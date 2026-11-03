"""Sequential real-machine Pilot, locked to exactly three previously failed IDs.

Secrets are loaded from a server-only JSON file and never written into the manifest.
Acceptance scripts below are derived from public issue descriptions, not gold tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PILOT_IDS = ("psf__requests-2674", "pytest-dev__pytest-8365", "sympy__sympy-17630")

PUBLIC_CHECKS = {
    PILOT_IDS[0]: """from requests import Response
from requests.exceptions import RequestException
from requests.packages.urllib3.exceptions import DecodeError, ReadTimeoutError
for failure in (DecodeError("broken decode"), ReadTimeoutError(None, None, "timeout")):
    class Raw:
        def stream(self, *args, **kwargs):
            raise failure
            yield b""
    response = Response()
    response.raw = Raw()
    try:
        list(response.iter_content())
    except RequestException:
        pass
    else:
        raise AssertionError("Expected a requests exception")
print("public regression passed")
""",
    PILOT_IDS[1]: """import os, tempfile
from unittest.mock import patch
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import pytest
from _pytest.tmpdir import TempPathFactory
with tempfile.TemporaryDirectory() as directory:
    os.environ["PYTEST_DEBUG_TEMPROOT"] = directory
    config = pytest.Config.fromdictargs({}, [])
    try:
        with patch("getpass.getuser", return_value="invalid/user"):
            factory = TempPathFactory.from_config(config)
            assert factory.getbasetemp().is_dir()
    finally:
        config._ensure_unconfigure()
print("public regression passed")
""",
    PILOT_IDS[
        2
    ]: """from sympy import MatrixSymbol, ZeroMatrix, BlockMatrix, block_collapse
a = MatrixSymbol("a", 2, 2)
z = ZeroMatrix(2, 2)
b = BlockMatrix([[a, z], [z, z]])
assert block_collapse(b*b*b).shape == (4, 4)
assert b._blockmul(b)._blockmul(b).shape == (4, 4)
print("public regression passed")
""",
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument(
        "--dataset", type=Path, default=Path("/data/datasets/swebench-lite/test.jsonl")
    )
    result.add_argument(
        "--secret-file", type=Path, default=Path("/data/secrets/deepseek.json")
    )
    result.add_argument("--worker", choices=PILOT_IDS)
    result.add_argument("--attempt", type=int, default=1)
    result.add_argument("--token-budget", type=int, default=300_000)
    return result


async def worker(args):
    secret = json.loads(args.secret_file.read_text())
    os.environ["DEEPSEEK_API_KEY"] = secret["api_key"]
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    model = secret["model"]
    from base.engine.async_llm import LLMsConfig
    from benchmark.bench_swebench import SWEBenchConfig, SWEBenchEnvironment
    from benchmark.swebench.data_loader import SWEBenchInstance
    from porchestra.runners.runtime_runner import RuntimeOptions, SWEBenchRuntimeRunner
    from porchestra.runtime.types import AcceptanceCheck

    LLMsConfig._default_config = LLMsConfig(
        {
            model: {
                "api_key_env": "DEEPSEEK_API_KEY",
                "base_url": secret["base_url"],
                "temperature": 0,
                "top_p": 1,
                "timeout": 90,
            }
        }
    )
    row = next(
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if json.loads(line)["instance_id"] == args.worker
    )
    instance = SWEBenchInstance.from_dict(row)
    attempt = args.run_dir / "attempts" / args.worker / f"attempt-{args.attempt}"
    config = SWEBenchConfig(
        dataset_name="princeton-nlp/SWE-bench_Lite",
        max_steps=100,
        docker_timeout=600,
        result_folder=attempt / "official",
        timestamp=f"pilot-{args.attempt}",
    )

    class SelectedBenchmark:
        def make_env(self, level):
            if level["id"] != args.worker:
                raise ValueError("Pilot instance escaped the three-ID allowlist")
            return SWEBenchEnvironment(level, config, instance)

    command = "python -c " + shlex.quote(PUBLIC_CHECKS[args.worker])
    check = AcceptanceCheck(
        "public_issue_regression",
        command,
        expected_output_pattern="public regression passed",
        timeout_seconds=120,
    )
    runner = SWEBenchRuntimeRunner(
        benchmark=SelectedBenchmark(),
        main_model=model,
        sub_models=[model],
        options=RuntimeOptions(
            (check,), token_budget=args.token_budget, node_concurrency=2, max_nodes=10
        ),
        max_steps=80,
        task_timeout_seconds=900,
        grade_terminal_failures=True,
    )
    results = await runner.run_levels(
        [{"id": args.worker, "problem_statement": instance.problem_statement}],
        1,
        attempt / "results.csv",
        attempt / "trajectories",
        "pilot",
    )
    result = results[args.worker]
    result["scored"] = "official_report" in result.get("grading", {})
    result["attempt"] = args.attempt
    args.run_dir.joinpath("status").mkdir(parents=True, exist_ok=True)
    args.run_dir.joinpath("status", args.worker + ".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "event": "PILOT_CASE_END",
                "instance_id": args.worker,
                "scored": result["scored"],
                "resolved": result["success"],
                "runtime_outcome": result["runtime_outcome"],
                "tokens_used": result["tokens_used"],
                "error": result["error"],
            }
        ),
        flush=True,
    )
    return 0 if result["scored"] else 2


def main():
    args = parser().parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.worker:
        return asyncio.run(worker(args))
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "pilot_ids": PILOT_IDS,
        "revision": revision,
        "case_concurrency": 1,
        "node_concurrency": 2,
        "token_budget_per_case": args.token_budget,
        "runtime_deadline_seconds": 900,
        "acceptance_source": "public issue only",
        "run_id": uuid4().hex,
    }
    args.run_dir.joinpath("pilot-manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    for identity in PILOT_IDS:
        status = args.run_dir / "status" / (identity + ".json")
        previous = json.loads(status.read_text()) if status.exists() else {}
        if previous.get("scored"):
            print(
                json.dumps({"event": "PILOT_RETAIN_SCORED", "instance_id": identity}),
                flush=True,
            )
            continue
        attempt = int(previous.get("attempt", 0)) + 1
        print(
            json.dumps(
                {
                    "event": "PILOT_CASE_START",
                    "instance_id": identity,
                    "attempt": attempt,
                }
            ),
            flush=True,
        )
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            identity,
            "--run-dir",
            str(args.run_dir),
            "--dataset",
            str(args.dataset),
            "--secret-file",
            str(args.secret_file),
            "--attempt",
            str(attempt),
            "--token-budget",
            str(args.token_budget),
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait() != 0:
            print(
                json.dumps(
                    {"event": "PILOT_PAUSED_FOR_REPAIR", "instance_id": identity}
                ),
                flush=True,
            )
            return 2
    print(json.dumps({"event": "PILOT_COMPLETE", "scored_instances": 3}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

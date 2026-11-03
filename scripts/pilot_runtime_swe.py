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
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PILOT_IDS = ("psf__requests-2674", "pytest-dev__pytest-8365", "sympy__sympy-17630")

PUBLIC_CHECKS = {
    PILOT_IDS[0]: """from requests import Response
from requests.exceptions import RequestException
from requests.packages.urllib3.exceptions import DecodeError, ReadTimeoutError, TimeoutError
for failure in (DecodeError("broken decode"), ReadTimeoutError(None, None, "timeout"), TimeoutError("proxy timeout")):
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
from _pytest.config import Config
from _pytest.tmpdir import TempPathFactory
with tempfile.TemporaryDirectory() as directory:
    os.environ["PYTEST_DEBUG_TEMPROOT"] = directory
    config = Config.fromdictargs({}, [])
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


from scripts.pilot_extra_cases import (
    FAILURE_IDS,
    SUCCESS_IDS,
    IMPORTS as EXTRA_IMPORTS,
    PUBLIC_CHECKS as EXTRA_CHECKS,
)

ORIGINAL_IDS = PILOT_IDS
PILOT_SETS = {
    "original": ORIGINAL_IDS,
    "baseline-failures-extra3": FAILURE_IDS,
    "baseline-success-control": SUCCESS_IDS,
}
IMPORT_MODULES = {
    ORIGINAL_IDS[0]: ("requests",),
    ORIGINAL_IDS[1]: ("pytest", "_pytest"),
    ORIGINAL_IDS[2]: ("sympy",),
    **EXTRA_IMPORTS,
}
PUBLIC_CHECKS.update(EXTRA_CHECKS)
ALL_IDS = tuple(
    dict.fromkeys(identity for group in PILOT_SETS.values() for identity in group)
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument(
        "--dataset", type=Path, default=Path("/data/datasets/swebench-lite/test.jsonl")
    )
    result.add_argument(
        "--secret-file", type=Path, default=Path("/data/secrets/deepseek.json")
    )
    result.add_argument("--pilot-set", choices=tuple(PILOT_SETS), default="original")
    result.add_argument("--worker", choices=ALL_IDS)
    result.add_argument("--attempt", type=int, default=1)
    result.add_argument("--token-budget", type=int, default=300_000)
    return result


async def worker(args):
    if args.worker not in PILOT_SETS[args.pilot_set]:
        raise ValueError("Worker escaped the selected explicit Pilot set")
    started = time.monotonic()
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
                "extra_body": {
                    "thinking": {"type": "disabled"},
                    "parallel_tool_calls": False,
                },
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
        import_modules=IMPORT_MODULES[args.worker],
    )

    class GuardedPilotRunner(SWEBenchRuntimeRunner):
        async def _prepare(self, env, directory):
            import time

            from porchestra.runtime import GlobalMemoryBoard, ToolGateway
            from porchestra.runtime.benchmark_bindings import (
                BoundSWEExecutor,
                SnapshotCheckRunner,
            )
            from porchestra.runtime.memory import ToolRequest

            prepared = await super()._prepare(env, directory)
            gateway = ToolGateway(GlobalMemoryBoard.instance())
            request = ToolRequest(
                "pilot-cache-probe",
                "infrastructure-probe",
                prepared.workspace_id,
                "view",
                {"operation": "git-head-readonly"},
            )
            calls = 0

            async def read_head():
                nonlocal calls
                calls += 1
                output, code = await BoundSWEExecutor(
                    prepared.workspaces, prepared.workspace_id
                ).execute_command("git rev-parse HEAD")
                return {"output": output, "exit_code": code}

            await gateway.execute(request, read_head)
            cached = await gateway.execute(request, read_head)
            print(
                json.dumps(
                    {
                        "event": "CACHE_PROBE",
                        "scope": "infrastructure_only_not_model_savings",
                        "annotation": cached.annotation,
                        "real_calls": calls,
                        "cache_hit": cached.cache_hit,
                    }
                ),
                flush=True,
            )
            if not cached.cache_hit or calls != 1:
                raise RuntimeError("Real-container cache probe failed")
            base = await prepared.workspaces.snapshot(prepared.workspace_id)
            baseline = await SnapshotCheckRunner(prepared.workspaces).run(
                check, base, deadline_monotonic=time.monotonic() + 120
            )
            print(
                json.dumps(
                    {
                        "event": "PILOT_BASELINE_CHECK",
                        "instance_id": args.worker,
                        "exit_code": baseline.exit_code,
                        "output_tail": (baseline.stdout + baseline.stderr)[-800:],
                    }
                ),
                flush=True,
            )
            if baseline.exit_code == 0:
                raise RuntimeError(
                    "Public DoD already passes the unmodified baseline; repair acceptance before model calls"
                )
            if baseline.category == "framework_check":
                raise RuntimeError(
                    "Framework import check failed before task regression: "
                    + baseline.stdout
                    + baseline.stderr
                )
            if any(
                marker in baseline.stdout + baseline.stderr
                for marker in (
                    "ModuleNotFoundError",
                    "ImportError",
                    "command not found",
                )
            ):
                raise RuntimeError(
                    "Public DoD baseline failed due to environment setup, not the target regression"
                )
            return prepared

    runner = GuardedPilotRunner(
        benchmark=SelectedBenchmark(),
        main_model=model,
        sub_models=[model],
        options=RuntimeOptions(
            (check,), token_budget=args.token_budget, node_concurrency=2, max_nodes=10
        ),
        max_steps=80,
        subagent_backend="mini",
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
    result["code_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result["duration_seconds"] = time.monotonic() - started
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
        "pilot_ids": PILOT_SETS[args.pilot_set],
        "pilot_set": args.pilot_set,
        "revision": revision,
        "case_concurrency": 1,
        "node_concurrency": 2,
        "token_budget_per_case": args.token_budget,
        "runtime_deadline_seconds": 900,
        "acceptance_source": "public issue only",
        "run_id": uuid4().hex,
    }
    manifest_path = args.run_dir / "pilot-manifest.json"
    if manifest_path.exists():
        history = args.run_dir / "history"
        history.mkdir(exist_ok=True)
        (history / f"manifest-{uuid4().hex}.json").write_text(manifest_path.read_text())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for identity in PILOT_SETS[args.pilot_set]:
        status = args.run_dir / "status" / (identity + ".json")
        previous = json.loads(status.read_text()) if status.exists() else {}
        if previous.get("scored"):
            print(
                json.dumps({"event": "PILOT_RETAIN_SCORED", "instance_id": identity}),
                flush=True,
            )
            continue
        existing = list((args.run_dir / "attempts" / identity).glob("attempt-*"))
        attempt = (
            max(
                [
                    int(previous.get("attempt", 0)),
                    *(int(path.name.split("-")[-1]) for path in existing),
                ]
            )
            + 1
        )
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
            "--pilot-set",
            args.pilot_set,
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
    print(
        json.dumps(
            {
                "event": "PILOT_COMPLETE",
                "scored_instances": len(PILOT_SETS[args.pilot_set]),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

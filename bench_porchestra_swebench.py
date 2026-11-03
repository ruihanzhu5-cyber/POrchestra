"""Run SWE-bench Verified with POrchestra.

Usage:
    python bench_porchestra_swebench.py --check-command 'pytest tests' --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from aorchestra.config import SWEBenchOrchestraConfig
from porchestra.runners.runtime_runner import (
    SWEBenchRuntimeRunner,
    add_runtime_arguments,
    runtime_options,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config/example/benchmarks/porchestra_swebench.yaml"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run SWE-bench with POrchestra")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max_concurrency", type=int, default=None)
    parser.add_argument(
        "--tasks", type=str, default=None, help="Comma-separated instance IDs"
    )
    parser.add_argument("--task_timeout_seconds", type=int, default=None)
    add_runtime_arguments(parser)
    args = parser.parse_args()

    cfg = SWEBenchOrchestraConfig.load(args.config)
    options = runtime_options(args, args.config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "runner": "NodeRuntime",
                    "benchmark": "swebench",
                    "token_budget": options.token_budget,
                    "node_concurrency": options.node_concurrency,
                    "checks": [check.command for check in options.acceptance_checks],
                    "topology": "<trajectory>/<timestamp>/<instance>/topology.jsonl",
                },
                indent=2,
            )
        )
        return 0
    from benchmark.bench_swebench import SWEBenchBenchmark, SWEBenchConfig

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.timestamp = timestamp
    result_folder = cfg.result_folder
    trajectory_folder = cfg.trajectory_dir or result_folder / "trajectories"
    csv_path = cfg.csv_summary_path or result_folder / "results.csv"
    result_folder.mkdir(parents=True, exist_ok=True)

    benchmark = SWEBenchBenchmark(
        SWEBenchConfig(
            dataset_name=cfg.dataset_name,
            split=cfg.split,
            subset_seed=cfg.subset_seed,
            subset_sizes=cfg.subset_sizes,
            subset_role=cfg.subset_role,
            max_steps=cfg.max_steps,
            max_tasks=None,
            docker_timeout=cfg.docker_timeout,
            result_folder=result_folder,
            trajectory_dir=trajectory_folder,
            csv_summary_path=csv_path,
            timestamp=timestamp,
            env_init=cfg.env_init,
            cache_dir=cfg.cache_dir,
            window_size=cfg.window_size,
        )
    )
    levels = benchmark.list_levels()

    selected_ids: set[str] | None = None
    if cfg.selected_ids_file:
        selected_ids = set(
            json.loads(cfg.selected_ids_file.read_text(encoding="utf-8"))
        )
    if args.tasks:
        command_ids = {item.strip() for item in args.tasks.split(",") if item.strip()}
        selected_ids = (
            command_ids if selected_ids is None else selected_ids & command_ids
        )
    if selected_ids is not None:
        levels = [level for level in levels if str(level.get("id")) in selected_ids]
    if cfg.max_tasks is not None:
        levels = levels[: int(cfg.max_tasks)]
    if not levels:
        print("No SWE-bench instances selected.", file=sys.stderr)
        return 1

    runner = SWEBenchRuntimeRunner(
        benchmark=benchmark,
        main_model=cfg.main_model,
        sub_models=cfg.sub_models,
        options=options,
        max_steps=cfg.max_total_steps,
        subagent_backend=cfg.subagent_backend,
        task_timeout_seconds=args.task_timeout_seconds or cfg.task_timeout_seconds,
    )
    results = await runner.run_levels(
        levels=levels,
        max_concurrency=args.max_concurrency or cfg.max_concurrency,
        csv_path=csv_path,
        trajectory_folder=trajectory_folder,
        timestamp=timestamp,
    )
    passed = sum(bool(result.get("success")) for result in results.values())
    print(f"POrchestra SWE-bench: {passed}/{len(results)} passed")
    print(f"Results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

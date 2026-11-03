"""GAIA benchmark using NodeRuntime-managed POrchestra sessions.

Usage:
    python bench_porchestra_gaia.py --check-command 'python -m json.tool answer.json' --dry-run
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.engine.logs import logger
from aorchestra.config import GAIAOrchestraConfig
from porchestra.runners.runtime_runner import (
    GAIARuntimeRunner,
    add_runtime_arguments,
    runtime_options,
)
from base.engine.async_llm import get_openai_proxy


DEFAULT_CONFIG_PATH = ROOT / "config/example/benchmarks/porchestra_gaia.yaml"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run GAIA benchmark using POrchestra.")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML."
    )
    parser.add_argument(
        "--max_concurrency", type=int, default=None, help="Override max_concurrency."
    )
    parser.add_argument(
        "--tasks", type=str, default=None, help="Comma-separated task IDs."
    )
    parser.add_argument(
        "--skip_completed",
        type=str,
        default=None,
        help="Path to existing CSV to skip completed tasks.",
    )
    parser.add_argument(
        "--result_folder",
        type=Path,
        default=None,
        help="Override the configured result folder.",
    )
    parser.add_argument(
        "--trajectory_folder",
        type=Path,
        default=None,
        help="Override the configured trajectory folder.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Use a deterministic run timestamp/name instead of wall-clock time.",
    )
    add_runtime_arguments(parser)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GAIA Benchmark with POrchestra")
    logger.info("=" * 60)

    cfg = GAIAOrchestraConfig.load(args.config)
    options = runtime_options(args, args.config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "runner": "NodeRuntime",
                    "benchmark": "gaia",
                    "token_budget": options.token_budget,
                    "checks": [check.command for check in options.acceptance_checks],
                    "topology": "<trajectory>/<timestamp>/<task>/topology.jsonl",
                },
                indent=2,
            )
        )
        return 0
    from benchmark.bench_gaia import GAIAConfig
    from benchmark.gaia.tools import (
        ExecuteCodeAction,
        ExtractUrlContentAction,
        GoogleSearchAction,
        ImageAnalysisAction,
        ParseAudioAction,
    )
    from benchmark.porchestra_bench_gaia import PorchestraGAIABenchmark

    if args.result_folder is not None:
        cfg.result_folder = args.result_folder
    if args.trajectory_folder is not None:
        cfg.trajectory_folder = args.trajectory_folder
    # Keep the compact/original GAIA handoff for this ablation: MainAgent sees
    # the SubAgent SAE summary, without injecting the final raw observation.
    os.environ["PORCHESTRA_INCLUDE_RAW_OBSERVATION"] = "false"
    if cfg.subagent_prompt_variant:
        os.environ["PORCHESTRA_GAIA_PROMPT_VARIANT"] = cfg.subagent_prompt_variant
    os.environ["PORCHESTRA_EVIDENCE_SOURCE"] = cfg.subagent_evidence_source
    if cfg.summary_model:
        os.environ["PORCHESTRA_TRACE_SUMMARY_MODEL"] = cfg.summary_model
    active_prompt_variant = os.getenv("PORCHESTRA_GAIA_PROMPT_VARIANT", "control_skill")
    if not cfg.dataset_path.exists():
        logger.error(f"Dataset not found: {cfg.dataset_path}")
        return 1

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.timestamp = timestamp

    gaia_tools = [
        GoogleSearchAction(),
        ExecuteCodeAction(),
        ExtractUrlContentAction(),
        ImageAnalysisAction(),
        ParseAudioAction(),
    ]
    logger.info(f"Dataset: {cfg.dataset_path}")
    logger.info(f"Attachments: {cfg.attachments_dir}")
    logger.info(
        f"Loaded {len(gaia_tools)} GAIA tools: {[tool.name for tool in gaia_tools]}"
    )
    logger.info(f"OpenAI-compatible proxy: {get_openai_proxy() or 'not set'}")
    logger.info(
        f"Serper proxy: {os.getenv('SERPER_PROXY') or os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY') or 'not set'}"
    )
    logger.info(
        f"Jina proxy: {os.getenv('JINA_PROXY') or os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY') or 'not set'}"
    )
    logger.info(f"POrchestra GAIA SubAgent prompt variant: {active_prompt_variant}")
    logger.info(
        f"POrchestra GAIA evidence source: {cfg.subagent_evidence_source}; "
        f"trace summary model: {cfg.summary_model or 'gemini-3-flash-preview'}"
    )

    gaia_cfg = GAIAConfig(
        dataset_path=cfg.dataset_path,
        attachments_dir=cfg.attachments_dir,
        max_steps=cfg.max_steps,
        level_filter=cfg.level_filter,
        max_tasks=cfg.max_tasks,
        result_folder=cfg.result_folder,
        trajectory_folder=cfg.trajectory_folder,
    )
    benchmark = PorchestraGAIABenchmark(gaia_cfg, tools=gaia_tools)
    levels = benchmark.list_levels()
    if not levels:
        logger.error("No tasks found in dataset.")
        return 1

    if args.tasks:
        task_ids = {
            task_id.strip() for task_id in args.tasks.split(",") if task_id.strip()
        }
        levels = [
            level
            for level in levels
            if (level.get("task_id") or level.get("id")) in task_ids
        ]
        logger.info(f"Filtered to {len(levels)} task(s)")
    elif cfg.max_tasks and len(levels) > cfg.max_tasks:
        levels = levels[: cfg.max_tasks]
        logger.info(f"Limited to {len(levels)} task(s)")

    if args.skip_completed:
        skip_csv_path = Path(args.skip_completed)
        if skip_csv_path.exists():
            completed = set()
            with skip_csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task_id = row.get("task_id") or row.get("id")
                    if task_id:
                        completed.add(task_id)
            before = len(levels)
            levels = [
                level
                for level in levels
                if (level.get("task_id") or level.get("id")) not in completed
            ]
            logger.info(f"Skipped {before - len(levels)} completed task(s)")

    cfg.result_folder.mkdir(parents=True, exist_ok=True)
    cfg.trajectory_folder.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.result_folder / f"gaia_porchestra_{timestamp}.csv"

    runner = GAIARuntimeRunner(
        benchmark=benchmark,
        main_model=cfg.main_model,
        sub_models=cfg.sub_models,
        options=options,
        max_steps=cfg.max_steps * cfg.max_attempts,
        gaia_tools=gaia_tools,
        task_timeout_seconds=cfg.task_timeout_seconds,
    )

    logger.info(
        f"[POrchestra GAIA] main_model={cfg.main_model}, sub_models={cfg.sub_models}"
    )
    logger.info(f"Running {len(levels)} task(s)")
    logger.info(f"Max concurrency: {args.max_concurrency or cfg.max_concurrency}")
    logger.info(f"Task timeout seconds: {cfg.task_timeout_seconds}")
    logger.info(f"Results: {csv_path}")

    results = await runner.run_levels(
        levels=levels,
        max_concurrency=args.max_concurrency or cfg.max_concurrency,
        csv_path=csv_path,
        trajectory_folder=cfg.trajectory_folder,
        timestamp=timestamp,
    )

    total = len(results)
    success_count = sum(1 for result in results.values() if result.get("success"))
    total_reward = sum(
        float(result.get("reward", 0) or 0) for result in results.values()
    )

    logger.info("\n" + "=" * 60)
    logger.info("POrchestra GAIA Benchmark Summary:")
    logger.info(f"  Total tasks: {total}")
    logger.info(f"  Successful: {success_count}/{total}")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Results: {csv_path}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

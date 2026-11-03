"""SWE-bench/GAIA entry wiring for NodeRuntime; dependencies are injectable in tests."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

from porchestra.runtime import (
    AcceptanceCheck,
    BudgetLedger,
    ContractGenerator,
    ContractVerifier,
    EventSink,
    GlobalMemoryBoard,
    HealthMonitor,
    NodeRegistry,
    NodeRuntime,
    RuntimeLimits,
    StateMerger,
    ToolGateway,
)
from porchestra.runtime.agent_driver import POrchestraAgentDriver, create_legacy_session
from porchestra.runtime.benchmark_bindings import (
    BoundSWEExecutor,
    GAIA_ANSWER_CHECK,
    GAIAToolBinding,
    MountedSWEWorkspace,
    SnapshotCheckRunner,
    SWEToolBinding,
    initialize_gaia_workspace,
)
from porchestra.runtime.git_workspace import GitWorkspaceBackend
from porchestra.runtime.types import NodeRecord
from porchestra.live_trace import LiveTrace


@dataclass(frozen=True)
class RuntimeOptions:
    acceptance_checks: tuple[AcceptanceCheck, ...]
    token_budget: int = 100_000
    node_concurrency: int = 4
    max_depth: int = 3
    max_nodes: int = 32

    def __post_init__(self):
        if not self.acceptance_checks:
            raise ValueError(
                "Configure runtime_acceptance_checks or --check-command before running"
            )
        if type(self.token_budget) is not int or self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        RuntimeLimits(
            max_active=self.node_concurrency,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
        )


def add_runtime_arguments(parser) -> None:
    parser.add_argument(
        "--check-command",
        action="append",
        help="Repeatable root DoD; commands run in isolated verifier checkouts",
    )
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--node-concurrency", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print runtime wiring; do not load datasets, create clients or start environments",
    )


def runtime_options(args, config_path) -> RuntimeOptions:
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    supplied = (
        args.check_command
        if args.check_command is not None
        else raw.get("runtime_acceptance_checks", [])
    )
    if not isinstance(supplied, list):
        raise ValueError(
            "runtime_acceptance_checks must be a list of commands/check objects"
        )
    checks = tuple(
        ContractGenerator._parse_check(check, i) for i, check in enumerate(supplied, 1)
    )
    return RuntimeOptions(
        checks,
        args.token_budget
        if args.token_budget is not None
        else int(raw.get("runtime_token_budget", 100_000)),
        args.node_concurrency
        if args.node_concurrency is not None
        else int(raw.get("runtime_node_concurrency", 4)),
        args.max_depth
        if args.max_depth is not None
        else int(raw.get("runtime_max_depth", 3)),
        int(raw.get("runtime_max_nodes", 32)),
    )


@dataclass
class PreparedTask:
    workspaces: Any
    workspace_id: str
    basic_info: Any
    observation: dict[str, Any]
    binding_factory: Callable[[NodeRecord], Any]
    finalize: Callable[[Any], Awaitable[dict[str, Any]]]
    check_runner: Any = None


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip(".")
    if cleaned != value:
        return (
            (cleaned or "task") + "-" + hashlib.sha256(value.encode()).hexdigest()[:8]
        )
    return cleaned


class RuntimeBenchmarkRunner:
    benchmark_type: str

    def __init__(
        self,
        *,
        benchmark: Any,
        main_model: str,
        sub_models: list[str],
        options: RuntimeOptions,
        max_steps: int = 500,
        subagent_backend: str = "react",
        task_timeout_seconds: float | None = None,
        session_factory: Callable[..., Any] = create_legacy_session,
        prepare_task: Callable[[Any, Path], Awaitable[PreparedTask]] | None = None,
        gaia_tools: list[Any] | None = None,
    ) -> None:
        self.benchmark, self.main_model, self.sub_models = (
            benchmark,
            main_model,
            sub_models,
        )
        self.options, self.max_steps, self.subagent_backend = (
            options,
            max_steps,
            subagent_backend,
        )
        self.task_timeout_seconds = task_timeout_seconds or 3600.0
        self.session_factory, self.prepare_task = session_factory, prepare_task
        self.gaia_tools = gaia_tools or []
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive action count")
        if not sub_models:
            raise ValueError("At least one child model is required")

    async def run_levels(
        self, levels, max_concurrency, csv_path, trajectory_folder, timestamp
    ):
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        results: dict[str, dict[str, Any]] = {}

        async def run_one(level):
            identity = str(level.get("id") or level.get("task_id"))
            async with semaphore:
                directory = Path(trajectory_folder) / timestamp / _safe(identity)
                results[identity] = await self._run_one(level, directory, timestamp)

        await asyncio.gather(*(run_one(level) for level in levels))
        if csv_path is not None:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as stream:
                columns = [
                    "task_id",
                    "instance_id",
                    "success",
                    "reward",
                    "runtime_outcome",
                    "tokens_used",
                    "nodes",
                    "topology_path",
                    "timestamp",
                    "error",
                ]
                writer = csv.DictWriter(
                    stream, fieldnames=columns, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(results.values())
        return results

    async def _run_one(self, level: dict[str, Any], directory: Path, timestamp: str):
        directory.mkdir(parents=True, exist_ok=True)
        identity = str(level.get("id") or level.get("task_id"))
        events = EventSink(str(directory / "topology.jsonl"))
        runtime = None
        env = None
        outcome = None
        error = ""
        grading: dict[str, Any] = {}
        sessions = []
        try:
            allowed = (
                ("aci_command",)
                if self.benchmark_type == "swebench"
                else tuple(tool.name for tool in self.gaia_tools)
            )
            question = str(
                level.get("Question")
                or level.get("problem_statement")
                or "Task objective from environment"
            )
            checks = self.options.acceptance_checks
            if self.benchmark_type == "gaia":
                checks = (
                    *checks,
                    AcceptanceCheck("answer_artifact", GAIA_ANSWER_CHECK),
                )
            # Validate DoD before any environment/container initialization.
            contract = await ContractGenerator().create(
                json.dumps(
                    {
                        "objective": question,
                        "acceptance_checks": [asdict(check) for check in checks],
                        "allowed_tools": list(allowed),
                    }
                ),
                budget_limit=self.options.token_budget,
            )
            if self.benchmark_type == "swebench":
                env = self.benchmark.make_env(level)
            else:
                env = self.benchmark.make_env(
                    {**level, "_run_id": timestamp}, tools=self.gaia_tools
                )
            prepared = await (
                self.prepare_task(env, directory)
                if self.prepare_task
                else self._prepare(env, directory)
            )
            contract = replace(
                contract,
                objective=prepared.basic_info.instruction,
                deadline_seconds=self.task_timeout_seconds,
            )
            workspaces = prepared.workspaces
            base = await workspaces.snapshot(prepared.workspace_id)
            board = GlobalMemoryBoard.instance()
            gateway = ToolGateway(board)

            def create_driver(node, tools):
                model = (
                    self.main_model
                    if node.parent_id is None
                    else self.sub_models[(node.depth - 1) % len(self.sub_models)]
                )
                workspace_path = (
                    workspaces.container_path(node.workspace_id)
                    if hasattr(workspaces, "container_path")
                    else node.workspace_id
                )
                trace = LiveTrace(
                    directory / "agents" / _safe(node.node_id) / "events.jsonl"
                )
                session = self.session_factory(
                    node,
                    model=model,
                    benchmark_type=self.benchmark_type,
                    basic_info=prepared.basic_info,
                    workspace_path=workspace_path,
                    backend=self.subagent_backend,
                    trace=trace,
                )
                sessions.append(session)
                return POrchestraAgentDriver(
                    session,
                    prepared.binding_factory(node),
                    tools,
                    checkpoint_dir=str(directory / "checkpoints"),
                    max_steps=self.max_steps,
                    initial_observation={
                        **(prepared.observation if node.parent_id is None else {}),
                        "message": "Node workspace ready",
                        "workspace": workspace_path,
                    },
                    trace=trace,
                )

            runtime = NodeRuntime(
                run_id=uuid4().hex,
                registry=NodeRegistry(),
                verifier=ContractVerifier(
                    prepared.check_runner or SnapshotCheckRunner(workspaces), workspaces
                ),
                contracts=ContractGenerator(),
                gateway=gateway,
                health=HealthMonitor(),
                merger=StateMerger(workspaces, board, events),
                events=events,
                workspaces=workspaces,
                budget=BudgetLedger(self.options.token_budget),
                driver_factory=create_driver,
                limits=RuntimeLimits(
                    max_active=self.options.node_concurrency,
                    max_depth=self.options.max_depth,
                    max_nodes=self.options.max_nodes,
                ),
            )
            # This replaces the MainAgent -> DelegateTaskTool decision loop.
            outcome = await asyncio.wait_for(
                runtime.run(contract, workspace_id=prepared.workspace_id),
                self.task_timeout_seconds,
            )
            if outcome.outcome == "verified":
                grading = await prepared.finalize(base)
            else:
                error = outcome.reason
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                try:
                    await env.close()
                except Exception as exc:
                    error = f"{error}; environment cleanup: {exc}".strip("; ")
        reward = float(grading.get("reward", 0.0))
        result = {
            "task_id": identity,
            "instance_id": identity,
            "success": bool(grading.get("success", reward == 1.0)),
            "reward": reward,
            "runtime_outcome": outcome.outcome if outcome else "failed",
            "tokens_used": 0,
            "nodes": runtime.registry.total_registered if runtime else 0,
            "topology_path": events.path,
            "timestamp": timestamp,
            "error": error,
            "final_answer": grading.get("final_answer"),
            "grading": grading,
            "total_cost": sum(
                float(session.llm.get_usage_summary().get("total_cost", 0.0))
                for session in sessions
            ),
        }
        if runtime and runtime.budget.used_by_node:
            root_id = next(iter(runtime.budget.parent_by_node))
            result["tokens_used"] = runtime.budget.used_by_node[root_id]
        (directory / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return result


class SWEBenchRuntimeRunner(RuntimeBenchmarkRunner):
    benchmark_type = "swebench"

    async def _prepare(self, env, directory) -> PreparedTask:
        from benchmark.swebench.aci_tools import ACIToolManager

        workspaces, workspace_id, observation = await MountedSWEWorkspace.prepare(
            env, str(directory / "container_workspace")
        )

        def binding(node):
            executor = BoundSWEExecutor(workspaces, node.workspace_id)
            manager = ACIToolManager(
                executor, window_size=int(getattr(env.config, "window_size", 100))
            )
            manager.state.working_dir = executor.cwd
            return SWEToolBinding(manager)

        async def finalize(base):
            patch = await workspaces.publish_for_grading(base, workspace_id)
            reward, report = await env._executor.run_tests()
            return {
                "reward": reward,
                "success": reward == 1.0,
                "patch": patch,
                "official_report": report,
            }

        return PreparedTask(
            workspaces,
            workspace_id,
            env.get_basic_info(),
            observation,
            binding,
            finalize,
        )


class GAIARuntimeRunner(RuntimeBenchmarkRunner):
    benchmark_type = "gaia"

    async def _prepare(self, env, directory) -> PreparedTask:
        observation = await env.reset()
        info = env.get_basic_info()
        workspace = await initialize_gaia_workspace(
            str(directory / "workspaces" / "root"), info.instruction
        )
        workspaces = GitWorkspaceBackend(str(directory / "runtime_git"))
        tools = {tool.name: tool for tool in self.gaia_tools}

        async def finalize(base):
            from benchmark.gaia.scorer import question_scorer

            answer = json.loads(
                (Path(workspace) / "answer.json").read_text(encoding="utf-8")
            )["answer"]
            reward = float(question_scorer(answer, env.expected_answer))
            return {"reward": reward, "success": reward > 0.5, "final_answer": answer}

        return PreparedTask(
            workspaces,
            workspace,
            info,
            observation,
            lambda node: GAIAToolBinding(tools),
            finalize,
        )

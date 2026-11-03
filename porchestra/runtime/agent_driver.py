"""One-action adapter around the existing POrchestra ReAct/Mini LLM sessions."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from .contracts import ContractGenerator
from .memory import ToolGateway, ToolRequest
from .types import NodeRecord, StepOutcome


@dataclass
class PreparedTool:
    request: ToolRequest
    call: Callable[[], Awaitable[Any]]


class ToolBinding(Protocol):
    def prepare(self, node: NodeRecord, action: dict[str, Any]) -> PreparedTool: ...
    def observation(
        self, value: Any, *, step: int, annotation: str
    ) -> dict[str, Any]: ...
    async def finish(
        self, node: NodeRecord, params: dict[str, Any], gateway: ToolGateway
    ) -> None: ...


RUNTIME_INSTRUCTION = """
NodeRuntime owns delegation, budgets and verification. finish is a verification
request; a rejected finish must be repaired in this SAME session. Do not submit
to the benchmark grader. To delegate, emit a JSON action with action="spawn" and
params.children containing two or more objects with objective, acceptance_checks
(real command strings) and budget_limit (tokens). In SWE DISCUSSION/COMMAND format,
use COMMAND followed by: spawn {"children": [...]} on one line. Preserve inherited
constraints. revise requests do not grant new tokens; use spawn to delegate work.
For GAIA finish include params.result with the exact proposed final answer.
"""


class POrchestraAgentDriver:
    """A private session/client per node; no env.reset or internal session loop.

    Existing step()/record_transition() methods preserve native tool-call IDs.
    Cumulative usage includes any memory compression performed by that session.
    """

    def __init__(
        self,
        session: Any,
        binding: ToolBinding,
        gateway: ToolGateway,
        *,
        checkpoint_dir: str,
        initial_observation: dict[str, Any] | None = None,
        max_steps: int = 500,
        contracts: ContractGenerator | None = None,
        trace: Any = None,
    ) -> None:
        self.session, self.binding, self.gateway = session, binding, gateway
        self.checkpoint_dir = Path(checkpoint_dir)
        self.observation = initial_observation or {"message": "Node ready"}
        self.history: list[dict[str, Any]] = []
        self.max_steps = max_steps
        self.contracts = contracts or ContractGenerator()
        self.trace = trace
        self._accounted_tokens = 0
        self._closed = False
        self._step_task: asyncio.Task | None = None
        self._node_id: str | None = None
        self._progress: dict[str, Any] = {}
        context = str(getattr(session, "context", ""))
        if RUNTIME_INSTRUCTION not in context:
            session.context = (context + "\n" + RUNTIME_INSTRUCTION).strip()

    def _usage(self) -> int:
        summary = self.session.llm.get_usage_summary()
        if "total_tokens" in summary:
            total = summary["total_tokens"]
        elif "total_input_tokens" in summary and "total_output_tokens" in summary:
            total = summary["total_input_tokens"] + summary["total_output_tokens"]
        else:
            raise ValueError(
                "LLM usage summary must expose actual cumulative token counts"
            )
        if type(total) is not int or total < self._accounted_tokens:
            raise ValueError("Invalid or decreasing LLM token usage")
        return total

    def _outcome(self, kind: str, *, reason: str = "", children=()) -> StepOutcome:
        total = self._usage()
        used = total - self._accounted_tokens
        self._accounted_tokens = total
        if self.trace is not None:
            self.trace.append(
                "runtime_agent_usage",
                node_id=self._node_id,
                kind=kind,
                tokens_used=used,
                cumulative_tokens=total,
            )
        return StepOutcome(
            kind,
            reason=reason,
            child_contracts=children,
            tokens_used=used,
            progress=self._progress,
        )

    async def step(self, node: NodeRecord) -> StepOutcome:
        self._node_id = node.node_id
        self._progress = {}
        if self._closed:
            raise RuntimeError("Driver is closed")
        if node.steps_used >= self.max_steps:
            return self._outcome("stop", reason="per-node action limit reached")
        self._step_task = asyncio.current_task()
        raw_response = ""
        try:
            action, raw_response, _ = await self.session.step(
                observation=self.observation,
                history=self.history,
                current_step=node.steps_used + 1,
                max_steps=self.max_steps,
            )
            if not isinstance(action, dict) or not isinstance(
                action.get("params", {}), dict
            ):
                raise ValueError(
                    "Session must return an action object with object params"
                )
            # Legacy SWE parsers expose new textual control words as ACI commands.
            if action.get("action") == "aci_command":
                command = str(action.get("params", {}).get("command", ""))
                control, _, payload = command.partition(" ")
                if control in {"spawn", "finish", "revise", "stop"}:
                    action = {"action": control, "params": json.loads(payload)}
            name = str(action.get("action", ""))
            params = action.get("params", {})
            if not isinstance(params, dict):
                raise ValueError("Control parameters must be an object")
            self.history.append({"action": action, "raw_response": raw_response})
            if self.trace is not None:
                self.trace.append(
                    "runtime_agent_action",
                    node_id=node.node_id,
                    action=action,
                    raw_response=raw_response,
                )
            if name in {"finish", "stop", "spawn", "revise"} and getattr(
                self.session, "_pending_tool_call_ids", []
            ):
                await self.session.record_transition(
                    action=action,
                    observation_after={
                        "message": "Control request forwarded to NodeRuntime"
                    },
                    reward=0.0,
                    raw_response=raw_response,
                )
            if name == "finish":
                await self.binding.finish(node, params, self.gateway)
                return self._outcome("finish")
            if name == "stop":
                return self._outcome(
                    "stop", reason=str(params.get("state", "agent stopped"))
                )
            if name == "spawn":
                children = params.get("children")
                if not isinstance(children, list) or not children:
                    raise ValueError("spawn requires nonempty params.children")
                contracts = []
                for child in children:
                    if (
                        not isinstance(child, dict)
                        or type(child.get("budget_limit")) is not int
                    ):
                        raise ValueError(
                            "Each child needs objective, DoD and an integer token budget"
                        )
                    contracts.append(
                        await self.contracts.create(
                            json.dumps(child),
                            budget_limit=child["budget_limit"],
                            parent=node.contract,
                        )
                    )
                return self._outcome("spawn", children=tuple(contracts))
            if name == "revise":
                await self.inject_context(node.node_id, RUNTIME_INSTRUCTION)
                return self._outcome("continue", reason="runtime guidance supplied")
            prepared = self.binding.prepare(node, action)
            result = await self.gateway.execute(prepared.request, prepared.call)
            if result.cache_hit:
                logging.getLogger(__name__).info(
                    "[Cache Hit] node=%s tool=%s",
                    node.node_id,
                    prepared.request.tool_name,
                )
            self.observation = self.binding.observation(
                result.value,
                step=node.steps_used + 1,
                annotation=result.annotation,
            )
            self._progress = {
                "tool": prepared.request.tool_name,
                "arguments": prepared.request.args,
                "exit_code": self.observation.get("exit_code"),
                "output_digest": hashlib.sha256(
                    str(self.observation.get("output", result.value)).encode()
                ).hexdigest(),
            }
            await self.session.record_transition(
                action=action,
                observation_after=self.observation,
                thinking=action.get("reasoning"),
                reward=0.0,
                raw_response=raw_response,
            )
            self.history[-1]["observation_after"] = self.observation
            if self.trace is not None:
                self.trace.append(
                    "runtime_tool_result",
                    node_id=node.node_id,
                    cache_hit=result.cache_hit,
                    observation=self.observation,
                )
            return self._outcome("continue")
        except Exception as exc:
            self.observation = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            # Complete pending native tool messages even when authorization/tool use failed.
            await self.session.record_transition(
                action={"action": "error"},
                observation_after=self.observation,
                reward=0.0,
                raw_response=raw_response,
            )
            await self.inject_context(node.node_id, self.observation["error"])
            return self._outcome("continue", reason=self.observation["error"])
        finally:
            self._step_task = None

    async def inject_context(self, node_id: str, message: str) -> None:
        if self.trace is not None:
            self.trace.append("runtime_context", node_id=node_id, message=message)
        apply_revision = getattr(self.session, "apply_revision", None)
        if callable(apply_revision):
            result = apply_revision(
                revision_context=message,
                task_instruction=self.session.task_instruction,
                context=str(getattr(self.session, "context", "")) + "\n" + message,
                max_steps=self.max_steps,
            )
            if inspect.isawaitable(result):
                await result
        else:
            self.session.context = (
                str(getattr(self.session, "context", "")) + "\n" + message
            )

    async def checkpoint(self, node_id: str) -> str:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Node IDs originate in contracts; use a digest so caller IDs cannot escape.
        import hashlib

        path = (
            self.checkpoint_dir / f"{hashlib.sha256(node_id.encode()).hexdigest()}.json"
        )
        data = {
            "node_id": node_id,
            "observation": self.observation,
            "history": self.history,
            "messages": getattr(self.session, "messages", None),
            "context": getattr(self.session, "context", ""),
            "tokens": self._accounted_tokens,
            "memory": self.session.memory.as_text()
            if getattr(self.session, "memory", None) is not None
            else None,
        }
        await asyncio.to_thread(
            path.write_text,
            json.dumps(data, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return str(path)

    async def cancel(self, node_id: str, *, reason: str) -> None:
        task = self._step_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def close(self, node_id: str) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.session.llm, "aclose", None)
        if not callable(close):
            close = getattr(getattr(self.session.llm, "aclient", None), "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


def create_legacy_session(
    node: NodeRecord,
    *,
    model: str,
    benchmark_type: str,
    basic_info: Any,
    workspace_path: str,
    backend: str = "react",
    trace: Any = None,
) -> Any:
    """Lazy imports keep CLI planning and Mock tests independent of model clients."""
    from base.engine.async_llm import LLMsConfig, create_llm_instance
    from porchestra.subagent import ReActSubAgent
    from porchestra.mini_swe_subagent import MiniSWESubAgent
    from porchestra.prompts.swebench_subagent import parse_swebench_subagent_response

    if backend not in {"react", "mini"}:
        raise ValueError("Unknown session backend")

    class WorkspaceMini(MiniSWESubAgent):
        def reset(self, env_info):
            super().reset(env_info)
            # Rewrite only the static system template, never file/tool observations.
            self._messages[0]["content"] = self._messages[0]["content"].replace(
                "/testbed", workspace_path
            )

        def _record_and_parse_payload(self, payload, *, current_step):
            action = super()._record_and_parse_payload(
                payload, current_step=current_step
            )
            # Mini's parser rejects textual environment commands. spawn belongs to
            # the runtime control plane, so recover this control after logging it.
            if not payload.get("tool_calls"):
                parsed = parse_swebench_subagent_response(
                    str(payload.get("content") or "")
                )
                command = str(parsed.get("params", {}).get("command", ""))
                if parsed.get("action") == "aci_command" and command.startswith(
                    "spawn "
                ):
                    try:
                        return {"action": "spawn", "params": json.loads(command[6:])}
                    except ValueError:
                        pass
            return action

        def apply_revision(self, **kwargs):
            super().apply_revision(**kwargs)
            self._messages[-1]["content"] = self._messages[-1]["content"].replace(
                "current /testbed workspace", f"current {workspace_path} workspace"
            )

    session_class = (
        WorkspaceMini
        if backend == "mini" and benchmark_type == "swebench"
        else ReActSubAgent
    )
    session = session_class(
        llm=create_llm_instance(LLMsConfig.default().get(model)),
        benchmark_type=benchmark_type,
        task_instruction=node.contract.objective,
        context=RUNTIME_INSTRUCTION
        + f"\nAssigned repository root: {workspace_path}. Use relative paths or this assigned root. The image's original /testbed is not your workspace.\n"
        + "\nContract:\n"
        + json.dumps(asdict(node.contract)),
        original_question=basic_info.instruction,
        allowed_tools=list(node.contract.allowed_tools),
        live_trace=trace,
        **(
            {"workspace_path": workspace_path} if session_class is WorkspaceMini else {}
        ),
    )
    session.llm.max_completion_tokens = 4096
    session.reset(basic_info)
    return session

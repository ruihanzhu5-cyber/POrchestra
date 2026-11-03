"""Mini-SWE-agent-style SWE-bench executor for POrchestra."""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field, PrivateAttr

from base.agent.base_agent import BaseAgent
from base.agent.truncating_memory import truncate_text
from base.engine.logs import LogLevel, logger
from benchmark.common.env import Action, BasicInfo, Observation
from porchestra.prompts.swebench_mini_subagent import (
    MINI_SWE_SYSTEM_PROMPT,
    build_mini_swe_bash_tool,
    build_mini_swe_final_prompt,
    build_mini_swe_task_prompt,
)
from porchestra.prompts.swebench_subagent import parse_swebench_subagent_response


class MiniSWESubAgent(BaseAgent):
    """Preserve a native assistant/tool history while exposing POrch controls."""

    name: str = Field(default="POrchestraMiniSWESubAgent")
    description: str = Field(default="Mini-SWE-agent-style SubAgent with SAE handoff")

    benchmark_type: str = Field(default="swebench")
    task_instruction: str = Field(default="")
    context: str = Field(default="")
    original_question: str = Field(default="")
    allowed_tools: list[str] | None = Field(default=None)
    live_trace: Any = Field(default=None, exclude=True)
    max_observation_chars: int = Field(default=10_000)
    workspace_path: str = Field(default="/testbed")

    _messages: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _initialized: bool = PrivateAttr(default=False)
    _pending_tool_call_ids: list[str] = PrivateAttr(default_factory=list)
    _last_current_step: int = PrivateAttr(default=0)
    _last_max_steps: int = PrivateAttr(default=0)

    class Config:
        arbitrary_types_allowed = True

    @property
    def messages(self) -> list[dict[str, Any]]:
        """A copy of the native conversation, primarily for traces/tests."""
        return [dict(message) for message in self._messages]

    def reset(self, env_info: BasicInfo) -> None:
        if not self.original_question:
            self.original_question = env_info.instruction
        self._messages = [{"role": "system", "content": MINI_SWE_SYSTEM_PROMPT}]
        self._initialized = False
        self._pending_tool_call_ids = []
        self._last_current_step = 0
        self._last_max_steps = 0

    def _budget_warning(self, remaining_steps: int) -> str:
        if remaining_steps <= 3:
            return (
                f"Only {remaining_steps} step(s) remain. Use finish, revise, or stop "
                "before the budget is exhausted."
            )
        if remaining_steps <= 5:
            return f"{remaining_steps} step(s) remain; prepare a handoff if needed."
        return ""

    def _ensure_initial_message(
        self,
        *,
        observation: Any,
        current_step: int,
        max_steps: int,
    ) -> None:
        if self._initialized:
            return
        remaining_steps = max(max_steps - current_step, 0)
        self._messages.append(
            {
                "role": "user",
                "content": build_mini_swe_task_prompt(
                    original_question=self.original_question,
                    task_instruction=self.task_instruction,
                    context=self.context,
                    observation=self._bounded_observation(observation),
                    current_step=current_step,
                    max_steps=max_steps,
                    remaining_steps=remaining_steps,
                    budget_warning=self._budget_warning(remaining_steps),
                    workspace_path=self.workspace_path,
                ),
            }
        )
        self._initialized = True

    async def step(
        self,
        observation: Observation,
        history: Any,
        current_step: int = 1,
        max_steps: int = 50,
    ) -> tuple[Action, str, str]:
        del history
        self._last_current_step = current_step
        self._last_max_steps = max_steps
        self._ensure_initial_message(
            observation=observation,
            current_step=current_step,
            max_steps=max_steps,
        )
        raw_input = json.dumps(self._messages, ensure_ascii=False)
        logger.log_to_file(
            LogLevel.INFO,
            f"POrchestra MiniSWESubAgent Messages:\n{raw_input}\n",
        )
        _append_live(
            self.live_trace,
            "subagent_prompt",
            backend="mini",
            current_step=current_step,
            max_steps=max_steps,
            task_instruction=self.task_instruction,
            context=self.context,
            observation=observation,
            messages=self.messages,
        )
        try:
            payload = await self.llm.call_messages_with_tools(
                self._messages,
                tools=build_mini_swe_bash_tool(self.workspace_path),
                tool_choice="auto",
            )
        except Exception as exc:
            logger.error(f"POrchestra MiniSWESubAgent LLM call failed: {exc}")
            payload = {"content": "", "tool_calls": []}

        action = self._record_and_parse_payload(payload, current_step=current_step)
        raw_response = json.dumps(payload, ensure_ascii=False)
        _append_live(
            self.live_trace,
            "subagent_response",
            backend="mini",
            current_step=current_step,
            response=payload,
            action=action,
        )
        logger.agent_action(f"POrchestra MiniSWESubAgent Action: {action}")
        return action, raw_response, raw_input

    def _record_and_parse_payload(
        self, payload: dict[str, Any], *, current_step: int
    ) -> Action:
        content = str(payload.get("content") or "")
        calls = payload.get("tool_calls") or []
        assistant_calls: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            call = call if isinstance(call, dict) else {}
            call_id = str(call.get("id") or f"mini_step_{current_step}_{index}")
            assistant_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(call.get("name") or ""),
                        "arguments": str(call.get("arguments") or "{}"),
                    },
                }
            )
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content if content else (None if assistant_calls else ""),
        }
        if assistant_calls:
            assistant_message["tool_calls"] = assistant_calls
        self._messages.append(assistant_message)

        # Every assistant tool call must receive a matching tool message before
        # the next model request, including malformed or unregistered calls.
        self._pending_tool_call_ids = [str(call["id"]) for call in assistant_calls]

        if not assistant_calls:
            action = parse_swebench_subagent_response(content)
            if action.get("action") == "aci_command":
                return {
                    "action": "error",
                    "params": {
                        "message": "Environment commands must use the native bash tool"
                    },
                    "reasoning": action.get("reasoning", ""),
                    "_parse_error": "text_environment_command",
                }
            return action

        if len(assistant_calls) != 1:
            return {
                "action": "error",
                "params": {
                    "message": f"Expected exactly one bash tool call, got {len(assistant_calls)}"
                },
                "reasoning": content,
                "_parse_error": "invalid_native_tool_count",
            }

        call = assistant_calls[0]
        name = str(call["function"]["name"]).lower()
        if name not in {"bash", "read_file", "read_range", "search"}:
            return {
                "action": "error",
                "params": {"message": f"Unknown native tool: {name or 'missing'}"},
                "reasoning": content,
                "_parse_error": "unknown_native_tool",
            }
        try:
            arguments = json.loads(call["function"]["arguments"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "action": "error",
                "params": {"message": f"Invalid bash tool JSON: {exc}"},
                "reasoning": content,
                "_parse_error": "invalid_native_tool_json",
            }
        if name != "bash" and isinstance(arguments, dict):
            return {"action":name, "params":arguments, "reasoning":content, "_native_tool_call":True}
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if not isinstance(command, str) or not command.strip():
            return {
                "action": "error",
                "params": {"message": "bash requires a non-empty command"},
                "reasoning": content,
                "_parse_error": "invalid_native_command",
            }
        return {
            "action": "aci_command",
            "params": {"command": command.strip()},
            "reasoning": content,
            "_native_tool_call": True,
        }

    async def record_transition(
        self,
        *,
        action: Action,
        observation_after: Observation,
        thinking: str | None = None,
        reward: float | None = None,
        raw_response: str | None = None,
    ) -> None:
        del action, thinking, reward, raw_response
        observation_text = self._bounded_observation(observation_after)
        remaining = max(self._last_max_steps - self._last_current_step - 1, 0)
        content = (
            f"{observation_text}\n\n"
            f"[Progress: completed step {self._last_current_step}/"
            f"{self._last_max_steps}; {remaining} step(s) remain.]"
        )
        if self._pending_tool_call_ids:
            for call_id in self._pending_tool_call_ids:
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content,
                    }
                )
            self._pending_tool_call_ids = []
        else:
            self._messages.append(
                {
                    "role": "user",
                    "content": "Environment rejected the previous action:\n" + content,
                }
            )

    def apply_revision(
        self,
        *,
        revision_context: str,
        task_instruction: str,
        context: str,
        max_steps: int,
    ) -> None:
        """Preserve the native history and append MainAgent's revised contract."""
        self.task_instruction = task_instruction
        self.context = context
        self._last_max_steps = max_steps
        self._messages.append(
            {
                "role": "user",
                "content": (
                    "MainAgent revised the active execution contract. Preserve the "
                    "current /testbed workspace and continue from the existing state.\n\n"
                    + (revision_context or "No additional revision details were provided.")
                    + f"\n\nCurrent assigned subtask:\n{task_instruction}"
                ),
            }
        )

    async def final_handoff(
        self,
        observation: Observation,
        history: Any,
        current_step: int = 1,
        max_steps: int = 50,
    ) -> tuple[Action, str, str]:
        del history
        self._ensure_initial_message(
            observation=observation,
            current_step=current_step,
            max_steps=max_steps,
        )
        final_prompt = build_mini_swe_final_prompt(
            observation=self._bounded_observation(observation),
            current_step=current_step,
            max_steps=max_steps,
        )
        self._messages.append({"role": "user", "content": final_prompt})
        raw_input = json.dumps(self._messages, ensure_ascii=False)
        try:
            payload = await self.llm.call_messages_with_tools(
                self._messages,
                tools=[],
                tool_choice="none",
            )
        except Exception as exc:
            logger.error(f"POrchestra MiniSWESubAgent final handoff failed: {exc}")
            payload = {"content": "", "tool_calls": []}
        action = self._record_and_parse_payload(payload, current_step=current_step)
        if action.get("action") not in {"finish", "revise", "stop"}:
            action = _invalid_final_handoff()
        raw_response = json.dumps(payload, ensure_ascii=False)
        _append_live(
            self.live_trace,
            "subagent_final_response",
            backend="mini",
            current_step=current_step,
            response=payload,
            action=action,
        )
        return action, raw_response, raw_input

    def _bounded_observation(self, observation: Any) -> str:
        if isinstance(observation, dict):
            value = observation.get("output") or observation.get("message") or observation
        else:
            value = observation
        return truncate_text(value, self.max_observation_chars)

    async def run(self, request: str | None = None) -> str:
        del request
        return ""


def _invalid_final_handoff() -> Action:
    return {
        "action": "revise",
        "params": {
            "state": "The final boundary produced no valid finish/revise/stop handoff.",
            "action": {
                "type": "revise",
                "task_scope": {
                    "operation": "keep",
                    "object": "current workspace",
                    "details": "Preserve the current patch and execution state.",
                },
                "resource": {
                    "operation": "request",
                    "object": "state_handoff",
                    "details": "A recoverable handoff decision is required.",
                },
                "request": "Preserve the workspace and revise or replace the executor.",
                "reason": "The executor did not emit a valid final control action.",
            },
            "evidence": {"summary": "No valid final SAE handoff was produced."},
        },
        "reasoning": "",
    }


def _append_live(live_trace: Any, event_name: str, **payload: Any) -> None:
    if live_trace is None:
        return
    try:
        live_trace.append(event_name, **payload)
    except Exception as exc:
        logger.debug(
            f"[POrchestra MiniSWESubAgent] Failed to write {event_name}: {exc}"
        )

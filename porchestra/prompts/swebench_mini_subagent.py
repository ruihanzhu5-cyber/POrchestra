"""Mini-SWE-agent-style prompt used by the POrchestra SWE executor."""

from __future__ import annotations

from typing import Any

from porchestra.prompts.swebench_subagent import SWE_REVISE_EXAMPLE, SWE_SAE_SCHEMA


MINI_SWE_SYSTEM_PROMPT = """You are a POrchestra SubAgent responsible for a local software-engineering task.
Your goal is to complete the GitHub issue subtask assigned by MainAgent.

You interact with the repository through real Bash tool calls. MainAgent owns task
orchestration and the final SWE-bench submission."""


MINI_SWE_TASK_PROMPT = """<PR_DESCRIPTION>
{original_question}
</PR_DESCRIPTION>

<TASK_INSTRUCTIONS>
==== Progress ====
Current step: {current_step}/{max_steps}. Remaining: {remaining_steps} step(s).
{budget_warning}

==== MainAgent Assignment ====
Subtask:
{task_instruction}

Context:
{context}

==== Environment ====
- The repository is already cloned at the correct commit in /testbed.
- The workspace may contain source changes inherited from an earlier SubAgent. These
  changes are shared task state, not disposable setup. Inspect them with `git diff`
  before editing and preserve useful existing work.
- Do not run Docker commands.
- Each Bash call runs in a fresh subshell. Include `cd /testbed` and any required
  environment activation in the command itself.
- For Python tests, activate the testbed conda environment when needed.
- For Django, use its `tests/runtests.py` test runner rather than pytest.

==== Inherited Workspace Safety ====
- Do not use `git reset`, `git checkout --`, `git restore`, or `git clean` to erase
  inherited changes.
- If you need the base version for comparison, read it without mutating the workspace
  using `git show HEAD:<path>` or copy it to a temporary file.
- If an inherited change is concretely wrong, repair the relevant hunk while preserving
  other valid work. Do not silently return the entire file or repository to HEAD.
- Only discard inherited work when MainAgent explicitly identifies that exact change as
  invalid and instructs you to remove it.

==== Modification Boundary ====
Modify ordinary non-test source files in /testbed. Do not modify tests, temporary
reproduction scripts, packaging/build configuration, or dependency metadata unless
the issue explicitly requires such a file.

==== Recommended Workflow ====
1. Locate and understand the relevant code.
2. Create a focused reproduction when useful.
3. Edit the source to fix the issue in the repository's existing style.
4. Run the reproduction or relevant existing tests.
5. Check relevant edge cases and regressions.

==== Execution and Communication ====
Use the native `bash` tool for repository inspection, editing, and testing. Tool
results are real environment observations and arrive in the following tool message.
Do not invent command output or claim an edit/test succeeded without observing it.

At any appropriate logical point, you may instead communicate with MainAgent:
- finish: the assigned subtask is complete and the current patch is ready to submit.
- revise: MainAgent must adjust task scope, context, tools, constraints, budget, or
  state handoff before this executor can continue. For budget requests, set
  resource.object="budget".
- stop: the current path is invalid, unsafe, ambiguous, or should be abandoned.

Never execute submit yourself. MainAgent performs the final SWE-bench evaluation.

For finish, revise, or stop, do not call Bash. Output exactly:

DISCUSSION
<concise current judgment>

COMMAND
<finish|revise|stop> <single-line SAE JSON>

SAE JSON schema:
{sae_schema}

Canonical revise shape:
{revise_example}

Choose one path per turn: Bash tool call or textual control action. A control JSON
must be on one line, and nothing may follow COMMAND.

==== Initial Environment Observation ====
{observation}
</TASK_INSTRUCTIONS>"""


MINI_SWE_FINAL_PROMPT = """The executor has reached its current step boundary.
Do not call Bash. Return one evidence-backed finish, revise, or stop SAE handoff.

- finish: the patch is ready for MainAgent to submit.
- revise: preserve the current workspace and request the concrete adjustment needed.
- stop: abandon this path.

Current progress: step {current_step}/{max_steps}.
Latest observation:
{observation}

Output exactly:
DISCUSSION
<concise final judgment>

COMMAND
<finish|revise|stop> <single-line SAE JSON>

SAE JSON schema:
{sae_schema}

Nothing may follow COMMAND."""


def build_mini_swe_task_prompt(
    *,
    original_question: str,
    task_instruction: str,
    context: str,
    observation: Any,
    current_step: int,
    max_steps: int,
    remaining_steps: int,
    budget_warning: str,
    workspace_path: str = "/testbed",
) -> str:
    return MINI_SWE_TASK_PROMPT.replace("/testbed", workspace_path).format(
        original_question=original_question,
        task_instruction=task_instruction,
        context=context or "No additional context provided.",
        observation=_observation_text(observation),
        current_step=current_step,
        max_steps=max_steps,
        remaining_steps=remaining_steps,
        budget_warning=budget_warning,
        sae_schema=SWE_SAE_SCHEMA,
        revise_example=SWE_REVISE_EXAMPLE,
    )


def build_mini_swe_final_prompt(
    *, observation: Any, current_step: int, max_steps: int
) -> str:
    return MINI_SWE_FINAL_PROMPT.format(
        observation=_observation_text(observation),
        current_step=current_step,
        max_steps=max_steps,
        sae_schema=SWE_SAE_SCHEMA,
    )


def build_mini_swe_bash_tool(workspace_path: str = "/testbed") -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": (
                    "Run one real non-interactive Bash command in the SWE-bench "
                    f"container. Include `cd {workspace_path}` when accessing the repository."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Complete Bash command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    ]


def _observation_text(observation: Any) -> str:
    if isinstance(observation, dict):
        return str(
            observation.get("output") or observation.get("message") or observation
        )
    return str(observation)

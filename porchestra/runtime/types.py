"""Passive runtime records and backend boundaries; no scheduling implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol


class NodeState(str, Enum):
    ACTIVE = "ACTIVE"
    WAITING = "WAITING"
    VERIFYING = "VERIFYING"
    DEAD = "DEAD"


@dataclass(frozen=True)
class AcceptanceCheck:
    check_id: str
    command: str
    expected_exit_code: int = 0
    timeout_seconds: float = 120.0
    # None means no output matcher; an empty/zero-test run is not proof by itself.
    expected_output_pattern: str | None = None
    import_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskContract:
    objective: str
    acceptance_checks: tuple[AcceptanceCheck, ...]
    budget_limit: int  # Total reported LLM tokens for this entire subtree.
    contract_id: str
    parent_contract_id: str | None = None
    version: int = 1
    constraints: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    deadline_seconds: float = 600.0  # Relative lifetime; never extends parent deadline.
    scope: tuple[str, ...] = ()  # Explicit repository paths owned by a helper.
    independent: bool = False  # Parent declares this duty independently verifiable.


@dataclass(frozen=True)
class WorkspaceSnapshot:
    workspace_id: str  # Globally unique across runs.
    revision: str  # Includes uncommitted/overlay state, not just Git HEAD.
    artifact_ref: str


@dataclass(frozen=True)
class WorkspaceDiff:
    base: WorkspaceSnapshot
    tip: WorkspaceSnapshot
    artifact_ref: str  # Includes tracked, untracked, deleted and binary changes.


@dataclass(frozen=True)
class VerifiedFact:
    key: str
    value: Any
    evidence_refs: tuple[str, ...]
    source_snapshot: WorkspaceSnapshot
    verification_id: str  # Issued by the verifier, not accepted from agent prose.


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    exit_code: int | None
    passed: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    category: Literal["framework_check", "task_check"] = "task_check"
    framework_output: str = ""


@dataclass(frozen=True)
class VerificationReport:
    verification_id: str
    node_id: str
    contract_id: str
    contract_version: int
    snapshot: WorkspaceSnapshot
    passed: bool
    checks: tuple[CheckResult, ...]
    error_log: str = ""
    framework_checks: tuple[CheckResult, ...] = ()


@dataclass(frozen=True)
class NodeResult:
    node_id: str
    outcome: Literal["verified", "failed", "timed_out", "cancelled"]
    reason: str
    diff: WorkspaceDiff | None = None
    verification: VerificationReport | None = None
    facts: tuple[VerifiedFact, ...] = ()
    artifact_refs: tuple[str, ...] = ()


@dataclass
class NodeRecord:
    node_id: str
    run_id: str
    parent_id: str | None
    depth: int
    contract: TaskContract
    workspace_id: str
    base_snapshot: WorkspaceSnapshot
    deadline_monotonic: float
    state: NodeState = NodeState.ACTIVE
    children: set[str] = field(default_factory=set)
    pending_children: set[str] = field(default_factory=set)
    steps_used: int = 0
    tokens_used: int = 0  # Attributed subtree LLM consumption; never reset on wake.
    unchanged_diff_steps: int = 0
    last_diff_fingerprint: str | None = None
    health_split_version: int | None = None  # Latch: once per contract version.
    wait_reason: str | None = None
    checkpoint_ref: str | None = None
    progress_seen: set[str] = field(default_factory=set)
    progress_log: list[dict[str, Any]] = field(default_factory=list)
    rejected_spawn_reasons: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class StepOutcome:
    kind: Literal["continue", "finish", "spawn", "pause", "stop"]
    reason: str = ""
    child_contracts: tuple[TaskContract, ...] = ()
    tokens_used: int = 0  # Driver supplies actual input + output usage for this step.
    progress: dict[str, Any] = field(default_factory=dict)
    recommendation: Literal["continue", "request_help", "converge"] = "continue"


@dataclass(frozen=True)
class HealthDecision:
    split_required: bool
    reason: str
    unchanged_diff_steps: int
    budget_ratio: float
    status: str = "investigating"


@dataclass(frozen=True)
class MergeReport:
    merge_id: str
    child_id: str
    parent_id: str | None
    committed: bool
    resulting_snapshot: WorkspaceSnapshot | None = None
    conflicts: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()


class AgentDriver(Protocol):
    """One bounded action per step; adapters retain the existing agent history."""

    async def step(self, node: NodeRecord) -> StepOutcome: ...
    async def inject_context(self, node_id: str, message: str) -> None: ...
    async def checkpoint(self, node_id: str) -> str: ...
    async def cancel(self, node_id: str, *, reason: str) -> None: ...
    async def close(self, node_id: str) -> None: ...


class WorkspaceBackend(Protocol):
    """Git snapshots for SWE; observable state revisions for ARE; no hidden oracle."""

    async def snapshot(self, workspace_id: str) -> WorkspaceSnapshot: ...
    async def diff_fingerprint(self, workspace_id: str) -> str: ...
    async def fork(self, base: WorkspaceSnapshot, *, node_id: str) -> str: ...
    async def export_diff(
        self, base: WorkspaceSnapshot, workspace_id: str
    ) -> WorkspaceDiff: ...
    async def merge_three_way(
        self, parent: NodeRecord, child: NodeResult
    ) -> MergeReport: ...
    async def release(self, workspace_id: str) -> None: ...


class CheckRunner(Protocol):
    """Run real commands on a pinned verifier checkout; bound/kill subprocesses."""

    async def run(
        self,
        check: AcceptanceCheck,
        snapshot: WorkspaceSnapshot,
        *,
        deadline_monotonic: float,
    ) -> CheckResult: ...

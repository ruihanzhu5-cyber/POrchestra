"""Event-driven node scheduling with token quotas and supervised lifecycle jobs."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from .contracts import ContractGenerator, ContractVerifier
from .handoff import summarize
from .health import HealthMonitor
from .lifecycle import EventSink, StateMerger
from .memory import ToolGateway
from .types import (
    AgentDriver,
    HealthDecision,
    NodeRecord,
    NodeResult,
    NodeState,
    StepOutcome,
    TaskContract,
    VerificationReport,
    WorkspaceBackend,
)


@dataclass(frozen=True)
class SpawnBudgetPolicy:
    # Conservative byte upper bound for prompt text; not a tokenizer claim.
    prompt_overhead_tokens: int = 4096
    response_tokens: int = 4096
    tool_tokens: int = 2048
    parent_min_tokens: int = 8192
    parent_reserve_ratio: float = 0.20

    def minimum_child(self, contract: TaskContract, shared_context: str = "") -> int:
        from dataclasses import asdict

        prompt_bytes = len(
            (json.dumps(asdict(contract), ensure_ascii=False) + shared_context).encode()
        )
        return (
            prompt_bytes
            + self.prompt_overhead_tokens
            + self.response_tokens
            + self.tool_tokens
        )

    def parent_reserve(self, contract: TaskContract, remaining: int) -> int:
        return max(
            self.parent_min_tokens,
            math.ceil(remaining * self.parent_reserve_ratio),
            self.minimum_child(contract),
        )


@dataclass(frozen=True)
class RuntimeLimits:
    max_active: int = 4
    max_depth: int = 3
    max_nodes: int = 32
    step_timeout_seconds: float = 120.0
    verifier_timeout_seconds: float = 300.0
    split_timeout_seconds: float = 60.0
    cleanup_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in ("max_active", "max_nodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_depth) is not int or self.max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        for name in (
            "step_timeout_seconds",
            "verifier_timeout_seconds",
            "split_timeout_seconds",
            "cleanup_timeout_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class NodeRegistry:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeRecord] = {}
        self.results: dict[str, NodeResult] = {}
        self.total_registered = 0
        self._lock = threading.RLock()

    def register(self, node: NodeRecord) -> None:
        with self._lock:
            if (
                node.node_id in self.nodes
                or node.node_id in self.results
                or node.state is NodeState.DEAD
            ):
                raise ValueError("Node identity is already used or DEAD")
            if node.parent_id is not None and node.parent_id not in self.nodes:
                raise ValueError("Parent must be registered before its children")
            self.nodes[node.node_id] = node
            self.total_registered += 1

    def get(self, node_id: str) -> NodeRecord:
        with self._lock:
            return self.nodes[node_id]

    def transition(
        self, node_id: str, *, expected: NodeState, target: NodeState
    ) -> None:
        allowed = {
            NodeState.ACTIVE: {NodeState.WAITING, NodeState.VERIFYING, NodeState.DEAD},
            NodeState.WAITING: {NodeState.ACTIVE, NodeState.DEAD},
            NodeState.VERIFYING: {NodeState.ACTIVE, NodeState.DEAD},
            NodeState.DEAD: set(),
        }
        with self._lock:
            node = self.nodes[node_id]
            if node.state is not expected or target not in allowed[expected]:
                raise ValueError(f"Illegal transition {node.state} -> {target}")
            node.state = target

    def remove(self, node_id: str, *, result: NodeResult) -> None:
        with self._lock:
            if node_id in self.results:
                return
            if (
                self.nodes[node_id].state is not NodeState.DEAD
                or result.node_id != node_id
            ):
                raise ValueError(
                    "Only DEAD nodes with matching receipts can be removed"
                )
            self.results[node_id] = result
            del self.nodes[node_id]


class BudgetLedger:
    """Token reservations are transferred, never duplicated, along the tree."""

    def __init__(self, root_limit: int) -> None:
        if type(root_limit) is not int or root_limit < 1:
            raise ValueError("root_limit must be a positive integer token count")
        self.root_limit = root_limit
        self.used_by_node: dict[str, int] = {}
        self.reserved_by_node: dict[str, int] = {}
        self.parent_by_node: dict[str, str | None] = {}
        self.limits_by_node: dict[str, int] = {}
        self._settled: set[str] = set()

    def register_root(self, node_id: str, contract: TaskContract) -> None:
        if self.limits_by_node or contract.budget_limit > self.root_limit:
            raise ValueError("Ledger already bound or root budget exceeds its limit")
        self.limits_by_node[node_id] = contract.budget_limit
        self.used_by_node[node_id] = self.reserved_by_node[node_id] = 0
        self.parent_by_node[node_id] = None

    def remaining(self, node_id: str) -> int:
        if node_id in self._settled:
            return 0
        return max(
            0,
            self.limits_by_node[node_id]
            - self.used_by_node[node_id]
            - self.reserved_by_node[node_id],
        )

    def reserve_children(
        self, parent_id: str, contracts: tuple[TaskContract, ...]
    ) -> None:
        ids = [contract.contract_id for contract in contracts]
        total = sum(contract.budget_limit for contract in contracts)
        if len(set(ids)) != len(ids) or any(
            node_id in self.limits_by_node for node_id in ids
        ):
            raise ValueError("Child identities must be new and unique")
        if total > self.remaining(parent_id) or any(
            c.budget_limit <= 0 for c in contracts
        ):
            raise ValueError("Insufficient unreserved parent tokens")
        self.reserved_by_node[parent_id] += total
        for contract in contracts:
            node_id = contract.contract_id
            self.limits_by_node[node_id] = contract.budget_limit
            self.parent_by_node[node_id] = parent_id
            self.used_by_node[node_id] = self.reserved_by_node[node_id] = 0

    def validate_spawn(self, parent_id, contracts, *, minimums, parent_reserve):
        if len(minimums) != len(contracts) or any(
            c.budget_limit < minimum for c, minimum in zip(contracts, minimums)
        ):
            raise ValueError(
                "Insufficient child startup budget (prompt + tool + response); converge instead"
            )
        if sum(c.budget_limit for c in contracts) + parent_reserve > self.remaining(
            parent_id
        ):
            raise ValueError(
                "Insufficient budget after reserving parent integration/verification/repair tokens; converge instead"
            )

    def charge(self, node_id: str, *, tokens: int) -> None:
        if type(tokens) is not int or tokens < 0 or node_id in self._settled:
            raise ValueError(
                "Token usage must be nonnegative and belong to a live allocation"
            )
        over_budget = tokens > self.remaining(node_id)
        current: str | None = node_id
        while current is not None:
            old_used = self.used_by_node[current]
            self.used_by_node[current] += tokens
            parent = self.parent_by_node[current]
            if parent is not None:
                released = min(tokens, max(0, self.limits_by_node[current] - old_used))
                self.reserved_by_node[parent] -= released
            current = parent
        # Preserve observed usage even if one provider call overshot its quota.
        if over_budget:
            raise ValueError("Reported token usage exceeded the node's remaining quota")

    def settle(self, node_id: str) -> None:
        if node_id in self._settled:
            return
        if self.reserved_by_node[node_id] or any(
            parent == node_id and child not in self._settled
            for child, parent in self.parent_by_node.items()
        ):
            raise ValueError("Settle descendants before their parent")
        parent = self.parent_by_node[node_id]
        if parent is not None:
            self.reserved_by_node[parent] -= max(
                0, self.limits_by_node[node_id] - self.used_by_node[node_id]
            )
        self._settled.add(node_id)


class NodeRuntime:
    """One event-loop owner per runtime; drivers must provide cancelable async I/O."""

    def __init__(
        self,
        *,
        run_id: str,
        registry: NodeRegistry,
        verifier: ContractVerifier,
        contracts: ContractGenerator,
        gateway: ToolGateway,
        health: HealthMonitor,
        merger: StateMerger,
        events: EventSink,
        workspaces: WorkspaceBackend,
        budget: BudgetLedger,
        driver_factory: Callable[[NodeRecord, ToolGateway], AgentDriver],
        limits: RuntimeLimits = RuntimeLimits(),
        spawn_budget_policy: SpawnBudgetPolicy = SpawnBudgetPolicy(),
    ) -> None:
        self.run_id, self.registry, self.verifier = run_id, registry, verifier
        self.contracts, self.gateway, self.health = contracts, gateway, health
        self.merger, self.events, self.workspaces = merger, events, workspaces
        self.gateway.artifact_reader = merger.artifacts
        self.budget, self.limits, self._driver_factory = budget, limits, driver_factory
        self.spawn_budget_policy = spawn_budget_policy
        self._drivers: dict[str, AgentDriver] = {}
        self._ready: asyncio.Queue[str] = asyncio.Queue()
        self._queued: set[str] = set()
        self._jobs: dict[str, asyncio.Task[Any]] = {}
        self._kinds: dict[str, str] = {}
        self._staged: dict[str, NodeResult] = {}
        self._batch_locks: dict[str, asyncio.Lock] = {}
        self._spawn_locks: dict[str, asyncio.Lock] = {}
        self._terminating: set[str] = set()
        self._settling: set[str] = set()
        self._notifications: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._messages: dict[str, list[str]] = {}
        self._running = False

    def _enqueue(self, node_id: str) -> None:
        if node_id not in self._queued:
            self._queued.add(node_id)
            self._ready.put_nowait(node_id)

    def _launch(self, node_id: str, kind: str, job: Awaitable[Any]) -> None:
        if node_id in self._jobs:
            raise RuntimeError(f"A job is already running for {node_id}")
        self._jobs[node_id] = asyncio.create_task(job, name=f"{kind}:{node_id}")
        self._kinds[node_id] = kind

    async def _bounded(self, work: Awaitable[Any], timeout: float) -> Any:
        return await asyncio.wait_for(work, timeout)

    async def _step(self, node: NodeRecord) -> tuple[StepOutcome, HealthDecision]:
        async def execute() -> tuple[StepOutcome, HealthDecision]:
            outcome = await self._drivers[node.node_id].step(node)
            if not isinstance(outcome, StepOutcome):
                raise TypeError("AgentDriver.step must return StepOutcome")
            node.steps_used += 1
            try:
                self.budget.charge(node.node_id, tokens=outcome.tokens_used)
            finally:
                for identity, record in self.registry.nodes.items():
                    record.tokens_used = self.budget.used_by_node[identity]
            fingerprint = await self.workspaces.diff_fingerprint(node.workspace_id)
            decision = self.health.observe(
                node, diff_fingerprint=fingerprint, progress=outcome.progress
            )
            await self.events.record(
                "PROGRESS",
                node,
                decision.status,
                progress=node.progress_log[-1],
                repeated_steps=decision.unchanged_diff_steps,
            )
            return outcome, decision

        return await self._bounded(execute(), self.limits.step_timeout_seconds)

    async def run(self, root: TaskContract, *, workspace_id: str) -> NodeResult:
        if self._running or self.registry.nodes or self.registry.results:
            raise ValueError("Each runtime instance runs exactly one task tree")
        self.verifier.validate_contract(root)
        self._running = True
        root_id = root.contract_id
        notification_task = asyncio.create_task(self._notifications.get())
        try:
            snapshot = await self._bounded(
                self.workspaces.snapshot(workspace_id), self.limits.step_timeout_seconds
            )
            node = NodeRecord(
                root_id,
                self.run_id,
                None,
                0,
                root,
                workspace_id,
                snapshot,
                time.monotonic() + root.deadline_seconds,
            )
            node.last_diff_fingerprint = await self._bounded(
                self.workspaces.diff_fingerprint(workspace_id),
                self.limits.step_timeout_seconds,
            )
            self.budget.register_root(root_id, root)
            self.registry.register(node)
            self._drivers[root_id] = self._driver_factory(node, self.gateway)
            self._enqueue(root_id)
            await self.events.record("SPAWN", node, "root task registered")
            while root_id not in self.registry.results:
                for identity, record in list(self.registry.nodes.items()):
                    if (
                        record.deadline_monotonic <= time.monotonic()
                        and identity not in self._terminating
                        and identity not in self._settling
                    ):
                        await self._cancel_job(identity)
                        if identity not in self.registry.nodes:
                            continue
                        self._mark_terminating(identity)
                        self._launch(
                            identity,
                            "terminate",
                            self.terminate(
                                identity, reason="node deadline expired", timed_out=True
                            ),
                        )
                slots = self.limits.max_active - sum(
                    kind == "step" for kind in self._kinds.values()
                )
                while slots > 0 and not self._ready.empty():
                    identity = self._ready.get_nowait()
                    self._queued.discard(identity)
                    record = self.registry.nodes.get(identity)
                    if (
                        record is None
                        or record.state is not NodeState.ACTIVE
                        or identity in self._jobs
                        or identity in self._terminating
                    ):
                        continue
                    if self.budget.remaining(identity) <= 0:
                        self._terminating.add(identity)
                        self._launch(
                            identity,
                            "terminate",
                            self.terminate(identity, reason="token budget exhausted"),
                        )
                    else:
                        self._launch(identity, "step", self._step(record))
                        slots -= 1
                deadlines = [
                    n.deadline_monotonic
                    for n in self.registry.nodes.values()
                    if n.node_id not in self._terminating
                    and n.node_id not in self._settling
                ]
                timeout = (
                    max(0, min(deadlines) - time.monotonic()) if deadlines else None
                )
                done, _ = await asyncio.wait(
                    (*self._jobs.values(), notification_task),
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if notification_task in done:
                    identity, message = notification_task.result()
                    if identity in self.registry.nodes:
                        self.gateway.board.invalidate(
                            self.registry.get(identity).workspace_id
                        )
                        self._messages.setdefault(identity, []).append(message)
                        if (
                            identity not in self._jobs
                            and identity not in self._terminating
                        ):
                            self._launch(
                                identity,
                                "notification",
                                self._deliver_messages(identity),
                            )
                    notification_task = asyncio.create_task(self._notifications.get())
                for identity, task in list(self._jobs.items()):
                    if task not in done:
                        continue
                    kind = self._kinds.pop(identity)
                    del self._jobs[identity]
                    if identity not in self.registry.nodes:
                        if not task.cancelled():
                            task.exception()
                        continue
                    record = self.registry.get(identity)
                    try:
                        result = task.result()
                        if kind == "step":
                            outcome, health = result
                            if outcome.kind == "finish":
                                self._launch(
                                    identity,
                                    "verify",
                                    self._verify_and_complete(identity),
                                )
                            elif outcome.kind == "stop":
                                self._terminating.add(identity)
                                self._launch(
                                    identity,
                                    "terminate",
                                    self.terminate(
                                        identity,
                                        reason=outcome.reason or "agent stopped",
                                    ),
                                )
                            elif outcome.kind == "spawn":
                                self._launch(
                                    identity,
                                    "spawn",
                                    self._bounded(
                                        self.spawn(
                                            identity,
                                            outcome.child_contracts,
                                            reason=outcome.reason or "agent delegation",
                                        ),
                                        self.limits.split_timeout_seconds,
                                    ),
                                )
                            elif health.split_required:
                                self._launch(
                                    identity,
                                    "split",
                                    self._bounded(
                                        self.forced_split(identity, health),
                                        self.limits.split_timeout_seconds,
                                    ),
                                )
                            elif outcome.kind == "pause":
                                self._launch(
                                    identity, "pause", self._pause(identity, "external")
                                )
                            elif outcome.kind != "continue":
                                raise ValueError(
                                    f"Unknown driver action {outcome.kind}"
                                )
                        if (
                            identity not in self._jobs
                            and identity in self.registry.nodes
                        ):
                            if self._messages.get(identity):
                                self._launch(
                                    identity,
                                    "notification",
                                    self._deliver_messages(identity),
                                )
                            elif record.state is NodeState.ACTIVE:
                                self._enqueue(identity)
                    except (Exception, asyncio.CancelledError) as exc:
                        if kind == "terminate":
                            raise RuntimeError(
                                f"Lifecycle cleanup failed for {identity}"
                            ) from exc
                        if kind in {"split", "spawn"}:
                            if (
                                isinstance(exc, ValueError)
                                and record.state is NodeState.ACTIVE
                                and not record.pending_children
                            ):
                                reason = str(exc)
                                if reason not in record.rejected_spawn_reasons:
                                    await self.events.record(
                                        "SPLIT_REJECTED", record, reason
                                    )
                                    record.rejected_spawn_reasons.add(reason)
                                await self._drivers[identity].inject_context(
                                    identity,
                                    f"Helper request rejected (recoverable): {reason}. "
                                    "Do not repeat the rejected plan. Continue locally or converge with existing evidence.",
                                )
                                self.health.acknowledge_split(record)
                                self._enqueue(identity)
                                continue
                            await self.events.record("SPLIT_REJECTED", record, str(exc))
                        self._terminating.add(identity)
                        self._launch(
                            identity,
                            "terminate",
                            self.terminate(
                                identity,
                                reason=f"{kind}: {type(exc).__name__}: {exc}",
                                timed_out=isinstance(exc, TimeoutError),
                            ),
                        )
            return self.registry.results[root_id]
        finally:
            notification_task.cancel()
            await asyncio.gather(notification_task, return_exceptions=True)
            for identity in list(self._jobs):
                await self._cancel_job(identity)
            for record in sorted(
                list(self.registry.nodes.values()), key=lambda n: n.depth
            ):
                if record.node_id in self.registry.nodes:
                    await self.terminate(record.node_id, reason="runtime shutdown")
            self._running = False

    async def _pause(self, node_id: str, reason: str) -> None:
        node = self.registry.get(node_id)
        self.registry.transition(
            node_id, expected=NodeState.ACTIVE, target=NodeState.WAITING
        )
        node.wait_reason = reason
        node.checkpoint_ref = await self._bounded(
            self._drivers[node_id].checkpoint(node_id), self.limits.step_timeout_seconds
        )
        await self.events.record(
            "SLEEP", node, reason, checkpoint_ref=node.checkpoint_ref
        )

    def _mark_terminating(self, node_id: str) -> None:
        self._terminating.add(node_id)
        node = self.registry.nodes.get(node_id)
        if node is not None:
            for child_id in node.pending_children:
                self._mark_terminating(child_id)

    async def spawn(
        self, parent_id: str, contracts: tuple[TaskContract, ...], *, reason: str
    ) -> tuple[str, ...]:
        async with self._spawn_locks.setdefault(parent_id, asyncio.Lock()):
            parent = self.registry.get(parent_id)
            if (
                parent.state is not NodeState.ACTIVE
                or parent.pending_children
                or not contracts
            ):
                raise ValueError(
                    "Spawn requires an ACTIVE node and a nonempty fresh batch"
                )
            if (
                parent.depth >= self.limits.max_depth
                or self.registry.total_registered + len(contracts)
                > self.limits.max_nodes
            ):
                raise ValueError("Recursive depth or lifetime node cap reached")
            for contract in contracts:
                self.verifier.validate_contract(contract, parent=parent.contract)
            self.contracts.validate_delegation(parent.contract, contracts)
            handoff = await self._build_handoff(parent)
            context = json.dumps(handoff, ensure_ascii=False)
            self.budget.validate_spawn(
                parent_id,
                contracts,
                minimums=[
                    self.spawn_budget_policy.minimum_child(c, context)
                    for c in contracts
                ],
                parent_reserve=self.spawn_budget_policy.parent_reserve(
                    parent.contract, self.budget.remaining(parent_id)
                ),
            )
            if sum(c.budget_limit for c in contracts) >= self.budget.remaining(
                parent_id
            ):
                raise ValueError("Child quotas must leave parent repair tokens")
            await self._pause(parent_id, "children")
            self.budget.reserve_children(parent_id, contracts)
            prepared: list[tuple[NodeRecord, AgentDriver]] = []
            created_workspaces: list[str] = []
            committed = False
            try:
                base = await self.workspaces.snapshot(parent.workspace_id)
                for contract in contracts:
                    workspace = await self.workspaces.fork(
                        base, node_id=contract.contract_id
                    )
                    created_workspaces.append(workspace)
                    node = NodeRecord(
                        contract.contract_id,
                        self.run_id,
                        parent_id,
                        parent.depth + 1,
                        contract,
                        workspace,
                        base,
                        min(
                            parent.deadline_monotonic,
                            time.monotonic() + contract.deadline_seconds,
                        ),
                    )
                    node.last_diff_fingerprint = await self.workspaces.diff_fingerprint(
                        workspace
                    )
                    node.handoff = {
                        **handoff,
                        "changes": {
                            **handoff["changes"],
                            "state": "inherited_from_parent",
                        },
                    }
                    driver = self._driver_factory(node, self.gateway)
                    prepared.append((node, driver))
                    await driver.inject_context(
                        node.node_id,
                        "PARENT_HANDOFF "
                        + json.dumps(node.handoff, ensure_ascii=False),
                    )
                for node, driver in prepared:
                    self.registry.register(node)
                    self._drivers[node.node_id] = driver
                    parent.children.add(node.node_id)
                    parent.pending_children.add(node.node_id)
                committed = True
                for node, _ in prepared:
                    await self.events.record(
                        "SPAWN", node, reason, budget_limit=node.contract.budget_limit
                    )
                    self._enqueue(node.node_id)
                return tuple(node.node_id for node, _ in prepared)
            except BaseException:
                if not committed:
                    for node, driver in prepared:
                        await driver.close(node.node_id)
                    for workspace in created_workspaces:
                        await self.workspaces.release(workspace)
                    for contract in contracts:
                        self.budget.settle(contract.contract_id)
                raise

    async def forced_split(
        self, node_id: str, trigger: HealthDecision
    ) -> tuple[str, ...]:
        node = self.registry.get(node_id)
        remaining = self.budget.remaining(node_id)
        retained = max(1, remaining // 5)
        children = await self.contracts.split(
            node, trigger, available_budget=remaining - retained
        )
        if not children:
            await self._drivers[node_id].inject_context(
                node_id,
                f"Health: {trigger.reason}. Continue investigating or converge. "
                "Request helpers only with distinct objectives, DoD and disjoint scope; no automatic task copies.",
            )
            self.health.acknowledge_split(node)
            return ()
        ids = await self.spawn(node_id, children, reason=trigger.reason)
        self.health.acknowledge_split(node)
        return ids

    async def _build_handoff(self, node):
        describe = getattr(self.workspaces, "describe_changes", None)
        summary = (
            await describe(node.base_snapshot, node.workspace_id) if describe else ""
        )
        return summarize(node, change_summary=summary)

    async def request_finish(self, node_id: str) -> VerificationReport:
        node = self.registry.get(node_id)
        if node.pending_children:
            raise ValueError("Cannot finish with unsettled children")
        self.registry.transition(
            node_id, expected=NodeState.ACTIVE, target=NodeState.VERIFYING
        )
        return await self._bounded(
            self.verifier.verify(node), self.limits.verifier_timeout_seconds
        )

    async def _verify_and_complete(self, node_id: str) -> None:
        report = await self.request_finish(node_id)
        node = self.registry.get(node_id)
        if report.passed:
            await self.events.record(
                "VERIFY_PASS",
                node,
                "all DoD checks passed",
                verification_id=report.verification_id,
            )
            await self.complete(node_id, report)
        else:
            await self.events.record("VERIFY_REJECT", node, report.error_log)
            if any(not c.passed for c in report.framework_checks):
                await self.terminate(
                    node_id,
                    reason="framework_check: " + self.verifier.format_feedback(report),
                )
                return
            await self._drivers[node_id].inject_context(
                node_id, self.verifier.format_feedback(report)
            )
            self.registry.transition(
                node_id, expected=NodeState.VERIFYING, target=NodeState.ACTIVE
            )

    async def complete(self, node_id: str, report: VerificationReport) -> NodeResult:
        node = self.registry.get(node_id)
        if node.state is not NodeState.VERIFYING or node.pending_children:
            raise ValueError("Completion requires a VERIFYING leaf")
        result = await self._bounded(
            self.merger.collect(node, report=report),
            self.limits.cleanup_timeout_seconds,
        )
        if node.parent_id is None:
            await self.destroy(node_id, result=result)
        else:
            self._staged[node_id] = result
            await self._settle_batch(node.parent_id)
        return result

    async def _settle_batch(self, parent_id: str) -> None:
        async with self._batch_locks.setdefault(parent_id, asyncio.Lock()):
            parent = self.registry.nodes.get(parent_id)
            if (
                parent is None
                or parent_id in self._terminating
                or not parent.pending_children
            ):
                return
            batch = sorted(parent.pending_children)
            if any(identity not in self._staged for identity in batch):
                return
            results = [self._staged[identity] for identity in batch]
            self._settling.update(batch)
            all_passed = all(result.outcome == "verified" for result in results)
            messages = []
            for result in results:
                merge = None
                child_contract = self.registry.get(result.node_id).contract
                independent = (
                    result.outcome == "verified" and child_contract.independent
                )
                if all_passed or independent:
                    merge = await self._bounded(
                        self.merger.merge(
                            parent,
                            result,
                            scope=child_contract.scope if independent else None,
                        ),
                        self.limits.cleanup_timeout_seconds,
                    )
                messages.append(
                    {
                        "node_id": result.node_id,
                        "outcome": result.outcome,
                        "reason": result.reason,
                        "artifacts": result.artifact_refs,
                        "merged": merge.committed if merge else False,
                        "conflicts": merge.conflicts if merge else (),
                        "facts": [fact.key for fact in result.facts],
                        "handoff": result.handoff,
                    }
                )
                await self.destroy(result.node_id, result=result)
                self._settling.discard(result.node_id)
                self._staged.pop(result.node_id, None)
            await self.wake(
                parent_id,
                reason="child batch settled",
                message=json.dumps(messages, ensure_ascii=False),
            )

    async def wake(self, node_id: str, *, reason: str, message: str) -> None:
        node = self.registry.get(node_id)
        if node.pending_children or node_id in self._terminating:
            return
        if node.state is not NodeState.WAITING:
            raise ValueError("Only WAITING nodes can wake")
        await self._drivers[node_id].inject_context(node_id, message)
        fingerprint = await self.workspaces.diff_fingerprint(node.workspace_id)
        self.health.reset_after_wake(node, diff_fingerprint=fingerprint)
        node.wait_reason = None
        self.registry.transition(
            node_id, expected=NodeState.WAITING, target=NodeState.ACTIVE
        )
        await self.events.record("WAKE", node, reason)
        self._enqueue(node_id)

    async def notify(self, node_id: str, *, message: str) -> None:
        await self._notifications.put((node_id, message))

    async def _deliver_messages(self, node_id: str) -> None:
        node = self.registry.get(node_id)
        messages = self._messages.pop(node_id, [])
        if node.state is NodeState.WAITING and not node.pending_children:
            await self.wake(
                node_id, reason="external notification", message="\n".join(messages)
            )
        else:
            await self._drivers[node_id].inject_context(node_id, "\n".join(messages))

    async def destroy(self, node_id: str, *, result: NodeResult) -> None:
        if node_id in self.registry.results:
            return
        node = self.registry.get(node_id)
        cleanup_error = ""
        try:
            await self._bounded(
                self.merger.release(node, self._drivers[node_id]),
                self.limits.cleanup_timeout_seconds,
            )
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
            result = replace(
                result, reason=f"{result.reason}; cleanup: {cleanup_error}"
            )
        await self.events.record(
            "DESTROY",
            node,
            result.reason,
            outcome=result.outcome,
            cleanup_error=cleanup_error,
        )
        self.registry.transition(node_id, expected=node.state, target=NodeState.DEAD)
        self.budget.settle(node_id)
        self._drivers.pop(node_id, None)
        self.registry.remove(node_id, result=result)
        if node.parent_id in self.registry.nodes:
            self.registry.get(node.parent_id).pending_children.discard(node_id)

    async def _cancel_job(self, node_id: str) -> None:
        task = self._jobs.pop(node_id, None)
        self._kinds.pop(node_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await self._bounded(
                    asyncio.gather(task, return_exceptions=True),
                    self.limits.cleanup_timeout_seconds,
                )
            except TimeoutError:
                pass

    async def terminate(
        self, node_id: str, *, reason: str, timed_out: bool = False
    ) -> NodeResult:
        if node_id in self.registry.results:
            return self.registry.results[node_id]
        self._terminating.add(node_id)
        node = self.registry.get(node_id)
        if timed_out:
            await self.events.record("TIMEOUT", node, reason)
        for child_id in list(node.pending_children):
            if child_id in self.registry.nodes:
                self._terminating.add(child_id)
                await self._cancel_job(child_id)
                await self.terminate(child_id, reason=f"ancestor terminated: {reason}")
        driver = self._drivers.get(node_id)
        if driver is not None:
            try:
                await self._bounded(
                    driver.cancel(node_id, reason=reason),
                    self.limits.cleanup_timeout_seconds,
                )
            except Exception:
                pass
        try:
            result = await self._bounded(
                self.merger.collect(
                    node, outcome="timed_out" if timed_out else "failed", reason=reason
                ),
                self.limits.cleanup_timeout_seconds,
            )
        except Exception as exc:
            result = await self.events.persist_result(
                NodeResult(
                    node_id,
                    "timed_out" if timed_out else "failed",
                    f"{reason}; partial collection failed: {exc}",
                )
            )
        if node.parent_id is None or node.parent_id in self._terminating:
            await self.destroy(node_id, result=result)
        else:
            self._staged[node_id] = result
            await self._settle_batch(node.parent_id)
        return result

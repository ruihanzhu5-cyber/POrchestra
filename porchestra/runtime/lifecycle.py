"""Durable result handoff, serialized three-way merges and topology JSONL events."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from .memory import GlobalMemoryBoard
from .types import (
    AgentDriver,
    MergeReport,
    NodeRecord,
    NodeResult,
    VerificationReport,
    WorkspaceBackend,
)

EventName = Literal[
    "PROGRESS",
    "SPAWN",
    "SLEEP",
    "VERIFY_PASS",
    "VERIFY_REJECT",
    "MERGE_COMMIT",
    "MERGE_CONFLICT",
    "DESTROY",
    "WAKE",
    "TIMEOUT",
    "SPLIT_REJECTED",
]


@dataclass(frozen=True)
class RuntimeEvent:
    event: EventName
    event_id: str
    run_id: str
    sequence: int
    timestamp_utc: str
    node_id: str
    parent_id: str | None
    reason: str
    depth: int
    contract_version: int
    details: dict[str, Any]


class EventSink:
    def __init__(self, path: str, *, console: bool = True) -> None:
        self.path = str(path)
        self.console = console
        self._lock = threading.RLock()
        self._sequences: dict[str, int] = {}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                run = record["run_id"]
                self._sequences[run] = max(
                    self._sequences.get(run, 0), record["sequence"]
                )

    async def emit(self, event: RuntimeEvent) -> None:
        await asyncio.to_thread(self._append, event)

    def _append(self, event: RuntimeEvent) -> None:
        with self._lock:
            sequence = self._sequences.get(event.run_id, 0) + 1
            record = asdict(replace(event, sequence=sequence))
            line = json.dumps(record, ensure_ascii=False, default=str)
            with Path(self.path).open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._sequences[event.run_id] = sequence
            if self.console:
                print(line, flush=True)

    async def record(
        self, event: EventName, node: NodeRecord, reason: str, **details: Any
    ) -> None:
        await self.emit(
            RuntimeEvent(
                event,
                uuid4().hex,
                node.run_id,
                0,
                datetime.now(timezone.utc).isoformat(),
                node.node_id,
                node.parent_id,
                reason,
                node.depth,
                node.contract.version,
                details,
            )
        )

    async def persist_result(self, result: NodeResult) -> NodeResult:
        """Write a durable receipt before any worker/workspace is released."""
        path = Path(self.path).parent / "artifacts" / f"{uuid4().hex}.json"

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(asdict(result), stream, ensure_ascii=False, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)

        await asyncio.to_thread(write)
        return replace(result, artifact_refs=(*result.artifact_refs, str(path)))


class StateMerger:
    def __init__(
        self, workspaces: WorkspaceBackend, board: GlobalMemoryBoard, events: EventSink
    ) -> None:
        self.workspaces = workspaces
        self.board = board
        self.events = events
        self._merge_locks: dict[str, asyncio.Lock] = {}
        self._release_locks: dict[str, asyncio.Lock] = {}
        self._merged: dict[tuple[str, str], MergeReport] = {}
        self._released: set[str] = set()

    async def collect(
        self,
        node: NodeRecord,
        *,
        outcome: Literal["verified", "failed", "timed_out", "cancelled"] = "verified",
        reason: str = "Task verified",
        report: VerificationReport | None = None,
    ) -> NodeResult:
        if outcome == "verified":
            if report is None or not report.passed or report.node_id != node.node_id:
                raise ValueError(
                    "Verified collection requires this node's approved report"
                )
            if (report.contract_id, report.contract_version) != (
                node.contract.contract_id,
                node.contract.version,
            ):
                raise ValueError("Verification belongs to another contract version")
        diff = await self.workspaces.export_diff(node.base_snapshot, node.workspace_id)
        if diff.tip.workspace_id != node.workspace_id:
            raise ValueError("Exported diff belongs to a different workspace")
        if outcome == "verified" and (
            report.snapshot.workspace_id != diff.tip.workspace_id
            or report.snapshot.revision != diff.tip.revision
        ):
            raise ValueError("Workspace changed after verification")
        facts = self.board.verified_facts(
            node.workspace_id, snapshot=report.snapshot if report else diff.tip
        )
        result = NodeResult(
            node.node_id,
            outcome,
            reason,
            diff=diff,
            verification=report,
            facts=facts,
            artifact_refs=(diff.artifact_ref, diff.tip.artifact_ref),
        )
        return await self.events.persist_result(result)

    async def merge(self, parent: NodeRecord, result: NodeResult) -> MergeReport:
        key = (parent.node_id, result.node_id)
        lock = self._merge_locks.setdefault(parent.node_id, asyncio.Lock())
        async with lock:
            if key in self._merged:
                return self._merged[key]
            if (
                result.outcome != "verified"
                or result.verification is None
                or not result.verification.passed
            ):
                raise ValueError("Unverified results cannot be automatically merged")
            if (
                result.diff is None
                or result.diff.tip.revision != result.verification.snapshot.revision
            ):
                raise ValueError("Diff does not match verified evidence")
            try:
                merged = await self.workspaces.merge_three_way(parent, result)
                if (
                    merged.child_id != result.node_id
                    or merged.parent_id != parent.node_id
                ):
                    raise ValueError("Backend returned a merge for a different node")
            except Exception as exc:
                merged = MergeReport(
                    uuid4().hex,
                    result.node_id,
                    parent.node_id,
                    False,
                    conflicts=(f"{type(exc).__name__}: {exc}",),
                    artifact_refs=result.artifact_refs,
                )
            # Even failed backends may have staged changes; never serve an old cache.
            self.board.invalidate(parent.workspace_id)
            self._merged[key] = merged
            await self.events.record(
                "MERGE_COMMIT" if merged.committed else "MERGE_CONFLICT",
                parent,
                "child patch merged"
                if merged.committed
                else "child patch needs conflict resolution",
                child_id=result.node_id,
                merge_id=merged.merge_id,
                conflicts=merged.conflicts,
                artifact_refs=merged.artifact_refs,
            )
            # Facts retain their child snapshot provenance in the durable result.
            # Parent verification must approve them before publication as parent facts.
            return merged

    async def release(self, node: NodeRecord, driver: AgentDriver) -> None:
        lock = self._release_locks.setdefault(node.node_id, asyncio.Lock())
        async with lock:
            if node.node_id in self._released:
                return
            self.board.release_workspace(node.workspace_id)
            failures = []
            # Stop the worker before deleting a workspace it may still reference.
            for cleanup in (
                lambda: driver.close(node.node_id),
                lambda: self.workspaces.release(node.workspace_id),
            ):
                try:
                    await cleanup()
                except Exception as error:
                    failures.append(str(error))
            if failures:
                raise RuntimeError("; ".join(failures))
            self._released.add(node.node_id)

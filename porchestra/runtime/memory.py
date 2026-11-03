"""Thread-safe, versioned tool caching with cross-loop single-flight delivery."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sys
import threading
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import Future as ThreadFuture
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from .types import VerificationReport, VerifiedFact, WorkspaceSnapshot


@dataclass(frozen=True)
class CacheEntry:
    workspace_id: str
    generation: int
    value: Any
    cacheable: bool = True


@dataclass(frozen=True)
class ToolRequest:
    run_id: str
    node_id: str
    workspace_id: str
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    force_refresh: bool = False
    version_provider: Callable[[], Awaitable[tuple[str, str]]] | None = None


@dataclass(frozen=True)
class ToolResult:
    value: Any
    cache_hit: bool = False
    annotation: str = ""


async def _wait_shared(future: ThreadFuture) -> Any:
    # Each loop owns its asyncio.Future; only the thread-safe bridge crosses loops.
    wrapped = asyncio.wrap_future(future)
    wrapped.add_done_callback(
        lambda done: None if done.cancelled() else done.exception()
    )
    return await asyncio.shield(wrapped)


class _WorkspaceGate:
    """Fair async read/write barrier shared across gateway instances and threads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._readers = 0
        self._writer = False
        self._queue: deque[tuple[bool, ThreadFuture]] = deque()

    def _drain(self) -> None:
        if self._writer:
            return
        while self._queue:
            write, ticket = self._queue[0]
            if write and self._readers:
                return
            self._queue.popleft()
            if write:
                self._writer = True
            else:
                self._readers += 1
            ticket.set_result(None)
            if write:
                return

    def _release(self, write: bool) -> None:
        with self._lock:
            if write:
                self._writer = False
            else:
                self._readers -= 1
            self._drain()

    @asynccontextmanager
    async def hold(self, *, write: bool) -> AsyncIterator[None]:
        ticket: ThreadFuture = ThreadFuture()
        with self._lock:
            self._queue.append((write, ticket))
            self._drain()
        try:
            await _wait_shared(ticket)
        except BaseException:
            with self._lock:
                if ticket.done():
                    self._release(write)
                else:
                    self._queue.remove((write, ticket))
                    ticket.cancel()
                    self._drain()
            raise
        try:
            yield
        finally:
            self._release(write)


@dataclass
class _Flight:
    result: ThreadFuture
    loop: asyncio.AbstractEventLoop
    workspace_id: str | None
    task: asyncio.Task[None] | None = None


def _size(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(_size(k, seen) + _size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        total += sum(_size(v, seen) for v in value)
    elif isinstance(value, CacheEntry):
        total += _size(value.value, seen) + _size(value.workspace_id, seen)
    return total


class GlobalMemoryBoard:
    """One instance per process. Sync state uses an RLock; async waits never hold it.

    Values are copied on publication/delivery to isolate mutable agent outputs.
    Metadata tombstones retain generations so late producers cannot resurrect a
    released workspace. Multiple processes still need a shared IPC owner.
    """

    _instance: ClassVar[GlobalMemoryBoard | None] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> GlobalMemoryBoard:
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._lock = threading.RLock()
                instance._cache: OrderedDict[str, CacheEntry] = OrderedDict()
                instance._sizes: dict[str, int] = {}
                instance._cache_bytes = 0
                instance._inflight: dict[str, _Flight] = {}
                instance._generations: dict[str, int] = {}
                instance._facts: dict[str, dict[str, VerifiedFact]] = {}
                instance._gates: dict[str, _WorkspaceGate] = {}
                instance.max_cache_bytes = 64 * 1024 * 1024
                cls._instance = instance
            return cls._instance

    @classmethod
    def instance(cls) -> GlobalMemoryBoard:
        return cls()

    def _gate(self, workspace_id: str) -> _WorkspaceGate:
        with self._lock:
            return self._gates.setdefault(workspace_id, _WorkspaceGate())

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.generation != self._generations.get(entry.workspace_id, 0):
                self._drop(key)
                return None
            self._cache.move_to_end(key)
            return copy.deepcopy(entry)

    def _drop(self, key: str) -> None:
        self._cache.pop(key, None)
        self._cache_bytes -= self._sizes.pop(key, 0)

    def set(self, key: str, value: CacheEntry) -> None:
        if not isinstance(value, CacheEntry):
            raise TypeError("value must be a CacheEntry with workspace and generation")
        if not value.cacheable:
            return
        stored = copy.deepcopy(value)
        size = _size(key) + _size(stored)
        with self._lock:
            if stored.generation != self._generations.get(stored.workspace_id, 0):
                return
            self._drop(key)
            if size > self.max_cache_bytes:
                return
            while self._cache and self._cache_bytes + size > self.max_cache_bytes:
                self._drop(next(iter(self._cache)))
            self._cache[key] = stored
            self._sizes[key] = size
            self._cache_bytes += size

    def generation(self, workspace_id: str) -> int:
        with self._lock:
            return self._generations.get(workspace_id, 0)

    def invalidate(self, workspace_id: str) -> None:
        with self._lock:
            self._generations[workspace_id] = self._generations.get(workspace_id, 0) + 1
            for key in [
                k
                for k, entry in self._cache.items()
                if entry.workspace_id == workspace_id
            ]:
                self._drop(key)
            self._facts.pop(workspace_id, None)

    async def get_or_compute(
        self,
        key: str,
        producer: Callable[[], Awaitable[CacheEntry]],
        *,
        workspace_id: str | None = None,
    ) -> tuple[CacheEntry, bool]:
        with self._lock:
            cached = self.get(key)
            if cached is not None:
                return cached, True
            flight = self._inflight.get(key)
            shared = flight is not None
            if flight is None:
                loop = asyncio.get_running_loop()
                flight = _Flight(ThreadFuture(), loop, workspace_id)
                self._inflight[key] = flight
                flight.task = loop.create_task(self._produce(key, flight, producer))
                flight.task.add_done_callback(
                    lambda task: self._finish_flight(key, flight, task)
                )
        entry = await _wait_shared(flight.result)
        return copy.deepcopy(entry), shared

    def _finish_flight(
        self, key: str, flight: _Flight, task: asyncio.Task[None]
    ) -> None:
        # Cancellation before a coroutine starts never enters its try/finally.
        with self._lock:
            if not flight.result.done():
                error = (
                    asyncio.CancelledError() if task.cancelled() else task.exception()
                )
                flight.result.set_exception(
                    error or RuntimeError("Producer exited without a result")
                )
            if self._inflight.get(key) is flight:
                del self._inflight[key]

    async def _produce(
        self,
        key: str,
        flight: _Flight,
        producer: Callable[[], Awaitable[CacheEntry]],
    ) -> None:
        try:
            entry = await producer()
            with self._lock:
                if not flight.result.done():
                    self.set(key, entry)
                    flight.result.set_result(entry)
        except BaseException as exc:
            with self._lock:
                if not flight.result.done():
                    flight.result.set_exception(exc)
        finally:
            with self._lock:
                if self._inflight.get(key) is flight:
                    del self._inflight[key]

    def verified_facts(
        self,
        workspace_id: str,
        *,
        snapshot: WorkspaceSnapshot,
    ) -> tuple[VerifiedFact, ...]:
        with self._lock:
            return copy.deepcopy(
                tuple(
                    fact
                    for fact in self._facts.get(workspace_id, {}).values()
                    if fact.source_snapshot == snapshot
                    and snapshot.workspace_id == workspace_id
                )
            )

    def publish_facts(
        self,
        workspace_id: str,
        facts: tuple[VerifiedFact, ...],
        *,
        report: VerificationReport,
    ) -> None:
        if (
            not report.passed
            or not report.checks
            or not all(c.passed for c in report.checks)
        ):
            raise ValueError("Facts require a successful non-empty verification report")
        if report.snapshot.workspace_id != workspace_id:
            raise ValueError("Verification belongs to a different workspace")
        if any(
            fact.source_snapshot != report.snapshot
            or fact.verification_id != report.verification_id
            or not fact.key
            or not fact.evidence_refs
            for fact in facts
        ):
            raise ValueError("Fact evidence does not match its verification report")
        with self._lock:
            self._facts.setdefault(workspace_id, {}).update(
                {f.key: copy.deepcopy(f) for f in facts}
            )

    def release_workspace(self, workspace_id: str) -> None:
        with self._lock:
            self.invalidate(workspace_id)
            flights = [
                (key, flight)
                for key, flight in self._inflight.items()
                if flight.workspace_id == workspace_id
            ]
            for key, flight in flights:
                if not flight.result.done():
                    flight.result.set_exception(asyncio.CancelledError())
                del self._inflight[key]
        for _, flight in flights:
            if flight.task is not None and not flight.loop.is_closed():
                try:
                    flight.loop.call_soon_threadsafe(flight.task.cancel)
                except RuntimeError:
                    # A concurrent owner-loop shutdown can race the is_closed check.
                    pass


_READ_TOOLS = frozenset(
    {
        "view",
        "view_file",
        "read",
        "read_file",
        "read_range",
        "read_document",
        "cat",
        "search",
        "search_file",
        "search_dir",
        "find_file",
        "ls",
        "list_directory",
        "info",
        "tree",
    }
)
_DYNAMIC_READS = frozenset(
    {
        "get_time",
        "current_time",
        "get_notifications",
        "list_notifications",
        "read_artifact",
    }
)


class ToolGateway:
    """Explicit read policy; unknown tools and arbitrary bash/code may write.

    Adapters must pass effective cwd/path/view arguments and perform local ACI state
    updates and step accounting even on a hit. All real calls must be awaitable and
    honor cancellation. An optional authorizer is always run before cache access.
    """

    def __init__(
        self,
        board: GlobalMemoryBoard,
        *,
        authorize: Callable[[ToolRequest], Awaitable[None]] | None = None,
    ) -> None:
        self.board = board
        self.authorize = authorize

    def classify(self, request: ToolRequest) -> Literal["read", "write", "uncacheable"]:
        namespace, separator, name = request.tool_name.lower().rpartition("__")
        name = name if separator else request.tool_name.lower()
        if name in _DYNAMIC_READS:
            return "uncacheable"
        if name in _READ_TOOLS:
            # Dynamic app searches cannot be treated as versioned filesystem reads.
            return (
                "read"
                if not separator or namespace in {"files", "file", "fs", "workspace"}
                else "uncacheable"
            )
        return "write"

    def cache_key(self, request: ToolRequest, *, generation: int) -> str:
        payload = json.dumps(
            [
                request.run_id,
                request.workspace_id,
                request.tool_name,
                request.args,
                generation,
            ],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _successful(value: Any) -> bool:
        if not isinstance(value, dict):
            return True
        return (
            value.get("success") is not False
            and not value.get("error")
            and (value.get("exit_code", 0) == 0)
        )

    async def execute(
        self,
        request: ToolRequest,
        call: Callable[[], Awaitable[Any]],
    ) -> ToolResult:
        if self.authorize is not None:
            await self.authorize(request)
        mode = self.classify(request)
        gate = self.board._gate(request.workspace_id)
        if mode == "read" and request.version_provider is not None:
            # The provider pins immutable Git objects while this workspace is read-locked.
            # The producer reads the durable object store, so waiter cancellation cannot
            # expose a changing/deleted live workspace to an in-flight read.
            async with gate.hold(write=False):
                namespace, revision = await request.version_provider()
                if request.force_refresh:
                    self.board.invalidate(namespace)
                generation = self.board.generation(namespace)
                payload = [
                    request.run_id,
                    namespace,
                    revision,
                    request.tool_name,
                    request.args,
                    generation,
                ]
                key = hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()
                ).hexdigest()

                async def immutable_read():
                    value = await call()
                    return CacheEntry(
                        namespace, generation, value, self._successful(value)
                    )

                entry, hit = await self.board.get_or_compute(
                    key, immutable_read, workspace_id=namespace
                )
                return ToolResult(
                    entry.value, cache_hit=hit, annotation="[Cache Hit]" if hit else ""
                )

        if mode == "write":
            async with gate.hold(write=True):
                self.board.invalidate(request.workspace_id)
                try:
                    return ToolResult(await call())
                finally:
                    self.board.invalidate(request.workspace_id)
        if mode == "uncacheable":
            async with gate.hold(write=False):
                return ToolResult(await call())
        if request.force_refresh:
            async with gate.hold(write=True):
                self.board.invalidate(request.workspace_id)
                generation = self.board.generation(request.workspace_id)
                value = await call()
                self.board.set(
                    self.cache_key(request, generation=generation),
                    CacheEntry(
                        request.workspace_id, generation, value, self._successful(value)
                    ),
                )
                return ToolResult(value)
        while True:
            generation = self.board.generation(request.workspace_id)
            key = self.cache_key(request, generation=generation)

            async def produce() -> CacheEntry:
                # The producer owns this lease, so canceling a waiter cannot let a
                # writer overlap the real read that other waiters still depend on.
                async with gate.hold(write=False):
                    if self.board.generation(request.workspace_id) != generation:
                        return CacheEntry(request.workspace_id, generation, None, False)
                    value = await call()
                    return CacheEntry(
                        request.workspace_id, generation, value, self._successful(value)
                    )

            entry, hit = await self.board.get_or_compute(
                key, produce, workspace_id=request.workspace_id
            )
            if self.board.generation(request.workspace_id) != generation:
                continue
            return ToolResult(
                entry.value, cache_hit=hit, annotation="[Cache Hit]" if hit else ""
            )

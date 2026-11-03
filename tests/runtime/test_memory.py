import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

from porchestra.runtime.memory import (
    CacheEntry,
    GlobalMemoryBoard,
    ToolGateway,
    ToolRequest,
)
from porchestra.runtime.types import (
    CheckResult,
    VerificationReport,
    VerifiedFact,
    WorkspaceSnapshot,
)


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.board = GlobalMemoryBoard.instance()
        self.workspace = uuid4().hex
        self.gateway = ToolGateway(self.board)
        self.request = ToolRequest(
            "run", "node", self.workspace, "view", {"path": "a.py"}
        )

    def tearDown(self):
        self.board.release_workspace(self.workspace)

    async def test_singleton_and_thread_safe_generation(self):
        def invalidate(_):
            self.assertIs(GlobalMemoryBoard(), self.board)
            self.board.invalidate(self.workspace)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(invalidate, range(100)))
        self.assertEqual(self.board.generation(self.workspace), 100)

    async def test_many_readers_share_one_call_and_values_are_isolated(self):
        calls = 0

        async def read():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return {"lines": [1]}

        results = await asyncio.gather(
            *(self.gateway.execute(self.request, read) for _ in range(20))
        )
        self.assertEqual(calls, 1)
        self.assertEqual(sum(not r.cache_hit for r in results), 1)
        results[0].value["lines"].append(2)
        cached = await self.gateway.execute(self.request, read)
        self.assertEqual(cached.value, {"lines": [1]})
        self.assertEqual(cached.annotation, "[Cache Hit]")

    async def test_keys_ignore_argument_order_but_separate_workspaces_and_runs(self):
        left = replace(self.request, args={"path": "a", "start": 1})
        right = replace(left, node_id="other", args={"start": 1, "path": "a"})
        key = self.gateway.cache_key(left, generation=0)
        self.assertEqual(key, self.gateway.cache_key(right, generation=0))
        self.assertNotEqual(
            key, self.gateway.cache_key(replace(left, run_id="other"), generation=0)
        )
        self.assertNotEqual(
            key,
            self.gateway.cache_key(replace(left, workspace_id="other"), generation=0),
        )

    async def test_cancel_first_waiter_does_not_cancel_shared_producer(self):
        entered, finish = asyncio.Event(), asyncio.Event()
        calls = 0

        async def read():
            nonlocal calls
            calls += 1
            entered.set()
            await finish.wait()
            return "content"

        first = asyncio.create_task(self.gateway.execute(self.request, read))
        await entered.wait()
        second = asyncio.create_task(self.gateway.execute(self.request, read))
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        finish.set()
        self.assertEqual((await asyncio.wait_for(second, 1)).value, "content")
        self.assertEqual(calls, 1)

    async def test_failed_producer_settles_waiters_and_allows_retry(self):
        calls = 0

        async def fail():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            raise RuntimeError("read failed")

        outcomes = await asyncio.gather(
            *(self.gateway.execute(self.request, fail) for _ in range(5)),
            return_exceptions=True,
        )
        self.assertEqual(calls, 1)
        self.assertTrue(all(isinstance(o, RuntimeError) for o in outcomes))
        self.assertEqual(
            (
                await self.gateway.execute(
                    self.request, lambda: asyncio.sleep(0, result="ok")
                )
            ).value,
            "ok",
        )

    async def test_error_outputs_are_not_cached(self):
        calls = 0

        async def read():
            nonlocal calls
            calls += 1
            return {"success": False, "error": "missing"}

        await self.gateway.execute(self.request, read)
        await self.gateway.execute(self.request, read)
        self.assertEqual(calls, 2)

    async def test_write_waits_for_reads_and_invalidates_across_gateways(self):
        entered, finish = asyncio.Event(), asyncio.Event()
        state = {"value": "old"}

        async def read():
            entered.set()
            await finish.wait()
            return state["value"]

        reader = asyncio.create_task(self.gateway.execute(self.request, read))
        await entered.wait()

        async def write():
            state["value"] = "new"

        writer = asyncio.create_task(
            ToolGateway(self.board).execute(
                replace(self.request, tool_name="edit"), write
            )
        )
        await asyncio.sleep(0)
        self.assertEqual(state["value"], "old")
        finish.set()
        await asyncio.wait_for(asyncio.gather(reader, writer), 1)
        self.assertEqual((await self.gateway.execute(self.request, read)).value, "new")

    async def test_external_invalidation_discards_inflight_old_result(self):
        entered, finish = asyncio.Event(), asyncio.Event()
        calls = 0

        async def read():
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await finish.wait()
                return "old"
            return "new"

        pending = asyncio.create_task(self.gateway.execute(self.request, read))
        await entered.wait()
        self.board.invalidate(self.workspace)
        finish.set()
        self.assertEqual((await pending).value, "new")
        self.assertEqual(calls, 2)

    async def test_failed_write_invalidates_and_cancelled_writer_does_not_block(self):
        await self.gateway.execute(self.request, lambda: asyncio.sleep(0, result="old"))

        async def fail():
            raise RuntimeError("partial write")

        with self.assertRaises(RuntimeError):
            await self.gateway.execute(replace(self.request, tool_name="bash"), fail)
        self.assertFalse(
            (
                await self.gateway.execute(
                    self.request, lambda: asyncio.sleep(0, result="new")
                )
            ).cache_hit
        )
        gate = self.board._gate(self.workspace)
        async with gate.hold(write=False):
            pending = asyncio.create_task(
                self.gateway.execute(replace(self.request, tool_name="edit"), fail)
            )
            await asyncio.sleep(0)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        result = await asyncio.wait_for(
            self.gateway.execute(self.request, lambda: asyncio.sleep(0, result="ok")), 1
        )
        self.assertTrue(result.cache_hit)

    async def test_cross_thread_event_loops_share_one_producer(self):
        calls = 0
        lock = threading.Lock()
        start = threading.Barrier(2)

        async def read():
            nonlocal calls
            with lock:
                calls += 1
            await asyncio.sleep(0.05)
            return "shared"

        def worker():
            start.wait(timeout=2)
            return asyncio.run(
                ToolGateway(self.board).execute(self.request, read)
            ).value

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker) for _ in range(2)]
            self.assertEqual(
                [f.result(timeout=3) for f in futures], ["shared", "shared"]
            )
        self.assertEqual(calls, 1)

    async def test_force_refresh_dynamic_reads_and_authorization(self):
        calls = 0

        async def read():
            nonlocal calls
            calls += 1
            return calls

        await self.gateway.execute(self.request, read)
        self.assertEqual(
            (
                await self.gateway.execute(
                    replace(self.request, force_refresh=True), read
                )
            ).value,
            2,
        )
        self.assertEqual((await self.gateway.execute(self.request, read)).value, 2)
        dynamic = replace(self.request, tool_name="Calendar__search")
        await self.gateway.execute(dynamic, read)
        await self.gateway.execute(dynamic, read)
        self.assertEqual(calls, 4)

        async def deny(request):
            raise PermissionError(request.node_id)

        with self.assertRaises(PermissionError):
            await ToolGateway(self.board, authorize=deny).execute(self.request, read)

    async def test_stale_publication_and_fact_validation(self):
        generation = self.board.generation(self.workspace)
        self.board.invalidate(self.workspace)
        self.board.set("stale", CacheEntry(self.workspace, generation, "old"))
        self.assertIsNone(self.board.get("stale"))
        snap = WorkspaceSnapshot(self.workspace, "v1", "artifact")
        report = VerificationReport(
            "verify", "node", "contract", 1, snap, True, (CheckResult("c", 0, True),)
        )
        fact = VerifiedFact("key", "value", ("evidence",), snap, "verify")
        self.board.publish_facts(self.workspace, (fact,), report=report)
        self.assertEqual(
            self.board.verified_facts(self.workspace, snapshot=snap), (fact,)
        )
        with self.assertRaises(ValueError):
            self.board.publish_facts(
                self.workspace, (replace(fact, verification_id="wrong"),), report=report
            )
        self.board.invalidate(self.workspace)
        self.assertEqual(self.board.verified_facts(self.workspace, snapshot=snap), ())

    async def test_cache_capacity_evicts_and_large_entries_bypass_storage(self):
        original_limit = self.board.max_cache_bytes
        generation = self.board.generation(self.workspace)
        try:
            self.board.max_cache_bytes = 1200
            for i in range(5):
                self.board.set(
                    f"{self.workspace}-{i}",
                    CacheEntry(self.workspace, generation, "x" * 200),
                )
            self.assertIsNone(self.board.get(f"{self.workspace}-0"))
            self.assertIsNotNone(self.board.get(f"{self.workspace}-4"))
            self.board.set(
                f"{self.workspace}-large",
                CacheEntry(self.workspace, generation, "x" * 5000),
            )
            self.assertIsNone(self.board.get(f"{self.workspace}-large"))
        finally:
            self.board.max_cache_bytes = original_limit

    async def test_releasing_workspace_cancels_producer_and_settles_waiters(self):
        entered = asyncio.Event()

        async def read():
            entered.set()
            await asyncio.Event().wait()

        pending = asyncio.create_task(self.gateway.execute(self.request, read))
        await entered.wait()
        self.board.release_workspace(self.workspace)
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 1)

    async def test_cancellation_before_producer_starts_does_not_leave_future_pending(
        self,
    ):
        async def produce():
            return CacheEntry(self.workspace, 0, "value")

        pending = asyncio.create_task(
            self.board.get_or_compute(
                self.workspace, produce, workspace_id=self.workspace
            )
        )
        await asyncio.sleep(0)
        self.board.release_workspace(self.workspace)
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 1)


if __name__ == "__main__":
    unittest.main()

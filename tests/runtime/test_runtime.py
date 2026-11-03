import asyncio
import copy
import hashlib
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from porchestra.runtime import (
    AcceptanceCheck,
    BudgetLedger,
    ContractGenerator,
    ContractVerifier,
    EventSink,
    GlobalMemoryBoard,
    HealthMonitor,
    HealthPolicy,
    NodeRegistry,
    NodeRuntime,
    RuntimeLimits,
    StateMerger,
    TaskContract,
    ToolGateway,
)
from porchestra.runtime.runtime import SpawnBudgetPolicy
from porchestra.runtime.types import (
    CheckResult,
    HealthDecision,
    MergeReport,
    NodeRecord,
    NodeState,
    StepOutcome,
    WorkspaceDiff,
    WorkspaceSnapshot,
)


def helper_plan(node, count=2, budget=None):
    return tuple(
        replace(
            node.contract,
            objective=("Investigate" if i == 0 else "Implement") + f" component {i}",
            contract_id=uuid4().hex,
            parent_contract_id=node.contract.contract_id,
            scope=(f"component_{i}",),
            acceptance_checks=(
                AcceptanceCheck(f"component_{i}", f"pytest component_{i}"),
            ),
            budget_limit=budget or node.contract.budget_limit // 4,
        )
        for i in range(count)
    )


class MockSpawnBudgetPolicy(SpawnBudgetPolicy):
    # These drivers have no model/prompt. Real prompt-budget guards are tested separately.
    def minimum_child(self, contract, shared_context=""):
        return 1


class MockWorkspaceBackend:
    def __init__(self):
        self.files = {"root": {}}
        self.snapshots = {}
        self.released = []

    async def snapshot(self, workspace_id):
        data = self.files[workspace_id]
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        ref = f"{workspace_id}:{digest}"
        self.snapshots[ref] = copy.deepcopy(data)
        return WorkspaceSnapshot(workspace_id, digest, ref)

    async def diff_fingerprint(self, workspace_id):
        return (await self.snapshot(workspace_id)).revision

    async def fork(self, base, *, node_id):
        self.files[node_id] = copy.deepcopy(self.snapshots[base.artifact_ref])
        return node_id

    async def export_diff(self, base, workspace_id):
        return WorkspaceDiff(
            base, await self.snapshot(workspace_id), f"patch:{workspace_id}"
        )

    async def merge_three_way(self, parent, child):
        base = self.snapshots[child.diff.base.artifact_ref]
        theirs = self.snapshots[child.diff.tip.artifact_ref]
        ours = self.files[parent.workspace_id]
        changed = {
            key for key in set(base) | set(theirs) if base.get(key) != theirs.get(key)
        }
        conflicts = tuple(
            key
            for key in changed
            if ours.get(key) not in (base.get(key), theirs.get(key))
        )
        if conflicts:
            return MergeReport(
                uuid4().hex,
                child.node_id,
                parent.node_id,
                False,
                conflicts=conflicts,
                artifact_refs=child.artifact_refs,
            )
        for key in changed:
            if key in theirs:
                ours[key] = theirs[key]
            else:
                ours.pop(key, None)
        return MergeReport(
            uuid4().hex,
            child.node_id,
            parent.node_id,
            True,
            await self.snapshot(parent.workspace_id),
        )

    async def release(self, workspace_id):
        self.released.append(workspace_id)


class MockCheckRunner:
    def __init__(self, workspaces):
        self.workspaces = workspaces

    async def run(self, check, snapshot, *, deadline_monotonic):
        await asyncio.sleep(0)
        passed = (
            self.workspaces.snapshots[snapshot.artifact_ref].get("fix.py") == "fixed"
        )
        return CheckResult(
            check.check_id,
            0 if passed else 1,
            passed,
            stdout="1 passed" if passed else "1 failed",
            stderr="" if passed else "fix.py is not fixed",
        )


class MockAgentDriver:
    def __init__(self, backend, *, hang_children=False):
        self.backend = backend
        self.hang_children = hang_children
        self.steps = 0
        self.messages = []
        self.closed = False
        self.cancelled = False

    async def step(self, node):
        self.steps += 1
        await asyncio.sleep(0)
        if node.depth == 0 and not self.messages:
            if self.steps < 5:
                return StepOutcome("continue", tokens_used=10)
            return StepOutcome(
                "spawn", child_contracts=helper_plan(node), tokens_used=10
            )
        if node.depth > 0:
            if self.hang_children:
                await asyncio.Event().wait()
            self.backend.files[node.workspace_id]["fix.py"] = "fixed"
        return StepOutcome("finish", tokens_used=3)

    async def inject_context(self, node_id, message):
        self.messages.append(message)

    async def checkpoint(self, node_id):
        return f"checkpoint:{node_id}:{self.steps}"

    async def cancel(self, node_id, *, reason):
        self.cancelled = True

    async def close(self, node_id):
        self.closed = True


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = MockWorkspaceBackend()
        self.board = GlobalMemoryBoard.instance()
        self.drivers = {}

    def build(self, *, factory=None, limits=None, console=False, checks=None):
        self.log = Path(self.temp.name) / "events.jsonl"
        events = EventSink(str(self.log), console=console)
        self.root = TaskContract(
            "Fix the regression",
            checks or (AcceptanceCheck("regression", "pytest tests/test_fix.py"),),
            1000,
            uuid4().hex,
            deadline_seconds=3,
        )

        def create(node, gateway):
            driver = factory(node) if factory else MockAgentDriver(self.backend)
            self.drivers[node.node_id] = driver
            return driver

        self.runtime = NodeRuntime(
            run_id=uuid4().hex,
            registry=NodeRegistry(),
            verifier=ContractVerifier(MockCheckRunner(self.backend), self.backend),
            contracts=ContractGenerator(),
            gateway=ToolGateway(self.board),
            health=HealthMonitor(),
            merger=StateMerger(self.backend, self.board, events),
            events=events,
            workspaces=self.backend,
            budget=BudgetLedger(1000),
            spawn_budget_policy=MockSpawnBudgetPolicy(parent_min_tokens=1),
            driver_factory=create,
            limits=limits
            or RuntimeLimits(
                max_active=1, step_timeout_seconds=1, cleanup_timeout_seconds=1
            ),
        )
        return self.runtime

    def records(self):
        return [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
        ]

    async def test_stagnation_split_verify_merge_wake_end_to_end_single_slot(self):
        runtime = self.build(console=True)
        result = await asyncio.wait_for(runtime.run(self.root, workspace_id="root"), 3)
        self.assertEqual(result.outcome, "verified")
        self.assertEqual(self.backend.files["root"], {"fix.py": "fixed"})
        self.assertEqual(len(runtime.registry.results), 3)
        self.assertFalse(runtime.registry.nodes)
        self.assertTrue(all(driver.closed for driver in self.drivers.values()))
        records = self.records()
        names = [record["event"] for record in records]
        self.assertEqual(names.count("SPAWN"), 3)
        self.assertEqual(names.count("SLEEP"), 1)
        self.assertEqual(names.count("VERIFY_PASS"), 3)
        self.assertEqual(names.count("MERGE_COMMIT"), 2)
        self.assertEqual(names.count("DESTROY"), 3)
        self.assertEqual(names.count("WAKE"), 1)
        self.assertLess(names.index("SLEEP"), names.index("MERGE_COMMIT"))
        self.assertLess(
            max(i for i, name in enumerate(names) if name == "MERGE_COMMIT"),
            names.index("WAKE"),
        )
        self.assertEqual(
            [r["sequence"] for r in records], list(range(1, len(records) + 1))
        )
        self.assertEqual(runtime.budget.used_by_node[self.root.contract_id], 59)
        self.assertEqual(runtime.budget.reserved_by_node[self.root.contract_id], 0)
        self.assertEqual(self.drivers[self.root.contract_id].steps, 6)

    async def test_failed_verification_repairs_same_session(self):
        class RepairDriver(MockAgentDriver):
            async def step(self, node):
                self.steps += 1
                if self.messages:
                    self.backend.files[node.workspace_id]["fix.py"] = "fixed"
                return StepOutcome("finish", tokens_used=2)

        runtime = self.build(factory=lambda node: RepairDriver(self.backend))
        result = await asyncio.wait_for(runtime.run(self.root, workspace_id="root"), 2)
        self.assertEqual(result.outcome, "verified")
        driver = self.drivers[self.root.contract_id]
        self.assertEqual(driver.steps, 2)
        self.assertIn("FINISH REJECTED", driver.messages[0])
        self.assertIn("VERIFY_REJECT", [r["event"] for r in self.records()])

    async def test_proactive_spawn_enforces_four_execution_slots(self):
        active, peak = 0, 0

        class ParallelDriver(MockAgentDriver):
            async def step(self, node):
                nonlocal active, peak
                if node.depth == 0:
                    if self.messages:
                        return StepOutcome("finish", tokens_used=1)
                    children = helper_plan(node, count=6, budget=100)
                    return StepOutcome("spawn", child_contracts=children, tokens_used=1)
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.03)
                    self.backend.files[node.workspace_id]["fix.py"] = "fixed"
                    return StepOutcome("finish", tokens_used=1)
                finally:
                    active -= 1

        runtime = self.build(
            factory=lambda node: ParallelDriver(self.backend),
            limits=RuntimeLimits(max_active=4),
        )
        result = await asyncio.wait_for(runtime.run(self.root, workspace_id="root"), 3)
        self.assertEqual(result.outcome, "verified")
        self.assertEqual(peak, 4)
        self.assertEqual(len(runtime.registry.results), 7)

    async def test_recursive_delegation_with_one_execution_slot(self):
        class RecursiveDriver(MockAgentDriver):
            async def step(self, node):
                if node.depth < 2 and not any(
                    message.startswith("[") for message in self.messages
                ):
                    children = helper_plan(node)
                    return StepOutcome("spawn", child_contracts=children, tokens_used=1)
                if node.depth == 2:
                    self.backend.files[node.workspace_id]["fix.py"] = "fixed"
                return StepOutcome("finish", tokens_used=1)

        runtime = self.build(factory=lambda node: RecursiveDriver(self.backend))
        result = await asyncio.wait_for(runtime.run(self.root, workspace_id="root"), 3)
        self.assertEqual(result.outcome, "verified")
        self.assertEqual(len(runtime.registry.results), 7)
        self.assertEqual(sum(r["event"] == "WAKE" for r in self.records()), 3)
        self.assertEqual(runtime.budget.reserved_by_node[self.root.contract_id], 0)

    async def test_token_threshold_advises_convergence_without_duplicating_tasks(self):
        class TokenDriver(MockAgentDriver):
            async def step(self, node):
                if not self.messages:
                    self.backend.files[node.workspace_id]["progress"] = "new diff"
                    return StepOutcome("continue", tokens_used=800)
                self.backend.files[node.workspace_id]["fix.py"] = "fixed"
                return StepOutcome("finish", tokens_used=1)

        runtime = self.build(factory=lambda node: TokenDriver(self.backend))
        result = await runtime.run(self.root, workspace_id="root")
        self.assertEqual(result.outcome, "verified")
        self.assertEqual(len(runtime.registry.results), 1)
        self.assertEqual(runtime.budget.used_by_node[self.root.contract_id], 801)
        self.assertTrue(
            any(
                "token_budget:800/1000" in message
                for message in self.drivers[self.root.contract_id].messages
            )
        )

    async def test_depth_cap_rejects_helpers_and_requester_continues(self):
        class Requester(MockAgentDriver):
            async def step(self, node):
                if not self.messages:
                    return StepOutcome("spawn", child_contracts=helper_plan(node))
                self.backend.files[node.workspace_id]["fix.py"] = "fixed"
                return StepOutcome("finish", tokens_used=1)

        runtime = self.build(
            factory=lambda node: Requester(self.backend),
            limits=RuntimeLimits(max_depth=0),
        )
        result = await runtime.run(self.root, workspace_id="root")
        self.assertEqual(result.outcome, "verified")
        self.assertEqual(len(runtime.registry.results), 1)
        self.assertIn("SPLIT_REJECTED", [r["event"] for r in self.records()])

    async def test_child_timeout_does_not_deadlock_waiting_parent(self):
        runtime = self.build(
            factory=lambda node: MockAgentDriver(self.backend, hang_children=True),
            limits=RuntimeLimits(
                step_timeout_seconds=0.02, cleanup_timeout_seconds=0.5
            ),
        )
        self.root = replace(self.root, deadline_seconds=0.3)
        result = await asyncio.wait_for(runtime.run(self.root, workspace_id="root"), 2)
        self.assertNotEqual(result.outcome, "verified")
        self.assertFalse(runtime.registry.nodes)
        names = [r["event"] for r in self.records()]
        self.assertIn("TIMEOUT", names)
        self.assertIn("WAKE", names)
        self.assertNotIn("MERGE_COMMIT", names)

    async def test_failed_sibling_prevents_batch_auto_merge(self):
        class FailedSiblingDriver(MockAgentDriver):
            async def step(self, node):
                if node.depth > 0 and node.contract.objective.startswith("Investigate"):
                    return StepOutcome(
                        "stop", reason="dependency unavailable", tokens_used=1
                    )
                if node.depth == 0 and self.messages:
                    return StepOutcome(
                        "stop", reason="child batch failed", tokens_used=1
                    )
                return await super().step(node)

        runtime = self.build(factory=lambda node: FailedSiblingDriver(self.backend))
        result = await asyncio.wait_for(runtime.run(self.root, workspace_id="root"), 3)
        self.assertEqual(result.outcome, "failed")
        self.assertFalse(runtime.registry.nodes)
        self.assertNotIn("fix.py", self.backend.files["root"])
        self.assertTrue(
            any(r.outcome == "verified" for r in runtime.registry.results.values())
        )
        names = [r["event"] for r in self.records()]
        self.assertNotIn("MERGE_COMMIT", names)
        self.assertEqual(names.count("WAKE"), 1)

    async def test_external_wait_wakes_on_notification(self):
        class WaitDriver(MockAgentDriver):
            async def step(self, node):
                if not self.messages:
                    return StepOutcome("pause", tokens_used=1)
                self.backend.files[node.workspace_id]["fix.py"] = "fixed"
                return StepOutcome("finish", tokens_used=1)

        runtime = self.build(factory=lambda node: WaitDriver(self.backend))
        task = asyncio.create_task(runtime.run(self.root, workspace_id="root"))
        for _ in range(100):
            node = runtime.registry.nodes.get(self.root.contract_id)
            if (
                node
                and node.state is NodeState.WAITING
                and self.root.contract_id not in runtime._jobs
            ):
                break
            await asyncio.sleep(0.002)
        await runtime.notify(self.root.contract_id, message="new evidence")
        self.assertEqual((await asyncio.wait_for(task, 2)).outcome, "verified")
        self.assertEqual([r["event"] for r in self.records()].count("WAKE"), 1)

    async def test_runtime_cancellation_reaps_all_descendants(self):
        runtime = self.build(
            factory=lambda node: MockAgentDriver(self.backend, hang_children=True)
        )
        task = asyncio.create_task(runtime.run(self.root, workspace_id="root"))
        for _ in range(200):
            if len(runtime.registry.nodes) == 3:
                break
            await asyncio.sleep(0.002)
        self.assertEqual(len(runtime.registry.nodes), 3)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertFalse(runtime.registry.nodes)
        self.assertTrue(all(driver.closed for driver in self.drivers.values()))

    async def test_conflict_preserved_and_parent_wakes(self):
        class ConflictBackend(MockWorkspaceBackend):
            async def merge_three_way(self, parent, child):
                return MergeReport(
                    uuid4().hex,
                    child.node_id,
                    parent.node_id,
                    False,
                    conflicts=("fix.py",),
                    artifact_refs=child.artifact_refs,
                )

        self.backend = ConflictBackend()

        class ResolveDriver(MockAgentDriver):
            async def step(self, node):
                if node.depth == 0 and self.messages:
                    self.backend.files[node.workspace_id]["fix.py"] = "fixed"
                    return StepOutcome("finish", tokens_used=1)
                return await super().step(node)

        runtime = self.build(factory=lambda node: ResolveDriver(self.backend))
        result = await asyncio.wait_for(runtime.run(self.root, workspace_id="root"), 3)
        self.assertEqual(result.outcome, "verified")
        records = self.records()
        self.assertEqual(sum(r["event"] == "MERGE_CONFLICT" for r in records), 2)
        self.assertEqual(sum(r["event"] == "WAKE" for r in records), 1)
        self.assertIn("fix.py", self.drivers[self.root.contract_id].messages[0])

    async def test_parent_deadline_during_batch_merge_reaps_whole_tree(self):
        entered = asyncio.Event()

        class SlowMergeBackend(MockWorkspaceBackend):
            async def merge_three_way(self, parent, child):
                entered.set()
                await asyncio.sleep(5)
                return await super().merge_three_way(parent, child)

        self.backend = SlowMergeBackend()
        runtime = self.build()
        self.root = replace(self.root, deadline_seconds=0.5)
        result = await asyncio.wait_for(runtime.run(self.root, workspace_id="root"), 2)
        self.assertTrue(entered.is_set())
        self.assertEqual(result.outcome, "timed_out")
        self.assertFalse(runtime.registry.nodes)
        self.assertEqual(len(runtime.registry.results), 3)
        self.assertTrue(all(driver.closed for driver in self.drivers.values()))


class HealthAndBudgetTests(unittest.IsolatedAsyncioTestCase):
    def node(self):
        contract = TaskContract("task", (AcceptanceCheck("c", "pytest"),), 100, "root")
        return NodeRecord(
            "root",
            "run",
            None,
            0,
            contract,
            "workspace",
            WorkspaceSnapshot("workspace", "v1", "ref"),
            time.monotonic() + 10,
        )

    async def test_five_unchanged_steps_token_boundary_and_latch(self):
        node = self.node()
        monitor = HealthMonitor()
        monitor.reset_after_wake(node, diff_fingerprint="baseline")
        for _ in range(4):
            self.assertFalse(
                monitor.observe(node, diff_fingerprint="baseline").split_required
            )
        self.assertTrue(
            monitor.observe(node, diff_fingerprint="baseline").split_required
        )
        monitor.acknowledge_split(node)
        node.tokens_used = 80
        monitor.reset_after_wake(node, diff_fingerprint="changed")
        self.assertFalse(
            monitor.observe(node, diff_fingerprint="changed").split_required
        )
        node.contract = replace(node.contract, version=2)
        self.assertTrue(monitor.observe(node, diff_fingerprint="new").split_required)
        node.tokens_used = 79
        self.assertFalse(monitor.observe(node, diff_fingerprint="newer").split_required)
        with self.assertRaises(ValueError):
            HealthPolicy(stagnant_steps=0)

    async def test_nested_budget_reservations_conserve_tokens(self):
        node = self.node()
        ledger = BudgetLedger(100)
        ledger.register_root("root", node.contract)
        a = replace(
            node.contract, contract_id="a", parent_contract_id="root", budget_limit=40
        )
        b = replace(a, contract_id="b", budget_limit=30)
        ledger.reserve_children("root", (a, b))
        grandchild = replace(
            a, contract_id="grandchild", parent_contract_id="a", budget_limit=20
        )
        ledger.reserve_children("a", (grandchild,))
        ledger.charge("grandchild", tokens=7)
        self.assertEqual(ledger.used_by_node["root"], 7)
        self.assertEqual(ledger.remaining("root"), 30)
        ledger.settle("grandchild")
        ledger.settle("a")
        ledger.settle("b")
        ledger.settle("b")
        self.assertEqual(ledger.remaining("root"), 93)
        self.assertEqual(ledger.reserved_by_node["root"], 0)

    async def test_default_split_declines_without_explicit_complementary_duties(self):
        duties = await ContractGenerator().split(
            self.node(), HealthDecision(True, "stagnation", 5, 0.5), available_budget=20
        )
        self.assertEqual(duties, ())


if __name__ == "__main__":
    unittest.main()

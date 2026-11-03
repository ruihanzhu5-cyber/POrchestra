import time
import unittest
from dataclasses import replace
from porchestra.runtime.contracts import ContractGenerator
from porchestra.runtime.runtime import BudgetLedger, SpawnBudgetPolicy

from porchestra.runtime.health import HealthMonitor
from porchestra.runtime.types import (
    AcceptanceCheck,
    NodeRecord,
    TaskContract,
    WorkspaceSnapshot,
)


def node():
    contract = TaskContract(
        "repair", (AcceptanceCheck("check", "pytest"),), 100000, "root"
    )
    return NodeRecord(
        "root",
        "run",
        None,
        0,
        contract,
        "repo",
        WorkspaceSnapshot("repo", "v1", "ref"),
        time.monotonic() + 60,
    )


class ProgressTests(unittest.TestCase):
    def test_same_read_range_stagnates(self):
        n = node()
        monitor = HealthMonitor()
        for _ in range(5):
            result = monitor.observe(
                n,
                diff_fingerprint="unchanged",
                progress={"tool": "read_range", "path": "a.py", "start": 1, "end": 20},
            )
        self.assertEqual(result.status, "circling")
        self.assertTrue(result.split_required)

    def test_distinct_files_commands_and_errors_are_progress(self):
        n = node()
        monitor = HealthMonitor()
        for i in range(8):
            for progress in (
                {"path": f"file{i}.py"},
                {"command": f"pytest test_{i}.py", "error": f"failure {i}"},
            ):
                result = monitor.observe(
                    n, diff_fingerprint="unchanged", progress=progress
                )
                self.assertFalse(result.split_required)
                self.assertEqual(result.status, "investigating")


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_has_no_duplicate_split(self):
        from porchestra.runtime.types import HealthDecision

        n = node()
        self.assertEqual(
            await ContractGenerator().split(
                n, HealthDecision(True, "repeat", 5, 0), available_budget=90000
            ),
            (),
        )


class SpawnBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_732_tokens_rejected_before_allocation(self):
        n = node()
        ledger = BudgetLedger(n.contract.budget_limit)
        ledger.register_root("root", n.contract)
        ledger.charge("root", tokens=n.contract.budget_limit - 732)
        child = replace(n.contract, contract_id="child", budget_limit=732)
        policy = SpawnBudgetPolicy()
        with self.assertRaisesRegex(ValueError, "startup budget"):
            ledger.validate_spawn(
                "root",
                (child,),
                minimums=[policy.minimum_child(child)],
                parent_reserve=policy.parent_reserve(n.contract, 732),
            )
        self.assertNotIn("child", ledger.limits_by_node)

    def test_parent_reserve_and_sufficient_children(self):
        n = node()
        ledger = BudgetLedger(n.contract.budget_limit)
        ledger.register_root("root", n.contract)
        child = replace(n.contract, contract_id="child", budget_limit=20000)
        policy = SpawnBudgetPolicy()
        reserve = policy.parent_reserve(n.contract, 100000)
        ledger.validate_spawn(
            "root",
            (child,),
            minimums=[policy.minimum_child(child)],
            parent_reserve=reserve,
        )
        too_large = replace(child, budget_limit=95000)
        with self.assertRaisesRegex(ValueError, "parent integration"):
            ledger.validate_spawn(
                "root",
                (too_large,),
                minimums=[policy.minimum_child(child)],
                parent_reserve=reserve,
            )
        a = replace(
            n.contract, contract_id="a", parent_contract_id="root", scope=("a.py",)
        )
        b = replace(a, contract_id="b", scope=("b.py",))
        with self.assertRaisesRegex(ValueError, "distinct"):
            ContractGenerator().validate_delegation(n.contract, (a, b))

    async def test_complementary_explicit_plan_is_accepted(self):
        from porchestra.runtime.types import HealthDecision

        n = node()
        a = replace(
            n.contract,
            objective="repair parser",
            acceptance_checks=(AcceptanceCheck("parser", "pytest test_parser.py"),),
            contract_id="a",
            parent_contract_id="root",
            scope=("parser.py",),
            budget_limit=10000,
        )
        b = replace(
            a,
            objective="repair renderer",
            acceptance_checks=(AcceptanceCheck("renderer", "pytest test_renderer.py"),),
            contract_id="b",
            scope=("renderer.py",),
        )
        self.assertEqual(
            await ContractGenerator().split(
                n,
                HealthDecision(True, "help", 0, 0),
                available_budget=90000,
                proposed=(a, b),
            ),
            (a, b),
        )

import os
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from porchestra.runtime.contracts import ContractGenerator
from porchestra.runtime.health import HealthMonitor
from porchestra.runtime.runtime import BudgetLedger, SpawnBudgetPolicy
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


class RecoverableSpawnTests(unittest.IsolatedAsyncioTestCase):
    async def test_depth_and_node_caps_keep_requester_alive(self):
        from test_runtime import MockAgentDriver, RuntimeTests

        from porchestra.runtime.runtime import RuntimeLimits
        from porchestra.runtime.types import StepOutcome

        for limits in (RuntimeLimits(max_depth=0), RuntimeLimits(max_nodes=1)):
            fixture = RuntimeTests()
            await fixture.asyncSetUp()
            self.addCleanup(fixture.doCleanups)

            class Requester(MockAgentDriver):
                async def step(self, record):
                    self.steps += 1
                    if self.steps == 1:
                        helper = replace(
                            record.contract,
                            objective="repair helper",
                            contract_id="helper",
                            parent_contract_id=record.contract.contract_id,
                            scope=("fix.py",),
                            budget_limit=100,
                        )
                        return StepOutcome("spawn", child_contracts=(helper,))
                    self.backend.files[record.workspace_id]["fix.py"] = "fixed"
                    return StepOutcome("finish", tokens_used=1)

            runtime = fixture.build(
                factory=lambda record: Requester(fixture.backend), limits=limits
            )
            result = await runtime.run(fixture.root, workspace_id="root")
            self.assertEqual(result.outcome, "verified")
            self.assertEqual(fixture.drivers[fixture.root.contract_id].steps, 2)
            events = fixture.records()
            self.assertEqual(sum(e["event"] == "SPLIT_REJECTED" for e in events), 1)
            self.assertEqual(sum(e["event"] == "DESTROY" for e in events), 1)
            self.assertTrue(
                any(
                    "recoverable" in m
                    for m in fixture.drivers[fixture.root.contract_id].messages
                )
            )


class ImportIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_src_precedes_installed_module_and_provenance_is_printed(self):
        from porchestra.runtime.benchmark_bindings import import_probe

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "installed").mkdir()
            (root / "src" / "target_v2.py").write_text("VALUE = 'workspace'")
            (root / "installed" / "target_v2.py").write_text("VALUE = 'installed'")
            env = dict(os.environ, PYTHONPATH=str(root / "installed"))
            run = subprocess.run(
                [sys.executable, "-c", import_probe(str(root), ("target_v2",))],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn(str(root / "src"), run.stdout)

    async def test_external_import_is_framework_failure_and_task_never_runs(self):
        from porchestra.runtime.benchmark_bindings import SnapshotCheckRunner

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            class Workspaces:
                async def fork(self, *args, **kwargs):
                    return str(root)

                async def release(self, *args):
                    pass

            marker = root / "task-ran"
            check = AcceptanceCheck(
                "imports", f'echo BAD > "{marker}"', import_modules=("json",)
            )
            result = await SnapshotCheckRunner(Workspaces()).run(
                check, None, deadline_monotonic=time.monotonic() + 10
            )
            self.assertFalse(result.passed)
            self.assertEqual(result.category, "framework_check")
            self.assertFalse(marker.exists())

    async def test_pytest_check_uses_supported_config_and_executes(self):
        from scripts.pilot_runtime_swe import PILOT_IDS, PUBLIC_CHECKS

        script = PUBLIC_CHECKS[PILOT_IDS[1]]
        self.assertIn("from _pytest.config import Config", script)
        self.assertNotIn("pytest.Config", script)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        # This assertion can fail for the target regression, but never the removed API.
        self.assertNotIn("has no attribute 'Config'", result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)


class HandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_spawn_handoff_has_reads_changes_checks_and_next_step(self):
        from test_runtime import MockAgentDriver, RuntimeTests

        from porchestra.runtime.runtime import BudgetLedger
        from porchestra.runtime.types import StepOutcome

        fixture = RuntimeTests()
        await fixture.asyncSetUp()
        self.addCleanup(fixture.doCleanups)

        class Requester(MockAgentDriver):
            async def step(self, record):
                self.steps += 1
                if record.depth == 0 and self.steps == 1:
                    helper = replace(
                        record.contract,
                        objective="repair helper",
                        contract_id="helper",
                        parent_contract_id=record.contract.contract_id,
                        scope=("fix.py",),
                        budget_limit=30000,
                    )
                    return StepOutcome(
                        "spawn",
                        child_contracts=(helper,),
                        progress={
                            "tool": "read_range",
                            "arguments": {"path": "fix.py", "start": 1, "end": 20},
                        },
                    )
                if record.depth > 0:
                    self.backend.files[record.workspace_id]["fix.py"] = "fixed"
                return StepOutcome("finish", tokens_used=1)

        runtime = fixture.build(factory=lambda record: Requester(fixture.backend))
        fixture.root = replace(fixture.root, budget_limit=100000)
        runtime.budget = BudgetLedger(100000)
        result = await runtime.run(fixture.root, workspace_id="root")
        self.assertEqual(result.outcome, "verified")
        msg = next(
            m
            for m in fixture.drivers["helper"].messages
            if m.startswith("PARENT_HANDOFF")
        )
        import json

        data = json.loads(msg.partition(" ")[2])
        self.assertEqual(
            set(data),
            {
                "read_locations",
                "changes",
                "checks_run",
                "unresolved_issues",
                "next_steps",
            },
        )
        self.assertEqual(data["read_locations"][0]["arguments"]["path"], "fix.py")
        self.assertEqual(data["changes"]["state"], "inherited_from_parent")


class StructuredCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_binding_shares_snapshot_reads_and_invalidates_after_edit(self):
        from types import SimpleNamespace

        from porchestra.runtime.benchmark_bindings import SWEToolBinding
        from porchestra.runtime.git_workspace import GitWorkspaceBackend
        from porchestra.runtime.memory import (
            GlobalMemoryBoard,
            ToolGateway,
            ToolRequest,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            for args in (
                ("init",),
                ("config", "user.name", "test"),
                ("config", "user.email", "test@local"),
            ):
                subprocess.run(
                    ["git", "-C", str(root), *args], check=True, capture_output=True
                )
            (root / "a.py").write_text("original\n")
            subprocess.run(
                ["git", "-C", str(root), "add", "."], check=True, capture_output=True
            )
            subprocess.run(
                ["git", "-C", str(root), "commit", "-m", "base"],
                check=True,
                capture_output=True,
            )
            backend = GitWorkspaceBackend(str(Path(temp) / "artifacts"))
            base = await backend.snapshot(str(root))
            fork = await backend.fork(base, node_id="child")
            n = replace(
                node(),
                workspace_id=str(root),
                contract=replace(node().contract, allowed_tools=("aci_command",)),
            )
            child = replace(n, node_id="child", workspace_id=fork)
            calls = 0
            original_git = backend._git

            async def counted(*args, **kwargs):
                nonlocal calls
                if len(args) > 1 and args[1] == "show":
                    calls += 1
                return await original_git(*args, **kwargs)

            backend._git = counted
            gateway = ToolGateway(GlobalMemoryBoard.instance())

            async def read(record):
                manager = SimpleNamespace(executor=SimpleNamespace(workspaces=backend))
                prepared = SWEToolBinding(manager).prepare(
                    record, {"action": "read_file", "params": {"path": "a.py"}}
                )
                return await gateway.execute(prepared.request, prepared.call)

            first = await read(n)
            second = await read(child)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(calls, 1)

            async def edit():
                (root / "a.py").write_text("changed\n")
                return {"exit_code": 0}

            await gateway.execute(
                ToolRequest(n.run_id, n.node_id, n.workspace_id, "write"), edit
            )
            changed = await read(n)
            self.assertFalse(changed.cache_hit)
            self.assertEqual(changed.value["output"], "changed\n")
            unchanged = await read(child)
            self.assertTrue(unchanged.cache_hit)
            self.assertEqual(unchanged.value["output"], "original\n")
            prepared = SWEToolBinding(
                SimpleNamespace(executor=SimpleNamespace(workspaces=backend))
            ).prepare(
                n,
                {
                    "action": "read_range",
                    "params": {"path": "a.py", "start": 1, "end": 1},
                },
            )
            ranged = await gateway.execute(prepared.request, prepared.call)
            self.assertEqual(ranged.value["output"], "1: changed")
            with self.assertRaises(ValueError):
                SWEToolBinding(SimpleNamespace()).prepare(
                    n, {"action": "read_file", "params": {"path": "../outside"}}
                )


class PartialMergeTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_verified_child_merges_despite_failed_sibling(self):
        from test_runtime import MockAgentDriver, RuntimeTests

        from porchestra.runtime.runtime import BudgetLedger
        from porchestra.runtime.types import StepOutcome

        fixture = RuntimeTests()
        await fixture.asyncSetUp()
        self.addCleanup(fixture.doCleanups)

        async def paths(diff):
            base = fixture.backend.snapshots[diff.base.artifact_ref]
            tip = fixture.backend.snapshots[diff.tip.artifact_ref]
            return tuple(k for k in set(base) | set(tip) if base.get(k) != tip.get(k))

        fixture.backend.changed_paths = paths

        class Requester(MockAgentDriver):
            async def step(self, record):
                self.steps += 1
                if record.depth == 0 and self.steps == 1:
                    a = replace(
                        record.contract,
                        objective="repair component A",
                        contract_id="a",
                        parent_contract_id=record.contract.contract_id,
                        scope=("fix.py",),
                        independent=True,
                        budget_limit=25000,
                    )
                    b = replace(
                        a,
                        objective="repair component B",
                        contract_id="b",
                        scope=("other.py",),
                        acceptance_checks=(AcceptanceCheck("b", "pytest other.py"),),
                    )
                    return StepOutcome("spawn", child_contracts=(a, b))
                if record.node_id == "b":
                    return StepOutcome("stop", reason="cannot repair sibling")
                if record.depth > 0:
                    self.backend.files[record.workspace_id]["fix.py"] = "fixed"
                return StepOutcome("finish", tokens_used=1)

        runtime = fixture.build(factory=lambda record: Requester(fixture.backend))
        fixture.root = replace(fixture.root, budget_limit=100000)
        runtime.budget = BudgetLedger(100000)
        result = await runtime.run(fixture.root, workspace_id="root")
        self.assertEqual(result.outcome, "verified")
        self.assertEqual(runtime.registry.results["b"].outcome, "failed")
        self.assertEqual(fixture.backend.files["root"]["fix.py"], "fixed")
        self.assertEqual(
            sum(e["event"] == "MERGE_COMMIT" for e in fixture.records()), 1
        )

    async def test_candidate_artifact_is_readable_without_becoming_trusted(self):
        from porchestra.runtime.lifecycle import ArtifactReader

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "candidate.patch"
            artifact.write_text("unverified change")
            reader = ArtifactReader((root,))
            reader.register(str(artifact), trusted=False)
            result = await reader.read_artifact(str(artifact))
            self.assertEqual(result["trust"], "candidate_unverified")
            self.assertEqual(result["output"], "unverified change")
            with self.assertRaises(PermissionError):
                await reader.read_artifact(str(root.parent / "escape"))


class CheckExitTests(unittest.IsolatedAsyncioTestCase):
    async def test_shell_pipeline_cannot_hide_failed_check(self):
        from porchestra.runtime.benchmark_bindings import SnapshotCheckRunner

        with tempfile.TemporaryDirectory() as temp:

            class Workspaces:
                async def fork(self, *args, **kwargs):
                    return temp

                async def release(self, *args):
                    pass

            runner = SnapshotCheckRunner(Workspaces())
            result = await runner.run(
                AcceptanceCheck("exit", "false | tail -n 1"),
                None,
                deadline_monotonic=time.monotonic() + 10,
            )
            self.assertNotEqual(result.exit_code, 0)

    async def test_framework_failure_stops_without_business_repair_feedback(self):
        from test_runtime import RuntimeTests, MockAgentDriver
        from porchestra.runtime.types import CheckResult, StepOutcome

        fixture = RuntimeTests()
        await fixture.asyncSetUp()
        self.addCleanup(fixture.doCleanups)

        class Requester(MockAgentDriver):
            async def step(self, record):
                self.steps += 1
                return StepOutcome("finish", tokens_used=1)

        class BadFramework:
            async def run(self, check, snapshot, **kwargs):
                return CheckResult(
                    check.check_id,
                    1,
                    False,
                    stderr="wrong module origin",
                    category="framework_check",
                )

        runtime = fixture.build(factory=lambda record: Requester(fixture.backend))
        runtime.verifier.runner = BadFramework()
        result = await runtime.run(fixture.root, workspace_id="root")
        self.assertEqual(result.outcome, "failed")
        self.assertIn("framework_check", result.reason)
        self.assertEqual(fixture.drivers[fixture.root.contract_id].steps, 1)
        self.assertEqual(fixture.drivers[fixture.root.contract_id].messages, [])


class NativePromptTests(unittest.TestCase):
    def test_registered_reads_match_prompt_and_native_parser(self):
        from porchestra.prompts.swebench_mini_subagent import (
            MINI_SWE_TASK_PROMPT,
            MINI_SWE_SYSTEM_PROMPT,
            build_mini_swe_bash_tool,
        )
        from porchestra.mini_swe_subagent import MiniSWESubAgent

        self.assertIn("read_file", MINI_SWE_SYSTEM_PROMPT)
        self.assertNotIn("bash` tool for repository inspection", MINI_SWE_TASK_PROMPT)
        tools = build_mini_swe_bash_tool("/workspace/root")
        self.assertTrue(
            {"bash", "read_file", "read_range", "search", "read_artifact"}.issubset(
                {t["function"]["name"] for t in tools}
            )
        )
        import json

        session = MiniSWESubAgent.model_construct()
        session._messages = []
        action = session._record_and_parse_payload(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "name": "read_range",
                        "arguments": json.dumps(
                            {"path": "a.py", "start": 1, "end": 10}
                        ),
                    }
                ],
            },
            current_step=1,
        )
        self.assertEqual(action["action"], "read_range")


class RuntimeSupportTests(unittest.IsolatedAsyncioTestCase):
    async def test_ignored_build_product_survives_fork_without_becoming_task_patch(
        self,
    ):
        from porchestra.runtime.git_workspace import GitWorkspaceBackend

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            for args in (
                ("init",),
                ("config", "user.name", "test"),
                ("config", "user.email", "test@local"),
            ):
                subprocess.run(
                    ["git", "-C", str(root), *args], check=True, capture_output=True
                )
            (root / ".gitignore").write_text("_version.py\n")
            package = root / "src" / "_pytest"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("from ._version import version\n")
            (package / "_version.py").write_text(
                "version = 'generated-build-version'\n"
            )
            subprocess.run(
                ["git", "-C", str(root), "add", "."], check=True, capture_output=True
            )
            subprocess.run(
                ["git", "-C", str(root), "commit", "-m", "base"],
                check=True,
                capture_output=True,
            )
            backend = GitWorkspaceBackend(
                str(Path(temp) / "artifacts"),
                support_paths=("src/_pytest/_version.py",),
            )
            base = await backend.snapshot(str(root))
            fork = await backend.fork(base, node_id="child")
            self.assertEqual(
                (Path(fork) / "src/_pytest/_version.py").read_text(),
                (package / "_version.py").read_text(),
            )
            diff = await backend.export_diff(base, fork)
            self.assertEqual(Path(diff.artifact_ref).read_bytes(), b"")
            self.assertNotIn(
                "_version.py",
                subprocess.check_output(
                    ["git", "-C", str(root), "ls-files"], text=True
                ),
            )


class AdditionalPilotSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_cannot_escape_fixed_selected_set(self):
        from types import SimpleNamespace
        from scripts.pilot_runtime_swe import worker

        with self.assertRaisesRegex(ValueError, "escaped"):
            await worker(
                SimpleNamespace(
                    worker="pytest-dev__pytest-11143",
                    pilot_set="baseline-failures-extra3",
                )
            )

    async def test_public_reproductions_compile_and_sets_do_not_overlap(self):
        from scripts.pilot_runtime_swe import PILOT_SETS, PUBLIC_CHECKS, IMPORT_MODULES

        self.assertEqual(len(PILOT_SETS["baseline-failures-extra3"]), 3)
        self.assertEqual(len(PILOT_SETS["baseline-success-control"]), 1)
        groups = list(PILOT_SETS.values())
        self.assertEqual(len(set(sum(groups, ()))), sum(map(len, groups)))
        for identity in groups[1] + groups[2]:
            compile(PUBLIC_CHECKS[identity], identity, "exec")
            self.assertTrue(IMPORT_MODULES[identity])

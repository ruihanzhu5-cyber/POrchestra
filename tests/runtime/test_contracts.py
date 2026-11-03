import asyncio
import json
import sys
import tempfile
import time
import unittest
from dataclasses import replace

from porchestra.runtime.contracts import ContractGenerator, ContractVerifier
from porchestra.runtime.types import (
    AcceptanceCheck,
    CheckResult,
    NodeRecord,
    TaskContract,
    WorkspaceSnapshot,
)


class Workspace:
    def __init__(self, path="workspace"):
        self.revision = "v1"
        self.path = path

    async def snapshot(self, workspace_id):
        return WorkspaceSnapshot(workspace_id, self.revision, self.path)


class Runner:
    def __init__(self, behavior=None):
        self.behavior = behavior
        self.calls = []

    async def run(self, check, snapshot, *, deadline_monotonic):
        self.calls.append((check, snapshot, deadline_monotonic))
        if self.behavior:
            return await self.behavior(check, snapshot)
        return CheckResult(check.check_id, 0, True, stdout="2 passed")


def node_for(checks):
    contract = TaskContract("fix issue", tuple(checks), 30, "contract")
    return NodeRecord(
        "node",
        "run",
        None,
        0,
        contract,
        "workspace",
        WorkspaceSnapshot("workspace", "v1", "workspace"),
        time.monotonic() + 10,
    )


class ContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_explicit_natural_language_and_deduplicate(self):
        contract = await ContractGenerator().create(
            "修复错误并运行 `pytest tests/test_bug.py`\nDoD: `pytest tests/test_bug.py`",
            budget_limit=20,
        )
        self.assertEqual(len(contract.acceptance_checks), 1)
        self.assertEqual(
            contract.acceptance_checks[0].command, "pytest tests/test_bug.py"
        )

    async def test_generate_structured_contract_and_inherit_constraints(self):
        parent = TaskContract(
            "root",
            (AcceptanceCheck("c", "pytest"),),
            30,
            "root",
            constraints=("no network",),
            allowed_tools=("view",),
        )
        data = {
            "objective": "repair",
            "acceptance_checks": [
                {"command": "pytest a.py", "expected_output_pattern": "passed"}
            ],
        }
        child = await ContractGenerator().create(
            json.dumps(data), budget_limit=10, parent=parent
        )
        self.assertEqual(child.parent_contract_id, "root")
        self.assertEqual(child.constraints, ("no network",))
        self.assertEqual(child.allowed_tools, ("view",))
        data["allowed_tools"] = ["bash"]
        with self.assertRaises(ValueError):
            await ContractGenerator().create(
                json.dumps(data), budget_limit=10, parent=parent
            )

    async def test_missing_dod_and_invalid_patterns_fail_closed(self):
        for text in ("fix the bug", '{"objective":"fix", "acceptance_checks":[]}'):
            with self.assertRaises(ValueError):
                await ContractGenerator().create(text, budget_limit=10)
        data = {
            "objective": "fix",
            "acceptance_checks": [
                {"command": "pytest", "expected_output_pattern": "["}
            ],
        }
        with self.assertRaises(ValueError):
            await ContractGenerator().create(json.dumps(data), budget_limit=10)

    async def test_caller_supplied_defaults_and_shell_fences(self):
        default = AcceptanceCheck("explicit-default", "pytest tests")
        contract = await ContractGenerator(default_checks=(default,)).create(
            "fix", budget_limit=10
        )
        self.assertEqual(contract.acceptance_checks, (default,))
        fenced = await ContractGenerator().create(
            "fix\n```bash\npytest tests\n```", budget_limit=10
        )
        self.assertEqual(fenced.acceptance_checks[0].command, "pytest tests")

    async def test_match_exit_and_pattern_independently_of_runner_pass_flag(self):
        node = node_for(
            (
                AcceptanceCheck("exit", "pytest"),
                AcceptanceCheck(
                    "pattern", "pytest", expected_output_pattern="3 passed"
                ),
            )
        )

        async def behavior(check, snapshot):
            return CheckResult(
                check.check_id,
                1 if check.check_id == "exit" else 0,
                True,
                "2 passed",
                "AssertionError: broken",
            )

        verifier = ContractVerifier(Runner(behavior), Workspace())
        report = await verifier.verify(node)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.checks), 2)
        self.assertTrue(all(not r.passed for r in report.checks))
        feedback = verifier.format_feedback(report)
        self.assertIn("2 passed", feedback)
        self.assertIn("AssertionError: broken", feedback)
        self.assertIn("FINISH REJECTED", feedback)

    async def test_expected_nonzero_exit_and_stderr_pattern(self):
        node = node_for(
            (
                AcceptanceCheck(
                    "negative",
                    "command",
                    expected_exit_code=2,
                    expected_output_pattern="expected error",
                ),
            )
        )

        async def behavior(check, snapshot):
            return CheckResult(check.check_id, 2, True, stderr="expected error")

        self.assertTrue(
            (await ContractVerifier(Runner(behavior), Workspace()).verify(node)).passed
        )

    async def test_timeout_exception_and_mismatched_result_are_failures(self):
        node = node_for(
            (
                AcceptanceCheck("timeout", "cmd", timeout_seconds=0.01),
                AcceptanceCheck("error", "cmd"),
                AcceptanceCheck("wrong", "cmd"),
            )
        )

        async def behavior(check, snapshot):
            if check.check_id == "timeout":
                await asyncio.sleep(1)
            if check.check_id == "error":
                raise OSError("runner unavailable")
            return CheckResult("other", 0, True)

        report = await ContractVerifier(Runner(behavior), Workspace()).verify(node)
        self.assertFalse(report.passed)
        self.assertTrue(report.checks[0].timed_out)
        self.assertIn("runner unavailable", report.checks[1].stderr)
        self.assertIn("mismatched", report.checks[2].stderr)

    async def test_workspace_or_contract_mutation_rejects_old_evidence(self):
        node = node_for((AcceptanceCheck("c", "pytest"),))
        workspace = Workspace()

        async def behavior(check, snapshot):
            workspace.revision = "v2"
            node.contract = replace(node.contract, version=2)
            return CheckResult(check.check_id, 0, True)

        report = await ContractVerifier(Runner(behavior), workspace).verify(node)
        self.assertFalse(report.passed)
        self.assertEqual(report.contract_version, 1)
        self.assertIn("Workspace changed", report.error_log)
        self.assertIn("Contract changed", report.error_log)

    async def test_expired_node_never_calls_runner(self):
        node = node_for((AcceptanceCheck("c", "pytest"),))
        node.deadline_monotonic = time.monotonic() - 1
        runner = Runner()
        report = await ContractVerifier(runner, Workspace()).verify(node)
        self.assertFalse(report.passed)
        self.assertEqual(runner.calls, [])

    async def test_feedback_bounded_and_keeps_error_tail(self):
        node = node_for((AcceptanceCheck("c", "pytest"),))

        async def behavior(check, snapshot):
            return CheckResult(
                check.check_id,
                1,
                False,
                "start\n" + "x" * 20000 + "\nERROR tail",
                "stderr reason",
            )

        verifier = ContractVerifier(Runner(behavior), Workspace())
        feedback = verifier.format_feedback(await verifier.verify(node))
        self.assertLessEqual(len(feedback), 16000)
        self.assertIn("ERROR tail", feedback)
        self.assertIn("stderr reason", feedback)

    async def test_real_subprocess_runner_matches_protocol_and_output(self):
        class SubprocessRunner:
            async def run(self, check, snapshot, *, deadline_monotonic):
                process = await asyncio.create_subprocess_shell(
                    check.command,
                    cwd=snapshot.artifact_ref,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), deadline_monotonic - time.monotonic()
                    )
                except BaseException:
                    if process.returncode is None:
                        process.kill()
                    await process.wait()
                    raise
                return CheckResult(
                    check.check_id,
                    process.returncode,
                    True,
                    stdout.decode(),
                    stderr.decode(),
                )

        with tempfile.TemporaryDirectory() as path:
            command = f'"{sys.executable}" -c "print(\'2 passed\')"'
            node = node_for(
                (AcceptanceCheck("real", command, expected_output_pattern="2 passed"),)
            )
            report = await ContractVerifier(SubprocessRunner(), Workspace(path)).verify(
                node
            )
            self.assertTrue(report.passed, report.error_log)
            self.assertIn("2 passed", report.checks[0].stdout)


if __name__ == "__main__":
    unittest.main()

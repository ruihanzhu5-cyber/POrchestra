import time
import unittest

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

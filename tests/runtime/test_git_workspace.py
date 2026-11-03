import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from porchestra.runtime import AcceptanceCheck, GitWorkspaceBackend, TaskContract
from porchestra.runtime.types import NodeRecord, NodeResult, WorkspaceSnapshot


class GitWorkspaceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        await self.git("init")
        await self.git("config", "user.email", "test@localhost")
        await self.git("config", "user.name", "Runtime Test")
        (self.root / "fix.py").write_text("original\n", encoding="utf-8")
        await self.git("add", ".")
        await self.git("commit", "-m", "base")
        self.backend = GitWorkspaceBackend(str(Path(self.temp.name) / "artifacts"))

    async def git(self, *args):
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(self.root),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        self.assertEqual(process.returncode, 0, stderr.decode(errors="replace"))
        return stdout

    async def parent(self, base: WorkspaceSnapshot):
        contract = TaskContract(
            "repair", (AcceptanceCheck("c", "pytest"),), 100, "parent"
        )
        return NodeRecord(
            "parent",
            "run",
            None,
            0,
            contract,
            str(self.root),
            base,
            time.monotonic() + 30,
        )

    async def test_snapshot_uncommitted_binary_delete_and_three_way_merge(self):
        (self.root / "local.txt").write_text("uncommitted\n", encoding="utf-8")
        base = await self.backend.snapshot(str(self.root))
        child = await self.backend.fork(base, node_id="child")
        self.assertEqual((Path(child) / "local.txt").read_text(), "uncommitted\n")
        (Path(child) / "fix.py").write_text("fixed\n", encoding="utf-8")
        (Path(child) / "binary.dat").write_bytes(b"\x00\xff\x01")
        (Path(child) / "local.txt").unlink()
        diff = await self.backend.export_diff(base, child)
        (self.root / "other.txt").write_text("parent work", encoding="utf-8")
        result = await self.backend.merge_three_way(
            await self.parent(base), NodeResult("child", "verified", "ready", diff=diff)
        )
        self.assertTrue(result.committed, result.conflicts)
        self.assertEqual((self.root / "fix.py").read_text(), "fixed\n")
        self.assertEqual((self.root / "binary.dat").read_bytes(), b"\x00\xff\x01")
        self.assertFalse((self.root / "local.txt").exists())
        self.assertEqual((self.root / "other.txt").read_text(), "parent work")
        await self.backend.release(child)
        self.assertFalse(Path(child).exists())
        # Child snapshots remain reconstructible after its .git directory is gone.
        restored = await self.backend.fork(diff.tip, node_id="restored")
        self.assertEqual((Path(restored) / "fix.py").read_text(), "fixed\n")
        await self.backend.release(restored)

    async def test_conflict_retains_staging_and_does_not_modify_parent(self):
        base = await self.backend.snapshot(str(self.root))
        child = await self.backend.fork(base, node_id="child")
        (Path(child) / "fix.py").write_text("child version\n", encoding="utf-8")
        diff = await self.backend.export_diff(base, child)
        (self.root / "fix.py").write_text("parent version\n", encoding="utf-8")
        result = await self.backend.merge_three_way(
            await self.parent(base), NodeResult("child", "verified", "ready", diff=diff)
        )
        self.assertFalse(result.committed)
        self.assertEqual((self.root / "fix.py").read_text(), "parent version\n")
        staging = Path(result.artifact_refs[0])
        self.assertTrue(staging.is_dir())
        self.assertIn("<<<<<<<", (staging / "fix.py").read_text())
        await self.backend.release(str(self.root))
        self.assertTrue(self.root.exists())
        await self.backend.release(child)


if __name__ == "__main__":
    unittest.main()

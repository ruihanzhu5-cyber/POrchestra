"""Git-backed snapshots and staged three-way patch application.

The artifact root must be outside task repositories. Parent workspace writes must
be exclusively owned by the runtime while a merge is published.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from uuid import uuid4

from .types import MergeReport, NodeRecord, NodeResult, WorkspaceDiff, WorkspaceSnapshot


class GitWorkspaceBackend:
    def __init__(
        self,
        artifact_root: str,
        *,
        timeout_seconds: float = 120.0,
        shared_clones: bool = True,
        support_paths: tuple[str, ...] = (),
    ) -> None:
        self.root = Path(artifact_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.shared_clones = shared_clones
        self.support_paths = support_paths
        self._store = self.root / "snapshots.git"
        self._store_lock = asyncio.Lock()
        self._store_ready = False
        self._owned: set[str] = set()
        self._preserved: set[str] = set()

    async def _ensure_store(self) -> None:
        async with self._store_lock:
            if not self._store_ready:
                if not (self._store / "HEAD").exists():
                    await self._git(str(self.root), "init", "--bare", str(self._store))
                self._store_ready = True

    async def _git(
        self,
        workspace: str,
        *args: str,
        env: dict[str, str] | None = None,
        allow_failure: bool = False,
    ) -> tuple[bytes, int]:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            workspace,
            *args,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), self.timeout_seconds
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode and not allow_failure:
            raise RuntimeError(stderr.decode(errors="replace").strip())
        return (stdout + (stderr if process.returncode else b"")), process.returncode

    async def snapshot(self, workspace_id: str) -> WorkspaceSnapshot:
        path = Path(workspace_id).resolve()
        if self.root == path or self.root.is_relative_to(path):
            raise ValueError("Artifact root must be outside the source workspace")
        top, _ = await self._git(str(path), "rev-parse", "--show-toplevel")
        if Path(top.decode().strip()).resolve() != path:
            raise ValueError("workspace_id must identify a repository root")
        indexes = self.root / "indexes"
        indexes.mkdir(exist_ok=True)
        index = indexes / uuid4().hex
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        try:
            await self._git(str(path), "read-tree", "HEAD", env=env)
            await self._git(str(path), "add", "-A", "--", ".", env=env)
            for relative in self.support_paths:
                support = path / relative
                support.resolve().relative_to(path)
                if support.is_file() and not support.is_symlink():
                    # Runtime build products belong to the initial snapshot, never
                    # a fabricated task patch. Forks must preserve their exact bytes.
                    await self._git(str(path), "add", "-f", "--", relative, env=env)
            tree, _ = await self._git(str(path), "write-tree", env=env)
        finally:
            index.unlink(missing_ok=True)
        revision = tree.decode().strip()
        ref = f"refs/porchestra/snapshots/{revision}"
        commit, status = await self._git(
            str(path), "rev-parse", "--verify", ref, allow_failure=True
        )
        if status:
            commit_env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "POrchestra",
                "GIT_AUTHOR_EMAIL": "runtime@localhost",
                "GIT_COMMITTER_NAME": "POrchestra",
                "GIT_COMMITTER_EMAIL": "runtime@localhost",
            }
            commit, _ = await self._git(
                str(path),
                "commit-tree",
                revision,
                "-p",
                "HEAD",
                "-m",
                "Runtime workspace snapshot",
                env=commit_env,
            )
            await self._git(str(path), "update-ref", ref, commit.decode().strip())
        await self._ensure_store()
        # Copy objects to durable storage before temporary child clones are removed.
        await self._git(
            str(self._store),
            "fetch",
            "--no-tags",
            "--no-write-fetch-head",
            "--",
            str(path),
            f"{ref}:refs/porchestra/commits/{commit.decode().strip()}",
        )
        snapshots = self.root / "snapshots"
        snapshots.mkdir(exist_ok=True)
        identity = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        manifest = snapshots / f"{identity}-{revision}.json"
        data = {
            "workspace": str(path),
            "tree": revision,
            "commit": commit.decode().strip(),
        }
        temporary = manifest.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(data), encoding="utf-8")
        temporary.replace(manifest)
        return WorkspaceSnapshot(workspace_id, revision, str(manifest))

    async def diff_fingerprint(self, workspace_id: str) -> str:
        return (await self.snapshot(workspace_id)).revision

    async def fork(self, base: WorkspaceSnapshot, *, node_id: str) -> str:
        data = json.loads(Path(base.artifact_ref).read_text(encoding="utf-8"))
        target = self.root / "workspaces" / uuid4().hex
        target.parent.mkdir(exist_ok=True)
        self._owned.add(str(target))
        try:
            await self._ensure_store()
            await self._git(
                str(self.root),
                "clone",
                "--shared" if self.shared_clones else "--no-hardlinks",
                "--no-checkout",
                "--",
                str(self._store),
                str(target),
            )
            await self._git(str(target), "checkout", "--detach", data["commit"])
            return str(target)
        except BaseException:
            await self.release(str(target))
            raise

    async def export_diff(
        self, base: WorkspaceSnapshot, workspace_id: str
    ) -> WorkspaceDiff:
        tip = await self.snapshot(workspace_id)
        patch, _ = await self._git(
            workspace_id, "diff", "--binary", base.revision, tip.revision, "--"
        )
        folder = self.root / "patches"
        folder.mkdir(exist_ok=True)
        path = folder / f"{uuid4().hex}.patch"
        path.write_bytes(patch)
        return WorkspaceDiff(base, tip, str(path))

    async def merge_three_way(
        self, parent: NodeRecord, child: NodeResult
    ) -> MergeReport:
        if child.diff is None:
            raise ValueError("Child has no exported patch")
        merge_id = uuid4().hex
        before = await self.snapshot(parent.workspace_id)
        patch = Path(child.diff.artifact_ref)
        if not patch.read_bytes():
            return MergeReport(
                merge_id,
                child.node_id,
                parent.node_id,
                True,
                before,
                artifact_refs=(str(patch),),
            )
        staging = await self.fork(before, node_id=merge_id)
        try:
            output, status = await self._git(
                staging,
                "apply",
                "--3way",
                "--index",
                "--binary",
                str(patch),
                allow_failure=True,
            )
            if status:
                self._preserved.add(staging)
                return MergeReport(
                    merge_id,
                    child.node_id,
                    parent.node_id,
                    False,
                    conflicts=(output.decode(errors="replace"),),
                    artifact_refs=(staging, str(patch), before.artifact_ref),
                )
            merged = await self.snapshot(staging)
            current = await self.snapshot(parent.workspace_id)
            if current.revision != before.revision:
                self._preserved.add(staging)
                return MergeReport(
                    merge_id,
                    child.node_id,
                    parent.node_id,
                    False,
                    conflicts=("Parent changed while merge was staged",),
                    artifact_refs=(staging, str(patch)),
                )
            # Publish the staged result as a patch without altering the caller's index.
            publish, _ = await self._git(
                staging, "diff", "--binary", before.revision, merged.revision, "--"
            )
            publication = self.root / "patches" / f"{merge_id}-publish.patch"
            publication.write_bytes(publish)
            if publish:
                await self._git(
                    parent.workspace_id,
                    "apply",
                    "--check",
                    "--binary",
                    str(publication),
                )
                await self._git(
                    parent.workspace_id, "apply", "--binary", str(publication)
                )
            after = await self.snapshot(parent.workspace_id)
            return MergeReport(
                merge_id,
                child.node_id,
                parent.node_id,
                True,
                after,
                artifact_refs=(str(patch), str(publication), after.artifact_ref),
            )
        except BaseException:
            self._preserved.add(staging)
            raise
        finally:
            if staging not in self._preserved:
                await self.release(staging)

    async def changed_paths(self, diff):
        output, _ = await self._git(
            str(self._store),
            "diff",
            "--name-only",
            "-z",
            diff.base.revision,
            diff.tip.revision,
            "--",
        )
        return tuple(
            path.decode(errors="replace") for path in output.split(b"\0") if path
        )

    async def describe_changes(self, base, workspace_id):
        current = await self.snapshot(workspace_id)
        output, _ = await self._git(
            workspace_id, "diff", "--stat", base.revision, current.revision, "--"
        )
        return output.decode(errors="replace")

    async def release(self, workspace_id: str) -> None:
        path = Path(workspace_id).resolve()
        key = str(path)
        if key not in self._owned or key in self._preserved:
            return
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError(
                "Refusing to release a workspace outside the artifact root"
            )
        if path.exists():

            def retry_readonly(function, target, error):
                if not isinstance(error, PermissionError):
                    raise error
                os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
                function(target)

            await asyncio.to_thread(shutil.rmtree, path, onexc=retry_readonly)
        self._owned.discard(key)

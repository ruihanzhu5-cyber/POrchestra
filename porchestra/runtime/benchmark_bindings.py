"""Workspace-aware SWE/GAIA tools and snapshot-isolated acceptance commands."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent_driver import PreparedTool
from .git_workspace import GitWorkspaceBackend
from .memory import ToolGateway, ToolRequest
from .types import CheckResult, NodeRecord


class MountedSWEWorkspace(GitWorkspaceBackend):
    """Host Git operations on an existing executor's persistent /workspace mount.

    Clones have self-contained objects; host-only alternates would be invalid in
    Docker. This assumes the existing executor and Docker daemon share the host.
    """

    def __init__(self, mount_root: str, executor: Any) -> None:
        self.mount_root = Path(mount_root).resolve()
        self.executor = executor
        super().__init__(str(self.mount_root / "runtime_git"), shared_clones=False)

    def container_path(self, host_path: str) -> str:
        relative = Path(host_path).resolve().relative_to(self.mount_root)
        return "/workspace/" + relative.as_posix()

    @classmethod
    async def prepare(
        cls, env: Any, mount_root: str
    ) -> tuple[MountedSWEWorkspace, str, Any]:
        executor = env._executor
        executor.set_runtime_workspace_directory(Path(mount_root))
        # Task initialization only: never called by a child-node factory.
        observation = await env.reset()
        backend = cls(mount_root, executor)
        workspace = str(backend.mount_root / f"root-{uuid4().hex}")
        container = backend.container_path(workspace)
        command = f"mkdir -p {shlex.quote(container)} && cp -a /testbed/. {shlex.quote(container)}/"
        output, status = await executor.execute_command(command, workdir="/testbed")
        if status:
            raise RuntimeError(f"Failed to initialize runtime workspace: {output}")
        return backend, workspace, observation

    async def publish_for_grading(self, base, workspace_id: str) -> str:
        diff = await self.export_diff(base, workspace_id)
        container_patch = self.container_path(diff.artifact_ref)
        if Path(diff.artifact_ref).stat().st_size:
            output, code = await self.executor.execute_command(
                f"git apply --check --binary {shlex.quote(container_patch)} && git apply --binary {shlex.quote(container_patch)}",
                workdir="/testbed",
            )
            if code:
                raise RuntimeError(f"Final patch publication failed: {output}")
        return diff.artifact_ref


class BoundSWEExecutor:
    def __init__(self, workspaces: MountedSWEWorkspace, workspace_id: str) -> None:
        self.workspaces = workspaces
        self.cwd = workspaces.container_path(workspace_id)

    async def execute_command(self, command: str, timeout: int | None = None, **kwargs):
        if re.search(r"(?<![\w])/testbed(?:/|\b)", command):
            return f"Use this node's assigned repository: {self.cwd}", 1
        prefix = (
            "set -o pipefail; source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && "
            f"export PYTHONPATH={shlex.quote(self.cwd + '/src')}:{shlex.quote(self.cwd)}:${{PYTHONPATH:-}} && "
        )
        return await self.workspaces.executor.execute_command(
            prefix + command, timeout=timeout, workdir=self.cwd
        )


class SWEToolBinding:
    def __init__(self, manager: Any) -> None:
        self.manager = manager

    def prepare(self, node: NodeRecord, action: dict[str, Any]) -> PreparedTool:
        if action.get("action") in {"read_file", "read_range", "search"}:
            return self._structured_read(node, action)
        if action.get("action") != "aci_command":
            raise ValueError("SWE environment actions must use aci_command")
        if "aci_command" not in node.contract.allowed_tools:
            raise PermissionError("aci_command was not granted by this contract")
        command = str(action.get("params", {}).get("command", "")).strip()
        if not command or command.split(maxsplit=1)[0] == "submit":
            raise ValueError(
                "Empty commands and agent-owned grader submissions are forbidden"
            )
        parts = command.split(maxsplit=1)
        name, argument = parts[0], parts[1] if len(parts) > 1 else ""
        state = self.manager.state
        tool = "bash"  # Arbitrary shell is conservatively a possible write.

        def call():
            return self.manager.execute(command)

        # Explicit ACI reads only; shell metacharacters never become cacheable reads.
        if re.fullmatch(r"[\w./*? +:\-]*", argument):
            if name == "open":
                args = shlex.split(argument)
                if not args or len(args) > 2:
                    raise ValueError("open requires file and optional line")
                state.current_file = args[0]
                state.current_line = max(1, int(args[1])) if len(args) == 2 else 1
                tool, call = "view", self.manager._display_window
            elif name in {"scroll_up", "scroll_down", "goto"}:
                if not state.current_file:
                    raise ValueError("Open a file before navigating")
                value = int(argument) if argument else state.window_size
                state.current_line = max(
                    1,
                    value
                    if name == "goto"
                    else state.current_line
                    + (-value if name == "scroll_up" else value),
                )
                tool, call = "view", self.manager._display_window
            elif name in {"search_file", "search_dir", "find_file"}:
                tool = "search"

        async def execute():
            output, exit_code = await call()
            return {"output": output, "exit_code": exit_code}

        request = ToolRequest(
            node.run_id,
            node.node_id,
            node.workspace_id,
            tool,
            {
                "command": command,
                "cwd": state.working_dir,
                "file": state.current_file,
                "line": state.current_line,
                "window": state.window_size,
            },
        )
        return PreparedTool(request, execute)

    def _structured_read(self, node, action):
        if "aci_command" not in node.contract.allowed_tools:
            raise PermissionError("SWE repository access was not granted")
        name, params = action["action"], action.get("params", {})
        relative = str(params.get("path", "." if name == "search" else ""))
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative or ".git" in path.parts:
            raise ValueError("Structured reads require a relative repository path")
        workspaces = self.manager.executor.workspaces
        live = Path(node.workspace_id).resolve()
        (live / path).resolve().relative_to(live)
        start, end = int(params.get("start", 1)), int(params.get("end", 200))
        if start < 1 or end < start or end - start > 2000:
            raise ValueError("Invalid read range (maximum 2001 lines)")
        query = str(params.get("query", ""))
        if name == "search" and not query:
            raise ValueError("search requires a literal query")
        args = {"path":path.as_posix()}
        if name == "read_range": args.update(start=start,end=end)
        if name == "search": args["query"] = query
        pinned = None
        async def version():
            nonlocal pinned
            pinned = await workspaces.snapshot(node.workspace_id)
            return str(workspaces.root.resolve()), pinned.revision
        async def read():
            if name == "search":
                output, code = await workspaces._git(str(workspaces._store), "grep", "-n", "-I", "-F", "-e", query, pinned.revision, "--", path.as_posix(), allow_failure=True)
                if code == 1: return {"output":"", "exit_code":0}
                text = output.decode(errors="replace").replace(pinned.revision + ":", "")
                return {"output":text, "exit_code":code}
            output, code = await workspaces._git(str(workspaces._store), "show", f"{pinned.revision}:{path.as_posix()}", allow_failure=True)
            text = output.decode(errors="replace")
            if code == 0 and name == "read_range":
                text = "\n".join(f"{i}: {line}" for i,line in enumerate(text.splitlines(),1) if start <= i <= end)
            return {"output":text, "exit_code":code}
        return PreparedTool(ToolRequest(node.run_id, node.node_id, node.workspace_id, name, args, version_provider=version), read)

    def observation(self, value: Any, *, step: int, annotation: str) -> dict[str, Any]:
        return {
            **value,
            "current_step": step,
            "state_info": self.manager.state.to_status_line(),
            "cache_annotation": annotation,
        }

    async def finish(
        self, node: NodeRecord, params: dict[str, Any], gateway: ToolGateway
    ) -> None:
        return None


class GAIAToolBinding:
    def __init__(self, tools: dict[str, Any]) -> None:
        self.tools = tools

    def prepare(self, node: NodeRecord, action: dict[str, Any]) -> PreparedTool:
        name = str(action.get("action", ""))
        if name not in self.tools or name not in node.contract.allowed_tools:
            raise PermissionError(f"Tool is not authorized: {name}")
        params = dict(action.get("params", {}))
        params.pop("_workdir", None)
        if name == "ExecuteCodeAction":
            params["_workdir"] = node.workspace_id
        policy_name = {
            "GoogleSearchAction": "search",
            "ExtractUrlContentAction": "view",
        }.get(name, name)
        return PreparedTool(
            ToolRequest(
                node.run_id,
                node.node_id,
                node.workspace_id,
                policy_name,
                {"original_tool": name, **params},
            ),
            lambda: self.tools[name](**params),
        )

    def observation(self, value: Any, *, step: int, annotation: str) -> dict[str, Any]:
        return {**value, "current_step": step, "cache_annotation": annotation}

    async def finish(
        self, node: NodeRecord, params: dict[str, Any], gateway: ToolGateway
    ) -> None:
        answer = params.get("result")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "GAIA finish requires params.result containing the proposed answer"
            )

        async def write():
            path = Path(node.workspace_id) / "answer.json"
            await asyncio.to_thread(
                path.write_text,
                json.dumps({"answer": answer}, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"success": True}

        await gateway.execute(
            ToolRequest(
                node.run_id,
                node.node_id,
                node.workspace_id,
                "write",
                {"path": "answer.json"},
            ),
            write,
        )


def import_probe(workspace: str, modules) -> str:
    return (
        "import importlib, pathlib, sys\n"
        f"root = pathlib.Path({workspace!r}).resolve()\n"
        "sys.path[:0] = [str(root / 'src'), str(root)]\n"
        f"for name in {tuple(modules)!r}:\n"
        "    module = importlib.import_module(name)\n"
        "    source = pathlib.Path(module.__file__).resolve()\n"
        "    print(f'FRAMEWORK_IMPORT {name}: {source}', flush=True)\n"
        "    assert source.is_relative_to(root), f'{name} imported outside workspace: {source}'\n"
    )


class SnapshotCheckRunner:
    """Pinned task checks run only after their module provenance is verified."""

    def __init__(self, workspaces: GitWorkspaceBackend) -> None:
        self.workspaces = workspaces

    async def _execute(self, fork, command, deadline):
        if isinstance(self.workspaces, MountedSWEWorkspace):
            output, code = await BoundSWEExecutor(
                self.workspaces, fork
            ).execute_command(
                command, timeout=max(1, math.ceil(deadline - time.monotonic()))
            )
            return code, output, ""
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            (str(Path(fork) / "src"), str(fork), env.get("PYTHONPATH", ""))
        )
        process = await asyncio.create_subprocess_shell(
            ("set -o pipefail; " + command) if os.name != "nt" else command,
            **({"executable": "/bin/bash"} if os.name != "nt" else {}),
            cwd=fork,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        return (
            process.returncode,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )

    async def run(self, check, snapshot, *, deadline_monotonic: float) -> CheckResult:
        fork = None
        try:
            async with asyncio.timeout_at(deadline_monotonic):
                fork = await self.workspaces.fork(
                    snapshot, node_id=f"verify-{uuid4().hex}"
                )
                framework_output = ""
                if check.import_modules:
                    workspace = (
                        self.workspaces.container_path(fork)
                        if isinstance(self.workspaces, MountedSWEWorkspace)
                        else str(fork)
                    )
                    probe = import_probe(workspace, check.import_modules)
                    encoded = base64.b64encode(probe.encode()).decode()
                    script = f"import base64; exec(base64.b64decode('{encoded}'))"
                    python = (
                        "python"
                        if isinstance(self.workspaces, MountedSWEWorkspace)
                        else sys.executable
                    )
                    command = (
                        f"{python} -c {shlex.quote(script)}"
                        if os.name != "nt"
                        or isinstance(self.workspaces, MountedSWEWorkspace)
                        else __import__("subprocess").list2cmdline(
                            [python, "-c", script]
                        )
                    )
                    code, stdout, stderr = await self._execute(
                        fork, command, deadline_monotonic
                    )
                    framework_output = stdout + stderr
                    if code != 0:
                        return CheckResult(
                            check.check_id,
                            code,
                            False,
                            stdout,
                            stderr,
                            category="framework_check",
                        )
                code, stdout, stderr = await self._execute(
                    fork, check.command, deadline_monotonic
                )
                return CheckResult(
                    check.check_id,
                    code,
                    code >= 0,
                    stdout,
                    stderr,
                    timed_out=code == -1,
                    framework_output=framework_output,
                )
        finally:
            if fork is not None:
                await self.workspaces.release(fork)


async def initialize_gaia_workspace(directory: str, objective: str) -> str:
    path = Path(directory).resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "task.txt").write_text(objective, encoding="utf-8")
    for args in (
        ("init",),
        ("add", "."),
        (
            "-c",
            "user.name=POrchestra",
            "-c",
            "user.email=runtime@localhost",
            "commit",
            "-m",
            "GAIA runtime baseline",
        ),
    ):
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(path),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode:
            raise RuntimeError(stderr.decode(errors="replace"))
    return str(path)


GAIA_ANSWER_CHECK = (
    f'"{sys.executable}" -c "import json; d=json.load(open(\'answer.json\')); '
    "assert isinstance(d.get('answer'),str) and d['answer'].strip()\""
)

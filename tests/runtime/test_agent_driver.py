import asyncio
import contextlib
import importlib
import io
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from porchestra.runtime import (
    AcceptanceCheck,
    GlobalMemoryBoard,
    TaskContract,
    ToolGateway,
)
from porchestra.runtime.agent_driver import (
    POrchestraAgentDriver,
    PreparedTool,
    create_legacy_session,
)
from porchestra.runtime.benchmark_bindings import (
    GAIAToolBinding,
    MountedSWEWorkspace,
    SWEToolBinding,
)
from porchestra.runtime.memory import ToolRequest
from porchestra.runtime.types import NodeRecord, WorkspaceSnapshot
from porchestra.runners.runtime_runner import (
    GAIARuntimeRunner,
    PreparedTask,
    RuntimeOptions,
    SWEBenchRuntimeRunner,
)
from test_runtime import MockCheckRunner, MockWorkspaceBackend


def node():
    contract = TaskContract(
        "repair",
        (AcceptanceCheck("c", "pytest"),),
        1000,
        uuid4().hex,
        allowed_tools=("aci_command", "view", "edit"),
    )
    return NodeRecord(
        contract.contract_id,
        "run",
        None,
        0,
        contract,
        uuid4().hex,
        WorkspaceSnapshot("root", "v1", "ref"),
        time.monotonic() + 10,
    )


class ReplaySession:
    def __init__(self, actions):
        self.actions = list(actions)
        self.context = ""
        self.task_instruction = "repair"
        self.tokens = 0
        self.transitions = []
        self.llm = SimpleNamespace(
            get_usage_summary=lambda: {"total_tokens": self.tokens, "total_cost": 0.0}
        )

    async def step(self, **kwargs):
        self.tokens += 5
        action = self.actions.pop(0)
        return action, json.dumps(action), "mock prompt"

    async def record_transition(self, **kwargs):
        self.transitions.append(kwargs)
        self.tokens += 1  # Account for session-owned memory/model work as well.


class ReplayBinding:
    def __init__(self, workspace=None):
        self.calls = 0
        self.workspace = workspace

    def prepare(self, record, action):
        async def execute():
            self.calls += 1
            if self.workspace is not None and action["action"] == "edit":
                self.workspace.files[record.workspace_id]["fix.py"] = "fixed"
            return {"output": "content", "exit_code": 0}

        return PreparedTool(
            ToolRequest(
                record.run_id, record.node_id, record.workspace_id, action["action"], {}
            ),
            execute,
        )

    def observation(self, value, *, step, annotation):
        return {**value, "current_step": step, "cache_annotation": annotation}

    async def finish(self, record, params, gateway):
        pass


class AgentDriverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.board = GlobalMemoryBoard.instance()
        self.record = node()
        self.addCleanup(self.board.release_workspace, self.record.workspace_id)

    def driver(self, session, binding=None):
        return POrchestraAgentDriver(
            session,
            binding or ReplayBinding(),
            ToolGateway(self.board),
            checkpoint_dir=self.temp.name,
        )

    async def test_gateway_cache_and_usage_delta_and_finish_gate(self):
        session = ReplaySession(
            [
                {"action": "view"},
                {"action": "view"},
                {"action": "edit"},
                {"action": "view"},
                {"action": "finish", "params": {}},
            ]
        )
        binding = ReplayBinding()
        driver = self.driver(session, binding)
        outcomes = []
        for _ in range(5):
            outcomes.append(await driver.step(self.record))
            self.record.steps_used += 1
        self.assertEqual(binding.calls, 3)
        self.assertEqual([o.tokens_used for o in outcomes], [6, 6, 6, 6, 5])
        self.assertEqual(outcomes[-1].kind, "finish")
        self.assertEqual(
            session.transitions[1]["observation_after"]["cache_annotation"],
            "[Cache Hit]",
        )
        checkpoint = await driver.checkpoint(self.record.node_id)
        self.assertEqual(json.loads(Path(checkpoint).read_text())["tokens"], 29)
        await driver.inject_context(self.record.node_id, "FINISH REJECTED: repair")
        self.assertIn("FINISH REJECTED", session.context)

    async def test_spawn_parsed_from_legacy_aci_and_preserves_native_tool_reply(self):
        children = [
            {"objective": "repair", "acceptance_checks": ["pytest"], "budget_limit": 20}
            for _ in range(2)
        ]
        session = ReplaySession(
            [
                {
                    "action": "aci_command",
                    "params": {
                        "command": "spawn " + json.dumps({"children": children})
                    },
                }
            ]
        )
        session._pending_tool_call_ids = ["native-call"]
        result = await self.driver(session).step(self.record)
        self.assertEqual(result.kind, "spawn")
        self.assertEqual(len(result.child_contracts), 2)
        self.assertEqual(result.tokens_used, 6)
        self.assertEqual(len(session.transitions), 1)
        self.assertTrue(
            all(
                c.parent_contract_id == self.record.contract.contract_id
                for c in result.child_contracts
            )
        )

    async def test_missing_usage_does_not_invent_zero_tokens(self):
        session = ReplaySession([{"action": "finish", "params": {}}])
        session.llm.get_usage_summary = lambda: {}
        with self.assertRaisesRegex(ValueError, "actual cumulative"):
            await self.driver(session).step(self.record)

    async def test_gaia_uses_node_directory_and_finish_writes_answer(self):
        self.record.workspace_id = self.temp.name
        self.record.contract = replace(
            self.record.contract, allowed_tools=("ExecuteCodeAction",)
        )
        tool = AsyncMock(return_value={"success": True, "output": "computed"})
        binding = GAIAToolBinding({"ExecuteCodeAction": tool})
        session = ReplaySession(
            [
                {
                    "action": "ExecuteCodeAction",
                    "params": {"code": "print(1)", "_workdir": "forged"},
                },
                {"action": "finish", "params": {"result": "42"}},
            ]
        )
        driver = self.driver(session, binding)
        await driver.step(self.record)
        self.assertEqual(tool.await_args.kwargs["_workdir"], self.temp.name)
        self.assertEqual((await driver.step(self.record)).kind, "finish")
        self.assertEqual(
            json.loads((Path(self.temp.name) / "answer.json").read_text())["answer"],
            "42",
        )

    async def test_swe_navigation_state_updates_even_on_cache_hit(self):
        state = SimpleNamespace(
            current_file=None,
            current_line=1,
            window_size=100,
            working_dir="/workspace/root",
            to_status_line=lambda: "node state",
        )
        manager = SimpleNamespace(
            state=state,
            _display_window=AsyncMock(return_value=("file content", 0)),
            execute=AsyncMock(return_value=("result", 0)),
        )
        binding = SWEToolBinding(manager)
        action = {"action": "aci_command", "params": {"command": "open file.py 10"}}
        gateway = ToolGateway(self.board)
        for _ in range(2):
            state.current_file = None
            prepared = binding.prepare(self.record, action)
            result = await gateway.execute(prepared.request, prepared.call)
        self.assertEqual(state.current_file, "file.py")
        self.assertEqual(state.current_line, 10)
        self.assertTrue(result.cache_hit)
        self.assertEqual(manager._display_window.await_count, 1)
        with self.assertRaises(ValueError):
            binding.prepare(
                self.record, {"action": "aci_command", "params": {"command": "submit"}}
            )

    async def test_mount_prepare_resets_once_and_paths_stay_in_mount(self):
        executor = SimpleNamespace(
            set_runtime_workspace_directory=lambda p: None,
            execute_command=AsyncMock(return_value=("", 0)),
        )
        env = SimpleNamespace(
            _executor=executor, reset=AsyncMock(return_value={"message": "ready"})
        )
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=AssertionError("No real process in Mock test"),
        ):
            backend, workspace, _ = await MountedSWEWorkspace.prepare(
                env, self.temp.name
            )
            self.assertTrue(
                backend.container_path(workspace).startswith("/workspace/root-")
            )
            self.assertFalse(backend.shared_clones)
            with self.assertRaises(ValueError):
                backend.container_path(str(Path(self.temp.name).parent / "outside"))
        env.reset.assert_awaited_once()

    async def test_existing_react_session_with_replay_llm_no_api(self):
        from base.engine.async_llm import AsyncLLM

        class ReplayLLM(AsyncLLM):
            def __init__(self):
                self.config = SimpleNamespace(model="mock")
                self.tokens = 0

            async def __call__(self, prompt, **kwargs):
                self.tokens += 7
                self.prompt = prompt
                return "DISCUSSION\nready\nCOMMAND\nfinish {}"

            def get_usage_summary(self):
                return {"total_tokens": self.tokens}

        llm = ReplayLLM()
        info = SimpleNamespace(
            instruction="repair /testbed",
            action_space="aci_command",
            meta_data={"command_docs": "commands"},
        )
        with (
            patch(
                "base.engine.async_llm.LLMsConfig.default",
                return_value=SimpleNamespace(get=lambda name: None),
            ),
            patch("base.engine.async_llm.create_llm_instance", return_value=llm),
            patch(
                "asyncio.create_subprocess_exec",
                side_effect=AssertionError("No external process"),
            ),
        ):
            session = create_legacy_session(
                self.record,
                model="mock",
                benchmark_type="swebench",
                basic_info=info,
                workspace_path="/workspace/child",
            )
            driver = self.driver(session)
            driver.observation = {"output": "literal /testbed sentinel from source"}
            outcome = await driver.step(self.record)
        self.assertEqual(outcome.kind, "finish")
        self.assertEqual(outcome.tokens_used, 7)
        self.assertIn("/workspace/child", llm.prompt)
        self.assertIn("literal /testbed sentinel from source", llm.prompt)

    async def test_runner_uses_runtime_and_mock_environment_without_subprocesses(self):
        backend = MockWorkspaceBackend()
        env = SimpleNamespace(reset=AsyncMock(), close=AsyncMock())
        benchmark = SimpleNamespace(make_env=lambda level: env)
        sessions = []
        finalizations = []

        async def prepare(environment, directory):
            await environment.reset()

            async def finalize(base):
                finalizations.append(base)
                return {"success": True, "reward": 1.0}

            return PreparedTask(
                backend,
                "root",
                SimpleNamespace(instruction="repair"),
                {},
                lambda record: ReplayBinding(backend),
                finalize,
                MockCheckRunner(backend),
            )

        def session_factory(record, **kwargs):
            actions = (
                [{"action": "view"}] * 5
                if record.parent_id is None
                else [{"action": "edit"}]
            ) + [{"action": "finish", "params": {}}]
            session = ReplaySession(actions)
            sessions.append(session)
            return session

        runner = SWEBenchRuntimeRunner(
            benchmark=benchmark,
            main_model="mock-main",
            sub_models=["mock-child"],
            options=RuntimeOptions(
                (AcceptanceCheck("c", "pytest"),), node_concurrency=1
            ),
            session_factory=session_factory,
            prepare_task=prepare,
        )
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=AssertionError("Docker/process creation forbidden"),
        ):
            results = await runner.run_levels(
                [{"id": "mock-task"}],
                1,
                Path(self.temp.name) / "results.csv",
                Path(self.temp.name),
                "mock",
            )
        result = results["mock-task"]
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["nodes"], 3)
        self.assertEqual(len(sessions), 3)
        env.reset.assert_awaited_once()
        env.close.assert_awaited_once()
        self.assertEqual(len(finalizations), 1)
        events = [
            json.loads(line)["event"]
            for line in Path(result["topology_path"]).read_text().splitlines()
        ]
        self.assertEqual(events.count("WAKE"), 1)
        self.assertEqual(events.count("DESTROY"), 3)

    async def test_existing_mini_session_can_spawn_without_native_tool_id_leak(self):
        from base.engine.async_llm import AsyncLLM

        children = [
            {"objective": "repair", "acceptance_checks": ["pytest"], "budget_limit": 20}
            for _ in range(2)
        ]

        class ReplayLLM(AsyncLLM):
            def __init__(self):
                self.config = SimpleNamespace(model="mock")
                self.tokens = 0

            async def call_messages_with_tools(self, messages, **kwargs):
                self.tokens += 8
                return {
                    "content": "DISCUSSION\ndelegate\nCOMMAND\nspawn "
                    + json.dumps({"children": children}),
                    "tool_calls": [],
                }

            def get_usage_summary(self):
                return {"total_tokens": self.tokens}

        info = SimpleNamespace(
            instruction="repair /testbed", action_space="aci_command", meta_data={}
        )
        with (
            patch(
                "base.engine.async_llm.LLMsConfig.default",
                return_value=SimpleNamespace(get=lambda name: None),
            ),
            patch(
                "base.engine.async_llm.create_llm_instance", return_value=ReplayLLM()
            ),
            patch(
                "asyncio.create_subprocess_exec",
                side_effect=AssertionError("No external process"),
            ),
        ):
            session = create_legacy_session(
                self.record,
                model="mock",
                benchmark_type="swebench",
                basic_info=info,
                workspace_path="/workspace/child",
                backend="mini",
            )
            outcome = await self.driver(session).step(self.record)
        self.assertEqual(outcome.kind, "spawn")
        self.assertEqual(outcome.tokens_used, 8)
        self.assertEqual(len(outcome.child_contracts), 2)
        self.assertFalse(session._pending_tool_call_ids)

    async def test_gaia_runner_records_answer_and_scores_only_after_verification(self):
        from porchestra.runtime.types import CheckResult

        workspace = str(Path(self.temp.name) / "gaia-root")
        Path(workspace).mkdir()
        backend = MockWorkspaceBackend()
        backend.files = {workspace: {}}
        env = SimpleNamespace(reset=AsyncMock(), close=AsyncMock())
        benchmark = SimpleNamespace(make_env=lambda level, **kwargs: env)

        class AnswerCheck:
            async def run(self, check, snapshot, *, deadline_monotonic):
                data = json.loads(
                    (Path(snapshot.workspace_id) / "answer.json").read_text()
                )
                passed = data.get("answer") == "42"
                return CheckResult(check.check_id, 0 if passed else 1, passed)

        async def prepare(environment, directory):
            await environment.reset()

            async def finalize(base):
                answer = json.loads((Path(workspace) / "answer.json").read_text())[
                    "answer"
                ]
                return {"reward": 1.0, "success": True, "final_answer": answer}

            return PreparedTask(
                backend,
                workspace,
                SimpleNamespace(instruction="compute answer"),
                {},
                lambda record: GAIAToolBinding({}),
                finalize,
                AnswerCheck(),
            )

        runner = GAIARuntimeRunner(
            benchmark=benchmark,
            main_model="mock",
            sub_models=["mock"],
            options=RuntimeOptions((AcceptanceCheck("answer", "check-answer"),)),
            session_factory=lambda *args, **kwargs: ReplaySession(
                [{"action": "finish", "params": {"result": "42"}}]
            ),
            prepare_task=prepare,
        )
        with (
            patch(
                "asyncio.create_subprocess_exec",
                side_effect=AssertionError("No external process"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = (
                await runner.run_levels(
                    [{"task_id": "gaia-mock", "Question": "compute"}],
                    1,
                    None,
                    Path(self.temp.name),
                    "gaia",
                )
            )["gaia-mock"]
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["final_answer"], "42")
        self.assertEqual(result["tokens_used"], 5)
        env.reset.assert_awaited_once()
        env.close.assert_awaited_once()

    async def test_swe_dry_run_never_constructs_benchmark_or_starts_process(self):
        module = importlib.import_module("bench_porchestra_swebench")
        config = Path(self.temp.name) / "config.yaml"
        config.write_text("main_model: mock\nsub_models: [mock]\n", encoding="utf-8")
        output = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "bench_porchestra_swebench.py",
                    "--config",
                    str(config),
                    "--check-command",
                    "pytest",
                    "--dry-run",
                ],
            ),
            patch(
                "asyncio.create_subprocess_exec",
                side_effect=AssertionError("No process in dry run"),
            ),
            patch(
                "benchmark.bench_swebench.SWEBenchBenchmark",
                side_effect=AssertionError("No dataset loading in dry run"),
            ),
            contextlib.redirect_stdout(output),
        ):
            code = await module.main()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["runner"], "NodeRuntime")

    async def test_executor_timeout_cleans_mock_process_without_docker(self):
        from benchmark.swebench.swebench_executor import SWEBenchExecutor

        executor = SWEBenchExecutor(
            SimpleNamespace(instance_id="mock"), Path(self.temp.name)
        )
        executor.container_id = "mock-container"
        process = SimpleNamespace(
            returncode=None,
            communicate=AsyncMock(side_effect=asyncio.TimeoutError()),
            kill=lambda: setattr(process, "returncode", -9),
            wait=AsyncMock(return_value=-9),
        )
        with patch(
            "asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)
        ):
            _, status = await executor.execute_command("read", timeout=0.01)
        self.assertEqual(status, -1)
        self.assertEqual(process.returncode, -9)
        process.wait.assert_awaited_once()

    async def test_native_api_options_and_reasoning_are_not_executable_content(self):
        from base.engine.async_llm import AsyncLLM

        client = AsyncMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        reasoning_content="COMMAND\nunsafe simulation",
                        tool_calls=[],
                    )
                )
            ],
        )
        llm = object.__new__(AsyncLLM)
        llm.config = SimpleNamespace(
            model="mock",
            temperature=0,
            top_p=1,
            extra_body={"thinking": {"type": "disabled"}},
        )
        llm.max_completion_tokens = 4096
        llm.aclient = client
        llm.usage_tracker = SimpleNamespace(add_usage=lambda *args: {"total_cost": 0.0})
        result = await llm.call_messages_with_tools(
            [{"role": "user", "content": "test"}], tools=[]
        )
        self.assertEqual(result["content"], "")
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )


if __name__ == "__main__":
    unittest.main()

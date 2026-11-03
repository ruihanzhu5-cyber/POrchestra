# POrchestra Pilot V2 修复与七题实机报告

远端开发目录：`/data/src/POrchestra-self-evolving`。远端备份 tag：`pre-fix-baseline-202611031539`；本地先前备份：`pre-fix-baseline-202611031530`。开发分支：`fix/pilot-v2`。

实际分支起点为 `80d36bae5b7ed1c20f61a65f90a0d96b3eaa383b`；用户指定对比基线为 `89fe4b3`。已确认两者运行时代码相同，差异仅新增结果收集脚本。没有切回旧提交，也没有在开发仓库执行 reset/restore/clean。

B1—B5最初在本地编写；用户要求后，通过SSH bundle将相同提交带到远端。B6—B8、后续修复、测试和实机运行均在远端完成。远端缺GitHub认证，使用本地已有认证转发远端提交；运行时代码不在本地继续修改。

## 独立修复提交

|条目|提交|涉及文件（runtime/前缀省略）|行为|定向测试|
|---|---|---|---|---|
|B1|7dfd59f|types.py, health.py, agent_driver.py, runtime.py, lifecycle.py|进展证据替代单纯Diff判定；相同读取打转，不同文件/命令计进展|b1-tests.log: 2；b1-driver-tests.log: 13|
|B2|a933b6c|types.py, contracts.py, agent_driver.py, runtime.py|默认不拆；显式职责、DoD、scope互补后才允许派生|b2-tests.log: 4|
|B3|3bf218c|runtime.py|子启动预算估算；父节点保留整合/验收/修复预算|b3-tests.log: 6|
|B4|ed066eb|types.py, runtime.py|可恢复拒绝不销毁节点，反馈并去重拒绝原因|b4-tests.log: 7|
|B5|dc946a4|types.py, contracts.py, runtime.py, benchmark_bindings.py, scripts/pilot_runtime_swe.py|Config入口、src优先、模块来源检查、pipefail、框架/任务验收分开|b5-tests.log: 10；b5-contract-tests.log: 11|
|B6|32d320b|types.py, handoff.py, agent_driver.py, runtime.py, lifecycle.py, git_workspace.py|交接读取位置、修改、检查、未解决项、下一步|b6-tests.log: 11|
|B7|13d8066|memory.py, benchmark_bindings.py, agent_driver.py, mini_swe_subagent.py, prompts/swebench_mini_subagent.py|结构化读取；按仓库、不可变快照及相对参数复用|b7-tests.log: 12|
|B8|56fbded|lifecycle.py, runtime.py, git_workspace.py, agent_driver.py, memory.py, mini_swe_subagent.py, prompts/swebench_mini_subagent.py|独立已验收且范围有效的子补丁受控合并；候选产物注册读取|b8-tests.log: 14|

新增/修改测试集中在 `tests/runtime/test_pilot_v2.py`、`test_runtime.py`、`test_agent_driver.py`。前五项本地日志随报告归档，远端B1—B5重验10项通过。

## 回归与接线补充提交

- `e35fe8d`：Honor explicit helper plans before health fallback; replace obsolete auto-clone and fatal-cap fixtures (B2/B4 regression)。文件：`porchestra/runtime/runtime.py`, `tests/runtime/test_agent_driver.py`, `tests/runtime/test_runtime.py`。
- `415618b`：Format v2 adapters and verify framework failures and real pipeline exit codes。文件：`porchestra/mini_swe_subagent.py`, `porchestra/prompts/swebench_mini_subagent.py`, `porchestra/runtime/agent_driver.py`, `porchestra/runtime/benchmark_bindings.py`, `porchestra/runtime/git_workspace.py`, `porchestra/runtime/handoff.py`, `porchestra/runtime/health.py`, `porchestra/runtime/lifecycle.py`, `porchestra/runtime/memory.py`, `porchestra/runtime/runtime.py`, `scripts/pilot_runtime_swe.py`, `tests/runtime/test_agent_driver.py`, `tests/runtime/test_pilot_v2.py`, `tests/runtime/test_runtime.py`。
- `1847fb4`：Align Mini prompt with registered structured reads instead of legacy bash-only inspection (B7 regression)。文件：`porchestra/prompts/swebench_mini_subagent.py`, `tests/runtime/test_pilot_v2.py`。
- `31b3cc2`：Request a single native call to match one-action runtime scheduling (B7 integration)。文件：`porchestra/mini_swe_subagent.py`, `porchestra/runtime/agent_driver.py`, `scripts/pilot_runtime_swe.py`, `tests/runtime/test_agent_driver.py`。
- `b77f258`：Preserve pytest generated version in workspace snapshots without exporting a task change (B5 integration)。文件：`porchestra/runtime/benchmark_bindings.py`, `porchestra/runtime/git_workspace.py`, `scripts/pilot_runtime_swe.py`, `tests/runtime/test_pilot_v2.py`。
- `e701aee`：Add locked three-failure and single-success Pilot sets from original baseline。文件：`scripts/pilot_extra_cases.py`, `scripts/pilot_runtime_swe.py`, `tests/runtime/test_pilot_v2.py`。

完整回归首轮70项：9失败、3报错；旧版自动复制/致命拒绝测试过期，另发现显式派生被健康兜底抢占的调度缺陷。更新显式Mock职责和修复优先级后70项通过，未跳过失败。随后新增管道退出码、框架失败不驱动业务修改、原生读取提示、忽略的版本生成文件、额外题组边界等测试，最终76项全部通过。

真实命令：

```bash
cd /data/src/POrchestra-self-evolving
/data/envs/porchestra-swe/bin/python -m unittest discover -s tests/runtime -v
/data/envs/porchestra-swe/bin/python -m compileall -q porchestra/runtime porchestra/runners scripts/pilot_runtime_swe.py
```

最终单测日志：`/data/runs/swebench/pilot-v2-validation/full-extra-cases.log`，76 tests / OK。关键语法类Ruff检查通过；不宣称全仓库所有lint规则无告警。宿主测试环境补装pytest、ruff；benchmark容器使用自己的依赖环境。

## 七题官方结果

模型为DeepSeek Flash，thinking disabled，温度0，预算300000，运行时900秒、官方测试超时600秒，题目串行、节点槽2。全部使用官方harness评分，公开DoD仅作运行时验收。

新增失败题是原版已评分失败中、排除原三题后按原耗时排序取前三；成功对照是已评分成功中耗时最短题。在新运行前固定，没有按新版结果换题。

|题目|原版官方|V2官方|公开DoD|补丁字节|根/子树Token|耗时秒|
|---|---|---|---|---:|---:|---:|
|psf__requests-2674|失败|失败|passed|2515|54174/54174|156.00|
|pytest-dev__pytest-8365|失败|失败|passed|934|45926/45926|16.58|
|sympy__sympy-17630|失败|失败|not_completed_budget_termination|0|314003/314003|48.36|
|sympy__sympy-21171|失败|失败|passed|1685|182549/182549|66.75|
|sympy__sympy-13177|失败|失败|passed|568|143080/143080|69.61|
|sphinx-doc__sphinx-8595|失败|失败|passed|1092|86772/86772|36.84|
|pytest-dev__pytest-11143|成功|成功|passed|532|48600/48600|20.30|

官方成功1/7：`pytest-dev__pytest-11143`，FAIL_TO_PASS 1/1、PASS_TO_PASS 114/114。原失败题恢复0/6，成功对照保留1/1。原版在相同精选七题中也是1/7，成功集合一致；不能据此声称解决率提升，也不能直接与全量50题65%的已评分口径比较。

原三题与上一轮89fe4b3运行时版本对比，Token为 Requests 253088→54174，Pytest8365 306453→45926，Sympy17630 304615→314003。总计864156→414103，但官方结果均为0/3，减少成本不等于改善解题能力。

七题最终尝试875104 Token；两次接线中断51121+51715=102836；V2全部轨迹观测977940 Token。另有原生API探针311 Token，合计观测加探针978251；取消而未返回usage的调用可能另计账单。Pytest8365第一次框架失败为0模型Token，其记录仍保留。

## 修复证据与剩余问题

- Git Diff不变不再自动复制任务：七题均只有根节点，子节点0、SLEEP 0、SPLIT_REJECTED 0；未出现732/765/1315 Token的派生节点。最低子预算、父预留和拒绝后存活由定向测试覆盖，七题没有触发派生，不能把“未出现”当作这些边界的实机压力测试。
- 框架与judge已区分：目标模块来源绑定隔离src，Pytest生成文件随快照保留且不导出为业务修改。公开DoD通过的6题中仅1题官方solved，剩余验收覆盖不足，必须以官方judge为准。
- 结构化read_file/read_range/search真实Binding与不可变Git快照共享已测，修改后读取新内容已测。但七题实际Agent缓存命中均为0；大部分动作仍使用Bash，不能声称已实现实机缓存收益或Token节省来自缓存。
- API请求设置parallel_tool_calls=False，但实机仍出现多工具响应，单动作解析器拒绝后继续；该provider设置不是可靠的单调用保证。需要后续兼容安全的多读取响应，或核查接口行为。
- 独立子结果合并、失败兄弟候选保留、read_artifact越界拒绝已单测覆盖。七题MERGE_COMMIT均为0，没有真实LLM+Docker非空合并证据；本轮也没有实机父子交接可评估。
- 拒绝后保活及拒绝原因去重已测；语义等价但不断改写的派生请求仍可能消耗Token，仍需步骤/预算约束，尚无语义计划级去重。
- Token按响应后真实usage记账，根节点单次响应可能超过预算，Sympy17630实测314003后终止；派生启动预算检查不等于严格的API账单硬上限。

## 未验证点

真实LLM+Docker的子节点非空合并、跨Agent实际缓存收益、递归拓扑的真实职责互补与交接效果，均未获得本轮七题实机证据。两次根节点中断后容器已清理，但容器内进程组、递归子树同时取消及恶意工具写入等场景未专门压测。未启动50题。

## 代码版本、结果与复现

Requests使用31b3cc2。Pytest8365在快照生成文件修复后使用b77f258，与Sympy17630相同；新增四题使用e701aee（只扩展题组入口，不再修改运行时）。每题版本记录在结果/导出汇总中。Requests耗时为容器日志秒级差值约156秒，其余为monotonic实测。

每次复现请使用新的run-dir，以免脚本跳过已评分状态：

```bash
cd /data/src/POrchestra-self-evolving
PY=/data/envs/porchestra-swe/bin/python
$PY -u scripts/pilot_runtime_swe.py --pilot-set original --run-dir /data/runs/swebench/v2-original-new-run
$PY -u scripts/pilot_runtime_swe.py --pilot-set baseline-failures-extra3 --run-dir /data/runs/swebench/v2-extra3-new-run
$PY -u scripts/pilot_runtime_swe.py --pilot-set baseline-success-control --run-dir /data/runs/swebench/v2-control-new-run
```

本轮原始结果目录为 `/data/runs/swebench/self-evolving-pilot-v2-run-20261103`、`self-evolving-pilot-v2-extra3-20261103`、`self-evolving-pilot-v2-control-20261103`。可读TXT在 `/data/runs/swebench/pilot-v2-seven-text-20261103`，七题完整包 `/data/runs/swebench/porchestra-v2-seven-readable-txt.zip`。

每题主TXT包括真实系统事件、动作、工具返回、Token、验收与官方结果；完整LLM输入消息在完整会话目录，原始所有尝试按文本证据归档。没有裁掉失败结果，没有把缓存探针或终端Success当作官方solved。

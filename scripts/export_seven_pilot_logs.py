"""Export the seven fixed V2 Pilot cases as readable UTF-8 text, without rerunning them."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path("/data/runs/swebench")
GROUPS = {
    "original_three": (
        "self-evolving-pilot-v2-run-20261103",
        ("psf__requests-2674", "pytest-dev__pytest-8365", "sympy__sympy-17630"),
    ),
    "additional_failures": (
        "self-evolving-pilot-v2-extra3-20261103",
        ("sympy__sympy-21171", "sympy__sympy-13177", "sphinx-doc__sphinx-8595"),
    ),
    "successful_control": (
        "self-evolving-pilot-v2-control-20261103",
        ("pytest-dev__pytest-11143",),
    ),
}


def records(path):
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def readable(value, level=0):
    prefix = "  " * level
    if isinstance(value, dict):
        return (
            "".join(
                prefix + str(key) + ":\n" + readable(item, level + 1)
                for key, item in value.items()
            )
            or prefix + "{}\n"
        )
    if isinstance(value, list):
        return (
            "".join(
                prefix + f"[{index}]\n" + readable(item, level + 1)
                for index, item in enumerate(value, 1)
            )
            or prefix + "[]\n"
        )
    if isinstance(value, str):
        return "\n".join(prefix + line for line in value.split("\n")) + "\n"
    return prefix + json.dumps(value, ensure_ascii=False) + "\n"


def timestamp(row):
    stamp = row.get("time") or row.get("timestamp_utc")
    if not stamp:
        return 0
    value = datetime.fromisoformat(stamp)
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return value.timestamp()


def show_time(row):
    return datetime.fromtimestamp(timestamp(row), ZoneInfo("Asia/Shanghai")).isoformat(
        timespec="milliseconds"
    )


def export(output):
    output.mkdir(parents=True, exist_ok=True)
    baseline = {
        row["instance_id"]: row
        for row in json.loads(
            (BASE / "pilot-v2-validation/baseline_all7.json").read_text(
                encoding="utf-8-sig"
            )
        )
    }
    cases, archive_sources = [], []
    for group, (run_name, identities) in GROUPS.items():
        run = BASE / run_name
        for identity in identities:
            result = json.loads((run / "status" / f"{identity}.json").read_text())
            if not result.get("scored"):
                raise RuntimeError(
                    f"{identity} has no completed official grade; refuse a final report"
                )
            attempt = int(result["attempt"])
            trace_root = (
                run
                / "attempts"
                / identity
                / f"attempt-{attempt}"
                / "trajectories"
                / "pilot"
                / identity
            )
            topology = list(records(trace_root / "topology.jsonl"))
            rows = []
            traces = list(trace_root.glob("agents/*/events.jsonl"))
            for trace in traces:
                for row in records(trace):
                    rows.append({**row, "_node": trace.parent.name})
            usage = [row for row in rows if row.get("event") == "runtime_agent_usage"]
            tools = [row for row in rows if row.get("event") == "runtime_tool_result"]
            roots = {
                row["node_id"]
                for row in topology
                if row["event"] == "SPAWN" and row["parent_id"] is None
            }
            reports = []
            for artifact in trace_root.glob("artifacts/*.json"):
                receipt = json.loads(artifact.read_text())
                if receipt.get("verification"):
                    reports.append(receipt["verification"])
            official = result["grading"]["official_report"]
            raw_official = official.get("official_report") or {}
            official_dir = (
                run / "attempts" / identity / f"attempt-{attempt}" / "official"
            )
            patches = list(official_dir.glob("*/logs/model_patch.diff"))
            patch_bytes = sum(path.stat().st_size for path in patches)
            counts = Counter(row["event"] for row in topology)
            old = baseline[identity]
            duration = result.get("duration_seconds")
            if duration is None and identity == "psf__requests-2674":
                duration = (
                    156.0  # Container logs 15:57:25 -> 16:00:01, second precision.
                )
            case = {
                "instance_id": identity,
                "group": group,
                "attempt": attempt,
                "original_baseline_resolved": str(old["resolved"]).lower() == "true",
                "original_baseline_tokens": int(old["prompt_tokens"])
                + int(old["completion_tokens"]),
                "original_baseline_duration_seconds": float(old["duration_seconds"]),
                "framework_check": "passed_before_agent_execution",
                "public_dod": "passed"
                if result["contract_accepted"]
                else "not_completed_budget_termination",
                "official_resolved": bool(raw_official.get("resolved")),
                "official_test_summary": official.get("summary"),
                "official_patch_is_none": raw_official.get("patch_is_None"),
                "final_patch_bytes": patch_bytes,
                "root_tokens": sum(
                    row["tokens_used"] for row in usage if row.get("node_id") in roots
                ),
                "observed_subtree_tokens": sum(row["tokens_used"] for row in usage),
                "duration_seconds": duration,
                "duration_source": "container logs, second precision"
                if result.get("duration_seconds") is None
                else "worker monotonic clock",
                "event_counts": dict(counts),
                "child_nodes": sum(
                    row["event"] == "SPAWN" and row["parent_id"] is not None
                    for row in topology
                ),
                "split_count": counts["SLEEP"],
                "split_rejected": counts["SPLIT_REJECTED"],
                "merge_commit_count": counts["MERGE_COMMIT"],
                "nonempty_merge": False,
                "agent_cache_hits": sum(bool(row.get("cache_hit")) for row in tools),
                "action_counts": dict(
                    Counter(
                        row["action"]["action"]
                        for row in rows
                        if row.get("event") == "runtime_agent_action"
                    )
                ),
                "runtime_outcome": result["runtime_outcome"],
                "error": result["error"],
                "code_revision": result.get("code_revision")
                or "31b3cc211afb8b09996fa1d2a946ea65323d0960",
                "run_directory": str(run),
            }
            if counts["MERGE_COMMIT"]:
                raise RuntimeError(
                    "Review merge artifacts before reporting whether a merge is nonempty"
                )
            cases.append(case)
            parts = [
                f"POrchestra V2 实机实验日志：{identity}\n",
                "时间统一显示为北京时间（UTC+8）。以下内容来自真实轨迹，不是模型生成的回顾。\n",
                "一、最终结果（官方 resolved 才代表 solved）\n",
                readable(case),
                "\n二、运行时验收报告与导入来源\n",
                readable(reports),
                "\n三、系统事件、Agent 动作、工具返回和 Token 时间线\n",
            ]
            selected = [
                row
                for row in rows
                if row.get("event")
                in {
                    "runtime_agent_action",
                    "runtime_tool_result",
                    "runtime_agent_usage",
                    "runtime_context",
                }
            ]
            timeline = sorted(topology + selected, key=timestamp)
            for index, row in enumerate(timeline, 1):
                event = row["event"]
                parts.append(
                    f"\n{'=' * 70}\n记录 {index} | {show_time(row)} | {event}\n"
                )
                if event == "PROGRESS":
                    parts.append(
                        f"判断：{row['reason']}；连续重复证据计数：{row['details']['repeated_steps']}\n"
                    )
                elif event == "runtime_agent_action":
                    parts.append(
                        "节点：" + row["_node"] + "\n动作：\n" + readable(row["action"])
                    )
                elif event == "runtime_tool_result":
                    parts.append(
                        "节点："
                        + row["_node"]
                        + "\n真实缓存命中："
                        + str(row.get("cache_hit"))
                        + "\n工具返回：\n"
                        + readable(row["observation"])
                    )
                elif event == "runtime_agent_usage":
                    parts.append(
                        f"节点：{row['_node']}；本步 Token：{row['tokens_used']}；该节点累计 Token：{row['cumulative_tokens']}\n"
                    )
                elif event == "runtime_context":
                    parts.append(
                        "节点："
                        + row["_node"]
                        + "\n注入的上下文：\n"
                        + row["message"]
                        + "\n"
                    )
                else:
                    parts.append(
                        readable(
                            {
                                key: value
                                for key, value in row.items()
                                if key != "details"
                            }
                        )
                        + readable(row.get("details", {}))
                    )
            parts.append("\n四、官方模型补丁\n")
            for path in patches:
                parts.append(path.read_text() or "(没有最终补丁)\n")
            (output / f"{identity}_实验日志.txt").write_text(
                "".join(parts), encoding="utf-8"
            )
            full = output / "完整会话" / f"{identity}_完整会话.txt"
            full.parent.mkdir(exist_ok=True)
            with full.open("w", encoding="utf-8") as stream:
                stream.write(
                    "全部节点原始事件字段，包括实际 LLM 输入消息，未截断或摘要。\n"
                )
                for trace in sorted(traces):
                    stream.write("\n原始文件：" + str(trace) + "\n")
                    for index, row in enumerate(records(trace), 1):
                        stream.write(f"\n记录 {index}\n" + readable(row))
            # Preserve every attempt, including framework failure, as text evidence.
            for source in (run / "attempts" / identity).rglob("*"):
                if (
                    source.is_file()
                    and not source.is_symlink()
                    and "container_workspace" not in source.parts
                ):
                    archive_sources.append(
                        (
                            source,
                            Path("原始文本证据") / run.name / source.relative_to(run),
                        )
                    )
        archive_sources.append(
            (run / "logs/pilot.log", Path("终端日志") / (run.name + ".txt"))
        )
    interrupted_tokens = 0
    for name in (
        "self-evolving-pilot-v2-20261103",
        "self-evolving-pilot-v2-final-20261103",
    ):
        run = BASE / name
        for trace in run.glob(
            "attempts/*/attempt-*/trajectories/pilot/*/agents/*/events.jsonl"
        ):
            interrupted_tokens += sum(
                row["tokens_used"]
                for row in records(trace)
                if row.get("event") == "runtime_agent_usage"
            )
        for source in run.rglob("*"):
            if (
                source.is_file()
                and not source.is_symlink()
                and "container_workspace" not in source.parts
                and source.suffix in {".json", ".jsonl", ".log", ".txt", ".csv"}
            ):
                archive_sources.append(
                    (source, Path("接线中断尝试") / name / source.relative_to(run))
                )
    summary = {
        "cases": cases,
        "scored": len(cases),
        "resolved": sum(c["official_resolved"] for c in cases),
        "original_failure_stratum": {
            "cases": sum(not c["original_baseline_resolved"] for c in cases),
            "recovered": sum(
                c["official_resolved"] and not c["original_baseline_resolved"]
                for c in cases
            ),
        },
        "original_success_control": {
            "cases": sum(c["original_baseline_resolved"] for c in cases),
            "retained": sum(
                c["official_resolved"] and c["original_baseline_resolved"]
                for c in cases
            ),
        },
        "final_case_tokens": sum(c["observed_subtree_tokens"] for c in cases),
        "interrupted_v2_tokens": interrupted_tokens,
        "all_observed_v2_tokens": sum(c["observed_subtree_tokens"] for c in cases)
        + interrupted_tokens,
        "separate_native_api_probe_tokens": 311,
        "accounting_note": "Cancelled calls without returned usage may incur additional billing. Infrastructure cache probes are not agent savings.",
        "unit_tests": 76,
        "unverified": [
            "Real LLM+Docker nonempty child merge (no helper nodes in these seven cases)",
            "Adversarial Docker process-group cancellation cleanup",
            "Actual cross-agent cache benefit (no child nodes; tests verify immutable sharing)",
            "Provider enforcement of parallel_tool_calls=False (some multi-call responses were still rejected and recovered)",
        ],
    }
    assert len(cases) == 7
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = [
        "POrchestra V2 七题实机汇总\n",
        "官方 resolved 是唯一 solved 口径；公开 DoD 和框架检查单独列出。\n\n",
        "题目 | 原版基线 | 新版官方结果 | 公开DoD | 最终补丁字节 | 根/子树Token | 耗时秒\n",
    ]
    for c in cases:
        text.append(
            f"{c['instance_id']} | {'成功' if c['original_baseline_resolved'] else '失败'} | {'成功' if c['official_resolved'] else '失败'} | {c['public_dod']} | {c['final_patch_bytes']} | {c['root_tokens']}/{c['observed_subtree_tokens']} | {c['duration_seconds']:.2f}\n"
        )
    text += [
        "\n全部统计与未验证点\n",
        readable(summary),
        "\n说明：Requests 时间来自秒级容器日志，约156秒；其代码版本31b3cc2。Pytest8365和Sympy17630版本b77f258；新增四题e701aee。\n",
        "Pytest8365首个框架失败尝试为0模型Token，原始记录保留。两次接线中断尝试另列，不冒充完成评分的题目。\n",
        "1/7是选择性诊断样本，不能与原版50题整体解决率直接比较；没有证据支持“新架构已提升解决率”。\n",
        "这七题没有子节点，SLEEP、SPLIT_REJECTED、MERGE_COMMIT均为0；因此不能将它们当作成功合并闭环的实机证据。\n",
    ]
    (output / "七题实验汇总.txt").write_text("".join(text), encoding="utf-8")
    bundle = output.parent / "porchestra-v2-seven-readable-txt.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in output.rglob("*.txt"):
            archive.write(source, source.relative_to(output))
        seen = set()
        for source, relative in archive_sources:
            name = str(relative.with_name(relative.name + ".txt"))
            if name in seen:
                continue
            seen.add(name)
            if source.suffix == ".jsonl":
                content = (
                    "原始文件："
                    + str(source)
                    + "\n"
                    + "".join(
                        f"\n记录 {i}\n" + readable(row)
                        for i, row in enumerate(records(source), 1)
                    )
                )
            elif source.suffix == ".json":
                content = (
                    "原始文件："
                    + str(source)
                    + "\n"
                    + readable(json.loads(source.read_text(encoding="utf-8-sig")))
                )
            else:
                content = source.read_text(encoding="utf-8", errors="replace")
            archive.writestr(name, content)
        for source in sorted((BASE / "pilot-v2-validation").glob("*.log")):
            archive.write(source, "单元测试日志/" + source.name + ".txt")
        assert all(name.endswith(".txt") for name in archive.namelist())
    print(
        json.dumps(
            {
                "scored": summary["scored"],
                "resolved": summary["resolved"],
                "final_case_tokens": summary["final_case_tokens"],
                "interrupted_tokens": interrupted_tokens,
                "bundle": str(bundle),
                "bundle_bytes": bundle.stat().st_size,
                "files": len(list(output.rglob("*.txt"))),
            },
            ensure_ascii=False,
        )
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args().output)

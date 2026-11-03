"""Collect scored Pilot results and topology without exposing prompts or secrets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def records(path: Path):
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # A live writer may not have completed its final line.


def summarize(run: Path):
    manifest = json.loads((run / "pilot-manifest.json").read_text())
    attempts, topology = [], []
    for path in sorted(
        run.glob("attempts/*/attempt-*/trajectories/pilot/*/topology.jsonl")
    ):
        identity = path.parent.name
        attempt = next(p.name for p in path.parents if p.name.startswith("attempt-"))
        events = list(records(path))
        for event in events:
            topology.append({**event, "instance_id": identity, "attempt": attempt})
        usage, tools = [], []
        for trace in path.parent.glob("agents/*/events.jsonl"):
            for row in records(trace):
                if row.get("event") == "runtime_agent_usage":
                    usage.append(row)
                if row.get("event") == "runtime_tool_result":
                    tools.append(row)
        result_path = path.parent / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        attempts.append(
            {
                "instance_id": identity,
                "attempt": attempt,
                "event_counts": dict(Counter(e["event"] for e in events)),
                "observed_api_tokens": sum(row["tokens_used"] for row in usage),
                "agent_steps_recorded": len(usage),
                "agent_tool_results": len(tools),
                "agent_cache_hits": sum(bool(row.get("cache_hit")) for row in tools),
                "runtime_outcome": result.get("runtime_outcome"),
                "contract_accepted": result.get("contract_accepted"),
                "official_resolved": result.get("success")
                if result.get("grading")
                else None,
                "topology_source": str(path),
            }
        )
    cases = []
    for identity in manifest["pilot_ids"]:
        path = run / "status" / f"{identity}.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        cases.append(
            {
                "instance_id": identity,
                **{
                    key: result.get(key)
                    for key in (
                        "attempt",
                        "scored",
                        "success",
                        "runtime_outcome",
                        "contract_accepted",
                        "tokens_used",
                        "error",
                        "grading",
                    )
                },
            }
        )
    summary = {
        "manifest": manifest,
        "cases": cases,
        "scored": sum(bool(case["scored"]) for case in cases),
        "resolved": sum(bool(case["success"]) for case in cases),
        "observed_api_tokens_all_attempts": sum(
            a["observed_api_tokens"] for a in attempts
        ),
        "token_accounting_note": "Trace-observed API tokens include retries and over-quota responses; cancelled requests without returned usage and separate API probes may be additional.",
        "cache_accounting_note": "Agent cache hits exclude explicitly labelled infrastructure CACHE_PROBE calls.",
        "attempts": attempts,
    }
    (run / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (run / "topology.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in topology), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(summarize(parser.parse_args().run_dir), indent=2))

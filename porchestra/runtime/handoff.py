"""Observed, bounded handoffs; never promote an agent assertion to a verified fact."""

from dataclasses import asdict


def summarize(node, *, change_summary="", inherited=False, outcome="active", checks=()):
    progress = node.progress_log[-24:]
    reads, ran = [], []
    for row in progress:
        tool, args = row.get("tool", ""), row.get("arguments", {})
        command = str(args.get("command", ""))
        evidence = {"tool": tool, "arguments": args, "exit_code": row.get("exit_code")}
        if tool in {"view", "read_file", "read_range", "search"} or any(
            w in command for w in ("cat ", "sed ", "grep ", "head ")
        ):
            reads.append(evidence)
        if any(
            w in command for w in ("pytest", "unittest", "python -c", "python - <<")
        ):
            ran.append({**evidence, "output": row.get("output_excerpt", "")})
    ran.extend(asdict(check) for check in checks)
    return {
        "read_locations": reads,
        "changes": {
            "state": "inherited_from_parent"
            if inherited
            else ("modified" if change_summary else "no_new_changes"),
            "summary": change_summary,
            "inherited_changes": node.handoff.get("changes", {}),
        },
        "checks_run": ran,
        "unresolved_issues": [] if outcome == "verified" else [outcome],
        "next_steps": "Verify the integrated result against your own contract"
        if outcome == "verified"
        else "Review evidence and candidate patches; continue locally or propose a complementary duty",
    }

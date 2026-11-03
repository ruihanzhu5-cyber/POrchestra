"""Module 4: end-of-step health decisions; the runtime owns forced spawning."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from .types import HealthDecision, NodeRecord


@dataclass(frozen=True)
class HealthPolicy:
    stagnant_steps: int = 5
    overload_ratio: float = 0.80

    def __post_init__(self) -> None:
        if type(self.stagnant_steps) is not int or self.stagnant_steps < 1:
            raise ValueError("stagnant_steps must be a positive integer")
        if not math.isfinite(self.overload_ratio) or not 0 < self.overload_ratio <= 1:
            raise ValueError("overload_ratio must be in (0, 1]")


class HealthMonitor:
    policy: HealthPolicy

    def __init__(self, policy: HealthPolicy = HealthPolicy()) -> None:
        self.policy = policy

    def observe(
        self, node: NodeRecord, *, diff_fingerprint: str, progress=None
    ) -> HealthDecision:
        """Track observed tool/result novelty as well as workspace changes.

        Missing driver evidence is conservative: identical empty steps repeat.
        Novel reads/commands/errors are progress even when Git has no new diff.
        """
        if type(node.tokens_used) is not int or node.tokens_used < 0:
            raise ValueError("tokens_used must be a nonnegative integer")
        if node.contract.budget_limit <= 0:
            raise ValueError("Contract token budget must be positive")
        evidence = {**(progress or {}), "diff": diff_fingerprint}
        fingerprint = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, default=str).encode()
        ).hexdigest()
        # A new read range, command, result/error or diff is observable progress.
        # Cycling through previously seen evidence still counts as repetition.
        if fingerprint in node.progress_seen:
            node.unchanged_diff_steps += 1
        else:
            node.unchanged_diff_steps = 1
        node.progress_seen.add(fingerprint)
        node.progress_log.append({**evidence, "fingerprint": fingerprint})
        node.last_diff_fingerprint = diff_fingerprint
        ratio = node.tokens_used / node.contract.budget_limit
        reasons = []
        if node.unchanged_diff_steps >= self.policy.stagnant_steps:
            reasons.append(
                f"stagnation:{node.unchanged_diff_steps} repeated progress signals"
            )
        if ratio >= self.policy.overload_ratio:
            reasons.append(
                f"token_budget:{node.tokens_used}/{node.contract.budget_limit}"
            )
        latched = node.health_split_version == node.contract.version
        return HealthDecision(
            bool(reasons) and not latched,
            "; ".join(reasons),
            node.unchanged_diff_steps,
            ratio,
            "circling"
            if node.unchanged_diff_steps >= self.policy.stagnant_steps
            else "investigating",
        )

    def acknowledge_split(self, node: NodeRecord) -> None:
        """Latch the contract version only after the child batch is registered."""
        node.health_split_version = node.contract.version

    def reset_after_wake(self, node: NodeRecord, *, diff_fingerprint: str) -> None:
        """Reset stagnation baseline; retain split latch and consumed subtree budget."""
        node.last_diff_fingerprint = diff_fingerprint
        node.unchanged_diff_steps = 0

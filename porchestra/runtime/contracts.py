"""Rule-based contract creation and verification against real runner observations."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import replace
from typing import Any
from uuid import uuid4

from .types import (
    AcceptanceCheck,
    CheckResult,
    CheckRunner,
    HealthDecision,
    NodeRecord,
    TaskContract,
    VerificationReport,
    WorkspaceBackend,
)


def _positive_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate(contract: TaskContract, parent: TaskContract | None = None) -> None:
    if not isinstance(contract, TaskContract):
        raise ValueError("Expected a TaskContract")
    if not isinstance(contract.objective, str) or not contract.objective.strip():
        raise ValueError("objective must be non-empty")
    if not isinstance(contract.contract_id, str) or not contract.contract_id.strip():
        raise ValueError("contract_id must be non-empty")
    if type(contract.version) is not int or contract.version < 1:
        raise ValueError("version must be a positive integer")
    if type(contract.budget_limit) is not int or contract.budget_limit < 1:
        raise ValueError("budget_limit must be a positive integer")
    if not _positive_number(contract.deadline_seconds):
        raise ValueError("deadline_seconds must be finite and positive")
    if (
        not isinstance(contract.acceptance_checks, tuple)
        or not contract.acceptance_checks
    ):
        raise ValueError("At least one executable acceptance check is required")
    seen: set[str] = set()
    for check in contract.acceptance_checks:
        if not isinstance(check, AcceptanceCheck):
            raise ValueError("acceptance_checks must contain AcceptanceCheck records")
        if (
            not isinstance(check.check_id, str)
            or not check.check_id.strip()
            or check.check_id in seen
        ):
            raise ValueError("check_id must be non-empty and unique")
        seen.add(check.check_id)
        if not isinstance(check.command, str) or not check.command.strip():
            raise ValueError(f"{check.check_id}: command must be non-empty")
        if type(check.expected_exit_code) is not int:
            raise ValueError(f"{check.check_id}: expected_exit_code must be an integer")
        if not _positive_number(check.timeout_seconds):
            raise ValueError(
                f"{check.check_id}: timeout_seconds must be finite and positive"
            )
        if check.expected_output_pattern is not None:
            if (
                not isinstance(check.expected_output_pattern, str)
                or not check.expected_output_pattern
            ):
                raise ValueError(
                    f"{check.check_id}: output pattern must be a non-empty regex"
                )
            try:
                re.compile(check.expected_output_pattern)
            except re.error as exc:
                raise ValueError(
                    f"{check.check_id}: invalid output regex: {exc}"
                ) from exc
    for name in ("constraints", "allowed_tools"):
        values = getattr(contract, name)
        if not isinstance(values, tuple) or any(
            not isinstance(v, str) or not v.strip() for v in values
        ):
            raise ValueError(f"{name} must be a tuple of non-empty strings")
    if parent is not None:
        if contract.parent_contract_id != parent.contract_id:
            raise ValueError("Child contract must reference its parent")
        if contract.contract_id == parent.contract_id:
            raise ValueError("Child and parent contract IDs must differ")
        if contract.budget_limit > parent.budget_limit:
            raise ValueError("Child budget exceeds parent budget")
        if contract.deadline_seconds > parent.deadline_seconds:
            raise ValueError("Child deadline exceeds parent deadline")
        if not set(parent.constraints).issubset(contract.constraints):
            raise ValueError("Child contract dropped inherited constraints")
        if not set(contract.allowed_tools).issubset(parent.allowed_tools):
            raise ValueError("Child contract expanded its tool permissions")


class ContractGenerator:
    """Parse JSON, shell fences, DoD lines or backticked test commands.

    Plain text requires explicit executable DoD or caller-configured defaults.
    This parser does not infer semantic coverage of an arbitrary task.
    """

    def __init__(
        self,
        *,
        default_checks: tuple[AcceptanceCheck, ...] = (),
        default_tools: tuple[str, ...] = (),
        default_deadline_seconds: float = 600.0,
    ) -> None:
        self.default_checks = default_checks
        self.default_tools = default_tools
        self.default_deadline_seconds = default_deadline_seconds

    async def create(
        self,
        objective: str,
        *,
        budget_limit: int,
        parent: TaskContract | None = None,
    ) -> TaskContract:
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be non-empty text or a JSON object")
        text = objective.strip()
        fenced_json = re.fullmatch(
            r"```json\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE
        )
        data: dict[str, Any] = {}
        if text.startswith("{") or fenced_json:
            parsed = json.loads(fenced_json.group(1) if fenced_json else text)
            if not isinstance(parsed, dict):
                raise ValueError("Structured contract must be a JSON object")
            data = parsed
            text = data.get("objective", "")
        raw_checks = data.get("acceptance_checks")
        if "acceptance_checks" in data:
            if not isinstance(raw_checks, list):
                raise ValueError("acceptance_checks must be a JSON array")
            checks = tuple(
                self._parse_check(item, i) for i, item in enumerate(raw_checks, 1)
            )
        else:
            checks = self._extract_checks(text) or self.default_checks
        limit = data.get("budget_limit", budget_limit)
        if (
            type(budget_limit) is not int
            or type(limit) is not int
            or limit > budget_limit
        ):
            raise ValueError(
                "Structured budget must not exceed the caller's integer budget"
            )
        constraints = self._strings(data.get("constraints", []), "constraints")
        inherited = parent.constraints if parent else ()
        tools = (
            self._strings(data["allowed_tools"], "allowed_tools")
            if "allowed_tools" in data
            else (parent.allowed_tools if parent else self.default_tools)
        )
        default_deadline = (
            min(self.default_deadline_seconds, parent.deadline_seconds)
            if parent
            else (self.default_deadline_seconds)
        )
        contract = TaskContract(
            objective=text,
            acceptance_checks=checks,
            budget_limit=limit,
            contract_id=uuid4().hex,
            parent_contract_id=parent.contract_id if parent else None,
            constraints=tuple(dict.fromkeys((*inherited, *constraints))),
            allowed_tools=tools,
            deadline_seconds=data.get("deadline_seconds", default_deadline),
        )
        _validate(contract, parent)
        return contract

    @staticmethod
    def _strings(value: Any, name: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(
            not isinstance(v, str) or not v.strip() for v in value
        ):
            raise ValueError(f"{name} must be an array of non-empty strings")
        return tuple(value)

    @staticmethod
    def _parse_check(value: Any, index: int) -> AcceptanceCheck:
        if isinstance(value, str):
            return AcceptanceCheck(check_id=f"check_{index}", command=value)
        if not isinstance(value, dict):
            raise ValueError(
                "Each acceptance check must be a command string or an object"
            )
        try:
            return AcceptanceCheck(**{"check_id": f"check_{index}", **value})
        except TypeError as exc:
            raise ValueError(f"Invalid acceptance check {index}: {exc}") from exc

    @staticmethod
    def _extract_checks(text: str) -> tuple[AcceptanceCheck, ...]:
        if not isinstance(text, str):
            raise ValueError("Structured objective must be a string")
        commands = re.findall(
            r"```(?:bash|sh|shell|powershell|pwsh|cmd)\s*\n(.*?)\n```", text, re.DOTALL
        )
        unfenced = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        commands.extend(
            re.findall(
                r"^\s*(?:[-*]\s*)?(?:DoD|验收命令|验收|check|command)\s*[:：]\s*(.+?)\s*$",
                unfenced,
                re.MULTILINE | re.IGNORECASE,
            )
        )
        test_prefix = re.compile(
            r"^(?:pytest\b|python(?:3)?\s+-m\s+(?:pytest|unittest)\b|npm\s+test\b|"
            r"npx\s+(?:vitest|jest)\b|go\s+test\b|cargo\s+test\b|dotnet\s+test\b|"
            r"mvn\s+test\b|(?:\./)?gradlew?\s+test\b|ctest\b|make\s+test\b|tox\b)"
        )
        commands.extend(
            cmd
            for cmd in re.findall(r"`([^`\n]+)`", unfenced)
            if test_prefix.match(cmd.strip())
        )
        unique = dict.fromkeys(
            cmd.strip().strip("`") for cmd in commands if cmd.strip()
        )
        return tuple(
            AcceptanceCheck(check_id=f"check_{i}", command=cmd)
            for i, cmd in enumerate(unique, 1)
        )

    async def split(
        self,
        node: NodeRecord,
        trigger: HealthDecision,
        *,
        available_budget: int,
    ) -> tuple[TaskContract, TaskContract]:
        """Conservative rule fallback: partition DoD without inventing weaker checks.

        With a single check both duties must pass it independently. Domain-specific
        generators can override this method to provide disjoint, checkable duties.
        available_budget excludes the parent's retained repair quota.
        """
        if type(available_budget) is not int or available_budget < 2:
            raise ValueError(
                "Splitting requires at least two remaining token allocations"
            )
        _validate(node.contract)
        checks = node.contract.acceptance_checks
        groups = (checks[::2], checks[1::2] or checks[::2])
        limits = (available_budget // 2, available_budget - available_budget // 2)
        children = tuple(
            TaskContract(
                objective=f"{role}: {node.contract.objective}\nTrigger: {trigger.reason}",
                acceptance_checks=group,
                budget_limit=limit,
                contract_id=uuid4().hex,
                parent_contract_id=node.contract.contract_id,
                constraints=node.contract.constraints,
                allowed_tools=node.contract.allowed_tools,
                deadline_seconds=node.contract.deadline_seconds,
            )
            for role, group, limit in zip(
                (
                    "Investigate and resolve a verifiable part",
                    "Implement and validate a repair",
                ),
                groups,
                limits,
                strict=True,
            )
        )
        for child in children:
            _validate(child, node.contract)
        return children[0], children[1]


class ContractVerifier:
    def __init__(self, runner: CheckRunner, workspaces: WorkspaceBackend) -> None:
        self.runner = runner
        self.workspaces = workspaces

    def validate_contract(
        self, contract: TaskContract, *, parent: TaskContract | None = None
    ) -> None:
        """Validate structure and inherited limits; semantic scope needs parent review."""
        _validate(contract, parent)

    async def verify(self, node: NodeRecord) -> VerificationReport:
        verification_id = uuid4().hex
        contract = node.contract
        snapshot = node.base_snapshot
        results: list[CheckResult] = []
        errors: list[str] = []
        try:
            self.validate_contract(contract)
            if not math.isfinite(node.deadline_monotonic):
                raise ValueError("Node deadline must be finite")
            remaining = node.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Node deadline expired before verification")
            snapshot = await asyncio.wait_for(
                self.workspaces.snapshot(node.workspace_id), remaining
            )
            if snapshot.workspace_id != node.workspace_id:
                raise ValueError("Snapshot belongs to a different workspace")
        except Exception as exc:
            errors.append(f"Verification setup failed: {type(exc).__name__}: {exc}")
        else:
            for check in contract.acceptance_checks:
                deadline = min(
                    node.deadline_monotonic, time.monotonic() + check.timeout_seconds
                )
                remaining = deadline - time.monotonic()
                reason = ""
                try:
                    if remaining <= 0:
                        raise TimeoutError("Node deadline expired")
                    raw = await asyncio.wait_for(
                        self.runner.run(check, snapshot, deadline_monotonic=deadline),
                        remaining,
                    )
                    if (
                        not isinstance(raw, CheckResult)
                        or raw.check_id != check.check_id
                    ):
                        raise ValueError(
                            "CheckRunner returned a mismatched CheckResult"
                        )
                    if (
                        type(raw.exit_code) not in (int, type(None))
                        or not isinstance(raw.stdout, str)
                        or not isinstance(raw.stderr, str)
                    ):
                        raise ValueError(
                            "CheckRunner returned invalid exit code or output types"
                        )
                    matches = (
                        check.expected_output_pattern is None
                        or re.search(
                            check.expected_output_pattern,
                            raw.stdout + "\n" + raw.stderr,
                            re.MULTILINE,
                        )
                        is not None
                    )
                    passed = (
                        raw.passed is True
                        and not raw.timed_out
                        and (raw.exit_code == check.expected_exit_code and matches)
                    )
                    result = replace(raw, passed=passed)
                    if not passed:
                        reason = (
                            f"expected exit={check.expected_exit_code}, actual={raw.exit_code}; "
                            f"output_pattern_match={matches}; timed_out={raw.timed_out}; "
                            f"runner_passed={raw.passed}"
                        )
                except TimeoutError as exc:
                    result = CheckResult(
                        check.check_id,
                        None,
                        False,
                        stderr=str(exc) or "Check timed out",
                        timed_out=True,
                    )
                    reason = "Acceptance check timed out"
                except Exception as exc:
                    result = CheckResult(
                        check.check_id,
                        None,
                        False,
                        stderr=f"{type(exc).__name__}: {exc}",
                    )
                    reason = "Acceptance check execution failed"
                results.append(result)
                if not result.passed:
                    errors.append(f"{check.check_id} ({check.command}): {reason}")
            try:
                remaining = node.deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Node deadline expired during verification")
                current = await asyncio.wait_for(
                    self.workspaces.snapshot(node.workspace_id), remaining
                )
                if (current.workspace_id, current.revision) != (
                    snapshot.workspace_id,
                    snapshot.revision,
                ):
                    errors.append(
                        "Workspace changed during verification; verify the new snapshot before finishing"
                    )
                if node.contract != contract:
                    errors.append(
                        "Contract changed during verification; verify the new contract version"
                    )
            except Exception as exc:
                errors.append(f"Snapshot recheck failed: {type(exc).__name__}: {exc}")
        return VerificationReport(
            verification_id=verification_id,
            node_id=node.node_id,
            contract_id=contract.contract_id,
            contract_version=contract.version,
            snapshot=snapshot,
            passed=not errors and bool(results) and all(r.passed for r in results),
            checks=tuple(results),
            error_log="\n".join(errors),
        )

    def format_feedback(self, report: VerificationReport) -> str:
        if report.passed:
            return f"Verification passed: {len(report.checks)} checks on snapshot {report.snapshot.revision}."
        sections = [
            "FINISH REJECTED: repair the failed criteria and request verification again.",
            f"Contract {report.contract_id} v{report.contract_version}; snapshot {report.snapshot.revision}",
            _excerpt(report.error_log, 4000),
        ]
        for result in report.checks:
            if not result.passed:
                sections.append(
                    f"[FAIL] {result.check_id}: exit={result.exit_code}, timed_out={result.timed_out}\n"
                    f"stdout:\n{_excerpt(result.stdout) or '(empty)'}\n"
                    f"stderr:\n{_excerpt(result.stderr) or '(empty)'}"
                )
        return _excerpt("\n\n".join(sections), 16000)


def _excerpt(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    marker = "\n[... output truncated ...]\n"
    head = (limit - len(marker)) // 3
    return text[:head] + marker + text[-(limit - len(marker) - head) :]

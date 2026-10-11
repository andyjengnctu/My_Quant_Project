"""Permanent SSOT regression contracts, consumed by the existing synthetic suite.

This is not a second test runner.  Golden expectations are approved, inert
fixture data; the implementation under test must never generate its own oracle.
"""

from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

from .meta_contracts import (
    load_synthetic_registry_entries_from_source,
    summarize_no_reverse_app_import_contract,
)

GOLDEN_RELATIVE_PATH = "tools/validate/baselines/ssot_regression_v1.json"
# A deliberately separate anchor: editing a Golden fixture is not sufficient
# to change what the existing regression gate accepts.
APPROVED_GOLDEN_SHA256 = "172b828579a2302be866092cbe12897a2e8c9ffe2d6ab7fd6bc8ada1b14d4a9b"
# A user-owned, protected Git tag is a stronger independent anchor after the
# first local Windows acceptance.  It cannot be created from a source ZIP.
USER_APPROVED_GIT_REF = "refs/tags/ssot-contract-approved-v1"

IDENTITY_PATTERN = re.compile(r"^(?:MR-[A-Za-z0-9-]+|ARCH-[A-Za-z0-9_]+|PARAM-P[A-Za-z0-9-]+)$")
GENERIC_CONSUMER_PATHS = (
    "core/portfolio_engine.py",
    "services/research/artifact_orchestrator.py",
    "services/research/strategy_compare_reuse.py",
    "services/optimizer/application.py",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _file_ast(project_root: Path, relative_path: str) -> ast.Module:
    return ast.parse((project_root / relative_path).read_text(encoding="utf-8"), filename=relative_path)


def _named_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    return next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name), None)


def frozen_test_function_hash(project_root: Path, relative_path: str, function_name: str) -> str | None:
    """Freeze meaningful AST, not formatting/comments or line numbers."""
    node = _named_function(_file_ast(project_root, relative_path), function_name)
    if node is None:
        return None
    return _sha256(ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8"))


def snapshot_structural_violations(project_root: Path) -> dict[str, list[str]]:
    """Invariants, not a frozen file count, new MR list or current config values."""
    result: dict[str, list[str]] = {
        "reverse_dependencies": [],
        "architecture_family_switch": [],
        "scientific_identity_in_generic_consumer": [],
    }
    direction = summarize_no_reverse_app_import_contract(project_root)
    result["reverse_dependencies"] = [
        f"{item['path']}:{item['lineno']}->{item['module']}"
        for item in direction["violations"]
    ]

    relative = "filters/breakout_quality/models/spec.py"
    tree = _file_ast(project_root, relative)
    sequence_validator = _named_function(tree, "validate_model_sequence_length")
    if sequence_validator is None:
        result["architecture_family_switch"].append(f"{relative}:missing_validator")
    else:
        for node in ast.walk(sequence_validator):
            if not isinstance(node, ast.Compare):
                continue
            # Disallow all direct family-name branching within generic sequence
            # validation. Descriptor-derived, parameterized spec fields are safe.
            if any(isinstance(sub, ast.Attribute) and sub.attr == "family" for sub in ast.walk(node)):
                result["architecture_family_switch"].append(f"{relative}:{node.lineno}")
        source_attributes = {
            node.attr for node in ast.walk(sequence_validator)
            if isinstance(node, ast.Attribute)
        }
        if not {"patch_transformer_patch_size", "patch_transformer_patch_stride"} <= source_attributes:
            result["architecture_family_switch"].append(f"{relative}:missing_patch_geometry_contract")

    for relative in GENERIC_CONSUMER_PATHS:
        for node in ast.walk(_file_ast(project_root, relative)):
            if not isinstance(node, (ast.If, ast.IfExp, ast.Match)):
                continue
            # Legitimate registration/manifest literal data elsewhere are not
            # affected. Only an identity-specific execution branch is a leak.
            expr = node.test if isinstance(node, (ast.If, ast.IfExp)) else node.subject
            for sub in ast.walk(expr):
                if (isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                        and IDENTITY_PATTERN.fullmatch(sub.value)):
                    result["scientific_identity_in_generic_consumer"].append(
                        f"{relative}:{sub.lineno}:{sub.value}"
                    )
    return result


def _tagged_file_bytes(project_root: Path, relative_path: str) -> bytes | None:
    """Read a user-approved, local Git tag without invoking network or changing Git."""
    try:
        probe = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--verify", "--quiet", f"{USER_APPROVED_GIT_REF}^{{commit}}"],
            capture_output=True, check=False, timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Source-ZIP environments lack .git. The pinned digest remains active;
        # external approval protection becomes available only after local tagging.
        return None
    if probe.returncode:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "show", f"{USER_APPROVED_GIT_REF}:{relative_path}"],
            capture_output=True, check=False, timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"無法核對 approved Git tag: {relative_path}") from exc
    if result.returncode:
        raise ValueError(f"approved Git tag 缺少必要契約: {relative_path}")
    return result.stdout


def read_approved_baseline(project_root: Path, *, raw_bytes: bytes | None = None) -> dict[str, Any]:
    if raw_bytes is None:
        raw_bytes = (project_root / GOLDEN_RELATIVE_PATH).read_bytes()
    if _sha256(raw_bytes) != APPROVED_GOLDEN_SHA256:
        raise ValueError("SSOT Golden 基準雜湊不符合已核准版本，禁止直接重寫 expected 值")
    tagged = _tagged_file_bytes(project_root, GOLDEN_RELATIVE_PATH)
    if tagged is not None and raw_bytes != tagged:
        raise ValueError("SSOT Golden 與使用者批准的 Git tag 不一致")
    baseline = json.loads(raw_bytes)
    if baseline.get("schema_version") != 1:
        raise ValueError("SSOT Golden schema_version 不支援")
    return baseline


def test_contract_violations(project_root: Path, approved: Mapping[str, Any]) -> list[str]:
    registry_entries = load_synthetic_registry_entries_from_source(project_root)
    registered = [entry["name"] for entry in registry_entries]
    problems: list[str] = []
    for item in approved["protected_tests"]:
        relative = item["path"]
        function = item["function"]
        if registered.count(function) != 1:
            problems.append(f"{function}:registration_count={registered.count(function)}")
        actual = frozen_test_function_hash(project_root, relative, function)
        if actual != item["ast_sha256"]:
            problems.append(f"{relative}:{function}:assertions_or_logic_changed")
    for item in approved["protected_guards"]:
        actual = frozen_test_function_hash(project_root, item["path"], item["function"])
        if actual != item["ast_sha256"]:
            problems.append(f"{item['path']}:{item['function']}:guard_implementation_changed")
    return problems


def _rounded_number(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _portfolio_digest(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Compare event chronology plus exact accounting and strategy outcomes."""
    keys = (
        "total_return", "mdd", "trade_count", "final_eq", "portfolio_buy_rows",
        "portfolio_full_exit_rows", "portfolio_half_take_profit_rows",
        "portfolio_missed_buy_rows", "total_missed",
    )
    return {
        **{name: _rounded_number(stats[name]) for name in keys},
        "completed_trades": [
            {
                "buy_date": trade["buy_date"],
                "exit_date": trade["exit_date"],
                "total_pnl": _rounded_number(trade["total_pnl"]),
                "half_exit_count": trade["half_exit_count"],
                "full_exit_action": trade["full_exit_action"],
            }
            for trade in stats["portfolio_completed_trades"]
        ],
    }


def compute_isolated_behavior_snapshot(approved: Mapping[str, Any]) -> dict[str, Any]:
    """Actual producer output for fixed, synthetic inputs; NEVER called to update expected."""
    from core.exact_accounting import build_buy_ledger_from_price, build_sell_ledger_from_price
    from core.strategy_params import V16StrategyParams
    from .synthetic_fixtures import write_synthetic_csv_bundle
    from .synthetic_portfolio_common import (
        build_synthetic_half_tp_full_year_case,
        build_synthetic_same_day_sell_block_case,
        run_portfolio_core_check_for_dir,
    )

    params = V16StrategyParams(**approved["fixed_params"])
    accounting: dict[str, Any] = {}
    for kind, ledger in (
        ("buy", build_buy_ledger_from_price(100.0, 10, params)),
        ("sell", build_sell_ledger_from_price(95.0, 10, params, ticker="2330", trade_date="2024-03-20")),
    ):
        accounting[kind] = {key: value for key, value in sorted(ledger.items()) if isinstance(value, int)}

    scenarios = {
        "same_day_no_reallocation": build_synthetic_same_day_sell_block_case,
        "half_take_profit": build_synthetic_half_tp_full_year_case,
    }
    portfolio: dict[str, Any] = {}
    for case_name, builder in scenarios.items():
        case = builder(params)
        with tempfile.TemporaryDirectory(prefix="ssot_golden_") as tmp:
            write_synthetic_csv_bundle(tmp, case["frames"])
            stats = run_portfolio_core_check_for_dir(
                tmp, case["params"], max_positions=case["max_positions"],
                enable_rotation=case["enable_rotation"],
                start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"],
            )
        portfolio[case_name] = _portfolio_digest(stats)
    return {"accounting": accounting, "portfolio": portfolio}


def behavior_snapshot_differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Return stable, addressable differences; no blind baseline auto-update."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = sorted(set(expected) | set(actual))
        differences = []
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in expected or key not in actual:
                differences.append(f"{child}:missing_or_extra")
            else:
                differences.extend(behavior_snapshot_differences(expected[key], actual[key], child))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}:length:{len(expected)}!={len(actual)}"]
        differences = []
        for index, (before, after) in enumerate(zip(expected, actual)):
            differences.extend(behavior_snapshot_differences(before, after, f"{path}[{index}]"))
        return differences
    return [] if type(expected) is type(actual) and expected == actual else [f"{path}:mismatch"]


def summarize_ssot_regression_guard(project_root: Path) -> dict[str, Any]:
    baseline = read_approved_baseline(project_root)
    actual = compute_isolated_behavior_snapshot(baseline)
    return {
        "structural": snapshot_structural_violations(project_root),
        "test_contract_violations": test_contract_violations(project_root, baseline),
        "behavior_differences": behavior_snapshot_differences(baseline["expected"], actual),
        "golden_sha256": APPROVED_GOLDEN_SHA256,
        "git_tag_available": _tagged_file_bytes(project_root, GOLDEN_RELATIVE_PATH) is not None,
        "scenario_count": len(actual["portfolio"]),
    }

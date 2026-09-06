"""Pre-live Trading operational audit.

The audit is read-only with respect to Trading account/order state.  It may
write only its own report artifacts under outputs/trading/operational_audit/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.research import RESEARCH_MARKET_DATA_CUTOFF
from core.console_report import project_relative_display_path
from core.dataset_dates import resolve_latest_dataset_date
from core.dataset_profiles import DATASET_PROFILE_FULL, DATASET_PROFILE_REDUCED
from core.file_integrity import atomic_write_json, atomic_write_text
from core.runtime_domains import (
    RUNTIME_DOMAIN_RESEARCH,
    RUNTIME_DOMAIN_TRADING,
    resolve_runtime_domain_paths,
    resolve_runtime_output_dir,
)
from core.runtime_utils import get_taipei_now
from core.trading_capabilities import build_trading_capability_snapshot
from core.trading_market_clock import latest_allowed_completed_daily_date
from services.trading.operations_status import build_trading_operations_status

TRADING_OPERATIONAL_AUDIT_SCHEMA_VERSION = 1
TRADING_OPERATIONAL_AUDIT_STATUS_LIVE_READY = "LIVE_READY"
TRADING_OPERATIONAL_AUDIT_STATUS_LIVE_BLOCKED = "LIVE_BLOCKED"


def resolve_trading_operational_audit_dir(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return Path(resolve_runtime_output_dir(root, domain=RUNTIME_DOMAIN_TRADING, category="operational_audit"))


def _inspect_research_dataset(root: Path, profile: str) -> dict[str, Any]:
    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_RESEARCH, dataset_profile=profile)
    data_dir = Path(paths.data_dir)
    if not data_dir.is_dir() or not any(data_dir.glob("*.csv")):
        return {
            "profile": profile,
            "data_dir": project_relative_display_path(data_dir, project_root=root),
            "available": False,
            "latest_data_date": None,
            "cutoff": str(RESEARCH_MARKET_DATA_CUTOFF),
            "within_cutoff": None,
            "error": None,
        }
    try:
        latest = resolve_latest_dataset_date(data_dir)
    except (OSError, ValueError) as exc:
        return {
            "profile": profile,
            "data_dir": project_relative_display_path(data_dir, project_root=root),
            "available": True,
            "latest_data_date": None,
            "cutoff": str(RESEARCH_MARKET_DATA_CUTOFF),
            "within_cutoff": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    cutoff = pd.Timestamp(RESEARCH_MARKET_DATA_CUTOFF).normalize()
    latest_ts = pd.Timestamp(latest).normalize()
    return {
        "profile": profile,
        "data_dir": project_relative_display_path(data_dir, project_root=root),
        "available": True,
        "latest_data_date": latest,
        "cutoff": str(RESEARCH_MARKET_DATA_CUTOFF),
        "within_cutoff": bool(latest_ts <= cutoff),
        "error": None,
    }


def build_trading_operational_audit(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    capability = build_trading_capability_snapshot()
    operations = build_trading_operations_status(root)
    research_rows = [
        _inspect_research_dataset(root, DATASET_PROFILE_FULL),
        _inspect_research_dataset(root, DATASET_PROFILE_REDUCED),
    ]

    blockers: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    for row in research_rows:
        if row["available"] is False:
            warnings.append(f"Research {row['profile']} dataset 在此環境不可驗證實體 cutoff")
            status = "WARN"
            detail = "dataset unavailable"
        elif row["within_cutoff"] is False:
            blockers.append(
                f"Research {row['profile']} dataset 超過 cutoff：latest={row['latest_data_date'] or '-'} > {row['cutoff']}"
            )
            status = "FAIL"
            detail = row.get("error") or "cutoff exceeded"
        else:
            status = "PASS"
            detail = f"latest={row['latest_data_date']} <= cutoff={row['cutoff']}"
        checks.append({"id": f"research_{row['profile']}_cutoff", "status": status, "detail": detail})

    latest_data_date = operations.get("latest_data_date")
    allowed_date = latest_allowed_completed_daily_date(now=get_taipei_now())
    if latest_data_date:
        completed_ok = pd.Timestamp(str(latest_data_date)).normalize() <= pd.Timestamp(allowed_date).normalize()
        if not completed_ok:
            blockers.append(
                f"Trading data 可能包含尚未完成的日K：latest={latest_data_date}, allowed_completed={allowed_date}"
            )
        checks.append(
            {
                "id": "trading_completed_daily_bar",
                "status": "PASS" if completed_ok else "FAIL",
                "detail": f"latest={latest_data_date}, allowed_completed={allowed_date}",
            }
        )
    else:
        warnings.append("Trading dataset 尚無可驗證最新交易日")
        checks.append({"id": "trading_completed_daily_bar", "status": "WARN", "detail": "no Trading data"})

    for name in list(capability.get("live_blocking_capabilities") or []):
        spec = dict((capability.get("capabilities") or {}).get(name) or {})
        blockers.append(f"缺少 live capability {name}: {spec.get('description') or '-'}")
        checks.append({"id": f"capability_{name}", "status": "FAIL", "detail": spec.get("description")})

    for name, spec in dict(capability.get("capabilities") or {}).items():
        if name in set(capability.get("live_blocking_capabilities") or []):
            continue
        checks.append(
            {
                "id": f"capability_{name}",
                "status": "PASS" if bool(spec.get("implemented")) else "WARN",
                "detail": spec.get("description"),
            }
        )

    account_initialized = bool(operations.get("account_initialized"))
    checks.append({"id":"trading_account_initialized","status":"PASS" if account_initialized else "FAIL","detail":"Trading account canonical state exists" if account_initialized else "Trading account 尚未初始化"})
    if not account_initialized:
        blockers.append("Trading account 尚未初始化")
    cash_configured = account_initialized and operations.get("cash") is not None
    checks.append({"id":"trading_cash_configured","status":"PASS" if cash_configured else "FAIL","detail":"cash configured" if cash_configured else "cash 尚未設定"})
    if account_initialized and not cash_configured:
        blockers.append("Trading cash 尚未設定")

    rollforward_due = int(operations.get("rollforward_due_count") or 0)
    checks.append({"id":"trading_position_rollforward_current","status":"PASS" if rollforward_due == 0 else "FAIL","detail":f"due_count={rollforward_due}"})
    if rollforward_due:
        blockers.append("Trading strategy positions 尚有 daily roll-forward 未完成")

    strategy_position_count = int(operations.get("strategy_position_count") or 0)
    indicator_plan_ready = strategy_position_count == 0 or bool(operations.get("indicator_exit_plan_fresh"))
    checks.append({"id":"trading_indicator_exit_plan_current","status":"PASS" if indicator_plan_ready else "FAIL","detail":f"strategy_positions={strategy_position_count}, fresh={bool(operations.get('indicator_exit_plan_fresh'))}"})
    if not indicator_plan_ready:
        blockers.append("Trading strategy positions 缺少 fresh Indicator SELL plan")

    sell_coverage_safe = bool(operations.get("open_position_sell_coverage_safe"))
    sell_coverage_blockers = [str(x) for x in list(operations.get("open_position_sell_coverage_blockers") or []) if str(x)]
    forced_stop_due = list(operations.get("forced_stop_exit_tickers") or [])
    forced_stop_unsubmitted = list(operations.get("forced_stop_unsubmitted_tickers") or [])
    forced_stop_conflict = list(operations.get("forced_stop_conflict_tickers") or [])
    checks.append({
        "id": "trading_stop_forced_exit_continuity",
        "status": "PASS" if not forced_stop_unsubmitted and not forced_stop_conflict else "FAIL",
        "detail": f"due={forced_stop_due}, unsubmitted={forced_stop_unsubmitted}, conflict={forced_stop_conflict}",
    })
    checks.append({
        "id": "trading_open_position_sell_coverage",
        "status": "PASS" if sell_coverage_safe else "FAIL",
        "detail": "; ".join(sell_coverage_blockers) or "Operations canonical sell coverage is safe",
    })
    if not sell_coverage_safe:
        blockers.append("Trading open positions 尚未具備安全且互斥的實際 SELL coverage: " + "；".join(sell_coverage_blockers))


    if str(operations.get("overall_status") or "") == "BLOCKED":
        for item in list(operations.get("blockers") or []):
            blockers.append(f"Operations Status: {item}")

    for item in list(operations.get("warnings") or []):
        warnings.append(f"Operations Status: {item}")

    status = (
        TRADING_OPERATIONAL_AUDIT_STATUS_LIVE_READY
        if not blockers
        else TRADING_OPERATIONAL_AUDIT_STATUS_LIVE_BLOCKED
    )
    generated_at = get_taipei_now().isoformat(timespec="seconds")
    payload = {
        "schema_version": TRADING_OPERATIONAL_AUDIT_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "status": status,
        "generated_at": generated_at,
        "research_cutoff": str(RESEARCH_MARKET_DATA_CUTOFF),
        "latest_allowed_completed_daily_date": allowed_date,
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
        "capabilities": capability,
        "operations_status": operations,
        "research_datasets": research_rows,
    }
    return payload


def _render_operational_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Trading Pre-live Operational Audit",
        "",
        f"- Status: **{audit['status']}**",
        f"- Generated at: `{audit['generated_at']}`",
        f"- Research cutoff: `{audit['research_cutoff']}`",
        f"- Latest allowed completed daily date: `{audit['latest_allowed_completed_daily_date']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = list(audit.get("blockers") or [])
    lines.extend([f"- {item}" for item in blockers] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    warnings = list(audit.get("warnings") or [])
    lines.extend([f"- {item}" for item in warnings] or ["- None"])
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for row in list(audit.get("checks") or []):
        detail = str(row.get("detail") or "-").replace("|", "\\|")
        lines.append(f"| `{row.get('id')}` | {row.get('status')} | {detail} |")
    lines.extend(
        [
            "",
            "## Current Operations Status",
            "",
            f"- Overall: `{audit['operations_status'].get('overall_status')}`",
            f"- Next action: `{audit['operations_status'].get('next_action_code')}` - {audit['operations_status'].get('next_action_label')}",
            "",
            "> LIVE_READY only means required Trading capabilities and the current operational safety state pass this audit. "
            "It never infers broker submission/fills or bypasses explicit user confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_trading_operational_audit(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    audit = build_trading_operational_audit(root)
    output_dir = resolve_trading_operational_audit_dir(root)
    json_path = output_dir / "prelive_audit.json"
    markdown_path = output_dir / "prelive_audit.md"
    atomic_write_json(json_path, audit)
    atomic_write_text(markdown_path, _render_operational_audit_markdown(audit))
    return {
        **audit,
        "json_path": project_relative_display_path(json_path, project_root=root),
        "markdown_path": project_relative_display_path(markdown_path, project_root=root),
    }


__all__ = [
    "TRADING_OPERATIONAL_AUDIT_SCHEMA_VERSION",
    "TRADING_OPERATIONAL_AUDIT_STATUS_LIVE_READY",
    "TRADING_OPERATIONAL_AUDIT_STATUS_LIVE_BLOCKED",
    "resolve_trading_operational_audit_dir",
    "build_trading_operational_audit",
    "run_trading_operational_audit",
]

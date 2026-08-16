from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic_breakout_quality_support import add_check, tempfile, Path


def _planned_risk_alignment_synthetic():
    from tools.audit.breakout_quality.planned_risk_40d_alignment import analyze_alignment_frames

    entry = "2020-01-02"
    dates = pd.bdate_range(entry, periods=50)
    executions = []
    trades = []
    targets = []
    market_frames = {}
    for index in range(20):
        ticker = f"T{index:02d}"
        realized_r = 0.5 + index * 0.12
        stop_gap = 10.0 - index * (8.0 / 19.0)
        stop_price = 100.0 - stop_gap
        highs = np.full(len(dates), 100.0)
        lows = np.full(len(dates), 99.5)
        highs[1:41] = 100.0 + realized_r * stop_gap
        market_frames[ticker] = pd.DataFrame(
            {
                "Open": 100.0,
                "High": highs,
                "Low": lows,
                "Close": 100.0,
                "Volume": 1000,
            },
            index=dates,
        )
        executions.append(
            {
                "ticker": ticker,
                "trade_date": entry,
                "signal_date": "2020-01-01",
                "entry_type": "normal",
                "entry_filled": True,
                "limit_px": 100.0,
                "init_sl": stop_price,
            }
        )
        trades.extend(
            (
                {
                    "Date": entry,
                    "Ticker": ticker,
                    "Type": "買進 (一般)",
                    "進場類型": "normal",
                    "候選類型": "normal",
                    "買訊日": "2020-01-01",
                    "候選日": entry,
                    "成交價": 100.0,
                },
                {
                    "Date": "2020-03-02",
                    "Ticker": ticker,
                    "Type": "全倉結算",
                    "成交價": 100.0,
                    "該筆總損益": 0.0,
                    "R_Multiple": realized_r,
                },
            )
        )
        targets.append(
            {
                "ticker": ticker,
                "trade_date": entry,
                "signal_date": "2020-01-01",
                "target_raw_r": float((index % 5) - 2),
                "target_available": True,
            }
        )
    return analyze_alignment_frames(
        executions=pd.DataFrame(executions),
        trades=pd.DataFrame(trades),
        selected_targets=pd.DataFrame(targets),
        market_frames=market_frames,
        horizon_bars=40,
        fixed_risk_budget_return=0.10,
        capital_bucket_count=5,
    )


def validate_breakout_quality_audit_framework_contract_case(_base_params):
    from config.audit import (
        AUDIT_OUTPUT_ROOT,
        get_audit_definitions,
        get_audit_module_ids,
        get_enabled_audit_definitions,
        validate_audit_config,
    )
    from tools.audit.catalog import get_audit_entry, validate_audit_catalog
    from tools.audit.runner import collect_audit_status

    case_id = "AUDIT_FRAMEWORK"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    validate_audit_config()
    definitions = get_audit_definitions("breakout_quality")
    enabled = get_enabled_audit_definitions("breakout_quality")
    validate_audit_catalog(definitions)

    expected_id = "min-roos-planned-risk-40d-alignment"
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "formal_audit_config_contains_only_current_planned_risk_alignment_gate",
        True,
        bool(
            get_audit_module_ids() == ("breakout_quality",)
            and len(definitions) == 1
            and len(enabled) == 1
            and enabled[0].audit_id == expected_id
            and enabled[0].audit_type == "planned_risk_40d_alignment"
            and AUDIT_OUTPUT_ROOT == "outputs/audit"
        ),
    )
    entry = get_audit_entry("planned_risk_40d_alignment")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "planned_risk_alignment_audit_is_formal_read_only_handler",
        (True, True, True),
        (entry.formal, entry.read_only, bool(entry.status_function and entry.run_function)),
    )

    with tempfile.TemporaryDirectory() as tmp:
        status_payload = collect_audit_status(
            module_id="breakout_quality",
            project_root=Path(tmp),
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "formal_audit_runner_blocks_when_completed_pair_or_dataset_is_missing",
        ("BLOCKED", 1, "BLOCKED"),
        (
            status_payload.get("overall_status"),
            len(status_payload.get("rows") or []),
            (status_payload.get("rows") or [{}])[0].get("status"),
        ),
    )

    detail, alignment = _planned_risk_alignment_synthetic()
    planned = alignment["metrics"]["planned_risk_40d"]
    fixed = alignment["metrics"]["entry_fixed_risk_40d"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "planned_risk_40d_uses_same_entry_date_control_and_can_recover_risk_normalized_ordering",
        True,
        bool(
            len(detail) == 20
            and alignment.get("decision") == "GO"
            and float(planned["spearman"]) > float(fixed["spearman"])
            and float(planned["top_bottom_realized_spread_r"])
            > float(fixed["top_bottom_realized_spread_r"])
            and alignment.get("path_contract")
            == "40 trading bars after actual entry date; same-bar risk breach adverse-first; entry day excluded"
        ),
    )

    summary["workflow"] = "planned_risk_40d_alignment_gate"
    return results, summary

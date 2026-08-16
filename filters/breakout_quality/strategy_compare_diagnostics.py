"""Post-replay Strategy Compare candidate/selection diagnostics."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from core.console_report import project_relative_display_path
from core.display_common import _display_width
from core.report_metrics import (
    R_ANALYSIS_GROUPED_SECTIONS,
    R_ANALYSIS_MERGED_METRICS,
    RAnalysisMetricSpec,
)
from core.report_style import best_worst_signals, styled_signal
from core.strategy_comparison import StrategyComparisonSettings

from core.exact_accounting import (
    calc_planned_initial_risk_from_prices_milli,
    milli_to_money,
)
from core.order_lot_policy import apply_board_lot_preferred_qty
from core.price_utils import calc_position_size

from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_selection_point_in_time_score_table,
    load_selection_point_in_time_score_table_from_path,
)
from filters.breakout_quality.strategy_compare_contracts import comparison_labels


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _exact_planned_initial_risk(
    *,
    limit_price: float,
    stop_price: float,
    qty: int,
    params,
    ticker: str,
    security_profile,
    trade_date: str,
) -> float | None:
    if qty <= 0:
        return 0.0
    try:
        risk_milli = calc_planned_initial_risk_from_prices_milli(
            limit_price,
            stop_price,
            int(qty),
            params,
            ticker=ticker,
            security_profile=security_profile,
            trade_date=trade_date,
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    return float(milli_to_money(risk_milli))


def _flatten_entry_execution_rows(replay_execution_rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    columns = [
        "execution_order", "ticker", "trade_date", "candidate_date", "signal_date", "entry_type",
        "candidate_qty", "chosen_qty", "filled_qty", "entry_filled",
        "limit_px", "init_sl", "entry_fill_price",
        "sizing_equity", "candidate_sizing_capital", "effective_entry_budget",
        "available_cash_before", "reserved_cost", "fixed_risk",
        "max_position_cap_pct", "max_qty", "core_qty", "qty_without_risk_cap",
        "qty_without_position_cap", "qty_without_risk_or_position_cap",
        "qty_after_max_qty_before_lot", "lot_rounding_qty_loss",
        "candidate_initial_risk", "chosen_initial_risk", "actual_initial_risk", "risk_budget",
        "candidate_risk_utilization", "chosen_risk_utilization", "actual_risk_utilization",
        "risk_cap_binding", "position_cap_binding", "risk_position_tie",
        "capital_binding", "max_qty_binding", "lot_rounding_binding",
        "entry_budget_cash_binding", "binding_signature",
    ]
    rows: list[dict[str, Any]] = []
    for raw in list(replay_execution_rows or []):
        item = dict(raw or {})
        if str(item.get("_event_type") or "") != "entry_execution":
            continue
        params = item.get("params_obj")
        if params is None:
            continue
        ticker = str(item.get("ticker") or "").strip()
        trade_date = str(item.get("trade_date") or "")
        limit_px = _finite_float(item.get("limit_px"))
        init_sl = _finite_float(item.get("init_sl"))
        sizing_capital = _finite_float(item.get("candidate_sizing_capital"))
        if sizing_capital is None:
            sizing_capital = _finite_float(item.get("sizing_equity"))
        candidate_qty = int(item.get("candidate_qty", 0) or 0)
        chosen_qty = int(item.get("chosen_qty", 0) or 0)
        filled_qty = int(item.get("filled_qty", 0) or 0)
        max_qty_raw = item.get("max_qty")
        try:
            max_qty = int(max_qty_raw) if max_qty_raw not in (None, "") else None
        except (TypeError, ValueError):
            max_qty = None

        core_qty = no_risk_qty = no_position_qty = no_both_qty = 0
        qty_after_max = 0
        lot_loss = 0
        risk_binding = position_binding = risk_position_tie = capital_binding = False
        max_qty_binding = lot_binding = cash_binding = False
        candidate_initial_risk = chosen_initial_risk = actual_initial_risk = risk_budget = None
        candidate_utilization = chosen_utilization = actual_utilization = None
        fixed_risk = _finite_float(getattr(params, "fixed_risk", None))
        max_position_cap_pct = _finite_float(getattr(params, "max_position_cap_pct", None))

        if limit_px is not None and init_sl is not None and sizing_capital is not None and sizing_capital > 0.0:
            core_qty = int(calc_position_size(
                limit_px, init_sl, sizing_capital, float(params.fixed_risk), params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            ))
            no_risk_qty = int(calc_position_size(
                limit_px, init_sl, sizing_capital, 1.0, params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            ))
            no_position_params = copy.copy(params)
            no_position_params.max_position_cap_pct = 1.0
            no_position_qty = int(calc_position_size(
                limit_px, init_sl, sizing_capital, float(params.fixed_risk), no_position_params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            ))
            no_both_qty = int(calc_position_size(
                limit_px, init_sl, sizing_capital, 1.0, no_position_params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            ))
            risk_gain = no_risk_qty - core_qty
            position_gain = no_position_qty - core_qty
            both_gain = no_both_qty - core_qty
            risk_binding = risk_gain > 0 and position_gain <= 0
            position_binding = position_gain > 0 and risk_gain <= 0
            risk_position_tie = risk_gain <= 0 and position_gain <= 0 and both_gain > 0
            capital_binding = both_gain <= 0
            qty_after_max = min(core_qty, max_qty) if max_qty is not None and max_qty > 0 else core_qty
            max_qty_binding = bool(max_qty is not None and max_qty > 0 and qty_after_max < core_qty)
            lot_qty = int(apply_board_lot_preferred_qty(limit_px, qty_after_max, params)) if qty_after_max > 0 else 0
            lot_loss = max(0, int(qty_after_max) - int(lot_qty))
            lot_binding = lot_loss > 0
            candidate_initial_risk = _exact_planned_initial_risk(
                limit_price=limit_px, stop_price=init_sl, qty=candidate_qty, params=params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            )
            chosen_initial_risk = _exact_planned_initial_risk(
                limit_price=limit_px, stop_price=init_sl, qty=chosen_qty, params=params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            )
            risk_budget = float(sizing_capital * float(params.fixed_risk))
            if risk_budget > 0.0 and candidate_initial_risk is not None:
                candidate_utilization = float(candidate_initial_risk / risk_budget)
            if risk_budget > 0.0 and chosen_initial_risk is not None:
                chosen_utilization = float(chosen_initial_risk / risk_budget)

        actual_risk_milli = int(item.get("actual_initial_risk_total_milli", 0) or 0)
        if actual_risk_milli > 0:
            actual_initial_risk = float(milli_to_money(actual_risk_milli))
            if risk_budget is not None and risk_budget > 0.0:
                actual_utilization = float(actual_initial_risk / risk_budget)
        cash_binding = chosen_qty > 0 and candidate_qty > 0 and chosen_qty < candidate_qty

        signature: list[str] = []
        if risk_binding:
            signature.append("RISK_CAP")
        if position_binding:
            signature.append("POSITION_CAP")
        if risk_position_tie:
            signature.append("RISK_POSITION_TIE")
        if capital_binding:
            signature.append("CAPITAL")
        if max_qty_binding:
            signature.append("MAX_QTY")
        if lot_binding:
            signature.append("LOT_ROUNDING")
        if cash_binding:
            signature.append("ENTRY_BUDGET_CASH")
        if not signature:
            signature.append("NONE_DETECTED")

        rows.append({
            "execution_order": int(item.get("execution_order", len(rows)) or 0),
            "ticker": ticker,
            "trade_date": trade_date,
            "candidate_date": str(item.get("candidate_date") or ""),
            "signal_date": str(item.get("signal_date") or ""),
            "entry_type": str(item.get("type") or "normal"),
            "candidate_qty": candidate_qty,
            "chosen_qty": chosen_qty,
            "filled_qty": filled_qty,
            "entry_filled": bool(item.get("entry_filled", False)),
            "limit_px": limit_px,
            "init_sl": init_sl,
            "entry_fill_price": _finite_float(item.get("entry_fill_price")),
            "sizing_equity": _finite_float(item.get("sizing_equity")),
            "candidate_sizing_capital": sizing_capital,
            "effective_entry_budget": _finite_float(item.get("effective_entry_budget")),
            "available_cash_before": _finite_float(item.get("available_cash_before")),
            "reserved_cost": _finite_float(item.get("reserved_cost")),
            "fixed_risk": fixed_risk,
            "max_position_cap_pct": max_position_cap_pct,
            "max_qty": max_qty,
            "core_qty": core_qty,
            "qty_without_risk_cap": no_risk_qty,
            "qty_without_position_cap": no_position_qty,
            "qty_without_risk_or_position_cap": no_both_qty,
            "qty_after_max_qty_before_lot": qty_after_max,
            "lot_rounding_qty_loss": lot_loss,
            "candidate_initial_risk": candidate_initial_risk,
            "chosen_initial_risk": chosen_initial_risk,
            "actual_initial_risk": actual_initial_risk,
            "risk_budget": risk_budget,
            "candidate_risk_utilization": candidate_utilization,
            "chosen_risk_utilization": chosen_utilization,
            "actual_risk_utilization": actual_utilization,
            "risk_cap_binding": risk_binding,
            "position_cap_binding": position_binding,
            "risk_position_tie": risk_position_tie,
            "capital_binding": capital_binding,
            "max_qty_binding": max_qty_binding,
            "lot_rounding_binding": lot_binding,
            "entry_budget_cash_binding": cash_binding,
            "binding_signature": "+".join(signature),
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame[columns].sort_values(
        ["execution_order"], kind="mergesort"
    ).reset_index(drop=True)

def _flatten_candidate_replay_rows(replay_counts: dict[str, dict[str, Any]], field: str) -> pd.DataFrame:
    if field not in {"candidate_rows", "orderable_rows"}:
        raise ValueError(f"不支援的candidate replay field: {field}")
    rows: list[dict[str, Any]] = []
    for ticker in sorted(replay_counts):
        bucket = replay_counts.get(ticker) or {}
        for raw in list(bucket.get(field) or []):
            row = dict(raw or {})
            row["ticker"] = str(row.get("ticker") or ticker)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "trade_date", "candidate_date", "signal_date",
            "candidate_type", "entry_source", "is_orderable", "high_len",
            "ensemble_vote_count", "qty", "sort_value", "historical_ev",
            "historical_win_rate", "historical_trade_count",
            "breakout_quality_score", "breakout_quality_score_date",
        ])
    frame = pd.DataFrame(rows)
    for column in ("trade_date", "candidate_date", "signal_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return frame.sort_values(
        ["trade_date", "ticker", "signal_date", "candidate_type"],
        kind="mergesort",
    ).reset_index(drop=True)

def _flatten_selector_trace_rows(replay_selector_trace_rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Serialize transient max-DL selector stage membership without runtime objects."""

    columns = [
        "stage", "stage_rank", "ticker", "trade_date", "candidate_date", "signal_date",
        "candidate_type", "entry_source", "breakout_quality_score",
        "breakout_quality_score_date", "breakout_quality_score_available",
        "breakout_quality_expected_r_available", "breakout_quality_expected_r",
        "breakout_quality_daily_score_percentile", "breakout_quality_expected_r_calibration_cutoff",
        "breakout_quality_expected_excess_r_available", "breakout_quality_expected_excess_r",
        "breakout_quality_expected_excess_r_calibration_cutoff",
        "pre_market_order_limit", "direct_score_order_feasible", "repair_steps", "ascent_steps",
    ]
    rows: list[dict[str, Any]] = []
    for raw in list(replay_selector_trace_rows or []):
        item = dict(raw or {})
        if str(item.get("trace_kind") or "basket_stage") != "basket_stage":
            continue
        rows.append({
            "stage": str(item.get("stage") or ""),
            "stage_rank": int(item.get("stage_rank", 0) or 0),
            "ticker": str(item.get("ticker") or ""),
            "trade_date": str(item.get("trade_date") or ""),
            "candidate_date": str(item.get("candidate_date") or ""),
            "signal_date": str(item.get("signal_date") or ""),
            "candidate_type": str(item.get("candidate_type") or ""),
            "entry_source": str(item.get("entry_source") or ""),
            "breakout_quality_score": _finite_float(item.get("breakout_quality_score")),
            "breakout_quality_score_date": str(item.get("breakout_quality_score_date") or ""),
            "breakout_quality_score_available": bool(item.get("breakout_quality_score_available", False)),
            "breakout_quality_expected_r_available": bool(
                item.get("breakout_quality_expected_r_available", False)
            ),
            "breakout_quality_expected_r": _finite_float(item.get("breakout_quality_expected_r")),
            "breakout_quality_daily_score_percentile": _finite_float(
                item.get("breakout_quality_daily_score_percentile")
            ),
            "breakout_quality_expected_r_calibration_cutoff": str(
                item.get("breakout_quality_expected_r_calibration_cutoff") or ""
            ),
            "breakout_quality_expected_excess_r_available": bool(
                item.get("breakout_quality_expected_excess_r_available", False)
            ),
            "breakout_quality_expected_excess_r": _finite_float(
                item.get("breakout_quality_expected_excess_r")
            ),
            "breakout_quality_expected_excess_r_calibration_cutoff": str(
                item.get("breakout_quality_expected_excess_r_calibration_cutoff") or ""
            ),
            "pre_market_order_limit": (
                None if item.get("pre_market_order_limit") in (None, "")
                else int(item.get("pre_market_order_limit"))
            ),
            "direct_score_order_feasible": bool(item.get("direct_score_order_feasible", False)),
            "repair_steps": int(item.get("repair_steps", 0) or 0),
            "ascent_steps": int(item.get("ascent_steps", 0) or 0),
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame[columns].sort_values(
        ["trade_date", "stage", "stage_rank", "ticker"], kind="mergesort"
    ).reset_index(drop=True)


def _flatten_repair_mechanism_rows(
    replay_selector_trace_rows: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """Serialize scalable repair-search certificates from production selector trace."""

    summary_columns = [
        "trace_kind", "trade_date", "status", "classification",
        "target_count", "candidate_count", "reserve_floor_milli",
        "raw_selected_count", "raw_reserved_cost_milli", "raw_count_deficit",
        "raw_reserve_deficit_milli", "actual_repair_steps",
        "actual_repair_replacement_distance", "raw_score_sum",
        "repair_seed_score_sum", "final_score_sum",
        "repair_seed_score_loss_from_raw", "final_score_change_from_repair",
        "repair_search_evaluations", "ascent_search_evaluations",
        "ascent_local_optimum",
    ]
    candidate_columns = [
        "repair_role", "repair_step", "ticker", "candidate_date",
        "signal_date", "candidate_type", "entry_source", "breakout_quality_score",
        "breakout_quality_score_date", "before_selected_count", "after_selected_count",
        "before_reserved_cost_milli", "after_reserved_cost_milli",
        "before_count_deficit", "after_count_deficit",
        "before_reserve_deficit_milli", "after_reserve_deficit_milli", "after_feasible",
        "evaluated_swap_count", "progress_swap_count", "feasible_swap_count",
    ]
    columns = summary_columns + candidate_columns
    rows = []
    for raw in list(replay_selector_trace_rows or []):
        item = dict(raw or {})
        trace_kind = str(item.get("trace_kind") or "")
        if trace_kind not in {"repair_summary", "repair_swap"}:
            continue
        row = {column: None for column in columns}
        row.update({
            "trace_kind": trace_kind,
            "trade_date": str(item.get("trade_date") or ""),
            "status": str(item.get("status") or ""),
            "classification": str(item.get("classification") or ""),
            "repair_role": str(item.get("repair_role") or ""),
            "ticker": str(item.get("ticker") or ""),
            "candidate_date": str(item.get("candidate_date") or ""),
            "signal_date": str(item.get("signal_date") or ""),
            "candidate_type": str(item.get("candidate_type") or ""),
            "entry_source": str(item.get("entry_source") or ""),
            "breakout_quality_score": _finite_float(item.get("breakout_quality_score")),
            "breakout_quality_score_date": str(item.get("breakout_quality_score_date") or ""),
            "ascent_local_optimum": bool(item.get("ascent_local_optimum", False)) if trace_kind == "repair_summary" else None,
        })
        int_columns = (
            "target_count", "candidate_count", "reserve_floor_milli", "raw_selected_count",
            "raw_reserved_cost_milli", "raw_count_deficit", "raw_reserve_deficit_milli",
            "actual_repair_steps", "actual_repair_replacement_distance",
            "repair_search_evaluations", "ascent_search_evaluations",
            "repair_step", "before_selected_count", "after_selected_count",
            "before_reserved_cost_milli", "after_reserved_cost_milli",
            "before_count_deficit", "after_count_deficit",
            "before_reserve_deficit_milli", "after_reserve_deficit_milli",
            "evaluated_swap_count", "progress_swap_count", "feasible_swap_count",
        )
        for column in int_columns:
            value = item.get(column)
            row[column] = None if value in (None, "") else int(value)
        for column in (
            "raw_score_sum", "repair_seed_score_sum", "final_score_sum",
            "repair_seed_score_loss_from_raw", "final_score_change_from_repair",
        ):
            row[column] = _finite_float(item.get(column))
        if trace_kind == "repair_swap":
            row["after_feasible"] = bool(item.get("after_feasible", False))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame[columns].sort_values(
        ["trade_date", "trace_kind", "repair_step", "repair_role", "ticker"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _flatten_selected_buy_rows(trade_history: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(trade_history).copy()
    columns = ["ticker", "trade_date", "signal_date", "type"]
    if frame.empty or not {"Date", "Ticker", "Type"}.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    buy_mask = frame["Type"].fillna("").astype(str).str.startswith("買進 (")
    out = frame.loc[buy_mask].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame({
        "ticker": out["Ticker"].fillna("").astype(str).str.strip(),
        "trade_date": pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "signal_date": pd.to_datetime(
            out.get("買訊日", pd.Series("", index=out.index)), errors="coerce"
        ).dt.strftime("%Y-%m-%d"),
        "type": out["Type"].fillna("").astype(str),
    })
    return out.dropna(subset=["trade_date"]).sort_values(
        ["trade_date", "ticker", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)

def _selection_target_lookup(
    *, root: Path, filter_id: str, architecture: str, profile: str,
    score_path_override: str | None = None, manifest_path_override: str | None = None,
) -> pd.DataFrame:
    scores = (
        load_selection_point_in_time_score_table_from_path(
            str(score_path_override), manifest_path=str(manifest_path_override)
        )
        if score_path_override not in (None, "")
        else load_selection_point_in_time_score_table(
            str(root), filter_id, architecture, profile
        )
    ).reset_index()
    bundle = load_profile_continuous_ranker_data(
        filter_id=filter_id,
        model_architecture=architecture,
        experiment_profile=profile,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    groups = bundle.group_table[["group_index", "ticker", "date", "label"]].copy()
    groups["ticker"] = groups["ticker"].fillna("").astype(str).str.strip()
    groups["date"] = pd.to_datetime(groups["date"], errors="raise").dt.strftime("%Y-%m-%d")
    groups["target_raw_r"] = bundle.raw_target
    groups["target_available"] = bundle.target_valid & pd.Series(bundle.raw_target).map(math.isfinite).to_numpy()
    joined = scores.merge(groups, on="group_index", how="left", validate="one_to_one", suffixes=("", "_dataset"))
    mismatch = (
        joined["ticker"] != joined["ticker_dataset"]
    ) | (
        joined["date"] != joined["date_dataset"]
    )
    if bool(mismatch.any()):
        raise ValueError("Selection PIT Score與Dataset group identity不一致")
    return joined[[
        "ticker", "date", "group_index", "breakout_quality_score", "fold_id",
        "model_information_cutoff", "label", "target_raw_r", "target_available",
    ]].rename(columns={"date": "signal_date"})

def _load_isolated_selection_pit_contract(
    *, score_path: str, manifest_path: str, filter_id: str, model_architecture: str,
    experiment_profile: str, expected_seed: int | None,
):
    score = Path(str(score_path)).resolve()
    manifest_file = Path(str(manifest_path)).resolve()
    for label, path in (("Selection PIT score", score), ("Selection PIT manifest", manifest_file)):
        if not path.is_file():
            raise FileNotFoundError(f"找不到isolated {label}: {path}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("isolated Selection PIT manifest根節點必須是object")
    if str(manifest.get("status") or "") != "BUILT":
        raise ValueError(f"isolated Selection PIT manifest尚未完成: {manifest.get('status')!r}")
    expected_identity = {
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
    }
    for field, expected in expected_identity.items():
        actual = str(manifest.get(field) or "")
        if actual != expected:
            raise ValueError(
                f"isolated Selection PIT identity不一致: field={field}, expected={expected}, actual={actual}"
            )
    if expected_seed is not None and int(manifest.get("seed", -1)) != int(expected_seed):
        raise ValueError(
            "isolated Selection PIT seed不一致: "
            f"expected={int(expected_seed)}, actual={manifest.get('seed')!r}"
        )
    lookahead = dict(manifest.get("lookahead_contract") or {})
    required_true = (
        "every_score_uses_model_not_trained_on_scored_event",
        "training_requires_label_eval_end_before_score_start",
    )
    required_false = (
        "oos_rows_or_target_statistics_used_for_training_or_epoch_selection",
        "future_target_in_score_table",
    )
    if any(lookahead.get(key) is not True for key in required_true) or any(
        lookahead.get(key) is not False for key in required_false
    ):
        raise ValueError("isolated Selection PIT lookahead contract不符合策略回放要求")
    score_record = dict((manifest.get("artifacts") or {}).get("scores") or {})
    if str(score_record.get("filename") or "") != score.name:
        raise ValueError("isolated Selection PIT score filename與manifest不一致")
    expected_hash = str(score_record.get("sha256") or "").lower()
    actual_hash = compute_file_sha256(score).lower()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("isolated Selection PIT score SHA256與manifest不一致")
    expected_size = int(score_record.get("size_bytes", -1))
    if expected_size != int(score.stat().st_size):
        raise ValueError("isolated Selection PIT score size與manifest不一致")
    table = load_selection_point_in_time_score_table_from_path(
        str(score), manifest_path=str(manifest_file)
    )
    available_from = str(table.attrs.get("available_from") or "")
    available_through = str(table.attrs.get("available_through") or "")
    if not available_from or not available_through or available_through < available_from:
        raise ValueError("isolated Selection PIT score period不合法")
    return SimpleNamespace(
        score_path=score,
        manifest_path=manifest_file,
        audit_path=None,
        manifest=manifest,
        model_architecture=str(model_architecture),
        experiment_profile=str(experiment_profile),
        seed=int(manifest.get("seed", 0) or 0),
        available_from=available_from,
        available_through=available_through,
        model_validation_gate={
            "status": "ISOLATED_MULTI_SEED_ROBUSTNESS",
            "strategy_metrics_used": False,
            "future_target_used_for_runtime_sort": False,
        },
    )

def _strategy_selection_diagnostics(
    *, orderable: pd.DataFrame, selected: pd.DataFrame, lookup: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    orderable_work = pd.DataFrame(orderable).copy()
    selected_work = pd.DataFrame(selected).copy()
    lookup_work = pd.DataFrame(lookup).copy()

    for frame, columns in (
        (orderable_work, ("trade_date", "signal_date")),
        (selected_work, ("trade_date", "signal_date")),
        (lookup_work, ("signal_date",)),
    ):
        for column in columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(
                    frame[column], errors="coerce"
                ).dt.strftime("%Y-%m-%d").fillna("")

    # Continuation／re-entry 的交易 signal_date 可以晚於目前模型資訊日；
    # runtime Score 與 Future Target 都必須以replay保存的 score_date 對回 PIT 工件。
    raw_score_dates = orderable_work.get(
        "breakout_quality_score_date",
        pd.Series("", index=orderable_work.index, dtype="object"),
    ).fillna("").astype(str).str.strip()
    parsed_score_dates = pd.to_datetime(raw_score_dates, errors="coerce")
    invalid_score_dates = raw_score_dates.ne("") & parsed_score_dates.isna()
    if bool(invalid_score_dates.any()):
        bad = orderable_work.loc[invalid_score_dates].iloc[0]
        bad_score_date = raw_score_dates.loc[invalid_score_dates].iloc[0]
        raise ValueError(
            "策略replay保存的Breakout Quality score_date無法解析: "
            f"ticker={bad.get('ticker')}, trade_date={bad.get('trade_date')}, "
            f"signal_date={bad.get('signal_date')}, score_date={bad_score_date!r}"
        )
    normalized_score_dates = parsed_score_dates.dt.strftime("%Y-%m-%d").fillna("")
    orderable_work["score_event_date"] = normalized_score_dates.where(
        normalized_score_dates.ne(""), orderable_work.get("signal_date", "")
    )

    lookup_work = lookup_work.rename(columns={
        "signal_date": "score_event_date",
        "breakout_quality_score": "pit_breakout_quality_score",
    })
    if bool(lookup_work.duplicated(["ticker", "score_event_date"]).any()):
        raise ValueError("Selection PIT diagnostic lookup同一ticker/score_event_date不唯一")

    orderable_joined = orderable_work.merge(
        lookup_work,
        on=["ticker", "score_event_date"],
        how="left",
        validate="many_to_one",
    )
    score_available = pd.to_numeric(
        orderable_joined.get(
            "pit_breakout_quality_score",
            pd.Series(float("nan"), index=orderable_joined.index),
        ),
        errors="coerce",
    ).map(math.isfinite)
    runtime_score_available = pd.to_numeric(
        orderable_joined.get(
            "breakout_quality_score",
            pd.Series(float("nan"), index=orderable_joined.index),
        ),
        errors="coerce",
    ).map(math.isfinite)
    declared_runtime_available = orderable_joined.get(
        "breakout_quality_score_available",
        pd.Series(False, index=orderable_joined.index),
    ).fillna(False).astype(bool)
    runtime_rows = declared_runtime_available | runtime_score_available
    if bool(runtime_rows.any()):
        expected = pd.to_numeric(
            orderable_joined.loc[runtime_rows, "pit_breakout_quality_score"],
            errors="coerce",
        )
        actual = pd.to_numeric(
            orderable_joined.loc[runtime_rows, "breakout_quality_score"],
            errors="coerce",
        )
        mismatch = (
            (~expected.map(math.isfinite))
            | (~actual.map(math.isfinite))
            | ((actual - expected).abs() > 1e-12)
        )
        if bool(mismatch.any()):
            bad_index = mismatch[mismatch].index[0]
            bad = orderable_joined.loc[bad_index]
            raise ValueError(
                "策略replay使用的Breakout Quality Score與PIT score table不一致: "
                f"ticker={bad.get('ticker')}, trade_date={bad.get('trade_date')}, "
                f"signal_date={bad.get('signal_date')}, "
                f"score_event_date={bad.get('score_event_date')}, "
                f"runtime_score={actual.loc[bad_index]}, pit_score={expected.loc[bad_index]}"
            )

    target_available = orderable_joined.get(
        "target_available", pd.Series(False, index=orderable_joined.index)
    ).fillna(False).astype(bool)

    occurrence_keys = ["ticker", "trade_date", "signal_date"]
    occurrence_columns = [
        *occurrence_keys,
        "score_event_date",
        "target_raw_r",
        "target_available",
        "pit_breakout_quality_score",
    ]
    occurrence_score_event_counts = orderable_joined.groupby(
        occurrence_keys, dropna=False
    )["score_event_date"].nunique(dropna=False)
    if bool((occurrence_score_event_counts > 1).any()):
        bad_key = occurrence_score_event_counts[occurrence_score_event_counts > 1].index[0]
        raise ValueError(
            "同一策略候選發生多個Breakout Quality score event date: "
            f"ticker={bad_key[0]}, trade_date={bad_key[1]}, signal_date={bad_key[2]}"
        )
    occurrence_lookup = orderable_joined[occurrence_columns].drop_duplicates(
        occurrence_keys, keep="first"
    )
    selected_joined = selected_work.merge(
        occurrence_lookup,
        on=occurrence_keys,
        how="left",
        validate="many_to_one",
    )

    day_rows = []
    valid_orderable = orderable_joined.loc[target_available].copy()
    for trade_date, day in valid_orderable.groupby("trade_date", sort=True):
        chosen = selected_joined[selected_joined["trade_date"] == trade_date]
        chosen_keys = set(zip(chosen["ticker"], chosen["signal_date"]))
        if not chosen_keys:
            continue
        day = day.drop_duplicates(["ticker", "signal_date"], keep="first").copy()
        day["target_percentile"] = day["target_raw_r"].rank(method="average", pct=True)
        selected_day = day[[
            (ticker, signal_date) in chosen_keys
            for ticker, signal_date in zip(day["ticker"], day["signal_date"])
        ]]
        if selected_day.empty:
            continue
        k = len(selected_day)
        top = day.nlargest(k, "target_raw_r", keep="first")
        top_keys = set(zip(top["ticker"], top["signal_date"]))
        retained = sum(key in top_keys for key in chosen_keys)
        day_rows.append({
            "trade_date": trade_date,
            "selected_count": k,
            "selected_target_mean_r": float(selected_day["target_raw_r"].mean()),
            "selected_target_percentile_mean": float(selected_day["target_percentile"].mean()),
            "target_top_k_retention": float(retained / k),
            "target_opportunity_gap_r": float(
                top["target_raw_r"].mean() - selected_day["target_raw_r"].mean()
            ),
        })
    daily = pd.DataFrame(day_rows)
    metrics = {
        "orderable_occurrences": int(len(orderable_joined)),
        "orderable_score_covered": int(score_available.sum()),
        "orderable_score_coverage_rate": float(score_available.mean()) if len(score_available) else None,
        "runtime_scored_orderable_occurrences": int(runtime_score_available.sum()),
        "runtime_score_identity_match": True,
        "orderable_target_covered": int(target_available.sum()),
        "orderable_target_coverage_rate": float(target_available.mean()) if len(target_available) else None,
        "selected_buy_rows": int(len(selected_joined)),
        "diagnostic_days": int(len(daily)),
        "selected_target_percentile_mean": float(daily["selected_target_percentile_mean"].mean()) if not daily.empty else None,
        "target_top_k_retention_mean": float(daily["target_top_k_retention"].mean()) if not daily.empty else None,
        "target_opportunity_gap_r_mean": float(daily["target_opportunity_gap_r"].mean()) if not daily.empty else None,
        "selected_target_mean_r": float(daily["selected_target_mean_r"].mean()) if not daily.empty else None,
        "future_target_used_for_runtime_sort": False,
    }
    return metrics, orderable_joined, selected_joined


def paired_target_selection_delta_r(
    baseline_selected: pd.DataFrame,
    active_selected: pd.DataFrame,
) -> dict[str, Any]:
    """Compare paired exclusive selections using the existing post-replay Future Target.

    The comparison basis mirrors direct-selection attribution conceptually: only
    selections unique to the active arm versus selections unique to its same-param
    DL-off baseline contribute.  This remains a post-replay diagnostic and never
    feeds runtime selection, sizing, cash allocation, or execution.
    """

    keys = ["ticker", "trade_date", "signal_date"]
    required = [*keys, "target_raw_r", "target_available"]
    frames = []
    for label, raw in (("baseline", baseline_selected), ("active", active_selected)):
        frame = pd.DataFrame(raw).copy()
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{label} selected target diagnostics缺欄位: {missing}")
        if bool(frame.duplicated(keys).any()):
            bad = frame.loc[frame.duplicated(keys, keep=False), keys].iloc[0].to_dict()
            raise ValueError(f"{label} selected target diagnostics selection key不唯一: {bad}")
        frame["target_raw_r"] = pd.to_numeric(frame["target_raw_r"], errors="coerce")
        available = frame["target_available"]
        if available.dtype == bool:
            frame["target_available"] = available.fillna(False)
        else:
            frame["target_available"] = (
                available.fillna(False).astype(str).str.strip().str.lower()
                .isin({"true", "1", "yes"})
            )
        frames.append(frame)
    baseline, active = frames
    baseline_keys = set(map(tuple, baseline[keys].itertuples(index=False, name=None)))
    active_keys = set(map(tuple, active[keys].itertuples(index=False, name=None)))
    baseline_only_keys = baseline_keys - active_keys
    active_only_keys = active_keys - baseline_keys

    def exclusive(frame: pd.DataFrame, selected_keys: set[tuple[Any, ...]]) -> pd.DataFrame:
        if not selected_keys:
            return frame.iloc[0:0].copy()
        mask = [tuple(row) in selected_keys for row in frame[keys].itertuples(index=False, name=None)]
        return frame.loc[mask].copy()

    baseline_only = exclusive(baseline, baseline_only_keys)
    active_only = exclusive(active, active_only_keys)
    all_exclusive = pd.concat([baseline_only, active_only], ignore_index=True)
    valid_mask = (
        all_exclusive["target_available"]
        & all_exclusive["target_raw_r"].map(math.isfinite)
    ) if not all_exclusive.empty else pd.Series(dtype=bool)
    exclusive_count = int(len(all_exclusive))
    covered_count = int(valid_mask.sum()) if exclusive_count else 0
    coverage = (float(covered_count / exclusive_count) if exclusive_count else 1.0)
    if exclusive_count and covered_count != exclusive_count:
        delta_r = None
        baseline_only_r = None
        active_only_r = None
    else:
        baseline_only_r = float(baseline_only["target_raw_r"].sum()) if len(baseline_only) else 0.0
        active_only_r = float(active_only["target_raw_r"].sum()) if len(active_only) else 0.0
        delta_r = float(active_only_r - baseline_only_r)
    return {
        "comparison_basis": "paired_exclusive_selected_occurrence_target_raw_r",
        "active_only_count": int(len(active_only)),
        "baseline_only_count": int(len(baseline_only)),
        "target_covered_exclusive_count": covered_count,
        "target_coverage_rate": coverage,
        "active_only_target_r": active_only_r,
        "baseline_only_target_r": baseline_only_r,
        "paired_target_selection_delta_r": delta_r,
        "future_target_used_for_runtime_sort": False,
    }


def backfill_pair_r_conversion_diagnostic(
    payload: dict[str, Any],
    *,
    pair_dir: Path,
) -> bool:
    """Backfill paired Target-selection R from existing canonical sidecars on REUSE.

    Returns True only when the payload was changed.  This deliberately avoids any
    market replay or raw-data reconstruction.
    """

    diagnostics = dict(payload.get("selection_diagnostics") or {})
    existing = dict(diagnostics.get("selection_r_conversion") or {})
    if _finite(existing.get("paired_target_selection_delta_r")) is not None:
        return False
    baseline_path = Path(pair_dir) / "no_filter_selected_target_diagnostics.csv"
    active_path = Path(pair_dir) / "score_ranking_selected_target_diagnostics.csv"
    if not baseline_path.is_file() or not active_path.is_file():
        return False
    baseline = pd.read_csv(baseline_path, encoding="utf-8-sig")
    active = pd.read_csv(active_path, encoding="utf-8-sig")
    diagnostics["selection_r_conversion"] = paired_target_selection_delta_r(
        baseline, active
    )
    payload["selection_diagnostics"] = diagnostics
    return True

# Stable public aliases for read-only consumers.
flatten_candidate_replay_rows = _flatten_candidate_replay_rows

# Human-readable aggregate report reuse layer.
def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_status_file(root: Path, status: dict[str, Any], dl_id: str, key: str) -> Path | None:
    row = (((status.get("dl_sources") or {}).get(dl_id) or {}).get("files") or {}).get(key) or {}
    raw = str(row.get("path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (root / path).resolve()


def _selection_pit_prediction_row(
    *, root: Path, status: dict[str, Any], dl_id: str, source: Any
) -> dict[str, Any]:
    path = _resolve_status_file(root, status, dl_id, "audit")
    payload = _read_json(path) if path is not None else None
    if payload is None:
        return {
            "dl_id": dl_id,
            "score_source": source.score_source,
            "scope": "-",
            "artifact": None if path is None else project_relative_display_path(path, project_root=root),
            "available": False,
        }
    contract = dict(payload.get("decision_contract") or {})
    scope = str(contract.get("primary_metric_scope") or "all_valid_target")
    metrics = dict((payload.get("metrics") or {}).get(scope) or {})
    direction = dict(payload.get("direction_summary") or {})
    return {
        "dl_id": dl_id,
        "score_source": source.score_source,
        "scope": str(contract.get("primary_metric_label") or scope),
        "global_spearman": _finite(metrics.get("global_spearman")),
        "mean_daily_spearman": _finite(metrics.get("mean_daily_spearman")),
        "pairwise_concordance": _finite(metrics.get("pairwise_concordance")),
        "top_target_r": _finite(metrics.get("top_decile_target_mean")),
        "bottom_target_r": _finite(metrics.get("bottom_decile_target_mean")),
        "top_bottom_target_spread_r": _finite(metrics.get("top_bottom_target_spread")),
        "valid_year_count": int(direction.get("valid_year_count") or 0),
        "positive_spearman_year_count": int(direction.get("positive_spearman_year_count") or 0),
        "positive_spearman_year_rate": _finite(direction.get("positive_spearman_year_rate")),
        "artifact": project_relative_display_path(path, project_root=root),
        "available": True,
    }


def _continuous_prediction_row(
    *, root: Path, status: dict[str, Any], dl_id: str, source: Any
) -> dict[str, Any]:
    path = _resolve_status_file(root, status, dl_id, "report")
    payload = _read_json(path) if path is not None else None
    if payload is None:
        return {
            "dl_id": dl_id,
            "score_source": source.score_source,
            "scope": "Forward OOS",
            "artifact": None if path is None else project_relative_display_path(path, project_root=root),
            "available": False,
        }
    metrics = dict((payload.get("split_metrics") or {}).get("oos") or {})
    top = _finite(metrics.get("top_score_decile_raw_target_mean"))
    bottom = _finite(metrics.get("bottom_score_decile_raw_target_mean"))
    spread = None if top is None or bottom is None else top - bottom
    return {
        "dl_id": dl_id,
        "score_source": source.score_source,
        "scope": "Forward OOS all eligible stock-days",
        "global_spearman": _finite(metrics.get("global_spearman_vs_raw_target")),
        "mean_daily_spearman": _finite(metrics.get("mean_daily_spearman")),
        "pairwise_concordance": _finite(metrics.get("pairwise_concordance")),
        "top_target_r": top,
        "bottom_target_r": bottom,
        "top_bottom_target_spread_r": spread,
        "valid_year_count": None,
        "positive_spearman_year_count": None,
        "positive_spearman_year_rate": None,
        "artifact": project_relative_display_path(path, project_root=root),
        "available": True,
    }


def _model_prediction_rows(
    *, root: Path, settings: StrategyComparisonSettings, status: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for arm in settings.enabled_arms:
        if not arm.dl_enabled or not arm.dl_id or arm.dl_id in seen:
            continue
        seen.add(arm.dl_id)
        source = settings.dl_sources[arm.dl_id]
        if source.score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            row = _selection_pit_prediction_row(root=root, status=status, dl_id=arm.dl_id, source=source)
        elif source.score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
            row = _continuous_prediction_row(root=root, status=status, dl_id=arm.dl_id, source=source)
        else:
            row = {
                "dl_id": arm.dl_id,
                "score_source": source.score_source,
                "scope": "-",
                "artifact": None,
                "available": False,
            }
        rows.append(row)
    return rows


def _selection_translation_rows(
    *, settings: StrategyComparisonSettings, pair_payloads: dict[str, dict[str, Any]], scenarios: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    by_arm: dict[str, dict[str, Any]] = {}
    for pair in pair_payloads.values():
        _param_source, _rule_policy, _off_arm, on_arm = pair["arm_contract"]
        if on_arm is None:
            continue
        payload = dict(pair.get("payload") or {})
        diagnostics = dict(payload.get("selection_diagnostics") or {})
        if not diagnostics:
            continue
        mode = str((payload.get("metadata") or {}).get("comparison_mode") or "")
        active_name = comparison_labels(mode)["active_name"]
        baseline = dict(diagnostics.get("no_filter") or {})
        active = dict(diagnostics.get(active_name) or {})
        if not active:
            continue

        def delta(key: str) -> float | None:
            left = _finite(active.get(key))
            right = _finite(baseline.get(key))
            return None if left is None or right is None else left - right

        scenario = scenarios.get(on_arm.arm_id) or {}
        conversion = dict(diagnostics.get("selection_r_conversion") or {})
        target_selection_delta_r = _finite(
            conversion.get("paired_target_selection_delta_r")
        )
        realized_selection_delta_r = _finite(scenario.get("direct_selection_delta_r"))
        r_conversion_efficiency = (
            None
            if target_selection_delta_r is None
            or target_selection_delta_r <= 1e-12
            or realized_selection_delta_r is None
            else float(realized_selection_delta_r / target_selection_delta_r)
        )
        by_arm[on_arm.arm_id] = {
            "arm_id": on_arm.arm_id,
            "name": on_arm.name,
            "dl_id": on_arm.dl_id,
            "score_coverage": _finite(active.get("orderable_score_coverage_rate")),
            "paired_target_selection_delta_r": target_selection_delta_r,
            "r_conversion_efficiency": r_conversion_efficiency,
            "selected_target_mean_r": _finite(active.get("selected_target_mean_r")),
            "selected_target_mean_r_delta": delta("selected_target_mean_r"),
            "selected_target_percentile": _finite(active.get("selected_target_percentile_mean")),
            "selected_target_percentile_delta": delta("selected_target_percentile_mean"),
            "target_top_k_retention": _finite(active.get("target_top_k_retention_mean")),
            "target_top_k_retention_delta": delta("target_top_k_retention_mean"),
            "target_opportunity_gap_r": _finite(active.get("target_opportunity_gap_r_mean")),
            "target_opportunity_gap_r_delta": delta("target_opportunity_gap_r_mean"),
            "direct_selection_delta_r": _finite(scenario.get("direct_selection_delta_r")),
        }
    return [by_arm[arm.arm_id] for arm in settings.enabled_arms if arm.arm_id in by_arm]


def _execution_rows(
    *, settings: StrategyComparisonSettings, scenarios: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for arm in settings.enabled_arms:
        scenario = scenarios[arm.arm_id]
        rows.append({
            "arm_id": arm.arm_id,
            "name": arm.name,
            "avg_exposure_pct": _finite(scenario.get("avg_exposure_pct")),
            "reserved_buy_fill_rate_pct": _finite(scenario.get("reserved_buy_fill_rate_pct")),
            "avg_orderable_candidates": _finite(scenario.get("avg_orderable_candidates")),
            "candidate_supply_gap_days": _finite(scenario.get("candidate_supply_gap_days")),
            "underfilled_end_days": _finite(scenario.get("underfilled_end_days")),
            "end_position_gap_slot_days": _finite(scenario.get("end_position_gap_slot_days")),
        })
    return rows


def _r_analysis_rows(
    *,
    settings: StrategyComparisonSettings,
    scenarios: dict[str, dict[str, Any]],
    model_prediction: list[dict[str, Any]],
    selection_translation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join canonical model/selection diagnostics to canonical strategy R outcomes."""

    model_by_dl = {
        str(row.get("dl_id") or ""): dict(row)
        for row in model_prediction
        if str(row.get("dl_id") or "")
    }
    translation_by_arm = {
        str(row.get("arm_id") or ""): dict(row)
        for row in selection_translation
        if str(row.get("arm_id") or "")
    }
    rows: list[dict[str, Any]] = []
    for arm in settings.enabled_arms:
        scenario = dict(scenarios.get(arm.arm_id) or {})
        model = dict(model_by_dl.get(str(arm.dl_id or "")) or {})
        translation = dict(translation_by_arm.get(arm.arm_id) or {})
        rows.append({
            "arm_id": arm.arm_id,
            "name": arm.name,
            "dl_id": arm.dl_id,
            "portfolio_avg_r": _finite(scenario.get("portfolio_avg_r")),
            "portfolio_median_r": _finite(scenario.get("portfolio_median_r")),
            "mean_daily_spearman": _finite(model.get("mean_daily_spearman")),
            "global_spearman": _finite(model.get("global_spearman")),
            "pairwise_concordance": _finite(model.get("pairwise_concordance")),
            "top_target_r": _finite(model.get("top_target_r")),
            "bottom_target_r": _finite(model.get("bottom_target_r")),
            "top_bottom_target_spread_r": _finite(model.get("top_bottom_target_spread_r")),
            "score_coverage": _finite(translation.get("score_coverage")),
            "paired_target_selection_delta_r": _finite(translation.get("paired_target_selection_delta_r")),
            "r_conversion_efficiency": _finite(translation.get("r_conversion_efficiency")),
            "selected_target_mean_r": _finite(translation.get("selected_target_mean_r")),
            "selected_target_mean_r_delta": _finite(translation.get("selected_target_mean_r_delta")),
            "selected_target_percentile": _finite(translation.get("selected_target_percentile")),
            "selected_target_percentile_delta": _finite(translation.get("selected_target_percentile_delta")),
            "target_top_k_retention": _finite(translation.get("target_top_k_retention")),
            "target_top_k_retention_delta": _finite(translation.get("target_top_k_retention_delta")),
            "target_opportunity_gap_r": _finite(translation.get("target_opportunity_gap_r")),
            "target_opportunity_gap_r_delta": _finite(translation.get("target_opportunity_gap_r_delta")),
            "direct_selection_delta_r": _finite(translation.get("direct_selection_delta_r")),
        })
    return rows


def build_strategy_diagnostics(
    *,
    project_root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
    pair_payloads: dict[str, dict[str, Any]],
    reference_arm_id: str | None = None,
) -> dict[str, Any]:
    """Build one reusable diagnostic payload from already-canonical artifacts."""

    root = Path(project_root).resolve()
    model_prediction = _model_prediction_rows(root=root, settings=settings, status=status)
    selection_translation = _selection_translation_rows(
        settings=settings, pair_payloads=pair_payloads, scenarios=scenarios
    )
    return {
        "reference_arm_id": str(reference_arm_id or ""),
        "model_prediction": model_prediction,
        "selection_translation": selection_translation,
        "r_analysis": _r_analysis_rows(
            settings=settings,
            scenarios=scenarios,
            model_prediction=model_prediction,
            selection_translation=selection_translation,
        ),
        "execution_conversion": _execution_rows(settings=settings, scenarios=scenarios),
        "contract": {
            "model_metrics_source": "validated existing PIT audit / continuous ranker report",
            "strategy_metrics_source": "canonical Strategy Compare pair payloads",
            "raw_market_or_trade_recalculation": False,
        },
    }


def _fmt(value: Any, *, digits: int = 3, unit: str = "") -> str:
    numeric = _finite(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}{unit}"


def _fmt_pct_fraction(value: Any) -> str:
    numeric = _finite(value)
    return "-" if numeric is None else f"{numeric * 100.0:.2f}%"


def _value_with_signal(text: str, signal: str | None, *, target: str) -> str:
    if text == "-" or not signal:
        return text
    return styled_signal(text, signal, target=target)


def _format_r_analysis_value(value: Any, metric: RAnalysisMetricSpec) -> str:
    if metric.format_kind == "fraction_pct":
        return _fmt_pct_fraction(value)
    return _fmt(value, digits=metric.digits, unit=metric.unit)


def _pad_cell(text: str, width: int, *, align: str = "left") -> str:
    value = str(text)
    padding = max(0, int(width) - _display_width(value))
    if align == "right":
        return " " * padding + value
    if align == "center":
        left = padding // 2
        return " " * left + value + " " * (padding - left)
    return value + " " * padding


def _render_grouped_table(headers_top: list[str], headers_bottom: list[str], rows: list[list[str]]) -> str:
    column_count = len(headers_bottom)
    widths = []
    for index in range(column_count):
        values = [headers_top[index], headers_bottom[index], *(row[index] for row in rows)]
        widths.append(max(_display_width(value) for value in values))
    def join_line(values: list[str], *, centers: set[int] | None = None) -> str:
        centers = centers or set()
        return "  ".join(
            _pad_cell(value, widths[idx], align="center" if idx in centers else "left")
            for idx, value in enumerate(values)
        )
    separator = "  ".join("-" * width for width in widths)
    return "\n".join([join_line(headers_top, centers=set(range(column_count))), join_line(headers_bottom, centers=set(range(1, column_count))), separator, *[join_line(row) for row in rows]])


def _render_metric_notes() -> str:
    lines = ["註解", "----"]
    for group_label, metrics in R_ANALYSIS_GROUPED_SECTIONS:
        lines.append(f"{group_label}：")
        for metric in metrics:
            lines.append(
                f"- {metric.label}：{metric.definition}｜理想方向：{metric.ideal_direction}"
            )
    return "\n".join(lines)


def render_strategy_r_analysis_table(diagnostics: dict[str, Any], *, target: str = "plain") -> str:
    """Render canonical R diagnostics as one grouped arm-comparison table."""

    rows = list(diagnostics.get("r_analysis") or [])
    if not rows:
        return "沒有可用的R預測／轉化診斷。"
    metrics = R_ANALYSIS_MERGED_METRICS
    top_headers = ["分群", ""]
    bottom_headers = ["編號", "比較對象"]
    for group_label, group_metrics in R_ANALYSIS_GROUPED_SECTIONS:
        for index, metric in enumerate(group_metrics):
            top_headers.append(group_label if index == 0 else "")
            bottom_headers.append(metric.label)
    body: list[list[str]] = []
    metric_signals = {
        metric.key: best_worst_signals(
            {str(row.get("arm_id") or "-"): row.get(metric.key) for row in rows},
            preference=metric.preference,
        )
        for metric in metrics
    }
    for row in rows:
        arm_id = str(row.get("arm_id") or "-")
        values = [arm_id, str(row.get("name") or "-")]
        for metric in metrics:
            text = _format_r_analysis_value(row.get(metric.key), metric)
            signal = metric_signals.get(metric.key, {}).get(arm_id)
            values.append(_value_with_signal(text, signal, target=target))
        body.append(values)
    return _render_grouped_table(top_headers, bottom_headers, body) + "\n\n" + _render_metric_notes()

def render_strategy_diagnostics_markdown(diagnostics: dict[str, Any]) -> str:
    """Persist the same canonical three-table R view used by the aggregate report."""

    return (
        "# Strategy Compare 間接指標\n\n"
        + render_strategy_r_analysis_table(diagnostics, target="markdown").rstrip()
        + "\n"
    )

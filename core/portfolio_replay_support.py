import math
import time

import pandas as pd

from core.trade_plans import clone_shadow_position


def _format_replay_elapsed(seconds):
    total_seconds = max(0, int(float(seconds or 0.0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_replay_date(today):
    if hasattr(today, "strftime"):
        return today.strftime("%Y-%m-%d")
    return str(today)


def _run_portfolio_replay_phase(today, phase, callback, /, *args, **kwargs):
    try:
        return callback(*args, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"portfolio replay 失敗 | date={_format_replay_date(today)} | "
            f"phase={phase} | {type(exc).__name__}: {exc}"
        ) from exc


def _print_slow_ensemble_phase_heartbeat(*, verbose, phase, today, completed, total, started_at):
    if not verbose or int(total or 0) <= 0:
        return
    elapsed = time.perf_counter() - float(started_at)
    if elapsed < 3.0:
        return
    print(
        f"\033[90m⏳ replay 階段: {_format_replay_date(today)} | "
        f"phase={phase} | member={int(completed)}/{int(total)} | "
        f"elapsed={_format_replay_elapsed(elapsed)}...\033[0m",
        end="\r",
        flush=True,
    )


def _candidate_replay_snapshot(candidate, *, fallback_trade_date, is_orderable):
    """Return a stable diagnostic row without leaking runtime objects into CSV/JSON."""

    row = dict(candidate or {})
    params_obj = row.get("params_obj")

    def _date_text(value):
        if value is None or value == "":
            return ""
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    def _optional_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    trade_date = row.get("trade_date") or row.get("candidate_date") or fallback_trade_date
    signal_date = row.get("signal_date")
    rank_payload = row.get("breakout_quality_rank") if isinstance(row.get("breakout_quality_rank"), dict) else {}
    return {
        "ticker": str(row.get("ticker") or ""),
        "trade_date": _date_text(trade_date),
        "candidate_date": _date_text(row.get("candidate_date") or trade_date),
        "signal_date": _date_text(signal_date),
        "candidate_type": str(row.get("type") or ""),
        "entry_source": str(row.get("entry_source") or ""),
        "is_orderable": bool(is_orderable),
        "high_len": int(getattr(params_obj, "high_len", 0) or 0),
        "ensemble_vote_count": int(row.get("ensemble_vote_count", 1) or 1),
        "qty": int(row.get("qty", 0) or 0),
        "sort_value": _optional_float(row.get("sort_value")),
        "historical_ev": _optional_float(row.get("ev")),
        "historical_win_rate": _optional_float(row.get("hist_win_rate")),
        "historical_trade_count": int(row.get("hist_trade_count", 0) or 0),
        "breakout_quality_score": _optional_float(row.get("breakout_quality_score")),
        "breakout_quality_score_date": str(row.get("breakout_quality_score_date") or ""),
        "breakout_quality_score_source": str(row.get("breakout_quality_score_source") or ""),
        "breakout_quality_safety_score": _optional_float(row.get("breakout_quality_safety_score")),
        "breakout_quality_safety_score_available": bool(row.get("breakout_quality_safety_score_available", False)),
        "breakout_quality_safety_score_date": str(row.get("breakout_quality_safety_score_date") or ""),
        "breakout_quality_safety_score_source": str(row.get("breakout_quality_safety_score_source") or ""),
        "breakout_quality_primary_score_percentile": _optional_float(row.get("breakout_quality_primary_score_percentile")),
        "breakout_quality_safety_score_percentile": _optional_float(row.get("breakout_quality_safety_score_percentile")),
        "breakout_quality_expected_safety_percentile": _optional_float(row.get("breakout_quality_expected_safety_percentile")),
        "breakout_quality_residual_safety_score": _optional_float(row.get("breakout_quality_residual_safety_score")),
        "breakout_quality_residual_safety_score_available": bool(row.get("breakout_quality_residual_safety_score_available", False)),
        "breakout_quality_ranking_policy": str(
            row.get("breakout_quality_ranking_policy") or "score"
        ),
        "breakout_quality_ranking_options": dict(
            row.get("breakout_quality_ranking_options") or {}
        ),
        "projected_capital_fraction": _optional_float(
            row.get("projected_capital_fraction")
        ),
        "projected_capital_deployment_rate": _optional_float(
            row.get("projected_capital_deployment_rate")
        ),
        "max_position_cap_pct": _optional_float(row.get("max_position_cap_pct")),
        "breakout_quality_score_available": bool(rank_payload.get("available", False)),
        "breakout_quality_score_unavailable_reason": str(rank_payload.get("unavailable_reason") or ""),
        "breakout_quality_expected_r_available": bool(rank_payload.get("expected_r_available", False)),
        "breakout_quality_expected_r": _optional_float(rank_payload.get("expected_r")),
        "breakout_quality_daily_score_percentile": _optional_float(
            rank_payload.get("daily_score_percentile")
        ),
        "breakout_quality_expected_r_calibration_cutoff": str(
            rank_payload.get("expected_r_calibration_cutoff") or ""
        ),
        "breakout_quality_expected_excess_r_available": bool(
            rank_payload.get("expected_excess_r_available", False)
        ),
        "breakout_quality_expected_excess_r": _optional_float(
            rank_payload.get("expected_excess_r")
        ),
        "breakout_quality_expected_excess_r_calibration_cutoff": str(
            rank_payload.get("expected_excess_r_calibration_cutoff") or ""
        ),
    }


def _candidate_execution_replay_snapshot(candidate, *, fallback_trade_date, all_dfs_fast, sizing_equity):
    """Return a compact in-memory sidecar row for research execution replay.

    This payload is never serialized by the canonical replay and never receives
    lifecycle callbacks.  Read-only params/market objects remain shared; only
    the small mutable shadow position is cloned.
    """

    row = dict(candidate or {})
    fields = (
        "ticker",
        "type",
        "entry_source",
        "signal_date",
        "candidate_date",
        "trade_date",
        "qty",
        "limit_px",
        "init_sl",
        "init_trail",
        "target_price",
        "entry_atr",
        "security_profile",
        "today_pos",
        "yesterday_pos",
        "sizing_capital",
        "params_obj",
        "orig_limit",
        "orig_atr",
        "max_qty",
    )
    snapshot = {key: row.get(key) for key in fields}
    snapshot["_sizing_equity"] = sizing_equity
    trade_date = row.get("trade_date") or row.get("candidate_date") or fallback_trade_date
    snapshot["trade_date"] = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    snapshot["candidate_date"] = pd.Timestamp(
        row.get("candidate_date") or trade_date
    ).strftime("%Y-%m-%d")
    signal_date = row.get("signal_date")
    snapshot["signal_date"] = (
        pd.Timestamp(signal_date).strftime("%Y-%m-%d") if signal_date not in (None, "") else ""
    )
    signal_state = row.get("signal_state") if isinstance(row.get("signal_state"), dict) else {}
    shadow = row.get("shadow_position_state")
    if shadow is None:
        shadow = signal_state.get("shadow_position")
    if shadow is not None:
        snapshot["shadow_position_state"] = clone_shadow_position(shadow)
    context = row.get("_ensemble_context") if isinstance(row.get("_ensemble_context"), dict) else {}
    candidate_all_dfs = context.get("all_dfs_fast") or all_dfs_fast or {}
    ticker = str(row.get("ticker") or "")
    fast_df = candidate_all_dfs.get(ticker) if ticker else None
    if fast_df is not None:
        snapshot["_candidate_fast_df"] = fast_df
    snapshot["_canonical_snapshot"] = _candidate_replay_snapshot(
        row,
        fallback_trade_date=fallback_trade_date,
        is_orderable=True,
    )
    return snapshot



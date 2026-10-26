"""Hindsight daily-universal Min trade-path score source for the formal Oracle.

Feature eligibility comes from the existing generic daily model dataset, NOT
from the retired MR-13CL label. Each seed's counterfactual 40-session trades
consume the identical DL-off parameters used by that seed's Min Portfolio.
Exact/Direct selectors read the same per-seed score table; neither can train.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.breakout_quality_registry import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.daily_trade40_target import (
    HORIZON_SESSIONS, ExecutableMinPeriod, collect_canonical_exchange_sessions,
    compute_counterfactual_trade40_batch, rank_trade40_outcome_frame,
)
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame

# Old MR-13CL's P2/P16 single-teacher label is a distinct retired scientific
# identity; no new result may masquerade as its supervised-label artifact.
TARGET_ID = "daily_universal_formal_min_roos_trade40_netr_v1"
LABEL_ID = "LABEL-daily_universal_formal_min_roos_trade40_netr_v1"


class DailyTrade40OracleBlockedError(ValueError):
    """A source, maturity or per-seed formal Min identity failed verification."""


def build_daily_universal_trade40_oracle_scores(
    *, project_root: Path, dataset: str, filter_id: str,
    profile_id: str, cutoff: str, evaluation_start: str,
    score_start: str | None = None,
    seed: int, param_sha256: str,
    canonical_schedule: tuple[ExecutableMinPeriod, ...],
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Compute all feature-eligible stock-day outcomes from a formal Min seed.

    Only information dates covered by the canonical PIT DL source are scored;
    the pre-OOS information bar may still be used to locate the first trade
    session, but cannot acquire a hindsight score outside the PIT score period.
    For each hypothetical entry, Min parameters become effective on the next
    observed *ticker* order date, matching Min signal->order semantics. Only
    the 40-session horizon is imposed on the otherwise canonical trade path.
    """
    root = Path(project_root).resolve()
    if not canonical_schedule or len(param_sha256) != 64:
        raise DailyTrade40OracleBlockedError("Missing verified formal Min parameters for this seed")
    if any(p.source_id != "formal_min_roos_seed" for p in canonical_schedule):
        raise DailyTrade40OracleBlockedError("Oracle parameter source is not the formal Min Portfolio")
    profile = get_breakout_quality_experiment_profile(profile_id)
    if (profile.training_sample_scope != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
            or profile.model_architecture != "inception_time_v1"
            or str(profile.continuous_target_id) == "daily_universal_counterfactual_mintrade40_netr_v1"):
        raise DailyTrade40OracleBlockedError("Feature-universe source must be a generic active daily profile")
    cutoff_day = pd.Timestamp(cutoff).normalize()
    start_day = pd.Timestamp(evaluation_start).normalize()
    if start_day > cutoff_day:
        raise DailyTrade40OracleBlockedError("Invalid Oracle trading period")
    # Source coverage is resolved from the actual Research PIT DL score contract.
    # No extra pre-start Oracle score is valid merely because the target matured.
    if not score_start:
        raise DailyTrade40OracleBlockedError("Missing canonical PIT DL score start")
    first_scored_day = pd.Timestamp(score_start).normalize()
    if first_scored_day < start_day or first_scored_day > cutoff_day:
        raise DailyTrade40OracleBlockedError("PIT DL score period outside Oracle evaluation")
    try:
        if progress_callback is not None:
            progress_callback("取得正式每日合格股票日母體")
        bundle = load_profile_continuous_ranker_data(
            filter_id=filter_id, model_architecture=str(profile.model_architecture),
            experiment_profile=profile_id, preload_feature_bank=False,
            allow_stale_source=False, project_root=root,
        )
        summary = dict(bundle.summary or {})
        group = pd.DataFrame(bundle.group_table).reset_index(drop=True)
        if (summary.get("dataset") != dataset or summary.get("filter_id") != filter_id
                or summary.get("sample_scope") != "daily_eligible_stock_days"
                or int(summary.get("score_eligible_sample_count", -1)) != len(group)
                or not {"ticker", "date"}.issubset(group.columns)):
            raise DailyTrade40OracleBlockedError("Generic daily feature-universe identity mismatch")
        if progress_callback is not None:
            progress_callback(f"已取得 {len(group):,} 筆股票日，盤點共同交易日曆")
        csv_inputs, duplicates = discover_dataset_csv_inputs(root, dataset)
        if duplicates:
            raise DailyTrade40OracleBlockedError("Duplicate canonical price ticker source")
        exchange = collect_canonical_exchange_sessions(
            csv_inputs, benchmark_ticker=str(DEFAULT_LABEL_POLICY.benchmark_ticker),
        )
        prior = exchange[exchange < start_day]
        if prior.empty:
            raise DailyTrade40OracleBlockedError("No legal pre-OOS information session")
        first_info = prior[-1]
        first_pos = int(exchange.get_indexer([first_info])[0])
        if first_pos + 1 >= len(exchange):
            raise DailyTrade40OracleBlockedError("No execution session after pre-OOS information")
        first_trade = exchange[first_pos + 1]
        if first_trade < start_day:
            raise DailyTrade40OracleBlockedError("Incorrect OOS trade start")
        source_paths = {normalize_ticker(ticker): Path(path) for ticker, path in csv_inputs}
        if len(source_paths) != len(csv_inputs):
            raise DailyTrade40OracleBlockedError("Duplicate normalized stock ticker")
        group["ticker"] = group["ticker"].map(normalize_ticker)
        group["date"] = pd.to_datetime(group["date"], errors="raise").dt.normalize()
        group = group.loc[(group["date"] >= first_scored_day) & (group["date"] <= cutoff_day)].reset_index(drop=True)
        if group.empty or group.duplicated(["ticker", "date"]).any():
            raise DailyTrade40OracleBlockedError("No unique feature-eligible daily stock-days")
        daily_calendar_pos = exchange.get_indexer(pd.DatetimeIndex(group["date"]))
        if (daily_calendar_pos < 0).any():
            raise DailyTrade40OracleBlockedError("Feature stock-day absent from canonical exchange calendar")
        eligible = daily_calendar_pos + HORIZON_SESSIONS < len(exchange)
        maturity = np.full(len(group), np.datetime64("NaT"), dtype="datetime64[D]")
        mature_positions = daily_calendar_pos[eligible] + HORIZON_SESSIONS
        maturity[eligible] = exchange.take(mature_positions).to_numpy(dtype="datetime64[D]")
        mature = eligible & (maturity <= np.datetime64(cutoff_day.date()))
        group = group.loc[mature].copy()
        if group.empty:
            raise DailyTrade40OracleBlockedError("No 40-session mature feature stock-days")
        records: list[pd.DataFrame] = []
        groups = group.groupby("ticker", sort=True)
        total_tickers = group["ticker"].nunique()
        if progress_callback is not None:
            progress_callback(f"開始逐股正式交易路徑（0/{total_tickers}）")
        for tick_index, (ticker, selected) in enumerate(groups, 1):
            path = source_paths.get(ticker)
            if path is None:
                raise DailyTrade40OracleBlockedError(f"Feature-universe ticker missing price source: {ticker}")
            frame = load_dataset_frame(path, ticker, min_rows=1)
            positions = pd.DatetimeIndex(frame.index).normalize().get_indexer(
                pd.DatetimeIndex(selected["date"])
            )
            if (positions < 0).any():
                raise DailyTrade40OracleBlockedError(f"Feature stock-day missing sanitized price: {ticker}")
            batch = compute_counterfactual_trade40_batch(
                frame, positions, ticker=ticker, schedule=canonical_schedule,
                exchange_calendar=exchange,
            )
            # No effective Min parameter is missing source truth, NOT an
            # unfilled trade outcome. Ranking it last would fabricate an
            # economic label for an unavailable rolling P16 parameter period.
            # Fail the whole controlled Oracle before publishing any scores.
            if len(batch.reasons) != len(selected):
                raise DailyTrade40OracleBlockedError(
                    f"Formal Min path result shape mismatch: {ticker}"
                )
            missing_param = next(
                (i for i, reason in enumerate(batch.reasons)
                 if reason == "no_historical_effective_Min_param"), None
            )
            if missing_param is not None:
                information_day = selected.iloc[missing_param]["date"].strftime("%Y-%m-%d")
                raise DailyTrade40OracleBlockedError(
                    f"Formal Min rolling parameter schedule missing at "
                    f"ticker={ticker} information_date={information_day}; "
                    "cannot censor an unavailable parameter as an unfilled trade"
                )
            expected_deadlines = np.asarray(maturity[selected.index], dtype="datetime64[D]")
            if not np.array_equal(batch.maturity_dates, expected_deadlines):
                raise DailyTrade40OracleBlockedError("40-session mature deadline disagrees with Min target path")
            raw = np.asarray(batch.target_raw_r, dtype=np.float64)
            valid = np.asarray(batch.valid_mask, dtype=bool)
            if ((valid & ~np.isfinite(raw)).any() or ((~valid) & np.isfinite(raw)).any()):
                raise DailyTrade40OracleBlockedError("NULL/valid Net-R contract invalid")
            records.append(pd.DataFrame({
                "ticker": ticker,
                "date": selected["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                "realized_net_r": np.where(valid, raw, np.nan),
            }))
            if progress_callback is not None and (tick_index % 100 == 0 or tick_index == total_tickers):
                progress_callback(f"逐股交易路徑 {tick_index}/{total_tickers}")
        outcome = pd.concat(records, ignore_index=True).sort_values(["date", "ticker"], kind="stable")
        if outcome.empty or not outcome["realized_net_r"].notna().any():
            raise DailyTrade40OracleBlockedError("No completed valid formal-Min Net-R outcomes")
        if outcome.groupby("date")["realized_net_r"].count().eq(0).any():
            raise DailyTrade40OracleBlockedError("At least one information date has no settled path")
        scores = rank_trade40_outcome_frame(outcome)
        if len(scores) != len(outcome):
            raise DailyTrade40OracleBlockedError("Oracle daily scores missing feature-eligible rows")
        raw_sha = sha256(outcome.to_csv(index=False, na_rep="NULL", float_format="%.10g").encode("utf-8")).hexdigest()
        rank_sha = sha256("".join(f"{t}|{d}|{scores[t,d]:.17g}\n" for t,d in sorted(scores)).encode("ascii")).hexdigest()
        cal_sha = sha256("\n".join(exchange.strftime("%Y-%m-%d")).encode("ascii")).hexdigest()
        teacher = {
            "origin": "formal_min_roos_rolling_per_seed", "seed": int(seed),
            "param_identity_sha256": str(param_sha256),
            "periods": [{"start": str(p.start.date()),
                         "end": None if p.end is None else str(p.end.date()),
                         "params_signature": p.params_signature} for p in canonical_schedule],
        }
        evidence = {
            "label_id": LABEL_ID, "target_id": TARGET_ID,
            "profile_id": profile_id, "key_mode": "information_date",
            "cohort": "daily_eligible_stock_days_formal_min_40_exchange_sessions_mature",
            "mature_eligible_rows": len(outcome),
            "settled_net_r_rows": int(outcome["realized_net_r"].notna().sum()),
            "censored_rank_last_rows": int(outcome["realized_net_r"].isna().sum()),
            "first_date": outcome["date"].min(), "last_date": outcome["date"].max(),
            "first_replay_trade_date": str(first_trade.date()),
            "research_cutoff": str(cutoff_day.date()),
            "score_start_date": str(first_scored_day.date()),
            "rank_sha256": rank_sha, "raw_label_sha256": raw_sha,
            "calendar_sha256": cal_sha, "teacher": teacher,
            "rank_policy": "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
            "oracle_evaluation_start_date": str(start_day.date()),
            "oracle_hindsight_only": True,
        }
        return scores, evidence
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DailyTrade40OracleBlockedError):
            raise
        raise DailyTrade40OracleBlockedError(
            f"Daily-universal formal Min source blocked: {type(exc).__name__}: "
            f"{str(exc).replace(str(root), '.')}"
        ) from exc


__all__ = ["TARGET_ID", "LABEL_ID", "DailyTrade40OracleBlockedError",
           "build_daily_universal_trade40_oracle_scores"]

"""Distinct Full/P17 Oracle truth adapter, sharing the canonical trade economics.

The existing Min/P16 truth producer remains byte-identical.  This module is
only an independent evidence/cache/universe orchestration for the verified
Full/P17 parameter schedule. It delegates orders, fills, stops, fees, closeout,
reference sizing and same-day ranking to the already signed canonical modules.
"""
from __future__ import annotations

from hashlib import sha256
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.file_integrity import canonical_json_sha256, is_sha256_hex
from core.breakout_quality_runtime import FULLTRADE_FULLCYCLE_NETPNL_TARGET_ID
from core.breakout_quality_registry import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.daily_trade40_target import (
    HORIZON_SESSIONS, ExecutableMinPeriod, collect_canonical_exchange_sessions,
    rank_trade40_outcome_frame,
)
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs
from services.audit.oracle_score_artifacts import oracle_dataset_summary_sha256
from services.audit.daily_trade40_oracle import (
    DailyTrade40OracleBlockedError, DailyTrade40TruthBuildRequired,
    NET_PNL_EVIDENCE, NET_PNL_RANK_POLICY, _TRUTH_CACHE_SCHEMA,
    _truth_cache_read, _truth_cache_write, _run_ticker_outcomes,
    truth_producer_fingerprint as min_truth_producer_fingerprint,
)

TARGET_ID = FULLTRADE_FULLCYCLE_NETPNL_TARGET_ID
LABEL_ID = "LABEL-" + TARGET_ID

@lru_cache(maxsize=1)
def truth_producer_fingerprint() -> str:
    """Unique identity including current Full adapter and the canonical economic engine."""
    root = Path(__file__).resolve().parents[2]
    digest = sha256(b"formal_full_p17_netpnl_truth_v1\0")
    digest.update(min_truth_producer_fingerprint("net_pnl").encode("ascii") + b"\n")
    for relative in (
        "services/audit/fulltrade_oracle_truth.py",
        "services/breakout_quality/formal_fulltrade_training_source.py",
    ):
        code = root / relative
        if not code.is_file():
            raise DailyTrade40OracleBlockedError(f"FullTrade Oracle source missing: {relative}")
        digest.update(relative.encode("utf-8") + b"\0" + sha256(code.read_bytes()).hexdigest().encode("ascii") + b"\n")
    return digest.hexdigest()

def build_fulltrade_oracle_scores(
    *, project_root: Path, dataset: str, filter_id: str,
    profile_id: str, cutoff: str, evaluation_start: str,
    score_start: str | None = None,
    seed: int, param_sha256: str,
    canonical_schedule: tuple[ExecutableMinPeriod, ...],
    horizon_sessions: int | None = HORIZON_SESSIONS,
    outcome_metric: str = "net_r",
    evaluation_end: str | None = None,
    price_sha256: str | None = None, cache_root: Path | None = None,
    feature_cohort_cache: dict[tuple[str, ...], tuple[pd.DataFrame, str]] | None = None,
    allow_build: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    ticker_workers: int = 1,
    exchange_calendar_cache: dict[tuple[str, str, str], pd.DatetimeIndex] | None = None,
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Compute all feature-eligible stock-day outcomes from a formal Full seed.

    Only information dates covered by the canonical PIT DL source are scored;
    the pre-OOS information bar may still be used to locate the first trade
    session, but cannot acquire a hindsight score outside the PIT score period.
    For each hypothetical entry, Min parameters become effective on the next
    observed *ticker* order date, matching Min signal->order semantics. Only
    the declared horizon and Research data cutoff are both imposed on the
    canonical trade path as earliest legal closeout boundaries.
    """
    if horizon_sessions is not None or outcome_metric != "net_pnl":
        raise DailyTrade40OracleBlockedError("FullTrade Oracle accepts full-cycle NetPnL only")
    target_id, label_id = TARGET_ID, LABEL_ID
    value_column = "realized_net_pnl" if outcome_metric == "net_pnl" else "realized_net_r"
    root = Path(project_root).resolve()
    if not canonical_schedule or len(param_sha256) != 64:
        raise DailyTrade40OracleBlockedError("Missing verified formal Full parameters for this seed")
    if any(p.source_id != "formal_full_roos_seed" for p in canonical_schedule):
        raise DailyTrade40OracleBlockedError("Oracle parameter source is not the formal Full Portfolio")
    profile = get_breakout_quality_experiment_profile(profile_id)
    if (profile.training_sample_scope != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
            or profile.model_architecture != "inception_time_v1"
            or str(profile.continuous_target_id) == "daily_universal_counterfactual_mintrade40_netr_v1"):
        raise DailyTrade40OracleBlockedError("Feature-universe source must be a generic active daily profile")
    cutoff_day = pd.Timestamp(cutoff).normalize()
    start_day = pd.Timestamp(evaluation_start).normalize()
    evaluation_end_day = (pd.Timestamp(evaluation_end).normalize()
                          if evaluation_end is not None else cutoff_day)
    if not (start_day <= evaluation_end_day <= cutoff_day):
        raise DailyTrade40OracleBlockedError("Evaluation window must end by research cutoff")
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
        # Feature eligibility is unchanged by the chosen holding horizon. One
        # invocation loads the certified multi-million-row cohort only once.
        dataset_summary_sha = oracle_dataset_summary_sha256(root=root, filter_id=filter_id)
        cohort_key = (str(root), str(dataset), str(filter_id), str(profile_id),
                      str(first_scored_day.date()), str(evaluation_end_day.date()),
                      str(dataset_summary_sha))
        snapshot = (feature_cohort_cache.get(cohort_key)
                    if feature_cohort_cache is not None else None)
        if snapshot is None:
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
            group["ticker"] = group["ticker"].map(normalize_ticker)
            group["date"] = pd.to_datetime(group["date"], errors="raise").dt.normalize()
            group = group.loc[(group["date"] >= first_scored_day) &
                              (group["date"] <= evaluation_end_day)].reset_index(drop=True)
            if group.empty or group.duplicated(["ticker", "date"]).any():
                raise DailyTrade40OracleBlockedError("No unique feature-eligible daily stock-days")
            group_sha = sha256(group[["ticker", "date"]].sort_values(
                ["ticker", "date"], kind="stable").to_csv(index=False).encode("utf-8")).hexdigest()
            if feature_cohort_cache is not None:
                feature_cohort_cache[cohort_key] = (group, group_sha)
        else:
            group, group_sha = snapshot
            if progress_callback is not None:
                progress_callback(f"合格股票日母體共用（{len(group):,} 筆）")
        cache_path = None
        cache_identity = None
        if cache_root is not None:
            if not is_sha256_hex(price_sha256):
                raise DailyTrade40OracleBlockedError("Oracle truth REUSE requires verified OHLCV SHA")
            cache_identity = {
                "schema": _TRUTH_CACHE_SCHEMA,
                "target_id": target_id, "dataset": str(dataset),
                "filter_id": str(filter_id), "profile_id": str(profile_id),
                "evaluation_start": str(start_day.date()),
                "evaluation_end": str(evaluation_end_day.date()),
                "score_start": str(first_scored_day.date()),
                "research_cutoff": str(cutoff_day.date()),
                "seed": int(seed), "param_sha256": str(param_sha256),
                "rolling_params": [{
                    "start": str(item.start.date()),
                    "end": None if item.end is None else str(item.end.date()),
                    "signature": item.params_signature,
                } for item in canonical_schedule],
                "horizon_sessions": horizon_sessions,
                "price_sha256": str(price_sha256),
                "feature_eligible_keys_sha256": group_sha,
                "truth_source_sha256": truth_producer_fingerprint(),
                "dataset_summary_sha256": dataset_summary_sha,
            }
            cache_path = Path(cache_root) / str(seed) / (
                f"{target_id}_{canonical_json_sha256(cache_identity)}.npz"
            )
            cached = _truth_cache_read(cache_path, cache_identity)
            if cached is not None:
                if progress_callback is not None:
                    progress_callback(f"Target 真值 REUSE（{len(cached[0]):,} 筆，無逐股模擬）")
                return cached
            if not allow_build:
                raise DailyTrade40TruthBuildRequired(f"{target_id} seed={seed} needs truth BUILD")
        if not allow_build:
            raise DailyTrade40TruthBuildRequired(f"{target_id} seed={seed} has no certified truth cache")
        if progress_callback is not None:
            progress_callback(f"真值 BUILD：{len(group):,} 個股票日，建立交易路徑")
        csv_inputs, duplicates = discover_dataset_csv_inputs(root, dataset)
        if duplicates:
            raise DailyTrade40OracleBlockedError("Duplicate canonical price ticker source")
        # Calendar depends only on the certified price inventory, not on
        # horizon or P16 seed; reuse within this worker for subsequent Targets.
        calendar_key = (str(root), str(dataset), str(price_sha256))
        exchange = (exchange_calendar_cache.get(calendar_key)
                    if exchange_calendar_cache is not None and is_sha256_hex(price_sha256)
                    else None)
        if exchange is None:
            exchange = collect_canonical_exchange_sessions(
                csv_inputs, benchmark_ticker=str(DEFAULT_LABEL_POLICY.benchmark_ticker),
                horizon_sessions=horizon_sessions,
            )
            if exchange_calendar_cache is not None and is_sha256_hex(price_sha256):
                exchange_calendar_cache[calendar_key] = exchange
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
        daily_calendar_pos = exchange.get_indexer(pd.DatetimeIndex(group["date"]))
        if (daily_calendar_pos < 0).any():
            raise DailyTrade40OracleBlockedError("Feature stock-day absent from canonical exchange calendar")
        # Eligible DL score rows must NEVER depend on the Target's future
        # holding period. The Research cutoff controls the outcome's final
        # closeout, not whether the date is allowed in the score cross-section.
        if not len(exchange[exchange <= cutoff_day]) or group.empty:
            raise DailyTrade40OracleBlockedError("No Research-cutoff feature stock-days")
        if type(ticker_workers) is not int or not 1 <= ticker_workers <= 8:
            raise DailyTrade40OracleBlockedError("Formal Full ticker_workers must be in [1, 8]")
        groups = group.groupby("ticker", sort=True)
        jobs = []
        for index, (ticker, selected) in enumerate(groups):
            path = source_paths.get(ticker)
            if path is None:
                raise DailyTrade40OracleBlockedError(
                    f"Feature-universe ticker missing price source: {ticker}"
                )
            jobs.append((index, ticker, str(path), pd.DatetimeIndex(selected["date"])))
        if progress_callback is not None:
            progress_callback(f"開始逐股正式交易路徑（0/{len(jobs)}）")
        records, total_truncated, total_cutoff_closeouts = _run_ticker_outcomes(
            jobs, schedule=canonical_schedule, exchange=exchange,
            horizon=horizon_sessions, cutoff=cutoff_day, metric=outcome_metric,
            workers=ticker_workers, progress=progress_callback,
        )
        outcome = pd.concat(records, ignore_index=True).sort_values(["date", "ticker"], kind="stable")
        if outcome.empty or not outcome[value_column].notna().any():
            raise DailyTrade40OracleBlockedError("No completed valid formal-Full economic outcomes")
        # An information date with no filled path is still a valid censored
        # cross-section: all its stocks tie for last place (canonical rank 0.5).
        # No R is imputed. Missing rolling Min parameters are blocked above;
        # genuine unfilled orders must not make the whole OOS window unusable.
        scores = rank_trade40_outcome_frame(outcome, value_column=value_column)
        if len(scores) != len(outcome):
            raise DailyTrade40OracleBlockedError("Oracle daily scores missing feature-eligible rows")
        # Preserve every TWD milli-unit carried from the canonical ledger.
        # Do not change the existing Net-R truth-cache serialization.
        raw_format = "%.17g" if outcome_metric == "net_pnl" else "%.10g"
        raw_sha = sha256(outcome.to_csv(index=False, na_rep="NULL", float_format=raw_format).encode("utf-8")).hexdigest()
        rank_sha = sha256("".join(f"{t}|{d}|{scores[t,d]:.17g}\n" for t,d in sorted(scores)).encode("ascii")).hexdigest()
        cal_sha = sha256("\n".join(exchange.strftime("%Y-%m-%d")).encode("ascii")).hexdigest()
        teacher = {
            "origin": "formal_full_roos_rolling_per_seed", "seed": int(seed),
            "param_identity_sha256": str(param_sha256),
            "periods": [{"start": str(p.start.date()),
                         "end": None if p.end is None else str(p.end.date()),
                         "params_signature": p.params_signature} for p in canonical_schedule],
        }
        evidence = {
            "label_id": label_id, "target_id": target_id,
            "profile_id": profile_id, "key_mode": "information_date",
            "cohort": "daily_eligible_stock_days_formal_full_research_cutoff_closeout",
            "mature_eligible_rows": len(outcome),
            "cutoff_truncated_rows": int(total_truncated),
            "research_cutoff_closeout_rows": int(total_cutoff_closeouts),
            "horizon_sessions": horizon_sessions,
            "evaluation_end_date": str(evaluation_end_day.date()),
            ("settled_net_pnl_rows" if outcome_metric == "net_pnl" else
             "settled_net_r_rows"): int(outcome[value_column].notna().sum()),
            "censored_rank_last_rows": int(outcome[value_column].isna().sum()),
            "first_date": outcome["date"].min(), "last_date": outcome["date"].max(),
            "first_replay_trade_date": str(first_trade.date()),
            "research_cutoff": str(cutoff_day.date()),
            "score_start_date": str(first_scored_day.date()),
            "truth_source_sha256": truth_producer_fingerprint(),
            "dataset_summary_sha256": dataset_summary_sha,
            "rank_sha256": rank_sha, "raw_label_sha256": raw_sha,
            "calendar_sha256": cal_sha, "teacher": teacher,
            **(NET_PNL_EVIDENCE if outcome_metric == "net_pnl" else {}),
            "rank_policy": (
                NET_PNL_RANK_POLICY
                if outcome_metric == "net_pnl" else
                "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation"
            ),
            "oracle_evaluation_start_date": str(start_day.date()),
            "oracle_hindsight_only": True,
        }
        if cache_path is not None:
            _truth_cache_write(cache_path, cache_identity, scores, evidence)
        return scores, {**evidence, "source_reuse_status": "BUILD"}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DailyTrade40OracleBlockedError):
            raise
        raise DailyTrade40OracleBlockedError(
            f"Daily-universal formal Full source blocked: {type(exc).__name__}: "
            f"{str(exc).replace(str(root), '.')}"
        ) from exc


__all__ = ["TARGET_ID", "LABEL_ID", "truth_producer_fingerprint", "build_fulltrade_oracle_scores"]

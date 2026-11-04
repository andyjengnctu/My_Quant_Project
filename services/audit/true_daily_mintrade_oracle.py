"""TrueDaily Min NetPnL truth source; reuse canonical Min math and signed stores.

Only the entry decision is new. The signal-driven source remains byte-identical
so its already-published truth/replay identities are not invalidated.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.file_integrity import canonical_json_sha256, compute_file_sha256, is_sha256_hex
from core.breakout_quality_registry import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.daily_trade40_target import (
    collect_canonical_exchange_sessions, rank_trade40_outcome_frame,
)
from filters.breakout_quality.true_daily_mintrade_target import compute_true_daily_mintrade_batch
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame
from services.audit import fullcycle_positive_truth as positive_witness
from services.audit.daily_trade40_oracle import (
    NET_PNL_EVIDENCE, NET_PNL_RANK_POLICY, DailyTrade40OracleBlockedError,
    _truth_cache_read, _truth_cache_write, truth_producer_fingerprint as original_min_fingerprint,
)
from services.audit.oracle_score_artifacts import oracle_dataset_summary_sha256

TARGET_ID = "true_daily_universal_mintrade_fulltrade_netpnl_v1"
LABEL_ID = f"LABEL-{TARGET_ID}"
ENTRY_POLICY = "next_exchange_session_information_close_limit_v1"


@lru_cache(maxsize=1)
def truth_producer_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    return canonical_json_sha256({
        "target_id": TARGET_ID, "entry_policy": ENTRY_POLICY,
        "producer": compute_file_sha256(Path(__file__)),
        "hypothetical_path": compute_file_sha256(
            root / "filters/breakout_quality/true_daily_mintrade_target.py"),
        "original_min_execution": original_min_fingerprint("net_pnl"),
    })


def _raw_path(root, subdir, seed, price_sha, param_sha, evidence):
    identity = positive_witness._identity(
        seed=seed, price_sha=price_sha, param_sha=param_sha,
        source_evidence=evidence, project_root=root,
    )
    return positive_witness._path(root, subdir, identity), identity


def load_positive_truth(*, project_root: Path, output_subdir: str, seed: int,
                        price_sha256: str, param_sha256: str,
                        canonical_scores: dict, canonical_evidence: dict,
                        execution=None, cache_root: Path | None = None) -> dict:
    """Read the original raw witness; only a signed effective reference permits provenance drift."""
    effective = canonical_evidence.get("effective_source_reference")
    if effective:
        if execution is None or cache_root is None:
            raise DailyTrade40OracleBlockedError("TrueDaily effective truth requires its signed input context")
        from services.audit.oracle_truth_references import OracleTruthReferences
        OracleTruthReferences(cache_root).validate_consumption(execution, canonical_evidence)
    if (canonical_evidence.get("target_id") != TARGET_ID or
            (not effective and
             canonical_evidence.get("truth_source_sha256") != truth_producer_fingerprint()) or
            canonical_evidence.get("entry_policy") != ENTRY_POLICY):
        raise DailyTrade40OracleBlockedError("TrueDaily raw NetPnL provenance mismatch")
    path, identity = _raw_path(Path(project_root), output_subdir, seed,
                               price_sha256, param_sha256, canonical_evidence)
    raw = positive_witness._load(path, identity)
    if raw is None or not set(raw).issubset(canonical_scores) or len(raw) != int(
            canonical_evidence["settled_net_pnl_rows"]):
        raise DailyTrade40OracleBlockedError("TrueDaily Positive signed raw witness missing/incompatible")
    return raw


def _stock_outcome(job, *, schedule, exchange, cutoff):
    index, ticker, path, dates = job
    frame = load_dataset_frame(Path(path), ticker, min_rows=1)
    positions = pd.DatetimeIndex(frame.index).normalize().get_indexer(dates)
    if (positions < 0).any():
        raise DailyTrade40OracleBlockedError(f"TrueDaily scored stock-day lacks OHLCV: {ticker}")
    batch = compute_true_daily_mintrade_batch(
        frame, positions, ticker=ticker, schedule=schedule,
        exchange_calendar=exchange, research_cutoff=cutoff,
    )
    if "no_historical_effective_Min_param" in batch.reasons:
        raise DailyTrade40OracleBlockedError(f"TrueDaily missing PIT P16: {ticker}")
    raw = np.asarray(batch.target_raw_pnl, dtype=np.float64)
    valid = np.asarray(batch.valid_mask, dtype=bool)
    if (len(raw) != len(dates) or len(batch.reasons) != len(dates) or
            (valid & ~np.isfinite(raw)).any() or ((~valid) & np.isfinite(raw)).any()):
        raise DailyTrade40OracleBlockedError(f"TrueDaily NULL/settled outcome mismatch: {ticker}")
    return index, pd.DataFrame({
        "ticker": ticker, "date": dates.strftime("%Y-%m-%d").to_numpy(),
        "realized_net_pnl": np.where(valid, raw, np.nan),
    }), int(batch.research_cutoff_closeout_rows)


_WORKER_STATE = None


def _init_worker(schedule, exchange, cutoff):
    global _WORKER_STATE
    _WORKER_STATE = (schedule, exchange, cutoff)


def _worker(job):
    if _WORKER_STATE is None:
        raise RuntimeError("TrueDaily worker missing per-seed execution contract")
    try:
        return _stock_outcome(job, schedule=_WORKER_STATE[0],
                              exchange=_WORKER_STATE[1], cutoff=_WORKER_STATE[2])
    except Exception as exc:
        raise DailyTrade40OracleBlockedError(f"TrueDaily ticker={job[1]} failed: {exc}") from exc


def _run_stocks(jobs, *, schedule, exchange, cutoff, workers, progress):
    if workers == 1 or len(jobs) < 2:
        results = (_stock_outcome(job, schedule=schedule, exchange=exchange,
                                  cutoff=cutoff) for job in jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)), mp_context=get_context("spawn"),
            initializer=_init_worker, initargs=(schedule, exchange, cutoff),
        )
        results = pool.map(_worker, jobs)
    frames, forced = [], 0
    try:
        for index, (job_id, frame, count) in enumerate(results):
            if job_id != index:
                raise DailyTrade40OracleBlockedError("TrueDaily unstable stock worker order")
            frames.append(frame)
            forced += count
            if progress and (len(frames) % 100 == 0 or len(frames) == len(jobs)):
                progress(f"TrueDaily 逐股假設交易 {len(frames)}/{len(jobs)}")
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    return frames, forced


def build_true_daily_universal_oracle_scores(
    *, project_root: Path, dataset: str, filter_id: str, profile_id: str,
    cutoff: str, evaluation_start: str, evaluation_end: str, score_start: str,
    seed: int, param_sha256: str, canonical_schedule: tuple, price_sha256: str,
    cache_root: Path, output_subdir: str, feature_cohort_cache: dict | None = None,
    exchange_calendar_cache: dict | None = None, ticker_workers: int = 1,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict, dict[str, Any]]:
    root = Path(project_root).resolve()
    profile = get_breakout_quality_experiment_profile(profile_id)
    if (profile.training_sample_scope != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS or
            profile.model_architecture != "inception_time_v1" or not canonical_schedule or
            not is_sha256_hex(param_sha256) or not is_sha256_hex(price_sha256) or
            type(ticker_workers) is not int or not 1 <= ticker_workers <= 8):
        raise DailyTrade40OracleBlockedError("TrueDaily invalid canonical cohort/P16/worker contract")
    start, end, research_end, first = (pd.Timestamp(v).normalize() for v in
                                       (evaluation_start, evaluation_end, cutoff, score_start))
    if not start <= first <= end <= research_end:
        raise DailyTrade40OracleBlockedError("TrueDaily score/evaluation/cutoff order invalid")
    dataset_sha = oracle_dataset_summary_sha256(root=root, filter_id=filter_id)
    cohort_key = (str(root), str(dataset), str(filter_id), str(profile_id),
                  str(first.date()), str(end.date()), str(dataset_sha))
    cached_cohort = feature_cohort_cache.get(cohort_key) if feature_cohort_cache is not None else None
    if cached_cohort is None:
        bundle = load_profile_continuous_ranker_data(
            filter_id=filter_id, model_architecture=profile.model_architecture,
            experiment_profile=profile_id, preload_feature_bank=False,
            allow_stale_source=False, project_root=root,
        )
        summary = dict(bundle.summary or {})
        cohort = pd.DataFrame(bundle.group_table).reset_index(drop=True)
        if (summary.get("dataset") != dataset or summary.get("filter_id") != filter_id or
                summary.get("sample_scope") != "daily_eligible_stock_days" or
                len(cohort) != int(summary.get("score_eligible_sample_count", -1)) or
                not {"ticker", "date"}.issubset(cohort)):
            raise DailyTrade40OracleBlockedError("TrueDaily cohort differs from canonical DL eligibility")
        cohort["ticker"] = cohort["ticker"].map(normalize_ticker)
        cohort["date"] = pd.to_datetime(cohort["date"], errors="raise").dt.normalize()
        cohort = cohort.loc[cohort["date"].between(first, end)].reset_index(drop=True)
        if cohort.empty or cohort.duplicated(["ticker", "date"]).any():
            raise DailyTrade40OracleBlockedError("TrueDaily empty/duplicated score cohort")
        group_sha = sha256(cohort[["ticker", "date"]].sort_values(
            ["ticker", "date"], kind="stable").to_csv(index=False).encode("utf-8")).hexdigest()
        if feature_cohort_cache is not None:
            feature_cohort_cache[cohort_key] = (cohort, group_sha)
    else:
        cohort, group_sha = cached_cohort
    identity = {
        "schema": 1, "target_id": TARGET_ID, "dataset": str(dataset),
        "filter_id": str(filter_id), "profile_id": str(profile_id),
        "evaluation_start": str(start.date()), "evaluation_end": str(end.date()),
        "score_start": str(first.date()), "research_cutoff": str(research_end.date()),
        "seed": int(seed), "param_sha256": str(param_sha256),
        "rolling_params": [{"start": str(p.start.date()),
                            "end": None if p.end is None else str(p.end.date()),
                            "signature": p.params_signature} for p in canonical_schedule],
        "horizon_sessions": None, "price_sha256": str(price_sha256),
        "feature_eligible_keys_sha256": group_sha,
        "truth_source_sha256": truth_producer_fingerprint(),
        "dataset_summary_sha256": dataset_sha, "entry_policy": ENTRY_POLICY,
    }
    cache_path = Path(cache_root) / str(seed) / f"{TARGET_ID}_{canonical_json_sha256(identity)}.npz"
    cached = _truth_cache_read(cache_path, identity)
    if cached is not None:
        load_positive_truth(project_root=root, output_subdir=output_subdir, seed=seed,
                            price_sha256=price_sha256, param_sha256=param_sha256,
                            canonical_scores=cached[0], canonical_evidence=cached[1])
        if progress_callback:
            progress_callback(f"TrueDaily signed truth REUSE ({len(cached[0]):,} stock-days)")
        return cached
    if progress_callback:
        progress_callback(f"TrueDaily truth BUILD ({len(cohort):,} stock-days)")
    csv_inputs, duplicates = discover_dataset_csv_inputs(root, dataset)
    if duplicates:
        raise DailyTrade40OracleBlockedError("TrueDaily duplicate price source")
    calendar_key = (str(root), str(dataset), str(price_sha256))
    exchange = (exchange_calendar_cache.get(calendar_key)
                if exchange_calendar_cache is not None else None)
    if exchange is None:
        exchange = collect_canonical_exchange_sessions(
            csv_inputs, benchmark_ticker=str(DEFAULT_LABEL_POLICY.benchmark_ticker),
            horizon_sessions=None,
        )
        if exchange_calendar_cache is not None:
            exchange_calendar_cache[calendar_key] = exchange
    if (exchange.get_indexer(pd.DatetimeIndex(cohort["date"])) < 0).any():
        raise DailyTrade40OracleBlockedError("TrueDaily stock-day missing canonical exchange date")
    prior = exchange[exchange < start]
    future = exchange[exchange >= start]
    if prior.empty or future.empty:
        raise DailyTrade40OracleBlockedError("TrueDaily missing legal OOS trade-start boundary")
    paths = {normalize_ticker(ticker): Path(path) for ticker, path in csv_inputs}
    if len(paths) != len(csv_inputs):
        raise DailyTrade40OracleBlockedError("TrueDaily normalized stock ticker collision")
    jobs = []
    for index, (ticker, selected) in enumerate(cohort.groupby("ticker", sort=True)):
        if ticker not in paths:
            raise DailyTrade40OracleBlockedError(f"TrueDaily stock missing price source: {ticker}")
        jobs.append((index, ticker, str(paths[ticker]), pd.DatetimeIndex(selected["date"])))
    frames, forced_count = _run_stocks(
        jobs, schedule=canonical_schedule, exchange=exchange,
        cutoff=research_end, workers=ticker_workers, progress=progress_callback,
    )
    outcome = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"], kind="stable")
    if outcome.empty or not outcome["realized_net_pnl"].notna().any():
        raise DailyTrade40OracleBlockedError("TrueDaily lacks settled NetPnL")
    scores = rank_trade40_outcome_frame(outcome, value_column="realized_net_pnl")
    if len(scores) != len(cohort):
        raise DailyTrade40OracleBlockedError("TrueDaily scoring widened/shrank DL cohort")
    raw_sha = sha256(outcome.to_csv(index=False, na_rep="NULL",
                                    float_format="%.17g").encode("utf-8")).hexdigest()
    rank_sha = sha256("".join(f"{t}|{d}|{scores[t,d]:.17g}\n"
                              for t, d in sorted(scores)).encode("ascii")).hexdigest()
    evidence = {
        "label_id": LABEL_ID, "target_id": TARGET_ID, "entry_policy": ENTRY_POLICY,
        "profile_id": profile_id, "key_mode": "information_date",
        "cohort": "true_daily_universal_hypothetical_min_fullcycle_netpnl",
        "mature_eligible_rows": len(outcome), "cutoff_truncated_rows": 0,
        "research_cutoff_closeout_rows": int(forced_count), "horizon_sessions": None,
        "evaluation_end_date": str(end.date()),
        "settled_net_pnl_rows": int(outcome["realized_net_pnl"].notna().sum()),
        "censored_rank_last_rows": int(outcome["realized_net_pnl"].isna().sum()),
        "first_date": outcome["date"].min(), "last_date": outcome["date"].max(),
        "first_replay_trade_date": str(future[0].date()),
        "research_cutoff": str(research_end.date()), "score_start_date": str(first.date()),
        "truth_source_sha256": truth_producer_fingerprint(),
        "dataset_summary_sha256": dataset_sha, "rank_sha256": rank_sha,
        "raw_label_sha256": raw_sha,
        "calendar_sha256": sha256("\n".join(exchange.strftime("%Y-%m-%d")).encode("ascii")).hexdigest(),
        "teacher": {"origin": "formal_min_roos_rolling_per_seed", "seed": int(seed),
                    "param_identity_sha256": str(param_sha256),
                    "periods": [{"start": str(p.start.date()),
                                 "end": None if p.end is None else str(p.end.date()),
                                 "params_signature": p.params_signature} for p in canonical_schedule]},
        **NET_PNL_EVIDENCE, "rank_policy": NET_PNL_RANK_POLICY,
        "oracle_evaluation_start_date": str(start.date()), "oracle_hindsight_only": True,
    }
    # Same completed ledger values feed both rank and Positive witnesses;
    # NULL rows remain unscored by the Positive head, never counted as losses.
    raw = {(row.ticker, row.date): float(row.realized_net_pnl)
           for row in outcome.itertuples(index=False, name="Row")}
    path, raw_identity = _raw_path(root, output_subdir, seed, price_sha256, param_sha256, evidence)
    positive_witness._save(path, raw_identity, raw)
    _truth_cache_write(cache_path, identity, scores, evidence)
    return scores, {**evidence, "source_reuse_status": "BUILD"}


__all__ = ["TARGET_ID", "LABEL_ID", "ENTRY_POLICY", "truth_producer_fingerprint",
           "load_positive_truth", "build_true_daily_universal_oracle_scores"]

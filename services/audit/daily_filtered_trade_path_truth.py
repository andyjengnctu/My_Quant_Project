"""Signed Oracle lifecycle for declared filtered Daily economic capabilities.

Only cohort, scheduling and evidence are orchestrated here. Price, admission,
trade outcomes, percentile mapping and broker accounting retain their owners.
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
from core.file_integrity import canonical_json_sha256, is_sha256_hex
from core.breakout_quality_registry import (
    get_breakout_quality_experiment_profile,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
)
from filters.breakout_quality.profile_ranker_data import (
    load_profile_continuous_ranker_data,
)
from filters.breakout_quality.workflow_io import (
    discover_dataset_csv_inputs,
    load_dataset_frame,
)
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.daily_trade40_target import (
    collect_canonical_exchange_sessions,
    rank_trade40_outcome_frame,
)
from filters.breakout_quality.daily_filtered_mintrade_target import (
    compute_daily_filtered_mintrade_batch,
)
from services.audit.daily_trade40_oracle import (
    _truth_cache_read,
    _truth_cache_write,
    DailyTrade40OracleBlockedError,
    NET_PNL_EVIDENCE,
    NET_PNL_RANK_POLICY,
)
from services.audit.oracle_score_artifacts import oracle_dataset_summary_sha256
from services.audit import fullcycle_positive_truth as witness

_WORKER_STATE = None


def _init_worker(schedule, exchange, cutoff):
    global _WORKER_STATE
    _WORKER_STATE = schedule, exchange, cutoff


def _stock_outcome(job, *, schedule, exchange, cutoff):
    index, ticker, path, dates = job
    frame = load_dataset_frame(Path(path), ticker, min_rows=1)
    positions = pd.DatetimeIndex(frame.index).normalize().get_indexer(dates)
    if (positions < 0).any():
        raise DailyTrade40OracleBlockedError(
            "Filtered Daily cohort lacks information-day prices"
        )
    batch = compute_daily_filtered_mintrade_batch(
        frame,
        positions,
        ticker=ticker,
        schedule=schedule,
        exchange_calendar=exchange,
        research_cutoff=cutoff,
    )
    if "no_historical_effective_Min_param" in batch.reasons:
        raise DailyTrade40OracleBlockedError(
            "Filtered Daily missing legal parameter teacher"
        )
    raw, valid = np.asarray(batch.target_raw_pnl), np.asarray(
        batch.valid_mask, dtype=bool
    )
    if ((valid & ~np.isfinite(raw)) | (~valid & np.isfinite(raw))).any():
        raise DailyTrade40OracleBlockedError("Filtered Daily NULL/outcome mismatch")
    return (
        index,
        pd.DataFrame(
            {
                "ticker": ticker,
                "date": dates.strftime("%Y-%m-%d"),
                "realized_net_pnl": raw,
            }
        ),
        int(batch.research_cutoff_closeout_rows),
    )


def _worker(job):
    if _WORKER_STATE is None:
        raise RuntimeError("Filtered Daily worker missing seed contract")
    return _stock_outcome(
        job,
        schedule=_WORKER_STATE[0],
        exchange=_WORKER_STATE[1],
        cutoff=_WORKER_STATE[2],
    )


def _raw_reference(root, subdir, seed, price_sha, param_sha, evidence):
    identity = witness._identity(
        seed=seed,
        price_sha=price_sha,
        param_sha=param_sha,
        source_evidence=evidence,
        project_root=root,
    )
    return witness._path(root, subdir, identity), identity


def load_filtered_positive_truth(
    *,
    spec,
    fingerprint,
    project_root,
    output_subdir,
    seed,
    price_sha256,
    param_sha256,
    canonical_scores,
    canonical_evidence,
    execution=None,
    cache_root=None,
):
    evidence = canonical_evidence
    effective = evidence.get("effective_source_reference")
    if effective:
        if execution is None or cache_root is None:
            raise DailyTrade40OracleBlockedError(
                "Effective Daily truth lacks its signed consumption context"
            )
        from services.audit.oracle_truth_references import OracleTruthReferences

        OracleTruthReferences(cache_root).validate_consumption(execution, evidence)
    teacher = evidence.get("teacher") or {}
    if (
        evidence.get("target_id") != spec["target_id"]
        or evidence.get("entry_policy") != spec["entry_policy"]
        or teacher.get("seed") != seed
        or (not effective and teacher.get("param_identity_sha256") != param_sha256)
        or teacher.get("origin") != spec["source_id"]
        or (not effective and evidence.get("truth_source_sha256") != fingerprint)
    ):
        raise DailyTrade40OracleBlockedError(
            "Filtered Daily Positive source/seed provenance mismatch"
        )
    _, identity = _raw_reference(
        Path(project_root), output_subdir, seed, price_sha256, param_sha256, evidence
    )
    found = witness._existing_raw_witness(Path(project_root), output_subdir, identity)
    raw = None if found is None else found[0]
    if (
        raw is None
        or not set(raw).issubset(canonical_scores)
        or len(raw) != evidence.get("settled_net_pnl_rows")
    ):
        raise DailyTrade40OracleBlockedError(
            "Filtered Daily signed Positive witness missing/incompatible"
        )
    return raw


def build_filtered_daily_oracle_scores(
    *,
    spec,
    fingerprint,
    project_root,
    dataset,
    filter_id,
    profile_id,
    cutoff,
    evaluation_start,
    evaluation_end,
    score_start,
    seed,
    param_sha256,
    canonical_schedule,
    price_sha256,
    cache_root,
    output_subdir,
    feature_cohort_cache=None,
    exchange_calendar_cache=None,
    ticker_workers=1,
    progress_callback=None,
):
    root = Path(project_root).resolve()
    profile = get_breakout_quality_experiment_profile(profile_id)
    start, end, first, limit = map(
        lambda v: pd.Timestamp(v).normalize(),
        (evaluation_start, evaluation_end, score_start, cutoff),
    )
    if (
        type(seed) is not int
        or seed <= 0
        or not canonical_schedule
        or any(p.source_id != spec["source_id"] for p in canonical_schedule)
        or not is_sha256_hex(param_sha256)
        or not is_sha256_hex(price_sha256)
        or not start <= first <= end <= limit
        or type(ticker_workers) is not int
        or not 1 <= ticker_workers <= 8
        or profile.training_sample_scope
        != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    ):
        raise DailyTrade40OracleBlockedError(
            "Filtered Daily canonical cohort/seed/schedule contract invalid"
        )
    dataset_sha = oracle_dataset_summary_sha256(root=root, filter_id=filter_id)
    key = (
        str(root),
        dataset,
        filter_id,
        profile_id,
        str(first.date()),
        str(end.date()),
        dataset_sha,
    )
    cached_cohort = (
        None if feature_cohort_cache is None else feature_cohort_cache.get(key)
    )
    if cached_cohort is None:
        bundle = load_profile_continuous_ranker_data(
            filter_id=filter_id,
            model_architecture=profile.model_architecture,
            experiment_profile=profile_id,
            preload_feature_bank=False,
            allow_stale_source=False,
            project_root=root,
        )
        summary = bundle.summary or {}
        cohort = pd.DataFrame(bundle.group_table).reset_index(drop=True)
        if (
            summary.get("dataset") != dataset
            or summary.get("filter_id") != filter_id
            or summary.get("sample_scope") != "daily_eligible_stock_days"
            or len(cohort) != summary.get("score_eligible_sample_count")
            or not {"ticker", "date"}.issubset(cohort)
        ):
            raise DailyTrade40OracleBlockedError(
                "Filtered Daily cohort differs from canonical DL eligibility"
            )
        cohort["ticker"] = cohort["ticker"].map(normalize_ticker)
        cohort["date"] = pd.to_datetime(cohort["date"], errors="raise").dt.normalize()
        cohort = cohort.loc[cohort["date"].between(first, end)].reset_index(drop=True)
        if cohort.empty or cohort.duplicated(["ticker", "date"]).any():
            raise DailyTrade40OracleBlockedError(
                "Filtered Daily empty/duplicated cohort"
            )
        cohort_sha = sha256(
            cohort[["ticker", "date"]]
            .sort_values(["ticker", "date"], kind="stable")
            .to_csv(index=False)
            .encode()
        ).hexdigest()
        if feature_cohort_cache is not None:
            feature_cohort_cache[key] = cohort, cohort_sha
    else:
        cohort, cohort_sha = cached_cohort
    periods = [
        {
            "start": str(p.start.date()),
            "end": None if p.end is None else str(p.end.date()),
            "params_signature": p.params_signature,
        }
        for p in canonical_schedule
    ]
    identity = {
        "schema": 1,
        "target_id": spec["target_id"],
        "seed": seed,
        "param_sha256": param_sha256,
        "periods": periods,
        "price_sha256": price_sha256,
        "profile_id": profile_id,
        "dataset": dataset,
        "filter_id": filter_id,
        "dataset_summary_sha256": dataset_sha,
        "feature_eligible_keys_sha256": cohort_sha,
        "evaluation_start": str(start.date()),
        "evaluation_end": str(end.date()),
        "score_start": str(first.date()),
        "research_cutoff": str(limit.date()),
        "truth_source_sha256": fingerprint,
        "entry_policy": spec["entry_policy"],
    }
    path = (
        Path(cache_root)
        / str(seed)
        / f"{spec['target_id']}_{canonical_json_sha256(identity)}.npz"
    )
    cached = _truth_cache_read(path, identity)
    if cached is not None:
        load_filtered_positive_truth(
            spec=spec,
            fingerprint=fingerprint,
            project_root=root,
            output_subdir=output_subdir,
            seed=seed,
            price_sha256=price_sha256,
            param_sha256=param_sha256,
            canonical_scores=cached[0],
            canonical_evidence=cached[1],
        )
        if progress_callback:
            progress_callback(
                f"Filtered Daily truth REUSE ({len(cached[0]):,} stock-days)"
            )
        return cached
    inputs, duplicates = discover_dataset_csv_inputs(root, dataset)
    paths = {normalize_ticker(t): Path(p) for t, p in inputs}
    if duplicates or len(paths) != len(inputs):
        raise DailyTrade40OracleBlockedError("Filtered Daily duplicate price inputs")
    calendar_key = str(root), dataset, price_sha256
    exchange = (
        None
        if exchange_calendar_cache is None
        else exchange_calendar_cache.get(calendar_key)
    )
    if exchange is None:
        exchange = collect_canonical_exchange_sessions(
            inputs,
            benchmark_ticker=str(DEFAULT_LABEL_POLICY.benchmark_ticker),
            horizon_sessions=None,
        )
        if exchange_calendar_cache is not None:
            exchange_calendar_cache[calendar_key] = exchange
    if (
        (exchange.get_indexer(pd.DatetimeIndex(cohort["date"])) < 0).any()
        or not (exchange < start).any()
        or not (exchange >= start).any()
    ):
        raise DailyTrade40OracleBlockedError(
            "Filtered Daily calendar/OOS boundary invalid"
        )
    jobs = []
    for index, (ticker, selected) in enumerate(cohort.groupby("ticker", sort=True)):
        if ticker not in paths:
            raise DailyTrade40OracleBlockedError(
                f"Filtered Daily missing stock price source: {ticker}"
            )
        jobs.append(
            (index, ticker, str(paths[ticker]), pd.DatetimeIndex(selected["date"]))
        )
    if progress_callback:
        progress_callback(f"Filtered Daily truth BUILD ({len(cohort):,} stock-days)")
    frames, forced = [], 0
    pool = None
    try:
        if ticker_workers == 1 or len(jobs) < 2:
            results = (
                _stock_outcome(
                    j, schedule=canonical_schedule, exchange=exchange, cutoff=limit
                )
                for j in jobs
            )
        else:
            pool = ProcessPoolExecutor(
                max_workers=min(ticker_workers, len(jobs)),
                mp_context=get_context("spawn"),
                initializer=_init_worker,
                initargs=(canonical_schedule, exchange, limit),
            )
            results = pool.map(_worker, jobs)
        for index, (job_id, frame, count) in enumerate(results):
            if index != job_id:
                raise DailyTrade40OracleBlockedError(
                    "Filtered Daily unstable worker order"
                )
            frames.append(frame)
            forced += count
            if progress_callback and ((index + 1) % 100 == 0 or index + 1 == len(jobs)):
                progress_callback(f"Filtered Daily stock paths {index+1}/{len(jobs)}")
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    outcome = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "ticker"], kind="stable"
    )
    if len(outcome) != len(cohort) or not outcome["realized_net_pnl"].notna().any():
        raise DailyTrade40OracleBlockedError("Filtered Daily outcome/cohort incomplete")
    scores = rank_trade40_outcome_frame(outcome, value_column="realized_net_pnl")
    evidence = {
        "label_id": f"LABEL-{spec['target_id']}",
        "target_id": spec["target_id"],
        "entry_policy": spec["entry_policy"],
        "profile_id": profile_id,
        "key_mode": "information_date",
        "cohort": "filtered_daily_hypothetical_fullcycle_netpnl",
        "mature_eligible_rows": len(outcome),
        "cutoff_truncated_rows": 0,
        "research_cutoff_closeout_rows": int(forced),
        "horizon_sessions": None,
        "evaluation_end_date": str(end.date()),
        "settled_net_pnl_rows": int(outcome["realized_net_pnl"].notna().sum()),
        "censored_rank_last_rows": int(outcome["realized_net_pnl"].isna().sum()),
        "first_date": outcome["date"].min(),
        "last_date": outcome["date"].max(),
        "first_replay_trade_date": str(exchange[exchange >= start][0].date()),
        "research_cutoff": str(limit.date()),
        "score_start_date": str(first.date()),
        "truth_source_sha256": fingerprint,
        "dataset_summary_sha256": dataset_sha,
        "rank_sha256": sha256(
            "".join(f"{t}|{d}|{scores[t,d]:.17g}\n" for t, d in sorted(scores)).encode()
        ).hexdigest(),
        "raw_label_sha256": sha256(
            outcome.to_csv(index=False, na_rep="NULL", float_format="%.17g").encode()
        ).hexdigest(),
        "calendar_sha256": sha256(
            "\n".join(exchange.strftime("%Y-%m-%d")).encode()
        ).hexdigest(),
        "teacher": {
            "origin": spec["source_id"],
            "optimizer_seed": seed,
            "seed": seed,
            "param_identity_sha256": param_sha256,
            "periods": periods,
        },
        **NET_PNL_EVIDENCE,
        "rank_policy": NET_PNL_RANK_POLICY,
        "oracle_evaluation_start_date": str(start.date()),
        "oracle_hindsight_only": True,
    }
    raw = {(r.ticker, r.date): float(r.realized_net_pnl) for r in outcome.itertuples()}
    raw_path, raw_id = _raw_reference(
        root, output_subdir, seed, price_sha256, param_sha256, evidence
    )
    witness._save(raw_path, raw_id, raw)
    _truth_cache_write(path, identity, scores, evidence)
    return scores, {**evidence, "source_reuse_status": "BUILD"}

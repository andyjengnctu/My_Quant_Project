"""Hindsight daily-universal Min trade-path score source for the formal Oracle.

Feature eligibility comes from the existing generic daily model dataset, NOT
from the retired MR-13CL label. Each seed's horizon-specific trades
consume the identical DL-off parameters used by that seed's Min Portfolio.
Exact/Direct selectors read the same per-seed score table; neither can train.
"""
from __future__ import annotations

from hashlib import sha256
from functools import lru_cache
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from core.file_integrity import (
    canonical_json_sha256, is_sha256_hex,
    atomic_replace_with_retry, exclusive_file_lock,
)

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
from services.audit.oracle_score_artifacts import (
    oracle_daily_source_fingerprint, oracle_dataset_summary_sha256,
)
from services.audit.mintrade_truth_source_attestation import attested_mintrade_source_sha256

# Old MR-13CL's P2/P16 single-teacher label is a distinct retired scientific
# identity; no new result may masquerade as its supervised-label artifact.
TARGET_ID = "daily_universal_formal_min_roos_trade40_netr_v1"
LABEL_ID = "LABEL-daily_universal_formal_min_roos_trade40_netr_v1"
# One owner for the monetary-outcome evidence, not a second economic model.
NET_PNL_EVIDENCE = {
    "sizing_basis": "canonical_single_backtest_initial_capital_not_portfolio_dynamic_equity",
    "economic_outcome": "broker_ledger_realized_net_pnl_twd",
}
NET_PNL_RANK_POLICY = "average_zero_based_completed_net_pnl_censored_rank_last_no_pnl_imputation"


def target_identity_for_horizon(
    horizon_sessions: int | None, *, outcome_metric: str = "net_r",
) -> tuple[str, str]:
    if outcome_metric not in {"net_r", "net_pnl"}:
        raise DailyTrade40OracleBlockedError("Unknown formal Min Oracle economic outcome")
    if horizon_sessions is not None and (
        not isinstance(horizon_sessions, int) or isinstance(horizon_sessions, bool) or horizon_sessions <= 0
    ):
        raise DailyTrade40OracleBlockedError("Invalid formal Min Oracle session horizon")
    # Holding horizon and broker-ledger outcome are independent capabilities.
    # The Audit Registry alone authorizes which combinations are compared.
    horizon_id = "fulltrade" if horizon_sessions is None else f"trade{horizon_sessions}"
    outcome_id = "netpnl" if outcome_metric == "net_pnl" else "netr"
    target = f"daily_universal_formal_min_roos_{horizon_id}_{outcome_id}_v1"
    return target, f"LABEL-{target}"


class DailyTrade40OracleBlockedError(ValueError):
    """A source, maturity or per-seed formal Min identity failed verification."""


class DailyTrade40TruthBuildRequired(RuntimeError):
    """Read-only preflight found no full-identity truth artifact; do not BUILD yet."""


# This is a persistent Target-truth cache, never a Replay or model score cache.
# Every key covers actual price bytes, feature-eligible ticker/date membership,
# formal rolling P16 parameters, horizon, PIT date bounds and source code lineage.
# Missing old truth artifacts must be computed once; a Replay observation alone
# cannot reconstruct or independently certify the underlying perfect scores.
_TRUTH_CACHE_SCHEMA = 1
_TRUTH_SOURCE_MODULES = (
    "services/audit/daily_trade40_oracle.py",
    "filters/breakout_quality/daily_trade40_target.py",
    "filters/breakout_quality/trade_path_label.py",
    "filters/breakout_quality/runtime.py",
    "core/backtest_finalize.py",
    "core/position_step.py",
    "core/position_management.py",
    "core/exit_priority.py",
    "core/entry_plans.py",
    "core/extended_signals.py",
    "core/order_lot_policy.py",
    "core/exact_accounting.py",
    "core/fee_rebate.py",
    "core/price_utils.py",
    "core/strategy_params.py",
    "strategies/breakout/schema.py",
    "core/selection_policy.py",
    "core/signal_utils.py",
    "core/breakout_false_filter.py",
    "core/feature_bank.py",
    "core/native_optimizer_hotpath.py",
    "core/native_optimizer_hotpath.c",
    "core/training_performance.py",
    "core/portfolio_param_runtime.py",
    "core/params_io.py",
    "core/active_param_ensemble.py",
    "core/rolling_oos_params.py",
    "core/seed_ensemble_policy.py",
    "core/capital_policy.py",
    "config/broker_accounting.py",
    "config/breakout_policy.py",
    "config/execution_policy.py",
    "config/training_performance_policy.py",
    "config/trading.py",
)


def _truth_source_sha(root: Path, *, outcome_metric: str = "net_r") -> str:
    if outcome_metric not in {"net_r", "net_pnl"}:
        raise DailyTrade40OracleBlockedError("Unknown formal Min Oracle outcome for source identity")
    # All MinTrade outcomes share the actual canonical execution source. A
    # substantive signal/order-state correction MUST invalidate historical
    # Net-R and NetPnL truth, never inherit a pre-fix compatibility fingerprint.
    digest = sha256()
    # The daily feature/population closure has one owner shared with price
    # Targets. Entry, management, sizing and ledger owners below add only the
    # Min execution semantics, including their delegated Python/native code.
    digest.update(b"canonical_daily_source\0")
    digest.update(oracle_daily_source_fingerprint(root).encode("ascii") + b"\n")
    for relative in _TRUTH_SOURCE_MODULES:
        path = root / relative
        if not path.is_file():
            raise DailyTrade40OracleBlockedError(f"Oracle truth source lineage missing: {relative}")
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(attested_mintrade_source_sha256(relative, path).encode("ascii") + b"\n")
    return digest.hexdigest()


@lru_cache(maxsize=2)
def truth_producer_fingerprint(outcome_metric: str = "net_r") -> str:
    """Fingerprint the executing canonical outcome producer without any label RUN.

    This is the same source closure used by the durable truth-cache identity.
    The imported module root (not a data project_root) identifies the code
    actually executing; hashing this small source closure never simulates trades.
    """
    return _truth_source_sha(Path(__file__).resolve().parents[2], outcome_metric=outcome_metric)


def _ranked_score_sha(scores: Mapping[tuple[str, str], float]) -> str:
    digest = sha256()
    for (ticker, day), value in sorted(scores.items()):
        score = float(value)
        if not np.isfinite(score) or score < 0.0 or score > 1.0:
            raise DailyTrade40OracleBlockedError("Oracle cached score is outside finite [0,1]")
        digest.update(f"{ticker}|{day}|{score:.17g}\n".encode("ascii"))
    return digest.hexdigest()


def _truth_cache_read(path: Path, identity: Mapping[str, Any]):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "tickers", "days", "scores"}:
                raise ValueError("unexpected cached truth payload fields")
            meta = json.loads(str(archive["metadata"].item()))
            tickers = archive["tickers"]
            days = archive["days"]
            values = archive["scores"]
        if meta.get("schema") != _TRUTH_CACHE_SCHEMA or meta.get("identity") != dict(identity):
            raise ValueError("truth cache identity mismatch")
        # The raw Net-R outcomes cannot be recovered from percentile scores.
        # Preserve their evidence as a single immutable, checksummed payload;
        # never allow a valid rank table with altered provenance to REUSE.
        payload = {name: meta.get(name) for name in ("schema", "identity", "evidence", "score_sha256")}
        if meta.get("payload_sha256") != canonical_json_sha256(payload):
            raise ValueError("truth cache evidence payload SHA mismatch")
        if (tickers.ndim != 1 or days.ndim != 1 or values.ndim != 1 or
                not len(values) or len(tickers) != len(values) or len(days) != len(values)):
            raise ValueError("truth cache data shape mismatch")
        if (tickers.dtype.kind != "U" or days.dtype.kind != "U" or
                values.dtype.kind != "f"):
            raise ValueError("truth cache data type mismatch")
        scores = dict(zip(zip(tickers.tolist(), days.tolist()), values.tolist()))
        evidence = meta["evidence"]
        teacher = dict(evidence["teacher"])
        if (evidence["target_id"] != identity["target_id"] or
                evidence["profile_id"] != identity["profile_id"] or
                evidence.get("truth_source_sha256") != identity["truth_source_sha256"] or
                "dataset_summary_sha256" not in evidence or
                evidence["dataset_summary_sha256"] != identity["dataset_summary_sha256"] or
                evidence["research_cutoff"] != identity["research_cutoff"] or
                evidence["score_start_date"] != identity["score_start"] or
                teacher.get("seed") != identity["seed"] or
                teacher.get("param_identity_sha256") != identity["param_sha256"] or
                len(scores) != len(values) or
                len(scores) != int(evidence["mature_eligible_rows"]) or
                _ranked_score_sha(scores) != evidence["rank_sha256"] or
                meta.get("score_sha256") != evidence["rank_sha256"]):
            raise ValueError("truth cache value/lineage SHA/row count mismatch")
        return scores, {**evidence, "source_reuse_status": "REUSE"}
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, zipfile.BadZipFile) as exc:
        raise DailyTrade40OracleBlockedError(
            f"Oracle Target-truth cache is corrupt and cannot be REUSE: {path.name}: {exc}"
        ) from exc


def _truth_cache_write(path: Path, identity: Mapping[str, Any],
                       scores: Mapping[tuple[str, str], float], evidence: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = {k: v for k, v in evidence.items() if k != "source_reuse_status"}
    if _ranked_score_sha(scores) != evidence["rank_sha256"]:
        raise DailyTrade40OracleBlockedError("Oracle truth source rank hash not reproducible")
    payload = {
        "schema": _TRUTH_CACHE_SCHEMA, "identity": identity,
        "evidence": evidence, "score_sha256": evidence["rank_sha256"],
    }
    metadata = json.dumps({**payload, "payload_sha256": canonical_json_sha256(payload)},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    keys = sorted(scores)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream, metadata=np.asarray(metadata),
                tickers=np.asarray([ticker for ticker, _ in keys]),
                days=np.asarray([day for _, day in keys]),
                scores=np.asarray([scores[key] for key in keys], dtype=np.float64),
            )
            stream.flush()
            os.fsync(stream.fileno())
        with exclusive_file_lock(path.parent / ".truth_cache.publish.lock"):
            previous = _truth_cache_read(path, identity)
            if previous is not None:
                if previous[0] != dict(scores) or {
                        k: v for k, v in previous[1].items() if k != "source_reuse_status"
                    } != evidence:
                    raise DailyTrade40OracleBlockedError("Oracle same-identity truth evidence or scores differ")
            else:
                atomic_replace_with_retry(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def build_daily_universal_trade40_oracle_scores(
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
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Compute all feature-eligible stock-day outcomes from a formal Min seed.

    Only information dates covered by the canonical PIT DL source are scored;
    the pre-OOS information bar may still be used to locate the first trade
    session, but cannot acquire a hindsight score outside the PIT score period.
    For each hypothetical entry, Min parameters become effective on the next
    observed *ticker* order date, matching Min signal->order semantics. Only
    the declared horizon and Research data cutoff are both imposed on the
    canonical trade path as earliest legal closeout boundaries.
    """
    target_id, label_id = target_identity_for_horizon(
        horizon_sessions, outcome_metric=outcome_metric,
    )
    value_column = "realized_net_pnl" if outcome_metric == "net_pnl" else "realized_net_r"
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
                "truth_source_sha256": truth_producer_fingerprint(outcome_metric),
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
        exchange = collect_canonical_exchange_sessions(
            csv_inputs, benchmark_ticker=str(DEFAULT_LABEL_POLICY.benchmark_ticker),
            horizon_sessions=horizon_sessions,
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
        daily_calendar_pos = exchange.get_indexer(pd.DatetimeIndex(group["date"]))
        if (daily_calendar_pos < 0).any():
            raise DailyTrade40OracleBlockedError("Feature stock-day absent from canonical exchange calendar")
        # Eligible DL score rows must NEVER depend on the Target's future
        # holding period. The Research cutoff controls the outcome's final
        # closeout, not whether the date is allowed in the score cross-section.
        if not len(exchange[exchange <= cutoff_day]) or group.empty:
            raise DailyTrade40OracleBlockedError("No Research-cutoff feature stock-days")
        records: list[pd.DataFrame] = []
        total_truncated = 0
        total_cutoff_closeouts = 0
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
                exchange_calendar=exchange, horizon_sessions=horizon_sessions,
                research_cutoff=cutoff_day,
                include_net_pnl=(outcome_metric == "net_pnl"),
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
            # The only allowed deadline is the earlier of horizon and research
            # cutoff. Cross-ticker calendars are verified by the source owner.
            if len(batch.maturity_dates) != len(selected):
                raise DailyTrade40OracleBlockedError("Min target deadline result shape mismatch")
            total_truncated += batch.cutoff_truncated_rows
            total_cutoff_closeouts += batch.research_cutoff_closeout_rows
            raw = np.asarray(
                batch.target_raw_pnl if outcome_metric == "net_pnl" else batch.target_raw_r,
                dtype=np.float64,
            )
            valid = np.asarray(batch.valid_mask, dtype=bool)
            if (len(raw) != len(valid) or
                    (valid & ~np.isfinite(raw)).any() or ((~valid) & np.isfinite(raw)).any()):
                raise DailyTrade40OracleBlockedError("NULL/valid canonical Min economic outcome invalid")
            records.append(pd.DataFrame({
                "ticker": ticker,
                "date": selected["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                value_column: np.where(valid, raw, np.nan),
            }))
            if progress_callback is not None and (tick_index % 100 == 0 or tick_index == total_tickers):
                progress_callback(f"逐股交易路徑 {tick_index}/{total_tickers}")
        outcome = pd.concat(records, ignore_index=True).sort_values(["date", "ticker"], kind="stable")
        if outcome.empty or not outcome[value_column].notna().any():
            raise DailyTrade40OracleBlockedError("No completed valid formal-Min economic outcomes")
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
            "origin": "formal_min_roos_rolling_per_seed", "seed": int(seed),
            "param_identity_sha256": str(param_sha256),
            "periods": [{"start": str(p.start.date()),
                         "end": None if p.end is None else str(p.end.date()),
                         "params_signature": p.params_signature} for p in canonical_schedule],
        }
        evidence = {
            "label_id": label_id, "target_id": target_id,
            "profile_id": profile_id, "key_mode": "information_date",
            "cohort": "daily_eligible_stock_days_formal_min_research_cutoff_closeout",
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
            "truth_source_sha256": truth_producer_fingerprint(outcome_metric),
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
            f"Daily-universal formal Min source blocked: {type(exc).__name__}: "
            f"{str(exc).replace(str(root), '.')}"
        ) from exc


__all__ = ["TARGET_ID", "LABEL_ID", "target_identity_for_horizon", "DailyTrade40OracleBlockedError",
           "build_daily_universal_trade40_oracle_scores", "truth_producer_fingerprint"]

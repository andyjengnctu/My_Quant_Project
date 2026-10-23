"""Reusable hindsight target economics Audit; NEVER a legal prediction source.

This audit compares future-outcome stock-day rankings by running the same frozen
Broker-Cash P16 Exact K/R0 portfolio replay. All historical truths are *ex post*.
No model training, PIT score publication, optimizer, or trading state is touched.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
import math

import numpy as np
import pandas as pd

from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
    CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.console_report import render_table, render_title
from core.dataset_profiles import get_dataset_dir
from core.path_utils import project_relative_display_path
from core.file_integrity import canonical_json_sha256
from core.strategy_compare_policy import (
    get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
)
from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame
from filters.breakout_quality.continuous_target import (
    StrategyAlignedContinuousTargetSpec,
    full_horizon_mae_safety_r_from_minimum_low_return,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.mfe_safety_geometry import (
    ensure_daily_percentile, normalize_ticker,
)
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.runtime import audit_oracle_truth_score_context
from filters.breakout_quality.strategy_compare_contracts import COMPARISON_MODE_SCORE_RANKING
from filters.breakout_quality.strategy_compare_reporting import _scenario_summary, _yearly_frame, _to_json_native
from filters.breakout_quality.strategy_compare_sources import (
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    _build_controlled_param_source_pair,
    _validate_requested_param_policy,
    load_strategy_param_evaluation_view,
)
from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES
from services.audit.reusable_report import persist_reusable_report
from services.audit.oracle_replay_artifacts import (
    HistoricalOracleIndex, OracleReplayStore, price_inventory_sha256, replay_identity,
)
from services.optimizer.strategy_param_service import (
    inspect_fixed_multiseed_robustness_strategy_parameter_artifact,
)
from config.execution_policy import (
    DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT, EXECUTION_POLICY_PARAM_SPECS,
)
from config.broker_accounting import (
    BROKER_ACCOUNTING_POLICY_ID, BROKER_FEE_RATE, BROKER_MIN_FEE_TWD,
    BROKER_ODD_LOT_MIN_FEE_TWD, BROKER_BOARD_LOT_SHARES,
    BROKER_STOCK_TAX_RATE, BROKER_CHARGE_ROUNDING_MODE,
)
from config.breakout_quality import BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
from core.active_param_ensemble import get_active_param_ensemble_date_range
from core.rolling_oos_params import get_active_param_date_range
from services.research.strategy_compare_replay import _run_scenario

from core.research_report_contract import (
    report_contract, require_approved_report_contract,
    persistent_report_contract_fingerprint,
)
from services.audit.reusable_report import format_contract_row

SUPPORTED_AUDIT_TYPE = "target_ground_truth_strategy_romd"
ORACLE_REPORT_SCHEMA_VERSION = 2
ORACLE_TARGETS = (
    ("AO40_EqualRank", "40D MFE + adverse-to-peak Safety, 50/50 daily rank"),
    ("CG40_EqualRank", "40D MFE + full D40 Safety, 50/50 daily rank"),
    ("CH20_EqualRank", "20D MFE + full D20 Safety, 50/50 daily rank"),
    ("U40", "40D MFE - D40, canonical R"),
    ("U20", "20D MFE - D20, canonical R"),
    ("FirstBreach_NetR", "MFE before first risk breach - adverse to that peak, R"),
    ("MFE40_Control", "40D MFE only (control, not target recommendation)"),
    ("MFE40_over_D40", "Exact MFE40 / D40 in canonical R; if D40=0 and MFE>0 rank as +infinity, 0/0 ranks as zero"),
    ("MFE20", "20D maximum favorable excursion, in canonical R"),
    ("MAE40", "Negative full-horizon 40D MAE (lower adverse excursion ranks higher), in canonical R"),
    ("MAE20", "Negative full-horizon 20D MAE (lower adverse excursion ranks higher), in canonical R"),
)


class OracleBlockedError(RuntimeError):
    """Insufficient or incompatible historical truth; do not publish a result."""


def _contract(definition) -> tuple[Any, Any, Any]:
    source = definition.source
    if definition.audit_type != SUPPORTED_AUDIT_TYPE:
        raise ValueError("Oracle Audit identity不一致")
    settings = get_strategy_comparison_settings(str(source["strategy_profile_id"]))
    base = settings.arms[str(source["baseline_arm_id"])]
    exact = settings.arms[str(source["exact_arm_id"])]
    if (
        base.dl_enabled or not exact.dl_enabled
        or base.param_source != exact.param_source
        or base.param_policy != exact.param_policy
        or base.rule_policy != "all_off" or exact.rule_policy != "all_off"
        or not settings.start_date or not settings.end_date
        or settings.dataset != "full" or settings.rotation != "off"
        or not exact.dl_runtime_options
        or not exact.dl_runtime_options.get("preserve_k_r0")
        or exact.dl_runtime_options.get("constrained_solver") != "exact_branch_and_bound_v1"
        or not exact.dl_runtime_options.get("selection_only")
    ):
        raise OracleBlockedError("Broker-Cash C96/C100 Exact K/R0 control contract已漂移")
    if base.param_policy != "base-finalist-best":
        raise OracleBlockedError("Oracle Audit需要canonical P16 Base-Finalist-Best")
    return settings, base, exact


def _robustness_artifacts(definition, root: Path):
    settings, baseline, exact = _contract(definition)
    robustness = get_strategy_multi_seed_robustness_settings(settings.profile_id)
    source = settings.parameter_sources[str(baseline.param_source)]
    builder = source.builder
    if (
        robustness.profile_id != settings.profile_id
        or not robustness.enabled or not robustness.benchmark_id
        or not robustness.resolved_seeds
        or len(robustness.resolved_seeds) != robustness.seed_count
        or builder is None or not builder.enabled
        or builder.builder_type != "optimizer_fixed_multiseed_strategy_params"
    ):
        raise OracleBlockedError("Oracle Benchmark無法對齊canonical 4-seed P16訓練合約")
    training_profile = str(dict(builder.options).get("training_profile_id") or "")
    if not training_profile:
        raise OracleBlockedError("P16缺少canonical training profile")
    records = []
    for seed in robustness.resolved_seeds:
        inspection = inspect_fixed_multiseed_robustness_strategy_parameter_artifact(
            root,
            training_profile_id=training_profile,
            benchmark_id=str(robustness.benchmark_id),
            replication_seed=int(seed),
            policy=str(baseline.param_policy),
            comparison_end_date=str(settings.end_date),
            dataset=settings.dataset,
            max_positions=settings.max_positions,
            rotation=settings.rotation,
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
        if not bool(inspection["ready"]):
            raise OracleBlockedError(
                f"Benchmark P16 seed={seed}非READY: {inspection['status']}（只讀Audit不自動重訓）"
            )
        records.append((int(seed), Path(inspection["path"])))
    return robustness, records


def collect_status(definition, *, project_root: Path) -> dict[str, Any]:
    """Read-only 4-seed P16 manifest and full Dataset preflight."""
    root = Path(project_root).resolve()
    display = "Full Research Dataset + 4-seed canonical P16 + daily target truth"
    try:
        settings, _base, _exact = _contract(definition)
        data_dir = Path(get_dataset_dir(str(root), settings.dataset)).resolve()
        if not data_dir.is_dir() or not next(data_dir.rglob("*.csv"), None):
            raise OracleBlockedError("缺少完整研究價格資料（full dataset）")
        from filters.breakout_quality.paths import resolve_filter_output_dir
        truth_root = resolve_filter_output_dir(root, filter_id=str(definition.source["filter_id"]))
        if not truth_root.is_dir():
            raise OracleBlockedError("缺少validated breakout-quality Dataset bundle；Audit不得代替Producer建置")
        robustness, records = _robustness_artifacts(definition, root)
        return {
            "status": "READY", "reason": "", "source": {
                "display": display, "preparable": False,
                "benchmark": robustness.benchmark_id,
                "seeds": [seed for seed, _path in records],
            },
        }
    except (ValueError, OSError, OracleBlockedError) as exc:
        return {
            "status": "BLOCKED", "reason": str(exc),
            "source": {"display": display, "preparable": False},
        }



_ORACLE_PROVIDER_FIELDS = ("pure_mfe40_provider", "mfe20_d20_provider", "first_breach_provider")


def _market_truth_for_oracle(
    definition, root: Path, *, cutoff: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Materialize ex-post truth for the *replay* price universe, not DL samples.

    The canonical daily-universal provider is deliberately feature-window and
    benchmark restricted.  Neither restriction defines which stocks the
    strategy can rank.  Reuse its canonical target calculator on each sanitized
    Research price path, including newly listed stocks and the benchmark.
    Invalid/unmatured paths never receive a score or a proxy value.
    """
    settings, _baseline, _exact = _contract(definition)
    specs = {}
    for field in _ORACLE_PROVIDER_FIELDS:
        profile_id = str(definition.source[field])
        profile = get_breakout_quality_experiment_profile(profile_id)
        recipe = get_continuous_ranker_execution_recipe(profile_id)
        policy = recipe.target_policy
        if (
            policy.materialization_mode != CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT
            or policy.postprocess != CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE
            or not policy.component_target_id
            or recipe.continuous_target_id != profile.continuous_target_id
        ):
            raise OracleBlockedError(f"{field}不能以canonical daily target calculator計算")
        horizon = int(policy.horizon_bars or DEFAULT_LABEL_POLICY.label_horizon_bars)
        maturity = int(policy.sample_universe_horizon_bars or horizon)
        if horizon <= 0 or maturity < horizon:
            raise OracleBlockedError(f"{field}的horizon／maturity contract不合法")
        specs[field] = (
            StrategyAlignedContinuousTargetSpec.from_label_policy(
                replace(DEFAULT_LABEL_POLICY, label_horizon_bars=horizon)
            ),
            maturity,
            str(policy.component_target_id),
        )

    csv_inputs, duplicates = discover_dataset_csv_inputs(root, settings.dataset)
    if duplicates:
        raise OracleBlockedError(f"Oracle Research Dataset含重複ticker來源：{duplicates[:3]}")
    paths = {}
    for ticker, path in csv_inputs:
        key = normalize_ticker(ticker)
        if key in paths:
            raise OracleBlockedError(f"Oracle Research Dataset正規化ticker重複：{key}")
        paths[key] = Path(path)
    benchmark = normalize_ticker(DEFAULT_LABEL_POLICY.benchmark_ticker)
    if benchmark not in paths:
        raise OracleBlockedError(f"Oracle缺少可交易Benchmark價格：{benchmark}")

    # Portfolio simulation can request yesterday's information on its first day.
    # Use the last actual benchmark trading day before the replay start; no
    # arbitrary calendar lookback or DL feature-window cutoff is imposed.
    bench_frame = load_dataset_frame(paths[benchmark], benchmark, min_rows=1)
    historical_dates = pd.DatetimeIndex(bench_frame.index).normalize()
    prior = historical_dates[historical_dates < pd.Timestamp(settings.start_date)]
    if prior.empty:
        raise OracleBlockedError("Oracle缺少策略起始日前的Benchmark交易日")
    score_start = prior[-1]
    limit = pd.Timestamp(cutoff)
    chunks = {field: [] for field in _ORACLE_PROVIDER_FIELDS}
    skipped = {}
    valid_counts = {field: 0 for field in _ORACLE_PROVIDER_FIELDS}
    for ticker, path in paths.items():
        try:
            frame = (bench_frame if ticker == benchmark else load_dataset_frame(path, ticker, min_rows=1))
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
            # Keep diagnostics traceable without storing host-absolute data paths.
            reason = str(exc).replace(str(path), path.name).replace(str(root), ".")
            skipped[ticker] = f"{type(exc).__name__}: {reason}"
            continue
        dates = pd.DatetimeIndex(frame.index).normalize()
        positions = np.flatnonzero((dates >= score_start) & (dates <= limit)).astype(np.int64)
        if not positions.size:
            continue
        for field, (spec, maturity, target_id) in specs.items():
            eligible = positions[positions + maturity < len(frame)]
            if not eligible.size:
                continue
            eligible = eligible[dates.take(eligible + maturity) <= limit]
            if not eligible.size:
                continue
            batch = compute_daily_opportunity_target_batch(
                frame, eligible, spec=spec, target_id=target_id,
            )
            valid = np.asarray(batch.valid_mask, dtype=bool)
            if maturity != int(spec.horizon_bars):
                cohort = compute_daily_opportunity_target_batch(
                    frame, eligible, spec=replace(spec, horizon_bars=maturity),
                    target_id=target_id,
                )
                valid &= np.asarray(cohort.valid_mask, dtype=bool)
            risk = float(spec.risk_budget_return)
            values = np.column_stack((
                np.asarray(batch.favorable_return, dtype=float) / risk,
                np.asarray(batch.adverse_return_to_peak, dtype=float) / risk,
                np.asarray(batch.minimum_low_return, dtype=float),
                np.asarray(batch.target_raw_r, dtype=float),
            ))
            valid &= np.isfinite(values).all(axis=1)
            if not valid.any():
                continue
            valid_counts[field] += int(valid.sum())
            chunks[field].append(pd.DataFrame({
                "ticker": ticker,
                "date": dates.take(eligible[valid]).strftime("%Y-%m-%d"),
                "mfe": values[valid, 0],
                "peak_adverse": values[valid, 1],
                "min_low": values[valid, 2],
                "raw": values[valid, 3],
            }))
    truth = {}
    for field in _ORACLE_PROVIDER_FIELDS:
        if not chunks[field]:
            raise OracleBlockedError(f"{field}沒有可用的已成熟Research價格真值")
        table = pd.concat(chunks[field], ignore_index=True)
        if table.duplicated(["ticker", "date"]).any():
            raise OracleBlockedError(f"{field}的Research價格真值ticker/date重複")
        truth[field] = table
    return truth, {
        "truth_source": "canonical full Research OHLCV via daily target calculator",
        "score_start": str(score_start.date()),
        "price_ticker_count": len(paths),
        "unreadable_tickers": skipped,
        "mature_rows_by_provider": valid_counts,
    }


def _reconcile_oracle_truth(
    source: pd.DataFrame, market: pd.DataFrame, *, field: str, score_start: str,
) -> tuple[pd.DataFrame, int]:
    """Ensure expanded trading universe agrees with official model truth."""
    old = source.loc[source["date"] >= score_start]
    check = old.merge(market, on=["ticker", "date"], how="left",
                      suffixes=("_provider", "_market"), validate="one_to_one", indicator=True)
    missing = check["_merge"] == "left_only"
    if missing.any():
        row = check.loc[missing, ["ticker", "date"]].iloc[0]
        raise OracleBlockedError(
            f"{field}官方真值找不到相應Research價格成熟路徑：{row['ticker']} {row['date']}"
        )
    for name in ("mfe", "peak_adverse", "min_low", "raw"):
        if not np.allclose(check[f"{name}_provider"].to_numpy(dtype=float),
                           check[f"{name}_market"].to_numpy(dtype=float),
                           rtol=2e-5, atol=2e-5):
            raise OracleBlockedError(f"{field}的official target與Research價格重建真值不一致：{name}")
    return market, len(market) - len(old)


def _load_truth(definition, root: Path, source_field: str, *, cutoff: str) -> pd.DataFrame:
    profile_id = str(definition.source[source_field])
    profile = get_breakout_quality_experiment_profile(profile_id)
    architecture = str(profile.model_architecture or definition.source["model_architecture"])
    bundle = load_profile_continuous_ranker_data(
        filter_id=str(definition.source["filter_id"]),
        model_architecture=architecture,
        experiment_profile=profile_id,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    manifest_target = str(dict((bundle.target_manifest or {}).get("target_contract") or {}).get("target_id") or "")
    if manifest_target != str(profile.continuous_target_id):
        raise OracleBlockedError(f"canonical target identity漂移：{source_field}")
    group = pd.DataFrame(bundle.group_table).reset_index(drop=True)
    required = {"ticker", "date", "label_eval_end_date", "target_favorable_r", "target_adverse_r", "target_minimum_low_return"}
    if not required.issubset(group):
        raise OracleBlockedError(f"{source_field}缺少官方daily future-path欄位：{sorted(required - set(group))}")
    if len(group) != len(bundle.raw_target) or len(group) != len(bundle.target_valid):
        raise OracleBlockedError(f"{source_field} group/target shape不一致")
    mature = (
        np.asarray(bundle.target_valid, dtype=bool)
        & (pd.to_datetime(group["label_eval_end_date"], errors="raise") <= pd.Timestamp(cutoff)).to_numpy()
    )
    frame = pd.DataFrame({
        "ticker": group["ticker"].map(normalize_ticker),
        "date": pd.to_datetime(group["date"], errors="raise").dt.strftime("%Y-%m-%d"),
        "mfe": pd.to_numeric(group["target_favorable_r"], errors="coerce"),
        "peak_adverse": pd.to_numeric(group["target_adverse_r"], errors="coerce"),
        "min_low": pd.to_numeric(group["target_minimum_low_return"], errors="coerce"),
        "raw": np.asarray(bundle.raw_target, dtype=float),
    })
    numeric = frame[["mfe", "peak_adverse", "min_low", "raw"]].to_numpy(dtype=float)
    mature &= np.isfinite(numeric).all(axis=1)
    frame = frame.loc[mature].copy()
    if frame.empty:
        raise OracleBlockedError(f"{source_field}沒有研究截止日前成熟且完整的共同樣本")
    if frame.duplicated(["ticker", "date"]).any():
        raise OracleBlockedError(f"{source_field}存在ticker/date重複")
    return frame


def _mfe40_over_d40_ratio(mfe40: pd.Series, safety_d40: pd.Series) -> pd.Series:
    """Raw MFE40/D40 (both in canonical R), never an arbitrary fitted risk floor.

    Signed MFE40 may be negative when every future high is below the anchor.
    Zero D40 with positive upside has mathematically unbounded ratio and is
    ordered above finite values. Zero upside with zero downside is zero utility
    (0/0 convention for ranking only). The *final* daily percentile is finite.
    """
    numerator = np.asarray(mfe40, dtype=float)
    downside = -np.asarray(safety_d40, dtype=float)
    if (
        numerator.shape != downside.shape
        or not np.isfinite(numerator).all()
        or not np.isfinite(downside).all()
        or (downside < 0.0).any()
    ):
        raise OracleBlockedError("MFE40/D40必須使用有限的canonical MFE40、非負的D40真值")
    ratio = np.zeros_like(numerator)
    np.divide(numerator, downside, out=ratio, where=downside > 0.0)
    ratio[(downside == 0.0) & (numerator > 0.0)] = np.inf
    return pd.Series(ratio, index=mfe40.index, dtype=float)


def build_oracle_score_tables(definition, *, project_root: Path, cutoff: str) -> tuple[dict[str, dict[tuple[str, str], float]], dict[str, Any]]:
    """Pure deterministic hindsight target construction, no data-driven fitting."""
    root = Path(project_root).resolve()
    market, market_evidence = _market_truth_for_oracle(definition, root, cutoff=cutoff)
    provider_truth = {
        field: _load_truth(definition, root, field, cutoff=cutoff)
        for field in _ORACLE_PROVIDER_FIELDS
    }
    complete = {}
    extra_rows = {}
    for field in _ORACLE_PROVIDER_FIELDS:
        complete[field], extra_rows[field] = _reconcile_oracle_truth(
            provider_truth[field], market[field], field=field,
            score_start=market_evidence["score_start"],
        )
    raw40 = complete["pure_mfe40_provider"]
    raw20 = complete["mfe20_d20_provider"]
    first = complete["first_breach_provider"]
    a = raw40.rename(columns={c: f"{c}40" for c in ("mfe", "peak_adverse", "min_low", "raw")})
    b = raw20.rename(columns={c: f"{c}20" for c in ("mfe", "peak_adverse", "min_low", "raw")})
    c = first.rename(columns={col: f"{col}fb" for col in ("mfe", "peak_adverse", "min_low", "raw")})
    joint = a.merge(b, on=["ticker", "date"], how="inner", validate="one_to_one")
    joint = joint.merge(c, on=["ticker", "date"], how="inner", validate="one_to_one")
    if joint.empty:
        raise OracleBlockedError("MFE40、MFE20/D20、First-Breach沒有共同已成熟stock-day")
    if bool(joint.duplicated(["ticker", "date"]).any()):
        raise OracleBlockedError("Oracle common cohort存在重複ticker/date")
    risk = float(StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY).risk_budget_return)
    for horizon in (20, 40):
        joint[f"safety_d{horizon}"] = full_horizon_mae_safety_r_from_minimum_low_return(
            joint[f"min_low{horizon}"].to_numpy(dtype=float), risk_budget_return=risk,
        )
        joint[f"safety_peak{horizon}"] = -joint[f"peak_adverse{horizon}"].astype(float)
    if not np.isfinite(joint[["safety_d20", "safety_d40"]].to_numpy(dtype=float)).all():
        raise OracleBlockedError("Oracle D20/D40 safety包含不合法值")

    method = str(definition.dimensions["percentile_method"])
    if method != "average_zero_based":
        raise OracleBlockedError("Oracle percentile必須採canonical average_zero_based")
    def percentile(values: pd.Series) -> pd.Series:
        temporary = pd.DataFrame({"date": joint["date"], "value": values})
        result, _ = ensure_daily_percentile(temporary, method=method)
        return result["percentile"]

    # The 50/50 equal-rank combination is the pre-existing canonical MR-13N
    # aggregation. The AO/CG/CH names indicate geometry, NOT an AO/CG/CH model head.
    raw = {
        "AO40_EqualRank": (percentile(joint["mfe40"]) + percentile(joint["safety_peak40"])) / 2,
        "CG40_EqualRank": (percentile(joint["mfe40"]) + percentile(joint["safety_d40"])) / 2,
        "CH20_EqualRank": (percentile(joint["mfe20"]) + percentile(joint["safety_d20"])) / 2,
        "U40": joint["mfe40"] + joint["safety_d40"],
        "U20": joint["mfe20"] + joint["safety_d20"],
        "FirstBreach_NetR": joint["rawfb"] - joint["peak_adversefb"],
        "MFE40_Control": joint["mfe40"],
        "MFE40_over_D40": _mfe40_over_d40_ratio(joint["mfe40"], joint["safety_d40"]),
        "MFE20": joint["mfe20"],
        "MAE40": joint["safety_d40"],
        "MAE20": joint["safety_d20"],
    }
    expected = {target for target, _ in ORACLE_TARGETS}
    if set(raw) != expected:
        raise OracleBlockedError("Oracle target inventory不一致")
    keys = list(zip(joint["ticker"].astype(str), joint["date"].astype(str)))
    scores: dict[str, dict[tuple[str, str], float]] = {}
    for name, value in raw.items():
        ranking = percentile(pd.Series(value, index=joint.index))
        if not np.isfinite(ranking.to_numpy(dtype=float)).all():
            raise OracleBlockedError(f"Oracle {name}存在非有限rank score")
        scores[name] = dict(zip(keys, ranking.to_numpy(dtype=float)))
    evidence = {
        "cohort_rows": len(keys),
        "first_common_day": min(joint["date"]),
        "last_common_day": max(joint["date"]),
        "research_cutoff": cutoff,
        "risk_budget_return": risk,
        "providers": {
            "40d": definition.source["pure_mfe40_provider"],
            "20d": definition.source["mfe20_d20_provider"],
            "first_breach": definition.source["first_breach_provider"],
        },
        "scoring": "All targets final-ranked by average_zero_based percentile within common-day universe",
        "oracle_only": True,
        "ratio_zero_denominator_policy": "D40=0,MFE40>0 -> +infinity (highest rank); 0/0 -> 0; final score = finite daily percentile",
        "mfe40_over_d40_zero_downside_positive_upside_count": int(((joint["safety_d40"] == 0.0) & (joint["mfe40"] > 0.0)).sum()),
        "mfe40_over_d40_zero_zero_count": int(((joint["safety_d40"] == 0.0) & (joint["mfe40"] == 0.0)).sum()),
        "market_truth": market_evidence,
        "rows_outside_dl_training_provider_by_source": extra_rows,
    }
    return scores, evidence


def _metrics(replay: dict[str, Any], *, target: str) -> dict[str, Any]:
    summary = _scenario_summary(replay)
    required = ("total_return_pct", "max_drawdown_pct", "return_over_max_drawdown", "expected_value_r", "trade_count", "avg_exposure_pct")
    if any(not math.isfinite(float(summary[name])) for name in (*required, "annual_return_pct")):
        raise OracleBlockedError(f"Oracle {target}缺少有效portfolio performance metrics")
    yearly = _yearly_frame(replay["profile"], "strategy")
    return {
        "target": target,
        "total_return_pct": float(summary["total_return_pct"]),
        "annualized_return_pct": float(summary["annual_return_pct"]),
        "mdd_pct": float(summary["max_drawdown_pct"]),
        "romd": float(summary["return_over_max_drawdown"]),
        "ev_a_r": float(summary["expected_value_r"]),
        "trades": int(summary["trade_count"]),
        "exposure_pct": float(summary["avg_exposure_pct"]),
        "yearly": _to_json_native(yearly.to_dict("records")),
    }


def _aggregate(records: list[dict[str, Any]], *, target: str) -> dict[str, Any]:
    if not records:
        raise OracleBlockedError("Oracle robustness missing seed observations")
    summary = {"target": target, "seeds": len(records)}
    fields = (
        "total_return_pct", "annualized_return_pct", "mdd_pct", "romd", "ev_a_r", "trades", "exposure_pct",
    )
    for field in fields:
        values = np.asarray([float(x[field]) for x in records], dtype=float)
        summary[field] = float(np.mean(values))
        summary[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    yearly = {}
    for row in records:
        for year_row in row["yearly"]:
            year = str(year_row["year"])
            yearly.setdefault(year, []).append(year_row)
    summary["yearly"] = [
        {
            "year": year,
            "strategy_return_pct": float(np.mean([float(z["strategy_return_pct"]) for z in values])),
            "n_seeds": len(values),
            "is_full_year": all(bool(z.get("is_full_year", False)) for z in values),
        }
        for year, values in sorted(yearly.items())
    ]
    summary["seed_observations"] = records
    return summary


def build_report(definition, *, project_root: Path, quiet: bool = False) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings, base_arm, exact_arm = _contract(definition)
    status = collect_status(definition, project_root=root)
    if status["status"] != "READY":
        raise OracleBlockedError(status["reason"])
    robustness, parameter_paths = _robustness_artifacts(definition, root)
    scores, cohort = build_oracle_score_tables(
        definition, project_root=root, cutoff=str(settings.end_date),
    )
    start = str(settings.start_date)
    end = min(str(settings.end_date), str(cohort["last_common_day"]))
    if end <= start:
        raise OracleBlockedError("共同成熟樣本沒有可回測期間")
    data_dir = Path(get_dataset_dir(str(root), settings.dataset)).resolve()
    replay_kwargs = {
        "data_dir": data_dir, "start_date": start, "end_date": end,
        "max_positions": settings.max_positions, "enable_rotation": False, "quiet": quiet,
    }
    # Same-seed paired replay is essential: benchmark parameter fitting varies by
    # seed and cannot be replaced by the single-run P16 parameters. Resolve and
    # verify every current canonical parameter identity *before* considering
    # any prior scientific observation for reuse.
    prepared = []
    param_evidence = []
    for seed, path in parameter_paths:
        source, param_sha = load_strategy_param_evaluation_view(
            path, evaluation_mode=str(definition.dimensions["evaluation_mode"]),
            start_date=start, end_date=end,
        )
        _validate_requested_param_policy(source, str(base_arm.param_policy))
        kind = str(source["kind"])
        if kind == "rolling_active_param_ensemble":
            schedule_start, schedule_end = get_active_param_ensemble_date_range(source["payload"])
        elif kind == "rolling_oos_param_schedule":
            schedule_start, schedule_end = get_active_param_date_range(source["payload"])
        else:
            raise OracleBlockedError("Oracle基準必須重用正式Rolling P16日程，禁止靜態前視參數")
        if str(schedule_start) > start or str(schedule_end) < end:
            raise OracleBlockedError(f"Oracle P16日程未完整覆蓋共同成熟期間: {schedule_start}~{schedule_end}")
        kind, base_params, oracle_params, *_ = _build_controlled_param_source_pair(
            source,
            filter_id=str(definition.source["filter_id"]),
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD), fixed_risk=None,
            comparison_mode=COMPARISON_MODE_SCORE_RANKING,
            optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
            shared_param_overrides=ALL_RULE_FILTERS_OFF_OVERRIDES,
        )
        prepared.append((seed, param_sha, kind, base_params, oracle_params))
        param_evidence.append({
            "seed": seed,
            "param_path": project_relative_display_path(path, project_root=root),
            "param_sha256": param_sha,
        })

    # Evaluation/report schema changes must not rebuild identical scientific
    # replays. Cache identity includes actual Research price bytes, resolved
    # P16 SHA, scientific comparison settings, cohort and ranking contract.
    # Incomplete work is saved per seed/target; a later run resumes missing
    # observations only. Historical V1 snapshots are checked separately.
    try:
        price_sha = price_inventory_sha256(root, settings.dataset)
        archived = HistoricalOracleIndex(root)
    except ValueError as exc:
        raise OracleBlockedError(str(exc)) from exc
    cache = OracleReplayStore(root, definition.output_subdir)
    definitions = dict(ORACLE_TARGETS)
    definitions_with_base = {"C96": "Canonical same-window Min P16 Base-Finalist-Best", **definitions}
    # Use the exact two-arm scientific contract; changes to unrelated compare
    # arms/reports must NOT invalidate these completed Oracle replays.
    comparison_sha = canonical_json_sha256({
        "dataset": settings.dataset,
        "start_date": start, "end_date": end,
        "max_positions": int(settings.max_positions),
        "rotation": settings.rotation,
        "parameter_source": settings.parameter_sources[str(base_arm.param_source)].as_dict(),
        "baseline_arm": base_arm.as_dict(),
        "exact_arm": exact_arm.as_dict(),
    })
    legacy_context = {
        "benchmark": robustness.benchmark_id,
        "strategy_profile_id": settings.profile_id,
        "dataset": settings.dataset,
        "start_date": start,
        "end_date": end,
        "seeds": [seed for seed, _ in parameter_paths],
        "param_sha_by_seed": {int(item["seed"]): item["param_sha256"] for item in param_evidence},
        "cohort": cohort,
        "definitions": definitions,
    }
    replay_policy = {
        "baseline_arm": str(definition.source["baseline_arm_id"]),
        "exact_arm": str(definition.source["exact_arm_id"]),
        "filter_id": str(definition.source["filter_id"]),
        "param_policy": str(base_arm.param_policy),
        "comparison_mode": COMPARISON_MODE_SCORE_RANKING,
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "all_rule_filters_off": dict(ALL_RULE_FILTERS_OFF_OVERRIDES),
        "score_threshold": float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        "fixed_risk": float(DEFAULT_FIXED_RISK),
        "max_position_cap_pct": float(DEFAULT_MAX_POSITION_CAP_PCT),
        "broker_accounting": {
            "policy_id": BROKER_ACCOUNTING_POLICY_ID,
            "fee_rate": str(BROKER_FEE_RATE),
            "min_fee": str(BROKER_MIN_FEE_TWD),
            "odd_lot_min_fee": str(BROKER_ODD_LOT_MIN_FEE_TWD),
            "board_lot_shares": int(BROKER_BOARD_LOT_SHARES),
            "stock_tax_rate": str(BROKER_STOCK_TAX_RATE),
            "rounding": BROKER_CHARGE_ROUNDING_MODE,
        },
        "execution_defaults": {
            key: str(spec["default"]) if spec["default"] is not None else None
            for key, spec in sorted(EXECUTION_POLICY_PARAM_SPECS.items())
        },
    }
    observations: dict[str, list[dict[str, Any]]] = {"C96": []}
    observations.update({target: [] for target, _ in ORACLE_TARGETS})
    for seed, param_sha, kind, base_params, oracle_params in prepared:
        if not quiet:
            print(f"[Oracle Audit] seed={seed}: verify C96 + {len(ORACLE_TARGETS)} Oracle results")

        def load_or_run(target: str, runner):
            identity = replay_identity(
                dataset=settings.dataset, dataset_sha256=price_sha,
                target=target, definition=definitions_with_base,
                settings_fingerprint=comparison_sha, start=start, end=end,
                benchmark=str(robustness.benchmark_id), param_sha256=param_sha,
                seed=seed, cohort=cohort,
                ranking_policy=str(exact_arm.dl_runtime_mode),
                ranking_options=dict(exact_arm.dl_runtime_options),
                evaluation_mode=str(definition.dimensions["evaluation_mode"]),
                max_positions=settings.max_positions, replay_policy=replay_policy,
            )
            try:
                record = cache.read(identity)
                origin = "verified per-arm cache"
                if record is None and not cache.has_other_identity(identity):
                    record = archived.lookup(target=target, seed=seed, context=legacy_context)
                    origin = "compatible completed Oracle report"
                    if record is not None:
                        cache.write(identity, record, origin="historical_report")
                if record is not None:
                    if not quiet:
                        print(f"[REUSE] seed={seed} {target} | {origin}")
                    return record
            except ValueError as exc:
                raise OracleBlockedError(str(exc)) from exc
            if not quiet:
                print(f"[RUN] seed={seed} {target} | missing completed identity")
            record = _metrics(runner(), target=target)
            record["seed"] = seed
            if target != "C96":
                record["delta_romd_vs_c96"] = record["romd"] - observations["C96"][-1]["romd"]
            try:
                cache.write(identity, record, origin="canonical_replay")
            except ValueError as exc:
                raise OracleBlockedError(str(exc)) from exc
            return record

        baseline = load_or_run("C96", lambda: _run_scenario(
            name=f"C96 seed={seed} same-window", params=base_params,
            param_source_kind=kind, **replay_kwargs,
        ))
        observations["C96"].append(baseline)
        for target, _description in ORACLE_TARGETS:
            def run_oracle(target=target):
                with audit_oracle_truth_score_context(
                    scores[target],
                    ranking_policy=str(exact_arm.dl_runtime_mode),
                    ranking_options=exact_arm.dl_runtime_options,
                ):
                    return _run_scenario(
                        name=f"Oracle {target} seed={seed}", params=oracle_params,
                        param_source_kind=kind, **replay_kwargs,
                    )
            record = load_or_run(target, run_oracle)
            if not math.isclose(
                float(record["delta_romd_vs_c96"]), float(record["romd"] - baseline["romd"]),
                rel_tol=1e-9, abs_tol=1e-9,
            ):
                raise OracleBlockedError(f"Oracle seed={seed} {target}歷史對照C96不一致，禁止重用")
            observations[target].append(record)
    baseline_summary = _aggregate(observations["C96"], target="C96")
    rows: list[dict[str, Any]] = []
    for target, _ in ORACLE_TARGETS:
        item = _aggregate(observations[target], target=target)
        diffs = np.array([record["delta_romd_vs_c96"] for record in observations[target]], dtype=float)
        item["delta_romd_vs_c96"] = float(np.mean(diffs))
        item["delta_romd_std"] = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        item["positive_seed_count"] = int((diffs > 0.0).sum())
        rows.append(item)
    return {
        "schema_version": ORACLE_REPORT_SCHEMA_VERSION, "status": "COMPLETED", "read_only": True,
        "hindsight_non_pit": True, "iterative_oos_evidence": True,
        "strategy_profile_id": settings.profile_id,
        "benchmark_id": robustness.benchmark_id,
        "seeds": [seed for seed, _ in parameter_paths],
        "param_artifacts": param_evidence,
        "dataset": settings.dataset,
        "start_date": start, "end_date": end,
        "common_cohort": cohort,
        "baseline": baseline_summary,
        "oracles": rows,
        "definitions": dict(ORACLE_TARGETS),
        "limitations": [
            "Ex-post future targets are NOT predictions, not PIT-legal, and never a deployable strategy.",
            "AO/CG/CH EqualRank are fixed 50/50 geometry proxies, not learned model-head fusion.",
            "Same-window C96 is verified/reused or replayed for each benchmark seed; full-period C96 is not comparable.",
            "N=4 paired end-to-end parameter seeds; oracle scores have no trainable seed.",
            "No best-target fitting from iterative OOS; multi-target search is exploratory.",
            "MAE40 and MAE20 rank low downside first; MFE40/D40 has a zero-downside singularity, reported separately and not tuned.",
        ],
    }


def render_result(report: Mapping[str, Any], *, target: str) -> str:
    # Persistent reusable report columns/order/precision are owned by the
    # approved machine-readable contract; no parallel headings live here.
    contract = report_contract("audit.target_oracle_romd")
    sections = contract.sections
    perf_table = sections[0].tables[0]
    yearly_table = sections[2].tables[0]
    paired_table = sections[3].tables[0]
    baseline = dict(report["baseline"])
    items = [dict(baseline, target="C96 Same-Window", delta_romd_vs_c96=0.0, wins="-"),
             *[dict(item, wins=f'{item["positive_seed_count"]}/{item["seeds"]}')
               for item in report["oracles"]]]
    rows = [format_contract_row(perf_table, item) for item in items]
    intro = [
        "Target Ground Truth -> Strategy RoMD (HINDSIGHT ORACLE / NOT PIT)",
        f'Period: {report["start_date"]} ~ {report["end_date"]} (40D-mature common window)',
        f'Common stock-day cohort: {report["common_cohort"]["cohort_rows"]:,}',
        f'Benchmark: {report["benchmark_id"]}, same-seed pairs N={len(report["seeds"])}',
        f'Report schema: {report["schema_version"]}, Oracle treatments: {len(report["oracles"])} (formal Research Audit; not Standard SOP)',
        'Ranking: Exact K/R0, Broker-Cash, same seed/P16; future truth changes ranking only.',
        (
            'MFE40/D40 zero denominator: '
            f'{report["common_cohort"]["mfe40_over_d40_zero_downside_positive_upside_count"]:,} positive-MFE (+infinity); '
            f'{report["common_cohort"]["mfe40_over_d40_zero_zero_count"]:,} zero/zero (rank 0).'
        ),
        'AO/CG/CH = equal-rank 50/50 geometry PROXIES, not neural-network model heads.',
    ]
    if target == "console":
        return "\n\n".join((
            render_title(intro[0]), "\n".join(intro[1:]),
            render_table(perf_table.headers, rows),
            "NOTICE: Future labels are hindsight only; not legal for PIT/Trading.",
        ))
    lines = [f"# {intro[0]}", *[f"- {line}" for line in intro[1:]], "",
             f"## {sections[0].display_title}",
             "| " + " | ".join(perf_table.headers) + " |",
             "|" + "|".join("---" for _ in perf_table.headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(["", f"## {sections[1].display_title}",
                  *[f"- **{key}** - {desc}" for key, desc in ORACLE_TARGETS],
                  "", f"## {sections[2].display_title}"])
    for item in items:
        lines.append(f'### {item["target"]}')
        lines.extend(("| " + " | ".join(yearly_table.headers) + " |",
                      "|" + "|".join("---" for _ in yearly_table.headers) + "|"))
        for year in item["yearly"]:
            lines.append("| " + " | ".join(format_contract_row(yearly_table, year)) + " |")
    lines.extend(["", f"## {sections[3].display_title}",
                  "| " + " | ".join(paired_table.headers) + " |",
                  "|" + "|".join("---" for _ in paired_table.headers) + "|"])
    baselines = {int(x["seed"]): float(x["romd"]) for x in baseline["seed_observations"]}
    for item in report["oracles"]:
        for obs in item["seed_observations"]:
            row = dict(obs, target=item["target"], c96_romd=baselines[int(obs["seed"])])
            lines.append("| " + " | ".join(format_contract_row(paired_table, row)) + " |")
    lines.extend(["", f"## {sections[4].display_title}",
                  *[f"- {limitation}" for limitation in report["limitations"]]])
    return "\n".join(lines)


def run_formal_audit(definition, *, project_root: Path, quiet: bool = False) -> dict[str, Any]:
    # Authorization, frozen schema, and historical snapshot must be checked
    # before initiating the costly hindsight-only portfolio replays.
    approved = require_approved_report_contract("audit.target_oracle_romd")
    payload = dict(build_report(definition, project_root=project_root, quiet=quiet))
    payload["report_contract_id"] = f"{approved.report_id}@v{approved.version}"
    payload["report_contract_fingerprint"] = persistent_report_contract_fingerprint(approved.report_id)
    console = render_result(payload, target="console")
    markdown = render_result(payload, target="markdown")
    adapter = SimpleNamespace(
        report_id=definition.audit_id, report_type=definition.audit_type,
        output_subdir=definition.output_subdir,
    )
    saved = persist_reusable_report(
        definition=adapter, project_root=project_root,
        payload=payload, console_text=console, markdown_text=markdown,
    )
    if not quiet:
        print(console)
    return saved


__all__ = [
    "SUPPORTED_AUDIT_TYPE", "ORACLE_TARGETS", "OracleBlockedError",
    "collect_status", "build_oracle_score_tables", "build_report", "render_result", "run_formal_audit",
]

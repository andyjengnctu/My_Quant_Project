"""One-time hindsight target economics Audit; NEVER a legal prediction source.

This audit compares future-outcome stock-day rankings by running the same frozen
Broker-Cash P16 Exact K/R0 portfolio replay. All historical truths are *ex post*.
No model training, PIT score publication, optimizer, or trading state is touched.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
import math

import numpy as np
import pandas as pd

from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from core.console_report import render_table, render_title
from core.dataset_profiles import get_dataset_dir
from core.path_utils import project_relative_display_path
from core.strategy_compare_policy import (
    get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
)
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
from services.optimizer.strategy_param_service import (
    inspect_fixed_multiseed_robustness_strategy_parameter_artifact,
)
from config.execution_policy import DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT
from config.breakout_quality import BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
from core.active_param_ensemble import get_active_param_ensemble_date_range
from core.rolling_oos_params import get_active_param_date_range
from services.research.strategy_compare_replay import _run_scenario

SUPPORTED_AUDIT_TYPE = "target_ground_truth_strategy_romd"
ORACLE_TARGETS = (
    ("AO40_EqualRank", "40D MFE + adverse-to-peak Safety, 50/50 daily rank"),
    ("CG40_EqualRank", "40D MFE + full D40 Safety, 50/50 daily rank"),
    ("CH20_EqualRank", "20D MFE + full D20 Safety, 50/50 daily rank"),
    ("U40", "40D MFE - D40, canonical R"),
    ("U20", "20D MFE - D20, canonical R"),
    ("FirstBreach_NetR", "MFE before first risk breach - adverse to that peak, R"),
    ("MFE40_Control", "40D MFE only (control, not target recommendation)"),
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


def build_oracle_score_tables(definition, *, project_root: Path, cutoff: str) -> tuple[dict[str, dict[tuple[str, str], float]], dict[str, Any]]:
    """Pure deterministic hindsight target construction, no data-driven fitting."""
    root = Path(project_root).resolve()
    raw40 = _load_truth(definition, root, "pure_mfe40_provider", cutoff=cutoff)
    raw20 = _load_truth(definition, root, "mfe20_d20_provider", cutoff=cutoff)
    first = _load_truth(definition, root, "first_breach_provider", cutoff=cutoff)
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
    # seed and cannot be replaced by the single-run P16 parameters.
    observations: dict[str, list[dict[str, Any]]] = {"C96": []}
    observations.update({target: [] for target, _ in ORACLE_TARGETS})
    param_evidence = []
    for seed, path in parameter_paths:
        if not quiet:
            print(f"[Oracle Audit] Benchmark seed={seed}: C96 + {len(ORACLE_TARGETS)} Oracle replays")
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
        param_evidence.append({
            "seed": seed,
            "param_path": project_relative_display_path(path, project_root=root),
            "param_sha256": param_sha,
        })
        baseline = _metrics(_run_scenario(
            name=f"C96 seed={seed} same-window", params=base_params,
            param_source_kind=kind, **replay_kwargs,
        ), target="C96")
        baseline["seed"] = seed
        observations["C96"].append(baseline)
        for target, _description in ORACLE_TARGETS:
            with audit_oracle_truth_score_context(
                scores[target],
                ranking_policy=str(exact_arm.dl_runtime_mode),
                ranking_options=exact_arm.dl_runtime_options,
            ):
                replay = _run_scenario(
                    name=f"Oracle {target} seed={seed}", params=oracle_params,
                    param_source_kind=kind, **replay_kwargs,
                )
            result = _metrics(replay, target=target)
            result["seed"] = seed
            result["delta_romd_vs_c96"] = result["romd"] - baseline["romd"]
            observations[target].append(result)
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
        "schema_version": 1, "status": "COMPLETED", "read_only": True,
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
            "Same-window C96 is recomputed for each benchmark seed; full-period C96 is not comparable.",
            "N=4 paired end-to-end parameter seeds; oracle scores have no trainable seed.",
            "No best-target fitting from iterative OOS; multi-target search is exploratory.",
        ],
    }


def render_result(report: Mapping[str, Any], *, target: str) -> str:
    headings = ("Target", "Return Mean", "Ann. Mean", "MDD Mean", "RoMD Mean", "RoMD SD", "Delta RoMD", "Win/Seed", "EV(A) Mean")
    baseline = dict(report["baseline"])
    rows = [dict(baseline, target="C96 Same-Window", delta_romd_vs_c96=0.0, positive_seed_count=0), *list(report["oracles"])]
    cells = [(
        x["target"], f'{x["total_return_pct"]:+.2f}%', f'{x["annualized_return_pct"]:+.2f}%',
        f'{x["mdd_pct"]:.2f}%', f'{x["romd"]:.2f}', f'{x["romd_std"]:.2f}',
        f'{x["delta_romd_vs_c96"]:+.2f}',
        "-" if x["target"] == "C96 Same-Window" else f'{x["positive_seed_count"]}/{x["seeds"]}',
        f'{x["ev_a_r"]:+.2f} R',
    ) for x in rows]
    intro = [
        "Target Ground Truth → Strategy RoMD (HINDSIGHT ORACLE / NOT PIT)",
        f'Period: {report["start_date"]} ～ {report["end_date"]} (40D-mature common window)',
        f'Common stock-day cohort: {report["common_cohort"]["cohort_rows"]:,}',
        f'Benchmark: {report["benchmark_id"]}, same-seed pairs N={len(report["seeds"])}',
        'Ranking: Exact K/R0, Broker-Cash, same seed/P16; future truth changes ranking only.',
        'AO/CG/CH = equal-rank 50/50 geometry PROXIES, not neural-network model heads.',
    ]
    if target == "console":
        return "\n\n".join((render_title(intro[0]), "\n".join(intro[1:]), render_table(headings, cells),
            "NOTICE: Future labels are hindsight only; not legal for PIT/Trading."))
    lines = [f"# {intro[0]}", *[f"- {line}" for line in intro[1:]], "", "## 1. Oracle Performance (4-seed mean)",
        "| " + " | ".join(headings) + " |", "|" + "|".join("---" for _ in headings) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in cells)
    lines.extend(["", "## 2. Target Definitions", *[f"- **{key}** — {desc}" for key, desc in ORACLE_TARGETS],
                  "", "## 3. Yearly Stability (same-window seed means)"])
    for x in rows:
        lines.append(f'### {x["target"]}')
        lines.extend(("| Year | Mean Return | Seed Count | Full Year |", "|---|---:|---:|---|"))
        for yr in x["yearly"]:
            lines.append(f'| {yr["year"]} | {yr["strategy_return_pct"]:+.2f}% | {yr["n_seeds"]} | {yr["is_full_year"]} |')
    lines += ["", "## 4. Robustness (paired same-seed ΔRoMD)",
              "| Target | Seed | RoMD | C96 Seed RoMD | Paired Delta |", "|---|---:|---:|---:|---:|"]
    baselines = {int(x["seed"]):float(x["romd"]) for x in baseline["seed_observations"]}
    for x in report["oracles"]:
        for obs in x["seed_observations"]:
            seed = int(obs["seed"])
            lines.append(f'| {x["target"]} | {seed} | {obs["romd"]:.2f} | {baselines[seed]:.2f} | {obs["delta_romd_vs_c96"]:+.2f} |')
    lines.extend(["", "## 5. Research Constraints", *[f"- {v}" for v in report["limitations"]]])
    return "\n".join(lines)


def run_formal_audit(definition, *, project_root: Path, quiet: bool = False) -> dict[str, Any]:
    payload = build_report(definition, project_root=project_root, quiet=quiet)
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

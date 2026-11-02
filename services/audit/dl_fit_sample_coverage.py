"""Read-only, registry-driven audit of actual Continuous DL FIT cohorts.

Only canonical PIT producers decide training membership and fitting identity.
The audit recomputes cohort IDs through their shared selector and requires
published Fold bytes, counts and source truth before marking any row VERIFIED.
It never generates Target labels, checkpoints, PIT scores or Optimization truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.breakout_quality_policy import (
    get_breakout_quality_model_test_settings,
    get_breakout_quality_workflow_settings,
    get_continuous_ranker_execution_recipe,
)
from core.breakout_quality_registry import TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
from core.breakout_quality_runtime import CONTINUOUS_RANKER_TARGET_BUILDER_BINARY_POSITIVE
from core.console_report import render_table
from core.research_report_contract import (
    require_approved_report_contract, section_contract, table_contract,
)
from core.training_policy import get_robustness_benchmark_policy_snapshot
from config.robustness_benchmark import MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.model_source_provenance import source_inventory_matches_current
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    resolve_model_research_robustness_seed_dirs,
)
from filters.breakout_quality.pit_fold_provenance import validate_pit_fold_evidence
from filters.breakout_quality.pit_group_selection import _fold_group_ids, build_training_scope_mask
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from services.breakout_quality.fitting_data_dependencies import (
    FitDataDependencies, fold_source_dependency_issues, mintrade40_fit_sample_sha256,
)
from services.breakout_quality.price_input_source import resolve_feature_price_source
from services.breakout_quality.trade_count_source import resolve_trade_count_feature_source
from services.breakout_quality.trade_path_target_source import loader_kwargs_for_trade_path
from services.audit.reusable_report import (
    audit_section, audit_title, format_contract_row, markdown_table,
    persist_reusable_report,
)

SUPPORTED_AUDIT_TYPE = "dl_fit_sample_coverage"
REPORT_CONTRACT_ID = "audit.dl_fit_sample_coverage"


def _inventory() -> tuple[tuple[str, str], ...]:
    return get_breakout_quality_model_test_settings().model_profiles


def _seeds() -> tuple[int, ...]:
    benchmark = get_robustness_benchmark_policy_snapshot(
        benchmark_id=MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID,
    )
    return tuple(int(v) for v in benchmark["resolved_seeds"])


def _source_path(root: Path, profile: str, seed: int) -> tuple[Path, str, str]:
    settings = get_breakout_quality_workflow_settings(experiment_profile=profile)
    path, _ = resolve_model_research_robustness_seed_dirs(
        root, settings.filter_id, settings.model_architecture, profile,
        seed=int(seed), rolling=True,
    )
    return path, settings.filter_id, settings.model_architecture


def collect_status(definition, *, project_root: Path) -> dict[str, Any]:
    """Cheap read-only preflight; run phase performs all expensive attestation."""
    root = Path(project_root).resolve()
    try:
        inventory, seeds = _inventory(), _seeds()
        existing = sum(
            (lambda directory: (directory / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME).is_file())(
                _source_path(root, profile, seed)[0]
            )
            for _model, profile in inventory for seed in seeds
        )
        total = len(inventory) * len(seeds)
        return {
            "status": "READY" if existing else "BLOCKED",
            "reason": (f"已發布manifest {existing}/{total}；缺件者將明列 BLOCKED，"
                       "不自動補建。" if existing else "無已發布的Rolling Robustness PIT manifest；Audit不建立前置工件"),
            "source": {"display": "canonical Model Compare/Test + Rolling PIT Fold + benchmark", "found": existing, "expected": total},
        }
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return {"status": "BLOCKED", "reason": str(exc), "source": {"display": "canonical Model Compare/Test + Rolling PIT Fold"}}


def _read_manifest(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("PIT manifest expected an object")
    return loaded


def _date_range(dates: pd.Series, ids: np.ndarray) -> dict[str, str | None]:
    if not len(ids):
        return {"start": None, "end": None}
    selected = dates.iloc[ids]
    return {"start": str(selected.min().date()), "end": str(selected.max().date())}


def _prepare_cohort(bundle) -> dict[str, Any]:
    """Prepare immutable diagnostics arrays; training predicates remain canonical."""
    table = bundle.group_table
    dates = pd.to_datetime(table["date"], errors="raise").dt.normalize()
    maturity = pd.to_datetime(table["label_eval_end_date"], errors="raise").dt.normalize()
    valid = np.asarray(bundle.target_valid, dtype=bool)
    scoped = build_training_scope_mask(bundle)
    if len(dates) != len(maturity) or len(valid) != len(dates) or len(scoped) != len(dates):
        raise ValueError("PIT bundle group/date/target scope長度不一致")
    event_index = np.asarray(bundle.event_group_index, dtype=np.int64)
    if event_index.ndim != 1 or np.any(event_index < 0) or np.any(event_index >= len(dates)):
        raise ValueError("PIT bundle event group index不合法")
    event_weights = np.bincount(event_index, minlength=len(dates)).astype(np.int64)
    return {
        "dates": dates, "maturity": maturity, "valid": valid, "scoped": scoped,
        "years": dates.dt.year.to_numpy(dtype=np.int64),
        "event_weights": event_weights,
        "raw": np.asarray(bundle.raw_target, dtype=np.float64),
    }


def _partitions(pre_score: np.ndarray, cohort: Mapping[str, Any], *, score_start: pd.Timestamp, ids) -> dict[str, Any]:
    valid = cohort["valid"]
    scoped = cohort["scoped"]
    matured = cohort["maturity"].lt(score_start).fillna(False).to_numpy(dtype=bool)
    missing = pre_score & ~valid
    excluded = pre_score & valid & ~scoped
    immature = pre_score & scoped & ~matured
    fit = np.zeros(len(valid), dtype=bool)
    fit[ids["final_ids"]] = True
    counts = {
        "pre_score_groups": int(pre_score.sum()),
        "valid_target": int((pre_score & valid).sum()),
        "missing_target": int(missing.sum()),
        "scope_excluded": int(excluded.sum()),
        "not_matured": int(immature.sum()),
        "final_refit": int(fit.sum()),
    }
    if sum(counts[k] for k in ("missing_target", "scope_excluded", "not_matured", "final_refit")) != counts["pre_score_groups"]:
        raise ValueError("PIT sample partition不能對帳：不能將缺值或未成熟視為負類")
    return {**counts, "fit_mask": fit, "missing_mask": missing,
            "excluded_mask": excluded, "immature_mask": immature}


def _positive_counts(cohort: Mapping[str, Any], fit_mask: np.ndarray, *, is_positive_binary: bool) -> tuple[int | None, int | None, float | None]:
    if not is_positive_binary:
        return None, None, None
    from core.positive_netpnl_contract import positive_labels
    labels = positive_labels(cohort["raw"][fit_mask])
    if not np.isfinite(labels).all():
        raise ValueError("Positive FIT內有未成熟/NULL target")
    yes, no = int((labels == 1).sum()), int((labels == 0).sum())
    if yes + no != int(fit_mask.sum()):
        raise ValueError("Positive FIT類別筆數不能對帳")
    return yes, no, (100.0 * yes / (yes + no)) if yes + no else None


def _year_rows(model: str, seed: int, fold: str, cohort: Mapping[str, Any], masks: Mapping[str, Any], *, is_positive_binary: bool) -> list[dict[str, Any]]:
    rows = []
    years = cohort["years"]
    fitted = masks["fit_mask"]
    for year in np.unique(years[masks["pre_score_mask"]]):
        year_scope = masks["pre_score_mask"] & (years == int(year))
        fit_mask = fitted & year_scope
        positive, negative, positive_rate = _positive_counts(cohort, fit_mask, is_positive_binary=is_positive_binary)
        rows.append({
            "model": model, "seed": seed, "fold": fold, "year": int(year),
            "pre_score_groups": int(year_scope.sum()),
            "valid_target": int((year_scope & cohort["valid"]).sum()),
            "missing_target": int((year_scope & masks["missing_mask"]).sum()),
            "scope_excluded": int((year_scope & masks["excluded_mask"]).sum()),
            "not_matured": int((year_scope & masks["immature_mask"]).sum()),
            "final_refit": int(fit_mask.sum()),
            "event_rows": int(cohort["event_weights"][fit_mask].sum()),
            "positive": positive, "nonpositive": negative, "positive_rate": positive_rate,
        })
    return rows


def _verify_fold(bundle, cohort: Mapping[str, Any], record: Mapping[str, Any], *, model: str, profile: str, seed: int, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Published FIT source/count/period identity must equal canonical sample selection."""
    fold_id = str(record["fold_id"])
    fold_manifest_path = _source_path(root, profile, seed)[0] / "folds" / fold_id / "manifest.json"
    manifest = _read_manifest(fold_manifest_path)
    settings = get_breakout_quality_workflow_settings(experiment_profile=profile)
    for source in (record, manifest):
        for key, value in (("fold_id", fold_id), ("seed", int(seed)),
                           ("experiment_profile", profile), ("filter_id", settings.filter_id),
                           ("model_architecture", settings.model_architecture)):
            # Aggregate fold records do not redundantly store every parent field.
            if source is record and key not in source:
                continue
            if source.get(key) != value:
                raise ValueError(f"{fold_id} {key}與正式FIT identity不一致")
    planned = dict(manifest.get("planned_periods") or {})
    record_planned = record.get("planned_periods")
    if not isinstance(record_planned, Mapping) or dict(record_planned) != planned:
        raise ValueError(f"{fold_id} aggregate/Fold periods不一致")
    score_start, score_end = pd.Timestamp(planned["score_start"]), pd.Timestamp(planned["score_end"])
    months = int(dict(manifest.get("training_settings") or {}).get(
        "train_window_months")) if dict(manifest.get("training_settings") or {}).get("train_window_months") is not None else None
    months_validation = (pd.Timestamp(planned["score_start"]).to_period("M") - pd.Timestamp(planned["validation_start"]).to_period("M")).n
    # Use the producer's single canonical PIT membership predicate, not a copy.
    fold = {"fold_id": fold_id, "score_start": score_start, "score_end": score_end}
    ids = _fold_group_ids(bundle, fold, validation_months=months_validation, train_window_months=months)
    for key, date in (("validation_start", ids["validation_start"]), ("history_start", ids["history_start"])):
        if (None if date is None else str(date.date())) != planned.get(key):
            raise ValueError(f"{fold_id} {key}不符合canonical PIT期間")
    phases = {"inner_train": ids["train_ids"], "validation": ids["validation_ids"],
              "final_refit": ids["final_ids"], "score": ids["score_ids"]}
    event_weights = cohort["event_weights"]
    for phase, group_ids in phases.items():
        if int(dict(manifest.get("group_counts") or {}).get(phase, -1)) != len(group_ids):
            raise ValueError(f"{fold_id} {phase} FIT groups與發布記錄不一致")
        if int(dict(manifest.get("event_row_counts") or {}).get(phase, -1)) != int(event_weights[group_ids].sum()):
            raise ValueError(f"{fold_id} {phase} Event rows與發布記錄不一致")
        if dict(dict(manifest.get("observed_periods") or {}).get(phase) or {}) != _date_range(cohort["dates"], group_ids):
            raise ValueError(f"{fold_id} {phase} 日期區間與發布記錄不一致")
    for key in ("group_counts", "event_row_counts", "observed_periods", "model_information_cutoff"):
        if record.get(key) != manifest.get(key):
            raise ValueError(f"{fold_id} aggregate與Fold {key}不一致")
    if not len(ids["final_ids"]):
        raise ValueError(f"{fold_id} FIT final_refit為空")
    cutoff = cohort["maturity"].iloc[ids["final_ids"]].max()
    if str(cutoff.date()) != manifest.get("model_information_cutoff"):
        raise ValueError(f"{fold_id} model_information_cutoff不一致")
    # FIT-source identity includes formal as-of teacher and the ordered sample
    # witness for MinTrade Targets; score-only/horizon bytes are not a FIT key.
    published_source = dict(manifest.get("source_contract") or {})
    start = (str(bundle.summary.get("training_universe_start_date"))
             if "training_universe_start_date" in published_source else None)
    expected_source = FitDataDependencies.from_bundle(bundle).rolling_source(
        daily_training_universe_start_date=start, fit_cutoff=cutoff,
    )
    digest = mintrade40_fit_sample_sha256(bundle, ids["final_ids"])
    if digest is not None:
        expected_source["formal_mintrade40_fit_samples_sha256"] = digest
    issues = fold_source_dependency_issues(
        manifest, {"source_contract": expected_source}, project_root=root,
        inventory_matcher=source_inventory_matches_current,
    )
    if issues:
        raise ValueError(f"{fold_id} FIT source mismatch: {'; '.join(issues)}")
    fit_range = _date_range(cohort["dates"], ids["final_ids"])
    return ({
        "model": model, "seed": seed, "fold": fold_id, "score_start": str(score_start.date()),
        "score_end": str(score_end.date()),
        "inner_train": len(ids["train_ids"]), "validation": len(ids["validation_ids"]),
        "final_refit": len(ids["final_ids"]), "event_rows": int(event_weights[ids["final_ids"]].sum()),
        "fit_start": fit_range["start"], "fit_end": fit_range["end"],
        "history_start": planned.get("history_start"), "status": "VERIFIED",
    }, ids)


def _audit_seed(root: Path, *, model: str, profile: str, seed: int) -> dict[str, Any]:
    base, filter_id, architecture = _source_path(root, profile, seed)
    manifest_path = base / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Rolling model PIT manifest缺少: {manifest_path.relative_to(root).as_posix()}")
    aggregate = _read_manifest(manifest_path)
    folds = aggregate.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("Rolling model PIT manifest缺少Fold records")
    # All declared Fold manifests, checkpoints, score and validation score bytes
    # must match the aggregate. No READY based on filename existence alone.
    validate_pit_fold_evidence(
        root, filter_id=filter_id, model_architecture=architecture,
        experiment_profile=profile, manifest=aggregate, point_in_time_dir_override=base,
    )
    feature_price = resolve_feature_price_source(
        model_architecture=architecture, project_root=root, create=False,
    )
    feature_trades = resolve_trade_count_feature_source(
        model_architecture=architecture, project_root=root,
    )
    kwargs = loader_kwargs_for_trade_path(
        experiment_profile=profile, project_root=root, seed=int(seed), authorized_build=False,
    )
    bundle = load_profile_continuous_ranker_data(
        filter_id=filter_id, model_architecture=architecture,
        experiment_profile=profile, project_root=root,
        preload_feature_bank=False, allow_stale_source=False,
        feature_price_source=feature_price,
        feature_trade_count_source=feature_trades,
        **kwargs,
    )
    cohort = _prepare_cohort(bundle)
    recipe = get_continuous_ranker_execution_recipe(profile)
    is_binary = recipe.objective_policy.target_builder == CONTINUOUS_RANKER_TARGET_BUILDER_BINARY_POSITIVE
    all_fold_rows: list[dict[str, Any]] = []
    all_year_rows: list[dict[str, Any]] = []
    last_observation = None
    for record in folds:
        row, ids = _verify_fold(bundle, cohort, record, model=model, profile=profile, seed=seed, root=root)
        score_start = pd.Timestamp(row["score_start"])
        pre_score = cohort["dates"].lt(score_start).to_numpy(dtype=bool)
        if row["history_start"] is not None:
            pre_score &= cohort["dates"].ge(pd.Timestamp(row["history_start"])).to_numpy(dtype=bool)
        parts = _partitions(pre_score, cohort, score_start=score_start, ids=ids)
        if parts["final_refit"] != row["final_refit"]:
            raise ValueError(f"{row['fold']} final_refit counts不能對帳")
        positive, nonpositive, positive_rate = _positive_counts(
            cohort, parts["fit_mask"], is_positive_binary=is_binary,
        )
        row.update({
            "pre_score_groups": parts["pre_score_groups"],
            "valid_target": parts["valid_target"],
            "missing_target": parts["missing_target"],
            "scope_excluded": parts["scope_excluded"],
            "not_matured": parts["not_matured"],
            "positive": positive, "nonpositive": nonpositive,
            "positive_rate": positive_rate,
        })
        year_rows = _year_rows(
            model, seed, row["fold"], cohort, {**parts, "pre_score_mask": pre_score},
            is_positive_binary=is_binary,
        )
        for key in ("pre_score_groups", "valid_target", "missing_target", "scope_excluded", "not_matured", "final_refit"):
            if sum(item[key] for item in year_rows) != parts[key]:
                raise ValueError(f"{row['fold']} 年度{key}加總不能對帳")
        all_fold_rows.append(row)
        all_year_rows.extend(year_rows)
        if last_observation is None or row["score_start"] > last_observation["score_start"]:
            last_observation = row
    if last_observation is None:
        raise ValueError("沒有合法已完成的PIT Fold")
    last_row = last_observation
    source_meta = dict(bundle.target_manifest.get("target_contract") or {})
    label_mode = str(recipe.target_policy.materialization_mode)
    sample_mode = str(bundle.profile.training_sample_scope)
    return {
        "model": model, "profile": profile, "seed": seed, "status": "VERIFIED",
        "sample_scope": sample_mode,
        "label_materialization": label_mode,
        "scope": ("daily" if sample_mode == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS else "event")
                 + (" / trade-path" if "trade_path" in label_mode else " / market"),
        "target_kind": source_meta.get("target_kind"),
        "target_id": str(bundle.profile.continuous_target_id),
        "target_available_from": bundle.target_manifest.get("risk_param_coverage_start"),
        "raw_groups": len(cohort["dates"]),
        "raw_event_rows": len(bundle.event_group_index),
        "all_target_valid": int(cohort["valid"].sum()),
        "last_fold": last_row,
        "seed_coverage": {**last_row},
        "folds": all_fold_rows, "years": all_year_rows,
        "evidence": {"manifest": manifest_path.relative_to(root).as_posix(),
                     "manifest_sha256": build_file_manifest(manifest_path)["sha256"],
                     "fold_count": len(folds), "source_contract_verified": True,
                     "fold_artifacts_verified": True},
    }


def _model_summary(model: str, entries: list[dict[str, Any]], expected_seeds: int) -> dict[str, Any]:
    verified = [x for x in entries if x["status"] == "VERIFIED"]
    if not verified:
        return {"model": model, "scope": "-", "seeds": 0, "expected_seeds": expected_seeds,
                "folds": None, "raw_groups": None, "target_valid": None,
                "final_refit": None, "fit_rate": None, "first_fit_date": None,
                "last_fit_date": None, "status": "BLOCKED"}
    observed = [row["seed_coverage"] for row in verified]
    avg = lambda key: int(round(float(np.mean([x[key] for x in observed]))))
    return {
        "model": model, "scope": verified[0]["scope"], "seeds": len(verified),
        "expected_seeds": expected_seeds, "folds": min(len(x["folds"]) for x in verified),
        "raw_groups": avg("pre_score_groups"), "target_valid": avg("valid_target"),
        "final_refit": avg("final_refit"),
        "fit_rate": float(np.mean([100.0 * x["final_refit"] / x["pre_score_groups"]
                                  if x["pre_score_groups"] else 0.0 for x in observed])),
        "first_fit_date": min(x["last_fold"]["fit_start"] for x in verified),
        "last_fit_date": max(x["last_fold"]["fit_end"] for x in verified),
        "status": "VERIFIED" if len(verified) == expected_seeds else "PARTIAL",
    }


def build_report(definition, *, project_root: Path) -> dict[str, Any]:
    """Validate every configured Model x benchmark Seed without best-seed selection."""
    require_approved_report_contract(REPORT_CONTRACT_ID)
    root = Path(project_root).resolve()
    seeds = _seeds()
    inventory = _inventory()
    by_model: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for model, profile in inventory:
        by_model[model] = []
        for seed in seeds:
            try:
                row = _audit_seed(root, model=model, profile=profile, seed=seed)
            except (ValueError, KeyError, TypeError, FileNotFoundError, OSError, RuntimeError) as exc:
                # This is an Audit, not a producer. Preserve per-seed diagnostic
                # and continue with other models; missing truth cannot be faked.
                row = {"model": model, "profile": profile, "seed": seed,
                       "status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
                errors.append(dict(row))
            by_model[model].append(row)
            print(f"[DL FIT AUDIT] {model} seed={seed} {row['status']}", flush=True)
    model_rows = [_model_summary(model, by_model[model], len(seeds)) for model, _ in inventory]
    seed_rows = [entry["seed_coverage"] for values in by_model.values() for entry in values if entry["status"] == "VERIFIED"]
    fold_rows = [fold for values in by_model.values() for entry in values if entry["status"] == "VERIFIED" for fold in entry["folds"]]
    year_rows = [year for values in by_model.values() for entry in values if entry["status"] == "VERIFIED" for year in entry["years"]]
    if not seed_rows:
        raise RuntimeError(f"DL FIT Audit沒有任何可驗證來源，禁止發布空的有效結果；blocked={len(errors)}")
    status = "VERIFIED" if not errors else "PARTIAL"
    return {
        "schema_version": 1, "status": status,
        "benchmark": get_robustness_benchmark_policy_snapshot(benchmark_id=MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID)["benchmark_id"],
        "seeds": list(seeds), "model_count": len(inventory),
        "verified_seed_count": len(seed_rows), "expected_seed_count": len(inventory) * len(seeds),
        "models": model_rows, "seed_coverage": seed_rows,
        "folds": fold_rows, "yearly": year_rows,
        "blocked": errors,
        "source_evidence": [
            {key: value for key, value in entry.items() if key not in {"folds", "years", "seed_coverage"}}
            for values in by_model.values() for entry in values
        ],
        "limitations": [
            "Daily stock-day universe與trade-path/event supervision為不同軸；不能將Daily scoring誤稱Daily FIT標籤。",
            "Last-Fold每Seed的pre-score/groups是可比較的橫截面；Rolling Fold重疊，不可將各Fold FIT加總當獨立觀測。",
            "Model summary counts為已驗證Seeds最新Fold的平均（不是聯集Unique樣本數）。",
            "No target只代表source未提供合法標籤；無schema證據不能細分成無訊號、未成交或其他失效原因。",
            "Label unmatured只限target已有效且時間未達score cutoff者；NULL不補零、不補負類。",
            "只有source contract、已簽章Fold檔案和canonical重算一致的Seed列為VERIFIED；其餘為BLOCKED。",
            "Binary positive-rate僅對宣告binary_positive_from_raw的訓練目標適用；各Fold正負數已分開登錄。",
        ],
    }


def _render_table(report_id: str, section: str, table_name: str, rows: list[dict[str, Any]], *, target: str) -> str:
    spec = table_contract(report_id, section, table_name)
    rendered = [format_contract_row(spec, r) for r in rows]
    return (render_table(spec.headers, rendered, alignments=spec.alignments)
            if target == "console" else markdown_table(spec.headers, rendered))


def render_result(result: Mapping[str, Any], *, target: str) -> str:
    require_approved_report_contract(REPORT_CONTRACT_ID)
    if target not in {"console", "markdown"}:
        raise ValueError("unsupported Audit render target")
    lines: list[str] = [audit_title("DL FIT 年度／有效樣本／Event-Daily Coverage", target=target)]
    lines.append(f"Status={result['status']} | Model={result['model_count']} | Verified Seed={result['verified_seed_count']}/{result['expected_seed_count']} | Benchmark={result['benchmark']}")
    for number, (section, table, key) in enumerate((
        ("model_coverage", "models", "models"),
        ("seed_coverage", "seeds", "seed_coverage"),
        ("fold_coverage", "folds", "folds"),
        ("yearly_coverage", "years", "yearly"),
    ), 1):
        lines.append(audit_section(section_contract(REPORT_CONTRACT_ID, section).title, number, target=target))
        rows = result[key]
        if target == "console" and key in {"folds", "yearly"}:
            # Full fold/year evidence remains in the published Markdown/JSON.
            lines.append(f"全量 {len(rows):,} 筆列於本次 audit.md / audit.json；Console顯示每模型第一個已驗證Seed（年度僅最新Fold）。")
            first = {}
            for item in result["seed_coverage"]:
                first.setdefault(item["model"], item["seed"])
            latest_fold = {r["model"]: r["fold"] for r in result["seed_coverage"]
                           if first.get(r["model"]) == r["seed"]}
            rows = [r for r in rows
                    if first.get(r["model"]) == r["seed"]
                    and (key != "yearly" or r["fold"] == latest_fold.get(r["model"]))]
        lines.append(_render_table(REPORT_CONTRACT_ID, section, table, rows, target=target))
    lines.append(audit_section(section_contract(REPORT_CONTRACT_ID, "evidence").title, 5, target=target))
    for item in result["limitations"]:
        lines.append(f"- {item}")
    if result["blocked"]:
        lines.append("Blocked seeds（來源缺少或FIT不匹配，未列入任何統計）：")
        for item in result["blocked"]:
            lines.append(f"- {item['model']} seed={item['seed']}: {item['reason']}")
    return "\n\n".join(lines)


def run_formal_audit(definition, *, project_root: Path, quiet: bool = False) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result = build_report(definition, project_root=root)
    console = render_result(result, target="console")
    markdown = render_result(result, target="markdown")
    saved = persist_reusable_report(
        definition=definition, project_root=root, payload=result,
        console_text=console, markdown_text=markdown,
    )
    if not quiet:
        print(console)
    return saved


__all__ = ["SUPPORTED_AUDIT_TYPE", "build_report", "collect_status", "render_result", "run_formal_audit"]

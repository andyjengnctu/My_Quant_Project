"""Profile-driven daily-universal continuous-ranker orchestration.

Learning/loss/epoch-selection stay centralized in ``train_continuous_ranker``. This module
only owns the daily stock/day sample universe, no-lookahead split, and daily/candidate
validation views so additional Daily DL experiments do not fork a new trainer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd

from config.breakout_quality import (
    CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
    TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    get_continuous_ranker_research_spec,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
)
from filters.breakout_quality.daily_ranker_data import (
    build_daily_ranker_split,
    load_daily_universal_ranker_data,
    select_breakout_candidate_group_ids,
)
from filters.breakout_quality.models.factory import count_trainable_parameters
from filters.breakout_quality.ranker_sample_contract import (
    build_score_eligibility_contract,
    resolve_forward_oos_score_group_ids,
)
from filters.breakout_quality.ranking_score_store import DAILY_RANKER_OOS_SCORE_FILENAME
from filters.breakout_quality.workflow_io import PROJECT_ROOT, write_json
from core.console_report import print_artifact_paths

from services.breakout_quality import ranker_training as ranker_api
from services.breakout_quality.continuous_ranker_pipeline import resolve_ranker_execution_plan

DAILY_SPLIT_FILENAME = "daily_split_by_date.csv"




def _empty_split_metrics(group_count: int, reason: str) -> dict:
    return {
        "group_count": int(group_count),
        "not_evaluated_reason": str(reason),
        "percentile_target_count": 0,
        "mse_vs_daily_percentile": None,
        "global_spearman_vs_raw_target": None,
        "global_spearman_vs_daily_percentile": None,
        "rankable_date_count": 0,
        "mean_daily_spearman": None,
        "median_daily_spearman": None,
        "pairwise_concordance": None,
        "comparable_pair_count": 0,
        "top_k_quality": None,
        "score_mean": None,
        "score_std": None,
        "top_score_decile_raw_target_mean": None,
        "bottom_score_decile_raw_target_mean": None,
        "binary_pr_auc": None,
        "p_at_50pct": None,
        "p_at_60pct": None,
        "p_at_70pct": None,
        "raw_r_regression": None,
    }


def _date_split_frame(
    bundle,
    split,
    *,
    forward_score_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    selection_dates = set(dates.iloc[split.selection_ids].tolist())
    train_dates = set(dates.iloc[split.inner_train_ids].tolist())
    validation_dates = set(dates.iloc[split.validation_ids].tolist())
    oos_group_ids = (
        split.oos_ids
        if forward_score_ids is None
        else np.asarray(forward_score_ids, dtype=np.int64)
    )
    oos_dates = set(dates.iloc[oos_group_ids].tolist())
    rows = []
    for date_value in sorted(selection_dates | oos_dates):
        if date_value in oos_dates:
            outer_split, role = "oos", "not_applicable"
        elif date_value in validation_dates:
            outer_split, role = "selection", "validation"
        elif date_value in train_dates:
            outer_split, role = "selection", "train"
        else:
            outer_split, role = "selection", "embargo"
        rows.append({"date": str(pd.Timestamp(date_value).date()), "outer_split": outer_split, "selection_role": role})
    return pd.DataFrame(rows)


def _dual_component_metrics(
    group_table: pd.DataFrame,
    group_ids: np.ndarray,
    predicted_favorable_r: np.ndarray,
    predicted_adverse_r: np.ndarray,
) -> dict:
    """Evaluate MR-13L primary component heads without changing the composite Gate."""

    ids = np.asarray(group_ids, dtype=np.int64)
    favorable_pred = np.asarray(predicted_favorable_r, dtype=np.float64)
    adverse_pred = np.asarray(predicted_adverse_r, dtype=np.float64)
    if favorable_pred.shape != ids.shape or adverse_pred.shape != ids.shape:
        raise ValueError("dual-component prediction length與group ids不一致")
    favorable_true = pd.to_numeric(
        group_table.iloc[ids]["target_favorable_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    adverse_true = pd.to_numeric(
        group_table.iloc[ids]["target_adverse_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    dates = pd.to_datetime(group_table.iloc[ids]["date"], errors="raise").to_numpy()

    def one_component(predicted: np.ndarray, actual: np.ndarray) -> dict:
        valid = np.isfinite(predicted) & np.isfinite(actual)
        if not bool(valid.any()):
            return {
                "count": 0,
                "mse": None,
                "mae": None,
                "rmse": None,
                "bias": None,
                "global_spearman": None,
                "mean_daily_spearman": None,
            }
        errors = predicted[valid] - actual[valid]
        daily = ranker_api.daily_rank_metrics(
            dates[valid], predicted[valid], actual[valid]
        )
        mse = float(np.mean(np.square(errors)))
        return {
            "count": int(valid.sum()),
            "mse": mse,
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(mse)),
            "bias": float(np.mean(errors)),
            "global_spearman": ranker_api.calculate_spearman(
                predicted[valid], actual[valid]
            ),
            "mean_daily_spearman": daily.get("mean_daily_spearman"),
        }

    return {
        "favorable_mfe_r": one_component(favorable_pred, favorable_true),
        "adverse_to_peak_r": one_component(adverse_pred, adverse_true),
    }


def _render_markdown(payload: dict) -> str:
    def fmt(value, digits=4):
        return "-" if value is None else f"{float(value):.{digits}f}"

    objective = str(payload["training"].get("objective") or "")
    direct_r = objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
    dual_component_r = objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION
    source_dataset = dict(payload.get("source_dataset") or {})
    target_manifest = dict(payload.get("target_manifest") or {})
    lines = [
        "# Daily Universal Continuous Model Report",
        "",
        f"- Experiment：`{payload['experiment']}`",
        f"- Profile：`{payload['experiment_profile']}`",
        f"- Sample scope：`{payload['training']['sample_scope']}`",
        f"- Target：`{payload['training']['target']}`",
        f"- Target-valid / score-eligible：`{int(source_dataset.get('target_valid_sample_count', 0) or 0):,}` / `{int(source_dataset.get('score_eligible_sample_count', 0) or 0):,}`",
        f"- Risk-param coverage start：`{source_dataset.get('risk_param_coverage_start') or '-'}`",
        f"- Context features：`{', '.join(target_manifest.get('context_features') or []) or '-'}`",
        f"- Score semantic：`{payload.get('score_semantic_id')}`",
        f"- Selected epoch：`{payload['training']['selected_epoch']}`",
        "- Feature storage：`lazy canonical OHLCV windows`；未建立 expanded daily 300×10 feature bank。",
        "- OOS 在 checkpoint 寫入後才推論，不參與 loss／gradient／epoch selection。",
    ]
    if direct_r:
        regression_contract = dict(payload["training"].get("raw_r_regression_contract") or {})
        loss_name = str(regression_contract.get("loss") or payload["training"].get("loss") or "")
        loss_detail = (
            f"Huber delta={fmt(regression_contract.get('huber_delta_r'))}R"
            if loss_name == "huber_raw_r"
            else "raw-R MSE / conditional-mean objective"
        )
        lines.extend([
            f"- Direct-R objective：two-logit margin直接解讀為Predicted R；loss=`{loss_name}`（{loss_detail}）。",
            "",
            "## Direct-R Regression",
            "",
            "| Scope | Groups | MSE | Huber | MAE | RMSE | Bias | Pred R Mean | Target R Mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label, key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout candidate slice", "breakout_candidate_oos")):
            row = payload["split_metrics"].get(key) or {}
            reg = dict(row.get("raw_r_regression") or {})
            lines.append(
                f"| {label} | {int(row.get('group_count', 0) or 0):,} | {fmt(reg.get('mse_raw_r'))} | {fmt(reg.get('huber_loss_raw_r'))} "
                f"| {fmt(reg.get('mae_raw_r'))} | {fmt(reg.get('rmse_raw_r'))} | {fmt(reg.get('bias_raw_r'))} "
                f"| {fmt(reg.get('predicted_r_mean'))} | {fmt(reg.get('target_r_mean'))} |"
            )
    if dual_component_r:
        lines.extend([
            "- Dual-component objective：兩個既有輸出分別為Predicted adverse-to-peak R與Predicted MFE R；"
            "正式score固定為MFE R−adverse R；兩分量MSE採同一mean reduction，無auxiliary loss／lambda。",
            "",
            "## Dual Component Regression",
            "",
            "| Scope | Component | Groups | RMSE | MAE | Bias | Global rho | Daily rho |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        component_eval = dict(payload.get("dual_component_evaluation") or {})
        for scope_label, scope_key in (
            ("Validation", "validation"),
            ("Forward OOS", "oos"),
            ("Breakout candidate slice", "breakout_candidate_oos"),
        ):
            scope = dict(component_eval.get(scope_key) or {})
            for component_label, component_key in (
                ("MFE", "favorable_mfe_r"),
                ("Adverse", "adverse_to_peak_r"),
            ):
                row = dict(scope.get(component_key) or {})
                lines.append(
                    f"| {scope_label} | {component_label} | {int(row.get('count', 0) or 0):,} "
                    f"| {fmt(row.get('rmse'))} | {fmt(row.get('mae'))} | {fmt(row.get('bias'))} "
                    f"| {fmt(row.get('global_spearman'))} | {fmt(row.get('mean_daily_spearman'))} |"
                )
    lines.extend([
        "",
        "## Ranking Diagnostics",
        "",
        "| Scope | Groups | Daily rho | Global rho | Pair concordance | Top 10% Target | Bottom 10% Target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, key in (("All eligible stock-days", "oos"), ("Breakout candidate slice", "breakout_candidate_oos")):
        row = payload["split_metrics"].get(key) or {}
        pair = row.get("pairwise_concordance")
        lines.append(
            f"| {label} | {int(row.get('group_count', 0) or 0):,} | {fmt(row.get('mean_daily_spearman'))} "
            f"| {fmt(row.get('global_spearman_vs_raw_target'))} | {'-' if pair is None else f'{float(pair)*100:.2f}%'} "
            f"| {fmt(row.get('top_score_decile_raw_target_mean'))} | {fmt(row.get('bottom_score_decile_raw_target_mean'))} |"
        )
    reference_eval = dict(payload.get("reference_target_evaluation") or {})
    if reference_eval.get("available"):
        lines.extend([
            "",
            "## Common Target Reference",
            "",
            f"- Reference profile：`{reference_eval.get('reference_profile')}`",
            f"- Reference target：`{reference_eval.get('reference_target_id')}`",
            "- 此reference target只在checkpoint寫入與forward inference後計算；不參與loss、gradient、epoch selection或final refit。",
            "",
            "| Scope | Groups | Daily rho | Global rho | Pair concordance | Top 10% Target | Bottom 10% Target |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label, key in (("All eligible stock-days", "oos"), ("Breakout candidate slice", "breakout_candidate_oos")):
            row = dict((reference_eval.get("split_metrics") or {}).get(key) or {})
            pair = row.get("pairwise_concordance")
            lines.append(
                f"| {label} | {int(row.get('group_count', 0) or 0):,} | {fmt(row.get('mean_daily_spearman'))} "
                f"| {fmt(row.get('global_spearman_vs_raw_target'))} | {'-' if pair is None else f'{float(pair)*100:.2f}%'} "
                f"| {fmt(row.get('top_score_decile_raw_target_mean'))} | {fmt(row.get('bottom_score_decile_raw_target_mean'))} |"
            )
    lines.extend([
        "",
        "## Boundary",
        "",
        f"- Research identity：`{payload['model_research_id']}`；trainer由profile metadata驅動，不綁定單一MR版本。",
        f"- Pairwise reduction：`{payload['training'].get('pairwise_reduction') or '-'}`；其餘learning semantics由profile固定。",
        "- Breakout candidate slice 只作 checkpoint 後診斷；candidate membership 不進模型輸入，也不進 training sample selection。",
        "- 本模型目前 research-only，不提供 strategy runtime source，不建立 PIT scores，不執行 ROOS。",
    ])
    return "\n".join(lines) + "\n"


def run(args) -> int:
    started = time.perf_counter()
    research_spec = get_continuous_ranker_research_spec(str(args.experiment_profile))
    if research_spec.trainer_family != CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL:
        raise ValueError(
            "daily ranker orchestrator只接受daily_universal research spec: "
            f"profile={args.experiment_profile}, family={research_spec.trainer_family}"
        )
    build_started = time.perf_counter()
    progress_state = {"last_bucket": -1}

    def _daily_build_progress(processed, total, target_valid, skipped):
        total = max(int(total), 1)
        processed = int(processed)
        bucket = min(20, int(processed * 20 / total))
        if processed not in {0, total} and bucket <= int(progress_state["last_bucket"]):
            return
        progress_state["last_bucket"] = bucket
        pct = min(100.0, 100.0 * processed / total)
        elapsed = time.perf_counter() - build_started
        print(
            "[Daily target/index] "
            f"{processed}/{total} tickers ({pct:5.1f}%)｜"
            f"target_valid={int(target_valid):,}｜skipped={int(skipped)}｜"
            f"elapsed={elapsed:,.1f}s",
            flush=True,
        )

    print("[Daily target/index] 開始建立daily stock-day index與40D target...", flush=True)
    bundle = load_daily_universal_ranker_data(
        filter_id=str(args.filter_id),
        model_architecture=str(args.model_architecture),
        experiment_profile=str(args.experiment_profile),
        preload_feature_bank=bool(args.preload_feature_bank),
        allow_stale_source=bool(args.allow_stale_source),
        progress_callback=_daily_build_progress,
    )
    target_id = str(bundle.profile.continuous_target_id or "").strip()
    if not target_id:
        raise ValueError("daily-universal continuous ranker缺少target identity")
    raw_r_loss_name = (
        str(bundle.profile.loss_name)
        if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
        else None
    )
    raw_r_huber_delta = (
        float(bundle.profile.raw_r_huber_delta_r)
        if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
        and bundle.profile.raw_r_huber_delta_r is not None
        else None
    )
    split = build_daily_ranker_split(bundle, inner_validation_months=int(args.inner_validation_months))
    forward_score_ids = resolve_forward_oos_score_group_ids(bundle)

    percentile_target = np.full(bundle.raw_target.shape, np.nan, dtype=np.float32)
    selection_mask = np.zeros(bundle.raw_target.shape, dtype=bool)
    selection_mask[split.selection_ids] = True
    selection_percentiles = ranker_api.build_daily_percentile_targets(
        bundle.raw_target, selection_mask, bundle.group_table["date"]
    )
    percentile_target[split.selection_ids] = selection_percentiles[split.selection_ids]

    torch, plan = resolve_ranker_execution_plan(args)
    print(
        f"torch=device={plan.device_type}, mixed_precision={plan.mixed_precision_enabled}, "
        f"dtype={plan.autocast_dtype_name}, deterministic={plan.deterministic_algorithms}, tf32={plan.allow_tf32}"
    )
    print(
        "Daily Universal Ranker｜"
        f"score_eligible={len(bundle.group_table):,} "
        f"target_valid={int(bundle.summary.get('target_valid_sample_count', 0) or 0):,} "
        f"tickers={int(bundle.summary['ticker_count']):,} "
        f"feature_storage={bundle.summary['feature_storage']}"
    )
    if bundle.summary.get("risk_param_coverage_start"):
        print(
            "Risk-normalized target｜"
            f"risk_param_coverage_start={bundle.summary['risk_param_coverage_start']} "
            f"context={','.join(bundle.summary.get('context_features') or []) or '-'}"
        )

    epoch_selection = ranker_api.select_epoch(
        torch,
        bundle.feature_bank,
        bundle.group_context,
        bundle.group_table,
        bundle.raw_target,
        percentile_target,
        split.inner_train_ids,
        split.validation_ids,
        args=args,
        plan=plan,
        evaluate_train_metrics=False,
    )
    selected_epoch = int(epoch_selection["best_epoch"])
    model, final_history = ranker_api.fit_final(
        torch,
        bundle.feature_bank,
        bundle.group_context,
        bundle.raw_target,
        percentile_target,
        bundle.group_table,
        split.selection_ids,
        epochs=selected_epoch,
        args=args,
        plan=plan,
        phase_label="Daily Selection完整重訓",
    )

    artifact_paths, output_dir = ranker_api.resolve_training_output_paths(args)
    artifact_paths.model_dir.mkdir(parents=True, exist_ok=True)
    trainable_parameter_count = count_trainable_parameters(model)
    total_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "feature_count": int(bundle.feature_bank.shape[2]),
            "context_count": int(bundle.group_context.shape[1]),
            "sequence_length": int(bundle.feature_bank.shape[1]),
            "model_spec": bundle.model_spec.as_manifest_payload(),
            "experiment_profile": str(args.experiment_profile),
            "experiment_settings": bundle.profile.as_manifest_payload(),
            "training_objective": bundle.profile.training_objective,
            "training_sample_scope": bundle.profile.training_sample_scope,
            "continuous_target_contract": bundle.target_manifest.get("target_contract"),
            "selected_epoch": selected_epoch,
            "torch_execution": plan.as_manifest_payload(),
            "trainable_parameter_count": int(trainable_parameter_count),
            "total_parameter_count": int(total_parameter_count),
        },
        artifact_paths.model_path,
    )

    # OOS target ranks and inference are intentionally deferred until after checkpoint write.
    oos_mask = np.zeros(bundle.raw_target.shape, dtype=bool)
    oos_mask[split.oos_ids] = True
    oos_percentiles = ranker_api.build_daily_percentile_targets(
        bundle.raw_target, oos_mask, bundle.group_table["date"]
    )
    percentile_target[split.oos_ids] = oos_percentiles[split.oos_ids]

    dual_component_evaluation = {}
    validation_components = None
    forward_components = None
    if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION:
        validation_components = ranker_api.predict_dual_component_r(
            torch, model, bundle.feature_bank, bundle.group_context, split.validation_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        forward_components = ranker_api.predict_dual_component_r(
            torch, model, bundle.feature_bank, bundle.group_context, forward_score_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        validation_scores = validation_components["model_score"]
        forward_scores = forward_components["model_score"]
        dual_component_evaluation["validation"] = _dual_component_metrics(
            bundle.group_table,
            split.validation_ids,
            validation_components["predicted_favorable_r"],
            validation_components["predicted_adverse_r"],
        )
        oos_positions = {int(group_id): pos for pos, group_id in enumerate(forward_score_ids)}
        oos_component_positions = np.asarray(
            [oos_positions[int(group_id)] for group_id in split.oos_ids], dtype=np.int64
        )
        dual_component_evaluation["oos"] = _dual_component_metrics(
            bundle.group_table,
            split.oos_ids,
            forward_components["predicted_favorable_r"][oos_component_positions],
            forward_components["predicted_adverse_r"][oos_component_positions],
        )
    else:
        validation_scores = ranker_api.predict_scores(
            torch, model, bundle.feature_bank, bundle.group_context, split.validation_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
            training_objective=bundle.profile.training_objective,
        )
        forward_scores = ranker_api.predict_scores(
            torch, model, bundle.feature_bank, bundle.group_context, forward_score_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
            training_objective=bundle.profile.training_objective,
        )
    score_by_group = np.full(len(bundle.group_table), np.nan, dtype=np.float32)
    score_by_group[forward_score_ids] = forward_scores
    oos_scores = score_by_group[split.oos_ids]
    validation_metrics = ranker_api.split_metrics(
        split.validation_ids, bundle.group_table, bundle.raw_target, percentile_target,
        validation_scores, include_top_k_quality=True,
        raw_r_regression_loss_name=raw_r_loss_name,
        raw_r_huber_delta_r=raw_r_huber_delta,
    )
    oos_metrics = ranker_api.split_metrics(
        split.oos_ids, bundle.group_table, bundle.raw_target, percentile_target,
        oos_scores, include_top_k_quality=True,
        raw_r_regression_loss_name=raw_r_loss_name,
        raw_r_huber_delta_r=raw_r_huber_delta,
    )
    candidate_ids = select_breakout_candidate_group_ids(
        bundle, split.oos_ids, allow_stale_source=bool(args.allow_stale_source)
    )
    if forward_components is not None:
        forward_position_by_group = {
            int(group_id): pos for pos, group_id in enumerate(forward_score_ids)
        }
        candidate_positions = np.asarray(
            [forward_position_by_group[int(group_id)] for group_id in candidate_ids],
            dtype=np.int64,
        )
        dual_component_evaluation["breakout_candidate_oos"] = _dual_component_metrics(
            bundle.group_table,
            candidate_ids,
            forward_components["predicted_favorable_r"][candidate_positions],
            forward_components["predicted_adverse_r"][candidate_positions],
        )
    candidate_metrics = (
        ranker_api.split_metrics(
            candidate_ids,
            bundle.group_table,
            bundle.raw_target,
            percentile_target,
            score_by_group[candidate_ids],
            include_top_k_quality=True,
            raw_r_regression_loss_name=raw_r_loss_name,
            raw_r_huber_delta_r=raw_r_huber_delta,
        )
        if len(candidate_ids) >= 2
        else _empty_split_metrics(
            len(candidate_ids), "breakout candidate OOS slice有效sample不足"
        )
    )

    reference_target_evaluation = {
        "available": False,
        "reason": "active research profile沒有設定reference target",
    }
    reference_raw_target = None
    if research_spec.reference_profile_name:
        reference_bundle = load_daily_universal_ranker_data(
            filter_id=str(args.filter_id),
            model_architecture=str(args.model_architecture),
            experiment_profile=str(research_spec.reference_profile_name),
            preload_feature_bank=False,
            allow_stale_source=bool(args.allow_stale_source),
            project_root=PROJECT_ROOT,
        )
        key_columns = ["ticker", "date", "source_pos", "label_eval_end_date"]
        if len(reference_bundle.group_table) != len(bundle.group_table) or not (
            reference_bundle.group_table[key_columns].astype(str).equals(
                bundle.group_table[key_columns].astype(str)
            )
        ):
            raise ValueError("reference target與candidate daily stock-day universe／順序不一致")
        if not np.array_equal(reference_bundle.target_valid, bundle.target_valid):
            raise ValueError("reference target與candidate target-valid universe不一致")
        reference_raw_target = np.asarray(reference_bundle.raw_target, dtype=np.float32)
        reference_percentile = np.full(reference_raw_target.shape, np.nan, dtype=np.float32)
        reference_oos_percentile = ranker_api.build_daily_percentile_targets(
            reference_raw_target, oos_mask, bundle.group_table["date"]
        )
        reference_percentile[split.oos_ids] = reference_oos_percentile[split.oos_ids]
        reference_oos_metrics = ranker_api.split_metrics(
            split.oos_ids, bundle.group_table, reference_raw_target, reference_percentile,
            oos_scores, include_top_k_quality=True,
        )
        reference_candidate_metrics = (
            ranker_api.split_metrics(
                candidate_ids, bundle.group_table, reference_raw_target, reference_percentile,
                score_by_group[candidate_ids], include_top_k_quality=True,
            )
            if len(candidate_ids) >= 2
            else _empty_split_metrics(
                len(candidate_ids), "breakout candidate OOS slice有效sample不足"
            )
        )
        reference_target_evaluation = {
            "available": True,
            "reference_profile": str(research_spec.reference_profile_name),
            "reference_target_id": str(reference_bundle.profile.continuous_target_id),
            "used_for_training_or_epoch_selection": False,
            "evaluated_after_checkpoint_write": True,
            "split_metrics": {
                "oos": reference_oos_metrics,
                "breakout_candidate_oos": reference_candidate_metrics,
            },
        }
        del reference_bundle

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / DAILY_RANKER_OOS_SCORE_FILENAME
    report_json_path = output_dir / ranker_api.RANKER_REPORT_JSON_FILENAME
    report_markdown_path = output_dir / ranker_api.RANKER_REPORT_MARKDOWN_FILENAME
    split_path = output_dir / DAILY_SPLIT_FILENAME

    oos_frame = bundle.group_table.iloc[forward_score_ids][["ticker", "date", "group_index"]].copy()
    evaluable_forward_mask = np.isin(forward_score_ids, split.oos_ids)
    oos_frame["target_raw_r"] = np.nan
    oos_frame["target_daily_percentile"] = np.nan
    oos_frame.loc[evaluable_forward_mask, "target_raw_r"] = bundle.raw_target[
        forward_score_ids[evaluable_forward_mask]
    ]
    oos_frame.loc[evaluable_forward_mask, "target_daily_percentile"] = percentile_target[
        forward_score_ids[evaluable_forward_mask]
    ]
    if reference_raw_target is not None:
        oos_frame["reference_target_raw_r"] = np.nan
        oos_frame.loc[evaluable_forward_mask, "reference_target_raw_r"] = reference_raw_target[
            forward_score_ids[evaluable_forward_mask]
        ]
    oos_frame["model_score"] = forward_scores
    if forward_components is not None:
        oos_frame["predicted_favorable_r"] = forward_components["predicted_favorable_r"]
        oos_frame["predicted_adverse_r"] = forward_components["predicted_adverse_r"]
        oos_frame["predicted_composite_r"] = forward_components["model_score"]
    if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
        oos_frame["predicted_r"] = forward_scores
    oos_frame.to_csv(score_path, index=False, encoding="utf-8-sig", compression="gzip")
    _date_split_frame(bundle, split, forward_score_ids=forward_score_ids).to_csv(
        split_path, index=False, encoding="utf-8-sig"
    )

    information_cutoff = str(pd.Timestamp(bundle.group_table.iloc[split.selection_ids]["label_eval_end_date"].max()).date())
    split_metrics = {
        "inner_train": _empty_split_metrics(len(split.inner_train_ids), "full-refit後不重跑整段inner train以避免重複daily window materialization"),
        "validation": validation_metrics,
        "selection": _empty_split_metrics(len(split.selection_ids), "full Selection metrics不另重跑；模型主Gate使用forward OOS"),
        "oos": oos_metrics,
        "breakout_candidate_oos": candidate_metrics,
    }
    payload = {
        "schema_version": ranker_api.RANKER_SCHEMA_VERSION + 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": research_spec.experiment_name,
        "phase": research_spec.phase,
        "model_research_id": research_spec.model_research_id,
        "score_semantic_id": research_spec.score_semantic_id,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
        "experiment_settings": bundle.profile.as_manifest_payload(),
        "training": {
            "objective": bundle.profile.training_objective,
            "loss": bundle.profile.loss_name,
            "sample_scope": bundle.profile.training_sample_scope,
            "training_label_scope": bundle.profile.training_label_scope,
            "target": target_id,
            "model_score": (
                "predicted_r_two_logit_margin"
                if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
                else "predicted_favorable_mfe_r_minus_predicted_adverse_to_peak_r"
                if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION
                else "softmax_pass_probability_monotonic_to_two_logit_margin"
            ),
            "batching": ranker_api.training_semantics(bundle.profile)["batching"],
            "pairwise_contract": ranker_api.training_semantics(bundle.profile)["pairwise_contract"],
            "raw_r_regression_contract": ranker_api.training_semantics(bundle.profile).get("raw_r_regression_contract"),
            "dual_component_r_regression_contract": ranker_api.training_semantics(bundle.profile).get("dual_component_r_regression_contract"),
            "pairwise_reduction": research_spec.pairwise_reduction,
            "selected_epoch": selected_epoch,
            "epoch_selection_metric": bundle.profile.epoch_selection_metric,
            "epoch_selection": epoch_selection,
            "final_refit_history": final_history,
            "seed": int(args.seed),
        },
        "split_report": split.report,
        "split_metrics": split_metrics,
        "all_group_split_metrics": split_metrics,
        "reference_target_evaluation": reference_target_evaluation,
        "dual_component_evaluation": dual_component_evaluation,
        "trade_alignment": {"available": False, "reason": "daily universal model先做模型本身驗證；未接策略trade attribution"},
        "target_manifest": bundle.target_manifest,
        "source_dataset": bundle.summary,
        "model_information_cutoff": information_cutoff,
        "oos_used_for_training_or_epoch_selection": False,
        "oos_evaluated_after_checkpoint_write": True,
        "score_eligibility_contract": build_score_eligibility_contract(bundle.profile),
        "forward_score_coverage": {
            "inference_eligible_groups": int(len(forward_score_ids)),
            "target_evaluable_groups": int(len(split.oos_ids)),
            "future_target_required_for_score": False,
        },
        "runtime_eligibility": {
            "eligible": False,
            "scope": "research_only",
            "reason": (
                f"{research_spec.model_research_id}先通過daily forward-OOS與breakout-candidate "
                "slice模型Gate後，才建立PIT／策略source"
            ),
        },
        "artifacts": {
            "model": build_file_manifest(artifact_paths.model_path),
            "oos_scores_gzip": build_file_manifest(score_path),
            "date_split": build_file_manifest(split_path),
        },
        "torch_execution": plan.as_manifest_payload(),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    write_json(report_json_path, payload)
    report_markdown_path.write_text(_render_markdown(payload), encoding="utf-8")

    manifest = {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "research_schema_version": payload["schema_version"],
        "filter_family": FILTER_FAMILY,
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "model_spec": bundle.model_spec.as_manifest_payload(),
        "experiment_profile": str(args.experiment_profile),
        "experiment_settings": bundle.profile.as_manifest_payload(),
        "training_objective": bundle.profile.training_objective,
        "training_label_scope": bundle.profile.training_label_scope,
        "training_sample_scope": bundle.profile.training_sample_scope,
        "training_semantics": ranker_api.training_semantics(bundle.profile),
        "continuous_target_id": target_id,
        "sequence_length": int(bundle.feature_bank.shape[1]),
        "feature_columns": list(FEATURE_COLUMNS),
        "context_columns": list(bundle.target_manifest.get("context_features") or []),
        "feature_storage": "lazy_from_canonical_ohlcv",
        "trainable_parameter_count": int(trainable_parameter_count),
        "total_parameter_count": int(total_parameter_count),
        "frozen_parameter_count": int(total_parameter_count - trainable_parameter_count),
        "model": build_file_manifest(artifact_paths.model_path),
        "date_split": build_file_manifest(split_path),
        "outer_oos_policy": bundle.outer_policy,
        "selected_epoch": selected_epoch,
        "model_information_cutoff": information_cutoff,
        "source_dataset": bundle.summary,
        "source_continuous_target": bundle.target_manifest,
        "reference_target_evaluation": reference_target_evaluation,
        "runtime_eligibility": payload["runtime_eligibility"],
        "score_eligibility_contract": payload["score_eligibility_contract"],
        "forward_score_coverage": payload["forward_score_coverage"],
        "research_outputs": {
            "oos_scores_gzip": build_file_manifest(score_path),
            "report_json": build_file_manifest(report_json_path),
            "report_markdown": build_file_manifest(report_markdown_path),
        },
    }
    write_json(artifact_paths.manifest_path, manifest)

    def _fmt_metric(value) -> str:
        return "-" if value is None else f"{float(value):.4f}"

    print(f"\nDaily Universal Continuous Model完成｜{research_spec.model_research_id}")
    if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
        regression = dict(oos_metrics.get("raw_r_regression") or {})
        if raw_r_loss_name == "huber_raw_r":
            primary = f"OOS Huber={_fmt_metric(regression.get('huber_loss_raw_r'))}"
        else:
            primary = f"OOS MSE={_fmt_metric(regression.get('mse_raw_r'))} | RMSE={_fmt_metric(regression.get('rmse_raw_r'))}R"
        print(
            f"selected_epoch={selected_epoch} | {primary} "
            f"| MAE={_fmt_metric(regression.get('mae_raw_r'))}R | bias={_fmt_metric(regression.get('bias_raw_r'))}R"
        )
    print(
        f"OOS daily rho={_fmt_metric(oos_metrics.get('mean_daily_spearman'))} | "
        f"pair={_fmt_metric(oos_metrics.get('pairwise_concordance'))}"
    )
    print(
        f"breakout candidate slice: groups={candidate_metrics['group_count']:,} | "
        f"daily rho={_fmt_metric(candidate_metrics.get('mean_daily_spearman'))} | "
        f"pair={_fmt_metric(candidate_metrics.get('pairwise_concordance'))}"
    )
    print_artifact_paths(
        (("Daily ranker model", artifact_paths.model_path), ("Markdown", report_markdown_path), ("OOS scores", score_path)),
        project_root=PROJECT_ROOT,
    )
    return 0


__all__ = ["run"]

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


def _render_markdown(payload: dict) -> str:
    def fmt(value, digits=4):
        return "-" if value is None else f"{float(value):.{digits}f}"

    objective = str(payload["training"].get("objective") or "")
    direct_r = objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
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
    bundle = load_daily_universal_ranker_data(
        filter_id=str(args.filter_id),
        model_architecture=str(args.model_architecture),
        experiment_profile=str(args.experiment_profile),
        preload_feature_bank=bool(args.preload_feature_bank),
        allow_stale_source=bool(args.allow_stale_source),
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
    oos_frame["model_score"] = forward_scores
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
                else "softmax_pass_probability_monotonic_to_two_logit_margin"
            ),
            "batching": ranker_api.training_semantics(bundle.profile)["batching"],
            "pairwise_contract": ranker_api.training_semantics(bundle.profile)["pairwise_contract"],
            "raw_r_regression_contract": ranker_api.training_semantics(bundle.profile).get("raw_r_regression_contract"),
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
        "context_columns": [],
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

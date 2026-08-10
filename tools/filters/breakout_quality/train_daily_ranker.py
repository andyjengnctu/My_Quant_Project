"""MR-13A daily-universal pairwise ranker orchestration.

Learning/loss/epoch-selection stay centralized in ``train_continuous_ranker``.  This module
only supplies the daily stock/day sample universe, no-lookahead split, and daily/candidate
validation views.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd

from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_target import DAILY_OPPORTUNITY_NO_TIME_TARGET_ID
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
from filters.breakout_quality.models.factory import count_trainable_parameters, require_torch
from filters.breakout_quality.ranking_score_store import DAILY_RANKER_OOS_SCORE_FILENAME
from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
from filters.breakout_quality.workflow_io import PROJECT_ROOT, write_json
from core.console_report import print_artifact_paths

DAILY_SPLIT_FILENAME = "daily_split_by_date.csv"


def _execution_plan(args):
    torch, _nn = require_torch()
    plan = resolve_torch_execution_plan(
        torch,
        requested_device=str(args.device),
        mixed_precision=bool(args.mixed_precision),
        mixed_precision_dtype=str(args.mixed_precision_dtype),
        deterministic_algorithms=bool(args.deterministic_algorithms),
        allow_tf32=bool(args.allow_tf32),
    )
    if plan.device_type == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            if "cannot set number of interop threads" not in str(exc):
                raise
    return torch, plan


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
    }


def _date_split_frame(bundle, split) -> pd.DataFrame:
    dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    selection_dates = set(dates.iloc[split.selection_ids].tolist())
    train_dates = set(dates.iloc[split.inner_train_ids].tolist())
    validation_dates = set(dates.iloc[split.validation_ids].tolist())
    oos_dates = set(dates.iloc[split.oos_ids].tolist())
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

    lines = [
        "# Daily Universal Ranker Report",
        "",
        f"- Experiment：`{payload['experiment']}`",
        f"- Profile：`{payload['experiment_profile']}`",
        f"- Sample scope：`{payload['training']['sample_scope']}`",
        f"- Target：`{payload['training']['target']}`",
        f"- Selected epoch：`{payload['training']['selected_epoch']}`",
        "- Feature storage：`lazy canonical OHLCV windows`；未建立 expanded daily 300×10 feature bank。",
        "- OOS 在 checkpoint 寫入後才推論，不參與 loss／gradient／epoch selection。",
        "",
        "## Forward OOS",
        "",
        "| Scope | Groups | Daily rho | Global rho | Pair concordance | Top 10% Target | Bottom 10% Target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
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
        "- 第一版固定沿用 MR-12B 的 InceptionTime、pairwise logistic、optimizer、LR 與 epoch metric；scientific change 只包含 daily stock-day sample universe 與對應 daily target identity。",
        "- Breakout candidate slice 只作 checkpoint 後診斷；candidate membership 不進模型輸入，也不進 training sample selection。",
        "- 本模型目前 research-only，不提供 strategy runtime source，不建立 PIT scores，不執行 ROOS。",
    ])
    return "\n".join(lines) + "\n"


def run(args, *, ranker_impl) -> int:
    started = time.perf_counter()
    bundle = load_daily_universal_ranker_data(
        filter_id=str(args.filter_id),
        model_architecture=str(args.model_architecture),
        experiment_profile=str(args.experiment_profile),
        preload_feature_bank=bool(args.preload_feature_bank),
        allow_stale_source=bool(args.allow_stale_source),
    )
    if str(bundle.profile.continuous_target_id) != DAILY_OPPORTUNITY_NO_TIME_TARGET_ID:
        raise ValueError("MR-13A target identity不一致")
    split = build_daily_ranker_split(bundle, inner_validation_months=int(args.inner_validation_months))

    percentile_target = np.full(bundle.raw_target.shape, np.nan, dtype=np.float32)
    selection_mask = np.zeros(bundle.raw_target.shape, dtype=bool)
    selection_mask[split.selection_ids] = True
    selection_percentiles = ranker_impl.build_daily_percentile_targets(
        bundle.raw_target, selection_mask, bundle.group_table["date"]
    )
    percentile_target[split.selection_ids] = selection_percentiles[split.selection_ids]

    torch, plan = _execution_plan(args)
    print(
        f"torch=device={plan.device_type}, mixed_precision={plan.mixed_precision_enabled}, "
        f"dtype={plan.autocast_dtype_name}, deterministic={plan.deterministic_algorithms}, tf32={plan.allow_tf32}"
    )
    print(
        "Daily Universal Ranker｜"
        f"samples={len(bundle.group_table):,} tickers={int(bundle.summary['ticker_count']):,} "
        f"feature_storage={bundle.summary['feature_storage']}"
    )

    epoch_selection = ranker_impl._select_epoch(
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
    model, final_history = ranker_impl._fit_final(
        torch,
        bundle.feature_bank,
        bundle.group_context,
        percentile_target,
        bundle.group_table,
        split.selection_ids,
        epochs=selected_epoch,
        args=args,
        plan=plan,
        phase_label="Daily Selection完整重訓",
    )

    artifact_paths, output_dir = ranker_impl._training_output_paths(args)
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
    oos_percentiles = ranker_impl.build_daily_percentile_targets(
        bundle.raw_target, oos_mask, bundle.group_table["date"]
    )
    percentile_target[split.oos_ids] = oos_percentiles[split.oos_ids]

    validation_scores = ranker_impl._predict_scores(
        torch, model, bundle.feature_bank, bundle.group_context, split.validation_ids,
        batch_size=int(args.evaluation_batch_size), plan=plan,
    )
    oos_scores = ranker_impl._predict_scores(
        torch, model, bundle.feature_bank, bundle.group_context, split.oos_ids,
        batch_size=int(args.evaluation_batch_size), plan=plan,
    )
    validation_metrics = ranker_impl._split_metrics(
        split.validation_ids, bundle.group_table, bundle.raw_target, percentile_target,
        validation_scores, include_top_k_quality=True,
    )
    oos_metrics = ranker_impl._split_metrics(
        split.oos_ids, bundle.group_table, bundle.raw_target, percentile_target,
        oos_scores, include_top_k_quality=True,
    )
    score_by_group = np.full(len(bundle.group_table), np.nan, dtype=np.float32)
    score_by_group[split.oos_ids] = oos_scores
    candidate_ids = select_breakout_candidate_group_ids(
        bundle, split.oos_ids, allow_stale_source=bool(args.allow_stale_source)
    )
    candidate_metrics = (
        ranker_impl._split_metrics(
            candidate_ids,
            bundle.group_table,
            bundle.raw_target,
            percentile_target,
            score_by_group[candidate_ids],
            include_top_k_quality=True,
        )
        if len(candidate_ids) >= 2
        else _empty_split_metrics(
            len(candidate_ids), "breakout candidate OOS slice有效sample不足"
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / DAILY_RANKER_OOS_SCORE_FILENAME
    report_json_path = output_dir / ranker_impl.RANKER_REPORT_JSON_FILENAME
    report_markdown_path = output_dir / ranker_impl.RANKER_REPORT_MARKDOWN_FILENAME
    split_path = output_dir / DAILY_SPLIT_FILENAME

    oos_frame = bundle.group_table.iloc[split.oos_ids][["ticker", "date", "group_index"]].copy()
    oos_frame["target_raw_r"] = bundle.raw_target[split.oos_ids]
    oos_frame["target_daily_percentile"] = percentile_target[split.oos_ids]
    oos_frame["model_score"] = oos_scores
    oos_frame.to_csv(score_path, index=False, encoding="utf-8-sig", compression="gzip")
    _date_split_frame(bundle, split).to_csv(split_path, index=False, encoding="utf-8-sig")

    information_cutoff = str(pd.Timestamp(bundle.group_table.iloc[split.selection_ids]["label_eval_end_date"].max()).date())
    split_metrics = {
        "inner_train": _empty_split_metrics(len(split.inner_train_ids), "full-refit後不重跑整段inner train以避免重複daily window materialization"),
        "validation": validation_metrics,
        "selection": _empty_split_metrics(len(split.selection_ids), "full Selection metrics不另重跑；模型主Gate使用forward OOS"),
        "oos": oos_metrics,
        "breakout_candidate_oos": candidate_metrics,
    }
    payload = {
        "schema_version": ranker_impl.RANKER_SCHEMA_VERSION + 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "MR-13A Daily Universal No-time Pairwise Ranker",
        "phase": "13A",
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
            "target": DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
            "model_score": "softmax_pass_probability_monotonic_to_two_logit_margin",
            "batching": ranker_impl.PAIRWISE_TRAINING_CONTRACT["batching"],
            "pairwise_contract": dict(ranker_impl.PAIRWISE_TRAINING_CONTRACT),
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
        "runtime_eligibility": {
            "eligible": False,
            "scope": "research_only",
            "reason": "MR-13A先通過daily forward-OOS與breakout-candidate slice模型Gate後，才建立PIT／策略source",
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
        "training_semantics": ranker_impl._training_semantics(bundle.profile),
        "continuous_target_id": DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
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
        "research_outputs": {
            "oos_scores_gzip": build_file_manifest(score_path),
            "report_json": build_file_manifest(report_json_path),
            "report_markdown": build_file_manifest(report_markdown_path),
        },
    }
    write_json(artifact_paths.manifest_path, manifest)

    def _fmt_metric(value) -> str:
        return "-" if value is None else f"{float(value):.4f}"

    print("\nDaily Universal Ranker完成")
    print(
        f"selected_epoch={selected_epoch} | OOS daily rho={_fmt_metric(oos_metrics.get('mean_daily_spearman'))} | "
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

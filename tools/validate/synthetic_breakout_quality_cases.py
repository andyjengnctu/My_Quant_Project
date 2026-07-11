from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.breakout_policy import (
    BREAKOUT_DEFAULT_HIGH_LEN,
    BREAKOUT_HIGH_LEN_SEARCH_MAX,
    BREAKOUT_HIGH_LEN_SEARCH_MIN,
    BREAKOUT_HIGH_LEN_SEARCH_STEP,
    build_breakout_optimizer_high_len_values,
)
from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES,
    BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
    build_breakout_quality_default_high_len_values,
)
from filters.breakout_quality.artifacts import (
    build_file_manifest,
    load_model_artifact_contract,
    load_runtime_artifact_contract,
    load_split_assignment_frame,
)
from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONTEXT_COLUMNS,
    DEFAULT_LABEL_POLICY,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
    SELECTION_ROLE_EMBARGO,
    SELECTION_ROLE_IGNORE,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_NOT_APPLICABLE,
    SELECTION_ROLE_TRAIN,
    SELECTION_ROLE_VALIDATION,
    OUTER_SPLIT_OOS,
    OUTER_SPLIT_OUT_OF_SCOPE,
    OUTER_SPLIT_SELECTION,
    RUNTIME_SCOPE_FORWARD_OOS,
    SCORE_COLUMN,
    SCORE_COMPARISON,
    SCORE_TABLE_REQUIRED_COLUMNS,
    SCORE_TABLE_SCHEMA_VERSION,
    SCORE_THRESHOLD_SOURCE,
    SPLIT_ASSIGNMENT_REQUIRED_COLUMNS,
    SPLIT_ASSIGNMENT_SCHEMA_VERSION,
    TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
)
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_research_score_path,
    resolve_filter_report_json_path,
    resolve_filter_report_markdown_path,
)
from filters.breakout_quality.score_store import build_pass_condition_from_score_table, load_score_table
from filters.breakout_quality.source_inventory import build_source_data_inventory
from core.signal_utils import generate_signals
from core.strategy_params import V16StrategyParams
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE
from tools.filters.breakout_quality import common as breakout_quality_common
from tools.filters.breakout_quality.report import (
    build_report_payload,
    render_console_summary,
    render_markdown_report,
)
from filters.breakout_quality.splits import (
    build_selection_oos_split_assignments,
    compute_outer_policy_fingerprint,
    resolve_breakout_quality_outer_policy,
)

from .checks import add_check


def validate_breakout_quality_policy_single_source_case(_base_params):
    case_id = "BREAKOUT_QUALITY_POLICY_SSOT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    optimizer_values = build_breakout_optimizer_high_len_values()
    quality_values = build_breakout_quality_default_high_len_values()
    search_spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE["high_len"]

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_default_uses_config",
        int(BREAKOUT_DEFAULT_HIGH_LEN),
        int(BREAKOUT_PARAM_SPECS["high_len"]["default"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "optimizer_range_uses_config",
        (int(BREAKOUT_HIGH_LEN_SEARCH_MIN), int(BREAKOUT_HIGH_LEN_SEARCH_MAX), int(BREAKOUT_HIGH_LEN_SEARCH_STEP)),
        (int(search_spec["low"]), int(search_spec["high"]), int(search_spec["step"])),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "quality_coverage_contains_optimizer_grid",
        True,
        set(optimizer_values).issubset(set(quality_values)),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "quality_coverage_contains_strategy_default",
        True,
        int(BREAKOUT_DEFAULT_HIGH_LEN) in set(quality_values),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "fixed_threshold_is_user_configured_and_legal",
        True,
        0.0 <= float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD) <= 1.0,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "training_defaults_are_user_configured_and_legal",
        True,
        int(BREAKOUT_QUALITY_DEFAULT_EPOCHS) >= 1
        and int(BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE) >= 1
        and float(BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE) > 0.0
        and int(BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED) >= 0
        and int(BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES) >= 1,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inner_validation_config_has_legal_types_and_ranges",
        True,
        isinstance(BREAKOUT_QUALITY_USE_INNER_VALIDATION, bool)
        and int(BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS) >= 1
        and int(BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE) >= 0
        and float(BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA) >= 0.0
        and int(BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES) >= 1,
    )
    try:
        V16StrategyParams(breakout_quality_filter_id=" ")
        empty_filter_id_rejected = False
    except ValueError as exc:
        empty_filter_id_rejected = "breakout_quality_filter_id" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "empty_filter_id_rejected_without_silent_default",
        True,
        empty_filter_id_rejected,
    )

    with tempfile.TemporaryDirectory(prefix="breakout_quality_source_inventory_") as temp_dir:
        project_root = Path(temp_dir)
        source_dir = project_root / "data" / "tw_stock_data_vip_reduced"
        source_dir.mkdir(parents=True, exist_ok=True)
        first_path = source_dir / "2330.csv"
        second_path = source_dir / "0050.csv"
        first_path.write_text("Date,Open\n2026-01-01,100\n", encoding="utf-8")
        second_path.write_text("Date,Open\n2026-01-01,50\n", encoding="utf-8")

        inventory_before = build_source_data_inventory(project_root, "reduced")
        inventory_repeat = build_source_data_inventory(project_root, "reduced")
        first_path.write_text(
            "Date,Open\n2026-01-01,100\n2026-01-02,101\n",
            encoding="utf-8",
        )
        inventory_after = build_source_data_inventory(project_root, "reduced")

        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "source_inventory_is_stable_without_changes",
            inventory_before,
            inventory_repeat,
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "source_inventory_detects_csv_update",
            True,
            inventory_before["csv_inventory_sha256"] != inventory_after["csv_inventory_sha256"],
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "source_inventory_tracks_unique_csv_count",
            2,
            inventory_after["csv_file_count"],
        )

        output_dir = project_root / "outputs" / "filters" / "breakout_quality" / "synthetic_quality"
        output_dir.mkdir(parents=True, exist_ok=True)
        stale_summary = {
            "filter_id": "synthetic_quality",
            "dataset": "reduced",
            "policy": DEFAULT_LABEL_POLICY.as_manifest_payload(),
            "feature_columns": list(FEATURE_COLUMNS),
            "context_columns": list(CONTEXT_COLUMNS),
            "source_data_inventory": inventory_before,
        }
        (output_dir / "dataset_summary.json").write_text(
            json.dumps(stale_summary),
            encoding="utf-8",
        )
        stale_source_rejected = False
        with (
            patch.object(breakout_quality_common, "PROJECT_ROOT", project_root),
            patch.object(
                breakout_quality_common,
                "dataset_output_dir",
                return_value=output_dir,
            ),
        ):
            try:
                breakout_quality_common.load_validated_dataset_bundle(
                    "synthetic_quality",
                    require_current_source=True,
                )
            except ValueError as exc:
                stale_source_rejected = "來源 CSV 已更新" in str(exc)
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "standalone_training_rejects_stale_source_dataset",
            True,
            stale_source_rejected,
        )

    summary["optimizer_high_len_count"] = len(optimizer_values)
    summary["quality_high_len_count"] = len(quality_values)
    return results, summary


def validate_breakout_quality_chronological_embargo_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CHRONOLOGICAL_EMBARGO"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    date_and_end = (
        ("2024-12-31", "2025-01-02"),
        ("2025-01-10", "2025-01-20"),
        ("2025-02-20", "2025-03-05"),
        ("2025-03-10", "2025-03-20"),
        ("2025-04-20", "2025-05-05"),
        ("2025-05-10", "2025-05-20"),
        ("2025-06-20", "2025-07-05"),
    )
    rows = []
    for event_date, label_end_date in date_and_end:
        for high_len in (100, 105):
            rows.append(
                {
                    "ticker": "2330",
                    "date": event_date,
                    "label_eval_end_date": label_end_date,
                    "high_len": high_len,
                }
            )
    events = pd.DataFrame(rows)
    labels = np.asarray([0, 1] * len(date_and_end), dtype=np.int64)
    outer_policy = {
        "policy_source": "core.walk_forward_policy.synthetic_override",
        "selection_start_date": "2025-01-01",
        "selection_end_date": "2025-04-30",
        "oos_start_date": "2025-05-01",
        "configured_oos_end_date": "2025-06-30",
        "effective_oos_end_date": "2025-06-30",
    }
    outer_policy["policy_fingerprint_sha256"] = compute_outer_policy_fingerprint(
        outer_policy
    )

    (
        assignments_off,
        train_off,
        validation_off,
        refit_off,
        oos_off,
        report_off,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=False,
        inner_validation_months=2,
        early_stopping_enabled=False,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_off_uses_full_selection",
        (6, 0, 6, False, False),
        (
            len(train_off),
            len(validation_off),
            len(refit_off),
            report_off["inner_validation_used"],
            report_off["early_stopping_used"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_off_selection_oos_embargo_rows",
        2,
        report_off["selection_oos_embargo_row_count"],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_off_role_counts",
        {
            SELECTION_ROLE_TRAIN: 6,
            SELECTION_ROLE_VALIDATION: 0,
            SELECTION_ROLE_INNER_EMBARGO: 0,
            SELECTION_ROLE_EMBARGO: 2,
            SELECTION_ROLE_IGNORE: 0,
            SELECTION_ROLE_NOT_APPLICABLE: 6,
        },
        report_off["selection_role_counts"],
    )

    (
        assignments_on,
        train_on,
        validation_on,
        refit_on,
        oos_on,
        report_on,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=True,
        inner_validation_months=2,
        early_stopping_enabled=True,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_on_period_and_roles",
        ("2025-03-01", 2, 2, 2, 6, True, True),
        (
            report_on["inner_validation_start_date"],
            len(train_on),
            len(validation_on),
            report_on["inner_train_validation_embargo_row_count"],
            len(refit_on),
            report_on["inner_validation_used"],
            report_on["early_stopping_used"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_on_refit_recovers_inner_embargo_rows",
        True,
        set(refit_on.tolist())
        == set(train_on.tolist())
        | set(validation_on.tolist())
        | set(
            np.flatnonzero(
                assignments_on["selection_role"].to_numpy()
                == SELECTION_ROLE_INNER_EMBARGO
            ).tolist()
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_on_role_counts",
        {
            SELECTION_ROLE_TRAIN: 2,
            SELECTION_ROLE_VALIDATION: 2,
            SELECTION_ROLE_INNER_EMBARGO: 2,
            SELECTION_ROLE_EMBARGO: 2,
            SELECTION_ROLE_IGNORE: 0,
            SELECTION_ROLE_NOT_APPLICABLE: 6,
        },
        report_on["selection_role_counts"],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inner_train_label_information_before_validation",
        True,
        pd.to_datetime(events.iloc[train_on]["label_eval_end_date"]).max()
        < pd.Timestamp(report_on["inner_validation_start_date"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "final_refit_label_information_before_oos",
        True,
        pd.to_datetime(events.iloc[refit_on]["label_eval_end_date"]).max()
        < pd.Timestamp(report_on["oos_start_date"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "oos_evaluable_and_tail_rows",
        (2, 2),
        (len(oos_on), report_on["oos_label_after_end_row_count"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "outer_and_inner_overlap_forbidden",
        (0, 0, 0, 0),
        (
            report_on["overlap_group_count"],
            report_on["overlap_event_date_count"],
            report_on["inner_train_validation_overlap_group_count"],
            report_on["inner_train_validation_overlap_event_date_count"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "outside_selection_has_no_selection_role",
        True,
        bool(
            (
                assignments_on.loc[
                    assignments_on["outer_split"] != OUTER_SPLIT_SELECTION,
                    "selection_role",
                ]
                == SELECTION_ROLE_NOT_APPLICABLE
            ).all()
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "split_assignment_key_unique",
        False,
        bool(
            assignments_on.duplicated(
                ["ticker", "date", "high_len"],
                keep=False,
            ).any()
        ),
    )

    project_root = Path(__file__).resolve().parents[2]
    rolling_fold_policy = resolve_breakout_quality_outer_policy(
        project_root,
        source_data_end_date="2021-12-31",
        environ={
            "V16_WF_SELECTION_START_DATE": "2019-01-01",
            "V16_WF_TRAIN_START_DATE": "2019-01-01",
            "V16_WF_SEARCH_TRAIN_END_DATE": "2020-12-31",
            "V16_WF_OOS_START_DATE": "2021-01-01",
            "V16_WF_OOS_END_DATE": "2021-12-31",
        },
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_fold_reuses_standard_walk_forward_overrides",
        ("2019-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        (
            rolling_fold_policy["selection_start_date"],
            rolling_fold_policy["selection_end_date"],
            rolling_fold_policy["oos_start_date"],
            rolling_fold_policy["effective_oos_end_date"],
        ),
    )
    summary["split_report_off"] = report_off
    summary["split_report_on"] = report_on
    return results, summary


def _clear_breakout_quality_caches() -> None:
    load_model_artifact_contract.cache_clear()
    load_runtime_artifact_contract.cache_clear()
    load_split_assignment_frame.cache_clear()
    load_score_table.cache_clear()



def _validate_signal_runtime_wiring(results, case_id: str) -> None:
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "Open": [9.0, 9.0, 9.0, 10.5, 10.0],
            "High": [10.0, 10.0, 10.0, 12.0, 12.0],
            "Low": [8.0, 8.0, 8.0, 10.0, 9.0],
            "Close": [9.0, 9.0, 9.0, 11.0, 10.0],
            "Volume": [100.0] * 5,
        },
        index=dates,
    )
    params = SimpleNamespace(
        atr_len=2,
        atr_times_trail=3.0,
        atr_buy_tol=1.0,
        high_len=2,
        use_breakout_buy=True,
        use_breakout_ema_filter=False,
        use_bb=False,
        use_vol=False,
        use_breakout_return_filter=False,
        use_breakout_false_filter=False,
        use_breakout_quality_filter=True,
        breakout_quality_filter_id="synthetic_quality",
        breakout_quality_score_threshold=0.63,
        use_kc=False,
    )
    captured = {}

    def _fake_quality_filter(_df, **kwargs):
        captured.update(kwargs)
        return np.ones(len(_df), dtype=bool)

    with patch("core.signal_utils.build_breakout_quality_filter_pass_condition", side_effect=_fake_quality_filter):
        _atr, buy_condition, _sell_condition, _buy_limits = generate_signals(frame, params, ticker="2330")

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "signal_runtime_receives_exact_crossover_candidates",
        [False, False, False, True, False],
        np.asarray(captured["candidate_condition"], dtype=bool).tolist(),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "signal_runtime_receives_active_threshold",
        0.63,
        float(captured["score_threshold"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "quality_filter_is_and_gate_for_breakout_buy",
        [False, False, False, True, False],
        np.asarray(buy_condition, dtype=bool).tolist(),
    )


def _validate_breakout_quality_report_rendering(results, case_id):
    def _metrics(split_name, *, base, precision, acceptance, recall, false_reject):
        total = 100.0
        actual_pass = float(base) * total
        actual_reject = total - actual_pass
        predicted_pass = float(acceptance) * total
        tp = float(recall) * actual_pass
        fn = actual_pass - tp
        fp = predicted_pass - tp
        tn = actual_reject - fp
        specificity = tn / actual_reject
        accuracy = (tp + tn) / total
        return {
            "filter_id": "synthetic_quality",
            "split": split_name,
            "selected_date_range": {"start": "2021-01-01", "end": "2022-12-31"},
            "ticker_date_group_weighted": {
                "group_count": 100,
                "row_count": 1000,
                "weight_sum": total,
                "base_pass_rate": base,
                "acceptance_rate": acceptance,
                "pass_precision": precision,
                "precision_lift_vs_all_pass": precision / base,
                "pass_recall": recall,
                "false_rejection_rate": false_reject,
                "reject_specificity": specificity,
                "accuracy": accuracy,
                "all_pass_baseline_accuracy": base,
                "avg_score": 0.45,
                "confusion": {
                    "true_pass_pred_pass": tp,
                    "true_pass_pred_reject": fn,
                    "true_reject_pred_pass": fp,
                    "true_reject_pred_reject": tn,
                },
            },
            "row_level": {"row_count": 1000},
        }

    context = {
        "filter_id": "synthetic_quality",
        "score_path": Path("research_scores.csv"),
        "model_manifest": {
            "training_mode": "inner_validation_epoch_selection_full_refit",
            "inner_validation_used": True,
            "max_epochs": 20,
            "selected_epoch": 2,
            "fixed_evaluation_threshold": 0.5,
            "learning_rate": 0.001,
            "batch_size": 256,
            "seed": 42,
            "epoch_selection_source": "inner_validation_loss",
            "early_stopping_enabled": True,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
            "inner_validation_epoch_selection": {
                "best_epoch": 2,
                "completed_epochs": 3,
                "best_validation_loss": 0.77253,
                "best_validation_metrics": {
                    "accuracy": 0.505773,
                    "pass_rate": 0.344325,
                },
                "history": [
                    {
                        "epoch": 1,
                        "batch_loss": 0.681608,
                        "inner_train_metrics": {
                            "loss": 0.670129,
                            "accuracy": 0.578197,
                            "pass_rate": 0.48634,
                        },
                        "inner_validation_metrics": {
                            "loss": 0.774969,
                            "accuracy": 0.500052,
                            "pass_rate": 0.363297,
                        },
                        "is_best_epoch": True,
                    },
                    {
                        "epoch": 2,
                        "batch_loss": 0.666133,
                        "inner_train_metrics": {
                            "loss": 0.655166,
                            "accuracy": 0.604912,
                            "pass_rate": 0.553797,
                        },
                        "inner_validation_metrics": {
                            "loss": 0.77253,
                            "accuracy": 0.505773,
                            "pass_rate": 0.344325,
                        },
                        "is_best_epoch": True,
                    },
                    {
                        "epoch": 3,
                        "batch_loss": 0.657169,
                        "inner_train_metrics": {
                            "loss": 0.651273,
                            "accuracy": 0.615911,
                            "pass_rate": 0.496689,
                        },
                        "inner_validation_metrics": {
                            "loss": 0.793311,
                            "accuracy": 0.519413,
                            "pass_rate": 0.220611,
                        },
                        "is_best_epoch": False,
                    },
                ],
            },
        },
    }
    payload = build_report_payload(
        metrics_by_split={
            "train": _metrics(
                "train",
                base=0.442189,
                precision=0.569538,
                acceptance=0.368851,
                recall=0.475079,
                false_reject=0.524921,
            ),
            "validation": _metrics(
                "validation",
                base=0.492563,
                precision=0.626719,
                acceptance=0.426307,
                recall=0.542417,
                false_reject=0.457583,
            ),
            "selection": _metrics(
                "selection",
                base=0.456199,
                precision=0.586467,
                acceptance=0.387072,
                recall=0.497601,
                false_reject=0.502399,
            ),
            "oos": _metrics(
                "oos",
                base=0.493898,
                precision=0.460815,
                acceptance=0.169293,
                recall=0.157953,
                false_reject=0.842047,
            ),
        },
        context=context,
    )
    markdown = render_markdown_report(payload)
    console = render_console_summary(payload)
    colored_console = render_console_summary(payload, color=True)
    no_oos_payload = build_report_payload(
        metrics_by_split={
            split_name: metrics
            for split_name, metrics in payload["full_metrics"].items()
            if split_name != "oos"
        },
        context=context,
    )
    no_oos_console = render_console_summary(no_oos_payload)
    add_check(results, "synthetic_breakout_quality", case_id, "report_oos_fail_status", "FAIL", payload["conclusion"]["status"])
    add_check(results, "synthetic_breakout_quality", case_id, "report_uses_group_weighted_headline", "ticker_date_group_weighted", payload["headline_basis"])
    add_check(results, "synthetic_breakout_quality", case_id, "report_schema_v2", 2, payload["schema_version"])
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_header_includes_fixed_training_parameters", True, "- **Threshold**：`0.5`" in markdown and "- **Learning Rate**：`0.001`" in markdown and "- **Batch Size**：`256`" in markdown and "- **Random Seed**：`42`" in markdown and "## 1. 固定訓練參數" not in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_has_epoch_comparison", True, "## 1. Epoch 選擇結果" in markdown and "最終模型" in markdown and "Validation Loss" in markdown)
    selection_matrix = markdown.split("## 2. Selection Confusion Matrix", 1)[1].split("### 分類品質", 1)[0]
    console_selection_matrix = console.split("2. Selection Confusion Matrix", 1)[1].split("分類品質", 1)[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_confusion_matrix_is_count_only_with_standard_margins",
        True,
        (
            all(token in selection_matrix for token in ("TP =", "FN =", "FP =", "TN =", "模型 PASS", "模型 REJECT", "原始 PASS", "原始 REJECT"))
            and all(token in console_selection_matrix for token in ("TP =", "FN =", "FP =", "TN =", "模型 PASS", "模型 REJECT", "原始 PASS", "原始 REJECT"))
            and all(token not in selection_matrix for token in ("Precision", "Recall", "Specificity", "NPV", "Accuracy"))
            and all(token not in console_selection_matrix for token in ("Precision", "Recall", "Specificity", "NPV", "Accuracy"))
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_confusion_margins_include_all_four_rates",
        True,
        (
            all(token in selection_matrix for token in ("原始 PASS 比例", "原始 REJECT 比例", "保留率", "拒絕率"))
            and all(token in console_selection_matrix for token in ("原始 PASS 比例", "原始 REJECT 比例", "保留率", "拒絕率"))
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_metric_tables_include_formulas_and_explanations",
        True,
        (
            "| 指標 | 公式 | 結果 | 解釋 |" in markdown
            and "| 指標 | 公式 | 結果 |" in markdown
            and "TP ÷ (TP + FP)" in markdown
            and "PASS Precision − 原始 PASS 比例" in markdown
            and "被保留的訊號中，有多少真的 PASS" in markdown
            and "篩選行為" not in markdown
            and "篩選行為" not in console
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_marks_oos_not_for_retuning", True, "不得使用同一段 OOS 回頭調整" in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_console_has_epoch_and_confusion_tables", True, "1. Epoch 選擇結果" in console and "2. Selection Confusion Matrix" in console and "3. OOS Confusion Matrix" in console and "4. 各資料區段比較" in console and "5. Selection 與 OOS 差異" in console and "Inner Train" in console and "Validation*" in console and "Precision" in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_header_merges_fixed_training_parameters", True, "Threshold       : 0.5" in console and "Learning Rate   : 0.001" in console and "Batch Size      : 256" in console and "Random Seed     : 42" in console and "固定訓練參數" not in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_epoch_summary_uses_bullets", True, "- Epoch 上限：20" in console and "- 最終模型：Inner Validation 選出 Epoch 2" in console and "| Epoch 上限" not in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_split_and_date_are_separate_columns", True, "區段 / 日期" not in console and "|    區段" in console and "|          日期" in console and "| 區段 | 日期 |" in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_epoch_selection_is_not_repeated_in_bullets", True, "- 最後選擇：" not in console and "是否選出 Best Epoch" not in console and console.count("- 最終模型：") == 1)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_metric_labels_are_consistent_across_tables",
        True,
        (
            all(
                label in markdown and label in console
                for label in (
                    "原始 PASS 比例",
                    "保留率",
                    "拒絕率",
                    "PASS Precision",
                    "PASS Recall",
                    "REJECT Specificity",
                    "REJECT NPV",
                    "Accuracy",
                    "Precision 絕對提升",
                    "Precision 相對提升",
                    "平均 Score",
                )
            )
            and "模型 PASS 比例" not in markdown
            and "模型 PASS 比例" not in console
            and "模型 REJECT 比例" not in markdown
            and "模型 REJECT 比例" not in console
            and "模型保留" not in markdown
            and "模型保留" not in console
            and "REJECT 辨識率" not in markdown
            and "REJECT 辨識率" not in console
            and "保留後 Precision" not in markdown
            and "保留後 Precision" not in console
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_deployment_summary_is_concise", True, "- OOS 判定依據：" in console and "- 部署決策：" in console and "- 研究限制：" in console and "是否選出 Best Epoch" not in console and "| 判定項目" not in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_header_does_not_duplicate_final_decision", True, "最終判定" not in console.split("1. Epoch 選擇結果", 1)[0] and "部署建議" not in console.split("1. Epoch 選擇結果", 1)[0])
    add_check(results, "synthetic_breakout_quality", case_id, "report_without_oos_explains_missing_sections", True, "OOS             : 未納入本次報表" in no_oos_console and "不會輸出 OOS Confusion Matrix 與 Selection/OOS 差異" in no_oos_console and "--include-oos" in no_oos_console and "OOS Confusion Matrix" not in no_oos_console.split("注意：", 1)[0] and "Selection 與 OOS 差異" not in no_oos_console.split("注意：", 1)[0])
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_has_semantic_colors", True, "color:#188038" in markdown and "color:#C62828" in markdown and "color:#42A5F5" in markdown and "color:#B06000" in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_console_color_is_opt_in", True, "\x1b[" not in console and "\x1b[96m" in colored_console)
    report_markdown_path = resolve_filter_report_markdown_path("/project", "synthetic_quality")
    report_json_path = resolve_filter_report_json_path("/project", "synthetic_quality")
    expected_report_dir_suffix = (
        "outputs",
        "filters",
        "breakout_quality",
        "synthetic_quality",
        "reports",
    )
    report_paths_are_under_filter_reports = (
        tuple(report_markdown_path.parts[-6:])
        == (*expected_report_dir_suffix, "evaluation_report.md")
        and tuple(report_json_path.parts[-6:])
        == (*expected_report_dir_suffix, "evaluation_metrics.json")
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_paths_are_under_filter_reports",
        True,
        report_paths_are_under_filter_reports,
    )
    research_only = build_report_payload(
        metrics_by_split={"selection": _metrics("selection", base=0.45, precision=0.55, acceptance=0.40, recall=0.50, false_reject=0.50)},
        context=context,
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_without_oos_is_research_only", "RESEARCH_ONLY", research_only["conclusion"]["status"])

def validate_breakout_quality_runtime_artifact_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_RUNTIME_ARTIFACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    filter_id = "synthetic_quality"
    high_len = int(BREAKOUT_DEFAULT_HIGH_LEN)
    with tempfile.TemporaryDirectory(prefix="breakout_quality_contract_") as tmp_dir:
        project_root = Path(tmp_dir)
        models_dir = project_root / "models"
        with patch.dict(os.environ, {"V16_MODELS_DIR": str(models_dir), "V16_BREAKOUT_QUALITY_SCORE_PATH": str(project_root / "legacy.csv")}, clear=False):
            paths = resolve_filter_artifact_paths(project_root, filter_id)
            research_score_path = resolve_filter_research_score_path(project_root, filter_id)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "research_output_cannot_replace_canonical_runtime_score",
                True,
                research_score_path != paths.score_path and "outputs" in research_score_path.parts,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "path_resolution_has_no_output_side_effect",
                False,
                (project_root / "outputs").exists(),
            )
            paths.model_dir.mkdir(parents=True, exist_ok=True)
            paths.model_path.write_bytes(b"synthetic-model")
            split_frame = pd.DataFrame(
                [
                    {
                        "ticker": "2330",
                        "date": "2024-01-02",
                        "high_len": high_len,
                        "outer_split": OUTER_SPLIT_SELECTION,
                        "selection_role": SELECTION_ROLE_TRAIN,
                    },
                    {
                        "ticker": "2330",
                        "date": "2024-12-02",
                        "high_len": high_len,
                        "outer_split": OUTER_SPLIT_SELECTION,
                        "selection_role": SELECTION_ROLE_TRAIN,
                    },
                ]
            )
            split_frame.to_csv(paths.split_path, index=False, encoding="utf-8-sig")
            score_frame = pd.DataFrame(
                [
                    {"ticker": "2330", "date": "2025-01-03", "high_len": high_len, SCORE_COLUMN: 0.60},
                    {"ticker": "2330", "date": "2025-01-04", "high_len": high_len, SCORE_COLUMN: 0.40},
                ]
            )
            score_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            legacy_path = project_root / "legacy.csv"
            pd.DataFrame(
                [{"ticker": "2330", "date": "2025-01-03", "high_len": high_len, SCORE_COLUMN: 0.0}]
            ).to_csv(legacy_path, index=False)

            score_record = build_file_manifest(paths.score_path)
            score_record.update(
                {
                    "schema_version": SCORE_TABLE_SCHEMA_VERSION,
                    "required_columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "row_count": len(score_frame),
                    "high_len_values": [high_len],
                    "event_date_range": {"start": "2025-01-03", "end": "2025-01-04"},
                }
            )
            split_record = build_file_manifest(paths.split_path)
            split_record.update(
                {
                    "schema_version": SPLIT_ASSIGNMENT_SCHEMA_VERSION,
                    "required_columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "row_count": len(split_frame),
                    "group_key": "ticker/date/high_len",
                    "outer_split_counts": {
                        OUTER_SPLIT_SELECTION: 2,
                        OUTER_SPLIT_OOS: 0,
                        OUTER_SPLIT_OUT_OF_SCOPE: 0,
                    },
                    "selection_role_counts": {
                        SELECTION_ROLE_TRAIN: 2,
                        SELECTION_ROLE_VALIDATION: 0,
                        SELECTION_ROLE_INNER_EMBARGO: 0,
                        SELECTION_ROLE_EMBARGO: 0,
                        SELECTION_ROLE_IGNORE: 0,
                        SELECTION_ROLE_NOT_APPLICABLE: 0,
                    },
                }
            )
            outer_policy = {
                "policy_source": "core.walk_forward_policy.synthetic_override",
                "selection_start_date": "2024-01-01",
                "selection_end_date": "2024-12-31",
                "oos_start_date": "2025-01-01",
                "configured_oos_end_date": "2025-12-31",
                "effective_oos_end_date": "2025-12-31",
            }
            outer_policy["policy_fingerprint_sha256"] = compute_outer_policy_fingerprint(outer_policy)
            manifest = {
                "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
                "filter_family": FILTER_FAMILY,
                "filter_id": filter_id,
                "model": build_file_manifest(paths.model_path),
                "split_assignments": split_record,
                "outer_oos_policy": outer_policy,
                "feature_columns": list(FEATURE_COLUMNS),
                "context_columns": list(CONTEXT_COLUMNS),
                "score_decision": {
                    "score_column": SCORE_COLUMN,
                    "comparison": SCORE_COMPARISON,
                    "threshold_source": SCORE_THRESHOLD_SOURCE,
                },
                "fixed_evaluation_threshold": 0.50,
                "threshold_policy": {
                    "mode": "fixed_before_oos",
                    "evaluation_threshold": 0.50,
                    "runtime_source": SCORE_THRESHOLD_SOURCE,
                    "optimized_by_train": False,
                    "oos_tuning_allowed": False,
                },
                "training_mode": "fixed_epoch_full_selection",
                "fixed_epochs": 2,
                "completed_epochs": 2,
                "early_stopping_enabled": False,
                "inner_validation_used": False,
                "training_uses_all_eligible_selection_rows": True,
                "oos_predictions_used_during_training": False,
                "oos_metrics_emitted_by_train": False,
                "score_table": score_record,
                "runtime_eligibility": {
                    "eligible": True,
                    "scope": RUNTIME_SCOPE_FORWARD_OOS,
                    "available_from": "2025-01-03",
                    "available_through": "2025-01-05",
                    "model_information_cutoff": "2025-01-02",
                },
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()

            frame = pd.DataFrame(index=pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-04", "2025-01-05"]))
            candidates = np.asarray([True, True, True, False], dtype=bool)
            pass_at_050 = build_pass_condition_from_score_table(
                frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.50,
                candidate_condition=candidates,
                project_root=str(project_root),
                filter_id=filter_id,
            )
            pass_at_070 = build_pass_condition_from_score_table(
                frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.70,
                candidate_condition=candidates,
                project_root=str(project_root),
                filter_id=filter_id,
            )
            add_check(results, "synthetic_breakout_quality", case_id, "canonical_score_path_only", [True, True, False, True], pass_at_050.tolist())
            add_check(results, "synthetic_breakout_quality", case_id, "active_threshold_controls_decision", [True, False, False, True], pass_at_070.tolist())

            split_frame.loc[1, "selection_role"] = SELECTION_ROLE_VALIDATION
            split_frame.to_csv(paths.split_path, index=False, encoding="utf-8-sig")
            validation_split_record = build_file_manifest(paths.split_path)
            validation_split_record.update(
                {
                    "schema_version": SPLIT_ASSIGNMENT_SCHEMA_VERSION,
                    "required_columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "row_count": len(split_frame),
                    "group_key": "ticker/date/high_len",
                    "outer_split_counts": {
                        OUTER_SPLIT_SELECTION: 2,
                        OUTER_SPLIT_OOS: 0,
                        OUTER_SPLIT_OUT_OF_SCOPE: 0,
                    },
                    "selection_role_counts": {
                        SELECTION_ROLE_TRAIN: 1,
                        SELECTION_ROLE_VALIDATION: 1,
                        SELECTION_ROLE_INNER_EMBARGO: 0,
                        SELECTION_ROLE_EMBARGO: 0,
                        SELECTION_ROLE_IGNORE: 0,
                        SELECTION_ROLE_NOT_APPLICABLE: 0,
                    },
                }
            )
            manifest.update(
                {
                    "split_assignments": validation_split_record,
                    "training_mode": TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
                    "max_epochs": 5,
                    "selected_epoch": 2,
                    "fixed_epochs": 2,
                    "completed_epochs": 2,
                    "epoch_selection_source": "inner_validation_loss",
                    "early_stopping_enabled": True,
                    "early_stopping_patience": 1,
                    "early_stopping_min_delta": 0.0,
                    "inner_validation_used": True,
                    "inner_validation_months": 2,
                    "inner_validation_epoch_selection": {
                        "best_epoch": 2,
                        "completed_epochs": 3,
                    },
                }
            )
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            validation_contract = load_model_artifact_contract(
                str(project_root),
                filter_id,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "inner_validation_model_contract_supported",
                TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
                validation_contract.manifest["training_mode"],
            )

            missing_candidate = np.asarray([False, False, False, True], dtype=bool)
            try:
                build_pass_condition_from_score_table(
                    frame,
                    ticker="2330",
                    high_len=high_len,
                    score_threshold=0.50,
                    candidate_condition=missing_candidate,
                    project_root=str(project_root),
                    filter_id=filter_id,
                )
                missing_rejected = False
            except ValueError as exc:
                missing_rejected = "缺少正式候選事件" in str(exc)
            add_check(results, "synthetic_breakout_quality", case_id, "missing_candidate_fails_fast", True, missing_rejected)

            stale_frame = pd.DataFrame(index=pd.to_datetime(["2025-01-06"]))
            no_candidate_after_coverage = build_pass_condition_from_score_table(
                stale_frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.50,
                candidate_condition=np.asarray([False], dtype=bool),
                project_root=str(project_root),
                filter_id=filter_id,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "no_candidate_after_coverage_does_not_require_score",
                [True],
                no_candidate_after_coverage.tolist(),
            )
            try:
                build_pass_condition_from_score_table(
                    stale_frame,
                    ticker="2330",
                    high_len=high_len,
                    score_threshold=0.50,
                    candidate_condition=np.asarray([True], dtype=bool),
                    project_root=str(project_root),
                    filter_id=filter_id,
                )
                stale_rejected = False
            except ValueError as exc:
                stale_rejected = "已過期" in str(exc)
            add_check(results, "synthetic_breakout_quality", case_id, "uncovered_candidate_fails_fast", True, stale_rejected)

            manifest["runtime_eligibility"] = {
                "eligible": False,
                "scope": "research",
                "reason": "synthetic research artifact",
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()
            try:
                load_runtime_artifact_contract(str(project_root), filter_id)
                research_rejected = False
            except ValueError as exc:
                research_rejected = "不可用於正式 runtime" in str(exc)
            add_check(results, "synthetic_breakout_quality", case_id, "research_artifact_rejected", True, research_rejected)

    _clear_breakout_quality_caches()
    _validate_signal_runtime_wiring(results, case_id)
    _validate_breakout_quality_report_rendering(results, case_id)
    summary["filter_id"] = filter_id
    summary["high_len"] = high_len
    return results, summary


__all__ = [
    "validate_breakout_quality_chronological_embargo_case",
    "validate_breakout_quality_policy_single_source_case",
    "validate_breakout_quality_runtime_artifact_contract_case",
]

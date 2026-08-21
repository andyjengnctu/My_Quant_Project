from __future__ import annotations

from .checks import bind_checks

from .synthetic_breakout_quality_support import (
    ARTIFACT_CONTRACT_VERSION,
    BASELINE_EXPERIMENT_PROFILE,
    BREAKOUT_DEFAULT_HIGH_LEN,
    BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_FINAL_REFIT_MODE,
    BREAKOUT_QUALITY_PRETRAINING_PROFILE,
    BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
    CONFIGURED_EXPERIMENT,
    CONTEXT_COLUMNS,
    DEFAULT_LABEL_POLICY,
    DEFAULT_MODEL_ARCHITECTURE,
    DEFAULT_UNAVAILABLE_SCORE_FILENAME,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
    MANTIS_V2_CHECKPOINT_FILENAME,
    MANTIS_V2_CHECKPOINT_SHA256,
    MANTIS_V2_CONFIG_FILENAME,
    MANTIS_V2_CONFIG_SHA256,
    MANTIS_V2_REPOSITORY,
    MANTIS_V2_REVISION,
    MOMENT_CHECKPOINT_FILENAME,
    MOMENT_CHECKPOINT_SHA256,
    MOMENT_CHECKPOINT_SIZE_BYTES,
    MOMENT_CONFIG_FILENAME,
    MOMENT_CONFIG_SIZE_BYTES,
    MOMENT_PACKAGE_NAME,
    MOMENT_PACKAGE_VERSION,
    MOMENT_REPOSITORY,
    MOMENT_REVISION,
    MOMENT_TRANSFORMERS_PACKAGE_NAME,
    MOMENT_TRANSFORMERS_VERSION,
    OUTER_SPLIT_OOS,
    OUTER_SPLIT_OUT_OF_SCOPE,
    OUTER_SPLIT_SELECTION,
    Path,
    RUNTIME_SCOPE_FORWARD_OOS,
    SCORE_COLUMN,
    SCORE_COMPARISON,
    SCORE_TABLE_REQUIRED_COLUMNS,
    SCORE_TABLE_SCHEMA_VERSION,
    SCORE_THRESHOLD_SOURCE,
    SELECTION_ROLE_EMBARGO,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_INVALID,
    SELECTION_ROLE_NOT_APPLICABLE,
    SELECTION_ROLE_TRAIN,
    SELECTION_ROLE_VALIDATION,
    SPLIT_ASSIGNMENT_REQUIRED_COLUMNS,
    SPLIT_ASSIGNMENT_SCHEMA_VERSION,
    SimpleNamespace,
    TIME_WEIGHT_MODE_DATE_BALANCED,
    TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
    TRAINING_SAMPLING_ALL_EVENT_ROWS,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    add_check,
    assign_market_regimes,
    breakout_quality_export_scores,
    breakout_quality_train,
    build_breakout_quality_model,
    build_breakout_quality_pretraining_profile_payload,
    build_file_manifest,
    build_pass_condition_from_score_table,
    build_regime_audit_payload,
    build_report_payload,
    compute_outer_policy_fingerprint,
    count_trainable_parameters,
    derive_benchmark_regime_features,
    generate_signals,
    get_breakout_quality_experiment_profile,
    get_model_spec,
    json,
    load_model_artifact_contract,
    load_runtime_artifact_contract,
    load_score_table,
    load_shared_group_score_table,
    load_split_assignment_frame,
    lookup_breakout_quality_candidate_score,
    np,
    os,
    patch,
    pd,
    render_console_summary,
    render_markdown_report,
    render_regime_audit_markdown,
    resolve_existing_filter_artifact_paths,
    resolve_filter_artifact_paths,
    resolve_filter_report_json_path,
    resolve_filter_report_markdown_path,
    resolve_filter_research_score_path,
    tempfile,
)

def _clear_breakout_quality_caches() -> None:
    load_model_artifact_contract.cache_clear()
    load_runtime_artifact_contract.cache_clear()
    load_split_assignment_frame.cache_clear()
    load_score_table.cache_clear()
    load_shared_group_score_table.cache_clear()

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
        group_metrics = {
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
            "ranking_and_calibration": {
                "average_precision_pr_auc": 0.61,
                "precision_at_coverage": {
                    "0.50": 0.62,
                    "0.60": 0.60,
                    "0.70": 0.58,
                },
                "recall_at_precision_60": 0.55,
                "brier_score": 0.24,
                "expected_calibration_error_10_bins": 0.03,
            },
            "confusion": {
                "true_pass_pred_pass": tp,
                "true_pass_pred_reject": fn,
                "true_reject_pred_pass": fp,
                "true_reject_pred_reject": tn,
            },
        }
        result = {
            "filter_id": "synthetic_quality",
            "split": split_name,
            "selected_date_range": {"start": "2021-01-01", "end": "2022-12-31"},
            "ticker_date_group_weighted": group_metrics,
            "row_level": {"row_count": 1000},
        }
        if split_name == "oos":
            result["yearly_ticker_date_group_weighted"] = [
                {
                    "year": 2021,
                    "date_range": {"start": "2021-01-04", "end": "2021-12-30"},
                    "policy_window": {"start": "2021-01-01", "end": "2021-12-31"},
                    "is_partial_calendar_year": False,
                    "metrics": dict(group_metrics),
                },
                {
                    "year": 2022,
                    "date_range": {"start": "2022-01-03", "end": "2022-06-30"},
                    "policy_window": {"start": "2022-01-01", "end": "2022-06-30"},
                    "is_partial_calendar_year": True,
                    "metrics": {**group_metrics, "group_count": 45, "row_count": 450},
                },
            ]
        return result

    synthetic_report_architecture = "inception_time_v1"
    synthetic_report_model = build_breakout_quality_model(
        10,
        4,
        architecture=synthetic_report_architecture,
    )
    synthetic_report_trainable_parameter_count = count_trainable_parameters(
        synthetic_report_model
    )
    synthetic_report_total_parameter_count = sum(
        int(parameter.numel()) for parameter in synthetic_report_model.parameters()
    )
    context = {
        "filter_id": "synthetic_quality",
        "score_path": Path("research_scores.csv"),
        "model_manifest": {
            "model_architecture": synthetic_report_architecture,
            "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            "experiment_settings": CONFIGURED_EXPERIMENT.as_manifest_payload(),
            "model_spec": get_model_spec(synthetic_report_architecture).as_manifest_payload(),
            "trainable_parameter_count": synthetic_report_trainable_parameter_count,
            "total_parameter_count": synthetic_report_total_parameter_count,
            "frozen_parameter_count": (
                synthetic_report_total_parameter_count
                - synthetic_report_trainable_parameter_count
            ),
            "self_supervised_pretraining": None,
            "sequence_length": int(DEFAULT_LABEL_POLICY.feature_window_bars),
            "training_mode": "inner_validation_epoch_selection_full_refit",
            "inner_validation_used": True,
            "max_epochs": 20,
            "selected_epoch": 2,
            "fixed_evaluation_threshold": 0.5,
            "optimizer_name": CONFIGURED_EXPERIMENT.optimizer_name,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "final_refit_plan": {
                "mode": "matched_optimizer_steps",
                "actual_optimizer_steps": 200,
                "equivalent_epochs": 1.5,
            },
            "class_weight_mode": "none",
            "time_weight_mode": BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
            "training_weight_reduction": CONFIGURED_EXPERIMENT.training_weight_reduction,
            "torch_execution": {
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "mixed_precision_requested": False,
                "mixed_precision_enabled": False,
                "autocast_dtype": "float32",
                "deterministic_algorithms": True,
                "allow_tf32": False,
            },
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

    legacy_ts2vec_architecture = "ts2vec_frozen_linear_v1"
    legacy_ts2vec_model = build_breakout_quality_model(
        10,
        4,
        architecture=legacy_ts2vec_architecture,
    )
    legacy_ts2vec_trainable_parameter_count = count_trainable_parameters(
        legacy_ts2vec_model
    )
    legacy_ts2vec_total_parameter_count = sum(
        int(parameter.numel()) for parameter in legacy_ts2vec_model.parameters()
    )
    legacy_ts2vec_manifest = {
        **context["model_manifest"],
        "model_architecture": legacy_ts2vec_architecture,
        "model_spec": get_model_spec(legacy_ts2vec_architecture).as_manifest_payload(),
        "trainable_parameter_count": legacy_ts2vec_trainable_parameter_count,
        "total_parameter_count": legacy_ts2vec_total_parameter_count,
        "frozen_parameter_count": (
            legacy_ts2vec_total_parameter_count
            - legacy_ts2vec_trainable_parameter_count
        ),
        "self_supervised_pretraining": {
            "manifest": {
                "pretraining_profile": build_breakout_quality_pretraining_profile_payload(
                    BREAKOUT_QUALITY_PRETRAINING_PROFILE
                )
            }
        },
    }
    legacy_ts2vec_payload = build_report_payload(
        metrics_by_split=payload["full_metrics"],
        context={
            **context,
            "model_manifest": legacy_ts2vec_manifest,
        },
    )
    legacy_ts2vec_markdown = render_markdown_report(legacy_ts2vec_payload)
    legacy_ts2vec_console = render_console_summary(legacy_ts2vec_payload)
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
    add_check(results, "synthetic_breakout_quality", case_id, "report_schema_v4", 4, payload["schema_version"])
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_header_includes_fixed_training_parameters", True, f"- **Experiment Profile**：`{BREAKOUT_QUALITY_EXPERIMENT_PROFILE}`" in markdown and "- **Pretraining Profile**" not in markdown and "- **Threshold**：`0.5`" in markdown and f"- **Optimizer**：`{CONFIGURED_EXPERIMENT.optimizer_name}`" in markdown and f"- **LR Schedule**：`{CONFIGURED_EXPERIMENT.lr_schedule_name}`" in markdown and f"- **LR Schedule Parameters**：`{CONFIGURED_EXPERIMENT.lr_schedule_parameters() or '-'}`" in markdown and f"- **Augmentation**：`{CONFIGURED_EXPERIMENT.augmentation_name}`" in markdown and f"- **Augmentation Parameters**：`{CONFIGURED_EXPERIMENT.augmentation_parameters() or '-'}`" in markdown and "- **Learning Rate**：`0.001`" in markdown and "- **Weight Decay**：`0.0001`" in markdown and "- **Gradient Clip Norm**：`1.0`" in markdown and "- **Batch Size**：`256`" in markdown and "- **Random Seed**：`42`" in markdown and "## 1. 固定訓練參數" not in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_legacy_ts2vec_markdown_header_includes_pretraining_parameters", True, f"- **Pretraining Profile**：`{BREAKOUT_QUALITY_PRETRAINING_PROFILE}`" in legacy_ts2vec_markdown and "- **Encoder Training**：`Selection-only self-supervised; frozen downstream`" in legacy_ts2vec_markdown and "- **Downstream Head**：`linear`" in legacy_ts2vec_markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_has_epoch_comparison", True, "## 1. Epoch 選擇結果" in markdown and "最終模型" in markdown and "Validation Loss" in markdown)
    selection_matrix = markdown.split("## 2. Selection Confusion Matrix", 1)[1].split("### 分類品質", 1)[0]
    normalized_selection_matrix = selection_matrix.replace("**", "")
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
        "report_confusion_margins_use_requested_labels_and_order",
        True,
        (
            all(
                token in normalized_selection_matrix
                for token in (
                    "原始PASS =",
                    "TP + FN =",
                    "原始REJECT =",
                    "FP + TN =",
                    "TP + FP =",
                    "模型PASS =",
                    "FN + TN =",
                    "模型REJECT =",
                )
            )
            and all(
                token in console_selection_matrix
                for token in (
                    "原始PASS =",
                    "TP + FN =",
                    "原始REJECT =",
                    "FP + TN =",
                    "TP + FP =",
                    "模型PASS =",
                    "FN + TN =",
                    "模型REJECT =",
                )
            )
            and normalized_selection_matrix.index("原始PASS =") < normalized_selection_matrix.index("TP + FN =")
            and normalized_selection_matrix.index("原始REJECT =") < normalized_selection_matrix.index("FP + TN =")
            and normalized_selection_matrix.index("TP + FP =") < normalized_selection_matrix.index("模型PASS =")
            and normalized_selection_matrix.index("FN + TN =") < normalized_selection_matrix.index("模型REJECT =")
            and console_selection_matrix.index("原始PASS =") < console_selection_matrix.index("TP + FN =")
            and console_selection_matrix.index("原始REJECT =") < console_selection_matrix.index("FP + TN =")
            and console_selection_matrix.index("TP + FP =") < console_selection_matrix.index("模型PASS =")
            and console_selection_matrix.index("FN + TN =") < console_selection_matrix.index("模型REJECT =")
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
            and "PASS Precision − 原始 PASS" in markdown
            and "被保留的訊號中，有多少真的 PASS" in markdown
            and "篩選行為" not in markdown
            and "篩選行為" not in console
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_marks_oos_not_for_retuning", True, "不得使用同一段 OOS 回頭調整" in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_console_has_epoch_and_confusion_tables", True, "1. Epoch 選擇結果" in console and "2. Selection Confusion Matrix" in console and "3. OOS Confusion Matrix" in console and "4. 各資料區段比較" in console and "5. 排序與校準診斷" in console and "6. OOS 年度診斷" in console and "7. OOS 綜合判定" in console and "Inner Train" in console and "Validation*" in console and "Precision" in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_confusion_omits_redundant_orientation_text", True, "統計口徑：Ticker/Date Group Weighted" not in console and "列 = 原始結果；欄 = 模型判定" not in console and "ticker/date group weighted`；列為原始結果" not in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_header_merges_fixed_training_parameters", True, f"Experiment      : {BREAKOUT_QUALITY_EXPERIMENT_PROFILE}" in console and "Pretrain Profile :" not in console and "Threshold       : 0.5" in console and f"Optimizer       : {CONFIGURED_EXPERIMENT.optimizer_name}" in console and f"LR Schedule     : {CONFIGURED_EXPERIMENT.lr_schedule_name}" in console and f"LR Schedule Args: {CONFIGURED_EXPERIMENT.lr_schedule_parameters() or '-' }" in console and f"Augmentation    : {CONFIGURED_EXPERIMENT.augmentation_name}" in console and f"Augmentation Args: {CONFIGURED_EXPERIMENT.augmentation_parameters() or '-'}" in console and "Learning Rate   : 0.001" in console and "Weight Decay    : 0.0001" in console and "Gradient Clip   : 1.0" in console and "Batch Size      : 256" in console and "Random Seed     : 42" in console and "固定訓練參數" not in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_legacy_ts2vec_header_includes_pretraining_parameters", True, f"Pretrain Profile : {BREAKOUT_QUALITY_PRETRAINING_PROFILE}" in legacy_ts2vec_console and "Encoder Training : Selection-only SSL; frozen downstream" in legacy_ts2vec_console and "Downstream Head  : linear" in legacy_ts2vec_console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_epoch_summary_uses_bullets", True, "- Epoch 上限：20" in console and "- 最終模型：Inner Validation 選出 Epoch 2" in console and "| Epoch 上限" not in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_split_and_date_are_separate_columns", True, "區段 / 日期" not in console and "|    區段" in console and "|          日期" in console and "| 區段 | 日期 |" in markdown)
    markdown_section_4 = markdown.split("## 4. 各資料區段比較", 1)[1].split("## 5. 排序與校準診斷", 1)[0]
    markdown_section_5 = markdown.split("## 7. OOS 綜合判定", 1)[1].split("## 指標白話說明", 1)[0]
    console_section_4 = console.split("4. 各資料區段比較", 1)[1].split("5. 排序與校準診斷", 1)[0]
    console_section_5 = console.split("7. OOS 綜合判定", 1)[1]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_sections_4_and_5_use_consistent_metric_names",
        True,
        (
            all("模型 PASS" in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
            and all("保留率" not in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
            and all("原始 PASS" in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
            and all("PASS Precision" in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
        ),
    )
    compact_section_4_headers = (
        "區段",
        "日期",
        "Groups",
        "原始 PASS",
        "模型 PASS",
        "PASS Precision",
        "Precision 絕對",
        "PASS Recall",
        "平均 Score",
    )
    removed_section_4_headers = (
        "REJECT Specificity",
        "REJECT NPV",
        "Accuracy",
        "Precision 相對",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_section_4_is_compact_and_ordered",
        True,
        (
            "| 區段 | 日期 | Groups | 原始 PASS | 模型 PASS | PASS Precision | Precision 絕對 | PASS Recall | 平均 Score |"
            in markdown_section_4
            and all(label in console_section_4 for label in compact_section_4_headers)
            and all(label not in markdown_section_4 for label in removed_section_4_headers)
            and all(label not in console_section_4 for label in removed_section_4_headers)
            and all(
                markdown_section_4.index(left) < markdown_section_4.index(right)
                for left, right in zip(compact_section_4_headers, compact_section_4_headers[1:])
            )
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_oos_assessment_has_three_metric_groups_and_judgements",
        True,
        (
            "| 類別 | 指標 | Selection | OOS | OOS - Selection | 判讀 |" in markdown_section_5
            and all(label in markdown_section_5 for label in ("主要成效", "過度篩選防線", "輔助診斷"))
            and all(label in markdown_section_5 for label in ("PASS Precision", "PASS Recall", "模型 PASS", "REJECT Specificity", "REJECT NPV", "Accuracy", "平均 Score"))
            and "綜合判定" in markdown_section_5
        ),
    )
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
                    "原始 PASS",
                    "模型 PASS",
                    "PASS Precision",
                    "PASS Recall",
                    "REJECT Specificity",
                    "REJECT NPV",
                    "Accuracy",
                    "Precision 絕對",
                    "Precision 相對",
                    "平均 Score",
                )
            )
            and "模型保留" not in markdown
            and "模型保留" not in console
            and "REJECT 辨識率" not in markdown
            and "REJECT 辨識率" not in console
            and "保留後 Precision" not in markdown
            and "保留後 Precision" not in console
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_oos_assessment_merges_comparison_and_deployment",
        True,
        (
            "5. 排序與校準診斷" in console and "6. OOS 年度診斷" in console and "7. OOS 綜合判定" in console
            and "類別" in console
            and "判讀" in console
            and "主要成效" in console
            and "過度篩選防線" in console
            and "輔助診斷" in console
            and "- 最終部署決策：" in console
            and "- 研究限制：" in console
            and "Selection 與 OOS 差異" not in console
            and "最終部署判定" not in console
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_header_does_not_duplicate_final_decision", True, "最終判定" not in console.split("1. Epoch 選擇結果", 1)[0] and "部署建議" not in console.split("1. Epoch 選擇結果", 1)[0])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_includes_yearly_oos_classification_ranking_and_partial_marker",
        True,
        (
            "## 6. OOS 年度診斷" in markdown
            and "| 年度 | 日期 | Groups | 原始 PASS | 模型 PASS | PASS Precision | Precision 絕對 | PASS Recall | 平均 Score |" in markdown
            and "| 年度 | PR-AUC | P@50% | P@60% | P@70% | R@P60% | Brier | ECE |" in markdown
            and "2022*" in markdown
            and "6. OOS 年度診斷" in console
            and "年度表只切分同一份固定 OOS score" in console
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_without_oos_explains_missing_sections", True, "OOS             : 未納入本次報表" in no_oos_console and "不會輸出 OOS Confusion Matrix 與 Selection/OOS 指標比較" in no_oos_console and "--include-oos" in no_oos_console and "OOS Confusion Matrix" not in no_oos_console.split("注意：", 1)[0] and "Selection 與 OOS 差異" not in no_oos_console.split("注意：", 1)[0])
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_has_semantic_colors", True, "color:#188038" in markdown and "color:#C62828" in markdown and "color:#42A5F5" in markdown and "color:#B06000" in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_console_color_is_opt_in", True, "\x1b[" not in console and "\x1b[96m" in colored_console)

    colored_matrix = colored_console.split("2. Selection Confusion Matrix", 1)[1].split("分類品質", 1)[0]
    colored_matrix_lines = [line for line in colored_matrix.splitlines() if "|" in line]

    def _matrix_cell(line_token: str, column_index: int) -> str:
        line = next(line for line in colored_matrix_lines if line_token in line)
        return line.split("|")[column_index]

    green = "\x1b[92m"
    red = "\x1b[91m"
    reset = "\x1b[0m"
    tp_label_cell = _matrix_cell("正確保留 PASS", 2)
    fn_label_cell = _matrix_cell("錯殺 PASS", 3)
    tp_value_cell = _matrix_cell("TP =", 2)
    fn_value_cell = _matrix_cell("FN =", 3)
    fp_label_cell = _matrix_cell("錯誤保留 REJECT", 2)
    tn_label_cell = _matrix_cell("正確拒絕 REJECT", 3)
    fp_value_cell = _matrix_cell("FP =", 2)
    tn_value_cell = _matrix_cell("TN =", 3)
    original_pass_margin = _matrix_cell("原始PASS =", 4)
    original_reject_margin = _matrix_cell("原始REJECT =", 4)
    model_pass_margin = _matrix_cell("模型PASS =", 2)
    model_reject_margin = _matrix_cell("模型REJECT =", 3)

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_console_confusion_colors_do_not_bleed",
        True,
        (
            all(green in cell and reset in cell and red not in cell for cell in (tp_label_cell, tp_value_cell, tn_label_cell, tn_value_cell))
            and all(red in cell and reset in cell and green not in cell for cell in (fn_label_cell, fn_value_cell, fp_label_cell, fp_value_cell))
            and all(green not in cell and red not in cell for cell in (original_pass_margin, original_reject_margin, model_pass_margin, model_reject_margin))
        ),
    )
    report_markdown_path = resolve_filter_report_markdown_path("/project", "synthetic_quality")
    report_json_path = resolve_filter_report_json_path("/project", "synthetic_quality")
    expected_report_dir_suffix = (
        "outputs",
        "filters",
        "breakout_quality",
        "synthetic_quality",
        DEFAULT_MODEL_ARCHITECTURE,
        BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        "reports",
    )
    report_paths_are_under_filter_reports = (
        tuple(report_markdown_path.parts[-8:])
        == (*expected_report_dir_suffix, "evaluation_report.md")
        and tuple(report_json_path.parts[-8:])
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

    regime_bank = np.zeros((12, 300, len(FEATURE_COLUMNS)), dtype=np.float32)
    benchmark_close_index = FEATURE_COLUMNS.index("benchmark_close_norm")
    for group_index in range(12):
        relative_close = (
            np.linspace(0.70, 1.0, 300)
            if group_index < 6
            else np.linspace(1.30, 1.0, 300)
        )
        relative_close = relative_close * (
            1.0
            + 0.002
            * (group_index % 3)
            * np.sin(np.linspace(0.0, 20.0, 300))
        )
        relative_close = relative_close / relative_close[-1]
        regime_bank[group_index, :, benchmark_close_index] = relative_close - 1.0
    regime_features = derive_benchmark_regime_features(
        regime_bank, np.arange(12, dtype=np.int64)
    )
    regime_groups = pd.DataFrame(
        {
            "ticker": [f"R{index}" for index in range(12)],
            "date": list(pd.date_range("2018-01-01", periods=6, freq="YS"))
            + list(pd.date_range("2024-01-02", periods=6, freq="MS")),
            "group_index": np.arange(12, dtype=np.int64),
            "outer_split": [OUTER_SPLIT_SELECTION] * 6 + [OUTER_SPLIT_OOS] * 6,
            "selection_role": [SELECTION_ROLE_TRAIN] * 6
            + [SELECTION_ROLE_NOT_APPLICABLE] * 6,
            "label": [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
            SCORE_COLUMN: [0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.7, 0.6, 0.8, 0.55, 0.75, 0.65],
        }
    ).merge(regime_features, on="group_index", validate="one_to_one")
    regime_groups, regime_thresholds = assign_market_regimes(regime_groups)
    (
        regime_payload,
        regime_cells,
        regime_focus_year_cells,
        enriched_regime_groups,
    ) = build_regime_audit_payload(
        regime_groups,
        threshold=0.5,
        min_selection_groups=2,
        min_support_share_ratio=0.5,
        focus_year=2024,
        regime_thresholds=regime_thresholds,
        score_group_diagnostics={"group_score_reduction": "synthetic"},
        metadata={
            "filter_id": "synthetic_quality",
            "model_architecture": "inception_time_v1",
            "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        },
    )
    regime_markdown = render_regime_audit_markdown(regime_payload)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_audit_uses_selection_only_volatility_quantiles",
        "selection_only",
        regime_payload["regime_feature_contract"]["thresholds"][
            "volatility_quantile_source"
        ],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_audit_preserves_one_row_per_ticker_date_group",
        12,
        len(enriched_regime_groups),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_audit_emits_all_dimensions_and_support_fields",
        True,
        (
            set(regime_cells["dimension"])
            == {
                "trend_state",
                "drawdown_state",
                "volatility_state",
                "combined_regime",
            }
            and enriched_regime_groups["selection_support_groups"].notna().all()
            and not regime_focus_year_cells.empty
            and "Selection低代表性" in regime_markdown
            and "2024 Combined regime 歸因" in regime_markdown
            and "Low-support 排除診斷" in regime_markdown
            and "不得依 OOS 結果回頭調整" in regime_markdown
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_audit_emits_focus_year_attribution_and_support_exclusion",
        True,
        (
            regime_payload["focus_year"] == 2024
            and set(
                regime_payload["focus_year_analysis"][
                    "support_exclusion_diagnostic"
                ]
            )
            == {
                "all_focus_year_events",
                "low_support_only",
                "supported_only",
                "supported_only_delta_vs_all",
            }
            and int(regime_focus_year_cells["group_count"].sum()) == 6
        ),
    )

def validate_breakout_quality_runtime_artifact_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_RUNTIME_ARTIFACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    filter_id = "synthetic_quality"
    runtime_fixture_architecture = "inception_time_v1"
    high_len = int(BREAKOUT_DEFAULT_HIGH_LEN)
    with tempfile.TemporaryDirectory(prefix="breakout_quality_contract_") as tmp_dir:
        project_root = Path(tmp_dir)
        models_dir = project_root / "models"
        with (
            patch.dict(
                os.environ,
                {
                    "V16_MODELS_DIR": str(models_dir),
                    "V16_BREAKOUT_QUALITY_SCORE_PATH": str(project_root / "legacy.csv"),
                },
                clear=False,
            ),
            patch(
                "filters.breakout_quality.paths.DEFAULT_MODEL_ARCHITECTURE",
                runtime_fixture_architecture,
            ),
        ):
            paths = resolve_filter_artifact_paths(
                project_root,
                filter_id,
                runtime_fixture_architecture,
                BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            )
            research_score_path = resolve_filter_research_score_path(project_root, filter_id)
            check_true(
                "research_output_cannot_replace_canonical_runtime_score",
                research_score_path != paths.score_path and "outputs" in research_score_path.parts,
            )
            check("path_resolution_has_no_output_side_effect", False, (project_root / "outputs").exists())
            legacy_model_dir = (
                models_dir
                / "filters"
                / FILTER_FAMILY
                / filter_id
                / runtime_fixture_architecture
            )
            legacy_model_dir.mkdir(parents=True, exist_ok=True)
            (legacy_model_dir / "manifest.json").write_text("{}", encoding="utf-8")
            legacy_loaded_paths = resolve_existing_filter_artifact_paths(
                project_root,
                filter_id,
                runtime_fixture_architecture,
                BASELINE_EXPERIMENT_PROFILE,
            )
            try:
                breakout_quality_export_scores._resolve_forward_export_write_paths(
                    filter_id=filter_id,
                    model_architecture=runtime_fixture_architecture,
                    experiment_profile=BASELINE_EXPERIMENT_PROFILE,
                    loaded_paths=legacy_loaded_paths,
                    project_root=project_root,
                )
                legacy_forward_write_rejected = False
            except ValueError as exc:
                legacy_forward_write_rejected = "僅供唯讀相容" in str(exc)
            canonical_baseline_paths = resolve_filter_artifact_paths(
                project_root,
                filter_id,
                runtime_fixture_architecture,
                BASELINE_EXPERIMENT_PROFILE,
            )
            canonical_forward_paths = breakout_quality_export_scores._resolve_forward_export_write_paths(
                filter_id=filter_id,
                model_architecture=runtime_fixture_architecture,
                experiment_profile=BASELINE_EXPERIMENT_PROFILE,
                loaded_paths=canonical_baseline_paths,
                project_root=project_root,
            )
            check_true(
                "legacy_baseline_fallback_is_read_only_for_forward_export",
                legacy_forward_write_rejected
                                and canonical_forward_paths.model_dir == canonical_baseline_paths.model_dir,
            )
            (legacy_model_dir / "manifest.json").unlink()
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
                    {"ticker": "2330", "date": "2025-01-02", "high_len": high_len, SCORE_COLUMN: 0.55},
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
                    "event_date_range": {"start": "2025-01-02", "end": "2025-01-04"},
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
                        SELECTION_ROLE_INVALID: 0,
                        SELECTION_ROLE_NOT_APPLICABLE: 0,
                    },
                }
            )
            outer_policy = {
                "policy_source": "core.walk_forward_policy.synthetic_override",
                "selection_start_date": "2024-01-01",
                "selection_end_date": "2024-12-31",
                "oos_start_date": "2025-01-03",
                "configured_oos_end_date": "2025-12-31",
                "effective_oos_end_date": "2025-12-31",
            }
            outer_policy["policy_fingerprint_sha256"] = compute_outer_policy_fingerprint(outer_policy)
            synthetic_learning_rate = 0.001
            synthetic_schedule_parameters = CONFIGURED_EXPERIMENT.lr_schedule_parameters()

            def _synthetic_schedule_record(*, total_steps: int, actual_steps: int):
                plan = breakout_quality_train._build_learning_rate_schedule_plan(
                    schedule_name=CONFIGURED_EXPERIMENT.lr_schedule_name,
                    base_learning_rate=synthetic_learning_rate,
                    total_optimizer_steps=total_steps,
                    warmup_fraction=CONFIGURED_EXPERIMENT.lr_warmup_fraction,
                    minimum_lr_ratio=CONFIGURED_EXPERIMENT.lr_minimum_ratio,
                )
                return {
                    **plan,
                    "actual_optimizer_steps": int(actual_steps),
                    "last_applied_learning_rate": breakout_quality_train._learning_rate_for_optimizer_step(
                        plan,
                        int(actual_steps) - 1,
                    ),
                }

            def _synthetic_sampling_phase(source_rows: int, group_count: int):
                mode = CONFIGURED_EXPERIMENT.training_sampling_mode
                if mode == TRAINING_SAMPLING_UNIQUE_TICKER_DATE:
                    sampled_rows = int(group_count)
                    return {
                        "mode": mode,
                        "sampling_unit": "unique_ticker_date_group",
                        "batch_size_unit": "unique_ticker_date_groups",
                        "source_row_count": int(source_rows),
                        "sampled_row_count": sampled_rows,
                        "unique_group_count": int(group_count),
                        "duplicate_rows_removed": int(source_rows) - sampled_rows,
                        "representative_rule": "minimum_original_event_row_index",
                        "uses_all_eligible_rows": sampled_rows == int(source_rows),
                        "uses_all_eligible_groups": True,
                    }
                return {
                    "mode": TRAINING_SAMPLING_ALL_EVENT_ROWS,
                    "sampling_unit": "event_row",
                    "batch_size_unit": "event_rows",
                    "source_row_count": int(source_rows),
                    "sampled_row_count": int(source_rows),
                    "unique_group_count": int(group_count),
                    "duplicate_rows_removed": 0,
                    "representative_rule": None,
                    "uses_all_eligible_rows": True,
                    "uses_all_eligible_groups": True,
                }

            fixed_inner_sampling = _synthetic_sampling_phase(2, 2)
            fixed_final_sampling = _synthetic_sampling_phase(2, 2)

            def _synthetic_weight_summary(group_count: int):
                if BREAKOUT_QUALITY_TIME_WEIGHT_MODE == TIME_WEIGHT_MODE_DATE_BALANCED:
                    return {
                        "mode": BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
                        "group_count": int(group_count),
                        "weight_sum": float(group_count),
                        "year_group_counts": {},
                        "year_weight_multipliers": {},
                        "date_count": int(group_count),
                        "date_group_count_min": 1,
                        "date_group_count_max": 1,
                        "date_group_count_mean": 1.0,
                        "date_weight_multiplier_min": 1.0,
                        "date_weight_multiplier_max": 1.0,
                        "target_total_weight_per_date": 1.0,
                        "actual_total_weight_per_date_min": 1.0,
                        "actual_total_weight_per_date_max": 1.0,
                    }
                return {
                    "mode": BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
                    "group_count": int(group_count),
                    "weight_sum": float(group_count),
                    "year_group_counts": {},
                    "year_weight_multipliers": {},
                }

            synthetic_runtime_model = build_breakout_quality_model(
                10, 4, architecture=paths.model_architecture
            )
            synthetic_trainable_parameter_count = count_trainable_parameters(
                synthetic_runtime_model
            )
            synthetic_total_parameter_count = sum(
                int(parameter.numel())
                for parameter in synthetic_runtime_model.parameters()
            )
            synthetic_frozen_parameter_count = (
                synthetic_total_parameter_count
                - synthetic_trainable_parameter_count
            )
            synthetic_pretraining_fingerprint = "a" * 64
            manifest = {
                "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
                "filter_family": FILTER_FAMILY,
                "filter_id": filter_id,
                "model_architecture": paths.model_architecture,
                "experiment_profile": paths.experiment_profile,
                "experiment_settings": get_breakout_quality_experiment_profile(
                    paths.experiment_profile
                ).as_manifest_payload(),
                "model_spec": get_model_spec(paths.model_architecture).as_manifest_payload(),
                "trainable_parameter_count": int(synthetic_trainable_parameter_count),
                "total_parameter_count": int(synthetic_total_parameter_count),
                "frozen_parameter_count": int(synthetic_frozen_parameter_count),
                "self_supervised_pretraining": None,
                "sequence_length": int(DEFAULT_LABEL_POLICY.feature_window_bars),
                "torch_execution": {
                    "requested_device": "cpu",
                    "resolved_device": "cpu",
                    "mixed_precision_requested": False,
                    "mixed_precision_enabled": False,
                    "autocast_dtype": "float32",
                    "deterministic_algorithms": True,
                    "allow_tf32": False,
                },
                "model": build_file_manifest(paths.model_path),
                "split_assignments": split_record,
                "outer_oos_policy": outer_policy,
                "feature_columns": list(FEATURE_COLUMNS),
                "context_columns": list(CONTEXT_COLUMNS),
                "policy": DEFAULT_LABEL_POLICY.as_manifest_payload(),
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
                "max_epochs": 2,
                "selected_epoch": 2,
                "fixed_epochs": 2,
                "completed_epochs": 2,
                "epoch_selection_source": "fixed_cli_epochs",
                "final_refit_plan": {
                    "mode": "selected_epochs",
                    "training_sampling_mode": CONFIGURED_EXPERIMENT.training_sampling_mode,
                    "training_weight_reduction": CONFIGURED_EXPERIMENT.training_weight_reduction,
                    "batch_size_unit": fixed_final_sampling["batch_size_unit"],
                    "inner_train_sampling_row_count": int(fixed_inner_sampling["sampled_row_count"]),
                    "final_refit_sampling_row_count": int(fixed_final_sampling["sampled_row_count"]),
                    "selected_optimizer_steps": None,
                    "final_refit_batches_per_epoch": 1,
                    "target_optimizer_steps": 2,
                    "actual_optimizer_steps": 2,
                    "completed_epoch_cycles": 2,
                    "all_eligible_selection_rows_seen_at_least_once": bool(fixed_final_sampling["uses_all_eligible_rows"]),
                    "all_eligible_selection_groups_seen_at_least_once": True,
                },
                "training_sampling": {
                    "mode": CONFIGURED_EXPERIMENT.training_sampling_mode,
                    "inner_train": fixed_inner_sampling,
                    "final_refit": fixed_final_sampling,
                    "validation_sampling_enabled": False,
                    "oos_sampling_enabled": False,
                },
                "optimizer_name": CONFIGURED_EXPERIMENT.optimizer_name,
                "learning_rate": synthetic_learning_rate,
                "lr_schedule_name": CONFIGURED_EXPERIMENT.lr_schedule_name,
                "augmentation_name": CONFIGURED_EXPERIMENT.augmentation_name,
                "training_augmentation": {
                    "name": CONFIGURED_EXPERIMENT.augmentation_name,
                    "parameters": CONFIGURED_EXPERIMENT.augmentation_parameters(),
                    "epoch_selection": None,
                    "final_refit": [
                        {
                            "name": CONFIGURED_EXPERIMENT.augmentation_name,
                            "sample_count": 2,
                            "augmented_sample_count": (
                                1
                                if CONFIGURED_EXPERIMENT.augmentation_name != "none"
                                else 0
                            ),
                            "masked_bar_count": (
                                10
                                if CONFIGURED_EXPERIMENT.augmentation_name != "none"
                                else 0
                            ),
                        }
                        for _ in range(2)
                    ],
                    "validation_augmented": False,
                    "oos_augmented": False,
                },
                "learning_rate_schedule": {
                    "name": CONFIGURED_EXPERIMENT.lr_schedule_name,
                    "parameters": synthetic_schedule_parameters,
                    "epoch_selection": None,
                    "final_refit": _synthetic_schedule_record(
                        total_steps=2,
                        actual_steps=2,
                    ),
                },
                "class_weight_mode": BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
                "class_weights_reject_pass": [1.0, 1.0],
                "time_weight_mode": BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
                "training_weight_reduction": CONFIGURED_EXPERIMENT.training_weight_reduction,
                "sample_weight_summaries": {
                    "final_refit": _synthetic_weight_summary(2)
                },
                "early_stopping_enabled": False,
                "inner_validation_used": False,
                "training_uses_all_eligible_selection_rows": bool(fixed_final_sampling["uses_all_eligible_rows"]),
                "training_uses_all_eligible_selection_groups": True,
                "oos_predictions_used_during_training": False,
                "oos_metrics_emitted_by_train": False,
                "score_table": score_record,
                "score_inference_execution": {
                    "inference_unit": "unique_ticker_date_feature_group",
                    "shared_group_score_broadcast": True,
                },
                "runtime_eligibility": {
                    "eligible": True,
                    "scope": RUNTIME_SCOPE_FORWARD_OOS,
                    "available_from": "2025-01-02",
                    "available_through": "2025-01-05",
                    "required_signal_start": "2025-01-02",
                    "execution_start": "2025-01-03",
                    "model_information_cutoff": "2025-01-02",
                },
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()

            runtime_contract = load_runtime_artifact_contract(str(project_root), filter_id)
            check(
                "runtime_score_signal_anchor_precedes_oos_execution_start",
                ("2025-01-02", "2025-01-03"),
                (
                                    runtime_contract.required_signal_start.isoformat(),
                                    runtime_contract.execution_start.isoformat(),
                                ),
            )
            anchor_score = lookup_breakout_quality_candidate_score(
                project_root=str(project_root),
                ticker="2330",
                signal_date="2025-01-02",
                high_len=high_len,
                filter_id=filter_id,
            )
            check(
                "pre_execution_signal_anchor_score_is_runtime_available",
                (True, 0.55),
                (bool(anchor_score["available"]), round(float(anchor_score["score"]), 2)),
            )

            original_runtime_eligibility = dict(manifest["runtime_eligibility"])
            manifest["runtime_eligibility"] = {
                **original_runtime_eligibility,
                "required_signal_start": "2025-01-03",
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()
            try:
                load_runtime_artifact_contract(str(project_root), filter_id)
                bad_signal_start_rejected = False
            except ValueError as exc:
                bad_signal_start_rejected = "required_signal_start" in str(exc)
            check_true("required_signal_start_tamper_fails_fast", bad_signal_start_rejected)

            manifest["runtime_eligibility"] = {
                **original_runtime_eligibility,
                "execution_start": "2025-01-04",
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()
            try:
                load_runtime_artifact_contract(str(project_root), filter_id)
                bad_execution_start_rejected = False
            except ValueError as exc:
                bad_execution_start_rejected = "execution_start" in str(exc)
            check_true("execution_start_tamper_fails_fast", bad_execution_start_rejected)
            manifest["runtime_eligibility"] = original_runtime_eligibility
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
            check("canonical_score_path_only", [True, True, False, True], pass_at_050.tolist())
            check("active_threshold_controls_decision", [True, False, False, True], pass_at_070.tolist())

            alternate_high_len = int(high_len) + 5
            shared_only_frame = pd.DataFrame(
                [
                    {
                        "ticker": "2330",
                        "date": "2025-01-02",
                        "high_len": high_len,
                        SCORE_COLUMN: 0.55,
                    },
                    {
                        "ticker": "2330",
                        "date": "2025-01-03",
                        "high_len": alternate_high_len,
                        SCORE_COLUMN: 0.60,
                    },
                    {
                        "ticker": "2330",
                        "date": "2025-01-04",
                        "high_len": high_len,
                        SCORE_COLUMN: 0.40,
                    },
                ]
            )
            shared_only_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            shared_score_record = build_file_manifest(paths.score_path)
            shared_score_record.update(
                {
                    "schema_version": SCORE_TABLE_SCHEMA_VERSION,
                    "required_columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "row_count": len(shared_only_frame),
                    "high_len_values": [high_len, alternate_high_len],
                    "event_date_range": {"start": "2025-01-02", "end": "2025-01-04"},
                }
            )
            manifest["score_table"] = shared_score_record
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            shared_lookup = build_pass_condition_from_score_table(
                frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.50,
                candidate_condition=candidates,
                project_root=str(project_root),
                filter_id=filter_id,
            )
            check(
                "sequence_only_runtime_uses_ticker_date_shared_score",
                [True, True, False, True],
                shared_lookup.tolist(),
            )

            inconsistent_shared_frame = pd.concat(
                [
                    shared_only_frame,
                    pd.DataFrame(
                        [
                            {
                                "ticker": "2330",
                                "date": "2025-01-03",
                                "high_len": alternate_high_len + 5,
                                SCORE_COLUMN: 0.61,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            inconsistent_shared_frame.to_csv(
                paths.score_path,
                index=False,
                encoding="utf-8-sig",
            )
            inconsistent_record = build_file_manifest(paths.score_path)
            inconsistent_record.update(
                {
                    "schema_version": SCORE_TABLE_SCHEMA_VERSION,
                    "required_columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "row_count": len(inconsistent_shared_frame),
                    "high_len_values": [high_len, alternate_high_len, alternate_high_len + 5],
                    "event_date_range": {"start": "2025-01-02", "end": "2025-01-04"},
                }
            )
            manifest["score_table"] = inconsistent_record
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                build_pass_condition_from_score_table(
                    frame,
                    ticker="2330",
                    high_len=high_len,
                    score_threshold=0.50,
                    candidate_condition=candidates,
                    project_root=str(project_root),
                    filter_id=filter_id,
                )
                inconsistent_shared_rejected = False
            except ValueError as exc:
                inconsistent_shared_rejected = "同一 ticker/date 出現不一致分數" in str(exc)
            check_true("shared_group_score_inconsistency_fails_fast", inconsistent_shared_rejected)

            shared_only_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            manifest["score_table"] = shared_score_record
            manifest.pop("score_inference_execution", None)
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                build_pass_condition_from_score_table(
                    frame,
                    ticker="2330",
                    high_len=high_len,
                    score_threshold=0.50,
                    candidate_condition=candidates,
                    project_root=str(project_root),
                    filter_id=filter_id,
                )
                legacy_exact_key_rejected = False
            except ValueError as exc:
                legacy_exact_key_rejected = "ticker/date/high_len event" in str(exc)
            check_true("legacy_runtime_keeps_exact_ticker_date_high_len_lookup", legacy_exact_key_rejected)
            manifest["score_inference_execution"] = {
                "inference_unit": "unique_ticker_date_feature_group",
                "shared_group_score_broadcast": True,
            }

            score_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            manifest["score_table"] = score_record
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

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
                        SELECTION_ROLE_INVALID: 0,
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
                    "final_refit_plan": {
                        "mode": BREAKOUT_QUALITY_FINAL_REFIT_MODE,
                        "training_sampling_mode": CONFIGURED_EXPERIMENT.training_sampling_mode,
                        "batch_size_unit": fixed_final_sampling["batch_size_unit"],
                        "inner_train_sampling_row_count": 1,
                        "final_refit_sampling_row_count": int(fixed_final_sampling["sampled_row_count"]),
                        "selected_optimizer_steps": 2,
                        "final_refit_batches_per_epoch": 1,
                        "target_optimizer_steps": 2,
                        "actual_optimizer_steps": 2,
                        "completed_epoch_cycles": 2,
                        "all_eligible_selection_rows_seen_at_least_once": bool(fixed_final_sampling["uses_all_eligible_rows"]),
                        "all_eligible_selection_groups_seen_at_least_once": True,
                    },
                    "training_sampling": {
                        "mode": CONFIGURED_EXPERIMENT.training_sampling_mode,
                        "inner_train": _synthetic_sampling_phase(1, 1),
                        "final_refit": fixed_final_sampling,
                        "validation_sampling_enabled": False,
                        "oos_sampling_enabled": False,
                    },
                    "epoch_selection_source": "inner_validation_loss",
                    "early_stopping_enabled": True,
                    "early_stopping_patience": 1,
                    "early_stopping_min_delta": 0.0,
                    "inner_validation_used": True,
                    "inner_validation_months": 2,
                    "inner_validation_epoch_selection": {
                        "best_epoch": 2,
                        "completed_epochs": 3,
                        "best_optimizer_steps": 2,
                        "batches_per_epoch": 1,
                    },
                    "training_augmentation": {
                        **manifest["training_augmentation"],
                        "epoch_selection": [
                            {
                                "name": CONFIGURED_EXPERIMENT.augmentation_name,
                                "sample_count": 2,
                                "augmented_sample_count": (
                                    1
                                    if CONFIGURED_EXPERIMENT.augmentation_name != "none"
                                    else 0
                                ),
                                "masked_bar_count": (
                                    10
                                    if CONFIGURED_EXPERIMENT.augmentation_name != "none"
                                    else 0
                                ),
                            }
                            for _ in range(3)
                        ],
                    },
                    "learning_rate_schedule": {
                        "name": CONFIGURED_EXPERIMENT.lr_schedule_name,
                        "parameters": synthetic_schedule_parameters,
                        "epoch_selection": _synthetic_schedule_record(
                            total_steps=5,
                            actual_steps=3,
                        ),
                        "final_refit": _synthetic_schedule_record(
                            total_steps=2,
                            actual_steps=2,
                        ),
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
            check(
                "inner_validation_model_contract_supported",
                TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
                validation_contract.manifest["training_mode"],
            )

            legacy_ts2vec_architecture = "ts2vec_frozen_linear_v1"
            legacy_ts2vec_paths = resolve_filter_artifact_paths(
                project_root,
                filter_id,
                legacy_ts2vec_architecture,
                paths.experiment_profile,
            )
            legacy_ts2vec_paths.model_dir.mkdir(parents=True, exist_ok=True)
            legacy_ts2vec_paths.model_path.write_bytes(b"synthetic-ts2vec-model")
            split_frame.to_csv(
                legacy_ts2vec_paths.split_path,
                index=False,
                encoding="utf-8-sig",
            )
            legacy_ts2vec_model = build_breakout_quality_model(
                10,
                4,
                architecture=legacy_ts2vec_architecture,
            )
            legacy_ts2vec_trainable_parameter_count = count_trainable_parameters(
                legacy_ts2vec_model
            )
            legacy_ts2vec_total_parameter_count = sum(
                int(parameter.numel())
                for parameter in legacy_ts2vec_model.parameters()
            )
            legacy_ts2vec_split_record = dict(validation_split_record)
            legacy_ts2vec_split_record.update(
                build_file_manifest(legacy_ts2vec_paths.split_path)
            )
            legacy_ts2vec_manifest = json.loads(
                json.dumps(manifest, ensure_ascii=False)
            )
            legacy_ts2vec_manifest.update(
                {
                    "model_architecture": legacy_ts2vec_architecture,
                    "model_spec": get_model_spec(
                        legacy_ts2vec_architecture
                    ).as_manifest_payload(),
                    "trainable_parameter_count": int(
                        legacy_ts2vec_trainable_parameter_count
                    ),
                    "total_parameter_count": int(
                        legacy_ts2vec_total_parameter_count
                    ),
                    "frozen_parameter_count": int(
                        legacy_ts2vec_total_parameter_count
                        - legacy_ts2vec_trainable_parameter_count
                    ),
                    "self_supervised_pretraining": {
                        "manifest": {
                            "schema_version": 1,
                            "model_architecture": legacy_ts2vec_architecture,
                            "experiment_profile": paths.experiment_profile,
                            "model_spec": get_model_spec(
                                legacy_ts2vec_architecture
                            ).as_manifest_payload(),
                            "pretraining_profile": build_breakout_quality_pretraining_profile_payload(
                                BREAKOUT_QUALITY_PRETRAINING_PROFILE
                            ),
                            "pretraining_dataset_fingerprint": synthetic_pretraining_fingerprint,
                            "oos_windows_used": False,
                            "pass_reject_labels_used": False,
                        },
                        "dataset_summary": {
                            "family": "ts2vec_v1",
                            "stride": 5,
                            "window_count": 100,
                            "selection_start_date": "2024-01-01",
                            "selection_end_date": "2024-12-31",
                            "configuration_fingerprint": synthetic_pretraining_fingerprint,
                        },
                    },
                    "model": build_file_manifest(legacy_ts2vec_paths.model_path),
                    "split_assignments": legacy_ts2vec_split_record,
                }
            )
            legacy_ts2vec_paths.manifest_path.write_text(
                json.dumps(legacy_ts2vec_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            legacy_ts2vec_contract = load_model_artifact_contract(
                str(project_root),
                filter_id,
                model_architecture=legacy_ts2vec_architecture,
                experiment_profile=paths.experiment_profile,
            )
            check(
                "legacy_ts2vec_artifact_contract_supported",
                legacy_ts2vec_architecture,
                legacy_ts2vec_contract.manifest["model_architecture"],
            )

            moment_architecture = "moment_1_base_frozen_linear_v1"
            moment_paths = resolve_filter_artifact_paths(
                project_root,
                filter_id,
                moment_architecture,
                paths.experiment_profile,
            )
            moment_paths.model_dir.mkdir(parents=True, exist_ok=True)
            moment_paths.model_path.write_bytes(b"synthetic-moment-model")
            split_frame.to_csv(moment_paths.split_path, index=False, encoding="utf-8-sig")
            moment_split_record = dict(validation_split_record)
            moment_split_record.update(build_file_manifest(moment_paths.split_path))
            moment_manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
            moment_model_spec = get_model_spec(moment_architecture)
            moment_manifest.update(
                {
                    "model_architecture": moment_architecture,
                    "model_spec": moment_model_spec.as_manifest_payload(),
                    "trainable_parameter_count": 15_362,
                    "total_parameter_count": 15_363,
                    "frozen_parameter_count": 1,
                    "self_supervised_pretraining": None,
                    "external_pretrained_encoder": {
                        "source_type": "hugging_face_snapshot",
                        "repository": MOMENT_REPOSITORY,
                        "requested_revision": MOMENT_REVISION,
                        "resolved_revision": MOMENT_REVISION,
                        "checkpoint": {
                            "filename": MOMENT_CHECKPOINT_FILENAME,
                            "sha256": MOMENT_CHECKPOINT_SHA256,
                            "size_bytes": MOMENT_CHECKPOINT_SIZE_BYTES,
                        },
                        "config": {
                            "filename": MOMENT_CONFIG_FILENAME,
                            "sha256": "a" * 64,
                            "size_bytes": MOMENT_CONFIG_SIZE_BYTES,
                            "semantic_validation": "exact_pinned_config",
                        },
                        "package": {
                            "name": MOMENT_PACKAGE_NAME,
                            "version": MOMENT_PACKAGE_VERSION,
                            "expected_version": MOMENT_PACKAGE_VERSION,
                        },
                        "runtime_dependencies": {
                            MOMENT_TRANSFORMERS_PACKAGE_NAME: {
                                "version": MOMENT_TRANSFORMERS_VERSION,
                                "expected_version": MOMENT_TRANSFORMERS_VERSION,
                            }
                        },
                        "model_architecture": moment_architecture,
                        "model_spec": moment_model_spec.as_manifest_payload(),
                        "encoder_frozen_downstream": True,
                        "project_selection_windows_used_for_encoder_training": False,
                        "project_oos_windows_used_for_encoder_training": False,
                        "project_pass_reject_labels_used_for_encoder_training": False,
                        "project_encoder_fine_tuning_used": False,
                        "publisher_pretraining_description": (
                            "Timeseries-PILE masked patch reconstruction pretraining"
                        ),
                    },
                    "model": build_file_manifest(moment_paths.model_path),
                    "split_assignments": moment_split_record,
                }
            )
            moment_paths.manifest_path.write_text(
                json.dumps(moment_manifest, ensure_ascii=False), encoding="utf-8"
            )
            _clear_breakout_quality_caches()
            moment_contract = load_model_artifact_contract(
                str(project_root),
                filter_id,
                model_architecture=moment_architecture,
                experiment_profile=paths.experiment_profile,
            )
            check(
                "moment_external_checkpoint_artifact_contract_supported",
                (moment_architecture, MOMENT_CHECKPOINT_SHA256, True),
                (
                                    moment_contract.manifest["model_architecture"],
                                    moment_contract.manifest["external_pretrained_encoder"]["checkpoint"]["sha256"],
                                    moment_contract.manifest["external_pretrained_encoder"]["encoder_frozen_downstream"],
                                ),
            )
            stale_moment_manifest = json.loads(json.dumps(moment_manifest, ensure_ascii=False))
            stale_moment_manifest["external_pretrained_encoder"]["checkpoint"]["sha256"] = "0" * 64
            moment_paths.manifest_path.write_text(
                json.dumps(stale_moment_manifest, ensure_ascii=False), encoding="utf-8"
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(
                    str(project_root),
                    filter_id,
                    model_architecture=moment_architecture,
                    experiment_profile=paths.experiment_profile,
                )
                stale_moment_checkpoint_rejected = False
            except ValueError as exc:
                stale_moment_checkpoint_rejected = "external checkpoint record" in str(exc)
            check_true("moment_checkpoint_hash_tamper_is_rejected", stale_moment_checkpoint_rejected)
            moment_paths.manifest_path.write_text(
                json.dumps(moment_manifest, ensure_ascii=False), encoding="utf-8"
            )
            _clear_breakout_quality_caches()

            mantis_architecture = "mantis_v2_frozen_linear_v1"
            mantis_paths = resolve_filter_artifact_paths(
                project_root,
                filter_id,
                mantis_architecture,
                paths.experiment_profile,
            )
            mantis_paths.model_dir.mkdir(parents=True, exist_ok=True)
            mantis_paths.model_path.write_bytes(b"synthetic-mantis-model")
            split_frame.to_csv(
                mantis_paths.split_path,
                index=False,
                encoding="utf-8-sig",
            )
            mantis_split_record = dict(validation_split_record)
            mantis_split_record.update(build_file_manifest(mantis_paths.split_path))
            mantis_manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
            mantis_model_spec = get_model_spec(mantis_architecture)
            mantis_manifest.update(
                {
                    "model_architecture": mantis_architecture,
                    "model_spec": mantis_model_spec.as_manifest_payload(),
                    "trainable_parameter_count": 10_242,
                    "total_parameter_count": 10_243,
                    "frozen_parameter_count": 1,
                    "self_supervised_pretraining": None,
                    "external_pretrained_encoder": {
                        "source_type": "hugging_face_snapshot",
                        "repository": MANTIS_V2_REPOSITORY,
                        "requested_revision": MANTIS_V2_REVISION,
                        "resolved_revision": MANTIS_V2_REVISION,
                        "checkpoint": {
                            "filename": MANTIS_V2_CHECKPOINT_FILENAME,
                            "sha256": MANTIS_V2_CHECKPOINT_SHA256,
                            "size_bytes": 16_771_648,
                        },
                        "config": {
                            "filename": MANTIS_V2_CONFIG_FILENAME,
                            "sha256": MANTIS_V2_CONFIG_SHA256,
                            "size_bytes": 375,
                        },
                        "package": {
                            "name": "mantis-tsfm",
                            "version": "1.0.0",
                            "expected_version": "1.0.0",
                        },
                        "model_architecture": mantis_architecture,
                        "model_spec": mantis_model_spec.as_manifest_payload(),
                        "encoder_frozen_downstream": True,
                        "project_selection_windows_used_for_encoder_training": False,
                        "project_oos_windows_used_for_encoder_training": False,
                        "project_pass_reject_labels_used_for_encoder_training": False,
                        "project_encoder_fine_tuning_used": False,
                        "publisher_pretraining_description": (
                            "CauKer-2M synthetic time-series pretraining"
                        ),
                    },
                    "model": build_file_manifest(mantis_paths.model_path),
                    "split_assignments": mantis_split_record,
                }
            )
            mantis_paths.manifest_path.write_text(
                json.dumps(mantis_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            mantis_contract = load_model_artifact_contract(
                str(project_root),
                filter_id,
                model_architecture=mantis_architecture,
                experiment_profile=paths.experiment_profile,
            )
            check(
                "mantis_external_checkpoint_artifact_contract_supported",
                (mantis_architecture, MANTIS_V2_CHECKPOINT_SHA256, True),
                (
                                    mantis_contract.manifest["model_architecture"],
                                    mantis_contract.manifest["external_pretrained_encoder"][
                                        "checkpoint"
                                    ]["sha256"],
                                    mantis_contract.manifest["external_pretrained_encoder"][
                                        "encoder_frozen_downstream"
                                    ],
                                ),
            )

            stale_mantis_manifest = json.loads(
                json.dumps(mantis_manifest, ensure_ascii=False)
            )
            stale_mantis_manifest["external_pretrained_encoder"]["checkpoint"][
                "sha256"
            ] = "0" * 64
            mantis_paths.manifest_path.write_text(
                json.dumps(stale_mantis_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(
                    str(project_root),
                    filter_id,
                    model_architecture=mantis_architecture,
                    experiment_profile=paths.experiment_profile,
                )
                stale_mantis_checkpoint_rejected = False
            except ValueError as exc:
                stale_mantis_checkpoint_rejected = (
                    "external checkpoint record" in str(exc)
                )
            check_true("mantis_checkpoint_hash_tamper_is_rejected", stale_mantis_checkpoint_rejected)
            mantis_paths.manifest_path.write_text(
                json.dumps(mantis_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            stale_pretraining_profile_manifest = json.loads(
                json.dumps(legacy_ts2vec_manifest, ensure_ascii=False)
            )
            stale_pretraining_profile_manifest["self_supervised_pretraining"][
                "manifest"
            ]["pretraining_profile"]["mask_probability"] = 0.25
            legacy_ts2vec_paths.manifest_path.write_text(
                json.dumps(stale_pretraining_profile_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(
                    str(project_root),
                    filter_id,
                    model_architecture=legacy_ts2vec_architecture,
                    experiment_profile=paths.experiment_profile,
                )
                stale_pretraining_profile_rejected = False
            except ValueError as exc:
                stale_pretraining_profile_rejected = "pretraining_profile" in str(exc)
            check_true("pretraining_profile_tamper_is_rejected", stale_pretraining_profile_rejected)
            legacy_ts2vec_paths.manifest_path.write_text(
                json.dumps(legacy_ts2vec_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            stale_sampling_manifest = json.loads(
                json.dumps(manifest, ensure_ascii=False)
            )
            stale_sampling_manifest["training_sampling"]["final_refit"][
                "representative_rule"
            ] = "unstable_first_seen_row"
            paths.manifest_path.write_text(
                json.dumps(stale_sampling_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(str(project_root), filter_id)
                stale_sampling_rejected = False
            except ValueError as exc:
                stale_sampling_rejected = "unique-group 契約不一致" in str(exc)
            check_true("training_sampling_contract_tamper_is_rejected", stale_sampling_rejected)
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            stale_architecture_manifest = dict(manifest)
            stale_architecture = (
                "tiny_cnn_v1"
                if paths.model_architecture != "tiny_cnn_v1"
                else "residual_tcn_v1"
            )
            stale_architecture_manifest["model_architecture"] = stale_architecture
            stale_architecture_manifest["model_spec"] = get_model_spec(
                stale_architecture
            ).as_manifest_payload()
            paths.manifest_path.write_text(
                json.dumps(stale_architecture_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(str(project_root), filter_id)
                stale_architecture_rejected = False
            except ValueError as exc:
                stale_architecture_rejected = (
                    "model_architecture" in str(exc) and "工件路徑" in str(exc)
                )
            check_true("stale_model_architecture_is_rejected", stale_architecture_rejected)
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            stale_policy_manifest = dict(manifest)
            stale_policy_manifest["policy"] = {
                **DEFAULT_LABEL_POLICY.as_manifest_payload(),
                "min_mfe_return": 0.99,
            }
            paths.manifest_path.write_text(
                json.dumps(stale_policy_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(str(project_root), filter_id)
                stale_policy_rejected = False
            except ValueError as exc:
                stale_policy_rejected = "model policy" in str(exc)
            check_true("stale_model_policy_is_rejected", stale_policy_rejected)
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

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
            check_true("missing_candidate_fails_fast", missing_rejected)

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
            check("no_candidate_after_coverage_does_not_require_score", [True], no_candidate_after_coverage.tolist())
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
            check_true("uncovered_candidate_fails_fast", stale_rejected)

            unavailable_path = paths.score_path.with_name(
                DEFAULT_UNAVAILABLE_SCORE_FILENAME
            )
            unavailable_frame = pd.DataFrame(
                [
                    {
                        "ticker": "2330",
                        "date": "2025-01-04",
                        "high_len": high_len,
                        "reason": "benchmark_date_missing",
                    }
                ]
            )
            unavailable_frame.to_csv(
                unavailable_path,
                index=False,
                encoding="utf-8-sig",
            )
            unavailable_record = build_file_manifest(unavailable_path)
            unavailable_record.update(
                {
                    "columns": ["ticker", "date", "high_len", "reason"],
                    "row_count": 1,
                    "reason_counts": {"benchmark_date_missing": 1},
                    "conservative_runtime_score": 0.0,
                    "runtime_action": "reject",
                }
            )
            audit_score_frame = score_frame.copy()
            audit_score_frame.loc[
                audit_score_frame["date"] == "2025-01-04",
                SCORE_COLUMN,
            ] = 0.0
            audit_score_frame.to_csv(
                paths.score_path,
                index=False,
                encoding="utf-8-sig",
            )
            audit_score_record = build_file_manifest(paths.score_path)
            audit_score_record.update(
                {
                    "schema_version": SCORE_TABLE_SCHEMA_VERSION,
                    "required_columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "row_count": len(audit_score_frame),
                    "high_len_values": [high_len],
                    "event_date_range": {
                        "start": "2025-01-02",
                        "end": "2025-01-04",
                    },
                }
            )
            manifest["score_table"] = audit_score_record
            manifest["conservative_unscorable_events"] = unavailable_record
            manifest["score_inference_execution"] = {
                "inference_unit": "unique_ticker_date_feature_group",
                "shared_group_score_broadcast": True,
                "model_scored_event_row_count": 2,
                "conservative_reject_event_row_count": 1,
                "output_event_row_count": 3,
            }
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            audited_scores = load_score_table(str(project_root), filter_id)
            audited_reject_score = float(
                audited_scores.loc[("2330", "2025-01-04", high_len), SCORE_COLUMN]
            )
            check("unscorable_runtime_candidate_is_audited_and_rejected", 0.0, audited_reject_score)

            tampered_audit_score_frame = audit_score_frame.copy()
            tampered_audit_score_frame.loc[
                tampered_audit_score_frame["date"] == "2025-01-04",
                SCORE_COLUMN,
            ] = 0.1
            tampered_audit_score_frame.to_csv(
                paths.score_path,
                index=False,
                encoding="utf-8-sig",
            )
            tampered_audit_score_record = build_file_manifest(paths.score_path)
            tampered_audit_score_record.update(
                {
                    **audit_score_record,
                    **build_file_manifest(paths.score_path),
                }
            )
            manifest["score_table"] = tampered_audit_score_record
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_score_table(str(project_root), filter_id)
                nonzero_unscorable_score_rejected = False
            except ValueError as exc:
                nonzero_unscorable_score_rejected = "保守分數為 0.0" in str(exc)
            check_true("unscorable_runtime_candidate_nonzero_score_is_rejected", nonzero_unscorable_score_rejected)

            score_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            manifest["score_table"] = score_record
            manifest.pop("conservative_unscorable_events", None)
            manifest["score_inference_execution"] = {
                "inference_unit": "unique_ticker_date_feature_group",
                "shared_group_score_broadcast": True,
            }
            unavailable_path.unlink(missing_ok=True)

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
            check_true("research_artifact_rejected", research_rejected)

    _clear_breakout_quality_caches()
    _validate_signal_runtime_wiring(results, case_id)
    _validate_breakout_quality_report_rendering(results, case_id)
    summary["filter_id"] = filter_id
    summary["high_len"] = high_len
    return results, summary

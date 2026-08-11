"""策略績效比較設定。

比較對象、差異、工件來源與前置建立政策全部逐項列出，直接以
``enabled`` 或本檔欄位調整。正式 App 與執行引擎不保存特定實驗矩陣。
"""

from __future__ import annotations

from config.breakout_quality import get_breakout_quality_workflow_settings
from config.training_policy import (
    OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    OUTER_ROLLING_OOS_HORIZON_MONTHS,
    OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
)
from core.strategy_comparison import (
    StrategyArtifactBuilder,
    StrategyComparisonArm,
    StrategyComparisonContrast,
    StrategyComparisonSettings,
    StrategyMultiSeedRobustnessSettings,
    StrategyDLSource,
    StrategyParameterSource,
    StrategyPreparationPolicy,
    validate_strategy_comparison_settings,
    validate_strategy_multi_seed_robustness_settings,
)

STRATEGY_COMPARE_SCHEMA_VERSION = 20

# =============================================================================
# 1. 常用設定
#    一般 Strategy Compare / robustness 實驗通常只需修改本區。
#    Dataset / param policy / max positions / rotation 不在此複製；它們直接
#    繼承 config/breakout_quality.py 的 BreakoutQualityWorkflowSettings SSOT。
# =============================================================================

STRATEGY_COMPARE_DEFAULT_PROFILE = "forward_oos"
STRATEGY_COMPARE_DEFAULT_ROBUSTNESS_PROFILE = "forward_oos"
STRATEGY_COMPARE_STALE_SCORE_MEMBERSHIP_GUARD_MAX_AGE_DAYS = 22

STRATEGY_COMPARE_ROBUSTNESS_SEED_COUNT = 8
STRATEGY_COMPARE_ROBUSTNESS_SEED_GENERATOR_SEED = 20260810
STRATEGY_COMPARE_ROBUSTNESS_GPU_TRAIN_WORKERS = 2
STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS = 1
STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED = True
STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE = "compact"
STRATEGY_COMPARE_ROBUSTNESS_PROGRESS_INTERVAL_SECONDS = 60.0
STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT = True
STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS = False
STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES = False
STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS = False

# Current Strategy Compare核心比較名稱的單一真理。
# Selection PIT／Forward-OOS由profile頁首區分，不把研究階段或固定selector語意塞進arm顯示名稱。
STRATEGY_COMPARE_DISPLAY_FULL_ROOS = "Full ROOS"
STRATEGY_COMPARE_DISPLAY_MIN_ROOS = "Min ROOS"
STRATEGY_COMPARE_DISPLAY_MIN_MR12B = "Min MR-12B"
STRATEGY_COMPARE_DISPLAY_MIN_MR13A = "Min MR-13A"

# Strategy Compare以研究階段profile隔離設定與輸出；App只顯示泛化階段名稱，
# arms／contrasts／period／output namespace全部由本檔驅動。
STRATEGY_COMPARE_PROFILES = {
    "selection_pit": {
        "label": "Selection PIT 策略比較",
        "description": "2014～2020 point-in-time策略轉化Gate；核心比較固定為Full／Min baseline與Min MR-12B／MR-13A。",
        "display_alignment_group": "core_strategy_compare",
        "start_date": "2014-01-01",
        "end_date": "2020-12-31",
        "output_root": "outputs/strategy_compare/selection_pit",
        "reuse_output_roots": ("outputs/strategy_compare",),
        "arm_ids": ("C32", "C23", "C25", "C28"),
        "contrast_ids": (
            "C32-C23",
            "C25-C23", "C28-C23", "C28-C25",
        ),
    },
    "forward_oos": {
        "label": "Forward-OOS 策略比較",
        "description": "2021+ frozen Forward-OOS策略Gate；核心比較固定為Full／Min baseline與Min MR-12B／MR-13A。",
        "display_alignment_group": "core_strategy_compare",
        "start_date": None,
        "end_date": None,
        "output_root": "outputs/strategy_compare/forward_oos",
        "reuse_output_roots": ("outputs/strategy_compare",),
        "arm_ids": ("C1", "C3", "C20", "C29"),
        "contrast_ids": (
            "C1-C3",
            "C20-C3", "C29-C3", "C29-C20",
        ),
    },
}

# Multiple-seed robustness各研究階段以獨立config profile呈現於正式選單。
# stochastic/fixed比較對象不在此重列arm ID，而由對應Strategy Compare profile的
# enabled arms + robustness_role動態解析。
STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES = {
    "selection_pit": {
        "label": "Selection PIT Multi-seed robustness",
        "enabled": True,
        "profile_id": "selection_pit",
        "seed_count": STRATEGY_COMPARE_ROBUSTNESS_SEED_COUNT,
        "seed_generator_seed": STRATEGY_COMPARE_ROBUSTNESS_SEED_GENERATOR_SEED,
        "gpu_train_workers": STRATEGY_COMPARE_ROBUSTNESS_GPU_TRAIN_WORKERS,
        "cpu_replay_workers": STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS,
        "reuse_completed": STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED,
        "console_mode": STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE,
        "progress_interval_seconds": STRATEGY_COMPARE_ROBUSTNESS_PROGRESS_INTERVAL_SECONDS,
        "yearly_report": STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT,
        "keep_checkpoints": STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS,
        "keep_scores": STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES,
        "keep_replay_details": STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS,
        "romd_reference_baselines": {
            "min": {"param_source": "selection_min_roos", "rule_policy": "all_off"},
            "full": {"param_source": "selection_full_roos", "rule_policy": "formal"},
        },
        "output_root": "outputs/strategy_compare/robustness/selection_pit",
        "model_work_root": "models/research/breakout_quality/strategy_compare/multi_seed_robustness/selection_pit",
    },
    "forward_oos": {
        "label": "Forward-OOS Multi-seed robustness",
        "enabled": True,
        "profile_id": "forward_oos",
        "seed_count": STRATEGY_COMPARE_ROBUSTNESS_SEED_COUNT,
        "seed_generator_seed": STRATEGY_COMPARE_ROBUSTNESS_SEED_GENERATOR_SEED,
        "gpu_train_workers": STRATEGY_COMPARE_ROBUSTNESS_GPU_TRAIN_WORKERS,
        "cpu_replay_workers": STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS,
        "reuse_completed": STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED,
        "console_mode": STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE,
        "progress_interval_seconds": STRATEGY_COMPARE_ROBUSTNESS_PROGRESS_INTERVAL_SECONDS,
        "yearly_report": STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT,
        "keep_checkpoints": STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS,
        "keep_scores": STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES,
        "keep_replay_details": STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS,
        "romd_reference_baselines": {
            "min": {"param_source": "min_roos", "rule_policy": "all_off"},
            "full": {"param_source": "full_roos", "rule_policy": "formal"},
        },
        "output_root": "outputs/strategy_compare/robustness",
        "model_work_root": "models/research/breakout_quality/strategy_compare/multi_seed_robustness",
    }
}


# =============================================================================
# 2. 前置工件政策
# =============================================================================

STRATEGY_COMPARE_PREPARATION = {
    "auto_prepare": True,
    "reuse_ready_artifacts": True,
    "rebuild_stale_artifacts": True,
    "resume_parameter_training": True,
    "require_confirmation": True,
    # 已完成且replay identity完全相同的pair直接重用歷史正式結果。
    "reuse_completed_results": True,
    # 同一param_source/rule_policy的新pair只執行一次DL-off baseline。
    "reuse_shared_baseline": True,
}

# =============================================================================
# 3. 策略參數工件來源
# =============================================================================

STRATEGY_PARAM_SOURCES = {
    "full_roos": {
        "path_template": None,
        "description": "正式Full ROOS；依param_policy解析canonical rolling工件",
        "identity_manifest_path": None,
        "trained_with_dl_id": None,
        "builder": None,
    },
    "min_roos": {
        "path_template": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p2_dl_off_trained/active_params/{param_filename}"
        ),
        "description": "forward rolling期間、rule-based filters全關、DL-off訓練的Min ROOS",
        "identity_manifest_path": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p2_dl_off_trained/rolling_preflight.json"
        ),
        "trained_with_dl_id": None,
        "builder": {
            "enabled": True,
            "builder_type": "binary_dl_min_roos_rolling",
            "options": {
                "parameter_set": "p2",
                "model_source_id": "TP1",
                "trials_per_fold": OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
                "resume": True,
                "fixed_risk": 0.01,
                "max_position_cap_pct": 0.30,
                "build_binary_pit": False,
                "binary_pit_resume": True,
                "quiet": False,
            },
        },
    },
    "min_dl_tp1_roos": {
        "path_template": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p3_dl_on_trained/active_params/{param_filename}"
        ),
        "description": "rule-based filters全關、TP1-on環境訓練的Min-TP1 ROOS",
        "identity_manifest_path": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p3_dl_on_trained/rolling_preflight.json"
        ),
        "trained_with_dl_id": "TP1",
        "builder": {
            "enabled": True,
            "builder_type": "binary_dl_min_roos_rolling",
            "options": {
                "parameter_set": "p3",
                "model_source_id": "TP1",
                "p3_variant": None,
                "trials_per_fold": OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
                "resume": True,
                "fixed_risk": 0.01,
                "max_position_cap_pct": 0.30,
                "build_binary_pit": True,
                "binary_pit_resume": True,
                "quiet": False,
            },
        },
    },
    "min_dl_a9_roos": {
        "path_template": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p3_dl_on_trained/A9/active_params/{param_filename}"
        ),
        "description": "rule-based filters全關、A9-on環境訓練的Min-A9 ROOS",
        "identity_manifest_path": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p3_dl_on_trained/A9/rolling_preflight.json"
        ),
        "trained_with_dl_id": "A9",
        "builder": {
            "enabled": True,
            "builder_type": "binary_dl_min_roos_rolling",
            "options": {
                "parameter_set": "p3",
                "model_source_id": "A9",
                "p3_variant": "A9",
                "trials_per_fold": OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
                "resume": True,
                "fixed_risk": 0.01,
                "max_position_cap_pct": 0.30,
                "build_binary_pit": True,
                "binary_pit_resume": True,
                "quiet": False,
            },
        },
    },
    "selection_min_roos": {
        "path_template": (
            "models/research/breakout_quality/trade_path_label/a2_teacher_params/"
            "p2_dl_off_trained/active_params/{param_filename}"
        ),
        "description": (
            "Selection 2014～2020 historical Min ROOS；rules全關、DL-off，"
            "單階段rolling直接搜尋high_len＋4個ATR欄位"
        ),
        "identity_manifest_path": None,
        "trained_with_dl_id": None,
        "artifact_contract": {
            "breakout_quality_param_adaptation": {
                "mode": "min_roos_training",
                "parameter_set": "P2_HISTORY",
                "search_fields": [
                    "high_len",
                    "atr_len",
                    "atr_buy_tol",
                    "atr_times_init",
                    "atr_times_trail",
                ],
                "fixed_rule_contract": "all_rule_filters_off",
                "training_dl_enabled": False,
            }
        },
        "builder": {
            "enabled": True,
            "builder_type": "selection_historical_p2",
            "options": {
                "parameter_set": "p2_history",
                "trials_per_fold": OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
                "train_window_months": OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
                "oos_months": OUTER_ROLLING_OOS_HORIZON_MONTHS,
                "resume": True,
                "fixed_risk": 0.01,
                "max_position_cap_pct": 0.30,
                "optimizer_seed": 42,
                "quiet": False,
            },
        },
    },
    "selection_full_roos": {
        "path_template": (
            "models/research/breakout_quality/strategy_compare/selection_full_roos/"
            "active_params/{param_filename}"
        ),
        "description": (
            "Selection 2014～2020 historical Full ROOS；使用canonical Full optimizer search space，"
            "DL filter/ranking與TP維持optimizer正式固定契約"
        ),
        "identity_manifest_path": (
            "models/research/breakout_quality/strategy_compare/selection_full_roos/"
            "rolling_preflight.json"
        ),
        "trained_with_dl_id": None,
        "artifact_contract": {
            "breakout_quality_param_adaptation": {
                "mode": "selection_full_roos_training",
                "parameter_set": "P4_HISTORY",
                "training_dl_enabled": False,
            }
        },
        "builder": {
            "enabled": True,
            "builder_type": "selection_historical_full_roos",
            "options": {
                "parameter_set": "p4_history",
                "trials_per_fold": OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
                "train_window_months": OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
                "oos_months": OUTER_ROLLING_OOS_HORIZON_MONTHS,
                "resume": True,
                "fixed_risk": 0.01,
                "max_position_cap_pct": 0.30,
                "optimizer_seed": 42,
                "quiet": False,
            },
        },
    },
}

# =============================================================================
# 4. DL工件來源
# =============================================================================

STRATEGY_DL_SOURCES = {
    "TP1": {
        "filter_id": "breakout_quality_a2_trade_path_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "unique_group_sampling",
        "threshold": 0.5,
        "score_source": "canonical_runtime",
        "description": "A2 realized trade-path Binary DL模型",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "forward_oos_scores",
            "options": {
                "scope": "forward_oos",
                "inference_batch_size": 4096,
                "inference_workers": 4,
                "device": "auto",
                "mixed_precision": True,
                "mixed_precision_dtype": "bfloat16",
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "preload_feature_bank": True,
            },
        },
    },
    "A9": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "unique_group_sampling",
        "threshold": 0.5,
        "score_source": "canonical_runtime",
        "description": "既有MFE／MAE 9A Binary DL模型",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "forward_oos_scores",
            "options": {
                "scope": "forward_oos",
                "inference_batch_size": 4096,
                "inference_workers": 4,
                "device": "auto",
                "mixed_precision": True,
                "mixed_precision_dtype": "bfloat16",
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "preload_feature_bank": True,
            },
        },
    },
    "CONT11G": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_pass_magnitude_mse",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": "MR-11G frozen OOS continuous ranker；只允許受控strategy research replay",
        "forward_scores_builder": None,
    },
    "CONT12A": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_all_event_mse",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": "MR-12A all-event no-time frozen OOS continuous ranker；只允許受控strategy research replay",
        "forward_scores_builder": None,
    },
    "CONT12B": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_all_event_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": "MR-12B all-event no-time pairwise ranker frozen OOS score；只允許受控strategy research replay",
        "forward_scores_builder": None,
    },
    "CONT13A": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_no_time_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": (
            "MR-13A Daily Universal frozen Forward-OOS continuous score；"
            "每個盤前決策使用最新已完成交易日資訊，供frozen strategy Gate"
        ),
        "forward_scores_builder": None,
    },
    "CONT12C": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_all_event_listwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": "MR-12C all-event no-time ListNet top-one listwise ranker frozen OOS score；只允許受控strategy research replay",
        "forward_scores_builder": None,
    },
    "CONT12B_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_all_event_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-12B Selection point-in-time continuous score；"
            "只供2014～2020無前視策略經濟驗證"
        ),
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {
                "resume": True,
                "allow_stale_source": False,
            },
        },
    },
    "CONT13A_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_no_time_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-13A Daily Universal Selection point-in-time score；"
            "每個盤前決策只使用最新已完成交易日資訊，供2014～2020無前視策略經濟驗證"
        ),
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {
                "resume": True,
                "allow_stale_source": False,
            },
        },
    },
}

# =============================================================================
# 5. 要比較的對象：逐項用enabled開關
# =============================================================================
# 同一param_source／rule_policy只定義一個共用DL-off基準，並可掛多個DL-on模型。
# 各DL-on arm與contrast可獨立開關；只要仍有DL-on啟用，共用DL-off就必須啟用。

STRATEGY_COMPARE_ARMS = {
    "C1": {
        "enabled": True,
        "name": STRATEGY_COMPARE_DISPLAY_FULL_ROOS,
        "description": "Full optimizer rolling active params；DL-off baseline",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C2": {
        "enabled": False,
        "name": "Full ROOS: TP1-on",
        "description": "Full ROOS參數，runtime開TP1",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "TP1",
        "dl_runtime_mode": "hard-filter",
    },
    "C3": {
        "enabled": True,
        "name": STRATEGY_COMPARE_DISPLAY_MIN_ROOS,
        "description": "只搜尋high_len＋4個ATR；rules全關；DL-off baseline",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C4": {
        "enabled": False,
        "name": "Min ROOS: TP1-on",
        "description": "Min ROOS參數，runtime開TP1",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "TP1",
        "dl_runtime_mode": "hard-filter",
    },
    "C5": {
        "enabled": False,
        "name": "Min-TP1 ROOS",
        "description": "TP1-on環境訓練參數，runtime DL-off",
        "param_source": "min_dl_tp1_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
    },
    "C6": {
        "enabled": False,
        "name": "Min-TP1 ROOS: DL-on",
        "description": "TP1-on環境訓練參數，runtime開其配對TP1",
        "param_source": "min_dl_tp1_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "TP1",
        "dl_runtime_mode": "hard-filter",
    },
    "C7": {
        "enabled": False,
        "name": "Full ROOS: A9-on",
        "description": "Full ROOS參數，runtime開A9",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "hard-filter",
    },
    "C8": {
        "enabled": False,
        "name": "Min ROOS: A9-on",
        "description": "Min ROOS參數，runtime開A9",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "hard-filter",
    },
    "C9": {
        "enabled": False,
        "name": "Min-A9 ROOS",
        "description": "A9-on環境訓練參數，runtime DL-off",
        "param_source": "min_dl_a9_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
    },
    "C10": {
        "enabled": False,
        "name": "Min-A9 ROOS: DL-on",
        "description": "A9-on環境訓練參數，runtime開其配對A9",
        "param_source": "min_dl_a9_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "hard-filter",
    },
    "C11": {
        "enabled": False,
        "name": "Min ROOS: A9 resource-aware",
        "description": "Min ROOS參數；A9只在盤前cash先成瓶頸時以first-improvement改善PASS預留資金",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "resource-aware-binary",
    },
    "C12": {
        "enabled": False,
        "name": "Min ROOS: A9 resource-aware basket",
        "description": "Min ROOS參數；沿用相同cash-binding Gate，每輪評估全部可行PASS promotion並採用最佳改善",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "resource-aware-binary-basket",
    },
    "C14": {
        "enabled": False,
        "name": "Min ROOS: Continuous resource-aware",
        "description": "歷史對照；capital-utilization first + MR-11G PASS-only continuous score",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT11G",
        "dl_runtime_mode": "resource-aware-continuous",
    },
    "C15": {
        "enabled": False,
        "name": "Min ROOS: All-event Continuous resource-aware",
        "description": "Min ROOS先維持資本利用；只有cash-binding的DL Selection Mode才使用MR-12A all-event frozen OOS continuous score排序",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12A",
        "dl_runtime_mode": "resource-aware-continuous",
    },
    "C16": {
        "enabled": False,
        "name": "Min ROOS: All-event Continuous capital-preserving",
        "description": (
            "Min ROOS exact reservation建立每日baseline；MR-12A frozen continuous score可在"
            "cash/slot瓶頸日重排，但selected count與reserved capital均不得低於baseline"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12A",
        "dl_runtime_mode": "resource-aware-continuous-capital-preserving",
    },
    "C17": {
        "enabled": False,
        "name": "Min ROOS: All-event Continuous max-DL constrained basket",
        "description": (
            "Min ROOS只固定每日預留單數K與exact reserved-capital floor；"
            "MR-12A frozen score先取純DL Top-K，不合法時只做deterministic minimum-repair，"
            "最後只允許K筆盤前預留單"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12A",
        "dl_runtime_mode": "resource-aware-continuous-max-dl",
    },
    "C18": {
        "enabled": False,
        "name": "Min ROOS: All-event Continuous max-DL feasible-ascent",
        "description": (
            "與C17使用完全相同K/R0、MR-12A與basket內Min ROOS執行順序；"
            "C17合法seed之後持續做best-feasible single-swap DL改善直到1-swap local optimum，"
            "capital只作hard feasibility，不參與objective"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12A",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C19": {
        "enabled": False,
        "name": "Min ROOS: MR-12B pairwise max-DL constrained basket",
        "description": (
            "與C17使用完全相同K/R0、minimum-repair與basket內Min ROOS執行順序；"
            "唯一模型差異為DL source改成MR-12B pairwise ranker"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12B",
        "dl_runtime_mode": "resource-aware-continuous-max-dl",
    },
    "C20": {
        "enabled": True,
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR12B,
        "description": (
            "與C18使用完全相同K/R0、feasible-ascent與basket內Min ROOS執行順序；"
            "唯一模型差異為DL source改成MR-12B pairwise ranker"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12B",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
        "robustness_role": "stochastic",
    },
    "C21": {
        "enabled": False,
        "name": "Min ROOS: MR-12C listwise max-DL constrained basket",
        "description": (
            "與C19使用完全相同C17 K/R0、minimum-repair與basket內Min ROOS執行順序；"
            "唯一模型差異為DL source改成MR-12C ListNet top-one listwise ranker"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12C",
        "dl_runtime_mode": "resource-aware-continuous-max-dl",
    },
    "C22": {
        "enabled": False,
        "name": "Min ROOS: MR-12C listwise max-DL feasible-ascent",
        "description": (
            "與C20使用完全相同C18 K/R0、feasible-ascent與basket內Min ROOS執行順序；"
            "唯一模型差異為DL source改成MR-12C ListNet top-one listwise ranker"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12C",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C23": {
        "enabled": False,
        "name": STRATEGY_COMPARE_DISPLAY_MIN_ROOS,
        "description": (
            "2014～2020 historical P2 Min ROOS active params；rules全關；DL關閉；"
            "作Selection PIT策略經濟驗證共同baseline"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C24": {
        "enabled": False,
        "name": "Selection PIT: MR-12B minimum-repair",
        "description": (
            "與C23使用完全相同historical P2 Min ROOS params；"
            "使用MR-12B Selection PIT score並完全沿用C17 minimum-repair selector"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12B_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl",
    },
    "C25": {
        "enabled": False,
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR12B,
        "description": (
            "與C23使用完全相同historical P2 Min ROOS params；"
            "使用MR-12B Selection PIT score並完全沿用C18 feasible-ascent selector"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12B_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
        "robustness_role": "stochastic",
    },
    "C26": {
        "enabled": False,
        "name": "Selection PIT: MR-12B feasible-ascent stale-score guard",
        "description": (
            "與C25完全相同MR-12B Selection PIT與feasible-ascent；"
            f"唯一變更為score age超過Selection預先凍結{STRATEGY_COMPARE_STALE_SCORE_MEMBERSHIP_GUARD_MAX_AGE_DAYS}日門檻時，"
            "該舊score不得驅動DL membership swap，候選本身仍保留並沿用Min ROOS資源契約"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12B_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard",
        "dl_runtime_options": {
            "stale_score_membership_guard_max_age_days": STRATEGY_COMPARE_STALE_SCORE_MEMBERSHIP_GUARD_MAX_AGE_DAYS,
        },
    },
    "C27": {
        "enabled": False,
        "name": "Selection PIT: MR-13A daily minimum-repair",
        "description": (
            "與C24使用完全相同historical P2 Min ROOS params、K/R0與minimum-repair selector；"
            "唯一DL差異為score source改成MR-13A Daily Universal Selection PIT，"
            "盤前每日依最新已完成交易日score重排"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13A_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl",
    },
    "C28": {
        "enabled": False,
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13A,
        "description": (
            "與C25使用完全相同historical P2 Min ROOS params、K/R0與feasible-ascent selector；"
            "唯一DL差異為score source改成MR-13A Daily Universal Selection PIT，"
            "盤前每日依最新已完成交易日score重排"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13A_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
        "robustness_role": "stochastic",
    },
    "C29": {
        "enabled": True,
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13A,
        "description": (
            "與C20使用相同current Min ROOS、all-off rules與frozen feasible-ascent；"
            "唯一DL source差異為MR-13A Daily Universal Forward-OOS score"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13A",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
        "robustness_role": "stochastic",
    },
    "C30": {
        "enabled": True,
        "name": "Full ROOS: MR-12B feasible-ascent",
        "description": (
            "與C1使用相同Full ROOS active params與formal rules；"
            "feasible-ascent的K/R0由同參數DL-off baseline逐日建立，DL source為MR-12B"
        ),
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "CONT12B",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C31": {
        "enabled": True,
        "name": "Full ROOS: MR-13A daily feasible-ascent",
        "description": (
            "與C30使用相同Full ROOS active params、formal rules及frozen feasible-ascent；"
            "唯一DL source差異為MR-13A Daily Universal Forward-OOS score"
        ),
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "CONT13A",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C32": {
        "enabled": False,
        "name": STRATEGY_COMPARE_DISPLAY_FULL_ROOS,
        "description": "2014～2020 historical Full ROOS active params；formal rules；DL-off共同baseline",
        "param_source": "selection_full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C33": {
        "enabled": False,
        "name": "Selection Full ROOS: MR-12B feasible-ascent",
        "description": (
            "與C32使用相同historical Full ROOS與formal rules；K/R0由同參數DL-off baseline建立；"
            "使用MR-12B Selection PIT與frozen feasible-ascent"
        ),
        "param_source": "selection_full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "CONT12B_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C34": {
        "enabled": False,
        "name": "Selection Full ROOS: MR-13A daily feasible-ascent",
        "description": (
            "與C33使用相同historical Full ROOS、formal rules與frozen feasible-ascent；"
            "唯一DL source差異為MR-13A Daily Universal Selection PIT"
        ),
        "param_source": "selection_full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "CONT13A_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
}

# =============================================================================
# 6. 報表差異：逐項用enabled開關
# =============================================================================

STRATEGY_COMPARE_CONTRASTS = {
    "C8-C3": {"enabled": False, "left": "C8", "right": "C3", "description": "Min ROOS下A9 hard-filter效果（既有對照重現）"},
    "C11-C3": {"enabled": False, "left": "C11", "right": "C3", "description": "Min ROOS下A9 resource-aware first-improvement效果"},
    "C12-C3": {"enabled": False, "left": "C12", "right": "C3", "description": "Min ROOS下A9 resource-aware best-improvement效果"},
    "C14-C3": {"enabled": False, "left": "C14", "right": "C3", "description": "Capital-utilization first下MR-11G continuous排序效果"},
    "C14-C12": {"enabled": False, "left": "C14", "right": "C12", "description": "MR-11G Continuous resource-aware相對A9 max-PASS resource-aware效果"},
    "C15-C3": {"enabled": False, "left": "C15", "right": "C3", "description": "Capital-utilization first下MR-12A all-event continuous排序效果"},
    "C15-C12": {"enabled": False, "left": "C15", "right": "C12", "description": "All-event continuous resource-aware相對A9 max-PASS resource-aware效果"},
    "C16-C15": {"enabled": False, "left": "C16", "right": "C15", "description": "同一MR-12A source下capital-preserving selector相對C15 cash-binding selector的純runtime效果"},
    "C16-C3": {"enabled": False, "left": "C16", "right": "C3", "description": "Capital-preserving MR-12A selector相對Min ROOS正式研究基準"},
    "C17-C16": {"enabled": False, "left": "C17", "right": "C16", "description": "同一MR-12A source下max-DL constrained basket相對C16 capital-preserving heuristic的純selector效果"},
    "C17-C3": {"enabled": False, "left": "C17", "right": "C3", "description": "Max-DL constrained basket在固定Min ROOS資源底線下相對正式研究基準"},
    "C18-C17": {"enabled": False, "left": "C18", "right": "C17", "description": "相同MR-12A與K/R0資源契約下，feasible-ascent相對C17 minimum-repair的純selector搜尋效果"},
    "C18-C3": {"enabled": False, "left": "C18", "right": "C3", "description": "Max-DL feasible-ascent在固定Min ROOS資源底線下相對正式研究基準"},
    "C21-C19": {"enabled": False, "left": "C21", "right": "C19", "description": "固定C17 selector下MR-12C listwise相對MR-12B pairwise的純DL模型效果"},
    "C22-C20": {"enabled": False, "left": "C22", "right": "C20", "description": "固定C18 selector下MR-12C listwise相對MR-12B pairwise的純DL模型效果"},
    "C22-C21": {"enabled": False, "left": "C22", "right": "C21", "description": "同一MR-12C source下C18 feasible-ascent相對C17 minimum-repair的selector轉化效果"},
    "C21-C3": {"enabled": False, "left": "C21", "right": "C3", "description": "MR-12C在C17 selector下相對Min ROOS研究基準"},
    "C22-C3": {"enabled": False, "left": "C22", "right": "C3", "description": "MR-12C在C18 selector下相對Min ROOS研究基準"},
    "C19-C17": {"enabled": False, "left": "C19", "right": "C17", "description": "固定C17 selector下MR-12B pairwise相對MR-12A MSE的純DL模型效果"},
    "C20-C18": {"enabled": False, "left": "C20", "right": "C18", "description": "固定C18 selector下MR-12B pairwise相對MR-12A MSE的純DL模型效果"},
    "C20-C19": {"enabled": False, "left": "C20", "right": "C19", "description": "同一MR-12B source下C18 feasible-ascent相對C17 minimum-repair的selector轉化效果"},
    "C19-C3": {"enabled": False, "left": "C19", "right": "C3", "description": "MR-12B在C17 selector下相對Min ROOS研究基準"},
    "C20-C3": {"enabled": True, "left": "C20", "right": "C3", "description": "current Min ROOS下MR-12B feasible-ascent相對DL-off baseline的Forward-OOS策略效果"},
    "C24-C23": {"enabled": False, "left": "C24", "right": "C23", "description": "Selection PIT下固定historical Min ROOS與C17 selector，MR-12B PIT ranking相對DL-off baseline的經濟效果"},
    "C25-C23": {"enabled": False, "left": "C25", "right": "C23", "description": "Selection PIT下固定historical Min ROOS與C18 selector，MR-12B PIT ranking相對DL-off baseline的經濟效果"},
    "C26-C25": {"enabled": False, "left": "C26", "right": "C25", "description": "Selection PIT MR-12B feasible-ascent固定其餘條件下，stale-score membership guard的純runtime效果"},
    "C26-C23": {"enabled": False, "left": "C26", "right": "C23", "description": "Selection PIT下固定historical Min ROOS，MR-12B feasible-ascent加stale-score membership guard相對DL-off baseline的經濟效果"},
    "C25-C24": {"enabled": False, "left": "C25", "right": "C24", "description": "Selection PIT MR-12B固定score source下，C18 feasible-ascent相對C17 minimum-repair的selector轉化效果"},
    "C27-C24": {"enabled": False, "left": "C27", "right": "C24", "description": "固定historical Min ROOS與minimum-repair selector，MR-13A daily PIT相對MR-12B event PIT的純DL source效果"},
    "C28-C25": {"enabled": False, "left": "C28", "right": "C25", "description": "固定historical Min ROOS與feasible-ascent selector，MR-13A daily PIT相對MR-12B event PIT的純DL source效果"},
    "C27-C23": {"enabled": False, "left": "C27", "right": "C23", "description": "Selection PIT下MR-13A daily minimum-repair相對DL-off historical Min ROOS baseline的策略經濟效果"},
    "C28-C23": {"enabled": False, "left": "C28", "right": "C23", "description": "Selection PIT下MR-13A daily feasible-ascent相對DL-off historical Min ROOS baseline的策略經濟效果"},
    "C28-C27": {"enabled": False, "left": "C28", "right": "C27", "description": "同一MR-13A daily PIT source下，feasible-ascent相對minimum-repair的selector轉化效果"},
    "C29-C3": {"enabled": True, "left": "C29", "right": "C3", "description": "current Min ROOS下MR-13A daily feasible-ascent相對DL-off baseline的Forward-OOS策略效果"},
    "C29-C20": {"enabled": True, "left": "C29", "right": "C20", "description": "固定current Min ROOS與feasible-ascent，MR-13A daily相對MR-12B event source的純DL Forward-OOS效果"},
    "C30-C1": {"enabled": True, "left": "C30", "right": "C1", "description": "Full ROOS下MR-12B feasible-ascent相對DL-off baseline的Forward-OOS策略效果"},
    "C31-C1": {"enabled": True, "left": "C31", "right": "C1", "description": "Full ROOS下MR-13A daily feasible-ascent相對DL-off baseline的Forward-OOS策略效果"},
    "C31-C30": {"enabled": True, "left": "C31", "right": "C30", "description": "固定Full ROOS與feasible-ascent，MR-13A daily相對MR-12B event source的純DL Forward-OOS效果"},
    "C1-C3": {"enabled": True, "left": "C1", "right": "C3", "description": "Full ROOS相對current Min ROOS的完整策略體系差異；不是單一參數效果"},
    "C31-C29": {"enabled": True, "left": "C31", "right": "C29", "description": "固定MR-13A daily與feasible-ascent下，Full ROOS相對Min ROOS的完整策略體系interaction"},
    "C12-C11": {"enabled": False, "left": "C12", "right": "C11", "description": "Best-improvement相對first-improvement改善"},
    "C11-C8": {"enabled": False, "left": "C11", "right": "C8", "description": "Resource-aware相對A9 hard-filter改善"},
    "C2-C1": {"enabled": False, "left": "C2", "right": "C1", "description": "Full ROOS下TP1 runtime效果"},
    "C7-C1": {"enabled": False, "left": "C7", "right": "C1", "description": "Full ROOS下A9 runtime效果"},
    "C4-C3": {"enabled": False, "left": "C4", "right": "C3", "description": "Min ROOS下TP1 runtime效果"},
    "C6-C5": {"enabled": False, "left": "C6", "right": "C5", "description": "Min-TP1 ROOS下配對DL效果"},
    "C10-C9": {"enabled": False, "left": "C10", "right": "C9", "description": "Min-A9 ROOS下配對DL效果"},
    "C3-C1": {"enabled": False, "left": "C3", "right": "C1", "description": "Min ROOS相對Full ROOS"},
    "C5-C3": {"enabled": False, "left": "C5", "right": "C3", "description": "TP1-aware參數本身效果"},
    "C9-C3": {"enabled": False, "left": "C9", "right": "C3", "description": "A9-aware參數本身效果"},
    "C6-C4": {"enabled": False, "left": "C6", "right": "C4", "description": "TP1-on下參數適應效果"},
    "C10-C8": {"enabled": False, "left": "C10", "right": "C8", "description": "A9-on下參數適應效果"},
    "C6-C1": {"enabled": False, "left": "C6", "right": "C1", "description": "Min-TP1完整方案相對正式基準"},
    "C10-C1": {"enabled": False, "left": "C10", "right": "C1", "description": "Min-A9完整方案相對正式基準"},
    "C33-C32": {"enabled": False, "left": "C33", "right": "C32", "description": "Selection Full ROOS下MR-12B feasible-ascent相對DL-off baseline的策略效果"},
    "C34-C32": {"enabled": False, "left": "C34", "right": "C32", "description": "Selection Full ROOS下MR-13A daily feasible-ascent相對DL-off baseline的策略效果"},
    "C34-C33": {"enabled": False, "left": "C34", "right": "C33", "description": "固定Selection Full ROOS與feasible-ascent，MR-13A daily相對MR-12B event PIT的純DL source效果"},
    "C32-C23": {"enabled": False, "left": "C32", "right": "C23", "description": "Selection Full ROOS相對Selection Min ROOS的完整策略體系差異；不是單一參數效果"},
    "C34-C28": {"enabled": False, "left": "C34", "right": "C28", "description": "固定MR-13A daily PIT與feasible-ascent下，Selection Full相對Min的完整策略體系interaction"},
}


def _builder(raw) -> StrategyArtifactBuilder | None:
    if raw in (None, {}):
        return None
    return StrategyArtifactBuilder(
        enabled=bool(raw.get("enabled")),
        builder_type=str(raw.get("builder_type") or "").strip(),
        options=dict(raw.get("options") or {}),
    )


def get_strategy_multi_seed_robustness_profiles() -> tuple[dict[str, str], ...]:
    return tuple(
        {"robustness_id": str(robustness_id), "label": str(raw.get("label") or robustness_id)}
        for robustness_id, raw in STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES.items()
        if bool(raw.get("enabled", True))
    )


def get_strategy_multi_seed_robustness_settings(
    robustness_id: str | None = None,
) -> StrategyMultiSeedRobustnessSettings:
    selected_id = str(robustness_id or STRATEGY_COMPARE_DEFAULT_ROBUSTNESS_PROFILE).strip()
    if selected_id not in STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES:
        raise ValueError(f"不存在的multi-seed robustness profile: {selected_id}")
    raw = dict(STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES[selected_id])
    settings = StrategyMultiSeedRobustnessSettings(
        robustness_id=selected_id,
        label=str(raw.get("label") or selected_id).strip(),
        enabled=bool(raw.get("enabled", True)),
        profile_id=str(raw.get("profile_id") or "").strip(),
        seed_count=int(raw.get("seed_count", 0) or 0),
        seed_generator_seed=int(raw.get("seed_generator_seed", 0) or 0),
        gpu_train_workers=int(raw.get("gpu_train_workers", STRATEGY_COMPARE_ROBUSTNESS_GPU_TRAIN_WORKERS)),
        cpu_replay_workers=int(raw.get("cpu_replay_workers", 0) or 0),
        reuse_completed=bool(raw.get("reuse_completed", True)),
        console_mode=str(raw.get("console_mode") or "compact").strip(),
        progress_interval_seconds=float(raw.get("progress_interval_seconds", 60.0) or 60.0),
        yearly_report=bool(raw.get("yearly_report", True)),
        keep_checkpoints=bool(raw.get("keep_checkpoints", False)),
        keep_scores=bool(raw.get("keep_scores", False)),
        keep_replay_details=bool(raw.get("keep_replay_details", False)),
        romd_reference_baselines={
            str(key).strip(): {
                "param_source": str(dict(value or {}).get("param_source") or "").strip(),
                "rule_policy": str(dict(value or {}).get("rule_policy") or "").strip(),
            }
            for key, value in dict(raw.get("romd_reference_baselines") or {}).items()
        },
        output_root=str(raw.get("output_root") or "").strip(),
        model_work_root=str(raw.get("model_work_root") or "").strip(),
    )
    validate_strategy_multi_seed_robustness_settings(settings)
    if settings.profile_id not in STRATEGY_COMPARE_PROFILES:
        raise ValueError(
            f"multi-seed robustness引用不存在的Strategy Compare profile: {settings.profile_id}"
        )
    profile_settings = get_strategy_comparison_settings(settings.profile_id)
    fixed = [arm for arm in profile_settings.enabled_arms if arm.robustness_role == "fixed_baseline"]
    stochastic = [arm for arm in profile_settings.enabled_arms if arm.robustness_role == "stochastic"]
    if not fixed:
        raise ValueError("multi-seed robustness至少需要一個fixed_baseline arm")
    if not stochastic:
        raise ValueError("multi-seed robustness至少需要一個stochastic arm")
    score_sources = {
        profile_settings.dl_sources[str(arm.dl_id)].score_source
        for arm in stochastic if arm.dl_id
    }
    expected_score_source = (
        "selection_point_in_time" if settings.profile_id == "selection_pit"
        else "continuous_ranker_oos"
    )
    if score_sources != {expected_score_source}:
        raise ValueError(
            "multi-seed stochastic arms的score source與robustness階段不一致: "
            f"expected={expected_score_source}, actual={sorted(score_sources)}"
        )
    for reference_key, spec in settings.romd_reference_baselines.items():
        matches = [
            arm for arm in fixed
            if arm.param_source == spec["param_source"]
            and arm.rule_policy == spec["rule_policy"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "multi-seed RoMD reference必須唯一對應一個fixed baseline: "
                f"reference={reference_key}, param_source={spec['param_source']}, "
                f"rule_policy={spec['rule_policy']}, matches={len(matches)}"
            )
    return settings


def get_strategy_comparison_profiles() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "profile_id": str(profile_id),
            "label": str(raw["label"]),
            "description": str(raw.get("description") or ""),
        }
        for profile_id, raw in STRATEGY_COMPARE_PROFILES.items()
    )


def get_strategy_comparison_settings(profile_id: str | None = None) -> StrategyComparisonSettings:
    selected_profile_id = str(profile_id or STRATEGY_COMPARE_DEFAULT_PROFILE).strip()
    if selected_profile_id not in STRATEGY_COMPARE_PROFILES:
        raise ValueError(f"未知Strategy Compare profile: {selected_profile_id}")
    profile = dict(STRATEGY_COMPARE_PROFILES[selected_profile_id])
    profile_arm_ids = tuple(str(value) for value in profile.get("arm_ids", ()))
    profile_contrast_ids = tuple(str(value) for value in profile.get("contrast_ids", ()))
    missing_arms = [arm_id for arm_id in profile_arm_ids if arm_id not in STRATEGY_COMPARE_ARMS]
    missing_contrasts = [
        contrast_id
        for contrast_id in profile_contrast_ids
        if contrast_id not in STRATEGY_COMPARE_CONTRASTS
    ]
    if missing_arms or missing_contrasts:
        raise ValueError(
            "Strategy Compare profile引用不存在的設定: "
            f"arms={missing_arms or '-'}, contrasts={missing_contrasts or '-'}"
        )

    preparation = StrategyPreparationPolicy(
        auto_prepare=bool(STRATEGY_COMPARE_PREPARATION.get("auto_prepare")),
        reuse_ready_artifacts=bool(
            STRATEGY_COMPARE_PREPARATION.get("reuse_ready_artifacts")
        ),
        rebuild_stale_artifacts=bool(
            STRATEGY_COMPARE_PREPARATION.get("rebuild_stale_artifacts")
        ),
        resume_parameter_training=bool(
            STRATEGY_COMPARE_PREPARATION.get("resume_parameter_training")
        ),
        require_confirmation=bool(
            STRATEGY_COMPARE_PREPARATION.get("require_confirmation")
        ),
        reuse_completed_results=bool(
            STRATEGY_COMPARE_PREPARATION.get("reuse_completed_results", True)
        ),
        reuse_shared_baseline=bool(
            STRATEGY_COMPARE_PREPARATION.get("reuse_shared_baseline", True)
        ),
    )
    parameter_sources = {
        source_id: StrategyParameterSource(
            source_id=source_id,
            path_template=raw.get("path_template"),
            description=str(raw.get("description") or "").strip(),
            identity_manifest_path=raw.get("identity_manifest_path"),
            trained_with_dl_id=raw.get("trained_with_dl_id"),
            artifact_contract=(
                None
                if raw.get("artifact_contract") in (None, {})
                else dict(raw.get("artifact_contract") or {})
            ),
            builder=_builder(raw.get("builder")),
        )
        for source_id, raw in STRATEGY_PARAM_SOURCES.items()
    }
    dl_sources = {
        dl_id: StrategyDLSource(
            dl_id=dl_id,
            filter_id=str(raw.get("filter_id") or "").strip(),
            model_architecture=str(raw.get("model_architecture") or "").strip(),
            experiment_profile=str(raw.get("experiment_profile") or "").strip(),
            threshold=(None if raw.get("threshold") in (None, "") else float(raw.get("threshold"))),
            description=str(raw.get("description") or "").strip(),
            score_source=str(raw.get("score_source") or "canonical_runtime").strip(),
            forward_scores_builder=_builder(raw.get("forward_scores_builder")),
        )
        for dl_id, raw in STRATEGY_DL_SOURCES.items()
    }
    ordered_arm_ids = tuple(dict.fromkeys((*profile_arm_ids, *STRATEGY_COMPARE_ARMS.keys())))
    arms = {
        arm_id: StrategyComparisonArm(
            arm_id=arm_id,
            enabled=arm_id in profile_arm_ids,
            name=str(raw.get("name") or "").strip(),
            description=str(raw.get("description") or "").strip(),
            param_source=str(raw.get("param_source") or "").strip(),
            rule_policy=str(raw.get("rule_policy") or "").strip(),
            dl_enabled=bool(raw.get("dl_enabled")),
            dl_id=(None if raw.get("dl_id") in (None, "") else str(raw.get("dl_id"))),
            dl_runtime_mode=(
                None
                if raw.get("dl_runtime_mode") in (None, "")
                else str(raw.get("dl_runtime_mode")).strip()
            ),
            dl_runtime_options=(
                None
                if raw.get("dl_runtime_options") in (None, {})
                else dict(raw.get("dl_runtime_options") or {})
            ),
            robustness_role=str(raw.get("robustness_role") or "off").strip(),
        )
        for arm_id in ordered_arm_ids
        for raw in (STRATEGY_COMPARE_ARMS[arm_id],)
    }
    ordered_contrast_ids = tuple(dict.fromkeys((*profile_contrast_ids, *STRATEGY_COMPARE_CONTRASTS.keys())))
    contrasts = {
        contrast_id: StrategyComparisonContrast(
            contrast_id=contrast_id,
            enabled=contrast_id in profile_contrast_ids,
            left=str(raw.get("left") or "").strip(),
            right=str(raw.get("right") or "").strip(),
            description=str(raw.get("description") or "").strip(),
        )
        for contrast_id in ordered_contrast_ids
        for raw in (STRATEGY_COMPARE_CONTRASTS[contrast_id],)
    }
    workflow_settings = get_breakout_quality_workflow_settings()
    settings = StrategyComparisonSettings(
        schema_version=int(STRATEGY_COMPARE_SCHEMA_VERSION),
        profile_id=selected_profile_id,
        profile_label=str(profile["label"]),
        dataset=str(workflow_settings.strategy_dataset).strip(),
        start_date=(
            None
            if profile.get("start_date") in (None, "")
            else str(profile.get("start_date")).strip()
        ),
        end_date=(
            None
            if profile.get("end_date") in (None, "")
            else str(profile.get("end_date")).strip()
        ),
        param_policy=str(workflow_settings.strategy_param_policy).strip(),
        max_positions=int(workflow_settings.strategy_max_positions),
        rotation=str(workflow_settings.strategy_rotation).strip(),
        output_root=str(profile["output_root"]).strip(),
        reuse_output_roots=tuple(
            str(value).strip()
            for value in tuple(profile.get("reuse_output_roots") or ())
            if str(value).strip()
        ),
        preparation=preparation,
        parameter_sources=parameter_sources,
        dl_sources=dl_sources,
        arms=arms,
        contrasts=contrasts,
    )
    validate_strategy_comparison_settings(settings)
    return settings


__all__ = [
    "STRATEGY_COMPARE_ARMS",
    "STRATEGY_COMPARE_CONTRASTS",
    "STRATEGY_COMPARE_PREPARATION",
    "STRATEGY_COMPARE_PROFILES",
    "STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES",
    "STRATEGY_DL_SOURCES",
    "STRATEGY_PARAM_SOURCES",
    "get_strategy_comparison_profiles",
    "get_strategy_multi_seed_robustness_profiles",
    "get_strategy_multi_seed_robustness_settings",
    "get_strategy_comparison_settings",
]

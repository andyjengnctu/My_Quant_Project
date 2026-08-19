"""目前正式 Strategy Compare 設定。

Active framework以Extending-Window Rolling作唯一策略績效主線：expanding history、
每個PIT fold重新選epoch/refit，再評分下一段。舊Selection PIT／Frozen Forward
profiles保留作歷史工件解讀／重現，但不再暴露於主選單或作current Gate。
Fixed-Window Rolling屬模型穩定性診斷，不建立另一套production strategy truth。
"""

from __future__ import annotations

from config.breakout_quality import get_breakout_quality_workflow_settings
from config.execution_policy import DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT
from config.compatibility.strategy_compare_history import (
    HISTORICAL_STRATEGY_COMPARE_ARMS,
    HISTORICAL_STRATEGY_COMPARE_CONTRASTS,
    HISTORICAL_STRATEGY_DL_SOURCES,
    HISTORICAL_STRATEGY_PARAM_SOURCES,
)
from config.training_policy import (
    OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    OPTIMIZER_RANDOM_SEED_DEFAULT,
    OUTER_ROLLING_OOS_HORIZON_MONTHS,
    OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
)
from core.strategy_comparison import (
    StrategyArtifactBuilder,
    StrategyComparisonArm,
    StrategyComparisonContrast,
    StrategyComparisonSettings,
    StrategyMultiSeedRobustnessSettings,
    StrategyRuntimeIntegrationSettings,
    StrategyDLSource,
    StrategyParameterSource,
    StrategyPreparationPolicy,
    validate_strategy_comparison_settings,
    validate_strategy_multi_seed_robustness_settings,
    validate_strategy_runtime_integration_settings,
)

STRATEGY_COMPARE_SCHEMA_VERSION = 46

# =============================================================================
# 1. 常用設定
#    一般 Strategy Compare / robustness 實驗通常只需修改本區。
#    Dataset / param policy / max positions / rotation 不在此複製；它們直接
#    繼承 config/breakout_quality.py 的 BreakoutQualityWorkflowSettings SSOT。
# =============================================================================

STRATEGY_COMPARE_DEFAULT_PROFILE = "pre_test"
# 主互動選單只暴露泛化工作階段；研究 profile identity 留在 config/報表。
STRATEGY_COMPARE_MENU_PROFILE_IDS = ("pre_test", "extending_window_rolling")
STRATEGY_COMPARE_DEFAULT_ROBUSTNESS_PROFILE = "extending_window_rolling"
STRATEGY_COMPARE_ROBUSTNESS_SEED_COUNT = 4
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
STRATEGY_COMPARE_ROBUSTNESS_KEEP_ATTRIBUTION_SOURCE = True

# Runtime promotion只讀取已存在的正式Strategy Compare / robustness工件；
# Gate本身不訓練模型、不重跑策略，也不自動切換runtime default。
STRATEGY_RUNTIME_INTEGRATION = {
    "label": "Runtime 整合 Gate",
    "enabled": False,
    "selection_profile_id": "selection_pit",
    "forward_profile_id": "forward_oos",
    # Runtime promotion candidate由config明確指定；研究matrix可同時存在其他DL arms。
    "selection_candidate_arm_id": "C42",
    "forward_candidate_arm_id": "C44",
    "selection_robustness_id": "selection_pit",
    "forward_robustness_id": "forward_oos",
    "output_root": "outputs/strategy_compare/runtime_integration",
    # Gate通過後workflow可升格；robustness對照仍固定為升格前production anchor。
    "comparison_anchor_experiment_profile": "strategy_aligned_no_time_all_event_pairwise",
    # 舊exact implementation曾出現>10秒困難case；正式候選不得退回該等級。
    "max_selector_latency_ms": 10000.0,
    # 多seed至少必須嚴格過半勝過目前workflow runtime anchor的RoMD。
    "require_strict_romd_majority": True,
}

# Current Strategy Compare核心比較名稱的單一真理。
# Selection PIT／Forward-OOS由profile頁首區分，不把研究階段或固定selector語意塞進arm顯示名稱。
STRATEGY_COMPARE_DISPLAY_FULL_ROOS = "Full ROOS"
STRATEGY_COMPARE_DISPLAY_MIN_ROOS = "Min ROOS"
STRATEGY_COMPARE_DISPLAY_MIN_MR13E_SCORE_CONSTRAINED = "Min MR-13E Constrained"
STRATEGY_COMPARE_DISPLAY_MIN_MR13K_SCORE_CONSTRAINED = "Min MR-13K Constrained"
STRATEGY_COMPARE_DISPLAY_MIN_MR13K_MR13M_SAFETY_CONSTRAINED = "Min MR-13K + MR-13M Safety"
STRATEGY_COMPARE_DISPLAY_MIN_MR13K_MR13M_RESIDUAL_SAFETY_CONSTRAINED = "Min MR-13K + MR-13M Residual Safety"

# Strategy Compare以研究階段profile隔離設定與輸出；App只顯示泛化階段名稱，
# arms／contrasts／period／output namespace全部由本檔驅動。
STRATEGY_COMPARE_PROFILES = {
    "pre_test": {
        "label": "Pre-Test 策略比較",
        "description": (
            "單模型快速研究Gate：沿用原本full-Selection refit後的2021+ continuous-ranker OOS scores，"
            "固定比較Full ROOS、Min ROOS、MR-13E reference與目前B2 candidate。只決定是否值得進正式Rolling，"
            "不得取代2016～2025十fold Extending-Window evidence。"
        ),
        "display_alignment_group": "pre_test_strategy_compare",
        "display_alignment_arm_ids": ("C1", "C3", "C44", "C56"),
        "start_date": None,
        "end_date": None,
        "output_root": "outputs/strategy_compare/pre_test",
        "reuse_output_roots": ("outputs/strategy_compare/forward_oos",),
        "arm_ids": ("C1", "C3", "C44", "C56"),
        "contrast_ids": (
            "C1-C3", "C44-C3", "C44-C1",
            "C56-C44", "C56-C3", "C56-C1",
        ),
    },
    "extending_window_rolling": {
        "label": "Extending-Window Rolling 策略比較",
        "description": (
            "2016→2025十個完整年度fold的單一PIT-safe operational chain；DL使用expanding history + "
            "annual refit；Full／Min ROOS都沿用各時期當時合法rolling params形成共同策略體系基準。"
            "2021不再形成evaluation policy斷點。"
        ),
        "display_alignment_group": "extending_strategy_compare",
        "display_alignment_arm_ids": ("C61", "C58", "C59", "C60"),
        "start_date": "2016-01-01",
        "end_date": "2025-12-31",
        "output_root": "outputs/strategy_compare/extending_window_rolling",
        "reuse_output_roots": (
            "outputs/strategy_compare/selection_pit",
            "outputs/strategy_compare/forward_oos",
        ),
        "arm_ids": ("C61", "C58", "C59", "C60"),
        "contrast_ids": (
            "C61-C58", "C59-C58", "C59-C61",
            "C60-C58", "C60-C61", "C60-C59",
        ),
    },
    # Legacy evaluation policies retained only for historical replay / artifact interpretation.
    "selection_pit": {
        "label": "Selection PIT 策略比較",
        "description": "2014～2020 Selection PIT策略Gate；固定Full/Min references與production MR-13E C42，current C56 full-flow只新增C57：MR-13K PIT primary objective + MR-13M PIT同日rank residual safety floor。",
        "display_alignment_group": "core_strategy_compare",
        "display_alignment_arm_ids": ("C32", "C23", "C42"),
        "start_date": "2014-01-01",
        "end_date": "2020-12-31",
        "output_root": "outputs/strategy_compare/selection_pit",
        "reuse_output_roots": ("outputs/strategy_compare",),
        "arm_ids": ("C32", "C23", "C42", "C57"),
        "contrast_ids": (
            "C32-C23",
            "C42-C23",
            "C42-C32",
            "C57-C42",
            "C57-C23",
            "C57-C32",
        ),
    },
    "forward_oos": {
        "label": "Forward-OOS 策略比較",
        "description": "2021+ frozen Forward-OOS策略Gate；固定Full/Min references與production MR-13E C44，current C56 full-flow只保留C56 B2：MR-13K upside objective + MR-13M同日rank-space residual safety floor。C54/C55保留historical result但不再跟跑。",
        "display_alignment_group": "core_strategy_compare",
        "display_alignment_arm_ids": ("C1", "C3", "C44"),
        "start_date": None,
        "end_date": None,
        "output_root": "outputs/strategy_compare/forward_oos",
        "reuse_output_roots": ("outputs/strategy_compare",),
        "arm_ids": ("C1", "C3", "C44", "C56"),
        "contrast_ids": (
            "C1-C3",
            "C44-C3",
            "C44-C1",
            "C56-C44",
            "C56-C3",
            "C56-C1",
        ),
    },
}

# Multiple-seed robustness各研究階段以獨立config profile呈現於正式選單。
# stochastic/fixed比較對象由robustness profile顯式指定；
# robustness orchestration不得改動單次Strategy Compare arm identity/fingerprint。
STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES = {
    "extending_window_rolling": {
        "label": "Extending-Window Rolling Multi-seed robustness",
        "enabled": True,
        "profile_id": "extending_window_rolling",
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
        "keep_attribution_source": STRATEGY_COMPARE_ROBUSTNESS_KEEP_ATTRIBUTION_SOURCE,
        "romd_reference_baselines": {
            "min": {"param_source": "extending_min_roos", "rule_policy": "all_off"},
        },
        "fixed_arm_ids": ("C58",),
        "stochastic_arm_ids": ("C60",),
        "paired_contrasts": (),
        "output_root": "outputs/strategy_compare/robustness/extending_window_rolling",
        "model_work_root": "models/research/breakout_quality/strategy_compare/multi_seed_robustness/extending_window_rolling",
    },
    "selection_pit": {
        "label": "Selection PIT Multi-seed robustness (Legacy)",
        "enabled": False,
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
        "keep_attribution_source": STRATEGY_COMPARE_ROBUSTNESS_KEEP_ATTRIBUTION_SOURCE,
        "romd_reference_baselines": {
            "min": {"param_source": "selection_min_roos", "rule_policy": "all_off"},
            "full": {"param_source": "selection_full_roos", "rule_policy": "formal"},
        },
        "fixed_arm_ids": ("C32", "C23"),
        "stochastic_arm_ids": ("C57",),
        "paired_contrasts": (),
        "output_root": "outputs/strategy_compare/robustness/selection_pit",
        "model_work_root": "models/research/breakout_quality/strategy_compare/multi_seed_robustness/selection_pit",
    },
    "forward_oos": {
        "label": "Forward-OOS Multi-seed robustness (Legacy)",
        "enabled": False,
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
        "keep_attribution_source": STRATEGY_COMPARE_ROBUSTNESS_KEEP_ATTRIBUTION_SOURCE,
        "romd_reference_baselines": {
            "min": {"param_source": "min_roos", "rule_policy": "all_off"},
            "full": {"param_source": "full_roos", "rule_policy": "formal"},
        },
        "fixed_arm_ids": ("C1", "C3"),
        "stochastic_arm_ids": ("C56",),
        "paired_contrasts": (),
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
    "extending_min_roos": {
        "path_template": (
            "models/research/breakout_quality/strategy_compare/extending_min_roos/"
            "active_params/{param_filename}"
        ),
        "description": (
            "Extending-Window 2016→2025 Min ROOS；stitch既有2014-2020 historical P2與2021+ "
            "current P2 rolling schedules；只在兩段rolling/search contract一致時建立，不重新最佳化。"
        ),
        "identity_manifest_path": (
            "models/research/breakout_quality/strategy_compare/extending_min_roos/"
            "extending_stitch_manifest.json"
        ),
        "trained_with_dl_id": None,
        "artifact_contract": {
            "breakout_quality_param_adaptation": {
                "mode": "min_roos_training",
                "parameter_set": "P2_EXTENDING",
                "search_fields": [
                    "high_len", "atr_len", "atr_buy_tol", "atr_times_init", "atr_times_trail"
                ],
                "fixed_rule_contract": "all_rule_filters_off",
                "training_dl_enabled": False,
            }
        },
        "builder": {
            "enabled": True,
            "builder_type": "extending_min_roos_stitch",
            "options": {
                "historical_params_path": (
                    "models/research/breakout_quality/trade_path_label/a2_teacher_params/"
                    "p2_dl_off_trained/active_params/{param_filename}"
                ),
                "current_params_path": (
                    "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
                    "risk_only_rolling/p2_dl_off_trained/active_params/{param_filename}"
                ),
                "output_relative_dir": (
                    "models/research/breakout_quality/strategy_compare/extending_min_roos"
                ),
                "quiet": False,
            },
        },
    },
    "extending_full_roos": {
        "path_template": (
            "models/research/breakout_quality/strategy_compare/extending_full_roos/"
            "active_params/{param_filename}"
        ),
        "description": (
            "Extending-Window 2016→2025 Full ROOS；stitch既有2014-2020 historical P4與2021+ "
            "canonical Full rolling schedules；只在rolling/search contract一致時建立，不重新最佳化。"
        ),
        "identity_manifest_path": (
            "models/research/breakout_quality/strategy_compare/extending_full_roos/"
            "extending_stitch_manifest.json"
        ),
        "trained_with_dl_id": None,
        "artifact_contract": {
            "breakout_quality_param_adaptation": {
                "mode": "extending_full_roos_stitch",
                "parameter_set": "P4_EXTENDING",
                "training_dl_enabled": False,
            }
        },
        "builder": {
            "enabled": True,
            "builder_type": "extending_full_roos_stitch",
            "options": {
                "historical_params_path": (
                    "models/research/breakout_quality/strategy_compare/selection_full_roos/"
                    "active_params/{param_filename}"
                ),
                "current_params_path": "models/{param_filename}",
                "output_relative_dir": (
                    "models/research/breakout_quality/strategy_compare/extending_full_roos"
                ),
                "quiet": False,
            },
        },
    },
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
                "fixed_risk": DEFAULT_FIXED_RISK,
                "max_position_cap_pct": DEFAULT_MAX_POSITION_CAP_PCT,
                "build_binary_pit": False,
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
                "fixed_risk": DEFAULT_FIXED_RISK,
                "max_position_cap_pct": DEFAULT_MAX_POSITION_CAP_PCT,
                "optimizer_seed": OPTIMIZER_RANDOM_SEED_DEFAULT,
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
                "fixed_risk": DEFAULT_FIXED_RISK,
                "max_position_cap_pct": DEFAULT_MAX_POSITION_CAP_PCT,
                "optimizer_seed": OPTIMIZER_RANDOM_SEED_DEFAULT,
                "quiet": False,
            },
        },
    },
}

# =============================================================================
# 4. DL工件來源
# =============================================================================

STRATEGY_DL_SOURCES = {
    "CONT13E_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_no_time_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": "MR-13E Extending-Window Rolling PIT-safe score；expanding history + annual refit，2016→2025。",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13K_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": "MR-13K Extending-Window Rolling PIT-safe pure-MFE score；expanding history + annual refit。",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13M_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": "MR-13M Extending-Window Rolling PIT-safe low-adverse score；只作C60 residual-safety secondary source。",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13E": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_no_time_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": (
            "MR-13E Daily Universal full-list Delta-NDCG frozen Forward-OOS score；"
            "盤前使用最新已完成交易日資訊，供current MR-13E exact strategy research"
        ),
        "forward_scores_builder": None,
    },
    "CONT13E_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_no_time_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-13E Daily Universal full-list Delta-NDCG Selection PIT score；"
            "盤前只使用最新已完成交易日資訊，供2014～2020 current MR-13E exact strategy research"
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
    "CONT13K": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": (
            "MR-13K Daily Universal full-horizon pure-MFE frozen Forward-OOS score；"
            "盤前使用最新已完成交易日資訊，供C54與C44作同K/R0/exact source-only策略比較"
        ),
        "forward_scores_builder": None,
    },
    "CONT13M": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": (
            "MR-13M Daily Universal full-horizon low-adverse frozen Forward-OOS score；"
            "只作C55/C56 secondary path-safety source；C55使用raw safety floor，C56使用同日rank-space residual safety floor；皆不與MR-13K score加權、不單獨作portfolio objective"
        ),
        "forward_scores_builder": None,
    },
    "CONT13M_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-13M Daily Universal full-horizon low-adverse Selection PIT score；"
            "只供C57作C56 B2的Selection secondary residual-safety source，不單獨作portfolio objective"
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
    "CONT13K_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-13K Daily Universal full-horizon pure-MFE Selection PIT score；"
            "盤前只使用最新已完成交易日資訊，供C53與C42作同K/R0/exact source-only策略比較"
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
# 5. Strategy arm definitions
# =============================================================================
# Arm 是否啟用只由 STRATEGY_COMPARE_PROFILES[*]["arm_ids"] 決定；
# arm definition 本身不再保存第二份 enabled 狀態。

STRATEGY_COMPARE_ARMS = {
    "C61": {
        "name": STRATEGY_COMPARE_DISPLAY_FULL_ROOS,
        "description": (
            "Extending-Window 2016→2025 stitched Full ROOS rolling params；formal rules；DL-off baseline"
        ),
        "param_source": "extending_full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C58": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_ROOS,
        "description": "Extending-Window 2016→2025 stitched Min ROOS rolling params；rules全關；DL-off baseline",
        "param_source": "extending_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C59": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13E_SCORE_CONSTRAINED,
        "description": "Extending-Window Rolling MR-13E exact constrained；每個score fold只用當時已成熟歷史訓練。",
        "param_source": "extending_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C60": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13K_MR13M_RESIDUAL_SAFETY_CONSTRAINED,
        "description": (
            "Extending-Window Rolling B2：MR-13K primary + MR-13M same-day rank OLS residual-safety floor；"
            "兩個score都由同一日期前已成熟資料的PIT-safe annual-refit model產生。"
        ),
        "param_source": "extending_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13K_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-residual-safety-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
            "safety_dl_id": "CONT13M_ROLL",
            "safety_constraint": "baseline_residual_coverage_and_score_sum_floor_v1",
            "safety_residualization": "same_day_rank_ols_v1",
        },
        "robustness_role": "off",
    },
    "C1": {
        "name": STRATEGY_COMPARE_DISPLAY_FULL_ROOS,
        "description": "Full optimizer rolling active params；DL-off baseline",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C3": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_ROOS,
        "description": "只搜尋high_len＋4個ATR；rules全關；DL-off baseline",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C23": {
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
    "C42": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13E_SCORE_CONSTRAINED,
        "description": (
            "Selection PIT current research arm：historical Min params/all-off + frozen MR-13E PIT score；"
            "固定K/R0、canonical sizing/cash/orderability/execution，以deterministic exact branch-and-bound"
            "在完整候選universe求K/R0/canonical-cash feasible MR-13E score-sum global optimum"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C44": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13E_SCORE_CONSTRAINED,
        "description": (
            "Forward-OOS current research arm：current Min params/all-off + frozen MR-13E Forward score；"
            "固定K/R0、canonical sizing/cash/orderability/execution，以與C42同源deterministic exact "
            "branch-and-bound在完整候選universe求K/R0/canonical-cash feasible MR-13E score-sum global optimum"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": False,
        },
        "robustness_role": "off",
    },
    "C56": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13K_MR13M_RESIDUAL_SAFETY_CONSTRAINED,
        "description": (
            "Forward-only Plan B2 controlled arm：完全沿用C54的Min params/all-off、K/R0、canonical "
            "sizing/cash/orderability/execution與MR-13K primary score objective；每天在當日orderable候選中，"
            "先把MR-13K/MR-13M frozen scores各自轉average-rank percentile，再以含intercept OLS估計"
            "expected safety rank given upside rank，Residual Safety=actual safety percentile−expected safety percentile。"
            "選中basket的Residual Safety coverage與score-sum不得低於同日DL-off Min ROOS baseline；13M residual"
            "只作hard floor，不與13K objective加權、不使用未來Target、不新增numeric threshold。"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13K",
        "dl_runtime_mode": "resource-aware-continuous-score-residual-safety-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": False,
            "safety_dl_id": "CONT13M",
            "safety_constraint": "baseline_residual_coverage_and_score_sum_floor_v1",
            "safety_residualization": "same_day_rank_ols_v1",
        },
        "robustness_role": "off",
    },
    "C57": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13K_MR13M_RESIDUAL_SAFETY_CONSTRAINED,
        "description": (
            "Selection PIT C56 full-flow counterpart：historical Min params/all-off、K/R0、canonical "
            "sizing/cash/orderability/execution與exact solver固定；primary改用CONT13K_PIT，secondary改用"
            "CONT13M_PIT。每天只在當日orderable候選把兩個PIT-safe score轉rank percentile並做同日OLS residual；"
            "Residual Safety只作baseline-relative hard floor，primary objective仍只最大化MR-13K score-sum。"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13K_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-residual-safety-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
            "safety_dl_id": "CONT13M_PIT",
            "safety_constraint": "baseline_residual_coverage_and_score_sum_floor_v1",
            "safety_residualization": "same_day_rank_ols_v1",
        },
        "robustness_role": "off",
    },
    "C32": {
        "name": STRATEGY_COMPARE_DISPLAY_FULL_ROOS,
        "description": "2014～2020 historical Full ROOS active params；formal rules；DL-off共同baseline",
        "param_source": "selection_full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
}

# =============================================================================
# 6. Contrast definitions
# =============================================================================
# Contrast 是否啟用只由 STRATEGY_COMPARE_PROFILES[*]["contrast_ids"] 決定。

STRATEGY_COMPARE_CONTRASTS = {
    "C61-C58": {"left": "C61", "right": "C58", "description": "Extending-Window Rolling Full ROOS相對同期間Min ROOS的完整策略體系差異；不是單一參數效果"},
    "C59-C58": {"left": "C59", "right": "C58", "description": "Extending-Window Rolling MR-13E相對同期間DL-off Min ROOS的增量策略效果"},
    "C59-C61": {"left": "C59", "right": "C61", "description": "Extending-Window Rolling MR-13E Min策略相對同期間Full ROOS的整體策略結果；不是單一DL效果"},
    "C60-C58": {"left": "C60", "right": "C58", "description": "Extending-Window Rolling B2相對同期間DL-off Min ROOS的增量策略效果"},
    "C60-C61": {"left": "C60", "right": "C61", "description": "Extending-Window Rolling B2 Min策略相對同期間Full ROOS的整體策略結果；不是單一DL效果"},
    "C60-C59": {"left": "C60", "right": "C59", "description": "Extending-Window Rolling B2相對MR-13E production reference語意的同政策比較"},
    "C42-C23": {"left": "C42", "right": "C23", "description": "Selection PIT frozen MR-13E score exact constrained optimum相對DL-off Min ROOS的策略經濟效果"},
    "C42-C32": {"left": "C42", "right": "C32", "description": "Selection PIT active research最終候選：Min MR-13E exact constrained相對Full ROOS的整體策略結果；不是單一參數或單一DL效果"},
    "C57-C42": {"left": "C57", "right": "C42", "description": "C56 full-flow Selection primary contrast：同historical Min/K/R0/exact/cash/execution下，以MR-13K PIT primary + MR-13M PIT residual safety對production MR-13E exact reference"},
    "C57-C23": {"left": "C57", "right": "C23", "description": "C56 full-flow Selection相對DL-off Min ROOS的策略經濟效果"},
    "C57-C32": {"left": "C57", "right": "C32", "description": "C56 full-flow Selection相對Full ROOS的整體策略結果；不是單一參數效果"},
    "C56-C44": {"left": "C56", "right": "C44", "description": "Plan B2 Forward相對production MR-13E exact constrained reference的整體策略結果"},
    "C56-C3": {"left": "C56", "right": "C3", "description": "Plan B2 Forward相對DL-off Min ROOS的策略經濟效果"},
    "C56-C1": {"left": "C56", "right": "C1", "description": "Plan B2 Forward相對Full ROOS的整體策略結果；不是單一參數效果"},
    "C44-C3": {"left": "C44", "right": "C3", "description": "current Min ROOS下MR-13E exact constrained score selector相對DL-off baseline的Forward-OOS策略效果"},
    "C44-C1": {"left": "C44", "right": "C1", "description": "Forward-OOS active research最終候選：Min MR-13E exact constrained相對Full ROOS的整體策略結果；不是單一參數或單一DL效果"},
    "C1-C3": {"left": "C1", "right": "C3", "description": "Full ROOS相對current Min ROOS的完整策略體系差異；不是單一參數效果"},
    "C32-C23": {"left": "C32", "right": "C23", "description": "Selection Full ROOS相對Selection Min ROOS的完整策略體系差異；不是單一參數效果"},
}

def _merge_compatibility_catalog(
    active: dict[str, dict],
    historical: dict[str, dict],
    *,
    catalog_name: str,
) -> dict[str, dict]:
    overlap = sorted(set(active).intersection(historical))
    if overlap:
        raise ValueError(
            f"Strategy Compare active/historical {catalog_name} ID重複: {overlap}"
        )
    return {**active, **historical}


def _compatibility_catalogs() -> tuple[
    dict[str, dict],
    dict[str, dict],
    dict[str, dict],
    dict[str, dict],
]:
    """Return runtime catalogs including read-only historical compatibility entries."""

    return (
        _merge_compatibility_catalog(
            STRATEGY_PARAM_SOURCES,
            HISTORICAL_STRATEGY_PARAM_SOURCES,
            catalog_name="parameter source",
        ),
        _merge_compatibility_catalog(
            STRATEGY_DL_SOURCES,
            HISTORICAL_STRATEGY_DL_SOURCES,
            catalog_name="DL source",
        ),
        _merge_compatibility_catalog(
            STRATEGY_COMPARE_ARMS,
            HISTORICAL_STRATEGY_COMPARE_ARMS,
            catalog_name="arm",
        ),
        _merge_compatibility_catalog(
            STRATEGY_COMPARE_CONTRASTS,
            HISTORICAL_STRATEGY_COMPARE_CONTRASTS,
            catalog_name="contrast",
        ),
    )


def _builder(raw) -> StrategyArtifactBuilder | None:
    if raw in (None, {}):
        return None
    return StrategyArtifactBuilder(
        enabled=bool(raw.get("enabled")),
        builder_type=str(raw.get("builder_type") or "").strip(),
        options=dict(raw.get("options") or {}),
    )


def get_strategy_runtime_integration_settings() -> StrategyRuntimeIntegrationSettings:
    raw = dict(STRATEGY_RUNTIME_INTEGRATION)
    settings = StrategyRuntimeIntegrationSettings(
        label=str(raw.get("label") or "").strip(),
        enabled=bool(raw.get("enabled", True)),
        selection_profile_id=str(raw.get("selection_profile_id") or "").strip(),
        forward_profile_id=str(raw.get("forward_profile_id") or "").strip(),
        selection_candidate_arm_id=str(raw.get("selection_candidate_arm_id") or "").strip(),
        forward_candidate_arm_id=str(raw.get("forward_candidate_arm_id") or "").strip(),
        selection_robustness_id=str(raw.get("selection_robustness_id") or "").strip(),
        forward_robustness_id=str(raw.get("forward_robustness_id") or "").strip(),
        output_root=str(raw.get("output_root") or "").strip(),
        comparison_anchor_experiment_profile=str(
            raw.get("comparison_anchor_experiment_profile") or ""
        ).strip(),
        require_strict_romd_majority=bool(raw.get("require_strict_romd_majority", True)),
        max_selector_latency_ms=float(raw.get("max_selector_latency_ms", 10000.0)),
    )
    validate_strategy_runtime_integration_settings(settings)
    profile_ids = set(STRATEGY_COMPARE_PROFILES)
    robustness_ids = set(STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES)
    if settings.selection_profile_id not in profile_ids or settings.forward_profile_id not in profile_ids:
        raise ValueError("runtime integration引用不存在的Strategy Compare profile")
    candidate_specs = (
        (settings.selection_profile_id, settings.selection_candidate_arm_id, "Selection"),
        (settings.forward_profile_id, settings.forward_candidate_arm_id, "Forward"),
    )
    for profile_id, arm_id, stage_label in candidate_specs:
        active_ids = {str(value) for value in STRATEGY_COMPARE_PROFILES[profile_id].get("arm_ids", ())}
        if arm_id not in active_ids:
            raise ValueError(
                f"runtime integration {stage_label} candidate不在active profile: {arm_id}"
            )
    if settings.selection_robustness_id not in robustness_ids or settings.forward_robustness_id not in robustness_ids:
        raise ValueError("runtime integration引用不存在的robustness profile")
    return settings


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
        keep_attribution_source=bool(raw.get("keep_attribution_source", True)),
        romd_reference_baselines={
            str(key).strip(): {
                "param_source": str(dict(value or {}).get("param_source") or "").strip(),
                "rule_policy": str(dict(value or {}).get("rule_policy") or "").strip(),
            }
            for key, value in dict(raw.get("romd_reference_baselines") or {}).items()
        },
        fixed_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("fixed_arm_ids") or ()) if str(value).strip()),
        stochastic_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("stochastic_arm_ids") or ()) if str(value).strip()),
        paired_contrasts=tuple(
            {
                "contrast_id": str(dict(item or {}).get("contrast_id") or "").strip(),
                "left": str(dict(item or {}).get("left") or "").strip(),
                "right": str(dict(item or {}).get("right") or "").strip(),
                "description": str(dict(item or {}).get("description") or "").strip(),
            }
            for item in tuple(raw.get("paired_contrasts") or ())
        ),
        output_root=str(raw.get("output_root") or "").strip(),
        model_work_root=str(raw.get("model_work_root") or "").strip(),
    )
    validate_strategy_multi_seed_robustness_settings(settings)
    if settings.profile_id not in STRATEGY_COMPARE_PROFILES:
        raise ValueError(
            f"multi-seed robustness引用不存在的Strategy Compare profile: {settings.profile_id}"
        )
    profile_settings = get_strategy_comparison_settings(settings.profile_id)
    enabled_by_id = {arm.arm_id: arm for arm in profile_settings.enabled_arms}
    missing_fixed = [arm_id for arm_id in settings.fixed_arm_ids if arm_id not in enabled_by_id]
    missing_stochastic = [arm_id for arm_id in settings.stochastic_arm_ids if arm_id not in enabled_by_id]
    if missing_fixed or missing_stochastic:
        raise ValueError(
            "multi-seed robustness引用未啟用或不存在的arm: "
            f"fixed={missing_fixed}, stochastic={missing_stochastic}"
        )
    fixed = [enabled_by_id[arm_id] for arm_id in settings.fixed_arm_ids]
    stochastic = [enabled_by_id[arm_id] for arm_id in settings.stochastic_arm_ids]
    stochastic_ids = {arm.arm_id for arm in stochastic}
    for spec in settings.paired_contrasts:
        left = str(dict(spec).get("left") or "")
        right = str(dict(spec).get("right") or "")
        if left not in stochastic_ids or right not in stochastic_ids:
            raise ValueError(
                "multi-seed paired contrast只能引用目前stochastic arms: "
                f"contrast={dict(spec).get('contrast_id')}, left={left}, right={right}, "
                f"stochastic={sorted(stochastic_ids)}"
            )
    score_sources = {
        profile_settings.dl_sources[str(arm.dl_id)].score_source
        for arm in stochastic if arm.dl_id
    }
    expected_score_source = (
        "selection_point_in_time"
        if settings.profile_id in {"selection_pit", "extending_window_rolling"}
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
    """Return the full configured profile catalog, including historical/replay profiles."""
    return tuple(
        {
            "profile_id": str(profile_id),
            "label": str(raw["label"]),
            "description": str(raw.get("description") or ""),
        }
        for profile_id, raw in STRATEGY_COMPARE_PROFILES.items()
    )


def get_strategy_comparison_menu_profiles() -> tuple[dict[str, str], ...]:
    """Return only the config-selected generic work stages exposed by the main menu."""
    missing = [
        profile_id
        for profile_id in STRATEGY_COMPARE_MENU_PROFILE_IDS
        if profile_id not in STRATEGY_COMPARE_PROFILES
    ]
    if missing:
        raise ValueError(f"Strategy Compare主選單引用不存在profile: {missing}")
    return tuple(
        {
            "profile_id": str(profile_id),
            "label": str(STRATEGY_COMPARE_PROFILES[profile_id]["label"]),
            "description": str(STRATEGY_COMPARE_PROFILES[profile_id].get("description") or ""),
        }
        for profile_id in STRATEGY_COMPARE_MENU_PROFILE_IDS
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

    (
        parameter_catalog,
        dl_catalog,
        arm_catalog,
        contrast_catalog,
    ) = _compatibility_catalogs()

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
        for source_id, raw in parameter_catalog.items()
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
        for dl_id, raw in dl_catalog.items()
    }
    ordered_arm_ids = tuple(dict.fromkeys((*profile_arm_ids, *arm_catalog.keys())))
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
        for raw in (arm_catalog[arm_id],)
    }
    ordered_contrast_ids = tuple(dict.fromkeys((*profile_contrast_ids, *contrast_catalog.keys())))
    contrasts = {
        contrast_id: StrategyComparisonContrast(
            contrast_id=contrast_id,
            enabled=contrast_id in profile_contrast_ids,
            left=str(raw.get("left") or "").strip(),
            right=str(raw.get("right") or "").strip(),
            description=str(raw.get("description") or "").strip(),
        )
        for contrast_id in ordered_contrast_ids
        for raw in (contrast_catalog[contrast_id],)
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
    "STRATEGY_COMPARE_MENU_PROFILE_IDS",
    "STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES",
    "STRATEGY_RUNTIME_INTEGRATION",
    "STRATEGY_DL_SOURCES",
    "STRATEGY_PARAM_SOURCES",
    "get_strategy_comparison_profiles",
    "get_strategy_comparison_menu_profiles",
    "get_strategy_multi_seed_robustness_profiles",
    "get_strategy_multi_seed_robustness_settings",
    "get_strategy_runtime_integration_settings",
    "get_strategy_comparison_settings",
]

"""Strategy Compare scientific/catalog SSOT.

This module owns stable/current comparison identities and catalog relationships.
It is not a user settings surface and contains no runtime resolver functions.
"""

from __future__ import annotations

from config.breakout_quality import (
    BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT,
)
from core.breakout_quality_policy import (
    get_breakout_quality_rolling_test_mode,
)
from config.strategy_compare import (
    STRATEGY_COMPARE_GPU_TRAIN_WORKERS,
    STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE,
    STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS,
    STRATEGY_COMPARE_ROBUSTNESS_KEEP_ATTRIBUTION_SOURCE,
    STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS,
    STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS,
    STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES,
    STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED,
    STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT,
    STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS,
)
from config.training_policy import (
    ROBUSTNESS_BENCHMARK_ID,
    ROBUSTNESS_BENCHMARK_SEED_COUNT,
    ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED,
)

STRATEGY_COMPARE_SCHEMA_VERSION = 67

_ROLLING_OOS = get_breakout_quality_rolling_test_mode("oos")
_ROLLING_ROLLING = get_breakout_quality_rolling_test_mode("rolling")
STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT = (
    BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT
)

# Nested OOS/Rolling bindings are derived catalog data, not user knobs.
STRATEGY_COMPARE_ROLLING_TEST_MODES = (
    {
        "mode_id": _ROLLING_OOS.mode_id,
        "label": _ROLLING_OOS.label,
        "score_start_date": _ROLLING_OOS.score_start_date,
        "score_end_date": _ROLLING_OOS.score_end_date,
        "fold_months": int(_ROLLING_OOS.fold_months),
        "fold_anchor_date": _ROLLING_OOS.fold_anchor_date,
        "single_score_block": bool(_ROLLING_OOS.single_score_block),
        "profile_id": "extending_window_oos",
        "robustness_id": "extending_window_oos",
    },
    {
        "mode_id": _ROLLING_ROLLING.mode_id,
        "label": _ROLLING_ROLLING.label,
        "score_start_date": _ROLLING_ROLLING.score_start_date,
        "score_end_date": _ROLLING_ROLLING.score_end_date,
        "fold_months": int(_ROLLING_ROLLING.fold_months),
        "fold_anchor_date": _ROLLING_ROLLING.fold_anchor_date,
        "single_score_block": bool(_ROLLING_ROLLING.single_score_block),
        "profile_id": "extending_window_rolling",
        "robustness_id": "extending_window_rolling",
    },
)

# Strategy arm的scientific base name只描述策略本體；current evaluation mode的
# OOS／Rolling suffix由profile renderer衍生，避免同一arm在各mode各自維護顯示名稱。
STRATEGY_COMPARE_DISPLAY_FULL_ROOS = "Full ROOS"
STRATEGY_COMPARE_DISPLAY_MIN_ROOS = "Min ROOS"
STRATEGY_COMPARE_DISPLAY_FULL_FINALISTS_AGREE = "Full Base-Finalists-Agree"
STRATEGY_COMPARE_DISPLAY_MIN_FINALISTS_AGREE = "Min Base-Finalists-Agree"
STRATEGY_COMPARE_DISPLAY_MIN_MR13E_SCORE_CONSTRAINED = "Min MR-13E Constrained"
STRATEGY_COMPARE_DISPLAY_MIN_MR13E_K_FLEX_R0 = "Min MR-13E K-Flex R0 Exact (Compute-Blocked)"
STRATEGY_COMPARE_DISPLAY_MIN_MR13E_FEASIBLE_ASCENT_CONTROL = "Min MR-13E Fixed-K Feasible-Ascent"
STRATEGY_COMPARE_DISPLAY_MIN_MR13E_K_FLEX_R0_FEASIBLE_ASCENT = "Min MR-13E K-Flex R0 Feasible-Ascent"
STRATEGY_COMPARE_DISPLAY_MIN_MR13K_SCORE_CONSTRAINED = "Min MR-13K Constrained"
STRATEGY_COMPARE_DISPLAY_MIN_MR13K_MR13M_SAFETY_CONSTRAINED = "Min MR-13K + MR-13M Safety"
STRATEGY_COMPARE_DISPLAY_MIN_MR13K_MR13M_RESIDUAL_SAFETY_CONSTRAINED = "Min MR-13K + MR-13M Residual Safety"
STRATEGY_COMPARE_DISPLAY_MIN_MR13P_CONDITIONAL_SAFETY_CONSTRAINED = "Min MR-13P Conditional Safety"
STRATEGY_COMPARE_DISPLAY_MIN_MR13Q_CONDITIONAL_MFE_SINGLE = "Min MR-13Q Conditional-MFE Single"
STRATEGY_COMPARE_DISPLAY_MIN_MR13R_CONDITIONAL_MFE_DUO = "Min MR-13R Conditional-MFE Duo"
STRATEGY_COMPARE_DISPLAY_MIN_MR13R_CONDITIONAL_MFE_RAW_SAFETY_GATE_50_NO_K_NO_R0 = "Min MR-13R Conditional-MFE Raw-Safety P50 No-K No-R0"
STRATEGY_COMPARE_DISPLAY_MIN_MR13R_CONDITIONAL_MFE_RAW_SAFETY_GATE_60_NO_K_NO_R0 = "Min MR-13R Conditional-MFE Raw-Safety P60 No-K No-R0"
STRATEGY_COMPARE_DISPLAY_MIN_MR13R_CONDITIONAL_MFE_RAW_SAFETY_GATE_70_NO_K_NO_R0 = "Min MR-13R Conditional-MFE Raw-Safety P70 No-K No-R0"
STRATEGY_COMPARE_DISPLAY_MIN_MR13R_SAFETY_MFE_PRODUCT_NO_K_NO_R0 = "Min MR-13R Raw-Safety × Conditional-MFE Product No-K No-R0"
STRATEGY_COMPARE_DISPLAY_MIN_MR13Z_JOINT_MIN_NO_K_NO_R0 = "Min MR-13Z Joint-Min No-K No-R0"
STRATEGY_COMPARE_DISPLAY_MIN_MR13H_NO_K_NO_R0 = "Min MR-13H No-K No-R0"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AC_NO_K_NO_R0 = "Min MR-13AC No-K No-R0"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AC_SCORE_CONSTRAINED = "Min MR-13AC Constrained"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AH_SCORE_CONSTRAINED = "Min MR-13AH Constrained"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AH_NO_K_NO_R0 = "Min MR-13AH No-K No-R0"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AK_SCORE_CONSTRAINED = "Min MR-13AK Constrained"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AK_NO_K_NO_R0 = "Min MR-13AK No-K No-R0"

# Current Compare Suite是「比較誰／比較哪些差」的唯一真理來源。
# OOS／Rolling／single-seed／multi-seed都只能引用suite，不得各自再列current arm matrix。
STRATEGY_COMPARE_SUITES = {
    "extending_current": {
        "arm_ids": ("C61", "C58", "C59", "C77", "C78", "C79", "C80", "C81", "C82", "C83"),
        "contrast_ids": (
            "C61-C58",
            "C59-C58",
            "C77-C58",
            "C78-C58", "C78-C77",
            "C79-C58", "C79-C59", "C79-C78",
            "C80-C58", "C80-C59", "C80-C79",
            "C81-C58", "C81-C77", "C81-C78",
            "C80-C81",
            "C82-C80", "C83-C81", "C82-C83",
        ),
        "display_name_bases": {
            "C61": "Full Base-Finalist-Best",
            "C58": "Min Base-Finalist-Best",
            "C59": "Min MR-13E Constrained",
            "C77": "Min MR-13H No-K No-R0",
            "C78": "Min MR-13AC No-K No-R0",
            "C79": "Min MR-13AC Constrained",
            "C80": "Min MR-13AH Constrained",
            "C81": "Min MR-13AH No-K No-R0",
            "C82": "Min MR-13AK Constrained",
            "C83": "Min MR-13AK No-K No-R0",
        },
    },
}


# Strategy Compare以研究階段profile隔離設定與輸出；App只顯示泛化階段名稱，
# arms／contrasts／period／output namespace全部由本檔驅動。
STRATEGY_COMPARE_PROFILES = {
    "pre_test": {
        "label": "Pre-Test 策略比較",
        "description": (
            "單模型快速研究Gate：沿用原本full-Selection refit後的2021+ continuous-ranker OOS scores，"
            "固定比較Full ROOS、Min ROOS、MR-13E reference與目前B2 candidate。只決定是否值得進正式Rolling，"
            "不得取代2021→latest的12M Extending-Window Rolling evidence。"
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
    "extending_window_oos": {
        "label": "Extending-Window Test | OOS Test",
        "description": (
            "固定information cutoff：只使用2021-01-01以前且Target已成熟的合法歷史做inner validation／epoch selection／final refit，"
            "只訓練一次並評分2021-01-01起至最新合法score date。OOS Test只作快速Gate，不取代12M Rolling evidence。"
        ),
        "display_alignment_group": "extending_strategy_compare",
        "suite_id": "extending_current",
        "display_suffix": "OOS",
        # None = 由三個PIT runtime工件的共同coverage動態解析2021→最新。
        "start_date": None,
        "end_date": None,
        "point_in_time_score_start_date": _ROLLING_OOS.score_start_date,
        "point_in_time_score_end_date": _ROLLING_OOS.score_end_date,
        "point_in_time_fold_months": int(_ROLLING_OOS.fold_months),
        "point_in_time_fold_anchor_date": _ROLLING_OOS.fold_anchor_date,
        "point_in_time_single_score_block": bool(_ROLLING_OOS.single_score_block),
        "point_in_time_dirname": _ROLLING_OOS.point_in_time_dirname,
        "output_root": "outputs/strategy_compare/extending_window/oos_2021_forward",
        "reuse_output_roots": (),
        "arm_param_source_overrides": {
            "C61": "full_oos",
            "C58": "min_oos",
            "C59": "min_oos",
            "C77": "min_oos",
            "C78": "min_oos",
            "C79": "min_oos",
            "C80": "min_oos",
            "C81": "min_oos",
            "C82": "min_oos",
            "C83": "min_oos",
        },
    },
    "extending_window_rolling": {
        "label": "Extending-Window Test | Rolling Test",
        "description": (
            "2021→latest的12M PIT-safe operational chain；DL使用expanding history + annual refit；"
            "Full／Min都沿用各時期當時合法rolling params。與OOS Test使用相同評估期間，"
            "兩者主要差異只保留是否按年度持續refit。"
        ),
        "display_alignment_group": "extending_strategy_compare",
        "suite_id": "extending_current",
        "display_suffix": "Rolling",
        # None = 由三個Rolling PIT runtime工件的共同coverage動態解析2021→latest。
        "start_date": None,
        "end_date": None,
        "point_in_time_score_start_date": _ROLLING_ROLLING.score_start_date,
        "point_in_time_score_end_date": _ROLLING_ROLLING.score_end_date,
        "point_in_time_fold_months": int(_ROLLING_ROLLING.fold_months),
        "point_in_time_fold_anchor_date": _ROLLING_ROLLING.fold_anchor_date,
        "point_in_time_single_score_block": bool(_ROLLING_ROLLING.single_score_block),
        "point_in_time_dirname": _ROLLING_ROLLING.point_in_time_dirname,
        "output_root": "outputs/strategy_compare/extending_window/rolling_2021_forward",
        # 舊2016～2025 aggregate保留historical-only；period不同時不得誤REUSE。
        "reuse_output_roots": ("outputs/strategy_compare/extending_window_rolling",),
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
# Current OOS／Rolling robustness只引用Compare Suite；所有suite arms與contrasts都逐seed重複，
# 僅model-seed-sensitive子集由arm的DL dependency推導，不得再於profile重複維護比較矩陣。
# Historical robustness profiles保留顯式membership供artifact compatibility；orchestration不得改動
# 單次Strategy Compare arm identity/fingerprint。
STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES = {
    "extending_window_oos": {
        "label": "Extending-Window Multi-seed Robustness Test | OOS Test",
        "enabled": True,
        "profile_id": "extending_window_oos",
        "suite_id": "extending_current",
        "benchmark_id": ROBUSTNESS_BENCHMARK_ID,
        "gpu_train_workers": STRATEGY_COMPARE_GPU_TRAIN_WORKERS,
        "cpu_replay_workers": STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS,
        "reuse_completed": STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED,
        "console_mode": STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE,
        "progress_interval_seconds": STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS,
        "yearly_report": STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT,
        "keep_checkpoints": STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS,
        "keep_scores": STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES,
        "keep_replay_details": STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS,
        "keep_attribution_source": STRATEGY_COMPARE_ROBUSTNESS_KEEP_ATTRIBUTION_SOURCE,
        "romd_reference_baselines": {
            "min": {"param_source": "min_oos", "param_policy": "base-finalist-best", "rule_policy": "all_off"},
            "full": {"param_source": "full_oos", "param_policy": "base-finalist-best", "rule_policy": "formal"},
        },
        "output_root": "outputs/strategy_compare/robustness/extending_window/oos_2021_forward",
        "model_work_root": "models/research/breakout_quality/strategy_compare/multi_seed_robustness/extending_window/oos_2021_forward",
        "checkpoint_cache_root": STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT,
    },
    "extending_window_rolling": {
        "label": "Extending-Window Multi-seed Robustness Test | Rolling Test",
        "enabled": True,
        "profile_id": "extending_window_rolling",
        "suite_id": "extending_current",
        "benchmark_id": ROBUSTNESS_BENCHMARK_ID,
        "gpu_train_workers": STRATEGY_COMPARE_GPU_TRAIN_WORKERS,
        "cpu_replay_workers": STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS,
        "reuse_completed": STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED,
        "console_mode": STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE,
        "progress_interval_seconds": STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS,
        "yearly_report": STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT,
        "keep_checkpoints": STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS,
        "keep_scores": STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES,
        "keep_replay_details": STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS,
        "keep_attribution_source": STRATEGY_COMPARE_ROBUSTNESS_KEEP_ATTRIBUTION_SOURCE,
        "romd_reference_baselines": {
            "min": {"param_source": "min_rolling", "param_policy": "base-finalist-best", "rule_policy": "all_off"},
            "full": {"param_source": "full_rolling", "param_policy": "base-finalist-best", "rule_policy": "formal"},
        },
        "output_root": "outputs/strategy_compare/robustness/extending_window/rolling_2021_forward",
        "model_work_root": "models/research/breakout_quality/strategy_compare/multi_seed_robustness/extending_window/rolling_2021_forward",
        "checkpoint_cache_root": STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT,
    },
    "selection_pit": {
        "label": "Selection PIT Multi-seed robustness (Legacy)",
        "enabled": False,
        "profile_id": "selection_pit",
        "seed_count": ROBUSTNESS_BENCHMARK_SEED_COUNT,
        "seed_generator_seed": ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED,
        "gpu_train_workers": STRATEGY_COMPARE_GPU_TRAIN_WORKERS,
        "cpu_replay_workers": STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS,
        "reuse_completed": STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED,
        "console_mode": STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE,
        "progress_interval_seconds": STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS,
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
        "seed_count": ROBUSTNESS_BENCHMARK_SEED_COUNT,
        "seed_generator_seed": ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED,
        "gpu_train_workers": STRATEGY_COMPARE_GPU_TRAIN_WORKERS,
        "cpu_replay_workers": STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS,
        "reuse_completed": STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED,
        "console_mode": STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE,
        "progress_interval_seconds": STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS,
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


# =============================================================================
# 3. 策略參數工件來源
# =============================================================================

STRATEGY_PARAM_SOURCES = {
    # Current strategy parameter SSOT.  Optimizer is the only producer; Research/
    # Strategy Compare only resolves/delegates these canonical identities; legacy migration is explicit one-time maintenance.
    "full_oos": {
        "path_template": None,
        "description": "Canonical Optimizer Full OOS strategy params (2020 cutoff, 2021→latest).",
        "identity_manifest_path": None,
        "trained_with_dl_id": None,
        "canonical_family": "full",
        "canonical_evaluation_mode": "oos",
        "artifact_contract": None,
        "builder": {"enabled": True, "builder_type": "canonical_optimizer_strategy_params", "options": {}},
    },
    "min_oos": {
        "path_template": None,
        "description": "Canonical Optimizer Min OOS strategy params; Research不得另建seed/trial policy。",
        "identity_manifest_path": None,
        "trained_with_dl_id": None,
        "canonical_family": "min",
        "canonical_evaluation_mode": "oos",
        "artifact_contract": None,
        "builder": {"enabled": True, "builder_type": "canonical_optimizer_strategy_params", "options": {}},
    },
    "full_rolling": {
        "path_template": None,
        "description": "Canonical Optimizer Full Rolling strategy params (2021→latest / 12M).",
        "identity_manifest_path": None,
        "trained_with_dl_id": None,
        "canonical_family": "full",
        "canonical_evaluation_mode": "rolling",
        "artifact_contract": None,
        "builder": {"enabled": True, "builder_type": "canonical_optimizer_strategy_params", "options": {}},
    },
    "min_rolling": {
        "path_template": None,
        "description": "Canonical Optimizer Min Rolling strategy params (2021→latest / 12M).",
        "identity_manifest_path": None,
        "trained_with_dl_id": None,
        "canonical_family": "min",
        "canonical_evaluation_mode": "rolling",
        "artifact_contract": None,
        "builder": {"enabled": True, "builder_type": "canonical_optimizer_strategy_params", "options": {}},
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
        "description": "MR-13E current OOS/Rolling PIT-safe score；C59保留exact K/R0 constrained reference，供C79作DL-source-only controlled comparison。",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13H_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": "MR-13H current OOS/Rolling PIT-safe full-horizon economic score；保留C77 historical/current direct-result comparison，不代表promotion。",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13AC_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_predicted_upside_context_v1",
        "experiment_profile": "daily_universal_predicted_upside_conditional_low_adverse_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-13AC current OOS/Rolling PIT-safe conditional Low-Adverse score；"
            "Stage-1 predicted-upside context仍完全沿用MR-13AC既定cross-fit + fixed-pre-OOS contract，"
            "current Rolling只對Stage-2 ranker依fold合法refit，不重定義Stage-1 scientific identity。"
        ),
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13AH_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_predicted_safety_product_weighted_pure_mfe_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": (
            "MR-13AH user-authorized Seed42 OOS/Rolling conversion source；"
            "Stage-1 canonical PIT-safe predicted-Safety context不變，Stage-2依evaluation mode合法refit。"
            "Model Gate仍維持FAIL；此source只授權controlled strategy conversion，不代表promotion。"
        ),
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13AK_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_shared_safety_mfe_v1",
        "experiment_profile": "daily_universal_shared_safety_weighted_pure_mfe_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": (
            "MR-13AK user-authorized Seed42 OOS/Rolling strategy conversion source；"
            "shared Safety/MFE model fitting與score semantics完全沿用MR-13AK canonical workflow。"
            "本source只把既有AK score綁入C82/C83 portfolio conversion，不代表production promotion。"
        ),
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
}

# =============================================================================
# 5. Strategy arm definitions
# =============================================================================
# Historical profile的arm集合仍可由profile-local arm_ids指定；current Extending profiles只引用Compare Suite。
# Arm definition本身不保存第二份enabled/mode membership；current matrix不得在profile或robustness重複列出。

STRATEGY_COMPARE_ARMS = {
    "C61": {
        "name": STRATEGY_COMPARE_DISPLAY_FULL_ROOS,
        "description": (
            "Extending-Window Full baseline；人讀名稱由evaluation mode渲染為Full OOS／Full Rolling；參數來源由mode綁定："
            "OOS固定2020 cutoff，Rolling使用Optimizer-owned canonical schedule；formal rules；DL-off"
        ),
        "param_source": "full_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C58": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_ROOS,
        "description": "Extending-Window Min baseline；Min base-finalist-best、all-off、DL-off；current DL strategy arms的同源Min baseline。",
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C59": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13E_SCORE_CONSTRAINED,
        "description": "既有MR-13E exact K/R0 constrained reference；C79完整複製其portfolio contract，只把DL source換成MR-13AC，因此C79-C59是DL-source-only controlled comparison。",
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
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
    "C77": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13H_NO_K_NO_R0,
        "description": (
            "MR-13H single-head full-horizon economic score direct conversion；沿用已完成的No-K/No-R0 Min contract，"
            "保留作single-head economic historical/current reference，不因C79新增而改變既有結果。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13H_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False,
            "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C78": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13AC_NO_K_NO_R0,
        "description": (
            "MR-13AC conditional Low-Adverse residual direct diagnostic；與C77沿用相同No-K/No-R0 Min direct contract，"
            "唯一scientific change是DL source換成MR-13AC。AC score沒有upside reward，因此此arm只回答standalone economic conversion，"
            "即使績效FAIL也不得反向否定MR-13AC Model Gate。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13AC_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False,
            "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C79": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13AC_SCORE_CONSTRAINED,
        "description": (
            "MR-13AC controlled replacement of C59：完整沿用C59的Min base-finalist-best、all-off、K/R0、"
            "resource-aware-continuous-score-constrained-optimal與exact_branch_and_bound_v1；"
            "唯一scientific change是DL source由CONT13E_ROLL換成CONT13AC_ROLL。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13AC_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C80": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13AH_SCORE_CONSTRAINED,
        "description": (
            "MR-13AH constrained conversion：逐欄沿用C59的Min base-finalist-best、all-off、K/R0、"
            "resource-aware-continuous-score-constrained-optimal與exact_branch_and_bound_v1；"
            "唯一scientific change是DL source由CONT13E_ROLL換成CONT13AH_ROLL。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13AH_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C81": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13AH_NO_K_NO_R0,
        "description": (
            "MR-13AH direct conversion：逐欄沿用C78的No-K/No-R0 Min direct allocator，"
            "唯一scientific change是DL source由CONT13AC_ROLL換成CONT13AH_ROLL；"
            "不加Safety gate、score fusion或其他runtime treatment。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13AH_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False,
            "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C82": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13AK_SCORE_CONSTRAINED,
        "description": (
            "MR-13AK constrained strategy conversion：逐欄沿用C80的Min base-finalist-best、all-off、K/R0、"
            "resource-aware-continuous-score-constrained-optimal與exact_branch_and_bound_v1；"
            "唯一scientific change是DL source由CONT13AH_ROLL換成CONT13AK_ROLL。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13AK_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C83": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13AK_NO_K_NO_R0,
        "description": (
            "MR-13AK direct strategy conversion：逐欄沿用C81的No-K/No-R0 Min direct allocator；"
            "唯一scientific change是DL source由CONT13AH_ROLL換成CONT13AK_ROLL；"
            "不加Safety gate、score fusion、threshold或其他runtime treatment。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13AK_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False,
            "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
}

# =============================================================================
# 6. Contrast definitions
# =============================================================================
# Contrast 是否啟用只由 STRATEGY_COMPARE_PROFILES[*]["contrast_ids"] 決定。

STRATEGY_COMPARE_CONTRASTS = {
    "C61-C58": {"left": "C61", "right": "C58", "description": "Full相對Min的整體策略體系reference；不是單一DL效果"},
    "C59-C58": {"left": "C59", "right": "C58", "description": "既有MR-13E exact K/R0 constrained相對Min baseline"},
    "C77-C58": {"left": "C77", "right": "C58", "description": "MR-13H No-K/No-R0 single-head economic score相對Min baseline"},
    "C78-C58": {"left": "C78", "right": "C58", "description": "MR-13AC No-K/No-R0 conditional-safety standalone diagnostic相對Min baseline"},
    "C78-C77": {"left": "C78", "right": "C77", "description": "同一No-K/No-R0 direct contract下conditional safety residual相對MR-13H single-head economic score"},
    "C79-C58": {"left": "C79", "right": "C58", "description": "MR-13AC exact K/R0 constrained相對Min DL-off baseline"},
    "C79-C59": {"left": "C79", "right": "C59", "description": "C59 portfolio contract完全不變，只把DL source由MR-13E換成MR-13AC；primary DL-source-only controlled contrast"},
    "C79-C78": {"left": "C79", "right": "C78", "description": "同一MR-13AC source下 exact K/R0 constrained相對No-K/No-R0 direct allocator；allocator/resource secondary diagnostic"},
    "C80-C58": {"left": "C80", "right": "C58", "description": "MR-13AH exact K/R0 constrained相對Min DL-off baseline"},
    "C80-C59": {"left": "C80", "right": "C59", "description": "C59 portfolio contract完全不變，只把DL source由MR-13E換成MR-13AH；AH constrained primary controlled contrast"},
    "C80-C79": {"left": "C80", "right": "C79", "description": "同一exact K/R0 constrained contract下MR-13AH相對MR-13AC；upside-safety score geometry comparison"},
    "C81-C58": {"left": "C81", "right": "C58", "description": "MR-13AH No-K/No-R0 direct conversion相對Min DL-off baseline"},
    "C81-C77": {"left": "C81", "right": "C77", "description": "同一No-K/No-R0 direct contract下MR-13AH相對MR-13H economic score"},
    "C81-C78": {"left": "C81", "right": "C78", "description": "C78 direct allocator contract完全不變，只把DL source由MR-13AC換成MR-13AH；AH direct primary controlled contrast"},
    "C80-C81": {"left": "C80", "right": "C81", "description": "同一MR-13AH source下 exact K/R0 constrained相對No-K/No-R0 direct allocator；allocator/resource diagnostic"},
    "C82-C80": {"left": "C82", "right": "C80", "description": "C80 constrained allocator contract完全不變，只把DL source由MR-13AH換成MR-13AK；AK constrained primary controlled contrast"},
    "C83-C81": {"left": "C83", "right": "C81", "description": "C81 No-K/No-R0 direct allocator contract完全不變，只把DL source由MR-13AH換成MR-13AK；AK direct primary controlled contrast"},
    "C82-C83": {"left": "C82", "right": "C83", "description": "同一MR-13AK source下 exact K/R0 constrained相對No-K/No-R0 direct allocator；allocator/resource secondary diagnostic"},
}


__all__ = [
    "STRATEGY_COMPARE_SCHEMA_VERSION",
    "STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT",
    "STRATEGY_COMPARE_ROLLING_TEST_MODES",
    "STRATEGY_COMPARE_SUITES",
    "STRATEGY_COMPARE_PROFILES",
    "STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES",
    "STRATEGY_PARAM_SOURCES",
    "STRATEGY_DL_SOURCES",
    "STRATEGY_COMPARE_ARMS",
    "STRATEGY_COMPARE_CONTRASTS",
]

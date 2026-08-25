"""目前正式 Strategy Compare 設定。

Active framework以Extending-Window Rolling作唯一策略績效主線：expanding history、
每個PIT fold重新選epoch/refit，再評分下一段。舊Selection PIT／Frozen Forward
profiles保留作歷史工件解讀／重現，但不再暴露於主選單或作current Gate。
Fixed-Window Rolling屬模型穩定性診斷，不建立另一套production strategy truth。
"""

from __future__ import annotations

from dataclasses import replace

from config.breakout_quality import (
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    get_breakout_quality_rolling_test_mode,
    get_breakout_quality_workflow_settings,
)
from config.research import get_research_artifact_preparation_policy
from config.compatibility.strategy_compare_history import (
    HISTORICAL_STRATEGY_COMPARE_ARMS,
    HISTORICAL_STRATEGY_COMPARE_CONTRASTS,
    HISTORICAL_STRATEGY_DL_SOURCES,
    HISTORICAL_STRATEGY_PARAM_SOURCES,
)
from config.training_policy import (
    ROBUSTNESS_BENCHMARK_ID,
    ROBUSTNESS_BENCHMARK_RESOLVED_SEEDS,
    ROBUSTNESS_BENCHMARK_SEED_COUNT,
    ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED,
    get_strategy_parameter_training_policy_snapshot,
    resolve_robustness_benchmark_seeds,
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
    resolve_strategy_comparison_arm_param_policy,
    validate_strategy_comparison_settings,
    validate_strategy_compare_gpu_train_workers,
    validate_strategy_multi_seed_robustness_settings,
    validate_strategy_runtime_integration_settings,
)

STRATEGY_COMPARE_SCHEMA_VERSION = 57

# =============================================================================
# 1. 常用設定
#    一般 Strategy Compare / robustness 實驗通常只需修改本區。
#    Dataset / param policy / max positions / rotation 不在此複製；它們直接
#    繼承 config/breakout_quality.py 的 BreakoutQualityWorkflowSettings SSOT。
# =============================================================================

_ROLLING_OOS = get_breakout_quality_rolling_test_mode("oos")
_ROLLING_ROLLING = get_breakout_quality_rolling_test_mode("rolling")

STRATEGY_COMPARE_DEFAULT_PROFILE = "extending_window_oos"
# Current UI只顯示研究問題；OOS／Rolling在下一層選單選擇。
# 此tuple仍列出兩個current internal profiles，供status／artifact resolver使用。
STRATEGY_COMPARE_MENU_PROFILE_IDS = (
    "extending_window_oos",
    "extending_window_rolling",
)
STRATEGY_COMPARE_ROLLING_TEST_MENU_LABEL = "Extending-Window Test"
STRATEGY_COMPARE_ROBUSTNESS_MENU_LABEL = "Extending-Window Multi-seed Robustness Test"
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
STRATEGY_COMPARE_DEFAULT_ROBUSTNESS_PROFILE = "extending_window_oos"
STRATEGY_COMPARE_GPU_TRAIN_WORKERS = 1
validate_strategy_compare_gpu_train_workers(STRATEGY_COMPARE_GPU_TRAIN_WORKERS)
STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS = 60.0
STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT = (
    "models/research/breakout_quality/strategy_compare/"
    "extending_window/shared_fitting_checkpoints"
)
STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS = 1
STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED = True
STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE = "compact"
STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT = True
STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS = False
STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES = False
STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS = False
STRATEGY_COMPARE_ROBUSTNESS_KEEP_ATTRIBUTION_SOURCE = True

# Strategy Compare read-only path-conversion diagnostics. These settings affect only
# post-replay attribution/reporting; changing them must never invalidate portfolio replay.
STRATEGY_COMPARE_UPSIDE_REALIZATION_R_THRESHOLDS = (1.0, 2.0, 3.0)
STRATEGY_COMPARE_UPSIDE_REALIZATION_ADVERSE_BUCKET_EDGES_R = (0.5, 1.0)
# Canonical profile owner for the path components consumed by Upside/First-Passage
# diagnostics.  Other research profiles may legitimately share the same primary target
# (for example MR-13P), so target-id uniqueness must not be used as ownership.
STRATEGY_COMPARE_UPSIDE_REALIZATION_PATH_PROFILE = (
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
)

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

# Current Compare Suite是「比較誰／比較哪些差」的唯一真理來源。
# OOS／Rolling／single-seed／multi-seed都只能引用suite，不得各自再列current arm matrix。
STRATEGY_COMPARE_SUITES = {
    "extending_current": {
        "arm_ids": ("C61", "C62", "C58", "C63", "C59", "C68", "C69", "C60", "C64", "C65", "C66"),
        "contrast_ids": (
            "C62-C61", "C63-C58", "C62-C63",
            "C61-C58", "C59-C58", "C59-C61",
            "C68-C59", "C69-C68", "C69-C59", "C69-C58",
            "C60-C58", "C60-C61", "C60-C59",
            "C64-C58", "C64-C59", "C64-C60",
            "C65-C58", "C65-C59", "C65-C64",
            "C66-C58", "C66-C59", "C66-C64", "C66-C65",
        ),
        "display_name_bases": {
            "C61": "Full Base-Finalist-Best",
            "C62": "Full Base-Finalists-Agree",
            "C58": "Min Base-Finalist-Best",
            "C63": "Min Base-Finalists-Agree",
            "C59": "Min MR-13E Constrained",
            "C68": "Min MR-13E Fixed-K Feasible-Ascent",
            "C69": "Min MR-13E K-Flex R0 Feasible-Ascent",
            "C60": "Min MR-13K + MR-13M Residual Safety",
            "C64": "Min MR-13P Conditional Safety",
            "C65": "Min MR-13Q Conditional-MFE Single",
            "C66": "Min MR-13R Conditional-MFE Duo",
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
            "C62": "full_oos",
            "C58": "min_oos",
            "C63": "min_oos",
            "C59": "min_oos",
            "C68": "min_oos",
            "C69": "min_oos",
            "C60": "min_oos",
            "C64": "min_oos",
            "C65": "min_oos",
            "C66": "min_oos",
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

_RESEARCH_PREPARATION = get_research_artifact_preparation_policy()
STRATEGY_COMPARE_PREPARATION = {
    "auto_prepare": bool(_RESEARCH_PREPARATION.auto_prepare),
    "reuse_ready_artifacts": bool(_RESEARCH_PREPARATION.reuse_ready_artifacts),
    "rebuild_stale_artifacts": bool(_RESEARCH_PREPARATION.rebuild_stale_artifacts),
    "resume_parameter_training": bool(_RESEARCH_PREPARATION.resume_partial_artifacts),
    "require_confirmation": bool(_RESEARCH_PREPARATION.require_single_confirmation),
    # 已完成且replay identity完全相同的pair直接重用歷史正式結果。
    "reuse_completed_results": True,
    # 同一param_source/rule_policy的新pair只執行一次DL-off baseline。
    "reuse_shared_baseline": True,
}

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
        "description": "MR-13E Extending-Window Rolling PIT-safe score；expanding history + 12M annual refit，current period=2021→latest。",
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
        "description": "MR-13K Extending-Window Rolling PIT-safe pure-MFE score；expanding history + mode-specific refit cadence。",
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


    "CONT13P_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_conditional_mfe_safety_v1",
        "experiment_profile": "daily_universal_conditional_mfe_safety_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": "MR-13P Extending-Window single-model dual-head PIT score；primary=Pure-MFE，secondary=learned Conditional Safety。",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13Q_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_conditional_mfe_single_head_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": "MR-13Q reverse-conditional MFE single-head PIT score；final score直接代表J=U-E(U|S)。",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13R_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_safety_conditional_mfe_v1",
        "experiment_profile": "daily_universal_safety_conditional_mfe_duo_head_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": "MR-13R Safety→Conditional-MFE duo-head PIT score；strategy只使用final Conditional-MFE head。",
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
            "Extending-Window Full baseline；人讀名稱由evaluation mode渲染為Full OOS／Full Rolling；參數來源由mode綁定：OOS固定2020 cutoff，Rolling使用Optimizer-owned canonical schedule；formal rules；DL-off"
        ),
        "param_source": "full_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C62": {
        "name": STRATEGY_COMPARE_DISPLAY_FULL_FINALISTS_AGREE,
        "description": (
            "Extending-Window Full DL-off parameter-policy reference；與C61完全相同formal rules／execution，"
            "唯一差異為canonical parameter policy使用base-finalists-agree。"
        ),
        "param_source": "full_rolling",
        "param_policy": "base-finalists-agree",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C58": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_ROOS,
        "description": "Extending-Window Min baseline；人讀名稱由evaluation mode渲染為Min OOS／Min Rolling；參數來源由mode綁定：OOS固定2020 cutoff，Rolling使用Optimizer-owned canonical schedule；rules全關；DL-off",
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C63": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_FINALISTS_AGREE,
        "description": (
            "Extending-Window Min DL-off parameter-policy reference；與C58完全相同all-off rules／execution，"
            "唯一差異為canonical parameter policy使用base-finalists-agree。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalists-agree",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C59": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13E_SCORE_CONSTRAINED,
        "description": "Extending-Window MR-13E exact constrained；模型與策略參數都依OOS／Rolling mode使用一致information-cutoff contract。",
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
    "C68": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13E_FEASIBLE_ASCENT_CONTROL,
        "description": (
            "SR-C68 fixed-K local-search control：與C59使用相同MR-13E score、Min base-finalist-best、"
            "all-off、baseline K/R0與canonical sizing/cash/orderability/execution；唯一差異為"
            "global exact solver改為既有deterministic best-improvement feasible-ascent。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-matched-feasible-ascent",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "membership_proposal": "score_priority_then_canonical_execution_v1",
            "local_search": "deterministic_single_swap_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C69": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13E_K_FLEX_R0_FEASIBLE_ASCENT,
        "description": (
            "SR-C69 matched K-Flex local-search treatment：與C68完全相同MR-13E source與"
            "deterministic feasible-ascent family，R0仍為hard floor；唯一scientific change是"
            "允許single-add把planned-order count由baseline K擴到physical free slots，"
            "count lexicographically優先，count內仍最大化原始MR-13E score。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-k-flex-r0-feasible-ascent",
        "dl_runtime_options": {
            "count_constraint": "baseline_k_to_physical_free_slots_local_v1",
            "preserve_r0": True,
            "membership_proposal": "score_priority_then_canonical_execution_v1",
            "local_search": "deterministic_single_add_swap_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C60": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13K_MR13M_RESIDUAL_SAFETY_CONSTRAINED,
        "description": (
            "Extending-Window B2：MR-13K primary + MR-13M same-day rank OLS residual-safety floor；"
            "兩個score必須來自同一evaluation mode／同一information-cutoff namespace。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
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
    "C64": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13P_CONDITIONAL_SAFETY_CONSTRAINED,
        "description": (
            "MR-13P single-model conversion Gate：Primary直接最大化Pure-MFE head；"
            "同一PIT artifact的conditional_safety_score直接作baseline-relative safety floor，"
            "training target已conditionalize，因此runtime不再做OLS residualization。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13P_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-safety-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
            "safety_dl_id": "CONT13P_ROLL",
            "safety_score_column": "conditional_safety_score",
            "safety_constraint": "baseline_coverage_and_score_sum_floor_v1",
        },
        "robustness_role": "off",
    },
    "C65": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13Q_CONDITIONAL_MFE_SINGLE,
        "description": (
            "MR-13Q Single-head reverse-Conditional-MFE conversion arm；與C58共用Min base-finalist-best、"
            "all-off、K/R0與canonical execution，唯一model change是直接最大化J=U-E(U|S) prediction。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13Q_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C66": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13R_CONDITIONAL_MFE_DUO,
        "description": (
            "MR-13R Duo-head Safety→Conditional-MFE conversion arm；與C65完全相同strategy contract，"
            "唯一差異是模型內顯式Raw Safety condition head；strategy仍只最大化final Conditional-MFE score。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13R_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
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
    "C62-C61": {"left": "C62", "right": "C61", "description": "{left}相對{right}的Full parameter-policy單變量差異"},
    "C63-C58": {"left": "C63", "right": "C58", "description": "{left}相對{right}的Min parameter-policy單變量差異"},
    "C62-C63": {"left": "C62", "right": "C63", "description": "base-finalists-agree政策下Full相對Min的完整策略體系差異"},
    "C61-C58": {"left": "C61", "right": "C58", "description": "{left}相對{right}的完整策略體系差異；不是單一參數效果"},
    "C59-C58": {"left": "C59", "right": "C58", "description": "{left}相對{right}的增量策略效果"},
    "C59-C61": {"left": "C59", "right": "C61", "description": "{left}相對{right}的整體策略結果；不是單一DL效果"},
    "C68-C59": {"left": "C68", "right": "C59", "description": "dedicated matched fixed-K feasible-ascent相對exact C59；隔離global exact→matched deterministic local-search solver effect"},
    "C69-C68": {"left": "C69", "right": "C68", "description": "同一feasible-ascent family下K-Flex相對fixed-K；唯一scientific change為baseline K可擴到physical free slots，R0保留"},
    "C69-C59": {"left": "C69", "right": "C59", "description": "practical K-Flex local treatment相對current exact-K C59的淨策略效果"},
    "C69-C58": {"left": "C69", "right": "C58", "description": "K-Flex / R0-preserved feasible-ascent MR-13E相對Min baseline整體效果"},
    "C60-C58": {"left": "C60", "right": "C58", "description": "{left}相對{right}的增量策略效果"},
    "C60-C61": {"left": "C60", "right": "C61", "description": "{left}相對{right}的整體策略結果；不是單一DL效果"},
    "C60-C59": {"left": "C60", "right": "C59", "description": "{left}相對{right}的同政策比較"},
    "C64-C58": {"left": "C64", "right": "C58", "description": "MR-13P conditional conversion相對Min baseline的增量策略效果"},
    "C64-C59": {"left": "C64", "right": "C59", "description": "MR-13P conditional conversion相對目前C59 MR-13E conversion reference"},
    "C64-C60": {"left": "C64", "right": "C60", "description": "model-level conditional safety相對C60 runtime post-hoc residual safety"},
    "C65-C58": {"left": "C65", "right": "C58", "description": "Single-head Conditional-MFE相對Min baseline的增量策略效果"},
    "C65-C59": {"left": "C65", "right": "C59", "description": "Single-head Conditional-MFE相對C59 conversion reference"},
    "C65-C64": {"left": "C65", "right": "C64", "description": "直接maximize Conditional-MFE相對MR-13P safety hard-floor formulation"},
    "C66-C58": {"left": "C66", "right": "C58", "description": "Duo-head Conditional-MFE相對Min baseline的增量策略效果"},
    "C66-C59": {"left": "C66", "right": "C59", "description": "Duo-head Conditional-MFE相對C59 conversion reference"},
    "C66-C64": {"left": "C66", "right": "C64", "description": "Duo-head Conditional-MFE相對MR-13P safety hard-floor formulation"},
    "C66-C65": {"left": "C66", "right": "C65", "description": "顯式Safety condition head相對Single-head的architecture controlled contrast"},
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


def _resolved_suite_id(profile: dict) -> str | None:
    value = str(profile.get("suite_id") or "").strip()
    return value or None


def _resolved_profile_matrix(profile: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    suite_id = _resolved_suite_id(profile)
    if suite_id is None:
        return (
            tuple(str(value) for value in profile.get("arm_ids", ())),
            tuple(str(value) for value in profile.get("contrast_ids", ())),
        )
    if suite_id not in STRATEGY_COMPARE_SUITES:
        raise ValueError(f"Strategy Compare profile引用不存在Compare Suite: {suite_id}")
    suite = get_strategy_compare_suite(suite_id)
    if profile.get("arm_ids") not in (None, (), []):
        raise ValueError(f"current suite profile不得另寫arm_ids: suite={suite_id}")
    if profile.get("contrast_ids") not in (None, (), []):
        raise ValueError(f"current suite profile不得另寫contrast_ids: suite={suite_id}")
    return (
        tuple(str(value) for value in suite.get("arm_ids", ())),
        tuple(str(value) for value in suite.get("contrast_ids", ())),
    )


def _arm_has_seed_sensitive_model_dependency(arm: StrategyComparisonArm) -> bool:
    return bool(arm.dl_enabled and str(arm.dl_id or "").strip())


def _derived_robustness_membership(
    profile_settings: StrategyComparisonSettings,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Derive current robustness roles from the shared Compare Suite.

    Current robustness is exactly the single-seed Compare Suite repeated for every
    benchmark seed.  Therefore every enabled suite arm is strategy-seed-sensitive;
    only arms with DL dependencies additionally retrain model sources with that seed.
    The first tuple is intentionally empty and exists only for the legacy fixed-arm
    compatibility shape used by disabled historical robustness profiles.
    """
    benchmark = tuple(arm.arm_id for arm in profile_settings.enabled_arms)
    model_sensitive = tuple(
        arm.arm_id
        for arm in profile_settings.enabled_arms
        if _arm_has_seed_sensitive_model_dependency(arm)
    )
    if not benchmark or not model_sensitive:
        raise ValueError(
            "Compare Suite end-to-end robustness角色不完整: "
            f"suite={profile_settings.suite_id}, benchmark={list(benchmark)}, "
            f"model_sensitive={list(model_sensitive)}"
        )
    return (), benchmark, model_sensitive


def _derived_stochastic_contrasts(
    profile_settings: StrategyComparisonSettings, stochastic_arm_ids: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    stochastic = set(stochastic_arm_ids)
    rows: list[dict[str, str]] = []
    for contrast in profile_settings.enabled_contrasts:
        if contrast.left in stochastic and contrast.right in stochastic:
            rows.append({
                "contrast_id": contrast.contrast_id,
                "left": contrast.left,
                "right": contrast.right,
                "description": contrast.description,
            })
    return tuple(rows)


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
        active_ids = set(_resolved_profile_matrix(dict(STRATEGY_COMPARE_PROFILES[profile_id]))[0])
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
        suite_id=(None if raw.get("suite_id") in (None, "") else str(raw.get("suite_id")).strip()),
        benchmark_id=(None if raw.get("benchmark_id") in (None, "") else str(raw.get("benchmark_id")).strip()),
        seed_count=int(raw.get("seed_count", ROBUSTNESS_BENCHMARK_SEED_COUNT) or 0),
        seed_generator_seed=int(raw.get("seed_generator_seed", ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED) or 0),
        resolved_seeds=tuple(
            int(value) for value in tuple(
                raw.get("resolved_seeds")
                or (ROBUSTNESS_BENCHMARK_RESOLVED_SEEDS if raw.get("benchmark_id") not in (None, "") else ())
                or ()
            )
        ),
        strategy_trials_per_fold=(
            int(get_strategy_parameter_training_policy_snapshot(evaluation_mode="rolling")["trials_per_fold"])
            if raw.get("benchmark_id") not in (None, "")
            else (
                None if raw.get("strategy_trials_per_fold") in (None, "")
                else int(raw.get("strategy_trials_per_fold"))
            )
        ),
        gpu_train_workers=int(raw.get("gpu_train_workers", STRATEGY_COMPARE_GPU_TRAIN_WORKERS)),
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
                "param_policy": str(dict(value or {}).get("param_policy") or "").strip(),
                "rule_policy": str(dict(value or {}).get("rule_policy") or "").strip(),
            }
            for key, value in dict(raw.get("romd_reference_baselines") or {}).items()
        },
        fixed_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("fixed_arm_ids") or ()) if str(value).strip()),
        stochastic_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("stochastic_arm_ids") or ()) if str(value).strip()),
        benchmark_strategy_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("benchmark_strategy_arm_ids") or raw.get("stochastic_arm_ids") or ()) if str(value).strip()),
        model_seed_sensitive_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("model_seed_sensitive_arm_ids") or raw.get("stochastic_arm_ids") or ()) if str(value).strip()),
        consensus_reference_arm_ids=tuple(str(value).strip() for value in tuple(raw.get("consensus_reference_arm_ids") or raw.get("fixed_arm_ids") or ()) if str(value).strip()),
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
        checkpoint_cache_root=(
            None
            if raw.get("checkpoint_cache_root") in (None, "")
            else str(raw.get("checkpoint_cache_root")).strip()
        ),
    )
    if settings.suite_id is None:
        if not settings.resolved_seeds:
            settings = replace(
                settings,
                resolved_seeds=resolve_robustness_benchmark_seeds(
                    seed_count=settings.seed_count, generator_seed=settings.seed_generator_seed
                ),
            )
        validate_strategy_multi_seed_robustness_settings(settings)
    if settings.profile_id not in STRATEGY_COMPARE_PROFILES:
        raise ValueError(
            f"multi-seed robustness引用不存在的Strategy Compare profile: {settings.profile_id}"
        )
    profile_settings = get_strategy_comparison_settings(settings.profile_id)
    if settings.suite_id is not None:
        if profile_settings.suite_id != settings.suite_id:
            raise ValueError(
                "multi-seed robustness suite與Strategy Compare profile不一致: "
                f"robustness={settings.suite_id}, profile={profile_settings.suite_id}"
            )
        fixed_ids, benchmark_ids, model_ids = _derived_robustness_membership(profile_settings)
        settings = replace(
            settings,
            fixed_arm_ids=fixed_ids,
            stochastic_arm_ids=benchmark_ids,
            benchmark_strategy_arm_ids=benchmark_ids,
            model_seed_sensitive_arm_ids=model_ids,
            consensus_reference_arm_ids=(),
            paired_contrasts=_derived_stochastic_contrasts(profile_settings, benchmark_ids),
        )
        validate_strategy_multi_seed_robustness_settings(settings)
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
    if len(score_sources) > 1:
        raise ValueError(
            "multi-seed stochastic DL arms必須使用同一score source: "
            f"actual={sorted(score_sources)}"
        )
    reference_pool = stochastic if settings.benchmark_id is not None else fixed
    reference_role = "same-seed benchmark baseline" if settings.benchmark_id is not None else "fixed baseline"
    for reference_key, spec in settings.romd_reference_baselines.items():
        expected_param_policy = str(
            spec.get("param_policy") or profile_settings.param_policy
        ).strip()
        matches = [
            arm for arm in reference_pool
            if arm.param_source == spec["param_source"]
            and resolve_strategy_comparison_arm_param_policy(profile_settings, arm)
            == expected_param_policy
            and arm.rule_policy == spec["rule_policy"]
            and not arm.dl_enabled
        ]
        if len(matches) != 1:
            raise ValueError(
                f"multi-seed RoMD reference必須唯一對應一個{reference_role}: "
                f"reference={reference_key}, param_source={spec['param_source']}, "
                f"param_policy={expected_param_policy}, rule_policy={spec['rule_policy']}, "
                f"matches={len(matches)}"
            )
    return settings


def get_strategy_rolling_test_modes() -> tuple[dict[str, object], ...]:
    """Return OOS/Rolling mode bindings for the nested Strategy Compare menus."""

    rows: list[dict[str, object]] = []
    seen_profiles: set[str] = set()
    seen_robustness: set[str] = set()
    for raw in STRATEGY_COMPARE_ROLLING_TEST_MODES:
        item = dict(raw)
        profile_id = str(item.get("profile_id") or "").strip()
        robustness_id = str(item.get("robustness_id") or "").strip()
        if profile_id not in STRATEGY_COMPARE_PROFILES:
            raise ValueError(f"Rolling Test mode引用不存在Strategy Compare profile: {profile_id!r}")
        if robustness_id not in STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES:
            raise ValueError(f"Rolling Test mode引用不存在robustness profile: {robustness_id!r}")
        if profile_id in seen_profiles or robustness_id in seen_robustness:
            raise ValueError("Rolling Test mode profile/robustness不可重複")
        rows.append(item)
        seen_profiles.add(profile_id)
        seen_robustness.add(robustness_id)
    if not rows:
        raise ValueError("Strategy Compare至少需要一個Rolling Test mode")
    current_suite_ids = {
        _resolved_suite_id(dict(STRATEGY_COMPARE_PROFILES[str(item["profile_id"])]))
        for item in rows
    }
    if None in current_suite_ids or len(current_suite_ids) != 1:
        raise ValueError(
            "current OOS／Rolling modes必須引用同一Compare Suite: "
            f"suite_ids={sorted(str(value) for value in current_suite_ids)}"
        )
    for item in rows:
        raw_robustness = dict(
            STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES[str(item["robustness_id"])]
        )
        if str(raw_robustness.get("suite_id") or "").strip() not in current_suite_ids:
            raise ValueError(
                "current robustness mode必須引用與single-seed相同Compare Suite: "
                f"robustness={item['robustness_id']}"
            )
    return tuple(rows)


def get_strategy_compare_suite(suite_id: str) -> dict[str, object]:
    selected = str(suite_id).strip()
    if selected not in STRATEGY_COMPARE_SUITES:
        raise ValueError(f"不存在的Strategy Compare suite: {selected}")
    raw = dict(STRATEGY_COMPARE_SUITES[selected])
    arm_ids = tuple(str(value) for value in raw.get("arm_ids", ()))
    contrast_ids = tuple(str(value) for value in raw.get("contrast_ids", ()))
    display_name_bases = {
        str(key): str(value).strip()
        for key, value in dict(raw.get("display_name_bases") or {}).items()
    }
    if not arm_ids or len(set(arm_ids)) != len(arm_ids):
        raise ValueError(f"Strategy Compare suite arm_ids不可空白／重複: {selected}")
    if len(set(contrast_ids)) != len(contrast_ids):
        raise ValueError(f"Strategy Compare suite contrast_ids不得重複: {selected}")
    if set(display_name_bases) != set(arm_ids):
        raise ValueError(
            f"Strategy Compare suite display_name_bases必須完整覆蓋arms: suite={selected}"
        )
    missing_arms = [arm_id for arm_id in arm_ids if arm_id not in STRATEGY_COMPARE_ARMS]
    missing_contrasts = [cid for cid in contrast_ids if cid not in STRATEGY_COMPARE_CONTRASTS]
    if missing_arms or missing_contrasts:
        raise ValueError(
            f"Strategy Compare suite引用不存在設定: arms={missing_arms}, contrasts={missing_contrasts}"
        )
    for cid in contrast_ids:
        spec = dict(STRATEGY_COMPARE_CONTRASTS[cid])
        if str(spec.get("left") or "") not in arm_ids or str(spec.get("right") or "") not in arm_ids:
            raise ValueError(f"Strategy Compare suite contrast端點不在suite: {selected}/{cid}")
    return {
        "suite_id": selected,
        "arm_ids": arm_ids,
        "contrast_ids": contrast_ids,
        "display_name_bases": display_name_bases,
    }


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
    profile_arm_ids, profile_contrast_ids = _resolved_profile_matrix(profile)
    profile_suite_id = _resolved_suite_id(profile)
    profile_suite_display_bases = (
        {} if profile_suite_id is None
        else {
            str(key): str(value).strip()
            for key, value in dict(get_strategy_compare_suite(profile_suite_id).get("display_name_bases") or {}).items()
        }
    )
    profile_display_suffix = (
        None if profile.get("display_suffix") in (None, "")
        else str(profile.get("display_suffix")).strip()
    )
    (
        parameter_catalog,
        dl_catalog,
        arm_catalog,
        contrast_catalog,
    ) = _compatibility_catalogs()
    missing_arms = [arm_id for arm_id in profile_arm_ids if arm_id not in arm_catalog]
    missing_contrasts = [
        contrast_id
        for contrast_id in profile_contrast_ids
        if contrast_id not in contrast_catalog
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
            canonical_family=(None if raw.get("canonical_family") in (None, "") else str(raw.get("canonical_family"))),
            canonical_evaluation_mode=(None if raw.get("canonical_evaluation_mode") in (None, "") else str(raw.get("canonical_evaluation_mode"))),
        )
        for source_id, raw in parameter_catalog.items()
    }
    profile_pit_score_start_date = (
        None
        if profile.get("point_in_time_score_start_date") in (None, "")
        else str(profile.get("point_in_time_score_start_date")).strip()
    )
    profile_pit_score_end_date = (
        None
        if profile.get("point_in_time_score_end_date") in (None, "")
        else str(profile.get("point_in_time_score_end_date")).strip()
    )
    profile_pit_single_score_block = bool(profile.get("point_in_time_single_score_block", False))
    profile_pit_fold_months = (
        None
        if profile.get("point_in_time_fold_months") in (None, "")
        else int(profile.get("point_in_time_fold_months"))
    )
    profile_pit_anchor_date = (
        None
        if profile.get("point_in_time_fold_anchor_date") in (None, "")
        else str(profile.get("point_in_time_fold_anchor_date")).strip()
    )
    profile_pit_dirname = (
        None
        if profile.get("point_in_time_dirname") in (None, "")
        else str(profile.get("point_in_time_dirname")).strip()
    )
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
            point_in_time_score_start_date=(
                profile_pit_score_start_date
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else None
            ),
            point_in_time_score_end_date=(
                profile_pit_score_end_date
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else None
            ),
            point_in_time_fold_months=(
                profile_pit_fold_months
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else None
            ),
            point_in_time_fold_anchor_date=(
                profile_pit_anchor_date
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else None
            ),
            point_in_time_single_score_block=(
                profile_pit_single_score_block
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else False
            ),
            point_in_time_dirname=(
                profile_pit_dirname
                if str(raw.get("score_source") or "canonical_runtime").strip() == "selection_point_in_time"
                else None
            ),
        )
        for dl_id, raw in dl_catalog.items()
    }
    profile_arm_param_source_overrides = {
        str(arm_id): str(source_id)
        for arm_id, source_id in dict(profile.get("arm_param_source_overrides") or {}).items()
    }
    unknown_param_override_arms = sorted(
        set(profile_arm_param_source_overrides) - set(arm_catalog)
    )
    unknown_param_override_sources = sorted(
        set(profile_arm_param_source_overrides.values()) - set(parameter_sources)
    )
    if unknown_param_override_arms or unknown_param_override_sources:
        raise ValueError(
            "Strategy Compare profile param-source override無效: "
            f"arms={unknown_param_override_arms or '-'}, "
            f"params={unknown_param_override_sources or '-'}"
        )
    ordered_arm_ids = tuple(dict.fromkeys((*profile_arm_ids, *arm_catalog.keys())))
    arms = {
        arm_id: StrategyComparisonArm(
            arm_id=arm_id,
            enabled=arm_id in profile_arm_ids,
            name=(
                f"{profile_suite_display_bases.get(arm_id, str(raw.get('name') or '').strip())} {profile_display_suffix}"
                if arm_id in profile_suite_display_bases and profile_display_suffix
                else str(raw.get("name") or "").strip()
            ),
            description=str(raw.get("description") or "").strip(),
            param_source=str(
                profile_arm_param_source_overrides.get(
                    arm_id, raw.get("param_source") or ""
                )
            ).strip(),
            param_policy=(
                None
                if raw.get("param_policy") in (None, "")
                else str(raw.get("param_policy")).strip()
            ),
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
            description=str(raw.get("description") or "").strip().format(
                left=arms[str(raw.get("left") or "").strip()].name,
                right=arms[str(raw.get("right") or "").strip()].name,
            ),
        )
        for contrast_id in ordered_contrast_ids
        for raw in (contrast_catalog[contrast_id],)
    }
    workflow_settings = get_breakout_quality_workflow_settings()
    settings = StrategyComparisonSettings(
        schema_version=int(STRATEGY_COMPARE_SCHEMA_VERSION),
        profile_id=selected_profile_id,
        profile_label=str(profile["label"]),
        suite_id=profile_suite_id,
        display_suffix=profile_display_suffix,
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
    "STRATEGY_COMPARE_SUITES",
    "STRATEGY_COMPARE_MENU_PROFILE_IDS",
    "STRATEGY_COMPARE_ROLLING_TEST_MENU_LABEL",
    "STRATEGY_COMPARE_ROBUSTNESS_MENU_LABEL",
    "STRATEGY_COMPARE_ROLLING_TEST_MODES",
    "STRATEGY_COMPARE_GPU_TRAIN_WORKERS",
    "STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT",
    "STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS",
    "STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES",
    "STRATEGY_RUNTIME_INTEGRATION",
    "STRATEGY_DL_SOURCES",
    "STRATEGY_PARAM_SOURCES",
    "get_strategy_compare_suite",
    "get_strategy_rolling_test_modes",
    "get_strategy_comparison_profiles",
    "get_strategy_comparison_menu_profiles",
    "get_strategy_multi_seed_robustness_profiles",
    "get_strategy_multi_seed_robustness_settings",
    "get_strategy_runtime_integration_settings",
    "get_strategy_comparison_settings",
]

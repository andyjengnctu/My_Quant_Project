"""Strategy Compare scientific/catalog SSOT.

This module owns stable/current comparison identities and catalog relationships.
It is not a user settings surface and contains no runtime resolver functions.
"""

from __future__ import annotations

from core.research_summary import get_research_summary_strategy_arm_ids

from config.breakout_quality import (
    BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT,
)
from core.breakout_quality_policy import (
    get_breakout_quality_rolling_test_mode,
)
from config.strategy_compare import (
    STRATEGY_COMPARE_CURRENT_OOS_PROFILE,
    STRATEGY_COMPARE_CURRENT_OOS_ROBUSTNESS,
    STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE,
    STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS,
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
from config.robustness_benchmark import (
    MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID as ROBUSTNESS_BENCHMARK_ID,
    ROBUSTNESS_BENCHMARK_SPECS,
)

# AI: Strategy Robustness and Target Oracle resolve their registered benchmark
# from the canonical config-driven specification.
_STRATEGY_ROBUSTNESS_SPEC = ROBUSTNESS_BENCHMARK_SPECS[ROBUSTNESS_BENCHMARK_ID]
ROBUSTNESS_BENCHMARK_SEED_COUNT = int(_STRATEGY_ROBUSTNESS_SPEC["seed_count"])
ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED = int(_STRATEGY_ROBUSTNESS_SPEC["seed_generator_seed"])

STRATEGY_COMPARE_SCHEMA_VERSION = 75

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
        "profile_id": STRATEGY_COMPARE_CURRENT_OOS_PROFILE,
        "robustness_id": STRATEGY_COMPARE_CURRENT_OOS_ROBUSTNESS,
    },
    {
        "mode_id": _ROLLING_ROLLING.mode_id,
        "label": _ROLLING_ROLLING.label,
        "score_start_date": _ROLLING_ROLLING.score_start_date,
        "score_end_date": _ROLLING_ROLLING.score_end_date,
        "fold_months": int(_ROLLING_ROLLING.fold_months),
        "fold_anchor_date": _ROLLING_ROLLING.fold_anchor_date,
        "single_score_block": bool(_ROLLING_ROLLING.single_score_block),
        "profile_id": STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE,
        "robustness_id": STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS,
    },
)

# Strategy arm的scientific base name只描述策略本體；current evaluation mode的
# OOS／Rolling suffix由profile renderer衍生，避免同一arm在各mode各自維護顯示名稱。
STRATEGY_COMPARE_DISPLAY_FULL_ROOS = "Full ROOS"
STRATEGY_COMPARE_DISPLAY_MIN_ROOS = "Min ROOS"
STRATEGY_COMPARE_DISPLAY_FULL_FINALISTS_AGREE = "Full Base-Finalists-Agree"
STRATEGY_COMPARE_DISPLAY_FULL_VALIDATION_BUDGET_REFIT = "Full Validation-Budget Refit"
STRATEGY_COMPARE_DISPLAY_FULL_VALIDATION_PLATEAU_REFIT = "Full Adaptive Plateau Validation Refit"
STRATEGY_COMPARE_DISPLAY_FULL_VALIDATION_PLATEAU_BEST_REFIT = "Full Adaptive Plateau Validation Refit / Finalist Best"
STRATEGY_COMPARE_DISPLAY_FULL_VALIDATION_PLATEAU_BEST_P100_REFIT = "Full Adaptive Plateau Validation / Finalist Best / Patience 100"
STRATEGY_COMPARE_DISPLAY_FULL_VALIDATION_PLATEAU_BEST_QUARTER_OFF_REFIT = "Full Adaptive Plateau Validation / Finalist Best / Quarter Off"
STRATEGY_COMPARE_DISPLAY_FULL_PRODUCTION_MULTI_SEED = "Full Production-Style Fixed Multi-Seed"
STRATEGY_COMPARE_DISPLAY_FULL_MULTI_SEED_8X300 = "Full Fixed Multi-Seed 8x300"
STRATEGY_COMPARE_DISPLAY_FULL_MULTI_SEED_4X1000 = "Full Fixed Multi-Seed 4x1000"
STRATEGY_COMPARE_DISPLAY_FULL_TRADING_ALIGNED_8X1000 = "Full Base-Finalists-Agree 8x1000"
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
STRATEGY_COMPARE_DISPLAY_MIN_MR13AC_NO_K_NO_R0 = "Min MR-13AC No-K No-R0"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AC_SCORE_CONSTRAINED = "Min MR-13AC Constrained"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AH_SCORE_CONSTRAINED = "Min MR-13AH Constrained"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AH_NO_K_NO_R0 = "Min MR-13AH No-K No-R0"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AK_SCORE_CONSTRAINED = "Min MR-13AK Constrained"
STRATEGY_COMPARE_DISPLAY_MIN_MR13AK_NO_K_NO_R0 = "Min MR-13AK No-K No-R0"

# Current Compare Suite是「比較誰／比較哪些差」的唯一真理來源。
# OOS／Rolling／single-seed／multi-seed都只能引用suite，不得各自再列current arm matrix。
# Suite invariants are intentionally stored separately from the mutable current membership:
# adding a treatment may append arms, but silently removing established references fails closed.
STRATEGY_COMPARE_SUITE_INVARIANTS = {
}

STRATEGY_COMPARE_SUITES = {
    "non_dl_broker_cash_v1": {
        "arm_ids": ("C96", "C97", "C98"),
        "contrast_ids": ("C97-C96", "C98-C97", "C98-C96"),
        "display_name_bases": {
            "C96": "Min Base-Finalist-Best / Broker-Cash",
            "C97": "Full Base-Finalist-Best / Broker-Cash",
            "C98": "Full Corrected Same-Signal Agree 4x300 / Broker-Cash",
        },
    },
    "broker_cash_dl_conversion_v1": {
        "membership_source": "research_summary",
        # Single scientific matrix shared by OOS/Rolling and multi-seed robustness.
        # Historical DL arms reuse C96/P16; source identity remains registry-owned.
        # AI: This catalog preserves historical ordering/display references.
        # Current membership is projected exclusively from Research Summary.
        # Current CJ/CK Direct and K/AH Exact/Direct use the same P16 broker-cash
        # accounting; they differ only by model score or selector.
        "arm_ids": ('C96', 'C97', 'C98', 'C103', 'C106', 'C107', 'C109', 'C189', 'C190', 'C111', 'C112', 'C118', 'C123', 'C124', 'C125', 'C126', 'C147', 'C148', 'C153', 'C154', 'C155', 'C156', 'C157', 'C158', 'C179', 'C181', 'C182', 'C183', 'C186', 'C168', 'C173', 'C172', 'C169'),
        # A compact connected contrast graph supports each selected model
        # versus C96 plus the matched CK/CJ label contrast. Historical AO/CH
        # contrasts remain registered and filter out with their inactive arms.
        "contrast_ids": ('C189-C96', 'C190-C96', 'C189-C190', 'C189-C106', 'C190-C118', 'C186-C183', 'C97-C96', 'C98-C97', 'C98-C96', 'C103-C96', 'C106-C96', 'C107-C96', 'C109-C96', 'C111-C96', 'C112-C96', 'C118-C96', 'C123-C96', 'C124-C96', 'C125-C96', 'C126-C96', 'C109-C107', 'C111-C112', 'C106-C118', 'C123-C124', 'C125-C126', 'C123-C125', 'C124-C126', 'C147-C107', 'C148-C109', 'C147-C148', 'C153-C97', 'C154-C97', 'C153-C107', 'C154-C109', 'C154-C153', 'C155-C97', 'C156-C97', 'C155-C156', 'C155-C153', 'C156-C154', 'C157-C97', 'C158-C97', 'C157-C158', 'C157-C153', 'C158-C154'),
        "display_name_bases": {'C96': 'Min Base-Finalist-Best / Broker-Cash', 'C97': 'Full Base-Finalist-Best / Broker-Cash', 'C98': 'Full Corrected Same-Signal Agree 4x300 / Broker-Cash', 'C103': 'Min MR-13CG Exact K/R0 / Broker-Cash', 'C106': 'Min MR-13CI Exact K/R0 / Broker-Cash', 'C189': 'Min MR-13CI Rank + MR-13DL R40 Positive Exact K/R0 / Broker-Cash', 'C190': 'Min MR-13CI Rank + MR-13DL R40 Positive Direct No-K No-R0 / Broker-Cash', 'C107': 'Min MR-13CJ Exact K/R0 / Broker-Cash', 'C109': 'Min MR-13CJ T40 Direct No-K No-R0 / Broker-Cash', 'C111': 'Min MR-13K Pure-MFE Control Exact K/R0 / Broker-Cash', 'C112': 'Min MR-13K Pure-MFE Direct No-K No-R0 / Broker-Cash', 'C118': 'Min MR-13CI R40 Direct No-K No-R0 / Broker-Cash', 'C123': 'Min MR-13CO Formal MinTrade Full-cycle NetPnL Exact K/R0 / Broker-Cash', 'C124': 'Min MR-13CO Formal MinTrade Full-cycle NetPnL Direct No-K No-R0 / Broker-Cash', 'C125': 'Min MR-13CO Rank + MR-13CP Positive Exact K/R0 / Broker-Cash', 'C126': 'Min MR-13CO Rank + MR-13CP Positive Direct No-K No-R0 / Broker-Cash', 'C147': 'Min MR-13CX T40 Pairwise-Confidence Exact K/R0 / Broker-Cash', 'C148': 'Min MR-13CX T40 Pairwise-Confidence Direct No-K No-R0 / Broker-Cash', 'C153': 'Full MR-13CJ T40 Exact K/R0 / Broker-Cash', 'C154': 'Full MR-13CJ T40 Direct No-K No-R0 / Broker-Cash', 'C155': 'Full MR-13CZ Rank + MR-13DA Positive Exact K/R0 / Broker-Cash', 'C156': 'Full MR-13CZ Rank + MR-13DA Positive Direct No-K No-R0 / Broker-Cash', 'C157': 'Full MR-13CZ NetPnL Rank Exact K/R0 / Broker-Cash', 'C158': 'Full MR-13CZ NetPnL Rank Direct No-K No-R0 / Broker-Cash', 'C179': 'Daily (Min/P21) No DL / Default Selector / Broker-Cash'},
    },
    # Historical Round-10 suite retained unchanged for artifact interpretation/replay.
}


# Strategy Compare以研究階段profile隔離設定與輸出；App只顯示泛化階段名稱，
# arms／contrasts／period／output namespace全部由本檔驅動。
STRATEGY_COMPARE_PROFILES = {
    "non_dl_broker_cash_oos": {
        "label": "No-DL Broker-Cash | OOS Test",
        "description": "C96/P16, C97/P17, C98/P15 all share broker_cash_v1, adjusted prices and PIT schedule.",
        "display_alignment_group": "non_dl_broker_cash_v1",
        "suite_id": "non_dl_broker_cash_v1",
        "display_suffix": "OOS",
        "start_date": "2021-01-01", "end_date": "2026-03-02",
        "point_in_time_score_start_date": _ROLLING_OOS.score_start_date,
        "point_in_time_score_end_date": _ROLLING_OOS.score_end_date,
        "point_in_time_fold_months": int(_ROLLING_OOS.fold_months),
        "point_in_time_fold_anchor_date": _ROLLING_OOS.fold_anchor_date,
        "point_in_time_single_score_block": bool(_ROLLING_OOS.single_score_block),
        "point_in_time_dirname": _ROLLING_OOS.point_in_time_dirname,
        "output_root": "outputs/strategy_compare/non_dl_broker_cash/oos_2021_forward",
        "reuse_output_roots": (),
        "arm_param_source_overrides": {
            "C96": "min_best_broker_cash_oos_experiment",
            "C97": "full_best_broker_cash_oos_experiment",
            "C98": "full_agree_broker_cash_oos_experiment",
        },
    },
    "non_dl_broker_cash_rolling": {
        "label": "No-DL Broker-Cash | Rolling Test",
        "description": "C96/P16, C97/P17, C98/P15 all share broker_cash_v1, adjusted prices and PIT schedule.",
        "display_alignment_group": "non_dl_broker_cash_v1",
        "suite_id": "non_dl_broker_cash_v1",
        "display_suffix": "Rolling",
        "start_date": "2021-01-01", "end_date": "2026-03-02",
        "point_in_time_score_start_date": _ROLLING_ROLLING.score_start_date,
        "point_in_time_score_end_date": _ROLLING_ROLLING.score_end_date,
        "point_in_time_fold_months": int(_ROLLING_ROLLING.fold_months),
        "point_in_time_fold_anchor_date": _ROLLING_ROLLING.fold_anchor_date,
        "point_in_time_single_score_block": bool(_ROLLING_ROLLING.single_score_block),
        "point_in_time_dirname": _ROLLING_ROLLING.point_in_time_dirname,
        "output_root": "outputs/strategy_compare/non_dl_broker_cash/rolling_2021_forward",
        "reuse_output_roots": (),
        "arm_param_source_overrides": {
            "C96": "min_best_broker_cash_rolling_experiment",
            "C97": "full_best_broker_cash_rolling_experiment",
            "C98": "full_agree_broker_cash_rolling_experiment",
        },
    },
    # Historical Round-10 profiles retained unchanged.
    # Legacy evaluation policies retained only for historical replay / artifact interpretation.
}

# Canonical broker-cash DL strategy conversion work stage. Reuse the registered
# No-DL parameter truth and PIT timing; the DL arms have no new optimizer stage.
# These are new profile identities, not in-place mutations of historical No-DL runs.
_BROKER_CASH_DL_ARM_IDS = tuple(
    arm_id for arm_id in STRATEGY_COMPARE_SUITES["broker_cash_dl_conversion_v1"]["arm_ids"]
    if arm_id not in STRATEGY_COMPARE_SUITES["non_dl_broker_cash_v1"]["arm_ids"]
)
for _mode in ("oos", "rolling"):
    _baseline_profile = STRATEGY_COMPARE_PROFILES[f"non_dl_broker_cash_{_mode}"]
    _namespace = f"outputs/strategy_compare/broker_cash_dl_conversion/{_mode}_2021_forward"
    STRATEGY_COMPARE_PROFILES[f"broker_cash_dl_conversion_{_mode}"] = {
        **_baseline_profile,
        "label": f"Broker-Cash DL Conversion | {_mode.upper() if _mode == 'oos' else 'Rolling'} Test",
        "description": (
            "C96/P16 broker-cash Min control plus configured model-list "
            "DL score conversions (authorized Exact K/R0 and Direct No-K/No-R0); "
            "C97/P17 and C98/P15 remain reference arms."
        ),
        "display_alignment_group": "broker_cash_dl_conversion_v1",
        "suite_id": "broker_cash_dl_conversion_v1",
        "output_root": _namespace,
        # Registered result roots are discovered by the generic identity-gated resolver.
        # Do not wire cross-profile reuse paths into a specific research stage.
        "arm_param_source_overrides": {
            **_baseline_profile["arm_param_source_overrides"],
        },
    }


# Multiple-seed robustness各研究階段以獨立config profile呈現於正式選單。
# Current OOS／Rolling robustness只引用Compare Suite；所有suite arms與contrasts都逐seed重複，
# 僅model-seed-sensitive子集由arm的DL dependency推導，不得再於profile重複維護比較矩陣。
# Historical robustness profiles保留顯式membership供artifact compatibility；orchestration不得改動
# 單次Strategy Compare arm identity/fingerprint。
STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES = {
    "non_dl_broker_cash_oos": {
        "label": "No-DL Broker-Cash Robustness | OOS Test",
        "enabled": True, "profile_id": "non_dl_broker_cash_oos",
        "suite_id": "non_dl_broker_cash_v1",
        "benchmark_id": ROBUSTNESS_BENCHMARK_ID,
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
            "min": {"param_source": "min_best_broker_cash_oos_experiment", "param_policy": "base-finalist-best", "rule_policy": "all_off"},
            "full": {"param_source": "full_best_broker_cash_oos_experiment", "param_policy": "base-finalist-best", "rule_policy": "formal"},
        },
        "output_root": "outputs/strategy_compare/robustness/non_dl_broker_cash/oos_2021_forward",
        "model_work_root": "models/research/breakout_quality/strategy_compare/multi_seed_robustness/non_dl_broker_cash/oos_2021_forward",
        "checkpoint_cache_root": STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT,
    },
    "non_dl_broker_cash_rolling": {
        "label": "No-DL Broker-Cash Robustness | Rolling Test",
        "enabled": True, "profile_id": "non_dl_broker_cash_rolling",
        "suite_id": "non_dl_broker_cash_v1",
        "benchmark_id": ROBUSTNESS_BENCHMARK_ID,
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
            "min": {"param_source": "min_best_broker_cash_rolling_experiment", "param_policy": "base-finalist-best", "rule_policy": "all_off"},
            "full": {"param_source": "full_best_broker_cash_rolling_experiment", "param_policy": "base-finalist-best", "rule_policy": "formal"},
        },
        "output_root": "outputs/strategy_compare/robustness/non_dl_broker_cash/rolling_2021_forward",
        "model_work_root": "models/research/breakout_quality/strategy_compare/multi_seed_robustness/non_dl_broker_cash/rolling_2021_forward",
        "checkpoint_cache_root": STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT,
    },

}


# The four formal modes resolve the SAME scientific suite.  No per-mode arm,
# model, seed or contrast lists are maintained by the robustness configuration.
for _mode in ("oos", "rolling"):
    _baseline_robustness = STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES[
        f"non_dl_broker_cash_{_mode}"
    ]
    STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES[
        f"broker_cash_dl_conversion_{_mode}"
    ] = {
        **_baseline_robustness,
        "label": f"Broker-Cash DL Conversion Robustness | {_mode.upper() if _mode == 'oos' else 'Rolling'} Test",
        "profile_id": f"broker_cash_dl_conversion_{_mode}",
        "suite_id": "broker_cash_dl_conversion_v1",
        "output_root": f"outputs/strategy_compare/robustness/broker_cash_dl_conversion/{_mode}_2021_forward",
        "model_work_root": (
            "models/research/breakout_quality/strategy_compare/"
            f"multi_seed_robustness/broker_cash_dl_conversion/{_mode}_2021_forward"
        ),
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

# Research-only parameter-source identities. These are deliberately outside the
# current Strategy Compare parameter-source catalog so temporary experiments do
# not expand the production/current menu contract. The resolver merges them for
# profiles that explicitly opt in.
STRATEGY_EXPERIMENTAL_PARAM_SOURCES = {
    # New scientific fee treatment uses one physical parameter truth for OOS/Rolling.
    # Compatibility builder name also accepts registered single-seed profiles;
    # the Optimizer is still the sole producer and validates profile capability.
    "min_best_broker_cash_oos_experiment": {
        "path_template": "models/research/breakout_quality/strategy_params/param_p16_min_best_broker_cash/{param_filename}",
        "identity_manifest_path": "models/research/breakout_quality/strategy_params/param_p16_min_best_broker_cash/identity_manifest.json",
        "description": "PARAM-P16 registered broker-cash parameter truth; oos consumer.",
        "trained_with_dl_id": None,
        "consumption_evaluation_mode": "oos",
        "artifact_contract": {"breakout_quality_param_adaptation": {
            "parameter_set": "PARAM-P16", "training_dl_enabled": False,
            "optimizer_search_protocol": "fixed_single_seed_v1",
        }},
        "builder": {"enabled": True, "builder_type": "optimizer_fixed_multiseed_strategy_params",
                    "options": {"training_profile_id": "broker_cash_min_best_seed42_current"}},
    },
    "min_best_broker_cash_rolling_experiment": {
        "path_template": "models/research/breakout_quality/strategy_params/param_p16_min_best_broker_cash/{param_filename}",
        "identity_manifest_path": "models/research/breakout_quality/strategy_params/param_p16_min_best_broker_cash/identity_manifest.json",
        "description": "PARAM-P16 registered broker-cash parameter truth; rolling consumer.",
        "trained_with_dl_id": None,
        "consumption_evaluation_mode": "rolling",
        "artifact_contract": {"breakout_quality_param_adaptation": {
            "parameter_set": "PARAM-P16", "training_dl_enabled": False,
            "optimizer_search_protocol": "fixed_single_seed_v1",
        }},
        "builder": {"enabled": True, "builder_type": "optimizer_fixed_multiseed_strategy_params",
                    "options": {"training_profile_id": "broker_cash_min_best_seed42_current"}},
    },
    "daily_min_six_best_broker_cash_oos_experiment": {
        "path_template": "models/research/breakout_quality/strategy_params/param_p21_daily_min_best_broker_cash/{param_filename}",
        "identity_manifest_path": "models/research/breakout_quality/strategy_params/param_p21_daily_min_best_broker_cash/identity_manifest.json",
        "description": "PARAM-P21 independent 6D close-limit EMA+volume Daily Min parameter truth; OOS evaluation view.",
        "trained_with_dl_id": None,
        "consumption_evaluation_mode": "oos",
        "artifact_contract": {"breakout_quality_param_adaptation": {
            "parameter_set": "PARAM-P21", "training_dl_enabled": False,
            "optimizer_search_protocol": "fixed_single_seed_v1",
        }},
        "builder": {"enabled": True, "builder_type": "optimizer_fixed_multiseed_strategy_params",
                    "options": {"training_profile_id": "broker_cash_daily_min_six_seed42_current"}},
    },
    "daily_min_six_best_broker_cash_rolling_experiment": {
        "path_template": "models/research/breakout_quality/strategy_params/param_p21_daily_min_best_broker_cash/{param_filename}",
        "identity_manifest_path": "models/research/breakout_quality/strategy_params/param_p21_daily_min_best_broker_cash/identity_manifest.json",
        "description": "PARAM-P21 independent 6D close-limit EMA+volume Daily Min parameter truth; Rolling evaluation view.",
        "trained_with_dl_id": None,
        "consumption_evaluation_mode": "rolling",
        "artifact_contract": {"breakout_quality_param_adaptation": {
            "parameter_set": "PARAM-P21", "training_dl_enabled": False,
            "optimizer_search_protocol": "fixed_single_seed_v1",
        }},
        "builder": {"enabled": True, "builder_type": "optimizer_fixed_multiseed_strategy_params",
                    "options": {"training_profile_id": "broker_cash_daily_min_six_seed42_current"}},
    },
    "daily_full_nineteen_best_broker_cash_oos_experiment": {
        "path_template": "models/research/breakout_quality/strategy_params/param_p20_daily_full_best_broker_cash/{param_filename}",
        "identity_manifest_path": "models/research/breakout_quality/strategy_params/param_p20_daily_full_best_broker_cash/identity_manifest.json",
        "description": "PARAM-P20 independently Daily-trained full strategy parameters; oos evaluation view.",
        "trained_with_dl_id": None,
        "consumption_evaluation_mode": "oos",
        "artifact_contract": {"breakout_quality_param_adaptation": {
            "parameter_set": "PARAM-P20", "training_dl_enabled": False,
            "optimizer_search_protocol": "fixed_single_seed_v1",
        }},
        "builder": {"enabled": True, "builder_type": "optimizer_fixed_multiseed_strategy_params",
                    "options": {"training_profile_id": "broker_cash_daily_full_nineteen_seed42_current"}},
    },
    "daily_full_nineteen_best_broker_cash_rolling_experiment": {
        "path_template": "models/research/breakout_quality/strategy_params/param_p20_daily_full_best_broker_cash/{param_filename}",
        "identity_manifest_path": "models/research/breakout_quality/strategy_params/param_p20_daily_full_best_broker_cash/identity_manifest.json",
        "description": "PARAM-P20 independently Daily-trained full strategy parameters; rolling evaluation view.",
        "trained_with_dl_id": None,
        "consumption_evaluation_mode": "rolling",
        "artifact_contract": {"breakout_quality_param_adaptation": {
            "parameter_set": "PARAM-P20", "training_dl_enabled": False,
            "optimizer_search_protocol": "fixed_single_seed_v1",
        }},
        "builder": {"enabled": True, "builder_type": "optimizer_fixed_multiseed_strategy_params",
                    "options": {"training_profile_id": "broker_cash_daily_full_nineteen_seed42_current"}},
    },
    "full_best_broker_cash_oos_experiment": {
        "path_template": "models/research/breakout_quality/strategy_params/param_p17_full_best_broker_cash/{param_filename}",
        "identity_manifest_path": "models/research/breakout_quality/strategy_params/param_p17_full_best_broker_cash/identity_manifest.json",
        "description": "PARAM-P17 registered broker-cash parameter truth; oos consumer.",
        "trained_with_dl_id": None,
        "consumption_evaluation_mode": "oos",
        "artifact_contract": {"breakout_quality_param_adaptation": {
            "parameter_set": "PARAM-P17", "training_dl_enabled": False,
            "optimizer_search_protocol": "fixed_single_seed_v1",
        }},
        "builder": {"enabled": True, "builder_type": "optimizer_fixed_multiseed_strategy_params",
                    "options": {"training_profile_id": "broker_cash_full_best_seed42_current"}},
    },
    "full_best_broker_cash_rolling_experiment": {
        "path_template": "models/research/breakout_quality/strategy_params/param_p17_full_best_broker_cash/{param_filename}",
        "identity_manifest_path": "models/research/breakout_quality/strategy_params/param_p17_full_best_broker_cash/identity_manifest.json",
        "description": "PARAM-P17 registered broker-cash parameter truth; rolling consumer.",
        "trained_with_dl_id": None,
        "consumption_evaluation_mode": "rolling",
        "artifact_contract": {"breakout_quality_param_adaptation": {
            "parameter_set": "PARAM-P17", "training_dl_enabled": False,
            "optimizer_search_protocol": "fixed_single_seed_v1",
        }},
        "builder": {"enabled": True, "builder_type": "optimizer_fixed_multiseed_strategy_params",
                    "options": {"training_profile_id": "broker_cash_full_best_seed42_current"}},
    },
    "full_agree_broker_cash_oos_experiment": {
        "path_template": "models/research/breakout_quality/strategy_params/param_p15_c85_same_signal_broker_cash/{param_filename}",
        "identity_manifest_path": "models/research/breakout_quality/strategy_params/param_p15_c85_same_signal_broker_cash/identity_manifest.json",
        "description": "PARAM-P15 registered broker-cash parameter truth; oos consumer.",
        "trained_with_dl_id": None,
        "consumption_evaluation_mode": "oos",
        "artifact_contract": {"breakout_quality_param_adaptation": {
            "parameter_set": "PARAM-P15", "training_dl_enabled": False,
            "optimizer_search_protocol": "fixed_multiseed_v1",
        }},
        "builder": {"enabled": True, "builder_type": "optimizer_fixed_multiseed_strategy_params",
                    "options": {"training_profile_id": "c85_same_signal_broker_cash_current"}},
    },
    "full_agree_broker_cash_rolling_experiment": {
        "path_template": "models/research/breakout_quality/strategy_params/param_p15_c85_same_signal_broker_cash/{param_filename}",
        "identity_manifest_path": "models/research/breakout_quality/strategy_params/param_p15_c85_same_signal_broker_cash/identity_manifest.json",
        "description": "PARAM-P15 registered broker-cash parameter truth; rolling consumer.",
        "trained_with_dl_id": None,
        "consumption_evaluation_mode": "rolling",
        "artifact_contract": {"breakout_quality_param_adaptation": {
            "parameter_set": "PARAM-P15", "training_dl_enabled": False,
            "optimizer_search_protocol": "fixed_multiseed_v1",
        }},
        "builder": {"enabled": True, "builder_type": "optimizer_fixed_multiseed_strategy_params",
                    "options": {"training_profile_id": "c85_same_signal_broker_cash_current"}},
    },
}

# =============================================================================
# 4. DL工件來源
# =============================================================================

STRATEGY_DL_SOURCES = {
    # Existing historical C60 Pure-MFE PIT source, promoted unchanged to the
    # current catalog so K Exact/Direct can share its canonical checkpoint/score.
    'CONT13K_ROLL': {'filter_id': 'breakout_quality_v1',
                     'model_architecture': 'inception_time_v1',
                     'experiment_profile': 'daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise',
                     'threshold': None,
                     'score_source': 'selection_point_in_time',
                     'description': 'MR-13K Extending-Window Rolling PIT-safe pure-MFE score；expanding history + '
                                    'mode-specific refit cadence。',
                     'forward_scores_builder': {'enabled': True,
                                                'builder_type': 'selection_pit_from_existing_folds',
                                                'options': {'resume': True, 'allow_stale_source': False}}},
    # Research-only strategy conversion sources.  These register no new model
    # fitting identity, do not grant Trading promotion, and use the canonical
    # model-list PIT workflow (including the 40D-mature CH stock-day cohort).
    "CONT13CG_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_shared_safety_mfe_v1",
        "experiment_profile": "daily_universal_shared_full_d40_safety_hs_conditional_mfe_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13CG full D40 Safety/true-HS conditional-MFE score; controlled broker-cash Strategy Compare only, no Trading.",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13DL_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_terminal_r40_positive_binary",
        "threshold": None, "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13DL independent R40>0 two-class PIT classifier, shared by P16 Exact/Direct.",
        "forward_scores_builder": {"enabled": True, "builder_type": "selection_pit_from_existing_folds",
                                   "options": {"resume": True, "allow_stale_source": False}},
    },
    "CONT13CI_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_terminal_r40_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13CI R40 canonical PIT score; research-only controlled P16 Exact conversion, not Trading.",
        "forward_scores_builder": {"enabled": True, "builder_type": "selection_pit_from_existing_folds",
                                   "options": {"resume": True, "allow_stale_source": False}},
    },
    "CONT13CJ_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_terminal_mae_gated_t40_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13CJ terminal T40 canonical PIT score; research-only P16 Exact conversion, not Trading.",
        "forward_scores_builder": {"enabled": True, "builder_type": "selection_pit_from_existing_folds",
                                   "options": {"resume": True, "allow_stale_source": False}},
    },
    "CONT13CX_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_terminal_mae_gated_t40_probabilistic_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": (
            "MR-13CX proper Bernoulli same-date T40 pair probabilities, "
            "combined into an expected-win-fraction score over PIT eligible stocks; "
            "one source for Breakout/Daily Exact/Direct; research only."
        ),
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13CO_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_formal_mintrade_fullcycle_netpnl_per_seed_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": (
            "MR-13CO: per-seed canonical Min ROOS full-cycle NetPnL,"
            "PIT-matured historical Optimizer/P16 source only; research comparison, not Trading."
        ),
        "forward_scores_builder": {
            "enabled": True, "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13CP_ROLL": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_formal_mintrade_fullcycle_netpnl_positive_binary",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": (
            "MR-13CP independent binary NetPnL>0 classifier; "
            "C125/C126 rank with CO and admit with CP PIT probability."
        ),
        "forward_scores_builder": {
            "enabled": True, "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
    "CONT13DH_ROLL": {
        "filter_id": "breakout_quality_v1", "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_min_p21_mintrade_netpnl_full_list_ndcg_pairwise",
        "threshold": None, "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13DH independently Daily Min/P21 PIT full-cycle broker NetPnL Rank",
        "forward_scores_builder": {"enabled": True, "builder_type": "selection_pit_from_existing_folds",
                                   "options": {"resume": True, "allow_stale_source": False}},
    },
    "CONT13DJ_ROLL": {
        "filter_id": "breakout_quality_v1", "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_min_p21_netpnl_utility_binary",
        "threshold": None, "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13DJ independent Positive learned decision; no upstream DL or tuned threshold",
        "forward_scores_builder": {"enabled": True, "builder_type": "selection_pit_from_existing_folds",
                                   "options": {"resume": True, "allow_stale_source": False}},
    },
    "CONT13DI_ROLL": {
        "filter_id": "breakout_quality_v1", "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_min_p21_mintrade_netpnl_positive_binary",
        "threshold": None, "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13DI independent Min/P21 PIT NetPnL Positive classifier",
        "forward_scores_builder": {"enabled": True, "builder_type": "selection_pit_from_existing_folds",
                                   "options": {"resume": True, "allow_stale_source": False}},
    },
    "CONT13DB_ROLL": {
        "filter_id": "breakout_quality_v1", "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_p20_fulltrade_netpnl_full_list_ndcg_pairwise",
        "threshold": None, "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13DB independent Daily Full/P20 19D PIT full-cycle broker NetPnL Rank",
        "forward_scores_builder": {"enabled": True, "builder_type": "selection_pit_from_existing_folds",
                                   "options": {"resume": True, "allow_stale_source": False}},
    },
    "CONT13DC_ROLL": {
        "filter_id": "breakout_quality_v1", "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_p20_fulltrade_netpnl_positive_binary",
        "threshold": None, "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13DC independent Daily Full/P20 PIT NetPnL>0 Positive Classifier",
        "forward_scores_builder": {"enabled": True, "builder_type": "selection_pit_from_existing_folds",
                                   "options": {"resume": True, "allow_stale_source": False}},
    },
    "CONT13CZ_ROLL": {
        "filter_id": "breakout_quality_v1", "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_formal_fulltrade_fullcycle_netpnl_full_list_ndcg_pairwise",
        "threshold": None, "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13CZ per-seed verified Full/P17 PIT broker NetPnL rank, not Min/P16",
        "forward_scores_builder": {"enabled": True, "builder_type": "selection_pit_from_existing_folds",
                                   "options": {"resume": True, "allow_stale_source": False}},
    },
    "CONT13DA_ROLL": {
        "filter_id": "breakout_quality_v1", "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_formal_fulltrade_fullcycle_netpnl_positive_binary",
        "threshold": None, "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13DA independent Full/P17 binary NetPnL>0 PIT classifier",
        "forward_scores_builder": {"enabled": True, "builder_type": "selection_pit_from_existing_folds",
                                   "options": {"resume": True, "allow_stale_source": False}},
    },
    'CONT13AO_ROLL': {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_shared_safety_mfe_v1",
        "experiment_profile": "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "single_seed_strategy_conversion_authorized": True,
        "description": "MR-13AO frozen 40D true-HS conditional-MFE score; controlled broker-cash Strategy Compare only, no Trading.",
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
    "C96": {
        "name": "Min Base-Finalist-Best / Broker-Cash",
        "description": "SR-C96 / PARAM-P16; Min/all-off, fixed Seed42 x300 and broker_cash_v1.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": False, "dl_id": None, "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C97": {
        "name": "Full Base-Finalist-Best / Broker-Cash",
        "description": "SR-C97 / PARAM-P17; Full/formal, fixed Seed42 x300 and broker_cash_v1.",
        "param_source": "full_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": False, "dl_id": None, "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C98": {
        "name": "Full Corrected Same-Signal Agree 4x300 / Broker-Cash",
        "description": "SR-C98 / PARAM-P15; Full/formal, fixed 4x300 same-original-signal consensus and broker_cash_v1.",
        "param_source": "full_agree_broker_cash_rolling_experiment",
        "param_policy": "base-finalists-agree", "rule_policy": "formal",
        "dl_enabled": False, "dl_id": None, "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    # P16 single truth and identical Min/all-off broker-cash economics across six
    # arms.  Score identity and Exact-vs-Direct are the only treatment dimensions.
    "C103": {
        "name": "Min MR-13CG Exact K/R0 / Broker-Cash",
        "description": "SR-C103: CG full-D40 score, P16 Min broker-cash; exact K/R0 controlled conversion.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CG_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C189": {
        "replay_contract_version": 1,
        "name": "Min MR-13CI Rank + MR-13DL R40 Positive Exact K/R0 / Broker-Cash",
        "description": "Independent CI R40 rank plus DL P(R40>0)>0.5; same P16 Breakout execution, costs and cash.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CI_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True, "constrained_solver": "exact_branch_and_bound_v1", "selection_only": True,
            "positive_score_column": "positive_p_positive",
            "positive_dl_id": "CONT13DL_ROLL",
            "positive_probability_threshold": 0.5,
        },
        "robustness_role": "off",
    },
    "C190": {
        "replay_contract_version": 1,
        "name": "Min MR-13CI Rank + MR-13DL R40 Positive Direct No-K No-R0 / Broker-Cash",
        "description": "Independent CI R40 rank plus DL P(R40>0)>0.5; same P16 Breakout execution, costs and cash.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CI_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1", "selection_only": True,
            "positive_score_column": "positive_p_positive",
            "positive_dl_id": "CONT13DL_ROLL",
            "positive_probability_threshold": 0.5,
        },
        "robustness_role": "off",
    },
    "C106": {
        "name": "Min MR-13CI Exact K/R0 / Broker-Cash",
        "description": "SR-C106: CI terminal R40 PIT score, fixed P16 Min broker-cash; Exact K/R0.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CI_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {"preserve_k_r0": True, "constrained_solver": "exact_branch_and_bound_v1", "selection_only": True},
        "robustness_role": "off",
    },
    "C107": {
        "name": "Min MR-13CJ Exact K/R0 / Broker-Cash",
        "description": "SR-C107: CJ terminal T40 PIT score, fixed P16 Min broker-cash; Exact K/R0.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CJ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {"preserve_k_r0": True, "constrained_solver": "exact_branch_and_bound_v1", "selection_only": True},
        "robustness_role": "off",
    },
    # Newly authorized P16 conversion controls. No new target/optimizer/model FIT;
    # each arm reuses its existing model PIT score and canonical selector.
    "C109": {
        "name": "Min MR-13CJ T40 Direct No-K No-R0 / Broker-Cash",
        "description": "SR-C109: C107 T40 PIT score with P16 broker cash; selector only changes Exact K/R0 to Direct No-K/No-R0.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CJ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C111": {
        "name": "Min MR-13K Pure-MFE Control Exact K/R0 / Broker-Cash",
        "description": "SR-C111: canonical MR-13K pure-MFE PIT score; C96/P16 broker cash and Exact K/R0 control selector.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13K_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C112": {
        "name": "Min MR-13K Pure-MFE Direct No-K No-R0 / Broker-Cash",
        "description": "SR-C112: same MR-13K pure-MFE PIT score and P16 broker cash as C111; Direct No-K/No-R0 selector.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13K_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    # AH is Safety-product-weighted Pure-MFE, not the unweighted MFE40 label.
    # Keep C113/C114 retired under historical compatibility; never reuse their IDs.
    "C118": {
        "name": "Min MR-13CI R40 Direct No-K No-R0 / Broker-Cash",
        "description": "SR-C118: same MR-13CI terminal R40 PIT scores and P16 broker cash as C106; canonical Direct No-K/No-R0 selector only.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CI_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C123": {
        "name": "Min MR-13CO Formal MinTrade Full-cycle NetPnL Exact K/R0 / Broker-Cash",
        "description": "MR-13CO same-seed canonical P16 MinTrade Full-cycle NetPnL PIT rank; C96/P16 broker-cash with Exact K/R0.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CO_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C124": {
        "name": "Min MR-13CO Formal MinTrade Full-cycle NetPnL Direct No-K No-R0 / Broker-Cash",
        "description": "Same MR-13CO PIT scores/P16/broker cash as C123; only selector changes to Direct No-K/No-R0.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CO_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C125": {
        "replay_contract_version": 3,
        "name": "Min MR-13CO Rank + MR-13CP Positive Exact K/R0 / Broker-Cash",
        "description": "Independent CO NetPnL rank and CP binary Positive classifier, both PIT; canonical Min selector, costs and cash.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CO_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True, "constrained_solver": "exact_branch_and_bound_v1", "selection_only": True,
            "positive_score_column": "positive_p_positive",
            "positive_dl_id": "CONT13CP_ROLL",
            "positive_probability_threshold": 0.5,
        },
        "robustness_role": "off",
    },
    "C126": {
        "replay_contract_version": 3,
        "name": "Min MR-13CO Rank + MR-13CP Positive Direct No-K No-R0 / Broker-Cash",
        "description": "Independent CO NetPnL rank and CP binary Positive classifier, both PIT; canonical Min selector, costs and cash.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CO_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1", "selection_only": True,
            "positive_score_column": "positive_p_positive",
            "positive_dl_id": "CONT13CP_ROLL",
            "positive_probability_threshold": 0.5,
        },
        "robustness_role": "off",
    },
    "C147": {
        "replay_contract_version": 1,
        "name": "Min MR-13CX T40 Pairwise-Confidence Exact K/R0 / Broker-Cash",
        "description": "CX PIT expected same-date pair-win fraction on Breakout/P16; otherwise matched to CJ C107 Exact.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CX_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True, "constrained_solver": "exact_branch_and_bound_v1", "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C148": {
        "replay_contract_version": 1,
        "name": "Min MR-13CX T40 Pairwise-Confidence Direct No-K No-R0 / Broker-Cash",
        "description": "CX PIT expected same-date pair-win fraction on Breakout/P16; otherwise matched to CJ C109 Direct.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13CX_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1", "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C153": {
        "replay_contract_version": 1,
        "name": "Full MR-13CJ T40 Exact K/R0 / Broker-Cash",
        "description": "Full P17 base-finalist-best Broker-Cash with original CJ T40 PIT score; Exact matched to C107, non-DL reference C97.",
        "param_source": "full_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13CJ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True, "constrained_solver": "exact_branch_and_bound_v1", "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C154": {
        "replay_contract_version": 1,
        "name": "Full MR-13CJ T40 Direct No-K No-R0 / Broker-Cash",
        "description": "Same Full P17 and CJ PIT score as C153; Direct selector matched to C109; non-DL reference C97.",
        "param_source": "full_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13CJ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1", "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C155": {
        "replay_contract_version": 1,
        "name": "Full MR-13CZ Rank + MR-13DA Positive Exact K/R0 / Broker-Cash",
        "description": "Two independently FIT Full/P17 models: CZ continuous NetPnL rank and DA true binary NetPnL>0 admission; same Full control C97.",
        "param_source": "full_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13CZ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True, "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True, "positive_score_column": "positive_p_positive",
            "positive_dl_id": "CONT13DA_ROLL", "positive_probability_threshold": 0.5,
        },
        "robustness_role": "off",
    },
    "C156": {
        "replay_contract_version": 1,
        "name": "Full MR-13CZ Rank + MR-13DA Positive Direct No-K No-R0 / Broker-Cash",
        "description": "Same Full/P17 CZ rank and DA Positive independent PIT sources as C155; only selector changes to Direct.",
        "param_source": "full_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13CZ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True, "positive_score_column": "positive_p_positive",
            "positive_dl_id": "CONT13DA_ROLL", "positive_probability_threshold": 0.5,
        },
        "robustness_role": "off",
    },
    "C157": {
        "replay_contract_version": 1,
        "name": "Full MR-13CZ NetPnL Rank Exact K/R0 / Broker-Cash",
        "description": "Full/P17 same broker NetPnL target and PIT CZ rank as historical C155, with no DA Positive admission; Full no-DL control C97.",
        "param_source": "full_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13CZ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True, "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C158": {
        "replay_contract_version": 1,
        "name": "Full MR-13CZ NetPnL Rank Direct No-K No-R0 / Broker-Cash",
        "description": "Same Full/P17 PIT CZ rank and broker cash as C157; no Positive admission, only Direct selector differs.",
        "param_source": "full_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13CZ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C168": {
        "replay_contract_version": 1,
        "name": "Daily (Full/P20) No DL / Default Selector / Broker-Cash",
        "description": "P20 new 19D common filters independent daily optimizer and unscored daily-stockday control.",
        "param_source": "daily_full_nineteen_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": False, "dl_id": None, "dl_runtime_mode": None,
        "robustness_role": "off", "entry_strategy": "Daily",
    },
    "C169": {
        "replay_contract_version": 1,
        "name": "T40 × Daily (Full/P20) Direct No-K No-R0 / Broker-Cash",
        "description": "PIT MR-13CJ T40 score with independently Daily-optimized Full/P20 execution; same-family C168 No-DL control.",
        "param_source": "daily_full_nineteen_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13CJ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off", "entry_strategy": "Daily",
    },
        "C181": {
        "replay_contract_version": 1,
        "name": "T40 x Daily (Min/P21) Direct No-K No-R0 / Broker-Cash",
        "description": "P21 six-dimensional EMA+volume Daily close-limit Direct; same-seed P21 parameters, C179 control.",
        "param_source": "daily_min_six_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13CJ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {"preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1", "selection_only": True,
        },
        "robustness_role": "off", "entry_strategy": "Daily",
    },
    "C182": {
        "replay_contract_version": 1,
        "name": "Daily_MinTrade_NetPnL x Daily (Min/P21) Direct No-K No-R0 / Broker-Cash",
        "description": "P21 six-dimensional EMA+volume Daily close-limit Direct; same-seed P21 parameters, C179 control.",
        "param_source": "daily_min_six_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13DH_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {"preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1", "selection_only": True,
        },
        "robustness_role": "off", "entry_strategy": "Daily",
    },
    "C186": {
        "replay_contract_version": 4,
        "name": "Daily_MinTrade_NetPnL_Positive x Daily (Min/P21) Direct / Learned Decision / Broker-Cash",
        "description": "Same C183 DH rank and P21 economics; second DL DJ learns payoff-sensitive trade/cash decision, no DI dependency or threshold search.",
        "param_source": "daily_min_six_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13DH_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {"preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1", "selection_only": True,
            "positive_dl_id": "CONT13DJ_ROLL", "positive_score_column": "positive_trade_allowed",
            "positive_source_mode": "model_decision",
        },
        "robustness_role": "off", "entry_strategy": "Daily",
    },
    "C183": {
        "replay_contract_version": 1,
        "name": "Daily_MinTrade_NetPnL_Positive x Daily (Min/P21) Direct No-K No-R0 / Broker-Cash",
        "description": "P21 six-dimensional EMA+volume Daily close-limit Direct; same-seed P21 parameters, C179 control.",
        "param_source": "daily_min_six_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13DH_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {"preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1", "selection_only": True,
            "positive_dl_id": "CONT13DI_ROLL", "positive_score_column": "positive_p_positive", "positive_probability_threshold": 0.5,
        },
        "robustness_role": "off", "entry_strategy": "Daily",
    },
    "C179": {
        "replay_contract_version": 1,
        "name": "Daily (Min/P21) No DL / Default Selector / Broker-Cash",
        "description": "P21 independently Daily-optimized 6D Min close-limit entry with EMA and prior-volume filters always on; no DL scores or Direct selector.",
        "param_source": "daily_min_six_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": False, "dl_id": None, "dl_runtime_mode": None,
        "robustness_role": "off", "entry_strategy": "Daily",
    },
    "C172": {
        "replay_contract_version": 1,
        "name": "DailyUniversal FullTrade NetPnL Positive x Daily (Full/P20) Direct",
        "description": "Daily Full/P20 19D execution; independent MR-13DB P20 full-cycle broker NetPnL rank and MR-13DC same-P20 binary Positive classification. C168 No-DL control, C173 rank-only contrast. Never reuse CU/CT or CZ/DA FIT.",
        "param_source": "daily_full_nineteen_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13DB_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True, "positive_score_column": "positive_p_positive",
            "positive_dl_id": "CONT13DC_ROLL", "positive_probability_threshold": 0.5,
        },
        "robustness_role": "off", "entry_strategy": "Daily",
    },
    "C173": {
        "replay_contract_version": 1,
        "name": "DailyUniversal FullTrade NetPnL x Daily (Full/P20) Direct",
        "description": "Daily Full/P20 19D execution and independent MR-13DB P20 full-cycle broker NetPnL rank without Positive admission; same teacher and PIT rank source as C172.",
        "param_source": "daily_full_nineteen_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13DB_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1", "selection_only": True,
        },
        "robustness_role": "off", "entry_strategy": "Daily",
    },
    # Historical C62 prose/name is intentionally frozen for completed-result fingerprint
    # compatibility.  Governance correction (2026-09-28/29): C62 is the Seed42
    # single-seed finalists-agree controlled reference, not exact Trading production.
    # New suites must use their own display_name_bases (see Round-2 C85 suite) rather
    # than mutating this historical arm payload and invalidating durable result reuse.
    # Required dormant Exact selector witnesses for current Oracle families.
    'C171': {
        "replay_contract_version": 1,
        "name": "T40 × Daily (Full/P20) Exact K/R0 / Oracle contract witness (INACTIVE)",
        "description": "INACTIVE registered Exact witness for the shared formal Oracle selector contract; not in current Summary or strategy comparison.",
        "param_source": "daily_full_nineteen_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13CJ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True, "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off", "entry_strategy": "Daily",
    },
    'C184': {
        "replay_contract_version": 1,
        "name": "T40 × Daily (Min/P21) Exact K/R0 / Oracle contract witness (INACTIVE)",
        "description": "INACTIVE registered Exact witness for the shared formal Oracle selector contract; not in current Summary or strategy comparison.",
        "param_source": "daily_min_six_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "formal",
        "dl_enabled": True, "dl_id": "CONT13CJ_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True, "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off", "entry_strategy": "Daily",
    },
    'C101': {
        "name": "Min MR-13AO No-K No-R0 / Broker-Cash",
        "description": "SR-C101: AO 40D score, P16 Min broker-cash; direct allocation without K/R0.",
        "param_source": "min_best_broker_cash_rolling_experiment",
        "param_policy": "base-finalist-best", "rule_policy": "all_off",
        "dl_enabled": True, "dl_id": "CONT13AO_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False, "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
}

# AI: Controlled candidates share the exact C123 execution contract.
from config.learning_studies import LEARNING_STUDIES as _LEARNING_STUDIES
for _study in _LEARNING_STUDIES.values():
    _source_id = "CONT" + _study["model_id"].removeprefix("MR-") + "_ROLL"
    STRATEGY_DL_SOURCES[_source_id] = {
        **STRATEGY_DL_SOURCES[STRATEGY_COMPARE_ARMS[_study["control_arm"]]["dl_id"]],
        "model_architecture": _study["architecture"], "experiment_profile": _study["profile"],
        "description": _study["label"],
    }
    STRATEGY_COMPARE_ARMS[_study["arm_id"]] = {
        **STRATEGY_COMPARE_ARMS[_study["control_arm"]],
        "name": _study["label"] + " / P16 Exact", "description": _study["label"], "dl_id": _source_id,
    }

# AI: Materialize mode-specific parameter-source overrides from the canonical
# arm definitions, after those definitions exist. This keeps Full P17 and Min
# P16 in the same generic Compare Suite without per-arm dispatch exceptions.
for _mode in ("oos", "rolling"):
    _profile_param_sources = STRATEGY_COMPARE_PROFILES[
        f"broker_cash_dl_conversion_{_mode}"
    ]["arm_param_source_overrides"]
    for _arm_id in dict.fromkeys((*_BROKER_CASH_DL_ARM_IDS, *get_research_summary_strategy_arm_ids())):
        _registered_source = str(STRATEGY_COMPARE_ARMS[_arm_id]["param_source"])
        if not _registered_source.endswith("_rolling_experiment"):
            raise ValueError(f"Unrecognized rolling parameter source: {_arm_id}/{_registered_source}")
        _profile_param_sources[_arm_id] = _registered_source.replace(
            "_rolling_experiment", f"_{_mode}_experiment"
        )

# =============================================================================
# 6. Contrast definitions
# =============================================================================
# Contrast 是否啟用只由 STRATEGY_COMPARE_PROFILES[*]["contrast_ids"] 決定。

STRATEGY_COMPARE_CONTRASTS = {
    "C189-C96": {"left": "C189", "right": "C96", "description": "CI R40 rank plus DL Positive Exact versus same-P16 no-DL control"},
    "C190-C96": {"left": "C190", "right": "C96", "description": "CI R40 rank plus DL Positive Direct versus same-P16 no-DL control"},
    "C189-C190": {"left": "C189", "right": "C190", "description": "Same CI R40 rank and independent DL Positive: Exact versus Direct"},
    "C189-C106": {"left": "C189", "right": "C106", "description": "Same CI R40 Exact: with versus without independent R40 Positive admission"},
    "C190-C118": {"left": "C190", "right": "C118", "description": "Same CI R40 Direct: with versus without independent R40 Positive admission"},
    "C97-C96": {"left": "C97", "right": "C96", "description": "same-fee broker cash Full vs Min overall policy comparison"},
    "C155-C97": {"left": "C155", "right": "C97", "description": "Full/P17 CZ/DA Exact versus Full DL-off C97"},
    "C156-C97": {"left": "C156", "right": "C97", "description": "Full/P17 CZ/DA Direct versus Full DL-off C97"},
    "C155-C156": {"left": "C155", "right": "C156", "description": "Full/P17 CZ+DA same scores, Exact versus Direct"},
    "C155-C153": {"left": "C155", "right": "C153", "description": "Full/P17 Exact: NetPnL Positive versus T40"},
    "C156-C154": {"left": "C156", "right": "C154", "description": "Full/P17 Direct: NetPnL Positive versus T40"},
    "C157-C97": {"left": "C157", "right": "C97", "description": "Full/P17 CZ NetPnL rank-only Exact versus same-Full no-DL C97"},
    "C158-C97": {"left": "C158", "right": "C97", "description": "Full/P17 CZ NetPnL rank-only Direct versus same-Full no-DL C97"},
    "C157-C158": {"left": "C157", "right": "C158", "description": "Same Full/P17 CZ NetPnL scores: Exact versus Direct selector"},
    "C157-C153": {"left": "C157", "right": "C153", "description": "Full/P17 Exact: NetPnL continuous ranking versus T40 ranking"},
    "C158-C154": {"left": "C158", "right": "C154", "description": "Full/P17 Direct: NetPnL continuous ranking versus T40 ranking"},
    "C186-C183": {"left": "C186", "right": "C183", "description": "Same DH rank and P21 economics; second Positive DL payoff-learned action versus win-probability gate"},
    "C153-C97": {"left": "C153", "right": "C97", "description": "Full P17 CJ T40 Exact vs same-Full no-DL control"},
    "C154-C97": {"left": "C154", "right": "C97", "description": "Full P17 CJ T40 Direct vs same-Full no-DL control"},
    "C153-C107": {"left": "C153", "right": "C107", "description": "CJ T40 Exact: Full P17 vs Min P16"},
    "C154-C109": {"left": "C154", "right": "C109", "description": "CJ T40 Direct: Full P17 vs Min P16"},
    "C154-C153": {"left": "C154", "right": "C153", "description": "Full P17 CJ T40: Direct vs Exact selector"},
    "C98-C97": {"left": "C98", "right": "C97", "description": "same-fee broker cash Full corrected 4x300 agree vs single-seed best"},
    "C98-C96": {"left": "C98", "right": "C96", "description": "same-fee broker cash corrected Full 4x300 vs Min best full system comparison"},
    # Broker-cash DL conversion: a non-redundant contrast graph.  Every
    # unlisted six-arm pair remains derivable from the one common main report.
    "C103-C96": {"left": "C103", "right": "C96", "description": "CG Exact K/R0 vs same-P16 broker-cash DL-off control"},
    "C106-C96": {"left": "C106", "right": "C96", "description": "CI Exact K/R0 vs same-P16 broker-cash DL-off control"},
    "C107-C96": {"left": "C107", "right": "C96", "description": "CJ Exact K/R0 vs same-P16 broker-cash DL-off control"},

    # Same-parameter Broker-Cash controls and score-vs-selector separations.
    "C109-C96": {"left": "C109", "right": "C96", "description": "CJ T40 Direct against matched P16 DL-off baseline"},
    "C111-C96": {"left": "C111", "right": "C96", "description": "K Pure-MFE Exact against matched P16 DL-off baseline"},
    "C112-C96": {"left": "C112", "right": "C96", "description": "K Pure-MFE Direct against matched P16 DL-off baseline"},
    "C109-C107": {"left": "C109", "right": "C107", "description": "Same CJ T40 score/P16: Direct versus Exact selector"},
    "C111-C112": {"left": "C111", "right": "C112", "description": "Same K Pure-MFE score/P16: Exact versus Direct selector"},
    "C118-C96": {"left": "C118", "right": "C96", "description": "CI terminal R40 Direct No-K/No-R0 versus matched P16 broker-cash DL-off reference"},
    "C106-C118": {"left": "C106", "right": "C118", "description": "Same MR-13CI terminal R40 PIT score and P16 broker cash: Exact K/R0 versus Direct No-K/No-R0"},
    "C123-C96": {"left": "C123", "right": "C96", "description": "Full-cycle Min NetPnL PIT model versus C96 Exact"},
    "C124-C96": {"left": "C124", "right": "C96", "description": "Full-cycle Min NetPnL PIT model versus C96 Direct"},
    "C125-C96": {"left": "C125", "right": "C96", "description": "CO ranking plus independent CP Positive, Exact versus C96"},
    "C126-C96": {"left": "C126", "right": "C96", "description": "CO ranking plus independent CP Positive, Direct versus C96"},
    "C125-C126": {"left": "C125", "right": "C126", "description": "Same CO rank and CP binary gate: Exact versus Direct"},
    "C147-C107": {"left": "C147", "right": "C107", "description": "Same T40/P16 Breakout Exact: probabilistic pairwise confidence CX versus CJ"},
    "C148-C109": {"left": "C148", "right": "C109", "description": "Same T40/P16 Breakout Direct: probabilistic pairwise confidence CX versus CJ"},
    "C147-C148": {"left": "C147", "right": "C148", "description": "CX Breakout Exact versus Direct"},
    "C123-C125": {"left": "C123", "right": "C125", "description": "CO rank without/with independent CP classifier: Exact"},
    "C124-C126": {"left": "C124", "right": "C126", "description": "CO rank without/with independent CP classifier: Direct"},
    "C123-C124": {"left": "C123", "right": "C124", "description": "Same MR-13CO PIT score, Exact versus Direct"},
}


__all__ = [
    "STRATEGY_COMPARE_SCHEMA_VERSION",
    "STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT",
    "STRATEGY_COMPARE_ROLLING_TEST_MODES",
    "STRATEGY_COMPARE_SUITE_INVARIANTS",
    "STRATEGY_COMPARE_SUITES",
    "STRATEGY_COMPARE_PROFILES",
    "STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES",
    "STRATEGY_PARAM_SOURCES",
    "STRATEGY_EXPERIMENTAL_PARAM_SOURCES",
    "STRATEGY_DL_SOURCES",
    "STRATEGY_COMPARE_ARMS",
    "STRATEGY_COMPARE_CONTRASTS",
]

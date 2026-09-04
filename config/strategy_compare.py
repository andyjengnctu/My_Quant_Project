"""Strategy Compare 使用者／執行設定。

本模組只保存可調的 current profile、執行資源、工件保留、診斷與 promotion-gate knobs。
Scientific Compare Suite／arm／contrast／artifact-source catalog 由
``core.strategy_compare_registry`` 唯一持有；resolver／validation 由
``core.strategy_compare_policy`` 持有。
"""

from __future__ import annotations

from config.breakout_quality import (
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
)

# Current work-stage selection.
STRATEGY_COMPARE_DEFAULT_PROFILE = "extending_window_oos"
STRATEGY_COMPARE_MENU_PROFILE_IDS = (
    "extending_window_oos",
    "extending_window_rolling",
)
STRATEGY_COMPARE_ROLLING_TEST_MENU_LABEL = "Extending-Window Test"
STRATEGY_COMPARE_ROBUSTNESS_MENU_LABEL = "Extending-Window Multi-seed Robustness Test"
STRATEGY_COMPARE_DEFAULT_ROBUSTNESS_PROFILE = "extending_window_oos"

# Execution / retention knobs.
STRATEGY_COMPARE_GPU_TRAIN_WORKERS = 1
STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS = 60.0
STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS = 1
STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED = True
STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE = "compact"
STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT = True
STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS = False
STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES = False
STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS = False
STRATEGY_COMPARE_ROBUSTNESS_KEEP_ATTRIBUTION_SOURCE = True

# Preparation behavior unique to Strategy Compare. Generic research preparation
# remains owned by config/research.py and is combined by the policy resolver.
STRATEGY_COMPARE_REUSE_COMPLETED_RESULTS = True
STRATEGY_COMPARE_REUSE_SHARED_BASELINE = True

# Read-only path-conversion diagnostics. These settings affect reporting only and
# must never invalidate portfolio replay.
STRATEGY_COMPARE_UPSIDE_REALIZATION_R_THRESHOLDS = (1.0, 2.0, 3.0)
STRATEGY_COMPARE_UPSIDE_REALIZATION_ADVERSE_BUCKET_EDGES_R = (0.5, 1.0)
STRATEGY_COMPARE_UPSIDE_REALIZATION_PATH_PROFILE = (
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
)

# Read-only main-report MFE × Safety geometry.
STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_ENABLED = True
STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PROFILE_IDS = (
    "extending_window_oos",
    "extending_window_rolling",
)
STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PROFILE = (
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
)
STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PERCENTILE_CUTOFF = 0.50
STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PERCENTILE_METHOD = "average_zero_based"

# Runtime promotion reads existing formal artifacts only; it never trains, replays,
# or changes the production default automatically.
STRATEGY_RUNTIME_INTEGRATION = {
    "label": "Runtime Promotion Gate",
    "enabled": False,
    "selection_profile_id": "selection_pit",
    "forward_profile_id": "forward_oos",
    "selection_candidate_arm_id": "C42",
    "forward_candidate_arm_id": "C44",
    "selection_robustness_id": "selection_pit",
    "forward_robustness_id": "forward_oos",
    "output_root": "outputs/strategy_compare/runtime_integration",
    "comparison_anchor_experiment_profile": "strategy_aligned_no_time_all_event_pairwise",
    "max_selector_latency_ms": 10000.0,
    "require_strict_romd_majority": True,
}

__all__ = [name for name in globals() if name.startswith("STRATEGY_")]

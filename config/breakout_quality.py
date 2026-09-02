"""Single user-facing source of truth for breakout-quality configuration.

Edit only the user-settings section at the top of this file. Named scientific/profile
declarations, validation, derived values, and helper functions remain centralized here.
Generic continuous-ranker execution capability contracts live in
``config.breakout_quality_runtime`` and are re-exported below for compatibility; that module
is not user-adjustable configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from config.research import RESEARCH_SINGLE_SEED
from config.breakout_policy import (
    BREAKOUT_DEFAULT_HIGH_LEN,
    build_breakout_optimizer_high_len_values,
)
from config.training_policy import OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
from config.execution_policy import (
    DEFAULT_FIXED_RISK,
    DEFAULT_MAX_POSITION_CAP_PCT,
    DEFAULT_PORTFOLIO_MAX_POSITIONS,
    DEFAULT_PORTFOLIO_ROTATION,
)
from config.breakout_quality_runtime import (
    PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID,
    PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID,
    PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    CONTINUOUS_RANKER_TRAINER_EVENT,
    CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE,
    CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE,
    CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT,
    CONTINUOUS_RANKER_CONTEXT_ROLE_TARGET_TRANSFORM,
    CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
    SUPPORTED_CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPES,
    SUPPORTED_CONTINUOUS_RANKER_PAIR_WEIGHT_POLICIES,
    get_continuous_ranker_pair_weight_policy,
    CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR,
    CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_PARETO_COMPONENTS,
    CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT,
    SUPPORTED_CONTINUOUS_RANKER_PAIRWISE_REDUCTIONS,
    ContinuousRankerContextPolicy,
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_EXTERNAL,
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_RISK_NORMALIZED,
    CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE,
    CONTINUOUS_RANKER_TARGET_POSTPROCESS_EQUAL_RANK_MFE_LOW_ADVERSE,
    CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_NONE,
    CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_UPSIDE_LOW_ADVERSE,
    CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_SAFETY_MFE,
    CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PRESERVE_PURE_MFE,
    CONTINUOUS_RANKER_BATCH_MODE_SHUFFLED,
    CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
    CONTINUOUS_RANKER_TARGET_BUILDER_PERCENTILE,
    CONTINUOUS_RANKER_TARGET_BUILDER_SCALAR_PAIRWISE,
    CONTINUOUS_RANKER_TARGET_BUILDER_RAW_R,
    CONTINUOUS_RANKER_TARGET_BUILDER_DUAL_COMPONENT_R,
    CONTINUOUS_RANKER_TARGET_BUILDER_PARETO_COMPONENTS,
    CONTINUOUS_RANKER_TARGET_BUILDER_CONDITIONAL_MFE_SAFETY,
    CONTINUOUS_RANKER_TARGET_BUILDER_CONDITIONAL_MFE_SINGLE,
    CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE,
    CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_PRIMARY,
    CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_TARGET_BUILDER_HS_PRIORITY_MFE,
    CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE_HMHS,
    CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE_JOINT_MIN,
    CONTINUOUS_RANKER_TARGET_BUILDER_DIRECT_HMHS,
    CONTINUOUS_RANKER_LOSS_HANDLER_PERCENTILE_MSE,
    CONTINUOUS_RANKER_LOSS_HANDLER_RAW_R,
    CONTINUOUS_RANKER_LOSS_HANDLER_DUAL_COMPONENT_R,
    CONTINUOUS_RANKER_LOSS_HANDLER_SINGLE_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_CONDITIONAL_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_WEIGHTED_MFE_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_HS_QUALIFICATION_SCOPED_MFE_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_DUAL_SUPERVISED_HS_SCOPED_MFE_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_TOP_HS_SAFETY_SCOPED_MFE_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_JOINT_TRI_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_LISTWISE,
    CONTINUOUS_RANKER_AUX_TARGET_NONE,
    CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_SAFETY,
    CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY,
    CONTINUOUS_RANKER_SEMANTICS_DEFAULT,
    CONTINUOUS_RANKER_SEMANTICS_PAIRWISE,
    CONTINUOUS_RANKER_SEMANTICS_LISTWISE,
    CONTINUOUS_RANKER_SEMANTICS_RAW_R,
    CONTINUOUS_RANKER_SEMANTICS_DUAL_COMPONENT_R,
    CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SAFETY,
    CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SINGLE,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_PRIMARY,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_HMHS,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_JOINT_MIN,
    CONTINUOUS_RANKER_SEMANTICS_DIRECT_HMHS,
    CONTINUOUS_RANKER_SCORE_TRANSFORM_PROBABILITY,
    CONTINUOUS_RANKER_SCORE_TRANSFORM_MARGIN_R,
    CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
    CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_WEIGHTED,
    ContinuousRankerTargetPolicy,
    ContinuousRankerTrainingPolicy,
    ContinuousRankerObjectivePolicy,
    ContinuousRankerDependencySpec,
    BreakoutQualityOutputSchema,
    BREAKOUT_QUALITY_OUTPUT_SCHEMA,
    ContinuousRankerExecutionRecipe,
    get_continuous_ranker_training_policy,
    build_continuous_ranker_execution_recipe,
)


# =============================================================================
# USER SETTINGS — edit this section only
# =============================================================================

# =============================================================================
# 0. Main menu workflow selector
# =============================================================================

# This is the only setting normally changed to switch the main menu model workflow.
# - 9A binary filter: "unique_group_sampling"
# - continuous PIT ranker (PASS-only): "strategy_aligned_no_time_pass_magnitude_mse"
# - MR-12A all-event continuous MSE: "strategy_aligned_no_time_all_event_mse"
# - MR-12B all-event pairwise ranker: "strategy_aligned_no_time_all_event_pairwise"
# - MR-12C all-event ListNet top-one listwise ranker: "strategy_aligned_no_time_all_event_listwise"
# - MR-13A daily-universal equal-pair ranker: "daily_universal_no_time_pairwise"
# - MR-13B daily-universal target-gap-weighted pairwise ranker: "daily_universal_no_time_pairwise_gap_weighted"
# - MR-13C daily-universal percentile regression: "daily_universal_no_time_percentile_mse"
# - MR-13D daily-universal upper-tail relevance pairwise ranker: "daily_universal_no_time_upper_tail_pairwise"
# - MR-13E daily-universal full-list delta-NDCG pairwise ranker: "daily_universal_no_time_full_list_ndcg_pairwise"
# - MR-13F daily-universal direct-R Huber regression: "daily_universal_no_time_r_huber"
# - MR-13G daily-universal direct-R mean/MSE regression: "daily_universal_no_time_r_mse"
# - MR-13H daily-universal full-horizon no-breach full-list ranker: "daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise"
# - MR-13K daily-universal full-horizon pure-MFE full-list ranker: "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise"
# - MR-13L daily-universal full-horizon decomposed MFE/adverse regression: "daily_universal_full_horizon_mfe_adverse_dual_mse"
# - MR-13M daily-universal full-horizon low-adverse full-list ranker: "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise"
# - MR-13N daily-universal full-horizon equal-rank MFE + low-adverse full-list ranker: "daily_universal_full_horizon_equal_rank_mfe_low_adverse_full_list_ndcg_pairwise"
# - MR-13O daily-universal full-horizon Pareto-dominance pairwise ranker: "daily_universal_full_horizon_pareto_mfe_low_adverse_pairwise"
# - MR-13P single-model conditional MFE–Safety full-list ranker: "daily_universal_conditional_mfe_safety_full_list_ndcg_pairwise"
# - MR-13Q reverse-conditional MFE single-head ranker: "daily_universal_conditional_mfe_single_head_full_list_ndcg_pairwise"
# - MR-13R reverse-conditional MFE duo-head ranker: "daily_universal_safety_conditional_mfe_duo_head_full_list_ndcg_pairwise"
# - MR-13S safety-conditioned absolute MFE duo-head ranker: "daily_universal_safety_raw_mfe_duo_head_full_list_ndcg_pairwise"
# - MR-13T raw-data direct HM/HS tri-head ranker: "daily_universal_safety_raw_mfe_hmhs_tri_head_full_list_ndcg_pairwise"
# - MR-13U raw-data direct HM/HS H-only single-head control: "daily_universal_hmhs_single_head_full_list_ndcg_pairwise"
# - MR-13V MR-13T control + nonlinear Direct HM/HS MLP head: "daily_universal_safety_raw_mfe_hmhs_mlp_head_full_list_ndcg_pairwise"
# - MR-13W MR-13V architecture + continuous joint-min target: "daily_universal_safety_raw_mfe_joint_min_mlp_head_full_list_ndcg_pairwise"
# - MR-13X MR-13W target + Joint-Min learned temporal attention pooling: "daily_universal_safety_raw_mfe_joint_min_attn_pool_mlp_head_full_list_ndcg_pairwise"
# - MR-13Y Joint-Min ModernTCN raw-data architecture comparison: "daily_universal_safety_raw_mfe_joint_min_modern_tcn_attn_pool_mlp_head_full_list_ndcg_pairwise"
# - MR-13Z Joint-Min Patch Transformer raw-data architecture comparison: "daily_universal_safety_raw_mfe_joint_min_patch_transformer_attn_pool_mlp_head_full_list_ndcg_pairwise"
# - MR-13AA MR-13H exact target + frozen Patch Transformer architecture control: "daily_universal_full_horizon_no_breach_patch_transformer_full_list_ndcg_pairwise"
# - MR-13AB Pure-MFE before first risk breach target control: "daily_universal_first_risk_breach_pure_mfe_full_list_ndcg_pairwise"
# - MR-13AC PIT-safe predicted-upside conditional low-adverse ranker: "daily_universal_predicted_upside_conditional_low_adverse_full_list_ndcg_pairwise"
# - MR-13AD PIT-safe predicted-safety conditional MFE reverse-control: "daily_universal_predicted_safety_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13AE PIT-safe predicted-safety context + canonical Pure-MFE control: "daily_universal_predicted_safety_context_pure_mfe_full_list_ndcg_pairwise"
# - MR-13AF PIT-safe high-Safety-weighted canonical Pure-MFE ranker: "daily_universal_predicted_safety_weighted_pure_mfe_full_list_ndcg_pairwise"
# - MR-13AG PIT-safe MFE-winner-Safety-weighted canonical Pure-MFE ranker: "daily_universal_predicted_safety_winner_weighted_pure_mfe_full_list_ndcg_pairwise"
# - MR-13AH PIT-safe Safety-product-weighted canonical Pure-MFE ranker: "daily_universal_predicted_safety_product_weighted_pure_mfe_full_list_ndcg_pairwise"
# - MR-13AJ PIT-safe conflict-only unsafe-winner-discounted Pure-MFE ranker: "daily_universal_predicted_safety_conflict_discounted_pure_mfe_full_list_ndcg_pairwise"
# - MR-13AK A1 Shared-AH: shared Safety/MFE encoder with detached Safety-product MFE loss weight: "daily_universal_shared_safety_weighted_pure_mfe_full_list_ndcg_pairwise"
# - MR-13AL A2 Shared-AH + Safety Context: A1 loss fixed; final MFE head additionally receives detach(Safety): "daily_universal_shared_safety_context_weighted_pure_mfe_full_list_ndcg_pairwise"
# - MR-13AM A3 AK shared architecture + H economic target: "daily_universal_shared_safety_weighted_full_horizon_opportunity_full_list_ndcg_pairwise"
# - MR-13AN AM + A2 Safety Context: economic target fixed; final head additionally receives detach(Safety): "daily_universal_shared_safety_context_weighted_full_horizon_opportunity_full_list_ndcg_pairwise"
# - MR-13AO true-HS Conditional-MFE: full-universe Safety supervision + true-HS-only MFE list supervision: "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13AP HS-Priority MFE: LS fixed worst; HS ranked by within-HS MFE on the full daily list: "daily_universal_shared_safety_hs_priority_mfe_full_list_ndcg_pairwise"
# - MR-13AQ AP truth + pair-stratified normalization: HS↔LS and HS↔HS Delta-NDCG strata normalized separately then 1:1 mean: "daily_universal_shared_safety_hs_priority_stratified_mfe_full_list_ndcg_pairwise"
# - MR-13AR direct HS qualification + AO true-HS Conditional-MFE: "daily_universal_shared_hs_qualification_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13AS boundary-focused HS qualification + true-HS Conditional-MFE: "daily_universal_shared_hs_boundary_weighted_qualification_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13AT dual-supervised Safety/HS qualification + true-HS Conditional-MFE: "daily_universal_shared_dual_supervised_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13AU Top-HS decision-aligned Safety NDCG@K + true-HS Conditional-MFE: "daily_universal_shared_top_hs_safety_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13I daily-universal canonical-cost risk-normalized 40D NDCG ranker: "daily_universal_risk_normalized_net_full_list_ndcg_pairwise"
# - MR-13J MR-13I target + explicit universal risk/economic geometry context: "daily_universal_risk_context_net_full_list_ndcg_pairwise"
# Runtime Integration Gate 於 2026-08-15 正式 GO；MR-13E 成為 production workflow anchor。
BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE = "daily_universal_no_time_full_list_ndcg_pairwise"
# Model-research menu may move ahead of strategy deployment. Active research models
# must not silently change strategy defaults or the deployed strategy PIT identity.
# Keep the user-facing MR identity and executable profile together as one canonical pair:
# [1]/[2] consume the profile from this pair, while [3]~[6] MUST derive their list
# from the same pair plus explicit historical/reference controls.  This makes
# "trainable current DL => present in every compare/robustness list" an invariant
# instead of a manual synchronization step whenever a new DL becomes the research focus.
BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE = (
    "MR-13AU",
    "daily_universal_shared_top_hs_safety_conditional_mfe_full_list_ndcg_pairwise",
)
# Compatibility alias for call sites that only need the executable profile slug.
BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE = (
    BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE[1]
)

# (AI註: Breakout-quality全部正式模型流程共用此Seed；CLI --seed只作單次覆寫。)
BREAKOUT_QUALITY_RANDOM_SEED = RESEARCH_SINGLE_SEED

# Daily-universal batch PIT Gate：只放「可共用同一Target/period/fold contract」的profile。
# 單一profile是否已被Forward證據授權PIT，改由ContinuousRankerResearchSpec.selection_pit_authorized控制；
# 此清單只控制模型研究選單的batch Selection PIT比較，不改Strategy workflow/runtime source。
BREAKOUT_QUALITY_CONTINUOUS_RANKER_PIT_GATE_PROFILES = (
    ("MR-13C", "daily_universal_no_time_percentile_mse"),
    ("MR-13D", "daily_universal_no_time_upper_tail_pairwise"),
    ("MR-13E", "daily_universal_no_time_full_list_ndcg_pairwise"),
)


# =============================================================================
# 1. Active model identity and runtime decision defaults
# =============================================================================

BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "inception_time_v1"  # 9A accepted：目前排序／高品質研究基準；10A Candidate-conditioned Market Set 已由完整 OOS 淘汰。
BREAKOUT_QUALITY_EXPERIMENT_PROFILE = "unique_group_sampling"  # 8F accepted 基準：每個 unique ticker/date group 每個 epoch 只參與一次 optimizer sampling。
BREAKOUT_QUALITY_DEFAULT_FILTER_ID = "breakout_quality_v1"  # 9A正式Dataset與模型工件根路徑；Market Set實驗工件保留於獨立legacy filter id。
BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD = 0.50  # 在查看 OOS 前鎖定的 PASS 分數門檻；不由 OOS 自動調整。


# =============================================================================
# 2. Dataset universe, input window, and label policy
# =============================================================================

BREAKOUT_QUALITY_FEATURE_WINDOW_BARS = 300  # 每個事件輸入模型的歷史特徵交易日數。
BREAKOUT_QUALITY_BENCHMARK_TICKER = "0050"  # 建立相對市場特徵時使用的基準 ETF 代號。

BREAKOUT_QUALITY_LABEL_HORIZON_BARS = 40  # 自突破訊號隔日起，用來判定 PASS 或 REJECT 的未來交易日數；資料不足或無效者不產生有效 Label。
BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS = 120  # 快取每個 ticker/date 的未來 K 線路徑長度；調整門檻或不超過此值的 horizon 時只需快速 relabel。
BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN = 0.05  # PASS 至少要求的最大有利漲幅（MFE）；必須嚴格大於此值。
BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO = 2.0  # PASS 的最低 MFE／MAE；必須嚴格大於 1，且實際判定也採嚴格大於。
BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN = -0.10  # 最大容許不利跌幅；Low 觸及或跌破此值即保守標記為 REJECT。

# Dataset 必須涵蓋 optimizer 搜尋網格，並可額外納入正式預設值或其他指定 high_len。
BREAKOUT_QUALITY_EXTRA_HIGH_LENS = (BREAKOUT_DEFAULT_HIGH_LEN,)


# =============================================================================
# 3. Active architecture settings: InceptionTime
# =============================================================================

# TARGET 是希望模型至少覆蓋的時間範圍；程式會依 depth 自動產生 3 個近似
# 1x／1/2x／1/4x 的正奇數 kernels，並將實際 receptive field 寫入 model manifest。
# 現行 target=228、depth=6 會得到 kernels=(39, 19, 9)，實際 receptive field=229。
# 若要測完整約 600 bars，可同時設定 FEATURE_WINDOW_BARS=600、
# TARGET_RECEPTIVE_FIELD_BARS=600；depth=6 時會得到 kernels=(101, 51, 25)，
# 實際 receptive field=601。
BREAKOUT_QUALITY_INCEPTION_DEPTH = 6
BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS = 228
BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY = 3

# Stage 0／1 learned market-set branch settings。Global-query v1與10A Candidate-conditioned v1
# 均已由完整 OOS 淘汰，只保留legacy checkpoint／manifest嚴格重建與歷史研究重現。
BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS = 300
BREAKOUT_QUALITY_MARKET_SET_BASE_FEATURES = (
    "close_return",
    "overnight_return",
    "intraday_return",
    "high_low_range",
    "log_volume_change",
)
BREAKOUT_QUALITY_MARKET_SET_STOCK_EMBEDDING_DIM = 32
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_CHANNELS = 32
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_KERNEL_SIZE = 15
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_STRIDE = 10
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS = (1, 2, 4)
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION = "group_norm"
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION_GROUPS = 8
BREAKOUT_QUALITY_MARKET_SET_QUERY_COUNT = 4  # Legacy Global Market Set v1 learned queries。
BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT = 1  # 10A每個候選事件產生一個動態query。
BREAKOUT_QUALITY_MARKET_SET_ATTENTION_HEADS = 4
BREAKOUT_QUALITY_MARKET_SET_EMBEDDING_DIM = 128
BREAKOUT_QUALITY_MARKET_SET_FUSION_HIDDEN_DIM = 128
BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO = 0.80
BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS = 0  # 0 = 使用當時資料集中全部股票；正整數可作資源受控實驗。
BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH = 4  # 每個 market microbatch 最多物化幾個日期；optimizer logical batch 仍固定使用 DEFAULT_BATCH_SIZE 個事件。


# =============================================================================
# 4. Supervised training and optimization
# =============================================================================

BREAKOUT_QUALITY_DEFAULT_EPOCHS = 200  # 關閉 inner validation 時為固定訓練輪數；開啟時為 epoch 搜尋上限。
BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE = 128  # 8F accepted 基準；unique_group_sampling 時代表 128 個 unique ticker/date groups。8H batch 64 已淘汰。
BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE = 0.0003  # optimizer 的預設 learning rate；9A首輪沿用8F以隔離架構差異。
BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY = 0.0001  # optimizer weight decay；0 表示關閉。Adam 為 coupled L2，AdamW 為 decoupled weight decay。
BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM = 1.0  # 每次更新前的全域 gradient norm 上限；0 表示關閉。
BREAKOUT_QUALITY_CLASS_WEIGHT_MODE = "none"  # Cross-entropy 類別權重；none 不平衡補償，inverse_frequency 依訓練資料加權。PASS／REJECT 接近均衡時建議 none。
BREAKOUT_QUALITY_TIME_WEIGHT_MODE = "none"  # 8F accepted 基準；8K date-balanced training 已由完整 OOS 淘汰。
BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES = 20  # 開始訓練前要求的最少有效 train rows。


# =============================================================================
# 5. Inner validation, early stopping, and final refit
# =============================================================================

BREAKOUT_QUALITY_USE_INNER_VALIDATION = True  # 是否以 Selection 尾端資料選 best epoch；正式模型後續採 best checkpoint 或完整 Selection refit，由 FINAL_REFIT_MODE 決定。
BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS = 24  # Inner validation 從 Selection 結尾往前保留的月份數。
BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE = 1  # 8F accepted 基準；Validation loss 連續 1 個 epoch 未改善即停止，避免 unique-group training 迅速重新過擬合。
BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA = 0.0  # Validation loss 至少下降多少才視為新最佳 epoch。
BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES = 20  # 開啟 inner validation 時要求的最少有效 validation rows。
BREAKOUT_QUALITY_FINAL_REFIT_MODE = "selected_epochs"  # 8F accepted 基準：Inner Validation 選出 epoch 後，重新初始化並以完整 eligible Selection 重訓相同 epoch 數。


# =============================================================================
# 6. Device, numerical reproducibility, and data-loading performance
# =============================================================================

BREAKOUT_QUALITY_TORCH_DEVICE = "auto"  # auto 優先使用 CUDA；CUDA 不可用時退回 CPU。
BREAKOUT_QUALITY_USE_MIXED_PRECISION = True  # 僅在 CUDA 啟用；auto dtype 優先 bfloat16，否則 float16。
BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE = "auto"  # auto／bfloat16／float16。
BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS = True  # 固定 PyTorch deterministic algorithms 與 cuDNN deterministic。
BREAKOUT_QUALITY_ALLOW_TF32 = False  # 保持跨裝置數值契約；不使用 TF32。
BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES = 0  # 訓練時預先準備後續 batches 的數量；RAM preload 開啟時預設 0，慢速磁碟可自行調高；不改 batch 順序或 optimizer 更新。
BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK = True  # 訓練與分數匯出前將去重 feature bank 與小型事件陣列載入 RAM；資料值與列順序不變。
BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES = 8  # Continuous/daily ranker CPU feature feeding queue；只預先物化後續batch，不改batch order、loss或optimizer step。
BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS = 4  # CPU feature materialization workers；結果仍按原batch順序消費，僅提升GPU feeding。


# =============================================================================
# 7. Evaluation and score-export performance
# =============================================================================

BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE = 4096  # Train／Validation／Selection 完整評估與分數匯出的分批大小；不抽樣、不改模型更新或輸出列序。
BREAKOUT_QUALITY_EVALUATION_WORKERS = 4  # CPU評估可平行；CUDA固定使用單一GPU serial batches，batch與最終列序不變。
BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION = False  # 同時評估 Inner Train 與 Validation；峰值最多使用 2 × EVALUATION_WORKERS，只做 read-only inference。
BREAKOUT_QUALITY_PIT_EPOCH_SELECTION_LIGHTWEIGHT_METRICS = True  # PIT每個epoch只計算真正參與checkpoint selection的Validation metric；完整Joint/geometry diagnostics保留於fold正式score/audit，避免GPU等待CPU O(N²)診斷。


# =============================================================================
# 8. Legacy compatibility: 9C Selection-only TS2Vec pretraining
# =============================================================================

# 只供舊工件重建；不是 active architecture 的新實驗入口。
BREAKOUT_QUALITY_PRETRAINING_PROFILE = "ts2vec_selection_only"
BREAKOUT_QUALITY_PRETRAINING_STRIDE = 5  # Dataset sampling設定；每個ticker在Selection endpoint每5個交易日建立一窗。

# =============================================================================
# 9. Rolling point-in-time evaluation contract
# =============================================================================

# Rolling OOS 是正式時間泛化評估：使用當時所有合法歷史資料（expanding）
# 並按固定score fold cadence重新選epoch／refit，再只評分下一段。每筆score都必須
# 滿足label completion < score_start；Fixed-Window Rolling 已退出 current workflow，
# 僅保留既有歷史工件／研究紀錄，不再提供正式執行入口。
BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE = "2021-01-01"
# Strategy/reporting只從此日期起視為正式operational evidence；更早的合法fold可保留
# 作模型warm-up與coverage，但不強迫策略比較納入。
BREAKOUT_QUALITY_POINT_IN_TIME_COVERAGE_REFERENCE_START_DATE = "2021-01-01"
# Current Rolling與OOS使用同一可比較期間：2021-01-01起至最新合法score date。
# Rolling仍以12M calendar folds逐期refit，因此資料尾端可形成partial latest fold。
BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE: str | None = "auto"

# Current時間驗證只保留兩種執行模式：
# OOS Test固定以2021-01-01作information cutoff後的單一forward score block，
# 只訓練一次並評分2021-01-01起至最新合法score date；Rolling Test才使用12M annual folds。
BREAKOUT_QUALITY_ROLLING_TEST_MODES = {
    "oos": {
        "label": "OOS Test",
        "score_start_date": "2021-01-01",
        "score_end_date": "auto",
        # single_score_block=True時fold_months只保留CLI/manifest相容欄位，不切分score period。
        "fold_months": 12,
        "fold_anchor_date": "2021-01-01",
        "single_score_block": True,
        # 新OOS語意使用獨立namespace，不誤REUSE舊Fast 60M工件。
        "point_in_time_dirname": "point_in_time_oos_2021_forward",
    },
    "rolling": {
        "label": "Rolling Test",
        "score_start_date": BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE,
        "score_end_date": BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE,
        "fold_months": 12,
        "fold_anchor_date": None,
        "single_score_block": False,
        # None = 沿用canonical point_in_time fold store；2021～既有年度fold可直接REUSE，僅補latest缺fold。
        "point_in_time_dirname": None,
    },
}
BREAKOUT_QUALITY_DEFAULT_ROLLING_TEST_MODE = "oos"
# Backward-compatible canonical PIT cadence；current Rolling Test由同一設定衍生。
BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS = int(
    BREAKOUT_QUALITY_ROLLING_TEST_MODES["rolling"]["fold_months"]
)
BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS = 24
# None = expanding history；正整數 = 完整fit history固定最近N個calendar months，
# 且必須大於inner validation months。Operational固定為None。
BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS: int | None = None
# Canonical fitting-identity checkpoint cache shared by model Rolling and Strategy Compare.
# Exact fitting identity is still mandatory; score/report identities remain evaluation-specific.
BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT = (
    "models/research/breakout_quality/strategy_compare/"
    "extending_window/shared_fitting_checkpoints"
)

# Rolling Timing Mode：只做 execution benchmark，不改正式 PIT 工件或模型科學契約。
# 第一次執行會建立改善前 baseline；之後同設定重跑時以目前程式作 candidate，
# 比較 wall-clock 與 exact-result hashes。預設只量最晚完整年度，避免為了 benchmark
# 再跑完整 10-fold；可自行加入較早年度觀察 Extending history 成長造成的 scaling。
BREAKOUT_QUALITY_ROLLING_TIMING_SCORE_YEARS = (2025,)
# None = 跟隨目前 Model Research Active Profile／共用 Seed。若要固定 benchmark 對象可明確指定。
BREAKOUT_QUALITY_ROLLING_TIMING_EXPERIMENT_PROFILE: str | None = None
BREAKOUT_QUALITY_ROLLING_TIMING_SEED: int | None = None

BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS = 20
BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS = 20
BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS = 1
BREAKOUT_QUALITY_POINT_IN_TIME_RESUME = True


# =============================================================================
# 10. Strategy workflow defaults
# =============================================================================

BREAKOUT_QUALITY_STRATEGY_DATASET = "full"
BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY = "base-finalist-best"
BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS = DEFAULT_PORTFOLIO_MAX_POSITIONS
# Continuous-ranker品質報表預設以策略最大持倉數作Top-K邊界。
# 兩者只影響checkpoint後評估／顯示，不參與訓練、score export或selector runtime。
BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K = BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS
BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH = 3
# Read-only target comparison inspects the rank/target cliff immediately around the
# canonical adverse-return barrier. This is diagnostic only and never changes labels.
BREAKOUT_QUALITY_TARGET_COMPARISON_BARRIER_BAND_RETURN = 0.01
BREAKOUT_QUALITY_STRATEGY_ROTATION = DEFAULT_PORTFOLIO_ROTATION

# Continuous-ranker read-only comparison is config-driven.  The interactive menu must
# never hard-code experiment/model/arm IDs; it only renders this configured work item.
BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_ENABLED = True
BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_MENU_LABEL = "比較設定中的 Continuous Rankers"
BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_PROFILES = (
    ("MR-12A", "strategy_aligned_no_time_all_event_mse"),
    ("MR-12B", "strategy_aligned_no_time_all_event_pairwise"),
    ("MR-12C", "strategy_aligned_no_time_all_event_listwise"),
)
# Dynamic-K candidate universe / daily K source.
BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_REFERENCE_ARM = "C17"
# Short console summary compares left minus right.  The pair must be adjacent in the model order above.
BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_SUMMARY_PAIR = ("MR-12B", "MR-12A")
# Forward-OOS top-prefix diagnostic. Values are descriptive only and never enter training.
BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_FIXED_K_VALUES = (1, 2, 3, 5, 10)
# Explicit historical/reference controls for model comparison.  This is NOT the full
# [3]~[6] membership list: the active [1]/[2] training model is injected below from
# BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE so it cannot be forgotten when a new DL
# becomes the current research model.
BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES = (
    ("MR-13H", "daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise"),
    ("MR-13AH", "daily_universal_predicted_safety_product_weighted_pure_mfe_full_list_ndcg_pairwise"),
    ("MR-13AK", "daily_universal_shared_safety_weighted_pure_mfe_full_list_ndcg_pairwise"),
    # Current true-HS Conditional-MFE family controls.  The active model is injected
    # separately below, so the list stays H/AH/AK + completed same-family controls + current.
    ("MR-13AO", "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"),
    ("MR-13AR", "daily_universal_shared_hs_qualification_conditional_mfe_full_list_ndcg_pairwise"),
    ("MR-13AS", "daily_universal_shared_hs_boundary_weighted_qualification_conditional_mfe_full_list_ndcg_pairwise"),
    ("MR-13AT", "daily_universal_shared_dual_supervised_hs_conditional_mfe_full_list_ndcg_pairwise"),
)


def _merge_breakout_quality_model_profiles(
    *groups: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Merge ordered model/profile groups while rejecting identity collisions.

    Exact duplicates are intentionally de-duplicated.  Reusing one MR id for a different
    profile, or one profile under a different MR id, is a configuration error and fails
    before any training/comparison workflow can run.
    """

    rows: list[tuple[str, str]] = []
    by_model_id: dict[str, str] = {}
    by_profile: dict[str, str] = {}
    for group in groups:
        for raw_model_id, raw_profile in group:
            model_id = str(raw_model_id).strip()
            profile = str(raw_profile).strip()
            if not model_id or not profile:
                raise ValueError("模型研究model id/profile不得為空")
            existing_profile = by_model_id.get(model_id)
            existing_model_id = by_profile.get(profile)
            if existing_profile is not None and existing_profile != profile:
                raise ValueError(
                    f"模型研究model id重複綁定不同profile: {model_id} -> "
                    f"{existing_profile!r} / {profile!r}"
                )
            if existing_model_id is not None and existing_model_id != model_id:
                raise ValueError(
                    f"模型研究profile重複綁定不同model id: {profile} -> "
                    f"{existing_model_id!r} / {model_id!r}"
                )
            if existing_profile is not None:
                continue
            by_model_id[model_id] = profile
            by_profile[profile] = model_id
            rows.append((model_id, profile))
    return tuple(rows)


# Canonical [3]~[6] Model Compare/Test List.  Its required subset is generated from the
# same canonical pair that drives [1]/[2]; reference controls may only add membership.
# Therefore every current training DL is structurally guaranteed to appear in Forward /
# Rolling comparison and Forward / Rolling robustness without a second manual edit.
BREAKOUT_QUALITY_MODEL_TEST_PROFILES = _merge_breakout_quality_model_profiles(
    BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES,
    (BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE,),
)
BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_MENU_LABEL = "模型比較（Standard SOP）"
# Backward-compatible alias; there is no second model list.
BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_PROFILES = BREAKOUT_QUALITY_MODEL_TEST_PROFILES


def get_breakout_quality_model_workflow_profile_names() -> tuple[str, ...]:
    """Return the single configured target set for current model workflows.

    [1]/[2] consume ``BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE`` while
    [3]～[6] consume ``BREAKOUT_QUALITY_MODEL_TEST_PROFILES``.  Membership in either
    current work-item SSOT is itself the authorization for current Forward/Rolling
    model evaluation; experiment-history flags remain compatibility evidence rather
    than a second current-workflow selector.
    """

    ordered = [str(BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE).strip()]
    ordered.extend(str(profile).strip() for _model_id, profile in BREAKOUT_QUALITY_MODEL_TEST_PROFILES)
    return tuple(dict.fromkeys(value for value in ordered if value))


def is_breakout_quality_model_workflow_profile(profile_name: str) -> bool:
    normalized = normalize_breakout_quality_experiment_profile(profile_name)
    return normalized in set(get_breakout_quality_model_workflow_profile_names())


def is_breakout_quality_model_test_profile(profile_name: str) -> bool:
    """Whether a profile is selected by the one [3]～[6] compare/test SSOT."""

    normalized = normalize_breakout_quality_experiment_profile(profile_name)
    return normalized in {
        normalize_breakout_quality_experiment_profile(profile)
        for _model_id, profile in BREAKOUT_QUALITY_MODEL_TEST_PROFILES
    }


# "auto" resolves from the selected experiment profile:
# - binary classification -> hard-filter / canonical_runtime / original
# - continuous ranker     -> score-ranking / selection_point_in_time /
#                            breakout_quality_score_desc
BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE = "auto"
BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE = "auto"
BREAKOUT_QUALITY_STRATEGY_BUY_SORT = "auto"

# 正式 production runtime contract。Strategy Compare 仍用各 stage context 顯式覆寫，
# 此處只控制沒有研究 context 的 scanner／portfolio runtime。
BREAKOUT_QUALITY_RUNTIME_STRATEGY_ENABLED = True
BREAKOUT_QUALITY_RUNTIME_RANKING_POLICY = "resource-aware-continuous-score-constrained-optimal"
BREAKOUT_QUALITY_RUNTIME_RANKING_OPTIONS = {
    "preserve_k_r0": True,
    "constrained_solver": "exact_branch_and_bound_v1",
    "selection_only": False,
}

# =============================================================================
# INTERNAL PROFILE DEFINITIONS AND SUPPORTED VALUES — normally do not edit
# =============================================================================

BASELINE_EXPERIMENT_PROFILE = "baseline"
ADAMW_ONLY_EXPERIMENT_PROFILE = "adamw_only"
ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE = "adam_warmup_cosine"
HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE = "history_masking_only"
UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE = "unique_group_sampling"
UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE = "unique_group_date_balanced"
STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE = "strategy_aligned_daily_percentile_mse"
STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE = "strategy_aligned_no_time_pass_magnitude_mse"
STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE = "strategy_aligned_no_time_all_event_mse"
STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE = "strategy_aligned_no_time_all_event_pairwise"
STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE = "strategy_aligned_no_time_all_event_listwise"
DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE = "daily_universal_no_time_pairwise"
DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE = "daily_universal_no_time_pairwise_gap_weighted"
DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE = "daily_universal_no_time_percentile_mse"
DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE = "daily_universal_no_time_upper_tail_pairwise"
DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE = "daily_universal_no_time_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE = "daily_universal_no_time_r_huber"
DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE = "daily_universal_no_time_r_mse"
DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_first_risk_breach_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_upside_conditional_low_adverse_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_context_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_weighted_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_winner_weighted_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_product_weighted_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_conflict_discounted_pure_mfe_full_list_ndcg_pairwise"
)

PREDICTED_UPSIDE_CONTEXT_SCHEMA_VERSION = 1
PREDICTED_UPSIDE_CONTEXT_STAGE1_PROFILE = (
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
)
PREDICTED_UPSIDE_CONTEXT_STAGE1_ARCHITECTURE = "inception_time_v1"
PREDICTED_UPSIDE_CONTEXT_STAGE1_RESEARCH_ID = "MR-13K"
PREDICTED_UPSIDE_CONTEXT_STAGE1_SEED = 42


def get_predicted_upside_context_contract() -> dict[str, Any]:
    """Lightweight MR-13AC stacking contract shared by config/training/artifact consumers."""

    return {
        "schema_version": PREDICTED_UPSIDE_CONTEXT_SCHEMA_VERSION,
        "stage1_profile": PREDICTED_UPSIDE_CONTEXT_STAGE1_PROFILE,
        "stage1_architecture": PREDICTED_UPSIDE_CONTEXT_STAGE1_ARCHITECTURE,
        "stage1_research_id": PREDICTED_UPSIDE_CONTEXT_STAGE1_RESEARCH_ID,
        "stage1_seed": PREDICTED_UPSIDE_CONTEXT_STAGE1_SEED,
        "context_semantic": "same_date_average_rank_percentile_of_stage1_predicted_pure_mfe",
        "selection_context": "expanding_cross_fitted_point_in_time",
        "forward_context": "single_fixed_pre_oos_fit",
        "full_fit_selection_score_forbidden": True,
        "oos_statistics_for_training_forbidden": True,
        "selection_training_universe": "pit_context_covered_rows_only",
        "stage2_response": "same_date_low_adverse_percentile",
        "stage2_target": "same_date_percentile_of_low_adverse_residual_given_predicted_upside_percentile",
        "stage2_context_used_as_input": True,
    }

PREDICTED_SAFETY_CONTEXT_SCHEMA_VERSION = 1
PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE = (
    "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise"
)
PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE = "inception_time_v1"
PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID = "MR-13M"
PREDICTED_SAFETY_CONTEXT_STAGE1_SEED = 42


def get_predicted_safety_context_contract() -> dict[str, Any]:
    """MR-13AD reverse-control stacking contract shared by all consumers."""

    return {
        "schema_version": PREDICTED_SAFETY_CONTEXT_SCHEMA_VERSION,
        "stage1_profile": PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE,
        "stage1_architecture": PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE,
        "stage1_research_id": PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID,
        "stage1_seed": PREDICTED_SAFETY_CONTEXT_STAGE1_SEED,
        "context_semantic": "same_date_average_rank_percentile_of_stage1_predicted_low_adverse_safety",
        "selection_context": "expanding_cross_fitted_point_in_time",
        "forward_context": "single_fixed_pre_oos_fit",
        "full_fit_selection_score_forbidden": True,
        "oos_statistics_for_training_forbidden": True,
        "selection_training_universe": "pit_context_covered_rows_only",
        "stage2_response": "same_date_pure_mfe_percentile",
        "stage2_target": "same_date_percentile_of_pure_mfe_residual_given_predicted_safety_percentile",
        "stage2_context_used_as_input": True,
    }


# Backward-compatible canonical owner of the reusable MR-13M predicted-safety PIT context.
# MR-13AE consumes the exact same Stage-1 artifact; the owner path remains MR-13AD so the
# already-built 10+1 fold context can be reused without retraining or copying.
PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE = (
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
)
PREDICTED_SAFETY_CONTEXT_OWNER_ARCHITECTURE = "inception_time_predicted_safety_context_v1"


def get_predicted_safety_pure_mfe_contract() -> dict[str, Any]:
    """MR-13AE consumer contract: canonical Pure-MFE target plus PIT-safe safety context."""

    return {
        "schema_version": 1,
        "stage1_profile": PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE,
        "stage1_architecture": PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE,
        "stage1_research_id": PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID,
        "stage1_seed": PREDICTED_SAFETY_CONTEXT_STAGE1_SEED,
        "context_semantic": "same_date_average_rank_percentile_of_stage1_predicted_low_adverse_safety",
        "context_artifact_owner_profile": PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE,
        "selection_context": "expanding_cross_fitted_point_in_time",
        "forward_context": "single_fixed_pre_oos_fit",
        "full_fit_selection_score_forbidden": True,
        "oos_statistics_for_training_forbidden": True,
        "selection_training_universe": "pit_context_covered_rows_only",
        "stage2_response": "canonical_full_horizon_pure_mfe_r",
        "stage2_target": "exact_mr13k_pure_mfe_order_no_residualization",
        "stage2_context_used_as_input": True,
    }


def get_predicted_safety_pair_weight_contract(pair_weight_policy: str) -> dict[str, Any]:
    """Canonical Pure-MFE pair-weight contract for PIT-safe predicted-Safety policies."""

    policy = get_continuous_ranker_pair_weight_policy(pair_weight_policy)
    if policy.context_source != CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY:
        raise ValueError(
            f"pair weight policy不是predicted-Safety context: {pair_weight_policy!r}"
        )
    if not policy.contract_pair_safety_weight or not policy.contract_pair_weight_combination:
        raise ValueError(f"pair weight policy缺少scientific contract metadata: {pair_weight_policy!r}")

    return {
        "schema_version": 1,
        "stage1_profile": PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE,
        "stage1_architecture": PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE,
        "stage1_research_id": PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID,
        "stage1_seed": PREDICTED_SAFETY_CONTEXT_STAGE1_SEED,
        "context_semantic": "same_date_average_rank_percentile_of_stage1_predicted_low_adverse_safety",
        "context_artifact_owner_profile": PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE,
        "selection_context": "expanding_cross_fitted_point_in_time",
        "forward_context": "single_fixed_pre_oos_fit",
        "full_fit_selection_score_forbidden": True,
        "oos_statistics_for_training_forbidden": True,
        "selection_training_universe": "pit_context_covered_rows_only",
        "stage2_response": "canonical_full_horizon_pure_mfe_r",
        "stage2_target": "exact_mr13k_pure_mfe_order_no_residualization",
        "stage2_context_used_as_input": False,
        "pair_base_relevance": "mr13k_full_list_delta_ndcg",
        "pair_safety_weight": policy.contract_pair_safety_weight,
        "pair_weight_combination": policy.contract_pair_weight_combination,
        "pair_weight_reduction": "normalized_weighted_mean_over_comparable_same_date_pairs",
        "bucket_or_threshold": None,
        "lambda_or_temperature": None,
    }


def get_high_safety_weighted_pure_mfe_contract() -> dict[str, Any]:
    """Backward-compatible MR-13AF min-Safety pair-weight contract."""

    return get_predicted_safety_pair_weight_contract(
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY
    )
DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE = (
    "daily_universal_full_horizon_mfe_adverse_dual_mse"
)
DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_equal_rank_mfe_low_adverse_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_pareto_mfe_low_adverse_pairwise"
)
DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_conditional_mfe_safety_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_conditional_mfe_single_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_conditional_mfe_duo_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_duo_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_weighted_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_context_weighted_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_weighted_full_horizon_opportunity_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_context_weighted_full_horizon_opportunity_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_priority_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_priority_stratified_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_hs_qualification_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_hs_boundary_weighted_qualification_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_dual_supervised_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_top_hs_safety_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_hmhs_tri_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_hmhs_single_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_hmhs_mlp_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_joint_min_mlp_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_joint_min_attn_pool_mlp_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_joint_min_modern_tcn_attn_pool_mlp_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_joint_min_patch_transformer_attn_pool_mlp_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_no_breach_patch_transformer_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE = "daily_universal_risk_normalized_net_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE = "daily_universal_risk_context_net_full_list_ndcg_pairwise"

TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE = "ts2vec_selection_only"

TRAINING_SAMPLING_ALL_EVENT_ROWS = "all_event_rows_group_weighted"
TRAINING_SAMPLING_UNIQUE_TICKER_DATE = "unique_ticker_date"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES = (
    TRAINING_SAMPLING_ALL_EVENT_ROWS,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
)

TIME_WEIGHT_MODE_NONE = "none"
TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT = "year_balanced_sqrt"
TIME_WEIGHT_MODE_DATE_BALANCED = "date_balanced"
SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES = (
    TIME_WEIGHT_MODE_NONE,
    TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT,
    TIME_WEIGHT_MODE_DATE_BALANCED,
)

TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM = "batch_weight_sum"
TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE = "fixed_batch_size"

TRAINING_LABEL_SCOPE_ALL = "all_labels"
TRAINING_LABEL_SCOPE_PASS_ONLY = "pass_only"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES = (
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
)

TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS = "breakout_event_groups"
TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS = "daily_eligible_stock_days"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLE_SCOPES = (
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
)

TRAINING_OBJECTIVE_BINARY_CLASSIFICATION = "binary_classification"
CONTINUOUS_RANKER_TRAINING_OBJECTIVES = (
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
)
SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES = (
    TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
    *CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
)
SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS = (
    TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM,
    TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
)

LR_SCHEDULE_NONE = "none"
LR_SCHEDULE_LINEAR_WARMUP_COSINE = "linear_warmup_cosine"
AUGMENTATION_NONE = "none"
AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK = "old_history_contiguous_mask"

SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS = ("adam", "adamw")
SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES = (
    LR_SCHEDULE_NONE,
    LR_SCHEDULE_LINEAR_WARMUP_COSINE,
)
SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS = (
    AUGMENTATION_NONE,
    AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK,
)


@dataclass(frozen=True)
class BreakoutQualityExperimentProfile:
    name: str
    optimizer_name: str
    lr_schedule_name: str = LR_SCHEDULE_NONE
    augmentation_name: str = "none"
    augmentation_probability: float = 0.0
    augmentation_protected_recent_bars: int = 0
    augmentation_min_mask_bars: int = 0
    augmentation_max_mask_bars: int = 0
    lr_warmup_fraction: float = 0.0
    lr_minimum_ratio: float = 1.0
    training_sampling_mode: str = TRAINING_SAMPLING_ALL_EVENT_ROWS
    time_weight_mode: str | None = None
    training_weight_reduction: str = TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM
    training_objective: str = TRAINING_OBJECTIVE_BINARY_CLASSIFICATION
    continuous_target_id: str | None = None
    loss_name: str = "cross_entropy"
    epoch_selection_metric: str = "validation_loss"
    training_label_scope: str = TRAINING_LABEL_SCOPE_ALL
    training_sample_scope: str = TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS
    raw_r_huber_delta_r: float | None = None
    model_architecture: str | None = None

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if not normalized_name or normalized_name != self.name:
            raise ValueError("experiment profile name 必須是非空白小寫名稱")
        if any(token in normalized_name for token in ("/", "\\", "\x00")):
            raise ValueError("experiment profile name 必須是安全的單一資料夾名稱")
        if self.optimizer_name not in SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS:
            raise ValueError(f"不支援的 optimizer: {self.optimizer_name!r}")
        if self.lr_schedule_name not in SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES:
            raise ValueError(f"不支援的 LR schedule: {self.lr_schedule_name!r}")
        if self.augmentation_name not in SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS:
            raise ValueError(f"不支援的 augmentation: {self.augmentation_name!r}")
        if self.training_sampling_mode not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES:
            raise ValueError(
                f"不支援的 training sampling mode: {self.training_sampling_mode!r}"
            )
        if (
            self.time_weight_mode is not None
            and self.time_weight_mode not in SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES
        ):
            raise ValueError(f"不支援的 time weight mode: {self.time_weight_mode!r}")
        if self.training_weight_reduction not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS:
            raise ValueError(
                f"不支援的 training weight reduction: {self.training_weight_reduction!r}"
            )
        if self.training_objective not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES:
            raise ValueError(f"不支援的 training objective: {self.training_objective!r}")
        if self.training_label_scope not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES:
            raise ValueError(f"不支援的 training label scope: {self.training_label_scope!r}")
        if self.training_sample_scope not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLE_SCOPES:
            raise ValueError(f"不支援的 training sample scope: {self.training_sample_scope!r}")
        if self.model_architecture is not None:
            architecture = str(self.model_architecture).strip().lower()
            if not architecture or architecture != self.model_architecture:
                raise ValueError("profile model_architecture 必須是非空白小寫名稱")
            if any(token in architecture for token in ("/", "\\", "\x00")):
                raise ValueError("profile model_architecture 必須是安全名稱")
        if self.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
            if self.continuous_target_id is not None:
                raise ValueError("binary classification profile 不得指定 continuous_target_id")
            if self.loss_name != "cross_entropy" or self.epoch_selection_metric != "validation_loss":
                raise ValueError("binary classification profile 必須使用 cross_entropy / validation_loss")
            if self.training_label_scope != TRAINING_LABEL_SCOPE_ALL:
                raise ValueError("binary classification profile 必須使用all_labels scope")
            if self.training_sample_scope != TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
                raise ValueError("binary classification profile 必須使用breakout_event_groups sample scope")
        elif self.training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
            if not str(self.continuous_target_id or "").strip():
                raise ValueError("continuous ranker profile 必須指定 continuous_target_id")
            if self.training_objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION:
                if self.loss_name != "dual_mse_raw_r":
                    raise ValueError(
                        "dual-component R regression必須使用 dual_mse_raw_r"
                    )
                expected_epoch_metric = "mean_daily_spearman"
            elif self.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
                allowed_losses = {"huber_raw_r", "mse_raw_r"}
                if self.loss_name not in allowed_losses:
                    raise ValueError(
                        "direct R regression loss不支援: "
                        f"expected one of {sorted(allowed_losses)}, actual={self.loss_name}"
                    )
                expected_epoch_metric = (
                    "validation_huber_raw_r"
                    if self.loss_name == "huber_raw_r"
                    else "validation_mse_raw_r"
                )
            else:
                expected_loss = {
                    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION: "mse",
                    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING: "pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING: "pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING: "pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING: "dual_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING: "tri_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING: "tri_head_pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING: "pairwise_logistic",
                    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING: "listnet_top_one_cross_entropy",
                }[self.training_objective]
                if self.loss_name != expected_loss:
                    raise ValueError(
                        "continuous ranker loss與training objective不一致: "
                        f"objective={self.training_objective}, expected={expected_loss}, actual={self.loss_name}"
                    )
                expected_epoch_metric = (
                    "mean_daily_pareto_pair_concordance"
                    if self.training_objective == TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING
                    else "conditional_safety_mean_daily_spearman"
                    if self.training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING
                    else "conditional_mfe_mean_daily_spearman"
                    if self.training_objective == TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING
                    else "hs_conditional_mfe_mean_daily_spearman"
                    if self.training_objective in {
                        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
                        TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
                        TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
                        TRAINING_OBJECTIVE_DAILY_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
                        TRAINING_OBJECTIVE_DAILY_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
                    }
                    else "hs_priority_mfe_mean_daily_spearman"
                    if self.training_objective in {
                        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
                        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
                    }
                    else "raw_mfe_mean_daily_spearman"
                    if self.training_objective in {
                        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
                        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
                    }
                    else "raw_mfe_mean_daily_spearman"
                    if self.training_objective in {
                        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
                        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
                    }
                    else "hmhs_pairwise_concordance"
                    if self.training_objective == TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING
                    else "mean_daily_spearman"
                )
            if self.epoch_selection_metric != expected_epoch_metric:
                raise ValueError(
                    "continuous ranker epoch selection與training objective不一致: "
                    f"objective={self.training_objective}, expected={expected_epoch_metric}, actual={self.epoch_selection_metric}"
                )
            if self.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
                if self.loss_name == "huber_raw_r":
                    if self.raw_r_huber_delta_r is None or not math.isfinite(float(self.raw_r_huber_delta_r)) or float(self.raw_r_huber_delta_r) <= 0.0:
                        raise ValueError("Huber direct R regression必須指定正有限 raw_r_huber_delta_r")
                elif self.raw_r_huber_delta_r is not None:
                    raise ValueError("MSE direct R regression不得指定 raw_r_huber_delta_r")
            elif self.raw_r_huber_delta_r is not None:
                raise ValueError("非direct R regression profile不得指定 raw_r_huber_delta_r")
            if self.training_sampling_mode != TRAINING_SAMPLING_UNIQUE_TICKER_DATE:
                raise ValueError("continuous ranker只允許 unique ticker/date sampling")
            if self.time_weight_mode not in {None, TIME_WEIGHT_MODE_NONE}:
                raise ValueError("continuous ranker不允許 time weighting")
            if self.training_weight_reduction != TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM:
                raise ValueError("continuous ranker只允許 batch_weight_sum")
        if (
            self.training_weight_reduction == TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE
            and self.time_weight_mode != TIME_WEIGHT_MODE_DATE_BALANCED
        ):
            raise ValueError(
                "fixed_batch_size training weight reduction 目前只允許 date_balanced profile"
            )
        warmup_fraction = float(self.lr_warmup_fraction)
        minimum_ratio = float(self.lr_minimum_ratio)
        if self.lr_schedule_name == LR_SCHEDULE_NONE:
            if warmup_fraction != 0.0 or minimum_ratio != 1.0:
                raise ValueError("無 LR schedule 時 warmup 必須為 0、minimum ratio 必須為 1")
        elif self.lr_schedule_name == LR_SCHEDULE_LINEAR_WARMUP_COSINE:
            if not 0.0 < warmup_fraction < 1.0:
                raise ValueError("linear warmup fraction 必須介於 0 與 1 之間")
            if not 0.0 < minimum_ratio <= 1.0:
                raise ValueError("minimum LR ratio 必須介於 0 與 1 之間")
        augmentation_parameters = self.augmentation_parameters()
        if self.augmentation_name == AUGMENTATION_NONE:
            if (
                float(self.augmentation_probability) != 0.0
                or int(self.augmentation_protected_recent_bars) != 0
                or int(self.augmentation_min_mask_bars) != 0
                or int(self.augmentation_max_mask_bars) != 0
                or augmentation_parameters
            ):
                raise ValueError("augmentation=none 時不可帶 augmentation 參數")
        elif self.augmentation_name == AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK:
            probability = float(self.augmentation_probability)
            protected_recent_bars = int(self.augmentation_protected_recent_bars)
            min_mask_bars = int(self.augmentation_min_mask_bars)
            max_mask_bars = int(self.augmentation_max_mask_bars)
            if not 0.0 < probability <= 1.0:
                raise ValueError("masking augmentation probability 必須介於 0 與 1 之間")
            if protected_recent_bars < 1:
                raise ValueError("masking protected_recent_bars 必須 >=1")
            if min_mask_bars < 1 or max_mask_bars < min_mask_bars:
                raise ValueError("masking bars 必須滿足 1 <= min <= max")

    def lr_schedule_parameters(self) -> dict[str, float]:
        if self.lr_schedule_name == LR_SCHEDULE_NONE:
            return {}
        return {
            "warmup_fraction": float(self.lr_warmup_fraction),
            "minimum_lr_ratio": float(self.lr_minimum_ratio),
        }

    def augmentation_parameters(self) -> dict[str, int | float]:
        if self.augmentation_name == AUGMENTATION_NONE:
            return {}
        if self.augmentation_name == AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK:
            return {
                "probability": float(self.augmentation_probability),
                "protected_recent_bars": int(self.augmentation_protected_recent_bars),
                "min_mask_bars": int(self.augmentation_min_mask_bars),
                "max_mask_bars": int(self.augmentation_max_mask_bars),
            }
        raise ValueError(f"不支援的 augmentation: {self.augmentation_name!r}")

    def as_manifest_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "optimizer_name": self.optimizer_name,
            "lr_schedule_name": self.lr_schedule_name,
            "augmentation_name": self.augmentation_name,
        }
        schedule_parameters = self.lr_schedule_parameters()
        if schedule_parameters:
            payload["lr_schedule_parameters"] = schedule_parameters
        augmentation_parameters = self.augmentation_parameters()
        if augmentation_parameters:
            payload["augmentation_parameters"] = augmentation_parameters
        if self.training_sampling_mode != TRAINING_SAMPLING_ALL_EVENT_ROWS:
            payload["training_sampling_mode"] = self.training_sampling_mode
        if self.time_weight_mode is not None:
            payload["time_weight_mode"] = self.time_weight_mode
        if self.training_weight_reduction != TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM:
            payload["training_weight_reduction"] = self.training_weight_reduction
        if self.training_objective != TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
            payload.update({
                "training_objective": self.training_objective,
                "continuous_target_id": self.continuous_target_id,
                "loss_name": self.loss_name,
                "epoch_selection_metric": self.epoch_selection_metric,
            })
            if self.raw_r_huber_delta_r is not None:
                payload["raw_r_huber_delta_r"] = float(self.raw_r_huber_delta_r)
            # The event-group scope is the historical continuous-ranker default.
            # Omit it so existing MR-12 artifact identities remain byte-for-byte
            # compatible; only new non-default sample scopes are explicit.
            if self.training_sample_scope != TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
                payload["training_sample_scope"] = self.training_sample_scope
            if self.training_label_scope != TRAINING_LABEL_SCOPE_ALL:
                payload["training_label_scope"] = self.training_label_scope
        if self.model_architecture is not None:
            payload["model_architecture"] = str(self.model_architecture)
        return payload



@dataclass(frozen=True)
class BreakoutQualityPretrainingProfile:
    name: str
    family: str
    optimizer_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    min_crop_bars: int
    mask_probability: float
    contrastive_alpha: float
    temporal_unit: int

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if not normalized_name or normalized_name != self.name:
            raise ValueError("pretraining profile name 必須是非空白小寫名稱")
        if any(token in normalized_name for token in ("/", "\\", "\x00")):
            raise ValueError("pretraining profile name 必須是安全的單一資料夾名稱")
        if str(self.family).strip().lower() != self.family or not self.family:
            raise ValueError("pretraining family 必須是非空白小寫名稱")
        if any(token in self.family for token in ("/", "\\", "\x00")):
            raise ValueError("pretraining family 必須是安全的單一資料夾名稱")
        if self.optimizer_name not in SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS:
            raise ValueError(f"不支援的 pretraining optimizer: {self.optimizer_name!r}")
        if int(self.epochs) < 1 or int(self.batch_size) < 2:
            raise ValueError("pretraining epochs 必須 >=1 且 batch_size 必須 >=2")
        if (
            float(self.learning_rate) <= 0.0
            or float(self.weight_decay) < 0.0
            or float(self.gradient_clip_norm) < 0.0
        ):
            raise ValueError(
                "pretraining learning_rate 必須 >0，weight_decay與gradient_clip_norm必須 >=0"
            )
        if int(self.min_crop_bars) < 2 or int(self.temporal_unit) < 0:
            raise ValueError("pretraining min_crop_bars 必須 >=2 且 temporal_unit 必須 >=0")
        if not 0.0 <= float(self.mask_probability) < 1.0:
            raise ValueError("pretraining mask_probability 必須介於0（含）與1（不含）")
        if not 0.0 <= float(self.contrastive_alpha) <= 1.0:
            raise ValueError("pretraining contrastive_alpha 必須介於0與1")

    def as_manifest_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "optimizer_name": self.optimizer_name,
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "min_crop_bars": int(self.min_crop_bars),
            "mask_probability": float(self.mask_probability),
            "contrastive_alpha": float(self.contrastive_alpha),
            "temporal_unit": int(self.temporal_unit),
        }


_PRETRAINING_PROFILES = {
    TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE: BreakoutQualityPretrainingProfile(
        name=TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE,
        family="ts2vec_v1",
        optimizer_name="adamw",
        epochs=10,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        min_crop_bars=60,
        mask_probability=0.5,
        contrastive_alpha=0.5,
        temporal_unit=0,
    ),
}
SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES = tuple(_PRETRAINING_PROFILES)


def normalize_breakout_quality_pretraining_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _PRETRAINING_PROFILES:
        allowed = ", ".join(SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES)
        raise ValueError(
            f"不支援的 breakout quality pretraining profile: {value!r}；可用值: {allowed}"
        )
    return normalized


def get_breakout_quality_pretraining_profile(
    value: str,
) -> BreakoutQualityPretrainingProfile:
    return _PRETRAINING_PROFILES[normalize_breakout_quality_pretraining_profile(value)]


def build_breakout_quality_pretraining_profile_payload(
    value: str,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    gradient_clip_norm: float | None = None,
    min_crop_bars: int | None = None,
    mask_probability: float | None = None,
    contrastive_alpha: float | None = None,
    temporal_unit: int | None = None,
) -> dict[str, Any]:
    """Return the named profile payload with explicit CLI overrides applied.

    Formal workflow runs use the profile defaults. The override path remains available for
    isolated development experiments, while downstream canonical training can reject an
    encoder whose stored payload differs from the active named profile.
    """

    profile = get_breakout_quality_pretraining_profile(value)
    resolved = BreakoutQualityPretrainingProfile(
        name=profile.name,
        family=profile.family,
        optimizer_name=profile.optimizer_name,
        epochs=profile.epochs if epochs is None else int(epochs),
        batch_size=profile.batch_size if batch_size is None else int(batch_size),
        learning_rate=(
            profile.learning_rate if learning_rate is None else float(learning_rate)
        ),
        weight_decay=profile.weight_decay if weight_decay is None else float(weight_decay),
        gradient_clip_norm=(
            profile.gradient_clip_norm
            if gradient_clip_norm is None
            else float(gradient_clip_norm)
        ),
        min_crop_bars=(
            profile.min_crop_bars if min_crop_bars is None else int(min_crop_bars)
        ),
        mask_probability=(
            profile.mask_probability
            if mask_probability is None
            else float(mask_probability)
        ),
        contrastive_alpha=(
            profile.contrastive_alpha
            if contrastive_alpha is None
            else float(contrastive_alpha)
        ),
        temporal_unit=profile.temporal_unit if temporal_unit is None else int(temporal_unit),
    )
    return resolved.as_manifest_payload()

_EXPERIMENT_PROFILES = {
    BASELINE_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=BASELINE_EXPERIMENT_PROFILE,
        optimizer_name="adam",
    ),
    ADAMW_ONLY_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=ADAMW_ONLY_EXPERIMENT_PROFILE,
        optimizer_name="adamw",
    ),
    ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        lr_schedule_name=LR_SCHEDULE_LINEAR_WARMUP_COSINE,
        lr_warmup_fraction=0.05,
        lr_minimum_ratio=0.10,
    ),
    HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        augmentation_name=AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK,
        augmentation_probability=0.50,
        augmentation_protected_recent_bars=60,
        augmentation_min_mask_bars=10,
        augmentation_max_mask_bars=30,
    ),
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    ),
    UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        time_weight_mode=TIME_WEIGHT_MODE_DATE_BALANCED,
        training_weight_reduction=TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
    ),
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
    ),
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_PASS_ONLY,
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="listnet_top_one_cross_entropy",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
    ),
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="patch_token_transformer_ranker_v1",
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_first_risk_breach_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id=PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID,
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_predicted_upside_context_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id=PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID,
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_predicted_safety_context_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id=PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID,
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_predicted_safety_context_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="dual_mse_raw_r",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_low_adverse_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_equal_rank_mfe_low_adverse_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_pareto_pair_concordance",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="conditional_safety_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_conditional_mfe_safety_v1",
    ),
    DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_conditional_mfe_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_conditional_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_conditional_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_conditional_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_priority_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_priority_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_raw_mfe_hmhs_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_raw_mfe_hmhs_mlp_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_raw_mfe_hmhs_mlp_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_raw_mfe_joint_attn_mlp_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="modern_tcn_safety_raw_mfe_joint_attn_mlp_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="patch_token_transformer_safety_raw_mfe_joint_attn_mlp_v1",
    ),
    DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="hmhs_pairwise_concordance",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="huber_raw_r",
        epoch_selection_metric="validation_huber_raw_r",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        raw_r_huber_delta_r=1.0,
    ),
    DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="mse_raw_r",
        epoch_selection_metric="validation_mse_raw_r",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_risk_normalized_net_opportunity_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_risk_normalized_net_opportunity_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_risk_context_v1",
    ),
}

SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES = tuple(_EXPERIMENT_PROFILES)
SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES = tuple(
    name
    for name, profile in _EXPERIMENT_PROFILES.items()
    if profile.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION
)


def normalize_breakout_quality_experiment_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _EXPERIMENT_PROFILES:
        allowed = ", ".join(SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES)
        raise ValueError(
            f"不支援的 breakout quality experiment profile: {value!r}；可用值: {allowed}"
        )
    return normalized


def get_breakout_quality_experiment_profile(
    value: str,
) -> BreakoutQualityExperimentProfile:
    return _EXPERIMENT_PROFILES[
        normalize_breakout_quality_experiment_profile(value)
    ]




@dataclass(frozen=True)
class ContinuousRankerResearchSpec:
    profile_name: str
    model_research_id: str
    experiment_name: str
    phase: str
    trainer_family: str
    target_description: str
    objective_description: str
    metric_scope: str
    score_semantic_id: str
    pairwise_reduction: str | None = None
    pair_weight_policy: str | None = None
    secondary_pair_scope: str = CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL
    secondary_pair_scope_threshold: float | None = None
    reference_profile_name: str | None = None
    evaluation_reference_profile_name: str | None = None
    model_gate_reference_profile_name: str | None = None
    # Historical/research PIT authorization.  This may remain true for archived
    # evidence that must still be readable/reconstructable.  Current OOS/Rolling
    # execution is governed separately by current_time_validation_authorized.
    selection_pit_authorized: bool = True
    current_time_validation_authorized: bool = False

    def __post_init__(self) -> None:
        if self.profile_name not in _EXPERIMENT_PROFILES:
            raise ValueError(f"continuous ranker research spec引用未知profile: {self.profile_name}")
        profile = _EXPERIMENT_PROFILES[self.profile_name]
        if profile.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
            raise ValueError(f"continuous ranker research spec只接受continuous profile: {self.profile_name}")
        if self.trainer_family not in {
            CONTINUOUS_RANKER_TRAINER_EVENT,
            CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        }:
            raise ValueError(f"不支援的continuous ranker trainer family: {self.trainer_family}")
        expected_family = (
            CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL
            if profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
            else CONTINUOUS_RANKER_TRAINER_EVENT
        )
        if self.trainer_family != expected_family:
            raise ValueError(
                "continuous ranker trainer family與training sample scope不一致: "
                f"profile={self.profile_name}, expected={expected_family}, actual={self.trainer_family}"
            )
        if profile.training_objective in {
            TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
        }:
            if self.pairwise_reduction not in SUPPORTED_CONTINUOUS_RANKER_PAIRWISE_REDUCTIONS:
                raise ValueError(
                    f"pairwise profile必須指定合法pairwise reduction: {self.profile_name}"
                )
            if (
                profile.training_objective == TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING
                and self.pairwise_reduction != CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE
            ):
                raise ValueError("Pareto pairwise profile必須使用pareto_dominance_equal_pair")
            if (
                profile.training_objective in {
                    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
                    TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
                }
                and self.pairwise_reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE
            ):
                raise ValueError("scalar pairwise profile不得使用Pareto dominance pair scope")
        elif self.pairwise_reduction is not None:
            raise ValueError(
                f"非pairwise profile不得指定pairwise reduction: {self.profile_name}"
            )
        if self.pair_weight_policy is not None:
            if profile.training_objective not in {
                TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
            }:
                raise ValueError(
                    f"非scalar pairwise profile不得指定pair weight policy: {self.profile_name}"
                )
            get_continuous_ranker_pair_weight_policy(self.pair_weight_policy)
            if self.pair_weight_policy == CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE:
                raise ValueError("research spec的pair_weight_policy=None即可表示未加權；不得顯式宣告none")
        if self.secondary_pair_scope not in SUPPORTED_CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPES:
            raise ValueError(f"不支援的secondary pair scope: {self.secondary_pair_scope!r}")
        if self.secondary_pair_scope == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL:
            if self.secondary_pair_scope_threshold is not None:
                raise ValueError("all-items secondary pair scope不得指定threshold")
        else:
            if profile.training_objective not in {
                TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
            }:
                raise ValueError("scoped secondary pair supervision只允許支援該capability的training objective")
            threshold = self.secondary_pair_scope_threshold
            if threshold is None or not (0.0 < float(threshold) < 1.0):
                raise ValueError("secondary pair scope threshold必須位於(0,1)")
        for reference_field in (
            "reference_profile_name",
            "evaluation_reference_profile_name",
            "model_gate_reference_profile_name",
        ):
            reference_value = getattr(self, reference_field)
            if reference_value is None:
                continue
            reference = str(reference_value).strip()
            if reference not in _EXPERIMENT_PROFILES:
                raise ValueError(
                    f"continuous ranker {reference_field}不存在: {reference_value}"
                )
            if reference == self.profile_name:
                raise ValueError(f"continuous ranker {reference_field}不得等於自身")
        if self.current_time_validation_authorized and not self.selection_pit_authorized:
            raise ValueError(
                "current time validation authorization必須建立在PIT research authorization上: "
                f"{self.profile_name}"
            )
        for field_name in (
            "model_research_id",
            "experiment_name",
            "phase",
            "target_description",
            "objective_description",
            "metric_scope",
            "score_semantic_id",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(
                    f"continuous ranker research spec缺少{field_name}: {self.profile_name}"
                )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "profile_name": self.profile_name,
            "model_research_id": self.model_research_id,
            "experiment_name": self.experiment_name,
            "phase": self.phase,
            "trainer_family": self.trainer_family,
            "target_description": self.target_description,
            "objective_description": self.objective_description,
            "metric_scope": self.metric_scope,
            "score_semantic_id": self.score_semantic_id,
            "pairwise_reduction": self.pairwise_reduction,
            "reference_profile_name": self.reference_profile_name,
            "evaluation_reference_profile_name": self.evaluation_reference_profile_name,
            "selection_pit_authorized": bool(self.selection_pit_authorized),
            "current_time_validation_authorized": bool(
                self.current_time_validation_authorized
            ),
        }
        if self.secondary_pair_scope != CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL:
            payload["secondary_pair_scope"] = self.secondary_pair_scope
            payload["secondary_pair_scope_threshold"] = float(self.secondary_pair_scope_threshold)
        if self.model_gate_reference_profile_name is not None:
            payload["model_gate_reference_profile_name"] = self.model_gate_reference_profile_name
        return payload


_CONTINUOUS_RANKER_RESEARCH_SPECS = {
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        model_research_id="MR-11B",
        experiment_name="11B Strategy-aligned Daily Percentile Ranker",
        phase="11B",
        trainer_family=CONTINUOUS_RANKER_TRAINER_EVENT,
        target_description="same_date_rank_percentile_of_strategy_aligned_opportunity_r_v1",
        objective_description="同日11A target percentile的MSE",
        metric_scope="all_labels",
        score_semantic_id="opportunity_rank",
    ),
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        model_research_id="MR-11G",
        experiment_name="11G PASS-conditional No-time Magnitude Ranker",
        phase="11G",
        trainer_family=CONTINUOUS_RANKER_TRAINER_EVENT,
        target_description="same_date_pass_only_rank_percentile_of_strategy_aligned_opportunity_no_time_r_v1",
        objective_description="同日PASS-only No-time target percentile的MSE",
        metric_scope="pass_only",
        score_semantic_id="opportunity_rank",
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
        model_research_id="MR-12A",
        experiment_name="MR-12A All-event No-time Continuous Ranker",
        phase="12A",
        trainer_family=CONTINUOUS_RANKER_TRAINER_EVENT,
        target_description="same_date_all_event_rank_percentile_of_strategy_aligned_opportunity_no_time_r_v1",
        objective_description="同日all-event No-time target percentile的MSE",
        metric_scope="all_labels",
        score_semantic_id="opportunity_rank",
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
        model_research_id="MR-12B",
        experiment_name="MR-12B All-event No-time Pairwise Ranker",
        phase="12B",
        trainer_family=CONTINUOUS_RANKER_TRAINER_EVENT,
        target_description="same_date_all_event_order_of_strategy_aligned_opportunity_no_time_r_v1",
        objective_description="同日all-event No-time target ordering的RankNet pairwise logistic loss",
        metric_scope="all_labels",
        score_semantic_id="opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE,
        model_research_id="MR-12C",
        experiment_name="MR-12C All-event No-time ListNet Top-one Ranker",
        phase="12C",
        trainer_family=CONTINUOUS_RANKER_TRAINER_EVENT,
        target_description="same_date_all_event_listnet_distribution_of_strategy_aligned_opportunity_no_time_r_v1",
        objective_description="同日all-event No-time完整候選榜單的ListNet top-one cross-entropy",
        metric_scope="all_labels",
        score_semantic_id="opportunity_rank",
    ),
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        model_research_id="MR-13A",
        experiment_name="MR-13A Daily Universal No-time Pairwise Ranker",
        phase="13A",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_opportunity_no_time_r_v1",
        objective_description="同日全部合法stock-day No-time target ordering的RankNet pairwise logistic loss",
        metric_scope="all_stock_days",
        score_semantic_id="daily_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    ),
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE,
        model_research_id="MR-13B",
        experiment_name="MR-13B Daily Universal Target-gap-weighted Pairwise Ranker",
        phase="13B",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_opportunity_no_time_r_v1",
        objective_description=(
            "同日全部合法stock-day No-time target ordering的RankNet pairwise logistic loss；"
            "pair依daily target percentile距離加權並於date內正規化"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED,
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE,
        model_research_id="MR-13C",
        experiment_name="MR-13C Daily Universal Percentile Regression",
        phase="13C",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_percentile_of_daily_opportunity_no_time_r_v1",
        objective_description="同日全部合法stock-day No-time opportunity target percentile的MSE",
        metric_scope="all_stock_days",
        score_semantic_id="daily_opportunity_rank",
    ),
    DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE,
        model_research_id="MR-13D",
        experiment_name="MR-13D Daily Universal Upper-tail Relevance-weighted Pairwise Ranker",
        phase="13D",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_opportunity_no_time_r_v1",
        objective_description=(
            "同日全部合法stock-day No-time target ordering的RankNet pairwise logistic loss；"
            "pair依兩端daily target percentile算術平均作upper-tail relevance權重"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE,
    ),
    DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13E",
        experiment_name="MR-13E Daily Universal Full-list Delta-NDCG-weighted Pairwise Ranker",
        phase="13E",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_opportunity_no_time_r_v1",
        objective_description=(
            "同日全部合法stock-day No-time target ordering的RankNet pairwise logistic loss；"
            "pair依目前預測完整榜單交換造成的raw-percentile Delta-NDCG作權重"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13H",
        experiment_name="MR-13H Daily Universal Full-horizon No-breach Ranker",
        phase="13H",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_full_horizon_opportunity_r_v1",
        objective_description=(
            "MR-13E同一daily-universal full-list Delta-NDCG pairwise objective；"
            "唯一變更為future low觸及risk barrier不再截斷固定40D target path"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        reference_profile_name=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AA",
        experiment_name="MR-13AA MR-13H Target + Frozen Patch Transformer Architecture Control",
        phase="13AA",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_full_horizon_opportunity_r_v1",
        objective_description=(
            "MR-13H exact controlled architecture contrast：target、daily-universal universe、"
            "full-list Delta-NDCG pairwise objective、Seed42、split、optimizer、epoch selection、"
            "training/sample scope全部不變；唯一scientific dimension為temporal architecture由"
            "InceptionTime換成MR-13Z已freeze的historical-recipe Patch Transformer。"
            "Patch recipe固定non-overlap patch=10、embedding=128、depth=3、heads=4、FFN=256、"
            "sinusoidal position、LayerNorm、dropout=0.10與mean pooling；不做architecture tuning。"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        reference_profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13K",
        experiment_name="MR-13K Daily Universal Full-horizon Pure-MFE Ranker",
        phase="13K",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_full_horizon_pure_mfe_r_v1",
        objective_description=(
            "MR-13H同一daily-universal full-list Delta-NDCG pairwise objective；"
            "唯一變更為adverse-to-peak仍保留診斷但不再從固定40D target扣除"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        reference_profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=True,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AB",
        experiment_name="MR-13AB Daily Universal Pure-MFE Before First Risk Breach Ranker",
        phase="13AB",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_first_risk_breach_pure_mfe_r_v1",
        objective_description=(
            "MR-13K exact controlled target contrast：daily-universal universe、InceptionTime、"
            "full-list Delta-NDCG pairwise objective、Seed42、split、optimizer、epoch selection、"
            "training/sample scope與40D fixed R scale全部不變；唯一scientific dimension為"
            "Pure-MFE future path沿用MR-13E same-bar adverse-first first-risk-breach truncation。"
            "adverse-to-peak只保留diagnostic，不從target扣除；首根即breach為0R。"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_first_risk_breach_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        reference_profile_name=DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AC",
        experiment_name="MR-13AC PIT-safe Predicted-Upside Conditional Low-Adverse Ranker",
        phase="13AC",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "same_date_percentile_of_low_adverse_residual_given_PIT_safe_predicted_pure_mfe_percentile"
        ),
        objective_description=(
            "Plan C-M controlled Model Gate：Stage-1固定MR-13K Pure-MFE ranker。Selection context只允許expanding cross-fitted/PIT-safe same-day predicted-upside percentile；"
            "因此Stage-2 Selection training universe只使用已有合法PIT context覆蓋的rows，不以前視或full-fit score回填較早rows。"
            "Forward OOS context固定單一pre-2021 Stage-1 fit，不在OOS期間refit。Stage-2固定MR-13M InceptionTime/full-list Delta-NDCG/Seed42/split/optimizer，"
            "輸入只新增1個PIT-safe upside percentile scalar；label為同日Low-Adverse percentile對該predicted-upside percentile含intercept OLS residual後再同日percentile化。"
            "不讀full-fit Selection score、不使用OOS統計、portfolio state、threshold、lambda、score fusion或backbone tuning。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_predicted_upside_conditional_low_adverse_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AD",
        experiment_name="MR-13AD PIT-safe Predicted-Safety Conditional MFE Reverse-Control",
        phase="13AD",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "same_date_percentile_of_pure_mfe_residual_given_PIT_safe_predicted_low_adverse_safety_percentile"
        ),
        objective_description=(
            "MR-13AC methodology reverse-control：Stage-1固定MR-13M Low-Adverse ranker。Selection context只允許expanding cross-fitted/PIT-safe same-day predicted-safety percentile；"
            "Stage-2 Selection training universe只使用已有合法PIT context覆蓋的rows，不以前視或full-fit score回填。Forward OOS context固定單一pre-2021 Stage-1 fit。"
            "Stage-2固定MR-13K InceptionTime/full-list Delta-NDCG/Seed42/split/optimizer/epoch-selection，輸入只新增1個PIT-safe safety percentile scalar；"
            "label為同日Pure-MFE percentile對predicted-safety percentile含intercept OLS residual後再同日percentile化。"
            "Model Gate同時評估own-target learnability、Pure-MFE/upside與Low-Adverse/downside evidence；不使用portfolio state、threshold、lambda、score fusion或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_predicted_safety_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AE",
        experiment_name="MR-13AE PIT-safe Predicted-Safety Context Pure-MFE Control",
        phase="13AE",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="exact_MR-13K_full_horizon_Pure-MFE_order_with_PIT_safe_predicted_safety_context",
        objective_description=(
            "MR-13AD follow-up controlled cell：Stage-1完全重用MR-13M Low-Adverse PIT-safe predicted-safety context；"
            "Stage-2回復MR-13K canonical full-horizon Pure-MFE target/order，不做conditional residualization。"
            "MR-13K InceptionTime/full-list Delta-NDCG/Seed42/split/optimizer/epoch-selection/training scope全部固定，"
            "唯一scientific change是GAP latent後direct concat 1個PIT-safe predicted-safety percentile scalar。"
            "Selection只使用cross-fitted context-covered rows，Forward只使用single fixed pre-OOS context；"
            "Model Gate先驗證Pure-MFE learnability是否接近MR-13K，再檢查高分股Adverse/High-Safety是否改善；"
            "不使用portfolio state、threshold、lambda、fusion、product或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_predicted_safety_context_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AF",
        experiment_name="MR-13AF PIT-safe High-Safety-Weighted Pure-MFE Ranker",
        phase="13AF",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="exact_MR-13K_full_horizon_Pure-MFE_order_with_PIT_safe_predicted_safety_pair_weighting",
        objective_description=(
            "MR-13AE anti-shortcut controlled cell：Stage-1完全重用MR-13M Low-Adverse PIT-safe predicted-safety context；"
            "Stage-2回復MR-13K inception_time_v1，不把Safety scalar輸入network，Pure-MFE target/order完全不變。"
            "MR-13K full-list Delta-NDCG pair relevance乘上min(S_i,S_j)，S為同日PIT-safe predicted-Safety percentile；"
            "最終loss以所有同日comparable pairs做normalized weighted mean，所以只改relative supervision importance、不改loss scale。"
            "高Safety×高Safety的MFE比較保留高權重；任何含低Safety股票的pair自然降權。"
            "沒有bucket、Safety cutoff、lambda、temperature、joint head、residual target、portfolio state或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_high_safety_weighted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AG",
        experiment_name="MR-13AG PIT-safe MFE-Winner-Safety-Weighted Pure-MFE Ranker",
        phase="13AG",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="exact_MR-13K_full_horizon_Pure-MFE_order_with_MFE_winner_predicted_safety_pair_weighting",
        objective_description=(
            "MR-13AF directional anti-shortcut follow-up：Stage-1完全重用MR-13M Low-Adverse PIT-safe predicted-safety context；"
            "Stage-2固定MR-13K inception_time_v1，Safety不進network，Pure-MFE target/order與full-list Delta-NDCG relevance完全不變。"
            "每個同日non-tied MFE pair只將Delta-NDCG乘上較高Pure-MFE item本身的PIT-safe predicted-Safety percentile S_winner；"
            "因此safe MFE winner > unsafe loser保留高權重，而unsafe MFE winner > safe loser自然降權，pair方向永不因Safety反轉。"
            "最終loss仍為normalized weighted mean；沒有額外mean normalization、bucket、cutoff、lambda、temperature、joint head、"
            "residual target、portfolio state或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_mfe_winner_safety_weighted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AH",
        experiment_name="MR-13AH PIT-safe Safety-Product-Weighted Pure-MFE Ranker",
        phase="13AH",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="exact_MR-13K_full_horizon_Pure-MFE_order_with_symmetric_predicted_safety_product_pair_weighting",
        objective_description=(
            "MR-13AF symmetric-weight concentration follow-up：Stage-1完全重用MR-13M Low-Adverse PIT-safe predicted-safety context；"
            "Stage-2固定MR-13K inception_time_v1，Safety不進network，Pure-MFE target/order與full-list Delta-NDCG relevance完全不變。"
            "每個同日non-tied MFE pair只將Delta-NDCG乘上兩端PIT-safe predicted-Safety percentile乘積 S_i*S_j；"
            "pair weighting保持完全對稱，不依MFE winner方向改變，因此不重複MR-13AG directional supervision。"
            "相較MR-13AF min(S_i,S_j)，product會更集中於high-Safety manifold並更強壓低low/low pair；"
            "最終loss仍為normalized weighted mean；沒有bucket、cutoff、lambda、exponent、temperature、joint head、residual target、portfolio state或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_safety_product_weighted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
        # Historical model-level flags remain frozen after the failed Forward Model Gate.
        # B313 makes current work-item membership the execution SSOT: when MR-13AH is
        # selected in the shared Model Compare/Test List it may run current Rolling /
        # Robustness workflows; outside that list these frozen flags remain fail-closed.
        # Fixed-Window remains retired and this does not imply model/production promotion.
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AJ",
        experiment_name="MR-13AJ PIT-safe Conflict-Only Unsafe-Winner-Discounted Pure-MFE Ranker",
        phase="13AJ",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="exact_MR-13K_full_horizon_Pure-MFE_order_with_conflict_only_unsafe_winner_predicted_safety_discount",
        objective_description=(
            "AF/AG/AH follow-up：Stage-1仍完全重用MR-13M/Seed42 PIT-safe predicted-Safety context；Stage-2固定MR-13K inception_time_v1、"
            "Pure-MFE target/order、full-list Delta-NDCG、split、optimizer與epoch selection。Safety不進network，pair truth永不反轉。"
            "若MFE ordering與predicted-Safety ordering一致或Safety tie，保留完整MR-13K pair weight=Delta-NDCG；"
            "只有MFE winner較不安全的conflict pair才乘該winner的Safety percentile。"
            "此設計保留AG遺失的aligned MFE supervision，同時比AF更精準地只削弱unsafe-high-MFE conflict pressure；"
            "沒有bucket、cutoff、lambda、exponent、temperature、joint head、residual target、portfolio state或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_conflict_discounted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE,
        model_research_id="MR-13L",
        experiment_name="MR-13L Daily Universal Decomposed MFE-Adverse Regression",
        phase="13L",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="daily_full_horizon_opportunity_r_v1_decomposed_into_favorable_r_and_adverse_to_peak_r",
        objective_description=(
            "MR-13H相同full-horizon target/universe/architecture；兩個既有輸出神經元分別直接預測"
            "favorable MFE R與adverse-to-peak R，primary loss為兩分量等權MSE平均，"
            "正式model score固定為Predicted MFE R - Predicted adverse R；不使用auxiliary loss、不調lambda"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_decomposed_opportunity_r",
        reference_profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13M",
        experiment_name="MR-13M Daily Universal Full-horizon Low-Adverse Ranker",
        phase="13M",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_negative_daily_full_horizon_adverse_to_peak_r_v1",
        objective_description=(
            "MR-13H/13K相同40D full-horizon earliest-max-MFE peak與fixed R scale；"
            "只以到達該peak前的adverse-to-peak R取負值作排序Target，越高代表path risk越小；"
            "沿用full-list Delta-NDCG weighted RankNet，不加入MFE或strategy state"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_low_adverse_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13N",
        experiment_name="MR-13N Daily Universal Equal-rank MFE + Low-Adverse Ranker",
        phase="13N",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_equal_weight_mean_of_mfe_percentile_and_low_adverse_percentile",
        objective_description=(
            "MR-13K Pure-MFE與MR-13M low-adverse兩個已證實可學component先各自轉為同日[0,1] percentile；"
            "正式Target固定為兩者等權平均，不掃lambda；沿用full-list Delta-NDCG weighted RankNet。"
            "MR-13H economic Target只在checkpoint後作reference evaluation，不參與training或epoch selection"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_equal_rank_mfe_low_adverse",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        evaluation_reference_profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE,
        model_research_id="MR-13O",
        experiment_name="MR-13O Daily Universal Pareto MFE + Low-Adverse Pairwise Ranker",
        phase="13O",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "same_date_strict_pareto_dominance_pairs_over_mfe_percentile_and_low_adverse_percentile; "
            "economic_target_remains_daily_full_horizon_opportunity_r_v1"
        ),
        objective_description=(
            "同日只有當一個stock-day在MFE percentile與low-adverse percentile兩者都嚴格高於另一個stock-day時才提供RankNet supervision；"
            "兩component互有優劣的trade-off pair完全排除。pair等權，不使用MFE/adverse混合係數；"
            "epoch selection只依Validation mean daily Pareto pair concordance，MR-13H economic R不參與選模"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_pareto_dominance_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE,
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13P",
        experiment_name="MR-13P Daily Universal Single-model Conditional MFE-Safety Ranker",
        phase="13P",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "primary=same_date_pure_mfe_percentile; "
            "secondary=same_date_percentile_of_low_adverse_residual_given_true_mfe_percentile"
        ),
        objective_description=(
            "單一shared InceptionTime encoder；Pure-MFE primary head與MFE-conditioned residual-safety head皆使用"
            "full-list Delta-NDCG RankNet。secondary target由同日low-adverse percentile對true MFE percentile"
            "含intercept OLS residual後再同日percentile化；conditional head只接收stop-gradient primary prediction，"
            "不做score fusion／threshold／loss-weight sweep"
        ),
        metric_scope="all_stock_days_dual_head",
        score_semantic_id="daily_conditional_mfe_safety_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13Q",
        experiment_name="MR-13Q Daily Universal Reverse-Conditional MFE Single-head Ranker",
        phase="13Q",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_percentile_of_pure_mfe_residual_given_true_low_adverse_safety_percentile",
        objective_description=(
            "Single-head InceptionTime直接學J=U-E(U|S)：U為同日Pure-MFE percentile、S為同日low-adverse Safety percentile；"
            "每日日內含intercept OLS後取MFE residual並再percentile化。只使用full-list Delta-NDCG RankNet，"
            "不建立Safety head、不做scalar MFE/adverse composite、threshold或loss-weight sweep。"
        ),
        metric_scope="all_stock_days_reverse_conditional_mfe",
        score_semantic_id="daily_reverse_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13R",
        experiment_name="MR-13R Daily Universal Safety-conditioned MFE Duo-head Ranker",
        phase="13R",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "condition=same_date_low_adverse_safety_percentile; "
            "final=same_date_percentile_of_pure_mfe_residual_given_true_safety_percentile"
        ),
        objective_description=(
            "單一shared InceptionTime encoder；Raw Safety auxiliary head先學S，Conditional-MFE final head接收shared latent與"
            "stop-gradient Safety prediction並學同一J=U-E(U|S)。兩head均使用full-list Delta-NDCG RankNet、固定等尺度loss平均；"
            "epoch selection與strategy score只看Conditional-MFE head，Safety head不直接進selector。"
        ),
        metric_scope="all_stock_days_safety_conditioned_mfe_dual_head",
        score_semantic_id="daily_safety_conditioned_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13S",
        experiment_name="MR-13S Daily Universal Safety-conditioned Raw-MFE Duo-head Ranker",
        phase="13S",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "condition=same_date_low_adverse_safety_percentile; "
            "final=same_date_pure_mfe_percentile"
        ),
        objective_description=(
            "MR-13R controlled contrast：shared InceptionTime、Raw Safety auxiliary、stop-gradient Safety context與"
            "兩head full-list Delta-NDCG等權loss全部不變；唯一scientific change是final head由residual J改學absolute Pure-MFE percentile U。"
            "epoch selection只依Raw-MFE head Validation mean daily Spearman；model-only Gate，不授權PIT或strategy conversion。"
        ),
        metric_scope="all_stock_days_safety_conditioned_raw_mfe_dual_head",
        score_semantic_id="daily_safety_conditioned_raw_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AK",
        experiment_name="MR-13AK A1 Shared-AH",
        phase="13AK",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile"
        ),
        objective_description=(
            "A0 MR-13AH的唯一control change：把external MR-13M Safety DL改成同一InceptionTime shared encoder的Raw Safety head。"
            "Safety head使用普通full-list Delta-NDCG RankNet；Pure-MFE final head只讀shared latent，不接Safety prediction。"
            "Raw Safety probability先按每個完整交易日轉成與AH相同的average-rank percentile S；"
            "MFE full-list Delta-NDCG pair weight乘detach(S_i)×detach(S_j)，pair direction仍100%由Pure-MFE決定；"
            "兩head loss固定等權平均，final runtime/model-gate score只使用MFE head。"
        ),
        metric_scope="all_stock_days_shared_safety_weighted_raw_mfe_dual_head",
        score_semantic_id="daily_shared_safety_weighted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AL",
        experiment_name="MR-13AL A2 Shared-AH + Safety Context",
        phase="13AL",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile"
        ),
        objective_description=(
            "A1 MR-13AK的唯一control change：Safety/MFE shared encoder、兩head truth、Safety full-list Delta-NDCG、"
            "AH detach(S_i)×detach(S_j) Pure-MFE pair weighting、pair direction、equal head-loss mean、Seed/split/optimizer全部固定；"
            "final MFE head由shared latent only改為concat(shared latent, detach(Raw Safety probability))。"
            "Safety context不接受MFE loss gradient；final model-gate score仍只使用MFE head。"
        ),
        metric_scope="all_stock_days_shared_safety_context_weighted_raw_mfe_dual_head",
        score_semantic_id="daily_shared_safety_context_weighted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AM",
        experiment_name="MR-13AM A3 Shared-AH Economic Target Control",
        phase="13AM",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_percentile_of_daily_full_horizon_opportunity_r_v1"
        ),
        objective_description=(
            "MR-13AK A1的單一target control：shared InceptionTime、獨立Raw Safety/final head topology、"
            "Safety full-list Delta-NDCG、same-date predicted-Safety percentile detach(S_i)×detach(S_j) pair weighting、"
            "pair direction、equal head-loss mean、Seed/split/optimizer全部固定；唯一scientific change為final head truth由"
            "Pure-MFE percentile改成MR-13H daily_full_horizon_opportunity_r_v1 percentile。"
            "不使用MR-13AL Safety-context input；final Model-Gate score只使用economic-target head。"
        ),
        metric_scope="all_stock_days_shared_safety_weighted_full_horizon_opportunity_dual_head",
        score_semantic_id="daily_shared_safety_weighted_full_horizon_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AN",
        experiment_name="MR-13AN AM Economic Target + A2 Safety Context",
        phase="13AN",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_percentile_of_daily_full_horizon_opportunity_r_v1"
        ),
        objective_description=(
            "MR-13AM的唯一control change：economic target、shared encoder、Safety truth/loss、"
            "same-date predicted-Safety percentile detach(S_i)×detach(S_j) pair weighting、pair direction、"
            "equal head-loss mean、Seed/split/optimizer全部固定；final economic-target head由shared latent only改為"
            "concat(shared latent, detach(Raw Safety probability))。Safety context不接受primary loss gradient；"
            "final Model-Gate score仍只使用economic-target head。"
        ),
        metric_scope="all_stock_days_shared_safety_context_weighted_full_horizon_opportunity_dual_head",
        score_semantic_id="daily_shared_safety_context_weighted_full_horizon_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AO",
        experiment_name="MR-13AO True-HS Conditional-MFE Shared Ranker",
        phase="13AO",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "Full-universe samples仍全部進shared InceptionTime encoder且全部監督Raw Safety head；"
            "true HS固定為same-date Safety percentile>=0.50。Conditional-MFE head只讀shared latent，"
            "且只有true-HS items先形成獨立sublist後才計算full-list Delta-NDCG RankNet；LS rows不參與MFE pair、"
            "predicted rank position、IDCG或Delta-NDCG geometry。MFE head不接predicted Safety、不使用Safety pair weight；"
            "兩head loss固定1:1。Seed42 Forward Model Gate only，不授權PIT/strategy/robustness。"
        ),
        metric_scope="full_universe_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AP",
        experiment_name="MR-13AP HS-Priority MFE Shared Ranker",
        phase="13AP",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=LS_relevance_0_else_0.5_plus_0.5_times_same_date_MFE_percentile_within_true_HS"
        ),
        objective_description=(
            "MR-13AO strict follow-up after true-HS upside learnability was confirmed but LS extrapolation amplified contamination. "
            "Architecture、Raw Safety head、full-universe encoder exposure、Seed42、split、optimizer與full-list Delta-NDCG固定。"
            "唯一核心改動是final ranking head不再mask LS：true LS全部明確監督為最差relevance=0；true HS依HS cohort內MFE percentile映射到[0.5,1.0]，"
            "因此任一HS truth嚴格高於任一LS，HS內仍依MFE排序，LS內全部tie。Final head只讀shared latent、不吃predicted Safety、不做Safety pair weighting；"
            "inference直接在all-daily universe使用final score，不使用Pred-Safety gate。Seed42 Forward Model Gate only。"
        ),
        metric_scope="all_daily_non_compensatory_hs_priority_then_mfe",
        score_semantic_id="daily_hs_priority_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AQ",
        experiment_name="MR-13AQ HS-Priority Stratified-MFE Shared Ranker",
        phase="13AQ",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=MR13AP_truth_LS_0_else_0.5_plus_0.5_times_same_date_MFE_percentile_within_true_HS"
        ),
        objective_description=(
            "MR-13AP strict loss-aggregation control：architecture、Safety auxiliary、AP final truth、all-daily sample exposure、"
            "Seed42/split/optimizer/full-list Delta-NDCG與direct all-daily inference全部固定。唯一change是final-head comparable pairs依"
            "true-HS boundary分為HS↔LS與HS↔HS兩個stratum；兩者保留同一full-list predicted-rank/IDCG/Delta-NDCG geometry，"
            "但各自先以自身Delta-NDCG weight sum正規化，再固定1:1平均。LS↔LS仍tie無direction。此設計只移除AP的boundary supervision-mass dominance，"
            "不改任何pair direction、不加入lambda sweep或Pred-Safety gate。Seed42 Forward Model Gate only。"
        ),
        metric_scope="all_daily_non_compensatory_hs_priority_pair_stratified_then_mfe",
        score_semantic_id="daily_hs_priority_stratified_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AR",
        experiment_name="MR-13AR Direct HS-Qualification + Conditional-MFE Shared Ranker",
        phase="13AR",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=true_HS_indicator_from_same_date_low_adverse_safety_percentile_gte_0.50; "
            "head2=same_date_pure_mfe_percentile_within_true_HS_only"
        ),
        objective_description=(
            "MR-13AO strict primary-head supervision control：shared InceptionTime、full-universe encoder exposure、"
            "true-HS definition、Conditional-MFE target/sublist、independent heads、1:1 head weighting、Seed42/split/optimizer、"
            "epoch selection與Pred-HS P50→Conditional-MFE inference全部固定。唯一scientific change是第一head不再學continuous "
            "Low-Adverse percentile ordering，而將same-date Safety percentile>=0.50轉成binary HS qualification truth；"
            "因此只有HS↔LS pairs有qualification direction，HS↔HS與LS↔LS為tie。Conditional-MFE仍只有true-HS sublist產生gradient，"
            "不吃qualification prediction、不使用Safety pair weight。Seed42 Forward Model Gate only。"
        ),
        metric_scope="direct_hs_qualification_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_hs_qualification_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AS",
        experiment_name="MR-13AS Boundary-Weighted HS-Qualification + Conditional-MFE Shared Ranker",
        phase="13AS",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=true_HS_indicator_from_same_date_low_adverse_safety_percentile_gte_0.50; "
            "head2=same_date_pure_mfe_percentile_within_true_HS_only"
        ),
        objective_description=(
            "MR-13AR strict boundary-supervision control：architecture、true-HS definition=P50、binary HS/LS pair direction、"
            "true-HS Conditional-MFE target/sublist、independent heads、1:1 head weighting、Seed42/split/optimizer、epoch selection與"
            "Pred-HS P50→Conditional-MFE inference全部固定。唯一scientific change是qualification head的HS↔LS pair supervision在"
            "canonical full-list Delta-NDCG weight之上再乘`1-|SafetyPct_i-SafetyPct_j|`；跨界pair越接近P50權重越高，"
            "P90↔P10等容易遠距pair降權。SafetyPct只作truth-side supervision weight，不進model input、不改pair direction、"
            "不新增cutoff/lambda/temperature。Conditional-MFE head與MR-13AR完全相同。Seed42 Forward Model Gate only。"
        ),
        metric_scope="boundary_weighted_hs_qualification_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_boundary_weighted_hs_qualification_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AT",
        experiment_name="MR-13AT Dual-Supervised HS Representation + Conditional-MFE Shared Ranker",
        phase="13AT",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_logits_dual_supervision_of_same_date_low_adverse_safety_percentile_and_true_HS_indicator_gte_0.50; "
            "head2=same_date_pure_mfe_percentile_within_true_HS_only"
        ),
        objective_description=(
            "MR-13AR/AS supervision-density control：architecture、true-HS=P50、same Safety/HS output head、true-HS Conditional-MFE "
            "target/sublist、independent MFE head、Seed42/split/optimizer、epoch selection與Pred-HS P50→Conditional-MFE inference固定。"
            "唯一scientific change是同一Safety/HS logits同時接受continuous Low-Adverse percentile full-list Delta-NDCG與binary HS/LS "
            "full-list Delta-NDCG supervision；Safety branch內兩loss固定1:1平均，再與Conditional-MFE branch固定1:1平均，等效component "
            "weights=continuous Safety 0.25 / binary HS 0.25 / Conditional-MFE 0.50，無loss-ratio sweep。Continuous Safety只作training "
            "representation shaping，runtime qualification仍只使用同一HS/Safety head的same-date P50 score；Conditional-MFE不吃Safety prediction。"
            "Seed42 Forward Model Gate only。"
        ),
        metric_scope="dual_supervised_hs_representation_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_dual_supervised_hs_qualification_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AU",
        experiment_name="MR-13AU Top-HS Safety NDCG@K + Conditional-MFE Shared Ranker",
        phase="13AU",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=top_HS_safety_relevance_where_LS_zero_and_HS_keep_same_date_low_adverse_percentile; "
            "head2=same_date_pure_mfe_percentile_within_true_HS_only"
        ),
        objective_description=(
            "MR-13AT後的decision-aligned Safety supervision control：architecture、true-HS=P50、Conditional-MFE target/sublist、"
            "Seed42/split/optimizer、epoch selection與Pred-HS P50→Conditional-MFE inference固定。唯一scientific change是Safety head"
            "不再用continuous/binary dual loss，而以LS relevance=0、HS relevance=原same-date Safety percentile建立Top-HS relevance，"
            "每天K固定為true-HS item數；primary Delta-NDCG discount在K之後歸零，因此直接優化預測top-half membership並在HS內"
            "讓越安全者優先。Conditional-MFE仍只有true-HS sublist supervision，不吃Safety prediction。無K sweep、cutoff sweep或loss ratio。"
        ),
        metric_scope="top_hs_safety_ndcg_at_true_hs_count_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_top_hs_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13T",
        experiment_name="MR-13T Daily Universal Safety + Raw-MFE + Direct HM/HS Tri-head Ranker",
        phase="13T",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=direct_hmhs_indicator_of_both_percentiles_ge_0.5"
        ),
        objective_description=(
            "MR-13S strict controlled contrast：300×10 raw normalized stock/0050 OHLCV、shared InceptionTime、"
            "Raw Safety head、stop-gradient Safety-conditioned Raw-MFE head、Seed42/split/optimizer與Raw-MFE epoch selection全部不變；"
            "唯一新增shared latent上的Direct HM/HS head，target=1[S>=0.5 and U>=0.5]。三head各自使用full-list Delta-NDCG RankNet，"
            "loss固定等權平均；HM/HS head不接Safety/MFE score arithmetic，不使用人工feature、breakout state、strategy state、threshold sweep或OOS fitting。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_direct_hmhs_tri_head",
        score_semantic_id="daily_safety_raw_mfe_direct_hmhs_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13V",
        experiment_name="MR-13V MR-13T Control + Nonlinear Direct HM/HS MLP Head",
        phase="13V",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=direct_hmhs_indicator_of_both_percentiles_ge_0.5"
        ),
        objective_description=(
            "MR-13T strict architecture contrast：canonical raw 300×10、shared InceptionTime trunk、Raw Safety head、"
            "stop-gradient Safety-conditioned Raw-MFE head、S/U/H targets、Seed42、daily-universal split、optimizer、"
            "full-list Delta-NDCG、三head等權loss與Raw-MFE Validation epoch selection全部不變；唯一scientific change是"
            "Direct HM/HS readout由Linear(latent,2)改為固定Linear(latent,latent)->ReLU->Linear(latent,2)。"
            "hidden width等於shared latent width，不設dropout、不做width/depth/activation/lambda sweep；Joint head仍不讀Safety/MFE predicted scores。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_direct_hmhs_nonlinear_head",
        score_semantic_id="daily_safety_raw_mfe_direct_hmhs_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13W",
        experiment_name="MR-13W MR-13V Architecture + Continuous Joint-Min Target",
        phase="13W",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=min(same_date_low_adverse_safety_percentile,same_date_pure_mfe_percentile)"
        ),
        objective_description=(
            "MR-13V strict target contrast：raw 300×10、shared InceptionTime trunk、Raw Safety head、"
            "stop-gradient Safety-conditioned Raw-MFE head、nonlinear Joint MLP architecture、Seed42、split、optimizer、"
            "full-list Delta-NDCG、三head固定等權loss與Raw-MFE Validation epoch selection全部不變；唯一fitting change是"
            "第三head supervision由binary 1[S>=0.5 and U>=0.5]改為continuous min(S,U)。"
            "不重新rank min target、不設threshold/temperature/weight，不使用OOS統計或strategy state。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_joint_min_nonlinear_head",
        score_semantic_id="daily_safety_raw_mfe_joint_min_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13X",
        experiment_name="MR-13X MR-13W Target + Joint Temporal Attention Pooling",
        phase="13X",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=min(same_date_low_adverse_safety_percentile,same_date_pure_mfe_percentile)"
        ),
        objective_description=(
            "MR-13W strict representation contrast：raw 300×10、InceptionTime trunk、Safety/Raw-MFE marginal global-average paths、"
            "S/U/Jmin targets、128->128->2 ReLU joint MLP、Seed42、split、optimizer、full-list Delta-NDCG、三head等權loss與"
            "Raw-MFE Validation epoch selection全部不變；唯一scientific change是Joint-Min branch在shared temporal feature map上"
            "使用parameter-free-shape scalar 1x1 attention scorer + softmax(time) + weighted temporal sum，取代Joint branch global mean。"
            "不加positional encoding/multi-head attention/dropout/temperature/attention-width sweep；marginal heads仍只讀原global mean。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_joint_min_attention_pool",
        score_semantic_id="daily_safety_raw_mfe_joint_min_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13Y",
        experiment_name="MR-13Y Joint-Min ModernTCN Raw-data Architecture Comparison",
        phase="13Y",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=min(same_date_low_adverse_safety_percentile,same_date_pure_mfe_percentile)"
        ),
        objective_description=(
            "MR-13X strict raw-data architecture-family contrast：canonical raw 300×10、S/U/Jmin targets、"
            "Safety/Raw-MFE marginal global-average semantics、Joint scalar temporal attention pooling、latent-width→latent-width→2 ReLU joint MLP、"
            "Seed42、split、optimizer、full-list Delta-NDCG、三head固定等權loss與Raw-MFE Validation epoch selection全部不變；"
            "唯一research dimension是temporal trunk由InceptionTime換成ModernTCN。ModernTCN固定沿用historical 9B未調參recipe："
            "6 blocks、96 channels、kernel51 depthwise temporal conv、4x pointwise expansion、BatchNorm、dropout0.10；"
            "head/attention widths僅隨trunk latent width自然為96，不另加adapter或capacity sweep。historical modern_tcn_v1仍維持legacy read-only。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_joint_min_modern_tcn_attention_pool",
        score_semantic_id="daily_safety_raw_mfe_joint_min_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13Z",
        experiment_name="MR-13Z Joint-Min Patch Transformer Raw-data Architecture Comparison",
        phase="13Z",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=min(same_date_low_adverse_safety_percentile,same_date_pure_mfe_percentile)"
        ),
        objective_description=(
            "MR-13X strict raw-data architecture-family contrast：canonical raw 300×10、S/U/Jmin targets、"
            "Safety/Raw-MFE marginal global-average semantics、Joint scalar temporal attention pooling、latent-width→latent-width→2 ReLU joint MLP、"
            "Seed42、split、optimizer、full-list Delta-NDCG、三head固定等權loss與Raw-MFE Validation epoch selection全部不變；"
            "唯一research dimension是temporal trunk由InceptionTime換成historical-recipe Patch Transformer。Patch Transformer固定9F未調參recipe："
            "non-overlap patch=10 bars、embedding=128、3 encoder layers、4 heads、FFN=256、sinusoidal position、dropout=0.10；"
            "marginal heads對30 patch tokens作global mean，Joint head對同一token map作scalar softmax attention；"
            "不做patch/depth/head/embedding/FFN/position/dropout/pooling sweep。historical patch_transformer_v1維持legacy read-only。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_joint_min_patch_transformer_attention_pool",
        score_semantic_id="daily_safety_raw_mfe_joint_min_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13U",
        experiment_name="MR-13U Daily Universal Direct HM/HS H-only Single-head Ranker",
        phase="13U",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "direct_hmhs_indicator=1[same_date_low_adverse_safety_percentile>=0.5 "
            "and same_date_pure_mfe_percentile>=0.5]"
        ),
        objective_description=(
            "MR-13T learnability ablation：canonical raw 300×10 normalized stock/0050 OHLCV、InceptionTime trunk、"
            "Seed42、daily-universal split、optimizer與full-list Delta-NDCG pairwise logistic不變；移除Safety/MFE heads與loss，"
            "只保留單一Direct HM/HS classifier，讓encoder僅受H supervision。Epoch selection事前固定為Validation HM/HS Pair concordance，"
            "同值才以Validation global PR-AUC tie-break；不加人工feature、不用breakout/strategy state、不做OOS fitting。"
        ),
        metric_scope="all_stock_days_direct_hmhs_h_only",
        score_semantic_id="daily_direct_hmhs_h_only_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE,
        model_research_id="MR-13F",
        experiment_name="MR-13F Daily Universal Direct-R Huber Regression",
        phase="13F",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="raw_daily_opportunity_no_time_r_v1_in_R_units",
        objective_description=(
            "同日全部合法stock-day直接預測daily_opportunity_no_time_r_v1 raw R；"
            "使用Huber loss且delta固定為1R，two-logit margin直接解讀為Predicted R"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_predicted_r",
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE,
        model_research_id="MR-13G",
        experiment_name="MR-13G Daily Universal Direct-R Mean Regression",
        phase="13G",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="raw_daily_opportunity_no_time_r_v1_in_R_units",
        objective_description=(
            "同日全部合法stock-day直接預測daily_opportunity_no_time_r_v1 raw R；"
            "使用MSE以population optimum對齊conditional mean E[R|X]，two-logit margin直接解讀為Predicted R"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_predicted_r",
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13I",
        experiment_name="MR-13I Daily Universal Cost-adjusted Risk-normalized 40D Ranker",
        phase="13I",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_risk_normalized_net_opportunity_r_v1",
        objective_description=(
            "MR-13E universe/backbone/40D horizon不變；只把target改為以historical-effective Min ROOS "
            "atr_len+atr_times_init定義generic initial risk distance，並使用canonical 1% sizing/fee/tax後的40D opportunity R；"
            "full-list Delta-NDCG pairwise objective不變"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_risk_normalized_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13J",
        experiment_name="MR-13J Daily Universal Risk-context 40D Ranker",
        phase="13J",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_target_as_MR-13I_daily_risk_normalized_net_opportunity_r_v1",
        objective_description=(
            "Target/universe/horizon/loss與MR-13I完全相同；唯一新增decision-time universal risk/economic geometry context "
            "(risk distance, capital per risk, cost per risk, risk capacity)"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_risk_normalized_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    ),
}

SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES = tuple(
    _CONTINUOUS_RANKER_RESEARCH_SPECS
)


def get_continuous_ranker_research_spec(
    experiment_profile: str,
) -> ContinuousRankerResearchSpec:
    profile_name = normalize_breakout_quality_experiment_profile(experiment_profile)
    spec = _CONTINUOUS_RANKER_RESEARCH_SPECS.get(profile_name)
    if spec is None:
        raise ValueError(
            f"continuous ranker profile缺少research spec登記: {profile_name}"
        )
    return spec




def resolve_breakout_quality_random_seed() -> int:
    """Return the single configured breakout-quality random seed."""

    resolved = int(BREAKOUT_QUALITY_RANDOM_SEED)
    if resolved < 0:
        raise ValueError("breakout quality random seed 必須是>=0的整數")
    return resolved

# =============================================================================
# DERIVED VALUES AND HELPER FUNCTIONS — do not edit unless changing implementation
# =============================================================================

# Workflow filter and architecture intentionally follow the active canonical identity.
# Strategy tools use the validated workflow identity; the model-research menu has its own active profile above.
BREAKOUT_QUALITY_WORKFLOW_FILTER_ID = BREAKOUT_QUALITY_DEFAULT_FILTER_ID
BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE = BREAKOUT_QUALITY_MODEL_ARCHITECTURE


# =============================================================================
# 9. Derived values and helper calculations (not user-adjustable)
# =============================================================================

# 下列 pretraining 欄位由具名 profile 自動展開，請調整 profile 定義而非直接修改衍生值。
_BREAKOUT_QUALITY_PRETRAINING_SETTINGS = get_breakout_quality_pretraining_profile(
    BREAKOUT_QUALITY_PRETRAINING_PROFILE
)
BREAKOUT_QUALITY_PRETRAINING_FAMILY = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.family
BREAKOUT_QUALITY_PRETRAINING_EPOCHS = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.epochs
BREAKOUT_QUALITY_PRETRAINING_BATCH_SIZE = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.batch_size
BREAKOUT_QUALITY_PRETRAINING_LEARNING_RATE = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.learning_rate
BREAKOUT_QUALITY_PRETRAINING_WEIGHT_DECAY = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.weight_decay
BREAKOUT_QUALITY_PRETRAINING_GRADIENT_CLIP_NORM = (
    _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.gradient_clip_norm
)
BREAKOUT_QUALITY_PRETRAINING_MIN_CROP_BARS = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.min_crop_bars
BREAKOUT_QUALITY_PRETRAINING_MASK_PROBABILITY = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.mask_probability
BREAKOUT_QUALITY_PRETRAINING_CONTRASTIVE_ALPHA = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.contrastive_alpha
BREAKOUT_QUALITY_PRETRAINING_TEMPORAL_UNIT = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.temporal_unit


def build_breakout_quality_default_high_len_values() -> tuple[int, ...]:
    values = set(build_breakout_optimizer_high_len_values())
    for value in BREAKOUT_QUALITY_EXTRA_HIGH_LENS:
        normalized = int(value)
        if normalized < 1:
            raise ValueError("BREAKOUT_QUALITY_EXTRA_HIGH_LENS 只能包含正整數")
        values.add(normalized)
    return tuple(sorted(values))


def _nearest_positive_odd(value: float, *, minimum: int = 3) -> int:
    lower = max(int(minimum), int(math.floor(float(value))))
    if lower % 2 == 0:
        lower -= 1
    lower = max(int(minimum), lower)
    if lower % 2 == 0:
        lower += 1
    upper = lower + 2
    if abs(float(value) - lower) <= abs(upper - float(value)):
        return int(lower)
    return int(upper)


def build_breakout_quality_inception_kernel_sizes() -> tuple[int, int, int]:
    depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
    target = int(BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS)
    residual_every = int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY)
    feature_window = int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS)
    if depth < 1:
        raise ValueError("BREAKOUT_QUALITY_INCEPTION_DEPTH 必須 >= 1")
    if residual_every < 1 or depth % residual_every != 0:
        raise ValueError(
            "BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY 必須 >= 1，且必須整除 INCEPTION_DEPTH"
        )
    minimum_target = 1 + depth * 8
    if target < minimum_target:
        raise ValueError(
            "BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS 過小；"
            f"depth={depth} 時至少需要 {minimum_target} bars，才能保留三尺度 kernels"
        )
    if target > feature_window:
        raise ValueError(
            "BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS 不得大於 "
            "BREAKOUT_QUALITY_FEATURE_WINDOW_BARS；模型不能從不存在的更早歷史學習"
        )

    minimum_max_kernel = math.ceil(1.0 + (target - 1) / depth)
    max_kernel = int(minimum_max_kernel)
    if max_kernel % 2 == 0:
        max_kernel += 1
    middle_kernel = _nearest_positive_odd(max_kernel / 2.0)
    short_kernel = _nearest_positive_odd(max_kernel / 4.0)
    kernels = (int(max_kernel), int(middle_kernel), int(short_kernel))
    if len(set(kernels)) != 3 or not (kernels[0] > kernels[1] > kernels[2] >= 3):
        raise ValueError(f"InceptionTime 自動產生的 kernels 不合法: {kernels}")
    return kernels


def resolve_breakout_quality_inception_receptive_field_bars() -> int:
    kernels = build_breakout_quality_inception_kernel_sizes()
    return 1 + int(BREAKOUT_QUALITY_INCEPTION_DEPTH) * (max(kernels) - 1)

def get_continuous_ranker_execution_recipe(
    experiment_profile: str,
) -> ContinuousRankerExecutionRecipe:
    """Resolve declarative profile identity into the canonical runtime contract."""

    profile_name = normalize_breakout_quality_experiment_profile(experiment_profile)
    profile = get_breakout_quality_experiment_profile(profile_name)
    spec = get_continuous_ranker_research_spec(profile_name)
    if profile.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        raise ValueError(f"continuous ranker recipe只接受continuous profile: {profile_name}")
    return build_continuous_ranker_execution_recipe(
        profile_name=profile_name,
        profile=profile,
        spec=spec,
        requires_continuous_target_artifact=(
            profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS
        ),
    )


@dataclass(frozen=True)
class BreakoutQualityContinuousRankerComparisonSettings:
    enabled: bool
    menu_label: str
    model_profiles: tuple[tuple[str, str], ...]
    reference_arm: str
    summary_pair: tuple[str, str]
    fixed_k_values: tuple[int, ...]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model_id for model_id, _profile in self.model_profiles)


def get_breakout_quality_continuous_ranker_comparison_settings(
) -> BreakoutQualityContinuousRankerComparisonSettings:
    model_profiles = tuple(
        (str(model_id).strip(), str(profile).strip())
        for model_id, profile in BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_PROFILES
    )
    if len(model_profiles) < 2:
        raise ValueError("continuous ranker comparison至少需要兩個model profile")
    model_ids = tuple(model_id for model_id, _profile in model_profiles)
    profiles = tuple(profile for _model_id, profile in model_profiles)
    if any(not model_id for model_id in model_ids) or any(not profile for profile in profiles):
        raise ValueError("continuous ranker comparison model id/profile不得為空")
    if len(set(model_ids)) != len(model_ids) or len(set(profiles)) != len(profiles):
        raise ValueError("continuous ranker comparison model id/profile不可重複")
    for profile in profiles:
        experiment = get_breakout_quality_experiment_profile(profile)
        if experiment.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
            raise ValueError(
                "continuous ranker comparison只允許continuous ranker profile: "
                f"{profile}"
            )

    menu_label = str(BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_MENU_LABEL).strip()
    if not menu_label:
        raise ValueError("continuous ranker comparison menu label不得為空")
    reference_arm = str(
        BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_REFERENCE_ARM
    ).strip()
    if not reference_arm:
        raise ValueError("continuous ranker comparison reference arm不得為空")

    raw_summary_pair = tuple(
        str(value).strip()
        for value in BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_SUMMARY_PAIR
    )
    if len(raw_summary_pair) != 2 or raw_summary_pair[0] == raw_summary_pair[1]:
        raise ValueError("continuous ranker comparison summary pair必須是兩個不同model id")
    if any(model_id not in model_ids for model_id in raw_summary_pair):
        raise ValueError("continuous ranker comparison summary pair必須存在於model profiles")
    available_contrasts = tuple(zip(model_ids[1:], model_ids[:-1]))
    if raw_summary_pair not in available_contrasts:
        raise ValueError(
            "continuous ranker comparison summary pair必須符合model profiles的相鄰比較順序"
        )

    fixed_k_values = tuple(
        int(value)
        for value in BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_FIXED_K_VALUES
    )
    if not fixed_k_values or any(value < 1 for value in fixed_k_values):
        raise ValueError("continuous ranker comparison fixed K sweep只能包含正整數")
    if len(set(fixed_k_values)) != len(fixed_k_values):
        raise ValueError("continuous ranker comparison fixed K sweep不可重複")
    if tuple(sorted(fixed_k_values)) != fixed_k_values:
        raise ValueError("continuous ranker comparison fixed K sweep必須遞增排序")

    return BreakoutQualityContinuousRankerComparisonSettings(
        enabled=bool(BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_ENABLED),
        menu_label=menu_label,
        model_profiles=model_profiles,
        reference_arm=reference_arm,
        summary_pair=(raw_summary_pair[0], raw_summary_pair[1]),
        fixed_k_values=fixed_k_values,
    )


@dataclass(frozen=True)
class BreakoutQualityModelTestSettings:
    model_profiles: tuple[tuple[str, str], ...]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model_id for model_id, _profile in self.model_profiles)


def get_breakout_quality_model_test_settings() -> BreakoutQualityModelTestSettings:
    model_profiles = tuple(
        (str(model_id).strip(), str(profile).strip())
        for model_id, profile in BREAKOUT_QUALITY_MODEL_TEST_PROFILES
    )
    required_training_model = tuple(
        str(value).strip() for value in BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
    )
    if len(required_training_model) != 2 or required_training_model not in model_profiles:
        raise ValueError(
            "模型比較／測試清單必須自動包含[1]/[2] canonical Training Model: "
            f"{required_training_model!r}"
        )
    if len(model_profiles) < 2:
        raise ValueError("模型比較／測試清單至少需要兩個model profile")
    model_ids = tuple(model_id for model_id, _profile in model_profiles)
    profiles = tuple(profile for _model_id, profile in model_profiles)
    if any(not value for value in (*model_ids, *profiles)):
        raise ValueError("模型比較／測試清單model id/profile不得為空")
    if len(set(model_ids)) != len(model_ids) or len(set(profiles)) != len(profiles):
        raise ValueError("模型比較／測試清單model id/profile不可重複")
    for model_id, profile_name in model_profiles:
        experiment = get_breakout_quality_experiment_profile(profile_name)
        if experiment.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
            raise ValueError(
                "模型比較／測試清單只允許continuous ranker profile: "
                f"{profile_name}"
            )
        research = get_continuous_ranker_research_spec(profile_name)
        if str(research.model_research_id) != str(model_id):
            raise ValueError(
                "模型比較／測試清單model id/profile research identity不一致: "
                f"configured={model_id}, resolved={research.model_research_id}, profile={profile_name}"
            )
    return BreakoutQualityModelTestSettings(model_profiles=model_profiles)


@dataclass(frozen=True)
class BreakoutQualityStandardModelComparisonSettings:
    menu_label: str
    model_profiles: tuple[tuple[str, str], ...]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model_id for model_id, _profile in self.model_profiles)


def get_breakout_quality_standard_model_comparison_settings(
) -> BreakoutQualityStandardModelComparisonSettings:
    model_profiles = get_breakout_quality_model_test_settings().model_profiles
    menu_label = str(BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_MENU_LABEL).strip()
    if not menu_label:
        raise ValueError("Standard Model SOP比較menu label不得為空")
    return BreakoutQualityStandardModelComparisonSettings(
        menu_label=menu_label,
        model_profiles=model_profiles,
    )


@dataclass(frozen=True)
class BreakoutQualityRollingTestModeSettings:
    mode_id: str
    label: str
    score_start_date: str
    score_end_date: str | None
    fold_months: int
    fold_anchor_date: str | None
    single_score_block: bool
    point_in_time_dirname: str | None


def get_breakout_quality_rolling_test_modes() -> tuple[BreakoutQualityRollingTestModeSettings, ...]:
    rows: list[BreakoutQualityRollingTestModeSettings] = []
    seen: set[str] = set()
    for mode_id, raw in dict(BREAKOUT_QUALITY_ROLLING_TEST_MODES).items():
        normalized_id = str(mode_id).strip()
        if not normalized_id or normalized_id in seen:
            raise ValueError(f"Rolling Test mode id空白或重複: {mode_id!r}")
        label = str(dict(raw).get("label") or "").strip()
        score_start_date = str(dict(raw).get("score_start_date") or "").strip()
        score_end_raw = dict(raw).get("score_end_date")
        score_end_date = None if score_end_raw in (None, "") else str(score_end_raw).strip()
        fold_months = int(dict(raw).get("fold_months", 0) or 0)
        single_score_block = bool(dict(raw).get("single_score_block", False))
        anchor_raw = dict(raw).get("fold_anchor_date")
        fold_anchor_date = None if anchor_raw in (None, "") else str(anchor_raw).strip()
        if fold_anchor_date is not None:
            try:
                date.fromisoformat(fold_anchor_date)
            except ValueError as exc:
                raise ValueError(
                    f"Rolling Test fold_anchor_date必須是YYYY-MM-DD: {fold_anchor_date!r}"
                ) from exc
        dirname_raw = dict(raw).get("point_in_time_dirname")
        dirname = None if dirname_raw in (None, "") else str(dirname_raw).strip()
        if not label:
            raise ValueError(f"Rolling Test mode label不可空白: {normalized_id}")
        if not score_start_date:
            raise ValueError(f"Rolling Test score_start_date不可空白: {normalized_id}")
        try:
            date.fromisoformat(score_start_date)
        except ValueError as exc:
            raise ValueError(
                f"Rolling Test score_start_date必須是YYYY-MM-DD: {score_start_date!r}"
            ) from exc
        if score_end_date is not None and score_end_date.lower() != "auto":
            try:
                date.fromisoformat(score_end_date)
            except ValueError as exc:
                raise ValueError(
                    f"Rolling Test score_end_date必須是YYYY-MM-DD或auto: {score_end_date!r}"
                ) from exc
        if fold_months < 1:
            raise ValueError(f"Rolling Test fold_months必須>=1: {normalized_id}")
        if dirname is not None and (Path(dirname).name != dirname or dirname in {".", ".."}):
            raise ValueError(f"Rolling Test point_in_time_dirname必須是安全單一資料夾名稱: {dirname!r}")
        rows.append(
            BreakoutQualityRollingTestModeSettings(
                mode_id=normalized_id,
                label=label,
                score_start_date=score_start_date,
                score_end_date=score_end_date,
                fold_months=fold_months,
                fold_anchor_date=fold_anchor_date,
                single_score_block=single_score_block,
                point_in_time_dirname=dirname,
            )
        )
        seen.add(normalized_id)
    if len(rows) < 2:
        raise ValueError("Rolling Test至少需要兩種執行深度")
    default_mode = str(BREAKOUT_QUALITY_DEFAULT_ROLLING_TEST_MODE).strip()
    if default_mode not in seen:
        raise ValueError(f"Rolling Test default mode不存在: {default_mode!r}")
    return tuple(rows)


def get_breakout_quality_rolling_test_mode(
    mode_id: str | None = None,
) -> BreakoutQualityRollingTestModeSettings:
    selected = str(mode_id or BREAKOUT_QUALITY_DEFAULT_ROLLING_TEST_MODE).strip()
    for item in get_breakout_quality_rolling_test_modes():
        if item.mode_id == selected:
            return item
    raise ValueError(f"未知Rolling Test mode: {selected!r}")


@dataclass(frozen=True)
class BreakoutQualityRollingTimingSettings:
    experiment_profile: str
    seed: int
    score_years: tuple[int, ...]


def get_breakout_quality_rolling_timing_settings() -> BreakoutQualityRollingTimingSettings:
    profile_name = str(
        BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE
        if BREAKOUT_QUALITY_ROLLING_TIMING_EXPERIMENT_PROFILE is None
        else BREAKOUT_QUALITY_ROLLING_TIMING_EXPERIMENT_PROFILE
    ).strip()
    if not profile_name:
        raise ValueError("Rolling Timing experiment profile不可空白")
    profile = get_breakout_quality_experiment_profile(profile_name)
    if profile.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        raise ValueError("Rolling Timing只支援continuous ranker profile")

    workflow = get_breakout_quality_workflow_settings(experiment_profile=profile_name)
    if not workflow.rolling_authorized:
        raise ValueError(
            f"Rolling Timing profile尚未授權current Rolling: {profile_name}"
        )

    seed = (
        resolve_breakout_quality_random_seed()
        if BREAKOUT_QUALITY_ROLLING_TIMING_SEED is None
        else int(BREAKOUT_QUALITY_ROLLING_TIMING_SEED)
    )
    if seed < 0:
        raise ValueError("Rolling Timing seed必須 >= 0")

    years = tuple(int(value) for value in BREAKOUT_QUALITY_ROLLING_TIMING_SCORE_YEARS)
    if not years:
        raise ValueError("Rolling Timing至少需要一個score year")
    if len(set(years)) != len(years):
        raise ValueError("Rolling Timing score years不可重複")
    if tuple(sorted(years)) != years:
        raise ValueError("Rolling Timing score years必須遞增排序")
    formal_start_raw = str(BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE).strip()
    formal_start = (
        None
        if formal_start_raw.lower() == "auto"
        else date.fromisoformat(formal_start_raw)
    )
    formal_end_raw = (
        ""
        if BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE is None
        else str(BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE).strip()
    )
    formal_end = (
        None
        if formal_end_raw.lower() in {"", "auto"}
        else date.fromisoformat(formal_end_raw)
    )
    for year in years:
        if year < 1900 or year > 2200:
            raise ValueError(f"Rolling Timing score year不合法: {year}")
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        if formal_start is not None and start < formal_start:
            raise ValueError(
                f"Rolling Timing year早於current Rolling起點: {year} < {formal_start.year}"
            )
        if formal_end is not None and end > formal_end:
            raise ValueError(
                f"Rolling Timing year晚於current完整Rolling終點: {year} > {formal_end.year}"
            )

    return BreakoutQualityRollingTimingSettings(
        experiment_profile=profile_name,
        seed=int(seed),
        score_years=years,
    )


# =============================================================================
# WORKFLOW RESOLUTION AND VALIDATION — do not edit unless changing implementation
# =============================================================================

WORKFLOW_STRATEGY_MODE_AUTO = "auto"
WORKFLOW_STRATEGY_MODE_HARD_FILTER = "hard-filter"
WORKFLOW_STRATEGY_MODE_SCORE_RANKING = "score-ranking"
SUPPORTED_WORKFLOW_STRATEGY_MODES = (
    WORKFLOW_STRATEGY_MODE_AUTO,
    WORKFLOW_STRATEGY_MODE_HARD_FILTER,
    WORKFLOW_STRATEGY_MODE_SCORE_RANKING,
)

WORKFLOW_SCORE_SOURCE_AUTO = "auto"
WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME = "selection_point_in_time"
WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS = "final_selection_model_oos"
WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME = "canonical_runtime"
SUPPORTED_WORKFLOW_SCORE_SOURCES = (
    WORKFLOW_SCORE_SOURCE_AUTO,
    WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS,
    WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
)

WORKFLOW_BUY_SORT_AUTO = "auto"
WORKFLOW_BUY_SORT_ORIGINAL = "original"
WORKFLOW_BUY_SORT_SCORE_DESC = "breakout_quality_score_desc"

@dataclass(frozen=True)
class BreakoutQualityWorkflowSettings:
    filter_id: str
    model_architecture: str
    experiment_profile: str
    training_objective: str
    continuous_target_id: str | None
    training_label_scope: str
    training_sample_scope: str
    seed: int
    point_in_time_score_start_date: str
    point_in_time_coverage_reference_start_date: str
    point_in_time_score_end_date: str | None
    point_in_time_fold_months: int
    point_in_time_inner_validation_months: int
    point_in_time_train_window_months: int | None
    point_in_time_min_train_groups: int
    point_in_time_min_validation_groups: int
    point_in_time_min_score_groups: int
    point_in_time_resume: bool
    strategy_dataset: str
    strategy_param_policy: str
    strategy_max_positions: int
    strategy_rotation: str
    strategy_trials_per_fold: int
    strategy_fixed_risk: float
    strategy_max_position_cap_pct: float
    strategy_comparison_mode: str
    strategy_score_source: str
    strategy_buy_sort: str
    runtime_strategy_enabled: bool
    runtime_ranking_policy: str
    runtime_ranking_options: dict[str, Any]

    @property
    def is_binary_classification(self) -> bool:
        return self.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION

    @property
    def is_continuous_ranker(self) -> bool:
        return self.training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES

    @property
    def supports_point_in_time_scores(self) -> bool:
        return bool(
            self.is_continuous_ranker
            and self.training_sample_scope
            in SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLE_SCOPES
        )

    @property
    def rolling_authorized(self) -> bool:
        """Whether current Rolling model evidence is authorized for this profile.

        Current model-workflow membership is the work-item SSOT: the active Training
        Profile drives [1]/[2], and the shared Model Compare/Test List drives [3]～[6].
        Historical recipe authorization remains accepted for compatibility, but it is
        no longer a second selector that can block a currently configured model target.
        """

        if not self.supports_point_in_time_scores:
            return False
        recipe = get_continuous_ranker_execution_recipe(self.experiment_profile)
        # B313/B325: current workflow membership is the sole current-mode selector.
        # Historical recipe authorization remains readable for profiles outside the
        # current workflow, but it cannot veto a configured Training/Compare member.
        return bool(
            recipe.current_time_validation_authorized
            or is_breakout_quality_model_workflow_profile(self.experiment_profile)
        )

    @property
    def robustness_authorized(self) -> bool:
        """Whether current multi-seed model evidence is authorized for this profile.

        Robustness is a property of shared Model Compare/Test membership.  The current
        Training Model is structurally injected into that same list, so no per-profile
        authorization flag may create a second selector.
        """

        return bool(
            self.is_continuous_ranker
            and is_breakout_quality_model_test_profile(self.experiment_profile)
        )

    def as_manifest_payload(self) -> dict[str, Any]:
        payload = {
            "filter_id": self.filter_id,
            "model_architecture": self.model_architecture,
            "experiment_profile": self.experiment_profile,
            "training_objective": self.training_objective,
            "continuous_target_id": self.continuous_target_id,
            "training_label_scope": self.training_label_scope,
            "seed": int(self.seed),
            "point_in_time": {
                "enabled": bool(self.rolling_authorized),
                "structurally_supported": bool(self.supports_point_in_time_scores),
                "research_authorized": bool(self.rolling_authorized),
                "score_start_date": self.point_in_time_score_start_date,
                "coverage_reference_start_date": (
                    self.point_in_time_coverage_reference_start_date
                ),
                "score_end_date": self.point_in_time_score_end_date,
                "fold_months": int(self.point_in_time_fold_months),
                "inner_validation_months": int(
                    self.point_in_time_inner_validation_months
                ),
                "train_window_months": (
                    None
                    if self.point_in_time_train_window_months is None
                    else int(self.point_in_time_train_window_months)
                ),
                "min_train_groups": int(self.point_in_time_min_train_groups),
                "min_validation_groups": int(
                    self.point_in_time_min_validation_groups
                ),
                "min_score_groups": int(self.point_in_time_min_score_groups),
                "resume": bool(self.point_in_time_resume),
            },
            "strategy": {
                "dataset": self.strategy_dataset,
                "param_policy": self.strategy_param_policy,
                "max_positions": int(self.strategy_max_positions),
                "rotation": self.strategy_rotation,
                "adaptation": {
                    "trials_per_fold": int(self.strategy_trials_per_fold),
                    "trial_source": "OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT",
                    "fixed_risk": float(self.strategy_fixed_risk),
                    "max_position_cap_pct": float(
                        self.strategy_max_position_cap_pct
                    ),
                },
                "comparison_mode": self.strategy_comparison_mode,
                "score_source": self.strategy_score_source,
                "buy_sort": self.strategy_buy_sort,
            },
            "runtime": {
                "enabled": bool(self.runtime_strategy_enabled),
                "score_source": WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
                "ranking_policy": self.runtime_ranking_policy,
                "ranking_options": dict(self.runtime_ranking_options),
            },
        }
        if self.training_sample_scope != TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
            payload["training_sample_scope"] = self.training_sample_scope
        return payload



def _resolve_strategy_defaults(training_objective: str) -> tuple[str, str, str]:
    if training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
        return (
            WORKFLOW_STRATEGY_MODE_HARD_FILTER,
            WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
            WORKFLOW_BUY_SORT_ORIGINAL,
        )
    if training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        return (
            WORKFLOW_STRATEGY_MODE_SCORE_RANKING,
            WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            WORKFLOW_BUY_SORT_SCORE_DESC,
        )
    raise ValueError(f"不支援的 workflow training objective: {training_objective!r}")


def _resolve_auto(value: str, *, auto_value: str, resolved_default: str) -> str:
    normalized = str(value).strip()
    return resolved_default if normalized == auto_value else normalized


def get_breakout_quality_workflow_settings(
    *, experiment_profile: str | None = None
) -> BreakoutQualityWorkflowSettings:
    random_seed = resolve_breakout_quality_random_seed()
    resolved_experiment_profile = str(
        BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE
        if experiment_profile is None
        else experiment_profile
    ).strip()
    profile = get_breakout_quality_experiment_profile(resolved_experiment_profile)
    if profile.training_objective not in {
        TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
        *CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
    }:
        raise ValueError(
            "workflow experiment profile必須是binary classification或continuous ranker"
        )
    try:
        coverage_reference_start = date.fromisoformat(
            str(BREAKOUT_QUALITY_POINT_IN_TIME_COVERAGE_REFERENCE_START_DATE).strip()
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "point-in-time coverage reference start date必須是YYYY-MM-DD合法日期"
        ) from exc
    if int(BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS) < 1:
        raise ValueError("point-in-time fold months 必須 >= 1")
    if int(BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS) < 1:
        raise ValueError("point-in-time inner validation months 必須 >= 1")
    if BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS is not None:
        if int(BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS) <= int(
            BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS
        ):
            raise ValueError(
                "point-in-time fixed train window必須大於inner validation months"
            )
    if min(
        int(BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS),
        int(BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS),
        int(BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS),
    ) < 1:
        raise ValueError("point-in-time minimum group counts 必須 >= 1")
    if BREAKOUT_QUALITY_STRATEGY_DATASET not in {"reduced", "full"}:
        raise ValueError("strategy dataset 必須是 reduced 或 full")
    if BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY not in {
        "auto",
        "base-finalist-best",
        "base-finalists-agree",
    }:
        raise ValueError(
            "strategy param policy 必須是auto、base-finalist-best或base-finalists-agree"
        )
    if int(BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS) < 1:
        raise ValueError("strategy max positions 必須 >= 1")
    if BREAKOUT_QUALITY_STRATEGY_ROTATION not in {"off", "on"}:
        raise ValueError("strategy rotation 必須是 off 或 on")
    if int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT) < 1:
        raise ValueError("outer rolling optimizer trials per fold 必須 >= 1")
    if not 0.0 < float(DEFAULT_FIXED_RISK) <= 1.0:
        raise ValueError("strategy fixed risk 必須介於0與1")
    if not 0.0 < float(DEFAULT_MAX_POSITION_CAP_PCT) <= 1.0:
        raise ValueError("strategy max position cap pct 必須介於0與1")

    raw_comparison_mode = str(BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE).strip()
    if raw_comparison_mode not in SUPPORTED_WORKFLOW_STRATEGY_MODES:
        raise ValueError(
            "strategy comparison mode 必須是auto、hard-filter或score-ranking"
        )
    raw_score_source = str(BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE).strip()
    if raw_score_source not in SUPPORTED_WORKFLOW_SCORE_SOURCES:
        raise ValueError(
            "strategy score source 必須是auto、selection_point_in_time、"
            "final_selection_model_oos或canonical_runtime"
        )
    raw_buy_sort = str(BREAKOUT_QUALITY_STRATEGY_BUY_SORT).strip()
    if not raw_buy_sort:
        raise ValueError("strategy buy sort不可為空白")

    default_mode, default_score_source, default_buy_sort = _resolve_strategy_defaults(
        profile.training_objective
    )
    strategy_comparison_mode = _resolve_auto(
        raw_comparison_mode,
        auto_value=WORKFLOW_STRATEGY_MODE_AUTO,
        resolved_default=default_mode,
    )
    strategy_score_source = _resolve_auto(
        raw_score_source,
        auto_value=WORKFLOW_SCORE_SOURCE_AUTO,
        resolved_default=default_score_source,
    )
    strategy_buy_sort = _resolve_auto(
        raw_buy_sort,
        auto_value=WORKFLOW_BUY_SORT_AUTO,
        resolved_default=default_buy_sort,
    )

    if strategy_comparison_mode == WORKFLOW_STRATEGY_MODE_HARD_FILTER:
        if strategy_score_source != WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME:
            raise ValueError("hard-filter策略比較只接受canonical_runtime score source")
        if strategy_buy_sort != WORKFLOW_BUY_SORT_ORIGINAL:
            raise ValueError("hard-filter策略比較必須沿用original buy-sort")
    elif strategy_comparison_mode == WORKFLOW_STRATEGY_MODE_SCORE_RANKING:
        if strategy_score_source not in {
            WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS,
            WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
        }:
            raise ValueError("score-ranking策略比較缺少合法score source")
        if strategy_buy_sort != WORKFLOW_BUY_SORT_SCORE_DESC:
            raise ValueError(
                "score-ranking策略比較目前只支援breakout_quality_score_desc"
            )
    else:
        raise ValueError(
            f"不支援的 resolved strategy comparison mode: {strategy_comparison_mode!r}"
        )

    runtime_ranking_policy = str(BREAKOUT_QUALITY_RUNTIME_RANKING_POLICY).strip()
    runtime_ranking_options = dict(BREAKOUT_QUALITY_RUNTIME_RANKING_OPTIONS or {})
    runtime_strategy_enabled = bool(
        BREAKOUT_QUALITY_RUNTIME_STRATEGY_ENABLED
        and resolved_experiment_profile == str(BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE).strip()
        and profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    )
    if runtime_strategy_enabled:
        if not runtime_ranking_policy:
            raise ValueError("正式runtime ranking policy不可為空白")
        expected_runtime_options = {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": False,
        }
        if runtime_ranking_options != expected_runtime_options:
            raise ValueError(
                "正式runtime必須維持Gate已驗證的exact K/R0 contract: "
                f"expected={expected_runtime_options}, actual={runtime_ranking_options}"
            )

    return BreakoutQualityWorkflowSettings(
        filter_id=str(BREAKOUT_QUALITY_WORKFLOW_FILTER_ID),
        model_architecture=str(profile.model_architecture or BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE),
        experiment_profile=resolved_experiment_profile,
        training_objective=str(profile.training_objective),
        continuous_target_id=(
            None
            if profile.continuous_target_id is None
            else str(profile.continuous_target_id)
        ),
        training_label_scope=str(profile.training_label_scope),
        training_sample_scope=str(profile.training_sample_scope),
        seed=random_seed,
        point_in_time_score_start_date=str(
            BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE
        ),
        point_in_time_coverage_reference_start_date=(
            coverage_reference_start.isoformat()
        ),
        point_in_time_score_end_date=(
            None
            if BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE is None
            else str(BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE)
        ),
        point_in_time_fold_months=int(BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS),
        point_in_time_inner_validation_months=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS
        ),
        point_in_time_train_window_months=(
            None
            if BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS is None
            else int(BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS)
        ),
        point_in_time_min_train_groups=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS
        ),
        point_in_time_min_validation_groups=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS
        ),
        point_in_time_min_score_groups=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS
        ),
        point_in_time_resume=bool(BREAKOUT_QUALITY_POINT_IN_TIME_RESUME),
        strategy_dataset=str(BREAKOUT_QUALITY_STRATEGY_DATASET),
        strategy_param_policy=str(BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY),
        strategy_max_positions=int(BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS),
        strategy_rotation=str(BREAKOUT_QUALITY_STRATEGY_ROTATION),
        strategy_trials_per_fold=int(
            OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
        ),
        strategy_fixed_risk=float(DEFAULT_FIXED_RISK),
        strategy_max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        strategy_comparison_mode=strategy_comparison_mode,
        strategy_score_source=strategy_score_source,
        strategy_buy_sort=strategy_buy_sort,
        runtime_strategy_enabled=runtime_strategy_enabled,
        runtime_ranking_policy=runtime_ranking_policy,
        runtime_ranking_options=runtime_ranking_options,
    )


@dataclass(frozen=True)
class BreakoutQualityContinuousRankerPITGateSettings:
    model_profiles: tuple[tuple[str, str], ...]
    seed: int
    point_in_time_score_start_date: str
    point_in_time_score_end_date: str | None
    point_in_time_fold_months: int
    point_in_time_inner_validation_months: int

    def __post_init__(self) -> None:
        if len(self.model_profiles) < 2:
            raise ValueError("PIT Gate batch至少需要兩個model profile")
        seen_ids: set[str] = set()
        seen_profiles: set[str] = set()
        reference = None
        for model_id, profile_name in self.model_profiles:
            model_id = str(model_id).strip()
            profile_name = normalize_breakout_quality_experiment_profile(profile_name)
            if not model_id or model_id in seen_ids:
                raise ValueError(f"PIT Gate model ID空白或重複: {model_id!r}")
            if profile_name in seen_profiles:
                raise ValueError(f"PIT Gate experiment profile重複: {profile_name}")
            seen_ids.add(model_id)
            seen_profiles.add(profile_name)
            spec = get_continuous_ranker_research_spec(profile_name)
            if spec.model_research_id != model_id:
                raise ValueError(
                    "PIT Gate model ID/profile research identity不一致: "
                    f"{model_id} != {spec.model_research_id}"
                )
            workflow = get_breakout_quality_workflow_settings(experiment_profile=profile_name)
            if not workflow.rolling_authorized:
                raise ValueError(f"PIT Gate profile尚未授權Rolling PIT: {profile_name}")
            contract = (
                workflow.filter_id,
                workflow.model_architecture,
                workflow.continuous_target_id,
                workflow.training_sample_scope,
                workflow.seed,
                workflow.point_in_time_score_start_date,
                workflow.point_in_time_score_end_date,
                workflow.point_in_time_fold_months,
                workflow.point_in_time_inner_validation_months,
            )
            if reference is None:
                reference = contract
            elif contract != reference:
                raise ValueError(
                    "PIT Gate profiles的Dataset/Target/Seed/period/fold contract不一致"
                )


def get_breakout_quality_continuous_ranker_pit_gate_settings(
) -> BreakoutQualityContinuousRankerPITGateSettings:
    return BreakoutQualityContinuousRankerPITGateSettings(
        model_profiles=tuple(
            (str(model_id), str(profile_name))
            for model_id, profile_name in BREAKOUT_QUALITY_CONTINUOUS_RANKER_PIT_GATE_PROFILES
        ),
        seed=resolve_breakout_quality_random_seed(),
        point_in_time_score_start_date=str(BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE),
        point_in_time_score_end_date=(
            None
            if BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE is None
            else str(BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE)
        ),
        point_in_time_fold_months=int(BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS),
        point_in_time_inner_validation_months=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS
        ),
    )


def get_breakout_quality_model_research_settings() -> BreakoutQualityWorkflowSettings:
    """Return the canonical [1]/[2] training identity without changing strategy defaults."""

    model_id, profile_name = (
        str(value).strip() for value in BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
    )
    if not model_id or not profile_name:
        raise ValueError("canonical Training Model model id/profile不得為空")
    research = get_continuous_ranker_research_spec(profile_name)
    if str(research.model_research_id).strip() != model_id:
        raise ValueError(
            "canonical Training Model model id/profile research identity不一致: "
            f"configured={model_id}, resolved={research.model_research_id}, "
            f"profile={profile_name}"
        )
    return get_breakout_quality_workflow_settings(experiment_profile=profile_name)


__all__ = [
    'BREAKOUT_QUALITY_BENCHMARK_TICKER',
    'BREAKOUT_QUALITY_TORCH_DEVICE',
    'BREAKOUT_QUALITY_USE_MIXED_PRECISION',
    'BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE',
    'BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS',
    'BREAKOUT_QUALITY_ALLOW_TF32',
    'BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE',
    'BREAKOUT_QUALITY_DEFAULT_EPOCHS',
    'BREAKOUT_QUALITY_DEFAULT_FILTER_ID',
    'BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE',
    'BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE',
    'BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE',
    'BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM',
    'BREAKOUT_QUALITY_RANDOM_SEED',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_PIT_GATE_PROFILES',
    'BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD',
    'BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY',
    'BREAKOUT_QUALITY_FINAL_REFIT_MODE',
    'BREAKOUT_QUALITY_CLASS_WEIGHT_MODE',
    'BREAKOUT_QUALITY_TIME_WEIGHT_MODE',
    'BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE',
    'BREAKOUT_QUALITY_EVALUATION_WORKERS',
    'BREAKOUT_QUALITY_PIT_EPOCH_SELECTION_LIGHTWEIGHT_METRICS',
    'BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K',
    'BREAKOUT_QUALITY_TARGET_COMPARISON_BARRIER_BAND_RETURN',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_ENABLED',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_MENU_LABEL',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_PROFILES',
    'BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES',
    'BREAKOUT_QUALITY_MODEL_TEST_PROFILES',
    'BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_MENU_LABEL',
    'BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_PROFILES',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_REFERENCE_ARM',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_SUMMARY_PAIR',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_FIXED_K_VALUES',
    'BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK',
    'BREAKOUT_QUALITY_PRETRAINING_PROFILE',
    'BREAKOUT_QUALITY_PRETRAINING_FAMILY',
    'BREAKOUT_QUALITY_PRETRAINING_STRIDE',
    'BREAKOUT_QUALITY_PRETRAINING_EPOCHS',
    'BREAKOUT_QUALITY_PRETRAINING_BATCH_SIZE',
    'BREAKOUT_QUALITY_PRETRAINING_LEARNING_RATE',
    'BREAKOUT_QUALITY_PRETRAINING_WEIGHT_DECAY',
    'BREAKOUT_QUALITY_PRETRAINING_GRADIENT_CLIP_NORM',
    'BREAKOUT_QUALITY_PRETRAINING_MIN_CROP_BARS',
    'BREAKOUT_QUALITY_PRETRAINING_MASK_PROBABILITY',
    'BREAKOUT_QUALITY_PRETRAINING_CONTRASTIVE_ALPHA',
    'BREAKOUT_QUALITY_PRETRAINING_TEMPORAL_UNIT',
    'BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES',
    'BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS',
    'BREAKOUT_QUALITY_EXTRA_HIGH_LENS',
    'BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA',
    'BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE',
    'BREAKOUT_QUALITY_FEATURE_WINDOW_BARS',
    'BREAKOUT_QUALITY_INCEPTION_DEPTH',
    'BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS',
    'BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY',
    'BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS',
    'BREAKOUT_QUALITY_MARKET_SET_BASE_FEATURES',
    'BREAKOUT_QUALITY_MARKET_SET_STOCK_EMBEDDING_DIM',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_CHANNELS',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_KERNEL_SIZE',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_STRIDE',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION_GROUPS',
    'BREAKOUT_QUALITY_MARKET_SET_QUERY_COUNT',
    'BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT',
    'BREAKOUT_QUALITY_MARKET_SET_ATTENTION_HEADS',
    'BREAKOUT_QUALITY_MARKET_SET_EMBEDDING_DIM',
    'BREAKOUT_QUALITY_MARKET_SET_FUSION_HIDDEN_DIM',
    'BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO',
    'BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS',
    'BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH',
    'BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS',
    'BREAKOUT_QUALITY_LABEL_HORIZON_BARS',
    'BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS',
    'BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN',
    'BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN',
    'BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO',
    'BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES',
    'BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES',
    'BREAKOUT_QUALITY_EXPERIMENT_PROFILE',
    'BREAKOUT_QUALITY_MODEL_ARCHITECTURE',
    'BREAKOUT_QUALITY_USE_INNER_VALIDATION',
    'build_breakout_quality_default_high_len_values',
    'build_breakout_quality_inception_kernel_sizes',
    'resolve_breakout_quality_inception_receptive_field_bars',
    'ADAMW_ONLY_EXPERIMENT_PROFILE',
    'ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE',
    'AUGMENTATION_NONE',
    'AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK',
    'BASELINE_EXPERIMENT_PROFILE',
    'HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE',
    'UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE',
    'UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE',
    'STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE',
    'STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE',
    'STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE',
    'STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE',
    'STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE',
    'TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE',
    'BreakoutQualityExperimentProfile',
    'BreakoutQualityPretrainingProfile',
    'LR_SCHEDULE_LINEAR_WARMUP_COSINE',
    'LR_SCHEDULE_NONE',
    'SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS',
    'SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES',
    'SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES',
    'SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES',
    'SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS',
    'SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES',
    'SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES',
    'SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES',
    'CONTINUOUS_RANKER_TRAINING_OBJECTIVES',
    'SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES',
    'TRAINING_LABEL_SCOPE_ALL',
    'TRAINING_LABEL_SCOPE_PASS_ONLY',
    'TRAINING_OBJECTIVE_BINARY_CLASSIFICATION',
    'TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION',
    'TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION',
    'TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION',
    'TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING',
    'TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING',
    'TRAINING_SAMPLING_ALL_EVENT_ROWS',
    'TRAINING_SAMPLING_UNIQUE_TICKER_DATE',
    'TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS',
    'TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS',
    'SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLE_SCOPES',
    'DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE',
    'DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE',
    'DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE',
    'DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE',
    'DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID',
    'PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID',
    'PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID',
    'PREDICTED_UPSIDE_CONTEXT_SCHEMA_VERSION',
    'PREDICTED_UPSIDE_CONTEXT_STAGE1_PROFILE',
    'PREDICTED_UPSIDE_CONTEXT_STAGE1_ARCHITECTURE',
    'PREDICTED_UPSIDE_CONTEXT_STAGE1_RESEARCH_ID',
    'PREDICTED_UPSIDE_CONTEXT_STAGE1_SEED',
    'get_predicted_upside_context_contract',
    'PREDICTED_SAFETY_CONTEXT_SCHEMA_VERSION',
    'PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE',
    'PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE',
    'PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID',
    'PREDICTED_SAFETY_CONTEXT_STAGE1_SEED',
    'get_predicted_safety_context_contract',
    'PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE',
    'PREDICTED_SAFETY_CONTEXT_OWNER_ARCHITECTURE',
    'get_predicted_safety_pure_mfe_contract',
    'get_predicted_safety_pair_weight_contract',
    'get_high_safety_weighted_pure_mfe_contract',
    'DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE',
    'DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE',
    'ContinuousRankerResearchSpec',
    'ContinuousRankerContextPolicy',
    'ContinuousRankerObjectivePolicy',
    'ContinuousRankerDependencySpec',
    'BreakoutQualityOutputSchema',
    'BREAKOUT_QUALITY_OUTPUT_SCHEMA',
    'CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE',
    'CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE',
    'CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY',
    'CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE',
    'CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT',
    'CONTINUOUS_RANKER_CONTEXT_ROLE_TARGET_TRANSFORM',
    'CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT',
    'CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE',
    'CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY',
    'CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY',
    'SUPPORTED_CONTINUOUS_RANKER_PAIR_WEIGHT_POLICIES',
    'CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR',
    'CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_PARETO_COMPONENTS',
    'CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT',
    'ContinuousRankerExecutionRecipe',
    'CONTINUOUS_RANKER_TRAINER_EVENT',
    'CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL',
    'CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR',
    'CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED',
    'CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE',
    'CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG',
    'CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG',
    'CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE',
    'SUPPORTED_CONTINUOUS_RANKER_PAIRWISE_REDUCTIONS',
    'get_continuous_ranker_research_spec',
    'get_continuous_ranker_execution_recipe',
    'SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES',
    'TIME_WEIGHT_MODE_DATE_BALANCED',
    'TIME_WEIGHT_MODE_NONE',
    'TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT',
    'SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES',
    'TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM',
    'TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE',
    'SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS',
    'get_breakout_quality_experiment_profile',
    'resolve_breakout_quality_random_seed',
    'get_breakout_quality_pretraining_profile',
    'build_breakout_quality_pretraining_profile_payload',
    'normalize_breakout_quality_experiment_profile',
    'normalize_breakout_quality_pretraining_profile',
    'BREAKOUT_QUALITY_RUNTIME_STRATEGY_ENABLED',
    'BREAKOUT_QUALITY_RUNTIME_RANKING_POLICY',
    'BREAKOUT_QUALITY_RUNTIME_RANKING_OPTIONS',
    'BREAKOUT_QUALITY_STRATEGY_BUY_SORT',
    'BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE',
    'BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE',
    'BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE',
    'BreakoutQualityContinuousRankerComparisonSettings',
    'BreakoutQualityRollingTestModeSettings',
    'BreakoutQualityRollingTimingSettings',
    'BreakoutQualityWorkflowSettings',
    'SUPPORTED_WORKFLOW_SCORE_SOURCES',
    'SUPPORTED_WORKFLOW_STRATEGY_MODES',
    'WORKFLOW_BUY_SORT_AUTO',
    'WORKFLOW_BUY_SORT_ORIGINAL',
    'WORKFLOW_BUY_SORT_SCORE_DESC',
    'WORKFLOW_SCORE_SOURCE_AUTO',
    'WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME',
    'WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS',
    'WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME',
    'WORKFLOW_STRATEGY_MODE_AUTO',
    'WORKFLOW_STRATEGY_MODE_HARD_FILTER',
    'WORKFLOW_STRATEGY_MODE_SCORE_RANKING',
    'get_breakout_quality_continuous_ranker_comparison_settings',
    'get_breakout_quality_standard_model_comparison_settings',
    'get_breakout_quality_rolling_test_modes',
    'get_breakout_quality_rolling_test_mode',
    'get_breakout_quality_rolling_timing_settings',
    'get_breakout_quality_workflow_settings',
    'BreakoutQualityContinuousRankerPITGateSettings',
    'get_breakout_quality_continuous_ranker_pit_gate_settings',
    'get_breakout_quality_model_research_settings',
    'get_breakout_quality_model_test_settings',
    'get_breakout_quality_model_workflow_profile_names',
    'is_breakout_quality_model_workflow_profile',
    'is_breakout_quality_model_test_profile',
]

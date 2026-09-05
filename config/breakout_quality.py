"""User/project-adjustable breakout-quality settings.

Only declarative settings and current workflow membership belong here.  Scientific/profile
registries live in ``core.breakout_quality_registry``; derived settings and runtime
resolution live in ``core.breakout_quality_policy``.
"""

from __future__ import annotations

from config.research import RESEARCH_SINGLE_SEED
from config.breakout_policy import BREAKOUT_DEFAULT_HIGH_LEN
from config.execution_policy import DEFAULT_PORTFOLIO_MAX_POSITIONS, DEFAULT_PORTFOLIO_ROTATION
from core.breakout_quality_registry import merge_breakout_quality_model_profiles

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
# - MR-13BK MR-13M target on BJ BiGRU backbone: "daily_universal_bigru_full_horizon_low_adverse_full_list_ndcg_pairwise"
# - MR-13BL AO + Price-Volume structural Safety representation: "daily_universal_price_volume_structure_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13BM BL + local Price-Volume zone tokens: "daily_universal_price_volume_structure_local_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13BN BL + high-resolution local Price-Volume 2D: "daily_universal_price_volume_multiscale_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13BO BN + explicit signed-price/time-age coordinate semantics: "daily_universal_price_volume_position_aware_multiscale_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13BP AO + explicit pairwise temporal-relation bias on Safety self-attention: "daily_universal_shared_safety_pairwise_relation_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13BQ AO + PIT-safe Stock-Code encoder pretraining before full AO fine-tune: "daily_universal_scc_pretrained_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
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
# - MR-13AV AO objective + task-specific final residual Safety/MFE representation: "daily_universal_task_specific_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13AW AO objective + Safety-specific scalar temporal attention pooling; MFE keeps GAP: "daily_universal_shared_safety_attn_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13AX AV task-specific high-level representation + Safety scalar temporal attention pooling: "daily_universal_task_specific_safety_attn_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13AY AO objective + Safety single-head temporal self-attention interaction; MFE keeps AO GAP: "daily_universal_shared_safety_self_attn_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13AZ AO objective + independent frozen-recipe Patch Safety encoder + InceptionTime Conditional-MFE encoder: "daily_universal_patch_safety_inception_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13BA AO objective + shared InceptionTime full-window receptive field: "daily_universal_shared_safety_hs_conditional_mfe_full_window_rf_full_list_ndcg_pairwise"
# - MR-13BB AO objective + shared InceptionTime pure width/capacity scaling: "daily_universal_shared_safety_hs_conditional_mfe_wide_full_list_ndcg_pairwise"
# - MR-13BC AO objective + 600-bar long-horizon input control: "daily_universal_shared_safety_hs_conditional_mfe_600bar_full_list_ndcg_pairwise"
# - MR-13BD AO objective + depth-12 parameter/RF-matched hierarchical control: "daily_universal_shared_safety_hs_conditional_mfe_deep_full_list_ndcg_pairwise"
# - MR-13BE AO objective + 600-bar × wide-capacity factorial interaction: "daily_universal_shared_safety_hs_conditional_mfe_600bar_wide_full_list_ndcg_pairwise"
# - MR-13BF AO objective + parameter-matched full-resolution day-token Transformer: "daily_universal_day_token_transformer_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13BG AO objective + parameter-matched gated recurrent GRU backbone: "daily_universal_gru_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
# - MR-13BJ BG controlled extension + parameter-matched bidirectional GRU: "daily_universal_bigru_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
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
    "MR-13AO",
    "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise",
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
BREAKOUT_QUALITY_TRADE_PATH_LABEL_WORKERS = 4  # A2 realized trade-path label builder 的 CPU workers；只改 execution strategy。


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
    ("MR-13AO", "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"),
)



# Canonical [3]~[6] Model Compare/Test List.  Its required subset is generated from the
# same canonical pair that drives [1]/[2]; reference controls may only add membership.
# Therefore every current training DL is structurally guaranteed to appear in Forward /
# Rolling comparison and Forward / Rolling robustness without a second manual edit.
BREAKOUT_QUALITY_MODEL_TEST_PROFILES = merge_breakout_quality_model_profiles(
    BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES,
    (BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE,),
)
BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_MENU_LABEL = "模型比較（Standard SOP）"
# Backward-compatible alias; there is no second model list.
BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_PROFILES = BREAKOUT_QUALITY_MODEL_TEST_PROFILES


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


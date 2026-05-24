import math

from core.seed_ensemble_policy import build_seed_ensemble_policy_snapshot

# 期望值 (EV) 算法切換
# 'A' = 嚴格 R_Multiple 期望值 (Mean R)
# 'B' = 傳統實際盈虧期望值 (Win% * Payoff - Loss%)
EV_CALC_METHOD = 'A' 

# 買入優先序切換開關
# 'EV' = 優先買入期望值最高的標的 (單筆質量極大化)
# 'PROJ_COST' = 優先買入能消耗最多資金的標的 (資金效率極大化)
# 'HIST_WIN_X_TRADES' = 優先買入歷史勝率 × 交易次數最高的標的 (穩定度 × 樣本數)
# 'ASSET_GROWTH' = 優先買入歷史資產成長最高的標的 (歷史複利成長極大化)
BUY_SORT_METHOD = 'PROJ_COST'  

# 系統評分 (Score) 算法切換
# 'RoMD' = 傳統報酬回撤比風格的基底分數
# 'LOG_R2' = 結合對數 R 平方與月度勝率的不對稱模型
SCORE_CALC_METHOD = 'RoMD'  
SCORE_MDD_POWER = 1.2 # 1.0 = 保持原本 RoMD 口徑；>1 加重 MDD 懲罰；0~1 降低 MDD 懲罰
SCORE_MDD_DENOMINATOR_EPSILON = 0.0001
SCORE_WIN_RATE_AMP_ENABLED = True # = win_rate / SCORE_WIN_RATE_TARGET
SCORE_WIN_RATE_TARGET = 50.0 # 完整交易勝率達此目標時 score 不加不扣；低於目標打折，高於目標放大。

# 系統評分分子切換
# 'TOTAL_RETURN' = 分子使用總報酬率
# 'ANNUAL_RETURN' = 分子使用年化報酬率
SCORE_NUMERATOR_METHOD = 'TOTAL_RETURN'  



# 停利比例固定開關
OPTIMIZER_FIXED_TP_PERCENT = 0.0 # None = 由 optimizer 搜尋 tp_percent; 0.0 = 固定關閉停利; 其他數值 = 固定停利比例

# Trade mode 實戰參數輸出與 promote 設定。selector 名稱沿用 rolling/OOS policy：
# base / local / retention / base_retention_gt_0_0 / base_retention_gt_0_2 / base_retention_gt_0_4 / base_retention_gt_0_6 / base_retention_gt_0_8 / base_retention_gt_min
TRADE_MODE_CANDIDATE_SELECTOR = 'base_retention_gt_0_0'
TRADE_MODE_RUN_BEST_SELECTOR = 'base_retention_gt_0_0'
TRADE_MODE_AUTO_PROMOTE_RUN_BEST = True
TRADE_PROMOTE_MIN_SCORE_DELTA = 0.10
TRADE_PROMOTE_ON_POLICY_MISMATCH = 'candidate_only'

# optimizer 指標輸出開關。False 會停用該指標的表格、replay 與 paramset 輸出。
OPTIMIZER_POLICY_INDICATOR_ENABLED = {
    "base": True,
    "base_retention_gt_0_0": True,
    "base_retention_gt_min": True,
    "base_finalists_agree": True,
    "local": True,
    "local_finalists_agree": True,
    "retention_finalists_agree": True,
    "retention": True,
}

# ============================== 區間/次數 ====================================

# optimizer 提供三種資料區間語意：
# trade = 最新實際交易參數訓練；最近 OUTER_ROLLING_TRAIN_WINDOW_MONTHS；無 OOS。
# oos = 單一 fold OOS validation。rolling OOS = 多 fold OOS validation。
DEFAULT_OPTIMIZER_MODEL_MODE = 'trade'
OOS_EVALUATION_START_YEAR = 2021
OUTER_ROLLING_TRAIN_WINDOW_MONTHS = 120
OUTER_ROLLING_OOS_HORIZON_MONTHS = 12

# Study 儲存策略：正式流程不使用長期硬碟 DB / resume；必要時才用 per-run temp DB。
OPTIMIZER_PERSIST_STUDY_DB = False
OPTIMIZER_STUDY_STORAGE_MODE = 'memory'
OPTIMIZER_ALLOW_PER_RUN_TEMP_DB = True

# local_min review 計算開關。
OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED = True
OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE = 0.02  # local_min_score finalist review 預設取訓練次數的比例
OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN = 5  # local_min_score finalist review 的最小候選數

# Rolling OOS optimizer search 預設 trial 數。
OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT = 1000

# random seed ensemble：每次 retrain 隨機抽 N 個 seeds，正式輸出用同一個 JSON 保存 N 組參數
OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED = True
OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE = 8
OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE = "auto" # "auto" = 過半數；整數 = 至少幾個 seed 同意。最大值永遠是 N。

#  finalists agree：先以每個 seed 的全部 finalists 加總選出單一 seed。
OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE = "auto" # "auto" = 該 seed finalist 數量的一半向上取整；整數 = 至少幾個 finalist 同意。
OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE = "auto" # "auto" = 該 seed finalist 數量的一半向上取整；整數 = 至少幾個 finalist 同意。
OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE = "auto" # "auto" = 該 seed finalist 數量的一半向上取整；整數 = 至少幾個 finalist 同意。


# ============================== Gates ====================================

# base (r>門檻) 版本的 retention 門檻
OPTIMIZER_BASE_RETENTION_GT_MIN = 0.5 # 選取邏輯：local_retention > 此值後，再依 base_rank 取第一名。

# dominant-year dependency anti-overfitting 開關
OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED = False
DOMINANT_YEAR_HIGH_POSITIVE_PNL_SHARE = 0.70 # 最大獲利年度佔比 + 該年度來源狹窄判斷。
DOMINANT_YEAR_NARROW_POSITIVE_TRADE_COUNT = 3 # 最大獲利年度若只有 3 筆以內賺錢交易，視為該年度交易來源狹窄。
DOMINANT_YEAR_NARROW_POSITIVE_SYMBOL_COUNT = 3 # 最大獲利年度若只有 3 檔以內賺錢股票，視為該年度標的來源狹窄。
DOMINANT_YEAR_TOP_TRADE_OUTLIER_PNL_SHARE = 0.50 # 最大獲利年度若單筆交易貢獻該年正獲利 50% 以上，視為單筆 outlier。

# inner validation anti-overfitting 開關
OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED = False
OPTIMIZER_INNER_VALIDATE_MIN_SCORE = 0.0 # inner validation score 下限。第一層 gate 仍要求 validate score > 0。
OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE = 0.50 # inner validation 相對排名上限。0.50 代表只接受 finalists validation 排名前半段。
OPTIMIZER_INNER_VALIDATE_HOLDOUT_YEARS = 1 # 第一版固定切最後 1 年做 validation。


# =========================== 共用硬門檻 (投組期未績效門檻) ====================

# 交易頻率
MIN_ANNUAL_TRADES = 5.0  # 最小年化交易次數門檻
MIN_BUY_FILL_RATE = 70.0  # 最小保留後買進成交率門檻
MIN_TRADE_WIN_RATE = 30.0  # 最小完整交易勝率門檻
# 績效風險
MIN_FULL_YEAR_RETURN_PCT = -35.0  # 完整年度最差報酬率下限
MAX_PORTFOLIO_MDD_PCT = 45.0  # 投組最大回撤上限
# 穩定度
MIN_MONTHLY_WIN_RATE = 35.0  # 最小月勝率門檻
MIN_EQUITY_CURVE_R_SQUARED = 0.40  # 權益曲線最小 R 平方門檻


#========================= Functions ===========================================

def is_optimizer_local_min_review_enabled() -> bool:
    return bool(OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED)


def resolve_optimizer_policy_indicator_enabled_map() -> dict[str, bool]:
    return {
        str(name): bool(enabled)
        for name, enabled in dict(OPTIMIZER_POLICY_INDICATOR_ENABLED or {}).items()
    }


def is_optimizer_policy_indicator_enabled(policy_name: str) -> bool:
    # 未列入 map 的新指標預設開啟，避免外部擴充 policy 被意外關閉。
    return bool(resolve_optimizer_policy_indicator_enabled_map().get(str(policy_name), True))


def resolve_optimizer_enabled_policy_indicators(policy_names=None) -> tuple[str, ...]:
    names = tuple(str(name) for name in list(policy_names or []) if str(name))
    return tuple(name for name in names if is_optimizer_policy_indicator_enabled(name))


def _resolve_optimizer_finalists_agree_min_agree(finalist_count, raw_value) -> int:
    n = max(1, int(finalist_count or 1))
    text = str(raw_value).strip().lower() if raw_value is not None else "auto"
    if text in {"", "none", "null", "auto", "half", "half_up", "ceil_half"}:
        requested = int(math.ceil(n / 2.0))
    else:
        try:
            requested = int(raw_value)
        except (TypeError, ValueError):
            requested = int(math.ceil(n / 2.0))
    return min(n, max(1, int(requested)))


def resolve_optimizer_base_finalists_agree_min_agree(finalist_count, min_agree=None) -> int:
    raw_value = OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE if min_agree is None else min_agree
    return _resolve_optimizer_finalists_agree_min_agree(finalist_count, raw_value)


def resolve_optimizer_local_finalists_agree_min_agree(finalist_count, min_agree=None) -> int:
    raw_value = OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE if min_agree is None else min_agree
    return _resolve_optimizer_finalists_agree_min_agree(finalist_count, raw_value)


def resolve_optimizer_retention_finalists_agree_min_agree(finalist_count, min_agree=None) -> int:
    raw_value = OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE if min_agree is None else min_agree
    return _resolve_optimizer_finalists_agree_min_agree(finalist_count, raw_value)


def resolve_optimizer_local_min_score_finalist_top_k(n_trials):
    requested_trials = max(0, int(n_trials))
    proportional_top_k = int(math.ceil(requested_trials * OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE))
    return max(int(OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN), proportional_top_k)

def _derive_fixed_window_selection_start_year(oos_start_year: int, train_window_months: int) -> int:
    oos_start_month_index = int(oos_start_year) * 12
    selection_start_month_index = oos_start_month_index - max(1, int(train_window_months))
    return int(selection_start_month_index // 12)

PREDEPLOY_SELECTION_START_YEAR = _derive_fixed_window_selection_start_year(
    OOS_EVALUATION_START_YEAR,
    OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
)

OPTIMIZER_TRAIN_START_YEAR = PREDEPLOY_SELECTION_START_YEAR
OPTIMIZER_MIN_TRAIN_YEARS = OOS_EVALUATION_START_YEAR - PREDEPLOY_SELECTION_START_YEAR
TRAINING_SPLIT_POLICY = {
    "selection_start_year": PREDEPLOY_SELECTION_START_YEAR,
    "train_start_year": OPTIMIZER_TRAIN_START_YEAR,
    "min_train_years": OPTIMIZER_MIN_TRAIN_YEARS,
    "search_train_end_year": OOS_EVALUATION_START_YEAR - 1,
    "oos_start_year": OOS_EVALUATION_START_YEAR,
    "objective_mode": 'split_train_romd',
}

SELECTION_POLICY_PARAM_SPECS = {
    "min_history_trades": {"type": int, "default": 0, "min_value": 0},  # 歷史績效最少交易次數門檻
    "min_history_ev": {"type": float, "default": -1.0},  # 歷史績效最小期望值門檻
    "min_history_win_rate": {"type": float, "default": 0.30, "min_value": 0.0, "max_value": 1.0},  # 歷史績效最小勝率門檻
}

def resolve_score_mdd_power(raw_value=None) -> float:
    value = SCORE_MDD_POWER if raw_value is None else raw_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MDD_POWER 必須是有限非負數，目前值: {value!r}") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"SCORE_MDD_POWER 必須是有限非負數，目前值: {value!r}")
    return resolved


def resolve_score_mdd_denominator_epsilon(raw_value=None) -> float:
    value = SCORE_MDD_DENOMINATOR_EPSILON if raw_value is None else raw_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MDD_DENOMINATOR_EPSILON 必須是有限正數，目前值: {value!r}") from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"SCORE_MDD_DENOMINATOR_EPSILON 必須是有限正數，目前值: {value!r}")
    return resolved


def is_score_win_rate_amp_enabled() -> bool:
    return bool(SCORE_WIN_RATE_AMP_ENABLED)


def resolve_score_win_rate_target(raw_value=None) -> float:
    value = SCORE_WIN_RATE_TARGET if raw_value is None else raw_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_WIN_RATE_TARGET 必須是有限正數，目前值: {value!r}") from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"SCORE_WIN_RATE_TARGET 必須是有限正數，目前值: {value!r}")
    return resolved


def build_training_threshold_snapshot():
    return {
        "MIN_FULL_YEAR_RETURN_PCT": MIN_FULL_YEAR_RETURN_PCT,
        "MIN_ANNUAL_TRADES": MIN_ANNUAL_TRADES,
        "MIN_BUY_FILL_RATE": MIN_BUY_FILL_RATE,
        "MIN_TRADE_WIN_RATE": MIN_TRADE_WIN_RATE,
        "MAX_PORTFOLIO_MDD_PCT": MAX_PORTFOLIO_MDD_PCT,
        "MIN_MONTHLY_WIN_RATE": MIN_MONTHLY_WIN_RATE,
        "MIN_EQUITY_CURVE_R_SQUARED": MIN_EQUITY_CURVE_R_SQUARED,
    }

def build_training_score_policy_snapshot():
    return {
        "EV_CALC_METHOD": EV_CALC_METHOD,
        "BUY_SORT_METHOD": BUY_SORT_METHOD,
        "SCORE_CALC_METHOD": SCORE_CALC_METHOD,
        "SCORE_NUMERATOR_METHOD": SCORE_NUMERATOR_METHOD,
        "SCORE_MDD_POWER": resolve_score_mdd_power(),
        "SCORE_MDD_DENOMINATOR_EPSILON": resolve_score_mdd_denominator_epsilon(),
        "SCORE_WIN_RATE_AMP_ENABLED": is_score_win_rate_amp_enabled(),
        "SCORE_WIN_RATE_TARGET": resolve_score_win_rate_target(),
        "OPTIMIZER_FIXED_TP_PERCENT": OPTIMIZER_FIXED_TP_PERCENT,
        "OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED": is_optimizer_local_min_review_enabled(),
        "OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE": OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE,
        "OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN": OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN,
        "OPTIMIZER_BASE_RETENTION_GT_MIN": OPTIMIZER_BASE_RETENTION_GT_MIN,
        "OPTIMIZER_POLICY_INDICATOR_ENABLED": resolve_optimizer_policy_indicator_enabled_map(),
        "OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE": OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE,
        "OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE": OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE,
        "OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE": OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE,
        "OPTIMIZER_RANDOM_SEED_ENSEMBLE": build_seed_ensemble_policy_snapshot(
            enabled=OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
            seed_count=OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
            min_agree=OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
        ),
        "OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED": OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED,
        "OPTIMIZER_INNER_VALIDATE_MIN_SCORE": OPTIMIZER_INNER_VALIDATE_MIN_SCORE,
        "OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE": OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE,
        "OPTIMIZER_INNER_VALIDATE_HOLDOUT_YEARS": OPTIMIZER_INNER_VALIDATE_HOLDOUT_YEARS,
        "TRADE_MODE_CANDIDATE_SELECTOR": TRADE_MODE_CANDIDATE_SELECTOR,
        "TRADE_MODE_RUN_BEST_SELECTOR": TRADE_MODE_RUN_BEST_SELECTOR,
        "TRADE_MODE_AUTO_PROMOTE_RUN_BEST": TRADE_MODE_AUTO_PROMOTE_RUN_BEST,
        "TRADE_PROMOTE_MIN_SCORE_DELTA": TRADE_PROMOTE_MIN_SCORE_DELTA,
        "TRADE_PROMOTE_ON_POLICY_MISMATCH": TRADE_PROMOTE_ON_POLICY_MISMATCH,
    }

def build_selection_policy_snapshot():
    return {field_name: spec["default"] for field_name, spec in SELECTION_POLICY_PARAM_SPECS.items()}

def build_optimizer_train_test_policy_snapshot():
    payload = dict(TRAINING_SPLIT_POLICY)
    payload["OUTER_ROLLING_TRAIN_WINDOW_MONTHS"] = int(OUTER_ROLLING_TRAIN_WINDOW_MONTHS)
    payload["OUTER_ROLLING_OOS_HORIZON_MONTHS"] = int(OUTER_ROLLING_OOS_HORIZON_MONTHS)
    payload["OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT"] = int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT)
    payload["TRADE_MODE_CANDIDATE_SELECTOR"] = str(TRADE_MODE_CANDIDATE_SELECTOR)
    payload["TRADE_MODE_RUN_BEST_SELECTOR"] = str(TRADE_MODE_RUN_BEST_SELECTOR)
    payload["TRADE_MODE_AUTO_PROMOTE_RUN_BEST"] = bool(TRADE_MODE_AUTO_PROMOTE_RUN_BEST)
    payload["TRADE_PROMOTE_MIN_SCORE_DELTA"] = float(TRADE_PROMOTE_MIN_SCORE_DELTA)
    payload["OPTIMIZER_PERSIST_STUDY_DB"] = bool(OPTIMIZER_PERSIST_STUDY_DB)
    payload["OPTIMIZER_STUDY_STORAGE_MODE"] = str(OPTIMIZER_STUDY_STORAGE_MODE)
    payload["OPTIMIZER_ALLOW_PER_RUN_TEMP_DB"] = bool(OPTIMIZER_ALLOW_PER_RUN_TEMP_DB)
    payload["OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED"] = is_optimizer_local_min_review_enabled()
    payload["OPTIMIZER_POLICY_INDICATOR_ENABLED"] = resolve_optimizer_policy_indicator_enabled_map()
    payload["OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE"] = OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE
    payload["OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE"] = OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE
    payload["OPTIMIZER_RANDOM_SEED_ENSEMBLE"] = build_seed_ensemble_policy_snapshot(
        enabled=OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
        seed_count=OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
        min_agree=OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
    )
    return payload

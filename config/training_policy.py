import math

from core.seed_ensemble_policy import build_seed_ensemble_policy_snapshot

# 期望值 (EV) 算法切換
# 'A' = 嚴格 R_Multiple 期望值 (Mean R)
# 'B' = 傳統實際盈虧期望值 (Win% * Payoff - Loss%)
EV_CALC_METHOD = 'A' 

# 買入優先序切換開關
# 'EV' = 優先買入期望值最高的標的 (單筆質量極大化)
# 'HIST_WIN_X_TRADES' = 優先買入歷史勝率 × 交易次數最高的標的 (穩定度 × 樣本數)
# 'ASSET_GROWTH' = 優先買入歷史資產成長最高的標的 (歷史複利成長極大化)
# 'PROJ_COST' = 優先買入能消耗最多資金的標的 (資金效率極大化)
# 'BUY_LIMIT_OVERAGE_THEN_PROJ_COST' = 優先買入前收未超出買入限價或超出幅度最小者，再按預估投入資金由大到小排序
# 'ENTRY_TYPE_THEN_PROJ_COST' = 優先買入新突破/Re-entry，延續候選靠後；同類再按買入前收未超出買入限價或超出幅度最小排序
BUY_SORT_METHOD = 'BUY_LIMIT_OVERAGE_THEN_PROJ_COST'  

# 系統評分 (Score) 算法切換
# 'RoMD' = 分子 / MDD 分母的報酬回撤比風格模型
# 'LOG_R2' = 在 RoMD 基底上結合對數 R 平方與月度勝率的不對稱模型s
SCORE_CALC_METHOD = 'RoMD'  
SCORE_MDD_POWER = 1.0 # 1.0 = 保持原本 RoMD 口徑；>1 加重 MDD 懲罰；0~1 降低 MDD 懲罰
SCORE_MDD_DENOMINATOR_EPSILON = 0.0001

SCORE_WIN_RATE_AMP_ENABLED = False  # True = 以完整交易勝率對 score 做目標式倍率校正。
SCORE_WIN_RATE_TARGET = 70.0  # 完整交易勝率達此目標時倍率為 1；低於目標會加速打折，高於目標會放大。
SCORE_MONTHLY_WIN_RATE_AMP_ENABLED = True  # True = 以月度獲利勝率對 score 做目標式倍率校正。
SCORE_MONTHLY_WIN_RATE_TARGET = 90.0  # 月度獲利勝率達此目標時倍率為 1；低於目標會加速打折，高於目標會放大。

SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED = False  # True = 以完整年度最差報酬對 score 做目標式倍率校正。
SCORE_MIN_FULL_YEAR_RETURN_TARGET = 10.0  # 完整年度最差報酬達此目標時倍率為 1；高於目標會放大。
SCORE_MIN_QUARTER_RETURN_AMP_ENABLED = True  # True = 以完整季度最差報酬對 score 做目標式倍率校正。
SCORE_MIN_QUARTER_RETURN_TARGET = 10.0  # 完整季度最差報酬達此目標時倍率為 1；高於目標會放大。
SCORE_MIN_MONTH_RETURN_AMP_ENABLED = False  # True = 以完整月度最差報酬對 score 做目標式倍率校正。
SCORE_MIN_MONTH_RETURN_TARGET = 10.0  # 完整月度最差報酬達此目標時倍率為 1；高於目標會放大。

SCORE_MEDIAN_R_AMP_ENABLED = False  # True = 以單股回測 R 中位數對 score 做目標式倍率校正。
SCORE_MEDIAN_R_FLOOR = -0.2  # R 中位數低於此值時倍率歸零；-1.0 代表完整 1R 虧損。
SCORE_MEDIAN_R_TARGET = 0.2  # R 中位數達此目標時倍率為 1；高於目標會放大。

# 系統評分分子切換
# 'TOTAL_RETURN' = 分子使用總報酬率
# 'ANNUAL_RETURN' = 分子使用年化報酬率
# 'TOTAL_R' = 分子使用單股回測總 R
# 'TOTAL_R_X_PORTFOLIO_RETURN' = 分子使用單股回測總 R × max(0, 投組總資產報酬率% / 100)
# 'TOTAL_R_X_ANNUAL_RETURN' = 分子使用單股回測總 R × max(0, 投組年化報酬率% / 100)
SCORE_NUMERATOR_METHOD = 'TOTAL_RETURN'  



# 停利比例固定開關
OPTIMIZER_FIXED_TP_PERCENT = 0.0 # None = 由 optimizer 搜尋 tp_percent; 0.0 = 固定關閉停利; 其他數值 = 固定停利比例

# Trade mode 實戰參數輸出與 promote 設定。selector 名稱沿用 rolling/OOS policy：
# base_finalist_best / local_finalist_best / retention_finalist_best / base_finalists_agree / local_finalists_agree / retention_finalists_agree / base / local / retention
TRADE_MODE_CANDIDATE_SELECTOR = 'base_finalists_agree'
TRADE_MODE_RUN_BEST_SELECTOR = 'base_finalists_agree'
TRADE_MODE_AUTO_PROMOTE_RUN_BEST = True
TRADE_PROMOTE_MIN_SCORE_DELTA = 0.10
TRADE_PROMOTE_ON_POLICY_MISMATCH = 'candidate_only'

# optimizer 指標輸出開關。False 會停用該指標的表格、replay 與 paramset 輸出。
OPTIMIZER_POLICY_INDICATOR_ENABLED = {
    "base_finalist_best": True,
    "local_finalist_best": True,
    "retention_finalist_best": True,
    "base_finalists_agree": True,
    "local_finalists_agree": True,
    "retention_finalists_agree": True,
    "base": True,
    "local": True,
    "retention": True,
}

# ============================== 區間/次數 ====================================

# optimizer 提供四種資料區間語意：
# study = 單 seed 研究模式；Study-Full / Study-OOS 由互動選單決定。
# full = seed ensemble 全期間訓練；使用 FULL_START_YEAR~FULL_END_YEAR；無 OOS。
# oos = seed ensemble 單一 fold OOS validation。rolling OOS = 多 fold OOS validation。
# trade = 最新實際交易參數訓練；最近 OUTER_ROLLING_TRAIN_WINDOW_MONTHS；無 OOS。
DEFAULT_OPTIMIZER_MODEL_MODE = 'trade'
FULL_START_YEAR = 2021
FULL_END_YEAR = None  # None = 使用最新資料日
OOS_EVALUATION_START_YEAR = 2021
OOS_EVALUATION_END_YEAR = None  # None = 使用最新資料日
OUTER_ROLLING_TRAIN_WINDOW_MONTHS = 120
OUTER_ROLLING_OOS_HORIZON_MONTHS = 12

# Study 儲存策略：正式流程不使用長期硬碟 DB / resume；必要時才用 per-run temp DB。
OPTIMIZER_PERSIST_STUDY_DB = False
OPTIMIZER_STUDY_STORAGE_MODE = 'memory'
OPTIMIZER_ALLOW_PER_RUN_TEMP_DB = True

# local_min review 計算開關。
# OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED 控制一般模式預設值；Full Mode 另有獨立預設，避免全期間正式訓練預設耗費 local review。
OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED = False
OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED = False
_OPTIMIZER_RUNTIME_MODEL_MODE = None
OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE = 0.01  # local_min_score finalist review 預設取訓練次數的比例
OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN = 6  # local_min_score finalist review 的最小候選數

# 單一 fold optimizer／最終 refit 預設 trial 數；rolling adaptation 不使用此值。
OPTIMIZER_SINGLE_FOLD_TRIALS_DEFAULT = 1000

# Rolling OOS 與 Selection rolling adaptation 每個 fold 的預設 trial 數。
OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT = 300

# Canonical optimizer stochastic seed.  Model-training seeds remain owned by their model config.
OPTIMIZER_RANDOM_SEED_DEFAULT = 42

# random seed ensemble：每次 retrain 隨機抽 N 個 seeds，正式輸出用同一個 JSON 保存 N 組參數
OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED = False
OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE = 8
OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE = "auto" # "auto" = 過半數；整數 = 至少幾個 seed 同意。最大值永遠是 N。

#  finalists agree：先以每個 seed 的全部 finalists 加總選出單一 seed。
OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE = "auto" # "auto" = 該 seed finalist 數量的一半向上取整；整數 = 至少幾個 finalist 同意。
OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE = "auto" # "auto" = 該 seed finalist 數量的一半向上取整；整數 = 至少幾個 finalist 同意。
OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE = "auto" # "auto" = 該 seed finalist 數量的一半向上取整；整數 = 至少幾個 finalist 同意。


# ============================== Gates ====================================

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
MIN_FULL_YEAR_RETURN_PCT = -25.0  # 完整年度最差報酬率下限
MAX_PORTFOLIO_MDD_PCT = 45.0  # 投組最大回撤上限
# 穩定度
MIN_MONTHLY_WIN_RATE = 35.0  # 最小月勝率門檻
MIN_EQUITY_CURVE_R_SQUARED = 0.40  # 權益曲線最小 R 平方門檻


#========================= Functions ===========================================

def set_optimizer_runtime_model_mode(model_mode) -> None:
    global _OPTIMIZER_RUNTIME_MODEL_MODE
    normalized = str(model_mode or "").strip().lower()
    _OPTIMIZER_RUNTIME_MODEL_MODE = normalized or None


def resolve_optimizer_runtime_model_mode() -> str:
    return str(_OPTIMIZER_RUNTIME_MODEL_MODE or "").strip().lower()


def is_optimizer_local_min_review_enabled(model_mode=None) -> bool:
    normalized = str(model_mode if model_mode is not None else resolve_optimizer_runtime_model_mode()).strip().lower()
    if normalized == "full":
        return bool(OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED)
    return bool(OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED)


def resolve_optimizer_policy_indicator_enabled_map() -> dict[str, bool]:
    return {
        str(name): bool(enabled)
        for name, enabled in dict(OPTIMIZER_POLICY_INDICATOR_ENABLED or {}).items()
    }


def is_optimizer_policy_indicator_enabled(policy_name: str) -> bool:
    # 未列入 map 的新指標預設開啟，避免外部擴充 policy 被意外關閉。
    name = str(policy_name)
    if name in {
        "local_finalist_best",
        "retention_finalist_best",
        "local_finalists_agree",
        "retention_finalists_agree",
        "local",
        "retention",
    } and not bool(is_optimizer_local_min_review_enabled()):
        return False
    return bool(resolve_optimizer_policy_indicator_enabled_map().get(name, True))


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
    "oos_end_year": OOS_EVALUATION_END_YEAR,
    "full_start_year": FULL_START_YEAR,
    "full_end_year": FULL_END_YEAR,
    "objective_mode": 'split_train_romd',
}

SELECTION_POLICY_PARAM_SPECS = {
    "use_history_threshold": {"type": bool, "default": True},  # 是否啟用歷史績效門檻
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
        raise ValueError(f"SCORE_WIN_RATE_TARGET 必須是 0~100 的有限正數，目前值: {value!r}") from exc
    if not math.isfinite(resolved) or resolved <= 0.0 or resolved > 100.0:
        raise ValueError(f"SCORE_WIN_RATE_TARGET 必須是 0~100 的有限正數，目前值: {value!r}")
    return resolved


def is_score_monthly_win_rate_amp_enabled() -> bool:
    return bool(SCORE_MONTHLY_WIN_RATE_AMP_ENABLED)


def resolve_score_monthly_win_rate_target(raw_value=None) -> float:
    value = SCORE_MONTHLY_WIN_RATE_TARGET if raw_value is None else raw_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MONTHLY_WIN_RATE_TARGET 必須是 0~100 的有限正數，目前值: {value!r}") from exc
    if not math.isfinite(resolved) or resolved <= 0.0 or resolved > 100.0:
        raise ValueError(f"SCORE_MONTHLY_WIN_RATE_TARGET 必須是 0~100 的有限正數，目前值: {value!r}")
    return resolved


def is_score_min_full_year_return_amp_enabled() -> bool:
    return bool(SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED)


def is_score_min_month_return_amp_enabled() -> bool:
    return bool(SCORE_MIN_MONTH_RETURN_AMP_ENABLED)


def is_score_min_quarter_return_amp_enabled() -> bool:
    return bool(SCORE_MIN_QUARTER_RETURN_AMP_ENABLED)


def is_score_median_r_amp_enabled() -> bool:
    return bool(SCORE_MEDIAN_R_AMP_ENABLED)


def resolve_score_median_r_floor(raw_value=None) -> float:
    value = SCORE_MEDIAN_R_FLOOR if raw_value is None else raw_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MEDIAN_R_FLOOR 必須是有限數，目前值: {value!r}") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"SCORE_MEDIAN_R_FLOOR 必須是有限數，目前值: {value!r}")
    return resolved


def resolve_score_median_r_target(raw_value=None, floor_r=None) -> float:
    value = SCORE_MEDIAN_R_TARGET if raw_value is None else raw_value
    floor = SCORE_MEDIAN_R_FLOOR if floor_r is None else floor_r
    try:
        resolved = float(value)
        resolved_floor = float(floor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"SCORE_MEDIAN_R_TARGET 必須是大於 SCORE_MEDIAN_R_FLOOR 的有限數，"
            f"目前 target={value!r}, floor={floor!r}"
        ) from exc
    if not math.isfinite(resolved) or not math.isfinite(resolved_floor) or resolved <= resolved_floor:
        raise ValueError(
            f"SCORE_MEDIAN_R_TARGET 必須是大於 SCORE_MEDIAN_R_FLOOR 的有限數，"
            f"目前 target={value!r}, floor={floor!r}"
        )
    return resolved


def resolve_score_min_full_year_return_target(raw_value=None, floor_pct=None) -> float:
    value = SCORE_MIN_FULL_YEAR_RETURN_TARGET if raw_value is None else raw_value
    floor = MIN_FULL_YEAR_RETURN_PCT if floor_pct is None else floor_pct
    try:
        resolved = float(value)
        resolved_floor = float(floor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"SCORE_MIN_FULL_YEAR_RETURN_TARGET 必須是大於 MIN_FULL_YEAR_RETURN_PCT 的有限數，"
            f"目前 target={value!r}, floor={floor!r}"
        ) from exc
    if not math.isfinite(resolved) or not math.isfinite(resolved_floor) or resolved <= resolved_floor:
        raise ValueError(
            f"SCORE_MIN_FULL_YEAR_RETURN_TARGET 必須是大於 MIN_FULL_YEAR_RETURN_PCT 的有限數，"
            f"目前 target={value!r}, floor={floor!r}"
        )
    return resolved


def resolve_score_min_month_return_floor(target_pct=None) -> float:
    value = SCORE_MIN_MONTH_RETURN_TARGET if target_pct is None else target_pct
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MIN_MONTH_RETURN_TARGET 必須是有限數，目前值: {value!r}") from exc
    if not math.isfinite(target):
        raise ValueError(f"SCORE_MIN_MONTH_RETURN_TARGET 必須是有限數，目前值: {value!r}")
    if target > 0.0:
        return -abs(target)
    return float(MIN_FULL_YEAR_RETURN_PCT) / 12.0


def resolve_score_min_month_return_target(raw_value=None, floor_pct=None) -> float:
    value = SCORE_MIN_MONTH_RETURN_TARGET if raw_value is None else raw_value
    floor = resolve_score_min_month_return_floor(value) if floor_pct is None else floor_pct
    try:
        resolved = float(value)
        resolved_floor = float(floor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"SCORE_MIN_MONTH_RETURN_TARGET 必須是大於月度最差報酬 floor 的有限數，"
            f"目前 target={value!r}, floor={floor!r}"
        ) from exc
    if not math.isfinite(resolved) or not math.isfinite(resolved_floor) or resolved <= resolved_floor:
        raise ValueError(
            f"SCORE_MIN_MONTH_RETURN_TARGET 必須是大於月度最差報酬 floor 的有限數，"
            f"目前 target={value!r}, floor={floor!r}"
        )
    return resolved


def resolve_score_min_quarter_return_floor(target_pct=None) -> float:
    value = SCORE_MIN_QUARTER_RETURN_TARGET if target_pct is None else target_pct
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MIN_QUARTER_RETURN_TARGET 必須是有限數，目前值: {value!r}") from exc
    if not math.isfinite(target):
        raise ValueError(f"SCORE_MIN_QUARTER_RETURN_TARGET 必須是有限數，目前值: {value!r}")
    if target > 0.0:
        return -abs(target)
    return float(MIN_FULL_YEAR_RETURN_PCT) / 4.0


def resolve_score_min_quarter_return_target(raw_value=None, floor_pct=None) -> float:
    value = SCORE_MIN_QUARTER_RETURN_TARGET if raw_value is None else raw_value
    floor = resolve_score_min_quarter_return_floor(value) if floor_pct is None else floor_pct
    try:
        resolved = float(value)
        resolved_floor = float(floor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"SCORE_MIN_QUARTER_RETURN_TARGET 必須是大於季度最差報酬 floor 的有限數，"
            f"目前 target={value!r}, floor={floor!r}"
        ) from exc
    if not math.isfinite(resolved) or not math.isfinite(resolved_floor) or resolved <= resolved_floor:
        raise ValueError(
            f"SCORE_MIN_QUARTER_RETURN_TARGET 必須是大於季度最差報酬 floor 的有限數，"
            f"目前 target={value!r}, floor={floor!r}"
        )
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
        "SCORE_MONTHLY_WIN_RATE_AMP_ENABLED": is_score_monthly_win_rate_amp_enabled(),
        "SCORE_MONTHLY_WIN_RATE_TARGET": resolve_score_monthly_win_rate_target(),
        "SCORE_MIN_MONTH_RETURN_AMP_ENABLED": is_score_min_month_return_amp_enabled(),
        "SCORE_MIN_MONTH_RETURN_TARGET": resolve_score_min_month_return_target(),
        "SCORE_MIN_MONTH_RETURN_FLOOR": resolve_score_min_month_return_floor(),
        "SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED": is_score_min_full_year_return_amp_enabled(),
        "SCORE_MIN_FULL_YEAR_RETURN_TARGET": resolve_score_min_full_year_return_target(),
        "SCORE_MIN_QUARTER_RETURN_AMP_ENABLED": is_score_min_quarter_return_amp_enabled(),
        "SCORE_MIN_QUARTER_RETURN_TARGET": resolve_score_min_quarter_return_target(),
        "SCORE_MIN_QUARTER_RETURN_FLOOR": resolve_score_min_quarter_return_floor(),
        "SCORE_MEDIAN_R_AMP_ENABLED": is_score_median_r_amp_enabled(),
        "SCORE_MEDIAN_R_FLOOR": resolve_score_median_r_floor(),
        "SCORE_MEDIAN_R_TARGET": resolve_score_median_r_target(),
        "OPTIMIZER_FIXED_TP_PERCENT": OPTIMIZER_FIXED_TP_PERCENT,
        "OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED": is_optimizer_local_min_review_enabled(),
        "OPTIMIZER_LOCAL_MIN_REVIEW_DEFAULT_ENABLED": bool(OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED),
        "OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED": bool(OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED),
        "OPTIMIZER_RUNTIME_MODEL_MODE": resolve_optimizer_runtime_model_mode(),
        "OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE": OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE,
        "OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN": OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN,
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



def get_strategy_parameter_training_policy_snapshot(*, evaluation_mode: str) -> dict:
    """Return the canonical optimizer-owned strategy-parameter training policy.

    Consumers (Research/Strategy Compare/Audit) may record this snapshot but must not
    own independent seed/trial settings for current strategy-parameter artifacts.
    """
    mode = str(evaluation_mode or "").strip().lower()
    if mode == "roos":
        mode = "rolling"
    if mode == "split":
        mode = "oos"
    rolling_like = mode == "rolling"
    return {
        "owner": "optimizer",
        "evaluation_mode": mode,
        "optimizer_seed": int(OPTIMIZER_RANDOM_SEED_DEFAULT),
        "trials_per_fold": int(
            OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
            if rolling_like
            else OPTIMIZER_SINGLE_FOLD_TRIALS_DEFAULT
        ),
        "train_window_months": int(OUTER_ROLLING_TRAIN_WINDOW_MONTHS),
        "oos_horizon_months": int(OUTER_ROLLING_OOS_HORIZON_MONTHS),
        "random_seed_ensemble": build_seed_ensemble_policy_snapshot(
            enabled=OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
            seed_count=OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
            min_agree=OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
        ),
        "finalists_agree": {
            "base_min_agree": OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE,
            "local_min_agree": OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE,
            "retention_min_agree": OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE,
        },
    }

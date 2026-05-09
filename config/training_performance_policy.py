# Optimizer 訓練效能參數區
#
# 本檔只放會影響訓練時間、記憶體、process/cache 併發行為的設定。
# 策略口徑、分數口徑與交易規則仍維持在 training_policy.py / execution_policy.py。

# OPTIMIZER_ROLLING_FOLD_WORKERS:
# - "fold_count" = timing/rolling 平行模式預設使用 fold 總數。
# - 正整數 = 固定 rolling fold process 數。
# - 環境變數 OPTIMIZER_ROLLING_FOLD_WORKERS 仍可覆寫此預設。
OPTIMIZER_ROLLING_FOLD_WORKERS = "fold_count"

# OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS:
# - 0 = 關閉 parallel rolling worker 內的 prepared trial input cache。
# - 正整數 = 每個 fold worker 最多保留的 prepared trial input 筆數。
# - 環境變數 OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS 仍可覆寫此預設。
OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS = 0

# OPTIMIZER_FEATURE_BANK_MAX_ITEMS:
# - 0 = 關閉 optimizer worker feature bank。
# - 正整數 = 每個 prep worker 的 feature bank 上限。
# - timing 顯示 hit rate 長期偏低時，應降低此值以減少多 process 記憶體疊加。
# - 環境變數 OPTIMIZER_FEATURE_BANK_MAX_ITEMS 仍可覆寫此預設。
OPTIMIZER_FEATURE_BANK_MAX_ITEMS = 2048


def _coerce_int(value, *, default: int, min_value: int = 0, max_value: int | None = None) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    resolved = max(int(min_value), resolved)
    if max_value is not None:
        resolved = min(int(max_value), resolved)
    return resolved


def resolve_optimizer_rolling_fold_workers_default(fold_count):
    resolved_fold_count = _coerce_int(fold_count, default=1, min_value=1)
    raw_value = OPTIMIZER_ROLLING_FOLD_WORKERS
    if raw_value is None:
        return resolved_fold_count
    text = str(raw_value).strip().lower()
    if text in {"", "fold_count", "folds", "auto"}:
        return resolved_fold_count
    return _coerce_int(raw_value, default=resolved_fold_count, min_value=1)


def resolve_optimizer_rolling_parallel_prep_cache_max_items_default():
    return _coerce_int(
        OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS,
        default=0,
        min_value=0,
        max_value=256,
    )


def resolve_optimizer_feature_bank_max_items_default():
    return _coerce_int(
        OPTIMIZER_FEATURE_BANK_MAX_ITEMS,
        default=2048,
        min_value=0,
        max_value=65536,
    )


def build_training_performance_policy_snapshot(fold_count=None):
    return {
        "OPTIMIZER_ROLLING_FOLD_WORKERS": OPTIMIZER_ROLLING_FOLD_WORKERS,
        "OPTIMIZER_ROLLING_FOLD_WORKERS_RESOLVED": (
            None
            if fold_count is None
            else resolve_optimizer_rolling_fold_workers_default(fold_count)
        ),
        "OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS": resolve_optimizer_rolling_parallel_prep_cache_max_items_default(),
        "OPTIMIZER_FEATURE_BANK_MAX_ITEMS": resolve_optimizer_feature_bank_max_items_default(),
    }

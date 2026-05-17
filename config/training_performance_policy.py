import os
from core.seed_ensemble_policy import resolve_seed_ensemble_parallel_backend, resolve_seed_ensemble_parallel_workers

# Optimizer 訓練效能參數區
#
# 本檔只放會影響訓練時間、記憶體、process/cache 併發行為的設定。
# 策略口徑、分數口徑與交易規則仍維持在 training_policy.py / execution_policy.py。
# 一般 outer rolling 與 timing outer rolling 預設共用本檔，避免效能口徑分叉。

# OPTIMIZER_ROLLING_FOLD_WORKERS:
# - "fold_count" = timing/rolling 平行模式預設使用 fold 總數。
# - 正整數 = 固定 rolling fold process 數。
# - 環境變數 OPTIMIZER_ROLLING_FOLD_WORKERS 仍可覆寫此預設。
OPTIMIZER_ROLLING_FOLD_WORKERS = 1

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
OPTIMIZER_FEATURE_BANK_MAX_ITEMS = 1024


# 單一 optimizer search unit 內加速參數區
#
# 這一區控制「單一 optimizer search unit 內部」的平行化。
# 適用範圍：
# - 非 rolling 一般 optimizer / 一般 OOS：整次 optimizer search 視為一個 single unit。
# - rolling OOS：每一個 rolling fold 視為一個 single unit。
#
# 與 OPTIMIZER_ROLLING_FOLD_WORKERS 不同：
# - OPTIMIZER_ROLLING_FOLD_WORKERS 只控制 rolling OOS 時幾個 OOS fold 同時跑。
# - 本區控制單一 search unit 內 optimizer search / local_min review 的併發程度。
#
# 預設保守，不改既有 trial 序列與正式結果；需要壓縮單一 unit 時間時再手動調高。

# OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS:
# - 1 = 保持既有單一 search unit trial 串行搜尋，正式模式預設使用。
# - >1 = 同一 search unit 內同時評估多個 Optuna trial。
# - TPE sampler 預設仍會被保護為 1，除非打開 OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH。
OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS = 1

# OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH:
# - False = 正式 TPE 搜尋維持 n_jobs=1，避免平行 ask 導致 trial 序列漂移。
# - True = 允許 TPE 也使用 OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS。
# - 只改搜尋路徑，不改策略、交易與評分口徑。
OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH = False

# OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_WORKERS:
# - 控制 local_min review 內鄰點 ordered prefetch 的 thread worker 數。
# - 預設 1，保留最穩定的 local_min 評估節奏；需要加速可調高，但不設硬性上限。
OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_WORKERS = 4

# OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PROCESS_WORKERS:
# - 保留給 local_min process-level 併發；目前正式流程仍以 thread ordered prefetch 為主。
# - 預設 0，避免 process 巢狀併發造成記憶體放大。
OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PROCESS_WORKERS = 0

# OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS:
# - 控制 random seed ensemble 內同時運行幾個 seed member。
# - 1 = 維持循序，最省記憶體。
# - >1 = 同時跑多個 seeds；實際值會 clamp 到 1~OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE。
# - "auto" = 自動使用最大併發，也就是目前 random seed ensemble 的 N。
# - 這是效能設定，只改執行併發，不改 seed ensemble 的選參、交易或統計口徑。
OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS = 8

# OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND:
# - "process" = 每個 seed member 使用獨立 Python process，CPU 利用率較高，但記憶體用量較大。
# - "thread" = 同一 Python process 內用 threads，較省記憶體，但 CPU-bound 訓練較難吃滿核心。
# - 這是效能設定，只改 seed member 執行 backend，不改選參、交易或統計口徑。
OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND = "process"


def _coerce_int(value, *, default: int, min_value: int = 0, max_value: int | None = None) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    resolved = max(int(min_value), resolved)
    if max_value is not None:
        resolved = min(int(max_value), resolved)
    return resolved


def _coerce_bool(value, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


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
        default=1024,
        min_value=0,
        max_value=65536,
    )


# OPTIMIZER_LOCAL_MIN_DEPENDENCY_STATS_ENABLED:
# - True = timing mode 追加 local_min 鄰點的 signal / portfolio 依賴分層統計。
# - False = 關閉此觀測欄位。
# - 僅影響 timing 統計，不改 optimizer 分數、交易規則或 local_min 評估順序。
OPTIMIZER_LOCAL_MIN_DEPENDENCY_STATS_ENABLED = True

# OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER:
# - "last" = local_min 鄰點排序時，將 portfolio-only 鄰點排在未命中快取/提示的 signal 鄰點之後。
#            不改鄰點集合與 local_min 定義；只讓 early stop/prune 更早遇到較可能需要重算 signal 的鄰點。
# - "original" = 保留原本鄰點產生順序。
# - 環境變數 OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER 仍可覆寫此預設。
OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER = "last"

# OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER:
# - "hard_fail_first" = local_min 鄰點排序時，signal-only 鄰點依較可能觸發 early stop/prune 的欄位順序排序。
#                       不改鄰點集合與 local_min 定義；只改完整 local_min 評估前的安全排序。
# - "original" = 保留原本鄰點產生順序。
# - 也可用逗號分隔欄位名稱覆寫，例如：
#   OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER=high_len,atr_times_trail,atr_buy_tol
OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER = "hard_fail_first"

_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER_HARD_FAIL_FIRST = (
    "high_len",
    "atr_times_trail",
    "atr_buy_tol",
    "bb_len",
    "atr_len",
    "bb_mult",
    "kc_len",
    "kc_mult",
    "vol_long_len",
    "vol_short_len",
)


def resolve_optimizer_local_min_portfolio_dependency_order():
    raw_value = os.environ.get(
        "OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER",
        OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER,
    )
    text = str(raw_value or "").strip().lower()
    if text in {"last", "deprioritize", "deprioritized", "tail"}:
        return "last"
    if text in {"original", "off", "0", "false", "no"}:
        return "original"
    return "last"


def resolve_optimizer_local_min_signal_dependency_field_order() -> tuple[str, ...]:
    raw_value = os.environ.get(
        "OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER",
        OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER,
    )
    text = str(raw_value or "").strip()
    normalized = text.lower()
    if normalized in {"", "hard_fail_first", "default", "on", "1", "true", "yes"}:
        return tuple(_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER_HARD_FAIL_FIRST)
    if normalized in {"original", "off", "0", "false", "no"}:
        return tuple()
    fields = []
    seen = set()
    for part in text.split(","):
        field = str(part or "").strip()
        if not field or field in seen:
            continue
        fields.append(field)
        seen.add(field)
    return tuple(fields)


def resolve_optimizer_single_fold_search_parallel_trials_default():
    return _coerce_int(
        OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS,
        default=1,
        min_value=1,
        max_value=16,
    )


def is_optimizer_single_fold_tpe_parallel_search_allowed_default():
    return _coerce_bool(
        OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH,
        default=False,
    )


def resolve_optimizer_single_fold_local_min_parallel_workers_default():
    return _coerce_int(
        OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_WORKERS,
        default=1,
        min_value=1,
    )


def resolve_optimizer_single_fold_local_min_process_workers_default():
    return _coerce_int(
        OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PROCESS_WORKERS,
        default=0,
        min_value=0,
    )


def resolve_optimizer_random_seed_ensemble_parallel_workers_default(seed_count):
    return resolve_seed_ensemble_parallel_workers(
        seed_count,
        OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS,
    )


def resolve_optimizer_random_seed_ensemble_parallel_backend_default():
    return resolve_seed_ensemble_parallel_backend(OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND)


def is_optimizer_local_min_dependency_stats_enabled():
    raw_value = os.environ.get(
        "OPTIMIZER_LOCAL_MIN_DEPENDENCY_STATS_ENABLED",
        OPTIMIZER_LOCAL_MIN_DEPENDENCY_STATS_ENABLED,
    )
    return _coerce_bool(raw_value, default=True)

def build_training_performance_policy_snapshot(fold_count=None, seed_ensemble_size=None):
    return {
        "OPTIMIZER_ROLLING_FOLD_WORKERS": OPTIMIZER_ROLLING_FOLD_WORKERS,
        "OPTIMIZER_ROLLING_FOLD_WORKERS_RESOLVED": (
            None
            if fold_count is None
            else resolve_optimizer_rolling_fold_workers_default(fold_count)
        ),
        "OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS": resolve_optimizer_rolling_parallel_prep_cache_max_items_default(),
        "OPTIMIZER_FEATURE_BANK_MAX_ITEMS": resolve_optimizer_feature_bank_max_items_default(),
        "OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS": resolve_optimizer_single_fold_search_parallel_trials_default(),
        "OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH": is_optimizer_single_fold_tpe_parallel_search_allowed_default(),
        "OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_WORKERS": resolve_optimizer_single_fold_local_min_parallel_workers_default(),
        "OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PROCESS_WORKERS": resolve_optimizer_single_fold_local_min_process_workers_default(),
        "OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS": OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS,
        "OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS_RESOLVED": (
            None
            if seed_ensemble_size is None
            else resolve_optimizer_random_seed_ensemble_parallel_workers_default(seed_ensemble_size)
        ),
        "OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND": OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND,
        "OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND_RESOLVED": resolve_optimizer_random_seed_ensemble_parallel_backend_default(),
        "OPTIMIZER_LOCAL_MIN_DEPENDENCY_STATS_ENABLED": is_optimizer_local_min_dependency_stats_enabled(),
        "OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER": resolve_optimizer_local_min_portfolio_dependency_order(),
        "OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER": ",".join(resolve_optimizer_local_min_signal_dependency_field_order()),
    }

"""Canonical Strategy Optimizer training-policy resolver/runtime contract.

User/project knobs live in ``config.training_policy``.  This module owns
validation, derived policy snapshots, robustness seed resolution, and the
optimizer runtime model-mode state so config remains declarative.
"""

import math
import random

import config.training_policy as _settings
from core.seed_ensemble_policy import build_seed_ensemble_policy_snapshot

_OPTIMIZER_RUNTIME_MODEL_MODE = None

def resolve_robustness_benchmark_seeds(
    *,
    seed_count: int | None = None,
    generator_seed: int | None = None,
) -> tuple[int, ...]:
    count = int(_settings.ROBUSTNESS_BENCHMARK_SEED_COUNT if seed_count is None else seed_count)
    seed0 = int(
        _settings.ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED
        if generator_seed is None
        else generator_seed
    )
    if count < 2:
        raise ValueError("robustness benchmark seed_count必須>=2")
    if seed0 < 0:
        raise ValueError("robustness benchmark generator seed必須>=0")
    rng = random.Random(seed0)
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < count:
        value = int(rng.randrange(1, 2**31 - 1))
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values)

def get_robustness_benchmark_policy_snapshot() -> dict:
    return {
        "benchmark_id": str(_settings.ROBUSTNESS_BENCHMARK_ID),
        "seed_count": int(_settings.ROBUSTNESS_BENCHMARK_SEED_COUNT),
        "seed_generator_seed": int(_settings.ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED),
        "resolved_seeds": [int(value) for value in resolve_robustness_benchmark_seeds()],
        # Optimizer budget is not benchmark-owned: always inherit the canonical rolling policy.
        "strategy_trials_per_fold": int(_settings.OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT),
        "train_window_months": int(_settings.OUTER_ROLLING_TRAIN_WINDOW_MONTHS),
        "oos_horizon_months": int(_settings.OUTER_ROLLING_OOS_HORIZON_MONTHS),
        "seed_pairing": "same_seed_strategy_optimizer_and_all_dl_sources",
        "production_ensemble_is_separate": True,
    }

def set_optimizer_runtime_model_mode(model_mode) -> None:
    global _OPTIMIZER_RUNTIME_MODEL_MODE
    normalized = str(model_mode or "").strip().lower()
    _OPTIMIZER_RUNTIME_MODEL_MODE = normalized or None

def resolve_optimizer_runtime_model_mode() -> str:
    return str(_OPTIMIZER_RUNTIME_MODEL_MODE or "").strip().lower()

def is_optimizer_local_min_review_enabled(model_mode=None) -> bool:
    normalized = str(model_mode if model_mode is not None else resolve_optimizer_runtime_model_mode()).strip().lower()
    if normalized == "full":
        return bool(_settings.OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED)
    return bool(_settings.OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED)

def resolve_optimizer_policy_indicator_enabled_map() -> dict[str, bool]:
    return {
        str(name): bool(enabled)
        for name, enabled in dict(_settings.OPTIMIZER_POLICY_INDICATOR_ENABLED or {}).items()
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
    raw_value = _settings.OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE if min_agree is None else min_agree
    return _resolve_optimizer_finalists_agree_min_agree(finalist_count, raw_value)

def resolve_optimizer_local_finalists_agree_min_agree(finalist_count, min_agree=None) -> int:
    raw_value = _settings.OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE if min_agree is None else min_agree
    return _resolve_optimizer_finalists_agree_min_agree(finalist_count, raw_value)

def resolve_optimizer_retention_finalists_agree_min_agree(finalist_count, min_agree=None) -> int:
    raw_value = _settings.OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE if min_agree is None else min_agree
    return _resolve_optimizer_finalists_agree_min_agree(finalist_count, raw_value)

def resolve_optimizer_local_min_score_finalist_top_k(n_trials):
    requested_trials = max(0, int(n_trials))
    proportional_top_k = int(math.ceil(requested_trials * _settings.OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE))
    return max(int(_settings.OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN), proportional_top_k)

def resolve_score_mdd_power(raw_value=None) -> float:
    value = _settings.SCORE_MDD_POWER if raw_value is None else raw_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MDD_POWER 必須是有限非負數，目前值: {value!r}") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"SCORE_MDD_POWER 必須是有限非負數，目前值: {value!r}")
    return resolved

def resolve_score_mdd_denominator_epsilon(raw_value=None) -> float:
    value = _settings.SCORE_MDD_DENOMINATOR_EPSILON if raw_value is None else raw_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MDD_DENOMINATOR_EPSILON 必須是有限正數，目前值: {value!r}") from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"SCORE_MDD_DENOMINATOR_EPSILON 必須是有限正數，目前值: {value!r}")
    return resolved

def is_score_win_rate_amp_enabled() -> bool:
    return bool(_settings.SCORE_WIN_RATE_AMP_ENABLED)

def resolve_score_win_rate_target(raw_value=None) -> float:
    value = _settings.SCORE_WIN_RATE_TARGET if raw_value is None else raw_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_WIN_RATE_TARGET 必須是 0~100 的有限正數，目前值: {value!r}") from exc
    if not math.isfinite(resolved) or resolved <= 0.0 or resolved > 100.0:
        raise ValueError(f"SCORE_WIN_RATE_TARGET 必須是 0~100 的有限正數，目前值: {value!r}")
    return resolved

def is_score_monthly_win_rate_amp_enabled() -> bool:
    return bool(_settings.SCORE_MONTHLY_WIN_RATE_AMP_ENABLED)

def resolve_score_monthly_win_rate_target(raw_value=None) -> float:
    value = _settings.SCORE_MONTHLY_WIN_RATE_TARGET if raw_value is None else raw_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MONTHLY_WIN_RATE_TARGET 必須是 0~100 的有限正數，目前值: {value!r}") from exc
    if not math.isfinite(resolved) or resolved <= 0.0 or resolved > 100.0:
        raise ValueError(f"SCORE_MONTHLY_WIN_RATE_TARGET 必須是 0~100 的有限正數，目前值: {value!r}")
    return resolved

def is_score_min_full_year_return_amp_enabled() -> bool:
    return bool(_settings.SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED)

def is_score_min_month_return_amp_enabled() -> bool:
    return bool(_settings.SCORE_MIN_MONTH_RETURN_AMP_ENABLED)

def is_score_min_quarter_return_amp_enabled() -> bool:
    return bool(_settings.SCORE_MIN_QUARTER_RETURN_AMP_ENABLED)

def is_score_median_r_amp_enabled() -> bool:
    return bool(_settings.SCORE_MEDIAN_R_AMP_ENABLED)

def resolve_score_median_r_floor(raw_value=None) -> float:
    value = _settings.SCORE_MEDIAN_R_FLOOR if raw_value is None else raw_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MEDIAN_R_FLOOR 必須是有限數，目前值: {value!r}") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"SCORE_MEDIAN_R_FLOOR 必須是有限數，目前值: {value!r}")
    return resolved

def resolve_score_median_r_target(raw_value=None, floor_r=None) -> float:
    value = _settings.SCORE_MEDIAN_R_TARGET if raw_value is None else raw_value
    floor = _settings.SCORE_MEDIAN_R_FLOOR if floor_r is None else floor_r
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
    value = _settings.SCORE_MIN_FULL_YEAR_RETURN_TARGET if raw_value is None else raw_value
    floor = _settings.MIN_FULL_YEAR_RETURN_PCT if floor_pct is None else floor_pct
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
    value = _settings.SCORE_MIN_MONTH_RETURN_TARGET if target_pct is None else target_pct
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MIN_MONTH_RETURN_TARGET 必須是有限數，目前值: {value!r}") from exc
    if not math.isfinite(target):
        raise ValueError(f"SCORE_MIN_MONTH_RETURN_TARGET 必須是有限數，目前值: {value!r}")
    if target > 0.0:
        return -abs(target)
    return float(_settings.MIN_FULL_YEAR_RETURN_PCT) / 12.0

def resolve_score_min_month_return_target(raw_value=None, floor_pct=None) -> float:
    value = _settings.SCORE_MIN_MONTH_RETURN_TARGET if raw_value is None else raw_value
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
    value = _settings.SCORE_MIN_QUARTER_RETURN_TARGET if target_pct is None else target_pct
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCORE_MIN_QUARTER_RETURN_TARGET 必須是有限數，目前值: {value!r}") from exc
    if not math.isfinite(target):
        raise ValueError(f"SCORE_MIN_QUARTER_RETURN_TARGET 必須是有限數，目前值: {value!r}")
    if target > 0.0:
        return -abs(target)
    return float(_settings.MIN_FULL_YEAR_RETURN_PCT) / 4.0

def resolve_score_min_quarter_return_target(raw_value=None, floor_pct=None) -> float:
    value = _settings.SCORE_MIN_QUARTER_RETURN_TARGET if raw_value is None else raw_value
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
        "MIN_FULL_YEAR_RETURN_PCT": _settings.MIN_FULL_YEAR_RETURN_PCT,
        "MIN_ANNUAL_TRADES": _settings.MIN_ANNUAL_TRADES,
        "MIN_BUY_FILL_RATE": _settings.MIN_BUY_FILL_RATE,
        "MIN_TRADE_WIN_RATE": _settings.MIN_TRADE_WIN_RATE,
        "MAX_PORTFOLIO_MDD_PCT": _settings.MAX_PORTFOLIO_MDD_PCT,
        "MIN_MONTHLY_WIN_RATE": _settings.MIN_MONTHLY_WIN_RATE,
        "MIN_EQUITY_CURVE_R_SQUARED": _settings.MIN_EQUITY_CURVE_R_SQUARED,
    }

def build_training_score_policy_snapshot():
    return {
        "EV_CALC_METHOD": _settings.EV_CALC_METHOD,
        "BUY_SORT_METHOD": _settings.BUY_SORT_METHOD,
        "SCORE_CALC_METHOD": _settings.SCORE_CALC_METHOD,
        "SCORE_NUMERATOR_METHOD": _settings.SCORE_NUMERATOR_METHOD,
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
        "OPTIMIZER_FIXED_TP_PERCENT": _settings.OPTIMIZER_FIXED_TP_PERCENT,
        "OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED": is_optimizer_local_min_review_enabled(),
        "OPTIMIZER_LOCAL_MIN_REVIEW_DEFAULT_ENABLED": bool(_settings.OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED),
        "OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED": bool(_settings.OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED),
        "OPTIMIZER_RUNTIME_MODEL_MODE": resolve_optimizer_runtime_model_mode(),
        "OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE": _settings.OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_RATE,
        "OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN": _settings.OPTIMIZER_LOCAL_MIN_SCORE_FINALIST_TOP_K_MIN,
        "OPTIMIZER_POLICY_INDICATOR_ENABLED": resolve_optimizer_policy_indicator_enabled_map(),
        "OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE": _settings.OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE,
        "OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE": _settings.OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE,
        "OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE": _settings.OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE,
        "OPTIMIZER_RANDOM_SEED_ENSEMBLE": build_seed_ensemble_policy_snapshot(
            enabled=_settings.OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
            seed_count=_settings.OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
            min_agree=_settings.OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
        ),
        "OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED": _settings.OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED,
        "OPTIMIZER_INNER_VALIDATE_MIN_SCORE": _settings.OPTIMIZER_INNER_VALIDATE_MIN_SCORE,
        "OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE": _settings.OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE,
        "OPTIMIZER_INNER_VALIDATE_HOLDOUT_YEARS": _settings.OPTIMIZER_INNER_VALIDATE_HOLDOUT_YEARS,
        "TRADE_MODE_CANDIDATE_SELECTOR": _settings.TRADE_MODE_CANDIDATE_SELECTOR,
        "TRADE_MODE_RUN_BEST_SELECTOR": _settings.TRADE_MODE_RUN_BEST_SELECTOR,
        "TRADE_MODE_AUTO_PROMOTE_RUN_BEST": _settings.TRADE_MODE_AUTO_PROMOTE_RUN_BEST,
        "TRADE_PROMOTE_MIN_SCORE_DELTA": _settings.TRADE_PROMOTE_MIN_SCORE_DELTA,
        "TRADE_PROMOTE_ON_POLICY_MISMATCH": _settings.TRADE_PROMOTE_ON_POLICY_MISMATCH,
    }

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
    # Current Strategy Parameter SSOT is one cross-time schedule.  OOS freezes the
    # member effective at its configured score start, while Rolling consumes the same
    # schedule, so both current modes share
    # the rolling per-fold optimizer budget; the 1000-trial single-fold default
    # remains for independent Study/OOS optimizer workflows, not this artifact SSOT.
    rolling_like = mode in {"oos", "rolling"}
    return {
        "owner": "optimizer",
        "evaluation_mode": mode,
        "optimizer_seed": int(_settings.OPTIMIZER_RANDOM_SEED_DEFAULT),
        "trials_per_fold": int(
            _settings.OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
            if rolling_like
            else _settings.OPTIMIZER_SINGLE_FOLD_TRIALS_DEFAULT
        ),
        "train_window_months": int(_settings.OUTER_ROLLING_TRAIN_WINDOW_MONTHS),
        "oos_horizon_months": int(_settings.OUTER_ROLLING_OOS_HORIZON_MONTHS),
        "random_seed_ensemble": build_seed_ensemble_policy_snapshot(
            enabled=_settings.OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
            seed_count=_settings.OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
            min_agree=_settings.OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
        ),
        "finalists_agree": {
            "base_min_agree": _settings.OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE,
            "local_min_agree": _settings.OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE,
            "retention_min_agree": _settings.OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE,
        },
    }


"""Deterministic market-regime context derived from the existing OHLCV sequence."""

from __future__ import annotations

import math


EXPECTED_REGIME_SEQUENCE_FEATURE_COUNT = 10
STOCK_CLOSE_FEATURE_INDEX = 3
BENCHMARK_CLOSE_FEATURE_INDEX = 8
REGIME_CONTEXT_LOOKBACK_BARS = (20, 60)
REGIME_CONTEXT_ANNUALIZATION_BARS = 252
REGIME_CONTEXT_FEATURES = (
    "benchmark_log_return_20",
    "benchmark_log_return_60",
    "benchmark_annualized_volatility_20",
    "benchmark_annualized_volatility_60",
    "stock_minus_benchmark_log_return_20",
    "stock_minus_benchmark_log_return_60",
)


def build_regime_context_from_level_sequence(
    torch,
    sequence,
    *,
    lookback_bars: tuple[int, ...] = REGIME_CONTEXT_LOOKBACK_BARS,
    annualization_bars: int = REGIME_CONTEXT_ANNUALIZATION_BARS,
):
    """Return low-dimensional market regime context from [batch, feature, time].

    The source sequence is the existing normalized stock/0050 OHLCV tensor, so no
    Dataset rebuild or future data is required. For each lookback, the transform
    derives 0050 log return, annualized 0050 close-to-close volatility, and stock
    minus 0050 log return using data available through the event date only.
    """

    if int(sequence.ndim) != 3:
        raise ValueError("regime context 需要 [batch, feature, time] tensor")
    if int(sequence.shape[1]) != EXPECTED_REGIME_SEQUENCE_FEATURE_COUNT:
        raise ValueError("regime context 需要 canonical 10-column OHLCV feature contract")

    normalized_lookbacks = tuple(int(value) for value in lookback_bars)
    if not normalized_lookbacks or any(value < 1 for value in normalized_lookbacks):
        raise ValueError("regime context lookback bars 必須是非空正整數")
    if tuple(sorted(set(normalized_lookbacks))) != normalized_lookbacks:
        raise ValueError("regime context lookback bars 必須遞增且不可重複")
    if int(annualization_bars) < 1:
        raise ValueError("regime context annualization bars 必須 >=1")
    if int(sequence.shape[2]) <= max(normalized_lookbacks):
        raise ValueError(
            "regime context sequence 長度不足: "
            f"sequence={int(sequence.shape[2])}, max_lookback={max(normalized_lookbacks)}"
        )

    epsilon = 1e-6

    def _log_close(feature_index: int):
        values = torch.clamp(
            sequence[:, int(feature_index), :],
            min=-1.0 + epsilon,
        )
        return torch.log1p(values)

    stock_log_close = _log_close(STOCK_CLOSE_FEATURE_INDEX)
    benchmark_log_close = _log_close(BENCHMARK_CLOSE_FEATURE_INDEX)

    benchmark_returns = []
    benchmark_volatilities = []
    relative_returns = []
    annualization_scale = math.sqrt(float(annualization_bars))
    for lookback in normalized_lookbacks:
        stock_total_return = stock_log_close[:, -1] - stock_log_close[:, -(lookback + 1)]
        benchmark_total_return = (
            benchmark_log_close[:, -1] - benchmark_log_close[:, -(lookback + 1)]
        )
        benchmark_daily_returns = (
            benchmark_log_close[:, -lookback:]
            - benchmark_log_close[:, -(lookback + 1) : -1]
        )
        benchmark_volatility = torch.std(
            benchmark_daily_returns,
            dim=1,
            unbiased=False,
        ) * annualization_scale

        benchmark_returns.append(benchmark_total_return)
        benchmark_volatilities.append(benchmark_volatility)
        relative_returns.append(stock_total_return - benchmark_total_return)

    result = torch.stack(
        [*benchmark_returns, *benchmark_volatilities, *relative_returns],
        dim=1,
    )
    if int(result.shape[1]) != len(REGIME_CONTEXT_FEATURES):
        raise AssertionError("regime context feature 數量與正式欄位契約不一致")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("regime context 衍生值必須全部有限")
    return result


__all__ = [
    "BENCHMARK_CLOSE_FEATURE_INDEX",
    "EXPECTED_REGIME_SEQUENCE_FEATURE_COUNT",
    "REGIME_CONTEXT_ANNUALIZATION_BARS",
    "REGIME_CONTEXT_FEATURES",
    "REGIME_CONTEXT_LOOKBACK_BARS",
    "STOCK_CLOSE_FEATURE_INDEX",
    "build_regime_context_from_level_sequence",
]

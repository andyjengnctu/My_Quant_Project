import numpy as np
import pandas as pd

from core.price_utils import adjust_long_buy_limit_array
from core.feature_bank import coerce_feature_bank


def tv_rma(source, length):
    source = np.asarray(source, dtype=np.float64)
    rma = np.full(source.shape, np.nan, dtype=np.float64)
    valid_idx = np.where(~np.isnan(source))[0]
    if len(valid_idx) < length:
        return rma
    first_valid = valid_idx[length - 1]
    tail_start = valid_idx[0]
    if valid_idx[-1] - tail_start + 1 == len(valid_idx):
        seed = np.mean(source[tail_start:first_valid + 1])
        tail = np.empty(source.size - first_valid, dtype=np.float64)
        tail[0] = seed
        if tail.size > 1:
            tail[1:] = source[first_valid + 1:]
        rma[first_valid:] = pd.Series(tail).ewm(alpha=1.0 / length, adjust=False).mean().to_numpy(dtype=np.float64, copy=False)
        return rma
    rma[first_valid] = np.mean(source[valid_idx[0]:first_valid + 1])
    alpha = 1.0 / length
    for i in range(first_valid + 1, len(source)):
        if not np.isnan(source[i]):
            rma[i] = alpha * source[i] + (1 - alpha) * rma[i - 1]
        else:
            rma[i] = rma[i - 1]
    return rma


def tv_atr(high, low, close, length):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if high.size == 0 or low.size == 0 or close.size == 0:
        return np.full(close.shape, np.nan, dtype=np.float64)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0], tr3[0] = np.nan, np.nan
    tr = np.nanmax([tr1, tr2, tr3], axis=0)
    return tv_rma(tr, length)


def tv_ema(source, length):
    source = np.asarray(source, dtype=np.float64)
    ema = np.full(source.shape, np.nan, dtype=np.float64)
    valid_idx = np.where(~np.isnan(source))[0]
    if len(valid_idx) == 0:
        return ema
    first_valid = valid_idx[0]
    if valid_idx[-1] - first_valid + 1 == len(valid_idx):
        ema[first_valid:] = pd.Series(source[first_valid:]).ewm(span=length, adjust=False).mean().to_numpy(dtype=np.float64, copy=False)
        return ema
    ema[first_valid] = source[first_valid]
    alpha = 2.0 / (length + 1)
    for i in range(first_valid + 1, len(source)):
        if not np.isnan(source[i]):
            if np.isnan(ema[i - 1]):
                ema[i] = source[i]
            else:
                ema[i] = alpha * source[i] + (1 - alpha) * ema[i - 1]
        else:
            ema[i] = ema[i - 1]
    return ema


def tv_supertrend(high, low, close, atr, multiplier):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64)
    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    final_ub = np.full(close.shape, np.nan, dtype=np.float64)
    final_lb = np.full(close.shape, np.nan, dtype=np.float64)
    direction = np.full(close.shape, 1, dtype=np.int8)
    first_valid = np.where(~np.isnan(atr))[0]
    if len(first_valid) == 0:
        return direction
    first = first_valid[0]
    final_ub[first], final_lb[first] = basic_ub[first], basic_lb[first]
    for i in range(first + 1, len(close)):
        if close[i - 1] > final_lb[i - 1]:
            final_lb[i] = max(basic_lb[i], final_lb[i - 1])
        else:
            final_lb[i] = basic_lb[i]
        if close[i - 1] < final_ub[i - 1]:
            final_ub[i] = min(basic_ub[i], final_ub[i - 1])
        else:
            final_ub[i] = basic_ub[i]
        if direction[i - 1] == -1 and close[i] < final_lb[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and close[i] > final_ub[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return direction



def generate_signals(df, params, ticker=None, feature_bank=None):
    resolved_ticker = ticker or df.attrs.get('ticker')
    if len(df) == 0:
        empty_float = np.array([], dtype=np.float64)
        empty_bool = np.array([], dtype=bool)
        return empty_float, empty_bool, empty_bool, empty_float
    H = df['High'].to_numpy(dtype=np.float64, copy=False)
    L = df['Low'].to_numpy(dtype=np.float64, copy=False)
    C = df['Close'].to_numpy(dtype=np.float64, copy=False)
    O = df['Open'].to_numpy(dtype=np.float64, copy=False)
    V = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    is_tradable_bar = V > 0
    feature_cache = coerce_feature_bank(feature_bank)
    atr_len = int(params.atr_len)
    high_len = int(params.high_len)
    atr_times_trail = float(params.atr_times_trail)
    use_bb = bool(getattr(params, 'use_bb', True))
    use_vol = bool(getattr(params, 'use_vol', True))
    use_kc = bool(getattr(params, 'use_kc', True))

    def _feature(feature_name, feature_args, builder):
        if feature_cache is None or resolved_ticker is None:
            return builder()
        return feature_cache.get_or_compute((resolved_ticker, feature_name, tuple(feature_args)), builder)

    ATR_main = _feature('atr', (atr_len,), lambda: tv_atr(H, L, C, atr_len))

    HighN = _feature(
        'rolling_high_shift1',
        (high_len,),
        lambda: pd.Series(H).shift(1).rolling(high_len, min_periods=high_len).max().values,
    )
    SuperTrend_Dir = _feature(
        'supertrend_dir',
        (atr_len, atr_times_trail),
        lambda: tv_supertrend(H, L, C, ATR_main, atr_times_trail),
    )

    prev_supertrend = np.empty_like(SuperTrend_Dir)
    prev_supertrend[0] = SuperTrend_Dir[0]
    prev_supertrend[1:] = SuperTrend_Dir[:-1]
    isSupertrend_Bearish_Flip = (SuperTrend_Dir == 1) & (prev_supertrend == -1)
    isSupertrend_Bearish_Flip[0] = False

    prev_close = np.empty_like(C)
    prev_close[0] = C[0]
    prev_close[1:] = C[:-1]
    prev_highn = np.empty_like(HighN)
    prev_highn[0] = HighN[0]
    prev_highn[1:] = HighN[:-1]
    isPriceCrossover = (C > HighN) & (prev_close <= prev_highn)
    isPriceCrossover[0] = False

    if use_bb:
        bb_len = int(params.bb_len)
        close_series = pd.Series(C)
        BB_Mid = _feature(
            'close_rolling_mean',
            (bb_len,),
            lambda: close_series.rolling(bb_len).mean().values,
        )
        BB_Std = _feature(
            'close_rolling_std0',
            (bb_len,),
            lambda: close_series.rolling(bb_len).std(ddof=0).values,
        )
        BB_Upper = BB_Mid + params.bb_mult * BB_Std
        bbCondition = C > BB_Upper
    else:
        bbCondition = np.ones_like(C, dtype=bool)

    if use_vol:
        vol_short_len = int(params.vol_short_len)
        vol_long_len = int(params.vol_long_len)
        volume_series = pd.Series(V)
        VolS = _feature(
            'volume_rolling_mean',
            (vol_short_len,),
            lambda: volume_series.rolling(vol_short_len).mean().values,
        )
        VolL = _feature(
            'volume_rolling_mean',
            (vol_long_len,),
            lambda: volume_series.rolling(vol_long_len).mean().values,
        )
        volCondition = np.isnan(V) | (VolS > VolL)
    else:
        volCondition = np.ones_like(C, dtype=bool)

    if use_kc:
        kc_len = int(params.kc_len)
        ATR_kc = ATR_main if kc_len == atr_len else _feature('atr', (kc_len,), lambda: tv_atr(H, L, C, kc_len))
        KC_Mid = _feature('ema_close', (kc_len,), lambda: tv_ema(C, kc_len))
        KC_Lower = KC_Mid - ATR_kc * params.kc_mult
        prev_kc_lower = np.empty_like(KC_Lower)
        prev_kc_lower[0] = KC_Lower[0]
        prev_kc_lower[1:] = KC_Lower[:-1]
        isKcCrossunder = (C < KC_Lower) & (prev_close >= prev_kc_lower)
        isKcCrossunder[0] = False
        kcSellCondition = isKcCrossunder & (C < O)
    else:
        kcSellCondition = np.zeros_like(C, dtype=bool)

    buyCondition = is_tradable_bar & (C > O) & isPriceCrossover & bbCondition & volCondition
    sellCondition = is_tradable_bar & ((isSupertrend_Bearish_Flip & (C <= O)) | kcSellCondition)
    raw_buy_limits = C + ATR_main * params.atr_buy_tol
    buy_limits = np.full_like(C, np.nan)
    valid_buy_mask = buyCondition & ~np.isnan(raw_buy_limits)
    if np.any(valid_buy_mask):
        buy_limits[valid_buy_mask] = adjust_long_buy_limit_array(raw_buy_limits[valid_buy_mask], ticker=resolved_ticker)
    return ATR_main, buyCondition, sellCondition, buy_limits

import numpy as np
import pandas as pd

from core.price_utils import adjust_long_buy_limit_array


def tv_rma(source, length):
    source = np.asarray(source, dtype=np.float64)
    rma = np.full(source.shape, np.nan, dtype=np.float64)
    valid_idx = np.where(~np.isnan(source))[0]
    if len(valid_idx) < length:
        return rma
    first_valid = valid_idx[length - 1]
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




def build_optimizer_signal_feature_cache_for_df(df, *, atr_lengths, high_lengths, bb_lengths, vol_lengths):
    if len(df) == 0:
        return {
            'close': np.array([], dtype=np.float64),
            'open': np.array([], dtype=np.float64),
            'high': np.array([], dtype=np.float64),
            'low': np.array([], dtype=np.float64),
            'volume': np.array([], dtype=np.float64),
            'is_tradable_bar': np.array([], dtype=bool),
            'close_gt_open': np.array([], dtype=bool),
            'atr_by_len': {},
            'highn_by_len': {},
            'bb_mid_by_len': {},
            'bb_std_by_len': {},
            'vol_mean_by_len': {},
        }

    H = df['High'].to_numpy(dtype=np.float64, copy=False)
    L = df['Low'].to_numpy(dtype=np.float64, copy=False)
    C = df['Close'].to_numpy(dtype=np.float64, copy=False)
    O = df['Open'].to_numpy(dtype=np.float64, copy=False)
    V = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    close_series = pd.Series(C)
    high_series = pd.Series(H)
    volume_series = pd.Series(V)

    atr_by_len = {int(length): tv_atr(H, L, C, int(length)) for length in atr_lengths}
    highn_by_len = {
        int(length): high_series.shift(1).rolling(int(length), min_periods=int(length)).max().values
        for length in high_lengths
    }
    bb_mid_by_len = {
        int(length): close_series.rolling(int(length)).mean().values
        for length in bb_lengths
    }
    bb_std_by_len = {
        int(length): close_series.rolling(int(length)).std(ddof=0).values
        for length in bb_lengths
    }
    vol_mean_by_len = {
        int(length): volume_series.rolling(int(length)).mean().values
        for length in vol_lengths
    }
    ema_by_len = {
        int(length): tv_ema(C, int(length))
        for length in atr_lengths
    }

    return {
        'close': C,
        'open': O,
        'high': H,
        'low': L,
        'volume': V,
        'is_tradable_bar': V > 0,
        'close_gt_open': C > O,
        'atr_by_len': atr_by_len,
        'highn_by_len': highn_by_len,
        'bb_mid_by_len': bb_mid_by_len,
        'bb_std_by_len': bb_std_by_len,
        'vol_mean_by_len': vol_mean_by_len,
        'ema_by_len': ema_by_len,
    }


def generate_signals_from_feature_cache(df, params, feature_cache, ticker=None):
    resolved_ticker = ticker or df.attrs.get('ticker')
    if len(df) == 0:
        empty_float = np.array([], dtype=np.float64)
        empty_bool = np.array([], dtype=bool)
        return empty_float, empty_bool, empty_bool, empty_float

    H = feature_cache['high']
    L = feature_cache['low']
    C = feature_cache['close']
    O = feature_cache['open']
    V = feature_cache['volume']
    is_tradable_bar = feature_cache['is_tradable_bar']
    close_gt_open = feature_cache['close_gt_open']
    atr_len = int(params.atr_len)
    high_len = int(params.high_len)
    ATR_main = feature_cache['atr_by_len'].get(atr_len)
    if ATR_main is None:
        ATR_main = tv_atr(H, L, C, atr_len)
    HighN = feature_cache['highn_by_len'].get(high_len)
    if HighN is None:
        high_series = pd.Series(H)
        HighN = high_series.shift(1).rolling(high_len, min_periods=high_len).max().values
    SuperTrend_Dir = tv_supertrend(H, L, C, ATR_main, params.atr_times_trail)

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

    if getattr(params, 'use_bb', True):
        bb_len = int(params.bb_len)
        bb_mid = feature_cache['bb_mid_by_len'].get(bb_len)
        bb_std = feature_cache['bb_std_by_len'].get(bb_len)
        if bb_mid is None or bb_std is None:
            close_series = pd.Series(C)
            bb_mid = close_series.rolling(bb_len).mean().values
            bb_std = close_series.rolling(bb_len).std(ddof=0).values
        BB_Upper = bb_mid + params.bb_mult * bb_std
        bbCondition = C > BB_Upper
    else:
        bbCondition = np.ones_like(C, dtype=bool)

    if getattr(params, 'use_vol', True):
        vol_short_len = int(params.vol_short_len)
        vol_long_len = int(params.vol_long_len)
        VolS = feature_cache['vol_mean_by_len'].get(vol_short_len)
        VolL = feature_cache['vol_mean_by_len'].get(vol_long_len)
        if VolS is None or VolL is None:
            volume_series = pd.Series(V)
            if VolS is None:
                VolS = volume_series.rolling(vol_short_len).mean().values
            if VolL is None:
                VolL = volume_series.rolling(vol_long_len).mean().values
        volCondition = np.isnan(V) | (VolS > VolL)
    else:
        volCondition = np.ones_like(C, dtype=bool)

    if getattr(params, 'use_kc', True):
        kc_len = int(params.kc_len)
        ATR_kc = feature_cache['atr_by_len'].get(kc_len)
        if ATR_kc is None:
            ATR_kc = tv_atr(H, L, C, kc_len)
        KC_Mid = feature_cache.get('ema_by_len', {}).get(kc_len)
        if KC_Mid is None:
            KC_Mid = tv_ema(C, kc_len)
        KC_Lower = KC_Mid - ATR_kc * params.kc_mult
        prev_kc_lower = np.empty_like(KC_Lower)
        prev_kc_lower[0] = KC_Lower[0]
        prev_kc_lower[1:] = KC_Lower[:-1]
        isKcCrossunder = (C < KC_Lower) & (prev_close >= prev_kc_lower)
        isKcCrossunder[0] = False
        kcSellCondition = isKcCrossunder & (C < O)
    else:
        kcSellCondition = np.zeros_like(C, dtype=bool)

    buyCondition = is_tradable_bar & close_gt_open & isPriceCrossover & bbCondition & volCondition
    sellCondition = is_tradable_bar & ((isSupertrend_Bearish_Flip & (C <= O)) | kcSellCondition)
    raw_buy_limits = C + ATR_main * params.atr_buy_tol
    buy_limits = np.full_like(C, np.nan)
    valid_buy_mask = buyCondition & ~np.isnan(raw_buy_limits)
    if np.any(valid_buy_mask):
        buy_limits[valid_buy_mask] = adjust_long_buy_limit_array(raw_buy_limits[valid_buy_mask], ticker=resolved_ticker)
    return ATR_main, buyCondition, sellCondition, buy_limits

def generate_signals(df, params, ticker=None):
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
    ATR_main = tv_atr(H, L, C, params.atr_len)

    close_series = pd.Series(C)
    high_series = pd.Series(H)
    volume_series = pd.Series(V)
    HighN = high_series.shift(1).rolling(params.high_len, min_periods=params.high_len).max().values
    SuperTrend_Dir = tv_supertrend(H, L, C, ATR_main, params.atr_times_trail)

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

    if getattr(params, 'use_bb', True):
        BB_Mid = close_series.rolling(params.bb_len).mean().values
        BB_Upper = BB_Mid + params.bb_mult * close_series.rolling(params.bb_len).std(ddof=0).values
        bbCondition = C > BB_Upper
    else:
        bbCondition = np.ones_like(C, dtype=bool)

    if getattr(params, 'use_vol', True):
        VolS = volume_series.rolling(params.vol_short_len).mean().values
        VolL = volume_series.rolling(params.vol_long_len).mean().values
        volCondition = np.isnan(V) | (VolS > VolL)
    else:
        volCondition = np.ones_like(C, dtype=bool)

    if getattr(params, 'use_kc', True):
        ATR_kc = tv_atr(H, L, C, params.kc_len)
        KC_Mid = tv_ema(C, params.kc_len)
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

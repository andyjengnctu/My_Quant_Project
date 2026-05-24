"""突破假突破濾網共用工具。"""

import numpy as np


def build_breakout_false_filter_pass_condition(close, atr, atr_pct_min):
    """回傳假突破濾網通過條件。

    規則只使用個股訊號日 ATR/Close，不依賴 benchmark 特徵。
    當資料不足或 ATR% 無法計算時，採保守不拒絕，避免資料缺口誤砍真突破。
    """
    close_arr = np.asarray(close, dtype=np.float64)
    atr_arr = np.asarray(atr, dtype=np.float64)
    atr_pct = np.divide(
        atr_arr,
        close_arr,
        out=np.full_like(close_arr, np.nan, dtype=np.float64),
        where=np.isfinite(close_arr) & (close_arr > 0),
    )
    reject = np.isfinite(atr_pct) & (atr_pct > float(atr_pct_min))
    return ~reject

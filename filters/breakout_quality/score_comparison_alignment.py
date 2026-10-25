"""Research-only, source-neutral daily score alignment and paired comparison checks.

This capability never changes persisted PIT scores, formal Oracle truth, training, or
Trading. A caller must supply fully provenance-validated stock-day score tables.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from filters.breakout_quality.mfe_safety_geometry import ensure_daily_percentile

ALIGNMENT_METHOD = "daily_average_zero_based_v1"


def rank_daily_stock_days(
    table: pd.DataFrame, *, score_column: str, start: str, end: str,
    cohort: pd.MultiIndex | None = None,
) -> pd.Series:
    """Use the canonical daily percentile formula on explicit, unique stock-days.

    `cohort` is the caller's independently validated *common* eligible population,
    not a candidate-only or hindsight-success subset. Missing members BLOCK.
    No row outside the requested period is allowed to enter the ranking denominator.
    """
    frame = pd.DataFrame(table)
    if not isinstance(frame.index, pd.MultiIndex) or tuple(frame.index.names) != ("ticker", "date"):
        raise ValueError("score alignment需要(ticker,date)唯一Index")
    if frame.index.has_duplicates or score_column not in frame:
        raise ValueError("score alignment的stock-day重複或缺少score欄位")
    if not start or not end or start > end:
        raise ValueError("score alignment期間無效")
    days = frame.index.get_level_values("date").astype(str)
    restricted = frame.loc[(days >= start) & (days <= end), [score_column]]
    if cohort is not None:
        if not isinstance(cohort, pd.MultiIndex) or cohort.has_duplicates:
            raise ValueError("score alignment共同母體必須是唯一MultiIndex")
        cohort_days = cohort.get_level_values("date").astype(str)
        if bool(((cohort_days < start) | (cohort_days > end)).any()):
            raise ValueError("score alignment共同母體超出比較期間")
        missing = cohort.difference(restricted.index)
        if len(missing):
            raise ValueError(f"score alignment共同母體缺少{len(missing)}筆合法score")
        restricted = restricted.reindex(cohort)
    if restricted.empty:
        raise ValueError("score alignment期間內沒有共同成熟stock-day")
    values = pd.to_numeric(restricted[score_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("score alignment包含無效或非有限分數")
    aligned, _ = ensure_daily_percentile(
        pd.DataFrame({"date": restricted.index.get_level_values("date").astype(str),
                      "value": values}),
        method="average_zero_based",
    )
    return pd.Series(aligned["percentile"].to_numpy(dtype=float), index=restricted.index,
                     name="aligned_percentile")


def common_stock_day_cohort(
    oracle: pd.DataFrame, model: pd.DataFrame, *, start: str, end: str,
    declared_eligible: pd.MultiIndex | None = None,
) -> pd.MultiIndex:
    """Fail closed rather than silently discard valid candidates or unseen stocks."""
    for name, table in (("Oracle", oracle), ("DL", model)):
        if (not isinstance(table.index, pd.MultiIndex)
                or tuple(table.index.names) != ("ticker", "date")
                or table.index.has_duplicates):
            raise ValueError(f"{name} stock-day identity不存在或重複")
    if declared_eligible is None:
        raise ValueError("必須使用模型及Oracle以外的獨立合法母體證據，不能以交集自動選樣")
    cohort = declared_eligible
    if cohort.has_duplicates or tuple(cohort.names) != ("ticker", "date"):
        raise ValueError("獨立合法母體stock-day identity錯誤")
    if len(cohort.difference(oracle.index)) or len(cohort.difference(model.index)):
        raise ValueError("Oracle與DL對獨立合法母體覆蓋不足，禁止只取交集美化結果")
    dates = cohort.get_level_values("date").astype(str)
    if bool(((dates < start) | (dates > end)).any()):
        raise ValueError("獨立合法母體的日期不在共同評估期間")
    return cohort


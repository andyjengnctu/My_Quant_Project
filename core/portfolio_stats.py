import pandas as pd
import numpy as np


def calc_portfolio_score(sys_ret, sys_mdd, m_win_rate, r_sq, annual_return_pct=None):
    from core.config import SCORE_CALC_METHOD, SCORE_NUMERATOR_METHOD

    annual_return = sys_ret if annual_return_pct is None else annual_return_pct
    if SCORE_NUMERATOR_METHOD == 'ANNUAL_RETURN':
        numerator = annual_return
    elif SCORE_NUMERATOR_METHOD == 'TOTAL_RETURN':
        numerator = sys_ret
    else:
        raise ValueError(f"未知 SCORE_NUMERATOR_METHOD: {SCORE_NUMERATOR_METHOD}")

    base_score = numerator / (abs(sys_mdd) + 0.0001)
    if SCORE_CALC_METHOD == 'LOG_R2':
        return base_score * (m_win_rate / 100.0) * r_sq
    if SCORE_CALC_METHOD == 'RoMD':
        return base_score
    raise ValueError(f"未知 SCORE_CALC_METHOD: {SCORE_CALC_METHOD}")


def calc_curve_stats(eq_list):
    r_squared, monthly_win_rate = 0.0, 0.0
    if len(eq_list) > 2:
        eq_array = np.array(eq_list)
        x = np.arange(len(eq_array))
        if np.std(eq_array) > 0:
            valid_idx = eq_array > 0
            if np.any(valid_idx):
                log_eq = np.log(eq_array[valid_idx])
                x_valid = x[valid_idx]
                if np.std(log_eq) > 0 and len(x_valid) > 1:
                    r_matrix = np.corrcoef(x_valid, log_eq)
                    r_squared = r_matrix[0, 1] ** 2 if not np.isnan(r_matrix[0, 1]) else 0.0
    if len(eq_list) > 1:
        eq_series = pd.Series(eq_list)
        monthly_rets = eq_series.pct_change().dropna()
        if len(monthly_rets) > 0:
            monthly_win_rate = (len(monthly_rets[monthly_rets > 0]) / len(monthly_rets)) * 100
    return r_squared, monthly_win_rate


# # (AI註: 只用完整年度做 min{r_y} > 0 的檢查；不把起始殘年與當前未完整年度算進去)
def build_full_year_return_stats(sorted_dates, year_start_equity, year_end_equity, year_first_sim_date, year_last_sim_date):
    yearly_return_rows = []
    year_market_bounds = {}

    for dt in sorted_dates:
        year = dt.year
        if year not in year_market_bounds:
            year_market_bounds[year] = {"first": dt, "last": dt}
        else:
            year_market_bounds[year]["last"] = dt

    for year in sorted(year_start_equity.keys()):
        start_equity = float(year_start_equity.get(year, 0.0))
        end_equity = float(year_end_equity.get(year, 0.0))
        first_sim_date = year_first_sim_date.get(year)
        last_sim_date = year_last_sim_date.get(year)
        market_bounds = year_market_bounds.get(year)

        if start_equity <= 0 or end_equity <= 0 or market_bounds is None or first_sim_date is None or last_sim_date is None:
            continue

        year_return_pct = (end_equity / start_equity - 1.0) * 100.0
        is_full_year = (first_sim_date == market_bounds["first"]) and (last_sim_date == market_bounds["last"])

        yearly_return_rows.append({
            "year": int(year),
            "year_return_pct": float(year_return_pct),
            "is_full_year": bool(is_full_year),
            "start_date": first_sim_date.strftime("%Y-%m-%d"),
            "end_date": last_sim_date.strftime("%Y-%m-%d"),
        })

    full_year_rows = [row for row in yearly_return_rows if row["is_full_year"]]
    min_full_year_return_pct = min((row["year_return_pct"] for row in full_year_rows), default=0.0)

    return {
        "full_year_count": len(full_year_rows),
        "min_full_year_return_pct": float(min_full_year_return_pct),
        "yearly_return_rows": yearly_return_rows,
    }


def build_benchmark_full_year_return_stats(sorted_dates, benchmark_data, yearly_return_rows):
    if benchmark_data is None or not yearly_return_rows:
        return {
            "bm_full_year_count": 0,
            "bm_min_full_year_return_pct": 0.0,
            "bm_yearly_return_rows": []
        }

    from core.portfolio_fast_data import has_fast_date, get_fast_close

    year_market_bounds = {}
    for dt in sorted_dates:
        year = dt.year
        if year not in year_market_bounds:
            year_market_bounds[year] = {"first": dt, "last": dt}
        else:
            year_market_bounds[year]["last"] = dt

    bm_yearly_rows = []
    full_years = [row["year"] for row in yearly_return_rows if row["is_full_year"]]

    for year in full_years:
        bounds = year_market_bounds.get(year)
        if bounds is None:
            continue
        if not has_fast_date(benchmark_data, bounds["first"]) or not has_fast_date(benchmark_data, bounds["last"]):
            continue

        start_value = get_fast_close(benchmark_data, date=bounds["first"])
        end_value = get_fast_close(benchmark_data, date=bounds["last"])
        if start_value is None or end_value is None or start_value <= 0:
            continue

        bm_yearly_rows.append({
            "year": int(year),
            "year_return_pct": float((end_value / start_value - 1.0) * 100.0),
            "is_full_year": True,
            "start_date": bounds["first"].strftime("%Y-%m-%d"),
            "end_date": bounds["last"].strftime("%Y-%m-%d"),
        })

    bm_min_full_year_return_pct = min((row["year_return_pct"] for row in bm_yearly_rows), default=0.0)

    return {
        "bm_full_year_count": len(bm_yearly_rows),
        "bm_min_full_year_return_pct": float(bm_min_full_year_return_pct),
        "bm_yearly_return_rows": bm_yearly_rows
    }


# # (AI註: 模擬起點索引單一真理來源；避免 engine / validate 各自複製起點規則導致完整年度判斷漂移)
def find_sim_start_idx(sorted_dates, start_year):
    if not sorted_dates:
        return 0

    start_dt = pd.to_datetime(f"{start_year}-01-01")
    return next((i for i, d in enumerate(sorted_dates) if d >= start_dt), len(sorted_dates))


# # (AI註: 年化報酬率與年化交易次數共用同一個回測期間口徑，避免統計不一致)
def calc_sim_years(sorted_dates, start_idx):
    if not sorted_dates or start_idx >= len(sorted_dates):
        return 0.0
    first_dt = pd.Timestamp(sorted_dates[start_idx])
    last_dt = pd.Timestamp(sorted_dates[-1])
    span_days = (last_dt - first_dt).days + 1
    if span_days <= 0:
        return 0.0
    return span_days / 365.25


# # (AI註: 用 CAGR 口徑統一系統與大盤年化報酬率；若期末值非正，直接回傳 -100% 避免數學異常)
def calc_annual_return_pct(start_value, end_value, years):
    if start_value <= 0 or years <= 0:
        return 0.0
    if end_value <= 0:
        return -100.0
    return ((end_value / start_value) ** (1.0 / years) - 1.0) * 100.0

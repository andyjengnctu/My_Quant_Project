import pandas as pd

from core.v16_portfolio_engine import find_sim_start_idx

from .trade_rebuild import rebuild_completed_trades_from_portfolio_trade_log


def calc_validation_sim_years(sorted_dates, start_year):
    if not sorted_dates:
        return 0.0

    start_idx = find_sim_start_idx(sorted_dates, start_year)
    if start_idx >= len(sorted_dates):
        return 0.0

    first_dt = pd.Timestamp(sorted_dates[start_idx])
    last_dt = pd.Timestamp(sorted_dates[-1])
    span_days = (last_dt - first_dt).days + 1
    if span_days <= 0:
        return 0.0
    return span_days / 365.25


def calc_validation_annual_return_pct(start_value, end_value, years):
    if start_value <= 0 or years <= 0:
        return 0.0
    if end_value <= 0:
        return -100.0
    return ((end_value / start_value) ** (1.0 / years) - 1.0) * 100.0


def normalize_yearly_return_rows(rows):
    normalized = []
    for row in rows or []:
        normalized.append({
            "year": int(row.get("year", 0)),
            "year_return_pct": float(row.get("year_return_pct", 0.0)),
            "is_full_year": bool(row.get("is_full_year", False)),
            "start_date": str(row.get("start_date", "")),
            "end_date": str(row.get("end_date", "")),
        })
    return normalized


def extract_yearly_profile_fields(profile_stats):
    yearly_rows = normalize_yearly_return_rows(profile_stats.get("yearly_return_rows", []))
    bm_yearly_rows = normalize_yearly_return_rows(profile_stats.get("bm_yearly_return_rows", []))
    return {
        "full_year_count": int(profile_stats.get("full_year_count", 0)),
        "min_full_year_return_pct": float(profile_stats.get("min_full_year_return_pct", 0.0)),
        "yearly_return_rows": yearly_rows,
        "bm_full_year_count": int(profile_stats.get("bm_full_year_count", 0)),
        "bm_min_full_year_return_pct": float(profile_stats.get("bm_min_full_year_return_pct", 0.0)),
        "bm_yearly_return_rows": bm_yearly_rows,
    }


def calc_expected_full_year_metrics(yearly_rows):
    full_year_rows = [row for row in (yearly_rows or []) if row["is_full_year"]]
    return {
        "full_year_count": len(full_year_rows),
        "min_full_year_return_pct": float(min((row["year_return_pct"] for row in full_year_rows), default=0.0)),
    }


def summarize_portfolio_trade_output(df_trades):
    portfolio_trade_types = df_trades["Type"].fillna("") if df_trades is not None and len(df_trades) > 0 and "Type" in df_trades.columns else pd.Series(dtype="object")
    portfolio_buy_rows = int(portfolio_trade_types.str.startswith("買進").sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_full_exit_rows = int(portfolio_trade_types.isin({"全倉結算(停損)", "全倉結算(指標)", "期末強制結算", "汰弱賣出(Open, T+1再評估買進)"}).sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_half_take_profit_rows = int((portfolio_trade_types == "半倉停利").sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_missed_buy_rows = int(portfolio_trade_types.str.startswith("錯失買進").sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_missed_sell_rows = int((portfolio_trade_types == "錯失賣出").sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_period_closeout_rows = int((portfolio_trade_types == "期末強制結算").sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_completed_trades = rebuild_completed_trades_from_portfolio_trade_log(df_trades)
    return {
        "portfolio_buy_rows": portfolio_buy_rows,
        "portfolio_full_exit_rows": portfolio_full_exit_rows,
        "portfolio_half_take_profit_rows": portfolio_half_take_profit_rows,
        "portfolio_missed_buy_rows": portfolio_missed_buy_rows,
        "portfolio_missed_sell_rows": portfolio_missed_sell_rows,
        "portfolio_period_closeout_rows": portfolio_period_closeout_rows,
        "portfolio_completed_trades": portfolio_completed_trades,
    }


def build_portfolio_stats_payload(*, module_path, df_trades, total_return, mdd, trade_count, win_rate, pf_ev, pf_payoff, final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct, profile_stats):
    payload = {
        "module_path": module_path,
        "total_return": total_return,
        "mdd": mdd,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "pf_ev": pf_ev,
        "pf_payoff": pf_payoff,
        "final_eq": final_eq,
        "avg_exp": avg_exp,
        "max_exp": max_exp,
        "bm_ret": bm_ret,
        "bm_mdd": bm_mdd,
        "total_missed": total_missed,
        "total_missed_sells": total_missed_sells,
        "r_sq": r_sq,
        "m_win_rate": m_win_rate,
        "bm_r_sq": bm_r_sq,
        "bm_m_win_rate": bm_m_win_rate,
        "normal_trade_count": normal_trade_count,
        "extended_trade_count": extended_trade_count,
        "annual_trades": annual_trades,
        "reserved_buy_fill_rate": reserved_buy_fill_rate,
        "annual_return_pct": annual_return_pct,
        "bm_annual_return_pct": bm_annual_return_pct,
        "df_trades": df_trades.copy() if isinstance(df_trades, pd.DataFrame) else pd.DataFrame(),
    }
    payload.update(summarize_portfolio_trade_output(payload["df_trades"]))
    payload.update(extract_yearly_profile_fields(profile_stats or {}))
    return payload

import pandas as pd


def rebuild_completed_trades_from_event_log(
    df_logs,
    *,
    tool_name,
    date_col,
    action_col,
    pnl_col,
    buy_prefix,
    half_action,
    full_exit_actions,
    ignored_actions,
    ignored_action_prefixes=(),
    missed_sell_actions,
):
    if df_logs is None or len(df_logs) == 0:
        return []

    required_cols = {date_col, action_col, pnl_col}
    missing_cols = [col for col in required_cols if col not in df_logs.columns]
    if missing_cols:
        raise KeyError(f"{tool_name} 缺少必要欄位: {missing_cols}")

    completed_trades = []
    active_trade = None

    for row in df_logs.itertuples(index=False):
        action = str(getattr(row, action_col))
        trade_date = pd.to_datetime(getattr(row, date_col)).strftime("%Y-%m-%d")
        realized_pnl = float(getattr(row, pnl_col))

        if action.startswith(buy_prefix):
            if active_trade is not None:
                raise ValueError(f"{tool_name} 出現連續買進，上一筆交易尚未完整結束。")
            active_trade = {
                "buy_date": trade_date,
                "exit_date": None,
                "total_pnl": 0.0,
                "half_exit_count": 0,
                "full_exit_action": None,
            }
            continue

        if action in ignored_actions:
            continue

        if any(action.startswith(prefix) for prefix in ignored_action_prefixes):
            continue

        if action == half_action:
            if active_trade is None:
                raise ValueError(f"{tool_name} 出現{half_action}，但前面沒有對應買進。")
            active_trade["total_pnl"] += realized_pnl
            active_trade["half_exit_count"] += 1
            continue

        if action in missed_sell_actions:
            if active_trade is None:
                raise ValueError(f"{tool_name} 出現 {action}，但前面沒有對應買進。")
            continue

        if action in full_exit_actions:
            if active_trade is None:
                raise ValueError(f"{tool_name} 出現 {action}，但前面沒有對應買進。")
            active_trade["total_pnl"] = round(active_trade["total_pnl"] + realized_pnl, 2)
            active_trade["exit_date"] = trade_date
            active_trade["full_exit_action"] = action
            completed_trades.append(active_trade)
            active_trade = None
            continue

        raise ValueError(f"{tool_name} 出現未納入驗證的動作: {action}")

    if active_trade is not None:
        raise ValueError(f"{tool_name} 最後仍有未完成交易，缺少完整賣出列。")

    return completed_trades


# # (AI註: 將 debug_trade_log 的逐列事件重建為 completed trades，
# # (AI註: 才能嚴格驗證半倉停利 + 尾倉結算後的總損益口徑是否與核心一致)
def rebuild_completed_trades_from_debug_log(debug_df):
    return rebuild_completed_trades_from_event_log(
        debug_df,
        tool_name="debug_trade_log",
        date_col="日期",
        action_col="動作",
        pnl_col="單筆實質損益",
        buy_prefix="買進",
        half_action="半倉停利",
        full_exit_actions={"停損殺出", "指標賣出", "期末強制結算"},
        ignored_actions=set(),
        ignored_action_prefixes=("錯失買進", "放棄進場"),
        missed_sell_actions={"錯失賣出"},
    )


# # (AI註: portfolio_sim 的 df_trades 也必須能重建成 completed trades，
# # (AI註: 否則即使 aggregate 指標一致，逐筆明細仍可能漂移而不自知)
def rebuild_completed_trades_from_portfolio_trade_log(df_trades):
    return rebuild_completed_trades_from_event_log(
        df_trades,
        tool_name="portfolio_sim df_trades",
        date_col="Date",
        action_col="Type",
        pnl_col="單筆損益",
        buy_prefix="買進",
        half_action="半倉停利",
        full_exit_actions={"全倉結算(停損)", "全倉結算(指標)", "期末強制結算", "汰弱賣出(Open, T+1再評估買進)"},
        ignored_actions=set(),
        ignored_action_prefixes=("錯失買進",),
        missed_sell_actions={"錯失賣出"},
    )

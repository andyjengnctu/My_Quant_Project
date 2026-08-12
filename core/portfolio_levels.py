import pandas as pd

from core.portfolio_exits import append_portfolio_position_active_level_row


def _append_portfolio_active_level_rows(active_level_rows, portfolio, today):
    if active_level_rows is None:
        return
    for ticker in sorted(portfolio.keys()):
        append_portfolio_position_active_level_row(
            active_level_rows,
            ticker=ticker,
            position=portfolio[ticker],
            today=today,
        )


def _is_portfolio_shadow_level_visible_day(today, signal_date):
    if signal_date is None:
        return True
    try:
        return pd.Timestamp(today) > pd.Timestamp(signal_date)
    except (TypeError, ValueError):
        return True


def _append_portfolio_extended_shadow_level_rows(active_level_rows, active_extended_signals, portfolio, today):
    if active_level_rows is None:
        return
    for ticker in sorted((active_extended_signals or {}).keys()):
        if ticker in portfolio:
            continue
        signal_state = active_extended_signals.get(ticker) or {}
        if not _is_portfolio_shadow_level_visible_day(today, signal_state.get('signal_date')):
            continue

        shadow_position = signal_state.get('shadow_position') or {}
        if int(shadow_position.get('qty', 0) or 0) <= 0:
            continue

        tp_half = shadow_position.get('tp_half')
        if bool(shadow_position.get('sold_half', False)) or shadow_position.get('pending_exit_action') == 'TP_HALF':
            tp_half = None

        orig_limit = signal_state.get('orig_limit')
        if orig_limit is None or orig_limit != orig_limit:
            orig_limit = shadow_position.get('limit_price')

        active_level_rows.append({
            'Date': today.strftime('%Y-%m-%d') if hasattr(today, 'strftime') else str(today),
            'Ticker': str(ticker),
            '停損價': shadow_position.get('sl'),
            '半倉停利價': tp_half,
            '買入限價': orig_limit,
            'Shadow買進價': shadow_position.get('entry_fill_price'),
            '成交價': None,
            'level_scope': 'extended_shadow',
            '進場類型': str(signal_state.get('source') or 'extended'),
            '買訊日': signal_state.get('signal_date'),
        })



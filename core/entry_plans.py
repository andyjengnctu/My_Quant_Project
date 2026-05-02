import numpy as np
import pandas as pd

from core.exact_accounting import (
    build_buy_ledger,
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    calc_initial_risk_total_milli,
    milli_to_money,
    milli_to_price,
    money_to_milli,
    price_to_milli,
    rate_to_ppm,
    sync_position_display_fields,
)
from core.order_lot_policy import (
    apply_board_lot_preferred_qty,
    calc_entry_notional_milli,
    entry_notional_meets_minimum,
    get_min_entry_notional_milli,
)
from core.price_utils import (
    adjust_long_buy_fill_price,
    calc_half_take_profit_sell_qty,
    calc_frozen_target_price,
    calc_initial_stop_from_reference,
    calc_initial_trailing_stop_from_reference,
    calc_position_size,
    is_locked_limit_up_bar,
)



def _find_affordable_qty(limit_price, max_qty, available_cash_milli, params):
    if max_qty <= 0 or available_cash_milli <= 0:
        return 0, 0

    price_milli = price_to_milli(limit_price)
    if price_milli <= 0:
        return 0, 0

    # # (AI註: 先用不含手續費的理論上限定位到可負擔股數附近，再由 exact accounting 微調；帳務仍由 build_buy_ledger() 單一真理來源決定)
    candidate_qty = min(int(max_qty), int(available_cash_milli) // int(price_milli))
    if candidate_qty <= 0:
        return 0, 0

    reserved_cost_milli = build_buy_ledger(price_milli, candidate_qty, params)["net_buy_total_milli"]
    while candidate_qty > 0 and reserved_cost_milli > int(available_cash_milli):
        candidate_qty -= 1
        if candidate_qty <= 0:
            return 0, 0
        reserved_cost_milli = build_buy_ledger(price_milli, candidate_qty, params)["net_buy_total_milli"]

    while candidate_qty < int(max_qty):
        next_qty = candidate_qty + 1
        next_cost_milli = build_buy_ledger(price_milli, next_qty, params)["net_buy_total_milli"]
        if next_cost_milli > int(available_cash_milli):
            break
        candidate_qty = next_qty
        reserved_cost_milli = next_cost_milli

    candidate_qty = apply_board_lot_preferred_qty(limit_price, candidate_qty, params)
    if candidate_qty <= 0:
        return 0, 0

    reserved_cost_milli = build_buy_ledger(price_milli, candidate_qty, params)["net_buy_total_milli"]
    if reserved_cost_milli > int(available_cash_milli):
        return 0, 0

    return candidate_qty, reserved_cost_milli


# # (AI註: 單一真理來源 - 候選單 sizing_capital 缺漏時統一回退到目前可用資金，避免 None / NaN 讓 exact accounting 路徑爆掉)
def _resolve_candidate_sizing_capital(candidate_plan, fallback_capital):
    if candidate_plan is None:
        return fallback_capital

    sizing_capital = candidate_plan.get("sizing_capital")
    if sizing_capital is None or pd.isna(sizing_capital):
        return fallback_capital
    return sizing_capital


# # (AI註: 單一真理來源 - 候選單在不同資金上限下的 qty / is_orderable 重算統一由此處理)
def resize_candidate_plan_to_capital(candidate_plan, sizing_capital, params):
    if candidate_plan is None:
        return None

    limit_price = candidate_plan.get("limit_price")
    init_sl = candidate_plan.get("init_sl")
    if pd.isna(limit_price) or pd.isna(init_sl):
        return None

    resolved_sizing_capital = _resolve_candidate_sizing_capital(candidate_plan, sizing_capital)
    resized_plan = dict(candidate_plan)
    qty = calc_position_size(
        limit_price,
        init_sl,
        resolved_sizing_capital,
        params.fixed_risk,
        params,
        ticker=candidate_plan.get("ticker"),
        security_profile=candidate_plan.get("security_profile"),
        trade_date=candidate_plan.get("trade_date"),
    )
    qty = apply_board_lot_preferred_qty(limit_price, qty, params)
    resized_plan["qty"] = qty
    resized_plan["is_orderable"] = qty > 0
    resized_plan["sizing_capital"] = float(resolved_sizing_capital)
    return resized_plan


# # (AI註: 單一真理來源 - 盤前依當下可用現金重算有效掛單規格與保留資金)
def build_cash_capped_entry_plan(candidate_plan, available_cash, params):
    sizing_capital = _resolve_candidate_sizing_capital(candidate_plan, available_cash)
    resized_plan = resize_candidate_plan_to_capital(candidate_plan, sizing_capital, params)
    if resized_plan is None or resized_plan["qty"] <= 0:
        return None

    available_cash_milli = money_to_milli(available_cash)
    affordable_qty, reserved_cost_milli = _find_affordable_qty(
        resized_plan["limit_price"],
        resized_plan["qty"],
        available_cash_milli,
        params,
    )
    if affordable_qty <= 0:
        return None

    resized_plan["qty"] = affordable_qty
    resized_plan["is_orderable"] = True
    resized_plan["reserved_cost_milli"] = reserved_cost_milli
    resized_plan["reserved_cost"] = milli_to_money(reserved_cost_milli)
    return resized_plan


# # (AI註: 單一真理來源 - 正常候選的盤前 limit / worst-case sizing stop / trail / qty 規格統一由此產生；候選資格與可掛單分離)
def build_normal_candidate_plan(limit_price, atr, sizing_capital, params, ticker=None, security_profile=None, trade_date=None):
    if pd.isna(limit_price) or pd.isna(atr):
        return None

    init_sl = calc_initial_stop_from_reference(limit_price, atr, params, ticker=ticker, security_profile=security_profile)
    init_trail = calc_initial_trailing_stop_from_reference(limit_price, atr, params, ticker=ticker, security_profile=security_profile)
    target_price = calc_frozen_target_price(limit_price, init_sl, ticker=ticker, security_profile=security_profile)
    base_plan = {
        "limit_price": limit_price,
        "init_sl": init_sl,
        "init_trail": init_trail,
        "target_price": target_price,
        "entry_atr": atr,
        "ticker": ticker,
        "security_profile": security_profile,
        "trade_date": trade_date,
    }
    return resize_candidate_plan_to_capital(base_plan, sizing_capital, params)


# # (AI註: 單一真理來源 - 正常單實際掛單規格；僅接受可掛單候選)
def build_normal_entry_plan(limit_price, atr, sizing_capital, params, ticker=None, security_profile=None, trade_date=None):
    candidate_plan = build_normal_candidate_plan(limit_price, atr, sizing_capital, params, ticker=ticker, security_profile=security_profile, trade_date=trade_date)
    if candidate_plan is None or candidate_plan["qty"] <= 0:
        return None
    return candidate_plan


# # (AI註: miss buy 正式定義單一真理來源 - 必須先有有效限價買單)
def should_count_miss_buy(order_qty, is_worse_than_initial_stop=False):
    if order_qty is None or order_qty <= 0:
        return False
    if is_worse_than_initial_stop:
        return False
    return True


def should_count_normal_miss_buy(order_qty, is_worse_than_initial_stop=False):
    return should_count_miss_buy(order_qty, is_worse_than_initial_stop=is_worse_than_initial_stop)


def _resolve_entry_fill_levels(*, buy_price, entry_atr, init_sl, init_trail, target_price, limit_price, params, ticker=None, security_profile=None):
    if entry_atr is not None and not pd.isna(entry_atr):
        actual_init_sl = calc_initial_stop_from_reference(buy_price, entry_atr, params, ticker=ticker, security_profile=security_profile)
        actual_init_trail = calc_initial_trailing_stop_from_reference(buy_price, entry_atr, params, ticker=ticker, security_profile=security_profile)
        actual_target_price = calc_frozen_target_price(buy_price, actual_init_sl, ticker=ticker, security_profile=security_profile)
        return actual_init_sl, actual_init_trail, actual_target_price

    resolved_init_sl = init_sl
    resolved_init_trail = init_trail
    resolved_target_price = target_price
    if resolved_target_price is None or pd.isna(resolved_target_price):
        target_basis = buy_price if limit_price is None else limit_price
        resolved_target_price = calc_frozen_target_price(target_basis, resolved_init_sl, ticker=ticker, security_profile=security_profile)
    return resolved_init_sl, resolved_init_trail, resolved_target_price


def _apply_entry_day_position_state(position, *, t_high):
    position["entry_day_stop_triggered"] = False
    position["entry_day_tp_triggered"] = False
    position["pending_exit_action"] = None
    position["pending_exit_trigger_price"] = np.nan

    highest_high_milli = position.get("highest_high_since_entry_milli", position["entry_fill_price_milli"])
    if not pd.isna(t_high):
        highest_high_milli = max(highest_high_milli, price_to_milli(t_high))
    position["highest_high_since_entry_milli"] = highest_high_milli
    position["highest_high_since_entry"] = milli_to_price(highest_high_milli)
    return position



def _apply_entry_day_pending_exit(position, *, t_high, t_low, params):
    if position is None or position.get("qty", 0) <= 0:
        return position

    stop_hit = (not pd.isna(t_low)) and price_to_milli(t_low) <= int(position["sl_milli"])
    half_sell_qty = calc_half_take_profit_sell_qty(position["qty"], params.tp_percent)
    tp_hit = (not pd.isna(t_high)) and price_to_milli(t_high) >= int(position["tp_half_milli"])

    if stop_hit and tp_hit:
        tp_hit = False

    position["entry_day_stop_triggered"] = bool(stop_hit)
    position["entry_day_tp_triggered"] = bool(tp_hit)

    if stop_hit:
        position["pending_exit_action"] = "STOP"
        position["pending_exit_trigger_price"] = milli_to_price(position["sl_milli"])
    elif tp_hit and half_sell_qty > 0 and not position.get("sold_half", False):
        position["pending_exit_action"] = "TP_HALF"
        position["pending_exit_trigger_price"] = milli_to_price(position["tp_half_milli"])
    return position


def _recompute_inherited_position_risk(position, *, params):
    if position is None or params is None:
        return position
    stop_price = position.get('sl')
    qty = int(position.get('initial_qty') or position.get('qty') or 0)
    if qty <= 0 or stop_price is None or pd.isna(stop_price):
        return sync_position_display_fields(position)

    stop_sell_ledger = build_sell_ledger_from_price(
        stop_price,
        qty,
        params,
        ticker=position.get('ticker'),
        security_profile=position.get('security_profile'),
        trade_date=position.get('entry_trade_date'),
    )
    position['initial_risk_total_milli'] = calc_initial_risk_total_milli(
        position.get('net_buy_total_milli', 0),
        stop_sell_ledger['net_sell_total_milli'],
        rate_to_ppm(params.fixed_risk),
    )
    return sync_position_display_fields(position)


def _apply_inherited_shadow_management(position, *, shadow_position, params=None):
    if position is None or shadow_position is None:
        return position

    inherited_fields = (
        'sl_milli',
        'initial_stop_milli',
        'trailing_stop_milli',
        'tp_half_milli',
        'sl',
        'initial_stop',
        'trailing_stop',
        'tp_half',
        'sold_half',
        'highest_high_since_entry_milli',
        'highest_high_since_entry',
        'pending_exit_action',
        'pending_exit_trigger_price',
    )
    for field in inherited_fields:
        if field in shadow_position:
            position[field] = shadow_position[field]

    position['inherited_shadow_management'] = True
    position['shadow_entry_fill_price_milli'] = shadow_position.get('entry_fill_price_milli', 0)
    position['shadow_entry_fill_price'] = shadow_position.get('entry_fill_price', float('nan'))
    position['shadow_sold_half'] = bool(shadow_position.get('sold_half', False))
    return _recompute_inherited_position_risk(position, params=params)


def _apply_inherited_shadow_entry_day_state(position, *, t_high):
    if position is None:
        return position

    position['entry_day_stop_triggered'] = False
    position['entry_day_tp_triggered'] = False

    highest_high_milli = int(position.get('highest_high_since_entry_milli', position['entry_fill_price_milli']))
    if not pd.isna(t_high):
        highest_high_milli = max(highest_high_milli, price_to_milli(t_high))
    position['highest_high_since_entry_milli'] = highest_high_milli
    position['highest_high_since_entry'] = milli_to_price(highest_high_milli)
    return position


# # (AI註: 單一真理來源 - 延續 shadow trade 啟動時，使用與正式進場同源的 fill / stop / target / entry-day pending exit 規則；不再依賴 low<=limit 才能存在)
def build_counterfactual_shadow_position_from_plan(entry_plan, *, t_open, t_high, t_low, params, ticker=None, security_profile=None, trade_date=None):
    if entry_plan is None:
        return None

    qty = int(entry_plan.get('qty', 0) or 0)
    resolved_ticker = ticker or entry_plan.get('ticker')
    resolved_security_profile = security_profile or entry_plan.get('security_profile')
    limit_price = entry_plan.get('limit_price', np.nan)
    if qty <= 0 or pd.isna(t_open) or pd.isna(limit_price):
        return None

    buy_price = adjust_long_buy_fill_price(min(t_open, limit_price), ticker=resolved_ticker, security_profile=resolved_security_profile)
    position = build_position_from_entry_fill(
        buy_price=buy_price,
        qty=qty,
        init_sl=entry_plan.get('init_sl'),
        init_trail=entry_plan.get('init_trail'),
        params=params,
        entry_type='extended_shadow',
        target_price=entry_plan.get('target_price'),
        limit_price=limit_price,
        entry_atr=entry_plan.get('entry_atr'),
        ticker=resolved_ticker,
        security_profile=resolved_security_profile,
        trade_date=trade_date,
    )
    position = _apply_entry_day_position_state(position, t_high=t_high)
    position = _apply_entry_day_pending_exit(position, t_high=t_high, t_low=t_low, params=params)
    return position


# # (AI註: 單一真理來源 - 成交後的部位欄位統一由此建立)
def build_position_from_entry_fill(
    buy_price,
    qty,
    init_sl=None,
    init_trail=None,
    params=None,
    entry_type="normal",
    *,
    target_price=None,
    limit_price=None,
    entry_atr=None,
    ticker=None,
    security_profile=None,
    trade_date=None,
):
    if params is None:
        raise ValueError("build_position_from_entry_fill 需要 params")
    if pd.isna(buy_price) or buy_price <= 0 or qty <= 0:
        raise ValueError(f"build_position_from_entry_fill 需要有效 buy_price/qty，收到 buy_price={buy_price!r}, qty={qty!r}")

    resolved_init_sl, resolved_init_trail, resolved_target_price = _resolve_entry_fill_levels(
        buy_price=buy_price,
        entry_atr=entry_atr,
        init_sl=init_sl,
        init_trail=init_trail,
        target_price=target_price,
        limit_price=limit_price,
        params=params,
        ticker=ticker,
        security_profile=security_profile,
    )
    if pd.isna(resolved_init_sl) or pd.isna(resolved_init_trail) or pd.isna(resolved_target_price):
        raise ValueError("build_position_from_entry_fill 無法建立有效的 stop / trail / target")

    buy_price_milli = price_to_milli(buy_price)
    initial_stop_milli = price_to_milli(resolved_init_sl)
    trailing_stop_milli = price_to_milli(resolved_init_trail)
    effective_stop_milli = max(initial_stop_milli, trailing_stop_milli)
    target_price_milli = price_to_milli(resolved_target_price)
    pure_limit_price_milli = None if limit_price is None or pd.isna(limit_price) else price_to_milli(limit_price)

    buy_ledger = build_buy_ledger_from_price(buy_price, qty, params)
    stop_sell_ledger = build_sell_ledger_from_price(
        resolved_init_sl,
        qty,
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
    )
    initial_risk_total_milli = calc_initial_risk_total_milli(
        buy_ledger["net_buy_total_milli"],
        stop_sell_ledger["net_sell_total_milli"],
        rate_to_ppm(params.fixed_risk),
    )

    position = {
        "qty": qty,
        "initial_qty": qty,
        "entry_fill_price_milli": buy_price_milli,
        "entry_fill_price": milli_to_price(buy_price_milli),
        "gross_buy_milli": buy_ledger["gross_buy_milli"],
        "buy_fee_milli": buy_ledger["buy_fee_milli"],
        "net_buy_total_milli": buy_ledger["net_buy_total_milli"],
        "remaining_cost_basis_milli": buy_ledger["net_buy_total_milli"],
        "sl_milli": effective_stop_milli,
        "initial_stop_milli": initial_stop_milli,
        "trailing_stop_milli": trailing_stop_milli,
        "tp_half_milli": target_price_milli,
        "sl": milli_to_price(effective_stop_milli),
        "initial_stop": milli_to_price(initial_stop_milli),
        "trailing_stop": milli_to_price(trailing_stop_milli),
        "tp_half": milli_to_price(target_price_milli),
        "sold_half": False,
        "pure_buy_price_milli": buy_price_milli,
        "pure_buy_price": milli_to_price(buy_price_milli),
        "realized_pnl_milli": 0,
        "display_realized_pnl_sum": 0.0,
        "initial_risk_total_milli": initial_risk_total_milli,
        "entry_type": entry_type,
        "ticker": ticker,
        "security_profile": security_profile,
        "entry_trade_date": trade_date,
        "limit_price": limit_price,
        "limit_price_milli": pure_limit_price_milli,
        "entry_day_stop_triggered": False,
        "entry_day_tp_triggered": False,
        "pending_exit_action": None,
        "pending_exit_trigger_price": np.nan,
        "highest_high_since_entry_milli": buy_price_milli,
        "highest_high_since_entry": milli_to_price(buy_price_milli),
    }
    return sync_position_display_fields(position)


# # (AI註: 單一真理來源 - 盤前有效買單的當日成交 / miss buy 邏輯統一由此判斷)
def execute_pre_market_entry_plan(entry_plan, t_open, t_high, t_low, t_close, t_volume, y_close, params, entry_type="normal", ticker=None, security_profile=None, trade_date=None):
    result = {
        "filled": False,
        "count_as_missed_buy": False,
        "is_worse_than_initial_stop": False,
        "is_locked_limit_up": False,
        "buy_price": np.nan,
        "entry_fill_price": np.nan,
        "entry_price": np.nan,
        "cost_basis_price": np.nan,
        "tp_half": np.nan,
        "position": None,
        "entry_type": entry_type,
        "entry_day_stop_triggered": False,
        "entry_day_tp_triggered": False,
        "entry_day_pending_action": None,
        "net_buy_total_milli": 0,
    }
    if entry_plan is None:
        return result

    qty = entry_plan.get("qty", 0)
    resolved_ticker = ticker or entry_plan.get("ticker")
    resolved_security_profile = security_profile or entry_plan.get("security_profile")
    result["count_as_missed_buy"] = should_count_miss_buy(qty)
    result["is_locked_limit_up"] = is_locked_limit_up_bar(t_open, t_high, t_low, t_close, y_close, ticker=resolved_ticker, security_profile=resolved_security_profile)

    if qty <= 0:
        result["count_as_missed_buy"] = False
        return result

    if pd.isna(t_volume) or t_volume <= 0 or pd.isna(t_open) or pd.isna(t_low) or result["is_locked_limit_up"]:
        return result

    limit_price = entry_plan["limit_price"]
    if price_to_milli(t_low) > price_to_milli(limit_price):
        return result

    buy_price = adjust_long_buy_fill_price(min(t_open, limit_price), ticker=resolved_ticker, security_profile=resolved_security_profile)
    result["buy_price"] = buy_price

    inherited_shadow_position = entry_plan.get("shadow_position_state")
    entry_atr_for_position = None if inherited_shadow_position is not None else entry_plan.get("entry_atr")
    position = build_position_from_entry_fill(
        buy_price=buy_price,
        qty=qty,
        init_sl=entry_plan.get("init_sl"),
        init_trail=entry_plan.get("init_trail"),
        params=params,
        entry_type=entry_type,
        target_price=entry_plan.get("target_price"),
        limit_price=limit_price,
        entry_atr=entry_atr_for_position,
        ticker=resolved_ticker,
        security_profile=resolved_security_profile,
        trade_date=trade_date,
    )
    if inherited_shadow_position is not None:
        position = _apply_inherited_shadow_management(position, shadow_position=inherited_shadow_position, params=params)
        position = _apply_inherited_shadow_entry_day_state(position, t_high=t_high)
    else:
        position = _apply_entry_day_position_state(position, t_high=t_high)
    position = _apply_entry_day_pending_exit(position, t_high=t_high, t_low=t_low, params=params)
    result["filled"] = True
    result["count_as_missed_buy"] = False
    result["position"] = position
    result["entry_fill_price"] = position["entry_fill_price"]
    result["entry_price"] = position["entry"]
    result["cost_basis_price"] = position["entry"]
    result["tp_half"] = position["tp_half"]
    result["entry_day_stop_triggered"] = bool(position["entry_day_stop_triggered"])
    result["entry_day_tp_triggered"] = bool(position["entry_day_tp_triggered"])
    result["entry_day_pending_action"] = position["pending_exit_action"]
    result["net_buy_total_milli"] = position["net_buy_total_milli"]
    result["entry_cost"] = milli_to_money(position["net_buy_total_milli"])
    return result

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from numbers import Integral
from typing import Any, Dict

MILLI_SCALE = 1000
PPM_SCALE = 1_000_000

_DECIMAL_THOUSAND = Decimal(MILLI_SCALE)
_DECIMAL_PPM = Decimal(PPM_SCALE)


def _to_decimal(value) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _round_decimal_to_int(value: Decimal, *, rounding=ROUND_HALF_UP) -> int:
    return int(value.quantize(Decimal("1"), rounding=rounding))


def price_to_milli(price) -> int:
    return _round_decimal_to_int(_to_decimal(price) * _DECIMAL_THOUSAND)


money_to_milli = price_to_milli


def milli_to_price(price_milli: int) -> float:
    return float(Decimal(int(price_milli)) / _DECIMAL_THOUSAND)


milli_to_money = milli_to_price


def is_money_milli_input(value) -> bool:
    return isinstance(value, Integral)


def coerce_money_like_to_milli(value) -> int:
    if is_money_milli_input(value):
        return int(value)
    return money_to_milli(value)


def restore_money_like_from_milli(value_milli: int, template):
    if is_money_milli_input(template):
        return int(value_milli)
    return milli_to_money(value_milli)


def rate_to_ppm(rate) -> int:
    return _round_decimal_to_int(_to_decimal(rate) * _DECIMAL_PPM)


MILLI_TICK_LADDER = (
    (1_000, 1),
    (10_000, 10),
    (50_000, 50),
    (100_000, 100),
    (500_000, 500),
    (1_000_000, 1000),
)


def get_tick_milli(price_milli: int) -> int:
    for threshold_milli, tick_milli in MILLI_TICK_LADDER:
        if price_milli < threshold_milli:
            return tick_milli
    return 5000


def tv_round_int(number: int, denominator: int) -> int:
    return (number + denominator // 2) // denominator


def round_price_milli_to_tick(price_milli: int, direction: str = "nearest") -> int:
    tick_milli = get_tick_milli(price_milli)
    if direction == "up":
        return ((price_milli + tick_milli - 1) // tick_milli) * tick_milli
    if direction == "down":
        return (price_milli // tick_milli) * tick_milli
    return tv_round_int(price_milli, tick_milli) * tick_milli


def calc_fee_milli(gross_milli: int, fee_ppm: int, min_fee_milli: int) -> int:
    fee_milli = _round_decimal_to_int(Decimal(gross_milli) * Decimal(fee_ppm) / _DECIMAL_PPM)
    return max(fee_milli, int(min_fee_milli))


def calc_tax_milli(gross_milli: int, tax_ppm: int) -> int:
    return _round_decimal_to_int(Decimal(gross_milli) * Decimal(tax_ppm) / _DECIMAL_PPM)


def _resolve_fee_schedule(params) -> Dict[str, int]:
    return {
        "buy_fee_ppm": rate_to_ppm(params.buy_fee),
        "sell_fee_ppm": rate_to_ppm(params.sell_fee),
        "tax_ppm": rate_to_ppm(params.tax_rate),
        "min_fee_milli": money_to_milli(params.min_fee),
        "fixed_risk_ppm": rate_to_ppm(params.fixed_risk),
    }


def build_buy_ledger(fill_price_milli: int, qty: int, params) -> Dict[str, int]:
    schedule = _resolve_fee_schedule(params)
    qty = int(qty)
    gross_buy_milli = int(fill_price_milli) * qty
    buy_fee_milli = calc_fee_milli(gross_buy_milli, schedule["buy_fee_ppm"], schedule["min_fee_milli"])
    net_buy_total_milli = gross_buy_milli + buy_fee_milli
    return {
        "fill_price_milli": int(fill_price_milli),
        "qty": qty,
        "gross_buy_milli": gross_buy_milli,
        "buy_fee_milli": buy_fee_milli,
        "net_buy_total_milli": net_buy_total_milli,
    }


def build_sell_ledger(exec_price_milli: int, qty: int, params) -> Dict[str, int]:
    schedule = _resolve_fee_schedule(params)
    qty = int(qty)
    gross_sell_milli = int(exec_price_milli) * qty
    sell_fee_milli = calc_fee_milli(gross_sell_milli, schedule["sell_fee_ppm"], schedule["min_fee_milli"])
    tax_milli = calc_tax_milli(gross_sell_milli, schedule["tax_ppm"])
    net_sell_total_milli = gross_sell_milli - sell_fee_milli - tax_milli
    return {
        "exec_price_milli": int(exec_price_milli),
        "qty": qty,
        "gross_sell_milli": gross_sell_milli,
        "sell_fee_milli": sell_fee_milli,
        "tax_milli": tax_milli,
        "net_sell_total_milli": net_sell_total_milli,
    }


def calc_initial_risk_total_milli(net_buy_total_milli: int, stop_net_sell_total_milli: int, fixed_risk_ppm: int) -> int:
    init_risk_milli = int(net_buy_total_milli) - int(stop_net_sell_total_milli)
    if init_risk_milli > 0:
        return init_risk_milli
    return max(
        _round_decimal_to_int(Decimal(int(net_buy_total_milli)) * Decimal(int(fixed_risk_ppm)) / _DECIMAL_PPM),
        0,
    )


def allocate_cost_basis_milli(total_cost_basis_milli: int, total_qty: int, sell_qty: int) -> int:
    total_qty = int(total_qty)
    sell_qty = int(sell_qty)
    if total_qty <= 0 or sell_qty <= 0:
        return 0
    if sell_qty >= total_qty:
        return int(total_cost_basis_milli)
    return (int(total_cost_basis_milli) * sell_qty) // total_qty


def calc_average_price_from_total_milli(total_milli: int, qty: int) -> float:
    qty = int(qty)
    if qty <= 0:
        return 0.0
    return float(Decimal(int(total_milli)) / Decimal(qty) / _DECIMAL_THOUSAND)


def build_buy_ledger_from_price(fill_price, qty: int, params) -> Dict[str, int]:
    return build_buy_ledger(price_to_milli(fill_price), qty, params)


def build_sell_ledger_from_price(exec_price, qty: int, params) -> Dict[str, int]:
    return build_sell_ledger(price_to_milli(exec_price), qty, params)


def calc_entry_total_cost(fill_price, qty: int, params) -> float:
    return milli_to_money(build_buy_ledger_from_price(fill_price, qty, params)["net_buy_total_milli"])


def calc_exit_net_total(exec_price, qty: int, params) -> float:
    return milli_to_money(build_sell_ledger_from_price(exec_price, qty, params)["net_sell_total_milli"])


def calc_entry_price_from_total(fill_price, qty: int, params) -> float:
    ledger = build_buy_ledger_from_price(fill_price, qty, params)
    return calc_average_price_from_total_milli(ledger["net_buy_total_milli"], qty)


def calc_net_sell_price_from_total(exec_price, qty: int, params) -> float:
    ledger = build_sell_ledger_from_price(exec_price, qty, params)
    return calc_average_price_from_total_milli(ledger["net_sell_total_milli"], qty)


def calc_limit_up_price_milli(reference_price_milli: int) -> int:
    raw_price_milli = _round_decimal_to_int(Decimal(int(reference_price_milli)) * Decimal(110) / Decimal(100))
    tick_milli = get_tick_milli(int(reference_price_milli))
    return (raw_price_milli // tick_milli) * tick_milli


def calc_limit_down_price_milli(reference_price_milli: int) -> int:
    raw_price_milli = _round_decimal_to_int(Decimal(int(reference_price_milli)) * Decimal(90) / Decimal(100))
    tick_milli = get_tick_milli(int(reference_price_milli))
    return ((raw_price_milli + tick_milli - 1) // tick_milli) * tick_milli


def is_same_price_milli(a, b) -> bool:
    return price_to_milli(a) == price_to_milli(b)




def round_money_for_display(value) -> float:
    return round(float(value), 2)


def calc_risk_budget_milli(cap, risk_pct) -> int:
    cap_milli = coerce_money_like_to_milli(cap)
    risk_ppm = rate_to_ppm(risk_pct)
    return _round_decimal_to_int(Decimal(int(cap_milli)) * Decimal(int(risk_ppm)) / _DECIMAL_PPM)


def get_display_realized_pnl_sum(position: Dict[str, Any]) -> float:
    return round_money_for_display(position.get("display_realized_pnl_sum", 0.0) or 0.0)


def register_display_realized_pnl(position: Dict[str, Any], pnl_display) -> float:
    updated_total = round_money_for_display(get_display_realized_pnl_sum(position) + round_money_for_display(pnl_display))
    position["display_realized_pnl_sum"] = updated_total
    return updated_total


def calc_reconciled_exit_display_pnl(position: Dict[str, Any], total_trade_pnl) -> float:
    return round_money_for_display(round_money_for_display(total_trade_pnl) - get_display_realized_pnl_sum(position))


def sync_position_display_fields(position: Dict[str, Any]) -> Dict[str, Any]:
    initial_qty = int(position.get("initial_qty") or position.get("qty") or 0)
    position["entry"] = calc_average_price_from_total_milli(position.get("net_buy_total_milli", 0), initial_qty)
    position["entry_capital_total"] = milli_to_money(position.get("net_buy_total_milli", 0))
    position["realized_pnl"] = milli_to_money(position.get("realized_pnl_milli", 0))
    position["initial_risk_total"] = milli_to_money(position.get("initial_risk_total_milli", 0))
    if "remaining_cost_basis_milli" in position:
        position["remaining_cost_basis"] = milli_to_money(position.get("remaining_cost_basis_milli", 0))
    return position

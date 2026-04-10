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


STOCK_MILLI_TICK_LADDER = (
    (1_000, 1),
    (10_000, 10),
    (50_000, 50),
    (100_000, 100),
    (500_000, 500),
    (1_000_000, 1000),
)
FUND_MILLI_TICK_LADDER = (
    (50_000, 10),
)
MILLI_TICK_LADDER = STOCK_MILLI_TICK_LADDER

_STOCK_DEFAULT_TICK_MILLI = 5000
_FUND_DEFAULT_TICK_MILLI = 50
_TICK_PROFILE_STOCK = "stock_six_tier"
_TICK_PROFILE_FUND = "fund_two_tier"

_SECURITY_BROAD_STOCK = "stock"
_SECURITY_BROAD_ETF = "etf"
_SECURITY_BROAD_LEVERAGED_INVERSE = "leveraged_inverse"
_SECURITY_BROAD_BOND = "bond"

_LEVERAGED_INVERSE_SUFFIXES = frozenset({"L", "M", "R", "S"})
_BOND_SUFFIXES = frozenset({"B", "C", "D"})
_ETF_GENERAL_SUFFIXES = frozenset({"A", "K", "T", "U", "V"})
_ETN_PREFIX = "02"


def normalize_security_symbol(symbol) -> str:
    if symbol is None:
        return ""
    normalized = str(symbol).strip().upper()
    if not normalized:
        return ""
    for suffix in (".CSV", ".TW", ".TWO"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized.strip()


def _normalize_security_name(security_name) -> str:
    if security_name is None:
        return ""
    return str(security_name).strip().upper()


def _build_security_profile(*, code: str, family: str, broad_type: str, tick_profile: str) -> Dict[str, str]:
    return {
        "normalized_code": code,
        "family": family,
        "broad_type": broad_type,
        "tick_profile": tick_profile,
    }


def infer_security_profile(symbol=None, *, cfi_code=None, security_name=None) -> Dict[str, str]:
    code = normalize_security_symbol(symbol)
    cfi = "" if cfi_code is None else str(cfi_code).strip().upper()
    name = _normalize_security_name(security_name)

    def _etf_like_profile(*, family: str, broad_type: str) -> Dict[str, str]:
        return _build_security_profile(code=code, family=family, broad_type=broad_type, tick_profile=_TICK_PROFILE_FUND)

    if cfi.startswith("ES"):
        return _build_security_profile(code=code, family="stock", broad_type=_SECURITY_BROAD_STOCK, tick_profile=_TICK_PROFILE_STOCK)

    if code:
        if len(code) == 6 and code.startswith(_ETN_PREFIX) and code[:5].isdigit() and (code[-1].isalpha() or code.isdigit()):
            broad_type = _SECURITY_BROAD_ETF
            if code[-1:] in _LEVERAGED_INVERSE_SUFFIXES:
                broad_type = _SECURITY_BROAD_LEVERAGED_INVERSE
            elif code[-1:] in {"B"}:
                broad_type = _SECURITY_BROAD_BOND
            elif "債" in name or "BOND" in name:
                broad_type = _SECURITY_BROAD_BOND
            return _etf_like_profile(family="etn", broad_type=broad_type)

        if code.startswith("00") and code[:-1].isdigit() and code[-1:].isalpha():
            suffix = code[-1]
            if suffix in _LEVERAGED_INVERSE_SUFFIXES:
                return _etf_like_profile(family="etf", broad_type=_SECURITY_BROAD_LEVERAGED_INVERSE)
            if suffix in _BOND_SUFFIXES:
                return _etf_like_profile(family="etf", broad_type=_SECURITY_BROAD_BOND)
            if suffix in _ETF_GENERAL_SUFFIXES:
                return _etf_like_profile(family="etf", broad_type=_SECURITY_BROAD_ETF)

        if code.startswith("00") and code.isdigit() and len(code) in {4, 5, 6}:
            return _etf_like_profile(family="etf", broad_type=_SECURITY_BROAD_ETF)

        if len(code) == 6 and code[:-1].isdigit() and code[-1] == "T" and code.startswith("01"):
            return _etf_like_profile(family="reit", broad_type=_SECURITY_BROAD_ETF)

        if code.isdigit() and len(code) == 4:
            return _build_security_profile(code=code, family="stock", broad_type=_SECURITY_BROAD_STOCK, tick_profile=_TICK_PROFILE_STOCK)

    if cfi.startswith("CE") or "ETF" in name or "交易所交易基金" in name:
        broad_type = _SECURITY_BROAD_ETF
        if any(token in name for token in ("槓桿", "反向", "正2", "反1", "LEVERAGED", "INVERSE")):
            broad_type = _SECURITY_BROAD_LEVERAGED_INVERSE
        elif any(token in name for token in ("債", "固定收益", "BOND")):
            broad_type = _SECURITY_BROAD_BOND
        return _etf_like_profile(family="etf", broad_type=broad_type)

    if "ETN" in name:
        broad_type = _SECURITY_BROAD_BOND if any(token in name for token in ("債", "BOND")) else _SECURITY_BROAD_ETF
        if any(token in name for token in ("槓桿", "反向", "LEVERAGED", "INVERSE")):
            broad_type = _SECURITY_BROAD_LEVERAGED_INVERSE
        return _etf_like_profile(family="etn", broad_type=broad_type)

    if any(token in name for token in ("REIT", "不動產投資信託")):
        return _etf_like_profile(family="reit", broad_type=_SECURITY_BROAD_ETF)

    return _build_security_profile(code=code, family="stock", broad_type=_SECURITY_BROAD_STOCK, tick_profile=_TICK_PROFILE_STOCK)


def resolve_security_profile(security_profile=None, *, ticker=None, cfi_code=None, security_name=None) -> Dict[str, str]:
    if isinstance(security_profile, dict) and security_profile:
        return security_profile
    return infer_security_profile(ticker, cfi_code=cfi_code, security_name=security_name)


def _resolve_tick_rule(*, security_profile=None, ticker=None, cfi_code=None, security_name=None):
    resolved_profile = resolve_security_profile(security_profile, ticker=ticker, cfi_code=cfi_code, security_name=security_name)
    if resolved_profile.get("tick_profile") == _TICK_PROFILE_FUND:
        return FUND_MILLI_TICK_LADDER, _FUND_DEFAULT_TICK_MILLI, resolved_profile
    return STOCK_MILLI_TICK_LADDER, _STOCK_DEFAULT_TICK_MILLI, resolved_profile


def _tick_decimal_from_milli(tick_milli: int) -> Decimal:
    return Decimal(int(tick_milli)) / _DECIMAL_THOUSAND


def get_tick_milli(price_milli: int, *, ticker=None, security_profile=None, cfi_code=None, security_name=None) -> int:
    tick_ladder, default_tick_milli, _resolved_profile = _resolve_tick_rule(
        security_profile=security_profile,
        ticker=ticker,
        cfi_code=cfi_code,
        security_name=security_name,
    )
    for threshold_milli, tick_milli in tick_ladder:
        if price_milli < threshold_milli:
            return tick_milli
    return default_tick_milli


def get_tick_milli_from_price(price, *, ticker=None, security_profile=None, cfi_code=None, security_name=None) -> int:
    price_decimal = _to_decimal(price)
    tick_ladder, default_tick_milli, _resolved_profile = _resolve_tick_rule(
        security_profile=security_profile,
        ticker=ticker,
        cfi_code=cfi_code,
        security_name=security_name,
    )
    for threshold_milli, tick_milli in tick_ladder:
        if price_decimal < (Decimal(int(threshold_milli)) / _DECIMAL_THOUSAND):
            return tick_milli
    return default_tick_milli


def tv_round_int(number: int, denominator: int) -> int:
    return (number + denominator // 2) // denominator


def round_price_milli_to_tick(price_milli: int, direction: str = "nearest", *, ticker=None, security_profile=None, cfi_code=None, security_name=None) -> int:
    tick_milli = get_tick_milli(
        price_milli,
        ticker=ticker,
        security_profile=security_profile,
        cfi_code=cfi_code,
        security_name=security_name,
    )
    if direction == "up":
        return ((price_milli + tick_milli - 1) // tick_milli) * tick_milli
    if direction == "down":
        return (price_milli // tick_milli) * tick_milli
    return tv_round_int(price_milli, tick_milli) * tick_milli


def round_price_to_tick_milli(price, direction: str = "nearest", *, ticker=None, security_profile=None, cfi_code=None, security_name=None) -> int:
    price_decimal = _to_decimal(price)
    tick_milli = get_tick_milli_from_price(
        price_decimal,
        ticker=ticker,
        security_profile=security_profile,
        cfi_code=cfi_code,
        security_name=security_name,
    )
    tick_decimal = _tick_decimal_from_milli(tick_milli)
    ratio = price_decimal / tick_decimal
    if direction == "up":
        rounded_ratio = ratio.quantize(Decimal("1"), rounding=ROUND_CEILING)
    elif direction == "down":
        rounded_ratio = ratio.quantize(Decimal("1"), rounding=ROUND_FLOOR)
    else:
        rounded_ratio = ratio.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return price_to_milli(rounded_ratio * tick_decimal)


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


def calc_total_from_average_price_milli(avg_price, qty: int) -> int:
    qty = int(qty)
    if qty <= 0:
        return 0
    return price_to_milli(avg_price) * qty


def calc_total_from_average_price(avg_price, qty: int) -> float:
    return milli_to_money(calc_total_from_average_price_milli(avg_price, qty))


def calc_entry_price_from_total(fill_price, qty: int, params) -> float:
    ledger = build_buy_ledger_from_price(fill_price, qty, params)
    return calc_average_price_from_total_milli(ledger["net_buy_total_milli"], qty)


def calc_net_sell_price_from_total(exec_price, qty: int, params) -> float:
    ledger = build_sell_ledger_from_price(exec_price, qty, params)
    return calc_average_price_from_total_milli(ledger["net_sell_total_milli"], qty)


def calc_limit_up_price_milli(reference_price_milli: int, *, ticker=None, security_profile=None) -> int:
    reference_price = Decimal(int(reference_price_milli)) / _DECIMAL_THOUSAND
    raw_limit_price = reference_price * Decimal(110) / Decimal(100)
    return round_price_to_tick_milli(raw_limit_price, direction="down", ticker=ticker, security_profile=security_profile)


def calc_limit_down_price_milli(reference_price_milli: int, *, ticker=None, security_profile=None) -> int:
    reference_price = Decimal(int(reference_price_milli)) / _DECIMAL_THOUSAND
    raw_limit_price = reference_price * Decimal(90) / Decimal(100)
    return round_price_to_tick_milli(raw_limit_price, direction="up", ticker=ticker, security_profile=security_profile)


def is_same_price_milli(a, b) -> bool:
    return price_to_milli(a) == price_to_milli(b)




def round_money_for_display(value) -> float:
    return float(_to_decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


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

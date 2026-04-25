from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from numbers import Integral
import math
from functools import lru_cache
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


@lru_cache(maxsize=131072)
def _price_text_to_milli_cached(price_text: str) -> int:
    return _round_decimal_to_int(Decimal(price_text) * _DECIMAL_THOUSAND)


def price_to_milli(price) -> int:
    price_type = type(price)
    if price_type is int:
        return price * MILLI_SCALE
    if price_type is float or price_type.__module__ == "numpy":
        value = float(price)
        if math.isfinite(value):
            scaled = value * MILLI_SCALE
            if scaled >= 0.0:
                return int(math.floor(scaled + 0.5))
            return int(math.ceil(scaled - 0.5))
    if isinstance(price, Decimal):
        return _round_decimal_to_int(price * _DECIMAL_THOUSAND)
    return _price_text_to_milli_cached(str(price))


money_to_milli = price_to_milli


def milli_to_price(price_milli: int) -> float:
    return int(price_milli) / MILLI_SCALE


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
    rate_type = type(rate)
    if rate_type is int:
        return rate * PPM_SCALE
    if rate_type is float or rate_type.__module__ == "numpy":
        value = float(rate)
        if math.isfinite(value):
            scaled = value * PPM_SCALE
            if scaled >= 0.0:
                return int(math.floor(scaled + 0.5))
            return int(math.ceil(scaled - 0.5))
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

_FUND_SELL_TAX_PPM = 1_000
_REIT_SELL_TAX_PPM = 0
_BOND_ETF_TAX_EXEMPTION_END = date(2026, 12, 31)


def _normalize_trade_date(trade_date):
    if trade_date is None:
        return None
    if isinstance(trade_date, datetime):
        return trade_date.date()
    if isinstance(trade_date, date):
        return trade_date
    text_value = str(trade_date).strip()
    if not text_value:
        return None
    candidate = text_value[:10]
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        return None


def resolve_sell_tax_ppm(params, *, ticker=None, security_profile=None, trade_date=None, cfi_code=None, security_name=None) -> int:
    resolved_profile = resolve_security_profile(security_profile, ticker=ticker, cfi_code=cfi_code, security_name=security_name)
    family = resolved_profile.get("family", "stock")
    broad_type = resolved_profile.get("broad_type", _SECURITY_BROAD_STOCK)
    trade_day = _normalize_trade_date(trade_date)
    stock_tax_ppm = rate_to_ppm(params.tax_rate)

    if family == "stock":
        return stock_tax_ppm
    if family == "reit":
        return _REIT_SELL_TAX_PPM
    if family == "etf" and broad_type == _SECURITY_BROAD_BOND:
        if trade_day is None or trade_day <= _BOND_ETF_TAX_EXEMPTION_END:
            return 0
        return _FUND_SELL_TAX_PPM
    if family in {"etf", "etn"}:
        return _FUND_SELL_TAX_PPM
    return stock_tax_ppm


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


@lru_cache(maxsize=512)
def _infer_security_profile_cached(code: str, cfi: str, name: str):
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


@lru_cache(maxsize=2048)
def _resolve_fee_schedule_cached(buy_fee, sell_fee, tax_rate, min_fee, family: str, broad_type: str, trade_day_ord):
    buy_fee_ppm = rate_to_ppm(buy_fee)
    sell_fee_ppm = rate_to_ppm(sell_fee)
    stock_tax_ppm = rate_to_ppm(tax_rate)
    if family == "stock":
        tax_ppm = stock_tax_ppm
    elif family == "reit":
        tax_ppm = _REIT_SELL_TAX_PPM
    elif family == "etf" and broad_type == _SECURITY_BROAD_BOND:
        if trade_day_ord is None or trade_day_ord <= _BOND_ETF_TAX_EXEMPTION_END.toordinal():
            tax_ppm = 0
        else:
            tax_ppm = _FUND_SELL_TAX_PPM
    elif family in {"etf", "etn"}:
        tax_ppm = _FUND_SELL_TAX_PPM
    else:
        tax_ppm = stock_tax_ppm
    return (buy_fee_ppm, sell_fee_ppm, tax_ppm, money_to_milli(min_fee))


def infer_security_profile(symbol=None, *, cfi_code=None, security_name=None) -> Dict[str, str]:
    code = normalize_security_symbol(symbol)
    cfi = "" if cfi_code is None else str(cfi_code).strip().upper()
    name = _normalize_security_name(security_name)
    return dict(_infer_security_profile_cached(code, cfi, name))


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


def _tick_rule_for_profile(tick_profile: str):
    if tick_profile == _TICK_PROFILE_FUND:
        return FUND_MILLI_TICK_LADDER, _FUND_DEFAULT_TICK_MILLI
    return STOCK_MILLI_TICK_LADDER, _STOCK_DEFAULT_TICK_MILLI


def _round_price_to_tick_milli_numeric(value: float, direction: str, tick_profile: str):
    if not math.isfinite(value):
        return None
    tick_ladder, default_tick_milli = _tick_rule_for_profile(tick_profile)
    for threshold_milli, tick_milli in tick_ladder:
        threshold_value = threshold_milli / MILLI_SCALE
        if abs(value - threshold_value) <= 1e-12:
            return None
        if value < threshold_value:
            break
    else:
        tick_milli = default_tick_milli

    scaled = value * MILLI_SCALE
    ratio = scaled / tick_milli
    if direction == "up":
        rounded_ratio = math.ceil(ratio - 1e-12)
    elif direction == "down":
        rounded_ratio = math.floor(ratio + 1e-12)
    else:
        lower = math.floor(ratio)
        fraction = ratio - lower
        if abs(fraction - 0.5) <= 1e-12:
            return None
        rounded_ratio = math.floor(ratio + 0.5)
    return int(rounded_ratio) * int(tick_milli)


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


@lru_cache(maxsize=131072)
def _round_price_to_tick_milli_cached(price_text: str, direction: str, tick_profile: str) -> int:
    price_decimal = Decimal(price_text)
    if tick_profile == _TICK_PROFILE_FUND:
        tick_ladder = FUND_MILLI_TICK_LADDER
        default_tick_milli = _FUND_DEFAULT_TICK_MILLI
    else:
        tick_ladder = STOCK_MILLI_TICK_LADDER
        default_tick_milli = _STOCK_DEFAULT_TICK_MILLI
    for threshold_milli, tick_milli in tick_ladder:
        if price_decimal < (Decimal(int(threshold_milli)) / _DECIMAL_THOUSAND):
            break
    else:
        tick_milli = default_tick_milli
    tick_decimal = _tick_decimal_from_milli(tick_milli)
    ratio = price_decimal / tick_decimal
    if direction == "up":
        rounded_ratio = ratio.quantize(Decimal("1"), rounding=ROUND_CEILING)
    elif direction == "down":
        rounded_ratio = ratio.quantize(Decimal("1"), rounding=ROUND_FLOOR)
    else:
        rounded_ratio = ratio.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return _price_text_to_milli_cached(str(rounded_ratio * tick_decimal))


def round_price_to_tick_milli(price, direction: str = "nearest", *, ticker=None, security_profile=None, cfi_code=None, security_name=None) -> int:
    resolved_profile = resolve_security_profile(security_profile, ticker=ticker, cfi_code=cfi_code, security_name=security_name)
    tick_profile = resolved_profile.get("tick_profile", _TICK_PROFILE_STOCK)
    price_type = type(price)
    if price_type is int:
        numeric_result = _round_price_to_tick_milli_numeric(float(price), direction, tick_profile)
        if numeric_result is not None:
            return numeric_result
    elif price_type is float or price_type.__module__ == "numpy":
        numeric_result = _round_price_to_tick_milli_numeric(float(price), direction, tick_profile)
        if numeric_result is not None:
            return numeric_result
    return _round_price_to_tick_milli_cached(str(price), direction, tick_profile)


def calc_fee_milli(gross_milli: int, fee_ppm: int, min_fee_milli: int) -> int:
    fee_milli = (int(gross_milli) * int(fee_ppm) + (PPM_SCALE // 2)) // PPM_SCALE
    return max(int(fee_milli), int(min_fee_milli))


def calc_tax_milli(gross_milli: int, tax_ppm: int) -> int:
    return (int(gross_milli) * int(tax_ppm) + (PPM_SCALE // 2)) // PPM_SCALE


def _resolve_fee_schedule_tuple(params, *, ticker=None, security_profile=None, trade_date=None, cfi_code=None, security_name=None):
    resolved_profile = resolve_security_profile(
        security_profile,
        ticker=ticker,
        cfi_code=cfi_code,
        security_name=security_name,
    )
    trade_day = _normalize_trade_date(trade_date)
    buy_fee_ppm, sell_fee_ppm, tax_ppm, min_fee_milli = _resolve_fee_schedule_cached(
        float(params.buy_fee),
        float(params.sell_fee),
        float(params.tax_rate),
        float(params.min_fee),
        resolved_profile.get("family", "stock"),
        resolved_profile.get("broad_type", _SECURITY_BROAD_STOCK),
        None if trade_day is None else trade_day.toordinal(),
    )
    return buy_fee_ppm, sell_fee_ppm, tax_ppm, min_fee_milli, rate_to_ppm(params.fixed_risk)


def _resolve_fee_schedule(params, *, ticker=None, security_profile=None, trade_date=None, cfi_code=None, security_name=None) -> Dict[str, int]:
    buy_fee_ppm, sell_fee_ppm, tax_ppm, min_fee_milli, fixed_risk_ppm = _resolve_fee_schedule_tuple(
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
        cfi_code=cfi_code,
        security_name=security_name,
    )
    return {
        "buy_fee_ppm": buy_fee_ppm,
        "sell_fee_ppm": sell_fee_ppm,
        "tax_ppm": tax_ppm,
        "min_fee_milli": min_fee_milli,
        "fixed_risk_ppm": fixed_risk_ppm,
    }


def build_buy_ledger(fill_price_milli: int, qty: int, params) -> Dict[str, int]:
    buy_fee_ppm, _sell_fee_ppm, _tax_ppm, min_fee_milli, _fixed_risk_ppm = _resolve_fee_schedule_tuple(params)
    qty = int(qty)
    gross_buy_milli = int(fill_price_milli) * qty
    buy_fee_milli = calc_fee_milli(gross_buy_milli, buy_fee_ppm, min_fee_milli)
    net_buy_total_milli = gross_buy_milli + buy_fee_milli
    return {
        "fill_price_milli": int(fill_price_milli),
        "qty": qty,
        "gross_buy_milli": gross_buy_milli,
        "buy_fee_milli": buy_fee_milli,
        "net_buy_total_milli": net_buy_total_milli,
    }


def build_sell_ledger(exec_price_milli: int, qty: int, params, *, ticker=None, security_profile=None, trade_date=None, cfi_code=None, security_name=None) -> Dict[str, int]:
    _buy_fee_ppm, sell_fee_ppm, tax_ppm, min_fee_milli, _fixed_risk_ppm = _resolve_fee_schedule_tuple(
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
        cfi_code=cfi_code,
        security_name=security_name,
    )
    qty = int(qty)
    gross_sell_milli = int(exec_price_milli) * qty
    sell_fee_milli = calc_fee_milli(gross_sell_milli, sell_fee_ppm, min_fee_milli)
    tax_milli = calc_tax_milli(gross_sell_milli, tax_ppm)
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
    risk_milli = (int(net_buy_total_milli) * int(fixed_risk_ppm) + (PPM_SCALE // 2)) // PPM_SCALE
    return max(risk_milli, 0)


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
    return int(total_milli) / qty / MILLI_SCALE


def build_buy_ledger_from_price(fill_price, qty: int, params) -> Dict[str, int]:
    return build_buy_ledger(price_to_milli(fill_price), qty, params)


def build_sell_ledger_from_price(exec_price, qty: int, params, *, ticker=None, security_profile=None, trade_date=None, cfi_code=None, security_name=None) -> Dict[str, int]:
    return build_sell_ledger(
        price_to_milli(exec_price),
        qty,
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
        cfi_code=cfi_code,
        security_name=security_name,
    )


def calc_entry_total_cost(fill_price, qty: int, params) -> float:
    return milli_to_money(build_buy_ledger_from_price(fill_price, qty, params)["net_buy_total_milli"])


def calc_exit_net_total(exec_price, qty: int, params, *, ticker=None, security_profile=None, trade_date=None, cfi_code=None, security_name=None) -> float:
    return milli_to_money(
        build_sell_ledger_from_price(
            exec_price,
            qty,
            params,
            ticker=ticker,
            security_profile=security_profile,
            trade_date=trade_date,
            cfi_code=cfi_code,
            security_name=security_name,
        )["net_sell_total_milli"]
    )


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


def calc_net_sell_price_from_total(exec_price, qty: int, params, *, ticker=None, security_profile=None, trade_date=None, cfi_code=None, security_name=None) -> float:
    ledger = build_sell_ledger_from_price(
        exec_price,
        qty,
        params,
        ticker=ticker,
        security_profile=security_profile,
        trade_date=trade_date,
        cfi_code=cfi_code,
        security_name=security_name,
    )
    return calc_average_price_from_total_milli(ledger["net_sell_total_milli"], qty)


def _get_tick_milli_for_ratio(numerator_milli: int, denominator: int, *, ticker=None, security_profile=None) -> int:
    tick_ladder, default_tick_milli, _resolved_profile = _resolve_tick_rule(
        security_profile=security_profile,
        ticker=ticker,
    )
    denominator = int(denominator)
    for threshold_milli, tick_milli in tick_ladder:
        if int(numerator_milli) < int(threshold_milli) * denominator:
            return tick_milli
    return default_tick_milli


def _round_ratio_price_milli_to_tick(numerator_milli: int, denominator: int, direction: str, *, ticker=None, security_profile=None) -> int:
    tick_milli = _get_tick_milli_for_ratio(
        numerator_milli,
        denominator,
        ticker=ticker,
        security_profile=security_profile,
    )
    step = int(denominator) * int(tick_milli)
    if direction == "up":
        return ((int(numerator_milli) + step - 1) // step) * int(tick_milli)
    return (int(numerator_milli) // step) * int(tick_milli)


def calc_limit_up_price_milli(reference_price_milli: int, *, ticker=None, security_profile=None) -> int:
    return _round_ratio_price_milli_to_tick(
        int(reference_price_milli) * 110,
        100,
        "down",
        ticker=ticker,
        security_profile=security_profile,
    )


def calc_limit_down_price_milli(reference_price_milli: int, *, ticker=None, security_profile=None) -> int:
    return _round_ratio_price_milli_to_tick(
        int(reference_price_milli) * 90,
        100,
        "up",
        ticker=ticker,
        security_profile=security_profile,
    )


def is_same_price_milli(a, b) -> bool:
    return price_to_milli(a) == price_to_milli(b)




def round_money_for_display(value) -> float:
    return float(_to_decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def calc_risk_budget_milli(cap, risk_pct) -> int:
    cap_milli = coerce_money_like_to_milli(cap)
    risk_ppm = rate_to_ppm(risk_pct)
    return (int(cap_milli) * int(risk_ppm) + (PPM_SCALE // 2)) // PPM_SCALE


def calc_ratio_from_milli(numerator_milli: int, denominator_milli: int) -> float:
    denominator = int(denominator_milli)
    if denominator <= 0:
        return 0.0
    return int(numerator_milli) / denominator


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

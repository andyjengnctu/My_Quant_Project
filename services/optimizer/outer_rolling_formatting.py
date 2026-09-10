from __future__ import annotations

import re

import pandas as pd

from core.report_style import signal_for_signed_value, terminal_signal
from core.runtime_utils import is_interactive_console
from services.optimizer.score_display import scale_optimizer_score_for_display


OOS_SCORE_DECIMALS = 2


def _fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "N/A"
    total = max(0, int(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_duration_compact(seconds: float | int | None) -> str:
    text = _fmt_duration(seconds)
    if text.startswith("00:"):
        return text[3:]
    return text







def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", str(text))


def _visible_len(text: str) -> int:
    return len(_strip_ansi(str(text)))


def _pad_ansi(text: str, width: int, *, align: str = "<") -> str:
    raw = str(text)
    pad = max(0, int(width) - _visible_len(raw))
    if align == ">":
        return " " * pad + raw
    if align == "^":
        left = pad // 2
        right = pad - left
        return " " * left + raw + " " * right
    return raw + " " * pad


def _color_numeric_text(text: str, value: float | int | None) -> str:
    if value is None:
        return str(text)
    return terminal_signal(str(text), signal_for_signed_value(_safe_float(value, 0.0)), enabled=True)




def _format_plain_score(value) -> str:
    raw_value = _safe_float(value, 0.0)
    return _color_numeric_text(f"{raw_value:.{OOS_SCORE_DECIMALS}f}", raw_value)


def _format_compare(reference_score, rank_1_score) -> str:
    ref = _safe_float(reference_score, 0.0)
    rank_1 = _safe_float(rank_1_score, 0.0)
    gap = rank_1 - ref
    return f"{_color_numeric_text(f'{ref:.{OOS_SCORE_DECIMALS}f}', ref)} ({_color_numeric_text(f'{gap:+.{OOS_SCORE_DECIMALS}f}', gap)})"


def _format_compare_plain(reference_score, rank_1_score) -> str:
    ref = _safe_float(reference_score, 0.0)
    rank_1 = _safe_float(rank_1_score, 0.0)
    return f"{ref:.{OOS_SCORE_DECIMALS}f} ({rank_1 - ref:+.{OOS_SCORE_DECIMALS}f})"


def _format_system_score_compare_plain(reference_score, rank_1_score) -> str:
    ref = scale_optimizer_score_for_display(reference_score)
    rank_1 = scale_optimizer_score_for_display(rank_1_score)
    return f"{ref:.{OOS_SCORE_DECIMALS}f} ({rank_1 - ref:+.{OOS_SCORE_DECIMALS}f})"


def _prompt_int(label: str, default: int, *, minimum: int | None = None) -> int:
    if not is_interactive_console():
        return int(default)
    raw = input(f"{label:<28} [{int(default)}] : ").strip()
    if raw == "":
        value = int(default)
    else:
        value = int(raw)
    if minimum is not None and value < int(minimum):
        raise ValueError(f"{label} 必須 >= {minimum}，收到: {value}")
    return int(value)




def _extract_cli_value(argv, option_name: str) -> str:
    args = list(argv or [])
    # AI註: ``run_outer_rolling_oos`` 同時接受真實CLI argv與內部option-only argv；
    # 內部service的第一個token本身就是option，不可固定從index 1開始掃描。
    first_option_index = 0 if args and str(args[0]).strip().startswith("-") else 1
    for idx in range(first_option_index, len(args)):
        raw = str(args[idx]).strip()
        if raw == option_name and idx + 1 < len(args):
            return str(args[idx + 1]).strip()
        if raw.startswith(option_name + "="):
            return raw.split("=", 1)[1].strip()
    return ""


def _has_cli_flag(argv, option_name: str) -> bool:
    args = list(argv or [])
    first_option_index = 0 if args and str(args[0]).strip().startswith("-") else 1
    return any(
        str(arg).strip() == option_name for arg in args[first_option_index:]
    )


def _month_start(value) -> pd.Timestamp:
    ts = pd.Timestamp(value).normalize()
    return pd.Timestamp(year=int(ts.year), month=int(ts.month), day=1)




def _parse_oos_boundary(value, *, default: pd.Timestamp, year_boundary: str = "start") -> pd.Timestamp:
    text = str(value or "").strip()
    if not text:
        return _month_start(default)
    if len(text) == 4 and text.isdigit():
        month = 12 if str(year_boundary).lower() == "end" else 1
        return pd.Timestamp(year=int(text), month=month, day=1)
    if len(text) == 6 and text.isdigit():
        return pd.Timestamp(year=int(text[:4]), month=int(text[4:]), day=1)
    return _month_start(pd.Timestamp(text))


def _period_key(value) -> int:
    ts = pd.Timestamp(value)
    return int(ts.year) * 100 + int(ts.month)


def _period_label(start, end) -> str:
    return f"{pd.Timestamp(start).strftime('%Y-%m-%d')}~{pd.Timestamp(end).strftime('%Y-%m-%d')}"


def _display_month_value(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "~" in text:
        return _display_month_period(text)
    try:
        if re.fullmatch(r"\d{6}", text):
            return f"{text[:4]}-{text[4:6]}"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return pd.Timestamp(text).strftime("%Y-%m")
        if re.fullmatch(r"\d{4}-\d{2}", text):
            return text
        if re.fullmatch(r"\d{4}", text):
            return text
        return pd.Timestamp(text).strftime("%Y-%m")
    except (TypeError, ValueError, OverflowError):
        return text


def _short_month_period_end(start_text: str, end_text: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}", start_text or "") and re.fullmatch(r"\d{4}-\d{2}", end_text or ""):
        if start_text[:2] == end_text[:2]:
            return end_text[2:]
    return end_text


def _display_month_period(value, end_value=None) -> str:
    if end_value is not None:
        start_text = _display_month_value(value)
        end_text = _display_month_value(end_value)
        if start_text and end_text and start_text != end_text:
            return f"{start_text}~{_short_month_period_end(start_text, end_text)}"
        return start_text or end_text
    text = str(value or "").strip()
    if "~" in text:
        start_text, end_text = text.split("~", 1)
        return _display_month_period(start_text, end_text)
    return _display_month_value(text)




def _display_short_date_value(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "~" in text:
        return _display_short_date_period(text)
    if text.lower() == "latest":
        return "latest"
    try:
        if re.fullmatch(r"\d{8}", text):
            return pd.Timestamp(f"{text[:4]}-{text[4:6]}-{text[6:8]}").strftime("%y-%m-%d")
        if re.fullmatch(r"\d{6}", text):
            return pd.Timestamp(f"{text[:4]}-{text[4:6]}-01").strftime("%y-%m-%d")
        if re.fullmatch(r"\d{4}-\d{2}$", text):
            return pd.Timestamp(f"{text}-01").strftime("%y-%m-%d")
        if re.fullmatch(r"\d{4}$", text):
            return pd.Timestamp(f"{text}-01-01").strftime("%y-%m-%d")
        return pd.Timestamp(text).strftime("%y-%m-%d")
    except (TypeError, ValueError, OverflowError):
        return text


def _display_short_date_period(value, end_value=None) -> str:
    if end_value is not None:
        start_text = _display_short_date_value(value)
        end_raw = str(end_value or "").strip()
        end_text = "latest" if end_raw.lower() == "latest" else _display_short_date_value(end_value)
        if start_text and end_text and start_text != end_text:
            return f"{start_text}~{end_text}"
        return start_text or end_text
    text = str(value or "").strip()
    if "~" in text:
        start_text, end_text = text.split("~", 1)
        return _display_short_date_period(start_text, end_text)
    if text.lower() == "latest":
        return "latest"
    return _display_short_date_value(text)


def _display_compact_month_period(value, end_value=None) -> str:
    return _display_short_date_period(value, end_value)


def _canonical_date_text(value, default: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return str(default or "")
    if text.lower() == "latest":
        return "latest"
    try:
        return pd.Timestamp(text).normalize().strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError):
        return text or str(default or "")


def _split_period_text(value) -> tuple[str, str]:
    text = str(value or "").replace(" ~ ", "~").strip()
    if "~" not in text:
        return text, ""
    start, end = text.split("~", 1)
    return start.strip(), end.strip()


def _infer_period_start(row: dict, *, period_key: str, start_key: str) -> str:
    direct = str((row or {}).get(start_key) or "").strip()
    if direct:
        return _canonical_date_text(direct)
    start, _end = _split_period_text((row or {}).get(period_key))
    return _canonical_date_text(start) if start else ""


def _infer_period_end(row: dict, *, period_key: str, end_key: str) -> str:
    direct = str((row or {}).get(end_key) or "").strip()
    if direct:
        return _canonical_date_text(direct)
    _start, end = _split_period_text((row or {}).get(period_key))
    return _canonical_date_text(end) if end else ""


def _timestamp_or_none(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def _timestamp_year_or_none(value) -> int | None:
    timestamp = _timestamp_or_none(value)
    if timestamp is None:
        return None
    return int(timestamp.year)


def _oos_key_from_start_date(value, default: int = 0) -> int:
    text = str(value or "").strip()
    if not text:
        return int(default)
    timestamp = _timestamp_or_none(text)
    if timestamp is not None:
        return _period_key(timestamp)
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 6:
        try:
            return int(digits[:6])
        except ValueError:
            return int(default)
    if len(digits) >= 4:
        try:
            return int(digits[:4]) * 100 + 1
        except ValueError:
            return int(default)
    return int(default)

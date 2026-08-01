"""Shared readable/color report helpers for breakout-quality strategy diagnostics."""

from __future__ import annotations

from html import escape
import math
from typing import Any

from core.display import C_GREEN, C_GRAY, C_RED, C_RESET, C_YELLOW

SIGNAL_POSITIVE = "positive"
SIGNAL_NEGATIVE = "negative"
SIGNAL_WARNING = "warning"
SIGNAL_NEUTRAL = "neutral"

_SIGNAL_MARKERS = {
    SIGNAL_POSITIVE: "🟢",
    SIGNAL_NEGATIVE: "🔴",
    SIGNAL_WARNING: "🟡",
    SIGNAL_NEUTRAL: "⚪",
}
_SIGNAL_LABELS = {
    SIGNAL_POSITIVE: "改善",
    SIGNAL_NEGATIVE: "惡化",
    SIGNAL_WARNING: "注意",
    SIGNAL_NEUTRAL: "中性",
}
_SIGNAL_COLORS = {
    SIGNAL_POSITIVE: C_GREEN,
    SIGNAL_NEGATIVE: C_RED,
    SIGNAL_WARNING: C_YELLOW,
    SIGNAL_NEUTRAL: C_GRAY,
}


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def signal_for_delta(
    value: Any,
    *,
    preference: str,
    tolerance: float = 1e-12,
    warning_threshold: float | None = None,
) -> str:
    """Classify a delta using an explicit metric direction contract."""

    numeric = finite_number(value)
    if numeric is None or abs(numeric) <= float(tolerance):
        return SIGNAL_NEUTRAL
    if preference == "higher":
        return SIGNAL_POSITIVE if numeric > 0 else SIGNAL_NEGATIVE
    if preference == "lower":
        return SIGNAL_POSITIVE if numeric < 0 else SIGNAL_NEGATIVE
    if preference == "attention":
        threshold = float(warning_threshold or 0.0)
        return SIGNAL_WARNING if abs(numeric) > threshold else SIGNAL_NEUTRAL
    if preference == "neutral":
        return SIGNAL_NEUTRAL
    raise ValueError(f"不支援的metric preference: {preference!r}")


def signal_marker(signal: str, *, include_label: bool = True) -> str:
    normalized = str(signal)
    marker = _SIGNAL_MARKERS.get(normalized, _SIGNAL_MARKERS[SIGNAL_NEUTRAL])
    if not include_label:
        return marker
    return f"{marker} {_SIGNAL_LABELS.get(normalized, _SIGNAL_LABELS[SIGNAL_NEUTRAL])}"


def terminal_signal(text: str, signal: str) -> str:
    color = _SIGNAL_COLORS.get(str(signal), C_GRAY)
    return f"{color}{text}{C_RESET}"


def html_signal_badge(signal: str) -> str:
    normalized = str(signal)
    label = signal_marker(normalized)
    return f'<span class="badge {escape(normalized)}">{escape(label)}</span>'


def html_delta_cell(text: str, signal: str) -> str:
    normalized = str(signal)
    return f'<td class="delta {escape(normalized)}">{escape(str(text))}</td>'


def html_page(*, title: str, body: str, subtitle: str = "") -> str:
    subtitle_html = f'<p class="subtitle">{escape(subtitle)}</p>' if subtitle else ""
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: light; --green:#147d3f; --green-bg:#eaf7ef; --red:#b42318; --red-bg:#fff0ee; --yellow:#8a5a00; --yellow-bg:#fff7df; --gray:#52606d; --gray-bg:#f4f6f8; --border:#d7dde3; --ink:#17212b; --muted:#5f6b76; }}
body {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",Arial,sans-serif; margin:0; background:#f6f8fa; color:var(--ink); line-height:1.55; }}
main {{ max-width:1180px; margin:28px auto; padding:0 20px 48px; }}
h1 {{ margin:0 0 4px; font-size:30px; }} h2 {{ margin-top:30px; border-bottom:2px solid #e5e9ed; padding-bottom:8px; }}
.subtitle {{ color:var(--muted); margin-top:0; }}
.card {{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:18px 20px; margin:16px 0; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
.meta-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:8px 20px; }}
.meta-grid div {{ overflow-wrap:anywhere; }}
table {{ width:100%; border-collapse:collapse; background:#fff; margin:12px 0 20px; font-variant-numeric:tabular-nums; }}
th,td {{ border:1px solid var(--border); padding:9px 10px; text-align:right; }} th:first-child,td:first-child {{ text-align:left; }}
th {{ background:#eef2f5; }}
tr:hover td {{ background:#fafcfd; }}
.delta.positive,.positive {{ color:var(--green); background:var(--green-bg); font-weight:700; }}
.delta.negative,.negative {{ color:var(--red); background:var(--red-bg); font-weight:700; }}
.delta.warning,.warning {{ color:var(--yellow); background:var(--yellow-bg); font-weight:700; }}
.delta.neutral,.neutral {{ color:var(--gray); background:var(--gray-bg); }}
.badge {{ display:inline-block; border-radius:999px; padding:3px 9px; font-size:13px; white-space:nowrap; }}
.callout {{ border-left:5px solid var(--gray); padding:12px 16px; background:#fff; margin:14px 0; }}
.callout.positive {{ border-color:var(--green); }} .callout.negative {{ border-color:var(--red); }} .callout.warning {{ border-color:var(--yellow); }}
.small {{ color:var(--muted); font-size:13px; }}
code {{ background:#eef2f5; border-radius:4px; padding:1px 5px; }}
ul.compact {{ margin:8px 0; padding-left:22px; }}
</style>
</head>
<body><main><h1>{escape(title)}</h1>{subtitle_html}{body}</main></body>
</html>
"""


__all__ = [
    "SIGNAL_POSITIVE", "SIGNAL_NEGATIVE", "SIGNAL_WARNING", "SIGNAL_NEUTRAL",
    "finite_number", "signal_for_delta", "signal_marker", "terminal_signal",
    "html_signal_badge", "html_delta_cell", "html_page",
]

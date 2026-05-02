import os
import re
import time
import warnings

import numpy as np
import pandas as pd


CHART_SIGNAL_BOX_ALPHA = 0.44
CHART_RIGHT_PADDING_BARS = 8
CHART_RIGHT_PADDING_RATIO = 0.1666666667
CHART_DEFAULT_LOOKBACK_MONTHS = 6
CHART_DEFAULT_LOOKBACK_FALLBACK_BARS = 126
CHART_FOCUS_PADDING_BARS = 15
CHART_FALLBACK_TAIL_BARS = 120
CHART_MIN_WINDOW_BARS = 40
CHART_PRICE_PADDING_RATIO = 0.035
CHART_VOLUME_PADDING_RATIO = 0.10
MATPLOTLIB_DEBUG_CHART_FIGSIZE = (18.2, 10.6)
MATPLOTLIB_SUBPLOT_LEFT = 0.046
MATPLOTLIB_SUBPLOT_RIGHT = 0.988
MATPLOTLIB_SUBPLOT_TOP = 0.986
MATPLOTLIB_SUBPLOT_BOTTOM = 0.058
MATPLOTLIB_CANDLE_WIDTH = 0.72
MATPLOTLIB_MARKER_SIZE = 172
MATPLOTLIB_VOLUME_ALPHA = 0.32
MATPLOTLIB_DARK_BG = "#000000"
MATPLOTLIB_GRID_COLOR = "#0a1824"
MATPLOTLIB_TEXT_COLOR = "#fbfdff"
MATPLOTLIB_MUTED_TEXT_COLOR = "#e4edf6"
MATPLOTLIB_UP_COLOR = "#ff5b6e"
MATPLOTLIB_DOWN_COLOR = "#18b26b"
MATPLOTLIB_STOP_COLOR = "#ff4d4f"
MATPLOTLIB_TP_COLOR = "#facc15"
MATPLOTLIB_INDICATOR_SELL_COLOR = "#22c55e"
MATPLOTLIB_LIMIT_COLOR = "#4f86ff"
MATPLOTLIB_ENTRY_COLOR = "#2f6df6"
MATPLOTLIB_INFO_BOX_FACE = (0.02, 0.04, 0.06, 0.80)
MATPLOTLIB_SIGNAL_BUY_COLOR = "#1f9cf0"
MATPLOTLIB_SIGNAL_SELL_COLOR = "#ff6174"
MATPLOTLIB_SIGNAL_TEXT_COLOR = "#f8fafc"
CHART_RUNTIME_FALLBACK_WARNING_KEYS = set()
MATPLOTLIB_VOLUME_OVERLAY_HEIGHT_RATIO = 0.14
MATPLOTLIB_VOLUME_OVERLAY_BOTTOM_GAP = 0.015
MATPLOTLIB_CROSSHAIR_COLOR = "#b8d1e8"
MATPLOTLIB_HOVER_BOX_FACE = (0.00, 0.00, 0.00, 0.00)
MATPLOTLIB_CJK_FONT_CANDIDATES = (
    "Microsoft JhengHei",
    "Microsoft JhengHei UI",
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "PingFang TC",
    "Heiti TC",
    "Source Han Sans TW",
    "Source Han Sans TC",
    "Sarasa Gothic TC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "SimHei",
    "Noto Sans CJK JP",
)

MATPLOTLIB_MIN_VISIBLE_BARS = 12
MATPLOTLIB_WHEEL_ZOOM_IN_FACTOR = 0.82
MATPLOTLIB_WHEEL_ZOOM_OUT_FACTOR = 1.22

MATPLOTLIB_PAN_CURSOR = "fleur"
MATPLOTLIB_PAN_REDRAW_MIN_INTERVAL_SEC = 0.012
MATPLOTLIB_PAN_REDRAW_MIN_PIXEL_DELTA = 2.0
MATPLOTLIB_KEY_PAN_STEP_RATIO = 0.16
MATPLOTLIB_KEY_EDGE_MARGIN_BARS = 2
MATPLOTLIB_DYNAMIC_BODY_WIDTH_RANGE = (1.8, 7.4)
MATPLOTLIB_DYNAMIC_WICK_WIDTH_RANGE = (0.7, 1.7)
MATPLOTLIB_VOLUME_WIDTH_SCALE = 0.72
MATPLOTLIB_STATUS_CHIP_BUY_FACE = (0.10, 0.46, 0.94, 0.92)
MATPLOTLIB_STATUS_CHIP_GATE_FACE = (0.98, 0.53, 0.10, 0.92)
MATPLOTLIB_STATUS_CHIP_SELL_FACE = (0.90, 0.25, 0.34, 0.92)
MATPLOTLIB_STATUS_CHIP_MUTED_FACE = (0.01, 0.02, 0.04, 0.96)
MATPLOTLIB_STATUS_CHIP_FONT_SIZE = 15
MATPLOTLIB_SUMMARY_FONT_SIZE = 14
MATPLOTLIB_SIGNAL_FONT_SIZE = 13
MATPLOTLIB_SIDEBAR_FONT_SIZE = 16
MATPLOTLIB_SIGNAL_ARROW_MUTATION_SCALE = 22
MATPLOTLIB_BUY_FILL_FACE = (0.08, 0.28, 0.86, 0.38)
MATPLOTLIB_SELL_PROFIT_FACE = (0.92, 0.22, 0.30, 0.40)
MATPLOTLIB_SELL_LOSS_FACE = (0.10, 0.60, 0.24, 0.40)
MATPLOTLIB_SIGNAL_BUY_FACE = (0.10, 0.46, 0.94, CHART_SIGNAL_BOX_ALPHA)
MATPLOTLIB_SIGNAL_SELL_PROFIT_FACE = (0.92, 0.22, 0.30, CHART_SIGNAL_BOX_ALPHA)
MATPLOTLIB_SIGNAL_SELL_LOSS_FACE = (0.10, 0.60, 0.24, CHART_SIGNAL_BOX_ALPHA)


ACTION_STYLE_MAP = {
    "限價買進": {"plotly_symbol": "line-ew-open", "mpl_marker": "_", "color": MATPLOTLIB_LIMIT_COLOR},
    "限價買進(延續候選)": {"plotly_symbol": "line-ew-open", "mpl_marker": "_", "color": "#7fb3ff"},
    "買進": {"plotly_symbol": "triangle-up", "mpl_marker": "^", "color": MATPLOTLIB_ENTRY_COLOR},
    "買進(延續候選)": {"plotly_symbol": "triangle-up", "mpl_marker": "^", "color": "#68d8ff"},
    "錯失買進(新訊號)": {"plotly_symbol": "circle-open", "mpl_marker": "o", "color": MATPLOTLIB_LIMIT_COLOR},
    "錯失買進(延續候選)": {"plotly_symbol": "circle-open", "mpl_marker": "o", "color": "#7fb3ff"},
    "半倉停利": {"plotly_symbol": "diamond", "mpl_marker": "D", "color": MATPLOTLIB_TP_COLOR},
    "停損殺出": {"plotly_symbol": "x", "mpl_marker": "x", "color": MATPLOTLIB_STOP_COLOR},
    "指標賣出": {"plotly_symbol": "line-ew-open", "mpl_marker": "_", "color": MATPLOTLIB_INDICATOR_SELL_COLOR},
    "期末強制結算": {"plotly_symbol": "square", "mpl_marker": "s", "color": "#facc15"},
    "錯失賣出": {"plotly_symbol": "circle-open", "mpl_marker": "o", "color": "#fbbf24"},
}

CHART_EVENT_LEGEND_ORDER = (
    "買訊",
    "賣訊",
    "限價買進",
    "限價買進(延續候選)",
    "買進",
    "買進(延續候選)",
    "錯失買進(新訊號)",
    "錯失買進(延續候選)",
    "半倉停利",
    "停損殺出",
    "指標賣出",
    "期末強制結算",
    "錯失賣出",
)

CHART_LINE_LEGEND_SPECS = (
    ("停損線", MATPLOTLIB_STOP_COLOR, "solid", 2.0),
    ("半倉停利線", MATPLOTLIB_TP_COLOR, "solid", 1.9),
    ("限價線", MATPLOTLIB_LIMIT_COLOR, (0, (4, 2)), 1.5),
    ("成交線", MATPLOTLIB_ENTRY_COLOR, "solid", 1.8),
)

CHART_SIGNAL_LEGEND_STYLE = {
    "買訊": {"mpl_marker": "v", "plotly_symbol": "triangle-down", "color": MATPLOTLIB_SIGNAL_BUY_COLOR},
    "賣訊": {"mpl_marker": "^", "plotly_symbol": "triangle-up", "color": MATPLOTLIB_SIGNAL_SELL_COLOR},
}

ORDER_STATUS_LABELS = {
    "filled": "成交",
    "missed": "未成交",
    "abandoned": "放棄進場",
}


def _warn_chart_runtime_fallback_once(key, exc, *, context):
    if key in CHART_RUNTIME_FALLBACK_WARNING_KEYS:
        return
    CHART_RUNTIME_FALLBACK_WARNING_KEYS.add(key)
    warnings.warn(f"{context}: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)


MATPLOTLIB_FONT_MANAGER_IMPORT_ERROR = ""


def get_matplotlib_cjk_font_candidates():
    return MATPLOTLIB_CJK_FONT_CANDIDATES


def get_matplotlib_font_manager_import_error():
    return MATPLOTLIB_FONT_MANAGER_IMPORT_ERROR


def create_debug_chart_context(df):
    dates = pd.DatetimeIndex(pd.to_datetime(df.index))
    total = len(dates)
    return {
        "dates": dates,
        "date_to_pos": {pd.Timestamp(dt): idx for idx, dt in enumerate(dates)},
        "stop_line": np.full(total, np.nan, dtype=np.float32),
        "tp_line": np.full(total, np.nan, dtype=np.float32),
        "limit_line": np.full(total, np.nan, dtype=np.float32),
        "entry_line": np.full(total, np.nan, dtype=np.float32),
        "order_markers": [],
        "trade_markers": [],
        "signal_annotations": [],
        "summary_box": None,
        "status_box": None,
        "future_preview": {},
    }


def _resolve_chart_pos(chart_context, current_date):
    current_ts = pd.Timestamp(current_date)
    pos = chart_context["date_to_pos"].get(current_ts)
    if pos is None:
        raise KeyError(f"chart context 找不到日期: {current_ts!s}")
    return pos


def _append_marker(marker_list, *, trace_name, current_date, price, qty, hover_text, note="", meta=None):
    if pd.isna(price):
        return
    marker_list.append(
        {
            "trace_name": trace_name,
            "date": pd.Timestamp(current_date),
            "price": float(price),
            "qty": int(qty) if qty is not None and not pd.isna(qty) else 0,
            "note": str(note or ""),
            "hover_text": hover_text,
            "meta": dict(meta or {}),
        }
    )


def record_active_levels(chart_context, *, current_date, stop_price=np.nan, tp_half_price=np.nan, limit_price=np.nan, entry_price=np.nan):
    if chart_context is None:
        return
    pos = _resolve_chart_pos(chart_context, current_date)
    if not pd.isna(stop_price):
        chart_context["stop_line"][pos] = float(stop_price)
    if not pd.isna(tp_half_price):
        chart_context["tp_line"][pos] = float(tp_half_price)
    if not pd.isna(limit_price):
        chart_context["limit_line"][pos] = float(limit_price)
    if not pd.isna(entry_price):
        chart_context["entry_line"][pos] = float(entry_price)


def record_limit_order(chart_context, *, current_date, limit_price, qty, entry_type, status, note=""):
    if chart_context is None or pd.isna(limit_price):
        return
    trace_name = "限價買進(延續候選)" if entry_type == "extended" else "限價買進"
    status_label = ORDER_STATUS_LABELS.get(status, status)
    hover_text = (
        f"{trace_name}<br>日期: {pd.Timestamp(current_date).strftime('%Y-%m-%d')}"
        f"<br>預掛限價: {float(limit_price):.2f}"
        f"<br>股數: {int(qty)}"
        f"<br>結果: {status_label}"
    )
    if note:
        hover_text += f"<br>備註: {note}"
    _append_marker(
        chart_context["order_markers"],
        trace_name=trace_name,
        current_date=current_date,
        price=limit_price,
        qty=qty,
        note=note,
        hover_text=hover_text,
    )


def record_trade_marker(chart_context, *, current_date, action, price, qty, note="", meta=None):
    if chart_context is None or pd.isna(price):
        return
    hover_text = (
        f"{action}<br>日期: {pd.Timestamp(current_date).strftime('%Y-%m-%d')}"
        f"<br>價格: {float(price):.2f}"
        f"<br>股數: {int(qty)}"
    )
    if note:
        hover_text += f"<br>備註: {note}"
    _append_marker(
        chart_context["trade_markers"],
        trace_name=action,
        current_date=current_date,
        price=price,
        qty=qty,
        note=note,
        hover_text=hover_text,
        meta=meta,
    )


def record_signal_annotation(chart_context, *, current_date, signal_type, anchor_price, title, detail_lines, note="", meta=None):
    if chart_context is None or pd.isna(anchor_price):
        return
    normalized_signal_type = str(signal_type).strip().lower()
    if normalized_signal_type not in {"buy", "sell"}:
        raise ValueError(f"不支援的 signal_type: {signal_type!r}")
    normalized_meta = dict(meta or {})
    detail_text = _build_signal_label_detail_text(str(title), normalized_meta)
    if not detail_text:
        detail_text = "\n".join(str(line) for line in detail_lines if str(line).strip())
    chart_context["signal_annotations"].append(
        {
            "date": pd.Timestamp(current_date),
            "signal_type": normalized_signal_type,
            "anchor_price": float(anchor_price),
            "title": str(title),
            "detail_text": detail_text,
            "note": str(note or ""),
            "meta": normalized_meta,
        }
    )


def set_chart_summary_box(chart_context, *, summary_lines):
    if chart_context is None:
        return
    chart_context["summary_box"] = [str(line) for line in summary_lines if str(line).strip()]


def set_chart_status_box(chart_context, *, status_lines, ok=True):
    if chart_context is None:
        return
    chart_context["status_box"] = {
        "lines": [str(line) for line in status_lines if str(line).strip()],
        "ok": bool(ok),
    }


def set_chart_future_preview(chart_context, *, limit_price=np.nan, stop_price=np.nan, tp_half_price=np.nan, entry_price=np.nan):
    if chart_context is None:
        return
    chart_context["future_preview"] = {
        "limit_price": np.nan if pd.isna(limit_price) else float(limit_price),
        "stop_price": np.nan if pd.isna(stop_price) else float(stop_price),
        "tp_half_price": np.nan if pd.isna(tp_half_price) else float(tp_half_price),
        "entry_price": np.nan if pd.isna(entry_price) else float(entry_price),
    }


def _normalize_line_array(values, expected_len):
    if values is None:
        return np.full(expected_len, np.nan, dtype=np.float32)
    normalized = np.asarray(values, dtype=np.float32)
    if normalized.size == expected_len:
        return normalized.copy()
    resized = np.full(expected_len, np.nan, dtype=np.float32)
    copy_len = min(expected_len, normalized.size)
    resized[:copy_len] = normalized[:copy_len]
    return resized


def _build_marker_groups(*, marker_lists, date_to_pos):
    groups = {}
    focus_positions = []
    for marker in marker_lists:
        marker_date = pd.Timestamp(marker["date"])
        pos = date_to_pos.get(marker_date)
        if pos is None:
            continue
        normalized_marker = {
            "trace_name": marker["trace_name"],
            "date": marker_date,
            "x": int(pos),
            "price": float(marker["price"]),
            "qty": int(marker.get("qty", 0) or 0),
            "note": str(marker.get("note", "") or ""),
            "hover_text": marker["hover_text"],
            "meta": dict(marker.get("meta") or {}),
        }
        groups.setdefault(marker["trace_name"], []).append(normalized_marker)
        focus_positions.append(int(pos))
    return groups, focus_positions


def _build_signal_annotations(*, signal_annotations, date_to_pos):
    normalized_annotations = []
    focus_positions = []
    for item in signal_annotations:
        annotation_date = pd.Timestamp(item["date"])
        pos = date_to_pos.get(annotation_date)
        if pos is None:
            continue
        normalized = {
            "date": annotation_date,
            "x": int(pos),
            "anchor_price": float(item["anchor_price"]),
            "signal_type": item["signal_type"],
            "title": item["title"],
            "detail_text": item.get("detail_text", ""),
            "note": item.get("note", ""),
            "meta": dict(item.get("meta") or {}),
        }
        normalized_annotations.append(normalized)
        focus_positions.append(int(pos))
    return normalized_annotations, focus_positions


def _expand_window_to_min_bars(start_idx, end_idx, *, total_bars, min_window_bars):
    if total_bars <= 0:
        return 0, 0
    current_width = end_idx - start_idx + 1
    if current_width >= min_window_bars:
        return max(0, start_idx), min(total_bars - 1, end_idx)
    extra_bars = min_window_bars - current_width
    left_expand = extra_bars // 2
    right_expand = extra_bars - left_expand
    start_idx = max(0, start_idx - left_expand)
    end_idx = min(total_bars - 1, end_idx + right_expand)
    current_width = end_idx - start_idx + 1
    if current_width >= min_window_bars:
        return start_idx, end_idx
    remaining = min_window_bars - current_width
    if start_idx == 0:
        end_idx = min(total_bars - 1, end_idx + remaining)
    elif end_idx == total_bars - 1:
        start_idx = max(0, start_idx - remaining)
    return start_idx, end_idx


def compute_default_view_window(dates, total_bars, focus_positions, *, default_lookback_months=CHART_DEFAULT_LOOKBACK_MONTHS, fallback_tail_bars=CHART_DEFAULT_LOOKBACK_FALLBACK_BARS, min_window_bars=CHART_MIN_WINDOW_BARS):
    if total_bars <= 0:
        return {"start_idx": 0, "end_idx": 0}
    if isinstance(dates, pd.DatetimeIndex) and len(dates) > 0:
        last_ts = pd.Timestamp(dates[-1])
        cutoff_ts = last_ts - pd.DateOffset(months=int(default_lookback_months))
        start_idx = int(dates.searchsorted(cutoff_ts, side="left"))
        start_idx = max(0, min(start_idx, total_bars - 1))
        end_idx = total_bars - 1
    elif focus_positions:
        start_idx = max(0, min(focus_positions) - int(CHART_FOCUS_PADDING_BARS))
        end_idx = min(total_bars - 1, max(focus_positions) + int(CHART_FOCUS_PADDING_BARS))
    else:
        tail_bars = min(total_bars, int(fallback_tail_bars))
        start_idx = max(0, total_bars - tail_bars)
        end_idx = total_bars - 1
    start_idx, end_idx = _expand_window_to_min_bars(int(start_idx), int(end_idx), total_bars=total_bars, min_window_bars=min(int(min_window_bars), total_bars))
    return {"start_idx": int(start_idx), "end_idx": int(end_idx)}


def compute_gui_render_window(chart_payload):
    total_bars = int(len(chart_payload["x"]))
    if total_bars <= 0:
        return {"start_idx": 0, "end_idx": 0}
    return {"start_idx": 0, "end_idx": total_bars - 1}


def build_debug_chart_payload(price_df, chart_context):
    df_chart = price_df.copy()
    df_chart.index = pd.DatetimeIndex(pd.to_datetime(df_chart.index))
    df_chart = df_chart.sort_index()
    dates = pd.DatetimeIndex(df_chart.index)
    total_bars = int(len(df_chart))
    x_positions = np.arange(total_bars, dtype=np.float32)
    date_to_pos = {pd.Timestamp(dt): idx for idx, dt in enumerate(dates)}
    marker_groups, focus_positions = _build_marker_groups(marker_lists=[*(chart_context or {}).get("order_markers", []), *((chart_context or {}).get("trade_markers", []))], date_to_pos=date_to_pos)
    signal_annotations, signal_focus_positions = _build_signal_annotations(signal_annotations=(chart_context or {}).get("signal_annotations", []), date_to_pos=date_to_pos)
    focus_positions = [*focus_positions, *signal_focus_positions]
    payload = {
        "dates": dates,
        "date_labels": [dt.strftime("%Y-%m-%d") for dt in dates],
        "x": x_positions,
        "open": df_chart["Open"].to_numpy(dtype=np.float32, copy=False),
        "high": df_chart["High"].to_numpy(dtype=np.float32, copy=False),
        "low": df_chart["Low"].to_numpy(dtype=np.float32, copy=False),
        "close": df_chart["Close"].to_numpy(dtype=np.float32, copy=False),
        "volume": df_chart["Volume"].to_numpy(dtype=np.float32, copy=False),
        "up_mask": (df_chart["Close"] >= df_chart["Open"]).to_numpy(dtype=bool, copy=False),
        "stop_line": _normalize_line_array((chart_context or {}).get("stop_line"), total_bars),
        "tp_line": _normalize_line_array((chart_context or {}).get("tp_line"), total_bars),
        "limit_line": _normalize_line_array((chart_context or {}).get("limit_line"), total_bars),
        "entry_line": _normalize_line_array((chart_context or {}).get("entry_line"), total_bars),
        "marker_groups": marker_groups,
        "signal_annotations": signal_annotations,
        "focus_positions": focus_positions,
        "summary_box": list((chart_context or {}).get("summary_box") or []),
        "status_box": dict((chart_context or {}).get("status_box") or {}),
        "future_preview": dict((chart_context or {}).get("future_preview") or {}),
    }
    payload["default_view"] = compute_default_view_window(dates, total_bars, focus_positions)
    payload["gui_render_window"] = compute_gui_render_window(payload)
    return payload


def normalize_chart_payload_contract(chart_payload):
    if chart_payload is None:
        raise ValueError("chart_payload 不可為空。")
    normalized = dict(chart_payload)
    x_positions = np.asarray(normalized.get("x", []), dtype=np.float32)
    if x_positions.size == 0:
        raise ValueError("chart_payload 不可為空。")
    total_bars = int(x_positions.size)
    normalized["x"] = x_positions

    dates = normalized.get("dates")
    if dates is None or len(dates) != total_bars:
        dates = pd.date_range("1970-01-01", periods=total_bars, freq="D")
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
    normalized["dates"] = dates

    date_labels = list(normalized.get("date_labels") or [])
    if len(date_labels) != total_bars:
        date_labels = [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in dates]
    normalized["date_labels"] = date_labels

    for key in ("open", "high", "low", "close"):
        normalized[key] = np.asarray(normalized[key], dtype=np.float32)
    normalized["volume"] = np.asarray(normalized.get("volume", np.zeros(total_bars, dtype=np.float32)), dtype=np.float32)

    up_mask = normalized.get("up_mask")
    if up_mask is None or len(up_mask) != total_bars:
        up_mask = normalized["close"] >= normalized["open"]
    normalized["up_mask"] = np.asarray(up_mask, dtype=bool)

    for key in ("stop_line", "tp_line", "limit_line", "entry_line"):
        normalized[key] = _normalize_line_array(normalized.get(key), total_bars)

    normalized["marker_groups"] = {str(name): list(markers) for name, markers in dict(normalized.get("marker_groups") or {}).items()}
    normalized["signal_annotations"] = list(normalized.get("signal_annotations") or [])
    focus_positions = normalized.get("focus_positions") or []
    normalized["focus_positions"] = [int(pos) for pos in focus_positions if 0 <= int(pos) < total_bars]
    normalized["summary_box"] = list(normalized.get("summary_box") or [])
    normalized["status_box"] = dict(normalized.get("status_box") or {})
    normalized["future_preview"] = dict(normalized.get("future_preview") or {})

    default_view = dict(normalized.get("default_view") or {})
    if not default_view:
        default_view = compute_default_view_window(dates, total_bars, normalized["focus_positions"])
    start_idx = int(default_view.get("start_idx", 0))
    end_idx = int(default_view.get("end_idx", total_bars - 1))
    start_idx, end_idx = _expand_window_to_min_bars(start_idx, end_idx, total_bars=total_bars, min_window_bars=min(CHART_MIN_WINDOW_BARS, total_bars))
    normalized["default_view"] = {"start_idx": int(start_idx), "end_idx": int(end_idx)}
    normalized["gui_render_window"] = dict(normalized.get("gui_render_window") or compute_gui_render_window(normalized))
    return normalized


def _slice_visible_window(array, start_idx, end_idx):
    return np.asarray(array[start_idx : end_idx + 1])


def compute_visible_value_ranges(chart_payload, *, start_idx, end_idx, price_padding_ratio=CHART_PRICE_PADDING_RATIO, volume_padding_ratio=CHART_VOLUME_PADDING_RATIO):
    total_bars = int(len(chart_payload["x"]))
    if total_bars <= 0:
        return {"price_min": 0.0, "price_max": 1.0, "volume_min": 0.0, "volume_max": 1.0}
    requested_end_idx = float(end_idx)
    start_idx = max(0, min(int(np.floor(start_idx)), total_bars - 1))
    end_idx = max(start_idx, min(int(np.ceil(end_idx)), total_bars - 1))
    candidate_price_arrays = [
        _slice_visible_window(chart_payload["low"], start_idx, end_idx),
        _slice_visible_window(chart_payload["high"], start_idx, end_idx),
        _slice_visible_window(chart_payload["stop_line"], start_idx, end_idx),
        _slice_visible_window(chart_payload["tp_line"], start_idx, end_idx),
        _slice_visible_window(chart_payload["limit_line"], start_idx, end_idx),
        _slice_visible_window(chart_payload["entry_line"], start_idx, end_idx),
    ]
    marker_prices = []
    for markers in chart_payload["marker_groups"].values():
        marker_prices.extend(marker["price"] for marker in markers if start_idx <= int(marker["x"]) <= end_idx and not pd.isna(marker["price"]))
    signal_annotation_prices = []
    for item in chart_payload.get("signal_annotations", []):
        if not (start_idx <= int(item["x"]) <= end_idx):
            continue
        anchor_price = item.get("anchor_price")
        if anchor_price is not None and not pd.isna(anchor_price):
            signal_annotation_prices.append(float(anchor_price))
        meta = dict(item.get("meta") or {})
        for key in ("tp_price", "limit_price", "stop_price", "entry_price"):
            value = meta.get(key)
            if value is not None and not pd.isna(value):
                signal_annotation_prices.append(float(value))
    if signal_annotation_prices:
        marker_prices.extend(signal_annotation_prices)
    preview_prices = []
    last_actual_x = float(len(chart_payload["x"]) - 1)
    if requested_end_idx >= last_actual_x + 0.25:
        preview = dict(chart_payload.get("future_preview") or {})
        for key in ("tp_half_price", "limit_price", "entry_price", "stop_price"):
            value = preview.get(key)
            if value is not None and not pd.isna(value):
                preview_prices.append(float(value))
    if marker_prices:
        candidate_price_arrays.append(np.asarray(marker_prices, dtype=np.float32))
    if preview_prices:
        candidate_price_arrays.append(np.asarray(preview_prices, dtype=np.float32))
    finite_prices = []
    for values in candidate_price_arrays:
        finite_values = values[np.isfinite(values)]
        if finite_values.size > 0:
            finite_prices.append(finite_values)
    if finite_prices:
        price_min = float(min(np.min(values) for values in finite_prices))
        price_max = float(max(np.max(values) for values in finite_prices))
    else:
        price_min = 0.0
        price_max = 1.0
    if price_max <= price_min:
        price_padding = max(abs(price_max) * float(price_padding_ratio), 1.0)
    else:
        price_padding = (price_max - price_min) * float(price_padding_ratio)
    visible_volume = _slice_visible_window(chart_payload["volume"], start_idx, end_idx)
    finite_volume = visible_volume[np.isfinite(visible_volume)]
    volume_max = float(np.max(finite_volume)) if finite_volume.size > 0 else 1.0
    volume_padding = max(volume_max * float(volume_padding_ratio), 1.0)
    return {
        "price_min": float(price_min - price_padding),
        "price_max": float(price_max + price_padding),
        "volume_min": 0.0,
        "volume_max": float(volume_max + volume_padding),
    }


def _clamp_chart_xlim(left, right, *, total_points, min_visible_bars=MATPLOTLIB_MIN_VISIBLE_BARS):
    if total_points <= 0:
        return -0.5, 0.5
    min_left = -0.5
    min_width = max(float(min_visible_bars), 4.0)
    width = max(float(right) - float(left), min_width)
    extra_right_padding = max(float(CHART_RIGHT_PADDING_BARS), width * float(CHART_RIGHT_PADDING_RATIO))
    max_right = float(total_points) - 0.5 + extra_right_padding
    full_width = max_right - min_left
    width = min(width, full_width)
    center = (float(left) + float(right)) / 2.0
    clamped_left = center - width / 2.0
    clamped_right = center + width / 2.0
    if clamped_left < min_left:
        shift = min_left - clamped_left
        clamped_left += shift
        clamped_right += shift
    if clamped_right > max_right:
        shift = clamped_right - max_right
        clamped_left -= shift
        clamped_right -= shift
    clamped_left = max(min_left, clamped_left)
    clamped_right = min(max_right, clamped_right)
    if clamped_right - clamped_left < min_width and full_width >= min_width:
        clamped_right = min(max_right, clamped_left + min_width)
        clamped_left = max(min_left, clamped_right - min_width)
    return float(clamped_left), float(clamped_right)


def _resolve_event_data_x(axis, event):
    if getattr(event, "xdata", None) is not None and np.isfinite(event.xdata):
        return float(event.xdata)
    if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
        return None
    transformed = axis.transData.inverted().transform((event.x, event.y))
    if transformed is None:
        return None
    x_value = float(transformed[0])
    return x_value if np.isfinite(x_value) else None


def _resolve_matplotlib_font_family():
    global MATPLOTLIB_FONT_MANAGER_IMPORT_ERROR
    try:
        from matplotlib import font_manager
    except ImportError as exc:
        MATPLOTLIB_FONT_MANAGER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    MATPLOTLIB_FONT_MANAGER_IMPORT_ERROR = ""
    available_font_names = {entry.name for entry in font_manager.fontManager.ttflist}
    for font_name in MATPLOTLIB_CJK_FONT_CANDIDATES:
        if font_name in available_font_names:
            return font_name
    return None


def _apply_axis_text_font(axis, font_properties):
    if font_properties is None:
        return
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontproperties(font_properties)


def _resolve_index_signal_annotation_meta(chart_payload, idx, *, signal_type=None):
    signal_type_normalized = None if signal_type is None else str(signal_type).strip().lower()
    for item in reversed(list(chart_payload.get("signal_annotations") or [])):
        if int(item.get("x", -1)) != int(idx):
            continue
        current_signal_type = str(item.get("signal_type", "")).strip().lower()
        if signal_type_normalized is not None and current_signal_type != signal_type_normalized:
            continue
        return dict(item.get("meta") or {})
    return {}



def _resolve_index_marker_meta(chart_payload, idx, *, trace_names):
    marker_groups = dict(chart_payload.get("marker_groups") or {})
    for trace_name in trace_names:
        markers = marker_groups.get(trace_name) or []
        for marker in reversed(list(markers)):
            if int(marker.get("x", -1)) != int(idx):
                continue
            return dict(marker.get("meta") or {}), marker
    return {}, None



def build_chart_hover_snapshot(chart_payload, index):
    idx = int(np.clip(int(index), 0, len(chart_payload["x"]) - 1))
    line_values = {
        "limit_price": float(chart_payload["limit_line"][idx]) if np.isfinite(chart_payload["limit_line"][idx]) else None,
        "entry_price": float(chart_payload["entry_line"][idx]) if np.isfinite(chart_payload["entry_line"][idx]) else None,
        "stop_price": float(chart_payload["stop_line"][idx]) if np.isfinite(chart_payload["stop_line"][idx]) else None,
        "tp_price": float(chart_payload["tp_line"][idx]) if np.isfinite(chart_payload["tp_line"][idx]) else None,
    }
    buy_signal_meta = _resolve_index_signal_annotation_meta(chart_payload, idx, signal_type="buy")
    buy_trade_meta, buy_trade_marker = _resolve_index_marker_meta(chart_payload, idx, trace_names=("買進", "買進(延續候選)", "錯失買進(新訊號)", "錯失買進(延續候選)"))
    reserved_capital = buy_signal_meta.get("reserved_capital")
    if reserved_capital is None:
        reserved_capital = buy_trade_meta.get("reserved_capital")
    return {
        "index": idx,
        "date_label": chart_payload["date_labels"][idx],
        "open": float(chart_payload["open"][idx]),
        "high": float(chart_payload["high"][idx]),
        "low": float(chart_payload["low"][idx]),
        "close": float(chart_payload["close"][idx]),
        "volume": float(chart_payload["volume"][idx]),
        "signal_capital": buy_signal_meta.get("current_capital"),
        "signal_qty": buy_signal_meta.get("qty"),
        "reserved_capital": reserved_capital,
        "buy_capital": buy_trade_meta.get("buy_capital"),
        "buy_qty": None if buy_trade_marker is None else int(buy_trade_marker.get("qty", 0) or 0),
        **line_values,
    }





def extract_trade_marker_indexes(chart_payload, *, trace_names=None):
    default_trace_names = {"買進", "買進(延續候選)", "錯失買進(新訊號)", "錯失買進(延續候選)", "半倉停利", "停損殺出", "指標賣出", "期末強制結算", "錯失賣出"}
    allowed_trace_names = default_trace_names if trace_names is None else {str(name) for name in trace_names}
    marker_groups = dict((chart_payload or {}).get("marker_groups") or {})
    indexes = []
    for trace_name, markers in marker_groups.items():
        if str(trace_name) not in allowed_trace_names:
            continue
        for marker in markers or []:
            try:
                indexes.append(int(marker.get("x")))
            except (TypeError, ValueError):
                continue
    return sorted(set(indexes))


def _build_hover_snapshot(chart_payload, index):
    return build_chart_hover_snapshot(chart_payload, index)


def _build_hover_text(chart_payload, index):
    snapshot = _build_hover_snapshot(chart_payload, index)
    parts = [
        snapshot["date_label"],
        f"開 {snapshot['open']:.2f}",
        f"高 {snapshot['high']:.2f}",
        f"低 {snapshot['low']:.2f}",
        f"收 {snapshot['close']:.2f}",
        f"量 {snapshot['volume'] / 1_000_000:.2f}M",
    ]
    return "   ".join(parts)


def _extract_signed_percent(text):
    if not text:
        return None
    match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', str(text))
    return float(match.group(1)) if match else None


def _resolve_signal_annotation_face(item):
    is_buy = item.get("signal_type") == "buy"
    if is_buy:
        return MATPLOTLIB_SIGNAL_BUY_FACE, MATPLOTLIB_SIGNAL_BUY_COLOR
    profit_pct = item.get("meta", {}).get("profit_pct")
    if profit_pct is None:
        profit_pct = _extract_signed_percent(item.get("detail_text", ""))
    if profit_pct is not None and float(profit_pct) >= 0.0:
        return MATPLOTLIB_SIGNAL_SELL_PROFIT_FACE, MATPLOTLIB_SIGNAL_SELL_COLOR
    return MATPLOTLIB_SIGNAL_SELL_LOSS_FACE, MATPLOTLIB_DOWN_COLOR


def _resolve_trade_box_style(trace_name, marker):
    meta = marker.get("meta") or {}
    if trace_name in {"買進", "買進(延續候選)"}:
        return MATPLOTLIB_BUY_FILL_FACE, MATPLOTLIB_LIMIT_COLOR, "below"
    if trace_name in {"錯失買進(新訊號)", "錯失買進(延續候選)"}:
        return MATPLOTLIB_INFO_BOX_FACE, MATPLOTLIB_LIMIT_COLOR, "below"
    if trace_name == "錯失賣出":
        return MATPLOTLIB_INFO_BOX_FACE, MATPLOTLIB_DOWN_COLOR, "above"
    if trace_name in {"停損殺出", "指標賣出", "期末強制結算"}:
        pnl_pct = meta.get("pnl_pct")
        if pnl_pct is None:
            pnl_pct = _extract_signed_percent(marker.get("note", ""))
        face = MATPLOTLIB_SELL_PROFIT_FACE if pnl_pct is not None and float(pnl_pct) >= 0.0 else MATPLOTLIB_SELL_LOSS_FACE
        color = MATPLOTLIB_UP_COLOR if pnl_pct is not None and float(pnl_pct) >= 0.0 else MATPLOTLIB_DOWN_COLOR
        return face, color, "below"
    if trace_name == "半倉停利":
        return MATPLOTLIB_SIGNAL_SELL_PROFIT_FACE, MATPLOTLIB_TP_COLOR, "above"
    return MATPLOTLIB_INFO_BOX_FACE, MATPLOTLIB_TEXT_COLOR, "above"


def _is_missing_info_value(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _first_present_info_value(meta, keys, *, marker=None):
    for key in keys:
        if key == "__marker_qty__" and marker is not None:
            value = marker.get("qty")
        elif key == "__marker_price__" and marker is not None:
            value = marker.get("price")
        else:
            value = meta.get(key)
        if not _is_missing_info_value(value):
            return value
    return None


def _format_chart_amount(value, *, signed=False):
    if _is_missing_info_value(value):
        return "-"
    sign_prefix = "+" if signed else ""
    return f"{float(value):{sign_prefix},.0f}"


def _format_chart_price(value):
    if _is_missing_info_value(value):
        return "-"
    return f"{float(value):.2f}"


def _format_chart_qty(value):
    if _is_missing_info_value(value):
        return "-"
    qty = int(value)
    if qty <= 0:
        return "-"
    return f"{qty:,}"


def _format_chart_pct(value, *, signed=False, digits=2):
    if _is_missing_info_value(value):
        return "-"
    sign_prefix = "+" if signed else ""
    return f"{float(value):{sign_prefix}.{int(digits)}f}%"


def _format_chart_entry_type(value):
    if _is_missing_info_value(value):
        return "-"
    normalized = str(value).strip().lower()
    if normalized in {"extended", "延續", "extended_candidate"}:
        return "延續"
    if normalized in {"normal", "正常"}:
        return "正常"
    return str(value).strip() or "-"


def _format_chart_trade_sequence(value):
    if _is_missing_info_value(value):
        return "-"
    return f"第 {int(value)} 次"


CHART_INFO_FIELD_SPECS = {
    "capital": {"label": "資金", "keys": ("current_capital", "capital"), "format": "amount"},
    "qty": {"label": "股數", "keys": ("qty", "__marker_qty__"), "format": "qty"},
    "limit_price": {"label": "限價", "keys": ("limit_price",), "format": "price"},
    "reserved_capital": {"label": "預留", "keys": ("reserved_capital",), "format": "amount"},
    "tp_price": {"label": "停利", "keys": ("tp_price", "target_price"), "format": "price"},
    "entry_price": {"label": "成交", "keys": ("entry_price", "sell_price", "exec_price", "__marker_price__"), "format": "price"},
    "stop_price": {"label": "停損", "keys": ("stop_price",), "format": "price"},
    "buy_capital": {"label": "實支", "keys": ("buy_capital",), "format": "amount"},
    "entry_type": {"label": "進場類型", "keys": ("entry_type",), "format": "entry_type"},
    "result": {"label": "結果", "keys": ("result",), "format": "text"},
    "reference_close": {"label": "參考收", "keys": ("reference_price", "reference_close", "close_price"), "format": "price"},
    "sell_capital": {"label": "金額", "keys": ("sell_capital",), "format": "amount"},
    "pnl": {"label": "損益", "keys": ("total_pnl", "pnl_value"), "format": "signed_amount"},
    "pnl_pct": {"label": "報酬率", "keys": ("pnl_pct",), "format": "signed_pct"},
    "win_rate": {"label": "勝率", "keys": ("win_rate",), "format": "pct1"},
    "max_drawdown": {"label": "最大回撤", "keys": ("max_drawdown",), "format": "pct2"},
    "trade_sequence": {"label": "交易次數", "keys": ("trade_sequence", "trade_count"), "format": "trade_sequence"},
}

CHART_INFO_BOX_SCHEMAS = {
    "買訊": ("capital", "qty", "limit_price", "reserved_capital"),
    "買進": ("capital", "qty", "tp_price", "limit_price", "entry_price", "stop_price", "buy_capital", "entry_type", "result"),
    "賣訊": ("capital", "qty", "reference_close"),
    "停利": ("capital", "qty", "entry_price", "sell_capital", "pnl", "pnl_pct"),
    "停損": ("capital", "qty", "entry_price", "sell_capital", "pnl", "pnl_pct", "win_rate", "max_drawdown", "trade_sequence", "result"),
    "指標賣出": ("capital", "qty", "entry_price", "sell_capital", "pnl", "pnl_pct", "win_rate", "max_drawdown", "trade_sequence", "result"),
    "期末結算": ("capital", "qty", "entry_price", "sell_capital", "pnl", "pnl_pct", "win_rate", "max_drawdown", "trade_sequence"),
}


def _normalize_chart_info_title(title):
    normalized = str(title or "").strip()
    if normalized.startswith("買訊"):
        return "買訊"
    if normalized.startswith("賣訊"):
        return "賣訊"
    if normalized in {"停損殺出", "停損"}:
        return "停損"
    if normalized in {"半倉停利", "停利"}:
        return "停利"
    if normalized in {"指標賣出", "賣出"}:
        return "指標賣出"
    if normalized in {"期末強制結算", "期未結算", "期末結算", "結算"}:
        return "期末結算"
    if normalized.startswith("買進"):
        return "買進"
    return normalized


def _format_chart_info_field(field_key, meta, *, marker=None):
    spec = CHART_INFO_FIELD_SPECS[field_key]
    value = _first_present_info_value(meta, spec["keys"], marker=marker)
    fmt = spec["format"]
    if fmt == "amount":
        rendered = _format_chart_amount(value)
    elif fmt == "signed_amount":
        rendered = _format_chart_amount(value, signed=True)
    elif fmt == "price":
        rendered = _format_chart_price(value)
    elif fmt == "qty":
        rendered = _format_chart_qty(value)
    elif fmt == "signed_pct":
        rendered = _format_chart_pct(value, signed=True, digits=2)
    elif fmt == "pct1":
        rendered = _format_chart_pct(value, digits=1)
    elif fmt == "pct2":
        rendered = _format_chart_pct(value, digits=2)
    elif fmt == "entry_type":
        rendered = _format_chart_entry_type(value)
    elif fmt == "trade_sequence":
        rendered = _format_chart_trade_sequence(value)
    else:
        rendered = "-" if _is_missing_info_value(value) else str(value)
    return f"{spec['label']}: {rendered}"


def _build_chart_info_box_text(title, meta, *, marker=None):
    normalized_title = _normalize_chart_info_title(title)
    schema = CHART_INFO_BOX_SCHEMAS.get(normalized_title)
    if not schema:
        return str(title or "")
    lines = [normalized_title]
    lines.extend(_format_chart_info_field(field_key, meta, marker=marker) for field_key in schema)
    return "\n".join(lines)


def _build_signal_label_detail_text(title, meta):
    normalized_title = _normalize_chart_info_title(title)
    schema = CHART_INFO_BOX_SCHEMAS.get(normalized_title)
    if normalized_title not in {"買訊", "賣訊"} or not schema:
        return ""
    return "\n".join(_format_chart_info_field(field_key, meta) for field_key in schema)


def _build_trade_label_text(trace_name, marker):
    meta = dict(marker.get("meta") or {})
    if trace_name in {"買進", "買進(延續候選)"}:
        meta.setdefault("result", "成交")
        meta.setdefault("entry_type", "extended" if trace_name == "買進(延續候選)" else "normal")
        return _build_chart_info_box_text("買進", meta, marker=marker)
    if trace_name in {"錯失買進(新訊號)", "錯失買進(延續候選)"}:
        meta.setdefault("result", "未成交")
        meta.setdefault("entry_type", "extended" if trace_name == "錯失買進(延續候選)" else "normal")
        return _build_chart_info_box_text("買進", meta, marker=marker)
    if trace_name == "錯失賣出":
        meta.setdefault("result", "賣出受阻")
        return _build_chart_info_box_text("指標賣出", meta, marker=marker)
    if trace_name == "半倉停利":
        return _build_chart_info_box_text("停利", meta, marker=marker)
    if trace_name in {"停損殺出", "指標賣出", "期末強制結算"}:
        title = "停損" if trace_name == "停損殺出" else ("指標賣出" if trace_name == "指標賣出" else "期末結算")
        if trace_name == "停損殺出":
            meta.setdefault("result", "停損")
        elif trace_name == "指標賣出":
            meta.setdefault("result", "指標賣出")
        return _build_chart_info_box_text(title, meta, marker=marker)
    return trace_name

def _build_status_chip_specs(status_box):
    status_lines = [str(line).strip() for line in (status_box.get("lines") or []) if str(line).strip()]
    chips = []
    for line in status_lines:
        if "買訊" in line or "買入訊號" in line or "候選" in line:
            is_active = all(token not in line for token in ("無", "否", "未", "不"))
            chips.append({"text": "出現買入訊號", "face": MATPLOTLIB_STATUS_CHIP_BUY_FACE if is_active else MATPLOTLIB_STATUS_CHIP_MUTED_FACE})
        elif "賣訊" in line:
            is_active = all(token not in line for token in ("無", "否", "未", "不"))
            chips.append({"text": "出現賣出訊號", "face": MATPLOTLIB_STATUS_CHIP_SELL_FACE if is_active else MATPLOTLIB_STATUS_CHIP_MUTED_FACE})
        elif "歷史績效" in line or "歷績門檻" in line:
            is_ok = any(token in line for token in ("合格", "符合"))
            chips.append({"text": "符合歷史績效", "face": MATPLOTLIB_STATUS_CHIP_GATE_FACE if is_ok else MATPLOTLIB_STATUS_CHIP_MUTED_FACE})
        else:
            chips.append({"text": line, "face": MATPLOTLIB_STATUS_CHIP_MUTED_FACE})
    return chips[:3]


def _render_status_chips(axis_price, status_box, label_font):
    artists = []
    chips = _build_status_chip_specs(status_box)
    if not chips:
        return artists
    y_cursor = 0.035
    for chip in reversed(chips):
        artist = axis_price.text(
            0.985,
            y_cursor,
            chip["text"],
            transform=axis_price.transAxes,
            ha="right",
            va="bottom",
            color=MATPLOTLIB_TEXT_COLOR,
            fontsize=MATPLOTLIB_STATUS_CHIP_FONT_SIZE if label_font is None else None,
            fontproperties=label_font,
            fontweight="bold" if label_font is None else None,
            bbox={"boxstyle": "round,pad=0.40", "fc": chip["face"], "ec": "none"},
            zorder=7,
        )
        artists.append(artist)
        y_cursor += 0.074
    return artists


def _compute_dynamic_linewidths(axis_price, figure):
    left, right = axis_price.get_xlim()
    visible_bars = max(float(right) - float(left), 1.0)
    axis_width_px = max(float(axis_price.bbox.width), 240.0)
    px_per_bar = axis_width_px / visible_bars
    body_px = min(max(px_per_bar * 0.72, MATPLOTLIB_DYNAMIC_BODY_WIDTH_RANGE[0]), MATPLOTLIB_DYNAMIC_BODY_WIDTH_RANGE[1])
    wick_px = min(max(px_per_bar * 0.18, MATPLOTLIB_DYNAMIC_WICK_WIDTH_RANGE[0]), MATPLOTLIB_DYNAMIC_WICK_WIDTH_RANGE[1])
    pt_scale = 72.0 / float(figure.dpi)
    return body_px * pt_scale, wick_px * pt_scale


def _resolve_annotation_slot_x_offset(slot_index, *, step=62):
    if slot_index <= 0:
        return 0
    magnitude = ((slot_index + 1) // 2) * step
    return magnitude if slot_index % 2 == 1 else -magnitude


def _count_nearby_annotation_slots(x_value, placement, occupied_positions, *, collision_window):
    return sum(
        1
        for item in occupied_positions
        if item.get("placement") == placement and abs(int(item.get("x", -10**9)) - int(x_value)) <= int(collision_window)
    )


def _resolve_trade_label_offsets(slot_index, *, placement, trace_name):
    if trace_name in {"買進", "買進(延續候選)"}:
        base_y = -64
        step_x = 94
        step_y = 28
    elif placement == "below":
        base_y = -82
        step_x = 82
        step_y = 24
    else:
        base_y = 18
        step_x = 74
        step_y = 22
    x_offset = _resolve_annotation_slot_x_offset(slot_index, step=step_x)
    tier = 0 if slot_index <= 0 else ((slot_index + 1) // 2)
    y_offset = base_y - (tier * step_y) if placement == "below" else base_y + (tier * step_y)
    return x_offset, y_offset


def _render_signal_annotations(axis_price, signal_annotations, label_font, *, start_idx=None, end_idx=None):
    rendered = []
    for item in signal_annotations:
        if start_idx is not None and int(item["x"]) < int(start_idx):
            continue
        if end_idx is not None and int(item["x"]) > int(end_idx):
            continue
        is_buy = item["signal_type"] == "buy"
        y_offset = -64 if is_buy else -76
        face_color, arrow_color = _resolve_signal_annotation_face(item)
        annotation_text = item["title"]
        if item.get("detail_text"):
            annotation_text = f"{annotation_text}\n{item['detail_text']}"
        rendered.append(
            axis_price.annotate(
                annotation_text,
                xy=(item["x"], item["anchor_price"]),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va="top",
                color=MATPLOTLIB_TEXT_COLOR,
                fontsize=MATPLOTLIB_SIGNAL_FONT_SIZE if label_font is None else None,
                fontproperties=label_font,
                bbox={"boxstyle": "round,pad=0.46", "fc": face_color, "ec": arrow_color},
                arrowprops={"arrowstyle": "-|>", "color": arrow_color, "lw": 1.55, "alpha": 0.96, "mutation_scale": MATPLOTLIB_SIGNAL_ARROW_MUTATION_SCALE},
                zorder=7,
                annotation_clip=True,
            )
        )
    return rendered


def _render_trade_labels(axis_price, marker_groups, label_font, *, signal_annotations=None, start_idx=None, end_idx=None):
    rendered = []
    supported_traces = {"買進", "買進(延續候選)", "錯失買進(新訊號)", "錯失買進(延續候選)", "半倉停利", "停損殺出", "指標賣出", "期末強制結算", "錯失賣出"}
    occupied_positions = []
    for item in signal_annotations or []:
        x_value = int(item.get("x", -10**9))
        if start_idx is not None and x_value < int(start_idx):
            continue
        if end_idx is not None and x_value > int(end_idx):
            continue
        placement = "below" if item.get("signal_type") == "buy" else "above"
        occupied_positions.append({"x": x_value, "placement": placement})

    trade_items = []
    for trace_name, markers in marker_groups.items():
        if trace_name not in supported_traces:
            continue
        for marker in markers:
            x_value = int(marker["x"])
            if start_idx is not None and x_value < int(start_idx):
                continue
            if end_idx is not None and x_value > int(end_idx):
                continue
            _, _, placement = _resolve_trade_box_style(trace_name, marker)
            trade_items.append((x_value, 0 if placement == "below" else 1, trace_name, marker))

    trade_items.sort(key=lambda item: (item[0], item[1], item[2]))
    for x_value, _, trace_name, marker in trade_items:
        face_color, text_color, placement = _resolve_trade_box_style(trace_name, marker)
        collision_window = 5 if placement == "below" else 4
        slot_index = _count_nearby_annotation_slots(x_value, placement, occupied_positions, collision_window=collision_window)
        x_offset, y_offset = _resolve_trade_label_offsets(slot_index, placement=placement, trace_name=trace_name)
        occupied_positions.append({"x": x_value, "placement": placement})
        va = "top" if placement == "below" else "bottom"
        rendered.append(
            axis_price.annotate(
                _build_trade_label_text(trace_name, marker),
                xy=(marker["x"], marker["price"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="center",
                va=va,
                color=MATPLOTLIB_TEXT_COLOR if placement == "below" else text_color,
                fontsize=MATPLOTLIB_SIGNAL_FONT_SIZE if label_font is None else None,
                fontproperties=label_font,
                bbox={"boxstyle": "round,pad=0.38", "fc": face_color, "ec": "none"},
                arrowprops={"arrowstyle": "-|>", "color": text_color, "lw": 1.45, "alpha": 0.94, "mutation_scale": MATPLOTLIB_SIGNAL_ARROW_MUTATION_SCALE},
                zorder=6,
                annotation_clip=True,
            )
        )
    return rendered


def _render_future_preview_lines(axis_price, chart_payload):
    preview = dict(chart_payload.get("future_preview") or {})
    if not preview:
        return []
    last_x = float(len(chart_payload["x"]) - 1)
    x_start = last_x + 0.55
    x_end = last_x + 1.55
    rendered = []
    preview_specs = (
        ("tp_half_price", MATPLOTLIB_TP_COLOR, 1.9, "solid"),
        ("limit_price", MATPLOTLIB_LIMIT_COLOR, 1.6, (0, (4, 2))),
        ("entry_price", MATPLOTLIB_ENTRY_COLOR, 1.8, "solid"),
        ("stop_price", MATPLOTLIB_STOP_COLOR, 2.0, "solid"),
    )
    for key, color, linewidth, linestyle in preview_specs:
        value = preview.get(key)
        if value is None or pd.isna(value):
            continue
        rendered.append(axis_price.hlines(float(value), x_start, x_end, colors=color, linewidth=linewidth, linestyles=linestyle, zorder=4))
    return rendered


def _build_complete_matplotlib_legend_handles():
    from matplotlib.lines import Line2D

    handles = []
    for label, color, linestyle, linewidth in CHART_LINE_LEGEND_SPECS:
        handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth, label=label))
    for label in CHART_EVENT_LEGEND_ORDER:
        style = CHART_SIGNAL_LEGEND_STYLE.get(label) or ACTION_STYLE_MAP.get(label)
        if not style:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["mpl_marker"],
                linestyle="None",
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                color=style["color"],
                markersize=8,
                label=label,
            )
        )
    return handles


def create_matplotlib_trade_chart_figure(*, chart_payload, ticker, show_volume=False):
    return create_matplotlib_debug_chart_figure(chart_payload=chart_payload, ticker=ticker, show_volume=show_volume)


def create_matplotlib_debug_chart_figure(*, chart_payload, ticker, show_volume=False):
    chart_payload = normalize_chart_payload_contract(chart_payload)
    try:
        from matplotlib.figure import Figure
        from matplotlib import rcParams
        from matplotlib import ticker as mticker
        from matplotlib.font_manager import FontProperties
    except ImportError as exc:
        raise RuntimeError("缺少 matplotlib，無法在 GUI 內嵌單股回測 K 線圖。") from exc
    x_positions = np.asarray(chart_payload["x"])
    if x_positions.size == 0:
        raise ValueError("chart_payload 不可為空。")
    font_family = _resolve_matplotlib_font_family()
    base_font = FontProperties(family=font_family) if font_family else None
    title_font = FontProperties(family=font_family, weight="bold", size=18) if font_family else None
    label_font = FontProperties(family=font_family, size=12) if font_family else None
    legend_font = FontProperties(family=font_family, size=10) if font_family else None
    rcParams["axes.unicode_minus"] = False
    if font_family:
        rcParams["font.sans-serif"] = [font_family, *MATPLOTLIB_CJK_FONT_CANDIDATES]
    figure = Figure(figsize=MATPLOTLIB_DEBUG_CHART_FIGSIZE, dpi=96, facecolor=MATPLOTLIB_DARK_BG)
    axis_price = figure.add_subplot(1, 1, 1)
    axis_volume = None
    figure.subplots_adjust(left=MATPLOTLIB_SUBPLOT_LEFT, right=MATPLOTLIB_SUBPLOT_RIGHT, top=MATPLOTLIB_SUBPLOT_TOP, bottom=MATPLOTLIB_SUBPLOT_BOTTOM)
    axis_price.set_facecolor(MATPLOTLIB_DARK_BG)
    axis_price.grid(True, color=MATPLOTLIB_GRID_COLOR, alpha=0.12, linewidth=0.68)
    axis_price.tick_params(axis="y", colors=MATPLOTLIB_TEXT_COLOR, labelsize=11, pad=6)
    axis_price.tick_params(axis="x", colors=MATPLOTLIB_TEXT_COLOR, labelsize=11)
    axis_price.spines["top"].set_visible(False)
    axis_price.spines["right"].set_visible(False)
    axis_price.spines["left"].set_color(MATPLOTLIB_GRID_COLOR)
    axis_price.spines["bottom"].set_color(MATPLOTLIB_GRID_COLOR)
    if show_volume:
        axis_volume = axis_price.inset_axes([0.0, MATPLOTLIB_VOLUME_OVERLAY_BOTTOM_GAP, 1.0, MATPLOTLIB_VOLUME_OVERLAY_HEIGHT_RATIO], sharex=axis_price)
        axis_volume.set_facecolor("none")
        axis_volume.grid(False)
        axis_volume.spines["top"].set_visible(False)
        axis_volume.spines["left"].set_visible(False)
        axis_volume.spines["bottom"].set_visible(False)
        axis_volume.spines["right"].set_color(MATPLOTLIB_GRID_COLOR)
        axis_volume.yaxis.tick_right()
        axis_volume.tick_params(axis="y", colors=MATPLOTLIB_MUTED_TEXT_COLOR, labelsize=9, pad=2)
        axis_volume.set_navigate(False)
        axis_volume.tick_params(axis="x", labelbottom=False, bottom=False)
        axis_volume.patch.set_alpha(0.0)
        axis_volume.set_zorder(1)
        axis_price.set_zorder(2)
    open_values = np.asarray(chart_payload["open"])
    close_values = np.asarray(chart_payload["close"])
    high_values = np.asarray(chart_payload["high"])
    low_values = np.asarray(chart_payload["low"])
    up_mask = np.asarray(chart_payload["up_mask"], dtype=bool)
    candle_colors = np.where(up_mask, MATPLOTLIB_UP_COLOR, MATPLOTLIB_DOWN_COLOR)
    body_low = np.minimum(open_values, close_values).astype(np.float32, copy=False)
    body_high = np.maximum(open_values, close_values).astype(np.float32, copy=False)
    median_price = float(np.nanmedian(close_values)) if np.isfinite(close_values).any() else 1.0
    min_body_height = max(median_price * 0.0008, 0.01)
    flat_mask = np.abs(close_values - open_values) <= 1e-9
    if np.any(flat_mask):
        body_low = body_low.copy()
        body_high = body_high.copy()
        body_low[flat_mask] -= float(min_body_height) / 2.0
        body_high[flat_mask] += float(min_body_height) / 2.0
    wick_collection = axis_price.vlines(x_positions, low_values, high_values, colors=candle_colors, linewidth=1.0, zorder=2)
    body_up_collection = None
    body_down_collection = None
    if np.any(up_mask):
        body_up_collection = axis_price.vlines(x_positions[up_mask], body_low[up_mask], body_high[up_mask], colors=MATPLOTLIB_UP_COLOR, linewidth=4.8, zorder=3)
    if np.any(~up_mask):
        body_down_collection = axis_price.vlines(x_positions[~up_mask], body_low[~up_mask], body_high[~up_mask], colors=MATPLOTLIB_DOWN_COLOR, linewidth=4.8, zorder=3)
    axis_price.step(x_positions, chart_payload["stop_line"], where="mid", color=MATPLOTLIB_STOP_COLOR, linewidth=2.0, zorder=4)
    if np.isfinite(chart_payload["tp_line"]).any():
        axis_price.step(x_positions, chart_payload["tp_line"], where="mid", color=MATPLOTLIB_TP_COLOR, linewidth=1.9, zorder=4)
    if np.isfinite(chart_payload["limit_line"]).any():
        axis_price.step(x_positions, chart_payload["limit_line"], where="mid", color=MATPLOTLIB_LIMIT_COLOR, linewidth=1.5, linestyle=(0, (4, 2)), zorder=4)
    if np.isfinite(chart_payload["entry_line"]).any():
        axis_price.step(x_positions, chart_payload["entry_line"], where="mid", color=MATPLOTLIB_ENTRY_COLOR, linewidth=1.8, zorder=4)
    for trace_name, markers in chart_payload["marker_groups"].items():
        style = ACTION_STYLE_MAP.get(trace_name, {"mpl_marker": "o", "color": MATPLOTLIB_TEXT_COLOR})
        axis_price.scatter(
            [item["x"] for item in markers],
            [item["price"] for item in markers],
            marker=style["mpl_marker"],
            s=MATPLOTLIB_MARKER_SIZE,
            color=style["color"],
            linewidths=1.8,
            zorder=5,
            label=trace_name if trace_name not in {"限價買進", "限價買進(延續候選)"} else "_nolegend_",
        )
    future_preview_artists = _render_future_preview_lines(axis_price, chart_payload)
    axis_price.set_ylabel("價格", color=MATPLOTLIB_TEXT_COLOR, fontsize=12 if label_font is None else None, fontproperties=label_font)
    volume_collection = None
    if show_volume and axis_volume is not None:
        volume_colors = np.where(up_mask, MATPLOTLIB_UP_COLOR, MATPLOTLIB_DOWN_COLOR)
        volume_collection = axis_volume.vlines(x_positions, 0.0, chart_payload["volume"], colors=volume_colors, linewidth=2.2, alpha=MATPLOTLIB_VOLUME_ALPHA, zorder=1)
        try:
            volume_collection.set_rasterized(True)
        except Exception as exc:
            warnings.warn(f"volume rasterization skipped: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
    date_labels = chart_payload["date_labels"]
    def _format_date_label(x_value, _pos):
        rounded = int(round(x_value))
        return date_labels[rounded] if 0 <= rounded < len(date_labels) else ""
    axis_price.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))
    axis_price.xaxis.set_major_formatter(mticker.FuncFormatter(_format_date_label))
    legend_handles = _build_complete_matplotlib_legend_handles()
    axis_price.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper left", ncol=min(8, max(1, len(legend_handles))), frameon=False, prop=legend_font, labelcolor=MATPLOTLIB_TEXT_COLOR, bbox_to_anchor=(0.012, 1.012), borderaxespad=0.0, handlelength=2.0, columnspacing=0.85)
    hover_text_artist = axis_price.text(0.01, 0.998, "", transform=axis_price.transAxes, ha="left", va="top", color=MATPLOTLIB_TEXT_COLOR, fontsize=10 if legend_font is None else None, fontproperties=legend_font, zorder=8)
    hover_text_artist.set_visible(False)
    crosshair_vline = axis_price.axvline(x=chart_payload["default_view"]["end_idx"], color=MATPLOTLIB_CROSSHAIR_COLOR, linewidth=0.8, linestyle=(0, (4, 4)), alpha=0.58, zorder=1)
    crosshair_hline = axis_price.axhline(y=chart_payload["close"][chart_payload["default_view"]["end_idx"]], color=MATPLOTLIB_CROSSHAIR_COLOR, linewidth=0.8, linestyle=(0, (4, 4)), alpha=0.35, zorder=1)
    rendered_signal_annotations = []
    rendered_trade_labels = []
    summary_lines = chart_payload.get("summary_box") or []
    summary_artist = None
    if summary_lines:
        summary_artist = axis_price.text(0.985, 0.56, "\n".join(summary_lines), transform=axis_price.transAxes, ha="right", va="center", color=MATPLOTLIB_TEXT_COLOR, fontsize=MATPLOTLIB_SUMMARY_FONT_SIZE if legend_font is None else None, fontproperties=legend_font, bbox={"boxstyle": "round,pad=0.36", "fc": MATPLOTLIB_INFO_BOX_FACE, "ec": MATPLOTLIB_GRID_COLOR, "lw": 0.9}, zorder=6)
    status_box = chart_payload.get("status_box") or {}
    status_chip_artists = _render_status_chips(axis_price, status_box, legend_font)
    default_view = chart_payload["default_view"]
    x_start, x_end = _clamp_chart_xlim(default_view["start_idx"] - 1, default_view["end_idx"] + 1, total_points=len(chart_payload["x"]))
    if axis_volume is not None:
        axis_volume.set_xlim(x_start, x_end, emit=False)
    ranges = compute_visible_value_ranges(chart_payload, start_idx=x_start, end_idx=x_end)
    axis_price.set_xlim(x_start, x_end)
    axis_price.set_ylim(ranges["price_min"], ranges["price_max"])
    if axis_volume is not None:
        axis_volume.set_ylim(ranges["volume_min"], ranges["volume_max"])
    interaction_flags = {"dragging": False}
    sync_state = {"updating": False, "last_window": None}

    def _replace_artist_list(target_list, new_items):
        for artist in target_list:
            try:
                artist.remove()
            except ValueError as exc:
                _warn_chart_runtime_fallback_once("artist_remove_value_error", exc, context="artist remove skipped")
        target_list[:] = list(new_items)

    def _refresh_overlay_annotations(start_idx, end_idx):
        render_start = max(0, int(start_idx) - 2)
        render_end = min(len(chart_payload["x"]) - 1, int(end_idx) + 2)
        visible_signal_annotations = chart_payload.get("signal_annotations", [])
        _replace_artist_list(rendered_signal_annotations, _render_signal_annotations(axis_price, visible_signal_annotations, legend_font, start_idx=render_start, end_idx=render_end))
        _replace_artist_list(rendered_trade_labels, _render_trade_labels(axis_price, chart_payload["marker_groups"], legend_font, signal_annotations=visible_signal_annotations, start_idx=render_start, end_idx=render_end))

    def _refresh_candle_widths():
        body_width, wick_width = _compute_dynamic_linewidths(axis_price, figure)
        wick_collection.set_linewidth(wick_width)
        if body_up_collection is not None:
            body_up_collection.set_linewidth(body_width)
        if body_down_collection is not None:
            body_down_collection.set_linewidth(body_width)
        if volume_collection is not None:
            volume_collection.set_linewidth(body_width * MATPLOTLIB_VOLUME_WIDTH_SCALE)

    def _sync_visible_ranges(*, force=False, redraw=True):
        if sync_state["updating"]:
            return
        if interaction_flags["dragging"] and not force:
            _refresh_candle_widths()
            return
        sync_state["updating"] = True
        try:
            left, right = axis_price.get_xlim()
            clamped_left, clamped_right = _clamp_chart_xlim(left, right, total_points=len(chart_payload["x"]))
            if abs(clamped_left - left) > 1e-9 or abs(clamped_right - right) > 1e-9:
                axis_price.set_xlim(clamped_left, clamped_right, emit=False)
                if axis_volume is not None:
                    axis_volume.set_xlim(clamped_left, clamped_right, emit=False)
            visible_window = (int(np.floor(clamped_left)), int(np.ceil(clamped_right)))
            if force or sync_state["last_window"] != visible_window:
                visible_ranges = compute_visible_value_ranges(chart_payload, start_idx=clamped_left, end_idx=clamped_right)
                axis_price.set_ylim(visible_ranges["price_min"], visible_ranges["price_max"])
                if axis_volume is not None:
                    axis_volume.set_ylim(visible_ranges["volume_min"], visible_ranges["volume_max"])
                _refresh_overlay_annotations(*visible_window)
                sync_state["last_window"] = visible_window
            _refresh_candle_widths()
            if redraw and figure.canvas is not None:
                figure.canvas.draw_idle()
        finally:
            sync_state["updating"] = False

    initial_visible_window = (int(np.floor(x_start)), int(np.ceil(x_end)))
    _refresh_overlay_annotations(*initial_visible_window)
    _refresh_candle_widths()
    sync_state["last_window"] = initial_visible_window
    axis_price.callbacks.connect("xlim_changed", lambda _axis: _sync_visible_ranges(force=False, redraw=True))
    _apply_axis_text_font(axis_price, base_font)
    if axis_volume is not None:
        _apply_axis_text_font(axis_volume, base_font)
    figure._stock_chart_contract = {
        "volume_visible": bool(show_volume),
        "volume_overlay_mode": "inset" if show_volume else "hidden",
        "volume_overlay_axis_present": bool(axis_volume is not None),
        "selected_font_family": font_family or "",
        "render_start_idx": 0,
        "render_end_idx": int(len(chart_payload["x"]) - 1),
        "render_bar_count": int(len(chart_payload["x"])),
        "total_bar_count": int(len(chart_payload["x"])),
        "default_view": chart_payload["default_view"],
        "full_history_navigation_enabled": True,
        "default_lookback_months": int(CHART_DEFAULT_LOOKBACK_MONTHS),
        "volume_overlay_ratio": float(MATPLOTLIB_VOLUME_OVERLAY_HEIGHT_RATIO),
        "mouse_wheel_zoom_enabled": False,
        "mouse_left_drag_pan_enabled": False,
        "keyboard_pan_enabled": False,
        "toolbar_required": False,
        "hover_value_display_enabled": True,
        "twse_up_color": MATPLOTLIB_UP_COLOR,
        "twse_down_color": MATPLOTLIB_DOWN_COLOR,
        "summary_box_present": bool(summary_lines),
        "status_box_present": bool(status_box.get("lines")),
        "status_chip_layout": "right_bottom",
        "signal_annotation_count": int(len(chart_payload.get("signal_annotations", []))),
        "dynamic_candle_width_enabled": True,
        "buy_trade_label_boxes_enabled": True,
        "mouse_drag_pan_mode": "pixel_anchor",
        "grid_alpha": 0.12,
    }
    figure._stock_chart_navigation_state = {
        "axis_price": axis_price,
        "axis_volume": axis_volume,
        "chart_payload": chart_payload,
        "total_points": int(len(chart_payload["x"])),
        "sync_visible_ranges": _sync_visible_ranges,
        "connection_ids": {},
        "hover_text_artist": hover_text_artist,
        "crosshair_vline": crosshair_vline,
        "crosshair_hline": crosshair_hline,
        "hover_last_index": int(chart_payload["default_view"]["end_idx"]),
        "summary_artist": summary_artist,
        "status_chip_artists": status_chip_artists,
        "signal_artists": rendered_signal_annotations,
        "trade_label_artists": rendered_trade_labels,
        "future_preview_artists": future_preview_artists,
        "external_hover_callback": None,
        "body_up_collection": body_up_collection,
        "body_down_collection": body_down_collection,
        "wick_collection": wick_collection,
        "volume_collection": volume_collection,
        "interaction_flags": interaction_flags,
    }
    return figure


def scroll_chart_to_latest(figure, *, redraw=True):
    state = getattr(figure, "_stock_chart_navigation_state", None)
    if state is None:
        return False
    axis_price = state["axis_price"]
    chart_payload = state["chart_payload"]
    total_points = int(state["total_points"])
    default_view = chart_payload.get("default_view") or {"start_idx": 0, "end_idx": max(0, total_points - 1)}
    window_width = max(float(default_view.get("end_idx", total_points - 1)) - float(default_view.get("start_idx", 0)) + 2.0, float(MATPLOTLIB_MIN_VISIBLE_BARS))
    target_right = float(total_points) - 0.5 + max(float(CHART_RIGHT_PADDING_BARS), window_width * float(CHART_RIGHT_PADDING_RATIO))
    target_left = target_right - window_width
    next_left, next_right = _clamp_chart_xlim(target_left, target_right, total_points=total_points)
    axis_price.set_xlim(next_left, next_right, emit=False)
    axis_volume = state.get("axis_volume")
    if axis_volume is not None:
        axis_volume.set_xlim(next_left, next_right, emit=False)
    sync_visible_ranges = state.get("sync_visible_ranges")
    if callable(sync_visible_ranges):
        sync_visible_ranges(force=True, redraw=redraw)
    hover_text_artist = state.get("hover_text_artist")
    crosshair_vline = state.get("crosshair_vline")
    crosshair_hline = state.get("crosshair_hline")
    latest_index = max(0, total_points - 1)
    state["hover_last_index"] = latest_index
    if hover_text_artist is not None:
        hover_text_artist.set_text(_build_hover_text(chart_payload, latest_index))
    if crosshair_vline is not None:
        crosshair_vline.set_xdata([latest_index, latest_index])
    if crosshair_hline is not None:
        close_price = float(chart_payload["close"][latest_index])
        crosshair_hline.set_ydata([close_price, close_price])
    external_hover_callback = state.get("external_hover_callback")
    if callable(external_hover_callback):
        external_hover_callback(_build_hover_snapshot(chart_payload, latest_index))
    if redraw and figure.canvas is not None:
        figure.canvas.draw_idle()
    return True


def scroll_chart_to_index(figure, target_index, *, redraw=True):
    state = getattr(figure, "_stock_chart_navigation_state", None)
    if state is None:
        return False
    axis_price = state["axis_price"]
    chart_payload = state["chart_payload"]
    total_points = int(state["total_points"])
    if total_points <= 0:
        return False

    target_index = int(np.clip(int(target_index), 0, total_points - 1))
    left, right = axis_price.get_xlim()
    current_width = max(float(right) - float(left), float(MATPLOTLIB_MIN_VISIBLE_BARS))
    default_view = chart_payload.get("default_view") or {"start_idx": 0, "end_idx": max(0, total_points - 1)}
    default_width = float(default_view.get("end_idx", total_points - 1)) - float(default_view.get("start_idx", 0)) + 2.0
    window_width = max(
        min(current_width, min(default_width, float(total_points))),
        float(CHART_MIN_WINDOW_BARS),
        float(MATPLOTLIB_MIN_VISIBLE_BARS),
    )
    target_left = float(target_index) - window_width * 0.46
    target_right = target_left + window_width
    next_left, next_right = _clamp_chart_xlim(target_left, target_right, total_points=total_points)

    axis_price.set_xlim(next_left, next_right, emit=False)
    axis_volume = state.get("axis_volume")
    if axis_volume is not None:
        axis_volume.set_xlim(next_left, next_right, emit=False)
    sync_visible_ranges = state.get("sync_visible_ranges")
    if callable(sync_visible_ranges):
        sync_visible_ranges(force=True, redraw=False)

    hover_text_artist = state.get("hover_text_artist")
    crosshair_vline = state.get("crosshair_vline")
    crosshair_hline = state.get("crosshair_hline")
    state["hover_last_index"] = target_index
    if hover_text_artist is not None:
        hover_text_artist.set_text(_build_hover_text(chart_payload, target_index))
    if crosshair_vline is not None:
        crosshair_vline.set_xdata([target_index, target_index])
    if crosshair_hline is not None:
        close_price = float(chart_payload["close"][target_index])
        crosshair_hline.set_ydata([close_price, close_price])
    external_hover_callback = state.get("external_hover_callback")
    if callable(external_hover_callback):
        external_hover_callback(_build_hover_snapshot(chart_payload, target_index))
    if redraw and figure.canvas is not None:
        figure.canvas.draw()
        try:
            figure.canvas.flush_events()
        except NotImplementedError as exc:
            _warn_chart_runtime_fallback_once("canvas_flush_events_not_implemented", exc, context="chart redraw flush skipped")
    return True


def resolve_adjacent_trade_index(current_index, trade_indexes, *, direction):
    normalized_indexes = sorted({int(idx) for idx in trade_indexes if idx is not None})
    if not normalized_indexes:
        return None

    try:
        cursor = int(current_index)
    except (TypeError, ValueError):
        cursor = normalized_indexes[-1]

    if int(direction) < 0:
        candidates = [idx for idx in normalized_indexes if idx < cursor]
        return candidates[-1] if candidates else normalized_indexes[-1]

    candidates = [idx for idx in normalized_indexes if idx > cursor]
    return candidates[0] if candidates else normalized_indexes[0]


def scroll_chart_to_adjacent_trade(figure, trade_indexes, *, direction, redraw=True):
    if figure is None:
        return False
    state = getattr(figure, "_stock_chart_navigation_state", None)
    current_index = None
    if isinstance(state, dict):
        current_index = state.get("hover_last_index")
    target_index = resolve_adjacent_trade_index(current_index, trade_indexes, direction=direction)
    if target_index is None:
        return False
    return scroll_chart_to_index(figure, target_index, redraw=redraw)


def bind_matplotlib_chart_navigation(figure, canvas):
    state = getattr(figure, "_stock_chart_navigation_state", None)
    if state is None:
        return {}
    existing = state.get("connection_ids")
    if existing:
        return existing
    axis_price = state["axis_price"]
    axis_volume = state.get("axis_volume")
    chart_payload = state["chart_payload"]
    total_points = int(state["total_points"])
    hover_text_artist = state["hover_text_artist"]
    crosshair_vline = state["crosshair_vline"]
    crosshair_hline = state["crosshair_hline"]
    sync_visible_ranges = state["sync_visible_ranges"]
    drag_state = {"active": False, "anchor_x": None, "anchor_px": None, "orig_xlim": None, "last_draw_px": None, "last_draw_at": 0.0}
    interaction_flags = state.get("interaction_flags") or {"dragging": False}
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.configure(cursor="", highlightthickness=0, bd=0, takefocus=1, background=MATPLOTLIB_DARK_BG)

    def _allowed_axis(event):
        return event.inaxes in {axis_price, axis_volume}

    def _set_hover_index(index, *, redraw=True):
        nearest_idx = int(np.clip(index, 0, total_points - 1))
        state["hover_last_index"] = nearest_idx
        hover_text_artist.set_text(_build_hover_text(chart_payload, nearest_idx))
        crosshair_vline.set_xdata([nearest_idx, nearest_idx])
        close_price = float(chart_payload["close"][nearest_idx])
        crosshair_hline.set_ydata([close_price, close_price])
        external_hover_callback = state.get("external_hover_callback")
        if callable(external_hover_callback):
            external_hover_callback(_build_hover_snapshot(chart_payload, nearest_idx))
        if redraw and figure.canvas is not None:
            figure.canvas.draw_idle()

    def _update_hover(event):
        if not _allowed_axis(event):
            return
        data_x = _resolve_event_data_x(axis_price, event)
        if data_x is None:
            return
        _set_hover_index(int(round(data_x)), redraw=True)

    def _pan_visible_window(delta_bars, *, relayout=False):
        left, right = axis_price.get_xlim()
        next_left, next_right = _clamp_chart_xlim(left + float(delta_bars), right + float(delta_bars), total_points=total_points)
        axis_price.set_xlim(next_left, next_right, emit=False)
        axis_volume = state.get("axis_volume")
        if axis_volume is not None:
            axis_volume.set_xlim(next_left, next_right, emit=False)
        if relayout:
            sync_visible_ranges(force=True, redraw=True)
        elif figure.canvas is not None:
            figure.canvas.draw_idle()

    def _ensure_hover_visible(index):
        left, right = axis_price.get_xlim()
        if index >= int(np.floor(right)) - MATPLOTLIB_KEY_EDGE_MARGIN_BARS:
            _pan_visible_window(index - (int(np.floor(right)) - MATPLOTLIB_KEY_EDGE_MARGIN_BARS), relayout=True)
        elif index <= int(np.ceil(left)) + MATPLOTLIB_KEY_EDGE_MARGIN_BARS:
            _pan_visible_window(index - (int(np.ceil(left)) + MATPLOTLIB_KEY_EDGE_MARGIN_BARS), relayout=True)

    def _on_press(event):
        if event.button != 1 or not _allowed_axis(event):
            return
        anchor_x = _resolve_event_data_x(axis_price, event)
        if anchor_x is None:
            return
        drag_state["active"] = True
        interaction_flags["dragging"] = True
        drag_state["anchor_x"] = anchor_x
        drag_state["anchor_px"] = float(getattr(event, "x", 0.0))
        drag_state["orig_xlim"] = axis_price.get_xlim()
        drag_state["last_draw_px"] = drag_state["anchor_px"]
        drag_state["last_draw_at"] = 0.0
        canvas_widget.focus_set()
        canvas_widget.configure(cursor=MATPLOTLIB_PAN_CURSOR)

    def _on_motion(event):
        if drag_state["active"]:
            current_px = getattr(event, "x", None)
            if current_px is None:
                return
            origin_left, origin_right = drag_state["orig_xlim"]
            axis_width_px = max(float(axis_price.bbox.width), 1.0)
            bars_per_pixel = (float(origin_right) - float(origin_left)) / axis_width_px
            delta = (float(drag_state["anchor_px"]) - float(current_px)) * bars_per_pixel
            next_left, next_right = _clamp_chart_xlim(origin_left + delta, origin_right + delta, total_points=total_points)
            axis_price.set_xlim(next_left, next_right, emit=False)
            if axis_volume is not None:
                axis_volume.set_xlim(next_left, next_right, emit=False)
            if figure.canvas is not None:
                now_ts = time.monotonic()
                last_px = drag_state.get("last_draw_px")
                if last_px is None or abs(float(current_px) - float(last_px)) >= MATPLOTLIB_PAN_REDRAW_MIN_PIXEL_DELTA or (now_ts - float(drag_state.get("last_draw_at", 0.0))) >= MATPLOTLIB_PAN_REDRAW_MIN_INTERVAL_SEC:
                    figure.canvas.draw_idle()
                    drag_state["last_draw_px"] = float(current_px)
                    drag_state["last_draw_at"] = now_ts
            return
        _update_hover(event)

    def _on_release(event):
        if event.button == 1 and drag_state["active"]:
            drag_state["active"] = False
            interaction_flags["dragging"] = False
            drag_state["anchor_x"] = None
            drag_state["anchor_px"] = None
            drag_state["orig_xlim"] = None
            drag_state["last_draw_px"] = None
            canvas_widget.configure(cursor="")
            sync_visible_ranges(force=True, redraw=True)
            _update_hover(event)

    def _on_scroll(event):
        if not _allowed_axis(event):
            return
        left, right = axis_price.get_xlim()
        current_width = max(float(right) - float(left), 1.0)
        zoom_in = getattr(event, "button", None) == "up" or getattr(event, "step", 0) > 0
        zoom_factor = MATPLOTLIB_WHEEL_ZOOM_IN_FACTOR if zoom_in else MATPLOTLIB_WHEEL_ZOOM_OUT_FACTOR
        new_width = current_width * zoom_factor
        new_width = min(max(new_width, float(MATPLOTLIB_MIN_VISIBLE_BARS)), max(float(total_points), float(MATPLOTLIB_MIN_VISIBLE_BARS)))
        focus_x = _resolve_event_data_x(axis_price, event)
        if focus_x is None:
            focus_x = (left + right) / 2.0
        focus_ratio = (focus_x - left) / current_width if current_width > 0 else 0.5
        focus_ratio = min(max(focus_ratio, 0.0), 1.0)
        next_left = focus_x - new_width * focus_ratio
        next_right = next_left + new_width
        next_left, next_right = _clamp_chart_xlim(next_left, next_right, total_points=total_points)
        axis_price.set_xlim(next_left, next_right, emit=False)
        if axis_volume is not None:
            axis_volume.set_xlim(next_left, next_right, emit=False)
        sync_visible_ranges(force=True, redraw=True)
        _set_hover_index(int(round(focus_x)), redraw=True)

    def _on_key_press(event):
        key = getattr(event, "key", None)
        if key not in {"left", "right"}:
            return
        current_index = state.get("hover_last_index")
        if current_index is None:
            current_index = int(round(sum(axis_price.get_xlim()) / 2.0))
        next_index = max(0, current_index - 1) if key == "left" else min(total_points - 1, current_index + 1)
        _set_hover_index(next_index, redraw=False)
        _ensure_hover_visible(next_index)
        if figure.canvas is not None:
            figure.canvas.draw_idle()

    def _on_leave(_event):
        if not drag_state["active"]:
            canvas_widget.configure(cursor="")

    connection_ids = {
        "button_press_event": canvas.mpl_connect("button_press_event", _on_press),
        "motion_notify_event": canvas.mpl_connect("motion_notify_event", _on_motion),
        "button_release_event": canvas.mpl_connect("button_release_event", _on_release),
        "scroll_event": canvas.mpl_connect("scroll_event", _on_scroll),
        "figure_leave_event": canvas.mpl_connect("figure_leave_event", _on_leave),
        "key_press_event": canvas.mpl_connect("key_press_event", _on_key_press),
    }
    state["connection_ids"] = connection_ids
    figure._stock_chart_contract["mouse_wheel_zoom_enabled"] = True
    figure._stock_chart_contract["mouse_left_drag_pan_enabled"] = True
    figure._stock_chart_contract["keyboard_pan_enabled"] = True
    figure._stock_chart_contract["toolbar_required"] = False
    return connection_ids


def export_debug_chart_html(price_df, *, ticker, output_dir, chart_context, chart_payload=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("缺少 plotly，無法輸出單股回測 K 線圖。") from exc
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"Debug_TradeChart_{ticker}.html")
    if chart_payload is None:
        chart_payload = build_debug_chart_payload(price_df, chart_context)
    chart_payload = normalize_chart_payload_contract(chart_payload)
    dates = chart_payload["dates"]
    x_start_idx = chart_payload["default_view"]["start_idx"]
    x_end_idx = chart_payload["default_view"]["end_idx"]
    default_ranges = compute_visible_value_ranges(chart_payload, start_idx=x_start_idx, end_idx=x_end_idx)
    volume_colors = np.where(chart_payload["up_mask"], MATPLOTLIB_UP_COLOR, MATPLOTLIB_DOWN_COLOR)
    font_family = _resolve_matplotlib_font_family() or "Microsoft JhengHei, Noto Sans CJK TC, sans-serif"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.82, 0.18])
    fig.add_trace(go.Candlestick(x=dates, open=chart_payload["open"], high=chart_payload["high"], low=chart_payload["low"], close=chart_payload["close"], name="K線", increasing_line_color=MATPLOTLIB_UP_COLOR, increasing_fillcolor=MATPLOTLIB_UP_COLOR, decreasing_line_color=MATPLOTLIB_DOWN_COLOR, decreasing_fillcolor=MATPLOTLIB_DOWN_COLOR), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=chart_payload["stop_line"], mode="lines", name="停損線", line={"color": MATPLOTLIB_STOP_COLOR, "width": 2}, line_shape="hv", connectgaps=False), row=1, col=1)
    if np.isfinite(chart_payload["tp_line"]).any():
        fig.add_trace(go.Scatter(x=dates, y=chart_payload["tp_line"], mode="lines", name="半倉停利線", line={"color": MATPLOTLIB_TP_COLOR, "width": 2}, line_shape="hv", connectgaps=False), row=1, col=1)
    if np.isfinite(chart_payload["limit_line"]).any():
        fig.add_trace(go.Scatter(x=dates, y=chart_payload["limit_line"], mode="lines", name="限價線", line={"color": MATPLOTLIB_LIMIT_COLOR, "width": 1.6, "dash": "dash"}, line_shape="hv", connectgaps=False), row=1, col=1)
    if np.isfinite(chart_payload["entry_line"]).any():
        fig.add_trace(go.Scatter(x=dates, y=chart_payload["entry_line"], mode="lines", name="成交線", line={"color": MATPLOTLIB_ENTRY_COLOR, "width": 1.8}, line_shape="hv", connectgaps=False), row=1, col=1)
    present_plotly_legends = {"K線", "停損線"}
    if np.isfinite(chart_payload["tp_line"]).any():
        present_plotly_legends.add("半倉停利線")
    if np.isfinite(chart_payload["limit_line"]).any():
        present_plotly_legends.add("限價線")
    if np.isfinite(chart_payload["entry_line"]).any():
        present_plotly_legends.add("成交線")
    for trace_name, markers in chart_payload["marker_groups"].items():
        style = ACTION_STYLE_MAP.get(trace_name, {"plotly_symbol": "circle", "color": MATPLOTLIB_TEXT_COLOR})
        fig.add_trace(go.Scatter(x=[item["date"] for item in markers], y=[item["price"] for item in markers], mode="markers", name=trace_name, marker={"symbol": style["plotly_symbol"], "size": 10, "color": style["color"], "line": {"width": 1, "color": style["color"]}}, hovertemplate="%{text}<extra></extra>", text=[item["hover_text"] for item in markers]), row=1, col=1)
        present_plotly_legends.add(trace_name)
    for label, color, dash, width in CHART_LINE_LEGEND_SPECS:
        if label in present_plotly_legends:
            continue
        plotly_dash = "dash" if dash != "solid" else "solid"
        fig.add_trace(go.Scatter(x=[dates[0]], y=[None], mode="lines", name=label, line={"color": color, "width": width, "dash": plotly_dash}, visible="legendonly", hoverinfo="skip"), row=1, col=1)
        present_plotly_legends.add(label)
    for label in CHART_EVENT_LEGEND_ORDER:
        if label in present_plotly_legends:
            continue
        style = CHART_SIGNAL_LEGEND_STYLE.get(label) or ACTION_STYLE_MAP.get(label)
        if not style:
            continue
        fig.add_trace(go.Scatter(x=[dates[0]], y=[None], mode="markers", name=label, marker={"symbol": style["plotly_symbol"], "size": 10, "color": style["color"], "line": {"width": 1, "color": style["color"]}}, visible="legendonly", hoverinfo="skip"), row=1, col=1)
        present_plotly_legends.add(label)
    fig.add_trace(go.Bar(x=dates, y=chart_payload["volume"], name="成交量", marker={"color": volume_colors}, opacity=0.7), row=2, col=1)
    fig.update_layout(title=f"<b>{ticker} 單股回測 K 線交易檢視</b>", template="plotly_dark", paper_bgcolor=MATPLOTLIB_DARK_BG, plot_bgcolor=MATPLOTLIB_DARK_BG, hovermode="x unified", dragmode="pan", xaxis_rangeslider_visible=False, legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0}, margin={"l": 40, "r": 40, "t": 70, "b": 40}, uirevision=ticker, font={"family": font_family, "color": MATPLOTLIB_TEXT_COLOR})
    fig.update_yaxes(title_text="價格", row=1, col=1, range=[default_ranges["price_min"], default_ranges["price_max"]])
    fig.update_yaxes(title_text="成交量", row=2, col=1, range=[default_ranges["volume_min"], default_ranges["volume_max"]])
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", range=[dates[x_start_idx], dates[x_end_idx]])
    fig.write_html(output_path, config={"displaylogo": False, "responsive": True, "scrollZoom": True})
    return output_path

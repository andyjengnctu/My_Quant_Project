import os

import numpy as np
import pandas as pd


CHART_SIGNAL_BOX_ALPHA = 0.32
CHART_DEFAULT_LOOKBACK_MONTHS = 18
CHART_DEFAULT_LOOKBACK_FALLBACK_BARS = 380
CHART_FOCUS_PADDING_BARS = 15
CHART_FALLBACK_TAIL_BARS = 120
CHART_MIN_WINDOW_BARS = 40
CHART_PRICE_PADDING_RATIO = 0.035
CHART_VOLUME_PADDING_RATIO = 0.10
MATPLOTLIB_DEBUG_CHART_FIGSIZE = (18.2, 10.6)
MATPLOTLIB_CANDLE_WIDTH = 0.72
MATPLOTLIB_MARKER_SIZE = 88
MATPLOTLIB_VOLUME_ALPHA = 0.42
MATPLOTLIB_DARK_BG = "#04070b"
MATPLOTLIB_GRID_COLOR = "#1a2634"
MATPLOTLIB_TEXT_COLOR = "#edf2f7"
MATPLOTLIB_MUTED_TEXT_COLOR = "#a8b3c2"
MATPLOTLIB_UP_COLOR = "#ff5b6e"
MATPLOTLIB_DOWN_COLOR = "#18b26b"
MATPLOTLIB_STOP_COLOR = "#ff4d4f"
MATPLOTLIB_TP_COLOR = "#22c55e"
MATPLOTLIB_LIMIT_COLOR = "#4f86ff"
MATPLOTLIB_ENTRY_COLOR = "#2f6df6"
MATPLOTLIB_INFO_BOX_FACE = (0.02, 0.04, 0.06, 0.72)
MATPLOTLIB_SIGNAL_BUY_COLOR = "#d9485f"
MATPLOTLIB_SIGNAL_SELL_COLOR = "#8b5cf6"
MATPLOTLIB_SIGNAL_TEXT_COLOR = "#f8fafc"
MATPLOTLIB_VOLUME_OVERLAY_HEIGHT_RATIO = 0.18
MATPLOTLIB_VOLUME_OVERLAY_BOTTOM_GAP = 0.015
MATPLOTLIB_CROSSHAIR_COLOR = "#9fb3c8"
MATPLOTLIB_HOVER_BOX_FACE = (0.02, 0.04, 0.06, 0.68)
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


ACTION_STYLE_MAP = {
    "限價買進": {"plotly_symbol": "line-ew-open", "mpl_marker": "_", "color": MATPLOTLIB_LIMIT_COLOR},
    "限價買進(延續候選)": {"plotly_symbol": "line-ew-open", "mpl_marker": "_", "color": "#7fb3ff"},
    "買進": {"plotly_symbol": "triangle-up", "mpl_marker": "^", "color": MATPLOTLIB_ENTRY_COLOR},
    "買進(延續候選)": {"plotly_symbol": "triangle-up", "mpl_marker": "^", "color": "#68d8ff"},
    "半倉停利": {"plotly_symbol": "diamond", "mpl_marker": "D", "color": MATPLOTLIB_TP_COLOR},
    "停損殺出": {"plotly_symbol": "x", "mpl_marker": "x", "color": MATPLOTLIB_STOP_COLOR},
    "指標賣出": {"plotly_symbol": "triangle-down", "mpl_marker": "v", "color": MATPLOTLIB_SIGNAL_SELL_COLOR},
    "期末強制結算": {"plotly_symbol": "square", "mpl_marker": "s", "color": "#cbd5e1"},
    "錯失賣出": {"plotly_symbol": "circle-open", "mpl_marker": "o", "color": "#fbbf24"},
}

ORDER_STATUS_LABELS = {
    "filled": "成交",
    "missed": "未成交",
    "abandoned": "先達停損放棄",
}


def get_matplotlib_cjk_font_candidates():
    return MATPLOTLIB_CJK_FONT_CANDIDATES


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
    }


def _resolve_chart_pos(chart_context, current_date):
    current_ts = pd.Timestamp(current_date)
    pos = chart_context["date_to_pos"].get(current_ts)
    if pos is None:
        raise KeyError(f"chart context 找不到日期: {current_ts!s}")
    return pos


def _append_marker(marker_list, *, trace_name, current_date, price, qty, hover_text, note=""):
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


def record_trade_marker(chart_context, *, current_date, action, price, qty, note=""):
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
    )


def record_signal_annotation(chart_context, *, current_date, signal_type, anchor_price, title, detail_lines, note=""):
    if chart_context is None or pd.isna(anchor_price):
        return
    normalized_signal_type = str(signal_type).strip().lower()
    if normalized_signal_type not in {"buy", "sell"}:
        raise ValueError(f"不支援的 signal_type: {signal_type!r}")
    detail_text = "\n".join(str(line) for line in detail_lines if str(line).strip())
    chart_context["signal_annotations"].append(
        {
            "date": pd.Timestamp(current_date),
            "signal_type": normalized_signal_type,
            "anchor_price": float(anchor_price),
            "title": str(title),
            "detail_text": detail_text,
            "note": str(note or ""),
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
    }
    payload["default_view"] = compute_default_view_window(dates, total_bars, focus_positions)
    payload["gui_render_window"] = compute_gui_render_window(payload)
    return payload


def _slice_visible_window(array, start_idx, end_idx):
    return np.asarray(array[start_idx : end_idx + 1])


def compute_visible_value_ranges(chart_payload, *, start_idx, end_idx, price_padding_ratio=CHART_PRICE_PADDING_RATIO, volume_padding_ratio=CHART_VOLUME_PADDING_RATIO):
    total_bars = int(len(chart_payload["x"]))
    if total_bars <= 0:
        return {"price_min": 0.0, "price_max": 1.0, "volume_min": 0.0, "volume_max": 1.0}
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
    marker_prices.extend(item["anchor_price"] for item in chart_payload.get("signal_annotations", []) if start_idx <= int(item["x"]) <= end_idx and not pd.isna(item["anchor_price"]))
    if marker_prices:
        candidate_price_arrays.append(np.asarray(marker_prices, dtype=np.float32))
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
    max_right = float(total_points) - 0.5
    full_width = max_right - min_left
    min_width = max(float(min_visible_bars), 4.0)
    width = max(float(right) - float(left), min_width)
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
    try:
        from matplotlib import font_manager
    except ImportError:
        return None
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


def _build_hover_text(chart_payload, index):
    idx = int(index)
    if idx < 0 or idx >= len(chart_payload["x"]):
        return ""
    parts = [
        chart_payload["date_labels"][idx],
        f"開 {chart_payload['open'][idx]:.2f}",
        f"高 {chart_payload['high'][idx]:.2f}",
        f"低 {chart_payload['low'][idx]:.2f}",
        f"收 {chart_payload['close'][idx]:.2f}",
        f"量 {chart_payload['volume'][idx] / 1_000_000:.2f}M",
    ]
    if np.isfinite(chart_payload["limit_line"][idx]):
        parts.append(f"限價 {chart_payload['limit_line'][idx]:.2f}")
    if np.isfinite(chart_payload["entry_line"][idx]):
        parts.append(f"成交 {chart_payload['entry_line'][idx]:.2f}")
    if np.isfinite(chart_payload["stop_line"][idx]):
        parts.append(f"停損 {chart_payload['stop_line'][idx]:.2f}")
    if np.isfinite(chart_payload["tp_line"][idx]):
        parts.append(f"停利 {chart_payload['tp_line'][idx]:.2f}")
    return "   ".join(parts)


def _render_signal_annotations(axis_price, signal_annotations, label_font, *, start_idx=None, end_idx=None):
    rendered = []
    for item in signal_annotations:
        if start_idx is not None and int(item["x"]) < int(start_idx):
            continue
        if end_idx is not None and int(item["x"]) > int(end_idx):
            continue
        is_buy = item["signal_type"] == "buy"
        y_offset = -34 if is_buy else -52
        arrow_color = MATPLOTLIB_SIGNAL_BUY_COLOR if is_buy else MATPLOTLIB_SIGNAL_SELL_COLOR
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
                color=MATPLOTLIB_SIGNAL_TEXT_COLOR,
                fontsize=9 if label_font is None else None,
                fontproperties=label_font,
                bbox={"boxstyle": "round,pad=0.3", "fc": arrow_color, "ec": arrow_color, "alpha": CHART_SIGNAL_BOX_ALPHA},
                arrowprops={"arrowstyle": "-|>", "color": arrow_color, "lw": 0.9, "alpha": 0.8},
                zorder=6,
                annotation_clip=True,
            )
        )
    return rendered


def _render_trade_labels(axis_price, marker_groups, label_font, *, start_idx=None, end_idx=None):
    rendered = []
    for trace_name, markers in marker_groups.items():
        if trace_name not in {"半倉停利", "停損殺出", "指標賣出", "期末強制結算"}:
            continue
        for marker in markers:
            if start_idx is not None and int(marker["x"]) < int(start_idx):
                continue
            if end_idx is not None and int(marker["x"]) > int(end_idx):
                continue
            if trace_name == "半倉停利":
                label = f"平倉: {marker['qty']:,}"
            elif trace_name == "停損殺出":
                label = "停損"
            elif trace_name == "指標賣出":
                label = "賣出"
            else:
                label = "結算"
            rendered.append(
                axis_price.annotate(
                    label,
                    xy=(marker["x"], marker["price"]),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color=ACTION_STYLE_MAP.get(trace_name, {}).get("color", MATPLOTLIB_TEXT_COLOR),
                    fontsize=9 if label_font is None else None,
                    fontproperties=label_font,
                    zorder=6,
                    annotation_clip=True,
                )
            )
    return rendered


def create_matplotlib_debug_chart_figure(*, chart_payload, ticker, show_volume=False):
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
    figure.subplots_adjust(left=0.042, right=0.985, top=0.94, bottom=0.07)
    axis_price.set_facecolor(MATPLOTLIB_DARK_BG)
    axis_price.grid(True, color=MATPLOTLIB_GRID_COLOR, alpha=0.72, linewidth=0.8)
    axis_price.tick_params(colors=MATPLOTLIB_TEXT_COLOR, labelsize=11)
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
    axis_price.vlines(x_positions, low_values, high_values, colors=candle_colors, linewidth=1.05, zorder=2)
    if np.any(up_mask):
        axis_price.vlines(x_positions[up_mask], body_low[up_mask], body_high[up_mask], colors=MATPLOTLIB_UP_COLOR, linewidth=5.8, zorder=3)
    if np.any(~up_mask):
        axis_price.vlines(x_positions[~up_mask], body_low[~up_mask], body_high[~up_mask], colors=MATPLOTLIB_DOWN_COLOR, linewidth=5.8, zorder=3)
    axis_price.step(x_positions, chart_payload["stop_line"], where="mid", color=MATPLOTLIB_STOP_COLOR, linewidth=1.9, label="停損線", zorder=4)
    if np.isfinite(chart_payload["tp_line"]).any():
        axis_price.step(x_positions, chart_payload["tp_line"], where="mid", color=MATPLOTLIB_TP_COLOR, linewidth=1.8, label="半倉停利線", zorder=4)
    if np.isfinite(chart_payload["limit_line"]).any():
        axis_price.step(x_positions, chart_payload["limit_line"], where="mid", color=MATPLOTLIB_LIMIT_COLOR, linewidth=1.5, linestyle=(0, (4, 2)), label="限價線", zorder=4)
    if np.isfinite(chart_payload["entry_line"]).any():
        axis_price.step(x_positions, chart_payload["entry_line"], where="mid", color=MATPLOTLIB_ENTRY_COLOR, linewidth=1.8, label="成交線", zorder=4)
    for trace_name, markers in chart_payload["marker_groups"].items():
        style = ACTION_STYLE_MAP.get(trace_name, {"mpl_marker": "o", "color": MATPLOTLIB_TEXT_COLOR})
        axis_price.scatter([item["x"] for item in markers], [item["price"] for item in markers], marker=style["mpl_marker"], s=MATPLOTLIB_MARKER_SIZE, color=style["color"], linewidths=1.5, zorder=5, label=trace_name)
    axis_price.set_title(f"{ticker} 單股回測 K 線交易檢視", fontsize=18, color=MATPLOTLIB_TEXT_COLOR, loc="left", pad=10, fontweight="bold" if title_font is None else None, fontproperties=title_font)
    axis_price.set_ylabel("價格", color=MATPLOTLIB_TEXT_COLOR, fontsize=12 if label_font is None else None, fontproperties=label_font)
    if show_volume and axis_volume is not None:
        volume_colors = np.where(up_mask, MATPLOTLIB_UP_COLOR, MATPLOTLIB_DOWN_COLOR)
        axis_volume.bar(x_positions, chart_payload["volume"], width=MATPLOTLIB_CANDLE_WIDTH, color=volume_colors, alpha=MATPLOTLIB_VOLUME_ALPHA, align="center", zorder=1, label="成交量")
    date_labels = chart_payload["date_labels"]
    def _format_date_label(x_value, _pos):
        rounded = int(round(x_value))
        return date_labels[rounded] if 0 <= rounded < len(date_labels) else ""
    axis_price.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))
    axis_price.xaxis.set_major_formatter(mticker.FuncFormatter(_format_date_label))
    line_handles, line_labels = axis_price.get_legend_handles_labels()
    if show_volume and axis_volume is not None:
        volume_handles, volume_labels = axis_volume.get_legend_handles_labels()
        line_handles = line_handles + volume_handles
        line_labels = line_labels + volume_labels
    if line_handles:
        axis_price.legend(line_handles, line_labels, loc="upper left", ncol=min(7, max(1, len(line_labels))), frameon=False, prop=legend_font, labelcolor=MATPLOTLIB_TEXT_COLOR, bbox_to_anchor=(0.0, 1.01), handlelength=2.2)
    hover_text_artist = axis_price.text(0.01, 0.965, "", transform=axis_price.transAxes, ha="left", va="top", color=MATPLOTLIB_TEXT_COLOR, fontsize=10 if legend_font is None else None, fontproperties=legend_font, bbox={"boxstyle": "round,pad=0.25", "fc": MATPLOTLIB_HOVER_BOX_FACE, "ec": "none"}, zorder=8)
    hover_text_artist.set_text(_build_hover_text(chart_payload, chart_payload["default_view"]["end_idx"]))
    crosshair_vline = axis_price.axvline(x=chart_payload["default_view"]["end_idx"], color=MATPLOTLIB_CROSSHAIR_COLOR, linewidth=0.8, linestyle=(0, (4, 4)), alpha=0.6, zorder=1)
    crosshair_hline = axis_price.axhline(y=chart_payload["close"][chart_payload["default_view"]["end_idx"]], color=MATPLOTLIB_CROSSHAIR_COLOR, linewidth=0.8, linestyle=(0, (4, 4)), alpha=0.35, zorder=1)
    rendered_signal_annotations = []
    rendered_trade_labels = []
    summary_lines = chart_payload.get("summary_box") or []
    summary_artist = None
    if summary_lines:
        summary_artist = axis_price.text(0.985, 0.52, "\n".join(summary_lines), transform=axis_price.transAxes, ha="right", va="center", color=MATPLOTLIB_TEXT_COLOR, fontsize=11 if legend_font is None else None, fontproperties=legend_font, bbox={"boxstyle": "round,pad=0.32", "fc": MATPLOTLIB_INFO_BOX_FACE, "ec": MATPLOTLIB_GRID_COLOR, "lw": 0.9}, zorder=6)
    status_box = chart_payload.get("status_box") or {}
    status_lines = status_box.get("lines") or []
    status_artist = None
    if status_lines:
        status_face = (0.02, 0.11, 0.08, 0.82) if status_box.get("ok", False) else (0.16, 0.10, 0.03, 0.85)
        status_artist = axis_price.text(0.985, 0.04, "\n".join(status_lines), transform=axis_price.transAxes, ha="right", va="bottom", color=MATPLOTLIB_TEXT_COLOR, fontsize=10 if legend_font is None else None, fontproperties=legend_font, bbox={"boxstyle": "round,pad=0.28", "fc": status_face, "ec": MATPLOTLIB_GRID_COLOR, "lw": 0.9}, zorder=6)
    default_view = chart_payload["default_view"]
    x_start, x_end = _clamp_chart_xlim(default_view["start_idx"] - 1, default_view["end_idx"] + 1, total_points=len(chart_payload["x"]))
    ranges = compute_visible_value_ranges(chart_payload, start_idx=x_start, end_idx=x_end)
    axis_price.set_xlim(x_start, x_end)
    axis_price.set_ylim(ranges["price_min"], ranges["price_max"])
    if axis_volume is not None:
        axis_volume.set_ylim(ranges["volume_min"], ranges["volume_max"])
    sync_state = {"updating": False, "last_window": None}

    def _replace_artist_list(target_list, new_items):
        for artist in target_list:
            try:
                artist.remove()
            except ValueError:
                pass
        target_list[:] = list(new_items)

    def _refresh_overlay_annotations(start_idx, end_idx):
        render_start = max(0, int(start_idx) - 2)
        render_end = min(len(chart_payload["x"]) - 1, int(end_idx) + 2)
        _replace_artist_list(
            rendered_signal_annotations,
            _render_signal_annotations(
                axis_price,
                chart_payload.get("signal_annotations", []),
                legend_font,
                start_idx=render_start,
                end_idx=render_end,
            ),
        )
        _replace_artist_list(
            rendered_trade_labels,
            _render_trade_labels(
                axis_price,
                chart_payload["marker_groups"],
                legend_font,
                start_idx=render_start,
                end_idx=render_end,
            ),
        )

    def _sync_visible_ranges(_axis):
        if sync_state["updating"]:
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
            if sync_state["last_window"] != visible_window:
                visible_ranges = compute_visible_value_ranges(chart_payload, start_idx=clamped_left, end_idx=clamped_right)
                axis_price.set_ylim(visible_ranges["price_min"], visible_ranges["price_max"])
                if axis_volume is not None:
                    axis_volume.set_ylim(visible_ranges["volume_min"], visible_ranges["volume_max"])
                _refresh_overlay_annotations(*visible_window)
                sync_state["last_window"] = visible_window
            if figure.canvas is not None:
                figure.canvas.draw_idle()
        finally:
            sync_state["updating"] = False
    initial_visible_window = (int(np.floor(x_start)), int(np.ceil(x_end)))
    _refresh_overlay_annotations(*initial_visible_window)
    sync_state["last_window"] = initial_visible_window
    axis_price.callbacks.connect("xlim_changed", _sync_visible_ranges)
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
        "toolbar_required": False,
        "hover_value_display_enabled": True,
        "twse_up_color": MATPLOTLIB_UP_COLOR,
        "twse_down_color": MATPLOTLIB_DOWN_COLOR,
        "summary_box_present": bool(summary_lines),
        "status_box_present": bool(status_lines),
        "signal_annotation_count": int(len(chart_payload.get("signal_annotations", []))),
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
        "hover_last_index": None,
        "summary_artist": summary_artist,
        "status_artist": status_artist,
        "signal_artists": rendered_signal_annotations,
        "trade_label_artists": rendered_trade_labels,
    }
    return figure


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
    drag_state = {"active": False, "anchor_x": None, "orig_xlim": None}
    canvas_widget = canvas.get_tk_widget()
    def _allowed_axis(event):
        return event.inaxes in {axis_price, axis_volume}
    def _update_hover(event):
        if not _allowed_axis(event):
            return
        data_x = _resolve_event_data_x(axis_price, event)
        if data_x is None:
            return
        nearest_idx = int(np.clip(round(data_x), 0, total_points - 1))
        if state.get("hover_last_index") == nearest_idx:
            return
        state["hover_last_index"] = nearest_idx
        hover_text_artist.set_text(_build_hover_text(chart_payload, nearest_idx))
        crosshair_vline.set_xdata([nearest_idx, nearest_idx])
        close_price = float(chart_payload["close"][nearest_idx])
        crosshair_hline.set_ydata([close_price, close_price])
        if figure.canvas is not None:
            figure.canvas.draw_idle()
    def _on_press(event):
        if event.button != 1 or not _allowed_axis(event):
            return
        anchor_x = _resolve_event_data_x(axis_price, event)
        if anchor_x is None:
            return
        drag_state["active"] = True
        drag_state["anchor_x"] = anchor_x
        drag_state["orig_xlim"] = axis_price.get_xlim()
        canvas_widget.configure(cursor="fleur")
    def _on_motion(event):
        if drag_state["active"]:
            current_x = _resolve_event_data_x(axis_price, event)
            if current_x is None:
                return
            origin_left, origin_right = drag_state["orig_xlim"]
            delta = drag_state["anchor_x"] - current_x
            next_left, next_right = _clamp_chart_xlim(origin_left + delta, origin_right + delta, total_points=total_points)
            axis_price.set_xlim(next_left, next_right)
            return
        _update_hover(event)
    def _on_release(event):
        if event.button == 1 and drag_state["active"]:
            drag_state["active"] = False
            drag_state["anchor_x"] = None
            drag_state["orig_xlim"] = None
            canvas_widget.configure(cursor="")
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
        axis_price.set_xlim(next_left, next_right)
        _update_hover(event)
    def _on_leave(_event):
        canvas_widget.configure(cursor="")
    connection_ids = {
        "button_press_event": canvas.mpl_connect("button_press_event", _on_press),
        "motion_notify_event": canvas.mpl_connect("motion_notify_event", _on_motion),
        "button_release_event": canvas.mpl_connect("button_release_event", _on_release),
        "scroll_event": canvas.mpl_connect("scroll_event", _on_scroll),
        "figure_leave_event": canvas.mpl_connect("figure_leave_event", _on_leave),
    }
    state["connection_ids"] = connection_ids
    figure._stock_chart_contract["mouse_wheel_zoom_enabled"] = True
    figure._stock_chart_contract["mouse_left_drag_pan_enabled"] = True
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
    for trace_name, markers in chart_payload["marker_groups"].items():
        style = ACTION_STYLE_MAP.get(trace_name, {"plotly_symbol": "circle", "color": MATPLOTLIB_TEXT_COLOR})
        fig.add_trace(go.Scatter(x=[item["date"] for item in markers], y=[item["price"] for item in markers], mode="markers", name=trace_name, marker={"symbol": style["plotly_symbol"], "size": 10, "color": style["color"], "line": {"width": 1, "color": style["color"]}}, hovertemplate="%{text}<extra></extra>", text=[item["hover_text"] for item in markers]), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=chart_payload["volume"], name="成交量", marker={"color": volume_colors}, opacity=0.7), row=2, col=1)
    fig.update_layout(title=f"<b>{ticker} 單股回測 K 線交易檢視</b>", template="plotly_dark", paper_bgcolor=MATPLOTLIB_DARK_BG, plot_bgcolor=MATPLOTLIB_DARK_BG, hovermode="x unified", dragmode="pan", xaxis_rangeslider_visible=False, legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0}, margin={"l": 40, "r": 40, "t": 70, "b": 40}, uirevision=ticker, font={"family": font_family, "color": MATPLOTLIB_TEXT_COLOR})
    fig.update_yaxes(title_text="價格", row=1, col=1, range=[default_ranges["price_min"], default_ranges["price_max"]])
    fig.update_yaxes(title_text="成交量", row=2, col=1, range=[default_ranges["volume_min"], default_ranges["volume_max"]])
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", range=[dates[x_start_idx], dates[x_end_idx]])
    fig.write_html(output_path, config={"displaylogo": False, "responsive": True, "scrollZoom": True})
    return output_path

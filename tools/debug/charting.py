import os

import numpy as np
import pandas as pd


CHART_FOCUS_PADDING_BARS = 15
CHART_FALLBACK_TAIL_BARS = 120
CHART_MIN_WINDOW_BARS = 40
CHART_PRICE_PADDING_RATIO = 0.03
CHART_VOLUME_PADDING_RATIO = 0.10
MATPLOTLIB_DEBUG_CHART_FIGSIZE = (17.8, 10.2)
MATPLOTLIB_CANDLE_WIDTH = 0.72
MATPLOTLIB_MARKER_SIZE = 96
MATPLOTLIB_VOLUME_ALPHA = 0.65
MATPLOTLIB_DARK_BG = "#0b0f14"
MATPLOTLIB_GRID_COLOR = "#243447"
MATPLOTLIB_TEXT_COLOR = "#e9ecef"
MATPLOTLIB_GUI_RENDER_CONTEXT_BARS = 36
MATPLOTLIB_GUI_MAX_RENDER_BARS = 180
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
    "限價買進": {"plotly_symbol": "line-ew-open", "mpl_marker": "_", "color": "#f59f00"},
    "限價買進(延續候選)": {"plotly_symbol": "line-ew-open", "mpl_marker": "_", "color": "#ffd43b"},
    "買進": {"plotly_symbol": "triangle-up", "mpl_marker": "^", "color": "#2f9e44"},
    "買進(延續候選)": {"plotly_symbol": "triangle-up", "mpl_marker": "^", "color": "#66d9e8"},
    "半倉停利": {"plotly_symbol": "diamond", "mpl_marker": "D", "color": "#4dabf7"},
    "停損殺出": {"plotly_symbol": "x", "mpl_marker": "x", "color": "#ff4d4f"},
    "指標賣出": {"plotly_symbol": "triangle-down", "mpl_marker": "v", "color": "#e599f7"},
    "期末強制結算": {"plotly_symbol": "square", "mpl_marker": "s", "color": "#adb5bd"},
    "錯失賣出": {"plotly_symbol": "circle-open", "mpl_marker": "o", "color": "#ffa94d"},
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
    return {
        "dates": dates,
        "date_to_pos": {pd.Timestamp(dt): idx for idx, dt in enumerate(dates)},
        "stop_line": np.full(len(dates), np.nan, dtype=np.float64),
        "tp_line": np.full(len(dates), np.nan, dtype=np.float64),
        "order_markers": [],
        "trade_markers": [],
    }


def _resolve_chart_pos(chart_context, current_date):
    current_ts = pd.Timestamp(current_date)
    pos = chart_context["date_to_pos"].get(current_ts)
    if pos is None:
        raise KeyError(f"chart context 找不到日期: {current_ts!s}")
    return pos


def _append_marker(marker_list, *, trace_name, current_date, price, hover_text):
    if pd.isna(price):
        return
    marker_list.append(
        {
            "trace_name": trace_name,
            "date": pd.Timestamp(current_date),
            "price": float(price),
            "hover_text": hover_text,
        }
    )


def record_active_levels(chart_context, *, current_date, stop_price=np.nan, tp_half_price=np.nan):
    if chart_context is None:
        return
    pos = _resolve_chart_pos(chart_context, current_date)
    if not pd.isna(stop_price):
        chart_context["stop_line"][pos] = float(stop_price)
    if not pd.isna(tp_half_price):
        chart_context["tp_line"][pos] = float(tp_half_price)


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
        hover_text=hover_text,
    )


def _normalize_line_array(values, expected_len):
    if values is None:
        return np.full(expected_len, np.nan, dtype=np.float64)
    normalized = np.asarray(values, dtype=np.float64)
    if normalized.size == expected_len:
        return normalized.copy()
    resized = np.full(expected_len, np.nan, dtype=np.float64)
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
            "hover_text": marker["hover_text"],
        }
        groups.setdefault(marker["trace_name"], []).append(normalized_marker)
        focus_positions.append(int(pos))
    return groups, focus_positions


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


def compute_default_view_window(total_bars, focus_positions, *, padding_bars=CHART_FOCUS_PADDING_BARS, fallback_tail_bars=CHART_FALLBACK_TAIL_BARS, min_window_bars=CHART_MIN_WINDOW_BARS):
    if total_bars <= 0:
        return {"start_idx": 0, "end_idx": 0}

    if focus_positions:
        start_idx = max(0, min(focus_positions) - int(padding_bars))
        end_idx = min(total_bars - 1, max(focus_positions) + int(padding_bars))
    else:
        tail_bars = min(total_bars, int(fallback_tail_bars))
        start_idx = max(0, total_bars - tail_bars)
        end_idx = total_bars - 1

    start_idx, end_idx = _expand_window_to_min_bars(
        int(start_idx),
        int(end_idx),
        total_bars=total_bars,
        min_window_bars=min(int(min_window_bars), total_bars),
    )
    return {"start_idx": int(start_idx), "end_idx": int(end_idx)}


def compute_gui_render_window(chart_payload, *, extra_context_bars=MATPLOTLIB_GUI_RENDER_CONTEXT_BARS, max_render_bars=MATPLOTLIB_GUI_MAX_RENDER_BARS):
    total_bars = int(len(chart_payload["x"]))
    if total_bars <= 0:
        return {"start_idx": 0, "end_idx": 0}

    default_view = chart_payload.get("default_view", {"start_idx": 0, "end_idx": total_bars - 1})
    start_idx = max(0, int(default_view.get("start_idx", 0)) - int(extra_context_bars))
    end_idx = min(total_bars - 1, int(default_view.get("end_idx", total_bars - 1)) + int(extra_context_bars))
    render_window = _expand_window_to_min_bars(
        start_idx,
        end_idx,
        total_bars=total_bars,
        min_window_bars=min(total_bars, int(max_render_bars)),
    )
    render_start, render_end = render_window
    if render_end - render_start + 1 > int(max_render_bars):
        width = int(max_render_bars)
        center = (render_start + render_end) // 2
        render_start = max(0, center - width // 2)
        render_end = min(total_bars - 1, render_start + width - 1)
        render_start = max(0, render_end - width + 1)
    return {"start_idx": int(render_start), "end_idx": int(render_end)}


def build_debug_chart_payload(price_df, chart_context):
    df_chart = price_df.copy()
    df_chart.index = pd.DatetimeIndex(pd.to_datetime(df_chart.index))
    df_chart = df_chart.sort_index()

    dates = pd.DatetimeIndex(df_chart.index)
    total_bars = int(len(df_chart))
    x_positions = np.arange(total_bars, dtype=np.float64)
    date_to_pos = {pd.Timestamp(dt): idx for idx, dt in enumerate(dates)}

    marker_groups, focus_positions = _build_marker_groups(
        marker_lists=[*(chart_context or {}).get("order_markers", []), *((chart_context or {}).get("trade_markers", []))],
        date_to_pos=date_to_pos,
    )

    payload = {
        "dates": dates,
        "date_labels": [dt.strftime("%Y-%m-%d") for dt in dates],
        "x": x_positions,
        "open": df_chart["Open"].to_numpy(dtype=np.float64, copy=False),
        "high": df_chart["High"].to_numpy(dtype=np.float64, copy=False),
        "low": df_chart["Low"].to_numpy(dtype=np.float64, copy=False),
        "close": df_chart["Close"].to_numpy(dtype=np.float64, copy=False),
        "volume": df_chart["Volume"].to_numpy(dtype=np.float64, copy=False),
        "up_mask": (df_chart["Close"] >= df_chart["Open"]).to_numpy(dtype=bool, copy=False),
        "stop_line": _normalize_line_array((chart_context or {}).get("stop_line"), total_bars),
        "tp_line": _normalize_line_array((chart_context or {}).get("tp_line"), total_bars),
        "marker_groups": marker_groups,
        "focus_positions": focus_positions,
    }
    payload["default_view"] = compute_default_view_window(total_bars, focus_positions)
    payload["gui_render_window"] = compute_gui_render_window(payload)
    return payload


def _slice_visible_window(array, start_idx, end_idx):
    return np.asarray(array[start_idx : end_idx + 1], dtype=np.float64)


def compute_visible_value_ranges(chart_payload, *, start_idx, end_idx, price_padding_ratio=CHART_PRICE_PADDING_RATIO, volume_padding_ratio=CHART_VOLUME_PADDING_RATIO):
    total_bars = int(len(chart_payload["x"]))
    if total_bars <= 0:
        return {
            "price_min": 0.0,
            "price_max": 1.0,
            "volume_min": 0.0,
            "volume_max": 1.0,
        }

    start_idx = max(0, min(int(np.floor(start_idx)), total_bars - 1))
    end_idx = max(start_idx, min(int(np.ceil(end_idx)), total_bars - 1))

    candidate_price_arrays = [
        _slice_visible_window(chart_payload["low"], start_idx, end_idx),
        _slice_visible_window(chart_payload["high"], start_idx, end_idx),
        _slice_visible_window(chart_payload["stop_line"], start_idx, end_idx),
        _slice_visible_window(chart_payload["tp_line"], start_idx, end_idx),
    ]

    marker_prices = []
    for markers in chart_payload["marker_groups"].values():
        marker_prices.extend(
            marker["price"]
            for marker in markers
            if start_idx <= int(marker["x"]) <= end_idx and not pd.isna(marker["price"])
        )
    if marker_prices:
        candidate_price_arrays.append(np.asarray(marker_prices, dtype=np.float64))

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
    price_range = {
        "price_min": float(price_min - price_padding),
        "price_max": float(price_max + price_padding),
    }

    visible_volume = _slice_visible_window(chart_payload["volume"], start_idx, end_idx)
    finite_volume = visible_volume[np.isfinite(visible_volume)]
    volume_max = float(np.max(finite_volume)) if finite_volume.size > 0 else 1.0
    volume_padding = max(volume_max * float(volume_padding_ratio), 1.0)
    return {
        **price_range,
        "volume_min": 0.0,
        "volume_max": float(volume_max + volume_padding),
    }


def _slice_chart_payload(chart_payload, *, start_idx, end_idx):
    total_bars = int(len(chart_payload["x"]))
    if total_bars <= 0:
        raise ValueError("chart_payload 不可為空。")

    start_idx = max(0, min(int(start_idx), total_bars - 1))
    end_idx = max(start_idx, min(int(end_idx), total_bars - 1))
    offset = int(start_idx)
    local_len = end_idx - start_idx + 1

    sliced_markers = {}
    for trace_name, markers in chart_payload["marker_groups"].items():
        local_markers = []
        for marker in markers:
            marker_x = int(marker["x"])
            if start_idx <= marker_x <= end_idx:
                local_markers.append({**marker, "x": marker_x - offset})
        if local_markers:
            sliced_markers[trace_name] = local_markers

    default_view = chart_payload.get("default_view", {"start_idx": 0, "end_idx": local_len - 1})
    return {
        "dates": chart_payload["dates"][start_idx : end_idx + 1],
        "date_labels": chart_payload["date_labels"][start_idx : end_idx + 1],
        "x": np.arange(local_len, dtype=np.float64),
        "open": _slice_visible_window(chart_payload["open"], start_idx, end_idx),
        "high": _slice_visible_window(chart_payload["high"], start_idx, end_idx),
        "low": _slice_visible_window(chart_payload["low"], start_idx, end_idx),
        "close": _slice_visible_window(chart_payload["close"], start_idx, end_idx),
        "volume": _slice_visible_window(chart_payload["volume"], start_idx, end_idx),
        "up_mask": np.asarray(chart_payload["up_mask"][start_idx : end_idx + 1], dtype=bool),
        "stop_line": _slice_visible_window(chart_payload["stop_line"], start_idx, end_idx),
        "tp_line": _slice_visible_window(chart_payload["tp_line"], start_idx, end_idx),
        "marker_groups": sliced_markers,
        "focus_positions": [int(pos - offset) for pos in chart_payload.get("focus_positions", []) if start_idx <= int(pos) <= end_idx],
        "default_view": {
            "start_idx": max(0, int(default_view.get("start_idx", 0)) - offset),
            "end_idx": min(local_len - 1, int(default_view.get("end_idx", local_len - 1)) - offset),
        },
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


def create_matplotlib_debug_chart_figure(*, chart_payload, ticker, show_volume=False):
    try:
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
        from matplotlib import rcParams
        from matplotlib import ticker as mticker
        from matplotlib.font_manager import FontProperties
    except ImportError as exc:
        raise RuntimeError("缺少 matplotlib，無法在 GUI 內嵌單股回測 K 線圖。") from exc

    x_positions = chart_payload["x"]
    if len(x_positions) == 0:
        raise ValueError("chart_payload 不可為空。")

    render_window = compute_gui_render_window(chart_payload)
    render_payload = _slice_chart_payload(
        chart_payload,
        start_idx=render_window["start_idx"],
        end_idx=render_window["end_idx"],
    )

    font_family = _resolve_matplotlib_font_family()
    base_font = FontProperties(family=font_family) if font_family else None
    title_font = FontProperties(family=font_family, weight="bold", size=18) if font_family else None
    label_font = FontProperties(family=font_family, size=13) if font_family else None
    legend_font = FontProperties(family=font_family, size=10) if font_family else None
    rcParams["axes.unicode_minus"] = False
    if font_family:
        rcParams["font.sans-serif"] = [font_family, *MATPLOTLIB_CJK_FONT_CANDIDATES]

    figure = Figure(figsize=MATPLOTLIB_DEBUG_CHART_FIGSIZE, dpi=96, facecolor=MATPLOTLIB_DARK_BG)
    if show_volume:
        axis_price = figure.add_subplot(2, 1, 1)
        axis_volume = figure.add_subplot(2, 1, 2, sharex=axis_price)
        figure.subplots_adjust(left=0.048, right=0.995, top=0.955, bottom=0.072, hspace=0.04)
        axes = (axis_price, axis_volume)
    else:
        axis_price = figure.add_subplot(1, 1, 1)
        axis_volume = None
        figure.subplots_adjust(left=0.048, right=0.995, top=0.955, bottom=0.072)
        axes = (axis_price,)

    for axis in axes:
        axis.set_facecolor(MATPLOTLIB_DARK_BG)
        axis.grid(True, color=MATPLOTLIB_GRID_COLOR, alpha=0.65, linewidth=0.8)
        axis.tick_params(colors=MATPLOTLIB_TEXT_COLOR, labelsize=11)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(MATPLOTLIB_GRID_COLOR)
        axis.spines["bottom"].set_color(MATPLOTLIB_GRID_COLOR)

    candle_width = MATPLOTLIB_CANDLE_WIDTH
    x_positions = render_payload["x"]
    for idx, x_pos in enumerate(x_positions):
        is_up = bool(render_payload["up_mask"][idx])
        body_color = "#2ec4b6" if is_up else "#ff6b6b"
        axis_price.vlines(
            x_pos,
            render_payload["low"][idx],
            render_payload["high"][idx],
            color=body_color,
            linewidth=1.2,
            zorder=2,
        )
        body_low = min(render_payload["open"][idx], render_payload["close"][idx])
        body_high = max(render_payload["open"][idx], render_payload["close"][idx])
        body_height = max(body_high - body_low, 0.01)
        axis_price.add_patch(
            Rectangle(
                (x_pos - candle_width / 2.0, body_low),
                candle_width,
                body_height,
                facecolor=body_color,
                edgecolor=body_color,
                linewidth=1.0,
                zorder=3,
            )
        )

    axis_price.step(
        x_positions,
        render_payload["stop_line"],
        where="mid",
        color="#ff4d4f",
        linewidth=2.0,
        label="停損線",
        zorder=1,
    )
    if np.isfinite(render_payload["tp_line"]).any():
        axis_price.step(
            x_positions,
            render_payload["tp_line"],
            where="mid",
            color="#4dabf7",
            linewidth=2.0,
            linestyle=":",
            label="半倉停利線",
            zorder=1,
        )

    for trace_name, markers in render_payload["marker_groups"].items():
        style = ACTION_STYLE_MAP.get(trace_name, {"mpl_marker": "o", "color": "#ffffff"})
        axis_price.scatter(
            [item["x"] for item in markers],
            [item["price"] for item in markers],
            marker=style["mpl_marker"],
            s=MATPLOTLIB_MARKER_SIZE,
            color=style["color"],
            linewidths=1.5,
            zorder=4,
            label=trace_name,
        )

    axis_price.set_title(
        f"{ticker} 單股回測 K 線交易檢視",
        fontsize=18,
        color=MATPLOTLIB_TEXT_COLOR,
        loc="left",
        pad=12,
        fontweight="bold" if title_font is None else None,
        fontproperties=title_font,
    )
    axis_price.set_ylabel("價格", color=MATPLOTLIB_TEXT_COLOR, fontsize=13 if label_font is None else None, fontproperties=label_font)

    if show_volume and axis_volume is not None:
        volume_colors = np.where(render_payload["up_mask"], "#2ec4b6", "#ff6b6b")
        axis_volume.bar(
            x_positions,
            render_payload["volume"],
            width=candle_width,
            color=volume_colors.tolist(),
            alpha=MATPLOTLIB_VOLUME_ALPHA,
            label="成交量",
        )
        axis_volume.set_ylabel("成交量", color=MATPLOTLIB_TEXT_COLOR, fontsize=13 if label_font is None else None, fontproperties=label_font)

    date_labels = render_payload["date_labels"]

    def _format_date_label(x_value, _pos):
        rounded = int(round(x_value))
        if 0 <= rounded < len(date_labels):
            return date_labels[rounded]
        return ""

    target_axis = axis_volume if axis_volume is not None else axis_price
    target_axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))
    target_axis.xaxis.set_major_formatter(mticker.FuncFormatter(_format_date_label))
    if axis_volume is not None:
        axis_price.tick_params(axis="x", labelbottom=False)

    line_handles, line_labels = axis_price.get_legend_handles_labels()
    volume_handles, volume_labels = (axis_volume.get_legend_handles_labels() if axis_volume is not None else ([], []))
    combined_handles = line_handles + volume_handles
    combined_labels = line_labels + volume_labels
    if combined_handles:
        axis_price.legend(
            combined_handles,
            combined_labels,
            loc="upper left",
            ncol=min(6, max(1, len(combined_labels))),
            frameon=False,
            prop=legend_font,
            labelcolor=MATPLOTLIB_TEXT_COLOR,
            bbox_to_anchor=(0.0, 1.02),
        )

    default_view = render_payload["default_view"]
    x_start = default_view["start_idx"] - 1
    x_end = default_view["end_idx"] + 1
    x_start, x_end = _clamp_chart_xlim(x_start, x_end, total_points=len(render_payload["x"]))
    ranges = compute_visible_value_ranges(render_payload, start_idx=x_start, end_idx=x_end)
    axis_price.set_xlim(x_start, x_end)
    axis_price.set_ylim(ranges["price_min"], ranges["price_max"])
    if axis_volume is not None:
        axis_volume.set_ylim(ranges["volume_min"], ranges["volume_max"])

    sync_state = {"updating": False}

    def _sync_visible_ranges(_axis):
        if sync_state["updating"]:
            return
        sync_state["updating"] = True
        try:
            left, right = axis_price.get_xlim()
            clamped_left, clamped_right = _clamp_chart_xlim(left, right, total_points=len(render_payload["x"]))
            if abs(clamped_left - left) > 1e-9 or abs(clamped_right - right) > 1e-9:
                axis_price.set_xlim(clamped_left, clamped_right, emit=False)
            visible_ranges = compute_visible_value_ranges(render_payload, start_idx=clamped_left, end_idx=clamped_right)
            axis_price.set_ylim(visible_ranges["price_min"], visible_ranges["price_max"])
            if axis_volume is not None:
                axis_volume.set_ylim(visible_ranges["volume_min"], visible_ranges["volume_max"])
            if figure.canvas is not None:
                figure.canvas.draw_idle()
        finally:
            sync_state["updating"] = False

    axis_price.callbacks.connect("xlim_changed", _sync_visible_ranges)

    for axis in axes:
        _apply_axis_text_font(axis, base_font)

    figure._stock_chart_contract = {
        "volume_visible": bool(show_volume),
        "selected_font_family": font_family or "",
        "render_start_idx": int(render_window["start_idx"]),
        "render_end_idx": int(render_window["end_idx"]),
        "render_bar_count": int(len(render_payload["x"])),
        "total_bar_count": int(len(chart_payload["x"])),
        "default_view": render_payload["default_view"],
        "mouse_wheel_zoom_enabled": False,
        "mouse_left_drag_pan_enabled": False,
        "toolbar_required": False,
    }
    figure._stock_chart_navigation_state = {
        "axis_price": axis_price,
        "axis_volume": axis_volume,
        "render_payload": render_payload,
        "total_points": int(len(render_payload["x"])),
        "sync_visible_ranges": _sync_visible_ranges,
        "connection_ids": {},
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
    total_points = int(state["total_points"])
    drag_state = {"active": False, "anchor_x": None, "orig_xlim": None}
    canvas_widget = canvas.get_tk_widget()

    def _allowed_axis(event):
        return event.inaxes in {axis_price, axis_volume}

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
        if not drag_state["active"]:
            return
        current_x = _resolve_event_data_x(axis_price, event)
        if current_x is None:
            return
        origin_left, origin_right = drag_state["orig_xlim"]
        delta = drag_state["anchor_x"] - current_x
        next_left, next_right = _clamp_chart_xlim(origin_left + delta, origin_right + delta, total_points=total_points)
        axis_price.set_xlim(next_left, next_right)
        canvas.draw_idle()

    def _on_release(event):
        if event.button == 1 and drag_state["active"]:
            drag_state["active"] = False
            drag_state["anchor_x"] = None
            drag_state["orig_xlim"] = None
            canvas_widget.configure(cursor="")

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
        canvas.draw_idle()

    connection_ids = {
        "button_press_event": canvas.mpl_connect("button_press_event", _on_press),
        "motion_notify_event": canvas.mpl_connect("motion_notify_event", _on_motion),
        "button_release_event": canvas.mpl_connect("button_release_event", _on_release),
        "scroll_event": canvas.mpl_connect("scroll_event", _on_scroll),
    }
    state["connection_ids"] = connection_ids
    figure._stock_chart_contract["mouse_wheel_zoom_enabled"] = True
    figure._stock_chart_contract["mouse_left_drag_pan_enabled"] = True
    figure._stock_chart_contract["toolbar_required"] = False
    return connection_ids


def export_debug_chart_html(price_df, *, ticker, output_dir, chart_context):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("缺少 plotly，無法輸出單股回測 K 線圖。") from exc

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"Debug_TradeChart_{ticker}.html")

    chart_payload = build_debug_chart_payload(price_df, chart_context)
    dates = chart_payload["dates"]
    x_start_idx = chart_payload["default_view"]["start_idx"]
    x_end_idx = chart_payload["default_view"]["end_idx"]
    default_ranges = compute_visible_value_ranges(chart_payload, start_idx=x_start_idx, end_idx=x_end_idx)
    volume_colors = np.where(chart_payload["up_mask"], "#2ec4b6", "#ff6b6b")
    font_family = _resolve_matplotlib_font_family() or "Microsoft JhengHei, Noto Sans CJK TC, sans-serif"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=chart_payload["open"],
            high=chart_payload["high"],
            low=chart_payload["low"],
            close=chart_payload["close"],
            name="K線",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=chart_payload["stop_line"],
            mode="lines",
            name="停損線",
            line={"color": "#ff4d4f", "width": 2},
            line_shape="hv",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    if np.isfinite(chart_payload["tp_line"]).any():
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=chart_payload["tp_line"],
                mode="lines",
                name="半倉停利線",
                line={"color": "#4dabf7", "width": 2, "dash": "dot"},
                line_shape="hv",
                connectgaps=False,
            ),
            row=1,
            col=1,
        )

    for trace_name, markers in chart_payload["marker_groups"].items():
        style = ACTION_STYLE_MAP.get(trace_name, {"plotly_symbol": "circle", "color": "#ffffff"})
        fig.add_trace(
            go.Scatter(
                x=[item["date"] for item in markers],
                y=[item["price"] for item in markers],
                mode="markers",
                name=trace_name,
                marker={
                    "symbol": style["plotly_symbol"],
                    "size": 11,
                    "color": style["color"],
                    "line": {"width": 1, "color": style["color"]},
                },
                hovertemplate="%{text}<extra></extra>",
                text=[item["hover_text"] for item in markers],
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=chart_payload["volume"],
            name="成交量",
            marker={"color": volume_colors.tolist()},
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"<b>{ticker} 單股回測 K 線交易檢視</b>",
        template="plotly_dark",
        hovermode="x unified",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
        margin={"l": 40, "r": 40, "t": 70, "b": 40},
        uirevision=ticker,
        font={"family": font_family},
    )
    fig.update_yaxes(title_text="價格", row=1, col=1, range=[default_ranges["price_min"], default_ranges["price_max"]])
    fig.update_yaxes(title_text="成交量", row=2, col=1, range=[default_ranges["volume_min"], default_ranges["volume_max"]])
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        range=[dates[x_start_idx], dates[x_end_idx]],
    )
    fig.write_html(
        output_path,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )
    return output_path

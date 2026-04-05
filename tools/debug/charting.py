import os

import numpy as np
import pandas as pd


CHART_FOCUS_PADDING_BARS = 15
CHART_FALLBACK_TAIL_BARS = 120
CHART_MIN_WINDOW_BARS = 40
CHART_PRICE_PADDING_RATIO = 0.03
CHART_VOLUME_PADDING_RATIO = 0.10
MATPLOTLIB_DEBUG_CHART_FIGSIZE = (13.8, 8.6)
MATPLOTLIB_CANDLE_WIDTH = 0.72
MATPLOTLIB_MARKER_SIZE = 96
MATPLOTLIB_VOLUME_ALPHA = 0.65
MATPLOTLIB_DARK_BG = "#0b0f14"
MATPLOTLIB_GRID_COLOR = "#243447"
MATPLOTLIB_TEXT_COLOR = "#e9ecef"


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


def create_matplotlib_debug_chart_figure(*, chart_payload, ticker):
    try:
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
        from matplotlib import ticker as mticker
    except ImportError as exc:
        raise RuntimeError("缺少 matplotlib，無法在 GUI 內嵌單股回測 K 線圖。") from exc

    x_positions = chart_payload["x"]
    if len(x_positions) == 0:
        raise ValueError("chart_payload 不可為空。")

    figure = Figure(figsize=MATPLOTLIB_DEBUG_CHART_FIGSIZE, dpi=100, facecolor=MATPLOTLIB_DARK_BG, constrained_layout=True)
    axis_price = figure.add_subplot(2, 1, 1)
    axis_volume = figure.add_subplot(2, 1, 2, sharex=axis_price)

    for axis in (axis_price, axis_volume):
        axis.set_facecolor(MATPLOTLIB_DARK_BG)
        axis.grid(True, color=MATPLOTLIB_GRID_COLOR, alpha=0.65, linewidth=0.8)
        axis.tick_params(colors=MATPLOTLIB_TEXT_COLOR, labelsize=11)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(MATPLOTLIB_GRID_COLOR)
        axis.spines["bottom"].set_color(MATPLOTLIB_GRID_COLOR)

    candle_width = MATPLOTLIB_CANDLE_WIDTH
    for idx, x_pos in enumerate(x_positions):
        is_up = bool(chart_payload["up_mask"][idx])
        body_color = "#2ec4b6" if is_up else "#ff6b6b"
        axis_price.vlines(
            x_pos,
            chart_payload["low"][idx],
            chart_payload["high"][idx],
            color=body_color,
            linewidth=1.2,
            zorder=2,
        )
        body_low = min(chart_payload["open"][idx], chart_payload["close"][idx])
        body_high = max(chart_payload["open"][idx], chart_payload["close"][idx])
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
        chart_payload["stop_line"],
        where="mid",
        color="#ff4d4f",
        linewidth=2.0,
        label="停損線",
        zorder=1,
    )
    if np.isfinite(chart_payload["tp_line"]).any():
        axis_price.step(
            x_positions,
            chart_payload["tp_line"],
            where="mid",
            color="#4dabf7",
            linewidth=2.0,
            linestyle=":",
            label="半倉停利線",
            zorder=1,
        )

    for trace_name, markers in chart_payload["marker_groups"].items():
        style = ACTION_STYLE_MAP.get(trace_name, {"mpl_marker": "o", "color": "#ffffff"})
        scatter = axis_price.scatter(
            [item["x"] for item in markers],
            [item["price"] for item in markers],
            marker=style["mpl_marker"],
            s=MATPLOTLIB_MARKER_SIZE,
            color=style["color"],
            linewidths=1.5,
            zorder=4,
            label=trace_name,
        )

    volume_colors = np.where(chart_payload["up_mask"], "#2ec4b6", "#ff6b6b")
    axis_volume.bar(
        x_positions,
        chart_payload["volume"],
        width=candle_width,
        color=volume_colors.tolist(),
        alpha=MATPLOTLIB_VOLUME_ALPHA,
        label="成交量",
    )

    axis_price.set_title(f"{ticker} 單股回測 K 線交易檢視", fontsize=18, color=MATPLOTLIB_TEXT_COLOR, loc="left", pad=12, fontweight="bold")
    axis_price.set_ylabel("價格", color=MATPLOTLIB_TEXT_COLOR, fontsize=13)
    axis_volume.set_ylabel("成交量", color=MATPLOTLIB_TEXT_COLOR, fontsize=13)

    date_labels = chart_payload["date_labels"]

    def _format_date_label(x_value, _pos):
        rounded = int(round(x_value))
        if 0 <= rounded < len(date_labels):
            return date_labels[rounded]
        return ""

    axis_volume.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))
    axis_volume.xaxis.set_major_formatter(mticker.FuncFormatter(_format_date_label))
    axis_price.tick_params(axis="x", labelbottom=False)

    line_handles, line_labels = axis_price.get_legend_handles_labels()
    volume_handles, volume_labels = axis_volume.get_legend_handles_labels()
    combined_handles = line_handles + volume_handles
    combined_labels = line_labels + volume_labels
    if combined_handles:
        axis_price.legend(
            combined_handles,
            combined_labels,
            loc="upper left",
            ncol=min(6, max(1, len(combined_labels))),
            frameon=False,
            fontsize=10,
            labelcolor=MATPLOTLIB_TEXT_COLOR,
            bbox_to_anchor=(0.0, 1.02),
        )

    default_view = chart_payload["default_view"]
    x_start = default_view["start_idx"] - 1
    x_end = default_view["end_idx"] + 1
    ranges = compute_visible_value_ranges(chart_payload, start_idx=default_view["start_idx"], end_idx=default_view["end_idx"])
    axis_price.set_xlim(x_start, x_end)
    axis_price.set_ylim(ranges["price_min"], ranges["price_max"])
    axis_volume.set_ylim(ranges["volume_min"], ranges["volume_max"])

    sync_state = {"updating": False}

    def _sync_visible_ranges(_axis):
        if sync_state["updating"]:
            return
        sync_state["updating"] = True
        try:
            left, right = axis_price.get_xlim()
            visible_ranges = compute_visible_value_ranges(chart_payload, start_idx=left, end_idx=right)
            axis_price.set_ylim(visible_ranges["price_min"], visible_ranges["price_max"])
            axis_volume.set_ylim(visible_ranges["volume_min"], visible_ranges["volume_max"])
            if figure.canvas is not None:
                figure.canvas.draw_idle()
        finally:
            sync_state["updating"] = False

    axis_price.callbacks.connect("xlim_changed", _sync_visible_ranges)
    figure.autofmt_xdate(rotation=0, ha="center")
    return figure


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

import os

import numpy as np
import pandas as pd


ACTION_STYLE_MAP = {
    "限價買進": {"symbol": "line-ew-open", "color": "#f59f00"},
    "限價買進(延續候選)": {"symbol": "line-ew-open", "color": "#ffd43b"},
    "買進": {"symbol": "triangle-up", "color": "#2f9e44"},
    "買進(延續候選)": {"symbol": "triangle-up", "color": "#66d9e8"},
    "半倉停利": {"symbol": "diamond", "color": "#4dabf7"},
    "停損殺出": {"symbol": "x", "color": "#ff4d4f"},
    "指標賣出": {"symbol": "triangle-down", "color": "#e599f7"},
    "期末強制結算": {"symbol": "square", "color": "#adb5bd"},
    "錯失賣出": {"symbol": "circle-open", "color": "#ffa94d"},
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
    marker_list.append({
        "trace_name": trace_name,
        "date": pd.Timestamp(current_date),
        "price": float(price),
        "hover_text": hover_text,
    })


def record_active_levels(chart_context, *, current_date, stop_price=np.nan, tp_half_price=np.nan):
    pos = _resolve_chart_pos(chart_context, current_date)
    if not pd.isna(stop_price):
        chart_context["stop_line"][pos] = float(stop_price)
    if not pd.isna(tp_half_price):
        chart_context["tp_line"][pos] = float(tp_half_price)


def record_limit_order(chart_context, *, current_date, limit_price, qty, entry_type, status, note=""):
    if pd.isna(limit_price):
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
    if pd.isna(price):
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


def export_debug_chart_html(price_df, *, ticker, output_dir, chart_context):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("缺少 plotly，無法輸出單股回測 K 線圖。") from exc

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"Debug_TradeChart_{ticker}.html")

    df_chart = price_df.copy()
    df_chart.index = pd.DatetimeIndex(pd.to_datetime(df_chart.index))
    df_chart = df_chart.sort_index()
    up_mask = df_chart["Close"] >= df_chart["Open"]
    volume_colors = np.where(up_mask.to_numpy(dtype=bool), "#2ec4b6", "#ff6b6b")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )
    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"],
            high=df_chart["High"],
            low=df_chart["Low"],
            close=df_chart["Close"],
            name="K線",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_context["dates"],
            y=chart_context["stop_line"],
            mode="lines",
            name="停損線",
            line={"color": "#ff4d4f", "width": 2},
            line_shape="hv",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    if np.isfinite(chart_context["tp_line"]).any():
        fig.add_trace(
            go.Scatter(
                x=chart_context["dates"],
                y=chart_context["tp_line"],
                mode="lines",
                name="半倉停利線",
                line={"color": "#4dabf7", "width": 2, "dash": "dot"},
                line_shape="hv",
                connectgaps=False,
            ),
            row=1,
            col=1,
        )

    marker_groups = {}
    for marker in [*chart_context["order_markers"], *chart_context["trade_markers"]]:
        marker_groups.setdefault(marker["trace_name"], []).append(marker)

    for trace_name, markers in marker_groups.items():
        style = ACTION_STYLE_MAP.get(trace_name, {"symbol": "circle", "color": "#ffffff"})
        fig.add_trace(
            go.Scatter(
                x=[item["date"] for item in markers],
                y=[item["price"] for item in markers],
                mode="markers",
                name=trace_name,
                marker={
                    "symbol": style["symbol"],
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
            x=df_chart.index,
            y=df_chart["Volume"],
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
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.write_html(
        output_path,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )
    return output_path

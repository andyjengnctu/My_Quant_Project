from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest_core import run_v16_backtest
from core.data_utils import get_required_min_rows, resolve_latest_trade_date_from_frame, sanitize_ohlcv_dataframe
from core.runtime_utils import run_cli_entrypoint
from tools.scanner.runtime_common import ACTIVE_PARAMS_PATH, load_strict_params
from tools.scanner.stock_processor import build_history_qualified_row_from_stats

REDUCED_DIR = PROJECT_ROOT / "data" / "tw_stock_data_vip_reduced"


def _load_clean_df(ticker: str, *, trim_tail_rows: int = 0):
    raw_df = pd.read_csv(REDUCED_DIR / f"{ticker}.csv")
    if trim_tail_rows > 0 and len(raw_df) > trim_tail_rows + 5:
        raw_df = raw_df.iloc[:-trim_tail_rows].copy()
    params = load_strict_params(ACTIVE_PARAMS_PATH)
    clean_df, sanitize_stats = sanitize_ohlcv_dataframe(
        raw_df,
        ticker,
        min_rows=get_required_min_rows(params),
    )
    return clean_df, sanitize_stats, params


def _build_stats_and_row(ticker: str, *, trim_tail_rows: int = 0):
    clean_df, sanitize_stats, params = _load_clean_df(ticker, trim_tail_rows=trim_tail_rows)
    stats = run_v16_backtest(clean_df.copy(), params, ticker=ticker)
    trade_date = resolve_latest_trade_date_from_frame(clean_df)
    row = build_history_qualified_row_from_stats(
        ticker=ticker,
        stats=stats,
        params=params,
        sanitize_stats=sanitize_stats,
        trade_date=trade_date,
    )
    return clean_df, sanitize_stats, params, stats, row


def _has_terminal_position(stats: Dict) -> bool:
    return bool(stats.get("hasOpenPositionAtEnd")) or int(stats.get("current_position", 0) or 0) > 0


def _validate_latest_reduced_snapshot() -> Tuple[bool, Dict]:
    params = load_strict_params(ACTIVE_PARAMS_PATH)
    terminal_leaks = []
    displayed_rows = []
    for csv_path in sorted(REDUCED_DIR.glob("*.csv")):
        ticker = csv_path.stem
        raw_df = pd.read_csv(csv_path)
        clean_df, sanitize_stats = sanitize_ohlcv_dataframe(
            raw_df,
            ticker,
            min_rows=get_required_min_rows(params),
        )
        stats = run_v16_backtest(clean_df.copy(), params, ticker=ticker)
        row = build_history_qualified_row_from_stats(
            ticker=ticker,
            stats=stats,
            params=params,
            sanitize_stats=sanitize_stats,
            trade_date=resolve_latest_trade_date_from_frame(clean_df),
        )
        if row is None:
            continue
        displayed_rows.append({"ticker": ticker, "kind": row["kind"]})
        if _has_terminal_position(stats):
            terminal_leaks.append({"ticker": ticker, "kind": row["kind"]})

    return len(terminal_leaks) == 0, {
        "displayed_rows": displayed_rows,
        "terminal_leaks": terminal_leaks,
    }


def _validate_reduced_derived_extended_fixture() -> Tuple[bool, Dict]:
    clean_df, sanitize_stats, params, stats, _row = _build_stats_and_row("00635U", trim_tail_rows=20)
    derived_stats = copy.deepcopy(stats)
    derived_stats["hasOpenPositionAtEnd"] = False
    derived_stats["current_position"] = 0
    row = build_history_qualified_row_from_stats(
        ticker="00635U",
        stats=derived_stats,
        params=params,
        sanitize_stats=sanitize_stats,
        trade_date=resolve_latest_trade_date_from_frame(clean_df),
    )
    return (row is not None and row["kind"] == "extended"), {
        "source_ticker": "00635U",
        "source_trim_tail_rows": 20,
        "source_trade_date": str(clean_df.index[-1].date()),
        "source_has_extended_candidate": stats.get("extended_candidate_today") is not None,
        "source_terminal": _has_terminal_position(stats),
        "derived_kind": None if row is None else row["kind"],
    }


def _validate_reduced_derived_extended_tbd_fixture() -> Tuple[bool, Dict]:
    clean_df, sanitize_stats, params, stats, _row = _build_stats_and_row("00679B", trim_tail_rows=0)
    derived_stats = copy.deepcopy(stats)
    derived_stats["is_setup_today"] = False
    derived_stats["hasOpenPositionAtEnd"] = False
    derived_stats["current_position"] = 0
    row = build_history_qualified_row_from_stats(
        ticker="00679B",
        stats=derived_stats,
        params=params,
        sanitize_stats=sanitize_stats,
        trade_date=resolve_latest_trade_date_from_frame(clean_df),
    )
    return (row is not None and row["kind"] == "extended_tbd"), {
        "source_ticker": "00679B",
        "source_trim_tail_rows": 0,
        "source_trade_date": str(clean_df.index[-1].date()),
        "source_has_extended_tbd_candidate": stats.get("extended_candidate_tbd_today") is not None,
        "source_terminal": _has_terminal_position(stats),
        "derived_kind": None if row is None else row["kind"],
    }


def main(argv=None):
    latest_ok, latest_payload = _validate_latest_reduced_snapshot()
    ext_ok, ext_payload = _validate_reduced_derived_extended_fixture()
    tbd_ok, tbd_payload = _validate_reduced_derived_extended_tbd_fixture()

    print("[latest_reduced_snapshot]")
    print(f"displayed_rows : {latest_payload['displayed_rows']}")
    print(f"terminal_leaks : {latest_payload['terminal_leaks']}")
    print()
    print("[reduced_derived_extended_fixture]")
    for key, value in ext_payload.items():
        print(f"{key}: {value}")
    print()
    print("[reduced_derived_extended_tbd_fixture]")
    for key, value in tbd_payload.items():
        print(f"{key}: {value}")

    all_ok = latest_ok and ext_ok and tbd_ok
    print()
    print(f"result: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    run_cli_entrypoint(main)

"""Reusable Trading dataset update application service."""
from __future__ import annotations

from core.console_report import project_relative_display_path
from core.trading_market_clock import assert_completed_daily_information_date
from core.trading_identity import normalize_trading_ticker
from services.downloader import runtime as rt
from services.downloader.sync import smart_download_vip_data
from services.downloader.universe import get_market_last_date, get_or_update_universe


def run_trading_dataset_update(*, required_tickers=None) -> dict[str, object]:
    """Update the Trading dataset and return a structured summary.

    The underlying downloader/universe implementation remains the canonical producer;
    this service only turns the existing CLI workflow into a reusable Workbench seam.
    """
    print(f"🤖 Trading 智能量化建庫系統 (VIP版) 啟動 | {rt.get_taipei_now().strftime('%Y-%m-%d %H:%M')}\n")
    market_date = assert_completed_daily_information_date(get_market_last_date(), now=rt.get_taipei_now())
    universe_tickers = [normalize_trading_ticker(item) for item in get_or_update_universe()]
    required = sorted({normalize_trading_ticker(item) for item in list(required_tickers or [])})
    target_tickers = list(dict.fromkeys([*universe_tickers, *required]))
    if not target_tickers:
        raise RuntimeError("未取得任何可下載標的；請檢查 universe 快篩條件、資料來源或快取內容。")

    summary = dict(smart_download_vip_data(target_tickers, market_date))
    if (
        int(summary.get("count_success") or 0) == 0
        and int(summary.get("count_skipped_latest") or 0) == 0
    ) or int(summary.get("download_error_count") or 0) > 0:
        issue_log_path = summary.get("issue_log_path")
        issue_log_suffix = f"；詳細請見 {issue_log_path}" if issue_log_path else ""
        raise RuntimeError(
            "VIP Trading 資料更新未達可交易完整性："
            f"成功 {summary.get('count_success', 0)} 檔、"
            f"已最新跳過 {summary.get('count_skipped_latest', 0)} 檔、"
            f"最後日期檢查失敗 {summary.get('last_date_check_error_count', 0)} 檔、"
            f"下載失敗 {summary.get('download_error_count', 0)} 檔。"
            "Trading 不允許在已知 ticker download failure 下繼續訓練／Scanner"
            f"{issue_log_suffix}"
        )

    return {
        **summary,
        "status": "READY",
        "runtime_domain": "trading",
        "market_date": str(market_date),
        "ticker_count": int(len(target_tickers)),
        "universe_ticker_count": int(len(universe_tickers)),
        "required_position_tickers": required,
        "required_position_ticker_count": int(len(required)),
        "required_position_tickers_added": sorted(set(required) - set(universe_tickers)),
        "data_dir": project_relative_display_path(rt.SAVE_DIR, project_root=rt.PROJECT_ROOT),
        "output_dir": project_relative_display_path(rt.OUTPUT_DIR, project_root=rt.PROJECT_ROOT),
        "issue_log_path": (
            project_relative_display_path(summary["issue_log_path"], project_root=rt.PROJECT_ROOT)
            if summary.get("issue_log_path")
            else None
        ),
    }


__all__ = ["run_trading_dataset_update"]

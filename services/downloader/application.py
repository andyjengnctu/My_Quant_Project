"""Reusable Trading dataset update application service."""
from __future__ import annotations

from core.console_report import project_relative_display_path
from core.trading_market_clock import assert_completed_daily_information_date
from core.trading_identity import normalize_trading_ticker
from core.trading_dataset_identity import resolve_trading_dataset_member_tickers
from services.downloader import runtime as rt
from services.downloader.sync import smart_download_vip_data
from services.downloader.universe import get_market_last_date, get_or_update_universe
from services.downloader.finmind_http import FinMindHttpError
from services.downloader.trading_price_refresh import (
    TradingBulkPriceUnsupported,
    inspect_local_price_freshness,
    probe_latest_adjusted_price_market_date,
    refresh_trading_adjusted_price_dataset,
)


def run_trading_dataset_update(*, required_tickers=None, provider_client=None) -> dict[str, object]:
    """Update the execution-critical Trading dataset and return a structured summary.

    With ``provider_client`` the canonical path deduplicates FinMind requests and
    refreshes current-vintage adjusted history through the lowest-request exact
    strategy.  The legacy no-client seam is retained for isolated compatibility
    tests and emergency fallback only.
    """
    print(f"🤖 Trading 智能量化建庫系統 (VIP版) 啟動 | {rt.get_taipei_now().strftime('%Y-%m-%d %H:%M')}\n")
    probe = None
    if provider_client is None:
        market_date = assert_completed_daily_information_date(get_market_last_date(), now=rt.get_taipei_now())
    else:
        try:
            probe = probe_latest_adjusted_price_market_date(client=provider_client, now=rt.get_taipei_now())
        except TradingBulkPriceUnsupported as exc:
            reason = f"{type(exc).__name__}: {exc}"
            rt.append_downloader_issues("Canonical PriceAdj bulk probe fallback", [reason])
            probe = None
        except FinMindHttpError as exc:
            if exc.quota_exhausted:
                raise
            reason = f"{type(exc).__name__}: {exc}"
            rt.append_downloader_issues("Canonical PriceAdj bulk probe fallback", [reason])
            probe = None
        if probe is None:
            market_date = assert_completed_daily_information_date(
                get_market_last_date(client=provider_client), now=rt.get_taipei_now()
            )
        else:
            market_date = assert_completed_daily_information_date(probe.market_date, now=rt.get_taipei_now())
            print(f"📅 台股最新完整交易日 (FinMind PriceAdj bulk) 為: {market_date}")

    universe_tickers = [
        normalize_trading_ticker(item)
        for item in get_or_update_universe(market_date=market_date, client=provider_client)
    ]
    required = sorted({normalize_trading_ticker(item) for item in list(required_tickers or [])})
    retained = [
        normalize_trading_ticker(item)
        for item in resolve_trading_dataset_member_tickers(rt.SAVE_DIR, required=False)
    ]
    # Physical retention is intentionally broader than today's actionable
    # universe because the canonical Trading Optimizer consumes the complete
    # data_dir.  Refresh every retained member to the same FinMind current
    # vintage; only actionable universe members require a target-date row.
    target_tickers = list(dict.fromkeys([*universe_tickers, *required, *retained]))
    if not target_tickers:
        raise RuntimeError("未取得任何可下載標的；請檢查 universe 快篩條件、資料來源或快取內容。")

    if provider_client is None:
        summary = dict(
            smart_download_vip_data(
                target_tickers,
                market_date,
                require_target_date_tickers=universe_tickers,
            )
        )
    else:
        try:
            if probe is None:
                raise TradingBulkPriceUnsupported("full-market range probe unavailable")
            summary = dict(
                refresh_trading_adjusted_price_dataset(
                    target_tickers,
                    market_date,
                    client=provider_client,
                    probe=probe,
                    universe_tickers=universe_tickers,
                )
            )
        except TradingBulkPriceUnsupported as exc:
            # Capability fallback preserves the legacy exact current-vintage
            # semantics; do not silently publish partial bulk history.
            reason = f"{type(exc).__name__}: {exc}"
            rt.append_downloader_issues("Canonical PriceAdj bulk fallback", [reason])
            summary = dict(
                smart_download_vip_data(
                    target_tickers,
                    market_date,
                    client=provider_client,
                    require_target_date_tickers=universe_tickers,
                )
            )
            summary.update(
                {
                    "price_fetch_strategy": "per_ticker_fallback_after_bulk_capability_check",
                    "bulk_fallback_reason": reason,
                }
            )

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

    actionable_freshness = inspect_local_price_freshness(
        universe_tickers,
        market_date=market_date,
    )
    if actionable_freshness.stale or actionable_freshness.unreadable:
        raise RuntimeError(
            "Trading downloader 完成後仍有 actionable universe ticker 未到確認 market date；"
            f"market_date={market_date} stale={list(actionable_freshness.stale)[:20]} "
            f"count={len(actionable_freshness.stale)}"
        )

    return {
        **summary,
        "status": "READY",
        "runtime_domain": "trading",
        "market_date": str(market_date),
        "ticker_count": int(len(target_tickers)),
        "universe_ticker_count": int(len(universe_tickers)),
        "universe_tickers": list(universe_tickers),
        "required_position_tickers": required,
        "required_position_ticker_count": int(len(required)),
        "required_position_tickers_added": sorted(set(required) - set(universe_tickers)),
        "retained_history_ticker_count": int(len(retained)),
        "retained_history_tickers": list(retained),
        "retained_history_tickers_added": sorted(set(retained) - set(universe_tickers) - set(required)),
        "data_dir": project_relative_display_path(rt.SAVE_DIR, project_root=rt.PROJECT_ROOT),
        "output_dir": project_relative_display_path(rt.OUTPUT_DIR, project_root=rt.PROJECT_ROOT),
        "issue_log_path": (
            project_relative_display_path(summary["issue_log_path"], project_root=rt.PROJECT_ROOT)
            if summary.get("issue_log_path")
            else None
        ),
    }


__all__ = ["run_trading_dataset_update"]

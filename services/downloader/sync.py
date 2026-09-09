from pathlib import Path

import pandas as pd

from core.console_report import project_relative_display_path
from core.file_integrity import atomic_write_text
from core.market_data_contract import FINMIND_RAW_PRICE_ARCHIVE_DATASET
from core.trading_dataset_identity import inspect_trading_dataset_member_date_evidence
from services.downloader import runtime as rt
from services.downloader.finmind_http import request_finmind_data_with_retry


def _provider_get_data(*, client, dataset: str, data_id: str, start_date: str):
    if client is None:
        loader = rt.get_finmind_loader()
        return loader.get_data(dataset=dataset, data_id=data_id, start_date=start_date)
    return request_finmind_data_with_retry(
        client,
        dataset=dataset,
        data_id=data_id,
        start_date=start_date,
    )


def _normalize_adjusted_ticker_history(df: pd.DataFrame, *, market_last_date: str) -> tuple[pd.DataFrame, int]:
    if df is None or df.empty:
        raise ValueError("FinMind 回傳空資料")

    frame = df.copy()
    frame.columns = [c.capitalize() for c in frame.columns]
    frame = frame.rename(columns={"Trading_volume": "Volume", "Max": "High", "Min": "Low"})
    if "Date" not in frame.columns:
        raise KeyError("下載資料缺少 Date 欄位")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any():
        raise ValueError("FinMind 下載資料含無法解析的 Date")
    frame.set_index("Date", inplace=True)
    frame = frame.sort_index()
    if frame.index.duplicated().any():
        raise ValueError("FinMind 下載資料含重複 Date")

    target_date = pd.Timestamp(market_last_date).normalize()
    future_mask = frame.index.normalize() > target_date
    trimmed_future_row_count = int(future_mask.sum())
    if bool(future_mask.any()):
        frame = frame.loc[~future_mask].copy()
    if frame.empty:
        raise ValueError(f"FinMind 在完整交易日 {market_last_date} 以前沒有可用資料")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in frame.columns]
    if missing_cols:
        raise KeyError(f"缺少必要欄位: {missing_cols}")
    for column in required_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[required_cols].isna().any(axis=None):
        raise ValueError("FinMind 下載 OHLCV 含不合法數值")
    return frame[required_cols], trimmed_future_row_count


def _raw_history_date_evidence(*, sid: str, market_last_date: str, client) -> set[str]:
    raw = _provider_get_data(
        client=client,
        dataset=FINMIND_RAW_PRICE_ARCHIVE_DATASET,
        data_id=sid,
        start_date=rt.PRICE_HISTORY_START_DATE,
    )
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError(f"FinMind {FINMIND_RAW_PRICE_ARCHIVE_DATASET} 對 {sid} 缺少 full-history date evidence")
    columns = {str(column).strip().lower(): str(column) for column in raw.columns}
    date_column = columns.get("date")
    if date_column is None:
        raise KeyError(f"FinMind {FINMIND_RAW_PRICE_ARCHIVE_DATASET} 缺少 Date 欄位")
    target = pd.Timestamp(market_last_date).normalize()
    parsed = pd.to_datetime(raw[date_column], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"FinMind {FINMIND_RAW_PRICE_ARCHIVE_DATASET} 含無法解析的 Date")
    dates = {
        item.date().isoformat()
        for item in parsed
        if item.normalize() <= target
    }
    if not dates:
        raise ValueError(f"FinMind {FINMIND_RAW_PRICE_ARCHIVE_DATASET} 對 {sid} 沒有 target-date 以前的 date evidence")
    return dates


def smart_download_vip_data(
    tickers,
    market_last_date,
    verbose=True,
    client=None,
    require_target_date_tickers=None,
):
    rt.ensure_runtime_dirs()
    total = len(tickers)
    require_target_date = {str(item) for item in (require_target_date_tickers or [])}

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"\n💎 啟動 VIP 庫更新 (目標: {total} 檔)")
    vprint("-" * 65)

    download_errors = []
    last_date_check_errors = []
    count_success = 0
    count_skipped_latest = 0
    trimmed_future_row_count = 0
    raw_history_evidence_request_count = 0

    for i, sid in enumerate(tickers, 1):
        file_path = rt.os.path.join(rt.SAVE_DIR, f"{sid}.csv")

        baseline = inspect_trading_dataset_member_date_evidence(
            Path(rt.SAVE_DIR) / f"{sid}.csv",
            through_date=market_last_date,
        )
        if baseline.exists and baseline.readable and baseline.last_date == str(pd.Timestamp(market_last_date).date()):
            count_skipped_latest += 1
            vprint(
                f"\r⏳ [{i:03d}/{total:03d}] 成功:{count_success:>4} | 跳過:{count_skipped_latest:>4} | "
                f"失敗:{len(download_errors):>4} | {sid:<6} 已最新",
                end="",
                flush=True
            )
            continue
        if baseline.exists and not baseline.readable:
            last_date_check_errors.append(f"{sid}: {baseline.error}")
            if rt.VERBOSE_LAST_DATE_CHECK_ERRORS:
                vprint(f"\n注意：{sid} 檢查既有歷史發生錯誤，將以 raw date evidence 強制重建: {baseline.error}")

        vprint(
            f"\r⚡ [{i:03d}/{total:03d}] 成功:{count_success:>4} | 跳過:{count_skipped_latest:>4} | "
            f"失敗:{len(download_errors):>4} | 正在下載 {sid:<6}",
            end="",
            flush=True
        )
        try:
            df = _provider_get_data(
                client=client,
                dataset=rt.FINMIND_PRICE_DATASET,
                data_id=sid,
                start_date=rt.PRICE_HISTORY_START_DATE,
            )
            normalized, trimmed = _normalize_adjusted_ticker_history(
                df,
                market_last_date=market_last_date,
            )
            trimmed_future_row_count += int(trimmed)
            refreshed_dates = {item.date().isoformat() for item in normalized.index}

            if baseline.readable:
                missing_existing = sorted(set(baseline.dates) - refreshed_dates)
                if missing_existing:
                    raise ValueError(
                        "FinMind TaiwanStockPriceAdj per-ticker full-history 遺失既有日期；"
                        f"missing={missing_existing[:20]} count={len(missing_existing)}"
                    )
            else:
                raw_dates = _raw_history_date_evidence(
                    sid=str(sid),
                    market_last_date=market_last_date,
                    client=client,
                )
                raw_history_evidence_request_count += 1
                missing_raw_dates = sorted(raw_dates - refreshed_dates)
                if missing_raw_dates:
                    raise ValueError(
                        "FinMind TaiwanStockPriceAdj per-ticker full-history 未覆蓋 raw price date evidence；"
                        f"missing={missing_raw_dates[:20]} count={len(missing_raw_dates)}"
                    )

            target_date_text = str(pd.Timestamp(market_last_date).date())
            if str(sid) in require_target_date and target_date_text not in refreshed_dates:
                raise ValueError(
                    "FinMind TaiwanStockPriceAdj actionable universe ticker 缺少 target-date row；"
                    f"ticker={sid} market_date={target_date_text}"
                )

            atomic_write_text(file_path, normalized.to_csv())
            count_success += 1
            rt.time.sleep(rt.FINMIND_DOWNLOAD_SLEEP_SEC)

        except rt.EXPECTED_DOWNLOAD_EXCEPTIONS as e:
            download_errors.append((sid, f"{type(e).__name__}: {e}"))
            if rt.VERBOSE_DOWNLOAD_ERRORS:
                vprint(f"\n❌ {sid} 失敗: {type(e).__name__}: {e}")

    vprint("\n" + "-" * 65)
    vprint(
        f"🏆 本地尊爵資料庫更新完畢！成功 {count_success} 檔 | "
        f"已最新跳過 {count_skipped_latest} 檔 | "
        f"最後日期檢查失敗 {len(last_date_check_errors)} 檔 | "
        f"下載失敗 {len(download_errors)} 檔"
    )

    if last_date_check_errors:
        rt.append_downloader_issues("最後日期檢查失敗", last_date_check_errors)

    if download_errors:
        download_log_lines = [f"{sid} -> {err}" for sid, err in download_errors]
        rt.append_downloader_issues("下載失敗", download_log_lines)

    issue_log_path = None
    if last_date_check_errors or download_errors:
        issue_log_path = rt.get_downloader_issue_log_path()
        vprint(f"注意：非致命問題詳細已寫入: {project_relative_display_path(issue_log_path, project_root=rt.PROJECT_ROOT)}")

    return {
        "total": total,
        "count_success": count_success,
        "count_skipped_latest": count_skipped_latest,
        "last_date_check_error_count": len(last_date_check_errors),
        "download_error_count": len(download_errors),
        "trimmed_future_row_count": int(trimmed_future_row_count),
        "raw_history_evidence_request_count": int(raw_history_evidence_request_count),
        "issue_log_path": issue_log_path,
    }

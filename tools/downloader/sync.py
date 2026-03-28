import pandas as pd

from tools.downloader import runtime as rt


def smart_download_vip_data(tickers, market_last_date, verbose=True):
    rt.ensure_runtime_dirs()
    total = len(tickers)

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"\n💎 啟動 VIP 庫更新 (目標: {total} 檔)")
    vprint("-" * 65)

    download_errors = []
    last_date_check_errors = []
    count_success = 0
    count_skipped_latest = 0

    for i, sid in enumerate(tickers, 1):
        file_path = rt.os.path.join(rt.SAVE_DIR, f"{sid}.csv")

        if rt.os.path.exists(file_path):
            try:
                # # (AI註: 修復 1 - 移除檔案修改時間判斷，嚴格依賴 CSV 內最後一筆日期，避免假更新跳過)
                last_date_in_csv = str(pd.read_csv(file_path, index_col=0).tail(1).index[0]).split(' ')[0]
                if last_date_in_csv == market_last_date:
                    count_skipped_latest += 1
                    vprint(
                        f"\r⏳ [{i:03d}/{total:03d}] 成功:{count_success:>4} | 跳過:{count_skipped_latest:>4} | "
                        f"失敗:{len(download_errors):>4} | {sid:<6} 已最新",
                        end="",
                        flush=True
                    )
                    continue
            except rt.EXPECTED_LAST_DATE_CHECK_EXCEPTIONS as e:
                last_date_check_errors.append(f"{sid}: {type(e).__name__}: {e}")
                if rt.VERBOSE_LAST_DATE_CHECK_ERRORS:
                    vprint(f"\n⚠️ {sid} 檢查最後日期發生錯誤，將強制重抓: {type(e).__name__}: {e}")

        vprint(
            f"\r⚡ [{i:03d}/{total:03d}] 成功:{count_success:>4} | 跳過:{count_skipped_latest:>4} | "
            f"失敗:{len(download_errors):>4} | 正在下載 {sid:<6}",
            end="",
            flush=True
        )
        try:
            loader = rt.get_finmind_loader()
            df = loader.get_data(dataset=rt.FINMIND_PRICE_DATASET, data_id=sid, start_date="1990-01-01")
            if df is None or df.empty:
                raise ValueError("FinMind 回傳空資料")

            df.columns = [c.capitalize() for c in df.columns]
            df = df.rename(columns={"Trading_volume": "Volume", "Max": "High", "Min": "Low"})

            if 'Date' not in df.columns:
                raise KeyError("下載資料缺少 Date 欄位")

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df.sort_index()

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise KeyError(f"缺少必要欄位: {missing_cols}")

            df[required_cols].to_csv(file_path)
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
        vprint(f"⚠️ 非致命問題詳細已寫入: {issue_log_path}")

    return {
        "total": total,
        "count_success": count_success,
        "count_skipped_latest": count_skipped_latest,
        "last_date_check_error_count": len(last_date_check_errors),
        "download_error_count": len(download_errors),
        "issue_log_path": issue_log_path,
    }

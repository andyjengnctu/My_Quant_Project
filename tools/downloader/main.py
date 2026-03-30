import sys
import os

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.downloader import runtime as rt
from tools.downloader.universe import get_market_last_date, get_or_update_universe
from tools.downloader import sync as sync_runtime
from core.runtime_utils import enable_line_buffered_stdout, has_help_flag, validate_cli_args

SAVE_DIR = rt.SAVE_DIR
FINMIND_PRICE_DATASET = rt.FINMIND_PRICE_DATASET
dl = rt.dl
time = rt.time


def smart_download_vip_data(tickers, market_last_date, verbose=True):
    global SAVE_DIR, dl
    rt.SAVE_DIR = SAVE_DIR
    rt.dl = dl
    result = sync_runtime.smart_download_vip_data(tickers, market_last_date, verbose=verbose)
    SAVE_DIR = rt.SAVE_DIR
    dl = rt.dl
    return result


def main(argv=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    try:
        validate_cli_args(argv)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    if has_help_flag(argv):
        print("用法: python apps/smart_downloader.py")
        print("說明: 下載或更新完整資料集到預設 full dataset 路徑。")
        return 0

    try:
        print(f"🤖 智能量化建庫系統 (VIP版) 啟動 | {rt.get_taipei_now().strftime('%Y-%m-%d %H:%M')}\n")
        market_date = get_market_last_date()
        target_tickers = get_or_update_universe()

        if not target_tickers:
            raise RuntimeError("未取得任何可下載標的；請檢查 universe 快篩條件、資料來源或快取內容。")

        summary = smart_download_vip_data(target_tickers, market_date)
        if summary["count_success"] == 0 and summary["count_skipped_latest"] == 0:
            issue_log_path = summary.get("issue_log_path")
            issue_log_suffix = f"；詳細請見 {issue_log_path}" if issue_log_path else ""
            raise RuntimeError(
                "VIP 資料庫更新失敗："
                f"成功 {summary['count_success']} 檔、"
                f"已最新跳過 {summary['count_skipped_latest']} 檔、"
                f"最後日期檢查失敗 {summary['last_date_check_error_count']} 檔、"
                f"下載失敗 {summary['download_error_count']} 檔"
                f"{issue_log_suffix}"
            )
        return 0
    except (
        RuntimeError,
        FileNotFoundError,
        ValueError,
        OSError,
        requests.RequestException,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ImportError,
        ModuleNotFoundError,
    ) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

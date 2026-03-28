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
from core.runtime_utils import enable_line_buffered_stdout, has_help_flag

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

        smart_download_vip_data(target_tickers, market_date)
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

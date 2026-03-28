import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.downloader import runtime as rt
from tools.downloader.universe import get_market_last_date, get_or_update_universe
from tools.downloader import sync as sync_runtime

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


def main():
    print(f"🤖 智能量化建庫系統 (VIP版) 啟動 | {rt.get_taipei_now().strftime('%Y-%m-%d %H:%M')}\n")
    market_date = get_market_last_date()
    target_tickers = get_or_update_universe()

    if not target_tickers:
        raise RuntimeError("未取得任何可下載標的；請檢查 universe 快篩條件、資料來源或快取內容。")

    smart_download_vip_data(target_tickers, market_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

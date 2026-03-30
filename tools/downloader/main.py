import sys
import os
import importlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, has_help_flag, resolve_cli_program_name, validate_cli_args

_RUNTIME_EXPORT_NAMES = {"SAVE_DIR", "FINMIND_PRICE_DATASET", "dl", "time"}


def _get_downloader_modules():
    rt = importlib.import_module("tools.downloader.runtime")
    sync_runtime = importlib.import_module("tools.downloader.sync")
    universe_module = importlib.import_module("tools.downloader.universe")

    return rt, sync_runtime, universe_module.get_market_last_date, universe_module.get_or_update_universe


def smart_download_vip_data(tickers, market_last_date, verbose=True):
    global SAVE_DIR, dl, FINMIND_PRICE_DATASET, time

    rt, sync_runtime, _get_market_last_date, _get_or_update_universe = _get_downloader_modules()
    rt.SAVE_DIR = globals().get("SAVE_DIR", rt.SAVE_DIR)
    rt.dl = globals().get("dl", rt.dl)
    result = sync_runtime.smart_download_vip_data(tickers, market_last_date, verbose=verbose)
    SAVE_DIR = rt.SAVE_DIR
    FINMIND_PRICE_DATASET = rt.FINMIND_PRICE_DATASET
    dl = rt.dl
    time = rt.time
    return result


def __getattr__(name):
    if name in _RUNTIME_EXPORT_NAMES:
        rt, _sync_runtime, _get_market_last_date, _get_or_update_universe = _get_downloader_modules()
        value = getattr(rt, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv)
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/downloader/main.py")
        print(f"用法: python {program_name}")
        print("說明: 下載或更新完整資料集到預設 full dataset 路徑。")
        return 0

    try:
        import pandas as pd
        import requests

        rt, _sync_runtime, get_market_last_date, get_or_update_universe = _get_downloader_modules()
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

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
    run_cli_entrypoint(main)

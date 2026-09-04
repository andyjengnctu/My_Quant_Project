import sys
import os
import importlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, has_help_flag, resolve_cli_program_name, validate_cli_args

_RUNTIME_EXPORT_NAMES = {"SAVE_DIR", "FINMIND_PRICE_DATASET", "dl", "time"}


def _get_downloader_modules():
    rt = importlib.import_module("services.downloader.runtime")
    sync_runtime = importlib.import_module("services.downloader.sync")
    universe_module = importlib.import_module("services.downloader.universe")

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
        program_name = resolve_cli_program_name(argv, "services/downloader/main.py")
        print(f"用法: python {program_name}")
        print("說明: 下載或更新 Trading 完整資料集；Research 資料不會被修改。")
        return 0

    try:
        import pandas as pd
        import requests
        from services.downloader.application import run_trading_dataset_update
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        run_trading_dataset_update()
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

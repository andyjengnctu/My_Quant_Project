import sys
import os
import importlib
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.path_utils import project_relative_display_path
from core.runtime_utils import (
    enable_line_buffered_stdout,
    has_help_flag,
    is_interactive_console,
    resolve_cli_program_name,
    run_cli_entrypoint,
    validate_cli_args,
)

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


def _run_trading_dataset_update() -> int:
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


def _run_market_data_v2_preflight() -> int:
    try:
        from services.downloader.market_data_preflight import run_market_data_v2_preflight
        rt = importlib.import_module("services.downloader.runtime")
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    token = rt.resolve_finmind_api_token()
    if not token:
        print("❌ 找不到 FinMind API token；請使用既有 FINMIND_API_TOKEN 設定。", file=sys.stderr)
        return 1

    output_dir = Path(rt.OUTPUT_DIR) / "market_data_v2"
    try:
        result = run_market_data_v2_preflight(
            token=token,
            output_dir=output_dir,
            now=rt.get_taipei_now(),
            timeout_sec=rt.REQUEST_TIMEOUT_SEC,
        )
    except (RuntimeError, ValueError, OSError, ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    status = str(result.get("status") or "BLOCKED")
    quota = result.get("quota") or {}
    print("=" * 88)
    print(" Market Data V2｜Backer Preflight + Exact Bootstrap Planner")
    print("=" * 88)
    print(f"狀態                 : {status}")
    print(f"資料共同檢查日       : {result.get('as_of_date')}")
    print(f"歷史 instrument 數   : {result.get('historical_instrument_count')}")
    print(f"納入 dataset 數      : {result.get('included_dataset_count')}")
    print(f"Preflight data requests: {result.get('preflight_data_request_count')}")
    print(f"Live quota            : {quota.get('user_count_after')} / {quota.get('api_request_limit')}")
    plan = result.get("plan") or {}
    if plan:
        print(f"Exact bootstrap requests: {plan.get('total_requests')}")
        print(f"理論最低 quota-hours    : {float(plan.get('minimum_quota_hours') or 0):.2f}")
        print(f"理論最低 quota windows  : {plan.get('minimum_quota_windows')}")
    failures = result.get("probe_failures") or []
    if failures:
        print(f"Blocking probes         : {len(failures)}")
        for item in failures[:10]:
            print(f"  - {item.get('dataset')}: {item.get('error')}")
        if len(failures) > 10:
            print(f"  ... 另有 {len(failures) - 10} 筆，請看報表")
    for label, key in (("Markdown", "markdown_path"), ("JSON", "json_path")):
        raw_path = result.get(key)
        if raw_path:
            display = project_relative_display_path(raw_path, project_root=PROJECT_ROOT)
            print(f"{label:<20}: {display}")
    warning = quota.get("accounting_warning")
    if warning:
        print(f"⚠ quota accounting: {warning}")
    return 0 if status == "READY" else 1


def _interactive_menu() -> int:
    print("=" * 72)
    print(" Smart Downloader")
    print("=" * 72)
    print("[1] Trading 資料更新（現行正式流程）")
    print("[2] Market Data V2｜Backer Preflight + Exact Bootstrap Plan")
    print("[0] 離開")
    while True:
        try:
            choice = input("請選擇: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "1":
            return _run_trading_dataset_update()
        if choice == "2":
            return _run_market_data_v2_preflight()
        if choice == "0":
            return 0
        print("請輸入 0、1 或 2。")


def main(argv=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv)
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "services/downloader/main.py")
        print(f"用法: python {program_name}")
        print("說明: 互動式入口提供現行 Trading 更新，以及 Market Data V2 Backer Preflight / Exact Planner。")
        print("非互動環境維持既有行為：直接執行 Trading 資料更新。")
        return 0

    if len(argv) == 1 and is_interactive_console():
        return _interactive_menu()
    return _run_trading_dataset_update()


if __name__ == "__main__":
    run_cli_entrypoint(main)

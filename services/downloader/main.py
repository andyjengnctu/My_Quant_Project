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
        from config.market_data import MARKET_DATA_V2_HTTP_TIMEOUT_SEC
        result = run_market_data_v2_preflight(
            token=token,
            output_dir=output_dir,
            now=rt.get_taipei_now(),
            timeout_sec=MARKET_DATA_V2_HTTP_TIMEOUT_SEC,
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



def _run_market_data_v2_bootstrap() -> int:
    try:
        from services.downloader.finmind_http import FinMindHttpClient, FinMindHttpError
        from services.downloader.market_data_bootstrap_activation import (
            MarketDataBootstrapActivationError,
            execute_market_data_v2_bootstrap,
            get_existing_bootstrap_summary,
            prepare_market_data_v2_bootstrap_activation,
        )
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
        activation = prepare_market_data_v2_bootstrap_activation(output_dir=output_dir)
        existing = get_existing_bootstrap_summary(project_root=PROJECT_ROOT, activation=activation)
        from config.market_data import MARKET_DATA_V2_HTTP_TIMEOUT_SEC
        client = FinMindHttpClient(token=token, timeout_sec=MARKET_DATA_V2_HTTP_TIMEOUT_SEC)
        usage = client.get_usage()
    except (MarketDataBootstrapActivationError, FinMindHttpError, RuntimeError, ValueError, OSError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    manifest = activation.manifest
    done = int(existing.done) if existing is not None else 0
    print("=" * 88)
    print(" Market Data V2｜完整 Bootstrap 開始 / 續傳")
    print("=" * 88)
    print(f"Preflight             : {project_relative_display_path(activation.preflight_path, project_root=PROJECT_ROOT)}")
    print(f"資料截止              : {manifest.as_of_date}")
    print(f"Historical instruments: {manifest.historical_instrument_count}")
    print(f"Logical requests      : {manifest.total_requests}")
    print(f"已完成 / 剩餘         : {done} / {manifest.total_requests - done}")
    print(f"Manifest fingerprint  : {manifest.manifest_fingerprint}")
    print(f"Live quota            : {usage.user_count} / {usage.api_request_limit}")
    print("說明                  : 啟動後會跨 quota window 自動等待並續傳；Ctrl+C 可安全中斷後再次由本選項續傳。")
    try:
        confirmation = input("輸入 START 確認開始/續傳完整 bootstrap；其他輸入取消: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    if confirmation != "START":
        print("已取消，未開始完整 bootstrap。")
        return 0

    def _progress(event: dict[str, object]) -> None:
        done_now = int(event.get("done") or 0)
        total = int(event.get("total") or 0)
        pct = 100.0 * done_now / total if total > 0 else 0.0
        suffix = " REUSE" if event.get("recovered") else " DONE"
        data_id = event.get("data_id")
        target = str(event.get("dataset") or "") + (f"/{data_id}" if data_id else "")
        print(f"[Bootstrap] {done_now}/{total} ({pct:.1f}%) | {target}{suffix}")

    try:
        result = execute_market_data_v2_bootstrap(
            activation=activation,
            token=token,
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
            timeout_sec=MARKET_DATA_V2_HTTP_TIMEOUT_SEC,
            client=client,
            now_fn=rt.get_taipei_now,
            progress_fn=_progress,
        )
    except KeyboardInterrupt:
        print("\n⚠ Bootstrap 已由使用者中斷；已完成的 request/Parquet/ledger 會保留，下次選 [3] 可續傳。")
        return 130
    except (MarketDataBootstrapActivationError, FinMindHttpError, RuntimeError, ValueError, OSError, ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("=" * 88)
    print(" Market Data V2｜Bootstrap 執行結果")
    print("=" * 88)
    print(f"狀態                 : {result.get('status')}")
    print(f"完成 / 總數          : {result.get('done')} / {result.get('total')}")
    print(f"未完成               : {result.get('unfinished')}")
    print(f"Blocked              : {result.get('blocked')}")
    print(f"累積 data HTTP attempts: {result.get('http_attempts')}")
    for label, key in (("Markdown", "markdown_path"), ("JSON", "json_path")):
        raw_path = result.get(key)
        if raw_path:
            print(f"{label:<20}: {project_relative_display_path(raw_path, project_root=PROJECT_ROOT)}")
    if str(result.get("status")) != "DONE":
        return 1
    print("Bootstrap requests 已全部 DONE；開始本機完整性驗證並建立 immutable provider snapshot（不會再打 FinMind data API）。")
    return _finalize_market_data_v2_provider_snapshot(activation=activation, output_dir=output_dir, rt=rt)

def _finalize_market_data_v2_provider_snapshot(*, activation, output_dir, rt) -> int:
    try:
        from services.downloader.market_data_bootstrap_completion import (
            MarketDataBootstrapCompletionError,
            finalize_market_data_v2_provider_snapshot,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    def _verify_progress(event: dict[str, object]) -> None:
        verified = int(event.get("verified") or 0)
        total = int(event.get("total") or 0)
        pct = 100.0 * verified / total if total > 0 else 0.0
        data_id = event.get("data_id")
        target = str(event.get("dataset") or "") + (f"/{data_id}" if data_id else "")
        print(f"[Verify] {verified}/{total} ({pct:.1f}%) | {target}")

    try:
        result = finalize_market_data_v2_provider_snapshot(
            activation=activation,
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
            now_fn=rt.get_taipei_now,
            progress_fn=_verify_progress,
        )
    except (MarketDataBootstrapCompletionError, RuntimeError, ValueError, OSError, ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("=" * 88)
    print(" Market Data V2｜Provider Snapshot Finalization")
    print("=" * 88)
    print(f"狀態                 : {result.get('status')}")
    print(f"資料截止              : {result.get('as_of_date')}")
    print(f"Verified requests     : {result.get('verified_requests')} / {result.get('total_requests')}")
    print(f"Dataset 數            : {result.get('dataset_count')}")
    print(f"總 rows               : {result.get('total_rows')}")
    print(f"Snapshot fingerprint  : {result.get('snapshot_fingerprint')}")
    print(f"既有 snapshot REUSE   : {result.get('reused_existing_snapshot')}")
    for label, key in (("Provider Snapshot", "provider_snapshot_path"), ("Markdown", "markdown_path"), ("JSON", "json_path")):
        raw_path = result.get(key)
        if raw_path:
            display = raw_path if key == "provider_snapshot_path" else project_relative_display_path(raw_path, project_root=PROJECT_ROOT)
            print(f"{label:<20}: {display}")
    print("說明                 : 這只是 neutral provider source READY；Research V2 / Trading 尚未因此自動切換。")
    return 0


def _run_market_data_v2_provider_snapshot_finalize() -> int:
    try:
        from services.downloader.market_data_bootstrap_activation import (
            MarketDataBootstrapActivationError,
            prepare_market_data_v2_bootstrap_activation,
        )
        rt = importlib.import_module("services.downloader.runtime")
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(rt.OUTPUT_DIR) / "market_data_v2"
    try:
        activation = prepare_market_data_v2_bootstrap_activation(output_dir=output_dir)
    except (MarketDataBootstrapActivationError, RuntimeError, ValueError, OSError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return _finalize_market_data_v2_provider_snapshot(activation=activation, output_dir=output_dir, rt=rt)


def _interactive_menu() -> int:
    print("=" * 72)
    print(" Smart Downloader")
    print("=" * 72)
    print("[1] Trading 資料更新（現行正式流程）")
    print("[2] Market Data V2｜Backer Preflight + Exact Bootstrap Plan")
    print("[3] Market Data V2｜開始 / 續傳完整 Bootstrap")
    print("[4] Market Data V2｜驗證 / 重建 Provider Snapshot（不使用 API quota）")
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
        if choice == "3":
            return _run_market_data_v2_bootstrap()
        if choice == "4":
            return _run_market_data_v2_provider_snapshot_finalize()
        if choice == "0":
            return 0
        print("請輸入 0、1、2、3 或 4。")


def main(argv=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv)
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "services/downloader/main.py")
        print(f"用法: python {program_name}")
        print("說明: 互動式入口提供現行 Trading 更新、Market Data V2 Backer Preflight / Exact Planner、完整 Bootstrap 開始/續傳，以及不使用 API quota 的 Provider Snapshot 完整性驗證。")
        print("非互動環境維持既有行為：直接執行 Trading 資料更新。")
        return 0

    if len(argv) == 1 and is_interactive_console():
        return _interactive_menu()
    return _run_trading_dataset_update()


if __name__ == "__main__":
    run_cli_entrypoint(main)

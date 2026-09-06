import sys
import os
import importlib
import time as _monotonic_time
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


def _format_bootstrap_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours < 100:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours}h{minutes:02d}m"


def _estimate_bootstrap_eta_seconds(
    *,
    done: int,
    initial_done: int,
    total: int,
    elapsed_seconds: float,
    quota_wait_seconds: float = 0.0,
    quota_limit: int | None,
    quota_reserve: int | None,
    observed_sample_floor: int = 1,
) -> float | None:
    """Estimate remaining wall-clock time without treating quota wait as slow I/O.

    The sustainable provider quota is always an upper bound on throughput.  An
    observed active-processing rate is used only after enough jobs have finished
    in this process; quota wait time is removed from that active-rate sample.
    """

    remaining = max(0, int(total) - int(done))
    if remaining == 0:
        return 0.0
    process_done = max(0, int(done) - int(initial_done))
    active_elapsed = max(0.0, float(elapsed_seconds) - max(0.0, float(quota_wait_seconds)))
    observed_rate = None
    if process_done >= max(1, int(observed_sample_floor)) and active_elapsed > 0:
        observed_rate = process_done / active_elapsed
    quota_rate = None
    if quota_limit is not None and int(quota_limit) > 0:
        safe_per_hour = max(1, int(quota_limit) - max(0, int(quota_reserve or 0)))
        quota_rate = safe_per_hour / 3600.0
    rates = [rate for rate in (observed_rate, quota_rate) if rate is not None and rate > 0]
    if not rates:
        return None
    effective_rate = min(rates)
    return remaining / effective_rate



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
        from config.market_data import MARKET_DATA_V2_HTTP_TIMEOUT_SEC, MARKET_DATA_V2_PREFLIGHT_RETRY_POLICY
        result = run_market_data_v2_preflight(
            token=token,
            output_dir=output_dir,
            now=rt.get_taipei_now(),
            timeout_sec=MARKET_DATA_V2_HTTP_TIMEOUT_SEC,
            retryable_attempts=int(MARKET_DATA_V2_PREFLIGHT_RETRY_POLICY["retryable_attempts"]),
            retry_backoff_seconds=tuple(MARKET_DATA_V2_PREFLIGHT_RETRY_POLICY["retry_backoff_seconds"]),
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

    progress_started = _monotonic_time.monotonic()
    initial_done = done
    from core.market_data_execution_policy import get_market_data_execution_policy
    progress_sample_floor = max(1, int(get_market_data_execution_policy().progress_every_committed_requests))
    wait_line_open = False
    wait_line_width = 0

    def _close_wait_line() -> None:
        nonlocal wait_line_open, wait_line_width
        if not wait_line_open:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
        wait_line_open = False
        wait_line_width = 0

    def _write_wait_line(text: str) -> None:
        nonlocal wait_line_open, wait_line_width
        rendered = str(text)
        padding = " " * max(0, wait_line_width - len(rendered))
        sys.stdout.write("\r" + rendered + padding)
        sys.stdout.flush()
        wait_line_open = True
        wait_line_width = max(wait_line_width, len(rendered))

    def _event_eta(event: dict[str, object], *, done_now: int, total: int, elapsed: float) -> float | None:
        quota_limit = event.get("quota_limit")
        quota_reserve = event.get("quota_reserve")
        quota_wait_seconds = float(event.get("quota_wait_seconds") or event.get("waited_seconds") or 0.0)
        return _estimate_bootstrap_eta_seconds(
            done=done_now,
            initial_done=initial_done,
            total=total,
            elapsed_seconds=elapsed,
            quota_wait_seconds=quota_wait_seconds,
            quota_limit=int(quota_limit) if quota_limit is not None else None,
            quota_reserve=int(quota_reserve) if quota_reserve is not None else None,
            observed_sample_floor=progress_sample_floor,
        )

    def _progress(event: dict[str, object]) -> None:
        _close_wait_line()
        done_now = int(event.get("done") or 0)
        total = int(event.get("total") or 0)
        pct = 100.0 * done_now / total if total > 0 else 0.0
        suffix = " REUSE" if event.get("recovered") else " DONE"
        data_id = event.get("data_id")
        target = str(event.get("dataset") or "") + (f"/{data_id}" if data_id else "")
        elapsed = max(0.0, _monotonic_time.monotonic() - progress_started)
        quota_limit = event.get("quota_limit")
        eta = _event_eta(event, done_now=done_now, total=total, elapsed=elapsed)
        quota_remaining = event.get("quota_remaining")
        quota_usable = event.get("quota_usable_remaining")
        if quota_remaining is None or quota_limit is None:
            quota_text = "quota=--"
        else:
            quota_text = f"quota≈{int(quota_remaining)}/{int(quota_limit)}"
            if quota_usable is not None:
                quota_text += f"(可用≈{int(quota_usable)})"
        print(
            f"[Bootstrap] {done_now}/{total} ({pct:.1f}%)"
            f" | 已過 {_format_bootstrap_duration(elapsed)}"
            f" | ETA≈{_format_bootstrap_duration(eta)}"
            f" | {quota_text}"
            f" | {target}{suffix}"
        )

    def _quota_wait(event: dict[str, object]) -> None:
        done_now = int(event.get("done") or initial_done)
        total = int(event.get("total") or manifest.total_requests)
        pct = 100.0 * done_now / total if total > 0 else 0.0
        elapsed = max(0.0, _monotonic_time.monotonic() - progress_started)
        waited = max(0.0, float(event.get("waited_seconds") or 0.0))
        eta = _event_eta(event, done_now=done_now, total=total, elapsed=elapsed)
        poll_seconds = max(0.0, float(event.get("poll_seconds") or 0.0))
        quota_used = event.get("quota_user_count")
        quota_limit = event.get("quota_limit")
        resume_used_max = event.get("quota_resume_used_max")
        needed_drop = event.get("quota_needed_drop")
        if quota_used is not None and quota_limit is not None:
            quota_text = f"Q{int(quota_used)}/{int(quota_limit)}"
            if resume_used_max is not None:
                quota_text += f"→{int(resume_used_max)}"
            if needed_drop is not None and int(needed_drop) > 0:
                quota_text += f" 差{int(needed_drop)}"
        else:
            quota_text = "Q查詢失敗"
        reason = str(event.get("reason") or "quota_capacity")
        error = event.get("error")
        extra = ""
        if reason != "quota_capacity":
            extra += f" | {reason}"
        if error:
            extra += f" | {error}"
        _write_wait_line(
            f"[WAIT] {done_now}/{total} {pct:.1f}%"
            f" | 過{_format_bootstrap_duration(elapsed)}"
            f" 等{_format_bootstrap_duration(waited)}"
            f" | ETA{_format_bootstrap_duration(eta)}"
            f" | {quota_text}"
            f" | {int(round(poll_seconds))}s{extra}"
        )

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
            quota_wait_fn=_quota_wait,
        )
    except KeyboardInterrupt:
        _close_wait_line()
        print("⚠ Bootstrap 已由使用者中斷；已完成的 request/Parquet/ledger 會保留，下次選 [3] 可續傳。")
        return 130
    except (MarketDataBootstrapActivationError, FinMindHttpError, RuntimeError, ValueError, OSError, ImportError, ModuleNotFoundError) as exc:
        _close_wait_line()
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _close_wait_line()
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

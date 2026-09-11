import sys
import os
import importlib
import time as _monotonic_time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, _strip_ansi
from core.display_common import console_color_enabled
from core.path_utils import project_relative_display_path
from core.runtime_utils import (
    enable_line_buffered_stdout,
    has_help_flag,
    is_interactive_console,
    resolve_cli_program_name,
    run_cli_entrypoint,
    validate_cli_args,
)

_COLOR_ENABLED = console_color_enabled()


def _paint(value: object, color: str) -> str:
    text = str(value)
    return f"{color}{text}{C_RESET}" if _COLOR_ENABLED else text


def _status_color(status: object) -> str:
    normalized = str(status or "").strip().upper()
    if normalized in {"PASS", "READY", "UPDATED", "DONE", "AVAILABLE", "YES", "SYNCED", "COMPLETE"}:
        return C_GREEN
    if normalized in {"DEFERRED", "WAIT_PUBLISH", "WAIT_QUOTA", "UNVERIFIED", "NO", "NO_DUE", "TARGET_ADVANCED"}:
        return C_YELLOW
    if normalized in {"FAIL", "BLOCKED", "ERROR", "UNAVAILABLE", "STALE"}:
        return C_RED
    return C_CYAN


def _request_window_policy_label(row: dict[str, object]) -> str:
    query_mode = str(row.get("trading_query_mode") or "").strip()
    lookback = int(row.get("trading_lookback_periods") or 0)
    request_mode = str(row.get("request_mode") or "").strip()
    if query_mode and query_mode != "auto":
        return f"{query_mode} × {lookback}"
    request_labels = {
        "trading_incremental": "incremental from last READY",
        "trading_recent_repair": "recent repair window",
        "trading_event_repair": "event repair window",
        "trading_periodic_repair": "periodic repair",
        "trading_static_refresh": "static refresh",
    }
    return request_labels.get(request_mode, request_mode or "auto")


def _format_ready_count(ready: object, total: object) -> tuple[str, str]:
    ready_i = int(ready or 0)
    total_i = int(total or 0)
    color = C_GREEN if total_i > 0 and ready_i == total_i else C_YELLOW
    return f"{ready_i} / {total_i}", color


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



def _run_market_data_v2_daily_update(*, prompt_mode: bool = False) -> int:
    try:
        from services.trading.market_data_auto_update import run_trading_market_data_auto_update
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    force_refresh = False
    if prompt_mode:
        print("-" * 88)
        print(" Daily Update 模式")
        print("-" * 88)
        print("[Enter] 正常更新：只處理目前 due datasets")
        print("[R]     重新下載 current target：新 batch、禁止 artifact/cache REUSE，重新驗證")
        print("[0]     返回")
        while True:
            try:
                mode = input("模式: ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if mode == "":
                break
            if mode == "R":
                force_refresh = True
                break
            if mode == "0":
                return 0
            print("請按 Enter、輸入 R 或 0。")

    progress_line_open = False
    progress_line_width = 0

    def _close_progress_line() -> None:
        nonlocal progress_line_open, progress_line_width
        if progress_line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            progress_line_open = False
            progress_line_width = 0

    def _write_progress_line(text: str) -> None:
        nonlocal progress_line_open, progress_line_width
        rendered = str(text)
        visible_width = len(_strip_ansi(rendered))
        padding = " " * max(0, progress_line_width - visible_width)
        sys.stdout.write("\r" + rendered + padding)
        sys.stdout.flush()
        progress_line_open = True
        progress_line_width = max(progress_line_width, visible_width)

    def _progress(event: dict[str, object]) -> None:
        kind = str(event.get("kind") or "")
        if kind == "PLAN":
            _close_progress_line()
            mode = "FORCE REFRESH" if event.get("force_refresh") else "DUE ONLY"
            print(
                _paint("[Daily]", C_CYAN)
                + f" {mode} | target={event.get('target_date')}"
                f" | datasets={event.get('dataset_count')} | requests={event.get('total')}"
            )
            return
        if kind != "REQUEST_PROGRESS":
            return
        done = int(event.get("done") or 0)
        total = int(event.get("total") or 0)
        pct = (100.0 * done / total) if total else 0.0
        dataset = str(event.get("dataset") or "-")
        data_id = event.get("data_id")
        target = dataset + (f"/{data_id}" if data_id else "")
        start_date = event.get("start_date")
        end_date = event.get("end_date")
        if start_date and end_date:
            request_scope = f"date={start_date}" if start_date == end_date else f"date={start_date}~{end_date}"
        else:
            request_scope = "date=STATIC"
        phase = str(event.get("phase") or "RUN")
        if bool(event.get("recovered")):
            phase = "REUSE"
        data_used = int(event.get("process_data_requests") or 0)
        usage_used = int(event.get("process_usage_requests") or 0)
        q_used = event.get("quota_user_count")
        q_limit = event.get("quota_limit")
        quota = "quota=--" if q_used is None or q_limit is None else f"quota≈{int(q_used)}/{int(q_limit)}"
        rendered_phase = _paint(phase, _status_color(phase))
        _write_progress_line(
            f"{_paint('[Daily]', C_CYAN)} {done}/{total} ({pct:5.1f}%) | {target} | {request_scope} | {rendered_phase}"
            f" | data={data_used} usage={usage_used} | {quota}"
        )

    def _quota_wait(event: dict[str, object]) -> None:
        _close_progress_line()
        print(
            f"{_paint('[WAIT]', C_YELLOW)} {event.get('done')}/{event.get('total')}"
            f" | quota={event.get('quota_user_count') or '-'} / {event.get('quota_limit') or '-'}"
            f" | reason={event.get('reason') or 'quota'}"
        )

    try:
        result = run_trading_market_data_auto_update(
            project_root=PROJECT_ROOT,
            force_market_date_discovery=True,
            force_refresh_current_target=force_refresh,
            progress_fn=_progress,
            quota_wait_fn=_quota_wait,
        )
    except (RuntimeError, FileNotFoundError, ValueError, OSError, ImportError, ModuleNotFoundError) as exc:
        _close_progress_line()
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _close_progress_line()

    status = str(result.get("status") or "UNKNOWN")
    print(_paint("=" * 88, C_CYAN))
    print(_paint(" Market Data V2｜Daily Update", C_CYAN))
    print(_paint("=" * 88, C_CYAN))
    print(f"狀態                    : {_paint(status, _status_color(status))}")
    mode_label = "FORCE REFRESH" if result.get("force_refresh") else "DUE ONLY"
    print(f"模式                    : {_paint(mode_label, C_CYAN)}")
    print(f"V2 target date          : {_paint(result.get('target_date') or '-', C_CYAN)}")
    advanced = "YES" if result.get("v2_target_advanced") else "NO"
    print(f"新 completed day        : {_paint(advanced, _status_color(advanced))}")
    print(f"Due/selected datasets   : {result.get('due_dataset_count', 0)}")
    print(f"Logical requests        : {result.get('request_count', 0)}")
    print(f"Provider data requests  : {result.get('data_requests', 0)}")
    print(f"Provider usage requests : {result.get('usage_requests', 0)}")
    if result.get("request_date_start") or result.get("request_date_end"):
        print(
            "Request date window      : "
            f"{_paint(result.get('request_date_start') or '-', C_CYAN)} ~ {_paint(result.get('request_date_end') or '-', C_CYAN)}"
        )
        start_sources = list(result.get("request_date_start_sources") or [])
        if start_sources:
            grouped: dict[str, list[str]] = {}
            for row in start_sources:
                item = dict(row or {})
                grouped.setdefault(_request_window_policy_label(item), []).append(str(item.get("dataset") or "-"))
            parts = [
                f"{','.join(sorted(datasets))} [{label}]"
                for label, datasets in sorted(grouped.items())
            ]
            print(f"Window start policy     : {_paint('; '.join(parts), C_GRAY)}")
    print(f"Exact-date total        : {result.get('exact_date_request_count', 0)} requests")
    print(f"  Full-market exact     : {result.get('full_market_exact_date_request_count', 0)}")
    print(f"  Data-id exact         : {result.get('fixed_data_id_exact_date_request_count', 0)}")
    print(f"Unique exact dates      : {result.get('unique_exact_date_count', 0)}")
    print(f"Range requests          : {result.get('range_request_count', 0)}")
    print(f"Undated/static requests : {result.get('undated_request_count', 0)}")

    archive_ready = result.get("archive_ready_dataset_count", result.get("ready_dataset_count"))
    archive_total = result.get("archive_dataset_count", result.get("dataset_count"))
    if archive_ready is not None or archive_total is not None:
        ready_text, ready_color = _format_ready_count(archive_ready, archive_total)
        print(f"Target freshness READY  : {_paint(ready_text, ready_color)}")
    trading_ready = result.get("trading_required_ready_dataset_count")
    trading_total = result.get("trading_required_dataset_count")
    strategy_id = result.get("trading_strategy_id")
    if trading_ready is not None or trading_total is not None:
        ready_text, ready_color = _format_ready_count(trading_ready, trading_total)
        print(f"Trading target READY    : {_paint(ready_text, ready_color)} ({strategy_id or '-'})")
    schema_ready = result.get("schema_ready_dataset_count")
    coverage_ready = result.get("coverage_ready_dataset_count")
    validation_count = result.get("current_validation_dataset_count")
    if validation_count is not None:
        print(f"Current validation      : {_paint(f'{validation_count} datasets', C_GREEN if int(validation_count or 0) == int(archive_total or 0) else C_YELLOW)}")
        print(f"Canonical schema valid  : {_paint(f'{schema_ready or 0} datasets', C_GREEN if int(schema_ready or 0) == int(validation_count or 0) else C_YELLOW)}")
        print(f"Request coverage valid  : {_paint(f'{coverage_ready or 0} datasets', C_GREEN if int(coverage_ready or 0) == int(validation_count or 0) else C_YELLOW)}")

    verification = dict(result.get("verification") or {})
    if verification:
        print(_paint("-" * 88, C_CYAN))
        print(_paint(" Force-refresh verification（instrument comparison 是非阻擋 reference diagnostic）", C_CYAN))
        print(_paint("-" * 88, C_CYAN))
        print(f"Fresh batch observed    : {verification.get('observed_dataset_count', 0)} datasets")
        print(f"Schema observed         : {verification.get('schema_verified_dataset_count', 0)} datasets")
        ref_status = verification.get("current_stockinfo_reference_status") or "UNAVAILABLE"
        ref_count = int(verification.get("current_stockinfo_reference_count") or 0)
        print(f"StockInfo broad ref     : {_paint(ref_status, _status_color(ref_status))} ({ref_count} stock/ETF identities)")
        completeness = verification.get("instrument_completeness_status") or "UNVERIFIED"
        print(f"Instrument completeness : {_paint(completeness, _status_color(completeness))}")
        print(_paint("說明                    : 不同 dataset 的合法 instrument universe 不同；目前沒有 authoritative dataset-specific expected universe，因此不再用同一 StockInfo 集合產生假 MATCH/DIFF。", C_GRAY))

    archive_incomplete = tuple(result.get("archive_incomplete_datasets") or ())
    if archive_incomplete:
        print(_paint("Target freshness pending:", C_YELLOW))
        for item in archive_incomplete:
            row = dict(item or {})
            pending_line = (
                "  - "
                f"{row.get('dataset')}: status={row.get('status')}, "
                f"latest={row.get('latest_data_date') or '-'}, "
                f"schema={row.get('schema_status')}, coverage={row.get('coverage_status')}"
            )
            print(_paint(pending_line, _status_color(row.get("status"))))
            if row.get("last_error"):
                print(f"      reason: {row.get('last_error')}")

    if result.get("target_date"):
        try:
            from config.downloader import DOWNLOADER_MIN_MARKET_CAP, DOWNLOADER_MIN_VOLUME
            from core.trading_data_dependencies import get_trading_data_dependency_spec
            from core.trading_policy import get_trading_strategy_profile
            from services.trading.market_data_consumer import resolve_trading_v2_current_execution_pool
            from services.trading.market_data_v2_view import TradingMarketDataV2View

            local_view = TradingMarketDataV2View.open(PROJECT_ROOT)
            profile = get_trading_strategy_profile()
            dependency = get_trading_data_dependency_spec(profile.strategy_id)
            horizon = local_view.training_horizon(required_datasets=dependency.required_v2_datasets)
            target_date = str(result.get("target_date"))
            safe_date = min(target_date, str(horizon.training_through_date))
            _tickers, pool = resolve_trading_v2_current_execution_pool(
                local_view,
                market_date=safe_date,
            )
            print(_paint("-" * 88, C_CYAN))
            print(_paint(f" Trading execution pool｜{safe_date}", C_CYAN))
            print(_paint("-" * 88, C_CYAN))
            if safe_date < target_date:
                print(
                    "Trading safe horizon    : "
                    f"{_paint(safe_date, C_YELLOW)} "
                    f"(target {_paint(target_date, C_YELLOW)} 尚未由全部 required datasets 共同 READY)"
                )
            else:
                print(f"Trading safe horizon    : {_paint(safe_date, C_GREEN)}")
            broad = int(pool.get("stockinfo_broad_reference_count") or 0)
            listed = int(pool.get("listed_count") or 0)
            exact = int(pool.get("listed_with_exact_price_count") or 0)
            no_price = int(pool.get("listed_without_exact_price_count") or 0)
            high_volume = int(pool.get("high_volume_count") or 0)
            low_volume = int(pool.get("below_min_volume_count") or 0)
            etf_pass = int(pool.get("high_volume_etf_count") or 0)
            stock_gate = int(pool.get("high_volume_stock_count") or 0)
            cap_pass = int(pool.get("market_cap_pass_stock_count") or 0)
            cap_fail = int(pool.get("market_cap_below_min_stock_count") or 0)
            final_count = int(pool.get("qualified_count") or 0)
            print(f"StockInfo broad ref     : {_paint(broad, C_GRAY)}")
            print(f"PIT market members      : {_paint(listed, C_CYAN)}")
            print(f"Exact-date price        : {_paint(exact, C_CYAN)}  ({_paint(f'excluded {no_price}', C_YELLOW)})")
            print(f"Volume >= {int(DOWNLOADER_MIN_VOLUME):,}     : {_paint(high_volume, C_CYAN)}  ({_paint(f'excluded {low_volume}', C_YELLOW)})")
            print(f"  ETF pass              : {_paint(etf_pass, C_CYAN)}")
            print(f"  Stock to cap gate     : {_paint(stock_gate, C_CYAN)}")
            print(f"Stock cap >= {int(DOWNLOADER_MIN_MARKET_CAP):,}: {_paint(cap_pass, C_CYAN)}  ({_paint(f'excluded {cap_fail}', C_YELLOW)})")
            print(f"Final execution pool    : {_paint(final_count, C_GREEN)}")
        except (RuntimeError, FileNotFoundError, ValueError, OSError, ImportError, ModuleNotFoundError) as exc:
            print(_paint("-" * 88, C_CYAN))
            print(_paint(" Trading execution pool", C_CYAN))
            print(_paint("-" * 88, C_CYAN))
            print(f"狀態                    : {_paint(f'UNAVAILABLE ({type(exc).__name__}: {exc})', C_RED)}")

    blockers = tuple(result.get("trading_blocking_datasets") or ())
    if blockers:
        pending_status = {
            str(dict(item or {}).get("dataset") or ""): str(dict(item or {}).get("status") or "")
            for item in archive_incomplete
        }
        blocker_color = C_RED if any(
            pending_status.get(str(dataset), "").upper() in {"BLOCKED", "ERROR", "STALE"}
            for dataset in blockers
        ) else C_YELLOW
        print(f"Trading target pending  : {_paint(', '.join(blockers), blocker_color)}")
    if result.get("next_check_at"):
        print(f"Next check              : {_paint(result.get('next_check_at'), C_CYAN)}")
    if result.get("error"):
        print(f"錯誤                    : {_paint(result.get('error'), C_RED)}")
    if status == "NO_TARGET":
        print("說明                    : 尚無 READY Provider Snapshot；請先完成 [2] → [3] → [4]。")
        return 1
    return 1 if status == "BLOCKED" else 0


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
    print(_paint("=" * 88, C_CYAN))
    print(_paint(" Market Data V2｜Backer Preflight + Exact Bootstrap Planner", C_CYAN))
    print(_paint("=" * 88, C_CYAN))
    print(f"狀態                 : {_paint(status, _status_color(status))}")
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
        print(f"Blocking probes         : {_paint(len(failures), C_RED)}")
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
        print(_paint(f"⚠ quota accounting: {warning}", C_YELLOW))
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
    print(_paint("=" * 88, C_CYAN))
    print(_paint(" Market Data V2｜完整 Bootstrap 開始 / 續傳", C_CYAN))
    print(_paint("=" * 88, C_CYAN))
    print(f"Preflight             : {project_relative_display_path(activation.preflight_path, project_root=PROJECT_ROOT)}")
    print(f"資料截止              : {manifest.as_of_date}")
    print(f"Historical instruments: {manifest.historical_instrument_count}")
    print(f"Logical requests      : {manifest.total_requests}")
    print(f"已完成 / 剩餘         : {done} / {manifest.total_requests - done}")
    print(f"Manifest fingerprint  : {manifest.manifest_fingerprint}")
    print(f"Live quota            : {usage.user_count} / {usage.api_request_limit}")
    print(_paint("說明                  : 啟動後會跨 quota window 自動等待並續傳；Ctrl+C 可安全中斷後再次由本選項續傳。", C_GRAY))
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
        visible_width = len(_strip_ansi(rendered))
        padding = " " * max(0, wait_line_width - visible_width)
        sys.stdout.write("\r" + rendered + padding)
        sys.stdout.flush()
        wait_line_open = True
        wait_line_width = max(wait_line_width, visible_width)

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
        suffix = _paint(" REUSE", C_CYAN) if event.get("recovered") else _paint(" DONE", C_GREEN)
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
            f"{_paint('[Bootstrap]', C_CYAN)} {done_now}/{total} ({pct:.1f}%)"
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
            f"{_paint('[WAIT]', C_YELLOW)} {done_now}/{total} {pct:.1f}%"
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
        print(_paint("⚠ Bootstrap 已由使用者中斷；已完成的 request/Parquet/ledger 會保留，下次選 [3] 可續傳。", C_YELLOW))
        return 130
    except (MarketDataBootstrapActivationError, FinMindHttpError, RuntimeError, ValueError, OSError, ImportError, ModuleNotFoundError) as exc:
        _close_wait_line()
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _close_wait_line()
    print(_paint("=" * 88, C_CYAN))
    print(_paint(" Market Data V2｜Bootstrap 執行結果", C_CYAN))
    print(_paint("=" * 88, C_CYAN))
    bootstrap_status = result.get("status")
    print(f"狀態                 : {_paint(bootstrap_status, _status_color(bootstrap_status))}")
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
    print(_paint("Bootstrap requests 已全部 DONE；開始本機完整性驗證並建立 immutable provider snapshot（不會再打 FinMind data API）。", C_GREEN))
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

    print(_paint("=" * 88, C_CYAN))
    print(_paint(" Market Data V2｜Provider Snapshot Finalization", C_CYAN))
    print(_paint("=" * 88, C_CYAN))
    snapshot_status = result.get("status")
    print(f"狀態                 : {_paint(snapshot_status, _status_color(snapshot_status))}")
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
    print(_paint("說明                 : 這只是 neutral provider source READY；Research V2 / Trading 尚未因此自動切換。", C_GRAY))
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


def _run_market_data_v2_full_integrity_audit() -> int:
    try:
        from services.downloader.market_data_integrity_audit import run_market_data_v2_full_integrity_audit
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    def _progress(event: dict[str, object]) -> None:
        phase = str(event.get("phase") or "VERIFY")
        verified = int(event.get("verified") or 0)
        total = event.get("total")
        target = str(event.get("dataset") or "-")
        data_id = event.get("data_id")
        if data_id:
            target += f"/{data_id}"
        if total is None:
            print(f"[Integrity:{phase}] {verified} | {target}")
        else:
            print(f"[Integrity:{phase}] {verified}/{int(total)} | {target}")

    try:
        result = run_market_data_v2_full_integrity_audit(
            project_root=PROJECT_ROOT,
            progress_fn=_progress,
        )
    except (RuntimeError, FileNotFoundError, ValueError, OSError, ImportError, ModuleNotFoundError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    validation = dict(result.get("dataset_validation") or {})
    print(_paint("=" * 88, C_CYAN))
    print(_paint(" Market Data V2｜Full Database Integrity（Local-only / no API quota）", C_CYAN))
    print(_paint("=" * 88, C_CYAN))
    integrity_status = result.get("status")
    print(f"Local canonical integrity : {_paint(integrity_status, _status_color(integrity_status))}")
    print(f"Provider Snapshot          : {result.get('provider_snapshot_status')} | requests={result.get('provider_requests_verified')} rows={result.get('provider_rows_verified')}")
    print(f"Trading DONE overlays      : {result.get('overlay_status')} | batches={result.get('overlay_done_batches_verified')} requests={result.get('overlay_requests_verified')} rows={result.get('overlay_rows_verified')}")
    print(f"Incomplete overlay batches : {result.get('overlay_incomplete_batches_ignored')} (resumable, not part of canonical read view)")
    print(f"Validation target          : {result.get('target_date')}")
    print(f"Dataset registry/state     : {validation.get('state_dataset_count', 0)} / {validation.get('dataset_count', 0)}")
    print(f"Current validation         : {validation.get('current_validation_count', 0)} / {validation.get('dataset_count', 0)}")
    print(f"Canonical schema valid     : {validation.get('schema_valid_count', 0)} / {validation.get('dataset_count', 0)}")
    print(f"Request coverage valid     : {validation.get('coverage_valid_count', 0)} / {validation.get('dataset_count', 0)}")
    print(f"Dataset READY              : {validation.get('ready_count', 0)} / {validation.get('dataset_count', 0)}")
    absolute_status = result.get("absolute_instrument_completeness")
    print(f"Absolute completeness      : {_paint(absolute_status, _status_color(absolute_status))}")
    print(_paint("Reason                     : provider 未提供每個 dataset 的 authoritative expected instrument universe；不可把 StockInfo broad ref 當所有 feed 的應有集合。", C_GRAY))
    print(f"Provider API requests      : {result.get('provider_requests_made', 0)}")
    return 0 if str(result.get("status")) == "PASS" else 1


def _interactive_menu() -> int:
    print(_paint("=" * 72, C_CYAN))
    print(_paint(" Smart Downloader", C_CYAN))
    print(_paint("=" * 72, C_CYAN))
    print(f"{_paint('[1]', C_GREEN)} Market Data V2｜Daily Update（Canonical：dataset × date bulk）")
    print(f"{_paint('[2]', C_CYAN)} Market Data V2｜Backer Preflight + Exact Bootstrap Plan")
    print(f"{_paint('[3]', C_CYAN)} Market Data V2｜開始 / 續傳完整 Bootstrap")
    print(f"{_paint('[4]', C_CYAN)} Market Data V2｜驗證 / 重建 Provider Snapshot（不使用 API quota）")
    print(f"{_paint('[5]', C_CYAN)} Market Data V2｜Full Database Integrity Audit（不使用 API quota）")
    print(f"{_paint('[0]', C_GRAY)} 離開")
    while True:
        try:
            choice = input("請選擇: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "1":
            return _run_market_data_v2_daily_update(prompt_mode=True)
        if choice == "2":
            return _run_market_data_v2_preflight()
        if choice == "3":
            return _run_market_data_v2_bootstrap()
        if choice == "4":
            return _run_market_data_v2_provider_snapshot_finalize()
        if choice == "5":
            return _run_market_data_v2_full_integrity_audit()
        if choice == "0":
            return 0
        print("請輸入 0、1、2、3、4 或 5。")


def main(argv=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv)
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "services/downloader/main.py")
        print(f"用法: python {program_name}")
        print("說明: [1] 為 Market Data V2 canonical Daily Update；[2]-[4] 提供 Backer Preflight、完整 Bootstrap 與 Provider Snapshot 驗證；[5] 執行全庫本機完整性稽核。")
        print("非互動環境直接執行 Market Data V2 Daily Update；不再由 Smart Downloader 先更新 Legacy Trading CSV。")
        return 0

    if len(argv) == 1 and is_interactive_console():
        return _interactive_menu()
    return _run_market_data_v2_daily_update()


if __name__ == "__main__":
    run_cli_entrypoint(main)

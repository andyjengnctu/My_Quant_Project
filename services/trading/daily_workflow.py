"""Trading daily Workbench workflow using canonical data/params/scanner producers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.file_integrity import atomic_write_json, compute_file_sha256
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_output_dir
from core.breakout_reentry import BREAKOUT_REENTRY_SOURCE
from core.portfolio_ensemble import (
    annotate_ensemble_candidate,
    aggregate_ensemble_candidate_rows,
    build_ensemble_candidate_display_metrics,
    sort_aggregated_ensemble_candidate_rows,
)
from services.trading.market_data_consumer import build_trading_v2_ohlcv_frame
from services.trading.market_data_update import run_trading_market_data_update
from services.trading.market_data_v2_view import TradingMarketDataV2View
from services.trading.live_reentry import build_trading_live_reentry_candidate_rows
from services.scanner.scan_runner import run_daily_scanner
from services.trading.scanner_state import (
    TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION,
    build_trading_daily_workflow_snapshot,
    get_trading_candidate_snapshot_read_model,
    load_trading_candidate_snapshot,
    load_trading_scanner_runtime,
    partition_trading_candidate_rows_for_information_date,
    resolve_trading_candidate_snapshot_path,
)
from services.trading.strategy_param_runtime import serialize_trading_candidate_member_params
from services.trading.strategy_param_training import (
    reuse_trading_strategy_params,
    run_trading_strategy_param_training,
)


TRADING_PARAM_MODE_REUSE = "reuse"
TRADING_PARAM_MODE_TRAIN = "train"
TRADING_PARAM_MODES = frozenset({TRADING_PARAM_MODE_REUSE, TRADING_PARAM_MODE_TRAIN})


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, 'item', None)
    if callable(item):
        return _json_safe(item())
    isoformat = getattr(value, 'isoformat', None)
    if callable(isoformat):
        return isoformat()
    return str(value)

def _persistable_trading_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    member_params = serialize_trading_candidate_member_params(payload)
    payload["ensemble_member_params_by_key"] = member_params
    payload.pop("params_obj", None)
    payload.pop("_ensemble_context", None)
    return _json_safe(payload)


def _run_trading_scanner_param_runtime(
    *,
    runtime: dict[str, Any],
    output_dir,
    expected_scanned_tickers: list[str],
    prepared_frames: dict[str, Any],
) -> dict[str, Any]:
    members = list(runtime.get("param_members") or [])
    if not members:
        raise RuntimeError("Trading Scanner Params runtime 沒有可用 member")

    member_results: list[dict[str, Any]] = []
    annotated_rows: list[dict[str, Any]] = []
    for member in members:
        result = run_daily_scanner(
            str(runtime["data_dir"]),
            member["params_obj"],
            output_dir=output_dir,
            include_execution_context=True,
            ticker_membership=expected_scanned_tickers,
            prepared_frames=prepared_frames,
        )
        actual_scanned_tickers = list(result.get("scanned_tickers") or [])
        if actual_scanned_tickers != expected_scanned_tickers:
            raise RuntimeError(
                "Trading Scanner 實際掃描 membership 與 canonical current universe 不一致；"
                f"expected={expected_scanned_tickers[:20]}, actual={actual_scanned_tickers[:20]}"
            )
        member_results.append(dict(result))
        member_key = str(member.get("member_key") or member.get("member_index") or member.get("seed") or "")
        for raw_row in list(result.get("candidate_rows") or []):
            seed = raw_row.get("execution_plan_seed") if isinstance(raw_row, dict) else None
            entry_source = str((seed or {}).get("entry_source") or raw_row.get("entry_source") or "").strip()
            if entry_source == BREAKOUT_REENTRY_SOURCE:
                continue
            annotated_rows.append(
                annotate_ensemble_candidate(
                    raw_row,
                    member=member,
                    params_obj=member["params_obj"],
                    member_key=member_key,
                )
            )

    min_agree = int(runtime.get("param_min_agree") or 1)
    if len(members) == 1:
        candidate_rows = list(annotated_rows)
        for row in candidate_rows:
            member_key = str(row.get("ensemble_member_key") or "1")
            row["ensemble_vote_count"] = 1
            row["ensemble_min_agree"] = 1
            row["ensemble_member_count"] = 1
            row["ensemble_member_keys"] = [member_key]
            row["ensemble_member_params_by_key"] = {member_key: row.get("params_obj")}
    else:
        candidate_rows = aggregate_ensemble_candidate_rows(annotated_rows, min_agree=min_agree)

    return {
        "count_scanned": len(expected_scanned_tickers),
        "elapsed_time": float(sum(float(item.get("elapsed_time") or 0.0) for item in member_results)),
        "count_history_qualified": len(candidate_rows),
        "count_skipped_insufficient": int(sum(int(item.get("count_skipped_insufficient") or 0) for item in member_results)),
        "count_sanitized_candidates": int(sum(int(item.get("count_sanitized_candidates") or 0) for item in member_results)),
        "max_workers": max((int(item.get("max_workers") or 0) for item in member_results), default=0),
        "pool_start_method": next((item.get("pool_start_method") for item in member_results if item.get("pool_start_method")), None),
        "scanned_tickers": list(expected_scanned_tickers),
        "candidate_rows": candidate_rows,
        "scanner_issue_log_path": None,
        "param_member_count": len(members),
        "param_min_agree": min_agree,
    }


def run_trading_candidate_scan(*, project_root: str | Path) -> dict[str, Any]:
    runtime = load_trading_scanner_runtime(project_root, verify_dataset_content=True)
    root = runtime["root"]
    output_dir = resolve_runtime_output_dir(
        root, domain=RUNTIME_DOMAIN_TRADING, category="scanner"
    )
    expected_scanned_tickers = list(runtime.get("current_universe_tickers") or [])
    if not expected_scanned_tickers:
        raise RuntimeError("Trading Scanner 缺少 canonical current universe membership；請重新更新 Trading 資料")
    v2_view = TradingMarketDataV2View.open_as_of_target(
        root, target_date=str(runtime["latest_data_date"])
    )
    prepared_frames = {
        ticker: build_trading_v2_ohlcv_frame(
            v2_view, ticker=ticker, through_date=str(runtime["latest_data_date"])
        )
        for ticker in expected_scanned_tickers
    }
    result = _run_trading_scanner_param_runtime(
        runtime=runtime,
        output_dir=output_dir,
        expected_scanned_tickers=expected_scanned_tickers,
        prepared_frames=prepared_frames,
    )
    current_rows = list(result.get("candidate_rows") or [])
    current_tickers = {str(row.get("ticker") or "") for row in current_rows}
    live_reentry_rows = build_trading_live_reentry_candidate_rows(
        root,
        information_date=str(runtime["latest_data_date"]),
        excluded_tickers=current_tickers,
    )
    result["candidate_rows"] = sort_aggregated_ensemble_candidate_rows(current_rows + live_reentry_rows)
    result["count_history_qualified"] = len(result["candidate_rows"])
    result["live_reentry_candidate_count"] = len(live_reentry_rows)
    live_reentry_tickers = {str(row.get("ticker") or "") for row in live_reentry_rows}
    actual_scanned_tickers = sorted(set(result.get("scanned_tickers") or []) | live_reentry_tickers)
    result["scanned_tickers"] = actual_scanned_tickers
    runtime_after = load_trading_scanner_runtime(project_root, verify_dataset_content=True)
    stable_fields = (
        "latest_data_date",
        "selected_params_sha256",
        "market_data_consumer_state_sha256",
        "market_data_source_view_fingerprint",
        "param_binding_sha256",
        "member_count",
        "param_min_agree",
    )
    changed = [field for field in stable_fields if str(runtime_after.get(field) or "") != str(runtime.get(field) or "")]
    if changed:
        raise RuntimeError(
            "Trading Scanner 執行期間 inputs 已變更，禁止發布混合 lineage candidate snapshot: " + ",".join(changed)
        )
    raw_candidate_rows = [_persistable_trading_candidate_row(dict(row)) for row in list(result.get('candidate_rows') or [])]
    candidate_rows, stale_candidate_rows = partition_trading_candidate_rows_for_information_date(
        raw_candidate_rows,
        information_date=runtime['latest_data_date'],
    )
    candidate_display_metrics = build_ensemble_candidate_display_metrics(
        total_member_count=runtime['member_count']
    )
    snapshot_path = resolve_trading_candidate_snapshot_path(root)
    snapshot_payload = {
        'schema_version': TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION,
        'runtime_domain': RUNTIME_DOMAIN_TRADING,
        'strategy_id': runtime['profile'].strategy_id,
        'param_selector': runtime['profile'].param_selector,
        'latest_data_date': runtime['latest_data_date'],
        'param_latest_data_date': runtime['param_latest_data_date'],
        'param_member_count': runtime['member_count'],
        'param_min_agree': runtime['param_min_agree'],
        'candidate_display_metrics': candidate_display_metrics,
        'selected_params_path': project_relative_display_path(runtime['selected_path'], project_root=root),
        'selected_params_sha256': runtime['selected_params_sha256'],
        'market_data_consumer_state_sha256': runtime['market_data_consumer_state_sha256'],
        'market_data_source_view_fingerprint': runtime['market_data_source_view_fingerprint'],
        'param_binding_sha256': runtime['param_binding_sha256'],
        'scanned_tickers': list(actual_scanned_tickers),
        'candidate_rows': candidate_rows,
        'stale_candidate_rows_skipped': stale_candidate_rows,
    }
    atomic_write_json(snapshot_path, snapshot_payload)
    return {
        **dict(result),
        'candidate_rows': candidate_rows,
        'stale_candidate_rows_skipped': stale_candidate_rows,
        'stale_candidate_count': len(stale_candidate_rows),
        "status": "READY",
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "strategy_id": runtime["profile"].strategy_id,
        "param_selector": runtime["profile"].param_selector,
        "latest_data_date": runtime["latest_data_date"],
        "param_latest_data_date": runtime["param_latest_data_date"],
        "param_member_count": runtime["member_count"],
        "param_min_agree": runtime["param_min_agree"],
        "candidate_display_metrics": candidate_display_metrics,
        "market_data_consumer_state_sha256": runtime["market_data_consumer_state_sha256"],
        "market_data_source_view_fingerprint": runtime["market_data_source_view_fingerprint"],
        "param_binding_sha256": runtime["param_binding_sha256"],
        "selected_params_path": project_relative_display_path(runtime["selected_path"], project_root=root),
        "scanner_output_dir": project_relative_display_path(output_dir, project_root=root),
        "candidate_snapshot_path": project_relative_display_path(snapshot_path, project_root=root),
        "candidate_snapshot_sha256": compute_file_sha256(snapshot_path),
    }


def run_trading_param_step(*, project_root: str | Path, mode: str, environ=None) -> dict[str, Any]:
    normalized = str(mode or "").strip().lower()
    if normalized == TRADING_PARAM_MODE_REUSE:
        return reuse_trading_strategy_params(project_root=project_root)
    if normalized == TRADING_PARAM_MODE_TRAIN:
        return run_trading_strategy_param_training(project_root=project_root, environ=environ)
    raise ValueError(f"不支援的 Trading Params 模式: {mode!r}")


def run_trading_daily_workflow(
    *,
    project_root: str | Path,
    environ=None,
    param_mode: str = TRADING_PARAM_MODE_TRAIN,
    data_progress_fn=None,
    data_quota_wait_fn=None,
) -> dict[str, Any]:
    from services.trading.indicator_exit_planning import build_trading_indicator_exit_plan
    from services.trading.position_rollforward import run_trading_position_rollforward

    data_result = run_trading_market_data_update(
        project_root=project_root,
        progress_fn=data_progress_fn,
        quota_wait_fn=data_quota_wait_fn,
    )
    rollforward_result = run_trading_position_rollforward(project_root=project_root)
    if str(rollforward_result.get("status") or "") == "NO_ACCOUNT":
        indicator_result = {"status": "NO_ACCOUNT", "exit_count": 0, "exits": []}
    else:
        indicator_result = build_trading_indicator_exit_plan(project_root=project_root)
    param_result = run_trading_param_step(
        project_root=project_root,
        mode=param_mode,
        environ=environ,
    )
    scan_result = run_trading_candidate_scan(project_root=project_root)
    return {
        "status": "READY",
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "data": data_result,
        "position_rollforward": rollforward_result,
        "indicator_exit": indicator_result,
        "params": param_result,
        "scanner": scan_result,
    }


__all__ = [
    "TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION",
    "build_trading_daily_workflow_snapshot",
    "load_trading_candidate_snapshot",
    "get_trading_candidate_snapshot_read_model",
    "load_trading_scanner_runtime",
    "resolve_trading_candidate_snapshot_path",
    "run_trading_market_data_update",
    "run_trading_candidate_scan",
    "run_trading_param_step",
    "run_trading_daily_workflow",
    "TRADING_PARAM_MODE_REUSE",
    "TRADING_PARAM_MODE_TRAIN",
    "TRADING_PARAM_MODES",
]

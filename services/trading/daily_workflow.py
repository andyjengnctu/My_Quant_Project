"""Trading daily Workbench workflow using canonical data/params/scanner producers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.file_integrity import atomic_write_json, compute_file_sha256
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths, resolve_runtime_output_dir
from core.trading_identity import normalize_trading_ticker
from services.downloader.application import run_trading_dataset_update
from services.trading.account_state import load_trading_account_state
from services.trading.market_data_state import publish_trading_market_data_snapshot
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
from services.trading.strategy_param_training import run_trading_strategy_param_training


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

def run_trading_market_data_update(*, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING)
    from services.downloader import runtime as downloader_runtime

    if Path(downloader_runtime.SAVE_DIR).resolve() != Path(paths.data_dir).resolve():
        raise RuntimeError("Trading downloader SAVE_DIR與runtime-domain data truth不一致")
    account = load_trading_account_state(root, required=False)
    required_position_tickers = sorted(
        normalize_trading_ticker(ticker)
        for ticker, record in ((account or {}).get("positions") or {}).items()
        if int(((record or {}).get("broker") or {}).get("qty") or 0) > 0
    )
    result = dict(run_trading_dataset_update(required_tickers=required_position_tickers))
    if str(result.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise RuntimeError("Trading downloader回傳的runtime domain不合法")
    snapshot = publish_trading_market_data_snapshot(
        root,
        market_date=result.get("market_date"),
        required_position_tickers=required_position_tickers,
    )
    return {
        **result,
        "market_data_snapshot_fingerprint": snapshot["snapshot_fingerprint"],
        "dataset_content_sha256": snapshot["dataset_fingerprint"]["csv_content_sha256"],
    }


def run_trading_candidate_scan(*, project_root: str | Path) -> dict[str, Any]:
    runtime = load_trading_scanner_runtime(project_root, verify_dataset_content=True)
    root = runtime["root"]
    output_dir = resolve_runtime_output_dir(
        root, domain=RUNTIME_DOMAIN_TRADING, category="scanner"
    )
    result = run_daily_scanner(
        str(runtime["data_dir"]),
        runtime["params"],
        output_dir=output_dir,
        include_execution_context=True,
    )
    runtime_after = load_trading_scanner_runtime(project_root, verify_dataset_content=True)
    stable_fields = (
        "latest_data_date", "selected_params_sha256", "dataset_content_sha256", "param_binding_sha256",
    )
    changed = [field for field in stable_fields if str(runtime_after.get(field) or "") != str(runtime.get(field) or "")]
    if changed:
        raise RuntimeError(
            "Trading Scanner 執行期間 inputs 已變更，禁止發布混合 lineage candidate snapshot: " + ",".join(changed)
        )
    runtime_after = load_trading_scanner_runtime(project_root, verify_dataset_content=True)
    stable_fields = (
        "latest_data_date", "selected_params_sha256", "market_data_snapshot_sha256",
        "dataset_content_sha256", "param_binding_sha256",
    )
    changed = [field for field in stable_fields if str(runtime_after.get(field) or "") != str(runtime.get(field) or "")]
    if changed:
        raise RuntimeError(
            "Trading Scanner 執行期間 inputs 已變更，禁止發布混合 lineage candidate snapshot: " + ",".join(changed)
        )
    raw_candidate_rows = [_json_safe(dict(row)) for row in list(result.get('candidate_rows') or [])]
    candidate_rows, stale_candidate_rows = partition_trading_candidate_rows_for_information_date(
        raw_candidate_rows,
        information_date=runtime['latest_data_date'],
    )
    snapshot_path = resolve_trading_candidate_snapshot_path(root)
    snapshot_payload = {
        'schema_version': TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION,
        'runtime_domain': RUNTIME_DOMAIN_TRADING,
        'strategy_id': runtime['profile'].strategy_id,
        'param_selector': runtime['profile'].param_selector,
        'latest_data_date': runtime['latest_data_date'],
        'param_latest_data_date': runtime['param_latest_data_date'],
        'selected_params_path': project_relative_display_path(runtime['selected_path'], project_root=root),
        'selected_params_sha256': runtime['selected_params_sha256'],
        'market_data_snapshot_sha256': runtime['market_data_snapshot_sha256'],
        'dataset_content_sha256': runtime['dataset_content_sha256'],
        'param_binding_sha256': runtime['param_binding_sha256'],
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
        "market_data_snapshot_sha256": runtime["market_data_snapshot_sha256"],
        "dataset_content_sha256": runtime["dataset_content_sha256"],
        "param_binding_sha256": runtime["param_binding_sha256"],
        "market_data_snapshot_sha256": runtime["market_data_snapshot_sha256"],
        "dataset_content_sha256": runtime["dataset_content_sha256"],
        "param_binding_sha256": runtime["param_binding_sha256"],
        "selected_params_path": project_relative_display_path(runtime["selected_path"], project_root=root),
        "scanner_output_dir": project_relative_display_path(output_dir, project_root=root),
        "candidate_snapshot_path": project_relative_display_path(snapshot_path, project_root=root),
        "candidate_snapshot_sha256": compute_file_sha256(snapshot_path),
    }


def run_trading_daily_workflow(*, project_root: str | Path, environ=None) -> dict[str, Any]:
    from services.trading.indicator_exit_planning import build_trading_indicator_exit_plan
    from services.trading.position_rollforward import run_trading_position_rollforward

    data_result = run_trading_market_data_update(project_root=project_root)
    rollforward_result = run_trading_position_rollforward(project_root=project_root)
    if str(rollforward_result.get("status") or "") == "NO_ACCOUNT":
        indicator_result = {"status": "NO_ACCOUNT", "exit_count": 0, "exits": []}
    else:
        indicator_result = build_trading_indicator_exit_plan(project_root=project_root)
    param_result = run_trading_strategy_param_training(project_root=project_root, environ=environ)
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
    "run_trading_daily_workflow",
]

"""Trading daily Workbench workflow using canonical data/params/scanner producers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.dataset_dates import resolve_latest_dataset_date
from core.file_integrity import load_json_strict
from core.portfolio_param_runtime import load_portfolio_param_source_from_json
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths, resolve_runtime_output_dir
from core.strategy_param_artifacts import resolve_strategy_param_manifest_path
from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
from services.downloader.application import run_trading_dataset_update
from services.scanner.scan_runner import run_daily_scanner
from services.trading.strategy_param_training import run_trading_strategy_param_training


def _selected_payload_latest_data_date(payload: dict[str, Any]) -> str:
    meta = dict(payload.get("meta") or {})
    policy = dict(meta.get("walk_forward_policy") or {})
    return str(policy.get("latest_data_date") or "").strip()


def _safe_latest_dataset_date(data_dir: Path) -> str | None:
    if not data_dir.is_dir():
        return None
    try:
        return resolve_latest_dataset_date(data_dir)
    except (OSError, ValueError):
        return None


def _load_trading_scanner_runtime(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    paths = resolve_runtime_domain_paths(
        root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile
    )
    data_dir = Path(paths.data_dir)
    latest_data_date = resolve_latest_dataset_date(data_dir)
    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    if not selected_path.is_file():
        raise FileNotFoundError(
            "Trading strategy params尚未產生；請先在Workbench執行「更新 Trading Params」。"
        )

    payload = load_json_strict(selected_path)
    if not isinstance(payload, dict):
        raise RuntimeError("Trading selected strategy params payload不合法")
    if str(payload.get("selector") or "").strip() != str(profile.param_selector):
        raise RuntimeError(
            "Trading selected strategy params selector與目前Trading設定不一致；請重新訓練。"
        )

    param_source = load_portfolio_param_source_from_json(selected_path)
    member_count = int(param_source.get("member_count") or 0)
    if member_count != 1:
        raise RuntimeError(
            "Trading Scanner目前要求selector解析成單一參數member；"
            f"目前artifact共有 {member_count} members，禁止靜默只取第一組。"
        )

    param_latest_data_date = _selected_payload_latest_data_date(payload)
    if param_latest_data_date != latest_data_date:
        raise RuntimeError(
            "Trading strategy params不是目前最新Trading資料的產物；"
            f"data={latest_data_date}, params={param_latest_data_date or '-'}。"
            "請先執行「更新 Trading Params」。"
        )

    return {
        "root": root,
        "profile": profile,
        "paths": paths,
        "data_dir": data_dir,
        "latest_data_date": latest_data_date,
        "selected_path": selected_path,
        "param_latest_data_date": param_latest_data_date,
        "member_count": member_count,
        "params": param_source["primary_params"],
    }


def build_trading_daily_workflow_snapshot(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    paths = resolve_runtime_domain_paths(
        root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile
    )
    data_dir = Path(paths.data_dir)
    latest_data_date = _safe_latest_dataset_date(data_dir)
    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    param_latest_data_date = ""
    member_count = 0
    param_error = ""
    if selected_path.is_file():
        try:
            payload = load_json_strict(selected_path)
            if not isinstance(payload, dict):
                raise ValueError("selected params root必須是object")
            param_latest_data_date = _selected_payload_latest_data_date(payload)
            param_source = load_portfolio_param_source_from_json(selected_path)
            member_count = int(param_source.get("member_count") or 0)
            if str(payload.get("selector") or "").strip() != str(profile.param_selector):
                param_error = "selector mismatch"
        except (OSError, ValueError, RuntimeError) as exc:
            param_error = f"{type(exc).__name__}: {exc}"
    params_ready = bool(
        latest_data_date
        and selected_path.is_file()
        and not param_error
        and member_count == 1
        and param_latest_data_date == latest_data_date
    )
    return {
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "strategy_id": profile.strategy_id,
        "param_selector": profile.param_selector,
        "data_dir": project_relative_display_path(data_dir, project_root=root),
        "latest_data_date": latest_data_date,
        "selected_params_path": project_relative_display_path(selected_path, project_root=root),
        "selected_params_exists": selected_path.is_file(),
        "param_latest_data_date": param_latest_data_date or None,
        "param_member_count": member_count,
        "param_error": param_error or None,
        "params_ready_for_scan": params_ready,
        "scanner_output_dir": project_relative_display_path(
            resolve_runtime_output_dir(root, domain=RUNTIME_DOMAIN_TRADING, category="scanner"),
            project_root=root,
        ),
    }


def run_trading_market_data_update(*, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING)
    from services.downloader import runtime as downloader_runtime

    if Path(downloader_runtime.SAVE_DIR).resolve() != Path(paths.data_dir).resolve():
        raise RuntimeError("Trading downloader SAVE_DIR與runtime-domain data truth不一致")
    result = dict(run_trading_dataset_update())
    if str(result.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise RuntimeError("Trading downloader回傳的runtime domain不合法")
    return result


def run_trading_candidate_scan(*, project_root: str | Path) -> dict[str, Any]:
    runtime = _load_trading_scanner_runtime(project_root)
    root = runtime["root"]
    output_dir = resolve_runtime_output_dir(
        root, domain=RUNTIME_DOMAIN_TRADING, category="scanner"
    )
    result = run_daily_scanner(
        str(runtime["data_dir"]),
        runtime["params"],
        output_dir=output_dir,
    )
    return {
        **dict(result),
        "status": "READY",
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "strategy_id": runtime["profile"].strategy_id,
        "param_selector": runtime["profile"].param_selector,
        "latest_data_date": runtime["latest_data_date"],
        "param_latest_data_date": runtime["param_latest_data_date"],
        "param_member_count": runtime["member_count"],
        "selected_params_path": project_relative_display_path(runtime["selected_path"], project_root=root),
        "scanner_output_dir": project_relative_display_path(output_dir, project_root=root),
    }


def run_trading_daily_workflow(*, project_root: str | Path, environ=None) -> dict[str, Any]:
    data_result = run_trading_market_data_update(project_root=project_root)
    param_result = run_trading_strategy_param_training(project_root=project_root, environ=environ)
    scan_result = run_trading_candidate_scan(project_root=project_root)
    return {
        "status": "READY",
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "data": data_result,
        "params": param_result,
        "scanner": scan_result,
    }


__all__ = [
    "build_trading_daily_workflow_snapshot",
    "run_trading_market_data_update",
    "run_trading_candidate_scan",
    "run_trading_daily_workflow",
]

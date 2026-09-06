"""Canonical Trading scanner runtime and candidate-snapshot state helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.dataset_dates import resolve_latest_dataset_date
from core.file_integrity import compute_file_sha256, load_json_strict
from core.portfolio_param_runtime import load_portfolio_param_source_from_json
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths, resolve_runtime_output_dir
from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from services.trading.market_data_state import (
    get_trading_market_data_snapshot_sha256,
    load_trading_market_data_snapshot,
)
from services.trading.strategy_param_state import (
    get_trading_strategy_param_binding_sha256,
    load_trading_strategy_param_binding,
)


TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION = 2


def partition_trading_candidate_rows_for_information_date(
    candidate_rows: list[dict[str, Any]],
    *,
    information_date: object,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep only actionable rows whose own signal bar is the information date.

    The dataset-wide latest date is not sufficient proof that every ticker was
    updated to that date (suspension/provider lag are both possible).  Trading
    may therefore persist an old per-ticker scanner row for diagnostics, but it
    must never turn that old signal into today's proposed order.
    """

    expected_date = normalize_trading_date(information_date, field_name="information_date", allow_none=False)
    current_rows: list[dict[str, Any]] = []
    stale_rows: list[dict[str, str]] = []
    for raw in list(candidate_rows or []):
        if not isinstance(raw, dict):
            raise TypeError("Trading Scanner candidate row 必須是 object")
        row = dict(raw)
        ticker = normalize_trading_ticker(row.get("ticker"))
        row_date = normalize_trading_date(row.get("trade_date"), field_name="trade_date", allow_none=False)
        seed = row.get("execution_plan_seed")
        if not isinstance(seed, dict):
            raise ValueError(f"Trading Scanner candidate 缺少 canonical execution_plan_seed: {ticker}")
        seed_ticker = normalize_trading_ticker(seed.get("ticker"))
        if seed_ticker != ticker:
            raise ValueError(f"Trading Scanner candidate ticker 與 execution_plan_seed 不一致: {ticker}/{seed_ticker or '-'}")
        seed_date = normalize_trading_date(seed.get("trade_date"), field_name="execution_plan_seed.trade_date", allow_none=False)
        if seed_date != row_date:
            raise ValueError(
                f"Trading Scanner candidate trade_date 與 execution_plan_seed 不一致: {ticker} {row_date}/{seed_date}"
            )
        if row_date != expected_date:
            stale_rows.append({"ticker": ticker, "trade_date": row_date})
            continue
        current_rows.append(row)
    return current_rows, stale_rows


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


def resolve_trading_candidate_snapshot_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return Path(resolve_runtime_output_dir(root, domain=RUNTIME_DOMAIN_TRADING, category="scanner")) / "candidate_snapshot.json"


def load_trading_scanner_runtime(
    project_root: str | Path,
    *,
    verify_dataset_content: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    paths = resolve_runtime_domain_paths(
        root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile
    )
    data_dir = Path(paths.data_dir)
    market_snapshot = load_trading_market_data_snapshot(
        root, required=True, verify_dataset_content=verify_dataset_content
    )
    latest_data_date = str(market_snapshot["market_date"])
    if resolve_latest_dataset_date(data_dir) != latest_data_date:
        raise RuntimeError("Trading dataset latest date 與 canonical market-data snapshot 不一致；請重新更新 Trading 資料")
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
    param_binding = load_trading_strategy_param_binding(
        root, required=True, verify_current=True, verify_dataset_content=verify_dataset_content
    )

    return {
        "root": root,
        "profile": profile,
        "paths": paths,
        "data_dir": data_dir,
        "latest_data_date": latest_data_date,
        "selected_path": selected_path,
        "selected_params_sha256": compute_file_sha256(selected_path),
        "param_latest_data_date": param_latest_data_date,
        "member_count": member_count,
        "params": param_source["primary_params"],
        "market_data_snapshot_sha256": get_trading_market_data_snapshot_sha256(root),
        "dataset_content_sha256": str(market_snapshot["dataset_fingerprint"]["csv_content_sha256"]),
        "param_binding_sha256": get_trading_strategy_param_binding_sha256(root),
        "param_binding_fingerprint": str(param_binding["binding_fingerprint"]),
    }


def build_trading_daily_workflow_snapshot(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    paths = resolve_runtime_domain_paths(
        root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile
    )
    data_dir = Path(paths.data_dir)
    raw_latest_data_date = _safe_latest_dataset_date(data_dir)
    market_error = ""
    market_snapshot = None
    try:
        market_snapshot = load_trading_market_data_snapshot(root, required=False, verify_dataset_content=False)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        market_error = f"{type(exc).__name__}: {exc}"
    market_date = None if market_snapshot is None else str(market_snapshot.get("market_date") or "") or None
    market_ready = bool(market_snapshot is not None and market_date and market_date == raw_latest_data_date and not market_error)
    latest_data_date = market_date or raw_latest_data_date

    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    param_latest_data_date = ""
    member_count = 0
    param_error = ""
    param_binding = None
    if selected_path.is_file():
        try:
            payload = load_json_strict(selected_path)
            if not isinstance(payload, dict):
                raise ValueError("selected params root必須是object")
            param_latest_data_date = _selected_payload_latest_data_date(payload)
            param_source = load_portfolio_param_source_from_json(selected_path)
            member_count = int(param_source.get("member_count") or 0)
            if str(payload.get("selector") or "").strip() != str(profile.param_selector):
                raise RuntimeError("selector mismatch")
            param_binding = load_trading_strategy_param_binding(
                root, required=True, verify_current=True, verify_dataset_content=False
            )
        except (OSError, ValueError, RuntimeError) as exc:
            param_error = f"{type(exc).__name__}: {exc}"
    params_ready = bool(
        market_ready
        and latest_data_date
        and selected_path.is_file()
        and not param_error
        and member_count == 1
        and param_latest_data_date == latest_data_date
        and param_binding is not None
    )
    return {
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "strategy_id": profile.strategy_id,
        "param_selector": profile.param_selector,
        "data_dir": project_relative_display_path(data_dir, project_root=root),
        "latest_data_date": latest_data_date,
        "raw_latest_data_date": raw_latest_data_date,
        "market_data_ready": market_ready,
        "market_data_error": market_error or None,
        "market_data_snapshot_sha256": (
            get_trading_market_data_snapshot_sha256(root) if market_snapshot is not None and not market_error else None
        ),
        "dataset_content_sha256": (
            None if market_snapshot is None else str((market_snapshot.get("dataset_fingerprint") or {}).get("csv_content_sha256") or "") or None
        ),
        "selected_params_path": project_relative_display_path(selected_path, project_root=root),
        "selected_params_exists": selected_path.is_file(),
        "param_latest_data_date": param_latest_data_date or None,
        "param_member_count": member_count,
        "param_error": param_error or None,
        "param_binding_sha256": (
            get_trading_strategy_param_binding_sha256(root) if param_binding is not None and not param_error else None
        ),
        "params_ready_for_scan": params_ready,
        "scanner_output_dir": project_relative_display_path(
            resolve_runtime_output_dir(root, domain=RUNTIME_DOMAIN_TRADING, category="scanner"),
            project_root=root,
        ),
    }


def _validate_trading_candidate_snapshot_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Trading candidate snapshot payload 必須是 object")
    if int(payload.get("schema_version", -1)) != TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Trading candidate snapshot schema_version 不相容；請重新執行 Scanner")
    if str(payload.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise ValueError("Trading candidate snapshot runtime domain 不合法")
    if not str(payload.get("strategy_id") or "").strip():
        raise ValueError("Trading candidate snapshot 缺少 strategy_id")
    if not str(payload.get("param_selector") or "").strip():
        raise ValueError("Trading candidate snapshot 缺少 param_selector")
    if not str(payload.get("latest_data_date") or "").strip():
        raise ValueError("Trading candidate snapshot 缺少 latest_data_date")
    if not str(payload.get("selected_params_sha256") or "").strip():
        raise ValueError("Trading candidate snapshot 缺少 selected_params_sha256")
    for field in ("market_data_snapshot_sha256", "dataset_content_sha256", "param_binding_sha256"):
        if not str(payload.get(field) or "").strip():
            raise ValueError(f"Trading candidate snapshot 缺少 {field}")
    if not isinstance(payload.get("candidate_rows"), list):
        raise ValueError("Trading candidate snapshot candidate_rows 必須是 list")
    current_rows, stale_rows = partition_trading_candidate_rows_for_information_date(
        list(payload.get("candidate_rows") or []),
        information_date=payload.get("latest_data_date"),
    )
    if stale_rows or len(current_rows) != len(list(payload.get("candidate_rows") or [])):
        raise ValueError("Trading candidate snapshot 含非 information-date 的舊訊號；請重新執行 Scanner")


def load_trading_candidate_snapshot(
    project_root: str | Path,
    *,
    require_current: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = resolve_trading_candidate_snapshot_path(root)
    if not path.is_file():
        raise FileNotFoundError("Trading Scanner candidate snapshot 尚未產生；請先執行「3 Scanner 候選」。")
    payload = load_json_strict(path)
    _validate_trading_candidate_snapshot_payload(payload)
    if not require_current:
        return payload

    runtime = load_trading_scanner_runtime(root)
    if str(payload.get("strategy_id") or "") != str(runtime["profile"].strategy_id):
        raise RuntimeError("Trading candidate snapshot strategy 與目前設定不一致；請重新執行 Scanner")
    if str(payload.get("param_selector") or "") != str(runtime["profile"].param_selector):
        raise RuntimeError("Trading candidate snapshot selector 與目前設定不一致；請重新執行 Scanner")
    if str(payload.get("latest_data_date") or "") != str(runtime["latest_data_date"]):
        raise RuntimeError("Trading candidate snapshot 已過期；請重新執行 Scanner")
    if str(payload.get("param_latest_data_date") or "") != str(runtime["param_latest_data_date"]):
        raise RuntimeError("Trading candidate snapshot params date 與目前設定不一致；請重新執行 Scanner")
    if str(payload.get("selected_params_sha256") or "") != str(runtime["selected_params_sha256"]):
        raise RuntimeError("Trading candidate snapshot 對應的 params 已改變；請重新執行 Scanner")
    if str(payload.get("dataset_content_sha256") or "") != str(runtime["dataset_content_sha256"]):
        raise RuntimeError("Trading candidate snapshot dataset content identity 已改變；請重新執行 Scanner")
    if str(payload.get("param_binding_sha256") or "") != str(runtime["param_binding_sha256"]):
        raise RuntimeError("Trading candidate snapshot 對應的 Params binding 已改變；請重新執行 Scanner")
    return payload


def get_trading_candidate_snapshot_read_model(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = resolve_trading_candidate_snapshot_path(root)
    if not path.is_file():
        return {
            "exists": False,
            "valid": False,
            "fresh": False,
            "candidate_count": 0,
            "information_date": None,
            "error": None,
            "path": project_relative_display_path(path, project_root=root),
        }
    try:
        payload = load_trading_candidate_snapshot(root, require_current=False)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return {
            "exists": True,
            "valid": False,
            "fresh": False,
            "candidate_count": 0,
            "information_date": None,
            "error": f"{type(exc).__name__}: {exc}",
            "path": project_relative_display_path(path, project_root=root),
        }
    freshness_error = None
    try:
        load_trading_candidate_snapshot(root, require_current=True)
    except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        freshness_error = f"{type(exc).__name__}: {exc}"
    return {
        "exists": True,
        "valid": True,
        "fresh": freshness_error is None,
        "candidate_count": len(payload.get("candidate_rows") or []),
        "stale_candidate_rows_skipped": list(payload.get("stale_candidate_rows_skipped") or []),
        "stale_candidate_count": len(payload.get("stale_candidate_rows_skipped") or []),
        "information_date": payload.get("latest_data_date"),
        "selected_params_sha256": payload.get("selected_params_sha256"),
        "market_data_snapshot_sha256": payload.get("market_data_snapshot_sha256"),
        "dataset_content_sha256": payload.get("dataset_content_sha256"),
        "param_binding_sha256": payload.get("param_binding_sha256"),
        "error": freshness_error,
        "path": project_relative_display_path(path, project_root=root),
    }


__all__ = [
    "TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION",
    "build_trading_daily_workflow_snapshot",
    "get_trading_candidate_snapshot_read_model",
    "load_trading_candidate_snapshot",
    "load_trading_scanner_runtime",
    "partition_trading_candidate_rows_for_information_date",
    "resolve_trading_candidate_snapshot_path",
]

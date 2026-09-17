"""Canonical Trading scanner runtime and candidate-snapshot state helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.file_integrity import compute_file_sha256, load_json_strict
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_output_dir
from core.portfolio_ensemble import build_ensemble_candidate_display_metrics
from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from services.trading.data_readiness import (
    assert_trading_data_readiness,
    build_trading_data_readiness_for_consumer_evidence,
)
from services.trading.market_data_consumer import (
    get_trading_v2_consumer_state_sha256,
    load_trading_v2_consumer_state,
)
from services.trading.market_data_v2_state import build_trading_market_data_v2_read_model
from services.trading.account_state import load_trading_account_state
from services.trading.entry_candidate_projection import project_trading_entry_candidate_payload
from services.trading.pending_entry_state import (
    load_trading_pending_entry_state,
    project_trading_pending_entry_state,
)
from services.trading.strategy_param_runtime import load_trading_strategy_param_runtime
from services.trading.strategy_param_state import (
    get_trading_strategy_param_binding_sha256,
    load_trading_strategy_param_binding,
)


TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION = 9


def _require_candidate_signal_date(row: dict[str, Any], *, trade_date: str) -> str:
    signal_date = normalize_trading_date(
        row.get("signal_date"), field_name="candidate.signal_date", allow_none=False
    )
    if signal_date > trade_date:
        raise ValueError(
            f"Trading candidate signal_date 晚於 trade_date: "
            f"{row.get('ticker') or '-'} {signal_date}/{trade_date}"
        )
    return signal_date


def partition_trading_candidate_rows_for_information_date(
    candidate_rows: list[dict[str, Any]],
    *,
    information_date: object,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep only actionable rows whose candidate information date is current.

    The dataset-wide latest date is not sufficient proof that every ticker was
    updated to that date (suspension/provider lag are both possible). Trading
    may therefore persist an old per-ticker scanner row for diagnostics, but it
    must never turn that old candidate into today's proposed order. The original
    buy-signal date is separate evidence and may legitimately precede trade_date.
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
        _require_candidate_signal_date(row, trade_date=row_date)
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
    market_state = load_trading_v2_consumer_state(
        root, required=True, verify_current_view=verify_dataset_content
    )
    latest_data_date = str(market_state["market_date"])
    data_readiness = build_trading_data_readiness_for_consumer_evidence(
        root,
        strategy_id=profile.strategy_id,
        target_date=latest_data_date,
        consumer_state_ready=True,
    )
    assert_trading_data_readiness(data_readiness)
    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    if not selected_path.is_file():
        raise FileNotFoundError(
            "Trading strategy params尚未產生；請先在 Workbench 執行「3 套用 Params」，並選擇重新訓練。"
        )

    payload = load_json_strict(selected_path)
    if not isinstance(payload, dict):
        raise RuntimeError("Trading selected strategy params payload不合法")
    if str(payload.get("selector") or "").strip() != str(profile.param_selector):
        raise RuntimeError(
            "Trading selected strategy params selector與目前Trading設定不一致；請重新訓練。"
        )

    param_runtime = load_trading_strategy_param_runtime(selected_path)
    member_count = int(param_runtime["member_count"])
    param_min_agree = int(param_runtime["min_agree"])

    param_latest_data_date = _selected_payload_latest_data_date(payload)
    if not param_latest_data_date:
        raise RuntimeError(
            "Trading selected strategy params 缺少訓練資料日；請重新訓練。"
        )
    if param_latest_data_date > latest_data_date:
        raise RuntimeError(
            "Trading strategy params 訓練資料日晚於目前 Trading data；"
            f"data={latest_data_date}, params={param_latest_data_date}。"
        )
    # ``market_state`` above already performed the expensive finalized-view
    # verification when requested.  Binding currentness only needs to compare
    # against that persisted consumer identity; repeating the same V2 view scan
    # here roughly doubles Scanner startup latency.
    param_binding = load_trading_strategy_param_binding(
        root, required=True, verify_current=True, verify_dataset_content=False
    )

    return {
        "root": root,
        "profile": profile,
        "data_dir": root / "data" / "trading" / "market_data_v2",
        "latest_data_date": latest_data_date,
        "selected_path": selected_path,
        "selected_params_sha256": compute_file_sha256(selected_path),
        "param_latest_data_date": param_latest_data_date,
        "param_usage_mode": str(param_binding.get("usage_mode") or ""),
        "member_count": member_count,
        "param_min_agree": param_min_agree,
        "param_members": list(param_runtime["members"]),
        "params": param_runtime["primary_params"],
        "market_data_consumer_state_sha256": get_trading_v2_consumer_state_sha256(root),
        "market_data_source_view_fingerprint": str(market_state["source_view_fingerprint"]),
        "market_data_source": str(market_state.get("source") or "trading_market_data_v2_historical_latest_view"),
        "current_universe_tickers": list(market_state.get("current_execution_pool_tickers") or []),
        "required_position_tickers": list(market_state.get("required_position_tickers") or []),
        "param_binding_sha256": get_trading_strategy_param_binding_sha256(root),
        "param_binding_fingerprint": str(param_binding["binding_fingerprint"]),
        "trading_data_readiness": data_readiness,
        "trading_data_dependency_fingerprint": data_readiness["dependency_fingerprint"],
    }


def build_trading_daily_workflow_snapshot(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    data_dir = root / "data" / "trading" / "market_data_v2"
    market_error = ""
    market_state = None
    try:
        market_state = load_trading_v2_consumer_state(root, required=False, verify_current_view=False)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        market_error = f"{type(exc).__name__}: {exc}"
    market_date = None if market_state is None else str(market_state.get("market_date") or "") or None
    market_ready = bool(market_state is not None and market_date and not market_error)
    latest_data_date = market_date

    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    param_latest_data_date = ""
    member_count = 0
    param_min_agree = 0
    param_error = ""
    param_binding_error = ""
    param_binding = None
    if selected_path.is_file():
        try:
            payload = load_json_strict(selected_path)
            if not isinstance(payload, dict):
                raise ValueError("selected params root必須是object")
            param_latest_data_date = _selected_payload_latest_data_date(payload)
            param_runtime = load_trading_strategy_param_runtime(selected_path)
            member_count = int(param_runtime["member_count"])
            param_min_agree = int(param_runtime["min_agree"])
            if str(payload.get("selector") or "").strip() != str(profile.param_selector):
                raise RuntimeError("selector mismatch")
            if not param_latest_data_date:
                raise RuntimeError("selected params 缺少訓練資料日")
            if latest_data_date and param_latest_data_date > latest_data_date:
                raise RuntimeError("selected params 訓練資料日晚於目前 Trading data")
        except (OSError, ValueError, RuntimeError) as exc:
            param_error = f"{type(exc).__name__}: {exc}"
        if not param_error:
            try:
                param_binding = load_trading_strategy_param_binding(
                    root, required=True, verify_current=True, verify_dataset_content=False
                )
            except (OSError, ValueError, RuntimeError) as exc:
                param_binding_error = f"{type(exc).__name__}: {exc}"
    current_execution_pool_stats = dict(
        (market_state or {}).get("current_execution_pool_stats") or {}
    )
    current_execution_pool_ticker_count = (
        None
        if market_state is None
        else int(
            market_state.get("current_execution_pool_ticker_count")
            or len(market_state.get("current_execution_pool_tickers") or [])
        )
    )
    data_readiness = build_trading_data_readiness_for_consumer_evidence(
        root,
        strategy_id=profile.strategy_id,
        target_date=latest_data_date,
        consumer_state_ready=market_ready,
        consumer_state_reason=market_error or (None if market_ready else "Trading V2 execution consumer state 尚未就緒"),
    )
    trading_data_ready = bool(data_readiness.get("ready"))
    params_reusable = bool(
        trading_data_ready
        and latest_data_date
        and selected_path.is_file()
        and not param_error
        and member_count >= 1
        and 1 <= param_min_agree <= member_count
        and param_latest_data_date
        and param_latest_data_date <= latest_data_date
    )
    params_ready = bool(
        params_reusable
        and not param_binding_error
        and param_binding is not None
    )
    try:
        v2_archive = build_trading_market_data_v2_read_model(root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        v2_archive = {
            "status": "STALE",
            "provider_ready": False,
            "latest_sync_target_date": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "strategy_id": profile.strategy_id,
        "param_selector": profile.param_selector,
        "data_dir": project_relative_display_path(data_dir, project_root=root),
        "latest_data_date": latest_data_date,
        "raw_latest_data_date": latest_data_date,
        "market_data_ready": market_ready,
        "market_data_error": market_error or None,
        "trading_data_ready": trading_data_ready,
        "trading_data_readiness_status": data_readiness.get("status"),
        "trading_data_dependency_fingerprint": data_readiness.get("dependency_fingerprint"),
        "trading_data_required_v2_count": data_readiness.get("required_v2_dataset_count"),
        "trading_data_ready_v2_count": data_readiness.get("ready_v2_dataset_count"),
        "trading_data_blockers": list(data_readiness.get("blocking_dependencies") or []),
        "market_data_consumer_state_sha256": (
            get_trading_v2_consumer_state_sha256(root) if market_state is not None and not market_error else None
        ),
        "market_data_source_view_fingerprint": (
            None if market_state is None else str(market_state.get("source_view_fingerprint") or "") or None
        ),
        "market_data_source": (None if market_state is None else market_state.get("source")),
        "current_execution_pool_ticker_count": current_execution_pool_ticker_count,
        "current_execution_pool_stats": current_execution_pool_stats,
        "market_data_v2_archive_status": v2_archive.get("status"),
        "market_data_v2_archive_provider_ready": bool(v2_archive.get("provider_ready")),
        "market_data_v2_archive_latest_date": v2_archive.get("latest_sync_target_date"),
        "market_data_v2_archive_error": v2_archive.get("error"),
        "selected_params_path": project_relative_display_path(selected_path, project_root=root),
        "selected_params_exists": selected_path.is_file(),
        "param_latest_data_date": param_latest_data_date or None,
        "param_member_count": member_count,
        "param_min_agree": param_min_agree or None,
        "param_error": param_error or None,
        "param_binding_error": param_binding_error or None,
        "param_usage_mode": None if param_binding is None else param_binding.get("usage_mode"),
        "params_reusable": params_reusable,
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
    member_count = int(payload.get("param_member_count") or 0)
    min_agree = int(payload.get("param_min_agree") or 0)
    if member_count < 1 or min_agree < 1 or min_agree > member_count:
        raise ValueError("Trading candidate snapshot Params ensemble metadata 不合法")
    if "candidate_display_metrics" not in payload:
        raise ValueError("Trading candidate snapshot 缺少 candidate_display_metrics；請重新執行 Scanner")
    display_metrics = payload.get("candidate_display_metrics")
    if not isinstance(display_metrics, list) or any(not isinstance(item, dict) for item in display_metrics):
        raise TypeError("Trading candidate snapshot candidate_display_metrics 必須是 object list")
    metric_keys = [str(item.get("key") or "").strip() for item in display_metrics]
    if any(not key for key in metric_keys) or len(metric_keys) != len(set(metric_keys)):
        raise ValueError("Trading candidate snapshot candidate display metric key 不合法")
    for item in display_metrics:
        if not str(item.get("label") or "").strip() or not str(item.get("value_field") or "").strip():
            raise ValueError("Trading candidate snapshot candidate display metric descriptor 不完整")
        format_kind = str(item.get("format_kind") or "")
        if format_kind not in {"text", "integer", "integer_grouped", "number", "percent", "r", "fraction"}:
            raise ValueError("Trading candidate snapshot candidate display metric formatter 不支援")
        if format_kind == "fraction":
            denominator = int(item.get("denominator") or 0)
            if denominator != member_count:
                raise ValueError("Trading candidate snapshot candidate display metric denominator 與 Params members 不一致")
    expected_display_metrics = build_ensemble_candidate_display_metrics(total_member_count=member_count)
    if display_metrics != expected_display_metrics:
        raise ValueError("Trading candidate snapshot candidate display metrics 與目前 canonical selector evidence contract 不一致；請重新執行 Scanner")
    for field in ("market_data_consumer_state_sha256", "market_data_source_view_fingerprint", "param_binding_sha256"):
        if not str(payload.get(field) or "").strip():
            raise ValueError(f"Trading candidate snapshot 缺少 {field}")
    scanned_tickers = payload.get("scanned_tickers")
    if not isinstance(scanned_tickers, list) or not scanned_tickers:
        raise ValueError("Trading candidate snapshot scanned_tickers 必須是非空 list")
    normalized_scanned = sorted({normalize_trading_ticker(item) for item in scanned_tickers})
    if list(scanned_tickers) != normalized_scanned:
        raise ValueError("Trading candidate snapshot scanned_tickers 必須為排序後唯一 canonical ticker")
    if not isinstance(payload.get("candidate_rows"), list):
        raise ValueError("Trading candidate snapshot candidate_rows 必須是 list")
    candidate_rows = list(payload.get("candidate_rows") or [])
    if any(not isinstance(row, dict) for row in candidate_rows):
        raise TypeError("Trading candidate snapshot candidate row 必須是 object")
    for row in candidate_rows:
        row_member_count = int(row.get("ensemble_member_count") or 0)
        row_min_agree = int(row.get("ensemble_min_agree") or 0)
        vote_count = int(row.get("ensemble_vote_count") or 0)
        member_keys = [str(item) for item in list(row.get("ensemble_member_keys") or [])]
        member_params = row.get("ensemble_member_params_by_key") or {}
        if row_member_count < 1 or row_min_agree < 1 or vote_count < row_min_agree or vote_count > row_member_count:
            raise ValueError("Trading candidate row-level ensemble metadata 不合法")
        if len(set(member_keys)) != vote_count:
            raise ValueError("Trading candidate agreeing member keys 與 vote_count 不一致")
        if not isinstance(member_params, dict) or set(member_keys) - set(str(key) for key in member_params):
            raise ValueError("Trading candidate 缺少 agreeing-voter immutable Params lineage")
        if not str(row.get("params_signature") or "").strip() or not str(row.get("ensemble_member_key") or "").strip():
            raise ValueError("Trading candidate 缺少 representative Params identity")
        trade_date = normalize_trading_date(row.get("trade_date"), field_name="candidate.trade_date", allow_none=False)
        _require_candidate_signal_date(row, trade_date=trade_date)
        lineage_source = str(row.get("param_lineage_source") or "current_selected_artifact")
        if lineage_source not in {"current_selected_artifact", "entry_order_frozen_ensemble"}:
            raise ValueError("Trading candidate param_lineage_source 不合法")
        entry_source = str((row.get("execution_plan_seed") or {}).get("entry_source") or "")
        if entry_source == "reentry" and (
            lineage_source != "entry_order_frozen_ensemble" or not str(row.get("source_entry_order_id") or "")
        ):
            raise ValueError("Trading live re-entry candidate 缺 broker-truth source entry lineage")
    candidate_tickers = {normalize_trading_ticker(row.get("ticker")) for row in candidate_rows}
    outside_membership = sorted(candidate_tickers - set(normalized_scanned))
    if outside_membership:
        raise ValueError(
            "Trading candidate snapshot 含 scanner membership 外的 candidate；"
            f"outside={outside_membership[:20]}"
        )
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
        raise FileNotFoundError("Trading Scanner candidate snapshot 尚未產生；請先執行「4 Scanner 候選」。")
    payload = load_json_strict(path)
    _validate_trading_candidate_snapshot_payload(payload)
    if not require_current:
        return payload

    runtime = load_trading_scanner_runtime(root, verify_dataset_content=True)
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
    if int(payload.get("param_member_count") or 0) != int(runtime["member_count"]):
        raise RuntimeError("Trading candidate snapshot Params member_count 已改變；請重新執行 Scanner")
    if int(payload.get("param_min_agree") or 0) != int(runtime["param_min_agree"]):
        raise RuntimeError("Trading candidate snapshot Params min_agree 已改變；請重新執行 Scanner")
    if str(payload.get("market_data_consumer_state_sha256") or "") != str(runtime["market_data_consumer_state_sha256"]):
        raise RuntimeError("Trading candidate snapshot 對應的 market-data membership 已改變；請重新執行 Scanner")
    if list(payload.get("scanned_tickers") or []) != list(runtime.get("current_universe_tickers") or []):
        raise RuntimeError("Trading candidate snapshot scanned membership 已過期；請重新執行 Scanner")
    if str(payload.get("market_data_source_view_fingerprint") or "") != str(runtime["market_data_source_view_fingerprint"]):
        raise RuntimeError("Trading candidate snapshot dataset content identity 已改變；請重新執行 Scanner")
    if str(payload.get("param_binding_sha256") or "") != str(runtime["param_binding_sha256"]):
        raise RuntimeError("Trading candidate snapshot 對應的 Params binding 已改變；請重新執行 Scanner")
    return payload


def _resolve_trading_held_tickers(project_root: str | Path) -> set[str]:
    """Return canonical currently-held tickers for account-aware Scanner views."""
    state = load_trading_account_state(project_root, required=False)
    if not state:
        return set()
    held: set[str] = set()
    for ticker, record in (state.get("positions") or {}).items():
        broker = (record or {}).get("broker") or {}
        if int(broker.get("qty") or 0) > 0:
            held.add(normalize_trading_ticker(ticker))
    return held


def filter_trading_candidate_rows_for_held_positions(
    candidate_rows: list[dict[str, Any]],
    *,
    held_tickers: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Mirror Research held-position exclusion in account-aware Scanner views."""
    normalized_held = {normalize_trading_ticker(ticker) for ticker in held_tickers if str(ticker or "").strip()}
    visible: list[dict[str, Any]] = []
    skipped: list[str] = []
    for raw in list(candidate_rows or []):
        row = dict(raw)
        ticker = normalize_trading_ticker(row.get("ticker"))
        if ticker in normalized_held:
            skipped.append(ticker)
            continue
        visible.append(row)
    return visible, sorted(set(skipped))


def load_trading_candidate_snapshot_for_account(
    project_root: str | Path,
    *,
    require_current: bool = False,
) -> dict[str, Any]:
    """Load Scanner truth projected through current account/pending entry state.

    The persisted Scanner snapshot stays immutable.  This function is the canonical
    actionable-candidate read path used by Workbench views: a ticker already held or
    already represented by an ACTIVE pending entry must not remain simultaneously
    actionable in Scanner Pool.  Closing/deleting that pending entry automatically
    makes the still-current Scanner candidate visible again.
    """
    root = Path(project_root).resolve()
    payload = dict(load_trading_candidate_snapshot(root, require_current=require_current))
    original_rows = [dict(row) for row in list(payload.get("candidate_rows") or [])]
    visible, skipped = filter_trading_candidate_rows_for_held_positions(
        original_rows,
        held_tickers=_resolve_trading_held_tickers(root),
    )
    payload["candidate_rows"] = visible
    payload["held_candidate_tickers_skipped"] = skipped

    pending_state = load_trading_pending_entry_state(root, required=False)
    pending_snapshot = project_trading_pending_entry_state(
        pending_state,
        current_information_date=payload.get("latest_data_date"),
    )
    projected = project_trading_entry_candidate_payload(
        payload,
        account_snapshot=None,  # held-position projection was applied above for legacy evidence.
        pending_snapshot=pending_snapshot,
    )
    before_pending = {
        normalize_trading_ticker(row.get("ticker"))
        for row in visible
        if str(row.get("ticker") or "").strip()
    }
    after_pending = {
        normalize_trading_ticker(row.get("ticker"))
        for row in list(projected.get("candidate_rows") or [])
        if str(row.get("ticker") or "").strip()
    }
    projected["active_pending_candidate_tickers_skipped"] = sorted(before_pending - after_pending)
    return projected


def get_trading_candidate_snapshot_read_model(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = resolve_trading_candidate_snapshot_path(root)
    if not path.is_file():
        return {
            "exists": False,
            "valid": False,
            "fresh": False,
            "candidate_count": 0,
            "candidate_tickers": [],
            "information_date": None,
            "error": None,
            "path": project_relative_display_path(path, project_root=root),
        }
    try:
        payload = load_trading_candidate_snapshot_for_account(root, require_current=False)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return {
            "exists": True,
            "valid": False,
            "fresh": False,
            "candidate_count": 0,
            "candidate_tickers": [],
            "information_date": None,
            "error": f"{type(exc).__name__}: {exc}",
            "path": project_relative_display_path(path, project_root=root),
        }
    freshness_error = None
    try:
        load_trading_candidate_snapshot(root, require_current=True)
    except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        freshness_error = f"{type(exc).__name__}: {exc}"
    candidate_rows = [dict(row) for row in list(payload.get("candidate_rows") or [])]
    candidate_tickers = sorted({
        str(row.get("ticker") or "").strip().upper()
        for row in candidate_rows
        if str(row.get("ticker") or "").strip()
    })
    return {
        "exists": True,
        "valid": True,
        "fresh": freshness_error is None,
        "candidate_count": len(candidate_rows),
        "candidate_tickers": candidate_tickers,
        "candidate_display_metrics": list(payload.get("candidate_display_metrics") or []),
        "scanned_tickers": list(payload.get("scanned_tickers") or []),
        "scanned_ticker_count": len(payload.get("scanned_tickers") or []),
        "stale_candidate_rows_skipped": list(payload.get("stale_candidate_rows_skipped") or []),
        "stale_candidate_count": len(payload.get("stale_candidate_rows_skipped") or []),
        "held_candidate_tickers_skipped": list(payload.get("held_candidate_tickers_skipped") or []),
        "active_pending_candidate_tickers_skipped": list(payload.get("active_pending_candidate_tickers_skipped") or []),
        "information_date": payload.get("latest_data_date"),
        "selected_params_sha256": payload.get("selected_params_sha256"),
        "market_data_consumer_state_sha256": payload.get("market_data_consumer_state_sha256"),
        "market_data_source_view_fingerprint": payload.get("market_data_source_view_fingerprint"),
        "param_binding_sha256": payload.get("param_binding_sha256"),
        "error": freshness_error,
        "path": project_relative_display_path(path, project_root=root),
    }


__all__ = [
    "TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION",
    "build_trading_daily_workflow_snapshot",
    "get_trading_candidate_snapshot_read_model",
    "load_trading_candidate_snapshot",
    "load_trading_candidate_snapshot_for_account",
    "filter_trading_candidate_rows_for_held_positions",
    "load_trading_scanner_runtime",
    "partition_trading_candidate_rows_for_information_date",
    "resolve_trading_candidate_snapshot_path",
]

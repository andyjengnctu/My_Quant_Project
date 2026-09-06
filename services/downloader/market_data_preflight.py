"""Market Data V2 Backer capability preflight and exact logical request planner."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.market_data_bootstrap_planner import DatasetProbeEvidence, build_bootstrap_request_plan
from core.market_data_dataset_registry import (
    BOOTSTRAP_BULK_REFERENCE_DATES,
    BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE,
    BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE,
    BOOTSTRAP_SINGLE_FULL_RANGE,
    BOOTSTRAP_SINGLE_NO_DATES,
    DEFAULT_EQUITY_PROBE_DATA_ID,
    get_market_dataset_specs,
    validate_market_dataset_registry,
)
from core.trading_market_clock import select_latest_completed_daily_date
from services.downloader.finmind_http import FinMindHttpClient, FinMindHttpError

PREFLIGHT_FULL_RANGE_START = "1900-01-01"
CURRENT_MARKET_TYPES = {"twse", "tpex"}


def _frame_dates(frame: pd.DataFrame) -> tuple[str, ...]:
    if "date" not in frame.columns or frame.empty:
        return ()
    parsed = pd.to_datetime(frame["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d")
    return tuple(sorted(set(parsed.tolist())))


def _build_evidence(dataset: str, frame: pd.DataFrame, *, request_count: int) -> DatasetProbeEvidence:
    dates = _frame_dates(frame)
    return DatasetProbeEvidence(
        dataset=dataset,
        status="PASS",
        request_count=int(request_count),
        row_count=int(len(frame)),
        columns=tuple(str(column) for column in frame.columns),
        observed_dates=dates,
        earliest_date=dates[0] if dates else None,
        latest_date=dates[-1] if dates else None,
        error=None,
    )


def _historical_instruments(stock_info: pd.DataFrame, delisting: pd.DataFrame) -> tuple[str, ...]:
    ids: set[str] = set()
    if not stock_info.empty:
        missing = {"stock_id", "type"}.difference(stock_info.columns)
        if missing:
            raise ValueError(f"TaiwanStockInfo 缺少欄位: {sorted(missing)}")
        market_type = stock_info["type"].astype(str).str.strip().str.lower()
        valid = stock_info.loc[market_type.isin(CURRENT_MARKET_TYPES), "stock_id"]
        ids.update(str(value).strip() for value in valid.tolist() if str(value).strip())
    if not delisting.empty:
        if "stock_id" not in delisting.columns:
            raise ValueError("TaiwanStockDelisting 缺少 stock_id")
        ids.update(str(value).strip() for value in delisting["stock_id"].tolist() if str(value).strip())
    normalized = tuple(sorted(ids))
    if not normalized:
        raise ValueError("無法由 TaiwanStockInfo + TaiwanStockDelisting 建立 historical instrument universe")
    return normalized


def _resolve_probe_stock(instruments: Iterable[str]) -> str:
    values = tuple(str(value).strip() for value in instruments if str(value).strip())
    if DEFAULT_EQUITY_PROBE_DATA_ID in values:
        return DEFAULT_EQUITY_PROBE_DATA_ID
    numeric = sorted(value for value in values if value.isdigit() and len(value) == 4)
    if numeric:
        return numeric[0]
    return sorted(values)[0]


def _verify_exact_date_bulk(frame: pd.DataFrame, *, dataset: str, expected_date: str) -> None:
    if frame.empty:
        raise ValueError(f"{dataset} all-market exact-date probe 回傳空資料: {expected_date}")
    if "date" not in frame.columns:
        raise ValueError(f"{dataset} all-market exact-date probe 缺少 date 欄")
    dates = set(pd.to_datetime(frame["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").tolist())
    if dates != {expected_date}:
        raise ValueError(f"{dataset} exact-date probe 日期不純: expected={expected_date}, actual={sorted(dates)[:10]}")


def _probe_spec(client: FinMindHttpClient, spec, *, as_of_date: str, probe_stock: str) -> DatasetProbeEvidence:
    before = client.data_request_count
    try:
        if spec.bootstrap_mode == BOOTSTRAP_SINGLE_NO_DATES:
            frame = client.get_data(dataset=spec.dataset)
        elif spec.bootstrap_mode == BOOTSTRAP_SINGLE_FULL_RANGE:
            frame = client.get_data(
                dataset=spec.dataset,
                start_date=PREFLIGHT_FULL_RANGE_START,
                end_date=as_of_date,
            )
        elif spec.bootstrap_mode in {BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, BOOTSTRAP_BULK_REFERENCE_DATES}:
            data_id = probe_stock if spec.probe_data_id == DEFAULT_EQUITY_PROBE_DATA_ID else (spec.probe_data_id or probe_stock)
            frame = client.get_data(
                dataset=spec.dataset,
                data_id=data_id,
                start_date=PREFLIGHT_FULL_RANGE_START,
                end_date=as_of_date,
            )
            dates = _frame_dates(frame)
            if spec.bootstrap_mode == BOOTSTRAP_BULK_REFERENCE_DATES:
                if not dates:
                    raise ValueError(f"{spec.dataset} reference-stock full-history probe 沒有 date evidence")
                exact_date = dates[-1]
                bulk = client.get_data(dataset=spec.dataset, start_date=exact_date, end_date=exact_date)
                _verify_exact_date_bulk(bulk, dataset=spec.dataset, expected_date=exact_date)
            elif spec.full_market_exact_date_expected and dates:
                exact_date = dates[-1]
                bulk = client.get_data(dataset=spec.dataset, start_date=exact_date, end_date=exact_date)
                _verify_exact_date_bulk(bulk, dataset=spec.dataset, expected_date=exact_date)
        elif spec.bootstrap_mode == BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE:
            frames = []
            for data_id in spec.fixed_data_ids:
                frames.append(
                    client.get_data(
                        dataset=spec.dataset,
                        data_id=data_id,
                        start_date=PREFLIGHT_FULL_RANGE_START,
                        end_date=as_of_date,
                    )
                )
            frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        else:
            raise ValueError(f"不支援 bootstrap_mode: {spec.bootstrap_mode}")
        return _build_evidence(spec.dataset, frame, request_count=client.data_request_count - before)
    except (FinMindHttpError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        return DatasetProbeEvidence(
            dataset=spec.dataset,
            status="FAIL",
            request_count=int(client.data_request_count - before),
            row_count=0,
            columns=(),
            observed_dates=(),
            earliest_date=None,
            latest_date=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _report_markdown(payload: dict[str, object]) -> str:
    plan = payload.get("plan") or {}
    quota = payload.get("quota") or {}
    lines = [
        "# Market Data V2 Backer Preflight / Bootstrap Plan",
        "",
        f"- status: `{payload.get('status')}`",
        f"- as_of_date: `{payload.get('as_of_date')}`",
        f"- historical instruments: **{payload.get('historical_instrument_count')}**",
        f"- included datasets: **{payload.get('included_dataset_count')}**",
        f"- excluded datasets: **{payload.get('excluded_dataset_count')}**",
        f"- actual preflight data requests: **{payload.get('preflight_data_request_count')}**",
        f"- live quota: **{quota.get('user_count_after')} / {quota.get('api_request_limit')}**",
        "",
    ]
    failures = payload.get("probe_failures") or []
    if failures:
        lines.extend(["## Blocking probe failures", ""])
        for item in failures:
            lines.append(f"- `{item.get('dataset')}`: {item.get('error')}")
        lines.append("")
    if plan:
        lines.extend(
            [
                "## Exact logical bootstrap plan",
                "",
                f"- total requests: **{plan.get('total_requests')}**",
                f"- per-instrument datasets: **{plan.get('per_instrument_dataset_count')}**",
                f"- minimum quota-hours: **{float(plan.get('minimum_quota_hours') or 0):.2f}**",
                f"- minimum quota windows: **{plan.get('minimum_quota_windows')}**",
                "",
                "| Dataset | Mode | Requests | Basis |",
                "|---|---|---:|---|",
            ]
        )
        for row in plan.get("rows") or []:
            lines.append(
                f"| `{row.get('dataset')}` | `{row.get('bootstrap_mode')}` | {row.get('request_count')} | {row.get('basis')} |"
            )
        lines.append("")
    lines.extend(["## Probe schemas", "", "| Dataset | Status | Rows | Columns | Date range | Requests |", "|---|---|---:|---|---|---:|"])
    for item in payload.get("probes") or []:
        columns = ", ".join(item.get("columns") or [])
        date_range = f"{item.get('earliest_date') or '-'} ~ {item.get('latest_date') or '-'}"
        lines.append(
            f"| `{item.get('dataset')}` | {item.get('status')} | {item.get('row_count')} | {columns} | {date_range} | {item.get('request_count')} |"
        )
    return "\n".join(lines) + "\n"


def run_market_data_v2_preflight(
    *,
    token: str,
    output_dir,
    now: datetime,
    timeout_sec: float = 30.0,
    client: FinMindHttpClient | None = None,
) -> dict[str, object]:
    registry_summary = validate_market_dataset_registry()
    specs = get_market_dataset_specs(included_only=True)
    excluded_specs = tuple(spec for spec in get_market_dataset_specs() if not spec.included)
    http = client or FinMindHttpClient(token=token, timeout_sec=timeout_sec)

    usage_before = http.get_usage()
    today = now.date().isoformat()

    # These three calls are also their own dataset probes and are reused below.
    stock_info = http.get_data(dataset="TaiwanStockInfo")
    delisting = http.get_data(
        dataset="TaiwanStockDelisting",
        start_date=PREFLIGHT_FULL_RANGE_START,
        end_date=today,
    )
    trading_dates = http.get_data(dataset="TaiwanStockTradingDate")
    if "date" not in trading_dates.columns:
        raise ValueError("TaiwanStockTradingDate preflight 缺少 date")
    as_of_date = select_latest_completed_daily_date(trading_dates["date"].tolist(), now=now)
    if as_of_date is None:
        raise ValueError("無法由 TaiwanStockTradingDate 解析 latest completed market date")

    instruments = _historical_instruments(stock_info, delisting)
    probe_stock = _resolve_probe_stock(instruments)
    preloaded = {
        "TaiwanStockInfo": _build_evidence("TaiwanStockInfo", stock_info, request_count=1),
        "TaiwanStockDelisting": _build_evidence("TaiwanStockDelisting", delisting, request_count=1),
        "TaiwanStockTradingDate": _build_evidence("TaiwanStockTradingDate", trading_dates, request_count=1),
    }

    evidence: dict[str, DatasetProbeEvidence] = {}
    for spec in specs:
        if spec.dataset in preloaded:
            evidence[spec.dataset] = preloaded[spec.dataset]
            continue
        evidence[spec.dataset] = _probe_spec(http, spec, as_of_date=as_of_date, probe_stock=probe_stock)

    usage_after = http.get_usage()
    failures = [item for item in evidence.values() if item.status != "PASS"]
    plan = None
    plan_error = None
    if not failures:
        try:
            plan = build_bootstrap_request_plan(
                specs=specs,
                historical_instruments=instruments,
                evidence_by_dataset=evidence,
                as_of_date=as_of_date,
                quota_limit=usage_after.api_request_limit,
            )
        except ValueError as exc:
            plan_error = f"{type(exc).__name__}: {exc}"

    observed_quota_delta = int(usage_after.user_count - usage_before.user_count)
    quota_warning = None
    if observed_quota_delta < 0:
        quota_warning = "user_count 在 preflight 前後下降，可能跨 quota window；不可用 delta 驗證 request accounting。"
    elif observed_quota_delta != int(http.data_request_count):
        quota_warning = (
            "user_count delta 與本程序 data request attempts 不一致；可能存在其他 concurrent requests、"
            "quota 計數延遲，或 user_info 本身採不同計數口徑。Bootstrap 必須繼續以 live quota guard 控制。"
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "READY" if plan is not None and not failures else "BLOCKED",
        "generated_at": now.isoformat(),
        "as_of_date": as_of_date,
        "probe_stock_id": probe_stock,
        "historical_instrument_count": len(instruments),
        "historical_instruments": list(instruments),
        "included_dataset_count": len(specs),
        "excluded_dataset_count": len(excluded_specs),
        "registry_summary": registry_summary,
        "preflight_data_request_count": int(http.data_request_count),
        "preflight_usage_request_count": int(http.usage_request_count),
        "quota": {
            "user_count_before": usage_before.user_count,
            "user_count_after": usage_after.user_count,
            "api_request_limit": usage_after.api_request_limit,
            "observed_usage_delta": observed_quota_delta,
            "accounting_warning": quota_warning,
        },
        "probes": [asdict(evidence[spec.dataset]) for spec in specs],
        "probe_failures": [asdict(item) for item in failures],
        "plan_error": plan_error,
        "plan": asdict(plan) if plan is not None else None,
        "excluded_datasets": [
            {"dataset": spec.dataset, "category": spec.category, "rationale": spec.rationale}
            for spec in excluded_specs
        ],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"market_data_v2_preflight_plan_{stamp}.json"
    md_path = out_dir / f"market_data_v2_preflight_plan_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_report_markdown(payload), encoding="utf-8")
    return {
        **payload,
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }


__all__ = [
    "PREFLIGHT_FULL_RANGE_START",
    "run_market_data_v2_preflight",
]

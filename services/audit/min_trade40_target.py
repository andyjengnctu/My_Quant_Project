"""P16 Min 40-session realized-net-R event target, hindsight-only.

This is a separate *event* cohort from the universal 40-bar stock-day Oracle.
It deliberately does not feed portfolio scores, DL fitting, Strategy Compare,
or any Trading state. Price/parameters/fee/entry/exit logic are canonical.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT
from core.file_integrity import canonical_json_sha256
from core.portfolio_param_runtime import (
    build_active_param_ensemble_objects_from_payload,
    build_active_param_objects_from_payload,
)
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from filters.breakout_quality.trade_path_label import (
    build_signal_cache, simulate_realized_trade_path_label,
)
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame
from services.audit.oracle_replay_artifacts import price_inventory_sha256


TARGET_ID = "LABEL-mintrade40_realized_net_r_v1"
TARGET_NAME = "MinTrade40_NetR"
# This existing scientific label ID is fixed to 40 market trading sessions.
TARGET_HORIZON_SESSIONS = 40
REPORT_SCHEMA = 1
# Reuse existing per-seed evidence path; this is NOT a separate active report.
OUTPUT_SUBDIR = "breakout_quality/reusable/min_trade40_netr"
CACHE_SCHEMA = 1


class MinTrade40BlockedError(RuntimeError):
    """Event data cannot legally support a completed, comparable target."""


@dataclass(frozen=True)
class MinTrade40P16Context:
    """Verified P16 views supplied by the owning Oracle orchestration service."""

    dataset: str
    start_date: str
    cutoff_date: str
    benchmark_id: str
    prepared: tuple[tuple[int, str, str, Any, Any], ...]
    parameter_evidence: tuple[Mapping[str, Any], ...]


def _target_spec(definition) -> dict[str, Any]:
    declared = tuple(definition.source.get("event_targets") or ())
    active = [dict(row) for row in declared if row.get("enabled", True)]
    if len(active) != 1:
        raise MinTrade40BlockedError("[1] Min event Target 須宣告恰一個啟用中的身分")
    spec = active[0]
    if (str(spec.get("target")) != TARGET_NAME or
            str(spec.get("label_id")) != TARGET_ID or
            type(spec.get("horizon_trading_bars")) is not int or
            int(spec["horizon_trading_bars"]) != TARGET_HORIZON_SESSIONS or
            spec.get("cohort") != "p16_min_signal_events" or
            spec.get("signal_membership_policy") != "next_observed_ticker_trade_day_p16_v1"):
        raise MinTrade40BlockedError("Min event Target 的科學身分／期限／母體契約無效")
    return spec


def _schedule(base_kind: str, base_params: Any) -> tuple[dict[str, Any], ...]:
    """Only formal canonical P16 param views; no A2/P2 teacher fallback."""
    if base_kind == "rolling_active_param_ensemble":
        records = build_active_param_ensemble_objects_from_payload(base_params)
        for record in records:
            members = record["members"]
            if len(members) != 1:
                raise MinTrade40BlockedError("P16 Base-Finalist-Best 不允許多member共識")
            record["params_obj"] = members[0]["params_obj"]
    elif base_kind == "rolling_oos_param_schedule":
        records = build_active_param_objects_from_payload(base_params)
    else:
        raise MinTrade40BlockedError(f"Min event Target需要rolling P16，不支援{base_kind}")
    dates = [pd.Timestamp(row["effective_date"]).normalize() for row in records]
    if not dates or dates != sorted(set(dates)):
        raise MinTrade40BlockedError("P16 生效日缺少或重複")
    return tuple(records)


def _benchmark_calendar(root: Path, dataset: str, ticker: str) -> tuple[pd.DatetimeIndex, dict[str, Path]]:
    sources, duplicates = discover_dataset_csv_inputs(root, dataset)
    if duplicates or not sources:
        raise MinTrade40BlockedError(f"Min event Target dataset 缺少或有重複 ticker：{duplicates[:3]}")
    paths = {}
    for name, source in sources:
        key = normalize_ticker(name)
        if key in paths:
            raise MinTrade40BlockedError(f"Min event Target normalized ticker重複：{key}")
        paths[key] = Path(source)
    benchmark = normalize_ticker(ticker)
    if benchmark not in paths:
        raise MinTrade40BlockedError(f"Min event Target缺少交易日曆：{benchmark}")
    bench = load_dataset_frame(paths[benchmark], benchmark, min_rows=1)
    dates = pd.DatetimeIndex(bench.index).normalize()
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise MinTrade40BlockedError("Min event Target benchmark交易日曆不合法")
    return dates, paths


def _build_seed_events(
    *, root: Path, dataset: str, start: str, cutoff: str,
    spec: Mapping[str, Any], seed: int, kind: str, base_params: Any,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compute realized R only on legal Min breakout signal events.

    Each event freezes that day's P16 member and its original signal date.
    Horizon is T+40 *benchmark sessions*, not 40 trading rows for an
    intermittently halted ticker. Empty, unfilled and unexecutable outcomes
    have null R and are excluded (never filled with 0/-1).
    """
    from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
    horizon = int(spec["horizon_trading_bars"])
    calendar, paths = _benchmark_calendar(root, dataset, DEFAULT_LABEL_POLICY.benchmark_ticker)
    mapping = {pd.Timestamp(d).normalize(): i for i, d in enumerate(calendar)}
    last = pd.Timestamp(cutoff).normalize()
    mature = {date: calendar[idx + horizon] for date, idx in mapping.items()
              if idx + horizon < len(calendar) and calendar[idx + horizon] <= last}
    schedules = _schedule(kind, base_params)
    events = []
    coverage = Counter()
    for index, (ticker, path) in enumerate(sorted(paths.items()), start=1):
        if progress and (index % 250 == 0 or index == len(paths)):
            progress(f"Seed={seed} event target {index}/{len(paths)} tickers")
        try:
            frame = load_dataset_frame(path, ticker, min_rows=1)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            raise MinTrade40BlockedError(f"Min event Target無法讀取 {ticker} price：{exc}") from exc
        if len(frame) < 2:
            continue
        dates = pd.DatetimeIndex(frame.index).normalize()
        # AI: Portfolio normal_setup_index creates a candidate on the next
        # observed ticker bar, not on the signal bar.  This also covers the
        # previous year's final signal and suspended/illiquid tickers whose
        # preceding observed bar is older than the prior benchmark session.
        candidate_trade_dates = dates[1:]
        for record in schedules:
            params = record["params_obj"]
            # All-off, no DL ranking is guaranteed by the canonical P16 pair.
            if (bool(params.use_breakout_quality_filter) or
                    bool(params.use_breakout_quality_ranking)):
                raise MinTrade40BlockedError("Min event Target上游P16含DL／quality，禁止使用")
            effective = pd.Timestamp(record["effective_date"]).normalize()
            raw_end = record.get("effective_end_date")
            end_day = pd.Timestamp(raw_end).normalize() if raw_end is not None else last
            # AI: The P16 member is resolved for the day the candidate first
            # becomes tradable.  Filtering by signal_date would omit legitimate
            # lookback signals (e.g. 2020-12-31 -> 2021-01-04), and would use
            # the previous fold's params on year-end signals.
            active = np.flatnonzero(
                (candidate_trade_dates >= max(pd.Timestamp(start).normalize(), effective))
                & (candidate_trade_dates <= min(last, end_day))
            )
            if not len(active):
                continue
            # Compute the same formal signal calculator as the A2/Min replay.
            signals = build_signal_cache(frame, params, ticker=ticker)
            signal_mask = np.asarray(signals[1], dtype=bool)
            for pos in active:
                if not signal_mask[pos]:
                    continue
                day = dates[pos]
                if day not in mapping:
                    coverage["signal_not_on_market_calendar"] += 1
                    continue
                deadline = mature.get(day)
                if deadline is None:
                    coverage["not_40_session_mature"] += 1
                    continue
                result = simulate_realized_trade_path_label(
                    frame, ticker=ticker, signal_pos=int(pos), params=params,
                    teacher_effective_date=effective.strftime("%Y-%m-%d"),
                    precomputed_signals=signals,
                    horizon_end_date=deadline,
                )
                reason = str(result.reason)
                coverage[reason] += 1
                events.append({
                    "ticker": ticker, "signal_date": day.strftime("%Y-%m-%d"),
                    "horizon_date": deadline.strftime("%Y-%m-%d"),
                    "p16_effective_date": effective.strftime("%Y-%m-%d"),
                    "status": str(result.status), "reason": reason,
                    "fill_date": result.fill_date,
                    "entry_price": result.entry_price,
                    "exit_date": result.exit_date,
                    "exit_price": result.exit_price,
                    "exit_reason": result.exit_reason,
                    "realized_net_pnl": result.realized_net_pnl,
                    "realized_net_r": result.realized_net_r,
                    "forced_horizon_closeout": bool(result.forced_closeout),
                })
    if not events:
        raise MinTrade40BlockedError("Min event Target沒有已成熟且符合P16的Signal事件")
    if len(events) != len({(row["ticker"], row["signal_date"]) for row in events}):
        raise MinTrade40BlockedError("Min event Target ticker/date重複，P16時期不得交疊")
    event_count = len(events)
    filled = [row for row in events if row["realized_net_r"] is not None]
    values = np.asarray([row["realized_net_r"] for row in filled], dtype=float)
    if not values.size:
        raise MinTrade40BlockedError("Min event Target沒有可合法結算的已成交交易，不能發布完成報表")
    if not np.isfinite(values).all():
        raise MinTrade40BlockedError("Min event Target已成交樣本含非有限淨R")
    reasons = dict(sorted(coverage.items()))
    return {
        "seed": int(seed), "signal_events": event_count,
        "completed_trades": len(filled), "excluded_events": event_count - len(filled),
        "horizon_closeouts": sum(bool(row["forced_horizon_closeout"]) for row in filled),
        "positive_trades": int((values > 0).sum()),
        "mean_net_r": float(values.mean()) if values.size else None,
        "median_net_r": float(np.median(values)) if values.size else None,
        "p10_net_r": float(np.percentile(values, 10)) if values.size else None,
        "p90_net_r": float(np.percentile(values, 90)) if values.size else None,
        "min_signal_date": min(row["signal_date"] for row in events),
        "max_signal_date": max(row["signal_date"] for row in events),
        "reasons": reasons, "events": events,
    }


def _cache_path(root: Path, seed: int, identity: Mapping[str, Any]) -> Path:
    digest = canonical_json_sha256(identity)
    return (root / AUDIT_OUTPUT_ROOT / OUTPUT_SUBDIR / "seed_truth" / str(seed) /
            f"{TARGET_NAME}_{digest}.json")


def _load_seed_cache(path: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MinTrade40BlockedError(f"Min event Target既有工件無法讀取：{path.name}") from exc
    digest = payload.pop("integrity_sha256", None)
    if (payload.get("identity") != dict(identity) or
            digest != canonical_json_sha256(payload) or
            not isinstance(payload.get("truth"), dict)):
        raise MinTrade40BlockedError(f"Min event Target既有工件身分／內容衝突：{path.name}")
    return dict(payload["truth"])


def _save_seed_cache(path: Path, identity: Mapping[str, Any], truth: Mapping[str, Any]) -> None:
    payload = {"identity": dict(identity), "truth": dict(truth)}
    payload["integrity_sha256"] = canonical_json_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    except FileExistsError:
        current = _load_seed_cache(path, identity)
        if current != dict(truth):
            raise MinTrade40BlockedError("Min event Target同身分並行寫入衝突")


def _seed_identity(*, spec: Mapping[str, Any], dataset: str, price_sha: str,
                   param_sha: str, seed: int, start: str, cutoff: str) -> dict[str, Any]:
    return {
        "cache_schema": CACHE_SCHEMA, "label_id": str(spec["label_id"]),
        "horizon_sessions": int(spec["horizon_trading_bars"]),
        "cohort": str(spec["cohort"]),
        "signal_membership_policy": str(spec["signal_membership_policy"]),
        "source": "canonical P16 Base-Finalist-Best / Min no-DL",
        "trade_path_primitive": "A2 canonical signal/entry/exit/ledger v1 + 40D endpoint",
        "dataset": dataset, "price_sha256": price_sha,
        "parameter_sha256": param_sha, "seed": int(seed),
        "start": start, "research_cutoff": cutoff,
    }


def _rank_seed_events(events: list[Mapping[str, Any]]) -> dict[tuple[str, str], float]:
    """Produce daily hindsight ranks; EXCLUDED has *no* PnL, always lowest.

    Completed events order by actual ledger-net R; excluded/unfilled events
    share a lower *ranking-only* value, not an invented realized R. The score
    lookup uses the original signal date also for continuation orders.
    """
    if not events:
        raise MinTrade40BlockedError("MinTrade40_NetR沒有可排名的已成熟訊號")
    from filters.breakout_quality.mfe_safety_geometry import ensure_daily_percentile

    keys = [(str(row["ticker"]), str(row["signal_date"])) for row in events]
    if len(set(keys)) != len(keys):
        raise MinTrade40BlockedError("MinTrade40_NetR ticker/signal-date重複")
    ranks = pd.DataFrame({"ticker": [key[0] for key in keys],
                          "date": [key[1] for key in keys],
                          "realized": [row.get("realized_net_r") for row in events]})
    is_filled = ranks["realized"].notna()
    numeric = pd.to_numeric(ranks["realized"], errors="raise")
    if not np.isfinite(numeric.loc[is_filled].to_numpy(dtype=float)).all():
        raise MinTrade40BlockedError("MinTrade40_NetR完成交易含無效淨R")
    if not is_filled.any():
        raise MinTrade40BlockedError("MinTrade40_NetR沒有已成交的真值")
    daily_min = numeric.groupby(ranks["date"]).transform("min").fillna(0.0)
    # This sentinel exists ONLY inside ranking. The underlying events retain
    # realized_net_r=None for every unfilled/unexecutable event.
    numeric.loc[~is_filled] = daily_min.loc[~is_filled] - (1.0 + daily_min.loc[~is_filled].abs())
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise MinTrade40BlockedError("MinTrade40_NetR ranking sentinel不合法")
    ranked, _ = ensure_daily_percentile(
        pd.DataFrame({"date": ranks["date"], "value": numeric}),
        method="average_zero_based",
    )
    scores = ranked["percentile"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise MinTrade40BlockedError("MinTrade40_NetR每日rank非有限")
    return dict(zip(keys, scores.tolist()))


def build_seed_oracle_scores(
    definition, *, project_root: Path,
    resolve_p16_context: Callable[[], MinTrade40P16Context],
    quiet: bool = False,
) -> tuple[dict[int, dict[tuple[str, str], float]], dict[int, dict[str, Any]]]:
    """One canonical P16 label producer feeding normal Oracle Replay arms.

    Reuses prior event truth with strict per-seed price/P16 identity. It does
    not run or publish a separate 'Target validation' report.
    """
    report = build_report(
        definition, project_root=project_root,
        resolve_p16_context=resolve_p16_context, quiet=quiet,
    )
    spec = _target_spec(definition)
    by_seed: dict[int, dict[tuple[str, str], float]] = {}
    evidence: dict[int, dict[str, Any]] = {}
    for row in report["rows"]:
        seed = int(row["seed"])
        param_record = next((item for item in report["parameter_artifacts"]
                             if int(item["seed"]) == seed), None)
        if param_record is None:
            raise MinTrade40BlockedError("MinTrade40_NetR缺少同seed P16身分")
        identity = _seed_identity(
            spec=spec, dataset=str(report["dataset"]),
            price_sha=str(report["price_sha256"]),
            param_sha=str(param_record["param_sha256"]), seed=seed,
            start=str(report["start_date"]), cutoff=str(report["research_cutoff"]),
        )
        truth = _load_seed_cache(_cache_path(Path(project_root), seed, identity), identity)
        if truth is None:
            raise MinTrade40BlockedError("MinTrade40_NetR真值快取消失，禁止空分數fallback")
        by_seed[seed] = _rank_seed_events(truth["events"])
        evidence[seed] = {
            "signal_events": int(truth["signal_events"]),
            "completed_trades": int(truth["completed_trades"]),
            "excluded_events": int(truth["excluded_events"]),
            "horizon_closeouts": int(truth["horizon_closeouts"]),
            "score_key_mode": "signal_origin_date",
            "excluded_rank_policy": "rank_last_no_realized_r_imputation",
            "label_id": str(spec["label_id"]),
        }
    return by_seed, evidence


def build_report(
    definition, *, project_root: Path,
    resolve_p16_context: Callable[[], MinTrade40P16Context],
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    spec = _target_spec(definition)
    # The caller owns the P16 authorization and parameter-view truth.  The
    # event producer consumes it, never imports the upstream Oracle service.
    context = resolve_p16_context()
    cutoff, start = context.cutoff_date, context.start_date
    prepared, evidences = context.prepared, list(context.parameter_evidence)
    try:
        price_sha = price_inventory_sha256(root, context.dataset)
    except (OSError, ValueError) as exc:
        raise MinTrade40BlockedError(f"Min event Target價格工件無法驗證：{exc}") from exc
    identities = {}
    cached = {}
    for seed, param_sha, kind, base_params, _oracle_params in prepared:
        identity = _seed_identity(
            spec=spec, dataset=context.dataset, price_sha=price_sha,
            param_sha=param_sha, seed=seed, start=start, cutoff=cutoff,
        )
        identities[seed] = identity
        cached[seed] = _load_seed_cache(_cache_path(root, seed, identity), identity)
    decisions = {seed: ("REUSE" if cached[seed] is not None else "RUN")
                 for seed in identities}
    if not quiet:
        print("[Target 驗證｜MinTrade40_NetR] 身分核對：" +
              " / ".join(f"S{i+1}={decisions[seed]}" for i, seed in enumerate(identities)),
              flush=True)
    all_rows = []
    for seed, param_sha, kind, base_params, _oracle_params in prepared:
        truth = cached[seed]
        if truth is None:
            if not quiet:
                print(f"[RUN] MinTrade40_NetR seed={seed}", flush=True)
            truth = _build_seed_events(
                root=root, dataset=context.dataset, start=start, cutoff=cutoff,
                spec=spec, seed=seed, kind=kind, base_params=base_params,
                progress=None if quiet else lambda txt: print(f"[Target 驗證｜前置] {txt}", flush=True),
            )
            _save_seed_cache(_cache_path(root, seed, identities[seed]), identities[seed], truth)
        elif not quiet:
            print(f"[REUSE] MinTrade40_NetR seed={seed}", flush=True)
        all_rows.append({k: v for k, v in truth.items() if k != "events"})
    return {
        "schema_version": REPORT_SCHEMA, "status": "COMPLETED",
        "label_id": str(spec["label_id"]), "target": str(spec["target"]),
        "horizon_trading_bars": int(spec["horizon_trading_bars"]),
        "cohort": str(spec["cohort"]),
        "start_date": start, "research_cutoff": cutoff, "dataset": context.dataset,
        "benchmark_id": context.benchmark_id, "seeds": list(identities),
        "price_sha256": price_sha, "parameter_artifacts": evidences,
        "rows": all_rows, "cache_decisions": decisions,
        "read_only": True, "hindsight_non_pit": True,
        "limitations": [
            "P16 Min signal-event cohort; NOT the existing 643617-stock-day 40D universal cohort.",
            "Signal date + 40 benchmark trading sessions; regular Min STOP/IND_SELL executes sooner.",
            "Unfilled, no legal 40th-session sell, and same-day entry/exit are EXCLUDED, not zero or a loss.",
            "Prices, single-stock entry/exit, ledger-net fees/tax and initial risk use canonical Min/A2 primitives.",
            "Event-level isolated sizing is not dynamic multi-position Broker-Cash capital or portfolio RoMD.",
            "Per-seed labels depend on P16; never share raw Target values across seeds.",
            "Hindsight labels not PIT-legal, cannot be fed to Trading or OOS fitting.",
        ],
    }



__all__ = [
    "TARGET_ID", "TARGET_NAME", "MinTrade40BlockedError", "MinTrade40P16Context",
    "build_seed_oracle_scores", "build_report",
]

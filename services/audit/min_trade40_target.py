"""P16 Min 40-session realized-net-R event target, hindsight-only.

Its event-level realized R feeds only the existing [1] Oracle portfolio
comparison as a future-truth ranking score.  It does not feed DL fitting,
PIT/Trading, or Strategy Compare. The canonical Portfolio prepared market,
setup index and single-stock ledger/entry/exit primitives own all execution
semantics; this module does not re-scan raw CSV breakout signals.
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
from core.dataset_profiles import get_dataset_dir
from core.portfolio_fast_access import get_fast_dates, is_packed_market_data
from core.raw_universe_contract import resolve_raw_universe_required_min_rows
from core.portfolio_param_runtime import (
    build_active_param_ensemble_objects_from_payload,
    build_active_param_objects_from_payload,
)
from filters.breakout_quality.trade_path_label import simulate_realized_trade_path_label
from services.audit.oracle_replay_artifacts import price_inventory_sha256


TARGET_ID = "LABEL-mintrade40_realized_net_r_v1"
TARGET_NAME = "MinTrade40_NetR"
# This existing scientific label ID is fixed to 40 market trading sessions.
TARGET_HORIZON_SESSIONS = 40
REPORT_SCHEMA = 1
# Reuse existing per-seed evidence path; this is NOT a separate active report.
OUTPUT_SUBDIR = "breakout_quality/reusable/min_trade40_netr"
CACHE_SCHEMA = 2


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
            spec.get("signal_membership_policy") != "canonical_portfolio_setup_index_p16_v2"):
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


def _prepared_min_contexts(root: Path, dataset: str, kind: str, base_params: Any,
                           schedules: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Consume the same prepared source + normal setup index as portfolio Replay.

    The source is the canonical active-param Portfolio preparer, including the
    full optimizer raw-universe boundary and OHLCV sanitization.  It is NOT a
    second breakout scanner or a standalone backtest with different prices.
    """
    from services.portfolio_replay import (
        _load_contexts_for_active_ensemble_schedule,
        _load_contexts_for_active_schedule,
    )

    data_dir = str(get_dataset_dir(str(root), dataset))
    required_rows = resolve_raw_universe_required_min_rows(base_params)
    if kind == "rolling_active_param_ensemble":
        contexts = _load_contexts_for_active_ensemble_schedule(
            data_dir, schedules, verbose=False, keep_trade_logs=False,
            raw_universe_required_min_rows=required_rows,
        )
        return {key: rows[0] for key, rows in contexts.items() if len(rows) == 1}
    if kind == "rolling_oos_param_schedule":
        return _load_contexts_for_active_schedule(
            data_dir, schedules, verbose=False,
            raw_universe_required_min_rows=required_rows,
        )
    raise MinTrade40BlockedError(f"Min event Target未授權的P16回放類型：{kind}")


def _prepared_trade_path_view(ticker: str, market: Mapping[str, Any]):
    """Zero new signal rules: take the Portfolio-prepared OHLCV + indicator arrays."""
    if not is_packed_market_data(market):
        raise MinTrade40BlockedError(f"Min event Target缺少Portfolio packed行情：{ticker}")
    dates = pd.DatetimeIndex(get_fast_dates(market)).normalize()
    columns = ("Open", "High", "Low", "Close", "Volume")
    frame = pd.DataFrame({field: market[field] for field in columns}, index=dates, copy=False)
    frame.attrs["ticker"] = ticker
    frame.attrs["security_profile"] = market.get("security_profile")
    signals = (market["ATR"], market["is_setup"],
               market["ind_sell_signal"], market["buy_limit"])
    return frame, signals


def _build_seed_events(
    *, root: Path, dataset: str, start: str, cutoff: str,
    spec: Mapping[str, Any], seed: int, kind: str, base_params: Any,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Source only the formal Min Portfolio setup index; freeze each active P16.

    Candidate availability is determined by the same *prepared* market source
    as the production Portfolio Engine. Continuation/re-entry ranking inherits
    the originating scored signal via the canonical trading signal state; this
    function never independently guesses candidate or re-entry eligibility.
    """
    from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
    from services.portfolio_replay import PORTFOLIO_DEFAULT_BENCHMARK_TICKER

    benchmark = str(DEFAULT_LABEL_POLICY.benchmark_ticker)
    if benchmark != PORTFOLIO_DEFAULT_BENCHMARK_TICKER:
        raise MinTrade40BlockedError("Min event Target與Portfolio benchmark交易日曆身分不一致")
    horizon = int(spec["horizon_trading_bars"])
    last = pd.Timestamp(cutoff).normalize()
    beginning = pd.Timestamp(start).normalize()
    schedules = _schedule(kind, base_params)
    for record in schedules:
        params = record["params_obj"]
        if bool(params.use_breakout_quality_filter) or bool(params.use_breakout_quality_ranking):
            raise MinTrade40BlockedError("Min event Target上游P16含DL／quality，禁止使用")
    contexts = _prepared_min_contexts(root, dataset, kind, base_params, schedules)
    if len(contexts) != len(schedules):
        raise MinTrade40BlockedError("Min event Target正式P16 context缺少生效期間")

    # The benchmark is the SAME sanitized, packed price series used by the
    # real Portfolio replay. Its market sessions define T+40 maturity.
    first_context = contexts[str(schedules[0]["effective_date_text"])]
    benchmark_fast = (first_context.get("all_dfs_fast") or {}).get(benchmark)
    if benchmark_fast is None:
        raise MinTrade40BlockedError("Min event Target正式Portfolio context缺少0050交易日曆")
    calendar = pd.DatetimeIndex(get_fast_dates(benchmark_fast)).normalize()
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise MinTrade40BlockedError("Min event Target正式Portfolio交易日曆不合法")
    mapping = {day: i for i, day in enumerate(calendar)}
    mature = {date: calendar[idx + horizon] for date, idx in mapping.items()
              if idx + horizon < len(calendar) and calendar[idx + horizon] <= last}
    events = []
    coverage = Counter()
    for member_i, record in enumerate(schedules, 1):
        effective_text = str(record["effective_date_text"])
        context = contexts[effective_text]
        market_by_ticker = context.get("all_dfs_fast") or {}
        setup_index = context.get("normal_setup_index")
        if not isinstance(setup_index, dict):
            raise MinTrade40BlockedError(f"正式Min缺少normal_setup_index：{effective_text}")
        params = record["params_obj"]
        effective = pd.Timestamp(record["effective_date"]).normalize()
        raw_end = record.get("effective_end_date")
        end_day = pd.Timestamp(raw_end).normalize() if raw_end is not None else last
        period_start = max(beginning, effective)
        period_end = min(last, end_day)
        if progress:
            progress(f"Seed={seed} canonical Min context {member_i}/{len(schedules)}")
        prepared_views: dict[str, tuple[pd.DataFrame, tuple[Any, ...]]] = {}
        for candidate_day, entries in sorted(setup_index.items()):
            trade_day = pd.Timestamp(candidate_day).normalize()
            if trade_day < period_start or trade_day > period_end:
                continue
            # Consume EXACTLY the canonical candidate event tuples.  In
            # particular, signal date is the prior OBSERVED ticker bar, not
            # assumed to be the previous benchmark session.
            for ticker, signal_pos, next_pos in entries:
                market = market_by_ticker.get(ticker)
                if market is None:
                    raise MinTrade40BlockedError(f"正式Min setup缺少股票快取：{ticker}")
                view = prepared_views.get(ticker)
                if view is None:
                    view = _prepared_trade_path_view(ticker, market)
                    prepared_views[ticker] = view
                frame, signals = view
                if next_pos != signal_pos + 1 or next_pos >= len(frame):
                    raise MinTrade40BlockedError("正式Min setup持有無效原始訊號位置")
                if pd.Timestamp(frame.index[next_pos]).normalize() != trade_day:
                    raise MinTrade40BlockedError("正式Min setup與交易日錯位")
                if not bool(signals[1][signal_pos]):
                    raise MinTrade40BlockedError("正式Min setup與prepared signal flags衝突")
                signal_day = pd.Timestamp(frame.index[signal_pos]).normalize()
                deadline = mature.get(signal_day)
                if deadline is None:
                    if signal_day not in mapping:
                        # A new signal without an exchange session cannot be
                        # assigned the specified forty-session label.
                        raise MinTrade40BlockedError(
                            f"正式Min訊號日不在交易日曆：ticker={ticker}, date={signal_day.date()}"
                        )
                    coverage["not_40_session_mature"] += 1
                    continue
                result = simulate_realized_trade_path_label(
                    frame, ticker=ticker, signal_pos=int(signal_pos), params=params,
                    teacher_effective_date=effective_text,
                    precomputed_signals=signals,
                    horizon_end_date=deadline,
                )
                reason = str(result.reason)
                coverage[reason] += 1
                events.append({
                    "ticker": str(ticker), "signal_date": signal_day.strftime("%Y-%m-%d"),
                    "horizon_date": deadline.strftime("%Y-%m-%d"),
                    "p16_effective_date": effective_text,
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

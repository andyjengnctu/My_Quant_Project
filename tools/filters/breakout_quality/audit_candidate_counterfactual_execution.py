"""11J research-only canonical per-candidate counterfactual execution audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality_policy import BREAKOUT_QUALITY_DEFAULT_FILTER_ID, BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
from core.exact_accounting import calc_ratio_from_milli
from core.portfolio_exits import closeout_open_positions
from core.portfolio_fast_access import get_fast_close, get_fast_pos, get_fast_value
from core.position_step import execute_bar_step
from core.portfolio_entries import build_candidate_plan_seed
from core.trade_plans import execute_pre_market_entry_plan
from filters.breakout_quality.continuous_target import STRATEGY_ALIGNED_NO_TIME_TARGET_ID
from filters.breakout_quality.contract import LABEL_PASS, LABEL_REJECT
from tools.filters.breakout_quality.audit_selection_strategy_realization import (
    AUDIT_JSON_FILENAME as SOURCE_AUDIT_JSON_FILENAME,
    AUDIT_DIRNAME as SOURCE_AUDIT_DIRNAME,
    DEFAULT_NESTED_OOS_END_DATE,
    _attach_targets,
    _build_controlled_param_source_pair,
    _date_text,
    _load_param_source,
    _output_dir as source_output_dir,
    _sha256_file,
    _target_lookup,
    _unique_signals,
    _validate_param_coverage,
)
from tools.filters.breakout_quality.common import PROJECT_ROOT, write_json
from tools.filters.breakout_quality.strategy_compare import (
    COMPARISON_MODE_HARD_FILTER,
    _flatten_candidate_replay_rows,
    _run_scenario,
    _scenario_summary,
)
from tools.filters.breakout_quality.train_continuous_ranker import _spearman

AUDIT_SCHEMA_VERSION = 1
EXPERIMENT_NAME = "11J Canonical Per-candidate Counterfactual Execution Audit"
AUDIT_DIRNAME = "candidate_counterfactual_execution_audit"
AUDIT_JSON_FILENAME = "candidate_counterfactual_execution_audit.json"
AUDIT_MARKDOWN_FILENAME = "candidate_counterfactual_execution_audit.md"
SIGNALS_FILENAME = "candidate_counterfactual_execution_signals.csv"
FILLED_FILENAME = "candidate_counterfactual_filled_trades.csv"
UNFILLED_FILENAME = "candidate_counterfactual_unfilled_signals.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "11J research-only canonical per-candidate counterfactual execution audit；"
            "忽略portfolio capacity／cash competition，但保留正式進場、shadow、停損、停利與出場SSOT"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _candidate_key(
    candidate: dict[str, Any],
    today: Any,
    *,
    snapshot: dict[str, Any] | None = None,
) -> tuple[str, str]:
    canonical = dict(snapshot or {})
    ticker = str(canonical.get("ticker") or candidate.get("ticker") or "").strip()
    signal_date = _date_text(
        canonical.get("signal_date")
        or canonical.get("candidate_date")
        or canonical.get("trade_date")
        or candidate.get("signal_date")
        or candidate.get("candidate_date")
        or candidate.get("trade_date")
        or today
    )
    if not ticker or not signal_date:
        raise ValueError(
            "11J candidate缺少canonical ticker／signal_date: "
            f"candidate={candidate}, snapshot={snapshot}"
        )
    return ticker, signal_date


def _entry_type_text(value: Any) -> str:
    text = str(value or "normal")
    return text if text else "normal"


class CandidateCounterfactualReplay(dict):
    """Observer threaded through canonical portfolio replay via the existing replay_counts object."""

    def __init__(self, *, candidate_cutoff: str, defer_finalize: bool = False):
        super().__init__()
        self.candidate_cutoff = _date_text(candidate_cutoff)
        self.defer_finalize = bool(defer_finalize)
        self.states: dict[tuple[str, str], dict[str, Any]] = {}
        self.finalized = False

    def _state_for_candidate(
        self,
        candidate: dict[str, Any],
        today: Any,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = _candidate_key(candidate, today, snapshot=snapshot)
        state = self.states.get(key)
        today_text = _date_text(today)
        if state is None:
            state = {
                "ticker": key[0],
                "signal_date": key[1],
                "candidate_type": _entry_type_text(candidate.get("type")),
                "entry_source": str(candidate.get("entry_source") or ""),
                "first_candidate_date": today_text,
                "last_candidate_date": today_text,
                "qualified_occurrence_count": 0,
                "orderable_occurrence_count": 0,
                "entry_attempt_count": 0,
                "missed_buy_count": 0,
                "locked_or_untradeable_count": 0,
                "filled": False,
                "closed": False,
                "entry_date": "",
                "exit_date": "",
                "exit_type": "",
                "r_multiple": math.nan,
                "position": None,
                "params_obj": None,
                "entry_context": None,
                "last_px": math.nan,
                "missed_sell_count": 0,
            }
            self.states[key] = state
        state["last_candidate_date"] = today_text
        return state

    def begin_replay_day(self, *, today, all_dfs_fast, fallback_params):
        today_text = _date_text(today)
        for key in sorted(self.states):
            state = self.states[key]
            position = state.get("position")
            if position is None or bool(state.get("closed")):
                continue
            ticker = state["ticker"]
            context = state.get("entry_context") if isinstance(state.get("entry_context"), dict) else {}
            position_all_dfs = context.get("all_dfs_fast") or all_dfs_fast
            fast_df = position_all_dfs.get(ticker)
            if fast_df is None:
                continue
            t_pos = get_fast_pos(fast_df, today)
            if t_pos <= 0:
                continue
            y_pos = t_pos - 1
            params = state.get("params_obj") or fallback_params
            position, _freed_cash, _pnl, events = execute_bar_step(
                position,
                get_fast_value(fast_df, "ATR", pos=y_pos),
                get_fast_value(fast_df, "ind_sell_signal", pos=y_pos),
                get_fast_close(fast_df, pos=y_pos),
                get_fast_value(fast_df, "Open", pos=t_pos),
                get_fast_value(fast_df, "High", pos=t_pos),
                get_fast_value(fast_df, "Low", pos=t_pos),
                get_fast_close(fast_df, pos=t_pos),
                get_fast_value(fast_df, "Volume", pos=t_pos),
                params,
                current_date=today,
                y_high=get_fast_value(fast_df, "High", pos=y_pos),
                return_milli=True,
                record_exec_contexts=False,
                sync_display_fields=True,
            )
            position["last_px"] = get_fast_close(fast_df, pos=t_pos)
            state["position"] = position
            state["last_px"] = position["last_px"]
            if "MISSED_SELL" in events:
                state["missed_sell_count"] = int(state.get("missed_sell_count", 0)) + 1
            if "STOP" in events or "IND_SELL" in events:
                state["closed"] = True
                state["exit_date"] = today_text
                state["exit_type"] = "STOP" if "STOP" in events else "IND_SELL"
                state["r_multiple"] = calc_ratio_from_milli(
                    int(position.get("realized_pnl_milli", 0) or 0),
                    int(position.get("initial_risk_total_milli", 0) or 0),
                )
                state["position"] = None

    def observe_replay_candidates(
        self,
        *,
        today,
        qualified_candidates,
        qualified_candidate_snapshots=None,
        orderable_candidates,
        orderable_candidate_snapshots=None,
        all_dfs_fast,
        sizing_equity,
        fallback_params,
    ):
        today_text = _date_text(today)
        if today_text > self.candidate_cutoff:
            return
        qualified_rows = list(qualified_candidates or [])
        qualified_snapshots = list(qualified_candidate_snapshots or [])
        orderable_rows = list(orderable_candidates or [])
        orderable_snapshots = list(orderable_candidate_snapshots or [])
        if qualified_snapshots and len(qualified_snapshots) != len(qualified_rows):
            raise ValueError(
                "11J qualified candidate與canonical snapshot數量不一致: "
                f"candidates={len(qualified_rows)}, snapshots={len(qualified_snapshots)}"
            )
        if orderable_snapshots and len(orderable_snapshots) != len(orderable_rows):
            raise ValueError(
                "11J orderable candidate與canonical snapshot數量不一致: "
                f"candidates={len(orderable_rows)}, snapshots={len(orderable_snapshots)}"
            )
        if not qualified_snapshots:
            qualified_snapshots = [None] * len(qualified_rows)
        if not orderable_snapshots:
            orderable_snapshots = [None] * len(orderable_rows)

        for candidate, snapshot in zip(qualified_rows, qualified_snapshots):
            state = self._state_for_candidate(candidate, today, snapshot=snapshot)
            state["qualified_occurrence_count"] = int(state["qualified_occurrence_count"]) + 1

        attempted_keys: set[tuple[str, str]] = set()
        for candidate, snapshot in zip(orderable_rows, orderable_snapshots):
            key = _candidate_key(candidate, today, snapshot=snapshot)
            if key in attempted_keys:
                continue
            attempted_keys.add(key)
            state = self._state_for_candidate(candidate, today, snapshot=snapshot)
            state["orderable_occurrence_count"] = int(state["orderable_occurrence_count"]) + 1
            if bool(state.get("filled")) or bool(state.get("closed")):
                continue
            qty = int(candidate.get("qty", 0) or 0)
            if qty <= 0:
                continue
            params = candidate.get("params_obj") or fallback_params
            plan = build_candidate_plan_seed(candidate, sizing_equity=sizing_equity)
            plan["qty"] = qty
            plan["is_orderable"] = True
            context = candidate.get("_ensemble_context") if isinstance(candidate.get("_ensemble_context"), dict) else {}
            candidate_all_dfs = context.get("all_dfs_fast") or all_dfs_fast
            ticker = state["ticker"]
            fast_df = candidate_all_dfs[ticker]
            t_pos = int(candidate.get("today_pos", get_fast_pos(fast_df, today)))
            y_pos = int(candidate.get("yesterday_pos", t_pos - 1))
            if t_pos <= 0 or y_pos < 0:
                continue
            state["entry_attempt_count"] = int(state["entry_attempt_count"]) + 1
            entry_result = execute_pre_market_entry_plan(
                entry_plan=plan,
                t_open=get_fast_value(fast_df, "Open", pos=t_pos),
                t_high=get_fast_value(fast_df, "High", pos=t_pos),
                t_low=get_fast_value(fast_df, "Low", pos=t_pos),
                t_close=get_fast_close(fast_df, pos=t_pos),
                t_volume=get_fast_value(fast_df, "Volume", pos=t_pos),
                y_close=get_fast_close(fast_df, pos=y_pos),
                params=params,
                entry_type=_entry_type_text(candidate.get("type")),
                ticker=ticker,
                security_profile=candidate.get("security_profile"),
                trade_date=today,
            )
            if not bool(entry_result.get("filled")):
                if bool(entry_result.get("count_as_missed_buy")):
                    state["missed_buy_count"] = int(state["missed_buy_count"]) + 1
                else:
                    state["locked_or_untradeable_count"] = int(state["locked_or_untradeable_count"]) + 1
                continue
            position = entry_result["position"]
            position["_entry_params_obj"] = params
            if context:
                position["_entry_context"] = context
            position["signal_date"] = state["signal_date"]
            position["candidate_date"] = today_text
            position["candidate_type"] = state["candidate_type"]
            position["ticker"] = ticker
            position["last_px"] = get_fast_close(fast_df, pos=t_pos)
            state["filled"] = True
            state["entry_date"] = today_text
            state["position"] = position
            state["params_obj"] = params
            state["entry_context"] = context
            state["last_px"] = position["last_px"]

    def finalize_replay(self, *, last_date, fallback_params):
        if self.defer_finalize:
            return
        if self.finalized:
            return
        self.finalized = True
        last_text = _date_text(last_date)
        for key in sorted(self.states):
            state = self.states[key]
            position = state.get("position")
            if position is None or bool(state.get("closed")):
                continue
            ticker = state["ticker"]
            stats: list[dict[str, Any]] = []
            portfolio = {ticker: position}
            closeout_open_positions(
                portfolio=portfolio,
                cash=0,
                params=state.get("params_obj") or fallback_params,
                trade_history=[],
                is_training=True,
                closed_trades_stats=stats,
                normal_trade_count=0,
                extended_trade_count=0,
                last_date=last_date,
            )
            if len(stats) != 1:
                raise ValueError(f"11J期末結算筆數異常: key={key}, rows={len(stats)}")
            state["closed"] = True
            state["exit_date"] = last_text
            state["exit_type"] = "FORCED_CLOSE"
            state["r_multiple"] = float(stats[0]["r_mult"])
            state["position"] = None

    def signal_frame(self) -> pd.DataFrame:
        rows=[]
        for key in sorted(self.states):
            state=self.states[key]
            rows.append({
                "ticker": state["ticker"],
                "signal_date": state["signal_date"],
                "candidate_type": state["candidate_type"],
                "entry_source": state["entry_source"],
                "first_candidate_date": state["first_candidate_date"],
                "last_candidate_date": state["last_candidate_date"],
                "qualified_occurrence_count": int(state["qualified_occurrence_count"]),
                "orderable_occurrence_count": int(state["orderable_occurrence_count"]),
                "entry_attempt_count": int(state["entry_attempt_count"]),
                "missed_buy_count": int(state["missed_buy_count"]),
                "locked_or_untradeable_count": int(state["locked_or_untradeable_count"]),
                "missed_sell_count": int(state["missed_sell_count"]),
                "was_orderable": int(state["orderable_occurrence_count"]) > 0,
                "filled": bool(state["filled"]),
                "closed": bool(state["closed"]),
                "entry_date": state["entry_date"],
                "exit_date": state["exit_date"],
                "exit_type": state["exit_type"],
                "r_multiple": state["r_multiple"],
            })
        return pd.DataFrame(rows)


def _source_report(filter_id: str) -> tuple[dict[str, Any], Path]:
    path = source_output_dir(filter_id) / SOURCE_AUDIT_JSON_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"11J需要已完成11I report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not str(payload.get("status") or "").startswith("RESULT_AVAILABLE"):
        raise ValueError("11J只接受已完成11I report")
    tm = dict(payload.get("trade_metrics") or {})
    overall = tm.get("spearman_target_vs_realized_r")
    pass_rho = ((tm.get("label_conditional") or {}).get("PASS") or {}).get("spearman_target_vs_realized_r")
    coverage = (payload.get("coverage") or {}).get("actual_trade_coverage_vs_qualified")
    if overall is None or pass_rho is None or float(overall) <= 0.0 or float(pass_rho) <= 0.0:
        raise ValueError("11J需要11I overall與PASS Target↔R均為正")
    if coverage is None or not (0.0 <= float(coverage) < 1.0):
        raise ValueError("11J需要有效且未滿覆蓋的11I actual-trade coverage")
    for item in (payload.get("artifacts") or {}).values():
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "")
        expected = str(item.get("sha256") or "")
        if not filename or not expected:
            continue
        artifact = path.parent / filename
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"11J偵測到11I artifact SHA256不一致: {artifact}")
    return payload, path


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame[
        frame["target_match"].astype(bool)
        & frame["filled"].astype(bool)
        & np.isfinite(pd.to_numeric(frame["r_multiple"], errors="coerce"))
    ].copy()
    result: dict[str, Any] = {
        "qualified_signal_count": int(len(frame)),
        "orderable_signal_count": int(frame["was_orderable"].sum()),
        "filled_signal_count": int(frame["filled"].sum()),
        "closed_signal_count": int(frame["closed"].sum()),
        "target_matched_filled_count": int(len(valid)),
        "orderable_coverage_vs_qualified": float(frame["was_orderable"].mean()) if len(frame) else None,
        "fill_coverage_vs_qualified": float(frame["filled"].mean()) if len(frame) else None,
        "fill_coverage_vs_orderable": (
            float(frame.loc[frame["was_orderable"], "filled"].mean())
            if bool(frame["was_orderable"].any()) else None
        ),
        "strategy_r_coverage_vs_qualified": float(len(valid) / len(frame)) if len(frame) else None,
    }
    if valid.empty:
        return result
    target = valid["target_raw_r"].to_numpy(dtype=np.float64)
    r = valid["r_multiple"].to_numpy(dtype=np.float64)
    count = max(1, int(math.ceil(len(valid) * 0.10)))
    ordered = valid.sort_values("target_raw_r", kind="mergesort")
    result.update({
        "spearman_target_vs_counterfactual_r": _spearman(target, r),
        "top_target_decile_average_r": float(ordered.tail(count)["r_multiple"].mean()),
        "bottom_target_decile_average_r": float(ordered.head(count)["r_multiple"].mean()),
        "label_conditional": {},
    })
    for name,value in (("PASS",LABEL_PASS),("REJECT",LABEL_REJECT)):
        sub=valid[valid["label"]==value]
        result["label_conditional"][name]={
            "rows": int(len(sub)),
            "spearman_target_vs_counterfactual_r": (
                _spearman(sub["target_raw_r"].to_numpy(dtype=np.float64),sub["r_multiple"].to_numpy(dtype=np.float64))
                if len(sub)>=2 else None
            ),
        }
    return result


def _render_markdown(payload: dict[str, Any]) -> str:
    def fmt(v,d=4): return "-" if v is None else f"{float(v):.{d}f}"
    def pct(v): return "-" if v is None else f"{float(v)*100:.2f}%"
    m=payload["metrics"]
    source=payload["source_11i"]
    return "\n".join([
        "# 11J Canonical Per-candidate Counterfactual Execution Audit",
        "",
        "- Runtime：research-only；Selection nested OOS candidate replay。",
        "- 忽略portfolio capacity與cash competition，但保留canonical order／shadow／exit／accounting規則。",
        "- 未成交候選維持unlabeled，不標0R。",
        "",
        "## 1. Coverage",
        "",
        f"- Qualified signals：`{m['qualified_signal_count']:,}`。",
        f"- Orderable：`{m['orderable_signal_count']:,}`（{pct(m.get('orderable_coverage_vs_qualified'))}）。",
        f"- Counterfactual filled：`{m['filled_signal_count']:,}`；vs qualified `{pct(m.get('fill_coverage_vs_qualified'))}`；vs orderable `{pct(m.get('fill_coverage_vs_orderable'))}`。",
        f"- Target-matched strategy R：`{m['target_matched_filled_count']:,}`；vs qualified `{pct(m.get('strategy_r_coverage_vs_qualified'))}`。",
        f"- 11I actual portfolio coverage：`{pct(source.get('actual_trade_coverage_vs_qualified'))}`。",
        "",
        "## 2. Counterfactual strategy realization",
        "",
        f"- No-time Target↔counterfactual R：`{fmt(m.get('spearman_target_vs_counterfactual_r'))}`。",
        f"- PASS內 Target↔counterfactual R：`{fmt((m.get('label_conditional') or {}).get('PASS',{}).get('spearman_target_vs_counterfactual_r'))}`。",
        f"- Target top／bottom decile R：`{fmt(m.get('top_target_decile_average_r'))}`／`{fmt(m.get('bottom_target_decile_average_r'))}`。",
        "",
        "## 3. Boundary",
        "",
        "- 本結果可判斷capacity／cash competition移除後的可執行strategy R coverage，但仍不為限價未成交訊號填值。",
        "- 本輪不建立Target arrays、不訓練、不選epoch、不調threshold或optimizer。",
        "- 是否建立strategy-realization target，必須等本報表review；11J本身不授權模型。",
        "",
    ])


def main(argv=None) -> int:
    args=parse_args(argv)
    started=time.perf_counter()
    source_report, source_report_path = _source_report(str(args.filter_id))
    params_path=Path(str((source_report.get("params") or {}).get("path") or "")).resolve()
    expected_params_hash=str((source_report.get("params") or {}).get("sha256") or "")
    if not params_path.is_file() or (expected_params_hash and _sha256_file(params_path)!=expected_params_hash):
        raise ValueError(f"11J nested params遺失或SHA256不一致: {params_path}")
    source=_load_param_source(params_path)
    start_date=str((source_report.get("period") or {}).get("start") or "")
    candidate_cutoff=str((source_report.get("period") or {}).get("end") or "")
    replay_end=DEFAULT_NESTED_OOS_END_DATE
    kind,param_first,param_last=_validate_param_coverage(source,start_date=start_date,end_date=replay_end)
    (
        param_source_kind,
        no_filter_params,
        _quality_params,
        _no_filter_payload,
        _quality_payload,
        _ensemble_policy,
    )=_build_controlled_param_source_pair(
        source,
        filter_id=str(args.filter_id),
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_HARD_FILTER,
    )
    if param_source_kind!=kind:
        raise ValueError("11J param source kind在rewrite前後不一致")
    data_dir=Path(str((source_report.get("dataset") or {}).get("path") or "")).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"11J找不到11I dataset: {data_dir}")
    strategy=dict(source_report.get("strategy") or {})
    max_positions=int(strategy.get("max_positions",10) or 10)
    enable_rotation=bool(strategy.get("rotation",False))
    tracker=CandidateCounterfactualReplay(
        candidate_cutoff=candidate_cutoff,
        defer_finalize=True,
    )
    discovery_scenario=_run_scenario(
        name="11J_candidate_discovery",
        data_dir=data_dir,
        param_source_kind=param_source_kind,
        params=no_filter_params,
        start_date=start_date,
        end_date=candidate_cutoff,
        max_positions=max_positions,
        enable_rotation=enable_rotation,
        quiet=bool(args.quiet),
        replay_counts=tracker,
    )
    discovery_summary=_scenario_summary(discovery_scenario)
    lookup,target_manifest=_target_lookup(str(args.filter_id))
    discovery_qualified=_flatten_candidate_replay_rows(tracker,"candidate_rows")
    discovery_qualified=_attach_targets(discovery_qualified,lookup)
    discovery_qualified_unique=_unique_signals(discovery_qualified)
    source_qualified_count=int((source_report.get("coverage") or {}).get("qualified_unique_signal_count",0) or 0)
    canonical_discovery_count=len(discovery_qualified_unique)
    if source_qualified_count and canonical_discovery_count!=source_qualified_count:
        raise ValueError(
            "11J canonical候選快照與11I qualified unique signals不一致: "
            f"expected={source_qualified_count}, actual={canonical_discovery_count}"
        )
    if len(tracker.states)!=canonical_discovery_count:
        canonical_keys=set(zip(
            discovery_qualified_unique["ticker"].astype(str),
            discovery_qualified_unique["target_date"].astype(str),
        ))
        state_keys=set(tracker.states)
        missing=sorted(canonical_keys-state_keys)[:10]
        unexpected=sorted(state_keys-canonical_keys)[:10]
        raise ValueError(
            "11J observer state未忠實採用canonical候選快照: "
            f"canonical={canonical_discovery_count}, states={len(tracker.states)}, "
            f"missing_sample={missing}, unexpected_sample={unexpected}"
        )

    discovered_state_count=len(tracker.states)
    management_start=_date_text(pd.Timestamp(candidate_cutoff)+pd.Timedelta(days=1))
    tracker.defer_finalize=False
    if management_start<=replay_end:
        management_scenario=_run_scenario(
            name="11J_counterfactual_management",
            data_dir=data_dir,
            param_source_kind=param_source_kind,
            params=no_filter_params,
            start_date=management_start,
            end_date=replay_end,
            max_positions=max_positions,
            enable_rotation=enable_rotation,
            quiet=bool(args.quiet),
            replay_counts=tracker,
        )
        management_summary=_scenario_summary(management_scenario)
    else:
        tracker.finalize_replay(last_date=pd.Timestamp(candidate_cutoff),fallback_params=None)
        management_summary={"skipped":True,"reason":"candidate cutoff equals replay end"}
    if len(tracker.states)!=discovered_state_count:
        raise ValueError(
            "11J持倉管理階段不得新增candidate states: "
            f"before={discovered_state_count}, after={len(tracker.states)}"
        )

    signals=tracker.signal_frame()
    signals=_attach_targets(signals,lookup)
    source_trade_path=source_report_path.parent / str(((source_report.get("artifacts") or {}).get("trade_matches") or {}).get("filename") or "")
    actual=pd.read_csv(source_trade_path,encoding="utf-8-sig") if source_trade_path.is_file() else pd.DataFrame()
    actual_keys=set()
    if not actual.empty and {"ticker","target_date"}.issubset(actual.columns):
        actual_keys=set(zip(actual["ticker"].astype(str),actual["target_date"].astype(str)))
    signals["was_actual_portfolio_trade"]=[(str(t),str(d)) in actual_keys for t,d in zip(signals["ticker"],signals["target_date"])]
    metrics=_metrics(signals)

    output_dir=source_output_dir(str(args.filter_id)) / AUDIT_DIRNAME
    output_dir.mkdir(parents=True,exist_ok=True)
    signals_path=output_dir / SIGNALS_FILENAME
    filled_path=output_dir / FILLED_FILENAME
    unfilled_path=output_dir / UNFILLED_FILENAME
    signals.to_csv(signals_path,index=False,encoding="utf-8-sig")
    signals[signals["filled"]].to_csv(filled_path,index=False,encoding="utf-8-sig")
    signals[~signals["filled"]].to_csv(unfilled_path,index=False,encoding="utf-8-sig")
    artifacts={}
    for key,path in (("signals",signals_path),("filled",filled_path),("unfilled",unfilled_path)):
        artifacts[key]={"filename":path.name,"sha256":_sha256_file(path),"size_bytes":int(path.stat().st_size)}
    payload={
        "schema_version":AUDIT_SCHEMA_VERSION,
        "experiment":EXPERIMENT_NAME,
        "status":"RESULT_AVAILABLE_PENDING_REVIEW",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "elapsed_sec":float(time.perf_counter()-started),
        "filter_id":str(args.filter_id),
        "source_11i":{
            "path":str(source_report_path),
            "sha256":_sha256_file(source_report_path),
            "actual_trade_coverage_vs_qualified":float((source_report.get("coverage") or {}).get("actual_trade_coverage_vs_qualified",math.nan)),
            "target_vs_actual_r":(source_report.get("trade_metrics") or {}).get("spearman_target_vs_realized_r"),
        },
        "period":{"candidate_start":start_date,"candidate_end":candidate_cutoff,"execution_end":replay_end},
        "params":{"path":str(params_path),"sha256":_sha256_file(params_path),"source_kind":param_source_kind,"coverage_start":param_first,"coverage_end":param_last},
        "target":{"target_id":STRATEGY_ALIGNED_NO_TIME_TARGET_ID,"manifest_generated_at_utc":target_manifest.get("generated_at_utc")},
        "metrics":metrics,
        "replay":{
            "candidate_discovery":{
                "start":start_date,
                "end":candidate_cutoff,
                "summary":discovery_summary,
            },
            "position_management":{
                "start":management_start,
                "end":replay_end,
                "summary":management_summary,
            },
        },
        "artifacts":artifacts,
        "training_performed":False,
        "runtime_eligible":False,
        "interpretation_contract":{
            "selection_nested_oos_only":True,
            "portfolio_capacity_and_cash_competition_ignored":True,
            "canonical_entry_shadow_exit_accounting_reused":True,
            "unfilled_candidates_remain_unlabeled":True,
            "new_model_not_authorized_until_result_review":True,
        },
    }
    json_path=output_dir / AUDIT_JSON_FILENAME
    md_path=output_dir / AUDIT_MARKDOWN_FILENAME
    write_json(json_path,payload)
    md_path.write_text(_render_markdown(payload),encoding="utf-8")
    print("11J candidate counterfactual execution audit完成")
    print(
        f"qualified={metrics['qualified_signal_count']:,} orderable={metrics['orderable_signal_count']:,} "
        f"filled={metrics['filled_signal_count']:,} target_matched={metrics['target_matched_filled_count']:,}"
    )
    print(
        "- Target↔counterfactual R: "
        f"overall={metrics.get('spearman_target_vs_counterfactual_r')} "
        f"pass={(metrics.get('label_conditional') or {}).get('PASS',{}).get('spearman_target_vs_counterfactual_r')}"
    )
    print(f"已輸出: {md_path}")
    print(f"已輸出: {json_path}")
    return 0


__all__=[
    "AUDIT_JSON_FILENAME",
    "AUDIT_MARKDOWN_FILENAME",
    "CandidateCounterfactualReplay",
    "main",
    "parse_args",
]

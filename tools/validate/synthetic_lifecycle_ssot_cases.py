"""Capability tests for the shared lifecycle, fact boundary and user discretion.

AI: Isolated fixtures mock market storage/provider I/O only in the persistent
integration case. They do not replace strategy transitions, risk sizing,
account/pending/OMS persistence, or chart projection. No formal runner is invoked.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from .checks import bind_synthetic_case
from core.entry_plans import (
    accept_entry_quantity_decision, build_position_from_frozen_entry_plan,
    build_counterfactual_shadow_position_from_plan, execute_pre_market_entry_plan,
)
from core.params_io import params_to_json_dict
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.position_management import POSITION_MANAGEMENT_FIELDS, complete_position_entry_session, step_position_management
from core.position_replay import replay_confirmed_position_management
from core.position_step import execute_bar_step
from core.file_integrity import canonical_json_sha256, atomic_write_json
from core.serialization_utils import json_native_value
from services.trading.lifecycle_sync_status import SYNC_STATUS_LATEST
from core.trading_account_state import effective_trading_account_events, validate_trading_account_state
from core.trading_order_state import build_empty_trading_order_state
from core.trading_lifecycle_plans import resolve_confirmed_entry_plan_from_frame
from services.trading.account_state import initialize_trading_account_state, record_strategy_trading_buy, load_trading_account_state
from services.trading.strategy_param_runtime import build_trading_candidate_strategy_lineage, resolve_trading_position_management_binding
from services.trading.pending_entry_state import create_trading_pending_entry, load_trading_pending_entry_state, validate_trading_pending_entry_state
from services.trading.pending_entry_service import _pending_entry_payload
from services.trading.lifecycle_sync import run_trading_lifecycle_sync
from services.trading.protection_planning import build_trading_protection_plan
from services.trading.single_stock_inspection import build_trading_single_stock_lifecycle_timeline


def _params(base):
    return replace(base, use_breakout_quality_filter=False, use_breakout_quality_ranking=False,
                   atr_times_init=2.0, atr_times_trail=1.0, tp_percent=0.5)


def _seed(ticker="2330", qty=1000):
    return {"ticker": ticker, "qty": qty, "max_qty": qty, "is_orderable": True,
            "limit_price": 100.0, "init_sl": 95.0, "init_trail": 95.0,
            "target_price": 110.0, "entry_atr": None, "entry_type": "normal",
            "trade_date": "2026-09-10", "sizing_capital": 1000000.0}


def _candidate(params, seed=None, ticker="2330", signal_date="2026-09-10", information_date="2026-09-10"):
    seed = deepcopy(seed or _seed(ticker))
    return {"ticker": ticker, "kind": "buy", "trade_date": information_date,
            "signal_date": signal_date, "proj_qty": int(seed["qty"]), "proj_cost": 100000.0,
            "execution_plan_seed": seed, "params_signature": build_portfolio_params_signature(params),
            "ensemble_member_key": "1", "ensemble_member_keys": ["1"],
            "ensemble_member_count": 1, "ensemble_vote_count": 1, "ensemble_min_agree": 1,
            "ensemble_member_params_by_key": {"1": params_to_json_dict(params)}}


def _frame(end="2026-09-18", periods=340):
    return pd.DataFrame({"Open":100.0,"High":101.0,"Low":99.0,"Close":100.0,"Volume":1000000.0},
                        index=pd.bdate_range(end=end, periods=periods))


@contextmanager
def _market_io(frames, state, reads=None):
    """Inject a pinned storage view, never a strategy result or persisted state."""
    reads = [] if reads is None else reads
    view = object()
    def load(_view=None, *, ticker, through_date=None, allowed_date=None, **kwargs):
        assert _view is view or kwargs.get("view") is view
        date = through_date or allowed_date
        reads.append((ticker, date))
        return frames[ticker].loc[:date].copy()
    def source(_root, *args, **kwargs):
        return deepcopy(state)
    with ExitStack() as stack:
        for module in ("services.trading.lifecycle_sync", "services.trading.lifecycle_context", "services.trading.market_data_consumer"):
            stack.enter_context(patch(module + ".load_trading_v2_consumer_state", side_effect=source))
        stack.enter_context(patch("services.trading.lifecycle_context.open_trading_v2_consumer_view", return_value=view))
        for module in ("services.trading.lifecycle_sync", "services.trading.pending_entry_service", "services.trading.entry_lifecycle_context"):
            stack.enter_context(patch(module + ".load_trading_v2_sanitized_ohlcv_frame", side_effect=load))
        stack.enter_context(patch("services.trading.position_rollforward.load_trading_position_market_frame", side_effect=load))
        yield reads


def validate_lifecycle_management_owner_contract_case(base_params):
    results, summary, check, check_true = bind_synthetic_case("LIFECYCLE_MANAGEMENT_OWNER", "lifecycle_ssot")
    p = _params(base_params)
    seed = _seed()
    shadow = build_counterfactual_shadow_position_from_plan(seed, t_open=100, t_high=107, t_low=99, params=p, ticker="2330", trade_date="2026-09-11")
    # A matured shadow has unique geometry that must survive actual acquisition.
    shadow.update(sl=102.0, sl_milli=102000, trailing_stop=102.0, trailing_stop_milli=102000)
    inherited = {**seed, "shadow_position_state": shadow}
    research = execute_pre_market_entry_plan(inherited, t_open=104.,t_high=105.,t_low=103.,t_close=104.,t_volume=100000.,y_close=104.,params=p,ticker="2330",trade_date="2026-09-14")
    # Limit must permit this fill; management geometry, not execution choice, is inherited.
    inherited["limit_price"] = 105.0
    research = execute_pre_market_entry_plan(inherited, t_open=104.,t_high=105.,t_low=103.,t_close=104.,t_volume=100000.,y_close=104.,params=p,ticker="2330",trade_date="2026-09-14")["position"]
    live = build_position_from_frozen_entry_plan(inherited,buy_price=104.,qty=1000,params=p,ticker="2330",trade_date="2026-09-14",t_high=105.,t_low=103.)
    check("shadow_actual_and_research_inherit_all_management_fields", {k:research.get(k) for k in POSITION_MANAGEMENT_FIELDS}, {k:live.get(k) for k in POSITION_MANAGEMENT_FIELDS})
    check("shadow_stop_is_not_reset_at_fill",102000,live["sl_milli"])
    check("shadow_high_water_is_inherited",107000,live["highest_high_since_entry_milli"])
    check("actual_fill_price_remains_actual",104000,live["entry_fill_price_milli"])

    for action, low, high in (("STOP",94.,111.),("TP_HALF",99.,111.)):
        pos=build_position_from_frozen_entry_plan(seed,buy_price=100.,qty=1000,params=p,ticker="2330",trade_date="2026-09-11",t_high=high,t_low=low)
        check("entry_day_"+action+"_is_deferred",action,pos["pending_exit_action"])
        before=deepcopy(pos);seen=[]
        step_position_management(pos,y_atr=5.,y_ind_sell=False,y_close=100.,y_high=high,t_open=108.,t_high=109.,t_low=107.,t_close=108.,t_volume=100000.,params=p,on_decision=lambda d:seen.append(d) or False,current_date="2026-09-14")
        check("unconfirmed_"+action+"_does_not_reduce_inventory",1000,pos["qty"])
        check("unconfirmed_"+action+"_does_not_change_cost",before["remaining_cost_basis_milli"],pos["remaining_cost_basis_milli"])
        check("unconfirmed_"+action+"_remains_due",action,pos["pending_exit_action"])
        check_true("deferred_"+action+"_uses_next_open_not_old_limit",bool(seen) and seen[0].deferred and seen[0].reference_price==108.)

    initial=build_position_from_frozen_entry_plan(seed,buy_price=100.,qty=1000,params=p,ticker="2330",trade_date="2026-09-11")
    frame=_frame(end="2026-09-15")
    frame.loc["2026-09-11",["High","Low"]]=[111.,99.]
    frame.loc["2026-09-14":,["Open","High","Low","Close"]]=[108.,109.,107.,108.]
    projection=replay_confirmed_position_management(initial,frame=frame,params=p,confirmed_quantity_events=[
        {"trade_date":"2026-09-14","qty_delta":-200,"event":"TP_HALF","tp_half_complete":False},
        {"trade_date":"2026-09-15","qty_delta":-300,"event":"TP_HALF","tp_half_complete":True}])
    tp_decisions=[d for d in projection["decisions"] if d["event"]=="TP_HALF"]
    check("partial_deferred_tp_does_not_resize_from_remaining_position",[500,300],[d["qty"] for d in tp_decisions])
    check("confirmed_tp_completion_is_acknowledged",True,projection["position_state"]["sold_half"])
    check("confirmed_tp_clears_deferred_obligation",None,projection["position_state"]["pending_exit_action"])
    check("only_confirmed_fills_reduce_qty",500,projection["position_state"]["qty"])

    selected=accept_entry_quantity_decision(seed,available_cash=1000000.,params=p,requested_qty=600)
    check("user_can_reduce_below_strategy_max",600,selected["qty"])
    check("user_choice_does_not_rewrite_original_strategy_max",1000,seed["qty"])
    check_true("selected_quantity_retains_separate_ceiling",selected["strategy_executable_qty_ceiling"]>=selected["qty"])
    rejected=False
    try: accept_entry_quantity_decision(seed,available_cash=1000000.,params=p,requested_qty=1001)
    except ValueError: rejected=True
    check("strategy_upper_bound_is_enforced",True,rejected)
    summary["checks"]=len(results)
    return results,summary


def validate_trading_lifecycle_persistent_simulation_case(base_params):
    results,summary,check,check_true=bind_synthetic_case("TRADING_LIFECYCLE_PERSISTENT_SIMULATION","lifecycle_integration")
    p=_params(base_params)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);account=initialize_trading_account_state(root,cash=1000000.)
        candidate=_candidate(p);lineage=build_trading_candidate_strategy_lineage(candidate)
        account=record_strategy_trading_buy(root,ticker="2330",qty=1000,price=100.,trade_date="2026-09-11",expected_revision=account["revision"],params=p,execution_plan_seed=candidate["execution_plan_seed"],strategy_lineage=lineage)
        pending_seed=accept_entry_quantity_decision(_seed("2317"),available_cash=800000.,params=p,requested_qty=600)
        pending_candidate=_candidate(p,pending_seed,"2317")
        pending_lineage=build_trading_candidate_strategy_lineage(pending_candidate)
        pending=create_trading_pending_entry(root,entry=_pending_entry_payload(origin="scanner_strategy",ticker="2317",information_date="2026-09-10",plan=pending_seed,lineage=pending_lineage,signal_date="2026-09-10",planned_trade_date="2026-09-11"))
        frames={"2330":_frame(),"2317":_frame()}
        state={"market_date":"2026-09-17","state_fingerprint":"batch-1"}
        cash=account["cash_milli"];broker=deepcopy(account["positions"]["2330"]["broker"])
        with _market_io(frames,state) as reads:
            result=run_trading_lifecycle_sync(root)
            check("all_lifecycles_sync_without_optional_oms", SYNC_STATUS_LATEST, result["status"])
            after=load_trading_account_state(root,required=True)
            pending_after=load_trading_pending_entry_state(root,required=True)
            validate_trading_account_state(after);validate_trading_pending_entry_state(pending_after)
            check("management_sync_never_changes_cash",cash,after["cash_milli"])
            check("management_sync_never_changes_broker_truth",broker,after["positions"]["2330"]["broker"])
            check("pending_preserves_user_reduced_qty",600,pending_after["entries"][pending["pending_entry_id"]]["planned_qty"])
            check("pending_and_position_share_finalized_boundary",pending_after["entries"][pending["pending_entry_id"]]["evaluated_through_date"],after["positions"]["2330"]["strategy_management"]["evaluated_through_date"])
            check_true("all_market_reads_use_pinned_not_newer_date",bool(reads) and all(date<="2026-09-17" for _,date in reads))
            stable=canonical_json_sha256({"account":after,"pending":pending_after})
            reads.clear();again=run_trading_lifecycle_sync(root)
            check("same_context_rerun_has_no_event_or_revision_change",stable,canonical_json_sha256({"account":load_trading_account_state(root),"pending":load_trading_pending_entry_state(root)}))
            check("same_context_rerun_needs_no_market_replay",[],list(reads))
            # Same day, new verified data: invalidate by full source identity, not date.
            state["state_fingerprint"]="batch-2"
            frames["2330"].loc["2026-09-17",["Open","High","Low","Close"]]=[100.,103.,94.,100.]
            corrected=run_trading_lifecycle_sync(root)
            after2=load_trading_account_state(root)
            check("same_date_corrected_market_batch_is_replayed", "STOP EXIT",after2["positions"]["2330"]["strategy_management"].get("sell_signal"))
            check("new_stop_obligation_does_not_infer_a_sell",1000,after2["positions"]["2330"]["broker"]["qty"])
            check("batch_correction_preserves_original_frozen_lineage",lineage,after2["positions"]["2330"]["strategy_lineage"])
            plan=build_trading_protection_plan(root)
            check("triggered_strategy_exit_is_market_not_wait_for_retouch","MARKET",plan["positions"][0]["legs"][0]["order_type"])
            # Chart consumes the same fact projection and frozen Params.
            record=after2["positions"]["2330"]
            inspection={"ticker":"2330","current_position":record,"position_binding":resolve_trading_position_management_binding(record),"account_events":effective_trading_account_events(after2),"account_audit_events":after2["events"],"entry_orders":[],"pending_entries":[],"finalized_date":"2026-09-17"}
            frame=frames["2330"].loc[:"2026-09-17"]
            chart={"date_labels":[d.strftime("%Y-%m-%d") for d in frame.index],**{c.lower():frame[c].to_numpy() for c in frame.columns}}
            timeline=build_trading_single_stock_lifecycle_timeline(inspection,chart,params=p)
            latest=timeline[max(timeline)]
            check("chart_and_position_share_management_stop",record["strategy_management"]["position_state"]["sl"],latest["stop_price"])
            check("chart_exposes_same_full_exit_obligation","STOP EXIT",latest["sell_signal"])
            # Restored corrected data must remove only a derived signal, never a fill fact.
            state["state_fingerprint"]="batch-3";frames["2330"]=_frame()
            run_trading_lifecycle_sync(root)
            restored=load_trading_account_state(root)
            check("correction_rebuild_does_not_leave_phantom_stop",None,restored["positions"]["2330"]["strategy_management"].get("sell_signal"))
            check("correction_rebuild_keeps_all_confirmed_economics",broker,restored["positions"]["2330"]["broker"])
    _check_accepted_pending_reservation_capability(base_params, check, check_true)
    summary["checks"]=len(results)
    return results,summary



def _check_accepted_pending_reservation_capability(base_params, check, check_true):
    """AI: Accepted intent is not a daily reallocation; cash checks remain live."""
    from core.entry_plans import (build_normal_candidate_plan, build_cash_capped_entry_plan,
                                  validate_accepted_entry_reservation)
    from core.exact_accounting import milli_to_money
    from services.trading import pending_entry_service as pending
    from services.trading.account_state import set_trading_cash_balance
    from services.trading.strategy_param_runtime import build_trading_manual_management_lineage
    from services.trading.single_stock_inspection import (build_trading_single_stock_inspection,
                                                         project_trading_single_stock_chart_payload)
    p = replace(_params(base_params), use_bb=False, use_kc=False, use_vol=False,
                use_breakout_false_filter=False, use_breakout_return_filter=False,
                min_entry_notional=0, fixed_risk=.01, max_position_cap_pct=.5, atr_times_trail=5.)
    frame = _frame()
    frame.loc[:, ["Open", "High", "Low", "Close"]] = [19.65, 19.9, 19.45, 19.65]
    market = {"market_date": "2026-09-18", "state_fingerprint": "accepted-reservation"}
    chart = {"date_labels": frame.index.strftime("%Y-%m-%d").tolist(),
             **{k.lower(): frame[k].to_numpy() for k in frame}, "marker_groups": {},
             "signal_annotations": [], "strategy_prefill_lifecycle_by_index": {},
             "summary_box": ["Independent Research result"]}
    def accepted(ticker, qty):
        seed = accept_entry_quantity_decision(
            build_normal_candidate_plan(19.65, .28, 1000000., p, ticker=ticker, trade_date="2026-09-17"),
            available_cash=1000000., params=p, requested_qty=qty)
        seed["entry_type"] = "manual"
        lineage = build_trading_manual_management_lineage(params=p, execution_plan_seed=seed,
            information_date="2026-09-17", origin="manual_pending_entry", planned_qty=qty,
            planned_cost=seed["reserved_cost"])
        return _pending_entry_payload(origin="manual_selected", ticker=ticker, information_date="2026-09-17",
            plan=seed, lineage=lineage, signal_date=None, planned_trade_date="2026-09-11")
    with tempfile.TemporaryDirectory() as td, _market_io({"2002": frame, "2836": frame}, market) as reads, _raw_market_io({"2002": frame, "2836": frame}), patch.object(pending, "load_trading_scanner_runtime", return_value={"latest_data_date": "2026-09-18", "params": p}):
        root = Path(td)
        account = initialize_trading_account_state(root, cash=1000000.)
        first = create_trading_pending_entry(root, entry=accepted("2002", 16000))
        sibling = create_trading_pending_entry(root, entry=accepted("2836", 8000))
        eid = first["pending_entry_id"]
        def project():
            inspection = build_trading_single_stock_inspection(root, "2002", market_frame=frame, consumer_state=market)
            return project_trading_single_stock_chart_payload(chart, inspection, params=p)
        before = project()
        result = run_trading_lifecycle_sync(root)
        check("accepted_sync_survives_new_strategy_ceiling_drop", SYNC_STATUS_LATEST, result["status"])
        state = load_trading_pending_entry_state(root, required=True)
        current = state["entries"][eid]
        recalculated = build_cash_capped_entry_plan(current["execution_plan_seed"], 1000000., p)
        check("fixture_actually_crosses_strategy_acceptance_boundary", [16000, 10000],
              [first["planned_qty"], recalculated["qty"]])
        fields = ("planned_qty", "limit_price", "reserved_cost", "reserved_cost_milli",
                  "planned_trade_date", "information_date", "management_lineage", "origin")
        for old in (first, sibling):
            row = state["entries"][old["pending_entry_id"]]
            check("management_refresh_preserves_accepted_facts_" + old["ticker"],
                  {k: old.get(k) for k in fields}, {k: row.get(k) for k in fields})
            for field in ("qty", "limit_price", "reserved_cost_milli", "reserved_cost",
                          "strategy_executable_qty_ceiling", "user_qty_override", "sizing_capital"):
                check("management_refresh_preserves_approval_" + old["ticker"] + "_" + field,
                      old["execution_plan_seed"].get(field), row["execution_plan_seed"].get(field))
        check("reservation_sync_does_not_touch_account", account, load_trading_account_state(root))
        after = project()
        for key in ("shadow_stop_line", "shadow_tp_line", "shadow_limit_line", "shadow_entry_line"):
            check("sizing_fix_preserves_entire_shadow_line_" + key,
                  json_native_value(before[key]), json_native_value(after[key]))
        idx = int(frame.index.get_loc("2026-09-15"))
        check("sizing_fix_retains_historical_shadow_state", "SHADOW", after["trading_lifecycle_by_index"][idx]["state"])
        check("sizing_fix_does_not_replace_research", before["summary_box"], after["summary_box"])
        precise_inspection = build_trading_single_stock_inspection(root, "2002", market_frame=frame, consumer_state=market)
        rounded_chart = {**chart, **{key: chart[key].astype(np.float32) for key in ("open", "high", "low", "close", "volume")}}
        rounded_projection = project_trading_single_stock_chart_payload(rounded_chart, precise_inspection, params=p)
        for key in ("shadow_stop_line", "shadow_tp_line", "shadow_limit_line", "shadow_entry_line"):
            check("display_precision_cannot_change_prefill_management_" + key,
                  json_native_value(after[key]), json_native_value(rounded_projection[key]))
        check("sidebar_shadow_stop_equals_pending_sync_stop", current["init_sl"],
              rounded_projection["trading_lifecycle_by_index"][idx]["stop_price"])
        start = idx - 1
        cropped_chart = {**rounded_chart, "date_labels": chart["date_labels"][start:],
                         **{key: rounded_chart[key][start:] for key in ("open", "high", "low", "close", "volume")}}
        cropped = project_trading_single_stock_chart_payload(cropped_chart, precise_inspection, params=p)
        check("cropped_display_uses_full_origin_and_date_mapping", after["trading_lifecycle_by_index"][idx],
              cropped["trading_lifecycle_by_index"][idx - start])
        future_frame = frame.copy()
        future_frame.loc[pd.Timestamp("2026-09-21")] = future_frame.iloc[-1]
        try:
            build_trading_single_stock_inspection(root, "2002", market_frame=future_frame, consumer_state=market)
        except ValueError as exc:
            rejected = "finalized boundary" in str(exc)
        else:
            rejected = False
        check_true("unfilled_inspection_rejects_future_market_input", rejected)
        reads.clear()
        rerun = run_trading_lifecycle_sync(root)
        check_true("accepted_refresh_rerun_is_noop", all(not row["changed"] for row in rerun["pending_results"]))
        check("accepted_refresh_rerun_does_not_write", state, load_trading_pending_entry_state(root))
        check("accepted_refresh_rerun_does_not_replay_market", [], reads)
        total = sum(row["reserved_cost_milli"] for row in state["entries"].values())
        def set_cash(amount):
            existing = load_trading_account_state(root)
            return set_trading_cash_balance(root, cash=milli_to_money(amount), expected_revision=existing["revision"])
        # A cache hit cannot conceal a cash correction, including one milli below
        # the exact aggregate. The other order's lock is never available twice.
        set_cash(total)
        check("exact_aggregate_reservation_is_covered", SYNC_STATUS_LATEST, run_trading_lifecycle_sync(root)["status"])
        set_cash(total - 1)
        shortage_account = load_trading_account_state(root)
        shortage = run_trading_lifecycle_sync(root)
        check("cash_shortage_detected_despite_management_cache", 2, len(shortage["pending_errors"]))
        check_true("cash_shortage_is_not_misreported_as_strategy_resizing",
                   all("after other pending reservations" in error for error in shortage["pending_errors"].values()))
        check("shortage_does_not_release_or_resize_orders", state, load_trading_pending_entry_state(root))
        check("shortage_does_not_change_confirmed_cash", shortage_account, load_trading_account_state(root))
        set_cash(total)
        check("cash_restore_recovers_without_market_identity_change", SYNC_STATUS_LATEST, run_trading_lifecycle_sync(root)["status"])
        check("cash_only_checks_do_not_replay_market", [], reads)
        # Pure validation keeps recorded approval intact, tolerates missing legacy
        # approval metadata, and refuses corrupt accepted economics.
        plan = current["execution_plan_seed"]
        legacy = deepcopy(plan)
        legacy.pop("strategy_executable_qty_ceiling", None)
        before_legacy = deepcopy(legacy)
        validate_accepted_entry_reservation(legacy, available_cash_milli=total, params=p)
        check("legacy_reservation_validation_is_read_only", before_legacy, legacy)
        for name, changes in (("quantity", {"qty": 16001}), ("cash", {"reserved_cost_milli": 1}),
                              ("display", {"reserved_cost": 1.}), ("limit", {"limit_price": 20.}),
                              ("fractional", {"qty": 15999.5}), ("boolean", {"qty": True})):
            bad = {**plan, **changes}
            try:
                validate_accepted_entry_reservation(bad, available_cash_milli=total, params=p)
            except ValueError:
                rejected = True
            else:
                rejected = False
            check_true("corrupt_accepted_reservation_rejected_" + name, rejected)
        set_cash(1000000000)
        for requested in (16000, 10001):
            try:
                pending.update_trading_pending_entry_intent(root, pending_entry_id=eid,
                    origin="manual", ticker="2002", qty=requested, limit_price=19.65,
                    planned_trade_date="2026-09-11")
            except ValueError as exc:
                rejected = "exceeds strategy/cash maximum 10000" in str(exc)
            else:
                rejected = False
            check_true("explicit_amendment_still_enforces_current_strategy_ceiling_" + str(requested), rejected)
        check("rejected_amendment_has_no_persistent_write", state, load_trading_pending_entry_state(root))
        for requested in (10000, 8000):
            amended = pending.update_trading_pending_entry_intent(root, pending_entry_id=eid,
                origin="manual", ticker="2002", qty=requested, limit_price=19.65,
                planned_trade_date="2026-09-11")
            check("legal_amendment_keeps_user_choice_" + str(requested), requested, amended["planned_qty"])
            check("legal_amendment_resynchronizes_" + str(requested), SYNC_STATUS_LATEST, run_trading_lifecycle_sync(root)["status"])
            check("sync_does_not_increase_reduced_choice_" + str(requested), requested,
                  load_trading_pending_entry_state(root)["entries"][eid]["planned_qty"])
        historical = project()["trading_lifecycle_by_index"][idx]
        check("later_reduction_does_not_rewrite_historical_order", 16000, historical["planned_qty"])

def _check_dated_prefill_origin_capability(base_params, check, check_true):
    """Causal materialization, checkpoints, factual intent and acquisition."""
    from core.entry_plans import build_normal_candidate_plan
    from core.exact_accounting import build_buy_ledger, price_to_milli, milli_to_money
    from core.prefill_origin import materialize_prefill_origin
    from core.trade_lifecycle import build_prefill_lifecycle_from_frame
    from core.trading_lifecycle_plans import build_pending_prefill_plan
    from services.trading import lifecycle_sync as sync, pending_entry_service as pending
    from services.trading.lifecycle_context import TradingLifecycleContext
    from services.trading.strategy_param_runtime import build_trading_manual_management_lineage
    from services.trading.single_stock_inspection import build_trading_single_stock_inspection, project_trading_single_stock_chart_payload
    from services.trading.pending_entry_state import update_trading_pending_entry
    from services.trading.account_state import record_manual_trading_sell
    p = replace(_params(base_params), use_bb=False, use_kc=False, use_vol=False,
                use_breakout_false_filter=False, use_breakout_return_filter=False,
                min_entry_notional=0, initial_capital=2000000., fixed_risk=.02)
    frame = _frame()
    frame.loc[:, ["Open", "High", "Low", "Close"]] = [19.5, 19.7, 19.3, 19.5]
    seed = build_normal_candidate_plan(19.65, .35, 2000000., p, ticker="2002", trade_date="2026-09-17")
    seed.update(qty=16000, max_qty=16000, entry_type="manual", init_sl=18.95, init_trail=18.95, target_price=20.35)
    cost = build_buy_ledger(price_to_milli(19.65), 16000, p)["cash_buy_total_milli"]
    seed.update(reserved_cost_milli=cost, reserved_cost=milli_to_money(cost))
    lineage = build_trading_manual_management_lineage(params=p, execution_plan_seed=seed,
        information_date="2026-09-17", origin="manual_pending_entry", planned_qty=16000, planned_cost=seed["reserved_cost"])
    entry = _pending_entry_payload(origin="manual_selected", ticker="2002", information_date="2026-09-17",
        plan=seed, lineage=lineage, signal_date=None, planned_trade_date="2026-09-11")
    plan = build_pending_prefill_plan(entry)
    resolved = materialize_prefill_origin(frame=frame, plan=plan, params=p)
    check("late_snapshot_retains_its_timestamp_not_as_historical_geometry", "2026-09-17", resolved["recorded_plan_as_of_date"])
    check("historical_geometry_uses_the_actual_order_origin", "2026-09-11", resolved["signal_date"])
    check("historical_atr_from_origin_bar_not_later_seed", .4, round(resolved["entry_atr"], 8))
    expected_seed = build_normal_candidate_plan(19.65, .4, 2000000., p, ticker="2002", trade_date="2026-09-11")
    for field, key in (("init_sl","stop_price"),("init_trail","init_trail"),("target_price","tp_price")):
        check("origin_uses_canonical_builder_"+field, expected_seed[field], resolved[key])
    timeline = build_prefill_lifecycle_from_frame(frame=frame, plan=plan, params=p)
    for day in ("2026-09-14","2026-09-15","2026-09-16","2026-09-17","2026-09-18"):
        row = timeline[int(frame.index.get_loc(day))]
        check("unfilled_history_is_shadow_"+day, "SHADOW", row["state"])
        check_true("unfilled_history_has_all_management_"+day, all(row.get(k) is not None for k in ("stop_price","tp_price","entry_price","limit_price")))
    first = timeline[int(frame.index.get_loc("2026-09-14"))]
    check("shadow_anchor_uses_first_post_order_open_not_late_snapshot",19.5,first["entry_price"])
    check("shadow_initial_stop_numeric_oracle",18.7,first["shadow_position_state"]["initial_stop"])
    check("shadow_effective_stop_numeric_oracle",19.1,first["stop_price"])
    # Stop/Target/ATR from the late snapshot may be arbitrarily different. They
    # must have zero influence on the reconstructed path, even after that date.
    poison_entry = deepcopy(entry)
    for owner in (poison_entry["execution_plan_seed"], poison_entry["management_lineage"]["execution_plan_seed"]):
        owner.update(init_sl=999.,init_trail=999.,target_price=1001.,entry_atr=999.)
    rebuilt = build_prefill_lifecycle_from_frame(frame=frame,plan=build_pending_prefill_plan(poison_entry),params=p)
    check("late_snapshot_poisoning_does_not_change_shadow_history",json_native_value(timeline),json_native_value(rebuilt))
    suffix = frame.copy(); suffix.loc["2026-09-17":,["Open","High","Low","Close"]]=[50.,99.,1.,50.]
    altered = build_prefill_lifecycle_from_frame(frame=suffix,plan=plan,params=p)
    cutoff = int(frame.index.get_loc("2026-09-16"))
    check("future_ohlcv_cannot_change_any_prior_shadow_row",json_native_value({i:r for i,r in timeline.items() if i<=cutoff}),json_native_value({i:r for i,r in altered.items() if i<=cutoff}))
    prefix = build_prefill_lifecycle_from_frame(frame=frame.loc[:"2026-09-16"],plan=plan,params=p)
    check("prefix_and_full_replay_have_identical_history",json_native_value(prefix),json_native_value({i:r for i,r in timeline.items() if i<=cutoff}))
    try: materialize_prefill_origin(frame=frame,plan=plan,params=replace(p,atr_len=2))
    except ValueError: rejected=True
    else: rejected=False
    check_true("cannot_substitute_current_params_for_frozen_policy",rejected)
    broken=deepcopy(entry);broken["management_lineage"]["frozen_params"]["atr_len"]=2
    try: build_pending_prefill_plan(broken)
    except ValueError: rejected=True
    else: rejected=False
    check_true("corrupt_frozen_binding_is_rejected",rejected)
    invalid={**plan,"signal_date":"2026-09-11"}
    try: build_prefill_lifecycle_from_frame(frame=frame,plan=invalid,params=p)
    except ValueError as exc: rejected="PIT" in str(exc)
    else: rejected=False
    check_true("cannot_backdate_snapshot_without_materializing_origin",rejected)
    # Existing legitimately early geometry remains EXACT, not normalized into a
    # new plan. Missing frozen evidence cannot be invented from caller Params.
    early=deepcopy(entry);early["management_lineage"]["execution_plan_seed"].update(trade_date="2026-09-11",init_sl=15.,init_trail=15.,target_price=30.)
    ep=build_pending_prefill_plan(early)
    check("verifiable_early_seed_is_not_rebuilt",ep,materialize_prefill_origin(frame=frame,plan=ep,params=p))
    missing=deepcopy(entry);missing["management_lineage"].pop("frozen_params")
    no_policy=build_prefill_lifecycle_from_frame(frame=frame.loc[:"2026-09-16"],plan=build_pending_prefill_plan(missing),params=None)
    check_true("missing_policy_preserves_order_without_inventing_shadow",bool(no_policy) and all(r["state"]=="PENDING" and r["stop_price"] is None for r in no_policy.values()))
    no_bar=materialize_prefill_origin(frame=frame.loc["2026-09-14":],plan=plan,params=p)
    check_true("missing_original_bar_is_explicit_not_current_fallback",bool(no_bar.get("origin_resolution_error")) and no_bar["signal_date"]=="2026-09-17")
    no_warmup=materialize_prefill_origin(frame=frame.loc["2026-09-11":],plan=plan,params=p)
    check_true("missing_atr_warmup_is_explicit",bool(no_warmup.get("origin_resolution_error")))
    # Date boundaries and real exits are preserved. The former test wrongly
    # treated a deliberately huge 9/14 downside excursion as a healthy order.
    for boundary in ({"end_date":"2026-09-11"},{"end_before_date":"2026-09-15"}):
        bounded=build_prefill_lifecycle_from_frame(frame=frame,plan={**plan,**boundary},params=p)
        check_true("cancel_or_fill_bounds_reconstructed_history_"+next(iter(boundary)),all(str(frame.index[i].date())<=boundary.get("end_date","9999-12-31") and str(frame.index[i].date())<boundary.get("end_before_date","9999-12-31") for i in bounded))
    for day in ("2026-09-14","2026-09-15","2026-09-18"):
        acquired=resolve_confirmed_entry_plan_from_frame(entry=entry,frame=frame,params=p,fill_date=day)
        check_true("historical_fill_uses_only_prior_management_"+day,acquired["management_information_date"]<day)
        poisoned=frame.copy();poisoned.loc[day:,["Open","High","Low","Close"]]=[50.,99.,1.,50.]
        same=resolve_confirmed_entry_plan_from_frame(entry=entry,frame=poisoned,params=p,fill_date=day)
        check("fill_and_future_bars_do_not_change_acquisition_"+day,json_native_value(acquired),json_native_value(same))
    chart={"date_labels":frame.index.strftime("%Y-%m-%d").tolist(),**{k.lower():frame[k].to_numpy() for k in frame},"marker_groups":{},"signal_annotations":[],"strategy_prefill_lifecycle_by_index":{},"summary_box":["Research untouched"]}
    # Read models, resource checks, immutable events and actual fill handoff run
    # through their public services against real temporary JSON state files.
    with tempfile.TemporaryDirectory() as td, _market_io({"2002":frame},{"market_date":"2026-09-18","state_fingerprint":"reconstruction"}), _raw_market_io({"2002":frame}), ExitStack() as stack:
        root=Path(td);account=initialize_trading_account_state(root,cash=2000000.)
        saved=create_trading_pending_entry(root,entry=entry)
        stack.enter_context(patch.object(pending,"load_trading_scanner_runtime",return_value={"latest_data_date":"2026-09-18","params":p}))
        result=run_trading_lifecycle_sync(root)
        check("reconstructed_pending_can_sync_normally",SYNC_STATUS_LATEST,result["status"])
        state=load_trading_pending_entry_state(root,required=True)
        current=state["entries"][saved["pending_entry_id"]]
        check("sync_preserves_order_date_chosen_qty_and_reservation",["2026-09-11",16000,cost],[current["planned_trade_date"],current["planned_qty"],current["reserved_cost_milli"]])
        check("sync_does_not_rewrite_frozen_lineage",lineage,current["management_lineage"])
        check("sync_does_not_change_account_facts",account,load_trading_account_state(root,required=True))
        again=run_trading_lifecycle_sync(root)
        check_true("identical_sync_is_idempotent",all(not x["changed"] for x in again["pending_results"]))
        check("idempotent_sync_has_no_state_write",state,load_trading_pending_entry_state(root,required=True))
        def projection():
            ins=build_trading_single_stock_inspection(root,"2002",market_frame=frame,consumer_state={"market_date":"2026-09-18"})
            return project_trading_single_stock_chart_payload(chart,ins,params=replace(p,atr_len=2))
        projected=projection();idx=int(frame.index.get_loc("2026-09-15"))
        row=projected["trading_lifecycle_by_index"][idx]
        check("chart_uses_same_shadow_as_sync",json_native_value({k:v for k,v in timeline[idx].items() if k not in {"source","pending_entry_id"}}),json_native_value({k:row.get(k) for k in timeline[idx] if k not in {"source","pending_entry_id"}}))
        check_true("all_four_shadow_lines_exist",all(pd.notna(projected[k][idx]) for k in ("shadow_stop_line","shadow_tp_line","shadow_limit_line","shadow_entry_line")))
        check("research_statistics_are_not_replaced",chart["summary_box"],projected["summary_box"])
        # Real strategy invalidation still blocks sync; it never auto-cancels or
        # silently reduces the accepted quantity to make the UI green.
        for reason,high,low in (("STOP",19.7,17.9),("TP_HALF",25.,19.3)):
            bad=frame.copy();bad.loc["2026-09-18",["High","Low"]]=[high,low]
            ctx=TradingLifecycleContext("2026-09-18","changed-"+reason,{},object())
            with patch.object(sync,"load_trading_v2_sanitized_ohlcv_frame",return_value=bad):
                try: sync._sync_one_pending(root,entry=current,active_entries=[current],account=account,latest_finalized_date="2026-09-18",lifecycle_context=ctx)
                except RuntimeError as exc: rejected="shadow exit" in str(exc)
                else: rejected=False
            check_true("real_shadow_exit_is_not_suppressed_"+reason,rejected)
            check("failed_sync_does_not_mutate_intent_"+reason,state,load_trading_pending_entry_state(root,required=True))
        transfer=pending.fill_trading_pending_entry(root,pending_entry_id=saved["pending_entry_id"],qty=16000,price=19.5,trade_date="2026-09-16",expected_account_revision=account["revision"])
        account=load_trading_account_state(root,required=True)
        filled=projection()
        check("filled_history_retains_prefill_shadow","SHADOW",filled["trading_lifecycle_by_index"][idx]["state"])
        buy_idx=int(frame.index.get_loc("2026-09-16"))
        check("only_confirmed_fill_creates_position",["POSITION",16000,19.5],[filled["trading_lifecycle_by_index"][buy_idx][k] for k in ("state","position_qty","entry_price")])
        expected=resolve_confirmed_entry_plan_from_frame(entry=entry,frame=frame,params=p,fill_date="2026-09-16")["shadow_position_state"]
        position=account["positions"]["2002"]["strategy_management"]["entry_position_state"]
        for field in ("sl_milli","tp_half_milli","highest_high_since_entry_milli"):
            check("confirmed_fill_inherits_shadow_"+field,expected[field],position[field])
        record_manual_trading_sell(root,ticker="2002",qty=16000,price=19.5,trade_date="2026-09-17",expected_revision=account["revision"])
        closed=projection()
        check("manual_close_preserves_prior_shadow","SHADOW",closed["trading_lifecycle_by_index"][idx]["state"])
        check("manual_close_has_zero_real_inventory",0,closed["trading_lifecycle_by_index"][int(frame.index.get_loc("2026-09-17"))]["position_qty"])
        check_true("closed_origin_never_reappears_after_exit",int(frame.index.get_loc("2026-09-18")) not in closed["trading_lifecycle_by_index"] and not closed.get("future_preview"))
    # A real Scanner signal, a later Pending and a still-later saved checkpoint
    # must be one lineage. Different current Params cannot change its prefix.
    sf=_frame();sf.loc["2026-09-03":,["Open","High","Low","Close"]]=[110.,111.,109.,110.];sf.loc["2026-09-03","Open"]=109.
    from core.signal_utils import generate_signals,unpack_precomputed_signals
    a,b,_s,limits=unpack_precomputed_signals(generate_signals(sf.loc[:"2026-09-03"],p,ticker="2330"))
    check_true("strategy_fixture_has_real_original_signal",bool(b[-1]))
    original=build_normal_candidate_plan(float(limits[-1]),float(a[-1]),1000000.,p,ticker="2330",trade_date="2026-09-03")
    original.update(qty=1000,max_qty=1000,entry_type="normal")
    late={**original,"trade_date":"2026-09-17","entry_atr":999.,"init_sl":999.,"init_trail":999.,"target_price":1999.}
    candidate=_candidate(p,late,signal_date="2026-09-03",information_date="2026-09-17")
    strategy=_pending_entry_payload(origin="scanner_strategy",ticker="2330",information_date="2026-09-17",plan=late,lineage=build_trading_candidate_strategy_lineage(candidate),signal_date="2026-09-03",planned_trade_date="2026-09-11")
    sp=build_pending_prefill_plan(strategy)
    sr=build_prefill_lifecycle_from_frame(frame=sf,plan=sp,params=p)
    direct={**sp,"signal_date":"2026-09-03","plan_as_of_date":"2026-09-03","limit_price":original["limit_price"],"stop_price":original["init_sl"],"init_trail":original["init_trail"],"tp_price":original["target_price"],"entry_atr":original["entry_atr"]}
    direct.pop("origin_reconstruction",None)
    oracle=build_prefill_lifecycle_from_frame(frame=sf,plan=direct,params=p)
    for i,row in oracle.items():
        check("strategy_reference_geometry_"+str(sf.index[i].date()),[row.get(k) for k in ("state","stop_price","tp_price","entry_price")],[sr[i].get(k) for k in ("state","stop_price","tp_price","entry_price")])
    check("order_does_not_restart_existing_shadow","SHADOW",sr[int(sf.index.get_loc("2026-09-11"))]["state"])
    mature=deepcopy(oracle[int(sf.index.get_loc("2026-09-17"))]["shadow_position_state"])
    strategy["management_lineage"]["execution_plan_seed"].update(shadow_position_state=mature,management_information_date="2026-09-17")
    resumed=build_prefill_lifecycle_from_frame(frame=sf,plan=build_pending_prefill_plan(strategy),params=p)
    birth=int(sf.index.get_loc("2026-09-17"))
    check_true("checkpoint_does_not_erase_original_prefix",int(sf.index.get_loc("2026-09-04")) in resumed)
    check("dated_checkpoint_is_preserved_exactly",json_native_value(mature),json_native_value(resumed[birth]["shadow_position_state"]))
    check("checkpoint_does_not_double_consume_its_bar",json_native_value(oracle[birth+1]["shadow_position_state"]),json_native_value(resumed[birth+1]["shadow_position_state"]))


    # Equal timestamps do not certify checkpoint compatibility either. A
    # derived checkpoint with another target/stop must not split the lineage.
    incompatible = deepcopy(strategy)
    checkpoint_state = incompatible["management_lineage"]["execution_plan_seed"]["shadow_position_state"]
    checkpoint_state.update(tp_half=1999., tp_half_milli=1999000,
                            sl=1., sl_milli=1000, trailing_stop=1., trailing_stop_milli=1000)
    repaired = build_prefill_lifecycle_from_frame(frame=sf, plan=build_pending_prefill_plan(incompatible), params=p)
    for date in ("2026-09-16", "2026-09-17", "2026-09-18"):
        idx = int(sf.index.get_loc(date))
        check("checkpoint_compatibility_uses_one_origin_" + date,
              [oracle[idx].get(k) for k in ("state", "stop_price", "tp_price", "entry_price")],
              [repaired[idx].get(k) for k in ("state", "stop_price", "tp_price", "entry_price")])
    check_true("incompatible_checkpoint_reconstruction_is_explicit",
               bool(repaired[birth].get("management_checkpoint_resolution")))
    check("incompatible_checkpoint_original_evidence_not_overwritten", 1999., checkpoint_state["tp_half"])

    # A live checkpoint cannot legitimize resurrecting a terminated prefix.
    broken=sf.copy();broken.loc["2026-09-14", "Low"]=1.
    try: build_prefill_lifecycle_from_frame(frame=broken,plan=build_pending_prefill_plan(strategy),params=p)
    except ValueError as exc: rejected="checkpoint conflicts" in str(exc)
    else: rejected=False
    check_true("checkpoint_cannot_revive_a_terminated_prefix",rejected)
    # All consumers read one accepted-decision history; latest quantities/prices
    # are not historical management inputs merely because an order is backdated.
    from services.trading.pending_entry_state import project_trading_pending_intent_entries
    with tempfile.TemporaryDirectory() as td, _market_io({"2002":frame},{"market_date":"2026-09-18","state_fingerprint":"intent-history"}), _raw_market_io({"2002":frame}), ExitStack() as stack:
        root=Path(td);account=initialize_trading_account_state(root,cash=2000000.)
        saved=create_trading_pending_entry(root,entry=entry)
        stack.enter_context(patch.object(pending,"load_trading_scanner_runtime",return_value={"latest_data_date":"2026-09-18","params":p}))
        check("original_late_intent_can_sync",SYNC_STATUS_LATEST,run_trading_lifecycle_sync(root)["status"])
        pending.update_trading_pending_entry_intent(root,pending_entry_id=saved["pending_entry_id"],origin="manual_selected",ticker="2002",qty=12000,planned_trade_date="2026-09-11",limit_price=19.7)
        check("edited_intent_can_sync",SYNC_STATUS_LATEST,run_trading_lifecycle_sync(root)["status"])
        state=load_trading_pending_entry_state(root,required=True)
        history=project_trading_pending_intent_entries(state)[saved["pending_entry_id"]]
        replay=build_prefill_lifecycle_from_frame(frame=frame,plan=build_pending_prefill_plan(history),params=p)
        earlier=replay[int(frame.index.get_loc("2026-09-15"))];later=replay[int(frame.index.get_loc("2026-09-18"))]
        check("past_order_keeps_original_limit_and_quantity",[19.65,16000],[earlier["limit_price"],earlier["planned_qty"]])
        check("new_order_choice_starts_on_its_decision_day",[19.7,12000],[later["limit_price"],later["planned_qty"]])
        check("price_edit_does_not_reset_shadow",earlier["shadow_position_state"]["entry_fill_price_milli"],later["shadow_position_state"]["entry_fill_price_milli"])
        ins=build_trading_single_stock_inspection(root,"2002",market_frame=frame,consumer_state={"market_date":"2026-09-18"})
        projected=project_trading_single_stock_chart_payload(chart,ins,params=p)
        for date in ("2026-09-15","2026-09-18"):
            i=int(frame.index.get_loc(date))
            check("chart_and_sync_share_accepted_history_"+date,[replay[i][k] for k in ("state","limit_price","planned_qty","stop_price")],[projected["trading_lifecycle_by_index"][i][k] for k in ("state","limit_price","planned_qty","stop_price")])
        check_true("derived_intent_history_is_not_persisted",all("order_intent_history" not in row for row in state["entries"].values()))
        check_true("decision_history_sync_is_idempotent",all(not x["changed"] for x in run_trading_lifecycle_sync(root)["pending_results"]))
        check("intent_edits_never_change_account_facts",account,load_trading_account_state(root,required=True))



def _check_confirmed_origin_upgrade_capability(base_params, check, check_true):
    """Legacy snapshots are evidence, not a second origin at the fill boundary."""
    from core.entry_plans import build_normal_candidate_plan
    from core.signal_utils import generate_signals, unpack_precomputed_signals
    from core.trade_lifecycle import build_prefill_lifecycle_from_frame
    from core.trading_lifecycle_plans import build_pending_prefill_plan
    from core.trading_position_projection import build_confirmed_position_origin
    from services.trading.strategy_param_runtime import build_trading_manual_management_lineage
    from services.trading.pending_entry_state import mark_trading_pending_entry_filled
    from services.trading.account_state import record_managed_manual_trading_buy, record_manual_trading_sell
    from services.trading.position_rollforward import run_trading_position_rollforward, build_trading_position_rollforward_snapshot
    from services.trading.single_stock_inspection import build_trading_single_stock_inspection, project_trading_single_stock_chart_payload

    p = replace(_params(base_params), use_bb=False, use_kc=False, use_vol=False,
        use_breakout_false_filter=False, use_breakout_return_filter=False,
        atr_len=2, atr_times_trail=5., min_entry_notional=0, fixed_risk=.01, max_position_cap_pct=.5)
    frame = _frame()
    frame.loc["2026-09-14":, ["High", "Low"]] = [100.1, 99.9]
    order_date, fill_date, registration = "2026-09-11", "2026-09-16", "2026-09-18"
    market = {"market_date": registration, "state_fingerprint": "confirmed-origin-upgrade-capability"}
    chart = {"date_labels": frame.index.strftime("%Y-%m-%d").tolist(),
        **{k.lower(): frame[k].to_numpy() for k in frame}, "marker_groups": {},
        "signal_annotations": [], "strategy_prefill_lifecycle_by_index": {}, "summary_box": ["Research evidence unchanged"]}
    rows_by_kind = {}
    for kind in ("late", "timely", "new"):
        with tempfile.TemporaryDirectory() as td, _market_io({"2330": frame}, market):
            root = Path(td)
            account = initialize_trading_account_state(root, cash=1000000.)
            seed_date = order_date if kind == "timely" else registration
            atr = unpack_precomputed_signals(generate_signals(frame.loc[:seed_date], p, ticker="2330"))[0]
            seed = accept_entry_quantity_decision(build_normal_candidate_plan(102., float(atr[-1]), 1000000., p,
                ticker="2330", trade_date=seed_date), available_cash=1000000., params=p, requested_qty=1000)
            seed["entry_type"] = "manual"
            lineage = build_trading_manual_management_lineage(params=p, execution_plan_seed=seed,
                information_date=seed_date, origin="manual_pending_entry", planned_qty=1000, planned_cost=seed["reserved_cost"])
            saved = create_trading_pending_entry(root, entry=_pending_entry_payload(origin="manual_selected", ticker="2330",
                information_date=seed_date, plan=seed, lineage=lineage, signal_date=None, planned_trade_date=order_date))
            if kind == "new":
                acquisition = resolve_confirmed_entry_plan_from_frame(entry=saved, frame=frame, params=p, fill_date=fill_date)
            else:
                # AI: Reproduce an immutable pre-upgrade acquisition containing
                # its saved late geometry. This is test evidence, not a second
                # historical strategy implementation or a mocked calculation.
                shadow = build_counterfactual_shadow_position_from_plan(seed,
                    t_open=100., t_high=100.1, t_low=99.9, params=p, ticker="2330", trade_date="2026-09-14")
                if kind == "late":
                    shadow["pending_exit_remaining_qty"] = 17
                acquisition = {**seed, "shadow_position_state": shadow}
            account = record_managed_manual_trading_buy(root, ticker="2330", qty=1000, price=99.95,
                trade_date=fill_date, expected_revision=account["revision"], params=p, execution_plan_seed=acquisition,
                management_lineage=lineage, management_start_date=fill_date)
            mark_trading_pending_entry_filled(root, pending_entry_id=saved["pending_entry_id"],
                fill={"ticker": "2330", "qty": 1000, "price": 99.95, "trade_date": fill_date})
            stored_account = deepcopy(account)
            stored_pending = load_trading_pending_entry_state(root)
            inspection = build_trading_single_stock_inspection(root, "2330", market_frame=frame, consumer_state=market)
            payload = project_trading_single_stock_chart_payload(chart, inspection, params=replace(p, atr_times_init=9.))
            rows = payload["trading_lifecycle_by_index"]
            pre, post = rows[frame.index.get_loc("2026-09-15")], rows[frame.index.get_loc(fill_date)]
            # Independent explicit golden values, not parity between two wrappers.
            check(kind + "_prefill_target_golden", 104., pre["tp_price"])
            check(kind + "_postfill_target_golden", 104., post["tp_price"])
            check(kind + "_stop_inherits_shadow_not_legacy_snapshot", [96., 96.], [pre["stop_price"], post["stop_price"]])
            check(kind + "_fill_price_is_actual_not_shadow", 99.95, post["entry_price"])
            check(kind + "_fill_qty_is_actual", 1000, post["position_qty"])
            check_true(kind + "_no_silent_projection_error", not inspection["decision_errors"])
            check(kind + "_research_summary_unchanged", chart["summary_box"], payload["summary_box"])
            check(kind + "_read_only_account", stored_account, load_trading_account_state(root))
            check(kind + "_read_only_pending", stored_pending, load_trading_pending_entry_state(root))
            fields = ("state", "stop_price", "tp_price", "entry_price", "position_qty")
            rows_by_kind[kind] = [[rows[i].get(k) for k in fields] for i in range(frame.index.get_loc(order_date), len(frame))]
            sync = run_trading_position_rollforward(root)
            check(kind + "_sync_succeeds", {}, sync["errors"])
            current = load_trading_account_state(root)
            management = current["positions"]["2330"]["strategy_management"]
            check(kind + "_sync_uses_same_target", 104., management["position_state"]["tp_half"])
            check_true(kind + "_obsolete_optional_obligation_removed", "pending_exit_remaining_qty" not in management["position_state"])
            check(kind + "_sync_keeps_actual_cash", stored_account["cash_milli"], current["cash_milli"])
            check(kind + "_sync_keeps_broker_record", stored_account["positions"]["2330"]["broker"], current["positions"]["2330"]["broker"])
            check(kind + "_sync_preserves_original_entry_snapshot", stored_account["positions"]["2330"]["strategy_management"]["entry_position_state"], management["entry_position_state"])
            check(kind + "_sync_preserves_original_entry_seed", stored_account["positions"]["2330"]["strategy_management"]["entry_execution_plan"], management["entry_execution_plan"])
            check_true(kind + "_sync_records_replay_evidence", bool(management.get("acquisition_replay", {}).get("resolved_entry_state_sha256")))
            check(kind + "_original_account_journal_is_prefix", stored_account["events"], current["events"][:len(stored_account["events"])])
            again = run_trading_position_rollforward(root)
            check(kind + "_rerun_is_idempotent", current, load_trading_account_state(root))
            check(kind + "_same_context_not_due", 0, build_trading_position_rollforward_snapshot(root)["due_count"])
            check(kind + "_pending_audit_unchanged", stored_pending, load_trading_pending_entry_state(root))
            # Partial sell and full manual exit are facts, not strategy proxies.
            current = record_manual_trading_sell(root, ticker="2330", qty=400, price=100., trade_date="2026-09-17", expected_revision=current["revision"])
            partial = build_trading_single_stock_inspection(root, "2330", market_frame=frame, consumer_state=market)
            pr = project_trading_single_stock_chart_payload(chart, partial, params=p)["trading_lifecycle_by_index"]
            check(kind + "_partial_fill_retains_management", [600, 104.], [pr[frame.index.get_loc("2026-09-17")]["position_qty"], pr[frame.index.get_loc("2026-09-17")]["tp_price"]])
            current = record_manual_trading_sell(root, ticker="2330", qty=600, price=100., trade_date=registration, expected_revision=current["revision"])
            closed = build_trading_single_stock_inspection(root, "2330", market_frame=frame, consumer_state=market)
            cr = project_trading_single_stock_chart_payload(chart, closed, params=p)["trading_lifecycle_by_index"]
            check(kind + "_closed_history_still_has_same_handoff", [104., 104., 0], [cr[frame.index.get_loc("2026-09-15")]["tp_price"], cr[frame.index.get_loc(fill_date)]["tp_price"], cr[len(frame)-1]["position_qty"]])
            # Source conflict is not a reason to fall back to the stale snapshot.
            record = stored_account["positions"]["2330"]
            binding = resolve_trading_position_management_binding(record)
            bad = deepcopy(saved); bad["management_lineage"]["lineage_id"] = "different-acquisition"
            try:
                build_confirmed_position_origin(record, binding=binding, params=p, account_events=stored_account["events"], frame=frame, prefill_entry=bad)
            except ValueError as exc:
                rejected = "mismatch" in str(exc)
            else:
                rejected = False
            check_true(kind + "_conflicting_prefill_rejected_not_snapshot_fallback", rejected)
            try:
                build_confirmed_position_origin(record, binding=binding, params=p, account_events=stored_account["events"], prefill_entry=saved)
            except RuntimeError as exc:
                rejected = "pre-entry market evidence" in str(exc)
            else:
                rejected = False
            check_true(kind + "_missing_market_cannot_certify_legacy_handoff", rejected)
    check("upgraded_late_and_valid_timely_full_lifecycle_match", rows_by_kind["timely"], rows_by_kind["late"])
    check("upgraded_legacy_and_new_full_lifecycle_match", rows_by_kind["new"], rows_by_kind["late"])


def validate_lifecycle_prefill_provenance_contract_case(base_params):
    results,summary,check,check_true=bind_synthetic_case("LIFECYCLE_PREFILL_PROVENANCE","lifecycle_provenance")
    p=_params(base_params);seed=_seed();seed["target_price"]=150.;seed["entry_atr"]=2.
    candidate=_candidate(p,seed);lineage=build_trading_candidate_strategy_lineage(candidate)
    entry=_pending_entry_payload(origin="scanner_strategy",ticker="2330",information_date="2026-09-10",plan=seed,lineage=lineage,signal_date="2026-09-10",planned_trade_date="2026-09-11")
    frame=_frame();frame.loc["2026-09-14",["Open","High","Low","Close"]]=[101.,103.,100.,102.]
    early=resolve_confirmed_entry_plan_from_frame(entry=entry,frame=frame,params=p,fill_date="2026-09-14")
    poisoned=frame.copy();poisoned.loc["2026-09-14":,["Open","High","Low","Close"]]=[500.,999.,1.,500.]
    again=resolve_confirmed_entry_plan_from_frame(entry=entry,frame=poisoned,params=p,fill_date="2026-09-14")
    check("backfilled_acquisition_excludes_fill_and_all_future_ohlc",canonical_json_sha256(json_native_value(early)),canonical_json_sha256(json_native_value(again)))
    check_true("backfilled_acquisition_inherits_actual_prior_shadow",bool(early.get("shadow_position_state")))
    entry["execution_plan_seed"]["init_sl"]=888.;entry["execution_plan_seed"]["target_price"]=999.
    third=resolve_confirmed_entry_plan_from_frame(entry=entry,frame=frame,params=p,fill_date="2026-09-14")
    check("latest_synced_pending_geometry_cannot_rewrite_past_acquisition",canonical_json_sha256(json_native_value(early)),canonical_json_sha256(json_native_value(third)))
    from core.trade_lifecycle import build_prefill_lifecycle_from_frame
    plan={"ticker":"2330","signal_date":"2026-09-10","information_date":"2026-09-10","limit_price":100.,"stop_price":96.,"init_trail":98.,"tp_price":108.,"entry_atr":2.,"planned_qty":1000}
    frozen=build_prefill_lifecycle_from_frame(frame=frame,plan=plan,params=p)
    # The chart's unrelated current-param arrays must not be consumed by this gateway.
    from services.trading.single_stock_inspection import _build_trading_prefill_lifecycle_timeline
    inspection={"ticker":"2330","candidate":candidate,"account_events":[],"entry_orders":[],"pending_entries":[]}
    chart={"date_labels":[d.strftime("%Y-%m-%d") for d in frame.index],**{c.lower():frame[c].to_numpy() for c in frame.columns},"atr":[999.]*len(frame),"sell_signals":[True]*len(frame)}
    first=_build_trading_prefill_lifecycle_timeline(inspection,chart,params=p)
    chart["atr"]=[.001]*len(frame);chart["sell_signals"]=[False]*len(frame)
    second=_build_trading_prefill_lifecycle_timeline(inspection,chart,params=replace(p,atr_times_init=4.))
    check("chart_binds_indicators_and_params_to_frozen_candidate",canonical_json_sha256(json_native_value(first)),canonical_json_sha256(json_native_value(second)))
    _check_dated_prefill_origin_capability(base_params, check, check_true)
    _check_confirmed_origin_upgrade_capability(base_params, check, check_true)
    summary["checks"]=len(results)
    return results,summary


def validate_trading_deferred_exit_full_cycle_case(base_params):
    """Real account/OMS files: buy, deferred TP partials, STOP, full close."""
    results, summary, check, check_true = bind_synthetic_case("TRADING_DEFERRED_FULL_CYCLE", "lifecycle_integration")
    from services.trading.order_state import load_trading_order_state, resolve_trading_order_state_path
    from services.trading.protection_order_submission import confirm_trading_protection_leg_submission
    from services.trading.fill_reconciliation import confirm_trading_protection_sell_order_fill, recover_trading_fill_transaction
    p = _params(base_params)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        account = initialize_trading_account_state(root, cash=1000000.)
        candidate = _candidate(p)
        account = record_strategy_trading_buy(root, ticker="2330", qty=1000, price=100., trade_date="2026-09-11",
            expected_revision=account["revision"], params=p, execution_plan_seed=candidate["execution_plan_seed"],
            strategy_lineage=build_trading_candidate_strategy_lineage(candidate))
        orders = build_empty_trading_order_state(timestamp="2026-09-11T18:00:00+08:00", mutation_id="cycle-init")
        atomic_write_json(resolve_trading_order_state_path(root), orders)
        frame = _frame()
        frame.loc["2026-09-11", ["High", "Low"]] = [111., 99.]
        frame.loc["2026-09-14":, ["Open", "High", "Low", "Close"]] = [108., 109., 107., 108.]
        state = {"market_date": "2026-09-11", "state_fingerprint": "cycle-entry"}
        with _market_io({"2330": frame}, state):
            synced = run_trading_lifecycle_sync(root)
            check("entry_day_tp_sync_is_current", SYNC_STATUS_LATEST, synced["status"])
            account = load_trading_account_state(root)
            check("touch_does_not_create_real_sell", 1000, account["positions"]["2330"]["broker"]["qty"])
            plan = build_trading_protection_plan(root)
            leg = next(leg for leg in plan["positions"][0]["legs"] if leg["action"] == "TP_HALF_DEFERRED_OPEN")
            check("deferred_tp_is_market_at_next_legal_open", ["MARKET", 500, None], [leg["order_type"], leg["qty"], leg.get("limit_price")])
            submitted = confirm_trading_protection_leg_submission(root, ticker="2330", action=leg["action"], expected_order_revision=orders["revision"])
            oid = next(iter(submitted["orders"]))
            rejected = False
            try:
                confirm_trading_protection_sell_order_fill(root, order_id=oid, fill_qty=200, fill_price=108., trade_date="2026-09-11",
                    expected_order_revision=submitted["revision"], expected_account_revision=account["revision"])
            except ValueError:
                rejected = True
            check("same_day_actual_sell_is_rejected_before_commit", True, rejected)
            check("rejected_fill_does_not_mutate_account", account, load_trading_account_state(root))
            partial = confirm_trading_protection_sell_order_fill(root, order_id=oid, fill_qty=200, fill_price=108., trade_date="2026-09-14",
                expected_order_revision=submitted["revision"], expected_account_revision=account["revision"])
            check("market_tp_accepts_actual_price_below_old_target", "PARTIAL", partial["status"])
            state.update(market_date="2026-09-14", state_fingerprint="cycle-partial")
            check("partial_cycle_resynchronizes", SYNC_STATUS_LATEST, run_trading_lifecycle_sync(root)["status"])
            plan = build_trading_protection_plan(root)
            tp = next(leg for leg in plan["positions"][0]["legs"] if leg["action"] == "TP_HALF_DEFERRED_OPEN")
            check("partial_keeps_original_half_target_not_half_of_remainder", 300, tp["qty"])
            account = load_trading_account_state(root)
            complete = confirm_trading_protection_sell_order_fill(root, order_id=oid, fill_qty=300, fill_price=108., trade_date="2026-09-15",
                expected_order_revision=partial["order_revision"], expected_account_revision=account["revision"])
            state.update(market_date="2026-09-15", state_fingerprint="cycle-tp-done")
            check("completed_tp_resynchronizes", SYNC_STATUS_LATEST, run_trading_lifecycle_sync(root)["status"])
            account = load_trading_account_state(root)
            management = account["positions"]["2330"]["strategy_management"]["position_state"]
            check("actual_half_completion_is_sticky_under_replay", [500, True, None], [management["qty"], management["sold_half"], management["pending_exit_action"]])
            frame.loc["2026-09-16", ["Open", "High", "Low", "Close"]] = [96., 97., 94., 95.]
            state.update(market_date="2026-09-16", state_fingerprint="cycle-stop")
            run_trading_lifecycle_sync(root)
            account = load_trading_account_state(root)
            check("stop_touch_retains_true_tail_inventory", 500, account["positions"]["2330"]["broker"]["qty"])
            plan = build_trading_protection_plan(root)
            stop = plan["positions"][0]["legs"][0]
            check("triggered_stop_routes_remaining_qty_to_market", ["STOP_EXIT_MARKET", "MARKET", 500], [stop["action"], stop["order_type"], stop["qty"]])
            orders = load_trading_order_state(root)
            submitted = confirm_trading_protection_leg_submission(root, ticker="2330", action=stop["action"], expected_order_revision=orders["revision"])
            stop_id = next(k for k, v in submitted["orders"].items() if v.get("action") == "STOP_EXIT_MARKET")
            done = confirm_trading_protection_sell_order_fill(root, order_id=stop_id, fill_qty=500, fill_price=93., trade_date="2026-09-17",
                expected_order_revision=submitted["revision"], expected_account_revision=account["revision"])
            check("only_confirmed_final_exit_closes_inventory", {}, done["account"]["positions"])
            check("completed_transaction_recovery_is_noop", False, recover_trading_fill_transaction(root))
            sells = [e for e in effective_trading_account_events(done["account"]) if e["mutation_type"] == "confirm_sell_fill"]
            check("exactly_three_confirmed_sell_events", 3, len(sells))
            check("confirmed_sell_quantities_conserve_acquisition", 1000, sum(int(e["details"].get("qty") or e["details"].get("fill_qty")) for e in sells))
    summary["checks"] = len(results)
    return results, summary


def validate_trading_signal_origin_persistence_case(base_params):
    """Isolated source-routing fixture; persistence and hash verification are real."""
    results, summary, check, check_true = bind_synthetic_case("TRADING_SIGNAL_ORIGIN", "lifecycle_provenance")
    from services.trading.signal_lineage import freeze_trading_candidate_lineages, resolve_trading_signal_lineage_path
    from core.file_integrity import load_json_strict
    p = _params(base_params)
    candidate = _candidate(p)
    frames = {"2330": _frame()}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        first = freeze_trading_candidate_lineages(root, candidate_rows=[candidate], prepared_frames=frames,
            information_date="2026-09-10", strategy_id="fixture", source_binding="params-v1")
        original = first[0]["signal_lineage"]
        p2 = replace(p, atr_times_init=4.)
        later = _candidate(p2, information_date="2026-09-17")
        frozen_result = _candidate(p, information_date="2026-09-17")
        frozen_result["kind"] = "extended"
        seen = []
        def producer(frame, ticker, params):
            seen.append((params.atr_times_init, str(frame.index.max().date())))
            return deepcopy(frozen_result)
        with patch("services.trading.signal_lineage.process_raw_stock_frame_actionable_detail", side_effect=producer):
            current = freeze_trading_candidate_lineages(root, candidate_rows=[later], prepared_frames=frames,
                information_date="2026-09-17", strategy_id="fixture", source_binding="params-v2")
        check("continuing_signal_calls_producer_with_original_params_and_cutoff", [(2., "2026-09-17")], seen)
        check("later_scan_cannot_replace_first_signal_identity", original, current[0]["signal_lineage"])
        check("continuing_candidate_is_recomputed_not_relabelled", "extended", current[0]["kind"])
        with patch("services.trading.signal_lineage.process_raw_stock_frame_actionable_detail", return_value=None):
            absent = freeze_trading_candidate_lineages(root, candidate_rows=[], prepared_frames=frames,
                information_date="2026-09-17", strategy_id="fixture", source_binding="params-v2")
        check("temporary_ineligibility_does_not_offer_a_candidate", [], absent)
        saved = load_json_strict(resolve_trading_signal_lineage_path(root))
        check("temporary_ineligibility_does_not_erase_original_binding", original, next(iter(saved["signals"].values())))
        with patch("services.trading.signal_lineage.process_raw_stock_frame_actionable_detail", side_effect=producer):
            returned = freeze_trading_candidate_lineages(root, candidate_rows=[later], prepared_frames=frames,
                information_date="2026-09-17", strategy_id="fixture", source_binding="params-v2")
        check("returning_candidate_uses_same_frozen_origin", original, returned[0]["signal_lineage"])
        key = next(iter(saved["signals"]))
        saved["signals"][key]["origin_candidate"]["execution_plan_seed"]["init_sl"] = 888.
        atomic_write_json(resolve_trading_signal_lineage_path(root), saved)
        rejected = False
        try:
            freeze_trading_candidate_lineages(root, candidate_rows=[later], prepared_frames=frames,
                information_date="2026-09-17", strategy_id="fixture", source_binding="params-v2")
        except ValueError:
            rejected = True
        check("tampered_frozen_origin_fails_closed", True, rejected)
    summary["checks"] = len(results)
    return results, summary


def validate_lifecycle_architecture_contract_case(base_params):
    """Dependency/ownership guards, separate from numerical parity tests."""
    import ast
    import inspect
    import core.position_management as owner
    import core.position_step as research
    import core.position_replay as replay
    results, summary, check, check_true = bind_synthetic_case("LIFECYCLE_ARCHITECTURE", "lifecycle_architecture")
    check("research_and_replay_use_identical_session_owner", owner.step_position_management, research.step_position_management)
    check("live_replay_imports_same_session_owner", owner.step_position_management, replay.step_position_management)
    check("legacy_public_helper_is_alias_not_second_implementation", owner.rollforward_position_management_from_completed_bar, research.rollforward_position_management_from_completed_bar)
    root = Path(__file__).resolve().parents[2]
    for name in ("position_management", "position_replay", "trading_position_projection", "trading_lifecycle_plans", "prefill_origin"):
        source = (root / "core" / (name + ".py")).read_text(encoding="utf-8")
        imports = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import): imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom): imports.append(node.module or "")
        check("core_has_no_app_ui_or_service_dependency_" + name, [], [x for x in imports if x.startswith(("services.", "apps.", "tools.", "PyQt"))])
    for name in ("position_rollforward", "single_stock_inspection"):
        source = (root / "services" / "trading" / (name + ".py")).read_text(encoding="utf-8")
        calls = {node.func.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        check("adapter_does_not_own_rule_primitive_sequence_" + name, [], sorted(calls & {"_update_trailing_stop", "resolve_position_intraday_exit_hits", "rollforward_position_management_from_completed_bar", "build_position_from_entry_fill", "execute_bar_step"}))
        check_true("adapter_delegates_to_canonical_replay_" + name, "replay_confirmed_position_management" in calls)
    import services.trading.lifecycle_sync as pending_sync
    calls = {node.func.id for node in ast.walk(ast.parse(inspect.getsource(pending_sync._sync_one_pending)))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    check("management_sync_cannot_accept_or_resize_order_decisions", [], sorted(calls & {
        "accept_entry_quantity_decision", "build_cash_capped_entry_plan", "resize_candidate_plan_to_capital",
        "calc_position_size"}))
    check_true("management_sync_uses_canonical_accepted_reservation_guard", "validate_accepted_entry_reservation" in calls)
    p = _params(base_params)
    pos = build_position_from_frozen_entry_plan(_seed(), buy_price=100., qty=1000, params=p, ticker="2330", trade_date="2026-09-11")
    economics = {k: deepcopy(v) for k, v in pos.items() if k not in POSITION_MANAGEMENT_FIELDS}
    complete_position_entry_session(pos, t_high=111., t_low=94., params=p)
    check("management_owner_can_only_change_management_fields", json_native_value(economics), json_native_value({k:v for k,v in pos.items() if k not in POSITION_MANAGEMENT_FIELDS}))
    summary["checks"] = len(results)
    return results, summary


@contextmanager
def _raw_market_io(frames):
    """Fixture storage only; real calendar, range/tick and fill guards execute."""
    from datetime import date
    from services.trading.market_data_v2_view import TradingMarketDataV2View
    class RawView:
        def read_dataset_frame(self, dataset, *, data_id=None, columns=None, start_date=None, end_date=None, **kwargs):
            frame = frames[data_id] if data_id else next(iter(frames.values()))
            if start_date is not None: frame = frame.loc[start_date:]
            if end_date is not None: frame = frame.loc[:end_date]
            result = pd.DataFrame({"date": frame.index.strftime("%Y-%m-%d"), "stock_id": data_id or "2330",
                "open": frame["Open"].to_numpy(), "max": frame["High"].to_numpy(),
                "min": frame["Low"].to_numpy(), "close": frame["Close"].to_numpy(),
                "Trading_Volume": frame["Volume"].to_numpy()})
            return result.loc[:, list(columns)] if columns else result
    with patch.object(TradingMarketDataV2View, "open", return_value=RawView()), patch("services.trading.actual_fill_validation._current_taipei_date", return_value=date(2026,9,19)):
        yield


def validate_trading_discretion_pending_fill_correction_case(base_params):
    """Actual file-backed intents, edits, fills, corrections and replay."""
    from services.trading.pending_entry_service import (create_scanner_trading_pending_entry,
        update_trading_pending_entry_intent, fill_trading_pending_entry, cancel_pending_entry_no_fill,
        recover_trading_pending_entry_transaction)
    from services.trading.account_trade_entry import correct_trading_account_transaction
    from core.trading_account_state import TRADE_MUTATION_STRATEGY_BUY
    results, summary, check, check_true = bind_synthetic_case("TRADING_DISCRETION_PENDING_CYCLE", "lifecycle_integration")
    p = _params(base_params)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); initialize_trading_account_state(root, cash=1000000.)
        first = _candidate(p, ticker="2330"); second = _candidate(p, ticker="2317")
        first["rank"] = 1; second["rank"] = 9
        frames = {"2330": _frame(), "2317": _frame()}
        state = {"market_date": "2026-09-18", "state_fingerprint": "discretion-1"}
        runtime = {"params": p, "latest_data_date": state["market_date"]}
        def resolve(_root, *, ticker, candidate_reference):
            row = {"2330": first, "2317": second}[ticker]
            if candidate_reference != row: raise ValueError("stale candidate fixture")
            return deepcopy(row)
        with _market_io(frames, state), _raw_market_io(frames), \
             patch("services.trading.pending_entry_service.load_trading_scanner_runtime", return_value=runtime), \
             patch("services.trading.account_trade_entry.load_trading_scanner_runtime", return_value=runtime), \
             patch("services.trading.pending_entry_service.resolve_current_trading_scanner_candidate", side_effect=resolve):
            pending = create_scanner_trading_pending_entry(root, candidate_reference=second, qty=600, planned_trade_date="2026-09-11")
            row = pending; eid = row["pending_entry_id"]
            pending = load_trading_pending_entry_state(root)
            check("lower_rank_may_be_selected_without_buying_first_rank", ["2317"], [v["ticker"] for v in pending["entries"].values()])
            check("accepted_quantity_is_user_selected_not_max", 600, row["planned_qty"])
            frozen_sha = row["management_lineage"]["frozen_params_sha256"]
            pending = update_trading_pending_entry_intent(root, pending_entry_id=eid, origin="scanner_strategy", ticker="2317", qty=400, planned_trade_date="2026-09-11")
            row = pending
            check("user_may_edit_existing_pending_downward", 400, row["planned_qty"])
            synced = run_trading_lifecycle_sync(root)
            check("synchronization_keeps_user_discretion", [SYNC_STATUS_LATEST, 400], [synced["status"], load_trading_pending_entry_state(root)["entries"][eid]["planned_qty"]])
            rejected = False
            try: fill_trading_pending_entry(root, pending_entry_id=eid, qty=401, price=100., trade_date="2026-09-14")
            except ValueError: rejected = True
            check("fill_above_accepted_quantity_is_rejected", True, rejected)
            filled = fill_trading_pending_entry(root, pending_entry_id=eid, qty=300, price=100., trade_date="2026-09-14")
            account = load_trading_account_state(root); position = account["positions"]["2317"]
            check("actual_quantity_does_not_rewrite_accepted_or_strategy_quantity", [1000,400,300], [second["proj_qty"],load_trading_pending_entry_state(root)["entries"][eid]["planned_qty"],position["broker"]["qty"]])
            check("pending_transfer_preserves_frozen_identity", frozen_sha, position["strategy_lineage"]["frozen_params_sha256"])
            check("backfilled_acquisition_excludes_later_shadow", "2026-09-11", position["strategy_management"]["entry_execution_plan"]["management_information_date"])
            check_true("shadow_management_transfers_to_actual_position", position["strategy_management"]["entry_position_state"].get("inherited_shadow_management", False))
            check("pending_transfer_restart_has_no_extra_action", None, recover_trading_pending_entry_transaction(root))
            original = next(e for e in effective_trading_account_events(account) if e["mutation_type"] == TRADE_MUTATION_STRATEGY_BUY)
            corrected = correct_trading_account_transaction(root, transaction_revision=original["revision"], qty=250, price=100., trade_date="2026-09-15", expected_account_revision=account["revision"])
            account = load_trading_account_state(root)
            before_cash = account["cash_milli"]; before_broker = deepcopy(account["positions"]["2317"]["broker"])
            synced = run_trading_lifecycle_sync(root)
            check("date_quantity_correction_replays_successfully", SYNC_STATUS_LATEST, synced["status"])
            account = load_trading_account_state(root)
            check("management_replay_does_not_rewrite_corrected_broker", before_broker, account["positions"]["2317"]["broker"])
            check("management_replay_does_not_rewrite_corrected_cash", before_cash, account["cash_milli"])
            check("correction_sets_true_acquisition_start", "2026-09-15", account["positions"]["2317"]["strategy_management"]["management_start_date"])
            repeat_before = canonical_json_sha256(account); run_trading_lifecycle_sync(root)
            check("corrected_fill_replay_is_idempotent", repeat_before, canonical_json_sha256(load_trading_account_state(root)))
            pending = create_scanner_trading_pending_entry(root, candidate_reference=first, qty=200, planned_trade_date="2026-09-11")
            cancel_id = pending["pending_entry_id"]
            cancelled = cancel_pending_entry_no_fill(root, pending_entry_id=cancel_id)
            check("user_can_decline_an_unfilled_candidate", "CANCELLED_NO_FILL", cancelled["status"])
            check("cancel_does_not_create_position", False, "2330" in load_trading_account_state(root)["positions"])
    summary["checks"] = len(results)
    return results, summary


def validate_lifecycle_same_session_confirmation_order_case(base_params):
    results, summary, check, check_true = bind_synthetic_case("LIFECYCLE_CONFIRMATION_ORDER", "lifecycle_ssot")
    p = _params(base_params); frame = _frame(end="2026-09-14")
    frame.loc["2026-09-11", ["High","Low"]] = [111.,99.]
    frame.loc["2026-09-14", ["Open","High","Low","Close"]] = [108.,109.,107.,108.]
    initial = build_position_from_frozen_entry_plan(_seed(), buy_price=100., qty=1000, params=p, ticker="2330", trade_date="2026-09-11")
    sells = [False] * len(frame); sells[frame.index.get_loc("2026-09-11")] = True
    with patch("core.position_replay.unpack_precomputed_signals", return_value=([5.] * len(frame), [False] * len(frame), sells, [None] * len(frame))):
        projection = replay_confirmed_position_management(initial, frame=frame, params=p, confirmed_quantity_events=[
            {"trade_date":"2026-09-14", "qty_delta":-500, "event":"TP_HALF", "tp_half_complete":True},
            {"trade_date":"2026-09-14", "qty_delta":-500, "event":"IND_SELL"}])
    check("confirmed_half_exit_is_applied_before_later_full_decision", [["TP_HALF",500],["IND_SELL",500]], [[d["event"],d["qty"]] for d in projection["decisions"]])
    check("no_double_sell_after_two_confirmed_legs", 0, projection["position_state"]["qty"])
    summary["checks"] = len(results)
    return results, summary


def validate_lifecycle_randomized_confirmed_replay_case(base_params):
    """Fixed-seed capability matrix: real signals, simulator facts, one replay."""
    import random
    from core.position_management import rollforward_position_management_from_completed_bar
    from core.position_step import SETTLEMENT_BASIS_LEDGER_NET
    from core.signal_utils import generate_signals, unpack_precomputed_signals
    results, summary, check, check_true = bind_synthetic_case("LIFECYCLE_RANDOMIZED_REPLAY", "lifecycle_ssot")
    rng = random.Random(838291)
    fields = ("qty", "sold_half", "pending_exit_action", "highest_high_since_entry_milli", "sl_milli")
    p = _params(base_params)
    for case in range(32):
        frame = _frame(); dates = frame.index[frame.index >= "2026-09-11"]; prior_close = 100.
        for date in dates:
            op = prior_close * rng.uniform(.96, 1.04)
            high, low = op * rng.uniform(1., 1.06), op * rng.uniform(.94, 1.)
            close = rng.uniform(low, high)
            frame.loc[date, ["Open", "High", "Low", "Close", "Volume"]] = [op, high, low, close, rng.choice([0., 100000., 100000.])]
            prior_close = close
        initial = build_position_from_frozen_entry_plan(_seed(), buy_price=100., qty=1000, params=p, ticker="2330", trade_date="2026-09-11")
        pos = deepcopy(initial)
        atr, _, sell, _ = unpack_precomputed_signals(generate_signals(frame, p, ticker="2330"))
        facts, expected = [], []
        for date in dates:
            idx = frame.index.get_loc(date); bar, prior = frame.iloc[idx], frame.iloc[idx - 1]
            day = str(date.date())
            if day == "2026-09-11":
                complete_position_entry_session(pos, t_high=bar.High, t_low=bar.Low, params=p)
            else:
                execute_bar_step(pos, y_atr=atr[idx - 1], y_ind_sell=bool(sell[idx - 1]), y_close=prior.Close, y_high=prior.High,
                    t_open=bar.Open, t_high=bar.High, t_low=bar.Low, t_close=bar.Close, t_volume=bar.Volume,
                    params=p, current_date=day, settlement_basis=SETTLEMENT_BASIS_LEDGER_NET)
                for leg in pos.get("_last_exec_contexts", []):
                    facts.append({"trade_date": day, "qty_delta": -leg["qty"], "event": leg["event"], "tp_half_complete": leg["event"] == "TP_HALF"})
            snapshot = deepcopy(pos)
            if day != "2026-09-11" and pos["qty"] > 0:
                rollforward_position_management_from_completed_bar(snapshot, completed_high=bar.High, completed_atr=atr[idx], params=p)
            expected.append((day, json_native_value({key: snapshot.get(key) for key in fields})))
            if pos["qty"] == 0:
                break
        replay = replay_confirmed_position_management(initial, frame=frame, params=p, confirmed_quantity_events=facts)
        actual = [(row["date"], json_native_value({key: row["position_state"].get(key) for key in fields})) for row in replay["sessions"]]
        check("simulator_facts_and_replay_agree_%03d" % case, expected, actual)
    summary["checks"] = len(results)
    return results, summary

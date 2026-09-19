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
    summary["checks"]=len(results)
    return results,summary


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
    for name in ("position_management", "position_replay", "trading_position_projection", "trading_lifecycle_plans"):
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

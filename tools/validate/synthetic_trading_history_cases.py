"""Historical Trading charts consume corrected facts and per-acquisition replay.

AI: Real account writes, frozen Params and lifecycle engine; storage injection
only where a consumer source is needed. No production settings are mutated.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
import tempfile
import hashlib

import numpy as np
import pandas as pd

from .checks import bind_synthetic_case
from core.file_integrity import canonical_json_sha256
from core.params_io import params_to_json_dict
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.serialization_utils import json_native_value
from core.trading_account_state import project_trading_account_transactions, apply_confirmed_sell_fill
from services.trading.account_state import (
    initialize_trading_account_state, record_managed_manual_trading_buy,
    record_manual_trading_sell, correct_trading_transaction, load_trading_account_state,
    resolve_trading_account_state_path, record_manual_trading_buy,
)
from services.trading.accounting_policy import build_standalone_trading_accounting_params
from services.trading.strategy_param_runtime import build_trading_manual_management_lineage
from services.trading.single_stock_inspection import (
    build_trading_single_stock_inspection, project_trading_single_stock_chart_payload,
    _inspection_account_events, _sell_trace_name,
)
from core.trading_position_projection import collect_confirmed_position_cycles
from services.trade_analysis.charting import (normalize_chart_payload_contract, build_chart_hover_snapshot,
    CHART_TRADE_LABEL_TRACE_NAMES, extract_trade_marker_indexes, _build_trade_label_text)

MANUAL_SELL = "\u624b\u52d5\u8ce3\u51fa"
INDICATOR_SELL = "\u6307\u6a19\u8ce3\u51fa"
BUY = "\u8cb7\u9032"


def _params(base):
    return replace(base, use_breakout_quality_filter=False, use_breakout_quality_ranking=False,
                   atr_times_init=2.0, atr_times_trail=1.0, tp_percent=0.5)


def _frame():
    frame = pd.DataFrame({"Open":100.0,"High":101.0,"Low":99.0,"Close":100.0,"Volume":1000000.0},
                         index=pd.bdate_range(end="2026-09-18", periods=340))
    frame.loc["2026-08-04":, ["Open","High","Low","Close"]] = [106.0,108.0,105.0,107.0]
    return frame


def _chart(frame):
    return {"x":np.arange(len(frame)), "dates":frame.index,
            "date_labels":frame.index.strftime("%Y-%m-%d").tolist(),
            **{key:frame[key.title()].to_numpy() for key in ("open","high","low","close","volume")},
            "marker_groups":{}, "signal_annotations":[],
            "summary_box":["Research canonical metrics unchanged"], "status_box":{}}


def _acquire(root, account, params, *, date="2026-08-03", qty=1000, price=100.0):
    seed = {"entry_type":"manual_direct_backfill", "entry_atr":2.0,
            "init_sl":price-4.0, "init_trail":price-2.0, "target_price":price+30.0,
            "target_reference_price":price, "limit_price":None, "trade_date":date}
    lineage = build_trading_manual_management_lineage(
        params=params, execution_plan_seed=seed, information_date="2026-09-18",
        origin="manual_direct_backfill", target_reference_close=price)
    account = record_managed_manual_trading_buy(root, ticker="2330", qty=qty, price=price,
        trade_date=date, expected_revision=account["revision"], params=params,
        execution_plan_seed=seed, management_lineage=lineage, management_start_date=date)
    return account, lineage


def _project(root, chart, params):
    frame = pd.DataFrame({key.title(): list(chart[key]) for key in ("open", "high", "low", "close", "volume")},
                         index=pd.to_datetime(chart["date_labels"]))
    inspection = build_trading_single_stock_inspection(root, "2330", market_frame=frame,
        consumer_state={"market_date": frame.index[-1].strftime("%Y-%m-%d")})
    return project_trading_single_stock_chart_payload(chart, inspection, params=params), inspection


def _idx(chart, date):
    return chart["date_labels"].index(date)


def _digest(value):
    def native(item):
        if isinstance(item, dict):
            return {str(key): native(val) for key, val in item.items()}
        if isinstance(item, (list, tuple, np.ndarray, pd.Index)):
            return [native(val) for val in item]
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        return json_native_value(item)
    return canonical_json_sha256(native(value))


def validate_trading_confirmed_cycle_history_contract_case(base_params):
    results, summary, check, check_true = bind_synthetic_case("TRADING_CONFIRMED_CYCLE_HISTORY", "trading_history")
    p = _params(base_params); frame = _frame(); chart = _chart(frame)
    chart["marker_groups"] = {INDICATOR_SELL:[{"x":_idx(chart,"2026-08-11"),"price":107.0,"meta":{"qty":9999}}]}
    chart["signal_annotations"] = [
        {"x":_idx(chart,"2026-08-11"),"date":"2026-08-11","signal_type":"buy","anchor_price":108.0,"meta":{"qty":9999}},
        {"x":_idx(chart,"2026-08-12"),"date":"2026-08-12","signal_type":"sell","anchor_price":108.0,"meta":{"qty":9999}},
    ]
    original = _digest(chart)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); account = initialize_trading_account_state(root, cash=1000000.0)
        account, lineage = _acquire(root, account, p)
        before, _ = _project(root, chart, p)
        check("actual_raised_trail_is_not_initial_snapshot", 106.0, before["stop_line"][_idx(chart,"2026-08-25")])
        account = record_manual_trading_sell(root, ticker="2330",qty=1000,price=107.0,trade_date="2026-08-26",expected_revision=account["revision"])
        state_file = resolve_trading_account_state_path(root)
        state_bytes = state_file.read_bytes()
        after, inspection = _project(root, chart, replace(p,atr_times_trail=5.0,atr_len=2))
        check("full_exit_has_no_current_position", None, inspection["current_position"])
        check("closed_cycle_retains_frozen_binding", lineage["lineage_id"], inspection["position_history"][0]["binding"]["lineage_id"])
        check_true("closing_inventory_preserves_every_pre_exit_management_bar",
            np.allclose(before["stop_line"][:_idx(chart,"2026-08-26")], after["stop_line"][:_idx(chart,"2026-08-26")], equal_nan=True))
        check("manual_sale_marker_date", ["2026-08-26"], [m["date"] for m in after["marker_groups"].get(MANUAL_SELL,[])])
        check_true("manual_sale_not_inferred_indicator_fill", not after["marker_groups"].get(INDICATOR_SELL))
        check("manual_reason_is_persisted_fact", "MANUAL_ACCOUNT_SELL", after["marker_groups"][MANUAL_SELL][0]["meta"]["execution_reason"])
        check_true("manual_exit_is_renderable_info_box", MANUAL_SELL in CHART_TRADE_LABEL_TRACE_NAMES)
        check_true("manual_exit_is_keyboard_navigable", _idx(chart,"2026-08-26") in extract_trade_marker_indexes(after))
        check_true("manual_exit_info_box_title_is_not_indicator", _build_trade_label_text(MANUAL_SELL, after["marker_groups"][MANUAL_SELL][0]).startswith(MANUAL_SELL))
        exit_row = after["trading_lifecycle_by_index"][_idx(chart,"2026-08-26")]
        check("exit_day_has_zero_remaining_position", 0, exit_row["position_qty"])
        check("exit_day_is_explicitly_closed", "\u5df2\u51fa\u5834", exit_row["display_state"])
        check("exit_day_line_equals_its_sidebar", after["stop_line"][_idx(chart,"2026-08-26")], exit_row["stop_price"])
        check_true("next_day_has_no_ghost_holding", _idx(chart,"2026-08-27") not in after["trading_lifecycle_by_index"])
        check_true("next_day_has_no_actual_lines", np.isnan(after["entry_line"][_idx(chart,"2026-08-27")]))
        check_true("research_hypothetical_entries_suppressed_while_held", not any(x.get("signal_type")=="buy" for x in after["signal_annotations"]))
        check("research_summary_preserved", chart["summary_box"], after["summary_box"])
        check("input_research_payload_not_mutated", original, _digest(chart))
        check("inspection_does_not_write_account", hashlib.sha256(state_bytes).hexdigest(), hashlib.sha256(state_file.read_bytes()).hexdigest())
        again = project_trading_single_stock_chart_payload(after, inspection, params=p)
        check("overlay_is_idempotent", _digest(after), _digest(again))
        normalized = normalize_chart_payload_contract(after)
        for date in ("2026-08-03","2026-08-05","2026-08-25","2026-08-26"):
            idx = _idx(chart,date); row = after["trading_lifecycle_by_index"][idx]
            check(f"actual_entry_not_clipped_above_cost_{date}", row["entry_price"], float(normalized["entry_line"][idx]))
            hover = build_chart_hover_snapshot(normalized,idx)
            check(f"hover_uses_same_lifecycle_{date}", row, hover.get("trading_lifecycle_state"))
        # Adjacent acquisitions must not look like one continuous stop/entry.
        from services.trade_analysis.charting import (
            _chart_transaction_line_connections, _render_bar_scoped_transaction_line,
            _transaction_line_plotly_points,
        )
        from matplotlib.figure import Figure
        connections = _chart_transaction_line_connections({
            "trading_overlay_source":"canonical_execution_lifecycle", "x":[0,1],
            "trading_lifecycle_by_index":{0:{"cycle_id":"old"},1:{"cycle_id":"new"}},
        })
        check("adjacent_cycles_do_not_share_render_connector", [False], connections)
        fig=Figure(); ax=fig.add_subplot(111)
        artists=_render_bar_scoped_transaction_line(ax,[0,1],[100.,107.],color="blue",linewidth=1.,connections=connections)
        check("each_adjacent_cycle_bar_still_visible",2,sum(len(a.get_segments()) for a in artists))
        check("no_cross_cycle_vertical_price_jump",1,len(artists))
        html_x,html_y=_transaction_line_plotly_points(["2026-08-31","2026-09-01"],[100.,107.],connections)
        check_true("html_preserves_bars_with_nan_boundary",html_y[0]==100. and np.isnan(html_y[1]) and html_y[2]==107.)
        # Same ticker, new acquisition with DIFFERENT frozen geometry.
        old_stops = deepcopy(after["stop_line"])
        p2 = replace(p,atr_times_init=4.0,atr_times_trail=5.0,atr_len=2)
        account, lineage2 = _acquire(root, account, p2, date="2026-09-01", qty=600, price=107.0)
        second, inspection2 = _project(root, chart, p2)
        check("two_separate_acquisition_cycles", 2, len(inspection2["position_history"]))
        check_true("new_position_does_not_rewrite_old_geometry", np.allclose(old_stops[:_idx(chart,"2026-08-27")], second["stop_line"][:_idx(chart,"2026-08-27")], equal_nan=True))
        check("new_cycle_actual_qty", 600, second["trading_lifecycle_by_index"][_idx(chart,"2026-09-01")]["position_qty"])
        check_true("cycles_keep_distinct_bindings", inspection2["position_history"][0]["binding"]["frozen_params_sha256"] != inspection2["position_history"][1]["binding"]["frozen_params_sha256"])
        # Full-source producer, cropped presentation window, no parquet dependency.
        reads=[]
        def load_frame(**kwargs):
            reads.append((kwargs["allowed_date"],kwargs["params"].atr_len))
            return frame.copy()
        with ExitStack() as stack:
            stack.enter_context(patch("services.trading.market_data_consumer.load_trading_v2_consumer_state", return_value={"market_date":"2026-09-18"}))
            stack.enter_context(patch("services.trading.lifecycle_context.resolve_trading_lifecycle_context",return_value=SimpleNamespace(view=object(),finalized_date="2026-09-18")))
            stack.enter_context(patch("services.trading.position_market_context.load_trading_position_market_frame",side_effect=load_frame))
            full_inspection=build_trading_single_stock_inspection(root,"2330")
        cropped = _chart(frame.loc["2026-08-20":"2026-08-27"])
        cropped_payload = project_trading_single_stock_chart_payload(cropped,full_inspection,params=p2)
        check("cropped_chart_uses_full_acquisition_history", second["stop_line"][_idx(chart,"2026-08-25")], cropped_payload["stop_line"][_idx(cropped,"2026-08-25")])
        check("frozen_param_specific_source_loads", sorted([p.atr_len,p2.atr_len]), sorted(x[1] for x in reads))
        invalid = deepcopy(full_inspection)
        invalid["position_history"][0]["projection_error"]="historical source unavailable"
        degraded=project_trading_single_stock_chart_payload(chart,invalid,params=p)
        row=degraded["trading_lifecycle_by_index"][_idx(chart,"2026-08-25")]
        check("failed_redundant_read_does_not_erase_verified_full_history", second["stop_line"][_idx(chart,"2026-08-25")], degraded["stop_line"][_idx(chart,"2026-08-25")])
        check_true("full_history_recovery_has_no_false_management_error", not row.get("management_projection_error"))
        check("missing_history_still_keeps_actual_fill", 100.0,row["entry_price"])
    return results,summary


def validate_trading_corrected_chart_ledger_contract_case(base_params):
    results,summary,check,check_true=bind_synthetic_case("TRADING_CORRECTED_CHART_LEDGER","trading_history")
    p=_params(base_params); chart=_chart(_frame())
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);account=initialize_trading_account_state(root,cash=1000000.0)
        account,lineage=_acquire(root,account,p); buy_revision=account["revision"]
        account=record_manual_trading_sell(root,ticker="2330",qty=1000,price=107.0,trade_date="2026-08-26",expected_revision=account["revision"])
        sale_revision=account["revision"]
        account,second_lineage=_acquire(root,account,replace(p,atr_times_trail=6.0),date="2026-09-01",qty=600,price=107.0)
        account=correct_trading_transaction(root,transaction_revision=buy_revision,qty=1000,price=102.0,trade_date="2026-08-04",expected_revision=account["revision"])
        path=resolve_trading_account_state_path(root);before_bytes=path.read_bytes()
        projected,inspection=_project(root,chart,p)
        check("correction_origin_binding_not_later_ticker_template",lineage["lineage_id"],inspection["position_history"][0]["binding"]["lineage_id"])
        check("later_cycle_keeps_own_binding",second_lineage["lineage_id"],inspection["position_history"][1]["binding"]["lineage_id"])
        check_true("old_buy_date_removed_after_correction",not any(m["date"]=="2026-08-03" for m in projected["marker_groups"].get(BUY,[])))
        row=projected["trading_lifecycle_by_index"][_idx(chart,"2026-08-04")]
        check("corrected_entry_date", "2026-08-04",row["entry_date"])
        check("corrected_entry_price",102.0,row["entry_price"])
        check_true("corrected_history_replay_succeeded",not row.get("management_projection_error"))
        economic=project_trading_account_transactions(account,accounting_params=build_standalone_trading_accounting_params())
        sale=next(e["details"] for e in economic if e["revision"]==sale_revision)
        raw_sale=next(e["details"] for e in account["events"] if e["revision"]==sale_revision)
        marker=projected["marker_groups"][MANUAL_SELL][0]["meta"]
        check_true("fixture_contains_stale_immutable_sell_cost",sale["allocated_cost_milli"]!=raw_sale["allocated_cost_milli"])
        check("chart_uses_account_table_corrected_pnl",sale["realized_pnl_milli"]/1000.,marker["pnl_value"])
        check("chart_uses_account_table_net_proceeds",sale["net_sell_total_milli"]/1000.,marker["sell_capital"])
        check("chart_uses_corrected_pnl_denominator",sale["realized_pnl_milli"]/sale["allocated_cost_milli"]*100.,marker["pnl_pct"])
        check("read_model_preserves_hash_chain",hashlib.sha256(before_bytes).hexdigest(),hashlib.sha256(path.read_bytes()).hexdigest())
        account=correct_trading_transaction(root,transaction_revision=sale_revision,qty=1000,price=108.0,trade_date="2026-08-27",expected_revision=account["revision"])
        corrected,_=_project(root,chart,p)
        check("corrected_sale_moves_only_actual_marker",["2026-08-27"],[m["date"] for m in corrected["marker_groups"][MANUAL_SELL]])
        check("prior_sale_day_holds_until_corrected_date",1000,corrected["trading_lifecycle_by_index"][_idx(chart,"2026-08-26")]["position_qty"])
        check("corrected_exit_day_qty_zero",0,corrected["trading_lifecycle_by_index"][_idx(chart,"2026-08-27")]["position_qty"])
    return results,summary


def validate_trading_partial_exit_chart_contract_case(base_params):
    results,summary,check,check_true=bind_synthetic_case("TRADING_PARTIAL_EXIT_CHART","trading_history")
    p=_params(base_params);chart=_chart(_frame())
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);account=initialize_trading_account_state(root,cash=1000000.0)
        account,lineage=_acquire(root,account,p)
        # Domain confirmation owner, including incomplete/complete TP facts.
        for date,qty,complete in (("2026-08-10",200,False),("2026-08-11",300,True)):
            account=apply_confirmed_sell_fill(account,ticker="2330",qty=qty,exec_price=107.0,
                params=build_standalone_trading_accounting_params(),timestamp=date+"T18:00:00+08:00",
                mutation_id="tp-"+date,trade_date=date,event="TP_HALF",mark_tp_half_complete=complete)
        events=_inspection_account_events(account)
        inspection={"ticker":"2330","account_events":events,"account_audit_events":account["events"],
            "current_position":account["positions"]["2330"],"entry_orders":[],"pending_entries":[],
            "position_history":collect_confirmed_position_cycles(events,ticker="2330",current_position=account["positions"]["2330"])}
        payload=project_trading_single_stock_chart_payload(chart,inspection,params=p)
        partial=payload["trading_lifecycle_by_index"][_idx(chart,"2026-08-10")]
        complete=payload["trading_lifecycle_by_index"][_idx(chart,"2026-08-11")]
        check("partial_tp_only_reduces_actual_filled_quantity",800,partial["position_qty"])
        check("partial_tp_keeps_unfulfilled_target",False,partial["sold_half"])
        check_true("partial_tp_target_still_visible",np.isfinite(payload["tp_line"][_idx(chart,"2026-08-10")]))
        check("complete_tp_remaining_quantity",500,complete["position_qty"])
        check("complete_tp_acknowledges_once",True,complete["sold_half"])
        check_true("complete_tp_target_no_longer_visible",np.isnan(payload["tp_line"][_idx(chart,"2026-08-11")]))
    for reason,expected in (("MANUAL_ACCOUNT_SELL",MANUAL_SELL),("MANUAL_CONFIRMED_SELL",MANUAL_SELL),
                           ("ACCOUNT_CORRECTION_SELL",MANUAL_SELL),("IND_SELL",INDICATOR_SELL),
                           ("STOP","\u505c\u640d\u8ce3\u51fa"),("TP_HALF","\u505c\u5229"),
                           ("UNKNOWN_BROKER_REASON","\u8ce3\u51fa\u6210\u4ea4"),("","\u8ce3\u51fa\u6210\u4ea4")):
        check("actual_reason_mapping_"+reason,expected,_sell_trace_name(reason))
    # Legacy unmanaged facts must not borrow current strategy geometry.
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);account=initialize_trading_account_state(root,cash=1000000.0)
        account=record_manual_trading_buy(root,ticker="2330",qty=500,price=100.0,trade_date="2026-08-03",expected_revision=account["revision"])
        account=record_manual_trading_sell(root,ticker="2330",qty=500,price=107.0,trade_date="2026-08-26",expected_revision=account["revision"])
        legacy,_=_project(root,chart,p)
        check_true("unbound_legacy_holding_has_no_invented_strategy_stop",np.isnan(legacy["stop_line"][_idx(chart,"2026-08-25")]))
        check("legacy_actual_entry_is_still_shown",100.,legacy["entry_line"][_idx(chart,"2026-08-25")])
        check("legacy_manual_sale_remains_manual",1,len(legacy["marker_groups"][MANUAL_SELL]))
    return results,summary


def validate_trading_prefill_position_exclusion_contract_case(base_params):
    """Real full shadow timelines, not empty payloads that cannot expose revival."""
    from core.trade_lifecycle import build_prefill_lifecycle_from_frame
    from services.trading.single_stock_inspection import resolve_trading_single_stock_sidebar_state
    results, summary, check, check_true = bind_synthetic_case("TRADING_PREFILL_POSITION_EXCLUSION", "trading_history")
    p = replace(_params(base_params), atr_times_init=8.0, atr_times_trail=8.0,
                use_bb=False, use_kc=False)
    frame = _frame(); chart = _chart(frame)
    def prefill(signal):
        plan = {"source":"strategy_prefill", "signal_date":signal, "information_date":signal,
                "limit_price":107.0, "stop_price":70.0, "init_trail":70.0,
                "tp_price":180.0, "entry_atr":None, "planned_qty":1000,
                "reserved_capital":107000.0, "ticker":"2330", "entry_type":"normal"}
        return build_prefill_lifecycle_from_frame(frame=frame, plan=plan, params=p)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); account=initialize_trading_account_state(root,cash=1000000.0)
        account, lineage = _acquire(root,account,p)
        account=record_manual_trading_sell(root,ticker="2330",qty=1000,price=107.0,
                                          trade_date="2026-08-26",expected_revision=account["revision"])
        for signal in ("2026-07-30", "2026-08-18", "2026-08-26"):
            original=prefill(signal)
            check_true(f"fixture_has_shadow_after_exit_{signal}", _idx(chart,"2026-09-10") in original)
            chart["strategy_prefill_lifecycle_by_index"]=original
            chart["signal_annotations"]=[{"x":_idx(chart,signal),"date":signal,"signal_type":"buy","title":"\u8cb7\u8a0a","anchor_price":107.0,"meta":{}}]
            projected,inspection=_project(root,chart,p)
            exit_idx=_idx(chart,"2026-08-26")
            check(f"exit_truth_wins_over_shadow_{signal}", 0, projected["trading_lifecycle_by_index"][exit_idx]["position_qty"])
            for date in ("2026-08-27", "2026-09-10", "2026-09-18"):
                idx=_idx(chart,date)
                check_true(f"consumed_or_held_signal_never_revives_{signal}_{date}", idx not in projected["trading_lifecycle_by_index"])
                check_true(f"no_ghost_lines_{signal}_{date}", all(np.isnan(projected[key][idx]) for key in ("entry_line","stop_line","tp_line","shadow_entry_line","shadow_stop_line","shadow_tp_line","shadow_limit_line")))
            check_true(f"no_ghost_future_preview_{signal}", not projected["future_preview"])
            buy_annotations=[x for x in projected["signal_annotations"] if x.get("signal_type")=="buy"]
            check(f"only_legitimate_prefill_anchor_survives_{signal}", int(signal<"2026-08-03"), len(buy_annotations))
            # The same exclusion also governs the direct sidebar plan resolver.
            inspection["candidate"]={"ticker":"2330","signal_date":signal,"trade_date":"2026-09-18",
                "params_signature":build_portfolio_params_signature(p),
                "ensemble_member_key":"1", "ensemble_member_params_by_key":{"1":params_to_json_dict(p)},
                "execution_plan_seed":{"entry_type":"normal","limit_price":107.,"init_sl":70.,"target_price":180.,"entry_atr":None}, "proj_qty":1000}
            check(f"sidebar_does_not_resurrect_same_signal_{signal}",None,resolve_trading_single_stock_sidebar_state(inspection,"2026-09-18",last_date="2026-09-18"))
        fresh="2026-08-28"
        chart["strategy_prefill_lifecycle_by_index"]={**prefill("2026-08-18"),**prefill(fresh)}
        chart["signal_annotations"]=[{"x":_idx(chart,date),"date":date,"signal_type":"buy","anchor_price":107.,"meta":{}} for date in ("2026-08-18",fresh)]
        projected,_=_project(root,chart,p)
        check("genuine_post_exit_signal_retained","SIGNAL",projected["trading_lifecycle_by_index"][_idx(chart,fresh)]["state"])
        check("genuine_post_exit_shadow_retained","SHADOW",projected["trading_lifecycle_by_index"][_idx(chart,"2026-09-10")]["state"])
        check("genuine_new_anchor_only",[fresh],[x["date"] for x in projected["signal_annotations"] if x.get("signal_type")=="buy"])
        check_true("fresh_post_exit_preview_remains_available",bool(projected["future_preview"]))
        # No confirmed acquisition: preserve every historical unfilled row.
        empty={"ticker":"2330","account_events":[],"candidate":{},"entry_orders":[],"pending_entries":[]}
        unfilled=project_trading_single_stock_chart_payload(chart,empty,params=p)
        check_true("unfilled_history_is_not_blanket_hidden",_idx(chart,"2026-08-19") in unfilled["trading_lifecycle_by_index"])
    return results, summary


def validate_trading_pinned_history_input_contract_case(base_params):
    """Both managed sources, real constructor, latest/cropped display, old origin."""
    from core.params_io import params_to_json_dict
    from core.portfolio_param_runtime import build_portfolio_params_signature
    from services.trading.strategy_param_runtime import build_trading_candidate_strategy_lineage
    from services.trading.account_state import record_strategy_trading_buy
    from core.trading_position_projection import build_confirmed_position_origin
    results,summary,check,check_true=bind_synthetic_case("TRADING_PINNED_HISTORY_INPUT","trading_history")
    p=_params(base_params); frame=_frame();chart=_chart(frame)
    for source in ("strategy", "manual"):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);account=initialize_trading_account_state(root,cash=1000000.)
            if source=="manual":
                account,lineage=_acquire(root,account,p)
            else:
                seed={"entry_type":"normal","trade_date":"2026-07-31","limit_price":102.,"init_sl":96.,"init_trail":98.,"target_price":130.,"entry_atr":2.0,"qty":1000}
                candidate={"ticker":"2330","kind":"buy","trade_date":"2026-07-31","signal_date":"2026-07-31","execution_plan_seed":seed,
                           "proj_qty":1000,"proj_cost":102000.,"params_signature":build_portfolio_params_signature(p),"ensemble_member_key":"1","ensemble_member_params_by_key":{"1":params_to_json_dict(p)}}
                lineage=build_trading_candidate_strategy_lineage(candidate)
                account=record_strategy_trading_buy(root,ticker="2330",qty=1000,price=100.,trade_date="2026-08-03",expected_revision=account["revision"],params=p,execution_plan_seed=seed,strategy_lineage=lineage)
            with patch("services.trading.position_market_context.load_trading_position_market_frame",side_effect=OSError("Redundant read forbidden")) as reader:
                full,inspection=_project(root,chart,replace(p,atr_len=2,atr_times_trail=9.))
                check(f"{source}_uses_already_verified_consumer_frame",0,reader.call_count)
            for date in ("2026-08-03","2026-08-25","2026-09-10"):
                idx=_idx(chart,date);row=full["trading_lifecycle_by_index"][idx]
                check_true(f"{source}_original_stop_tp_lines_present_{date}",all(np.isfinite(full[k][idx]) for k in ("entry_line","stop_line","tp_line")))
                check(f"{source}_sidebar_stop_equals_line_{date}",row["stop_price"],full["stop_line"][idx])
                check_true(f"{source}_valid_management_not_turned_into_error_{date}",not row.get("management_projection_error"))
            cropped=_chart(frame.loc["2026-09-08":])
            visible=project_trading_single_stock_chart_payload(cropped,inspection,params=replace(p,atr_len=2,atr_times_trail=9.))
            check(f"{source}_cropped_window_does_not_reinitialize_trail",full["stop_line"][_idx(chart,"2026-09-10")],visible["stop_line"][_idx(cropped,"2026-09-10")])
            # Legacy record lacks newly introduced origin fields, but the
            # immutable BUY state is already sufficient evidence for replay.
            record=deepcopy(account["positions"]["2330"])
            record["strategy_management"].pop("entry_position_state",None)
            record["strategy_management"].pop("entry_execution_plan",None)
            record["strategy_management"]["position_state"]["sl"]=999.0
            record["strategy_management"]["position_state"]["sl_milli"]=999000
            origin=build_confirmed_position_origin(record,binding=inspection["position_binding"],params=p,account_events=inspection["account_events"],frame=frame)
            check_true(f"{source}_legacy_origin_uses_initial_event_not_latest_state",origin["sl"]<100.0)
            old=deepcopy(inspection)
            old["management_projection"]=None
            for cycle in old["position_history"]:
                cycle.pop("projection",None);cycle["projection_error"]="source reload failed"
            recovered=project_trading_single_stock_chart_payload(chart,old,params=p)
            check_true(f"{source}_recoverable_source_error_does_not_erase_valid_lines",np.allclose(full["stop_line"],recovered["stop_line"],equal_nan=True))
    return results,summary


def validate_trading_direct_fill_source_contract_case(base_params):
    """Real snapshot validation, account mutation, corrected history and UI handler.

    AI: Only external market/Params providers are injected. Source discovery,
    currentness checks, frozen bindings, fill validation and ledgers are real.
    """
    from decimal import Decimal
    from core.file_integrity import atomic_write_json
    from core.trading_policy import get_trading_strategy_profile
    from services.trading import scanner_state, account_trade_entry as entry
    from services.trading.account_dashboard import _build_transaction_details
    from services.trading.strategy_param_runtime import build_trading_candidate_strategy_lineage
    from services.workbench_ui import trading_account_panel as ui
    from services.workbench_ui.trading_source_labels import trading_source_display_label
    from .synthetic_trading_cases import _build_synthetic_candidate_snapshot_payload

    results, summary, check, check_true = bind_synthetic_case("TRADING_DIRECT_FILL_SOURCE", "trading_history")
    p = _params(base_params); frame = _frame()
    current = replace(p, atr_len=2, atr_times_init=8., atr_times_trail=8.)
    signature = build_portfolio_params_signature(p)
    runtime = {
        "params": current, "latest_data_date": "2026-09-18", "param_latest_data_date": "2026-07-01",
        "profile": get_trading_strategy_profile(), "member_count": 1, "min_agree": 1, "param_min_agree": 1,
        "selected_params_sha256": "isolated-selected-params", "market_data_consumer_state_sha256": "isolated-consumer",
        "market_data_source_view_fingerprint": "isolated-view", "param_binding_sha256": "isolated-binding",
        "current_universe_tickers": ["2330", "2454"],
        "param_members": [{"member_key": "1", "member_index": 1, "params_obj": p, "params_signature": signature}],
    }
    seed = {"ticker": "2330", "trade_date": "2026-09-18", "entry_type": "normal", "entry_atr": 2.,
            "init_sl": 96., "init_trail": 98., "limit_price": 102., "target_price": 130., "qty": 1000}
    raw = {"ticker": "2330", "trade_date": "2026-09-18", "signal_date": "2026-07-31", "kind": "extended",
           "execution_plan_seed": seed, "limit_price": 102., "proj_qty": 1000, "proj_cost": 102000., "rank": 9}

    class Market:
        def read_dataset_frame(self, dataset, **kwargs):
            day = pd.Timestamp(kwargs["start_date"])
            if day not in frame.index:
                return pd.DataFrame()
            row = frame.loc[day]
            return pd.DataFrame([{"date": day.strftime("%Y-%m-%d"), "stock_id": kwargs["data_id"],
                                  "max": row["High"], "min": row["Low"], "Trading_Volume": row["Volume"]}])

    with tempfile.TemporaryDirectory() as td, ExitStack() as stack:
        roots = {key: Path(td) / key for key in ("typed", "selected", "custom", "guards", "ui")}
        for root in roots.values():
            initialize_trading_account_state(root, cash=1000000.)
        stack.enter_context(patch.object(scanner_state, "load_trading_scanner_runtime", return_value=runtime))
        stack.enter_context(patch.object(entry, "load_trading_scanner_runtime", return_value=runtime))
        stack.enter_context(patch("services.trading.actual_fill_validation.TradingMarketDataV2View.open", return_value=Market()))
        stack.enter_context(patch("services.trading.entry_lifecycle_context.resolve_trading_lifecycle_context",
                                  return_value=SimpleNamespace(view=Market(), finalized_date="2026-09-18")))
        stack.enter_context(patch("services.trading.entry_lifecycle_context.load_trading_v2_sanitized_ohlcv_frame",
                                  side_effect=lambda view, **kw: frame.loc[:kw["through_date"]].copy()))
        stack.enter_context(patch.object(entry, "open_trading_v2_consumer_view", return_value=Market()))
        stack.enter_context(patch.object(entry, "load_trading_v2_sanitized_ohlcv_frame",
                                  side_effect=lambda view, **kw: frame.loc[:kw["through_date"]].copy()))
        snapshot = _build_synthetic_candidate_snapshot_payload(roots["typed"], candidate_rows=[raw])
        candidate = snapshot["candidate_rows"][0]
        for root in roots.values():
            path = scanner_state.resolve_trading_candidate_snapshot_path(root)
            atomic_write_json(path, snapshot)
        kwargs = dict(ticker="2330", qty=600, price=Decimal("100"), trade_date="2026-08-03")
        expected_lineage = build_trading_candidate_strategy_lineage(candidate)
        preview = entry.preview_trading_account_buy(roots["typed"], **kwargs)
        check("typed_scanner_ticker_discovers_strategy_without_selection", "scanner_strategy_buy", preview["route"])
        check("typed_preview_reuses_exact_candidate", _digest(candidate), _digest(preview["candidate"]))
        check("strategy_discovery_does_not_construct_custom_context", None, preview["manual_management"])
        outcomes = {}
        for key, reference in (("typed", None), ("selected", candidate)):
            outcomes[key] = entry.record_trading_account_buy(roots[key], **kwargs, candidate=reference,
                                                           expected_route="scanner_strategy_buy")
            record = outcomes[key]["account"]["positions"]["2330"]
            check(key + "_source", "strategy_fill", record["source"])
            check(key + "_visible_source", "策略", trading_source_display_label(source=record["source"]))
            check(key + "_frozen_lineage", expected_lineage["lineage_id"], record["strategy_lineage"]["lineage_id"])
            check(key + "_frozen_params_not_current", p.atr_len, record["strategy_lineage"]["frozen_params"]["atr_len"])
            check(key + "_lower_user_quantity_kept", 600, record["broker"]["qty"])
            check(key + "_reference_quantity_unchanged", 1000, record["strategy_lineage"]["planned_qty"])
            buy, sell = _build_transaction_details(outcomes[key]["account"])
            check(key + "_buy_details_source", "策略", trading_source_display_label(source=buy[0]["source"]))
        check("typed_and_selected_economics_identical", outcomes["selected"]["account"]["cash_milli"],
              outcomes["typed"]["account"]["cash_milli"])
        check("typed_and_selected_management_identical", _digest(outcomes["selected"]["account"]["positions"]["2330"]["strategy_management"]),
              _digest(outcomes["typed"]["account"]["positions"]["2330"]["strategy_management"]))
        account = outcomes["typed"]["account"]
        buy_revision = account["revision"]
        account = entry.correct_trading_account_transaction(roots["typed"], transaction_revision=buy_revision,
                    qty=400, price=100., trade_date="2026-08-03", expected_account_revision=account["revision"])
        check("quantity_correction_keeps_strategy_source", "strategy_fill", account["positions"]["2330"]["source"])
        check("quantity_correction_keeps_lineage", expected_lineage["lineage_id"], account["positions"]["2330"]["strategy_lineage"]["lineage_id"])
        account = record_manual_trading_sell(roots["typed"], ticker="2330", qty=400, price=107., trade_date="2026-08-26",
                                             expected_revision=account["revision"])
        buys, sells = _build_transaction_details(account)
        check("manual_sell_does_not_reclassify_strategy_source", "策略", trading_source_display_label(source=sells[0]["source"]))
        check("sell_reason_independent_from_stock_source", "MANUAL_ACCOUNT_SELL", sells[0]["execution_reason"])
        payload, inspection = _project(roots["typed"], _chart(frame), current)
        check("backfilled_strategy_manual_exit_label", 1, len(payload["marker_groups"].get(MANUAL_SELL, [])))
        check_true("backfilled_strategy_keeps_management_lines", all(np.isfinite(payload[k][_idx(payload, "2026-08-25")])
                                                                   for k in ("entry_line", "stop_line", "tp_line")))
        check("backfilled_strategy_exit_quantity_zero", 0, payload["trading_lifecycle_by_index"][_idx(payload, "2026-08-26")]["position_qty"])
        check("backfilled_strategy_no_post_exit_shadow", False, _idx(payload, "2026-09-10") in payload["trading_lifecycle_by_index"])

        # A genuinely custom ticker remains custom, with the same managed engine.
        custom = entry.record_trading_account_buy(roots["custom"], **dict(kwargs, ticker="2454"), expected_route="manual_managed_buy")
        check("non_candidate_stays_custom", "manual_managed", custom["account"]["positions"]["2454"]["source"])
        check("non_candidate_visible_source", "自選", trading_source_display_label(source=custom["account"]["positions"]["2454"]["source"]))
        custom = record_manual_trading_sell(roots["custom"], ticker="2454", qty=600, price=107., trade_date="2026-08-26",
                                           expected_revision=custom["account"]["revision"])
        check("custom_sell_stays_custom", "自選", trading_source_display_label(source=_build_transaction_details(custom)[1][0]["source"]))

        # Reject instead of relabel when evidence or accepted choices change.
        guard = roots["guards"]; before = load_trading_account_state(guard)["revision"]
        for label, overrides in (("quantity_cap", {"qty": 1001}), ("fractional_quantity", {"qty": 1.5}),
                                 ("signal_day", {"trade_date": "2026-07-31"})):
            try:
                entry.record_trading_account_buy(guard, **dict(kwargs, **overrides))
            except (ValueError, RuntimeError):
                rejected = True
            else:
                rejected = False
            check(label + "_rejected_not_custom", True, rejected)
        changed = dict(candidate, proj_qty=900)
        for label, ref in (("stale_candidate", changed), ("wrong_ticker", dict(candidate, ticker="2454"))):
            try:
                entry.record_trading_account_buy(guard, **kwargs, candidate=ref)
            except (ValueError, RuntimeError):
                rejected = True
            else:
                rejected = False
            check(label + "_rejected_not_custom", True, rejected)
        with patch.object(scanner_state, "load_trading_scanner_runtime", return_value=dict(runtime, selected_params_sha256="changed")):
            try:
                entry.record_trading_account_buy(guard, **kwargs)
            except RuntimeError:
                rejected = True
            else:
                rejected = False
            check("stale_snapshot_not_silently_custom", True, rejected)
        try:
            entry.record_trading_account_buy(guard, **kwargs, expected_route="manual_managed_buy")
        except RuntimeError:
            rejected = True
        else:
            rejected = False
        check("preview_source_change_requires_reconfirmation", True, rejected)
        check("rejected_operations_do_not_write_ledger", before, load_trading_account_state(guard)["revision"])

        # Execute the real UI handler with no selected row, then inspect its
        # queued command after confirmation. No UI-only source guessing.
        queued = []; confirmations = []; errors = []
        harness = SimpleNamespace(_simple_trade_values=lambda: ("2330",600,Decimal("100"),"2026-08-03"),
                                  _selected_candidate_row=lambda: None,
                                  _submit_trading_command=lambda title, worker, **kw: queued.append(worker))
        with patch.object(ui, "WORKBENCH_PROJECT_ROOT", roots["ui"]), \
             patch.object(ui.messagebox, "askyesno", side_effect=lambda title, text, **kw: confirmations.append(text) or True), \
             patch.object(ui.messagebox, "showerror", side_effect=lambda title, text, **kw: errors.append(text)):
            ui.TradingAccountPanel._record_simple_trade(harness, "BUY")
            check("ui_no_selection_previews_without_error", [], errors)
            check("ui_confirmation_labels_strategy", True, bool(confirmations and "來源：策略" in confirmations[0]))
            result = queued[0]()
            check("ui_worker_preserves_preview_strategy", "strategy_fill", result["account"]["positions"]["2330"]["source"])
        check("scanner_evidence_unchanged_by_account_mutations", _digest(snapshot), _digest(scanner_state.load_trading_candidate_snapshot(guard)))
    return results, summary

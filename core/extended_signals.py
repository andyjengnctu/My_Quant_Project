import numpy as np
import pandas as pd


from core.entry_plans import (
    build_counterfactual_shadow_position_from_plan,
    execute_pre_market_entry_plan,
    resize_candidate_plan_to_capital,
)
from core.price_utils import (
    calc_frozen_target_price,
    calc_initial_stop_from_reference,
    calc_initial_trailing_stop_from_reference,
    is_limit_buy_price_reachable_for_day,
    price_to_milli,
)


def clone_shadow_position(shadow_position):
    """Clone mutable shadow-trade state without recursively copying read-only metadata.

    Shadow positions are flat accounting states.  The only mutable nested runtime
    payload is ``_last_exec_contexts``; clone that list explicitly so callers
    retain isolation without paying the recursive ``deepcopy`` cost in the daily
    replay hot path.
    """
    if shadow_position is None:
        return None
    cloned_position = dict(shadow_position)
    exec_contexts = shadow_position.get("_last_exec_contexts")
    if exec_contexts is not None:
        cloned_position["_last_exec_contexts"] = [dict(context) for context in exec_contexts]
    return cloned_position


def _resolve_shadow_position(signal_state):
    if signal_state is None:
        return None
    shadow_position = signal_state.get("shadow_position")
    if shadow_position is None:
        return None
    if int(shadow_position.get("qty", 0) or 0) <= 0:
        return None
    return shadow_position


def _has_live_shadow_position(signal_state):
    return _resolve_shadow_position(signal_state) is not None


def resolve_extended_signal_order_limit(signal_state):
    if signal_state is None:
        return np.nan
    return signal_state.get("orig_limit", np.nan)


def resolve_extended_signal_effective_limit(signal_state):
    return resolve_extended_signal_order_limit(signal_state)


def resolve_signal_tracking_params(signal_state, fallback_params):
    if isinstance(signal_state, dict):
        return signal_state.get("_params_obj") or fallback_params
    return fallback_params


def normalize_breakout_quality_rank_payload(quality_rank):
    """Normalize one breakout-event ranking payload without changing availability semantics."""

    if not isinstance(quality_rank, dict):
        raise TypeError("breakout quality rank 必須是 dict")

    normalized = dict(quality_rank)
    available = bool(quality_rank.get("available", False))
    score_date = str(quality_rank.get("score_date") or "").strip()
    score_source = str(quality_rank.get("score_source") or "canonical_runtime").strip()
    if not score_date:
        raise ValueError("breakout quality rank 必須保存原始 breakout score_date")
    if not score_source:
        raise ValueError("breakout quality rank 必須保存 score_source")

    if available:
        try:
            score = float(quality_rank.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError("可評分 breakout quality rank score 必須是 0~1 有限值") from exc
        if not np.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError("可評分 breakout quality rank score 必須是 0~1 有限值")
        unavailable_reason = ""
    else:
        score = None
        unavailable_reason = str(
            quality_rank.get("unavailable_reason") or "unavailable"
        ).strip()

    normalized.update({
        "score": score,
        "available": available,
        "unavailable_reason": unavailable_reason,
        "score_date": score_date,
        "score_source": score_source,
        "shared_group_score": bool(quality_rank.get("shared_group_score", False)),
        "filter_id": str(quality_rank.get("filter_id") or "").strip(),
    })
    return normalized


def attach_breakout_quality_rank(signal_state, quality_rank):
    """Attach the original breakout-event ranking state to continuation/re-entry state."""

    if signal_state is None:
        return None
    if quality_rank is None:
        signal_state.pop("breakout_quality_rank", None)
        return signal_state

    # (AI註: PIT缺分不是REJECT；延續／re-entry需保存原事件的不可評分狀態，
    #       後續候選才能一致回退原buy-sort，而不是重新查詢或中止replay。)
    signal_state["breakout_quality_rank"] = normalize_breakout_quality_rank_payload(
        quality_rank
    )
    return signal_state


def resolve_breakout_quality_rank(signal_state):
    if not isinstance(signal_state, dict):
        return None
    rank = signal_state.get("breakout_quality_rank")
    if rank is None:
        return None
    if not isinstance(rank, dict):
        raise TypeError("signal_state.breakout_quality_rank 必須是 dict")
    holder = {}
    attach_breakout_quality_rank(holder, rank)
    return dict(holder["breakout_quality_rank"])


def _did_extended_signal_touch_barrier(signal_state, *, day_low, day_high):
    if signal_state is None:
        return False

    shadow_position = _resolve_shadow_position(signal_state)
    if shadow_position is not None:
        if shadow_position.get("pending_exit_action") in {"STOP", "TP_HALF"}:
            return True
        if bool(shadow_position.get("sold_half", False)):
            return True

    invalidation_barrier = signal_state.get("continuation_invalidation_barrier", np.nan)
    completion_barrier = signal_state.get("continuation_completion_barrier", np.nan)
    stop_hit = (
        not pd.isna(day_low)
        and not pd.isna(invalidation_barrier)
        and price_to_milli(day_low) <= price_to_milli(invalidation_barrier)
    )
    target_hit = (
        not pd.isna(day_high)
        and not pd.isna(completion_barrier)
        and price_to_milli(day_high) >= price_to_milli(completion_barrier)
    )
    return stop_hit or target_hit


# # (AI註: scanner/單股顯示層：是否列為 extended_tbd 只看今日 Low 是否到原始買入限價；shadow_price 僅負責管理線)
def is_extended_tbd_display_day(signal_state, day_low):
    if signal_state is None or pd.isna(day_low):
        return False
    order_limit = resolve_extended_signal_order_limit(signal_state)
    if pd.isna(order_limit):
        return False
    return float(day_low) <= float(order_limit)


# # (AI註: 單一真理來源 - 延續候選原始訊號狀態統一由此建立；共用同一份 shadow trade 狀態，不再拆 extended / TBD 雙核心)
def create_signal_tracking_state(original_limit, atr, params, ticker=None, security_profile=None, signal_date=None):
    if pd.isna(original_limit) or pd.isna(atr):
        return None

    return {
        "orig_limit": original_limit,
        "orig_atr": atr,
        "entry_ref_price": np.nan,
        "continuation_invalidation_barrier": np.nan,
        "continuation_completion_barrier": np.nan,
        "shadow_position": None,
        "ticker": ticker,
        "security_profile": security_profile,
        "signal_date": signal_date,
        "_params_obj": params,
    }


def _sync_signal_shadow_fields(signal_state, shadow_position, *, copy_shadow_position=True):
    if signal_state is None:
        return None
    if shadow_position is None or int(shadow_position.get("qty", 0) or 0) <= 0:
        signal_state["shadow_position"] = None
        return signal_state

    signal_state["shadow_position"] = clone_shadow_position(shadow_position) if copy_shadow_position else shadow_position
    signal_state["entry_ref_price"] = shadow_position.get("entry_fill_price", np.nan)
    signal_state["continuation_invalidation_barrier"] = shadow_position.get("sl", np.nan)
    signal_state["continuation_completion_barrier"] = shadow_position.get("tp_half", np.nan)
    return signal_state


# # (AI註: 單一真理來源 - 進入 extended 後即啟動共同 shadow trade；不再把 shadow existence 綁在 low<=limit，避免 extended 在 low>limit 時無法判定失效)
def ensure_extended_signal_counterfactual_anchor(
    signal_state,
    *,
    t_open,
    t_high=np.nan,
    t_low,
    t_close=np.nan,
    t_volume=np.nan,
    y_close=np.nan,
    sizing_capital=None,
    params,
    current_date=None,
    copy_shadow_position=True,
):
    if signal_state is None or params is None or sizing_capital is None:
        return signal_state
    if _has_live_shadow_position(signal_state):
        return signal_state

    original_limit = signal_state.get("orig_limit", np.nan)
    original_atr = signal_state.get("orig_atr", np.nan)
    if pd.isna(original_limit) or pd.isna(original_atr) or pd.isna(t_open):
        return signal_state

    candidate_plan = build_extended_candidate_plan_from_signal(
        signal_state,
        sizing_capital,
        params,
        ticker=signal_state.get("ticker"),
        security_profile=signal_state.get("security_profile"),
        trade_date=current_date,
    )
    if candidate_plan is None or candidate_plan.get("qty", 0) <= 0:
        return signal_state

    shadow_position = build_counterfactual_shadow_position_from_plan(
        candidate_plan,
        t_open=t_open,
        t_high=t_high,
        t_low=t_low,
        params=params,
        ticker=signal_state.get("ticker"),
        security_profile=signal_state.get("security_profile"),
        trade_date=current_date,
    )
    if shadow_position is None:
        return signal_state

    return _sync_signal_shadow_fields(signal_state, shadow_position, copy_shadow_position=copy_shadow_position)


# # (AI註: 單一真理來源 - 延續 shadow trade 每日只用同一套 execute_bar_step 推進，不可再疊另一套 barrier 規則)
def update_extended_tbd_shadow_trade_for_bar(
    tbd_state,
    *,
    y_atr,
    y_ind_sell,
    y_close,
    y_high,
    t_open,
    t_high,
    t_low,
    t_close,
    t_volume,
    params,
    current_date=None,
    copy_shadow_position=True,
):
    if tbd_state is None:
        return None

    from core.position_step import execute_bar_step

    resolved_shadow_position = _resolve_shadow_position(tbd_state)
    shadow_position = clone_shadow_position(resolved_shadow_position) if copy_shadow_position else resolved_shadow_position
    if shadow_position is None:
        return tbd_state

    shadow_position, _freed_cash, _pnl_realized, _events = execute_bar_step(
        shadow_position,
        y_atr,
        y_ind_sell,
        y_close,
        t_open,
        t_high,
        t_low,
        t_close,
        t_volume,
        params,
        current_date=current_date,
        y_high=y_high,
    )
    if int(shadow_position.get("qty", 0) or 0) <= 0:
        return None

    return _sync_signal_shadow_fields(tbd_state, shadow_position, copy_shadow_position=copy_shadow_position)


# # (AI註: 單一真理來源 - 延續候選盤後狀態推進與失效統一由此處理；若啟動當日已跌破影子初始停損，則當日盤後直接失效)
def should_clear_extended_signal(
    signal_state,
    t_low,
    t_high=None,
    *,
    t_open=np.nan,
    t_close=np.nan,
    t_volume=np.nan,
    y_close=np.nan,
    y_high=np.nan,
    y_atr=np.nan,
    y_ind_sell=False,
    sizing_capital=None,
    current_date=None,
    params=None,
    copy_shadow_position=True,
):
    if signal_state is None:
        return False

    if _has_live_shadow_position(signal_state):
        updated_state = update_extended_tbd_shadow_trade_for_bar(
            signal_state,
            y_atr=y_atr,
            y_ind_sell=y_ind_sell,
            y_close=y_close,
            y_high=y_high,
            t_open=t_open,
            t_high=t_high,
            t_low=t_low,
            t_close=t_close,
            t_volume=t_volume,
            params=params,
            current_date=current_date,
            copy_shadow_position=copy_shadow_position,
        )
        if updated_state is None:
            signal_state["shadow_position"] = None
            return True
        return _did_extended_signal_touch_barrier(updated_state, day_low=t_low, day_high=t_high)

    ensure_extended_signal_counterfactual_anchor(
        signal_state,
        t_open=t_open,
        t_high=t_high,
        t_low=t_low,
        t_close=t_close,
        t_volume=t_volume,
        y_close=y_close,
        sizing_capital=sizing_capital,
        params=params,
        current_date=current_date,
        copy_shadow_position=copy_shadow_position,
    )
    shadow_position = _resolve_shadow_position(signal_state)
    if shadow_position is None:
        return False

    return _did_extended_signal_touch_barrier(signal_state, day_low=t_low, day_high=t_high)


# # (AI註: 單一真理來源 - 延續候選掛單價固定使用原始 limit；shadow_price 只繼承管理線 / 失效 / 達標狀態)
def build_extended_candidate_plan_from_signal(signal_state, sizing_capital, params, ticker=None, security_profile=None, trade_date=None):
    if signal_state is None:
        return None

    resolved_ticker = ticker or signal_state.get("ticker")
    resolved_security_profile = security_profile or signal_state.get("security_profile")
    entry_atr = signal_state.get("orig_atr", np.nan)
    if pd.isna(entry_atr):
        return None

    orig_limit = signal_state.get("orig_limit", np.nan)
    shadow_position = _resolve_shadow_position(signal_state)
    if shadow_position is not None:
        limit_reference = resolve_extended_signal_order_limit(signal_state)
        sizing_stop_ref = shadow_position.get("sl", np.nan)
        sizing_trail_ref = shadow_position.get("trailing_stop", np.nan)
        target_price = shadow_position.get("tp_half", np.nan)
    else:
        limit_reference = orig_limit
        sizing_stop_ref = calc_initial_stop_from_reference(limit_reference, entry_atr, params, ticker=resolved_ticker, security_profile=resolved_security_profile)
        sizing_trail_ref = calc_initial_trailing_stop_from_reference(limit_reference, entry_atr, params, ticker=resolved_ticker, security_profile=resolved_security_profile)
        target_price = calc_frozen_target_price(limit_reference, sizing_stop_ref, ticker=resolved_ticker, security_profile=resolved_security_profile)

    if pd.isna(limit_reference) or pd.isna(sizing_stop_ref) or pd.isna(sizing_trail_ref) or pd.isna(target_price):
        return None

    base_plan = {
        "limit_price": limit_reference,
        "init_sl": sizing_stop_ref,
        "init_trail": sizing_trail_ref,
        "orig_limit": orig_limit,
        "orig_atr": entry_atr,
        "target_price": target_price,
        "entry_atr": entry_atr,
        "entry_ref_price": signal_state.get("entry_ref_price", np.nan),
        "continuation_invalidation_barrier": sizing_stop_ref,
        "continuation_completion_barrier": target_price,
        "ticker": resolved_ticker,
        "security_profile": resolved_security_profile,
        "trade_date": trade_date,
        "signal_date": signal_state.get("signal_date"),
        "max_qty": signal_state.get("max_qty"),
    }
    if shadow_position is not None:
        base_plan["shadow_position_state"] = clone_shadow_position(shadow_position)
    return resize_candidate_plan_to_capital(base_plan, sizing_capital, params)


# # (AI註: 單一真理來源 - TBD 只是同一份 shadow trade 的顯示視圖，不再另建第二套狀態機)
def create_extended_tbd_tracking_state(signal_state, shadow_position):
    if signal_state is None:
        return None
    cloned_state = dict(signal_state)
    return _sync_signal_shadow_fields(cloned_state, shadow_position)


def _resolve_tbd_signal_state(tbd_state):
    return tbd_state


def _resolve_tbd_shadow_position(tbd_state):
    return _resolve_shadow_position(tbd_state)


def is_extended_tbd_shadow_alive(tbd_state):
    return _has_live_shadow_position(tbd_state)


def build_extended_tbd_candidate_plan_from_state(tbd_state, sizing_capital, params, ticker=None, security_profile=None, trade_date=None):
    if tbd_state is None or not is_extended_tbd_shadow_alive(tbd_state):
        return None
    shadow_position = _resolve_tbd_shadow_position(tbd_state)
    if shadow_position is None:
        return None

    candidate_plan = build_extended_candidate_plan_from_signal(
        tbd_state,
        sizing_capital,
        params,
        ticker=ticker or tbd_state.get("ticker"),
        security_profile=security_profile or tbd_state.get("security_profile"),
        trade_date=trade_date,
    )
    if candidate_plan is None:
        return None

    enriched_plan = dict(candidate_plan)
    enriched_plan["shadow_sl"] = shadow_position.get("sl", np.nan)
    enriched_plan["shadow_initial_stop"] = shadow_position.get("initial_stop", np.nan)
    enriched_plan["shadow_trailing_stop"] = shadow_position.get("trailing_stop", np.nan)
    enriched_plan["shadow_tp_half"] = shadow_position.get("tp_half", np.nan)
    enriched_plan["shadow_sold_half"] = bool(shadow_position.get("sold_half", False))
    enriched_plan["shadow_entry_price"] = shadow_position.get("entry_fill_price", np.nan)
    enriched_plan["shadow_pending_exit_action"] = shadow_position.get("pending_exit_action")
    enriched_plan["shadow_pending_exit_trigger_price"] = shadow_position.get("pending_exit_trigger_price", np.nan)
    enriched_plan["shadow_position_state"] = clone_shadow_position(shadow_position)
    return enriched_plan


# # (AI註: 單一真理來源 - 延續訊號今日能否掛單；若 shadow 已進入待出場狀態，則不得再掛)
def has_extended_signal_orderable_context_for_day(signal_state, candidate_plan, y_close, ticker=None, security_profile=None):
    if signal_state is None or candidate_plan is None:
        return False
    shadow_position = _resolve_shadow_position(signal_state)
    if shadow_position is not None and shadow_position.get("pending_exit_action") is not None:
        return False
    resolved_ticker = ticker or candidate_plan.get("ticker") or signal_state.get("ticker")
    resolved_security_profile = security_profile or candidate_plan.get("security_profile") or signal_state.get("security_profile")
    return is_limit_buy_price_reachable_for_day(candidate_plan.get("limit_price"), y_close, ticker=resolved_ticker, security_profile=resolved_security_profile)


def is_extended_signal_orderable_for_day(signal_state, candidate_plan, y_close, ticker=None, security_profile=None):
    if not has_extended_signal_orderable_context_for_day(signal_state, candidate_plan, y_close, ticker=ticker, security_profile=security_profile):
        return False
    return candidate_plan.get("qty", 0) > 0


def has_extended_tbd_orderable_context_for_day(tbd_state, candidate_plan, y_close, ticker=None, security_profile=None):
    if tbd_state is None or not is_extended_tbd_shadow_alive(tbd_state):
        return False
    signal_state = _resolve_tbd_signal_state(tbd_state)
    return has_extended_signal_orderable_context_for_day(
        signal_state,
        candidate_plan,
        y_close,
        ticker=ticker or tbd_state.get("ticker"),
        security_profile=security_profile or tbd_state.get("security_profile"),
    )


def is_extended_tbd_orderable_for_day(tbd_state, candidate_plan, y_close, ticker=None, security_profile=None):
    if not has_extended_tbd_orderable_context_for_day(tbd_state, candidate_plan, y_close, ticker=ticker, security_profile=security_profile):
        return False
    return candidate_plan.get("qty", 0) > 0


# # (AI註: 單一真理來源 - 延續單實際掛單規格；若已啟動 shadow，實際成交後直接繼承 shadow 管理狀態)
def build_extended_entry_plan_from_signal(signal_state, sizing_capital, params, *, y_close, ticker=None, security_profile=None, trade_date=None):
    candidate_plan = build_extended_candidate_plan_from_signal(signal_state, sizing_capital, params, ticker=ticker, security_profile=security_profile, trade_date=trade_date)
    if not is_extended_signal_orderable_for_day(signal_state, candidate_plan, y_close, ticker=ticker, security_profile=security_profile):
        return None
    return candidate_plan


# # (AI註: 延續候選資格評估)
def evaluate_extended_candidate_eligibility(original_limit, atr, sizing_capital, params, ticker=None, security_profile=None, trade_date=None, signal_date=None):
    signal_state = create_signal_tracking_state(original_limit, atr, params, ticker=ticker, security_profile=security_profile, signal_date=signal_date)
    return build_extended_candidate_plan_from_signal(signal_state, sizing_capital, params, ticker=ticker, security_profile=security_profile, trade_date=trade_date)

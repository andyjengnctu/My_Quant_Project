import copy
import os
import threading
from collections import OrderedDict
from tools.optimizer.callbacks import run_optimizer_monitoring_callback
from tools.optimizer.objective import run_optimizer_objective
from config.training_performance_policy import resolve_optimizer_rolling_parallel_prep_cache_max_items_default
from tools.optimizer.trial_inputs import _build_process_pool_executor
from core.raw_universe_contract import coerce_raw_universe_required_min_rows
from core.portfolio_fast_data import get_fast_dates, pack_static_market_data
from core.signal_utils import OPTIMIZER_TRUE_RANGE_ATTR, tv_true_range


def close_study_storage(study):
    storage = getattr(study, "_storage", None)
    if storage is None:
        return
    remove_session = getattr(storage, "remove_session", None)
    if callable(remove_session):
        remove_session()
    backend = getattr(storage, "_backend", None)
    backend_remove_session = getattr(backend, "remove_session", None)
    if callable(backend_remove_session):
        backend_remove_session()
    engine = getattr(backend, "engine", None)
    dispose = getattr(engine, "dispose", None)
    if callable(dispose):
        dispose()


def _empty_local_min_dependency_stats():
    return {
        "dependency_signal_total": 0,
        "dependency_signal_evaluated": 0,
        "dependency_portfolio_total": 0,
        "dependency_portfolio_evaluated": 0,
        "dependency_mixed_total": 0,
        "dependency_mixed_evaluated": 0,
        "dependency_unknown_total": 0,
        "dependency_unknown_evaluated": 0,
        "signal_reuse_candidate_total": 0,
        "signal_reuse_candidate_evaluated": 0,
        "signal_recompute_required_total": 0,
        "signal_recompute_required_evaluated": 0,
        "dependency_field_total_counts": {},
        "dependency_field_evaluated_counts": {},
    }


def _merge_int_count_dict(target: dict, source) -> None:
    if not isinstance(target, dict) or not isinstance(source, dict):
        return
    for key, value in source.items():
        field_name = str(key or "unknown")
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            continue
        target[field_name] = int(target.get(field_name, 0) or 0) + int(count)


def _format_int_count_dict(counts: dict, *, max_items: int = 24) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    items = []
    for key, value in counts.items():
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            continue
        items.append((str(key), count))
    items.sort(key=lambda item: (-item[1], item[0]))
    if max_items is not None:
        items = items[:max(1, int(max_items))]
    return ";".join(f"{field}:{count}" for field, count in items)


class OptimizerSession:
    def __init__(
        self,
        *,
        output_dir,
        session_ts,
        profile_recorder_cls,
        build_optimizer_trial_params,
        get_best_completed_trial_or_none,
        objective_mode,
        search_train_end_year,
        walk_forward_policy,
        resolve_optimizer_tp_percent,
        print_strategy_dashboard,
        colors,
        optimizer_fixed_tp_percent,
        train_max_positions,
        train_start_year,
        train_enable_rotation,
        default_max_workers,
        enable_optimizer_profiling,
        enable_profile_console_print,
        profile_print_every_n_trials,
    ):
        self.raw_data_cache = {}
        self._cache_lock = threading.RLock()
        self._local_min_stats_lock = threading.RLock()
        self.raw_data_cache_data_dir = None
        self.current_session_trial = 0
        self.n_trials = 0
        self.prep_summary = {
            "trials_with_insufficient": 0,
            "insufficient_count_total": 0,
        }
        self.output_dir = output_dir
        self.session_ts = session_ts
        self.build_optimizer_trial_params = build_optimizer_trial_params
        self.get_best_completed_trial_or_none = get_best_completed_trial_or_none
        self.objective_mode = str(objective_mode)
        self.search_train_end_year = int(search_train_end_year)
        self.walk_forward_policy = dict(walk_forward_policy)
        self.resolve_optimizer_tp_percent = resolve_optimizer_tp_percent
        self.print_strategy_dashboard = print_strategy_dashboard
        self.colors = colors
        self.optimizer_fixed_tp_percent = optimizer_fixed_tp_percent
        self.train_max_positions = train_max_positions
        self.train_start_year = train_start_year
        self.train_enable_rotation = train_enable_rotation
        self.default_max_workers = default_max_workers
        self.profile_recorder = profile_recorder_cls(
            output_dir=output_dir,
            session_ts=session_ts,
            enabled=enable_optimizer_profiling,
            console_print=enable_profile_console_print,
            print_every_n_trials=profile_print_every_n_trials,
        )
        self._trial_prep_executor_bundle = None
        self._shared_trial_prep_executor_holder = None
        self.prep_executor_stats = {
            "created": 0,
            "reused": 0,
        }
        self._optimizer_trial_milestone_inputs = {}
        self._prepared_trial_input_cache = OrderedDict()
        self._prepared_trial_input_cache_max_items = self._resolve_prepared_trial_input_cache_max_items()
        self._prepared_trial_input_cache_is_shared = False
        self.prep_cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }
        self.prep_phase_stats = self._empty_prep_phase_stats()
        self.local_min_review_stats = {
            "total_neighbors": 0,
            "evaluated_neighbors": 0,
            "payload_score_cache_hits": 0,
            "prep_cache_prioritized": 0,
            "order_score_prioritized": 0,
            "field_order_score_prioritized": 0,
            "portfolio_dependency_deprioritized": 0,
            "signal_dependency_field_prioritized": 0,
            "early_stops": 0,
            "selection_prunes": 0,
            "parallel_workers_max": 0,
            "parallel_submitted": 0,
            "parallel_completed": 0,
            "parallel_cancelled": 0,
            "hard_fail_stops": 0,
            "hard_fail_neighbors_skipped": 0,
            "hard_fail_cancelled": 0,
            **_empty_local_min_dependency_stats(),
        }
        self.local_min_order_score_cache = {}
        self.local_min_field_order_score_cache = {}
        self.local_min_field_order_score_update_cache = {}
        self._shared_local_min_field_order_score_cache = None
        self._full_evaluation_cache = OrderedDict()
        self._full_evaluation_cache_max_items = self._resolve_full_evaluation_cache_max_items()
        self.static_fast_cache = {}
        self.raw_data_cache_required_min_rows = None
        self.master_dates = set()
        self.sorted_master_dates = []

    
    def _resolve_prepared_trial_input_cache_max_items(self):
        policy_default = resolve_optimizer_rolling_parallel_prep_cache_max_items_default()
        raw_value = os.environ.get("OPTIMIZER_PREP_CACHE_MAX_ITEMS")
        if raw_value is None or str(raw_value).strip() == "":
            raw_value = os.environ.get("OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS", str(policy_default))
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = int(policy_default)
        return max(0, min(256, value))

    def _resolve_full_evaluation_cache_max_items(self):
        raw_value = os.environ.get("OPTIMIZER_FULL_EVAL_CACHE_MAX_ITEMS", "512")
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = 512
        return max(0, min(4096, value))

    def attach_shared_prepared_trial_input_cache(self, cache, *, max_items=None):
        if cache is None:
            return
        self._prepared_trial_input_cache = cache
        self._prepared_trial_input_cache_is_shared = True
        if max_items is not None:
            try:
                resolved_max_items = int(max_items)
            except (TypeError, ValueError):
                resolved_max_items = self._prepared_trial_input_cache_max_items
            self._prepared_trial_input_cache_max_items = max(0, min(4096, resolved_max_items))
        with self._cache_lock:
            while len(self._prepared_trial_input_cache) > int(self._prepared_trial_input_cache_max_items):
                self._prepared_trial_input_cache.popitem(last=False)
                self.prep_cache_stats["evictions"] = int(self.prep_cache_stats.get("evictions", 0)) + 1

    def attach_shared_trial_prep_executor_holder(self, holder):
        if isinstance(holder, dict):
            self._shared_trial_prep_executor_holder = holder

    def attach_shared_local_min_order_score_cache(self, cache):
        if isinstance(cache, dict):
            self.local_min_order_score_cache = cache

    def attach_shared_local_min_field_order_score_cache(self, cache):
        if isinstance(cache, dict):
            # AI註: field-level order hints are deliberately fold-lagged.
            # Reading a snapshot prevents noisy hints learned earlier in the
            # same fold from reordering later candidates and hurting early-stop.
            self._shared_local_min_field_order_score_cache = cache
            self.local_min_field_order_score_cache = dict(cache)
            self.local_min_field_order_score_update_cache = {}

    def flush_shared_local_min_field_order_score_cache(self):
        shared_cache = getattr(self, "_shared_local_min_field_order_score_cache", None)
        update_cache = getattr(self, "local_min_field_order_score_update_cache", None)
        if not isinstance(shared_cache, dict) or not isinstance(update_cache, dict):
            return 0
        merged = 0
        for key, score in list(update_cache.items()):
            prior_score = shared_cache.get(key)
            try:
                should_update = prior_score is None or float(score) < float(prior_score)
            except (TypeError, ValueError):
                should_update = True
            if bool(should_update):
                shared_cache[key] = float(score)
                merged += 1
        update_cache.clear()
        return int(merged)

    @staticmethod
    def _empty_prep_phase_stats():
        return {
            "prep_call_count": 0,
            "prep_wall_sum_sec": 0.0,
            "prep_worker_total_sum_sec": 0.0,
            "prep_total_sum_sec": 0.0,
            "prep_copy_sum_sec": 0.0,
            "prep_generate_signals_sum_sec": 0.0,
            "prep_assign_sum_sec": 0.0,
            "prep_run_backtest_sum_sec": 0.0,
            "prep_to_dict_sum_sec": 0.0,
            "prep_static_pack_sum_sec": 0.0,
            "prep_executor_setup_sum_sec": 0.0,
            "prep_executor_submit_sum_sec": 0.0,
            "prep_executor_collect_sum_sec": 0.0,
            "prep_executor_shutdown_sum_sec": 0.0,
            "prep_merge_sum_sec": 0.0,
            "prep_master_union_sum_sec": 0.0,
            "prep_ticker_count_sum": 0,
            "prep_batch_count_sum": 0,
            "prep_ok_count_sum": 0,
            "prep_fail_count_sum": 0,
            "prep_feature_bank_hits": 0,
            "prep_feature_bank_misses": 0,
            "prep_feature_bank_size_max": 0,
            "prep_feature_bank_max_items": 0,
        }

    def _record_prepared_trial_inputs_profile(self, prep_result):
        if not isinstance(prep_result, dict):
            return
        profile = dict(prep_result.get("prep_profile") or {})
        stats = self.prep_phase_stats
        stats["prep_call_count"] = int(stats.get("prep_call_count", 0) or 0) + 1
        stats["prep_wall_sum_sec"] = float(stats.get("prep_wall_sum_sec", 0.0) or 0.0) + float(prep_result.get("prep_wall_sec", 0.0) or 0.0)
        stats["prep_worker_total_sum_sec"] = float(stats.get("prep_worker_total_sum_sec", 0.0) or 0.0) + float(profile.get("worker_total_sum_sec", 0.0) or 0.0)
        stats["prep_total_sum_sec"] = float(stats.get("prep_total_sum_sec", 0.0) or 0.0) + float(profile.get("prep_total_sum_sec", 0.0) or 0.0)
        stats["prep_copy_sum_sec"] = float(stats.get("prep_copy_sum_sec", 0.0) or 0.0) + float(profile.get("copy_sum_sec", 0.0) or 0.0)
        stats["prep_generate_signals_sum_sec"] = float(stats.get("prep_generate_signals_sum_sec", 0.0) or 0.0) + float(profile.get("generate_signals_sum_sec", 0.0) or 0.0)
        stats["prep_assign_sum_sec"] = float(stats.get("prep_assign_sum_sec", 0.0) or 0.0) + float(profile.get("assign_sum_sec", 0.0) or 0.0)
        stats["prep_run_backtest_sum_sec"] = float(stats.get("prep_run_backtest_sum_sec", 0.0) or 0.0) + float(profile.get("run_backtest_sum_sec", 0.0) or 0.0)
        stats["prep_to_dict_sum_sec"] = float(stats.get("prep_to_dict_sum_sec", 0.0) or 0.0) + float(profile.get("to_dict_sum_sec", 0.0) or 0.0)
        stats["prep_static_pack_sum_sec"] = float(stats.get("prep_static_pack_sum_sec", 0.0) or 0.0) + float(profile.get("static_pack_sec", 0.0) or 0.0)
        stats["prep_executor_setup_sum_sec"] = float(stats.get("prep_executor_setup_sum_sec", 0.0) or 0.0) + float(profile.get("executor_setup_sec", 0.0) or 0.0)
        stats["prep_executor_submit_sum_sec"] = float(stats.get("prep_executor_submit_sum_sec", 0.0) or 0.0) + float(profile.get("executor_submit_sec", 0.0) or 0.0)
        stats["prep_executor_collect_sum_sec"] = float(stats.get("prep_executor_collect_sum_sec", 0.0) or 0.0) + float(profile.get("executor_collect_sec", 0.0) or 0.0)
        stats["prep_executor_shutdown_sum_sec"] = float(stats.get("prep_executor_shutdown_sum_sec", 0.0) or 0.0) + float(profile.get("executor_shutdown_sec", 0.0) or 0.0)
        stats["prep_merge_sum_sec"] = float(stats.get("prep_merge_sum_sec", 0.0) or 0.0) + float(profile.get("merge_sum_sec", 0.0) or 0.0)
        stats["prep_master_union_sum_sec"] = float(stats.get("prep_master_union_sum_sec", 0.0) or 0.0) + float(profile.get("master_union_sec", 0.0) or 0.0)
        stats["prep_ticker_count_sum"] = int(stats.get("prep_ticker_count_sum", 0) or 0) + int(profile.get("ticker_count", 0) or 0)
        stats["prep_batch_count_sum"] = int(stats.get("prep_batch_count_sum", 0) or 0) + int(profile.get("batch_count", 0) or 0)
        stats["prep_ok_count_sum"] = int(stats.get("prep_ok_count_sum", 0) or 0) + int(profile.get("ok_count", 0) or 0)
        stats["prep_fail_count_sum"] = int(stats.get("prep_fail_count_sum", 0) or 0) + int(profile.get("fail_count", 0) or 0)
        stats["prep_feature_bank_hits"] = int(stats.get("prep_feature_bank_hits", 0) or 0) + int(profile.get("feature_bank_hits", 0) or 0)
        stats["prep_feature_bank_misses"] = int(stats.get("prep_feature_bank_misses", 0) or 0) + int(profile.get("feature_bank_misses", 0) or 0)
        stats["prep_feature_bank_size_max"] = max(int(stats.get("prep_feature_bank_size_max", 0) or 0), int(profile.get("feature_bank_size_max", 0) or 0))
        stats["prep_feature_bank_max_items"] = max(int(stats.get("prep_feature_bank_max_items", 0) or 0), int(profile.get("feature_bank_max_items", 0) or 0))

    def reset_prep_cache_stats(self):
        self.prep_cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }
        self.prep_phase_stats = self._empty_prep_phase_stats()
        self.local_min_review_stats = {
            "total_neighbors": 0,
            "evaluated_neighbors": 0,
            "payload_score_cache_hits": 0,
            "prep_cache_prioritized": 0,
            "order_score_prioritized": 0,
            "field_order_score_prioritized": 0,
            "portfolio_dependency_deprioritized": 0,
            "signal_dependency_field_prioritized": 0,
            "early_stops": 0,
            "selection_prunes": 0,
            "parallel_workers_max": 0,
            "parallel_submitted": 0,
            "parallel_completed": 0,
            "parallel_cancelled": 0,
            "hard_fail_stops": 0,
            "hard_fail_neighbors_skipped": 0,
            "hard_fail_cancelled": 0,
            **_empty_local_min_dependency_stats(),
        }

    def record_local_min_review_stats(
        self,
        *,
        total_neighbors=0,
        evaluated_neighbors=0,
        payload_score_cache_hits=0,
        prep_cache_prioritized=0,
        order_score_prioritized=0,
        field_order_score_prioritized=0,
        portfolio_dependency_deprioritized=0,
        signal_dependency_field_prioritized=0,
        early_stopped=False,
        selection_pruned=False,
        parallel_workers=0,
        parallel_submitted=0,
        parallel_completed=0,
        parallel_cancelled=0,
        hard_fail_stopped=False,
        hard_fail_neighbors_skipped=0,
        hard_fail_cancelled=0,
        dependency_signal_total=0,
        dependency_signal_evaluated=0,
        dependency_portfolio_total=0,
        dependency_portfolio_evaluated=0,
        dependency_mixed_total=0,
        dependency_mixed_evaluated=0,
        dependency_unknown_total=0,
        dependency_unknown_evaluated=0,
        signal_reuse_candidate_total=0,
        signal_reuse_candidate_evaluated=0,
        signal_recompute_required_total=0,
        signal_recompute_required_evaluated=0,
        dependency_field_total_counts=None,
        dependency_field_evaluated_counts=None,
    ):
        with self._local_min_stats_lock:
            stats = self.local_min_review_stats
            stats["total_neighbors"] = int(stats.get("total_neighbors", 0)) + int(total_neighbors or 0)
            stats["evaluated_neighbors"] = int(stats.get("evaluated_neighbors", 0)) + int(evaluated_neighbors or 0)
            stats["payload_score_cache_hits"] = int(stats.get("payload_score_cache_hits", 0)) + int(payload_score_cache_hits or 0)
            stats["prep_cache_prioritized"] = int(stats.get("prep_cache_prioritized", 0)) + int(prep_cache_prioritized or 0)
            stats["order_score_prioritized"] = int(stats.get("order_score_prioritized", 0)) + int(order_score_prioritized or 0)
            stats["field_order_score_prioritized"] = int(stats.get("field_order_score_prioritized", 0)) + int(field_order_score_prioritized or 0)
            stats["portfolio_dependency_deprioritized"] = int(stats.get("portfolio_dependency_deprioritized", 0)) + int(portfolio_dependency_deprioritized or 0)
            stats["signal_dependency_field_prioritized"] = int(stats.get("signal_dependency_field_prioritized", 0)) + int(signal_dependency_field_prioritized or 0)
            stats["parallel_workers_max"] = max(int(stats.get("parallel_workers_max", 0)), int(parallel_workers or 0))
            stats["parallel_submitted"] = int(stats.get("parallel_submitted", 0)) + int(parallel_submitted or 0)
            stats["parallel_completed"] = int(stats.get("parallel_completed", 0)) + int(parallel_completed or 0)
            stats["parallel_cancelled"] = int(stats.get("parallel_cancelled", 0)) + int(parallel_cancelled or 0)
            stats["hard_fail_neighbors_skipped"] = int(stats.get("hard_fail_neighbors_skipped", 0)) + int(hard_fail_neighbors_skipped or 0)
            stats["hard_fail_cancelled"] = int(stats.get("hard_fail_cancelled", 0)) + int(hard_fail_cancelled or 0)
            stats["dependency_signal_total"] = int(stats.get("dependency_signal_total", 0)) + int(dependency_signal_total or 0)
            stats["dependency_signal_evaluated"] = int(stats.get("dependency_signal_evaluated", 0)) + int(dependency_signal_evaluated or 0)
            stats["dependency_portfolio_total"] = int(stats.get("dependency_portfolio_total", 0)) + int(dependency_portfolio_total or 0)
            stats["dependency_portfolio_evaluated"] = int(stats.get("dependency_portfolio_evaluated", 0)) + int(dependency_portfolio_evaluated or 0)
            stats["dependency_mixed_total"] = int(stats.get("dependency_mixed_total", 0)) + int(dependency_mixed_total or 0)
            stats["dependency_mixed_evaluated"] = int(stats.get("dependency_mixed_evaluated", 0)) + int(dependency_mixed_evaluated or 0)
            stats["dependency_unknown_total"] = int(stats.get("dependency_unknown_total", 0)) + int(dependency_unknown_total or 0)
            stats["dependency_unknown_evaluated"] = int(stats.get("dependency_unknown_evaluated", 0)) + int(dependency_unknown_evaluated or 0)
            stats["signal_reuse_candidate_total"] = int(stats.get("signal_reuse_candidate_total", 0)) + int(signal_reuse_candidate_total or 0)
            stats["signal_reuse_candidate_evaluated"] = int(stats.get("signal_reuse_candidate_evaluated", 0)) + int(signal_reuse_candidate_evaluated or 0)
            stats["signal_recompute_required_total"] = int(stats.get("signal_recompute_required_total", 0)) + int(signal_recompute_required_total or 0)
            stats["signal_recompute_required_evaluated"] = int(stats.get("signal_recompute_required_evaluated", 0)) + int(signal_recompute_required_evaluated or 0)
            if not isinstance(stats.get("dependency_field_total_counts"), dict):
                stats["dependency_field_total_counts"] = {}
            if not isinstance(stats.get("dependency_field_evaluated_counts"), dict):
                stats["dependency_field_evaluated_counts"] = {}
            _merge_int_count_dict(stats["dependency_field_total_counts"], dependency_field_total_counts)
            _merge_int_count_dict(stats["dependency_field_evaluated_counts"], dependency_field_evaluated_counts)
            if bool(hard_fail_stopped):
                stats["hard_fail_stops"] = int(stats.get("hard_fail_stops", 0)) + 1
            if bool(early_stopped):
                stats["early_stops"] = int(stats.get("early_stops", 0)) + 1
            if bool(selection_pruned):
                stats["selection_prunes"] = int(stats.get("selection_prunes", 0)) + 1

    def get_local_min_review_stats(self):
        with self._local_min_stats_lock:
            stats = dict(getattr(self, "local_min_review_stats", {}) or {})
        total_neighbors = int(stats.get("total_neighbors", 0) or 0)
        evaluated_neighbors = int(stats.get("evaluated_neighbors", 0) or 0)
        return {
            "total_neighbors": total_neighbors,
            "evaluated_neighbors": evaluated_neighbors,
            "skipped_neighbors": max(0, total_neighbors - evaluated_neighbors),
            "payload_score_cache_hits": int(stats.get("payload_score_cache_hits", 0) or 0),
            "prep_cache_prioritized": int(stats.get("prep_cache_prioritized", 0) or 0),
            "order_score_prioritized": int(stats.get("order_score_prioritized", 0) or 0),
            "field_order_score_prioritized": int(stats.get("field_order_score_prioritized", 0) or 0),
            "portfolio_dependency_deprioritized": int(stats.get("portfolio_dependency_deprioritized", 0) or 0),
            "signal_dependency_field_prioritized": int(stats.get("signal_dependency_field_prioritized", 0) or 0),
            "early_stops": int(stats.get("early_stops", 0) or 0),
            "selection_prunes": int(stats.get("selection_prunes", 0) or 0),
            "parallel_workers_max": int(stats.get("parallel_workers_max", 0) or 0),
            "parallel_submitted": int(stats.get("parallel_submitted", 0) or 0),
            "parallel_completed": int(stats.get("parallel_completed", 0) or 0),
            "parallel_cancelled": int(stats.get("parallel_cancelled", 0) or 0),
            "hard_fail_stops": int(stats.get("hard_fail_stops", 0) or 0),
            "hard_fail_neighbors_skipped": int(stats.get("hard_fail_neighbors_skipped", 0) or 0),
            "hard_fail_cancelled": int(stats.get("hard_fail_cancelled", 0) or 0),
            "dependency_signal_total": int(stats.get("dependency_signal_total", 0) or 0),
            "dependency_signal_evaluated": int(stats.get("dependency_signal_evaluated", 0) or 0),
            "dependency_portfolio_total": int(stats.get("dependency_portfolio_total", 0) or 0),
            "dependency_portfolio_evaluated": int(stats.get("dependency_portfolio_evaluated", 0) or 0),
            "dependency_mixed_total": int(stats.get("dependency_mixed_total", 0) or 0),
            "dependency_mixed_evaluated": int(stats.get("dependency_mixed_evaluated", 0) or 0),
            "dependency_unknown_total": int(stats.get("dependency_unknown_total", 0) or 0),
            "dependency_unknown_evaluated": int(stats.get("dependency_unknown_evaluated", 0) or 0),
            "signal_reuse_candidate_total": int(stats.get("signal_reuse_candidate_total", 0) or 0),
            "signal_reuse_candidate_evaluated": int(stats.get("signal_reuse_candidate_evaluated", 0) or 0),
            "signal_recompute_required_total": int(stats.get("signal_recompute_required_total", 0) or 0),
            "signal_recompute_required_evaluated": int(stats.get("signal_recompute_required_evaluated", 0) or 0),
            "dependency_field_total_counts": _format_int_count_dict(stats.get("dependency_field_total_counts", {}) or {}),
            "dependency_field_evaluated_counts": _format_int_count_dict(stats.get("dependency_field_evaluated_counts", {}) or {}),
        }

    def get_prep_cache_stats(self):
        return {
            "hits": int(self.prep_cache_stats.get("hits", 0)),
            "misses": int(self.prep_cache_stats.get("misses", 0)),
            "stores": int(self.prep_cache_stats.get("stores", 0)),
            "evictions": int(self.prep_cache_stats.get("evictions", 0)),
            "items": int(len(self._prepared_trial_input_cache)),
            "max_items": int(self._prepared_trial_input_cache_max_items),
            "shared": bool(self._prepared_trial_input_cache_is_shared),
            "executor_created": int(self.prep_executor_stats.get("created", 0)),
            "executor_reused": int(self.prep_executor_stats.get("reused", 0)),
            "executor_shared": self._shared_trial_prep_executor_holder is not None,
            **dict(getattr(self, "prep_phase_stats", {}) or {}),
        }

    def _precompute_optimizer_true_range(self):
        for df in self.raw_data_cache.values():
            if df is None or len(df) == 0:
                continue
            if OPTIMIZER_TRUE_RANGE_ATTR in getattr(df, 'attrs', {}):
                continue
            df.attrs[OPTIMIZER_TRUE_RANGE_ATTR] = tv_true_range(
                df['High'].to_numpy(dtype='float64', copy=False),
                df['Low'].to_numpy(dtype='float64', copy=False),
                df['Close'].to_numpy(dtype='float64', copy=False),
            )

    def install_raw_data_cache(
        self,
        data_dir,
        raw_data_cache,
        *,
        static_fast_cache=None,
        master_dates=None,
        sorted_master_dates=None,
        required_min_rows=None,
    ):
        self.close_trial_prep_executor()
        self.raw_data_cache = dict(raw_data_cache or {})
        self._precompute_optimizer_true_range()
        self.raw_data_cache_data_dir = data_dir
        self.raw_data_cache_required_min_rows = coerce_raw_universe_required_min_rows(required_min_rows, allow_none=True)
        if not bool(self._prepared_trial_input_cache_is_shared):
            self._prepared_trial_input_cache.clear()
        self._full_evaluation_cache.clear()

        if static_fast_cache is None:
            self.static_fast_cache = {ticker: pack_static_market_data(df) for ticker, df in self.raw_data_cache.items()}
        else:
            self.static_fast_cache = dict(static_fast_cache)

        if master_dates is None:
            resolved_master_dates = set()
            for fast_df in self.static_fast_cache.values():
                resolved_master_dates.update(get_fast_dates(fast_df))
        else:
            resolved_master_dates = set(master_dates)
        self.master_dates = resolved_master_dates
        self.sorted_master_dates = list(sorted_master_dates) if sorted_master_dates is not None else sorted(resolved_master_dates)

    def load_raw_data(self, data_dir, *, load_all_raw_data, required_min_rows, verbose=True):
        if not callable(load_all_raw_data):
            raise TypeError(
                "load_raw_data 需要接收 tools.optimizer.prep.load_all_raw_data 函式；"
                f"目前收到 {type(load_all_raw_data).__name__}"
            )
        raw_data_cache = load_all_raw_data(
            data_dir=data_dir,
            required_min_rows=required_min_rows,
            output_dir=self.output_dir,
            verbose=bool(verbose),
        )
        self.install_raw_data_cache(data_dir, raw_data_cache, required_min_rows=required_min_rows)

    def cache_trial_milestone_inputs(self, trial_number, *, sorted_master_dates=None, all_pit_stats_index=None, all_dfs_fast=None):
        if all_pit_stats_index is None and all_dfs_fast is None:
            return
        cache = self._optimizer_trial_milestone_inputs
        cache[int(trial_number)] = {
            'sorted_master_dates': tuple(sorted_master_dates or ()),
            'all_pit_stats_index': all_pit_stats_index,
            'all_dfs_fast': all_dfs_fast,
        }
        if len(cache) > 3:
            oldest_trial = sorted(cache.keys())[0]
            if oldest_trial != int(trial_number):
                cache.pop(oldest_trial, None)

    def consume_trial_milestone_inputs(self, trial_number):
        return self._optimizer_trial_milestone_inputs.pop(int(trial_number), None)

    def discard_trial_milestone_inputs(self, trial_number):
        self._optimizer_trial_milestone_inputs.pop(int(trial_number), None)

    def has_prepared_trial_inputs_in_cache(self, cache_key):
        if cache_key is None or int(self._prepared_trial_input_cache_max_items) <= 0:
            return False
        with self._cache_lock:
            return cache_key in self._prepared_trial_input_cache

    def get_prepared_trial_inputs_from_cache(self, cache_key):
        if int(self._prepared_trial_input_cache_max_items) <= 0:
            return None
        with self._cache_lock:
            cached = self._prepared_trial_input_cache.get(cache_key)
            if cached is None:
                self.prep_cache_stats["misses"] = int(self.prep_cache_stats.get("misses", 0)) + 1
                return None
            self.prep_cache_stats["hits"] = int(self.prep_cache_stats.get("hits", 0)) + 1
            self._prepared_trial_input_cache.move_to_end(cache_key)
            return {
                "all_dfs_fast": cached["all_dfs_fast"],
                "all_trade_logs": cached["all_trade_logs"],
                "all_pit_stats_index": cached["all_pit_stats_index"],
                "master_dates": cached["master_dates"],
                "prep_failures": cached["prep_failures"],
                "prep_profile": {
                    "worker_total_sum_sec": 0.0,
                    "prep_total_sum_sec": 0.0,
                    "copy_sum_sec": 0.0,
                    "generate_signals_sum_sec": 0.0,
                    "assign_sum_sec": 0.0,
                    "run_backtest_sum_sec": 0.0,
                    "to_dict_sum_sec": 0.0,
                    "feature_bank_hits": 0,
                    "feature_bank_misses": 0,
                    "feature_bank_size_max": 0,
                    "feature_bank_max_items": 0,
                    "static_pack_sec": 0.0,
                    "executor_setup_sec": 0.0,
                    "executor_submit_sec": 0.0,
                    "executor_collect_sec": 0.0,
                    "executor_shutdown_sec": 0.0,
                    "merge_sum_sec": 0.0,
                    "master_union_sec": 0.0,
                    "ticker_count": 0,
                    "batch_count": 0,
                    "ok_count": int(cached["ok_count"]),
                    "fail_count": int(cached["fail_count"]),
                },
                "prep_mode": "cache_hit",
                "pool_start_method": "cache",
                "pool_error_text": None,
                "prep_wall_sec": 0.0,
            }

    def cache_prepared_trial_inputs(self, cache_key, prep_result):
        if cache_key is None:
            return
        with self._cache_lock:
            self._record_prepared_trial_inputs_profile(prep_result)
            if int(self._prepared_trial_input_cache_max_items) <= 0:
                return
            self.prep_cache_stats["stores"] = int(self.prep_cache_stats.get("stores", 0)) + 1
            self._prepared_trial_input_cache[cache_key] = {
                "all_dfs_fast": prep_result["all_dfs_fast"],
                "all_trade_logs": prep_result["all_trade_logs"],
                "all_pit_stats_index": prep_result.get("all_pit_stats_index"),
                "master_dates": prep_result["master_dates"],
                "prep_failures": tuple(prep_result.get("prep_failures", ())),
                "ok_count": int(prep_result.get("prep_profile", {}).get("ok_count", 0)),
                "fail_count": int(prep_result.get("prep_profile", {}).get("fail_count", 0)),
            }
            self._prepared_trial_input_cache.move_to_end(cache_key)
            while len(self._prepared_trial_input_cache) > int(self._prepared_trial_input_cache_max_items):
                self._prepared_trial_input_cache.popitem(last=False)
                self.prep_cache_stats["evictions"] = int(self.prep_cache_stats.get("evictions", 0)) + 1

    def get_full_evaluation_from_cache(self, cache_key):
        if cache_key is None:
            return None
        with self._cache_lock:
            cached = self._full_evaluation_cache.get(cache_key)
            if cached is None:
                return None
            self._full_evaluation_cache.move_to_end(cache_key)
            return copy.deepcopy(cached)

    def cache_full_evaluation(self, cache_key, evaluation):
        if cache_key is None or evaluation is None:
            return
        with self._cache_lock:
            if int(self._full_evaluation_cache_max_items) <= 0:
                return
            self._full_evaluation_cache[cache_key] = copy.deepcopy(dict(evaluation))
            self._full_evaluation_cache.move_to_end(cache_key)
            while len(self._full_evaluation_cache) > int(self._full_evaluation_cache_max_items):
                self._full_evaluation_cache.popitem(last=False)

    @staticmethod
    def _shutdown_trial_prep_executor_bundle(bundle):
        if bundle is None:
            return
        executor = bundle.get("executor")
        shutdown = getattr(executor, "shutdown", None)
        if callable(shutdown):
            shutdown(wait=True, cancel_futures=False)

    def get_trial_prep_executor_bundle(self, max_workers):
        try:
            requested_workers = int(max_workers)
        except (TypeError, ValueError):
            requested_workers = int(self.default_max_workers)
        requested_workers = max(1, requested_workers)

        with self._cache_lock:
            shared_holder = self._shared_trial_prep_executor_holder
            existing = shared_holder.get("bundle") if shared_holder is not None else self._trial_prep_executor_bundle
            if existing is not None:
                existing_data_dir = existing.get("data_dir")
                existing_workers = int(existing.get("max_workers", 0))
                if existing_data_dir == self.raw_data_cache_data_dir and existing_workers == requested_workers:
                    self.prep_executor_stats["reused"] = int(self.prep_executor_stats.get("reused", 0)) + 1
                    return existing
                if shared_holder is not None:
                    self._shutdown_trial_prep_executor_bundle(shared_holder.pop("bundle", None))
                else:
                    self.close_trial_prep_executor()

            if not self.raw_data_cache:
                return None

            executor, pool_start_method, supports_initializer = _build_process_pool_executor(requested_workers, self.raw_data_cache)
            executor_kind = 'process'
            if executor_kind != 'thread' and not supports_initializer:
                executor.shutdown(wait=True, cancel_futures=False)
                return None

            bundle = {
                "executor": executor,
                "max_workers": requested_workers,
                "data_dir": self.raw_data_cache_data_dir,
                "pool_start_method": pool_start_method,
                "executor_kind": executor_kind,
            }
            self.prep_executor_stats["created"] = int(self.prep_executor_stats.get("created", 0)) + 1
            if shared_holder is not None:
                shared_holder["bundle"] = bundle
            else:
                self._trial_prep_executor_bundle = bundle
            return bundle

    def close_trial_prep_executor(self, *, force_shared=False):
        if self._shared_trial_prep_executor_holder is not None and not bool(force_shared):
            return
        if self._shared_trial_prep_executor_holder is not None:
            bundle = self._shared_trial_prep_executor_holder.pop("bundle", None)
            self._shutdown_trial_prep_executor_bundle(bundle)
            return
        bundle = self._trial_prep_executor_bundle
        self._trial_prep_executor_bundle = None
        self._shutdown_trial_prep_executor_bundle(bundle)

    def record_optimizer_prep_failures(self, insufficient_failures):
        insufficient_count = len(insufficient_failures)
        if insufficient_count <= 0:
            return
        with self._cache_lock:
            self.prep_summary["trials_with_insufficient"] += 1
            self.prep_summary["insufficient_count_total"] += insufficient_count

    def print_optimizer_prep_summary(self):
        insufficient_total = self.prep_summary["insufficient_count_total"]
        if insufficient_total == 0:
            return
        print(
            f"{self.colors['yellow']}⚠️ 本輪預處理摘要："
            f"資料不足 trial={self.prep_summary['trials_with_insufficient']} 次 / 累計 {insufficient_total} 檔。{self.colors['reset']}"
        )

    def objective(self, trial):
        return run_optimizer_objective(self, trial)

    def monitoring_callback(self, study, trial):
        return run_optimizer_monitoring_callback(self, study, trial)

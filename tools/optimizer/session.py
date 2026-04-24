from tools.optimizer.callbacks import run_optimizer_monitoring_callback
from tools.optimizer.objective import run_optimizer_objective
from tools.optimizer.trial_inputs import _build_process_pool_executor
from core.portfolio_fast_data import get_fast_dates, pack_static_market_data


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
        self._optimizer_trial_milestone_inputs = {}
        self.static_fast_cache = {}
        self.master_dates = set()
        self.sorted_master_dates = []

    def load_raw_data(self, data_dir, *, load_all_raw_data, required_min_rows):
        self.close_trial_prep_executor()
        self.raw_data_cache = load_all_raw_data(
            data_dir=data_dir,
            required_min_rows=required_min_rows,
            output_dir=self.output_dir,
        )
        self.raw_data_cache_data_dir = data_dir
        self.static_fast_cache = {ticker: pack_static_market_data(df) for ticker, df in self.raw_data_cache.items()}
        self.master_dates = set()
        for fast_df in self.static_fast_cache.values():
            self.master_dates.update(get_fast_dates(fast_df))
        self.sorted_master_dates = sorted(self.master_dates)

    def cache_trial_milestone_inputs(self, trial_number, *, sorted_master_dates=None, all_pit_stats_index=None):
        if all_pit_stats_index is None:
            return
        cache = self._optimizer_trial_milestone_inputs
        cache[int(trial_number)] = {
            'sorted_master_dates': tuple(sorted_master_dates or ()),
            'all_pit_stats_index': all_pit_stats_index,
        }
        if len(cache) > 3:
            oldest_trial = sorted(cache.keys())[0]
            if oldest_trial != int(trial_number):
                cache.pop(oldest_trial, None)

    def consume_trial_milestone_inputs(self, trial_number):
        return self._optimizer_trial_milestone_inputs.pop(int(trial_number), None)

    def discard_trial_milestone_inputs(self, trial_number):
        self._optimizer_trial_milestone_inputs.pop(int(trial_number), None)

    def get_trial_prep_executor_bundle(self, max_workers):
        try:
            requested_workers = int(max_workers)
        except (TypeError, ValueError):
            requested_workers = int(self.default_max_workers)
        requested_workers = max(1, requested_workers)

        existing = self._trial_prep_executor_bundle
        if existing is not None:
            existing_data_dir = existing.get("data_dir")
            existing_workers = int(existing.get("max_workers", 0))
            if existing_data_dir == self.raw_data_cache_data_dir and existing_workers == requested_workers:
                return existing
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
        self._trial_prep_executor_bundle = bundle
        return bundle

    def close_trial_prep_executor(self):
        bundle = self._trial_prep_executor_bundle
        self._trial_prep_executor_bundle = None
        if bundle is None:
            return
        executor = bundle.get("executor")
        shutdown = getattr(executor, "shutdown", None)
        if callable(shutdown):
            shutdown(wait=True, cancel_futures=False)

    def record_optimizer_prep_failures(self, insufficient_failures):
        insufficient_count = len(insufficient_failures)
        if insufficient_count <= 0:
            return
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

"""Execution-only invariants for independent canonical producer scheduling."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch

from services.optimizer import independent_seed_fold_queue as queue
from services.optimizer.execution_settings import OptimizerExecutionSettings


class IndependentSeedFoldQueueTests(unittest.TestCase):
    def test_worker_limit_uses_config_and_environment(self):
        with patch("core.training_performance.OPTIMIZER_INDEPENDENT_SEED_FOLD_GLOBAL_WORKERS", 3):
            self.assertEqual(queue.resolve_optimizer_independent_seed_fold_global_workers({}), 3)
            self.assertEqual(queue.resolve_optimizer_independent_seed_fold_global_workers({
                "OPTIMIZER_INDEPENDENT_SEED_FOLD_GLOBAL_WORKERS": "2"}), 2)

    def test_worker_environment_replaces_and_restores_each_seed_snapshot(self):
        from services.optimizer import outer_rolling_oos
        keys = ("OPTIMIZER_TEST_SEED", "OPTIMIZER_TEST_STALE", "BREAKOUT_QUALITY_BINARY_PIT_SCORES")
        observations = []
        def observe(task):
            observations.append(tuple(os.environ.get(key) for key in keys))
            return {"seed": task["seed"]}
        with patch.dict(os.environ, {keys[0]: "ambient", keys[1]: "stale"}, clear=True):
            before = dict(os.environ)
            with patch.object(outer_rolling_oos, "_run_outer_rolling_oos_fold_task", side_effect=observe):
                for seed, extra in ((11, {keys[2]: "first.csv"}), (22, {})):
                    settings = OptimizerExecutionSettings.from_environ({keys[0]: str(seed), **extra})
                    self.assertEqual(queue._execute_isolated_fold({"seed": seed, "execution_settings": settings}), {"seed": seed})
                    self.assertEqual(dict(os.environ), before)
        self.assertEqual(observations, [("11", None, "first.csv"), ("22", None, None)])

    def test_adapter_uses_canonical_fold_worker_and_preserves_results(self):
        from services.optimizer.outer_rolling_oos import _run_outer_rolling_oos_fold_task
        broker = Mock()
        broker.run_fold.side_effect = lambda task: {"seed": task["seed"]}
        with queue._SharedFoldExecutor(broker, 2) as executor:
            results = [executor.submit(_run_outer_rolling_oos_fold_task, {"seed": seed}) for seed in (9, 4)]
            self.assertEqual([future.result() for future in results], [{"seed": 9}, {"seed": 4}])
            with self.assertRaises(ValueError):
                executor.submit(str, {})
        broker.shutdown.assert_not_called()

    def test_adapter_failure_aborts_only_its_broker(self):
        broker = Mock()
        with self.assertRaisesRegex(RuntimeError, "fold failed"):
            with queue._SharedFoldExecutor(broker, 1):
                raise RuntimeError("fold failed")
        broker.shutdown.assert_called_once_with(abort=True)

    def test_one_worker_budget_includes_finalization_and_queued_folds(self):
        entered = Event()
        release = Event()
        seen = []
        def execute(task):
            seen.append(task["seed"])
            entered.set()
            release.wait(2)
            return {"seed": task["seed"]}
        with patch.object(queue, "ProcessPoolExecutor", side_effect=lambda **kwargs: ThreadPoolExecutor(max_workers=2)):
            broker = queue._FoldBroker(1)
        with patch.object(queue, "_execute_isolated_fold", side_effect=execute):
            broker.acquire_finalization()
            with ThreadPoolExecutor(max_workers=2) as clients:
                first = clients.submit(broker.run_fold, {"seed": 1})
                second = clients.submit(broker.run_fold, {"seed": 2})
                self.assertFalse(entered.wait(.03), "Finalization must consume the sole heavy slot")
                broker.release_finalization()
                self.assertTrue(entered.wait(2))
                self.assertEqual(len(seen), 1)
                release.set()
                self.assertEqual({first.result()["seed"], second.result()["seed"]}, {1, 2})
            self.assertEqual([task["queue_status"] for task in broker.progress()], ["DONE", "DONE"])
        broker.shutdown()

    def test_abort_wakes_queued_callers_and_prevents_new_work(self):
        with patch.object(queue, "ProcessPoolExecutor", return_value=Mock()):
            broker = queue._FoldBroker(1)
        broker.acquire_finalization()
        with ThreadPoolExecutor(max_workers=1) as clients:
            waiting = clients.submit(broker.run_fold, {"seed": 1})
            with patch("services.optimizer.outer_rolling_parallel_runner._cancel_parallel_fold_executor_after_failure") as cancel:
                broker.shutdown(abort=True)
                cancel.assert_called_once_with(broker._pool, set())
            with self.assertRaisesRegex(RuntimeError, "queue is closed"):
                waiting.result(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "queue is closed"):
            broker.run_fold({"seed": 2})

    def test_progress_counts_same_fold_in_different_seeds_separately(self):
        tasks = [{"optimizer_seed": seed, "fold_idx": 1, "fold_count": 2, "queue_status": "RUNNING"} for seed in (3, 8)]
        observation = {"fold_idx": 1, "completed": 5, "stage": "SEARCH", "search_started_ts": 10, "search_last_done_ts": 20}
        with patch("services.optimizer.outer_rolling_parallel_progress._read_latest_parallel_fold_progress", return_value=observation):
            lines = queue._progress_lines(tasks, seeds=2, workers=1, started=time.perf_counter())
        self.assertIn("folds=2 | seeds=2", lines[1])
        self.assertIn("avg_trial=1.000s", lines[1])
        self.assertIn("completed_jobs=0/2", lines[1])
        self.assertTrue(lines[2].startswith("seed=3 "))
        self.assertTrue(lines[3].startswith("seed=8 "))

    def test_empty_requests_do_not_start_processes(self):
        with patch.object(queue, "_FoldBrokerManager") as manager:
            self.assertEqual(queue.run_independent_optimizer_producers_shared_fold_queue(str, []), [])
        manager.assert_not_called()

    def test_uncached_chain_preparation_stays_in_reserved_slot(self):
        from services.optimizer import outer_rolling_active_replay_runtime as replay
        import pandas as pd
        record = {"params_signature": "same-contract", "params_obj": object(), "year": 2008}
        prepared = {"all_pit_stats_index": {"2330": []}, "all_dfs_fast": {}, "master_dates": [], "prep_mode": "inline"}
        with ExitStack() as stack:
            stack.enter_context(patch.object(queue, "_FOLD_TASK_ACTIVE", True))
            stack.enter_context(patch.object(replay, "_iter_active_replay_context_records", return_value=[record]))
            stack.enter_context(patch.object(replay, "is_optimizer_active_replay_use_prepared_cache_enabled", return_value=False))
            stack.enter_context(patch.object(replay, "is_optimizer_active_replay_write_prepared_cache_enabled", return_value=False))
            stack.enter_context(patch("core.data_utils.get_required_min_rows", return_value=1))
            stack.enter_context(patch("services.optimizer.raw_cache.load_all_raw_data", return_value={"2330": pd.DataFrame(index=pd.to_datetime(["2008-01-02"]))}))
            stack.enter_context(patch("core.portfolio_fast_data.pack_static_market_data", return_value={}))
            nested = stack.enter_context(patch.object(replay, "_build_process_pool_executor", side_effect=AssertionError("nested pool")))
            prep = stack.enter_context(patch.object(replay, "prepare_trial_inputs", return_value=prepared))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            actual = replay._load_active_replay_contexts_by_signature(data_dir="unused", schedule_groups={"best": [record]}, output_dir="unused")
        nested.assert_not_called()
        self.assertEqual(prep.call_args.kwargs["executor_bundle"]["executor_kind"], "inline")
        self.assertEqual(prep.call_args.kwargs["default_max_workers"], 1)
        self.assertEqual(actual["same-contract"]["all_pit_stats_index"], prepared["all_pit_stats_index"])

    def test_startup_failure_closes_started_manager(self):
        manager = Mock()
        manager.FoldBroker.side_effect = RuntimeError("broker startup")
        with patch.object(queue, "_FoldBrokerManager", return_value=manager):
            with self.assertRaisesRegex(RuntimeError, "broker startup"):
                queue.run_independent_optimizer_producers_shared_fold_queue(str, [{}], max_workers=1)
        manager.shutdown.assert_called_once()

    def test_coordinator_logs_do_not_interleave_console_and_restores_context(self):
        broker = Mock()
        def producer(**kwargs):
            print("canonical producer output")
            queue.acquire_independent_finalization_slot()
            return kwargs["seed"]
        with TemporaryDirectory() as directory:
            log = Path(directory) / "coordinator.log"
            with patch("sys.stdout", new_callable=io.StringIO) as console:
                self.assertEqual(queue._run_producer(producer, {"seed": 17}, broker, 1, {}, str(log)), 17)
                self.assertEqual(console.getvalue(), "")
            self.assertIn("canonical producer output", log.read_text(encoding="utf-8"))
        self.assertIsNone(queue.independent_fold_queue_workers())
        broker.acquire_finalization.assert_called_once()
        broker.release_finalization.assert_called_once()


if __name__ == "__main__":
    unittest.main()

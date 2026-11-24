"""Regression checks for optimizer neighbor throughput telemetry."""
import json
from pathlib import Path
import tempfile
import unittest

from services.optimizer import outer_rolling_progress as progress


def _local_event(**overrides):
    return {
        "stage": "LOCAL_MIN_REVIEW", "status": "DONE", "fold_idx": 1,
        "seed_ensemble_member_index": 1, "finalist_idx": 6,
        "neighbor_done": 1, "neighbor_total": 1,
        "local_min_started_ts": 100.0, "ts": 328.0,
        **overrides,
    }


def _header(metrics):
    return progress.format_optimizer_progress_performance_header(
        folds=1, seeds=1, min_agree=None, parallel_workers=1, backend="process",
        total_elapsed_sec=228.0,
        completed_local_min_trials=metrics["completed_local_min_trials"],
        local_min_wall_elapsed_sec=metrics["local_min_wall_elapsed_sec"],
    )


class OptimizerProgressTests(unittest.TestCase):
    def test_explicit_neighbor_counter_including_zero_precedes_legacy_position(self):
        cases = [
            ({"local_min_neighbor_completed": 0}, 0),
            ({"local_min_neighbor_completed": "0"}, 0),
            ({"completed_local_min_neighbors": 0}, 0),
            ({"local_min_neighbor_completed": 7}, 7),
            ({"completed_local_min_neighbors": "9"}, 9),
            ({"local_min_neighbor_completed": 0, "completed_local_min_neighbors": 9}, 0),
            ({"local_min_neighbor_completed": 7, "completed_local_min_neighbors": 0}, 7),
        ]
        for counters, expected in cases:
            with self.subTest(counters=counters):
                self.assertEqual(
                    progress._count_local_min_completed_neighbors_from_progress(_local_event(**counters)),
                    expected,
                )

    def test_missing_or_invalid_counter_preserves_legacy_fallback(self):
        for value in (None, "", "bad", -1, float("inf"), float("nan")):
            with self.subTest(value=value):
                self.assertEqual(progress._count_local_min_completed_neighbors_from_progress(
                    _local_event(local_min_neighbor_completed=value, completed_local_min_neighbors=0)), 0)
                self.assertEqual(progress._count_local_min_completed_neighbors_from_progress(
                    _local_event(local_min_neighbor_completed=value)), 6)
        self.assertEqual(progress._count_local_min_completed_neighbors_from_progress(_local_event()), 6)
        self.assertEqual(progress._count_local_min_completed_neighbors_from_progress(
            _local_event(finalist_idx=0, neighbor_done=3)), 3)
        self.assertEqual(progress._count_local_min_completed_neighbors_from_progress(
            _local_event(stage="OOS_REPLAY")), 0)

    def test_zero_neighbors_never_render_local_average_or_inflate_all_average(self):
        events = [
            _local_event(finalist_idx=index, local_min_neighbor_completed=0,
                         completed=4, search_active_intervals=[[100.0, 104.0]])
            for index in range(1, 7)
        ]
        metrics = progress._collect_optimizer_progress_phase_metrics(events)
        self.assertEqual(metrics["completed_local_min_trials"], 0)
        self.assertEqual(metrics["completed_local_min_finalists"], 6)
        self.assertNotIn("avg_local=", _header(metrics))
        header = progress.format_optimizer_progress_performance_header(
            folds=1, seeds=1, min_agree=None, parallel_workers=1, backend="process",
            total_elapsed_sec=228.0, completed_trials=metrics["completed_trials"],
            search_wall_elapsed_sec=metrics["search_wall_elapsed_sec"],
            completed_local_min_trials=metrics["completed_local_min_trials"],
            local_min_wall_elapsed_sec=metrics["local_min_wall_elapsed_sec"],
        )
        self.assertIn("avg_trial=1.000s", header)
        self.assertIn("avg_all=57.000s", header)
        self.assertNotIn("avg_local=", header)

    def test_actual_neighbors_keep_average_across_later_phases(self):
        events = [
            _local_event(local_min_neighbor_completed=0, ts=100.0),
            _local_event(local_min_neighbor_completed=4, local_min_last_neighbor_done_ts=108.0, ts=108.0),
            _local_event(stage="OOS_REPLAY", local_min_neighbor_completed=0, ts=300.0),
        ]
        metrics = progress._collect_optimizer_progress_phase_metrics(events)
        self.assertEqual(metrics["completed_local_min_trials"], 4)
        self.assertEqual(metrics["local_min_wall_elapsed_sec"], 8.0)
        self.assertIn("avg_local=2.000s", _header(metrics))

    def test_fold_log_and_live_board_keep_disabled_local_count_zero(self):
        from services.optimizer.outer_rolling_parallel_progress import _read_latest_parallel_fold_progress
        from services.optimizer.outer_rolling_live_boards import OptimizerSeedEnsembleProgressBoard

        events = [_local_event(finalist_idx=index, local_min_neighbor_completed=0) for index in range(1, 7)]
        with tempfile.TemporaryDirectory(prefix="optimizer-progress-") as temp:
            path = Path(temp) / "fold.log"
            path.write_text("\n".join(progress.PARALLEL_FOLD_PROGRESS_PREFIX + "\t" + json.dumps(event)
                                      for event in events), encoding="utf-8")
            latest = _read_latest_parallel_fold_progress(str(path))
            members = progress._read_latest_parallel_fold_seed_progresses(str(path))
        self.assertEqual(latest["local_min_neighbor_completed"], 0)
        self.assertEqual(members[1]["local_min_neighbor_completed"], 0)
        board = OptimizerSeedEnsembleProgressBoard([])
        for event in events:
            board._record_progress_metrics((1, 1), event)
        self.assertEqual(board.get_completed_local_min_trial_count(), 0)
        self.assertEqual(board.get_completed_local_min_finalist_count(), 6)


if __name__ == "__main__":
    unittest.main()

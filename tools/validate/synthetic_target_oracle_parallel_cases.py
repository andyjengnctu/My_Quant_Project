"""Independent contract tests for dynamic Target Oracle seed-level Replay workers.

No market or model artifacts are required, and no formal suite is invoked.
This is a new execution primitive: the canonical scientific Replay remains the
same service and only the authorized RUN cells may reach a spawned worker.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audit_policy import get_reusable_audit_definitions
from services.audit import target_oracle_romd as oracle


def _spawn_worker_probe(seed: int, target: str, project_root: str):
    """Executed in a fresh spawn process to verify worker isolation/routing."""
    from unittest.mock import patch

    definition = SimpleNamespace(output_subdir="probe")
    job = {
        "project_root": Path(project_root),
        "prepared": [(seed, "sha", "kind", None, None)],
        "decisions": {(seed, target): "RUN"},
        "execution_contexts": {(seed, target): {"seed": seed, "target": target}},
    }
    observed = {}

    def fake_replay(_definition, **kwargs):
        observed.update(kwargs)
        return {(seed, target): {"seed": seed, "target": target}}

    with patch.object(oracle, "_execute_single_stage_missing", side_effect=fake_replay):
        result = oracle._oracle_missing_seed_worker(
            definition, seed=seed, job=job, price_targets=frozenset(),
        )
    return (result, set(observed["decisions"]),
            observed["cache"].cache_root.as_posix(),
            observed["price_score_allow_build"])


class _InstantFuture:
    def __init__(self, answer):
        self.answer = answer
        self.cancelled = False

    def result(self):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def cancel(self):
        self.cancelled = True


class _InstantPool:
    seen = []
    max_workers = None
    response_error_seed = None

    def __init__(self, *, max_workers, mp_context):
        type(self).max_workers = max_workers
        assert mp_context.get_start_method() == "spawn"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, _worker, definition, *, seed, job, price_targets):
        self.seen.append((seed, job, price_targets))
        if seed == type(self).response_error_seed:
            return _InstantFuture(ValueError("injected worker failure"))
        return _InstantFuture({key: {"seed": seed, "target": key[1]}
                               for key in job["decisions"]})


class OracleParallelSeedWorkCases(TestCase):
    def setUp(self):
        self.definition = next(
            definition for definition in get_reusable_audit_definitions("breakout_quality")
            if definition.audit_type == oracle.SUPPORTED_AUDIT_TYPE
        )
        self.plan = oracle._oracle_comparison_plan(self.definition)
        self.targets = ("C96", *(target for target, _ in self.plan["targets"]))
        self.seeds = (693545351, 2014432738, 104504529, 762252050,
                      9734312, 1020806211, 116293459, 1234683984)
        _InstantPool.seen = []
        _InstantPool.response_error_seed = None

    def _args(self, root, decisions, *, backfill=None):
        return dict(
            project_root=root, quiet=True, plan=self.plan, contracts={},
            current_keys={}, settings=SimpleNamespace(dataset="synthetic", end_date="2026-03-02"),
            robustness=SimpleNamespace(benchmark_id="synthetic_8"),
            price_sha="f" * 64,
            prepared=[(seed, str(seed), "synthetic", {}, {}) for seed in self.seeds],
            decisions=decisions, preflight_cached={(seed, "C96"): {"romd": 9.0}
                                                   for seed in self.seeds},
            start="2021-01-01", end="2026-03-02",
            base_arm=None, exact_arm=None, direct_arm=None,
            base_policy={}, cache=None,
            allow_historical_execution_input_drift=True,
            metric_backfill=backfill or {},
            execution_contexts={key: {"seed": key[0], "target": key[1]}
                                for key, action in decisions.items() if action == "RUN"},
        )

    def _patch_source_validation(self, args):
        prepared = args["prepared"]
        return (
            patch.object(oracle, "price_inventory_sha256", return_value="f" * 64),
            patch.object(oracle, "_robustness_artifacts", return_value=(None, [
                (seed, Path("p16.json")) for seed in self.seeds])),
            patch.object(oracle, "_prepare_oracle_parameters", return_value=(prepared, [])),
            patch.object(oracle, "_load_or_build_oracle_price_scores", return_value=({}, {})),
            patch.object(oracle, "ProcessPoolExecutor", _InstantPool),
            patch.object(oracle, "as_completed", side_effect=lambda futures: tuple(futures)),
        )

    def test_sparse_seed6_and_full_seed7_seed8_are_dynamic_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            decisions = {(seed, target): ("RUN" if
                (seed in self.seeds[6:] or
                 seed == self.seeds[5] and target in self.targets[-3:])
                else "REUSE")
                for seed in self.seeds for target in self.targets}
            args = self._args(Path(tmp), decisions)
            from contextlib import ExitStack
            with ExitStack() as stack:
                for ctx in self._patch_source_validation(args):
                    stack.enter_context(ctx)
                stack.enter_context(patch.object(oracle, "TARGET_ORACLE_REPLAY_SEED_WORKERS", 2))
                result = oracle._execute_oracle_missing_parallel(self.definition, **args)
            self.assertEqual(_InstantPool.max_workers, 2)
            self.assertEqual([seed for seed, _job, _targets in _InstantPool.seen], list(self.seeds[5:]))
            self.assertEqual(len(_InstantPool.seen[0][1]["decisions"]), 3)
            self.assertEqual(len(_InstantPool.seen[1][1]["decisions"]), len(self.targets))
            self.assertEqual(len(_InstantPool.seen[2][1]["decisions"]), len(self.targets))
            self.assertEqual(len(result), 3 + len(self.targets) * 2)
            self.assertFalse(any(seed in self.seeds[:5] for seed, _target in result))
            self.assertEqual({seed for seed, _target in result}, set(self.seeds[5:]))
            for seed, job, _targets in _InstantPool.seen:
                self.assertEqual(set(job["execution_contexts"]), set(job["decisions"]))
                self.assertTrue(all(key[0] == seed for key in job["execution_contexts"]))

    def test_all_reuse_never_calls_any_producer_or_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp), {(seed, target): "REUSE"
                                         for seed in self.seeds for target in self.targets})
            with patch.object(oracle, "ProcessPoolExecutor", side_effect=AssertionError("spawned")), \
                 patch.object(oracle, "price_inventory_sha256", side_effect=AssertionError("hashed")):
                self.assertEqual(oracle._execute_oracle_missing_parallel(self.definition, **args), {})

    def test_one_run_seed_calls_original_execution_without_parallel_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp), {(self.seeds[5], self.targets[-1]): "RUN"})
            expected = {(self.seeds[5], self.targets[-1]): {"romd": 1}}
            with patch.object(oracle, "_execute_single_stage_missing", return_value=expected) as single, \
                 patch.object(oracle, "ProcessPoolExecutor", side_effect=AssertionError("spawned")), \
                 patch.object(oracle, "TARGET_ORACLE_REPLAY_SEED_WORKERS", 3):
                self.assertEqual(oracle._execute_oracle_missing_parallel(self.definition, **args), expected)
            single.assert_called_once()

    def test_foreign_backfill_is_blocked_before_any_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp), {(self.seeds[5], self.targets[-1]): "RUN"},
                              backfill={(self.seeds[4], self.targets[-1]): ({}, {})})
            with patch.object(oracle, "_execute_single_stage_missing", side_effect=AssertionError("ran")):
                with self.assertRaisesRegex(oracle.OracleBlockedError, "authorized RUN"):
                    oracle._execute_oracle_missing_parallel(self.definition, **args)

    def test_missing_execution_contract_is_blocked_before_any_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp), {(self.seeds[0], self.targets[-1]): "RUN"})
            args["execution_contexts"] = {}
            with patch.object(oracle, "price_inventory_sha256", side_effect=AssertionError("No source reads")), \
                 patch.object(oracle, "_execute_single_stage_missing", side_effect=AssertionError("No Replay")), \
                 patch.object(oracle, "ProcessPoolExecutor", side_effect=AssertionError("No process")):
                with self.assertRaisesRegex(oracle.OracleBlockedError, "effective execution context"):
                    oracle._execute_oracle_missing_parallel(self.definition, **args)

    def test_failed_worker_does_not_create_unapproved_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            decisions = {(seed, self.targets[-1]): "RUN" for seed in self.seeds[6:]}
            args = self._args(Path(tmp), decisions)
            _InstantPool.response_error_seed = self.seeds[7]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for ctx in self._patch_source_validation(args):
                    stack.enter_context(ctx)
                with self.assertRaisesRegex(oracle.OracleBlockedError, "injected worker failure"):
                    oracle._execute_oracle_missing_parallel(self.definition, **args)
            self.assertEqual(len(_InstantPool.seen), 2)

    def test_spawn_worker_keeps_cache_and_decisions_seed_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = self.seeds[5]
            with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
                result, scoped, path, build = pool.submit(
                    _spawn_worker_probe, seed, self.targets[-1], tmp,
                ).result(timeout=45)
            self.assertEqual(set(result), {(seed, self.targets[-1])})
            self.assertEqual(scoped, {(seed, self.targets[-1])})
            self.assertIn("replay_cache", path)
            self.assertFalse(build)


if __name__ == "__main__":
    main()

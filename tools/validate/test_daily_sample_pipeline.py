"""Daily collection, teacher binding and exclusion contracts without market I/O."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from filters.breakout_quality import daily_ranker_data as loader
from filters.breakout_quality import daily_model_input_context as contexts
from tools.validate.synthetic_daily_trade40_cases import fixture


def _outcomes(*, frame, positions, **kwargs):
    valid = (positions % 3 != 0) & (frame.attrs.get("ticker") != "NULL")
    return SimpleNamespace(
        raw_net_r=np.where(valid, positions / 100.0, np.nan),
        raw_net_pnl=np.where(valid, positions * 10.0, np.nan),
        completed=valid,
        maturity_dates=frame.index.take(positions + 1).to_numpy(dtype="datetime64[D]"),
    )


def build_isolated_daily_bundle(owner=loader, *, workers=1, context=False,
                                failures=False, profile=None, teacher=None):
    """Exercise the real loader with deterministic source/teacher boundaries."""
    from filters.breakout_quality import daily_trade40_target
    days, frame, _params, _period = fixture(count=520, start="2018-01-01")
    inputs = [("0050", Path("0050.csv")), ("NULL", Path("NULL.csv")),
              ("2330", Path("2330.csv")), ("2317", Path("2317.csv"))]
    if failures:
        inputs.insert(2, ("BROKEN", Path("BROKEN.csv")))
        inputs.append(("SHORT", Path("SHORT.csv")))

    def read(_path, ticker, **kwargs):
        if ticker == "BROKEN":
            raise ValueError("probe malformed OHLCV")
        result = frame.iloc[:10].copy() if ticker == "SHORT" else frame.copy()
        result.attrs["ticker"] = ticker
        return result

    recipe_profile = profile or "daily_universal_formal_mintrade_fullcycle_netpnl_per_seed_full_list_ndcg_pairwise"
    recipe = owner.get_continuous_ranker_execution_recipe(recipe_profile)
    real_path = recipe.target_policy.materialization_mode == owner.CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH
    teacher = object() if teacher is None else teacher
    kwargs = dict(
        filter_id="breakout_quality_v1", preload_feature_bank=False,
        allow_stale_source=False, project_root=Path.cwd(),
        model_architecture="inception_time_v1", experiment_profile=recipe_profile,
    )
    if real_path:
        kwargs.update(
            target_materializer=partial(_outcomes, source=teacher),
            target_lineage={
                "contract": recipe.target_policy.fit_teacher_contract or "formal_mintrade40_seed_historical_plus_p16_v1",
                "seed": 42, "source_sha256": "a" * 64,
                "first_effective_date": str(days[0].date()), "teacher_periods": [],
                "value_field": "raw_net_pnl",
                "target_kind": recipe.target_policy.fit_target_kind or "formal_seed_mintrade_fullcycle_broker_net_pnl",
            },
        )
        # The public loader's new optional input is process-local, not persisted.
        import inspect
        if "target_teacher" in inspect.signature(owner.load_daily_universal_ranker_data).parameters:
            kwargs["target_teacher"] = teacher

    with ExitStack() as stack:
        for name, replacement in {
            "load_validated_dataset_bundle": lambda *a, **kw: (
                {"dataset": "full", "source_data_date_range": {"end": str(days[-1].date())}},
                None, None, None, pd.DataFrame()),
            "resolve_breakout_quality_outer_policy": lambda *a, **kw: {
                "effective_oos_end_date": str(days[-1].date()), "oos_start_date": "2019-01-01"},
            "discover_dataset_csv_inputs": lambda *a, **kw: (inputs, []),
            "load_dataset_frame": read,
            "BREAKOUT_QUALITY_DAILY_REAL_PATH_WORKERS": workers,
            "research_ticker_workers": lambda n: n,
            # Worker boundary uses the same futures interface; pickle is checked separately.
            "ProcessPoolExecutor": ThreadPoolExecutor,
        }.items():
            stack.enter_context(patch.object(owner, name, replacement))
        stack.enter_context(patch.object(daily_trade40_target, "collect_canonical_exchange_sessions", return_value=days))
        if context:
            model = replace(owner.get_model_spec("inception_time_v1"), use_dataset_context=True)
            descriptor = SimpleNamespace(has_capability=lambda name: name == "trade_decision_context")
            stack.enter_context(patch.object(owner, "get_model_spec", return_value=model))
            stack.enter_context(patch.object(owner, "get_architecture_descriptor", return_value=descriptor))
            from filters.breakout_quality import trade_decision_context
            def materialize(*, source, positions, **kw):
                if source is not teacher:
                    raise AssertionError("teacher identity changed")
                return np.full((len(positions), len(trade_decision_context.CONTEXT_FEATURES)), 2.0, dtype=np.float32)
            stack.enter_context(patch.object(trade_decision_context, "build_trade_decision_context", side_effect=materialize))
        return owner.load_daily_universal_ranker_data(**kwargs)


class DailySamplePipelineTests(unittest.TestCase):
    def assert_same_bundle(self, left, right):
        pd.testing.assert_frame_equal(left.group_table, right.group_table)
        for name in ("labels", "event_group_index", "group_context", "raw_target", "target_valid"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)
        self.assertEqual(left.summary, right.summary)
        self.assertEqual(left.target_manifest, right.target_manifest)
        np.testing.assert_array_equal(left.feature_bank[:5], right.feature_bank[:5])

    def test_serial_parallel_preserve_valid_prefix_and_inference_only_ticker_order(self):
        serial = build_isolated_daily_bundle()
        parallel = build_isolated_daily_bundle(workers=2)
        self.assert_same_bundle(serial, parallel)
        count = int(serial.target_valid.sum())
        self.assertTrue(serial.target_valid[:count].all())
        self.assertFalse(serial.target_valid[count:].any())
        self.assertEqual(list(dict.fromkeys(serial.group_table.ticker)), ["2330", "2317", "NULL"])
        self.assertTrue(serial.group_table.loc[serial.group_table.ticker == "NULL", "group_index"].min() >= count)

    def test_bad_input_reasons_are_equal_and_do_not_change_legal_samples(self):
        clean = build_isolated_daily_bundle()
        with self.assertLogs(loader.__name__, level="INFO") as serial_logs:
            serial = build_isolated_daily_bundle(failures=True)
        with self.assertLogs(loader.__name__, level="INFO") as parallel_logs:
            parallel = build_isolated_daily_bundle(failures=True, workers=2)
        self.assert_same_bundle(serial, parallel)
        self.assertEqual(serial_logs.output, parallel_logs.output)
        self.assertIn("ticker=BROKEN", serial_logs.output[0])
        self.assertIn("exception=ValueError", serial_logs.output[0])
        self.assertIn("probe malformed OHLCV", serial_logs.output[0])
        self.assertIn("reason=insufficient_history", serial_logs.output[1])
        self.assertEqual(serial.summary["skipped_ticker_count"], 2)
        pd.testing.assert_frame_equal(clean.group_table, serial.group_table)
        self.assertEqual(clean.target_manifest, serial.target_manifest)

    def test_exclusions_keep_cause_and_are_spawn_serializable(self):
        for error in (OSError("io"), UnicodeDecodeError("utf8", b"\xff", 0, 1, "decode"),
                      ValueError("format"), KeyError("Close"), TypeError("type")):
            with self.subTest(error=type(error).__name__), patch.object(loader, "load_dataset_frame", side_effect=error):
                outcome = loader._read_daily_ticker("2330", Path("unused"), min_rows=1)
                self.assertIsNone(outcome.frame)
                self.assertEqual(outcome.exclusion.exception_type, type(error).__name__)
                self.assertEqual(outcome.exclusion.message, str(error))
                self.assertEqual(pickle.loads(pickle.dumps(outcome)), outcome)

    def test_explicit_teacher_is_independent_of_callable_representation(self):
        architecture = SimpleNamespace(has_capability=lambda key: key == "trade_decision_context")
        teacher = object()
        with patch("filters.breakout_quality.trade_decision_context.build_trade_decision_context") as materialize:
            bound = contexts.bind_daily_model_input_context(architecture, trade_path_mode=True, target_teacher=teacher)
            bound.materialize(ticker="2330")
            materialize.assert_called_once_with(source=teacher, ticker="2330")
        with self.assertRaisesRegex(ValueError, "canonical target teacher"):
            contexts.bind_daily_model_input_context(architecture, trade_path_mode=True, target_teacher=None)

    def test_input_context_preserves_target_population_and_manifest_projection(self):
        plain = build_isolated_daily_bundle()
        contextual = build_isolated_daily_bundle(context=True)
        np.testing.assert_array_equal(plain.target_valid, contextual.target_valid)
        np.testing.assert_array_equal(plain.raw_target, contextual.raw_target)
        pd.testing.assert_frame_equal(plain.group_table, contextual.group_table)
        self.assertEqual(plain.target_manifest["target_contract"], contextual.target_manifest["target_contract"])
        for name in ("input_decision_context", "context_features", "pair_weight_context_features"):
            self.assertEqual(contextual.summary[name], contextual.target_manifest[name])
        self.assertIsNot(contextual.summary["context_features"], contextual.target_manifest["context_features"])
        np.testing.assert_array_equal(contextual.group_context, 2.0)


if __name__ == "__main__":
    unittest.main()

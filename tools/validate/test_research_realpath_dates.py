"""Full-output parity and exact source pins for RealPath date indexing."""
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from filters.breakout_quality import daily_trade40_target as owner
from core.research_execution_equivalence import (
    reviewed_calculation_sha256, reviewed_execution_source_sha256,
)
from services.research import oracle_artifact_contract as contracts
from tools.validate.synthetic_daily_trade40_cases import fixture


ROOT = Path(__file__).resolve().parents[2]
RELATIVE = "filters/breakout_quality/daily_trade40_target.py"
BASELINE_SHA = "d716d6007511d42f19388d5e0535b145c275adebb7563cfe06bd6e0605f32883"
DATE_PREPARATION = (
    "    # Reuse the same normalized Timestamp objects during overlapping paths.\n"
    "    # Keep the index for vectorized alignment and signal-origin lookup.\n"
    "    date_values = tuple(dates)\n"
)


def indexed_reference():
    """Recover the independently pinned predecessor, not a second algorithm."""
    source = (ROOT / RELATIVE).read_text(encoding="utf-8")
    if source.count(DATE_PREPARATION) != 1:
        raise AssertionError("RealPath date preparation no longer matches its proof")
    source = source.replace(DATE_PREPARATION, "").replace("date_values[", "dates[")
    if sha256(source.encode("utf-8")).hexdigest() != BASELINE_SHA:
        raise AssertionError("Reconstructed RealPath reference is not the reviewed predecessor")
    import ast
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "compute_counterfactual_trade40_batch")
    namespace = dict(vars(owner))
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 "<pinned-indexed-realpath-reference>", "exec"), namespace)
    return namespace[function.name]


class RealPathDateIndexingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = staticmethod(indexed_reference())

    def assert_batch_equal(self, actual, expected):
        for field in fields(expected):
            left, right = getattr(actual, field.name), getattr(expected, field.name)
            with self.subTest(field=field.name):
                if isinstance(right, np.ndarray):
                    self.assertEqual(left.dtype, right.dtype)
                    np.testing.assert_array_equal(left, right)
                    self.assertEqual(left.tobytes(), right.tobytes())
                else:
                    self.assertEqual(left, right)

    def test_complete_outcomes_equal_across_horizons_cutoffs_and_signal_resets(self):
        for rising in (False, True):
            days, frame, _, period = fixture(count=150, rising=rising)
            # A real schedule boundary exercises both origin and active dates.
            schedule = (replace(period, end=days[79]), replace(period, start=days[80]))
            for horizon in (1, 40, 60, None):
                for reset in (None, days[80]):
                    for include_pnl in (False, True):
                        with self.subTest(rising=rising, horizon=horizon,
                                          reset=reset, include_pnl=include_pnl):
                            kwargs = dict(
                                ticker="2330", schedule=schedule, exchange_calendar=days,
                                horizon_sessions=horizon, research_cutoff=days[130],
                                include_net_pnl=include_pnl, signal_state_reset_date=reset,
                            )
                            positions = np.array([0, 45, 48, 49, 74, 75, 79, 80,
                                                  89, 90, 118, 129, 130, 149])
                            expected = self.reference(frame, positions, **kwargs)
                            self.assert_batch_equal(
                                owner.compute_counterfactual_trade40_batch(frame, positions, **kwargs),
                                expected,
                            )

    def test_episode_inventory_and_empty_cohort_equal(self):
        days, frame, _, period = fixture(count=150)
        for positions in (np.array([], dtype=np.int64), np.arange(45, 120, dtype=np.int64)):
            for keys_only in (False, True):
                expected_keys, actual_keys = {}, {}
                kwargs = dict(ticker="2330", schedule=(period,), exchange_calendar=days,
                              horizon_sessions=None, research_cutoff=days[130],
                              include_net_pnl=True, episode_keys_only=keys_only)
                expected = self.reference(frame, positions, episode_keys_out=expected_keys, **kwargs)
                actual = owner.compute_counterfactual_trade40_batch(
                    frame, positions, episode_keys_out=actual_keys, **kwargs)
                self.assert_batch_equal(actual, expected)
                self.assertEqual(actual_keys, expected_keys)

    def test_invalid_positions_keep_the_same_failure(self):
        days, frame, _, period = fixture(count=150)
        for positions in (np.array([-1]), np.array([150]), np.array([[48]])):
            errors = []
            for function in (self.reference, owner.compute_counterfactual_trade40_batch):
                with self.assertRaises(ValueError) as raised:
                    function(frame, positions, ticker="2330", schedule=(period,),
                             exchange_calendar=days)
                errors.append(str(raised.exception))
            self.assertEqual(*errors)

    def test_exact_source_certificate_preserves_only_reviewed_bytes(self):
        raw = (ROOT / RELATIVE).read_bytes().replace(b"\r\n", b"\n")
        for content, expected in (
            (raw, "4df90ba6d95a22cd0a78845141dc6bfb91683e7fa2ecc612cafa6f66e6bd3370"),
            (raw.replace(b"\n", b"\r\n"),
             "e569741dc258f637757d42cdd3febd39818f2136224a23b4fc26d7c505d4ada1"),
        ):
            actual = sha256(content).hexdigest()
            self.assertEqual(reviewed_execution_source_sha256(RELATIVE, actual), expected)
            changed = sha256(content + b"\n_UNKNOWN_REALPATH_EDIT = True\n").hexdigest()
            self.assertEqual(reviewed_execution_source_sha256(RELATIVE, changed), changed)

    def test_calculation_certificate_is_pinned_to_source_and_calculation(self):
        raw = (ROOT / RELATIVE).read_bytes().replace(b"\r\n", b"\n")
        variants = (
            (None, "7d22c5edbd455bb9715a24be5ae71feab98a2248620d00b9238c731bdf2f5b9e"),
            ({"signal_state_reset_date": None, "include_net_pnl": False},
             "98c80c47584be11046fb2dc562c7184fa2112ec4672e695bad016d967c1e0513"),
            ({"signal_state_reset_date": None, "include_net_pnl": True},
             "d09aa292059cd3d446e52aed5ddd18b78f17121e1d5e9d0ee225b1414c813f73"),
        )
        for bound, expected in variants:
            with patch.object(contracts, "reviewed_calculation_sha256", lambda _r, _f, s: s):
                syntax = contracts.calculation_source(
                    RELATIVE, "compute_counterfactual_trade40_batch", bound_literals=bound)
            for content in (raw, raw.replace(b"\n", b"\r\n")):
                file_sha = sha256(content).hexdigest()
                self.assertEqual(reviewed_calculation_sha256(RELATIVE, file_sha, syntax), expected)
                mutation = sha256(content + b"\n_UNKNOWN_REALPATH_EDIT = True\n").hexdigest()
                self.assertEqual(reviewed_calculation_sha256(RELATIVE, mutation, syntax), syntax)
                self.assertEqual(reviewed_calculation_sha256(RELATIVE, file_sha, "f" * 64), "f" * 64)


if __name__ == "__main__":
    unittest.main()

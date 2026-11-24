"""Regression tests for causal history composition and controlled progression."""
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd

from config.breakout_quality_history import CO_PROFILE, HISTORY_PROFILE
from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from services.breakout_quality.training_history import _prefix_materializer
from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITLabels


class HistoryLearningContractTests(unittest.TestCase):
    def test_only_history_recipe_changes(self):
        a = get_breakout_quality_experiment_profile(CO_PROFILE)
        b = get_breakout_quality_experiment_profile(HISTORY_PROFILE)
        self.assertEqual(a, replace(b, name=a.name))

    def test_prefix_preserves_original_rows_and_order(self):
        dates = pd.bdate_range('2013-12-27', periods=6)
        frame = pd.DataFrame(index=dates)
        calls = []
        def producer(value):
            def materialize(*, frame, positions, **kwargs):
                calls.append((value, tuple(positions)))
                n = len(positions)
                return FormalMinTrade40PITLabels(
                    np.full(n,value,dtype=np.float32), np.ones(n,bool),np.zeros(n,bool),
                    np.ones(n,bool),np.full(n,np.datetime64('2014-02-01')),{},np.full(n,value,dtype=np.float64))
            return materialize
        result = _prefix_materializer(control=producer(11),extended=producer(99),
            boundary='2014-01-01',frame=frame,positions=np.array([0,1,2,3]))
        np.testing.assert_array_equal(result.raw_net_pnl, [99,99,11,11])
        # Labels retain their natural future maturity, including across the join.
        self.assertTrue((result.maturity_dates == np.datetime64('2014-02-01')).all())
        self.assertEqual(calls, [(11,(2,3)),(99,(0,1))])



if __name__ == '__main__':
    unittest.main()

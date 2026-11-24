"""Regression tests for causal history composition and controlled progression."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd

from config.learning_studies import CO_PROFILE, HISTORY_PROFILE, LEARNING_STUDIES
from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from services.breakout_quality.learning_data import _prefix_materializer
from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITLabels
from services.research.learning_study_metrics import paired_rank_metrics, screen_gate, strategy_gate


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

    def test_common_cohort_rejects_label_change_or_missing_stock(self):
        a = pd.DataFrame({'ticker':['a','b'],'date':[pd.Timestamp('2020-01-02')]*2,
                          'score':[1.,0.], 'target':[2.,-1.]})
        b = a.copy(); b.loc[0,'target'] = 3.
        with self.assertRaises(ValueError):
            paired_rank_metrics(a,b)
        with self.assertRaises(ValueError):
            paired_rank_metrics(a,a.iloc[:1])

    def test_ties_and_competition_days(self):
        a = pd.DataFrame({'ticker':[str(i) for i in range(12)], 'date':[pd.Timestamp('2020-01-02')]*12,
                          'score':[0.]*12, 'target':list(range(12))})
        result = paired_rank_metrics(a,a)
        self.assertEqual(result['a']['pair_accuracy'],.5)
        self.assertEqual(result['a']['competition_days'],1)
        small = paired_rank_metrics(a.iloc[:10],a.iloc[:10])
        self.assertIsNone(small['a']['top10_lift'])

    def test_screen_requires_all_years_and_no_pair_decline(self):
        def row(delta,pair=0):
            return {'a':{'top10_lift':1.,'pair_accuracy':.51},
                    'b':{'top10_lift':1.+delta,'pair_accuracy':.51+pair}}
        years = {'2018':row(1),'2019':row(1),'2020':row(-.1)}
        gate = dict(expected_years=(2018,2019,2020),minimum_positive_years=2)
        self.assertEqual(screen_gate(years,**gate)['status'],'PASS')
        self.assertEqual(screen_gate({'2018':row(1)},**gate)['status'],'INCOMPLETE')
        years['2020'] = row(-.1,-.02)
        self.assertEqual(screen_gate(years,**gate)['status'],'REJECT')

    def test_strategy_gate_does_not_reward_one_lucky_seed(self):
        rows = [{'a':{'romd':10.,'mdd':15.},'b':{'romd':d,'mdd':15.}} for d in (30,9,9,9)]
        self.assertEqual(strategy_gate(rows,minimum_wins=3)['status'],'REJECT')
        self.assertEqual(strategy_gate([],minimum_wins=3)['status'],'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()

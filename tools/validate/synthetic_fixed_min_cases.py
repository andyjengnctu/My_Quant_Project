"""Fixed Min scientific contract tests (no market data or optimization required).

Run: python -m unittest tools.validate.synthetic_fixed_min_cases -q
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from core.fixed_min_policy import (
    FIXED_MIN_ID, FIXED_MIN_TARGET, fixed_min_params, fixed_min_signature,
    fixed_min_source, fixed_min_artifact_path, ensure_fixed_min_param_artifact,
)
from core.strategy_compare_policy import get_strategy_comparison_settings
from core.breakout_quality_policy import get_breakout_quality_model_test_settings
from services.breakout_quality.formal_mintrade40_pit_source import (
    verify_formal_mintrade40_pit_source, slice_formal_mintrade40_pit_source_for_fold,
    materialize_formal_mintrade40_pit_labels, FormalMinTrade40PITBlocked,
)
from services.breakout_quality.trade_path_target_source import loader_kwargs_for_trade_path

PROFILE = "fixed_mintrade_fullcycle_netpnl_full_list_ndcg_pairwise"


class FixedMinContractTests(TestCase):
    def test_one_set_of_constants_without_year_seed_or_optimizer_drift(self):
        p = fixed_min_params()
        self.assertEqual((p.high_len, p.atr_len, p.atr_buy_tol,
                          p.atr_times_init, p.atr_times_trail, p.tp_percent),
                         (200, 14, 1.5, 2.0, 3.5, 0.0))
        self.assertFalse(p.use_breakout_quality_filter)
        self.assertFalse(p.use_breakout_quality_ranking)
        first, second = fixed_min_source(42), fixed_min_source(1234683984)
        self.assertEqual(first.source_id, FIXED_MIN_ID)
        self.assertEqual(first.param_sha256, second.param_sha256)
        self.assertEqual(first.schedule[0].params_signature, fixed_min_signature())
        self.assertEqual(first.schedule[0].params_signature, second.schedule[0].params_signature)
        self.assertEqual(len(first.schedule), 1)
        for year in (2013, 2018, 2020, 2021, 2026):
            self.assertLessEqual(first.schedule[0].start, pd.Timestamp(f'{year}-07-01'))
            self.assertIsNone(first.schedule[0].end)
        for cutoff in ('2013-07-01', '2017-06-30', '2020-12-31', '2026-03-02'):
            truncated, digest = slice_formal_mintrade40_pit_source_for_fold(
                first, fit_cutoff=cutoff,
            )
            self.assertEqual(len(truncated.schedule), 1)
            self.assertEqual(truncated.schedule[0].end, pd.Timestamp(cutoff))
            self.assertEqual(len(digest), 64)
        self.assertNotEqual(FIXED_MIN_TARGET, 'MinTrade_NetPnL')

    def test_label_owner_and_training_pit_maturity(self):
        args = loader_kwargs_for_trade_path(
            experiment_profile=PROFILE, project_root=Path('/tmp/fixed_min_synthetic'),
            seed=42, authorized_build=False,
        )
        self.assertEqual(args['target_lineage']['contract'], FIXED_MIN_ID)
        self.assertEqual(args['target_lineage']['teacher_periods'][0]['end'], None)
        self.assertEqual(args['target_lineage']['horizon_mode'], 'full_trade_cycle')
        self.assertEqual(args['target_materializer'].keywords['outcome_metric'], 'net_pnl')
        self.assertIsNone(args['target_materializer'].keywords['horizon_sessions'])
        source = args['target_materializer'].keywords['source']
        dates = pd.date_range('2013-04-01', periods=270, freq='B')
        frame = pd.DataFrame({
            'Open': 100 + .25 * np.arange(len(dates)),
            'High': 102 + .25 * np.arange(len(dates)),
            'Low': 98 + .25 * np.arange(len(dates)),
            'Close': 100 + .25 * np.arange(len(dates)),
            'Volume': np.full(len(dates), 1_000_000),
        }, index=dates)
        evidence = verify_formal_mintrade40_pit_source(
            source, expected_seed=42,
            information_dates=[dates[230]], executable_dates=[dates[231]],
            fit_cutoff=dates[250],
        )
        self.assertEqual(evidence['sample_count'], 1)
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked, 'Post-FIT-cutoff'):
            verify_formal_mintrade40_pit_source(
                source, expected_seed=42,
                information_dates=[dates[251]], executable_dates=[dates[252]],
                fit_cutoff=dates[250],
            )
        labels = materialize_formal_mintrade40_pit_labels(
            source, expected_seed=42, frame=frame,
            positions=np.array([215, 244], dtype=np.int64), ticker='2330',
            exchange_calendar=dates, fit_cutoff=dates[250],
            horizon_sessions=None, outcome_metric='net_pnl',
        )
        self.assertEqual(len(labels.raw_net_pnl), 2)
        self.assertFalse(bool(labels.completed[1]))
        self.assertTrue(np.isnan(labels.raw_net_pnl[1]))
        self.assertEqual(labels.preflight['sample_count'], 2)

    def test_strategy_and_model_reuse_same_parameter_source(self):
        setting = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        enabled = {a.arm_id: a for a in setting.enabled_arms}
        self.assertTrue({'C139','C140','C141'}.issubset(enabled))
        for key in ('C139','C140','C141'):
            self.assertEqual(enabled[key].param_source, FIXED_MIN_ID)
            self.assertEqual(enabled[key].param_policy, 'base-finalist-best')
            self.assertEqual(enabled[key].rule_policy, 'all_off')
            self.assertNotEqual(enabled[key].param_source,
                                enabled['C96'].param_source)
        self.assertFalse(enabled['C139'].dl_enabled)
        self.assertTrue(enabled['C140'].dl_enabled)
        self.assertTrue(enabled['C141'].dl_enabled)
        self.assertEqual(enabled['C140'].dl_id, 'CONT13CV_ROLL')
        self.assertEqual(enabled['C141'].dl_id, 'CONT13CV_ROLL')
        self.assertTrue(enabled['C140'].dl_runtime_options['preserve_k_r0'])
        self.assertFalse(enabled['C141'].dl_runtime_options['preserve_k'])
        self.assertFalse(enabled['C141'].dl_runtime_options['preserve_r0'])
        selected = dict(get_breakout_quality_model_test_settings().model_profiles)
        self.assertEqual(selected['MR-13CV'], PROFILE)
        from config.breakout_quality import BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS
        self.assertIsNone(BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS)

    def test_immutable_artifact_reuse_and_drift_rejection(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = ensure_fixed_min_param_artifact(root)
            self.assertEqual(path, fixed_min_artifact_path(root))
            self.assertEqual(path.read_bytes(), ensure_fixed_min_param_artifact(root).read_bytes())
            payload = __import__('json').loads(path.read_text(encoding='utf-8'))
            self.assertEqual(len(payload['params_by_effective_date']), 1)
            self.assertEqual(payload['params_by_effective_date']['1900-01-01']['high_len'], 200)
            payload['params_by_effective_date']['1900-01-01']['high_len'] = 201
            path.write_text(__import__('json').dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from canonical'):
                ensure_fixed_min_param_artifact(root)

    def test_tampered_fixed_source_rejected(self):
        source = fixed_min_source(42)
        invalid = replace(source, schedule=(replace(
            source.schedule[0], params=replace(fixed_min_params(), high_len=201)),))
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked, 'identity drift'):
            slice_formal_mintrade40_pit_source_for_fold(
                invalid, fit_cutoff='2020-12-31',
            )


def validate_fixed_min_end_to_end_contract_case(_base_params):
    """Formal reduced-consistency coverage for the one-source Fixed Min treatment."""
    from .checks import bind_synthetic_case, raises_expected
    from services.research.strategy_compare_preparation_status import _validate_param_artifact
    results, summary, check, check_true = bind_synthetic_case(
        'FIXED_MIN', 'Fixed Min SSOT + PIT + Strategy + REUSE',
    )
    setting = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
    enabled = {arm.arm_id: arm for arm in setting.enabled_arms}
    check('registered fixed Min identities', ['C139', 'C140', 'C141'],
          sorted(set(enabled) & {'C139', 'C140', 'C141'}))
    check('shared fixed Min parameter source', [FIXED_MIN_ID] * 3,
          [enabled[name].param_source for name in ('C139', 'C140', 'C141')])
    check('selector Exact vs Direct', [True, False],
          [bool(enabled['C140'].dl_runtime_options.get('preserve_k_r0')),
           bool(enabled['C141'].dl_runtime_options.get('preserve_k'))])
    check('selected training target', PROFILE,
          dict(get_breakout_quality_model_test_settings().model_profiles).get('MR-13CV'))
    from config.breakout_quality import BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS
    check_true('expanding FIT history', BREAKOUT_QUALITY_POINT_IN_TIME_TRAIN_WINDOW_MONTHS is None)
    with TemporaryDirectory() as temp:
        path = ensure_fixed_min_param_artifact(Path(temp))
        contract = setting.parameter_sources[FIXED_MIN_ID].artifact_contract
        ready, reason, _ = _validate_param_artifact(
            path, param_policy='base-finalist-best',
            comparison_start='2021-01-01', comparison_end='2026-03-02',
            artifact_contract=dict(contract or {}),
        )
        check('valid source reusable', (True, 'READY'), (ready, reason))
        import json
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['params_by_effective_date']['1900-01-01']['high_len'] = 201
        path.write_text(json.dumps(payload), encoding='utf-8')
        ready, reason, _ = _validate_param_artifact(
            path, param_policy='base-finalist-best',
            comparison_start='2021-01-01', comparison_end='2026-03-02',
            artifact_contract=dict(contract or {}),
        )
        check('drifted source never REUSE', (False, 'FIXED_MIN_CONSTANT_DRIFT'),
              (ready, reason))
    source = fixed_min_source(42)
    check('one all-history schedule', 1, len(source.schedule))
    check('1900 effective date', '1900-01-01',
          str(source.schedule[0].start.date()))
    check_true('immutable scientific signature',
               source.schedule[0].params_signature == fixed_min_signature())
    return results, summary

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
    fixed_min_artifact_path, ensure_fixed_min_param_artifact,
)
from services.breakout_quality.fixed_min_source import fixed_min_source
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

    def test_fixed_min_oracle_replay_accepts_absent_universe_boundary(self):
        """No Optimizer boundary must not crash the real ensemble replay entry."""
        from unittest.mock import patch
        from core.data_utils import get_required_min_rows
        from core.fixed_min_policy import fixed_min_param_payload
        from core.portfolio_param_runtime import build_active_param_ensemble_objects_from_payload
        from core.raw_universe_contract import resolve_raw_universe_required_min_rows
        from services import portfolio_replay as replay

        payload = fixed_min_param_payload()
        self.assertIsNone(resolve_raw_universe_required_min_rows(payload))
        records = build_active_param_ensemble_objects_from_payload(payload)
        self.assertEqual(len(records), 1)
        self.assertEqual(replay._resolve_required_min_rows_for_raw_universe(
            [fixed_min_params()], raw_universe_required_min_rows=None,
        ), get_required_min_rows(fixed_min_params()))

        def key_with(boundary):
            return replay._replay_preparation_key(
                '/tmp/fixed-min-oracle', records, ensemble=True,
                raw_universe_required_min_rows=boundary, keep_trade_logs=False,
                use_prepared_cache=False, write_prepared_cache=False,
            )

        no_optimizer = key_with(None)
        p16_pinned = key_with(260)
        self.assertIsNone(no_optimizer[3])
        self.assertEqual(p16_pinned[3], 260)
        self.assertNotEqual(no_optimizer, p16_pinned,
                            'P16 and Fixed Min must not share prepared-market cache')
        self.assertEqual(key_with('260'), p16_pinned)
        for invalid in (True, 0, -1, 260.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                key_with(invalid)

        market = {'1900-01-01': [{
            'sorted_dates': [pd.Timestamp('2021-01-04'), pd.Timestamp('2021-01-05')],
            'all_dfs_fast': {}, 'all_trade_logs': {}, 'all_pit_stats_index': {},
            'prep_wall_sec': 0.0,
        }]}
        cache = {}
        with (
            patch.object(replay, '_load_contexts_for_active_ensemble_schedule',
                         return_value=market) as build_market,
            patch.object(replay, '_apply_active_single_stock_score_stats'),
            patch.object(replay, 'run_portfolio_timeline',
                         side_effect=['C139-timeline', 'C140-timeline']) as portfolio,
            patch.object(replay, 'append_portfolio_replay_profile',
                         side_effect=lambda value, profile: (value, profile)),
        ):
            outputs = [replay.run_portfolio_simulation_with_param_ensemble(
                '/tmp/fixed-min-oracle', payload, start_year=2021,
                start_date='2021-01-04', end_date='2021-01-05', verbose=False,
                replay_preparation_cache=cache,
            ) for _ in range(2)]
        self.assertEqual(build_market.call_count, 1,
                         'Same Fixed Min preparation must be reused within a session')
        self.assertEqual(build_market.call_args.kwargs['raw_universe_required_min_rows'], None)
        self.assertEqual(portfolio.call_count, 2,
                         'Same market preparation must not reuse a portfolio result')
        self.assertEqual([output[0] for output in outputs],
                         ['C139-timeline', 'C140-timeline'])
        self.assertTrue(all(out[1]['raw_universe_required_min_rows'] is None
                            for out in outputs))

    def test_fixed_min_oracle_receipt_identity_includes_portfolio_engine(self):
        from unittest.mock import patch
        from services.audit import fixed_min_oracle_scope as fixed
        from services.research.oracle_artifact_contract import portfolio_candidate_execution_sha256

        owner = portfolio_candidate_execution_sha256()
        self.assertEqual(len(owner), 64)
        digest = fixed._source_sha()
        with patch.object(fixed, 'portfolio_candidate_execution_sha256', return_value='f' * 64):
            changed = fixed._source_sha()
        self.assertNotEqual(digest, changed,
                            'Fixed Oracle must bind the shared Portfolio candidate/capital owner')

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

    def test_fixed_min_robustness_uses_one_truth_for_all_seeds(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from core.fixed_min_policy import (
            fixed_min_producer_manifest_path,
            inspect_fixed_min_robustness_parameter_artifact,
            ensure_fixed_min_robustness_parameter_artifact,
        )
        from services.research import strategy_multi_seed_robustness as robustness_owner

        settings = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        arms = tuple(settings.arms[k] for k in ('C139', 'C140', 'C141'))
        benchmark = SimpleNamespace(benchmark_id='fixed-min-test', resolved_seeds=(17, 31))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = fixed_min_artifact_path(root)
            manifest = fixed_min_producer_manifest_path(root)
            with (
                patch.object(robustness_owner, 'PROJECT_ROOT', root),
                patch.object(robustness_owner,
                             'ensure_fixed_min_robustness_parameter_artifact',
                             wraps=ensure_fixed_min_robustness_parameter_artifact) as constant_producer,
                patch.object(robustness_owner,
                             'ensure_robustness_benchmark_strategy_parameter_artifact',
                             side_effect=AssertionError('Optimizer must not run')),
            ):
                rows, blockers, plans = robustness_owner._benchmark_parameter_plan_rows(
                    settings=settings, robustness=benchmark, benchmark_arms=arms,
                    comparison_start='2021-01-01', comparison_end='2026-03-02',
                )
                self.assertFalse(blockers)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][0], 'BUILD')
                self.assertIn(':constant=all:', rows[0][1])
                self.assertEqual(len(plans), 6)
                self.assertEqual({str(b['path']) for b in plans.values()}, {str(path)})
                bound = robustness_owner._prepare_benchmark_strategy_parameter_artifacts(
                    settings=settings, robustness=benchmark, benchmark_arms=arms,
                    comparison_start='2021-01-01', comparison_end='2026-03-02',
                )
                self.assertEqual(constant_producer.call_count, 1)
                self.assertEqual(len(bound), 6)
                self.assertEqual({str(b['path']) for b in bound.values()}, {str(path)})
                self.assertEqual({str(b['manifest_path']) for b in bound.values()}, {str(manifest)})
                self.assertEqual(len({b['sha256'] for b in bound.values()}), 1)
                self.assertEqual({b['seed'] for b in bound.values()}, {17, 31})
                self.assertEqual(inspect_fixed_min_robustness_parameter_artifact(root)['action'], 'REUSE')
                rows, blockers, _ = robustness_owner._benchmark_parameter_plan_rows(
                    settings=settings, robustness=benchmark, benchmark_arms=arms,
                    comparison_start='2021-01-01', comparison_end='2026-03-02',
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][0], 'REUSE')
                self.assertFalse(blockers)
                import json
                payload = json.loads(path.read_text(encoding='utf-8'))
                payload['params_by_effective_date']['1900-01-01']['high_len'] = 201
                path.write_text(json.dumps(payload), encoding='utf-8')
                rows, blockers, _ = robustness_owner._benchmark_parameter_plan_rows(
                    settings=settings, robustness=benchmark, benchmark_arms=arms,
                    comparison_start='2021-01-01', comparison_end='2026-03-02',
                )
                self.assertEqual(rows[0][0], 'BLOCKED')
                self.assertTrue(blockers)
                with self.assertRaisesRegex(ValueError, 'source drift'):
                    ensure_fixed_min_robustness_parameter_artifact(root)

    def test_existing_fixed_min_truth_repairs_only_missing_manifest(self):
        from core.fixed_min_policy import (
            fixed_min_producer_manifest_path,
            inspect_fixed_min_robustness_parameter_artifact,
            ensure_fixed_min_robustness_parameter_artifact,
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = ensure_fixed_min_param_artifact(root)
            original = path.read_bytes()
            self.assertEqual(
                inspect_fixed_min_robustness_parameter_artifact(root),
                {'action': 'REPAIR', 'status': 'FIXED_MIN_MANIFEST_MISSING'},
            )
            published = ensure_fixed_min_robustness_parameter_artifact(root)
            self.assertEqual(published['action'], 'REPAIR')
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(fixed_min_producer_manifest_path(root).is_file())
            self.assertEqual(ensure_fixed_min_robustness_parameter_artifact(root)['action'], 'REUSE')

    def test_fixed_min_manifest_exclusive_create_race_is_verified(self):
        """Concurrent identical publication may REUSE; altered bytes must fail closed."""
        import json
        from unittest.mock import patch
        from core.fixed_min_policy import (
            fixed_min_producer_manifest_path,
            inspect_fixed_min_robustness_parameter_artifact,
            ensure_fixed_min_robustness_parameter_artifact,
            _fixed_min_producer_manifest_payload,
        )
        original_open = Path.open
        for tamper in (False, True):
            with self.subTest(tamper=tamper), TemporaryDirectory() as temp:
                root = Path(temp)
                source_path = ensure_fixed_min_param_artifact(root)
                manifest_path = fixed_min_producer_manifest_path(root)
                payload = _fixed_min_producer_manifest_payload(source_path)
                if tamper:
                    payload["parameter_file_sha256"] = "0" * 64

                def racing_open(path, mode="r", *args, **kwargs):
                    if path == manifest_path and mode == "x":
                        # Another publisher won between exists() and open('x').
                        with original_open(path, "w", encoding="utf-8") as f:
                            json.dump(payload, f)
                        raise FileExistsError("simulated concurrent manifest creation")
                    return original_open(path, mode, *args, **kwargs)

                with patch.object(Path, "open", racing_open):
                    if tamper:
                        with self.assertRaisesRegex(ValueError, "concurrent manifest publication invalid"):
                            ensure_fixed_min_robustness_parameter_artifact(root)
                    else:
                        result = ensure_fixed_min_robustness_parameter_artifact(root)
                        self.assertEqual(result["action"], "REPAIR")
                self.assertEqual(
                    inspect_fixed_min_robustness_parameter_artifact(root)["action"],
                    "BLOCKED" if tamper else "REUSE",
                )

    def test_fixed_min_robustness_manifest_tamper_rejected(self):
        from core.fixed_min_policy import (
            fixed_min_producer_manifest_path,
            inspect_fixed_min_robustness_parameter_artifact,
            ensure_fixed_min_robustness_parameter_artifact,
        )
        import json
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ready = ensure_fixed_min_robustness_parameter_artifact(root)
            self.assertEqual(ready['action'], 'BUILD')
            self.assertEqual(ensure_fixed_min_robustness_parameter_artifact(root)['action'], 'REUSE')
            manifest_path = fixed_min_producer_manifest_path(root)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['parameter_file_sha256'] = '0' * 64
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            state = inspect_fixed_min_robustness_parameter_artifact(root)
            self.assertEqual(state, {'action': 'BLOCKED', 'status': 'FIXED_MIN_MANIFEST_DRIFT'})
            with self.assertRaisesRegex(ValueError, 'source drift'):
                ensure_fixed_min_robustness_parameter_artifact(root)

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
    # Robustness must not reinterpret timeless parameters as seed-fitted Optimizer output.
    from unittest.mock import patch
    from types import SimpleNamespace
    from core.fixed_min_policy import (
        inspect_fixed_min_robustness_parameter_artifact,
        ensure_fixed_min_robustness_parameter_artifact,
    )
    from services.research import strategy_multi_seed_robustness as robustness_owner
    with TemporaryDirectory() as temp:
        root = Path(temp)
        arms = tuple(enabled[k] for k in ('C139', 'C140', 'C141'))
        bench = SimpleNamespace(benchmark_id='fixed-min-synthetic', resolved_seeds=(17, 31))
        with patch.object(robustness_owner, 'PROJECT_ROOT', root):
            rows, blocked, bindings = robustness_owner._benchmark_parameter_plan_rows(
                settings=setting, robustness=bench, benchmark_arms=arms,
                comparison_start='2021-01-01', comparison_end='2026-03-02',
            )
            check('fixed robustness one unbuilt constant source', ('BUILD', 1, 0),
                  (rows[0][0], len(rows), len(blocked)))
            check_true('all seeded arms refer to one fixed parameter truth',
                       len(bindings) == 6 and len({str(b['path']) for b in bindings.values()}) == 1)
            artifact = ensure_fixed_min_param_artifact(root)
            rows, blocked, _ = robustness_owner._benchmark_parameter_plan_rows(
                settings=setting, robustness=bench, benchmark_arms=arms,
                comparison_start='2021-01-01', comparison_end='2026-03-02',
            )
            check('fixed robustness missing manifest is repair only', ('REPAIR', 0),
                  (rows[0][0], len(blocked)))
            before = artifact.read_bytes()
            ensure_fixed_min_robustness_parameter_artifact(root)
            check_true('fixed robustness REUSE does not replace constants',
                       artifact.read_bytes() == before
                       and inspect_fixed_min_robustness_parameter_artifact(root)['action'] == 'REUSE')
    source = fixed_min_source(42)
    check('one all-history schedule', 1, len(source.schedule))
    check('1900 effective date', '1900-01-01',
          str(source.schedule[0].start.date()))
    check_true('immutable scientific signature',
               source.schedule[0].params_signature == fixed_min_signature())
    return results, summary

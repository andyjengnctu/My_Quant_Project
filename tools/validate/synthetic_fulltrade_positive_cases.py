"""Independent primitive invariants for the Full/P17 NetPnL Positive capability.

Synthetic, temporary, never a live optimizer run or formal Test Suite.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

from core.audit_policy import get_reusable_audit_definitions
from config.breakout_quality import BREAKOUT_QUALITY_MODEL_TEST_PROFILES
from core.breakout_quality_runtime import get_continuous_ranker_training_policy
from core.strategy_compare_policy import get_strategy_multi_seed_robustness_settings
from core.file_integrity import canonical_json_sha256
import json
from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from core.strategy_compare_policy import get_strategy_comparison_settings
from services.audit.fulltrade_oracle_truth import (
    build_fulltrade_oracle_scores, truth_producer_fingerprint,
)
from services.audit.fullcycle_positive_truth import load_or_build_positive_truth
from services.breakout_quality.formal_fulltrade_training_source import (
    FULL_SOURCE_ID, FULL_BASELINE_ARM, _full_execution_schedule,
    historical_params_path, inspect_full_history, prepare_full_history,
)
from services.breakout_quality.trade_path_target_source import registered_trade_path_source
from services.research.oracle_artifact_contract import source_calculation_contract
from tools.validate.synthetic_daily_trade40_oracle_cases import PROFILE, source_fixture


class FullTradePositiveContracts(TestCase):
    def test_independent_rank_positive_and_p17_strategy(self):
        models = dict(BREAKOUT_QUALITY_MODEL_TEST_PROFILES)
        for model in ("MR-13CZ", "MR-13DA"):
            self.assertIsNotNone(get_breakout_quality_experiment_profile(models[model]))
        self.assertNotEqual(models["MR-13CZ"], models["MR-13DA"])
        rank = get_breakout_quality_experiment_profile(models["MR-13CZ"])
        sign = get_breakout_quality_experiment_profile(models["MR-13DA"])
        self.assertEqual(rank.continuous_target_id, sign.continuous_target_id)
        self.assertEqual(get_continuous_ranker_training_policy(rank.training_objective).loss_handler,
                         "single_pairwise")
        policy = get_continuous_ranker_training_policy(sign.training_objective)
        self.assertEqual(policy.loss_handler, "categorical_cross_entropy")
        self.assertEqual(policy.categorical_class_count, 2)
        self.assertEqual(FULL_BASELINE_ARM, "C97")
        s = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        for cid in ("C155", "C156"):
            arm = s.arms[cid]
            self.assertEqual(arm.param_source, s.arms["C97"].param_source)
            self.assertEqual(arm.param_policy, s.arms["C97"].param_policy)
        robust = get_strategy_multi_seed_robustness_settings(s.profile_id)
        self.assertEqual(len(robust.resolved_seeds), 4)
        self.assertTrue({"C155", "C156"}.issubset(set(robust.stochastic_arm_ids)))
        self.assertTrue({"C155", "C156"}.issubset(set(robust.model_seed_sensitive_arm_ids)))
        defs = get_reusable_audit_definitions("breakout_quality")
        source = next(d.source for d in defs if d.audit_type == "target_ground_truth_strategy_romd")
        full = [c for c in source["oracle_comparisons"]
                if str(c["target"]).startswith("FullTrade_NetPnL_Positive")]
        self.assertEqual(len(full), 2)
        self.assertEqual({str(c["strategy_arm_id"]) for c in full}, {"C155", "C156"})

    def test_full_p17_teacher_distinct_from_min_p16(self):
        _, _, _, min_schedule = source_fixture()
        source = tuple(replace(p, source_id=FULL_SOURCE_ID) for p in min_schedule)
        self.assertTrue(all(p.source_id == FULL_SOURCE_ID for p in source))
        # The actual source adapter must re-label as Full without rewriting the
        # sole economic mapper or the original Min/P16 teacher identity.
        records = [dict(effective_date=p.start.strftime("%Y-%m-%d"),
                        effective_end_date=p.end, params_obj=p.params,
                        params_signature=p.params_signature) for p in min_schedule]
        adapted, receipt = _full_execution_schedule(records, seed=693545351, param_sha256="a"*64)
        self.assertEqual(tuple(x.params_signature for x in adapted),
                         tuple(x.params_signature for x in min_schedule))
        self.assertEqual({p.source_id for p in adapted}, {FULL_SOURCE_ID})
        self.assertEqual(receipt["origin"], "formal_full_roos_rolling_per_seed")

    def test_target_capability_is_registered(self):
        for name in ("MR-13CZ", "MR-13DA"):
            profile_id = dict(BREAKOUT_QUALITY_MODEL_TEST_PROFILES)[name]
            source = registered_trade_path_source(profile_id)
            self.assertEqual(source.contract_kind, "formal_seed_fulltrade_fullcycle_netpnl")
            self.assertEqual(len(source.dependencies), 2)

    def test_new_oracle_calculation_has_independent_identity(self):
        self.assertEqual(len(truth_producer_fingerprint()), 64)
        self.assertNotEqual(truth_producer_fingerprint(),
                            __import__("services.audit.daily_trade40_oracle", fromlist=["truth_producer_fingerprint"])
                            .truth_producer_fingerprint("net_pnl"))
        profiles = dict(BREAKOUT_QUALITY_MODEL_TEST_PROFILES)
        new = source_calculation_contract(recipe="daily_trade_path_full_net_pnl",
                profile_id=profiles["MR-13CZ"])
        old = source_calculation_contract(recipe="daily_trade_path_net_pnl",
                profile_id=profiles["MR-13CO"])
        self.assertNotEqual(new, old)

    def test_combined_oracle_contract_uses_full_family_selector_not_min(self):
        from services.audit import target_oracle_romd as oracle
        source = next(d for d in get_reusable_audit_definitions("breakout_quality")
                      if d.audit_type == "target_ground_truth_strategy_romd")
        settings, _baseline, _exact = oracle._contract(source)
        plan = oracle._oracle_comparison_plan(source)
        bound = oracle._resolve_oracle_target_score_contracts(source, settings, plan)
        self.assertEqual(bound["FullTrade_NetPnL_Positive"]["reference_arm_id"], "C155")
        self.assertEqual(bound["FullTrade_NetPnL_Positive_Direct"]["reference_arm_id"], "C156")
        self.assertEqual(bound["FullTrade_NetPnL_Positive"]["target_id"],
                         __import__("core.breakout_quality_runtime", fromlist=["FULLTRADE_FULLCYCLE_NETPNL_TARGET_ID"])
                         .FULLTRADE_FULLCYCLE_NETPNL_TARGET_ID)
        self.assertEqual(bound["T40"]["reference_arm_id"], _exact.arm_id)
        # The economic family cannot be spoofed by pointing a Full treatment at a Min arm.
        bad_settings = __import__("dataclasses", fromlist=["replace"]).replace(settings, arms={
            **settings.arms,
            "C155": __import__("dataclasses", fromlist=["replace"]).replace(
                settings.arms["C155"], param_source=settings.arms["C96"].param_source),
        })
        with self.assertRaisesRegex(oracle.OracleBlockedError, "strategy differs"):
            oracle._resolve_oracle_target_score_contracts(source, bad_settings, plan)

    def test_historical_teacher_canonical_build_resume_and_unauthed_block(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = inspect_full_history(project_root=root, seed=693545351)
            self.assertEqual(missing["action"], "BUILD")
            self.assertEqual(missing["path"], historical_params_path(root, 693545351))
            with self.assertRaisesRegex(Exception, "authorized"):
                prepare_full_history(project_root=root, seed=693545351)
            preflight = missing["path"].parent.parent / "rolling_preflight.json"
            preflight.parent.mkdir(parents=True)
            body = {"schema_version": 4, "mode": "selection_full_roos_parameter_training",
                    "dataset": "full", "seed_isolated": True}
            digest = canonical_json_sha256(body)
            preflight.write_text(json.dumps({**body, "runtime_identity_sha256": digest,
                                             "status": "P4_HISTORY_PREFLIGHT_PASS",
                                             "created_at": "2026-11-08T00:00:00+08:00"}))
            # A certified preflight alone supports RESUME, not fake REUSE.
            with patch("services.breakout_quality.formal_fulltrade_training_source._optimizer_policy",
                       return_value=({}, {}, None, {"runtime_identity_sha256": digest})):
                state = inspect_full_history(project_root=root, seed=693545351)
            self.assertEqual(state["action"], "RESUME")

    def test_signed_full_netpnl_build_reuse_and_positive_witness(self):
        trading, frames, bundle, min_schedule = source_fixture()
        full = tuple(replace(p, source_id=FULL_SOURCE_ID) for p in min_schedule)
        inputs = ([('0050', Path('0050.csv')), ('2330', Path('2330.csv')),
                   ('2317', Path('2317.csv'))], [])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('services.audit.oracle_score_artifacts.oracle_dataset_summary_sha256', return_value='c'*64), \
                 patch('services.audit.fulltrade_oracle_truth.oracle_dataset_summary_sha256', return_value='c'*64), \
                 patch('services.audit.fulltrade_oracle_truth.load_profile_continuous_ranker_data', return_value=bundle), \
                 patch('services.audit.fulltrade_oracle_truth.discover_dataset_csv_inputs', return_value=inputs), \
                 patch('services.audit.fulltrade_oracle_truth.collect_canonical_exchange_sessions', return_value=trading), \
                 patch('services.audit.daily_trade40_oracle.load_dataset_frame', side_effect=lambda path,ticker,**kw: frames[ticker]):
                kw=dict(project_root=root, dataset='full', filter_id='breakout_quality_v1',
                        profile_id=PROFILE, cutoff='2021-05-15', evaluation_start='2021-01-01',
                        score_start='2021-01-01', evaluation_end='2021-01-05',
                        seed=693545351, param_sha256='a'*64,
                        canonical_schedule=full, horizon_sessions=None,
                        outcome_metric='net_pnl', cache_root=root / 'cache',
                        price_sha256='b'*64, ticker_workers=1)
                a, signed = build_fulltrade_oracle_scores(**kw)
                b, cached = build_fulltrade_oracle_scores(**kw)
            self.assertEqual(a, b)
            self.assertEqual(cached['source_reuse_status'], 'REUSE')
            self.assertEqual(signed['source_reuse_status'], 'BUILD')
            self.assertEqual(signed['teacher']['origin'], 'formal_full_roos_rolling_per_seed')
            self.assertEqual(signed['truth_source_sha256'], truth_producer_fingerprint())
            with patch('services.audit.oracle_score_artifacts.oracle_dataset_summary_sha256', return_value='c'*64), \
                 patch('services.audit.fullcycle_positive_truth.load_profile_continuous_ranker_data', return_value=bundle), \
                 patch('services.audit.fullcycle_positive_truth.discover_dataset_csv_inputs', return_value=inputs), \
                 patch('services.audit.fullcycle_positive_truth.collect_canonical_exchange_sessions', return_value=trading), \
                 patch('services.audit.fullcycle_positive_truth.load_dataset_frame', side_effect=lambda path,ticker,**kw: frames[ticker]):
                args=dict(project_root=root, output_subdir='audit', dataset='full',
                          filter_id='breakout_quality_v1', profile_id=PROFILE,
                          seed=693545351, price_sha256='b'*64, param_sha256='a'*64,
                          canonical_schedule=full, research_cutoff='2021-05-15',
                          score_start='2021-01-01', evaluation_end='2021-01-05',
                          canonical_scores=a, canonical_evidence=signed)
                raw, saved = load_or_build_positive_truth(**args)
                reloaded, saved_again = load_or_build_positive_truth(**args)
            self.assertEqual(raw, reloaded)
            self.assertEqual(saved, saved_again)
            self.assertTrue(saved.is_file())
            self.assertTrue(all(isinstance(v, float) for v in raw.values()))


if __name__ == '__main__':
    main()

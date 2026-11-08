"""New Daily entry primitive and one-owner Oracle Strategy regression guards.

Execute: python -m unittest tools.validate.synthetic_daily_entry_strategy_cases -q
No real market data, formal optimizer or signed replay is executed here.
"""
from __future__ import annotations

from unittest import TestCase, main
from unittest.mock import patch

import pandas as pd

from core.audit_policy import get_reusable_audit_definitions
from core.portfolio_candidates import _resolve_qualified_normal_setup
from core.portfolio_entry_policy import (
    DAILY_ENTRY_CONTRACT, DailyEntrySource, current_daily_entry_source,
    portfolio_entry_policy_context,
)
from core.research_report_contract import require_approved_report_contract
from services.audit import target_oracle_romd as oracle


class DailyEntryStrategyCases(TestCase):
    def test_optimizer_trial_prepares_daily_data_with_trial_scoped_execution_contract(self):
        """The P20 signal mask must be computed under the same policy as replay.

        Optuna invokes ``session.objective`` outside the OOS evaluation context
        (and may use threads). The old implementation bound the context only in
        portfolio evaluation, so signal preparation silently used legacy buys.
        """
        from contextlib import contextmanager
        from types import SimpleNamespace
        from core.portfolio_entry_policy import (
            DAILY_EXECUTION_LEGACY, DAILY_EXECUTION_ATR_LIMIT,
            DAILY_EXECUTION_FULL_FILTERS, current_daily_execution_contract,
            portfolio_entry_policy_context,
        )
        from core.strategy_params import V16StrategyParams
        from services.optimizer.objective_runner import run_optimizer_objective

        class StopAfterPreparing(Exception):
            pass

        for execution in (DAILY_EXECUTION_LEGACY, DAILY_EXECUTION_ATR_LIMIT,
                          DAILY_EXECUTION_FULL_FILTERS):
            observed = []
            @contextmanager
            def runtime_context():
                with portfolio_entry_policy_context(
                    "Daily", unscored=True, daily_execution_contract=execution
                ):
                    yield
            session = SimpleNamespace(
                runtime_cache_identity="synthetic", fixed_strategy_param_overrides={},
                optimizer_runtime_context=runtime_context,
                apply_fixed_strategy_param_overrides=lambda params: params,
                get_prepared_trial_inputs_from_cache=lambda key: None,
                get_trial_prep_executor_bundle=lambda maximum: None,
                raw_data_cache={}, default_max_workers=1, static_fast_cache={},
                master_dates=set(), profile_recorder=SimpleNamespace(enabled=False),
            )
            trial = SimpleNamespace(set_user_attr=lambda key, value: None)
            def inspect_preparation(**kwargs):
                observed.append(current_daily_execution_contract())
                raise StopAfterPreparing
            with self.subTest(execution=execution), patch(
                "services.optimizer.objective_runner.build_trial_params",
                return_value=V16StrategyParams(use_breakout_buy=False),
            ), patch(
                "services.optimizer.objective_runner._can_defer_optimizer_pit_stats",
                return_value=False,
            ), patch(
                "services.optimizer.objective_runner.prepare_trial_inputs",
                side_effect=inspect_preparation,
            ):
                with self.assertRaises(StopAfterPreparing):
                    run_optimizer_objective(session, trial)
                self.assertEqual(observed, [execution])
                self.assertEqual(current_daily_execution_contract(), DAILY_EXECUTION_LEGACY)

    def test_optimizer_local_min_preparation_uses_session_entry_contract(self):
        """Local-Min diagnostics must not reprepare P20 as legacy Breakout."""
        from contextlib import contextmanager
        from types import SimpleNamespace
        from core.portfolio_entry_policy import (
            DAILY_EXECUTION_FULL_FILTERS, current_daily_execution_contract,
            portfolio_entry_policy_context,
        )
        from core.strategy_params import V16StrategyParams, strategy_params_to_dict
        from services.optimizer.robustness import _evaluate_local_min_neighbor_payload

        class StopAfterPreparing(Exception):
            pass

        @contextmanager
        def runtime_context():
            with portfolio_entry_policy_context(
                "Daily", unscored=True,
                daily_execution_contract=DAILY_EXECUTION_FULL_FILTERS,
            ):
                yield

        session = SimpleNamespace(
            optimizer_runtime_context=runtime_context,
            get_prepared_trial_inputs_from_cache=lambda key: None,
            get_trial_prep_executor_bundle=lambda max_workers: None,
            raw_data_cache={}, default_max_workers=1, static_fast_cache={},
            master_dates=set(), runtime_cache_identity="test_p20",
        )
        observed = []

        def observe_preparation(**kwargs):
            observed.append(current_daily_execution_contract())
            raise StopAfterPreparing

        with patch(
            "services.optimizer.robustness._build_session_params_from_mapping",
            return_value=V16StrategyParams(use_breakout_buy=False),
        ), patch(
            "services.optimizer.robustness.prepare_trial_inputs",
            side_effect=observe_preparation,
        ):
            with self.assertRaises(StopAfterPreparing):
                _evaluate_local_min_neighbor_payload(
                    session, strategy_params_to_dict(V16StrategyParams(use_breakout_buy=False)), {}
                )
        self.assertEqual(observed, [DAILY_EXECUTION_FULL_FILTERS])

    def test_daily_full_preparation_and_replay_are_one_execution_semantic(self):
        """Prove the former P20 0-trade failure and corrected P20 portfolio path.

        The exact same market and trading parameters must trade when the
        preparation uses the declared Full contract, but the old implicit
        legacy preparation creates no setup rows for a non-Breakout strategy.
        """
        from dataclasses import replace
        from core.portfolio_fast_data import (
            prep_optimizer_stock_data_bundle, pack_static_market_data,
            merge_static_market_with_dynamic,
        )
        from core.portfolio_engine import run_portfolio_timeline
        from core.portfolio_entry_policy import (
            DAILY_EXECUTION_LEGACY, DAILY_EXECUTION_FULL_FILTERS,
            daily_signal_execution_context, portfolio_entry_policy_context,
        )
        from tools.validate.synthetic_daily_trade40_cases import fixture
        days, frame, original, _ = fixture(count=160)
        params = replace(
            original, use_breakout_buy=False, use_bb=False, use_kc=False,
            use_vol=False, use_breakout_ema_filter=False,
            use_breakout_return_filter=False, use_breakout_false_filter=False,
        )
        trade_counts = {}
        for preparation in (DAILY_EXECUTION_LEGACY, DAILY_EXECUTION_FULL_FILTERS):
            with daily_signal_execution_context(preparation):
                prepared, _, _ = prep_optimizer_stock_data_bundle(
                    frame, params, ticker="2330", include_trade_logs=False,
                    include_pit_stats_index=False,
                )
            market = {"2330": merge_static_market_with_dynamic(
                pack_static_market_data(frame), prepared,
            )}
            with portfolio_entry_policy_context(
                "Daily", unscored=True,
                daily_execution_contract=DAILY_EXECUTION_FULL_FILTERS,
            ):
                result = run_portfolio_timeline(
                    market, {}, list(days), 2021, params, 10, False,
                    benchmark_ticker="2330", benchmark_data=market["2330"],
                    is_training=True, verbose=False,
                )
            trade_counts[preparation] = int(result[4])
        self.assertEqual(trade_counts[DAILY_EXECUTION_LEGACY], 0)
        self.assertGreater(trade_counts[DAILY_EXECUTION_FULL_FILTERS], 0)

    def test_optimizer_p20_study_implementation_revision_rebuilds_only_full_daily(self):
        """Old zero-candidate P20 trials may not be reused after the prep fix."""
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from core.file_integrity import canonical_json_sha256
        from services.optimizer.strategy_param_training import _runtime_contract
        from services.optimizer.outer_rolling_runtime import _resolve_outer_rolling_db_file

        def contract(profile):
            args = SimpleNamespace(
                strategy_id="daily", daily_search_profile=profile, dataset="full",
                trials_per_fold=300, fixed_risk=0.01, max_position_cap_pct=1.0,
                max_positions=10, rotation="off", optimizer_search_protocol="",
            )
            return _runtime_contract(
                root=Path(__file__).resolve().parents[2], args=args,
                baseline_contract={"sha256": "a" * 64, "meta": {
                    "window_mode": "fixed", "first_oos_date": "2021-01-01",
                    "last_oos_date": "2026-03-02", "train_window_months": 120,
                    "oos_horizon_months": 12,
                }},
                fold_overrides={}, model_artifact=None, arm_id="P2",
                training_dl_enabled=False, binary_pit=None, dataset_inventory={},
            )
        for profile in (None, "min_atr_limit_v2", "full_common_filters_v2"):
            with self.subTest(profile=profile):
                payload = contract(profile)
                previous = {key: value for key, value in payload.items()
                            if key not in ("runtime_identity_sha256", "optimizer_trial_prep_revision")}
                old_identity = canonical_json_sha256(previous)
                if profile == "full_common_filters_v2":
                    self.assertEqual(payload["optimizer_trial_prep_revision"],
                                     "daily_full_scoped_input_preparation_v2")
                    self.assertNotEqual(payload["runtime_identity_sha256"], old_identity)
                    with TemporaryDirectory() as temp_dir:
                        old_db, _ = _resolve_outer_rolling_db_file(
                            output_dir=temp_dir, session_ts="test", oos_year=202401,
                            task={"runtime_cache_identity": old_identity},
                            environ={"OPTIMIZER_ROLLING_RESUME_EXISTING_STUDIES": "1"},
                        )
                        Path(old_db).touch()
                        new_db, resume = _resolve_outer_rolling_db_file(
                            output_dir=temp_dir, session_ts="test", oos_year=202401,
                            task={"runtime_cache_identity": payload["runtime_identity_sha256"]},
                            environ={"OPTIMIZER_ROLLING_RESUME_EXISTING_STUDIES": "1"},
                        )
                        self.assertFalse(resume)
                        self.assertNotEqual(old_db, new_db)
                        self.assertTrue(Path(old_db).is_file())
                else:
                    self.assertNotIn("optimizer_trial_prep_revision", payload)
                    self.assertEqual(payload["runtime_identity_sha256"], old_identity)

    def test_optimizer_no_finalist_evidence_keeps_unqualified_trials_invalid(self):
        """Canonical Parameter Producer may not publish an invalid Trial as best."""
        import optuna
        from services.optimizer.outer_rolling_runtime import summarize_optimizer_no_finalist
        from services.optimizer.study_utils import INVALID_TRIAL_VALUE

        study = optuna.create_study(direction="maximize")
        study.add_trial(optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=float(INVALID_TRIAL_VALUE),
            user_attrs={"objective_mode": "split_train_romd", "fail_reason": "年化交易次數過低"},
        ))
        study.add_trial(optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=float(INVALID_TRIAL_VALUE),
            user_attrs={"objective_mode": "split_train_romd", "fail_reason": "年化交易次數過低"},
        ))
        study.add_trial(optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=float(INVALID_TRIAL_VALUE),
            user_attrs={"objective_mode": "split_train_romd", "fail_reason": "權益曲線過度震盪"},
        ))
        summary = summarize_optimizer_no_finalist(study, objective_mode="split_train_romd")
        self.assertIn("trials=3", summary)
        self.assertIn("qualified=0", summary)
        self.assertIn("invalid=3", summary)
        self.assertIn("年化交易次數過低 ×2", summary)

    def test_optimizer_no_finalist_is_non_retryable_and_does_not_relaunch_folds(self):
        from services.optimizer.outer_rolling_runtime import _is_non_retryable_fold_failure
        from services.optimizer.outer_rolling_oos import _run_missing_parallel_fold_tasks_sequentially

        failure = RuntimeError("NON_RETRYABLE_NO_FINALIST: trials=300 | qualified=0")
        self.assertTrue(_is_non_retryable_fold_failure(failure))
        with patch("services.optimizer.outer_rolling_oos._run_outer_rolling_oos_fold_task",
                   side_effect=AssertionError("不應重新搜尋")) as retried:
            with self.assertRaisesRegex(RuntimeError, "NON_RETRYABLE_OPTIMIZER_ERROR"):
                _run_missing_parallel_fold_tasks_sequentially(
                    tasks=[{"oos_year": 202101, "fold_idx": 1, "fold_count": 1}],
                    rows=[], fold_timing_rows=[], chain_state={}, cause=failure,
                )
        retried.assert_not_called()

    def test_optimizer_parameter_fold_payload_demands_completed_schedule(self):
        from services.optimizer.outer_rolling_oos import _build_outer_rolling_fold_task

        task = _build_outer_rolling_fold_task(
            mode="parallel",
            fold={"oos_year": 202101, "oos_start_date": "2021-01-01",
                  "oos_end_date": "2021-12-31", "oos_period": "2021-01~2021-12",
                  "selection_start_date": "2011-01-01", "selection_end_date": "2020-12-31",
                  "selection_period": "2011-01~2020-12"},
            fold_idx=1, fold_count=6,
            config_payload={},
            common={"project_root": ".", "output_dir": ".", "base_policy": {},
                    "selected_data_dir": ".", "optimizer_required_min_rows": 1,
                    "session_ts": "fixture", "optimizer_seed": 42, "sampler_kind": "random",
                    "timing_mode": False, "fold_workers": 1,
                    "optimizer_session_spec": {}, "progress_display_contract": {},
                    "require_complete_param_schedule": True},
            log_dir="logs", seed_ensemble_policy={},
        )
        self.assertTrue(task["require_complete_param_schedule"])

    def test_registered_parameter_policy_controls_daily_optimizer_diagnostics(self):
        """A declared single-parameter policy cannot trigger an unrelated agree replay."""
        from types import SimpleNamespace
        from config.strategy_params import STRATEGY_PARAM_PROFILES
        from services.optimizer.outer_rolling_policy import (
            _build_policy_items, _session_report_policy_names,
            _is_policy_ensemble_item, REPORT_POLICY_NAMES,
        )
        from services.optimizer.outer_rolling_evaluation import _evaluate_finalist_oos_diagnostics

        registered = STRATEGY_PARAM_PROFILES["broker_cash_daily_min_best_seed42_current"]
        self.assertEqual(registered["strategy_id"], "daily")
        self.assertEqual(registered["parameter_policy"], "base-finalist-best")
        declared = registered["parameter_policy"].replace("-", "_")
        session = SimpleNamespace(strategy_param_scientific_contract={
            "scientific_contract": {"strategy": dict(registered)}
        })
        finalists = [
            {"trial": SimpleNamespace(number=i, params={}, user_attrs={}),
             "base_rank": i + 1, "base_score": 10.0 - i, "gate_pass": True}
            for i in range(3)
        ]
        items = _build_policy_items(finalists, objective_mode="split_train_romd", session=session)
        self.assertEqual(tuple(items), (declared,))
        self.assertFalse(any(_is_policy_ensemble_item(item) for item in items.values()))
        self.assertEqual(_session_report_policy_names(None), REPORT_POLICY_NAMES)

        # The historical failing call was to multi-member portfolio replay
        # inside _evaluate_finalist_oos_diagnostics. It must remain unreachable.
        fake_report = {"period": {"test_score_romd": 2.0, "ret_pct": 20.0,
                                  "mdd": -10.0, "trade_count": 2,
                                  "benchmark_score_romd": 1.0}}
        with patch("services.optimizer.outer_rolling_evaluation._evaluate_period_oos",
                   return_value=fake_report) as period, patch(
                       "services.optimizer.outer_rolling_evaluation._evaluate_finalist_ensemble_oos_metrics",
                       side_effect=AssertionError("Unrequested consensus replay")) as consensus:
            output = _evaluate_finalist_oos_diagnostics(
                session=session, finalists=finalists, policy_items=items,
                oos_year=2021, oos_start_date="2021-01-01", oos_end_date="2021-12-31")
        consensus.assert_not_called()
        self.assertTrue(output["policies"][declared]["available"])
        self.assertEqual(output["policies"][declared]["rank_1_trial"], 1)
        self.assertGreaterEqual(period.call_count, 1)

    def test_optimizer_parameter_policy_selection_fails_closed_when_invalid(self):
        from types import SimpleNamespace
        from services.optimizer.outer_rolling_policy import _session_report_policy_names
        for policy in ("not-an-approved-policy", ""):
            with self.subTest(policy=policy):
                session = SimpleNamespace(strategy_param_scientific_contract={
                    "scientific_contract": {"strategy": {"parameter_policy": policy}}
                })
                with self.assertRaisesRegex(ValueError, "scientific contract"):
                    _session_report_policy_names(session)

    def test_optimizer_does_not_publish_empty_unselected_parameter_artifacts(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from types import SimpleNamespace
        from services.optimizer.outer_rolling_artifacts import _write_policy_paramset_files
        from services.optimizer.outer_rolling_active_replay import _build_active_param_replay_payload_from_rows
        from core.active_param_ensemble import is_active_param_ensemble_payload

        params = {"atr_len": 12}
        rows = [{"fold": "1/1", "oos_year": 2021,
                 "oos_start_date": "2021-01-01", "oos_end_date": "2021-12-31",
                 "best_finalist_params": params,
                 "policy_schedules": {"base_finalist_best": {
                     "oos_year": 2021, "effective_start": "2021-01-01",
                     "effective_end": "2021-12-31", "params": params,
                 }}}]
        payload = _build_active_param_replay_payload_from_rows(rows, policy_name="base_finalist_best")
        self.assertFalse(is_active_param_ensemble_payload(payload))
        self.assertEqual(payload["params_by_effective_date"]["2021-01-01"], params)
        with TemporaryDirectory() as temp_dir, patch(
            "services.optimizer.outer_rolling_artifacts._build_policy_paramset_payload",
            side_effect=lambda **kw: {"policy": kw["policy_name"]}), patch(
            "services.optimizer.outer_rolling_artifacts.normalize_strategy_param_payload_for_persistence",
            side_effect=lambda data: data):
            result = _write_policy_paramset_files(
                models_dir=temp_dir, rows=rows,
                config=SimpleNamespace(), summary={})
            self.assertEqual(tuple(result), ("base_finalist_best",))
            self.assertEqual(len(list(Path(temp_dir).glob("*.json"))), 1)

    def test_daily_strategy_records_follow_canonical_comparisons(self):
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        rows = tuple(definition.source["oracle_comparisons"])
        from core.positive_netpnl_contract import requires_positive_score
        from core.audit_registry import oracle_supervised_target_id, oracle_parameter_family
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        from core.breakout_quality_runtime import get_continuous_ranker_score_output_columns
        # AI註: Validate each declaration without freezing current roster membership.
        scoped = oracle._oracle_strategy_scopes(definition)
        expected = {}
        for row in rows:
            family = oracle_parameter_family(row)
            strategy = str(row.get("strategy", "Breakout"))
            scope_name = strategy if family == "min" else f"{strategy}::{family.capitalize()}"
            expected[scope_name] = expected.get(scope_name, 0) + 1
        self.assertEqual({s: len(d.source["oracle_comparisons"]) for s, d in scoped},
                         expected)
        for strategy, scope in scoped:
            settings, base, exact = oracle._contract(scope)
            plan = oracle._oracle_comparison_plan(scope)
            direct = oracle._direct_arm(scope, settings, base, exact)
            contracts = oracle._resolve_oracle_target_score_contracts(scope, settings, plan)
            if strategy.split("::", 1)[0] != "Daily":
                continue
            for row in scope.source["oracle_comparisons"]:
                arm = oracle._target_reference_arm(scope, settings, plan, row["target"], exact, direct)
                profile = get_breakout_quality_experiment_profile(
                    settings.dl_sources[arm.dl_id].experiment_profile)
                outputs = get_continuous_ranker_score_output_columns(
                    profile.training_objective, score_calibration=profile.score_calibration)
                self.assertEqual(arm.entry_strategy, "Daily")
                self.assertEqual(requires_positive_score(arm.dl_runtime_options), bool(row.get("positive_output")))
                if row.get('positive_output'):
                    positive_id = arm.dl_runtime_options['positive_dl_id']
                    classifier = get_breakout_quality_experiment_profile(
                        settings.dl_sources[positive_id].experiment_profile)
                    positive_columns = get_continuous_ranker_score_output_columns(
                        classifier.training_objective,score_calibration=classifier.score_calibration)
                    self.assertIn('positive_p_positive',positive_columns.values())
                    self.assertNotIn('positive_p_positive',outputs.values())
                    self.assertEqual(classifier.continuous_target_id,profile.continuous_target_id)
                else:
                    self.assertNotIn('positive_p_positive',outputs.values())
                # Trade-path recipes sign the raw producer identity; a scalar
                # price recipe preserves its established Oracle target name.
                # The model must still supervise the registered raw Target.
                source = row
                if row.get("score_source"):
                    source = next(item for item in scope.source["oracle_source_catalog"]
                                  if item["target"] == row["score_source"])
                self.assertEqual(profile.continuous_target_id,
                                 oracle_supervised_target_id(source))
                expected_oracle_target = (
                    oracle._daily_truth_identity(
                        source["score"],
                        None if source.get("horizon_mode") == "full_trade_cycle" else
                        source.get("horizon_sessions", oracle.HORIZON_SESSIONS),
                    )[0]
                    if source.get("score", "").startswith("daily_trade_path_")
                    else source["target"]
                )
                self.assertEqual(contracts[row["target"]]["target_id"], expected_oracle_target)
        self.assertTrue(all(oracle._oracle_target_strategy(d, r["target"]) == strategy.split("::", 1)[0]
                            for strategy, d in scoped
                            for r in d.source["oracle_comparisons"]))
        self.assertEqual(oracle._oracle_target_strategy(definition, "C96"), "Breakout")
        contract = require_approved_report_contract("audit.target_oracle_romd")
        self.assertGreater(contract.version, 0)
        self.assertEqual(tuple(c.label for c in contract.sections[0].tables[0].columns[:3]),
                         ("Target", "Strategy", "Selector"))
        self.assertTrue(DAILY_ENTRY_CONTRACT.startswith("daily_stockday_close_limit_"))

    def test_daily_fit_has_separate_event_truth_and_cutoff_safe_cache(self):
        """Real canonical Min settlement, not mocked Oracle truth or future FIT rows."""
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import numpy as np
        from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS, STRATEGY_DL_SOURCES
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        from services.breakout_quality.trade_path_target_source import registered_trade_path_source
        from services.breakout_quality.true_daily_mintrade_pit_source import (
            materialize_true_daily_mintrade_pit_labels,
        )
        from tools.validate.synthetic_daily_trade40_cases import (
            FormalMinTrade40TrainingSourcePreflightCases, fixture,
        )
        models = {}
        for arm_id in ("C125", "C126", "C133", "C134"):
            arm = STRATEGY_COMPARE_ARMS[arm_id]
            models[arm_id] = (arm["dl_id"], arm["dl_runtime_options"]["positive_dl_id"])
            self.assertNotEqual(*models[arm_id])
        self.assertEqual(models["C125"], models["C126"])
        self.assertEqual(models["C133"], models["C134"])
        self.assertNotEqual(models["C125"], models["C133"])
        event_profile = STRATEGY_DL_SOURCES[models["C125"][0]]["experiment_profile"]
        daily_profile = STRATEGY_DL_SOURCES[models["C133"][0]]["experiment_profile"]
        self.assertNotEqual(registered_trade_path_source(event_profile).contract_kind,
                            registered_trade_path_source(daily_profile).contract_kind)
        daily_positive_profile = STRATEGY_DL_SOURCES[models["C133"][1]]["experiment_profile"]
        self.assertEqual(get_breakout_quality_experiment_profile(daily_profile).continuous_target_id,
                         get_breakout_quality_experiment_profile(daily_positive_profile).continuous_target_id)
        days, source = FormalMinTrade40TrainingSourcePreflightCases._source()
        _, frame, _, _ = fixture(count=140, start="2018-01-02")
        args = dict(source=source, expected_seed=source.seed, frame=frame,
                    positions=np.array([70, 71, 72, 90]), ticker="2330",
                    exchange_calendar=days, fit_cutoff=days[110])
        original = materialize_true_daily_mintrade_pit_labels(**args)
        with TemporaryDirectory() as directory:
            cached = materialize_true_daily_mintrade_pit_labels(**args, cache_project_root=directory)
            reused = materialize_true_daily_mintrade_pit_labels(**args, cache_project_root=directory)
            self.assertEqual(len(list(Path(directory).rglob("*.npz"))), 1)
            for result in (cached, reused):
                np.testing.assert_array_equal(result.completed, original.completed)
                np.testing.assert_allclose(result.raw_net_pnl, original.raw_net_pnl, equal_nan=True)
                np.testing.assert_array_equal(result.maturity_dates, original.maturity_dates)
        self.assertEqual(int(original.completed.sum()), 3)
        self.assertTrue(np.isnan(original.raw_net_pnl[~original.completed]).all())
        self.assertTrue(np.isnat(original.maturity_dates[~original.completed]).all())

    def test_single_registry_pair_identity_and_duplicate_rejection(self):
        from core.audit_registry import oracle_comparison_key, resolve_current_oracle_comparisons
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        rows = resolve_current_oracle_comparisons(definition.source)
        keys = [oracle_comparison_key(row) for row in rows]
        self.assertEqual(len(keys), len(rows))
        self.assertEqual(len(set(keys)), len(rows))
        for name in ("TrueDailyUniversal_MinTrade_NetPnL_Positive",
                     "TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct"):
            pair = [row for row in rows if row["target"] == name]
            present_strategies = {p.get("strategy", "Breakout") for p in pair}
            self.assertIn("Breakout", present_strategies)
            if name.endswith("_Direct"):
                self.assertIn("Daily", present_strategies)
            self.assertEqual({tuple(sorted((k, repr(v)) for k,v in p.items()
                                           if k not in {"strategy", "strategy_arm_id", "parameter_family"})) for p in pair}.__len__(), 1)
        with self.assertRaisesRegex(ValueError, "duplicate Target x Strategy"):
            resolve_current_oracle_comparisons({
                **definition.source, "oracle_comparisons": (*rows, dict(rows[-1])),
            })
        # The inventory's second dimension is not a new scientific Target ID.
        self.assertNotIn("TrueDailyUniversal_MinTrade_NetPnL_Positive::Daily", keys)
        self.assertIn("TrueDailyUniversal_MinTrade_NetPnL_Positive", keys)
        self.assertIn("TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct::Daily::Daily_min", keys)

    def test_one_approval_matrix_and_same_engine_for_declared_treatments(self):
        from core.audit_registry import oracle_baseline_arm_ids, oracle_comparison_key
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        seeds = tuple(range(1, 9))
        comparisons = tuple(definition.source["oracle_comparisons"])
        scope_counts = [(strategy, len(scope.source["oracle_comparisons"]))
                        for strategy, scope in oracle._oracle_strategy_scopes(definition)]
        baselines = oracle_baseline_arm_ids(definition.source)
        reuse_count = len(seeds) * next(n for name, n in scope_counts if name == "Breakout")
        run_count = len(seeds) * (len(comparisons) - reuse_count // len(seeds))
        calls = []

        def fake_run(scoped, *, preview_only=False, **kwargs):
            strategy = scoped.source["oracle_comparisons"][0].get("strategy", "Breakout")
            family_baseline = str(scoped.source["baseline_arm_id"])
            family = next(f for f, arm in baselines.items() if arm == family_baseline)
            scope_name = (strategy if family == "min" else
                          f"{strategy}::{family.capitalize()}")
            targets = tuple(row["target"] for row in scoped.source["oracle_comparisons"])
            calls.append((scope_name, preview_only, kwargs.get("approved_replay_decisions")))
            decisions = {(seed, name): "REUSE" if scope_name == "Breakout" else "RUN"
                         for seed in seeds for name in (family_baseline, *targets)}
            for seed in seeds:
                if family_baseline == baselines["min"]:
                    decisions[(seed, family_baseline)] = "REUSE"
            if preview_only:
                return {"status":"PREVIEW", "preflight": decisions,
                        "benchmark_id":"end_to_end_v2", "completed_observations":{},
                        "start_date":"2021-01-01", "end_date":"2026-03-02",
                        "seed_set":seeds,"missing_p16":(),
                        "producer_kwargs":{"method":"canonical", "baseline":family_baseline},
                        "targets": (family_baseline,*targets)}
            self.assertTrue(kwargs["preflight_approval"]("single scope recheck"))
            self.assertEqual(kwargs["approved_replay_decisions"], decisions)
            return {"status":"COMPLETED", "seeds":list(seeds),
                    "benchmark_id":"end_to_end_v2", "dataset":"full",
                    "start_date":"2021-01-01", "end_date":"2026-03-02",
                    "strategy_profile_id": "test", "param_artifacts":{},
                    "common_cohort":{"cohort_rows":630589},
                    "baseline":{"target":family_baseline, "strategy":"Breakout",
                                "seed_observations": [], "romd": 9.0},
                    "oracles":[{"target": target, "strategy":strategy,
                                "romd":25.0 if strategy == "Breakout" else 40.0}
                               for target in targets],
                    "definitions":{row["target"]:row["description"]
                                   for row in scoped.source["oracle_comparisons"]}}
        with patch.object(oracle,"_run_single_stage_oracle",side_effect=fake_run):
            preview = oracle.preview_oracle_replay_plan(definition,project_root="/tmp")
            self.assertEqual(len(preview["targets"]), len(comparisons))
            self.assertEqual(len(preview["control_dependencies"]), len(baselines))
            self.assertIn(f"Portfolio REUSE {reuse_count} / RUN（需回放）{run_count} / BLOCKED 0", preview["text"])
            self.assertIn("DailyUniversal_MinTrade_NetPnL_Positive", preview["text"])
            self.assertIn("Direct", preview["text"])
            self.assertIn("Daily",preview["text"])
            result = oracle.build_report(
                definition,project_root="/tmp",quiet=True,
                preflight_approval=lambda _:True,
                approved_replay_decisions=preview["preflight"],
            )
        self.assertEqual(len(result["oracles"]), len(comparisons))
        self.assertEqual(len(result["definitions"]), len({row["target"] for row in comparisons}))
        self.assertEqual(len(result["comparison_inventory"]), len(baselines) + len(comparisons))
        target = "TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct"
        values = {(row["target"],row["strategy"]):row["romd"] for row in result["oracles"]}
        self.assertEqual(values[(target,"Breakout")],25.0)
        self.assertEqual(values[(target,"Daily")],40.0)
        self.assertEqual([c[1] for c in calls], [True] * len(scope_counts) + [False] * len(scope_counts))
        self.assertEqual([(c[0],len(c[2])) for c in calls[-len(scope_counts):]],
                         [(strategy, len(seeds) * (1 + count)) for strategy, count in scope_counts])

    def test_scoped_daily_scores_do_not_leak_or_read_positive_truth(self):
        days = tuple(pd.Timestamp(t) for t in
                     ("2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"))
        market = {
            "2330": {d: {"Close": 100.0} for d in days},
            "2317": {days[0]: {"Close": 90.0}, days[2]: {"Close": 92.0}},
        }
        keys = {
            ("2330", "2021-01-04"), ("2330", "2021-01-06"),
            ("2317", "2021-01-04"), ("00958B", "2021-01-04"),
        }
        self.assertIsNone(current_daily_entry_source())
        with portfolio_entry_policy_context("Breakout"):
            self.assertIsNone(current_daily_entry_source())
        with portfolio_entry_policy_context("Daily", score_keys=keys):
            source = current_daily_entry_source()
            self.assertIsInstance(source, DailyEntrySource)
            source.configure_calendar(days)
            # Market-covered score stocks only; today must follow information day.
            self.assertEqual(source.entries_for(days[1], market), {days[1]: [("2330", 0, 1)]})
            self.assertEqual(source.excluded_missing_market_ticker, 1)
            self.assertEqual(source.excluded_missing_adjacent_quote, 1)
            self.assertEqual(source.entries_for(days[2], market), {})
            self.assertEqual(source.entries_for(days[3], market), {days[3]: [("2330", 2, 3)]})
            with self.assertRaisesRegex(RuntimeError, "Nested Daily"):
                with portfolio_entry_policy_context("Daily", score_keys=keys):
                    pass
        self.assertIsNone(current_daily_entry_source())
        with self.assertRaises(ValueError):
            with portfolio_entry_policy_context("Daily"):
                pass
        self.assertIsNone(current_daily_entry_source())

    def test_entry_limit_prior_close_and_history_gate_is_breakout_only(self):
        d1, d2 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
        # The old Breakout setup carries 110 limit, but Daily must use Close=100.
        fast = {d1: {"buy_limit": 110.0, "ATR": 3.0, "Close": 100.0},
                d2: {"buy_limit": 115.0, "ATR": 3.0, "Close": 105.0}}
        kwargs = dict(
            ticker="2330", y_pos=0, portfolio={}, sold_today=set(),
            all_dfs_fast={"2330": fast}, active_extended_signals={},
            pit_stats_index={"2330": []}, pit_stats_cursor={},
            today=d2, params=object(), profile_stats=None,
        )
        with patch("core.portfolio_candidates.get_pit_stats_from_index",
                   side_effect=AssertionError("Breakout history gate must not run for Daily")):
            daily = _resolve_qualified_normal_setup(**kwargs, entry_policy="daily")
        self.assertEqual(daily[0], 100.0)
        self.assertEqual(daily[1:4], (3.0, 100.0, d1))
        with patch("core.portfolio_candidates.get_pit_stats_from_index",
                   return_value=(True, 1.0, .5, 2, 10.0)) as historical:
            breakout = _resolve_qualified_normal_setup(**kwargs)
            historical.assert_called_once()
        self.assertEqual(breakout[0], 110.0)

    def test_real_portfolio_engine_changes_only_daily_buy_admission(self):
        # A small real canonical Portfolio Engine integration (not a fake
        # replay): the descending-price no-Breakout stock is score eligible.
        from tools.validate.synthetic_daily_trade40_oracle_cases import source_fixture
        from core.portfolio_fast_data import prep_stock_data_and_trades, pack_prepared_stock_data
        from core.portfolio_engine import run_portfolio_timeline
        exchange, frames, _bundle, schedule = source_fixture()
        params = schedule[0].params
        market, trade_logs = {}, {}
        for ticker, frame in frames.items():
            ready, logs = prep_stock_data_and_trades(frame, params)
            market[ticker] = pack_prepared_stock_data(ready)
            trade_logs[ticker] = logs
        keys = {(ticker, str(day.date())) for ticker in market
                for day in exchange if day.year == 2021}
        kwargs = dict(
            all_dfs_fast=market, all_standalone_logs=trade_logs,
            sorted_dates=list(exchange), start_year=2021,
            params=params, max_positions=10, enable_rotation=False,
            benchmark_ticker="2330", benchmark_data=market["2330"],
            is_training=False, verbose=False, profile_stats={},
        )
        breakout = run_portfolio_timeline(**kwargs)
        with portfolio_entry_policy_context("Daily", score_keys=keys):
            daily = run_portfolio_timeline(**kwargs)
        self.assertEqual(int(breakout[4]), 1)
        self.assertGreater(int(daily[4]), int(breakout[4]))
        self.assertIsNone(current_daily_entry_source())

    def test_daily_unfilled_buy_expires_and_next_day_needs_own_score(self):
        # Real Portfolio Engine, not a mocked candidate cache. Only 1/4 has
        # a score; 1/5 low > 1/4 close; 1/6 would fill the *expired* order.
        from tools.validate.synthetic_daily_trade40_oracle_cases import source_fixture
        from core.portfolio_fast_data import prep_stock_data_and_trades, pack_prepared_stock_data
        from core.portfolio_engine import run_portfolio_timeline
        dates, frames, _, schedule = source_fixture()
        frame = frames["2317"].copy()
        frame.loc["2021-01-05", ["Open", "High", "Low", "Close"]] = [90.9, 91.5, 90.3, 90.7]
        frame.loc["2021-01-06", ["Open", "High", "Low", "Close"]] = [89.7, 91.8, 88.7, 89.8]
        ready, trades = prep_stock_data_and_trades(frame, schedule[0].params)
        market = {"2317": pack_prepared_stock_data(ready)}
        args = dict(
            all_dfs_fast=market, all_standalone_logs={"2317": trades},
            sorted_dates=list(dates), start_year=2021,
            params=schedule[0].params, max_positions=10, enable_rotation=False,
            benchmark_ticker="2317", benchmark_data=market["2317"],
            is_training=False, verbose=False, profile_stats={},
        )
        rows = []
        with portfolio_entry_policy_context("Daily", score_keys={("2317", "2021-01-04")}):
            result = run_portfolio_timeline(**args, replay_execution_rows=rows)
        self.assertEqual(int(result[4]), 0)
        self.assertFalse(any(
            str(row.get("trade_date")) == "2021-01-06" and row.get("entry_source") == "extended"
            for row in rows
        ))
        # A *fresh* 1/5 score is allowed to submit a different next-day limit.
        fresh_rows = []
        with portfolio_entry_policy_context("Daily", score_keys={("2317", "2021-01-05")}):
            fresh = run_portfolio_timeline(**args, replay_execution_rows=fresh_rows)
        self.assertEqual(int(fresh[4]), 1)
        self.assertTrue(any(
            str(row.get("signal_date")) == "2021-01-05" and
            str(row.get("candidate_date")) == "2021-01-06"
            for row in fresh_rows
        ))

    def test_entry_policy_and_replay_version_are_economic_cache_inputs(self):
        from copy import deepcopy
        from services.research.strategy_compare_reuse import _pair_cache_fingerprint_from_payload
        from services.research.strategy_robustness_units import scientific_unit_signature
        from services.research.target_dl_overview import _replay_policy_current
        from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
        current = STRATEGY_COMPARE_ARMS["C133"]
        self.assertTrue(_replay_policy_current(current, current))
        self.assertFalse(_replay_policy_current(current, dict(current, replay_contract_version=2)))
        self.assertFalse(_replay_policy_current(current, dict(current, entry_strategy="Breakout")))
        self.assertFalse(_replay_policy_current(current, dict(current, replay_contract_version="invalid")))
        self.assertTrue(_replay_policy_current(STRATEGY_COMPARE_ARMS["C135"],
                                               STRATEGY_COMPARE_ARMS["C135"]))
        settings = {"dataset": "full", "parameter_sources": {}, "dl_sources": {}}
        base = {"arm_id": "C133", "param_source": "p", "param_policy": "base-finalist-best", "rule_policy": "all_off",
                "dl_enabled": False, "dl_runtime_mode": "", "dl_id": None}
        def cache(on):
            return _pair_cache_fingerprint_from_payload(
                settings_payload=settings, artifact_identities={}, comparison_period={},
                off_arm_payload=base, on_arm_payload=on, engine_schema_version=1,
            )
        breakout = cache(base)
        daily = cache(dict(base, entry_strategy="Daily"))
        v2 = cache(dict(base, entry_strategy="Daily", replay_contract_version=2))
        v3 = cache(dict(base, entry_strategy="Daily", replay_contract_version=3))
        self.assertEqual(len({breakout, daily, v2, v3}), 4)
        unit_base = {"stochastic_arms": [dict(base, param_policy="p", param_source="p",
                                             strategy_seed_sensitive=True, model_seed_sensitive=False)],
                     "benchmark_parameter_artifact_identities": {"C133:seed=42": {
                         "sha256": "a"*64, "family": "x", "identity_schema": "x",
                         "evaluation_mode": "rolling", "param_policy": "p"}}}
        contract = dict(unit_base)
        before = scientific_unit_signature(contract=contract, arm_id="C133", seed=42)
        contract = deepcopy(unit_base)
        contract["stochastic_arms"][0]["entry_strategy"] = "Daily"
        contract["stochastic_arms"][0]["replay_contract_version"] = 3
        after = scientific_unit_signature(contract=contract, arm_id="C133", seed=42)
        self.assertNotEqual(before, after)
        self.assertEqual(after["arm"]["entry_strategy"], "Daily")
        self.assertEqual(after["arm"]["replay_contract_version"], 3)

    def test_registered_daily_and_positive_arms_pit_and_oracle_replay_parity(self):
        """PIT and perfect-output sources use one real Broker-Cash replay owner.

        This synthetic case is not signed 8-seed E13 parity. It does assert
        identical trades/equity and that a zero Positive score changes fills.
        """
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.positive_netpnl_contract import requires_positive_score
        from core.portfolio_fast_data import prep_stock_data_and_trades, pack_prepared_stock_data
        from core.portfolio_engine import run_portfolio_timeline
        from filters.breakout_quality.runtime import (
            audit_oracle_truth_score_context, breakout_quality_ranking_source_context,
        )
        from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
        from tools.validate.synthetic_daily_trade40_oracle_cases import source_fixture

        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        settings = get_strategy_comparison_settings(definition.source["strategy_profile_id"])
        days, frames, _bundle, schedule = source_fixture()
        params = schedule[0].params
        params.use_breakout_quality_ranking = True
        params.breakout_quality_filter_id = definition.source["filter_id"]
        market, logs = {}, {}
        for ticker, frame in frames.items():
            prepared, standalone = prep_stock_data_and_trades(frame, params)
            market[ticker] = pack_prepared_stock_data(prepared)
            logs[ticker] = standalone
        scores = {(ticker, str(day.date())): (0.9 if ticker == "2330" else 0.1)
                  for ticker in market for day in days}
        arguments = dict(
            all_dfs_fast=market, all_standalone_logs=logs, sorted_dates=list(days),
            start_year=2021, params=params, max_positions=10, enable_rotation=False,
            benchmark_ticker="2330", benchmark_data=market["2330"],
            is_training=False, verbose=False, profile_stats={},
        )
        arms = [arm for arm in settings.enabled_arms
                if arm.dl_enabled and (arm.entry_strategy == "Daily"
                                       or requires_positive_score(arm.dl_runtime_options))]
        self.assertTrue(arms)
        for arm in arms:
            arm_id = arm.arm_id
            self.assertIn(arm.entry_strategy, {"Daily", "Breakout"})
            self.assertTrue(arm.enabled)
            primary = settings.dl_sources[arm.dl_id]
            options = dict(arm.dl_runtime_options)
            if 'positive_dl_id' in options:
                second = settings.dl_sources[options['positive_dl_id']]
                options.update(positive_filter_id=second.filter_id,
                    positive_score_source=second.score_source,
                    positive_model_architecture=second.model_architecture,
                    positive_experiment_profile=second.experiment_profile)
            unfiltered_trades = None
            qualified = requires_positive_score(options)
            for excluded in ((False, True) if qualified else (False,)):
                positives = {key: (0.0 if excluded and key[0] == "2330" else 1.0)
                             for key in scores}

                def pit_score(*, ticker, signal_date, filter_id, **kwargs):
                    key = (str(ticker), pd.Timestamp(signal_date).strftime("%Y-%m-%d"))
                    is_positive = kwargs.get("score_column") == "positive_p_positive"
                    number = (positives if is_positive else scores).get(key)
                    return {
                        "score": number, "available": number is not None,
                        "score_date": key[1], "filter_id": filter_id,
                        "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                    }

                with portfolio_entry_policy_context(arm.entry_strategy,
                        score_keys=scores.keys() if arm.entry_strategy == "Daily" else None), \
                        audit_oracle_truth_score_context(
                            scores, ranking_policy=arm.dl_runtime_mode,
                            ranking_options=options, key_mode="information_date",
                            positive_scores=positives if qualified else None,
                        ):
                    oracle_result = run_portfolio_timeline(**arguments)
                with portfolio_entry_policy_context(arm.entry_strategy,
                        score_keys=scores.keys() if arm.entry_strategy == "Daily" else None), \
                        breakout_quality_ranking_source_context(
                            score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                            model_architecture=primary.model_architecture,
                            experiment_profile=primary.experiment_profile,
                            ranking_policy=arm.dl_runtime_mode, ranking_options=options,
                        ), patch(
                            "filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score",
                            side_effect=pit_score,
                        ):
                    pit_result = run_portfolio_timeline(**arguments)

                with self.subTest(arm=arm_id, positive_zero_exclusion=excluded):
                    self.assertEqual(len(oracle_result), len(pit_result))
                    for index, (observed, expected) in enumerate(zip(oracle_result, pit_result)):
                        if isinstance(observed, pd.DataFrame):
                            pd.testing.assert_frame_equal(observed, expected, check_exact=True)
                        elif isinstance(observed, float) and pd.isna(observed) and pd.isna(expected):
                            continue
                        else:
                            self.assertEqual(observed, expected, f"portfolio output index {index}")
                    if not excluded:
                        unfiltered_trades = int(oracle_result[4])
                        self.assertGreater(unfiltered_trades, 0)
                    else:
                        self.assertLess(int(oracle_result[4]), unfiltered_trades)
                    if arm.entry_strategy == "Daily":
                        self.assertEqual(int(oracle_result[4]), 3 if excluded else 4)
                self.assertIsNone(current_daily_entry_source())

    def test_formal_daily_arm_identity_and_old_breakout_reuse_payload(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        settings = get_strategy_comparison_settings(definition.source["strategy_profile_id"])
        for old_id, new_id in (("C135", "C133"), ("C136", "C134")):
            old_arm, new_arm = settings.arms[old_id], settings.arms[new_id]
            self.assertEqual(old_arm.entry_strategy, "Breakout")
            self.assertNotIn("entry_strategy", old_arm.as_dict())
            self.assertEqual(new_arm.entry_strategy, "Daily")
            self.assertEqual(new_arm.as_dict()["entry_strategy"], "Daily")
            self.assertEqual(old_arm.dl_runtime_options, new_arm.dl_runtime_options)
            self.assertEqual(old_arm.dl_id, new_arm.dl_id)
            self.assertEqual(old_arm.param_source, new_arm.param_source)
            self.assertEqual(old_arm.rule_policy, new_arm.rule_policy)
            self.assertEqual(new_arm.replay_contract_version, 4)
            self.assertEqual(old_arm.replay_contract_version, 2)  # Current two-model identity.
            self.assertNotIn("entry_strategy", STRATEGY_COMPARE_ARMS[old_id])
        for strategy, scoped in oracle._oracle_strategy_scopes(definition):
            plan = oracle._oracle_comparison_plan(scoped)
            resolved, base, exact = oracle._contract(scoped)
            direct = oracle._direct_arm(scoped, resolved, base, exact)
            for target in (
                "TrueDailyUniversal_MinTrade_NetPnL_Positive",
                "TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct",
            ):
                if target not in plan["selectors"]:
                    continue
                arm = oracle._target_reference_arm(scoped, resolved, plan, target, exact, direct)
                family_scope = strategy.split("::", 1)[0]
                self.assertEqual(arm.entry_strategy, family_scope)
                if strategy == "Daily::Daily_min":
                    expected = "C164"
                else:
                    expected = ("C133" if strategy == "Daily" else "C135") if (
                        plan["selectors"][target] == "exact") else (
                        "C134" if strategy == "Daily" else "C136")
                self.assertEqual(arm.arm_id, expected)

    def test_declared_strategy_rejects_wrong_scientific_target(self):
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        # Fail closed: changing an unbound Breakout treatment to Daily cannot
        # inherit a price or trade-path entry policy without a formal arm.
        from dataclasses import replace
        source = dict(definition.source)
        source["oracle_comparisons"] = tuple(
            {**row, "strategy": "Daily"} for row in definition.source["oracle_comparisons"]
            if row["target"] == "R40" and row.get("strategy", "Breakout") == "Breakout"
        )
        corrupted = replace(definition, source=source)
        with self.assertRaisesRegex(oracle.OracleBlockedError, "registered daily stock-day Target and formal arm"):
            oracle._oracle_target_strategy(corrupted, "R40")


    def test_daily_parameter_profiles_reuse_original_full_bounds_and_keep_p18(self):
        from pathlib import Path
        from config.strategy_params import STRATEGY_PARAM_PROFILES
        from core.strategy_param_scientific_contract import (
            build_strategy_param_scientific_contract,
        )
        from strategies.daily.search_space import resolve_daily_search_profile
        from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE
        from services.optimizer.strategy_param_training import _min_roos_search_fields

        cases = (
            ("broker_cash_daily_min_best_seed42_current", None, 3, "PARAM-P18"),
            ("broker_cash_daily_min_four_seed42_current", "min_atr_limit_v2", 4, "PARAM-P19"),
            ("broker_cash_daily_full_nineteen_seed42_current", "full_common_filters_v2", 19, "PARAM-P20"),
        )
        fingerprints = set()
        for profile_id, profile_name, count, identity in cases:
            with self.subTest(profile=profile_id):
                profile = STRATEGY_PARAM_PROFILES[profile_id]
                self.assertEqual(profile["parameter_set_identity"], identity)
                self.assertEqual(profile.get("daily_search_profile"), profile_name)
                search = _min_roos_search_fields("daily", profile_name)
                self.assertEqual(len(search), count)
                self.assertEqual(set(search), set(resolve_daily_search_profile(profile_name)["fields"]))
                for field in search:
                    self.assertIn(field, BREAKOUT_OPTIMIZER_SEARCH_SPACE)
                evidence = build_strategy_param_scientific_contract(Path(__file__).resolve().parents[2], profile_id)
                fingerprint = evidence["scientific_fingerprint_sha256"]
                self.assertEqual(fingerprint, profile["registered_scientific_fingerprint_sha256"])
                fingerprints.add(fingerprint)
        self.assertEqual(len(fingerprints), 3)
        self.assertEqual(_min_roos_search_fields("daily", "min_atr_limit_v2"),
                         ("atr_len", "atr_buy_tol", "atr_times_init", "atr_times_trail"))

    def test_daily_full_only_searches_live_filters_and_no_breakout_signals(self):
        from types import SimpleNamespace
        from config.strategy_params import STRATEGY_PARAM_PROFILES
        from services.optimizer.strategy_param_training import (
            _build_min_roos_fixed_overrides, _min_roos_search_fields,
        )
        entries = (
            ("broker_cash_daily_min_best_seed42_current", 3),
            ("broker_cash_daily_min_four_seed42_current", 4),
            ("broker_cash_daily_full_nineteen_seed42_current", 19),
        )
        for profile_id, count in entries:
            spec = STRATEGY_PARAM_PROFILES[profile_id]
            args = SimpleNamespace(
                strategy_id="daily", daily_search_profile=spec.get("daily_search_profile"),
                filter_id="breakout_quality_v1", fixed_risk=0.01,
                max_position_cap_pct=1.0,
            )
            overrides = _build_min_roos_fixed_overrides(args=args, training_dl_enabled=False)
            search = _min_roos_search_fields("daily", args.daily_search_profile)
            self.assertEqual(len(search), count)
            self.assertFalse(overrides["use_breakout_buy"])
            self.assertFalse(overrides["use_breakout_reclaim_reentry"])
            self.assertNotIn("high_len", search)
            self.assertTrue(all(field not in overrides for field in search))
            if count == 3:
                self.assertIn("atr_buy_tol", overrides)
            elif count == 4:
                self.assertFalse(overrides["use_bb"])
                self.assertFalse(overrides["use_vol"])
            else:
                switches = {"use_bb", "use_kc", "use_vol", "use_breakout_ema_filter",
                            "use_breakout_return_filter", "use_breakout_false_filter"}
                self.assertTrue(switches.issubset(search))
                self.assertTrue(switches.isdisjoint(overrides))

    def test_oracle_legacy_provenance_isolated_from_new_daily_candidate_capability(self):
        """An unrelated P20 candidate edit cannot invalidate old Oracle receipts.

        A shared numerical primitive edit *must* invalidate the legacy source.
        The P19 signature excludes the P20-only filter owner, but P20 includes it.
        """
        from pathlib import Path
        from unittest.mock import patch
        from services.research.oracle_artifact_contract import (
            _LEGACY_V1_REPLAY_IMPLEMENTATION,
            replay_implementation_sha256, portfolio_candidate_execution_sha256,
            execution_context,
        )
        from core.portfolio_entry_policy import (
            DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS,
        )
        legacy = replay_implementation_sha256()
        self.assertEqual(legacy, _LEGACY_V1_REPLAY_IMPLEMENTATION)
        p19 = portfolio_candidate_execution_sha256(
            daily_execution_contract=DAILY_EXECUTION_ATR_LIMIT)
        p20 = portfolio_candidate_execution_sha256(
            daily_execution_contract=DAILY_EXECUTION_FULL_FILTERS)
        self.assertNotEqual(p19, p20)
        original = Path.read_text
        daily_file = Path(__file__).resolve().parents[2] / "core/daily_common_entry_filters.py"

        def edited_daily_file(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            return value.replace("eligible = (volume > 0).copy()",
                                 "eligible = (volume >= 0).copy()") if path == daily_file else value

        portfolio_candidate_execution_sha256.cache_clear()
        try:
            with patch.object(Path, "read_text", edited_daily_file):
                self.assertEqual(replay_implementation_sha256(), legacy)
                self.assertEqual(portfolio_candidate_execution_sha256(
                    daily_execution_contract=DAILY_EXECUTION_ATR_LIMIT), p19)
                self.assertNotEqual(portfolio_candidate_execution_sha256(
                    daily_execution_contract=DAILY_EXECUTION_FULL_FILTERS), p20)
        finally:
            portfolio_candidate_execution_sha256.cache_clear()

        def digest(policy):
            return execution_context(
                dataset="full", price_sha256="a" * 64, param_sha256="b" * 64,
                seed=42, target="T40", start="2021-01-01", end="2026-03-02",
                cutoff="2026-03-02", max_positions=10, evaluation_mode="rolling",
                ranking_policy="direct", ranking_options={}, replay_policy=policy,
                score_contract=None, source_contract=None, dataset_inputs=None,
                execution_gate=None,
            )
        old = digest({"entry_policy": "daily_stockday_close_limit_next_exchange_session_v1"})
        self.assertNotIn("daily_candidate_implementation_sha256", old["replay_policy"])
        self.assertEqual(old["replay_implementation_sha256"], legacy)
        for mode, expected in ((DAILY_EXECUTION_ATR_LIMIT, p19),
                               (DAILY_EXECUTION_FULL_FILTERS, p20)):
            actual = digest({"daily_execution_contract": mode})
            self.assertEqual(actual["replay_policy"]["daily_candidate_implementation_sha256"], expected)
            self.assertNotEqual(actual, old)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            digest({"daily_execution_contract": DAILY_EXECUTION_FULL_FILTERS,
                    "daily_candidate_implementation_sha256": "0" * 64})

        # A change to an actually shared stop/entry primitive is never ignored.
        signal_file = Path(__file__).resolve().parents[2] / "core/signal_utils.py"
        def edited_shared_file(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == signal_file:
                self.assertIn("raw_buy_limits = C + ATR_main * params.atr_buy_tol", value)
                return value.replace("raw_buy_limits = C + ATR_main * params.atr_buy_tol",
                                     "raw_buy_limits = C + 2.0 * ATR_main * params.atr_buy_tol")
            return value
        replay_implementation_sha256.cache_clear()
        try:
            with patch.object(Path, "read_text", edited_shared_file):
                self.assertNotEqual(replay_implementation_sha256(), legacy)
        finally:
            replay_implementation_sha256.cache_clear()

    def test_daily_full_signal_primitive_parity_in_canonical_bundle(self):
        """Daily Full filtering uses shared features; legacy generator stays unchanged."""
        from dataclasses import replace
        import numpy as np
        from tools.validate.synthetic_daily_trade40_cases import fixture
        from core.portfolio_fast_data import prep_optimizer_stock_data_bundle
        from core.signal_utils import generate_signals
        from core.portfolio_entry_policy import (
            DAILY_EXECUTION_FULL_FILTERS, daily_signal_execution_context,
        )
        _, frame, params, _ = fixture(count=160)
        params = replace(params, use_breakout_buy=False)
        scenarios = (
            (params, 160),
            (replace(params, use_vol=True, vol_long_len=5, vol_breakout_mult=1.5), 0),
            (replace(params, use_breakout_ema_filter=True, breakout_ema_len=21), 156),
            (replace(params, use_bb=True, bb_len=14, bb_mult=2.0), 6),
            (replace(params, use_breakout_return_filter=True, breakout_return_min=0.01), 3),
            (replace(params, use_breakout_false_filter=True,
                     breakout_false_filter_atr_pct_min=0.02), 4),
        )
        for configuration, expected in scenarios:
            with self.subTest(expected=expected), daily_signal_execution_context(
                    DAILY_EXECUTION_FULL_FILTERS):
                self.assertFalse(np.any(generate_signals(frame, configuration)[1]),
                                 "The legacy signal owner cannot acquire P20-only branches")
                dynamic, *_ = prep_optimizer_stock_data_bundle(
                    frame, configuration, include_trade_logs=False,
                    include_pit_stats_index=False)
                self.assertEqual(int(np.count_nonzero(dynamic["is_setup"])), expected)
                self.assertEqual(int(np.count_nonzero(np.isfinite(dynamic["buy_limit"]))),
                                 int(np.count_nonzero(dynamic["is_setup"] &
                                                        np.isfinite(dynamic["ATR"]))))

    def test_scoped_oracle_policy_uses_registered_parameter_execution(self):
        from core.audit_policy import get_reusable_audit_definitions
        from services.audit.target_oracle_romd import (
            _oracle_strategy_scopes, _contract, _canonical_oracle_replay_policy,
        )
        from core.portfolio_entry_policy import (
            DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS,
        )
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == "target_ground_truth_strategy_romd")
        names = {}
        for scope, selected in _oracle_strategy_scopes(definition):
            _, base, exact = _contract(selected)
            names[base.arm_id] = _canonical_oracle_replay_policy(
                selected, base_arm=base, exact_arm=exact)
        self.assertNotIn("daily_execution_contract", names["C96"])
        self.assertNotIn("daily_execution_contract", names["C161"])
        self.assertEqual(names["C166"]["daily_execution_contract"], DAILY_EXECUTION_ATR_LIMIT)
        self.assertEqual(names["C168"]["daily_execution_contract"], DAILY_EXECUTION_FULL_FILTERS)

    def test_daily_full_signal_filters_are_real_and_other_modes_preserved(self):
        from dataclasses import replace
        import numpy as np
        from tools.validate.synthetic_daily_trade40_cases import fixture
        from core.portfolio_fast_data import prep_optimizer_stock_data_bundle
        from core.portfolio_entry_policy import (
            DAILY_EXECUTION_FULL_FILTERS, DAILY_EXECUTION_ATR_LIMIT,
            DAILY_EXECUTION_LEGACY, daily_signal_execution_context,
            current_daily_execution_contract,
        )
        _, frame, params, _ = fixture(count=160)
        params = replace(params, use_breakout_buy=False)
        def count(contract, cfg):
            with daily_signal_execution_context(contract):
                dynamic, _, _ = prep_optimizer_stock_data_bundle(
                    frame, cfg, include_trade_logs=False, include_pit_stats_index=False,
                )
                return int(np.count_nonzero(dynamic["is_setup"]))
        self.assertEqual(count(DAILY_EXECUTION_LEGACY, params), 0)
        self.assertEqual(count(DAILY_EXECUTION_ATR_LIMIT, params), 0)
        self.assertGreater(count(DAILY_EXECUTION_FULL_FILTERS, params), 0)
        vol_gate = replace(params, use_vol=True, vol_long_len=5, vol_breakout_mult=1.5)
        self.assertEqual(count(DAILY_EXECUTION_FULL_FILTERS, vol_gate), 0)
        self.assertEqual(current_daily_execution_contract(), DAILY_EXECUTION_LEGACY)

    def test_daily_new_limit_and_full_candidate_gate_are_replay_effective(self):
        from types import SimpleNamespace
        from core.portfolio_entry_policy import (
            DAILY_EXECUTION_FULL_FILTERS, DAILY_EXECUTION_ATR_LIMIT,
            portfolio_entry_policy_context,
        )
        info, session = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
        values = {info: {"buy_limit": 110.0, "ATR": 3.0, "Close": 100.0,
                         "is_setup": True},
                  session: {"buy_limit": 111.0, "ATR": 3.0, "Close": 101.0,
                            "is_setup": True}}
        kwargs = dict(
            ticker="2330", y_pos=0, portfolio={}, sold_today=set(),
            all_dfs_fast={"2330": values}, active_extended_signals={},
            pit_stats_index={"2330": []}, pit_stats_cursor={}, today=session,
            params=SimpleNamespace(atr_buy_tol=1.5), profile_stats=None,
            entry_policy="daily",
        )
        with portfolio_entry_policy_context("Daily", unscored=True):
            self.assertEqual(_resolve_qualified_normal_setup(**kwargs)[0], 100.0)
        for mode in (DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS):
            with portfolio_entry_policy_context("Daily", unscored=True,
                                                daily_execution_contract=mode):
                actual = _resolve_qualified_normal_setup(**kwargs)
                self.assertEqual(actual[0], 104.5)
        values[info]["is_setup"] = False
        with portfolio_entry_policy_context("Daily", unscored=True,
                                            daily_execution_contract=DAILY_EXECUTION_FULL_FILTERS):
            self.assertIsNone(_resolve_qualified_normal_setup(**kwargs))
        with portfolio_entry_policy_context("Daily", unscored=True,
                                            daily_execution_contract=DAILY_EXECUTION_ATR_LIMIT):
            self.assertIsNotNone(_resolve_qualified_normal_setup(**kwargs))

    def test_daily_signal_contract_is_explicit_in_worker_and_replay_caches(self):
        from concurrent.futures import ThreadPoolExecutor
        from dataclasses import replace
        import numpy as np
        from tools.validate.synthetic_daily_trade40_cases import fixture
        from services.optimizer.trial_inputs import worker_prep_data
        from services.portfolio_replay import _replay_preparation_key
        from core.portfolio_entry_policy import (
            DAILY_EXECUTION_FULL_FILTERS, DAILY_EXECUTION_LEGACY,
            daily_signal_execution_context,
        )
        _, frame, params, _ = fixture(count=160)
        params = replace(params, use_breakout_buy=False)
        def prepare(contract):
            return worker_prep_data("2330", frame, params, include_trade_logs=False,
                                    include_pit_stats_index=False, profile_enabled=False,
                                    daily_execution_contract=contract)
        with ThreadPoolExecutor(max_workers=2) as pool:
            old, new = list(pool.map(prepare, (DAILY_EXECUTION_LEGACY, DAILY_EXECUTION_FULL_FILTERS)))
        self.assertTrue(old["ok"] and new["ok"])
        self.assertFalse(np.any(old["dynamic"]["is_setup"]))
        self.assertTrue(np.any(new["dynamic"]["is_setup"]))
        record = {"effective_date_text": "2021-01-01", "effective_end_date_text": "2021-12-31",
                  "params_signature": "signed"}
        kw = dict(ensemble=False, raw_universe_required_min_rows=None,
                  keep_trade_logs=False, use_prepared_cache=False, write_prepared_cache=False)
        before = _replay_preparation_key(".", [record], **kw)
        with daily_signal_execution_context(DAILY_EXECUTION_FULL_FILTERS):
            after = _replay_preparation_key(".", [record], **kw)
        self.assertNotEqual(before, after)
        self.assertEqual(after[-1], DAILY_EXECUTION_FULL_FILTERS)

    def test_new_daily_direct_arms_have_separate_control_and_oracle_truth(self):
        from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
        live = {
            "C166": ("daily_min_four_best_broker_cash_rolling_experiment", False),
            "C167": ("daily_min_four_best_broker_cash_rolling_experiment", True),
            "C168": ("daily_full_nineteen_best_broker_cash_rolling_experiment", False),
            "C169": ("daily_full_nineteen_best_broker_cash_rolling_experiment", True),
        }
        for arm_id, (param_source, dl_enabled) in live.items():
            arm = STRATEGY_COMPARE_ARMS[arm_id]
            self.assertEqual(arm["param_source"], param_source)
            self.assertEqual(arm["entry_strategy"], "Daily")
            self.assertEqual(arm["dl_enabled"], dl_enabled)
            self.assertEqual(arm["dl_id"], "CONT13CJ_ROLL" if dl_enabled else None)
            if dl_enabled:
                self.assertIn("no-k-no-r0", arm["dl_runtime_mode"])
        # Witness registrations are immutable but never enter current execution.
        # Current inventory is generated from the same canonical registry.
        from config.research_summary import RESEARCH_SUMMARY_ROWS
        bindings = str(RESEARCH_SUMMARY_ROWS)
        for name in ("C166", "C167", "C168", "C169"):
            self.assertIn(name, bindings)
        self.assertNotIn("C170", bindings)
        self.assertNotIn("C171", bindings)
        self.assertEqual(STRATEGY_COMPARE_ARMS["C160"]["param_source"],
                         "daily_min_best_broker_cash_rolling_experiment")


if __name__ == "__main__":
    main()

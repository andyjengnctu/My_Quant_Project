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
    def test_daily_strategy_records_follow_canonical_comparisons(self):
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        rows = tuple(definition.source["oracle_comparisons"])
        from core.positive_netpnl_contract import requires_positive_score
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        from core.breakout_quality_runtime import get_continuous_ranker_score_output_columns
        # AI註: Validate each declaration without freezing current roster membership.
        settings, base, exact = oracle._contract(definition)
        scoped = oracle._oracle_strategy_scopes(definition)
        self.assertEqual([(s, len(d.source["oracle_comparisons"])) for s,d in scoped],
                         [(strategy, sum(r.get("strategy", "Breakout") == strategy for r in rows))
                          for strategy in ("Breakout", "Daily")
                          if any(r.get("strategy", "Breakout") == strategy for r in rows)])
        for strategy, scope in scoped:
            plan = oracle._oracle_comparison_plan(scope)
            direct = oracle._direct_arm(scope, settings, base, exact)
            contracts = oracle._resolve_oracle_target_score_contracts(scope, settings, plan)
            if strategy != "Daily":
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
                self.assertEqual(profile.continuous_target_id, contracts[row["target"]]["target_id"])
                self.assertEqual(contracts[row["target"]]["target_id"],
                                 "true_daily_universal_mintrade_fulltrade_netpnl_v1")
        self.assertTrue(all(oracle._oracle_target_strategy(d, r["target"]) == strategy
                            for strategy, d in scoped
                            for r in d.source["oracle_comparisons"]))
        self.assertEqual(oracle._oracle_target_strategy(definition, "C96"), "Breakout")
        contract = require_approved_report_contract("audit.target_oracle_romd")
        self.assertEqual(contract.version, 13)
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
            self.assertEqual({p.get("strategy", "Breakout") for p in pair},
                             {"Breakout", "Daily"})
            self.assertEqual({tuple(sorted((k, repr(v)) for k,v in p.items()
                                           if k not in {"strategy", "strategy_arm_id"})) for p in pair}.__len__(), 1)
        with self.assertRaisesRegex(ValueError, "duplicate Target x Strategy"):
            resolve_current_oracle_comparisons({
                **definition.source, "oracle_comparisons": (*rows, dict(rows[-1])),
            })
        # The inventory's second dimension is not a new scientific Target ID.
        self.assertIn("TrueDailyUniversal_MinTrade_NetPnL_Positive::Daily", keys)
        self.assertIn("TrueDailyUniversal_MinTrade_NetPnL_Positive", keys)

    def test_one_approval_matrix_and_same_engine_for_declared_treatments(self):
        from core.audit_registry import oracle_comparison_key
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        seeds = tuple(range(1, 9))
        comparisons = tuple(definition.source["oracle_comparisons"])
        scope_counts = [(strategy, len(scope.source["oracle_comparisons"]))
                        for strategy, scope in oracle._oracle_strategy_scopes(definition)]
        reuse_count = len(seeds) * (1 + sum(row.get("strategy", "Breakout") != "Daily"
                                          for row in comparisons))
        run_count = len(seeds) * sum(row.get("strategy") == "Daily" for row in comparisons)
        calls = []
        common_baseline = {"target":"C96", "strategy":"Breakout",
                           "seed_observations": [], "romd": 9.0}

        def fake_run(scoped, *, preview_only=False, **kwargs):
            strategy = scoped.source["oracle_comparisons"][0].get("strategy", "Breakout")
            targets = tuple(row["target"] for row in scoped.source["oracle_comparisons"])
            calls.append((strategy, preview_only, kwargs.get("approved_replay_decisions")))
            decisions = {(seed, name): "REUSE" if strategy == "Breakout" else "RUN"
                         for seed in seeds for name in ("C96", *targets)}
            for seed in seeds:
                decisions[(seed,"C96")] = "REUSE"
            if preview_only:
                return {"status":"PREVIEW", "preflight": decisions,
                        "benchmark_id":"end_to_end_v2", "completed_observations":{},
                        "start_date":"2021-01-01", "end_date":"2026-03-02",
                        "seed_set":seeds,"missing_p16":(),
                        "producer_kwargs":{"method":"canonical"},
                        "targets": ("C96",*targets)}
            self.assertTrue(kwargs["preflight_approval"]("single scope recheck"))
            self.assertEqual(kwargs["approved_replay_decisions"], decisions)
            return {"status":"COMPLETED", "seeds":list(seeds),
                    "benchmark_id":"end_to_end_v2", "dataset":"full",
                    "start_date":"2021-01-01", "end_date":"2026-03-02",
                    "strategy_profile_id": "test", "param_artifacts":{},
                    "common_cohort":{"cohort_rows":630589},
                    "baseline":common_baseline,
                    "oracles":[{"target": target, "strategy":strategy,
                                "romd":25.0 if strategy == "Breakout" else 40.0}
                               for target in targets],
                    "definitions":{row["target"]:row["description"]
                                   for row in scoped.source["oracle_comparisons"]}}
        with patch.object(oracle,"_run_single_stage_oracle",side_effect=fake_run):
            preview = oracle.preview_oracle_replay_plan(definition,project_root="/tmp")
            self.assertEqual(len(preview["targets"]), 1 + len(comparisons))
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
        self.assertEqual(len(result["comparison_inventory"]), 1 + len(comparisons))
        target = "TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct"
        values = {(row["target"],row["strategy"]):row["romd"] for row in result["oracles"]}
        self.assertEqual(values[(target,"Breakout")],25.0)
        self.assertEqual(values[(target,"Daily")],40.0)
        self.assertEqual([c[1] for c in calls], [True,True,False,False])
        self.assertEqual([(c[0],len(c[2])) for c in calls[-2:]],
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
        arms = [arm for arm in settings.arms.values()
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
                self.assertEqual(arm.entry_strategy, strategy)
                expected = ("C133" if strategy == "Daily" else "C135") if (
                    plan["selectors"][target] == "exact") else (
                    "C134" if strategy == "Daily" else "C136")
                self.assertEqual(arm.arm_id, expected)

    def test_declared_strategy_rejects_wrong_scientific_target(self):
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        # Fail-closed: a new unrelated Daily target must not inherit this policy.
        from dataclasses import replace
        source = dict(definition.source)
        source["oracle_comparisons"] = tuple(
            dict(row, strategy="Daily") if row["target"] == "T40" else row
            for row in definition.source["oracle_comparisons"]
        )
        corrupted = replace(definition, source=source)
        with self.assertRaisesRegex(oracle.OracleBlockedError, "signed TrueDaily"):
            oracle._oracle_target_strategy(corrupted, "T40")


if __name__ == "__main__":
    main()

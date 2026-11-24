"""Independent invariants for the P20 Daily Full full-cycle NetPnL primitive.

P20 has its own economic teacher and model FIT identities.  These tests do not
run or emulate the formal suite and never create research results.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from tools.validate.oracle_fixture_build import OracleFixtureTestCase as TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.file_integrity import canonical_json_sha256
from core.breakout_quality_registry import get_continuous_ranker_research_spec
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.research_summary import get_research_summary_strategy_arm_ids
from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
from filters.breakout_quality.daily_trade40_target import ExecutableMinPeriod
from filters.breakout_quality.daily_full_p20_target import compute_daily_full_p20_batch
from services.breakout_quality.daily_full_p20_pit_source import materialize_daily_full_p20_pit_labels
from services.breakout_quality.daily_full_p20_training_source import (
    SOURCE_ID, inspect_history, execute_history, historical_params_path,
)
from services.optimizer.optimizer_workspace_recovery import (
    adopt_nested_optimizer_workspace, inspect_nested_optimizer_workspace,
)
from services.breakout_quality.formal_mintrade40_pit_source import (
    FormalMinTrade40PITBlocked, FormalMinTrade40PITSource,
)
from tools.validate.synthetic_daily_trade40_cases import fixture


RANK_PROFILE = "daily_universal_full_p20_fulltrade_netpnl_full_list_ndcg_pairwise"
POS_PROFILE = "daily_universal_full_p20_fulltrade_netpnl_positive_binary"


def _fixture_source(seed=42):
    days, frame, original, _ = fixture(count=160)
    params = replace(original, use_breakout_buy=False)
    period = ExecutableMinPeriod(
        start=days[0], end=None, params=params, source_id=SOURCE_ID,
        params_signature=build_portfolio_params_signature(params),
    )
    source = FormalMinTrade40PITSource(
        seed=seed, param_sha256="a" * 64, schedule=(period,),
        publication_dates=(str((days[0] - pd.Timedelta(days=1)).date()),),
        source_id=SOURCE_ID, oos_start_date=str(days[0].date()),
    )
    return days, frame, source


class DailyFullP20NetPnLContractCases(TestCase):
    @staticmethod
    def _seeds():
        from config.research import RESEARCH_SINGLE_SEED
        from core.strategy_compare_policy import get_strategy_multi_seed_robustness_settings
        return (RESEARCH_SINGLE_SEED, *get_strategy_multi_seed_robustness_settings(
            "broker_cash_dl_conversion_rolling").resolved_seeds)

    @staticmethod
    def _record(source, seed, *, start=None, end=None):
        period = source.schedule[0]
        return {"params_obj": period.params, "params_signature": period.params_signature,
                "effective_date": start or str(period.start.date()),
                "effective_end_date": end, "members": [{"seed": seed}]}

    def test_seed_paths_never_fall_back_to_single_seed_history(self):
        from services.breakout_quality.daily_full_p20_training_source import TEACHER as owner
        with TemporaryDirectory() as tmp:
            seeds = self._seeds()
            original = historical_params_path(tmp, seeds[0])
            original.parent.mkdir(parents=True)
            original.write_bytes(b"original-seed-only")
            with patch.object(owner, "_validated_records"):
                self.assertEqual(inspect_history(project_root=tmp, seed=seeds[0])["action"], "REUSE")
                for seed in seeds[1:]:
                    state = inspect_history(project_root=tmp, seed=seed)
                    self.assertEqual(state["action"], "BUILD")
                    self.assertEqual(state["path"], historical_params_path(tmp, seed))
                    self.assertNotEqual(state["path"], original)
            self.assertEqual(original.read_bytes(), b"original-seed-only")
            for seed in (None, True, 0, -1, "42"):
                with self.subTest(seed=seed), self.assertRaises(FormalMinTrade40PITBlocked):
                    historical_params_path(tmp, seed)

    def test_each_teacher_dependency_delegates_matching_seed_to_optimizer(self):
        from services.breakout_quality.daily_full_p20_training_source import TEACHER as owner
        from services.breakout_quality.trade_path_target_source import registered_trade_path_source
        rank = registered_trade_path_source(RANK_PROFILE)
        self.assertIs(rank, registered_trade_path_source(POS_PROFILE))
        self.assertEqual(len(rank.dependencies), 2)
        with TemporaryDirectory() as tmp:
            for seed in self._seeds():
                expected = historical_params_path(tmp, seed)
                with (self.subTest(seed=seed),
                    patch.object(owner, "_validated_records") as verify,
                    patch.object(owner, "prepare_selection_historical_p2_params",
                                 return_value={"params_path": expected}) as build,
                    patch.object(owner, "inspect_formal_mintrade40_p16_teacher", return_value={}) as inspect,
                    patch.object(owner, "_current_p16_seed_artifact") as current,
                ):
                    rank.dependencies[0].execute(project_root=tmp, seed=seed)
                    self.assertEqual(build.call_args.kwargs["optimizer_seed"], seed)
                    self.assertEqual(build.call_args.kwargs["output_relative_dir"], expected.parent.parent.parent)
                    self.assertEqual(verify.call_args.kwargs["seed"], seed)
                    rank.dependencies[1].inspect(project_root=tmp, seed=seed)
                    self.assertEqual(inspect.call_args.kwargs["seed"], seed)
                    self.assertEqual(inspect.call_args.kwargs["baseline_arm"], owner.BASELINE_ARM)
                    rank.dependencies[1].execute(project_root=tmp, seed=seed)
                    self.assertEqual(current.call_args.args[1], seed)
                    self.assertTrue(current.call_args.kwargs["authorized_build"])

    def test_historical_and_current_source_and_both_heads_bind_the_same_seed(self):
        from services.breakout_quality.daily_full_p20_training_source import TEACHER as owner
        from services.breakout_quality.trade_path_target_source import loader_kwargs_for_trade_path
        from core.breakout_quality_runtime import trade_path_fit_capability
        contracts = []
        with TemporaryDirectory() as tmp:
            for seed in self._seeds():
                _, _, fixture_source = _fixture_source(seed)
                past = self._record(fixture_source, seed, start="2014-01-01", end="2020-12-31")
                current = self._record(fixture_source, seed, start="2021-01-01")
                history_payload = {"folds": [{"effective_start": "2014-01-01", "selection_end_date": "2013-12-31"}]}
                current_payload = {"folds": [{"effective_start": "2021-01-01", "selection_end_date": "2020-12-31"}]}
                with (self.subTest(seed=seed),
                    patch.object(owner, "_current_p16_seed_artifact", return_value=(Path(tmp)/"current", "2021-01-01", "2026-03-02", "unused")) as get_current,
                    patch.object(owner, "_validated_records", return_value=(history_payload, [past])) as get_history,
                    patch.object(owner, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": current_payload}, "c"*64)),
                    patch.object(owner, "resolve_formal_min_param_records", return_value=[current]),
                    patch.object(owner, "compute_strategy_param_file_sha256", return_value="b"*64),
                ):
                    loaders = [loader_kwargs_for_trade_path(experiment_profile=p, project_root=tmp, seed=seed)
                               for p in (RANK_PROFILE, POS_PROFILE)]
                    self.assertEqual(loaders[0]["target_lineage"], loaders[1]["target_lineage"])
                    self.assertEqual(get_current.call_args.args[1], seed)
                    self.assertEqual(get_history.call_args.kwargs["seed"], seed)
                    self.assertEqual(get_history.call_args.args[0], historical_params_path(tmp, seed))
                    lineage = loaders[0]["target_lineage"]
                    source = loaders[0]["target_materializer"].keywords["source"]
                    self.assertEqual(source.seed, seed)
                    self.assertEqual(loaders[0]["target_materializer"].keywords["expected_seed"], seed)
                    self.assertEqual(lineage["seed"], lineage["optimizer_teacher_seed"])
                    self.assertEqual(lineage["contract"], trade_path_fit_capability(lineage).fit_teacher_contract)
                    self.assertEqual([p.source_id for p in source.schedule], [SOURCE_ID, SOURCE_ID])
                    contracts.append(source.param_sha256)
                    # A copied other-seed current artifact must fail even if its settings match.
                    current["members"] = [{"seed": seed + 1}]
                    with self.assertRaisesRegex(FormalMinTrade40PITBlocked, "seed/member"):
                        owner.load_training_source(project_root=tmp, seed=seed)
        self.assertEqual(len(set(contracts)), len(contracts))

    def test_oracle_schedule_rejects_foreign_seed_members_before_truth_build(self):
        from services.audit.target_oracle_romd import _oracle_daily_schedule
        for seed in self._seeds():
            _, _, source = _fixture_source(seed)
            record = self._record(source, seed)
            args = dict(seed=seed, param_sha256="a"*64, recipe="daily_trade_path_daily_full_p20_net_pnl")
            schedule, evidence = _oracle_daily_schedule([record], **args)
            self.assertEqual(schedule, source.schedule)
            self.assertEqual(evidence["seed"], seed)
            for members in ([], [{"seed": seed + 1}], [{"seed": seed}, {"seed": seed + 1}]):
                with self.subTest(seed=seed, members=members), self.assertRaisesRegex(FormalMinTrade40PITBlocked, "seed/member"):
                    _oracle_daily_schedule([{**record, "members": members}], **args)

    def test_current_teacher_builds_the_registered_replica_without_production_fallback(self):
        from services.breakout_quality.daily_full_p20_training_source import TEACHER as teacher
        from services.breakout_quality import formal_mintrade40_training_source as canonical
        from core.strategy_compare_policy import get_strategy_comparison_settings
        settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        arm = settings.arms[teacher.BASELINE_ARM]
        training_profile = settings.parameter_sources[arm.param_source].builder.options["training_profile_id"]
        with TemporaryDirectory() as tmp:
            for seed in self._seeds()[1:]:
                path = Path(tmp)/str(seed)/"params.json"
                missing = {"ready": False, "action": "BUILD", "path": path}
                ready = {**missing, "ready": True, "action": "REUSE"}
                with (self.subTest(seed=seed),
                    patch.object(canonical, "resolve_fixed_multiseed_robustness_strategy_parameter_artifact",
                                 side_effect=[missing, missing, ready]) as inspect,
                    patch.object(canonical, "ensure_fixed_multiseed_robustness_strategy_parameter_artifact") as build,
                    patch.object(canonical, "ensure_fixed_multiseed_strategy_parameter_artifact",
                                 side_effect=AssertionError("No production seed fallback")),
                ):
                    self.assertEqual(teacher.inspect_current(project_root=tmp, seed=seed)["action"], "BUILD")
                    self.assertEqual(teacher.execute_current(project_root=tmp, seed=seed)[0], path)
                    self.assertEqual(build.call_args.kwargs["replication_seed"], seed)
                    self.assertEqual(build.call_args.kwargs["training_profile_id"], training_profile)
                    self.assertTrue(all(call.kwargs["replication_seed"] == seed for call in inspect.call_args_list))

    def test_robustness_plans_and_builds_each_teacher_once_for_rank_and_positive(self):
        from services.research import strategy_multi_seed_robustness as robust
        from services.breakout_quality.daily_full_p20_training_source import TEACHER as teacher
        from core.strategy_compare_policy import get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings
        settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        cfg = get_strategy_multi_seed_robustness_settings(settings.profile_id)
        ready, built = set(), []
        def state(kind, **kwargs):
            key = (kind, kwargs["seed"])
            return {"action": "REUSE" if key in ready else "BUILD", "reason": kind}
        def build(kind, **kwargs):
            key = (kind, kwargs["seed"])
            built.append(key)
            ready.add(key)
        with (
            patch.object(teacher, "inspect_history", side_effect=lambda **kw: state("history", **kw)),
            patch.object(teacher, "inspect_current", side_effect=lambda **kw: state("current", **kw)),
            patch.object(teacher, "execute_history", side_effect=lambda **kw: build("history", **kw)),
            patch.object(teacher, "execute_current", side_effect=lambda **kw: build("current", **kw)),
        ):
            plan_args = dict(cfg=cfg, settings=settings, model_arms=(settings.arms["C172"], settings.arms["C173"]),
                             robustness_unit_states={}, model_source_reuse={})
            rows, blocked = robust._trade_path_seed_plan(**plan_args)
            self.assertEqual(blocked, [])
            self.assertEqual(len(rows), 2 * len(cfg.resolved_seeds))
            self.assertEqual(len({key for _, key, _ in rows}), len(rows))
            self.assertTrue(all(action == "BUILD" for action, _, _ in rows))
            for seed in cfg.resolved_seeds:
                for profile in (RANK_PROFILE, POS_PROFILE):
                    robust._prepare_trade_path_teacher_for_model(experiment_profile=profile, seed=seed, authorized=True)
            self.assertEqual(set(built), {(kind, seed) for kind in ("history", "current") for seed in cfg.resolved_seeds})
            self.assertEqual(len(built), len(set(built)))
            rows, blocked = robust._trade_path_seed_plan(**plan_args)
            self.assertEqual(blocked, [])
            self.assertTrue(all(action == "REUSE" for action, _, _ in rows))

    def test_fixed_teacher_fit_contract_is_historical_only(self):
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        from core.breakout_quality_runtime import trade_path_fit_capability
        for profile in (RANK_PROFILE, POS_PROFILE):
            spec = get_breakout_quality_experiment_profile(profile)
            self.assertEqual(spec.fit_contract_version, 3)
            self.assertNotEqual(spec.continuous_target_id, "daily_universal_full_p20_fullcycle_netpnl_seed42_v1")
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            trade_path_fit_capability({"target_id": "daily_universal_full_p20_fullcycle_netpnl_seed42_v1",
                "target_kind": "daily_full_p20_seed42_fullcycle_netpnl",
                "teacher_lineage": {"contract": "daily_full_p20_historical_plus_current_v1", "seed": 42}})

    def test_real_per_seed_oracle_truth_matches_labels_and_reuses_signed_witness(self):
        from types import SimpleNamespace
        from services.audit import daily_full_p20_oracle_truth as oracle
        from services.audit.target_oracle_romd import _build_daily_truth, _oracle_daily_schedule
        from filters.breakout_quality.daily_trade40_target import rank_trade40_outcome_frame
        recipe = "daily_trade_path_daily_full_p20_net_pnl"
        label_arrays = []
        with TemporaryDirectory() as tmp:
            for index, seed in enumerate(self._seeds()):
                days, frame, source = _fixture_source(seed)
                # Different effective teacher parameters must affect economics, not only hashes.
                params = replace(source.schedule[0].params, atr_times_init=1.0 + index, atr_times_trail=1.0 + index)
                period = replace(source.schedule[0], params=params, params_signature=build_portfolio_params_signature(params))
                source = replace(source, schedule=(period,))
                schedule, _ = _oracle_daily_schedule([self._record(source, seed)], seed=seed,
                    param_sha256="a"*64, recipe=recipe)
                positions = np.arange(40, 105, dtype=np.int64)
                cohort = pd.DataFrame({"ticker": "2330", "date": days.take(positions)})
                bundle = SimpleNamespace(group_table=cohort, summary={"dataset": "full", "filter_id": "test",
                    "sample_scope": "daily_eligible_stock_days", "score_eligible_sample_count": len(cohort)})
                args = dict(project_root=Path(tmp), dataset="full", filter_id="test", profile_id=RANK_PROFILE,
                    cutoff=str(days[-1].date()), evaluation_start=str(days[1].date()), evaluation_end=str(days[-1].date()),
                    score_start=str(days[1].date()), seed=seed, param_sha256="a"*64, canonical_schedule=schedule,
                    price_sha256="b"*64, cache_root=Path(tmp)/"truth", output_subdir="audit")
                with (self.subTest(seed=seed),
                    patch.object(oracle, "oracle_dataset_summary_sha256", return_value="c"*64),
                    patch.object(oracle, "load_profile_continuous_ranker_data", return_value=bundle),
                    patch.object(oracle, "discover_dataset_csv_inputs", return_value=([("2330", Path(tmp)/"2330.csv")], [])),
                    patch.object(oracle, "collect_canonical_exchange_sessions", return_value=days),
                    patch.object(oracle, "load_dataset_frame", return_value=frame),
                ):
                    scores, evidence = _build_daily_truth(recipe, **args)
                    self.assertEqual(evidence["teacher"]["seed"], seed)
                    self.assertEqual(evidence["teacher"]["optimizer_seed"], seed)
                    self.assertEqual(evidence["teacher"]["periods"][0]["params_signature"], period.params_signature)
                    labels = materialize_daily_full_p20_pit_labels(source, expected_seed=seed,
                        frame=frame, positions=positions, ticker="2330", exchange_calendar=days,
                        fit_cutoff=days[-1], research_end_is_terminal=True)
                    label_arrays.append(labels.raw_net_pnl)
                    outcomes = pd.DataFrame({"ticker": "2330", "date": days.take(positions).strftime("%Y-%m-%d"),
                                             "realized_net_pnl": labels.raw_net_pnl})
                    self.assertEqual(scores, rank_trade40_outcome_frame(outcomes, value_column="realized_net_pnl"))
                    raw = oracle.load_positive_truth(project_root=Path(tmp), output_subdir="audit", seed=seed,
                        price_sha256="b"*64, param_sha256="a"*64, canonical_scores=scores, canonical_evidence=evidence)
                    expected_raw = {(row.ticker, row.date): row.realized_net_pnl for row in outcomes.itertuples() if np.isfinite(row.realized_net_pnl)}
                    self.assertEqual(raw, expected_raw)
                    with patch.object(oracle, "_run_stocks", side_effect=AssertionError("REUSE must not recompute")):
                        reused, receipt = _build_daily_truth(recipe, **args)
                        self.assertEqual(reused, scores)
                        self.assertEqual(receipt["teacher"], evidence["teacher"])
                    with self.assertRaisesRegex(FormalMinTrade40PITBlocked, "[Ss]eed"):
                        materialize_daily_full_p20_pit_labels(source, expected_seed=seed + 1,
                            frame=frame, positions=positions, ticker="2330", exchange_calendar=days, fit_cutoff=days[-1])
        self.assertTrue(any(not np.array_equal(label_arrays[0], other, equal_nan=True) for other in label_arrays[1:]))

    def test_current_oracle_source_catalog_has_registered_calculation_capabilities(self):
        """Active Summary recipes must resolve at the canonical source owner."""
        from core.audit_policy import get_reusable_audit_definitions
        from services.audit import target_oracle_romd as oracle
        from services.research.oracle_artifact_contract import source_calculation_contract

        definition = next(item for item in get_reusable_audit_definitions("breakout_quality")
                          if item.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        plan = oracle._oracle_comparison_plan(definition)
        source_recipes = (*plan["recipes"].values(), *plan["daily_recipes"].values())
        for recipe, provider, *horizon in source_recipes:
            with self.subTest(recipe=recipe):
                contract = source_calculation_contract(
                    recipe=recipe, profile_id=str(definition.source[provider]),
                    horizon=horizon[0] if horizon else None,
                    model_architecture=str(definition.source["model_architecture"]),
                )
                self.assertEqual(contract["recipe"], recipe)
                self.assertEqual(len(contract["implementation_sha256"]), 64)
                self.assertGreater(contract["calculation_revision"], 0)

    def test_p20_oracle_uses_its_actual_entry_and_truth_not_legacy_min_path(self):
        from services.research.oracle_artifact_contract import (
            source_calculation_contract, source_implementation_sha256,
        )
        recipe = "daily_trade_path_daily_full_p20_net_pnl"
        contract = source_calculation_contract(recipe=recipe, profile_id=RANK_PROFILE)
        self.assertEqual(contract["calculation_revision"], 3)
        self.assertIsNone(contract["horizon_sessions"])
        self.assertEqual(contract["score_mapping"], "canonical_daily_truth_percentile_v2")
        self.assertEqual(contract["implementation_sha256"],
                         source_implementation_sha256(recipe, RANK_PROFILE))
        self.assertNotEqual(contract["implementation_sha256"],
                            source_implementation_sha256("daily_trade_path_net_pnl", RANK_PROFILE))

    def test_p20_source_fingerprint_changes_without_invalidating_legacy_recipes(self):
        from services.research import oracle_artifact_contract as contracts
        recipe = "daily_trade_path_daily_full_p20_net_pnl"
        before = {key: contracts.source_implementation_sha256(key, RANK_PROFILE)
                  for key in (recipe, "daily_trade_path_net_pnl",
                              "daily_trade_path_true_net_pnl", "daily_trade_path_full_net_pnl")}
        original = contracts.calculation_source
        for relative, name in (
            ("core/daily_common_entry_filters.py", "daily_common_filter_signals"),
            ("filters/breakout_quality/daily_full_p20_target.py", "compute_daily_full_p20_batch"),
            ("services/audit/daily_full_p20_oracle_truth.py", "build_daily_full_p20_oracle_scores"),
        ):
            def edited(path, function, **kwargs):
                result = original(path, function, **kwargs)
                return "f" * 64 if (path, function) == (relative, name) else result

            contracts.source_implementation_sha256.cache_clear()
            with self.subTest(capability=name), patch.object(contracts, "calculation_source",
                                                      side_effect=edited):
                changed = {key: contracts.source_implementation_sha256(key, RANK_PROFILE)
                           for key in before}
            self.assertNotEqual(changed[recipe], before[recipe])
            self.assertEqual({key: value for key, value in changed.items() if key != recipe},
                             {key: value for key, value in before.items() if key != recipe})
        contracts.source_implementation_sha256.cache_clear()

    def test_new_rank_and_binary_head_share_p20_economic_truth_not_old_models(self):
        rank = get_continuous_ranker_research_spec(RANK_PROFILE)
        positive = get_continuous_ranker_research_spec(POS_PROFILE)
        self.assertEqual((rank.model_research_id, positive.model_research_id),
                         ("MR-13DB", "MR-13DC"))
        rank_runtime = get_continuous_ranker_execution_recipe(RANK_PROFILE)
        pos_runtime = get_continuous_ranker_execution_recipe(POS_PROFILE)
        self.assertEqual(rank_runtime.continuous_target_id, pos_runtime.continuous_target_id)
        self.assertEqual(rank_runtime.target_policy.contract_kind,
                         "daily_full_p20_fullcycle_netpnl_per_seed")
        self.assertEqual(pos_runtime.target_policy.contract_kind,
                         rank_runtime.target_policy.contract_kind)
        self.assertEqual(rank_runtime.training_policy.target_builder, "scalar_pairwise")
        self.assertEqual(pos_runtime.training_policy.target_builder, "netpnl_positive_binary")
        self.assertEqual(pos_runtime.training_policy.categorical_class_count, 2)
        self.assertEqual(rank_runtime.target_policy.horizon_bars, None)

    def test_current_strategy_registry_retirement_and_positive_only_on_c172(self):
        current = set(get_research_summary_strategy_arm_ids())
        self.assertTrue({"C168", "C169", "C172", "C173"}.issubset(current))
        self.assertTrue({"C166", "C167"}.isdisjoint(current))
        a, b = STRATEGY_COMPARE_ARMS["C172"], STRATEGY_COMPARE_ARMS["C173"]
        for item in (a, b):
            self.assertEqual(item["entry_strategy"], "Daily")
            self.assertEqual(item["param_source"],
                             "daily_full_nineteen_best_broker_cash_rolling_experiment")
            self.assertEqual(item["dl_id"], "CONT13DB_ROLL")
            self.assertFalse(item["dl_runtime_options"]["preserve_k"])
            self.assertFalse(item["dl_runtime_options"]["preserve_r0"])
        self.assertEqual(a["dl_runtime_options"]["positive_dl_id"], "CONT13DC_ROLL")
        self.assertEqual(a["dl_runtime_options"]["positive_probability_threshold"], 0.5)
        self.assertNotIn("positive_dl_id", b["dl_runtime_options"])

    def test_real_next_session_broker_label_has_both_signs_and_nulls(self):
        days, frame, source = _fixture_source()
        values = compute_daily_full_p20_batch(
            frame, np.arange(40, 105, dtype=np.int64), ticker="2330",
            schedule=source.schedule, exchange_calendar=days, research_cutoff=days[-1],
        )
        valid = np.asarray(values.valid_mask, dtype=bool)
        raw = np.asarray(values.target_raw_pnl, dtype=float)
        self.assertGreater(int(valid.sum()), 0)
        self.assertTrue(np.isnan(raw[~valid]).all())
        self.assertTrue(np.isfinite(raw[valid]).all())
        self.assertTrue(np.any(raw[valid] > 0))
        self.assertTrue(np.any(raw[valid] < 0))
        self.assertEqual(len(values.reasons), len(raw))

    def test_pit_maturity_gates_full_cycle_and_rejects_wrong_teacher(self):
        days, frame, source = _fixture_source()
        positions = np.arange(30, 93, dtype=np.int64)
        args = dict(source=source, expected_seed=42, frame=frame,
                    positions=positions, ticker="2330", exchange_calendar=days,
                    fit_cutoff=days[100], cache_project_root=None)
        labels = materialize_daily_full_p20_pit_labels(**args)
        valid = np.asarray(labels.completed, dtype=bool)
        self.assertGreater(int(valid.sum()), 0)
        self.assertTrue(np.isnan(np.asarray(labels.raw_net_pnl)[~valid]).all())
        self.assertTrue(np.isfinite(np.asarray(labels.raw_net_pnl)[valid]).all())
        self.assertTrue((labels.maturity_dates[valid] <=
                         np.datetime64(days[100].date(), "D")).all())
        self.assertEqual(labels.preflight["entry_policy"],
                         "next_exchange_session_daily_full_p20_atr_limit_v1")
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked,
                                    "Daily trade-path source identity mismatch"):
            materialize_daily_full_p20_pit_labels(**{
                **args, "source": replace(source, source_id="formal_full_roos_seed")
            })

    def test_teacher_missing_can_build_but_incompatible_artifact_blocks(self):
        with TemporaryDirectory() as tmp:
            state = inspect_history(project_root=tmp, seed=42)
            self.assertEqual(state["action"], "BUILD")
            path = Path(state["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            state = inspect_history(project_root=tmp, seed=42)
            self.assertEqual(state["action"], "BLOCKED")

    @staticmethod
    def _nested_history_fixture(root: Path, *, completed: bool = True):
        canonical_path = historical_params_path(root, 42)
        nested = canonical_path.parent.parent / "p2_dl_off_trained"
        nested.mkdir(parents=True, exist_ok=True)
        contract = {"test_scientific_identity": "original_p20_seed42"}
        identity = canonical_json_sha256(contract)
        import json
        (nested / "rolling_preflight.json").write_text(json.dumps({
            **contract, "runtime_identity_sha256": identity,
            "status": "DAILY_FULL_P20_LABEL_HISTORY_V1_PREFLIGHT_PASS",
        }), encoding="utf-8")
        (nested / "optimizer_runtime").mkdir()
        (nested / "optimizer_runtime" / "original_study.sqlite").write_bytes(b"study-bytes")
        if completed:
            (nested / "active_params").mkdir()
            (nested / "active_params" / canonical_path.name).write_bytes(b"signed-params-bytes")
            (nested / "rolling_optimizer_summary.json").write_text(
                '{"status":"COMPLETED"}', encoding="utf-8",
            )
        return canonical_path, nested, identity

    def test_completed_legacy_nested_history_reuses_without_optimizer_trials(self):
        from services.breakout_quality.daily_full_p20_training_source import TEACHER as owner
        with TemporaryDirectory() as tmp:
            canonical, nested, identity = self._nested_history_fixture(Path(tmp))
            with (patch.object(owner, "_expected_historical_preflight",
                               return_value={"runtime_identity_sha256": identity}),
                  patch.object(owner, "_validated_records", return_value=({}, ())),
                  patch.object(owner, "prepare_selection_historical_p2_params") as optimizer):
                plan = inspect_history(project_root=tmp, seed=42)
                self.assertEqual(plan["action"], "RESUME")
                self.assertFalse(canonical.is_file())  # Read-only planning.
                result = execute_history(project_root=tmp, seed=42)
                self.assertEqual(result, canonical)
                self.assertEqual(canonical.read_bytes(), b"signed-params-bytes")
                self.assertEqual((canonical.parent.parent / "optimizer_runtime" /
                                  "original_study.sqlite").read_bytes(), b"study-bytes")
                self.assertFalse(nested.exists())
                self.assertEqual(inspect_history(project_root=tmp, seed=42)["action"], "REUSE")
                optimizer.assert_not_called()

    def test_partial_nested_history_relocates_resumable_study_without_discarding_it(self):
        with TemporaryDirectory() as tmp:
            canonical, nested, identity = self._nested_history_fixture(Path(tmp), completed=False)
            source = inspect_nested_optimizer_workspace(
                canonical_arm_dir=canonical.parent.parent,
                expected_runtime_identity_sha256=identity,
            )
            self.assertEqual(source, nested)
            self.assertTrue(adopt_nested_optimizer_workspace(
                canonical_arm_dir=canonical.parent.parent,
                expected_runtime_identity_sha256=identity,
            ))
            self.assertTrue((canonical.parent.parent / "optimizer_runtime" /
                             "original_study.sqlite").is_file())
            self.assertFalse(canonical.is_file())
            self.assertFalse(nested.exists())

    def test_nested_history_mismatched_signature_and_conflicts_are_blocked(self):
        from services.breakout_quality.daily_full_p20_training_source import TEACHER as owner
        with TemporaryDirectory() as tmp:
            canonical, nested, identity = self._nested_history_fixture(Path(tmp))
            bad_identity = "f" * 64
            with patch.object(owner, "_expected_historical_preflight",
                              return_value={"runtime_identity_sha256": bad_identity}):
                self.assertEqual(inspect_history(project_root=tmp, seed=42)["action"], "BLOCKED")
                with self.assertRaises(FormalMinTrade40PITBlocked):
                    execute_history(project_root=tmp, seed=42)
            self.assertTrue((nested / "active_params" / canonical.name).is_file())
            (canonical.parent.parent / "rolling_optimizer_summary.json").write_bytes(
                b"unrelated-canonical-bytes")
            with patch.object(owner, "_expected_historical_preflight",
                              return_value={"runtime_identity_sha256": identity}):
                self.assertEqual(inspect_history(project_root=tmp, seed=42)["action"], "BLOCKED")
            self.assertTrue((nested / "active_params" / canonical.name).is_file())

    def test_new_history_uses_optimizer_workspace_root_not_arm_directory(self):
        from services.breakout_quality.daily_full_p20_training_source import TEACHER as owner
        with TemporaryDirectory() as tmp:
            expected = historical_params_path(tmp, 42)
            with (patch.object(owner, "_canonical_schedule_build_contract",
                               return_value={"trials_per_fold": 300,
                                             "train_window_months": 120,
                                             "oos_horizon_months": 12}),
                  patch.object(owner, "_validated_records", return_value=({}, ())),
                  patch.object(owner, "prepare_selection_historical_p2_params",
                               return_value={"params_path": expected}) as optimizer):
                self.assertEqual(execute_history(project_root=tmp, seed=42), expected)
                kwargs = optimizer.call_args.kwargs
                self.assertEqual(kwargs["output_relative_dir"], expected.parent.parent.parent)
                self.assertEqual(kwargs["parameter_set_identity"], owner.HISTORY_ID)
                self.assertEqual(kwargs["strategy_id"], "daily")
                self.assertEqual(kwargs["daily_search_profile"], owner.PROFILE)
                self.assertTrue(kwargs["recover_completed_strategy_compare"] is False)

    def test_nested_relocation_interrupted_between_renames_is_resumable(self):
        with TemporaryDirectory() as tmp:
            canonical, nested, identity = self._nested_history_fixture(Path(tmp))
            staged = canonical.parent.parent.parent / ".p2_dl_off_trained.nested_relocation"
            nested.rename(staged)
            self.assertEqual(inspect_nested_optimizer_workspace(
                canonical_arm_dir=canonical.parent.parent,
                expected_runtime_identity_sha256=identity,
            ), staged)
            self.assertTrue(adopt_nested_optimizer_workspace(
                canonical_arm_dir=canonical.parent.parent,
                expected_runtime_identity_sha256=identity,
            ))
            self.assertEqual(canonical.read_bytes(), b"signed-params-bytes")
            self.assertFalse(staged.exists())

    def test_unrelated_model_registration_keeps_existing_oracle_price_identity(self):
        from services.audit import oracle_score_artifacts as owner
        original = owner.compute_file_sha256
        before = owner.oracle_daily_source_fingerprint()
        def unrelated_model_registration(path):
            if Path(path).name in {"breakout_quality_registry.py", "breakout_quality_runtime.py"}:
                return "f" * 64
            return original(Path(path))
        with patch.object(owner, "compute_file_sha256", side_effect=unrelated_model_registration):
            self.assertEqual(owner.oracle_daily_source_fingerprint(), before)
        def changed_market_features(path):
            if Path(path).name == "features.py":
                return "e" * 64
            return original(Path(path))
        with patch.object(owner, "compute_file_sha256", side_effect=changed_market_features):
            self.assertNotEqual(owner.oracle_daily_source_fingerprint(), before)


if __name__ == "__main__":
    import unittest
    unittest.main()

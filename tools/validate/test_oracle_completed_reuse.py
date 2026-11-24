"""Signed historical Replay consumer regressions; no scientific producers."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from services.audit import target_oracle_romd as oracle
from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
from services.research import oracle_artifact_contract as contracts
from services.research.target_oracle_application import get_formal_oracle_definition


# Recorded in the original, signed local Oracle execution references. This is
# fixture provenance, not a production alias: the consumer below uses the real
# current implementation attestation and must independently prove equivalence.
_HISTORICAL_REPLAY_IMPLEMENTATION = (
    "ec364be05b82d31a326868aebb7119589ee4b814dcd292243e553986b7f9c472"
)
_REPLAY_IMPLEMENTATION_4539330D = (
    "a8ef2c4f42caafb521bbd2139d6e7f9bbeb74079ed58358e6bf6a95616f6af48"
)
_CANDIDATE_IMPLEMENTATION_4539330D = (
    "e81d2f676feab2e058573da3b75b45801e9a9b647f89ed6fce21936029ed7acf"
)


class OracleCompletedReuseTests(TestCase):
    def _fixture(self, root, *, replay_implementation=_HISTORICAL_REPLAY_IMPLEMENTATION,
                 scope="Breakout", selected_target="T40", candidate_implementation=None,
                 source_implementation=None):
        from core.file_integrity import compute_file_sha256
        from filters.breakout_quality.dataset_store import resolve_dataset_paths
        from filters.breakout_quality.paths import resolve_filter_output_dir
        definition = next(scoped for name, scoped in oracle._oracle_strategy_scopes(
            get_formal_oracle_definition("breakout_quality")) if name == scope)
        definition = replace(definition, source={**definition.source,
            "oracle_comparisons": tuple(row for row in definition.source["oracle_comparisons"]
                if row["target"] == selected_target)})
        settings, base, exact = oracle._contract(definition)
        exact = exact or settings.arms[definition.source["exact_arm_id"]]
        plan = oracle._oracle_comparison_plan(definition)
        direct = (oracle._direct_arm(definition, settings, base, exact)
                  if "direct" in plan["selectors"].values() else None)
        score_contracts = oracle._resolve_oracle_target_score_contracts(definition, settings, plan)
        policy = oracle._canonical_oracle_replay_policy(definition, base_arm=base, exact_arm=exact)
        baseline = definition.source["baseline_arm_id"]
        summary_path = resolve_dataset_paths(resolve_filter_output_dir(
            root, definition.source["filter_id"])).summary
        if plan["daily_sources"]:
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("{}", encoding="utf-8")
        seed, price, params = 693545351, "a" * 64, "b" * 64
        start, end = str(settings.start_date), str(definition.dimensions["oracle_evaluation_end_date"])
        store = OracleReplayStore(root, definition.output_subdir)
        index = contracts.OracleExecutionIndex(store)
        completed = {}
        for target in (baseline, selected_target):
            arm = exact if target == baseline else oracle._target_reference_arm(
                definition, settings, plan, target, exact, direct)
            source = None
            horizon = None
            if target != baseline:
                if target in plan["daily_sources"]:
                    recipe, provider, horizon = plan["daily_recipes"][plan["daily_sources"][target]]
                else:
                    recipe, provider = plan["recipes"][plan["aliases"].get(target, target)]
                source = contracts.source_calculation_contract(recipe=recipe,
                    profile_id=definition.source[provider],
                    horizon=horizon,
                    model_architecture=definition.source["model_architecture"])
                if source_implementation is not None:
                    source["implementation_sha256"] = source_implementation
            with patch.object(contracts, "replay_implementation_sha256",
                              return_value=replay_implementation):
                context = contracts.execution_context(
                    dataset=settings.dataset, price_sha256=price, param_sha256=params,
                    seed=seed, target=target, start=start, end=end, cutoff=str(settings.end_date),
                    max_positions=settings.max_positions,
                    evaluation_mode=str(definition.dimensions["evaluation_mode"]),
                    ranking_policy=arm.dl_runtime_mode, ranking_options=arm.dl_runtime_options,
                    replay_policy={**policy,
                        "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
                        "period_stats_implementation": oracle.PERIOD_STATS_IMPLEMENTATION,
                        **({"entry_policy": oracle.DAILY_ENTRY_CONTRACT}
                           if oracle._oracle_target_strategy(definition, target) == "Daily" else {})},
                    score_contract=score_contracts.get(target), source_contract=source,
                    dataset_inputs=contracts.dataset_input_contract(root, definition.source["filter_id"]),
                    execution_gate=None)
            if candidate_implementation is not None:
                context["replay_policy"]["daily_candidate_implementation_sha256"] = candidate_implementation
            score_args = {} if target == baseline else {
                "score_sha256": "d" * 64,
                "score_contract": {**score_contracts[target],
                    "canonical_feature_eligible_keys_sha256": "c" * 64,
                    "oracle_scored_rows": 2,
                    "source_evidence": "validated_generic_daily_feature_universe_and_research_cutoff_closeout"}}
            if target in plan["daily_sources"]:
                from services.audit.daily_trade40_oracle import NET_PNL_EVIDENCE, NET_PNL_RANK_POLICY
                target_id, label_id = oracle._daily_truth_identity(recipe, horizon)
                score_args["daily_evidence"] = {
                    "target_id": target_id, "label_id": label_id,
                    "profile_id": definition.source[provider],
                    "truth_source_sha256": oracle._daily_truth_fingerprint(recipe),
                    "dataset_summary_sha256": compute_file_sha256(summary_path),
                    "entry_policy": oracle._daily_truth_entry_policy(recipe),
                    "rank_policy": NET_PNL_RANK_POLICY, "rank_sha256": "d" * 64,
                    "raw_label_sha256": "e" * 64, "calendar_sha256": "f" * 64,
                    "teacher": {"seed": seed, "param_identity_sha256": params, "periods": []},
                    "mature_eligible_rows": 2, "censored_rank_last_rows": 0,
                    "settled_net_pnl_rows": 2, "score_start_date": start,
                    "research_cutoff": str(settings.end_date), **NET_PNL_EVIDENCE}
            physical_policy = oracle._arm_replay_policy(definition, plan=plan,
                target=target, replay_policy=policy, project_root=root,
                effective_context=context, **score_args)
            if candidate_implementation is not None:
                physical_policy["daily_candidate_implementation_sha256"] = candidate_implementation
            identity = replay_identity(dataset=settings.dataset, dataset_sha256=price,
                target=target, definition={target: target}, start=start, end=end,
                benchmark="end_to_end_v2", param_sha256=params, seed=seed,
                cohort={"cohort_rows": 2, "first_common_day": start, "last_common_day": end,
                        "research_cutoff": str(settings.end_date)},
                ranking_policy=arm.dl_runtime_mode, ranking_options=arm.dl_runtime_options,
                evaluation_mode=str(definition.dimensions["evaluation_mode"]),
                max_positions=settings.max_positions, replay_policy=physical_policy)
            observation = {"target": target, "seed": seed, "romd": 10.0,
                "total_return_pct": 100.0, "annualized_return_pct": 20.0,
                "mdd_pct": 10.0, "ev_a_r": 1.0, "trades": 10,
                "exposure_pct": 40.0, "win_rate_pct": 50.0,
                "yearly": [{"year": year, "strategy_return_pct": 10.0,
                            "is_full_year": year < int(end[:4])}
                           for year in range(int(start[:4]), int(end[:4]) + 1)],
                **({f"delta_romd_vs_{baseline.lower()}": 0.0} if target != baseline else {})}
            store.write(identity, observation, origin="historical_source_consumer_fixture")
            index.publish(context, identity, observation)
            completed[target] = (context, identity, observation)
        return definition, seed, price, params, store, index, completed

    def _environment(self, root, fixture):
        definition, seed, price, params, *_ = fixture
        stack = ExitStack()
        stack.enter_context(patch.object(oracle, "inspect_oracle_parameter_prerequisites", return_value={
            "robustness": SimpleNamespace(benchmark_id="end_to_end_v2"),
            "rows": [{"seed": seed, "path": root / "p16.json", "ready": True}],
            "producer_kwargs": {}}))
        stack.enter_context(patch.object(oracle, "_prepare_oracle_parameters", return_value=(
            [(seed, params, "synthetic", None, None)], {})))
        stack.enter_context(patch.object(oracle, "price_inventory_sha256", return_value=price))
        stack.enter_context(patch.object(oracle, "compute_strategy_param_scientific_sha256", return_value=params))
        for name in ("_read_only_score_keys", "load_profile_continuous_ranker_data",
                     "_load_or_build_oracle_price_scores", "_build_daily_truth", "_run_scenario",
                     "_execute_oracle_missing_parallel"):
            stack.enter_context(patch.object(oracle, name,
                side_effect=AssertionError(f"Read-only consumer invoked scientific producer: {name}")))
        return stack

    @staticmethod
    def _files(root):
        return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    @staticmethod
    def _current_context(recorded):
        """Resolve today's requested contract through the real calculation owners."""
        source = recorded["source_contract"]
        if source is not None:
            source = contracts.source_calculation_contract(
                recipe=source["recipe"], horizon=source["horizon_sessions"],
                profile_id=source["feature_provider"]["profile_id"],
                model_architecture=source["feature_provider"]["model_architecture"])
        policy = {key: value for key, value in recorded["replay_policy"].items()
                  if key != "daily_candidate_implementation_sha256"}
        return contracts.execution_context(
            dataset=recorded["dataset"], price_sha256=recorded["price_sha256"],
            param_sha256=recorded["param_sha256"], seed=recorded["seed"], target=recorded["target"],
            start=recorded["start_date"], end=recorded["end_date"], cutoff=recorded["research_cutoff"],
            max_positions=recorded["max_positions"], evaluation_mode=recorded["evaluation_mode"],
            ranking_policy=recorded["ranking_policy"], ranking_options=recorded["ranking_options"],
            replay_policy=policy, score_contract=recorded["score_contract"], source_contract=source,
            dataset_inputs=recorded["dataset_inputs"], execution_gate=recorded["execution_gate"])

    def test_historical_signed_observations_reuse_without_publication_or_replay(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._fixture(root)
            before = self._files(root)
            with self._environment(root, fixture):
                preview = oracle.preview_oracle_replay_plan(fixture[0], project_root=root)
                summary = oracle.collect_completed_oracle_results(fixture[0], project_root=root)
            self.assertEqual(preview["preflight"], {(fixture[1], target): "REUSE" for target in ("C96", "T40")})
            for target, (_context, identity, observation) in fixture[6].items():
                self.assertEqual(preview["completed_observations"][(fixture[1], target)], observation)
                self.assertEqual(preview["completed_identities"][(fixture[1], target)], identity)
            self.assertEqual(summary["metrics"]["T40"], {"oracle_romd": 10.0, "win_rate_pct": 50.0})
            self.assertEqual(before, self._files(root))

    def test_input_score_economic_and_unknown_code_changes_fail_closed(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._fixture(root)
            index, (context, identity, observation) = fixture[5], fixture[6]["T40"]
            self.assertEqual(index.read(context), (identity, observation))
            before = self._files(root)
            mutations = (
                (("price_sha256",), "1" * 64), (("param_sha256",), "2" * 64),
                (("seed",), 123), (("dataset",), "reduced"),
                (("ranking_policy",), "other_selector"), (("max_positions",), 1),
                (("score_contract", "score_start_date"), "2030-01-01"),
                (("replay_policy", "score_threshold"), 0.12345),
                (("replay_implementation_sha256",), "3" * 64),
                (("source_contract", "implementation_sha256"), "4" * 64),
                (("source_contract", "calculation_revision"), 999),
            )
            for keys, value in mutations:
                with self.subTest(field=".".join(keys)):
                    changed = deepcopy(context)
                    owner = changed
                    for key in keys[:-1]:
                        owner = owner[key]
                    owner[keys[-1]] = value
                    self.assertIsNone(index.read(changed))
                    self.assertIsNone(index.native_completed(changed))
            self.assertEqual(before, self._files(root))

    def test_missing_indexes_recover_signed_contexts_without_writing_aliases(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._fixture(root)
            for context, _identity, _observation in fixture[6].values():
                fixture[5]._path(context).unlink()
            before = self._files(root)
            with self._environment(root, fixture):
                preview = oracle.preview_oracle_replay_plan(fixture[0], project_root=root)
            self.assertEqual(set(preview["preflight"].values()), {"REUSE"})
            for target, (_context, identity, observation) in fixture[6].items():
                self.assertEqual(preview["completed_observations"][(fixture[1], target)], observation)
                self.assertEqual(preview["completed_identities"][(fixture[1], target)], identity)
            self.assertEqual(before, self._files(root))

    def test_replayed_4539330d_receipts_reuse_with_and_without_indexes(self):
        """A completed rerun under the interim full-source witness stays valid."""
        for indexed in (True, False):
            with self.subTest(indexed=indexed), TemporaryDirectory() as folder:
                root = Path(folder)
                fixture = self._fixture(root, replay_implementation=_REPLAY_IMPLEMENTATION_4539330D)
                if not indexed:
                    for context, _identity, _observation in fixture[6].values():
                        fixture[5]._path(context).unlink()
                before = self._files(root)
                with self._environment(root, fixture):
                    preview = oracle.preview_oracle_replay_plan(fixture[0], project_root=root)
                self.assertEqual(set(preview["preflight"].values()), {"REUSE"})
                for target, (_context, identity, observation) in fixture[6].items():
                    self.assertEqual(preview["completed_observations"][(fixture[1], target)], observation)
                    self.assertEqual(preview["completed_identities"][(fixture[1], target)], identity)
                self.assertEqual(before, self._files(root))

    def test_daily_4539330d_source_and_candidate_receipts_reuse_without_republication(self):
        # These are the complete 4539330d witnesses observed before this fix,
        # including the raw named-filter syntax and legal Min source encoding.
        variants = (
            ("Daily::Daily_full_p20", "TrueDailyUniversal_FullTrade_NetPnL_Direct",
             _CANDIDATE_IMPLEMENTATION_4539330D,
             "a5bfa8aa2b1498b55643754481f33673b1acd4fa798d448c81693039bcc8956b"),
            ("Daily::Daily_min_p21", "TrueDailyUniversal_MinTrade_P21_NetPnL_Direct", None,
             "98abdd17a1054c82e76567e49f2fda62dbda36e46bebff733b6907a380fb0c71"),
        )
        for scope, target, candidate, source in variants:
            for indexed in (True, False):
                with self.subTest(scope=scope, indexed=indexed), TemporaryDirectory() as folder:
                    root = Path(folder)
                    fixture = self._fixture(root, replay_implementation=_REPLAY_IMPLEMENTATION_4539330D,
                        scope=scope, selected_target=target, candidate_implementation=candidate,
                        source_implementation=source)
                    index = fixture[5]
                    if not indexed:
                        for context, _identity, _observation in fixture[6].values():
                            index._path(context).unlink()
                    before = self._files(root)
                    with self._environment(root, fixture):
                        for recorded, identity, observation in fixture[6].values():
                            current = self._current_context(recorded)
                            self.assertNotEqual(recorded, current)
                            actual = index.read(current) if indexed else index.native_completed(current)
                            self.assertEqual(actual, (identity, observation))
                    self.assertEqual(before, self._files(root))

    def test_changed_observation_consumer_rejects_4539330d_generic_source_receipts(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._fixture(root, replay_implementation=_REPLAY_IMPLEMENTATION_4539330D,
                scope="Daily::Daily_full_p20", selected_target="T40_Direct",
                candidate_implementation=_CANDIDATE_IMPLEMENTATION_4539330D)
            index = fixture[5]
            recorded, identity, observation = fixture[6]["T40_Direct"]
            self.assertEqual(index.read(self._current_context(recorded)), (identity, observation))
            before = self._files(root)
            consumer = Path(oracle.__file__).resolve()
            read_text = Path.read_text
            old = '"romd": float(summary["return_over_max_drawdown"]),'
            new = '"romd": float(summary["return_over_max_drawdown"]) + 1.0,'
            self.assertEqual(read_text(consumer, encoding="utf-8-sig").count(old), 1)

            def changed_source(path, *args, **kwargs):
                content = read_text(path, *args, **kwargs)
                return content.replace(old, new) if path.resolve() == consumer else content

            try:
                contracts.replay_implementation_sha256.cache_clear()
                contracts.portfolio_candidate_execution_sha256.cache_clear()
                with patch.object(Path, "read_text", changed_source):
                    current = self._current_context(recorded)
                    self.assertFalse(contracts.oracle_observation_equivalence_proven(
                        ranking_policy=recorded["ranking_policy"]))
                    retained = contracts.oracle_execution_contract(recorded)
                    self.assertEqual(retained["replay_implementation_sha256"],
                                     _REPLAY_IMPLEMENTATION_4539330D)
                    self.assertEqual(retained["replay_policy"]["daily_candidate_implementation_sha256"],
                                     _CANDIDATE_IMPLEMENTATION_4539330D)
                    self.assertIsNone(index.read(current))
                    self.assertIsNone(index.native_completed(current))
            finally:
                contracts.replay_implementation_sha256.cache_clear()
                contracts.portfolio_candidate_execution_sha256.cache_clear()
            self.assertEqual(before, self._files(root))

    def test_same_source_different_signed_score_cannot_reuse_old_observation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._fixture(root)
            store, index = fixture[4:6]
            context, identity, observation = fixture[6]["T40"]
            conflict = deepcopy(identity)
            conflict["replay_policy"]["perfect_score_sha256"] = "e" * 64
            store.write(conflict, observation, origin="contradictory_same_source_score_fixture")
            before = self._files(root)
            with self.assertRaisesRegex(ValueError, "conflict"):
                index.read(context)
            self.assertEqual(before, self._files(root))

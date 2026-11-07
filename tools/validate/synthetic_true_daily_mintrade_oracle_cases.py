"""Isolated, small-sample tests of new TrueDaily stock-day Min capability.

Run directly: python -m unittest tools.validate.synthetic_true_daily_mintrade_oracle_cases -q
No formal suite, optimizer, replay or model training is started here.
"""
from __future__ import annotations

from pathlib import Path
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
import copy
import json
import shutil
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.audit_policy import get_reusable_audit_definitions
from filters.breakout_quality.daily_trade40_target import compute_counterfactual_trade40_batch
from filters.breakout_quality.true_daily_mintrade_target import compute_true_daily_mintrade_batch
from services.audit import target_oracle_romd as oracle
from services.audit.daily_trade40_oracle import truth_producer_fingerprint as original_fingerprint
from services.audit.oracle_truth_references import load_or_build_daily_truth, source_context
from services.audit.true_daily_mintrade_oracle import (
    ENTRY_POLICY, TARGET_ID, build_true_daily_universal_oracle_scores, load_positive_truth,
    truth_producer_fingerprint as new_fingerprint,
)
from services.research.oracle_artifact_contract import source_implementation_sha256
from tools.validate.synthetic_daily_trade40_oracle_cases import PROFILE, source_fixture


class TrueDailyUniversalOracleCases(TestCase):
    def test_preflight_identifies_daily_only_in_strategy_column(self):
        from core.audit_policy import get_active_audit_module_id
        from core.audit_registry import oracle_comparison_key
        from services.research.target_oracle_application import get_formal_oracle_definition
        from core.display_common import _strip_ansi
        definition = get_formal_oracle_definition(get_active_audit_module_id())
        selected = [row for row in definition.source["oracle_comparisons"]
                    if row["target"] in {
                        "MFE40_Control", "TrueDailyUniversal_MinTrade_NetPnL_Positive",
                        "TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct",
                    }]
        keys = ["C96", *(oracle_comparison_key(row) for row in selected)]
        strategies = {"C96": "Breakout", **{
            oracle_comparison_key(row): row.get("strategy", "Breakout")
            for row in selected
        }}
        display_targets = {"C96": "C96", **{
            oracle_comparison_key(row): row["target"] for row in selected
        }}
        decisions = {(seed, key): ("RUN" if strategies[key] == "Daily" else "REUSE")
                     for seed in (11, 22) for key in keys}
        output = _strip_ansi(oracle.render_replay_preflight(
            label="Fixed", seeds=[11, 22], targets=keys, decisions=decisions,
            strategies=strategies, display_targets=display_targets,
        ))
        self.assertIn("Target", output)
        self.assertIn("Strategy", output)
        self.assertIn("Portfolio REUSE 8 / RUN（需回放）4 / BLOCKED 0", output)
        self.assertIn("Target 真值及 Score 另依各自簽章 REUSE／RUN", output)
        for line in output.splitlines():
            if "TrueDailyUniversal_MinTrade_NetPnL_Positive" in line:
                if "Daily" in line.split():
                    self.assertIn("RUN", line)
                else:
                    self.assertIn("Breakout", line)
                    self.assertIn("REUSE", line)
            elif "MFE40_Control" in line:
                self.assertIn("Breakout", line)
                self.assertIn("REUSE", line)

    def test_comparisons_and_previous_source_identities(self):
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        plan = oracle._oracle_comparison_plan(definition)
        exact = "TrueDailyUniversal_MinTrade_NetPnL_Positive"
        direct = exact + "_Direct"
        self.assertEqual(len(plan["targets"]), len({row["target"]
            for row in definition.source["oracle_comparisons"]}))
        self.assertEqual(plan["daily_recipes"][exact],
                         ("daily_trade_path_true_net_pnl", "daily_trade40_profile", None))
        self.assertEqual(plan["daily_sources"][direct], exact)
        self.assertEqual((plan["selectors"][exact], plan["selectors"][direct]),
                         ("exact", "direct"))
        self.assertEqual(oracle._daily_truth_identity("daily_trade_path_true_net_pnl", None)[0], TARGET_ID)
        for target in (exact, direct):
            record = next(r for r in definition.source["oracle_comparisons"] if r["target"] == target)
            self.assertEqual(record["positive_output"], "netpnl_positive_probability_v1")
        # New producer must not revise any old Min outcome/effective-scoring calculation.
        # Physical provenance may change across unrelated upstream releases;
        # the numerical owner below is the stable calculation invariant.
        self.assertEqual(source_implementation_sha256(
            "daily_trade_path_net_pnl", PROFILE),
            "3693bbbf2d8822688febf60f39b00ddc93a5fed06626916f867788813c4947c6")
        self.assertNotEqual(new_fingerprint(), original_fingerprint("net_pnl"))

    def test_trade_without_breakout_is_supervised_not_zero_filled(self):
        calendar, frames, _, schedule = source_fixture()
        frame = frames["2317"]  # This stock never emits a formal Min setup.
        positions = frame.index.get_indexer(pd.DatetimeIndex(["2021-01-04", "2021-01-05"]))
        old = compute_counterfactual_trade40_batch(
            frame, positions, ticker="2317", schedule=schedule,
            exchange_calendar=calendar, horizon_sessions=None,
            research_cutoff="2021-05-15", include_net_pnl=True,
        )
        new = compute_true_daily_mintrade_batch(
            frame, positions, ticker="2317", schedule=schedule,
            exchange_calendar=calendar, research_cutoff="2021-05-15",
        )
        self.assertEqual(old.valid_mask.tolist(), [False, False])
        self.assertTrue(all(value == "no_formal_min_signal_or_continuation" for value in old.reasons))
        self.assertEqual(new.valid_mask.tolist(), [True, True])
        self.assertTrue((new.target_raw_pnl < 0).all())

    def test_next_exchange_session_unfilled_cannot_become_later_continuation_fill(self):
        # 1/4 info-close 90.10; 1/5 never touches limit; 1/6 does.
        # Old Min continuation wrongly produced a completed 1/6 trade.
        calendar, frames, _, schedule = source_fixture()
        frame = frames["2317"].copy()
        frame.loc["2021-01-05", ["Open", "High", "Low", "Close"]] = [90.9, 91.5, 90.3, 90.7]
        frame.loc["2021-01-06", ["Open", "High", "Low", "Close"]] = [89.7, 91.8, 88.7, 89.8]
        info_day = pd.Timestamp("2021-01-04")
        self.assertEqual(float(frame.loc[info_day, "Close"]), 90.10)
        self.assertGreater(float(frame.loc["2021-01-05", "Low"]), 90.10)
        self.assertLess(float(frame.loc["2021-01-06", "Low"]), 90.10)
        observed = compute_true_daily_mintrade_batch(
            frame, frame.index.get_indexer(pd.DatetimeIndex([info_day])),
            ticker="2317", schedule=schedule,
            exchange_calendar=calendar, research_cutoff="2021-05-15",
        )
        self.assertEqual(observed.reasons, ("unfilled_at_horizon",))
        self.assertEqual(observed.valid_mask.tolist(), [False])
        self.assertTrue(np.isnan(observed.target_raw_pnl).all())
        self.assertTrue(np.isnat(observed.maturity_dates).all())

    def test_unavailable_next_market_session_stays_null(self):
        calendar, frames, _, schedule = source_fixture()
        frame = frames["2317"].drop(pd.Timestamp("2021-01-05"))
        positions = frame.index.get_indexer(pd.DatetimeIndex(["2021-01-04"]))
        outcome = compute_true_daily_mintrade_batch(
            frame, positions, ticker="2317", schedule=schedule,
            exchange_calendar=calendar, research_cutoff="2021-05-15",
        )
        self.assertEqual(outcome.reasons, ("next_exchange_session_quote_unavailable",))
        self.assertFalse(bool(outcome.valid_mask[0]))
        self.assertTrue(np.isnan(outcome.target_raw_pnl[0]))

    def test_missing_effective_p16_not_replaced_with_future_param(self):
        from dataclasses import replace
        calendar, frames, _, schedule = source_fixture()
        future_schedule = (replace(schedule[0], start=pd.Timestamp("2021-02-01")),)
        frame = frames["2317"]
        outcome = compute_true_daily_mintrade_batch(
            frame, frame.index.get_indexer(pd.DatetimeIndex(["2021-01-04"])),
            ticker="2317", schedule=future_schedule,
            exchange_calendar=calendar, research_cutoff="2021-05-15",
        )
        self.assertEqual(outcome.reasons, ("no_historical_effective_Min_param",))
        self.assertTrue(np.isnan(outcome.target_raw_pnl[0]))

    def test_signed_rank_positive_witness_and_effective_reference_reuse(self):
        calendar, frames, bundle, schedule = source_fixture()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            kwargs = dict(
                project_root=root, dataset="full", filter_id="breakout_quality_v1",
                profile_id=PROFILE, cutoff="2021-05-15", evaluation_start="2021-01-01",
                evaluation_end="2021-01-05", score_start="2021-01-01",
                seed=693545351, param_sha256="a" * 64, canonical_schedule=schedule,
                price_sha256="b" * 64, cache_root=root / "truth_cache",
                output_subdir="audit", ticker_workers=1,
            )
            with patch("services.audit.true_daily_mintrade_oracle.oracle_dataset_summary_sha256",
                       return_value="1" * 64), \
                 patch("services.audit.true_daily_mintrade_oracle.load_profile_continuous_ranker_data",
                       return_value=bundle), \
                 patch("services.audit.true_daily_mintrade_oracle.discover_dataset_csv_inputs",
                       return_value=([("0050", Path("0050.csv")),
                                      ("2330", Path("2330.csv")),
                                      ("2317", Path("2317.csv"))], [])), \
                 patch("services.audit.true_daily_mintrade_oracle.collect_canonical_exchange_sessions",
                       return_value=calendar), \
                 patch("services.audit.true_daily_mintrade_oracle.load_dataset_frame",
                       side_effect=lambda path, ticker, **options: frames[ticker]):
                scores, evidence = build_true_daily_universal_oracle_scores(**kwargs)
                self.assertEqual(len(scores), 4)
                # Same [0,1] within-day target percentile consumed by the
                # established MinTrade Oracle; raw broker PnL is NEVER used
                # as the injected Selector score.
                self.assertTrue(all(0.0 <= score <= 1.0 for score in scores.values()))
                self.assertEqual(evidence["settled_net_pnl_rows"], 4)
                self.assertEqual(evidence["entry_policy"], ENTRY_POLICY)
                raw = load_positive_truth(
                    project_root=root, output_subdir="audit", seed=693545351,
                    price_sha256="b" * 64, param_sha256="a" * 64,
                    canonical_scores=scores, canonical_evidence=evidence,
                )
                self.assertEqual(len(raw), 4)
                self.assertEqual(sum(value > 0.0 for value in raw.values()), 2)
                with patch("services.audit.true_daily_mintrade_oracle._run_stocks",
                           side_effect=AssertionError("Truth cache MUST REUSE")):
                    score2, proof2 = build_true_daily_universal_oracle_scores(**kwargs)
                self.assertEqual(scores, score2)
                self.assertEqual(proof2["source_reuse_status"], "REUSE")

                execution = {
                    "source_contract": {"recipe": "daily_trade_path_true_net_pnl",
                        "feature_provider": {"profile_id": PROFILE}, "horizon_sessions": None},
                    "dataset": "full", "price_sha256": "b" * 64,
                    "param_sha256": "a" * 64, "seed": 693545351,
                    "start_date": "2021-01-01", "end_date": "2021-01-05",
                    "research_cutoff": "2021-05-15", "dataset_inputs": {},
                    "replay_calculation_revision": 1,
                    "replay_implementation_sha256": "c" * 64,
                    "replay_policy": {
                        "filter_id": "breakout_quality_v1",
                        "fixed_risk": {}, "max_position_cap_pct": 1.0,
                        "broker_accounting": {}, "execution_defaults": {},
                    },
                    "score_contract": {"score_start_date": "2021-01-01"},
                }
                self.assertEqual(source_context(execution)["source_contract"]["recipe"],
                                 "daily_trade_path_true_net_pnl")
                score3, attested = load_or_build_daily_truth(
                    execution=execution, cache_root=root / "truth_cache",
                    producer=lambda: (scores, evidence),
                )
                self.assertEqual(scores, score3)
                self.assertIn("effective_source_reference", attested)
                score4, attested2 = load_or_build_daily_truth(
                    execution=execution, cache_root=root / "truth_cache",
                    producer=lambda: self.fail("Reference must reuse producer"),
                )
                self.assertEqual(scores, score4)
                self.assertEqual(attested["effective_source_reference"],
                                 attested2["effective_source_reference"])
                # Model/runtime-only edits change file provenance, not Min
                # arithmetic. Preserve signed original NPZ + raw witness.
                with patch("services.audit.true_daily_mintrade_oracle.truth_producer_fingerprint",
                           return_value="9" * 64):
                    preserved = load_positive_truth(
                        project_root=root, output_subdir="audit", seed=693545351,
                        price_sha256="b" * 64, param_sha256="a" * 64,
                        canonical_scores=score4, canonical_evidence=attested2,
                        execution=execution, cache_root=root / "truth_cache")
                    self.assertEqual(preserved, raw)
                    with self.assertRaises(ValueError):
                        load_positive_truth(
                            project_root=root, output_subdir="audit", seed=693545351,
                            price_sha256="b" * 64, param_sha256="a" * 64,
                            canonical_scores=score4,
                            canonical_evidence={**attested2, "effective_source_reference": "0" * 64},
                            execution=execution, cache_root=root / "truth_cache")



class OraclePerfectOutputReuseCases(TestCase):
    """Actual immutable old-format receipts; no FIT, truth build or Replay."""

    def _fixture(self, root, *, seed_count=8):
        from core.audit_policy import get_active_audit_module_id
        from core.file_integrity import canonical_json_sha256, compute_file_sha256
        from filters.breakout_quality.dataset_store import resolve_dataset_paths
        from filters.breakout_quality.paths import resolve_filter_output_dir
        from services.research.target_oracle_application import get_formal_oracle_definition
        from services.research.oracle_artifact_contract import (
            OracleExecutionIndex, execution_context, dataset_input_contract, source_calculation_contract,
        )
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        from services.audit.daily_trade40_oracle import NET_PNL_EVIDENCE, NET_PNL_RANK_POLICY

        definition = get_formal_oracle_definition(get_active_audit_module_id())
        settings, base, exact = oracle._contract(definition)
        seeds = (693545351, 2014432738, 104504529, 762252050,
                 9734312, 1020806211, 116293459, 1234683984)[:seed_count]
        summary = resolve_dataset_paths(resolve_filter_output_dir(
            root, definition.source["filter_id"])).summary
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({"source_data_date_range": {"end": settings.end_date}}))
        price, params = "a" * 64, "b" * 64
        schedule = source_fixture()[3]
        periods = [{"start": str(p.start.date()), "end": None if p.end is None else str(p.end.date()),
                    "params_signature": p.params_signature} for p in schedule]
        store = OracleReplayStore(root, definition.output_subdir)
        index = OracleExecutionIndex(store)
        contexts = {}
        for strategy, scoped in oracle._oracle_strategy_scopes(definition):
            # Historical fixture certifies only Min/P16. Newly registered
            # Full/P17 must remain RUN until its own receipts are produced.
            if scoped.source["baseline_arm_id"] != "C96":
                continue
            # Recover current certified receipts without scientific recomputation.
            # Retired dual-source evidence is exercised separately below.
            plan = oracle._oracle_comparison_plan(scoped)
            contracts = oracle._resolve_oracle_target_score_contracts(scoped, settings, plan)
            direct = oracle._direct_arm(scoped, settings, base, exact)
            common = oracle._canonical_oracle_replay_policy(scoped, base_arm=base, exact_arm=exact)
            for seed in seeds:
                for target in ("C96", *(name for name, _ in plan["targets"])):
                    arm = exact if target == "C96" else oracle._target_reference_arm(
                        scoped, settings, plan, target, exact, direct)
                    source = None
                    if target in plan["daily_sources"]:
                        recipe, provider, horizon = plan["daily_recipes"][plan["daily_sources"][target]]
                        source = source_calculation_contract(recipe=recipe,
                            profile_id=scoped.source[provider], horizon=horizon,
                            model_architecture=scoped.source["model_architecture"])
                    elif target != "C96":
                        recipe, provider = plan["recipes"][plan["aliases"].get(target, target)]
                        source = source_calculation_contract(recipe=recipe,
                            profile_id=scoped.source[provider],
                            model_architecture=scoped.source["model_architecture"])
                    daily = target != "C96" and strategy == "Daily"
                    context = execution_context(dataset=settings.dataset, price_sha256=price,
                        param_sha256=params, seed=seed, target=target, start=settings.start_date,
                        end=settings.end_date, cutoff=settings.end_date, max_positions=settings.max_positions,
                        evaluation_mode=str(scoped.dimensions["evaluation_mode"]),
                        ranking_policy=arm.dl_runtime_mode, ranking_options=arm.dl_runtime_options,
                        replay_policy={**common,
                            "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
                            "period_stats_implementation": oracle.PERIOD_STATS_IMPLEMENTATION,
                            **({"entry_policy": oracle.DAILY_ENTRY_CONTRACT} if daily else {})},
                        score_contract=contracts.get(target), source_contract=source,
                        dataset_inputs=dataset_input_contract(root, scoped.source["filter_id"]), execution_gate=None)
                    context["ranking_options"] = dict(arm.dl_runtime_options)
                    raw_contract = {key: value for key, value in contracts.get(target, {}).items()
                                    if key not in {"reference_arm_id", "reference_score_source_id", "reference_profile_id"}}
                    if target != "C96":
                        context["score_contract"] = raw_contract
                    args = {} if target == "C96" else {
                        "score_sha256": "d" * 64, "score_contract": {**raw_contract,
                            "canonical_feature_eligible_keys_sha256": "c" * 64, "oracle_scored_rows": 4,
                            "source_evidence": "validated_generic_daily_feature_universe_and_research_cutoff_closeout"}}
                    if target in plan["daily_sources"]:
                        target_id, label_id = oracle._daily_truth_identity(recipe, horizon)
                        args["daily_evidence"] = {
                            "target_id": target_id, "label_id": label_id, "profile_id": scoped.source[provider],
                            "truth_source_sha256": oracle._daily_truth_fingerprint(recipe),
                            "dataset_summary_sha256": compute_file_sha256(summary),
                            "rank_policy": NET_PNL_RANK_POLICY if "pnl" in recipe else
                                "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
                            "rank_sha256": "d" * 64, "raw_label_sha256": "e" * 64, "calendar_sha256": "f" * 64,
                            "teacher": {"seed": seed, "param_identity_sha256": params, "periods": periods},
                            "mature_eligible_rows": 4, "censored_rank_last_rows": 0,
                            "settled_net_pnl_rows": 4, "settled_net_r_rows": 4,
                            "score_start_date": settings.start_date, "research_cutoff": settings.end_date,
                            **NET_PNL_EVIDENCE,
                            **({"entry_policy": oracle._daily_truth_entry_policy(recipe)}
                               if recipe == "daily_trade_path_true_net_pnl" else {})}
                    policy = oracle._arm_replay_policy(scoped, plan=plan, target=target,
                        replay_policy=common, project_root=root, effective_context=context, **args)
                    identity = replay_identity(dataset=settings.dataset, dataset_sha256=price, target=target,
                        definition={target: target}, start=settings.start_date, end=settings.end_date,
                        benchmark="end_to_end_v2", param_sha256=params, seed=seed,
                        cohort={"cohort_rows": 4, "first_common_day": "2021-01-04",
                                "last_common_day": "2021-01-05", "research_cutoff": settings.end_date},
                        ranking_policy=arm.dl_runtime_mode, ranking_options=arm.dl_runtime_options,
                        evaluation_mode=str(scoped.dimensions["evaluation_mode"]),
                        max_positions=settings.max_positions, replay_policy=policy)
                    identity["ranking_options"] = dict(arm.dl_runtime_options)
                    if target != "C96":
                        identity["replay_policy"]["canonical_dl_score_contract"] = args["score_contract"]
                    if "formal_positive_score_output" in policy:
                        identity["replay_policy"]["formal_positive_score_output"] = policy["formal_positive_score_output"]
                    if "direct_arm" in policy:
                        identity["replay_policy"]["direct_arm"] = policy["direct_arm"]
                    romd = 8. if target == "C96" else (30. if daily else 10.) + (2 if target.endswith("_Direct") else 1)
                    observation = {"target": target, "seed": seed, "romd": romd,
                        "win_rate_pct": 90. if daily else 60., "total_return_pct": 100.,
                        "annualized_return_pct": 10., "mdd_pct": 12., "ev_a_r": .5,
                        "trades": 10, "exposure_pct": 70.,
                        "yearly": [{"year": y, "strategy_return_pct": 10., "is_full_year": y < 2026}
                                   for y in range(2021, 2027)],
                        **({"delta_romd_vs_c96": romd - 8.} if target != "C96" else {})}
                    store.write(identity, observation, origin="synthetic_old_reference_replay")
                    index.publish(context, identity, observation)
                    contexts[(seed, oracle._oracle_scoped_key(target, strategy))] = context
        return definition, seeds, price, params, schedule, store, contexts

    def _environment(self, root, fixture):
        definition, seeds, price, params, schedule, _, _ = fixture
        stack = ExitStack()
        stack.enter_context(patch.object(oracle, "inspect_oracle_parameter_prerequisites", return_value={
            "robustness": SimpleNamespace(benchmark_id="end_to_end_v2"),
            "rows": [{"seed": seed, "path": root / f"p16_{seed}.json", "ready": True}
                     for seed in seeds], "producer_kwargs": {}}))
        stack.enter_context(patch.object(oracle, "collect_status", return_value={"status": "READY"}))
        stack.enter_context(patch.object(oracle, "_robustness_artifacts", return_value=(
            SimpleNamespace(benchmark_id="end_to_end_v2"), [(seed, root / f"p16_{seed}.json") for seed in seeds])))
        stack.enter_context(patch.object(oracle, "_prepare_oracle_parameters", return_value=(
            [(seed, params, "synthetic", None, None) for seed in seeds], {})))
        stack.enter_context(patch.object(oracle, "price_inventory_sha256", return_value=price))
        stack.enter_context(patch("filters.breakout_quality.daily_trade40_target.resolve_formal_min_param_records", return_value=[]))
        stack.enter_context(patch("filters.breakout_quality.daily_trade40_target.build_formal_min_execution_schedule", return_value=(schedule, {})))
        for name in ("_read_only_score_keys", "load_profile_continuous_ranker_data",
                     "_load_or_build_oracle_price_scores", "_build_daily_truth", "_run_scenario",
                     "_execute_oracle_missing_parallel"):
            stack.enter_context(patch.object(oracle, name, side_effect=AssertionError(f"No scientific RUN: {name}")))
        return stack

    def test_all_eight_seed_rows_and_overview_recover_without_publication_or_run(self):
        from services.research.target_dl_overview import _oracle_rows
        with TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with self._environment(root, fixture):
                preview = oracle.preview_oracle_replay_plan(fixture[0], project_root=root)
                comparison_count = len(preview["targets"])
                self.assertEqual(len(preview["preflight"]), len(fixture[1]) * comparison_count)
                self.assertEqual({preview["preflight"][(seed, key)]
                                  for seed in fixture[1] for key in preview["targets"][:25]}, {"REUSE"})
                self.assertEqual({preview["preflight"][(seed, key)]
                                  for seed in fixture[1] for key in preview["targets"][25:]}, {"RUN"})
                _, _, _, values, info = _oracle_rows(root, fixture[0], {})
            target = "TrueDailyUniversal_MinTrade_NetPnL_Positive"
            self.assertEqual(len(values), comparison_count - 3)  # Full has no signed receipts.
            self.assertTrue({"C97", "T40::Full", "T40_Direct::Full"}.isdisjoint(values))
            self.assertEqual(values[target]["oracle_romd"], 11.)
            self.assertEqual(values[target + "::Daily"]["oracle_romd"], 31.)
            self.assertEqual(values[target + "_Direct::Daily"]["oracle_romd"], 32.)
            self.assertEqual(len(info["seeds"]), 8)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_lost_index_recovers_from_exact_old_signed_context_hash(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._fixture(root)
            shutil.rmtree(fixture[5].cache_root.parent / "execution_index")
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with self._environment(root, fixture):
                preview = oracle.preview_oracle_replay_plan(fixture[0], project_root=root)
            self.assertEqual({action for (_seed, key), action in preview["preflight"].items()
                              if key not in {"C97", "T40::Full", "T40_Direct::Full"}}, {"REUSE"})
            self.assertEqual({action for (_seed, key), action in preview["preflight"].items()
                              if key in {"C97", "T40::Full", "T40_Direct::Full"}}, {"RUN"})
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_legacy_oracle_positive_receipts_reuse_two_model_oracle_without_replay(self):
        """Signed old output schema and signed new output schema mean one Oracle.

        This exercises the actual immutable-origin index and its no-index
        recovery. The old replay is not rewritten, and no target or portfolio
        producer is called merely to display the REUSE plan.
        """
        from services.research.oracle_artifact_contract import (
            OracleExecutionIndex, oracle_execution_contract,
        )
        from core.file_integrity import canonical_json_sha256
        legacy_sha = "56fdce12e331011e7423c7fa0f1c8b27563b43bbd731fcff938c6a0c602edd05"
        with TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._fixture(root, seed_count=1)
            seed = fixture[1][0]
            store = fixture[5]
            index = OracleExecutionIndex(store)
            contexts = fixture[6]
            positives = [key for key in contexts if "Positive" in key[1]]
            self.assertEqual(len(positives), 6)
            for key in positives:
                current = contexts[key]
                native_identity, result = index.read(current)
                old_context = copy.deepcopy(current)
                old_options = old_context["ranking_options"]
                self.assertEqual(old_options.pop('positive_dl_id'),
                                 'CONT13CP_ROLL' if 'DailyUniversal_' in key[1] and 'TrueDaily' not in key[1] else 'CONT13CT_ROLL')
                old_options['positive_prediction_contract'] = 'single_ranker_calibrated_positive_v2' 
                old_context["score_contract"]["positive_score_contract_revision"] = (
                    "positive_netpnl_dual_output_v1")
                self.assertEqual(oracle_execution_contract(old_context),
                                 oracle_execution_contract(current))
                legacy_identity = copy.deepcopy(native_identity)
                legacy_identity["ranking_options"] = copy.deepcopy(old_options)
                policy = legacy_identity["replay_policy"]
                policy["canonical_dl_score_contract"]["positive_score_contract_revision"] = (
                    "positive_netpnl_dual_output_v1")
                policy["effective_execution_contract_sha256"] = canonical_json_sha256(old_context)
                policy["formal_positive_score_output"]["shared_admission_code_sha256"] = legacy_sha
                store._path(native_identity).unlink()
                index._path(current).unlink()
                store.write(legacy_identity, result, origin="verified_legacy_dual_predictor_fixture")
                index.publish(old_context, legacy_identity, result)
                restored = index.read(current)
                self.assertEqual(restored[1], result)
                self.assertEqual(restored[0], legacy_identity)
                for field, value in (("price_sha256", "c" * 64),
                                     ("param_sha256", "e" * 64)):
                    incompatible = dict(current, **{field: value})
                    self.assertIsNone(index.read(incompatible))
                    self.assertIsNone(index.native_completed(incompatible))
                for field, value in (("positive_probability_threshold", .6),
                                     ("positive_score_column", "wrong_column"),
                                     ("positive_dl_id", "")):
                    incompatible = copy.deepcopy(current)
                    incompatible["ranking_options"][field] = value
                    self.assertIsNone(index.read(incompatible))
                    self.assertIsNone(index.native_completed(incompatible))
                incompatible = copy.deepcopy(current)
                incompatible["score_contract"]["positive_score_contract_revision"] = "unknown_v3"
                self.assertIsNone(index.read(incompatible))
                self.assertIsNone(index.native_completed(incompatible))
                invalid_producer = copy.deepcopy(legacy_identity)
                invalid_producer["replay_policy"]["formal_positive_score_output"][
                    "shared_admission_code_sha256"] = "9" * 64
                with self.assertRaisesRegex(ValueError, "Positive admission producer not certified"):
                    index._validate_binding(current, invalid_producer, result)
                # A future executable gate edit cannot silently inherit this
                # old signed 0/1 admission semantics with unchanged price/P16.
                with patch("services.research.oracle_artifact_contract."
                           "_positive_admission_source_certificate",
                           return_value=("f" * 64, False)):
                    with self.assertRaisesRegex(ValueError, "Positive admission producer not certified"):
                        index._validate_binding(current, legacy_identity, result)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with self._environment(root, fixture):
                preview = oracle.preview_oracle_replay_plan(fixture[0], project_root=root)
            self.assertEqual({status for (_seed, key), status in preview["preflight"].items()
                              if key not in {"C97", "T40::Full", "T40_Direct::Full"}}, {"REUSE"})
            self.assertEqual({status for (_seed, key), status in preview["preflight"].items()
                              if key in {"C97", "T40::Full", "T40_Direct::Full"}}, {"RUN"})
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
            shutil.rmtree(store.cache_root.parent / "execution_index")
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with self._environment(root, fixture):
                preview = oracle.preview_oracle_replay_plan(fixture[0], project_root=root)
            self.assertEqual({status for (_seed, key), status in preview["preflight"].items()
                              if key not in {"C97", "T40::Full", "T40_Direct::Full"}}, {"REUSE"})
            self.assertEqual({status for (_seed, key), status in preview["preflight"].items()
                              if key in {"C97", "T40::Full", "T40_Direct::Full"}}, {"RUN"})
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_one_missing_daily_seed_hides_only_its_aggregate_without_replay(self):
        from services.research.oracle_artifact_contract import OracleExecutionIndex
        from services.research.target_dl_overview import _oracle_rows
        with TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self._fixture(root)
            key = "TrueDailyUniversal_MinTrade_NetPnL_Positive::Daily"
            context = fixture[6][(fixture[1][-1], key)]
            index = OracleExecutionIndex(fixture[5])
            identity, _ = index.read(context)
            fixture[5]._path(identity).unlink()
            index._path(context).unlink()
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with self._environment(root, fixture):
                preview = oracle.preview_oracle_replay_plan(fixture[0], project_root=root)
                _, _, _, values, _ = _oracle_rows(root, fixture[0], {})
            self.assertEqual(list(preview["preflight"].values()).count("RUN"),
                             3 * len(fixture[1]) + 1)
            self.assertEqual(preview["preflight"][(fixture[1][-1], key)], "RUN")
            self.assertEqual(len(values), len(preview["targets"]) - 4)  # Missing Min seed + three Full RUN rows.
            self.assertNotIn(key, values)
            self.assertIn(key.replace("::Daily", ""), values)
            self.assertIn(key.replace("::Daily", "_Direct::Daily"), values)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_true_execution_changes_never_inherit_reference_only_reuse(self):
        from services.research.oracle_artifact_contract import OracleExecutionIndex, oracle_execution_contract
        with TemporaryDirectory() as folder:
            fixture = self._fixture(Path(folder), seed_count=1)
            key = (fixture[1][0], "TrueDailyUniversal_MinTrade_NetPnL_Positive::Daily")
            context = oracle_execution_contract(fixture[6][key])
            index = OracleExecutionIndex(fixture[5])
            self.assertIsNotNone(index.read(context))
            changes = (("price_sha256", "f" * 64), ("param_sha256", "e" * 64),
                       ("seed", 123), ("end_date", "2025-12-31"), ("max_positions", 9),
                       ("ranking_policy", "other_selector"))
            for field, value in changes:
                with self.subTest(field=field):
                    changed = dict(context, **{field: value})
                    self.assertIsNone(index.read(changed))
                    self.assertIsNone(index.native_completed(changed))
            for field, key, value in (
                ("ranking_options", "positive_probability_threshold", .6),
                ("ranking_options", "positive_dl_id", None),
                ("score_contract", "score_start_date", "2021-01-04"),
                ("replay_policy", "entry_policy", "breakout"),
                ("replay_policy", "oracle_parity_implementation", "changed_implementation"),
                ("source_contract", "calculation_revision",
                 int(context["source_contract"]["calculation_revision"]) + 1),
            ):
                with self.subTest(field=field, key=key):
                    changed = copy.deepcopy(context)
                    changed[field][key] = value
                    self.assertIsNone(index.read(changed))
                    self.assertIsNone(index.native_completed(changed))

    def test_corrupt_reference_and_conflicting_signed_observations_fail_closed(self):
        from core.file_integrity import canonical_json_sha256
        from services.research.oracle_artifact_contract import OracleExecutionIndex
        for corruption in ("reference", "receipt", "conflict"):
            with self.subTest(corruption=corruption), TemporaryDirectory() as folder:
                fixture = self._fixture(Path(folder), seed_count=1)
                index = OracleExecutionIndex(fixture[5])
                context = fixture[6][(fixture[1][0], "TrueDailyUniversal_MinTrade_NetPnL_Positive::Daily")]
                identity, observation = index.read(context)
                if corruption == "reference":
                    index._path(context).write_text('{"bad":1}')
                elif corruption == "receipt":
                    fixture[5]._path(identity).write_text('{"bad":1}')
                else:
                    other = copy.deepcopy(identity)
                    other["target_definition"] = "old presentation description"
                    fixture[5].write(other, {**observation, "romd": observation["romd"] + 1}, origin="synthetic_conflict")
                with self.assertRaises(ValueError):
                    index.read(context)

    def test_reference_rebinding_preserves_source_and_verifies_legacy_consumer_evidence(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
        from core.file_integrity import canonical_json_sha256
        from services.research.oracle_artifact_contract import (
            _reference_arm_payload, reference_arm_execution_matches, reference_arm_execution_binding,
        )
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        before = original_fingerprint("net_pnl"), new_fingerprint()
        from dataclasses import replace
        from copy import deepcopy
        for current in ("C133", "C134"):
            alias = "TEST_ALIAS_" + current
            local = replace(settings, arms={**settings.arms,
                alias: replace(settings.arms[current], arm_id=alias)})
            comparison = {"strategy_arm_id": current}
            receipt = {"arm_id": alias, "execution_sha256": canonical_json_sha256(
                _reference_arm_payload(alias, local, effective=False))}
            self.assertTrue(reference_arm_execution_matches(receipt, comparison, local))
            receipt["execution_sha256"] = "f" * 64
            self.assertFalse(reference_arm_execution_matches(receipt, comparison, local))
            published = reference_arm_execution_binding({"strategy_arm_id": alias}, local)
            self.assertTrue(reference_arm_execution_matches(published, comparison, local))
            for change in ("threshold", "old_version", "dual_source"):
                old = deepcopy(published)
                arm = old["execution_contract"]["arm"]
                if change == "threshold":
                    arm["dl_runtime_options"]["positive_probability_threshold"] = .6
                elif change == "old_version":
                    arm["replay_contract_version"] = 1
                else:
                    arm["dl_runtime_options"]["positive_dl_id"] = "FAKE_SECONDARY"
                old["execution_sha256"] = canonical_json_sha256(old["execution_contract"])
                self.assertFalse(reference_arm_execution_matches(old, comparison, local))
            self.assertFalse(reference_arm_execution_matches(
                {"arm_id": "C131", "execution_sha256": "f" * 64}, comparison, local))
        original_fingerprint.cache_clear()
        new_fingerprint.cache_clear()
        self.assertEqual(before, (original_fingerprint("net_pnl"), new_fingerprint()))


if __name__ == "__main__":
    main()

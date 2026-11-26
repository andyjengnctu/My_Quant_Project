"""Independent decision-local randomization and fixed-repeat publication fixtures."""
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from core.candidate_score_permutation import (
    POLICY, REPEATS, permute_candidate_scores, permutation_replay_scope, permutation_spec,
)
from services.research.strategy_score_permutation import (
    KIND, ROMD, MDD, _mean_records, distribution, evaluate_permutation_control,
    load_bundle, run_fixed_permutations, validate_bundle,
)


def candidate(ticker, score, available=True):
    return dict(ticker=ticker, breakout_quality_score=score,
                breakout_quality_rank=dict(available=available, score=score),
                signal_date="2021-01-04", price=10, pending={"age": 2})


class CandidateScorePermutationTests(TestCase):
    def test_uniform_multiset_missing_and_input_immutability(self):
        rows = [candidate(str(i), value) for i, value in enumerate((-9., 0., 3., 3., 105.))]
        rows += [candidate("missing", None), candidate("invalid", 8, False), candidate("infinite", float("inf"))]
        before = deepcopy(rows)
        projected, evidence = permute_candidate_scores(rows, seed=42, repeat_id=1, decision_date="2021-01-05")
        self.assertEqual(sorted(r["breakout_quality_score"] for r in projected[:5]), [-9., 0., 3., 3., 105.])
        self.assertEqual(rows, before)
        self.assertEqual(projected[5:], rows[5:])
        self.assertEqual(evidence["missing_score_count"], 3)
        self.assertEqual(evidence["valid_score_count"], 5)
        self.assertEqual([r["price"] for r in projected], [10] * len(rows))
        self.assertEqual([r["signal_date"] for r in projected], ["2021-01-04"] * len(rows))

    def test_key_is_order_selector_schedule_and_score_value_independent(self):
        rows = [candidate(str(i), float(i)) for i in range(20)]
        def mapping(items, seed=42, repeat=1, date="2021-01-05"):
            _, evidence = permute_candidate_scores(items, seed=seed, repeat_id=repeat, decision_date=date)
            return [(r["recipient"], r["donor"]) for r in evidence["mapping"]]
        expected = mapping(rows)
        self.assertEqual(expected, mapping(list(reversed(rows))))
        self.assertEqual(expected, mapping([dict(r, future_target=1e9, selector="Direct") for r in rows]))
        self.assertEqual(expected, mapping([dict(r, breakout_quality_score=-1000*r["breakout_quality_score"]) for r in rows]))
        self.assertNotEqual(expected, mapping(rows, repeat=2))
        self.assertNotEqual(expected, mapping(rows, seed=17))
        self.assertNotEqual(expected, mapping(rows, date="2021-01-06"))

    def test_small_and_tied_universes_and_fixed_points_are_legal(self):
        for rows in ([], [candidate("A", 1)], [candidate("A", 3), candidate("B", 3)]):
            out, evidence = permute_candidate_scores(rows, seed=42, repeat_id=1, decision_date="2021-01-05")
            self.assertEqual(out, rows)
            self.assertFalse(evidence["exchangeable"])
        rows = [candidate("A", 1), candidate("B", 2)]
        outcomes = [permute_candidate_scores(rows, seed=42, repeat_id=i, decision_date="2021-01-05")[0]
                    for i in range(1, 100)]
        self.assertTrue(any(x == rows for x in outcomes))
        self.assertTrue(any(x != rows for x in outcomes))
        with self.assertRaises(ValueError):
            permute_candidate_scores(rows * 2, seed=42, repeat_id=1, decision_date="2021-01-05")
        for repeat in (0, 100):
            with self.assertRaises(ValueError):
                permute_candidate_scores(rows, seed=42, repeat_id=repeat, decision_date="2021-01-05")

    def test_projection_is_once_before_selector_and_does_not_change_state_or_pending(self):
        from core.portfolio_day_candidates import select_portfolio_candidate_orders
        calls = []
        rows = [candidate(str(i), float(i)) for i in range(10)]
        before = deepcopy(rows)
        def reorder(items, **kwargs):
            calls.append((deepcopy(items), kwargs))
            return sorted(items, key=lambda r: -r["breakout_quality_score"]), {"k": 2}
        arguments = dict(today="2021-01-05", orderable_rows=rows, replay_selector_trace_rows=[],
                         replay_execution_rows=None, replay_counts={}, is_training=False,
                         params=SimpleNamespace(use_breakout_quality_ranking=True), available_cash=12345,
                         sizing_equity=23456, pre_market_occupied=3, max_positions=5,
                         replay_phase=lambda day, phase, fn, *a, **kw: fn(*a, **kw),
                         reorder_callback=reorder, select_action_callback=lambda rows, d: rows[:d["k"]])
        evidence = StringIO()
        with permutation_replay_scope(seed=42, repeat_id=1, evidence_file=evidence) as state:
            selected = select_portfolio_candidate_orders(**arguments)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(calls[0][1]["available_cash"], 12345)
        self.assertEqual(calls[0][0], before)
        expected, _ = permute_candidate_scores(rows, seed=42, repeat_id=1, decision_date="2021-01-05")
        self.assertEqual(calls[1][0], expected)
        self.assertEqual(rows, before)
        self.assertEqual(state.summary()["decision_days"], 1)
        record = json.loads(evidence.getvalue())
        self.assertEqual(record["permuted_selected"], [r["ticker"] for r in selected.action_rows])
        calls.clear()
        select_portfolio_candidate_orders(**arguments)
        self.assertEqual(len(calls), 1)
        # A second day starts from refreshed original scores, never yesterday's permutation.
        self.assertEqual(calls[0][0], before)

    def test_no_dl_baseline_is_not_permuted_inside_scope(self):
        from core.candidate_score_permutation import active_permutation
        with permutation_replay_scope(seed=42, repeat_id=1, evidence_file=StringIO()):
            self.assertIsNone(active_permutation(SimpleNamespace(use_breakout_quality_ranking=False)))
        self.assertIsNone(active_permutation(SimpleNamespace(use_breakout_quality_ranking=True)))

    def test_registered_capability_keeps_source_control_and_four_seed_membership(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings
        from core.research_summary import get_research_summary_rows
        settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        robust = get_strategy_multi_seed_robustness_settings(settings.profile_id)
        treated = [arm for arm in settings.enabled_arms if permutation_spec(arm.dl_runtime_options)]
        self.assertTrue(treated)
        for arm in treated:
            spec = permutation_spec(arm.dl_runtime_options)
            reference = settings.arms[spec["control_arm_id"]]
            self.assertEqual(arm.dl_id, reference.dl_id)
            self.assertEqual(arm.dl_runtime_mode, reference.dl_runtime_mode)
            self.assertIn(arm.arm_id, robust.model_seed_sensitive_arm_ids)
            self.assertEqual(len(robust.resolved_seeds), 4)
            groups = [row for row in get_research_summary_rows()
                      if any(b.get("arm_id") == arm.arm_id for b in row.get("bindings", ()))]
            self.assertEqual(len(groups), 1)
            self.assertNotIn("oracle", groups[0])

    def test_canonical_exact_and_direct_replay_accept_projected_candidates(self):
        import pandas as pd
        from core import portfolio_engine as engine
        from core.strategy_params import V16StrategyParams
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.breakout_quality_ranking_context import breakout_quality_ranking_source_context
        from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
        from tools.validate import synthetic_portfolio_common as synthetic
        from tools.validate.synthetic_frame_utils import write_synthetic_csv_bundle
        settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        case = synthetic.build_synthetic_competing_candidates_case(V16StrategyParams())
        case["params"].use_breakout_quality_ranking = True
        def lookup(*, ticker, signal_date, **kwargs):
            day = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
            score = (sorted(case["frames"]).index(ticker) + 1) / len(case["frames"])
            return {"available": True, "score": score, "score_date": day,
                    "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME}
        first_mappings = []
        for arm in settings.enabled_arms:
            if permutation_spec(arm.dl_runtime_options) is None:
                continue
            source = settings.dl_sources[arm.dl_id]
            records = StringIO()
            with TemporaryDirectory() as temporary:
                write_synthetic_csv_bundle(temporary, case["frames"])
                with breakout_quality_ranking_source_context(
                    score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                    model_architecture=source.model_architecture, experiment_profile=source.experiment_profile,
                    ranking_policy=arm.dl_runtime_mode, ranking_options=arm.dl_runtime_options,
                ), patch("filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score", side_effect=lookup), \
                    permutation_replay_scope(seed=42, repeat_id=7, evidence_file=records) as state:
                    synthetic.run_portfolio_core_check_for_dir(
                        temporary, case["params"], max_positions=case["max_positions"],
                        enable_rotation=case["enable_rotation"], start_year=case["start_year"],
                        benchmark_ticker=case["benchmark_ticker"])
            days = [json.loads(line) for line in records.getvalue().splitlines()]
            self.assertGreater(state.counts["candidate_days"], 0)
            self.assertTrue(any(day["valid_score_count"] >= 2 for day in days))
            first_mappings.append(next(day["mapping_sha256"] for day in days if day["valid_score_count"] >= 2))
        self.assertEqual(len(set(first_mappings)), 1)


def synthetic_bundle(seed, values):
    rows = [{"repeat_id": i, "metrics": {ROMD: float(v), MDD: 2., "total_return_pct": v*2},
             "coverage": {"candidate_days": 10, "selection_change_fraction": .2}}
            for i, v in enumerate(values, 1)]
    return dict(kind=KIND, policy=POLICY, status="COMPLETED", seed=seed, repeats=99, completed=99,
                observations=rows, mean_metrics=_mean_records([r["metrics"] for r in rows]),
                distributions={k: distribution([r["metrics"] for r in rows], k) for k in (ROMD, MDD, "total_return_pct")},
                coverage=_mean_records([r["coverage"] for r in rows]))


class FixedPermutationObservationTests(TestCase):
    def test_four_seed_gate_uses_99_paired_means_with_conservative_ties(self):
        real = {s: {ROMD: 10., MDD: 1.} for s in (1, 2, 3, 4)}
        bundles = [synthetic_bundle(s, [0.] * 95 + [10.] * 4) for s in real]
        gate = evaluate_permutation_control(bundles, real)
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["empirical_tail_fraction"], .05)
        self.assertEqual(gate["null_ge_real_count"], 4)
        fail = [synthetic_bundle(s, [0.] * 94 + [10.] * 5) for s in real]
        self.assertEqual(evaluate_permutation_control(fail, real)["status"], "FAIL")
        crossed = [synthetic_bundle(s, [v] * 99) for s, v in zip(real, (200., -200., 0., 0.))]
        self.assertEqual(evaluate_permutation_control(crossed, real)["null_mean_romd_distribution"]["p95"], 0.)
        real[1][MDD] = 100
        self.assertFalse(evaluate_permutation_control(bundles, real)["mdd_gate"])
        self.assertEqual(evaluate_permutation_control(bundles[:3], real)["status"], "INCOMPLETE")

    def test_aggregate_rejects_partial_duplicates_and_edited_statistics(self):
        bundle = synthetic_bundle(42, range(99))
        validate_bundle(bundle, root=Path.cwd(), verify_children=False)
        for mutate in (lambda b: b["observations"].pop(),
                       lambda b: b["observations"][0].update(repeat_id=2),
                       lambda b: b["mean_metrics"].update(return_over_max_drawdown=999),
                       lambda b: b["distributions"][ROMD].update(p95=999)):
            bad = deepcopy(bundle)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_bundle(bad, root=Path.cwd(), verify_children=False)

    def test_checkpoint_resume_reuses_only_complete_pinned_children_and_means_ratios(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls, fail = [], [True]
            def replay(**kwargs):
                directory = kwargs["output_dir_override"]
                repeat = int(directory.name.rsplit("_", 1)[1])
                calls.append(repeat)
                if repeat == 11 and fail[0]:
                    raise RuntimeError("interrupted")
                payload = {"metadata": {}, "score_ranking": {ROMD: 100/repeat, MDD: repeat,
                           "total_return_pct": 100., "trade_count": 0},
                           "no_filter": {ROMD: 1., MDD: 5., "total_return_pct": 5., "trade_count": 0},
                           "yearly": [{"year": 2021, "score_ranking_return_pct": repeat,
                                       "is_complete_year": True}]}
                for name in ("no_filter_trades.csv", "score_ranking_trades.csv"):
                    (directory / name).write_text("\n", encoding="utf-8")
                for name in ("score_ranking_equity.csv", "score_ranking_daily_capacity.csv", "score_ranking_selected_buys.csv"):
                    (directory / name).write_text("Date\n2021-01-01\n", encoding="utf-8")
                (directory / "strategy_comparison.json").write_text(json.dumps(payload), encoding="utf-8")
                return payload
            arguments = dict(output_dir_override=root / "pair", replay_active_only=False)
            spec = {"policy": POLICY, "repeats": REPEATS, "control_arm_id": "original"}
            call = dict(run_once=replay, arguments=arguments, root=root, spec=spec, seed=42, identity={"input": "a"})
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_fixed_permutations(**call)
            self.assertFalse((root / "pair/score_permutation.json").exists())
            self.assertEqual(len(list(root.glob("outputs/**/permutation_receipt.json"))), 10)
            fail[0] = False
            calls.clear()
            payload = run_fixed_permutations(**call)
            self.assertEqual(calls, list(range(11, 100)))
            self.assertAlmostEqual(payload["score_ranking"][ROMD], sum(100/i for i in range(1, 100))/99)
            self.assertNotAlmostEqual(payload["score_ranking"][ROMD], 100/50)
            self.assertFalse((root / "pair/score_ranking_equity.csv").exists())
            self.assertEqual(load_bundle(root / "pair", root=root)["completed"], 99)
            calls.clear()
            run_fixed_permutations(**call)
            self.assertEqual(calls, [])
            receipt = next(root.glob("outputs/**/repeat_050/permutation_receipt.json"))
            original_receipt = receipt.read_text(encoding="utf-8")
            broken = json.loads(original_receipt)
            broken["files"] = []
            receipt.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned native replay"):
                run_fixed_permutations(**call)
            receipt.write_text(original_receipt, encoding="utf-8")
            pinned = next(root.glob("outputs/**/repeat_050/score_ranking_trades.csv"))
            pinned.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                run_fixed_permutations(**call)
            self.assertEqual(calls, [])

    def test_permutation_cannot_use_legacy_unpinned_pair_admission(self):
        from services.research.strategy_compare_pair_integrity import completed_pair_file_integrity_matches
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "manifest.json").write_text("{}", encoding="utf-8")
            pair = directory / "pairs/test"
            pair.mkdir(parents=True)
            args = dict(run_dir=directory, run_payload={}, pair_id="test", pair_dir=pair, required_paths=())
            self.assertTrue(completed_pair_file_integrity_matches(**args))
            (pair / "score_permutation.json").write_text("{}", encoding="utf-8")
            self.assertFalse(completed_pair_file_integrity_matches(**args))

    def test_compact_attribution_preserves_complete_distribution_after_pair_cleanup(self):
        import shutil
        from services.research import strategy_multi_seed_robustness as robustness
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from filters.breakout_quality.strategy_compare_runtime import _arm_runtime_spec
        settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        arm = next(a for a in settings.enabled_arms if permutation_spec(a.dl_runtime_options))
        source = settings.dl_sources[arm.dl_id]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = root / "pair"
            pair.mkdir()
            # This test isolates attribution storage from replay validation,
            # which the interrupted/resume fixture exercises with all 99 receipts.
            bundle = synthetic_bundle(42, range(99))
            (pair / "score_permutation.json").write_text(json.dumps(bundle), encoding="utf-8")
            run = root / "run"
            destination = robustness._attribution_unit_dir(run, arm.arm_id, 42)
            job = dict(scientific_fingerprint="fixture", robustness_id="fixture", profile_id=settings.profile_id,
                       arm_order=1, seed=42, seed_order=1, comparison_start="2021-01-01", comparison_end="2021-12-31")
            with patch.object(robustness, "PROJECT_ROOT", root), \
                 patch("core.score_permutation_observation.load_bundle", return_value=bundle), \
                 patch("core.score_permutation_observation.validate_bundle", side_effect=lambda b, **kw: b):
                manifest = robustness._write_compact_attribution_source(
                    pair_dir=pair, destination_dir=destination, job=job, arm=arm, dl=source,
                    runtime_spec=_arm_runtime_spec(arm))
                self.assertEqual(robustness._attribution_roles(manifest), ("permutations",))
                shutil.rmtree(pair)
                read = robustness._read_attribution_unit_manifest(
                    run, arm_id=arm.arm_id, seed=42, expected_fingerprint="fixture", require_verified=False)
                self.assertIsNotNone(read)
                self.assertEqual(robustness._read_permutation_attribution(read), bundle)
                pinned = root / manifest["files"]["permutations"]["path"]
                pinned.write_bytes(b"corrupt")
                self.assertIsNone(robustness._read_attribution_unit_manifest(
                    run, arm_id=arm.arm_id, seed=42, expected_fingerprint="fixture", require_verified=False))


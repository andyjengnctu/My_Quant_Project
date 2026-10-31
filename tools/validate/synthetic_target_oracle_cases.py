"""Isolated contract regression for reusable Oracle target → RoMD Audit.

Run directly; requires no historical market data and does not touch formal
Test Suite frozen entry checks, PIT artifacts, or portfolio model fitting.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sys
import tempfile
import pandas as pd
from unittest import TestCase, main
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audit_policy import get_audit_definitions, get_reusable_audit_definitions
from core.breakout_quality_ranking_context import (
    SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH,
    SUPPORTED_RANKING_SCORE_SOURCES,
    breakout_quality_ranking_source_context,
    get_breakout_quality_ranking_source_context,
)
from filters.breakout_quality.runtime import (
    audit_oracle_truth_score_context,
    resolve_breakout_quality_candidate_rank,
)
from services.audit import target_oracle_romd as oracle


def _synthetic_full_window_yearly(*, return_pct: float = 5.0) -> list[dict]:
    """Complete 2021-2025 simulated replay evidence, not a one-year cache stub."""
    return [{"year": year, "strategy_return_pct": float(return_pct),
             "is_full_year": year != 2025} for year in range(2021, 2026)]

# Historical stock-day Oracle regressions must not instantiate a P16 path
# provider in a temporary no-market fixture. The active suite is tested below.
_ACTIVE_ORACLE_DEFINITION = next(
    d for d in get_reusable_audit_definitions("breakout_quality")
    if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE
)
_NON_PRICE_SCORE_TARGETS = frozenset(
    set(oracle._oracle_comparison_plan(_ACTIVE_ORACLE_DEFINITION)["daily_sources"])
)
LEGACY_ORACLE_TARGETS = tuple(
    item for item in oracle._oracle_comparison_plan(_ACTIVE_ORACLE_DEFINITION)["targets"]
    if item[0] not in _NON_PRICE_SCORE_TARGETS
)
from services.audit.runner import collect_audit_menu_state, collect_audit_status
from services.audit.catalog import get_audit_methods


def _forty_only(definition):
    """Scope old fixed-40D fixtures without changing the active Registry."""
    return replace(definition, dimensions={**definition.dimensions,
        "oracle_evaluation_end_date": "2025-12-22"},
        source={**definition.source, "oracle_comparisons": tuple(
        item for item in definition.source["oracle_comparisons"]
        if item["target"] not in {
            "DailyUniversal_MinTrade60_NetR",
            "DailyUniversal_MinTrade60_NetR_Direct",
            "DailyUniversal_MinTrade60_NetPnL",
            "DailyUniversal_MinTrade60_NetPnL_Direct",
            "DailyUniversal_MinTrade_NetR",
            "DailyUniversal_MinTrade_NetR_Direct",
            "DailyUniversal_MinTrade_NetPnL",
            "DailyUniversal_MinTrade_NetPnL_Direct",
            "DailyUniversal_MinTrade_NetPnL_Positive_Direct",
        })})


def _legacy_definition():
    """Historical fixed-nine fixtures, independent of currently active Target rows."""
    active = next(d for d in get_reusable_audit_definitions("breakout_quality")
                  if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
    return replace(active, dimensions={**active.dimensions,
        "oracle_evaluation_end_date": "2025-12-22"},
        source={**active.source, "oracle_comparisons": tuple(
        item for item in active.source["oracle_comparisons"]
        if item["target"] not in _NON_PRICE_SCORE_TARGETS)})


def _mock_panels(common40):
    """Canonical one-window fixture: Exact and Direct share the identical daily truth."""
    return ({name: {("2330", "2025-01-02"): 0.8,
                    ("2330", "2025-12-22"): 0.8}
             for name, _ in LEGACY_ORACLE_TARGETS}, common40)


def _fixture_perfect_score_contracts(definition, settings, plan):
    """Replay-only deterministic contract fixture, not used for real resolution."""
    return {name: {
        "reference_arm_id": "synthetic_authorized",
        "target_id": "synthetic_perfect_label",
        "source_profile_id": str(definition.source[
            plan["daily_recipes"][plan["daily_sources"][name]][1]
            if name in plan["daily_sources"] else
            plan["recipes"][plan["aliases"].get(name, name)][1]]),
        "perfect_score_mapping": "canonical_daily_truth_percentile_v2",
        "score_start_date": "2021-01-01",
    } for name, _ in plan["targets"]}


def _fixture_read_score_keys(_definition, *, contracts, **_kwargs):
    """Independent two-row eligibility evidence for replay-only fixtures."""
    import hashlib
    keys = [("2330", "2025-01-02"), ("2330", "2025-12-22")]
    digest = hashlib.sha256("".join(f"{ticker}|{day}\n" for ticker, day in keys).encode()).hexdigest()
    proof = (digest, len(keys), keys[0][1], keys[-1][1])
    return {name: proof for name in contracts}


def _fixture_price_scores(definition, *, project_root, cutoff, **_kwargs):
    """Replay fixtures supply selected panels; source-cache behavior has real-store tests."""
    panels, cohort = oracle._build_oracle_score_panels(
        definition, project_root=project_root, cutoff=cutoff)
    selected = {entry["target"] for entry in definition.source["oracle_comparisons"]}
    return {name: scores for name, scores in panels.items() if name in selected}, cohort


class OracleResearchCutoff2026Cases(TestCase):
    """The active Oracle consumes full Research OOS, not 40D-mature labels."""

    def test_oracle_cold_start_inherits_optional_profile_architecture(self):
        """A None profile override is inherited, never passed as string 'None'."""
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        definition = _ACTIVE_ORACLE_DEFINITION
        price_profile = get_breakout_quality_experiment_profile(
            str(definition.source["pure_mfe40_provider"]))
        self.assertIsNone(price_profile.model_architecture)

        plan = oracle._oracle_comparison_plan(definition)
        settings, _base, _exact = oracle._contract(definition)
        all_contracts = oracle._resolve_oracle_target_score_contracts(
            definition, settings, plan)
        selected = ("R40", "DailyUniversal_MinTrade40_NetR")
        contracts = {name: all_contracts[name] for name in selected}
        group = pd.DataFrame({
            "ticker": ["2330", "2330"],
            "date": ["2021-01-04", "2026-03-02"],
        })
        bundle = SimpleNamespace(
            summary={"filter_id": definition.source["filter_id"]},
            group_table=group, target_valid=np.array([True, False]),
            raw_target=np.array([0.5, np.nan]),
        )
        architectures = []
        def load_data(**kwargs):
            architectures.append(kwargs["model_architecture"])
            return bundle
        with patch.object(oracle, "load_profile_continuous_ranker_data",
                          side_effect=load_data):
            cohorts = oracle._read_only_score_keys(
                definition, project_root=ROOT, contracts=contracts, plan=plan,
                evaluation_end="2026-03-02", cutoff="2026-03-02",
            )
        self.assertEqual(architectures,
                         [definition.source["model_architecture"]] * len(selected))
        self.assertEqual(len({item[0] for item in cohorts.values()}), 1)
        self.assertEqual(cohorts["R40"][1:], (2, "2021-01-04", "2026-03-02"))

        overridden = replace(definition, source={**definition.source,
            "model_architecture": "multiscale_cnn_v3"})
        override_calls = []
        with patch.object(oracle, "load_profile_continuous_ranker_data",
                          side_effect=lambda **kw: override_calls.append(
                              kw["model_architecture"]) or bundle):
            oracle._read_only_score_keys(
                overridden, project_root=ROOT,
                contracts={"R40": contracts["R40"]}, plan=plan,
                evaluation_end="2026-03-02", cutoff="2026-03-02",
            )
        self.assertEqual(override_calls, ["multiscale_cnn_v3"])

    def test_active_oracle_evaluates_through_research_cutoff(self):
        from config.market_data import RESEARCH_REQUIRED_CUTOFF
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, _base, _exact = oracle._contract(definition)
        self.assertEqual(str(definition.dimensions["oracle_evaluation_end_date"]),
                         str(settings.end_date))
        self.assertEqual(str(settings.end_date), RESEARCH_REQUIRED_CUTOFF)
        self.assertGreaterEqual(int(settings.end_date[:4]), 2026)

    def test_price_labels_keep_inference_only_tail_without_fitting_it(self):
        provider = "pure_mfe40_provider"
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        profile = get_breakout_quality_experiment_profile(
            str(_ACTIVE_ORACLE_DEFINITION.source[provider]))
        old_day = "2026-02-26"
        tail_day = "2026-03-02"
        bundle = SimpleNamespace(
            target_manifest={"target_contract": {"target_id": profile.continuous_target_id}},
            group_table=pd.DataFrame({
                "ticker": ["2330", "2330"], "date": [old_day, tail_day],
                "label_eval_end_date": ["2026-03-02", pd.NaT],
                "target_favorable_r": [0.5, np.nan],
                "target_adverse_r": [0.2, np.nan],
                "target_minimum_low_return": [-0.03, np.nan],
            }),
            raw_target=np.array([0.5, np.nan]),
            target_valid=np.array([True, False]),
        )
        with patch.object(oracle, "load_profile_continuous_ranker_data", return_value=bundle):
            official = oracle._load_truth(
                _ACTIVE_ORACLE_DEFINITION, ROOT, provider, cutoff=tail_day)
        self.assertEqual(set(official["date"]), {old_day, tail_day})
        self.assertEqual(official["_oracle_provider_mature"].tolist(), [True, False])
        market = pd.DataFrame({
            "ticker": ["2330", "2330"], "date": [old_day, tail_day],
            "mfe": [0.5, 0.0], "peak_adverse": [0.2, 0.0],
            "min_low": [-0.03, 0.0], "raw": [0.5, 0.0],
            "scalar_raw_terminal_r": [0.1, 0.0],
        })
        matched, clipped_rows = oracle._reconcile_oracle_truth(
            official, market, field=provider, score_start="2021-01-01")
        self.assertEqual(len(matched), 2)
        self.assertEqual(clipped_rows, 1)
        self.assertEqual(float(matched.loc[matched["date"] == tail_day, "raw"].iloc[0]), 0.0)
        # Training-label maturity cannot be retroactively changed to Oracle truth.
        self.assertTrue(np.isnan(bundle.raw_target[-1]))
        self.assertFalse(bundle.target_valid[-1])

    def test_price_truth_clips_horizon_to_cutoff_and_ignores_later_bars(self):
        dates = pd.to_datetime(["2026-02-25", "2026-02-26", "2026-02-27",
                                "2026-03-02", "2026-03-03", "2026-03-04"])
        prices = pd.DataFrame({
            "Close": [100.0, 101.0, 102.0, 105.0, 900.0, 1200.0],
            "High": [101.0, 102.0, 104.0, 110.0, 990.0, 1300.0],
            "Low": [99.0, 100.0, 101.0, 103.0, 890.0, 1190.0],
        }, index=dates)
        provider = "pure_mfe40_provider"
        keys = ["2026-02-26", "2026-02-27", "2026-03-02"]
        eligible = {provider: pd.DataFrame({"ticker": ["2330"] * len(keys), "date": keys})}
        def market_price(frame):
            with patch.object(oracle, "discover_dataset_csv_inputs", return_value=(
                         [("0050", Path("0050.csv")), ("2330", Path("2330.csv"))], [])), \
                 patch.object(oracle, "load_dataset_frame", side_effect=lambda path, ticker, **kw: (
                     prices if ticker == "0050" else frame)):
                return oracle._market_truth_for_oracle(
                    _ACTIVE_ORACLE_DEFINITION, ROOT, cutoff="2026-03-02",
                    provider_fields=(provider,), eligible_truth=eligible)
        truth, _evidence = market_price(prices)
        by_day = truth[provider].set_index("date")
        self.assertEqual(set(by_day.index), set(keys))
        risk = oracle.StrategyAlignedContinuousTargetSpec.from_label_policy(
            oracle.DEFAULT_LABEL_POLICY).risk_budget_return
        self.assertAlmostEqual(float(by_day.loc["2026-02-27", "scalar_raw_terminal_r"]),
                               (105.0 / 102.0 - 1.0) / risk)
        self.assertAlmostEqual(float(by_day.loc["2026-02-27", "mfe"]),
                               (110.0 / 102.0 - 1.0) / risk)
        self.assertEqual(float(by_day.loc["2026-03-02", "mfe"]), 0.0)
        self.assertEqual(float(by_day.loc["2026-03-02", "scalar_raw_terminal_r"]), 0.0)
        # Every scalar label uses exactly the observed path before cutoff,
        # including one-bar clips and the final zero-length information day.
        from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
        target_ids = {
            "mfe": "daily_full_horizon_pure_mfe_r_v1",
            "terminal_r": "daily_terminal_r40_v1",
            "terminal_mae_gate": "daily_terminal_mae_gated_t40_v1",
            "mfe_mae_gate": "daily_full_horizon_mae_gated_mfe_t40_v1",
        }
        spec = oracle.StrategyAlignedContinuousTargetSpec.from_label_policy(oracle.DEFAULT_LABEL_POLICY)
        observed = prices.loc[:"2026-03-02"]
        for day in keys:
            position = int(observed.index.get_loc(day))
            remaining = len(observed) - position - 1
            for recipe, target_id in target_ids.items():
                expected = np.float32(0) if remaining == 0 else compute_daily_opportunity_target_batch(
                    observed, np.array([position]), spec=replace(spec, horizon_bars=remaining),
                    target_id=target_id).target_raw_r[0]
                self.assertEqual(by_day.loc[day, f"scalar_raw_{recipe}"], expected)
        changed = prices.copy()
        changed.loc["2026-03-03":, ["High", "Low", "Close"]] = [40000., 0.01, 40000.]
        changed_truth, _ = market_price(changed)
        pd.testing.assert_frame_equal(
            truth[provider].sort_values(["ticker", "date"]).reset_index(drop=True),
            changed_truth[provider].sort_values(["ticker", "date"]).reset_index(drop=True),
        )

    def test_full_research_period_requires_2026_partial_year_per_seed(self):
        # The Oracle must not merely use 2026 prices for Target truth while
        # silently dropping 2026 portfolio performance from RoMD/yearly evidence.
        records = []
        for seed in (101, 202):
            records.append({
                "seed": seed, "target": "C96", "total_return_pct": 50.0,
                "annualized_return_pct": 10.0, "mdd_pct": 10.0,
                "romd": 5.0, "ev_a_r": 0.3, "trades": 10,
                "exposure_pct": 70.0, "win_rate_pct": 56.0,
                "yearly": [{"year": year, "strategy_return_pct": float(year - 2020),
                            "is_full_year": year < 2026}
                           for year in range(2021, 2027)],
            })
        full = oracle._aggregate(
            records, target="C96", period_start="2021-01-01",
            period_end="2026-03-02",
        )
        self.assertEqual([int(row["year"]) for row in full["yearly"]],
                         list(range(2021, 2027)))
        self.assertEqual(full["yearly"][-1]["strategy_return_pct"], 6.0)
        self.assertFalse(full["yearly"][-1]["is_full_year"])
        missing = [dict(row, yearly=row["yearly"][:-1]) for row in records]
        with self.assertRaisesRegex(oracle.OracleBlockedError, "annual coverage"):
            oracle._aggregate(missing, target="C96", period_start="2021-01-01",
                              period_end="2026-03-02")

    def test_full_cutoff_rejects_old_end_date_cache_without_truth_precompute(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            seed = 123
            base = dict(dataset="full", dataset_sha256="a" * 64,
                target="C96", definition={"C96": "C96"}, start="2021-01-01",
                benchmark="end_to_end_v1", param_sha256="b" * 64, seed=seed,
                cohort={"cohort_rows": 2, "research_cutoff": "2026-03-02"},
                ranking_policy="exact", ranking_options={},
                evaluation_mode="rolling", max_positions=10,
                replay_policy={"oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION})
            old = replay_identity(**base, end="2025-12-22")
            new = replay_identity(**base, end="2026-03-02")
            self.assertNotEqual(old, new)
            self.assertEqual(store.plan_execution({(seed, "C96"): new},
                            allow_historical_execution_input_drift=True).action(seed, "C96"), "RUN")


class OracleScoreParityContractCases(TestCase):
    def setUp(self):
        # Preserve the real callable fingerprint before individual tests
        # replace external panel loading. Both input checks see one fixture.
        source = patch.object(oracle, "_price_score_algorithm_sha256",
                              return_value=oracle._price_score_algorithm_sha256())
        source.start()
        self.addCleanup(source.stop)
        prices = patch("services.audit.oracle_replay_artifacts.price_inventory_sha256",
                       side_effect=lambda *args: oracle.price_inventory_sha256(*args))
        prices.start()
        self.addCleanup(prices.stop)

    @staticmethod
    def _feature_bundle(keys):
        ordered = sorted(keys)
        return SimpleNamespace(
            group_table=pd.DataFrame(ordered, columns=["ticker", "date"]),
            summary={"filter_id": "breakout_quality_v1"},
            target_valid=np.ones(len(ordered), dtype=bool),
            raw_target=np.zeros(len(ordered), dtype=np.float32))

    def test_hypothetical_oracle_resolves_registered_targets_without_existing_target_models(self):
        # This is the *actual* canonical Registry, no mocked contract resolver.
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, _base, exact = oracle._contract(definition)
        plan = oracle._oracle_comparison_plan(definition)
        mapping = oracle._resolve_oracle_target_score_contracts(definition, settings, plan)
        self.assertEqual(len(mapping), len(_ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]))
        self.assertEqual(mapping["T40"]["reference_arm_id"], exact.arm_id)
        self.assertEqual(mapping["CG40_EqualRank"]["perfect_score_mapping"],
                         "canonical_daily_truth_percentile_v2")
        self.assertFalse(any(x["model_fit_required"] for x in mapping.values()))
        self.assertTrue(all("not_unique_raw_pairwise_logits" in x["score_semantics"]
                            for x in mapping.values()))
        from services.audit.daily_trade40_oracle import TARGET_ID
        for name in ("DailyUniversal_MinTrade40_NetR", "DailyUniversal_MinTrade40_NetR_Direct"):
            self.assertEqual(mapping[name]["target_id"], TARGET_ID)
            self.assertNotIn("counterfactual_mintrade40", mapping[name]["target_id"])

    def test_direct_only_price_and_daily_do_not_require_selected_exact_replay(self):
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, _base, _exact = oracle._contract(definition)
        for selected, provider, hidden_source in (
            ("R40_Direct", "pure_mfe40_provider", "R40"),
            ("DailyUniversal_MinTrade40_NetR_Direct", "daily_trade40_profile",
             "DailyUniversal_MinTrade40_NetR"),
        ):
            restricted = replace(definition, source={**definition.source,
                "oracle_comparisons": tuple(row for row in definition.source["oracle_comparisons"]
                                            if row["target"] == selected)})
            plan = oracle._oracle_comparison_plan(restricted)
            resolved = oracle._resolve_oracle_target_score_contracts(restricted, settings, plan)
            self.assertEqual(set(resolved), {selected})
            self.assertEqual([name for name, _ in plan["targets"]], [selected])
            self.assertTrue(plan["selectors"][selected] == "direct")
            self.assertIn(hidden_source, plan["recipes"] or plan["daily_recipes"])
            self.assertEqual(restricted.source[provider], definition.source[provider])
            self.assertNotIn(hidden_source, resolved)
            self.assertNotIn(hidden_source, plan["daily_sources"])

    def test_direct_only_with_no_enabled_exact_uses_formal_direct_contract(self):
        # Direct-only must not require an unrelated, disabled Exact selector.
        # Historical C101 is explicitly registered for this controlled Audit.
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, _base, _exact = oracle._contract(definition)
        disabled_exact = replace(settings, arms={
            name: replace(arm, enabled=False)
            if arm.dl_runtime_mode == "resource-aware-continuous-score-constrained-optimal"
            else arm for name, arm in settings.arms.items()
        })
        for target in ("R40_Direct", "DailyUniversal_MinTrade40_NetR_Direct"):
            selected = replace(definition, source={**definition.source,
                "oracle_comparisons": tuple(
                    row for row in definition.source["oracle_comparisons"]
                    if row["target"] == target)})
            plan = oracle._oracle_comparison_plan(selected)
            _settings, _base, exact = oracle._contract(selected, settings=disabled_exact)
            self.assertIsNone(exact)  # No hidden Exact selector is required.
            direct = oracle._direct_arm(selected, disabled_exact, _base, exact)
            self.assertEqual(direct.dl_runtime_mode,
                             "resource-aware-continuous-score-no-k-no-r0")
            self.assertEqual(oracle._formal_dl_score_start(selected, disabled_exact),
                             "2021-01-01")
            contracts = oracle._resolve_oracle_target_score_contracts(
                selected, disabled_exact, plan)
            self.assertEqual(set(contracts), {target})
            self.assertEqual(contracts[target]["selector"], "direct")
        with self.assertRaisesRegex(oracle.OracleBlockedError,
                                    "active Exact selector unavailable"):
            oracle._contract(definition, settings=disabled_exact)

    def test_direct_only_rejects_invalid_registered_direct_reference(self):
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, _base, _exact = oracle._contract(definition)
        selected = replace(definition, source={**definition.source,
            "oracle_comparisons": tuple(
                row for row in definition.source["oracle_comparisons"]
                if row["target"] == "R40_Direct")})
        without_exact = replace(settings, arms={
            name: replace(arm, enabled=False)
            if arm.dl_runtime_mode == "resource-aware-continuous-score-constrained-optimal"
            else arm for name, arm in settings.arms.items()
        })
        direct = without_exact.arms[str(selected.source["direct_arm_id"])]
        broken = replace(without_exact, arms={**without_exact.arms,
            direct.arm_id: replace(direct, dl_runtime_options={
                **direct.dl_runtime_options, "preserve_k": True})})
        with self.assertRaisesRegex(oracle.OracleBlockedError, "Direct selector contract"):
            oracle._contract(selected, settings=broken)

    def test_direct_first_exact_second_resolves_without_order_dependency(self):
        definition = _ACTIVE_ORACLE_DEFINITION
        wanted = ("R40_Direct", "R40")
        comparison = tuple(next(row for row in definition.source["oracle_comparisons"]
                                if row["target"] == name) for name in wanted)
        changed = replace(definition, source={**definition.source,
                                             "oracle_comparisons": comparison})
        plan = oracle._oracle_comparison_plan(changed)
        self.assertEqual([name for name, _ in plan["targets"]], list(wanted))
        self.assertEqual(plan["recipes"]["R40"], ("terminal_r", "pure_mfe40_provider"))

    def test_direct_only_price_materializes_only_selected_treatment(self):
        definition = _ACTIVE_ORACLE_DEFINITION
        selected = "R40_Direct"
        restricted = replace(definition, source={**definition.source,
            "oracle_comparisons": tuple(row for row in definition.source["oracle_comparisons"]
                                        if row["target"] == selected)})
        rows = pd.DataFrame({
            "ticker": ["2330", "2317"], "date": ["2021-01-04"] * 2,
            "mfe": [1.1, .4], "peak_adverse": [-.3, -.4],
            "min_low": [-.02, -.04], "raw": [1.1, .4],
            "scalar_raw_terminal_r": [1.0, -.5],
        })
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(oracle, "_load_truth", return_value=rows), \
             patch.object(oracle, "_market_truth_for_oracle", return_value=(
                 {"pure_mfe40_provider": rows},
                 {"score_start":"2021-01-01", "mature_rows_by_provider": {}})), \
             patch.object(oracle, "_reconcile_oracle_truth", return_value=(rows, 0)):
            scores, _evidence = oracle._load_or_build_oracle_price_scores(
                restricted, project_root=Path(directory), cutoff="2026-03-02", price_sha="f" * 64)
        self.assertEqual(set(scores), {selected})
        self.assertEqual(set(scores[selected].values()), {0.0, 1.0})
        expected = {("2317", "2021-01-04"): 0.0, ("2330", "2021-01-04"): 1.0}
        self.assertEqual(oracle._canonical_score_map_sha256(scores[selected]),
                         oracle._canonical_score_map_sha256(expected))

    def test_target_score_contract_uses_provided_current_settings(self):
        # The caller-resolved settings are the only authorized selector input.
        # A second implicit settings fetch could pin the wrong Exact arm ID.
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, _base, first_exact = oracle._contract(definition)
        current_arms = {**settings.arms}
        current_arms[first_exact.arm_id] = replace(first_exact, enabled=False)
        changed_settings = replace(settings, arms=current_arms)
        _current, _baseline, second_exact = oracle._contract(
            definition, settings=changed_settings,
        )
        self.assertNotEqual(first_exact.arm_id, second_exact.arm_id)
        resolved = oracle._resolve_oracle_target_score_contracts(
            definition, changed_settings, oracle._oracle_comparison_plan(definition),
        )
        self.assertEqual(resolved["T40"]["reference_arm_id"], second_exact.arm_id)
        self.assertEqual(resolved["T40"]["reference_score_source_id"], second_exact.dl_id)

    def test_reference_dl_must_be_daily_feature_eligible(self):
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, _base, _exact = oracle._contract(definition)
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        original = get_breakout_quality_experiment_profile
        def source_profile(profile_id):
            profile = original(profile_id)
            return replace(profile, training_sample_scope="breakout_event_groups")
        with patch.object(oracle, "get_breakout_quality_experiment_profile", side_effect=source_profile):
            with self.assertRaisesRegex(oracle.OracleBlockedError, "feature eligibility"):
                oracle._formal_dl_score_start(definition, settings)

    def test_irrelevant_enabled_dl_source_does_not_change_oracle_score_start(self):
        # A concurrently enabled, differently-scoped model is NOT an input
        # to the registered Exact/Direct Oracle selector. Previously the
        # score-start resolver scanned every enabled model and blocked.
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, _base, exact = oracle._contract(definition)
        extra_arm = replace(
            exact, arm_id="C_UNRELATED_SYNTHETIC", dl_id="DL_UNRELATED_SYNTHETIC",
            dl_runtime_mode="independent_synthetic_ranking_mode",
        )
        extra_source = replace(
            settings.dl_sources[exact.dl_id], dl_id="DL_UNRELATED_SYNTHETIC",
            point_in_time_score_start_date="2022-01-01",
        )
        changed_settings = replace(settings,
            arms={**settings.arms, extra_arm.arm_id: extra_arm},
            dl_sources={**settings.dl_sources, extra_source.dl_id: extra_source},
        )
        self.assertEqual(
            oracle._formal_dl_score_start(definition, changed_settings),
            oracle._formal_dl_score_start(definition, settings),
        )
        # A drift of the actual selected Exact score period must still fail.
        exact_source = settings.dl_sources[exact.dl_id]
        invalid = replace(settings, dl_sources={
            **settings.dl_sources, exact.dl_id: replace(
                exact_source, point_in_time_score_start_date="2022-01-01",
            ),
        })
        with self.assertRaisesRegex(oracle.OracleBlockedError, "score start"):
            oracle._formal_dl_score_start(definition, invalid)
        # A second eligible Exact source, unlike the unrelated research arm,
        # must not be allowed to define an incompatible score period.
        matching_arm = replace(
            exact, arm_id="C_SECOND_EXACT_SYNTHETIC", dl_id="DL_SECOND_EXACT_SYNTHETIC",
        )
        mismatched_source = replace(
            settings.dl_sources[exact.dl_id], dl_id="DL_SECOND_EXACT_SYNTHETIC",
            point_in_time_score_start_date="2022-01-01",
        )
        changed = replace(settings,
            arms={**settings.arms, matching_arm.arm_id: matching_arm},
            dl_sources={**settings.dl_sources,
                        mismatched_source.dl_id: mismatched_source},
        )
        with self.assertRaisesRegex(oracle.OracleBlockedError, "score start"):
            oracle._formal_dl_score_start(definition, changed)

    def test_other_eligible_exact_reference_cannot_use_another_filter_universe(self):
        # A same-selector reference with different filter cannot silently
        # redefine the PIT-eligible candidate population of the Oracle.
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, _base, exact = oracle._contract(definition)
        other_arm = replace(
            exact, arm_id="C_OTHER_FILTER_SYNTHETIC", dl_id="DL_OTHER_FILTER_SYNTHETIC",
        )
        other_source = replace(
            settings.dl_sources[exact.dl_id],
            dl_id="DL_OTHER_FILTER_SYNTHETIC", filter_id="other-filter-identity",
        )
        changed = replace(settings,
            arms={**settings.arms, other_arm.arm_id: other_arm},
            dl_sources={**settings.dl_sources, other_source.dl_id: other_source},
        )
        with self.assertRaisesRegex(oracle.OracleBlockedError, "filter_id mismatch"):
            oracle._formal_dl_score_start(definition, changed)
        # A genuinely unrelated selector source can have its own filter.
        unrelated = replace(other_arm, dl_runtime_mode="other_ranking_mode")
        unrelated_settings = replace(settings,
            arms={**settings.arms, unrelated.arm_id: unrelated},
            dl_sources={**settings.dl_sources, other_source.dl_id: other_source},
        )
        self.assertEqual(oracle._formal_dl_score_start(definition, unrelated_settings),
                         oracle._formal_dl_score_start(definition, settings))

    def test_exact_only_does_not_depend_on_unselected_direct_selector(self):
        definition = _ACTIVE_ORACLE_DEFINITION
        exact_only = replace(definition, source={
            **definition.source,
            "oracle_comparisons": tuple(
                row for row in definition.source["oracle_comparisons"]
                if row["selector"] == "exact"
            ),
        })
        self.assertFalse(oracle._oracle_requires_direct(exact_only))
        settings, _base, _exact = oracle._contract(definition)
        direct_id = str(definition.source["direct_arm_id"])
        invalid_direct = replace(
            settings.arms[direct_id], dl_runtime_mode="retired_synthetic_mode",
        )
        modified = replace(settings, arms={
            **settings.arms, direct_id: invalid_direct,
        })
        with patch.object(oracle, "get_strategy_comparison_settings", return_value=modified):
            selected, _base, _exact = oracle._contract(exact_only)
            self.assertEqual(selected.profile_id, modified.profile_id)
            plan = oracle._oracle_comparison_plan(exact_only)
            contracts = oracle._resolve_oracle_target_score_contracts(
                exact_only, modified, plan,
            )
            self.assertEqual(set(contracts), {row["target"] for row in exact_only.source["oracle_comparisons"]})
            self.assertEqual(
                oracle._formal_dl_score_start(exact_only, modified), "2021-01-01",
            )
            # The same invalid Direct selector MUST fail if selected.
            with self.assertRaisesRegex(oracle.OracleBlockedError, "Direct selector"):
                oracle._contract(definition)

    def test_reference_dl_filter_must_match_oracle_candidate_universe(self):
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, _base, exact = oracle._contract(definition)
        source = settings.dl_sources[exact.dl_id]
        invalid_source = replace(source, filter_id="different_filter")
        changed_settings = replace(settings, dl_sources={
            **settings.dl_sources, exact.dl_id: invalid_source,
        })
        plan = oracle._oracle_comparison_plan(definition)
        with patch.object(oracle, "get_strategy_comparison_settings", return_value=changed_settings):
            with self.assertRaisesRegex(oracle.OracleBlockedError, "filter_id"):
                oracle._resolve_oracle_target_score_contracts(definition, changed_settings, plan)

    def test_unapproved_score_mapping_still_fails_closed(self):
        altered = replace(_ACTIVE_ORACLE_DEFINITION, dimensions={
            **_ACTIVE_ORACLE_DEFINITION.dimensions, "oracle_score_method": "arbitrary_logits",
        })
        with self.assertRaisesRegex(oracle.OracleBlockedError, "CONTRACT_UNRESOLVED"):
            oracle._oracle_comparison_plan(altered)
        with tempfile.TemporaryDirectory() as directory:
            status = oracle.collect_status(altered, project_root=Path(directory))
            self.assertIn("CONTRACT_UNRESOLVED", status["reason"])

    def test_score_value_changes_sha_even_when_candidate_set_does_not(self):
        earlier = {("2330", "2025-01-02"): 0.25,
                   ("2317", "2025-01-02"): 0.75}
        later = dict(earlier)
        later[("2330", "2025-01-02")] = 0.35
        self.assertNotEqual(oracle._canonical_score_map_sha256(earlier),
                            oracle._canonical_score_map_sha256(later))

    def test_perfect_score_sha_requires_approved_percentile_numerical_range(self):
        key = ("2330", "2021-01-04")
        self.assertEqual(len(oracle._canonical_score_map_sha256({key: 0.0})), 64)
        self.assertEqual(len(oracle._canonical_score_map_sha256({key: 1.0})), 64)
        for invalid in (-0.01, 1.01, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    oracle.OracleBlockedError, "percentile scale"):
                oracle._canonical_score_map_sha256({key: invalid})

    def test_no_model_gate_or_fitted_pit_file_is_required_for_hypothetical_coverage(self):
        # Score eligibility is independently verified by each canonical daily
        # feature/target producer; a matching future *model* may not exist.
        definition = _forty_only(_ACTIVE_ORACLE_DEFINITION)
        settings, _base, _exact = oracle._contract(definition)
        plan = oracle._oracle_comparison_plan(definition)
        contracts = oracle._resolve_oracle_target_score_contracts(definition, settings, plan)
        source_name = "DailyUniversal_MinTrade40_NetR"
        key = ("2330", "2021-01-04")
        daily = {1: {source_name: {key: 0.8}}, 2: {source_name: {key: 0.9}}}
        prices = {name: {key: 0.5} for name in plan["recipes"]}
        for direct, source in plan["aliases"].items():
            if source in prices:
                prices[direct] = prices[source]
        with patch.object(oracle, "load_profile_continuous_ranker_data",
                          return_value=self._feature_bundle({key})):
            certified = oracle._read_only_score_keys(
                definition, project_root=Path("/isolated"), contracts=contracts, plan=plan,
                evaluation_end="2025-12-22", cutoff=str(settings.end_date))
        self.assertEqual(len(certified), 11)
        self.assertEqual(certified[source_name][1], 1)
        self.assertEqual(len(certified[source_name][0]), 64)
        self.assertFalse(any("pit_score_manifest_sha256" in item for item in contracts.values()))
        from tools.validate.synthetic_daily_trade40_oracle_cases import execute_synthetic_score_fixture
        completed = execute_synthetic_score_fixture(
            definition, price_scores=prices, daily_scores=daily, keys={key})
        self.assertEqual(len(completed), 22, "same keys may carry different same-seed P16 outcomes")
        daily[2][source_name][("2317", "2021-01-04")] = 0.2
        with self.assertRaisesRegex(oracle.OracleBlockedError, "producer score eligibility differs"):
            execute_synthetic_score_fixture(
                definition, price_scores=prices, daily_scores=daily, keys={key})
        daily[2][source_name].pop(("2317", "2021-01-04"))
        prices["MFE40_Control"][("2330", "2020-12-31")] = 0.4
        with self.assertRaisesRegex(oracle.OracleBlockedError, "producer score eligibility differs"):
            execute_synthetic_score_fixture(
                definition, price_scores=prices, daily_scores=daily, keys={key})

    def test_cross_target_scored_cohort_must_be_identical_not_just_same_count(self):
        definition = _forty_only(_ACTIVE_ORACLE_DEFINITION)
        settings, _base, _exact = oracle._contract(definition)
        plan = oracle._oracle_comparison_plan(definition)
        contracts = oracle._resolve_oracle_target_score_contracts(definition, settings, plan)
        row = ("2330", "2021-01-04")
        alternative = ("2317", "2021-01-04")
        prices = {name: {row: 0.5} for name in plan["recipes"]}
        for alias, source in plan["aliases"].items():
            if source in prices:
                prices[alias] = prices[source]
        daily_name = "DailyUniversal_MinTrade40_NetR"
        daily = {seed: {daily_name: {alternative: 0.8}}
                 for seed in (693545351, 2014432738, 104504529, 762252050)}
        from tools.validate.synthetic_daily_trade40_oracle_cases import execute_synthetic_score_fixture
        with self.assertRaisesRegex(oracle.OracleBlockedError, "producer score eligibility differs"):
            execute_synthetic_score_fixture(definition, price_scores=prices, daily_scores=daily, keys={row})
        daily = {seed: {daily_name: {row: 0.8}} for seed in daily}
        with patch.object(oracle, "load_profile_continuous_ranker_data",
                          return_value=self._feature_bundle({row})):
            certified = oracle._read_only_score_keys(
                definition, project_root=Path("/isolated"), contracts=contracts, plan=plan,
                evaluation_end="2025-12-22", cutoff=str(settings.end_date))
        self.assertEqual(set(certified.values()), {certified[daily_name]})
        # Price-only comparisons and legitimate Daily-only comparisons do not
        # require unselected providers to exist.
        for selected in ({"T40", "T40-MFE"}, {daily_name, daily_name+"_Direct"}):
            subset = replace(definition, source={**definition.source,
                "oracle_comparisons": tuple(item for item in definition.source["oracle_comparisons"]
                                            if item["target"] in selected)})
            subplan = oracle._oracle_comparison_plan(subset)
            subcontracts = oracle._resolve_oracle_target_score_contracts(subset, settings, subplan)
            with patch.object(oracle, "load_profile_continuous_ranker_data",
                              return_value=self._feature_bundle({row})):
                certified = oracle._read_only_score_keys(
                    subset, project_root=Path("/isolated"), contracts=subcontracts, plan=subplan,
                    evaluation_end="2025-12-22", cutoff=str(settings.end_date))
            self.assertEqual(len(certified), len(selected))

    def test_cross_target_cohort_rejects_different_keys_in_a_price_alias(self):
        definition = _forty_only(_ACTIVE_ORACLE_DEFINITION)
        settings, _, _ = oracle._contract(definition)
        plan = oracle._oracle_comparison_plan(definition)
        contracts = oracle._resolve_oracle_target_score_contracts(definition, settings, plan)
        row = ("2330", "2021-01-04")
        prices = {name: {row: 0.5} for name in plan["recipes"]}
        for alias, source in plan["aliases"].items():
            if source in prices:
                prices[alias] = prices[source]
        daily = {1: {"DailyUniversal_MinTrade40_NetR": {row: 0.9}}}
        prices["R40_Direct"] = {("2317", "2021-01-04"): 0.5}
        from tools.validate.synthetic_daily_trade40_oracle_cases import execute_synthetic_score_fixture
        replayed = []
        with self.assertRaisesRegex(oracle.OracleBlockedError, "producer score eligibility differs"):
            execute_synthetic_score_fixture(
                definition, price_scores=prices, daily_scores=daily, keys={row}, replayed=replayed)
        self.assertFalse(any(name.startswith("Oracle R40_Direct ") for name in replayed))

    def test_mixed_incompatible_mature_cohorts_block_before_cache_and_replay(self):
        from services.audit import daily_trade40_oracle as daily_owner
        from services.audit.daily_trade40_oracle import TARGET_ID, LABEL_ID
        from filters.breakout_quality import daily_trade40_target as trade_owner
        definition = _forty_only(_ACTIVE_ORACLE_DEFINITION)
        # This test isolates cohort membership, not the fixed evaluation-end
        # boundary. Supply a matching synthetic single-day evaluation contract.
        definition = replace(definition, dimensions={
            **definition.dimensions, "oracle_evaluation_end_date": "2021-01-04",
        })
        seed, param_sha = 693545351, "a" * 64
        plan = oracle._oracle_comparison_plan(definition)
        price_row, daily_row = ("2330", "2021-01-04"), ("2317", "2021-01-04")
        scores = {name: {price_row: .7} for name in plan["recipes"]}
        for alias, source in plan["aliases"].items():
            if source in scores:
                scores[alias] = scores[source]
        evidence = {
            "teacher": {"seed": seed, "param_identity_sha256": param_sha},
            "profile_id": definition.source["daily_trade40_profile"],
            "target_id": TARGET_ID, "label_id": LABEL_ID,
            "first_date": "2021-01-04", "first_replay_trade_date": "2021-01-04",
            "mature_eligible_rows": 1, "settled_net_r_rows": 1,
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
             patch.object(oracle, "_robustness_artifacts", return_value=(
                 SimpleNamespace(benchmark_id="end_to_end_v1"), [(seed, Path(tmp)/"params.json")]
             )), \
             patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                 [(seed, param_sha, "rolling_active_param_ensemble", {}, {})], []
             )), \
             patch.object(trade_owner, "resolve_formal_min_param_records", return_value=()), \
             patch.object(trade_owner, "build_formal_min_execution_schedule", return_value=((
                 SimpleNamespace(start=pd.Timestamp("2021-01-01"), end=None, params_signature="fixture"),), {})), \
             patch.object(daily_owner, "build_daily_universal_trade40_oracle_scores",
                          return_value=({daily_row: .8}, evidence)), \
             patch.object(oracle, "_build_oracle_score_panels", return_value=(
                 scores, {"cohort_rows": 1, "last_common_day": "2025-12-22"}
             )), \
             patch.object(oracle, "price_inventory_sha256", return_value="f" * 64) as inventory, \
             patch.object(oracle, "load_profile_continuous_ranker_data", side_effect=lambda **kwargs:
                 self._feature_bundle({daily_row} if kwargs["experiment_profile"] ==
                    definition.source["daily_trade40_profile"] else {price_row})), \
             patch.object(oracle, "_run_scenario", side_effect=AssertionError("replay reached")) as replay:
            with self.assertRaisesRegex(oracle.OracleBlockedError, "different certified stock-day populations"):
                oracle.build_report(definition, project_root=Path(tmp), quiet=True)
        inventory.assert_called_once()
        replay.assert_not_called()

    def test_inventory_is_not_blocked_by_untrained_target_but_remains_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            status = oracle.collect_status(_ACTIVE_ORACLE_DEFINITION, project_root=Path(directory))
            self.assertEqual(status["status"], "BLOCKED")
            self.assertIn("價格資料", status["reason"])
            self.assertNotIn("CONTRACT_UNRESOLVED", status["reason"])

    def test_daily_only_exact_direct_build_plan_does_not_need_price_target_providers(self):
        from services.audit.daily_trade40_oracle import TARGET_ID
        from services.audit import daily_trade40_oracle as daily_owner
        from filters.breakout_quality import daily_trade40_target as trade_owner
        definition = replace(_ACTIVE_ORACLE_DEFINITION,
            dimensions={**_ACTIVE_ORACLE_DEFINITION.dimensions,
                        "oracle_evaluation_end_date": "2025-12-22"},
            source={
            **_ACTIVE_ORACLE_DEFINITION.source,
            "oracle_comparisons": tuple(
                item for item in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                if item["target"] in {
                    "DailyUniversal_MinTrade40_NetR",
                    "DailyUniversal_MinTrade40_NetR_Direct",
                }
            ),
        })
        plan = oracle._oracle_comparison_plan(definition)
        self.assertEqual(len(plan["targets"]), 2)
        self.assertFalse(plan["providers"])
        self.assertEqual(len(plan["daily_recipes"]), 1)
        seed, param_sha = 693545351, "b" * 64
        source = "DailyUniversal_MinTrade40_NetR"
        truth = {("2330", "2021-01-04"): 0.9, ("2317", "2021-01-04"): 0.1,
                 ("2330", "2025-12-22"): 0.5}
        evidence = {
            "label_id": "LABEL-" + TARGET_ID, "target_id": TARGET_ID,
            "profile_id": definition.source["daily_trade40_profile"],
            "teacher": {"seed": seed, "param_identity_sha256": param_sha},
            "mature_eligible_rows": len(truth), "settled_net_r_rows": len(truth),
            "first_date": "2021-01-04", "last_date": "2025-12-22",
            "first_replay_trade_date": "2021-01-04", "score_start_date": "2021-01-01",
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
             patch.object(oracle, "_robustness_artifacts", return_value=(
                 SimpleNamespace(benchmark_id="end_to_end_v1"),
                 [(seed, Path(directory) / "P16.json")],
             )), \
             patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                 [(seed, param_sha, "rolling_active_param_ensemble", {}, {})], []
             )), \
             patch.object(trade_owner, "resolve_formal_min_param_records", return_value=()), \
             patch.object(trade_owner, "build_formal_min_execution_schedule", return_value=((
                 SimpleNamespace(start=pd.Timestamp("2021-01-01"), end=None, params_signature="fixture"),), {})), \
             patch.object(daily_owner, "build_daily_universal_trade40_oracle_scores", return_value=(truth, evidence)) as daily_builder, \
             patch.object(oracle, "_build_oracle_score_panels", side_effect=AssertionError("price truth is unused")) as price_builder, \
             patch.object(oracle, "_read_only_score_keys", side_effect=AssertionError("cancelled source proof")) as eligibility, \
             patch.object(oracle, "price_inventory_sha256", return_value="f" * 64), \
             patch.object(oracle, "_run_scenario", side_effect=AssertionError("cancelled Replay")):
            result = oracle.build_report(definition, project_root=Path(directory), quiet=True,
                                         preflight_approval=lambda _: False)
        self.assertEqual(result["status"], "CANCELLED")
        self.assertEqual(set(result["preflight"]), {
            (seed, "C96"), (seed, source), (seed, source + "_Direct")})
        price_builder.assert_not_called()
        daily_builder.assert_not_called()
        eligibility.assert_not_called()

    def test_daily_only_full_synthetic_replay_shares_score_and_uses_c96_window(self):
        from services.audit.daily_trade40_oracle import TARGET_ID
        from services.audit import daily_trade40_oracle as daily_owner
        from filters.breakout_quality import daily_trade40_target as trade_owner
        definition = replace(_forty_only(_ACTIVE_ORACLE_DEFINITION), source={
            **_ACTIVE_ORACLE_DEFINITION.source,
            "oracle_comparisons": tuple(item for item in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                                        if "MinTrade40" in item["target"]),
        })
        seed, param_sha = 104504529, "c" * 64
        source = "DailyUniversal_MinTrade40_NetR"
        direct = "DailyUniversal_MinTrade40_NetR_Direct"
        truth = {("2330", "2021-01-04"): 0.9, ("2317", "2021-01-04"): 0.1,
                 ("2330", "2025-12-22"): 0.6}
        evidence = {
            "label_id": "LABEL-" + TARGET_ID, "target_id": TARGET_ID,
            "profile_id": definition.source["daily_trade40_profile"],
            "teacher": {"origin": "formal_min_roos_rolling_per_seed", "seed": seed,
                        "param_identity_sha256": param_sha, "periods": [{"start": "2021-01-01",
                            "end": None, "params_signature": "fixture"}]},
            "rank_policy": "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
            "truth_source_sha256": daily_owner.truth_producer_fingerprint(),
            "dataset_summary_sha256": None,
            "rank_sha256": "1"*64, "raw_label_sha256": "2"*64, "calendar_sha256": "3"*64,
            "mature_eligible_rows": len(truth), "settled_net_r_rows": len(truth),
            "censored_rank_last_rows": 0, "research_cutoff": "2026-03-02",
            "first_date": "2021-01-04", "last_date": "2025-12-22",
            "first_replay_trade_date": "2021-01-04", "score_start_date": "2021-01-01",
        }
        observed = []
        def fake_scenario(*, name, **kwargs):
            observed.append((name, str(kwargs["start_date"]), str(kwargs["end_date"])))
            return {"name": name}
        def fake_metrics(result, *, target):
            return {"target": target, "total_return_pct": 40., "annualized_return_pct": 8.,
                    "mdd_pct": 6., "romd": {"C96": 8., source: 12., direct: 15.}[target],
                    "ev_a_r": .6, "trades": 9, "exposure_pct": 30., "win_rate_pct": 56.0,
                    "yearly": _synthetic_full_window_yearly()}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
             patch.object(oracle, "_robustness_artifacts", return_value=(
                 SimpleNamespace(benchmark_id="end_to_end_v1"), [(seed, Path(directory)/"params.json")],
             )), \
             patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                 [(seed, param_sha, "rolling_active_param_ensemble", "baseline", "oracle")],
                 [{"seed": seed, "param_sha256": param_sha, "param_path": "params.json"}],
             )), \
             patch.object(trade_owner, "resolve_formal_min_param_records", return_value=()), \
             patch.object(trade_owner, "build_formal_min_execution_schedule", return_value=((
                 SimpleNamespace(start=pd.Timestamp("2021-01-01"), end=None, params_signature="fixture"),), {})), \
             patch.object(daily_owner, "build_daily_universal_trade40_oracle_scores", return_value=(truth, evidence)), \
             patch.object(oracle, "load_profile_continuous_ranker_data", return_value=self._feature_bundle(truth)), \
             patch.object(oracle, "price_inventory_sha256", return_value="f" * 64), \
             patch.object(oracle, "_build_oracle_score_panels", side_effect=AssertionError("unused price provider")) as unused, \
             patch.object(oracle, "_run_scenario", side_effect=fake_scenario), \
             patch.object(oracle, "_metrics", side_effect=fake_metrics):
            result = oracle.build_report(definition, project_root=Path(directory), quiet=True)
        unused.assert_not_called()
        self.assertEqual([row["target"] for row in result["oracles"]], [source, direct])
        self.assertEqual(result["common_cohort"]["cohort_rows"], len(truth))
        self.assertEqual([result["baseline"]["romd"],
                          result["oracles"][0]["romd"], result["oracles"][1]["romd"]],
                         [8., 12., 15.])
        self.assertEqual(len(observed), 3)
        self.assertTrue(all(start == "2021-01-01" and end == "2025-12-22"
                            for _name, start, end in observed))

    def test_direct_only_daily_replays_only_c96_and_selected_direct(self):
        from services.audit import daily_trade40_oracle as daily_owner
        from filters.breakout_quality import daily_trade40_target as trade_owner
        d = _ACTIVE_ORACLE_DEFINITION
        direct = "DailyUniversal_MinTrade40_NetR_Direct"
        exact = "DailyUniversal_MinTrade40_NetR"
        definition = replace(d, dimensions={**d.dimensions,
            "oracle_evaluation_end_date": "2025-12-22"},
            source={**d.source, "oracle_comparisons": tuple(
            row for row in d.source["oracle_comparisons"] if row["target"] == direct)})
        seed, param_sha = 104504529, "c" * 64
        scores = {("2330", "2021-01-04"): 0.9,
                  ("2317", "2021-01-04"): 0.1,
                  ("2330", "2025-12-22"): 0.6}
        evidence = {
            "label_id": daily_owner.LABEL_ID, "target_id": daily_owner.TARGET_ID,
            "profile_id": definition.source["daily_trade40_profile"],
            "teacher": {"seed": seed, "param_identity_sha256": param_sha,
                        "periods": [{"start": "2021-01-01", "end": None, "params_signature": "fixture"}]},
            "rank_policy": "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
            "truth_source_sha256": daily_owner.truth_producer_fingerprint(),
            "dataset_summary_sha256": None,
            "rank_sha256": "1"*64, "raw_label_sha256": "2"*64, "calendar_sha256": "3"*64,
            "mature_eligible_rows": len(scores), "settled_net_r_rows": len(scores),
            "censored_rank_last_rows": 0, "research_cutoff": "2026-03-02",
            "first_date": "2021-01-04", "last_date": "2025-12-22",
            "first_replay_trade_date": "2021-01-04", "score_start_date": "2021-01-01",
        }
        called = []
        def replay(*, name, **kwargs):
            called.append(name)
            return {"name": name}
        def metrics(result, *, target):
            return {"target": target, "total_return_pct": 40., "annualized_return_pct": 8.,
                    "mdd_pct": 6., "romd": {"C96": 8., direct: 15.}[target],
                    "ev_a_r": .6, "trades": 9, "exposure_pct": 30., "win_rate_pct": 56.0,
                    "yearly": _synthetic_full_window_yearly()}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(oracle, "collect_status", return_value={"status":"READY"}), \
             patch.object(oracle, "_robustness_artifacts", return_value=(
                 SimpleNamespace(benchmark_id="end_to_end_v1"),
                 [(seed, Path(directory)/"params.json")])), \
             patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                 [(seed, param_sha, "rolling_active_param_ensemble", "baseline", "oracle")], [])), \
             patch.object(trade_owner, "resolve_formal_min_param_records", return_value=()), \
             patch.object(trade_owner, "build_formal_min_execution_schedule", return_value=((
                 SimpleNamespace(start=pd.Timestamp("2021-01-01"), end=None, params_signature="fixture"),), {})), \
             patch.object(daily_owner, "build_daily_universal_trade40_oracle_scores", return_value=(scores, evidence)), \
             patch.object(oracle, "load_profile_continuous_ranker_data", return_value=self._feature_bundle(scores)), \
             patch.object(oracle, "_build_oracle_score_panels", side_effect=AssertionError("unused price")), \
             patch.object(oracle, "price_inventory_sha256", return_value="f"*64), \
             patch.object(oracle, "_run_scenario", side_effect=replay), \
             patch.object(oracle, "_metrics", side_effect=metrics):
            result = oracle.build_report(definition, project_root=Path(directory), quiet=True)
            again = oracle.build_report(definition, project_root=Path(directory), quiet=True)
        self.assertEqual([row["target"] for row in result["oracles"]], [direct])
        self.assertEqual([row["target"] for row in again["oracles"]], [direct])
        self.assertEqual(len(called), 2)
        self.assertTrue(called[0].startswith("C96"))
        self.assertIn(direct, called[1])
        self.assertNotIn("Oracle " + exact, called)

    def test_changed_formal_min_parameters_between_truth_and_replay_block(self):
        """Different P16 snapshots must never pair daily truth with a new Replay."""
        from services.audit import daily_trade40_oracle as daily_owner
        from filters.breakout_quality import daily_trade40_target as trade_owner
        definition = replace(_forty_only(_ACTIVE_ORACLE_DEFINITION), source={
            **_ACTIVE_ORACLE_DEFINITION.source,
            "oracle_comparisons": tuple(
                item for item in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                if "MinTrade40" in item["target"]),
        })
        seed, truth_param_sha, changed_param_sha = 693545351, "a" * 64, "b" * 64
        truth = {("2330", "2021-01-04"): 0.8, ("2317", "2021-01-04"): 0.2,
                 ("2330", "2025-12-22"): 0.5}
        evidence = {
            "label_id": daily_owner.LABEL_ID, "target_id": daily_owner.TARGET_ID,
            "profile_id": definition.source["daily_trade40_profile"],
            "teacher": {"seed": seed, "param_identity_sha256": truth_param_sha},
            "mature_eligible_rows": len(truth), "settled_net_r_rows": len(truth),
            "first_date": "2021-01-04", "last_date": "2025-12-22",
            "first_replay_trade_date": "2021-01-04", "score_start_date": "2021-01-01",
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
             patch.object(oracle, "_robustness_artifacts", return_value=(
                 SimpleNamespace(benchmark_id="end_to_end_v1"),
                 [(seed, Path(directory) / "P16.json")],
             )), \
             patch.object(oracle, "_prepare_oracle_parameters", side_effect=[
                 ([(seed, truth_param_sha, "rolling_active_param_ensemble", {}, {})], []),
                 ([(seed, changed_param_sha, "rolling_active_param_ensemble", {}, {})], []),
             ]), \
             patch.object(trade_owner, "resolve_formal_min_param_records", return_value=()), \
             patch.object(trade_owner, "build_formal_min_execution_schedule", return_value=((
                 SimpleNamespace(start=pd.Timestamp("2021-01-01"), end=None, params_signature="fixture"),), {})), \
             patch.object(daily_owner, "build_daily_universal_trade40_oracle_scores",
                          return_value=(truth, evidence)) as source, \
             patch.object(oracle, "load_profile_continuous_ranker_data", return_value=self._feature_bundle(truth)), \
             patch.object(oracle, "price_inventory_sha256", return_value="c" * 64), \
             patch.object(oracle, "_run_scenario", side_effect=AssertionError("replay must not run")) as replay:
            with self.assertRaisesRegex(oracle.OracleBlockedError, "P16 inputs changed"):
                oracle.build_report(definition, project_root=Path(directory), quiet=True)
            source.assert_not_called()
            replay.assert_not_called()

    def test_retired_daily_target_identity_is_rejected_before_market_replay(self):
        from tools.validate.synthetic_daily_trade40_oracle_cases import execute_synthetic_score_fixture
        source = "DailyUniversal_MinTrade40_NetR"
        definition = replace(_forty_only(_ACTIVE_ORACLE_DEFINITION), source={
            **_ACTIVE_ORACLE_DEFINITION.source,
            "oracle_comparisons": tuple(item for item in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                                        if item["target"] in {source, source + "_Direct"}),
        })
        seed = 693545351
        key = ("2330", "2021-01-04")
        # Keep all other provenance valid so this cannot pass accidentally
        # because an earlier fixture field is missing. The existing C96 is
        # paired; no incorrect scored Replay is permitted to start.
        for wrong in (
            {"label_id": "LABEL-daily_universal_counterfactual_mintrade40_netr_v1",
             "target_id": "daily_universal_counterfactual_mintrade40_netr_v1"},
            {"profile_id": "wrong_registered_source"},
        ):
            replayed = []
            with self.subTest(wrong=wrong), self.assertRaisesRegex(
                    oracle.OracleBlockedError, "(not based on this formal Min|Target.*identity|Target.*profile)"):
                execute_synthetic_score_fixture(
                    definition, price_scores={}, daily_scores={seed: {source: {key: .7}}},
                    keys={key}, replayed=replayed,
                    evidence_overrides=lambda _seed, _name, _evidence: wrong)
            self.assertEqual(replayed, [])

    def test_daily_cold_truth_rejects_another_seed_or_p16_snapshot(self):
        from tools.validate.synthetic_daily_trade40_oracle_cases import execute_synthetic_score_fixture
        source = "DailyUniversal_MinTrade40_NetR"
        definition = replace(_forty_only(_ACTIVE_ORACLE_DEFINITION), source={
            **_ACTIVE_ORACLE_DEFINITION.source,
            "oracle_comparisons": tuple(row for row in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                                        if row["target"] == source)})
        seed, key = 693545351, ("2330", "2021-01-04")
        for change in ({"seed": seed + 1}, {"param_identity_sha256": "b" * 64}):
            replayed = []
            with self.subTest(change=change), self.assertRaisesRegex(
                    oracle.OracleBlockedError, "Target and P16 teacher differ"):
                execute_synthetic_score_fixture(
                    definition, price_scores={}, daily_scores={seed: {source: {key: .7}}},
                    keys={key}, replayed=replayed,
                    evidence_overrides=lambda _seed, _name, evidence: {
                        "teacher": {**evidence["teacher"], **change}})
            self.assertEqual(replayed, [])

    def test_current_selector_is_from_enabled_research_arm(self):
        settings, base, exact = oracle._contract(_ACTIVE_ORACLE_DEFINITION)
        self.assertTrue(exact.enabled)
        self.assertIn(exact.arm_id, {arm.arm_id for arm in settings.enabled_arms})
        self.assertNotEqual(exact.arm_id, "C100")


class OracleCanonicalPriceLabelCases(TestCase):
    """Oracle scalar scores must preserve the canonical label's exact ties."""

    def _score_paths(self, paths, selected):
        from filters.breakout_quality.continuous_target import StrategyAlignedContinuousTargetSpec
        from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch

        spec = StrategyAlignedContinuousTargetSpec.from_label_policy(oracle.DEFAULT_LABEL_POLICY)
        dates = pd.bdate_range("2025-01-02", periods=spec.horizon_bars + 1)
        frames, records = {}, []
        for ticker, anchor, terminal, peak, minimum in paths:
            close = np.full(len(dates), anchor, dtype=np.float64)
            close[-1] = terminal
            frame = pd.DataFrame({"Close": close, "High": np.maximum(close, peak),
                                  "Low": np.full(len(dates), minimum, dtype=np.float64)}, index=dates)
            frames[ticker] = frame
            batch = compute_daily_opportunity_target_batch(
                frame, np.array([0]), spec=spec, target_id="daily_full_horizon_pure_mfe_r_v1")
            records.append({"ticker": ticker, "date": str(dates[0].date()),
                            "mfe": float((batch.favorable_return / spec.risk_budget_return)[0]),
                            "peak_adverse": float((batch.adverse_return_to_peak / spec.risk_budget_return)[0]),
                            "min_low": float(batch.minimum_low_return[0]),
                            "raw": float(batch.target_raw_r[0]), "_oracle_provider_mature": True})
        official = pd.DataFrame(records)
        definition = replace(_ACTIVE_ORACLE_DEFINITION, source={
            **_ACTIVE_ORACLE_DEFINITION.source, "oracle_comparisons": tuple(
                item for item in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                if item["target"] in selected)})
        inputs = [(ticker, Path(ticker + ".csv")) for ticker in ("0050", *frames)]
        with patch.object(oracle, "_load_truth", return_value=official), \
             patch.object(oracle, "discover_dataset_csv_inputs", return_value=(inputs, [])), \
             patch.object(oracle, "load_dataset_frame", side_effect=lambda path, ticker, **kw: (
                 next(iter(frames.values())) if ticker == "0050" else frames[ticker])):
            scores, _ = oracle._build_oracle_score_panels(
                definition, project_root=ROOT, cutoff="2026-03-02")
        return scores, official, frames, spec

    def test_scalar_price_scores_preserve_canonical_raw_label_ties(self):
        from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
        from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets

        cases = (
            ("R40", "daily_terminal_r40_v1", (
                ("2330", 100., 115., 115., 95.), ("2317", 10.1, 11.615, 11.615, 9.595))),
            ("T40", "daily_terminal_mae_gated_t40_v1", (
                ("2330", 100., 115., 115., 95.), ("2317", 160., 169., 169., 156.8),
                ("2454", 100., 102.5, 102.5, 100.))),
            ("T40-MFE", "daily_full_horizon_mae_gated_mfe_t40_v1", (
                ("2330", 100., 115., 115., 95.), ("2317", 160., 169., 169., 156.8),
                ("2454", 100., 102.5, 102.5, 100.))),
            ("MFE40_Control", "daily_full_horizon_pure_mfe_r_v1", (
                ("2330", 100., 105., 137.1463013634724, 99.),
                ("2317", 100., 105., 137.1463023634724, 99.))),
        )
        for name, target_id, paths in cases:
            direct = "MFE40_Direct" if name == "MFE40_Control" else name + "_Direct"
            with self.subTest(target=name):
                scores, official, frames, spec = self._score_paths(paths, {name, direct})
                raw = np.array([compute_daily_opportunity_target_batch(
                    frames[ticker], np.array([0]), spec=spec, target_id=target_id
                ).target_raw_r[0] for ticker, *_ in paths])
                np.testing.assert_array_equal(raw, np.full(len(paths), raw[0]))
                expected = build_same_date_percentile_targets(
                    raw, np.ones(len(raw), dtype=bool), official["date"])
                actual = np.array([scores[name][(ticker, "2025-01-02")] for ticker, *_ in paths])
                np.testing.assert_array_equal(actual, expected)
                self.assertIs(scores[name], scores[direct])

    def test_cg_proxy_retains_approved_component_rank_source(self):
        from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets

        paths = (("2330", 100., 105., 137.1463013634724, 99.),
                 ("2317", 100., 105., 137.1463023634724, 99.),
                 ("2454", 100., 105., 140., 90.))
        scores, official, _frames, spec = self._score_paths(paths, {"CG40_EqualRank"})
        rank = lambda values: build_same_date_percentile_targets(
            np.asarray(values), np.ones(len(paths), dtype=bool), official["date"])
        safety = oracle.full_horizon_mae_safety_r_from_minimum_low_return(
            official["min_low"].to_numpy(), risk_budget_return=spec.risk_budget_return)
        expected = rank((rank(official["mfe"]) + rank(safety)) / 2)
        actual = [scores["CG40_EqualRank"][(ticker, "2025-01-02")] for ticker, *_ in paths]
        np.testing.assert_array_equal(actual, expected)

    def test_only_selected_scalar_labels_are_materialized(self):
        original = oracle.compute_daily_opportunity_target_batch
        paths = (("2330", 100., 115., 115., 95.),)
        for selected, expected_targets in (
            ({"MFE40_Control"}, ["daily_full_horizon_pure_mfe_r_v1"]),
            ({"R40_Direct"}, ["daily_full_horizon_pure_mfe_r_v1", "daily_terminal_r40_v1"]),
            ({"CG40_EqualRank"}, ["daily_full_horizon_pure_mfe_r_v1"]),
        ):
            with self.subTest(selected=selected), patch.object(
                oracle, "compute_daily_opportunity_target_batch", wraps=original
            ) as producer:
                self._score_paths(paths, selected)
                self.assertEqual([call.kwargs["target_id"] for call in producer.call_args_list],
                                 expected_targets)

    def test_cg_inference_tail_keeps_formal_component_float32_ties(self):
        # Distinct float32 favorable returns can have the SAME formal
        # float32 favorable/R component. Float64 division would split it.
        tickers = ("2330", "2317")
        favorable = np.array([0.8661496043205261, 0.8661496639251709], dtype=np.float32)
        dates = pd.to_datetime(["2026-02-27", "2026-03-02", "2026-03-03"])
        frames = {ticker: pd.DataFrame({
            "Close": [100., 105., 900.],
            "High": [101., 100. * (1 + float(value)), 1000.],
            "Low": [99., 99., 850.],
        }, index=dates) for ticker, value in zip(tickers, favorable)}
        official = pd.DataFrame({"ticker": tickers, "date": ["2026-02-27"] * 2,
                                 "mfe": np.nan, "peak_adverse": np.nan, "min_low": np.nan,
                                 "raw": np.nan, "_oracle_provider_mature": False})
        definition = replace(_ACTIVE_ORACLE_DEFINITION, source={
            **_ACTIVE_ORACLE_DEFINITION.source, "oracle_comparisons": tuple(
                item for item in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                if item["target"] == "CG40_EqualRank")})
        inputs = [(ticker, Path(ticker + ".csv")) for ticker in ("0050", *tickers)]
        with patch.object(oracle, "_load_truth", return_value=official), \
             patch.object(oracle, "discover_dataset_csv_inputs", return_value=(inputs, [])), \
             patch.object(oracle, "load_dataset_frame", side_effect=lambda path, ticker, **kw: (
                 frames["2330"] if ticker == "0050" else frames[ticker])):
            scores, evidence = oracle._build_oracle_score_panels(
                definition, project_root=ROOT, cutoff="2026-03-02")
        expected_components = favorable / 0.1
        self.assertEqual(expected_components[0], expected_components[1])
        self.assertEqual(evidence["rows_outside_dl_training_provider_by_source"],
                         {"pure_mfe40_provider": 2})
        self.assertEqual(set(scores["CG40_EqualRank"].values()), {0.5})

    def test_mfe_scalar_preserves_first_breach_provider_path(self):
        from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch

        provider = "first_breach_provider"
        definition = replace(_ACTIVE_ORACLE_DEFINITION, source={
            **_ACTIVE_ORACLE_DEFINITION.source, "oracle_comparisons": ({
                "target": "MFE_FirstBreach_Test", "description": "isolated first-breach capability",
                "score": "mfe", "provider": provider, "selector": "exact"},)})
        target_id = oracle.get_continuous_ranker_execution_recipe(
            str(definition.source[provider])).target_policy.component_target_id
        spec = oracle.StrategyAlignedContinuousTargetSpec.from_label_policy(oracle.DEFAULT_LABEL_POLICY)
        dates = pd.bdate_range("2025-01-02", periods=spec.horizon_bars + 1)
        first = pd.DataFrame({"Close": 100., "High": 110., "Low": 95.}, index=dates)
        first.loc[dates[2], "Low"] = 85.
        first.loc[dates[3]:, "High"] = 200.  # Must be ignored after the first risk breach.
        second = pd.DataFrame({"Close": 100., "High": 115., "Low": 95.}, index=dates)
        frames, records = {"2330": first, "2317": second}, []
        for ticker, frame in frames.items():
            batch = compute_daily_opportunity_target_batch(
                frame, np.array([0]), spec=spec, target_id=target_id)
            records.append({"ticker": ticker, "date": "2025-01-02",
                            "mfe": float((batch.favorable_return / spec.risk_budget_return)[0]),
                            "peak_adverse": float((batch.adverse_return_to_peak / spec.risk_budget_return)[0]),
                            "min_low": float(batch.minimum_low_return[0]),
                            "raw": float(batch.target_raw_r[0])})
        official = pd.DataFrame(records)
        inputs = [(ticker, Path(ticker + ".csv")) for ticker in ("0050", *frames)]
        with patch.object(oracle, "_load_truth", return_value=official), \
             patch.object(oracle, "discover_dataset_csv_inputs", return_value=(inputs, [])), \
             patch.object(oracle, "load_dataset_frame", side_effect=lambda path, ticker, **kw: (
                 first if ticker == "0050" else frames[ticker])):
            scores, _ = oracle._build_oracle_score_panels(
                definition, project_root=ROOT, cutoff="2026-03-02")
        self.assertEqual(scores["MFE_FirstBreach_Test"],
                         {("2330", "2025-01-02"): 0., ("2317", "2025-01-02"): 1.})


class OracleAuditContractCases(TestCase):
    def setUp(self):
        # Replay tests replace price producers with synthetic fixtures. Pin the
        # real source identity before mocking callables; lineage drift has its
        # own unmocked-source regression cases below.
        source = patch.object(oracle, "_price_score_algorithm_sha256",
                              return_value=oracle._price_score_algorithm_sha256())
        source.start()
        self.addCleanup(source.stop)
        # These replay fixtures deliberately inject a complete deterministic
        # score contract; they must not mask the real contract-fail-closed tests.
        for attribute, side_effect in (
            ("_resolve_oracle_target_score_contracts", _fixture_perfect_score_contracts),
            ("_read_only_score_keys", _fixture_read_score_keys),
            ("_load_or_build_oracle_price_scores", _fixture_price_scores),
        ):
            patcher = patch.object(oracle, attribute, side_effect=side_effect)
            patcher.start()
            self.addCleanup(patcher.stop)
        prices = patch("services.audit.oracle_replay_artifacts.price_inventory_sha256",
                       side_effect=lambda *args: oracle.price_inventory_sha256(*args))
        prices.start()
        self.addCleanup(prices.stop)

    @classmethod
    def setUpClass(cls):
        cls.active_definition = next(
            definition for definition in get_reusable_audit_definitions("breakout_quality")
            if definition.audit_type == oracle.SUPPORTED_AUDIT_TYPE
        )
        cls.definition = _legacy_definition()
        cls.exact_policy = oracle._contract(cls.definition)[2].dl_runtime_mode
        cls.exact_options = oracle._contract(cls.definition)[2].dl_runtime_options

    def test_oracle_reusable_route_remains_registered_after_one_time_retirement(self):
        self.assertEqual(collect_audit_menu_state("breakout_quality")["enabled_count"], 0)
        from core.audit_policy import get_audit_definitions
        self.assertEqual(get_audit_definitions("breakout_quality"), ())
        self.assertEqual(collect_audit_menu_state("breakout_quality")["enabled_count"], 0)
        self.assertEqual(
            collect_audit_menu_state("breakout_quality", method_id=oracle.SUPPORTED_AUDIT_TYPE)["enabled_count"], 1
        )
        self.assertEqual(
            [x.method_id for x in get_audit_methods(menu_location="research_root")],
            [oracle.SUPPORTED_AUDIT_TYPE],
        )
        self.assertNotIn(oracle.SUPPORTED_AUDIT_TYPE,
                         [x.method_id for x in get_audit_methods(menu_location="audit")])
        self.assertFalse(any(
            definition.audit_type == oracle.SUPPORTED_AUDIT_TYPE
            for definition in get_audit_definitions("breakout_quality")
        ))
        self.assertEqual(
            {name for name, _ in LEGACY_ORACLE_TARGETS},
            {"CG40_EqualRank", "MFE40_Control", "R40", "T40",
             "T40-MFE", "MFE40_Direct", "R40_Direct", "T40_Direct", "T40-MFE_Direct"},
        )
        self.assertEqual(len(oracle._oracle_comparison_plan(self.active_definition)["targets"]), 16)
        self.assertNotIn("MinTrade40_NetR", dict(oracle._oracle_comparison_plan(self.active_definition)["targets"]))
        self.assertNotIn("MinTrade40_NetR_Direct", dict(oracle._oracle_comparison_plan(self.active_definition)["targets"]))
        self.assertEqual(oracle._oracle_comparison_plan(self.active_definition)["aliases"], {
            "MFE40_Direct": "MFE40_Control", "R40_Direct": "R40", "T40_Direct": "T40",
            "T40-MFE_Direct": "T40-MFE",
            "DailyUniversal_MinTrade40_NetR_Direct": "DailyUniversal_MinTrade40_NetR",
            "DailyUniversal_MinTrade_NetR_Direct": "DailyUniversal_MinTrade_NetR",
            "DailyUniversal_MinTrade_NetPnL_Direct": "DailyUniversal_MinTrade_NetPnL",
            "DailyUniversal_MinTrade_NetPnL_Positive_Direct": "DailyUniversal_MinTrade_NetPnL",
        })
        self.assertEqual(self.definition.output_subdir, "breakout_quality/reusable/target_oracle_romd")
        with tempfile.TemporaryDirectory() as tmp:
            status = collect_audit_status(
                "breakout_quality", project_root=Path(tmp), method_id=oracle.SUPPORTED_AUDIT_TYPE,
            )
            self.assertEqual(status["overall_status"], "BLOCKED")
            self.assertIn("full dataset", status["rows"][0]["reason"])

    def test_perfect_score_mapping_cannot_be_silently_redefined(self):
        # Exact compares numerical score sums. The original Oracle registry,
        # not a new per-Target SC, fixes the Target-to-score convention.
        from dataclasses import replace
        changed = replace(self.active_definition, dimensions={
            **self.active_definition.dimensions,
            "oracle_score_method": "raw_R_without_percentile",
        })
        with self.assertRaisesRegex(oracle.OracleBlockedError, "CONTRACT_UNRESOLVED"):
            oracle._oracle_comparison_plan(changed)
        changed = replace(self.active_definition, dimensions={
            **self.active_definition.dimensions,
            "percentile_method": "min_rank",
        })
        with self.assertRaisesRegex(oracle.OracleBlockedError, "CONTRACT_UNRESOLVED"):
            oracle._oracle_comparison_plan(changed)

    def test_oracle_cache_accepts_safe_hyphenated_target_without_path_escape(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), self.definition.output_subdir)
            identity = {"seed": 11, "target": "T40-MFE"}
            self.assertEqual(store._path(identity).parent.name, "11")
            self.assertTrue(store._path(identity).name.startswith("T40-MFE_"))
            direct = {**identity, "target": "T40-MFE_Direct"}
            self.assertTrue(store._path(direct).name.startswith("T40-MFE_Direct_"))
            self.assertNotEqual(store._path(identity), store._path(direct))
            for unsafe in ("../T40-MFE", "T40/MFE", "T40.MFE", "T40 MFE"):
                with self.assertRaisesRegex(ValueError, "invalid cache target"):
                    store._path({**identity, "target": unsafe})

    def test_summary_display_is_forwarded_only_to_declared_formal_oracle(self):
        from services.audit import runner
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner, "collect_audit_status", return_value={"overall_status": "READY"}), \
             patch.object(oracle, "run_formal_audit", return_value={"status": "COMPLETED"}) as execute:
            runner.run_enabled_audits(
                "breakout_quality", project_root=Path(tmp),
                method_id=oracle.SUPPORTED_AUDIT_TYPE, preflight_display="summary")
            self.assertEqual(execute.call_args.kwargs["preflight_display"], "summary")
            with self.assertRaisesRegex(ValueError, "簡要 Replay preflight"):
                runner.run_enabled_audits(
                    "breakout_quality", project_root=Path(tmp),
                    method_id="opportunity_selection_attribution", preflight_display="summary")


    def test_formal_runner_never_prompts_or_starts_optimizer_for_audit(self):
        from services.audit import runner
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner, "collect_audit_status", return_value={"overall_status": "READY"}), \
             patch("builtins.input", side_effect=AssertionError("second confirmation")) as user_input, \
             patch("services.research.strategy_compare_preparation.prepare_strategy_parameter_artifacts",
                   side_effect=AssertionError("unexpected Optimizer")) as optimizer, \
             patch.object(oracle, "run_formal_audit", return_value={"status": "COMPLETED"}) as execute:
            result = runner.run_enabled_audits(
                "breakout_quality", project_root=Path(tmp),
                method_id=oracle.SUPPORTED_AUDIT_TYPE, preflight_display="summary",
            )
            self.assertEqual(result["target_oracle_romd"]["status"], "COMPLETED")
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(execute.call_args.kwargs["preflight_display"], "summary")
            self.assertNotIn("before_replay", execute.call_args.kwargs)
            user_input.assert_not_called()
            optimizer.assert_not_called()

    def test_preflight_action_matrix_and_canonical_best_worst_metric_colors(self):
        from core.research_report_contract import report_contract
        from services.audit.reusable_report import format_comparison_rows
        from core.report_style import best_worst_signals
        from core.display_common import _strip_ansi
        self.assertEqual(best_worst_signals({"a": 2, "b": 2}, preference="higher"), {})
        perf = report_contract("audit.target_oracle_romd").sections[0].tables[0]
        values = [
            {"target": "X", "romd": 5.0, "mdd_pct": 20.0, "total_return_pct": 80},
            {"target": "Y", "romd": 9.0, "mdd_pct": 8.0, "total_return_pct": 120},
            {"target": "Z", "romd": 7.0, "mdd_pct": 12.0, "total_return_pct": 100},
        ]
        cells = format_comparison_rows(perf, values, target="markdown")
        columns = {col.key: i for i, col in enumerate(perf.columns)}
        self.assertIn("#188038", cells[1][columns["romd"]])
        self.assertIn("#C62828", cells[0][columns["romd"]])
        self.assertIn("#188038", cells[1][columns["mdd_pct"]])
        self.assertIn("#C62828", cells[0][columns["mdd_pct"]])
        self.assertNotIn("<span", cells[2][columns["romd"]])
        self.assertNotIn("<span", cells[0][columns["target"]])
        # Year-group values must not be compared to results from other years.
        grouped = format_comparison_rows(perf, values, target="markdown", groups=["2022", "2022", "2023"])
        self.assertNotIn("<span", grouped[2][columns["romd"]])
        plan = oracle.render_replay_preflight(
            label="40D", seeds=[11, 22], targets=["C96", "R40", "T40"],
            decisions={
                (11, "C96"): "REUSE", (22, "C96"): "REUSE",
                (11, "R40"): "RUN", (22, "R40"): "REUSE",
                (11, "T40"): "BLOCKED", (22, "T40"): "RUN",
            },
        )
        plain = _strip_ansi(plan)
        self.assertIn("S1=11", plain)
        self.assertIn("S2=22", plain)
        self.assertIn("Portfolio REUSE 3 / RUN（需回放）2 / BLOCKED 1", plain)
        self.assertLess(plain.index("C96"), plain.index("R40"))
        self.assertLess(plain.index("R40"), plain.index("T40"))

    def test_persistent_contract_must_be_approved_before_replay(self):
        with patch.object(oracle, "require_approved_report_contract", side_effect=ValueError("UNAUTHORIZED")), \
             patch.object(oracle, "build_report") as builder:
            with self.assertRaisesRegex(ValueError, "UNAUTHORIZED"):
                oracle.run_formal_audit(self.definition, project_root=ROOT, quiet=True)
            builder.assert_not_called()

    def test_future_source_cannot_be_registered_as_formal_pit_score(self):
        self.assertNotIn(SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH, SUPPORTED_RANKING_SCORE_SOURCES)
        with self.assertRaises(ValueError):
            with breakout_quality_ranking_source_context(score_source=SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH):
                self.fail("future truth must never enter normal score registry")

    def test_future_context_returns_exact_score_and_resets(self):
        before = get_breakout_quality_ranking_source_context()
        scores = {("2330", "2025-02-03"): 0.8125}
        with audit_oracle_truth_score_context(
            scores, ranking_policy=self.exact_policy, ranking_options=self.exact_options,
        ):
            rank = resolve_breakout_quality_candidate_rank(
                ticker="2330", signal_date="2025-02-03", information_date="2025-02-03", high_len=260,
            )
            self.assertEqual(rank["score"], 0.8125)
            self.assertEqual(rank["score_source"], SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)
            self.assertEqual(rank["score_date"], "2025-02-03")
            with self.assertRaisesRegex(ValueError, "未覆蓋"):
                resolve_breakout_quality_candidate_rank(
                    ticker="2317", signal_date="2025-02-03", information_date="2025-02-03", high_len=260,
                )
            with self.assertRaisesRegex(RuntimeError, "巢狀"):
                with audit_oracle_truth_score_context(
                    scores, ranking_policy=self.exact_policy, ranking_options=self.exact_options,
                ):
                    pass
        after = get_breakout_quality_ranking_source_context()
        self.assertEqual(after, before)

    def test_score_context_restores_after_replay_failure(self):
        before = get_breakout_quality_ranking_source_context()
        with self.assertRaisesRegex(RuntimeError, "forced abort"):
            with audit_oracle_truth_score_context(
                {("2330", "2025-02-03"): 0.5}, ranking_policy=self.exact_policy,
                ranking_options=self.exact_options,
            ):
                raise RuntimeError("forced abort")
        self.assertEqual(get_breakout_quality_ranking_source_context(), before)

    def test_direct_selector_is_same_canonical_c101_and_audit_only(self):
        from core.breakout_quality_ranking_context import audit_oracle_ranking_policy_context
        settings, base, exact = oracle._contract(self.definition)
        direct = oracle._direct_arm(self.definition, settings, base, exact)
        self.assertEqual(direct.dl_runtime_mode, "resource-aware-continuous-score-no-k-no-r0")
        self.assertIs(direct.dl_runtime_options["preserve_k"], False)
        self.assertIs(direct.dl_runtime_options["preserve_r0"], False)
        with audit_oracle_truth_score_context(
            {("2330", "2025-01-02"): 0.8}, ranking_policy=direct.dl_runtime_mode,
            ranking_options=direct.dl_runtime_options,
        ):
            state = get_breakout_quality_ranking_source_context()
            self.assertEqual(state.score_source, SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)
            self.assertEqual(state.ranking_policy, direct.dl_runtime_mode)
        with self.assertRaisesRegex(ValueError, "只允許"):
            with audit_oracle_ranking_policy_context(ranking_policy="score"):
                pass
        self.assertNotIn(SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH, SUPPORTED_RANKING_SCORE_SOURCES)

    def _truth(self, provider: str) -> pd.DataFrame:
        values = {
            "pure_mfe40_provider": {
                "mfe": [3.0, 2.0, 1.0], "peak_adverse": [0.8, 0.1, 0.2],
                "min_low": [-0.1, -0.3, -0.2], "raw": [3.0, 2.0, 1.0],
            },
            "mfe20_d20_provider": {
                "mfe": [1.0, 3.0, 2.0], "peak_adverse": [0.1, 0.2, 0.3],
                "min_low": [-0.2, -0.1, -0.3], "raw": [1.0, 3.0, 2.0],
            },
            "first_breach_provider": {
                "mfe": [0.9, 0.6, 0.4], "peak_adverse": [0.1, 0.5, 0.05],
                "min_low": [-0.1, -0.1, -0.1], "raw": [0.9, 0.6, 0.4],
            },
        }[provider]
        return pd.DataFrame({"ticker": ["2330", "2317", "2454"],
                             "date": ["2025-01-02"] * 3, **values})

    def _mock_market(self, omit_first_breach_last: bool = False):
        market = {field: self._truth(field) for field in oracle._oracle_comparison_plan(self.definition)["providers"]}
        market["pure_mfe40_provider"] = market["pure_mfe40_provider"].assign(
            scalar_raw_mfe=[3.0, 2.0, 1.0], scalar_raw_terminal_r=[1.5, 2.0, 0.5],
            scalar_raw_terminal_mae_gate=[-1.0, -1.0, -1.0],
            scalar_raw_mfe_mae_gate=[-1.0, -1.0, -1.0])
        if omit_first_breach_last:
            market["pure_mfe40_provider"] = market["pure_mfe40_provider"].iloc[:2]
        return market, {"score_start": "2025-01-01", "truth_source": "test"}

    def test_oracle_target_scores_use_matched_cohort_and_correct_directions(self):
        with patch.object(oracle, "_load_truth", side_effect=lambda definition, root, source_field, cutoff: self._truth(source_field)), \
             patch.object(oracle, "_market_truth_for_oracle", return_value=self._mock_market()):
            scores, evidence = oracle._build_oracle_score_panels(
                self.definition, project_root=ROOT, cutoff="2026-03-02",
            )
        self.assertEqual(evidence["cohort_rows"], 3)
        self.assertEqual(evidence["mfe40_over_d40_zero_downside_positive_upside_count"], 0)
        self.assertEqual(set(scores), {label for label, _ in LEGACY_ORACLE_TARGETS})
        key = lambda ticker: (ticker, "2025-01-02")
        self.assertEqual(scores["MFE40_Control"][key("2330")], 1.0)
        self.assertEqual(scores["CG40_EqualRank"][key("2330")], 1.0)
        self.assertNotIn("CH20_EqualRank", scores)
        self.assertNotIn("AO40_EqualRank", scores)
        self.assertNotIn("U40", scores)
        self.assertNotIn("FirstBreach_NetR", scores)
        self.assertEqual(scores["R40"][key("2317")], 1.0)
        self.assertEqual(scores["T40"][key("2330")], 0.5)  # All three reach MAE>=1R.
        for direct_name, exact_name in oracle._oracle_comparison_plan(self.definition)["aliases"].items():
            if exact_name in scores:
                self.assertIs(scores[direct_name], scores[exact_name], "Direct must reuse identical truth")
        self.assertTrue(all(0.0 <= v <= 1.0 for table in scores.values() for v in table.values()))

    def test_selected_score_providers_do_not_run_retired_truth_dependencies(self):
        market, evidence = self._mock_market()
        evidence.update({"price_ticker_count": 3, "unreadable_tickers": {}})
        with patch.object(oracle, "_load_truth", side_effect=lambda definition, root, source_field, cutoff: self._truth(source_field)) as loader, \
             patch.object(oracle, "_market_truth_for_oracle", return_value=(market, evidence)):
            scores40, cohort40 = oracle._build_oracle_score_panels(
                self.definition, project_root=ROOT, cutoff="2026-03-02",
            )
        self.assertEqual(cohort40["cohort_rows"], 3)
        self.assertEqual(set(cohort40["providers"]), {"40d"})
        self.assertEqual(loader.call_count, 1)
        self.assertFalse(any(name.endswith("60") or name.startswith("CH20") for name in scores40))
        self.assertEqual(len(scores40), len(LEGACY_ORACLE_TARGETS))

    def test_provider_dependencies_follow_selected_comparisons_not_hardcoded(self):
        from dataclasses import replace
        original = list(self.definition.source["oracle_comparisons"])
        mfe40 = next(item for item in original if item["target"] == "MFE40_Control")
        # Remove all other Exact and Direct arms; only 40D remains.
        subset = replace(self.definition, source={**self.definition.source,
                                                 "oracle_comparisons": (mfe40,)})
        p1 = oracle._oracle_comparison_plan(subset)
        self.assertEqual(p1["providers"], ("pure_mfe40_provider",))
        self.assertEqual([name for name, _ in p1["targets"]], ["MFE40_Control"])
        toggled = replace(self.definition, source={**self.definition.source,
            "oracle_comparisons": tuple({**item, "enabled": item["target"] == "MFE40_Control"}
                                        for item in original)})
        self.assertEqual(oracle._oracle_comparison_plan(toggled)["targets"], p1["targets"])
        # Add a test-only score referencing the catalogued 20D provider.
        # Same code path discovers the new dependency without a new allowlist.
        addition = {"target": "MFE20_Test", "description": "20D canonical test",
                    "score": "mfe", "provider": "mfe20_d20_provider", "selector": "exact"}
        extended = replace(self.definition, source={**self.definition.source,
                                                   "oracle_comparisons": (mfe40, addition)})
        p2 = oracle._oracle_comparison_plan(extended)
        self.assertEqual(p2["providers"], ("pure_mfe40_provider", "mfe20_d20_provider"))
        with patch.object(oracle, "_load_truth", side_effect=lambda d, root, field, cutoff: self._truth(field)) as loader, \
             patch.object(oracle, "_market_truth_for_oracle", return_value=({
                 **self._mock_market()[0], "mfe20_d20_provider": self._truth("mfe20_d20_provider").assign(
                     scalar_raw_mfe=[1.0, 3.0, 2.0])},
                 self._mock_market()[1])):
            scores, cohort = oracle._build_oracle_score_panels(extended, project_root=ROOT,
                                                              cutoff="2026-03-02")
        self.assertEqual(set(scores), {"MFE40_Control", "MFE20_Test"})
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(set(cohort["providers"]), {"40d", "20d"})
        self.assertNotIn("first_breach", cohort["providers"])

    def test_complete_verified_report_rechecks_source_without_recomputing_truth(self):
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"

        source = self.definition.source
        full = {
            "cohort_rows": 3, "first_common_day": "2021-01-04",
            "last_common_day": "2025-12-22", "research_cutoff": "2026-03-02",
            "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
            "risk_budget_return": 0.1,
            "providers": {"40d": source["pure_mfe40_provider"]},
            "scoring": "All targets final-ranked by average_zero_based percentile within common-day universe",
            "oracle_only": True,
            "market_truth": {"score_start": "2020-12-31", "mature_rows_by_provider": {
                "pure_mfe40_provider": 3}},
            "rows_outside_dl_training_provider_by_source": {"pure_mfe40_provider": 0},
        }
        outcomes = []
        def fake_replay(*, name, **kwargs):
            outcomes.append(name)
            return {"name": name}
        def fake_metrics(replay, *, target):
            seed = int(replay["name"].split("seed=")[1].split()[0])
            return {"target": target, "total_return_pct": 120.0,
                    "annualized_return_pct": 15.0, "mdd_pct": 12.0,
                    "romd": (8.0 if target == "C96" else 12.0) + seeds.index(seed) * 0.2,
                    "ev_a_r": 0.4, "trades": 50, "exposure_pct": 45.0, "win_rate_pct": 56.0,
                    "yearly": _synthetic_full_window_yearly(return_pct=4.0)}
        def matching_panels(selected_definition, **_kwargs):
            complete, evidence = _mock_panels(full)
            selected = {str(item["target"]) for item in selected_definition.source["oracle_comparisons"]}
            return {key: value for key, value in complete.items() if key in selected}, evidence

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [
                     (seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", side_effect=matching_panels) as panels, \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=(
                     {"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=(
                     "2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=(
                     "rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=fake_replay), \
                 patch.object(oracle, "_metrics", side_effect=fake_metrics), \
                 patch.object(oracle, "_formal_dl_score_start", return_value="2021-01-01"):
                first = oracle.run_formal_audit(self.definition, project_root=root, quiet=True)
                self.assertEqual(len(outcomes), 40)
                self.assertEqual(panels.call_count, 1)
                second = oracle.build_report(self.definition, project_root=root, quiet=True)
                self.assertEqual(len(outcomes), 40)
                self.assertEqual(panels.call_count, 1, "Signed current-source REUSE cannot rebuild truth")
                self.assertEqual(second["oracles"], first["oracles"])
                # A reduced comparison list must reuse the SAME certified
                # historical seed-level replays without any provider work.
                from dataclasses import replace
                selected = tuple(item for item in self.definition.source["oracle_comparisons"]
                                 if item["target"] in {"MFE40_Control", "R40"})
                subset = replace(self.definition, source={**self.definition.source,
                                                          "oracle_comparisons": selected})
                reduced = oracle.build_report(subset, project_root=root, quiet=True)
                self.assertEqual([row["target"] for row in reduced["oracles"]],
                                 ["MFE40_Control", "R40"])
                self.assertEqual(len(reduced["definitions"]), 2)
                self.assertEqual(len(outcomes), 40)
                self.assertEqual(panels.call_count, 1)
                with patch.object(oracle, "price_inventory_sha256", return_value="c" * 64):
                    with self.assertRaises(oracle.OracleBlockedError):
                        oracle.build_report(self.definition, project_root=root, quiet=True)
                self.assertEqual(len(outcomes), 40)

    def test_retired_provider_cohort_reuse_requires_mathematical_proof(self):
        from services.audit.oracle_replay_artifacts import oracle_cohorts_compatible
        new = {"cohort_rows": 3, "first_common_day": "2021-01-01", "last_common_day": "2021-05-01",
               "research_cutoff": "2021-06-01", "risk_budget_return": 0.1,
               "providers": {"40d": "mfe_40"}, "scoring": "percentile",
               "market_truth": {"score_start": "2020-12-31", "mature_rows_by_provider": {"pure_mfe40_provider": 3}},
               "rows_outside_dl_training_provider_by_source": {"pure_mfe40_provider": 1}}
        old = {**new, "providers": {"40d": "mfe_40", "20d": "mfe_20", "first_breach": "breach"},
               "market_truth": {"score_start": "2020-12-31", "mature_rows_by_provider": {
                   "pure_mfe40_provider": 3, "mfe20_d20_provider": 3, "first_breach_provider": 3}},
               "rows_outside_dl_training_provider_by_source": {
                   "pure_mfe40_provider": 1, "mfe20_d20_provider": 1, "first_breach_provider": 1}}
        self.assertTrue(oracle_cohorts_compatible(old, new))
        not_proven = {**old, "market_truth": {"score_start": "2020-12-31", "mature_rows_by_provider": {
            "pure_mfe40_provider": 4, "mfe20_d20_provider": 3, "first_breach_provider": 3}}}
        self.assertFalse(oracle_cohorts_compatible(not_proven, new))
        self.assertFalse(oracle_cohorts_compatible({**old, "providers": {"40d": "changed"}}, new))

    def test_t40_terminal_return_with_mae_clamp_and_negative_return(self):
        from filters.breakout_quality.continuous_target import full_horizon_mae_gated_reward_r

        # The actual 40D stop risk is a *research* fixed 1R, not an ATR stop.
        # At MAE>=1R, the result must equal -1R irrespective of positive R40.
        r40 = pd.Series([1.5, 1.5, 1.5, 1.5, 1.5, -0.4])
        safety = pd.Series([0.0, -0.2, -0.5, -0.9, -1.2, -0.5])
        target = full_horizon_mae_gated_reward_r(r40, safety)
        np.testing.assert_allclose(target, [1.5, 1.0, 0.25, -0.75, -1.0, -0.7])
        with self.assertRaisesRegex(ValueError, "nonpositive safety"):
            full_horizon_mae_gated_reward_r(np.array([1.0]), np.array([0.2]))
        with self.assertRaisesRegex(ValueError, "finite matched R"):
            full_horizon_mae_gated_reward_r(np.array([float("nan")]), np.array([-0.1]))

    def test_registered_r40_t40_t40mfe_training_targets_match_oracle_economics(self):
        # One canonical 40-bar price path drives both supervised DL truth and the
        # hindsight Oracle T40/T40-MFE formulas; no alternative target engine.
        from core.breakout_quality_registry import (
            SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
            get_breakout_quality_experiment_profile,
            get_continuous_ranker_research_spec,
        )
        from filters.breakout_quality.continuous_target import (
            StrategyAlignedContinuousTargetSpec, full_horizon_mae_gated_reward_r,
        )
        from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
        spec = StrategyAlignedContinuousTargetSpec.from_label_policy(oracle.DEFAULT_LABEL_POLICY)
        self.assertEqual(spec.horizon_bars, 40)
        # Test registered historical Target contracts; do not force retired CI into
        # the mutable current Model Compare/Test list merely to keep regression coverage.
        by_model = {
            research.model_research_id: profile_name
            for profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
            for research in (get_continuous_ranker_research_spec(profile_name),)
        }
        self.assertTrue({"MR-13CI", "MR-13CJ", "MR-13CK"}.issubset(by_model))
        by_target = {
            "MR-13CI": "daily_terminal_r40_v1",
            "MR-13CJ": "daily_terminal_mae_gated_t40_v1",
            "MR-13CK": "daily_full_horizon_mae_gated_mfe_t40_v1",
        }
        policies = []
        for mr, expected_id in by_target.items():
            profile = get_breakout_quality_experiment_profile(by_model[mr])
            research = get_continuous_ranker_research_spec(by_model[mr])
            self.assertEqual(profile.continuous_target_id, expected_id)
            self.assertEqual(profile.model_architecture, "inception_time_v1")
            self.assertEqual(research.model_research_id, mr)
            policies.append((profile.training_objective, profile.loss_name,
                             research.pairwise_reduction, research.trainer_family))
        self.assertEqual(policies[0], policies[1], "R40/T40 use identical learner and Loss")
        self.assertEqual(policies[1], policies[2], "T40 and T40-MFE change only the target")

        def price_frame(close_40, lowest_40, future_peak=None):
            close = np.full(41, 100.0)
            close[40] = close_40
            high = np.maximum(close, 101.0)
            if future_peak is not None:
                high[10] = future_peak
            low = np.minimum(close, 99.0)
            low[10] = lowest_40
            low[40] = min(low[40], close_40)
            return pd.DataFrame({"Close": close, "High": high, "Low": low})

        for terminal_close, min_low, peak, expected_r, expected_t, expected_ck in [
            (115.0, 95.0, None, 1.5, 0.25, 0.25),
            (115.0, 89.0, None, 1.5, -1.0, -1.0),
            (96.0, 95.0, None, -0.4, -0.7, -0.45),
            (105.0, 95.0, 130.0, 0.5, -0.25, 1.0),
        ]:
            frame = price_frame(terminal_close, min_low, peak)
            scored = {}
            for mr, target_id in by_target.items():
                batch = compute_daily_opportunity_target_batch(
                    frame, np.array([0], dtype=np.int64), spec=spec, target_id=target_id,
                )
                self.assertTrue(batch.valid_mask[0], mr)
                scored[mr] = float(batch.target_raw_r[0])
            self.assertAlmostEqual(scored["MR-13CI"], expected_r, places=5)
            self.assertAlmostEqual(scored["MR-13CJ"], expected_t, places=5)
            self.assertAlmostEqual(scored["MR-13CK"], expected_ck, places=5)
            mae_r = max(0, 1.0 - min_low / 100.0) / spec.risk_budget_return
            oracle_t = float(full_horizon_mae_gated_reward_r(
                np.array([expected_r]), np.array([-mae_r]),
            )[0])
            self.assertAlmostEqual(scored["MR-13CJ"], oracle_t, places=5)
            mfe_r = (float(frame.High.iloc[1:41].max()) / 100.0 - 1.0) / spec.risk_budget_return
            oracle_ck = float(full_horizon_mae_gated_reward_r(
                np.array([mfe_r]), np.array([-mae_r]),
            )[0])
            self.assertAlmostEqual(scored["MR-13CK"], oracle_ck, places=5)

    def test_mismatched_stock_days_cannot_be_silently_filled(self):
        # If an official source claims a valid row but Research prices cannot
        # produce the same 40D-mature future path, stop instead of imputing it.
        with patch.object(oracle, "_load_truth", side_effect=lambda definition, root, source_field, cutoff: self._truth(source_field)), \
             patch.object(oracle, "_market_truth_for_oracle", return_value=self._mock_market(omit_first_breach_last=True)):
            with self.assertRaisesRegex(oracle.OracleBlockedError, "canonical score-eligible stock-day lacks Research price path"):
                oracle._build_oracle_score_panels(
                    self.definition, project_root=ROOT, cutoff="2026-03-02",
                )

    def test_price_truth_restricts_scores_to_official_dl_eligible_stock_days(self):
        # The benchmark 0050 and an early-listed stock may be Min candidates,
        # but neither may receive an Oracle score without a formal DL sample.
        days = pd.bdate_range("2020-01-01", periods=700)
        short_days = pd.bdate_range("2021-01-04", periods=180)
        def price_frame(dates):
            close = np.arange(len(dates), dtype=float) / 50.0 + 100.0
            return pd.DataFrame({"Open": close, "High": close * 1.01,
                                 "Low": close * 0.99, "Close": close,
                                 "Volume": 1_000_000.0}, index=dates)
        prices = {"0050": price_frame(days), "2330": price_frame(days),
                  "2211": price_frame(short_days)}
        paths = [(ticker, Path(ticker + ".csv")) for ticker in prices]
        settings = SimpleNamespace(start_date="2021-01-01", end_date="2021-06-01", dataset="full")
        providers = oracle._oracle_comparison_plan(self.definition)["providers"]
        valid = {field: pd.DataFrame({"ticker": ["2330"], "date": ["2021-02-26"]})
                 for field in providers}
        with patch.object(oracle, "_contract", return_value=(settings, None, None)), \
             patch.object(oracle, "_formal_dl_score_start", return_value="2021-01-01"), \
             patch.object(oracle, "discover_dataset_csv_inputs", return_value=(paths, [])), \
             patch.object(oracle, "load_dataset_frame", side_effect=lambda path, ticker, **kw: prices[ticker]):
            truth, evidence = oracle._market_truth_for_oracle(
                self.definition, ROOT, cutoff="2021-06-01", eligible_truth=valid,
            )
        self.assertEqual(evidence["score_start"], "2021-01-01")
        self.assertEqual(evidence["price_ticker_count"], 3)
        for provider in providers:
            table = truth[provider]
            self.assertEqual(set(table["ticker"]), {"2330"})
            self.assertEqual(set(table["date"]), {"2021-02-26"})
            self.assertFalse(table.duplicated(["ticker", "date"]).any())
            self.assertTrue(np.isfinite(table[["mfe", "peak_adverse", "min_low", "raw"]]
                                        .to_numpy()).all())
            if provider == "pure_mfe40_provider":
                index = prices["2330"].index.get_loc(pd.Timestamp("2021-02-26"))
                risk = oracle.StrategyAlignedContinuousTargetSpec.from_label_policy(
                    oracle.DEFAULT_LABEL_POLICY).risk_budget_return
                expected = (prices["2330"].iloc[index + 40]["Close"] /
                            prices["2330"].iloc[index]["Close"] - 1) / risk
                self.assertAlmostEqual(float(table["scalar_raw_terminal_r"].iloc[0]), expected, places=6)

    def test_missing_benchmark_price_is_blocked_not_neutral_filled(self):
        eligible = {field: pd.DataFrame({"ticker": ["2330"], "date": ["2025-01-02"]})
                    for field in oracle._oracle_comparison_plan(self.definition)["providers"]}
        with patch.object(oracle, "discover_dataset_csv_inputs", return_value=([("2330", Path("2330.csv"))], [])):
            with self.assertRaisesRegex(oracle.OracleBlockedError, "Benchmark價格"):
                oracle._market_truth_for_oracle(self.definition, ROOT, cutoff="2026-03-02",
                                                eligible_truth=eligible)

    def test_reconcile_rejects_oracle_only_extra_price_stock_days(self):
        official = pd.DataFrame({"ticker": ["2330"], "date": ["2025-01-02"],
                                 "mfe": [1.5], "peak_adverse": [0.2],
                                 "min_low": [-0.08], "raw": [1.5]})
        market = pd.concat([official, official.assign(ticker="2211", mfe=2.0, raw=2.0)],
                           ignore_index=True)
        with self.assertRaisesRegex(oracle.OracleBlockedError, "Oracle prices must equal the complete eligible DL cohort"):
            oracle._reconcile_oracle_truth(official, market, field="pure_mfe40_provider",
                                           score_start="2025-01-01")
        matched, extras = oracle._reconcile_oracle_truth(
            official, official.assign(scalar_raw_terminal_r=1.25), field="pure_mfe40_provider",
            score_start="2025-01-01",
        )
        self.assertEqual(extras, 0)
        self.assertEqual(set(matched.ticker), {"2330"})
        self.assertEqual(float(matched.scalar_raw_terminal_r.iloc[0]), 1.25)
        incompatible = official.assign(mfe=2.0)
        with self.assertRaisesRegex(oracle.OracleBlockedError, "mature DL target disagrees with OHLCV"):
            oracle._reconcile_oracle_truth(official, incompatible,
                                           field="pure_mfe40_provider", score_start="2025-01-01")

    def test_price_oracle_unscored_tickers_follow_pit_missing_rank(self):
        provider = pd.DataFrame({"ticker": ["2330"], "date": ["2021-02-26"],
                                 "mfe": [1.0], "peak_adverse": [0.3],
                                 "min_low": [-0.07], "raw": [1.0]})
        market = ({field: (provider.assign(
                              scalar_raw_mfe=[1.0], scalar_raw_terminal_r=[0.8],
                              scalar_raw_terminal_mae_gate=[-0.46], scalar_raw_mfe_mae_gate=[-0.4])
                           if field == "pure_mfe40_provider" else provider)
                   for field in oracle._oracle_comparison_plan(self.definition)["providers"]},
                  {"score_start": "2021-01-01", "truth_source": "synthetic"})
        with patch.object(oracle, "_market_truth_for_oracle", return_value=market), \
             patch.object(oracle, "_load_truth", return_value=provider):
            scores, evidence = oracle._build_oracle_score_panels(
                self.definition, project_root=ROOT, cutoff="2026-03-02",
            )
        self.assertEqual(evidence["cohort_rows"], 1)
        self.assertEqual(evidence["rows_outside_dl_training_provider_by_source"],
                         {field: 0 for field in oracle._oracle_comparison_plan(self.definition)["providers"]})
        for target, _ in LEGACY_ORACLE_TARGETS:
            self.assertEqual(set(scores[target]), {("2330", "2021-02-26")})
            with audit_oracle_truth_score_context(
                scores[target], ranking_policy=self.exact_policy,
                ranking_options=self.exact_options,
                missing_key_policy="unavailable_outside_mature_feature_universe",
                score_start="2021-01-01",
            ):
                for ticker in ("0050", "2211"):
                    rank = resolve_breakout_quality_candidate_rank(
                        ticker=ticker, signal_date="2021-02-26",
                        information_date="2021-02-26", high_len=260,
                    )
                    self.assertFalse(rank["available"])
                    self.assertIsNone(rank["score"])
                    self.assertEqual(rank["unavailable_reason"], "missing_ticker_date_score")

    def test_canonical_provider_loader_keeps_official_rows(self):
        provider = "pure_mfe40_provider"
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        profile = get_breakout_quality_experiment_profile(str(self.definition.source[provider]))
        fake_bundle = SimpleNamespace(
            target_manifest={"target_contract": {"target_id": profile.continuous_target_id}},
            group_table=pd.DataFrame({
                "ticker": ["2330"], "date": ["2025-01-02"],
                "label_eval_end_date": ["2025-03-10"],
                "target_favorable_r": [1.5], "target_adverse_r": [0.2],
                "target_minimum_low_return": [-0.08],
            }),
            raw_target=np.array([1.5]), target_valid=np.array([True]),
        )
        with patch.object(oracle, "load_profile_continuous_ranker_data", return_value=fake_bundle):
            result = oracle._load_truth(self.definition, ROOT, provider, cutoff="2026-03-02")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.ticker.iloc[0], "2330")

    def test_ssot_golden_is_strict_across_windows_line_endings(self):
        from tools.validate.ssot_regression_guard import (
            APPROVED_GOLDEN_SHA256, read_approved_baseline,
        )
        import hashlib

        original = (ROOT / "tools/validate/baselines/ssot_regression.json").read_bytes()
        lf = original.replace(b"\r\n", b"\n")
        crlf = lf.replace(b"\n", b"\r\n")
        self.assertEqual(hashlib.sha256(lf).hexdigest(), APPROVED_GOLDEN_SHA256)
        self.assertEqual(
            read_approved_baseline(ROOT, raw_bytes=lf),
            read_approved_baseline(ROOT, raw_bytes=crlf),
        )
        for changed in (lf + b" ", crlf + b" "):
            with self.assertRaisesRegex(ValueError, "Golden hash mismatch"):
                read_approved_baseline(ROOT, raw_bytes=changed)
        # A real JSON value mutation must still fail, regardless of line ending.
        changed = lf.replace(b'"schema_version": 1', b'"schema_version": 2', 1)
        self.assertNotEqual(changed, lf)
        with self.assertRaisesRegex(ValueError, "Golden hash mismatch"):
            read_approved_baseline(ROOT, raw_bytes=changed)

    def test_retired_history_never_required_for_current_report_or_oracle_reuse(self):
        from core.report_contract_versions import (
            load_report_contract_versions, validate_active_report_contract_versions,
        )
        from core.research_report_contract import (
            PERSISTENT_REPORT_CONTRACTS, persistent_report_contract_fingerprints,
        )
        from tools.validate.ssot_regression_guard import (
            read_approved_baseline, report_contract_version_history_violations,
        )

        with tempfile.TemporaryDirectory(prefix="standalone_report_contract_") as tmp:
            project = Path(tmp)
            for name in (
                "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json",
                "tools/validate/baselines/ssot_regression.json",
            ):
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            self.assertFalse((project / "doc/scientific_contracts/CONTRACT_EVIDENCE_HISTORY.zip").exists())
            registry = load_report_contract_versions(project)
            current = persistent_report_contract_fingerprints()
            self.assertEqual(validate_active_report_contract_versions({
                report_id: (contract.version, current[report_id])
                for report_id, contract in PERSISTENT_REPORT_CONTRACTS.items()
            }, project_root=project), ())
            approved = read_approved_baseline(project)
            self.assertEqual(report_contract_version_history_violations(project, approved), [])
            # All past report IDs/fingerprints needed for completed Oracle
            # payload compatibility remain available in one small ledger.
            oracle_history = registry["audit.target_oracle_romd"]
            self.assertEqual(tuple(row["version"] for row in oracle_history), tuple(range(1, 12)))
            self.assertEqual(oracle_history[-1]["fingerprint"], current["audit.target_oracle_romd"])
            self.assertTrue(all(set(row) == {"contract_id", "version", "fingerprint"}
                                for rows in registry.values() for row in rows))
            # Mutation rejection is independent of the retired file inventory.
            ledger_path = project / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            row = next(x for x in ledger["reports"] if x["report_id"] == "audit.target_oracle_romd")
            row["versions"][0]["fingerprint"] = "0" * 16
            ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
            self.assertTrue(report_contract_version_history_violations(project, approved))
            fixture = project / "tools/validate/baselines/ssot_regression.json"
            fixture.write_bytes(fixture.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Golden hash mismatch"):
                read_approved_baseline(project)

    def test_future_report_versions_need_only_compact_ledger_row(self):
        from core.report_contract_versions import load_report_contract_versions
        from tools.validate.ssot_regression_guard import (
            read_approved_baseline, report_contract_version_history_violations,
        )
        approved = read_approved_baseline(ROOT)
        self.assertEqual(approved["schema_version"], 1)
        with tempfile.TemporaryDirectory(prefix="report_ledger_append_") as tmp:
            project = Path(tmp)
            folder = project / "doc/scientific_contracts"
            folder.mkdir(parents=True)
            ledger_path = folder / "REPORT_CONTRACT_VERSIONS.json"
            shutil.copy2(ROOT / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json", ledger_path)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            entry = next(x for x in ledger["reports"] if x["report_id"] == "audit.target_oracle_romd")
            self.assertEqual(entry["versions"][-1]["version"], 11)
            entry["versions"].append({
                "contract_id": "audit.target_oracle_romd@v12",
                "version": 12,
                "fingerprint": "1" * 16,
            })
            ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(load_report_contract_versions(project)["audit.target_oracle_romd"][-1]["version"], 12)
            self.assertEqual(report_contract_version_history_violations(project, approved), [])
            entry["versions"][7]["fingerprint"] = "0" * 16
            ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
            self.assertTrue(report_contract_version_history_violations(project, approved))

    def test_reusable_research_report_publishes_schema7_exact_and_direct(self):
        def summary(name: str):
            return {
                "target": name, "total_return_pct": 100.0, "annualized_return_pct": 14.0,
                "mdd_pct": 12.0, "romd": 8.0, "romd_std": 0.0,
                "ev_a_r": 0.5, "seeds": 1, "yearly": [],
                "seed_observations": [{"seed": 123, "romd": 8.0, "delta_romd_vs_c96": 0.0}],
                "delta_romd_vs_c96": 0.0, "positive_seed_count": 0,
            }
        result = {
            "schema_version": 8, "status": "COMPLETED",
            "start_date": "2021-01-01", "end_date": "2025-12-22",
            "common_cohort": {"cohort_rows": 3},
            "benchmark_id": "test", "seeds": [123], "limitations": ["NOT PIT"],
            "baseline": summary("C96"),
            "oracles": [summary(name) for name, _ in LEGACY_ORACLE_TARGETS],
            "definitions": dict(LEGACY_ORACLE_TARGETS),
        }
        from services.audit.runner import render_latest_audit_summary
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch.object(oracle, "build_report", return_value=result):
                published = oracle.run_formal_audit(self.definition, project_root=base, quiet=True)
            self.assertEqual(published["schema_version"], 8)
            self.assertEqual(published["report_contract_id"], "audit.target_oracle_romd@v11")
            latest = base / "outputs/audit" / self.definition.output_subdir / "latest"
            body = (latest / "audit.md").read_text(encoding="utf-8")
            for name in oracle._oracle_comparison_plan(self.definition)["aliases"]:
                if name in dict(LEGACY_ORACLE_TARGETS):
                    self.assertIn("**" + name + "**", body)
            self.assertIn("**R40**", body)
            self.assertIn("**T40**", body)
            self.assertIn("**T40-MFE**", body)
            self.assertIn("**T40-MFE_Direct**", body)
            self.assertNotIn("60D Mature Oracle", body)
            self.assertNotIn("**CH20_EqualRank**", body)
            self.assertIn("MAE40 >= 1R", body)
            self.assertNotIn("Delta RoMD", body)
            # Display titles describe stable schema, not today's selected inventory.
            self.assertIn("## 2. Target Definitions\n", body)
            self.assertIn(f"Oracle treatments: {len(LEGACY_ORACLE_TARGETS)} ", body)
            self.assertNotIn("5 × Exact / 4 × Direct", body)
            # A new comparison row changes rendered data but not report identity.
            expanded = {**result, "oracles": [*result["oracles"], summary("FUTURE_TEST_ONLY")]}
            expanded_body = oracle.render_result(expanded, target="markdown")
            self.assertIn(f"Oracle treatments: {len(LEGACY_ORACLE_TARGETS) + 1} ", expanded_body)
            self.assertIn("## 2. Target Definitions\n", expanded_body)
            self.assertEqual(published["report_contract_id"], "audit.target_oracle_romd@v11")
            json_data = json.loads((latest / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(len(json_data["oracles"]), len(LEGACY_ORACLE_TARGETS))
            self.assertIn("**MFE40**", body)
            self.assertNotIn("**MFE40_Control**", body)
            for old_name in ("AO40_EqualRank", "U40", "FirstBreach_NetR"):
                self.assertNotIn("**" + old_name + "**", body)
            self.assertIn("MFE40_Control", json.dumps(json_data))
            self.assertNotIn("additional_panels", json_data)
            listing = render_latest_audit_summary(
                "breakout_quality", project_root=base, method_id=oracle.SUPPORTED_AUDIT_TYPE,
            )
            self.assertIn("存在", listing)
            self.assertIn(self.definition.report_id, listing)

    def test_four_seed_paired_replay_and_report_without_real_market(self):
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        seeds = (693545351, 2014432738, 104504529, 762252050)
        self.assertEqual(len(seeds), 4)
        seen = []
        def fake_replay(*, name, **kwargs):
            if name.startswith("C96"):
                self.assertNotEqual(get_breakout_quality_ranking_source_context().score_source,
                                    SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)
            else:
                context = get_breakout_quality_ranking_source_context()
                self.assertEqual(context.score_source, SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)
                expected_policy = (
                    "resource-aware-continuous-score-no-k-no-r0"
                    if "_Direct" in name else self.exact_policy
                )
                self.assertEqual(context.ranking_policy, expected_policy)
            seen.append(name)
            return {"name": name}
        def fake_metrics(replay, *, target):
            seed = int(replay["name"].split("seed=")[1].split()[0])
            base_romd = 8.0 + seeds.index(seed) * 0.3
            return {
                "target": target, "total_return_pct": 100.0, "annualized_return_pct": 15.0,
                "mdd_pct": 12.0, "romd": base_romd if target == "C96" else base_romd + 1.5,
                "ev_a_r": 0.3, "trades": 50, "exposure_pct": 75.0, "win_rate_pct": 56.0,
                "yearly": _synthetic_full_window_yearly(return_pct=4.0),
            }
        with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
             patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
             patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels({"cohort_rows": 4, "last_common_day": "2025-12-31", "mfe40_over_d40_zero_downside_positive_upside_count": 0, "mfe40_over_d40_zero_zero_count": 0})), \
             patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
             patch.object(oracle, "_validate_requested_param_policy"), \
             patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
             patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
             patch.object(oracle, "_run_scenario", side_effect=fake_replay), \
             patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
             patch.object(oracle, "_metrics", side_effect=fake_metrics):
            with tempfile.TemporaryDirectory() as tmp:
                report = oracle.build_report(self.definition, project_root=Path(tmp), quiet=True)
                original_calls = len(seen)
                reused = oracle.build_report(self.definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(len(seen), original_calls, "identical replay must REUSE all arms")
                self.assertEqual(reused["baseline"], report["baseline"])
                self.assertEqual(reused["oracles"], report["oracles"])
                # A scientifically different replay identity is NOT permission
                # to silently run it again when matching-panel cache exists.
                with patch.object(oracle, "price_inventory_sha256", return_value="c" * 64):
                    with self.assertRaisesRegex(oracle.OracleBlockedError, "dataset_sha256"):
                        oracle.build_report(self.definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(len(seen), original_calls)
        self.assertEqual(len(seen), len(seeds) * (1 + len(LEGACY_ORACLE_TARGETS)))
        self.assertEqual(report["seeds"], list(seeds))
        self.assertEqual(report["schema_version"], 8)
        self.assertEqual(len(report["oracles"]), len(LEGACY_ORACLE_TARGETS))
        self.assertEqual(report["end_date"], "2025-12-22")
        self.assertNotIn("additional_panels", report)
        self.assertEqual(report["baseline"]["romd_std"], report["oracles"][0]["romd_std"])
        self.assertEqual(report["oracles"][0]["delta_romd_vs_c96"], 1.5)
        self.assertEqual(report["oracles"][0]["positive_seed_count"], len(seeds))
        self.assertIn("Oracle Performance", oracle.render_result(report, target="markdown"))
        self.assertIn("**R40**", oracle.render_result(report, target="markdown"))
        self.assertIn("**T40**", oracle.render_result(report, target="markdown"))
        self.assertIn("**T40_Direct**", oracle.render_result(report, target="markdown"))
        self.assertNotIn("MFE40/D40 zero denominator", oracle.render_result(report, target="console"))
        self.assertIn("RoMD Mean", oracle.render_result(report, target="console"))
        json.dumps(report, ensure_ascii=False)
        self.assertNotEqual(get_breakout_quality_ranking_source_context().score_source,
                            SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)



    def test_oracle_and_daily_pit_reuse_identical_portfolio_rank_input_contract(self):
        """Verify the same candidate/selector contract in every configured time mode.

        Source replacement alone must not change the signal-date policy, daily
        refresh, score ordering or constrained-selector numeric objective. This
        is NOT a claim that archived Oracle percentiles equal raw DL scores.
        """
        from core.portfolio_candidates import _resolve_candidate_quality_ranking
        from core.portfolio_entry_selection_max_dl import (
            _max_dl_score_order, _max_dl_basket_quality_key,
        )
        from core.strategy_compare_policy import (
            get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
        )
        from core.strategy_compare_registry import STRATEGY_COMPARE_ROLLING_TEST_MODES
        from core.extended_signals import attach_breakout_quality_rank
        from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME

        oracle_exact = oracle._contract(self.active_definition)[2]
        oracle_benchmark = get_strategy_multi_seed_robustness_settings(
            str(self.active_definition.source["strategy_profile_id"])
        )
        modes_checked = set()
        checked_arms = set()
        for mode in STRATEGY_COMPARE_ROLLING_TEST_MODES:
            profile_id = str(mode["profile_id"])
            settings = get_strategy_comparison_settings(profile_id)
            benchmark = get_strategy_multi_seed_robustness_settings(profile_id)
            self.assertEqual(benchmark.benchmark_id, oracle_benchmark.benchmark_id)
            self.assertEqual(tuple(benchmark.resolved_seeds), tuple(oracle_benchmark.resolved_seeds))
            base = settings.arms[str(self.active_definition.source["baseline_arm_id"])]
            matched = [
                arm for arm in settings.enabled_arms
                if (arm.dl_enabled and arm.dl_id
                    and arm.param_source == base.param_source
                    and arm.param_policy == base.param_policy
                    and arm.rule_policy == "all_off"
                    and arm.dl_runtime_mode == oracle_exact.dl_runtime_mode
                    and dict(arm.dl_runtime_options or {}) == dict(oracle_exact.dl_runtime_options or {}))
            ]
            self.assertTrue(matched, f"{profile_id}: no current Daily-universal Exact arm")
            modes_checked.add(str(mode["mode_id"]))
            for arm in matched:
                with self.subTest(mode=profile_id, arm_id=arm.arm_id):
                    checked_arms.add((profile_id, arm.arm_id))
                    source = settings.dl_sources[str(arm.dl_id)]
                    self.assertEqual(source.score_source, SCORE_SOURCE_SELECTION_POINT_IN_TIME)
                    self.assertEqual(oracle_exact.dl_runtime_options, arm.dl_runtime_options)
                    params = SimpleNamespace(
                        use_breakout_quality_ranking=True,
                        breakout_quality_filter_id=str(source.filter_id), high_len=260,
                    )
                    score_map = {
                        (ticker, day): score
                        for ticker, first, second in (
                            ("2330", 0.21, 0.91), ("2317", 0.85, 0.16),
                        )
                        for day, score in (
                            ("2025-02-03", first), ("2025-02-04", second),
                        )
                    }

                    def fake_pit_lookup(*, ticker, signal_date, filter_id, **_kwargs):
                        day = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
                        return {
                            "score": score_map[(ticker, day)], "score_date": day,
                            "available": True, "unavailable_reason": "",
                            "filter_id": filter_id, "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                        }

                    def run_source(use_oracle):
                        context = (
                            audit_oracle_truth_score_context(
                                score_map, ranking_policy=oracle_exact.dl_runtime_mode,
                                ranking_options=oracle_exact.dl_runtime_options,
                                key_mode="information_date",
                            ) if use_oracle else breakout_quality_ranking_source_context(
                                score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                                model_architecture=str(source.model_architecture),
                                experiment_profile=str(source.experiment_profile),
                                ranking_policy=arm.dl_runtime_mode,
                                ranking_options=arm.dl_runtime_options,
                            )
                        )
                        with context, patch(
                            "filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score",
                            side_effect=fake_pit_lookup,
                        ) as pit_lookup:
                            signals = {}
                            history = []
                            for info_date in ("2025-02-03", "2025-02-04"):
                                candidates = []
                                for ticker in ("2330", "2317"):
                                    signal_state = signals.setdefault(ticker, {})
                                    rank = _resolve_candidate_quality_ranking(
                                        params=params, ticker=ticker,
                                        signal_date="2025-02-03", information_date=info_date,
                                        signal_state=signal_state,
                                    )
                                    attach_breakout_quality_rank(signal_state, rank)
                                    self.assertEqual(rank["score_date"], info_date)
                                    candidates.append({
                                        "ticker": ticker, "breakout_quality_score": rank["score"],
                                        "breakout_quality_rank": rank,
                                    })
                                base_rank = {id(row): i for i, row in enumerate(candidates)}
                                order = _max_dl_score_order(candidates, base_rank=base_rank)
                                objective = _max_dl_basket_quality_key(order, base_rank=base_rank)
                                ordered_scores = tuple(
                                    (r["ticker"], r["breakout_quality_score"]) for r in order
                                )
                                history.append((ordered_scores, objective))
                            if use_oracle:
                                pit_lookup.assert_not_called()
                            else:
                                self.assertEqual(pit_lookup.call_count, 4)
                            return history

                    oracle_history, dl_history = run_source(True), run_source(False)
                    self.assertEqual(oracle_history, dl_history)
                    self.assertNotEqual(
                        oracle_history[0][0], oracle_history[1][0],
                        "Daily-universal continued signals must refresh rankings",
                    )
        self.assertGreaterEqual(len(modes_checked), 2)
        self.assertGreaterEqual(len(checked_arms), len(modes_checked))

    def test_exact_score_sum_is_not_merely_a_rank_order(self):
        """Retain the historical numeric objective; never silently change its scale.

        Two score sets can have exactly the same stock order but different
        optimal fixed-K baskets. A new common-percentile research comparison
        would therefore require its own scientific identity and evidence.
        """
        from core.portfolio_entry_selection_max_dl import _max_dl_basket_quality_key

        def quality(scores, selected):
            rows = [
                {"ticker": str(i), "breakout_quality_score": float(score),
                 "breakout_quality_rank": {"available": True, "score": float(score)}}
                for i, score in enumerate(scores)
            ]
            rank = {id(row): i for i, row in enumerate(rows)}
            return _max_dl_basket_quality_key([rows[i] for i in selected], base_rank=rank)

        tight = (0.99, 0.70, 0.69, 0.10)
        spread = (0.99, 0.20, 0.19, 0.10)
        self.assertTrue(quality(tight, (1, 2)) > quality(tight, (0, 3)))
        self.assertTrue(quality(spread, (0, 3)) > quality(spread, (1, 2)))

    def test_oracle_and_pit_identical_scores_produce_identical_broker_cash_replay(self):
        """One actual Broker-Cash engine for both score sources, Exact AND Direct.

        This is behavioral evidence, not a mock _run_scenario. Reversing two
        stock rankings must change the actual filled ticker; with identical
        scores each policy must agree on all canonical trade/equity outputs.
        """
        from core.strategy_params import V16StrategyParams
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.strategy_compare_registry import STRATEGY_COMPARE_ROLLING_TEST_MODES
        from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
        from tools.validate.synthetic_portfolio_common import (
            build_synthetic_competing_candidates_case, run_portfolio_core_check_for_dir,
        )
        from tools.validate.synthetic_frame_utils import write_synthetic_csv_bundle

        case = build_synthetic_competing_candidates_case(V16StrategyParams())
        params = case["params"]
        params.use_breakout_quality_ranking = True
        params.breakout_quality_filter_id = str(self.active_definition.source["filter_id"])
        settings, baseline, exact = oracle._contract(self.active_definition)
        direct = oracle._direct_arm(self.active_definition, settings, baseline, exact)
        policies = {"exact": exact, "direct": direct}
        with tempfile.TemporaryDirectory() as tmp:
            write_synthetic_csv_bundle(tmp, case["frames"])
            for mode in STRATEGY_COMPARE_ROLLING_TEST_MODES:
                settings = get_strategy_comparison_settings(str(mode["profile_id"]))
                base = settings.arms[str(self.active_definition.source["baseline_arm_id"])]
                arms = [
                    arm for arm in settings.enabled_arms
                    if (arm.dl_enabled and arm.dl_id
                        and arm.param_source == base.param_source
                        and arm.param_policy == base.param_policy
                        and arm.rule_policy == "all_off"
                        and arm.dl_runtime_mode == exact.dl_runtime_mode
                        and dict(arm.dl_runtime_options or {}) == dict(exact.dl_runtime_options or {}))
                ]
                self.assertTrue(arms)
                # Pick one real registered PIT score provider per evaluation
                # mode. The separate Direct policy comes from canonical config,
                # not an Audit-specific selector or an invented new DL arm.
                source = settings.dl_sources[str(arms[0].dl_id)]
                for policy_name, policy in policies.items():
                    selected_tickers = set()
                    for winner in case["frames"]:
                        score_map = {
                            (ticker, day): (0.9 if ticker == winner else 0.1)
                            for ticker, frame in case["frames"].items()
                            for day in frame["Date"].to_list()
                        }

                        def fake_pit_lookup(*, ticker, signal_date, filter_id, **_kwargs):
                            date = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
                            return {
                                "score": score_map[(ticker, date)], "score_date": date,
                                "available": True, "unavailable_reason": "",
                                "filter_id": filter_id,
                                "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                            }

                        def execute(use_oracle):
                            context = (
                                audit_oracle_truth_score_context(
                                    score_map, ranking_policy=policy.dl_runtime_mode,
                                    ranking_options=policy.dl_runtime_options,
                                    key_mode="information_date",
                                ) if use_oracle else breakout_quality_ranking_source_context(
                                    score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                                    model_architecture=str(source.model_architecture),
                                    experiment_profile=str(source.experiment_profile),
                                    ranking_policy=policy.dl_runtime_mode,
                                    ranking_options=policy.dl_runtime_options,
                                )
                            )
                            with context, patch(
                                "filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score",
                                side_effect=fake_pit_lookup,
                            ):
                                return run_portfolio_core_check_for_dir(
                                    tmp, params, max_positions=case["max_positions"],
                                    enable_rotation=case["enable_rotation"],
                                    start_year=case["start_year"],
                                    benchmark_ticker=case["benchmark_ticker"],
                                )

                        oracle_result, dl_result = execute(True), execute(False)
                        with self.subTest(mode=mode["mode_id"], policy=policy_name, winner=winner):
                            for field in (
                                "total_return", "mdd", "final_eq", "trade_count",
                                "win_rate", "pf_ev", "pf_payoff", "filled_buy_count",
                                "avg_exp", "max_exp", "reserved_buy_fill_rate",
                            ):
                                self.assertEqual(oracle_result[field], dl_result[field], field)
                            pd.testing.assert_frame_equal(
                                oracle_result["df_trades"], dl_result["df_trades"],
                            )
                            self.assertEqual(
                                oracle_result["portfolio_completed_trades"],
                                dl_result["portfolio_completed_trades"],
                            )
                            bought = oracle_result["df_trades"].loc[
                                oracle_result["df_trades"]["Type"].str.startswith("買進"), "Ticker"
                            ].tolist()
                            self.assertEqual(bought, [winner])
                            selected_tickers.add(winner)
                    self.assertEqual(selected_tickers, set(case["frames"]))


class RemovedMinTrade40OracleComparisonCases(TestCase):
    """The obsolete signal-event arms are absent, not given a new science ID."""

    def test_retired_target_names_stay_reserved_in_canonical_registry(self):
        from core.audit_registry import AUDIT_REUSABLE_REPORTS
        active = _ACTIVE_ORACLE_DEFINITION
        catalog = AUDIT_REUSABLE_REPORTS["breakout_quality"]["target_oracle_romd"]["source"]
        retired = catalog["retired_oracle_targets"]
        self.assertEqual(set(retired), {"MinTrade40_NetR", "MinTrade40_NetR_Direct"})
        original = catalog["oracle_comparisons"]
        self.assertTrue(set(retired).isdisjoint({row["target"] for row in original}))
        existing_daily_exact = next(x for x in original
                                    if x["target"] == "DailyUniversal_MinTrade40_NetR")
        existing_daily_direct = next(x for x in original
                                     if x["target"] == "DailyUniversal_MinTrade40_NetR_Direct")
        for old, template in (("MinTrade40_NetR", existing_daily_exact),
                              ("MinTrade40_NetR_Direct", existing_daily_direct)):
            altered = replace(active, source={**active.source, "oracle_comparisons": (
                *original, {**template, "target": old},
            )})
            with self.subTest(old=old), self.assertRaisesRegex(oracle.OracleBlockedError,
                                                              "retired Target identity"):
                oracle._oracle_comparison_plan(altered)
        # A non-retired Direct row cannot use a reserved old score source.
        altered = replace(active, source={**active.source, "oracle_comparisons": (
            *original, {"target": "Unrelated_Direct", "description": "bad source",
                        "score_source": "MinTrade40_NetR", "selector": "direct"},
        )})
        with self.assertRaisesRegex(oracle.OracleBlockedError, "retired Target identity"):
            oracle._oracle_comparison_plan(altered)

    def test_retired_signal_event_target_is_not_selectable(self):
        current = _ACTIVE_ORACLE_DEFINITION
        planned = oracle._oracle_comparison_plan(current)
        names = {name for name, _ in planned["targets"]}
        self.assertEqual(len(names), 16)
        self.assertNotIn("MinTrade40_NetR", names)
        self.assertNotIn("MinTrade40_NetR_Direct", names)
        self.assertIn("DailyUniversal_MinTrade40_NetR", names)
        self.assertIn("DailyUniversal_MinTrade40_NetR_Direct", names)
        # Previous signal-event recipes must not reappear via ad-hoc config.
        source = {**current.source, "oracle_comparisons": (*current.source["oracle_comparisons"], {
            "target": "MinTrade40_NetR", "description": "retired", "score": "trade_path_realized_net_r",
            "provider": "event_targets", "selector": "exact",
        })}
        with self.assertRaisesRegex(oracle.OracleBlockedError, "retired Target identity"):
            oracle._oracle_comparison_plan(replace(current, source=source))


class OracleSameWindowEvidenceInvariantCases(TestCase):
    @staticmethod
    def _observation(seed, years=(2021, 2022), *, flags=(True, True)):
        return {
            "seed": seed, "target": "T40", "total_return_pct": 13.0,
            "annualized_return_pct": 4.0, "mdd_pct": 3.0, "romd": 4.0,
            "ev_a_r": 0.4, "trades": 10, "exposure_pct": 40.0, "win_rate_pct": 56.0,
            "yearly": [{"year": year, "strategy_return_pct": 12.0,
                        "is_full_year": full}
                       for year, full in zip(years, flags)],
        }

    def test_rejects_a_year_missing_from_one_seed(self):
        with self.assertRaisesRegex(oracle.OracleBlockedError, "annual seed coverage"):
            oracle._aggregate([
                self._observation(1), self._observation(2, (2021,), flags=(True,)),
            ], target="T40")

    def test_rejects_every_seed_missing_the_same_replay_years(self):
        # Prior check only compared seeds with each other; all four could
        # silently drop 2022-2025 and still publish a 'COMPLETED' report.
        four = [self._observation(seed, years=(2021,), flags=(True,))
                for seed in (11, 22, 33, 44)]
        with self.assertRaisesRegex(oracle.OracleBlockedError, "annual coverage does not match replay period"):
            oracle._aggregate(four, target="T40",
                              period_start="2021-01-01", period_end="2025-12-22")
        complete = [self._observation(seed, years=(2021, 2022, 2023, 2024, 2025),
                                      flags=(True, True, True, True, False))
                    for seed in (11, 22, 33, 44)]
        yearly = oracle._aggregate(complete, target="T40",
                                   period_start="2021-01-01", period_end="2025-12-22")["yearly"]
        self.assertEqual([row["year"] for row in yearly], [str(y) for y in range(2021, 2026)])
        self.assertFalse(yearly[-1]["is_full_year"])

    def test_rejects_duplicate_seed_and_duplicate_year(self):
        with self.assertRaisesRegex(oracle.OracleBlockedError, "duplicate benchmark seeds"):
            oracle._aggregate([self._observation(1), self._observation(1)], target="T40")
        duplicate_year = self._observation(2, (2021, 2021))
        with self.assertRaisesRegex(oracle.OracleBlockedError, "duplicated annual evidence"):
            oracle._aggregate([self._observation(1), duplicate_year], target="T40")

    def test_rejects_inconsistent_full_year_flags(self):
        with self.assertRaisesRegex(oracle.OracleBlockedError, "annual seed coverage"):
            oracle._aggregate([
                self._observation(1), self._observation(2, flags=(True, False)),
            ], target="T40")

    def test_rejects_missing_or_non_boolean_full_year_evidence(self):
        for malformed in (None, "False", 0):
            first = self._observation(1)
            first["yearly"][0]["is_full_year"] = malformed
            with self.subTest(malformed=malformed), self.assertRaisesRegex(
                    oracle.OracleBlockedError, "invalid full-year evidence"):
                oracle._aggregate([first, self._observation(2)], target="T40")

    def test_full_and_partial_years_with_four_seeds_are_kept(self):
        records = [self._observation(seed, flags=(True, False)) for seed in (11, 22, 33, 44)]
        result = oracle._aggregate(records, target="T40")
        self.assertEqual(result["seeds"], 4)
        self.assertEqual(result["yearly"], [
            {"year": "2021", "strategy_return_pct": 12.0, "n_seeds": 4, "is_full_year": True},
            {"year": "2022", "strategy_return_pct": 12.0, "n_seeds": 4, "is_full_year": False},
        ])


class OracleReplayEvidenceReuseCases(TestCase):
    """Exact-identity per-arm REUSE and data-driven local report discovery."""

    def setUp(self):
        source = patch.object(oracle, "_price_score_algorithm_sha256",
                              return_value=oracle._price_score_algorithm_sha256())
        source.start()
        self.addCleanup(source.stop)
        for attribute, side_effect in (
            ("_resolve_oracle_target_score_contracts", _fixture_perfect_score_contracts),
            ("_read_only_score_keys", _fixture_read_score_keys),
            ("_load_or_build_oracle_price_scores", _fixture_price_scores),
        ):
            patcher = patch.object(oracle, attribute, side_effect=side_effect)
            patcher.start()
            self.addCleanup(patcher.stop)
        prices = patch("services.audit.oracle_replay_artifacts.price_inventory_sha256",
                       side_effect=lambda *args: oracle.price_inventory_sha256(*args))
        prices.start()
        self.addCleanup(prices.stop)

    @staticmethod
    def _old_definitions():
        active = dict(LEGACY_ORACLE_TARGETS)
        # Frozen *test fixture* from the historical eleven-treatment evidence.
        # Production reuse must not have any historical target-name allowlist.
        retired = {
            "AO40_EqualRank": "40D MFE + adverse-to-peak Safety, 50/50 daily rank",
            "U40": "40D MFE - D40, canonical R",
            "FirstBreach_NetR": "MFE before first risk breach - adverse to that peak, R",
            "CH20_EqualRank": "20D MFE + full D20 Safety, 50/50 daily rank",
            "U20": "20D MFE - D20, canonical R",
            "MFE40_over_D40": "Exact MFE40 / D40 in canonical R; if D40=0 and MFE>0 rank as +infinity, 0/0 ranks as zero",
            "MFE20": "20D maximum favorable excursion, in canonical R",
            "MAE40": "Negative full-horizon 40D MAE (lower adverse excursion ranks higher), in canonical R",
            "MAE20": "Negative full-horizon 20D MAE (lower adverse excursion ranks higher), in canonical R",
        }
        names = ("AO40_EqualRank", "CG40_EqualRank", "CH20_EqualRank", "U40", "U20",
                 "FirstBreach_NetR", "MFE40_Control", "MFE40_over_D40", "MFE20", "MAE40", "MAE20")
        source = {**retired, **active}
        return {name: source[name] for name in names}


    @staticmethod
    def _observation(target, seed):
        romd = float(seed) + (0.0 if target == "C96" else 2.0)
        observation = {
            "target": target, "seed": seed,
            "total_return_pct": 123.0, "annualized_return_pct": 16.0,
            "mdd_pct": 12.0, "romd": romd, "ev_a_r": 0.7,
            "trades": 100, "exposure_pct": 75.0, "win_rate_pct": 56.0,
            "yearly": _synthetic_full_window_yearly(return_pct=14.0),
        }
        if target != "C96":
            observation["delta_romd_vs_c96"] = 2.0
        return observation

    @staticmethod
    def _identity(seed=1, target="CG40_EqualRank", dataset_sha="a" * 64, param_sha="b" * 64):
        from services.audit.oracle_replay_artifacts import replay_identity
        return replay_identity(
            dataset="full", dataset_sha256=dataset_sha,
            target=target, definition={
                "C96": "Canonical same-window Min P16 Base-Finalist-Best",
                **dict(LEGACY_ORACLE_TARGETS),
            },
            start="2021-01-01", end="2025-12-22",
            benchmark="end_to_end_v1", param_sha256=param_sha, seed=seed,
            cohort={"cohort_rows": 123, "research_cutoff": "2026-03-02"},
            ranking_policy="score", ranking_options={"preserve_k_r0": True},
            evaluation_mode="rolling", max_positions=10,
            replay_policy={"exact_k_r0": True},
        )

    @staticmethod
    def _write_legacy_cache(store, identity, observation):
        from core.file_integrity import canonical_json_sha256
        path = store._path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_schema": 1, "identity": identity,
            "identity_sha256": canonical_json_sha256(identity),
            "observation": observation,
            "observation_sha256": canonical_json_sha256(observation),
            "origin": "historical_fixture",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return path

    def test_content_sha_changes_even_when_csv_length_is_unchanged(self):
        from services.audit import oracle_replay_artifacts as artifacts
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "2330.csv"
            csv.write_text("100,200\n", encoding="utf-8")
            with patch.object(artifacts, "discover_dataset_csv_inputs", return_value=([("2330", csv)], [])):
                before = artifacts.price_inventory_sha256(Path(tmp), "full")
                csv.write_text("100,201\n", encoding="utf-8")
                after = artifacts.price_inventory_sha256(Path(tmp), "full")
                self.assertNotEqual(before, after)
                self.assertEqual(after, artifacts.price_inventory_sha256(Path(tmp), "full"))

    def test_bugfix_replay_revision_invalidates_old_scores_without_deleting_history(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        from core.file_integrity import canonical_json_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = OracleReplayStore(root, "breakout_quality/reusable/target_oracle_romd")
            old = self._identity()
            old_result = self._observation("CG40_EqualRank", 1)
            store.write(old, old_result, origin="canonical_replay")
            original_path = store._path(old)
            original_bytes = original_path.read_bytes()
            corrected = dict(old, replay_policy={**old["replay_policy"],
                "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION})
            found, status = store.resolve_completed(corrected)
            self.assertIsNone(found)
            self.assertEqual(status, "none")
            self.assertFalse(any(
                (identity.get("replay_policy") or {}).get("oracle_parity_implementation")
                    == oracle.ORACLE_PARITY_IMPLEMENTATION
                for identity in store.existing_completed_identities(seed=1, target="CG40_EqualRank")
            ))
            new_result = {**old_result, "romd": old_result["romd"]+1.0}
            store.write(corrected, new_result, origin="canonical_replay")
            self.assertEqual(store.resolve_completed(corrected)[0], new_result)
            self.assertEqual(original_path.read_bytes(), original_bytes)
            self.assertEqual(store.read(old), old_result)
            self.assertTrue(any(
                (identity.get("replay_policy") or {}).get("oracle_parity_implementation")
                    == oracle.ORACLE_PARITY_IMPLEMENTATION
                for identity in store.existing_completed_identities(seed=1, target="CG40_EqualRank")
            ))
            # Same new revision, changed price SHA is a real conflict.
            another = dict(corrected, dataset_sha256="f"*64)
            with self.assertRaisesRegex(ValueError, "DIFFERENT identity"):
                store.resolve_completed(another)

    def test_equivalent_reference_dl_arm_does_not_change_replay_identity(self):
        from services.audit.oracle_replay_artifacts import replay_identity
        def make(arm_name, score_source, reference_profile):
            return replay_identity(
                dataset="full", dataset_sha256="a" * 64, target="T40",
                definition={"T40": "immutable Target"}, start="2021-01-01",
                end="2025-12-22", benchmark="end_to_end_v1",
                param_sha256="b" * 64, seed=1,
                cohort={"cohort_rows": 99, "research_cutoff": "2026-03-02"},
                ranking_policy="resource-aware-continuous-score-constrained-optimal",
                ranking_options={"preserve_k_r0": True},
                evaluation_mode="rolling", max_positions=10,
                replay_policy={"exact_arm": arm_name, "fixed_risk": 0.03,
                    "perfect_score_sha256": "c" * 64,
                    "canonical_dl_score_contract": {
                        "reference_arm_id": arm_name,
                        "reference_score_source_id": score_source,
                        "reference_profile_id": reference_profile,
                        "source_target_id": "canonical_t40",
                        "canonical_feature_eligible_keys_sha256": "d" * 64,
                        "perfect_score_mapping": "canonical_daily_truth_percentile_v2",
                    }},
            )
        first = make("C103", "CONT13CG_ROLL", "model_cg")
        second = make("C105", "CONT13AK_ROLL", "model_ak")
        self.assertEqual(first, second)
        self.assertNotIn("exact_arm", first["replay_policy"])
        self.assertNotIn("reference_arm_id",
                         first["replay_policy"]["canonical_dl_score_contract"])
        with tempfile.TemporaryDirectory() as tmp:
            store = oracle.OracleReplayStore(Path(tmp),
                "breakout_quality/reusable/target_oracle_romd")
            stored = dict(first, replay_policy={**first["replay_policy"],
                "exact_arm": "C103", "canonical_dl_score_contract": {
                    **first["replay_policy"]["canonical_dl_score_contract"],
                    "reference_arm_id": "C103",
                    "reference_score_source_id": "CONT13CG_ROLL",
                    "reference_profile_id": "model_cg",
                }})
            record = self._observation("T40", 1)
            store.write(stored, record, origin="old_v2_reference_fixture")
            self.assertEqual(store.resolve_completed(second)[0], record)
            # A changed actual score is not reference-only metadata.
            changed = dict(second, replay_policy={**second["replay_policy"],
                "perfect_score_sha256": "f" * 64})
            with self.assertRaisesRegex(ValueError, "DIFFERENT identity"):
                store.resolve_completed(changed)

    def test_corrupt_direct_target_does_not_block_exact_with_shared_prefix(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            direct = self._identity(target="R40_Direct")
            store.write(direct, self._observation("R40_Direct", 1), origin="canonical_replay")
            store._path(direct).write_text("{damaged", encoding="utf-8")
            exact = self._identity(target="R40")
            self.assertEqual(store.resolve_completed(exact), (None, "none"))
            self.assertEqual(store.plan_execution({(1, "R40"): exact}).action(1, "R40"), "RUN")
            self.assertEqual(store.existing_completed_identities(seed=1, target="R40"), ())
            with self.assertRaisesRegex(ValueError, "corrupt"):
                store.resolve_completed(direct)

    def test_canonical_identity_ignores_explanations_but_not_economic_input(self):
        from services.audit.oracle_replay_artifacts import replay_identity
        original = self._identity(target="R40")
        changed = replay_identity(
            dataset=original["dataset"], dataset_sha256=original["dataset_sha256"],
            target="R40", definition={"R40": "Report wording changed only"},
            start=original["start_date"], end=original["end_date"],
            benchmark=original["benchmark"], param_sha256=original["param_sha256"],
            seed=original["seed"],
            cohort={**original["cohort"], "scoring": "new prose in report"},
            ranking_policy=original["ranking_policy"],
            ranking_options=original["ranking_options"],
            evaluation_mode=original["evaluation_mode"],
            max_positions=original["max_positions"], replay_policy=original["replay_policy"],
        )
        self.assertEqual(changed, original)
        self.assertNotIn("scoring", changed["cohort"])
        self.assertEqual(changed["target_definition"], "R40")
        changed["cohort"]["cohort_rows"] += 1
        self.assertNotEqual(changed, original)

    def test_v2_display_only_identity_drifts_reuse_with_full_score_evidence(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            current = self._identity(target="R40")
            current = dict(current, replay_policy={
                **current["replay_policy"],
                "perfect_score_sha256": "1" * 64,
                "canonical_dl_score_contract": {"target_id": "R40", "score_start_date": "2021-01-01"},
            })
            old = dict(current, target_definition="40D target written for people",
                       cohort={**current["cohort"], "scoring": "legacy rendered explanation"})
            observation = self._observation("R40", 1)
            store.write(old, observation, origin="canonical_replay")
            old_path = store._path(old)
            old_bytes = old_path.read_bytes()
            reused, origin = store.resolve_completed(current)
            self.assertEqual(reused, observation)
            self.assertIn("legacy replay cache", origin)
            self.assertEqual(old_path.read_bytes(), old_bytes)
            # It is not enough to have the same description: score-value and
            # portfolio-economics changes must remain strict conflicts.
            changed = dict(current, replay_policy={**current["replay_policy"],
                "perfect_score_sha256": "2" * 64})
            with self.assertRaisesRegex(ValueError, "DIFFERENT identity"):
                store.resolve_completed(changed)
            changed_price = dict(current, dataset_sha256="3" * 64)
            with self.assertRaisesRegex(ValueError, "DIFFERENT identity"):
                store.resolve_completed(changed_price)

    def test_v2_baseline_display_only_identity_does_not_require_model_score(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            baseline = self._identity(target="C96")
            previous = dict(baseline, target_definition="Human baseline description",
                            cohort={**baseline["cohort"], "scoring": "old user text"})
            store.write(previous, self._observation("C96", 1), origin="canonical_replay")
            self.assertEqual(store.resolve_completed(baseline)[0], self._observation("C96", 1))

    def test_partial_unit_cache_reuses_only_identical_input_identity(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            cache = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            identity = self._identity()
            record = self._observation("CG40_EqualRank", 1)
            self.assertIsNone(cache.read(identity))
            cache.write(identity, record, origin="canonical_replay")
            self.assertEqual(cache.read(identity), record)
            cache.write(identity, record, origin="canonical_replay")  # idempotent
            self.assertIsNone(cache.read(self._identity(dataset_sha="c" * 64)))
            self.assertEqual(cache.plan_execution({(1, identity["target"]): self._identity(dataset_sha="c" * 64)}).action(1, identity["target"]), "BLOCKED")
            self.assertIsNone(cache.read(self._identity(param_sha="d" * 64)))
            with self.assertRaisesRegex(ValueError, "conflict"):
                cache.write(identity, {**record, "romd": 999}, origin="canonical_replay")
            location = cache._path(identity)
            raw = json.loads(location.read_text(encoding="utf-8"))
            raw["observation"]["romd"] = 999
            location.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                cache.read(identity)


    def test_conflict_reports_exact_fields_and_relative_cache_path_without_replay(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            cached = self._identity(target="C96", dataset_sha="a" * 64, param_sha="b" * 64)
            # Same dates and panel, but inputs and policy changed; never REUSE.
            requested = {
                **cached,
                "dataset_sha256": "c" * 64,
                "param_sha256": "d" * 64,
                "cohort": {**cached["cohort"], "cohort_rows": 124},
                "replay_policy": {**cached["replay_policy"], "exact_k_r0": False},
            }
            store.write(cached, self._observation("C96", 1), origin="canonical_replay")
            planned = store.plan_execution({(1, "C96"): requested})
            self.assertEqual(planned.action(1, "C96"), "BLOCKED")
            message = planned.blocked()[(1, "C96")]
            for field in ("dataset_sha256", "param_sha256", "cohort.cohort_rows",
                          "replay_policy.exact_k_r0"):
                self.assertIn(field, message)
            self.assertIn("outputs/audit/breakout_quality/reusable/target_oracle_romd/replay_cache/1/", message)
            self.assertIn("cache=aaaaaaaaaaaaaaaa...", message)
            self.assertIn("current=cccccccccccccccc...", message)
            self.assertIsNone(store.read(requested))
            self.assertEqual(store.read(cached), self._observation("C96", 1))
            self.assertEqual(store.plan_execution({(1, "C96"): requested}).blocked()[(1, "C96")], message)

    def test_legacy_comparison_digest_only_can_reuse_and_preserve_old_evidence(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            current = self._identity(target="C96")
            old = {**current, "cache_schema": 1, "strategy_settings_fingerprint": "7" * 64}
            record = self._observation("C96", 1)
            self._write_legacy_cache(store, old, record)
            legacy_path = store._path(old)
            original_bytes = legacy_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "read-only"):
                store.write(old, record, origin="illegal_legacy_rewrite")
            self.assertIsNone(store.read(current))
            self.assertEqual(store.plan_execution({(1, "C96"): current}).action(1, "C96"), "REUSE")
            reused, origin = store.resolve_completed(current)
            self.assertEqual(reused, record)
            self.assertIn("verified legacy replay cache", origin)
            store.write(current, reused, origin="verified_legacy_replay_cache")
            self.assertEqual(store.resolve_completed(current)[1], "verified per-arm cache")
            self.assertEqual(store.read(old), record)
            self.assertEqual(legacy_path.read_bytes(), original_bytes)

    def test_legacy_migration_rejects_any_replay_semantic_change(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        modifications = (
            {"dataset_sha256": "d" * 64},
            {"param_sha256": "e" * 64},
            {"target_definition": "different mathematical truth"},
            {"ranking_options": {"preserve_k_r0": False}},
            {"replay_policy": {"exact_k_r0": False}},
            {"cohort": {"cohort_rows": 124, "research_cutoff": "2026-03-02"}},
            {"max_positions": 9},
            {"end_date": "2025-12-23"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            base = self._identity(target="C96")
            old = {**base, "cache_schema": 1, "strategy_settings_fingerprint": "8" * 64}
            self._write_legacy_cache(store, old, self._observation("C96", 1))
            for changed in modifications:
                candidate = dict(base, **changed)
                if candidate == base:
                    continue
                # AI: A different time window is a separate cache cohort, not
                # AI: a same-window conflict, but must never be reused.
                if candidate["end_date"] != base["end_date"]:
                    self.assertEqual(store.resolve_completed(candidate), (None, "none"))
                else:
                    with self.subTest(changed=changed), self.assertRaisesRegex(
                        ValueError, "DIFFERENT identity"
                    ):
                        store.resolve_completed(candidate)
            self.assertEqual(store.plan_execution({(1, "C96"): base}).action(1, "C96"), "REUSE")

    def test_equivalent_legacy_caches_with_different_observations_fail_closed(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            identity = self._identity(target="C96")
            for digest, romd in (("a" * 64, 1.0), ("b" * 64, 99.0)):
                old = {**identity, "cache_schema": 1, "strategy_settings_fingerprint": digest}
                self._write_legacy_cache(store, old, {**self._observation("C96", 1), "romd": romd})
            with self.assertRaisesRegex(ValueError, "disagree on observations"):
                store.resolve_completed(identity)

    def test_incomplete_legacy_identity_is_not_admitted(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            current = self._identity(target="C96")
            old = {**current, "cache_schema": 1, "strategy_settings_fingerprint": "a" * 64}
            # AI: A missing broker/score policy is not an innocuous display change.
            old.pop("replay_policy")
            self._write_legacy_cache(store, old, self._observation("C96", 1))
            with self.assertRaisesRegex(ValueError, "DIFFERENT identity"):
                store.resolve_completed(current)

    def test_same_scored_arm_reuses_across_comparison_provider_inventories(self):
        # Switching from all targets to a Daily-only or single-price target
        # changes report provider diagnostics, not the injected score or Replay.
        from services.audit.oracle_replay_artifacts import replay_identity, OracleReplayStore
        shared = dict(
            dataset="full", dataset_sha256="a" * 64,
            target="DailyUniversal_MinTrade40_NetR_Direct",
            definition={"DailyUniversal_MinTrade40_NetR_Direct": "existing Target"},
            start="2021-01-01", end="2025-12-22", benchmark="end_to_end_v1",
            param_sha256="b" * 64, seed=1,
            ranking_policy="resource-aware-continuous-score-no-k-no-r0",
            ranking_options={"selection_only": True}, evaluation_mode="rolling",
            max_positions=10, replay_policy={
                "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
                "perfect_score_sha256": "c" * 64,
                "canonical_dl_score_contract": {
                    "target_id": "daily_universal_formal_min_roos_trade40_netr_v1",
                    "canonical_feature_eligible_keys_sha256": "d" * 64,
                },
            },
        )
        eligible = {"cohort_rows": 100, "first_common_day": "2021-01-04",
                    "last_common_day": "2025-12-22", "research_cutoff": "2026-03-02",
                    "risk_budget_return": 0.03, "oracle_only": True,
                    "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION}
        many = replay_identity(**shared, cohort={**eligible,
            "providers": {"price": "A", "market_truth": "B"},
            "market_truth": {"price_ticker_count": 2000}})
        daily_only = replay_identity(**shared, cohort={**eligible,
            "providers": {"daily_trade40": "B"},
            "scoring": "another report description"})
        self.assertEqual(many, daily_only)
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            result = self._observation(shared["target"], 1)
            store.write(many, result, origin="canonical_replay")
            self.assertEqual(store.resolve_completed(daily_only)[0], result)
            # An earlier V2 cache retained the provider summary in its key.
            old_v2 = dict(many, cohort={**eligible,
                "providers": {"full_compare": "diagnostic only"},
                "market_truth": {"ticker_count": 123}})
            with tempfile.TemporaryDirectory() as old_tmp:
                old_store = OracleReplayStore(Path(old_tmp), "breakout_quality/reusable/target_oracle_romd")
                old_store.write(old_v2, result, origin="canonical_replay")
                self.assertEqual(old_store.resolve_completed(daily_only)[0], result)
                actual_score_change = dict(daily_only, replay_policy={
                    **daily_only["replay_policy"], "perfect_score_sha256": "f" * 64})
                with self.assertRaisesRegex(ValueError, "DIFFERENT identity"):
                    old_store.resolve_completed(actual_score_change)
                changed_period = dict(daily_only, end_date="2025-12-19")
                self.assertIsNone(old_store.resolve_completed(changed_period)[0])

    def test_unrelated_maturity_panel_does_not_block_baseline_reuse(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            identity = self._identity(target="C96")
            different_panel = {
                **identity,
                "cohort": {**identity["cohort"], "providers": {"horizon": "60D"}},
            }
            store.write(different_panel, self._observation("C96", 1), origin="canonical_replay")
            self.assertEqual(store.plan_execution({(1, "C96"): identity}).action(1, "C96"), "REUSE")

    def test_prefix_distinct_targets_are_not_cache_conflicts(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            direct = self._identity(target="R40_Direct")
            store.write(direct, self._observation("R40_Direct", 1), origin="canonical_replay")
            self.assertEqual(store.plan_execution({(1, "R40"): self._identity(target="R40")}).action(1, "R40"), "RUN")


    def test_corrupt_archived_report_cannot_block_or_supply_canonical_replay(self):
        seeds = (11, 22, 33, 44)
        definitions = dict(LEGACY_ORACLE_TARGETS)
        common = {"cohort_rows": 4, "first_common_day": "2021-01-01",
                  "last_common_day": "2025-12-22", "research_cutoff": "2026-03-02",
                  "risk_budget_return": 0.03, "providers": {}, "scoring": "test",
                  "market_truth": {}, "rows_outside_dl_training_provider_by_source": {},
                  "ratio_zero_denominator_policy": "D40=0,MFE40>0 -> +infinity (highest rank); 0/0 -> 0; final score = finite daily percentile",
                  "mfe40_over_d40_zero_downside_positive_upside_count": 0,
                  "mfe40_over_d40_zero_zero_count": 0}
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        seen = []
        def replay(*, name, **kwargs):
            seen.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        definition = _legacy_definition()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd/20261023T223000Z"
            previous.mkdir(parents=True)
            (previous / "audit.json").write_text("{corrupt archived report", encoding="utf-8")
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                result = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(seen), 40, "old baseline yearly flags and Oracle scores must be rerun")
                self.assertEqual(sum(name.startswith("C96") for name in seen), 4)
                self.assertEqual(sum(name.startswith("Oracle ") for name in seen), len(seen) - 4)
                self.assertEqual([r["target"] for r in result["oracles"]], list(definitions))
                second = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(seen), 40, "corrected per-arm caches must be reused")
                self.assertEqual(second["oracles"], result["oracles"])

    def test_completed_v4_ten_target_report_reuses_all_seven_current_arms(self):
        # Resizing the current study must never repeat seven scientifically
        # identical historical observations or the same-window C96 control.
        from core.report_contract_versions import load_report_contract_versions
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        definitions = dict(LEGACY_ORACLE_TARGETS)
        old_definitions = {**self._old_definitions(), **definitions}
        # Old V4 had the ten active controls, not all 11 legacy controls.
        old_order = ("AO40_EqualRank", "CG40_EqualRank", "U40", "FirstBreach_NetR",
                     "MFE40_Control", "R40", "T40", "MFE40_Direct", "R40_Direct", "T40_Direct")
        old_definitions = {name: old_definitions[name] for name in old_order}
        old_report = {
            "schema_version": 5, "status": "COMPLETED", "read_only": True,
            "hindsight_non_pit": True, "report_id": "target_oracle_romd",
            "report_type": oracle.SUPPORTED_AUDIT_TYPE,
            "report_contract_id": "audit.target_oracle_romd@v4",
            "report_contract_fingerprint": load_report_contract_versions()["audit.target_oracle_romd"][3]["fingerprint"],
            "strategy_profile_id": "broker_cash_dl_conversion_rolling",
            "benchmark_id": Benchmark.benchmark_id, "dataset": "full",
            "start_date": "2021-01-01", "end_date": "2025-12-22",
            "seeds": list(seeds), "common_cohort": common,
            "param_artifacts": [{"seed": seed, "param_sha256": "a" * 64} for seed in seeds],
            "definitions": old_definitions,
            "baseline": oracle._aggregate([self._observation("C96", seed) for seed in seeds], target="C96"),
            "oracles": [oracle._aggregate([self._observation(name, seed) for seed in seeds], target=name)
                        for name in old_order],
        }
        from core.audit_policy import get_reusable_audit_definitions
        definition = _legacy_definition()
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd/20261024T110500Z"
            previous.mkdir(parents=True)
            (previous / "audit.json").write_text(json.dumps(old_report), encoding="utf-8")
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=lambda *, name, **kwargs: (seen.append(name), {"name": name})[1]), \
                 patch.object(oracle, "_metrics", side_effect=lambda replay, *, target: self._observation(target, int(replay["name"].split("seed=")[1].split()[0]))):
                rebuilt = oracle.build_report(definition, project_root=root, quiet=True)
        self.assertEqual(len(seen), 40, "historical baseline/yearly and Oracle predate parity fix")
        self.assertEqual(sum(name.startswith("C96") for name in seen), 4)
        self.assertEqual(sum(name.startswith("Oracle ") for name in seen), len(seen) - 4)
        self.assertEqual([x["target"] for x in rebuilt["oracles"]], list(definitions))
        self.assertEqual(rebuilt["schema_version"], 8)

    def test_incompatible_old_report_cannot_block_or_supply_corrected_oracle_results(self):
        from core.report_contract_versions import load_report_contract_versions
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        definitions = dict(LEGACY_ORACLE_TARGETS)
        report = {
            "schema_version": 5, "status": "COMPLETED", "read_only": True,
            "hindsight_non_pit": True, "report_id": "target_oracle_romd",
            "report_type": oracle.SUPPORTED_AUDIT_TYPE,
            "report_contract_id": "audit.target_oracle_romd@v4",
            "report_contract_fingerprint": load_report_contract_versions()["audit.target_oracle_romd"][3]["fingerprint"],
            "strategy_profile_id": "broker_cash_dl_conversion_rolling",
            "benchmark_id": Benchmark.benchmark_id, "dataset": "full",
            "start_date": "2021-01-01", "end_date": "2025-12-22",
            "seeds": list(seeds), "common_cohort": common,
            "param_artifacts": [{"seed": seed, "param_sha256": "c" * 64} for seed in seeds],
            "definitions": definitions,
            "baseline": oracle._aggregate([self._observation("C96", seed) for seed in seeds], target="C96"),
            "oracles": [oracle._aggregate([self._observation(name, seed) for seed in seeds], target=name)
                        for name in definitions],
        }
        definition = _legacy_definition()
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd/20261024T110500Z"
            directory.mkdir(parents=True)
            previous = directory / "audit.json"
            old_bytes = json.dumps(report).encode("utf-8")
            previous.write_bytes(old_bytes)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=lambda *, name, **kwargs: (seen.append(name), {"name": name})[1]), \
                 patch.object(oracle, "_metrics", side_effect=lambda result, *, target: self._observation(target, int(result["name"].split("seed=")[1].split()[0]))):
                revised = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(seen), 40, "old param/score report must not supply the new results")
                self.assertEqual([row["target"] for row in revised["oracles"]], list(definitions))
                self.assertEqual(previous.read_bytes(), old_bytes, "historical evidence must remain unchanged")
                again = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(seen), 40, "corrected cache is now reusable")
                self.assertEqual(again["oracles"], revised["oracles"])

    def test_partial_replay_is_saved_before_later_failure_and_resumes(self):
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        definition = _legacy_definition()
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        state = {"calls": 0, "crash": True}
        def replay(*, name, **kwargs):
            state["calls"] += 1
            if state["crash"] and state["calls"] == 10:
                raise RuntimeError("injected replay interruption")
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                with self.assertRaisesRegex(RuntimeError, "injected replay interruption"):
                    oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(state["calls"], 10)
                state["crash"] = False
                resumed = oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(state["calls"], 41, "the first nine completed arms must not replay")
                self.assertEqual(len(resumed["oracles"]), len(LEGACY_ORACLE_TARGETS))
                oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(state["calls"], 41)

    def test_compare_matrix_display_changes_do_not_replay_oracle(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        definition = _legacy_definition()
        settings = get_strategy_comparison_settings(str(definition.source["strategy_profile_id"]))
        modified_arms = dict(settings.arms)
        modified_arms["C100"] = replace(settings.arms["C100"], enabled=True,
                                        name="new display title", description="new comparison-only description")
        modified_arms["C96"] = replace(settings.arms["C96"], name="renamed control")
        modified_sources = dict(settings.parameter_sources)
        baseline_source = settings.arms["C96"].param_source
        modified_sources[baseline_source] = replace(settings.parameter_sources[baseline_source],
                                                    description="new catalog documentation")
        display_changed = replace(settings, arms=modified_arms, parameter_sources=modified_sources)
        calls = []
        def run(*, name, **kwargs):
            calls.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=run), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                report = oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(len(calls), 40)
                with patch.object(oracle, "get_strategy_comparison_settings", return_value=display_changed):
                    reused = oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(len(calls), 40, "presentation/membership must not cause new replay")
                self.assertEqual(reused["oracles"], report["oracles"])

    def test_legacy_c96_reuse_in_complete_four_seed_report_without_replay(self):
        from core.file_integrity import canonical_json_sha256
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        definition = _legacy_definition()
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        runs = []
        def replay(*, name, **kwargs):
            runs.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                original = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(runs), 40)
                store = OracleReplayStore(root, definition.output_subdir)
                canonical_path, = tuple((store.cache_root / "11").glob("C96_*.json"))
                current_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
                legacy_identity = {
                    **current_payload["identity"], "cache_schema": 1,
                    "strategy_settings_fingerprint": "7e7058c39ad23498" + "0" * 48,
                }
                legacy_payload = {
                    **current_payload, "cache_schema": 1, "identity": legacy_identity,
                    "identity_sha256": canonical_json_sha256(legacy_identity),
                }
                legacy_path = store._path(legacy_identity)
                legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
                original_legacy_bytes = legacy_path.read_bytes()
                canonical_path.unlink()  # AI: Fixture: old cache exists, new key does not.
                rebuilt = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(runs), 40, "verified old C96 must not run")
                self.assertEqual(rebuilt["oracles"], original["oracles"])
                self.assertEqual(rebuilt["baseline"], original["baseline"])
                self.assertEqual(legacy_path.read_bytes(), original_legacy_bytes)
                self.assertFalse(canonical_path.exists(), "Verified legacy REUSE needs no artifact rewrite")
                self.assertEqual(oracle.build_report(definition, project_root=root, quiet=True)["baseline"],
                                 original["baseline"])
                self.assertEqual(len(runs), 40)

    def test_conflicting_replay_cache_blocks_before_any_trade_or_report(self):
        # AI: Scientific identity checks, not additional user input, are the
        # AI: only preflight Replay gate after the Research menu's one Enter.
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        definition = _legacy_definition()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels({"cohort_rows": 4, "last_common_day": "2025-12-22"})) as panels, \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle.OracleReplayStore, "resolve_completed", side_effect=ValueError("incompatible cache identity")), \
                 patch("builtins.input", side_effect=AssertionError("domain must not prompt")) as user_input, \
                 patch.object(oracle, "_run_scenario") as replay, \
                 patch.object(oracle, "persist_reusable_report") as persist:
                with self.assertRaisesRegex(oracle.OracleBlockedError, "BLOCKED"):
                    oracle.run_formal_audit(definition, project_root=root, quiet=True)
                self.assertEqual(panels.call_count, 1)
                user_input.assert_not_called()
                replay.assert_not_called()
                persist.assert_not_called()
                self.assertFalse((root / "outputs/audit").exists())

    def test_completed_schema6_report_restores_32_old_units_and_only_runs_new_target(self):
        from core.report_contract_versions import load_report_contract_versions
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        common40 = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        recorded = []
        def replay(*, name, **kwargs):
            recorded.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        definition = _legacy_definition()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common40)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                report = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(recorded), 40)
                # Emulate a genuine historical v6 report with seven treatments.
                new_targets = {"T40-MFE", "T40-MFE_Direct"}
                for name in new_targets:
                    report["definitions"].pop(name)
                report["oracles"] = [row for row in report["oracles"] if row["target"] not in new_targets]
                report.update({
                    "report_id": "target_oracle_romd", "report_type": oracle.SUPPORTED_AUDIT_TYPE,
                    "schema_version": 6,
                    "report_contract_id": "audit.target_oracle_romd@v6",
                    "report_contract_fingerprint": load_report_contract_versions()["audit.target_oracle_romd"][5]["fingerprint"],
                })
                directory = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd"
                snapshot = directory / "20261024T104500Z"
                snapshot.mkdir(parents=True)
                (snapshot / "audit.json").write_text(json.dumps(report), encoding="utf-8")
                shutil.rmtree(directory / "replay_cache")
                cache = oracle.OracleReplayStore(root, definition.output_subdir)
                self.assertTrue(all(
                    not cache.existing_completed_identities(seed=seed, target=name)
                    for seed in seeds
                    for name in ("C96", *(item for item, _ in LEGACY_ORACLE_TARGETS))
                ), "historical reports must not masquerade as replay cache")
                rebuilt = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(recorded), 80, "old Oracle scores and baseline yearly statistics are invalid")
                self.assertEqual([row for row in rebuilt["oracles"] if row["target"] not in new_targets], report["oracles"])
                self.assertTrue(new_targets <= {row["target"] for row in rebuilt["oracles"]})
                console = io.StringIO()
                with redirect_stdout(console):
                    oracle.build_report(definition, project_root=root, quiet=False,
                                        preflight_display="summary")
                self.assertEqual(len(recorded), 80)
                self.assertIn("身分核對完成] REUSE 40 / RUN（需計算）0 / BLOCKED 0", console.getvalue())
                self.assertNotIn("Replay preflight（執行前）", console.getvalue())
                self.assertNotIn("[REUSE]", console.getvalue())

    def test_completed_schema7_reuses_all_36_old_units_and_runs_only_mfe_direct(self):
        from core.report_contract_versions import load_report_contract_versions
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        recorded = []
        def replay(*, name, **kwargs):
            recorded.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        definition = _legacy_definition()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                current = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(recorded), 40)
                # The prior completed report is a physically separate historical
                # snapshot; only its missing current Direct Replay is calculated.
                old = {**current,
                       "definitions": {name: desc for name, desc in current["definitions"].items() if name != "T40-MFE_Direct"},
                       "oracles": [row for row in current["oracles"] if row["target"] != "T40-MFE_Direct"],
                       "report_id": "target_oracle_romd",
                       "report_type": oracle.SUPPORTED_AUDIT_TYPE,
                       "schema_version": 7,
                       "report_contract_id": "audit.target_oracle_romd@v7",
                       "report_contract_fingerprint": load_report_contract_versions()["audit.target_oracle_romd"][6]["fingerprint"]}
                old_dir = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd/20261024T154600Z"
                old_dir.mkdir(parents=True)
                (old_dir / "audit.json").write_text(json.dumps(old), encoding="utf-8")
                cache_root = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd/replay_cache"
                shutil.rmtree(cache_root)
                self.assertTrue(all(
                    not oracle.OracleReplayStore(root, definition.output_subdir).existing_completed_identities(
                        seed=seed, target=name,
                    )
                    for seed in seeds
                    for name in ("C96", *(item for item, _ in LEGACY_ORACLE_TARGETS))
                ))
                previous = len(recorded)
                report = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(recorded) - previous, 40)
                self.assertEqual(sum(name.startswith("C96") for name in recorded[previous:]), 4)
                self.assertEqual(sum(name.startswith("Oracle ") for name in recorded[previous:]), 36)
                self.assertEqual([x for x in report["oracles"] if x["target"] != "T40-MFE_Direct"], old["oracles"])
                previous = len(recorded)
                again = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(recorded), previous)
                self.assertEqual(again["oracles"], report["oracles"])




class OracleUnifiedReuseArchitectureCases(TestCase):
    """Invariants required by the Research [1] single-plan REUSE contract."""

    def test_execution_plan_seals_reuse_without_invoking_producer(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        store = OracleReplayStore(Path("/tmp/oracle_plan_synthetic"), "synthetic")
        identities = {
            (11, "C96"): {"seed": 11, "target": "C96"},
            (11, "R40"): {"seed": 11, "target": "R40"},
        }
        observation = {"seed": 11, "target": "C96", "romd": 4.0}
        with patch.object(store, "resolve_completed", side_effect=[
                (observation, "verified per-arm cache"), (None, "none")]) as resolver, \
             patch.object(store, "write") as writer:
            plan = store.plan_execution(identities)
            self.assertEqual(plan.actions(), {(11, "C96"): "REUSE", (11, "R40"): "RUN"})
            self.assertEqual(resolver.call_count, 2)
            cached, _ = plan.execute(11, "C96", lambda: self.fail("REUSE must not produce"))
            self.assertEqual(cached, observation)
            self.assertEqual(writer.call_count, 0)
            fresh, _ = plan.execute(11, "R40", lambda: {"target": "R40", "seed": 11, "romd": 6.0})
            self.assertEqual(fresh["romd"], 6.0)
            self.assertEqual(writer.call_count, 1)
            self.assertEqual(resolver.call_count, 2, "No re-decision at execution")

    def test_execution_plan_blocks_all_runs_if_one_identity_is_conflicting(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        store = OracleReplayStore(Path("/tmp/oracle_plan_synthetic"), "synthetic")
        with patch.object(store, "resolve_completed", side_effect=[
                 (None, "none"), ValueError("scientific SHA conflict")]), \
             patch.object(store, "write") as writer:
            plan = store.plan_execution({
                (11, "C96"): {"seed": 11, "target": "C96"},
                (11, "R40"): {"seed": 11, "target": "R40"},
            })
            self.assertEqual(plan.action(11, "R40"), "BLOCKED")
            with self.assertRaisesRegex(ValueError, "BLOCKED"):
                plan.execute(11, "C96", lambda: self.fail("No partial batch Replay"))
            writer.assert_not_called()

    def test_daily_truth_build_requires_approval_and_cached_truth_skips_trade_path(self):
        from services.audit import daily_trade40_oracle as src
        from core.breakout_quality_registry import TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
        calendar = pd.DatetimeIndex(["2020-12-31", "2021-01-04", "2021-01-05"])
        info_day = pd.Timestamp("2021-01-04")
        group = pd.DataFrame({"ticker": ["2330"], "date": [info_day]})
        profile = SimpleNamespace(
            training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
            model_architecture="inception_time_v1", continuous_target_id="other",
        )
        bundle = SimpleNamespace(summary={"dataset": "full", "filter_id": "test",
            "sample_scope": "daily_eligible_stock_days", "score_eligible_sample_count": 1},
            group_table=group)
        prices = pd.DataFrame({"Open": [100.0, 100.0, 100.0],
            "High": [101.0, 101.0, 101.0], "Low": [99.0, 99.0, 99.0],
            "Close": [100.0, 100.0, 100.0], "Volume": [1000000]*3}, index=calendar)
        simulated = SimpleNamespace(
            reasons=["settled"], maturity_dates=[pd.Timestamp("2021-01-05")],
            target_raw_r=np.asarray([2.0]), valid_mask=np.asarray([True]),
            cutoff_truncated_rows=0, research_cutoff_closeout_rows=0,
        )
        period = SimpleNamespace(
            source_id="formal_min_roos_seed", start=pd.Timestamp("2021-01-01"),
            end=None, params_signature="P16-test",
        )
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(src, "get_breakout_quality_experiment_profile", return_value=profile), \
             patch.object(src, "load_profile_continuous_ranker_data", return_value=bundle) as cohort_loader, \
             patch.object(src, "_truth_source_sha", return_value="d"*64), \
             patch.object(src, "discover_dataset_csv_inputs", return_value=([("2330", Path(temp)/"2330.csv")], [])), \
             patch.object(src, "collect_canonical_exchange_sessions", return_value=calendar), \
             patch.object(src, "load_dataset_frame", return_value=prices), \
             patch.object(src, "compute_counterfactual_trade40_batch", return_value=simulated) as compute:
            args = dict(project_root=Path(temp), dataset="full", filter_id="test",
                profile_id="daily-example", cutoff="2021-01-05",
                evaluation_start="2021-01-04", evaluation_end="2021-01-04",
                score_start="2021-01-04", seed=11, param_sha256="a"*64,
                canonical_schedule=(period,), horizon_sessions=40,
                price_sha256="b"*64, cache_root=Path(temp)/"truth_cache",
                feature_cohort_cache={},
            )
            with self.assertRaises(src.DailyTrade40TruthBuildRequired):
                src.build_daily_universal_trade40_oracle_scores(**args, allow_build=False)
            compute.assert_not_called()
            first, evidence = src.build_daily_universal_trade40_oracle_scores(**args)
            self.assertEqual(evidence["source_reuse_status"], "BUILD")
            self.assertEqual(compute.call_count, 1)
            cached, evidence2 = src.build_daily_universal_trade40_oracle_scores(**args, allow_build=False)
            self.assertEqual(cached, first)
            self.assertEqual(evidence2["source_reuse_status"], "REUSE")
            self.assertEqual(compute.call_count, 1, "Certified truth REUSE must not trade")
            self.assertEqual(cohort_loader.call_count, 1, "Feature cohort is shared per run")

    def test_price_score_cache_reuses_certified_matrix_and_blocks_corruption(self):
        from services.audit.oracle_score_artifacts import OraclePriceScoreStore
        from services.audit.target_oracle_romd import _canonical_score_map_sha256
        identity = {"schema": 1, "dataset": "full", "algorithm": "synthetic"}
        panel = {"R40": {("2330", "2021-01-04"): 0.3,
                          ("2317", "2021-01-04"): 0.9}}
        cohort = {"cohort_rows": 2, "first_common_day": "2021-01-04",
                  "last_common_day": "2021-01-04"}
        with tempfile.TemporaryDirectory() as temp:
            store = OraclePriceScoreStore(Path(temp))
            self.assertIsNone(store.read(identity, score_sha=_canonical_score_map_sha256))
            store.write(identity, panel, cohort, score_sha=_canonical_score_map_sha256)
            self.assertEqual(store.read(identity, score_sha=_canonical_score_map_sha256), (panel, cohort))
            with open(store._path(identity), "wb") as stream:
                stream.write(b"corrupt scientific evidence")
            with self.assertRaisesRegex(ValueError, "BLOCKED"):
                store.read(identity, score_sha=_canonical_score_map_sha256)



    def test_formal_single_stage_dispatch_has_one_approval(self):
        calls = []
        def planned(_definition, **kwargs):
            calls.append(kwargs)
            self.assertTrue(kwargs["preflight_approval"]("FORMAL REUSE/RUN/BLOCKED"))
            return {"status": "COMPLETED"}
        with patch.object(oracle, "_run_single_stage_oracle", side_effect=planned):
            displays = []
            result = oracle.build_report(
                _ACTIVE_ORACLE_DEFINITION, project_root=Path("/tmp"), quiet=True,
                preflight_approval=lambda view: displays.append(view) or True,
            )
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(displays, ["FORMAL REUSE/RUN/BLOCKED"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["allow_historical_execution_input_drift"])

    def test_c96_can_be_certified_before_truth_using_real_cache_identity(self):
        from types import SimpleNamespace
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        definition = _ACTIVE_ORACLE_DEFINITION
        settings, base_arm, exact_arm = oracle._contract(definition)
        seed, price_sha, param_sha = 693545351, "a" * 64, "b" * 64
        identity_arm = exact_arm or settings.arms[definition.source["exact_arm_id"]]
        policy = oracle._arm_replay_policy(
            definition, plan={"selectors": {}, "daily_sources": {}}, target="C96",
            replay_policy=oracle._canonical_oracle_replay_policy(
                definition, base_arm=base_arm, exact_arm=identity_arm,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = replay_identity(
                dataset=settings.dataset, dataset_sha256=price_sha,
                target="C96", definition={"C96": "Canonical same-window Min P16 Base-Finalist-Best"},
                start=settings.start_date, end=definition.dimensions["oracle_evaluation_end_date"],
                benchmark="end_to_end_v1", param_sha256=param_sha, seed=seed,
                cohort={"cohort_rows": 608204, "research_cutoff": settings.end_date},
                ranking_policy=identity_arm.dl_runtime_mode,
                ranking_options=dict(identity_arm.dl_runtime_options),
                evaluation_mode=definition.dimensions["evaluation_mode"],
                max_positions=settings.max_positions, replay_policy=policy,
            )
            OracleReplayStore(root, definition.output_subdir).write(
                identity, {
                    "target": "C96", "seed": seed, "total_return_pct": 100.0,
                    "annualized_return_pct": 16.0, "mdd_pct": 12.0, "romd": 8.0,
                    "ev_a_r": 0.7, "trades": 100, "exposure_pct": 75.0, "win_rate_pct": 56.0,
                    "yearly": _synthetic_full_window_yearly(return_pct=14.0),
                }, origin="synthetic_baseline_fixture",
            )
            args = dict(
                project_root=root, settings=settings, base_arm=base_arm, exact_arm=exact_arm,
                robustness=SimpleNamespace(benchmark_id="end_to_end_v1"),
                prepared_parameters=[(seed, param_sha, "synthetic", None, None)],
                evaluation_end=definition.dimensions["oracle_evaluation_end_date"],
            )
            mapped = {}
            self.assertEqual(oracle._certify_unscored_baseline_before_truth(
                definition, price_sha=price_sha, certified_identity_out=mapped, **args),
                {seed: "REUSE"})
            self.assertEqual(mapped[seed], identity)
            # Changed price inputs are a new authorized execution identity;
            # the old signed receipt remains historical, never falsely REUSE.
            self.assertEqual(oracle._certify_unscored_baseline_before_truth(
                definition, price_sha="f" * 64,
                allow_historical_execution_input_drift=True, **args), {seed: "RUN"})
            self.assertEqual(oracle._certify_unscored_baseline_before_truth(
                definition, price_sha="f" * 64, **args), {seed: "BLOCKED"})
            # A scored target is never authorized by this C96 proof.
            self.assertEqual(set(mapped), {seed})

    def test_formal_blocked_preflight_never_requests_authorization(self):
        def blocked(_definition, **kwargs):
            raise oracle.OracleBlockedError("source identity BLOCKED")
        with patch.object(oracle, "_run_single_stage_oracle", side_effect=blocked):
            with self.assertRaisesRegex(oracle.OracleBlockedError, "BLOCKED"):
                oracle.build_report(
                    _ACTIVE_ORACLE_DEFINITION, project_root=Path("/tmp"), quiet=True,
                    preflight_approval=lambda _view: self.fail("BLOCKED must not request approval"),
                )


    def test_formal_preflight_approval_passes_once_and_keeps_one_plan(self):
        from services.audit import runner
        seen = []
        def fake_audit(_definition, *, project_root, quiet, preflight_approval):
            seen.append(preflight_approval("canonical matrix"))
            return {"status": "CANCELLED"}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner, "collect_audit_status", return_value={"overall_status": "READY"}), \
             patch.object(oracle, "run_formal_audit", side_effect=fake_audit), \
             patch("builtins.input", side_effect=AssertionError("Runner must not prompt")):
            result = runner.run_enabled_audits(
                "breakout_quality", project_root=Path(tmp),
                method_id=oracle.SUPPORTED_AUDIT_TYPE,
                preflight_approval=lambda text: text == "canonical matrix",
            )
            self.assertEqual(seen, [True])
            self.assertEqual(result["target_oracle_romd"]["status"], "CANCELLED")



class OracleReplayRunReasonEvidenceCases(TestCase):
    """Read-only source reasons; never grant REUSE from a diagnostic."""

    def test_signed_inventory_reports_missing_and_changed_inputs_without_replay(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        from services.audit import target_oracle_romd as target
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definition, benchmark, seed, price_sha, param_sha, _proof, _sample = (
                OracleOnePlanRealCacheCases()._fixture(root)
            )
            store = OracleReplayStore(root, definition.output_subdir)
            settings, _base, _exact = target._contract(definition)
            expected = {
                "start_date": str(settings.start_date),
                "end_date": str(definition.dimensions["oracle_evaluation_end_date"]),
                "dataset": settings.dataset,
                "dataset_sha256": price_sha,
                "param_sha256": param_sha,
                "benchmark": benchmark.benchmark_id,
                "replay_policy": {
                    "oracle_parity_implementation": target.ORACLE_PARITY_IMPLEMENTATION,
                    "period_stats_implementation": target.PERIOD_STATS_IMPLEMENTATION,
                },
            }
            diag = {}
            candidates, conflict = store.inspect_current_source(
                seed=seed, target="R40", expected_inputs=expected,
                diagnostic_out=diag,
            )
            self.assertEqual(conflict, None)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(diag["signed_receipts"], 1)
            self.assertEqual(diag["source_candidates"], 1)
            self.assertEqual(diag["differing_fields"], ())
            diag_wrong = {}
            current, conflict = store.inspect_current_source(
                seed=seed, target="R40",
                expected_inputs={**expected, "param_sha256": "f"*64},
                diagnostic_out=diag_wrong,
                allow_historical_execution_input_drift=True,
            )
            self.assertEqual(current, ())
            self.assertIsNone(conflict)
            self.assertEqual(diag_wrong["signed_receipts"], 1)
            self.assertIn("param_sha256", diag_wrong["differing_fields"])
            self.assertIn("param_sha256", target._source_run_diagnostic(diag_wrong))
            diag_none = {}
            missing, conflict = store.inspect_current_source(
                seed=seed + 12, target="R40",
                expected_inputs=expected, diagnostic_out=diag_none,
            )
            self.assertEqual((missing, conflict), ((), None))
            self.assertEqual(diag_none["signed_receipts"], 0)
            self.assertEqual(target._source_run_diagnostic(diag_none),
                             "沒有逐筆簽章 Replay 收據")
            # A source diagnostic never changes an identity, writes a replay,
            # or invents a scored observation.
            self.assertEqual(len(store.existing_completed_identities(
                seed=seed, target="R40")), 1)

    def test_reason_summary_does_not_change_preflight_actions(self):
        from services.audit import target_oracle_romd as target
        seeds = [1, 2]
        items = ["C96", "R40"]
        decisions = {(1,"C96"):"REUSE",(2,"C96"):"REUSE",
                     (1,"R40"):"RUN",(2,"R40"):"RUN"}
        reasons = {(1,"R40"):"沒有逐筆簽章 Replay 收據",
                   (2,"R40"):"沒有逐筆簽章 Replay 收據"}
        line = target._render_reuse_run_reasons(decisions=decisions, reasons=reasons)
        self.assertEqual(line.count("沒有逐筆簽章 Replay 收據"), 1)
        self.assertIn(" 2", line)
        preview = target.render_replay_preflight(
            label="Fixed", seeds=seeds, targets=items, decisions=decisions,
            reuse_diagnostic=line,
        )
        self.assertIn("REUSE 2 / RUN（需回放）2", preview)
        self.assertIn("[Target REUSE 核對原因｜唯讀]", preview)
        self.assertNotIn("[Target REUSE 核對原因｜唯讀]", target.render_replay_preflight(
            label="Fixed", seeds=seeds, targets=items, decisions=decisions,
        ))
        self.assertEqual(decisions[(1,"R40")],"RUN")


class OracleOnePlanRealCacheCases(TestCase):
    """Actual signed replay receipts, a single pre-execution table and NO truth on reuse."""

    def _fixture(self, root):
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        original = _ACTIVE_ORACLE_DEFINITION
        definition = replace(original, dimensions={**original.dimensions,
            "oracle_evaluation_end_date": "2025-12-22"},
            source={**original.source, "oracle_comparisons": tuple(
            row for row in original.source["oracle_comparisons"]
            if row["target"] in {"R40", "T40"})})
        settings, base, exact = oracle._contract(definition)
        plan = oracle._oracle_comparison_plan(definition)
        contracts = oracle._resolve_oracle_target_score_contracts(definition, settings, plan)
        bench = SimpleNamespace(benchmark_id="end_to_end_v1")
        seed, price_sha, param_sha = 693545351, "a" * 64, "b" * 64
        start, end = str(settings.start_date), str(definition.dimensions["oracle_evaluation_end_date"])
        source_proof = ("c" * 64, 20, start, end)
        cohort = {
            "cohort_rows": 20, "first_common_day": start,
            "last_common_day": end, "research_cutoff": settings.end_date,
            "risk_budget_return": float(oracle.StrategyAlignedContinuousTargetSpec.from_label_policy(
                oracle.DEFAULT_LABEL_POLICY).risk_budget_return),
            "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
            "oracle_only": True,
        }
        common = oracle._canonical_oracle_replay_policy(
            definition, base_arm=base, exact_arm=exact,
        )
        store = OracleReplayStore(root, definition.output_subdir)
        sample = lambda target, romd, diff=None: {
            "target": target, "seed": seed,
            "total_return_pct": 100.0, "annualized_return_pct": 16.0,
            "mdd_pct": 12.0, "romd": romd, "ev_a_r": 0.7,
            "trades": 100, "exposure_pct": 75.0, "win_rate_pct": 56.0,
            "yearly": _synthetic_full_window_yearly(return_pct=14.0),
            **({} if diff is None else {"delta_romd_vs_c96": diff}),
        }
        for target, romd in (("C96", 8.0), ("R40", 18.0)):
            args = {} if target == "C96" else dict(
                score_sha256="d" * 64,
                score_contract={**contracts[target],
                                "canonical_feature_eligible_keys_sha256": source_proof[0],
                                "oracle_scored_rows": source_proof[1],
                                "source_evidence": "validated_generic_daily_feature_universe_and_research_cutoff_closeout"},
            )
            arm_policy = oracle._arm_replay_policy(
                definition, plan=plan, target=target, replay_policy=common,
                project_root=root, **args,
            )
            identity = replay_identity(
                dataset=settings.dataset, dataset_sha256=price_sha,
                target=target, definition={target: target},
                start=start, end=end, benchmark=bench.benchmark_id,
                param_sha256=param_sha, seed=seed, cohort=cohort,
                ranking_policy=str(exact.dl_runtime_mode),
                ranking_options=dict(exact.dl_runtime_options),
                evaluation_mode=definition.dimensions["evaluation_mode"],
                max_positions=settings.max_positions, replay_policy=arm_policy,
            )
            store.write(identity, sample(target, romd, 10.0 if target != "C96" else None),
                        origin="synthetic_certified_existing_replay")
        return definition, bench, seed, price_sha, param_sha, source_proof, sample

    def test_before_first_enter_certifies_reuse_and_only_run_missing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definition, bench, seed, price_sha, param_sha, proof, sample = self._fixture(root)
            prompts = []
            planned = []
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(bench, [(seed, root/"p16.json")])), \
                 patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                     [(seed, param_sha, "synthetic", None, None)], {})), \
                 patch.object(oracle, "price_inventory_sha256", return_value=price_sha), \
                 patch.object(oracle, "_read_only_score_keys", side_effect=AssertionError(
                     "REUSE and warm missing-only planning must NOT recalculate 617 price labels")), \
                 patch.object(oracle, "load_profile_continuous_ranker_data", side_effect=AssertionError(
                     "REUSE must not build Daily DL target labels")), \
                 patch.object(oracle, "_execute_single_stage_missing", side_effect=lambda *a, **kw:
                     planned.append(kw["decisions"].copy()) or {(seed,"T40"): sample("T40", 20., 12.)}), \
                 patch.object(oracle, "_load_or_build_oracle_price_scores", side_effect=AssertionError(
                     "No price/target producer before authorization")):
                cancelled = oracle.build_report(
                    definition, project_root=root, quiet=True,
                    preflight_approval=lambda view: prompts.append(view) or False,
                )
                self.assertEqual(cancelled["status"], "CANCELLED")
                self.assertEqual(len(prompts), 1)
                self.assertEqual(planned, [])
                self.assertIn("REUSE 2 / RUN（需回放）1 / BLOCKED 0", prompts[0])
                self.assertIn("沒有逐筆簽章 Replay 收據 1", prompts[0])
                self.assertIn("R40", prompts[0])
                self.assertIn("T40", prompts[0])
                result = oracle.build_report(
                    definition, project_root=root, quiet=True,
                    preflight_approval=lambda view: prompts.append(view) or True,
                )
            self.assertEqual(len(prompts), 2, "One confirmation each invocation")
            self.assertEqual(planned, [{(seed, "C96"): "REUSE", (seed, "R40"): "REUSE",
                                        (seed, "T40"): "RUN"}])
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual([row["target"] for row in result["oracles"]], ["R40", "T40"])
            self.assertEqual(result["baseline"]["romd"], 8.)
            self.assertEqual(result["oracles"][1]["romd"], 20.)
            self.assertIsInstance(result["limitations"], list)

    def test_stale_c96_is_run_before_approval_without_recalculating_scored_reuse(self):
        """Old C96 executions must not BLOCK the entire preflight.

        Signed R40 (price target) remains reusable; the old C96 receipt is
        preserved, a changed baseline economic input is a new authorized RUN.
        This must require NO 558/617 price/label Producer before the prompt.
        """
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definition, bench, seed, price_sha, param_sha, _proof, _sample = self._fixture(root)
            store = OracleReplayStore(root, definition.output_subdir)
            baseline = store.existing_completed_identities(seed=seed, target="C96")[0]
            original = store._path(baseline)
            original_bytes = original.read_bytes()
            # Reproduce a signed *different* C96 replay-policy source from an
            # earlier execution, not a damaged or self-attested current cache.
            prior_policy = {**baseline["replay_policy"],
                            "execution_defaults": {"legacy": "different"}}
            prior = dict(baseline, replay_policy=prior_policy)
            previous = store.read(baseline)
            original.unlink()
            store.write(prior, previous, origin="historical_same_window_control")
            prior_path = store._path(prior)
            old_bytes = prior_path.read_bytes()
            shown=[]
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(bench, [(seed, root/"p16.json")])), \
                 patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                     [(seed, param_sha, "synthetic", None, None)], {})), \
                 patch.object(oracle, "price_inventory_sha256", return_value=price_sha), \
                 patch.object(oracle, "_read_only_score_keys", side_effect=AssertionError(
                     "Never precompute scores for a status matrix")), \
                 patch.object(oracle, "_execute_single_stage_missing", side_effect=AssertionError(
                     "No Portfolio/Target producer until user approval")):
                result = oracle.build_report(definition, project_root=root, quiet=True,
                    preflight_approval=lambda view: shown.append(view) or False)
            self.assertEqual(result["status"], "CANCELLED")
            self.assertIn("REUSE 1 / RUN（需回放）2 / BLOCKED 0", shown[0])
            self.assertEqual(result["preflight"][(seed,"C96")], "RUN")
            self.assertEqual(result["preflight"][(seed,"R40")], "REUSE")
            self.assertEqual(result["preflight"][(seed,"T40")], "RUN")
            self.assertEqual(prior_path.read_bytes(),old_bytes)
            self.assertFalse(original.exists())

    def test_signed_stale_receipt_is_not_corruption_but_corrupt_c96_still_blocks(self):
        """Generic planner only downgrades signed old *input revisions*."""
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            definition, _bench, seed, _price, _param, _proof, _sample=self._fixture(root)
            store=OracleReplayStore(root,definition.output_subdir)
            current=store.existing_completed_identities(seed=seed,target="C96")[0]
            different=dict(current, dataset_sha256="e"*64)
            self.assertEqual(store.plan_execution({(seed,"C96"):different}).action(
                seed,"C96"),"BLOCKED", "Read-only Audit cannot authorize a different input")
            self.assertEqual(store.plan_execution({(seed,"C96"):different},
                allow_historical_execution_input_drift=True).action(seed,"C96"),"RUN")
            store._path(current).write_text('{broken',encoding="utf-8")
            self.assertEqual(store.plan_execution({(seed,"C96"):different},
                allow_historical_execution_input_drift=True).action(seed,"C96"),"BLOCKED")

    def test_authorized_rebuild_keeps_stale_bytes_and_reuses_new_c96_on_next_run(self):
        """Immutable old control, new key only; future preflight is direct REUSE."""
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            definition, _bench, seed, _price, _param, _proof, sample=self._fixture(root)
            store=OracleReplayStore(root,definition.output_subdir)
            historical=store.existing_completed_identities(seed=seed,target="C96")[0]
            original_bytes=store._path(historical).read_bytes()
            current=dict(historical, replay_policy={**historical["replay_policy"],
                "execution_defaults": {"version": "new authoritatively configured"}})
            policy=dict(seed=seed,target="C96")
            blocked=store.plan_execution({(seed,"C96"):current})
            self.assertEqual(blocked.action(seed,"C96"),"BLOCKED",
                "Unauthorized noninteractive Audit may not silently recompute")
            approved=store.plan_execution({(seed,"C96"):current},
                allow_historical_execution_input_drift=True)
            self.assertEqual(approved.action(seed,"C96"),"RUN")
            produced=[]
            result,_=approved.execute(seed,"C96",lambda: produced.append(1) or sample("C96",9.0))
            self.assertEqual(produced,[1])
            self.assertEqual(result["romd"],9.0)
            self.assertEqual(store._path(historical).read_bytes(),original_bytes)
            next_plan=store.plan_execution({(seed,"C96"):current},
                allow_historical_execution_input_drift=True)
            self.assertEqual(next_plan.action(seed,"C96"),"REUSE")
            found,_=next_plan.execute(seed,"C96",lambda: self.fail("REUSE cannot run"))
            self.assertEqual(found,result)

    def test_approved_all_reuse_avoids_every_target_and_portfolio_producer(self):
        # Completed results may predate a standalone truth-cache migration.
        # A signed Replay receipt is enough to certify the same executed
        # score/selector with unchanged source inputs; do not backfill truth.
        from services.audit import oracle_replay_artifacts as artifacts
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definition, bench, seed, price_sha, param_sha, proof, _sample = self._fixture(root)
            definition = replace(definition, source={
                **definition.source,
                "oracle_comparisons": tuple(item for item in definition.source["oracle_comparisons"]
                                            if item["target"] == "R40"),
            })
            shown=[]
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(bench, [(seed, root/"p16.json")])), \
                 patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                     [(seed, param_sha, "synthetic", None, None)], {})), \
                 patch.object(oracle, "price_inventory_sha256", return_value=price_sha), \
                 patch.object(artifacts, "price_inventory_sha256", return_value=price_sha), \
                 patch.object(oracle, "_read_only_score_keys", side_effect=AssertionError(
                     "Full REUSE must not calculate price Target or DL 40D labels")), \
                 patch.object(oracle, "load_profile_continuous_ranker_data", side_effect=AssertionError(
                     "Full REUSE must not call model data loader")), \
                 patch.object(oracle, "_run_scenario", side_effect=AssertionError(
                     "Full REUSE must not run a Broker-Cash portfolio")), \
                 patch.object(oracle, "_load_or_build_oracle_price_scores", side_effect=AssertionError(
                     "Full REUSE must not materialize any price score")):
                result=oracle.build_report(
                    definition, project_root=root, quiet=True,
                    preflight_approval=lambda table: shown.append(table) or True,
                )
            self.assertEqual(len(shown),1)
            self.assertIn("REUSE 2 / RUN（需回放）0 / BLOCKED 0", shown[0])
            self.assertEqual(result["status"],"COMPLETED")
            self.assertEqual(result["baseline"]["romd"],8.)
            self.assertEqual(result["oracles"][0]["romd"],18.)

    def test_first_run_cancel_before_any_price_label_or_trade_producer(self):
        definition = replace(_ACTIVE_ORACLE_DEFINITION, source={
            **_ACTIVE_ORACLE_DEFINITION.source,
            "oracle_comparisons": tuple(item for item in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                                        if item["target"] in {"R40", "T40"}),
        })
        settings, _base, _exact = oracle._contract(definition)
        bench = SimpleNamespace(benchmark_id="end_to_end_v1")
        seed, param_sha, price_sha = 693545351, "b" * 64, "a" * 64
        preflights = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(bench, [(seed, root/"p16.json")])), \
                 patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                     [(seed, param_sha, "synthetic", None, None)], {})), \
                 patch.object(oracle, "price_inventory_sha256", return_value=price_sha), \
                 patch.object(oracle, "_read_only_score_keys", side_effect=AssertionError(
                     "No eligibility producer before approval")), \
                 patch.object(oracle, "load_profile_continuous_ranker_data", side_effect=AssertionError(
                     "No DL label computation before approval")), \
                 patch.object(oracle, "_load_or_build_oracle_price_scores", side_effect=AssertionError(
                     "No Oracle price score before approval")), \
                 patch.object(oracle, "_execute_single_stage_missing", side_effect=AssertionError(
                     "No Replay before approval")):
                result = oracle.build_report(
                    definition, project_root=root, quiet=True,
                    preflight_approval=lambda table: preflights.append(table) or False,
                )
        self.assertEqual(result["status"], "CANCELLED")
        self.assertEqual(len(preflights), 1)
        self.assertIn("REUSE 0 / RUN（需回放）3 / BLOCKED 0", preflights[0])

    def test_authorized_single_missing_arm_runs_only_that_portfolio(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        import hashlib
        definition = replace(_ACTIVE_ORACLE_DEFINITION, source={
            **_ACTIVE_ORACLE_DEFINITION.source,
            "oracle_comparisons": tuple(entry for entry in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                                        if entry["target"] in {"R40", "T40"}),
        })
        settings, baseline, exact = oracle._contract(definition)
        plan = oracle._oracle_comparison_plan(definition)
        contracts = oracle._resolve_oracle_target_score_contracts(definition, settings, plan)
        seed, price_sha, param_sha = 693545351, "a" * 64, "b" * 64
        scores = {("0050", "2021-01-01"): 0.2, ("1101", "2025-12-22"): 0.8}
        digest = hashlib.sha256()
        for ticker, day in sorted(scores):
            digest.update(f"{ticker}|{day}\n".encode("utf-8"))
        cohort_keys = (digest.hexdigest(), 2, "2021-01-01", "2025-12-22")
        current_keys = {"R40": cohort_keys, "T40": cohort_keys}
        prepared = [(seed, param_sha, "synthetic", "base", "oracle")]
        benchmark = SimpleNamespace(benchmark_id="end_to_end_v1")
        decisions = {(seed, "C96"): "REUSE", (seed, "R40"): "REUSE", (seed, "T40"): "RUN"}
        baseline_row = {"target": "C96", "seed": seed, "romd": 8.0}
        calls = []
        def economic_replay(*, name, **kwargs):
            calls.append(name)
            return {"name": name}
        def convert(replay, *, target):
            return {"target": target, "romd": 20.0, "seed": seed,
                    "yearly": _synthetic_full_window_yearly(),
                    "total_return_pct": 250., "annualized_return_pct": 25.,
                    "mdd_pct": 12.5, "ev_a_r": 1.5, "trades": 200,
                    "exposure_pct": 85., "win_rate_pct": 56.0}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = OracleReplayStore(root, definition.output_subdir)
            with patch("services.audit.oracle_replay_artifacts.price_inventory_sha256", return_value=price_sha), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(
                     benchmark, [(seed, root/"p16.json")])), \
                 patch.object(oracle, "_prepare_oracle_parameters", return_value=(prepared, {})), \
                 patch.object(oracle, "_load_or_build_oracle_price_scores", return_value=(
                     {"T40": scores}, {"cohort_rows": 2})) as price_producer, \
                 patch.object(oracle, "_run_scenario", side_effect=economic_replay) as replay_producer, \
                 patch.object(oracle, "_metrics", side_effect=convert):
                result = oracle._execute_single_stage_missing(
                    definition, project_root=root, quiet=True,
                    plan=plan, contracts=contracts, current_keys=current_keys,
                    settings=settings, robustness=benchmark, price_sha=price_sha,
                    prepared=prepared, decisions=decisions,
                    preflight_cached={(seed, "C96"): baseline_row},
                    start="2021-01-01", end="2025-12-22", base_arm=baseline,
                    exact_arm=exact, direct_arm=None,
                    base_policy=oracle._canonical_oracle_replay_policy(
                        definition, base_arm=baseline, exact_arm=exact), cache=store,
                )
            self.assertEqual(set(result), {(seed, "T40")})
            self.assertEqual(result[(seed,"T40")]["delta_romd_vs_c96"], 12.)
            self.assertEqual(replay_producer.call_count, 1)
            self.assertEqual(price_producer.call_count, 1)
            self.assertEqual(calls, [f"Oracle T40 seed={seed}"])
            self.assertIsNotNone(store.existing_completed_identities(seed=seed, target="T40"))
            self.assertEqual(store.existing_completed_identities(seed=seed, target="R40"), ())

    def test_formal_daily_40d_historical_receipt_reuses_before_truth_producer(self):
        """A signed same-producer receipt survives diagnostic-only schema drift.

        This signed receipt includes current producer lineage; another legacy
        receipt lacking producer lineage must NOT be reused after a behavior
        change even if score bytes and user-facing Target name look identical.
        """
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        from services.audit.daily_trade40_oracle import (
            target_identity_for_horizon, truth_producer_fingerprint,
        )
        import hashlib
        target = "DailyUniversal_MinTrade40_NetR"
        definition = replace(_ACTIVE_ORACLE_DEFINITION,
            dimensions={**_ACTIVE_ORACLE_DEFINITION.dimensions,
                        "oracle_evaluation_end_date": "2025-12-22"},
            source={**_ACTIVE_ORACLE_DEFINITION.source,
            "oracle_comparisons": tuple(entry for entry in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                                        if entry["target"] == target),
        })
        settings, base_arm, exact_arm = oracle._contract(definition)
        plan = oracle._oracle_comparison_plan(definition)
        contract = oracle._resolve_oracle_target_score_contracts(definition, settings, plan)[target]
        seed, param_sha, price_sha = 693545351, "b"*64, "a"*64
        evidence_key = ("0050", "2021-01-04")
        digest = hashlib.sha256((evidence_key[0] + "|" + evidence_key[1] + "\n").encode()).hexdigest()
        cohort_keys = (digest, 1, evidence_key[1], evidence_key[1])
        periods = [{"start": "2021-01-01", "end": "2021-12-31", "params_signature": "p16-a"}]
        tgt_id, label_id = target_identity_for_horizon(40)
        teacher = {"origin": "formal_min_roos_rolling_per_seed", "seed": seed,
                   "param_identity_sha256": param_sha, "periods": periods}
        record = {
            "target_id": tgt_id, "label_id": label_id,
            "profile_id": str(definition.source["daily_trade40_profile"]),
            "rank_policy": "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
            "rank_sha256": "1"*64, "raw_label_sha256": "2"*64,
            "truth_source_sha256": truth_producer_fingerprint(),
            "dataset_summary_sha256": None,
            "calendar_sha256": "3"*64, "teacher": teacher,
            "score_start_date": contract["score_start_date"],
            "mature_eligible_rows": 1, "settled_net_r_rows": 1,
            "censored_rank_last_rows": 0,
            "research_cutoff": str(settings.end_date),
            "oracle_evaluation_start_date": str(settings.start_date),
        }
        score_contract = {**contract, "canonical_feature_eligible_keys_sha256": digest,
                          "oracle_scored_rows": 1,
                          "source_evidence": "validated_generic_daily_feature_universe_and_research_cutoff_closeout"}
        base_policy = oracle._canonical_oracle_replay_policy(
            definition, base_arm=base_arm, exact_arm=exact_arm)
        policy = oracle._arm_replay_policy(
            definition, plan=plan, target=target, replay_policy=base_policy,
            daily_evidence=record, score_sha256="4"*64, score_contract=score_contract)
        # Old signed provenance has no newer diagnostic counters.
        old_policy = {**policy, "hindsight_target_contract": {
            key: value for key, value in policy["hindsight_target_contract"].items()
            if key not in ("cutoff_truncated_rows", "research_cutoff_closeout_rows")}}
        benchmark = SimpleNamespace(benchmark_id="end_to_end_v1")
        kwargs = dict(
            dataset=settings.dataset, dataset_sha256=price_sha,
            target=target, definition={target: target}, start=str(settings.start_date),
            end="2025-12-22", benchmark=benchmark.benchmark_id,
            param_sha256=param_sha, seed=seed,
            cohort={"cohort_rows": 1, "first_common_day": evidence_key[1],
                    "last_common_day": evidence_key[1], "research_cutoff": str(settings.end_date)},
            ranking_policy=str(exact_arm.dl_runtime_mode),
            ranking_options=dict(exact_arm.dl_runtime_options),
            evaluation_mode=str(definition.dimensions["evaluation_mode"]),
            max_positions=settings.max_positions)
        observation = {"target": target, "seed": seed, "total_return_pct": 55.,
                       "annualized_return_pct": 12., "mdd_pct": 10., "romd": 5.5,
                       "ev_a_r": 1.0, "trades": 12, "exposure_pct": 75., "win_rate_pct": 56.0,
                       "delta_romd_vs_c96": 1.0, "yearly": _synthetic_full_window_yearly()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = OracleReplayStore(root, definition.output_subdir)
            legacy = replay_identity(**kwargs, replay_policy=old_policy)
            store.write(legacy, observation, origin="synthetic_historical")
            signed = store.existing_completed_identities(seed=seed, target=target)
            self.assertEqual(len(signed), 1)
            keys_from_receipt = oracle._replay_attested_score_keys(
                definition=definition, store=store, contracts={target: contract}, plan=plan,
                prepared=[(seed, param_sha, "synthetic", None, None)],
                price_sha=price_sha, settings=settings, robustness=benchmark,
                evaluation_end="2025-12-22",
            )
            self.assertEqual(keys_from_receipt, {target: cohort_keys},
                             "Old completed MinTrade score membership must be reusable without a trade-path BUILD")
            self.assertEqual(oracle._replay_attested_score_keys(
                definition=definition, store=store, contracts={target: contract}, plan=plan,
                prepared=[(seed, param_sha, "synthetic", None, None)],
                price_sha="e" * 64, settings=settings, robustness=benchmark,
                evaluation_end="2025-12-22"), {},
                "Changed price bytes must not certify a historical witness")
            expected = oracle._certified_saved_oracle_identity(
                definition, project_root=root, store=store, saved=signed[0],
                plan=plan, contract=contract, keys=cohort_keys,
                param_sha=param_sha, price_sha=price_sha, seed=seed, target=target,
                settings=settings, robustness=benchmark, base_policy=base_policy,
                arm=exact_arm, rolling_periods=periods, evaluation_end="2025-12-22",
            )
            self.assertEqual(store.plan_execution({(seed,target):expected}).action(seed,target), "REUSE")
            self.assertEqual(expected, legacy)
            with self.assertRaises(ValueError):
                oracle._certified_saved_oracle_identity(
                    definition, project_root=root, store=store, saved=signed[0],
                    plan=plan, contract=contract, keys=cohort_keys,
                    param_sha="e"*64, price_sha=price_sha, seed=seed, target=target,
                    settings=settings, robustness=benchmark, base_policy=base_policy,
                    arm=exact_arm, rolling_periods=periods, evaluation_end="2025-12-22",
                )
            with self.assertRaises(ValueError):
                oracle._certified_saved_oracle_identity(
                    definition, project_root=root, store=store, saved=signed[0],
                    plan=plan, contract=contract, keys=("f"*64, 1, evidence_key[1], evidence_key[1]),
                    param_sha=param_sha, price_sha=price_sha, seed=seed, target=target,
                    settings=settings, robustness=benchmark, base_policy=base_policy,
                    arm=exact_arm, rolling_periods=periods, evaluation_end="2025-12-22",
                )

    def test_outcome_producer_revision_is_generic_and_does_not_force_precompute(self):
        """A signed old receipt never certifies a new producer just by score SHA.

        Target labels and horizons deliberately are arbitrary. The policy uses
        only the producer's signed implementation receipt, not a 40/60D list.
        """
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            for suffix in ("Seven", "Forty", "Sixty", "Unbounded"):
                target = "Synthetic_" + suffix
                kwargs = dict(dataset="full", dataset_sha256="a"*64,
                              target=target, definition={target: target},
                              start="2021-01-01", end="2025-12-22",
                              benchmark="end_to_end_v1", param_sha256="b"*64,
                              seed=11, cohort={"cohort_rows": 10},
                              ranking_policy="exact", ranking_options={},
                              evaluation_mode="rolling", max_positions=10)
                policy = {"oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
                          "hindsight_target_contract": {"target_id": target, "raw_label_sha256": "e"*64}}
                old = replay_identity(**kwargs, replay_policy=policy)
                receipt = {"target": target, "seed": 11, "total_return_pct": 50.,
                           "annualized_return_pct": 10., "mdd_pct": 10., "romd": 5.,
                           "ev_a_r": 0.5, "trades": 22, "exposure_pct": 50., "win_rate_pct": 56.0,
                           "delta_romd_vs_c96": 1.,
                           "yearly": _synthetic_full_window_yearly()}
                store.write(old, receipt, origin="historical_producer")
                new_policy = {**policy, "hindsight_target_contract": {
                    **policy["hindsight_target_contract"], "truth_source_sha256": "f"*64}}
                new = replay_identity(**kwargs, replay_policy=new_policy)
                self.assertEqual(store.plan_execution({(11, target): new}).action(11, target), "RUN")
                self.assertIsNotNone(store.read(old), "Historical evidence must remain intact")
                store.write(new, receipt, origin="corrected_producer")
                self.assertEqual(store.plan_execution({(11, target): new}).action(11, target), "REUSE")
                self.assertEqual(len(store.existing_completed_identities(seed=11, target=target)), 2)
                mutated = replay_identity(**kwargs, replay_policy={**new_policy,
                    "hindsight_target_contract": {**new_policy["hindsight_target_contract"],
                                                     "raw_label_sha256": "0"*64}})
                self.assertEqual(store.plan_execution({(11, target): mutated}).action(11, target),
                                 "BLOCKED", "Same producer with different raw truth must fail closed")

    def test_formal_planner_rejects_legacy_mintrade_receipt_without_truth_build(self):
        """Certified price/C96 remain REUSE; obsolete Min outcome becomes RUN only.

        This test uses real signed cache JSON, no fake preflight results. The
        historical-score receipt has matching price/P16/score SHA, but no
        current outcome producer attestation. It must not claim REUSE, and the
        user must see the entire decision table BEFORE any 558-ticker work.
        """
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        from services.audit.daily_trade40_oracle import (
            target_identity_for_horizon, truth_producer_fingerprint,
        )
        from filters.breakout_quality import daily_trade40_target as trade_owner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definition, bench, seed, price_sha, param_sha, proof, sample = self._fixture(root)
            from_source = _ACTIVE_ORACLE_DEFINITION.source
            requested = "DailyUniversal_MinTrade40_NetR"
            definition = replace(definition, source={**definition.source,
                "oracle_comparisons": tuple(row for row in from_source["oracle_comparisons"]
                                            if row["target"] in {"R40", requested})})
            settings, base, exact = oracle._contract(definition)
            plan = oracle._oracle_comparison_plan(definition)
            contract = oracle._resolve_oracle_target_score_contracts(definition, settings, plan)[requested]
            source_id, label_id = target_identity_for_horizon(40)
            period = {"start": "2021-01-01", "end": "2026-01-01", "params_signature": "P16"}
            evidence = {
                "target_id": source_id, "label_id": label_id,
                "profile_id": str(definition.source["daily_trade40_profile"]),
                "rank_policy": "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
                "rank_sha256": "1"*64, "raw_label_sha256": "2"*64,
                "calendar_sha256": "3"*64, "teacher": {
                    "origin": "formal_min_roos_rolling_per_seed", "seed": seed,
                    "param_identity_sha256": param_sha, "periods": [period]},
                "score_start_date": contract["score_start_date"],
                "mature_eligible_rows": proof[1], "settled_net_r_rows": proof[1],
                "censored_rank_last_rows": 0, "research_cutoff": str(settings.end_date),
                "oracle_evaluation_start_date": str(settings.start_date),
                "truth_source_sha256": truth_producer_fingerprint(),
                "dataset_summary_sha256": None,
            }
            common = oracle._canonical_oracle_replay_policy(
                definition, base_arm=base, exact_arm=exact)
            full_score = {**contract, "canonical_feature_eligible_keys_sha256": proof[0],
                          "oracle_scored_rows": proof[1],
                          "source_evidence": "validated_generic_daily_feature_universe_and_research_cutoff_closeout"}
            current_policy = oracle._arm_replay_policy(
                definition, plan=plan, target=requested, replay_policy=common,
                daily_evidence=evidence, score_sha256="4"*64, score_contract=full_score)
            prior_policy = {**current_policy, "hindsight_target_contract": {
                key: value for key, value in current_policy["hindsight_target_contract"].items()
                if key != "truth_source_sha256"}}
            cohort = {"cohort_rows": proof[1], "first_common_day": proof[2],
                      "last_common_day": proof[3], "research_cutoff": settings.end_date,
                      "risk_budget_return": float(oracle.StrategyAlignedContinuousTargetSpec.from_label_policy(
                          oracle.DEFAULT_LABEL_POLICY).risk_budget_return),
                      "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
                      "oracle_only": True}
            def receipt_identity(policy):
                return replay_identity(
                    dataset=settings.dataset, dataset_sha256=price_sha,
                    target=requested, definition={requested: requested},
                    start=str(settings.start_date), end=str(definition.dimensions["oracle_evaluation_end_date"]),
                    benchmark=bench.benchmark_id, param_sha256=param_sha, seed=seed,
                    cohort=cohort, ranking_policy=str(exact.dl_runtime_mode),
                    ranking_options=dict(exact.dl_runtime_options),
                    evaluation_mode=str(definition.dimensions["evaluation_mode"]),
                    max_positions=settings.max_positions, replay_policy=policy)
            store = OracleReplayStore(root, definition.output_subdir)
            old = receipt_identity(prior_policy)
            store.write(old, sample(requested, 19., 11.), origin="pre_cutoff_bugfix")
            display = []
            schedule = (SimpleNamespace(start=pd.Timestamp("2021-01-01"),
                                        end=pd.Timestamp("2026-01-01"),
                                        params_signature="P16"),)
            def preflight():
                with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                     patch.object(oracle, "_robustness_artifacts", return_value=(
                         bench, [(seed, root/"p16.json")])), \
                     patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                         [(seed, param_sha, "synthetic", None, None)], {})), \
                     patch.object(oracle, "price_inventory_sha256", return_value=price_sha), \
                     patch.object(trade_owner, "resolve_formal_min_param_records", return_value=()), \
                     patch.object(trade_owner, "build_formal_min_execution_schedule", return_value=(schedule, {})), \
                     patch.object(oracle, "_read_only_score_keys", side_effect=AssertionError(
                         "No DL/617 price producer before first confirmation")), \
                     patch.object(oracle, "_execute_single_stage_missing", side_effect=AssertionError(
                         "No MinTrade/Portfolio producer before first confirmation")):
                    return oracle.build_report(
                        definition, project_root=root, quiet=True,
                        preflight_approval=lambda table: display.append(table) or False)
            self.assertEqual(preflight()["status"], "CANCELLED")
            self.assertEqual(len(display), 1)
            self.assertIn("REUSE 2 / RUN（需回放）1 / BLOCKED 0", display[-1])
            self.assertEqual(store.existing_completed_identities(seed=seed, target=requested), (old,))
            store.write(receipt_identity(current_policy), sample(requested, 19., 11.),
                        origin="current_producer")
            self.assertEqual(preflight()["status"], "CANCELLED")
            self.assertIn("REUSE 3 / RUN（需回放）0 / BLOCKED 0", display[-1])

    def test_diagnostic_only_counters_do_not_block_saved_trading_result(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            policy = {"oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
                      "period_stats_implementation": oracle.PERIOD_STATS_IMPLEMENTATION,
                      "perfect_score_sha256": "c"*64,
                      "canonical_dl_score_contract": {
                          "target_id": "daily40", "canonical_feature_eligible_keys_sha256": "d"*64},
                      "hindsight_target_contract": {
                          "label_id": "netr40", "raw_label_sha256": "e"*64,
                          "rank_sha256": "f"*64, "cutoff_truncated_rows": 0,
                          "research_cutoff_closeout_rows": 0}}
            kwargs = dict(dataset="full", dataset_sha256="a"*64,
                          target="DailyUniversal_MinTrade40_NetR",
                          definition={"DailyUniversal_MinTrade40_NetR": "description"},
                          start="2021-01-01", end="2025-12-22", benchmark="end_to_end_v1",
                          param_sha256="b"*64, seed=11, cohort={"cohort_rows": 10},
                          ranking_policy="exact", ranking_options={}, evaluation_mode="rolling",
                          max_positions=10)
            # Historical code stored the extra counters; scientific score SHA
            # and every economic input are identical to the new identity.
            previous_policy = {**policy,
                               "hindsight_target_contract": {
                                   k: v for k, v in policy["hindsight_target_contract"].items()
                                   if k not in ("cutoff_truncated_rows", "research_cutoff_closeout_rows")}}
            old = replay_identity(**kwargs, replay_policy=previous_policy)
            store.write(old, {"target": kwargs["target"], "seed": 11,
                              "total_return_pct": 50., "annualized_return_pct": 10.,
                              "mdd_pct": 10., "romd": 5., "ev_a_r": 0.5,
                              "trades": 22, "exposure_pct": 50., "win_rate_pct": 56.0,
                              "delta_romd_vs_c96": 1.,
                              "yearly": _synthetic_full_window_yearly()}, origin="synthetic")
            current = replay_identity(**kwargs, replay_policy=policy)
            self.assertEqual(current, old)
            self.assertEqual(store.plan_execution({(11, kwargs["target"]):current}).action(
                11, kwargs["target"]), "REUSE")


class OracleReuseMenuAndPreparationCases(TestCase):
    """User-facing menu and shared-input preparation; not mere status assertions."""

    def test_replay_session_prepares_once_per_identical_seed_contract(self):
        from services import portfolio_replay as replay
        first = {"param_signature": "P16-a", "contexts": object()}
        calls = []
        def prepare(label):
            calls.append(label)
            return first
        key = replay._replay_preparation_key(
            "/tmp/research", [{"effective_date_text": "2021-01-01",
                "effective_end_date_text": "2021-12-31",
                "members": [{"params_signature": "P16-a", "member_index": 1}]}],
            ensemble=True, raw_universe_required_min_rows=260,
            keep_trade_logs=False, use_prepared_cache=False,
            write_prepared_cache=False,
        )
        cache = {}
        self.assertIs(replay._load_session_replay_preparation(cache, key, lambda: prepare("first")), first)
        self.assertIs(replay._load_session_replay_preparation(cache, key, lambda: prepare("wrong")), first)
        self.assertEqual(calls, ["first"])
        self.assertIs(replay._load_session_replay_preparation(None, key, lambda: prepare("no-cache")), first)
        self.assertEqual(calls, ["first", "no-cache"])
        changed = replay._replay_preparation_key(
            "/tmp/research", [{"effective_date_text": "2021-01-01",
                "effective_end_date_text": "2021-12-31",
                "members": [{"params_signature": "P16-b", "member_index": 1}]}],
            ensemble=True, raw_universe_required_min_rows=260,
            keep_trade_logs=False, use_prepared_cache=False,
            write_prepared_cache=False,
        )
        self.assertNotEqual(changed, key)
        self.assertIs(replay._load_session_replay_preparation(cache, changed, lambda: prepare("different-P16")), first)
        self.assertEqual(calls[-1], "different-P16")

    def test_target_scenario_passes_same_session_cache_only_to_run(self):
        from services.research import strategy_compare_replay as scenario
        observed = []
        cache = {}
        def fake_ensemble(*args, **kwargs):
            observed.append(kwargs["replay_preparation_cache"])
            return tuple([None] * len(scenario.PortfolioReplayResult._fields))
        def fake_unpack(result):
            return {"economic": 17.5}
        with patch.object(scenario, "run_portfolio_simulation_with_param_ensemble", side_effect=fake_ensemble), \
             patch.object(scenario, "_unpack_result", side_effect=fake_unpack):
            for target in ("T40", "R40"):
                result = scenario._run_scenario(
                    name=target, data_dir=Path("/tmp/research"),
                    param_source_kind="rolling_active_param_ensemble", params={},
                    start_date="2021-01-01", end_date="2025-12-22", max_positions=10,
                    enable_rotation=False, quiet=True, replay_preparation_cache=cache,
                )
                self.assertEqual(result, {"economic": 17.5})
        self.assertEqual(len(observed), 2)
        self.assertIs(observed[0], cache)
        self.assertIs(observed[1], cache)


    def test_separate_replay_calls_share_preparation_but_not_portfolio_timeline(self):
        from services import portfolio_replay as replay
        import datetime
        date = pd.Timestamp("2021-01-04")
        schedule = [{"effective_date_text": "2021-01-01",
                     "effective_end_date_text": "2021-01-05",
                     "effective_date": datetime.date(2021, 1, 1),
                     "year": 2021, "members": [{"params_signature": "P16",
                         "member_index": 1, "params_obj": object()}]}]
        context = {"sorted_dates": [date, pd.Timestamp("2021-01-05")],
                   "all_dfs_fast": {}, "all_trade_logs": {},
                   "all_pit_stats_index": {}, "prep_wall_sec": 0.25}
        source = {"2021-01-01": [context]}
        seen = []
        def simulate(*args, **kwargs):
            seen.append(kwargs["active_context_ensemble_resolver"](date))
            return f"independent-timeline-{len(seen)}"
        with patch.object(replay, "build_active_param_ensemble_objects_from_payload", return_value=schedule), \
             patch.object(replay, "resolve_raw_universe_required_min_rows", return_value=260), \
             patch.object(replay, "get_active_param_ensemble_policy", return_value={"seed_count":1,"min_agree":1}), \
             patch.object(replay, "resolve_active_param_ensemble_mode", return_value="rolling"), \
             patch.object(replay, "get_active_param_ensemble_date_range", return_value=("2021-01-01","2021-01-05")), \
             patch.object(replay, "_load_contexts_for_active_ensemble_schedule", return_value=source) as loader, \
             patch.object(replay, "_apply_active_single_stock_score_stats"), \
             patch.object(replay, "run_portfolio_timeline", side_effect=simulate) as simulator, \
             patch.object(replay, "append_portfolio_replay_profile", side_effect=lambda result, profile: (result, profile)):
            cache = {}
            outcomes = []
            for _ in range(2):
                outcomes.append(replay.run_portfolio_simulation_with_param_ensemble(
                    "/tmp/research", {"policy": "synthetic"}, start_date="2021-01-04",
                    end_date="2021-01-05", replay_preparation_cache=cache,
                    verbose=False,
                ))
        self.assertEqual(loader.call_count, 1, "Market and P16 preparation must occur once per run/seed")
        self.assertEqual(simulator.call_count, 2, "Every RUN must simulate its own portfolio")
        self.assertIs(seen[0], seen[1], "Immutable prepared context may be shared")
        self.assertEqual([result for result, _ in outcomes],
                         ["independent-timeline-1", "independent-timeline-2"])

class OraclePriceReplayLineageCases(TestCase):
    """A signed old score cannot certify changed formula/feature producers."""

    def _preflight(self, root, fixture):
        definition, benchmark, seed, prices, params, _proof, _sample = fixture
        with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
             patch.object(oracle, "_robustness_artifacts", return_value=(benchmark, [(seed, root / "p16.json")])), \
             patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                 [(seed, params, "synthetic", None, None)], {})), \
             patch.object(oracle, "price_inventory_sha256", return_value=prices), \
             patch.object(oracle, "_read_only_score_keys", side_effect=AssertionError("No labels in preflight")), \
             patch.object(oracle, "_build_oracle_score_panels", side_effect=AssertionError("No price truth")), \
             patch.object(oracle, "_execute_single_stage_missing", side_effect=AssertionError("Cancelled")):
            return oracle.build_report(definition, project_root=root, quiet=True,
                                       preflight_approval=lambda _view: False)["preflight"]

    def test_changed_price_formula_is_run_before_any_truth_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # This test replaces producer callables to forbid work. Their real
            # algorithm digest is captured before installing those mocks.
            original = oracle._price_score_algorithm_sha256()
            with patch.object(oracle, "_price_score_algorithm_sha256", return_value=original):
                fixture = OracleOnePlanRealCacheCases()._fixture(root)
                seed = fixture[2]
                self.assertEqual(self._preflight(root, fixture)[(seed, "R40")], "REUSE")
            with patch.object(oracle, "_price_score_algorithm_sha256", return_value="f" * 64):
                changed = self._preflight(root, fixture)
            self.assertEqual(changed, {(seed, "C96"): "REUSE", (seed, "R40"): "RUN",
                                       (seed, "T40"): "RUN"})

    def test_legacy_price_receipt_without_lineage_runs_and_preserves_history(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = oracle._price_score_algorithm_sha256()
            with patch.object(oracle, "_price_score_algorithm_sha256", return_value=original):
                fixture = OracleOnePlanRealCacheCases()._fixture(root)
                definition, _benchmark, seed, *_unused = fixture
                store = OracleReplayStore(root, definition.output_subdir)
                current, = store.existing_completed_identities(seed=seed, target="R40")
                record = store.read(current)
                old = {**current, "replay_policy": {key: value for key, value in current["replay_policy"].items()
                                                    if key != "hindsight_target_contract"}}
                store.write(old, record, origin="historical_missing_producer_proof")
                store._path(current).unlink()
                preserved = store._path(old).read_bytes()
                self.assertEqual(self._preflight(root, fixture)[(seed, "R40")], "RUN")
                planned = store.plan_execution({(seed, "R40"): current})
                planned.require_ready()
                self.assertEqual(planned.action(seed, "R40"), "RUN")
                planned.execute(seed, "R40", lambda: record)
                self.assertEqual(store.plan_execution({(seed, "R40"): current}).action(seed, "R40"), "REUSE")
                self.assertEqual(store._path(old).read_bytes(), preserved)

    def test_dataset_source_metadata_drift_is_not_certified_from_old_keys(self):
        from filters.breakout_quality.dataset_store import resolve_dataset_paths
        from filters.breakout_quality.paths import resolve_filter_output_dir
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = oracle._price_score_algorithm_sha256()
            summary = resolve_dataset_paths(resolve_filter_output_dir(
                root, str(_ACTIVE_ORACLE_DEFINITION.source["filter_id"]))).summary
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text('{"source_data_end_date":"2026-03-02"}', encoding="utf-8")
            with patch.object(oracle, "_price_score_algorithm_sha256", return_value=original):
                fixture = OracleOnePlanRealCacheCases()._fixture(root)
                seed = fixture[2]
                self.assertEqual(self._preflight(root, fixture)[(seed, "R40")], "REUSE")
                summary.write_text('{"source_data_end_date":"2025-12-31"}', encoding="utf-8")
                changed = self._preflight(root, fixture)
                self.assertEqual(changed[(seed, "R40")], "RUN")
                self.assertEqual(changed[(seed, "C96")], "REUSE")

    def test_price_feature_primitive_dependencies_share_one_fingerprint(self):
        from services.audit import oracle_score_artifacts as source
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in source._DAILY_SCORE_SOURCE_MODULES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source fixture\n", encoding="utf-8")
            before = source.oracle_daily_source_fingerprint(root)
            for relative in ("core/data_utils.py", "filters/breakout_quality/features.py",
                             "filters/breakout_quality/csv_io.py",
                             "filters/breakout_quality/models/spec_registry.py",
                             "core/breakout_quality_policy.py", "core/walk_forward_policy.py"):
                with self.subTest(owner=relative):
                    path = root / relative
                    path.write_text("changed eligibility primitive\n", encoding="utf-8")
                    self.assertNotEqual(source.oracle_daily_source_fingerprint(root), before)
                    path.write_text("source fixture\n", encoding="utf-8")
            self.assertEqual(source.oracle_daily_source_fingerprint(root), before)


class OracleDailySourceMetadataCases(TestCase):
    """Daily-only REUSE must attest metadata that bounds its feature cohort."""

    def _fixture(self, root):
        from hashlib import sha256
        from core.file_integrity import compute_file_sha256
        from filters.breakout_quality.dataset_store import resolve_dataset_paths
        from filters.breakout_quality.paths import resolve_filter_output_dir
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        from tools.validate.synthetic_daily_trade40_oracle_cases import run_source

        names = ("DailyUniversal_MinTrade40_NetR", "DailyUniversal_MinTrade40_NetR_Direct")
        definition = replace(_ACTIVE_ORACLE_DEFINITION, source={**_ACTIVE_ORACLE_DEFINITION.source,
            "oracle_comparisons": tuple(row for row in _ACTIVE_ORACLE_DEFINITION.source["oracle_comparisons"]
                                        if row["target"] in names)})
        settings, base, exact = oracle._contract(definition)
        direct = oracle._direct_arm(definition, settings, base, exact)
        plan = oracle._oracle_comparison_plan(definition)
        contracts = oracle._resolve_oracle_target_score_contracts(definition, settings, plan)
        summary = resolve_dataset_paths(resolve_filter_output_dir(root, str(definition.source["filter_id"]))).summary
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"source_data_date_range": {"end": str(settings.end_date)}}), encoding="utf-8")
        scores, evidence = run_source(cutoff=str(settings.end_date))
        evidence["dataset_summary_sha256"] = compute_file_sha256(summary)
        key_sha = sha256("".join(f"{ticker}|{day}\n" for ticker, day in sorted(scores)).encode()).hexdigest()
        cohort = {"cohort_rows": len(scores), "first_common_day": min(day for _, day in scores),
                  "last_common_day": max(day for _, day in scores), "research_cutoff": str(settings.end_date)}
        seed, price_sha, param_sha = 693545351, "b" * 64, "a" * 64
        benchmark = SimpleNamespace(benchmark_id="end_to_end_v1")
        common = oracle._canonical_oracle_replay_policy(definition, base_arm=base, exact_arm=exact)
        store = OracleReplayStore(root, definition.output_subdir)
        for target in ("C96", *names):
            arm = direct if target == names[1] else exact
            args = {} if target == "C96" else dict(daily_evidence=evidence,
                score_sha256=oracle._canonical_score_map_sha256(scores),
                score_contract={**contracts[target], "canonical_feature_eligible_keys_sha256": key_sha,
                                "oracle_scored_rows": len(scores),
                                "source_evidence": "validated_generic_daily_feature_universe_and_research_cutoff_closeout"})
            policy = oracle._arm_replay_policy(definition, plan=plan, target=target,
                replay_policy=common, project_root=root, **args)
            identity = replay_identity(dataset=settings.dataset, dataset_sha256=price_sha,
                target=target, definition={target: target}, start=str(settings.start_date),
                end=str(definition.dimensions["oracle_evaluation_end_date"]),
                benchmark=benchmark.benchmark_id, param_sha256=param_sha, seed=seed,
                cohort=cohort, ranking_policy=arm.dl_runtime_mode,
                ranking_options=dict(arm.dl_runtime_options),
                evaluation_mode=str(definition.dimensions["evaluation_mode"]),
                max_positions=settings.max_positions, replay_policy=policy)
            observation = {"target": target, "seed": seed, "romd": 1.,
                "total_return_pct": 1., "annualized_return_pct": 1., "mdd_pct": 1.,
                "ev_a_r": .1, "trades": 1, "exposure_pct": 1., "win_rate_pct": 56.0,
                "yearly": _synthetic_full_window_yearly(),
                **({} if target == "C96" else {"delta_romd_vs_c96": 0.})}
            store.write(identity, observation, origin="synthetic_dataset_source_witness")
        periods = tuple(SimpleNamespace(start=pd.Timestamp(row["start"]),
            end=None if row["end"] is None else pd.Timestamp(row["end"]),
            params_signature=row["params_signature"]) for row in evidence["teacher"]["periods"])
        return definition, benchmark, seed, price_sha, param_sha, summary, periods, names

    def _preflight(self, root, fixture, approval=None):
        from filters.breakout_quality import daily_trade40_target as trade_target
        definition, benchmark, seed, prices, params, _summary, periods, _names = fixture
        with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
             patch.object(oracle, "_robustness_artifacts", return_value=(benchmark, [(seed, root / "p16.json")])), \
             patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                 [(seed, params, "synthetic", None, None)], {})), \
             patch.object(oracle, "price_inventory_sha256", return_value=prices), \
             patch.object(trade_target, "resolve_formal_min_param_records", return_value=[]), \
             patch.object(trade_target, "build_formal_min_execution_schedule", return_value=(periods, {})), \
             patch.object(oracle, "_read_only_score_keys", side_effect=AssertionError("No labels in preflight")), \
             patch.object(oracle, "load_profile_continuous_ranker_data", side_effect=AssertionError("No DL provider")), \
             patch("services.audit.daily_trade40_oracle.build_daily_universal_trade40_oracle_scores",
                   side_effect=AssertionError("No MinTrade truth in preflight")), \
             patch.object(oracle, "_execute_single_stage_missing", side_effect=AssertionError("Cancelled")):
            return oracle.build_report(definition, project_root=root, quiet=True,
                                       preflight_approval=approval or (lambda _view: False))["preflight"]

    def test_daily_only_changed_summary_is_run_without_scientific_precompute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            _definition, _benchmark, seed, _prices, _params, summary, _periods, names = fixture
            before = self._preflight(root, fixture)
            self.assertEqual(set(before.values()), {"REUSE"})
            summary.write_text('{"source_data_date_range":{"end":"2025-12-31"}}', encoding="utf-8")
            changed = self._preflight(root, fixture)
            self.assertEqual(changed, {(seed, "C96"): "REUSE", **{(seed, name): "RUN" for name in names}})

    def test_daily_only_receipt_without_source_metadata_is_not_reused(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            definition, _benchmark, seed, _prices, _params, _summary, _periods, names = fixture
            store = OracleReplayStore(root, definition.output_subdir)
            preserved = []
            for target in names:
                current, = store.existing_completed_identities(seed=seed, target=target)
                record = store.read(current)
                old_lineage = {key: value for key, value in current["replay_policy"]["hindsight_target_contract"].items()
                               if key != "dataset_summary_sha256"}
                old = {**current, "replay_policy": {**current["replay_policy"], "hindsight_target_contract": old_lineage}}
                store.write(old, record, origin="historical_no_dataset_source_metadata")
                store._path(current).unlink()
                preserved.append((store._path(old), store._path(old).read_bytes()))
            self.assertEqual(self._preflight(root, fixture),
                             {(seed, "C96"): "REUSE", **{(seed, name): "RUN" for name in names}})
            self.assertTrue(all(path.read_bytes() == content for path, content in preserved))

    def test_daily_source_changed_during_approval_stops_without_any_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            def approval(_view):
                fixture[5].write_text('{"source_data_date_range":{"end":"2025-12-31"}}', encoding="utf-8")
                return True
            with self.assertRaisesRegex(oracle.OracleBlockedError, "metadata changed after approved preflight"):
                self._preflight(root, fixture, approval=approval)

    def test_daily_truth_and_feature_cache_include_same_source_metadata(self):
        from core.file_integrity import compute_file_sha256
        from filters.breakout_quality.dataset_store import resolve_dataset_paths
        from filters.breakout_quality.paths import resolve_filter_output_dir
        from services.audit import daily_trade40_oracle as source
        from tools.validate.synthetic_daily_trade40_oracle_cases import source_fixture, PROFILE
        trading, frames, bundle, schedule = source_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = resolve_dataset_paths(resolve_filter_output_dir(root, "breakout_quality_v1")).summary
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text('{"source_data_date_range":{"end":"2021-05-15"}}', encoding="utf-8")
            kwargs = dict(project_root=root, dataset="full", filter_id="breakout_quality_v1",
                profile_id=PROFILE, cutoff="2021-05-15", evaluation_start="2021-01-01",
                score_start="2021-01-01", seed=693545351, param_sha256="a" * 64,
                canonical_schedule=schedule, price_sha256="b" * 64,
                cache_root=root / "truth_cache", feature_cohort_cache={})
            with patch.object(source, "load_profile_continuous_ranker_data", return_value=bundle) as loader, \
                 patch.object(source, "discover_dataset_csv_inputs", return_value=(
                     [(ticker, Path(ticker + ".csv")) for ticker in frames], [])), \
                 patch.object(source, "collect_canonical_exchange_sessions", return_value=trading), \
                 patch.object(source, "load_dataset_frame", side_effect=lambda path, ticker, **kw: frames[ticker]), \
                 patch.object(source, "compute_counterfactual_trade40_batch", wraps=source.compute_counterfactual_trade40_batch) as compute:
                first, evidence = source.build_daily_universal_trade40_oracle_scores(**kwargs)
                calls = compute.call_count
                second, reused = source.build_daily_universal_trade40_oracle_scores(**kwargs, allow_build=False)
                self.assertEqual(first, second)
                self.assertEqual(reused["source_reuse_status"], "REUSE")
                self.assertEqual(compute.call_count, calls)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(evidence["dataset_summary_sha256"], compute_file_sha256(summary))
                summary.write_text('{"source_data_date_range":{"end":"2021-01-04"}}', encoding="utf-8")
                smaller = bundle.group_table.loc[pd.to_datetime(bundle.group_table["date"]) <= pd.Timestamp("2021-01-04")].copy()
                loader.return_value = SimpleNamespace(group_table=smaller, summary={**bundle.summary,
                    "score_eligible_sample_count": len(smaller)})
                third, rebuilt = source.build_daily_universal_trade40_oracle_scores(**kwargs)
                self.assertEqual(rebuilt["source_reuse_status"], "BUILD")
                self.assertEqual(loader.call_count, 2, "Changed metadata must not reuse the old in-run cohort")
                self.assertLess(len(third), len(first))
                self.assertNotEqual(rebuilt["dataset_summary_sha256"], evidence["dataset_summary_sha256"])
                self.assertEqual(rebuilt["dataset_summary_sha256"], compute_file_sha256(summary))


class OraclePriceScoreAliasCacheCases(TestCase):
    """One price truth cache must serve any Exact/Direct recipient subset."""

    def setUp(self):
        from core.file_integrity import compute_file_sha256
        from filters.breakout_quality.dataset_store import resolve_dataset_paths
        from filters.breakout_quality.paths import resolve_filter_output_dir
        from services.audit import oracle_score_artifacts as artifacts

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = oracle._contract(_ACTIVE_ORACLE_DEFINITION)[0]
        self.algorithm_sha = oracle._price_score_algorithm_sha256()
        self.price_sha = "a" * 64
        self.scores = {("2330", "2025-01-02"): 0.0,
                       ("2317", "2025-01-02"): 1.0}
        self.cohort = {"cohort_rows": 2, "first_common_day": "2025-01-02",
                       "last_common_day": "2025-01-02"}
        # Real source identities and npz receipts; the numerical producer is
        # isolated below because this regression concerns recipient routing.
        for relative in artifacts._DAILY_SCORE_SOURCE_MODULES:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        paths = resolve_dataset_paths(resolve_filter_output_dir(
            self.root, str(_ACTIVE_ORACLE_DEFINITION.source["filter_id"])))
        paths.summary.parent.mkdir(parents=True, exist_ok=True)
        records = {}
        for name, path in paths.artifact_paths().items():
            path.write_bytes(f"isolated cache source: {name}".encode("utf-8"))
            records[name] = {"sha256": compute_file_sha256(path)}
        paths.summary.write_text(json.dumps({
            "dataset": self.settings.dataset,
            "filter_id": _ACTIVE_ORACLE_DEFINITION.source["filter_id"],
            "dataset_artifacts": records,
        }), encoding="utf-8")

    def _selected(self, *names):
        definition = _ACTIVE_ORACLE_DEFINITION
        return replace(definition, source={**definition.source,
            "oracle_comparisons": tuple(row for row in definition.source["oracle_comparisons"]
                                        if row["target"] in names)})

    def _identity(self, definition):
        from services.audit.oracle_score_artifacts import price_score_source_identity
        plan = oracle._oracle_comparison_plan(definition)
        return price_score_source_identity(
            root=self.root, dataset=str(self.settings.dataset),
            filter_id=str(definition.source["filter_id"]),
            price_sha256=self.price_sha,
            providers={field: str(definition.source[field]) for field in plan["providers"]},
            recipes=plan["recipes"],
            score_start=oracle._formal_dl_score_start(definition, self.settings),
            cutoff="2026-03-02", score_algorithm_sha256=self.algorithm_sha,
            label_policy=oracle.DEFAULT_LABEL_POLICY.as_manifest_payload(),
        )

    def _store(self, definition):
        from services.audit.oracle_score_artifacts import OraclePriceScoreStore
        return OraclePriceScoreStore(
            self.root / oracle.AUDIT_OUTPUT_ROOT / definition.output_subdir / "truth_cache")

    def _load(self, definition, *, allow_build=True):
        return oracle._load_or_build_oracle_price_scores(
            definition, project_root=self.root, cutoff="2026-03-02",
            price_sha=self.price_sha, allow_build=allow_build)

    def _producer(self, definition, **_kwargs):
        return ({name: dict(self.scores)
                 for name, _description in oracle._oracle_comparison_plan(definition)["targets"]},
                dict(self.cohort))

    def _assert_new_cache_serves_both_selectors(self, first_name):
        first = self._selected(first_name)
        exact = self._selected("R40")
        direct = self._selected("R40_Direct")
        combined = self._selected("R40", "R40_Direct")
        identity = self._identity(first)
        self.assertEqual(identity, self._identity(exact))
        self.assertEqual(identity, self._identity(direct))
        self.assertEqual(identity, self._identity(combined))
        with patch.object(oracle, "_build_oracle_score_panels", side_effect=self._producer) as producer, \
             patch.object(oracle, "_price_score_algorithm_sha256", return_value=self.algorithm_sha):
            first_scores, first_cohort = self._load(first)
            self.assertEqual(set(first_scores), {first_name})
            self.assertEqual(first_cohort, self.cohort)
            for definition, names in ((exact, {"R40"}), (direct, {"R40_Direct"}),
                                      (combined, {"R40", "R40_Direct"})):
                scores, cohort = self._load(definition, allow_build=False)
                self.assertEqual(set(scores), names)
                self.assertTrue(all(panel == self.scores for panel in scores.values()))
                self.assertEqual(cohort, self.cohort)
                if len(names) == 2:
                    self.assertIs(scores["R40"], scores["R40_Direct"])
            self.assertEqual(producer.call_count, 1)
        stored, _cohort = self._store(first).read(
            identity, score_sha=oracle._canonical_score_map_sha256)
        self.assertEqual(set(stored), {"R40"}, "Cache stores the source recipe, not its selector")

    def test_exact_cache_supplies_direct_and_combined_without_rebuilding(self):
        self._assert_new_cache_serves_both_selectors("R40")

    def test_direct_cache_stores_canonical_source_and_supplies_exact(self):
        self._assert_new_cache_serves_both_selectors("R40_Direct")

    def test_legacy_direct_panels_reuse_without_rewriting_history(self):
        exact = self._selected("R40")
        direct = self._selected("R40_Direct")
        combined = self._selected("R40", "R40_Direct")
        identity = self._identity(exact)
        store = self._store(exact)
        store.write(identity, {"R40_Direct": dict(self.scores)}, self.cohort,
                    score_sha=oracle._canonical_score_map_sha256)
        path = store._path(identity)
        before = path.read_bytes()
        with patch.object(oracle, "_build_oracle_score_panels",
                          side_effect=AssertionError("Certified source must not rebuild")) as producer, \
             patch.object(oracle, "_price_score_algorithm_sha256", return_value=self.algorithm_sha):
            for definition, names in ((exact, {"R40"}), (direct, {"R40_Direct"}),
                                      (combined, {"R40", "R40_Direct"})):
                scores, cohort = self._load(definition, allow_build=False)
                self.assertEqual(set(scores), names)
                self.assertTrue(all(panel == self.scores for panel in scores.values()))
                self.assertEqual(cohort, self.cohort)
                if len(names) == 2:
                    self.assertIs(scores["R40"], scores["R40_Direct"])
            producer.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_legacy_same_source_score_conflict_is_blocked(self):
        definition = self._selected("R40", "R40_Direct")
        identity = self._identity(definition)
        store = self._store(definition)
        conflicting = {**self.scores, ("2330", "2025-01-02"): 0.5}
        # Each saved panel has a valid SHA; their shared-source claim conflicts.
        store.write(identity, {"R40": dict(self.scores), "R40_Direct": conflicting},
                    self.cohort, score_sha=oracle._canonical_score_map_sha256)
        path = store._path(identity)
        before = path.read_bytes()
        with patch.object(oracle, "_build_oracle_score_panels",
                          side_effect=AssertionError("Conflict cannot trigger a rebuild")) as producer, \
             patch.object(oracle, "_price_score_algorithm_sha256", return_value=self.algorithm_sha):
            with self.assertRaisesRegex(oracle.OracleBlockedError, "source conflict"):
                self._load(definition)
            producer.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_unrecognized_saved_source_is_not_guessed_from_equal_values(self):
        definition = self._selected("R40")
        identity = self._identity(definition)
        store = self._store(definition)
        store.write(identity, {"T40": dict(self.scores)}, self.cohort,
                    score_sha=oracle._canonical_score_map_sha256)
        with patch.object(oracle, "_build_oracle_score_panels",
                          side_effect=AssertionError("Wrong source cannot trigger a rebuild")) as producer, \
             patch.object(oracle, "_price_score_algorithm_sha256", return_value=self.algorithm_sha):
            with self.assertRaisesRegex(oracle.OracleBlockedError, "unrecognized source"):
                self._load(definition)
            producer.assert_not_called()


class OracleSingleOwnerExecutionCases(TestCase):
    """UI and noninteractive runs share execution, with explicit drift authorization."""

    def _context(self, root, fixture, *, price_sha=None, param_sha=None):
        from contextlib import ExitStack
        from services.audit import oracle_replay_artifacts as artifacts
        from services.audit import daily_trade40_oracle as daily
        _definition, benchmark, seed, prices, params, _proof, _sample = fixture
        stack = ExitStack()
        # Capture real code provenance before deliberately forbidding producers.
        stack.enter_context(patch.object(oracle, "_price_score_algorithm_sha256",
                                         return_value=oracle._price_score_algorithm_sha256()))
        stack.enter_context(patch.object(oracle, "collect_status", return_value={"status": "READY"}))
        stack.enter_context(patch.object(oracle, "_robustness_artifacts", return_value=(
            benchmark, [(seed, root / "p16.json")])))
        stack.enter_context(patch.object(oracle, "_prepare_oracle_parameters", return_value=(
            [(seed, param_sha or params, "synthetic", None, None)], {})))
        for owner in (oracle, artifacts):
            stack.enter_context(patch.object(owner, "price_inventory_sha256", return_value=price_sha or prices))
        for owner, name in ((oracle, "_read_only_score_keys"),
                            (oracle, "load_profile_continuous_ranker_data"),
                            (oracle, "_build_oracle_score_panels"),
                            (oracle, "_load_or_build_oracle_price_scores"),
                            (oracle, "_run_scenario"),
                            (daily, "build_daily_universal_trade40_oracle_scores")):
            stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
                f"Unexpected scientific producer: {name}")))
        return stack

    def _fixture(self, root):
        fixture = OracleOnePlanRealCacheCases()._fixture(root)
        definition = fixture[0]
        definition = replace(definition, source={**definition.source,
            "oracle_comparisons": tuple(row for row in definition.source["oracle_comparisons"]
                                        if row["target"] == "R40")})
        return (definition, *fixture[1:])

    def test_both_call_modes_reuse_identical_signed_results_without_any_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            original = {path: path.read_bytes() for path in root.rglob("*.json")}
            prompts = []
            with self._context(root, fixture):
                cli = oracle.build_report(fixture[0], project_root=root, quiet=True)
                ui = oracle.build_report(fixture[0], project_root=root, quiet=True,
                    preflight_approval=lambda view: prompts.append(view) or True)
            self.assertEqual(cli, ui)
            self.assertEqual(len(prompts), 1)
            self.assertIn("REUSE 2 / RUN（需回放）0 / BLOCKED 0", prompts[0])
            self.assertEqual({path: path.read_bytes() for path in root.rglob("*.json")}, original)

    def test_noninteractive_price_and_p16_drift_block_before_science(self):
        for field, changed in (("dataset_sha256", {"price_sha": "e" * 64}),
                               ("param_sha256", {"param_sha": "f" * 64})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture = self._fixture(root)
                original = {path: path.read_bytes() for path in root.rglob("*.json")}
                with self._context(root, fixture, **changed):
                    with self.assertRaisesRegex(oracle.OracleBlockedError, field):
                        oracle.build_report(fixture[0], project_root=root, quiet=True)
                    cancelled = oracle.build_report(fixture[0], project_root=root, quiet=True,
                        preflight_approval=lambda _view: False)
                self.assertEqual(set(cancelled["preflight"].values()), {"RUN"})
                self.assertEqual({path: path.read_bytes() for path in root.rglob("*.json")}, original)

    def test_current_receipt_does_not_hide_same_source_score_contradiction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            store = oracle.OracleReplayStore(root, fixture[0].output_subdir)
            saved, = store.existing_completed_identities(seed=fixture[2], target="R40")
            conflicting = {**saved, "replay_policy": {**saved["replay_policy"],
                                                    "perfect_score_sha256": "e" * 64}}
            store.write(conflicting, store.read(saved), origin="synthetic_same_source_conflict")
            original = {path: path.read_bytes() for path in root.rglob("*.json")}
            with self._context(root, fixture):
                for approval in (None, lambda _view: self.fail("BLOCKED cannot ask for approval")):
                    with self.subTest(interactive=approval is not None), self.assertRaisesRegex(
                            oracle.OracleBlockedError, "multiple same-window scored replays"):
                        oracle.build_report(fixture[0], project_root=root, quiet=True,
                                            preflight_approval=approval)
            self.assertEqual({path: path.read_bytes() for path in root.rglob("*.json")}, original)

    def test_strict_source_probe_keeps_approved_producer_revision_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            store = oracle.OracleReplayStore(root, fixture[0].output_subdir)
            saved, = store.existing_completed_identities(seed=fixture[2], target="R40")
            known = {field: saved[field] for field in (
                "dataset", "dataset_sha256", "start_date", "end_date", "param_sha256", "benchmark")}
            known["replay_policy"] = {
                "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
                "period_stats_implementation": oracle.PERIOD_STATS_IMPLEMENTATION,
                "hindsight_target_contract": {"truth_source_sha256": "f" * 64},
            }
            candidates, conflict = store.inspect_current_source(
                seed=fixture[2], target="R40", expected_inputs=known)
            self.assertEqual(candidates, ())
            self.assertIsNone(conflict)
            self.assertNotIn("perfect_score_sha256", known["replay_policy"])

    def test_cold_runs_in_both_call_modes_create_identical_replay_identities(self):
        reports, identities = [], []
        for interactive in (False, True):
            with self.subTest(interactive=interactive), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture = self._fixture(root)
                store = oracle.OracleReplayStore(root, fixture[0].output_subdir)
                for path in store.cache_root.rglob("*.json"):
                    path.unlink()
                scores = {("2330", "2025-01-02"): .2, ("2330", "2025-12-22"): .8}
                observed = []
                def played(*, name, **_kwargs):
                    observed.append(name)
                    if name.startswith("Oracle"):
                        self.assertEqual(get_breakout_quality_ranking_source_context().score_source,
                                         SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)
                    return {"name": name}
                def metrics(_played, *, target):
                    return fixture[-1](target, 8. if target == "C96" else 18.,
                                       None if target == "C96" else 10.)
                prompts = []
                with self._context(root, fixture), \
                     patch.object(oracle, "_read_only_score_keys", side_effect=_fixture_read_score_keys), \
                     patch.object(oracle, "_load_or_build_oracle_price_scores", return_value=({"R40": scores}, {})) as source, \
                     patch.object(oracle, "_run_scenario", side_effect=played), \
                     patch.object(oracle, "_metrics", side_effect=metrics):
                    reports.append(oracle.build_report(fixture[0], project_root=root, quiet=True,
                        preflight_approval=(lambda view: prompts.append(view) or True) if interactive else None))
                    self.assertEqual(source.call_count, 1)
                self.assertEqual(len(observed), 2)
                self.assertEqual(len(prompts), int(interactive))
                identities.append({target: store.existing_completed_identities(seed=fixture[2], target=target)
                                   for target in ("C96", "R40")})
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(identities[0], identities[1])

    def test_old_complete_replay_adds_win_rate_without_overwriting_original(self):
        """One missing metric is a declared RUN, then an attested 100% REUSE."""
        from core.file_integrity import canonical_json_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            store = oracle.OracleReplayStore(root, fixture[0].output_subdir)
            seed = fixture[2]
            scores = {("2330", "2025-01-02"): 0.2, ("2330", "2025-12-22"): 0.8}
            import hashlib
            key_sha = hashlib.sha256("".join(
                f"{ticker}|{day}\n" for ticker, day in sorted(scores)
            ).encode()).hexdigest()
            # The independent owner fixture uses placeholder score/key hashes.
            # Construct a *real* digest-backed score/eligibility identity here,
            # rather than bypassing the formal RUN source-cohort proof.
            for name in ("C96", "R40"):
                previous, = store.existing_completed_identities(seed=seed, target=name)
                current = dict(previous)
                current["cohort"] = {**previous["cohort"], "cohort_rows": len(scores),
                    "first_common_day":"2025-01-02", "last_common_day":"2025-12-22"}
                if name != "C96":
                    policy = dict(current["replay_policy"])
                    policy["perfect_score_sha256"] = oracle._canonical_score_map_sha256(scores)
                    policy["canonical_dl_score_contract"] = {
                        **policy["canonical_dl_score_contract"],
                        "canonical_feature_eligible_keys_sha256": key_sha,
                        "oracle_scored_rows": len(scores),
                    }
                    current["replay_policy"] = policy
                store.write(current, store.read(previous), origin="synthetic_digest_attested")
                store._path(previous).unlink()
            identity, = store.existing_completed_identities(seed=seed, target="R40")
            path = store._path(identity)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("win_rate_pct", payload["observation"])
            payload["observation"].pop("win_rate_pct")
            payload["observation_sha256"] = canonical_json_sha256(payload["observation"])
            path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            original = path.read_bytes()
            self.assertIsNone(store.with_win_rate(identity, store.read(identity)))
            calls = []
            def simulate(*, name, **_kw):
                calls.append(name)
                return {"name": name}
            def summarize(_played, *, target):
                return fixture[-1](target, 8.0 if target == "C96" else 18.0,
                                   None if target == "C96" else 10.0)
            prompts = []
            with self._context(root, fixture), \
                 patch.object(oracle, "_load_or_build_oracle_price_scores", return_value=(
                     {"R40": scores}, {})) as source, \
                 patch.object(oracle, "_run_scenario", side_effect=simulate), \
                 patch.object(oracle, "_metrics", side_effect=summarize):
                cancelled = oracle.build_report(fixture[0], project_root=root, quiet=True,
                    preflight_approval=lambda view: prompts.append(view) or False)
                self.assertEqual(cancelled["status"], "CANCELLED")
                self.assertIn("REUSE 1 / RUN（需回放）1 / BLOCKED 0", prompts[-1])
                self.assertEqual(calls, [])
                report = oracle.build_report(fixture[0], project_root=root, quiet=True,
                    preflight_approval=lambda _view: True)
                self.assertEqual(len(calls), 1)
                self.assertEqual(source.call_count, 1)
                self.assertEqual(report["oracles"][0]["win_rate_pct"], 56.0)
                self.assertEqual(path.read_bytes(), original, "historical signed cache must be immutable")
                self.assertEqual(store.with_win_rate(identity, store.read(identity))["win_rate_pct"], 56.0)
                oracle.build_report(fixture[0], project_root=root, quiet=True,
                    preflight_approval=lambda view: prompts.append(view) or False)
                self.assertIn("REUSE 2 / RUN（需回放）0 / BLOCKED 0", prompts[-1])
                self.assertEqual(len(calls), 1)
            evidence = store._win_rate_path(identity)
            evidence_json = json.loads(evidence.read_text(encoding="utf-8"))
            evidence_json["win_rate_pct"] = 99.0
            evidence.write_text(json.dumps(evidence_json, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "win-rate evidence is corrupt"):
                store.with_win_rate(identity, store.read(identity))


class OracleWinRateReportCases(TestCase):
    def test_completed_trade_win_rate_aggregation_and_order(self):
        rows = []
        for seed, rate in zip((10, 20, 30, 40), (20.0, 40.0, 60.0, 80.0)):
            rows.append({"target":"R40", "seed":seed, "total_return_pct":90.0,
                "annualized_return_pct":15.0, "mdd_pct":10.0, "romd":9.0,
                "ev_a_r":1.0, "win_rate_pct":rate, "trades":10,
                "exposure_pct":45.0, "delta_romd_vs_c96":1.0,
                "yearly":[{"year":"2025", "strategy_return_pct":12.0,
                           "is_full_year":False}]})
        performance = oracle._aggregate(rows, target="R40")
        self.assertAlmostEqual(performance["win_rate_pct"], 50.0)
        self.assertAlmostEqual(performance["win_rate_pct_std"],
                               float(np.std([20.0,40.0,60.0,80.0],ddof=1)))
        baseline = {**performance, "target":"C96", "romd":8.0}
        better = {**performance, "target":"T40", "romd":12.0, "positive_seed_count":4}
        report = {"baseline":baseline, "oracles":[better,
                 {**performance, "positive_seed_count":4}],
                 "start_date":"2021-01-01", "end_date":"2026-03-02",
                 "common_cohort":{"cohort_rows":100}, "benchmark_id":"test",
                 "seeds":[10,20,30,40], "schema_version":8,
                 "definitions":{"R40":"return", "T40":"return"},
                 "limitations":["NOT PIT"]}
        md = oracle.render_result(report, target="markdown")
        table = md.split("## 1. Oracle Performance (8-seed mean)",1)[1].split("## 2.",1)[0]
        self.assertIn("Win Rate", table)
        self.assertNotIn("Delta RoMD", table)
        self.assertLess(table.index("C96 Same-Window"),table.index("R40"))
        self.assertLess(table.index("R40"),table.index("T40"))
        self.assertIn("50.00%", table)
        self.assertNotIn("Delta RoMD", md)
        self.assertNotIn("Paired Delta", md)
        self.assertIn("C96 Seed RoMD", md, "robustness retains independent same-seed baseline")
        self.assertIn("RoMD Mean", oracle.render_result(report, target="console"))

    def test_changed_economic_replay_blocks_win_rate_backfill(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        identity = replay_identity(dataset="synthetic", dataset_sha256="a"*64,
            target="C96", definition={"C96":"same-window"}, start="2021-01-01",
            end="2026-03-02", benchmark="synthetic", param_sha256="b"*64,
            seed=42, cohort={"cohort_rows":1}, ranking_policy="exact",
            ranking_options={}, evaluation_mode="oos", max_positions=10,
            replay_policy={})
        original = {"target":"C96", "seed":42, "total_return_pct":12.,
            "annualized_return_pct":8., "mdd_pct":5., "romd":2.4,
            "ev_a_r":0.4, "trades":10, "exposure_pct":50.,
            "yearly":[{"year":2025,"strategy_return_pct":12.,"is_full_year":True}]}
        with tempfile.TemporaryDirectory() as temp:
            store = OracleReplayStore(Path(temp), "oracle_test")
            store.write(identity, original, origin="legacy_signed")
            self.assertIsNone(store.with_win_rate(identity, original))
            with self.assertRaisesRegex(ValueError, "changes prior economic evidence"):
                store.write_win_rate_evidence(identity, original,
                    {**original, "romd":99., "win_rate_pct":70.0})
            self.assertFalse(store._win_rate_path(identity).exists())
            stored = store.write_win_rate_evidence(identity, original,
                                                  {**original, "win_rate_pct":70.0})
            self.assertEqual(stored["win_rate_pct"], 70.)
            self.assertEqual(store.read(identity),original)
            self.assertEqual(store.with_win_rate(identity, original),stored)
            with self.assertRaisesRegex(ValueError, "evidence conflict"):
                store.write_win_rate_evidence(identity, original,
                                              {**original, "win_rate_pct":20.0})



class OraclePriceSourceExecutionPolicyIsolationCases(TestCase):
    """Execution-only arms must never invalidate certified price score sources."""

    # This version's historical signed price-score implementation fingerprint.
    # A genuine price-score formula change must knowingly update this evidence
    # and invalidate affected artifacts; a new execution-only gate must not.
    _CERTIFIED_PRICE_FORMULA_SHA = (
        "d73b4ce20dd958cb75b094f49d6abac94d31e34ca7222fa5806675fd931964bd"
    )

    def test_execution_only_policy_does_not_enter_price_producer_identity(self):
        from services.audit.oracle_execution_policy import execution_gate_policies
        from core.oracle_positive_gate import POSITIVE_FULLCYCLE_NETPNL_GATE
        definition = _ACTIVE_ORACLE_DEFINITION
        before = replace(definition, source={**definition.source,
            "oracle_comparisons": tuple(row for row in definition.source["oracle_comparisons"]
                                        if row.get("positive_gate") is None)})
        before_plan, after_plan = (oracle._oracle_comparison_plan(item)
                                   for item in (before, definition))
        self.assertEqual(before_plan["recipes"], after_plan["recipes"])
        self.assertEqual(before_plan["providers"], after_plan["providers"])
        self.assertEqual(execution_gate_policies(before, before_plan), {})
        gates = execution_gate_policies(definition, after_plan)
        self.assertEqual(len(gates), 1)
        self.assertEqual(list(gates.values()), [POSITIVE_FULLCYCLE_NETPNL_GATE])
        self.assertEqual(oracle._price_score_algorithm_sha256(),
                         self._CERTIFIED_PRICE_FORMULA_SHA)
        self.assertEqual(oracle._price_score_replay_lineage(before),
                         oracle._price_score_replay_lineage(definition))

        # The nine preexisting price recipients retain byte-identical score
        # consumption and economic replay inputs, despite the new diagnostic.
        settings, base, exact = oracle._contract(definition)
        before_contracts = oracle._resolve_oracle_target_score_contracts(
            before, settings, before_plan)
        after_contracts = oracle._resolve_oracle_target_score_contracts(
            definition, settings, after_plan)
        for name in before_plan["selectors"]:
            self.assertEqual(before_contracts[name], after_contracts[name])
        price_target = next(name for name in before_plan["selectors"]
                            if name not in before_plan["daily_sources"])
        old_policy = oracle._arm_replay_policy(
            before, plan=before_plan, target=price_target, replay_policy={},
            score_sha256="a" * 64, score_contract=before_contracts[price_target])
        new_policy = oracle._arm_replay_policy(
            definition, plan=after_plan, target=price_target, replay_policy={},
            score_sha256="a" * 64, score_contract=after_contracts[price_target])
        self.assertEqual(old_policy, new_policy)
        self.assertNotIn("hindsight_extra_gate", new_policy)

    def test_execution_policy_must_be_registered_and_fullcycle_direct(self):
        from services.audit.oracle_execution_policy import (
            execution_gate_policies, OracleExecutionPolicyError,
        )
        definition = _ACTIVE_ORACLE_DEFINITION
        rows = list(definition.source["oracle_comparisons"])
        ordinary = next(i for i, row in enumerate(rows)
                        if row.get("positive_gate") is None)
        rows[ordinary] = {**rows[ordinary], "positive_gate": "unexpected_non_pit"}
        modified = replace(definition, source={**definition.source,
            "oracle_comparisons": tuple(rows)})
        with self.assertRaisesRegex(OracleExecutionPolicyError, "execution policy changed"):
            execution_gate_policies(modified, oracle._oracle_comparison_plan(modified))

    def test_generic_admission_hook_is_scoped_and_not_an_oracle_dependency(self):
        from core.portfolio_candidate_admission import (
            admitted_resource_candidates, scoped_candidate_admission,
        )
        rows = [{"ticker": "2330"}, {"ticker": "2317"}]
        self.assertEqual(admitted_resource_candidates(rows), rows)
        with self.assertRaisesRegex(RuntimeError, "nested"):
            with scoped_candidate_admission(lambda row: row["ticker"] == "2330"):
                self.assertEqual(admitted_resource_candidates(rows), rows[:1])
                with scoped_candidate_admission(lambda row: True):
                    pass
        self.assertEqual(admitted_resource_candidates(rows), rows)
        with self.assertRaisesRegex(ValueError, "intentional failure"):
            with scoped_candidate_admission(lambda row: True):
                raise ValueError("intentional failure")
        self.assertEqual(admitted_resource_candidates(rows), rows)



class OracleEightSeedSinglePipelineCases(TestCase):
    """An expanded aggregate consumes the same signed per-seed engines and receipts."""

    @staticmethod
    def _identity(seed, target, benchmark, *, price_sha="a" * 64, param_sha=None):
        from services.audit.oracle_replay_artifacts import replay_identity
        return replay_identity(
            dataset="full", dataset_sha256=price_sha,
            target=target, definition={"C96": "same-window", "FUTURE_TARGET": "generic target",
                                       "NEXT_TARGET": "new generic target"},
            start="2021-01-01", end="2026-03-02", benchmark=benchmark,
            param_sha256=param_sha or f"{seed:064x}", seed=seed,
            cohort={"cohort_rows": 100, "research_cutoff": "2026-03-02"},
            ranking_policy="exact", ranking_options={"max_rank": 10},
            evaluation_mode="rolling", max_positions=10,
            replay_policy={"oracle_parity_implementation": "same"},
        )

    @staticmethod
    def _observation(seed, target):
        observation = {
            "target": target, "seed": int(seed),
            "total_return_pct": 20.0, "annualized_return_pct": 10.0,
            "mdd_pct": 5.0, "romd": 4.0, "ev_a_r": 0.4,
            "trades": 8, "exposure_pct": 40.0,
            "win_rate_pct": 50.0, "yearly": _synthetic_full_window_yearly(return_pct=12.0),
        }
        if target != "C96":
            observation["delta_romd_vs_c96"] = 0.0
        return observation

    def test_single_8_seed_current_registration_preserves_historical_prefix(self):
        from core.training_policy import (
            get_robustness_benchmark_policy_snapshot,
            benchmark_seed_execution_equivalent,
        )
        old = get_robustness_benchmark_policy_snapshot("end_to_end_v1")
        current = get_robustness_benchmark_policy_snapshot()
        self.assertEqual(current["seed_count"], 8)
        self.assertEqual(current["benchmark_id"], "end_to_end_v2")
        self.assertEqual(tuple(old["resolved_seeds"]), tuple(current["resolved_seeds"][:4]))
        self.assertEqual(len(set(current["resolved_seeds"])), 8)
        self.assertTrue(all(benchmark_seed_execution_equivalent(
            recorded_benchmark=old["benchmark_id"], active_benchmark=current["benchmark_id"], seed=s,
        ) for s in old["resolved_seeds"]))
        self.assertFalse(benchmark_seed_execution_equivalent(
            recorded_benchmark=old["benchmark_id"], active_benchmark=current["benchmark_id"],
            seed=current["resolved_seeds"][-1],
        ))
        self.assertFalse(benchmark_seed_execution_equivalent(
            recorded_benchmark=current["benchmark_id"], active_benchmark=old["benchmark_id"],
            seed=old["resolved_seeds"][0],
        ))

    def test_old_four_signed_replays_reuse_new_four_run_and_future_target_uses_8(self):
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        v1 = get_robustness_benchmark_policy_snapshot("end_to_end_v1")
        current = get_robustness_benchmark_policy_snapshot()
        with tempfile.TemporaryDirectory(prefix="oracle_eight_seed_single_pipeline_") as temp:
            store = OracleReplayStore(Path(temp), "breakout_quality/reusable/target_oracle_romd")
            originals = {}
            for seed in v1["resolved_seeds"]:
                for target in ("C96", "FUTURE_TARGET"):
                    identity = self._identity(seed, target, v1["benchmark_id"])
                    store.write(identity, self._observation(seed, target), origin="canonical_replay")
                    originals[store._path(identity)] = store._path(identity).read_bytes()
            planned = {
                (seed, target): self._identity(seed, target, current["benchmark_id"])
                for seed in current["resolved_seeds"]
                for target in ("C96", "FUTURE_TARGET")
            }
            plan = store.plan_execution(planned)
            for seed in current["resolved_seeds"]:
                for target in ("C96", "FUTURE_TARGET"):
                    self.assertEqual(plan.action(seed, target),
                                     "REUSE" if seed in v1["resolved_seeds"] else "RUN")
            self.assertEqual(sum(action == "REUSE" for action in plan.actions().values()), 8)
            self.assertEqual(sum(action == "RUN" for action in plan.actions().values()), 8)
            self.assertFalse(plan.blocked())
            self.assertTrue(all(path.read_bytes() == old_bytes for path, old_bytes in originals.items()))
            self.assertTrue(all(not store._path(identity).exists() for identity in planned.values()))
            # A genuinely new Target has no four-seed compatibility execution path.
            unseen = {
                (seed, "NEXT_TARGET"): self._identity(seed, "NEXT_TARGET", current["benchmark_id"])
                for seed in current["resolved_seeds"]
            }
            self.assertEqual(sum(action == "RUN" for action in store.plan_execution(unseen).actions().values()), 8)

    def test_unrelated_changed_inputs_are_never_reused_as_8_seed_expansion(self):
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        v1 = get_robustness_benchmark_policy_snapshot("end_to_end_v1")
        current = get_robustness_benchmark_policy_snapshot()
        seed = int(v1["resolved_seeds"][0])
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            old = self._identity(seed, "C96", v1["benchmark_id"])
            store.write(old, self._observation(seed, "C96"), origin="canonical_replay")
            altered = self._identity(seed, "C96", current["benchmark_id"], param_sha="f" * 64)
            self.assertEqual(store.plan_execution({(seed, "C96"): altered}).action(seed, "C96"), "BLOCKED")
            self.assertEqual(store.plan_execution({(seed, "C96"): altered},
                    allow_historical_execution_input_drift=True).action(seed, "C96"), "RUN")
            self.assertEqual(store.plan_execution({(seed, "C96"): self._identity(
                    seed, "C96", current["benchmark_id"])}).action(seed, "C96"), "REUSE")

    def test_canonical_optimizer_artifact_reader_uses_verified_prior_seed_provenance(self):
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        from services.optimizer import strategy_param_service as producer
        v1 = get_robustness_benchmark_policy_snapshot("end_to_end_v1")
        v2 = get_robustness_benchmark_policy_snapshot()
        seed = int(v1["resolved_seeds"][0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signed = root / "optimizer_p16_v1.json"
            signed.write_text('{"fitting":"identical"}', encoding="utf-8")
            old_contract = {"benchmark_id": v1["benchmark_id"], "optimizer_seed": seed,
                            "trials_per_fold": 300}
            new_contract = {**old_contract, "benchmark_id": v2["benchmark_id"]}
            ready = {"ready": True, "status": "READY", "action": "REUSE",
                     "path": signed, "manifest_path": root / "signed_manifest.json",
                     "scientific_identity": {"fitting": "identical"},
                     "build_contract": old_contract}
            missing = {"ready": False, "status": "ARTIFACT_MISSING_OR_INVALID", "action": "BUILD",
                       "path": root / "optimizer_p16_v2_missing.json",
                       "manifest_path": root / "optimizer_p16_v2_missing_manifest.json",
                       "scientific_identity": {"fitting": "identical"},
                       "build_contract": new_contract}
            observations = []

            def inspect(_root, *, benchmark_id, **_kwargs):
                observations.append(benchmark_id)
                return dict(ready if benchmark_id == v1["benchmark_id"] else missing)

            kwargs = dict(training_profile_id="production_multiseed_current",
                          benchmark_id=v2["benchmark_id"], replication_seed=seed,
                          policy="base-finalists-agree", comparison_end_date="2026-03-02",
                          dataset="full", max_positions=10, rotation="off",
                          fixed_risk=1.0, max_position_cap_pct=100.0)
            with patch.object(producer, "inspect_fixed_multiseed_robustness_strategy_parameter_artifact",
                              side_effect=inspect):
                result = producer.resolve_fixed_multiseed_robustness_strategy_parameter_artifact(
                    root, **kwargs)
            self.assertEqual(observations, [v2["benchmark_id"], v1["benchmark_id"]])
            self.assertTrue(result["ready"])
            self.assertEqual(result["path"], signed)
            self.assertEqual(result["current_benchmark_id"], v2["benchmark_id"])
            self.assertEqual(result["origin_benchmark_id"], v1["benchmark_id"])
            # A damaged current manifest is NOT a reason to silently consume the old P16.
            missing["manifest_path"].write_text('{"damaged":true}', encoding="utf-8")
            observations.clear()
            with patch.object(producer, "inspect_fixed_multiseed_robustness_strategy_parameter_artifact",
                              side_effect=inspect):
                damaged = producer.resolve_fixed_multiseed_robustness_strategy_parameter_artifact(
                    root, **kwargs)
            self.assertFalse(damaged["ready"])
            self.assertEqual(observations, [v2["benchmark_id"]])
            # New seeds have no ancestor and therefore no old execution path.
            missing["manifest_path"].unlink()
            observations.clear()
            with patch.object(producer, "inspect_fixed_multiseed_robustness_strategy_parameter_artifact",
                              side_effect=inspect):
                newer = producer.resolve_fixed_multiseed_robustness_strategy_parameter_artifact(
                    root, **{**kwargs, "replication_seed": v2["resolved_seeds"][-1]})
            self.assertFalse(newer["ready"])
            self.assertEqual(observations, [v2["benchmark_id"]])

    def test_canonical_optimizer_producer_reuses_verified_history_without_copying(self):
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        from services.optimizer import strategy_param_service as producer
        v1 = get_robustness_benchmark_policy_snapshot("end_to_end_v1")
        v2 = get_robustness_benchmark_policy_snapshot()
        seed = int(v1["resolved_seeds"][0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            historical_path = root / "historical_signed.json"
            historical_path.write_text('{"selector":"base_finalists_agree"}', encoding="utf-8")
            kwargs = dict(training_profile_id="production_multiseed_current",
                          benchmark_id=v2["benchmark_id"], replication_seed=seed,
                          policy="base-finalists-agree", comparison_end_date="2026-03-02",
                          dataset="full", max_positions=10, rotation="off",
                          fixed_risk=1.0, max_position_cap_pct=100.0)
            historical = dict(
                ready=True, action="REUSE", status="READY", path=historical_path,
                manifest_path=root / "original_manifest.json", coverage_end="2026-03-02",
                build_contract={"benchmark_id": v1["benchmark_id"],
                                "robustness_replication_seed": seed},
                scientific_identity={"same_seed": True},
            )
            original_bytes = historical_path.read_bytes()
            with patch.object(producer, "resolve_fixed_multiseed_robustness_strategy_parameter_artifact",
                              return_value=historical) as canonical, \
                 patch.object(producer, "compute_strategy_param_scientific_sha256",
                              return_value="0" * 64), \
                 patch.object(producer, "inspect_fixed_multiseed_robustness_strategy_parameter_artifact",
                              side_effect=AssertionError("legacy bypass must not occur")):
                result = producer.ensure_fixed_multiseed_robustness_strategy_parameter_artifact(
                    root, **kwargs)
            self.assertEqual(result["action"], "REUSE")
            self.assertEqual(result["path"], historical_path)
            canonical.assert_called_once()
            self.assertEqual(historical_path.read_bytes(), original_bytes)
            with patch.object(producer, "resolve_fixed_multiseed_robustness_strategy_parameter_artifact",
                              side_effect=AssertionError("force must never target historical path")), \
                 patch.object(producer, "inspect_fixed_multiseed_robustness_strategy_parameter_artifact",
                              side_effect=RuntimeError("CURRENT_ONLY")) as current:
                with self.assertRaisesRegex(RuntimeError, "CURRENT_ONLY"):
                    producer.ensure_fixed_multiseed_robustness_strategy_parameter_artifact(
                        root, force_rebuild=True, **kwargs)
            current.assert_called_once()
            self.assertEqual(historical_path.read_bytes(), original_bytes)


class OracleEightSeedParameterDependencyRegressionCases(TestCase):
    """[1] must coordinate missing P16 without changing read-only Oracle semantics."""

    @staticmethod
    def _fixture(*, ready_seeds, invalid_seed=None):
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        seeds = get_robustness_benchmark_policy_snapshot()["resolved_seeds"]
        rows = []
        for seed in seeds:
            ready = int(seed) in ready_seeds
            invalid = seed == invalid_seed
            rows.append(dict(
                seed=seed, ready=ready,
                action="REUSE" if ready else "REBUILD" if invalid else "BUILD",
                status="READY" if ready else "MANIFEST_IDENTITY_MISMATCH" if invalid
                       else "ARTIFACT_MISSING_OR_INVALID",
                path=Path(f"/unittest/P16/{seed}.json"),
                manifest_path=Path(f"/unittest/P16/{seed}.manifest.json"),
            ))
        return dict(
            robustness=SimpleNamespace(benchmark_id="end_to_end_v2"),
            producer_kwargs={
                "training_profile_id": "broker_cash_min_best_seed42_current",
                "benchmark_id": "end_to_end_v2", "policy": "base-finalist-best",
                "comparison_end_date": "2026-03-02", "dataset": "full",
                "max_positions": 10, "rotation": "off", "fixed_risk": 0.05,
                "max_position_cap_pct": 1.0,
            },
            rows=tuple(rows),
        )

    def test_p16_inspection_uses_one_registered_source_for_all_benchmark_seeds(self):
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        current = get_robustness_benchmark_policy_snapshot()
        legacy = get_robustness_benchmark_policy_snapshot("end_to_end_v1")
        seen = []

        def inspect(_root, *, replication_seed, **kwargs):
            seen.append((replication_seed, kwargs))
            return dict(
                ready=replication_seed in legacy["resolved_seeds"],
                action="REUSE" if replication_seed in legacy["resolved_seeds"] else "BUILD",
                status="READY" if replication_seed in legacy["resolved_seeds"] else "ARTIFACT_MISSING_OR_INVALID",
                path=Path(f"/mock/P16/{replication_seed}.json"),
                manifest_path=Path(f"/mock/P16/{replication_seed}.manifest.json"),
            )

        with patch.object(oracle, "resolve_fixed_multiseed_robustness_strategy_parameter_artifact",
                          side_effect=inspect):
            plan = oracle._oracle_parameter_reuse_plan(_ACTIVE_ORACLE_DEFINITION, Path("/mock"))
        self.assertEqual([row["seed"] for row in plan["rows"]], current["resolved_seeds"])
        self.assertEqual([row["action"] for row in plan["rows"]], ["REUSE"] * 4 + ["BUILD"] * 4)
        self.assertEqual(len(seen), len(current["resolved_seeds"]))
        self.assertTrue(all(kwargs["benchmark_id"] == current["benchmark_id"] for _, kwargs in seen))
        self.assertTrue(all(kwargs["training_profile_id"] ==
                            "broker_cash_min_best_seed42_current" for _, kwargs in seen))
        self.assertTrue(all(kwargs["policy"] == "base-finalist-best" for _, kwargs in seen))
        self.assertEqual(plan["producer_kwargs"]["comparison_end_date"], "2026-03-02")

    def test_p16_missing_cannot_be_accepted_by_read_only_audit(self):
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        seeds = get_robustness_benchmark_policy_snapshot()["resolved_seeds"]
        fixture = self._fixture(ready_seeds=set(seeds[:4]))
        with patch.object(oracle, "inspect_oracle_parameter_prerequisites", return_value=fixture):
            status = oracle.collect_status(_ACTIVE_ORACLE_DEFINITION, project_root=Path("/mock"))
        self.assertEqual(status["status"], "BLOCKED")
        self.assertIn(str(seeds[4]), status["reason"])
        self.assertIn("只讀Audit", status["reason"])

    @staticmethod
    def _approved_preview(fixture):
        rows = tuple(fixture["rows"])
        seeds = tuple(row["seed"] for row in rows)
        return {
            "seed_set": seeds,
            "producer_kwargs": fixture["producer_kwargs"],
            "missing_p16": tuple(row["seed"] for row in rows if not row["ready"]),
            "preflight": {(row["seed"], "C96"): "REUSE" if row["ready"] else "RUN"
                          for row in rows},
            "text": "[Target 驗證 Fixed | Portfolio Replay 正式計畫]\n"
                    "Target  S1  S2  S3  S4  S5  S6  S7  S8\n"
                    "C96     REUSE REUSE REUSE REUSE RUN RUN RUN RUN",
        }

    def test_research_builds_only_four_approved_missing_seeds(self):
        from services.research import target_oracle_application as application
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        seeds = get_robustness_benchmark_policy_snapshot()["resolved_seeds"]
        ready = set(seeds[:4])
        built = []
        def inspect(_definition, *, project_root):
            return self._fixture(ready_seeds=ready)
        def resolve(_root, *, replication_seed, **kwargs):
            return {"ready": replication_seed in ready,
                    "action": "REUSE" if replication_seed in ready else "BUILD",
                    "status": "READY" if replication_seed in ready else "ARTIFACT_MISSING_OR_INVALID"}
        def build(_root, *, replication_seed, **kwargs):
            self.assertNotIn(replication_seed, ready)
            built.append(replication_seed)
            ready.add(replication_seed)
            return {"action": "BUILD"}
        preview = self._approved_preview(self._fixture(ready_seeds=ready))
        with patch.object(application, "inspect_oracle_parameter_prerequisites", side_effect=inspect), \
             patch.object(application, "resolve_fixed_multiseed_robustness_strategy_parameter_artifact", side_effect=resolve), \
             patch.object(application, "ensure_fixed_multiseed_robustness_strategy_parameter_artifact", side_effect=build), \
             redirect_stdout(io.StringIO()):
            application.ensure_authorized_oracle_parameters(
                project_root=Path("/mock"), definition=_ACTIVE_ORACLE_DEFINITION,
                approved_plan=preview,
            )
        self.assertEqual(built, seeds[4:])
        self.assertEqual(ready, set(seeds))

    def test_all_ready_never_invokes_optimizer(self):
        from services.research import target_oracle_application as application
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        seeds = get_robustness_benchmark_policy_snapshot()["resolved_seeds"]
        fixture = self._fixture(ready_seeds=set(seeds))
        with patch.object(application, "inspect_oracle_parameter_prerequisites", return_value=fixture), \
             patch.object(application, "resolve_fixed_multiseed_robustness_strategy_parameter_artifact") as probe, \
             patch.object(application, "ensure_fixed_multiseed_robustness_strategy_parameter_artifact") as produce, \
             redirect_stdout(io.StringIO()):
            application.ensure_authorized_oracle_parameters(
                project_root=Path("/mock"), definition=_ACTIVE_ORACLE_DEFINITION,
                approved_plan=self._approved_preview(fixture),
            )
        probe.assert_not_called()
        produce.assert_not_called()

    def test_p16_conflict_and_new_unapproved_seed_never_fit(self):
        from services.research import target_oracle_application as application
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        seeds = get_robustness_benchmark_policy_snapshot()["resolved_seeds"]
        fixture = self._fixture(ready_seeds=set(seeds[:4]), invalid_seed=seeds[4])
        authorized = self._approved_preview(self._fixture(ready_seeds=set(seeds[:4])))
        with patch.object(application, "inspect_oracle_parameter_prerequisites", return_value=fixture), \
             patch.object(application, "ensure_fixed_multiseed_robustness_strategy_parameter_artifact") as producer, \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "身分衝突"):
                application.ensure_authorized_oracle_parameters(
                    project_root=Path("/mock"), definition=_ACTIVE_ORACLE_DEFINITION,
                    approved_plan=authorized,
                )
            producer.assert_not_called()
        lost = self._fixture(ready_seeds=set(seeds[:3]))
        with patch.object(application, "inspect_oracle_parameter_prerequisites", return_value=lost), \
             patch.object(application, "ensure_fixed_multiseed_robustness_strategy_parameter_artifact") as producer, \
             self.assertRaisesRegex(RuntimeError, "原為 REUSE"):
            application.ensure_authorized_oracle_parameters(
                project_root=Path("/mock"), definition=_ACTIVE_ORACLE_DEFINITION,
                approved_plan=authorized,
            )
        producer.assert_not_called()

    def test_approved_replay_cells_cannot_gain_new_run_or_target(self):
        """Canonical Oracle, not UI text, enforces the approved cell matrix."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = OracleOnePlanRealCacheCases()._fixture(root)
            definition, bench, seed, price_sha, param_sha, *_ = fixture
            previous = {(seed, "C96"): "REUSE", (seed, "R40"): "REUSE",
                        (seed, "T40"): "REUSE"}  # T40 is actually RUN
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(bench, [(seed, root/"p16.json")])), \
                 patch.object(oracle, "_prepare_oracle_parameters", return_value=(
                     [(seed, param_sha, "synthetic", None, None)], {})), \
                 patch.object(oracle, "price_inventory_sha256", return_value=price_sha), \
                 patch.object(oracle, "_execute_single_stage_missing", side_effect=AssertionError(
                     "No unauthorized Target or Portfolio production")) as producer:
                with self.assertRaisesRegex(oracle.OracleBlockedError, "身分或計畫已改變"):
                    oracle.build_report(
                        definition, project_root=root, quiet=True,
                        preflight_approval=lambda _view: self.fail("Unapproved new RUN"),
                        approved_replay_decisions=previous,
                    )
                # An original RUN becoming REUSE consumes *less* work and is legal.
                authorized = {**previous, (seed, "T40"): "RUN"}
                cancelled = oracle.build_report(
                    definition, project_root=root, quiet=True,
                    preflight_approval=lambda _view: False,
                    approved_replay_decisions=authorized,
                )
            self.assertEqual(cancelled["status"], "CANCELLED")
            producer.assert_not_called()

    def test_four_signed_old_seeds_all_reuse_after_8_seed_upgrade(self):
        """Four existing v1 signed Replay receipts survive v2 membership change."""
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definition, _b, old_seed, price_sha, param_sha, *_ = (
                OracleOnePlanRealCacheCases()._fixture(root))
            seeds = get_robustness_benchmark_policy_snapshot()["resolved_seeds"]
            store = OracleReplayStore(root, definition.output_subdir)
            # Isolated synthetic receipts from the same source contract; the
            # production system never copies one seed's real P16 or Replay.
            for target in ("C96", "R40"):
                identity, = store.existing_completed_identities(seed=old_seed, target=target)
                original = store.read(identity)
                for seed in seeds[1:4]:
                    store.write({**identity, "seed": seed}, {**original, "seed": seed},
                                origin="synthetic_8seed_upgrade_reuse")
            fixture = self._fixture(ready_seeds=set(seeds[:4]))
            fixture["robustness"] = SimpleNamespace(benchmark_id="end_to_end_v2")
            def prepare(_definition, *, root, parameter_paths, **_kwargs):
                return ([(seed, param_sha, "synthetic", None, None)
                         for seed, _path in parameter_paths], {})
            with patch.object(oracle, "inspect_oracle_parameter_prerequisites", return_value=fixture), \
                 patch.object(oracle, "_prepare_oracle_parameters", side_effect=prepare), \
                 patch.object(oracle, "price_inventory_sha256", return_value=price_sha), \
                 patch.object(oracle, "_read_only_score_keys", side_effect=AssertionError(
                     "No expensive Target truth build")), \
                 patch.object(oracle, "_execute_single_stage_missing", side_effect=AssertionError(
                     "No Portfolio replay before authorization")):
                view = oracle.preview_oracle_replay_plan(definition, project_root=root)
            for seed in seeds[:4]:
                self.assertEqual(view["preflight"][(seed, "C96")], "REUSE")
                self.assertEqual(view["preflight"][(seed, "R40")], "REUSE")
            for seed in seeds[4:]:
                self.assertEqual({view["preflight"][(seed, key)]
                                  for key in ("C96", "R40", "T40")}, {"RUN"})
            self.assertEqual(sum(x == "REUSE" for x in view["preflight"].values()), 8)

    def test_partial_missing_p16_still_previews_registered_target_matrix(self):
        """Real signed first-seed Replay proves REUSE; other P16s are not guessed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definition, _benchmark, first, price_sha, param_sha, _proof, _sample = (
                OracleOnePlanRealCacheCases()._fixture(root))
            from core.training_policy import get_robustness_benchmark_policy_snapshot
            seeds = get_robustness_benchmark_policy_snapshot()["resolved_seeds"]
            fixture = self._fixture(ready_seeds=set(seeds[:4]))
            fixture["robustness"] = SimpleNamespace(benchmark_id="end_to_end_v2")
            def prepare(_definition, *, root, parameter_paths, **_kwargs):
                return ([(seed, param_sha, "synthetic", None, None)
                         for seed, _path in parameter_paths], {})
            with patch.object(oracle, "inspect_oracle_parameter_prerequisites", return_value=fixture), \
                 patch.object(oracle, "_prepare_oracle_parameters", side_effect=prepare), \
                 patch.object(oracle, "price_inventory_sha256", return_value=price_sha), \
                 patch.object(oracle, "_read_only_score_keys", side_effect=AssertionError("No Target producer before approval")), \
                 patch.object(oracle, "_execute_single_stage_missing", side_effect=AssertionError("No Portfolio Replay")), \
                 patch.object(oracle, "load_profile_continuous_ranker_data", side_effect=AssertionError("No DL features")), \
                 redirect_stdout(io.StringIO()):
                preview = oracle.preview_oracle_replay_plan(definition, project_root=root)
            self.assertEqual(preview["status"], "PREVIEW")
            self.assertEqual(preview["seed_set"], tuple(seeds))
            self.assertEqual(preview["missing_p16"], tuple(seeds[4:]))
            self.assertEqual(len(preview["preflight"]), 8 * 3)
            self.assertEqual(preview["preflight"][(first, "C96")], "REUSE")
            self.assertEqual(preview["preflight"][(first, "R40")], "REUSE")
            for seed in seeds[4:]:
                self.assertEqual({preview["preflight"][(seed, target)]
                                  for target in ("C96", "R40", "T40")}, {"RUN"})
            self.assertIn("Target", preview["text"])
            self.assertIn("T40", preview["text"])
            self.assertNotIn("P16 前置工件計畫", preview["text"])


class OracleInventoryAndSourceIsolationRegressionCases(TestCase):
    """Adding/removing research Targets must not invalidate unrelated Replay."""

    def test_both_formal_research_one_views_exclude_historical_60d(self):
        from core.audit_registry import resolve_current_oracle_comparisons
        from services.research.target_dl_overview import (
            collect_target_dl_overview, render_target_dl_overview,
        )
        definition = _ACTIVE_ORACLE_DEFINITION
        live = resolve_current_oracle_comparisons(definition.source)
        names = [row["target"] for row in live]
        self.assertEqual(len(names), 16)
        self.assertEqual(names, [name for name, _ in oracle._oracle_comparison_plan(definition)["targets"]])
        self.assertTrue(set(definition.source["historical_only_oracle_targets"]).isdisjoint(names))
        self.assertFalse(any("MinTrade60" in name for name in names))
        # The actual [1] -> [1] renderer and [1] -> [2] collector, not just
        # registration metadata, must use the very same active membership.
        decisions = {(seed, name): "REUSE" for seed in range(1, 9)
                     for name in ("C96", *names)}
        visible = oracle.render_replay_preflight(
            label="Fixed", seeds=tuple(range(1, 9)), targets=["C96", *names],
            decisions=decisions,
        )
        overview = render_target_dl_overview(collect_target_dl_overview(ROOT))
        self.assertNotIn("MinTrade60", visible)
        self.assertNotIn("MinTrade60", overview)
        self.assertEqual(len(decisions), 8 * 17)

    def test_historical_only_target_cannot_enter_plan_even_if_misregistered(self):
        from core.audit_registry import resolve_current_oracle_comparisons
        definition = _ACTIVE_ORACLE_DEFINITION
        past = "DailyUniversal_MinTrade60_NetR"
        inserted = {"target": past, "description": "accidentally enabled",
                    "score": "daily_trade_path_net_r", "provider": "daily_trade40_profile",
                    "horizon_sessions": 60, "selector": "exact"}
        source = {**definition.source,
                  "oracle_comparisons": (*definition.source["oracle_comparisons"], inserted)}
        with self.assertRaisesRegex(ValueError, "Historical-only Oracle target"):
            resolve_current_oracle_comparisons(source)
        with self.assertRaisesRegex(oracle.OracleBlockedError, "retired Target identity"):
            oracle._oracle_comparison_plan(replace(definition, source=source))

    def test_new_unrelated_target_does_not_change_old_price_implementation_sha(self):
        from services.audit.oracle_score_artifacts import oracle_daily_source_fingerprint
        original = _ACTIVE_ORACLE_DEFINITION
        recipe = {"target": "SyntheticPriceNew", "description": "new unrelated price recipient",
                  "score": "mfe", "provider": "pure_mfe40_provider", "selector": "exact"}
        new = replace(original, source={**original.source,
                      "oracle_comparisons": (*original.source["oracle_comparisons"], recipe)})
        before = oracle._price_score_replay_lineage(original)
        self.assertEqual(before, oracle._price_score_replay_lineage(new))
        self.assertEqual(
            oracle._price_score_algorithm_sha256(),
            "d73b4ce20dd958cb75b094f49d6abac94d31e34ca7222fa5806675fd931964bd",
        )
        # Independently verified 154445 source code had the same four Oracle
        # provider contracts and executing price algorithms.  No Target list
        # mutation is allowed to invalidate the signed source lineage.
        self.assertEqual(oracle_daily_source_fingerprint(),
                         "b7bdeb766babdb66ac5084125d08f2bf59390843e4dc09eb84645c093a3c770d")

    def test_unrelated_model_registration_is_not_price_source_drift(self):
        from services.audit import oracle_score_artifacts as source
        from core.file_integrity import compute_file_sha256 as original
        from config import breakout_quality as cfg
        before = source.oracle_daily_source_fingerprint()
        ignored = {"core/breakout_quality_registry.py", "core/breakout_quality_runtime.py",
                   "config/breakout_quality.py"}
        def only_registry_changes(path):
            if Path(path).resolve().relative_to(ROOT).as_posix() in ignored:
                return "f" * 64
            return original(Path(path))
        with patch.object(source, "compute_file_sha256", side_effect=only_registry_changes):
            self.assertEqual(source.oracle_daily_source_fingerprint(), before)
        # The actual price cohort / feature owner must still invalidate.
        def feature_changes(path):
            if Path(path).resolve() == ROOT / "filters/breakout_quality/features.py":
                return "e" * 64
            return original(Path(path))
        with patch.object(source, "compute_file_sha256", side_effect=feature_changes):
            self.assertNotEqual(source.oracle_daily_source_fingerprint(), before)
        # Price-affecting configuration cannot use the historical alias.
        with patch.object(cfg, "BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE", "2022-01-01"):
            self.assertNotEqual(source.oracle_daily_source_fingerprint(), before)
        # Source code unrelated to a new model must never bypass a changed
        # general data loader unless that exact change was attested.
        def loader_changes(path):
            if Path(path).resolve() == ROOT / "filters/breakout_quality/daily_ranker_data.py":
                return "d" * 64
            return original(Path(path))
        with patch.object(source, "compute_file_sha256", side_effect=loader_changes):
            self.assertNotEqual(source.oracle_daily_source_fingerprint(), before)


class OracleTargetCatalogAdditionDeletionSafetyCases(TestCase):
    """Add/delete Target names without modifying any executable price/Min trade.

    These fast source-contract cases complement the physical multi-copy signed
    Replay regression run at delivery. Both directions are necessary: a change
    that only works for adding but not removing a Target is not a valid SSOT.
    """

    def test_add_then_delete_oracle_target_does_not_mutate_price_lineage(self):
        from services.audit.oracle_score_artifacts import oracle_daily_source_fingerprint
        original = _ACTIVE_ORACLE_DEFINITION
        before = oracle._price_score_replay_lineage(original)
        before_catalog = oracle._oracle_comparison_plan(original)
        add = {"target": "SyntheticUnrelatedPriceControl", "description": "read-only registry test",
               "score": "mfe", "provider": "pure_mfe40_provider", "selector": "exact"}
        added = replace(original, source={**original.source,
             "oracle_comparisons": (*original.source["oracle_comparisons"], add)})
        self.assertEqual(len(oracle._oracle_comparison_plan(added)["targets"]),
                         len(before_catalog["targets"]) + 1)
        self.assertEqual(before, oracle._price_score_replay_lineage(added))
        reduced = replace(added, source={**added.source,
             "oracle_comparisons": tuple(item for item in added.source["oracle_comparisons"]
                                         if item["target"] not in {"CG40_EqualRank", add["target"]})})
        self.assertEqual(len(oracle._oracle_comparison_plan(reduced)["targets"]),
                         len(before_catalog["targets"]) - 1)
        self.assertEqual(before, oracle._price_score_replay_lineage(reduced))
        restored = replace(reduced, source=original.source)
        self.assertEqual(before, oracle._price_score_replay_lineage(restored))
        self.assertEqual(oracle_daily_source_fingerprint(), before["source_code_sha256"])

    def test_add_and_remove_model_selection_configs_preserve_only_price_identity(self):
        from config import breakout_quality as config
        from services.audit import oracle_score_artifacts as producer
        price = producer.oracle_daily_source_fingerprint()
        config_sha = producer._oracle_price_configuration_sha256()
        with patch.object(config, "BREAKOUT_QUALITY_MODEL_TEST_FUTURE_PROFILE", "synthetic", create=True):
            self.assertEqual(producer._oracle_price_configuration_sha256(), config_sha)
            self.assertEqual(producer.oracle_daily_source_fingerprint(), price)
        with patch.object(config, "BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_PROFILES",
                          (*config.BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_PROFILES,
                           ("MR-SYNTHETIC", "synthetic_profile"))):
            self.assertEqual(producer.oracle_daily_source_fingerprint(), price)
        self.assertEqual(producer.oracle_daily_source_fingerprint(), price)
        with patch.object(config, "BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE", "2022-01-01"):
            self.assertNotEqual(producer.oracle_daily_source_fingerprint(), price)

    def test_min_trade_new_target_catalog_branch_does_not_rewrite_economic_producer(self):
        from services.audit import mintrade_truth_source_attestation as owner
        src = ROOT / owner._SOURCE_OWNER
        current = owner.attested_mintrade_source_sha256(owner._SOURCE_OWNER, src)
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / src.name
            source = src.read_text(encoding="utf-8")
            marker = '    if outcome_metric not in {"net_r", "net_pnl"}:\n'
            self.assertGreaterEqual(source.count(marker), 1)
            # The first match is the naming-only target resolver, not the
            # independently attested economic source function below it.
            added = source.replace(marker,
                '    if horizon_sessions == 777 and outcome_metric == "net_r":\n'
                '        return "synthetic_target_777", "LABEL-synthetic_target_777"\n' + marker, 1)
            staged.write_text(added, encoding="utf-8")
            self.assertEqual(owner.attested_mintrade_source_sha256(owner._SOURCE_OWNER, staged), current)
            self.assertEqual(owner._executable_owner_ast_sha256(staged),
                             owner._CERTIFIED_EXECUTABLE_OWNER_AST_SHA256)
            # Delete that new Target: the exact previously signed identity returns.
            staged.write_text(source, encoding="utf-8")
            self.assertEqual(owner.attested_mintrade_source_sha256(owner._SOURCE_OWNER, staged), current)
            # A documentation-only change also remains non-executable.
            original_doc = 'Compute all feature-eligible stock-day outcomes from a formal Min seed.'
            self.assertEqual(source.count(original_doc), 1)
            staged.write_text(source.replace(original_doc,
                'Document-only description of the same canonical Min producer.', 1),
                encoding="utf-8")
            self.assertEqual(owner.attested_mintrade_source_sha256(owner._SOURCE_OWNER, staged), current)
            # A change to the real executed money/NetR routing MUST invalidate.
            old = 'include_net_pnl=(metric == "net_pnl")'
            self.assertEqual(source.count(old), 1)
            staged.write_text(source.replace(old, 'include_net_pnl=(metric == "net_r")', 1),
                              encoding="utf-8")
            self.assertNotEqual(owner.attested_mintrade_source_sha256(owner._SOURCE_OWNER, staged), current)
            self.assertNotEqual(owner._executable_owner_ast_sha256(staged),
                                owner._CERTIFIED_EXECUTABLE_OWNER_AST_SHA256)


class OracleCertifiedSourceLineEndingEquivalenceCases(TestCase):
    """One audited historic ZIP newline conversion is not a new economic RUN."""

    def test_signed_crlf_price_replay_reuses_on_lf_code_without_any_producer(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        from services.audit.oracle_score_artifacts import (
            certified_price_lineages_equivalent,
            _CERTIFIED_CRLF_PRICE_SOURCE_SHA256,
        )
        from core.file_integrity import canonical_json_sha256
        scenario = OracleSingleOwnerExecutionCases()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = scenario._fixture(root)
            definition, bench, seed, price_sha, param_sha, *_ = fixture
            cache = OracleReplayStore(root, definition.output_subdir)
            original, = cache.existing_completed_identities(seed=seed, target="R40")
            saved = cache.read(original)
            current_lineage = original["replay_policy"]["hindsight_target_contract"]
            previous_lineage = dict(current_lineage)
            previous_lineage["source_code_sha256"] = _CERTIFIED_CRLF_PRICE_SOURCE_SHA256
            previous_lineage["truth_source_sha256"] = canonical_json_sha256({
                key: previous_lineage[key] for key in (
                    "source_code_sha256", "score_algorithm_sha256",
                    "dataset_summary_sha256", "label_policy",
                )
            })
            self.assertTrue(certified_price_lineages_equivalent(previous_lineage, current_lineage))
            previous = {**original, "replay_policy": {
                **original["replay_policy"], "hindsight_target_contract": previous_lineage,
            }}
            cache._path(original).unlink()  # temp fixture only; retain signed old receipt
            cache.write(previous, saved, origin="signed_170351_crlf_revision")
            original_bytes = {path: path.read_bytes() for path in root.rglob("*.json")}
            with scenario._context(root, fixture):
                published = oracle.build_report(definition, project_root=root, quiet=True,
                                                preflight_approval=lambda _view: True)
            self.assertEqual(published["status"], "COMPLETED")
            self.assertEqual(published["oracles"][0]["target"], "R40")
            self.assertEqual(published["oracles"][0]["romd"], 18.)
            self.assertEqual({path: path.read_bytes() for path in root.rglob("*.json")}, original_bytes)

    def test_semantic_change_never_uses_a_line_ending_alias(self):
        from services.audit.oracle_score_artifacts import (
            certified_price_lineages_equivalent,
            _CERTIFIED_CRLF_PRICE_SOURCE_SHA256,
        )
        from core.file_integrity import canonical_json_sha256
        current = oracle._price_score_replay_lineage(_ACTIVE_ORACLE_DEFINITION)
        previous = {**current, "source_code_sha256": _CERTIFIED_CRLF_PRICE_SOURCE_SHA256}
        previous["truth_source_sha256"] = canonical_json_sha256({
            key: previous[key] for key in (
                "source_code_sha256", "score_algorithm_sha256",
                "dataset_summary_sha256", "label_policy",
            )
        })
        self.assertTrue(certified_price_lineages_equivalent(previous, current))
        for changed in ({"score_algorithm_sha256": "f"*64},
                        {"dataset_summary_sha256": "a"*64},
                        {"label_policy": {"different": True}},
                        {"source_code_sha256": "e"*64}):
            with self.subTest(changed=changed):
                new = {**previous, **changed}
                new["truth_source_sha256"] = canonical_json_sha256({
                    key: new[key] for key in (
                        "source_code_sha256", "score_algorithm_sha256",
                        "dataset_summary_sha256", "label_policy",
                    )
                })
                self.assertFalse(certified_price_lineages_equivalent(new, current))
        invalid_signature = {**previous, "truth_source_sha256": "0"*64}
        self.assertFalse(certified_price_lineages_equivalent(invalid_signature, current))

if __name__ == "__main__":
    main(verbosity=2)

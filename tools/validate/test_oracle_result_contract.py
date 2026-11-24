"""Behavior-level identities for the independent Target calculation stages."""
from copy import deepcopy
from unittest import TestCase

from core.file_integrity import canonical_json_sha256
from services.research.oracle_result_contract import (
    OracleResultContract, oracle_truth_result_contract, oracle_score_result_contract,
    oracle_replay_result_contract, oracle_observation_result_contract,
)


def execution_fixture(*, daily=True):
    return {
        "schema": 1, "purpose": "hindsight_oracle_completed_replay",
        "dataset": "isolated-fixture", "price_sha256": "a" * 64,
        "param_sha256": "b" * 64, "seed": 17, "target": "fixture",
        "start_date": "2021-01-01", "end_date": "2022-12-30",
        "research_cutoff": "2022-12-30", "dataset_inputs": {"calendar": "c" * 64},
        "replay_calculation_revision": 1, "replay_implementation_sha256": "d" * 64,
        "max_positions": 10, "rotation": "off", "evaluation_mode": "oos",
        "ranking_policy": "fixture-selector", "ranking_options": {"k": 10},
        "source_contract": {
            "recipe": "daily_trade_path_net_pnl" if daily else "mfe",
            "calculation_revision": 1, "feature_universe_revision": 1,
            "implementation_sha256": "e" * 64, "horizon_sessions": 40,
            "score_mapping": "canonical_daily_truth_percentile_v2",
            "percentile_method": "average_zero_based", "maturity": "complete_horizon",
        },
        "score_contract": {
            "score_start_date": "2021-01-01", "perfect_score_sha256": "f" * 64,
            "canonical_feature_eligible_keys_sha256": "1" * 64,
            "maturity_sha256": "2" * 64, "ordered_keys": ["A|2021-01-04", "B|2021-01-04"],
        },
        "replay_policy": {
            "filter_id": "fixture-filter", "fixed_risk": 0.01,
            "max_position_cap_pct": 0.1, "broker_accounting": {"fee": 0.001425},
            "execution_defaults": {"limit_policy": "next_day_only"},
            "daily_candidate_implementation_sha256": "3" * 64,
            "formal_positive_score_output": {
                "contract": "perfect-positive", "shared_admission_code_sha256": "4" * 64,
            },
            "period_stats_implementation": "portfolio_period_stats_v1",
        },
    }


def stage_keys(execution):
    return {name: canonical_json_sha256(builder(execution)) for name, builder in (
        ("truth", oracle_truth_result_contract), ("score", oracle_score_result_contract),
        ("replay", oracle_replay_result_contract), ("observation", oracle_observation_result_contract),
    )}


class OracleResultContractTests(TestCase):
    def test_source_provenance_changes_do_not_change_any_result_stage(self):
        before = execution_fixture()
        after = deepcopy(before)
        after["replay_implementation_sha256"] = "5" * 64
        after["source_contract"]["implementation_sha256"] = "6" * 64
        after["replay_policy"]["daily_candidate_implementation_sha256"] = "7" * 64
        after["replay_policy"]["formal_positive_score_output"]["shared_admission_code_sha256"] = "8" * 64
        self.assertEqual(stage_keys(before), stage_keys(after))
        self.assertEqual(OracleResultContract.from_execution(before), OracleResultContract.from_execution(after))

    def test_trade_path_inputs_invalidate_truth_and_all_consumers(self):
        before = execution_fixture()
        changes = (
            ("price_sha256", "9" * 64), ("param_sha256", "9" * 64),
            ("seed", 19), ("research_cutoff", "2022-11-30"),
            ("source_contract.calculation_revision", 2),
            ("source_contract.horizon_sessions", 60),
            ("source_contract.maturity", "different_mask"),
            ("replay_policy.broker_accounting.fee", 0.002),
        )
        for field, value in changes:
            with self.subTest(field=field):
                after = deepcopy(before)
                parent = after
                *parts, last = field.split(".")
                for part in parts:
                    parent = parent[part]
                parent[last] = value
                left, right = stage_keys(before), stage_keys(after)
                self.assertEqual({name for name in left if left[name] != right[name]}, set(left))

    def test_selector_and_capacity_do_not_revise_upstream_truth_or_scores(self):
        before = execution_fixture()
        for field, value in (("ranking_options", {"k": 3}), ("max_positions", 3),
                             ("ranking_policy", "different-selector"), ("replay_calculation_revision", 2)):
            with self.subTest(field=field):
                after = deepcopy(before)
                after[field] = value
                left, right = stage_keys(before), stage_keys(after)
                self.assertEqual({name for name in left if left[name] != right[name]}, {"replay", "observation"})

    def test_score_values_order_and_membership_invalidate_only_score_consumers(self):
        before = execution_fixture()
        for field, value in (("perfect_score_sha256", "9" * 64),
                             ("canonical_feature_eligible_keys_sha256", "9" * 64),
                             ("maturity_sha256", "9" * 64),
                             ("ordered_keys", ["B|2021-01-04", "A|2021-01-04"])):
            with self.subTest(field=field):
                after = deepcopy(before)
                after["score_contract"][field] = value
                left, right = stage_keys(before), stage_keys(after)
                self.assertEqual({name for name in left if left[name] != right[name]}, {"score", "replay", "observation"})

    def test_price_truth_excludes_teacher_and_portfolio_economics(self):
        before = execution_fixture(daily=False)
        after = deepcopy(before)
        after["param_sha256"] = "9" * 64
        after["max_positions"] = 1
        after["replay_policy"]["broker_accounting"]["fee"] = 0.01
        left, right = stage_keys(before), stage_keys(after)
        self.assertEqual(left["truth"], right["truth"])
        self.assertEqual(left["score"], right["score"])
        self.assertNotEqual(left["replay"], right["replay"])

    def test_observation_revision_does_not_revise_trading_inputs(self):
        before = execution_fixture()
        after = deepcopy(before)
        after["replay_policy"]["period_stats_implementation"] = "portfolio_period_stats_v2"
        left, right = stage_keys(before), stage_keys(after)
        self.assertEqual({name for name in left if left[name] != right[name]}, {"observation"})

    def test_unknown_effective_fields_are_never_silently_discarded(self):
        before = execution_fixture()
        for scope in (None, "replay_policy", "source_contract", "score_contract"):
            with self.subTest(scope=scope):
                after = deepcopy(before)
                (after if scope is None else after[scope])["new_effective_rule"] = {"value": 1}
                self.assertNotEqual(OracleResultContract.from_execution(before).sha256,
                                    OracleResultContract.from_execution(after).sha256)

    def test_result_contract_is_immutable_and_serialization_order_independent(self):
        original = execution_fixture()
        contract = OracleResultContract.from_execution(original)
        expected = contract.sha256
        reordered = dict(reversed(list(original.items())))
        self.assertEqual(expected, OracleResultContract.from_execution(reordered).sha256)
        original["ranking_options"]["k"] = 100
        exported = contract.as_payload()
        exported["ranking_options"]["k"] = 1000
        self.assertEqual(contract.sha256, expected)

    def test_wrong_purpose_and_nonfinite_inputs_are_rejected(self):
        context = execution_fixture()
        context["purpose"] = "another_artifact"
        with self.assertRaises(ValueError):
            OracleResultContract.from_execution(context)
        context = execution_fixture()
        context["ranking_options"]["k"] = float("nan")
        with self.assertRaises(ValueError):
            OracleResultContract.from_execution(context)

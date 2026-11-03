"""Isolated, small-sample tests of new TrueDaily stock-day Min capability.

Run directly: python -m unittest tools.validate.synthetic_true_daily_mintrade_oracle_cases -q
No formal suite, optimizer, replay or model training is started here.
"""
from __future__ import annotations

from pathlib import Path
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
        self.assertEqual(len(definition.source["oracle_comparisons"]), 20)
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
        self.assertEqual(original_fingerprint("net_pnl"),
                         "6978ded3a808149755dd3cda328fe96da12ab9a37c17b11d6103cc96dc45fc07")
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


if __name__ == "__main__":
    main()

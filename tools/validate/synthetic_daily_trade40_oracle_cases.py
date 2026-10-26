"""Formal Min per-seed Oracle source and audit REUSE invariants (synthetic-only)."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from core.audit_policy import get_reusable_audit_definitions
from core.research_report_contract import require_approved_report_contract
from filters.breakout_quality.daily_trade40_target import (
    ExecutableMinPeriod, build_formal_min_execution_schedule,
)
from services.audit import target_oracle_romd as oracle
from services.audit.daily_trade40_oracle import (
    DailyTrade40OracleBlockedError, build_daily_universal_trade40_oracle_scores,
    TARGET_ID, LABEL_ID,
)
from tools.validate.synthetic_daily_trade40_cases import fixture

EXACT = "DailyUniversal_MinTrade40_NetR"
DIRECT = "DailyUniversal_MinTrade40_NetR_Direct"
PROFILE = "daily_universal_full_horizon_mae_gated_mfe_t40_full_list_ndcg_pairwise"


def definition():
    return next(d for d in get_reusable_audit_definitions("breakout_quality")
                if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)


def source_fixture():
    trading = pd.bdate_range("2020-10-01", periods=160).difference(
        pd.DatetimeIndex(["2021-01-01"]))
    frames = {}
    rows = []
    for ticker, price_delta in (("2330", .22), ("2317", -.15)):
        base_days, frame, params, period = fixture(count=len(trading))
        frame.index = trading
        close = 100. + price_delta * np.arange(len(trading))
        for c in ("Open", "High", "Low", "Close"):
            frame[c] = close + ({"Open":0.,"High":2.,"Low":-2.,"Close":0.}[c])
        frames[ticker] = frame
        for day in ("2020-12-31", "2021-01-04", "2021-01-05"):
            rows.append({"ticker":ticker,"date":day})
    group = pd.DataFrame(rows)
    bundle = SimpleNamespace(
        group_table=group,
        summary={"dataset":"full", "filter_id":"breakout_quality_v1",
                 "sample_scope":"daily_eligible_stock_days",
                 "score_eligible_sample_count":len(group)},
    )
    spec = [
        {"effective_date":"2021-01-01", "effective_end_date":None,
         "params_obj":params, "params_signature":period.params_signature},
    ]
    schedule,_ = build_formal_min_execution_schedule(spec,seed=693545351,param_sha256="a"*64)
    return trading, frames, bundle, schedule


def run_source(*, schedule=None, seed=693545351, dataset="full", group=None,
               evaluation_start="2021-01-01", cutoff="2021-05-15"):
    trading, frames, bundle, original = source_fixture()
    if group is not None: bundle.group_table=group;bundle.summary["score_eligible_sample_count"]=len(group)
    with patch("services.audit.daily_trade40_oracle.load_profile_continuous_ranker_data",return_value=bundle), \
         patch("services.audit.daily_trade40_oracle.discover_dataset_csv_inputs",return_value=(
             [("0050",Path("0050.csv")),("2330",Path("2330.csv")),("2317",Path("2317.csv"))],[])), \
         patch("services.audit.daily_trade40_oracle.collect_canonical_exchange_sessions",return_value=trading), \
         patch("services.audit.daily_trade40_oracle.load_dataset_frame",side_effect=lambda p,t,**kw: frames[t]):
        return build_daily_universal_trade40_oracle_scores(
            project_root=Path("/isolated"),dataset=dataset,
            filter_id="breakout_quality_v1",profile_id=PROFILE,
            cutoff=cutoff,evaluation_start=evaluation_start,score_start="2021-01-01",seed=seed,
            param_sha256="a"*64,canonical_schedule=schedule or original,
        )


def synthetic_perfect_score_contracts(_definition, _settings, plan):
    """Authorized synthetic score maps only, never production PIT evidence."""
    return {name: {
        "reference_arm_id": "synthetic_authorized",
        "score_start_date": "2021-01-01",
        "target_id": "synthetic_target",
    } for name, _ in plan["targets"]}


def synthetic_score_eligibility(_definition, *, contracts, plan,
                           price_scores, daily_scores):
    return {name: {**contract, "canonical_feature_eligible_keys_sha256": "a" * 64}
            for name, contract in contracts.items()}


def synthetic_score_parity_setup(testcase):
    # Replay-integrity fixtures verify cached-score identity rather than
    # reconstructing historical model PIT files. Original hypothetical score
    # method and absence of a FIT dependency are tested without mocks in
    # synthetic_target_oracle_cases.py.
    for name, action in (
        ("_resolve_oracle_target_score_contracts", synthetic_perfect_score_contracts),
        ("_certify_oracle_score_eligibility", synthetic_score_eligibility),
    ):
        guard = patch.object(oracle, name, side_effect=action)
        guard.start()
        testcase.addCleanup(guard.stop)


class FormalMinDailyOracleCases(TestCase):
    def setUp(self):
        synthetic_score_parity_setup(self)

    def test_existing_comparisons_are_preserved_and_report_contract_unchanged(self):
        d=definition()
        p=oracle._oracle_comparison_plan(d)
        self.assertEqual(len(p["targets"]),11)
        self.assertEqual(p["daily_recipes"],{EXACT:("daily_trade_path_net_r","daily_trade40_profile")})
        self.assertEqual(p["daily_sources"],{EXACT:EXACT,DIRECT:EXACT})
        self.assertEqual(p["selectors"][EXACT],"exact")
        self.assertEqual(p["selectors"][DIRECT],"direct")
        self.assertEqual(d.source["daily_trade40_profile"],PROFILE)
        self.assertEqual(require_approved_report_contract("audit.target_oracle_romd").version,9)

    def test_daily_oracle_real_csv_calendar_and_price_path(self):
        """Exercise the production CSV -> sanitized exchange sessions -> Oracle path.

        The other synthetic source fixture mocks the calendar collector and
        therefore cannot detect a missing runtime import in that function.
        """
        trading, frames, bundle, schedule = source_fixture()
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            csv_inputs = []
            for ticker, frame in (("0050", frames["2330"]),
                                  ("2330", frames["2330"]),
                                  ("2317", frames["2317"])):
                path = temp / f"{ticker}.csv"
                frame.to_csv(path, index_label="Date")
                csv_inputs.append((ticker, path))
            with patch(
                "services.audit.daily_trade40_oracle.load_profile_continuous_ranker_data",
                return_value=bundle,
            ), patch(
                "services.audit.daily_trade40_oracle.discover_dataset_csv_inputs",
                return_value=(csv_inputs, []),
            ):
                # Do NOT patch collect_canonical_exchange_sessions or
                # load_dataset_frame: both must execute their real code.
                actual, evidence = build_daily_universal_trade40_oracle_scores(
                    project_root=temp, dataset="full", filter_id="breakout_quality_v1",
                    profile_id=PROFILE, cutoff="2021-05-15", evaluation_start="2021-01-01",
                    score_start="2021-01-01", seed=693545351, param_sha256="a" * 64,
                    canonical_schedule=schedule,
                )
        expected, reference = run_source()
        self.assertEqual(actual, expected)
        for key in ("first_replay_trade_date", "calendar_sha256",
                    "raw_label_sha256", "rank_sha256", "settled_net_r_rows"):
            self.assertEqual(evidence[key], reference[key])
        self.assertEqual(evidence["first_replay_trade_date"], "2021-01-04")

    def test_first_legal_information_day_uses_formal_pit_score_start(self):
        scores,evidence=run_source()
        self.assertNotIn(("2330","2020-12-31"),scores)
        self.assertNotIn(("2317","2020-12-31"),scores)
        self.assertEqual(evidence["first_replay_trade_date"],"2021-01-04")
        self.assertEqual(evidence["first_date"],"2021-01-04")
        self.assertEqual(evidence["score_start_date"],"2021-01-01")
        self.assertEqual(evidence["label_id"],LABEL_ID)
        self.assertEqual(evidence["target_id"],TARGET_ID)
        self.assertEqual(evidence["teacher"]["seed"],693545351)
        self.assertEqual(evidence["teacher"]["param_identity_sha256"],"a"*64)
        self.assertEqual(len(scores),4)
        self.assertEqual(evidence["mature_eligible_rows"],4)
        self.assertEqual(evidence["settled_net_r_rows"]+evidence["censored_rank_last_rows"],4)

    def test_exact_direct_share_scores_but_not_selector(self):
        scores,evidence=run_source()
        d=definition();plan=oracle._oracle_comparison_plan(d)
        base={"fixed_risk":.025,"ranking_policy":"canonical"}
        evidence_sha=oracle._canonical_score_map_sha256(scores)
        contract={"reference_arm_id":"synthetic_authorized", "target_id":TARGET_ID}
        e1=oracle._arm_replay_policy(d,plan=plan,target=EXACT,replay_policy=base,daily_evidence=evidence,
                                     score_sha256=evidence_sha,score_contract=contract)
        e2=oracle._arm_replay_policy(d,plan=plan,target=DIRECT,replay_policy=base,daily_evidence=evidence,
                                     score_sha256=evidence_sha,score_contract=contract)
        self.assertEqual(e1["hindsight_target_contract"],e2["hindsight_target_contract"])
        self.assertEqual(
            e1["hindsight_target_contract"]["missing_key_policy"],
            "unavailable_outside_mature_feature_universe",
        )
        self.assertNotIn("direct_arm",e1)
        self.assertIn("direct_arm",e2)
        self.assertEqual(e1["hindsight_target_contract"]["label_id"],LABEL_ID)
        self.assertNotIn("MR-13CL",str(e1))
        self.assertGreater(scores[("2330","2021-01-04")],scores[("2317","2021-01-04")])

    def test_benchmark_is_not_a_daily_model_sample_and_stays_unscored_in_oracle(self):
        """Reproduce 0050/2021-01-13: no fake R and no new candidate engine.

        The daily model uses 0050 only as a feature benchmark. The Min Portfolio
        can still present 0050 as a candidate. Its ranked availability must
        behave exactly as a canonical PIT model with no ticker/date score.
        """
        from filters.breakout_quality.runtime import (
            audit_oracle_truth_score_context, resolve_breakout_quality_candidate_rank,
        )
        from core.portfolio_entry_selection_max_dl import _max_dl_score_order
        scores, _evidence = run_source()
        self.assertIn(("2330", "2021-01-04"), scores)
        self.assertNotIn(("0050", "2021-01-04"), scores)
        settings, _base, exact = oracle._contract(definition())
        direct = oracle._direct_arm(definition(), settings, _base, exact)
        for arm in (exact, direct):
            with self.subTest(selector=arm.dl_runtime_mode):
                with audit_oracle_truth_score_context(
                    scores, ranking_policy=arm.dl_runtime_mode,
                    ranking_options=arm.dl_runtime_options,
                    missing_key_policy="unavailable_outside_mature_feature_universe",
                ):
                    scored = resolve_breakout_quality_candidate_rank(
                        ticker="2330", signal_date="2020-12-31",
                        information_date="2021-01-04", high_len=260,
                    )
                    unscored = resolve_breakout_quality_candidate_rank(
                        ticker="0050", signal_date="2020-12-31",
                        information_date="2021-01-04", high_len=260,
                    )
                    self.assertTrue(scored["available"])
                    self.assertIsInstance(scored["score"], float)
                    self.assertEqual(unscored["score"], None)
                    self.assertFalse(unscored["available"])
                    self.assertEqual(unscored["unavailable_reason"], "missing_ticker_date_score")
                    # The canonical DL selector keeps the same Min candidate,
                    # but scores it after available candidates: no 0R imputation.
                    rows = [
                        {"ticker": "0050", "breakout_quality_rank": unscored,
                         "breakout_quality_score": None},
                        {"ticker": "2330", "breakout_quality_rank": scored,
                         "breakout_quality_score": scored["score"]},
                    ]
                    sorted_rows = _max_dl_score_order(rows, base_rank={id(v): k for k, v in enumerate(rows)})
                    self.assertEqual([x["ticker"] for x in sorted_rows], ["2330", "0050"])
        # No new fallback is permitted for event or other strict Oracle sources.
        with audit_oracle_truth_score_context(
            scores, ranking_policy=exact.dl_runtime_mode,
            ranking_options=exact.dl_runtime_options,
        ):
            with self.assertRaisesRegex(ValueError, "未覆蓋"):
                resolve_breakout_quality_candidate_rank(
                    ticker="0050", signal_date="2020-12-31",
                    information_date="2021-01-04", high_len=260,
                )

    def test_daily_unscored_scope_rejects_nonfinite_truth_and_event_scope(self):
        from filters.breakout_quality.runtime import (
            audit_oracle_truth_score_context, resolve_breakout_quality_candidate_rank,
        )
        settings, _base, exact = oracle._contract(definition())
        with self.assertRaisesRegex(ValueError, "Only the mature daily"):
            with audit_oracle_truth_score_context(
                {("2330", "2025-01-02"): 0.8}, ranking_policy=exact.dl_runtime_mode,
                ranking_options=exact.dl_runtime_options,
                key_mode="signal_origin_date",
                missing_key_policy="unavailable_outside_mature_feature_universe",
            ):
                pass
        with audit_oracle_truth_score_context(
            {("2330", "2025-01-02"): float("nan")}, ranking_policy=exact.dl_runtime_mode,
            ranking_options=exact.dl_runtime_options,
            missing_key_policy="unavailable_outside_mature_feature_universe",
        ):
            with self.assertRaisesRegex(ValueError, "非有限真值"):
                resolve_breakout_quality_candidate_rank(
                    ticker="2330", signal_date="2025-01-02",
                    information_date="2025-01-02", high_len=260,
                )
        with audit_oracle_truth_score_context(
            {("2330", "2025-01-02"): 0.8}, ranking_policy=exact.dl_runtime_mode,
            ranking_options=exact.dl_runtime_options,
        ):
            with self.assertRaisesRegex(ValueError, "未覆蓋"):
                resolve_breakout_quality_candidate_rank(
                    ticker="0050", signal_date="2025-01-02",
                    information_date="2025-01-02", high_len=260,
                )

    def test_wrong_dataset_or_missing_source_is_blocked_not_zero(self):
        with self.assertRaisesRegex(DailyTrade40OracleBlockedError,"identity mismatch"):
            run_source(dataset="reduced")
        trading,frames,bundle,schedule=source_fixture()
        with self.assertRaisesRegex(DailyTrade40OracleBlockedError,"not the formal Min Portfolio"):
            run_source(schedule=(ExecutableMinPeriod(schedule[0].start,None,schedule[0].params,
                                                    "old_historical_teacher",schedule[0].params_signature),))

    def test_missing_effective_min_parameter_is_blocked_not_ranked_as_worst(self):
        """An absent parameter period is missing evidence, not a bad trade.

        Previously one missing-parameter row was ranked last when a different
        stock on the same date had a settled path, silently producing RoMD.
        """
        from filters.breakout_quality.daily_trade40_target import CounterfactualTrade40Batch
        from services.audit import daily_trade40_oracle as provider

        original = provider.compute_counterfactual_trade40_batch

        def missing_one_parameter(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs["ticker"] != "2317":
                return result
            values = np.asarray(result.target_raw_r, dtype=np.float32).copy()
            valid = np.asarray(result.valid_mask, dtype=bool).copy()
            reasons = list(result.reasons)
            self.assertTrue(len(reasons))
            values[0] = np.nan
            valid[0] = False
            reasons[0] = "no_historical_effective_Min_param"
            return CounterfactualTrade40Batch(
                values, valid, result.maturity_dates, tuple(reasons),
            )

        with patch.object(
            provider, "compute_counterfactual_trade40_batch",
            side_effect=missing_one_parameter,
        ):
            with self.assertRaisesRegex(
                DailyTrade40OracleBlockedError,
                "Formal Min rolling parameter schedule missing.*2317",
            ):
                run_source()

    def test_real_min_schedule_gap_blocks_before_any_daily_rank(self):
        """Exercise the canonical trade-path primitive, not a mocked reason."""
        from dataclasses import replace
        _calendar, _frames, _bundle, schedule = source_fixture()
        future_only = (replace(schedule[0], start=pd.Timestamp("2021-03-01")),)
        with self.assertRaisesRegex(
            DailyTrade40OracleBlockedError, "rolling parameter schedule missing",
        ):
            run_source(schedule=future_only)

    def test_normal_unfilled_counterfactual_remains_censored(self):
        """A genuine unfilled order is still a legitimate ranked-last label."""
        from filters.breakout_quality.daily_trade40_target import CounterfactualTrade40Batch
        from services.audit import daily_trade40_oracle as provider

        original = provider.compute_counterfactual_trade40_batch

        def unfilled_one_order(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs["ticker"] != "2317":
                return result
            values = np.asarray(result.target_raw_r, dtype=np.float32).copy()
            valid = np.asarray(result.valid_mask, dtype=bool).copy()
            reasons = list(result.reasons)
            values[0] = np.nan
            valid[0] = False
            reasons[0] = "unfilled_limit_order"
            return CounterfactualTrade40Batch(
                values, valid, result.maturity_dates, tuple(reasons),
            )

        with patch.object(
            provider, "compute_counterfactual_trade40_batch", side_effect=unfilled_one_order,
        ):
            scores, evidence = run_source()
        self.assertEqual(len(scores), 4)
        self.assertGreater(evidence["censored_rank_last_rows"], 0)

    def test_date_with_all_unfilled_orders_is_a_valid_equal_last_tie(self):
        """No filled trade on a date is not missing parameters or a fabricated R.

        The approved censored-rank-last mapping has an unambiguous tied rank
        even when every stock on that day is censored. Do not block a five-year
        Oracle solely because one information date has no filled Min path.
        """
        from filters.breakout_quality.daily_trade40_target import CounterfactualTrade40Batch
        from services.audit import daily_trade40_oracle as provider
        original = provider.compute_counterfactual_trade40_batch

        def unfilled_last_day(*args, **kwargs):
            result = original(*args, **kwargs)
            values = np.asarray(result.target_raw_r, dtype=np.float32).copy()
            valid = np.asarray(result.valid_mask, dtype=bool).copy()
            reasons = list(result.reasons)
            # The first day still has a real executed outcome; the second
            # day's entire cohort has zero fills, but valid Min parameters.
            if kwargs["ticker"] == "2330":
                values[0] = 1.5
                valid[0] = True
                reasons[0] = "filled_executed_net_R"
            else:
                values[0] = np.nan
                valid[0] = False
                reasons[0] = "unfilled_limit_order"
            values[-1] = np.nan
            valid[-1] = False
            reasons[-1] = "unfilled_limit_order"
            return CounterfactualTrade40Batch(
                values, valid, result.maturity_dates, tuple(reasons),
            )

        with patch.object(provider, "compute_counterfactual_trade40_batch",
                          side_effect=unfilled_last_day):
            scores, evidence = run_source()
        self.assertEqual(scores[("2330", "2021-01-05")], 0.5)
        self.assertEqual(scores[("2317", "2021-01-05")], 0.5)
        self.assertEqual(evidence["settled_net_r_rows"], 1)
        self.assertEqual(evidence["censored_rank_last_rows"], 3)

    def test_source_is_seed_specific_even_with_same_stock_days(self):
        _,_,_,s1=source_fixture()
        p=s1[0].params
        from core.portfolio_param_runtime import build_portfolio_params_signature
        from dataclasses import replace
        altered=replace(p,atr_times_init=p.atr_times_init*0.35)
        sched2,_=build_formal_min_execution_schedule([
            {"effective_date":"2021-01-01","params_obj":altered,
             "params_signature":build_portfolio_params_signature(altered)}],
             seed=2014432738,param_sha256="b"*64)
        _,ev1=run_source(schedule=s1,seed=693545351)
        _,ev2=run_source(schedule=sched2,seed=2014432738)
        self.assertNotEqual(ev1["teacher"]["seed"],ev2["teacher"]["seed"])
        self.assertNotEqual(ev1["teacher"]["periods"],ev2["teacher"]["periods"])
        self.assertNotEqual(ev1["teacher"]["periods"][0]["params_signature"],
                            ev2["teacher"]["periods"][0]["params_signature"])

    def test_no_future_param_backfill_and_no_legacy_teacher_import(self):
        import filters.breakout_quality.daily_trade40_target as primitive
        self.assertFalse(hasattr(primitive,"load_historical_min_execution_schedule"))
        self.assertFalse(hasattr(primitive,"resolve_historical_min_teacher_paths"))
        self.assertFalse(hasattr(primitive,"counterfactual_trade40_contract"))
        self.assertFalse(hasattr(oracle,"_missing_daily_teacher_source"))

    def test_blocked_upstream_does_not_compute_daily_truth_or_start_optimizer(self):
        d = definition()
        with patch.object(oracle, "collect_status", return_value={
                "status": "BLOCKED", "reason": "formal Min seed parameters unavailable",
             }), \
             patch.object(oracle, "_robustness_artifacts") as sources, \
             patch.object(oracle, "_build_oracle_score_panels") as truth, \
             patch.object(oracle, "_run_scenario") as replay, \
             patch("builtins.input", side_effect=AssertionError("domain-level confirmation")) as prompt, \
             patch("services.research.strategy_compare_preparation.prepare_strategy_parameter_artifacts",
                   side_effect=AssertionError("unexpected Optimizer")) as optimizer:
            with self.assertRaisesRegex(oracle.OracleBlockedError, "formal Min seed"):
                oracle.build_report(d, project_root=Path("/isolated"), quiet=True)
            sources.assert_not_called()
            truth.assert_not_called()
            replay.assert_not_called()
            optimizer.assert_not_called()
            prompt.assert_not_called()

    def test_cache_identity_is_per_target_and_same_period_no_global_shift(self):
        from services.audit.oracle_replay_artifacts import replay_identity
        kw=dict(dataset="full",dataset_sha256="a"*64,
                definition={"C96":"base",EXACT:"formal Min 40d"},
                start="2021-01-01",end="2026-03-02",benchmark="synthetic",
                param_sha256="b"*64,seed=693545351,cohort={"same":"source"},
                ranking_policy="rank",ranking_options={},evaluation_mode="rolling",
                max_positions=10,replay_policy={"x":1})
        c96=replay_identity(target="C96",**kw)
        enriched=replay_identity(target="C96",**{**kw,"definition":{**kw["definition"],
            DIRECT:"new unrelated target"}})
        self.assertEqual(c96,enriched)
        treatment=replay_identity(target=EXACT,**kw)
        self.assertNotEqual(treatment,c96)


class FormalMinReplayIntegrationCases(TestCase):
    def setUp(self):
        synthetic_score_parity_setup(self)

    def test_four_seed_existing_48_reuse_new_8_runs_same_2021_window(self):
        # An exact replay-cache regression: adding a stock-day target must NOT
        # change the original time window or force old 48 treatments to RUN.
        from dataclasses import replace
        from filters.breakout_quality.runtime import resolve_breakout_quality_candidate_rank
        from core.breakout_quality_ranking_context import get_breakout_quality_ranking_source_context
        active=definition()
        old=replace(active,source={**active.source,"oracle_comparisons":tuple(
            item for item in active.source["oracle_comparisons"]
            if item["target"] not in (EXACT,DIRECT))})
        plan=oracle._oracle_comparison_plan(old)
        price_scores={name:{("2330","2025-01-03"):0.8} for name in plan["recipes"]}
        for name,source in plan["aliases"].items():
            if source in price_scores:
                price_scores[name]=price_scores[source]
        cohort={"cohort_rows":100,"first_common_day":"2021-01-01",
                "last_common_day":"2025-12-24","research_cutoff":"2026-03-02",
                "providers":{"40d":active.source["pure_mfe40_provider"]}}
        seeds=(693545351,2014432738,104504529,762252050)
        benchmark=SimpleNamespace(benchmark_id="synthetic_same_source")
        observed=[]
        def replay(*,name,**kwargs):
            if name.startswith("Oracle DailyUniversal_"):
                score=resolve_breakout_quality_candidate_rank(
                    ticker="2330",signal_date="2025-01-02",
                    information_date="2025-01-03",high_len=260,
                    filter_id="breakout_quality_v1")
                context=get_breakout_quality_ranking_source_context()
                observed.append((name,score["score"],score["score_date"],context.ranking_policy))
            return {"name":name}
        def metrics(result,*,target):
            seed=int(result["name"].split("seed=")[1].split()[0])
            romd=(7. if target=="C96" else
                  9. if target==EXACT else
                  6. if target==DIRECT else 8.)
            return {"target":target,"seed":seed,"total_return_pct":100.,
                    "annualized_return_pct":12.,"mdd_pct":10.,"romd":romd,
                    "ev_a_r":.5,"trades":25,"exposure_pct":60.,
                    "yearly":[{"year":year,"strategy_return_pct":5.,"is_full_year":year!=2025}
                              for year in range(2021,2026)]}
        formal=source_fixture()[3]
        def scores(*,seed,param_sha256,**kwargs):
            score_map={("2330","2025-01-03"):0.625}
            evidence={"label_id":LABEL_ID,"target_id":TARGET_ID,"profile_id":PROFILE,
                "rank_policy":"average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
                "rank_sha256":"c"*64,"raw_label_sha256":"d"*64,
                "calendar_sha256":"e"*64,
                "teacher":{"origin":"formal_min_roos_rolling_per_seed",
                           "seed":int(seed),"param_identity_sha256":str(param_sha256),
                           "periods":[{"start":"2021-01-01","params_signature":"sig"}]},
                "mature_eligible_rows":25,"settled_net_r_rows":20,
                "censored_rank_last_rows":5,"research_cutoff":"2026-03-02",
                "first_date":"2021-01-04","first_replay_trade_date":"2021-01-04", "score_start_date":"2021-01-01"}
            return score_map,evidence
        with TemporaryDirectory() as tmp, \
             patch.object(oracle,"collect_status",return_value={"status":"READY"}), \
             patch.object(oracle,"_robustness_artifacts",return_value=(benchmark,[(seed,Path(f"seed_{seed}.json")) for seed in seeds])), \
             patch.object(oracle,"_build_oracle_score_panels",return_value=(price_scores,cohort)), \
             patch.object(oracle,"load_strategy_param_evaluation_view",return_value=({"kind":"rolling_active_param_ensemble","payload":{}},"a"*64)), \
             patch.object(oracle,"_validate_requested_param_policy"), \
             patch.object(oracle,"get_active_param_ensemble_date_range",return_value=("2020-01-01","2026-03-02")), \
             patch.object(oracle,"_build_controlled_param_source_pair",return_value=("rolling_active_param_ensemble","base","oracle",None,None,None)), \
             patch.object(oracle,"price_inventory_sha256",return_value="b"*64), \
             patch("filters.breakout_quality.daily_trade40_target.resolve_formal_min_param_records",return_value=({
                 "effective_date":"2021-01-01","params_obj":formal[0].params,
                 "params_signature":formal[0].params_signature,
             },)), \
             patch("services.audit.daily_trade40_oracle.build_daily_universal_trade40_oracle_scores",side_effect=scores) as source, \
             patch.object(oracle,"_run_scenario",side_effect=replay) as run, \
             patch.object(oracle,"_metrics",side_effect=metrics):
            root=Path(tmp)
            previous=oracle.run_formal_audit(old,project_root=root,quiet=True)
            self.assertEqual(len(previous["oracles"]),9)
            self.assertEqual(run.call_count,40)
            updated=oracle.run_formal_audit(active,project_root=root,quiet=True)
            self.assertEqual(run.call_count,48,"40 certified reuses + only 8 new replay runs")
            self.assertEqual({call.kwargs["start_date"] for call in run.call_args_list},{"2021-01-01"})
            self.assertEqual(len(updated["oracles"]),11)
            self.assertEqual(len(observed),8)
            self.assertTrue(all(score==.625 and dt=="2025-01-03" for _,score,dt,_ in observed))
            exact_mode={mode for name,_,_,mode in observed if "_Direct" not in name}
            direct_mode={mode for name,_,_,mode in observed if "_Direct" in name}
            self.assertNotEqual(exact_mode,direct_mode)
            self.assertEqual(source.call_count,4)
            rows={row["target"]:row for row in updated["oracles"]}
            self.assertEqual(rows[EXACT]["delta_romd_vs_c96"],2.0)
            self.assertEqual(rows[DIRECT]["delta_romd_vs_c96"],-1.0)
            oracle.run_formal_audit(active,project_root=root,quiet=True)
            self.assertEqual(run.call_count,48,"second run must REUSE all 48")
            self.assertEqual(source.call_count,8)
            # A changed Target hash cannot be silently accepted as the old truth.
            changed_hash="0"*64
            def conflicting(*,seed,param_sha256,**kwargs):
                x,y=scores(seed=seed,param_sha256=param_sha256,**kwargs)
                y["raw_label_sha256"]=changed_hash
                return x,y
            source.side_effect=conflicting
            with self.assertRaisesRegex(oracle.OracleBlockedError,"(INCOMPATIBLE|DIFFERENT|identity)"):
                oracle.run_formal_audit(active,project_root=root,quiet=True)
            self.assertEqual(run.call_count,48)


if __name__=="__main__":
    main()

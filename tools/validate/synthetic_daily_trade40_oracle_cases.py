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
            cutoff=cutoff,evaluation_start=evaluation_start,seed=seed,
            param_sha256="a"*64,canonical_schedule=schedule or original,
        )


class FormalMinDailyOracleCases(TestCase):
    def test_existing_comparisons_are_preserved_and_report_contract_unchanged(self):
        d=definition()
        p=oracle._oracle_comparison_plan(d)
        self.assertEqual(len(p["targets"]),13)
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
                    seed=693545351, param_sha256="a" * 64,
                    canonical_schedule=schedule,
                )
        expected, reference = run_source()
        self.assertEqual(actual, expected)
        for key in ("first_replay_trade_date", "calendar_sha256",
                    "raw_label_sha256", "rank_sha256", "settled_net_r_rows"):
            self.assertEqual(evidence[key], reference[key])
        self.assertEqual(evidence["first_replay_trade_date"], "2021-01-04")

    def test_first_legal_information_day_is_previous_year_and_trade_is_jan4(self):
        scores,evidence=run_source()
        self.assertIn(("2330","2020-12-31"),scores)
        self.assertIn(("2317","2020-12-31"),scores)
        self.assertEqual(evidence["first_replay_trade_date"],"2021-01-04")
        self.assertEqual(evidence["first_date"],"2020-12-31")
        self.assertEqual(evidence["label_id"],LABEL_ID)
        self.assertEqual(evidence["target_id"],TARGET_ID)
        self.assertEqual(evidence["teacher"]["seed"],693545351)
        self.assertEqual(evidence["teacher"]["param_identity_sha256"],"a"*64)
        self.assertEqual(len(scores),6)
        self.assertEqual(evidence["mature_eligible_rows"],6)
        self.assertEqual(evidence["settled_net_r_rows"]+evidence["censored_rank_last_rows"],6)

    def test_exact_direct_share_scores_but_not_selector(self):
        scores,evidence=run_source()
        d=definition();plan=oracle._oracle_comparison_plan(d)
        base={"fixed_risk":.025,"ranking_policy":"canonical"}
        e1=oracle._arm_replay_policy(d,plan=plan,target=EXACT,replay_policy=base,daily_evidence=evidence)
        e2=oracle._arm_replay_policy(d,plan=plan,target=DIRECT,replay_policy=base,daily_evidence=evidence)
        self.assertEqual(e1["hindsight_target_contract"],e2["hindsight_target_contract"])
        self.assertNotIn("direct_arm",e1)
        self.assertIn("direct_arm",e2)
        self.assertEqual(e1["hindsight_target_contract"]["label_id"],LABEL_ID)
        self.assertNotIn("MR-13CL",str(e1))
        self.assertGreater(scores[("2330","2020-12-31")],scores[("2317","2020-12-31")])

    def test_wrong_dataset_or_missing_source_is_blocked_not_zero(self):
        with self.assertRaisesRegex(DailyTrade40OracleBlockedError,"identity mismatch"):
            run_source(dataset="reduced")
        trading,frames,bundle,schedule=source_fixture()
        with self.assertRaisesRegex(DailyTrade40OracleBlockedError,"not the formal Min Portfolio"):
            run_source(schedule=(ExecutableMinPeriod(schedule[0].start,None,schedule[0].params,
                                                    "old_historical_teacher",schedule[0].params_signature),))

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
        event_scores={seed:{("2330","2025-01-02"):0.6} for seed in seeds}
        event_evidence={seed:{"label_id":"LABEL-mintrade40_realized_net_r_v1"} for seed in seeds}
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
                    "yearly":[{"year":2025,"strategy_return_pct":5.,"is_full_year":True}]}
        formal=source_fixture()[3]
        def scores(*,seed,param_sha256,**kwargs):
            score_map={("2330","2020-12-31"):0.5,("2330","2025-01-03"):0.625}
            evidence={"label_id":LABEL_ID,"target_id":TARGET_ID,"profile_id":PROFILE,
                "rank_policy":"average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
                "rank_sha256":"c"*64,"raw_label_sha256":"d"*64,
                "calendar_sha256":"e"*64,
                "teacher":{"origin":"formal_min_roos_rolling_per_seed",
                           "seed":int(seed),"param_identity_sha256":str(param_sha256),
                           "periods":[{"start":"2021-01-01","params_signature":"sig"}]},
                "mature_eligible_rows":25,"settled_net_r_rows":20,
                "censored_rank_last_rows":5,"research_cutoff":"2026-03-02",
                "first_date":"2020-12-31","first_replay_trade_date":"2021-01-04"}
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
             patch("services.audit.min_trade40_target.build_seed_oracle_scores",return_value=(event_scores,event_evidence)), \
             patch("services.audit.min_trade40_target._schedule",return_value=({
                 "effective_date":"2021-01-01","params_obj":formal[0].params,
                 "params_signature":formal[0].params_signature,
             },)), \
             patch("services.audit.daily_trade40_oracle.build_daily_universal_trade40_oracle_scores",side_effect=scores) as source, \
             patch.object(oracle,"_run_scenario",side_effect=replay) as run, \
             patch.object(oracle,"_metrics",side_effect=metrics):
            root=Path(tmp)
            previous=oracle.run_formal_audit(old,project_root=root,quiet=True)
            self.assertEqual(len(previous["oracles"]),11)
            self.assertEqual(run.call_count,48)
            updated=oracle.run_formal_audit(active,project_root=root,quiet=True)
            self.assertEqual(run.call_count,56,"48 certified reuses + only 8 new replay runs")
            self.assertEqual({call.kwargs["start_date"] for call in run.call_args_list},{"2021-01-01"})
            self.assertEqual(len(updated["oracles"]),13)
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
            self.assertEqual(run.call_count,56,"second run must REUSE all 56")
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
            self.assertEqual(run.call_count,56)


if __name__=="__main__":
    main()

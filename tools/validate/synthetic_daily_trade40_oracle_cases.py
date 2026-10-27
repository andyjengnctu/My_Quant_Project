"""Formal Min per-seed Oracle source and audit REUSE invariants (synthetic-only)."""
from __future__ import annotations

from datetime import date
from dataclasses import replace
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
    truth_producer_fingerprint,
    TARGET_ID, LABEL_ID,
)
from tools.validate.synthetic_daily_trade40_cases import fixture

EXACT = "DailyUniversal_MinTrade40_NetR"
DIRECT = "DailyUniversal_MinTrade40_NetR_Direct"
PROFILE = "daily_universal_full_horizon_mae_gated_mfe_t40_full_list_ndcg_pairwise"
_ORIGINAL_SCORE_CONTRACT_RESOLVER = oracle._resolve_oracle_target_score_contracts


def definition(*, full_cycle: bool = False):
    active = next(d for d in get_reusable_audit_definitions("breakout_quality")
                  if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
    if full_cycle:
        return active
    # Prior 40D/60D behavioral fixtures isolate existing targets. Dedicated
    # full-cycle tests below cover the added capability on the actual catalog.
    return replace(active, source={**active.source, "oracle_comparisons": tuple(
        entry for entry in active.source["oracle_comparisons"]
        if entry["target"] not in {"DailyUniversal_MinTrade_NetR",
                                   "DailyUniversal_MinTrade_NetR_Direct"})})


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
               evaluation_start="2021-01-01", cutoff="2021-05-15",
               horizon_sessions=40):
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
            horizon_sessions=horizon_sessions,
        )


def synthetic_perfect_score_contracts(_definition, _settings, plan):
    """Authorized synthetic score maps only, never production PIT evidence."""
    return {name: {
        "reference_arm_id": "synthetic_authorized",
        "score_start_date": "2021-01-01",
        "target_id": "synthetic_target",
        "source_profile_id": str(_definition.source[
            (plan["daily_recipes"][plan["daily_sources"][name]] if name in plan["daily_sources"]
             else plan["recipes"][plan["aliases"].get(name, name)])[1]]),
    } for name, _ in plan["targets"]}


def score_key_proof(keys):
    from hashlib import sha256
    ordered = sorted(keys)
    return (sha256("".join(f"{ticker}|{day}\n" for ticker, day in ordered).encode()).hexdigest(),
            len(ordered), min(day for _, day in ordered), max(day for _, day in ordered))


def synthetic_read_score_keys(_definition, *, contracts, **_kwargs):
    proof = score_key_proof({("2330", "2025-01-03"), ("2330", "2025-12-22")})
    return {name: proof for name in contracts}


def synthetic_price_scores(definition, *, project_root, cutoff, **_kwargs):
    panels, cohort = oracle._build_oracle_score_panels(
        definition, project_root=project_root, cutoff=cutoff)
    selected = {row["target"] for row in definition.source["oracle_comparisons"]}
    return {name: scores for name, scores in panels.items() if name in selected}, cohort


def execute_synthetic_score_fixture(definition, *, price_scores, daily_scores, keys,
                                    evidence_overrides=None, replayed=None):
    """Exercise the current executor gates with paired C96 and no market data.

    The input-key proof is independent of injected outcome values. A real
    Replay store verifies each produced identity; only economic execution and
    external source loading are synthetic.
    """
    from services.audit import daily_trade40_oracle as daily_owner
    from services.audit.oracle_replay_artifacts import OracleReplayStore
    from filters.breakout_quality import daily_trade40_target as trade_owner
    plan = oracle._oracle_comparison_plan(definition)
    settings, base, exact = oracle._contract(definition)
    contracts = _ORIGINAL_SCORE_CONTRACT_RESOLVER(definition, settings, plan)
    direct = (oracle._direct_arm(definition, settings, base, exact)
              if "direct" in plan["selectors"].values() else None)
    baseline_arm = exact or settings.arms[str(definition.source["exact_arm_id"])]
    base_policy = oracle._canonical_oracle_replay_policy(
        definition, base_arm=base, exact_arm=baseline_arm)
    seeds = tuple(daily_scores) or (1,)
    prepared = [(seed, "a" * 64, "rolling_active_param_ensemble", {}, {}) for seed in seeds]
    proof = score_key_proof(keys)
    schedule = source_fixture()[3]
    periods = [{"start": str(p.start.date()), "end": None if p.end is None else str(p.end.date()),
                "params_signature": p.params_signature} for p in schedule]
    observed = replayed if replayed is not None else []
    def source(*, seed, param_sha256, horizon_sessions, **_kwargs):
        name = next(name for name, row in plan["daily_recipes"].items() if row[2] == horizon_sessions)
        values = daily_scores[seed][name]
        target_id, label_id = daily_owner.target_identity_for_horizon(horizon_sessions)
        evidence = {
            "target_id": target_id, "label_id": label_id,
            "profile_id": str(definition.source[plan["daily_recipes"][name][1]]),
            "teacher": {"seed": seed, "param_identity_sha256": param_sha256, "periods": periods},
            "truth_source_sha256": daily_owner.truth_producer_fingerprint(),
            "dataset_summary_sha256": None,
            "rank_policy": "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
            "rank_sha256": "1" * 64, "raw_label_sha256": "2" * 64, "calendar_sha256": "3" * 64,
            "mature_eligible_rows": len(values), "settled_net_r_rows": len(values),
            "censored_rank_last_rows": 0, "research_cutoff": str(settings.end_date),
            "score_start_date": contracts[next(t for t, s in plan["daily_sources"].items() if s == name)]["score_start_date"],
        }
        if evidence_overrides:
            evidence.update(evidence_overrides(seed, name, evidence))
        return values, evidence
    def scenario(*, name, **_kwargs):
        observed.append(name)
        return {"name": name}
    end = str(definition.dimensions["oracle_evaluation_end_date"])
    def metrics(_result, *, target):
        return {"target": target, "total_return_pct": 5., "annualized_return_pct": 1.,
                "mdd_pct": 1., "romd": 5., "ev_a_r": .5, "trades": 1, "exposure_pct": 5.,
                "yearly": [{"year": year, "strategy_return_pct": 1.,
                            "is_full_year": year < pd.Timestamp(end).year}
                           for year in range(pd.Timestamp(settings.start_date).year, pd.Timestamp(end).year + 1)]}
    with TemporaryDirectory() as directory, \
         patch("services.audit.oracle_replay_artifacts.price_inventory_sha256", return_value="f" * 64), \
         patch.object(oracle, "_prepare_oracle_parameters", return_value=(prepared, [])), \
         patch.object(oracle, "_robustness_artifacts", return_value=(None, [(seed, Path("P16.json")) for seed in seeds])), \
         patch.object(oracle, "_load_or_build_oracle_price_scores", return_value=(price_scores, {})), \
         patch.object(oracle, "_price_score_algorithm_sha256", return_value="e" * 64), \
         patch.object(trade_owner, "resolve_formal_min_param_records", return_value=()), \
         patch.object(trade_owner, "build_formal_min_execution_schedule", return_value=(schedule, {})), \
         patch.object(daily_owner, "build_daily_universal_trade40_oracle_scores", side_effect=source), \
         patch.object(oracle, "_run_scenario", side_effect=scenario), \
         patch.object(oracle, "_metrics", side_effect=metrics):
        return oracle._execute_single_stage_missing(
            definition, project_root=Path(directory), quiet=True, plan=plan,
            contracts=contracts, current_keys={name: proof for name in contracts},
            settings=settings, robustness=SimpleNamespace(benchmark_id="synthetic_current_executor"),
            price_sha="f" * 64, prepared=prepared,
            decisions={(seed, name): "RUN" for seed in seeds for name in contracts},
            preflight_cached={(seed, "C96"): {"romd": 1.} for seed in seeds},
            start=str(settings.start_date), end=end, base_arm=base, exact_arm=exact,
            direct_arm=direct, base_policy=base_policy,
            cache=OracleReplayStore(Path(directory), definition.output_subdir))


def synthetic_score_parity_setup(testcase):
    # Replay-integrity fixtures verify cached-score identity rather than
    # reconstructing historical model PIT files. Original hypothetical score
    # method and absence of a FIT dependency are tested without mocks in
    # synthetic_target_oracle_cases.py.
    for name, action in (
        ("_resolve_oracle_target_score_contracts", synthetic_perfect_score_contracts),
        ("_read_only_score_keys", synthetic_read_score_keys),
        ("_load_or_build_oracle_price_scores", synthetic_price_scores),
        # The integration cases inject a synthetic price-panel producer. Its
        # callable is a Mock, so attest that controlled stub explicitly instead
        # of asking production inspect.getsource to hash the Mock object.
        ("_price_score_algorithm_sha256", lambda: "a" * 64),
    ):
        guard = patch.object(oracle, name, side_effect=action)
        guard.start()
        testcase.addCleanup(guard.stop)
    prices = patch("services.audit.oracle_replay_artifacts.price_inventory_sha256",
                   side_effect=lambda *args: oracle.price_inventory_sha256(*args))
    prices.start()
    testcase.addCleanup(prices.stop)


class FormalMinDailyOracleCases(TestCase):
    def setUp(self):
        synthetic_score_parity_setup(self)

    def test_truth_fingerprint_covers_delegated_execution_and_feature_owners(self):
        """A wrapper's unchanged bytes cannot attest changed economic code."""
        import inspect
        import shutil
        from services.audit import daily_trade40_oracle as source
        from services.audit import oracle_score_artifacts as shared_source
        from core.position_management import step_position_management
        from core.exit_priority import resolve_stop_tp_hits
        from core.order_lot_policy import apply_board_lot_preferred_qty_milli
        from core.capital_policy import resolve_single_backtest_sizing_capital
        from core.fee_rebate import validate_fee_rebate_settlement_basis
        from core.breakout_false_filter import build_breakout_false_filter_pass_condition
        from core.rolling_oos_params import build_active_param_schedule
        from core.active_param_ensemble import build_active_param_ensemble_schedule
        from core.params_io import build_params_from_mapping
        from core.training_performance import is_optimizer_native_hotpath_kernels_enabled
        from strategies.breakout.schema import validate_breakout_param_ranges
        from config import broker_accounting, execution_policy
        from core.data_utils import sanitize_ohlcv_dataframe
        from filters.breakout_quality.daily_ranker_data import resolve_daily_training_universe_start
        from filters.breakout_quality.models.spec_registry import get_model_spec
        from filters.breakout_quality.features import normalize_ohlcv_array_windows
        from core import native_optimizer_hotpath

        source_root = Path(source.__file__).resolve().parents[2]
        delegated_owners = {
            Path(inspect.getsourcefile(owner)).resolve().relative_to(source_root).as_posix()
            for owner in (
                step_position_management, resolve_stop_tp_hits,
                apply_board_lot_preferred_qty_milli, sanitize_ohlcv_dataframe,
                resolve_daily_training_universe_start, get_model_spec,
                normalize_ohlcv_array_windows,
                resolve_single_backtest_sizing_capital, validate_fee_rebate_settlement_basis,
                build_breakout_false_filter_pass_condition, build_active_param_schedule,
                build_active_param_ensemble_schedule, build_params_from_mapping,
                is_optimizer_native_hotpath_kernels_enabled, validate_breakout_param_ranges,
                broker_accounting, execution_policy,
            )
        }
        delegated_owners.add(native_optimizer_hotpath._SOURCE.relative_to(source_root).as_posix())
        declared = set(source._TRUTH_SOURCE_MODULES) | set(shared_source._DAILY_SCORE_SOURCE_MODULES)
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in declared | delegated_owners:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / relative, destination)
            original = source._truth_source_sha(root)
            for relative in sorted(delegated_owners):
                with self.subTest(owner=relative):
                    path = root / relative
                    previous = path.read_bytes()
                    if relative.endswith("/position_management.py"):
                        changed = previous.replace(
                            b"is_stop_hit = price_to_milli(t_low) <= int(position['sl_milli'])",
                            b"is_stop_hit = price_to_milli(t_low) < int(position['sl_milli'])",
                        )
                        self.assertNotEqual(changed, previous)
                    else:
                        changed = previous + (b"\n/* owner mutation fixture */\n" if path.suffix == ".c"
                                              else b"\n# owner mutation fixture\n")
                    path.write_bytes(changed)
                    self.assertNotEqual(source._truth_source_sha(root), original)
                    path.write_bytes(previous)
                    self.assertEqual(source._truth_source_sha(root), original)
            # Neither the requested horizon list nor renderer bytes are a
            # second source of the primitive's scientific implementation.
            unrelated = root / "doc" / "unrelated_report.md"
            unrelated.parent.mkdir(parents=True, exist_ok=True)
            unrelated.write_text("unrelated renderer evidence", encoding="utf-8")
            self.assertEqual(source._truth_source_sha(root), original)

    def test_warm_mintrade_receipt_rejects_changed_execution_owner_without_producing(self):
        from hashlib import sha256
        from services.audit import daily_trade40_oracle as source
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity

        active = definition()
        active = replace(active, source={**active.source, "oracle_comparisons": tuple(
            row for row in active.source["oracle_comparisons"] if row["target"] == EXACT)})
        settings, baseline, exact = oracle._contract(active)
        plan = oracle._oracle_comparison_plan(active)
        contracts = _ORIGINAL_SCORE_CONTRACT_RESOLVER(active, settings, plan)
        scores, evidence = run_source(cutoff=str(settings.end_date))
        keys = sorted(scores)
        key_sha = sha256("".join(f"{ticker}|{day}\n" for ticker, day in keys).encode()).hexdigest()
        proof = (key_sha, len(keys), min(day for _, day in keys), max(day for _, day in keys))
        cohort = {
            "cohort_rows": proof[1], "first_common_day": proof[2],
            "last_common_day": proof[3], "research_cutoff": str(settings.end_date),
        }
        score_contract = {**contracts[EXACT],
            "canonical_feature_eligible_keys_sha256": key_sha,
            "oracle_scored_rows": len(keys),
            "source_evidence": "validated_generic_daily_feature_universe_and_research_cutoff_closeout",
        }
        common = oracle._canonical_oracle_replay_policy(active, base_arm=baseline, exact_arm=exact)
        policy = oracle._arm_replay_policy(
            active, plan=plan, target=EXACT, replay_policy=common,
            daily_evidence=evidence, score_sha256=oracle._canonical_score_map_sha256(scores),
            score_contract=score_contract,
        )
        seed, price_sha, param_sha = 693545351, "b" * 64, "a" * 64
        benchmark = SimpleNamespace(benchmark_id="synthetic_owner_provenance")
        end = str(active.dimensions["oracle_evaluation_end_date"])
        identity = replay_identity(
            dataset=settings.dataset, dataset_sha256=price_sha, target=EXACT,
            definition={EXACT: EXACT}, start=str(settings.start_date), end=end,
            benchmark=benchmark.benchmark_id, param_sha256=param_sha, seed=seed,
            cohort=cohort, ranking_policy=str(exact.dl_runtime_mode),
            ranking_options=dict(exact.dl_runtime_options),
            evaluation_mode=str(active.dimensions["evaluation_mode"]),
            max_positions=settings.max_positions, replay_policy=policy,
        )
        observation = {"target": EXACT, "seed": seed, "romd": 1.,
            "total_return_pct": 1., "annualized_return_pct": 1., "mdd_pct": 1.,
            "ev_a_r": .1, "trades": 1, "exposure_pct": 1., "delta_romd_vs_c96": 0.,
            "yearly": [{"year": year, "strategy_return_pct": 1., "is_full_year": year < 2026}
                       for year in range(2021, 2027)]}
        with TemporaryDirectory() as temp_dir:
            store = OracleReplayStore(Path(temp_dir), active.output_subdir)
            store.write(identity, observation, origin="synthetic_execution_owner_provenance")
            arguments = dict(definition=active, store=store, contracts=contracts, plan=plan,
                prepared=[(seed, param_sha, None, None, None)], price_sha=price_sha,
                settings=settings, robustness=benchmark, evaluation_end=end)
            with patch.object(source, "compute_counterfactual_trade40_batch", side_effect=AssertionError("no truth precompute")), \
                 patch.object(oracle, "load_profile_continuous_ranker_data", side_effect=AssertionError("no DL labels")), \
                 patch.object(oracle, "_run_scenario", side_effect=AssertionError("no Replay")):
                self.assertEqual(oracle._replay_attested_score_keys(**arguments), {EXACT: proof})
                with patch.object(source, "truth_producer_fingerprint", return_value="f" * 64):
                    self.assertEqual(oracle._replay_attested_score_keys(**arguments), {})

    def test_existing_comparisons_are_preserved_and_report_contract_unchanged(self):
        d=definition()
        p=oracle._oracle_comparison_plan(d)
        self.assertEqual(len(p["targets"]),13)
        self.assertEqual(p["daily_recipes"], {
            EXACT: ("daily_trade_path_net_r", "daily_trade40_profile", 40),
            "DailyUniversal_MinTrade60_NetR":
                ("daily_trade_path_net_r", "daily_trade40_profile", 60),
        })
        self.assertEqual(p["daily_sources"], {
            EXACT: EXACT, DIRECT: EXACT,
            "DailyUniversal_MinTrade60_NetR": "DailyUniversal_MinTrade60_NetR",
            "DailyUniversal_MinTrade60_NetR_Direct": "DailyUniversal_MinTrade60_NetR",
        })
        self.assertEqual(p["selectors"][EXACT],"exact")
        self.assertEqual(p["selectors"][DIRECT],"direct")
        self.assertEqual(d.source["daily_trade40_profile"],PROFILE)
        self.assertEqual(require_approved_report_contract("audit.target_oracle_romd").version,9)

    def test_mixed_horizons_reject_incomplete_stock_day_not_silent_intersection(self):
        d = definition()
        plan = oracle._oracle_comparison_plan(d)
        keys = {("2330", "2025-11-25"), ("2317", "2025-11-25")}
        price = {name: {key: .5 for key in keys}
                 for name, _ in plan["targets"] if name not in plan["daily_sources"]}
        daily = {693545351: {
            source: {key: .5 for key in keys
                     if horizon == 40 or key[0] == "2330"}
            for source, (_recipe, _provider, horizon) in plan["daily_recipes"].items()
        }}
        with self.assertRaisesRegex(oracle.OracleBlockedError,
                                    "producer score eligibility differs"):
            execute_synthetic_score_fixture(d, price_scores=price, daily_scores=daily, keys=keys)

    def test_60_source_has_distinct_approved_identity_and_maturity(self):
        from services.audit.daily_trade40_oracle import target_identity_for_horizon
        forty, e40 = run_source(horizon_sessions=40)
        sixty, e60 = run_source(horizon_sessions=60)
        self.assertEqual(set(forty), set(sixty))
        self.assertEqual(target_identity_for_horizon(40), (TARGET_ID, LABEL_ID))
        self.assertEqual((e60["target_id"], e60["label_id"]),
                         target_identity_for_horizon(60))
        self.assertNotEqual(e60["target_id"], e40["target_id"])
        self.assertIn("research_cutoff_closeout", e60["cohort"])
        self.assertEqual(e60["calendar_sha256"], e40["calendar_sha256"])
        self.assertEqual(e60["teacher"], e40["teacher"])
        self.assertEqual(e60["first_replay_trade_date"], e40["first_replay_trade_date"])
        # Both periods retain the SAME feature rows. An H=60 observation
        # without 60 available sessions is settled at the Research cutoff,
        # not excluded or given a fabricated zero R.
        a40, e40_short = run_source(horizon_sessions=40, cutoff="2021-03-15")
        a60, e60_short = run_source(horizon_sessions=60, cutoff="2021-03-15")
        self.assertEqual(set(a40), set(a60))
        self.assertTrue(a60)
        self.assertGreater(e60_short["cutoff_truncated_rows"], 0)
        self.assertGreaterEqual(e60_short["cutoff_truncated_rows"], e40_short["cutoff_truncated_rows"])

    def test_all_declared_min_horizons_share_feature_universe_and_cutoff_policy(self):
        from services.audit.daily_trade40_oracle import target_identity_for_horizon
        full = definition(full_cycle=True)
        self.assertEqual(len(full.source["oracle_comparisons"]), 15)
        plan = oracle._oracle_comparison_plan(full)
        self.assertEqual({name: horizon for name, (_, _, horizon) in plan["daily_recipes"].items()}, {
            "DailyUniversal_MinTrade40_NetR": 40,
            "DailyUniversal_MinTrade60_NetR": 60,
            "DailyUniversal_MinTrade_NetR": None,
        })
        self.assertEqual(plan["daily_sources"]["DailyUniversal_MinTrade_NetR_Direct"],
                         "DailyUniversal_MinTrade_NetR")
        self.assertEqual(len(plan["targets"]), 15)
        settings, _base, _exact = oracle._contract(full)
        contracts = _ORIGINAL_SCORE_CONTRACT_RESOLVER(full, settings, plan)
        self.assertEqual(contracts["DailyUniversal_MinTrade_NetR"]["target_id"],
                         target_identity_for_horizon(None)[0])
        self.assertEqual(contracts["DailyUniversal_MinTrade_NetR_Direct"]["target_id"],
                         target_identity_for_horizon(None)[0])

        evidence = {}
        scored = {}
        for horizon in (40, 60, 87, 100, None):
            scored[horizon], evidence[horizon] = run_source(
                horizon_sessions=horizon, cutoff="2021-03-15",
            )
        keys = set(scored[40])
        self.assertTrue(keys)
        self.assertTrue(all(set(scores) == keys for scores in scored.values()))
        self.assertTrue(all(row["cutoff_truncated_rows"] > 0
                            for key, row in evidence.items() if key != 40))
        self.assertTrue(all(row["research_cutoff"] == "2021-03-15"
                            for row in evidence.values()))
        self.assertEqual(len({v["target_id"] for v in evidence.values()}), 5)

        # Direct can consume full-cycle truth without replaying an unselected
        # full-cycle Exact arm, or requiring a second Target producer.
        only_direct = replace(full, source={**full.source,
            "oracle_comparisons": tuple(item for item in full.source["oracle_comparisons"]
                if item["target"] == "DailyUniversal_MinTrade_NetR_Direct")})
        direct_plan = oracle._oracle_comparison_plan(only_direct)
        self.assertEqual(list(direct_plan["daily_recipes"]), ["DailyUniversal_MinTrade_NetR"])
        self.assertEqual(list(direct_plan["selectors"]), ["DailyUniversal_MinTrade_NetR_Direct"])

    def test_new_positive_integer_horizon_needs_only_registry_declaration(self):
        full = definition(full_cycle=True)
        name = "DailyUniversal_MinTrade87_NetR"
        additional = {"target": name, "description": "87-session hypothetical formal Min NetR",
                      "score": "daily_trade_path_net_r", "provider": "daily_trade40_profile",
                      "horizon_sessions": 87, "selector": "exact"}
        augmented = replace(full, source={**full.source,
            "oracle_comparisons": (*full.source["oracle_comparisons"], additional)})
        plan = oracle._oracle_comparison_plan(augmented)
        self.assertEqual(plan["daily_recipes"][name][2], 87)
        for invalid in (0, -3, True):
            changed = replace(augmented, source={**augmented.source,
                "oracle_comparisons": (*full.source["oracle_comparisons"],
                    {**additional, "horizon_sessions": invalid})})
            with self.assertRaisesRegex(oracle.OracleBlockedError, "horizon/mode"):
                oracle._oracle_comparison_plan(changed)

    def test_60_exact_direct_only_reuses_one_source_and_no_hidden_exact(self):
        from services.audit.daily_trade40_oracle import target_identity_for_horizon
        definition_ = definition()
        names = ("DailyUniversal_MinTrade60_NetR",
                 "DailyUniversal_MinTrade60_NetR_Direct")
        restricted = replace(definition_, source={**definition_.source,
            "oracle_comparisons": tuple(row for row in definition_.source["oracle_comparisons"]
                                        if row["target"] in names)})
        plan = oracle._oracle_comparison_plan(restricted)
        self.assertEqual(plan["daily_recipes"][names[0]][2], 60)
        self.assertEqual(plan["daily_sources"][names[1]], names[0])
        settings, _base, _exact = oracle._contract(restricted)
        mapping = _ORIGINAL_SCORE_CONTRACT_RESOLVER(restricted, settings, plan)
        self.assertEqual({mapping[name]["target_id"] for name in names},
                         {target_identity_for_horizon(60)[0]})
        solo = replace(restricted, source={**restricted.source,
            "oracle_comparisons": tuple(row for row in restricted.source["oracle_comparisons"]
                                        if row["target"] == names[1])})
        solo_plan = oracle._oracle_comparison_plan(solo)
        self.assertEqual(set(solo_plan["daily_recipes"]), {names[0]})
        self.assertEqual([x for x, _ in solo_plan["targets"]], [names[1]])
        altered = tuple({**row, "horizon_sessions": 40} if row["target"] == names[0]
                        else row for row in restricted.source["oracle_comparisons"])
        with self.assertRaisesRegex(oracle.OracleBlockedError, "scientific|score/selector|score"):
            oracle._oracle_comparison_plan(replace(restricted, source={**restricted.source,
                "oracle_comparisons": altered}))

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
        # Historical cache-reuse fixture deliberately uses the 2025-mature
        # observation window. The *current* Oracle now runs through the full
        # Research cutoff; this regression solely tests invariant per-arm
        # identities while adding a comparison within one old window.
        all_targets = definition()
        all_targets = replace(all_targets, dimensions={
            **all_targets.dimensions, "oracle_evaluation_end_date": "2025-12-22",
        })
        active = replace(all_targets, source={**all_targets.source,
            "oracle_comparisons": tuple(item for item in all_targets.source["oracle_comparisons"]
                                        if "MinTrade60" not in item["target"])})
        old=replace(active,source={**active.source,"oracle_comparisons":tuple(
            item for item in active.source["oracle_comparisons"]
            if item["target"] not in (EXACT,DIRECT))})
        plan=oracle._oracle_comparison_plan(old)
        price_scores={name:{("2330","2025-01-03"):0.8,
                            ("2330","2025-12-22"):0.8} for name in plan["recipes"]}
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
            score_map={("2330","2025-01-03"):0.625,
                       ("2330","2025-12-22"):0.625}
            evidence={"label_id":LABEL_ID,"target_id":TARGET_ID,"profile_id":PROFILE,
                "rank_policy":"average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
                "rank_sha256":"c"*64,"raw_label_sha256":"d"*64,
                "truth_source_sha256": truth_producer_fingerprint(),
                "dataset_summary_sha256": None,
                "calendar_sha256":"e"*64,
                "teacher":{"origin":"formal_min_roos_rolling_per_seed",
                           "seed":int(seed),"param_identity_sha256":str(param_sha256),
                           "periods":[{"start":"2021-01-01", "end":None,
                                       "params_signature":formal[0].params_signature}]},
                "mature_eligible_rows":2,"settled_net_r_rows":2,
                "censored_rank_last_rows":0,"research_cutoff":"2026-03-02",
                "first_date":"2025-01-03","last_date":"2025-12-22",
                "first_replay_trade_date":"2021-01-04", "score_start_date":"2021-01-01"}
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
            self.assertEqual(source.call_count,4,"warm Replay never regenerates Min truth")
            # A new producer revision must build its new truth and preserve the
            # previous completed identities; unchanged warm receipts need no
            # speculative producer call merely to rediscover their score SHA.
            changed_hash="0"*64
            def conflicting(*,seed,param_sha256,**kwargs):
                x,y=scores(seed=seed,param_sha256=param_sha256,**kwargs)
                y["raw_label_sha256"]=changed_hash
                y["truth_source_sha256"]="f"*64
                return x,y
            source.side_effect=conflicting
            with patch("services.audit.daily_trade40_oracle.truth_producer_fingerprint",return_value="f"*64):
                oracle.run_formal_audit(active,project_root=root,quiet=True)
            self.assertEqual(run.call_count,56,"only changed Min source replays need RUN")
            self.assertEqual(source.call_count,8)


    def test_60_mixed_40_window_replays_and_preserves_prior_caches(self):
        """The registry-fixed evaluation end is independent of requested horizons.

        Adding 60D must not alter the baseline or invalidate 40D identities.
        No shortened maturity cohort and no hidden intersection are allowed.
        """
        from filters.breakout_quality.runtime import resolve_breakout_quality_candidate_rank
        from services.audit.daily_trade40_oracle import target_identity_for_horizon

        full = definition()
        first = ("2330", "2025-11-25")
        second = ("2330", "2025-12-01")
        full = replace(full, dimensions={
            **full.dimensions, "oracle_evaluation_end_date": second[1],
        })
        forty = replace(full, source={**full.source,
            "oracle_comparisons": tuple(x for x in full.source["oracle_comparisons"]
                                        if "MinTrade60" not in x["target"])})
        price_plan = oracle._oracle_comparison_plan(forty)
        price = {name: {first: .25, second: .75} for name in price_plan["recipes"]}
        for alias, original in price_plan["aliases"].items():
            if original in price:
                price[alias] = price[original]
        cohort = {"cohort_rows": 2, "first_common_day": first[1],
                  "last_common_day": second[1], "research_cutoff": "2026-03-02",
                  "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
                  "oracle_only": True, "risk_budget_return": .1,
                  "providers": {"40d": full.source["pure_mfe40_provider"]}}
        seed = 693545351
        benchmark = SimpleNamespace(benchmark_id="synthetic_60_maturity")
        formal = source_fixture()[3]
        source_observations = []
        score_observations = []

        def source(*, seed, param_sha256, horizon_sessions=40, **kwargs):
            target, label = target_identity_for_horizon(horizon_sessions)
            values = {first: .625, second: .875}
            source_observations.append((horizon_sessions, tuple(values)))
            evidence = {"label_id": label, "target_id": target, "profile_id": PROFILE,
                "rank_policy": "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
                "rank_sha256": "c" * 64, "raw_label_sha256": "d" * 64,
                "truth_source_sha256": truth_producer_fingerprint(),
                "dataset_summary_sha256": None,
                "calendar_sha256": "e" * 64,
                "teacher": {"origin": "formal_min_roos_rolling_per_seed",
                            "seed": int(seed), "param_identity_sha256": str(param_sha256),
                            "periods": [{"start": "2021-01-01", "end": None,
                                         "params_signature": formal[0].params_signature}]},
                "mature_eligible_rows": len(values), "settled_net_r_rows": len(values),
                "censored_rank_last_rows": 0, "research_cutoff": "2026-03-02",
                "first_date": first[1], "last_date": max(k[1] for k in values),
                "first_replay_trade_date": "2021-01-04", "score_start_date": "2021-01-01",
            }
            return values, evidence

        def run(*, name, start_date, end_date, **kwargs):
            if name.startswith("Oracle "):
                target = name.removeprefix("Oracle ").split(" seed=")[0]
                score = resolve_breakout_quality_candidate_rank(
                    ticker="2330", signal_date="2025-11-24",
                    information_date=first[1], high_len=260,
                    filter_id="breakout_quality_v1")
                score_observations.append((target, end_date, score["score"], score["score_date"]))
            return {"name": name, "end_date": end_date}

        def metrics(replay, *, target):
            return {"target": target, "total_return_pct": 100.,
                "annualized_return_pct": 12., "mdd_pct": 10.,
                "romd": 7. if target == "C96" else 9., "ev_a_r": .5,
                "trades": 25, "exposure_pct": 60.,
                "yearly": [{"year": year, "strategy_return_pct": 5.,
                            "is_full_year": year != 2025} for year in range(2021, 2026)]}

        with TemporaryDirectory() as tmp, \
             patch.object(oracle, "_read_only_score_keys",
                          side_effect=lambda _d, *, contracts, **_kw: {
                              name: score_key_proof({first, second}) for name in contracts}), \
             patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
             patch.object(oracle, "_robustness_artifacts",
                          return_value=(benchmark, [(seed, Path("seed.json"))])), \
             patch.object(oracle, "_build_oracle_score_panels", return_value=(price, cohort)), \
             patch.object(oracle, "load_strategy_param_evaluation_view",
                          return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
             patch.object(oracle, "_validate_requested_param_policy"), \
             patch.object(oracle, "get_active_param_ensemble_date_range",
                          return_value=("2020-01-01", "2026-03-02")), \
             patch.object(oracle, "_build_controlled_param_source_pair",
                          return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
             patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
             patch("filters.breakout_quality.daily_trade40_target.resolve_formal_min_param_records",
                   return_value=({"effective_date": "2021-01-01",
                                  "params_obj": formal[0].params,
                                  "params_signature": formal[0].params_signature},)), \
             patch("services.audit.daily_trade40_oracle.build_daily_universal_trade40_oracle_scores",
                   side_effect=source) as source_mock, \
             patch.object(oracle, "_run_scenario", side_effect=run) as runner, \
             patch.object(oracle, "_metrics", side_effect=metrics):
            root = Path(tmp)
            old = oracle.run_formal_audit(forty, project_root=root, quiet=True)
            self.assertEqual(old["end_date"], second[1])
            self.assertEqual(len(old["oracles"]), 11)
            self.assertEqual(runner.call_count, 12)
            self.assertEqual(source_mock.call_count, 1)

            mixed = oracle.run_formal_audit(full, project_root=root, quiet=True)
            self.assertEqual(mixed["end_date"], second[1])
            self.assertEqual(mixed["common_cohort"]["cohort_rows"], 2)
            self.assertEqual(len(mixed["oracles"]), 13)
            markdown = oracle.render_result(mixed, target="markdown")
            self.assertIn("fixed evaluation window", markdown)
            self.assertIn("DailyUniversal_MinTrade60_NetR_Direct", markdown)
            self.assertEqual(runner.call_count, 14, "old 40D and baseline results reuse: only 60D exact/direct run")
            self.assertEqual(source_mock.call_count, 2, "only newly added 60D truth is prepared")
            self.assertEqual(source_observations, [
                (40, (first, second)), (60, (first, second))])
            self.assertTrue(all((record[2], record[3]) ==
                                ((.625 if "DailyUniversal" in record[0] else .25), first[1])
                                for record in score_observations))
            self.assertEqual({end for _, end, _, _ in score_observations}, {second[1]})
            self.assertEqual({row["target"] for row in mixed["oracles"]},
                             {row["target"] for row in full.source["oracle_comparisons"]})
            oracle.run_formal_audit(full, project_root=root, quiet=True)
            self.assertEqual(runner.call_count, 14, "identical fixed-evaluation identities REUSE")
            self.assertEqual(source_mock.call_count, 2, "warm Replay does not build either source")
            oracle.run_formal_audit(forty, project_root=root, quiet=True)
            self.assertEqual(runner.call_count, 14, "40D-only completed identities remain reusable")
            sixty = replace(full, source={**full.source, "oracle_comparisons": tuple(
                row for row in full.source["oracle_comparisons"]
                if "MinTrade60" in row["target"])})
            sixty_only = oracle.run_formal_audit(sixty, project_root=root, quiet=True)
            self.assertEqual(len(sixty_only["oracles"]), 2)
            self.assertEqual(sixty_only["end_date"], second[1])
            self.assertEqual(runner.call_count, 14,
                             "60D-only consumes completed 60D mixed cache")

            full_cycle = definition(full_cycle=True)
            full_cycle = replace(full_cycle, dimensions={
                **full_cycle.dimensions, "oracle_evaluation_end_date": second[1],
            })
            expanded = oracle.run_formal_audit(full_cycle, project_root=root, quiet=True)
            self.assertEqual(expanded["end_date"], second[1])
            self.assertEqual(expanded["common_cohort"]["cohort_rows"], 2)
            self.assertEqual(len(expanded["oracles"]), 15)
            self.assertEqual(runner.call_count, 16,
                             "adding full-cycle retains 40D and 60D Replay identities")
            self.assertEqual(source_mock.call_count, 3, "only new full-cycle source is built")
            oracle.run_formal_audit(full_cycle, project_root=root, quiet=True)
            self.assertEqual(runner.call_count, 16, "all fixed-end arms re-use on repeat")
            full_only = replace(full_cycle, source={**full_cycle.source,
                "oracle_comparisons": tuple(row for row in full_cycle.source["oracle_comparisons"]
                                            if row["target"].startswith("DailyUniversal_MinTrade_") )})
            alone = oracle.run_formal_audit(full_only, project_root=root, quiet=True)
            self.assertEqual(len(alone["oracles"]), 2)
            self.assertEqual(alone["end_date"], second[1])
            self.assertEqual(runner.call_count, 16,
                             "full-cycle-only reuses without altering period or other targets")
            self.assertEqual(source_mock.call_count, 3, "all repeated subsets reuse existing truth receipts")


if __name__=="__main__":
    main()

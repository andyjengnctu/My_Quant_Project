"""Isolated same-truth U40 evidence regression, independent of Windows formal runner."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile

import numpy as np
import pandas as pd

from core.breakout_quality_policy import get_breakout_quality_workflow_settings
from core.research_report_contract import validate_approved_persistent_report_contracts
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_target import StrategyAlignedContinuousTargetSpec
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from services.breakout_quality import u40_paired_comparison as service
from services.research.breakout_quality_application import _render_model_comparison_specific_extensions

AO_PROFILE = "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
CE_PROFILE = "daily_universal_full_horizon_mfe_minus_d40_huber"
CF_PROFILE = "daily_universal_full_horizon_mfe_minus_d40_full_list_ndcg_pairwise"
CG_PROFILE = "daily_universal_shared_full_d40_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
CH_PROFILE = "daily_universal_shared_full_d20_safety_hs_conditional_mfe_full_list_ndcg_pairwise"


def _fixture():
    budget = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY).risk_budget_return
    records = []
    for date in pd.to_datetime(("2021-03-01", "2021-04-01", "2022-03-01", "2022-04-01")):
        for k in range(20):
            mfe = 2.5 if k % 2 == 0 else 3.6
            d40 = 0.15 if k % 2 == 0 else 2.75
            records.append({
                "ticker": f"S{k:03d}", "date": date,
                "label_eval_end_date": date + pd.Timedelta(days=50),
                "target_favorable_r": mfe,
                "target_minimum_low_return": -budget * d40,
                "group_index": len(records), "actual_u40": mfe - d40,
            })
    group = pd.DataFrame(records)
    raw = group["actual_u40"].to_numpy(dtype=np.float32)
    group.drop(columns=["actual_u40"], inplace=True)
    # An inference-only tail with no mature label must never enter paired truth.
    group = pd.concat([group, pd.DataFrame([{
        "ticker": "LATE", "date": pd.Timestamp("2026-03-02"),
        "label_eval_end_date": pd.NaT,
        "group_index": len(group), "target_favorable_r": np.nan,
        "target_minimum_low_return": np.nan,
    }])], ignore_index=True)
    bundle = SimpleNamespace(
        group_table=group,
        target_valid=np.array([True] * (len(group) - 1) + [False], dtype=bool),
        raw_target=np.r_[raw, np.nan],
    )
    candidates = set(zip(group.loc[group["ticker"].isin([f"S{k:03d}" for k in range(12)]), "ticker"],
                         group.loc[group["ticker"].isin([f"S{k:03d}" for k in range(12)]), "date"]))
    return bundle, candidates


def _items(tmp: Path, bundle, *, mode, include_ranker: bool = False,
           include_full_d40: bool = False, include_direct: bool = True,
           include_short_horizon: bool = False):
    scores = bundle.group_table.iloc[:-1][list(service.PRIMARY_COLUMNS)].copy()
    profiles = (("AO", AO_PROFILE),)
    if include_direct:
        profiles += (("CE", CE_PROFILE),)
    if include_ranker:
        profiles += (("CF", CF_PROFILE),)
    if include_full_d40:
        profiles += (("CG", CG_PROFILE),)
    if include_short_horizon:
        profiles += (("CH", CH_PROFILE),)
    items=[]
    for name, slug in profiles:
        settings = get_breakout_quality_workflow_settings(experiment_profile=slug)
        frame = scores.copy()
        is_even = frame["ticker"].str[-1].astype(int) % 2 == 0
        col = "model_score" if mode == "forward_oos" else "breakout_quality_score"
        frame[col] = (np.where(is_even, 0.2, 0.9) if name == "AO"
                      else np.where(is_even, 2.35, -0.55) if name == "CE"
                      else np.where(is_even, 0.85, 0.10) if name == "CF"
                      else np.where(is_even, 0.25, 0.75))
        if mode == "rolling_oos":
            frame["fold_id"] = np.where(frame["date"].dt.year == 2021, "fold_2021", "fold_2022")
            frame["model_information_cutoff"] = np.where(frame["date"].dt.year == 2021, "2020-12-31", "2021-12-31")
        else:
            frame["target_raw_r"] = 0.1  # Forward mature OOS only
        path = tmp/f"{name}_{mode}.csv.gz"
        frame.to_csv(path, index=False, encoding="utf-8-sig", compression="infer")
        report = {"artifacts": {"oos_scores_gzip": build_file_manifest(path)}}
        items.append({"model_id": name, "settings": settings,
                      "contract": SimpleNamespace(score_path=path, report=report,
                                                  continuous_target_id=settings.continuous_target_id),
                      "payload": {"immutable_original_metric": name}})
    return items


def case():
    issues = validate_approved_persistent_report_contracts()
    assert not issues, issues
    bundle, candidates = _fixture()
    truth = service._canonical_u40_truth(bundle)
    assert len(truth)==80
    assert np.allclose(truth["u40_r"], truth["mfe40_r"]-truth["d40_r"], atol=1e-5)
    assert (truth["d40_r"] >= 0).all()
    with tempfile.TemporaryDirectory(prefix="u40_paired_case_") as temp:
        root=Path(temp)
        for mode in ("rolling_oos", "forward_oos"):
            items = _items(root, bundle, mode=mode)
            with patch.object(service,"load_daily_universal_ranker_data",return_value=bundle), patch.object(
                  service, "load_official_breakout_candidate_keys",return_value=candidates):
                output = service.attach_paired_u40_extension(items, evaluation_mode=mode, project_root=root)
            ext = {i["model_id"]: next(x for x in i["payload"]["comparison_extensions"]
                          if x["id"] == service.EXTENSION_ID) for i in output}
            ao, ce = ext["AO"],ext["CE"]
            assert len(ao["rows"]) == len(ce["rows"]) == 2
            assert len(ao["year_rows"]) == len(ce["year_rows"]) == 4
            assert ao["cohort_sha256"] == ce["cohort_sha256"]
            assert output[0]["payload"]["immutable_original_metric"] == "AO"
            assert output[1]["payload"]["immutable_original_metric"] == "CE"
            for a,c in zip(ao["rows"],ce["rows"]):
                assert a["split"] == c["split"]
                assert a["top10_n"] == c["top10_n"] and a["stock_days"] == c["stock_days"]
                assert c["top10_u40_r"] > a["top10_u40_r"]
                assert c["top10_mfe40_r"] < a["top10_mfe40_r"]
                assert c["top10_d40_r"] < a["top10_d40_r"]
                assert c["delta_u40_vs_reference_r"] > 0
                assert abs(c["top10_mfe40_r"] - c["top10_d40_r"] - c["top10_u40_r"]) < 1e-5
            for a,c in zip(ao["year_rows"],ce["year_rows"]):
                assert a["year"] == c["year"]
                assert c["delta_u40_vs_reference_r"] > 0
            views=[{"model_id":i["model_id"],
                    "view":{"extensions":i["payload"]["comparison_extensions"]}} for i in output]
            rendered="\n".join(_render_model_comparison_specific_extensions(views,target="markdown"))
            for token in ("MFE40 - D40", "Top10 U40", "Top10 MFE40", "Top10 D40", "vs original U40", "Year"):
                assert token in rendered, token
            # Every violation must refuse, not silently shrink a pair or coerce signed CE to [0,1].
            good_ref=service._read_validated_score(items[0],mode=mode)
            good_ce=service._read_validated_score(items[1],mode=mode)
            corrupted=good_ce.copy()
            corrupted.loc[0,"group_index"] = -99
            with np.testing.assert_raises_regex(ValueError,"group index mismatch"):
                service._matched_rows(truth=truth,reference=good_ref,treatment=corrupted,mode=mode)
            if mode=="rolling_oos":
                corrupted=good_ce.copy()
                corrupted.loc[0,"fold_id"] = "fold_wrong"
                with np.testing.assert_raises_regex(ValueError,"fold or information cutoff"):
                    service._matched_rows(truth=truth,reference=good_ref,treatment=corrupted,mode=mode)
                corrupted=good_ce.copy()
                corrupted.loc[0,"model_information_cutoff"] = pd.Timestamp("2022-12-31")
                with np.testing.assert_raises_regex(ValueError,"fold or information cutoff"):
                    service._matched_rows(truth=truth,reference=good_ref,treatment=corrupted,mode=mode)
            else:
                # The trust boundary for Forward scores includes the published SHA.
                file=items[1]["contract"].score_path
                corrupt_csv=pd.read_csv(file,encoding="utf-8-sig")
                corrupt_csv.loc[0,"model_score"] = 19.5
                corrupt_csv.to_csv(file,index=False,encoding="utf-8-sig",compression="infer")
                with np.testing.assert_raises_regex(ValueError,"SHA/file manifest"):
                    service._read_validated_score(items[1],mode=mode)
            # No relevant direct model: the current Standard SOP payload remains unchanged.
            out=service.attach_paired_u40_extension(items[:1],evaluation_mode=mode,project_root=root)
            assert out[0]["payload"]==items[0]["payload"]
            assert "comparison_extensions" not in out[0]["payload"]
            if mode=="rolling_oos":
                csv_path=items[0]["contract"].score_path
                bad=pd.read_csv(csv_path,encoding="utf-8-sig")
                bad.loc[0,"breakout_quality_score"]=2.0
                bad.to_csv(csv_path,index=False,encoding="utf-8-sig",compression="infer")
                with np.testing.assert_raises_regex(ValueError,"\\[0,1\\]"):
                    service._read_validated_score(items[0],mode=mode)
            corrupted=SimpleNamespace(group_table=bundle.group_table,
                                       raw_target=bundle.raw_target.copy(),target_valid=bundle.target_valid)
            corrupted.raw_target[0]+=1
            with np.testing.assert_raises_regex(ValueError,"inconsistent with the canonical"):
                service._canonical_u40_truth(corrupted)




def three_way_case() -> None:
    bundle, candidates = _fixture()
    with tempfile.TemporaryDirectory(prefix="u40_aocf_three_way_") as temp:
        root = Path(temp)
        for mode in ("rolling_oos", "forward_oos"):
            items = _items(root, bundle, mode=mode, include_ranker=True)
            with patch.object(service, "load_daily_universal_ranker_data", return_value=bundle), patch.object(
                service, "load_official_breakout_candidate_keys", return_value=candidates
            ):
                output = service.attach_paired_u40_extension(
                    items, evaluation_mode=mode, project_root=root
                )
            ext = {i["model_id"]: next(x for x in i["payload"]["comparison_extensions"]
                          if x["id"] == service.EXTENSION_ID) for i in output}
            assert set(ext) == {"AO", "CE", "CF"}
            assert {len(ex["rows"]) for ex in ext.values()} == {2}
            assert {len(ex["year_rows"]) for ex in ext.values()} == {4}
            assert all(ex["cohort_sha256"] == ext["AO"]["cohort_sha256"] for ex in ext.values())
            rendered = "\n".join(_render_model_comparison_specific_extensions(
                [{"model_id": item["model_id"],
                  "view": {"extensions": item["payload"]["comparison_extensions"]}}
                 for item in output], target="markdown"
            ))
            for token in ("MFE40 - D40", "AO", "CE", "CF", "Top10 U40", "Top10 D40"):
                assert token in rendered, token
            for split_index in (0, 1):
                metrics = [ext[k]["rows"][split_index] for k in ("AO", "CE", "CF")]
                assert len({r["top10_n"] for r in metrics}) == 1
                assert len({r["stock_days"] for r in metrics}) == 1
                assert metrics[1]["top10_u40_r"] > metrics[0]["top10_u40_r"]
                assert metrics[2]["top10_u40_r"] > metrics[0]["top10_u40_r"]
                assert metrics[2]["delta_u40_vs_reference_r"] > 0
                assert all(abs(r["top10_mfe40_r"] - r["top10_d40_r"]
                               - r["top10_u40_r"]) < 1e-5 for r in metrics)
            ref = service._read_validated_score(items[0], mode=mode)
            ce = service._read_validated_score(items[1], mode=mode)
            cf = service._read_validated_score(items[2], mode=mode)
            bad = cf.copy()
            bad.loc[0, "group_index"] = -999
            with np.testing.assert_raises_regex(ValueError, "group index mismatch"):
                service._matched_rows(truth=service._canonical_u40_truth(bundle),
                    reference=ref, treatments={"direct_0": ce, "direct_1": bad}, mode=mode)
            if mode == "rolling_oos":
                bad = cf.copy()
                bad.loc[0, "fold_id"] = "fold_wrong"
                with np.testing.assert_raises_regex(ValueError, "fold or information cutoff"):
                    service._matched_rows(truth=service._canonical_u40_truth(bundle),
                        reference=ref, treatments={"direct_0": ce, "direct_1": bad}, mode=mode)
                # Pairwise score is a bounded ranking probability, unlike CE's signed margin.
                path = items[2]["contract"].score_path
                frame = pd.read_csv(path, encoding="utf-8-sig")
                frame.loc[0, "breakout_quality_score"] = 1.1
                frame.to_csv(path, index=False, encoding="utf-8-sig", compression="infer")
                with np.testing.assert_raises_regex(ValueError, "\\[0,1\\]"):
                    service._read_validated_score(items[2], mode=mode)
            else:
                path = items[2]["contract"].score_path
                frame = pd.read_csv(path, encoding="utf-8-sig")
                frame.loc[0, "model_score"] = 0.4
                frame.to_csv(path, index=False, encoding="utf-8-sig", compression="infer")
                with np.testing.assert_raises_regex(ValueError, "SHA/file manifest"):
                    service._read_validated_score(items[2], mode=mode)

def full_d40_four_way_case() -> None:
    bundle,candidates=_fixture()
    with tempfile.TemporaryDirectory(prefix="u40_aocfcg_four_way_") as temp:
        root=Path(temp)
        for mode in ("rolling_oos","forward_oos"):
            items=_items(root,bundle,mode=mode,include_ranker=True,include_full_d40=True)
            with patch.object(service,"load_daily_universal_ranker_data",return_value=bundle), patch.object(
                service,"load_official_breakout_candidate_keys",return_value=candidates
            ):
                output=service.attach_paired_u40_extension(items,evaluation_mode=mode,project_root=root)
            ext={item["model_id"]:next(x for x in item["payload"]["comparison_extensions"]
                  if x["id"] == service.EXTENSION_ID) for item in output}
            assert set(ext)=={"AO","CE","CF","CG"}
            assert all(len(value["rows"])==2 and len(value["year_rows"])==4
                       for value in ext.values())
            assert len({value["cohort_sha256"]["Rolling OOS" if mode=="rolling_oos" else "Forward OOS"]
                        for value in ext.values()})==1
            assert all(ext[name]["rows"][0]["top10_n"]==ext["AO"]["rows"][0]["top10_n"]
                       for name in ext)
            assert all(abs(row["top10_u40_r"]-row["top10_mfe40_r"]+row["top10_d40_r"])<1e-5
                       for value in ext.values() for row in value["rows"])
            rendered="\n".join(_render_model_comparison_specific_extensions(
                [{"model_id":item["model_id"],"view":{"extensions":item["payload"]["comparison_extensions"]}}
                 for item in output],target="markdown"))
            assert all(token in rendered for token in ("AO","CE","CF","CG","Top10 U40","Top10 D40"))
            ref=service._read_validated_score(items[0],mode=mode)
            valid={name:service._read_validated_score(item,mode=mode)
                   for name,item in zip(("CE","CF","CG"),items[1:])}
            corrupted=valid["CG"].copy()
            corrupted.loc[0,"group_index"]=-7
            with np.testing.assert_raises_regex(ValueError,"group index mismatch"):
                service._matched_rows(truth=service._canonical_u40_truth(bundle),reference=ref,
                    treatments={"direct_0":valid["CE"],"direct_1":valid["CF"],"direct_2":corrupted},mode=mode)
            if mode=="rolling_oos":
                corrupted=valid["CG"].copy();corrupted.loc[0,"fold_id"]="other_fold"
                with np.testing.assert_raises_regex(ValueError,"fold or information cutoff"):
                    service._matched_rows(truth=service._canonical_u40_truth(bundle),reference=ref,
                        treatments={"direct_0":valid["CE"],"direct_1":valid["CF"],"direct_2":corrupted},mode=mode)
                file=items[-1]["contract"].score_path
                frame=pd.read_csv(file,encoding="utf-8-sig");frame.loc[0,"breakout_quality_score"]=1.5
                frame.to_csv(file,index=False,encoding="utf-8-sig",compression="infer")
                with np.testing.assert_raises_regex(ValueError,r"\[0,1\]"):
                    service._read_validated_score(items[-1],mode=mode)
            else:
                file=items[-1]["contract"].score_path
                frame=pd.read_csv(file,encoding="utf-8-sig");frame.loc[0,"model_score"]=0.1
                frame.to_csv(file,index=False,encoding="utf-8-sig",compression="infer")
                with np.testing.assert_raises_regex(ValueError,"SHA/file manifest"):
                    service._read_validated_score(items[-1],mode=mode)



def ao_cg_without_direct_u40_case() -> None:
    """Active AO/CG must retain paired U40 evidence without CE/CF score artifacts."""
    bundle, candidates = _fixture()
    pure_mfe = bundle.group_table["target_favorable_r"].to_numpy(dtype=np.float64)
    ao_bundle = SimpleNamespace(
        group_table=bundle.group_table,
        target_valid=bundle.target_valid,
        raw_target=pure_mfe,
    )
    truth = service._canonical_u40_truth(
        ao_bundle, source_target_id=service.DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    )
    assert np.allclose(truth["u40_r"], truth["mfe40_r"] - truth["d40_r"], atol=1e-5)
    assert (truth["u40_r"] < truth["mfe40_r"]).any()
    assert len(truth) == 80

    with tempfile.TemporaryDirectory(prefix="u40_aocg_without_cecf_") as temp:
        root = Path(temp)
        for mode in ("rolling_oos", "forward_oos"):
            items = _items(root, bundle, mode=mode, include_direct=False, include_full_d40=True)
            assert [i["model_id"] for i in items] == ["AO", "CG"]
            with patch.object(service, "load_daily_universal_ranker_data", return_value=ao_bundle) as producer, \
                 patch.object(service, "load_official_breakout_candidate_keys", return_value=candidates):
                output = service.attach_paired_u40_extension(
                    items, evaluation_mode=mode, project_root=root,
                )
            assert producer.call_count == 1
            assert producer.call_args.kwargs["experiment_profile"] == AO_PROFILE
            ext = {
                i["model_id"]: next(
                    x for x in i["payload"]["comparison_extensions"]
                    if x["id"] == service.EXTENSION_ID
                ) for i in output
            }
            assert set(ext) == {"AO", "CG"}
            assert ext["AO"]["cohort_sha256"] == ext["CG"]["cohort_sha256"]
            assert len(ext["AO"]["rows"]) == len(ext["CG"]["rows"]) == 2
            assert len(ext["AO"]["year_rows"]) == len(ext["CG"]["year_rows"]) == 4
            for a, c in zip(ext["AO"]["rows"], ext["CG"]["rows"]):
                assert a["stock_days"] == c["stock_days"]
                assert a["top10_n"] == c["top10_n"]
                assert abs(c["top10_u40_r"] - c["top10_mfe40_r"] + c["top10_d40_r"]) < 1e-5
            rendered = "\n".join(_render_model_comparison_specific_extensions(
                [{"model_id": i["model_id"], "view": {
                    "extensions": i["payload"]["comparison_extensions"]}}
                 for i in output], target="markdown",
            ))
            assert all(token in rendered for token in ("AO", "CG", "Top10 U40", "Top10 D40"))
            assert "CE" not in [item["model_id"] for item in output]
            assert "CF" not in [item["model_id"] for item in output]
            # The AO Producer's raw target must match the Pure-MFE raw-R contract;
            # a mismatched target cannot silently be treated as U40 truth.
            altered = SimpleNamespace(
                group_table=bundle.group_table, target_valid=bundle.target_valid,
                raw_target=pure_mfe.copy(),
            )
            altered.raw_target[0] += 0.2
            with np.testing.assert_raises_regex(ValueError, "inconsistent with the canonical"):
                service._canonical_u40_truth(
                    altered, source_target_id=service.DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
                )
            with np.testing.assert_raises_regex(ValueError, "requires a canonical mature"):
                service._canonical_u40_truth(ao_bundle, source_target_id="unsupported")
            ref = service._read_validated_score(items[0], mode=mode)
            cg = service._read_validated_score(items[1], mode=mode)
            corrupt = cg.copy()
            corrupt.loc[0, "group_index"] = -13
            with np.testing.assert_raises_regex(ValueError, "group index mismatch"):
                service._matched_rows(truth=truth, reference=ref, treatment=corrupt, mode=mode)
            if mode == "rolling_oos":
                corrupt = cg.copy()
                corrupt.loc[0, "fold_id"] = "wrong_fold"
                with np.testing.assert_raises_regex(ValueError, "fold or information cutoff"):
                    service._matched_rows(truth=truth, reference=ref, treatment=corrupt, mode=mode)
            else:
                path = items[1]["contract"].score_path
                frame = pd.read_csv(path, encoding="utf-8-sig")
                frame.loc[0, "model_score"] = 0.8
                frame.to_csv(path, index=False, encoding="utf-8-sig", compression="infer")
                with np.testing.assert_raises_regex(ValueError, "SHA/file manifest"):
                    service._read_validated_score(items[1], mode=mode)



def fixed_20bar_loader_cohort_case() -> None:
    """Verify the real data-provider's 20D target and 40D PIT maturity on one stock universe."""
    from filters.breakout_quality import daily_ranker_data as loader
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from core.breakout_quality_registry import get_breakout_quality_experiment_profile

    cg_semantics = training_semantics(get_breakout_quality_experiment_profile(CG_PROFILE))[
        "shared_safety_hs_conditional_mfe_duo_head_contract"]
    ch_semantics = training_semantics(get_breakout_quality_experiment_profile(CH_PROFILE))[
        "shared_safety_hs_conditional_mfe_duo_head_contract"]
    assert "full_40bar" in cg_semantics["safety_target"]
    assert "full_d40" in cg_semantics["conditional_mfe_supervision_scope"]
    assert "full_20bar" in ch_semantics["safety_target"]
    assert "full_d20" in ch_semantics["conditional_mfe_supervision_scope"]

    n = 380
    calendar = pd.bdate_range("2020-01-01", periods=n)
    benchmark_ticker = str(DEFAULT_LABEL_POLICY.benchmark_ticker)
    tickers = (benchmark_ticker, "2330", "2317")
    frames = {}
    for ticker in tickers:
        positions = np.arange(n)
        high = (np.where(positions % 43 > 20, 111., 108.) if ticker == "2330"
                else np.where(positions % 47 > 23, 103., 107.))
        low = (np.where(positions % 43 > 20, 89., 98.) if ticker == "2330"
               else np.where(positions % 47 > 23, 86., 94.))
        frames[ticker] = pd.DataFrame({
            "Open": 100., "High": high, "Low": low, "Close": 100., "Volume": 1e6,
        }, index=calendar)
    summary = {"dataset": "full", "source_data_date_range": {
        "start": str(calendar[0].date()), "end": str(calendar[-1].date()),
    }}
    outer = {"effective_oos_end_date": str(calendar[-1].date()),
             "oos_start_date": str(calendar[0].date())}
    with patch.object(loader, "load_validated_dataset_bundle",
                      return_value=(summary, None, None, None, None)), \
         patch.object(loader, "resolve_breakout_quality_outer_policy", return_value=outer), \
         patch.object(loader, "discover_dataset_csv_inputs", return_value=(
             [(ticker, Path(ticker)) for ticker in tickers], None)), \
         patch.object(loader, "load_dataset_frame",
                      side_effect=lambda path, ticker, *, min_rows: frames[ticker]):
        kwargs = dict(filter_id="breakout_quality_v1",
                      model_architecture="inception_time_shared_safety_mfe_v1",
                      preload_feature_bank=False, allow_stale_source=False)
        cg = loader.load_daily_universal_ranker_data(experiment_profile=CG_PROFILE, **kwargs)
        ch = loader.load_daily_universal_ranker_data(experiment_profile=CH_PROFILE, **kwargs)
    cols = ["ticker", "date", "group_index"]
    assert cg.group_table[cols].astype(str).equals(ch.group_table[cols].astype(str))
    assert np.array_equal(cg.target_valid, ch.target_valid)
    assert cg.group_table["label_eval_end_date"].equals(ch.group_table["label_eval_end_date"])
    assert ch.target_manifest["target_contract"]["horizon_bars"] == 20
    assert ch.target_manifest["target_contract"]["target_id"] == (
        "daily_20bar_full_horizon_pure_mfe_r_v1")
    assert ch.target_manifest["sample_universe_horizon_bars"] == 40
    assert bool(np.any(cg.target_valid))
    assert not np.allclose(cg.raw_target[cg.target_valid], ch.raw_target[ch.target_valid])
    matured = ch.group_table.loc[ch.target_valid, ["date", "label_eval_end_date"]]
    assert bool((matured["label_eval_end_date"] > matured["date"]).all())


def fixed_20bar_utility_case() -> None:
    """20D CH truth is independently materialized; 40D AO/CG truth is frozen."""
    from dataclasses import replace
    from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
    from filters.breakout_quality.continuous_target import StrategyAlignedContinuousTargetSpec
    from filters.breakout_quality.hs_conditional_mfe import build_hs_conditional_mfe_targets
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from core.breakout_quality_runtime import (
        DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        DAILY_20BAR_FULL_HORIZON_PURE_MFE_TARGET_ID,
        CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE,
    )

    cg = get_continuous_ranker_execution_recipe(CG_PROFILE)
    ch = get_continuous_ranker_execution_recipe(CH_PROFILE)
    assert cg.model_architecture == ch.model_architecture
    assert cg.training_objective == ch.training_objective
    assert cg.loss_name == ch.loss_name
    assert cg.objective_policy == ch.objective_policy
    assert cg.target_policy.horizon_bars is None
    assert ch.target_policy.horizon_bars == 20
    assert ch.target_policy.sample_universe_horizon_bars == 40
    assert ch.continuous_target_id == DAILY_20BAR_FULL_HORIZON_PURE_MFE_TARGET_ID

    # Same signal-day close but a larger rally and a deep post-20-bar loss.
    # The 20-day target MUST exclude both post-20 values.
    frame = pd.DataFrame({
        "Close": np.full(41, 100.0),
        "High": np.r_[100.0, np.full(20, 105.0), np.full(20, 125.0)],
        "Low": np.r_[100.0, np.full(20, 99.0), np.full(20, 72.0)],
    })
    spec40 = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    spec20 = replace(spec40, horizon_bars=20)
    a = compute_daily_opportunity_target_batch(
        frame, np.array([0]), spec=spec40,
        target_id=DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    )
    b = compute_daily_opportunity_target_batch(
        frame, np.array([0]), spec=spec20,
        target_id=DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    )
    budget = spec40.risk_budget_return
    assert bool(a.valid_mask[0]) and bool(b.valid_mask[0])
    assert np.isclose(float(a.target_raw_r[0]), .25/budget)
    assert np.isclose(float(b.target_raw_r[0]), .05/budget)
    assert np.isclose(float(a.minimum_low_return[0]), -.28)
    assert np.isclose(float(b.minimum_low_return[0]), -.01)
    assert np.isclose(float(b.adverse_return_to_peak[0]), .01)

    full, candidates = _fixture()
    short_group = full.group_table.copy()
    # Distinct 20D path on exact same comparison cohort; published 20D target
    # must match its own MFE20 and the 40D target must remain unchanged.
    short_group["target_favorable_r"] = np.where(
        short_group["ticker"].str[-1].str.isdigit().fillna(False),
        np.nan_to_num(full.group_table["target_favorable_r"], nan=0.) * .52,
        0.,
    )
    short_group["target_minimum_low_return"] = (
        np.nan_to_num(full.group_table["target_minimum_low_return"], nan=0.) * .48
    )
    short_group.loc[~full.target_valid, ["target_favorable_r", "target_minimum_low_return"]] = np.nan
    short_mfe = short_group["target_favorable_r"].to_numpy(dtype=np.float64)
    short_bundle = SimpleNamespace(
        group_table=short_group, target_valid=full.target_valid,
        raw_target=short_mfe,
        target_manifest={"target_contract": {"target_id": DAILY_20BAR_FULL_HORIZON_PURE_MFE_TARGET_ID,
                                             "horizon_bars": 20}},
    )
    truth20 = service._canonical_horizon_utility_truth(
        short_bundle, source_target_id=DAILY_20BAR_FULL_HORIZON_PURE_MFE_TARGET_ID,
        horizon_bars=20,
    )
    truth40 = service._canonical_u40_truth(full)
    assert len(truth20) == len(truth40) == 80
    assert not np.allclose(truth20["u20_r"], truth40["u40_r"])
    assert np.allclose(truth20["u20_r"], truth20["mfe20_r"] - truth20["d20_r"])
    broken = SimpleNamespace(group_table=short_group, target_valid=full.target_valid,
                              raw_target=short_mfe, target_manifest={"target_contract": {
                                  "target_id": DAILY_20BAR_FULL_HORIZON_PURE_MFE_TARGET_ID,
                                  "horizon_bars": 40}})
    with np.testing.assert_raises_regex(ValueError, "horizon or identity mismatch"):
        service._canonical_horizon_utility_truth(
            broken, source_target_id=DAILY_20BAR_FULL_HORIZON_PURE_MFE_TARGET_ID,
            horizon_bars=20,
        )
    pure_group = full.group_table.copy()
    ao_bundle = SimpleNamespace(group_table=pure_group, target_valid=full.target_valid,
        raw_target=pure_group["target_favorable_r"].to_numpy(dtype=np.float64))

    with tempfile.TemporaryDirectory(prefix="ch_20bar_utility_") as temp:
        root = Path(temp)
        for mode in ("rolling_oos", "forward_oos"):
            items = _items(root, full, mode=mode, include_direct=False,
                include_full_d40=True, include_short_horizon=True)
            assert [i["model_id"] for i in items] == ["AO", "CG", "CH"]
            def producer(**kwargs):
                profile = kwargs["experiment_profile"]
                if profile == AO_PROFILE:
                    return ao_bundle
                if profile == CH_PROFILE:
                    return short_bundle
                raise AssertionError(profile)
            with patch.object(service, "load_daily_universal_ranker_data", side_effect=producer) as loader, \
                 patch.object(service, "load_official_breakout_candidate_keys", return_value=candidates):
                output = service.attach_paired_u40_extension(
                    items, evaluation_mode=mode, project_root=root)
            assert loader.call_count == 2
            ext = {i["model_id"]: {e["id"]: e for e in i["payload"]["comparison_extensions"]}
                   for i in output}
            for horizon, key in ((20, service.SHORT_EXTENSION_ID), (40, service.EXTENSION_ID)):
                for model in ("AO", "CG", "CH"):
                    entry = ext[model][key]
                    assert len(entry["rows"]) == 2
                    assert len(entry["year_rows"]) == 4
                    assert entry["cohort_sha256"] == ext["AO"][key]["cohort_sha256"]
                    for row in entry["rows"] + entry["year_rows"]:
                        assert np.isclose(row[f"top10_u{horizon}_r"],
                                          row[f"top10_mfe{horizon}_r"] - row[f"top10_d{horizon}_r"])
            assert ext["CH"][service.SHORT_EXTENSION_ID]["rows"][0]["top10_u20_r"] != (
                ext["CH"][service.EXTENSION_ID]["rows"][0]["top10_u40_r"])
            rendered = "\n".join(_render_model_comparison_specific_extensions(
                [{"model_id": item["model_id"], "view": {
                    "extensions": item["payload"]["comparison_extensions"]}}
                 for item in output], target="markdown"))
            assert all(token in rendered for token in
                       ("MFE20 - D20", "Top10 U20", "Top10 D20", "Top10 U40", "AO", "CG", "CH"))
            # Neither the canonical CH20 source nor the AO40 source may be
            # silently replaced with an unrelated truth contract.
            bad20 = SimpleNamespace(group_table=short_group,
                target_valid=full.target_valid, raw_target=short_mfe + 1.,
                target_manifest=short_bundle.target_manifest)
            with np.testing.assert_raises_regex(ValueError, "inconsistent with the canonical"):
                service._canonical_horizon_utility_truth(
                    bad20, source_target_id=DAILY_20BAR_FULL_HORIZON_PURE_MFE_TARGET_ID,
                    horizon_bars=20)


if __name__ == "__main__":
    case()
    three_way_case()
    full_d40_four_way_case()
    ao_cg_without_direct_u40_case()
    fixed_20bar_utility_case()
    fixed_20bar_loader_cohort_case()
    print("U40 AO/CG and historical AO/CE/CF/CG same mature truth, Forward/Rolling and fail-closed checks PASS")

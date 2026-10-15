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
           include_full_d40: bool = False, include_direct: bool = True):
    scores = bundle.group_table.iloc[:-1][list(service.PRIMARY_COLUMNS)].copy()
    profiles = (("AO", AO_PROFILE),)
    if include_direct:
        profiles += (("CE", CE_PROFILE),)
    if include_ranker:
        profiles += (("CF", CF_PROFILE),)
    if include_full_d40:
        profiles += (("CG", CG_PROFILE),)
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

if __name__ == "__main__":
    case()
    three_way_case()
    full_d40_four_way_case()
    ao_cg_without_direct_u40_case()
    print("U40 AO/CG and historical AO/CE/CF/CG same mature truth, Forward/Rolling and fail-closed checks PASS")

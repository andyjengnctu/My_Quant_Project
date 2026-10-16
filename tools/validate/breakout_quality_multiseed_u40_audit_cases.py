"""Independent synthetic invariants for one-off AO/CG/CH 4-seed U40 Audit."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.research_report_contract import validate_approved_persistent_report_contracts
from core.training_policy import get_robustness_benchmark_policy_snapshot
from filters.breakout_quality.continuous_target import StrategyAlignedContinuousTargetSpec
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from services.breakout_quality.u40_paired_comparison import _canonical_u40_truth
from tools.audit.breakout_quality import ao_cg_ch_multiseed_u40 as audit


def fixture():
    budget = StrategyAlignedContinuousTargetSpec.from_label_policy(
        DEFAULT_LABEL_POLICY,
    ).risk_budget_return
    records = []
    for date in pd.to_datetime(("2021-03-01", "2021-04-01", "2022-03-01", "2022-04-01")):
        for k in range(24):
            mfe = 0.9 + (k % 7) * 0.31
            d40 = 0.15 + ((k * 3) % 9) * 0.27
            records.append({
                "ticker": f"S{k:03d}", "date": date,
                "group_index": len(records),
                "label_eval_end_date": date + pd.Timedelta(days=50),
                "target_favorable_r": mfe,
                "target_minimum_low_return": -budget * d40,
            })
    group = pd.DataFrame(records)
    raw = group["target_favorable_r"].to_numpy(dtype=np.float64)
    bundle = SimpleNamespace(group_table=group, target_valid=np.ones(len(group), dtype=bool),
                             raw_target=raw)
    breakout = set(zip(group.loc[group.ticker.isin([f"S{k:03d}" for k in range(15)]), "ticker"],
                       group.loc[group.ticker.isin([f"S{k:03d}" for k in range(15)]), "date"]))
    truth = _canonical_u40_truth(bundle, source_target_id=audit._model_settings()["MR-13AO"].continuous_target_id)
    return truth, breakout


def scores(*, truth: pd.DataFrame, seed_index: int):
    result = {}
    for model in audit.MODELS:
        frame = truth[["ticker", "date", "group_index"]].copy()
        ticker = frame["ticker"].str[-3:].astype(int).to_numpy()
        utility = truth["u40_r"].to_numpy()
        base = ((ticker * 7 + 2 * seed_index) % 24) / 24
        utility_rank = pd.Series(utility).groupby(frame["date"], sort=False).rank(pct=True).to_numpy()
        if model == "MR-13AO":
            frame["score"] = base
        elif model == "MR-13CG":
            frame["score"] = utility_rank
        else:
            frame["score"] = 0.5 * (base + utility_rank)
        frame["fold_id"] = np.where(frame.date.dt.year == 2021, "fold_2021", "fold_2022")
        frame["model_information_cutoff"] = pd.to_datetime(np.where(
            frame.date.dt.year == 2021, "2020-12-31", "2021-12-31"
        ))
        result[model] = frame
    return result


def case():
    assert not validate_approved_persistent_report_contracts()
    benchmark = get_robustness_benchmark_policy_snapshot()
    assert len(benchmark["resolved_seeds"]) == 4
    truth, breakout = fixture()
    assert len(truth) == 96
    results = []
    for index, seed in enumerate(benchmark["resolved_seeds"]):
        frames = scores(truth=truth, seed_index=index)
        result = audit._paired_seed_result(seed=seed, truth=truth, frames=frames,
                                           candidates=breakout)
        assert result["complete_score_rows"] == result["common_mature_rows"] == 96
        assert len(result["rows"]["MR-13AO"]) == 2
        assert len(result["year_rows"]["MR-13AO"]) == 4
        results.append(result)
        # Detect partial intersection and invalid information cutoff even when scores exist.
        missing = {k: v.copy() for k, v in frames.items()}
        missing["MR-13CH"] = missing["MR-13CH"].iloc[1:].copy()
        try:
            audit._paired_seed_result(seed=seed, truth=truth, frames=missing, candidates=breakout)
        except ValueError as exc:
            assert "mismatched" in str(exc)
        else:
            raise AssertionError("Missing CH row silently shrank the common cohort")
        stale = {k: v.copy() for k, v in frames.items()}
        stale["MR-13CG"].loc[0, "model_information_cutoff"] = pd.Timestamp("2021-02-01")
        try:
            audit._paired_seed_result(seed=seed, truth=truth, frames=stale, candidates=breakout)
        except ValueError as exc:
            assert "mismatched" in str(exc)
        else:
            raise AssertionError("Mixed PIT cutoff was accepted")

    report = audit._make_report(seed_results=results, benchmark=benchmark,
                                model_manifests={int(r["seed"]): {} for r in results})
    assert report["status"] == "RESULT_AVAILABLE"
    assert len(report["overall_mean_sigma"]) == 2 * 3 * 3
    assert len(report["overall_paired_deltas"]) == 2 * 3 * 3
    for row in report["overall_paired_deltas"]:
        assert row["seed_count"] == 4
        assert abs(row["delta_mean_r"] - np.mean(row["paired_delta_by_seed_r"])) < 1e-12
        assert row["positive_seeds"] + row["negative_seeds"] + row["zero_seeds"] == 4
    preview = audit._render_markdown(report)
    assert "CH" in preview and "Breakout slice" in preview
    assert "Year × seed" in preview

    repeated = [dict(result) for result in results]
    repeated[-1] = dict(results[0])
    try:
        audit._make_report(seed_results=repeated, benchmark=benchmark, model_manifests={})
    except ValueError as exc:
        assert "benchmark seeds" in str(exc)
    else:
        raise AssertionError("Repeated seed was accepted")
    tampered = [dict(result) for result in results]
    tampered[-1] = {**results[-1], "cohort_sha256": {"Rolling OOS": "tampered"}}
    try:
        audit._make_report(seed_results=tampered, benchmark=benchmark, model_manifests={})
    except ValueError as exc:
        assert "cohort SHA" in str(exc)
    else:
        raise AssertionError("Different cohort sha was accepted")

    # Reader must be fail-closed on official READY, benchmark seed and target ID.
    model_settings = audit._model_settings()
    settings = model_settings["MR-13CH"]
    source_frames = scores(truth=truth, seed_index=0)
    good_manifest = {"seed": int(benchmark["resolved_seeds"][0]),
                     "continuous_target_id": str(settings.continuous_target_id),
                     "artifacts": {"scores": {"sha256": "fixture-hash"}}}
    def fake_paths(_settings, *, seed, rolling):
        return audit.PROJECT_ROOT / "models" / "fixture" / f"seed_{seed}", audit.PROJECT_ROOT
    with patch.object(audit, "_load_rolling_robustness_seed", return_value=({}, None, None)), patch.object(
        audit, "_model_robustness_dirs", side_effect=fake_paths), patch.object(
        audit, "_validate_score_artifacts", return_value=(source_frames["MR-13CH"].rename(
            columns={"score": "breakout_quality_score"}), good_manifest)):
        frame, provenance = audit._validated_score_for_seed(
            model_id="MR-13CH", settings=settings, seed=benchmark["resolved_seeds"][0]
        )
        assert frame["score"].equals(source_frames["MR-13CH"]["score"])
        assert provenance["source_seed"] == benchmark["resolved_seeds"][0]
        try:
            audit._validated_score_for_seed(model_id="MR-13CH", settings=settings, seed=12345)
        except ValueError as exc:
            assert "unexpected source seed" in str(exc)
        else:
            raise AssertionError("Unrelated benchmark seed accepted")
        tampered_target = {**good_manifest, "continuous_target_id": "stale-target"}
        with patch.object(audit, "_validate_score_artifacts", return_value=(
            source_frames["MR-13CH"], tampered_target
        )):
            try:
                audit._validated_score_for_seed(
                    model_id="MR-13CH", settings=settings, seed=benchmark["resolved_seeds"][0]
                )
            except ValueError as exc:
                assert "target identity" in str(exc)
            else:
                raise AssertionError("Different target identity accepted")
    with patch.object(audit, "_load_rolling_robustness_seed", return_value=(None, None, "stale")):
        try:
            audit._validated_score_for_seed(
                model_id="MR-13CH", settings=settings, seed=benchmark["resolved_seeds"][0]
            )
        except ValueError as exc:
            assert "not READY" in str(exc)
        else:
            raise AssertionError("Non-READY robustness artifact accepted")

    # End-to-end orchestration dry run: truth and scores are supplied by synthetic
    # fixtures, so this never calls a trainer or writes even a diagnostic artifact.
    by_seed = {seed: scores(truth=truth, seed_index=i)
               for i, seed in enumerate(benchmark["resolved_seeds"])}
    def loader(*, model_id, settings, seed):
        return by_seed[seed][model_id], {"source_seed": seed}
    with patch.object(audit, "_validated_score_for_seed", side_effect=loader), patch.object(
        audit, "load_daily_universal_ranker_data", return_value=SimpleNamespace(
            group_table=pd.DataFrame(), target_valid=[], raw_target=[]
        )), patch.object(audit, "_canonical_u40_truth", return_value=truth), patch.object(
        audit, "load_official_breakout_candidate_keys", return_value=breakout):
        completed = audit.run(write_report=False)
    assert len(completed["seed_results"]) == 4
    print("AO/CG/CH 4-seed common-U40 audit synthetic: PASS")


if __name__ == "__main__":
    case()

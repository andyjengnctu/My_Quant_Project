"""Read-only common-truth MFE40/D40/U40 contrast of existing continuous models.

No model fitting, label redesign, strategy replay or threshold selection.  The
canonical mature 40-bar MFE/D40 outcomes are loaded from the same Daily-Universal
Producer used by the selected reference. A pure-MFE reference may supply the
matured path components directly; previously fitted direct-U40 models are not
required to participate in the active comparison.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.breakout_quality_runtime import (
    DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    CONTINUOUS_RANKER_SAFETY_SUPERVISION_ADVERSE_TO_PEAK,
    CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
    get_continuous_ranker_training_policy,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_target import (
    StrategyAlignedContinuousTargetSpec,
    full_horizon_mae_safety_r_from_minimum_low_return,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.daily_ranker_data import (
    load_daily_universal_ranker_data,
    load_official_breakout_candidate_keys,
)
from filters.breakout_quality.ranking_score_store import validate_continuous_ranker_score_values

EXTENSION_ID = "paired_mfe_d40_direct_utility_v1"
TOP_DECILE = 0.1
MIN_RANKABLE_PER_DATE = 10
REFERENCE_EVIDENCE_FAMILY = "hs_conditional_mfe"
PRIMARY_COLUMNS = ("ticker", "date", "group_index")


def _participants(models: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...], Mapping[str, Any]] | None:
    """Find one original-Safety reference, matched U40 outcomes, and their truth Producer."""
    references, treatments, direct = [], [], []
    for item in models:
        settings = item["settings"]
        policy = get_continuous_ranker_training_policy(str(settings.training_objective))
        recipe = get_continuous_ranker_execution_recipe(str(settings.experiment_profile))
        is_conditional = REFERENCE_EVIDENCE_FAMILY in policy.report_evidence_families
        supervised_source = recipe.objective_policy.safety_supervision_source
        is_primary_scoped = (
            recipe.objective_policy.secondary_pair_scope
            == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN
        )
        if str(settings.continuous_target_id) == DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID:
            direct.append(item)
            treatments.append(item)
        elif is_conditional and is_primary_scoped:
            if supervised_source == CONTINUOUS_RANKER_SAFETY_SUPERVISION_ADVERSE_TO_PEAK:
                references.append(item)
            elif supervised_source == CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE:
                treatments.append(item)
    if not references or not treatments:
        return None
    if len(references) != 1 or any(item is references[0] for item in treatments):
        raise ValueError("U40 paired comparison requires one original-Safety Conditional-MFE reference")
    reference = references[0]
    all_items = (reference, *treatments)
    if len({str(item["model_id"]) for item in all_items}) != len(all_items):
        raise ValueError("U40 comparison contains duplicate model identity")
    if len({str(item["settings"].filter_id) for item in all_items}) != 1:
        raise ValueError("U40 paired comparison mixed incompatible filter universes")
    for item in all_items:
        contract = item.get("contract")
        if contract is None or str(contract.continuous_target_id) != str(item["settings"].continuous_target_id):
            raise ValueError("U40 paired comparison scientific target contract mismatch")
    # Historical direct-U40 controls can still supply the identical canonical
    # target. With only AO/CG active, AO supplies the original mature 40D path.
    return reference, tuple(treatments), direct[0] if direct else reference


def _read_validated_score(item: Mapping[str, Any], *, mode: str) -> pd.DataFrame:
    contract = item.get("contract")
    if contract is None or not hasattr(contract, "score_path"):
        raise ValueError("U40 paired score missing validated Research OOS contract")
    path = Path(contract.score_path)
    if mode == "forward_oos":
        expected = dict((contract.report.get("artifacts") or {}).get("oos_scores_gzip") or {})
        if not expected or build_file_manifest(path) != expected:
            raise ValueError("U40 paired Forward scores fail the approved SHA/file manifest")
    source_column = "model_score" if mode == "forward_oos" else "breakout_quality_score"
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": str})
    required = {*PRIMARY_COLUMNS, source_column}
    if mode == "forward_oos":
        required.add("target_raw_r")
    else:
        required |= {"fold_id", "model_information_cutoff"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"U40 paired source missing declared OOS columns: {missing}")
    if mode == "forward_oos":
        # Future inference-only predictions carry no mature evaluation truth.
        frame = frame.loc[pd.to_numeric(frame["target_raw_r"], errors="coerce").notna()].copy()
    if frame.empty:
        raise ValueError("U40 paired OOS scores contain no mature evaluable rows")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["group_index"] = pd.to_numeric(frame["group_index"], errors="raise").astype("int64")
    frame["score"] = pd.to_numeric(frame[source_column], errors="raise").astype("float64")
    if frame.duplicated(["ticker", "date"]).any() or frame["group_index"].duplicated().any():
        raise ValueError("U40 paired source duplicated ticker/date or group_index")
    validate_continuous_ranker_score_values(
        frame["score"].to_numpy(dtype=np.float64),
        experiment_profile=str(item["settings"].experiment_profile),
        column=source_column,
        label="U40 paired OOS score",
    )
    if mode == "rolling_oos":
        frame["fold_id"] = frame["fold_id"].astype("string")
        if frame["fold_id"].isna().any() or (frame["fold_id"].str.strip() == "").any():
            raise ValueError("U40 paired Rolling fold identity absent")
        cutoff = pd.to_datetime(frame["model_information_cutoff"], errors="raise").dt.normalize()
        if cutoff.isna().any() or bool((frame["date"] <= cutoff).any()):
            raise ValueError("U40 paired Rolling score violates model information cutoff")
        frame["model_information_cutoff"] = cutoff
    return frame[list(PRIMARY_COLUMNS) + ["score"] +
                 (["fold_id", "model_information_cutoff"] if mode == "rolling_oos" else [])].copy()


def _canonical_u40_truth(
    bundle, *, source_target_id: str = DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID,
) -> pd.DataFrame:
    """Verify mature MFE/D40/R truth against the producer's own raw target."""
    if source_target_id not in {
        DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID,
        DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    }:
        raise ValueError("U40 common truth requires a canonical mature U40 or Pure-MFE target")
    group = bundle.group_table
    required = {"ticker", "date", "group_index", "label_eval_end_date",
                "target_favorable_r", "target_minimum_low_return"}
    missing = required - set(group.columns)
    if missing:
        raise ValueError(f"U40 canonical truth missing source columns: {sorted(missing)}")
    valid = np.asarray(bundle.target_valid, dtype=bool)
    raw = np.asarray(bundle.raw_target, dtype=np.float64)
    if valid.shape != (len(group),) or raw.shape != (len(group),):
        raise ValueError("U40 canonical target/maturity shape does not match stock-day universe")
    if not valid.any():
        raise ValueError("U40 canonical truth has no mature 40D path")
    table = group.loc[valid, list(required)].copy()
    table["ticker"] = table["ticker"].astype(str)
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.normalize()
    table["label_eval_end_date"] = pd.to_datetime(table["label_eval_end_date"], errors="raise").dt.normalize()
    if table["label_eval_end_date"].isna().any() or (table["label_eval_end_date"] <= table["date"]).any():
        raise ValueError("U40 canonical truth missing or invalid 40D maturity date")
    if table.duplicated(["ticker", "date"]).any() or table["group_index"].duplicated().any():
        raise ValueError("U40 canonical truth duplicated stock-day key")
    risk_budget = StrategyAlignedContinuousTargetSpec.from_label_policy(
        DEFAULT_LABEL_POLICY
    ).risk_budget_return
    mfe = pd.to_numeric(table["target_favorable_r"], errors="raise").to_numpy(dtype=np.float64)
    min_low = pd.to_numeric(table["target_minimum_low_return"], errors="raise").to_numpy(dtype=np.float64)
    d40 = -full_horizon_mae_safety_r_from_minimum_low_return(min_low, risk_budget_return=risk_budget)
    utility = mfe - d40
    published = raw[np.flatnonzero(valid)]
    if (not np.isfinite(mfe).all() or not np.isfinite(d40).all()
            or not np.isfinite(published).all() or (d40 < 0.0).any()):
        raise ValueError("U40 canonical truth contains invalid MFE/D40/R components")
    expected_raw = (
        utility if source_target_id == DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID
        else mfe
    )
    if not np.allclose(expected_raw, published, atol=5e-5, rtol=1e-5):
        raise ValueError("U40 common truth is inconsistent with the canonical target Producer")
    table["mfe40_r"] = mfe
    table["d40_r"] = d40
    table["u40_r"] = utility
    return table[["ticker", "date", "group_index", "mfe40_r", "d40_r", "u40_r"]].copy()


def _matched_rows(*, truth: pd.DataFrame, reference: pd.DataFrame,
                  treatment: pd.DataFrame | None = None,
                  treatments: Mapping[str, pd.DataFrame] | None = None,
                  mode: str) -> pd.DataFrame:
    """Intersect all evaluated U40 arms with identical keys, folds and PIT cutoffs.

    The original single-treatment signature remains valid for historical
    evidence. Every active treatment shares ONE common cohort with AO.
    """
    if (treatment is None) == (treatments is None):
        raise ValueError("U40 matched comparison requires either one or many treatment frames")
    if treatments is not None and (not treatments or "reference" in treatments):
        raise ValueError("U40 matched comparison cannot override the reference arm")
    arms = {"reference": reference, **(
        {"direct": treatment} if treatments is None else dict(treatments)
    )}
    if "reference" not in arms or len(arms) < 2:
        raise ValueError("U40 matched comparison requires distinct score roles")
    cohort = truth.copy()
    for role, frame in arms.items():
        if role != "reference" and (not role.startswith("direct") or not role.replace("_", "").isalnum()):
            raise ValueError("U40 treatment score role must be a validated direct role")
        src = frame.rename(columns={"score": f"score_{role}",
                                    "group_index": f"group_index_{role}",
                                    "fold_id": f"fold_{role}",
                                    "model_information_cutoff": f"cutoff_{role}"})
        cohort = cohort.merge(src, on=["ticker", "date"], how="inner", validate="one_to_one", sort=False)
        if (cohort[f"group_index_{role}"] != cohort["group_index"]).any():
            raise ValueError(f"U40 common stock-day group index mismatch: {role}")
        cohort.drop(columns=[f"group_index_{role}"], inplace=True)
        if mode == "rolling_oos" and role != "reference":
            if (cohort[f"fold_{role}"] != cohort["fold_reference"]).any() or (
                cohort[f"cutoff_{role}"] != cohort["cutoff_reference"]
            ).any():
                raise ValueError("U40 common cohort has mismatched Rolling fold or information cutoff")
    if cohort.empty:
        raise ValueError("U40 paired common mature OOS cohort is empty")
    if mode == "rolling_oos":
        # Prevent a mixed fold/year cohort from hiding a validity regression.
        per_year = cohort.groupby(cohort["date"].dt.year)["fold_reference"].nunique(dropna=False)
        if (per_year != 1).any():
            raise ValueError("U40 paired Rolling year contains mixed fold identities")
    cohort.sort_values(["date", "ticker"], inplace=True, kind="mergesort")
    return cohort.reset_index(drop=True)


def _top_decile_metrics(cohort: pd.DataFrame, score_column: str) -> dict[str, Any]:
    """Identical date inclusion and Top10 slots for every model in a cohort."""
    totals = np.zeros(3, dtype=np.float64)
    selected = 0
    dates = 0
    for _, day in cohort.groupby("date", sort=True):
        if len(day) <= MIN_RANKABLE_PER_DATE:
            continue
        count = max(1, math.ceil(len(day) * TOP_DECILE))
        top = day.sort_values([score_column, "ticker"], ascending=[False, True], kind="mergesort").head(count)
        totals += top[["u40_r", "mfe40_r", "d40_r"]].to_numpy(dtype=np.float64).sum(axis=0)
        selected += len(top)
        dates += 1
    if selected == 0:
        raise ValueError("U40 paired model score has no Top10 rankable common dates")
    return {
        "top10_n": int(selected), "rankable_dates": dates,
        "top10_u40_r": float(totals[0] / selected),
        "top10_mfe40_r": float(totals[1] / selected),
        "top10_d40_r": float(totals[2] / selected),
    }


def _paired_report_rows(
    *, common: pd.DataFrame, reference_id: str, treatment_id: str | None, mode: str,
    breakout_candidate_keys: set[tuple[str, pd.Timestamp]],
    treatment_score_columns: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Report original and one/many U40 rankers on a strictly shared cohort."""
    label = "Rolling OOS" if mode == "rolling_oos" else "Forward OOS"
    membership = np.fromiter(
        ((str(t), pd.Timestamp(d).normalize()) in breakout_candidate_keys
         for t, d in zip(common["ticker"], common["date"])),
        dtype=bool, count=len(common),
    )
    compared = ({str(treatment_id): "score_direct"} if treatment_score_columns is None
                else dict(treatment_score_columns))
    if not compared or reference_id in compared:
        raise ValueError("U40 paired report requires distinct treatment models")
    arms = {str(reference_id): "score_reference", **compared}
    if len(arms) != len(compared) + 1:
        raise ValueError("U40 paired report contains duplicate arm identities")
    result = {model: {"id": EXTENSION_ID, "rows": [], "year_rows": []} for model in arms}
    for scope, subset in ((label, common), ("Breakout slice", common.loc[membership])):
        if subset.empty or not bool((subset.groupby("date").size() > MIN_RANKABLE_PER_DATE).any()):
            continue
        digest = hashlib.sha256(
            "\n".join(f"{date:%Y-%m-%d}|{ticker}|{int(group)}"
                      for ticker, date, group in zip(subset["ticker"], subset["date"], subset["group_index"]))
            .encode("utf-8")
        ).hexdigest()
        def records(part: pd.DataFrame, *, year: int | None) -> None:
            controls = _top_decile_metrics(part, "score_reference")
            for model_id, score_column in arms.items():
                metric = controls if model_id == reference_id else _top_decile_metrics(part, score_column)
                if (controls["top10_n"], controls["rankable_dates"]) != (
                    metric["top10_n"], metric["rankable_dates"]
                ):
                    raise ValueError("U40 matched arms differ in Top10 selection slots")
                row = {
                    "split": scope, "stock_days": int(len(part)),
                    "dates": int(part["date"].nunique()), **metric,
                    "delta_u40_vs_reference_r": metric["top10_u40_r"] - controls["top10_u40_r"],
                }
                if year is not None:
                    row["year"] = int(year)
                result[model_id]["year_rows" if year is not None else "rows"].append(row)
        records(subset, year=None)
        for year, frame in subset.groupby(subset["date"].dt.year, sort=True):
            if not bool((frame.groupby("date").size() > MIN_RANKABLE_PER_DATE).any()):
                continue
            records(frame, year=int(year))
        for model in result.values():
            model.setdefault("cohort_sha256", {})[scope] = digest
    return result


def attach_paired_u40_extension(
    models: Sequence[Mapping[str, Any]], *, evaluation_mode: str, project_root: Path,
) -> list[dict[str, Any]]:
    """Add post-fit same-U40 evidence to the existing Model-specific Extension."""
    output = [{**item, "payload": dict(item.get("payload") or {})} for item in models]
    if evaluation_mode not in {"forward_oos", "rolling_oos"}:
        return output
    participants = _participants(output)
    if participants is None:
        return output
    reference, treatment_models, canonical_u40_source = participants
    ref_scores = _read_validated_score(reference, mode=evaluation_mode)
    treatment_scores = {
        f"direct_{index}": _read_validated_score(item, mode=evaluation_mode)
        for index, item in enumerate(treatment_models)
    }
    # A single U40 Producer owns the shared mature MFE/D40 truth. No score,
    # classification or historical OOS outcome is allowed into model fitting.
    settings = canonical_u40_source["settings"]
    bundle = load_daily_universal_ranker_data(
        filter_id=str(settings.filter_id), model_architecture=str(settings.model_architecture),
        experiment_profile=str(settings.experiment_profile), preload_feature_bank=False,
        allow_stale_source=False, project_root=project_root,
    )
    truth = _canonical_u40_truth(
        bundle, source_target_id=str(settings.continuous_target_id),
    )
    matched = _matched_rows(truth=truth, reference=ref_scores,
                            treatments=treatment_scores, mode=evaluation_mode)
    candidates = load_official_breakout_candidate_keys(
        str(settings.filter_id), allow_stale_source=False,
    )
    entries = _paired_report_rows(
        common=matched, reference_id=str(reference["model_id"]), treatment_id=None,
        treatment_score_columns={str(item["model_id"]): f"score_direct_{index}"
                                 for index, item in enumerate(treatment_models)},
        mode=evaluation_mode, breakout_candidate_keys=candidates,
    )
    for item in output:
        extension = entries.get(str(item["model_id"]))
        if extension is None:
            continue
        existing = [dict(ext) for ext in (item["payload"].get("comparison_extensions") or [])
                    if str(ext.get("id")) != EXTENSION_ID]
        item["payload"]["comparison_extensions"] = [*existing, extension]
    return output


__all__ = ["EXTENSION_ID", "attach_paired_u40_extension"]

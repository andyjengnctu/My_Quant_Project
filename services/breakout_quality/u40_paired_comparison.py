"""Read-only common-truth MFE40/D40/U40 contrast of existing continuous models.

No model fitting, label redesign, strategy replay or threshold selection.  The
canonical mature 40-bar U40 target is loaded through the same Daily-Universal
Producer that owns MR-13CE; the original AO score is evaluated against that
truth rather than its historical MFE-centric Standard-SOP target.
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
    get_continuous_ranker_training_policy,
)
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


def _participants(models: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    """Resolve registered capability and target rather than hardcoding MR IDs."""
    references, direct = [], []
    for item in models:
        settings = item["settings"]
        policy = get_continuous_ranker_training_policy(str(settings.training_objective))
        if str(settings.continuous_target_id) == DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID:
            direct.append(item)
        if REFERENCE_EVIDENCE_FAMILY in policy.report_evidence_families:
            references.append(item)
    if not direct or not references:
        return None
    if len(direct) != 1 or len(references) != 1 or direct[0] is references[0]:
        raise ValueError("U40 paired comparison requires exactly one direct target and one Conditional-MFE reference")
    reference, treatment = references[0], direct[0]
    if str(reference["settings"].filter_id) != str(treatment["settings"].filter_id):
        raise ValueError("U40 paired comparison mixed incompatible filter universes")
    for item in (reference, treatment):
        contract = item.get("contract")
        if contract is None or str(contract.continuous_target_id) != str(item["settings"].continuous_target_id):
            raise ValueError("U40 paired comparison scientific target contract mismatch")
    return reference, treatment


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


def _canonical_u40_truth(bundle) -> pd.DataFrame:
    """Use only fully mature producer labels. Verify all three R-unit components."""
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
    if not np.allclose(utility, published, atol=5e-5, rtol=1e-5):
        raise ValueError("U40 common truth is inconsistent with the canonical direct target Producer")
    table["mfe40_r"] = mfe
    table["d40_r"] = d40
    table["u40_r"] = published
    return table[["ticker", "date", "group_index", "mfe40_r", "d40_r", "u40_r"]].copy()


def _matched_rows(*, truth: pd.DataFrame, reference: pd.DataFrame,
                  treatment: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Never let an inner join silently paper over mismatched group/fold identity."""
    cohort = truth.copy()
    for role, frame in (("reference", reference), ("direct", treatment)):
        src = frame.rename(columns={"score": f"score_{role}",
                                    "group_index": f"group_index_{role}",
                                    "fold_id": f"fold_{role}",
                                    "model_information_cutoff": f"cutoff_{role}"})
        cohort = cohort.merge(src, on=["ticker", "date"], how="inner", validate="one_to_one", sort=False)
        if (cohort[f"group_index_{role}"] != cohort["group_index"]).any():
            raise ValueError(f"U40 common stock-day group index mismatch: {role}")
        cohort.drop(columns=[f"group_index_{role}"], inplace=True)
    if cohort.empty:
        raise ValueError("U40 AO/direct paired common mature OOS cohort is empty")
    if mode == "rolling_oos":
        if (cohort["fold_reference"] != cohort["fold_direct"]).any() or (
            cohort["cutoff_reference"] != cohort["cutoff_direct"]
        ).any():
            raise ValueError("U40 common cohort has mismatched Rolling fold or information cutoff")
        # Protect against accidental overlap: one canonical fold per year.
        per_year = cohort.groupby(cohort["date"].dt.year)["fold_direct"].nunique(dropna=False)
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
    *, common: pd.DataFrame, reference_id: str, treatment_id: str, mode: str,
    breakout_candidate_keys: set[tuple[str, pd.Timestamp]],
) -> dict[str, dict[str, Any]]:
    label = "Rolling OOS" if mode == "rolling_oos" else "Forward OOS"
    membership = np.fromiter(
        ((str(t), pd.Timestamp(d).normalize()) in breakout_candidate_keys
         for t, d in zip(common["ticker"], common["date"])),
        dtype=bool, count=len(common),
    )
    result = {model: {"id": EXTENSION_ID, "rows": [], "year_rows": []}
              for model in (reference_id, treatment_id)}
    identities = (reference_id, treatment_id)
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
            treatments = _top_decile_metrics(part, "score_direct")
            if (controls["top10_n"], controls["rankable_dates"]) != (
                treatments["top10_n"], treatments["rankable_dates"]
            ):
                raise ValueError("U40 matched arms differ in Top10 selection slots")
            for role, model_id, metric in (("reference", identities[0], controls),
                                           ("direct", identities[1], treatments)):
                row = {
                    "split": scope, "stock_days": int(len(part)),
                    "dates": int(part["date"].nunique()),
                    **metric,
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
    """Add post-fit same-U40 evidence to Model-specific Extension only."""
    output = [{**item, "payload": dict(item.get("payload") or {})} for item in models]
    if evaluation_mode not in {"forward_oos", "rolling_oos"}:
        return output
    participants = _participants(output)
    if participants is None:
        return output
    reference, direct = participants
    ref_scores = _read_validated_score(reference, mode=evaluation_mode)
    direct_scores = _read_validated_score(direct, mode=evaluation_mode)
    settings = direct["settings"]
    bundle = load_daily_universal_ranker_data(
        filter_id=str(settings.filter_id), model_architecture=str(settings.model_architecture),
        experiment_profile=str(settings.experiment_profile), preload_feature_bank=False,
        allow_stale_source=False, project_root=project_root,
    )
    truth = _canonical_u40_truth(bundle)
    matched = _matched_rows(truth=truth, reference=ref_scores,
                            treatment=direct_scores, mode=evaluation_mode)
    candidates = load_official_breakout_candidate_keys(
        str(settings.filter_id), allow_stale_source=False,
    )
    entries = _paired_report_rows(
        common=matched, reference_id=str(reference["model_id"]),
        treatment_id=str(direct["model_id"]), mode=evaluation_mode,
        breakout_candidate_keys=candidates,
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

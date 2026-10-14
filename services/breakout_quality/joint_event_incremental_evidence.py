"""Post-fit paired joint-event ranking and train-frozen probability baselines.

All fitting, model selection, score production and scientific identities are
read-only.  Joint truth is the *same* canonical class-3 label already used by
the categorical model.  Train climatology is reconstructed from the immutable
per-fold split contract, never from held-out rows or OOS statistics.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.breakout_quality_registry import get_continuous_ranker_research_spec
from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_SAFETY_SUPERVISION_ADVERSE_TO_PEAK,
    CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE,
    get_continuous_ranker_training_policy,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.joint_mfe_d40 import JOINT_CLASS_COUNT, JOINT_FAVORABLE_AND_LOW_MAE_CLASS
from services.breakout_quality.point_in_time_scores import _fold_group_ids

EXTENSION_ID = "joint_mfe_d40_incremental_skill_v1"
PAIR_KEYS = ["ticker", "date", "group_index"]
PROBABILITY_BIN_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)


def _read_compared_model_scores(item: Mapping[str, Any], *, mode: str) -> pd.DataFrame:
    """Consume only a previously validated Research Compare score contract."""
    contract = item.get("contract")
    if contract is None or not hasattr(contract, "score_path"):
        raise ValueError("joint paired event requires validated Research OOS contract")
    path = Path(contract.score_path)
    if mode == "forward_oos":
        saved = dict((contract.report.get("artifacts") or {}).get("oos_scores_gzip") or {})
        if not saved or build_file_manifest(path) != saved:
            raise ValueError("joint paired Forward score SHA manifest mismatch")
    score = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": str})
    primary = "model_score" if mode == "forward_oos" else "breakout_quality_score"
    needed = set(PAIR_KEYS) | {primary}
    if mode == "forward_oos":
        needed.add("target_raw_r")
    else:
        needed |= {"fold_id", "model_information_cutoff"}
    missing = needed - set(score)
    if missing:
        raise ValueError(f"joint paired OOS score missing canonical columns: {sorted(missing)}")
    if mode == "forward_oos":
        # Forward sidecar contains decision-time inference tails with no mature labels.
        score = score.loc[pd.to_numeric(score["target_raw_r"], errors="coerce").notna()].copy()
    score["ticker"] = score["ticker"].astype(str)
    score["date"] = pd.to_datetime(score["date"], errors="raise").dt.normalize()
    score["group_index"] = pd.to_numeric(score["group_index"], errors="raise").astype("int64")
    score["model_score"] = pd.to_numeric(score[primary], errors="raise")
    if score.empty or score.duplicated(["ticker", "date"]).any() or score["group_index"].duplicated().any():
        raise ValueError("joint paired OOS score empty or has duplicate stock-day identity")
    values = score["model_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("joint paired OOS ranking probabilities must be finite [0,1]")
    if mode == "rolling_oos":
        cutoff = pd.to_datetime(score["model_information_cutoff"], errors="raise").dt.normalize()
        if score["fold_id"].isna().any() or (score["date"] <= cutoff).any():
            raise ValueError("joint paired Rolling score has invalid fold/cutoff")
        score["fold_id"] = score["fold_id"].astype(str)
    return score[PAIR_KEYS + ["model_score"] + (["fold_id"] if mode == "rolling_oos" else [])]


def _join_comparison_scores(
    joint: pd.DataFrame, reference: pd.DataFrame, full_horizon: pd.DataFrame, *, mode: str,
) -> pd.DataFrame:
    """Same stock-days and source fold for every ranking arm; no relabelling."""
    required = {*PAIR_KEYS, "joint_truth", "joint_event_probability"}
    if mode == "rolling_oos":
        required.add("fold_id")
    if not required.issubset(joint.columns):
        raise ValueError("canonical joint prediction frame missing required paired fields")
    common = joint[PAIR_KEYS + ["joint_truth", "joint_event_probability"] + (["fold_id"] if mode == "rolling_oos" else [])].copy()
    if common.duplicated(["ticker", "date"]).any():
        raise ValueError("categorical OOS paired keys are not unique")
    common.rename(columns={"fold_id": "joint_fold"}, inplace=True)
    for role, other in (("reference", reference), ("full_horizon", full_horizon)):
        other = other.rename(columns={"model_score": f"{role}_score", "group_index": f"{role}_gid", "fold_id": f"{role}_fold"})
        common = common.merge(other, on=["ticker", "date"], how="inner", validate="one_to_one", sort=False)
        if (common[f"{role}_gid"] != common["group_index"]).any():
            raise ValueError(f"joint same-truth {role} group_index does not match canonical keys")
        if mode == "rolling_oos" and (common[f"{role}_fold"] != common["joint_fold"]).any():
            raise ValueError(f"joint same-truth {role} fold differs from categorical OOS fold")
        common.drop(columns=[f"{role}_gid"] + ([f"{role}_fold"] if mode == "rolling_oos" else []), inplace=True)
    if common.empty:
        raise ValueError("joint same-truth paired OOS cohort is empty")
    values = common["joint_truth"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values, np.arange(JOINT_CLASS_COUNT)).all():
        raise ValueError("joint event common truth invalid")
    p = common["joint_event_probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("joint event probability invalid")
    return common.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)


def _cohort_hash(frame: pd.DataFrame) -> str:
    entries = (
        f"{pd.Timestamp(date):%Y-%m-%d}|{ticker}|{int(group_id)}"
        for ticker, date, group_id in zip(frame["ticker"], frame["date"], frame["group_index"])
    )
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def _ranked_top10_hits(frame: pd.DataFrame, score_column: str) -> tuple[float | None, int]:
    observed = 0
    selected = 0
    for _date, day in frame.groupby("date", sort=True):
        if len(day) < 10:
            continue
        n = max(1, int(np.ceil(len(day) * 0.1)))
        top = day.sort_values([score_column, "ticker"], ascending=[False, True], kind="mergesort").head(n)
        observed += int((top["joint_truth"] == JOINT_FAVORABLE_AND_LOW_MAE_CLASS).sum())
        selected += n
    return (None if selected == 0 else observed * 100.0 / selected), selected


def _rank_table(frame: pd.DataFrame, *, split: str, role_to_model: Mapping[str, str]) -> list[tuple[str, dict[str, Any]]]:
    baseline = float(np.mean(frame["joint_truth"].to_numpy() == JOINT_FAVORABLE_AND_LOW_MAE_CLASS)) * 100.0
    ref_hit, ref_n = _ranked_top10_hits(frame, "reference_score")
    values = []
    for role, col in (("reference", "reference_score"), ("full_horizon", "full_horizon_score"), ("joint", "joint_event_probability")):
        hit, n = _ranked_top10_hits(frame, col)
        if n != ref_n:
            raise ValueError("paired ranking top-decile sample size differs")
        values.append((role_to_model[role], {
            "split": split,
            "stock_days": int(len(frame)),
            "dates": int(frame["date"].nunique()),
            "top10_selected": n,
            "base_event_rate_pct": baseline,
            "top10_event_hit_pct": hit,
            "event_lift_pp": None if hit is None else hit - baseline,
            "paired_delta_vs_reference_pp": None if hit is None or ref_hit is None else hit - ref_hit,
        }))
    return values


def _train_class_counts(bundle, joint_labels: np.ndarray, *, mode: str, item: Mapping[str, Any], fold: str) -> tuple[np.ndarray, int]:
    contract = item["contract"]
    table = bundle.group_table
    maturity = pd.to_datetime(table["label_eval_end_date"], errors="raise").dt.normalize()
    dates = pd.to_datetime(table["date"], errors="raise").dt.normalize()
    if mode == "rolling_oos":
        manifest = dict(contract.manifest or {})
        parts = [dict(value) for value in manifest.get("folds") or [] if str(value.get("fold_id")) == fold]
        if len(parts) != 1:
            raise ValueError(f"joint train prior requires a unique canonical PIT fold: {fold}")
        part = parts[0]
        planned = dict(part.get("planned_periods") or {})
        start = pd.Timestamp(planned["score_start"]).normalize()
        end = pd.Timestamp(planned["score_end"]).normalize()
        if not start <= end or not str(fold).startswith("fold_"):
            raise ValueError("joint train prior fold period contract invalid")
        ids = _fold_group_ids(
            bundle, {"fold_id": fold, "score_start": start, "score_end": end},
            validation_months=int(manifest["inner_validation_months"]),
            train_window_months=(
                None if (manifest.get("evaluation_policy") or {}).get("train_window_months") is None
                else int(manifest["evaluation_policy"]["train_window_months"])
            ),
        )["train_ids"]
        if len(ids) != int((part.get("group_counts") or {})["inner_train"]):
            raise ValueError("joint Train-only reconstructed fold count disagrees with saved PIT manifest")
        if len(ids) and bool((maturity.iloc[ids] >= start).any()):
            raise ValueError("joint train prior includes immature labels at score start")
    else:
        split = dict((contract.report or {}).get("split_report") or {})
        counts = dict(split.get("counts") or {})
        required = ("selection_start_date", "selection_end_date", "inner_validation_start_date", "oos_start_date")
        if any(not split.get(field) for field in required):
            raise ValueError("joint Forward prior missing frozen split dates")
        select_start, select_end, validation_start, oos_start = (
            pd.Timestamp(split[field]).normalize() for field in required
        )
        if not (select_start < validation_start <= select_end < oos_start):
            raise ValueError("joint Forward train-period chronology invalid")
        valid = np.asarray(bundle.target_valid, dtype=bool)
        mask = (valid & (dates >= select_start).to_numpy(dtype=bool)
                & (dates < validation_start).to_numpy(dtype=bool)
                & (maturity < validation_start).to_numpy(dtype=bool)
                & (maturity < oos_start).to_numpy(dtype=bool))
        ids = np.flatnonzero(mask)
        if len(ids) != int(counts["inner_train"]):
            raise ValueError("joint Train-only Forward counts disagree with frozen model split")
    if len(ids) < 1 or not np.isfinite(joint_labels[ids]).all():
        raise ValueError("joint train prior has missing or non-mature targets")
    counts = np.bincount(joint_labels[ids].astype(np.int64), minlength=JOINT_CLASS_COUNT)
    if counts.shape != (JOINT_CLASS_COUNT,) or not (counts > 0).all():
        raise ValueError("joint train prior missing one or more categorical classes")
    return counts, int(len(ids))


def _prior_skill_rows(
    frame: pd.DataFrame, *,
    split: str, fold: str, prior_counts: np.ndarray, train_n: int,
    categorical_columns: Sequence[str],
) -> dict[str, Any]:
    classes = frame["joint_truth"].to_numpy(dtype=np.int64)
    probs = frame[list(categorical_columns)].to_numpy(dtype=np.float64)
    if not len(classes) or not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=2e-5, rtol=0):
        raise ValueError("joint proper scoring expected finite categorical predictions")
    prior = np.asarray(prior_counts, dtype=np.float64) / float(train_n)
    baseline = prior[JOINT_FAVORABLE_AND_LOW_MAE_CLASS]
    truth = (classes == JOINT_FAVORABLE_AND_LOW_MAE_CLASS).astype(np.float64)
    prob = probs[:, JOINT_FAVORABLE_AND_LOW_MAE_CLASS]
    ce = float(-np.log(np.clip(probs[np.arange(len(classes)), classes], np.finfo(float).tiny, 1)).mean())
    prior_ce = float(-np.log(prior[classes]).mean())
    brier = float(np.mean((prob - truth) ** 2))
    prior_brier = float(np.mean((baseline - truth) ** 2))
    if prior_brier <= 0:
        raise ValueError("joint baseline Brier degenerate")
    return {
        "split": split, "fold": fold, "stock_days": int(len(frame)),
        "train_stock_days": train_n,
        "train_event_pct": 100.0 * baseline,
        "oos_event_pct": 100.0 * truth.mean(),
        "joint_ce": ce, "train_only_ce": prior_ce,
        "joint_brier": brier, "train_only_brier": prior_brier,
        "brier_skill_pct": 100.0 * (1.0 - brier / prior_brier),
    }


def _calibration_rows(frame: pd.DataFrame, *, split: str) -> list[dict[str, Any]]:
    prob = frame["joint_event_probability"].to_numpy(dtype=np.float64)
    truth = (frame["joint_truth"].to_numpy(dtype=np.int64) == JOINT_FAVORABLE_AND_LOW_MAE_CLASS)
    bins = np.searchsorted(PROBABILITY_BIN_EDGES, prob, side="right") - 1
    bins = np.clip(bins, 0, len(PROBABILITY_BIN_EDGES) - 2)
    rows = []
    for i in range(len(PROBABILITY_BIN_EDGES) - 1):
        mask = bins == i
        if not mask.any():
            continue
        expected = 100.0 * float(prob[mask].mean())
        actual = 100.0 * float(truth[mask].mean())
        rows.append({
            "split": split, "prob_bin": f"{int(100*PROBABILITY_BIN_EDGES[i])}-{int(100*PROBABILITY_BIN_EDGES[i+1])}%",
            "stock_days": int(mask.sum()),
            "mean_predicted_pct": expected,
            "actual_event_pct": actual,
            "calibration_gap_pp": expected - actual,
        })
    return rows


def build_joint_event_incremental_extensions(
    *, models: Sequence[Mapping[str, Any]], categorical_item: Mapping[str, Any],
    scored: pd.DataFrame, bundle, candidate_keys: set[tuple[str, pd.Timestamp]],
    mode: str, categorical_columns: Sequence[str], full_truth: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Return extension evidence keyed by config-selected scientific role, not MR IDs."""
    if mode not in {"forward_oos", "rolling_oos"}:
        return {}
    controls: list[Mapping[str, Any]] = []
    treatments: list[Mapping[str, Any]] = []
    for item in models:
        settings = item["settings"]
        policy = get_continuous_ranker_training_policy(str(settings.training_objective))
        if policy.categorical_class_count is not None:
            continue
        spec = get_continuous_ranker_research_spec(str(settings.experiment_profile))
        if spec.safety_supervision_source == CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE:
            treatments.append(item)
    if not treatments:
        return {}  # A single categorical model can still report its existing proper-score extension.
    if len(treatments) != 1:
        raise ValueError("joint same-event comparison expects one declared full-horizon Safety treatment")
    treatment = treatments[0]
    for item in models:
        settings = item["settings"]
        spec = get_continuous_ranker_research_spec(str(settings.experiment_profile))
        if (str(settings.training_objective) == str(treatment["settings"].training_objective)
            and spec.safety_supervision_source == CONTINUOUS_RANKER_SAFETY_SUPERVISION_ADVERSE_TO_PEAK):
            controls.append(item)
    if len(controls) != 1:
        raise ValueError("joint same-event comparison requires exactly one original-Safety paired control")
    reference = controls[0]
    for item in (reference, treatment):
        if (str(item["settings"].filter_id) != str(categorical_item["settings"].filter_id)
            or str(item["contract"].continuous_target_id) != str(categorical_item["contract"].continuous_target_id)):
            raise ValueError("joint paired event mixed incompatible dataset/target scientific contracts")
    ref = _read_compared_model_scores(reference, mode=mode)
    d40 = _read_compared_model_scores(treatment, mode=mode)
    primary = scored[PAIR_KEYS + ["joint_truth", categorical_columns[JOINT_FAVORABLE_AND_LOW_MAE_CLASS]]
                     + (["fold_id"] if mode == "rolling_oos" else [])].copy()
    primary.rename(columns={categorical_columns[JOINT_FAVORABLE_AND_LOW_MAE_CLASS]: "joint_event_probability"}, inplace=True)
    common = _join_comparison_scores(primary, ref, d40, mode=mode)
    cohort_digest = _cohort_hash(common)
    candidate_mask = pd.Series(list(zip(common["ticker"], common["date"]))).isin(candidate_keys).to_numpy(dtype=bool)
    split_label = "Rolling OOS" if mode == "rolling_oos" else "Forward OOS"
    cohorts = [(split_label, common), ("Breakout slice", common.loc[candidate_mask].copy())]
    model_roles = {
        "reference": str(reference["model_id"]),
        "full_horizon": str(treatment["model_id"]),
        "joint": str(categorical_item["model_id"]),
    }
    result: dict[str, dict[str, Any]] = {value: {"id": EXTENSION_ID, "paired_rows": [], "cohort_sha256": cohort_digest,
                                                     "cohort_semantic": "all canonical mature class-3 truth; three-model ticker/date/group_index intersection"}
                                         for value in model_roles.values()}
    for split, frame in cohorts:
        if frame.empty:
            continue
        for model_id, row in _rank_table(frame, split=split, role_to_model=model_roles):
            result[model_id]["paired_rows"].append(row)
    joint_extension = result[model_roles["joint"]]
    joint_extension["fold_skill_rows"] = []
    joint_extension["fold_rank_rows"] = []
    joint_extension["calibration_rows"] = []
    joint_extension["fold_calibration_detail"] = []
    joint_extension["prior_semantic"] = "canonical inner_train rows: label_eval_end before validation_start; no OOS or validation labels"
    joint_extension["calibration_semantic"] = "fixed 10-percentage-point probability bins; display only; no fitting"
    cd_full = scored[PAIR_KEYS + ["joint_truth", *categorical_columns] + (["fold_id"] if mode == "rolling_oos" else [])].copy()
    cd_full["date"] = pd.to_datetime(cd_full["date"], errors="raise").dt.normalize()
    # Baseline and calibration evaluate the original categorical OOS universe,
    # independently of the matched three-model ranking cohort.
    if cd_full.duplicated(["ticker", "date"]).any():
        raise ValueError("joint prior source OOS rows are duplicated")
    full_calibration = [(split_label, cd_full),
                        ("Breakout slice", cd_full.loc[pd.Series(list(zip(cd_full["ticker"], cd_full["date"]))).isin(candidate_keys).to_numpy(dtype=bool)])]
    for split, frame in full_calibration:
        if frame.empty:
            continue
        copy = frame.copy()
        copy["joint_event_probability"] = copy[categorical_columns[JOINT_FAVORABLE_AND_LOW_MAE_CLASS]]
        joint_extension["calibration_rows"].extend(_calibration_rows(copy, split=split))
    # Reuse the original canonical whole-universe target, never rebuild a
    # percentile truth on a post-fit intersection or breakout slice.
    labels = np.asarray(full_truth, dtype=np.float64)
    if labels.shape != (len(bundle.group_table),):
        raise ValueError("joint train prior entire canonical truth has invalid length")
    fold_ids = (sorted(cd_full["fold_id"].astype(str).unique()) if mode == "rolling_oos" else ["forward"])
    for fold in fold_ids:
        train_counts, train_n = _train_class_counts(bundle, labels, mode=mode, item=categorical_item, fold=fold)
        for split, full in cohorts:
            native = full_calibration[0][1] if split == split_label else full_calibration[1][1]
            evaluation = native.loc[native["fold_id"].astype(str) == fold] if mode == "rolling_oos" else native
            if not evaluation.empty:
                row = _prior_skill_rows(evaluation, split=split, fold=fold, prior_counts=train_counts,
                                        train_n=train_n, categorical_columns=categorical_columns)
                calibration = evaluation.copy()
                calibration["joint_event_probability"] = calibration[categorical_columns[JOINT_FAVORABLE_AND_LOW_MAE_CLASS]]
                bins = _calibration_rows(calibration, split=split)
                if sum(v["stock_days"] for v in bins) != len(evaluation):
                    raise ValueError("joint fixed-bin calibration omits OOS rows")
                row["calibration_ece_pp"] = sum(
                    v["stock_days"] * abs(v["calibration_gap_pp"]) for v in bins
                ) / len(evaluation)
                joint_extension["fold_skill_rows"].append(row)
                joint_extension["fold_calibration_detail"].append({
                    "split": split, "fold": fold, "bins": bins,
                })
            selected = full.loc[full["joint_fold"] == fold] if mode == "rolling_oos" else full
            if selected.empty:
                continue
            ref_hit, _ = _ranked_top10_hits(selected, "reference_score")
            d40_hit, _ = _ranked_top10_hits(selected, "full_horizon_score")
            cd_hit, _ = _ranked_top10_hits(selected, "joint_event_probability")
            joint_extension["fold_rank_rows"].append({
                "split": split, "fold": fold, "stock_days": int(len(selected)),
                "reference_top10_pct": ref_hit,
                "full_horizon_top10_pct": d40_hit,
                "joint_top10_pct": cd_hit,
                "joint_minus_reference_pp": None if cd_hit is None or ref_hit is None else cd_hit - ref_hit,
            })
    return result

"""Independent regression for signed continuous-R vs bounded probability scores.

A signed primary score is valid for the registered margin transformation only.
Historical probability models and optional sidecar probabilities remain [0,1].
This is not a formal-suite runner and does not create fitted/OOS artifacts.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json

import numpy as np
import pandas as pd

from core.breakout_quality_runtime import get_continuous_ranker_score_output_columns
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.ranking_score_store import (
    validate_continuous_ranker_score_values,
    load_continuous_ranker_oos_score_table_from_path,
    load_selection_point_in_time_score_table_from_path,
    lookup_continuous_ranker_oos_candidate_score,
    lookup_selection_point_in_time_candidate_score,
)
from services.breakout_quality import point_in_time_audit as pit_audit
from services.breakout_quality.point_in_time_scores import (
    _validate_score_frame,
    _validate_validation_score_frame,
    POINT_IN_TIME_SCHEMA_VERSION,
)

MARGIN_PROFILE = "daily_universal_full_horizon_mfe_minus_d40_huber"
HISTORICAL_MARGIN_PROFILE = "daily_universal_no_time_r_huber"
PROBABILITY_PROFILE = "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
CATEGORICAL_PROFILE = "daily_universal_joint_mfe_d40_quadrant_cross_entropy"
FOLD_ID = "fold_20210101_20211231"
CUTOFF = "2020-12-31"


def _expect_rejected(fn, text: str = "") -> None:
    try:
        fn()
    except ValueError as exc:
        assert not text or text in str(exc), (text, str(exc))
    else:
        raise AssertionError(f"Out-of-contract input accepted: {text}")


def _score_frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [f"T{idx:04d}" for idx in range(len(scores))],
        "date": ["2021-01-04"] * len(scores),
        "group_index": list(range(len(scores))),
        "breakout_quality_score": scores,
        "fold_id": [FOLD_ID] * len(scores),
        "model_information_cutoff": [CUTOFF] * len(scores),
    })


def _fold_contract(profile: str) -> dict:
    return {
        "fold_id": FOLD_ID, "model_information_cutoff": CUTOFF,
        "planned_periods": {"score_start": "2021-01-01", "score_end": "2021-12-31"},
        "experiment_profile": profile,
    }


def _write_audit_fixture(folder: Path, scores: list[float], *, profile: str) -> tuple[SimpleNamespace, dict]:
    score_path = folder / "selection_point_in_time_scores.csv"
    coverage_path = folder / "selection_point_in_time_coverage.csv"
    manifest_path = folder / "selection_point_in_time_manifest.json"
    fold_dir = folder / "folds" / FOLD_ID
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "scores.csv").write_text("canonical fold score fixture", encoding="utf-8")
    (fold_dir / "model.pt").write_bytes(b"synthetic checkpoint only")
    (fold_dir / "manifest.json").write_text('{"synthetic":true}', encoding="utf-8")
    _score_frame(scores).to_csv(score_path, index=False, encoding="utf-8-sig")
    coverage_path.write_text("fixture\n", encoding="utf-8")
    fold_record = {
        "fold_id": FOLD_ID,
        "model_information_cutoff": CUTOFF,
        "planned_periods": {"score_start": "2021-01-01", "score_end": "2021-12-31"},
        "group_counts": {"score": len(scores)},
        "fold_manifest": build_file_manifest(fold_dir / "manifest.json"),
        "artifacts": {
            "checkpoint": build_file_manifest(fold_dir / "model.pt"),
            "scores": build_file_manifest(fold_dir / "scores.csv"),
        },
    }
    manifest = {
        "schema_version": POINT_IN_TIME_SCHEMA_VERSION,
        "status": "BUILT",
        "score_column": "breakout_quality_score",
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1" if profile == MARGIN_PROFILE else "inception_time_shared_safety_mfe_v1",
        "experiment_profile": profile,
        "lookahead_contract": {
            "every_score_uses_model_not_trained_on_scored_event": True,
            "training_requires_label_eval_end_before_score_start": True,
            "oos_rows_or_target_statistics_used_for_training_or_epoch_selection": False,
            "future_target_in_score_table": False,
        },
        "runtime_eligibility": {"per_score_fold_information_cutoff_required": True},
        "artifacts": {
            "scores": build_file_manifest(score_path),
            "coverage": build_file_manifest(coverage_path),
        },
        "score_columns": get_continuous_ranker_score_output_columns(
            "daily_raw_r_regression" if profile == MARGIN_PROFILE
            else "daily_shared_safety_hs_conditional_mfe_pairwise_ranking"
        ),
        "score_period": {"start": "2021-01-01", "end": "2021-12-31"},
        "fold_count": 1,
        "folds": [fold_record],
        "coverage": {
            "scored_group_count": len(scores),
            "duplicate_group_count": 0,
            "missing_group_count": 0,
            "extra_group_count": 0,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(
        filter_id="breakout_quality_v1",
        model_architecture=manifest["model_architecture"],
        experiment_profile=profile, point_in_time_dir_override=None,
    )
    paths = {"score": score_path, "manifest": manifest_path, "coverage": coverage_path}
    return args, paths


def case() -> None:
    signed = np.array([-3.25, -0.13, 0.0, 0.95, 4.70], dtype=np.float64)
    for profile in (MARGIN_PROFILE, HISTORICAL_MARGIN_PROFILE):
        validate_continuous_ranker_score_values(
            signed, experiment_profile=profile, column="breakout_quality_score", label="signed"
        )
        validate_continuous_ranker_score_values(
            signed, experiment_profile=profile, column="model_score", label="signed"
        )
        _expect_rejected(lambda: validate_continuous_ranker_score_values(
            [-0.4], experiment_profile=profile, column="conditional_safety_score", label="sidecar"
        ), "[0,1]")
    for profile in (PROBABILITY_PROFILE, CATEGORICAL_PROFILE, None):
        _expect_rejected(lambda pro=profile: validate_continuous_ranker_score_values(
            signed, experiment_profile=pro, column="breakout_quality_score", label="probability"
        ), "[0,1]")
    for nonfinite in (float("nan"), float("inf"), -float("inf")):
        for profile in (MARGIN_PROFILE, PROBABILITY_PROFILE):
            _expect_rejected(lambda v=nonfinite, pro=profile: validate_continuous_ranker_score_values(
                [v], experiment_profile=pro, column="breakout_quality_score", label="value"
            ), "非有限")

    scores = signed.tolist()
    frame = _score_frame(scores)
    validated = _validate_score_frame(frame, fold_contract=_fold_contract(MARGIN_PROFILE))
    assert validated["breakout_quality_score"].to_list() == scores
    _validate_validation_score_frame(frame, ids=np.arange(len(scores)), fold_id=FOLD_ID)
    _expect_rejected(lambda: _validate_score_frame(frame, fold_contract=_fold_contract(PROBABILITY_PROFILE)), "[0,1]")
    unknown = _fold_contract(MARGIN_PROFILE)
    unknown.pop("experiment_profile")
    _expect_rejected(lambda: _validate_score_frame(frame, fold_contract=unknown), "[0,1]")
    broken = frame.copy()
    broken.loc[0, "breakout_quality_score"] = np.nan
    _expect_rejected(lambda: _validate_score_frame(broken, fold_contract=_fold_contract(MARGIN_PROFILE)), "非有限")
    broken = frame.copy()
    broken["conditional_safety_score"] = 0.4
    broken.loc[0, "conditional_safety_score"] = -0.4
    _expect_rejected(lambda: _validate_score_frame(broken, fold_contract=_fold_contract(MARGIN_PROFILE)), "[0,1]")

    with TemporaryDirectory(prefix="u40_score_domain_") as tmp:
        root = Path(tmp)
        selection = root / "scores.csv"
        manifest = root / "manifest.json"
        frame.to_csv(selection, index=False, encoding="utf-8-sig")
        manifest.write_text(json.dumps({
            "experiment_profile": MARGIN_PROFILE,
            "score_period": {"start": "2021-01-01", "end": "2021-12-31"},
            "coverage": {"scored_group_count": len(scores)},
        }), encoding="utf-8")
        loaded = load_selection_point_in_time_score_table_from_path(
            str(selection), manifest_path=str(manifest), experiment_profile=MARGIN_PROFILE
        )
        assert loaded.loc[("T0000", "2021-01-04"), "breakout_quality_score"] == scores[0]
        assert loaded.loc[("T0004", "2021-01-04"), "breakout_quality_score"] == scores[4]
        looked_up = lookup_selection_point_in_time_candidate_score(
            project_root=str(root), filter_id="breakout_quality_v1",
            model_architecture="inception_time_v1", experiment_profile=MARGIN_PROFILE,
            ticker="T0000", signal_date="2021-01-04",
            score_path_override=str(selection), manifest_path_override=str(manifest),
        )
        assert looked_up["available"] and looked_up["score"] == scores[0]
        _expect_rejected(lambda: lookup_selection_point_in_time_candidate_score(
            project_root=str(root), filter_id="breakout_quality_v1",
            model_architecture="inception_time_v1", experiment_profile=PROBABILITY_PROFILE,
            ticker="T0000", signal_date="2021-01-04",
            score_path_override=str(selection), manifest_path_override=str(manifest),
        ), "profile不一致")
        bare = load_selection_point_in_time_score_table_from_path(
            str(selection), experiment_profile=MARGIN_PROFILE,
        )
        assert len(bare) == len(scores)
        _expect_rejected(lambda: load_selection_point_in_time_score_table_from_path(str(selection)), "[0,1]")

        forward = root / "forward.csv"
        forward_frame = frame[["ticker", "date", "group_index", "breakout_quality_score"]].rename(
            columns={"breakout_quality_score": "model_score"}
        )
        forward_frame.to_csv(forward, index=False, encoding="utf-8-sig")
        fwd = load_continuous_ranker_oos_score_table_from_path(str(forward), MARGIN_PROFILE)
        assert len(fwd) == len(scores)
        found = lookup_continuous_ranker_oos_candidate_score(
            project_root=str(root), filter_id="breakout_quality_v1",
            model_architecture="inception_time_v1", experiment_profile=MARGIN_PROFILE,
            ticker="T0004", signal_date="2021-01-04", score_path_override=str(forward),
        )
        assert found["available"] and found["score"] == scores[4]
        _expect_rejected(lambda: load_continuous_ranker_oos_score_table_from_path(
            str(forward), PROBABILITY_PROFILE
        ), "[0,1]")
        broken_forward = forward_frame.copy()
        broken_forward["conditional_safety_score"] = -0.15
        broken_forward.to_csv(forward, index=False, encoding="utf-8-sig")
        load_continuous_ranker_oos_score_table_from_path.cache_clear()
        _expect_rejected(lambda: load_continuous_ranker_oos_score_table_from_path(
            str(forward), MARGIN_PROFILE
        ), "0~1")

    with TemporaryDirectory(prefix="u40_audit_domain_") as tmp:
        folder = Path(tmp)
        args, paths = _write_audit_fixture(folder, scores, profile=MARGIN_PROFILE)
        with patch.object(pit_audit, "_pit_artifact_paths", return_value=paths), patch.object(
            pit_audit, "resolve_filter_point_in_time_fold_dir", return_value=folder / "folds" / FOLD_ID
        ):
            checked, check_manifest = pit_audit._validate_score_artifacts(args)
        assert len(checked) == len(scores) and check_manifest["fold_count"] == 1
        corrupted = _score_frame([float("inf"), *scores[1:]])
        corrupted.to_csv(paths["score"], index=False, encoding="utf-8-sig")
        written_manifest = json.loads(paths["manifest"].read_text())
        written_manifest["artifacts"]["scores"] = build_file_manifest(paths["score"])
        paths["manifest"].write_text(json.dumps(written_manifest), encoding="utf-8")
        with patch.object(pit_audit, "_pit_artifact_paths", return_value=paths), patch.object(
            pit_audit, "resolve_filter_point_in_time_fold_dir", return_value=folder / "folds" / FOLD_ID
        ):
            _expect_rejected(lambda: pit_audit._validate_score_artifacts(args), "非有限")



if __name__ == "__main__":
    case()
    print("Signed PIT/Forward producer-reader-audit domains: PASS")

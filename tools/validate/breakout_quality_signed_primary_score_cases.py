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
from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from filters.breakout_quality.ranker_sample_contract import build_score_eligibility_contract
from filters.breakout_quality.artifacts import build_file_manifest
from core.file_integrity import compute_file_sha256
from filters.breakout_quality.ranking_score_store import (
    validate_continuous_ranker_score_values,
    load_continuous_ranker_oos_score_table_from_path,
    load_selection_point_in_time_score_table_from_path,
    load_selection_point_in_time_score_source_contract,
    selection_pit_lookahead_contract,
    validate_selection_pit_lookahead_contract,
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
    (fold_dir / "validation_scores.csv").write_text(
        "group_index,date,breakout_quality_score,fold_id\n"
        f"0,2020-12-30,0.5,{FOLD_ID}\n", encoding="utf-8",
    )
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
            "validation_scores": build_file_manifest(fold_dir / "validation_scores.csv"),
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
        experiment_profile=profile, point_in_time_dir_override=str(folder),
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
        owner_profile = get_breakout_quality_experiment_profile(MARGIN_PROFILE)
        persisted = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        persisted.update({
            "training_sample_scope": str(owner_profile.training_sample_scope),
            "seed": 42,
            "source_continuous_target": {"target_contract": {"synthetic": True}},
            "lookahead_contract": selection_pit_lookahead_contract(),
        })
        paths["manifest"].write_text(json.dumps(persisted), encoding="utf-8")

        def verify_owned_score_source(*, allow_stale_source=False):
            return load_selection_point_in_time_score_source_contract(
                folder, args.filter_id, args.model_architecture, args.experiment_profile,
                point_in_time_dir_override=folder,
                allow_stale_source=allow_stale_source,
            )

        # A daily-model PIT SCORE without canonical eligibility is invalid
        # even if its Audit-specific fold and checksum fields all look sound.
        _expect_rejected(verify_owned_score_source, "eligibility contract")
        persisted["score_eligibility_contract"] = build_score_eligibility_contract(owner_profile)
        paths["manifest"].write_text(json.dumps(persisted), encoding="utf-8")

        # The score-only loader owns the four declaration semantics. A valid
        # historical-style producer manifest may retain unrelated fields, but
        # omissions, opposite values and truthy string/integer substitutes
        # must not reach either PIT Audit or strategy ranking as legal SCORE.
        assert verify_owned_score_source(allow_stale_source=True).seed == 42
        expectations = selection_pit_lookahead_contract()
        validate_selection_pit_lookahead_contract({
            "lookahead_contract": {**expectations, "legacy_optional_field": "kept"},
        })
        for field, correct in expectations.items():
            for value in ("__omitted__", not correct, str(correct).lower(), int(correct), None):
                broken = dict(expectations)
                if value == "__omitted__":
                    del broken[field]
                else:
                    broken[field] = value
                persisted["lookahead_contract"] = broken
                paths["manifest"].write_text(json.dumps(persisted), encoding="utf-8")
                _expect_rejected(
                    lambda: verify_owned_score_source(allow_stale_source=True),
                    "Selection PIT lookahead contract",
                )
        persisted["lookahead_contract"] = expectations
        paths["manifest"].write_text(json.dumps(persisted), encoding="utf-8")
        assert verify_owned_score_source(allow_stale_source=True).seed == 42
        with patch(
            "filters.breakout_quality.ranking_score_store.validate_current_pit_score_source_lineage",
            side_effect=ValueError("synthetic stale publication"),
        ):
            _expect_rejected(verify_owned_score_source, "synthetic stale publication")
            assert verify_owned_score_source(allow_stale_source=True).seed == 42

        # This fixture isolates Audit-specific signed-score/fold checks from the
        # canonical SCORE loader, whose full upstream lineage is tested by its
        # own producer-reader synthetic contracts.
        def verified_fixture_source(*_args, **_kwargs):
            return SimpleNamespace(manifest=json.loads(paths["manifest"].read_text(encoding="utf-8")))

        with patch.object(pit_audit, "_pit_artifact_paths", return_value=paths), patch.object(
            pit_audit, "load_selection_point_in_time_score_source_contract",
            side_effect=verified_fixture_source,
        ) as canonical:
            checked, check_manifest = pit_audit._validate_score_artifacts(args)
            original_manifest_sha256 = compute_file_sha256(paths["manifest"])
            assert canonical.call_count == 1
            pit_audit._assert_audit_source_revision_unchanged(
                    args, check_manifest, expected_manifest_sha256=original_manifest_sha256,
                )
            # Same logical JSON is not the same physical manifest evidence.
            persisted_manifest_bytes = paths["manifest"].read_bytes()
            paths["manifest"].write_text(
                json.dumps(check_manifest, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            assert json.loads(paths["manifest"].read_text()) == check_manifest
            try:
                pit_audit._assert_audit_source_revision_unchanged(
                    args, check_manifest, expected_manifest_sha256=original_manifest_sha256,
                )
            except RuntimeError as exc:
                assert "manifest" in str(exc) and "SHA256" in str(exc), str(exc)
            else:
                raise AssertionError("Audit accepted re-serialized manifest bytes")
            paths["manifest"].write_bytes(persisted_manifest_bytes)
            # The aggregate manifest remains byte-for-byte unchanged when a
            # Fold FIT/score/validation sidecar is replaced.  The final audit
            # guard must reject all of them, rather than publish old metrics
            # under a valid-looking aggregate revision.
            fold_dir = folder / "folds" / FOLD_ID
            fold_artifacts = {
                "fold_manifest": fold_dir / "manifest.json",
                "checkpoint": fold_dir / "model.pt",
                "scores": fold_dir / "scores.csv",
                "validation_scores": fold_dir / "validation_scores.csv",
            }
            unchanged_aggregate = paths["manifest"].read_bytes()
            for field, artifact in fold_artifacts.items():
                original = artifact.read_bytes()
                artifact.write_bytes(original + b"\nchanged during audit")
                try:
                    pit_audit._assert_audit_source_revision_unchanged(
                    args, check_manifest, expected_manifest_sha256=original_manifest_sha256,
                )
                except RuntimeError as exc:
                    assert field in str(exc) and "audit期間更新" in str(exc), str(exc)
                else:
                    raise AssertionError(f"Audit accepted replaced Fold {field}")
                artifact.write_bytes(original)
                pit_audit._assert_audit_source_revision_unchanged(
                    args, check_manifest, expected_manifest_sha256=original_manifest_sha256,
                )
            artifact = fold_artifacts["checkpoint"]
            original = artifact.read_bytes()
            artifact.unlink()
            try:
                pit_audit._assert_audit_source_revision_unchanged(
                    args, check_manifest, expected_manifest_sha256=original_manifest_sha256,
                )
            except RuntimeError as exc:
                assert "checkpoint" in str(exc) and "遺失或不可讀" in str(exc), str(exc)
            else:
                raise AssertionError("Audit accepted a missing Fold checkpoint")
            artifact.write_bytes(original)
            assert paths["manifest"].read_bytes() == unchanged_aggregate
            pit_audit._assert_audit_source_revision_unchanged(
                    args, check_manifest, expected_manifest_sha256=original_manifest_sha256,
                )
            for dangerous_fold in ("..", "a/../b", r"a\b"):
                _expect_rejected(
                    lambda key=dangerous_fold: pit_audit._pit_fold_integrity_paths(args, key),
                    "fold_id",
                )
            # Audit work also consumes a continuous target. If that target
            # changes after scoring but before publication, the final guard
            # must reject it rather than bind earlier metrics to newer bytes.
            target_file = folder / "synthetic_target_manifest.json"
            target_file.write_text('{"revision":"a"}', encoding="utf-8")
            original_target_record = build_file_manifest(target_file)
            with patch.object(
                pit_audit, "resolve_continuous_target_dir", return_value=folder,
            ), patch.object(
                pit_audit, "TARGET_MANIFEST_FILENAME", target_file.name,
            ):
                pit_audit._assert_audit_source_revision_unchanged(
                    args, check_manifest,
                    expected_manifest_sha256=original_manifest_sha256,
                    expected_target_manifest_record=original_target_record,
                )
                target_file.write_text('{"revision":"b"}', encoding="utf-8")
                try:
                    pit_audit._assert_audit_source_revision_unchanged(
                        args, check_manifest,
                        expected_manifest_sha256=original_manifest_sha256,
                        expected_target_manifest_record=original_target_record,
                    )
                except RuntimeError as exc:
                    assert "Continuous Target manifest在audit期間更新" in str(exc), str(exc)
                else:
                    raise AssertionError("Audit accepted replaced continuous target")
                target_file.write_text('{"revision":"a"}', encoding="utf-8")
                pit_audit._assert_audit_source_revision_unchanged(
                    args, check_manifest,
                    expected_manifest_sha256=original_manifest_sha256,
                    expected_target_manifest_record=original_target_record,
                )
            with patch.object(
                pit_audit, "load_selection_point_in_time_score_source_contract",
                side_effect=ValueError("canonical SCORE eligibility rejected"),
            ):
                _expect_rejected(
                    lambda: pit_audit._validate_score_artifacts(args),
                    "canonical SCORE eligibility rejected",
                )
        assert len(checked) == len(scores) and check_manifest["fold_count"] == 1
        corrupted = _score_frame([float("inf"), *scores[1:]])
        corrupted.to_csv(paths["score"], index=False, encoding="utf-8-sig")
        written_manifest = json.loads(paths["manifest"].read_text())
        written_manifest["artifacts"]["scores"] = build_file_manifest(paths["score"])
        paths["manifest"].write_text(json.dumps(written_manifest), encoding="utf-8")
        with patch.object(pit_audit, "_pit_artifact_paths", return_value=paths), patch.object(
            pit_audit, "load_selection_point_in_time_score_source_contract",
            side_effect=verified_fixture_source,
        ):
            _expect_rejected(lambda: pit_audit._validate_score_artifacts(args), "非有限")
            try:
                pit_audit._assert_audit_source_revision_unchanged(
                    args, check_manifest, expected_manifest_sha256=original_manifest_sha256,
                )
            except RuntimeError as exc:
                assert "audit期間更新" in str(exc)
            else:
                raise AssertionError("Audit accepted changed SCORE as the original revision")



if __name__ == "__main__":
    case()
    print("Signed PIT/Forward producer-reader-audit domains: PASS")

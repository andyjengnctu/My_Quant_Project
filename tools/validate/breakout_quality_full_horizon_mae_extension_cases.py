"""Independent capability-focused synthetic contract checks (not formal suite)."""
from __future__ import annotations

from dataclasses import replace
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.research_report_contract import (
    MODEL_COMPARISON_ONLY_EXTENSION_SCHEMAS,
    comparison_extension_contract, extension_contract,
    independently_frozen_model_extension_contract_fingerprints,
    persistent_report_contract_fingerprints,
    validate_approved_persistent_report_contracts,
)
from services.breakout_quality.full_horizon_mae_comparison import (
    build_paired_full_horizon_mae_rows, attach_full_horizon_mae_common_truth_extension,
)
from services.research.breakout_quality_application import _render_model_comparison_specific_extensions


def case() -> None:
    assert validate_approved_persistent_report_contracts() == ()
    assert persistent_report_contract_fingerprints()["model.standard_sop"] == "56e5fb1173404d7f"
    # The approved v11 Standard Comparison is immutable; newly authorized
    # model-specific evidence has its own schema-version/fingerprint guard.
    assert persistent_report_contract_fingerprints()["model.standard_comparison"] == "0626f6512f145d23"
    assert independently_frozen_model_extension_contract_fingerprints() == {
        "full_horizon_mae_common_truth": "94ccca56a41ca540",
    }
    ext = extension_contract("full_horizon_mae_common_truth")
    assert ext.contract_version == 1
    with patch.dict(MODEL_COMPARISON_ONLY_EXTENSION_SCHEMAS, {ext.extension_id: replace(ext, title=ext.title + " tampered")}):
        assert any("UNAUTHORIZED_MODEL_SPECIFIC_EXTENSION_CHANGE" in reason
                   for reason in validate_approved_persistent_report_contracts())
    assert validate_approved_persistent_report_contracts() == ()
    assert ext.comparison_scope_key == "split"
    assert "model" == comparison_extension_contract(ext.extension_id).tables[0].columns[0].key
    base = pd.DataFrame({
        "date": list(pd.to_datetime(["2023-01-10"] * 6 + ["2023-01-11"] * 6)),
        "ticker": [f"{i:04d}" for i in range(1, 7)] * 2,
        "group_index": list(range(12)),
        "d40_r": [.1, .3, .5, .7, .9, 1.1] * 2,
        "old_safety_r": [-.1, -.2, -.3, -.4, -.5, -.6] * 2,
    })
    forward_a = base[["ticker", "date", "group_index"]].copy()
    forward_a["safety_prediction"] = [.85, .8, .75, .7, .65, .6] * 2
    forward_b = base[["ticker", "date", "group_index"]].copy()
    forward_b["safety_prediction"] = [.95, .85, .75, .65, .55, .45] * 2
    # Missing OOS ticker/day must make both models use exactly the same joint mask.
    forward_a = forward_a.iloc[1:]
    rows, shared = build_paired_full_horizon_mae_rows(
        truth=base,
        predictions_by_model={"MR-13AO": forward_a, "MR-13CC": forward_b},
        split_label="Forward OOS",
    )
    assert rows["MR-13AO"]["paired_stock_days"] == 11
    assert rows["MR-13CC"]["paired_stock_days"] == 11
    assert shared["d40_strictly_worse_than_peak_pct"] > 0
    assert rows["MR-13CC"]["d40_daily_rho"] is not None
    assert rows["MR-13CC"]["d40_pair"] is not None
    assert rows["MR-13CC"]["top10_mean_d40_r"] is not None
    # A truly mismatched group ID may not be swallowed by a ticker/date join.
    corrupt = forward_b.copy()
    corrupt.loc[1, "group_index"] = 555
    try:
        build_paired_full_horizon_mae_rows(
            truth=base, predictions_by_model={"MR-13AO": forward_a, "MR-13CC": corrupt}, split_label="Forward OOS"
        )
    except ValueError as exc:
        assert "group_index" in str(exc)
    else:
        raise AssertionError("mismatched group ID was accepted")

    # Synthetic integration without any formal Test Suite, market download or retraining.
    import services.breakout_quality.full_horizon_mae_comparison as svc
    from config.breakout_quality import BREAKOUT_QUALITY_MODEL_TEST_PROFILES
    from core.breakout_quality_policy import get_breakout_quality_workflow_settings
    ao = next((m, p) for m, p in BREAKOUT_QUALITY_MODEL_TEST_PROFILES if m == "MR-13AO")
    cc = next((m, p) for m, p in BREAKOUT_QUALITY_MODEL_TEST_PROFILES if m == "MR-13CC")
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        reports = []
        for (mid, name), model_frame in zip((ao, cc), (forward_a, forward_b)):
            path = Path(tmp) / (mid + ".csv.gz")
            score = model_frame.rename(columns={"safety_prediction": "raw_safety_score"})
            score["target_raw_r"] = 1.0
            score.to_csv(path, compression="gzip", index=False)
            from filters.breakout_quality.artifacts import build_file_manifest
            contract = SimpleNamespace(
                score_path=path,
                continuous_target_id="same_target",
                report={"artifacts": {"oos_scores_gzip": build_file_manifest(path)}},
            )
            settings = get_breakout_quality_workflow_settings(experiment_profile=name)
            paths.append((mid, settings, contract))
        bundle = SimpleNamespace(target_valid=np.ones(len(base), dtype=bool), group_table=pd.DataFrame({
            **base.to_dict("list"),
            "target_minimum_low_return": [-.01, -.03, -.05, -.07, -.09, -.11] * 2,
            "target_adverse_r": [.01, .02, .03, .04, .05, .06] * 2,
            "label_eval_end_date": list(pd.to_datetime(["2023-03-01"] * 12)),
        }))
        models = [{"model_id": mid, "settings": settings, "contract": contract, "payload": {}}
                  for mid, settings, contract in paths]
        # Regression: the daily AO/CC comparison must not load the event-group
        # Continuous Target manifest.  This temp root deliberately has no such
        # manifest, while the canonical daily loader supplies its embedded truth.
        with patch.object(svc, "load_daily_universal_ranker_data", return_value=bundle) as daily_loader:
            result = attach_full_horizon_mae_common_truth_extension(
                models, evaluation_mode="forward_oos", project_root=Path(tmp)
            )
            daily_loader.assert_called_once()
            kwargs = daily_loader.call_args.kwargs
            assert kwargs["experiment_profile"] == cc[1]
            assert kwargs["project_root"] == Path(tmp)
            assert kwargs["preload_feature_bank"] is False
            assert kwargs["allow_stale_source"] is False
        assert all(len(item["payload"].get("comparison_extensions", [])) == 1 for item in result)
        assert all(not item["payload"] for item in models), "source reports were mutated"
        # Same persisted rows render identically after RUN and REUSE.
        views = [{"model_id": item["model_id"], "view": {"extensions": item["payload"]["comparison_extensions"]}}
                 for item in result]
        text_a = _render_model_comparison_specific_extensions(views, target="markdown")
        text_b = _render_model_comparison_specific_extensions(views, target="markdown")
        assert text_a == text_b and "D40 Daily rho" in str(text_a)
        # Rolling OOS must retain the same evidence/split semantics on PIT scores.
        rolling_items = []
        for mid, settings, old_contract in paths:
            rolling_path = Path(tmp) / (mid + "_rolling.csv")
            raw_frame = forward_a if mid == "MR-13AO" else forward_b
            rolling_frame = raw_frame.rename(columns={"safety_prediction": "raw_safety_score"}).copy()
            rolling_frame["model_information_cutoff"] = "2022-12-31"
            rolling_frame["fold_id"] = "fold_2023"
            rolling_frame.to_csv(rolling_path, index=False)
            rolling_items.append({
                "model_id": mid, "settings": settings,
                "contract": SimpleNamespace(score_path=rolling_path, continuous_target_id="same_target"),
                "payload": {},
            })
        with patch.object(svc, "load_daily_universal_ranker_data", return_value=bundle):
            rolling = attach_full_horizon_mae_common_truth_extension(
                rolling_items, evaluation_mode="rolling_oos", project_root=Path(tmp)
            )
        assert all(item["payload"]["comparison_extensions"][0]["rows"][0]["split"] == "Rolling OOS" for item in rolling)
        unsafe_path = rolling_items[0]["contract"].score_path
        unsafe_frame = pd.read_csv(unsafe_path)
        unsafe_frame.loc[unsafe_frame.index[0], "model_information_cutoff"] = "2023-01-10"
        unsafe_frame.to_csv(unsafe_path, index=False)
        try:
            with patch.object(svc, "load_daily_universal_ranker_data", return_value=bundle):
                attach_full_horizon_mae_common_truth_extension(
                    rolling_items, evaluation_mode="rolling_oos", project_root=Path(tmp)
                )
        except ValueError as exc:
            assert "information cutoff" in str(exc)
        else:
            raise AssertionError("unsafe PIT information cutoff was accepted")
    print("D40 paired common-truth extension: independent synthetic PASS")


if __name__ == "__main__":
    case()

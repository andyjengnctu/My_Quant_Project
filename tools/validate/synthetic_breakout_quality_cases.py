from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.breakout_policy import (
    BREAKOUT_DEFAULT_HIGH_LEN,
    BREAKOUT_HIGH_LEN_SEARCH_MAX,
    BREAKOUT_HIGH_LEN_SEARCH_MIN,
    BREAKOUT_HIGH_LEN_SEARCH_STEP,
    build_breakout_optimizer_high_len_values,
)
from config.breakout_quality_policy import build_breakout_quality_default_high_len_values
from filters.breakout_quality.artifacts import (
    build_file_manifest,
    load_model_artifact_contract,
    load_runtime_artifact_contract,
)
from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
    RUNTIME_SCOPE_FORWARD_OOS,
    SCORE_COLUMN,
    SCORE_COMPARISON,
    SCORE_TABLE_REQUIRED_COLUMNS,
    SCORE_TABLE_SCHEMA_VERSION,
    SCORE_THRESHOLD_SOURCE,
)
from filters.breakout_quality.paths import resolve_filter_artifact_paths, resolve_filter_research_score_path
from filters.breakout_quality.score_store import build_pass_condition_from_score_table, load_score_table
from core.signal_utils import generate_signals
from core.strategy_params import V16StrategyParams
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE
from tools.filters.breakout_quality.common import chronological_group_split_indices

from .checks import add_check


def validate_breakout_quality_policy_single_source_case(_base_params):
    case_id = "BREAKOUT_QUALITY_POLICY_SSOT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    optimizer_values = build_breakout_optimizer_high_len_values()
    quality_values = build_breakout_quality_default_high_len_values()
    search_spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE["high_len"]

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_default_uses_config",
        int(BREAKOUT_DEFAULT_HIGH_LEN),
        int(BREAKOUT_PARAM_SPECS["high_len"]["default"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "optimizer_range_uses_config",
        (int(BREAKOUT_HIGH_LEN_SEARCH_MIN), int(BREAKOUT_HIGH_LEN_SEARCH_MAX), int(BREAKOUT_HIGH_LEN_SEARCH_STEP)),
        (int(search_spec["low"]), int(search_spec["high"]), int(search_spec["step"])),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "quality_coverage_contains_optimizer_grid",
        True,
        set(optimizer_values).issubset(set(quality_values)),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "quality_coverage_contains_strategy_default",
        True,
        int(BREAKOUT_DEFAULT_HIGH_LEN) in set(quality_values),
    )
    try:
        V16StrategyParams(breakout_quality_filter_id=" ")
        empty_filter_id_rejected = False
    except ValueError as exc:
        empty_filter_id_rejected = "breakout_quality_filter_id" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "empty_filter_id_rejected_without_silent_default",
        True,
        empty_filter_id_rejected,
    )
    summary["optimizer_high_len_count"] = len(optimizer_values)
    summary["quality_high_len_count"] = len(quality_values)
    return results, summary


def validate_breakout_quality_chronological_embargo_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CHRONOLOGICAL_EMBARGO"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    rows = []
    for event_date, label_end_date in (
        ("2025-01-01", "2025-01-02"),
        ("2025-01-02", "2025-01-03"),
        ("2025-01-03", "2025-01-06"),
        ("2025-01-04", "2025-01-07"),
    ):
        for high_len in (100, 105):
            rows.append(
                {
                    "ticker": "2330",
                    "date": event_date,
                    "label_eval_end_date": label_end_date,
                    "high_len": high_len,
                }
            )
    events = pd.DataFrame(rows)
    labels = np.asarray([0, 1] * 4, dtype=np.int64)
    train_idx, val_idx, report = chronological_group_split_indices(
        events,
        labels,
        val_ratio=0.5,
        label_pass=1,
        label_reject=0,
    )

    add_check(results, "synthetic_breakout_quality", case_id, "validation_start_date", "2025-01-03", report["validation_start_date"])
    add_check(results, "synthetic_breakout_quality", case_id, "safe_train_rows_only", 2, len(train_idx))
    add_check(results, "synthetic_breakout_quality", case_id, "embargo_rows_removed", 2, report["embargo_dropped_row_count"])
    add_check(results, "synthetic_breakout_quality", case_id, "validation_rows", 4, len(val_idx))
    add_check(results, "synthetic_breakout_quality", case_id, "group_overlap_forbidden", 0, report["overlap_group_count"])
    add_check(results, "synthetic_breakout_quality", case_id, "event_date_overlap_forbidden", 0, report["overlap_event_date_count"])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "train_label_information_before_validation",
        True,
        pd.to_datetime(events.iloc[train_idx]["label_eval_end_date"]).max() < pd.Timestamp(report["validation_start_date"]),
    )
    summary["split_report"] = report
    return results, summary


def _clear_breakout_quality_caches() -> None:
    load_model_artifact_contract.cache_clear()
    load_runtime_artifact_contract.cache_clear()
    load_score_table.cache_clear()



def _validate_signal_runtime_wiring(results, case_id: str) -> None:
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "Open": [9.0, 9.0, 9.0, 10.5, 10.0],
            "High": [10.0, 10.0, 10.0, 12.0, 12.0],
            "Low": [8.0, 8.0, 8.0, 10.0, 9.0],
            "Close": [9.0, 9.0, 9.0, 11.0, 10.0],
            "Volume": [100.0] * 5,
        },
        index=dates,
    )
    params = SimpleNamespace(
        atr_len=2,
        atr_times_trail=3.0,
        atr_buy_tol=1.0,
        high_len=2,
        use_breakout_buy=True,
        use_breakout_ema_filter=False,
        use_bb=False,
        use_vol=False,
        use_breakout_return_filter=False,
        use_breakout_false_filter=False,
        use_breakout_quality_filter=True,
        breakout_quality_filter_id="synthetic_quality",
        breakout_quality_score_threshold=0.63,
        use_kc=False,
    )
    captured = {}

    def _fake_quality_filter(_df, **kwargs):
        captured.update(kwargs)
        return np.ones(len(_df), dtype=bool)

    with patch("core.signal_utils.build_breakout_quality_filter_pass_condition", side_effect=_fake_quality_filter):
        _atr, buy_condition, _sell_condition, _buy_limits = generate_signals(frame, params, ticker="2330")

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "signal_runtime_receives_exact_crossover_candidates",
        [False, False, False, True, False],
        np.asarray(captured["candidate_condition"], dtype=bool).tolist(),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "signal_runtime_receives_active_threshold",
        0.63,
        float(captured["score_threshold"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "quality_filter_is_and_gate_for_breakout_buy",
        [False, False, False, True, False],
        np.asarray(buy_condition, dtype=bool).tolist(),
    )

def validate_breakout_quality_runtime_artifact_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_RUNTIME_ARTIFACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    filter_id = "synthetic_quality"
    high_len = int(BREAKOUT_DEFAULT_HIGH_LEN)
    with tempfile.TemporaryDirectory(prefix="breakout_quality_contract_") as tmp_dir:
        project_root = Path(tmp_dir)
        models_dir = project_root / "models"
        with patch.dict(os.environ, {"V16_MODELS_DIR": str(models_dir), "V16_BREAKOUT_QUALITY_SCORE_PATH": str(project_root / "legacy.csv")}, clear=False):
            paths = resolve_filter_artifact_paths(project_root, filter_id)
            research_score_path = resolve_filter_research_score_path(project_root, filter_id)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "research_output_cannot_replace_canonical_runtime_score",
                True,
                research_score_path != paths.score_path and "outputs" in research_score_path.parts,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "path_resolution_has_no_output_side_effect",
                False,
                (project_root / "outputs").exists(),
            )
            paths.model_dir.mkdir(parents=True, exist_ok=True)
            paths.model_path.write_bytes(b"synthetic-model")
            score_frame = pd.DataFrame(
                [
                    {"ticker": "2330", "date": "2025-01-03", "high_len": high_len, SCORE_COLUMN: 0.60},
                    {"ticker": "2330", "date": "2025-01-04", "high_len": high_len, SCORE_COLUMN: 0.40},
                ]
            )
            score_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            legacy_path = project_root / "legacy.csv"
            pd.DataFrame(
                [{"ticker": "2330", "date": "2025-01-03", "high_len": high_len, SCORE_COLUMN: 0.0}]
            ).to_csv(legacy_path, index=False)

            score_record = build_file_manifest(paths.score_path)
            score_record.update(
                {
                    "schema_version": SCORE_TABLE_SCHEMA_VERSION,
                    "required_columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "row_count": len(score_frame),
                    "high_len_values": [high_len],
                    "event_date_range": {"start": "2025-01-03", "end": "2025-01-04"},
                }
            )
            manifest = {
                "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
                "filter_family": FILTER_FAMILY,
                "filter_id": filter_id,
                "model": build_file_manifest(paths.model_path),
                "feature_columns": list(FEATURE_COLUMNS),
                "context_columns": list(CONTEXT_COLUMNS),
                "score_decision": {
                    "score_column": SCORE_COLUMN,
                    "comparison": SCORE_COMPARISON,
                    "threshold_source": SCORE_THRESHOLD_SOURCE,
                },
                "score_table": score_record,
                "runtime_eligibility": {
                    "eligible": True,
                    "scope": RUNTIME_SCOPE_FORWARD_OOS,
                    "available_from": "2025-01-03",
                    "available_through": "2025-01-05",
                    "model_information_cutoff": "2025-01-02",
                },
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()

            frame = pd.DataFrame(index=pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-04", "2025-01-05"]))
            candidates = np.asarray([True, True, True, False], dtype=bool)
            pass_at_050 = build_pass_condition_from_score_table(
                frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.50,
                candidate_condition=candidates,
                project_root=str(project_root),
                filter_id=filter_id,
            )
            pass_at_070 = build_pass_condition_from_score_table(
                frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.70,
                candidate_condition=candidates,
                project_root=str(project_root),
                filter_id=filter_id,
            )
            add_check(results, "synthetic_breakout_quality", case_id, "canonical_score_path_only", [True, True, False, True], pass_at_050.tolist())
            add_check(results, "synthetic_breakout_quality", case_id, "active_threshold_controls_decision", [True, False, False, True], pass_at_070.tolist())

            missing_candidate = np.asarray([False, False, False, True], dtype=bool)
            try:
                build_pass_condition_from_score_table(
                    frame,
                    ticker="2330",
                    high_len=high_len,
                    score_threshold=0.50,
                    candidate_condition=missing_candidate,
                    project_root=str(project_root),
                    filter_id=filter_id,
                )
                missing_rejected = False
            except ValueError as exc:
                missing_rejected = "缺少正式候選事件" in str(exc)
            add_check(results, "synthetic_breakout_quality", case_id, "missing_candidate_fails_fast", True, missing_rejected)

            stale_frame = pd.DataFrame(index=pd.to_datetime(["2025-01-06"]))
            no_candidate_after_coverage = build_pass_condition_from_score_table(
                stale_frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.50,
                candidate_condition=np.asarray([False], dtype=bool),
                project_root=str(project_root),
                filter_id=filter_id,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "no_candidate_after_coverage_does_not_require_score",
                [True],
                no_candidate_after_coverage.tolist(),
            )
            try:
                build_pass_condition_from_score_table(
                    stale_frame,
                    ticker="2330",
                    high_len=high_len,
                    score_threshold=0.50,
                    candidate_condition=np.asarray([True], dtype=bool),
                    project_root=str(project_root),
                    filter_id=filter_id,
                )
                stale_rejected = False
            except ValueError as exc:
                stale_rejected = "已過期" in str(exc)
            add_check(results, "synthetic_breakout_quality", case_id, "uncovered_candidate_fails_fast", True, stale_rejected)

            manifest["runtime_eligibility"] = {
                "eligible": False,
                "scope": "research",
                "reason": "synthetic research artifact",
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()
            try:
                load_runtime_artifact_contract(str(project_root), filter_id)
                research_rejected = False
            except ValueError as exc:
                research_rejected = "不可用於正式 runtime" in str(exc)
            add_check(results, "synthetic_breakout_quality", case_id, "research_artifact_rejected", True, research_rejected)

    _clear_breakout_quality_caches()
    _validate_signal_runtime_wiring(results, case_id)
    summary["filter_id"] = filter_id
    summary["high_len"] = high_len
    return results, summary


__all__ = [
    "validate_breakout_quality_chronological_embargo_case",
    "validate_breakout_quality_policy_single_source_case",
    "validate_breakout_quality_runtime_artifact_contract_case",
]

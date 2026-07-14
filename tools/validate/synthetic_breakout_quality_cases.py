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
from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    BREAKOUT_QUALITY_FINAL_REFIT_MODE,
    BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
    BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_EVALUATION_WORKERS,
    BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_LABEL_HORIZON_BARS,
    BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN,
    BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN,
    BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO,
    BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS,
    BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES,
    BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
    build_breakout_quality_default_high_len_values,
)
from filters.breakout_quality.artifacts import (
    build_file_manifest,
    load_model_artifact_contract,
    load_runtime_artifact_contract,
    load_split_assignment_frame,
)
from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONTEXT_COLUMNS,
    DEFAULT_LABEL_POLICY,
    DEFAULT_MODEL_ARCHITECTURE,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
    LABEL_INVALID,
    LABEL_OBJECTIVE,
    LEGACY_LABEL_OBJECTIVE,
    LABEL_PASS,
    LABEL_REJECT,
    BreakoutQualityLabelPolicy,
    label_manifest_payload_from_policy_manifest,
    SELECTION_ROLE_EMBARGO,
    SELECTION_ROLE_INVALID,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_NOT_APPLICABLE,
    SELECTION_ROLE_TRAIN,
    SELECTION_ROLE_VALIDATION,
    OUTER_SPLIT_OOS,
    OUTER_SPLIT_OUT_OF_SCOPE,
    OUTER_SPLIT_SELECTION,
    RUNTIME_SCOPE_FORWARD_OOS,
    SCORE_COLUMN,
    SCORE_COMPARISON,
    SCORE_TABLE_REQUIRED_COLUMNS,
    SCORE_TABLE_SCHEMA_VERSION,
    SCORE_THRESHOLD_SOURCE,
    SPLIT_ASSIGNMENT_REQUIRED_COLUMNS,
    SPLIT_ASSIGNMENT_SCHEMA_VERSION,
    TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
)
from filters.breakout_quality.models import (
    build_model as build_breakout_quality_model,
    count_trainable_parameters,
    get_model_spec,
)
from filters.breakout_quality.models.multiscale_cnn import (
    build_market_relative_return_delta_representation,
    build_return_delta_representation,
)
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_output_dir,
    resolve_filter_research_score_path,
    resolve_filter_report_json_path,
    resolve_filter_report_markdown_path,
)
from filters.breakout_quality.score_store import build_pass_condition_from_score_table, load_score_table
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.dataset_store import (
    DATASET_STORAGE_FORMAT,
    DATASET_STORAGE_SCHEMA_VERSION,
    IndexedFeatureBank,
)
from filters.breakout_quality.features import build_event_label, label_from_cached_path
from filters.breakout_quality.inference import strict_parallel_batched_logits
from core.signal_utils import generate_signals
from core.strategy_params import V16StrategyParams
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE
from tools.filters.breakout_quality import common as breakout_quality_common
from tools.filters.breakout_quality import export_scores as breakout_quality_export_scores
from tools.filters.breakout_quality import train as breakout_quality_train
from tools.filters.breakout_quality.report import (
    build_report_payload,
    render_console_summary,
    render_markdown_report,
)
from filters.breakout_quality.splits import (
    build_selection_oos_split_assignments,
    compute_outer_policy_fingerprint,
    resolve_breakout_quality_outer_policy,
)

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
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "fixed_threshold_is_user_configured_and_legal",
        True,
        0.0 <= float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD) <= 1.0,
    )
    configured_model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "model_architecture_is_versioned_and_user_configured",
        DEFAULT_MODEL_ARCHITECTURE,
        configured_model_spec.architecture,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "dataset_policy_does_not_include_model_architecture",
        False,
        "model_architecture" in DEFAULT_LABEL_POLICY.as_manifest_payload(),
    )
    tiny_model = build_breakout_quality_model(10, 4, architecture="tiny_cnn_v1")
    multiscale_model = build_breakout_quality_model(10, 4, architecture="multiscale_cnn_v1")
    multiscale_v2_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v2"
    )
    multiscale_v3_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v3"
    )
    multiscale_v4_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v4"
    )
    residual_model = build_breakout_quality_model(10, 4, architecture="residual_tcn_v1")
    tiny_parameter_count = count_trainable_parameters(tiny_model)
    multiscale_parameter_count = count_trainable_parameters(multiscale_model)
    multiscale_v2_parameter_count = count_trainable_parameters(multiscale_v2_model)
    multiscale_v3_parameter_count = count_trainable_parameters(multiscale_v3_model)
    multiscale_v4_parameter_count = count_trainable_parameters(multiscale_v4_model)
    residual_parameter_count = count_trainable_parameters(residual_model)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_cnn_is_medium_capacity_with_long_receptive_field",
        True,
        (
            tiny_parameter_count < multiscale_parameter_count < residual_parameter_count
            and 15000 <= multiscale_parameter_count <= 25000
            and get_model_spec("multiscale_cnn_v1").receptive_field_bars >= 240
            and get_model_spec("multiscale_cnn_v1").normalization == "group_norm"
            and "max" not in get_model_spec("multiscale_cnn_v1").pooling
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v2_changes_representation_without_changing_capacity",
        (
            multiscale_parameter_count,
            ("return_delta", "return_delta", "level"),
        ),
        (
            multiscale_v2_parameter_count,
            get_model_spec("multiscale_cnn_v2").branch_input_representations,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v3_adds_market_relative_returns_without_changing_capacity",
        (
            multiscale_parameter_count,
            (
                "market_relative_return_delta",
                "market_relative_return_delta",
                "level",
            ),
        ),
        (
            multiscale_v3_parameter_count,
            get_model_spec("multiscale_cnn_v3").branch_input_representations,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v4_only_reduces_long_branch_channels",
        (
            True,
            (16, 16, 8),
            (),
            get_model_spec("multiscale_cnn_v1").receptive_field_bars,
        ),
        (
            tiny_parameter_count < multiscale_v4_parameter_count < multiscale_parameter_count,
            get_model_spec("multiscale_cnn_v4").branch_channels,
            get_model_spec("multiscale_cnn_v4").branch_input_representations,
            get_model_spec("multiscale_cnn_v4").receptive_field_bars,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v1_v2_v3_manifest_specs_remain_backward_compatible",
        False,
        any(
            "branch_channels" in get_model_spec(architecture).as_manifest_payload()
            for architecture in (
                "multiscale_cnn_v1",
                "multiscale_cnn_v2",
                "multiscale_cnn_v3",
            )
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v1_manifest_spec_remains_backward_compatible",
        False,
        "branch_input_representations"
        in get_model_spec("multiscale_cnn_v1").as_manifest_payload(),
    )
    torch, _nn = breakout_quality_train.require_torch()
    level_sequence = torch.zeros((1, 10, 3), dtype=torch.float32)
    level_sequence[0, 0:5, 1] = torch.tensor(
        [0.01, 0.03, -0.01, 0.02, 0.50], dtype=torch.float32
    )
    level_sequence[0, 0:5, 2] = torch.tensor(
        [0.03, 0.05, 0.01, 0.04, 0.20], dtype=torch.float32
    )
    level_sequence[0, 5:10, 1] = torch.tensor(
        [0.005, 0.02, -0.005, 0.01, 0.25], dtype=torch.float32
    )
    level_sequence[0, 5:10, 2] = torch.tensor(
        [0.02, 0.03, 0.00, 0.025, 0.40], dtype=torch.float32
    )
    return_delta = build_return_delta_representation(torch, level_sequence)
    expected_second_stock = np.asarray(
        [
            np.log1p(0.01),
            np.log1p(0.03),
            np.log1p(-0.01),
            np.log1p(0.02),
            0.50,
        ],
        dtype=np.float32,
    )
    expected_third_stock = np.asarray(
        [
            np.log1p(0.03) - np.log1p(0.02),
            np.log1p(0.05) - np.log1p(0.02),
            np.log1p(0.01) - np.log1p(0.02),
            np.log1p(0.04) - np.log1p(0.02),
            -0.30,
        ],
        dtype=np.float32,
    )
    expected_second_benchmark = np.asarray(
        [
            np.log1p(0.005),
            np.log1p(0.02),
            np.log1p(-0.005),
            np.log1p(0.01),
            0.25,
        ],
        dtype=np.float32,
    )
    expected_third_benchmark = np.asarray(
        [
            np.log1p(0.02) - np.log1p(0.01),
            np.log1p(0.03) - np.log1p(0.01),
            np.log1p(0.00) - np.log1p(0.01),
            np.log1p(0.025) - np.log1p(0.01),
            0.15,
        ],
        dtype=np.float32,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v2_return_delta_transform_matches_canonical_ohlcv_semantics",
        True,
        bool(
            np.allclose(
                return_delta[0, :, 0].detach().cpu().numpy(),
                np.zeros(10, dtype=np.float32),
                atol=1e-7,
            )
            and np.allclose(
                return_delta[0, 0:5, 1].detach().cpu().numpy(),
                expected_second_stock,
                atol=1e-6,
            )
            and np.allclose(
                return_delta[0, 0:5, 2].detach().cpu().numpy(),
                expected_third_stock,
                atol=1e-6,
            )
            and np.allclose(
                return_delta[0, 5:10, 1].detach().cpu().numpy(),
                expected_second_benchmark,
                atol=1e-6,
            )
            and np.allclose(
                return_delta[0, 5:10, 2].detach().cpu().numpy(),
                expected_third_benchmark,
                atol=1e-6,
            )
            and np.isfinite(return_delta.detach().cpu().numpy()).all()
        ),
    )
    market_relative_return_delta = (
        build_market_relative_return_delta_representation(torch, level_sequence)
    )
    expected_second_relative_stock = expected_second_stock.copy()
    expected_second_relative_stock[0:4] -= expected_second_benchmark[0:4]
    expected_third_relative_stock = expected_third_stock.copy()
    expected_third_relative_stock[0:4] -= expected_third_benchmark[0:4]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v3_market_relative_transform_preserves_contract",
        True,
        bool(
            np.allclose(
                market_relative_return_delta[0, :, 0].detach().cpu().numpy(),
                np.zeros(10, dtype=np.float32),
                atol=1e-7,
            )
            and np.allclose(
                market_relative_return_delta[0, 0:5, 1].detach().cpu().numpy(),
                expected_second_relative_stock,
                atol=1e-6,
            )
            and np.allclose(
                market_relative_return_delta[0, 0:5, 2].detach().cpu().numpy(),
                expected_third_relative_stock,
                atol=1e-6,
            )
            and np.allclose(
                market_relative_return_delta[0, 5:10, :].detach().cpu().numpy(),
                return_delta[0, 5:10, :].detach().cpu().numpy(),
                atol=1e-7,
            )
            and np.isfinite(
                market_relative_return_delta.detach().cpu().numpy()
            ).all()
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v2_forward_shape_matches_existing_contract",
        (2, 2),
        tuple(
            multiscale_v2_model(
                torch.zeros((2, 300, 10), dtype=torch.float32),
                torch.zeros((2, 4), dtype=torch.float32),
            ).shape
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v3_forward_shape_matches_existing_contract",
        (2, 2),
        tuple(
            multiscale_v3_model(
                torch.zeros((2, 300, 10), dtype=torch.float32),
                torch.zeros((2, 4), dtype=torch.float32),
            ).shape
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v4_forward_shape_matches_existing_contract",
        (2, 2),
        tuple(
            multiscale_v4_model(
                torch.zeros((2, 300, 10), dtype=torch.float32),
                torch.zeros((2, 4), dtype=torch.float32),
            ).shape
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v4_runtime_branch_widths_match_spec",
        (16, 16, 8),
        tuple(
            int(branch.network[0].conv.out_channels)
            for branch in multiscale_v4_model.branches
        ),
    )
    try:
        build_breakout_quality_model(9, 4, architecture="multiscale_cnn_v2")
        noncanonical_feature_contract_rejected = False
    except ValueError as exc:
        noncanonical_feature_contract_rejected = "canonical 10-column" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v2_rejects_noncanonical_feature_contract",
        True,
        noncanonical_feature_contract_rejected,
    )
    try:
        build_breakout_quality_model(9, 4, architecture="multiscale_cnn_v3")
        v3_noncanonical_feature_contract_rejected = False
    except ValueError as exc:
        v3_noncanonical_feature_contract_rejected = "canonical 10-column" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v3_rejects_noncanonical_feature_contract",
        True,
        v3_noncanonical_feature_contract_rejected,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "residual_tcn_has_larger_receptive_field_and_parameter_count",
        True,
        get_model_spec("residual_tcn_v1").receptive_field_bars
        > get_model_spec("tiny_cnn_v1").receptive_field_bars
        and residual_parameter_count > tiny_parameter_count,
    )
    tiny_paths = resolve_filter_artifact_paths(
        "/project", "synthetic_quality", "tiny_cnn_v1"
    )
    multiscale_paths = resolve_filter_artifact_paths(
        "/project", "synthetic_quality", "multiscale_cnn_v1"
    )
    multiscale_v2_paths = resolve_filter_artifact_paths(
        "/project", "synthetic_quality", "multiscale_cnn_v2"
    )
    multiscale_v3_paths = resolve_filter_artifact_paths(
        "/project", "synthetic_quality", "multiscale_cnn_v3"
    )
    multiscale_v4_paths = resolve_filter_artifact_paths(
        "/project", "synthetic_quality", "multiscale_cnn_v4"
    )
    residual_paths = resolve_filter_artifact_paths(
        "/project", "synthetic_quality", "residual_tcn_v1"
    )
    tiny_research = resolve_filter_research_score_path(
        "/project", "synthetic_quality", "tiny_cnn_v1"
    )
    multiscale_research = resolve_filter_research_score_path(
        "/project", "synthetic_quality", "multiscale_cnn_v1"
    )
    multiscale_v2_research = resolve_filter_research_score_path(
        "/project", "synthetic_quality", "multiscale_cnn_v2"
    )
    multiscale_v3_research = resolve_filter_research_score_path(
        "/project", "synthetic_quality", "multiscale_cnn_v3"
    )
    multiscale_v4_research = resolve_filter_research_score_path(
        "/project", "synthetic_quality", "multiscale_cnn_v4"
    )
    residual_research = resolve_filter_research_score_path(
        "/project", "synthetic_quality", "residual_tcn_v1"
    )
    shared_dataset_dir = resolve_filter_output_dir("/project", "synthetic_quality")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "model_artifacts_are_architecture_scoped_but_dataset_is_shared",
        True,
        len(
            {
                tiny_paths.model_path,
                multiscale_paths.model_path,
                multiscale_v2_paths.model_path,
                multiscale_v3_paths.model_path,
                multiscale_v4_paths.model_path,
                residual_paths.model_path,
            }
        )
        == 6
        and len(
            {
                tiny_research,
                multiscale_research,
                multiscale_v2_research,
                multiscale_v3_research,
                multiscale_v4_research,
                residual_research,
            }
        )
        == 6
        and tiny_paths.model_dir.name == "tiny_cnn_v1"
        and multiscale_paths.model_dir.name == "multiscale_cnn_v1"
        and multiscale_v2_paths.model_dir.name == "multiscale_cnn_v2"
        and multiscale_v3_paths.model_dir.name == "multiscale_cnn_v3"
        and multiscale_v4_paths.model_dir.name == "multiscale_cnn_v4"
        and residual_paths.model_dir.name == "residual_tcn_v1"
        and shared_dataset_dir.name == "synthetic_quality",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "training_defaults_are_user_configured_and_legal",
        True,
        int(BREAKOUT_QUALITY_DEFAULT_EPOCHS) >= 1
        and int(BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE) >= 1
        and float(BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE) > 0.0
        and float(BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY) >= 0.0
        and float(BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM) >= 0.0
        and int(BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED) >= 0
        and int(BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE) >= 1
        and int(BREAKOUT_QUALITY_EVALUATION_WORKERS) >= 1
        and isinstance(BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION, bool)
        and int(BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES) >= 0
        and isinstance(BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK, bool)
        and int(BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES) >= 1
        and BREAKOUT_QUALITY_FINAL_REFIT_MODE in {"matched_optimizer_steps", "selected_epochs"}
        and BREAKOUT_QUALITY_CLASS_WEIGHT_MODE in {"none", "inverse_frequency"}
        and BREAKOUT_QUALITY_TIME_WEIGHT_MODE in {"none", "year_balanced_sqrt"},
    )
    train_defaults = breakout_quality_train.parse_args([])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "training_weight_and_refit_defaults_follow_config",
        (
            BREAKOUT_QUALITY_FINAL_REFIT_MODE,
            BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
            BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
        ),
        (
            str(train_defaults.final_refit_mode),
            str(train_defaults.class_weight_mode),
            str(train_defaults.time_weight_mode),
        ),
    )
    matched_target, minimum_pass = breakout_quality_train._resolve_final_refit_target_steps(
        mode="matched_optimizer_steps",
        selected_epoch=2,
        selected_optimizer_steps=8422,
        final_batches_per_epoch=5701,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "matched_refit_preserves_selected_optimizer_steps",
        (8422, False),
        (matched_target, minimum_pass),
    )
    minimum_target, minimum_pass = breakout_quality_train._resolve_final_refit_target_steps(
        mode="matched_optimizer_steps",
        selected_epoch=1,
        selected_optimizer_steps=4211,
        final_batches_per_epoch=5701,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "matched_refit_still_uses_all_selection_rows_once",
        (5701, True),
        (minimum_target, minimum_pass),
    )
    synthetic_weight_events = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "date": [
                "2020-01-02", "2020-01-02",
                "2020-02-03", "2020-02-03",
                "2020-03-04", "2020-03-04",
                "2020-04-05", "2020-04-05",
                "2021-01-06", "2021-01-06",
            ],
            "high_len": [60, 65] * 5,
        }
    )
    weight_indices = np.arange(len(synthetic_weight_events), dtype=np.int64)
    year_weights, year_summary = breakout_quality_train._time_weighted_group_weights(
        synthetic_weight_events,
        weight_indices,
        mode="year_balanced_sqrt",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "year_balanced_weights_keep_group_total_and_soften_year_dominance",
        True,
        bool(
            abs(float(year_weights.sum()) - 5.0) < 1e-6
            and year_summary["year_group_counts"] == {"2020": 4, "2021": 1}
            and year_summary["year_weight_multipliers"]["2021"]
            > year_summary["year_weight_multipliers"]["2020"]
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "class_weight_none_is_identity",
        [1.0, 1.0],
        breakout_quality_train._class_weights(
            np.asarray([0, 1], dtype=np.int64),
            np.ones((2,), dtype=np.float32),
            mode="none",
        ).tolist(),
    )

    export_defaults = breakout_quality_export_scores.parse_args([])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_export_performance_defaults_share_training_policy",
        (
            int(BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE),
            int(BREAKOUT_QUALITY_EVALUATION_WORKERS),
            bool(BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK),
        ),
        (
            int(export_defaults.inference_batch_size),
            int(export_defaults.inference_workers),
            bool(export_defaults.preload_feature_bank),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inner_validation_config_has_legal_types_and_ranges",
        True,
        isinstance(BREAKOUT_QUALITY_USE_INNER_VALIDATION, bool)
        and int(BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS) >= 1
        and int(BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE) >= 0
        and float(BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA) >= 0.0
        and int(BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES) >= 1
        and int(BREAKOUT_QUALITY_LABEL_HORIZON_BARS) >= 1
        and int(BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS) >= int(BREAKOUT_QUALITY_LABEL_HORIZON_BARS)
        and float(BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN) > 0.0
        and float(BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO) > 1.0
        and -1.0 < float(BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN) < 0.0,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "dataset_storage_contract_uses_indexed_feature_bank",
        (3, "indexed_feature_bank_npy_v2"),
        (DATASET_STORAGE_SCHEMA_VERSION, DATASET_STORAGE_FORMAT),
    )
    feature_bank = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    indexed_features = IndexedFeatureBank(feature_bank, np.asarray([0, 1, 0], dtype=np.int32))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "indexed_feature_bank_reuses_ticker_date_sequence",
        True,
        indexed_features.shape == (3, 3, 4)
        and np.array_equal(indexed_features[0], indexed_features[2])
        and np.array_equal(indexed_features[1], feature_bank[1]),
    )
    label_policy = BreakoutQualityLabelPolicy(
        feature_window_bars=2,
        label_horizon_bars=3,
        label_path_cache_bars=5,
        high_len_values=(2,),
        min_mfe_return=0.05,
        min_reward_risk_ratio=1.20,
        max_adverse_return=-0.10,
        benchmark_ticker="0050",
    )

    def _label_case(highs, lows):
        frame = pd.DataFrame(
            {
                "Open": [100.0] * 4,
                "High": [100.0, *highs],
                "Low": [100.0, *lows],
                "Close": [100.0] * 4,
                "Volume": [1000.0] * 4,
            },
            index=pd.date_range("2025-01-01", periods=4, freq="D"),
        )
        return build_event_label(frame, event_pos=0, policy=label_policy)[:2]

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_passes_low_risk_moderate_gain",
        (LABEL_PASS, "risk_adjusted_opportunity"),
        _label_case([104.0, 106.0, 108.0], [99.0, 98.0, 97.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_rejects_insufficient_ratio",
        (LABEL_REJECT, "no_risk_adjusted_opportunity"),
        _label_case([104.0, 106.0, 108.0], [96.0, 94.0, 93.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_rejects_when_downside_limit_hits_first",
        (LABEL_REJECT, "downside_first"),
        _label_case([104.0, 106.0, 108.0], [90.0, 92.0, 93.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_uses_conservative_same_bar_order",
        (LABEL_REJECT, "same_bar_adverse_first"),
        _label_case([113.0, 114.0, 115.0], [90.0, 92.0, 93.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_rejects_below_minimum_mfe",
        (LABEL_REJECT, "no_risk_adjusted_opportunity"),
        _label_case([104.0, 104.5, 104.9], [99.0, 99.0, 99.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_requires_strictly_more_than_minimum_mfe",
        (LABEL_REJECT, "no_risk_adjusted_opportunity"),
        _label_case([105.0, 105.0, 105.0], [100.0, 100.0, 100.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_handles_zero_mae_without_division_error",
        (LABEL_PASS, "risk_adjusted_opportunity"),
        _label_case([106.0, 107.0, 108.0], [100.0, 100.0, 100.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_requires_strictly_more_than_ratio_threshold",
        (LABEL_REJECT, "no_risk_adjusted_opportunity"),
        _label_case([106.0, 106.0, 106.0], [95.0, 95.0, 95.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_pass_is_not_reversed_by_later_drawdown",
        (LABEL_PASS, "risk_adjusted_opportunity"),
        _label_case([104.0, 106.0, 107.0], [99.0, 98.0, 89.0]),
    )
    invalid_frame = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 104.0, 106.0],
            "Low": [100.0, 99.0, 98.0],
            "Close": [100.0, 100.0, 100.0],
            "Volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.date_range("2025-02-01", periods=3, freq="D"),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "incomplete_future_path_is_invalid_not_a_third_label",
        (LABEL_INVALID, "insufficient_future"),
        build_event_label(invalid_frame, event_pos=0, policy=label_policy)[:2],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "label_manifest_declares_risk_adjusted_opportunity_objective",
        LABEL_OBJECTIVE,
        label_policy.label_manifest_payload().get("label_objective"),
    )
    legacy_policy_payload = {
        "label_objective": LEGACY_LABEL_OBJECTIVE,
        "label_horizon_bars": 40,
        "pass_return_threshold": 0.15,
        "reject_return_threshold": -0.07,
    }
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "legacy_label_policy_remains_readable_for_fast_relabel",
        legacy_policy_payload,
        label_manifest_payload_from_policy_manifest(legacy_policy_payload),
    )
    cached_result = label_from_cached_path(
        np.asarray([104.0, 106.0, 108.0, np.nan, np.nan], dtype=np.float64),
        np.asarray([99.0, 98.0, 97.0, np.nan, np.nan], dtype=np.float64),
        anchor_price=100.0,
        available_bars=3,
        policy=label_policy,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "cached_future_path_relabel_matches_risk_adjusted_contract",
        (LABEL_PASS, "risk_adjusted_opportunity", 2.0),
        (cached_result.label, cached_result.reason, cached_result.first_hit_bar),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "label_thresholds_are_separate_from_feature_cache_policy",
        True,
        "min_mfe_return" not in label_policy.feature_cache_manifest_payload()
        and "min_reward_risk_ratio" not in label_policy.feature_cache_manifest_payload()
        and "max_adverse_return" not in label_policy.feature_cache_manifest_payload()
        and label_policy.label_manifest_payload()["min_mfe_return"] == 0.05
        and label_policy.label_manifest_payload()["min_reward_risk_ratio"] == 1.20
        and label_policy.label_manifest_payload()["max_adverse_return"] == -0.10,
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

    torch, nn = breakout_quality_train.require_torch()

    class _DeterministicEvaluationModel(nn.Module):
        def forward(self, x, context):
            pooled = x.mean(dim=(1, 2))
            return torch.stack(
                [pooled + context[:, 0], -pooled + context[:, 1]],
                dim=1,
            )

    evaluation_model = _DeterministicEvaluationModel()
    evaluation_features = np.arange(12 * 3 * 2, dtype=np.float32).reshape(12, 3, 2) / 100.0
    evaluation_context = np.arange(12 * 2, dtype=np.float32).reshape(12, 2) / 50.0
    evaluation_labels = np.asarray([0, 1] * 6, dtype=np.int64)
    evaluation_indices = np.asarray([11, 2, 8, 1, 6, 4, 9, 0, 5], dtype=np.int64)
    evaluation_weights = np.linspace(0.5, 1.5, 12, dtype=np.float32)
    evaluation_class_weights = torch.tensor([1.25, 0.75], dtype=torch.float32)
    one_shot_metrics = breakout_quality_train._evaluate(
        torch,
        evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        evaluation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=len(evaluation_indices),
        evaluation_workers=1,
    )
    chunked_metrics = breakout_quality_train._evaluate(
        torch,
        evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        evaluation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=1,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "chunked_full_evaluation_preserves_one_shot_metrics",
        one_shot_metrics,
        chunked_metrics,
    )

    parallel_metrics = breakout_quality_train._evaluate(
        torch,
        evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        evaluation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=4,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "parallel_chunked_evaluation_preserves_serial_metrics",
        chunked_metrics,
        parallel_metrics,
    )

    torch.manual_seed(20260712)
    actual_evaluation_model = breakout_quality_train.build_model(2, 2, architecture="tiny_cnn_v1")
    state_before_evaluation = {
        key: value.detach().clone()
        for key, value in actual_evaluation_model.state_dict().items()
    }
    actual_serial_metrics = breakout_quality_train._evaluate(
        torch,
        actual_evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        evaluation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=1,
    )
    actual_parallel_metrics = breakout_quality_train._evaluate(
        torch,
        actual_evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        evaluation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=4,
    )
    state_after_evaluation = actual_evaluation_model.state_dict()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "parallel_actual_model_evaluation_preserves_serial_metrics",
        actual_serial_metrics,
        actual_parallel_metrics,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "parallel_evaluation_does_not_mutate_model_state",
        True,
        all(
            torch.equal(state_before_evaluation[key], state_after_evaluation[key])
            for key in state_before_evaluation
        ),
    )

    split_train_indices = evaluation_indices[:6]
    split_validation_indices = evaluation_indices[6:]
    serial_split_metrics = breakout_quality_train._evaluate_inner_splits(
        torch,
        actual_evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        split_train_indices,
        split_validation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=2,
        parallel=False,
    )
    parallel_split_metrics = breakout_quality_train._evaluate_inner_splits(
        torch,
        actual_evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        split_train_indices,
        split_validation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=2,
        parallel=True,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "concurrent_train_validation_evaluation_preserves_serial_metrics",
        serial_split_metrics,
        parallel_split_metrics,
    )

    indexed_features = IndexedFeatureBank(
        evaluation_features[:4],
        np.asarray([0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3], dtype=np.int64),
    )
    preloaded_features, preloaded_context, preloaded_labels = (
        breakout_quality_train._preload_training_arrays(
            indexed_features,
            evaluation_context,
            evaluation_labels.astype(np.int8),
            enabled=True,
        )
    )
    serial_export_logits = strict_parallel_batched_logits(
        torch,
        actual_evaluation_model,
        indexed_features,
        evaluation_context,
        indices=None,
        batch_size=3,
        workers=1,
    )
    parallel_export_logits = strict_parallel_batched_logits(
        torch,
        actual_evaluation_model,
        preloaded_features,
        preloaded_context,
        indices=None,
        batch_size=3,
        workers=4,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strict_parallel_score_inference_preserves_serial_logits",
        True,
        np.array_equal(serial_export_logits, parallel_export_logits),
    )

    preload_probe = np.asarray([11, 0, 7, 4, 2], dtype=np.int64)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "feature_bank_preload_preserves_values_and_row_mapping",
        True,
        np.array_equal(indexed_features[preload_probe], preloaded_features[preload_probe])
        and np.array_equal(evaluation_context, preloaded_context)
        and np.array_equal(evaluation_labels, preloaded_labels),
    )

    with tempfile.TemporaryDirectory(prefix="breakout_quality_source_inventory_") as temp_dir:
        project_root = Path(temp_dir)
        source_dir = project_root / "data" / "tw_stock_data_vip_reduced"
        source_dir.mkdir(parents=True, exist_ok=True)
        first_path = source_dir / "2330.csv"
        second_path = source_dir / "0050.csv"
        first_path.write_text("Date,Open\n2026-01-01,100\n", encoding="utf-8")
        second_path.write_text("Date,Open\n2026-01-01,50\n", encoding="utf-8")

        inventory_before = build_source_data_inventory(project_root, "reduced")
        inventory_repeat = build_source_data_inventory(project_root, "reduced")
        first_path.write_text(
            "Date,Open\n2026-01-01,100\n2026-01-02,101\n",
            encoding="utf-8",
        )
        inventory_after = build_source_data_inventory(project_root, "reduced")

        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "source_inventory_is_stable_without_changes",
            inventory_before,
            inventory_repeat,
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "source_inventory_detects_csv_update",
            True,
            inventory_before["csv_inventory_sha256"] != inventory_after["csv_inventory_sha256"],
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "source_inventory_tracks_unique_csv_count",
            2,
            inventory_after["csv_file_count"],
        )

        output_dir = project_root / "outputs" / "filters" / "breakout_quality" / "synthetic_quality"
        output_dir.mkdir(parents=True, exist_ok=True)
        stale_summary = {
            "filter_id": "synthetic_quality",
            "dataset": "reduced",
            "dataset_storage_schema_version": DATASET_STORAGE_SCHEMA_VERSION,
            "dataset_storage_format": DATASET_STORAGE_FORMAT,
            "policy": DEFAULT_LABEL_POLICY.as_manifest_payload(),
            "label_policy": DEFAULT_LABEL_POLICY.label_manifest_payload(),
            "feature_columns": list(FEATURE_COLUMNS),
            "context_columns": list(CONTEXT_COLUMNS),
            "source_data_inventory": inventory_before,
        }
        (output_dir / "dataset_summary.json").write_text(
            json.dumps(stale_summary),
            encoding="utf-8",
        )
        stale_source_rejected = False
        with (
            patch.object(breakout_quality_common, "PROJECT_ROOT", project_root),
            patch.object(
                breakout_quality_common,
                "dataset_output_dir",
                return_value=output_dir,
            ),
        ):
            try:
                breakout_quality_common.load_validated_dataset_bundle(
                    "synthetic_quality",
                    require_current_source=True,
                )
            except ValueError as exc:
                stale_source_rejected = "來源 CSV 已更新" in str(exc)
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "standalone_training_rejects_stale_source_dataset",
            True,
            stale_source_rejected,
        )

    summary["optimizer_high_len_count"] = len(optimizer_values)
    summary["quality_high_len_count"] = len(quality_values)
    return results, summary


def validate_breakout_quality_chronological_embargo_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CHRONOLOGICAL_EMBARGO"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    date_and_end = (
        ("2024-12-31", "2025-01-02"),
        ("2025-01-10", "2025-01-20"),
        ("2025-02-20", "2025-03-05"),
        ("2025-03-10", "2025-03-20"),
        ("2025-04-20", "2025-05-05"),
        ("2025-05-10", "2025-05-20"),
        ("2025-06-20", "2025-07-05"),
    )
    rows = []
    for event_date, label_end_date in date_and_end:
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
    labels = np.asarray([0, 1] * len(date_and_end), dtype=np.int64)
    outer_policy = {
        "policy_source": "core.walk_forward_policy.synthetic_override",
        "selection_start_date": "2025-01-01",
        "selection_end_date": "2025-04-30",
        "oos_start_date": "2025-05-01",
        "configured_oos_end_date": "2025-06-30",
        "effective_oos_end_date": "2025-06-30",
    }
    outer_policy["policy_fingerprint_sha256"] = compute_outer_policy_fingerprint(
        outer_policy
    )

    (
        assignments_off,
        train_off,
        validation_off,
        refit_off,
        oos_off,
        report_off,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=False,
        inner_validation_months=2,
        early_stopping_enabled=False,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_off_uses_full_selection",
        (6, 0, 6, False, False),
        (
            len(train_off),
            len(validation_off),
            len(refit_off),
            report_off["inner_validation_used"],
            report_off["early_stopping_used"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_off_selection_oos_embargo_rows",
        2,
        report_off["selection_oos_embargo_row_count"],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_off_role_counts",
        {
            SELECTION_ROLE_TRAIN: 6,
            SELECTION_ROLE_VALIDATION: 0,
            SELECTION_ROLE_INNER_EMBARGO: 0,
            SELECTION_ROLE_EMBARGO: 2,
            SELECTION_ROLE_INVALID: 0,
            SELECTION_ROLE_NOT_APPLICABLE: 6,
        },
        report_off["selection_role_counts"],
    )

    (
        assignments_on,
        train_on,
        validation_on,
        refit_on,
        oos_on,
        report_on,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=True,
        inner_validation_months=2,
        early_stopping_enabled=True,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_on_period_and_roles",
        ("2025-03-01", 2, 2, 2, 6, True, True),
        (
            report_on["inner_validation_start_date"],
            len(train_on),
            len(validation_on),
            report_on["inner_train_validation_embargo_row_count"],
            len(refit_on),
            report_on["inner_validation_used"],
            report_on["early_stopping_used"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_on_refit_recovers_inner_embargo_rows",
        True,
        set(refit_on.tolist())
        == set(train_on.tolist())
        | set(validation_on.tolist())
        | set(
            np.flatnonzero(
                assignments_on["selection_role"].to_numpy()
                == SELECTION_ROLE_INNER_EMBARGO
            ).tolist()
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_on_role_counts",
        {
            SELECTION_ROLE_TRAIN: 2,
            SELECTION_ROLE_VALIDATION: 2,
            SELECTION_ROLE_INNER_EMBARGO: 2,
            SELECTION_ROLE_EMBARGO: 2,
            SELECTION_ROLE_INVALID: 0,
            SELECTION_ROLE_NOT_APPLICABLE: 6,
        },
        report_on["selection_role_counts"],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inner_train_label_information_before_validation",
        True,
        pd.to_datetime(events.iloc[train_on]["label_eval_end_date"]).max()
        < pd.Timestamp(report_on["inner_validation_start_date"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "final_refit_label_information_before_oos",
        True,
        pd.to_datetime(events.iloc[refit_on]["label_eval_end_date"]).max()
        < pd.Timestamp(report_on["oos_start_date"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "oos_evaluable_and_tail_rows",
        (2, 2),
        (len(oos_on), report_on["oos_label_after_end_row_count"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "outer_and_inner_overlap_forbidden",
        (0, 0, 0, 0),
        (
            report_on["overlap_group_count"],
            report_on["overlap_event_date_count"],
            report_on["inner_train_validation_overlap_group_count"],
            report_on["inner_train_validation_overlap_event_date_count"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "outside_selection_has_no_selection_role",
        True,
        bool(
            (
                assignments_on.loc[
                    assignments_on["outer_split"] != OUTER_SPLIT_SELECTION,
                    "selection_role",
                ]
                == SELECTION_ROLE_NOT_APPLICABLE
            ).all()
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "split_assignment_key_unique",
        False,
        bool(
            assignments_on.duplicated(
                ["ticker", "date", "high_len"],
                keep=False,
            ).any()
        ),
    )

    project_root = Path(__file__).resolve().parents[2]
    rolling_fold_policy = resolve_breakout_quality_outer_policy(
        project_root,
        source_data_end_date="2021-12-31",
        environ={
            "V16_WF_SELECTION_START_DATE": "2019-01-01",
            "V16_WF_TRAIN_START_DATE": "2019-01-01",
            "V16_WF_SEARCH_TRAIN_END_DATE": "2020-12-31",
            "V16_WF_OOS_START_DATE": "2021-01-01",
            "V16_WF_OOS_END_DATE": "2021-12-31",
        },
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_fold_reuses_standard_walk_forward_overrides",
        ("2019-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        (
            rolling_fold_policy["selection_start_date"],
            rolling_fold_policy["selection_end_date"],
            rolling_fold_policy["oos_start_date"],
            rolling_fold_policy["effective_oos_end_date"],
        ),
    )
    summary["split_report_off"] = report_off
    summary["split_report_on"] = report_on
    return results, summary


def _clear_breakout_quality_caches() -> None:
    load_model_artifact_contract.cache_clear()
    load_runtime_artifact_contract.cache_clear()
    load_split_assignment_frame.cache_clear()
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


def _validate_breakout_quality_report_rendering(results, case_id):
    def _metrics(split_name, *, base, precision, acceptance, recall, false_reject):
        total = 100.0
        actual_pass = float(base) * total
        actual_reject = total - actual_pass
        predicted_pass = float(acceptance) * total
        tp = float(recall) * actual_pass
        fn = actual_pass - tp
        fp = predicted_pass - tp
        tn = actual_reject - fp
        specificity = tn / actual_reject
        accuracy = (tp + tn) / total
        return {
            "filter_id": "synthetic_quality",
            "split": split_name,
            "selected_date_range": {"start": "2021-01-01", "end": "2022-12-31"},
            "ticker_date_group_weighted": {
                "group_count": 100,
                "row_count": 1000,
                "weight_sum": total,
                "base_pass_rate": base,
                "acceptance_rate": acceptance,
                "pass_precision": precision,
                "precision_lift_vs_all_pass": precision / base,
                "pass_recall": recall,
                "false_rejection_rate": false_reject,
                "reject_specificity": specificity,
                "accuracy": accuracy,
                "all_pass_baseline_accuracy": base,
                "avg_score": 0.45,
                "confusion": {
                    "true_pass_pred_pass": tp,
                    "true_pass_pred_reject": fn,
                    "true_reject_pred_pass": fp,
                    "true_reject_pred_reject": tn,
                },
            },
            "row_level": {"row_count": 1000},
        }

    context = {
        "filter_id": "synthetic_quality",
        "score_path": Path("research_scores.csv"),
        "model_manifest": {
            "model_architecture": DEFAULT_MODEL_ARCHITECTURE,
            "model_spec": get_model_spec(DEFAULT_MODEL_ARCHITECTURE).as_manifest_payload(),
            "trainable_parameter_count": count_trainable_parameters(
                build_breakout_quality_model(10, 4, architecture=DEFAULT_MODEL_ARCHITECTURE)
            ),
            "sequence_length": int(DEFAULT_LABEL_POLICY.feature_window_bars),
            "training_mode": "inner_validation_epoch_selection_full_refit",
            "inner_validation_used": True,
            "max_epochs": 20,
            "selected_epoch": 2,
            "fixed_evaluation_threshold": 0.5,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "final_refit_plan": {
                "mode": "matched_optimizer_steps",
                "actual_optimizer_steps": 200,
                "equivalent_epochs": 1.5,
            },
            "class_weight_mode": "none",
            "time_weight_mode": "none",
            "batch_size": 256,
            "seed": 42,
            "epoch_selection_source": "inner_validation_loss",
            "early_stopping_enabled": True,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
            "inner_validation_epoch_selection": {
                "best_epoch": 2,
                "completed_epochs": 3,
                "best_validation_loss": 0.77253,
                "best_validation_metrics": {
                    "accuracy": 0.505773,
                    "pass_rate": 0.344325,
                },
                "history": [
                    {
                        "epoch": 1,
                        "batch_loss": 0.681608,
                        "inner_train_metrics": {
                            "loss": 0.670129,
                            "accuracy": 0.578197,
                            "pass_rate": 0.48634,
                        },
                        "inner_validation_metrics": {
                            "loss": 0.774969,
                            "accuracy": 0.500052,
                            "pass_rate": 0.363297,
                        },
                        "is_best_epoch": True,
                    },
                    {
                        "epoch": 2,
                        "batch_loss": 0.666133,
                        "inner_train_metrics": {
                            "loss": 0.655166,
                            "accuracy": 0.604912,
                            "pass_rate": 0.553797,
                        },
                        "inner_validation_metrics": {
                            "loss": 0.77253,
                            "accuracy": 0.505773,
                            "pass_rate": 0.344325,
                        },
                        "is_best_epoch": True,
                    },
                    {
                        "epoch": 3,
                        "batch_loss": 0.657169,
                        "inner_train_metrics": {
                            "loss": 0.651273,
                            "accuracy": 0.615911,
                            "pass_rate": 0.496689,
                        },
                        "inner_validation_metrics": {
                            "loss": 0.793311,
                            "accuracy": 0.519413,
                            "pass_rate": 0.220611,
                        },
                        "is_best_epoch": False,
                    },
                ],
            },
        },
    }
    payload = build_report_payload(
        metrics_by_split={
            "train": _metrics(
                "train",
                base=0.442189,
                precision=0.569538,
                acceptance=0.368851,
                recall=0.475079,
                false_reject=0.524921,
            ),
            "validation": _metrics(
                "validation",
                base=0.492563,
                precision=0.626719,
                acceptance=0.426307,
                recall=0.542417,
                false_reject=0.457583,
            ),
            "selection": _metrics(
                "selection",
                base=0.456199,
                precision=0.586467,
                acceptance=0.387072,
                recall=0.497601,
                false_reject=0.502399,
            ),
            "oos": _metrics(
                "oos",
                base=0.493898,
                precision=0.460815,
                acceptance=0.169293,
                recall=0.157953,
                false_reject=0.842047,
            ),
        },
        context=context,
    )
    markdown = render_markdown_report(payload)
    console = render_console_summary(payload)
    colored_console = render_console_summary(payload, color=True)
    no_oos_payload = build_report_payload(
        metrics_by_split={
            split_name: metrics
            for split_name, metrics in payload["full_metrics"].items()
            if split_name != "oos"
        },
        context=context,
    )
    no_oos_console = render_console_summary(no_oos_payload)
    add_check(results, "synthetic_breakout_quality", case_id, "report_oos_fail_status", "FAIL", payload["conclusion"]["status"])
    add_check(results, "synthetic_breakout_quality", case_id, "report_uses_group_weighted_headline", "ticker_date_group_weighted", payload["headline_basis"])
    add_check(results, "synthetic_breakout_quality", case_id, "report_schema_v2", 2, payload["schema_version"])
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_header_includes_fixed_training_parameters", True, "- **Threshold**：`0.5`" in markdown and "- **Learning Rate**：`0.001`" in markdown and "- **Weight Decay**：`0.0001`" in markdown and "- **Gradient Clip Norm**：`1.0`" in markdown and "- **Batch Size**：`256`" in markdown and "- **Random Seed**：`42`" in markdown and "## 1. 固定訓練參數" not in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_has_epoch_comparison", True, "## 1. Epoch 選擇結果" in markdown and "最終模型" in markdown and "Validation Loss" in markdown)
    selection_matrix = markdown.split("## 2. Selection Confusion Matrix", 1)[1].split("### 分類品質", 1)[0]
    normalized_selection_matrix = selection_matrix.replace("**", "")
    console_selection_matrix = console.split("2. Selection Confusion Matrix", 1)[1].split("分類品質", 1)[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_confusion_matrix_is_count_only_with_standard_margins",
        True,
        (
            all(token in selection_matrix for token in ("TP =", "FN =", "FP =", "TN =", "模型 PASS", "模型 REJECT", "原始 PASS", "原始 REJECT"))
            and all(token in console_selection_matrix for token in ("TP =", "FN =", "FP =", "TN =", "模型 PASS", "模型 REJECT", "原始 PASS", "原始 REJECT"))
            and all(token not in selection_matrix for token in ("Precision", "Recall", "Specificity", "NPV", "Accuracy"))
            and all(token not in console_selection_matrix for token in ("Precision", "Recall", "Specificity", "NPV", "Accuracy"))
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_confusion_margins_use_requested_labels_and_order",
        True,
        (
            all(
                token in normalized_selection_matrix
                for token in (
                    "原始PASS =",
                    "TP + FN =",
                    "原始REJECT =",
                    "FP + TN =",
                    "TP + FP =",
                    "模型PASS =",
                    "FN + TN =",
                    "模型REJECT =",
                )
            )
            and all(
                token in console_selection_matrix
                for token in (
                    "原始PASS =",
                    "TP + FN =",
                    "原始REJECT =",
                    "FP + TN =",
                    "TP + FP =",
                    "模型PASS =",
                    "FN + TN =",
                    "模型REJECT =",
                )
            )
            and normalized_selection_matrix.index("原始PASS =") < normalized_selection_matrix.index("TP + FN =")
            and normalized_selection_matrix.index("原始REJECT =") < normalized_selection_matrix.index("FP + TN =")
            and normalized_selection_matrix.index("TP + FP =") < normalized_selection_matrix.index("模型PASS =")
            and normalized_selection_matrix.index("FN + TN =") < normalized_selection_matrix.index("模型REJECT =")
            and console_selection_matrix.index("原始PASS =") < console_selection_matrix.index("TP + FN =")
            and console_selection_matrix.index("原始REJECT =") < console_selection_matrix.index("FP + TN =")
            and console_selection_matrix.index("TP + FP =") < console_selection_matrix.index("模型PASS =")
            and console_selection_matrix.index("FN + TN =") < console_selection_matrix.index("模型REJECT =")
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_metric_tables_include_formulas_and_explanations",
        True,
        (
            "| 指標 | 公式 | 結果 | 解釋 |" in markdown
            and "| 指標 | 公式 | 結果 |" in markdown
            and "TP ÷ (TP + FP)" in markdown
            and "PASS Precision − 原始 PASS" in markdown
            and "被保留的訊號中，有多少真的 PASS" in markdown
            and "篩選行為" not in markdown
            and "篩選行為" not in console
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_marks_oos_not_for_retuning", True, "不得使用同一段 OOS 回頭調整" in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_console_has_epoch_and_confusion_tables", True, "1. Epoch 選擇結果" in console and "2. Selection Confusion Matrix" in console and "3. OOS Confusion Matrix" in console and "4. 各資料區段比較" in console and "5. OOS 綜合判定" in console and "Inner Train" in console and "Validation*" in console and "Precision" in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_confusion_omits_redundant_orientation_text", True, "統計口徑：Ticker/Date Group Weighted" not in console and "列 = 原始結果；欄 = 模型判定" not in console and "ticker/date group weighted`；列為原始結果" not in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_header_merges_fixed_training_parameters", True, "Threshold       : 0.5" in console and "Learning Rate   : 0.001" in console and "Weight Decay    : 0.0001" in console and "Gradient Clip   : 1.0" in console and "Batch Size      : 256" in console and "Random Seed     : 42" in console and "固定訓練參數" not in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_epoch_summary_uses_bullets", True, "- Epoch 上限：20" in console and "- 最終模型：Inner Validation 選出 Epoch 2" in console and "| Epoch 上限" not in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_split_and_date_are_separate_columns", True, "區段 / 日期" not in console and "|    區段" in console and "|          日期" in console and "| 區段 | 日期 |" in markdown)
    markdown_section_4 = markdown.split("## 4. 各資料區段比較", 1)[1].split("## 5. OOS 綜合判定", 1)[0]
    markdown_section_5 = markdown.split("## 5. OOS 綜合判定", 1)[1].split("## 指標白話說明", 1)[0]
    console_section_4 = console.split("4. 各資料區段比較", 1)[1].split("5. OOS 綜合判定", 1)[0]
    console_section_5 = console.split("5. OOS 綜合判定", 1)[1]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_sections_4_and_5_use_consistent_metric_names",
        True,
        (
            all("模型 PASS" in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
            and all("保留率" not in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
            and all("原始 PASS" in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
            and all("PASS Precision" in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
        ),
    )
    compact_section_4_headers = (
        "區段",
        "日期",
        "Groups",
        "原始 PASS",
        "模型 PASS",
        "PASS Precision",
        "Precision 絕對",
        "PASS Recall",
        "平均 Score",
    )
    removed_section_4_headers = (
        "REJECT Specificity",
        "REJECT NPV",
        "Accuracy",
        "Precision 相對",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_section_4_is_compact_and_ordered",
        True,
        (
            "| 區段 | 日期 | Groups | 原始 PASS | 模型 PASS | PASS Precision | Precision 絕對 | PASS Recall | 平均 Score |"
            in markdown_section_4
            and all(label in console_section_4 for label in compact_section_4_headers)
            and all(label not in markdown_section_4 for label in removed_section_4_headers)
            and all(label not in console_section_4 for label in removed_section_4_headers)
            and all(
                markdown_section_4.index(left) < markdown_section_4.index(right)
                for left, right in zip(compact_section_4_headers, compact_section_4_headers[1:])
            )
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_oos_assessment_has_three_metric_groups_and_judgements",
        True,
        (
            "| 類別 | 指標 | Selection | OOS | OOS - Selection | 判讀 |" in markdown_section_5
            and all(label in markdown_section_5 for label in ("主要成效", "過度篩選防線", "輔助診斷"))
            and all(label in markdown_section_5 for label in ("PASS Precision", "PASS Recall", "模型 PASS", "REJECT Specificity", "REJECT NPV", "Accuracy", "平均 Score"))
            and "綜合判定" in markdown_section_5
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_epoch_selection_is_not_repeated_in_bullets", True, "- 最後選擇：" not in console and "是否選出 Best Epoch" not in console and console.count("- 最終模型：") == 1)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_metric_labels_are_consistent_across_tables",
        True,
        (
            all(
                label in markdown and label in console
                for label in (
                    "原始 PASS",
                    "模型 PASS",
                    "PASS Precision",
                    "PASS Recall",
                    "REJECT Specificity",
                    "REJECT NPV",
                    "Accuracy",
                    "Precision 絕對",
                    "Precision 相對",
                    "平均 Score",
                )
            )
            and "模型保留" not in markdown
            and "模型保留" not in console
            and "REJECT 辨識率" not in markdown
            and "REJECT 辨識率" not in console
            and "保留後 Precision" not in markdown
            and "保留後 Precision" not in console
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_oos_assessment_merges_comparison_and_deployment",
        True,
        (
            "5. OOS 綜合判定" in console
            and "類別" in console
            and "判讀" in console
            and "主要成效" in console
            and "過度篩選防線" in console
            and "輔助診斷" in console
            and "- 最終部署決策：" in console
            and "- 研究限制：" in console
            and "Selection 與 OOS 差異" not in console
            and "最終部署判定" not in console
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_header_does_not_duplicate_final_decision", True, "最終判定" not in console.split("1. Epoch 選擇結果", 1)[0] and "部署建議" not in console.split("1. Epoch 選擇結果", 1)[0])
    add_check(results, "synthetic_breakout_quality", case_id, "report_without_oos_explains_missing_sections", True, "OOS             : 未納入本次報表" in no_oos_console and "不會輸出 OOS Confusion Matrix 與 Selection/OOS 指標比較" in no_oos_console and "--include-oos" in no_oos_console and "OOS Confusion Matrix" not in no_oos_console.split("注意：", 1)[0] and "Selection 與 OOS 差異" not in no_oos_console.split("注意：", 1)[0])
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_has_semantic_colors", True, "color:#188038" in markdown and "color:#C62828" in markdown and "color:#42A5F5" in markdown and "color:#B06000" in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_console_color_is_opt_in", True, "\x1b[" not in console and "\x1b[96m" in colored_console)

    colored_matrix = colored_console.split("2. Selection Confusion Matrix", 1)[1].split("分類品質", 1)[0]
    colored_matrix_lines = [line for line in colored_matrix.splitlines() if "|" in line]

    def _matrix_cell(line_token: str, column_index: int) -> str:
        line = next(line for line in colored_matrix_lines if line_token in line)
        return line.split("|")[column_index]

    green = "\x1b[92m"
    red = "\x1b[91m"
    reset = "\x1b[0m"
    tp_label_cell = _matrix_cell("正確保留 PASS", 2)
    fn_label_cell = _matrix_cell("錯殺 PASS", 3)
    tp_value_cell = _matrix_cell("TP =", 2)
    fn_value_cell = _matrix_cell("FN =", 3)
    fp_label_cell = _matrix_cell("錯誤保留 REJECT", 2)
    tn_label_cell = _matrix_cell("正確拒絕 REJECT", 3)
    fp_value_cell = _matrix_cell("FP =", 2)
    tn_value_cell = _matrix_cell("TN =", 3)
    original_pass_margin = _matrix_cell("原始PASS =", 4)
    original_reject_margin = _matrix_cell("原始REJECT =", 4)
    model_pass_margin = _matrix_cell("模型PASS =", 2)
    model_reject_margin = _matrix_cell("模型REJECT =", 3)

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_console_confusion_colors_do_not_bleed",
        True,
        (
            all(green in cell and reset in cell and red not in cell for cell in (tp_label_cell, tp_value_cell, tn_label_cell, tn_value_cell))
            and all(red in cell and reset in cell and green not in cell for cell in (fn_label_cell, fn_value_cell, fp_label_cell, fp_value_cell))
            and all(green not in cell and red not in cell for cell in (original_pass_margin, original_reject_margin, model_pass_margin, model_reject_margin))
        ),
    )
    report_markdown_path = resolve_filter_report_markdown_path("/project", "synthetic_quality")
    report_json_path = resolve_filter_report_json_path("/project", "synthetic_quality")
    expected_report_dir_suffix = (
        "outputs",
        "filters",
        "breakout_quality",
        "synthetic_quality",
        DEFAULT_MODEL_ARCHITECTURE,
        "reports",
    )
    report_paths_are_under_filter_reports = (
        tuple(report_markdown_path.parts[-7:])
        == (*expected_report_dir_suffix, "evaluation_report.md")
        and tuple(report_json_path.parts[-7:])
        == (*expected_report_dir_suffix, "evaluation_metrics.json")
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_paths_are_under_filter_reports",
        True,
        report_paths_are_under_filter_reports,
    )
    research_only = build_report_payload(
        metrics_by_split={"selection": _metrics("selection", base=0.45, precision=0.55, acceptance=0.40, recall=0.50, false_reject=0.50)},
        context=context,
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_without_oos_is_research_only", "RESEARCH_ONLY", research_only["conclusion"]["status"])

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
            split_frame = pd.DataFrame(
                [
                    {
                        "ticker": "2330",
                        "date": "2024-01-02",
                        "high_len": high_len,
                        "outer_split": OUTER_SPLIT_SELECTION,
                        "selection_role": SELECTION_ROLE_TRAIN,
                    },
                    {
                        "ticker": "2330",
                        "date": "2024-12-02",
                        "high_len": high_len,
                        "outer_split": OUTER_SPLIT_SELECTION,
                        "selection_role": SELECTION_ROLE_TRAIN,
                    },
                ]
            )
            split_frame.to_csv(paths.split_path, index=False, encoding="utf-8-sig")
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
            split_record = build_file_manifest(paths.split_path)
            split_record.update(
                {
                    "schema_version": SPLIT_ASSIGNMENT_SCHEMA_VERSION,
                    "required_columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "row_count": len(split_frame),
                    "group_key": "ticker/date/high_len",
                    "outer_split_counts": {
                        OUTER_SPLIT_SELECTION: 2,
                        OUTER_SPLIT_OOS: 0,
                        OUTER_SPLIT_OUT_OF_SCOPE: 0,
                    },
                    "selection_role_counts": {
                        SELECTION_ROLE_TRAIN: 2,
                        SELECTION_ROLE_VALIDATION: 0,
                        SELECTION_ROLE_INNER_EMBARGO: 0,
                        SELECTION_ROLE_EMBARGO: 0,
                        SELECTION_ROLE_INVALID: 0,
                        SELECTION_ROLE_NOT_APPLICABLE: 0,
                    },
                }
            )
            outer_policy = {
                "policy_source": "core.walk_forward_policy.synthetic_override",
                "selection_start_date": "2024-01-01",
                "selection_end_date": "2024-12-31",
                "oos_start_date": "2025-01-01",
                "configured_oos_end_date": "2025-12-31",
                "effective_oos_end_date": "2025-12-31",
            }
            outer_policy["policy_fingerprint_sha256"] = compute_outer_policy_fingerprint(outer_policy)
            manifest = {
                "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
                "filter_family": FILTER_FAMILY,
                "filter_id": filter_id,
                "model_architecture": paths.model_architecture,
                "model_spec": get_model_spec(paths.model_architecture).as_manifest_payload(),
                "trainable_parameter_count": 1,
                "sequence_length": int(DEFAULT_LABEL_POLICY.feature_window_bars),
                "model": build_file_manifest(paths.model_path),
                "split_assignments": split_record,
                "outer_oos_policy": outer_policy,
                "feature_columns": list(FEATURE_COLUMNS),
                "context_columns": list(CONTEXT_COLUMNS),
                "policy": DEFAULT_LABEL_POLICY.as_manifest_payload(),
                "score_decision": {
                    "score_column": SCORE_COLUMN,
                    "comparison": SCORE_COMPARISON,
                    "threshold_source": SCORE_THRESHOLD_SOURCE,
                },
                "fixed_evaluation_threshold": 0.50,
                "threshold_policy": {
                    "mode": "fixed_before_oos",
                    "evaluation_threshold": 0.50,
                    "runtime_source": SCORE_THRESHOLD_SOURCE,
                    "optimized_by_train": False,
                    "oos_tuning_allowed": False,
                },
                "training_mode": "fixed_epoch_full_selection",
                "max_epochs": 2,
                "selected_epoch": 2,
                "fixed_epochs": 2,
                "completed_epochs": 2,
                "epoch_selection_source": "fixed_cli_epochs",
                "final_refit_plan": {
                    "mode": "selected_epochs",
                    "selected_optimizer_steps": None,
                    "final_refit_batches_per_epoch": 1,
                    "target_optimizer_steps": 2,
                    "actual_optimizer_steps": 2,
                    "completed_epoch_cycles": 2,
                    "all_eligible_selection_rows_seen_at_least_once": True,
                },
                "class_weight_mode": BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
                "class_weights_reject_pass": [1.0, 1.0],
                "time_weight_mode": BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
                "sample_weight_summaries": {
                    "final_refit": {"mode": BREAKOUT_QUALITY_TIME_WEIGHT_MODE}
                },
                "early_stopping_enabled": False,
                "inner_validation_used": False,
                "training_uses_all_eligible_selection_rows": True,
                "oos_predictions_used_during_training": False,
                "oos_metrics_emitted_by_train": False,
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

            split_frame.loc[1, "selection_role"] = SELECTION_ROLE_VALIDATION
            split_frame.to_csv(paths.split_path, index=False, encoding="utf-8-sig")
            validation_split_record = build_file_manifest(paths.split_path)
            validation_split_record.update(
                {
                    "schema_version": SPLIT_ASSIGNMENT_SCHEMA_VERSION,
                    "required_columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "row_count": len(split_frame),
                    "group_key": "ticker/date/high_len",
                    "outer_split_counts": {
                        OUTER_SPLIT_SELECTION: 2,
                        OUTER_SPLIT_OOS: 0,
                        OUTER_SPLIT_OUT_OF_SCOPE: 0,
                    },
                    "selection_role_counts": {
                        SELECTION_ROLE_TRAIN: 1,
                        SELECTION_ROLE_VALIDATION: 1,
                        SELECTION_ROLE_INNER_EMBARGO: 0,
                        SELECTION_ROLE_EMBARGO: 0,
                        SELECTION_ROLE_INVALID: 0,
                        SELECTION_ROLE_NOT_APPLICABLE: 0,
                    },
                }
            )
            manifest.update(
                {
                    "split_assignments": validation_split_record,
                    "training_mode": TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
                    "max_epochs": 5,
                    "selected_epoch": 2,
                    "fixed_epochs": 2,
                    "completed_epochs": 2,
                    "final_refit_plan": {
                        "mode": BREAKOUT_QUALITY_FINAL_REFIT_MODE,
                        "selected_optimizer_steps": 2,
                        "final_refit_batches_per_epoch": 1,
                        "target_optimizer_steps": 2,
                        "actual_optimizer_steps": 2,
                        "completed_epoch_cycles": 2,
                        "all_eligible_selection_rows_seen_at_least_once": True,
                    },
                    "epoch_selection_source": "inner_validation_loss",
                    "early_stopping_enabled": True,
                    "early_stopping_patience": 1,
                    "early_stopping_min_delta": 0.0,
                    "inner_validation_used": True,
                    "inner_validation_months": 2,
                    "inner_validation_epoch_selection": {
                        "best_epoch": 2,
                        "completed_epochs": 3,
                        "best_optimizer_steps": 2,
                    },
                }
            )
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            validation_contract = load_model_artifact_contract(
                str(project_root),
                filter_id,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "inner_validation_model_contract_supported",
                TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
                validation_contract.manifest["training_mode"],
            )

            stale_architecture_manifest = dict(manifest)
            stale_architecture = (
                "tiny_cnn_v1"
                if paths.model_architecture != "tiny_cnn_v1"
                else "residual_tcn_v1"
            )
            stale_architecture_manifest["model_architecture"] = stale_architecture
            stale_architecture_manifest["model_spec"] = get_model_spec(
                stale_architecture
            ).as_manifest_payload()
            paths.manifest_path.write_text(
                json.dumps(stale_architecture_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(str(project_root), filter_id)
                stale_architecture_rejected = False
            except ValueError as exc:
                stale_architecture_rejected = (
                    "model_architecture" in str(exc) and "工件路徑" in str(exc)
                )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "stale_model_architecture_is_rejected",
                True,
                stale_architecture_rejected,
            )
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            stale_policy_manifest = dict(manifest)
            stale_policy_manifest["policy"] = {
                **DEFAULT_LABEL_POLICY.as_manifest_payload(),
                "min_mfe_return": 0.99,
            }
            paths.manifest_path.write_text(
                json.dumps(stale_policy_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(str(project_root), filter_id)
                stale_policy_rejected = False
            except ValueError as exc:
                stale_policy_rejected = "model policy" in str(exc)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "stale_model_policy_is_rejected",
                True,
                stale_policy_rejected,
            )
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

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
    _validate_breakout_quality_report_rendering(results, case_id)
    summary["filter_id"] = filter_id
    summary["high_len"] = high_len
    return results, summary


__all__ = [
    "validate_breakout_quality_chronological_embargo_case",
    "validate_breakout_quality_policy_single_source_case",
    "validate_breakout_quality_runtime_artifact_contract_case",
]

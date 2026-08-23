"""Public continuous-ranker training, inference, and metric API.

This module is the stable service boundary shared by event rankers, daily-universal
rankers, point-in-time folds, and multi-seed robustness.  Training implementations
remain single-sourced in ``train_continuous_ranker``; consumers must not depend on
its private helpers or inject the trainer module as an implementation object.
"""

from __future__ import annotations

from services.breakout_quality.train_continuous_ranker import (
    CONDITIONAL_MFE_SAFETY_TRAINING_CONTRACT,
    LISTWISE_TRAINING_CONTRACT,
    PAIRWISE_TRAINING_CONTRACT,
    RANKER_REPORT_JSON_FILENAME,
    RANKER_REPORT_MARKDOWN_FILENAME,
    RANKER_SCHEMA_VERSION,
    RANKER_SCORE_FILENAME,
    RANKER_TARGET_FILENAME,
    build_conditional_targets,
    build_daily_percentile_targets,
    build_pareto_component_percentile_targets,
    pareto_pair_concordance_metrics,
    calculate_spearman,
    daily_rank_metrics,
    daily_top_k_metrics,
    conditional_mfe_safety_metrics,
    fit_final,
    predict_conditional_mfe_safety_scores,
    predict_scores,
    predict_dual_component_r,
    resolve_training_output_paths,
    select_epoch,
    split_metrics,
    training_semantics,
)

__all__ = [
    "CONDITIONAL_MFE_SAFETY_TRAINING_CONTRACT",
    "LISTWISE_TRAINING_CONTRACT",
    "PAIRWISE_TRAINING_CONTRACT",
    "RANKER_REPORT_JSON_FILENAME",
    "RANKER_REPORT_MARKDOWN_FILENAME",
    "RANKER_SCHEMA_VERSION",
    "RANKER_SCORE_FILENAME",
    "RANKER_TARGET_FILENAME",
    "build_conditional_targets",
    "build_daily_percentile_targets",
    "build_pareto_component_percentile_targets",
    "pareto_pair_concordance_metrics",
    "calculate_spearman",
    "daily_rank_metrics",
    "daily_top_k_metrics",
    "conditional_mfe_safety_metrics",
    "fit_final",
    "predict_conditional_mfe_safety_scores",
    "predict_scores",
    "predict_dual_component_r",
    "resolve_training_output_paths",
    "select_epoch",
    "split_metrics",
    "training_semantics",
]

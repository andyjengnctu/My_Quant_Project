"""Stable Strategy Compare mode contracts shared by orchestration and reporting."""

from __future__ import annotations

STRATEGY_COMPARE_SCHEMA_VERSION = 7

COMPARISON_MODE_HARD_FILTER = "hard-filter"
COMPARISON_MODE_SCORE_RANKING = "score-ranking"
COMPARISON_MODES = (COMPARISON_MODE_HARD_FILTER, COMPARISON_MODE_SCORE_RANKING)


def comparison_switch_spec(comparison_mode: str) -> tuple[str, bool, bool]:
    mode = str(comparison_mode)
    if mode == COMPARISON_MODE_HARD_FILTER:
        return "use_breakout_quality_filter", False, True
    if mode == COMPARISON_MODE_SCORE_RANKING:
        return "use_breakout_quality_ranking", False, True
    raise ValueError(f"不支援的 comparison_mode: {comparison_mode}")


def comparison_labels(comparison_mode: str) -> dict[str, str]:
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        return {
            "active_name": "quality_filter",
            "active_title": "Active quality filter",
            "output_dir": "strategy_compare",
            "difference_text": "use_breakout_quality_filter=False vs True",
        }
    if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
        return {
            "active_name": "score_ranking",
            "active_title": "Quality score ranking",
            "output_dir": "strategy_compare_score_ranking",
            "difference_text": "use_breakout_quality_ranking=False vs True（hard filter 兩組皆 False）",
        }
    raise ValueError(f"不支援的 comparison_mode: {comparison_mode}")

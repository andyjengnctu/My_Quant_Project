"""Human-readable reporting for Strategy Compare multi-seed robustness."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from core.strategy_compare_policy import get_strategy_comparison_settings
from core.console_report import render_section, render_table
from core.report_metrics import ROBUSTNESS_ROMD_DISTRIBUTION_METRICS
from core.research_report_contract import strategy_mode_extension_contract
from core.report_style import (
    SIGNAL_NEGATIVE,
    SIGNAL_POSITIVE,
    best_worst_signals,
    signal_for_delta,
    styled_signal,
)
from services.research.strategy_comparison import render_strategy_aggregate_report

def _report_settings_for_contract(contract: dict[str, Any], settings):
    contract_ids = {
        str(dict(item or {}).get("arm_id") or "").strip()
        for item in (*tuple(contract.get("fixed_arms") or ()), *tuple(contract.get("stochastic_arms") or ()))
    }
    contract_ids.discard("")
    if not contract_ids:
        return settings
    missing_ids = contract_ids.difference(settings.arms)
    if missing_ids:
        raise ValueError(
            "robustness report contract引用不存在的arm: " + ", ".join(sorted(missing_ids))
        )
    # Report order belongs to the Compare Suite, not to execution roles.  Fixed/benchmark
    # membership may change without changing the human comparison matrix/order.
    selected = {
        arm.arm_id: replace(arm, enabled=True)
        for arm in settings.enabled_arms
        if arm.arm_id in contract_ids
    }
    selected_ids = set(selected)
    selected_contrasts = {
        contrast_id: contrast
        for contrast_id, contrast in settings.contrasts.items()
        if contrast.enabled and contrast.left in selected_ids and contrast.right in selected_ids
    }
    return replace(settings, arms=selected, contrasts=selected_contrasts)

def _fmt(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"

def _style_text(text: str, signal: str | None, *, target: str) -> str:
    if not signal or text == "-":
        return text
    return styled_signal(text, signal, target=target)

def _format_metric_value(value: Any, metric, *, target: str, signal: str | None = None) -> str:
    text = _fmt(value, digits=int(metric.digits), suffix=str(metric.unit))
    return _style_text(text, signal, target=target)

def _distribution_signals(
    rows: list[dict[str, Any]],
    *,
    metrics,
    identity_key: str,
) -> dict[str, dict[str, str]]:
    return {
        metric.key: best_worst_signals(
            {str(row[identity_key]): row.get(metric.key) for row in rows},
            preference=str(metric.preference),
        )
        for metric in metrics
    }

def _right_minus_left_signal(value: Any, *, preference: str) -> str | None:
    if preference not in {"higher", "lower"}:
        return None
    return signal_for_delta(value, preference=preference)

def _win_count_signals(right_count: int, left_count: int) -> tuple[str | None, str | None]:
    if right_count == left_count:
        return None, None
    if right_count > left_count:
        return SIGNAL_POSITIVE, SIGNAL_NEGATIVE
    return SIGNAL_NEGATIVE, SIGNAL_POSITIVE

def render_multi_seed_robustness_report(
    summary: dict[str, Any],
    *,
    target: str = "markdown",
) -> str:
    """Render canonical Strategy SOP plus the compact robustness extension.

    The Standard SOP is rendered byte-for-byte through the same aggregate renderer
    used by single-seed Strategy Compare.  Multi-seed presentation is append-only:
    detailed seed/year/attribution evidence stays in the durable JSON/CSV artifacts
    and must not replace or renumber the Standard SOP.
    """

    contract = dict(summary["contract"])
    settings = _report_settings_for_contract(
        contract, get_strategy_comparison_settings(str(contract["profile_id"]))
    )
    common = dict(summary.get("common_strategy_report") or {})
    scenarios = dict(common.get("scenarios") or {})
    r_analysis = list(common.get("r_analysis") or [])
    yearly_by_id = {
        str(arm_id): {int(year): value for year, value in dict(values or {}).items()}
        for arm_id, values in dict(common.get("yearly_by_id") or {}).items()
    }
    if not scenarios:
        raise ValueError("robustness報表缺少canonical核心策略結果")
    if not yearly_by_id:
        raise ValueError("robustness報表缺少canonical年度結果")

    canonical = render_strategy_aggregate_report(
        settings=settings,
        comparison_period=dict(contract.get("comparison_period") or {}),
        fingerprint=str(contract["fingerprint"]),
        scenarios=scenarios,
        diagnostics={"r_analysis": r_analysis},
        yearly_by_id=yearly_by_id,
        target=target,
        title="策略績效比較",
        fingerprint_label="Scientific fingerprint",
        extra_metadata=(
            ("比較階段", str(contract.get("label") or "Multiple-seed robustness")),
            ("Benchmark ID", str(contract.get("benchmark_id") or "-")),
            ("Strategy trials/fold", str(contract.get("strategy_trials_per_fold") or "-")),
            ("Seed generator", str(contract.get("seed_generator_seed") or "-")),
            (
                "Seeds",
                f"{int(contract['seed_count'])}（deterministic generated；不作best-seed選擇）",
            ),
            (
                "Report schema",
                f"{int(summary['schema_version'])}（不影響scientific fingerprint）",
            ),
        ),
    ).rstrip()

    extension = strategy_mode_extension_contract("robustness_stability")
    romd_table, contrast_table = extension.tables
    sections: list[str] = [canonical, render_section(extension.title)]

    # Compact across-seed RoMD distribution.  Full CV/P25/P75/reference-win
    # diagnostics remain in robustness_summary.json and seed_results.csv.
    romd_metric_by_key = {metric.key: metric for metric in ROBUSTNESS_ROMD_DISTRIBUTION_METRICS}
    compact_metric_keys = ("mean", "median", "std", "min", "max")
    compact_metrics = tuple(romd_metric_by_key[key] for key in compact_metric_keys)
    romd_source_rows = [dict(source) for source in summary.get("romd_statistics") or ()]
    romd_signals = _distribution_signals(
        romd_source_rows, metrics=compact_metrics, identity_key="arm_id"
    )
    romd_rows: list[list[str]] = []
    for row in romd_source_rows:
        arm_id = str(row["arm_id"])
        metric_values = [
            _format_metric_value(
                row.get(key),
                romd_metric_by_key[key],
                target=target,
                signal=romd_signals.get(key, {}).get(arm_id),
            )
            for key in compact_metric_keys
        ]
        romd_rows.append([row["name"], str(int(row["n"])), *metric_values])
    sections.append(render_table(list(romd_table.headers), romd_rows))

    paired = list(summary.get("paired_comparisons") or [])
    if not paired:
        legacy_same = summary.get("romd_same_seed_comparison")
        if isinstance(legacy_same, dict):
            paired = [{
                "contrast_id": "legacy_default",
                "left": legacy_same.get("left"),
                "right": legacy_same.get("right"),
                "romd_same_seed": legacy_same,
            }]

    contrast_rows: list[list[str]] = []
    for item in paired:
        same = item.get("romd_same_seed")
        if not isinstance(same, dict):
            continue
        n = int(same["n"])
        right_wins = int(same["right_gt_left_count"])
        left_wins = int(same["left_gt_right_count"])
        ties = int(same["tie_count"])
        right_signal, left_signal = _win_count_signals(right_wins, left_wins)
        contrast_rows.append([
            f"{same['right']} − {same['left']}",
            str(n),
            _style_text(f"{right_wins}/{n}", right_signal, target=target),
            _style_text(f"{left_wins}/{n}", left_signal, target=target),
            f"{ties}/{n}",
            _style_text(
                f"{float(same['right_minus_left_mean']):+.2f}",
                _right_minus_left_signal(same.get("right_minus_left_mean"), preference="higher"),
                target=target,
            ),
            _style_text(
                f"{float(same['right_minus_left_median']):+.2f}",
                _right_minus_left_signal(same.get("right_minus_left_median"), preference="higher"),
                target=target,
            ),
            _fmt(same.get("right_minus_left_std")),
        ])
    if contrast_rows:
        sections.extend([
            render_section("Robustness Extension｜Same-seed Contrasts"),
            render_table(list(contrast_table.headers), contrast_rows),
        ])

    references = dict(contract.get("romd_reference_baselines") or {})
    min_name = str(dict(references.get("min") or {}).get("name") or "Min baseline")
    full_name = str(dict(references.get("full") or {}).get("name") or "Full baseline")
    detail_paths = (
        "`robustness_summary.json`、`seed_results.csv`、`seed_yearly_returns.csv`"
    )
    sections.extend([
        render_section("Robustness Extension｜說明"),
        "\n".join((
            "- 上方Standard SOP完整沿用Strategy Compare canonical renderer；Robustness只追加本extension，不重新定義主報表。",
            f"- 完整CV/P25/P75、all-pairs probability、逐seed attribution、逐年same-seed delta與跨seed年度分布保留於{detail_paths}，不在主報表重複展開。",
            "- 舊robustness工件缺少較新的model prediction／Future Target conversion欄位時，report-only refresh顯示`-`；不得為補報表重新train或strategy replay。",
            "- resolved seeds只用於重現；不得挑best seed或依本報表組seed ensemble。",
            (
                f"- {full_name}／{min_name}及其餘current Compare Suite arms逐benchmark seed使用same-seed strategy params；"
                "有DL dependency的arms再以相同seed訓練模型。"
                if "full" in references
                else f"- {min_name}及其餘current Compare Suite arms逐benchmark seed使用same-seed strategy params。"
            ),
            "- Extending-Window Rolling robustness只評估config既定scientific condition；不得依結果回頭調整training semantics或使用future fold結果擬合當下模型。",
        )),
    ])
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"


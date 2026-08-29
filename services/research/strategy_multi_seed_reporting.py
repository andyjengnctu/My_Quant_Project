"""Human-readable reporting for Strategy Compare multi-seed robustness."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from config.strategy_compare import get_strategy_comparison_settings
from core.console_report import render_section, render_table
from core.report_metrics import (
    ROBUSTNESS_ROMD_DISTRIBUTION_METRICS,
    ROBUSTNESS_SEED_DELTA_METRICS,
    ROBUSTNESS_YEARLY_DELTA_METRICS,
    ROBUSTNESS_YEARLY_DISTRIBUTION_METRICS,
)
from core.report_style import (
    SIGNAL_NEGATIVE,
    SIGNAL_POSITIVE,
    SIGNAL_WARNING,
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

def _direction_signal(selection_delta: float, romd_delta: float) -> str:
    selection_signal = signal_for_delta(selection_delta, preference="higher")
    romd_signal = signal_for_delta(romd_delta, preference="higher")
    if selection_signal == SIGNAL_POSITIVE and romd_signal == SIGNAL_POSITIVE:
        return SIGNAL_POSITIVE
    if selection_signal == SIGNAL_NEGATIVE and romd_signal == SIGNAL_NEGATIVE:
        return SIGNAL_NEGATIVE
    return SIGNAL_WARNING

def render_multi_seed_robustness_report(
    summary: dict[str, Any],
    *,
    target: str = "markdown",
) -> str:
    """Render robustness using the canonical Strategy Compare report format.

    Sections 1-4 come from the exact same top-level Strategy Compare renderer used
    by the configured comparison profile.  Multi-seed-only evidence is appended
    from section 5 onward.
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

    sections: list[str] = [canonical]

    romd_source_rows: list[dict[str, Any]] = []
    for source in summary["romd_statistics"]:
        row = dict(source)
        n = int(row["n"])
        row["beats_min_rate"] = (
            None if row.get("beats_min_count") is None or n <= 0
            else float(row["beats_min_count"]) / float(n)
        )
        row["beats_full_rate"] = (
            None if row.get("beats_full_count") is None or n <= 0
            else float(row["beats_full_count"]) / float(n)
        )
        romd_source_rows.append(row)
    romd_signals = _distribution_signals(
        romd_source_rows,
        metrics=ROBUSTNESS_ROMD_DISTRIBUTION_METRICS,
        identity_key="arm_id",
    )
    romd_rows = []
    romd_metric_by_key = {metric.key: metric for metric in ROBUSTNESS_ROMD_DISTRIBUTION_METRICS}
    for row in romd_source_rows:
        n = int(row["n"])
        arm_id = str(row["arm_id"])
        metric_values = []
        for key in ("mean", "median", "std", "cv", "min", "p25", "p75", "max"):
            metric = romd_metric_by_key[key]
            metric_values.append(
                _format_metric_value(
                    row.get(key),
                    metric,
                    target=target,
                    signal=romd_signals.get(key, {}).get(arm_id),
                )
            )
        beat_values = []
        for count_key, rate_key in (
            ("beats_min_count", "beats_min_rate"),
            ("beats_full_count", "beats_full_rate"),
        ):
            if row.get(count_key) is None:
                beat_values.append("-")
                continue
            text = f"{int(row[count_key])}/{n}"
            beat_values.append(
                _style_text(
                    text,
                    romd_signals.get(rate_key, {}).get(arm_id),
                    target=target,
                )
            )
        romd_rows.append([row["name"], str(n), *metric_values, *beat_values])
    sections.extend([
        render_section("5. RoMD完整統計"),
        render_table(
            ["比較對象", "N", "Mean", "Median", "Std", "CV", "Min", "P25", "P75", "Max", "勝Min", "勝Full"],
            romd_rows,
        ),
    ])

    paired = list(summary.get("paired_comparisons") or [])
    if not paired:
        legacy_same = summary.get("romd_same_seed_comparison")
        if isinstance(legacy_same, dict):
            paired = [{
                "contrast_id": "legacy_default",
                "description": "歷史兩arm robustness同seed比較",
                "left": legacy_same.get("left"),
                "right": legacy_same.get("right"),
                "romd_same_seed": legacy_same,
                "direct_selection_r_same_seed": summary.get("direct_selection_r_same_seed_comparison"),
                "selection_r_to_strategy": summary.get("selection_r_to_strategy_same_seed_translation"),
                "romd_distribution": summary.get("romd_distribution_comparison"),
                "yearly_same_seed": summary.get("yearly_same_seed_comparison") or [],
            }]

    next_section = 6
    if paired:
        sections.append(render_section(f"{next_section}. 設定中的同seed contrasts"))
        paired_blocks: list[str] = []
        for item in paired:
            left = str(item.get("left") or "Left")
            right = str(item.get("right") or "Right")
            block: list[str] = [f"{right} − {left}"]
            description = str(item.get("description") or "").strip()
            if description:
                block.append(f"用途：{description}")

            same = item.get("romd_same_seed")
            if isinstance(same, dict):
                n = int(same["n"])
                right_signal, left_signal = _win_count_signals(
                    int(same["right_gt_left_count"]),
                    int(same["left_gt_right_count"]),
                )
                right_count_text = _style_text(
                    f"{same['right_gt_left_count']}/{n}", right_signal, target=target
                )
                left_count_text = _style_text(
                    f"{same['left_gt_right_count']}/{n}", left_signal, target=target
                )
                delta_parts = []
                for label, key, preference in (
                    ("Mean", "right_minus_left_mean", "higher"),
                    ("Median", "right_minus_left_median", "higher"),
                    ("Std", "right_minus_left_std", "neutral"),
                    ("Min", "right_minus_left_min", "higher"),
                    ("P25", "right_minus_left_p25", "higher"),
                    ("P75", "right_minus_left_p75", "higher"),
                    ("Max", "right_minus_left_max", "higher"),
                ):
                    value = same.get(key)
                    signal = _right_minus_left_signal(value, preference=preference)
                    delta_parts.append(
                        f"{label} {_style_text(_fmt(value), signal, target=target)}"
                    )
                block.extend([
                    f"RoMD：{same['right']} > {same['left']} {right_count_text}；"
                    f"{same['left']} > {same['right']} {left_count_text}；Tie {same['tie_count']}/{n}。",
                    "ΔRoMD " + "；".join(delta_parts) + "。",
                ])

            direct_same = item.get("direct_selection_r_same_seed")
            if isinstance(direct_same, dict):
                n = int(direct_same["n"])
                right_signal, left_signal = _win_count_signals(
                    int(direct_same["right_gt_left_count"]),
                    int(direct_same["left_gt_right_count"]),
                )
                right_count_text = _style_text(
                    f"{direct_same['right_gt_left_count']}/{n}", right_signal, target=target
                )
                left_count_text = _style_text(
                    f"{direct_same['left_gt_right_count']}/{n}", left_signal, target=target
                )
                mean_text = _style_text(
                    _fmt(direct_same.get("right_minus_left_mean"), suffix=" R"),
                    _right_minus_left_signal(
                        direct_same.get("right_minus_left_mean"), preference="higher"
                    ),
                    target=target,
                )
                median_text = _style_text(
                    _fmt(direct_same.get("right_minus_left_median"), suffix=" R"),
                    _right_minus_left_signal(
                        direct_same.get("right_minus_left_median"), preference="higher"
                    ),
                    target=target,
                )
                block.extend([
                    f"DL選擇R：{direct_same['right']} > {direct_same['left']} {right_count_text}；"
                    f"{direct_same['left']} > {direct_same['right']} {left_count_text}；Tie {direct_same['tie_count']}/{n}。",
                    f"ΔDL選擇R Mean {mean_text}；Median {median_text}；"
                    f"Std {_fmt(direct_same['right_minus_left_std'], suffix=' R')}。",
                ])

            translation = item.get("selection_r_to_strategy")
            if isinstance(translation, dict):
                pair_rows = []
                delta_metric_by_key = {
                    metric.key: metric for metric in ROBUSTNESS_SEED_DELTA_METRICS
                }
                for row in translation.get("seed_rows") or []:
                    selection_delta = float(row["right_minus_left_direct_selection_r"])
                    romd_delta = float(row["right_minus_left_return_over_max_drawdown"])
                    if selection_delta > 0 and romd_delta > 0:
                        verdict = "ranking↑／RoMD↑"
                    elif selection_delta > 0:
                        verdict = "ranking↑／RoMD↓"
                    elif romd_delta > 0:
                        verdict = "ranking↓／RoMD↑"
                    else:
                        verdict = "ranking↓／RoMD↓"
                    values = []
                    for key in (
                        "right_minus_left_direct_selection_r",
                        "right_minus_left_total_return_pct",
                        "right_minus_left_max_drawdown_pct",
                        "right_minus_left_return_over_max_drawdown",
                        "right_minus_left_expected_value_r",
                    ):
                        metric = delta_metric_by_key[key]
                        value = row.get(key)
                        values.append(
                            _format_metric_value(
                                value,
                                metric,
                                target=target,
                                signal=_right_minus_left_signal(
                                    value, preference=metric.preference
                                ),
                            )
                        )
                    pair_rows.append([
                        f"S{int(row['seed_index'])}",
                        *values,
                        _style_text(
                            verdict,
                            _direction_signal(selection_delta, romd_delta),
                            target=target,
                        ),
                    ])
                if pair_rows:
                    block.extend([
                        "",
                        render_table(
                            ["Seed", "ΔDL選擇R", "ΔReturn", "ΔMDD", "ΔRoMD", "ΔEV", "方向"],
                            pair_rows,
                        ),
                    ])
                positive_n = int(translation["selection_r_positive_count"])
                total_n = int(translation["n"])
                translated_n = int(translation["selection_r_positive_romd_positive_count"])
                positive_signal = signal_for_delta(
                    (float(positive_n) / float(total_n)) - 0.5,
                    preference="higher",
                ) if total_n > 0 else None
                translated_signal = signal_for_delta(
                    (float(translated_n) / float(positive_n)) - 0.5,
                    preference="higher",
                ) if positive_n > 0 else None
                concordant_n = int(translation["sign_concordant_count"])
                discordant_n = int(translation["sign_discordant_count"])
                concordant_signal, discordant_signal = _win_count_signals(
                    concordant_n, discordant_n
                )
                spearman_romd = translation.get("selection_r_delta_vs_romd_spearman")
                spearman_return = translation.get("selection_r_delta_vs_return_spearman")
                block.extend([
                    f"ΔDL選擇R>0："
                    f"{_style_text(f'{positive_n}/{total_n}', positive_signal, target=target)}；"
                    f"其中ΔRoMD>0："
                    f"{_style_text(f'{translated_n}/{positive_n if positive_n else 0}', translated_signal, target=target)}。",
                    f"方向一致："
                    f"{_style_text(f'{concordant_n}/{translation['sign_non_tie_n']}', concordant_signal, target=target)}；"
                    f"方向相反："
                    f"{_style_text(f'{discordant_n}/{translation['sign_non_tie_n']}', discordant_signal, target=target)}。",
                    f"Spearman(ΔDL選擇R, ΔRoMD)="
                    f"{_style_text(_fmt(spearman_romd, digits=3), _right_minus_left_signal(spearman_romd, preference='higher'), target=target)}；"
                    f"Spearman(ΔDL選擇R, ΔReturn)="
                    f"{_style_text(_fmt(spearman_return, digits=3), _right_minus_left_signal(spearman_return, preference='higher'), target=target)}。",
                ])

            compare = item.get("romd_distribution")
            if isinstance(compare, dict):
                probability = float(compare["pairwise_left_gt_right_probability"])
                probability_text = _style_text(
                    f"{probability * 100:.2f}%",
                    signal_for_delta(probability - 0.5, preference="higher"),
                    target=target,
                )
                left_n = int(compare.get("left_n") or 0)
                right_n = int(compare.get("right_n") or 0)
                pair_shape = (
                    f"{left_n}×{right_n}={int(compare['pair_count'])} all-pairs"
                    if left_n > 0 and right_n > 0
                    else f"{int(compare['pair_count'])} all-pairs"
                )
                block.append(
                    f"跨seed分布 P({compare['left']} > {compare['right']})="
                    f"{probability_text}"
                    f"（{pair_shape}；非same-seed配對勝率）。"
                )

            annual_pair = list(item.get("yearly_same_seed") or [])
            if annual_pair:
                yearly_delta_metric_by_key = {
                    metric.key: metric for metric in ROBUSTNESS_YEARLY_DELTA_METRICS
                }
                pair_rows = []
                for row in annual_pair:
                    metric_values = []
                    for key in (
                        "right_minus_left_mean",
                        "right_minus_left_median",
                        "right_minus_left_std",
                    ):
                        metric = yearly_delta_metric_by_key[key]
                        value = row.get(key)
                        metric_values.append(
                            _format_metric_value(
                                value,
                                metric,
                                target=target,
                                signal=_right_minus_left_signal(
                                    value, preference=metric.preference
                                ),
                            )
                        )
                    right_signal, left_signal = _win_count_signals(
                        int(row["right_gt_left_count"]),
                        int(row["left_gt_right_count"]),
                    )
                    pair_rows.append([
                        str(row["year"]),
                        str(row["n"]),
                        *metric_values,
                        _style_text(
                            f"{row['right_gt_left_count']}/{row['n']}",
                            right_signal,
                            target=target,
                        ),
                        _style_text(
                            f"{row['left_gt_right_count']}/{row['n']}",
                            left_signal,
                            target=target,
                        ),
                        f"{row['tie_count']}/{row['n']}",
                    ])
                block.extend([
                    "",
                    render_table(
                        ["年度", "N", "Δ右-左 Mean", "Median", "Std", "右勝", "左勝", "Tie"],
                        pair_rows,
                    ),
                ])

            paired_blocks.append("\n".join(block))
        sections.append("\n\n".join(paired_blocks))
        next_section += 1

    yearly = list(summary.get("yearly_statistics") or [])
    if yearly:
        rows = []
        yearly_metric_by_key = {
            metric.key: metric for metric in ROBUSTNESS_YEARLY_DISTRIBUTION_METRICS
        }
        yearly_by_year: dict[int, list[dict[str, Any]]] = {}
        for source in yearly:
            yearly_by_year.setdefault(int(source["year"]), []).append(dict(source))
        for year in sorted(yearly_by_year):
            year_rows = yearly_by_year[year]
            year_signals = _distribution_signals(
                year_rows,
                metrics=ROBUSTNESS_YEARLY_DISTRIBUTION_METRICS,
                identity_key="arm_id",
            )
            for row in sorted(year_rows, key=lambda x: (str(x["type"]), str(x["name"]))):
                arm_id = str(row["arm_id"])
                metric_values = []
                for key in ("mean", "median", "std", "min", "p25", "p75", "max"):
                    metric = yearly_metric_by_key[key]
                    metric_values.append(
                        _format_metric_value(
                            row.get(key),
                            metric,
                            target=target,
                            signal=year_signals.get(key, {}).get(arm_id),
                        )
                    )
                rows.append([
                    str(row["year"]) + ("" if row.get("is_complete_year") else "*"),
                    row["name"],
                    row["type"],
                    str(row["n"]),
                    *metric_values,
                ])
        sections.extend([
            render_section(f"{next_section}. 歷年報酬跨seed完整統計"),
            render_table(
                ["年度", "比較對象", "類型", "N", "Mean", "Median", "Std", "Min", "P25", "P75", "Max"],
                rows,
            ) + "\n* 非完整年度。",
        ])
        next_section += 1

    references = dict(contract.get("romd_reference_baselines") or {})
    min_name = str(dict(references.get("min") or {}).get("name") or "Min baseline")
    full_name = str(dict(references.get("full") or {}).get("name") or "Full baseline")
    sections.extend([
        render_section(f"{next_section}. 限制"),
        "\n".join((
            "- 前四張表直接重用Strategy Compare canonical aggregate renderer；Multi-seed arm顯示per-seed canonical metric的Mean，Fixed arm顯示正式baseline值。",
            "- Multi-seed專屬分布表同樣使用core/report_style.py：有明確方向的metric在同欄可比較arm間只標綠＝最佳、紅＝最差、其餘白；same-seed右減左欄位則依共用metric direction判讀，ΔMDD採lower-is-better，N／Type／Tie等中性欄不硬判。",
            "- 舊robustness工件若未永久保存model prediction／Future Target conversion欄位，report-only refresh會顯示`-`，不為補報表重訓或重跑strategy replay。",
            "- resolved seeds只用於重現；不得挑best seed或依本報表組seed ensemble。",
            (
                f"- {full_name}／{min_name}及其餘current Compare Suite arms全部逐benchmark seed使用same-seed strategy params；"
                "有DL dependency的arms再以相同seed訓練模型。"
                if "full" in references
                else f"- {min_name}及其餘current Compare Suite arms全部逐benchmark seed使用same-seed strategy params。"
            ),
            "- 同一DL source／seed只訓練一次，允許fan-out到不同runtime selector replay；此reuse不改變模型scientific condition。",
            "- Extending-Window Rolling robustness只評估config既定scientific condition；不得依結果回頭調整training semantics或使用future fold結果擬合當下模型。",
        )),
    ])
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"

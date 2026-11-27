"""Canonical Model SOP views and Console/Markdown rendering.

Consumes approved report contracts and already validated scientific evidence.
Training, workflow planning, artifact publication and model selection stay with
their application/producer owners.
"""
from __future__ import annotations

import math
from core.console_report import console_color_enabled, paint, render_section, render_table
from core.research_report_contract import (
    format_contract_value, mode_extension_contract, section_contract, table_contract,
)
from core.report_style import best_worst_signals, markdown_tone, signal_for_delta, styled_signal
from services.breakout_quality.model_report_extensions import build_core_model_report_extensions


def _fmt_simple_metric(value, *, digits: int = 4, percent: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if percent:
        return f"{number * 100.0:.2f}%"
    return f"{number:.{digits}f}"


def _model_sop_view(payload: dict, *, include_legacy_extension_evidence: bool = False) -> dict:
    """Project the canonical M1-M4 SOP. Build legacy diagnostics only for explicit consumers.

    The v8/v12 archived machine payload still includes separately versioned
    diagnostic evidence for backwards compatibility; the public renderer and
    its completeness checks must not compute or require that evidence.
    Removing those historical machine fields requires a new report contract.
    """

    source_payload = dict(payload or {})
    standard = dict(source_payload.get("standard_model_sop") or {})
    if not standard and str(source_payload.get("schema") or "").startswith("standard_model_sop_v"):
        standard = source_payload
    metrics = dict(standard.get("split_metrics") or {})
    if not metrics:
        return {}
    evaluation_mode = str(standard.get("evaluation_mode") or "forward_oos").strip().lower()
    rolling_oos = evaluation_mode == "rolling_oos"
    oos_scope_label = "Rolling OOS" if rolling_oos else "Forward OOS"
    daily_universal = bool(metrics.get("breakout_candidate_oos"))
    split_names = (
        ["validation", "oos", "breakout_candidate_oos"]
        if daily_universal
        else ["validation", "selection", "oos"]
    )

    learnability_rows = []
    for name in split_names:
        row = dict(metrics.get(name) or {})
        top_value = row.get("top_score_decile_raw_target_mean")
        bottom_value = row.get("bottom_score_decile_raw_target_mean")
        learnability_rows.append({
            "split": name,
            "group_count": int(row.get("group_count", 0) or 0),
            "mean_daily_spearman": row.get("mean_daily_spearman"),
            "global_spearman_vs_raw_target": row.get("global_spearman_vs_raw_target"),
            "pairwise_concordance": row.get("pairwise_concordance"),
            "top_score_decile_raw_target_mean": top_value,
            "bottom_score_decile_raw_target_mean": bottom_value,
            "top_bottom_raw_target_gap": (
                None
                if top_value is None or bottom_value is None
                else float(top_value) - float(bottom_value)
            ),
        })

    generalization_rows = []
    val = dict(metrics.get("validation") or {})
    oos = dict(metrics.get("oos") or {})
    breakout = dict(metrics.get("breakout_candidate_oos") or {})

    def top_bottom(row: dict) -> float | None:
        top = row.get("top_score_decile_raw_target_mean")
        bottom = row.get("bottom_score_decile_raw_target_mean")
        return None if top is None or bottom is None else float(top) - float(bottom)

    def delta(left, right):
        return None if left is None or right is None else float(right) - float(left)

    if val and oos:
        generalization_rows.append({
            "comparison": "Validation → OOS",
            "delta_daily_rho": delta(val.get("mean_daily_spearman"), oos.get("mean_daily_spearman")),
            "delta_pair": None if delta(val.get("pairwise_concordance"), oos.get("pairwise_concordance")) is None else delta(val.get("pairwise_concordance"), oos.get("pairwise_concordance")) * 100.0,
            "delta_top_bottom": delta(top_bottom(val), top_bottom(oos)),
        })
    if oos and daily_universal and breakout:
        generalization_rows.append({
            "comparison": "Global OOS → Breakout OOS",
            "delta_daily_rho": delta(oos.get("mean_daily_spearman"), breakout.get("mean_daily_spearman")),
            "delta_pair": None if delta(oos.get("pairwise_concordance"), breakout.get("pairwise_concordance")) is None else delta(oos.get("pairwise_concordance"), breakout.get("pairwise_concordance")) * 100.0,
            "delta_top_bottom": delta(top_bottom(oos), top_bottom(breakout)),
        })

    # Historical Extension evidence is not a Standard SOP obligation.
    # Only an explicit read-only caller requests this compatibility projection.
    ranking = None
    extensions: list[dict] = []
    if include_legacy_extension_evidence:
        from services.audit.model_extension_compat import archived_model_evidence

        ranking, extensions = archived_model_evidence(
            source_payload,
            metrics=metrics,
            split_names=split_names,
            daily_universal=daily_universal,
            core_extensions=build_core_model_report_extensions(
                source_payload, oos_scope_label=oos_scope_label,
            ),
            delta=delta,
            fmt_simple_metric=_fmt_simple_metric,
        )

    alignment_eval = dict(standard.get("upside_downside_alignment_evaluation") or {})
    alignment_rows = []
    top_tail_rows = []

    # The Standard report consumes the canonical stored per-stock outcome evidence.
    # Its two scopes are explicit: Global OOS and Breakout OOS. Validation remains
    # required for M1/M2 completeness but is not a model-comparison result row.
    alignment_scopes = (
        ("Validation", "validation"),
        (oos_scope_label, "oos"),
        ("Breakout OOS", "breakout_candidate_oos"),
    )
    for scope_label, scope_key in alignment_scopes:
        row = dict(alignment_eval.get(scope_key) or {})
        if not row:
            continue
        top = dict(row.get("top_10pct") or {})
        alignment_rows.append({
            "split": scope_label,
            "target_to_full_mfe_daily_spearman": row.get("target_to_full_mfe_daily_spearman"),
            "target_to_safety_daily_spearman": row.get("target_to_low_adverse_daily_spearman"),
            "score_to_full_mfe_daily_spearman": row.get("score_to_full_mfe_daily_spearman"),
            "score_to_safety_daily_spearman": row.get("score_to_low_adverse_daily_spearman"),
        })
        top_tail_rows.append({
            "split": scope_label,
            "top10_n": top.get("n"),
            "top10_mfe_pct": top.get("mfe_pct_mean"),
            "top10_adverse_pct": top.get("adverse_pct_mean"),
            "top10_joint_quality": top.get("joint_quality_pct"),
        })

    evidence = [
        ("Learnability", "AVAILABLE" if metrics.get("oos") else "N/A"),
        ("Generalization", "AVAILABLE" if (metrics.get("validation") and metrics.get("oos") and metrics.get("breakout_candidate_oos")) else "N/A"),
        ("Upside / Downside Alignment", "AVAILABLE" if alignment_rows else "N/A"),
        ("Joint MFE / Adverse Quality", "AVAILABLE" if top_tail_rows else "N/A"),
        ("Breakout application slice", "AVAILABLE" if metrics.get("breakout_candidate_oos") else "N/A"),
        ("Ranking / Boundary", "AVAILABLE" if ranking is not None else "N/A"),
    ]


    return {
        "split_names": split_names,
        "learnability": learnability_rows,
        "generalization": generalization_rows,
        "ranking": ranking,
        "evidence": evidence,
        "upside_downside_alignment": alignment_rows,
        "joint_mfe_adverse_quality": top_tail_rows,
        "extensions": extensions,
        "evaluation_mode": evaluation_mode,
        "rolling_specific": dict((standard.get("mode_extensions") or {}).get("rolling") or {}),
        "robustness_specific": dict((standard.get("mode_extensions") or {}).get("robustness") or {}),
        "comparison_extension_aggregation": dict(source_payload.get("comparison_extension_aggregation") or {}),
    }


_MODEL_STANDARD_MACHINE_VIEW_KEYS = (
    "learnability", "generalization", "upside_downside_alignment",
    "joint_mfe_adverse_quality", "evaluation_mode", "rolling_specific", "robustness_specific",
)


def _standard_model_machine_view(payload: dict) -> dict:
    view = _model_sop_view(payload, include_legacy_extension_evidence=False)
    if not view:
        raise ValueError("Model Standard SOP missing canonical view")
    if any(key not in view for key in _MODEL_STANDARD_MACHINE_VIEW_KEYS):
        raise ValueError("Model Standard SOP missing approved view fields")
    return {key: view[key] for key in _MODEL_STANDARD_MACHINE_VIEW_KEYS}


def _render_model_contract_table(
    table,
    rows,
    *,
    target: str,
    scope_styler=None,
    delta_style: bool = False,
    best_worst_style: bool = False,
    best_worst_group_keys: tuple[str, ...] = (),
) -> str:
    comparison_signals: dict[str, dict[str, str]] = {}
    if best_worst_style:
        for column in table.columns:
            if column.preference not in {"higher", "lower"}:
                continue
            grouped_values: dict[tuple[str, ...], dict[str, object]] = {}
            for index, row in enumerate(rows):
                group_key = tuple(str(row.get(key, "")) for key in best_worst_group_keys)
                grouped_values.setdefault(group_key, {})[str(index)] = row.get(
                    f"__compare__{column.key}", row.get(column.key)
                )
            signals: dict[str, str] = {}
            for values in grouped_values.values():
                signals.update(best_worst_signals(values, preference=column.preference))
            comparison_signals[column.key] = signals

    rendered = []
    for index, row in enumerate(rows):
        cells = []
        for column in table.columns:
            value = row.get(column.key)
            text = format_contract_value(column, value)
            if column.key in {"model", "model_id"}:
                text = _model_identity_text(text, target=target)
            if column.key == "split" and scope_styler is not None:
                text = scope_styler(text)
            if delta_style and column.preference in {"higher", "lower"} and value is not None:
                signal = signal_for_delta(value, preference=column.preference)
                text = styled_signal(
                    text, signal, target=target,
                    enabled=console_color_enabled() if target == "console" else None,
                    bold=True,
                )
            elif best_worst_style:
                signal = comparison_signals.get(column.key, {}).get(str(index))
                if signal is not None:
                    text = styled_signal(
                        text, signal, target=target,
                        enabled=console_color_enabled() if target == "console" else None,
                        bold=True,
                    )
            cells.append(text)
        rendered.append(tuple(cells))
    if target == "console":
        return render_table(table.headers, rendered, alignments=table.alignments)
    header = "| " + " | ".join(table.headers) + " |"
    separator = "|" + "|".join("---" if align == "left" else "---:" for align in table.alignments) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rendered]
    return "\n".join([header, separator, *body])


def _model_comparison_section_title(section_id: str, *, suffix: str = "") -> str:
    section = section_contract("model.standard_comparison", section_id)
    base = f"MODEL SOP | {section.number}. {section.title}"
    return base + (f" | {suffix}" if suffix else "")


def _render_model_comparison_contract_table(
    section_id: str,
    table_id: str,
    rows: list[dict],
    *,
    target: str,
    scope_styler=None,
    delta_style: bool = False,
    best_worst_style: bool = False,
) -> str:
    return _render_model_contract_table(
        table_contract("model.standard_comparison", section_id, table_id),
        rows,
        target=target,
        scope_styler=scope_styler,
        delta_style=delta_style,
        best_worst_style=best_worst_style,
    )


def _standard_model_comparison_rows(views: list[dict]) -> dict[str, object]:
    """Return only four user-approved model SOP sections from canonical source."""
    keys = ("learnability", "generalization", "upside_downside_alignment", "joint_mfe_adverse_quality")
    rows: dict[str, object] = {key: [] for key in keys}
    for item in views:
        for key in keys:
            rows[key].extend(
                {"model": str(item["model_id"]), **dict(row)}
                for row in list(dict(item["view"]).get(key) or [])
            )
    return rows


def _rolling_specific_comparison_rows(views: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in views:
        model = str(item.get("model_id") or "-")
        view = dict(item.get("view") or {})
        rolling = dict(view.get("rolling_specific") or {})
        if not rolling:
            continue
        direction = dict(rolling.get("direction_summary") or {})
        drift = dict(rolling.get("fold_drift") or {})
        valid_years = int(direction.get("valid_year_count", 0) or 0)
        positive_rho = int(direction.get("positive_spearman_year_count", 0) or 0)
        positive_spread = int(direction.get("positive_spread_year_count", 0) or 0)
        rows.append({
            "model": model,
            "fold_count": int(rolling.get("fold_count", 0) or 0),
            "fold_months": int(rolling.get("fold_months", 0) or 0),
            "valid_year_count": valid_years,
            "positive_rho_years": f"{positive_rho} / {valid_years}",
            "positive_spread_years": f"{positive_spread} / {valid_years}",
            "max_adjacent_mean_shift": drift.get("max_adjacent_mean_shift_in_pooled_std"),
            "drift_flag": "YES" if bool(drift.get("drift_flag")) else "NO",
        })
    return rows


def _robustness_specific_comparison_rows(views: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in views:
        model = str(item.get("model_id") or "-")
        view = dict(item.get("view") or {})
        robustness = dict(view.get("robustness_specific") or {})
        if not robustness:
            continue
        oos = next(
            (dict(row) for row in list(view.get("learnability") or []) if str(row.get("split")) == "oos"),
            {},
        )
        joint_rows = [
            dict(row) for row in list(view.get("joint_mfe_adverse_quality") or [])
            if str(row.get("split") or "").strip().lower()
            in {"oos", "forward oos", "rolling oos"}
        ]
        if len(joint_rows) != 1:
            raise ValueError(f"Robustness Model SOP需要唯一OOS Joint Quality: {model}")
        joint = joint_rows[0]
        quality = joint.get("top10_joint_quality")
        if not isinstance(quality, (int, float)) or not math.isfinite(float(quality)):
            raise ValueError(f"Robustness Model SOP缺少有效的OOS Joint Quality: {model}")
        rows.append({
            "model": model,
            "seed_count": int(robustness.get("seed_count", 0) or 0),
            "daily_rho_mean": oos.get("mean_daily_spearman"),
            "daily_rho_std": robustness.get("oos_daily_rho_std"),
            "pair_mean": oos.get("pairwise_concordance"),
            "pair_std": robustness.get("oos_pair_std"),
            "top_bottom_mean": oos.get("top_bottom_raw_target_gap"),
            "top_bottom_std": robustness.get("oos_top_bottom_std"),
            "joint_quality_mean": joint.get("top10_joint_quality"),
            "joint_quality_std": robustness.get("oos_top10_joint_quality_std"),
        })
    return rows


def render_model_comparison(views: list[dict], *, target: str) -> str:
    """Render the common Standard SOP for multiple models.

    [1][4] does not render Validation as a raw split table. For each scope-bearing
    Standard section, one numbered section title is followed by Forward OOS and
    Breakout slice tables. Generalization uses two independent transition tables:
    Validation → OOS and OOS → Breakout slice. The common scorecard remains
    invariant. Model-specific analyses remain in separate read-only Audits.
    """

    rows = _standard_model_comparison_rows(views)
    color = console_color_enabled()
    rolling_oos = all(
        str(dict(item.get("view") or {}).get("evaluation_mode") or "forward_oos") == "rolling_oos"
        for item in views
    )

    def section(section_id: str) -> str:
        title = _model_comparison_section_title(section_id)
        if target == "console":
            return render_section(paint(title, "cyan", enabled=color, bold=True))
        return f"## {markdown_tone(title, 'blue', bold=True)}"

    def scope_heading(label: str) -> str:
        # Scope / transition labels are secondary headings. Keep the main section
        # light-blue and use a dedicated light-yellow presentation tone here so
        # warning-yellow semantics remain unchanged.
        if target == "console":
            return paint(str(label), "light_yellow", enabled=color, bold=True)
        return f"### {markdown_tone(label, 'light_yellow', bold=True)}"

    def split_matches(row: dict, aliases: set[str]) -> bool:
        normalized = str(row.get("split") or "").strip().lower()
        return normalized in {alias.strip().lower() for alias in aliases}

    scope_specs = (
        ("Global", {"oos", "forward oos", "rolling oos"}),
        ("Breakout", {"breakout_candidate_oos", "breakout slice", "breakout oos"}),
    )
    parts: list[str] = []

    def append_scoped_section(section_id: str, table_id: str, source_rows: list[dict]) -> None:
        scoped = []
        for scope_label, aliases in scope_specs:
            table_rows = [dict(row) for row in source_rows if split_matches(dict(row), aliases)]
            if table_rows:
                scoped.append((scope_label, table_rows))
        if not scoped:
            return
        parts.append(section(section_id))
        for scope_label, table_rows in scoped:
            parts.append(scope_heading(scope_label))
            parts.append(_render_model_comparison_contract_table(
                section_id,
                table_id,
                table_rows,
                target=target,
                # The same approved higher/lower preference owns M1, M3 and M4;
                # neutral columns and across-model ties remain uncolored.
                best_worst_style=True,
            ))

    # 1. Learnability
    append_scoped_section("learnability", "learnability", list(rows.get("learnability") or []))

    # 2. Generalization. One section title, then two independent transition tables.
    generalization_rows = [dict(row) for row in list(rows.get("generalization") or [])]
    transition_specs = ("Validation → OOS", "Global OOS → Breakout OOS")
    scoped_generalization = []
    for transition_label in transition_specs:
        table_rows = [
            dict(row) for row in generalization_rows
            if str(row.get("comparison") or "").strip() == transition_label
        ]
        if table_rows:
            scoped_generalization.append((transition_label, table_rows))
    if scoped_generalization:
        parts.append(section("generalization"))
        for transition_label, table_rows in scoped_generalization:
            parts.append(scope_heading(transition_label))
            parts.append(_render_model_comparison_contract_table(
                "generalization",
                "generalization",
                table_rows,
                target=target,
                # Generalization compares models within the same transition.
                # Ranking uses each metric's canonical higher/lower preference.
                best_worst_style=True,
            ))

    # 3. Upside / Downside Alignment
    append_scoped_section(
        "upside_downside_alignment",
        "upside_downside_alignment",
        list(rows.get("upside_downside_alignment") or []),
    )

    # 4. Joint MFE / Adverse Quality, with the same OOS scopes for every DL.
    append_scoped_section(
        "joint_mfe_adverse_quality", "joint_mfe_adverse_quality",
        list(rows.get("joint_mfe_adverse_quality") or []),
    )
    # Model-specific comparisons belong to a separate, hypothesis-bound Audit.
    # Never append temporary extension tables to the persistent SOP.

    # Rolling / robustness evidence remains available without changing M1-M4.
    if rolling_oos:
        rolling_rows = _rolling_specific_comparison_rows(views)
        if rolling_rows:
            extension_title = "Rolling Stability | Fold / Year"
            if target == "console":
                parts.append(render_section(paint(extension_title, "cyan", enabled=color, bold=True)))
            else:
                parts.append(f"## {markdown_tone(extension_title, 'blue', bold=True)}")
            parts.append(_render_model_contract_table(
                mode_extension_contract("rolling_stability").tables[1],
                rolling_rows, target=target, best_worst_style=True,
            ))

    robustness_rows = _robustness_specific_comparison_rows(views)
    if robustness_rows:
        extension_title = "Across-seed Stability"
        if target == "console":
            parts.append(render_section(paint(extension_title, "cyan", enabled=color, bold=True)))
        else:
            parts.append(f"## {markdown_tone(extension_title, 'blue', bold=True)}")
        robustness_contract = mode_extension_contract("robustness_stability")
        parts.append(_render_model_contract_table(
            robustness_contract.tables[0],
            robustness_rows, target=target, best_worst_style=True,
        ))


    separator = "\n" if target == "console" else "\n\n"
    return separator.join(part for part in parts if part)


def _model_identity_text(value: object, *, target: str = "console") -> str:
    """Render user-facing model identities in the shared light-blue palette."""

    text = str(value)
    if target == "console":
        return paint(text, "cyan", enabled=console_color_enabled(), bold=True)
    return markdown_tone(text, "blue", bold=True)

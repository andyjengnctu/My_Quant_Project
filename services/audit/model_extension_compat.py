"""Archived Model-specific Extension evidence adapters for explicit read-only Audit.

The official Model SOP / Model Comparison publishes only approved M1-M4 and
mode-specific evidence.  Keep historical payload interpretation and rendering
here rather than a parallel path inside the formal Research application.
No CLI registration, fitting, score/materialization or report publication.
"""

from __future__ import annotations

from core.breakout_quality_runtime import TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.console_report import console_color_enabled, paint, render_section
from core.report_style import markdown_tone
from core.research_report_contract import (
    comparison_extension_contract,
    comparison_extension_ids_for_evidence_families,
    extension_contract,
    visible_extension_table,
)
from services.breakout_quality.model_report_extensions import extension_missing_evidence


def archived_model_evidence(
    source_payload: dict, *, metrics: dict, split_names: list[str],
    daily_universal: bool, core_extensions: list[dict],
    delta, fmt_simple_metric,
) -> tuple[dict | None, list[dict]]:
    """Read archived diagnostic payloads without changing approved M1-M4 evidence."""
    standard = dict(source_payload.get("standard_model_sop") or {})
    if not standard and str(source_payload.get("schema") or "").startswith("standard_model_sop_v"):
        standard = source_payload
    training = dict(standard.get("training") or {})
    direct_hmhs_only = (
        training.get("objective") == TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING
    )
    val = dict(metrics.get("validation") or {})
    oos = dict(metrics.get("oos") or {})
    breakout = dict(metrics.get("breakout_candidate_oos") or {})
    joint_min_eval = dict(source_payload.get("safety_raw_mfe_joint_min_evaluation") or {})
    raw_eval = dict(
        joint_min_eval
        or source_payload.get("safety_raw_mfe_hmhs_evaluation")
        or source_payload.get("safety_raw_mfe_evaluation")
        or {}
    )

    # Pre-v8 ranking/boundary diagnostics are retained only for historical
    # machine readers and explicitly requested read-only Audit evidence.
    sample = dict((metrics.get("oos") or {}).get("top_k_quality") or {})
    ranking = None
    if sample:
        ranking = {
            "top_k": int(sample.get("top_k", 0) or 0),
            "boundary_width": int(sample.get("boundary_width", 0) or 0),
            "competition_scope": "只看同日樣本數>K" if daily_universal else "只看候選數>K",
            "rows": [],
        }
        for name in split_names:
            quality = dict((metrics.get(name) or {}).get("top_k_quality") or {})
            ranking["rows"].append({
                "split": name,
                "ndcg_at_k": quality.get("ndcg_at_k"),
                "top_k_raw_target_lift": quality.get("top_k_raw_target_lift"),
                "oracle_top_k_overlap": quality.get("oracle_top_k_overlap"),
                "boundary_concordance": quality.get("boundary_concordance"),
                "boundary_raw_target_gap": quality.get("boundary_raw_target_gap"),
                "competition_date_pool": (
                    f"{int(quality.get('competition_date_count', quality.get('top_k_date_count', 0)) or 0):,} / "
                    f"{int(quality.get('all_date_count', 0) or 0):,}"
                ),
            })

    extensions = list(core_extensions)
    if direct_hmhs_only:
        ext_rows = []
        for label, key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout slice", "breakout_candidate_oos")):
            row = dict(metrics.get(key) or {})
            if not row:
                continue
            top10 = dict(row.get("top_10pct") or {})
            top20 = dict(row.get("top_20pct") or {})
            ext_rows.append({
                "split": label,
                "population_hmhs_pct": row.get("population_hmhs_pct"),
                "pairwise_concordance": row.get("pairwise_concordance"),
                "global_average_precision": row.get("global_average_precision"),
                "mean_daily_average_precision": row.get("mean_daily_average_precision"),
                "top10": f"{fmt_simple_metric(None if top10.get('hmhs_pct') is None else float(top10['hmhs_pct']) / 100.0, percent=True)} / {fmt_simple_metric(top10.get('hmhs_enrichment'))}×",
                "top20": f"{fmt_simple_metric(None if top20.get('hmhs_pct') is None else float(top20['hmhs_pct']) / 100.0, percent=True)} / {fmt_simple_metric(top20.get('hmhs_enrichment'))}×",
            })
        def enrich(row: dict) -> float | None:
            return dict(row.get("top_10pct") or {}).get("hmhs_enrichment")
        ext_gen = []
        if val and oos:
            ext_gen.append({
                "comparison": "Validation → Forward OOS",
                "delta_pair": None if delta(val.get("pairwise_concordance"), oos.get("pairwise_concordance")) is None else delta(val.get("pairwise_concordance"), oos.get("pairwise_concordance")) * 100.0,
                "delta_pr_auc": delta(val.get("global_average_precision"), oos.get("global_average_precision")),
                "delta_top10_enrichment": delta(enrich(val), enrich(oos)),
            })
            if breakout:
                ext_gen.append({
                    "comparison": "Forward OOS → Breakout slice",
                    "delta_pair": None if delta(oos.get("pairwise_concordance"), breakout.get("pairwise_concordance")) is None else delta(oos.get("pairwise_concordance"), breakout.get("pairwise_concordance")) * 100.0,
                    "delta_pr_auc": delta(oos.get("global_average_precision"), breakout.get("global_average_precision")),
                    "delta_top10_enrichment": delta(enrich(oos), enrich(breakout)),
                })
        extensions.append({"id": "direct_hmhs_h_only", "rows": ext_rows, "generalization": ext_gen})

    joint_rows = []
    for scope_label, scope_key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout slice", "breakout_candidate_oos")):
        scope = dict(raw_eval.get(scope_key) or {})
        joint = dict(scope.get("joint_hmhs") or {})
        product = dict(scope.get("joint_product_control") or {})
        if not joint:
            continue
        top10 = dict(joint.get("top_10pct") or {})
        product10 = dict(product.get("top_10pct") or {})
        joint_rows.append({
            "split": scope_label,
            "population_hmhs_pct": joint.get("population_hmhs_pct"),
            "direct_pair": joint.get("pairwise_concordance"),
            "product_pair": product.get("pairwise_concordance"),
            "direct_pr_auc": joint.get("global_average_precision"),
            "product_pr_auc": product.get("global_average_precision"),
            "direct_top10_pct": top10.get("hmhs_pct"),
            "direct_top10_enrichment": top10.get("hmhs_enrichment"),
            "product_top10_pct": product10.get("hmhs_pct"),
            "product_top10_enrichment": product10.get("hmhs_enrichment"),
        })
    if joint_rows:
        extensions.append({"id": "direct_hmhs_joint_retrieval", "rows": joint_rows})

    joint_min_rows = []
    for scope_label, scope_key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout slice", "breakout_candidate_oos")):
        scope = dict(joint_min_eval.get(scope_key) or {})
        joint = dict(scope.get("joint_min") or {})
        if not joint:
            continue
        top10 = dict(joint.get("top_10pct") or {})
        top20 = dict(joint.get("top_20pct") or {})
        joint_min_rows.append({
            "split": scope_label,
            "population_joint_min_mean": joint.get("population_joint_min_mean"),
            "joint_min_daily_rho": joint.get("mean_daily_spearman"),
            "joint_min_pair": joint.get("pairwise_concordance"),
            "top10_joint_min": top10.get("mean_joint_min"),
            "top10_safety": top10.get("mean_safety"),
            "top10_mfe": top10.get("mean_mfe"),
            "top10_hmhs_pct": top10.get("hmhs_pct"),
            "top10_hmhs_enrichment": top10.get("hmhs_enrichment"),
            "top20_joint_min": top20.get("mean_joint_min"),
            "top20_hmhs_enrichment": top20.get("hmhs_enrichment"),
        })
    if joint_min_rows:
        extensions.append({"id": "joint_min_retrieval", "rows": joint_min_rows})

    # Robustness aggregation may provide normalized comparison extensions directly.
    # Merge by extension ID so the same renderer consumes single-seed and multi-seed
    # evidence without a workflow-specific schema branch.
    for raw_extension in list(source_payload.get("comparison_extensions") or []):
        normalized_extension = dict(raw_extension or {})
        extension_id = str(normalized_extension.get("id") or "").strip()
        if not extension_id or extension_id == "truth_prediction_geometry":
            continue
        extensions = [
            extension for extension in extensions
            if str(dict(extension).get("id") or "") != extension_id
        ]
        extensions.append(normalized_extension)
    return ranking, extensions


def historical_comparison_evidence_issues(payload: dict, settings, *, view_builder) -> tuple[str, ...]:
    """Describe missing archived Extension evidence for explicit read-only consumers."""
    recipe = get_continuous_ranker_execution_recipe(str(settings.experiment_profile))
    expected = comparison_extension_ids_for_evidence_families(
        recipe.training_policy.report_evidence_families
    )
    if not expected:
        return ()
    view = view_builder(dict(payload or {}), include_legacy_extension_evidence=True)
    available = {
        str(dict(extension).get("id") or ""): dict(extension)
        for extension in list(view.get("extensions") or [])
    }
    issues: list[str] = []
    for extension_id in expected:
        extension = available.get(extension_id)
        if extension is None:
            issues.append(f"model_extension:{extension_id} missing")
            continue
        issues.extend(
            f"model_extension:{extension_id}:{reason}"
            for reason in extension_missing_evidence(extension)
        )
    return tuple(dict.fromkeys(issues))


def render_historical_comparison_extensions(
    views: list[dict], *, target: str, allowed_ids: tuple[str, ...], render_table,
) -> list[str]:
    """Render archived model-comparison extensions on explicit read-only request.

    Not a second Standard SOP renderer; the caller supplies its existing
    canonical table renderer and this function never publishes a report.

    Scope-bearing extensions are split into independent OOS/Breakout subtables.
    ``comparison_population=all_models`` keeps non-applicable models visible with
    ``-`` cells; otherwise only models exposing that evidence capability appear.
    """

    if not allowed_ids:
        return []
    color = console_color_enabled()

    def extension_heading(extension_id: str) -> str:
        title = f"Model-specific Extension｜{comparison_extension_contract(extension_id).title}"
        if target == "console":
            return render_section(paint(title, "cyan", enabled=color, bold=True))
        return f"## {markdown_tone(title, 'blue', bold=True)}"

    def scope_heading(value: str) -> str:
        if target == "console":
            return paint(str(value), "light_yellow", enabled=color, bold=True)
        return f"### {markdown_tone(value, 'light_yellow', bold=True)}"

    def ordered_scopes(values: list[str]) -> list[str]:
        unique = list(dict.fromkeys(str(value) for value in values if str(value).strip()))
        preferred = ("Forward OOS", "Rolling OOS", "Breakout slice")
        return [value for value in preferred if value in unique] + [
            value for value in unique if value not in preferred
        ]

    all_models = [str(item.get("model_id") or "MODEL") for item in views]
    extension_by_model: dict[str, dict[str, dict]] = {}
    for item in views:
        model_id = str(item.get("model_id") or "MODEL")
        view = dict(item.get("view") or {})
        extension_by_model[model_id] = {
            str(ext.get("id") or ""): dict(ext)
            for ext in list(view.get("extensions") or [])
        }

    parts: list[str] = []
    for extension_id in allowed_ids:
        spec = extension_contract(extension_id)
        present_models = [
            model_id for model_id in all_models
            if extension_id in extension_by_model.get(model_id, {})
        ]
        if not present_models:
            continue
        population = (
            all_models
            if spec.comparison_population == "all_models"
            else present_models
        )
        comparison_tables = {
            table.table_id: table
            for table in comparison_extension_contract(extension_id).tables
        }
        extension_parts: list[str] = []

        if spec.comparison_mode == "row_tables":
            scope_key = str(spec.comparison_scope_key or "")
            for table_id, row_key in spec.comparison_row_keys:
                comparison_table = comparison_tables[f"{table_id}_comparison"]
                source_model_key = comparison_table.columns[0].key
                rows_by_model = {
                    model_id: [
                        dict(row)
                        for row in list(
                            extension_by_model.get(model_id, {})
                            .get(extension_id, {})
                            .get(row_key)
                            or []
                        )
                    ]
                    for model_id in population
                }

                if scope_key:
                    scopes = ordered_scopes([
                        str(row.get(scope_key) or "")
                        for model_rows in rows_by_model.values()
                        for row in model_rows
                    ])
                    for scope in scopes:
                        table_rows: list[dict] = []
                        for model_id in population:
                            matches = [
                                row for row in rows_by_model[model_id]
                                if str(row.get(scope_key) or "") == scope
                            ]
                            if not matches and spec.comparison_population == "all_models":
                                matches = [{}]
                            for row in matches:
                                table_rows.append({source_model_key: model_id, **row})
                        if not table_rows:
                            continue
                        visible_table = visible_extension_table(
                            extension_id, comparison_table, table_rows
                        )
                        # A table with only Model contains no applicable head/evidence.
                        if len(visible_table.columns) <= 1:
                            continue
                        extension_parts.append(scope_heading(scope))
                        extension_parts.append(render_table(
                            visible_table,
                            table_rows,
                            target=target,
                            best_worst_style=True,
                        ))
                else:
                    table_rows = [
                        {source_model_key: model_id, **row}
                        for model_id in population
                        for row in rows_by_model[model_id]
                    ]
                    if table_rows:
                        visible_table = visible_extension_table(
                            extension_id, comparison_table, table_rows
                        )
                        extension_parts.append(render_table(
                            visible_table,
                            table_rows,
                            target=target,
                            best_worst_style=True,
                        ))

        if extension_parts:
            parts.append(extension_heading(extension_id))
            parts.extend(extension_parts)

    return parts


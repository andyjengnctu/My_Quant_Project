"""Canonical persistent Research report schemas and authorization freeze.

Persistent/standard/reusable Research renderers consume the schemas in this
module.  A model-specific extension may add evidence without changing the
standard Model SOP skeleton.  Persistent schema changes require explicit user
authorization under ``doc/PROJECT_SETTINGS.md`` B.18 and must update the approved
fingerprint map in the same authorized change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from statistics import fmean
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ReportColumnContract:
    key: str
    label: str
    digits: int | None = None
    unit: str = ""
    preference: str = "neutral"
    format_kind: str = "text"
    alignment: str = "right"


@dataclass(frozen=True)
class ReportTableContract:
    table_id: str
    columns: tuple[ReportColumnContract, ...]

    @property
    def headers(self) -> tuple[str, ...]:
        return tuple(column.label for column in self.columns)

    @property
    def alignments(self) -> tuple[str, ...]:
        return tuple(column.alignment for column in self.columns)


@dataclass(frozen=True)
class ReportSectionContract:
    section_id: str
    number: int
    title: str
    applicability: str = "all"
    tables: tuple[ReportTableContract, ...] = ()

    @property
    def display_title(self) -> str:
        return f"{self.number}. {self.title}"


@dataclass(frozen=True)
class PersistentReportContract:
    report_id: str
    version: int
    role: str
    menu_path: tuple[str, ...]
    sections: tuple[ReportSectionContract, ...]


@dataclass(frozen=True)
class ModelExtensionContract:
    extension_id: str
    title: str
    applicability: str
    tables: tuple[ReportTableContract, ...] = ()
    # Cross-model comparison is an evidence capability declared on the extension
    # itself.  This prevents the persistent report from maintaining a second
    # extension-ID whitelist that can drift when a new model/evidence family lands.
    comparison_mode: str | None = None
    comparison_row_keys: tuple[tuple[str, str], ...] = ()
    # When set, the scope dimension remains in canonical rows but is rendered as
    # independent OOS/Breakout subtables rather than a repeated table column.
    comparison_scope_key: str | None = None
    # Most extensions show only applicable models.  Multi-head comparison is the
    # deliberate exception: all compared models stay visible and non-applicable
    # head cells render as ``-`` instead of disappearing from the comparison set.
    comparison_population: str = "applicable_models"
    # Optional column groups are part of the persistent contract.  A group may be
    # omitted only when every row is non-applicable/missing for every key in it.
    # This keeps the fixed canonical schema while avoiding permanently-wide empty
    # head families on a particular comparison surface.
    optional_column_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Stable evidence-family handshake with the training-composition owner.
    # A workflow never names model IDs or extension IDs to decide applicability.
    evidence_family: str | None = None
    # Cross-seed aggregation is a scientific property of the extension itself.
    # ``row_mean`` is valid for contract rows; ``single_seed_only`` explicitly
    # forbids inventing an across-seed aggregate for distribution geometry.
    robustness_aggregation: str = "single_seed_only"


@dataclass(frozen=True)
class HeadLearnabilitySemanticContract:
    semantic_id: str
    label: str

    @property
    def metric_keys(self) -> tuple[str, str, str]:
        prefix = str(self.semantic_id)
        return (f"{prefix}_daily_rho", f"{prefix}_global_rho", f"{prefix}_pair")


# Canonical display registry for generic head-level rank learnability.  Runtime
# head applicability is NOT defined here; it comes from the existing
# ContinuousRankerScoreOutputPolicy / evidence payload.  Adding a genuinely new
# comparable head semantic therefore changes one registry, not every renderer.
MODEL_HEAD_LEARNABILITY_SEMANTICS: tuple[HeadLearnabilitySemanticContract, ...] = (
    HeadLearnabilitySemanticContract("raw_safety", "Raw Safety"),
    HeadLearnabilitySemanticContract("raw_mfe", "Raw MFE"),
    HeadLearnabilitySemanticContract("conditional_mfe", "Conditional MFE"),
    HeadLearnabilitySemanticContract("primary_mfe", "Primary MFE"),
    HeadLearnabilitySemanticContract("conditional_safety", "Conditional Safety"),
    HeadLearnabilitySemanticContract("primary_target", "Economic Target"),
    HeadLearnabilitySemanticContract("hs_priority_mfe", "HS-Priority MFE"),
)


def _head_learnability_columns() -> tuple[ReportColumnContract, ...]:
    columns: list[ReportColumnContract] = [C("split", "Split", alignment="left")]
    for semantic in MODEL_HEAD_LEARNABILITY_SEMANTICS:
        daily_key, global_key, pair_key = semantic.metric_keys
        columns.extend((
            C(daily_key, f"{semantic.label} Daily", 4, preference="higher", format_kind="number"),
            C(global_key, f"{semantic.label} Global", 4, preference="higher", format_kind="number"),
            C(pair_key, f"{semantic.label} Pair", 2, "%", "higher", "fraction_pct"),
        ))
    return tuple(columns)


def _head_learnability_optional_groups() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((semantic.semantic_id, semantic.metric_keys) for semantic in MODEL_HEAD_LEARNABILITY_SEMANTICS)


C = ReportColumnContract
T = ReportTableContract
S = ReportSectionContract

MODEL_STANDARD_SOP = PersistentReportContract(
    report_id="model.standard_sop",
    version=9,
    role="persistent_standard_model_sop_all_evaluation_modes",
    menu_path=("Research", "模型訓練／驗證"),
    sections=(
        S("learnability", 1, "Learnability", "all_continuous_dl", (
            T("learnability", (
                C("split", "Split", alignment="left"), C("group_count", "Groups", 0, format_kind="int"),
                C("mean_daily_spearman", "Daily rho", 4, preference="higher", format_kind="number"),
                C("global_spearman_vs_raw_target", "Global rho", 4, preference="higher", format_kind="number"),
                C("pairwise_concordance", "Pair", 2, "%", "higher", "fraction_pct"),
                C("top_score_decile_raw_target_mean", "Top 10% Target", 4, preference="higher", format_kind="number"),
                C("bottom_score_decile_raw_target_mean", "Bottom 10% Target", 4, preference="lower", format_kind="number"),
                C("top_bottom_raw_target_gap", "Top-Bottom Target", 4, preference="higher", format_kind="number"),
            )),
        )),
        S("generalization", 2, "Generalization", "validation_and_oos", (
            T("generalization", (
                C("comparison", "Comparison", alignment="left"),
                C("delta_daily_rho", "Δ Daily rho", 4, preference="higher", format_kind="signed_number"),
                C("delta_pair", "Δ Pair", 2, "pp", "higher", "signed_pp"),
                C("delta_top_bottom", "Δ Top-Bottom", 4, preference="higher", format_kind="signed_number"),
            )),
        )),
        S("upside_downside_alignment", 3, "Upside / Downside Alignment", "daily_universal_mfe_adverse_truth", (
            T("upside_downside_alignment", (
                C("split", "Split", alignment="left"),
                C("target_to_full_mfe_daily_spearman", "Target→MFE rho", 4, preference="higher", format_kind="number"),
                C("target_to_safety_daily_spearman", "Target→Safety rho", 4, preference="higher", format_kind="number"),
                C("score_to_full_mfe_daily_spearman", "Score→MFE rho", 4, preference="higher", format_kind="number"),
                C("score_to_safety_daily_spearman", "Score→Safety rho", 4, preference="higher", format_kind="number"),
            )),
        )),
        S("top_tail_economic_quality", 4, "Top-tail Economic Quality", "daily_universal_mfe_adverse_truth", (
            T("top_tail_economic_quality", (
                C("split", "Split", alignment="left"),
                C("top10_n", "Top10 N", 0, format_kind="int"),
                C("top10_full_mfe_r_mean", "Top10 MFE", 4, "R", "higher", "number"),
                C("top10_low_adverse_r_mean", "Top10 Low-Adverse", 4, "R", "higher", "number"),
                C("top10_high_mfe_pct", "High-MFE", 2, "%", "higher", "pct"),
                C("top10_high_safety_pct", "High-Safety", 2, "%", "higher", "pct"),
                C("top10_hmhs", "HM/HS", preference="higher", alignment="right"),
                C("top10_hmls", "HM/LS", preference="lower", alignment="right"),
                C("top10_lmhs", "LM/HS", preference="lower", alignment="right"),
                C("top10_lmls", "LM/LS", preference="lower", alignment="right"),
            )),
        )),
        S("ranking_boundary", 5, "Ranking / Boundary", "ranking_quality_available", (
            T("ranking_boundary", (
                C("split", "Split", alignment="left"),
                C("ndcg_at_k", "NDCG@K", 4, preference="higher", format_kind="number"),
                C("top_k_raw_target_lift", "Top-K Lift", 4, preference="higher", format_kind="number"),
                C("oracle_top_k_overlap", "Oracle overlap", 2, "%", "higher", "fraction_pct"),
                C("boundary_concordance", "Boundary", 2, "%", "higher", "fraction_pct"),
                C("boundary_raw_target_gap", "Boundary gap", 4, preference="higher", format_kind="number"),
                C("competition_date_pool", "競爭日 / Pool日", alignment="right"),
            )),
        )),
        S("evidence_coverage", 6, "Evidence Coverage", "all_continuous_dl", (
            T("evidence_coverage", (
                C("evidence", "Evidence", alignment="left"), C("status", "Status", alignment="left"),
            )),
        )),
    ),
)

def _model_comparison_table(table: ReportTableContract) -> ReportTableContract:
    """Derive multi-model comparison schema from the Standard Model SOP table SSOT.

    [1][4] renders Forward OOS and Breakout as separate tables, so the Standard
    SOP ``Split`` column is structural/redundant there and is omitted. All
    actual metric columns continue to come directly from the Standard SOP.
    """

    # Split / Comparison are structural labels in [1][4]. Each scope/transition
    # is rendered as its own plain-text subheading above an independent table.
    metric_columns = tuple(
        column for column in table.columns if column.key not in {"split", "comparison"}
    )
    return T(
        table.table_id,
        (C("model", "Model", alignment="left"), *metric_columns),
    )


def _model_comparison_generalization_table() -> ReportTableContract:
    """Pivot the two canonical Generalization transitions into one comparison row.

    The source evidence remains the Standard-SOP ``comparison + three deltas`` rows.
    Cross-model comparison only reshapes those already-canonical deltas so one model is
    one row; it does not recompute any metric from raw scores.
    """

    return T(
        "generalization",
        (
            C("model", "Model", alignment="left"),
            C("validation_to_oos_delta_daily_rho", "Val→OOS Δ Daily rho", 4, preference="higher", format_kind="signed_number"),
            C("validation_to_oos_delta_pair", "Val→OOS Δ Pair", 2, "pp", "higher", "signed_pp"),
            C("validation_to_oos_delta_top_bottom", "Val→OOS Δ Top-Bottom", 4, preference="higher", format_kind="signed_number"),
            C("oos_to_breakout_delta_daily_rho", "OOS→Breakout Δ Daily rho", 4, preference="higher", format_kind="signed_number"),
            C("oos_to_breakout_delta_pair", "OOS→Breakout Δ Pair", 2, "pp", "higher", "signed_pp"),
            C("oos_to_breakout_delta_top_bottom", "OOS→Breakout Δ Top-Bottom", 4, preference="higher", format_kind="signed_number"),
        ),
    )


def _model_comparison_sections() -> tuple[ReportSectionContract, ...]:
    sections: list[ReportSectionContract] = []
    for section in MODEL_STANDARD_SOP.sections:
        # Evidence Coverage remains a single-model SOP diagnostic. The authorized
        # cross-model surface deliberately stops at Ranking / Boundary.
        if section.section_id == "evidence_coverage":
            continue
        tables = (
            (_model_comparison_generalization_table(),)
            if section.section_id == "generalization"
            else tuple(_model_comparison_table(table) for table in section.tables)
        )
        sections.append(S(
            section.section_id,
            section.number,
            section.title,
            section.applicability,
            tables,
        ))
    return tuple(sections)


MODEL_STANDARD_COMPARISON = PersistentReportContract(
    report_id="model.standard_comparison",
    version=14,
    role="persistent_multi_model_comparison_all_evaluation_modes",
    menu_path=("Research", "模型訓練／驗證"),
    sections=_model_comparison_sections(),
)

MODEL_MODE_EXTENSION_SCHEMAS: Mapping[str, ModelExtensionContract] = {
    "rolling_stability": ModelExtensionContract(
        "rolling_stability", "Standard Mode Evidence｜Rolling Stability", "rolling_oos_only",
        (T("rolling_stability", (
            C("metric", "Metric", alignment="left"), C("value", "Value", alignment="right"),
        )),
         T("rolling_stability_comparison", (
            C("model", "Model", alignment="left"),
            C("fold_count", "Fold count", 0, format_kind="int"),
            C("fold_months", "Cadence", 0, "M", format_kind="number"),
            C("valid_year_count", "Valid years", 0, preference="higher", format_kind="int"),
            C("positive_rho_years", "Positive-rho years", alignment="right"),
            C("positive_spread_years", "Positive Top-Bottom years", alignment="right"),
            C("max_adjacent_mean_shift", "Max adjacent score-mean drift", 4, " pooled σ", "lower", "number"),
            C("drift_flag", "Drift flag", alignment="center"),
        ))),
    ),
    "robustness_stability": ModelExtensionContract(
        "robustness_stability", "Standard Mode Evidence｜Across-seed Stability", "robustness_only",
        (T("robustness_stability", (
            C("model", "Model", alignment="left"),
            C("seed_count", "Seeds", 0, format_kind="int"),
            C("daily_rho_mean", "OOS Daily rho mean", 4, preference="higher", format_kind="number"),
            C("daily_rho_std", "OOS Daily rho σ", 4, preference="lower", format_kind="number"),
            C("pair_mean", "OOS Pair mean", 2, "%", "higher", "fraction_pct"),
            C("pair_std", "OOS Pair σ", 2, "%", "lower", "fraction_pct"),
            C("top_bottom_mean", "OOS Top-Bottom mean", 4, preference="higher", format_kind="number"),
            C("top_bottom_std", "OOS Top-Bottom σ", 4, preference="lower", format_kind="number"),
        )),
         T("robustness_extension_capabilities", (
            C("extension", "Model Extension", alignment="left"),
            C("aggregation", "Across-seed aggregation", alignment="left"),
            C("status", "Status", alignment="left"),
        ))),
    ),
}


MODEL_EXTENSION_SCHEMAS: Mapping[str, ModelExtensionContract] = {
    "multi_head_learnability": ModelExtensionContract(
        "multi_head_learnability", "Multi-head Learnability",
        "multi_head_rank_output",
        (T("multi_head_learnability", _head_learnability_columns()),),
        comparison_mode="row_tables",
        comparison_row_keys=(("multi_head_learnability", "rows"),),
        comparison_scope_key="split",
        comparison_population="all_models",
        optional_column_groups=_head_learnability_optional_groups(),
        evidence_family="multi_head_learnability",
        robustness_aggregation="row_mean",
    ),
    "hs_conditional_mfe_gate": ModelExtensionContract(
        "hs_conditional_mfe_gate", "HS Qualification / Conditional-MFE Quality",
        "true_hs_conditional_mfe_duo",
        (
            T("hs_conditional_quality", (
                C("split", "Split", alignment="left"),
                C("pred_hs_true_ls_pct", "Pred-HS true-LS", 2, "%", "lower", "pct"),
                C("true_hs_recall_pct", "True-HS recall", 2, "%", "higher", "pct"),
                C("p45_p55_pair", "P45–P55 Pair", 2, "%", "higher", "fraction_pct"),
                C("actual_hmhs_pct", "Pred-HS HM/HS", 2, "%", "higher", "pct"),
                C("actual_high_mfe_pct", "Pred-HS High-MFE", 2, "%", "higher", "pct"),
                C("actual_mean_mfe_r", "Pred-HS MFE", 3, "R", "higher", "number"),
                C("mean_mfe_gap_r", "Δ MFE vs Oracle", 3, "R", "higher", "signed_number"),
            )),
        ),
        comparison_mode="row_tables",
        comparison_row_keys=(("hs_conditional_quality", "rows"),),
        comparison_scope_key="split",
        evidence_family="hs_conditional_mfe",
        robustness_aggregation="row_mean",
    ),
    "direct_hmhs_joint_retrieval": ModelExtensionContract(
        "direct_hmhs_joint_retrieval", "Direct HM/HS Joint Retrieval",
        "marginal_safety_raw_mfe_plus_direct_hmhs_head",
        (T("joint_retrieval", (
            C("split", "Split", alignment="left"), C("population_hmhs_pct", "HM/HS Pop", 2, "%", format_kind="pct"),
            C("direct_pair", "Direct Pair", 2, "%", "higher", "fraction_pct"), C("product_pair", "Product Pair", 2, "%", "higher", "fraction_pct"),
            C("direct_pr_auc", "Direct PR-AUC", 4, preference="higher", format_kind="number"), C("product_pr_auc", "Product PR-AUC", 4, preference="higher", format_kind="number"),
            C("direct_top10_pct", "Direct Top10", 2, "%", "higher", "pct"), C("direct_top10_enrichment", "Direct ×", 4, preference="higher", format_kind="number"),
            C("product_top10_pct", "Product Top10", 2, "%", "higher", "pct"), C("product_top10_enrichment", "Product ×", 4, preference="higher", format_kind="number"),
        )),),
    ),
    "direct_hmhs_h_only": ModelExtensionContract(
        "direct_hmhs_h_only", "Direct HM/HS H-only Learnability",
        "direct_hmhs_single_head",
        (
            T("h_only_learnability", (
                C("split", "Split", alignment="left"), C("population_hmhs_pct", "HM/HS Pop", 2, "%", format_kind="pct"),
                C("pairwise_concordance", "Pair", 2, "%", "higher", "fraction_pct"), C("global_average_precision", "PR-AUC", 4, preference="higher", format_kind="number"),
                C("mean_daily_average_precision", "Daily PR-AUC", 4, preference="higher", format_kind="number"),
                C("top10", "Top10 / ×", alignment="right"), C("top20", "Top20 / ×", alignment="right"),
            )),
            T("h_only_generalization", (
                C("comparison", "Comparison", alignment="left"), C("delta_pair", "Δ HM/HS Pair", 2, "pp", "higher", "signed_pp"),
                C("delta_pr_auc", "Δ PR-AUC", 4, preference="higher", format_kind="signed_number"), C("delta_top10_enrichment", "Δ Top10×", 4, preference="higher", format_kind="signed_number"),
            )),
        ),
    ),
    "joint_min_retrieval": ModelExtensionContract(
        "joint_min_retrieval", "Continuous Joint-Min Retrieval",
        "marginal_safety_raw_mfe_plus_continuous_joint_min_head",
        (T("joint_min_retrieval", (
            C("split", "Split", alignment="left"),
            C("population_joint_min_mean", "Pop Min", 4, preference="higher", format_kind="number"),
            C("joint_min_daily_rho", "Joint-Min rho", 4, preference="higher", format_kind="number"),
            C("joint_min_pair", "Pair", 2, "%", "higher", "fraction_pct"),
            C("top10_joint_min", "Top10 Min", 4, preference="higher", format_kind="number"),
            C("top10_safety", "Top10 S", 4, preference="higher", format_kind="number"),
            C("top10_mfe", "Top10 MFE", 4, preference="higher", format_kind="number"),
            C("top10_hmhs_pct", "Top10 HM/HS", 2, "%", "higher", "pct"),
            C("top10_hmhs_enrichment", "HM/HS ×", 4, preference="higher", format_kind="number"),
            C("top20_joint_min", "Top20 Min", 4, preference="higher", format_kind="number"),
            C("top20_hmhs_enrichment", "Top20 ×", 4, preference="higher", format_kind="number"),
        )),),
    ),
}

STRATEGY_STANDARD_SOP = PersistentReportContract(
    report_id="strategy.standard_sop", version=1, role="persistent_standard",
    menu_path=("Research", "策略組合比較", "Extending-Window OOS / Rolling"),
    sections=tuple(S(key, i, title) for i, (key, title) in enumerate((
        ("core_performance", "Core Performance"),
        ("trade_quality", "Trade Quality / MFE × Safety（Filled buys）"),
        ("selection_quality", "Selection Quality"),
        ("upside_survival", "Upside Survival / First-Passage"),
        ("capital_execution", "Capital / Execution"),
        ("yearly_results", "Yearly Results"),
        ("execution_summary", "Execution Summary"),
    ), start=1)),
)

OPPORTUNITY_SELECTION_REPORT = PersistentReportContract(
    report_id="audit.opportunity_selection", version=1, role="persistent_reusable",
    menu_path=("Research", "Audit／診斷", "可重複使用的原因分析", "Opportunity／Selection Attribution"),
    sections=(
        S("actual_opportunity",1,"Actual Opportunity Baseline", tables=(T("truth_summary",(
            C("scope","Scope",alignment="left"),C("n","N",0,format_kind="int"),C("rho","Actual S↔MFE Daily rho",4,format_kind="number"),
            C("s5_m5","S5×M5 N/Pop/×Exp"),C("s4plus_m4plus","S4+×M4+ N/Pop/×Exp"),)),)),
        S("selection_funnel",2,"Breakout → Orderable → Planned", tables=(T("geometry",(
            C("cohort","Cohort",alignment="left"),C("n","N",0,format_kind="int"),C("truth_coverage_pct","Truth Cov",2,"%",format_kind="pct"),
            C("high_mfe_high_safety_pct","HM/HS",2,"%","higher","pct"),C("high_mfe_low_safety_pct","HM/LS",2,"%","lower","pct"),
            C("low_mfe_high_safety_pct","LM/HS",2,"%",format_kind="pct"),C("low_mfe_low_safety_pct","LM/LS",2,"%","lower","pct"),
            C("high_mfe_total_pct","High-MFE",2,"%","higher","pct"),C("high_safety_total_pct","High-Safety",2,"%",format_kind="pct"),)),)),
        S("planned_geometry",3,"Final Planned 5×5", tables=(T("planned_delta",(
            C("safety","Treatment-Control",alignment="left"),C("m1","M1",2,"pp",format_kind="signed_pp"),C("m2","M2",2,"pp",format_kind="signed_pp"),
            C("m3","M3",2,"pp",format_kind="signed_pp"),C("m4","M4",2,"pp",format_kind="signed_pp"),C("m5","M5",2,"pp",format_kind="signed_pp"),)),)),
        S("pair_membership",4,"Pair Membership Attribution", tables=(T("pair_cohorts",(
            C("cohort","Cohort",alignment="left"),C("n","N",0,format_kind="int"),C("score_percentile_mean","Score %ile",3,preference="higher",format_kind="number"),
            C("high_mfe_high_safety_pct","HM/HS",2,"%","higher","pct"),C("high_mfe_low_safety_pct","HM/LS",2,"%","lower","pct"),
            C("high_mfe_total_pct","High-MFE",2,"%","higher","pct"),C("high_safety_total_pct","High-Safety",2,"%",format_kind="pct"),)),)),
        S("selection_quality",5,"Selection Quality", tables=(T("selection_quality",(
            C("arm","Arm",alignment="left"),C("selected_target_mean_r","Target mean",3,"R","higher","number"),C("selected_target_percentile","Target %ile",3,preference="higher",format_kind="number"),
            C("target_top_k_retention","Top-K retention",2,"%","higher","pct"),C("target_opportunity_gap_r","Opp gap",3,"R","lower","number"),C("r_conversion_efficiency","RCE",3,preference="higher",format_kind="number"),)),)),
        S("key_evidence",6,"Key Evidence"),
    ),
)

TRADE_OUTCOME_FIRST_PASSAGE_THRESHOLDS_R = (1.0, 2.0, 3.0)

TRADE_OUTCOME_PATH_REPORT = PersistentReportContract(
    report_id="audit.trade_outcome_path", version=1, role="persistent_reusable",
    menu_path=("Research", "Audit／診斷", "可重複使用的原因分析", "Trade Outcome／Path Attribution"),
    sections=(
        S("planned_to_filled",1,"Planned → Filled",tables=(T("planned_to_filled",(
            C("arm","Arm",alignment="left"),C("planned_count","Planned",0,format_kind="int"),C("filled_count","Filled",0,format_kind="int"),C("fill_rate_pct","Fill",2,"%","higher","pct"),C("path_coverage_pct","Path coverage",2,"%","higher","pct"),)),)),
        S("realized_first_passage",2,"Realized Outcome / First Passage",tables=(T("realized_first_passage",(
            C("arm","Arm",alignment="left"),C("realized_mean_r","Avg R",3,"R","higher","number"),C("realized_median_r","Median R",3,"R","higher","number"),
            C("full_horizon_mfe_mean_r","Full MFE",3,"R","higher","number"),C("adverse_to_peak_mean_r","Adverse",3,"R","lower","number"),
            C("first_1r_reached_pct","+1R reached",2,"%","higher","pct"),C("initial_stop_before_1r_pct","Stop before +1R",2,"%","lower","pct"),
            C("first_2r_reached_pct","+2R reached",2,"%","higher","pct"),C("initial_stop_before_2r_pct","Stop before +2R",2,"%","lower","pct"),
            C("first_3r_reached_pct","+3R reached",2,"%","higher","pct"),C("initial_stop_before_3r_pct","Stop before +3R",2,"%","lower","pct"),
        )),)),
        S("pair_cohorts",3,"Pair Cohorts",tables=(T("pair_cohorts",(
            C("cohort","Cohort",alignment="left"),C("filled_count","Filled",0,format_kind="int"),C("realized_mean_r","Avg R",3,"R","higher","number"),
            C("full_horizon_mfe_mean_r","Full MFE",3,"R","higher","number"),C("adverse_to_peak_mean_r","Adverse",3,"R","lower","number"),C("initial_stop_before_2r_pct","Stop before +2R",2,"%","lower","pct"),)),)),
        S("truth_cohort_outcome",4,"Truth Cohort Outcome",tables=(T("truth_cohort_outcome",(
            C("arm","Arm",alignment="left"),C("truth_cohort","Truth cohort",alignment="left"),C("filled_count","N",0,format_kind="int"),C("realized_mean_r","Avg R",3,"R","higher","number"),
            C("full_horizon_mfe_mean_r","Full MFE",3,"R","higher","number"),C("adverse_to_peak_mean_r","Adverse",3,"R","lower","number"),C("initial_stop_before_2r_pct","Stop before +2R",2,"%","lower","pct"),)),)),
        S("key_evidence",5,"Key Evidence"),
    ),
)

PORTFOLIO_DRAWDOWN_REPORT = PersistentReportContract(
    report_id="audit.portfolio_drawdown", version=1, role="persistent_reusable",
    menu_path=("Research", "Audit／診斷", "可重複使用的原因分析", "Portfolio／Drawdown Attribution"),
    sections=(
        S("capital_exposure",1,"Capital / Exposure",tables=(T("capital_exposure",(
            C("arm","Arm",alignment="left"),C("average_exposure_pct","Exposure",2,"%",format_kind="pct"),C("median_stop_distance_pct","Median stop dist",2,"%",format_kind="pct"),
            C("median_reserved_fraction_pct","Reserved/Equity",2,"%",format_kind="pct"),C("mean_chosen_risk_utilization","Risk util",3,format_kind="number"),C("mean_holding_calendar_days","Holding days",1,"d",format_kind="number"),C("round_trip_count","Trades",0,format_kind="int"),)),)),
        S("max_drawdown_mtm",2,"Max Drawdown MTM",tables=(T("max_drawdown_mtm",(
            C("arm","Arm",alignment="left"),C("max_drawdown_pct","MDD",2,"%","lower","pct"),C("peak_date","Peak",alignment="left"),C("trough_date","Trough",alignment="left"),
            C("position_mtm_contribution_pct_peak_equity","ΣPosition MTM / Peak",2,"%",format_kind="pct"),C("reconciliation_delta","Reconcile Δ",3,format_kind="number"),)),)),
        S("position_lifecycle",3,"Position Lifecycle",tables=(T("position_lifecycle",(
            C("arm","Arm",alignment="left"),C("peak_held_count","Peak-held",0,format_kind="int"),C("entered_during_drawdown_count","Entered during DD",0,format_kind="int"),
            C("exited_during_drawdown_count","Exited during DD",0,format_kind="int"),C("trough_held_count","Trough-held",0,format_kind="int"),C("max_same_day_entries","Max same-day entries",0,format_kind="int"),)),)),
        S("drawdown_truth",4,"Drawdown Truth Contribution",tables=(T("drawdown_truth",(
            C("arm","Arm",alignment="left"),C("hmhs_mtm_contribution_pct_peak_equity","HM/HS",2,"%",format_kind="pct"),C("hmls_mtm_contribution_pct_peak_equity","HM/LS",2,"%",format_kind="pct"),
            C("lmhs_mtm_contribution_pct_peak_equity","LM/HS",2,"%",format_kind="pct"),C("lmls_mtm_contribution_pct_peak_equity","LM/LS",2,"%",format_kind="pct"),)),)),
        S("key_evidence",5,"Key Evidence"),
    ),
)

STRATEGY_CONSISTENCY_REPORT = PersistentReportContract(
    report_id="strategy.oos_rolling_consistency", version=1, role="persistent_standard",
    menu_path=("Research", "策略組合比較", "OOS + Rolling 一鍵／Consistency"),
    sections=(
        S("arm_values",1,"Arm-level values",tables=(T("arm_values",(
            C("arm","Arm",alignment="left"),C("metric","Metric",alignment="left"),C("oos","OOS"),C("rolling","Rolling"),)),)),
        S("contrast_consistency",2,"Configured contrast consistency",tables=(T("contrast_consistency",(
            C("contrast","Contrast",alignment="left"),C("metric","Metric",alignment="left"),C("oos_delta","OOS Δ"),C("rolling_delta","Rolling Δ"),C("direction","Direction",alignment="left"),)),)),
    ),
)
STRATEGY_CONSISTENCY_METRIC_KEYS = (
    "total_return_pct", "max_drawdown_pct", "return_over_max_drawdown",
    "expected_value_r", "avg_exposure_pct", "high_mfe_high_safety_pct", "high_mfe_total_pct",
)

PERSISTENT_REPORT_CONTRACTS: Mapping[str, PersistentReportContract] = {
    contract.report_id: contract for contract in (
        MODEL_STANDARD_SOP, MODEL_STANDARD_COMPARISON, STRATEGY_STANDARD_SOP, OPPORTUNITY_SELECTION_REPORT,
        TRADE_OUTCOME_PATH_REPORT, PORTFOLIO_DRAWDOWN_REPORT, STRATEGY_CONSISTENCY_REPORT,
    )
}


def format_contract_value(column: ReportColumnContract, value) -> str:
    """Format one report value using the canonical column contract."""

    if value is None:
        return "-"
    kind = str(column.format_kind)
    if kind == "text":
        return str(value)
    if kind == "int":
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    digits = int(column.digits or 0)
    if kind == "fraction_pct":
        return f"{number * 100.0:.{digits}f}%"
    if kind == "pct":
        return f"{number:.{digits}f}%"
    if kind == "signed_pp":
        return f"{number * 100.0:+.{digits}f}pp" if column.unit == "pp_fraction" else f"{number:+.{digits}f}pp"
    if kind == "signed_number":
        return f"{number:+.{digits}f}{column.unit}"
    return f"{number:.{digits}f}{column.unit}"


def report_contract(report_id: str) -> PersistentReportContract:
    try:
        return PERSISTENT_REPORT_CONTRACTS[str(report_id)]
    except KeyError as exc:
        raise KeyError(f"未知persistent Research report contract: {report_id}") from exc


def section_contract(report_id: str, section_id: str) -> ReportSectionContract:
    for section in report_contract(report_id).sections:
        if section.section_id == str(section_id):
            return section
    raise KeyError(f"{report_id}沒有section: {section_id}")


def table_contract(report_id: str, section_id: str, table_id: str) -> ReportTableContract:
    section = section_contract(report_id, section_id)
    for table in section.tables:
        if table.table_id == str(table_id):
            return table
    raise KeyError(f"{report_id}/{section_id}沒有table: {table_id}")


def column_contract(report_id: str, section_id: str, table_id: str, column_key: str) -> ReportColumnContract:
    table = table_contract(report_id, section_id, table_id)
    for column in table.columns:
        if column.key == str(column_key):
            return column
    raise KeyError(f"{report_id}/{section_id}/{table_id}沒有column: {column_key}")


def _comparison_source_column(table: ReportTableContract) -> ReportColumnContract:
    """Return the source-model dimension without colliding with evidence-internal Model."""

    has_model_column = any(column.key == "model" for column in table.columns)
    return C(
        "source_model" if has_model_column else "model",
        "Source Model" if has_model_column else "Model",
        alignment="left",
    )


def _derive_row_table_comparison_contract(
    extension: ModelExtensionContract,
) -> ModelExtensionContract:
    table_by_id = {table.table_id: table for table in extension.tables}
    tables: list[ReportTableContract] = []
    for table_id, _row_key in extension.comparison_row_keys:
        try:
            base = table_by_id[str(table_id)]
        except KeyError as exc:
            raise KeyError(
                f"{extension.extension_id} comparison row table不存在: {table_id}"
            ) from exc
        scope_key = str(extension.comparison_scope_key or "")
        metric_columns = tuple(
            column for column in base.columns
            if not scope_key or column.key != scope_key
        )
        tables.append(T(
            f"{base.table_id}_comparison",
            (_comparison_source_column(base), *metric_columns),
        ))
    return ModelExtensionContract(
        extension.extension_id,
        extension.title,
        extension.applicability,
        tuple(tables),
        comparison_mode=extension.comparison_mode,
        comparison_row_keys=extension.comparison_row_keys,
        comparison_scope_key=extension.comparison_scope_key,
        comparison_population=extension.comparison_population,
        optional_column_groups=extension.optional_column_groups,
        evidence_family=extension.evidence_family,
        robustness_aggregation=extension.robustness_aggregation,
    )


def _derive_comparison_extension_contract(
    extension: ModelExtensionContract,
) -> ModelExtensionContract:
    if extension.comparison_mode == "row_tables":
        return _derive_row_table_comparison_contract(extension)
    raise ValueError(
        f"未知Model extension comparison_mode: {extension.extension_id}={extension.comparison_mode}"
    )



def visible_extension_table(
    extension_id: str,
    table: ReportTableContract,
    rows: Sequence[Mapping[str, Any]],
) -> ReportTableContract:
    """Project optional extension column groups using only canonical contract rules.

    A head family disappears only when every row has no value for every metric in
    that family.  If at least one model exposes the family, the full family remains
    visible and non-applicable models render ``-`` in those cells.
    """

    extension = extension_contract(extension_id)
    omit_keys: set[str] = set()
    normalized = [dict(row) for row in rows]
    for _group_id, keys in extension.optional_column_groups:
        if all(
            row.get(key) in {None, "", "-"}
            for row in normalized
            for key in keys
        ):
            omit_keys.update(keys)
    if not omit_keys:
        return table
    return T(
        table.table_id,
        tuple(column for column in table.columns if column.key not in omit_keys),
    )


def comparison_extension_ids() -> tuple[str, ...]:
    """Cross-model evidence capabilities, derived from the single extension registry."""

    return tuple(
        extension_id
        for extension_id, extension in MODEL_EXTENSION_SCHEMAS.items()
        if extension.comparison_mode is not None
    )


def comparison_extension_ids_for_evidence_families(
    evidence_families: Sequence[str],
) -> tuple[str, ...]:
    """Resolve cross-model extension IDs from capability families, never model identity."""

    families = {str(value).strip() for value in evidence_families if str(value).strip()}
    return tuple(
        extension_id
        for extension_id, extension in MODEL_EXTENSION_SCHEMAS.items()
        if extension.comparison_mode is not None
        and extension.evidence_family is not None
        and extension.evidence_family in families
    )


def aggregate_robustness_row_extension(
    extension_id: str,
    seed_extensions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate one row-table extension across benchmark seeds by its own contract.

    Identity/text columns must match exactly across seeds. Numeric contract columns
    use an arithmetic mean, matching Standard-SOP robustness semantics.
    """

    extension = extension_contract(extension_id)
    if extension.robustness_aggregation != "row_mean":
        raise ValueError(
            f"Model extension不支援row-mean robustness aggregation: {extension_id}="
            f"{extension.robustness_aggregation}"
        )
    if extension.comparison_mode != "row_tables":
        raise ValueError(f"row-mean robustness只支援row_tables extension: {extension_id}")
    normalized = [dict(value or {}) for value in seed_extensions]
    if len(normalized) < 2:
        raise ValueError("Model extension robustness至少需要兩個seed payload")
    tables = {table.table_id: table for table in extension.tables}
    result: dict[str, Any] = {
        "id": extension_id,
        "robustness_aggregation": "arithmetic_mean_across_benchmark_seeds",
    }
    for table_id, row_key in extension.comparison_row_keys:
        table = tables[table_id]
        identity_columns = tuple(
            column.key for column in table.columns
            if column.format_kind == "text" or column.digits is None
        )
        metric_columns = tuple(column.key for column in table.columns if column.key not in identity_columns)
        per_seed: list[dict[tuple[Any, ...], dict[str, Any]]] = []
        for payload in normalized:
            index: dict[tuple[Any, ...], dict[str, Any]] = {}
            for raw_row in list(payload.get(row_key) or []):
                row = dict(raw_row)
                identity = tuple(row.get(key) for key in identity_columns)
                if identity in index:
                    raise ValueError(f"{extension_id}/{table_id} seed row identity重複: {identity}")
                index[identity] = row
            per_seed.append(index)
        identity_sets = [set(index) for index in per_seed]
        if any(values != identity_sets[0] for values in identity_sets[1:]):
            raise ValueError(f"{extension_id}/{table_id} row identity跨seed不一致")
        rows: list[dict[str, Any]] = []
        for identity in sorted(identity_sets[0], key=lambda value: tuple(str(x) for x in value)):
            row = {key: value for key, value in zip(identity_columns, identity)}
            for key in metric_columns:
                values = [index[identity].get(key) for index in per_seed]
                finite_values = [float(value) for value in values if value is not None]
                row[key] = None if not finite_values else float(fmean(finite_values))
            rows.append(row)
        result[row_key] = rows
    return result


MODEL_COMPARISON_EXTENSION_SCHEMAS: Mapping[str, ModelExtensionContract] = {
    extension_id: _derive_comparison_extension_contract(extension)
    for extension_id, extension in MODEL_EXTENSION_SCHEMAS.items()
    if extension.comparison_mode is not None
}


def extension_contract(extension_id: str) -> ModelExtensionContract:
    try:
        return MODEL_EXTENSION_SCHEMAS[str(extension_id)]
    except KeyError as exc:
        raise KeyError(f"未知Model-specific extension: {extension_id}") from exc


def comparison_extension_contract(extension_id: str) -> ModelExtensionContract:
    try:
        return MODEL_COMPARISON_EXTENSION_SCHEMAS[str(extension_id)]
    except KeyError as exc:
        raise KeyError(f"未知Model comparison extension: {extension_id}") from exc


def mode_extension_contract(extension_id: str) -> ModelExtensionContract:
    try:
        return MODEL_MODE_EXTENSION_SCHEMAS[str(extension_id)]
    except KeyError as exc:
        raise KeyError(f"未知Model mode extension: {extension_id}") from exc


def _persistent_schema_payload(contract: PersistentReportContract) -> dict:
    payload = asdict(contract)
    if contract.report_id == "model.standard_comparison":
        # Comparison extensions are discovered from evidence capability metadata;
        # there is deliberately no second persistent-report extension allow-list.
        payload["model_specific_extensions"] = [
            asdict(comparison_extension_contract(extension_id))
            for extension_id in comparison_extension_ids()
        ]
    return payload


def persistent_report_contract_fingerprint(report_id: str) -> str:
    payload = json.dumps(
        _persistent_schema_payload(report_contract(report_id)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def persistent_report_contract_fingerprints() -> dict[str, str]:
    return {
        report_id: persistent_report_contract_fingerprint(report_id)
        for report_id in sorted(PERSISTENT_REPORT_CONTRACTS)
    }


# User-approved persistent-report freeze.  This map is deliberately separate
# from schema construction.  Updating a persistent schema and this approval map
# is only legal after explicit user authorization under PROJECT_SETTINGS B.18.
APPROVED_PERSISTENT_REPORT_CONTRACT_FINGERPRINTS: Mapping[str, str] = {
    "audit.opportunity_selection": "fcdc3c51c70f74db",
    "audit.portfolio_drawdown": "b30ce69159e1f31a",
    "audit.trade_outcome_path": "c50943f97734da39",
    "model.standard_comparison": "e1a9b52b45992a62",
    "model.standard_sop": "7a2be4dbb363e3fd",
    "strategy.oos_rolling_consistency": "deb471377e80ff80",
    "strategy.standard_sop": "c4e92dcc1e1c731e",
}


def validate_approved_persistent_report_contracts() -> tuple[str, ...]:
    current = persistent_report_contract_fingerprints()
    reasons: list[str] = []
    if set(current) != set(APPROVED_PERSISTENT_REPORT_CONTRACT_FINGERPRINTS):
        reasons.append(
            "approved persistent report set mismatch: "
            f"current={sorted(current)}, approved={sorted(APPROVED_PERSISTENT_REPORT_CONTRACT_FINGERPRINTS)}"
        )
    for report_id, fingerprint in current.items():
        approved = APPROVED_PERSISTENT_REPORT_CONTRACT_FINGERPRINTS.get(report_id)
        if approved != fingerprint:
            reasons.append(
                f"UNAUTHORIZED_PERSISTENT_REPORT_CONTRACT_CHANGE: {report_id} "
                f"current={fingerprint} approved={approved or '-'}"
            )
    return tuple(reasons)


__all__ = [
    "APPROVED_PERSISTENT_REPORT_CONTRACT_FINGERPRINTS",
    "MODEL_EXTENSION_SCHEMAS", "MODEL_MODE_EXTENSION_SCHEMAS", "MODEL_STANDARD_SOP",
    "OPPORTUNITY_SELECTION_REPORT", "PORTFOLIO_DRAWDOWN_REPORT",
    "MODEL_STANDARD_COMPARISON", "PERSISTENT_REPORT_CONTRACTS", "STRATEGY_CONSISTENCY_METRIC_KEYS",
    "STRATEGY_CONSISTENCY_REPORT", "STRATEGY_STANDARD_SOP",
    "TRADE_OUTCOME_FIRST_PASSAGE_THRESHOLDS_R", "TRADE_OUTCOME_PATH_REPORT", "ModelExtensionContract",
    "PersistentReportContract", "ReportColumnContract", "ReportSectionContract",
    "ReportTableContract", "HeadLearnabilitySemanticContract", "MODEL_HEAD_LEARNABILITY_SEMANTICS",
    "aggregate_robustness_row_extension", "column_contract", "comparison_extension_contract", "comparison_extension_ids", "comparison_extension_ids_for_evidence_families", "extension_contract", "mode_extension_contract", "visible_extension_table", "format_contract_value", "persistent_report_contract_fingerprint",
    "persistent_report_contract_fingerprints", "report_contract", "section_contract",
    "table_contract", "validate_approved_persistent_report_contracts",
]

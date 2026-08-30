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
from typing import Mapping


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


C = ReportColumnContract
T = ReportTableContract
S = ReportSectionContract

MODEL_STANDARD_SOP = PersistentReportContract(
    report_id="model.standard_sop",
    version=5,
    role="persistent_standard",
    menu_path=("Research", "模型訓練／驗證", "訓練目前模型 → Forward-OOS 標準模型 SOP 報表"),
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

    metric_columns = tuple(column for column in table.columns if column.key != "split")
    return T(
        table.table_id,
        (C("model", "Model", alignment="left"), *metric_columns),
    )


def _model_comparison_sections() -> tuple[ReportSectionContract, ...]:
    return tuple(
        S(
            section.section_id,
            section.number,
            section.title,
            section.applicability,
            tuple(_model_comparison_table(table) for table in section.tables),
        )
        for section in MODEL_STANDARD_SOP.sections
    )


MODEL_STANDARD_COMPARISON = PersistentReportContract(
    report_id="model.standard_comparison",
    version=3,
    role="persistent_multi_model_comparison_oos_breakout",
    menu_path=("Research", "模型訓練／驗證", "模型比較（Standard SOP）"),
    sections=_model_comparison_sections(),
)


MODEL_EXTENSION_SCHEMAS: Mapping[str, ModelExtensionContract] = {
    "multi_head_learnability": ModelExtensionContract(
        "multi_head_learnability", "Multi-head Learnability",
        "multi_head_only",
        (T("multi_head", (
            C("split", "Split", alignment="left"), C("head", "Head", alignment="left"),
            C("mean_daily_spearman", "Daily rho", 4, preference="higher", format_kind="number"),
            C("global_spearman_vs_raw_target", "Global rho", 4, preference="higher", format_kind="number"),
            C("pairwise_concordance", "Pair", 2, "%", "higher", "fraction_pct"),
        )),),
    ),
    "truth_prediction_geometry": ModelExtensionContract(
        "truth_prediction_geometry", "Truth / Prediction Geometry",
        "geometry_models",
        (
            T("geometry_summary", (
                C("metric", "Metric", alignment="left"), C("value", "Value", alignment="right"),
            )),
            T("truth_5x5", (
                C("safety", "Actual Safety \\ Pure-MFE", alignment="left"),
                C("m1", "M1"), C("m2", "M2"), C("m3", "M3"), C("m4", "M4"), C("m5", "M5"),
            )),
            T("predicted_5x5", (
                C("safety", "Pred Safety \\ Raw-MFE", alignment="left"),
                C("m1", "M1"), C("m2", "M2"), C("m3", "M3"), C("m4", "M4"), C("m5", "M5"),
            )),
            T("safety_cohorts", (
                C("safety", "Pred Safety", alignment="left"), C("n", "N", 0, format_kind="int"),
                C("raw_mfe_to_actual_mfe_mean_daily_spearman", "Raw-MFE→MFE rho", 4, preference="higher", format_kind="number"),
                C("high_mfe_pct", "High-MFE", 2, "%", "higher", "pct"),
                C("hmhs_pct", "HM/HS", 2, "%", "higher", "pct"),
            )),
        ),
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


def extension_contract(extension_id: str) -> ModelExtensionContract:
    try:
        return MODEL_EXTENSION_SCHEMAS[str(extension_id)]
    except KeyError as exc:
        raise KeyError(f"未知Model-specific extension: {extension_id}") from exc


def _persistent_schema_payload(contract: PersistentReportContract) -> dict:
    return asdict(contract)


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
    "model.standard_comparison": "7bf0ef0427519507",
    "model.standard_sop": "9ec49fe2aaf3a2d6",
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
    "MODEL_EXTENSION_SCHEMAS", "MODEL_STANDARD_SOP",
    "OPPORTUNITY_SELECTION_REPORT", "PORTFOLIO_DRAWDOWN_REPORT",
    "MODEL_STANDARD_COMPARISON", "PERSISTENT_REPORT_CONTRACTS", "STRATEGY_CONSISTENCY_METRIC_KEYS",
    "STRATEGY_CONSISTENCY_REPORT", "STRATEGY_STANDARD_SOP",
    "TRADE_OUTCOME_FIRST_PASSAGE_THRESHOLDS_R", "TRADE_OUTCOME_PATH_REPORT", "ModelExtensionContract",
    "PersistentReportContract", "ReportColumnContract", "ReportSectionContract",
    "ReportTableContract", "column_contract", "extension_contract", "format_contract_value", "persistent_report_contract_fingerprint",
    "persistent_report_contract_fingerprints", "report_contract", "section_contract",
    "table_contract", "validate_approved_persistent_report_contracts",
]

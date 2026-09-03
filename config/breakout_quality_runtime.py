"""Generic continuous-ranker runtime capability contracts.

This module owns execution-only primitives used by breakout-quality consumers.
Scientific/profile declarations and user-adjustable settings remain in
``config.breakout_quality``.  This module is deliberately dependency-free from that
scientific configuration owner; profile resolution is composed outside this module.

Do not add MR/profile-specific branches here.  New behavior belongs in reusable
target/training/context/objective capabilities and is selected declaratively.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable


# Stable execution-capability identities live with the runtime policies that consume them.
# ``config.breakout_quality`` re-exports these names for backward compatibility and uses
# them when declaring scientific profiles; there must not be a reverse runtime->config edge.
PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID = (
    "daily_predicted_upside_conditional_low_adverse_v1"
)
PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID = (
    "daily_predicted_safety_conditional_mfe_v1"
)
PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID = (
    "daily_predicted_safety_context_pure_mfe_r_v1"
)

TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION = "daily_percentile_regression"
TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION = "daily_raw_r_regression"
TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION = "daily_dual_component_r_regression"
TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING = "daily_pairwise_ranking"
TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING = "daily_pareto_pairwise_ranking"
TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING = (
    "daily_conditional_mfe_safety_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING = (
    "daily_conditional_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING = (
    "daily_safety_conditional_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING = (
    "daily_safety_raw_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING = (
    "daily_shared_safety_weighted_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING = (
    "daily_shared_safety_weighted_primary_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING = (
    "daily_shared_safety_hs_conditional_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING = (
    "daily_shared_hs_qualification_conditional_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING = (
    "daily_shared_hs_boundary_weighted_qualification_conditional_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_PAIRWISE_RANKING = (
    "daily_shared_dual_supervised_hs_conditional_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING = (
    "daily_shared_top_hs_safety_conditional_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING = (
    "daily_shared_safety_hs_priority_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING = (
    "daily_shared_safety_hs_priority_stratified_mfe_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING = (
    "daily_safety_raw_mfe_hmhs_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING = (
    "daily_safety_raw_mfe_joint_min_pairwise_ranking"
)
TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING = "daily_hmhs_pairwise_ranking"
TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING = "daily_listwise_ranking"

CONTINUOUS_RANKER_TRAINER_EVENT = "event"
CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL = "daily_universal"
CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR = "equal_pair_weight"
CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED = "target_gap_weighted_within_date"
CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE = "upper_tail_relevance_weighted"
CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG = "full_list_delta_ndcg_weighted"
CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG = "full_list_delta_ndcg_times_min_predicted_safety"
CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE = "pareto_dominance_equal_pair"

# Generic runtime capability primitives.  Scientific/research identity must not leak into
# execution consumers; consumers resolve these policies from ContinuousRankerExecutionRecipe.
CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE = "none"
CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE = "predicted_upside"
CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY = "predicted_safety"
CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE = "coverage"
CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT = "model_input"
CONTINUOUS_RANKER_CONTEXT_ROLE_TARGET_TRANSFORM = "target_transform"
CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT = "pair_weight"
CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE = "none"
CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY = "min_predicted_safety"
CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY = "mfe_winner_predicted_safety"
CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY = "product_predicted_safety"
CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY = "conflict_unsafe_winner_predicted_safety"
CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_NONE = "none"
CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY = "binary_boundary_proximity"
SUPPORTED_CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICIES = {
    CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_NONE,
    CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY,
}
CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL = "all_items"
CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN = "primary_target_min"
CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_ALL = "all"
CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_CROSS = "cross"
CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_WITHIN_POSITIVE = "within_positive"
SUPPORTED_CONTINUOUS_RANKER_PAIR_PARTITION_RELATIONS = {
    CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_ALL,
    CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_CROSS,
    CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_WITHIN_POSITIVE,
}
SUPPORTED_CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPES = (
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
)


@dataclass(frozen=True)
class ContinuousRankerPairWeightPolicy:
    """First-class pair-weight plugin consumed by generic training/reporting code."""

    policy_id: str
    context_source: str
    compatible_reductions: tuple[str, ...]
    multiplier: Callable[[Any, Any, Any, Any], Any] | None = None
    contract_pair_safety_weight: str | None = None
    contract_pair_weight_combination: str | None = None
    report_extension_title: str | None = None
    report_first_note: str | None = None
    report_second_note: str | None = None

    @property
    def weighted(self) -> bool:
        return self.multiplier is not None

    def apply(self, torch, target_diff, left_context, right_context):
        if self.multiplier is None:
            raise ValueError(f"pair weight policy不提供multiplier: {self.policy_id!r}")
        return self.multiplier(torch, target_diff, left_context, right_context)


def _minimum_predicted_safety_pair_weight(torch, _target_diff, left_safety, right_safety):
    return torch.minimum(left_safety, right_safety)


def _mfe_winner_predicted_safety_pair_weight(torch, target_diff, left_safety, right_safety):
    return torch.where(target_diff > 0, left_safety, right_safety)


def _product_predicted_safety_pair_weight(torch, _target_diff, left_safety, right_safety):
    return left_safety * right_safety


def _conflict_unsafe_winner_predicted_safety_pair_weight(
    torch, target_diff, left_safety, right_safety
):
    """Discount only MFE/Safety conflict pairs by the unsafe MFE winner's Safety."""

    winner_safety = torch.where(target_diff > 0, left_safety, right_safety)
    safety_diff = left_safety - right_safety
    aligned_or_tied = (target_diff * safety_diff) >= 0
    return torch.where(aligned_or_tied, torch.ones_like(winner_safety), winner_safety)


_CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_REGISTRY: dict[str, ContinuousRankerPairWeightPolicy] = {
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE: ContinuousRankerPairWeightPolicy(
        policy_id=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
        context_source=CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
        compatible_reductions=(),
    ),
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY: ContinuousRankerPairWeightPolicy(
        policy_id=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY,
        context_source=CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        compatible_reductions=(CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,),
        multiplier=_minimum_predicted_safety_pair_weight,
        contract_pair_safety_weight="min(predicted_safety_percentile_i,predicted_safety_percentile_j)",
        contract_pair_weight_combination="delta_ndcg_times_min_predicted_safety",
        report_extension_title="Pure-MFE × High-Safety Pair Weight",
        report_first_note=(
            "- Target/order與MR-13K相同；Predicted Safety不進network，只把MR-13K full-list "
            "ΔNDCG pair weight乘上min(S_i,S_j)。"
        ),
        report_second_note=(
            "- Actual Safety/MFE metrics只作checkpoint寫入後診斷；epoch selection仍固定Pure-MFE "
            "Validation Daily rho，無bucket/cutoff/lambda。"
        ),
    ),
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY: ContinuousRankerPairWeightPolicy(
        policy_id=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY,
        context_source=CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        compatible_reductions=(CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,),
        multiplier=_mfe_winner_predicted_safety_pair_weight,
        contract_pair_safety_weight="predicted_safety_percentile_of_higher_pure_mfe_item",
        contract_pair_weight_combination="delta_ndcg_times_mfe_winner_predicted_safety",
        report_extension_title="Pure-MFE × MFE-Winner Safety Pair Weight",
        report_first_note=(
            "- Target/order與MR-13K相同；Predicted Safety不進network，只把MR-13K full-list "
            "ΔNDCG pair weight乘上較高Pure-MFE item本身的S_winner；pair方向永不因Safety反轉。"
        ),
        report_second_note=(
            "- Actual Safety/MFE metrics只作checkpoint寫入後診斷；epoch selection仍固定Pure-MFE "
            "Validation Daily rho；無額外mean normalization、bucket/cutoff/lambda/temperature。"
        ),
    ),
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY: ContinuousRankerPairWeightPolicy(
        policy_id=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
        context_source=CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        compatible_reductions=(CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,),
        multiplier=_product_predicted_safety_pair_weight,
        contract_pair_safety_weight="predicted_safety_percentile_i_times_predicted_safety_percentile_j",
        contract_pair_weight_combination="delta_ndcg_times_product_predicted_safety",
        report_extension_title="Pure-MFE × Safety-Product Pair Weight",
        report_first_note=(
            "- Target/order與MR-13K相同；Predicted Safety不進network，只把MR-13K full-list "
            "ΔNDCG pair weight乘上S_i×S_j；pair weighting保持對稱，不依MFE winner方向改變。"
        ),
        report_second_note=(
            "- Actual Safety/MFE metrics只作checkpoint寫入後診斷；epoch selection仍固定Pure-MFE "
            "Validation Daily rho；無bucket/cutoff/lambda/exponent/temperature。"
        ),
    ),
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY: ContinuousRankerPairWeightPolicy(
        policy_id=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY,
        context_source=CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        compatible_reductions=(CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,),
        multiplier=_conflict_unsafe_winner_predicted_safety_pair_weight,
        contract_pair_safety_weight=(
            "1_if_mfe_and_predicted_safety_order_agree_or_tie_else_"
            "predicted_safety_percentile_of_mfe_winner"
        ),
        contract_pair_weight_combination=(
            "delta_ndcg_times_conflict_only_unsafe_mfe_winner_predicted_safety"
        ),
        report_extension_title="Pure-MFE × Conflict-Only Unsafe-Winner Discount",
        report_first_note=(
            "- Target/order與MR-13K相同；Predicted Safety不進network。MFE與Safety ordering一致或Safety tie時保留完整MR-13K ΔNDCG；"
            "只有MFE winner較不安全的conflict pair才乘該winner的PIT-safe Safety percentile。"
        ),
        report_second_note=(
            "- Pair truth永不因Safety反轉；此cell保留aligned MFE supervision，僅削弱unsafe-winner conflict pressure；"
            "epoch selection仍固定Pure-MFE Validation Daily rho，無bucket/cutoff/lambda/exponent/temperature。"
        ),
    ),
}

SUPPORTED_CONTINUOUS_RANKER_PAIR_WEIGHT_POLICIES = tuple(
    _CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_REGISTRY
)
PREDICTED_SAFETY_CONTINUOUS_RANKER_PAIR_WEIGHT_POLICIES = tuple(
    policy_id
    for policy_id, policy in _CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_REGISTRY.items()
    if policy.context_source == CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY
)


def get_continuous_ranker_pair_weight_policy(
    policy: str | ContinuousRankerPairWeightPolicy | None,
) -> ContinuousRankerPairWeightPolicy:
    if isinstance(policy, ContinuousRankerPairWeightPolicy):
        return policy
    policy_id = CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE if policy is None else str(policy)
    try:
        return _CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_REGISTRY[policy_id]
    except KeyError as exc:
        raise ValueError(f"不支援的continuous-ranker pair weight policy: {policy_id!r}") from exc


def normalize_continuous_ranker_pair_weight_configuration(
    reduction: str | None,
    pair_weight_policy: str | ContinuousRankerPairWeightPolicy | None,
) -> tuple[str | None, ContinuousRankerPairWeightPolicy]:
    """Normalize legacy combined reduction while keeping generic consumers identity-free."""

    declared_policy = get_continuous_ranker_pair_weight_policy(pair_weight_policy)
    base_reduction = reduction
    if reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG:
        if declared_policy.policy_id not in {
            CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
            CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY,
        }:
            raise ValueError("legacy High-Safety min reduction不得宣告不同pair weight policy")
        declared_policy = get_continuous_ranker_pair_weight_policy(
            CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY
        )
        base_reduction = CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG
    if declared_policy.weighted and base_reduction not in declared_policy.compatible_reductions:
        raise ValueError(
            f"pair weight policy {declared_policy.policy_id!r}不支援reduction {base_reduction!r}"
        )
    return base_reduction, declared_policy
CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR = "scalar"
CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_PARETO_COMPONENTS = "pareto_components"
CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT = "scalar_with_context_weight"

SUPPORTED_CONTINUOUS_RANKER_PAIRWISE_REDUCTIONS = (
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE,
)

@dataclass(frozen=True)
class ContinuousRankerContextPolicy:
    """Execution-only context capability; no MR/profile identity is allowed here."""

    source: str = CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        supported_sources = {
            CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
            CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
            CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        }
        supported_roles = {
            CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE,
            CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT,
            CONTINUOUS_RANKER_CONTEXT_ROLE_TARGET_TRANSFORM,
            CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT,
        }
        if self.source not in supported_sources:
            raise ValueError(f"不支援的continuous-ranker context source: {self.source!r}")
        unknown_roles = sorted(set(self.roles) - supported_roles)
        if unknown_roles:
            raise ValueError(f"不支援的continuous-ranker context roles: {unknown_roles}")
        if self.source == CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE and self.roles:
            raise ValueError("context source=none時不得宣告context roles")
        if self.source != CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE and CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE not in self.roles:
            raise ValueError("persistent context source必須宣告coverage role")

    def has_role(self, role: str) -> bool:
        return str(role) in self.roles


CONTINUOUS_RANKER_TARGET_MATERIALIZATION_EXTERNAL = "external_continuous_target"
CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT = "daily_component_target"
CONTINUOUS_RANKER_TARGET_MATERIALIZATION_RISK_NORMALIZED = "risk_normalized_target"
CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE = "none"
CONTINUOUS_RANKER_TARGET_POSTPROCESS_EQUAL_RANK_MFE_LOW_ADVERSE = "equal_rank_mfe_low_adverse"
CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_NONE = "none"
CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_UPSIDE_LOW_ADVERSE = (
    "predicted_upside_conditional_low_adverse"
)
CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_SAFETY_MFE = (
    "predicted_safety_conditional_mfe"
)
CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PRESERVE_PURE_MFE = "preserve_pure_mfe"

CONTINUOUS_RANKER_BATCH_MODE_SHUFFLED = "shuffled_unique_group_batches"
CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT = "whole_date_pack_no_date_split"
CONTINUOUS_RANKER_TARGET_BUILDER_PERCENTILE = "percentile"
CONTINUOUS_RANKER_TARGET_BUILDER_SCALAR_PAIRWISE = "scalar_pairwise"
CONTINUOUS_RANKER_TARGET_BUILDER_RAW_R = "raw_r"
CONTINUOUS_RANKER_TARGET_BUILDER_DUAL_COMPONENT_R = "dual_component_r"
CONTINUOUS_RANKER_TARGET_BUILDER_PARETO_COMPONENTS = "pareto_components"
CONTINUOUS_RANKER_TARGET_BUILDER_CONDITIONAL_MFE_SAFETY = "conditional_mfe_safety"
CONTINUOUS_RANKER_TARGET_BUILDER_CONDITIONAL_MFE_SINGLE = "conditional_mfe_single"
CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_CONDITIONAL_MFE = "safety_conditional_mfe"
CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE = "safety_raw_mfe"
CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_PRIMARY = "safety_primary"
CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE = "hs_conditional_mfe"
CONTINUOUS_RANKER_TARGET_BUILDER_HS_PRIORITY_MFE = "hs_priority_mfe"
CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE_HMHS = "safety_raw_mfe_hmhs"
CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE_JOINT_MIN = "safety_raw_mfe_joint_min"
CONTINUOUS_RANKER_TARGET_BUILDER_DIRECT_HMHS = "direct_hmhs"
CONTINUOUS_RANKER_LOSS_HANDLER_PERCENTILE_MSE = "percentile_mse"
CONTINUOUS_RANKER_LOSS_HANDLER_RAW_R = "raw_r_regression"
CONTINUOUS_RANKER_LOSS_HANDLER_DUAL_COMPONENT_R = "dual_component_r_regression"
CONTINUOUS_RANKER_LOSS_HANDLER_SINGLE_PAIRWISE = "single_pairwise"
CONTINUOUS_RANKER_LOSS_HANDLER_CONDITIONAL_DUO_PAIRWISE = "conditional_duo_pairwise"
CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_DUO_PAIRWISE = "safety_mfe_duo_pairwise"
CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_WEIGHTED_MFE_DUO_PAIRWISE = (
    "shared_safety_weighted_mfe_duo_pairwise"
)
CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE = (
    "shared_safety_scoped_mfe_duo_pairwise"
)
CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_HS_QUALIFICATION_SCOPED_MFE_DUO_PAIRWISE = (
    "shared_hs_qualification_scoped_mfe_duo_pairwise"
)
CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_DUAL_SUPERVISED_HS_SCOPED_MFE_DUO_PAIRWISE = (
    "shared_dual_supervised_hs_scoped_mfe_duo_pairwise"
)
CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_TOP_HS_SAFETY_SCOPED_MFE_DUO_PAIRWISE = (
    "shared_top_hs_safety_scoped_mfe_duo_pairwise"
)
CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_STRATIFIED_MFE_DUO_PAIRWISE = (
    "shared_safety_stratified_mfe_duo_pairwise"
)
CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_JOINT_TRI_PAIRWISE = "safety_mfe_joint_tri_pairwise"
CONTINUOUS_RANKER_LOSS_HANDLER_LISTWISE = "listwise"
CONTINUOUS_RANKER_AUX_TARGET_NONE = "none"
CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_SAFETY = "conditional_mfe_safety"
CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY = "conditional_mfe_opportunity"
CONTINUOUS_RANKER_SEMANTICS_DEFAULT = "default"
CONTINUOUS_RANKER_SEMANTICS_PAIRWISE = "pairwise"
CONTINUOUS_RANKER_SEMANTICS_LISTWISE = "listwise"
CONTINUOUS_RANKER_SEMANTICS_RAW_R = "raw_r"
CONTINUOUS_RANKER_SEMANTICS_DUAL_COMPONENT_R = "dual_component_r"
CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SAFETY = "conditional_mfe_safety"
CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SINGLE = "conditional_mfe_single"
CONTINUOUS_RANKER_SEMANTICS_SAFETY_CONDITIONAL_MFE = "safety_conditional_mfe"
CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE = "safety_raw_mfe"
CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_MFE = "shared_safety_weighted_mfe"
CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_PRIMARY = "shared_safety_weighted_primary"
CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_CONDITIONAL_MFE = "shared_safety_hs_conditional_mfe"
CONTINUOUS_RANKER_SEMANTICS_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE = (
    "shared_hs_qualification_conditional_mfe"
)
CONTINUOUS_RANKER_SEMANTICS_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE = (
    "shared_dual_supervised_hs_conditional_mfe"
)
CONTINUOUS_RANKER_SEMANTICS_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE = (
    "shared_top_hs_safety_conditional_mfe"
)
CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_MFE = "shared_safety_hs_priority_mfe"
CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE = (
    "shared_safety_hs_priority_stratified_mfe"
)
CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_HMHS = "safety_raw_mfe_hmhs"
CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_JOINT_MIN = "safety_raw_mfe_joint_min"
CONTINUOUS_RANKER_SEMANTICS_DIRECT_HMHS = "direct_hmhs"
CONTINUOUS_RANKER_SCORE_TRANSFORM_PROBABILITY = "pass_probability"
CONTINUOUS_RANKER_SCORE_TRANSFORM_MARGIN_R = "pass_minus_reject_margin"
CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH = "mean_batch"
CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_WEIGHTED = "weighted_supervision"
CONTINUOUS_RANKER_PAIRWISE_KIND_NONE = "none"
CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR = "scalar"
CONTINUOUS_RANKER_PAIRWISE_KIND_PARETO = "pareto"
CONTINUOUS_RANKER_PRIMARY_SUPERVISION_CONTINUOUS = "continuous"
CONTINUOUS_RANKER_PRIMARY_SUPERVISION_BINARY_HS = "binary_hs"
CONTINUOUS_RANKER_PRIMARY_SUPERVISION_DUAL_HS = "dual_hs"
CONTINUOUS_RANKER_PRIMARY_SUPERVISION_TOP_HS = "top_hs"
CONTINUOUS_RANKER_SECONDARY_SUPERVISION_STANDARD = "standard"
CONTINUOUS_RANKER_SECONDARY_SUPERVISION_STRATIFIED = "stratified"
CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_SINGLE = "single"
CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED = "equal_mean_required"
CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE = "equal_mean_available"


@dataclass(frozen=True)
class ContinuousRankerTargetPolicy:
    """Daily/external target materialization semantics, independent of experiment identity."""

    materialization_mode: str
    component_target_id: str | None = None
    postprocess: str = CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE
    context_source: str = CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE
    context_roles: tuple[str, ...] = ()
    context_transform: str = CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_NONE
    contract_kind: str = "external"

    def __post_init__(self) -> None:
        if self.materialization_mode not in {
            CONTINUOUS_RANKER_TARGET_MATERIALIZATION_EXTERNAL,
            CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            CONTINUOUS_RANKER_TARGET_MATERIALIZATION_RISK_NORMALIZED,
        }:
            raise ValueError(f"不支援的continuous-ranker target materialization: {self.materialization_mode!r}")
        if self.postprocess not in {
            CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE,
            CONTINUOUS_RANKER_TARGET_POSTPROCESS_EQUAL_RANK_MFE_LOW_ADVERSE,
        }:
            raise ValueError(f"不支援的continuous-ranker target postprocess: {self.postprocess!r}")
        if self.context_transform not in {
            CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_NONE,
            CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_UPSIDE_LOW_ADVERSE,
            CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_SAFETY_MFE,
            CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PRESERVE_PURE_MFE,
        }:
            raise ValueError(f"不支援的continuous-ranker target context transform: {self.context_transform!r}")
        ContinuousRankerContextPolicy(
            source=self.context_source,
            roles=self.context_roles,
        )
        if (
            self.materialization_mode == CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT
            and not str(self.component_target_id or "").strip()
        ):
            raise ValueError("daily-component target policy必須指定component_target_id")
        if (
            self.materialization_mode != CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT
            and self.component_target_id is not None
        ):
            raise ValueError("非daily-component target policy不得指定component_target_id")


@dataclass(frozen=True)
class ContinuousRankerTrainingPolicy:
    """Canonical objective/composition capability, independent of MR/profile identity.

    Profile validation, research-spec legality, trainer supervision geometry and artifact
    semantics all consume this descriptor.  New objectives should register their
    composition here rather than teaching generic consumers another objective-name switch.
    """

    batch_mode: str
    target_builder: str
    loss_handler: str
    profile_loss_metrics: tuple[tuple[str, str], ...] = ()
    pairwise_kind: str = CONTINUOUS_RANKER_PAIRWISE_KIND_NONE
    allows_context_pair_weight: bool = False
    allowed_secondary_pair_scopes: tuple[str, ...] = (
        CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
    )
    primary_supervision_mode: str = CONTINUOUS_RANKER_PRIMARY_SUPERVISION_CONTINUOUS
    secondary_supervision_mode: str = CONTINUOUS_RANKER_SECONDARY_SUPERVISION_STANDARD
    primary_pair_weight_policy: str = CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_NONE
    head_loss_combination: str = CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_SINGLE
    head_loss_component_count: int = 1
    auxiliary_target_bundle: str = CONTINUOUS_RANKER_AUX_TARGET_NONE
    semantics_contract_key: str = CONTINUOUS_RANKER_SEMANTICS_DEFAULT
    score_transform: str = CONTINUOUS_RANKER_SCORE_TRANSFORM_PROBABILITY
    epoch_loss_aggregation: str = CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_WEIGHTED
    uses_pairwise_loss: bool = False
    score_output_policy: ContinuousRankerScoreOutputPolicy | None = None
    # Comparison/report evidence is a declarative capability of the training
    # composition, not a workflow/model-ID whitelist.  Consumers discover
    # applicable Model-specific Extensions through these stable family IDs.
    report_evidence_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.batch_mode not in {
            CONTINUOUS_RANKER_BATCH_MODE_SHUFFLED,
            CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
        }:
            raise ValueError(f"不支援的continuous-ranker batch mode: {self.batch_mode!r}")
        if self.auxiliary_target_bundle not in {
            CONTINUOUS_RANKER_AUX_TARGET_NONE,
            CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_SAFETY,
            CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY,
        }:
            raise ValueError(f"不支援的continuous-ranker auxiliary target bundle: {self.auxiliary_target_bundle!r}")
        if self.score_transform not in {
            CONTINUOUS_RANKER_SCORE_TRANSFORM_PROBABILITY,
            CONTINUOUS_RANKER_SCORE_TRANSFORM_MARGIN_R,
        }:
            raise ValueError(f"不支援的continuous-ranker score transform: {self.score_transform!r}")
        if self.epoch_loss_aggregation not in {
            CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_WEIGHTED,
        }:
            raise ValueError(
                f"不支援的continuous-ranker epoch loss aggregation: {self.epoch_loss_aggregation!r}"
            )
        if self.score_output_policy is not None and not isinstance(
            self.score_output_policy, ContinuousRankerScoreOutputPolicy
        ):
            raise TypeError("score_output_policy必須是ContinuousRankerScoreOutputPolicy")
        evidence_families = tuple(str(value).strip() for value in self.report_evidence_families)
        if any(not value for value in evidence_families) or len(set(evidence_families)) != len(evidence_families):
            raise ValueError("report_evidence_families不得包含空值或重複值")
        loss_names = tuple(str(loss_name) for loss_name, _metric in self.profile_loss_metrics)
        if any(not value for value in loss_names) or len(loss_names) != len(set(loss_names)):
            raise ValueError("profile_loss_metrics loss_name不得為空或重複")
        if any(not str(metric).strip() for _loss_name, metric in self.profile_loss_metrics):
            raise ValueError("profile_loss_metrics epoch metric不得為空")
        if self.pairwise_kind not in {
            CONTINUOUS_RANKER_PAIRWISE_KIND_NONE,
            CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            CONTINUOUS_RANKER_PAIRWISE_KIND_PARETO,
        }:
            raise ValueError(f"不支援的continuous-ranker pairwise kind: {self.pairwise_kind!r}")
        if bool(self.uses_pairwise_loss) != (
            self.pairwise_kind != CONTINUOUS_RANKER_PAIRWISE_KIND_NONE
        ):
            raise ValueError("uses_pairwise_loss與pairwise_kind不一致")
        if self.allows_context_pair_weight and self.pairwise_kind != CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR:
            raise ValueError("context pair weight只允許scalar pairwise composition")
        scopes = tuple(str(scope) for scope in self.allowed_secondary_pair_scopes)
        if not scopes or len(scopes) != len(set(scopes)):
            raise ValueError("allowed_secondary_pair_scopes不得為空或重複")
        if any(scope not in SUPPORTED_CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPES for scope in scopes):
            raise ValueError("allowed_secondary_pair_scopes含不支援值")
        if CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL not in scopes:
            raise ValueError("所有training composition都必須允許all-items secondary scope")
        if self.primary_supervision_mode not in {
            CONTINUOUS_RANKER_PRIMARY_SUPERVISION_CONTINUOUS,
            CONTINUOUS_RANKER_PRIMARY_SUPERVISION_BINARY_HS,
            CONTINUOUS_RANKER_PRIMARY_SUPERVISION_DUAL_HS,
            CONTINUOUS_RANKER_PRIMARY_SUPERVISION_TOP_HS,
        }:
            raise ValueError(f"不支援的primary supervision mode: {self.primary_supervision_mode!r}")
        if self.secondary_supervision_mode not in {
            CONTINUOUS_RANKER_SECONDARY_SUPERVISION_STANDARD,
            CONTINUOUS_RANKER_SECONDARY_SUPERVISION_STRATIFIED,
        }:
            raise ValueError(f"不支援的secondary supervision mode: {self.secondary_supervision_mode!r}")
        if self.primary_pair_weight_policy not in SUPPORTED_CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICIES:
            raise ValueError(
                f"不支援的primary pair weight policy: {self.primary_pair_weight_policy!r}"
            )
        if (
            self.primary_pair_weight_policy != CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_NONE
            and self.primary_supervision_mode != CONTINUOUS_RANKER_PRIMARY_SUPERVISION_BINARY_HS
        ):
            raise ValueError("truth-side primary pair weight目前只允許binary-HS supervision")
        if self.head_loss_combination not in {
            CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_SINGLE,
            CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED,
            CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE,
        }:
            raise ValueError(f"不支援的head loss combination: {self.head_loss_combination!r}")
        if int(self.head_loss_component_count) < 1:
            raise ValueError("head_loss_component_count必須>=1")
        if (
            self.head_loss_combination == CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_SINGLE
            and int(self.head_loss_component_count) != 1
        ):
            raise ValueError("single head-loss combination必須宣告component_count=1")
        if (
            self.head_loss_combination != CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_SINGLE
            and int(self.head_loss_component_count) < 2
        ):
            raise ValueError("multi-head loss combination必須宣告component_count>=2")

    def artifact_head_weighting_semantic(self) -> str | None:
        """Return canonical persisted head-weighting wording for this composition.

        The wording is intentionally historical-contract compatible; the composition
        descriptor, not ranker_training_contract.py, owns which weighting applies.
        """

        if self.head_loss_combination == CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_SINGLE:
            return None
        if self.primary_supervision_mode == CONTINUOUS_RANKER_PRIMARY_SUPERVISION_DUAL_HS:
            if int(self.head_loss_component_count) != 2:
                raise ValueError("dual-HS composition預期兩個outer head losses")
            return (
                "safety_branch_0.5_conditional_mfe_0.5_with_"
                "safety_branch_split_0.5_0.5"
            )
        if int(self.head_loss_component_count) == 2:
            return "fixed_equal_mean_no_lambda_sweep"
        if int(self.head_loss_component_count) == 3:
            return "fixed_equal_mean_three_heads_no_lambda_sweep"
        raise ValueError(
            "artifact head-weighting尚未定義此multi-head component count: "
            f"{self.head_loss_component_count}"
        )

    def expected_epoch_metric(self, loss_name: str) -> str:
        mapping = dict(self.profile_loss_metrics)
        if not mapping:
            raise ValueError("continuous-ranker training capability尚未宣告profile loss/epoch metric contract")
        try:
            return str(mapping[str(loss_name)])
        except KeyError as exc:
            raise ValueError(
                "continuous ranker loss與training objective composition不一致: "
                f"expected one of {sorted(mapping)}, actual={loss_name!r}"
            ) from exc

    def validate_profile_loss_metric(
        self,
        *,
        loss_name: str,
        epoch_selection_metric: str,
    ) -> None:
        expected_metric = self.expected_epoch_metric(loss_name)
        if str(epoch_selection_metric) != expected_metric:
            raise ValueError(
                "continuous ranker epoch selection與training objective composition不一致: "
                f"expected={expected_metric}, actual={epoch_selection_metric}"
            )

    def validate_research_spec(
        self,
        *,
        pairwise_reduction: str | None,
        pair_weight_policy: str | None,
        secondary_pair_scope: str,
        secondary_pair_scope_threshold: float | None,
    ) -> None:
        reduction = pairwise_reduction
        if self.pairwise_kind == CONTINUOUS_RANKER_PAIRWISE_KIND_NONE:
            if reduction is not None:
                raise ValueError("非pairwise composition不得指定pairwise reduction")
        else:
            if reduction not in SUPPORTED_CONTINUOUS_RANKER_PAIRWISE_REDUCTIONS:
                raise ValueError("pairwise composition必須指定合法pairwise reduction")
            if self.pairwise_kind == CONTINUOUS_RANKER_PAIRWISE_KIND_PARETO:
                if reduction != CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE:
                    raise ValueError("Pareto pairwise composition必須使用pareto_dominance_equal_pair")
            elif reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE:
                raise ValueError("scalar pairwise composition不得使用Pareto dominance pair scope")

        if pair_weight_policy is not None:
            if not self.allows_context_pair_weight:
                raise ValueError("此training composition不得指定context pair weight policy")
            get_continuous_ranker_pair_weight_policy(pair_weight_policy)
            if pair_weight_policy == CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE:
                raise ValueError("pair_weight_policy=None即可表示未加權；不得顯式宣告none")

        scope = str(secondary_pair_scope)
        if scope not in self.allowed_secondary_pair_scopes:
            raise ValueError(
                f"secondary pair scope不適用於此training composition: {scope!r}"
            )
        if scope == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL:
            if secondary_pair_scope_threshold is not None:
                raise ValueError("all-items secondary pair scope不得指定threshold")
        else:
            threshold = secondary_pair_scope_threshold
            if threshold is None or not (0.0 < float(threshold) < 1.0):
                raise ValueError("secondary pair scope threshold必須位於(0,1)")


@dataclass(frozen=True)
class ContinuousRankerObjectivePolicy:
    """Executable objective semantics independent of research naming/history."""

    pairwise_reduction: str | None
    pair_weight_policy: str = CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE
    pair_target_schema: str = CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR
    secondary_pair_scope: str = CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL
    secondary_pair_scope_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.pairwise_reduction is not None and self.pairwise_reduction not in SUPPORTED_CONTINUOUS_RANKER_PAIRWISE_REDUCTIONS:
            raise ValueError(f"不支援的continuous-ranker pairwise reduction: {self.pairwise_reduction!r}")
        get_continuous_ranker_pair_weight_policy(self.pair_weight_policy)
        if self.pair_target_schema not in {
            CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR,
            CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_PARETO_COMPONENTS,
            CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT,
        }:
            raise ValueError(f"不支援的continuous-ranker pair target schema: {self.pair_target_schema!r}")
        weighted = self.pair_weight_policy != CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE
        if weighted != (self.pair_target_schema == CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT):
            raise ValueError("pair weight policy與pair target schema不一致")
        if self.secondary_pair_scope not in SUPPORTED_CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPES:
            raise ValueError(f"不支援的secondary pair scope: {self.secondary_pair_scope!r}")
        if self.secondary_pair_scope == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL:
            if self.secondary_pair_scope_threshold is not None:
                raise ValueError("all-items secondary pair scope不得指定threshold")
        else:
            threshold = self.secondary_pair_scope_threshold
            if threshold is None or not (0.0 < float(threshold) < 1.0):
                raise ValueError("scoped secondary pair supervision必須指定(0,1) threshold")


@dataclass(frozen=True)
class ContinuousRankerDependencySpec:
    """Persistent upstream requirements expressed as capabilities, not artifact paths."""

    requires_continuous_target_artifact: bool
    context_source: str = CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE

    def __post_init__(self) -> None:
        if self.context_source not in {
            CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
            CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
            CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        }:
            raise ValueError(
                f"不支援的continuous-ranker dependency context source: {self.context_source!r}"
            )


@dataclass(frozen=True)
class BreakoutQualityOutputSchema:
    """Canonical model-output width contract shared by inference consumers."""

    head_widths: tuple[tuple[str, int], ...]

    def width_for(self, output_head: str | None) -> int:
        head = str(output_head or "").strip().lower()
        widths = dict(self.head_widths)
        try:
            return int(widths[head])
        except KeyError as exc:
            raise ValueError(f"未知 breakout-quality output_head width contract: {output_head!r}") from exc


BREAKOUT_QUALITY_OUTPUT_SCHEMA = BreakoutQualityOutputSchema(
    head_widths=(
        ("", 2),
        ("primary", 2),
        ("mfe", 2),
        ("primary_mfe", 2),
        ("conditional_safety", 2),
        ("safety", 2),
        ("raw_safety", 2),
        ("safety_condition", 2),
        ("conditional_mfe", 2),
        ("final", 2),
        ("joint_hmhs", 2),
        ("hmhs", 2),
        ("conditional_both", 4),
        ("both", 4),
        ("tri_head", 6),
        ("safety_raw_mfe_hmhs", 6),
        ("all_three", 6),
    )
)


@dataclass(frozen=True)
class ContinuousRankerScoreOutputPolicy:
    """Canonical multi-head score-output contract consumed by PIT/inference services."""

    output_head: str
    head_names: tuple[str, ...]
    primary_head: str
    persisted_columns: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        names = tuple(str(value).strip() for value in self.head_names)
        if not names or any(not value for value in names):
            raise ValueError("score-output head_names不得為空")
        if len(names) != len(set(names)):
            raise ValueError("score-output head_names不得重複")
        if str(self.primary_head) not in names:
            raise ValueError("score-output primary_head必須存在於head_names")
        expected_width = 2 * len(names)
        actual_width = BREAKOUT_QUALITY_OUTPUT_SCHEMA.width_for(self.output_head)
        if actual_width != expected_width:
            raise ValueError(
                "score-output output_head width與head_names不一致: "
                f"output_head={self.output_head!r}, width={actual_width}, heads={names}"
            )
        columns = tuple((str(head), str(column)) for head, column in self.persisted_columns)
        column_heads = tuple(head for head, _column in columns)
        if len(column_heads) != len(set(column_heads)):
            raise ValueError("score-output persisted head不得重複")
        unknown = sorted(set(column_heads) - set(names))
        if unknown:
            raise ValueError(f"score-output persisted head不存在: {unknown}")
        if any(not column for _head, column in columns):
            raise ValueError("score-output persisted column不得為空")

    def manifest_columns(self) -> dict[str, str]:
        return {"primary": "breakout_quality_score", **dict(self.persisted_columns)}


SCORE_OUTPUT_POLICY_CONDITIONAL_MFE_SAFETY = ContinuousRankerScoreOutputPolicy(
    output_head="conditional_both",
    head_names=("primary_mfe", "conditional_safety"),
    primary_head="primary_mfe",
    persisted_columns=(
        ("primary_mfe", "primary_mfe_score"),
        ("conditional_safety", "conditional_safety_score"),
    ),
)
SCORE_OUTPUT_POLICY_SAFETY_CONDITIONAL_MFE = ContinuousRankerScoreOutputPolicy(
    output_head="both",
    head_names=("raw_safety", "conditional_mfe"),
    primary_head="conditional_mfe",
    persisted_columns=(
        ("conditional_mfe", "breakout_quality_score"),
        ("raw_safety", "raw_safety_score"),
    ),
)
SCORE_OUTPUT_POLICY_SAFETY_RAW_MFE = ContinuousRankerScoreOutputPolicy(
    output_head="both",
    head_names=("raw_safety", "raw_mfe"),
    primary_head="raw_mfe",
    persisted_columns=(
        ("raw_safety", "raw_safety_score"),
        ("raw_mfe", "raw_mfe_score"),
    ),
)
SCORE_OUTPUT_POLICY_SAFETY_PRIMARY = ContinuousRankerScoreOutputPolicy(
    output_head="both",
    head_names=("raw_safety", "primary_target"),
    primary_head="primary_target",
    persisted_columns=(
        ("raw_safety", "raw_safety_score"),
        ("primary_target", "primary_target_score"),
    ),
)
SCORE_OUTPUT_POLICY_SAFETY_RAW_MFE_JOINT_MIN = ContinuousRankerScoreOutputPolicy(
    output_head="tri_head",
    head_names=("raw_safety", "raw_mfe", "joint_min"),
    primary_head="raw_mfe",
    persisted_columns=(
        ("raw_safety", "raw_safety_score"),
        ("raw_mfe", "raw_mfe_score"),
        ("joint_min", "joint_min_score"),
    ),
)


@dataclass(frozen=True)
class ContinuousRankerExecutionRecipe:
    """Experiment-agnostic execution contract derived from canonical declarations.

    Services consume this recipe for trainer/target/context/dependency/output semantics.
    Research identity (MR id, phase, experiment wording) remains in
    ContinuousRankerResearchSpec for reports/history only.  Compatibility properties keep
    existing consumers stable while Round 2 migrates them to the explicit policy objects.
    """

    profile_name: str
    trainer_family: str
    training_objective: str
    continuous_target_id: str
    loss_name: str
    model_architecture: str | None
    training_label_scope: str
    training_sample_scope: str
    score_semantic_id: str
    objective_policy: ContinuousRankerObjectivePolicy
    target_policy: ContinuousRankerTargetPolicy
    training_policy: ContinuousRankerTrainingPolicy
    context_policy: ContinuousRankerContextPolicy
    dependency_spec: ContinuousRankerDependencySpec
    output_schema: BreakoutQualityOutputSchema
    historical_pit_authorized: bool
    current_time_validation_authorized: bool

    @property
    def pairwise_reduction(self) -> str | None:
        return self.objective_policy.pairwise_reduction

    def as_dict(self) -> dict[str, Any]:
        # Preserve the pre-Round-1 payload exactly: this helper may participate in
        # manifests/fingerprints outside this module and architecture refactoring must not
        # silently create a new scientific identity.
        return {
            "profile_name": self.profile_name,
            "trainer_family": self.trainer_family,
            "training_objective": self.training_objective,
            "continuous_target_id": self.continuous_target_id,
            "loss_name": self.loss_name,
            "model_architecture": self.model_architecture,
            "training_label_scope": self.training_label_scope,
            "training_sample_scope": self.training_sample_scope,
            "score_semantic_id": self.score_semantic_id,
            "pairwise_reduction": self.pairwise_reduction,
            "historical_pit_authorized": bool(self.historical_pit_authorized),
            "current_time_validation_authorized": bool(
                self.current_time_validation_authorized
            ),
        }

@lru_cache(maxsize=1)
def _continuous_ranker_target_policies() -> dict[str, ContinuousRankerTargetPolicy]:
    return {
        "strategy_aligned_opportunity_r_v1": ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_EXTERNAL,
        ),
        "strategy_aligned_opportunity_no_time_r_v1": ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_EXTERNAL,
        ),
        "daily_opportunity_no_time_r_v1": ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            component_target_id="daily_opportunity_no_time_r_v1",
            contract_kind="daily_opportunity_no_time",
        ),
        "daily_full_horizon_opportunity_r_v1": ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            component_target_id="daily_full_horizon_opportunity_r_v1",
            contract_kind="daily_full_horizon_opportunity",
        ),
        "daily_full_horizon_pure_mfe_r_v1": ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            component_target_id="daily_full_horizon_pure_mfe_r_v1",
            contract_kind="daily_full_horizon_pure_mfe",
        ),
        "daily_first_risk_breach_pure_mfe_r_v1": ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            component_target_id="daily_first_risk_breach_pure_mfe_r_v1",
            contract_kind="daily_first_risk_breach_pure_mfe",
        ),
        "daily_full_horizon_low_adverse_r_v1": ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            component_target_id="daily_full_horizon_low_adverse_r_v1",
            contract_kind="daily_full_horizon_low_adverse",
        ),
        "daily_full_horizon_equal_rank_mfe_low_adverse_v1": ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            component_target_id="daily_full_horizon_opportunity_r_v1",
            postprocess=CONTINUOUS_RANKER_TARGET_POSTPROCESS_EQUAL_RANK_MFE_LOW_ADVERSE,
            contract_kind="daily_full_horizon_equal_rank_mfe_low_adverse",
        ),
        PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID: ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            component_target_id="daily_full_horizon_low_adverse_r_v1",
            context_source=CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
            context_roles=(
                CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE,
                CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT,
                CONTINUOUS_RANKER_CONTEXT_ROLE_TARGET_TRANSFORM,
            ),
            context_transform=CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_UPSIDE_LOW_ADVERSE,
            contract_kind="predicted_upside_context",
        ),
        PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID: ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            component_target_id="daily_full_horizon_pure_mfe_r_v1",
            context_source=CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
            context_roles=(
                CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE,
                CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT,
                CONTINUOUS_RANKER_CONTEXT_ROLE_TARGET_TRANSFORM,
            ),
            context_transform=CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_SAFETY_MFE,
            contract_kind="predicted_safety_context",
        ),
        PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID: ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            component_target_id="daily_full_horizon_pure_mfe_r_v1",
            context_source=CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
            context_roles=(
                CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE,
                CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT,
            ),
            context_transform=CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PRESERVE_PURE_MFE,
            contract_kind="predicted_safety_pure_mfe",
        ),
        "daily_risk_normalized_net_opportunity_r_v1": ContinuousRankerTargetPolicy(
            materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_RISK_NORMALIZED,
            contract_kind="risk_normalized",
        ),
    }

@lru_cache(maxsize=1)
def _continuous_ranker_training_policies() -> dict[str, ContinuousRankerTrainingPolicy]:
    return {
        TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_SHUFFLED,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_PERCENTILE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_PERCENTILE_MSE,
            profile_loss_metrics=(("mse", "mean_daily_spearman"),),
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
        ),
        TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_SHUFFLED,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_RAW_R,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_RAW_R,
            profile_loss_metrics=(
                ("huber_raw_r", "validation_huber_raw_r"),
                ("mse_raw_r", "validation_mse_raw_r"),
            ),
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_RAW_R,
            score_transform=CONTINUOUS_RANKER_SCORE_TRANSFORM_MARGIN_R,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
        ),
        TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_SHUFFLED,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_DUAL_COMPONENT_R,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_DUAL_COMPONENT_R,
            profile_loss_metrics=(("dual_mse_raw_r", "mean_daily_spearman"),),
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_DUAL_COMPONENT_R,
            score_transform=CONTINUOUS_RANKER_SCORE_TRANSFORM_MARGIN_R,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
        ),
        TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_SCALAR_PAIRWISE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SINGLE_PAIRWISE,
            profile_loss_metrics=(("pairwise_logistic", "mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_PAIRWISE,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_PARETO_COMPONENTS,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SINGLE_PAIRWISE,
            profile_loss_metrics=(("pairwise_logistic", "mean_daily_pareto_pair_concordance"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_PARETO,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_PAIRWISE,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_CONDITIONAL_MFE_SAFETY,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_CONDITIONAL_DUO_PAIRWISE,
            profile_loss_metrics=(("dual_head_pairwise_logistic", "conditional_safety_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED,
            head_loss_component_count=2,
            auxiliary_target_bundle=CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_SAFETY,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SAFETY,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_CONDITIONAL_MFE_SAFETY,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_CONDITIONAL_MFE_SINGLE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SINGLE_PAIRWISE,
            profile_loss_metrics=(("pairwise_logistic", "mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            auxiliary_target_bundle=CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SINGLE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_CONDITIONAL_MFE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_DUO_PAIRWISE,
            profile_loss_metrics=(("dual_head_pairwise_logistic", "conditional_mfe_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED,
            head_loss_component_count=2,
            auxiliary_target_bundle=CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SAFETY_CONDITIONAL_MFE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_CONDITIONAL_MFE,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_DUO_PAIRWISE,
            profile_loss_metrics=(("dual_head_pairwise_logistic", "raw_mfe_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED,
            head_loss_component_count=2,
            auxiliary_target_bundle=CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_RAW_MFE,
            report_evidence_families=("safety_raw_mfe",),
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_WEIGHTED_MFE_DUO_PAIRWISE,
            profile_loss_metrics=(("dual_head_pairwise_logistic", "raw_mfe_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED,
            head_loss_component_count=2,
            auxiliary_target_bundle=CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_MFE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_RAW_MFE,
            report_evidence_families=("safety_raw_mfe",),
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_PRIMARY,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_WEIGHTED_MFE_DUO_PAIRWISE,
            profile_loss_metrics=(("dual_head_pairwise_logistic", "mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED,
            head_loss_component_count=2,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_PRIMARY,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_PRIMARY,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE,
            profile_loss_metrics=(("dual_head_pairwise_logistic", "hs_conditional_mfe_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            allowed_secondary_pair_scopes=(
                CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
                CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
            ),
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE,
            head_loss_component_count=2,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_CONDITIONAL_MFE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_CONDITIONAL_MFE,
            report_evidence_families=("hs_conditional_mfe",),
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_HS_QUALIFICATION_SCOPED_MFE_DUO_PAIRWISE,
            profile_loss_metrics=(("dual_head_pairwise_logistic", "hs_conditional_mfe_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allowed_secondary_pair_scopes=(
                CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
                CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
            ),
            primary_supervision_mode=CONTINUOUS_RANKER_PRIMARY_SUPERVISION_BINARY_HS,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE,
            head_loss_component_count=2,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_CONDITIONAL_MFE,
            report_evidence_families=("hs_conditional_mfe",),
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_HS_QUALIFICATION_SCOPED_MFE_DUO_PAIRWISE,
            profile_loss_metrics=(("dual_head_pairwise_logistic", "hs_conditional_mfe_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allowed_secondary_pair_scopes=(
                CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
                CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
            ),
            primary_supervision_mode=CONTINUOUS_RANKER_PRIMARY_SUPERVISION_BINARY_HS,
            primary_pair_weight_policy=CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE,
            head_loss_component_count=2,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_CONDITIONAL_MFE,
            report_evidence_families=("hs_conditional_mfe",),
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_DUAL_SUPERVISED_HS_SCOPED_MFE_DUO_PAIRWISE,
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            primary_supervision_mode=CONTINUOUS_RANKER_PRIMARY_SUPERVISION_DUAL_HS,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE,
            head_loss_component_count=2,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_CONDITIONAL_MFE,
            report_evidence_families=("hs_conditional_mfe",),
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_TOP_HS_SAFETY_SCOPED_MFE_DUO_PAIRWISE,
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            primary_supervision_mode=CONTINUOUS_RANKER_PRIMARY_SUPERVISION_TOP_HS,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE,
            head_loss_component_count=2,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_CONDITIONAL_MFE,
            report_evidence_families=("hs_conditional_mfe",),
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_HS_PRIORITY_MFE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE,
            profile_loss_metrics=(("dual_head_pairwise_logistic", "hs_priority_mfe_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE,
            head_loss_component_count=2,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_MFE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_CONDITIONAL_MFE,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_HS_PRIORITY_MFE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_STRATIFIED_MFE_DUO_PAIRWISE,
            profile_loss_metrics=(("dual_head_pairwise_logistic", "hs_priority_mfe_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            secondary_supervision_mode=CONTINUOUS_RANKER_SECONDARY_SUPERVISION_STRATIFIED,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE,
            head_loss_component_count=2,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_CONDITIONAL_MFE,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE_HMHS,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_JOINT_TRI_PAIRWISE,
            profile_loss_metrics=(("tri_head_pairwise_logistic", "raw_mfe_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED,
            head_loss_component_count=3,
            auxiliary_target_bundle=CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_HMHS,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE_JOINT_MIN,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_JOINT_TRI_PAIRWISE,
            profile_loss_metrics=(("tri_head_pairwise_logistic", "raw_mfe_mean_daily_spearman"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            head_loss_combination=CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED,
            head_loss_component_count=3,
            auxiliary_target_bundle=CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_JOINT_MIN,
            epoch_loss_aggregation=CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
            score_output_policy=SCORE_OUTPUT_POLICY_SAFETY_RAW_MFE_JOINT_MIN,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_DIRECT_HMHS,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_SINGLE_PAIRWISE,
            profile_loss_metrics=(("pairwise_logistic", "hmhs_pairwise_concordance"),),
            pairwise_kind=CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
            allows_context_pair_weight=True,
            auxiliary_target_bundle=CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY,
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_DIRECT_HMHS,
            uses_pairwise_loss=True,
        ),
        TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING: ContinuousRankerTrainingPolicy(
            batch_mode=CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
            target_builder=CONTINUOUS_RANKER_TARGET_BUILDER_PERCENTILE,
            loss_handler=CONTINUOUS_RANKER_LOSS_HANDLER_LISTWISE,
            profile_loss_metrics=(("listnet_top_one_cross_entropy", "mean_daily_spearman"),),
            semantics_contract_key=CONTINUOUS_RANKER_SEMANTICS_LISTWISE,
        ),
    }

def get_continuous_ranker_primary_pair_weight_policy(training_objective: str) -> str:
    """Return truth-side primary supervision weighting from the canonical composition."""

    return str(
        get_continuous_ranker_training_policy(
            training_objective
        ).primary_pair_weight_policy
    )


def get_continuous_ranker_training_policy(
    training_objective: str,
) -> ContinuousRankerTrainingPolicy:
    try:
        return _continuous_ranker_training_policies()[str(training_objective)]
    except KeyError as exc:
        raise ValueError(
            f"continuous ranker objective缺少training capability登記: {training_objective!r}"
        ) from exc


def get_profile_enabled_continuous_ranker_training_objectives() -> tuple[str, ...]:
    """Return objectives with a complete declarative Profile loss/metric contract."""

    return tuple(
        objective
        for objective, policy in _continuous_ranker_training_policies().items()
        if policy.profile_loss_metrics
    )


def get_continuous_ranker_score_output_columns(training_objective: str) -> dict[str, str]:
    """Return the canonical persisted PIT/Forward score-column contract for an objective.

    Score capability is owned by ``ContinuousRankerTrainingPolicy.score_output_policy``.
    Consumers must not maintain objective-specific column maps: a new multi-head composition
    becomes reusable/reportable everywhere as soon as its score-output policy is registered.
    """

    policy = get_continuous_ranker_training_policy(str(training_objective))
    output_policy = policy.score_output_policy
    return (
        {"primary": "breakout_quality_score"}
        if output_policy is None
        else dict(output_policy.manifest_columns())
    )


def get_continuous_ranker_persisted_score_columns() -> tuple[str, ...]:
    """Return the runtime-owned union of optional persisted score sidecar columns."""

    columns: list[str] = []
    for training_policy in _continuous_ranker_training_policies().values():
        output_policy = training_policy.score_output_policy
        if output_policy is None:
            continue
        for _head_name, column in output_policy.persisted_columns:
            normalized = str(column)
            if normalized == "breakout_quality_score" or normalized in columns:
                continue
            columns.append(normalized)
    return tuple(columns)


def get_continuous_ranker_report_evidence_families(training_objective: str) -> tuple[str, ...]:
    """Return comparison/report evidence families from the training-composition owner."""

    return tuple(get_continuous_ranker_training_policy(training_objective).report_evidence_families)


def _resolve_continuous_ranker_target_policy(profile: Any) -> ContinuousRankerTargetPolicy:
    target_id = str(profile.continuous_target_id or "")
    try:
        return _continuous_ranker_target_policies()[target_id]
    except KeyError as exc:
        raise ValueError(
            f"continuous ranker target缺少runtime capability登記: {target_id!r}"
        ) from exc


def _resolve_continuous_ranker_context_policy(
    target_policy: ContinuousRankerTargetPolicy,
    objective_policy: ContinuousRankerObjectivePolicy,
) -> ContinuousRankerContextPolicy:
    source = str(target_policy.context_source)
    roles = list(target_policy.context_roles)
    pair_weight_policy = get_continuous_ranker_pair_weight_policy(
        objective_policy.pair_weight_policy
    )
    if pair_weight_policy.weighted:
        if source not in {
            CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
            pair_weight_policy.context_source,
        }:
            raise ValueError(
                "pair weighting不能與不同persistent context source併用"
            )
        source = pair_weight_policy.context_source
        for role in (
            CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE,
            CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT,
        ):
            if role not in roles:
                roles.append(role)
    return ContinuousRankerContextPolicy(source=source, roles=tuple(roles))


def _resolve_continuous_ranker_objective_policy(spec: Any) -> ContinuousRankerObjectivePolicy:
    reduction = spec.pairwise_reduction
    secondary_pair_scope = str(
        getattr(spec, "secondary_pair_scope", CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL)
        or CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL
    )
    secondary_pair_scope_threshold = getattr(spec, "secondary_pair_scope_threshold", None)
    declared_pair_weight_policy = getattr(spec, "pair_weight_policy", None)
    pair_weight_policy = (
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE
        if declared_pair_weight_policy is None
        else str(declared_pair_weight_policy)
    )
    _base_reduction, resolved_pair_weight_policy = (
        normalize_continuous_ranker_pair_weight_configuration(
            reduction, pair_weight_policy
        )
    )
    if reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE:
        if resolved_pair_weight_policy.weighted:
            raise ValueError("Pareto pair scope不得再疊加scalar pair weight policy")
        return ContinuousRankerObjectivePolicy(
            pairwise_reduction=reduction,
            pair_target_schema=CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_PARETO_COMPONENTS,
            secondary_pair_scope=secondary_pair_scope,
            secondary_pair_scope_threshold=secondary_pair_scope_threshold,
        )
    if resolved_pair_weight_policy.weighted:
        return ContinuousRankerObjectivePolicy(
            pairwise_reduction=reduction,
            pair_weight_policy=resolved_pair_weight_policy.policy_id,
            pair_target_schema=CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT,
            secondary_pair_scope=secondary_pair_scope,
            secondary_pair_scope_threshold=secondary_pair_scope_threshold,
        )
    return ContinuousRankerObjectivePolicy(
        pairwise_reduction=reduction,
        secondary_pair_scope=secondary_pair_scope,
        secondary_pair_scope_threshold=secondary_pair_scope_threshold,
    )


def build_continuous_ranker_execution_recipe(
    *,
    profile_name: str,
    profile: Any,
    spec: Any,
    requires_continuous_target_artifact: bool,
) -> ContinuousRankerExecutionRecipe:
    """Build executable semantics from already-resolved declarative inputs.

    This module deliberately does not import the scientific/profile configuration owner.
    Resolution of profile identity belongs to ``config.breakout_quality``; the runtime
    owner only turns resolved declarations into reusable execution capabilities.
    """

    target_policy = _resolve_continuous_ranker_target_policy(profile)
    training_policy = get_continuous_ranker_training_policy(profile.training_objective)
    objective_policy = _resolve_continuous_ranker_objective_policy(spec)
    context_policy = _resolve_continuous_ranker_context_policy(
        target_policy, objective_policy
    )
    dependency_spec = ContinuousRankerDependencySpec(
        requires_continuous_target_artifact=bool(requires_continuous_target_artifact),
        context_source=context_policy.source,
    )
    return ContinuousRankerExecutionRecipe(
        profile_name=str(profile_name),
        trainer_family=spec.trainer_family,
        training_objective=profile.training_objective,
        continuous_target_id=str(profile.continuous_target_id),
        loss_name=profile.loss_name,
        model_architecture=profile.model_architecture,
        training_label_scope=profile.training_label_scope,
        training_sample_scope=profile.training_sample_scope,
        score_semantic_id=spec.score_semantic_id,
        objective_policy=objective_policy,
        target_policy=target_policy,
        training_policy=training_policy,
        context_policy=context_policy,
        dependency_spec=dependency_spec,
        output_schema=BREAKOUT_QUALITY_OUTPUT_SCHEMA,
        historical_pit_authorized=bool(spec.selection_pit_authorized),
        current_time_validation_authorized=bool(
            spec.current_time_validation_authorized
        ),
    )


__all__ = (
    "CONTINUOUS_RANKER_TRAINER_EVENT",
    "CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL",
    "CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR",
    "CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED",
    "CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE",
    "CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG",
    "CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG",
    "CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE",
    "CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE",
    "CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE",
    "CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY",
    "CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE",
    "CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT",
    "CONTINUOUS_RANKER_CONTEXT_ROLE_TARGET_TRANSFORM",
    "CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT",
    "CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE",
    "CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY",
    "CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY",
    "CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY",
    "CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY",
    "CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_NONE",
    "CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY",
    "SUPPORTED_CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICIES",
    "CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL",
    "CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN",
    "SUPPORTED_CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPES",
    "CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_ALL",
    "CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_CROSS",
    "CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_WITHIN_POSITIVE",
    "SUPPORTED_CONTINUOUS_RANKER_PAIR_PARTITION_RELATIONS",
    "SUPPORTED_CONTINUOUS_RANKER_PAIR_WEIGHT_POLICIES",
    "PREDICTED_SAFETY_CONTINUOUS_RANKER_PAIR_WEIGHT_POLICIES",
    "ContinuousRankerPairWeightPolicy",
    "get_continuous_ranker_pair_weight_policy",
    "normalize_continuous_ranker_pair_weight_configuration",
    "CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR",
    "CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_PARETO_COMPONENTS",
    "CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT",
    "SUPPORTED_CONTINUOUS_RANKER_PAIRWISE_REDUCTIONS",
    "ContinuousRankerContextPolicy",
    "CONTINUOUS_RANKER_TARGET_MATERIALIZATION_EXTERNAL",
    "CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT",
    "CONTINUOUS_RANKER_TARGET_MATERIALIZATION_RISK_NORMALIZED",
    "CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE",
    "CONTINUOUS_RANKER_TARGET_POSTPROCESS_EQUAL_RANK_MFE_LOW_ADVERSE",
    "CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_NONE",
    "CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_UPSIDE_LOW_ADVERSE",
    "CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_SAFETY_MFE",
    "CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PRESERVE_PURE_MFE",
    "CONTINUOUS_RANKER_BATCH_MODE_SHUFFLED",
    "CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT",
    "CONTINUOUS_RANKER_TARGET_BUILDER_PERCENTILE",
    "CONTINUOUS_RANKER_TARGET_BUILDER_SCALAR_PAIRWISE",
    "CONTINUOUS_RANKER_TARGET_BUILDER_RAW_R",
    "CONTINUOUS_RANKER_TARGET_BUILDER_DUAL_COMPONENT_R",
    "CONTINUOUS_RANKER_TARGET_BUILDER_PARETO_COMPONENTS",
    "CONTINUOUS_RANKER_TARGET_BUILDER_CONDITIONAL_MFE_SAFETY",
    "CONTINUOUS_RANKER_TARGET_BUILDER_CONDITIONAL_MFE_SINGLE",
    "CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_CONDITIONAL_MFE",
    "CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE",
    "CONTINUOUS_RANKER_TARGET_BUILDER_HS_PRIORITY_MFE",
    "CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE_HMHS",
    "CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE_JOINT_MIN",
    "CONTINUOUS_RANKER_TARGET_BUILDER_DIRECT_HMHS",
    "CONTINUOUS_RANKER_LOSS_HANDLER_PERCENTILE_MSE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_RAW_R",
    "CONTINUOUS_RANKER_LOSS_HANDLER_DUAL_COMPONENT_R",
    "CONTINUOUS_RANKER_LOSS_HANDLER_SINGLE_PAIRWISE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_CONDITIONAL_DUO_PAIRWISE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_DUO_PAIRWISE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_WEIGHTED_MFE_DUO_PAIRWISE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_HS_QUALIFICATION_SCOPED_MFE_DUO_PAIRWISE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_DUAL_SUPERVISED_HS_SCOPED_MFE_DUO_PAIRWISE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_TOP_HS_SAFETY_SCOPED_MFE_DUO_PAIRWISE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_STRATIFIED_MFE_DUO_PAIRWISE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_JOINT_TRI_PAIRWISE",
    "CONTINUOUS_RANKER_LOSS_HANDLER_LISTWISE",
    "CONTINUOUS_RANKER_AUX_TARGET_NONE",
    "CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_SAFETY",
    "CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY",
    "CONTINUOUS_RANKER_SEMANTICS_DEFAULT",
    "CONTINUOUS_RANKER_SEMANTICS_PAIRWISE",
    "CONTINUOUS_RANKER_SEMANTICS_LISTWISE",
    "CONTINUOUS_RANKER_SEMANTICS_RAW_R",
    "CONTINUOUS_RANKER_SEMANTICS_DUAL_COMPONENT_R",
    "CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SAFETY",
    "CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SINGLE",
    "CONTINUOUS_RANKER_SEMANTICS_SAFETY_CONDITIONAL_MFE",
    "CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE",
    "CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_MFE",
    "CONTINUOUS_RANKER_SEMANTICS_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE",
    "CONTINUOUS_RANKER_SEMANTICS_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE",
    "CONTINUOUS_RANKER_SEMANTICS_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE",
    "CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_MFE",
    "CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE",
    "CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_HMHS",
    "CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_JOINT_MIN",
    "CONTINUOUS_RANKER_SEMANTICS_DIRECT_HMHS",
    "CONTINUOUS_RANKER_SCORE_TRANSFORM_PROBABILITY",
    "CONTINUOUS_RANKER_SCORE_TRANSFORM_MARGIN_R",
    "CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH",
    "CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_WEIGHTED",
    "CONTINUOUS_RANKER_PAIRWISE_KIND_NONE",
    "CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR",
    "CONTINUOUS_RANKER_PAIRWISE_KIND_PARETO",
    "CONTINUOUS_RANKER_PRIMARY_SUPERVISION_CONTINUOUS",
    "CONTINUOUS_RANKER_PRIMARY_SUPERVISION_BINARY_HS",
    "CONTINUOUS_RANKER_PRIMARY_SUPERVISION_DUAL_HS",
    "CONTINUOUS_RANKER_PRIMARY_SUPERVISION_TOP_HS",
    "CONTINUOUS_RANKER_SECONDARY_SUPERVISION_STANDARD",
    "CONTINUOUS_RANKER_SECONDARY_SUPERVISION_STRATIFIED",
    "CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_SINGLE",
    "CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED",
    "CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE",
    "ContinuousRankerTargetPolicy",
    "ContinuousRankerTrainingPolicy",
    "ContinuousRankerScoreOutputPolicy",
    "SCORE_OUTPUT_POLICY_CONDITIONAL_MFE_SAFETY",
    "SCORE_OUTPUT_POLICY_SAFETY_CONDITIONAL_MFE",
    "SCORE_OUTPUT_POLICY_SAFETY_RAW_MFE",
    "SCORE_OUTPUT_POLICY_SAFETY_RAW_MFE_JOINT_MIN",
    "ContinuousRankerObjectivePolicy",
    "ContinuousRankerDependencySpec",
    "BreakoutQualityOutputSchema",
    "BREAKOUT_QUALITY_OUTPUT_SCHEMA",
    "ContinuousRankerExecutionRecipe",
    "get_continuous_ranker_training_policy",
    "get_profile_enabled_continuous_ranker_training_objectives",
    "get_continuous_ranker_primary_pair_weight_policy",
    "get_continuous_ranker_score_output_columns",
    "get_continuous_ranker_persisted_score_columns",
    "get_continuous_ranker_report_evidence_families",
    "build_continuous_ranker_execution_recipe",
)

"""Execution-effective Oracle artifact contracts and completed-result references.

Like Strategy Compare's evaluation/score projections, these contracts exclude
publication metadata, model inventories and physical source-code bytes. A
result-affecting implementation fix revises its capability's calculation
revision here; a refactor or unused training capability preserves it. Source
SHA256 stays in the immutable originating receipt as provenance.

This owner never loads prices, score arrays, labels, models or an Optimizer.
"""
from __future__ import annotations

import json
import ast
import importlib
from hashlib import sha256
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from core.file_integrity import (
    canonical_json_sha256, is_sha256_hex, exclusive_file_lock, atomic_replace_with_retry,
)
from core.research_execution_equivalence import reviewed_calculation_sha256

from core.daily_trade_path_capabilities import DAILY_TRUTH_CAPABILITIES, daily_truth_capability

# Engineering calculation revisions, not scientific Target/model identities.
# Change only the affected capability after a result-affecting fix. Existing
# recipes are stable when a new recipe is registered alongside them.
PRICE_CALCULATION_REVISIONS = {
    "equal_rank_mfe_safety": 1, "mfe": 1, "terminal_r": 1,
    "terminal_mae_gate": 1, "mfe_mae_gate": 1,
}
DAILY_CALCULATION_REVISIONS = {
    "daily_trade_path_net_r": 1,
    "daily_trade_path_net_pnl": 1,
    "daily_trade_path_true_net_pnl": 2,
    "daily_trade_path_full_net_pnl": 1,
    "daily_trade_path_daily_min_p19_net_pnl": 1,
    "daily_trade_path_daily_full_p20_net_pnl": 1,
}
DAILY_CALCULATION_REVISIONS.update({recipe: spec["calculation_revision"]
                                    for recipe, spec in DAILY_TRUTH_CAPABILITIES.items()})
FEATURE_UNIVERSE_CALCULATION_REVISION = 1
REPLAY_CALCULATION_REVISION = 1
_SCHEMA = 1
_REFERENCE_FIELDS = {"reference_arm_id", "reference_score_source_id", "reference_profile_id"}
_SCORE_REFERENCE_FIELDS = _REFERENCE_FIELDS | {"positive_score_source_id", "positive_profile_id"}
_SCORE_WITNESS_FIELDS = {"canonical_feature_eligible_keys_sha256", "oracle_scored_rows", "source_evidence"}

# The perfect-Positive intervention supplies the *same* signed, continuous
# Target-percentile rank and exact Positive 0/1 column before and after a
# predictor-topology change. These are approved Oracle output interfaces,
# NOT an equivalence of the historical fitted models or their PIT scores.
_ORACLE_PERFECT_POSITIVE_INTERFACE = "oracle_perfect_netpnl_rank_and_positive_v1"
_LEGACY_POSITIVE_INTERFACE = "positive_netpnl_dual_output_v1"
_SINGLE_RANKER_POSITIVE_INTERFACE = "single_ranker_calibrated_positive_v2"
_POSITIVE_OUTPUT_COLUMN = "positive_p_positive"
_POSITIVE_OUTPUT_ID = "netpnl_positive_probability_v1"
_LEGACY_POSITIVE_SOURCE_SHA256 = "56fdce12e331011e7423c7fa0f1c8b27563b43bbd731fcff938c6a0c602edd05"
# Reviewed executable AST of requires_positive_score, positive_score_admitted,
# and admit_positive_score_rows, *not* training/label-calibration code. An
# executable gate change fails this check and cannot inherit the old Replay.
# Executable AST audited for the current distinct-DL score consumer. The row
# admission decision on valid two-score Oracle inputs is unchanged: identical
# information day, available scores, finite probability > 0.5.
_REVIEWED_POSITIVE_ADMISSION_AST_SHA256 = "bbda38329e4ba1237c69b7082783a8519fbcf9c02daa2e58c72e2267d5e1795a"
# The immediately preceding single-ranker code had the same row-level
# admission behavior and is a reviewed, immutable past signed Oracle producer.
_PREVIOUS_SINGLE_RANKER_SOURCE_SHA256 = "bd21e2953973e6602fd6a901c6bca2f2a5593c518b2b4c7c31482376fbf2943a"


@lru_cache(maxsize=1)
def _positive_admission_source_certificate() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    path = root / "core/positive_netpnl_contract.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = {"requires_positive_score", "positive_score_admitted", "admit_positive_score_rows"}
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names):
        return sha256(path.read_bytes()).hexdigest(), False
    projected = _CalculationSyntax().visit(ast.Module(body=selected, type_ignores=[]))
    digest = canonical_json_sha256(_syntax_payload(projected))
    return sha256(path.read_bytes()).hexdigest(), digest == _REVIEWED_POSITIVE_ADMISSION_AST_SHA256


def _approved_perfect_positive_admission(value: Any) -> bool:
    # Historical binary/dual-model routing is accepted only for the original
    # reviewed signed producer. The current score-admission function must
    # retain its reviewed executable meaning; unrelated FIT-only edits do not
    # affect that check. Unknown historical revisions cannot be asserted equal.
    current_sha, approved = _positive_admission_source_certificate()
    return (approved and
            (value == _ORACLE_PERFECT_POSITIVE_INTERFACE
             or (is_sha256_hex(value) and value in {
                 _LEGACY_POSITIVE_SOURCE_SHA256, _PREVIOUS_SINGLE_RANKER_SOURCE_SHA256, current_sha})))


def oracle_score_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Only predictor topology is erased from the signed perfect-score interface."""
    result = {key: value for key, value in contract.items() if key not in _SCORE_REFERENCE_FIELDS}
    if (result.get("perfect_positive_output") == _POSITIVE_OUTPUT_ID
            and result.get("positive_score_column") == _POSITIVE_OUTPUT_COLUMN
            and result.get("positive_score_contract_revision") in {
                _LEGACY_POSITIVE_INTERFACE, _SINGLE_RANKER_POSITIVE_INTERFACE}):
        result["positive_score_contract_revision"] = _ORACLE_PERFECT_POSITIVE_INTERFACE
    return result


def oracle_ranking_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize *only* two certified perfect-Positive score-consumers.

    Ordinary strategy FIT/Score/Replay identities are not affected; they still
    distinguish independent binary predictors from calibrated rankers. Unknown
    or malformed Positive options are retained as distinct economic inputs.
    """
    result = dict(options)
    positive_keys = {str(key) for key in result if str(key).startswith("positive_")}
    if not positive_keys:
        return result
    allowed = {"positive_dl_id", "positive_prediction_contract", "positive_probability_threshold",
               "positive_score_column"}
    old = result.get("positive_dl_id")
    current = result.get("positive_prediction_contract")
    threshold = result.get("positive_probability_threshold")
    valid_threshold = (isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
                       and 0.0 < float(threshold) < 1.0)
    if (positive_keys <= allowed and valid_threshold
            and result.get("positive_score_column") == _POSITIVE_OUTPUT_COLUMN
            and (("positive_dl_id" in result and "positive_prediction_contract" not in result
                  and isinstance(old, str) and bool(old.strip()))
                 or ("positive_dl_id" not in result and "positive_prediction_contract" in result
                     and current == _SINGLE_RANKER_POSITIVE_INTERFACE))):
        result.pop("positive_dl_id", None)
        result.pop("positive_prediction_contract", None)
        result["oracle_positive_output_interface"] = _ORACLE_PERFECT_POSITIVE_INTERFACE
    return result


def oracle_replay_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Remove reference names, not score values, effective gates or economics."""
    result = {key: value for key, value in policy.items() if key not in {"exact_arm", "direct_arm"}}
    positive = result.get("formal_positive_score_output")
    if isinstance(positive, Mapping):
        positive = {key: value for key, value in positive.items() if key != "selector_arm_id"}
        if (positive.get("contract") == _POSITIVE_OUTPUT_ID
                and set(positive) == {"contract", "shared_admission_code_sha256"}
                and _approved_perfect_positive_admission(
                    positive["shared_admission_code_sha256"])):
            positive["shared_admission_code_sha256"] = _ORACLE_PERFECT_POSITIVE_INTERFACE
        result["formal_positive_score_output"] = positive
    contract = result.get("canonical_dl_score_contract")
    if isinstance(contract, Mapping):
        result["canonical_dl_score_contract"] = oracle_score_contract(contract)
    return result


def oracle_execution_contract(context: Mapping[str, Any]) -> dict[str, Any]:
    """The same effective projection serves planning, recovery and publication."""
    result = dict(context)
    result["ranking_options"] = oracle_ranking_options(context["ranking_options"])
    result["replay_policy"] = oracle_replay_policy(context["replay_policy"])
    if context.get("score_contract") is not None:
        result["score_contract"] = oracle_score_contract(context["score_contract"])
    return result


class _CalculationSyntax(ast.NodeTransformer):
    """Ignore comments/docstrings and inactive target-ID branches, not math.

    Only literal comparisons with the bound target ID are folded. No source
    expression is executed, and every unknown condition remains in the proof.
    Registering another target branch therefore cannot revise an existing
    target's calculation, while changing its selected arithmetic does.
    """
    def __init__(self, literals=None):
        self.literals = dict(literals or {})

    def _literal(self, node):
        if isinstance(node, ast.Name) and node.id in self.literals:
            return self.literals[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            return tuple(self._literal(value) for value in node.elts)
        raise ValueError("not a bound literal")

    def _condition(self, node):
        if isinstance(node, (ast.Name, ast.Constant)):
            value = self._literal(node)
            if isinstance(value, bool):
                return value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self._literal(node.left), self._literal(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.In):
                return left in right
            if isinstance(op, ast.NotIn):
                return left not in right
            if isinstance(op, ast.Eq):
                return left == right
            if isinstance(op, ast.NotEq):
                return left != right
            if isinstance(op, ast.Is):
                return left is right
            if isinstance(op, ast.IsNot):
                return left is not right
        if isinstance(node, ast.BoolOp):
            known = []
            for value in node.values:
                try:
                    known.append(self._condition(value))
                except ValueError:
                    known.append(None)
            if isinstance(node.op, ast.And) and False in known:
                return False
            if isinstance(node.op, ast.Or) and True in known:
                return True
            if None not in known:
                return all(known) if isinstance(node.op, ast.And) else any(known)
        raise ValueError("not a bound comparison")

    def visit_If(self, node):
        try:
            branch = node.body if self._condition(node.test) else node.orelse
        except ValueError:
            before = dict(self.literals)
            node.body = self._statements(node.body)
            self.literals = dict(before)
            node.orelse = self._statements(node.orelse)
            self.literals = before
            return node
        return self._statements(branch)

    def _statements(self, statements):
        result = []
        for statement in statements:
            transformed = self.visit(statement)
            if transformed is not None:
                result.extend(transformed if isinstance(transformed, list) else [transformed])
        return result

    def visit_IfExp(self, node):
        try:
            selected = node.body if self._condition(node.test) else node.orelse
        except ValueError:
            return self.generic_visit(node)
        return self.visit(selected)

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    self.literals[target.id] = self._literal(node.value)
                except ValueError:
                    self.literals.pop(target.id, None)
        return node

    def _local_scope(self, node):
        before = dict(self.literals)
        node = self.generic_visit(node)
        self.literals = before
        return node

    visit_For = _local_scope
    visit_While = _local_scope
    visit_Try = _local_scope
    visit_ClassDef = _local_scope

    def visit_FunctionDef(self, node):
        # Annotations and docstrings do not execute under future annotations.
        node.returns = None
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            argument.annotation = None
        return self._local_scope(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def _syntax_payload(value):
    # ast.dump's treatment of optional/empty fields changed in Python 3.13.
    # This structural encoding is stable across the supported interpreters,
    # including the user's Python 3.14. Locations/empty new AST fields are not
    # execution semantics and must not invalidate an existing result.
    if isinstance(value, ast.AST):
        return {"node": type(value).__name__, **{
            name: _syntax_payload(item) for name, item in ast.iter_fields(value)
            if item is not None and item != []}}
    if isinstance(value, list):
        return [_syntax_payload(item) for item in value]
    return value


def calculation_source(relative: str, name: str, *, bound_literals=None) -> str:
    """Read a named numerical owner, independent of runtime provider mocks."""
    code_root = Path(__file__).resolve().parents[2]
    module = importlib.import_module(relative[:-3].replace("/", "."))
    tree = ast.parse((code_root / relative).read_text(encoding="utf-8-sig"))
    selected = next(item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.ClassDef))
                    and item.name == name)
    literals = {key: value for key, value in vars(module).items()
                if key.isupper() and isinstance(value, (str, int, float, bool, type(None)))}
    literals.update(bound_literals or {})
    projected = _CalculationSyntax(literals).visit(ast.Module(body=[selected], type_ignores=[]))
    return reviewed_calculation_sha256(
        relative, sha256((code_root / relative).read_bytes()).hexdigest(),
        canonical_json_sha256(_syntax_payload(projected)),
    )


@lru_cache(maxsize=1)
def replay_implementation_sha256() -> str:
    """Numerical trade owners only; models, loaders and report code excluded."""
    code_root = Path(__file__).resolve().parents[2]
    owners = (
        "core/position_step.py", "core/position_management.py", "core/exit_priority.py",
        "core/entry_plans.py", "core/extended_signals.py", "core/order_lot_policy.py",
        "core/exact_accounting.py", "core/fee_rebate.py", "core/price_utils.py",
        "core/selection_policy.py", "core/signal_utils.py", "core/breakout_false_filter.py",
        "core/capital_policy.py",
    )
    return canonical_json_sha256({relative: _syntax_payload(_CalculationSyntax().visit(
        ast.parse((code_root / relative).read_text(encoding="utf-8-sig")))) for relative in owners})


def broker_cash_economics_contract() -> dict[str, Any]:
    """Economic terms actually consumed by Research Broker-Cash replay.

    Do not include the electronic-discount reference ratio: refunds are manual
    Trading cash operations, not current Research fee receivables.
    """
    from config import broker_accounting as broker
    from config.execution_policy import EXECUTION_POLICY_PARAM_SPECS
    return {
        "broker_policy_id": broker.BROKER_ACCOUNTING_POLICY_ID,
        "fee_rate": str(broker.BROKER_FEE_RATE),
        "min_fee_twd": str(broker.BROKER_MIN_FEE_TWD),
        "odd_lot_min_fee_twd": str(broker.BROKER_ODD_LOT_MIN_FEE_TWD),
        "board_lot_shares": int(broker.BROKER_BOARD_LOT_SHARES),
        "stock_tax_rate": str(broker.BROKER_STOCK_TAX_RATE),
        "rounding_mode": broker.BROKER_CHARGE_ROUNDING_MODE,
        "strategy_param_defaults": {
            name: spec["default"] for name, spec in EXECUTION_POLICY_PARAM_SPECS.items()
        },
    }


@lru_cache(maxsize=4)
def portfolio_candidate_execution_sha256(*, daily_execution_contract: str | None = None) -> str:
    """Attest executable portfolio capabilities for a registered execution scope.

    This extends the shared replay witness without revising the frozen
    ``replay_implementation_sha256`` consumed by existing P16 Oracle receipts.
    The dependency inventory covers the candidate/capacity/entry/exit/equity
    owners reached through ``run_portfolio_timeline`` and its replay adapter.
    AST projection ignores comments and docstrings but captures executable
    changes, including daily candidate capacity and selector decisions.
    """
    code_root = Path(__file__).resolve().parents[2]
    owners = (
        "core/portfolio_engine.py", "core/portfolio_day_candidates.py",
        "core/portfolio_candidate_admission.py", "core/portfolio_candidates.py",
        "core/portfolio_day_context.py", "core/portfolio_day_events.py",
        "core/portfolio_replay_support.py", "core/portfolio_benchmark.py",
        "core/portfolio_day_equity.py", "core/portfolio_day_signal_lifecycle.py",
        "core/portfolio_capacity_rows.py", "core/portfolio_terminal_equity.py",
        "core/portfolio_terminal_statistics.py",
        "core/portfolio_terminal_reporting.py", "core/portfolio_run_result.py",
        "core/portfolio_entries.py",
        "core/portfolio_exits.py", "core/portfolio_entry_selection.py",
        "core/portfolio_entry_selection_common.py",
        "core/portfolio_entry_selection_max_dl.py",
        "core/portfolio_ensemble.py", "core/portfolio_entry_plans.py",
        "core/portfolio_entry_policy.py", "core/portfolio_param_runtime.py",
        "core/portfolio_fast_data.py", "core/portfolio_stats.py",
        "core/portfolio_fast_access.py",
        "core/buy_sort.py", "core/backtest_core.py", "core/backtest_finalize.py",
        "core/seed_ensemble_policy.py", "core/history_filters.py",
        "services/portfolio_replay.py", "services/portfolio_replay_runtime.py",
    )
    # Optional capabilities belong only to the arms that execute them.  A
    # revision to Daily Full candidate admission must never invalidate a
    # Min ATR-limit or any legacy Breakout/Daily replay.  The shared portfolio
    # owners above remain strict for all newly registered execution scopes.
    from core.portfolio_entry_policy import (
        DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS,
        resolve_daily_execution_contract,
    )
    mode = resolve_daily_execution_contract(daily_execution_contract)
    if mode not in {DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS}:
        raise ValueError("Scoped portfolio source requires registered new Daily execution")
    optional = (("core/daily_common_entry_filters.py",)
                if mode == DAILY_EXECUTION_FULL_FILTERS else ())
    parts = {"daily_execution_contract": mode}
    for relative in dict.fromkeys((*owners, *optional)):
        path = code_root / relative
        if not path.is_file():
            raise ValueError(f"Oracle replay numerical owner missing: {relative}")
        parsed = ast.parse(path.read_text(encoding="utf-8-sig"))
        parts[relative] = canonical_json_sha256(
            _syntax_payload(_CalculationSyntax().visit(parsed)))
    return canonical_json_sha256(parts)


@lru_cache(maxsize=32)
def source_implementation_sha256(recipe: str, profile_id: str) -> str:
    parts = {"daily_percentile": calculation_source(
        "filters/breakout_quality/continuous_ranker_data.py", "build_same_date_percentile_targets")}
    capability = daily_truth_capability(recipe)
    if capability is not None:
        parts.update({key: calculation_source(relative, name)
                      for key, relative, name in capability["calculation_owners"]})
    elif recipe == "daily_trade_path_daily_min_p19_net_pnl":
        # P19 is a 4D ATR-limit/no-setup Daily primitive. Both the wrapper
        # and the reused canonical broker simulation are signed; no P18
        # source/teacher is permitted as an alias.
        parts.update({
            "p19_entry": calculation_source(
                "filters/breakout_quality/daily_min_p19_target.py", "compute_daily_min_p19_batch"),
            "shared_next_session_broker": calculation_source(
                "filters/breakout_quality/daily_full_p20_target.py", "compute_daily_full_p20_batch"),
            "daily_atr_limit": calculation_source(
                "core/daily_common_entry_filters.py", "daily_common_filter_signals"),
            "stock_outcomes": calculation_source(
                "services/audit/daily_min_p19_oracle_truth.py", "_stock_outcome"),
            "truth_cohort": calculation_source(
                "services/audit/daily_min_p19_oracle_truth.py", "build_daily_min_p19_oracle_scores"),
            "rank_outcomes": calculation_source(
                "filters/breakout_quality/daily_trade40_target.py", "rank_trade40_outcome_frame"),
            "schedule": calculation_source(
                "filters/breakout_quality/daily_trade40_target.py", "build_formal_min_execution_schedule"),
            "active_period": calculation_source(
                "filters/breakout_quality/daily_trade40_target.py", "resolve_historical_min_period"),
            "broker_path": calculation_source(
                "filters/breakout_quality/trade_path_label.py", "simulate_realized_trade_path_label"),
        })
    elif recipe == "daily_trade_path_daily_full_p20_net_pnl":
        # P20 uses a separate stock-day entry/filter/buy-limit decision. It
        # shares canonical broker/period/ranking primitives with Min, but the
        # original event-driven and TrueDaily source hashes must not change.
        # Bind the source to the actual P20 numerical owners, not the legacy
        # compute_counterfactual_trade40_batch path (which is not executed).
        parts.update({
            "daily_full_entry": calculation_source(
                "filters/breakout_quality/daily_full_p20_target.py",
                "compute_daily_full_p20_batch"),
            "daily_full_filters": calculation_source(
                "core/daily_common_entry_filters.py", "daily_common_filter_signals"),
            "stock_outcomes": calculation_source(
                "services/audit/daily_full_p20_oracle_truth.py", "_stock_outcome"),
            "truth_cohort": calculation_source(
                "services/audit/daily_full_p20_oracle_truth.py",
                "build_daily_full_p20_oracle_scores"),
            "rank_outcomes": calculation_source(
                "filters/breakout_quality/daily_trade40_target.py", "rank_trade40_outcome_frame"),
            "schedule": calculation_source(
                "filters/breakout_quality/daily_trade40_target.py", "build_formal_min_execution_schedule"),
            "active_period": calculation_source(
                "filters/breakout_quality/daily_trade40_target.py", "resolve_historical_min_period"),
            "broker_path": calculation_source(
                "filters/breakout_quality/trade_path_label.py", "simulate_realized_trade_path_label"),
        })
    elif recipe == "daily_trade_path_true_net_pnl":
        # New source capability uses the same canonical Min ledger but a new
        # independent every-stock-day entry decision. Old recipes are unchanged.
        parts.update({
            "true_daily_entry": calculation_source(
                "filters/breakout_quality/true_daily_mintrade_target.py",
                "compute_true_daily_mintrade_batch"),
            "rank_outcomes": calculation_source(
                "filters/breakout_quality/daily_trade40_target.py", "rank_trade40_outcome_frame"),
            "schedule": calculation_source(
                "filters/breakout_quality/daily_trade40_target.py", "build_formal_min_execution_schedule"),
            "active_period": calculation_source(
                "filters/breakout_quality/daily_trade40_target.py", "resolve_historical_min_period"),
            "broker_path": calculation_source(
                "filters/breakout_quality/trade_path_label.py", "simulate_realized_trade_path_label"),
        })
    elif recipe in DAILY_CALCULATION_REVISIONS:
        parts.update({key: calculation_source("filters/breakout_quality/daily_trade40_target.py", name,
                         bound_literals=({"signal_state_reset_date": None,
                                          "include_net_pnl": recipe in {"daily_trade_path_net_pnl", "daily_trade_path_full_net_pnl"}}
                                         if key == "min_path" else None))
                      for key, name in (
                          ("min_path", "compute_counterfactual_trade40_batch"),
                          ("rank_outcomes", "rank_trade40_outcome_frame"),
                          ("schedule", "build_formal_min_execution_schedule"),
                          ("active_period", "resolve_historical_min_period"))})
        parts["broker_path"] = calculation_source(
            "filters/breakout_quality/trade_path_label.py", "simulate_realized_trade_path_label")
        if recipe == "daily_trade_path_full_net_pnl":
            # The economic order/exit engine is shared; the Full/P17 source
            # and historical PIT parameter lineage must have a distinct AST.
            # The old Min recipe fingerprint is deliberately unchanged.
            parts["full_source_schedule"] = calculation_source(
                "services/breakout_quality/formal_fulltrade_training_source.py",
                "_full_execution_schedule")
            parts["full_oracle_truth"] = calculation_source(
                "services/audit/fulltrade_oracle_truth.py",
                "build_fulltrade_oracle_scores")
    else:
        from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
        from filters.breakout_quality import continuous_target as targets
        selected = {
            "terminal_r": targets.DAILY_TERMINAL_R40_TARGET_ID,
            "terminal_mae_gate": targets.DAILY_TERMINAL_MAE_GATED_T40_TARGET_ID,
            "mfe_mae_gate": targets.DAILY_FULL_HORIZON_MAE_GATED_MFE_T40_TARGET_ID,
        }.get(recipe, get_continuous_ranker_execution_recipe(profile_id).target_policy.component_target_id)
        parts["selected_target"] = calculation_source(
            "filters/breakout_quality/daily_ranker_data.py", "compute_daily_opportunity_target_batch",
            bound_literals={"target_id": selected})
        if recipe in {"equal_rank_mfe_safety", "mfe_mae_gate"}:
            parts["mae_safety"] = calculation_source("filters/breakout_quality/continuous_target.py",
                                                   "full_horizon_mae_safety_r_from_minimum_low_return")
        if recipe in {"terminal_mae_gate", "mfe_mae_gate"}:
            parts["mae_gate"] = calculation_source("filters/breakout_quality/continuous_target.py",
                                                 "full_horizon_mae_gated_reward_r")
    return canonical_json_sha256(parts)


# Frozen *format migration* boundary, not a producer-SHA alias list. Legacy
# receipt format 1 did not record these capability projections. Import is
# allowed only into the reviewed initial calculation, never into a later fix.
# New revisions/recipes do not add entries here; they publish native indexes.
_LEGACY_V1_REPLAY_IMPLEMENTATION = "ec364be05b82d31a326868aebb7119589ee4b814dcd292243e553986b7f9c472"
_LEGACY_V1_FEATURE_INPUT_REQUIREMENTS = "c6989308a50302e25c1e0cb46c146f7bef24e2280955a54f585e57b13474c00c"
_LEGACY_V1_SOURCE_IMPLEMENTATIONS = {
    "equal_rank_mfe_safety": "d12696ad9f69742e9e52c5abfb5ca74365e8cc43dd4d1279a6d5901f99d95a80",
    "mfe": "1851a82f7617df3baef866446c5f66bec0f1d7a4667aeed6dccc4c852d3d6c1b",
    "terminal_r": "bfedfcb1cb79947e71420d15fcd09d603383b104956f58fe99e9480b3e0d97d5",
    "terminal_mae_gate": "074344163eb6b800c23be81f8936c3ae05d44f8b6c6a76576528dd881502f743",
    "mfe_mae_gate": "255e6968508b4567ab8e8225ededfd95fa8dbad2992ba67046542ed58c06d871",
    "daily_trade_path_net_r": "60f35bf5fd5ef11da38f18999088c542d271f83fe3440e8f5d100bb65a92c24d",
    "daily_trade_path_net_pnl": "3693bbbf2d8822688febf60f39b00ddc93a5fed06626916f867788813c4947c6",
}


def dataset_input_contract(root: Path, filter_id: str) -> dict[str, Any] | None:
    """Only fields consumed by daily eligibility; content prices bind separately.

    Event-label counts, build timings, storage paths, publication timestamps
    and event-bank serialization do not define the universal stock-day rows.
    The canonical daily loader consumes source end/outer policy and the label
    policy; feature column order and market-set contract remain explicit.
    """
    from filters.breakout_quality.dataset_store import resolve_dataset_paths
    from filters.breakout_quality.paths import resolve_filter_output_dir
    path = resolve_dataset_paths(resolve_filter_output_dir(Path(root), filter_id)).summary
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Oracle Dataset summary must be an object")
    return {key: data[key] for key in (
        "dataset", "filter_id", "source_data_date_range", "source_data_end_date", "policy", "label_policy",
        "feature_columns", "context_columns", "market_set_contract",
    ) if key in data}


def _feature_input_requirements(architecture: str) -> dict[str, Any]:
    from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.models.architectures import get_architecture_descriptor
    spec = get_model_spec(architecture)
    descriptor = get_architecture_descriptor(spec.architecture)
    return {
        "window_bars": int(spec.input_window_bars or DEFAULT_LABEL_POLICY.feature_window_bars),
        "dataset_context": bool(spec.use_dataset_context),
        "requires_market_set": bool(spec.requires_market_set),
        "derived_context_features": list(spec.derived_context_features),
        "eligibility_capabilities": {name: descriptor.has_capability(name) for name in (
            "raw_unadjusted_price_input", "lagged_trade_count_input", "risk_context",
            "predicted_upside_context", "predicted_safety_context")},
    }


def source_calculation_contract(*, recipe: str, profile_id: str,
                                horizon: int | None = None,
                                model_architecture: str | None = None) -> dict[str, Any]:
    from core.breakout_quality_registry import get_breakout_quality_experiment_profile
    from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
    profile = get_breakout_quality_experiment_profile(profile_id)
    from dataclasses import asdict
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    architecture = profile.model_architecture or model_architecture
    if not architecture:
        raise ValueError("Oracle feature provider requires the canonical source architecture")
    revisions = DAILY_CALCULATION_REVISIONS if recipe in DAILY_CALCULATION_REVISIONS else PRICE_CALCULATION_REVISIONS
    if recipe not in revisions:
        raise ValueError(f"Oracle unregistered source calculation capability: {recipe}")
    return {
        "recipe": recipe, "calculation_revision": revisions[recipe],
        "implementation_sha256": source_implementation_sha256(recipe, profile_id),
        "horizon_sessions": horizon if recipe in DAILY_CALCULATION_REVISIONS else
                            int(DEFAULT_LABEL_POLICY.label_horizon_bars),
        "feature_universe_revision": FEATURE_UNIVERSE_CALCULATION_REVISION,
        "feature_provider": {
            "profile_id": profile_id, "sample_scope": profile.training_sample_scope,
            "model_architecture": architecture,
            "source_target": profile.continuous_target_id,
            "target_policy": asdict(get_continuous_ranker_execution_recipe(profile_id).target_policy),
            # Only requirements consumed by the daily eligibility loader.
            # Head widths, channels, dropout and training objectives do not
            # determine the Oracle's feature-eligible stock-day population.
            "input_requirements": _feature_input_requirements(architecture),
        },
        "label_policy": DEFAULT_LABEL_POLICY.as_manifest_payload(),
        "score_mapping": "canonical_daily_truth_percentile_v2",
        "percentile_method": "average_zero_based",
    }



def attest_daily_candidate_execution(policy: Mapping[str, Any]) -> dict[str, Any]:
    """One canonical P19/P20 numerical witness for planning AND signed replay.

    Legacy policies are byte-for-byte unchanged: new Daily-only capabilities
    cannot silently invalidate old Breakout or P18 Oracle receipts.  A caller
    cannot supply a foreign witness to obtain a fake REUSE authorization.
    """
    result = dict(policy)
    from core.portfolio_entry_policy import (
        DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS,
        resolve_daily_execution_contract,
    )
    daily_mode = resolve_daily_execution_contract(result.get("daily_execution_contract"))
    if daily_mode in {DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS}:
        required_sha = portfolio_candidate_execution_sha256(
            daily_execution_contract=daily_mode,
        )
        if ("daily_candidate_implementation_sha256" in result and
                result["daily_candidate_implementation_sha256"] != required_sha):
            raise ValueError("Daily candidate capability fingerprint mismatch")
        result["daily_candidate_implementation_sha256"] = required_sha
    elif "daily_candidate_implementation_sha256" in result:
        raise ValueError("Unscoped Daily candidate implementation fingerprint")
    return result


def execution_context(*, dataset: str, price_sha256: str, param_sha256: str,
                      seed: int, target: str, start: str, end: str, cutoff: str,
                      max_positions: int, evaluation_mode: str,
                      ranking_policy: str, ranking_options: Mapping[str, Any],
                      replay_policy: Mapping[str, Any], score_contract: Mapping[str, Any] | None,
                      source_contract: Mapping[str, Any] | None,
                      dataset_inputs: Mapping[str, Any] | None,
                      execution_gate: str | None) -> dict[str, Any]:
    if not is_sha256_hex(price_sha256) or not is_sha256_hex(param_sha256):
        raise ValueError("Oracle execution context requires verified price/P16 identity")
    if execution_gate is not None:
        raise ValueError("Retired raw-outcome Oracle gate; Positive must enter through scores")
    # The same witness must be present in the physical signed Replay policy
    # and the derived effective context (see _arm_replay_policy). A read-only
    # context witness alone is insufficient to authorize native receipt reuse.
    policy = attest_daily_candidate_execution(oracle_replay_policy(replay_policy))
    payload = {
        "schema": _SCHEMA, "purpose": "hindsight_oracle_completed_replay",
        "replay_calculation_revision": REPLAY_CALCULATION_REVISION,
        "replay_implementation_sha256": replay_implementation_sha256(),
        "dataset": str(dataset), "price_sha256": price_sha256,
        "param_sha256": param_sha256, "seed": int(seed), "target": target,
        "start_date": start, "end_date": end, "research_cutoff": cutoff,
        "max_positions": int(max_positions), "rotation": "off",
        "evaluation_mode": evaluation_mode, "ranking_policy": ranking_policy,
        "ranking_options": oracle_ranking_options(ranking_options), "replay_policy": policy,
        "score_contract": (oracle_score_contract(score_contract)
                           if score_contract is not None else None),
        "source_contract": dict(source_contract) if source_contract is not None else None,
        "dataset_inputs": dict(dataset_inputs) if dataset_inputs is not None and source_contract is not None else None,
        "execution_gate": execution_gate,
        "execution_gate_implementation": None,
    }
    # Effective provider contracts contain tuples; persisted JSON represents
    # them as arrays. Normalize once at the boundary, before equality checks.
    return json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def legacy_import_permitted(context: Mapping[str, Any]) -> bool:
    """Legacy producers certify only the calculation revision they executed.

    Revising a formula cannot re-import an older result under the new revision
    just because its old provenance still happens to match a source checksum.
    """
    source = context.get("source_contract")
    return (context.get("replay_calculation_revision") == 1 and
            context.get("replay_implementation_sha256") == _LEGACY_V1_REPLAY_IMPLEMENTATION and
            (source is None or (source.get("calculation_revision") == 1
                                and source.get("feature_universe_revision") == 1
                                and canonical_json_sha256(source.get("feature_provider", {}).get("input_requirements"))
                                    == _LEGACY_V1_FEATURE_INPUT_REQUIREMENTS
                                and source.get("implementation_sha256") ==
                                _LEGACY_V1_SOURCE_IMPLEMENTATIONS.get(source.get("recipe")))))


class OracleExecutionIndex:
    """Immutable effective-contract -> original signed completed Replay.

    Publication is possible only AFTER the existing canonical resolver has
    certified the full Replay identity, score/membership and Win Rate. The
    historical receipt is preserved and checked on every reference read.
    """
    def __init__(self, replay_store):
        self.store = replay_store
        self.root = replay_store.cache_root.parent / "execution_index"

    def _path(self, context: Mapping[str, Any]) -> Path:
        target = str(context["target"])
        if not target.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Oracle invalid execution-index target")
        return self.root / str(int(context["seed"])) / f"{target}_{canonical_json_sha256(context)}.json"

    def has_history(self, context: Mapping[str, Any]) -> bool:
        path = self._path(context)
        prefix = str(context["target"]) + "_"
        return any(is_sha256_hex(candidate.stem[len(prefix):])
                   for candidate in path.parent.glob(prefix + "*.json"))

    @staticmethod
    def keys(identity: Mapping[str, Any]) -> tuple[str, int, str, str] | None:
        policy = identity.get("replay_policy") or {}
        contract = policy.get("canonical_dl_score_contract") or {}
        cohort = identity.get("cohort") or {}
        digest = contract.get("canonical_feature_eligible_keys_sha256")
        rows = contract.get("oracle_scored_rows")
        first, last = cohort.get("first_common_day"), cohort.get("last_common_day")
        if (not is_sha256_hex(digest) or type(rows) is not int or rows <= 0
                or rows != cohort.get("cohort_rows")
                or not isinstance(first, str) or not isinstance(last, str)
                or not identity["start_date"] <= first <= last <= identity["end_date"]):
            return None
        return digest, rows, first, last

    def read(self, context: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        wanted = oracle_execution_contract(context)
        path = self._path(wanted)
        prefix = str(wanted["target"]) + "_"
        matches = []
        # Old indexes may contain predictor routing names. Validate their
        # original SHA and receipt before comparing the effective contracts.
        # This is a generic projection, never an arm-ID/source-SHA allowlist.
        for candidate in sorted(path.parent.glob(prefix + "*.json")):
            if not is_sha256_hex(candidate.stem[len(prefix):]):
                continue
            completed = self._read_reference(candidate, wanted)
            if completed is not None:
                matches.append(completed)
        if matches and any(item[1] != matches[0][1] for item in matches[1:]):
            raise ValueError("Oracle equivalent execution references have conflicting completed results")
        return matches[0] if matches else None

    def _read_reference(self, path, wanted):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            payload = {k: raw[k] for k in ("schema", "context", "origin_identity", "observation_sha256")}
            recorded = payload["context"]
            if (payload["schema"] != _SCHEMA or self._path(recorded) != path
                    or raw.get("reference_sha256") != canonical_json_sha256(payload)):
                raise ValueError("execution reference identity/SHA mismatch")
            if oracle_execution_contract(recorded) != wanted:
                return None
            identity = payload["origin_identity"]
            observation = self.store.read(identity)
            if observation is None:
                raise ValueError("execution reference origin receipt missing")
            complete = self.store.with_win_rate(identity, observation)
            if complete is None or canonical_json_sha256(complete) != payload["observation_sha256"]:
                raise ValueError("execution reference completed metrics differ")
            self._validate_binding(recorded, identity, complete)
            return dict(identity), complete
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Oracle completed execution reference corrupt: {path.name}: {exc}") from exc

    def _validate_binding(self, context, identity, observation):
        for key in ("target", "seed", "start_date", "end_date", "max_positions", "evaluation_mode",
                    "ranking_policy", "rotation", "dataset"):
            if identity.get(key) != context.get(key):
                raise ValueError(f"execution reference origin mismatch: {key}")
        if (identity.get("dataset_sha256") != context["price_sha256"]
                or identity.get("param_sha256") != context["param_sha256"]):
            raise ValueError("execution reference price/P16 differs")
        if oracle_ranking_options(identity["ranking_options"]) != oracle_ranking_options(context["ranking_options"]):
            raise ValueError("execution reference selector options differ")
        policy = oracle_replay_policy(identity.get("replay_policy") or {})
        differences = self._economic_policy_differences(context, identity)
        if differences:
            raise ValueError(f"execution reference economic policy differs: {differences[0]}")
        if context["score_contract"] is not None:
            expected_score = context["score_contract"]
            if expected_score.get("perfect_positive_output") == _POSITIVE_OUTPUT_ID:
                producer = (identity.get("replay_policy") or {}).get("formal_positive_score_output")
                if (not isinstance(producer, Mapping)
                        or producer.get("contract") != _POSITIVE_OUTPUT_ID
                        or not _approved_perfect_positive_admission(
                            producer.get("shared_admission_code_sha256"))):
                    raise ValueError("execution reference Positive admission producer not certified")
            if self.keys(identity) is None:
                raise ValueError("execution reference missing certified stock-day membership")
            original = oracle_score_contract(policy.get("canonical_dl_score_contract") or {})
            for key, value in oracle_score_contract(context["score_contract"]).items():
                if original.get(key) != value:
                    raise ValueError(f"execution reference score consumption differs: {key}")
        self.store.validate_completed_origin(identity, observation)

    @staticmethod
    def _economic_policy_differences(context, identity) -> tuple[str, ...]:
        """Verify signed economics independently of the claimed context SHA.

        Oracle parity describes the perfect-score intervention, not the
        unscored canonical No-DL control. Both Min and Full controls omit it
        from their signed Replay policy; their remaining economic inputs and
        effective execution context must still match in full.
        """
        signed = oracle_replay_policy(identity.get("replay_policy") or {})
        required = oracle_replay_policy(context["replay_policy"])
        unscored_control = (
            context.get("score_contract") is None
            and context.get("source_contract") is None
            and required.get("baseline_arm") == context.get("target")
            and signed.get("baseline_arm") == context.get("target")
        )
        return tuple(key for key, value in required.items()
                     if not (key == "oracle_parity_implementation" and unscored_control)
                     and signed.get(key) != value)

    @staticmethod
    def _native_context_matches(context, identity):
        # A historical Replay can carry a matching effective-context digest
        # while its actual signed economic implementation is different.
        # It is not a current REUSE candidate. Preserve the historical signed
        # receipt and let the authorized canonical Replay create a new one.
        if OracleExecutionIndex._economic_policy_differences(context, identity):
            return False
        digest = (identity.get("replay_policy") or {}).get("effective_execution_contract_sha256")
        if digest == canonical_json_sha256(context):
            return True
        # Recover a missing old index only if the signed origin's exact
        # effective-context SHA can be reconstructed. Unknown calculation,
        # source, eligibility or economic drift cannot pass this proof.
        recorded = dict(context, ranking_options=identity["ranking_options"])
        original = (identity.get("replay_policy") or {}).get("canonical_dl_score_contract")
        if original is not None:
            recorded["score_contract"] = {key: value for key, value in original.items()
                                          if key not in _REFERENCE_FIELDS | _SCORE_WITNESS_FIELDS}
        return (canonical_json_sha256(recorded) == digest
                and oracle_execution_contract(recorded) == oracle_execution_contract(context))

    def native_completed(self, context):
        """Recover an index from a native effective-contract signed receipt.

        Source-byte aliases and numerical artifacts are unnecessary: native
        receipts already name this exact execution-effective contract.
        """
        matches = []
        for identity in self.store.existing_completed_identities(seed=context["seed"], target=context["target"]):
            if not self._native_context_matches(context, identity):
                continue
            complete = self.store.with_win_rate(identity, self.store.read(identity))
            if complete is not None:
                self._validate_binding(context, identity, complete)
                matches.append((identity, complete))
        if matches and any(item[1] != matches[0][1] for item in matches[1:]):
            raise ValueError("Oracle same effective contract has conflicting native Replay receipts")
        return matches[0] if matches else None

    def publish(self, context: Mapping[str, Any], identity: Mapping[str, Any],
                observation: Mapping[str, Any]) -> None:
        # Never turn a caller's metric dictionary into evidence. Resolve the
        # actual signed origin, including any immutable metric supplement.
        origin = self.store.read(identity)
        if origin is None or self.store.with_win_rate(identity, origin) != dict(observation):
            raise ValueError("Oracle cannot index an unverified completed Replay")
        self._validate_binding(context, identity, observation)
        path = self._path(context)
        payload = {"schema": _SCHEMA, "context": dict(context),
                   "origin_identity": dict(identity),
                   "observation_sha256": canonical_json_sha256(observation)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(path.with_name("." + path.name + ".lock")):
            current = self.read(context)
            if current is not None:
                if current[1] != dict(observation):
                    raise ValueError("Oracle same effective contract has conflicting completed results")
                return
            temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
            try:
                temporary.write_text(json.dumps({**payload, "reference_sha256": canonical_json_sha256(payload)},
                    ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                atomic_replace_with_retry(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        self.read(context)


def _reference_arm_payload(arm_id, settings, *, effective):
    if arm_id not in settings.arms:
        raise ValueError(f"Oracle reference arm missing: {arm_id}")
    arm = settings.arms[arm_id]
    source_ids = [str(arm.dl_id)]
    from core.positive_score_admission import (requires_positive_score, uses_primary_positive_classifier)
    requires_positive_score(arm.dl_runtime_options or {})
    primary_positive_only = uses_primary_positive_classifier(arm.dl_runtime_options or {})
    sources = {}
    for index, source_id in enumerate(source_ids):
        if source_id not in settings.dl_sources:
            raise ValueError(f"Oracle reference DL source missing: {source_id}")
        source = settings.dl_sources[source_id]
        payload = {
            key: value for key, value in settings.dl_sources[source_id].as_dict().items()
            if key != "description"
        }
        if effective:
            from core.breakout_quality_registry import get_breakout_quality_experiment_profile
            profile = get_breakout_quality_experiment_profile(source.experiment_profile)
            payload = {key: value for key, value in payload.items() if key not in {
                "dl_id", "experiment_profile", "model_architecture", "forward_scores_builder",
                "single_seed_strategy_conversion_authorized", "point_in_time_dirname"}}
            payload.update(sample_scope=profile.training_sample_scope,
                           input_requirements=_feature_input_requirements(source.model_architecture))
        role = "positive" if primary_positive_only else ("rank" if index == 0 else "positive")
        sources[role if effective else source_id] = payload
    arm_contract = {
        key: value for key, value in arm.as_dict().items()
        if key not in {"name", "description"}
    }
    if effective:
        arm_contract = {key: value for key, value in arm_contract.items() if key not in {
            "arm_id", "dl_id", "enabled", "robustness_role"}}
        # The published reference binding describes the *actual* Research
        # consumer, not the hypothetical perfect-score substitution. Retain
        # model-output topology and Replay version in this independent proof.
        arm_contract["dl_runtime_options"] = dict(arm_contract["dl_runtime_options"])
    return json.loads(json.dumps({"arm": arm_contract, "sources": sources}, sort_keys=True))


def reference_arm_execution_binding(
    comparison: Mapping[str, Any], settings: Any,
) -> dict[str, Any] | None:
    """Record the inspected arm separately from its effective perfect-score consumer.

    Fitted predictors are substituted by the declared Oracle outputs. Their
    identities cannot invalidate those outputs; eligibility, timing, selector,
    Positive admission and trading semantics must still match exactly.
    """
    arm_id = str(comparison.get("strategy_arm_id") or "").strip()
    if not arm_id:
        return None
    contract = _reference_arm_payload(arm_id, settings, effective=True)
    return {"arm_id": arm_id, "execution_contract": contract,
            "execution_sha256": canonical_json_sha256(contract)}


def reference_arm_execution_matches(published, comparison, settings) -> bool:
    """Accept old evidence only after proving the same execution contract."""
    if not isinstance(published, Mapping):
        return False
    required = reference_arm_execution_binding(comparison, settings)
    if required is None:
        return False
    contract = published.get("execution_contract")
    if isinstance(contract, Mapping):
        return (published.get("execution_sha256") == canonical_json_sha256(contract)
                and contract == required["execution_contract"])
    arm_id = str(published.get("arm_id") or "")
    if arm_id not in settings.arms:
        return False
    legacy = _reference_arm_payload(arm_id, settings, effective=False)
    effective = _reference_arm_payload(arm_id, settings, effective=True)
    return (published.get("execution_sha256") in {
                canonical_json_sha256(legacy), canonical_json_sha256(effective)}
            and effective == required["execution_contract"])

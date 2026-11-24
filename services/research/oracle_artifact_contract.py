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
from dataclasses import dataclass

from core.execution_source_dependencies import execution_source_inventory, project_ranking_policy_source
from core.research_execution_equivalence import (
    reviewed_ranking_source_sha256, ORACLE_STRATEGY_OBSERVATION_SCOPE,
    oracle_observation_consumer_sources, project_oracle_observation_sources,
    project_daily_filter_calculation,
)
from core.file_integrity import (
    canonical_json_sha256, is_sha256_hex, exclusive_file_lock, atomic_replace_with_retry,
)
from core.research_execution_equivalence import (
    reviewed_calculation_sha256, project_reviewed_source_inventory,
    reviewed_replay_predecessor_sources,
)

from core.daily_trade_path_capabilities import DAILY_TRUTH_CAPABILITIES, daily_truth_capability
from services.research.oracle_result_contract import (
    PRICE_CALCULATION_REVISIONS, DAILY_CALCULATION_REVISIONS,
    FEATURE_UNIVERSE_CALCULATION_REVISION, REPLAY_CALCULATION_REVISION,
    OracleResultContract, oracle_source_result_contract, oracle_result_policy,
    oracle_score_contract, oracle_ranking_options, _REFERENCE_FIELDS, _SCORE_REFERENCE_FIELDS,
    _ORACLE_PERFECT_POSITIVE_INTERFACE, _LEGACY_POSITIVE_INTERFACE,
    _SINGLE_RANKER_POSITIVE_INTERFACE, _POSITIVE_OUTPUT_COLUMN, _POSITIVE_OUTPUT_ID,
)
from services.research.oracle_legacy_admission import verified_legacy_origin_pin

# Engineering calculation revisions, not scientific Target/model identities.
# Change only the affected capability after a result-affecting fix. Existing
# recipes are stable when a new recipe is registered alongside them.
_SCHEMA = 1
_RESULT_INDEX_SCHEMA = 2
_SCORE_WITNESS_FIELDS = {"canonical_feature_eligible_keys_sha256", "oracle_scored_rows", "source_evidence"}

# The perfect-Positive intervention supplies the *same* signed, continuous
# Target-percentile rank and exact Positive 0/1 column before and after a
# predictor-topology change. These are approved Oracle output interfaces,
# NOT an equivalence of the historical fitted models or their PIT scores.
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
    ranking_policy = context.get("ranking_policy")
    result["ranking_options"] = oracle_ranking_options(context["ranking_options"])
    result["replay_policy"] = oracle_observation_policy(
        context["replay_policy"], ranking_policy=ranking_policy,
    )
    # A completed Oracle may have been published under the full Portfolio
    # witness. Accept it only while both exact producer and consumer closures
    # prove that its persisted observation has the same narrower semantics.
    if (context.get("purpose") == "hindsight_oracle_completed_replay"
            and context.get("replay_implementation_sha256") ==
                replay_implementation_sha256(ranking_policy=ranking_policy)
            and oracle_observation_equivalence_proven(ranking_policy=ranking_policy)):
        result["replay_implementation_sha256"] = replay_implementation_sha256(
            ranking_policy=ranking_policy, observation_scope=ORACLE_STRATEGY_OBSERVATION_SCOPE,
        )
    if context.get("score_contract") is not None:
        result["score_contract"] = oracle_score_contract(context["score_contract"])
    if context.get("source_contract") is not None:
        result["source_contract"] = oracle_source_contract(context["source_contract"])
    return result


def oracle_result_contract(context: Mapping[str, Any]) -> OracleResultContract:
    """Normalize established Oracle interfaces before projecting provenance."""
    return OracleResultContract.from_execution(oracle_execution_contract(context))


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


def calculation_source(relative: str, name: str, *, bound_literals=None, project_refactor=True) -> str:
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
    digest = reviewed_calculation_sha256(
        relative, sha256((code_root / relative).read_bytes()).hexdigest(),
        canonical_json_sha256(_syntax_payload(projected)),
    )
    if project_refactor and relative == "core/daily_common_entry_filters.py" and name == "daily_common_filter_signals":
        evidence = {"calculation": digest, **{
            owner: _source_owner_attestation(code_root / owner)[1]
            for owner in (relative, "core/portfolio_entry_policy.py")
        }}
        evidence = project_daily_filter_calculation(evidence, project_root=code_root)
        return (evidence["calculation"] if set(evidence) == {"calculation"}
                else canonical_json_sha256(evidence))
    return digest


@lru_cache(maxsize=512)
def _source_content_attestation(content: str | bytes, *, is_python: bool,
                                is_ranking: bool = False, ranking_policy=None):
    """Cache pure parsing by complete source content and selected semantics.

    Callers still read every actual file. No path/stat/PASS cache can hide a
    same-size edit, restored timestamp, test mutation or changed descriptor.
    """
    if is_python:
        parsed = ast.parse(content)
        if is_ranking:
            parsed = project_ranking_policy_source(parsed, ranking_policy)
        payload = _syntax_payload(_CalculationSyntax().visit(parsed))
    else:
        payload = {"native_source_sha256": sha256(content).hexdigest()}
    return payload, canonical_json_sha256(payload)


def _source_owner_attestation(path: Path, *, ranking_policy=None):
    is_python = path.suffix == ".py"
    is_ranking = path.name == "ranking_policy.py"
    content = path.read_text(encoding="utf-8-sig") if is_python else path.read_bytes()
    return _source_content_attestation(
        content, is_python=is_python, is_ranking=is_ranking,
        ranking_policy=ranking_policy if is_ranking else None,
    )


def _source_owner_payload(path: Path, *, ranking_policy=None):
    return _source_owner_attestation(path, ranking_policy=ranking_policy)[0]


@lru_cache(maxsize=128)
def replay_implementation_sha256(*, ranking_policy: str | None = None,
                                 observation_scope: str | None = None) -> str:
    """Attest the selected Portfolio capability and its delegated numerical owners."""
    code_root = Path(__file__).resolve().parents[2]
    if observation_scope not in (None, ORACLE_STRATEGY_OBSERVATION_SCOPE):
        raise ValueError("Unknown Portfolio observation scope")
    owners = execution_source_inventory("replay")
    records = {relative: _source_owner_attestation(code_root / relative, ranking_policy=ranking_policy)
               for relative in owners}
    parts = {relative: record[0] for relative, record in records.items()}
    # A removed dependency must not recreate the old shorter inventory by
    # accident. The expanded graph is itself an attested input.
    parts["@execution_sources"] = list(owners)
    # The old witness hashes syntax payloads directly. Certified newly covered
    # dependencies can be omitted only as one unchanged group; original payloads
    # are always read live, so this is not an alias for a historical result hash.
    hashes = {relative: record[1] for relative, record in records.items()}
    hashes["@execution_sources"] = canonical_json_sha256(parts["@execution_sources"])
    ranking_owner = "core/ranking_policy.py"
    hashes[ranking_owner] = reviewed_ranking_source_sha256(hashes[ranking_owner], project_root=code_root)
    predecessor_sources = reviewed_replay_predecessor_sources(hashes, project_root=code_root)
    if observation_scope == ORACLE_STRATEGY_OBSERVATION_SCOPE:
        for relative in oracle_observation_consumer_sources():
            payload, digest = _source_owner_attestation(code_root / relative)
            parts[relative], hashes[relative] = payload, digest
        hashes = project_oracle_observation_sources(
            hashes, scope="replay_numerics", project_root=code_root,
        )
    projected = project_reviewed_source_inventory(
        hashes, scope="replay_numerics", project_root=code_root,
    )
    payloads = {relative: parts[relative] for relative in projected}
    predecessors = {}
    for relative, (source, expected_digest) in predecessor_sources.items():
        payload, digest = _source_content_attestation(source, is_python=True)
        if digest != expected_digest:
            predecessors = {}
            break
        predecessors[relative] = payload
    return canonical_json_sha256({**payloads, **predecessors})


@lru_cache(maxsize=128)
def portfolio_candidate_execution_sha256(*, daily_execution_contract: str | None = None, ranking_policy: str | None = None,
                                         observation_scope: str | None = None) -> str:
    """Attest executable portfolio capabilities for a registered execution scope.

    This extends the shared replay witness without revising the frozen
    ``replay_implementation_sha256`` consumed by existing P16 Oracle receipts.
    The dependency inventory covers the candidate/capacity/entry/exit/equity
    owners reached through ``run_portfolio_timeline`` and its replay adapter.
    AST projection ignores comments and docstrings but captures executable
    changes, including daily candidate capacity and selector decisions.
    """
    code_root = Path(__file__).resolve().parents[2]
    if observation_scope not in (None, ORACLE_STRATEGY_OBSERVATION_SCOPE):
        raise ValueError("Unknown Portfolio observation scope")
    owners = execution_source_inventory("portfolio_candidates")
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
    optional = (execution_source_inventory("daily_common_filters")
                if mode == DAILY_EXECUTION_FULL_FILTERS else ())
    parts = {"daily_execution_contract": mode}
    inventory = tuple(dict.fromkeys((*owners, *optional)))
    # Optional mode owners remain ordinary entries outside the common migration
    # group. Their presence/absence changes the hash independently.
    parts["@execution_sources"] = canonical_json_sha256(list(owners))
    for relative in inventory:
        path = code_root / relative
        if not path.is_file():
            raise ValueError(f"Oracle replay numerical owner missing: {relative}")
        _, digest = _source_owner_attestation(path, ranking_policy=ranking_policy)
        if relative == "core/ranking_policy.py":
            parts[relative] = reviewed_ranking_source_sha256(digest, project_root=code_root)
        else:
            parts[relative] = digest
    if observation_scope == ORACLE_STRATEGY_OBSERVATION_SCOPE:
        parts.update({relative: _source_owner_attestation(code_root / relative)[1]
                      for relative in oracle_observation_consumer_sources()})
        parts = project_oracle_observation_sources(
            parts, scope="portfolio_candidates", project_root=code_root,
        )
    if mode == DAILY_EXECUTION_FULL_FILTERS:
        parts = project_reviewed_source_inventory(parts, scope="portfolio_full_filters", project_root=code_root)
    return canonical_json_sha256(project_reviewed_source_inventory(
        parts, scope="portfolio_candidates", project_root=code_root,
    ))


def oracle_observation_equivalence_proven(*, ranking_policy: str | None = None,
                                         daily_execution_contract: str | None = None) -> bool:
    """Re-read the complete reviewed producer/consumer closure before migration."""
    root = Path(__file__).resolve().parents[2]
    from core.portfolio_entry_policy import DAILY_EXECUTION_FULL_FILTERS
    scope = "portfolio_candidates" if daily_execution_contract is not None else "replay_numerics"
    owners = execution_source_inventory("portfolio_candidates" if daily_execution_contract is not None else "replay")
    hashes = {relative: _source_owner_attestation(root / relative, ranking_policy=ranking_policy)[1]
              for relative in owners}
    hashes["@execution_sources"] = canonical_json_sha256(list(owners))
    ranking_owner = "core/ranking_policy.py"
    hashes[ranking_owner] = reviewed_ranking_source_sha256(hashes[ranking_owner], project_root=root)
    if daily_execution_contract == DAILY_EXECUTION_FULL_FILTERS:
        relative = "core/daily_common_entry_filters.py"
        hashes[relative] = _source_owner_attestation(root / relative)[1]
    consumers = oracle_observation_consumer_sources()
    hashes.update({relative: _source_owner_attestation(root / relative)[1] for relative in consumers})
    projected = project_oracle_observation_sources(hashes, scope=scope, project_root=root)
    return bool(consumers) and all(relative not in projected for relative in consumers)


def oracle_observation_policy(policy: Mapping[str, Any], *, ranking_policy: str | None = None) -> dict[str, Any]:
    """Recognize an already-signed full-result witness only through the exact proof."""
    from core.portfolio_entry_policy import DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS
    result = oracle_replay_policy(policy)
    mode = result.get("daily_execution_contract")
    recorded = result.get("daily_candidate_implementation_sha256")
    if recorded is not None and mode in (DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS):
        complete = portfolio_candidate_execution_sha256(daily_execution_contract=mode, ranking_policy=ranking_policy)
        if recorded == complete and oracle_observation_equivalence_proven(
                ranking_policy=ranking_policy, daily_execution_contract=mode):
            result["daily_candidate_implementation_sha256"] = portfolio_candidate_execution_sha256(
                daily_execution_contract=mode, ranking_policy=ranking_policy,
                observation_scope=ORACLE_STRATEGY_OBSERVATION_SCOPE,
            )
    return result


def _source_implementation_parts(recipe: str, profile_id: str) -> dict[str, str]:
    parts = {"daily_percentile": calculation_source(
        "filters/breakout_quality/continuous_ranker_data.py", "build_same_date_percentile_targets")}
    capability = daily_truth_capability(recipe)
    if capability is not None:
        parts.update({key: calculation_source(relative, name)
                      for key, relative, name in capability["calculation_owners"]})
        closure = capability.get("calculation_closure")
        if closure is not None:
            module, function, *arguments = closure
            parts["delegated_execution"] = getattr(importlib.import_module(module), function)(*arguments)
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
    return parts


@lru_cache(maxsize=32)
def source_implementation_sha256(recipe: str, profile_id: str) -> str:
    return canonical_json_sha256(_source_implementation_parts(recipe, profile_id))


@lru_cache(maxsize=32)
def _source_implementation_encodings(recipe: str, profile_id: str) -> frozenset[str]:
    """Rebuild declared encoding variants and exact reviewed filter-extraction parts.

    Every calculation owner is evaluated from the executing implementation;
    a changed formula, admission or delegated source cannot inherit an old
    digest. No archived result hash or model/strategy ID is an authority.
    """
    capability = daily_truth_capability(recipe)
    parts = _source_implementation_parts(recipe, profile_id)
    variants = [parts]
    if capability:
        for key, relative, name in capability["calculation_owners"]:
            if relative != "core/daily_common_entry_filters.py" or name != "daily_common_filter_signals":
                continue
            raw = calculation_source(relative, name, project_refactor=False)
            root = Path(__file__).resolve().parents[2]
            evidence = {"calculation": raw, **{
                owner: _source_owner_attestation(root / owner)[1]
                for owner in (relative, "core/portfolio_entry_policy.py")
            }}
            projected = project_daily_filter_calculation(evidence, project_root=root)
            if set(projected) == {"calculation"} and projected["calculation"] == parts[key]:
                variants.extend({**item, key: raw} for item in tuple(variants))
    closure = capability.get("calculation_closure") if capability else None
    if closure and tuple(closure[:2]) == (
            "services.audit.daily_trade40_oracle", "truth_producer_fingerprint"):
        from services.audit.mintrade_truth_source_attestation import _mintrade_source_fingerprint_proof
        executing, encodings = _mintrade_source_fingerprint_proof()
        if parts.get("delegated_execution") != executing:
            return frozenset()
        variants = [{**item, "delegated_execution": value}
                    for item in variants for value in encodings]
    return frozenset(canonical_json_sha256(item) for item in variants)


def oracle_source_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    """Project proven encoding/extraction variants; retain every other source input."""
    result = dict(source)
    recipe = str(source.get("recipe") or "")
    provider = source.get("feature_provider") or {}
    profile_id = str(provider.get("profile_id") or "")
    if source.get("implementation_sha256") in _source_implementation_encodings(recipe, profile_id):
        result["implementation_sha256"] = source_implementation_sha256(recipe, profile_id)
    return result


def legacy_source_contract_matches(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Bounded pre-migration proof; a matching semantic key alone is insufficient.

    An attested migration may instead pin a previously validated original
    reference. This fallback admits only source encodings independently proven
    from the executing owners, never an unknown historical source SHA.
    """
    if oracle_source_result_contract(recorded) != oracle_source_result_contract(current):
        return False
    recipe = str(recorded.get("recipe") or "")
    profile = str((recorded.get("feature_provider") or {}).get("profile_id") or "")
    encodings = _source_implementation_encodings(recipe, profile)
    return (recorded.get("implementation_sha256") in encodings
            and current.get("implementation_sha256", source_implementation_sha256(recipe, profile)) in encodings)


def legacy_execution_context_matches(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Keep the former exact proof only at the read-only migration boundary."""
    return oracle_execution_contract(recorded) == oracle_execution_contract(current)


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
            "target_policy": get_continuous_ranker_execution_recipe(profile_id).target_policy.as_target_calculation_payload(),
            # Only requirements consumed by the daily eligibility loader.
            # Head widths, channels, dropout and training objectives do not
            # determine the Oracle's feature-eligible stock-day population.
            "input_requirements": _feature_input_requirements(architecture),
        },
        "label_policy": DEFAULT_LABEL_POLICY.as_manifest_payload(),
        "score_mapping": "canonical_daily_truth_percentile_v2",
        "percentile_method": "average_zero_based",
    }


def attest_daily_candidate_execution(policy: Mapping[str, Any], *, ranking_policy: str | None = None) -> dict[str, Any]:
    """One canonical P19/P20 numerical witness for planning AND signed replay.

    Legacy policies are byte-for-byte unchanged: new Daily-only capabilities
    cannot silently invalidate old Breakout or P18 Oracle receipts.  A caller
    cannot supply a foreign witness to obtain a fake REUSE authorization.
    """
    result = dict(policy)
    if "daily_candidate_implementation_sha256" in result:
        result["daily_candidate_implementation_sha256"] = oracle_observation_policy(
            result, ranking_policy=ranking_policy,
        )["daily_candidate_implementation_sha256"]
    from core.portfolio_entry_policy import (
        DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS,
        resolve_daily_execution_contract,
    )
    daily_mode = resolve_daily_execution_contract(result.get("daily_execution_contract"))
    if daily_mode in {DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS}:
        required_sha = portfolio_candidate_execution_sha256(
            daily_execution_contract=daily_mode, ranking_policy=ranking_policy,
            observation_scope=ORACLE_STRATEGY_OBSERVATION_SCOPE,
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
    policy = attest_daily_candidate_execution(oracle_replay_policy(replay_policy), ranking_policy=ranking_policy)
    payload = {
        "schema": _SCHEMA, "purpose": "hindsight_oracle_completed_replay",
        "replay_calculation_revision": REPLAY_CALCULATION_REVISION,
        "replay_implementation_sha256": replay_implementation_sha256(
            ranking_policy=ranking_policy, observation_scope=ORACLE_STRATEGY_OBSERVATION_SCOPE),
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


@dataclass(frozen=True)
class OracleReuseInspection:
    decision: str
    reason: str
    identity: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    origin_pins: tuple[tuple[str, str], ...] = ()
    adoption_required: bool = False
    differences: tuple[str, ...] = ()


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
        return self.root / str(int(context["seed"])) / f"{target}_{oracle_result_contract(context).sha256}.json"

    def _legacy_path(self, context: Mapping[str, Any]) -> Path:
        return self.root / str(int(context["seed"])) / f"{context['target']}_{canonical_json_sha256(context)}.json"

    @staticmethod
    def _build_admission():
        from services.research.oracle_build_attestation import require_oracle_build_attestation
        return require_oracle_build_attestation(Path(__file__).resolve().parents[2])

    @staticmethod
    def _current_provenance(context) -> bool:
        """Do not let a caller relabel unknown code as today's semantic key."""
        ranking = context.get("ranking_policy")
        if context.get("replay_implementation_sha256") != replay_implementation_sha256(
                ranking_policy=ranking, observation_scope=ORACLE_STRATEGY_OBSERVATION_SCOPE):
            return False
        source = context.get("source_contract")
        if source is not None:
            recipe = str(source.get("recipe") or "")
            profile = str((source.get("feature_provider") or {}).get("profile_id") or "")
            if source.get("implementation_sha256") != source_implementation_sha256(recipe, profile):
                return False
        policy = context.get("replay_policy") or {}
        if "daily_candidate_implementation_sha256" in policy:
            expected = portfolio_candidate_execution_sha256(
                daily_execution_contract=policy.get("daily_execution_contract"), ranking_policy=ranking,
                observation_scope=ORACLE_STRATEGY_OBSERVATION_SCOPE)
            if policy["daily_candidate_implementation_sha256"] != expected:
                return False
        return True

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.store.project_root.resolve()).as_posix()

    def _migration_admitted(self, path, context, identity, build) -> bool:
        """A baseline-validated transition binds bytes AND the semantic result."""
        signed_context = (identity.get("replay_policy") or {}).get("effective_execution_contract_sha256")
        return verified_legacy_origin_pin(
            self.store.project_root, path, build,
            result_contract_sha256=oracle_result_contract(context).sha256,
            identity_sha256=canonical_json_sha256(identity),
            context_sha256=(canonical_json_sha256(context) if path.parent.parent == self.root
                            else signed_context),
        ) is not None

    def _origin_pins(self, identity) -> dict[str, str]:
        pins = {}
        pending = [identity]
        while pending:
            current = pending.pop()
            path = self.store._path(current)
            relative = self._relative(path)
            if relative in pins:
                raise ValueError("Oracle completed origin has a provenance cycle")
            raw_bytes = path.read_bytes()
            pins[relative] = sha256(raw_bytes).hexdigest()
            raw = json.loads(raw_bytes)
            supplement = self.store._win_rate_path(current)
            if supplement.is_file():
                pins[self._relative(supplement)] = sha256(supplement.read_bytes()).hexdigest()
            prior = (raw.get("reuse_certificate") or {}).get("origin_identity")
            if prior is not None:
                pending.append(prior)
        return pins

    def _verify_adoption(self, admission, context, identity, observation) -> None:
        if (not isinstance(admission, Mapping)
                or admission.get("kind") not in {
                    "current_signed_producer", "verified_build_transition", "reviewed_legacy_source"}
                or not is_sha256_hex(admission.get("build_attestation_sha256"))
                or admission.get("context_sha256") != canonical_json_sha256(context)
                or admission.get("result_contract_sha256") != oracle_result_contract(context).sha256
                or admission.get("origin_identity_sha256") != canonical_json_sha256(identity)
                or admission.get("observation_sha256") != canonical_json_sha256(observation)
                or admission.get("origin_pins") != self._origin_pins(identity)):
            raise ValueError("Oracle immutable result adoption provenance differs")

    def has_history(self, context: Mapping[str, Any]) -> bool:
        path = self._path(context)
        prefix = str(context["target"]) + "_"
        return any(is_sha256_hex(candidate.stem[len(prefix):])
                   for candidate in path.parent.glob(prefix + "*.json"))

    def inspect(self, context: Mapping[str, Any]) -> OracleReuseInspection:
        """One read-only decision for planning and authorized later adoption."""
        from services.research.oracle_build_attestation import OracleBuildBlockedError
        try:
            self._build_admission()
            if not self._current_provenance(context):
                return OracleReuseInspection("BLOCKED", "Unknown current implementation context",
                                             differences=("implementation_provenance",))
            completed = self.read(context)
            if completed is None:
                completed = self.native_completed(context)
            if completed is None:
                return OracleReuseInspection("RUN", "No compatible completed Oracle result",
                    differences=(("result_contract",) if self.has_history(context) else ()))
            identity, observation = completed
            pins = self._origin_pins(identity)
            checked = self.store.read(identity)
            if (checked is None or self.store.with_win_rate(identity, checked) != observation
                    or self._origin_pins(identity) != pins):
                raise ValueError("Oracle completed origin changed during inspection")
            stable_path = self._path(context)
            adopted = (stable_path.is_file()
                       and json.loads(stable_path.read_text(encoding="utf-8")).get("schema") == _RESULT_INDEX_SCHEMA)
            return OracleReuseInspection(
                "REUSE", "Verified completed Oracle result", dict(identity), dict(observation),
                tuple(sorted(pins.items())), not adopted,
            )
        except (ValueError, OSError, OracleBuildBlockedError) as exc:
            return OracleReuseInspection("BLOCKED", str(exc), differences=("artifact_or_build_provenance",))

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
        build = self._build_admission()
        if not self._current_provenance(context):
            return None
        wanted = oracle_result_contract(context)
        path = self._path(context)
        prefix = str(context["target"]) + "_"
        matches = []
        # Old indexes may contain predictor routing names. Validate their
        # original SHA and receipt before comparing the effective contracts.
        # This is a generic projection, never an arm-ID/source-SHA allowlist.
        for candidate in sorted(path.parent.glob(prefix + "*.json")):
            if not is_sha256_hex(candidate.stem[len(prefix):]):
                continue
            completed = self._read_reference(candidate, context, wanted, build)
            if completed is not None:
                matches.append(completed)
        if matches and any(item[1] != matches[0][1] for item in matches[1:]):
            raise ValueError("Oracle equivalent execution references have conflicting completed results")
        return matches[0] if matches else None

    def _read_reference(self, path, context, wanted, build):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            schema = raw.get("schema")
            fields = ("schema", "context", "origin_identity", "observation_sha256")
            if schema == _RESULT_INDEX_SCHEMA:
                fields += ("result_contract", "admission")
            payload = {k: raw[k] for k in fields}
            recorded = payload["context"]
            expected_path = self._legacy_path(recorded) if schema == _SCHEMA else self._path(recorded)
            if (schema not in {_SCHEMA, _RESULT_INDEX_SCHEMA} or expected_path != path
                    or raw.get("reference_sha256") != canonical_json_sha256(payload)):
                raise ValueError("execution reference identity/SHA mismatch")
            if oracle_result_contract(recorded) != wanted:
                return None
            identity = payload["origin_identity"]
            observation = self.store.read(identity)
            if observation is None:
                raise ValueError("execution reference origin receipt missing")
            complete = self.store.with_win_rate(identity, observation)
            if complete is None or canonical_json_sha256(complete) != payload["observation_sha256"]:
                raise ValueError("execution reference completed metrics differ")
            if schema == _RESULT_INDEX_SCHEMA:
                if payload["result_contract"] != wanted.as_payload():
                    raise ValueError("execution reference result contract mismatch")
                self._verify_adoption(payload["admission"], recorded, identity, complete)
            elif not (self._migration_admitted(path, recorded, identity, build)
                      or self._migration_admitted(self.store._path(identity), context, identity, build)
                      or legacy_execution_context_matches(recorded, context)):
                return None
            self._validate_binding(recorded, identity, complete, semantic=True)
            return dict(identity), complete
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Oracle completed execution reference corrupt: {path.name}: {exc}") from exc

    def _validate_binding(self, context, identity, observation, *, semantic=False):
        if not self._source_lineage_present(context, identity):
            raise ValueError("execution reference missing signed Target source lineage")
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
        differences = self._economic_policy_differences(context, identity, semantic=semantic)
        if differences:
            raise ValueError(f"execution reference economic policy differs: {differences[0]}")
        if context["score_contract"] is not None:
            expected_score = context["score_contract"]
            if expected_score.get("perfect_positive_output") == _POSITIVE_OUTPUT_ID:
                producer = (identity.get("replay_policy") or {}).get("formal_positive_score_output")
                if (not isinstance(producer, Mapping)
                        or producer.get("contract") != _POSITIVE_OUTPUT_ID
                        or (not semantic and not _approved_perfect_positive_admission(
                            producer.get("shared_admission_code_sha256")))):
                    raise ValueError("execution reference Positive admission producer not certified")
            if self.keys(identity) is None:
                raise ValueError("execution reference missing certified stock-day membership")
            original = oracle_score_contract(policy.get("canonical_dl_score_contract") or {})
            for key, value in oracle_score_contract(context["score_contract"]).items():
                if original.get(key) != value:
                    raise ValueError(f"execution reference score consumption differs: {key}")
        self.store.validate_completed_origin(identity, observation)

    @staticmethod
    def _economic_policy_differences(context, identity, *, semantic=False) -> tuple[str, ...]:
        """Verify signed economics independently of the claimed context SHA.

        Oracle parity describes the perfect-score intervention, not the
        unscored canonical No-DL control. Both Min and Full controls omit it
        from their signed Replay policy; their remaining economic inputs and
        effective execution context must still match in full.
        """
        signed = oracle_observation_policy(
            identity.get("replay_policy") or {}, ranking_policy=context.get("ranking_policy"),
        )
        required = oracle_observation_policy(
            context["replay_policy"], ranking_policy=context.get("ranking_policy"),
        )
        if semantic:
            signed, required = oracle_result_policy(signed), oracle_result_policy(required)
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
        if not OracleExecutionIndex._source_lineage_present(context, identity):
            return False
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
        candidates = [recorded]
        if oracle_observation_equivalence_proven(ranking_policy=context.get("ranking_policy")):
            policies = [recorded["replay_policy"]]
            signed_candidate = (identity.get("replay_policy") or {}).get("daily_candidate_implementation_sha256")
            if signed_candidate is not None:
                policies.append({**recorded["replay_policy"],
                                 "daily_candidate_implementation_sha256": signed_candidate})
            implementations = {
                recorded["replay_implementation_sha256"],
                replay_implementation_sha256(ranking_policy=context.get("ranking_policy")),
            }
            candidates = [{**recorded, "replay_policy": policy,
                           "replay_implementation_sha256": implementation}
                          for policy in policies for implementation in sorted(implementations)]
        source = recorded.get("source_contract")
        if source is not None:
            encodings = _source_implementation_encodings(
                str(source.get("recipe") or ""),
                str((source.get("feature_provider") or {}).get("profile_id") or ""))
            if source.get("implementation_sha256") in encodings:
                candidates = [{**candidate, "source_contract": {
                    **source, "implementation_sha256": value}}
                    for candidate in candidates for value in sorted(encodings)]
        return any(canonical_json_sha256(candidate) == digest
                   and oracle_execution_contract(candidate) == oracle_execution_contract(context)
                   for candidate in candidates)

    @staticmethod
    def _source_lineage_present(context, identity):
        if context.get("source_contract") is None:
            return True
        policy = identity.get("replay_policy") or {}
        return (is_sha256_hex(policy.get("perfect_score_sha256"))
                and isinstance(policy.get("hindsight_target_contract"), Mapping)
                and bool(policy["hindsight_target_contract"]))

    def matches_signed_context(self, context, identity, *, allow_input_drift=False) -> bool:
        """Read-only provenance admission, independent of WinRate completeness.

        The caller can inspect a historical data/P16 contract to classify an
        input change. This does not authorize reuse under the requested data,
        and all other inputs and the original signed digest must still match.
        """
        build = self._build_admission()
        if not self._current_provenance(context) or not self._source_lineage_present(context, identity):
            return False
        recorded = dict(context)
        if allow_input_drift:
            recorded.update(dataset=identity.get("dataset"), price_sha256=identity.get("dataset_sha256"),
                            param_sha256=identity.get("param_sha256"))
        migrated = self._migration_admitted(self.store._path(identity), recorded, identity, build)
        if not migrated and not self._native_context_matches(recorded, identity):
            return False
        observation = self.store.read(identity)
        if observation is None:
            return False
        self._validate_binding(recorded, identity, observation, semantic=migrated)
        return True

    def native_completed(self, context):
        """Recover an index from a native effective-contract signed receipt.

        Reconstruct the original signed context, including proven encoding
        variants, without publishing aliases or opening numerical artifacts.
        """
        build = self._build_admission()
        if not self._current_provenance(context):
            return None
        matches = []
        for identity in self.store.existing_completed_identities(seed=context["seed"], target=context["target"]):
            migrated = self._migration_admitted(self.store._path(identity), context, identity, build)
            if not migrated and not self._native_context_matches(context, identity):
                continue
            complete = self.store.with_win_rate(identity, self.store.read(identity))
            if complete is not None:
                self._validate_binding(context, identity, complete, semantic=migrated)
                matches.append((identity, complete))
        if matches and any(item[1] != matches[0][1] for item in matches[1:]):
            raise ValueError("Oracle same effective contract has conflicting native Replay receipts")
        return matches[0] if matches else None

    def publish(self, context: Mapping[str, Any], identity: Mapping[str, Any],
                observation: Mapping[str, Any], *, inspection: OracleReuseInspection | None = None) -> None:
        build = self._build_admission()
        if inspection is not None and (
                inspection.decision != "REUSE" or inspection.identity != dict(identity)
                or inspection.observation != dict(observation)
                or dict(inspection.origin_pins) != self._origin_pins(identity)):
            raise ValueError("Oracle completed origin changed after inspection")
        if not self._current_provenance(context):
            raise ValueError("Oracle cannot publish an unknown current implementation context")
        # Never turn a caller's metric dictionary into evidence. Resolve the
        # actual signed origin, including any immutable metric supplement.
        origin = self.store.read(identity)
        if origin is None or self.store.with_win_rate(identity, origin) != dict(observation):
            raise ValueError("Oracle cannot index an unverified completed Replay")
        result_contract = oracle_result_contract(context)
        migrated = self._migration_admitted(self.store._path(identity), context, identity, build)
        current_signed = (
            (identity.get("replay_policy") or {}).get("effective_result_contract_sha256") == result_contract.sha256
            and self._native_context_matches(context, identity)
        )
        reviewed = self._native_context_matches(context, identity)
        if not (migrated or current_signed or reviewed):
            raise ValueError("Oracle origin lacks a verified result-contract adoption proof")
        self._validate_binding(context, identity, observation, semantic=migrated or current_signed)
        path = self._path(context)
        admission = {
            "kind": ("verified_build_transition" if migrated else
                     "current_signed_producer" if current_signed else "reviewed_legacy_source"),
            "build_attestation_sha256": canonical_json_sha256(build),
            "context_sha256": canonical_json_sha256(context),
            "result_contract_sha256": result_contract.sha256,
            "origin_identity_sha256": canonical_json_sha256(identity),
            "observation_sha256": canonical_json_sha256(observation),
            "origin_pins": self._origin_pins(identity),
        }
        payload = {"schema": _RESULT_INDEX_SCHEMA, "context": dict(context),
                   "result_contract": result_contract.as_payload(), "admission": admission,
                   "origin_identity": dict(identity),
                   "observation_sha256": canonical_json_sha256(observation)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(path.with_name("." + path.name + ".lock")):
            if self._origin_pins(identity) != admission["origin_pins"]:
                raise ValueError("Oracle completed origin changed before adoption")
            current = self.read(context)
            if current is not None:
                if current[1] != dict(observation):
                    raise ValueError("Oracle same effective contract has conflicting completed results")
                if path.is_file():
                    return
            temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
            try:
                temporary.write_text(json.dumps({**payload, "reference_sha256": canonical_json_sha256(payload)},
                    ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                if self._origin_pins(identity) != admission["origin_pins"]:
                    raise ValueError("Oracle completed origin changed during adoption")
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

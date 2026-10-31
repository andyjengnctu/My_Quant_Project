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
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from core.file_integrity import (
    canonical_json_sha256, is_sha256_hex, exclusive_file_lock, atomic_replace_with_retry,
)

# Engineering calculation revisions, not scientific Target/model identities.
# Change only the affected capability after a result-affecting fix. Existing
# recipes are stable when a new recipe is registered alongside them.
PRICE_CALCULATION_REVISIONS = {
    "equal_rank_mfe_safety": 1, "mfe": 1, "terminal_r": 1,
    "terminal_mae_gate": 1, "mfe_mae_gate": 1,
}
DAILY_CALCULATION_REVISIONS = {"daily_trade_path_net_r": 1, "daily_trade_path_net_pnl": 1}
FEATURE_UNIVERSE_CALCULATION_REVISION = 1
REPLAY_CALCULATION_REVISION = 1
_SCHEMA = 1
_REFERENCE_FIELDS = {"reference_arm_id", "reference_score_source_id", "reference_profile_id"}


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
    return canonical_json_sha256(_syntax_payload(projected))


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


@lru_cache(maxsize=32)
def source_implementation_sha256(recipe: str, profile_id: str) -> str:
    parts = {"daily_percentile": calculation_source(
        "filters/breakout_quality/continuous_ranker_data.py", "build_same_date_percentile_targets")}
    if recipe in DAILY_CALCULATION_REVISIONS:
        parts.update({key: calculation_source("filters/breakout_quality/daily_trade40_target.py", name,
                         bound_literals=({"signal_state_reset_date": None,
                                          "include_net_pnl": recipe == "daily_trade_path_net_pnl"}
                                         if key == "min_path" else None))
                      for key, name in (
                          ("min_path", "compute_counterfactual_trade40_batch"),
                          ("rank_outcomes", "rank_trade40_outcome_frame"),
                          ("schedule", "build_formal_min_execution_schedule"),
                          ("active_period", "resolve_historical_min_period"))})
        parts["broker_path"] = calculation_source(
            "filters/breakout_quality/trade_path_label.py", "simulate_realized_trade_path_label")
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


def source_calculation_contract(*, recipe: str, profile_id: str,
                                horizon: int | None = None,
                                model_architecture: str | None = None) -> dict[str, Any]:
    from core.breakout_quality_registry import get_breakout_quality_experiment_profile
    from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
    profile = get_breakout_quality_experiment_profile(profile_id)
    from dataclasses import asdict
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.models.architectures import get_architecture_descriptor
    architecture = profile.model_architecture or model_architecture
    if not architecture:
        raise ValueError("Oracle feature provider requires the canonical source architecture")
    input_spec = get_model_spec(architecture)
    descriptor = get_architecture_descriptor(input_spec.architecture)
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
            "input_requirements": {
                "window_bars": int(input_spec.input_window_bars or DEFAULT_LABEL_POLICY.feature_window_bars),
                "dataset_context": bool(input_spec.use_dataset_context),
                "requires_market_set": bool(input_spec.requires_market_set),
                "derived_context_features": list(input_spec.derived_context_features),
                "eligibility_capabilities": {name: descriptor.has_capability(name) for name in (
                    "raw_unadjusted_price_input", "lagged_trade_count_input", "risk_context",
                    "predicted_upside_context", "predicted_safety_context")},
            },
        },
        "label_policy": DEFAULT_LABEL_POLICY.as_manifest_payload(),
        "score_mapping": "canonical_daily_truth_percentile_v2",
        "percentile_method": "average_zero_based",
    }


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
    policy = dict(replay_policy)
    policy.pop("exact_arm", None)
    payload = {
        "schema": _SCHEMA, "purpose": "hindsight_oracle_completed_replay",
        "replay_calculation_revision": REPLAY_CALCULATION_REVISION,
        "replay_implementation_sha256": replay_implementation_sha256(),
        "dataset": str(dataset), "price_sha256": price_sha256,
        "param_sha256": param_sha256, "seed": int(seed), "target": target,
        "start_date": start, "end_date": end, "research_cutoff": cutoff,
        "max_positions": int(max_positions), "rotation": "off",
        "evaluation_mode": evaluation_mode, "ranking_policy": ranking_policy,
        "ranking_options": dict(ranking_options), "replay_policy": policy,
        "score_contract": ({k: v for k, v in score_contract.items() if k not in _REFERENCE_FIELDS}
                           if score_contract is not None else None),
        "source_contract": dict(source_contract) if source_contract is not None else None,
        "dataset_inputs": dict(dataset_inputs) if dataset_inputs is not None and source_contract is not None else None,
        "execution_gate": execution_gate,
        "execution_gate_implementation": (_gate_implementation_sha256() if execution_gate is not None else None),
    }
    # Effective provider contracts contain tuples; persisted JSON represents
    # them as arrays. Normalize once at the boundary, before equality checks.
    return json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False))


@lru_cache(maxsize=1)
def _gate_implementation_sha256() -> str:
    code_root = Path(__file__).resolve().parents[2]
    return canonical_json_sha256({relative: _syntax_payload(_CalculationSyntax().visit(
        ast.parse((code_root / relative).read_text(encoding="utf-8-sig"))))
        for relative in ("core/oracle_positive_gate.py", "services/audit/oracle_execution_policy.py")})


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
        path = self._path(context)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            payload = {k: raw[k] for k in ("schema", "context", "origin_identity", "observation_sha256")}
            if (payload["schema"] != _SCHEMA or payload["context"] != dict(context)
                    or raw.get("reference_sha256") != canonical_json_sha256(payload)):
                raise ValueError("execution reference identity/SHA mismatch")
            identity = payload["origin_identity"]
            observation = self.store.read(identity)
            if observation is None:
                raise ValueError("execution reference origin receipt missing")
            complete = self.store.with_win_rate(identity, observation)
            if complete is None or canonical_json_sha256(complete) != payload["observation_sha256"]:
                raise ValueError("execution reference completed metrics differ")
            self._validate_binding(context, identity, complete)
            return dict(identity), complete
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Oracle completed execution reference corrupt: {path.name}: {exc}") from exc

    def _validate_binding(self, context, identity, observation):
        for key in ("target", "seed", "start_date", "end_date", "max_positions", "evaluation_mode",
                    "ranking_policy", "ranking_options", "rotation", "dataset"):
            if identity.get(key) != context.get(key):
                raise ValueError(f"execution reference origin mismatch: {key}")
        if (identity.get("dataset_sha256") != context["price_sha256"]
                or identity.get("param_sha256") != context["param_sha256"]):
            raise ValueError("execution reference price/P16 differs")
        policy = identity.get("replay_policy") or {}
        for key, value in context["replay_policy"].items():
            if key == "oracle_parity_implementation" and context["target"] == "C96":
                continue
            if policy.get(key) != value:
                raise ValueError(f"execution reference economic policy differs: {key}")
        if context["score_contract"] is not None:
            if self.keys(identity) is None:
                raise ValueError("execution reference missing certified stock-day membership")
            original = policy.get("canonical_dl_score_contract") or {}
            for key, value in context["score_contract"].items():
                if original.get(key) != value:
                    raise ValueError(f"execution reference score consumption differs: {key}")
        self.store.validate_completed_origin(identity, observation)

    def native_completed(self, context):
        """Recover an index from a native effective-contract signed receipt.

        Source-byte aliases and numerical artifacts are unnecessary: native
        receipts already name this exact execution-effective contract.
        """
        digest = canonical_json_sha256(context)
        matches = []
        for identity in self.store.existing_completed_identities(seed=context["seed"], target=context["target"]):
            if (identity.get("replay_policy") or {}).get("effective_execution_contract_sha256") != digest:
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

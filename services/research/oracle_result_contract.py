"""Stable Oracle result inputs, separate from a producer's source attestation.

Only named implementation witnesses are provenance. Unknown contract fields
remain effective inputs: adding a new economic gate cannot silently reuse an
older result. Source admission is performed by the index/Truth owners, never
by this pure projection.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.daily_trade_path_capabilities import DAILY_TRUTH_CAPABILITIES
from core.file_integrity import canonical_json_sha256


# Existing engineering revisions; these are not another scientific version.
PRICE_CALCULATION_REVISIONS = {
    "equal_rank_mfe_safety": 1, "mfe": 1, "terminal_r": 1,
    "terminal_mae_gate": 1, "mfe_mae_gate": 1,
}
DAILY_CALCULATION_REVISIONS = {
    "daily_trade_path_net_r": 1, "daily_trade_path_net_pnl": 1,
    "daily_trade_path_true_net_pnl": 2, "daily_trade_path_full_net_pnl": 1,
}
DAILY_CALCULATION_REVISIONS.update({
    recipe: spec["calculation_revision"] for recipe, spec in DAILY_TRUTH_CAPABILITIES.items()
})
FEATURE_UNIVERSE_CALCULATION_REVISION = 1
REPLAY_CALCULATION_REVISION = 1


def oracle_component_revisions() -> dict[str, int]:
    """The build certificate uses the same revision owners as result keys."""
    return {
        "replay": REPLAY_CALCULATION_REVISION,
        "feature_universe": FEATURE_UNIVERSE_CALCULATION_REVISION,
        **{"price/" + name: value for name, value in PRICE_CALCULATION_REVISIONS.items()},
        **{"daily/" + name: value for name, value in DAILY_CALCULATION_REVISIONS.items()},
    }


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def oracle_source_result_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    """Project a source already admitted by its original provenance owner."""
    return _json_copy({key: value for key, value in source.items()
                       if key != "implementation_sha256"})


_REFERENCE_FIELDS = {"reference_arm_id", "reference_score_source_id", "reference_profile_id"}

_SCORE_REFERENCE_FIELDS = _REFERENCE_FIELDS | {"positive_score_source_id", "positive_profile_id"}

_ORACLE_PERFECT_POSITIVE_INTERFACE = "oracle_perfect_netpnl_rank_and_positive_v1"

_LEGACY_POSITIVE_INTERFACE = "positive_netpnl_dual_output_v1"

_SINGLE_RANKER_POSITIVE_INTERFACE = "single_ranker_calibrated_positive_v2"

_POSITIVE_OUTPUT_COLUMN = "positive_p_positive"

_POSITIVE_OUTPUT_ID = "netpnl_positive_probability_v1"

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


def oracle_result_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(policy)
    result.pop("exact_arm", None)
    result.pop("direct_arm", None)
    result.pop("daily_candidate_implementation_sha256", None)
    positive = result.get("formal_positive_score_output")
    if isinstance(positive, dict):
        positive.pop("shared_admission_code_sha256", None)
        positive.pop("selector_arm_id", None)
    if isinstance(result.get("canonical_dl_score_contract"), dict):
        result["canonical_dl_score_contract"] = oracle_score_contract(result["canonical_dl_score_contract"])
    return result


def oracle_truth_result_contract(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Truth depends on teacher/data/economics, not Portfolio rank or capacity."""
    source = execution.get("source_contract")
    if source is None or not str(source.get("recipe", "")).startswith("daily_trade_path_"):
        return _json_copy({
            **{key: execution.get(key) for key in (
                "dataset", "price_sha256", "start_date", "end_date", "research_cutoff", "dataset_inputs")},
            "schema": 2,
            "source_contract": oracle_source_result_contract(source) if source is not None else None,
        })
    return _json_copy({
        **{key: execution[key] for key in (
            "dataset", "price_sha256", "param_sha256", "seed", "start_date", "end_date",
            "research_cutoff", "dataset_inputs",
        )},
        "schema": 2,
        "source_contract": oracle_source_result_contract(execution["source_contract"]),
        "filter_id": execution["replay_policy"]["filter_id"],
        "score_start": execution["score_contract"]["score_start_date"],
        "economics": {key: execution["replay_policy"][key] for key in (
            "fixed_risk", "max_position_cap_pct", "broker_accounting", "execution_defaults")},
    })


def oracle_score_result_contract(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Score mapping binds the admitted Truth and exact eligible membership."""
    return _json_copy({"truth": oracle_truth_result_contract(execution),
                       "score_contract": (oracle_score_contract(execution["score_contract"])
                                          if execution.get("score_contract") is not None else None)})


def oracle_replay_result_contract(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Replay retains all effective gates, selected ranking and portfolio inputs."""
    result = _json_copy(execution)
    result.pop("replay_implementation_sha256", None)
    result["ranking_options"] = oracle_ranking_options(result["ranking_options"])
    if result.get("score_contract") is not None:
        result["score_contract"] = oracle_score_contract(result["score_contract"])
    result["replay_policy"] = oracle_result_policy(result["replay_policy"])
    result["replay_policy"].pop("period_stats_implementation", None)
    if result.get("source_contract") is not None:
        result["source_contract"] = oracle_source_result_contract(result["source_contract"])
    return result


def oracle_observation_result_contract(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Persisted Oracle metrics use Replay's existing calculation/period revisions."""
    result = oracle_replay_result_contract(execution)
    policy = execution["replay_policy"]
    if "period_stats_implementation" in policy:
        result["replay_policy"]["period_stats_implementation"] = _json_copy(policy["period_stats_implementation"])
    return result


@dataclass(frozen=True)
class OracleResultContract:
    """Immutable canonical JSON; returned dictionaries never alias the key."""
    _canonical_json: str

    @classmethod
    def from_execution(cls, execution: Mapping[str, Any]) -> "OracleResultContract":
        if execution.get("purpose") != "hindsight_oracle_completed_replay":
            raise ValueError("Oracle result contract requires a completed Replay context")
        result = oracle_observation_result_contract(execution)
        return cls(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")))

    def as_payload(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.as_payload())


def oracle_build_source_paths(project_root: Path | None = None) -> tuple[str, ...]:
    """Capability inventory plus the transitive local Python import closure.

    A newly imported helper is automatically part of the build certificate.
    This inventory deliberately concerns admission, not result identity, so a
    conservative dependency can require a new certificate without a Replay.
    """
    from core.execution_source_dependencies import execution_source_inventory
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    pending = list(execution_source_inventory(
        "replay", "daily_prices", "mintrade_truth", "fulltrade_truth", "daily_common_filters"))
    pending.extend((
        "services/research/oracle_result_contract.py",
        "services/research/oracle_artifact_contract.py",
        "services/audit/target_oracle_romd.py",
        "services/audit/oracle_replay_artifacts.py",
        "services/audit/oracle_score_artifacts.py",
        "services/audit/oracle_truth_references.py",
        "services/audit/oracle_observation.py",
        "services/audit/mintrade_truth_source_attestation.py",
        "core/execution_source_dependencies.py",
    ))
    excluded = {"services/research/oracle_build_attestation.py"}
    allowed_roots = {"core", "filters", "services", "strategies", "config"}
    visited: set[str] = set()

    def enqueue_module(module: str) -> None:
        pieces = module.split(".")
        if not pieces or pieces[0] not in allowed_roots:
            return
        # Include package initializers, which execute before the imported owner.
        for count in range(1, len(pieces) + 1):
            package = "/".join(pieces[:count]) + "/__init__.py"
            if (root / package).is_file():
                pending.append(package)
        relative = "/".join(pieces) + ".py"
        if (root / relative).is_file():
            pending.append(relative)

    while pending:
        relative = pending.pop()
        if relative in visited or relative in excluded:
            continue
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Oracle build dependency is missing: {relative}")
        visited.add(relative)
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        package = relative[:-3].split("/")
        if package[-1] == "__init__":
            package.pop()
        else:
            package = package[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    enqueue_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                parent = package[:len(package) - node.level + 1] if node.level else []
                prefix = ".".join(parent + ([node.module] if node.module else []))
                enqueue_module(prefix)
                for alias in node.names:
                    if alias.name != "*":
                        enqueue_module(prefix + "." + alias.name)
            elif (isinstance(node, ast.Call) and node.args
                  and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)
                  and ((isinstance(node.func, ast.Name) and node.func.id in {"import_module", "__import__"})
                       or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"))):
                enqueue_module(node.args[0].value)
    return tuple(sorted(visited))

"""Immutable scientific observations and delta work plans for Strategy Robustness.

A comparison is a view over arm/seed observations, not their cache identity.
The store owns committed observations; it does not train models, resolve Latest
pointers, or compute results. Legacy evidence must be validated by its producer
before it is imported. A corrupt or conflicting receipt is never a cache miss.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from core.file_integrity import (
    atomic_write_text, canonical_json_sha256, exclusive_file_lock,
    file_change_signature,
)
from core.serialization_utils import json_native_value

Unit = tuple[str, int]
STORE_DIRECTORY = "scientific_units"
RECEIPT_SCHEMA = "strategy_robustness_scientific_unit_v1"


def scientific_unit_signature(
    *, contract: Mapping[str, Any], arm_id: str, seed: int,
) -> dict[str, Any] | None:
    """Project persisted science onto exactly one arm and its own dependencies.

    Keep the historical signature schema and field values. Adding members to a
    comparison, changing their order, or publishing another model is not an
    input to this observation. Training defaults do belong to model observations.
    """
    arm = next((dict(x) for x in contract.get("stochastic_arms", ())
                if str(x.get("arm_id", "")) == str(arm_id)), None)
    if arm is None:
        return None
    parameter = dict(dict(contract.get("benchmark_parameter_artifact_identities") or {})
                     .get(f"{arm_id}:seed={int(seed)}") or {})
    if not str(parameter.get("sha256") or "").strip():
        return None
    model = bool(arm.get("model_seed_sensitive"))
    signature = {
        "unit_result_identity_schema": "robustness_strategy_seed_result_v1",
        "scientific_contract_version": int(contract.get("scientific_contract_version") or 0),
        "dataset": str(contract.get("dataset") or ""),
        "dataset_identity": dict(contract.get("dataset_identity") or {}),
        "max_positions": int(contract.get("max_positions") or 0),
        "rotation": str(contract.get("rotation") or ""),
        "comparison_period": dict(contract.get("comparison_period") or {}),
        "benchmark_id": str(contract.get("benchmark_id") or ""),
        "seed": int(seed),
        "strategy_trials_per_fold": contract.get("strategy_trials_per_fold"),
        "seed_pairing": str(arm.get("seed_pairing") or contract.get("seed_pairing") or ""),
        "arm": {key: arm.get(key) for key in (
            "arm_id", "param_source", "param_policy", "rule_policy",
            "strategy_seed_sensitive", "model_seed_sensitive",
            *(("dl_id", "dl_runtime_mode", "dl_runtime_options", "runtime_dl_sources")
              if model else ()),
        )},
        "benchmark_parameter": {key: parameter[key] for key in (
            "identity_schema", "family", "evaluation_mode", "param_policy", "sha256",
        ) if parameter.get(key) not in (None, "")},
    }
    # These are economic replay inputs, but not fields of model FIT identity.
    # Keep the implicit historical Breakout/v1 representation unchanged.
    for key in ("entry_strategy", "replay_contract_version"):
        if key in arm:
            signature["arm"][key] = arm[key]
    if model:
        signature["training_defaults"] = dict(contract.get("training_defaults") or {})
    return signature


def model_source_signature(contract: Mapping[str, Any], dl_id: str) -> dict[str, Any]:
    """One score-workspace identity, independent of other sources/arms/seeds."""
    sources = [dict(source) for arm in contract.get("stochastic_arms", ())
               for source in arm.get("runtime_dl_sources", ())
               if str(source.get("dl_id") or "") == str(dl_id)]
    if not sources or any(source != sources[0] for source in sources[1:]):
        raise ValueError(f"Missing or inconsistent model source contract: {dl_id}")
    return {
        "dataset": contract.get("dataset"),
        "dataset_identity": contract.get("dataset_identity"),
        "comparison_period": contract.get("comparison_period"),
        "training_defaults": contract.get("training_defaults"),
        "source": sources[0],
    }


@dataclass(frozen=True)
class RobustnessWorkPlan:
    """The approved work set. Neither renderer nor executor can enlarge it."""
    fingerprint: str
    units: frozenset[Unit]
    reuse_units: frozenset[Unit]
    replay_units: frozenset[Unit]
    source_units: frozenset[Unit]
    source_identities: tuple[tuple[Unit, str], ...] = ()

    @classmethod
    def build(
        cls, *, fingerprint: str, units: Iterable[Unit], reuse_units: Iterable[Unit],
        dependencies: Mapping[str, Iterable[str]],
    ) -> "RobustnessWorkPlan":
        all_units = frozenset((str(a), int(s)) for a, s in units)
        reuse = frozenset((str(a), int(s)) for a, s in reuse_units)
        if not reuse <= all_units:
            raise ValueError("REUSE evidence contains units outside the current comparison")
        pending = all_units - reuse
        sources = frozenset((str(dl), seed) for arm, seed in pending
                            for dl in dependencies.get(arm, ()))
        return cls(str(fingerprint), all_units, reuse, pending, sources)

    def bind_sources(self, evidence: Iterable[tuple[Unit, Mapping[str, Any]]]) -> "RobustnessWorkPlan":
        """Pin shared source observations for pending controlled comparisons only."""
        identities = dict(self.source_identities)
        for unit, identity in evidence:
            if unit not in self.source_units:
                continue
            digest = canonical_json_sha256(json_native_value(dict(identity)))
            if unit in identities and identities[unit] != digest:
                raise ValueError(f"Conflicting original model-source observations: {unit}")
            identities[unit] = digest
        return replace(self, source_identities=tuple(sorted(identities.items())))

    def require_source_identity(self, unit: Unit, identity: Mapping[str, Any]) -> None:
        self.require_source(unit)
        expected = dict(self.source_identities).get(unit)
        if expected and expected != canonical_json_sha256(json_native_value(dict(identity))):
            raise ValueError(
                f"Pending strategy source differs from its completed same-source control: {unit}; "
                "completed observations remain unchanged"
            )

    def require_replay(self, unit: Unit) -> None:
        if unit not in self.replay_units:
            raise RuntimeError(f"Replay outside the approved delta plan: {unit}")

    def require_source(self, unit: Unit) -> None:
        if unit not in self.source_units:
            raise RuntimeError(f"Model work outside the approved delta plan: {unit}")

    def verify_completed(self, completed: Iterable[Unit]) -> None:
        if frozenset(completed) != self.reuse_units:
            raise RuntimeError("Scientific evidence changed after planning; re-open the original menu")


def pin_files(paths: Iterable[str | Path]) -> tuple[tuple[str, tuple], ...]:
    """TOCTOU guard for already SHA-verified files; never a substitute for SHA."""
    return tuple((str(path), file_change_signature(path))
                 for path in sorted({Path(x).resolve() for x in paths}, key=str))


def require_pinned_files(pins: Iterable[tuple[str, tuple]]) -> None:
    for path, signature in pins:
        if file_change_signature(path) != signature:
            raise RuntimeError(f"Verified dependency changed after planning: {Path(path).name}")


class ScientificUnitStore:
    """An atomic, immutable result index in the existing robustness domain.

    The receipt keeps the original score provenance, economic rows and references
    to verified trade evidence. It never substitutes the latest model score for
    the score actually consumed by an already completed experiment.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, signature: Mapping[str, Any]) -> Path:
        digest = canonical_json_sha256(json_native_value(dict(signature)))
        return self.root / STORE_DIRECTORY / digest[:2] / f"{digest}.json"

    def load(self, signature: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self.path(signature)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Invalid scientific-unit receipt")
        body = payload.get("body")
        if not isinstance(body, dict) or payload.get("sha256") != canonical_json_sha256(body):
            raise ValueError("Scientific-unit receipt SHA mismatch")
        if body.get("schema") != RECEIPT_SCHEMA or body.get("signature") != json_native_value(dict(signature)):
            raise ValueError("Scientific-unit receipt identity mismatch")
        evidence = body.get("observation")
        if not isinstance(evidence, dict):
            raise ValueError("Scientific-unit receipt has no observation")
        return evidence

    def publish(
        self, *, signature: Mapping[str, Any], observation: Mapping[str, Any],
        validate_equivalent: Callable[[dict[str, Any], dict[str, Any]], None],
    ) -> None:
        path = self.path(signature)
        current = json_native_value(dict(observation))
        with exclusive_file_lock(path.with_suffix(".lock")):
            previous = self.load(signature)
            if previous is not None:
                validate_equivalent(previous, current)
                return  # never rewrite a scientific receipt or its provenance
            body = {
                "schema": RECEIPT_SCHEMA,
                "signature": json_native_value(dict(signature)),
                "observation": current,
            }
            atomic_write_text(path, json.dumps(
                {"body": body, "sha256": canonical_json_sha256(body)},
                ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
            ) + "\n")


def model_observation_provenance_matches(
    signature: Mapping[str, Any], row: Mapping[str, Any], attribution: Mapping[str, Any],
) -> bool:
    """Validate the original multi-source score provenance, not a mutable Latest.

    The caller separately authenticates the result rows and all attribution bytes.
    All declared heads must be present and covered by the observation's SHA.
    """
    arm = dict(signature.get("arm") or {})
    declared = {str(source.get("dl_id") or "")
                for source in arm.get("runtime_dl_sources", ())}
    actual = dict(attribution.get("runtime_training_identities") or {})
    primary_id = str(arm.get("dl_id") or "")
    if not declared or set(actual) != declared or primary_id not in actual:
        return False
    norm = lambda value: str(value or "").strip().lower()
    runtime_sha = canonical_json_sha256(actual)
    primary = dict(actual[primary_id] or {})
    training = dict(attribution.get("training_identity") or {})
    if not norm(row.get("score_sha256")) or any(
        norm(row.get(key)) != norm(training.get(key))
        for key in ("score_sha256", "runtime_source_identity_sha256")
    ):
        return False
    if norm(row.get("runtime_source_identity_sha256")) != runtime_sha:
        return False
    if norm(primary.get("score_sha256")) != norm(row.get("score_sha256")):
        return False
    if any(not norm(dict(value or {}).get("score_sha256")) for value in actual.values()):
        return False
    return True

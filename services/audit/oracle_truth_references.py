"""Effective Min truth references shared by Exact, Direct and gate consumers.

The original NPZ and its source provenance are immutable. A reference records
which already certified calculation/input contract can consume it. First
adoption verifies arrays through the original truth owner; later inspections
validate the sealed original bytes and membership without decoding arrays.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from core.artifact_inspection import ArtifactInspection, InspectionStatus
from core.file_integrity import (
    canonical_json_sha256, atomic_replace_with_retry, exclusive_file_lock,
    compute_file_sha256, file_change_signature,
)


def source_context(execution: Mapping[str, Any]) -> dict[str, Any]:
    from services.research.oracle_result_contract import oracle_truth_result_contract
    source = execution["source_contract"]
    if not source or not str(source["recipe"]).startswith("daily_trade_path_"):
        raise ValueError("Min truth reference requires a daily trade-path capability")
    return oracle_truth_result_contract(execution)


def _effective_source_context(context):
    from services.research.oracle_result_contract import oracle_source_result_contract
    return {key: value for key, value in context.items()
            if key not in {"schema", "source_contract", "replay_calculation_revision",
                           "replay_implementation_sha256"}} | {
        "schema": 2,
        "source_contract": oracle_source_result_contract(context["source_contract"]),
    }


@dataclass(frozen=True)
class TruthArtifactPin:
    path: Path
    signature: tuple
    sha256: str


@dataclass(frozen=True)
class VerifiedTruthAdoption:
    context: dict[str, Any]
    origin_identity: dict[str, Any]
    evidence: dict[str, Any]
    pins: tuple[TruthArtifactPin, ...]
    evidence_sha256: str
    score_membership: tuple[str, int, str, str]


def _pin_truth_file(path: Path) -> TruthArtifactPin:
    before = file_change_signature(path)
    digest = compute_file_sha256(path)
    if file_change_signature(path) != before:
        raise ValueError("Oracle truth changed during inspection")
    return TruthArtifactPin(path.resolve(), before, digest)


def _verify_truth_pins(pins: tuple[TruthArtifactPin, ...]) -> None:
    for pin in pins:
        if _pin_truth_file(pin.path) != pin:
            raise ValueError("Oracle truth changed after inspection; adoption refused")


def _evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evidence.items()
            if key not in {"source_reuse_status", "effective_source_reference"}}


def _score_membership(scores):
    ordered = sorted(scores)
    if not ordered:
        raise ValueError("Oracle referenced truth has no score keys")
    digest = sha256()
    for ticker, day in ordered:
        digest.update(f"{ticker}|{day}\n".encode("utf-8"))
    return digest.hexdigest(), len(ordered), min(day for _, day in ordered), max(day for _, day in ordered)


def verify_truth_score_membership(
    scores: Mapping[tuple[str, str], float], origin_identity: Mapping[str, Any],
    *, certified_keys: tuple[str, int, str, str] | None = None,
) -> None:
    """Verify NPZ keys against the canonical Oracle score-universe witness.

    With an independently certified preflight witness, compare the *actual*
    ticker/date keys in the NPZ to the same canonical ``ticker|date`` digest
    required by Portfolio Replay. The original feature-cohort CSV digest is
    retained in the signed NPZ identity, but its DataFrame serialization is
    not a substitute for the signed Replay membership proof. Without such a
    witness (e.g. a standalone origin inspection), require the original CSV
    digest as before. No comparison uses score counts alone.
    """
    ordered = sorted(scores)
    if not ordered:
        raise ValueError("Oracle referenced truth has no score keys")
    if certified_keys is not None:
        from core.file_integrity import is_sha256_hex
        if (not isinstance(certified_keys, tuple) or len(certified_keys) != 4
                or not is_sha256_hex(certified_keys[0])
                or type(certified_keys[1]) is not int or certified_keys[1] <= 0
                or not isinstance(certified_keys[2], str)
                or not isinstance(certified_keys[3], str)):
            raise ValueError("Oracle certified Replay score membership proof is invalid")
        if _score_membership(scores) != certified_keys:
            raise ValueError("Oracle referenced truth score membership differs from certified Replay keys")
        return

    # Standalone legacy inspection has no separately certified Replay cohort.
    # Reconstruct the producer's original two-column CSV, including its
    # platform line endings, rather than assuming an unverified key set.
    import pandas as pd
    keys = pd.DataFrame.from_records(ordered, columns=["ticker", "date"])
    csv_digests = {
        sha256(keys.to_csv(index=False, lineterminator=newline).encode("utf-8")).hexdigest()
        for newline in ("\n", "\r\n")
    }
    if origin_identity["feature_eligible_keys_sha256"] not in csv_digests:
        raise ValueError("Oracle referenced truth feature membership differs from actual arrays")


class OracleTruthReferences:
    def __init__(self, cache_root: Path, *, project_root: Path | None = None):
        self.cache_root = Path(cache_root).resolve()
        self.root = self.cache_root.parent / "truth_index"
        self.project_root = (Path(project_root).resolve() if project_root is not None
                             else Path(__file__).resolve().parents[2])

    def _require_build(self):
        from services.research.oracle_build_attestation import require_oracle_build_attestation
        return require_oracle_build_attestation(self.project_root)

    def _path(self, context):
        return self.root / str(int(context["seed"])) / f"{canonical_json_sha256(context)}.json"

    @staticmethod
    def _matches(identity, context):
        mapping = {
            "dataset": "dataset", "price_sha256": "price_sha256", "param_sha256": "param_sha256",
            "seed": "seed", "filter_id": "filter_id", "evaluation_start": "start_date",
            "evaluation_end": "end_date", "research_cutoff": "research_cutoff", "score_start": "score_start",
        }
        return (all(identity.get(old) == context[new] for old, new in mapping.items()) and
                identity.get("profile_id") == context["source_contract"]["feature_provider"]["profile_id"] and
                identity.get("horizon_sessions") == context["source_contract"]["horizon_sessions"])

    def read(self, context):
        build = self._require_build()
        path = self._path(context)
        from services.research.oracle_artifact_contract import legacy_source_contract_matches
        from services.research.oracle_legacy_admission import verified_legacy_origin_pin
        wanted = _effective_source_context(context)
        matches = []
        for candidate in sorted(path.parent.glob("*.json")):
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                fields = ("schema", "context", "origin_identity", "evidence_sha256")
                if raw.get("schema") == 2:
                    fields += ("origin_sha256", "score_membership")
                payload = {key: raw[key] for key in fields}
                recorded = payload["context"]
                if (payload["schema"] not in (1, 2) or self._path(recorded) != candidate
                        or raw.get("reference_sha256") != canonical_json_sha256(payload)
                        or not self._matches(payload["origin_identity"], recorded)):
                    raise ValueError("truth reference contract/SHA mismatch")
                if payload["schema"] == 1:
                    pinned = (verified_legacy_origin_pin(
                        self.project_root, candidate, build,
                        result_contract_sha256=canonical_json_sha256(wanted),
                        identity_sha256=canonical_json_sha256(payload["origin_identity"]),
                        context_sha256=canonical_json_sha256(recorded),
                    ) if candidate.is_relative_to(self.project_root) else None)
                    if pinned is None and not legacy_source_contract_matches(
                            recorded["source_contract"], context["source_contract"]):
                        continue
                effective = _effective_source_context(recorded)
                if effective == wanted:
                    matches.append(raw)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Oracle truth reference corrupt: {candidate.name}: {exc}") from exc
        if matches and any(item["evidence_sha256"] != matches[0]["evidence_sha256"] for item in matches[1:]):
            raise ValueError("Oracle same effective truth contract has conflicting evidence")
        return max(matches, key=lambda item: item["schema"]) if matches else None

    def origin_path(self, identity):
        name = str(identity["target_id"])
        if not name.replace("_", "").isalnum():
            raise ValueError("Oracle invalid truth artifact name")
        return self.cache_root / str(int(identity["seed"])) / f"{name}_{canonical_json_sha256(identity)}.npz"

    def _publish(self, context, identity, evidence, *, pins: tuple[TruthArtifactPin, ...] = (),
                 score_membership=None):
        self._require_build()
        if not self._matches(identity, context):
            raise ValueError("Oracle truth origin differs from effective inputs")
        origin = self.origin_path(identity)
        if not pins:
            # Compatibility callers still pass through the canonical full reader.
            from services.audit.daily_trade40_oracle import _truth_cache_read
            verified = _truth_cache_read(origin, identity)
            if verified is None or _evidence(verified[1]) != _evidence(evidence):
                raise ValueError("Oracle truth publication requires its verified original NPZ")
            pins = (_pin_truth_file(origin),)
            score_membership = _score_membership(verified[0])
        origin_pin = next((pin for pin in pins if pin.path == origin.resolve()), None)
        if origin_pin is None:
            raise ValueError("Oracle truth adoption has no original artifact pin")
        payload = {"schema": 2, "context": context, "origin_identity": identity,
                   "evidence_sha256": canonical_json_sha256(_evidence(evidence)),
                   "origin_sha256": origin_pin.sha256,
                   "score_membership": list(score_membership)}
        path = self._path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(path.with_name("." + path.name + ".lock")):
            _verify_truth_pins(pins)
            existing = self.read(context)
            if existing is not None:
                if existing["evidence_sha256"] != payload["evidence_sha256"]:
                    raise ValueError("Oracle same truth contract has conflicting certified evidence")
                if existing.get("schema") == 2:
                    return
            temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
            try:
                temporary.write_text(json.dumps({**payload, "reference_sha256": canonical_json_sha256(payload)},
                    sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                _verify_truth_pins(pins)
                atomic_replace_with_retry(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def inspect_existing(self, execution, *, replay_identity=None, evidence=None,
                         certified_keys=None, producer_fingerprint=None) -> ArtifactInspection[VerifiedTruthAdoption]:
        """Validate an existing truth origin without publishing any reference.

        Preview must detect corrupt metadata just as an authorized migration
        would, but inspecting a signed Replay must never write truth indexes.
        """
        self._require_build()
        try:
            return self._inspect_existing(execution, replay_identity=replay_identity,
                evidence=evidence, certified_keys=certified_keys, producer_fingerprint=producer_fingerprint)
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
            return ArtifactInspection(InspectionStatus.INVALID, str(exc))

    def publish_existing(self, execution, *, replay_identity=None, evidence=None,
                         certified_keys=None, inspection=None, producer_fingerprint=None):
        """Publish only an already certified truth origin after authorization.

        A missing truth NPZ does not invalidate a completed Portfolio. Without
        an NPZ no source reference is published; a future consumer must BUILD.
        This never infers scores from portfolio summary statistics.
        """
        self._require_build()
        inspected = inspection if inspection is not None else self.inspect_existing(
            execution, replay_identity=replay_identity, evidence=evidence,
            certified_keys=certified_keys, producer_fingerprint=producer_fingerprint)
        if inspected.status is InspectionStatus.MISSING:
            return inspected
        if not inspected.ready or not isinstance(inspected.value, VerifiedTruthAdoption):
            raise ValueError(f"Oracle truth adoption rejected: {inspected.reason}")
        adoption = inspected.value
        if (adoption.context != source_context(execution)
                or adoption.evidence_sha256 != canonical_json_sha256(_evidence(adoption.evidence))):
            raise ValueError("Oracle truth adoption plan differs from inspected evidence")
        _verify_truth_pins(adoption.pins)
        self._publish(adoption.context, adoption.origin_identity, adoption.evidence,
                      pins=adoption.pins, score_membership=adoption.score_membership)
        return inspected

    def _inspect_existing(self, execution, *, replay_identity, evidence, certified_keys,
                          producer_fingerprint):
        build = self._require_build()
        context = source_context(execution)
        from services.research.oracle_artifact_contract import source_calculation_contract
        from services.research.oracle_result_contract import oracle_source_result_contract
        declared = execution["source_contract"]
        provider = declared.get("feature_provider") or {}
        current_source = source_calculation_contract(
            recipe=declared["recipe"], profile_id=provider["profile_id"],
            horizon=declared["horizon_sessions"], model_architecture=provider.get("model_architecture"),
        )
        if oracle_source_result_contract(declared) != oracle_source_result_contract(current_source):
            return ArtifactInspection(InspectionStatus.INCOMPATIBLE, "Truth calculation contract changed")
        reference = self.read(context)
        if replay_identity is not None:
            wanted = replay_identity["replay_policy"]["hindsight_target_contract"]
            if certified_keys is None:
                from services.research.oracle_artifact_contract import OracleExecutionIndex
                certified_keys = OracleExecutionIndex.keys(replay_identity)
        elif evidence is not None:
            wanted = _evidence(evidence)
        elif reference is not None:
            wanted = None
        else:
            return ArtifactInspection(InspectionStatus.MISSING, "No signed truth source evidence")
        # NPZ score_sha256 signs the producer's text-encoded rank sequence.
        # Replay perfect_score_sha256 signs a binary float64 score map. They
        # cover the same values using DIFFERENT encodings and cannot be equal
        # by contract. The signed Replay carries the producer's rank_sha256
        # explicitly; that is the matching provenance for an NPZ reference.
        target = str((wanted or reference["origin_identity"])["target_id"])
        if not target.replace("_", "").isalnum():
            raise ValueError("Oracle invalid truth artifact name")
        paths = sorted((self.cache_root / str(context["seed"])).glob(target + "_*.npz"))
        for path in paths:
            pin = _pin_truth_file(path)
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"].item()))
            identity = meta["identity"]
            if not self._matches(identity, context):
                continue
            original = meta["evidence"]
            if wanted is not None and any(original.get(key) != wanted.get(key) for key in (
                    "raw_label_sha256", "rank_sha256", "truth_source_sha256", "dataset_summary_sha256",
                    "calendar_sha256", "teacher", "target_id", "label_id", "mature_eligible_rows")):
                continue
            payload = {key: meta.get(key) for key in ("schema", "identity", "evidence", "score_sha256")}
            relative = path.relative_to(self.cache_root).as_posix()
            if path != self.origin_path(identity):
                raise ValueError(f"Oracle signed truth filename/identity mismatch: {relative}")
            if meta.get("payload_sha256") != canonical_json_sha256(payload):
                raise ValueError(f"Oracle signed truth metadata payload SHA mismatch: {relative}")
            if meta.get("score_sha256") != original["rank_sha256"]:
                raise ValueError(f"Oracle signed truth producer rank SHA mismatch: {relative}")
            trusted = (reference is not None and reference.get("schema") == 2
                       and reference["origin_identity"] == identity
                       and reference.get("origin_sha256") == pin.sha256
                       and reference["evidence_sha256"] == canonical_json_sha256(_evidence(original)))
            if not trusted:
                from services.research.oracle_legacy_admission import verified_legacy_origin_pin
                origin_proof = (verified_legacy_origin_pin(
                    self.project_root, path, build,
                    result_contract_sha256=canonical_json_sha256(context),
                    identity_sha256=canonical_json_sha256(identity),
                ) if path.is_relative_to(self.project_root) else None)
                if (origin_proof is None and (producer_fingerprint is None
                        or original.get("truth_source_sha256") != producer_fingerprint)):
                    return ArtifactInspection(InspectionStatus.INCOMPATIBLE,
                                              "Original truth producer is not attested")
            membership = tuple(reference["score_membership"]) if trusted else None
            if trusted and certified_keys is not None and membership != certified_keys:
                raise ValueError("Oracle referenced truth score membership differs from certified Replay keys")
            if not trusted:
                from services.audit.daily_trade40_oracle import _truth_cache_read
                loaded = _truth_cache_read(path, identity)
                if loaded is None:
                    raise ValueError("Oracle truth disappeared during inspection")
                verify_truth_score_membership(loaded[0], identity, certified_keys=certified_keys)
                membership = _score_membership(loaded[0])
            _verify_truth_pins((pin,))
            adoption = VerifiedTruthAdoption(context, dict(identity), dict(original), (pin,),
                                             canonical_json_sha256(_evidence(original)), membership)
            return ArtifactInspection(InspectionStatus.READY, "Verified original truth artifact", adoption)
        return ArtifactInspection(
            InspectionStatus.INCOMPATIBLE if paths else InspectionStatus.MISSING,
            "No truth artifact matches the signed source" if paths else "Original truth NPZ missing",
        )

    def load(self, execution, *, certified_keys: tuple[str, int, str, str] | None = None):
        context = source_context(execution)
        reference = self.read(context)
        if reference is None:
            return None
        from services.audit.daily_trade40_oracle import _truth_cache_read
        loaded = _truth_cache_read(self.origin_path(reference["origin_identity"]), reference["origin_identity"])
        if loaded is None:
            raise ValueError("Oracle referenced truth artifact is missing")
        scores, evidence = loaded
        if canonical_json_sha256(_evidence(evidence)) != reference["evidence_sha256"]:
            raise ValueError("Oracle referenced truth evidence differs from its certified origin")
        verify_truth_score_membership(
            scores, reference["origin_identity"], certified_keys=certified_keys,
        )
        return scores, {**evidence, "effective_source_reference": reference["reference_sha256"]}

    def validate_consumption(self, execution, evidence):
        reference = self.read(source_context(execution))
        if (reference is None or evidence.get("effective_source_reference") != reference["reference_sha256"]
                or canonical_json_sha256(_evidence(evidence)) != reference["evidence_sha256"]):
            raise ValueError("Oracle consumed truth lacks its effective calculation reference")
        return reference["reference_sha256"]


def load_or_build_daily_truth(*, execution, cache_root, producer,
                              certified_keys: tuple[str, int, str, str] | None = None,
                              producer_fingerprint: str | None = None):
    references = OracleTruthReferences(cache_root)
    loaded = references.load(execution, certified_keys=certified_keys)
    if loaded is not None:
        return loaded
    scores, evidence = producer()
    # This branch has actually invoked the selected producer; its canonical
    # signed NPZ must agree with the returned scores and provenance before adoption.
    from services.audit.daily_trade40_oracle import _ranked_score_sha
    if _ranked_score_sha(scores) != evidence.get("rank_sha256"):
        raise ValueError("Oracle produced scores differ from signed truth evidence")
    references.publish_existing(execution, evidence=evidence, certified_keys=certified_keys,
        producer_fingerprint=producer_fingerprint or evidence.get("truth_source_sha256"))
    if references.read(source_context(execution)) is not None:
        reference = references.read(source_context(execution))
        evidence = {**evidence, "effective_source_reference": reference["reference_sha256"]}
        references.validate_consumption(execution, evidence)
    return scores, evidence


def load_or_build_positive_truth(*, execution, cache_root, **kwargs):
    """Consume a certified source without relabelling its immutable provenance.

    An existing raw witness is read with its original source identity. If no
    raw witness exists, the canonical positive producer reconstructs values
    with current code and independently checks EVERY raw/rank value against
    the certified source. Its new witness then correctly names current code.
    """
    from services.audit import fullcycle_positive_truth as positive
    evidence = kwargs["canonical_evidence"]
    if "effective_source_reference" not in evidence:
        return positive.load_or_build_positive_truth(**kwargs)
    OracleTruthReferences(cache_root).validate_consumption(execution, evidence)
    identity = positive._identity(seed=kwargs["seed"], price_sha=kwargs["price_sha256"],
        param_sha=kwargs["param_sha256"], source_evidence=evidence, project_root=kwargs["project_root"])
    witness = positive._existing_raw_witness(kwargs["project_root"], kwargs["output_subdir"], identity)
    if witness is not None:
        existing, path = witness
        if not set(existing).issubset(kwargs["canonical_scores"]):
            raise ValueError("positive raw witness differs from certified score membership")
        if len(existing) != int(evidence["settled_net_pnl_rows"]):
            raise ValueError("positive raw witness settled count differs from certified truth")
        return existing, path
    source_ids = {str(item.source_id) for item in kwargs["canonical_schedule"]}
    if source_ids == {"formal_full_roos_seed"}:
        from services.audit.fulltrade_oracle_truth import truth_producer_fingerprint
        current_sha = truth_producer_fingerprint()
        if evidence.get("truth_source_sha256") != current_sha:
            raise ValueError("FullTrade source reference cannot be relabelled as current Full code")
    elif source_ids == {"formal_min_roos_seed"}:
        from services.audit.daily_trade40_oracle import truth_producer_fingerprint
        current_sha = truth_producer_fingerprint("net_pnl")
    else:
        raise ValueError("Positive witness requires a uniform formal teacher parameter family")
    current_evidence = {**evidence, "truth_source_sha256": current_sha}
    return positive.load_or_build_positive_truth(**{**kwargs, "canonical_evidence": current_evidence})

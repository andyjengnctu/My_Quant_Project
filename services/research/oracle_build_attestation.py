"""Admission of a verified Oracle build, separate from scientific result identity.

The engineering producer combines an explicitly reviewed source transition with
formal evidence. Runtime never manufactures a proof or falls back to a candidate.
The certificate and package are deliberately outside their own source inventory.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping
import zipfile
from collections import OrderedDict
from threading import RLock

from core.file_integrity import atomic_replace_with_retry, atomic_write_json, canonical_json_sha256


ATTESTATION_RELATIVE_PATH = "outputs/local_regression/oracle_contract/current.json"
REVIEW_RELATIVE_PATH = "outputs/local_regression/oracle_contract/review.json"
ATTESTATION_SCHEMA = "oracle_build_attestation.v1"
REVIEW_SCHEMA = "oracle_build_equivalence.v1"
EVIDENCE_DIRECTORY = "outputs/local_regression/oracle_contract/evidence"
_OWNER_PATH = "services/research/oracle_build_attestation.py"
_SOURCE_CAPTURE_CACHE = OrderedDict()
_SOURCE_CAPTURE_LOCK = RLock()


def _is_sha256(value) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


class OracleBuildBlockedError(RuntimeError):
    """The executing build has no applicable, complete source proof."""


@dataclass(frozen=True)
class OracleBuildAttestationStatus:
    ready: bool
    reason: str
    component_revisions: Mapping[str, int]
    source_digest: str
    payload: Mapping[str, Any] | None = None


def _relative_path(value: str) -> str:
    name = str(value)
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in ("..", ".") for part in path.parts)
            or path.as_posix() != name):
        raise OracleBuildBlockedError("Oracle build source path is not canonical")
    return name


def source_inventory_digest(inventory: Mapping[str, Mapping[str, str]]) -> str:
    if not inventory or any(not _is_sha256(record.get("sha256")) or not _is_sha256(record.get("syntax_sha256"))
                            for record in inventory.values()):
        raise OracleBuildBlockedError("Oracle source inventory has invalid digests")
    return canonical_json_sha256({
        _relative_path(name): str(record["syntax_sha256"])
        for name, record in sorted(inventory.items())
    })


def capture_oracle_source_inventory(project_root: str | Path) -> dict[str, dict[str, str]]:
    """Read every dependency's actual bytes; reuse only a content-identical projection.

Missing import candidates are also rechecked, so a newly introduced local helper
cannot inherit an earlier closure. Neither timestamps nor a previous PASS grant
admission. The small cache saves repeated AST traversal for a Target matrix.
"""
    from services.research.oracle_numerical_sources import oracle_build_source_parts_from_reader
    root = Path(project_root).resolve()
    content = {}
    observed = {}
    def read_source(name):
        name = _relative_path(name)
        if name in content:
            if content[name] is None:
                raise FileNotFoundError(name)
            return content[name]
        source = root / name
        if not source.resolve().is_relative_to(root):
            raise OracleBuildBlockedError("Oracle source closure escapes the checkout")
        try:
            raw = source.read_bytes()
        except FileNotFoundError:
            content[name], observed[name] = None, None
            raise
        content[name], observed[name] = raw, sha256(raw).hexdigest()
        return raw
    with _SOURCE_CAPTURE_LOCK:
        previous = _SOURCE_CAPTURE_CACHE.get(root)
    if previous is not None:
        expected, records = previous
        fresh = {}
        for name in expected:
            try:
                fresh[name] = sha256((root / name).read_bytes()).hexdigest()
            except FileNotFoundError:
                fresh[name] = None
        if fresh == expected:
            return {name: dict(record) for name, record in records.items()}
    parts = oracle_build_source_parts_from_reader(read_source)
    if not parts:
        raise OracleBuildBlockedError("Oracle source closure is empty")
    result = {}
    for owner, projection in parts.items():
        name = _relative_path(owner)
        if name == _OWNER_PATH or name.startswith(("apps/", "outputs/")) or name.endswith(".zip"):
            raise OracleBuildBlockedError("Oracle source closure contains a self-referential build owner")
        result[name] = {"sha256": sha256(read_source(name)).hexdigest(),
                        "syntax_sha256": sha256(projection.encode("utf-8")).hexdigest()}
    for engineering_owner in ("services/research/oracle_numerical_sources.py", "services/research/oracle_result_contract.py"):
        try:
            read_source(engineering_owner)
        except FileNotFoundError:
            continue
    with _SOURCE_CAPTURE_LOCK:
        _SOURCE_CAPTURE_CACHE[root] = (dict(observed), result)
        _SOURCE_CAPTURE_CACHE.move_to_end(root)
        while len(_SOURCE_CAPTURE_CACHE) > 8:
            _SOURCE_CAPTURE_CACHE.popitem(last=False)
    return result


def _component_revisions() -> dict[str, int]:
    from services.research.oracle_result_contract import oracle_component_revisions
    revisions = dict(oracle_component_revisions())
    if not revisions or any(type(value) is not int or value < 1 for value in revisions.values()):
        raise OracleBuildBlockedError("Oracle calculation revisions are unresolved")
    return revisions


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise OracleBuildBlockedError("Oracle evidence must be a JSON object")
    return payload


def _sealed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("attestation_sha256", None)
    return {**body, "attestation_sha256": canonical_json_sha256(body)}


def _validate_certificate(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != ATTESTATION_SCHEMA or payload.get("status") != "VERIFIED":
        raise OracleBuildBlockedError("Oracle build certificate is not VERIFIED")
    if _sealed_payload(payload)["attestation_sha256"] != payload.get("attestation_sha256"):
        raise OracleBuildBlockedError("Oracle build certificate integrity mismatch")
    if source_inventory_digest(payload["source_inventory"]) != payload.get("source_digest"):
        raise OracleBuildBlockedError("Oracle build source inventory digest mismatch")
    formal = dict(payload.get("formal_evidence") or {})
    if (formal.get("overall_status") != "PASS" or not formal.get("summary_sha256")
            or not formal.get("required_steps") or formal.get("selected_steps") != formal.get("required_steps")):
        raise OracleBuildBlockedError("Oracle build certificate lacks complete formal evidence")
    transition = dict(payload.get("equivalence_evidence") or {})
    if transition.get("kind") not in ("unchanged_sources", "reviewed_transition", "revision_change"):
        raise OracleBuildBlockedError("Oracle build certificate lacks source equivalence evidence")
    if transition.get("current_source_digest") != payload.get("source_digest"):
        raise OracleBuildBlockedError("Oracle source review belongs to another build")
    if not transition.get("baseline_source_digest"):
        raise OracleBuildBlockedError("Oracle source review lacks its independent baseline")
    if transition["kind"] == "unchanged_sources":
        if not transition.get("previous_attestation_sha256"):
            raise OracleBuildBlockedError("Oracle unchanged-source proof has no previous certificate")
    elif not transition.get("review_sha256") or not transition.get("proofs"):
        raise OracleBuildBlockedError("Oracle source transition has no reviewed evidence")


def inspect_oracle_build_attestation(project_root: str | Path) -> OracleBuildAttestationStatus:
    try:
        root = Path(project_root).resolve()
        payload = _load_json(root / ATTESTATION_RELATIVE_PATH)
        _validate_certificate(payload)
        revisions = _component_revisions()
        if payload.get("component_revisions") != revisions:
            raise OracleBuildBlockedError("Oracle build calculation revision is not certified")
        inventory = capture_oracle_source_inventory(root)
        digest = source_inventory_digest(inventory)
        if digest != payload.get("source_digest"):
            raise OracleBuildBlockedError("Oracle build source closure changed; equivalence review required")
        return OracleBuildAttestationStatus(True, "verified", revisions, digest, payload)
    except OSError as exc:
        return OracleBuildAttestationStatus(False, f"IMPLEMENTATION_UNVERIFIED: evidence/source unavailable ({type(exc).__name__})", {}, "")
    except (ValueError, KeyError, TypeError, SyntaxError, OracleBuildBlockedError) as exc:
        detail = str(exc) if isinstance(exc, OracleBuildBlockedError) else type(exc).__name__
        return OracleBuildAttestationStatus(False, f"IMPLEMENTATION_UNVERIFIED: {detail}", {}, "")


def require_oracle_build_attestation(project_root: str | Path) -> dict[str, Any]:
    status = inspect_oracle_build_attestation(project_root)
    if not status.ready:
        raise OracleBuildBlockedError(status.reason)
    return dict(status.payload or {})


def _read_pinned_json(reference: Mapping[str, Any], *, relative_to: Path) -> dict[str, Any]:
    path = Path(str(reference["path"]))
    if not path.is_absolute():
        path = relative_to / path
    content = path.read_bytes()
    if sha256(content).hexdigest() != reference.get("sha256"):
        raise OracleBuildBlockedError("Oracle equivalence receipt bytes changed")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise OracleBuildBlockedError("Oracle equivalence receipt is not an object")
    return payload


def _verify_review(root: Path, path: Path, inventory, revisions) -> dict[str, Any]:
    review = _load_json(path)
    if review.get("schema") != REVIEW_SCHEMA:
        raise OracleBuildBlockedError("Oracle source review schema mismatch")
    current = dict(review.get("current") or {})
    baseline = dict(review.get("baseline") or {})
    if current.get("source_inventory") != inventory or current.get("component_revisions") != revisions:
        raise OracleBuildBlockedError("Oracle review does not match the exact current source closure")
    if current.get("source_digest") != source_inventory_digest(inventory):
        raise OracleBuildBlockedError("Oracle review current source digest mismatch")
    old_revisions = dict(baseline.get("component_revisions") or {})
    if set(old_revisions) != set(revisions) or any(
        type(old_revisions[name]) is not int or old_revisions[name] < 1 or revisions[name] < old_revisions[name]
        for name in revisions
    ):
        raise OracleBuildBlockedError("Oracle revision rollback or unresolved component is forbidden")
    changes = {name: {"from": old_revisions[name], "to": revisions[name]}
               for name in revisions if old_revisions[name] != revisions[name]}
    review_kind = str(review.get("kind") or "reviewed_transition")
    if changes:
        if (review_kind != "revision_change" or review.get("revision_changes") != changes
                or review.get("decision_kind") not in ("correctness_fix", "approved_semantic_change")
                or not str(review.get("decision_reference") or "").strip()):
            raise OracleBuildBlockedError("Oracle revision change requires an explicit component decision")
    elif review_kind != "reviewed_transition":
        raise OracleBuildBlockedError("Oracle review classification does not match its revisions")
    snapshot = Path(str(baseline["snapshot_path"]))
    if not snapshot.is_absolute():
        snapshot = path.parent / snapshot
    if sha256(snapshot.read_bytes()).hexdigest() != baseline.get("snapshot_sha256"):
        raise OracleBuildBlockedError("Oracle baseline snapshot integrity mismatch")
    baseline_inventory = dict(baseline["source_inventory"])
    if not baseline_inventory or source_inventory_digest(baseline_inventory) != baseline.get("source_digest"):
        raise OracleBuildBlockedError("Oracle baseline source inventory mismatch")
    with zipfile.ZipFile(snapshot) as archive:
        from services.research.oracle_numerical_sources import oracle_build_source_parts_from_reader
        baseline_parts = oracle_build_source_parts_from_reader(archive.read)
        if set(baseline_parts) != set(baseline_inventory):
            raise OracleBuildBlockedError("Oracle baseline numerical closure is incomplete")
        for name, record in baseline_inventory.items():
            reconstructed = {"sha256": sha256(archive.read(_relative_path(name))).hexdigest(),
                             "syntax_sha256": sha256(baseline_parts[name].encode("utf-8")).hexdigest()}
            if reconstructed != record:
                raise OracleBuildBlockedError("Oracle baseline source does not match its snapshot")
    evidence_files = {}
    def retain(name, source, digest):
        file_name = Path(name)
        relative = f"{EVIDENCE_DIRECTORY}/{file_name.stem}_{digest[:16]}{file_name.suffix}"
        evidence_files[relative] = {"path": str(source), "sha256": digest}
        return {"path": relative, "sha256": digest}
    baseline_reference = retain("baseline_sources.zip", snapshot, baseline["snapshot_sha256"])
    retain("review.json", path, sha256(path.read_bytes()).hexdigest())
    proofs = list(review.get("proofs") or [])
    parity_count = 0
    proved_components = set()
    changed_sources = {
        name for name in set(baseline_inventory) | set(inventory)
        if dict(baseline_inventory.get(name) or {}).get("syntax_sha256")
        != dict(inventory.get(name) or {}).get("syntax_sha256")
    }
    covered_sources = set()
    retained_proofs = []
    for index, proof in enumerate(proofs):
        if proof.get("kind") not in ("independent_observation_parity", "independent_component_parity"):
            raise OracleBuildBlockedError("Unknown Oracle equivalence proof kind")
        covered = proof.get("covered_changed_sources")
        if (not isinstance(covered, list) or any(not isinstance(name, str) for name in covered)
                or len(set(covered)) != len(covered) or not set(covered).issubset(changed_sources)):
            raise OracleBuildBlockedError("Oracle parity proof has missing or unknown source coverage")
        covered_sources.update(covered)
        left = _read_pinned_json(proof["baseline_receipt"], relative_to=path.parent)
        right = _read_pinned_json(proof["current_receipt"], relative_to=path.parent)
        retained_proof = {"kind": proof["kind"], "covered_changed_sources": sorted(covered)}
        for side in ("baseline", "current"):
            reference = proof[f"{side}_receipt"]
            receipt_path = Path(reference["path"])
            if not receipt_path.is_absolute():
                receipt_path = path.parent / receipt_path
            retained_proof[f"{side}_receipt"] = retain(f"{side}_parity_{index}.json", receipt_path, reference["sha256"])
        retained_proofs.append(retained_proof)
        for observed, expected in ((left, baseline), (right, current)):
            if (observed.get("status") != "PASS" or observed.get("source_digest") != expected["source_digest"]
                    or not observed.get("process_token") or not _is_sha256(observed.get("input_sha256"))
                    or not _is_sha256(observed.get("observation_sha256"))
                    or type(observed.get("case_count")) is not int or observed["case_count"] < 1):
                raise OracleBuildBlockedError("Oracle independent parity evidence is incomplete")
        if (left["process_token"] == right["process_token"]
                or any(left[key] != right[key] for key in ("input_sha256", "case_count"))):
            raise OracleBuildBlockedError("Oracle independent baseline/current parity differs")
        if proof["kind"] == "independent_observation_parity":
            if changes or left["observation_sha256"] != right["observation_sha256"]:
                raise OracleBuildBlockedError("Oracle same-revision observation parity differs")
            proved_components.update(revisions)
        else:
            left_components = dict(left.get("component_observations") or {})
            right_components = dict(right.get("component_observations") or {})
            for name in set(revisions) - set(changes):
                if left_components.get(name) and left_components.get(name) == right_components.get(name):
                    proved_components.add(name)
        parity_count += int(left["case_count"])
    if not parity_count or not (set(revisions) - set(changes)).issubset(proved_components):
        raise OracleBuildBlockedError("Formal PASS cannot replace independent Oracle equivalence evidence")
    if covered_sources != changed_sources:
        raise OracleBuildBlockedError("Oracle changed numerical sources lack scoped parity evidence")
    origins = dict(review.get("legacy_origins") or {})
    if origins:
        audit_reference = dict(review.get("legacy_audit_receipt") or {})
        audit = _read_pinned_json(audit_reference, relative_to=path.parent)
        audited_origins = dict(audit.get("validated_origins") or {})
        if (audit.get("status") != "PASS" or audit.get("source_digest") != baseline["source_digest"]
                or audit.get("snapshot_sha256") != baseline["snapshot_sha256"]
                or not audit.get("process_token")
                or any(audited_origins.get(name) != record for name, record in origins.items())):
            raise OracleBuildBlockedError("Oracle legacy origins lack independent baseline validation")
        audit_path = Path(audit_reference["path"])
        if not audit_path.is_absolute():
            audit_path = path.parent / audit_path
        retain("baseline_legacy_audit.json", audit_path, audit_reference["sha256"])
    for name, record in origins.items():
        if changes:
            relevant = dict(record.get("component_revisions") or {})
            if (not relevant or any(component not in revisions or type(value) is not int
                                    or value != old_revisions[component] or value != revisions[component]
                                    for component, value in relevant.items())):
                raise OracleBuildBlockedError("Legacy origin has changed or unproven component revisions")
        receipt = root / _relative_path(name)
        if not receipt.resolve().is_relative_to(root) or sha256(receipt.read_bytes()).hexdigest() != record.get("sha256"):
            raise OracleBuildBlockedError("Oracle legacy origin receipt changed")
    return {
        "kind": review_kind, "review_sha256": sha256(path.read_bytes()).hexdigest(),
        "baseline_commit": str(baseline.get("commit") or ""),
        "baseline_source_digest": baseline["source_digest"],
        "current_source_digest": current["source_digest"], "parity_case_count": parity_count,
        "proofs": retained_proofs, "baseline_snapshot": baseline_reference,
        "structurally_unchanged_sources": sorted((set(baseline_inventory) & set(inventory)) - changed_sources),
        "legacy_origins": origins, "evidence_files": evidence_files,
        **({"revision_changes": changes, "decision_kind": review["decision_kind"],
            "decision_reference": review["decision_reference"]} if changes else {}),
    }


def prepare_oracle_build_candidate(project_root: str | Path, *, review_path: Path | None = None) -> dict[str, Any]:
    """Validate engineering evidence without publishing or granting runtime admission."""
    root = Path(project_root).resolve()
    inventory = capture_oracle_source_inventory(root)
    revisions = _component_revisions()
    digest = source_inventory_digest(inventory)
    previous = None
    certificate = root / ATTESTATION_RELATIVE_PATH
    if certificate.is_file():
        previous = _load_json(certificate)
        _validate_certificate(previous)
    if (previous is not None and previous.get("source_digest") == digest
            and previous.get("component_revisions") == revisions):
        evidence = {"kind": "unchanged_sources", "previous_attestation_sha256": previous["attestation_sha256"],
                    "baseline_source_digest": digest, "current_source_digest": digest}
        origins = dict(previous.get("legacy_origins") or {})
        evidence_files = {name: {"path": str(root / name), "sha256": digest}
                          for name, digest in dict(previous.get("evidence_artifacts") or {}).items()}
    else:
        evidence = _verify_review(root, review_path or root / REVIEW_RELATIVE_PATH, inventory, revisions)
        origins = evidence.pop("legacy_origins")
        evidence_files = evidence.pop("evidence_files")
    return {"source_inventory": inventory, "source_digest": digest,
            "component_revisions": revisions, "equivalence_evidence": evidence, "legacy_origins": origins,
            "evidence_files": evidence_files,
            "evidence_artifacts": {name: record["sha256"] for name, record in evidence_files.items()}}


def finalize_oracle_build_attestation(project_root: str | Path, *, candidate: Mapping[str, Any],
                                     formal_summary: Mapping[str, Any], required_steps: tuple[str, ...],
                                     formal_summary_sha256: str, commit: str) -> dict[str, Any]:
    """Construct a final certificate only for the exact sources actually tested."""
    root = Path(project_root).resolve()
    if capture_oracle_source_inventory(root) != candidate.get("source_inventory"):
        raise OracleBuildBlockedError("Oracle sources changed during formal verification")
    if _component_revisions() != candidate.get("component_revisions"):
        raise OracleBuildBlockedError("Oracle revisions changed during formal verification")
    scripts = list(formal_summary.get("scripts") or [])
    selected = tuple(formal_summary.get("selected_steps") or ())
    statuses = {row.get("name"): row.get("status") for row in scripts}
    recorded_commit = str(formal_summary.get("git_commit") or "")
    commit_matches = bool(re.fullmatch(r"[0-9a-f]{7,40}", recorded_commit)) and commit.startswith(recorded_commit)
    if (formal_summary.get("overall_status") != "PASS" or formal_summary.get("failures") != 0
            or formal_summary.get("payload_failures") or formal_summary.get("not_run_step_names")
            or formal_summary.get("failed_step_names") or sorted(selected) != sorted(required_steps)
            or len(statuses) != len(scripts)
            or not required_steps or any(statuses.get(name) != "PASS" for name in required_steps)
            or not commit_matches or not _is_sha256(formal_summary_sha256)):
        raise OracleBuildBlockedError("Oracle certificate requires this build's complete formal PASS")
    payload = _sealed_payload({
        "schema": ATTESTATION_SCHEMA, "status": "VERIFIED",
        **{key: value for key, value in candidate.items() if key != "evidence_files"}, "git_commit": commit,
        "formal_evidence": {"overall_status": "PASS", "summary_sha256": formal_summary_sha256,
                            "selected_steps": sorted(selected), "required_steps": sorted(required_steps)},
    })
    _validate_certificate(payload)
    return payload


def publish_oracle_build_attestation(project_root: str | Path, payload: Mapping[str, Any], *,
                                    evidence_files: Mapping[str, Mapping[str, str]] | None = None) -> Path:
    """Called after frozen-package finalization; an earlier proof survives failures."""
    _validate_certificate(payload)
    root = Path(project_root).resolve()
    if source_inventory_digest(capture_oracle_source_inventory(root)) != payload.get("source_digest"):
        raise OracleBuildBlockedError("Oracle source closure changed before certificate publication")
    for name, expected in dict(payload.get("evidence_artifacts") or {}).items():
        if not _relative_path(name).startswith(EVIDENCE_DIRECTORY + "/"):
            raise OracleBuildBlockedError("Oracle evidence destination is outside its owner")
        record = dict((evidence_files or {}).get(name) or {})
        source = Path(record.get("path") or root / name)
        content = source.read_bytes()
        if sha256(content).hexdigest() != expected:
            raise OracleBuildBlockedError("Oracle evidence changed before publication")
        destination = root / name
        if source.resolve() == destination.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as pending:
            pending.write(content)
            pending_path = Path(pending.name)
        try:
            atomic_replace_with_retry(pending_path, destination)
        finally:
            pending_path.unlink(missing_ok=True)
    return atomic_write_json(root / ATTESTATION_RELATIVE_PATH, dict(payload))

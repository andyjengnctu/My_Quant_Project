"""Isolated evidence lifecycle tests; no production certificate or artifact writes."""
from __future__ import annotations

from copy import deepcopy
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from services.research import oracle_build_attestation as owner


class OracleBuildAttestationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="oracle_build_proof_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.revisions = {"replay": 1, "observation": 1}
        self.addCleanup(patch.stopall)
        def projected_sources(reader):
            from services.research.oracle_numerical_sources import _Syntax
            raw = reader("calculation.py")
            result = {"calculation.py": ast.dump(_Syntax().visit(ast.parse(raw)), include_attributes=False)}
            if b"import new" in raw:
                result["new.py"] = ast.dump(ast.parse(reader("new.py")), include_attributes=False)
            return result
        patch("services.research.oracle_numerical_sources.oracle_build_source_parts_from_reader", side_effect=projected_sources).start()
        patch("services.research.oracle_result_contract.oracle_component_revisions", side_effect=lambda: self.revisions).start()
        self.source = self.root / "calculation.py"
        self.source.write_text("def calculate(x=1):\n    return x + 1\n", encoding="utf-8")
        self.baseline = owner.capture_oracle_source_inventory(self.root)
        self.snapshot = self.root / "baseline.zip"
        with zipfile.ZipFile(self.snapshot, "w") as archive:
            archive.writestr("calculation.py", self.source.read_bytes())
        self.current = owner.capture_oracle_source_inventory(self.root)
        self.review_path = self.root / owner.REVIEW_RELATIVE_PATH
        self.review_path.parent.mkdir(parents=True)
        self.review = self.make_review()
        self.write_review()

    def write_json(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}

    def make_review(self):
        old_digest = owner.source_inventory_digest(self.baseline)
        current_digest = owner.source_inventory_digest(self.current)
        proofs = {}
        for side, digest in (("baseline", old_digest), ("current", current_digest)):
            proofs[f"{side}_receipt"] = self.write_json(f"{side}.json", {
                "status": "PASS", "source_digest": digest, "process_token": side,
                "input_sha256": "a" * 64, "observation_sha256": "b" * 64, "case_count": 4,
                "component_observations": {"replay": "c" * 64, "observation": "d" * 64},
            })
        return {
            "schema": owner.REVIEW_SCHEMA,
            "baseline": {"source_inventory": self.baseline, "source_digest": old_digest,
                         "component_revisions": dict(self.revisions), "commit": "a" * 40,
                         "snapshot_path": str(self.snapshot), "snapshot_sha256": sha256(self.snapshot.read_bytes()).hexdigest()},
            "current": {"source_inventory": self.current, "source_digest": current_digest,
                        "component_revisions": dict(self.revisions)},
            "proofs": [{"kind": "independent_observation_parity", "covered_changed_sources": [
                name for name in set(self.baseline) | set(self.current)
                if self.baseline.get(name, {}).get("syntax_sha256") != self.current.get(name, {}).get("syntax_sha256")
            ], **proofs}],
        }

    def write_review(self):
        self.review_path.write_text(json.dumps(self.review), encoding="utf-8")

    def finalize(self, candidate=None, **overrides):
        summary = {"overall_status": "PASS", "failures": 0, "scripts": [{"name": "formal", "status": "PASS"}],
                   "selected_steps": ["formal"], "git_commit": "a" * 40, **overrides}
        return owner.finalize_oracle_build_attestation(
            self.root, candidate=candidate or owner.prepare_oracle_build_candidate(self.root),
            formal_summary=summary, required_steps=("formal",), formal_summary_sha256="e" * 64, commit="a" * 40,
        )

    def publish(self):
        candidate = owner.prepare_oracle_build_candidate(self.root)
        payload = self.finalize(candidate)
        owner.publish_oracle_build_attestation(self.root, payload, evidence_files=candidate["evidence_files"])
        return payload

    def test_missing_candidate_and_corrupt_certificate_never_admit_runtime(self):
        status = owner.inspect_oracle_build_attestation(self.root)
        self.assertFalse(status.ready)
        self.assertIn("IMPLEMENTATION_UNVERIFIED", status.reason)
        self.assertNotIn(str(self.root), status.reason)
        owner.prepare_oracle_build_candidate(self.root)
        with self.assertRaises(owner.OracleBuildBlockedError):
            owner.require_oracle_build_attestation(self.root)
        self.publish()
        self.assertTrue(owner.inspect_oracle_build_attestation(self.root).ready)
        path = self.root / owner.ATTESTATION_RELATIVE_PATH
        payload = json.loads(path.read_text())
        payload["component_revisions"]["replay"] = 99
        path.write_text(json.dumps(payload))
        self.assertFalse(owner.inspect_oracle_build_attestation(self.root).ready)

    def test_unknown_same_size_source_and_dependency_drift_are_blocked(self):
        self.publish()
        original = self.source.read_bytes()
        timestamp = self.source.stat()
        self.source.write_bytes(original.replace(b"x + 1", b"x - 1"))
        os.utime(self.source, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
        self.assertFalse(owner.inspect_oracle_build_attestation(self.root).ready)
        self.source.write_bytes(original)
        self.source.write_bytes(b"import new\n" + original)
        (self.root / "new.py").write_text("pass\n")
        self.assertFalse(owner.inspect_oracle_build_attestation(self.root).ready)

    def test_only_comments_docstrings_and_encoding_are_structurally_neutral(self):
        payload = self.publish()
        self.source.write_bytes(b'# comment\r\n"""module docs"""\r\ndef calculate(x=1):\r\n    """function docs"""\r\n    return x + 1\r\n')
        self.assertTrue(owner.inspect_oracle_build_attestation(self.root).ready)
        next_candidate = owner.prepare_oracle_build_candidate(self.root)
        self.assertEqual(next_candidate["source_digest"], payload["source_digest"])
        self.assertEqual(next_candidate["equivalence_evidence"]["kind"], "unchanged_sources")
        for changed in ("def calculate(x=2):\n    return x + 1\n", "def calculate(x: int=1):\n    return x + 1\n",
                        "@decorator\ndef calculate(x=1):\n    return x + 1\n"):
            self.source.write_text(changed)
            self.assertFalse(owner.inspect_oracle_build_attestation(self.root).ready)

    def test_pass_partial_fail_or_drift_cannot_certify(self):
        candidate = owner.prepare_oracle_build_candidate(self.root)
        self.assertEqual(self.finalize(candidate, git_commit="aaaaaaaa")["git_commit"], "a" * 40)
        for override in ({"overall_status": "FAIL"}, {"selected_steps": []}, {"failures": 1},
                         {"payload_failures": ["error"]}, {"git_commit": "wrong"},
                         {"selected_steps": ["formal", "formal"]},
                         {"scripts": [{"name": "formal", "status": "SKIP"}]}):
            with self.subTest(override=override), self.assertRaises(owner.OracleBuildBlockedError):
                self.finalize(candidate, **override)
        self.source.write_text("changed = 1\n")
        with self.assertRaises(owner.OracleBuildBlockedError):
            self.finalize(candidate)

    def test_unproved_transition_and_non_independent_or_different_parity_are_rejected(self):
        original = deepcopy(self.review)
        for replacement in ([], [{"kind": "unknown"}]):
            self.review["proofs"] = replacement
            self.write_review()
            with self.assertRaises(owner.OracleBuildBlockedError):
                owner.prepare_oracle_build_candidate(self.root)
        for field, value in (("process_token", "baseline"), ("observation_sha256", "f" * 64), ("input_sha256", "f" * 64)):
            self.review = deepcopy(original)
            receipt = json.loads((self.root / "current.json").read_text())
            receipt[field] = value
            self.review["proofs"][0]["current_receipt"] = self.write_json("bad.json", receipt)
            self.write_review()
            with self.assertRaises(owner.OracleBuildBlockedError):
                owner.prepare_oracle_build_candidate(self.root)

    def test_legacy_pin_needs_independent_baseline_audit(self):
        origin = self.root / "receipt.json"
        origin.write_text('{"immutable": true}')
        origins = {"receipt.json": {"sha256": sha256(origin.read_bytes()).hexdigest(),
                                    "identity_sha256": "f" * 64, "result_contract_sha256": "d" * 64}}
        self.review["legacy_origins"] = origins
        self.write_review()
        with self.assertRaises((owner.OracleBuildBlockedError, KeyError)):
            owner.prepare_oracle_build_candidate(self.root)
        self.review["legacy_audit_receipt"] = self.write_json("audit.json", {
            "status": "PASS", "source_digest": self.review["baseline"]["source_digest"],
            "snapshot_sha256": self.review["baseline"]["snapshot_sha256"], "process_token": "independent-baseline",
            "validated_origins": origins,
        })
        self.write_review()
        self.assertEqual(owner.prepare_oracle_build_candidate(self.root)["legacy_origins"], origins)
        origin.write_text("changed")
        with self.assertRaises(owner.OracleBuildBlockedError):
            owner.prepare_oracle_build_candidate(self.root)

    def test_changed_numerical_owner_needs_explicit_probe_coverage(self):
        self.source.write_text("def calculate(x=1):\n    return x + 2\n")
        self.current = owner.capture_oracle_source_inventory(self.root)
        self.review = self.make_review()
        self.write_review()
        self.assertEqual(owner.prepare_oracle_build_candidate(self.root)["source_inventory"], self.current)
        for coverage in (None, [], ["unrelated_teacher.py"]):
            self.review["proofs"][0]["covered_changed_sources"] = coverage
            self.write_review()
            with self.assertRaises(owner.OracleBuildBlockedError):
                owner.prepare_oracle_build_candidate(self.root)
    def test_revision_increase_is_explicit_and_unaffected_components_are_proved(self):
        self.revisions = {"replay": 2, "observation": 1}
        self.review["current"]["component_revisions"] = dict(self.revisions)
        self.write_review()
        with self.assertRaises(owner.OracleBuildBlockedError):
            owner.prepare_oracle_build_candidate(self.root)
        self.review.update(kind="revision_change", revision_changes={"replay": {"from": 1, "to": 2}},
                           decision_kind="correctness_fix", decision_reference="reviewed existing execution contract")
        self.review["proofs"][0]["kind"] = "independent_component_parity"
        self.write_review()
        candidate = owner.prepare_oracle_build_candidate(self.root)
        self.assertEqual(self.finalize(candidate)["component_revisions"]["replay"], 2)
        self.review["baseline"]["component_revisions"]["replay"] = 3
        self.write_review()
        with self.assertRaises(owner.OracleBuildBlockedError):
            owner.prepare_oracle_build_candidate(self.root)

    def test_snapshot_tamper_and_failed_finalization_preserve_existing_certificate(self):
        previous = self.publish()
        certificate = self.root / owner.ATTESTATION_RELATIVE_PATH
        before = certificate.read_bytes()
        candidate = owner.prepare_oracle_build_candidate(self.root)
        self.source.write_text("def calculate(x=1):\n    return x + 2\n")
        with self.assertRaises(owner.OracleBuildBlockedError):
            self.finalize(candidate)
        self.assertEqual(certificate.read_bytes(), before)
        certificate.unlink()
        self.source.write_text("def calculate(x=1):\n    return x + 1\n")
        self.snapshot.write_bytes(b"corrupt")
        with self.assertRaises(owner.OracleBuildBlockedError):
            owner.prepare_oracle_build_candidate(self.root)

    def test_revision_change_keeps_only_audited_unaffected_origin_components(self):
        self.revisions = {"replay": 2, "observation": 1}
        self.review["current"]["component_revisions"] = dict(self.revisions)
        self.review.update(kind="revision_change", revision_changes={"replay": {"from": 1, "to": 2}},
                           decision_kind="correctness_fix", decision_reference="existing execution contract")
        self.review["proofs"][0]["kind"] = "independent_component_parity"
        origin = self.root / "origin.json"
        origin.write_text('{"observation": "unchanged"}')
        record = {"sha256": sha256(origin.read_bytes()).hexdigest(), "component_revisions": {"observation": 1}}
        for relevant, accepted in (({"observation": 1}, True), ({"observation": 1, "replay": 1}, False), ({}, False)):
            with self.subTest(relevant=relevant):
                record["component_revisions"] = relevant
                origins = {"origin.json": dict(record)}
                self.review["legacy_origins"] = origins
                self.review["legacy_audit_receipt"] = self.write_json("scoped_audit.json", {
                    "status": "PASS", "source_digest": self.review["baseline"]["source_digest"],
                    "snapshot_sha256": self.review["baseline"]["snapshot_sha256"],
                    "process_token": "independent-baseline", "validated_origins": origins,
                })
                self.write_review()
                if accepted:
                    self.assertEqual(owner.prepare_oracle_build_candidate(self.root)["legacy_origins"], origins)
                else:
                    with self.assertRaises(owner.OracleBuildBlockedError):
                        owner.prepare_oracle_build_candidate(self.root)


if __name__ == "__main__":
    unittest.main()

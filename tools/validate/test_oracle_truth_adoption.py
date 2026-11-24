"""Truth inspection/adoption preserves original arrays and refuses races."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from core.artifact_inspection import InspectionStatus
from services.audit import daily_trade40_oracle as daily
from services.audit.oracle_truth_references import OracleTruthReferences, source_context
from tools.validate.oracle_fixture_build import OracleFixtureTestCase


class OracleTruthAdoptionTests(OracleFixtureTestCase):
    def _fixture(self, root):
        from services.research.target_oracle_application import get_formal_oracle_definition
        from services.research.oracle_artifact_contract import source_calculation_contract
        definition = get_formal_oracle_definition("breakout_quality")
        profile = definition.source["pure_mfe40_provider"]
        source = source_calculation_contract(recipe="daily_trade_path_net_pnl", profile_id=profile,
            horizon=None, model_architecture=definition.source["model_architecture"])
        scores = {("2330", "2021-01-04"): 0.0, ("2317", "2021-01-04"): 1.0}
        execution = {
            "dataset": "full", "price_sha256": "a" * 64, "param_sha256": "b" * 64,
            "seed": 42, "start_date": "2021-01-01", "end_date": "2021-01-05",
            "research_cutoff": "2026-03-02", "dataset_inputs": {"data": "c" * 64},
            "source_contract": source,
            "replay_calculation_revision": 1, "replay_implementation_sha256": "e" * 64,
            "score_contract": {"score_start_date": "2021-01-01"},
            "replay_policy": {"filter_id": "breakout_quality_v1", "fixed_risk": 0.01,
                "max_position_cap_pct": 0.3, "broker_accounting": {"fee": "0.001425"},
                "execution_defaults": {"initial_capital": "1000000.0"}},
        }
        identity = {
            "schema": daily._TRUTH_CACHE_SCHEMA, "target_id": "daily_full_net_pnl",
            "dataset": execution["dataset"], "price_sha256": execution["price_sha256"],
            "param_sha256": execution["param_sha256"], "seed": execution["seed"],
            "filter_id": "breakout_quality_v1", "profile_id": profile,
            "evaluation_start": execution["start_date"], "evaluation_end": execution["end_date"],
            "research_cutoff": execution["research_cutoff"], "score_start": "2021-01-01",
            "horizon_sessions": None, "truth_source_sha256": "f" * 64,
            "dataset_summary_sha256": "1" * 64,
            "feature_eligible_keys_sha256": sha256(pd.DataFrame(sorted(scores),
                columns=["ticker", "date"]).to_csv(index=False).encode("utf-8")).hexdigest(),
        }
        evidence = {
            "target_id": identity["target_id"], "profile_id": identity["profile_id"],
            "truth_source_sha256": identity["truth_source_sha256"],
            "dataset_summary_sha256": identity["dataset_summary_sha256"],
            "research_cutoff": identity["research_cutoff"], "score_start_date": identity["score_start"],
            "teacher": {"seed": 42, "param_identity_sha256": "b" * 64},
            "mature_eligible_rows": len(scores), "rank_sha256": daily._ranked_score_sha(scores),
            "raw_label_sha256": "2" * 64, "calendar_sha256": "3" * 64,
        }
        references = OracleTruthReferences(root / "truth_cache", project_root=root)
        origin = references.origin_path(identity)
        daily._truth_cache_write(origin, identity, scores, evidence)
        return references, execution, identity, scores, evidence, origin

    @staticmethod
    def _snapshot(root):
        return {path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in root.rglob("*") if path.is_file()}

    def test_inspection_is_read_only_and_adoption_preserves_original_truth(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            refs, execution, identity, scores, evidence, origin = self._fixture(root)
            before = self._snapshot(root)
            inspected = refs.inspect_existing(execution, evidence=evidence,
                producer_fingerprint=evidence["truth_source_sha256"])
            self.assertEqual(inspected.status, InspectionStatus.READY)
            self.assertEqual(before, self._snapshot(root))
            refs.publish_existing(execution, evidence=evidence, inspection=inspected)
            self.assertEqual(before[origin.relative_to(root)], (origin.read_bytes(), origin.stat().st_mtime_ns))
            reference = refs.read(source_context(execution))
            self.assertEqual(reference["origin_identity"], identity)
            self.assertEqual(reference["origin_sha256"], inspected.value.pins[0].sha256)
            self.assertEqual(refs.load(execution)[0], scores)
            # A warm inspection checks sealed bytes/membership without re-reading arrays.
            with patch.object(daily, "_truth_cache_read", side_effect=AssertionError("Unexpected array decode")):
                self.assertTrue(refs.inspect_existing(execution, evidence=evidence).ready)

    def test_missing_npz_is_missing_truth_even_when_reference_survives(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            refs, execution, _identity, _scores, evidence, origin = self._fixture(root)
            refs.publish_existing(execution, evidence=evidence,
                producer_fingerprint=evidence["truth_source_sha256"])
            origin.unlink()
            before = self._snapshot(root)
            self.assertEqual(refs.inspect_existing(execution, evidence=evidence).status, InspectionStatus.MISSING)
            self.assertEqual(refs.publish_existing(execution, evidence=evidence).status, InspectionStatus.MISSING)
            self.assertEqual(before, self._snapshot(root))

    def test_changed_original_after_inspection_cannot_publish_reference(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            refs, execution, _identity, _scores, evidence, origin = self._fixture(root)
            inspected = refs.inspect_existing(execution, evidence=evidence,
                producer_fingerprint=evidence["truth_source_sha256"])
            self.assertTrue(inspected.ready)
            origin.write_bytes(origin.read_bytes() + b"changed_after_inspection")
            with self.assertRaisesRegex(ValueError, "changed after inspection"):
                refs.publish_existing(execution, evidence=evidence, inspection=inspected)
            self.assertFalse(refs.root.exists())

    def test_replay_only_changes_preserve_truth_but_math_teacher_inputs_do_not(self):
        with TemporaryDirectory() as folder:
            refs, execution, _identity, _scores, evidence, _origin = self._fixture(Path(folder))
            refs.publish_existing(execution, evidence=evidence,
                producer_fingerprint=evidence["truth_source_sha256"])
            changed = deepcopy(execution)
            changed["replay_implementation_sha256"] = "9" * 64
            changed["replay_calculation_revision"] += 1
            changed["ranking_options"] = {"selector": "different_recipient"}
            self.assertEqual(source_context(changed), source_context(execution))
            self.assertIsNotNone(refs.read(source_context(changed)))
            for field in ("param_sha256", "price_sha256"):
                changed = {**execution, field: "8" * 64}
                self.assertIsNone(refs.read(source_context(changed)))
            changed = deepcopy(execution)
            changed["source_contract"]["calculation_revision"] += 1
            self.assertIsNone(refs.read(source_context(changed)))
            changed = deepcopy(execution)
            changed["replay_policy"]["broker_accounting"]["fee"] = "0.9"
            self.assertIsNone(refs.read(source_context(changed)))

    def test_first_adoption_requires_attested_source_and_canonical_calculation(self):
        with TemporaryDirectory() as folder:
            refs, execution, _identity, _scores, evidence, _origin = self._fixture(Path(folder))
            before = self._snapshot(Path(folder))
            for expected in (None, "0" * 64):
                result = refs.inspect_existing(execution, evidence=evidence,
                    producer_fingerprint=expected)
                self.assertEqual(result.status, InspectionStatus.INCOMPATIBLE)
                self.assertIn("producer is not attested", result.reason)
            changed = deepcopy(execution)
            changed["source_contract"]["calculation_revision"] += 1
            result = refs.inspect_existing(changed, evidence=evidence,
                producer_fingerprint=evidence["truth_source_sha256"])
            self.assertEqual(result.status, InspectionStatus.INCOMPATIBLE)
            self.assertIn("calculation contract changed", result.reason)
            self.assertEqual(before, self._snapshot(Path(folder)))

    def test_interrupted_reference_publish_preserves_origin_and_can_retry(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            refs, execution, _identity, _scores, evidence, origin = self._fixture(root)
            before = (origin.read_bytes(), origin.stat().st_mtime_ns)
            inspected = refs.inspect_existing(execution, evidence=evidence,
                producer_fingerprint=evidence["truth_source_sha256"])
            with patch("services.audit.oracle_truth_references.atomic_replace_with_retry",
                       side_effect=OSError("injected publication interruption")):
                with self.assertRaisesRegex(OSError, "publication interruption"):
                    refs.publish_existing(execution, inspection=inspected)
            self.assertEqual(before, (origin.read_bytes(), origin.stat().st_mtime_ns))
            self.assertIsNone(refs.read(source_context(execution)))
            self.assertEqual(list(refs.root.rglob("*.tmp")), [])
            refs.publish_existing(execution, inspection=inspected)
            self.assertIsNotNone(refs.read(source_context(execution)))

    def test_reviewed_legacy_npz_pin_requires_exact_bytes_identity_and_result(self):
        from core.file_integrity import canonical_json_sha256
        with TemporaryDirectory() as folder:
            root = Path(folder)
            refs, execution, identity, _scores, evidence, origin = self._fixture(root)
            proof = {"sha256": sha256(origin.read_bytes()).hexdigest(),
                     "identity_sha256": canonical_json_sha256(identity),
                     "result_contract_sha256": canonical_json_sha256(source_context(execution))}
            relative = origin.relative_to(root).as_posix()
            before = self._snapshot(root)
            for mutated in ("sha256", "identity_sha256", "result_contract_sha256", None):
                admitted = dict(proof)
                if mutated:
                    admitted[mutated] = "0" * 64
                with patch("services.research.oracle_build_attestation.require_oracle_build_attestation",
                           return_value={"legacy_origins": {relative: admitted}}):
                    result = refs.inspect_existing(execution, evidence=evidence,
                        producer_fingerprint="9" * 64)
                self.assertEqual(result.ready, mutated is None, mutated)
            self.assertEqual(before, self._snapshot(root))

    def test_target_adopts_schema1_after_approval_and_preserves_original_receipt(self):
        import json
        from core.file_integrity import canonical_json_sha256
        from services.audit import target_oracle_romd as oracle
        from services.research.oracle_artifact_contract import OracleExecutionIndex
        from tools.validate.synthetic_target_oracle_cases import OracleSingleOwnerExecutionCases
        with TemporaryDirectory() as folder:
            root = Path(folder)
            case = OracleSingleOwnerExecutionCases()
            fixture = case._fixture(root)
            contexts = []
            original = oracle.execution_context
            def capture(**kwargs):
                context = original(**kwargs)
                contexts.append(context)
                return context
            with case._context(root, fixture), patch.object(oracle, "execution_context", side_effect=capture):
                cancelled = oracle.build_report(fixture[0], project_root=root, quiet=True,
                    preflight_approval=lambda _view: False)
            self.assertEqual(cancelled["status"], "CANCELLED")
            store = oracle.OracleReplayStore(root, fixture[0].output_subdir)
            index = OracleExecutionIndex(store)
            context = next(item for item in contexts if item["target"] == "R40")
            identity, = store.existing_completed_identities(seed=fixture[2], target="R40")
            observation = store.read(identity)
            old_payload = {"schema": 1, "context": context, "origin_identity": identity,
                           "observation_sha256": canonical_json_sha256(observation)}
            old_path = index._legacy_path(context)
            old_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.write_text(json.dumps({**old_payload,
                "reference_sha256": canonical_json_sha256(old_payload)}), encoding="utf-8")
            preserved = self._snapshot(root)
            with case._context(root, fixture):
                oracle.build_report(fixture[0], project_root=root, quiet=True,
                    preflight_approval=lambda _view: False)
                self.assertEqual(preserved, self._snapshot(root))
                oracle.build_report(fixture[0], project_root=root, quiet=True,
                    preflight_approval=lambda _view: True)
            self.assertEqual(json.loads(index._path(context).read_text())["schema"], 2)
            self.assertFalse(index.inspect(context).adoption_required)
            for relative, pinned in preserved.items():
                path = root / relative
                self.assertEqual(pinned, (path.read_bytes(), path.stat().st_mtime_ns), str(relative))

    def test_target_rejects_origin_byte_change_after_preflight_approval(self):
        from services.audit import target_oracle_romd as oracle
        from services.research.oracle_artifact_contract import OracleExecutionIndex
        from tools.validate.synthetic_target_oracle_cases import OracleSingleOwnerExecutionCases
        with TemporaryDirectory() as folder:
            root = Path(folder)
            case = OracleSingleOwnerExecutionCases()
            fixture = case._fixture(root)
            store = oracle.OracleReplayStore(root, fixture[0].output_subdir)
            identity, = store.existing_completed_identities(seed=fixture[2], target="R40")
            origin = store._path(identity)
            observation = store.read(identity)
            def approve_after_origin_changes(_view):
                # Its canonical JSON signatures and numerical observation stay
                # valid; the approved physical-origin pin must still reject it.
                origin.write_bytes(origin.read_bytes() + b"\n")
                return True
            with case._context(root, fixture):
                with self.assertRaisesRegex(oracle.OracleBlockedError, "changed after inspection"):
                    oracle.build_report(fixture[0], project_root=root, quiet=True,
                        preflight_approval=approve_after_origin_changes)
            self.assertEqual(store.read(identity), observation)
            index = OracleExecutionIndex(store)
            self.assertEqual(list(index.root.rglob("R40_*.json")), [])


class OracleRuntimeBuildGateTests(TestCase):
    def test_missing_certificate_blocks_before_parameter_or_portfolio_work(self):
        from services.audit.target_oracle_romd import OracleBlockedError, require_oracle_execution_build
        from services.research.target_oracle_application import ensure_authorized_oracle_parameters
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(OracleBlockedError, "IMPLEMENTATION_UNVERIFIED"):
                require_oracle_execution_build(root)
            with patch("services.optimizer.strategy_param_service."
                       "ensure_fixed_multiseed_robustness_strategy_parameter_artifact") as producer:
                with self.assertRaisesRegex(OracleBlockedError, "IMPLEMENTATION_UNVERIFIED"):
                    ensure_authorized_oracle_parameters(project_root=root, definition=None, approved_plan={})
            producer.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

"""Disabled candidate proof: exact source reconstruction and unknown-edit rejection."""
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

from core import research_execution_equivalence as proof
from core.execution_source_dependencies import execution_source_inventory
from services.research import oracle_artifact_contract as contracts

ROOT = Path(__file__).resolve().parents[2]


class OptimizerExecutionEquivalenceTests(unittest.TestCase):
    def tearDown(self):
        contracts.replay_implementation_sha256.cache_clear()
        contracts.portfolio_candidate_execution_sha256.cache_clear()

    def _source_parts(self):
        owners = execution_source_inventory('replay')
        parts = {name: contracts._source_owner_attestation(ROOT / name)[1] for name in owners}
        parts['@execution_sources'] = contracts.canonical_json_sha256(list(owners))
        parts['core/ranking_policy.py'] = proof.reviewed_ranking_source_sha256(
            parts['core/ranking_policy.py'], project_root=ROOT)
        return parts

    def test_disabled_candidate_cannot_authorize_any_source(self):
        with patch.object(proof, '_OPTIMIZER_EXECUTION_REVIEW_ENABLED', False):
            for name, pins in proof._OPTIMIZER_EXECUTION_PHYSICAL_PINS.items():
                for current in pins:
                    self.assertEqual(proof._reviewed_optimizer_execution_sha256(name, current), current)
            self.assertEqual(proof.reviewed_replay_predecessor_sources({}, project_root=ROOT), {})

    def test_known_physical_sources_and_unknown_mutations(self):
        with patch.object(proof, '_OPTIMIZER_EXECUTION_REVIEW_ENABLED', True):
            for name, pins in proof._OPTIMIZER_EXECUTION_PHYSICAL_PINS.items():
                raw = (ROOT / name).read_bytes()
                current = sha256(raw).hexdigest()
                self.assertIn(current, pins, name)
                self.assertEqual(proof.reviewed_execution_source_sha256(name, current), pins[current])
                mutation = sha256(raw + b'\n_UNKNOWN_EXECUTION_EDIT = True\n').hexdigest()
                self.assertNotEqual(current, mutation)
                self.assertEqual(proof.reviewed_execution_source_sha256(name, mutation), mutation)

    def test_complete_group_reconstructs_formal_zip_predecessor(self):
        with patch.object(proof, '_OPTIMIZER_EXECUTION_REVIEW_ENABLED', True):
            restored = proof.reviewed_replay_predecessor_sources(self._source_parts(), project_root=ROOT)
        self.assertEqual(set(restored), set(proof._OPTIMIZER_LEGACY_SOURCE_RECONSTRUCTIONS))
        for name, (source, syntax) in restored.items():
            self.assertEqual(sha256(source.encode()).hexdigest(),
                             proof._OPTIMIZER_LEGACY_SOURCE_RECONSTRUCTIONS[name]['predecessor_lf_sha256'])
            self.assertEqual(contracts._source_content_attestation(source, is_python=True)[1], syntax)

    def test_unknown_owner_helper_or_dependency_graph_is_live(self):
        with patch.object(proof, '_OPTIMIZER_EXECUTION_REVIEW_ENABLED', True):
            parts = self._source_parts()
            self.assertTrue(proof.reviewed_replay_predecessor_sources(parts, project_root=ROOT))
            for name in ('core/price_utils.py', 'core/portfolio_entry_selection_common.py', '@execution_sources'):
                altered = {**parts, name: 'f' * 64}
                self.assertNotEqual(parts, altered)
                self.assertEqual(proof.reviewed_replay_predecessor_sources(altered, project_root=ROOT), {})
            self.assertEqual(proof.reviewed_replay_predecessor_sources(
                {**parts, 'core/new_helper.py': 'f' * 64}, project_root=ROOT), {})
            missing = dict(parts); missing.pop('core/price_utils.py')
            self.assertEqual(proof.reviewed_replay_predecessor_sources(missing, project_root=ROOT), {})
            self.assertEqual(proof.reviewed_replay_predecessor_sources(parts, project_root=ROOT.parent), {})

    def test_physical_unknown_edit_cannot_inherit_known_syntax(self):
        original = Path.read_bytes
        path = ROOT / 'core/price_utils.py'
        raw = original(path)
        mutation = raw + b'\n_UNKNOWN_PHYSICAL_EDIT = True\n'
        self.assertNotEqual(raw, mutation)
        with patch.object(proof, '_OPTIMIZER_EXECUTION_REVIEW_ENABLED', True):
            parts = self._source_parts()
            with patch.object(Path, 'read_bytes', lambda p: mutation if p == path else original(p)):
                self.assertEqual(proof.reviewed_replay_predecessor_sources(parts, project_root=ROOT), {})

    def test_reconstruction_hash_mismatch_fails_closed(self):
        name = 'core/price_utils.py'
        corrupt = {**proof._OPTIMIZER_LEGACY_SOURCE_RECONSTRUCTIONS[name], 'predecessor_lf_sha256': '0' * 64}
        with (patch.object(proof, '_OPTIMIZER_EXECUTION_REVIEW_ENABLED', True),
              patch.dict(proof._OPTIMIZER_LEGACY_SOURCE_RECONSTRUCTIONS, {name: corrupt})):
            self.assertEqual(proof.reviewed_replay_predecessor_sources(self._source_parts(), project_root=ROOT), {})


if __name__ == '__main__':
    unittest.main()

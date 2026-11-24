"""Reviewed optimizer proof: exact source reconstruction and unknown-edit rejection."""
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
                resource = proof._independent_queue_resource_predecessor(name, raw)
                syntax_owner = proof._calculation_syntax_owner_certificate(name)
                if syntax_owner is not None:
                    resource = syntax_owner[1]
                predecessor = current if resource is None else resource
                self.assertIn(predecessor, pins, name)
                self.assertEqual(proof.reviewed_execution_source_sha256(name, current), pins[predecessor])
                mutation = sha256(raw + b'\n_UNKNOWN_EXECUTION_EDIT = True\n').hexdigest()
                self.assertNotEqual(current, mutation)
                self.assertEqual(proof.reviewed_execution_source_sha256(name, mutation), mutation)

    def test_queue_capacity_is_literal_only_and_cannot_hide_unknown_config(self):
        name = 'config/training_performance_policy.py'
        raw = (ROOT / name).read_bytes()
        import ast
        tree = ast.parse(raw)
        knob = 'OPTIMIZER_INDEPENDENT_SEED_FOLD_GLOBAL_WORKERS'
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == knob
                                  for target in node.targets))
        declaration = ast.get_source_segment(raw.decode(), assignment).encode()
        expected = proof._independent_queue_resource_predecessor(name, raw)
        self.assertIsNotNone(expected)
        for value, admitted in ((1, True), (64, True), ('int(6)', False),
                                ('True', False), (0, False), (65, False)):
            changed = raw.replace(declaration, f'{knob} = {value}'.encode())
            observed = proof._independent_queue_resource_predecessor(name, changed)
            self.assertEqual(observed, expected if admitted else None)
        for tail in (b'\n_UNKNOWN_TRAINING_SETTING = 1\n', b'\n' + declaration + b'\n'):
            self.assertIsNone(proof._independent_queue_resource_predecessor(name, raw + tail))

    def test_queue_resolver_requires_exact_body_and_complete_predecessor(self):
        name = 'core/training_performance.py'
        raw = (ROOT / name).read_bytes()
        self.assertIsNotNone(proof._independent_queue_resource_predecessor(name, raw))
        resolver = b'def resolve_optimizer_independent_seed_fold_global_workers(environ=None) -> int:'
        changed = raw.replace(resolver, resolver + b'\n    return 2')
        self.assertIsNone(proof._independent_queue_resource_predecessor(name, changed))
        self.assertIsNone(proof._independent_queue_resource_predecessor(
            name, raw + b'\n_UNKNOWN_RESOURCE_CONSUMER = True\n'))
        self.assertIsNone(proof._independent_queue_resource_predecessor(
            name, raw.replace(b'"OPTIMIZER_ROLLING_FOLD_WORKERS"', b'"UNKNOWN_WORKER_OVERRIDE"')))

    def test_unknown_resource_source_cannot_inherit_syntax_group(self):
        name = 'core/training_performance.py'
        path = ROOT / name
        raw = path.read_bytes()
        mutate = raw + b'\n_UNKNOWN_RESOURCE_CONSUMER = True\n'
        original = Path.read_bytes
        with patch.object(Path, 'read_bytes', lambda p: mutate if p == path else original(p)):
            self.assertEqual(proof.reviewed_execution_source_sha256(name, sha256(mutate).hexdigest()),
                             sha256(mutate).hexdigest())
            self.assertNotEqual(proof._reviewed_source_group('portfolio_candidates')['current'][name],
                                contracts._source_content_attestation(mutate, is_python=True)[1])

    def test_calculation_owner_extraction_requires_entire_helper_and_provider(self):
        owner = (ROOT / 'services/research/oracle_artifact_contract.py').read_bytes()
        helper = (ROOT / 'core/source_calculation_syntax.py').read_bytes()
        self.assertIsNotNone(proof._calculation_syntax_owner_predecessor(owner, helper))
        for mutation in (
            helper + b'\n_UNKNOWN_HELPER_GLOBAL = True\n',
            helper.replace(b'return payload, canonical_json_sha256(payload)', b'return payload, "0" * 64'),
        ):
            self.assertIsNone(proof._calculation_syntax_owner_predecessor(owner, mutation))
        self.assertIsNone(proof._calculation_syntax_owner_predecessor(
            owner + b'\n_UNKNOWN_SOURCE_SCOPE = True\n', helper))

    def test_calculation_source_aliases_share_one_low_level_owner(self):
        from core import source_calculation_syntax as owner
        self.assertIs(contracts._CalculationSyntax, owner._CalculationSyntax)
        self.assertIs(contracts._syntax_payload, owner._syntax_payload)

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

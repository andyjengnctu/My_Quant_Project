"""Source scopes follow consumed capability metadata and executable dependencies."""
import ast
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.execution_source_dependencies import execution_source_inventory, project_ranking_policy_source
from core.portfolio_entry_policy import DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS
from core.ranking_policy import RANKING_POLICY_SPECS
from core.research_execution_equivalence import ORACLE_STRATEGY_OBSERVATION_SCOPE
from services.research import oracle_artifact_contract as replay

ROOT = Path(__file__).resolve().parents[2]


class ExecutionSourceDependencyTests(unittest.TestCase):
    def tearDown(self):
        replay.replay_implementation_sha256.cache_clear()
        replay.portfolio_candidate_execution_sha256.cache_clear()

    def _digests(self, policy=None, observation_scope=None):
        self.tearDown()
        return (replay.replay_implementation_sha256(ranking_policy=policy, observation_scope=observation_scope), *(
            replay.portfolio_candidate_execution_sha256(daily_execution_contract=mode, ranking_policy=policy,
                                                        observation_scope=observation_scope)
            for mode in (DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS)))

    def test_shared_numerical_dependencies_are_registered_in_replay_and_truth(self):
        for scope in ("replay", "portfolio_candidates", "mintrade_truth"):
            owners = execution_source_inventory(scope)
            self.assertEqual(len(owners), len(set(owners)))
            for owner in ("core/native_optimizer_hotpath.py", "core/native_optimizer_hotpath.c",
                          "config/broker_accounting.py", "config/execution_policy.py"):
                self.assertIn(owner, owners)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            execution_source_inventory("unregistered")

    def test_additive_unused_ranking_descriptor_does_not_change_any_scope(self):
        path = ROOT / "core/ranking_policy.py"
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        declaration = next(n for n in tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "RANKING_POLICY_SPECS" for t in n.targets))
        declaration.value.elts.append(ast.parse("RankingPolicySpec(policy_id='unused-capability-probe', prefix_key='score')", mode="eval").body)
        changed = ast.unparse(tree)
        original = Path.read_text
        for selected in (None, RANKING_POLICY_SPECS[0].policy_id, RANKING_POLICY_SPECS[-1].policy_id):
            before = self._digests(selected)
            with patch.object(Path, "read_text", lambda p, *a, **kw: changed if p == path else original(p, *a, **kw)):
                self.assertEqual(before, self._digests(selected))

    def test_selected_descriptor_and_primitive_changes_invalidate(self):
        path = ROOT / "core/ranking_policy.py"
        original = Path.read_text
        source = original(path, encoding="utf-8-sig")
        changed = source.replace("prefix_key='score',", "prefix_key='baseline',", 1)
        self.assertNotEqual(source, changed)
        policy = RANKING_POLICY_SPECS[0].policy_id
        before = self._digests(policy)
        with patch.object(Path, "read_text", lambda p, *a, **kw: changed if p == path else original(p, *a, **kw)):
            self.assertTrue(all(a != b for a, b in zip(before, self._digests(policy))))
        path = ROOT / "core/portfolio_entry_selection_common.py"
        changed = original(path, encoding="utf-8-sig") + "\n_SOURCE_DEPENDENCY_MUTATION = True\n"
        with patch.object(Path, "read_text", lambda p, *a, **kw: changed if p == path else original(p, *a, **kw)):
            self.assertTrue(all(a != b for a, b in zip(before, self._digests(policy))))

    def test_dynamic_unused_registration_cannot_hide_import_time_execution(self):
        path = ROOT / "core/ranking_policy.py"
        read = Path.read_text
        tree = ast.parse(read(path, encoding="utf-8-sig"))
        declaration = next(n for n in tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "RANKING_POLICY_SPECS" for t in n.targets))
        declaration.value.elts.append(ast.parse(
            "RankingPolicySpec(policy_id='dynamic-probe', prefix_key=mutate_existing_policy())", mode="eval").body)
        changed = ast.unparse(tree)
        policy = RANKING_POLICY_SPECS[0].policy_id
        before = self._digests(policy)
        with patch.object(Path, "read_text", lambda p, *a, **kw: changed if p == path else read(p, *a, **kw)):
            self.assertTrue(all(a != b for a, b in zip(before, self._digests(policy))))

    def test_unused_policy_dictionary_value_cannot_hide_import_time_execution(self):
        path = ROOT / "core/ranking_policy.py"
        read = Path.read_text
        tree = ast.parse(read(path, encoding="utf-8-sig"))
        policy = RANKING_POLICY_SPECS[0].policy_id
        unused = RANKING_POLICY_SPECS[-1].policy_id
        before = self._digests(policy)
        tree.body.append(ast.parse(
            f"UNUSED_PREFIX_PROBE = {{{unused!r}: mutate_existing_policy()}}" ).body[0])
        changed = ast.unparse(tree)
        with patch.object(Path, "read_text", lambda p, *a, **kw: changed if p == path else read(p, *a, **kw)):
            self.assertTrue(all(a != b for a, b in zip(before, self._digests(policy))))

    def test_native_and_accounting_configuration_changes_invalidate(self):
        original_bytes, original_text = Path.read_bytes, Path.read_text
        before = self._digests(RANKING_POLICY_SPECS[0].policy_id)
        native = ROOT / "core/native_optimizer_hotpath.c"
        changed = original_bytes(native).replace(b"return PyLong_FromLongLong((long long)qty);",
                                                b"return PyLong_FromLongLong((long long)(qty + 1000));")
        self.assertNotEqual(original_bytes(native), changed)
        with patch.object(Path, "read_bytes", lambda p: changed if p == native else original_bytes(p)):
            self.assertTrue(all(a != b for a, b in zip(before, self._digests(RANKING_POLICY_SPECS[0].policy_id))))
        config = ROOT / "config/broker_accounting.py"
        changed_text = original_text(config, encoding="utf-8-sig") + "\n_BROKER_DEPENDENCY_MUTATION = True\n"
        with patch.object(Path, "read_text", lambda p, *a, **kw: changed_text if p == config else original_text(p, *a, **kw)):
            self.assertTrue(all(a != b for a, b in zip(before, self._digests(RANKING_POLICY_SPECS[0].policy_id))))

    def test_registry_dependency_removal_and_missing_helper_fail_closed(self):
        from core import execution_source_dependencies as registry
        purposes = (None, ORACLE_STRATEGY_OBSERVATION_SCOPE)
        before = {purpose: self._digests(observation_scope=purpose) for purpose in purposes}
        scopes = dict(registry.SOURCE_CAPABILITIES)
        runtime = scopes["numerical_runtime"]
        scopes["numerical_runtime"] = replace(runtime, owners=tuple(
            name for name in runtime.owners if name != "core/native_optimizer_hotpath.c"))
        with patch.object(registry, "SOURCE_CAPABILITIES", scopes):
            for purpose in purposes:
                self.assertTrue(all(a != b for a, b in zip(before[purpose], self._digests(observation_scope=purpose))))
        scopes["numerical_runtime"] = replace(runtime, dependencies=("missing-helper-capability",))
        with patch.object(registry, "SOURCE_CAPABILITIES", scopes):
            with self.assertRaisesRegex(ValueError, "Unknown execution source capability"):
                self._digests()
        scopes["numerical_runtime"] = replace(runtime, owners=(*runtime.owners, "core/unpublished_numeric_helper.py"))
        with patch.object(registry, "SOURCE_CAPABILITIES", scopes):
            with self.assertRaises(FileNotFoundError):
                self._digests()

    def test_new_declared_helper_is_inherited_by_all_consuming_scopes(self):
        from core import execution_source_dependencies as registry
        scopes = dict(registry.SOURCE_CAPABILITIES)
        helper = ROOT / "core/isolated_numeric_helper.py"
        scopes["isolated_helper"] = registry.ExecutionSourceCapability(("core/isolated_numeric_helper.py",))
        scopes["numerical_runtime"] = replace(scopes["numerical_runtime"], dependencies=("isolated_helper",))
        read_text, is_file = Path.read_text, Path.is_file
        with patch.object(registry, "SOURCE_CAPABILITIES", scopes), \
             patch.object(Path, "is_file", lambda p: True if p == helper else is_file(p)):
            self.assertIn("core/isolated_numeric_helper.py", execution_source_inventory("mintrade_truth"))
            with patch.object(Path, "read_text", lambda p, *a, **kw: "def quantity(x): return x\n" if p == helper else read_text(p, *a, **kw)):
                before = {purpose: self._digests(observation_scope=purpose)
                          for purpose in (None, ORACLE_STRATEGY_OBSERVATION_SCOPE)}
            with patch.object(Path, "read_text", lambda p, *a, **kw: "def quantity(x): return x + 1\n" if p == helper else read_text(p, *a, **kw)):
                for purpose, original in before.items():
                    self.assertTrue(all(a != b for a, b in zip(original, self._digests(observation_scope=purpose))))

    def test_every_registered_replay_owner_remains_effective_after_migration(self):
        read_text, read_bytes = Path.read_text, Path.read_bytes
        policy = RANKING_POLICY_SPECS[0].policy_id
        purposes = (None, ORACLE_STRATEGY_OBSERVATION_SCOPE)
        before = {purpose: self._digests(policy, purpose) for purpose in purposes}
        for relative in execution_source_inventory("replay"):
            target = ROOT / relative
            with self.subTest(owner=relative):
                if target.suffix == ".py":
                    changed = read_text(target, encoding="utf-8-sig") + "\n_SOURCE_OWNER_MUTATION = True\n"
                    scope = patch.object(Path, "read_text", lambda p, *a, **kw: changed if p == target else read_text(p, *a, **kw))
                else:
                    changed = read_bytes(target) + b"\n#define SOURCE_OWNER_MUTATION 1\n"
                    scope = patch.object(Path, "read_bytes", lambda p: changed if p == target else read_bytes(p))
                with scope:
                    for purpose in purposes:
                        with self.subTest(observation_scope=purpose):
                            self.assertTrue(all(a != b for a, b in zip(before[purpose], self._digests(policy, purpose))))

    def test_oracle_observation_proof_requires_unchanged_consumers_and_bindings(self):
        from core.research_execution_equivalence import oracle_observation_consumer_sources
        before = self._digests(observation_scope=ORACLE_STRATEGY_OBSERVATION_SCOPE)
        full_result = self._digests()
        self.assertTrue(all(a != b for a, b in zip(before, full_result)))
        read = Path.read_text
        for relative in oracle_observation_consumer_sources():
            path = ROOT / relative
            source = read(path, encoding="utf-8-sig")
            for suffix in ("\n_OBSERVATION_BINDING = 1\n", "\n_OBSERVATION_BINDING = 2\n"):
                with self.subTest(owner=relative, revision=suffix):
                    with patch.object(Path, "read_text", lambda p, *a, **kw: source + suffix if p == path else read(p, *a, **kw)):
                        changed = self._digests(observation_scope=ORACLE_STRATEGY_OBSERVATION_SCOPE)
                        self.assertTrue(all(a != b for a, b in zip(before, changed)))
                        self.assertEqual(full_result, self._digests())
                    if suffix.endswith("2\n"):
                        self.assertTrue(all(a != b for a, b in zip(first_changed, changed)))
                    first_changed = changed
        for function, kwargs in (
            (replay.replay_implementation_sha256, {}),
            (replay.portfolio_candidate_execution_sha256, {"daily_execution_contract": DAILY_EXECUTION_FULL_FILTERS}),
        ):
            with self.assertRaisesRegex(ValueError, "Unknown Portfolio observation scope"):
                function(observation_scope="all_reports_are_equal", **kwargs)

    def test_named_daily_filter_projection_tracks_delegated_execution_contract(self):
        relative = "core/daily_common_entry_filters.py"
        before = replay.calculation_source(relative, "daily_common_filter_signals")
        read = Path.read_text
        for name in (relative, "core/portfolio_entry_policy.py"):
            path = ROOT / name
            source = read(path, encoding="utf-8-sig")
            altered = []
            for value in (1, 2):
                changed = source + f"\n_DAILY_EXECUTION_MUTATION = {value}\n"
                with patch.object(Path, "read_text", lambda p, *a, **kw: changed if p == path else read(p, *a, **kw)):
                    altered.append(replay.calculation_source(relative, "daily_common_filter_signals"))
            self.assertEqual(3, len({before, *altered}), name)

    def test_new_policy_can_reuse_different_registered_selector_without_source_list_edit(self):
        from dataclasses import asdict
        from core.ranking_policy import RankingPolicySpec
        from core.portfolio_entry_selection import _SELECTOR_HANDLERS
        base = next(spec for spec in RANKING_POLICY_SPECS if spec.selector_key == "binary_greedy")
        new = RankingPolicySpec(**{**asdict(base), "policy_id": "new-combination-probe"})
        self.assertIn(new.selector_key, _SELECTOR_HANDLERS)
        before_owners = execution_source_inventory("portfolio_candidates")
        path = ROOT / "core/ranking_policy.py"
        original = Path.read_text
        tree = ast.parse(original(path, encoding="utf-8-sig"))
        declaration = next(n for n in tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "RANKING_POLICY_SPECS" for t in n.targets))
        declaration.value.elts.append(ast.parse(
            "RankingPolicySpec(policy_id='new-combination-probe', resource_aware=True, selector_key='binary_greedy')",
            mode="eval").body)
        changed = ast.unparse(tree)
        before = self._digests(base.policy_id)
        with patch.object(Path, "read_text", lambda p, *a, **kw: changed if p == path else original(p, *a, **kw)):
            self.assertEqual(before, self._digests(base.policy_id))
            self.assertNotEqual(before, self._digests(new.policy_id))
        self.assertEqual(before_owners, execution_source_inventory("portfolio_candidates"))

    def test_external_fixture_cannot_inherit_local_projection_certificate(self):
        from core.research_execution_equivalence import reviewed_ranking_source_sha256
        with TemporaryDirectory() as root:
            for value in ("0" * 64, "a" * 64):
                self.assertEqual(value, reviewed_ranking_source_sha256(value, project_root=root))

    def test_full_optional_primitive_changes_only_consuming_execution_scope(self):
        before = self._digests(RANKING_POLICY_SPECS[0].policy_id)
        path = ROOT / "core/daily_common_entry_filters.py"
        read = Path.read_text
        source = read(path, encoding="utf-8-sig")
        changed = source.replace("eligible = (volume > 0).copy()", "eligible = (volume >= 0).copy()")
        self.assertNotEqual(source, changed)
        with patch.object(Path, "read_text", lambda p, *a, **kw: changed if p == path else read(p, *a, **kw)):
            after = self._digests(RANKING_POLICY_SPECS[0].policy_id)
        self.assertEqual(before[:2], after[:2])
        self.assertNotEqual(before[2], after[2])

    def test_source_parse_cache_reads_content_and_selected_descriptor(self):
        path = ROOT / "core/ranking_policy.py"
        read = Path.read_text
        source = read(path, encoding="utf-8-sig")
        changed = source.replace("prefix_key='score',", "prefix_key='other',", 1)
        self.assertEqual(len(source), len(changed))
        replay._source_content_attestation.cache_clear()
        with patch.object(replay.ast, "parse", wraps=ast.parse) as parse:
            first = replay._source_owner_attestation(path, ranking_policy=RANKING_POLICY_SPECS[0].policy_id)[1]
            self.assertEqual(first, replay._source_owner_attestation(path, ranking_policy=RANKING_POLICY_SPECS[0].policy_id)[1])
            self.assertEqual(parse.call_count, 1)
            other = replay._source_owner_attestation(path, ranking_policy=RANKING_POLICY_SPECS[1].policy_id)[1]
            self.assertNotEqual(first, other)
            self.assertEqual(parse.call_count, 2)
            with patch.object(Path, "read_text", lambda p, *a, **kw: changed if p == path else read(p, *a, **kw)):
                self.assertNotEqual(first, replay._source_owner_attestation(path, ranking_policy=RANKING_POLICY_SPECS[0].policy_id)[1])
            self.assertEqual(parse.call_count, 3)


if __name__ == "__main__":
    unittest.main()

"""Exact source-subgraph certificates must never hide a later owner/helper edit."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.file_integrity import canonical_json_sha256
from core.research_execution_equivalence import (
    _REVIEWED_SOURCE_INVENTORIES,
    project_reviewed_source_inventory,
)
from services.audit import oracle_score_artifacts as prices
from services.audit import fulltrade_oracle_truth as full_truth
from services.research import oracle_artifact_contract as replay


ROOT = Path(__file__).resolve().parents[2]


def raw_inventory():
    return {name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in prices._DAILY_SCORE_SOURCE_MODULES}


class SourceInventoryProofTests(unittest.TestCase):
    def tearDown(self):
        replay.portfolio_candidate_execution_sha256.cache_clear()
        full_truth.truth_producer_fingerprint.cache_clear()

    def test_whole_group_is_required_and_unknown_owners_remain_effective(self):
        for scope, proof in _REVIEWED_SOURCE_INVENTORIES.items():
            current = {name: value if isinstance(value, str) else value[0]
                       for name, value in proof["current"].items()}
            current["unrelated/future_owner.py"] = "1" * 64
            projected = project_reviewed_source_inventory(current, scope=scope)
            self.assertEqual("1" * 64, projected["unrelated/future_owner.py"])
            for name in proof["current"]:
                with self.subTest(scope=scope, changed=name):
                    changed = {**current, name: "2" * 64}
                    self.assertEqual(changed, project_reviewed_source_inventory(changed, scope=scope))
                    missing = {key: value for key, value in current.items() if key != name}
                    self.assertEqual(missing, project_reviewed_source_inventory(missing, scope=scope))
            changed_extra = {**current, "unrelated/future_owner.py": "3" * 64}
            self.assertNotEqual(canonical_json_sha256(projected), canonical_json_sha256(
                project_reviewed_source_inventory(changed_extra, scope=scope)))

    def test_external_source_root_cannot_use_project_local_proof(self):
        with TemporaryDirectory() as directory:
            for scope, proof in _REVIEWED_SOURCE_INVENTORIES.items():
                values = {name: value if isinstance(value, str) else value[0]
                          for name, value in proof["current"].items()}
                with self.subTest(scope=scope):
                    self.assertEqual(values, project_reviewed_source_inventory(
                        values, scope=scope, project_root=directory))

    def test_price_source_equals_old_inventory_under_same_unchanged_sources(self):
        current = prices.oracle_daily_source_fingerprint()
        proof = _REVIEWED_SOURCE_INVENTORIES["oracle_daily_provider"]
        old_names = tuple(name for name in prices._DAILY_SCORE_SOURCE_MODULES
                          if name not in proof["current"] or name in proof["baseline"])
        read_sha = prices.compute_file_sha256
        def before_refactor(path):
            name = Path(path).relative_to(ROOT).as_posix()
            prior = proof["baseline"].get(name)
            return prior[0] if prior is not None else read_sha(path)
        # Reconstruct the original inventory/raw hashes, then run the unchanged
        # original per-file attestation and ordered digest. Non-scope bytes stay live.
        with patch.object(prices, "_DAILY_SCORE_SOURCE_MODULES", old_names), \
             patch.object(prices, "compute_file_sha256", before_refactor), \
             patch.object(prices, "project_reviewed_source_inventory", lambda parts, **_kwargs: dict(parts)):
            previous = prices.oracle_daily_source_fingerprint()
        self.assertEqual(previous, current)

    def test_lf_crlf_and_mixed_encoding_proofs_keep_all_reviewed_owners(self):
        proof = _REVIEWED_SOURCE_INVENTORIES["oracle_daily_provider"]
        values = {name: {None: record[0], b"\n": record[1], b"\r\n": record[2]}
                  for name, record in proof["current"].items()}
        for encoding in (1, 2):
            encoded = {name: {**options, None: proof["current"][name][encoding]}
                       for name, options in values.items()}
            projected = project_reviewed_source_inventory(encoded, scope="oracle_daily_provider")
            self.assertEqual(set(proof["baseline"]), set(projected))
            for name, record in proof["baseline"].items():
                self.assertEqual({None: record[0], b"\n": record[1], b"\r\n": record[2]}, projected[name])
        current, variants = prices.oracle_daily_source_fingerprint_proof()
        self.assertEqual(prices.oracle_daily_source_fingerprint(), current)
        self.assertIn(current, variants)

    def test_every_new_price_owner_mutation_changes_executing_source(self):
        original = prices.oracle_daily_source_fingerprint()
        read_bytes = Path.read_bytes
        read_sha = prices.compute_file_sha256
        for relative in ("filters/breakout_quality/daily_sample_accumulator.py",
                         "filters/breakout_quality/daily_model_input_context.py"):
            with self.subTest(helper=relative):
                target = ROOT / relative
                modified = read_bytes(target) + b"\n_REFACTOR_MUTATION_PROBE = True\n"
                def changed_bytes(path):
                    return modified if path == target else read_bytes(path)
                def changed_sha(path):
                    return sha256(modified).hexdigest() if Path(path) == target else read_sha(path)
                with patch.object(Path, "read_bytes", changed_bytes), patch.object(prices, "compute_file_sha256", changed_sha):
                    actual = prices.oracle_daily_source_fingerprint()
                    executing, variants = prices.oracle_daily_source_fingerprint_proof()
                self.assertNotEqual(original, actual)
                self.assertEqual(actual, executing)
                self.assertNotIn(original, variants)

    def test_candidate_scopes_preserve_reviewed_sources_but_invalidate_corrected_benchmark(self):
        previous = {
            "daily_atr_limit_v2": "48139e7e98220b53659fc42060761c063c3914942dacd8b038f3c5f591c4d302",
            "daily_atr_limit_common_filters_v2": "1d3791e5965caff6548b6eeada3d571ab0472bf49eedf4fc9c26ab3989ed7803",
        }
        replay.portfolio_candidate_execution_sha256.cache_clear()
        proof = _REVIEWED_SOURCE_INVENTORIES["portfolio_candidates"]
        benchmark = "core/portfolio_benchmark.py"
        self.assertNotIn(benchmark, proof["current"])
        self.assertNotIn(benchmark, proof["baseline"])
        for mode, old_digest in previous.items():
            parts = {"daily_execution_contract": mode, **proof["baseline"],
                     benchmark: canonical_json_sha256(replay._source_owner_payload(ROOT / benchmark))}
            if mode == "daily_atr_limit_common_filters_v2":
                optional = "core/daily_common_entry_filters.py"
                parts[optional] = _REVIEWED_SOURCE_INVENTORIES["portfolio_full_filters"]["baseline"][optional]
            current = replay.portfolio_candidate_execution_sha256(daily_execution_contract=mode)
            self.assertEqual(canonical_json_sha256(parts), current)
            self.assertNotEqual(old_digest, current)

    def test_full_oracle_teacher_matches_original_inventory_and_tracks_contract_helper(self):
        proof = _REVIEWED_SOURCE_INVENTORIES["full_oracle_teacher"]
        prior = sha256(b"formal_full_p17_netpnl_truth_v1\0")
        prior.update(full_truth.min_truth_producer_fingerprint("net_pnl").encode("ascii") + b"\n")
        for name, hashes in proof["baseline"].items():
            prior.update(name.encode("utf-8") + b"\0" + hashes[0].encode("ascii") + b"\n")
        full_truth.truth_producer_fingerprint.cache_clear()
        original = full_truth.truth_producer_fingerprint()
        self.assertEqual(prior.hexdigest(), original)
        read_bytes = Path.read_bytes
        for relative in proof["current"]:
            with self.subTest(owner=relative):
                target = ROOT / relative
                def changed_bytes(path):
                    data = read_bytes(path)
                    return data + b"\n_REFACTOR_MUTATION_PROBE = True\n" if path == target else data
                with patch.object(Path, "read_bytes", changed_bytes):
                    full_truth.truth_producer_fingerprint.cache_clear()
                    self.assertNotEqual(original, full_truth.truth_producer_fingerprint())
                full_truth.truth_producer_fingerprint.cache_clear()

    def test_each_extracted_candidate_helper_mutation_invalidates_both_scopes(self):
        modes = ("daily_atr_limit_v2", "daily_atr_limit_common_filters_v2")
        replay.portfolio_candidate_execution_sha256.cache_clear()
        original = {mode: replay.portfolio_candidate_execution_sha256(daily_execution_contract=mode) for mode in modes}
        read_text = Path.read_text
        for relative in ("core/ranking_policy.py", "core/portfolio_run_state.py"):
            with self.subTest(helper=relative):
                target = ROOT / relative
                def changed_text(path, *args, **kwargs):
                    text = read_text(path, *args, **kwargs)
                    return text + "\n_REFACTOR_MUTATION_PROBE = True\n" if path == target else text
                with patch.object(Path, "read_text", changed_text):
                    replay.portfolio_candidate_execution_sha256.cache_clear()
                    for mode in modes:
                        self.assertNotEqual(original[mode], replay.portfolio_candidate_execution_sha256(daily_execution_contract=mode))
                replay.portfolio_candidate_execution_sha256.cache_clear()

    def test_min_truth_proof_tracks_added_dependencies_and_all_reviewed_encodings(self):
        from services.audit import daily_trade40_oracle as truth, mintrade_truth_source_attestation as attestation
        proof = _REVIEWED_SOURCE_INVENTORIES["mintrade_source_graph"]
        original = truth._truth_source_sha(ROOT)
        attestation._mintrade_source_fingerprint_proof.cache_clear()
        executing, variants = attestation._mintrade_source_fingerprint_proof()
        self.assertEqual(original, executing)
        self.assertIn(original, variants)
        source_values = {name: value if isinstance(value, str) else value[0]
                         for name, value in proof["current"].items()}
        for index, encoding in enumerate((None, b"\n", b"\r\n")):
            projected = project_reviewed_source_inventory(
                source_values, scope="mintrade_source_graph", source_encoding=encoding,
            )
            self.assertEqual(list(projected), list(proof["baseline_order"]))
            self.assertEqual(projected, {name: value[index] for name, value in proof["baseline"].items()})
        read_bytes, read_sha = Path.read_bytes, attestation.compute_file_sha256
        for relative in set(proof["current"]) - set(proof["baseline"]) - {"@execution_sources"}:
            with self.subTest(helper=relative):
                target = ROOT / relative
                changed = read_bytes(target) + b"\n_MIN_SOURCE_MUTATION = True\n"
                with patch.object(Path, "read_bytes", lambda path: changed if path == target else read_bytes(path)), \
                     patch.object(attestation, "compute_file_sha256", lambda path: sha256(changed).hexdigest() if Path(path) == target else read_sha(path)):
                    self.assertNotEqual(original, truth._truth_source_sha(ROOT))
                    attestation._mintrade_source_fingerprint_proof.cache_clear()
                    current, altered_variants = attestation._mintrade_source_fingerprint_proof()
                    self.assertEqual(current, truth._truth_source_sha(ROOT))
                    self.assertNotIn(original, altered_variants)
        attestation._mintrade_source_fingerprint_proof.cache_clear()


if __name__ == "__main__":
    unittest.main()

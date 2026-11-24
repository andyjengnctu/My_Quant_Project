"""Historical receipt encoding is reconstructed only from the verified source graph."""
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

from services.audit import daily_trade40_oracle as truth
from services.audit import mintrade_truth_source_attestation as attestation


ROOT = Path(__file__).resolve().parents[2]
# Actual historical receipt evidence, not a production fingerprint allowlist.
HISTORICAL_MIN_RECEIPT_SOURCE = "c6f1a63e16f5b053fd4560a42fcfcaf6d923d1f2ccb282c9c0c8b5d6519b35b4"


class MinTradeEncodingProjectionTests(unittest.TestCase):
    def setUp(self):
        self._clear_proof_caches()
        self.addCleanup(self._clear_proof_caches)

    @staticmethod
    def _clear_proof_caches():
        for function in (
            attestation._truth_source_inventory, attestation._mintrade_source_fingerprint_proof,
            attestation._encoding_equivalent_mintrade_sources, attestation._archived_default_oracle_sources,
            truth.truth_producer_fingerprint,
        ):
            function.cache_clear()

    def test_historical_mixed_adapter_encoding_is_proven_without_changing_executing_identity(self):
        executing, variants = attestation._mintrade_source_fingerprint_proof()
        self.assertEqual(executing, truth._truth_source_sha(ROOT))
        self.assertEqual(executing, truth.truth_producer_fingerprint("net_r"))
        self.assertEqual(executing, truth.truth_producer_fingerprint("net_pnl"))
        self.assertIn(executing, variants)
        self.assertIn(HISTORICAL_MIN_RECEIPT_SOURCE, variants)
        self.assertTrue(attestation.mintrade_sources_equivalent(HISTORICAL_MIN_RECEIPT_SOURCE, executing))

    def test_economic_native_and_fee_changes_reject_historical_encoding(self):
        original, _ = attestation._mintrade_source_fingerprint_proof()
        read_bytes, read_sha = Path.read_bytes, attestation.compute_file_sha256
        for relative in ("core/native_optimizer_hotpath.c", "core/fee_rebate.py"):
            with self.subTest(owner=relative):
                target = ROOT / relative
                source = read_bytes(target)
                if target.suffix == ".c":
                    changed = source.replace(b"return PyLong_FromLongLong((long long)qty);",
                                             b"return PyLong_FromLongLong((long long)(qty + 1000));")
                else:
                    changed = source.replace(b"+ amount_milli", b"+ amount_milli + 1000", 1)
                self.assertNotEqual(source, changed)
                with patch.object(Path, "read_bytes", lambda p: changed if p == target else read_bytes(p)), \
                     patch.object(attestation, "compute_file_sha256", lambda p: sha256(changed).hexdigest() if Path(p) == target else read_sha(p)):
                    self._clear_proof_caches()
                    executing, variants = attestation._mintrade_source_fingerprint_proof()
                    self.assertNotEqual(original, executing)
                    self.assertEqual(executing, truth._truth_source_sha(ROOT))
                    self.assertNotIn(HISTORICAL_MIN_RECEIPT_SOURCE, variants)
                    self.assertFalse(attestation.mintrade_sources_equivalent(HISTORICAL_MIN_RECEIPT_SOURCE, executing))
                self._clear_proof_caches()

    def test_unknown_inventory_cannot_use_adapter_encoding_to_recreate_old_receipt(self):
        inventory = attestation._truth_source_inventory()
        original, _ = attestation._mintrade_source_fingerprint_proof()
        unknown_owner = "core/ranking_policy.py"
        self.assertNotIn(unknown_owner, inventory)
        changed_inventories = (
            (*inventory, unknown_owner),
            tuple(name for name in inventory if name != "core/native_optimizer_hotpath.c"),
        )
        for changed in changed_inventories:
            with self.subTest(inventory=changed):
                self._clear_proof_caches()
                with patch.object(attestation, "_truth_source_inventory", return_value=changed), \
                     patch.object(truth, "_TRUTH_SOURCE_MODULES", changed):
                    executing, variants = attestation._mintrade_source_fingerprint_proof()
                    self.assertNotEqual(original, executing)
                    self.assertEqual(executing, truth._truth_source_sha(ROOT))
                    self.assertNotIn(HISTORICAL_MIN_RECEIPT_SOURCE, variants)
                    self.assertFalse(attestation.mintrade_sources_equivalent(HISTORICAL_MIN_RECEIPT_SOURCE, executing))


if __name__ == "__main__":
    unittest.main()

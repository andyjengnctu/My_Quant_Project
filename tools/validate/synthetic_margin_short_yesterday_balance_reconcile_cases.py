"""Capability tests: offline T-1 FinMind YesterdayBalance vs saved official 前日餘額."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.file_integrity import canonical_json_sha256
from tools.audit import margin_short_official_publication_probe as probe
from tools.audit import margin_short_yesterday_balance_reconcile as target
from tools.validate.synthetic_margin_short_official_publication_probe_cases import _payload, _finmind

REPORT_DATE = "2025-01-06"
BALANCE_DATE = "2025-01-03"


def _source():
    frame = _finmind()
    frame["MarginPurchaseYesterdayBalance"] = [99, 20, 102, 21, 0]
    frame["ShortSaleYesterdayBalance"] = [3, 0, 3, 0, 0]
    return frame


def _seed_probe(root: Path, *, tamper_capture=False) -> tuple[dict, Path]:
    blob = json.dumps(_payload(), ensure_ascii=False).encode("utf-8")
    official = probe.parse_twse_mi_margn(_payload(), report_date=REPORT_DATE)
    old = probe.compare_official_confirmed_balances(official, _source().loc[:, probe.FINMIND_COLUMNS],
                                                      balance_date=BALANCE_DATE)
    sample = {"report_date": REPORT_DATE, "status": "OK", "capture_utc": "2026-10-10T04:08:51+00:00",
              "source_url": probe._url(REPORT_DATE), "raw_sha256": sha256(blob).hexdigest(),
              "source_vintage_grade": "CURRENT_VINTAGE_REQUERY_ONLY", "official_ticker_count": 2,
              "alignment": {"confirmed_balance_date": BALANCE_DATE,
                            "official_report_date": REPORT_DATE,
                            "prediction_date": "2025-01-07",
                            "information_rule": "official_(T-1)_previous_day_balance_is_T-2; use_before_T_open_only_if_vintage_proven"},
              "comparison": old}
    report = {"audit_contract": probe.AUDIT_ID, "research_cutoff": "2026-03-02",
              "model_input_authorized": False, "historical_contemporaneous_vintage_proven": False,
              "reports_acquisition": "public_historical_requery",
              "sample_dates": [REPORT_DATE], "samples": [sample], "sample_success": 1,
              "provider_snapshot_fingerprint": "a" * 64}
    report["audit_fingerprint"] = canonical_json_sha256({
        "contract": probe.AUDIT_ID, "snapshot": report["provider_snapshot_fingerprint"],
        "dates": report["sample_dates"],
        "capture": [(sample["report_date"], sample["raw_sha256"], sample["capture_utc"], sample["status"])]})
    path = root / probe.OUTPUT_ROOT / report["audit_fingerprint"]
    (path / "captured_requeries").mkdir(parents=True)
    (path / "publication_probe.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (path / "captured_requeries" / "20250106.json").write_bytes(blob + (b"tamper" if tamper_capture else b""))
    return report, path


def _fake_view(frame: pd.DataFrame, *, calendar=None):
    class FakeView:
        def __init__(self, *args, **kwargs):
            pass
        def dataset_artifacts(self, dataset):
            return [SimpleNamespace(request_id="f", content_sha256="c" * 64, row_count=len(frame))]
        def iter_verified_frames(self, dataset, columns):
            if dataset != probe.CALENDAR_DATASET:
                raise AssertionError("unexpected dataset")
            yield pd.DataFrame({"date": calendar or ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]})
        def iter_verified_frames_many_data_ids(self, dataset, columns, data_ids):
            assert dataset == probe.DATASET
            assert "0050" in data_ids
            yield frame.loc[:, [c for c in columns if c in frame.columns]].copy()
    return FakeView


class YesterdayBalanceReconciliationTests(unittest.TestCase):
    def _run(self, tmp: str, frame: pd.DataFrame, *, tamper=False):
        report, old_dir = _seed_probe(Path(tmp), tamper_capture=tamper)
        old_blob = (old_dir / "publication_probe.json").read_bytes()
        old_capture = (old_dir / "captured_requeries" / "20250106.json").read_bytes()
        archive = SimpleNamespace(as_of_date="2026-03-02", snapshot_fingerprint="a" * 64)
        with (patch.object(target, "ProviderSnapshotView", _fake_view(frame)),
              patch.object(target, "load_ready_provider_snapshot_archive", return_value=archive),
              patch.object(probe, "_fetch_official", side_effect=AssertionError("no network"))):
            result, folder = target.run_reconcile(Path(tmp), report_date=REPORT_DATE)
        self.assertFalse(result["model_input_authorized"])
        self.assertFalse(result["historical_contemporaneous_vintage_proven"])
        self.assertEqual(result["external_requests"], 0)
        self.assertEqual((old_dir / "publication_probe.json").read_bytes(), old_blob)
        self.assertEqual((old_dir / "captured_requeries" / "20250106.json").read_bytes(), old_capture)
        self.assertTrue((folder / "reconcile.md").is_file())
        self.assertTrue((folder / "reconcile.json").is_file())
        return result, folder

    def test_offline_exact_yesterday_comparison_reconciles_old_mismatch(self):
        with TemporaryDirectory() as tmp:
            result, folder = self._run(tmp, _source())
            self.assertEqual(result["status"], "CURRENT_VINTAGE_NUMERIC_RECONCILIATION_AVAILABLE_PIT_BLOCKED")
            c = result["comparison"]
            self.assertEqual(c["old_confirmed_date_today"]["margin_equal"], 1)
            self.assertEqual(c["new_report_date_yesterday"]["margin_equal"], 2)
            self.assertEqual(c["new_report_date_yesterday"]["short_equal"], 2)
            self.assertEqual(c["previous_mismatch_reconciled"], 1)
            self.assertEqual(c["previous_mismatch_details"][0]["ticker"], "2330")
            self.assertEqual(c["previous_mismatch_details"][0]["new_margin_delta"], 0.0)
            self.assertIn("PIT_REVIEW_REQUIRED", (folder / "reconcile.md").read_text(encoding="utf-8"))
            self.assertEqual(result["prediction_date"], "2025-01-07")

    def test_missing_yesterday_columns_fails_not_autofilled(self):
        with TemporaryDirectory() as tmp:
            without = _source().drop(columns=["MarginPurchaseYesterdayBalance"])
            result, _ = self._run(tmp, without)
            self.assertEqual(result["status"], "RECONCILIATION_BLOCKED")
            self.assertIn("YesterdayBalance", result["block_reason"])

    def test_duplicate_ticker_date_never_choose_one(self):
        with TemporaryDirectory() as tmp:
            frame = _source()
            extra = frame.loc[(frame["stock_id"] == "0050") & (frame["date"] == REPORT_DATE)]
            result, _ = self._run(tmp, pd.concat([frame, extra], ignore_index=True))
            self.assertEqual(result["status"], "RECONCILIATION_BLOCKED")
            self.assertIn("重複", result["block_reason"])

    def test_previous_t_minus_two_source_drift_fails_closed(self):
        with TemporaryDirectory() as tmp:
            frame = _source()
            frame.loc[(frame["stock_id"] == "0050") & (frame["date"] == BALANCE_DATE),
                      "MarginPurchaseTodayBalance"] = 101
            result, _ = self._run(tmp, frame)
            self.assertEqual(result["status"], "RECONCILIATION_BLOCKED")
            self.assertIn("READY Snapshot不符", result["block_reason"])

    def test_capture_tamper_fails_closed(self):
        with TemporaryDirectory() as tmp:
            result, _ = self._run(tmp, _source(), tamper=True)
            self.assertEqual(result["status"], "RECONCILIATION_BLOCKED")
            self.assertIn("SHA256", result["block_reason"])

    def test_genuine_new_discrepancy_not_suppressed(self):
        with TemporaryDirectory() as tmp:
            frame = _source()
            frame.loc[(frame["stock_id"] == "0050") & (frame["date"] == REPORT_DATE),
                      "MarginPurchaseYesterdayBalance"] = 101
            result, _ = self._run(tmp, frame)
            self.assertEqual(result["status"], "CURRENT_VINTAGE_NUMERIC_RECONCILIATION_AVAILABLE_PIT_BLOCKED")
            self.assertEqual(result["comparison"]["new_report_date_yesterday"]["margin_equal"], 1)
            self.assertEqual(result["comparison"]["new_report_date_yesterday"]["mismatch_count"], 1)

    def test_missing_source_yesterday_not_misread_as_zero(self):
        with TemporaryDirectory() as tmp:
            frame = _source()
            frame.loc[(frame["stock_id"] == "2330") & (frame["date"] == REPORT_DATE),
                      "MarginPurchaseYesterdayBalance"] = None
            result, _ = self._run(tmp, frame)
            self.assertEqual(result["status"], "CURRENT_VINTAGE_NUMERIC_RECONCILIATION_AVAILABLE_PIT_BLOCKED")
            self.assertEqual(result["comparison"]["new_report_date_yesterday"]["finmind_invalid_count"], 1)
            self.assertEqual(result["comparison"]["previous_mismatch_reconciled"], 0)

    def test_numeric_stock_id_is_rejected_to_preserve_0050(self):
        with TemporaryDirectory() as tmp:
            frame = _source()
            frame["stock_id"] = [50, 2330, 50, 2330, 50]
            result, _ = self._run(tmp, frame)
            self.assertEqual(result["status"], "RECONCILIATION_BLOCKED")
            self.assertIn("stock_id", result["block_reason"])

    def test_no_official_saved_probe_stays_blocked(self):
        with TemporaryDirectory() as tmp:
            archive = SimpleNamespace(as_of_date="2026-03-02", snapshot_fingerprint="a" * 64)
            with patch.object(target, "load_ready_provider_snapshot_archive", return_value=archive):
                result, folder = target.run_reconcile(Path(tmp), report_date=REPORT_DATE)
            self.assertEqual(result["status"], "RECONCILIATION_BLOCKED")
            self.assertEqual(result["external_requests"], 0)
            self.assertTrue((folder / "reconcile.json").is_file())

    def test_future_data_alone_cannot_change_result(self):
        with TemporaryDirectory() as tmp:
            frame = _source()
            future = frame.loc[frame["date"] == "2025-01-07"].copy()
            future["MarginPurchaseYesterdayBalance"] = 2_000_000_000
            result, _ = self._run(tmp, pd.concat([frame, future], ignore_index=True))
            self.assertEqual(result["status"], "CURRENT_VINTAGE_NUMERIC_RECONCILIATION_AVAILABLE_PIT_BLOCKED")
            self.assertEqual(result["comparison"]["new_report_date_yesterday"]["both_equal"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

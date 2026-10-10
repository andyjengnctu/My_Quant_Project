"""Isolated tests for official T-1 previous balance (T-2) vintage-safe probe."""
from __future__ import annotations

import json
import ssl
from urllib.error import URLError
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.audit import margin_short_official_publication_probe as probe


FIELDS = ["代號", "名稱", "買進", "賣出", "現金償還", "前日餘額", "今日餘額", "次一營業日限額",
          "買進", "賣出", "現券償還", "前日餘額", "今日餘額", "次一營業日限額", "資券互抵", "註記"]


def _payload(rows=None, title="114年01月06日 融資融券彙總"):
    if rows is None:
        rows = [
            ["0050", "元大台灣50", "10", "2", "0", "102", "110", "1,000", "0", "1", "0", "3", "2", "1,000", "0", ""],
            ["2330", "台積電", "5", "4", "0", "21", "22", "2,000", "1", "0", "0", "0", "1", "2,000", "0", ""],
        ]
    return {
        "stat": "OK", "title": title,
        "tables": [
            {"title": "114年01月06日 信用交易統計", "fields": ["項目", "買進", "賣出", "現金償還", "前日餘額", "今日餘額"],
             "data": [["融資(交易單位)", "20", "10", "0", "100", "110"]]},
            {"title": title, "fields": list(FIELDS), "data": [list(row) for row in rows]},
        ],
    }


def _finmind():
    return pd.DataFrame([
        ("0050", "2025-01-03", 102, 3),
        ("2330", "2025-01-03", 20, 0),
        ("0050", "2025-01-06", 110, 2),
        ("2330", "2025-01-06", 22, 1),
        ("0050", "2025-01-07", 99999, 8888),
    ], columns=probe.FINMIND_COLUMNS)


class OfficialMarginPublicationProbeTests(unittest.TestCase):
    def test_explicit_official_date_and_two_previous_balances(self):
        rows = probe.parse_twse_mi_margn(_payload(), report_date="2025-01-06")
        self.assertEqual(rows["0050"], (102, 3))
        self.assertEqual(rows["2330"], (21, 0))
        self.assertNotIn("融資(交易單位)", rows)
        self.assertEqual(probe._reported_dates(_payload()), {"2025-01-06"})

    def test_reject_wrong_date_even_if_query_date_claimed(self):
        with self.assertRaisesRegex(ValueError, "日期"):
            wrong = _payload(title="114年01月07日 融資融券彙總")
            wrong["tables"][0]["title"] = "114年01月07日 信用交易統計"
            probe.parse_twse_mi_margn(wrong, report_date="2025-01-06")
        with self.assertRaisesRegex(ValueError, "日期"):
            probe.parse_twse_mi_margn({"tables": _payload()["tables"][1:1]}, report_date="2025-01-06")

    def test_reject_bad_schema_duplicate_and_zero_as_legit(self):
        payload = _payload()
        payload["tables"][1]["fields"][11] = "今日餘額"
        with self.assertRaisesRegex(ValueError, "數量不為1"):
            probe.parse_twse_mi_margn(payload, report_date="2025-01-06")
        duplicated = _payload()
        duplicated["tables"][1]["data"].append(duplicated["tables"][1]["data"][0])
        with self.assertRaisesRegex(ValueError, "重複"):
            probe.parse_twse_mi_margn(duplicated, report_date="2025-01-06")
        zero_payload = _payload(rows=[
            ["0050", "元大台灣50", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "1", "0", ""]])
        self.assertEqual(probe.parse_twse_mi_margn(zero_payload, report_date="2025-01-06")["0050"], (0, 0))

    def test_official_known_aggregate_row_skipped_without_discarding_real_bad_rows(self):
        # Some TWSE per-category tables have a first blank-code / aggregate row.
        payload = _payload()
        total = ["　", "合計", "1", "2", "3", "100", "200", "1000",
                 "1", "2", "0", "4", "5", "1000", "0", ""]
        payload["tables"][1]["data"].insert(0, total)
        parsed = probe.parse_twse_mi_margn(payload, report_date="2025-01-06")
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed["0050"], (102, 3))
        payload["tables"][1]["data"][0][1] = "未知摘要"
        with self.assertRaisesRegex(ValueError, "代碼不合法"):
            probe.parse_twse_mi_margn(payload, report_date="2025-01-06")

    def test_exact_failure_type_is_in_human_audit(self):
        from urllib.error import HTTPError, URLError
        self.assertEqual(probe._source_error_kind(HTTPError(
            "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN", 403,
            "Forbidden", None, None)), "HTTP_403")
        self.assertEqual(probe._source_error_kind(URLError("blocked")), "NETWORK_OR_TLS_FAILURE")
        report = probe.markdown_report({
            "status": "NO_OFFICIAL_COMPARISON_EVIDENCE", "research_cutoff": "2026-03-02",
            "sample_dates": ["2025-02-03"], "sample_success": 0,
            "samples": [{"report_date": "2025-02-03", "status": "SOURCE_NOT_VERIFIABLE",
                         "error_kind": "HTTP_403", "error": "HTTPError: Forbidden"}],
        })
        self.assertIn("HTTP_403: HTTPError: Forbidden", report)
        self.assertIn("模型 Input 授權：**NO**", report)
        self.assertIn("| 失敗原因 |", report)

    def test_existing_diagnosis_never_fetches_or_reads_provider(self):
        with TemporaryDirectory(prefix="margin_official_existing_") as tmp:
            base = Path(tmp) / probe.OUTPUT_ROOT / ("f" * 64)
            base.mkdir(parents=True)
            (base / "publication_probe.json").write_text(json.dumps({
                "status": "NO_OFFICIAL_COMPARISON_EVIDENCE", "sample_success": 0,
                "samples": [{"report_date": "2025-02-03", "status": "SOURCE_NOT_VERIFIABLE",
                             "error": "HTTPError: HTTP Error 403: Forbidden"}]}, ensure_ascii=False), encoding="utf-8")
            with (patch.object(probe, "_fetch_official", side_effect=AssertionError("network")),
                  patch.object(probe, "load_ready_provider_snapshot_archive", side_effect=AssertionError("provider"))):
                diagnosis, code = probe.existing_probe_diagnosis(Path(tmp))
            self.assertEqual(code, 0)
            self.assertIn("2025-02-03", diagnosis)
            self.assertIn("HTTP Error 403", diagnosis)
            self.assertIn("LEGACY_UNCLASSIFIED", diagnosis)
            self.assertIn("沒有重新下載TWSE", diagnosis)
            self.assertIn("模型Input授權維持NO", diagnosis)

    def test_report_tminus_one_confirms_tminus_two_not_same_day(self):
        calendar = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
        alignment = probe.forecast_alignment(calendar, "2025-01-06", cutoff="2025-01-07")
        self.assertEqual(alignment["confirmed_balance_date"], "2025-01-03")
        self.assertEqual(alignment["official_report_date"], "2025-01-06")
        self.assertEqual(alignment["prediction_date"], "2025-01-07")
        for wrong in ("2025-01-02", "2025-01-07", "2025-01-05"):
            with self.assertRaises(ValueError):
                probe.forecast_alignment(calendar, wrong, cutoff="2025-01-07")
        with self.assertRaisesRegex(ValueError, "cutoff"):
            probe.forecast_alignment(calendar, "2025-01-06", cutoff="2025-01-06")

    def test_compare_current_vintage_diagnostic_mismatch(self):
        official = probe.parse_twse_mi_margn(_payload(), report_date="2025-01-06")
        result = probe.compare_official_confirmed_balances(official, _finmind(), balance_date="2025-01-03")
        self.assertEqual(result["paired_count"], 2)
        self.assertEqual(result["margin_equal"], 1)
        self.assertEqual(result["short_equal"], 2)
        self.assertEqual(result["both_equal"], 1)
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["sample_mismatches"][0]["ticker"], "2330")
        self.assertEqual(result["sample_mismatches"][0]["official_margin_previous"], 21)
        # What is recorded at or AFTER T may not alter the T-2 comparison.
        mutated = _finmind()
        mutated.loc[mutated.date == "2025-01-07", "MarginPurchaseTodayBalance"] = 10**9
        self.assertEqual(probe.compare_official_confirmed_balances(official, mutated,
                         balance_date="2025-01-03"), result)

    def test_stock_id_leading_zero_and_source_duplicate(self):
        official = probe.parse_twse_mi_margn(_payload(), report_date="2025-01-06")
        missing = _finmind().loc[lambda frame: frame.stock_id != "0050"]
        compare = probe.compare_official_confirmed_balances(official, missing, balance_date="2025-01-03")
        self.assertEqual(compare["finmind_missing_count"], 1)
        duplicate = pd.concat([_finmind(), _finmind().iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "重複"):
            probe.compare_official_confirmed_balances(official, duplicate, balance_date="2025-01-03")
        broken = _finmind()
        broken["stock_id"] = [50, 2330, 50, 2330, 50]
        with self.assertRaisesRegex(ValueError, "前導零"):
            probe.compare_official_confirmed_balances(official, broken, balance_date="2025-01-03")

    def test_sample_dates_based_on_calendar_not_labels(self):
        dates = pd.DatetimeIndex(["2021-01-05", "2021-02-01", "2021-02-02", "2022-02-07", "2022-02-08",
                                  "2026-02-02", "2026-02-03"])
        self.assertEqual(probe._sample_dates(dates, "2026-03-02"),
                         ["2021-02-01", "2022-02-07", "2026-02-02"])

    def test_report_says_no_model_and_no_vintage(self):
        report = probe.markdown_report({"status": "HISTORICAL_VINTAGE_NOT_PROVEN", "research_cutoff": "2026-03-02",
             "sample_dates": ["2025-01-06"], "sample_success": 1,
             "samples": [{"report_date": "2025-01-06", "status": "OK", "alignment": {
                  "official_report_date": "2025-01-06", "confirmed_balance_date": "2025-01-03", "prediction_date": "2025-01-07"},
                  "comparison": {"official_stock_count": 2, "paired_count": 2, "margin_equal": 1,
                                 "short_equal": 2, "both_equal": 1}}]})
        self.assertIn("模型 Input 授權：**NO**", report)
        self.assertIn("不是當時原始發布版本", report)
        self.assertIn("TPEx", report)
        self.assertIn("限額", report)

    def test_raw_capture_is_byte_exact(self):
        with TemporaryDirectory(prefix="margin_twse_capture_test_") as tmp:
            output = Path(tmp) / "capture.json"
            blob = json.dumps(_payload(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            probe._write_exact_capture(output, blob)
            self.assertEqual(output.read_bytes(), blob)
            self.assertEqual(probe.sha256(output.read_bytes()).hexdigest(), probe.sha256(blob).hexdigest())


    def test_twse_strict_tls_success_requires_no_compatibility(self):
        class Response:
            url = probe._url("2025-01-06")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, _limit):
                return b"valid-response"
        evidence = {}
        with patch.object(probe, "urlopen", return_value=Response()) as fetch:
            raw, target = probe._fetch_official("2025-01-06", tls_evidence=evidence)
        self.assertEqual(raw, b"valid-response")
        self.assertEqual(target, probe._url("2025-01-06"))
        self.assertEqual(evidence["tls_validation_mode"], "DEFAULT_STRICT")
        self.assertEqual(fetch.call_count, 1)
        self.assertNotIn("context", fetch.call_args.kwargs)

    def test_twse_missing_ski_compatibility_keeps_ca_and_hostname_validation(self):
        class Response:
            url = probe._url("2025-01-06")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, _limit):
                return b"official-bytes"
        missing_ski = URLError(ssl.SSLCertVerificationError(
            1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: Missing Subject Key Identifier"))
        self.assertTrue(probe._missing_subject_key_identifier_error(missing_ski))
        self.assertEqual(probe._source_error_kind(missing_ski),
                         "TLS_X509_MISSING_SUBJECT_KEY_IDENTIFIER")
        evidence = {}
        with patch.object(probe, "urlopen", side_effect=[missing_ski, Response()]) as fetch:
            raw, _ = probe._fetch_official("2025-01-06", tls_evidence=evidence)
        self.assertEqual(raw, b"official-bytes")
        self.assertEqual(fetch.call_count, 2)
        first, second = fetch.call_args_list
        self.assertNotIn("context", first.kwargs)
        context = second.kwargs["context"]
        self.assertIsInstance(context, ssl.SSLContext)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_flags & ssl.VERIFY_X509_STRICT, 0)
        self.assertEqual(evidence["tls_validation_mode"], "CA_CHAIN_AND_HOSTNAME_REQUIRED_X509_FORMAT_COMPAT")
        # A per-request context must not weaken the Python/global TLS defaults.
        self.assertTrue(ssl.create_default_context().verify_flags & ssl.VERIFY_X509_STRICT)

    def test_twse_does_not_retry_unrelated_certificate_or_network_failures(self):
        errors = [URLError(ssl.SSLCertVerificationError(1, "certificate has expired")),
                  URLError(ssl.SSLCertVerificationError(1, "unable to get local issuer certificate")),
                  URLError("connection reset")]
        for exc in errors:
            with self.subTest(reason=str(exc)), patch.object(probe, "urlopen", side_effect=exc) as fetch:
                with self.assertRaises(URLError):
                    probe._fetch_official("2025-01-06")
                self.assertEqual(fetch.call_count, 1)
        self.assertEqual(probe._source_error_kind(errors[0]), "TLS_CERTIFICATE_VERIFY_FAILED")
        self.assertFalse(probe._missing_subject_key_identifier_error(URLError(
            "Missing Subject Key Identifier text from untrusted generic error")))

    def test_twse_missing_ski_retry_is_bounded_and_still_fails_closed(self):
        missing_ski = URLError(ssl.SSLCertVerificationError(1, "Missing Subject Key Identifier"))
        untrusted = URLError(ssl.SSLCertVerificationError(1, "unable to get local issuer certificate"))
        with patch.object(probe, "urlopen", side_effect=[missing_ski, untrusted]) as fetch:
            with self.assertRaises(URLError) as raised:
                probe._fetch_official("2025-01-06")
            self.assertIs(raised.exception, untrusted)
            self.assertEqual(fetch.call_count, 2)

    def test_tls_retry_failure_is_audited_and_does_not_authorize(self):
        from types import SimpleNamespace
        archive = SimpleNamespace(as_of_date="2026-03-02", snapshot_fingerprint="a" * 64)
        class FakeVerifiedView:
            def __init__(self, *args, **kwargs):
                pass
            def dataset_artifacts(self, dataset):
                return [SimpleNamespace(request_id="r", content_sha256="b" * 64, row_count=4)]
            def iter_verified_frames(self, dataset, columns):
                yield pd.DataFrame({"date": ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]})
        missing_ski = URLError(ssl.SSLCertVerificationError(1, "Missing Subject Key Identifier"))
        untrusted = URLError(ssl.SSLCertVerificationError(1, "unable to get local issuer certificate"))
        with TemporaryDirectory(prefix="margin_official_tls_fail_closed_") as tmp:
            with (patch.object(probe, "load_ready_provider_snapshot_archive", return_value=archive),
                  patch.object(probe, "ProviderSnapshotView", FakeVerifiedView),
                  patch.object(probe, "urlopen", side_effect=[missing_ski, untrusted])):
                result, folder = probe.run_probe(Path(tmp), dates=["2025-01-06"], fetch_public=True)
            self.assertEqual(result["status"], "NO_OFFICIAL_COMPARISON_EVIDENCE")
            self.assertEqual(result["sample_success"], 0)
            self.assertFalse(result["model_input_authorized"])
            entry = result["samples"][0]
            self.assertEqual(entry["error_kind"], "TLS_CERTIFICATE_VERIFY_FAILED")
            self.assertEqual(entry["tls_validation_mode"], "CA_CHAIN_AND_HOSTNAME_REQUIRED_X509_FORMAT_COMPAT")
            self.assertIn("TLS_CERTIFICATE_VERIFY_FAILED", (folder / "publication_probe.md").read_text(encoding="utf-8"))
            self.assertFalse((folder / "captured_requeries").exists())

    def test_tls_failure_artifacts_are_immutable_across_distinct_retries(self):
        from types import SimpleNamespace
        archive = SimpleNamespace(as_of_date="2026-03-02", snapshot_fingerprint="a" * 64)
        class FakeVerifiedView:
            def __init__(self, *args, **kwargs):
                pass
            def dataset_artifacts(self, dataset):
                return [SimpleNamespace(request_id="r", content_sha256="b" * 64, row_count=4)]
            def iter_verified_frames(self, dataset, columns):
                yield pd.DataFrame({"date": ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]})
        expired = URLError(ssl.SSLCertVerificationError(1, "certificate has expired"))
        untrusted = URLError(ssl.SSLCertVerificationError(1, "unable to get local issuer certificate"))
        with TemporaryDirectory(prefix="margin_official_error_identity_") as tmp:
            with (patch.object(probe, "load_ready_provider_snapshot_archive", return_value=archive),
                  patch.object(probe, "ProviderSnapshotView", FakeVerifiedView)):
                with patch.object(probe, "urlopen", side_effect=expired):
                    first, first_folder = probe.run_probe(Path(tmp), dates=["2025-01-06"], fetch_public=True)
                original = (first_folder / "publication_probe.json").read_bytes()
                with patch.object(probe, "urlopen", side_effect=untrusted):
                    second, second_folder = probe.run_probe(Path(tmp), dates=["2025-01-06"], fetch_public=True)
            self.assertNotEqual(first["audit_fingerprint"], second["audit_fingerprint"])
            self.assertNotEqual(first_folder, second_folder)
            self.assertEqual((first_folder / "publication_probe.json").read_bytes(), original)
            self.assertTrue((second_folder / "publication_probe.json").exists())

    def test_official_url_is_hardcoded_twse_https(self):
        url = probe._url("2025-01-06")
        self.assertTrue(url.startswith("https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?"))
        self.assertIn("date=20250106", url)
        self.assertIn("selectType=ALL", url)

    def test_run_with_official_requery_and_verified_snapshot_stays_blocked(self):
        from types import SimpleNamespace
        report = json.dumps(_payload(), ensure_ascii=False).encode("utf-8")
        archive = SimpleNamespace(as_of_date="2026-03-02", snapshot_fingerprint="a" * 64)
        class FakeVerifiedView:
            def __init__(self, *args, **kwargs):
                pass
            def dataset_artifacts(self, name):
                return [SimpleNamespace(request_id="r", content_sha256="b" * 64, row_count=5)]
            def iter_verified_frames(self, dataset, columns):
                self.assert_wanted = dataset
                yield pd.DataFrame({"date": ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]})
            def iter_verified_frames_many_data_ids(self, dataset, columns, data_ids):
                if dataset != probe.DATASET:
                    raise AssertionError("wrong provider source")
                if "0050" not in data_ids:
                    raise AssertionError("leading-zero stock ID not selected")
                yield _finmind().loc[:, list(columns)].copy()
        with TemporaryDirectory(prefix="margin_twse_full_probe_") as tmp:
            with (patch.object(probe, "load_ready_provider_snapshot_archive", return_value=archive),
                  patch.object(probe, "ProviderSnapshotView", FakeVerifiedView),
                  patch.object(probe, "_fetch_official", return_value=(report, probe._url("2025-01-06")))):
                result, folder = probe.run_probe(Path(tmp), dates=["2025-01-06"], fetch_public=True)
            self.assertEqual(result["status"], "HISTORICAL_VINTAGE_NOT_PROVEN")
            self.assertEqual(result["sample_success"], 1)
            self.assertEqual(result["samples"][0]["comparison"]["paired_count"], 2)
            self.assertEqual(result["samples"][0]["alignment"]["confirmed_balance_date"], "2025-01-03")
            self.assertFalse(result["historical_contemporaneous_vintage_proven"])
            self.assertFalse(result["model_input_authorized"])
            self.assertEqual((folder / "captured_requeries" / "20250106.json").read_bytes(), report)
            self.assertTrue((folder / "publication_probe.md").exists())
            self.assertTrue((folder / "publication_probe.json").exists())

    def test_error_response_retains_raw_bytes_and_diagnostic_in_markdown(self):
        from types import SimpleNamespace
        archive = SimpleNamespace(as_of_date="2026-03-02", snapshot_fingerprint="a" * 64)
        class FakeVerifiedView:
            def __init__(self, *args, **kwargs):
                pass
            def dataset_artifacts(self, dataset):
                return [SimpleNamespace(request_id="r", content_sha256="b" * 64, row_count=4)]
            def iter_verified_frames(self, dataset, columns):
                yield pd.DataFrame({"date": ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]})
        blob = b"not json: invalid response"
        with TemporaryDirectory(prefix="margin_official_bad_response_") as tmp:
            with (patch.object(probe, "load_ready_provider_snapshot_archive", return_value=archive),
                  patch.object(probe, "ProviderSnapshotView", FakeVerifiedView),
                  patch.object(probe, "_fetch_official", return_value=(blob, probe._url("2025-01-06")))):
                result, folder = probe.run_probe(Path(tmp), dates=["2025-01-06"], fetch_public=True)
            self.assertEqual(result["status"], "NO_OFFICIAL_COMPARISON_EVIDENCE")
            self.assertEqual(result["samples"][0]["error_kind"], "NON_JSON_OFFICIAL_RESPONSE")
            self.assertEqual((folder / "captured_requeries" / "20250106.response").read_bytes(), blob)
            self.assertFalse((folder / "captured_requeries" / "20250106.json").exists())
            self.assertIn("NON_JSON_OFFICIAL_RESPONSE", (folder / "publication_probe.md").read_text(encoding="utf-8"))
            self.assertFalse(result["model_input_authorized"])

    def test_no_snapshot_does_not_create_authorization(self):
        with TemporaryDirectory(prefix="margin_twse_missing_snapshot_") as tmp:
            with patch.object(probe, "load_ready_provider_snapshot_archive", side_effect=FileNotFoundError("absent")):
                result, folder = probe.run_probe(Path(tmp), dates=["2025-01-06"], fetch_public=False)
            self.assertEqual(result["status"], "SOURCE_OR_CONTRACT_BLOCKED")
            self.assertFalse(result["model_input_authorized"])
            self.assertFalse(result["historical_contemporaneous_vintage_proven"])
            self.assertTrue((folder / "publication_probe.md").is_file())
            self.assertTrue((folder / "publication_probe.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)

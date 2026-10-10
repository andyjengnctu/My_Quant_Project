"""Isolated invariants for the one-off margin/short-sale PIT Data Gate."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.audit.margin_short_pit_data_gate import audit_coverage, markdown_report


def _toy_control():
    dates = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    return {"0050": dates, "2330": dates}


def _toy_source():
    return pd.DataFrame([
        ("0050", "2025-01-02", 0, 100, 0, 10),
        ("0050", "2025-01-03", 10, 0, 2, 10),
        ("0050", "2025-01-06", 15, 100, 5, 10),
        ("0050", "2025-01-07", 2, 100, 2, 10),
        ("2330", "2025-01-02", 0, 100, 0, 10),
        ("2330", "2025-01-03", -1, 100, 2, 10),
    ], columns=["stock_id", "date", "MarginPurchaseTodayBalance", "MarginPurchaseLimit",
                "ShortSaleTodayBalance", "ShortSaleLimit"])


def _audit(frame):
    return audit_coverage(_toy_control(), [frame], end_date="2025-01-07",
                          oos_start_date="2025-01-02")


def test_pit_and_missing():
    result = _audit(_toy_source())
    overall = result["oos_control_window"]
    assert overall["control_stock_days"] == 8
    assert overall["previous_session_available"] == 6
    assert overall["source_row_present"] == 5
    assert overall["source_row_missing"] == 1
    assert overall["both_ratio_valid"] == 3
    assert overall["margin_ratio_valid"] == 3
    assert overall["short_ratio_valid"] == 5
    assert overall["margin_limit_zero"] == 1
    assert overall["margin_balance_zero_valid"] == 2
    assert overall["short_balance_zero_valid"] == 2
    assert overall["margin_values_invalid"] == 2
    assert overall["short_values_invalid"] == 0
    assert overall["both_ratio_coverage_pct"] == 50.0
    assert result["control_ticker_count"] == 2
    assert {item["ticker"] for item in result["source_missing_examples"]} == {"2330"}
    assert result["availability_rule"].startswith("source.date = strictly_previous")
    assert "2025" in result["by_prediction_year"]
    assert result["source"]["source_earliest_control_ticker_date"] == "2025-01-02"
    return 15


def test_no_future_lookahead():
    source = _toy_source()
    base = _audit(source)
    changed = source.copy()
    changed.loc[changed["date"] == "2025-01-07", "MarginPurchaseTodayBalance"] = 1_000_000
    assert base["oos_control_window"] == _audit(changed)["oos_control_window"]
    assert base["by_prediction_year"] == _audit(changed)["by_prediction_year"]
    return 2


def test_missing_and_zero_are_distinct():
    source = _toy_source()
    zero = _audit(source)
    absent = _audit(source.loc[~((source["stock_id"] == "2330") & (source["date"] == "2025-01-02"))])
    assert zero["oos_control_window"]["margin_balance_zero_valid"] == 2
    assert absent["oos_control_window"]["margin_balance_zero_valid"] == 1
    assert absent["oos_control_window"]["source_row_missing"] == 2
    return 3


def test_duplicate_and_invalid_dates():
    source = _toy_source()
    duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    audit = _audit(duplicated)
    assert audit["source"]["source_duplicate_ticker_date_rows"] == 1
    assert audit["oos_control_window"]["both_ratio_valid"] == 2
    assert "0050" == audit["source_duplicate_examples"][0]["ticker"]
    bad = source.copy()
    bad.loc[0, "date"] = "garbage"
    audit = _audit(bad)
    assert audit["source"]["source_invalid_date_rows"] == 1
    return 4


def test_blank_and_negative_limits():
    source = _toy_source()
    invalid = source.copy()
    invalid.loc[0, "MarginPurchaseLimit"] = None
    invalid.loc[0, "ShortSaleLimit"] = -5
    audit = _audit(invalid)
    assert audit["oos_control_window"]["source_row_present"] == 5
    assert audit["oos_control_window"]["both_ratio_valid"] == 2
    assert audit["oos_control_window"]["margin_values_invalid"] == 3
    assert audit["oos_control_window"]["short_values_invalid"] == 1
    return 4


def test_multi_fragment_and_year_split():
    source = _toy_source()
    audit = audit_coverage(_toy_control(), [source.iloc[:3], source.iloc[3:]],
                           end_date="2025-01-07", oos_start_date="2025-01-07")
    assert audit["all_control_history"]["both_ratio_valid"] == 3
    assert audit["oos_control_window"]["control_stock_days"] == 2
    assert audit["oos_control_window"]["both_ratio_valid"] == 1
    return 3


def test_schema_control_and_report():
    source = _toy_source()
    audit = _audit(source.drop(columns=["ShortSaleLimit"]))
    assert len(audit["source"]["schema_errors"]) == 1
    assert audit["oos_control_window"]["both_ratio_valid"] == 0
    try:
        audit_coverage({"2330": pd.DatetimeIndex(["2025-01-02", "2025-01-02"])}, [source],
                       end_date="2025-01-07", oos_start_date="2025-01-02")
        raise AssertionError("duplicate control date must be blocked")
    except ValueError as exc:
        assert "重複" in str(exc)
    try:
        audit_coverage(_toy_control(), [source], end_date="2025-01-07", oos_start_date="2025-01-02",
                       trading_dates=pd.DatetimeIndex(["2024-12-31", "2025-01-02", "2025-01-06", "2025-01-07"]))
        raise AssertionError("canonical control date outside verified calendar must be blocked")
    except ValueError as exc:
        assert "已驗證台股交易日曆" in str(exc)
    sample_result = {"status": "PIT_REVIEW_REQUIRED", "cutoff": "2025-01-07", "coverage": _audit(source)}
    report = markdown_report(sample_result)
    assert "歷史逐年覆蓋" in report
    assert "模型 Input 授權：**NO**" in report
    return 5


def test_legacy_control_before_provider_calendar_is_scoped_not_fabricated():
    """1992 Research history cannot block a provable 2021 OOS Data Gate."""
    control = {
        "0050": pd.DatetimeIndex(["1992-01-04", "2020-12-30", "2021-01-04", "2021-01-05"]),
        "2330": pd.DatetimeIndex(["1992-01-06", "2021-01-04", "2021-01-05"]),
    }
    verified = pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
    source = pd.DataFrame([
        (ticker, day, 10, 100, 0, 10)
        for ticker in ("0050", "2330") for day in ("2020-12-31", "2021-01-04")
    ], columns=_toy_source().columns)
    audit = audit_coverage(control, [source], end_date="2021-01-05",
                           oos_start_date="2021-01-01", trading_dates=verified)
    scope = audit["calendar_scope"]
    assert scope["verified_calendar_first_date"] == "2020-12-30"
    assert scope["effective_start_date"] == "2020-12-30"
    assert scope["pre_calendar_control_date_count"] == 2
    assert scope["pre_calendar_control_stock_days"] == 2
    assert scope["pre_calendar_control_first_date"] == "1992-01-04"
    assert scope["pre_calendar_control_last_date"] == "1992-01-06"
    assert scope["pre_calendar_history_audited"] is False
    assert audit["oos_control_window"]["control_stock_days"] == 4
    assert audit["oos_control_window"]["both_ratio_valid"] == 4
    assert "TaiwanStockTradingDate" in audit["availability_rule"]
    report = markdown_report({"status": "PIT_REVIEW_REQUIRED", "cutoff": "2021-01-05", "coverage": audit})
    assert "未稽核／不納入覆蓋率分母" in report
    # A change on a score day cannot affect that day's previous-session input.
    mutated = pd.concat([source, pd.DataFrame([
        ("0050", "2021-01-05", 1000, 100, 0, 10)
    ], columns=source.columns)], ignore_index=True)
    other = audit_coverage(control, [mutated], end_date="2021-01-05",
                           oos_start_date="2021-01-01", trading_dates=verified)
    assert audit["oos_control_window"] == other["oos_control_window"]
    return 12


def test_verified_calendar_scope_never_masks_in_range_gaps():
    control = {"0050": pd.DatetimeIndex(["1992-01-04", "2021-01-04", "2021-01-05"])}
    source = _toy_source()
    expected = (
        (pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-05"]),
         None, "已驗證台股交易日曆涵蓋期間"),
        (pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-04"]),
         None, "已驗證台股交易日曆涵蓋期間"),
        (pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"]),
         "1992-01-04", "指定Audit起始日早於"),
        (pd.DatetimeIndex(["2021-01-01", "2021-01-04", "2021-01-05"]),
         None, "未涵蓋OOS開始前"),
    )
    for verified, begin, fragment in expected:
        try:
            audit_coverage(control, [source], end_date="2021-01-05", oos_start_date="2021-01-01",
                           trading_dates=verified, start_date=begin)
            raise AssertionError(f"Must block insufficient calendar evidence: {fragment}")
        except ValueError as exc:
            assert fragment in str(exc), str(exc)
    return 4


def main():
    tests = [test_pit_and_missing, test_no_future_lookahead, test_missing_and_zero_are_distinct,
             test_duplicate_and_invalid_dates, test_blank_and_negative_limits,
             test_multi_fragment_and_year_split, test_schema_control_and_report,
             test_legacy_control_before_provider_calendar_is_scoped_not_fabricated,
             test_verified_calendar_scope_never_masks_in_range_gaps,
             test_offline_report_integration, test_canonical_snapshot_view_sha]
    count = 0
    for test in tests:
        steps = test()
        print(f"PASS {test.__name__} ({steps})")
        count += steps
    print(f"PASS {len(tests)} scenarios, {count} assertions")



def test_offline_report_integration():
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    from core.file_integrity import load_json_strict
    from filters.breakout_quality.source_inventory import build_source_data_inventory
    from tools.audit import margin_short_pit_data_gate as gate

    with TemporaryDirectory(prefix="margin_pit_gate_") as scratch:
        root = Path(scratch)
        data_dir = root / "data/tw_stock_data_vip"
        data_dir.mkdir(parents=True)
        for ticker in ("0050", "2330"):
            pd.DataFrame({
                "Date": ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
                "Open": [10.0, 10.1, 10.2, 10.3],
                "High": [11.0, 11.1, 11.2, 11.3],
                "Low": [9.0, 9.1, 9.2, 9.3],
                "Close": [10.5, 10.6, 10.7, 10.8],
                "Volume": [100, 200, 100, 300],
            }).to_csv(data_dir / f"{ticker}.csv", index=False)
        summary = root / "outputs/filters/breakout_quality/breakout_quality_v1/dataset_summary.json"
        summary.parent.mkdir(parents=True)
        import json
        summary.write_text(json.dumps({"dataset":"full", "source_data_inventory":
                                       build_source_data_inventory(root, "full")}), encoding="utf-8")

        fake_artifact = SimpleNamespace(request_id="fixture", content_sha256="f"*64, row_count=6)
        fake_archive = SimpleNamespace(snapshot_fingerprint="1"*64, manifest_fingerprint="2"*64,
                                       as_of_date="2026-03-02")
        class FakeView:
            def __init__(self, **kwargs):
                pass
            def dataset_artifacts(self, dataset):
                assert dataset in (gate.DATASET, "TaiwanStockTradingDate")
                return (fake_artifact,)
            def iter_verified_frames(self, dataset, *, columns):
                assert dataset == "TaiwanStockTradingDate" and columns == ("date",)
                yield pd.DataFrame({"date": ["2024-12-31", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]})
            def iter_verified_frames_many_data_ids(self, dataset, *, columns, data_ids):
                assert dataset == gate.DATASET and "0050" in data_ids and "2330" in data_ids
                assert set(columns) == set(gate.REQUIRED_COLUMNS)
                yield _toy_source()
        with patch.object(gate, "load_ready_provider_snapshot_archive", return_value=fake_archive), \
             patch.object(gate, "ProviderSnapshotView", FakeView), \
             patch.object(gate, "BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE", "2025-01-01"):
            result, path = gate.run_data_gate(root)
        assert result["status"] == "PIT_REVIEW_REQUIRED"
        assert result["model_input_authorized"] is False
        assert result["coverage"]["oos_control_window"]["both_ratio_valid"] == 3
        assert load_json_strict(path / "data_gate.json")["provider_snapshot"]["fingerprint"] == "1"*64
        assert "資料檢查：**PIT_REVIEW_REQUIRED**" in (path / "data_gate.md").read_text(encoding="utf-8")
        assert "outputs/filters/breakout_quality/audits" in path.as_posix()
        # Missing provider evidence is an explicit blocker; no silent source fallback.
        with patch.object(gate, "load_ready_provider_snapshot_archive", side_effect=FileNotFoundError("missing")):
            no_source, blocked_path = gate.run_data_gate(root)
        assert no_source["status"] == "DATA_GATE_BLOCKED"
        assert no_source["model_input_authorized"] is False
        assert blocked_path.is_dir()
        return 9



def test_canonical_snapshot_view_sha():
    """Exercise the REAL neutral Snapshot View read path (without pyarrow)."""
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from core.file_integrity import compute_file_sha256
    from core.market_data_bootstrap_requests import BootstrapHttpRequest
    from core.market_data_storage_contract import resolve_market_data_request_parquet_path
    from services.downloader.market_data_ledger import LedgerCommittedArtifact
    from services.market_data.provider_snapshot_view import ProviderSnapshotView

    with TemporaryDirectory(prefix="margin_snapshot_sha_") as scratch:
        root = Path(scratch)
        manifest = "a" * 64
        requests = (
            BootstrapHttpRequest(dataset="TaiwanStockMarginPurchaseShortSale",
                                 bootstrap_mode="per_instrument_full_range", data_id="0050",
                                 start_date="2025-01-01", end_date="2025-01-08"),
            BootstrapHttpRequest(dataset="TaiwanStockTradingDate",
                                 bootstrap_mode="single_no_dates", data_id=None,
                                 start_date=None, end_date=None),
        )
        frames = (_toy_source().loc[lambda f: f.stock_id == "0050"],
                  pd.DataFrame({"date": ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]}))
        artifacts = []
        mapped = {}
        for ordinal, (request, frame) in enumerate(zip(requests, frames)):
            path = resolve_market_data_request_parquet_path(root, manifest, request)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"snapshot-{ordinal}".encode("ascii"))
            mapped[str(path)] = frame
            artifacts.append(LedgerCommittedArtifact(
                workload_id="fixture", request_id=request.request_id, ordinal=ordinal,
                dataset=request.dataset, bootstrap_mode=request.bootstrap_mode,
                data_id=request.data_id, start_date=request.start_date, end_date=request.end_date,
                row_count=len(frame), content_sha256=compute_file_sha256(path),
            ))
        archive = SimpleNamespace(artifacts=tuple(artifacts), manifest_fingerprint=manifest)
        view = ProviderSnapshotView(project_root=root, archive=archive,
                                    frame_reader=lambda p, columns: mapped[str(p)].loc[:, list(columns)].copy())
        assert len(view.dataset_artifacts("TaiwanStockMarginPurchaseShortSale")) == 1
        margin_rows = list(view.iter_verified_frames_many_data_ids(
            "TaiwanStockMarginPurchaseShortSale",
            columns=("date", "stock_id", "MarginPurchaseTodayBalance", "MarginPurchaseLimit",
                     "ShortSaleTodayBalance", "ShortSaleLimit"), data_ids=("0050",)))
        assert len(margin_rows) == 1 and len(margin_rows[0]) == 4
        official = list(view.iter_verified_frames("TaiwanStockTradingDate", columns=("date",)))
        assert len(official) == 1 and len(official[0]) == 4
        request_path = resolve_market_data_request_parquet_path(root, manifest, requests[0])
        request_path.write_bytes(b"tampered")
        try:
            list(view.iter_verified_frames_many_data_ids("TaiwanStockMarginPurchaseShortSale",
                 columns=("date", "stock_id"), data_ids=("0050",)))
            raise AssertionError("SHA tampering must fail closed")
        except ValueError as exc:
            assert "SHA256 drift" in str(exc)
    return 4


if __name__ == "__main__":
    main()

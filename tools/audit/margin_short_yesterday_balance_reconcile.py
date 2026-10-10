"""One-off, offline reconciliation of FinMind YesterdayBalance and saved TWSE 前日餘額.

No TWSE requests, FinMind API calls, provider modifications, model authorization,
training, or historical-vintage assertion. A freshly retrieved *historical* report
is CURRENT VINTAGE even if exact values agree with FinMind YesterdayBalance.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Mapping

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.research import RESEARCH_MARKET_DATA_CUTOFF
from core.file_integrity import atomic_write_json, atomic_write_text, canonical_json_sha256
from core.market_data_research_pit_contract import build_research_v2_pit_review_contracts
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.market_data.provider_snapshot_view import ProviderSnapshotView
from tools.audit import margin_short_official_publication_probe as official_probe

AUDIT_CONTRACT = "margin_short_finmind_yesterday_official_reconcile_v1"
OUTPUT_ROOT = Path("outputs/filters/breakout_quality/audits/margin_short_yesterday_balance_reconcile")
YESTERDAY_COLUMNS = ("MarginPurchaseYesterdayBalance", "ShortSaleYesterdayBalance")
REQUIRED_COLUMNS = (*official_probe.FINMIND_COLUMNS, *YESTERDAY_COLUMNS)
BASELINE_COUNT_KEYS = (
    "official_stock_count", "finmind_rows", "paired_count", "margin_equal",
    "short_equal", "both_equal", "mismatch_count", "finmind_missing_count", "finmind_invalid_count",
)


def _verified_previous_probe(root: Path, report_date: str, *, probe_fingerprint: str | None = None):
    """Load one *successful* persisted report; fail on ambiguous matches."""
    reports_root = (root / official_probe.OUTPUT_ROOT).resolve()
    if probe_fingerprint is not None:
        if len(probe_fingerprint) != 64 or any(c not in "0123456789abcdef" for c in probe_fingerprint):
            raise ValueError("指定的probe fingerprint不是小寫64位hex")
        paths = [reports_root / probe_fingerprint / "publication_probe.json"]
    else:
        paths = sorted(reports_root.glob("*/publication_probe.json"))
    candidates = []
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(reports_root):
            continue
        if path.stat().st_size > 5_000_000:
            raise ValueError("既有official probe JSON超過上限")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or not isinstance(report.get("samples"), list):
            raise ValueError("既有official probe JSON格式錯誤")
        found = [item for item in report["samples"] if isinstance(item, dict)
                 and item.get("report_date") == report_date and item.get("status") == "OK"]
        if found:
            if len(found) != 1:
                raise ValueError("同一既有官方probe有重複成功報表日期")
            candidates.append((path, report, found[0]))
    if len(candidates) != 1:
        raise ValueError(f"已保存成功官方報表數量不為1 ({len(candidates)})；不重新下載；若多筆請指定--probe-fingerprint")
    path, report, sample = candidates[0]
    if report.get("audit_contract") != official_probe.AUDIT_ID:
        raise ValueError("既有官方probe contract不一致")
    if report.get("research_cutoff") != str(RESEARCH_MARKET_DATA_CUTOFF):
        raise ValueError("既有官方probe Research截止日不同")
    if report.get("model_input_authorized") is not False or report.get("historical_contemporaneous_vintage_proven") is not False:
        raise ValueError("既有官方probe授權／vintage欄位不可冒稱已通過")
    if report.get("reports_acquisition") != "public_historical_requery":
        raise ValueError("既有官方probe不是public historical requery")
    if sample.get("source_url") != official_probe._url(report_date):
        raise ValueError("既有官方URL不符固定TWSE來源")
    if sample.get("source_vintage_grade") != "CURRENT_VINTAGE_REQUERY_ONLY":
        raise ValueError("既有官方數值版本等級不符")
    identity = {"contract": official_probe.AUDIT_ID,
                "snapshot": report.get("provider_snapshot_fingerprint"),
                "dates": report.get("sample_dates"),
                "capture": [(r.get("report_date"), r.get("raw_sha256"), r.get("capture_utc"), r.get("status"))
                            for r in report["samples"]]}
    failures = [(r.get("report_date"), r.get("error_kind"), r.get("error"), r.get("tls_validation_mode"))
                for r in report["samples"] if r.get("error")]
    if failures:
        identity["source_failure_evidence"] = failures
    if canonical_json_sha256(identity) != report.get("audit_fingerprint") or path.parent.name != report.get("audit_fingerprint"):
        raise ValueError("既有官方probe fingerprint不一致")
    capture = path.parent / "captured_requeries" / (report_date.replace("-", "") + ".json")
    raw = capture.read_bytes()
    if len(raw) > 12_000_000 or sha256(raw).hexdigest() != sample.get("raw_sha256"):
        raise ValueError("已保存官方JSON raw SHA256不一致")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("已保存官方JSON不是object")
    official = official_probe.parse_twse_mi_margn(payload, report_date=report_date)
    if len(official) != sample.get("official_ticker_count"):
        raise ValueError("官方JSON重新解析的股票筆數與已保存報告不同")
    return report, sample, official


def _columns_for_comparison(frame: pd.DataFrame, source_date: str, *, confirmed_date: str,
                            use_yesterday: bool) -> pd.DataFrame:
    """Map SAME-DATE YesterdayBalance to old comparator's column names, with explicit date scope."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"FinMind READY Snapshot缺少YesterdayBalance必要欄位: {sorted(missing)}")
    subset = frame.loc[frame["date"].astype("string") == source_date].copy()
    if pd.api.types.is_numeric_dtype(subset["stock_id"]):
        raise ValueError("FinMind stock_id為數值型態，可能丟失前導零")
    if use_yesterday:
        subset["MarginPurchaseTodayBalance"] = subset["MarginPurchaseYesterdayBalance"]
        subset["ShortSaleTodayBalance"] = subset["ShortSaleYesterdayBalance"]
        subset["date"] = confirmed_date
    return subset.loc[:, official_probe.FINMIND_COLUMNS]


def compare_existing_official_yesterday(
    official: Mapping[str, tuple[int, int]], frames: pd.DataFrame,
    *, report_date: str, balance_date: str, prior_comparison: Mapping[str, object],
) -> dict[str, object]:
    """Same report TWSE Yesterday vs FinMind same-date Yesterday; no tolerant matching."""
    original_today = official_probe.compare_official_confirmed_balances(
        official, _columns_for_comparison(frames, balance_date, confirmed_date=balance_date,
                                          use_yesterday=False), balance_date=balance_date)
    for key in BASELINE_COUNT_KEYS:
        if original_today.get(key) != prior_comparison.get(key):
            raise ValueError(f"已保存T-2 TodayBalance比較的{key}與目前READY Snapshot不符")
    if original_today["sample_mismatches"] != prior_comparison.get("sample_mismatches"):
        raise ValueError("已保存5筆差異與目前READY Snapshot不一致，不可假裝已修復")
    confirmed_yesterday = official_probe.compare_official_confirmed_balances(
        official, _columns_for_comparison(frames, report_date, confirmed_date=balance_date,
                                          use_yesterday=True), balance_date=balance_date)
    previous_mismatches = {item["ticker"]: item for item in original_today["sample_mismatches"]}
    yesterday_frame = frames.loc[frames["date"].astype("string") == report_date].copy()
    if not yesterday_frame.empty:
        yesterday_frame["stock_id"] = yesterday_frame["stock_id"].astype("string").str.strip()
    lookup = yesterday_frame.set_index("stock_id")
    details = []
    for ticker, old in sorted(previous_mismatches.items()):
        if ticker not in lookup.index:
            details.append({"ticker": ticker, "previous_margin_delta": int(old["official_margin_previous"] - old["finmind_margin_t_minus_two"]),
                            "finmind_report_day_yesterday_margin": None, "reconciled": False,
                            "reason": "MISSING_REPORT_DATE_SOURCE_ROW"})
            continue
        source = lookup.loc[ticker]
        try:
            value = float(pd.to_numeric(source["MarginPurchaseYesterdayBalance"], errors="raise"))
            if not pd.notna(value) or not (0 <= value < float("inf")):
                raise ValueError("invalid")
        except (ValueError, TypeError, OverflowError):
            details.append({"ticker": ticker, "previous_margin_delta": int(old["official_margin_previous"] - old["finmind_margin_t_minus_two"]),
                            "finmind_report_day_yesterday_margin": None, "reconciled": False,
                            "reason": "INVALID_YESTERDAY_SOURCE_VALUE"})
            continue
        details.append({"ticker": ticker,
                        "previous_margin_delta": int(old["official_margin_previous"] - old["finmind_margin_t_minus_two"]),
                        "finmind_report_day_yesterday_margin": value,
                        "official_margin_previous": int(old["official_margin_previous"]),
                        "new_margin_delta": float(old["official_margin_previous"] - value),
                        "reconciled": bool(value == old["official_margin_previous"]),
                        "reason": "MATCHED" if value == old["official_margin_previous"] else "STILL_DIFFERENT"})
    return {
        "old_confirmed_date_today": original_today,
        "new_report_date_yesterday": confirmed_yesterday,
        "previous_mismatch_details": details,
        "previous_mismatch_reconciled": sum(int(item["reconciled"]) for item in details),
        "previous_mismatch_count": original_today["mismatch_count"],
    }


def markdown_report(result: Mapping[str, object]) -> str:
    lines = ["# FinMind YesterdayBalance 同日 TWSE 確認餘額對照（唯讀）", "",
             f"- 狀態：**{result['status']}**；模型 Input 授權：**NO**",
             "- 資料證據：FinMind READY Snapshot + 已保存 TWSE current-vintage historical requery；不重新下載",
             "- 即使數值完全一致，仍不證明當時首發／修訂版本，**PIT_REVIEW_REQUIRED**。", ""]
    if result.get("block_reason"):
        lines.extend(["## 阻擋原因", "", str(result["block_reason"]), ""])
    c = result.get("comparison")
    if isinstance(c, dict):
        o, n = c["old_confirmed_date_today"], c["new_report_date_yesterday"]
        lines.extend(["## 同日欄位核對", "",
                      f"- 官方報表日 T−1：`{result['report_date']}`；確認餘額日 T−2：`{result['confirmed_balance_date']}`；預測日 T：`{result['prediction_date']}`。",
                      "- 舊比對：FinMind **T−2 的 TodayBalance**；新比對：FinMind **T−1 的 YesterdayBalance**。",
                      "- 數值依相同數字單位假設直接比對；單位未獨立證實，且無容忍誤差、補值或欄位日期推測。", "",
                      "| 指標 | 舊 T−2 TodayBalance | 新 T−1 YesterdayBalance |", "|---|---:|---:|",
                      f"| 可比股票 | {o['paired_count']:,} | {n['paired_count']:,} |",
                      f"| 融資一致 | {o['margin_equal']:,} | {n['margin_equal']:,} |",
                      f"| 融券一致 | {o['short_equal']:,} | {n['short_equal']:,} |",
                      f"| 雙欄一致 | {o['both_equal']:,} | {n['both_equal']:,} |", "",
                      "## 先前差異逐筆結果", "",
                      "| 代碼 | 舊融資差額（官方−FinMind） | 新報表日 YesterdayBalance | 新融資差額 | 判定 |",
                      "|---|---:|---:|---:|---|"])
        for item in c["previous_mismatch_details"]:
            value = item.get("finmind_report_day_yesterday_margin")
            delta = item.get("new_margin_delta")
            lines.append(f"| {item['ticker']} | {item['previous_margin_delta']:+d} | "
                         f"{value if value is not None else 'N/A'} | "
                         f"{delta if delta is not None else 'N/A'} | {item['reason']} |")
        lines.extend(["", f"- 先前差異已消除：**{c['previous_mismatch_reconciled']}/{c['previous_mismatch_count']}**。", ""])
    lines.extend(["## 研究邊界", "", "- 此工具沒有打TWSE或FinMind API，沒有重算AO、模型學習、Target或Trading。",
                  "- 只限TWSE上市及這個官方報表日；TPEx／早期Train覆蓋／歷史當時不可變版本未證明。",
                  "- 不加入限額比率，限額的effective-date仍未驗證。",
                  "- 本工具結果不得修改原Provider Snapshot或Research PIT authorization。", ""])
    return "\n".join(lines)


def run_reconcile(root: Path, *, report_date: str = "2024-02-01",
                  probe_fingerprint: str | None = None) -> tuple[dict[str, object], Path]:
    """Single bounded offline check; writes only a new one-off Audit report."""
    root = Path(root).resolve()
    result: dict[str, object] = {
        "audit_contract": AUDIT_CONTRACT, "status": "NOT_RUN",
        "model_input_authorized": False, "research_pit_review_status": "REVIEW_REQUIRED",
        "historical_contemporaneous_vintage_proven": False,
        "research_cutoff": str(RESEARCH_MARKET_DATA_CUTOFF),
        "provider_scope": "TWSE_LISTED_ONLY", "report_date": report_date,
        "official_evidence_type": "CURRENT_VINTAGE_REQUERY_ONLY", "finmind_source": "READY_PROVIDER_SNAPSHOT",
        "unit_equivalence_independently_proven": False,
        "external_requests": 0,
    }
    try:
        stamp = pd.Timestamp(report_date).strftime("%Y-%m-%d")
        if stamp != report_date:
            raise ValueError("報表日期必須是YYYY-MM-DD")
        official_report, sample, official = _verified_previous_probe(root, report_date,
                                                                      probe_fingerprint=probe_fingerprint)
        archive = load_ready_provider_snapshot_archive(
            root, snapshot_fingerprint=official_report["provider_snapshot_fingerprint"])
        if archive.as_of_date < str(RESEARCH_MARKET_DATA_CUTOFF):
            raise ValueError("Provider Snapshot research cutoff不足")
        canonical = next((c for c in build_research_v2_pit_review_contracts()
                          if c.dataset == official_probe.DATASET), None)
        if canonical is None:
            raise ValueError("Research V2缺MarginShort PIT contract")
        result["research_pit_review_status"] = canonical.review_status
        if canonical.review_status != "REVIEW_REQUIRED":
            raise ValueError("本專題只驗證既有REVIEW_REQUIRED契約，不提供新授權")
        view = ProviderSnapshotView(project_root=root, archive=archive)
        if not view.dataset_artifacts(official_probe.DATASET) or not view.dataset_artifacts(official_probe.CALENDAR_DATASET):
            raise ValueError("READY Provider Snapshot缺個股融資融券或交易日曆")
        alignment = official_probe.forecast_alignment(official_probe._verified_trading_days(view),
                                                      report_date, cutoff=str(RESEARCH_MARKET_DATA_CUTOFF))
        if sample.get("alignment") != alignment:
            raise ValueError("既有官方probe的T−2/T−1/T日期與已驗證TradingDate不一致")
        result.update({"official_probe_fingerprint": official_report["audit_fingerprint"],
                       "official_raw_sha256": sample["raw_sha256"],
                       "provider_snapshot_fingerprint": archive.snapshot_fingerprint,
                       "confirmed_balance_date": alignment["confirmed_balance_date"],
                       "prediction_date": alignment["prediction_date"]})
        frames = []
        wanted_dates = {alignment["confirmed_balance_date"], report_date}
        for frame in view.iter_verified_frames_many_data_ids(
            official_probe.DATASET, columns=REQUIRED_COLUMNS, data_ids=tuple(sorted(official))
        ):
            if not set(REQUIRED_COLUMNS).issubset(frame.columns):
                raise ValueError("FinMind來源缺YesterdayBalance欄位")
            if pd.api.types.is_numeric_dtype(frame["stock_id"]):
                raise ValueError("FinMind來源stock_id已變為數字，拒絕丟失前導零")
            part = frame.loc[frame["date"].astype("string").isin(wanted_dates)].copy()
            if not part.empty:
                if not part["stock_id"].astype("string").str.fullmatch(r"[0-9A-Z]{4,8}").all():
                    raise ValueError("FinMind來源包含不合法ticker")
                frames.append(part)
        source = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=REQUIRED_COLUMNS)
        if source.loc[:, ["stock_id", "date"]].duplicated().any():
            raise ValueError("FinMind兩日股票＋日期主鍵重複；不可自行挑選")
        result["comparison"] = compare_existing_official_yesterday(
            official, source, report_date=report_date, balance_date=alignment["confirmed_balance_date"],
            prior_comparison=sample["comparison"])
        result["status"] = "CURRENT_VINTAGE_NUMERIC_RECONCILIATION_AVAILABLE_PIT_BLOCKED"
        result["block_reason"] = "歷史當時首次公布／修訂vintage仍未證實；數值相等不能自動授權DL input"
    except (OSError, KeyError, TypeError, ValueError, UnicodeError, RuntimeError) as exc:
        result["status"] = "RECONCILIATION_BLOCKED"
        result["block_reason"] = f"{type(exc).__name__}: {exc}"
    fingerprint = canonical_json_sha256({
        "contract": AUDIT_CONTRACT, "official_probe": result.get("official_probe_fingerprint"),
        "snapshot": result.get("provider_snapshot_fingerprint"), "date": report_date,
        "status": result["status"], "comparison": result.get("comparison"),
        "block_reason": result.get("block_reason"),
    })
    result["audit_fingerprint"] = fingerprint
    out = root / OUTPUT_ROOT / fingerprint
    atomic_write_json(out / "reconcile.json", result)
    atomic_write_text(out / "reconcile.md", markdown_report(result))
    return result, out


def main() -> int:
    parser = argparse.ArgumentParser(description="使用已保存TWSE報表與FinMind YesterdayBalance完成離線對照；不授權模型")
    parser.add_argument("--project-root", type=Path, default=_ROOT)
    parser.add_argument("--report-date", default="2024-02-01")
    parser.add_argument("--probe-fingerprint", default=None)
    args = parser.parse_args()
    result, folder = run_reconcile(args.project_root, report_date=args.report_date,
                                   probe_fingerprint=args.probe_fingerprint)
    print(f"YesterdayBalance離線對照：{result['status']} | 模型Input授權：NO")
    if result.get("comparison"):
        before = result["comparison"]["old_confirmed_date_today"]
        after = result["comparison"]["new_report_date_yesterday"]
        print(f"TWSE可比股票 {after['paired_count']} | 融資相等 {before['margin_equal']}→{after['margin_equal']}"
              f" | 融券相等 {before['short_equal']}→{after['short_equal']}"
              f" | 先前差異消除 {result['comparison']['previous_mismatch_reconciled']}/{result['comparison']['previous_mismatch_count']}")
    elif result.get("block_reason"):
        print(f"阻擋：{result['block_reason']}")
    print(f"報告：{(folder / 'reconcile.md').resolve()}")
    return 0 if result["status"] == "CURRENT_VINTAGE_NUMERIC_RECONCILIATION_AVAILABLE_PIT_BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

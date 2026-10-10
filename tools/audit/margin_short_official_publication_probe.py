"""One-off TWSE official historical-report / FinMind T-2 balance probe.

A historical query made TODAY is a current-vintage reconstruction. Neither a
TWSE report date, an HTTP response Date header nor this tool's capture time
proves the returned bytes were available unchanged before an earlier forecast.
This read-only research probe NEVER authorizes training or changes the Research
V2 PIT contract. TPEx and next-day credit LIMIT semantics are not validated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import os
import math
import sys
import tempfile
from typing import Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.research import RESEARCH_MARKET_DATA_CUTOFF
from core.file_integrity import atomic_replace_with_retry, atomic_write_json, atomic_write_text, canonical_json_sha256
from core.market_data_research_pit_contract import build_research_v2_pit_review_contracts
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.market_data.provider_snapshot_view import ProviderSnapshotView

DATASET = "TaiwanStockMarginPurchaseShortSale"
CALENDAR_DATASET = "TaiwanStockTradingDate"
AUDIT_ID = "margin_short_twse_confirmed_balance_probe_v1"
OUTPUT_ROOT = Path("outputs/filters/breakout_quality/audits/margin_short_official_publication_probe")
OFFICIAL_BASE = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
FINMIND_COLUMNS = ("stock_id", "date", "MarginPurchaseTodayBalance", "ShortSaleTodayBalance")
_PREVIOUS_LABEL = "前日餘額"
_TICKER_PATTERN = re.compile(r"^[0-9A-Z]{4,8}$")
_TAG = re.compile(r"<[^>]+>")
_ROC_DATE = re.compile(r"(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_ISO_DATE = re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b")


def _clean(value: object) -> str:
    return html.unescape(_TAG.sub("", str(value))).replace("\xa0", " ").strip()


def _number(value: object) -> int:
    cleaned = _clean(value).replace(",", "")
    if not cleaned or not re.fullmatch(r"[0-9]+", cleaned):
        raise ValueError(f"官方前日餘額不是有效非負整數: {value!r}")
    return int(cleaned)


def _reported_dates(payload: Mapping[str, object]) -> set[str]:
    """Extract only an explicit official date, never a user-supplied URL date."""
    dates: set[str] = set()
    for value in (payload.get("date"), payload.get("title")):
        text = _clean(value) if value is not None else ""
        if re.fullmatch(r"\d{8}", text):
            dates.add(pd.Timestamp(text).strftime("%Y-%m-%d"))
        for roc_y, month, day in _ROC_DATE.findall(text):
            dates.add(f"{int(roc_y)+1911:04d}-{int(month):02d}-{int(day):02d}")
        for year, month, day in _ISO_DATE.findall(text):
            dates.add(f"{int(year):04d}-{int(month):02d}-{int(day):02d}")
    for table in payload.get("tables", []) if isinstance(payload.get("tables"), list) else []:
        if not isinstance(table, dict):
            continue
        for roc_y, month, day in _ROC_DATE.findall(_clean(table.get("title", ""))):
            dates.add(f"{int(roc_y)+1911:04d}-{int(month):02d}-{int(day):02d}")
    return dates


def _table_candidates(payload: Mapping[str, object]) -> Iterable[tuple[list[object], list[object]]]:
    for table in payload.get("tables", []) if isinstance(payload.get("tables"), list) else []:
        if isinstance(table, dict) and isinstance(table.get("fields"), list) and isinstance(table.get("data"), list):
            yield table["fields"], table["data"]
    for name, data in payload.items():
        if name.startswith("data") and isinstance(data, list):
            fields = payload.get("fields" + name[4:])
            if isinstance(fields, list):
                yield fields, data


def parse_twse_mi_margn(payload: Mapping[str, object], *, report_date: str) -> dict[str, tuple[int, int]]:
    """Read TWSE *individual securities*, not the all-market summary table."""
    expected = pd.Timestamp(report_date).strftime("%Y-%m-%d")
    published_dates = _reported_dates(payload)
    if expected not in published_dates:
        raise ValueError(f"TWSE回應沒有可核對的指定報表日期: {expected}; response_dates={sorted(published_dates)}")
    if str(payload.get("stat", "OK")).upper() not in ("OK", "SUCCESS"):
        raise ValueError(f"TWSE報表非正常狀態: {payload.get('stat')}")
    valid_tables: list[dict[str, tuple[int, int]]] = []
    for raw_fields, raw_rows in _table_candidates(payload):
        fields = [_clean(x) for x in raw_fields]
        previous_cols = [i for i, name in enumerate(fields) if name == _PREVIOUS_LABEL]
        if len(previous_cols) != 2 or len(fields) < 14 or not any("代號" in s for s in fields):
            continue  # e.g. the 6-column summary, not per-stock evidence.
        if previous_cols[0] >= previous_cols[1]:
            raise ValueError("TWSE融資／融券欄位次序不明")
        ticker_pos = next(i for i, name in enumerate(fields) if "代號" in name)
        rows: dict[str, tuple[int, int]] = {}
        for row in raw_rows:
            if not isinstance(row, list) or len(row) != len(fields):
                raise ValueError("TWSE個股報表欄位數量不符合schema；拒絕猜測欄位位置")
            ticker = _clean(row[ticker_pos]).upper()
            if not _TICKER_PATTERN.fullmatch(ticker):
                raise ValueError(f"TWSE個股代碼不合法: {ticker!r}")
            if ticker in rows:
                raise ValueError(f"TWSE官方同報表重複個股: {ticker}")
            rows[ticker] = (_number(row[previous_cols[0]]), _number(row[previous_cols[1]]))
        if rows:
            valid_tables.append(rows)
    if len(valid_tables) != 1:
        raise ValueError(f"TWSE單一融資融券個股報表解析數量不為1: {len(valid_tables)}")
    return valid_tables[0]


def forecast_alignment(trading_days: pd.DatetimeIndex, report_date: str, *, cutoff: str) -> dict[str, str]:
    """For prediction T, TWSE T-1 report's 前日餘額 belongs to T-2."""
    days = pd.DatetimeIndex(trading_days).normalize().sort_values()
    if days.empty or days.hasnans or days.has_duplicates:
        raise ValueError("已驗證台股交易日曆不可為空／重複")
    stamp = pd.Timestamp(report_date).normalize()
    pos = days.searchsorted(stamp)
    if pos == 0 or pos >= len(days) - 1 or days[pos] != stamp:
        raise ValueError("官方報表日期缺T-2來源日或下一個預測日，或不是已驗證交易日")
    prediction = days[pos + 1]
    if prediction > pd.Timestamp(cutoff).normalize():
        raise ValueError("官方報表所對應的預測日超出Research cutoff")
    return {
        "confirmed_balance_date": days[pos - 1].strftime("%Y-%m-%d"),
        "official_report_date": days[pos].strftime("%Y-%m-%d"),
        "prediction_date": prediction.strftime("%Y-%m-%d"),
        "information_rule": "official_(T-1)_previous_day_balance_is_T-2; use_before_T_open_only_if_vintage_proven",
    }


def compare_official_confirmed_balances(
    official: Mapping[str, tuple[int, int]], finmind: pd.DataFrame,
    *, balance_date: str,
) -> dict[str, object]:
    """Numeric agreement is diagnostic only and never a historical vintage proof.

    Assumes the same number unit for both columns (not silently converted).
    The audit explicitly reports this as a comparison-unit hypothesis.
    """
    needed = set(FINMIND_COLUMNS)
    if not needed.issubset(finmind.columns):
        raise ValueError(f"FinMind欄位不足: {sorted(needed - set(finmind.columns))}")
    source = finmind.loc[finmind["date"].astype("string") == balance_date].copy()
    if pd.api.types.is_numeric_dtype(source["stock_id"]):
        raise ValueError("FinMind stock_id已轉數字，會遺失前導零")
    source["stock_id"] = source["stock_id"].astype("string").str.strip()
    if source["stock_id"].duplicated().any():
        raise ValueError("FinMind同一天個股主鍵重複，不能任選一筆")
    lookup = source.set_index("stock_id")
    match = {"official_stock_count": len(official), "finmind_rows": len(source),
             "paired_count": 0, "margin_equal": 0, "short_equal": 0,
             "both_equal": 0, "mismatch_count": 0,
             "finmind_missing_count": 0, "finmind_invalid_count": 0,
             "sample_mismatches": []}
    for ticker, (margin, short) in official.items():
        if ticker not in lookup.index:
            match["finmind_missing_count"] += 1
            continue
        row = lookup.loc[ticker]
        try:
            source_margin = pd.to_numeric(row["MarginPurchaseTodayBalance"], errors="raise")
            source_short = pd.to_numeric(row["ShortSaleTodayBalance"], errors="raise")
        except (TypeError, ValueError):
            match["finmind_invalid_count"] += 1
            continue
        if (pd.isna(source_margin) or pd.isna(source_short)
                or not math.isfinite(float(source_margin)) or not math.isfinite(float(source_short))
                or source_margin < 0 or source_short < 0):
            match["finmind_invalid_count"] += 1
            continue
        match["paired_count"] += 1
        m_eq, s_eq = source_margin == margin, source_short == short
        match["margin_equal"] += int(m_eq)
        match["short_equal"] += int(s_eq)
        match["both_equal"] += int(m_eq and s_eq)
        if not (m_eq and s_eq):
            match["mismatch_count"] += 1
            if len(match["sample_mismatches"]) < 12:
                match["sample_mismatches"].append({"ticker": ticker,
                    "official_margin_previous": margin, "finmind_margin_t_minus_two": float(source_margin),
                    "official_short_previous": short, "finmind_short_t_minus_two": float(source_short)})
    return match


def _sample_dates(days: pd.DatetimeIndex, cutoff: str) -> list[str]:
    """Pre-declared yearwise date selection; no OOS label/score/feature search."""
    eligible = days[(days >= pd.Timestamp("2021-01-01")) & (days < pd.Timestamp(cutoff))]
    choices: list[str] = []
    for year in sorted(set(eligible.year)):
        year_days = eligible[eligible.year == year]
        # Early February avoids cutting 2026 OOS short and has a prior session.
        after = year_days[year_days >= pd.Timestamp(f"{year}-02-01")]
        if len(after):
            choices.append(after[0].strftime("%Y-%m-%d"))
    return choices


def _url(date: str) -> str:
    return OFFICIAL_BASE + "?" + urlencode({"date": date.replace("-", ""),
                                             "selectType": "ALL", "response": "json"})


def _fetch_official(date: str) -> tuple[bytes, str]:
    target = _url(date)
    req = Request(target, headers={"User-Agent": "Mozilla/5.0 (Research-PIT-evidence; read-only)",
                                   "Accept": "application/json"})
    with urlopen(req, timeout=20) as response:
        if response.url.split("?")[0] != OFFICIAL_BASE:
            raise ValueError("官方報表非允許的TWSE URL")
        raw = response.read(12_000_001)
    if len(raw) > 12_000_000:
        raise ValueError("TWSE報表超出上限")
    return raw, target


def _write_exact_capture(target: Path, raw: bytes) -> None:
    """Keep the actual response bytes, not a reserialized JSON approximation."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tempname = tempfile.mkstemp(prefix=".twse_capture_", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace_with_retry(tempname, target)
    finally:
        if os.path.exists(tempname):
            os.unlink(tempname)


def _verified_trading_days(view: ProviderSnapshotView) -> pd.DatetimeIndex:
    parts = list(view.iter_verified_frames(CALENDAR_DATASET, columns=("date",)))
    if not parts or any("date" not in part.columns for part in parts):
        raise ValueError("未取得SHA驗證台股TradingDate資料集")
    dates = pd.to_datetime(pd.concat([p["date"] for p in parts], ignore_index=True),
                           errors="coerce", format="%Y-%m-%d")
    if dates.isna().any():
        raise ValueError("Provider交易日曆含不合法日期")
    return pd.DatetimeIndex(dates.dt.normalize().unique()).sort_values()


def markdown_report(result: Mapping[str, object]) -> str:
    lines = ["# 融資融券官方確認餘額來源／歷史Vintage稽核（一次性）", "",
             f"- 狀態：**{result['status']}**；模型 Input 授權：**NO**",
             f"- 審查對象：`{DATASET}`；Research截止：`{result['research_cutoff']}`",
             "- 方法：T-1官方個股報表的「前日餘額」代表T-2；T日盤前僅可使用已發布報表。",
             "- **歷史查詢於今日取得的JSON，不是當時原始發布版本。** 捕獲時間只證明本次取得時間，不證明歷史首次發布或當時內容。",
             "- 本稽核僅查TWSE上市證券；TPEx上櫃、限額適用日／使用率、完整OOS適用性均未通過本稽核。",
             "- 官方與FinMind比較採同數字單位假設；一致率非最終單位確認或PIT授權。", "",
             "## 取樣結果（固定按年份，非依OOS表現選擇）", "",
             "| 官方日期T-1 | 確認餘額日T-2 | 預測日T | TWSE個股 | 可比較 | 融資一致 | 融券一致 | 雙欄一致 | 證據層級 |",
             "|---|---|---|---:|---:|---:|---:|---:|---|" ]
    for entry in result.get("samples", []):
        if entry.get("status") == "OK":
            a, match = entry["alignment"], entry["comparison"]
            lines.append(f"| {a['official_report_date']} | {a['confirmed_balance_date']} | {a['prediction_date']} | "
                         f"{match['official_stock_count']} | {match['paired_count']} | {match['margin_equal']} | "
                         f"{match['short_equal']} | {match['both_equal']} | CURRENT-VINTAGE REQUERY |")
        else:
            lines.append(f"| {entry['report_date']} | — | — | — | — | — | — | — | {entry['status']} |")
    lines.extend(["", "## 資料缺口與正式Gate", "",
                  f"- 成功取樣：{result.get('sample_success', 0)}/{len(result.get('sample_dates', []))}。",
                  "- 取樣只可確認報表欄位語意與最新歷史數值的局部對照；不可推論完整歷史coverage或不可變版本。",
                  "- T-1報表的『次一營業日限額』與T-2確認餘額不是同一effective-date，不能自動組成T-2使用率。",
                  "- 若要嚴格PIT，必須有可稽核的當年發布檔及不晚於預測T盤前的第三方保存／可信交付時間，"
                  "並驗證TWSE、TPEx、各歷史期間及所需欄位；歷史網站重查不足以代替。",
                  "- 正式Research V2仍為PIT_REVIEW_REQUIRED；不得修改AO、模型輸入、Target、checkpoint或Trading。", ""])
    return "\n".join(lines)


def run_probe(root: Path, *, snapshot_fingerprint: str | None = None,
              dates: list[str] | None = None, reports_dir: Path | None = None,
              fetch_public: bool = False) -> tuple[dict[str, object], Path]:
    """Read old snapshot; optionally capture a few contemporary *requeries*."""
    root = Path(root).resolve()
    cutoff = str(RESEARCH_MARKET_DATA_CUTOFF)
    canonical = next(c for c in build_research_v2_pit_review_contracts() if c.dataset == DATASET)
    result: dict[str, object] = {
        "audit_contract": AUDIT_ID, "research_cutoff": cutoff,
        "status": "NOT_RUN", "model_input_authorized": False,
        "research_pit_review_status": canonical.review_status,
        "historical_contemporaneous_vintage_proven": False,
        "provider_scope": "TWSE_LISTED_ONLY", "limits_effective_date_proven": False,
        "reports_acquisition": "public_historical_requery" if fetch_public else "local_untrusted_historical_files",
        "samples": [], "sample_dates": [], "sample_success": 0,
    }
    raw_captures: dict[str, bytes] = {}
    try:
        archive = load_ready_provider_snapshot_archive(root, snapshot_fingerprint=snapshot_fingerprint)
        view = ProviderSnapshotView(project_root=root, archive=archive)
        if archive.as_of_date < cutoff:
            raise ValueError("Provider Snapshot截止日不足")
        if not view.dataset_artifacts(DATASET) or not view.dataset_artifacts(CALENDAR_DATASET):
            raise ValueError("Provider Snapshot缺融資融券或正式交易日曆")
        calendar = _verified_trading_days(view)
        selected_dates = dates if dates is not None else _sample_dates(calendar, cutoff)
        if not selected_dates:
            raise ValueError("無可取樣的交易日期")
        if len(selected_dates) > 16 or len(selected_dates) != len(set(selected_dates)):
            raise ValueError("專題稽核取樣最多16個不重複交易日")
        alignments = {stamp: forecast_alignment(calendar, stamp, cutoff=cutoff) for stamp in selected_dates}
        result["sample_dates"] = selected_dates
        result["provider_snapshot_fingerprint"] = archive.snapshot_fingerprint
        result["trading_calendar_fingerprint"] = canonical_json_sha256([
            (a.request_id, a.content_sha256, a.row_count) for a in view.dataset_artifacts(CALENDAR_DATASET)])
        official_items: list[tuple[str, dict[str, tuple[int, int]], dict[str, object]]] = []
        for report_date in selected_dates:
            entry: dict[str, object] = {"report_date": report_date, "status": "NOT_RUN"}
            result["samples"].append(entry)
            try:
                if fetch_public:
                    raw, url = _fetch_official(report_date)
                elif reports_dir is not None:
                    path = Path(reports_dir) / (pd.Timestamp(report_date).strftime("%Y%m%d") + ".json")
                    raw, url = path.read_bytes(), "local_file_unverified_vintage:" + path.name
                else:
                    entry["status"] = "NO_HISTORICAL_OFFICIAL_RELEASE_INPUT"
                    continue
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("官方JSON根節點不是object")
                rows = parse_twse_mi_margn(payload, report_date=report_date)
                entry.update({"status": "PARSED_PENDING_COMPARE", "capture_utc": datetime.now(timezone.utc).isoformat(),
                              "source_url": url, "raw_sha256": sha256(raw).hexdigest(),
                              "source_vintage_grade": "CURRENT_VINTAGE_REQUERY_ONLY",
                              "official_ticker_count": len(rows)})
                official_items.append((report_date, rows, entry))
                raw_captures[report_date] = raw
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                safe = str(exc).replace(str(root) + "/", "").replace(str(root) + "\\", "")
                if reports_dir is not None:
                    safe = safe.replace(str(Path(reports_dir).resolve()) + "/", "")
                    safe = safe.replace(str(Path(reports_dir).resolve()) + "\\", "")
                entry.update({"status": "SOURCE_NOT_VERIFIABLE", "error": f"{type(exc).__name__}: {safe}"})
        if official_items:
            source_dates = {alignments[date]["confirmed_balance_date"] for date, _, _ in official_items}
            source_ids = {ticker for _, rows, _ in official_items for ticker in rows}
            frames: list[pd.DataFrame] = []
            for part in view.iter_verified_frames_many_data_ids(
                DATASET, columns=FINMIND_COLUMNS, data_ids=tuple(sorted(source_ids))):
                if not set(FINMIND_COLUMNS).issubset(part.columns):
                    raise ValueError("FinMind來源缺必要餘額欄位")
                sub = part.loc[part["date"].astype("string").isin(source_dates)]
                if len(sub):
                    frames.append(sub.loc[:, FINMIND_COLUMNS].copy())
            finmind = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FINMIND_COLUMNS)
            for date, rows, entry in official_items:
                entry["alignment"] = alignments[date]
                entry["comparison"] = compare_official_confirmed_balances(
                    rows, finmind, balance_date=alignments[date]["confirmed_balance_date"])
                if entry["comparison"]["paired_count"]:
                    entry["status"] = "OK"
                    result["sample_success"] += 1
                else:
                    entry["status"] = "NO_COMPARABLE_FINMIND_ROWS"
        result["status"] = ("HISTORICAL_VINTAGE_NOT_PROVEN" if result["sample_success"]
                            else "NO_OFFICIAL_COMPARISON_EVIDENCE")
        result["block_reason"] = (
            "即使歷史重查與FinMind數值相同，亦缺決策時可取得的原始發布版本、完整TPEx、限額有效日證據；"
            "不得將Current-vintage歷史重查當作PIT首次發布。"
        )
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as exc:
        result["status"] = "SOURCE_OR_CONTRACT_BLOCKED"
        result["block_reason"] = f"{type(exc).__name__}: {exc}".replace(str(root) + "/", "")
    result["audit_fingerprint"] = canonical_json_sha256({
        "contract": AUDIT_ID, "snapshot": result.get("provider_snapshot_fingerprint"),
        "dates": result.get("sample_dates"),
        "capture": [(r.get("report_date"), r.get("raw_sha256"), r.get("capture_utc"),
                     r.get("status")) for r in result["samples"]],
    })
    output = root / OUTPUT_ROOT / str(result["audit_fingerprint"])
    for report_date, raw in raw_captures.items():
        _write_exact_capture(output / "captured_requeries" / (report_date.replace("-", "") + ".json"), raw)
    atomic_write_json(output / "publication_probe.json", result)
    atomic_write_text(output / "publication_probe.md", markdown_report(result))
    return result, output


def main() -> int:
    parser = argparse.ArgumentParser(description="TWSE歷史報表T-2確認餘額來源稽核（單次、唯讀、不授權模型）")
    parser.add_argument("--project-root", type=Path, default=_ROOT)
    parser.add_argument("--snapshot-fingerprint", default=None)
    parser.add_argument("--report-date", action="append", dest="dates", default=None,
                        help="官方交易日YYYY-MM-DD；可重複，預設每年二月第一個交易日")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fetch-public", action="store_true",
                       help="查詢TWSE公開歷史網頁（取得的是今日的歷史requery，非原始vintage）")
    group.add_argument("--reports-dir", type=Path,
                       help="匯入YYYYMMDD.json官方查詢檔；仍不能自稱當年vintage")
    args = parser.parse_args()
    result, folder = run_probe(args.project_root, snapshot_fingerprint=args.snapshot_fingerprint,
                               dates=args.dates, fetch_public=args.fetch_public,
                               reports_dir=args.reports_dir)
    try:
        path = folder.relative_to(args.project_root.resolve()).as_posix()
    except ValueError:
        path = OUTPUT_ROOT.as_posix()
    print(f"官方確認餘額比對：{result['status']} | 模型Input授權：NO")
    print(f"TWSE sample={result['sample_success']}/{len(result['sample_dates'])} | "
          f"TPEx=NOT_VERIFIED | historical_vintage=NOT_PROVEN")
    print(f"報表：{path}/publication_probe.md")
    return 0 if result["status"] == "HISTORICAL_VINTAGE_NOT_PROVEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())

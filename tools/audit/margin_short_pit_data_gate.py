"""One-off, read-only PIT data gate for FinMind individual margin/short-sale data.

This audit does NOT authorize a new model input.  It measures conservative
previous-Taiwan-trading-session coverage on the existing adjusted-price Research
control grid using an immutable, SHA-verified Market Data V2 Provider Snapshot.
The Trading refresh schedule is NOT proof of a historical publication vintage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.breakout_quality import BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE
from config.research import RESEARCH_MARKET_DATA_CUTOFF
from core.file_integrity import atomic_write_json, atomic_write_text, canonical_json_sha256, load_json_strict
from core.market_data_research_pit_contract import build_research_v2_pit_review_contracts
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.market_data.provider_snapshot_view import ProviderSnapshotView

DATASET = "TaiwanStockMarginPurchaseShortSale"
AUDIT_CONTRACT = "margin_short_pit_data_gate_v2"
AUDIT_RELATIVE_ROOT = Path("outputs/filters/breakout_quality/audits/margin_short_pit_data_gate")
NUMERIC_COLUMNS = (
    "MarginPurchaseTodayBalance", "MarginPurchaseLimit",
    "ShortSaleTodayBalance", "ShortSaleLimit",
)
REQUIRED_COLUMNS = ("date", "stock_id", *NUMERIC_COLUMNS)


def _date(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"無效日期: {value!r}")
    return stamp.normalize()


def _ratio_pct(count: int, total: int) -> float | None:
    return round(100.0 * count / total, 3) if total else None


def _new_counts() -> dict[str, int]:
    return dict.fromkeys((
        "control_stock_days", "previous_session_available", "source_row_present",
        "both_ratio_valid", "margin_ratio_valid", "short_ratio_valid",
        "margin_limit_zero", "short_limit_zero", "margin_balance_zero_valid",
        "short_balance_zero_valid", "margin_values_invalid", "short_values_invalid",
        "source_row_missing",
    ), 0)


def _add_counts(destination: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        destination[key] += int(value)


def _summarize_year(values: Mapping[str, int]) -> dict[str, int | float | None]:
    summary = dict(values)
    denominator = int(values["previous_session_available"])
    summary["both_ratio_coverage_pct"] = _ratio_pct(int(values["both_ratio_valid"]), denominator)
    summary["margin_ratio_coverage_pct"] = _ratio_pct(int(values["margin_ratio_valid"]), denominator)
    summary["short_ratio_coverage_pct"] = _ratio_pct(int(values["short_ratio_valid"]), denominator)
    summary["source_row_presence_pct"] = _ratio_pct(int(values["source_row_present"]), denominator)
    return summary


def audit_coverage(
    control_dates_by_ticker: Mapping[str, pd.DatetimeIndex],
    source_frames: Iterable[pd.DataFrame],
    *,
    end_date: str,
    oos_start_date: str,
    start_date: str | None = None,
    trading_dates: pd.DatetimeIndex | None = None,
) -> dict[str, object]:
    """Count exact previous-session observations, with no fill or future lookup.

    Input control dates must already be canonical adjusted-sanitizer retained dates;
    this deliberately measures a broad control universe, NOT an exact AO eligible
    stock-day population. If verified trading dates are supplied, pre-calendar
    control history is explicitly excluded from audited coverage (never filled),
    and any gap within the available calendar still fails. No model fitting or
    feature-value fabrication occurs.
    """
    if not control_dates_by_ticker:
        raise ValueError("Audit control calendar不可為空")
    end = _date(end_date)
    oos_start = _date(oos_start_date)
    begin = _date(start_date) if start_date else None
    if begin is not None and begin > end:
        raise ValueError("Audit start_date晚於end_date")
    ticker_set = {str(k).strip() for k in control_dates_by_ticker}
    if len(ticker_set) != len(control_dates_by_ticker) or "" in ticker_set:
        raise ValueError("Control ticker不可空白或重複")
    control: dict[str, pd.DatetimeIndex] = {}
    calendar = pd.DatetimeIndex([])
    for ticker, dates in control_dates_by_ticker.items():
        idx = pd.DatetimeIndex(dates).normalize()
        if idx.hasnans or idx.has_duplicates:
            raise ValueError(f"Control retained dates含缺值／重複: {ticker}")
        idx = idx.sort_values()
        control[str(ticker).strip()] = idx
        calendar = calendar.union(idx)
    calendar = calendar.sort_values()
    # The Full Research adjusted series may predate the provider trading-calendar
    # archive (e.g. 1992 stock-days).  Only report PIT coverage where a dated,
    # SHA-verified calendar actually exists; do not invent historical sessions.
    # Any unexplained gap *inside* that calendar's evidenced period still fails.
    calendar_scope: dict[str, object] = {
        "calendar_source": "control_date_union_unverified" if trading_dates is None
                           else "TaiwanStockTradingDate_provider_snapshot",
        "requested_start_date": begin.strftime("%Y-%m-%d") if begin is not None else None,
        "verified_calendar_first_date": None,
        "verified_calendar_last_date": None,
        "effective_start_date": begin.strftime("%Y-%m-%d") if begin is not None else None,
        "pre_calendar_control_date_count": 0,
        "pre_calendar_control_stock_days": 0,
        "pre_calendar_control_first_date": None,
        "pre_calendar_control_last_date": None,
        "pre_calendar_history_audited": trading_dates is None,
    }
    if trading_dates is not None:
        verified_calendar = pd.DatetimeIndex(trading_dates).normalize().sort_values()
        if verified_calendar.empty or verified_calendar.hasnans or verified_calendar.has_duplicates:
            raise ValueError("已驗證交易日曆含空白、重複或不合法日期")
        first_calendar_date = verified_calendar[0]
        last_calendar_date = verified_calendar[-1]
        # Require a predecessor to the first OOS prediction day, otherwise an
        # apparently good OOS percentage could hide a truncated recent calendar.
        if first_calendar_date >= oos_start:
            raise ValueError("台股交易日曆未涵蓋OOS開始前的交易日: "
                             f"calendar_first={first_calendar_date:%Y-%m-%d}, "
                             f"oos_start={oos_start:%Y-%m-%d}")
        if begin is not None and begin < first_calendar_date:
            raise ValueError("指定Audit起始日早於Provider台股交易日曆可驗證範圍: "
                             f"requested={begin:%Y-%m-%d}, "
                             f"calendar_first={first_calendar_date:%Y-%m-%d}")
        legacy_days = calendar[(calendar < first_calendar_date) & (calendar <= end)]
        effective_begin = first_calendar_date if begin is None else begin
        in_verified_scope = calendar[(calendar >= effective_begin) & (calendar <= end)]
        # This also rejects uncovered control dates after the last calendar day;
        # it never silently shrinks a requested OOS or verified-history window.
        uncovered_control_days = in_verified_scope.difference(verified_calendar)
        if len(uncovered_control_days):
            raise ValueError("Adjusted控制日期不在已驗證台股交易日曆涵蓋期間: "
                             + str([day.strftime("%Y-%m-%d") for day in uncovered_control_days[:10]]))
        calendar_scope.update({
            "verified_calendar_first_date": first_calendar_date.strftime("%Y-%m-%d"),
            "verified_calendar_last_date": last_calendar_date.strftime("%Y-%m-%d"),
            "effective_start_date": effective_begin.strftime("%Y-%m-%d"),
            "pre_calendar_control_date_count": len(legacy_days),
            "pre_calendar_control_stock_days": sum(
                int(((dates < first_calendar_date) & (dates <= end)).sum())
                for dates in control.values()
            ),
            "pre_calendar_control_first_date": legacy_days[0].strftime("%Y-%m-%d") if len(legacy_days) else None,
            "pre_calendar_control_last_date": legacy_days[-1].strftime("%Y-%m-%d") if len(legacy_days) else None,
            "pre_calendar_history_audited": len(legacy_days) == 0,
        })
        begin = effective_begin
        calendar = verified_calendar[verified_calendar <= end]

    groups: dict[str, list[pd.DataFrame]] = defaultdict(list)
    schema_errors: list[str] = []
    malformed_dates = 0
    raw_rows = 0
    selected_rows = 0
    source_first_date: str | None = None
    source_last_date: str | None = None
    malformed_ticker_rows = 0
    for part_no, frame in enumerate(source_frames, start=1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Provider Snapshot reader必須回傳DataFrame")
        if frame.empty and not len(frame.columns):
            continue  # Empty provider artifact without materialized schema.
        missing_cols = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing_cols:
            schema_errors.append(f"part={part_no} missing={','.join(missing_cols)}")
            continue
        raw_rows += len(frame)
        part = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
        # Preserve 0050 / ETF leading zeros from the declared provider string ID.
        if pd.api.types.is_numeric_dtype(part["stock_id"]):
            schema_errors.append(f"part={part_no} stock_id_numeric_type_loses_leading_zero")
            continue
        part["stock_id"] = part["stock_id"].astype("string").str.strip()
        malformed_ticker_rows += int(part["stock_id"].isna().sum() + part["stock_id"].eq("").fillna(False).sum())
        part["date"] = pd.to_datetime(part["date"], errors="coerce", format="%Y-%m-%d").dt.normalize()
        malformed_dates += int(part["date"].isna().sum())
        part = part.loc[part["date"].notna() & part["stock_id"].isin(ticker_set)
                        & (part["date"] <= end) & (part["date"] >= calendar[0])]
        selected_rows += len(part)
        if not part.empty:
            earliest = part["date"].min().strftime("%Y-%m-%d")
            latest = part["date"].max().strftime("%Y-%m-%d")
            source_first_date = min(source_first_date, earliest) if source_first_date else earliest
            source_last_date = max(source_last_date, latest) if source_last_date else latest
        for column in NUMERIC_COLUMNS:
            part[column] = pd.to_numeric(part[column], errors="coerce")
        for ticker, subset in part.groupby("stock_id", sort=False):
            groups[str(ticker)].append(subset.drop(columns=["stock_id"]))

    issues = {
        "source_raw_rows": raw_rows,
        "source_rows_for_control_tickers_through_cutoff": selected_rows,
        "source_invalid_date_rows": malformed_dates,
        "source_invalid_ticker_rows": malformed_ticker_rows,
        "source_earliest_control_ticker_date": source_first_date,
        "source_latest_control_ticker_date": source_last_date,
        "source_duplicate_ticker_date_rows": 0,
        "schema_errors": schema_errors[:20],
        "source_tickers_with_rows": len(groups),
    }
    per_year: dict[str, dict[str, int]] = defaultdict(_new_counts)
    all_counts = _new_counts()
    oos_counts = _new_counts()
    ratio_over_one = {"margin": 0, "short": 0}
    per_ticker: dict[str, dict[str, int | float | None]] = {}
    missing_examples: list[dict[str, str]] = []
    duplicate_examples: list[dict[str, str]] = []

    for ticker, days in control.items():
        in_range = (days <= end) & ((days >= begin) if begin is not None else True)
        prediction_days = days[in_range]
        if len(prediction_days) == 0:
            continue
        prior_positions = calendar.searchsorted(prediction_days, side="left") - 1
        has_prior_session = prior_positions >= 0
        prev_dates = pd.DatetimeIndex(
            [calendar[pos] if pos >= 0 else pd.NaT for pos in prior_positions]
        )
        fragments = groups.get(ticker, [])
        if fragments:
            observations = pd.concat(fragments, ignore_index=True)
            duplicates = observations["date"].duplicated(keep=False)
            duplicate_count = int(observations["date"].duplicated().sum())
            issues["source_duplicate_ticker_date_rows"] += duplicate_count
            if duplicates.any() and len(duplicate_examples) < 12:
                duplicate_examples.extend(
                    {"ticker": ticker, "date": stamp.strftime("%Y-%m-%d")}
                    for stamp in observations.loc[duplicates, "date"].iloc[:12 - len(duplicate_examples)]
                )
            # Ambiguous source keys have NO usable observation, even if their values match.
            observations = observations.loc[~duplicates].set_index("date").sort_index()
            observations["_observed"] = True
            joined = observations.reindex(prev_dates)
            present = joined["_observed"].eq(True).to_numpy(dtype=bool) & has_prior_session
            mp_bal = joined[NUMERIC_COLUMNS[0]].to_numpy(dtype=float)
            mp_limit = joined[NUMERIC_COLUMNS[1]].to_numpy(dtype=float)
            ss_bal = joined[NUMERIC_COLUMNS[2]].to_numpy(dtype=float)
            ss_limit = joined[NUMERIC_COLUMNS[3]].to_numpy(dtype=float)
        else:
            present = np.zeros(len(prediction_days), dtype=bool)
            mp_bal = mp_limit = ss_bal = ss_limit = np.full(len(prediction_days), np.nan)
        # Zero balance is valid if quota/limit is positive; an absent/quota-zero
        # value is not a zero-valued utilization feature.
        mp_valid = present & np.isfinite(mp_bal) & (mp_bal >= 0) & np.isfinite(mp_limit) & (mp_limit > 0)
        ss_valid = present & np.isfinite(ss_bal) & (ss_bal >= 0) & np.isfinite(ss_limit) & (ss_limit > 0)
        ratio_over_one["margin"] += int((mp_valid & (mp_bal > mp_limit)).sum())
        ratio_over_one["short"] += int((ss_valid & (ss_bal > ss_limit)).sum())
        vectors = {
            "control_stock_days": np.ones(len(prediction_days), dtype=bool),
            "previous_session_available": has_prior_session,
            "source_row_present": present,
            "both_ratio_valid": mp_valid & ss_valid,
            "margin_ratio_valid": mp_valid,
            "short_ratio_valid": ss_valid,
            "margin_limit_zero": present & np.isfinite(mp_limit) & (mp_limit == 0),
            "short_limit_zero": present & np.isfinite(ss_limit) & (ss_limit == 0),
            "margin_balance_zero_valid": mp_valid & (mp_bal == 0),
            "short_balance_zero_valid": ss_valid & (ss_bal == 0),
            "margin_values_invalid": present & ~mp_valid,
            "short_values_invalid": present & ~ss_valid,
            "source_row_missing": has_prior_session & ~present,
        }
        year_strs = prediction_days.strftime("%Y").tolist()
        ticker_totals = {key: int(value.sum()) for key, value in vectors.items()}
        per_ticker[ticker] = _summarize_year(ticker_totals)
        for year in sorted(set(year_strs)):
            mask = np.asarray([y == year for y in year_strs])
            sums = {key: int(value[mask].sum()) for key, value in vectors.items()}
            _add_counts(per_year[year], sums)
            _add_counts(all_counts, sums)
            if int(year) >= oos_start.year:
                oos_mask = mask & (prediction_days >= oos_start)
                _add_counts(oos_counts, {key: int(value[oos_mask].sum()) for key, value in vectors.items()})
        if len(missing_examples) < 12:
            for dt, prev_dt, observed in zip(prediction_days, prev_dates, present):
                if pd.notna(prev_dt) and not observed:
                    missing_examples.append({"ticker": ticker, "prediction_date": dt.strftime("%Y-%m-%d"),
                                             "required_source_date": prev_dt.strftime("%Y-%m-%d")})
                    if len(missing_examples) >= 12:
                        break

    return {
        "control_ticker_count": len(control),
        "calendar_scope": calendar_scope,
        "control_trading_session_count": len(calendar),
        "control_earliest_date": calendar.min().strftime("%Y-%m-%d"),
        "control_latest_date": calendar.max().strftime("%Y-%m-%d"),
        "coverage_scope": "canonical adjusted retained ticker-days (not AO exact eligible sample)",
        "availability_rule": (
            "source.date = strictly_previous_TaiwanStockTradingDate_provider_session(prediction_date); NO_FILL"
            if trading_dates is not None else
            "source.date = strictly_previous_union_control_trading_session(prediction_date); NO_FILL"
        ),
        "source": issues,
        "source_duplicate_examples": duplicate_examples,
        "source_missing_examples": missing_examples,
        "ratios_exceed_one_observations": ratio_over_one,
        "all_control_history": _summarize_year(all_counts),
        "oos_control_window": _summarize_year(oos_counts),
        "by_prediction_year": {y: _summarize_year(v) for y, v in sorted(per_year.items())},
        "per_control_ticker": dict(sorted(per_ticker.items())),
    }


def load_control(root: Path) -> tuple[dict[str, pd.DatetimeIndex], dict[str, object]]:
    summary_path = root / "outputs/filters/breakout_quality/breakout_quality_v1/dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Full Research Dataset摘要不存在: outputs/filters/breakout_quality/breakout_quality_v1/dataset_summary.json")
    summary = load_json_strict(summary_path)
    if summary.get("dataset") != "full":
        raise ValueError("Margin Data Gate只允許Full Research控制資料")
    current_inventory = build_source_data_inventory(root, "full")
    if summary.get("source_data_inventory") != current_inventory:
        raise ValueError("Dataset summary source_data_inventory不符合目前控制資料；須先修復來源READY")
    paths, duplicates = discover_dataset_csv_inputs(root, "full")
    if duplicates:
        raise ValueError("Full Research控制資料出現重複ticker CSV")
    control: dict[str, pd.DatetimeIndex] = {}
    for ticker, path in paths:
        frame = load_dataset_frame(path, ticker, min_rows=1)
        control[str(ticker)] = pd.DatetimeIndex(frame.index)
    return control, {"source_data_inventory": current_inventory, "dataset_summary_path":
                     "outputs/filters/breakout_quality/breakout_quality_v1/dataset_summary.json"}


def markdown_report(result: Mapping[str, object]) -> str:
    status = result["status"]
    lines = [
        "# 個股融資融券 PIT Data Gate（專題 Audit／唯讀）", "",
        f"- 資料檢查：**{status}**；模型 Input 授權：**NO**", 
        f"- Dataset：`{DATASET}`；研究截止：`{result['cutoff']}`", 
        "- 成交日 t 的 Input 僅允許 t 以前的上一個台股交易日資料；當日融資融券資料不可當日使用。", 
        "- FinMind 文件的平日 21:00 是一般更新時間，**不是**各歷史列的實際首發與修訂時間證據。", 
        "- 既有 Research V2 registry 是 `PIT_REVIEW_REQUIRED`；本報告不解除此限制。", "",
    ]
    if "block_reason" in result:
        lines.extend(["## 阻擋原因", "", str(result["block_reason"]), ""])
    if "coverage" in result:
        coverage = result["coverage"]
        overall = coverage["oos_control_window"]
        scope = coverage["calendar_scope"]
        if scope["verified_calendar_first_date"] is not None:
            lines.extend([
                "## 已驗證日曆與歷史涵蓋界線", "",
                f"- Provider日曆範圍：`{scope['verified_calendar_first_date']}`～`{scope['verified_calendar_last_date']}`。",
                f"- 本次有效稽核起點：`{scope['effective_start_date']}`。",
                f"- 早於Provider日曆的控制日期：{scope['pre_calendar_control_date_count']:,} 日、"
                f"{scope['pre_calendar_control_stock_days']:,} 股票日；**未稽核／不納入覆蓋率分母**。",
                "- 既有Adjusted歷史沒有被刪除；未涵蓋期間不能聲稱已完成PIT驗證。", 
            "- 此日期列表已驗證Provider Snapshot來源SHA；實際開市與歷史首發／修訂時點仍須獨立證明。", "",
            ])
        lines.extend([
            "## 覆蓋率（OOS 年度控制格；非精確 AO Eligible 樣本）", "",
            "| 指標 | 數值 |", "|---|---:|",
            f"| 控制股票日 | {overall['control_stock_days']:,} |",
            f"| 存在上一交易日 | {overall['previous_session_available']:,} |",
            f"| 前一交易日來源列存在率 | {overall['source_row_presence_pct']}% |",
            f"| 融資使用率可計算率 | {overall['margin_ratio_coverage_pct']}% |",
            f"| 融券使用率可計算率 | {overall['short_ratio_coverage_pct']}% |",
            f"| 兩者同時可計算率 | {overall['both_ratio_coverage_pct']}% |",
            f"| 融資限額為零 | {overall['margin_limit_zero']:,} |",
            f"| 融券限額為零 | {overall['short_limit_zero']:,} |", "",
            "## 歷史逐年覆蓋（按預測日、以前一交易日為來源）", "",
            "| 年度 | 股票日 | 來源存在% | 融資可用% | 融券可用% | 雙者可用% |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for year, row in coverage["by_prediction_year"].items():
            lines.append(
                f"| {year} | {row['control_stock_days']:,} | {row['source_row_presence_pct']} | "
                f"{row['margin_ratio_coverage_pct']} | {row['short_ratio_coverage_pct']} | "
                f"{row['both_ratio_coverage_pct']} |"
            )
        source = coverage["source"]
        lines.extend([
            "", "## 資料異常與缺漏", "",
            f"- 來源最早／最晚日期：{source['source_earliest_control_ticker_date']}～{source['source_latest_control_ticker_date']}",
            f"- 缺失／無效日期來源：{source['source_invalid_date_rows']:,} 筆",
            f"- 無效股票代碼：{source['source_invalid_ticker_rows']:,} 筆", 
            f"- 相同股票＋日期重複：{source['source_duplicate_ticker_date_rows']:,} 筆；不得擇一自行修復。", 
            f"- Schema 問題：{source['schema_errors']}", 
            f"- 比率大於 1（僅供稽核，不視為自動錯誤）：{coverage['ratios_exceed_one_observations']}",
            "", "## 下一步", "",
            "先確認歷史發布／修訂版本契約，再依控制格的逐年覆蓋情況設定受控模型 Input；"
            "不可把缺值當作零、用當日資料或拿 OOS 統計作 fitting／feature selection。", "",
        ])
    lines.extend([
        "## 研究邊界", "",
        "此為專題資料稽核；不會修改 Provider Snapshot、Research V1、AO checkpoint、標籤、"
        "模型 Comparison 或任何 Trading 參數。", "",
    ])
    return "\n".join(lines)


def run_data_gate(
    root: Path, *, snapshot_fingerprint: str | None = None,
    start_date: str | None = None,
) -> tuple[dict[str, object], Path]:
    root = Path(root).resolve()
    cutoff = str(RESEARCH_MARKET_DATA_CUTOFF)
    pit_contract = next(c for c in build_research_v2_pit_review_contracts() if c.dataset == DATASET)
    result: dict[str, object] = {
        "audit_contract": AUDIT_CONTRACT,
        "dataset": DATASET,
        "cutoff": cutoff,
        "oos_start": BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE,
        "start_date": start_date,
        "read_only": True,
        "model_input_authorized": False,
        "status": "NOT_RUN",
        "pit": {
            "canonical_review_status": pit_contract.review_status,
            "canonical_date_audit_mode": pit_contract.audit_mode,
            "availability_rule": "strict_previous_taiwan_trading_session",
            "same_day_use_allowed": False,
            "documented_update_time_taipei": "weekdays 21:00 (actual API update may differ)",
            "historical_publication_vintages_proven": False,
            "historical_revisions_proven": False,
            "model_fitting_authorized": False,
        },
    }
    try:
        control, control_identity = load_control(root)
        result["control"] = control_identity
        archive = load_ready_provider_snapshot_archive(root, snapshot_fingerprint=snapshot_fingerprint)
        view = ProviderSnapshotView(project_root=root, archive=archive)
        dataset_artifacts = view.dataset_artifacts(DATASET)
        result["provider_snapshot"] = {
            "fingerprint": archive.snapshot_fingerprint,
            "manifest_fingerprint": archive.manifest_fingerprint,
            "as_of_date": archive.as_of_date,
            "dataset_artifact_count": len(dataset_artifacts),
            "dataset_artifact_sha256": canonical_json_sha256([
                (artifact.request_id, artifact.content_sha256, artifact.row_count) for artifact in dataset_artifacts
            ]),
        }
        if archive.as_of_date < cutoff:
            raise ValueError(f"Provider Snapshot as_of不足: {archive.as_of_date} < {cutoff}")
        if not dataset_artifacts:
            raise ValueError("READY Provider Snapshot沒有個股融資融券Dataset工件")
        calendar_dataset = "TaiwanStockTradingDate"
        calendar_artifacts = view.dataset_artifacts(calendar_dataset)
        if not calendar_artifacts:
            raise ValueError("READY Provider Snapshot缺少台股交易日曆，不得以個股日期union代替")
        calendar_parts = list(view.iter_verified_frames(calendar_dataset, columns=("date",)))
        if not calendar_parts or any("date" not in part.columns for part in calendar_parts):
            raise ValueError("台股交易日曆Snapshot沒有有效date欄位")
        trading_days = pd.to_datetime(
            pd.concat([part["date"] for part in calendar_parts], ignore_index=True), errors="coerce", format="%Y-%m-%d"
        ).dt.normalize()
        if trading_days.isna().any():
            raise ValueError("台股交易日曆Snapshot含無效date")
        trading_days = pd.DatetimeIndex(trading_days.unique()).sort_values()
        result["provider_snapshot"]["trading_calendar_sha256"] = canonical_json_sha256([
            (artifact.request_id, artifact.content_sha256, artifact.row_count) for artifact in calendar_artifacts
        ])
        frames = view.iter_verified_frames_many_data_ids(
            DATASET, columns=REQUIRED_COLUMNS,
            data_ids=tuple(control),
        )
        coverage = audit_coverage(
            control, frames, end_date=cutoff,
            oos_start_date=BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE,
            start_date=start_date,
            trading_dates=trading_days,
        )
        result["coverage"] = coverage
        structural_issues = coverage["source"]
        if (structural_issues["schema_errors"] or structural_issues["source_duplicate_ticker_date_rows"]
                or structural_issues["source_invalid_date_rows"] or structural_issues["source_invalid_ticker_rows"]):
            result["status"] = "SCHEMA_OR_SOURCE_KEY_BLOCKED"
            result["block_reason"] = "Schema、來源日期或主鍵存在不合法證據；需由Provider來源修復。"
        elif not structural_issues["source_rows_for_control_tickers_through_cutoff"]:
            result["status"] = "NO_MATCHING_MARGIN_SOURCE_ROWS"
            result["block_reason"] = "Provider Snapshot沒有對應Full控制股票與研究截止日的來源列。"
        elif not coverage["oos_control_window"]["source_row_present"]:
            result["status"] = "NO_OOS_MARGIN_SOURCE_ROWS"
            result["block_reason"] = "OOS控制日期沒有可用的前一交易日融資融券資料；不能建立OOS Input。"
        elif not coverage["oos_control_window"]["margin_ratio_valid"] and not coverage["oos_control_window"]["short_ratio_valid"]:
            result["status"] = "NO_OOS_VALID_MARGIN_SHORT_RATIO"
            result["block_reason"] = "OOS沒有可計算的融資或融券使用率；限額為0或數值不合法。"
        else:
            result["status"] = "PIT_REVIEW_REQUIRED"
            result["block_reason"] = (
                "歷史資料覆蓋已完成唯讀統計，但 Provider Snapshot 缺少各歷史列首發與修訂vintage，"
                "且正式Research PIT contract未授權模型Input。需分開確認覆蓋與PIT來源證據。"
            )
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        result["status"] = "DATA_GATE_BLOCKED"
        detail = str(exc).replace(str(root) + "/", "").replace(str(root) + "\\", "")
        result["block_reason"] = f"{type(exc).__name__}: {detail}"
    result["audit_fingerprint"] = canonical_json_sha256({
        "contract": AUDIT_CONTRACT,
        "cutoff": cutoff,
        "oos_start": BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE,
        "start_date": start_date,
        "source_snapshot": result.get("provider_snapshot"),
        "control": result.get("control"),
    })
    target = root / AUDIT_RELATIVE_ROOT / str(result["audit_fingerprint"])
    atomic_write_json(target / "data_gate.json", result)
    atomic_write_text(target / "data_gate.md", markdown_report(result))
    return result, target


def main() -> int:
    parser = argparse.ArgumentParser(description="個股融資融券 PIT Data Gate（唯讀、一次性 Research Audit）")
    parser.add_argument("--project-root", type=Path, default=_ROOT)
    parser.add_argument("--snapshot-fingerprint", default=None,
                        help="指定已READY的immutable Provider Snapshot；預設最新READY")
    parser.add_argument("--start-date", default=None,
                        help="統計起始日期（預設以Provider日曆可驗證期間為界；更早的Adjusted歷史另行列報）")
    args = parser.parse_args()
    result, folder = run_data_gate(args.project_root, snapshot_fingerprint=args.snapshot_fingerprint,
                                   start_date=args.start_date)
    try:
        display_path = folder.resolve().relative_to(args.project_root.resolve()).as_posix()
    except ValueError:
        display_path = AUDIT_RELATIVE_ROOT.as_posix()
    print(f"資料檢查：{result['status']} | 模型Input授權：NO")
    if "coverage" in result:
        overall = result["coverage"]["oos_control_window"]
        print("OOS控制格 | "
              f"source_presence={overall['source_row_presence_pct']}% | "
              f"margin_valid={overall['margin_ratio_coverage_pct']}% | "
              f"short_valid={overall['short_ratio_coverage_pct']}% | "
              f"both={overall['both_ratio_coverage_pct']}%")
    if result.get("block_reason"):
        print(f"PIT/資料阻擋：{result['block_reason']}")
    print(f"報表：{display_path}/data_gate.md")
    return 0 if result["status"] == "PIT_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only MFE × Safety quadrant Audit over canonical truth and strategy artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from config.breakout_quality import get_breakout_quality_experiment_profile
from core.path_utils import project_relative_display_path
from filters.breakout_quality.continuous_target import (
    DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
)
from filters.breakout_quality.dataset_store import resolve_dataset_paths
from filters.breakout_quality.paths import resolve_filter_output_dir
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from services.audit.strategy_compare_source import (
    AuditSourceBlockedError,
    load_strategy_compare_source,
)
from core.report_metrics import (
    MFE_SAFETY_QUADRANT_DISTRIBUTION_METRICS,
    MFE_SAFETY_QUADRANT_ENRICHMENT_METRICS,
)

SUPPORTED_AUDIT_TYPE = "continuous_truth_strategy_quadrants"

_QUADRANT_KEYS = (
    "high_mfe_high_safety_pct",
    "high_mfe_low_safety_pct",
    "low_mfe_high_safety_pct",
    "low_mfe_low_safety_pct",
)
_QUADRANT_ENRICHMENT_KEYS = (
    "high_mfe_high_safety_enrichment",
    "high_mfe_low_safety_enrichment",
)

_TICKER_KEYS = ("ticker", "symbol", "stock_id", "stock", "code")
_DATE_KEYS = (
    "date",
    "candidate_date",
    "breakout_quality_score_date",
    "score_date",
    "signal_date",
    "trade_date",
    "entry_date",
    "buy_date",
)
_ROW_COLLECTION_KEYS = {
    "candidate_rows": "candidate",
    "orderable_rows": "orderable",
    "selected_rows": "selected",
    "selection_rows": "selected",
    "trade_rows": "trade",
    "closed_trade_rows": "closed_trade",
}
_SELECTION_PRIORITY = ("selected", "trade", "closed_trade")


AuditBlockedError = AuditSourceBlockedError


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping.get(key)
            if value not in (None, ""):
                return value
    return None


def _normalize_ticker(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _normalize_date(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ensure_daily_percentile(
    frame: pd.DataFrame,
    *,
    method: str,
) -> tuple[pd.DataFrame, str]:
    table = frame.copy()
    if "percentile" in table:
        percentile = pd.to_numeric(table["percentile"], errors="coerce")
        valid = percentile.notna() & percentile.between(0.0, 1.0, inclusive="both")
        if valid.all():
            table["percentile"] = percentile.astype(float)
            return table, "canonical percentile column"
    rank_method = str(method).strip().lower()
    if rank_method != "average_zero_based":
        raise ValueError(
            "same-day percentile method目前只接受與既有project rank contract一致的average_zero_based"
        )
    grouped = table.groupby("date", sort=False)["value"]
    average_rank = grouped.rank(method="average") - 1.0
    group_size = grouped.transform("size").astype(float)
    denominator = group_size - 1.0
    denominator_array = denominator.to_numpy(dtype=float)
    percentile = np.full(len(table), 0.5, dtype=float)
    np.divide(
        average_rank.to_numpy(dtype=float),
        denominator_array,
        out=percentile,
        where=denominator_array > 0.0,
    )
    table["percentile"] = percentile
    return table, "derived from canonical raw truth via same-day average zero-based rank / (N-1)"


def _extract_date_range(payload: Any) -> tuple[str | None, str | None]:
    starts: list[str] = []
    ends: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).strip().lower()
                if lowered in {"start", "start_date", "period_start", "evaluation_start"}:
                    date_text = _normalize_date(child)
                    if date_text:
                        starts.append(date_text)
                elif lowered in {"end", "end_date", "period_end", "evaluation_end"}:
                    date_text = _normalize_date(child)
                    if date_text:
                        ends.append(date_text)
                elif isinstance(child, (Mapping, list, tuple)):
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child)

    visit(payload)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditBlockedError(f"無法讀取正式JSON工件: {path.name} | {exc}") from exc


def _referenced_artifact_paths(project_root: Path, result_dir: Path, payloads: Sequence[Any]) -> set[Path]:
    paths: set[Path] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            text = value.strip()
            if not text or len(text) > 500:
                return
            if not any(text.lower().endswith(suffix) for suffix in (".json", ".csv", ".parquet", ".pq")):
                return
            raw_path = Path(text)
            candidates = (
                (raw_path,) if raw_path.is_absolute() else (project_root / raw_path, result_dir / raw_path)
            )
            for candidate in candidates:
                try:
                    if candidate.is_file():
                        paths.add(candidate.resolve())
                except OSError:
                    continue

    for payload in payloads:
        visit(payload)
    return paths


def _record_row_collection(
    target: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    arm_id: str,
    row_kind: str,
    rows: Any,
) -> None:
    if not isinstance(rows, list):
        return
    valid_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not valid_rows:
        return
    target.setdefault(arm_id, {}).setdefault(row_kind, []).extend(valid_rows)


def _collect_embedded_arm_rows(
    payload: Any,
    arm_ids: Sequence[str],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    collected: dict[str, dict[str, list[dict[str, Any]]]] = {}
    arm_set = set(arm_ids)

    def visit(value: Any, active_arm: str | None = None) -> None:
        if isinstance(value, Mapping):
            explicit_arm = str(
                value.get("arm_id")
                or value.get("comparison_id")
                or value.get("strategy_id")
                or ""
            ).strip()
            current_arm = explicit_arm if explicit_arm in arm_set else active_arm
            for key, child in value.items():
                key_text = str(key).strip()
                child_arm = key_text if key_text in arm_set else current_arm
                row_kind = _ROW_COLLECTION_KEYS.get(key_text.lower())
                if row_kind and child_arm:
                    _record_row_collection(
                        collected,
                        arm_id=child_arm,
                        row_kind=row_kind,
                        rows=child,
                    )
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child, child_arm)
        elif isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child, active_arm)

    visit(payload)
    return collected


def _merge_row_collections(
    target: dict[str, dict[str, list[dict[str, Any]]]],
    source: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> None:
    for arm_id, kinds in source.items():
        for row_kind, rows in kinds.items():
            _record_row_collection(target, arm_id=arm_id, row_kind=row_kind, rows=list(rows))


def _infer_arm_from_path(path: Path, arm_ids: Sequence[str]) -> str | None:
    for part in reversed(path.parts):
        tokens = str(part).replace("-", "_").replace(".", "_").upper()
        matches = [arm_id for arm_id in arm_ids if arm_id.upper() in tokens]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    return None


def _infer_row_kind_from_path(path: Path) -> str | None:
    lower = path.name.lower()
    for token, kind in (
        ("orderable", "orderable"),
        ("selected", "selected"),
        ("selection", "selected"),
        ("candidate", "candidate"),
        ("closed_trade", "closed_trade"),
        ("trades", "trade"),
        ("trade", "trade"),
    ):
        if token in lower:
            return kind
    return None


def _load_generic_rows_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False).to_dict(orient="records")
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path).to_dict(orient="records")
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [dict(row) for row in payload if isinstance(row, Mapping)]
            if isinstance(payload, Mapping):
                for key in ("rows", "records", "data"):
                    rows = payload.get(key)
                    if isinstance(rows, list):
                        return [dict(row) for row in rows if isinstance(row, Mapping)]
    except (OSError, ValueError, TypeError, json.JSONDecodeError, ImportError):
        return []
    return []


def load_strategy_rows(
    project_root: Path,
    *,
    profile_id: str,
    arm_ids: Sequence[str],
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    Path,
    tuple[str | None, str | None],
    str,
]:
    strategy_source = load_strategy_compare_source(
        project_root, profile_id=profile_id
    )
    settings = strategy_source.settings
    result_dir = strategy_source.result_dir
    payloads: list[Any] = [strategy_source.result]
    if strategy_source.manifest is not None:
        payloads.append(strategy_source.manifest)
    source_fingerprint = strategy_source.config_fingerprint
    collected: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for payload in payloads:
        _merge_row_collections(collected, _collect_embedded_arm_rows(payload, arm_ids))

    artifact_paths = _referenced_artifact_paths(project_root, result_dir, payloads)
    for path in result_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".parquet", ".pq"}:
            artifact_paths.add(path.resolve())
    for path in sorted(artifact_paths, key=lambda item: item.as_posix()):
        arm_id = _infer_arm_from_path(path, arm_ids)
        row_kind = _infer_row_kind_from_path(path)
        if arm_id is not None and row_kind is not None:
            rows = _load_generic_rows_file(path)
            _record_row_collection(collected, arm_id=arm_id, row_kind=row_kind, rows=rows)
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            _merge_row_collections(collected, _collect_embedded_arm_rows(payload, arm_ids))

    period_start, period_end = _extract_date_range(payloads)
    if settings.start_date:
        period_start = str(settings.start_date)
    if settings.end_date:
        period_end = str(settings.end_date)
    return collected, result_dir, (period_start, period_end), source_fingerprint


def _cohort_keys_from_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    records: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        ticker = _normalize_ticker(_first_present(row, _TICKER_KEYS))
        date_text = _normalize_date(_first_present(row, _DATE_KEYS))
        if ticker and date_text:
            records.append((ticker, date_text))
    if not records:
        return pd.DataFrame(columns=["ticker", "date"])
    frame = pd.DataFrame(records, columns=["ticker", "date"])
    return frame.drop_duplicates(["ticker", "date"]).sort_values(["date", "ticker"], kind="stable")


def _pick_arm_selection_rows(kinds: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], str]:
    for kind in _SELECTION_PRIORITY:
        rows = kinds.get(kind)
        if rows:
            return [dict(row) for row in rows], kind
    return [], "missing"


def _filter_period(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    result = frame
    if start_date:
        result = result.loc[result["date"] >= start_date]
    if end_date:
        result = result.loc[result["date"] <= end_date]
    return result.copy()


def _resolve_truth_provider_contract(
    definition: AuditDefinition,
) -> tuple[str, str, str, str, str]:
    source = definition.source
    filter_id = str(source.get("filter_id") or "").strip()
    model_architecture = str(source.get("model_architecture") or "").strip()
    provider_profile_id = str(source.get("truth_provider_profile_id") or "").strip()
    mfe_target_id = str(source.get("mfe_target_id") or "").strip()
    safety_target_id = str(source.get("safety_target_id") or "").strip()
    missing = [
        name
        for name, value in (
            ("filter_id", filter_id),
            ("model_architecture", model_architecture),
            ("truth_provider_profile_id", provider_profile_id),
            ("mfe_target_id", mfe_target_id),
            ("safety_target_id", safety_target_id),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"{definition.audit_id} truth provider設定不完整: {', '.join(missing)}"
        )
    profile = get_breakout_quality_experiment_profile(provider_profile_id)
    if str(profile.continuous_target_id or "") != mfe_target_id:
        raise ValueError(
            f"{definition.audit_id} truth provider profile target不一致: "
            f"profile={provider_profile_id}, target={profile.continuous_target_id!r}, "
            f"expected={mfe_target_id!r}"
        )
    if safety_target_id != DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID:
        raise ValueError(
            f"{definition.audit_id} safety target目前必須使用canonical "
            f"{DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID!r}"
        )
    return filter_id, model_architecture, provider_profile_id, mfe_target_id, safety_target_id


def build_truth_geometry(
    definition: AuditDefinition,
    project_root: Path,
    *,
    percentile_method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the canonical daily-universal truth geometry without a duplicate artifact.

    Daily-universal targets are generated by the profile-aware sample provider and do
    not own the older event-style ``continuous_targets/<target_id>`` physical bundle.
    Pure-MFE already carries the same selected-peak adverse component, so Low-Adverse
    Safety is the canonical ``-target_adverse_r`` component from that same provider.
    This is the same ownership contract used by Strategy Compare path diagnostics.
    """

    (
        filter_id,
        model_architecture,
        provider_profile_id,
        mfe_target_id,
        safety_target_id,
    ) = _resolve_truth_provider_contract(definition)
    bundle = load_profile_continuous_ranker_data(
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=provider_profile_id,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=project_root,
    )
    groups = bundle.group_table.copy()
    required = {"ticker", "date", "target_adverse_r"}
    missing = sorted(required - set(groups.columns))
    if missing:
        raise AuditBlockedError(
            f"canonical daily truth provider缺少欄位: {missing}"
        )
    if len(groups) != len(bundle.raw_target) or len(groups) != len(bundle.target_valid):
        raise AuditBlockedError("canonical daily truth provider group/target長度不一致")
    target_contract = dict(dict(bundle.target_manifest or {}).get("target_contract") or {})
    if str(target_contract.get("target_id") or "") != mfe_target_id:
        raise AuditBlockedError("canonical daily truth provider target identity不一致")

    valid = np.asarray(bundle.target_valid, dtype=bool)
    mfe_value = pd.to_numeric(pd.Series(bundle.raw_target, index=groups.index), errors="coerce")
    adverse_value = pd.to_numeric(groups["target_adverse_r"], errors="coerce")
    valid &= np.isfinite(mfe_value.to_numpy(dtype=float))
    valid &= np.isfinite(adverse_value.to_numpy(dtype=float))
    if not bool(valid.any()):
        raise AuditBlockedError("canonical daily truth provider沒有有效MFE/Safety rows")

    base = pd.DataFrame(
        {
            "ticker": groups.loc[valid, "ticker"].astype(str).str.strip().to_numpy(),
            "date": pd.to_datetime(groups.loc[valid, "date"], errors="raise")
            .dt.strftime("%Y-%m-%d")
            .to_numpy(),
            "mfe_value": mfe_value.loc[valid].to_numpy(dtype=float),
            # Low-Adverse Safety target is exactly negative adverse-to-peak R.
            "safety_value": (-adverse_value.loc[valid]).to_numpy(dtype=float),
        }
    )
    if bool(base.duplicated(["ticker", "date"]).any()):
        raise AuditBlockedError("canonical daily truth provider存在重複ticker/date")

    mfe = base[["ticker", "date", "mfe_value"]].rename(columns={"mfe_value": "value"})
    safety = base[["ticker", "date", "safety_value"]].rename(
        columns={"safety_value": "value"}
    )
    mfe, mfe_percentile_source = _ensure_daily_percentile(mfe, method=percentile_method)
    safety, safety_percentile_source = _ensure_daily_percentile(
        safety, method=percentile_method
    )
    joined = mfe[["ticker", "date", "percentile"]].rename(
        columns={"percentile": "mfe_percentile"}
    ).merge(
        safety[["ticker", "date", "percentile"]].rename(
            columns={"percentile": "safety_percentile"}
        ),
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise AuditBlockedError("Pure-MFE與Low-Adverse Safety canonical truth沒有共同ticker/date")

    dataset_root = resolve_filter_output_dir(project_root, filter_id=filter_id)
    source = {
        "provider": "canonical profile-aware daily sample provider",
        "filter_id": filter_id,
        "model_architecture": model_architecture,
        "provider_profile_id": provider_profile_id,
        "dataset_root": project_relative_display_path(dataset_root, project_root=project_root),
        "mfe_target_id": mfe_target_id,
        "mfe_truth_source": "bundle.raw_target",
        "mfe_percentile_source": mfe_percentile_source,
        "safety_target_id": safety_target_id,
        "safety_truth_source": "-bundle.group_table.target_adverse_r",
        "safety_percentile_source": safety_percentile_source,
        "joined_truth_rows": int(len(joined)),
    }
    return joined, source


def _quadrant_columns(frame: pd.DataFrame, *, cutoff: float) -> pd.DataFrame:
    table = frame.copy()
    mfe_high = table["mfe_percentile"] >= float(cutoff)
    safety_high = table["safety_percentile"] >= float(cutoff)
    conditions = [
        mfe_high & safety_high,
        mfe_high & ~safety_high,
        ~mfe_high & safety_high,
        ~mfe_high & ~safety_high,
    ]
    labels = list(_QUADRANT_KEYS)
    table["quadrant"] = np.select(conditions, labels, default="")
    return table


def _distribution_for_keys(
    truth: pd.DataFrame,
    keys: pd.DataFrame | None,
) -> dict[str, Any]:
    if keys is None:
        covered = truth
        raw_count = int(len(truth))
    else:
        raw_count = int(len(keys))
        covered = keys.merge(truth, on=["ticker", "date"], how="inner", validate="one_to_one")
    covered_count = int(len(covered))
    if covered_count <= 0:
        raise AuditBlockedError("cohort與MFE/Safety truth沒有任何共同ticker/date")
    counts = covered["quadrant"].value_counts().to_dict()
    result: dict[str, Any] = {
        "raw_rows": raw_count,
        "truth_covered_rows": covered_count,
        "truth_coverage_pct": (covered_count / raw_count * 100.0) if raw_count else 100.0,
    }
    for key in _QUADRANT_KEYS:
        result[key] = float(counts.get(key, 0)) / covered_count * 100.0
    return result


def _enrichment(distribution: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for distribution_key, enrichment_key in zip(_QUADRANT_KEYS[:2], _QUADRANT_ENRICHMENT_KEYS):
        base = _finite_float(population.get(distribution_key))
        value = _finite_float(distribution.get(distribution_key))
        result[enrichment_key] = None if base in (None, 0.0) or value is None else value / base
    return result


def _fingerprint_payload(definition: AuditDefinition, source_refs: Mapping[str, Any]) -> str:
    payload = {
        "definition": definition.as_dict(),
        "sources": source_refs,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _format_metric(value: Any, *, unit: str, digits: int) -> str:
    number = _finite_float(value)
    if number is None:
        return "-"
    text = f"{number:.{digits}f}"
    return f"{text}{unit}"


def _render_distribution_table(mode_result: Mapping[str, Any], cohort_order: Sequence[str]) -> str:
    from core.console_report import render_table

    metrics = MFE_SAFETY_QUADRANT_DISTRIBUTION_METRICS
    headers = ["來源", "N", "Truth coverage", *(metric.label for metric in metrics)]
    rows = []
    cohorts = mode_result["cohorts"]
    for cohort_id in cohort_order:
        item = cohorts[cohort_id]
        rows.append(
            [
                item["display_name"],
                str(item["truth_covered_rows"]),
                f"{item['truth_coverage_pct']:.2f}%",
                *[
                    _format_metric(item[metric.key], unit=metric.unit, digits=metric.digits)
                    for metric in metrics
                ],
            ]
        )
    return render_table(headers, rows)


def _render_enrichment_table(mode_result: Mapping[str, Any], cohort_order: Sequence[str]) -> str:
    from core.console_report import render_table

    metrics = MFE_SAFETY_QUADRANT_ENRICHMENT_METRICS
    headers = ["來源", *(metric.label for metric in metrics)]
    rows = []
    cohorts = mode_result["cohorts"]
    for cohort_id in cohort_order:
        item = cohorts[cohort_id]
        enrichment = item["enrichment_vs_population"]
        rows.append(
            [
                item["display_name"],
                *[
                    _format_metric(
                        enrichment.get(metric.key),
                        unit=metric.unit,
                        digits=metric.digits,
                    )
                    for metric in metrics
                ],
            ]
        )
    return render_table(headers, rows)


def render_result(result: Mapping[str, Any]) -> str:
    from core.console_report import render_key_values, render_section, render_title

    lines = [render_title("MFE × Safety 四象限 Audit")]
    lines.append(
        render_key_values(
            [
                ("Audit", result["audit_id"]),
                ("Threshold", f"same-day percentile >= {result['percentile_cutoff']:.2f} = High"),
                ("Percentile", result["percentile_semantics"]),
                ("決策問題", result["decision_question"]),
            ]
        )
    )
    for index, mode_id in enumerate(result["evaluation_order"], start=1):
        mode_result = result["evaluations"][mode_id]
        lines.append(render_section(f"{mode_result['display_name']}｜表 1：MFE × Safety 四象限分布", number=index * 2 - 1))
        lines.append(_render_distribution_table(mode_result, result["cohort_order"]))
        lines.append(render_section(f"{mode_result['display_name']}｜表 2：相對母體 enrichment", number=index * 2))
        lines.append(_render_enrichment_table(mode_result, result["cohort_order"]))
        focus_arm_id = str(result.get("focus_arm_id") or "").strip()
        focus = mode_result["cohorts"].get(focus_arm_id) if focus_arm_id else None
        if focus:
            hm_ls = focus["enrichment_vs_population"].get("high_mfe_low_safety_enrichment")
            hm_hs = focus["enrichment_vs_population"].get("high_mfe_high_safety_enrichment")
            hm_ls_text = _format_metric(hm_ls, unit="×", digits=2)
            hm_hs_text = _format_metric(hm_hs, unit="×", digits=2)
            lines.append(
                f"{focus_arm_id} 觀察：High-MFE / Low-Safety = {focus['high_mfe_low_safety_pct']:.2f}% "
                f"({hm_ls_text}母體)；High-MFE / High-Safety = {focus['high_mfe_high_safety_pct']:.2f}% "
                f"({hm_hs_text}母體)。"
            )
    lines.append(render_section("判讀邊界"))
    lines.append(
        "本 Audit 不以OOS結果反向調 threshold，也不建立新模型／target。"
        f"若 {result.get('focus_arm_id', 'focus arm')} 在 OOS 與 Rolling 都仍把 "
        "High-MFE / Low-Safety 的占比推高至母體以上，"
        "即可直接支持『conditional J 可學，但未保證 absolute Safety 高』的 geometry 疑慮；"
        "後續是否改 target 由研究決策另行確定。"
    )
    lines.append(render_section("工件輸出"))
    lines.append(render_key_values([(label, path) for label, path in result["artifacts"].items()]))
    return "\n".join(lines)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = ["# MFE × Safety 四象限 Audit", ""]
    lines += [
        f"- Audit: `{result['audit_id']}`",
        f"- Threshold: same-day percentile >= {result['percentile_cutoff']:.2f} = High",
        f"- Percentile: {result['percentile_semantics']}",
        f"- 決策問題: {result['decision_question']}",
        "",
    ]
    for mode_id in result["evaluation_order"]:
        mode_result = result["evaluations"][mode_id]
        lines += [f"## {mode_result['display_name']}", "", "### 四象限分布", ""]
        metrics = MFE_SAFETY_QUADRANT_DISTRIBUTION_METRICS
        headers = ["來源", "N", "Truth coverage", *(metric.label for metric in metrics)]
        rows: list[list[str]] = []
        for cohort_id in result["cohort_order"]:
            item = mode_result["cohorts"][cohort_id]
            rows.append(
                [
                    item["display_name"],
                    str(item["truth_covered_rows"]),
                    f"{item['truth_coverage_pct']:.2f}%",
                    *[f"{item[metric.key]:.{metric.digits}f}{metric.unit}" for metric in metrics],
                ]
            )
        lines += [_markdown_table(headers, rows), "", "### 相對母體 enrichment", ""]
        e_metrics = MFE_SAFETY_QUADRANT_ENRICHMENT_METRICS
        e_headers = ["來源", *(metric.label for metric in e_metrics)]
        e_rows: list[list[str]] = []
        for cohort_id in result["cohort_order"]:
            item = mode_result["cohorts"][cohort_id]
            enrichment = item["enrichment_vs_population"]
            e_rows.append(
                [
                    item["display_name"],
                    *[
                        _format_metric(enrichment.get(metric.key), unit=metric.unit, digits=metric.digits)
                        for metric in e_metrics
                    ],
                ]
            )
        lines += [_markdown_table(e_headers, e_rows), ""]
    return "\n".join(lines)


def _write_csv(path: Path, result: Mapping[str, Any]) -> None:
    fieldnames = [
        "evaluation",
        "cohort",
        "display_name",
        "row_source",
        "raw_rows",
        "truth_covered_rows",
        "truth_coverage_pct",
        *_QUADRANT_KEYS,
        *_QUADRANT_ENRICHMENT_KEYS,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode_id in result["evaluation_order"]:
            mode_result = result["evaluations"][mode_id]
            for cohort_id in result["cohort_order"]:
                item = mode_result["cohorts"][cohort_id]
                row = {
                    "evaluation": mode_id,
                    "cohort": cohort_id,
                    "display_name": item["display_name"],
                    "row_source": item.get("row_source", ""),
                    "raw_rows": item["raw_rows"],
                    "truth_covered_rows": item["truth_covered_rows"],
                    "truth_coverage_pct": item["truth_coverage_pct"],
                    **{key: item[key] for key in _QUADRANT_KEYS},
                    **item["enrichment_vs_population"],
                }
                writer.writerow(row)


def _relative_artifact_path(path: Path, project_root: Path) -> str:
    return project_relative_display_path(path, project_root=project_root)


def run_audit(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    if definition.audit_type != SUPPORTED_AUDIT_TYPE:
        raise ValueError(f"unsupported audit_type: {definition.audit_type}")
    source = definition.source
    dimensions = definition.dimensions
    evaluation_order = tuple(str(value) for value in source.get("evaluation_profile_ids", ()))
    arm_ids = tuple(str(value) for value in source.get("strategy_arm_ids", ()))
    candidate_pool_arm_id = str(source.get("candidate_pool_arm_id") or "").strip()
    focus_arm_id = str(source.get("focus_arm_id") or "").strip()
    mfe_target_id = str(source.get("mfe_target_id") or "").strip()
    safety_target_id = str(source.get("safety_target_id") or "").strip()
    cohort_order = tuple(str(value) for value in dimensions.get("cohort_order", ()))
    cutoff = float(dimensions.get("same_day_percentile_cutoff"))
    percentile_method = str(dimensions.get("percentile_method") or "average")
    if (
        not evaluation_order
        or not arm_ids
        or not candidate_pool_arm_id
        or not focus_arm_id
        or not mfe_target_id
        or not safety_target_id
    ):
        raise ValueError(f"{definition.audit_id} source設定不完整")
    if candidate_pool_arm_id not in arm_ids:
        raise ValueError(f"{definition.audit_id} candidate_pool_arm_id必須存在於strategy_arm_ids")
    if focus_arm_id not in arm_ids:
        raise ValueError(f"{definition.audit_id} focus_arm_id必須存在於strategy_arm_ids")
    if not 0.0 < cutoff < 1.0:
        raise ValueError(f"{definition.audit_id} same_day_percentile_cutoff必須介於0與1")
    expected_cohorts = {"population", "candidate_pool", *arm_ids}
    if set(cohort_order) != expected_cohorts:
        raise ValueError(
            f"{definition.audit_id} cohort_order必須且只能包含 {sorted(expected_cohorts)}"
        )

    truth, truth_source = build_truth_geometry(
        definition,
        project_root,
        percentile_method=percentile_method,
    )
    truth = _quadrant_columns(truth, cutoff=cutoff)
    strategy_payloads: dict[str, Any] = {}
    source_refs: dict[str, Any] = {"truth": truth_source, "strategy": {}}

    for profile_id in evaluation_order:
        rows_by_arm, result_dir, period, source_fingerprint = load_strategy_rows(
            project_root,
            profile_id=profile_id,
            arm_ids=arm_ids,
        )
        period_truth = _filter_period(truth, period[0], period[1])
        if period_truth.empty:
            raise AuditBlockedError(f"{profile_id} evaluation period內沒有MFE/Safety truth")
        missing_arms = [arm_id for arm_id in arm_ids if arm_id not in rows_by_arm]
        if missing_arms:
            raise AuditBlockedError(
                f"{profile_id} Strategy Compare工件缺少row-level arm evidence: {missing_arms}; "
                "Audit不得重跑策略補資料"
            )
        pool_rows = rows_by_arm.get(candidate_pool_arm_id, {}).get("orderable", [])
        if not pool_rows:
            raise AuditBlockedError(
                f"{profile_id}/{candidate_pool_arm_id} 缺少orderable_rows；"
                "無法建立Breakout/orderable candidate pool，Audit不得以trade rows替代"
            )
        pool_keys = _filter_period(_cohort_keys_from_rows(pool_rows), period[0], period[1])
        population = _distribution_for_keys(period_truth, None)
        cohort_results: dict[str, Any] = {
            "population": {
                "display_name": "實際母體（All eligible stock-days）",
                "row_source": "canonical daily eligible stock-day truth",
                **population,
                "enrichment_vs_population": {
                    "high_mfe_high_safety_enrichment": 1.0,
                    "high_mfe_low_safety_enrichment": 1.0,
                },
            }
        }
        pool_distribution = _distribution_for_keys(period_truth, pool_keys)
        cohort_results["candidate_pool"] = {
            "display_name": "Breakout / orderable candidate pool",
            "row_source": f"{candidate_pool_arm_id}.orderable_rows",
            **pool_distribution,
            "enrichment_vs_population": _enrichment(pool_distribution, population),
        }
        arm_row_sources: dict[str, str] = {}
        for arm_id in arm_ids:
            selected_rows, row_source = _pick_arm_selection_rows(rows_by_arm[arm_id])
            if not selected_rows:
                raise AuditBlockedError(
                    f"{profile_id}/{arm_id} 缺少selected_rows/trade_rows/closed_trade_rows；"
                    "無法建立實際策略選股cohort"
                )
            arm_keys = _filter_period(_cohort_keys_from_rows(selected_rows), period[0], period[1])
            distribution = _distribution_for_keys(period_truth, arm_keys)
            cohort_results[arm_id] = {
                "display_name": arm_id,
                "row_source": f"{arm_id}.{row_source}",
                **distribution,
                "enrichment_vs_population": _enrichment(distribution, population),
            }
            arm_row_sources[arm_id] = row_source
        strategy_settings = load_strategy_compare_source(
            project_root, profile_id=profile_id
        ).settings
        strategy_payloads[profile_id] = {
            "display_name": strategy_settings.profile_label,
            "period": {"start": period[0], "end": period[1]},
            "strategy_result_dir": _relative_artifact_path(result_dir, project_root),
            "candidate_pool_arm_id": candidate_pool_arm_id,
            "focus_arm_id": focus_arm_id,
            "strategy_config_fingerprint": source_fingerprint,
            "arm_row_sources": arm_row_sources,
            "cohorts": cohort_results,
        }
        source_refs["strategy"][profile_id] = {
            "result_dir": _relative_artifact_path(result_dir, project_root),
            "period": period,
            "strategy_config_fingerprint": source_fingerprint,
            "candidate_pool_row_source": f"{candidate_pool_arm_id}.orderable_rows",
            "arm_row_sources": arm_row_sources,
        }

    fingerprint = _fingerprint_payload(definition, source_refs)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = project_root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    run_dir = output_root / "runs" / f"{timestamp}_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "mfe_safety_quadrants.json"
    csv_path = run_dir / "mfe_safety_quadrants.csv"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"
    latest_path = output_root / "latest.json"

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "module_id": definition.module_id,
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "decision_question": str(definition.outcomes.get("decision_question") or ""),
        "focus_arm_id": focus_arm_id,
        "stopping_condition": str(definition.outcomes.get("stopping_condition") or ""),
        "percentile_cutoff": cutoff,
        "percentile_method": percentile_method,
        "percentile_semantics": (
            "優先重用canonical percentile column；缺少時只從canonical raw truth做同日cross-sectional percentile"
        ),
        "truth_sources": truth_source,
        "evaluation_order": list(evaluation_order),
        "cohort_order": list(cohort_order),
        "evaluations": strategy_payloads,
        "artifacts": {
            "完整JSON": _relative_artifact_path(json_path, project_root),
            "彙總CSV": _relative_artifact_path(csv_path, project_root),
            "詳細Markdown": _relative_artifact_path(report_path, project_root),
            "Manifest": _relative_artifact_path(manifest_path, project_root),
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, result)
    report_path.write_text(_render_markdown(result), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "audit_definition": definition.as_dict(),
        "config_fingerprint": fingerprint,
        "source_refs": source_refs,
        "artifacts": result["artifacts"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    output_root.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(
            {
                "run_dir": _relative_artifact_path(run_dir, project_root),
                "report": _relative_artifact_path(report_path, project_root),
                "result": _relative_artifact_path(json_path, project_root),
                "config_fingerprint": fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    latest_dir = output_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "audit.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / "audit.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / "audit.csv").write_bytes(csv_path.read_bytes())
    (latest_dir / "manifest.json").write_text(
        manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return result


def preflight(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    blockers: list[str] = []
    source = definition.source
    truth_sources: list[str] = []
    try:
        (
            filter_id,
            _model_architecture,
            provider_profile_id,
            mfe_target_id,
            safety_target_id,
        ) = _resolve_truth_provider_contract(definition)
        dataset_dir = resolve_filter_output_dir(project_root, filter_id=filter_id)
        dataset = resolve_dataset_paths(dataset_dir)
        truth_sources.append(
            project_relative_display_path(dataset.summary, project_root=project_root)
        )
        if not dataset.summary.is_file():
            blockers.append(
                "缺少canonical Dataset summary: "
                + project_relative_display_path(dataset.summary, project_root=project_root)
            )
        truth_sources.append(
            f"profile:{provider_profile_id} / target:{mfe_target_id} / safety:{safety_target_id}"
        )
    except (ValueError, OSError) as exc:
        blockers.append(str(exc))

    strategy_paths: list[str] = []
    for profile_id in tuple(source.get("evaluation_profile_ids") or ()):
        try:
            strategy_source = load_strategy_compare_source(
                project_root, profile_id=str(profile_id)
            )
            strategy_paths.append(
                project_relative_display_path(
                    strategy_source.result_dir, project_root=project_root
                )
            )
        except (ValueError, AuditBlockedError, OSError, json.JSONDecodeError) as exc:
            blockers.append(str(exc))
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": blockers,
        "target_paths": truth_sources,
        "strategy_paths": strategy_paths,
    }


def load_latest_result(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any] | None:
    output_root = project_root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    canonical_latest_result = output_root / "latest" / "audit.json"
    if canonical_latest_result.exists():
        payload = _load_json(canonical_latest_result)
        return dict(payload) if isinstance(payload, Mapping) else None

    latest_path = output_root / "latest.json"
    if not latest_path.exists():
        return None
    pointer = _load_json(latest_path)
    if not isinstance(pointer, Mapping):
        return None
    result_path_text = str(pointer.get("result") or "").strip()
    if not result_path_text:
        return None
    result_path = Path(result_path_text)
    if not result_path.is_absolute():
        result_path = project_root / result_path
    if not result_path.exists():
        return None
    payload = _load_json(result_path)
    return dict(payload) if isinstance(payload, Mapping) else None


def collect_status(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    state = preflight(definition, project_root=Path(project_root))
    blockers = tuple(str(value) for value in state.get("blockers", ()))
    source = {
        "display": "Daily-universal MFE × Safety truth provider + OOS/Rolling Strategy Compare",
        "target_paths": list(state.get("target_paths", ())),
        "strategy_paths": list(state.get("strategy_paths", ())),
    }
    return {
        "status": str(state.get("status") or "BLOCKED"),
        "reason": "；".join(blockers),
        "source": source,
    }


def run_formal_audit(
    definition: AuditDefinition,
    *,
    project_root: Path,
    quiet: bool = False,
) -> dict[str, Any]:
    result = run_audit(definition, project_root=Path(project_root))
    if not quiet:
        print(render_result(result))
    return result


__all__ = [
    "AuditBlockedError",
    "SUPPORTED_AUDIT_TYPE",
    "build_truth_geometry",
    "collect_status",
    "load_latest_result",
    "load_strategy_rows",
    "preflight",
    "render_result",
    "run_audit",
    "run_formal_audit",
]

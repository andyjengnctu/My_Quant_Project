"""Read-only all-seed portfolio-translation attribution for multi-seed robustness.

This Audit never trains a model or replays a strategy.  It consumes only the
compact attribution source persisted by the canonical multi-seed robustness
orchestrator and reuses the existing strategy-attribution primitives for each
matched seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from config.strategy_compare import get_strategy_multi_seed_robustness_settings
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from core.runtime_utils import get_taipei_now
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.trade_attribution import assign_canonical_trade_match_key
from filters.breakout_quality.strategy_multi_seed_robustness import (
    ATTRIBUTION_SOURCE_DIRNAME,
    ATTRIBUTION_SOURCE_SCHEMA_VERSION,
    LATEST_FILENAME,
    MANIFEST_FILENAME,
    SEED_RESULTS_FILENAME,
)
from tools.audit.breakout_quality.c15_strategy_attribution import (
    build_strategy_attribution_pair_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class CompactArmArtifacts:
    arm_id: str
    summary: dict[str, Any]
    trades_path: Path
    equity_path: Path
    capacity_path: Path
    selected_path: Path
    execution_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "缺少JSON工件: "
            + project_relative_display_path(path, project_root=PROJECT_ROOT)
        )
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root必須是object: {path.name}")
    return payload


def _validate_definition(definition: AuditDefinition) -> tuple[str, str, str, int, int, int]:
    source = dict(definition.source)
    if str(source.get("kind") or "") != "multi_seed_robustness":
        raise ValueError(f"{definition.audit_id}.source.kind必須是multi_seed_robustness")
    robustness_id = str(source.get("robustness_id") or "").strip()
    run_selector = str(source.get("run") or "latest").strip().lower()
    if run_selector != "latest":
        raise ValueError(f"{definition.audit_id}.source.run目前只允許latest")
    candidate = str(source.get("candidate_arm_id") or "").strip()
    comparator = str(source.get("comparator_arm_id") or "").strip()
    if not robustness_id or not candidate or not comparator or candidate == comparator:
        raise ValueError(f"{definition.audit_id}.source robustness/candidate/comparator設定不合法")
    dimensions = dict(definition.dimensions)
    try:
        focus_year = int(dimensions.get("focus_year"))
        top_month_count = int(dimensions.get("top_month_count"))
        top_trade_count = int(dimensions.get("top_trade_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{definition.audit_id}.dimensions必須是整數") from exc
    if focus_year < 1900 or top_month_count < 1 or top_trade_count < 1:
        raise ValueError(f"{definition.audit_id}.dimensions超出合法範圍")
    return robustness_id, candidate, comparator, focus_year, top_month_count, top_trade_count


def _resolve_run_root(root: Path, robustness_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    cfg = get_strategy_multi_seed_robustness_settings(robustness_id)
    latest_path = (root / cfg.output_root / LATEST_FILENAME).resolve()
    latest = _read_json(latest_path)
    summary_rel = str(latest.get("summary_path") or "").strip()
    if not summary_rel:
        raise ValueError("robustness latest缺少summary_path")
    summary_path = (root / summary_rel).resolve()
    try:
        summary_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("robustness summary必須位於專案root內") from exc
    summary = _read_json(summary_path)
    run_root = summary_path.parent
    manifest = _read_json(run_root / MANIFEST_FILENAME)
    if str(manifest.get("status") or "") != "COMPLETED":
        raise ValueError("robustness run尚未完成")
    contract = dict(manifest.get("contract") or summary.get("contract") or {})
    if str(contract.get("robustness_id") or "") != robustness_id:
        raise ValueError("robustness latest identity與Audit config不一致")
    return run_root, summary, contract


def _load_seed_results(run_root: Path) -> pd.DataFrame:
    path = run_root / SEED_RESULTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError("robustness缺少seed_results.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "arm_id", "name", "seed", "seed_order", "arm_order",
        "total_return_pct", "max_drawdown_pct", "return_over_max_drawdown",
        "expected_value_r", "direct_selection_r",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("robustness seed_results缺少欄位: " + ", ".join(missing))
    frame["arm_id"] = frame["arm_id"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["seed_order"] = pd.to_numeric(frame["seed_order"], errors="raise").astype(int)
    if frame.duplicated(["arm_id", "seed"], keep=False).any():
        raise ValueError("robustness seed_results存在重複arm/seed observation")
    if frame.duplicated(["arm_id", "seed_order"], keep=False).any():
        raise ValueError("robustness seed_results存在重複arm/seed_order observation")
    return frame


def _unit_manifest(
    root: Path,
    run_root: Path,
    *,
    arm_id: str,
    seed: int,
    fingerprint: str,
) -> tuple[Path, dict[str, Any]]:
    unit_dir = run_root / ATTRIBUTION_SOURCE_DIRNAME / f"{arm_id}__seed_{int(seed)}"
    payload = _read_json(unit_dir / MANIFEST_FILENAME)
    if int(payload.get("schema_version") or 0) != ATTRIBUTION_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"compact attribution schema不一致: {arm_id}/seed={seed}")
    if str(payload.get("scientific_fingerprint") or "") != str(fingerprint):
        raise ValueError(f"compact attribution fingerprint不一致: {arm_id}/seed={seed}")
    if str(payload.get("arm_id") or "") != arm_id or int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"compact attribution unit identity不一致: {arm_id}/seed={seed}")
    validation = dict(payload.get("scientific_observation_validation") or {})
    if str(validation.get("status") or "") != "VERIFIED":
        raise ValueError(f"compact attribution尚未通過scientific observation驗證: {arm_id}/seed={seed}")
    files = dict(payload.get("files") or {})
    for role in ("trades", "equity", "daily_capacity", "selected_buys", "execution"):
        item = dict(files.get(role) or {})
        rel = str(item.get("path") or "")
        if not rel:
            raise ValueError(f"compact attribution缺少{role}: {arm_id}/seed={seed}")
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("compact attribution path必須位於專案root內") from exc
        if not path.is_file():
            raise FileNotFoundError(
                "compact attribution缺少檔案: "
                + project_relative_display_path(path, project_root=root)
            )
        expected_sha = str(item.get("sha256") or "").lower()
        if expected_sha and compute_file_sha256(path).lower() != expected_sha:
            raise ValueError(f"compact attribution SHA256不一致: {arm_id}/seed={seed}/{role}")
    return unit_dir, payload


def _artifacts_from_unit(
    root: Path,
    unit_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> CompactArmArtifacts:
    files = dict(manifest["files"])
    def path_for(role: str) -> Path:
        return (root / str(dict(files[role])["path"])).resolve()
    return CompactArmArtifacts(
        arm_id=str(manifest["arm_id"]),
        summary=dict(summary),
        trades_path=path_for("trades"),
        equity_path=path_for("equity"),
        capacity_path=path_for("daily_capacity"),
        selected_path=path_for("selected_buys"),
        execution_path=path_for("execution"),
    )


def _resolve_source(
    root: Path,
    definition: AuditDefinition,
) -> dict[str, Any]:
    robustness_id, candidate, comparator, focus_year, top_month_count, top_trade_count = _validate_definition(definition)
    run_root, robustness_summary, contract = _resolve_run_root(root, robustness_id)
    fingerprint = str(contract.get("fingerprint") or robustness_summary.get("fingerprint") or "")
    if not fingerprint:
        raise ValueError("robustness contract缺少scientific fingerprint")
    seed_frame = _load_seed_results(run_root)
    candidate_rows = seed_frame[seed_frame["arm_id"] == candidate].sort_values("seed_order")
    comparator_rows = seed_frame[seed_frame["arm_id"] == comparator].sort_values("seed_order")
    if candidate_rows.empty or comparator_rows.empty:
        raise ValueError("robustness結果缺少Audit candidate/comparator arm")
    candidate_seeds = list(zip(candidate_rows["seed_order"].astype(int), candidate_rows["seed"].astype(int)))
    comparator_seeds = list(zip(comparator_rows["seed_order"].astype(int), comparator_rows["seed"].astype(int)))
    if candidate_seeds != comparator_seeds:
        raise ValueError("robustness candidate/comparator resolved seeds不一致")
    expected_count = int(contract.get("seed_count") or 0)
    if expected_count < 2 or len(candidate_seeds) != expected_count:
        raise ValueError(
            f"Audit要求完整all-seed source: expected={expected_count}, actual={len(candidate_seeds)}"
        )
    resolved_seeds = [int(value) for value in list(contract.get("resolved_seeds") or [])]
    expected_seed_pairs = list(enumerate(resolved_seeds, start=1))
    if len(resolved_seeds) != expected_count or candidate_seeds != expected_seed_pairs:
        raise ValueError("robustness seed_results與scientific contract resolved_seeds不一致")
    units: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for _seed_order, seed in candidate_seeds:
        for arm_id in (candidate, comparator):
            units[(arm_id, seed)] = _unit_manifest(
                root, run_root, arm_id=arm_id, seed=seed, fingerprint=fingerprint
            )
    return {
        "robustness_id": robustness_id,
        "candidate_arm_id": candidate,
        "comparator_arm_id": comparator,
        "focus_year": focus_year,
        "top_month_count": top_month_count,
        "top_trade_count": top_trade_count,
        "run_root": run_root,
        "summary": robustness_summary,
        "contract": contract,
        "fingerprint": fingerprint,
        "seed_frame": seed_frame,
        "seed_pairs": candidate_seeds,
        "units": units,
    }


def collect_forward_robustness_portfolio_translation_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        source = _resolve_source(root, definition)
        return {
            "audit_id": definition.audit_id,
            "status": "READY",
            "reason": "",
            "source": {
                "candidate_arm_id": source["candidate_arm_id"],
                "comparator_arm_id": source["comparator_arm_id"],
                "display": f"{source['candidate_arm_id']} vs {source['comparator_arm_id']} / {len(source['seed_pairs'])} seeds",
                "robustness_id": source["robustness_id"],
                "scientific_fingerprint": source["fingerprint"],
                "run": project_relative_display_path(source["run_root"], project_root=root),
            },
        }
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        return {
            "audit_id": definition.audit_id,
            "status": "BLOCKED",
            "reason": str(exc),
            "source": {
                "candidate_arm_id": str(definition.source.get("candidate_arm_id") or ""),
                "comparator_arm_id": str(definition.source.get("comparator_arm_id") or ""),
                "display": f"{definition.source.get('candidate_arm_id', '-')} vs {definition.source.get('comparator_arm_id', '-')}",
            },
        }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(v) for v in value]
    return value


def _boolean_series(values: pd.Series, *, field: str, arm_label: str) -> pd.Series:
    series = pd.Series(values).copy()
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    allowed = {"", "true", "false", "1", "0"}
    invalid = sorted(set(normalized.unique()) - allowed)
    if invalid:
        raise ValueError(f"{arm_label} execution attribution欄位{field}含無效布林值: {invalid}")
    return normalized.isin({"true", "1"})


def _execution_with_match_key(frame: pd.DataFrame, *, arm_label: str) -> pd.DataFrame:
    rows = pd.DataFrame(frame).copy()
    required = {
        "execution_order", "ticker", "trade_date", "entry_type", "entry_filled",
        "actual_risk_utilization", "chosen_risk_utilization", "candidate_risk_utilization",
        "risk_cap_binding", "position_cap_binding", "risk_position_tie",
        "capital_binding", "max_qty_binding", "lot_rounding_binding",
        "entry_budget_cash_binding", "candidate_qty", "chosen_qty", "filled_qty",
        "binding_signature",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{arm_label} execution attribution缺少欄位: {missing}")
    if rows.empty:
        rows["entry_date"] = pd.Series(dtype=str)
        return assign_canonical_trade_match_key(rows)
    filled_mask = _boolean_series(rows["entry_filled"], field="entry_filled", arm_label=arm_label)
    filled = rows[filled_mask].copy()
    filled["ticker"] = filled["ticker"].fillna("").astype(str).str.strip()
    filled["entry_date"] = pd.to_datetime(
        filled["trade_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d").fillna("")
    filled["entry_type"] = filled["entry_type"].fillna("normal").astype(str)
    filled["execution_order"] = pd.to_numeric(
        filled["execution_order"], errors="coerce"
    )
    if filled["execution_order"].isna().any():
        raise ValueError(f"{arm_label} execution attribution含無效execution_order")
    filled = filled.sort_values(["execution_order"], kind="mergesort").reset_index(drop=True)
    return assign_canonical_trade_match_key(filled)


def _exclusive_execution_side_summary(
    trade_rows: pd.DataFrame,
    execution_rows: pd.DataFrame,
    *,
    category: str,
    r_col: str,
    side_label: str,
) -> dict[str, Any]:
    trades = pd.DataFrame(trade_rows).copy()
    if trades.empty:
        exclusive = trades.copy()
    else:
        missing_trade_columns = sorted({"category", "match_key", r_col} - set(trades.columns))
        if missing_trade_columns:
            raise ValueError(
                f"{side_label} trade attribution缺少欄位: {missing_trade_columns}"
            )
        exclusive = trades[trades["category"] == category].copy()
    execution = _execution_with_match_key(execution_rows, arm_label=side_label)
    if exclusive.empty:
        return {
            "trade_count": 0,
            "actual_risk_utilization_mean": None,
            "candidate_risk_utilization_mean": None,
            "chosen_risk_utilization_mean": None,
            "winner_actual_risk_utilization_mean": None,
            "loser_actual_risk_utilization_mean": None,
            "winner_minus_loser_actual_risk_utilization": None,
            "cash_binding_count": 0,
            "risk_cap_binding_count": 0,
            "position_cap_binding_count": 0,
            "risk_position_tie_count": 0,
            "capital_binding_count": 0,
            "max_qty_binding_count": 0,
            "lot_rounding_binding_count": 0,
            "candidate_to_chosen_qty_reduction_mean": 0.0,
            "chosen_to_filled_qty_reduction_mean": 0.0,
            "binding_signature_counts": {},
        }
    if execution.empty:
        raise RuntimeError(f"{side_label} exclusive trades存在但execution attribution為空")
    if execution["match_key"].duplicated().any():
        raise RuntimeError(f"{side_label} execution attribution canonical match key重複")
    missing_keys = sorted(
        set(exclusive["match_key"].astype(str)) - set(execution["match_key"].astype(str))
    )
    if missing_keys:
        raise RuntimeError(
            f"{side_label} exclusive trades缺少execution attribution: count={len(missing_keys)}"
        )
    joined = exclusive[["match_key", r_col]].merge(
        execution, on="match_key", how="left", validate="one_to_one"
    )
    realized_r = pd.to_numeric(joined[r_col], errors="coerce")
    actual_util = pd.to_numeric(joined["actual_risk_utilization"], errors="coerce")
    chosen_util = pd.to_numeric(joined["chosen_risk_utilization"], errors="coerce")
    candidate_util = pd.to_numeric(joined["candidate_risk_utilization"], errors="coerce")

    def mean_for(mask: pd.Series, values: pd.Series) -> float | None:
        subset = pd.to_numeric(values[mask], errors="coerce")
        subset = subset[np.isfinite(subset)]
        return float(subset.mean()) if len(subset) else None

    winner_mean = mean_for(realized_r > 0.0, actual_util)
    loser_mean = mean_for(realized_r < 0.0, actual_util)
    finite_actual = actual_util[np.isfinite(actual_util)]
    finite_chosen = chosen_util[np.isfinite(chosen_util)]
    finite_candidate = candidate_util[np.isfinite(candidate_util)]
    bool_fields = (
        "entry_budget_cash_binding", "risk_cap_binding", "position_cap_binding",
        "risk_position_tie", "capital_binding", "max_qty_binding",
        "lot_rounding_binding",
    )
    counts = {
        field: int(_boolean_series(joined[field], field=field, arm_label=side_label).sum())
        for field in bool_fields
    }
    candidate_qty = pd.to_numeric(joined["candidate_qty"], errors="coerce").fillna(0.0)
    chosen_qty = pd.to_numeric(joined["chosen_qty"], errors="coerce").fillna(0.0)
    filled_qty = pd.to_numeric(joined["filled_qty"], errors="coerce").fillna(0.0)
    signature_counts = {
        str(key): int(value)
        for key, value in joined["binding_signature"].fillna("NONE_DETECTED").astype(str).value_counts().to_dict().items()
    }
    return {
        "trade_count": int(len(joined)),
        "actual_risk_utilization_mean": float(finite_actual.mean()) if len(finite_actual) else None,
        "chosen_risk_utilization_mean": float(finite_chosen.mean()) if len(finite_chosen) else None,
        "candidate_risk_utilization_mean": float(finite_candidate.mean()) if len(finite_candidate) else None,
        "winner_actual_risk_utilization_mean": winner_mean,
        "loser_actual_risk_utilization_mean": loser_mean,
        "winner_minus_loser_actual_risk_utilization": (
            None if winner_mean is None or loser_mean is None else float(winner_mean - loser_mean)
        ),
        "cash_binding_count": counts["entry_budget_cash_binding"],
        "risk_cap_binding_count": counts["risk_cap_binding"],
        "position_cap_binding_count": counts["position_cap_binding"],
        "risk_position_tie_count": counts["risk_position_tie"],
        "capital_binding_count": counts["capital_binding"],
        "max_qty_binding_count": counts["max_qty_binding"],
        "lot_rounding_binding_count": counts["lot_rounding_binding"],
        "candidate_to_chosen_qty_reduction_mean": float((candidate_qty - chosen_qty).mean()),
        "chosen_to_filled_qty_reduction_mean": float((chosen_qty - filled_qty).mean()),
        "binding_signature_counts": signature_counts,
    }


def _exclusive_execution_binding_summary(
    trade_rows: pd.DataFrame,
    candidate_execution: pd.DataFrame,
    comparator_execution: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "candidate_only": _exclusive_execution_side_summary(
            trade_rows, candidate_execution,
            category="candidate_only", r_col="candidate_r", side_label="candidate",
        ),
        "comparator_only": _exclusive_execution_side_summary(
            trade_rows, comparator_execution,
            category="comparator_only", r_col="comparator_r", side_label="comparator",
        ),
    }


def _driver_label(row: dict[str, Any]) -> str:
    selection = float(row["delta_direct_selection_r"])
    ret = float(row["delta_return_pct"])
    mdd = float(row["delta_mdd_pct"])
    romd = float(row["delta_romd"])
    exclusive_pnl = float(row["direct_pair_exclusive_delta_pnl"])
    common_pnl = float(row["common_trade_pnl_delta"])
    if selection > 0 and romd > 0:
        return "TRANSLATION_SUCCESS"
    if selection > 0 and romd <= 0:
        if ret <= 0:
            return "TRADE_SET_DOLLAR_TRANSLATION" if abs(exclusive_pnl) >= abs(common_pnl) else "PORTFOLIO_SIZING_PATH"
        if mdd > 0:
            return "MDD_WEALTH_PATH"
        return "MIXED_TRANSLATION"
    if selection <= 0 and romd > 0:
        return "PORTFOLIO_OVERRIDE"
    return "ALIGNED_DECLINE"


def _row_summary(
    *,
    seed_order: int,
    seed: int,
    candidate_row: dict[str, Any],
    comparator_row: dict[str, Any],
    pair: dict[str, Any],
    execution_binding: dict[str, Any],
) -> dict[str, Any]:
    trade = dict(pair.get("trade_contribution") or {})
    risk = dict(pair.get("risk_dollar_translation") or {})
    common = dict(risk.get("common") or {})
    candidate_only = dict(risk.get("candidate_only") or {})
    comparator_only = dict(risk.get("comparator_only") or {})
    exclusive_bridge = dict(risk.get("exclusive_bridge") or {})
    slots = dict(pair.get("slot_occupancy") or {})
    candidate_binding = dict(execution_binding.get("candidate_only") or {})
    comparator_binding = dict(execution_binding.get("comparator_only") or {})
    row = {
        "seed_order": int(seed_order),
        "seed": int(seed),
        "delta_direct_selection_r": float(candidate_row["direct_selection_r"]) - float(comparator_row["direct_selection_r"]),
        "delta_return_pct": float(candidate_row["total_return_pct"]) - float(comparator_row["total_return_pct"]),
        "delta_mdd_pct": float(candidate_row["max_drawdown_pct"]) - float(comparator_row["max_drawdown_pct"]),
        "delta_romd": float(candidate_row["return_over_max_drawdown"]) - float(comparator_row["return_over_max_drawdown"]),
        "delta_ev_r": float(candidate_row["expected_value_r"]) - float(comparator_row["expected_value_r"]),
        # delta_direct_selection_r is a baseline-relative edge: each arm's canonical
        # same-param direct-selection R is measured against the shared DL-off baseline,
        # then candidate - comparator is taken.  The direct pair attribution below is
        # a different basis and must remain separate rather than be forced equal.
        "direct_pair_exclusive_delta_r": float(trade.get("exclusive_selection_delta_r") or 0.0),
        "direct_pair_exclusive_delta_pnl": float(trade.get("exclusive_selection_delta_pnl") or 0.0),
        "direct_pair_common_delta_r": float(trade.get("common_trade_delta_r") or 0.0),
        "common_trade_pnl_delta": float(trade.get("common_trade_pnl_delta") or 0.0),
        "direct_pair_all_trade_delta_r": float(trade.get("all_trade_delta_r") or 0.0),
        "all_trade_pnl_delta": float(trade.get("all_trade_pnl_delta") or 0.0),
        "candidate_only_trade_count": int(trade.get("candidate_only_trade_count") or 0),
        "comparator_only_trade_count": int(trade.get("comparator_only_trade_count") or 0),
        "common_trade_count": int(trade.get("common_trade_count") or 0),
        "candidate_only_total_r": float(candidate_only.get("total_r") or 0.0),
        "comparator_only_total_r": float(comparator_only.get("total_r") or 0.0),
        "candidate_only_risk_weighted_r": _finite(candidate_only.get("risk_weighted_r")),
        "comparator_only_risk_weighted_r": _finite(comparator_only.get("risk_weighted_r")),
        "candidate_only_total_implied_initial_risk": float(candidate_only.get("total_implied_initial_risk") or 0.0),
        "comparator_only_total_implied_initial_risk": float(comparator_only.get("total_implied_initial_risk") or 0.0),
        "candidate_only_avg_implied_initial_risk": _finite(candidate_only.get("avg_implied_initial_risk")),
        "comparator_only_avg_implied_initial_risk": _finite(comparator_only.get("avg_implied_initial_risk")),
        "candidate_only_winner_avg_implied_initial_risk": _finite(candidate_only.get("winner_avg_implied_initial_risk")),
        "comparator_only_winner_avg_implied_initial_risk": _finite(comparator_only.get("winner_avg_implied_initial_risk")),
        "candidate_only_loser_avg_implied_initial_risk": _finite(candidate_only.get("loser_avg_implied_initial_risk")),
        "comparator_only_loser_avg_implied_initial_risk": _finite(comparator_only.get("loser_avg_implied_initial_risk")),
        "candidate_only_total_pnl": float(candidate_only.get("total_pnl") or 0.0),
        "comparator_only_total_pnl": float(comparator_only.get("total_pnl") or 0.0),
        "exclusive_reference_risk": _finite(exclusive_bridge.get("reference_risk")),
        "exclusive_equal_risk_selection_effect_pnl": float(exclusive_bridge.get("equal_risk_selection_effect_pnl") or 0.0),
        "exclusive_average_risk_scale_effect_pnl": float(exclusive_bridge.get("average_risk_scale_effect_pnl") or 0.0),
        "exclusive_within_set_weighting_effect_pnl": float(exclusive_bridge.get("within_set_weighting_effect_pnl") or 0.0),
        "exclusive_uncovered_pnl_delta": float(exclusive_bridge.get("uncovered_pnl_delta") or 0.0),
        "exclusive_bridge_total_pnl_delta": float(exclusive_bridge.get("total_pnl_delta") or 0.0),
        "exclusive_bridge_residual_pnl": float(exclusive_bridge.get("total_decomposition_residual_pnl") or 0.0),
        "common_risk_size_effect_pnl": float(common.get("risk_size_effect_pnl") or 0.0),
        "common_r_difference_effect_pnl": float(common.get("r_difference_effect_pnl") or 0.0),
        "common_decomposition_residual_pnl": float(common.get("decomposition_residual_pnl") or 0.0),
        "candidate_end_position_gap_slot_days": int(slots.get("candidate_end_position_gap_slot_days") or 0),
        "comparator_end_position_gap_slot_days": int(slots.get("comparator_end_position_gap_slot_days") or 0),
        "delta_end_position_gap_slot_days": int(slots.get("candidate_end_position_gap_slot_days") or 0) - int(slots.get("comparator_end_position_gap_slot_days") or 0),
        "wealth_advantage_pct": float(dict(pair.get("wealth_path") or {}).get("final_relative_wealth_advantage_pct") or 0.0),
        "candidate_only_actual_risk_utilization_mean": _finite(candidate_binding.get("actual_risk_utilization_mean")),
        "comparator_only_actual_risk_utilization_mean": _finite(comparator_binding.get("actual_risk_utilization_mean")),
        "candidate_only_candidate_risk_utilization_mean": _finite(candidate_binding.get("candidate_risk_utilization_mean")),
        "comparator_only_candidate_risk_utilization_mean": _finite(comparator_binding.get("candidate_risk_utilization_mean")),
        "candidate_only_chosen_risk_utilization_mean": _finite(candidate_binding.get("chosen_risk_utilization_mean")),
        "comparator_only_chosen_risk_utilization_mean": _finite(comparator_binding.get("chosen_risk_utilization_mean")),
        "candidate_only_winner_minus_loser_risk_utilization": _finite(candidate_binding.get("winner_minus_loser_actual_risk_utilization")),
        "comparator_only_winner_minus_loser_risk_utilization": _finite(comparator_binding.get("winner_minus_loser_actual_risk_utilization")),
        "candidate_only_cash_binding_count": int(candidate_binding.get("cash_binding_count") or 0),
        "comparator_only_cash_binding_count": int(comparator_binding.get("cash_binding_count") or 0),
        "candidate_only_risk_cap_binding_count": int(candidate_binding.get("risk_cap_binding_count") or 0),
        "comparator_only_risk_cap_binding_count": int(comparator_binding.get("risk_cap_binding_count") or 0),
        "candidate_only_position_cap_binding_count": int(candidate_binding.get("position_cap_binding_count") or 0),
        "comparator_only_position_cap_binding_count": int(comparator_binding.get("position_cap_binding_count") or 0),
        "candidate_only_risk_position_tie_count": int(candidate_binding.get("risk_position_tie_count") or 0),
        "comparator_only_risk_position_tie_count": int(comparator_binding.get("risk_position_tie_count") or 0),
        "candidate_only_capital_binding_count": int(candidate_binding.get("capital_binding_count") or 0),
        "comparator_only_capital_binding_count": int(comparator_binding.get("capital_binding_count") or 0),
        "candidate_only_max_qty_binding_count": int(candidate_binding.get("max_qty_binding_count") or 0),
        "comparator_only_max_qty_binding_count": int(comparator_binding.get("max_qty_binding_count") or 0),
        "candidate_only_lot_rounding_binding_count": int(candidate_binding.get("lot_rounding_binding_count") or 0),
        "comparator_only_lot_rounding_binding_count": int(comparator_binding.get("lot_rounding_binding_count") or 0),
        "candidate_only_qty_cash_reduction_mean": float(candidate_binding.get("candidate_to_chosen_qty_reduction_mean") or 0.0),
        "comparator_only_qty_cash_reduction_mean": float(comparator_binding.get("candidate_to_chosen_qty_reduction_mean") or 0.0),
        "candidate_only_binding_signature_counts": dict(candidate_binding.get("binding_signature_counts") or {}),
        "comparator_only_binding_signature_counts": dict(comparator_binding.get("binding_signature_counts") or {}),
    }
    row["selection_basis_gap_r"] = (
        row["direct_pair_exclusive_delta_r"] - row["delta_direct_selection_r"]
    )
    c_rw = row["candidate_only_risk_weighted_r"]
    b_rw = row["comparator_only_risk_weighted_r"]
    row["exclusive_risk_weighted_r_gap"] = (
        float(c_rw) - float(b_rw) if c_rw is not None and b_rw is not None else None
    )
    row["exclusive_total_implied_risk_delta"] = (
        row["candidate_only_total_implied_initial_risk"]
        - row["comparator_only_total_implied_initial_risk"]
    )
    c_util = row["candidate_only_actual_risk_utilization_mean"]
    b_util = row["comparator_only_actual_risk_utilization_mean"]
    row["exclusive_actual_risk_utilization_gap"] = (
        None if c_util is None or b_util is None else float(c_util - b_util)
    )
    row["driver"] = _driver_label(row)
    return row


def _distribution(values: pd.Series) -> dict[str, Any]:
    numbers = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(numbers):
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": int(len(numbers)),
        "mean": float(np.mean(numbers)),
        "median": float(np.median(numbers)),
        "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None,
        "min": float(np.min(numbers)),
        "max": float(np.max(numbers)),
    }


def _render_report(payload: dict[str, Any]) -> str:
    seed_rows = []
    for row in payload["seed_summary"]:
        seed_rows.append((
            f"S{row['seed_order']}",
            f"{row['delta_direct_selection_r']:+.2f} R",
            f"{row['delta_return_pct']:+.2f}%",
            f"{row['delta_mdd_pct']:+.2f}%",
            f"{row['delta_romd']:+.2f}",
            f"{row['direct_pair_exclusive_delta_r']:+.2f} R",
            f"{row['direct_pair_exclusive_delta_pnl']:+,.0f}",
            f"{row['common_risk_size_effect_pnl']:+,.0f}",
            f"{row['delta_end_position_gap_slot_days']:+d}",
            row["driver"],
        ))
    bridge_rows = []
    for row in payload["seed_summary"]:
        def _rw(value: Any) -> str:
            return "-" if value is None else f"{float(value):+.3f} R"

        def _money(value: Any) -> str:
            return "-" if value is None else f"{float(value):,.0f}"

        bridge_rows.append((
            f"S{row['seed_order']}",
            f"{row['candidate_only_total_r']:+.2f} R / {_rw(row['candidate_only_risk_weighted_r'])}",
            f"{row['comparator_only_total_r']:+.2f} R / {_rw(row['comparator_only_risk_weighted_r'])}",
            _money(row['candidate_only_avg_implied_initial_risk']),
            _money(row['comparator_only_avg_implied_initial_risk']),
            f"{_money(row['candidate_only_winner_avg_implied_initial_risk'])} / {_money(row['candidate_only_loser_avg_implied_initial_risk'])}",
            f"{_money(row['comparator_only_winner_avg_implied_initial_risk'])} / {_money(row['comparator_only_loser_avg_implied_initial_risk'])}",
            f"{row['direct_pair_exclusive_delta_pnl']:+,.0f}",
        ))
    exact_bridge_rows = []
    for row in payload["seed_summary"]:
        exact_bridge_rows.append((
            f"S{row['seed_order']}",
            "-" if row["exclusive_reference_risk"] is None else f"{float(row['exclusive_reference_risk']):,.0f}",
            f"{row['exclusive_equal_risk_selection_effect_pnl']:+,.0f}",
            f"{row['exclusive_average_risk_scale_effect_pnl']:+,.0f}",
            f"{row['exclusive_within_set_weighting_effect_pnl']:+,.0f}",
            f"{row['exclusive_uncovered_pnl_delta']:+,.0f}",
            f"{row['direct_pair_exclusive_delta_pnl']:+,.0f}",
            f"{row['exclusive_bridge_residual_pnl']:+,.2f}",
        ))
    binding_rows = []
    for row in payload["seed_summary"]:
        def _util(value: Any) -> str:
            return "-" if value is None else f"{float(value) * 100.0:.1f}%"

        def _pair_count(left_key: str, right_key: str) -> str:
            return f"{int(row[left_key])}/{int(row[right_key])}"

        binding_rows.append((
            f"S{row['seed_order']}",
            f"{_util(row['candidate_only_candidate_risk_utilization_mean'])} → {_util(row['candidate_only_chosen_risk_utilization_mean'])} → {_util(row['candidate_only_actual_risk_utilization_mean'])}",
            f"{_util(row['comparator_only_candidate_risk_utilization_mean'])} → {_util(row['comparator_only_chosen_risk_utilization_mean'])} → {_util(row['comparator_only_actual_risk_utilization_mean'])}",
            _util(row['candidate_only_winner_minus_loser_risk_utilization']),
            _util(row['comparator_only_winner_minus_loser_risk_utilization']),
            _pair_count('candidate_only_cash_binding_count', 'comparator_only_cash_binding_count'),
            _pair_count('candidate_only_position_cap_binding_count', 'comparator_only_position_cap_binding_count'),
            _pair_count('candidate_only_risk_cap_binding_count', 'comparator_only_risk_cap_binding_count'),
            _pair_count('candidate_only_risk_position_tie_count', 'comparator_only_risk_position_tie_count'),
            _pair_count('candidate_only_capital_binding_count', 'comparator_only_capital_binding_count'),
            _pair_count('candidate_only_max_qty_binding_count', 'comparator_only_max_qty_binding_count'),
            _pair_count('candidate_only_lot_rounding_binding_count', 'comparator_only_lot_rounding_binding_count'),
        ))
    drivers = payload["aggregate"]["driver_counts"]
    driver_rows = [(key, value) for key, value in sorted(drivers.items())]
    aggregate = payload["aggregate"]
    return "\n\n".join((
        render_title("Forward-OOS Multi-seed Portfolio Translation Audit"),
        render_key_values((
            ("Audit", payload["audit_id"]),
            ("Scientific fingerprint", payload["metadata"]["scientific_fingerprint"]),
            ("Seeds", payload["metadata"]["seed_count"]),
            ("比較", f"{payload['metadata']['candidate_arm_id']} − {payload['metadata']['comparator_arm_id']}"),
            ("契約", "read-only compact attribution source；不train、不replay"),
        )),
        render_section("同seed translation attribution", number=1),
        render_table(
            ("Seed", "ΔDL選擇R", "ΔReturn", "ΔMDD", "ΔRoMD", "Direct-pair Exclusive ΔR", "Exclusive ΔPnL", "Common sizing ΔPnL", "ΔGap slot-days", "主要機制"),
            seed_rows,
        ),
        render_section("Exclusive trade R→Dollar bridge", number=2),
        render_table(
            ("Seed", "MR-13A-only ΣR / RW-R", "MR-12B-only ΣR / RW-R", "13A-only Avg risk", "12B-only Avg risk", "13A Winner/Loser risk", "12B Winner/Loser risk", "Exclusive ΔPnL"),
            bridge_rows,
        ),
        render_section("Exclusive ΔPnL exact decomposition", number=3),
        render_table(
            ("Seed", "Reference risk", "Equal-risk selection", "Avg-risk scale", "Within-set weighting", "Uncovered", "Actual Exclusive ΔPnL", "Residual"),
            exact_bridge_rows,
        ),
        render_section("Exclusive risk utilization / binding attribution", number=4),
        render_table(
            ("Seed", "13A-only Pre→Chosen→Actual util", "12B-only Pre→Chosen→Actual util", "13A Winner−Loser util", "12B Winner−Loser util", "Cash bind 13A/12B", "Position cap", "Risk cap", "Risk/Pos tie", "Capital", "MaxQty", "Lot"),
            binding_rows,
        ),
        "Pre→Chosen→Actual util依序表示candidate sizing、cash-capped entry plan、實際成交後initial risk占預定risk budget比例。Winner−Loser util > 0 表示winner平均取得較高actual utilization；< 0 表示risk dollars較偏向loser。各binding欄為13A-only/12B-only exclusive trade count；constraint stages可重疊，例如upstream risk cap後仍可能再被MaxQty或cash cap縮小。",
        render_section("8-seed aggregate", number=5),
        render_key_values((
            ("Ranking↑ seeds", f"{aggregate['ranking_positive_count']}/{aggregate['seed_count']}"),
            ("Ranking↑ 且 RoMD↑", f"{aggregate['ranking_positive_romd_positive_count']}/{aggregate['ranking_positive_count']}"),
            ("Ranking/RoMD同方向", f"{aggregate['direction_aligned_count']}/{aggregate['seed_count']}"),
            ("Direct-pair Exclusive ΔR Mean", f"{aggregate['direct_pair_exclusive_delta_r']['mean']:+.2f} R"),
            ("Selection basis gap Mean", f"{aggregate['selection_basis_gap_r']['mean']:+.2f} R"),
            ("Exclusive ΔPnL Mean", f"{aggregate['direct_pair_exclusive_delta_pnl']['mean']:+,.2f}"),
            ("Exclusive RW-R gap Mean", "-" if aggregate['exclusive_risk_weighted_r_gap']['mean'] is None else f"{aggregate['exclusive_risk_weighted_r_gap']['mean']:+.3f} R"),
            ("Exclusive implied-risk Δ Mean", f"{aggregate['exclusive_total_implied_risk_delta']['mean']:+,.2f}"),
            ("Equal-risk selection effect Mean", f"{aggregate['exclusive_equal_risk_selection_effect_pnl']['mean']:+,.2f}"),
            ("Average-risk scale effect Mean", f"{aggregate['exclusive_average_risk_scale_effect_pnl']['mean']:+,.2f}"),
            ("Within-set weighting effect Mean", f"{aggregate['exclusive_within_set_weighting_effect_pnl']['mean']:+,.2f}"),
            ("Uncovered exclusive ΔPnL Mean", f"{aggregate['exclusive_uncovered_pnl_delta']['mean']:+,.2f}"),
            ("Exclusive bridge residual Mean", f"{aggregate['exclusive_bridge_residual_pnl']['mean']:+,.6f}"),
            ("Actual risk-util gap Mean", "-" if aggregate['exclusive_actual_risk_utilization_gap']['mean'] is None else f"{aggregate['exclusive_actual_risk_utilization_gap']['mean'] * 100.0:+.2f}pp"),
            ("13A Winner−Loser util Mean", "-" if aggregate['candidate_only_winner_minus_loser_risk_utilization']['mean'] is None else f"{aggregate['candidate_only_winner_minus_loser_risk_utilization']['mean'] * 100.0:+.2f}pp"),
            ("12B Winner−Loser util Mean", "-" if aggregate['comparator_only_winner_minus_loser_risk_utilization']['mean'] is None else f"{aggregate['comparator_only_winner_minus_loser_risk_utilization']['mean'] * 100.0:+.2f}pp"),
            ("Cash binding trades 13A/12B", f"{aggregate['candidate_only_cash_binding_total']}/{aggregate['comparator_only_cash_binding_total']}"),
            ("Position-cap binding trades 13A/12B", f"{aggregate['candidate_only_position_cap_binding_total']}/{aggregate['comparator_only_position_cap_binding_total']}"),
            ("Risk-cap binding trades 13A/12B", f"{aggregate['candidate_only_risk_cap_binding_total']}/{aggregate['comparator_only_risk_cap_binding_total']}"),
            ("Risk/Position tie trades 13A/12B", f"{aggregate['candidate_only_risk_position_tie_total']}/{aggregate['comparator_only_risk_position_tie_total']}"),
            ("Capital binding trades 13A/12B", f"{aggregate['candidate_only_capital_binding_total']}/{aggregate['comparator_only_capital_binding_total']}"),
            ("MaxQty binding trades 13A/12B", f"{aggregate['candidate_only_max_qty_binding_total']}/{aggregate['comparator_only_max_qty_binding_total']}"),
            ("Lot-rounding binding trades 13A/12B", f"{aggregate['candidate_only_lot_rounding_binding_total']}/{aggregate['comparator_only_lot_rounding_binding_total']}"),
            ("Common risk-size effect Mean", f"{aggregate['common_risk_size_effect_pnl']['mean']:+,.2f}"),
            ("ΔGap slot-days Mean", f"{aggregate['delta_end_position_gap_slot_days']['mean']:+.2f}"),
        )),
        render_table(("機制", "Seeds"), driver_rows),
        render_section("限制", number=6),
        "本Audit只做既有8-seed結果的trade-set／risk-dollar／slot／wealth-path歸因；ΔDL選擇R是兩個arm各自相對共同DL-off baseline的差，Direct-pair Exclusive ΔR則是MR-13A對MR-12B直接trade partition，兩者比較基準不同不得強制相等；不得挑best seed，不是新的promotion gate，也不得回流模型training semantics。",
    )).rstrip() + "\n"


def run_forward_robustness_portfolio_translation_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source = _resolve_source(root, definition)
    frame = source["seed_frame"]
    candidate = source["candidate_arm_id"]
    comparator = source["comparator_arm_id"]
    seed_summaries: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    wealth_frames: list[pd.DataFrame] = []
    capacity_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    execution_frames: list[pd.DataFrame] = []

    for seed_order, seed in source["seed_pairs"]:
        c_row = frame[(frame["arm_id"] == candidate) & (frame["seed"] == seed)].iloc[0].to_dict()
        b_row = frame[(frame["arm_id"] == comparator) & (frame["seed"] == seed)].iloc[0].to_dict()
        c_dir, c_manifest = source["units"][(candidate, seed)]
        b_dir, b_manifest = source["units"][(comparator, seed)]
        c_artifacts = _artifacts_from_unit(root, c_dir, c_manifest, c_row)
        b_artifacts = _artifacts_from_unit(root, b_dir, b_manifest, b_row)
        pair = build_strategy_attribution_pair_payload(
            candidate_artifacts=c_artifacts,
            comparator_artifacts=b_artifacts,
            focus_year=source["focus_year"],
            top_month_count=source["top_month_count"],
            top_trade_count=source["top_trade_count"],
        )
        trade_frame_for_binding = pd.DataFrame(
            dict(pair.get("frames") or {}).get("trade_contributions")
        ).copy()
        candidate_execution = pd.read_csv(c_artifacts.execution_path, encoding="utf-8-sig")
        comparator_execution = pd.read_csv(b_artifacts.execution_path, encoding="utf-8-sig")
        execution_binding = _exclusive_execution_binding_summary(
            trade_frame_for_binding, candidate_execution, comparator_execution
        )
        for arm_id, side, execution_frame in (
            (candidate, "candidate", candidate_execution),
            (comparator, "comparator", comparator_execution),
        ):
            ef = pd.DataFrame(execution_frame).copy()
            ef.insert(0, "arm_id", str(arm_id))
            ef.insert(0, "side", side)
            ef.insert(0, "seed", int(seed))
            ef.insert(0, "seed_order", int(seed_order))
            execution_frames.append(ef)
        summary_row = _row_summary(
            seed_order=seed_order,
            seed=seed,
            candidate_row=c_row,
            comparator_row=b_row,
            pair=pair,
            execution_binding=execution_binding,
        )
        if not math.isclose(
            summary_row["direct_pair_exclusive_delta_r"]
            + summary_row["direct_pair_common_delta_r"],
            summary_row["direct_pair_all_trade_delta_r"],
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                f"seed S{seed_order} direct-pair R decomposition不閉合"
            )
        bridge_sum = (
            summary_row["exclusive_equal_risk_selection_effect_pnl"]
            + summary_row["exclusive_average_risk_scale_effect_pnl"]
            + summary_row["exclusive_within_set_weighting_effect_pnl"]
            + summary_row["exclusive_uncovered_pnl_delta"]
        )
        if (
            not math.isclose(
                bridge_sum,
                summary_row["direct_pair_exclusive_delta_pnl"],
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                summary_row["exclusive_bridge_total_pnl_delta"],
                summary_row["direct_pair_exclusive_delta_pnl"],
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or abs(summary_row["exclusive_bridge_residual_pnl"]) > 1e-6
        ):
            raise RuntimeError(
                f"seed S{seed_order} exclusive R→Dollar decomposition不閉合"
            )
        seed_summaries.append(summary_row)
        frames = pair.pop("frames")
        for collection, key in (
            (trade_frames, "trade_contributions"),
            (wealth_frames, "daily_log_wealth"),
            (capacity_frames, "capacity_days"),
            (selection_frames, "selection_days"),
        ):
            f = pd.DataFrame(frames[key]).copy()
            f.insert(0, "seed", int(seed))
            f.insert(0, "seed_order", int(seed_order))
            collection.append(f)

    seed_df = pd.DataFrame(seed_summaries).sort_values("seed_order").reset_index(drop=True)
    ranking_positive = seed_df["delta_direct_selection_r"] > 0
    romd_positive = seed_df["delta_romd"] > 0
    direction_aligned = np.sign(seed_df["delta_direct_selection_r"]) == np.sign(seed_df["delta_romd"])
    aggregate = {
        "seed_count": int(len(seed_df)),
        "ranking_positive_count": int(ranking_positive.sum()),
        "ranking_positive_romd_positive_count": int((ranking_positive & romd_positive).sum()),
        "direction_aligned_count": int(direction_aligned.sum()),
        "direct_pair_exclusive_delta_r": _distribution(seed_df["direct_pair_exclusive_delta_r"]),
        "selection_basis_gap_r": _distribution(seed_df["selection_basis_gap_r"]),
        "direct_pair_exclusive_delta_pnl": _distribution(seed_df["direct_pair_exclusive_delta_pnl"]),
        "exclusive_risk_weighted_r_gap": _distribution(seed_df["exclusive_risk_weighted_r_gap"]),
        "exclusive_total_implied_risk_delta": _distribution(seed_df["exclusive_total_implied_risk_delta"]),
        "exclusive_equal_risk_selection_effect_pnl": _distribution(seed_df["exclusive_equal_risk_selection_effect_pnl"]),
        "exclusive_average_risk_scale_effect_pnl": _distribution(seed_df["exclusive_average_risk_scale_effect_pnl"]),
        "exclusive_within_set_weighting_effect_pnl": _distribution(seed_df["exclusive_within_set_weighting_effect_pnl"]),
        "exclusive_uncovered_pnl_delta": _distribution(seed_df["exclusive_uncovered_pnl_delta"]),
        "exclusive_bridge_residual_pnl": _distribution(seed_df["exclusive_bridge_residual_pnl"]),
        "exclusive_actual_risk_utilization_gap": _distribution(seed_df["exclusive_actual_risk_utilization_gap"]),
        "candidate_only_winner_minus_loser_risk_utilization": _distribution(seed_df["candidate_only_winner_minus_loser_risk_utilization"]),
        "comparator_only_winner_minus_loser_risk_utilization": _distribution(seed_df["comparator_only_winner_minus_loser_risk_utilization"]),
        "candidate_only_cash_binding_count": _distribution(seed_df["candidate_only_cash_binding_count"]),
        "comparator_only_cash_binding_count": _distribution(seed_df["comparator_only_cash_binding_count"]),
        "candidate_only_position_cap_binding_count": _distribution(seed_df["candidate_only_position_cap_binding_count"]),
        "comparator_only_position_cap_binding_count": _distribution(seed_df["comparator_only_position_cap_binding_count"]),
        "candidate_only_risk_position_tie_count": _distribution(seed_df["candidate_only_risk_position_tie_count"]),
        "comparator_only_risk_position_tie_count": _distribution(seed_df["comparator_only_risk_position_tie_count"]),
        "candidate_only_capital_binding_count": _distribution(seed_df["candidate_only_capital_binding_count"]),
        "comparator_only_capital_binding_count": _distribution(seed_df["comparator_only_capital_binding_count"]),
        "candidate_only_cash_binding_total": int(pd.to_numeric(seed_df["candidate_only_cash_binding_count"], errors="coerce").fillna(0).sum()),
        "comparator_only_cash_binding_total": int(pd.to_numeric(seed_df["comparator_only_cash_binding_count"], errors="coerce").fillna(0).sum()),
        "candidate_only_position_cap_binding_total": int(pd.to_numeric(seed_df["candidate_only_position_cap_binding_count"], errors="coerce").fillna(0).sum()),
        "comparator_only_position_cap_binding_total": int(pd.to_numeric(seed_df["comparator_only_position_cap_binding_count"], errors="coerce").fillna(0).sum()),
        "candidate_only_risk_position_tie_total": int(pd.to_numeric(seed_df["candidate_only_risk_position_tie_count"], errors="coerce").fillna(0).sum()),
        "comparator_only_risk_position_tie_total": int(pd.to_numeric(seed_df["comparator_only_risk_position_tie_count"], errors="coerce").fillna(0).sum()),
        "candidate_only_capital_binding_total": int(pd.to_numeric(seed_df["candidate_only_capital_binding_count"], errors="coerce").fillna(0).sum()),
        "comparator_only_capital_binding_total": int(pd.to_numeric(seed_df["comparator_only_capital_binding_count"], errors="coerce").fillna(0).sum()),
        "candidate_only_risk_cap_binding_total": int(pd.to_numeric(seed_df["candidate_only_risk_cap_binding_count"], errors="coerce").fillna(0).sum()),
        "comparator_only_risk_cap_binding_total": int(pd.to_numeric(seed_df["comparator_only_risk_cap_binding_count"], errors="coerce").fillna(0).sum()),
        "candidate_only_max_qty_binding_total": int(pd.to_numeric(seed_df["candidate_only_max_qty_binding_count"], errors="coerce").fillna(0).sum()),
        "comparator_only_max_qty_binding_total": int(pd.to_numeric(seed_df["comparator_only_max_qty_binding_count"], errors="coerce").fillna(0).sum()),
        "candidate_only_lot_rounding_binding_total": int(pd.to_numeric(seed_df["candidate_only_lot_rounding_binding_count"], errors="coerce").fillna(0).sum()),
        "comparator_only_lot_rounding_binding_total": int(pd.to_numeric(seed_df["comparator_only_lot_rounding_binding_count"], errors="coerce").fillna(0).sum()),
        "direct_pair_common_delta_r": _distribution(seed_df["direct_pair_common_delta_r"]),
        "direct_pair_all_trade_delta_r": _distribution(seed_df["direct_pair_all_trade_delta_r"]),
        "common_trade_pnl_delta": _distribution(seed_df["common_trade_pnl_delta"]),
        "common_risk_size_effect_pnl": _distribution(seed_df["common_risk_size_effect_pnl"]),
        "common_r_difference_effect_pnl": _distribution(seed_df["common_r_difference_effect_pnl"]),
        "delta_end_position_gap_slot_days": _distribution(seed_df["delta_end_position_gap_slot_days"]),
        "driver_counts": {str(k): int(v) for k, v in seed_df["driver"].value_counts().to_dict().items()},
    }
    payload = {
        "schema_version": AUDIT_RESULT_SCHEMA_VERSION,
        "created_at": get_taipei_now().isoformat(),
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "config": definition.as_dict(),
        "metadata": {
            "robustness_id": source["robustness_id"],
            "scientific_fingerprint": source["fingerprint"],
            "robustness_run": project_relative_display_path(source["run_root"], project_root=root),
            "candidate_arm_id": candidate,
            "comparator_arm_id": comparator,
            "seed_count": int(len(seed_df)),
            "all_seeds_required": True,
            "read_only": True,
            "training_performed": False,
            "portfolio_replay_executed": False,
        },
        "seed_summary": seed_df.to_dict("records"),
        "aggregate": aggregate,
    }
    payload = _json_native(payload)

    output_root = root / Path(AUDIT_OUTPUT_ROOT) / Path(definition.output_subdir)
    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")
    audit_run_dir = output_root / "runs" / timestamp
    latest_dir = output_root / "latest"
    audit_run_dir.mkdir(parents=True, exist_ok=False)
    report_path = audit_run_dir / "audit.md"
    json_path = audit_run_dir / "audit.json"
    report_path.write_text(_render_report(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    seed_df.to_csv(audit_run_dir / "seed_summary.csv", index=False, encoding="utf-8-sig")
    artifacts = [report_path, json_path, audit_run_dir / "seed_summary.csv"]
    for filename, frames in (
        ("trade_attribution.csv.gz", trade_frames),
        ("daily_wealth.csv.gz", wealth_frames),
        ("daily_capacity.csv.gz", capacity_frames),
        ("selection_days.csv.gz", selection_frames),
        ("execution_binding.csv.gz", execution_frames),
    ):
        path = audit_run_dir / filename
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
        artifacts.append(path)

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source_path in artifacts:
        shutil.copy2(source_path, latest_dir / source_path.name)

    if not quiet:
        print("\n" + _render_report(payload))
        print_artifact_paths(
            (("Audit Markdown", report_path), ("Audit JSON", json_path), ("最新Audit", latest_dir)),
            project_root=root,
        )
    return payload


__all__ = [
    "AUDIT_RESULT_SCHEMA_VERSION",
    "collect_forward_robustness_portfolio_translation_status",
    "run_forward_robustness_portfolio_translation_audit",
]

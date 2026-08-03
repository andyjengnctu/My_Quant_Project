"""CLI-only A～E optional-entry-filter × ranking gate for Selection PIT replay."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

from config.breakout_quality import get_breakout_quality_workflow_settings
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
)
from filters.breakout_quality.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
from tools.filters.breakout_quality.strategy_compare import (
    COMPARISON_MODE_SCORE_RANKING,
    OPTIONAL_ENTRY_FILTER_FIELDS,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    PARAM_POLICIES,
    PARAM_POLICY_BASE_FINALIST_BEST,
    run_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1

SCENARIO_SPECS: dict[str, dict[str, Any]] = {
    "A": {
        "label": "目前 filters＋原 buy-sort",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
        "ranking_policy": None,
    },
    "B": {
        "label": "目前 filters＋R3",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
        "ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    },
    "C": {
        "label": "Optional entry filters 全關＋原 buy-sort",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "ranking_policy": None,
    },
    "D": {
        "label": "Optional entry filters 全關＋R3",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    },
    "E": {
        "label": "Optional entry filters 全關＋原始 Score sort",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    },
}

PAIR_SPECS = (
    ("AB", "A", "B", OPTIONAL_ENTRY_FILTER_POLICY_CURRENT, BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET),
    ("CD", "C", "D", OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF, BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET),
    ("CE", "C", "E", OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF, BREAKOUT_QUALITY_RANKING_POLICY_SCORE),
)

_PORTFOLIO_METRICS = (
    ("淨總報酬", "total_return_pct", "%"),
    ("最大回撤", "max_drawdown_pct", "%"),
    ("報酬／最大回撤", "return_over_max_drawdown", ""),
    ("年化報酬", "annual_return_pct", "%"),
    ("Log R²", "log_r_squared", ""),
    ("月勝率", "monthly_win_rate_pct", "%"),
    ("EV", "expected_value_r", " R"),
    ("平均曝險", "avg_exposure_pct", "%"),
    ("保留買單成交率", "reserved_buy_fill_rate_pct", "%"),
)

_CAPTURE_METRICS = (
    ("平均實際投入", "avg_invested_total", ""),
    ("平均初始停損距離", "avg_stop_distance_pct", "%"),
    ("平均 Realized R", "avg_realized_r", " R"),
    ("平均 Target R", "avg_target_r", " R"),
    ("Aggregate capture", "aggregate_target_capture_ratio", ""),
    ("平均投入資金報酬", "avg_capital_return_pct", "%"),
)

_SELECTION_METRICS = (
    ("Orderable Score coverage", "orderable_score_coverage_rate", ""),
    ("Target percentile", "selected_target_percentile_mean", ""),
    ("Target top-k retention", "target_top_k_retention_mean", ""),
    ("Target opportunity gap", "target_opportunity_gap_r_mean", " R"),
    ("Target mean", "selected_target_mean_r", " R"),
)


def build_filter_gate_scenario_specs() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in SCENARIO_SPECS.items()}


def _parse_args(argv=None):
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "固定舊ROOS執行A～E Optional entry filters × Ranking Selection gate；"
            "不重訓模型、不執行optimizer。"
        )
    )
    parser.add_argument("--dataset", choices=("reduced", "full"), default=settings.strategy_dataset)
    parser.add_argument("--filter-id", default=settings.filter_id)
    parser.add_argument("--model-architecture", default=settings.model_architecture)
    parser.add_argument("--experiment-profile", default=settings.experiment_profile)
    parser.add_argument("--params", default=None)
    parser.add_argument("--param-policy", choices=PARAM_POLICIES, default=PARAM_POLICY_BASE_FINALIST_BEST)
    parser.add_argument("--max-positions", type=int, default=settings.strategy_max_positions)
    parser.add_argument("--rotation", choices=("off", "on"), default=settings.strategy_rotation)
    parser.add_argument("--fixed-risk", type=float, default=None)
    parser.add_argument("--max-position-cap-pct", type=float, default=None)
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default="2020-12-31")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_delta(right: dict[str, Any], left: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in sorted(set(left) | set(right)):
        lv = left.get(key)
        rv = right.get(key)
        if (
            isinstance(lv, (int, float))
            and not isinstance(lv, bool)
            and isinstance(rv, (int, float))
            and not isinstance(rv, bool)
            and math.isfinite(float(lv))
            and math.isfinite(float(rv))
        ):
            output[key] = float(rv) - float(lv)
    return output


def _delta_of_deltas(
    right_delta: dict[str, float], left_delta: dict[str, float]
) -> dict[str, float]:
    return {
        key: float(right_delta[key]) - float(left_delta[key])
        for key in sorted(set(right_delta) & set(left_delta))
    }


def _scenario_from_pair(payload: dict[str, Any], *, active: bool) -> dict[str, Any]:
    scenario_key = "score_ranking" if active else "no_filter"
    selection_key = scenario_key
    capture_key = "score_sort" if active else "baseline"
    capture_payload = dict(payload.get("score_ranking_capture_audit") or {})
    selection_payload = dict(payload.get("selection_diagnostics") or {})
    return {
        "portfolio": dict(payload.get(scenario_key) or {}),
        "selection": dict(selection_payload.get(selection_key) or {}),
        "capture": dict(capture_payload.get(capture_key) or {}),
    }


def _scenario_delta(right: dict[str, Any], left: dict[str, Any]) -> dict[str, Any]:
    return {
        section: _numeric_delta(
            dict(right.get(section) or {}),
            dict(left.get(section) or {}),
        )
        for section in ("portfolio", "selection", "capture")
    }


def _interaction_delta(
    right: dict[str, Any], left: dict[str, Any]
) -> dict[str, Any]:
    return {
        section: _delta_of_deltas(
            dict(right.get(section) or {}),
            dict(left.get(section) or {}),
        )
        for section in ("portfolio", "selection", "capture")
    }


def _assert_equal_files(left: Path, right: Path, *, label: str) -> None:
    if not left.is_file() or not right.is_file():
        raise FileNotFoundError(f"{label}一致性檢查缺少工件: {left} / {right}")
    left_hash = _sha256_file(left)
    right_hash = _sha256_file(right)
    if left_hash != right_hash:
        raise ValueError(
            f"{label}必須完全一致但hash不同: left={left_hash}, right={right_hash}"
        )


def _run_pair(
    *,
    project_root: Path,
    pair_id: str,
    output_dir: Path,
    optional_entry_filter_policy: str,
    ranking_policy: str,
    args,
) -> dict[str, Any]:
    print(f"[{pair_id}] 執行 {optional_entry_filter_policy} filters × {ranking_policy}")
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            payload = run_comparison(
                project_root=project_root,
                dataset=args.dataset,
                params_path=args.params,
                param_policy=args.param_policy,
                max_positions=args.max_positions,
                enable_rotation=args.rotation == "on",
                fixed_risk=args.fixed_risk,
                max_position_cap_pct=args.max_position_cap_pct,
                comparison_mode=COMPARISON_MODE_SCORE_RANKING,
                ranking_policy=ranking_policy,
                optional_entry_filter_policy=optional_entry_filter_policy,
                filter_id=args.filter_id,
                score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                model_architecture=args.model_architecture,
                experiment_profile=args.experiment_profile,
                output_dir_override=output_dir,
                comparison_start_date=args.start_date,
                comparison_end_date=args.end_date,
                quiet=args.quiet,
            )
    except Exception:
        captured = buffer.getvalue().strip()
        if captured:
            print(captured)
        raise
    print(f"[{pair_id}] 完成")
    return payload


def _fmt(value: Any, *, digits: int = 2, unit: str = "", signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}{unit}"


def _scenario_table(
    scenarios: dict[str, dict[str, Any]],
    specs: tuple[tuple[str, str, str], ...],
    section: str,
) -> list[tuple[str, ...]]:
    rows = []
    for label, key, unit in specs:
        digits = 4 if key == "log_r_squared" else 0 if key == "avg_invested_total" else 2
        rows.append(
            (
                label,
                *(
                    _fmt(
                        scenarios[scenario][section].get(key),
                        digits=digits,
                        unit=unit,
                    )
                    for scenario in ("A", "B", "C", "D", "E")
                ),
            )
        )
    return rows


def _comparison_rows(comparisons: dict[str, dict[str, Any]]) -> list[tuple[str, ...]]:
    keys = (
        ("B−A：目前filters下R3效果", "B_minus_A"),
        ("D−C：filters全關下R3效果", "D_minus_C"),
        ("E−C：filters全關下原始Score效果", "E_minus_C"),
        ("C−A：關filters對原排序效果", "C_minus_A"),
        ("D−B：關filters對R3效果", "D_minus_B"),
        ("D−E：filters全關時R3相對原始Score", "D_minus_E"),
        ("R3 interaction：(D−C)−(B−A)", "r3_filter_interaction"),
    )
    rows = []
    for label, key in keys:
        delta = comparisons[key]
        portfolio = delta["portfolio"]
        capture = delta["capture"]
        selection = delta["selection"]
        rows.append((
            label,
            _fmt(portfolio.get("total_return_pct"), digits=2, unit="pp", signed=True),
            _fmt(portfolio.get("max_drawdown_pct"), digits=2, unit="pp", signed=True),
            _fmt(portfolio.get("return_over_max_drawdown"), digits=2, signed=True),
            _fmt(capture.get("avg_realized_r"), digits=2, unit=" R", signed=True),
            _fmt(selection.get("selected_target_mean_r"), digits=2, unit=" R", signed=True),
            _fmt(capture.get("aggregate_target_capture_ratio"), digits=2, signed=True),
        ))
    return rows


def _render_console(result: dict[str, Any]) -> str:
    metadata = result["metadata"]
    scenarios = result["scenarios"]
    comparisons = result["comparisons"]
    lines = [
        render_title("Optional Entry Filters × Ranking A～E Gate"),
        render_key_values((
            ("期間", f"{metadata['comparison_period']['start']} ～ {metadata['comparison_period']['end']}"),
            ("參數", metadata["param_policy"]),
            ("Score source", SCORE_SOURCE_SELECTION_POINT_IN_TIME),
            ("Optional filters", ", ".join(metadata["optional_entry_filter_fields"])),
            ("Future Target runtime", "未使用"),
        )),
        render_section("情境"),
        render_table(
            ["組別", "Filters", "Ranking"],
            [
                (
                    key,
                    "目前" if spec["optional_entry_filter_policy"] == OPTIONAL_ENTRY_FILTER_POLICY_CURRENT else "全關",
                    "原 buy-sort" if spec["ranking_policy"] is None else spec["ranking_policy"],
                )
                for key, spec in SCENARIO_SPECS.items()
            ],
        ),
        render_section("投組報酬與風險", number=1),
        render_table(
            ["指標", "A", "B", "C", "D", "E"],
            _scenario_table(scenarios, _PORTFOLIO_METRICS, "portfolio"),
        ),
        render_section("資金、Target與Capture", number=2),
        render_table(
            ["指標", "A", "B", "C", "D", "E"],
            _scenario_table(scenarios, _CAPTURE_METRICS, "capture"),
        ),
        render_section("模型選股能力", number=3),
        render_table(
            ["指標", "A", "B", "C", "D", "E"],
            _scenario_table(scenarios, _SELECTION_METRICS, "selection"),
        ),
        render_section("主要差異", number=4),
        render_table(
            ["比較", "總報酬", "MDD", "RoMD", "Realized R", "Target mean", "Capture"],
            _comparison_rows(comparisons),
        ),
        render_section("判讀契約", number=5),
        "- D−C 與 B−A比較R3在filters全關前後的效果。",
        "- E−C檢查filters全關後，原始Score sort是否恢復；D−E判斷資金分桶是否仍有必要。",
        "- R3 interaction = (D−C)−(B−A)；正值只代表關filters後R3相對效果增加，不等於正式採用。",
        "- 不重訓模型、不執行optimizer；Future Target只在回放後離線join。",
    ]
    return "\n".join(lines)


def _render_markdown(result: dict[str, Any]) -> str:
    metadata = result["metadata"]
    scenarios = result["scenarios"]
    comparisons = result["comparisons"]
    lines = [
        "# Optional Entry Filters × Ranking A～E Gate",
        "",
        f"- 期間：`{metadata['comparison_period']['start']}` ～ `{metadata['comparison_period']['end']}`",
        f"- 參數 policy：`{metadata['param_policy']}`",
        f"- Optional filters：`{', '.join(metadata['optional_entry_filter_fields'])}`",
        "- Score source：`selection_point_in_time`",
        "- Future Target：只於portfolio replay完成後離線join，未進入runtime。",
        "",
        "## 情境",
        "",
        "| 組別 | Filters | Ranking |",
        "|---|---|---|",
    ]
    for key, spec in SCENARIO_SPECS.items():
        filters = "目前" if spec["optional_entry_filter_policy"] == OPTIONAL_ENTRY_FILTER_POLICY_CURRENT else "全關"
        ranking = "原 buy-sort" if spec["ranking_policy"] is None else spec["ranking_policy"]
        lines.append(f"| {key} | {filters} | `{ranking}` |")

    def add_metric_table(title: str, section: str, specs):
        lines.extend(["", f"## {title}", "", "| 指標 | A | B | C | D | E |", "|---|---:|---:|---:|---:|---:|"])
        for row in _scenario_table(scenarios, specs, section):
            lines.append("| " + " | ".join(row) + " |")

    add_metric_table("投組報酬與風險", "portfolio", _PORTFOLIO_METRICS)
    add_metric_table("資金、Target與Capture", "capture", _CAPTURE_METRICS)
    add_metric_table("模型選股能力", "selection", _SELECTION_METRICS)
    lines.extend([
        "", "## 主要差異", "",
        "| 比較 | 總報酬 | MDD | RoMD | Realized R | Target mean | Capture |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in _comparison_rows(comparisons):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend([
        "", "## 判讀契約", "",
        "- `D−C` 與 `B−A` 比較R3在filters全關前後的效果。",
        "- `E−C` 檢查filters全關後原始Score sort是否恢復；`D−E`判斷資金分桶是否仍有必要。",
        "- `R3 interaction = (D−C)−(B−A)`；正值只代表關filters後R3相對效果增加，不等於正式採用。",
        "- 本Gate不重訓模型、不執行optimizer，也不修改正式Baseline。",
        "",
    ])
    return "\n".join(lines)


def run_filter_gate(*, args, project_root=PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if args.max_positions < 1:
        raise ValueError("max_positions必須>=1")
    if (args.start_date is None) != (args.end_date is None):
        raise ValueError("--start-date與--end-date必須同時提供")
    output_dir = (
        resolve_filter_model_output_dir(
            str(root), args.filter_id, args.model_architecture, args.experiment_profile
        )
        / f"strategy_filter_gate_{str(args.param_policy).replace('-', '_')}_selection_point_in_time"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_payloads: dict[str, dict[str, Any]] = {}
    pair_dirs: dict[str, Path] = {}
    for pair_id, _left, _right, filter_policy, ranking_policy in PAIR_SPECS:
        pair_dir = output_dir / f"pair_{pair_id.lower()}"
        pair_dirs[pair_id] = pair_dir
        pair_payloads[pair_id] = _run_pair(
            project_root=root,
            pair_id=pair_id,
            output_dir=pair_dir,
            optional_entry_filter_policy=filter_policy,
            ranking_policy=ranking_policy,
            args=args,
        )

    for filename in ("no_filter_equity.csv", "no_filter_trades.csv", "no_filter_daily_capacity.csv"):
        _assert_equal_files(
            pair_dirs["CD"] / filename,
            pair_dirs["CE"] / filename,
            label=f"C scenario {filename}",
        )

    scenarios = {
        "A": _scenario_from_pair(pair_payloads["AB"], active=False),
        "B": _scenario_from_pair(pair_payloads["AB"], active=True),
        "C": _scenario_from_pair(pair_payloads["CD"], active=False),
        "D": _scenario_from_pair(pair_payloads["CD"], active=True),
        "E": _scenario_from_pair(pair_payloads["CE"], active=True),
    }
    comparisons = {
        "B_minus_A": _scenario_delta(scenarios["B"], scenarios["A"]),
        "D_minus_C": _scenario_delta(scenarios["D"], scenarios["C"]),
        "E_minus_C": _scenario_delta(scenarios["E"], scenarios["C"]),
        "C_minus_A": _scenario_delta(scenarios["C"], scenarios["A"]),
        "D_minus_B": _scenario_delta(scenarios["D"], scenarios["B"]),
        "D_minus_E": _scenario_delta(scenarios["D"], scenarios["E"]),
    }
    comparisons["r3_filter_interaction"] = _interaction_delta(
        comparisons["D_minus_C"], comparisons["B_minus_A"]
    )

    ab_meta = pair_payloads["AB"]["metadata"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "comparison_period": dict(ab_meta.get("comparison_period") or {}),
            "dataset": args.dataset,
            "filter_id": args.filter_id,
            "model_architecture": args.model_architecture,
            "experiment_profile": args.experiment_profile,
            "param_policy": args.param_policy,
            "params_path": ab_meta.get("params_path"),
            "params_file_sha256": ab_meta.get("params_file_sha256"),
            "optional_entry_filter_fields": list(OPTIONAL_ENTRY_FILTER_FIELDS),
            "scenario_specs": build_filter_gate_scenario_specs(),
            "pair_artifacts": {
                key: project_relative_display_path(value, project_root=root)
                for key, value in pair_dirs.items()
            },
            "future_target_runtime_used": False,
            "model_retrained": False,
            "optimizer_executed": False,
        },
        "scenarios": scenarios,
        "comparisons": comparisons,
    }
    json_path = output_dir / "strategy_filter_gate.json"
    markdown_path = output_dir / "strategy_filter_gate.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    print("\n" + _render_console(result))
    print_artifact_paths(
        (("A～E Gate Markdown", markdown_path), ("A～E Gate JSON", json_path)),
        project_root=root,
    )
    return result


def main(argv=None):
    args = _parse_args(argv)
    run_filter_gate(args=args)
    return 0


__all__ = [
    "main",
    "run_filter_gate",
    "build_filter_gate_scenario_specs",
    "SCENARIO_SPECS",
    "PAIR_SPECS",
]

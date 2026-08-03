"""CLI-only A/B/C/F gate for replacing optional entry filters with the binary DL filter."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
from typing import Any

from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_STRATEGY_DATASET,
    BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS,
    BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY,
    BREAKOUT_QUALITY_STRATEGY_ROTATION,
)
from core.buy_sort import BREAKOUT_QUALITY_RANKING_POLICY_SCORE
from filters.breakout_quality.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_CANONICAL_RUNTIME
from tools.filters.breakout_quality.strategy_compare import (
    COMPARISON_MODE_HARD_FILTER,
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
        "label": "目前 optional filters＋原 buy-sort",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
        "dl_filter_enabled": False,
    },
    "B": {
        "label": "目前 optional filters＋Binary DL filter＋原 buy-sort",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
        "dl_filter_enabled": True,
    },
    "C": {
        "label": "Optional entry filters 全關＋原 buy-sort",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "dl_filter_enabled": False,
    },
    "F": {
        "label": "Optional entry filters 全關＋Binary DL filter＋原 buy-sort",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "dl_filter_enabled": True,
    },
}

PAIR_SPECS = (
    ("AB", "A", "B", OPTIONAL_ENTRY_FILTER_POLICY_CURRENT),
    ("CF", "C", "F", OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF),
)

_PORTFOLIO_METRICS = (
    ("淨總報酬", "total_return_pct", "%"),
    ("最大回撤", "max_drawdown_pct", "%"),
    ("報酬／最大回撤", "return_over_max_drawdown", ""),
    ("年化報酬", "annual_return_pct", "%"),
    ("Log R²", "log_r_squared", ""),
    ("月勝率", "monthly_win_rate_pct", "%"),
    ("交易數", "trade_count", ""),
    ("勝率", "win_rate_pct", "%"),
    ("Payoff", "payoff_ratio", ""),
    ("EV", "expected_value_r", " R"),
    ("平均曝險", "avg_exposure_pct", "%"),
    ("保留買單成交率", "reserved_buy_fill_rate_pct", "%"),
    ("平均每日可掛單候選", "avg_orderable_candidates", ""),
    ("候選供給不足日", "candidate_supply_gap_days", " 日"),
    ("每日結束未滿倉日", "underfilled_end_days", " 日"),
    ("每日結束持股缺口", "end_position_gap_slot_days", " 格日"),
)

_COMPARISON_SPECS = (
    ("B−A：DL 疊加目前 filters", "B_minus_A"),
    ("F−C：DL 取代 optional filters", "F_minus_C"),
    ("C−A：只移除 optional filters", "C_minus_A"),
    ("F−A：DL-only quality gate 對目前策略", "F_minus_A"),
    ("DL replacement interaction：(F−C)−(B−A)", "replacement_interaction"),
)

_ATTRIBUTION_SPECS = (
    ("總 R 差異", "reconstructed_total_r_delta", " R"),
    ("獨有交易選擇差異", "exclusive_selection_delta_r", " R"),
    ("被排除贏家 R", "excluded_winner_r", " R"),
    ("避開輸家 |R|", "avoided_loser_r_abs", " R"),
    ("替代贏家 R", "replacement_winner_r", " R"),
    ("替代輸家 |R|", "replacement_loser_r_abs", " R"),
    ("直接 DL 拒絕交易數", "direct_filter_reject_count", ""),
    ("投組路徑擠出交易數", "portfolio_path_displacement_count", ""),
)


def build_dl_filter_gate_scenario_specs() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in SCENARIO_SPECS.items()}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "固定正式rolling params執行A/B/C/F Binary DL filter replacement gate；"
            "不重訓模型、不調threshold、不改原buy-sort。"
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("reduced", "full"),
        default=BREAKOUT_QUALITY_STRATEGY_DATASET,
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument("--model-architecture", default=BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    parser.add_argument("--experiment-profile", default=BREAKOUT_QUALITY_EXPERIMENT_PROFILE)
    parser.add_argument("--params", default=None)
    parser.add_argument(
        "--param-policy",
        choices=PARAM_POLICIES,
        default=(
            BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY
            if BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY in PARAM_POLICIES
            else PARAM_POLICY_BASE_FINALIST_BEST
        ),
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=int(BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS),
    )
    parser.add_argument(
        "--rotation",
        choices=("off", "on"),
        default=str(BREAKOUT_QUALITY_STRATEGY_ROTATION),
    )
    parser.add_argument("--fixed-risk", type=float, default=None)
    parser.add_argument("--max-position-cap-pct", type=float, default=None)
    parser.add_argument(
        "--start-date",
        default=None,
        help="選填；未指定時使用9A canonical runtime score的正式execution start。",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="選填；未指定時使用9A canonical runtime score的available through。",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _numeric_delta(right: dict[str, Any], left: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in sorted(set(left) | set(right)):
        left_value = left.get(key)
        right_value = right.get(key)
        if (
            isinstance(left_value, (int, float))
            and not isinstance(left_value, bool)
            and isinstance(right_value, (int, float))
            and not isinstance(right_value, bool)
            and math.isfinite(float(left_value))
            and math.isfinite(float(right_value))
        ):
            output[key] = float(right_value) - float(left_value)
    return output


def _delta_of_deltas(
    right_delta: dict[str, float], left_delta: dict[str, float]
) -> dict[str, float]:
    return {
        key: float(right_delta[key]) - float(left_delta[key])
        for key in sorted(set(right_delta) & set(left_delta))
    }


def _fmt(value: Any, *, digits: int = 2, unit: str = "", signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}{unit}"


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"讀取{label}失敗: {path}｜{type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}根節點必須是object: {path}")
    return payload


def _run_pair(
    *,
    project_root: Path,
    pair_id: str,
    output_dir: Path,
    optional_entry_filter_policy: str,
    args,
) -> dict[str, Any]:
    label = "current" if optional_entry_filter_policy == OPTIONAL_ENTRY_FILTER_POLICY_CURRENT else "all-off"
    print(f"[{pair_id}] 執行 {label} optional filters × Binary DL hard filter")
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            comparison = run_comparison(
                project_root=project_root,
                dataset=args.dataset,
                params_path=args.params,
                param_policy=args.param_policy,
                max_positions=args.max_positions,
                enable_rotation=args.rotation == "on",
                fixed_risk=args.fixed_risk,
                max_position_cap_pct=args.max_position_cap_pct,
                comparison_mode=COMPARISON_MODE_HARD_FILTER,
                ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
                optional_entry_filter_policy=optional_entry_filter_policy,
                filter_id=args.filter_id,
                score_source=SCORE_SOURCE_CANONICAL_RUNTIME,
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
    attribution = _read_json_object(
        output_dir / "trade_attribution.json",
        label=f"{pair_id} trade attribution",
    )
    print(f"[{pair_id}] 完成")
    return {"comparison": comparison, "attribution": attribution}


def _scenario_from_pair(payload: dict[str, Any], *, active: bool) -> dict[str, Any]:
    comparison = dict(payload.get("comparison") or {})
    scenario_key = "quality_filter" if active else "no_filter"
    return dict(comparison.get(scenario_key) or {})


def _assert_pair_compatibility(ab: dict[str, Any], cf: dict[str, Any]) -> None:
    left = dict((ab.get("comparison") or {}).get("metadata") or {})
    right = dict((cf.get("comparison") or {}).get("metadata") or {})
    shared_fields = (
        "comparison_mode",
        "score_source",
        "dataset",
        "params_file_sha256",
        "param_source_kind",
        "requested_param_policy",
        "param_selector",
        "runtime_member_count_min",
        "runtime_member_count_max",
        "runtime_min_agree",
        "comparison_design",
        "lookahead_safe_active_param_schedule",
        "filter_id",
        "model_architecture",
        "experiment_profile",
        "threshold",
        "benchmark_ticker",
        "max_positions",
        "enable_rotation",
        "comparison_period",
        "score_signal_coverage",
        "runtime_manifest_path",
        "runtime_score_path",
    )
    mismatched = [field for field in shared_fields if left.get(field) != right.get(field)]
    if mismatched:
        raise ValueError(f"A/B與C/F Gate共同契約不一致: {mismatched}")
    if left.get("optional_entry_filter_policy") != OPTIONAL_ENTRY_FILTER_POLICY_CURRENT:
        raise ValueError("A/B pair必須沿用目前optional entry filters")
    if right.get("optional_entry_filter_policy") != OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF:
        raise ValueError("C/F pair必須關閉五個optional entry filters")
    forced = dict(right.get("optional_entry_filter_forced_values") or {})
    expected_forced = {field: False for field in OPTIONAL_ENTRY_FILTER_FIELDS}
    if forced != expected_forced:
        raise ValueError(f"C/F optional filter override不完整: {forced}")


def _scenario_table(
    scenarios: dict[str, dict[str, Any]],
    specs: tuple[tuple[str, str, str], ...],
) -> list[tuple[str, ...]]:
    rows = []
    for label, key, unit in specs:
        digits = (
            4
            if key == "log_r_squared"
            else 0
            if key in {
                "trade_count",
                "candidate_supply_gap_days",
                "underfilled_end_days",
                "end_position_gap_slot_days",
            }
            else 2
        )
        rows.append(
            (
                label,
                *(
                    _fmt(scenarios[scenario].get(key), digits=digits, unit=unit)
                    for scenario in ("A", "B", "C", "F")
                ),
            )
        )
    return rows


def _comparison_rows(comparisons: dict[str, dict[str, float]]) -> list[tuple[str, ...]]:
    rows = []
    for label, key in _COMPARISON_SPECS:
        delta = comparisons[key]
        rows.append(
            (
                label,
                _fmt(delta.get("total_return_pct"), digits=2, unit="pp", signed=True),
                _fmt(delta.get("max_drawdown_pct"), digits=2, unit="pp", signed=True),
                _fmt(delta.get("return_over_max_drawdown"), digits=2, signed=True),
                _fmt(delta.get("expected_value_r"), digits=2, unit=" R", signed=True),
                _fmt(delta.get("avg_exposure_pct"), digits=2, unit="pp", signed=True),
                _fmt(delta.get("trade_count"), digits=0, signed=True),
            )
        )
    return rows


def _attribution_rows(pair_effects: dict[str, dict[str, Any]]) -> list[tuple[str, ...]]:
    rows = []
    for label, key, unit in _ATTRIBUTION_SPECS:
        digits = 0 if key.endswith("_count") else 2
        rows.append(
            (
                label,
                _fmt(pair_effects["B_minus_A"].get(key), digits=digits, unit=unit),
                _fmt(pair_effects["F_minus_C"].get(key), digits=digits, unit=unit),
            )
        )
    return rows


def _yearly_rows(pair_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ab_rows = list((pair_payloads["AB"].get("comparison") or {}).get("yearly") or [])
    cf_rows = list((pair_payloads["CF"].get("comparison") or {}).get("yearly") or [])
    ab_by_year = {int(row["year"]): dict(row) for row in ab_rows}
    cf_by_year = {int(row["year"]): dict(row) for row in cf_rows}
    years = sorted(set(ab_by_year) | set(cf_by_year))
    output = []
    for year in years:
        ab = ab_by_year.get(year, {})
        cf = cf_by_year.get(year, {})
        output.append(
            {
                "year": year,
                "A_return_pct": ab.get("no_filter_return_pct"),
                "B_return_pct": ab.get("quality_filter_return_pct"),
                "C_return_pct": cf.get("no_filter_return_pct"),
                "F_return_pct": cf.get("quality_filter_return_pct"),
                "F_minus_A_pct": (
                    float(cf["quality_filter_return_pct"])
                    - float(ab["no_filter_return_pct"])
                    if "quality_filter_return_pct" in cf and "no_filter_return_pct" in ab
                    else None
                ),
                "is_full_year": bool(ab.get("is_full_year", cf.get("is_full_year", False))),
            }
        )
    return output


def _render_console(result: dict[str, Any]) -> str:
    metadata = result["metadata"]
    scenarios = result["scenarios"]
    comparisons = result["comparisons"]
    yearly = result["yearly"]
    lines = [
        render_title("Binary DL Filter Replacement A／B／C／F Gate"),
        render_key_values(
            (
                (
                    "期間",
                    f"{metadata['comparison_period']['start']} ～ {metadata['comparison_period']['end']}",
                ),
                ("參數", metadata["param_policy"]),
                (
                    "Binary model",
                    f"{metadata['model_architecture']} / {metadata['experiment_profile']}",
                ),
                ("Threshold", _fmt(metadata["threshold"], digits=2)),
                ("Score source", SCORE_SOURCE_CANONICAL_RUNTIME),
                ("Buy sort", "原 position-aware buy-sort（四組相同）"),
            )
        ),
        render_section("情境"),
        render_table(
            ["組別", "Optional filters", "Binary DL filter", "Ranking"],
            [
                (
                    key,
                    "目前"
                    if spec["optional_entry_filter_policy"]
                    == OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
                    else "全關",
                    "開" if spec["dl_filter_enabled"] else "關",
                    "原 buy-sort",
                )
                for key, spec in SCENARIO_SPECS.items()
            ],
        ),
        render_section("投組報酬、風險與資金使用", number=1),
        render_table(
            ["指標", "A", "B", "C", "F"],
            _scenario_table(scenarios, _PORTFOLIO_METRICS),
        ),
        render_section("主要差異", number=2),
        render_table(
            ["比較", "總報酬", "MDD", "RoMD", "EV", "平均曝險", "交易數"],
            _comparison_rows(comparisons),
        ),
        render_section("DL Filter交易歸因", number=3),
        render_table(
            ["指標", "B−A 疊加", "F−C 替代"],
            _attribution_rows(result["pair_effects"]),
        ),
        render_section("年度報酬", number=4),
        render_table(
            ["年度", "A", "B", "C", "F", "F−A", "完整年度"],
            [
                (
                    str(row["year"]),
                    _fmt(row.get("A_return_pct"), digits=2, unit="%"),
                    _fmt(row.get("B_return_pct"), digits=2, unit="%"),
                    _fmt(row.get("C_return_pct"), digits=2, unit="%"),
                    _fmt(row.get("F_return_pct"), digits=2, unit="%"),
                    _fmt(row.get("F_minus_A_pct"), digits=2, unit="pp", signed=True),
                    "是" if row.get("is_full_year") else "否",
                )
                for row in yearly
            ],
        ),
        render_section("判讀契約", number=5),
        "- F−C回答Binary DL在沒有規則品質filters時，能否單獨提供有效買入確認。",
        "- B−A回答Binary DL疊加目前filters後是否有效；F−A才是DL-only replacement的正式比較。",
        "- interaction=(F−C)−(B−A)只判斷替代是否優於疊加，不可取代F相對A的絕對績效。",
        "- 四組沿用原position-aware buy-sort；不使用continuous Score、R3、Future Target或optimizer。",
    ]
    return "\n".join(lines)


def _render_markdown(result: dict[str, Any]) -> str:
    metadata = result["metadata"]
    scenarios = result["scenarios"]
    comparisons = result["comparisons"]
    lines = [
        "# Binary DL Filter Replacement A／B／C／F Gate",
        "",
        f"- 期間：`{metadata['comparison_period']['start']}` ～ `{metadata['comparison_period']['end']}`",
        f"- 參數 policy：`{metadata['param_policy']}`",
        f"- Binary model：`{metadata['model_architecture']} / {metadata['experiment_profile']}`",
        f"- Threshold：`{metadata['threshold']}`",
        "- Score source：`canonical_runtime`",
        "- Ranking：四組均使用原 position-aware buy-sort。",
        "- Future Target：未進入本Gate。",
        "",
        "## 情境",
        "",
        "| 組別 | Optional filters | Binary DL filter | Ranking |",
        "|---|---|---|---|",
    ]
    for key, spec in SCENARIO_SPECS.items():
        filter_text = (
            "目前"
            if spec["optional_entry_filter_policy"] == OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
            else "全關"
        )
        lines.append(
            f"| {key} | {filter_text} | {'開' if spec['dl_filter_enabled'] else '關'} | 原 buy-sort |"
        )
    lines.extend(
        [
            "",
            "## 投組報酬、風險與資金使用",
            "",
            "| 指標 | A | B | C | F |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in _scenario_table(scenarios, _PORTFOLIO_METRICS):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## 主要差異",
            "",
            "| 比較 | 總報酬 | MDD | RoMD | EV | 平均曝險 | 交易數 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _comparison_rows(comparisons):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## DL Filter交易歸因",
            "",
            "| 指標 | B−A 疊加 | F−C 替代 |",
            "|---|---:|---:|",
        ]
    )
    for row in _attribution_rows(result["pair_effects"]):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## 年度報酬",
            "",
            "| 年度 | A | B | C | F | F−A | 完整年度 |",
            "|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in result["yearly"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["year"]),
                    _fmt(row.get("A_return_pct"), digits=2, unit="%"),
                    _fmt(row.get("B_return_pct"), digits=2, unit="%"),
                    _fmt(row.get("C_return_pct"), digits=2, unit="%"),
                    _fmt(row.get("F_return_pct"), digits=2, unit="%"),
                    _fmt(row.get("F_minus_A_pct"), digits=2, unit="pp", signed=True),
                    "是" if row.get("is_full_year") else "否",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 判讀契約",
            "",
            "- `F−C`回答Binary DL在沒有規則品質filters時，能否單獨提供有效買入確認。",
            "- `B−A`回答Binary DL疊加目前filters後是否有效；`F−A`才是DL-only replacement的正式比較。",
            "- `interaction=(F−C)−(B−A)`只判斷替代是否優於疊加，不可取代F相對A的絕對績效。",
            "- 四組沿用原position-aware buy-sort；不使用continuous Score、R3、Future Target或optimizer。",
            "",
        ]
    )
    return "\n".join(lines)


def run_dl_filter_gate(*, args, project_root=PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if args.max_positions < 1:
        raise ValueError("max_positions必須>=1")
    if (args.start_date is None) != (args.end_date is None):
        raise ValueError("--start-date與--end-date必須同時提供")
    output_dir = (
        resolve_filter_model_output_dir(
            str(root), args.filter_id, args.model_architecture, args.experiment_profile
        )
        / (
            "strategy_dl_filter_gate_"
            f"{str(args.param_policy).replace('-', '_')}_canonical_runtime"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_payloads: dict[str, dict[str, Any]] = {}
    pair_dirs: dict[str, Path] = {}
    for pair_id, _left, _right, filter_policy in PAIR_SPECS:
        pair_dir = output_dir / f"pair_{pair_id.lower()}"
        pair_dirs[pair_id] = pair_dir
        pair_payloads[pair_id] = _run_pair(
            project_root=root,
            pair_id=pair_id,
            output_dir=pair_dir,
            optional_entry_filter_policy=filter_policy,
            args=args,
        )

    _assert_pair_compatibility(pair_payloads["AB"], pair_payloads["CF"])
    scenarios = {
        "A": _scenario_from_pair(pair_payloads["AB"], active=False),
        "B": _scenario_from_pair(pair_payloads["AB"], active=True),
        "C": _scenario_from_pair(pair_payloads["CF"], active=False),
        "F": _scenario_from_pair(pair_payloads["CF"], active=True),
    }
    comparisons = {
        "B_minus_A": _numeric_delta(scenarios["B"], scenarios["A"]),
        "F_minus_C": _numeric_delta(scenarios["F"], scenarios["C"]),
        "C_minus_A": _numeric_delta(scenarios["C"], scenarios["A"]),
        "F_minus_A": _numeric_delta(scenarios["F"], scenarios["A"]),
    }
    comparisons["replacement_interaction"] = _delta_of_deltas(
        comparisons["F_minus_C"], comparisons["B_minus_A"]
    )
    pair_effects = {
        "B_minus_A": dict(
            (pair_payloads["AB"].get("attribution") or {}).get("r_attribution") or {}
        ),
        "F_minus_C": dict(
            (pair_payloads["CF"].get("attribution") or {}).get("r_attribution") or {}
        ),
    }
    ab_meta = dict((pair_payloads["AB"].get("comparison") or {}).get("metadata") or {})
    result = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "comparison_period": dict(ab_meta.get("comparison_period") or {}),
            "dataset": args.dataset,
            "filter_id": args.filter_id,
            "model_architecture": args.model_architecture,
            "experiment_profile": args.experiment_profile,
            "threshold": float(ab_meta.get("threshold", BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD)),
            "param_policy": args.param_policy,
            "params_path": ab_meta.get("params_path"),
            "params_file_sha256": ab_meta.get("params_file_sha256"),
            "optional_entry_filter_fields": list(OPTIONAL_ENTRY_FILTER_FIELDS),
            "scenario_specs": build_dl_filter_gate_scenario_specs(),
            "pair_artifacts": {
                key: project_relative_display_path(value, project_root=root)
                for key, value in pair_dirs.items()
            },
            "score_source": SCORE_SOURCE_CANONICAL_RUNTIME,
            "buy_sort": "original_position_aware_buy_sort",
            "future_target_runtime_used": False,
            "model_retrained": False,
            "threshold_tuned": False,
            "optimizer_executed": False,
        },
        "scenarios": scenarios,
        "comparisons": comparisons,
        "pair_effects": pair_effects,
        "yearly": _yearly_rows(pair_payloads),
    }
    json_path = output_dir / "strategy_dl_filter_gate.json"
    markdown_path = output_dir / "strategy_dl_filter_gate.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    print("\n" + _render_console(result))
    print_artifact_paths(
        (
            ("DL Filter Gate Markdown", markdown_path),
            ("DL Filter Gate JSON", json_path),
            ("A/B Trade Attribution", pair_dirs["AB"] / "trade_attribution.md"),
            ("C/F Trade Attribution", pair_dirs["CF"] / "trade_attribution.md"),
        ),
        project_root=root,
    )
    return result


def main(argv=None):
    args = _parse_args(argv)
    run_dl_filter_gate(args=args)
    return 0


__all__ = [
    "main",
    "run_dl_filter_gate",
    "build_dl_filter_gate_scenario_specs",
    "SCENARIO_SPECS",
    "PAIR_SPECS",
]

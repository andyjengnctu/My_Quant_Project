"""CLI-only rule-ablation matrix for the binary breakout-quality DL filter."""

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
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_model_architecture_dir,
    resolve_filter_model_output_dir,
)
from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_CANONICAL_RUNTIME
from filters.breakout_quality.strategy_compare_engine import (
    COMPARISON_MODE_HARD_FILTER,
    OPTIONAL_ENTRY_FILTER_FIELDS,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    PARAM_POLICIES,
    PARAM_POLICY_BASE_FINALIST_BEST,
    run_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 2

RULE_LEVEL_SPECS: dict[str, dict[str, Any]] = {
    "0": {
        "label": "原正式規則",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
        "shared_param_overrides": {},
    },
    "1": {
        "label": "Optional entry filters 全關",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "shared_param_overrides": {},
    },
    "2": {
        "label": "再關歷史門檻",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "shared_param_overrides": {"use_history_threshold": False},
    },
    "3": {
        "label": "再關 Re-entry",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "shared_param_overrides": {
            "use_history_threshold": False,
            "use_breakout_reclaim_reentry": False,
        },
    },
    "4": {
        "label": "再關 KC 出場",
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "shared_param_overrides": {
            "use_history_threshold": False,
            "use_breakout_reclaim_reentry": False,
            "use_kc": False,
        },
    },
}

SCENARIO_SPECS: dict[str, dict[str, Any]] = {}
for _level_id, _level_spec in RULE_LEVEL_SPECS.items():
    for _side, _enabled in (("A", False), ("B", True)):
        _scenario_id = f"{_side}{_level_id}"
        SCENARIO_SPECS[_scenario_id] = {
            "level": _level_id,
            "label": f"{_level_spec['label']}＋DL {'開' if _enabled else '關'}",
            "optional_entry_filter_policy": _level_spec[
                "optional_entry_filter_policy"
            ],
            "shared_param_overrides": dict(
                _level_spec["shared_param_overrides"]
            ),
            "dl_filter_enabled": _enabled,
        }

PAIR_SPECS = tuple(
    (level_id, f"A{level_id}", f"B{level_id}")
    for level_id in RULE_LEVEL_SPECS
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
    return {
        key: {
            **value,
            "shared_param_overrides": dict(value["shared_param_overrides"]),
        }
        for key, value in SCENARIO_SPECS.items()
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "固定正式rolling params執行A0/B0至A4/B4 Binary DL rule-ablation gate；"
            "不重訓模型、不調threshold、不改原buy-sort。"
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("reduced", "full"),
        default=BREAKOUT_QUALITY_STRATEGY_DATASET,
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument(
        "--model-architecture", default=BREAKOUT_QUALITY_MODEL_ARCHITECTURE
    )
    parser.add_argument(
        "--experiment-profile", default=BREAKOUT_QUALITY_EXPERIMENT_PROFILE
    )
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


def _discover_matching_binary_manifests(
    *,
    project_root: Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    canonical_manifest_path: Path,
) -> list[Path]:
    architecture_dir = resolve_filter_model_architecture_dir(
        project_root,
        filter_id,
        model_architecture,
    )
    if not architecture_dir.exists():
        return []
    candidates = [architecture_dir / "manifest.json"]
    candidates.extend(sorted(architecture_dir.glob("*/manifest.json")))
    matched: list[Path] = []
    for candidate in candidates:
        if (
            not candidate.is_file()
            or candidate.resolve() == canonical_manifest_path.resolve()
        ):
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("filter_id") or "").strip() != str(filter_id).strip():
            continue
        if str(payload.get("model_architecture") or "").strip() != str(
            model_architecture
        ).strip():
            continue
        if str(payload.get("experiment_profile") or "").strip() != str(
            experiment_profile
        ).strip():
            continue
        matched.append(candidate)
    return matched


def _validate_binary_runtime_preflight(*, project_root: Path, args) -> dict[str, Any]:
    paths = resolve_filter_artifact_paths(
        project_root,
        args.filter_id,
        args.model_architecture,
        args.experiment_profile,
    )
    artifacts = {
        "manifest": paths.manifest_path,
        "model": paths.model_path,
        "split": paths.split_path,
        "runtime_score": paths.score_path,
    }
    present = {name: path.is_file() for name, path in artifacts.items()}
    missing_model_artifacts = [
        name for name in ("manifest", "model", "split") if not present[name]
    ]
    if missing_model_artifacts:
        matching_manifests = _discover_matching_binary_manifests(
            project_root=project_root,
            filter_id=args.filter_id,
            model_architecture=args.model_architecture,
            experiment_profile=args.experiment_profile,
            canonical_manifest_path=paths.manifest_path,
        )
        status_lines = [
            f"[{('存在' if present[name] else '缺少')}] "
            f"{name}: {project_relative_display_path(path, project_root=project_root)}"
            for name, path in artifacts.items()
        ]
        recovery_note = ""
        if matching_manifests:
            recovery_note = (
                "\n偵測到其他位置的相同identity manifest：\n- "
                + "\n- ".join(
                    project_relative_display_path(path, project_root=project_root)
                    for path in matching_manifests
                )
                + "\n請從原始備份恢復完整profile資料夾（manifest.json、model.pt、split.csv）；"
                "程式不會只搬單檔或猜測缺失identity。"
            )
        raise FileNotFoundError(
            "Binary DL Filter Gate缺少9A canonical模型工件：\n"
            + "\n".join(status_lines)
            + recovery_note
            + "\n歷史報表、research_scores.csv或策略結果不能替代checkpoint manifest。"
            + "\n若沒有可恢復備份，現有Dataset／Label工件可沿用，通常不必重新build-dataset；"
            + "請先執行：\n"
            + f"python apps/breakout_quality.py train --filter-id {args.filter_id} "
            + f"--experiment-profile {args.experiment_profile}\n"
            + "訓練完成後再執行：\n"
            + f"python apps/breakout_quality.py export-scores --filter-id {args.filter_id} "
            + f"--experiment-profile {args.experiment_profile} --scope forward_oos"
        )
    if not present["runtime_score"]:
        raise FileNotFoundError(
            "9A canonical模型工件完整，但缺少正式forward-OOS scores.csv： "
            f"{project_relative_display_path(paths.score_path, project_root=project_root)}。"
            "不需重訓；請執行：\n"
            f"python apps/breakout_quality.py export-scores --filter-id {args.filter_id} "
            f"--experiment-profile {args.experiment_profile} --scope forward_oos"
        )
    return {"paths": paths, "present": present}


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


def _fmt(
    value: Any,
    *,
    digits: int = 2,
    unit: str = "",
    signed: bool = False,
) -> str:
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
        raise RuntimeError(
            f"讀取{label}失敗: {path}｜{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}根節點必須是object: {path}")
    return payload


def _run_pair(
    *,
    project_root: Path,
    level_id: str,
    output_dir: Path,
    args,
) -> dict[str, Any]:
    spec = RULE_LEVEL_SPECS[level_id]
    pair_label = f"A{level_id}/B{level_id}"
    print(f"[{pair_label}] 執行 {spec['label']} × Binary DL hard filter")
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
                optional_entry_filter_policy=spec[
                    "optional_entry_filter_policy"
                ],
                shared_param_overrides=spec["shared_param_overrides"],
                filter_id=args.filter_id,
                score_source=SCORE_SOURCE_CANONICAL_RUNTIME,
                model_architecture=args.model_architecture,
                experiment_profile=args.experiment_profile,
                output_dir_override=output_dir,
                comparison_start_date=args.start_date,
                comparison_end_date=args.end_date,
                quiet=args.quiet,
            )
    except Exception as exc:
        captured = buffer.getvalue().strip()
        if captured:
            print(captured)
        _ = exc
        raise
    attribution = _read_json_object(
        output_dir / "trade_attribution.json",
        label=f"{pair_label} trade attribution",
    )
    print(f"[{pair_label}] 完成")
    return {"comparison": comparison, "attribution": attribution}


def _scenario_from_pair(payload: dict[str, Any], *, active: bool) -> dict[str, Any]:
    comparison = dict(payload.get("comparison") or {})
    scenario_key = "quality_filter" if active else "no_filter"
    return dict(comparison.get(scenario_key) or {})


def _assert_pair_matrix_compatibility(
    pair_payloads: dict[str, dict[str, Any]],
) -> None:
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
    first_level = next(iter(RULE_LEVEL_SPECS))
    reference = dict(
        (pair_payloads[first_level].get("comparison") or {}).get("metadata") or {}
    )
    for level_id, level_spec in RULE_LEVEL_SPECS.items():
        metadata = dict(
            (pair_payloads[level_id].get("comparison") or {}).get("metadata") or {}
        )
        mismatched = [
            field for field in shared_fields if metadata.get(field) != reference.get(field)
        ]
        if mismatched:
            raise ValueError(
                f"A{level_id}/B{level_id}與共同Gate契約不一致: {mismatched}"
            )
        expected_policy = level_spec["optional_entry_filter_policy"]
        if metadata.get("optional_entry_filter_policy") != expected_policy:
            raise ValueError(
                f"A{level_id}/B{level_id} optional filter policy不一致"
            )
        expected_overrides = dict(level_spec["shared_param_overrides"])
        if dict(metadata.get("shared_param_overrides") or {}) != expected_overrides:
            raise ValueError(
                f"A{level_id}/B{level_id} shared override不一致: "
                f"{metadata.get('shared_param_overrides')}"
            )
        forced = metadata.get("optional_entry_filter_forced_values")
        if expected_policy == OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF:
            expected_forced = {field: False for field in OPTIONAL_ENTRY_FILTER_FIELDS}
            if dict(forced or {}) != expected_forced:
                raise ValueError(
                    f"A{level_id}/B{level_id} optional filter override不完整: {forced}"
                )
        elif forced is not None:
            raise ValueError(f"A{level_id}/B{level_id}不應強制關閉optional filters")


def _state_text(level_id: str, field_name: str) -> str:
    spec = RULE_LEVEL_SPECS[level_id]
    if field_name == "optional":
        return (
            "目前"
            if spec["optional_entry_filter_policy"]
            == OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
            else "全關"
        )
    override_key = {
        "history": "use_history_threshold",
        "reentry": "use_breakout_reclaim_reentry",
        "kc": "use_kc",
    }[field_name]
    return "關" if spec["shared_param_overrides"].get(override_key) is False else "依原參數"


def _main_performance_rows(
    scenarios: dict[str, dict[str, Any]],
    pair_deltas: dict[str, dict[str, float]],
) -> list[tuple[str, ...]]:
    rows = []
    for level_id in RULE_LEVEL_SPECS:
        left = scenarios[f"A{level_id}"]
        right = scenarios[f"B{level_id}"]
        delta = pair_deltas[f"B{level_id}_minus_A{level_id}"]
        rows.append(
            (
                level_id,
                _fmt(left.get("total_return_pct"), unit="%"),
                _fmt(right.get("total_return_pct"), unit="%"),
                _fmt(delta.get("total_return_pct"), unit="pp", signed=True),
                _fmt(left.get("max_drawdown_pct"), unit="%"),
                _fmt(right.get("max_drawdown_pct"), unit="%"),
                _fmt(delta.get("max_drawdown_pct"), unit="pp", signed=True),
                _fmt(left.get("return_over_max_drawdown")),
                _fmt(right.get("return_over_max_drawdown")),
                _fmt(delta.get("return_over_max_drawdown"), signed=True),
            )
        )
    return rows


def _trading_capital_rows(
    scenarios: dict[str, dict[str, Any]],
    pair_deltas: dict[str, dict[str, float]],
) -> list[tuple[str, ...]]:
    rows = []
    for level_id in RULE_LEVEL_SPECS:
        left = scenarios[f"A{level_id}"]
        right = scenarios[f"B{level_id}"]
        delta = pair_deltas[f"B{level_id}_minus_A{level_id}"]
        rows.append(
            (
                level_id,
                _fmt(left.get("expected_value_r"), unit=" R"),
                _fmt(right.get("expected_value_r"), unit=" R"),
                _fmt(delta.get("expected_value_r"), unit=" R", signed=True),
                _fmt(left.get("avg_exposure_pct"), unit="%"),
                _fmt(right.get("avg_exposure_pct"), unit="%"),
                _fmt(delta.get("avg_exposure_pct"), unit="pp", signed=True),
                _fmt(left.get("trade_count"), digits=0),
                _fmt(right.get("trade_count"), digits=0),
                _fmt(delta.get("trade_count"), digits=0, signed=True),
            )
        )
    return rows


def _rule_ablation_rows(
    scenarios: dict[str, dict[str, Any]],
) -> list[tuple[str, ...]]:
    baseline = scenarios["A0"]
    rows = []
    for level_id in ("1", "2", "3", "4"):
        delta = _numeric_delta(scenarios[f"A{level_id}"], baseline)
        rows.append(
            (
                f"A{level_id}−A0",
                RULE_LEVEL_SPECS[level_id]["label"],
                _fmt(delta.get("total_return_pct"), unit="pp", signed=True),
                _fmt(delta.get("max_drawdown_pct"), unit="pp", signed=True),
                _fmt(delta.get("return_over_max_drawdown"), signed=True),
                _fmt(delta.get("expected_value_r"), unit=" R", signed=True),
                _fmt(delta.get("avg_exposure_pct"), unit="pp", signed=True),
                _fmt(delta.get("trade_count"), digits=0, signed=True),
            )
        )
    return rows


def _attribution_rows(
    pair_effects: dict[str, dict[str, Any]],
) -> list[tuple[str, ...]]:
    rows = []
    for level_id in RULE_LEVEL_SPECS:
        effect = pair_effects[f"B{level_id}_minus_A{level_id}"]
        rows.append(
            (
                f"B{level_id}−A{level_id}",
                *(
                    _fmt(
                        effect.get(key),
                        digits=0 if key.endswith("_count") else 2,
                        unit=unit,
                    )
                    for _label, key, unit in _ATTRIBUTION_SPECS
                ),
            )
        )
    return rows


def _yearly_rows(pair_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_level: dict[str, dict[int, dict[str, Any]]] = {}
    years: set[int] = set()
    for level_id in RULE_LEVEL_SPECS:
        rows = list(
            (pair_payloads[level_id].get("comparison") or {}).get("yearly") or []
        )
        mapping = {int(row["year"]): dict(row) for row in rows}
        by_level[level_id] = mapping
        years.update(mapping)
    output = []
    for year in sorted(years):
        row: dict[str, Any] = {"year": year, "is_full_year": True}
        for level_id in RULE_LEVEL_SPECS:
            source = by_level[level_id].get(year, {})
            left_value = source.get("no_filter_return_pct")
            right_value = source.get("quality_filter_return_pct")
            row[f"A{level_id}_return_pct"] = left_value
            row[f"B{level_id}_return_pct"] = right_value
            row[f"B{level_id}_minus_A{level_id}_pct"] = (
                float(right_value) - float(left_value)
                if left_value is not None and right_value is not None
                else None
            )
            row["is_full_year"] = bool(
                row["is_full_year"] and source.get("is_full_year", False)
            )
        output.append(row)
    return output


def _render_console(result: dict[str, Any]) -> str:
    metadata = result["metadata"]
    scenarios = result["scenarios"]
    pair_deltas = result["pair_deltas"]
    lines = [
        render_title("Binary DL Filter Rule Ablation A0／B0 ～ A4／B4 Gate"),
        render_key_values(
            (
                (
                    "期間",
                    f"{metadata['comparison_period']['start']} ～ "
                    f"{metadata['comparison_period']['end']}",
                ),
                ("參數", metadata["param_policy"]),
                (
                    "Binary model",
                    f"{metadata['model_architecture']} / "
                    f"{metadata['experiment_profile']}",
                ),
                ("Threshold", _fmt(metadata["threshold"], digits=2)),
                ("Score source", SCORE_SOURCE_CANONICAL_RUNTIME),
                ("Buy sort", "原 position-aware buy-sort（十組相同）"),
            )
        ),
        render_section("情境矩陣"),
        render_table(
            [
                "層級",
                "A組",
                "B組",
                "Optional filters",
                "歷史門檻",
                "Re-entry",
                "KC出場",
            ],
            [
                (
                    level_id,
                    f"A{level_id}：DL關",
                    f"B{level_id}：DL開",
                    _state_text(level_id, "optional"),
                    _state_text(level_id, "history"),
                    _state_text(level_id, "reentry"),
                    _state_text(level_id, "kc"),
                )
                for level_id in RULE_LEVEL_SPECS
            ],
        ),
        render_section("每層投組成效與DL增量", number=1),
        render_table(
            [
                "層級",
                "A報酬",
                "B報酬",
                "Δ報酬",
                "A MDD",
                "B MDD",
                "ΔMDD",
                "A RoMD",
                "B RoMD",
                "ΔRoMD",
            ],
            _main_performance_rows(scenarios, pair_deltas),
        ),
        render_section("每層交易品質與資金使用", number=2),
        render_table(
            [
                "層級",
                "A EV",
                "B EV",
                "ΔEV",
                "A曝險",
                "B曝險",
                "Δ曝險",
                "A交易",
                "B交易",
                "Δ交易",
            ],
            _trading_capital_rows(scenarios, pair_deltas),
        ),
        render_section("逐步移除規則的影響（只看DL關閉的A系列）", number=3),
        render_table(
            ["比較", "新增關閉", "報酬", "MDD", "RoMD", "EV", "曝險", "交易數"],
            _rule_ablation_rows(scenarios),
        ),
        render_section("DL交易歸因（每層B−A）", number=4),
        render_table(
            ["比較", *(_label for _label, _key, _unit in _ATTRIBUTION_SPECS)],
            _attribution_rows(result["pair_effects"]),
        ),
        render_section("年度DL增量（B−A）", number=5),
        render_table(
            ["年度", "B0−A0", "B1−A1", "B2−A2", "B3−A3", "B4−A4", "完整年度"],
            [
                (
                    str(row["year"]),
                    *(
                        _fmt(
                            row.get(f"B{level_id}_minus_A{level_id}_pct"),
                            unit="pp",
                            signed=True,
                        )
                        for level_id in RULE_LEVEL_SPECS
                    ),
                    "是" if row.get("is_full_year") else "否",
                )
                for row in result["yearly"]
            ],
        ),
        render_section("判讀契約", number=6),
        "- 每一層只以同層 Bn−An 判斷Binary DL增量，避免把規則消融與DL效果混在一起。",
        "- A1−A0至A4−A0只判斷逐步關閉規則本身的影響；不代表DL有效。",
        "- A4／B4只關KC出場，不關半倉停利、ATR初始停損、trailing、fixed-risk sizing或原buy-sort。",
        "- 十組不使用continuous Score、R3、Future Target或optimizer；threshold固定為OOS前0.5。",
    ]
    return "\n".join(lines)


def _render_markdown(result: dict[str, Any]) -> str:
    metadata = result["metadata"]
    scenarios = result["scenarios"]
    pair_deltas = result["pair_deltas"]
    lines = [
        "# Binary DL Filter Rule Ablation A0／B0 ～ A4／B4 Gate",
        "",
        f"- 期間：`{metadata['comparison_period']['start']}` ～ "
        f"`{metadata['comparison_period']['end']}`",
        f"- 參數 policy：`{metadata['param_policy']}`",
        f"- Binary model：`{metadata['model_architecture']} / "
        f"{metadata['experiment_profile']}`",
        f"- Threshold：`{metadata['threshold']}`",
        "- Score source：`canonical_runtime`",
        "- Ranking：十組均使用原 position-aware buy-sort。",
        "- Future Target：未進入本Gate。",
        "",
        "## 情境矩陣",
        "",
        "| 層級 | A組 | B組 | Optional filters | 歷史門檻 | Re-entry | KC出場 |",
        "|---:|---|---|---|---|---|---|",
    ]
    for level_id in RULE_LEVEL_SPECS:
        lines.append(
            "| "
            + " | ".join(
                (
                    level_id,
                    f"A{level_id}：DL關",
                    f"B{level_id}：DL開",
                    _state_text(level_id, "optional"),
                    _state_text(level_id, "history"),
                    _state_text(level_id, "reentry"),
                    _state_text(level_id, "kc"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 每層投組成效與DL增量",
            "",
            "| 層級 | A報酬 | B報酬 | Δ報酬 | A MDD | B MDD | ΔMDD | A RoMD | B RoMD | ΔRoMD |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _main_performance_rows(scenarios, pair_deltas):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## 每層交易品質與資金使用",
            "",
            "| 層級 | A EV | B EV | ΔEV | A曝險 | B曝險 | Δ曝險 | A交易 | B交易 | Δ交易 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _trading_capital_rows(scenarios, pair_deltas):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## 逐步移除規則的影響（只看DL關閉的A系列）",
            "",
            "| 比較 | 新增關閉 | 報酬 | MDD | RoMD | EV | 曝險 | 交易數 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _rule_ablation_rows(scenarios):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## DL交易歸因（每層B−A）",
            "",
            "| 比較 | "
            + " | ".join(label for label, _key, _unit in _ATTRIBUTION_SPECS)
            + " |",
            "|---|" + "---:|" * len(_ATTRIBUTION_SPECS),
        ]
    )
    for row in _attribution_rows(result["pair_effects"]):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## 年度DL增量（B−A）",
            "",
            "| 年度 | B0−A0 | B1−A1 | B2−A2 | B3−A3 | B4−A4 | 完整年度 |",
            "|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in result["yearly"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["year"]),
                    *(
                        _fmt(
                            row.get(f"B{level_id}_minus_A{level_id}_pct"),
                            unit="pp",
                            signed=True,
                        )
                        for level_id in RULE_LEVEL_SPECS
                    ),
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
            "- 每一層只以同層 `Bn−An` 判斷Binary DL增量，避免把規則消融與DL效果混在一起。",
            "- `A1−A0`至`A4−A0`只判斷逐步關閉規則本身的影響；不代表DL有效。",
            "- `A4／B4`只關KC出場，不關半倉停利、ATR初始停損、trailing、fixed-risk sizing或原buy-sort。",
            "- 十組不使用continuous Score、R3、Future Target或optimizer；threshold固定為OOS前0.5。",
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
    _validate_binary_runtime_preflight(project_root=root, args=args)
    output_dir = (
        resolve_filter_model_output_dir(
            str(root),
            args.filter_id,
            args.model_architecture,
            args.experiment_profile,
        )
        / (
            "strategy_dl_filter_rule_ablation_gate_"
            f"{str(args.param_policy).replace('-', '_')}_canonical_runtime"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_payloads: dict[str, dict[str, Any]] = {}
    pair_dirs: dict[str, Path] = {}
    for level_id, left_id, right_id in PAIR_SPECS:
        pair_dir = output_dir / f"pair_{left_id.lower()}_{right_id.lower()}"
        pair_dirs[level_id] = pair_dir
        pair_payloads[level_id] = _run_pair(
            project_root=root,
            level_id=level_id,
            output_dir=pair_dir,
            args=args,
        )

    _assert_pair_matrix_compatibility(pair_payloads)
    scenarios: dict[str, dict[str, Any]] = {}
    pair_deltas: dict[str, dict[str, float]] = {}
    pair_effects: dict[str, dict[str, Any]] = {}
    for level_id, left_id, right_id in PAIR_SPECS:
        scenarios[left_id] = _scenario_from_pair(
            pair_payloads[level_id], active=False
        )
        scenarios[right_id] = _scenario_from_pair(
            pair_payloads[level_id], active=True
        )
        pair_key = f"{right_id}_minus_{left_id}"
        pair_deltas[pair_key] = _numeric_delta(
            scenarios[right_id], scenarios[left_id]
        )
        pair_effects[pair_key] = dict(
            (pair_payloads[level_id].get("attribution") or {}).get(
                "r_attribution"
            )
            or {}
        )

    first_meta = dict(
        (pair_payloads["0"].get("comparison") or {}).get("metadata") or {}
    )
    rule_ablation = {
        f"A{level_id}_minus_A0": _numeric_delta(
            scenarios[f"A{level_id}"], scenarios["A0"]
        )
        for level_id in ("1", "2", "3", "4")
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "comparison_period": dict(first_meta.get("comparison_period") or {}),
            "dataset": args.dataset,
            "filter_id": args.filter_id,
            "model_architecture": args.model_architecture,
            "experiment_profile": args.experiment_profile,
            "threshold": float(
                first_meta.get(
                    "threshold", BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
                )
            ),
            "param_policy": args.param_policy,
            "params_path": first_meta.get("params_path"),
            "params_file_sha256": first_meta.get("params_file_sha256"),
            "optional_entry_filter_fields": list(OPTIONAL_ENTRY_FILTER_FIELDS),
            "rule_level_specs": {
                key: {
                    **value,
                    "shared_param_overrides": dict(
                        value["shared_param_overrides"]
                    ),
                }
                for key, value in RULE_LEVEL_SPECS.items()
            },
            "scenario_specs": build_dl_filter_gate_scenario_specs(),
            "pair_artifacts": {
                f"A{key}/B{key}": project_relative_display_path(
                    value, project_root=root
                )
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
        "pair_deltas": pair_deltas,
        "rule_ablation": rule_ablation,
        "pair_effects": pair_effects,
        "yearly": _yearly_rows(pair_payloads),
    }
    json_path = output_dir / "strategy_dl_filter_rule_ablation_gate.json"
    markdown_path = output_dir / "strategy_dl_filter_rule_ablation_gate.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    print("\n" + _render_console(result))
    artifact_rows: list[tuple[str, Path]] = [
        ("DL Filter Rule Ablation Markdown", markdown_path),
        ("DL Filter Rule Ablation JSON", json_path),
    ]
    artifact_rows.extend(
        (
            f"A{level_id}/B{level_id} Trade Attribution",
            pair_dirs[level_id] / "trade_attribution.md",
        )
        for level_id in RULE_LEVEL_SPECS
    )
    print_artifact_paths(artifact_rows, project_root=root)
    return result


def main(argv=None):
    args = _parse_args(argv)
    run_dl_filter_gate(args=args)
    return 0


__all__ = [
    "main",
    "run_dl_filter_gate",
    "build_dl_filter_gate_scenario_specs",
    "RULE_LEVEL_SPECS",
    "SCENARIO_SPECS",
    "PAIR_SPECS",
    "_validate_binary_runtime_preflight",
]

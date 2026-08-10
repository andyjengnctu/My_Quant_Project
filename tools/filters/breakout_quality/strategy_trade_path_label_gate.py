"""Compare A2 no-DL with old and realized trade-path Binary DL labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


from config.breakout_quality import get_breakout_quality_workflow_settings
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.contract import DEFAULT_MODEL_ARCHITECTURE
from filters.breakout_quality.paths import (
    resolve_filter_report_json_path,
    resolve_filter_artifact_paths,
)
from filters.breakout_quality.trade_path_label import (
    TRADE_PATH_BASE_FILTER_ID,
    TRADE_PATH_FORWARD_TEACHER_PARAMS_RELATIVE_PATH,
    TRADE_PATH_LABEL_ID,
    TRADE_PATH_RESEARCH_FILTER_ID,
)
from filters.breakout_quality.workflow_io import PROJECT_ROOT, write_json
from filters.breakout_quality.export_scores import main as export_scores_main
from filters.breakout_quality.strategy_compare_engine import (
    COMPARISON_MODE_HARD_FILTER,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    PARAM_POLICY_BASE_FINALIST_BEST,
    run_comparison,
)
from filters.breakout_quality.strategy_param_training import (
    ALL_RULE_FILTERS_OFF_OVERRIDES,
)

SCHEMA_VERSION = 2
DIRECT_SELECTION_R_KEY = "exclusive_selection_delta_r"
BASE_IDENTITY_FILENAMES = (
    "no_filter_equity.csv",
    "no_filter_trades.csv",
    "no_filter_daily_capacity.csv",
)
OUTPUT_RELATIVE_DIR = Path(
    "models/research/breakout_quality/trade_path_label_gate/a2_realized_trade_path_v1"
)


def parse_args(argv=None) -> argparse.Namespace:
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "比較A2 no-DL、原MFE/MAE 9A DL與A2 realized trade-path Label DL；"
            "只使用既有已凍結模型，不自動訓練"
        )
    )
    parser.add_argument("--dataset", choices=("reduced", "full"), default="full")
    parser.add_argument(
        "--param-policy",
        choices=(PARAM_POLICY_BASE_FINALIST_BEST,),
        default=PARAM_POLICY_BASE_FINALIST_BEST,
    )
    parser.add_argument("--max-positions", type=int, default=int(settings.strategy_max_positions))
    parser.add_argument("--rotation", choices=("off", "on"), default=str(settings.strategy_rotation))
    parser.add_argument("--fixed-risk", type=float, default=float(settings.strategy_adapt_fixed_risk))
    parser.add_argument(
        "--max-position-cap-pct",
        type=float,
        default=float(settings.strategy_adapt_max_position_cap_pct),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _require_model_and_report(
    root: Path,
    *,
    filter_id: str,
    architecture: str,
    profile: str,
) -> None:
    artifacts = resolve_filter_artifact_paths(root, filter_id, architecture, profile)
    report_path = resolve_filter_report_json_path(root, filter_id, architecture, profile)
    missing = [
        path
        for path in (artifacts.model_path, artifacts.manifest_path, report_path)
        if not path.is_file()
    ]
    if missing:
        display = ", ".join(
            project_relative_display_path(path, project_root=root) for path in missing
        )
        raise FileNotFoundError(
            f"策略Gate只讀取既有已驗證模型；缺少工件: {display}。"
            "請先由選單「[1]  建立新Label → 重新訓練 → 模型預測報表  (Enter)」完成前置流程。"
        )


def _export_forward_scores(
    *,
    filter_id: str,
    profile: str,
) -> None:
    code = export_scores_main(
        [
            "--filter-id",
            filter_id,
            "--experiment-profile",
            profile,
            "--scope",
            "forward_oos",
        ]
    )
    if int(code or 0) != 0:
        raise RuntimeError(f"forward-OOS score export失敗: filter_id={filter_id}")


def _metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trade_attribution(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "trade_attribution.json"
    if not path.is_file():
        raise FileNotFoundError(
            "策略Gate缺少交易歸因工件: "
            f"{project_relative_display_path(path, project_root=PROJECT_ROOT)}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    r_attribution = payload.get("r_attribution")
    if not isinstance(r_attribution, dict):
        raise ValueError(f"交易歸因工件缺少r_attribution: {path}")
    direct_selection_r = _metric(r_attribution, DIRECT_SELECTION_R_KEY)
    if direct_selection_r is None:
        raise ValueError(
            f"交易歸因工件缺少有限{DIRECT_SELECTION_R_KEY}: {path}"
        )
    return payload


def _direct_selection_r(payload: dict[str, Any]) -> float:
    value = _metric(dict(payload.get("r_attribution") or {}), DIRECT_SELECTION_R_KEY)
    if value is None:
        raise ValueError(f"交易歸因缺少有限{DIRECT_SELECTION_R_KEY}")
    return float(value)


def _assert_same_base_artifacts(
    *, old_output_dir: Path, new_output_dir: Path
) -> dict[str, str]:
    identity: dict[str, str] = {}
    mismatches: list[tuple[str, str, str]] = []
    for filename in BASE_IDENTITY_FILENAMES:
        old_path = old_output_dir / filename
        new_path = new_output_dir / filename
        if not old_path.is_file() or not new_path.is_file():
            missing = [str(path) for path in (old_path, new_path) if not path.is_file()]
            raise FileNotFoundError(
                "Old／New Label比較缺少A2 no-DL base工件: " + ", ".join(missing)
            )
        old_sha = _sha256_file(old_path)
        new_sha = _sha256_file(new_path)
        if old_sha != new_sha:
            mismatches.append((filename, old_sha, new_sha))
        identity[filename] = old_sha
    if mismatches:
        raise ValueError(
            "Old／New Label比較的A2 no-DL base工件不一致: "
            f"{mismatches}"
        )
    return identity


def _fmt(value: Any, *, digits: int = 2, unit: str = "") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/A"
    number = float(value)
    if not math.isfinite(number):
        return "N/A"
    if digits == 0:
        return f"{number:.0f}{unit}"
    return f"{number:.{digits}f}{unit}"


def _delta(newer: dict[str, Any], older: dict[str, Any], key: str) -> float | None:
    a = _metric(newer, key)
    b = _metric(older, key)
    return None if a is None or b is None else a - b


def _assert_same_base(old_payload: dict[str, Any], new_payload: dict[str, Any]) -> None:
    old_base = dict(old_payload.get("no_filter") or {})
    new_base = dict(new_payload.get("no_filter") or {})
    fields = (
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "annual_return_pct",
        "expected_value_r",
        "avg_exposure_pct",
        "trade_count",
    )
    mismatches = []
    for field in fields:
        left = _metric(old_base, field)
        right = _metric(new_base, field)
        if left is None or right is None or not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9):
            mismatches.append((field, left, right))
    if mismatches:
        raise ValueError(f"Old／New Label比較的A2 no-DL base不一致: {mismatches}")


def _render_summary_table(
    *,
    base: dict[str, Any],
    old_dl: dict[str, Any],
    new_dl: dict[str, Any],
    old_attribution: dict[str, Any],
    new_attribution: dict[str, Any],
) -> str:
    rows = []
    for label, payload, direct_selection_r in (
        ("A2 Base（DL關）", base, 0.0),
        ("Old Label 9A（DL開）", old_dl, _direct_selection_r(old_attribution)),
        (
            "New Trade-path Label（DL開）",
            new_dl,
            _direct_selection_r(new_attribution),
        ),
    ):
        rows.append(
            (
                label,
                _fmt(payload.get("total_return_pct"), unit="%"),
                _fmt(payload.get("max_drawdown_pct"), unit="%"),
                _fmt(payload.get("return_over_max_drawdown")),
                _fmt(payload.get("annual_return_pct"), unit="%"),
                _fmt(payload.get("expected_value_r"), unit=" R"),
                _fmt(direct_selection_r, unit=" R"),
                _fmt(payload.get("avg_exposure_pct"), unit="%"),
                _fmt(payload.get("trade_count"), digits=0),
            )
        )
    return render_table(
        (
            "組別",
            "報酬",
            "MDD",
            "RoMD",
            "年化",
            "EV",
            "直接選擇R",
            "曝險",
            "交易數",
        ),
        rows,
    )


def _render_delta_table(
    *,
    base: dict[str, Any],
    old_dl: dict[str, Any],
    new_dl: dict[str, Any],
    old_attribution: dict[str, Any],
    new_attribution: dict[str, Any],
) -> str:
    metrics = (
        ("報酬", "total_return_pct", "pp"),
        ("MDD", "max_drawdown_pct", "pp"),
        ("RoMD", "return_over_max_drawdown", ""),
        ("年化", "annual_return_pct", "pp"),
        ("EV", "expected_value_r", " R"),
        ("曝險", "avg_exposure_pct", "pp"),
        ("交易數", "trade_count", ""),
    )
    direct_old = _direct_selection_r(old_attribution)
    direct_new = _direct_selection_r(new_attribution)
    rows = []
    for label, newer, older, direct_delta in (
        ("New−Base", new_dl, base, direct_new),
        ("New−Old", new_dl, old_dl, direct_new - direct_old),
        ("Old−Base", old_dl, base, direct_old),
    ):
        values = []
        for _metric_label, key, unit in metrics:
            values.append(
                _fmt(
                    _delta(newer, older, key),
                    digits=0 if key == "trade_count" else 2,
                    unit=unit,
                )
            )
        rows.append(
            (
                label,
                *values[:5],
                _fmt(direct_delta, unit=" R"),
                *values[5:],
            )
        )
    return render_table(
        (
            "比較",
            "報酬",
            "MDD",
            "RoMD",
            "年化",
            "EV",
            "直接選擇R",
            "曝險",
            "交易數",
        ),
        rows,
    )


def _yearly_table(
    *,
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
) -> str:
    old_rows = {int(row["year"]): row for row in old_payload.get("yearly") or []}
    new_rows = {int(row["year"]): row for row in new_payload.get("yearly") or []}
    rows = []
    for year in sorted(set(old_rows) | set(new_rows)):
        old_row = dict(old_rows.get(year) or {})
        new_row = dict(new_rows.get(year) or {})
        base_value = old_row.get("no_filter_return_pct")
        old_value = old_row.get("quality_filter_return_pct")
        new_value = new_row.get("quality_filter_return_pct")
        rows.append(
            (
                year,
                _fmt(base_value, unit="%"),
                _fmt(old_value, unit="%"),
                _fmt(new_value, unit="%"),
                _fmt(
                    None
                    if not isinstance(new_value, (int, float)) or not isinstance(base_value, (int, float))
                    else float(new_value) - float(base_value),
                    unit="pp",
                ),
                _fmt(
                    None
                    if not isinstance(new_value, (int, float)) or not isinstance(old_value, (int, float))
                    else float(new_value) - float(old_value),
                    unit="pp",
                ),
            )
        )
    return render_table(
        ("年度", "A2 Base", "Old Label", "New Label", "New−Base", "New−Old"),
        rows,
    )


def _render_report(
    *,
    metadata: dict[str, Any],
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
    old_attribution: dict[str, Any],
    new_attribution: dict[str, Any],
) -> str:
    base = dict(old_payload.get("no_filter") or {})
    old_dl = dict(old_payload.get("quality_filter") or {})
    new_dl = dict(new_payload.get("quality_filter") or {})
    lines = [
        render_title("A2 Trade-path Label Strategy Gate"),
        render_key_values(
            (
                ("期間", metadata["comparison_period"]),
                ("A2 Params", metadata["params_path"]),
                ("Rules", "all-off"),
                ("Base", "A2 no-DL"),
                ("Old Label", metadata["old_filter_id"]),
                ("New Label", metadata["new_filter_id"]),
                ("Threshold", metadata["threshold"]),
                ("唯一研究變數", "Binary Label／model identity"),
            )
        ),
        render_section("1. 三組策略結果"),
        _render_summary_table(
            base=base, old_dl=old_dl, new_dl=new_dl,
            old_attribution=old_attribution, new_attribution=new_attribution,
        ),
        render_section("2. 關鍵差異"),
        _render_delta_table(
            base=base, old_dl=old_dl, new_dl=new_dl,
            old_attribution=old_attribution, new_attribution=new_attribution,
        ),
        render_section("3. 年度結果"),
        _yearly_table(old_payload=old_payload, new_payload=new_payload),
        render_section("4. 判讀契約"),
        "主判定看New−Base：新Label至少須同時改善RoMD、EV、直接交易選擇R及年度穩定性，才可進入Binary PIT與DL-on optimizer。New−Old只判斷Label alignment是否優於舊MFE／MAE Label。不得依本次OOS結果回頭調整threshold、Label或模型。",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    if int(args.max_positions) < 1:
        raise ValueError("--max-positions必須>=1")
    root = PROJECT_ROOT.resolve()
    settings = get_breakout_quality_workflow_settings()
    architecture = str(settings.model_architecture or DEFAULT_MODEL_ARCHITECTURE)
    profile = str(settings.experiment_profile)
    params_path = root / TRADE_PATH_FORWARD_TEACHER_PARAMS_RELATIVE_PATH
    if not params_path.is_file():
        raise FileNotFoundError(
            "缺少A2／P2 active params: "
            f"{project_relative_display_path(params_path, project_root=root)}"
        )
    _require_model_and_report(
        root,
        filter_id=TRADE_PATH_RESEARCH_FILTER_ID,
        architecture=architecture,
        profile=profile,
    )
    _require_model_and_report(
        root,
        filter_id=TRADE_PATH_BASE_FILTER_ID,
        architecture=architecture,
        profile=profile,
    )

    print("\n" + render_title("A2 Trade-path Label Strategy Gate"))
    print("使用既有已凍結模型；只更新forward-OOS Scores並執行策略回放，不重新訓練。")
    print("\n[1/3] 更新舊Label forward-OOS Scores")
    _export_forward_scores(filter_id=TRADE_PATH_BASE_FILTER_ID, profile=profile)
    print("\n[2/3] 更新新Label forward-OOS Scores")
    _export_forward_scores(filter_id=TRADE_PATH_RESEARCH_FILTER_ID, profile=profile)

    common = {
        "project_root": root,
        "dataset": str(args.dataset),
        "params_path": str(params_path),
        "param_policy": str(args.param_policy),
        "max_positions": int(args.max_positions),
        "enable_rotation": str(args.rotation) == "on",
        "fixed_risk": float(args.fixed_risk),
        "max_position_cap_pct": float(args.max_position_cap_pct),
        "comparison_mode": COMPARISON_MODE_HARD_FILTER,
        "optional_entry_filter_policy": OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        "model_architecture": architecture,
        "experiment_profile": profile,
        "quiet": bool(args.quiet),
        "shared_param_overrides": ALL_RULE_FILTERS_OFF_OVERRIDES,
    }
    output_dir = root / OUTPUT_RELATIVE_DIR
    print("\n[3/3] 回放A2 Base／Old Label／New Label")
    old_output_dir = output_dir / "old_label_pair"
    new_output_dir = output_dir / "new_label_pair"
    old_payload = run_comparison(
        **common,
        filter_id=TRADE_PATH_BASE_FILTER_ID,
        output_dir_override=old_output_dir,
    )
    new_payload = run_comparison(
        **common,
        filter_id=TRADE_PATH_RESEARCH_FILTER_ID,
        output_dir_override=new_output_dir,
    )
    _assert_same_base(old_payload, new_payload)
    base_artifact_sha256 = _assert_same_base_artifacts(
        old_output_dir=old_output_dir, new_output_dir=new_output_dir
    )
    old_attribution = _load_trade_attribution(old_output_dir)
    new_attribution = _load_trade_attribution(new_output_dir)

    old_meta = dict(old_payload.get("metadata") or {})
    new_meta = dict(new_payload.get("metadata") or {})
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "label_id": TRADE_PATH_LABEL_ID,
        "old_filter_id": TRADE_PATH_BASE_FILTER_ID,
        "new_filter_id": TRADE_PATH_RESEARCH_FILTER_ID,
        "model_architecture": architecture,
        "experiment_profile": profile,
        "threshold": old_meta.get("threshold"),
        "params_path": project_relative_display_path(params_path, project_root=root),
        "comparison_period": old_meta.get("comparison_period"),
        "old_score_source": old_meta.get("score_source"),
        "new_score_source": new_meta.get("score_source"),
        "rule_policy": "all-off",
        "strategy_param_policy": str(args.param_policy),
        "base_identity_contract": "exact_sha256_equity_trades_daily_capacity",
        "base_artifact_sha256": base_artifact_sha256,
        "direct_selection_r_contract": DIRECT_SELECTION_R_KEY,
    }
    report = _render_report(
        metadata=metadata,
        old_payload=old_payload,
        new_payload=new_payload,
        old_attribution=old_attribution,
        new_attribution=new_attribution,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "strategy_trade_path_label_gate.md"
    json_path = output_dir / "strategy_trade_path_label_gate.json"
    report_path.write_text(report, encoding="utf-8")
    write_json(
        json_path,
        {
            "metadata": metadata,
            "base": old_payload.get("no_filter"),
            "old_label": old_payload.get("quality_filter"),
            "new_label": new_payload.get("quality_filter"),
            "old_pair": old_payload,
            "new_pair": new_payload,
            "old_trade_attribution": old_attribution,
            "new_trade_attribution": new_attribution,
        },
    )
    print("\n" + report)
    print_artifact_paths(
        (
            ("Gate Markdown", report_path),
            ("Gate JSON", json_path),
            ("Old Label pair", old_output_dir / "strategy_comparison.md"),
            ("Old Label歸因", old_output_dir / "trade_attribution.md"),
            ("New Label pair", new_output_dir / "strategy_comparison.md"),
            ("New Label歸因", new_output_dir / "trade_attribution.md"),
        ),
        project_root=root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

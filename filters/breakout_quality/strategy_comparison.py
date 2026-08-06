"""Config-driven Breakout Quality strategy performance comparison orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

from config.strategy_compare import get_strategy_comparison_settings
from core.runtime_utils import get_taipei_now
from core.strategy_comparison import (
    StrategyComparisonArm,
    StrategyComparisonSettings,
    StrategyDLSource,
    StrategyPreparationPlan,
    strategy_comparison_fingerprint,
)
from filters.breakout_quality.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.strategy_compare_engine import (
    COMPARISON_MODE_HARD_FILTER,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    run_comparison,
)
from filters.breakout_quality.strategy_rule_policies import (
    ALL_RULE_FILTERS_OFF_OVERRIDES,
)
from filters.breakout_quality.strategy_compare_preparation import (
    collect_artifact_status as collect_preparation_status,
    prepare_strategy_comparison_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA_VERSION = 4


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_native(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _resolve_relative_path(root: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"設定路徑必須是專案root相對路徑: {value}")
    return (root / path).resolve()


def collect_artifact_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
) -> dict[str, Any]:
    current = settings or get_strategy_comparison_settings()
    status = collect_preparation_status(
        project_root=Path(project_root).resolve(),
        settings=current,
    )
    status["config_fingerprint"] = strategy_comparison_fingerprint(
        current,
        artifact_identities=status["artifact_identities"],
    )
    return status


def render_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
    status: dict[str, Any] | None = None,
) -> str:
    root = Path(project_root).resolve()
    current = settings or get_strategy_comparison_settings()
    current_status = status or collect_artifact_status(
        project_root=root,
        settings=current,
    )
    arm_rows = [
        (
            "ON" if arm.enabled else "OFF",
            arm.arm_id,
            arm.name,
            arm.param_source,
            arm.rule_policy,
            "DL-on" if arm.dl_enabled else "DL-off",
            arm.description,
        )
        for arm in current.arms.values()
    ]
    contrast_rows = [
        (
            "ON" if item.enabled else "OFF",
            item.contrast_id,
            item.left,
            item.right,
            item.description,
        )
        for item in current.contrasts.values()
    ]
    artifact_rows = []
    for source_id, row in current_status["parameters"].items():
        artifact_rows.append((f"param:{source_id}", row["status"], row["action"], row["path"]))
        if row.get("identity_manifest_path"):
            artifact_rows.append(
                (
                    f"param:{source_id}:identity",
                    row["identity_status"],
                    row["action"],
                    row["identity_manifest_path"],
                )
            )
    for dl_id, row in current_status["dl_sources"].items():
        for key, file_row in row["files"].items():
            artifact_rows.append(
                (f"dl:{dl_id}:{key}", file_row["status"], file_row["action"], file_row["path"])
            )
    return "\n\n".join(
        (
            render_title("策略績效比較設定與工件狀態"),
            render_key_values(
                (
                    ("設定檔", "config/strategy_compare.py"),
                    ("Dataset", current.dataset),
                    (
                        "期間",
                        (
                            f"{current_status['comparison_period']['start']} ～ "
                            f"{current_status['comparison_period']['end']}"
                            if current_status.get("comparison_period")
                            else f"{current.start_date or 'artifact start'} ～ {current.end_date or 'artifact end'}"
                        ),
                    ),
                    ("Param policy", current.param_policy),
                    ("Max positions", current.max_positions),
                    ("Rotation", current.rotation),
                    ("Config fingerprint", current_status["config_fingerprint"]),
                    ("比較狀態", current_status["overall_status"]),
                    ("自動前置", "on" if current.preparation.auto_prepare else "off"),
                )
            ),
            render_section("1. 比較對象"),
            render_table(
                ("開關", "編號", "名稱", "參數來源", "Rules", "DL", "用途"),
                arm_rows,
            ),
            render_section("2. 差異比較"),
            render_table(
                ("開關", "比較", "左側", "右側", "用途"),
                contrast_rows,
            ),
            render_section("3. 前置工件與預計動作"),
            render_table(("工件", "狀態", "預計動作", "路徑"), artifact_rows),
        )
    )


def render_execution_plan(
    *,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
) -> str:
    plan: StrategyPreparationPlan = status["preparation_plan"]
    rows = [
        (item.action, item.artifact_key, item.description)
        for item in plan.actions
    ]
    rows.extend(
        ("RUN", arm.arm_id, arm.name) for arm in settings.enabled_arms
    )
    rows.extend(
        ("REPORT", item.contrast_id, item.description)
        for item in settings.enabled_contrasts
    )
    return "\n\n".join(
        (
            render_title("本次執行計畫"),
            render_key_values(
                (
                    ("整體狀態", plan.overall_status),
                    ("設定檔", "config/strategy_compare.py"),
                    ("Config fingerprint", status["config_fingerprint"]),
                )
            ),
            render_table(("動作", "項目", "說明"), rows),
        )
    )


def _metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: Any, *, unit: str = "", digits: int = 2) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    number = float(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}{unit}"


def _load_direct_selection_r(output_dir: Path, *, root: Path) -> float:
    path = output_dir / "trade_attribution.json"
    payload = _read_json(path)
    if payload is None:
        raise FileNotFoundError(
            "缺少交易歸因工件: "
            + project_relative_display_path(path, project_root=root)
        )
    value = _metric(
        dict(payload.get("r_attribution") or {}),
        "exclusive_selection_delta_r",
    )
    if value is None:
        raise ValueError("交易歸因缺少exclusive_selection_delta_r")
    return float(value)


def _execution_pairs(
    settings: StrategyComparisonSettings,
) -> tuple[tuple[str, str, StrategyComparisonArm, StrategyComparisonArm], ...]:
    """Return one replay pair per enabled DL source in config order.

    A ``param_source`` / ``rule_policy`` group owns one shared DL-off baseline
    and may expose multiple DL-on arms.  Each DL-on arm is replayed against the
    same baseline; downstream aggregation verifies that repeated baseline
    summaries and yearly returns remain identical.
    """
    grouped: dict[
        tuple[str, str],
        dict[str, StrategyComparisonArm | list[StrategyComparisonArm] | None],
    ] = {}
    ordered_keys: list[tuple[str, str]] = []
    for arm in settings.enabled_arms:
        key = (arm.param_source, arm.rule_policy)
        if key not in grouped:
            grouped[key] = {"off": None, "on": []}
            ordered_keys.append(key)
        group = grouped[key]
        if not arm.dl_enabled:
            if group["off"] is not None:
                raise ValueError(
                    "啟用比較群組重複定義DL-off基準: "
                    f"{arm.param_source}/{arm.rule_policy}"
                )
            group["off"] = arm
            continue
        on_arms = group["on"]
        if not isinstance(on_arms, list):
            raise TypeError("strategy comparison execution group contract錯誤")
        if not arm.dl_id:
            raise ValueError(f"DL-on arm缺少dl_id: {arm.arm_id}")
        if any(existing.dl_id == arm.dl_id for existing in on_arms):
            raise ValueError(
                "啟用比較群組重複定義相同DL source: "
                f"{arm.param_source}/{arm.rule_policy}/{arm.dl_id}"
            )
        on_arms.append(arm)

    pairs: list[
        tuple[str, str, StrategyComparisonArm, StrategyComparisonArm]
    ] = []
    for param_source, rule_policy in ordered_keys:
        group = grouped[(param_source, rule_policy)]
        off_arm = group["off"]
        on_arms = group["on"]
        if not isinstance(off_arm, StrategyComparisonArm) or not isinstance(on_arms, list):
            raise ValueError(
                "啟用比較群組缺少共用DL-off基準: "
                f"{param_source}/{rule_policy}"
            )
        if not on_arms:
            raise ValueError(
                "啟用比較群組至少需要一個DL-on arm: "
                f"{param_source}/{rule_policy}"
            )
        for on_arm in on_arms:
            pairs.append((param_source, rule_policy, off_arm, on_arm))
    return tuple(pairs)


def _assert_same_shared_baseline(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    *,
    arm_id: str,
) -> None:
    ignored = {
        "arm_id",
        "param_source",
        "rule_policy",
        "dl_enabled",
        "dl_id",
        "direct_selection_delta_r",
    }
    keys = (set(existing) | set(candidate)) - ignored
    for key in sorted(keys):
        left = existing.get(key)
        right = candidate.get(key)
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(
                    f"多DL比較的共用基準不一致: arm={arm_id}, key={key}, "
                    f"first={left}, repeated={right}"
                )
        elif left != right:
            raise ValueError(
                f"多DL比較的共用基準不一致: arm={arm_id}, key={key}, "
                f"first={left!r}, repeated={right!r}"
            )


def _scenario_payloads(
    pair_payloads: dict[str, dict[str, Any]],
    direct_r: dict[str, float],
    *,
    settings: StrategyComparisonSettings,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    enabled_ids = {arm.arm_id for arm in settings.enabled_arms}
    for group_id, pair in pair_payloads.items():
        param_source, rule_policy, off_arm, on_arm = pair["arm_contract"]
        if off_arm.arm_id in enabled_ids:
            baseline = {
                **dict(pair["payload"].get("no_filter") or {}),
                "arm_id": off_arm.arm_id,
                "param_source": param_source,
                "rule_policy": rule_policy,
                "dl_enabled": False,
                "dl_id": None,
                "direct_selection_delta_r": 0.0,
            }
            existing = output.get(off_arm.arm_id)
            if existing is None:
                output[off_arm.arm_id] = baseline
            else:
                _assert_same_shared_baseline(
                    existing,
                    baseline,
                    arm_id=off_arm.arm_id,
                )
        if on_arm.arm_id in enabled_ids:
            output[on_arm.arm_id] = {
                **dict(pair["payload"].get("quality_filter") or {}),
                "arm_id": on_arm.arm_id,
                "param_source": param_source,
                "rule_policy": rule_policy,
                "dl_enabled": True,
                "dl_id": on_arm.dl_id,
                "direct_selection_delta_r": float(direct_r[group_id]),
            }
    return output


def _summary_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    rows = []
    for arm in settings.enabled_arms:
        payload = scenarios[arm.arm_id]
        rows.append(
            (
                arm.arm_id,
                arm.name,
                _fmt(payload.get("total_return_pct"), unit="%"),
                _fmt(payload.get("max_drawdown_pct"), unit="%"),
                _fmt(payload.get("return_over_max_drawdown")),
                _fmt(payload.get("annual_return_pct"), unit="%"),
                _fmt(payload.get("expected_value_r"), unit=" R"),
                _fmt(payload.get("payoff_ratio")),
                _fmt(payload.get("avg_exposure_pct"), unit="%"),
                _fmt(payload.get("trade_count"), digits=0),
                _fmt(payload.get("direct_selection_delta_r"), unit=" R"),
            )
        )
    return render_table(
        (
            "編號",
            "比較對象",
            "報酬",
            "MDD",
            "RoMD",
            "年化",
            "EV",
            "Payoff",
            "曝險",
            "交易",
            "同參數DL選擇R",
        ),
        rows,
    )


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    a = _metric(left, key)
    b = _metric(right, key)
    return None if a is None or b is None else a - b


def _contrast_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    rows = []
    for contrast in settings.enabled_contrasts:
        left = scenarios[contrast.left]
        right = scenarios[contrast.right]
        rows.append(
            (
                contrast.contrast_id,
                contrast.description,
                _fmt(_delta(left, right, "total_return_pct"), unit="pp"),
                _fmt(_delta(left, right, "max_drawdown_pct"), unit="pp"),
                _fmt(_delta(left, right, "return_over_max_drawdown")),
                _fmt(_delta(left, right, "annual_return_pct"), unit="pp"),
                _fmt(_delta(left, right, "expected_value_r"), unit=" R"),
                _fmt(_delta(left, right, "avg_exposure_pct"), unit="pp"),
                _fmt(_delta(left, right, "trade_count"), digits=0),
                _fmt(
                    _delta(left, right, "direct_selection_delta_r"),
                    unit=" R",
                ),
            )
        )
    return render_table(
        (
            "比較",
            "用途",
            "Δ報酬",
            "ΔMDD",
            "ΔRoMD",
            "Δ年化",
            "ΔEV",
            "Δ曝險",
            "Δ交易",
            "Δ直接選擇R",
        ),
        rows,
    )


def _yearly_table(
    pair_payloads: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    enabled_ids = tuple(arm.arm_id for arm in settings.enabled_arms)
    by_id: dict[str, dict[int, float | None]] = {arm_id: {} for arm_id in enabled_ids}
    for pair in pair_payloads.values():
        _param_source, _rule_policy, off_arm, on_arm = pair["arm_contract"]
        for row in pair["payload"].get("yearly") or []:
            year = int(row["year"])
            if off_arm.arm_id in by_id:
                value = row.get("no_filter_return_pct")
                existing = by_id[off_arm.arm_id].get(year)
                if existing is not None and value is not None:
                    if not math.isclose(
                        float(existing), float(value), rel_tol=0.0, abs_tol=1e-10
                    ):
                        raise ValueError(
                            "多DL比較的共用基準年度報酬不一致: "
                            f"arm={off_arm.arm_id}, year={year}, "
                            f"first={existing}, repeated={value}"
                        )
                else:
                    by_id[off_arm.arm_id][year] = value
            if on_arm.arm_id in by_id:
                by_id[on_arm.arm_id][year] = row.get("quality_filter_return_pct")
    years = sorted({year for values in by_id.values() for year in values})
    rows = [
        (
            year,
            *(_fmt(by_id[arm_id].get(year), unit="%") for arm_id in enabled_ids),
        )
        for year in years
    ]
    return render_table(("年度", *enabled_ids), rows)


def _comparison_period(pair_payloads: dict[str, dict[str, Any]]) -> Any:
    periods = {
        json.dumps(
            dict(pair["payload"].get("metadata") or {}).get("comparison_period"),
            sort_keys=True,
        )
        for pair in pair_payloads.values()
    }
    if len(periods) != 1:
        raise ValueError(f"策略比較期間不一致: {periods}")
    return dict(next(iter(pair_payloads.values()))["payload"].get("metadata") or {}).get(
        "comparison_period"
    )


def _render_report(
    *,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
    pair_payloads: dict[str, dict[str, Any]],
) -> str:
    return "\n\n".join(
        (
            render_title("策略績效比較"),
            render_key_values(
                (
                    ("期間", _comparison_period(pair_payloads)),
                    ("Dataset", settings.dataset),
                    ("Param policy", settings.param_policy),
                    ("Max positions", settings.max_positions),
                    ("Rotation", settings.rotation),
                    ("Config fingerprint", status["config_fingerprint"]),
                    ("比較設定", "config/strategy_compare.py"),
                )
            ),
            render_section("1. 比較結果"),
            _summary_table(scenarios, settings=settings),
            render_section("2. 設定中的差異比較"),
            _contrast_table(scenarios, settings=settings),
            render_section("3. 年度結果"),
            _yearly_table(pair_payloads, settings=settings),
            render_section("4. 判讀原則"),
            (
                "以config中啟用的contrast逐項判讀；不得用單一年份改善取代"
                "全期RoMD、EV、直接交易選擇R與年度穩定性。比較流程不建立Label、"
                "不選模型也不訓練模型權重；可依config透過正式共用服務補建既有模型"
                "的forward-OOS scores與比較所需策略參數工件。"
            ),
        )
    ).rstrip() + "\n"


def _collect_ready_status_after_preparation(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    requested_plan: StrategyPreparationPlan,
) -> dict[str, Any]:
    """Refresh the full orchestration status after prerequisite builders finish.

    The preparation layer owns artifact readiness only; config fingerprints and
    replay-ready resolved paths belong to this orchestration layer. Recollecting
    through the public wrapper prevents the low-level preparation payload from
    replacing the richer status contract expected by report and output writers.
    """
    refreshed = collect_artifact_status(project_root=root, settings=settings)
    refreshed["requested_preparation_plan"] = requested_plan
    required_keys = (
        "config_fingerprint",
        "artifact_identities",
        "resolved_parameter_paths",
        "preparation_plan",
        "comparison_period",
    )
    missing = [key for key in required_keys if key not in refreshed]
    if missing:
        raise RuntimeError(
            "前置完成後狀態契約不完整: missing=" + ",".join(missing)
        )
    if not bool(refreshed.get("comparison_ready")) or refreshed.get("overall_status") != "READY":
        raise RuntimeError("前置完成後正式比較狀態仍非READY")
    return refreshed


def _run_directory(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    fingerprint: str,
) -> tuple[Path, Path]:
    output_root = _resolve_relative_path(root, settings.output_root)
    enabled_ids = "-".join(arm.arm_id for arm in settings.enabled_arms)
    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / f"{timestamp}_{enabled_ids}_{fingerprint}"
    latest_dir = output_root / "latest"
    return run_dir, latest_dir


def run_strategy_comparison(
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
    status: dict[str, Any] | None = None,
    auto_prepare: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = get_strategy_comparison_settings()
    status = status or collect_artifact_status(project_root=root, settings=settings)
    requested_fingerprint = str(status["config_fingerprint"])
    requested_plan = status["preparation_plan"]
    if status["overall_status"] == "BLOCKED":
        print("\n" + render_status(project_root=root, settings=settings, status=status))
        raise RuntimeError(
            "目前啟用比較缺少不可自動產生的上游工件；請依狀態頁使用正式模型入口處理。"
        )
    if status["overall_status"] == "PREPARABLE":
        if not auto_prepare:
            raise RuntimeError("目前工件可自動準備，但本次已停用auto_prepare")
        prepare_strategy_comparison_artifacts(
            project_root=root,
            settings=settings,
            status=status,
        )
        status = _collect_ready_status_after_preparation(
            root=root,
            settings=settings,
            requested_plan=requested_plan,
        )

    comparison_period = dict(status.get("comparison_period") or {})
    comparison_start = str(comparison_period.get("start") or "")
    comparison_end = str(comparison_period.get("end") or "")
    if not comparison_start or not comparison_end:
        raise RuntimeError("正式比較缺少已解析的共同comparison period")

    run_dir, latest_dir = _run_directory(
        root=root,
        settings=settings,
        fingerprint=status["config_fingerprint"],
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    pair_payloads: dict[str, dict[str, Any]] = {}
    direct_r: dict[str, float] = {}
    for param_source, rule_policy, off_arm, on_arm in _execution_pairs(settings):
        if not on_arm.dl_id:
            raise ValueError(f"DL-on arm缺少dl_id: {on_arm.arm_id}")
        dl: StrategyDLSource = settings.dl_sources[on_arm.dl_id]
        group_id = f"{param_source}__{rule_policy}__{on_arm.dl_id}"
        pair_dir = run_dir / "pairs" / group_id
        all_off = rule_policy == "all_off"
        pair_payload = run_comparison(
            project_root=root,
            dataset=settings.dataset,
            params_path=str(status["resolved_parameter_paths"][param_source]),
            param_policy=settings.param_policy,
            max_positions=settings.max_positions,
            enable_rotation=settings.rotation == "on",
            fixed_risk=None,
            max_position_cap_pct=None,
            comparison_mode=COMPARISON_MODE_HARD_FILTER,
            optional_entry_filter_policy=(
                OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
                if all_off
                else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
            ),
            filter_id=dl.filter_id,
            model_architecture=dl.model_architecture,
            experiment_profile=dl.experiment_profile,
            threshold=dl.threshold,
            output_dir_override=pair_dir,
            comparison_start_date=comparison_start,
            comparison_end_date=comparison_end,
            quiet=quiet,
            shared_param_overrides=(
                ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None
            ),
        )
        pair_payloads[group_id] = {
            "arm_contract": (param_source, rule_policy, off_arm, on_arm),
            "payload": pair_payload,
        }
        direct_r[group_id] = _load_direct_selection_r(pair_dir, root=root)

    scenarios = _scenario_payloads(pair_payloads, direct_r, settings=settings)
    report = _render_report(
        settings=settings,
        status=status,
        scenarios=scenarios,
        pair_payloads=pair_payloads,
    )
    report_path = run_dir / "strategy_comparison.md"
    json_path = run_dir / "strategy_comparison.json"
    manifest_path = run_dir / "manifest.json"
    report_path.write_text(report, encoding="utf-8")
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "COMPLETED",
        "created_at": get_taipei_now().isoformat(),
        "config_fingerprint": status["config_fingerprint"],
        "requested_config_fingerprint": requested_fingerprint,
        "settings": settings.as_dict(),
        "artifact_identities": status["artifact_identities"],
        "comparison_period": comparison_period,
        "comparison_period_source": status.get("comparison_period_source"),
        "requested_preparation_plan": requested_plan.as_dict(),
        "final_preparation_plan": status["preparation_plan"].as_dict(),
        "scenarios": scenarios,
        "contrasts": {
            item.contrast_id: {
                "left": item.left,
                "right": item.right,
                "description": item.description,
            }
            for item in settings.enabled_contrasts
        },
        "pairs": {
            key: value["payload"] for key, value in pair_payloads.items()
        },
    }
    _write_json(json_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "created_at": payload["created_at"],
            "config_path": "config/strategy_compare.py",
            "config_fingerprint": status["config_fingerprint"],
            "requested_config_fingerprint": requested_fingerprint,
            "enabled_arms": [arm.as_dict() for arm in settings.enabled_arms],
            "enabled_contrasts": [
                item.as_dict() for item in settings.enabled_contrasts
            ],
            "artifact_identities": status["artifact_identities"],
            "comparison_period": comparison_period,
            "comparison_period_source": status.get("comparison_period_source"),
            "preparation_policy": settings.preparation.as_dict(),
            "requested_preparation_plan": requested_plan.as_dict(),
            "final_preparation_plan": status["preparation_plan"].as_dict(),
            "run_dir": project_relative_display_path(run_dir, project_root=root),
        },
    )

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source, filename in (
        (report_path, "strategy_comparison.md"),
        (json_path, "strategy_comparison.json"),
        (manifest_path, "manifest.json"),
    ):
        shutil.copy2(source, latest_dir / filename)

    print("\n" + report)
    print_artifact_paths(
        (
            ("策略比較Markdown", report_path),
            ("策略比較JSON", json_path),
            ("執行Manifest", manifest_path),
            ("最新結果", latest_dir),
        ),
        project_root=root,
    )
    return payload


def show_strategy_comparison_status(
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = get_strategy_comparison_settings()
    status = collect_artifact_status(project_root=root, settings=settings)
    print("\n" + render_status(project_root=root, settings=settings, status=status))
    return status


__all__ = [
    "collect_artifact_status",
    "render_execution_plan",
    "render_status",
    "run_strategy_comparison",
    "show_strategy_comparison_status",
]

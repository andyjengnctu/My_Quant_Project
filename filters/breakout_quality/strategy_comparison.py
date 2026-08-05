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
    strategy_comparison_fingerprint,
)
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.paths import resolve_filter_artifact_paths
from filters.breakout_quality.strategy_compare_engine import (
    COMPARISON_MODE_HARD_FILTER,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    PARAM_POLICY_SPECS,
    _load_param_source,
    _resolve_params_path,
    _validate_requested_param_policy,
    run_comparison,
)
from filters.breakout_quality.strategy_rule_policies import (
    ALL_RULE_FILTERS_OFF_OVERRIDES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA_VERSION = 1


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


def _resolve_param_source_path(
    root: Path,
    settings: StrategyComparisonSettings,
    source_id: str,
) -> Path:
    source = settings.parameter_sources[source_id]
    if source.path_template in (None, ""):
        return _resolve_params_path(
            root=root,
            params_path=None,
            param_policy=settings.param_policy,
            allow_static_diagnostic=False,
        ).resolve()
    filename = str(PARAM_POLICY_SPECS[settings.param_policy]["filename"])
    try:
        rendered = str(source.path_template).format(param_filename=filename)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"parameter source {source_id}路徑模板只支援{{param_filename}}"
        ) from exc
    return _resolve_relative_path(root, rendered)


def _configured_groups(
    settings: StrategyComparisonSettings,
) -> dict[tuple[str, str], dict[bool, StrategyComparisonArm]]:
    groups: dict[tuple[str, str], dict[bool, StrategyComparisonArm]] = {}
    for arm in settings.arms.values():
        groups.setdefault((arm.param_source, arm.rule_policy), {})[
            bool(arm.dl_enabled)
        ] = arm
    return groups


def _execution_pairs(
    settings: StrategyComparisonSettings,
) -> tuple[tuple[str, str, StrategyComparisonArm, StrategyComparisonArm], ...]:
    enabled_groups = {
        (arm.param_source, arm.rule_policy) for arm in settings.enabled_arms
    }
    groups = _configured_groups(settings)
    pairs = []
    for param_source, rule_policy in groups:
        if (param_source, rule_policy) not in enabled_groups:
            continue
        states = groups[(param_source, rule_policy)]
        off_arm = states[False]
        on_arm = states[True]
        pairs.append((param_source, rule_policy, off_arm, on_arm))
    return tuple(pairs)


def _validate_param_artifact(
    path: Path,
    *,
    param_policy: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    if not path.is_file():
        return False, "MISSING", None
    try:
        source = _load_param_source(path)
        policy = _validate_requested_param_policy(source, param_policy)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"INVALID: {type(exc).__name__}: {exc}", None
    if str(source.get("kind") or "") != "rolling_active_param_ensemble":
        return False, f"INVALID_KIND: {source.get('kind')}", None
    return True, "READY", policy


def _validate_param_training_identity(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    source_id: str,
) -> tuple[bool, str, Path | None]:
    source = settings.parameter_sources[source_id]
    if not source.trained_with_dl_id:
        return True, "NOT_REQUIRED", None
    if not source.identity_manifest_path:
        return False, "IDENTITY_MANIFEST_NOT_CONFIGURED", None
    manifest_path = _resolve_relative_path(root, source.identity_manifest_path)
    payload = _read_json(manifest_path)
    if payload is None:
        return False, "IDENTITY_MANIFEST_MISSING_OR_INVALID", manifest_path
    dl = settings.dl_sources[source.trained_with_dl_id]
    binary_runtime = dict(payload.get("binary_runtime") or {})
    actual_filter = str(
        binary_runtime.get("filter_id") or payload.get("filter_id") or ""
    )
    actual_architecture = str(
        binary_runtime.get("model_architecture")
        or payload.get("model_architecture")
        or ""
    )
    actual_profile = str(
        binary_runtime.get("experiment_profile")
        or payload.get("experiment_profile")
        or ""
    )
    training_dl_enabled = payload.get("training_dl_enabled")
    if training_dl_enabled is not None and not bool(training_dl_enabled):
        return False, "TRAINING_DL_DISABLED", manifest_path
    if (
        actual_filter != dl.filter_id
        or actual_architecture != dl.model_architecture
        or actual_profile != dl.experiment_profile
    ):
        return False, "DL_IDENTITY_MISMATCH", manifest_path
    return True, "READY", manifest_path


def collect_artifact_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    current = settings or get_strategy_comparison_settings()
    required_param_sources = {
        arm.param_source for arm in current.enabled_arms
    }
    required_dl_sources = {
        arm.dl_id
        for arm in current.enabled_arms
        if arm.dl_enabled and arm.dl_id
    }

    parameter_rows: dict[str, Any] = {}
    resolved_parameter_paths: dict[str, Path] = {}
    artifact_identities: dict[str, Any] = {}
    for source_id in current.parameter_sources:
        if source_id not in required_param_sources:
            continue
        path = _resolve_param_source_path(root, current, source_id)
        resolved_parameter_paths[source_id] = path
        ready, status, policy = _validate_param_artifact(
            path,
            param_policy=current.param_policy,
        )
        identity_ready, identity_status, identity_path = _validate_param_training_identity(
            root=root,
            settings=current,
            source_id=source_id,
        )
        ready = bool(ready and identity_ready)
        if status == "READY" and not identity_ready:
            status = identity_status
        sha256 = compute_file_sha256(path) if path.is_file() else None
        artifact_identities[f"param:{source_id}"] = {
            "path": project_relative_display_path(path, project_root=root),
            "sha256": sha256,
            "identity_status": identity_status,
        }
        parameter_rows[source_id] = {
            "ready": ready,
            "status": status,
            "path": project_relative_display_path(path, project_root=root),
            "sha256": sha256,
            "selector": None if policy is None else policy.get("selector"),
            "identity_status": identity_status,
            "identity_manifest_path": (
                None
                if identity_path is None
                else project_relative_display_path(identity_path, project_root=root)
            ),
        }

    dl_rows: dict[str, Any] = {}
    for dl_id in current.dl_sources:
        if dl_id not in required_dl_sources:
            continue
        source = current.dl_sources[dl_id]
        artifacts = resolve_filter_artifact_paths(
            root,
            source.filter_id,
            source.model_architecture,
            source.experiment_profile,
        )
        files = {
            "model": artifacts.model_path,
            "manifest": artifacts.manifest_path,
            "forward_scores": artifacts.score_path,
        }
        file_rows = {}
        ready = True
        for key, path in files.items():
            exists = path.is_file()
            ready = bool(ready and exists)
            sha256 = compute_file_sha256(path) if exists else None
            file_rows[key] = {
                "ready": exists,
                "status": "READY" if exists else "MISSING",
                "path": project_relative_display_path(path, project_root=root),
                "sha256": sha256,
            }
            artifact_identities[f"dl:{dl_id}:{key}"] = {
                "path": file_rows[key]["path"],
                "sha256": sha256,
            }
        dl_rows[dl_id] = {
            "ready": ready,
            "status": "READY" if ready else "MISSING",
            "identity": source.as_dict(),
            "files": file_rows,
        }

    comparison_ready = all(row["ready"] for row in parameter_rows.values()) and all(
        row["ready"] for row in dl_rows.values()
    )
    fingerprint = strategy_comparison_fingerprint(
        current,
        artifact_identities=artifact_identities,
    )
    return {
        "comparison_ready": comparison_ready,
        "parameters": parameter_rows,
        "dl_sources": dl_rows,
        "artifact_identities": artifact_identities,
        "config_fingerprint": fingerprint,
        "resolved_parameter_paths": resolved_parameter_paths,
    }


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
        artifact_rows.append((f"param:{source_id}", row["status"], row["path"]))
        if row.get("identity_manifest_path"):
            artifact_rows.append(
                (
                    f"param:{source_id}:identity",
                    row["identity_status"],
                    row["identity_manifest_path"],
                )
            )
    for dl_id, row in current_status["dl_sources"].items():
        for key, file_row in row["files"].items():
            artifact_rows.append(
                (f"dl:{dl_id}:{key}", file_row["status"], file_row["path"])
            )
    return "\n\n".join(
        (
            render_title("策略績效比較設定與工件狀態"),
            render_key_values(
                (
                    ("設定檔", "config/strategy_compare.py"),
                    ("Dataset", current.dataset),
                    ("期間", f"{current.start_date or 'artifact start'} ～ {current.end_date or 'artifact end'}"),
                    ("Param policy", current.param_policy),
                    ("Max positions", current.max_positions),
                    ("Rotation", current.rotation),
                    ("Config fingerprint", current_status["config_fingerprint"]),
                    ("比較狀態", "READY" if current_status["comparison_ready"] else "NOT READY"),
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
            render_section("3. 所需工件"),
            render_table(("工件", "狀態", "路徑"), artifact_rows),
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
            output[off_arm.arm_id] = {
                **dict(pair["payload"].get("no_filter") or {}),
                "arm_id": off_arm.arm_id,
                "param_source": param_source,
                "rule_policy": rule_policy,
                "dl_enabled": False,
                "direct_selection_delta_r": 0.0,
            }
        if on_arm.arm_id in enabled_ids:
            output[on_arm.arm_id] = {
                **dict(pair["payload"].get("quality_filter") or {}),
                "arm_id": on_arm.arm_id,
                "param_source": param_source,
                "rule_policy": rule_policy,
                "dl_enabled": True,
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
                by_id[off_arm.arm_id][year] = row.get("no_filter_return_pct")
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
                "全期RoMD、EV、直接交易選擇R與年度穩定性。比較流程只讀"
                "既有模型、score與策略參數工件，不執行模型或optimizer訓練。"
            ),
        )
    ).rstrip() + "\n"


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
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = get_strategy_comparison_settings()
    status = collect_artifact_status(project_root=root, settings=settings)
    if not status["comparison_ready"]:
        print("\n" + render_status(project_root=root, settings=settings, status=status))
        raise FileNotFoundError(
            "目前啟用比較所需工件不完整；比較App不會自動訓練模型、匯出score或執行optimizer。"
        )

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
            comparison_start_date=settings.start_date,
            comparison_end_date=settings.end_date,
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
        "settings": settings.as_dict(),
        "artifact_identities": status["artifact_identities"],
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
            "enabled_arms": [arm.as_dict() for arm in settings.enabled_arms],
            "enabled_contrasts": [
                item.as_dict() for item in settings.enabled_contrasts
            ],
            "artifact_identities": status["artifact_identities"],
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
    "render_status",
    "run_strategy_comparison",
    "show_strategy_comparison_status",
]

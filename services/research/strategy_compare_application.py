"""Strategy Compare application/service boundary for Research.

User-facing Research entry points call this module rather than importing
Strategy Compare implementation modules from ``filters/`` directly.  The
service owns application-level dispatch and the fail-closed boundary for the
historical runtime-integration gate; scientific settings remain in ``config/``
and comparison/model/optimizer implementations remain in their canonical
owners.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from config.strategy_compare import (
    get_strategy_comparison_menu_profiles,
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_profiles,
    get_strategy_multi_seed_robustness_settings,
    get_strategy_runtime_integration_settings,
)
from core.console_report import (
    print_artifact_paths,
    render_key_values,
    render_table,
    render_title,
)
from core.file_integrity import atomic_write_json
from core.research_report_contract import (
    STRATEGY_CONSISTENCY_METRIC_KEYS,
    section_contract,
    table_contract,
)
from core.report_metrics import (
    CORE_STRATEGY_RESULT_METRICS,
    MFE_SAFETY_COMPARE_RESULT_METRICS,
)
from core.runtime_utils import get_taipei_now


_RUNTIME_INTEGRATION_MUTATING_ACTIONS = frozenset({"run", "gate", "promote", "apply", "refresh"})
_RUNTIME_INTEGRATION_READ_ONLY_ACTIONS = frozenset({"status", "show", "latest"})

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _metric_spec_map() -> dict[str, object]:
    return {
        metric.key: metric
        for metric in (*CORE_STRATEGY_RESULT_METRICS, *MFE_SAFETY_COMPARE_RESULT_METRICS)
    }


def _format_metric_value(key: str, value) -> str:
    if value is None:
        return "-"
    spec = _metric_spec_map().get(str(key))
    if spec is None:
        return f"{float(value):.4f}"
    digits = int(spec.digits)
    unit = str(spec.unit or "")
    return f"{float(value):.{digits}f}{unit}"


def _payload_geometry_arm(payload: dict, arm_id: str) -> dict:
    geometry = dict((payload.get("diagnostics") or {}).get("mfe_safety_geometry") or {})
    if str(geometry.get("status") or "") != "AVAILABLE":
        return {}
    return dict((geometry.get("arms") or {}).get(str(arm_id)) or {})


def _payload_metric(payload: dict, arm_id: str, key: str):
    if key in {"high_mfe_high_safety_pct", "high_mfe_total_pct"}:
        return _payload_geometry_arm(payload, arm_id).get(key)
    return dict((payload.get("scenarios") or {}).get(str(arm_id)) or {}).get(key)


def _signed_delta(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _direction_consistency(oos_delta, rolling_delta) -> str:
    if oos_delta is None or rolling_delta is None:
        return "N/A"
    eps = 1e-12
    oos_sign = 0 if abs(float(oos_delta)) <= eps else (1 if float(oos_delta) > 0 else -1)
    rolling_sign = 0 if abs(float(rolling_delta)) <= eps else (1 if float(rolling_delta) > 0 else -1)
    if oos_sign == 0 and rolling_sign == 0:
        return "FLAT"
    if oos_sign == rolling_sign:
        return "SAME"
    return "DIVERGED"


def build_strategy_oos_rolling_consistency(
    oos_payload: dict,
    rolling_payload: dict,
) -> dict:
    """Build one read-only consistency view from two canonical Strategy Compare results."""

    oos_settings = dict(oos_payload.get("settings") or {})
    rolling_settings = dict(rolling_payload.get("settings") or {})
    suite_id = str(oos_settings.get("suite_id") or "")
    if not suite_id or suite_id != str(rolling_settings.get("suite_id") or ""):
        raise ValueError("OOS／Rolling Consistency要求相同Compare Suite")
    oos_arms = set((oos_payload.get("scenarios") or {}).keys())
    rolling_arms = set((rolling_payload.get("scenarios") or {}).keys())
    if oos_arms != rolling_arms:
        raise ValueError("OOS／Rolling Consistency arms不一致")
    oos_contrasts = dict(oos_payload.get("contrasts") or {})
    rolling_contrasts = dict(rolling_payload.get("contrasts") or {})
    if oos_contrasts != rolling_contrasts:
        raise ValueError("OOS／Rolling Consistency contrasts不一致")

    metric_keys = STRATEGY_CONSISTENCY_METRIC_KEYS
    arm_rows = []
    for arm_id in sorted(oos_arms):
        for key in metric_keys:
            arm_rows.append({
                "arm_id": arm_id,
                "metric": key,
                "oos": _payload_metric(oos_payload, arm_id, key),
                "rolling": _payload_metric(rolling_payload, arm_id, key),
            })
    contrast_rows = []
    for contrast_id, raw in oos_contrasts.items():
        contrast = dict(raw or {})
        left = str(contrast.get("left") or "")
        right = str(contrast.get("right") or "")
        for key in metric_keys:
            oos_delta = _signed_delta(
                _payload_metric(oos_payload, left, key),
                _payload_metric(oos_payload, right, key),
            )
            rolling_delta = _signed_delta(
                _payload_metric(rolling_payload, left, key),
                _payload_metric(rolling_payload, right, key),
            )
            contrast_rows.append({
                "contrast_id": str(contrast_id),
                "left": left,
                "right": right,
                "metric": key,
                "oos_delta_left_minus_right": oos_delta,
                "rolling_delta_left_minus_right": rolling_delta,
                "direction": _direction_consistency(oos_delta, rolling_delta),
            })
    return {
        "schema_version": 1,
        "created_at": get_taipei_now().isoformat(),
        "suite_id": suite_id,
        "oos_profile_id": str(oos_settings.get("profile_id") or ""),
        "rolling_profile_id": str(rolling_settings.get("profile_id") or ""),
        "oos_fingerprint": str(oos_payload.get("config_fingerprint") or ""),
        "rolling_fingerprint": str(rolling_payload.get("config_fingerprint") or ""),
        "oos_period": dict(oos_payload.get("comparison_period") or {}),
        "rolling_period": dict(rolling_payload.get("comparison_period") or {}),
        "arm_rows": arm_rows,
        "contrast_rows": contrast_rows,
        "contract": "descriptive consistency only; left-minus-right contrast; no promotion decision",
    }


def render_strategy_oos_rolling_consistency(payload: dict) -> str:
    specs = _metric_spec_map()
    arm_rows = []
    for row in list(payload.get("arm_rows") or []):
        key = str(row.get("metric") or "")
        spec = specs.get(key)
        arm_rows.append((
            str(row.get("arm_id") or ""),
            str(getattr(spec, "label", key)),
            _format_metric_value(key, row.get("oos")),
            _format_metric_value(key, row.get("rolling")),
        ))
    contrast_rows = []
    for row in list(payload.get("contrast_rows") or []):
        key = str(row.get("metric") or "")
        spec = specs.get(key)
        contrast_rows.append((
            str(row.get("contrast_id") or ""),
            str(getattr(spec, "label", key)),
            _format_metric_value(key, row.get("oos_delta_left_minus_right")),
            _format_metric_value(key, row.get("rolling_delta_left_minus_right")),
            str(row.get("direction") or "N/A"),
        ))
    return "\n\n".join((
        render_title("OOS ↔ Rolling Consistency"),
        render_key_values((
            ("Compare Suite", payload.get("suite_id")),
            ("OOS fingerprint", payload.get("oos_fingerprint")),
            ("Rolling fingerprint", payload.get("rolling_fingerprint")),
            ("Contrast delta", "left - right"),
        )),
        section_contract("strategy.oos_rolling_consistency", "arm_values").title + "\n" + render_table(
            table_contract("strategy.oos_rolling_consistency", "arm_values", "arm_values").headers,
            arm_rows,
            alignments=table_contract("strategy.oos_rolling_consistency", "arm_values", "arm_values").alignments,
        ),
        section_contract("strategy.oos_rolling_consistency", "contrast_consistency").title + "\n" + render_table(
            table_contract("strategy.oos_rolling_consistency", "contrast_consistency", "contrast_consistency").headers,
            contrast_rows,
            alignments=table_contract("strategy.oos_rolling_consistency", "contrast_consistency", "contrast_consistency").alignments,
        ),
    ))


def materialize_strategy_oos_rolling_consistency(
    oos_payload: dict,
    rolling_payload: dict,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    root = Path(project_root).resolve()
    payload = build_strategy_oos_rolling_consistency(oos_payload, rolling_payload)
    out_dir = root / "outputs" / "strategy_compare" / "extending_window" / "consistency" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "strategy_consistency.json"
    md_path = out_dir / "strategy_consistency.md"
    atomic_write_json(json_path, payload)
    md_path.write_text(render_strategy_oos_rolling_consistency(payload) + "\n", encoding="utf-8")
    print("\n" + render_strategy_oos_rolling_consistency(payload))
    print_artifact_paths(
        (("Consistency Markdown", md_path), ("Consistency JSON", json_path)),
        project_root=root,
    )
    return payload


def resolve_strategy_comparison_execution(profile_id: str):
    """Resolve one comparison profile and its auditable dependency plan."""

    from services.research.strategy_comparison import (
        render_execution_plan,
        resolve_comparison_plan,
    )

    settings = get_strategy_comparison_settings(str(profile_id))
    resolved_plan = resolve_comparison_plan(settings=settings)
    rendered_plan = render_execution_plan(
        settings=settings,
        status=resolved_plan.status_dict(),
    )
    return settings, resolved_plan, rendered_plan


def execute_strategy_comparison(
    *,
    resolved_plan,
    settings,
    producer_handlers: dict[str, Callable] | None = None,
) -> dict:
    """Execute an already-resolved comparison through the canonical engine."""

    from services.research.strategy_comparison import run_strategy_comparison

    return run_strategy_comparison(
        resolved_plan=resolved_plan,
        auto_prepare=True,
        settings=settings,
        quiet=True,
        producer_handlers=dict(producer_handlers or {}),
    )


def show_strategy_comparison_profile_status(*, profile_id: str) -> None:
    """Render readiness for one configured Strategy Compare profile."""

    from services.research.strategy_comparison import show_strategy_comparison_status

    settings = get_strategy_comparison_settings(str(profile_id))
    show_strategy_comparison_status(settings=settings)


def run_strategy_multi_seed_robustness(
    *,
    robustness_id: str,
    confirm: bool,
    model_upstream_preparer: Callable[[], int] | None = None,
) -> dict:
    """Run one configured robustness profile through its canonical implementation."""

    from services.research.strategy_multi_seed_robustness import run_multi_seed_robustness

    return run_multi_seed_robustness(
        robustness_id=str(robustness_id),
        confirm=bool(confirm),
        model_upstream_preparer=model_upstream_preparer,
    )


def show_strategy_multi_seed_robustness_status(*, robustness_id: str) -> None:
    from services.research.strategy_multi_seed_robustness import (
        show_multi_seed_robustness_status,
    )

    show_multi_seed_robustness_status(robustness_id=str(robustness_id))


def show_latest_strategy_multi_seed_robustness_report(*, robustness_id: str) -> None:
    from services.research.strategy_multi_seed_robustness import (
        show_latest_multi_seed_robustness_report,
    )

    show_latest_multi_seed_robustness_report(robustness_id=str(robustness_id))


def runtime_integration_execution_enabled() -> bool:
    """Return whether the gate is both enabled and bound only to current profiles."""

    cfg = get_strategy_runtime_integration_settings()
    current_profiles = {row["profile_id"] for row in get_strategy_comparison_menu_profiles()}
    current_robustness = {row["robustness_id"] for row in get_strategy_multi_seed_robustness_profiles()}
    return bool(
        cfg.enabled
        and cfg.selection_profile_id in current_profiles
        and cfg.forward_profile_id in current_profiles
        and cfg.selection_robustness_id in current_robustness
        and cfg.forward_robustness_id in current_robustness
    )


def runtime_integration_allowed_actions() -> tuple[str, ...]:
    """Return the CLI-visible actions allowed by current config and profile reachability."""

    if runtime_integration_execution_enabled():
        return ("run", "promote", "status", "latest")
    return ("status", "latest")


def dispatch_runtime_integration_action(action: str) -> None:
    """Dispatch the legacy runtime gate with a config-authoritative fail-closed guard.

    Read-only historical inspection remains available for compatibility.  Any
    action that can create gate output or change runtime state is rejected before
    importing the producer when the integration is disabled in config.
    """

    normalized = str(action).strip().lower()
    supported = _RUNTIME_INTEGRATION_MUTATING_ACTIONS | _RUNTIME_INTEGRATION_READ_ONLY_ACTIONS
    if normalized not in supported:
        raise ValueError(f"compare integration不支援的命令: {action}")

    cfg = get_strategy_runtime_integration_settings()
    if normalized in _RUNTIME_INTEGRATION_MUTATING_ACTIONS and not runtime_integration_execution_enabled():
        raise RuntimeError(
            f"{cfg.label}目前未綁定可執行的current Strategy Compare profiles；"
            "historical compatibility只允許status/latest唯讀檢視。"
        )

    from services.research.runtime_integration_gate import (
        run_runtime_integration_gate,
        show_latest_runtime_integration_report,
        show_runtime_integration_status,
    )

    if normalized in {"run", "gate"}:
        run_runtime_integration_gate()
        return
    if normalized in {"promote", "apply", "refresh"}:
        from services.breakout_quality.runtime_promotion import (
            apply_or_refresh_runtime_promotion,
        )

        apply_or_refresh_runtime_promotion()
        return
    if normalized in {"status", "show"}:
        show_runtime_integration_status()
        return
    show_latest_runtime_integration_report()


__all__ = [
    "build_strategy_oos_rolling_consistency",
    "dispatch_runtime_integration_action",
    "execute_strategy_comparison",
    "materialize_strategy_oos_rolling_consistency",
    "render_strategy_oos_rolling_consistency",
    "resolve_strategy_comparison_execution",
    "run_strategy_multi_seed_robustness",
    "runtime_integration_allowed_actions",
    "runtime_integration_execution_enabled",
    "show_latest_strategy_multi_seed_robustness_report",
    "show_strategy_comparison_profile_status",
    "show_strategy_multi_seed_robustness_status",
]

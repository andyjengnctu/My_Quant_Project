"""Apply or refresh the Gate-approved breakout-quality production runtime."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from config.breakout_quality import (
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_EVALUATION_WORKERS,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
)
from core.breakout_quality_policy import (
    get_breakout_quality_workflow_settings,
)
from core.strategy_compare_policy import (
    get_strategy_comparison_settings,
    get_strategy_runtime_integration_settings,
)
from core.active_param_ensemble import build_active_param_ensemble_schedule
from core.console_report import print_artifact_paths, render_key_values, render_title
from core.file_integrity import compute_file_sha256
from core.params_io import build_params_from_mapping, load_params_from_json, params_to_json_dict
from services.breakout_quality.export_scores import export_workflow_runtime_scores
from services.research.runtime_integration_gate import collect_runtime_integration_status
from filters.breakout_quality.strategy_compare_sources import OPTIONAL_ENTRY_FILTER_FIELDS
from services.research.strategy_comparison import collect_artifact_status
from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES
from filters.breakout_quality.workflow_runtime_score_store import (
    load_workflow_runtime_score_bundle,
    resolve_workflow_runtime_score_paths,
)
from filters.breakout_quality.workflow_io import write_json
from services.optimizer.strategy_param_repository import write_strategy_parameter_state_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _resolve_latest_min_roos_params(*, root: Path) -> tuple[Path, dict[str, Any], str]:
    cfg = get_strategy_runtime_integration_settings()
    settings = get_strategy_comparison_settings(cfg.forward_profile_id)
    status = collect_artifact_status(project_root=root, settings=settings)
    if str(status.get("overall_status") or "") != "READY":
        raise RuntimeError("Forward-OOS Strategy Compare工件尚未READY，不能套用正式runtime")

    candidate = next(
        (
            arm
            for arm in settings.enabled_arms
            if arm.dl_enabled
            and str(arm.dl_runtime_mode or "").endswith("score-constrained-optimal")
        ),
        None,
    )
    if candidate is None:
        raise RuntimeError("找不到Gate-approved exact candidate")
    if candidate.rule_policy != "all_off":
        raise RuntimeError(f"正式runtime candidate必須是all_off，收到 {candidate.rule_policy!r}")
    source_path_text = str((status.get("resolved_parameter_paths") or {}).get(candidate.param_source) or "")
    source_path = Path(source_path_text)
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到Min ROOS active params: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    schedule = build_active_param_ensemble_schedule(payload)
    if not schedule:
        raise ValueError("Min ROOS active params沒有任何生效紀錄")
    latest = schedule[-1]
    members = list(latest.get("members") or ())
    if len(members) != 1:
        raise ValueError(
            "正式base-finalist-best runtime必須在最新生效日只有一個member: "
            f"effective_date={latest.get('effective_date_text')}, member_count={len(members)}"
        )
    raw_params = dict(members[0].get("params") or {})
    if not raw_params:
        raise ValueError("Min ROOS最新member缺少params")
    return source_path, raw_params, str(latest.get("effective_date_text") or "")


def _build_official_runtime_params(raw_params: dict[str, Any], *, filter_id: str) -> dict[str, Any]:
    params = build_params_from_mapping(raw_params)
    overrides: dict[str, Any] = {
        **{field: False for field in OPTIONAL_ENTRY_FILTER_FIELDS},
        **dict(ALL_RULE_FILTERS_OFF_OVERRIDES),
        "use_breakout_quality_filter": False,
        "use_breakout_quality_ranking": True,
        "breakout_quality_filter_id": str(filter_id),
    }
    promoted = replace(params, **overrides)
    return params_to_json_dict(promoted)


def _render_promotion_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# MR-13E Exact Runtime Promotion",
            "",
            f"- Status: **{payload['status']}**",
            f"- Applied at: `{payload['applied_at']}`",
            f"- Workflow profile: `{payload['workflow']['experiment_profile']}`",
            f"- Ranking policy: `{payload['workflow']['ranking_policy']}`",
            f"- Param source: `{payload['params']['source_path']}`",
            f"- Param effective date: `{payload['params']['effective_date']}`",
            f"- Official params: `{payload['params']['official_path']}`",
            f"- Runtime scores: `{payload['scores']['score_path']}`",
            f"- Score coverage: `{payload['scores']['available_from']} ～ {payload['scores']['available_through']}`",
            f"- Gate fingerprint: `{payload['gate']['fingerprint']}`",
            "",
            "正式 runtime 使用 Min ROOS/all-off + MR-13E daily score + exact K/R0 constrained selector。",
        )
    ) + "\n"


def apply_or_refresh_runtime_promotion(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Require a current GO, refresh causal scores, then atomically activate current Min ROOS params."""

    root = Path(project_root).resolve()
    gate = collect_runtime_integration_status(project_root=root)
    if str(gate.get("status") or "") != "GO":
        raise RuntimeError(
            "Runtime promotion只接受目前Gate=GO；"
            f"目前={gate.get('status')}。請先執行Runtime整合Gate。"
        )
    workflow = get_breakout_quality_workflow_settings()
    if not workflow.runtime_strategy_enabled:
        raise RuntimeError("config目前未啟用正式 breakout-quality runtime strategy")

    source_path, raw_params, effective_date = _resolve_latest_min_roos_params(root=root)

    export_workflow_runtime_scores(
        project_root=root,
        filter_id=workflow.filter_id,
        model_architecture=workflow.model_architecture,
        experiment_profile=workflow.experiment_profile,
        inference_batch_size=int(BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE),
        inference_workers=int(BREAKOUT_QUALITY_EVALUATION_WORKERS),
        device=str(BREAKOUT_QUALITY_TORCH_DEVICE),
        mixed_precision=bool(BREAKOUT_QUALITY_USE_MIXED_PRECISION),
        mixed_precision_dtype=str(BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE),
        deterministic_algorithms=bool(BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS),
        allow_tf32=bool(BREAKOUT_QUALITY_ALLOW_TF32),
        preload_feature_bank=bool(BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK),
    )
    load_workflow_runtime_score_bundle.cache_clear()
    bundle = load_workflow_runtime_score_bundle(
        str(root), workflow.filter_id, workflow.model_architecture, workflow.experiment_profile
    )
    runtime = dict(bundle["manifest"]["runtime_eligibility"])
    runtime_paths = resolve_workflow_runtime_score_paths(
        project_root=root,
        filter_id=workflow.filter_id,
        model_architecture=workflow.model_architecture,
        experiment_profile=workflow.experiment_profile,
    )

    official_payload = _build_official_runtime_params(raw_params, filter_id=workflow.filter_id)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    state_write = write_strategy_parameter_state_artifact(
        root,
        artifact="active",
        payload=official_payload,
        backup_existing=True,
        backup_label=f"before_mr13e_{timestamp}",
    )
    official_path = Path(state_write["path"]).resolve()
    backup_path = state_write.get("backup_path")
    load_params_from_json(str(official_path))

    output_dir = root / "outputs" / "strategy_compare" / "runtime_integration" / "promotion" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PROMOTED",
        "applied_at": datetime.now().astimezone().isoformat(),
        "gate": {
            "status": gate.get("status"),
            "fingerprint": gate.get("fingerprint"),
            "comparison_anchor_experiment_profile": gate.get("settings", {}).get(
                "comparison_anchor_experiment_profile"
            ),
        },
        "workflow": {
            "filter_id": workflow.filter_id,
            "model_architecture": workflow.model_architecture,
            "experiment_profile": workflow.experiment_profile,
            "ranking_policy": workflow.runtime_ranking_policy,
            "ranking_options": dict(workflow.runtime_ranking_options),
        },
        "params": {
            "source_path": str(source_path.relative_to(root)),
            "source_sha256": compute_file_sha256(source_path),
            "effective_date": effective_date,
            "official_path": str(official_path.relative_to(root)),
            "official_sha256": compute_file_sha256(official_path),
            "backup_path": None if backup_path is None else str(backup_path.relative_to(root)),
            "rule_policy": "all_off",
            "use_breakout_quality_filter": False,
            "use_breakout_quality_ranking": True,
        },
        "scores": {
            "manifest_path": str(runtime_paths["manifest"].relative_to(root)),
            "manifest_sha256": compute_file_sha256(runtime_paths["manifest"]),
            "score_path": str(runtime_paths["score"].relative_to(root)),
            "score_sha256": compute_file_sha256(runtime_paths["score"]),
            "available_from": runtime.get("available_from"),
            "available_through": runtime.get("available_through"),
            "causal_information_contract": runtime.get("causal_information_contract"),
        },
    }
    json_path = output_dir / "runtime_promotion.json"
    md_path = output_dir / "runtime_promotion.md"
    write_json(json_path, payload)
    _write_text_atomic(md_path, _render_promotion_report(payload))

    latest_dir = output_dir.parent / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    write_json(latest_dir / "runtime_promotion.json", payload)
    _write_text_atomic(latest_dir / "runtime_promotion.md", _render_promotion_report(payload))

    print(render_title("MR-13E Exact 正式 Runtime"))
    print(
        render_key_values(
            (
                ("Status", "PROMOTED"),
                ("Workflow", workflow.experiment_profile),
                ("Ranking", workflow.runtime_ranking_policy),
                ("Min ROOS effective", effective_date),
                ("Score coverage", f"{runtime.get('available_from')} ～ {runtime.get('available_through')}"),
            )
        )
    )
    print_artifact_paths(
        (("Official params", official_path), ("Runtime score manifest", runtime_paths["manifest"]), ("Promotion report", md_path)),
        project_root=root,
    )
    return payload


__all__ = ["apply_or_refresh_runtime_promotion"]

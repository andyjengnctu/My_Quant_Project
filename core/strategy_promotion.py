"""Strategy Promotion SSOT: same current Compare Suite, immutable rule identity.

Only the user explicitly approves/activates. No scientific fitting or backtest is
performed in this module; historical evidence is linked, never copied or altered.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
from config.strategy_params import STRATEGY_PARAM_CANONICAL_PROFILE_BY_FAMILY_POLICY
from core.file_integrity import atomic_write_json, canonical_json_sha256, compute_file_sha256, load_json_strict
from core.portfolio_ensemble import get_ensemble_candidate_consensus_contract
from core.runtime_utils import get_taipei_now
from core.strategy_compare_policy import get_strategy_comparison_settings
from core.strategy_param_policy import get_strategy_param_fitting_seed_protocol, get_strategy_param_training_profile

PROMOTION_SCHEMA_VERSION = 1
PROMOTION_RULE_VERSION = "canonical_breakout_rule_v1"


def current_strategy_rules() -> tuple[dict, ...]:
    """Derive the entire visible set and order from [3], without a second list."""
    settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
    rows = []
    for arm in settings.enabled_arms:
        source = settings.parameter_sources[str(arm.param_source)]
        family = str(source.canonical_family or "").strip()
        profile_id = ""
        if source.builder is not None:
            profile_id = str(source.builder.options.get("training_profile_id") or "")
        if profile_id:
            research_profile = get_strategy_param_training_profile(profile_id)
            family = str(research_profile.filter_profile)
        else:
            canonical = STRATEGY_PARAM_CANONICAL_PROFILE_BY_FAMILY_POLICY.get((family, str(arm.param_policy)))
            if canonical:
                profile_id = canonical
        if not profile_id or not family:
            raise ValueError(f"Compare arm 缺少 canonical Strategy Parameter source: {arm.arm_id}")
        fitting_profile = get_strategy_param_training_profile(profile_id)
        seed_protocol = get_strategy_param_fitting_seed_protocol(fitting_profile.fitting_seed_protocol)
        # This is a capability test, NOT a second promotion membership filter.
        # Single-seed or DL work still appears in the same list, but the current
        # Trading multi-seed producer cannot silently reinterpret its contract.
        problems = []
        if arm.dl_enabled:
            problems.append("Trading 尚不支援這個 DL source adapter")
        if str(fitting_profile.price_basis) != "adjusted":
            problems.append("Trading 僅支援 adjusted-price")
        if bool(fitting_profile.validation_enabled):
            problems.append("Trading 尚不支援 validation fitting")
        if not seed_protocol.enabled or seed_protocol.seed_count < 2:
            problems.append("Trading 目前只支援 multi-seed fitting producer")
        if not str(fitting_profile.registered_scientific_fingerprint_sha256 or ""):
            problems.append("參數研究未註冊 scientific fingerprint")
        rule_payload = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "rule_version": PROMOTION_RULE_VERSION,
            "rule_policy": str(arm.rule_policy),
            "parameter_policy": str(arm.param_policy),
            "filter_family": family,
            "price_basis": str(fitting_profile.price_basis),
            "dl_enabled": bool(arm.dl_enabled),
            "dl_id": str(arm.dl_id or "") if arm.dl_enabled else None,
            "consensus": (
                get_ensemble_candidate_consensus_contract()
                if str(arm.param_policy) == "base-finalists-agree"
                else {"version": "canonical_single_best_v1"}
            ),
            "signal_lifecycle": "canonical_signal_shadow_pending_position_v1",
        }
        rows.append({
            "arm_id": str(arm.arm_id),
            "name": str(arm.name),
            "suite_id": str(settings.suite_id),
            "rolling_profile_id": str(settings.profile_id),
            "rule": rule_payload,
            "rule_sha256": canonical_json_sha256(rule_payload),
            "training_profile_id": profile_id,
            "parameter_set_identity": str(fitting_profile.parameter_set_identity),
            "registered_scientific_fingerprint_sha256": str(fitting_profile.registered_scientific_fingerprint_sha256 or ""),
            "trading_supported": not problems,
            "trading_blockers": tuple(problems),
        })
    return tuple(rows)


def current_strategy_rule(arm_id: str) -> dict:
    for row in current_strategy_rules():
        if row["arm_id"] == str(arm_id):
            return row
    raise ValueError(f"策略不在目前 [3] Compare Suite: {arm_id}")


def _approvals_path(root: str | Path) -> Path:
    return Path(root).resolve() / "outputs" / "strategy_promotion" / "approved_rules.json"


def _release_path(root: str | Path) -> Path:
    return Path(root).resolve() / "outputs" / "trading" / "strategy_promotion" / "release_state.json"


def _read_state(path: Path, *, fields: tuple[str, ...]) -> dict:
    if not path.is_file():
        return {"schema_version": PROMOTION_SCHEMA_VERSION, **{key: ({} if key == "approvals" else [] if key == "history" else None) for key in fields}}
    state = load_json_strict(path)
    if not isinstance(state, dict) or state.get("schema_version") != PROMOTION_SCHEMA_VERSION:
        raise ValueError(f"Promotion state schema 不合法: {path.name}")
    if state.get("state_sha256") != canonical_json_sha256({k: v for k, v in state.items() if k != "state_sha256"}):
        raise ValueError(f"Promotion state fingerprint 不一致: {path.name}")
    return state


def _write_state(path: Path, state: dict) -> None:
    data = dict(state)
    data.pop("state_sha256", None)
    data["state_sha256"] = canonical_json_sha256(data)
    atomic_write_json(path, data)


def load_promotion_approvals(root: str | Path) -> dict:
    return _read_state(_approvals_path(root), fields=("approvals", "history"))


def _valid_approval(root: str | Path, arm: dict) -> dict | None:
    approval = (load_promotion_approvals(root).get("approvals") or {}).get(arm["arm_id"])
    if not isinstance(approval, dict):
        return None
    if (approval.get("rule_sha256") != arm["rule_sha256"]
            or approval.get("suite_id") != arm["suite_id"]
            or approval.get("status") != "APPROVED"):
        return None
    evidence = Path(root).resolve() / str(approval.get("evidence_path") or "")
    if not evidence.resolve().is_relative_to(Path(root).resolve()) or not evidence.is_file():
        return None
    if compute_file_sha256(evidence) != str(approval.get("evidence_sha256") or ""):
        return None
    manifest = Path(root).resolve() / str(approval.get("manifest_path") or "")
    if not manifest.resolve().is_relative_to(Path(root).resolve()) or not manifest.is_file():
        return None
    if compute_file_sha256(manifest) != str(approval.get("manifest_sha256") or ""):
        return None
    return dict(approval)


def get_approved_rule(root: str | Path, arm_id: str) -> dict:
    arm = current_strategy_rule(arm_id)
    approved = _valid_approval(root, arm)
    if approved is None:
        raise RuntimeError(f"{arm_id} 尚未獲 Research 有效核准；或 Compare Suite／證據已改變")
    return approved


def _rolling_evidence(root: str | Path, arm: dict) -> tuple[Path, str, Path, str]:
    settings = get_strategy_comparison_settings(arm["rolling_profile_id"])
    relative_root = Path(str(settings.output_root))
    project_root = Path(root).resolve()
    evidence = project_root / relative_root / "latest" / "strategy_comparison.json"
    if not evidence.is_file():
        raise FileNotFoundError("Research 尚缺目前 Compare Suite 的 Rolling OOS 結果；請先執行 [3] → [2]")
    payload = load_json_strict(evidence)
    if not isinstance(payload, dict) or payload.get("status") != "COMPLETED":
        raise RuntimeError("Research Rolling OOS 結果未完成")
    if str((payload.get("settings") or {}).get("suite_id") or "") != arm["suite_id"]:
        raise RuntimeError("Research 結果 Compare Suite 與目前選單不同；不可核准")
    if arm["arm_id"] not in (payload.get("scenarios") or {}):
        raise RuntimeError("Research 結果缺少所選 arm；不可核准")
    if not str(payload.get("config_fingerprint") or ""):
        raise RuntimeError("Research 結果缺少 config fingerprint；不可核准")
    manifest = evidence.with_name("manifest.json")
    if not manifest.is_file():
        raise FileNotFoundError("Research Rolling OOS 缺少正式 manifest；不可核准")
    manifest_payload = load_json_strict(manifest)
    if str(manifest_payload.get("config_fingerprint") or "") != str(payload["config_fingerprint"]):
        raise RuntimeError("Research 結果與 manifest fingerprint 不一致")
    current_arms = [candidate.as_dict() for candidate in settings.enabled_arms]
    if manifest_payload.get("enabled_arms") != current_arms:
        raise RuntimeError("Research manifest strategy arms differ from current Compare Suite")
    if sorted((payload.get("scenarios") or {}).keys()) != sorted(candidate["arm_id"] for candidate in current_arms):
        raise RuntimeError("Research result scenarios differ from current Compare Suite")
    sha = compute_file_sha256(evidence)
    # Pin an immutable run artifact, not the mutable "latest" alias.  Promotion
    # evidence must survive subsequent Research runs without silently drifting.
    resolved = evidence.resolve()
    if resolved.parent.name == "latest":
        runs_dir = project_root / relative_root / "runs"
        if runs_dir.is_dir():
            for candidate in sorted(runs_dir.glob("*/strategy_comparison.json"), reverse=True):
                if candidate.stat().st_size == evidence.stat().st_size and compute_file_sha256(candidate) == sha:
                    resolved = candidate.resolve()
                    break
    if resolved.parent.name == "latest":
        raise RuntimeError("Research latest 證據尚無可封存 run；禁止只引用可變 latest")
    if not resolved.is_relative_to(project_root):
        raise RuntimeError("Research run 證據不可超出專案目錄")
    resolved_manifest = resolved.with_name("manifest.json")
    if not resolved_manifest.is_file() or compute_file_sha256(resolved_manifest) != compute_file_sha256(manifest):
        raise RuntimeError("Research pinned manifest differs from latest manifest")
    return resolved, sha, resolved_manifest, compute_file_sha256(resolved_manifest)


def approve_current_strategy_rule(root: str | Path, arm_id: str) -> dict:
    arm = current_strategy_rule(arm_id)
    evidence, evidence_sha, manifest, manifest_sha = _rolling_evidence(root, arm)
    relative = evidence.relative_to(Path(root).resolve()).as_posix()
    result = {
        "arm_id": arm["arm_id"], "suite_id": arm["suite_id"],
        "rule_sha256": arm["rule_sha256"],
        "evidence_path": relative, "evidence_sha256": evidence_sha,
        "manifest_path": manifest.relative_to(Path(root).resolve()).as_posix(),
        "manifest_sha256": manifest_sha,
        "status": "APPROVED", "approved_at": get_taipei_now().isoformat(),
    }
    state = load_promotion_approvals(root)
    state.setdefault("approvals", {})[arm_id] = result
    state.setdefault("history", []).append(result)
    _write_state(_approvals_path(root), state)
    return result


def load_trading_release_state(root: str | Path) -> dict:
    return _read_state(_release_path(root), fields=("active", "pending", "history"))


def get_trading_release(root: str | Path, *, purpose: str = "active", require_current: bool = False) -> dict | None:
    if purpose not in {"active", "training"}:
        raise ValueError(f"未知 Trading release purpose: {purpose}")
    state = load_trading_release_state(root)
    release = state.get("pending") if purpose == "training" and state.get("pending") else state.get("active")
    if release is None:
        return None
    if not isinstance(release, dict):
        raise ValueError("Trading release 必須是 object")
    if not str(release.get("release_id") or "").isalnum() or len(str(release["release_id"])) != 32:
        raise RuntimeError("Trading Release ID 不合法")
    for required in ("arm_id", "suite_id", "rule_sha256", "training_profile_id"):
        if not str(release.get(required) or ""):
            raise RuntimeError(f"Trading Release 缺少 {required}")
    # Frozen positions and read-only charts must remain readable even after a
    # suite migration.  Only NEW entries / activation enforce current approval.
    if require_current:
        arm = current_strategy_rule(str(release["arm_id"]))
        get_approved_rule(root, arm["arm_id"])
        if (str(release["rule_sha256"]) != arm["rule_sha256"]
                or str(release["suite_id"]) != arm["suite_id"]
                or str(release["training_profile_id"]) != arm["training_profile_id"]):
            raise RuntimeError("Trading Release 與目前 Compare Suite canonical rule 不一致；禁止新增交易")
    return dict(release)


def stage_trading_strategy_rule(root: str | Path, arm_id: str) -> dict:
    arm = current_strategy_rule(arm_id)
    get_approved_rule(root, arm_id)
    if not arm["trading_supported"]:
        raise RuntimeError(f"{arm_id} 目前無法啟用 Trading: {'; '.join(arm['trading_blockers'])}")
    state = load_trading_release_state(root)
    release = {
        "release_id": uuid4().hex,
        "suite_id": arm["suite_id"], "arm_id": arm_id,
        "rule_sha256": arm["rule_sha256"],
        "training_profile_id": arm["training_profile_id"],
        "status": "PENDING_PARAMS", "staged_at": get_taipei_now().isoformat(),
    }
    state["pending"] = release
    state.setdefault("history", []).append({**release, "action": "STAGED"})
    _write_state(_release_path(root), state)
    return release


def commit_trading_release(root: str | Path, pending_release_id: str, *, selected_params_sha256: str, binding_sha256: str, manifest_sha256: str) -> dict:
    state = load_trading_release_state(root)
    pending = get_trading_release(root, purpose="training", require_current=True)
    if not state.get("pending") or not pending or pending.get("release_id") != pending_release_id:
        raise RuntimeError("Trading 待啟用策略已變更；不得啟用過期 Params")
    if not selected_params_sha256 or not binding_sha256 or not manifest_sha256:
        raise ValueError("Trading Release 缺少 Params / binding / manifest SHA256")
    result = {
        **pending, "status": "ACTIVE", "activated_at": get_taipei_now().isoformat(),
        "selected_params_sha256": selected_params_sha256,
        "binding_sha256_at_activation": binding_sha256,
        "param_manifest_sha256": manifest_sha256,
    }
    state["active"] = result
    state["pending"] = None
    state.setdefault("history", []).append({**result, "action": "ACTIVATED"})
    _write_state(_release_path(root), state)
    return result


def record_active_release_fitting(
    root: str | Path, *, expected_release_id: str,
    selected_params_sha256: str, binding_sha256: str, manifest_sha256: str,
) -> None:
    """Rebind a fitting instance under an unchanged canonical Strategy Rule.

    Every legitimate retrain/rebind leaves an append-only audit entry.  A staged
    fitting run can never alter the active release's selected Params.
    """
    state = load_trading_release_state(root)
    active = state.get("active")
    if not isinstance(active, dict) or active.get("release_id") != expected_release_id:
        raise RuntimeError("ACTIVE Release 已改變；禁止提交過期 fitting 產物")
    get_trading_release(root, purpose="active", require_current=True)
    updated = dict(active)
    updated["selected_params_sha256"] = str(selected_params_sha256)
    updated["binding_sha256_at_activation"] = str(binding_sha256)
    updated["param_manifest_sha256"] = str(manifest_sha256)
    state["active"] = updated
    state.setdefault("history", []).append({
        "action": "PARAMS_BOUND", "release_id": expected_release_id,
        "selected_params_sha256": str(selected_params_sha256),
        "binding_sha256": str(binding_sha256),
        "param_manifest_sha256": str(manifest_sha256),
        "at": get_taipei_now().isoformat(),
    })
    _write_state(_release_path(root), state)


__all__ = [
    "current_strategy_rules", "current_strategy_rule", "load_promotion_approvals",
    "get_approved_rule", "approve_current_strategy_rule", "load_trading_release_state",
    "get_trading_release", "stage_trading_strategy_rule", "commit_trading_release",
    "record_active_release_fitting",
]

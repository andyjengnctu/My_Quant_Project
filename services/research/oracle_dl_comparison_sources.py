"""Read-only canonical source bindings for matched Oracle / PIT DL strategy evidence.

No training, promotion, report publication, artifact mutation or alternative
candidate engine lives here. The registered Strategy Compare / Audit sources
alone determine the eligible comparison pairs and benchmark seeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.audit_policy import get_reusable_audit_definitions
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.file_integrity import (
    atomic_write_json, atomic_write_text, canonical_json_sha256,
    load_json_object_or_none, require_sha256_hex,
)
from core.strategy_compare_policy import (
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_settings,
)
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.ranking_score_store import (
    load_selection_point_in_time_score_source_contract,
    load_selection_point_in_time_score_table_from_path,
)

# Scientific *equivalence* of a daily component target and an Oracle recipe.
# Multi-head MFE/Safety scores are deliberately not interchangeable with
# the raw MFE geometry or a learned fused prediction.
MATCHED_RESEARCH_CONTRACT_ID = "oracle_dl_daily_matched_v1"

_DAILY_COMPONENT_TO_ORACLE_RECIPE = {
    "daily_terminal_mae_gated_t40": "terminal_mae_gate",
    "daily_full_horizon_mae_gated_mfe_t40": "mfe_mae_gate",
}


def resolve_matched_oracle_dl_pairs(profile_id: str) -> tuple[dict[str, str], ...]:
    """Resolve active Exact-K/R0 arm↔Oracle matches from the two registries."""
    settings = get_strategy_comparison_settings(str(profile_id))
    definitions = (
        definition for definition in get_reusable_audit_definitions("breakout_quality", enabled_only=True)
        if definition.report_id == "target_oracle_romd"
        and str(definition.source.get("strategy_profile_id")) == str(settings.profile_id)
    )
    definition = next(definitions, None)
    if definition is None:
        return ()
    recipes: dict[str, list[str]] = {}
    for comparison in definition.source.get("oracle_comparisons") or ():
        if comparison.get("enabled", True) and comparison.get("selector") == "exact":
            name = str(comparison.get("target") or "")
            recipe = str(comparison.get("score") or "")
            if recipe and name:
                recipes.setdefault(recipe, []).append(name)
    results = []
    for arm in settings.enabled_arms:
        if not arm.dl_enabled or not arm.dl_id:
            continue
        if (arm.param_source != settings.arms[str(definition.source["baseline_arm_id"])].param_source
                or arm.param_policy != settings.arms[str(definition.source["baseline_arm_id"])].param_policy
                or arm.rule_policy != "all_off"):
            continue
        if (str(arm.dl_runtime_mode) != str(settings.arms[str(definition.source["exact_arm_id"])].dl_runtime_mode)
                or dict(arm.dl_runtime_options or {}) != dict(settings.arms[str(definition.source["exact_arm_id"])].dl_runtime_options or {})):
            continue
        source = settings.dl_sources[arm.dl_id]
        recipe = get_continuous_ranker_execution_recipe(source.experiment_profile)
        oracle_recipe = _DAILY_COMPONENT_TO_ORACLE_RECIPE.get(str(recipe.target_policy.contract_kind))
        if oracle_recipe is None:
            continue
        matches = recipes.get(oracle_recipe, ())
        if len(matches) != 1:
            raise ValueError(f"Oracle目標 {oracle_recipe} 必須唯一且啟用，不能推測配對")
        results.append({
            "arm_id": str(arm.arm_id), "dl_id": str(arm.dl_id),
            "experiment_profile": str(source.experiment_profile),
            "model_architecture": str(source.model_architecture),
            "oracle_target": str(matches[0]),
            "continuous_target_id": str(recipe.continuous_target_id),
            "selector": "exact", "score_source": "selection_point_in_time",
        })
    return tuple(results)


def _registered_oracle_definition(profile_id: str):
    matches = [definition for definition in get_reusable_audit_definitions("breakout_quality", enabled_only=True)
               if definition.report_id == "target_oracle_romd"
               and str(definition.source.get("strategy_profile_id")) == str(profile_id)]
    if len(matches) != 1:
        raise ValueError("未找到對應且唯一的正式Oracle定義")
    return matches[0]


def _validated_robustness_model_root(project_root: Path, profile_id: str) -> Path:
    """Resolve an isolated-seed worktree only from a completed formal manifest.

    Latest is a pointer, not scientific authority. Nothing is inferred by
    listing filenames, and pruned score files are never reconstructed from RoMD.
    """
    root = Path(project_root).resolve()
    cfg = get_strategy_multi_seed_robustness_settings(str(profile_id))
    output_root = (root / str(cfg.output_root)).resolve()
    pointer = load_json_object_or_none(output_root / "latest.json")
    if not isinstance(pointer, dict):
        raise FileNotFoundError("缺少正式Robustness latest.json；不可猜測四Seed分數路徑")
    fingerprint = str(pointer.get("fingerprint") or "").strip().lower()
    if not fingerprint or len(fingerprint) != 16 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Robustness latest fingerprint非法")
    run_root = (output_root / fingerprint).resolve()
    if run_root.parent != output_root:
        raise ValueError("Robustness latest輸出路徑逃逸")
    manifest = load_json_object_or_none(run_root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("status") != "COMPLETED":
        raise ValueError("Robustness尚無已提交COMPLETED manifest")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Robustness缺少核准的Scientific Contract")
    if (str(contract.get("fingerprint")) != fingerprint
            or str(contract.get("profile_id")) != str(profile_id)
            or str(contract.get("robustness_id")) != str(cfg.robustness_id)
            or str(contract.get("benchmark_id")) != str(cfg.benchmark_id)
            or tuple(int(s) for s in (contract.get("resolved_seeds") or ())) != tuple(cfg.resolved_seeds)):
        raise ValueError("Robustness最新結果與目前profile／benchmark／四Seed身分不一致")
    model_fingerprint = str(contract.get("model_artifact_fingerprint") or fingerprint).strip().lower()
    if len(model_fingerprint) != 16 or any(c not in "0123456789abcdef" for c in model_fingerprint):
        raise ValueError("Robustness模型身分指紋非法")
    return (root / str(cfg.model_work_root) / model_fingerprint).resolve()


def load_verified_seed_pit_scores(
    *, project_root: Path, profile_id: str, match: Mapping[str, str],
    model_root_override: Path | None = None,
) -> dict[int, dict[str, Any]]:
    """Load 4 actual, different seed-specific PIT predictions using official loaders.

    Caller cannot replace seed N with a global PIT score: each manifest is
    validated against its requested seed, SHA, provenance and model identity.
    """
    root = Path(project_root).resolve()
    cfg = get_strategy_multi_seed_robustness_settings(str(profile_id))
    oracle = _registered_oracle_definition(profile_id)
    model_root = (Path(model_root_override).resolve() if model_root_override is not None
                  else _validated_robustness_model_root(root, profile_id))
    if not cfg.resolved_seeds or len(set(cfg.resolved_seeds)) != len(cfg.resolved_seeds):
        raise ValueError("四Seed契約缺失或重複")
    dl_id = str(match["dl_id"])
    source = get_strategy_comparison_settings(profile_id).dl_sources[dl_id]
    verified: dict[int, dict[str, Any]] = {}
    for seed in cfg.resolved_seeds:
        model_dir = model_root / f"{dl_id}__seed_{int(seed)}" / "model"
        try:
            contract = load_selection_point_in_time_score_source_contract(
                root, str(oracle.source["filter_id"]), str(source.model_architecture),
                str(source.experiment_profile), point_in_time_dir_override=model_dir,
                expected_seed=int(seed), allow_stale_source=False,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"P1-P3 BLOCKED: {dl_id} seed={seed} PIT SCORE未保留或驗證失敗：{exc}; "
                "不可拿完成的策略RoMD或其他Seed分數替代"
            ) from exc
        if str(contract.continuous_target_id) != str(match["continuous_target_id"]):
            raise ValueError(f"seed={seed} PIT Target與Oracle配對身分不一致")
        score_hash = compute_file_sha256(contract.score_path)
        require_sha256_hex(score_hash, field="PIT score SHA256")
        table = load_selection_point_in_time_score_table_from_path(
            str(contract.score_path), manifest_path=str(contract.manifest_path),
            experiment_profile=str(source.experiment_profile),
            expected_score_sha256=score_hash,
        )
        if not table.index.is_unique:
            raise ValueError(f"seed={seed} PIT股票日分數索引重複")
        verified[int(seed)] = {
            "table": table, "score_sha256": score_hash,
            "manifest_sha256": compute_file_sha256(contract.manifest_path),
            "available_from": str(contract.available_from),
            "available_through": str(contract.available_through),
            "seed": int(contract.seed),
        }
    return verified


def collect_matched_oracle_dl_readiness(
    *, project_root: Path, profile_id: str,
) -> dict[str, Any]:
    """Fast, read-only P1/P2/P3 source preflight; never starts Replay/FIT."""
    root = Path(project_root).resolve()
    pairs = resolve_matched_oracle_dl_pairs(profile_id)
    if not pairs:
        return {"status": "NOT_APPLICABLE", "pairs": (), "reason": "尚無同Target及Selector的正式Oracle／DL配對"}
    try:
        definition = _registered_oracle_definition(profile_id)
        cfg = get_strategy_multi_seed_robustness_settings(profile_id)
        from services.audit.target_oracle_romd import _robustness_artifacts
        _robustness_artifacts(definition, root)
        model_root = _validated_robustness_model_root(root, profile_id)
    except (OSError, ValueError, RuntimeError) as exc:
        return {"status": "BLOCKED", "reason": str(exc), "pairs": tuple({**pair, "status": "BLOCKED"} for pair in pairs)}
    rows = []
    for pair in pairs:
        try:
            verified = load_verified_seed_pit_scores(
                project_root=root, profile_id=profile_id, match=pair,
                model_root_override=model_root,
            )
            if set(verified) != set(cfg.resolved_seeds):
                raise ValueError("PIT Seed集合未完整對齊")
            rows.append({**pair, "status": "SOURCE_READY", "seed_count": len(verified)})
        except (OSError, ValueError, RuntimeError) as exc:
            rows.append({**pair, "status": "BLOCKED", "reason": str(exc)})
    return {
        "status": "SOURCES_READY" if all(row["status"] == "SOURCE_READY" for row in rows) else "BLOCKED",
        "pairs": tuple(rows), "reason": "" if all(row["status"] == "SOURCE_READY" for row in rows)
        else "至少一組模型／Seed PIT Score 不完整；不允許策略對照",
        "replay_status": "NOT_EXECUTED",
        "research_window": {"start": str(get_strategy_comparison_settings(profile_id).start_date),
                            "end": "from_verified_40d_oracle_cohort"},
    }



def _matched_research_output_root(root: Path, settings, arm_id: str) -> Path:
    # Dedicated historical research evidence beneath the existing Strategy Compare domain.
    # No Standard SOP renderer, second Replay producer, or latest pointer.
    safe_arm = str(arm_id)
    if not safe_arm or not safe_arm.replace("_", "").isalnum():
        raise ValueError("matched research arm identity is not path-safe")
    return (Path(root).resolve() / str(settings.output_root) /
            "oracle_dl_matched_research" / safe_arm)


def _read_completed_matched_evidence(directory: Path, identity: dict) -> dict | None:
    """A same-identity completed result is reusable only with intact published bytes."""
    from core.file_integrity import load_json_strict
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        if directory.exists():
            raise RuntimeError("P1-P3相同研究身分存在未完成工件；保留證據並BLOCK，不覆寫")
        return None
    manifest = load_json_strict(manifest_path)
    if (not isinstance(manifest, dict) or manifest.get("status") != "COMPLETED"
            or manifest.get("identity") != identity):
        raise RuntimeError("P1-P3完成工件身分衝突；不得REUSE或覆寫")
    for filename, expected in (("result.json", "result_sha256"),
                               ("result.md", "markdown_sha256")):
        path = directory / filename
        if not path.is_file() or compute_file_sha256(path) != manifest.get(expected):
            raise RuntimeError("P1-P3已完成工件內容與Hash不一致；禁止覆寫")
    payload = load_json_strict(directory / "result.json")
    if (not isinstance(payload, dict) or payload.get("identity") != identity
            or not isinstance(payload.get("result"), dict)
            or payload["result"].get("status") != "COMPARABLE"
            or payload["result"].get("scientific_contract_id") != MATCHED_RESEARCH_CONTRACT_ID):
        raise RuntimeError("P1-P3歷史研究結果契約不一致")
    return dict(payload["result"])


def _matched_research_readable(result: Mapping[str, Any]) -> str:
    """Compact read-only research evidence; not a Standard SOP report."""
    rows = list(result.get("same_seed") or ())
    if not rows:
        raise ValueError("P1-P3配對結果缺少各Seed RoMD")
    lines = ["# Oracle / PIT DL matched research evidence (NON-PIT Oracle)", "",
             f"- Status: {result['status']}",
             f"- Contract: {result['scientific_contract_id']}",
             f"- Research arm: {result['research_arm']}",
             f"- Oracle Target: {result['oracle_target']}",
             f"- Evaluation window: {result['evaluation_period']['start']} ～ {result['evaluation_period']['end']}",
             "- Score: same daily information date / shared feature-history eligibility / canonical percentile",
             "- Same four P16 seeds / Exact K-R0 / broker-cash portfolio rules",
             "- This is controlled hindsight research, never PIT-legal Oracle trading evidence.", "",
             "| Seed | No-DL RoMD | Oracle RoMD | PIT DL RoMD | Oracle-DL |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {int(row['seed'])} | {float(row['no_dl_romd']):.4f} | "
                     f"{float(row['oracle_romd']):.4f} | {float(row['dl_romd']):.4f} | "
                     f"{float(row['oracle_minus_dl']):+.4f} |")
    lines.extend(["", f"- Mean Oracle-DL RoMD gap: {float(result['oracle_minus_dl_romd_mean']):+.4f}",
                  "- Raw identity, score coverage and per-seed provenance are retained in result.json.", ""])
    return "\n".join(lines)


def _publish_completed_matched_evidence(directory: Path, identity: dict, result: dict) -> None:
    """Create an immutable same-identity result; never replace a completed history."""
    # No directory is created until all expensive and scientific checks have passed.
    if directory.exists():
        existing = _read_completed_matched_evidence(directory, identity)
        if existing is not None:
            return
    directory.mkdir(parents=True, exist_ok=False)
    payload = {"identity": identity, "result": result}
    try:
        atomic_write_json(directory / "result.json", payload)
        atomic_write_text(directory / "result.md", _matched_research_readable(result))
        atomic_write_json(directory / "manifest.json", {
            "status": "COMPLETED", "identity": identity,
            "result_sha256": compute_file_sha256(directory / "result.json"),
            "markdown_sha256": compute_file_sha256(directory / "result.md"),
        })
    except Exception:
        # Partial state intentionally remains for operator inspection; never silently
        # replace an incomplete or identity-conflicted research result.
        raise


def render_matched_research_plan(readiness: Mapping[str, Any]) -> str:
    """Show the same planning source used by the actual Research dispatcher."""
    pairs = tuple(readiness.get("pairs") or ())
    if not pairs:
        return ""
    from core.report_style import styled_workflow_status
    lines = ["[P1-P3 | Oracle / PIT DL 配對｜Research-only]", "正式策略比較維持原有RUN/REUSE；以下不會改寫正式報表："]
    for pair in pairs:
        action = "BLOCKED" if pair.get("status") == "BLOCKED" else "RUN"
        suffix = (str(pair.get("reason") or readiness.get("reason") or "")
                  if action == "BLOCKED" else "RUN前核對完整身份；相同已完成工件將REUSE")
        lines.append(f"- {styled_workflow_status(action)} {pair['arm_id']} ↔ {pair['oracle_target']} | {suffix}")
    return "\n".join(lines)


def run_matched_research_from_strategy_compare(
    *, project_root: Path, profile_id: str, readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One optional same-profile research action; missing sources don't block Standard SOP.

    Execution is authorized by the current compare-suite arm-to-target mapping;
    score/benchmark identity conflicts are always explicit BLOCKED, never fallback.
    """
    state = dict(readiness) if readiness is not None else collect_matched_oracle_dl_readiness(
        project_root=project_root, profile_id=profile_id,
    )
    outcomes: list[dict[str, str]] = []
    for pair in state.get("pairs", ()):
        arm = str(pair["arm_id"])
        target = str(pair["oracle_target"])
        if pair.get("status") == "BLOCKED":
            reason = str(pair.get("reason") or state.get("reason") or "P1-P3來源缺失")
            print(f"[P1-P3 BLOCKED] {arm} ↔ {target} | {reason}")
            outcomes.append({"arm_id": arm, "oracle_target": target, "action": "BLOCKED", "reason": reason})
            continue
        try:
            result = run_verified_matched_oracle_dl_comparison(
                project_root=project_root, profile_id=profile_id, arm_id=arm,
                quiet=True, persist_evidence=True,
            )
            action = str(result.get("artifact_action") or "RUN")
            print(f"[P1-P3 {action}] {arm} ↔ {target} | Oracle-DL ΔRoMD={result['oracle_minus_dl_romd_mean']:+.4f}"
                  f" | {result.get('evidence_path', '研究結果未發佈')}")
            outcomes.append({"arm_id": arm, "oracle_target": target, "action": action})
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[P1-P3 BLOCKED] {arm} ↔ {target} | {reason}")
            outcomes.append({"arm_id": arm, "oracle_target": target, "action": "BLOCKED", "reason": reason})
    return {"status": "COMPLETED" if outcomes and all(x["action"] != "BLOCKED" for x in outcomes)
            else "BLOCKED", "arms": tuple(outcomes)}

def run_verified_matched_oracle_dl_comparison(
    *, project_root: Path, profile_id: str, arm_id: str,
    quiet: bool = True, persist_evidence: bool = False,
) -> dict[str, Any]:
    """Replay a READY same-target four-seed No-DL/Oracle/PIT-DL matched case.

    This is deliberately an explicit research service action. The existing
    Strategy Compare/Robustness entry and its official reports are not mutated,
    and missing retained scores BLOCK instead of invoking model training.
    """
    root = Path(project_root).resolve()
    matches = {pair["arm_id"]: pair for pair in resolve_matched_oracle_dl_pairs(profile_id)}
    if arm_id not in matches:
        raise ValueError("不存在同Target同Selector的正式DL/Oracle配對")
    pair = matches[arm_id]
    definition = _registered_oracle_definition(profile_id)
    settings = get_strategy_comparison_settings(profile_id)
    robustness = get_strategy_multi_seed_robustness_settings(profile_id)
    if settings.rotation != "off" or settings.dataset != "full":
        raise ValueError("只允許已核准的Full Dataset / Rotation Off同口徑Replay")
    # Fast fail on absent canonical model/P16 artifacts, before loading the
    # full future price truth or preparing the expensive Portfolio context.
    from services.audit.target_oracle_romd import (
        _robustness_artifacts, _prepare_oracle_parameters, build_oracle_score_tables,
    )
    _robustness, parameter_paths = _robustness_artifacts(definition, root)
    verified = load_verified_seed_pit_scores(
        project_root=root, profile_id=profile_id, match=pair,
    )
    if set(verified) != set(int(s) for s in robustness.resolved_seeds):
        raise ValueError("四Seed PIT分數尚未具備獨立身份驗證")
    # Canonical Oracle target calculator provides the full matured stock-day
    # truth; only the selected registered target enters this experiment.
    oracle_scores, evidence = build_oracle_score_tables(
        definition, project_root=root, cutoff=str(settings.end_date),
    )
    first = str(settings.start_date)
    last = min(str(settings.end_date), str(evidence["last_common_day"]))
    warmup = str(dict(evidence.get("market_truth") or {}).get("score_start") or "")
    if not warmup or warmup > first or last <= first:
        raise ValueError("共同成熟窗與期初延續候選資訊窗不完整")
    if pair["oracle_target"] not in oracle_scores:
        raise ValueError("Oracle Registry配對Target沒有正式Ground Truth工件")
    oracle_map = oracle_scores[pair["oracle_target"]]
    oracle_table = pd.DataFrame(
        [(ticker, date, value) for (ticker, date), value in oracle_map.items()],
        columns=["ticker", "date", "score"],
    ).set_index(["ticker", "date"]).sort_index()

    # Independent feature eligibility is the canonical daily-universal
    # sample provider's *entire* stock-day inventory (train-valid + inference),
    # NOT its target_valid mask, Oracle validity, or PIT-score intersection.
    from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
    bundle = load_profile_continuous_ranker_data(
        filter_id=str(definition.source["filter_id"]),
        model_architecture=str(pair["model_architecture"]),
        experiment_profile=str(pair["experiment_profile"]),
        preload_feature_bank=False, allow_stale_source=False, project_root=root,
    )
    group = pd.DataFrame(bundle.group_table)
    if not {"ticker", "date"}.issubset(group) or len(group) != len(bundle.target_valid):
        raise ValueError("feature-history eligibility未通過正式Dataset結構驗證")
    group_days = pd.to_datetime(group["date"], errors="raise").dt.strftime("%Y-%m-%d")
    eligible_frame = pd.DataFrame({
        "ticker": group["ticker"].astype(str).to_numpy(),
        "date": group_days.to_numpy(),
    })
    eligible_frame = eligible_frame.loc[
        (eligible_frame["date"] >= warmup) & (eligible_frame["date"] <= last)
    ]
    if eligible_frame.empty or eligible_frame.duplicated(["ticker", "date"]).any():
        raise ValueError("獨立feature-history stock-day母體不存在或重複")
    cohort = pd.MultiIndex.from_frame(eligible_frame).set_names(["ticker", "date"])
    # The comparison uses the official P16 evaluation view and controlled
    # No-DL/score-ranking parameter pair, not another param calculator.
    baseline_arm = settings.arms[str(definition.source["baseline_arm_id"])]
    prepared, _param_evidence = _prepare_oracle_parameters(
        definition, root=root, parameter_paths=parameter_paths,
        start=first, end=last, base_arm=baseline_arm,
    )
    from core.dataset_profiles import get_dataset_dir
    from services.audit.oracle_replay_artifacts import price_inventory_sha256
    from config.broker_accounting import BROKER_ACCOUNTING_POLICY_ID
    from services.research.oracle_dl_comparison_contract import run_matched_oracle_dl_replays
    from hashlib import sha256

    # PIT manifest proves the producer's eligibility contract; the Oracle
    # provider separately proves the concrete future-score values.
    truth_hash = sha256()
    for (ticker, date), score in sorted(oracle_map.items()):
        truth_hash.update(f"{ticker}\0{date}\0{float(score):.17g}\n".encode("utf-8"))
    panels = {}
    for seed, param_sha, kind, baseline, scored in prepared:
        item = verified.get(int(seed))
        if item is None:
            raise ValueError(f"seed={seed}無對應PIT Score來源")
        if item["available_from"] > warmup or item["available_through"] < last:
            raise ValueError(f"seed={seed} PIT Score不覆蓋含期初延續候選的共同期間")
        model_table = item["table"]
        if "breakout_quality_score" not in model_table:
            raise ValueError(f"seed={seed} PIT Score缺少正式分數欄位")
        panels[int(seed)] = {
            "oracle": oracle_table, "dl": model_table[["breakout_quality_score"]].rename(
                columns={"breakout_quality_score": "score"}),
            "eligible": cohort, "param_sha256": str(param_sha), "kind": str(kind),
            "baseline_params": baseline, "scored_params": scored,
            "pit_score_sha256": item["score_sha256"],
            "oracle_truth_sha256": truth_hash.hexdigest(),
            "eligibility_source": "feature_history_only",
            "eligibility_manifest_sha256": item["manifest_sha256"],
        }
    exact = settings.arms[str(definition.source["exact_arm_id"])]
    dataset_sha = price_inventory_sha256(root, settings.dataset)
    # A completed result can be reused only after *current* source, dataset,
    # feature eligibility, P16, PIT and future-truth identities are verified.
    digest = canonical_json_sha256({
        "cohort": sorted((str(t), str(d)) for t, d in cohort.tolist()),
    })
    identity = {
        "scientific_contract_id": MATCHED_RESEARCH_CONTRACT_ID,
        "profile_id": str(profile_id), "research_arm": str(arm_id),
        "oracle_target": str(pair["oracle_target"]),
        "model_profile": str(pair["experiment_profile"]),
        "continuous_target_id": str(pair["continuous_target_id"]),
        "period": {"start": first, "end": last, "score_start": warmup},
        "dataset_sha256": dataset_sha,
        "broker_policy": str(BROKER_ACCOUNTING_POLICY_ID),
        "selector": str(exact.dl_runtime_mode),
        "ranking_options": dict(exact.dl_runtime_options or {}),
        "max_positions": int(settings.max_positions), "rotation": False,
        "alignment": "daily_average_zero_based_v1",
        "eligible_cohort_sha256": digest,
        "future_truth_sha256": truth_hash.hexdigest(),
        "seed_sources": [{
            "seed": int(seed), "param_sha256": str(panels[seed]["param_sha256"]),
            "pit_score_sha256": str(panels[seed]["pit_score_sha256"]),
            "pit_manifest_sha256": str(panels[seed]["eligibility_manifest_sha256"]),
        } for seed in sorted(panels)],
    }
    output_directory = (_matched_research_output_root(root, settings, arm_id) /
                        canonical_json_sha256(identity, length=24))
    if persist_evidence:
        cached = _read_completed_matched_evidence(output_directory, identity)
        if cached is not None:
            cached["artifact_action"] = "REUSE"
            return cached
    result = dict(run_matched_oracle_dl_replays(
        seed_panels=panels, start_date=first, end_date=last,
        score_start_date=warmup,
        data_dir=str(get_dataset_dir(str(root), settings.dataset)),
        dataset_sha256=dataset_sha,
        broker_policy=str(BROKER_ACCOUNTING_POLICY_ID),
        selector_policy=str(exact.dl_runtime_mode),
        ranking_options=dict(exact.dl_runtime_options or {}),
        max_positions=int(settings.max_positions), rotation=False, quiet=quiet,
    ))
    result["research_arm"] = str(arm_id)
    result["oracle_target"] = pair["oracle_target"]
    result["scientific_contract_id"] = MATCHED_RESEARCH_CONTRACT_ID
    result["source_provenance"] = tuple({
        "seed": seed,
        "pit_score_sha256": panels[seed]["pit_score_sha256"],
        "pit_manifest_sha256": panels[seed]["eligibility_manifest_sha256"],
        "oracle_truth_sha256": panels[seed]["oracle_truth_sha256"],
        "p16_evaluation_sha256": panels[seed]["param_sha256"],
    } for seed in sorted(panels))
    result["scientific_status"] = "RESEARCH_NON_PIT_ORACLE_VS_PIT_DL"
    result["source_verified_seeds"] = len(panels)
    result["read_only"] = True
    if persist_evidence:
        if (result.get("status") != "COMPARABLE"
                or int(result.get("matched_seeds", -1)) != len(panels)):
            raise ValueError("P1-P3必須完成全部四Seed一致性驗證後才能發布研究證據")
        result["evidence_path"] = str((output_directory / "result.md").relative_to(root)).replace("\\", "/")
        _publish_completed_matched_evidence(output_directory, identity, result)
        result["artifact_action"] = "RUN"
    return result

"""One-time CO ranking audit; read existing sources, delegate canonical replay.

The retained C IDs identify historical replay contracts, not current candidates.
This handler cannot fit models, optimize parameters, or publish formal comparisons.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from config.audit import AUDIT_OUTPUT_ROOT
from core.candidate_score_permutation_contract import OPTION, REPEATS, permutation_spec
from core.file_integrity import (
    atomic_write_json, atomic_write_text, canonical_json_sha256,
    compute_observed_file_sha256, exclusive_file_lock,
)
from core.score_permutation_observation import ROMD, evaluate_permutation_control, render_permutation_appendix
from core.strategy_compare_policy import (
    get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
)
from core.strategy_comparison import resolve_strategy_comparison_arm_param_policy


def resolve_audit_settings(definition):
    """Project only registered audit contracts; never change the global roster."""
    if int(definition.dimensions.get("repeats", 0)) != REPEATS:
        raise ValueError("This Audit requires exactly 99 repeats")
    if definition.dimensions.get("include_workflow_seed") is not True:
        raise ValueError("This Audit retains the supplementary workflow seed")
    settings = get_strategy_comparison_settings(definition.source["strategy_profile_id"])
    treatments = tuple(settings.arms[key] for key in definition.source["treatment_arm_ids"])
    if len(treatments) != 2 or len({arm.dl_runtime_mode for arm in treatments}) != 2:
        raise ValueError("Audit requires the two registered Exact/Direct contracts")
    enabled = {definition.source["baseline_arm_id"]}
    for arm in treatments:
        spec = permutation_spec(arm.dl_runtime_options)
        if spec is None:
            raise ValueError("Audit treatment has no permutation contract")
        control = settings.arms[spec["control_arm_id"]]
        options = dict(arm.dl_runtime_options)
        options.pop(OPTION)
        if (arm.dl_id != control.dl_id or options != dict(control.dl_runtime_options or {})
                or arm.dl_runtime_mode != control.dl_runtime_mode
                or arm.param_source != control.param_source):
            raise ValueError("Permutation and control must differ only in score assignment")
        enabled.update((arm.arm_id, control.arm_id))
    if len({arm.dl_id for arm in treatments}) != 1:
        raise ValueError("Audit requires one shared CO source")
    return replace(
        settings,
        arms={key: replace(arm, enabled=key in enabled) for key, arm in settings.arms.items()},
        contrasts={key: replace(item, enabled=False) for key, item in settings.contrasts.items()},
        preparation=replace(settings.preparation, auto_prepare=False),
    )


def _context(definition):
    from core.breakout_quality_policy import get_breakout_quality_workflow_settings
    settings = resolve_audit_settings(definition)
    arms = tuple(settings.arms[key] for key in definition.source["treatment_arm_ids"])
    source = settings.dl_sources[arms[0].dl_id]
    workflow = get_breakout_quality_workflow_settings(experiment_profile=source.experiment_profile)
    benchmark = get_strategy_multi_seed_robustness_settings(settings.profile_id)
    seeds = tuple(int(seed) for seed in benchmark.resolved_seeds)
    if len(set(seeds)) != 4 or int(workflow.seed) in seeds:
        raise ValueError("Audit requires four benchmark seeds distinct from the workflow seed")
    return settings, arms, source, workflow, benchmark, seeds


def collect_status(definition, *, project_root: Path):
    """Presence-only menu check; full source/PIT attestation happens before replay."""
    from services.research.strategy_multi_seed_robustness import _benchmark_param_binding
    from filters.breakout_quality.paths import (
        resolve_model_research_robustness_seed_dirs,
        SELECTION_POINT_IN_TIME_SCORE_FILENAME, SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
        SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME,
    )
    from filters.breakout_quality.strategy_compare_sources import resolve_param_source_path
    from services.research.strategy_compare_execution import selection_pit_mode_paths
    root = Path(project_root).resolve()
    try:
        settings, arms, source, workflow, benchmark, seeds = _context(definition)
        arm = arms[0]
        policy = resolve_strategy_comparison_arm_param_policy(settings, arm)
        paths = list(selection_pit_mode_paths(source, project_root=root).values())
        paths.append(resolve_param_source_path(root, settings, arm.param_source, param_policy=policy))
        cells = [(int(workflow.seed), all(path.is_file() for path in paths))]
        for seed in seeds:
            model, _ = resolve_model_research_robustness_seed_dirs(
                root, source.filter_id, source.model_architecture, source.experiment_profile,
                seed=seed, rolling=True)
            binding = _benchmark_param_binding(settings=settings, robustness=benchmark,
                                               arm=arm, seed=seed, project_root=root)
            paths = [model / name for name in (
                SELECTION_POINT_IN_TIME_SCORE_FILENAME, SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
                SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME)]
            paths.extend((Path(binding["path"]), Path(binding["manifest_path"])))
            cells.append((seed, all(path.is_file() for path in paths)))
        ready = all(present for _, present in cells)
        return {"status": "READY" if ready else "BLOCKED",
                "reason": "來源檔案檢查（執行前再完整驗證）：" + "、".join(
                    f"{seed} {'存在' if present else '缺件'}" for seed, present in cells)
                    + "；每 Seed Exact／Direct 各 99 次；不訓練或最佳化。",
                "source": {"display": "既有 CO PIT 分數＋同 Seed P16；結果只寫入 Audit"}}
    except (KeyError, ValueError, TypeError, OSError) as exc:
        return {"status": "BLOCKED", "reason": str(exc), "source": {}}


def _validated_inputs(definition, root):
    from services.research.strategy_compare_preparation_status import collect_preparation_status
    from services.research.strategy_compare_training import inspect_reusable_model_research_rolling_score
    from services.research.strategy_multi_seed_robustness import (
        _benchmark_param_binding, _inspect_benchmark_strategy_parameter_binding,
    )
    settings, arms, source, workflow, benchmark, seeds = _context(definition)
    status = collect_preparation_status(project_root=root, settings=settings, include_seed_dependencies=False)
    if status["overall_status"] != "READY":
        raise RuntimeError("Audit 來源未通過完整驗證；請由原 Research 入口處理缺件，不自動補建")
    period = status["comparison_period"]
    arm = arms[0]
    files = [root / item["path"] for item in status["dl_sources"][arm.dl_id]["files"].values()]
    params = Path(status["resolved_arm_parameter_paths"][arm.arm_id])
    files.append(params)
    manifest = status["parameters"][arm.param_source].get("identity_manifest_path")
    if manifest:
        files.append(root / manifest)
    jobs = [dict(seed=int(workflow.seed), benchmark=False, params=params, overrides=None,
                 mode=status["resolved_arm_parameter_identities"][arm.arm_id]["evaluation_mode"])]
    for seed in seeds:
        binding = _benchmark_param_binding(settings=settings, robustness=benchmark,
                                           arm=arm, seed=seed, project_root=root)
        state = _inspect_benchmark_strategy_parameter_binding(
            settings=settings, robustness=benchmark, binding=binding, seed=seed,
            comparison_start=period["start"], comparison_end=period["end"], project_root=root)
        if state is None or state["action"] != "REUSE":
            raise RuntimeError(f"Audit P16 seed={seed} 未通過驗證；不自動最佳化")
        artifact, reason = inspect_reusable_model_research_rolling_score(
            project_root=root, source=source, workflow=workflow, seed=seed,
            comparison_start=period["start"], comparison_end=period["end"])
        if artifact is None:
            raise RuntimeError(f"Audit CO seed={seed} 未通過驗證：{reason}；不自動訓練")
        files.extend((binding["path"], binding["manifest_path"],
                      artifact["score"], artifact["manifest"], artifact["audit"]))
        jobs.append(dict(seed=seed, benchmark=True, params=binding["path"],
                         mode=binding["evaluation_mode"],
                         overrides={arm.dl_id: {"score_path": artifact["score"],
                                                "manifest_path": artifact["manifest"]}}))
    return settings, arms, period, jobs, files


def run_formal_audit(definition, *, project_root: Path, quiet: bool = False):
    from filters.breakout_quality.source_inventory import build_source_data_inventory, source_data_inventory_read_scope
    from services.research.strategy_compare_execution import run_strategy_compare_active_arm
    from services.research.strategy_robustness_units import pin_files, require_pinned_files
    from services.research.oracle_build_attestation import capture_oracle_source_inventory, source_inventory_digest
    root = Path(project_root).resolve()
    settings, arms, period, jobs, files = _validated_inputs(definition, root)
    pins = pin_files(files)
    with source_data_inventory_read_scope(force_new=True):
        market = build_source_data_inventory(root, settings.dataset)
    identity = dict(definition=definition.as_dict(), period=period,
                    arms=[arm.as_dict() for arm in settings.enabled_arms],
                    seeds=[job["seed"] for job in jobs], dataset=market,
                    files={Path(path).resolve().relative_to(root).as_posix(): compute_observed_file_sha256(path)
                           for path in files},
                    replay_sources=source_inventory_digest(capture_oracle_source_inventory(root)),
                    audit_owner=compute_observed_file_sha256(Path(__file__)))
    base = root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    destination = base / canonical_json_sha256(identity)
    destination.mkdir(parents=True, exist_ok=True)
    rows, appendices = [], []
    bundles, controls = {arm.arm_id: [] for arm in arms}, {arm.arm_id: {} for arm in arms}
    with exclusive_file_lock(base / "writer.lock"):
        for job in jobs:
            baseline = None
            for arm in arms:
                require_pinned_files(pins)
                control = settings.arms[permutation_spec(arm.dl_runtime_options)["control_arm_id"]]
                cell = destination / f"seed_{job['seed']}" / arm.arm_id
                common = dict(settings=settings, project_root=root, params_path=job["params"],
                              comparison_start=period["start"], comparison_end=period["end"],
                              param_evaluation_mode=job["mode"], score_overrides=job["overrides"],
                              expected_seed_override=job["seed"] if job["benchmark"] else None,
                              quiet=True, capture_selection_target_diagnostics=False)
                if not quiet:
                    print(f"[Audit] seed={job['seed']} {control.arm_id} 原排名＋同日置換 99 次", flush=True)
                real = run_strategy_compare_active_arm(
                    **common, arm=control, output_dir=cell / "control", baseline_reuse_dir=baseline)
                baseline = baseline or cell / "control"
                null = run_strategy_compare_active_arm(
                    **common, arm=arm, output_dir=cell / "permutations", replay_active_only=True,
                    repetition_baseline_trades_path=baseline / "no_filter_trades.csv")
                require_pinned_files(pins)
                bundle = null["score_permutation"]
                appendices.append(render_permutation_appendix(bundle))
                rows.append(dict(seed=job["seed"], benchmark=job["benchmark"], control=control.arm_id,
                                 real=real["score_ranking"], distributions=bundle["distributions"],
                                 coverage=bundle["coverage"],
                                 evidence=(cell.relative_to(root).as_posix())))
                if job["benchmark"]:
                    bundles[arm.arm_id].append(bundle)
                    controls[arm.arm_id][job["seed"]] = real["score_ranking"]
        require_pinned_files(pins)
        with source_data_inventory_read_scope(force_new=True):
            if build_source_data_inventory(root, settings.dataset) != market:
                raise RuntimeError("Audit market inputs changed; no result published")
        gates = {arm.arm_id: evaluate_permutation_control(bundles[arm.arm_id], controls[arm.arm_id]) for arm in arms}
        result = dict(audit_id=definition.audit_id, status="COMPLETED", identity=identity,
                      rows=rows, gates=gates, iterative_research_oos=True)
        lines = ["# CO 同日候選分數置換 Audit", "", "排名貢獻診斷；不是新模型、正式候選或晉級結果。",
                 "", "| 原策略 | Seed | 原 RoMD | 置換 P05 / P50 / P95 | 籃子改變率 |",
                 "|---|---:|---:|---:|---:|"]
        for row in rows:
            dist = row["distributions"][ROMD]
            lines.append(f"| {row['control']} | {row['seed']} | {row['real'][ROMD]:.4f} | "
                         f"{dist['p05']:.4f} / {dist['p50']:.4f} / {dist['p95']:.4f} | "
                         f"{row['coverage'].get('selection_change_fraction', 0):.2%} |")
        for arm in arms:
            gate = gates[arm.arm_id]
            control_id = permutation_spec(arm.dl_runtime_options)["control_arm_id"]
            lines.extend(("", f"{control_id} 四 Seed：{gate['status']}；尾端比例 "
                          f"{gate['empirical_tail_fraction']:.2%}；高於各 Seed 中位數 "
                          f"{gate['individual_seed_median_wins']}/4；平均 MDD "
                          f"{gate['real_mean_mdd']:.4f}% / 置換 {gate['null_mean_mdd']:.4f}%。"))
        lines.extend(("", "Exact 為主判定，Direct 不能抵銷失敗。Seed42 僅補充。",
                      "迭代研究 OOS；置換也破壞跨日分數持續性，尾端比例不代表正式統計顯著性。"))
        lines.extend(appendices)
        report = "\n".join(lines) + "\n"
        for output in (destination, base / "latest"):
            output.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output / "audit.json", result)
            atomic_write_text(output / "audit.md", report)
        if not quiet:
            print(report)
        return result

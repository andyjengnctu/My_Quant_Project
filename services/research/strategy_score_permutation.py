"""Fixed-repeat observations over the canonical Strategy Compare replay owner.

No portfolio, accounting, metrics or selection implementation lives here. Each
child is an ordinary replay; only completed, pinned children enter the mean.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from core.candidate_score_permutation import POLICY, REPEATS, permutation_replay_scope
from core.file_integrity import (atomic_write_json, canonical_json_sha256,
                                 compute_observed_file_sha256, exclusive_file_lock,
                                 load_json_strict)

BUNDLE = "score_permutation.json"
KIND = "fixed_candidate_score_permutations_v1"
ROMD = "return_over_max_drawdown"
MDD = "max_drawdown_pct"


def is_permutation_payload(payload):
    return dict(payload.get("score_permutation") or {}).get("kind") == KIND


def _relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _pin(root, path):
    return {"path": _relative(root, path), "sha256": compute_observed_file_sha256(path)}


def _verified_path(root, item):
    path = (Path(root) / item["path"]).resolve()
    path.relative_to(Path(root).resolve())
    if not path.is_file() or compute_observed_file_sha256(path) != item["sha256"]:
        raise ValueError("Permutation observation evidence missing or SHA mismatch: " + item["path"])
    return path


def _mean_records(records):
    result = {}
    for key in set().union(*(record.keys() for record in records)):
        values = [record.get(key) for record in records]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in values):
            result[key] = float(math.fsum(values) / len(values))
        elif all(value == values[0] for value in values):
            result[key] = values[0]
        else:
            # Never silently drop missing observations from a metric's mean.
            result[key] = None
    return result


def _mean_yearly(payloads):
    years = [[str(row["year"]) for row in payload["yearly"]] for payload in payloads]
    if any(value != years[0] for value in years):
        raise ValueError("Permutation replays have different annual coverage")
    result = []
    for index, year in enumerate(years[0]):
        row = _mean_records([payload["yearly"][index] for payload in payloads])
        row["year"] = payloads[0]["yearly"][index]["year"]
        result.append(row)
    return result


def distribution(records, key):
    values = [row.get(key) for row in records]
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return {name: None for name in ("mean", "p05", "p50", "p95")}
    return {"mean": float(np.mean(values)), **dict(zip(
        ("p05", "p50", "p95"), (float(x) for x in np.quantile(values, [.05, .5, .95]))))}


def validate_bundle(bundle, *, root, verify_children=True):
    if (bundle.get("kind") != KIND or bundle.get("policy") != POLICY
            or bundle.get("status") != "COMPLETED" or bundle.get("repeats") != REPEATS
            or bundle.get("completed") != REPEATS or type(bundle.get("seed")) is not int):
        raise ValueError("Incomplete or incompatible score permutation observation")
    records = bundle.get("observations", [])
    if [item.get("repeat_id") for item in records] != list(range(1, REPEATS + 1)):
        raise ValueError("Permutation repeats must be exactly 1..99 without duplicates")
    for item in records:
        if any(not isinstance(item.get("metrics", {}).get(key), (int, float))
               or not math.isfinite(item["metrics"][key]) for key in (ROMD, MDD, "total_return_pct")):
            raise ValueError("Permutation repeat missing finite portfolio metrics")
        if verify_children:
            receipt = load_json_strict(_verified_path(root, item["receipt"]))
            if (receipt.get("identity") != bundle["identity"] or receipt.get("seed") != bundle["seed"]
                    or receipt.get("repeat_id") != item["repeat_id"]
                    or receipt.get("metrics") != item["metrics"]
                    or receipt.get("coverage") != item["coverage"]):
                raise ValueError("Permutation child differs from its committed aggregate")
            for pin in receipt["files"]:
                _verified_path(root, pin)
    if bundle.get("mean_metrics") != _mean_records([item["metrics"] for item in records]):
        raise ValueError("Permutation aggregate is not the arithmetic mean of 99 observations")
    expected = {key: distribution([item["metrics"] for item in records], key)
                for key in (ROMD, MDD, "total_return_pct")}
    if (bundle.get("distributions") != expected or bundle.get("coverage") != _mean_records(
            [item["coverage"] for item in records])):
        raise ValueError("Permutation distribution or coverage differs from its 99 observations")
    return bundle


def load_bundle(pair_dir, *, root):
    return validate_bundle(load_json_strict(Path(pair_dir) / BUNDLE), root=root)


def run_permutation_comparison(*, run_once, arguments, root, spec, seed, arm,
                               settings, score_paths, baseline_trades_path=None):
    """Bind replay inputs before entering the checkpoint/aggregation owner."""
    from filters.breakout_quality.source_inventory import build_source_data_inventory, source_data_inventory_read_scope
    from services.research.strategy_robustness_units import pin_files, require_pinned_files
    from services.research.oracle_build_attestation import capture_oracle_source_inventory, source_inventory_digest
    params_path = arguments["params_path"]
    identity = {
        "arm": arm.as_dict(), "seed": seed,
        "period": {"start": arguments["comparison_start_date"], "end": arguments["comparison_end_date"]},
        "params_sha256": compute_observed_file_sha256(params_path),
        "source_files": {key: compute_observed_file_sha256(path) for key, path in score_paths.items()},
        "dataset": build_source_data_inventory(root, settings.dataset),
        "execution_sources": source_inventory_digest(capture_oracle_source_inventory(root)),
        "repeat_owner_sha256": compute_observed_file_sha256(Path(__file__)),
        "max_positions": settings.max_positions, "rotation": settings.rotation,
        "param_evaluation_mode": arguments["param_evaluation_mode"],
        "ranking_options": arguments["ranking_options"],
        "active_only": bool(arguments["replay_active_only"]),
        "diagnostic_baseline_sha256": (compute_observed_file_sha256(baseline_trades_path)
                                        if baseline_trades_path is not None else None),
    }
    input_pins = pin_files([params_path, *score_paths.values(),
                           *([baseline_trades_path] if baseline_trades_path else [])])
    def verify_repeat_inputs():
        require_pinned_files(input_pins)
        # A nested fresh scope cannot inherit a plan's earlier market witness.
        with source_data_inventory_read_scope(force_new=True):
            if build_source_data_inventory(root, settings.dataset) != identity["dataset"]:
                raise ValueError("Permutation input market inventory changed during replay")
    return run_fixed_permutations(
        run_once=run_once, arguments=arguments, root=root, spec=spec, seed=seed,
        identity=identity, baseline_trades_path=baseline_trades_path,
        verify_inputs=verify_repeat_inputs,
    )


def run_fixed_permutations(*, run_once, arguments, root, spec, seed, identity,
                           baseline_trades_path=None, verify_inputs=None):
    """Resume immutable children in a stable store, independent of run directories."""
    root = Path(root).resolve()
    destination = Path(arguments["output_dir_override"]).resolve()
    identity = {**identity, "kind": KIND, "policy": POLICY, "seed": int(seed), "spec": spec}
    fingerprint = canonical_json_sha256(identity)
    store = root / "outputs/strategy_compare/permutation_observations" / fingerprint
    store.mkdir(parents=True, exist_ok=True)
    payloads, observations = [], []
    with exclusive_file_lock(store / "writer.lock"):
        for repeat in range(1, REPEATS + 1):
            if verify_inputs is not None:
                verify_inputs()
            directory = store / f"repeat_{repeat:03d}"
            receipt_path = directory / "permutation_receipt.json"
            reused = receipt_path.is_file()
            if receipt_path.is_file():
                receipt = load_json_strict(receipt_path)
                if (receipt.get("identity") != identity or receipt.get("seed") != seed
                        or receipt.get("repeat_id") != repeat):
                    raise ValueError("Permutation checkpoint identity mismatch")
                for pin in receipt["files"]:
                    _verified_path(root, pin)
                payload = load_json_strict(directory / "permutation_result.json")
                if payload["score_ranking"] != receipt["metrics"]:
                    raise ValueError("Permutation checkpoint metrics mismatch")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                call = {**arguments, "output_dir_override": directory,
                        "capture_selection_target_diagnostics": False}
                # The original same-parameter baseline is shared, never randomized.
                if payloads and not call.get("replay_active_only"):
                    call["baseline_reuse_dir"] = store / "repeat_001"
                with gzip.open(directory / "permutation_decisions.jsonl.gz", "wt", encoding="utf-8") as evidence:
                    with permutation_replay_scope(seed=seed, repeat_id=repeat, evidence_file=evidence) as state:
                        payload = run_once(**call)
                    coverage = state.summary()
                from services.research.strategy_comparison import _exclusive_selection_r_from_trade_files
                baseline = (Path(baseline_trades_path) if baseline_trades_path is not None else
                            directory / "no_filter_trades.csv")
                baseline_payload = load_json_strict(baseline.parent / "strategy_comparison.json")
                payload["score_ranking"]["direct_selection_r"] = _exclusive_selection_r_from_trade_files(
                    directory, active_trades_filename="score_ranking_trades.csv",
                    baseline_trades_path=baseline,
                    baseline_trade_count=baseline_payload["no_filter"]["trade_count"],
                    active_trade_count=payload["score_ranking"]["trade_count"])
                if verify_inputs is not None:
                    verify_inputs()
                payload["metadata"]["candidate_score_randomization"] = {
                    "policy": POLICY, "seed": seed, "repeat_id": repeat,
                    "mapping_evidence": "permutation_decisions.jsonl.gz"}
                atomic_write_json(directory / "permutation_result.json", payload)
                receipt = {"identity": identity, "seed": seed, "repeat_id": repeat,
                           "metrics": payload["score_ranking"], "coverage": coverage,
                           "files": [_pin(root, path) for path in sorted(directory.iterdir())
                                     if path.is_file() and path != receipt_path]}
                # Commit last. A partial run never qualifies as a finished repeat.
                atomic_write_json(receipt_path, receipt)
            payloads.append(payload)
            observations.append({"repeat_id": repeat, "metrics": receipt["metrics"],
                                 "coverage": receipt["coverage"], "receipt": _pin(root, receipt_path)})
            if arguments.get("progress_callback"):
                arguments["progress_callback"]("permutation_repeat_done", {
                    "repeat_id": repeat, "repeats": REPEATS, "seed": seed, "reused": reused})
            elif arguments.get("quiet"):
                print(f"[{'REUSE' if reused else 'DONE'}] seed={seed} 同日置換 {repeat}/{REPEATS}", flush=True)
        bundle = {"kind": KIND, "policy": POLICY, "status": "COMPLETED", "seed": seed,
                  "repeats": REPEATS, "completed": len(observations), "identity": identity,
                  "control_arm_id": spec["control_arm_id"], "observations": observations,
                  "mean_metrics": _mean_records([item["metrics"] for item in observations]),
                  "distributions": {key: distribution([x["metrics"] for x in observations], key)
                                    for key in (ROMD, MDD, "total_return_pct")},
                  "coverage": _mean_records([item["coverage"] for item in observations])}
        validate_bundle(bundle, root=root, verify_children=False)
        result = deepcopy(payloads[0])
        result["score_ranking"] = bundle["mean_metrics"]
        result["yearly"] = _mean_yearly(payloads)
        result["selection_diagnostics"] = None
        result["score_permutation"] = bundle
        result["metadata"]["aggregation"] = "arithmetic_mean_of_99_complete_replays"
        result["metadata"].pop("candidate_score_randomization", None)
        result["metadata"]["output_dir"] = _relative(root, destination)
        result["metadata"]["selection_target_diagnostics_captured"] = False
        if "no_filter" in result:
            from filters.breakout_quality.strategy_compare_reporting import _delta
            result["score_ranking_minus_no_filter"] = _delta(result["score_ranking"], result["no_filter"])
        destination.mkdir(parents=True, exist_ok=True)
        # Retain a real common baseline for the existing baseline reuse owner.
        for source in (store / "repeat_001").glob("no_filter_*.csv"):
            shutil.copy2(source, destination / source.name)
        atomic_write_json(destination / BUNDLE, bundle)
        pd.DataFrame(result["yearly"]).to_csv(destination / "yearly_returns_comparison.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"repeat_id": x["repeat_id"], **x["metrics"]} for x in observations]).to_csv(
            destination / "permutation_metrics.csv", index=False, encoding="utf-8-sig")
        atomic_write_json(destination / "strategy_comparison.json", result)
        return result


def render_permutation_appendix(bundle):
    lines = ["", "### 同日候選分數置換（固定 99 次平均）", "",
             f"Seed {bundle['seed']}：{bundle['completed']}/{bundle['repeats']}；控制組 {bundle['control_arm_id']}。",
             "標準績效欄為每次完整回放指標的算術平均；沒有平均投組或代表性交易歷史。",
             "", "| 指標 | Mean | P05 | P50 | P95 |", "|---|---:|---:|---:|---:|"]
    for key, label in ((ROMD, "RoMD"), (MDD, "MDD %"), ("total_return_pct", "總報酬 %")):
        values = bundle["distributions"][key]
        lines.append("| " + label + " | " + " | ".join(f"{values[x]:.4f}" for x in ("mean", "p05", "p50", "p95")) + " |")
    coverage = bundle["coverage"]
    lines += ["", "候選覆蓋（99 次平均）：" + ", ".join(f"{key}={value}" for key, value in coverage.items()),
              "選股變更率＝與同路徑、同狀態原分數影子選股相比，籃子不同的候選日比例。",
              "迭代研究 OOS；置換亦破壞分數時間持續性，尾端比例不是統計顯著性保證。"]
    return "\n".join(lines)


def evaluate_permutation_control(bundles, real_by_seed):
    """Predeclared four-seed gate; ties count against the real score assignment."""
    seeds = sorted(real_by_seed)
    if len(seeds) != 4 or sorted(bundle["seed"] for bundle in bundles) != seeds:
        return {"status": "INCOMPLETE", "reason": "requires all four paired benchmark seeds"}
    by_seed = {bundle["seed"]: bundle for bundle in bundles}
    for bundle in bundles:
        validate_bundle(bundle, root=Path.cwd(), verify_children=False)
    null = [float(np.mean([by_seed[seed]["observations"][b]["metrics"][ROMD] for seed in seeds]))
            for b in range(REPEATS)]
    real = float(np.mean([real_by_seed[seed][ROMD] for seed in seeds]))
    count = sum(value >= real for value in null)
    wins = sum(real_by_seed[seed][ROMD] > by_seed[seed]["distributions"][ROMD]["p50"] for seed in seeds)
    real_mdd = float(np.mean([real_by_seed[seed][MDD] for seed in seeds]))
    null_mdd = float(np.mean([by_seed[seed]["mean_metrics"][MDD] for seed in seeds]))
    passed = count <= 4 and wins >= 3 and real_mdd <= null_mdd
    return {"status": "PASS" if passed else "FAIL", "seed_count": 4, "repeats": REPEATS,
            "real_mean_romd": real, "null_mean_romd_distribution": distribution([{ROMD: x} for x in null], ROMD),
            "null_ge_real_count": count, "empirical_tail_fraction": (1 + count) / 100,
            "individual_seed_median_wins": wins, "real_mean_mdd": real_mdd, "null_mean_mdd": null_mdd,
            "tail_gate": count <= 4, "seed_consistency_gate": wins >= 3,
            "mdd_gate": real_mdd <= null_mdd, "iterative_research_oos": True}

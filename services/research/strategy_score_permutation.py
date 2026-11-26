"""Fixed repeats over the existing Strategy Compare replay, with durable resume."""
from __future__ import annotations
from copy import deepcopy
import gzip
from pathlib import Path
import shutil
import pandas as pd
from core.candidate_score_permutation import permutation_replay_scope
from core.candidate_score_permutation_contract import POLICY, REPEATS
from core.file_integrity import atomic_write_json, canonical_json_sha256, compute_observed_file_sha256, exclusive_file_lock, load_json_strict
from core.score_permutation_observation import (
    BUNDLE, KIND, ROMD, MDD, is_permutation_payload, _relative, _pin, _verified_path, _mean_records, _mean_yearly, distribution, validate_bundle, load_bundle, render_permutation_appendix, evaluate_permutation_control
)

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
        "aggregate_owner_sha256": compute_observed_file_sha256(Path(root) / "core/score_permutation_observation.py"),
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
                from filters.breakout_quality.strategy_compare_diagnostics import exclusive_selection_r_from_trade_files
                baseline = (Path(baseline_trades_path) if baseline_trades_path is not None else
                            directory / "no_filter_trades.csv")
                baseline_payload = load_json_strict(baseline.parent / "strategy_comparison.json")
                payload["score_ranking"]["direct_selection_r"] = exclusive_selection_r_from_trade_files(
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


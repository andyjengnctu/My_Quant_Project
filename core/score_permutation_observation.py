"""Complete repeated-replay observations, statistics and diagnostic presentation.

This owner never executes a strategy or imports a Research service.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from core.candidate_score_permutation_contract import POLICY, REPEATS
from core.file_integrity import compute_observed_file_sha256, load_json_strict

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


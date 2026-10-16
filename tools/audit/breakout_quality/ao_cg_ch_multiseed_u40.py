"""One-off, read-only AO/CG/CH Rolling 4-seed common-truth U40 audit.

This is evaluation-only research evidence: it never fits, tunes, selects seeds,
changes the persistent Standard SOP, or writes to model/score/manifest folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.breakout_quality_policy import (
    get_breakout_quality_model_test_settings,
    get_breakout_quality_workflow_settings,
)
from core.training_policy import (
    get_robustness_benchmark_policy_snapshot,
    resolve_robustness_benchmark_seeds,
)
from filters.breakout_quality.daily_ranker_data import (
    load_daily_universal_ranker_data,
    load_official_breakout_candidate_keys,
)
from services.breakout_quality.point_in_time_audit import _validate_score_artifacts
from services.breakout_quality.u40_paired_comparison import (
    _canonical_u40_truth,
    _matched_rows,
    _paired_report_rows,
)
from services.research.breakout_quality_application import (
    _load_rolling_robustness_seed,
    _model_robustness_dirs,
)

AUDIT_ID = "AUD-ao-cg-ch-four-seed-common-u40"
MODE = "rolling_oos"
MODELS = ("MR-13AO", "MR-13CG", "MR-13CH")
CONTRASTS = (("MR-13CG", "MR-13AO"), ("MR-13CH", "MR-13AO"),
             ("MR-13CH", "MR-13CG"))
SCOPE_ALL = "Rolling OOS"
SCOPE_BREAKOUT = "Breakout slice"
VALUE_KEYS = ("top10_u40_r", "top10_mfe40_r", "top10_d40_r")


def _model_settings() -> dict[str, Any]:
    configured = dict(get_breakout_quality_model_test_settings().model_profiles)
    missing = [model for model in MODELS if model not in configured]
    if missing:
        raise ValueError(f"Current Research compare list does not include {missing}; audit cannot change the list")
    settings = {model: get_breakout_quality_workflow_settings(
        experiment_profile=configured[model],
    ) for model in MODELS}
    targets = {model: str(item.continuous_target_id) for model, item in settings.items()}
    from core.breakout_quality_runtime import (
        DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        DAILY_20BAR_FULL_HORIZON_PURE_MFE_TARGET_ID,
    )
    if (targets[MODELS[0]] != DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID
            or targets[MODELS[1]] != DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID
            or targets[MODELS[2]] != DAILY_20BAR_FULL_HORIZON_PURE_MFE_TARGET_ID):
        raise ValueError(f"Unexpected AO/CG/CH scientific target contracts: {targets}")
    if len({str(s.filter_id) for s in settings.values()}) != 1:
        raise ValueError("Models use different filter universes")
    return settings


def _validated_score_for_seed(*, model_id: str, settings: Any, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require existing official READY and the same strict PIT artifact validator."""
    report, _path, reason = _load_rolling_robustness_seed(settings, seed=seed)
    if report is None:
        raise ValueError(f"{model_id} seed={seed} not READY: {reason}; this audit cannot build or train")
    pit_dir, _unused = _model_robustness_dirs(settings, seed=seed, rolling=True)
    args = SimpleNamespace(
        filter_id=str(settings.filter_id),
        model_architecture=str(settings.model_architecture),
        experiment_profile=str(settings.experiment_profile),
        point_in_time_dir_override=str(pit_dir),
    )
    frame, manifest = _validate_score_artifacts(args)
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError(f"{model_id} unexpected source seed: {manifest.get('seed')} != {seed}")
    if str(manifest.get("continuous_target_id")) != str(settings.continuous_target_id):
        raise ValueError(f"{model_id} target identity stale")
    return (frame[["ticker", "date", "group_index", "breakout_quality_score",
                   "fold_id", "model_information_cutoff"]]
            .rename(columns={"breakout_quality_score": "score"})), {
        "source_manifest": str(pit_dir.relative_to(PROJECT_ROOT) / "selection_point_in_time_manifest.json"),
        "source_score_sha256": str((manifest.get("artifacts") or {}).get("scores", {}).get("sha256", "")),
        "source_seed": int(seed),
        "source_target_id": str(manifest["continuous_target_id"]),
    }


def _score_key_digest(frame: pd.DataFrame) -> str:
    """Same complete scored stock-days/folds/cutoffs, regardless of CSV row order."""
    columns = ["ticker", "date", "group_index", "fold_id", "model_information_cutoff"]
    if len(frame) == 0 or frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Empty or duplicate Rolling source score cohort")
    identity = frame[columns].sort_values(["date", "ticker"], kind="mergesort")
    hashed = pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _rows_by_scope(extension: Mapping[str, Any], *, year: bool = False) -> dict[tuple[str, int | None], dict[str, Any]]:
    key = "year_rows" if year else "rows"
    result = {}
    for row in extension[key]:
        name = (str(row["split"]), int(row["year"]) if year else None)
        if name in result:
            raise ValueError("Duplicated audit scope/year")
        result[name] = dict(row)
    return result


def _paired_seed_result(*, seed: int, truth: pd.DataFrame,
                        frames: Mapping[str, pd.DataFrame],
                        candidates: set[tuple[str, pd.Timestamp]]) -> dict[str, Any]:
    """Re-use official U40 mature-truth, stock-day join and exact Top10 selection."""
    if set(frames) != set(MODELS):
        raise ValueError("AO/CG/CH audit requires all three model scores for the same seed")
    identities = {model: _score_key_digest(frame) for model, frame in frames.items()}
    if len(set(identities.values())) != 1:
        raise ValueError(f"Seed={seed} incomplete/mismatched score, Fold or PIT cutoff universe")
    common = _matched_rows(
        truth=truth, reference=frames["MR-13AO"],
        treatments={"direct_0": frames["MR-13CG"], "direct_1": frames["MR-13CH"]},
        mode=MODE,
    )
    reference_mature = frames["MR-13AO"][["ticker", "date"]].merge(
        truth[["ticker", "date"]], on=["ticker", "date"], how="inner", validate="one_to_one"
    )
    if len(common) != len(reference_mature):
        raise ValueError("Mature AO OOS rows were lost when matching CG/CH")
    extensions = _paired_report_rows(
        common=common, reference_id="MR-13AO", treatment_id=None,
        treatment_score_columns={"MR-13CG": "score_direct_0", "MR-13CH": "score_direct_1"},
        mode=MODE, breakout_candidate_keys=candidates,
    )
    row_map = {model: _rows_by_scope(extensions[model]) for model in MODELS}
    year_map = {model: _rows_by_scope(extensions[model], year=True) for model in MODELS}
    if not all(row_map[model].keys() == row_map[MODELS[0]].keys() for model in MODELS):
        raise ValueError("Same-seed AO/CG/CH scope rows are incompatible")
    if not all(year_map[model].keys() == year_map[MODELS[0]].keys() for model in MODELS):
        raise ValueError("Same-seed AO/CG/CH year rows are incompatible")
    for rows in (row_map, year_map):
        for scope in rows[MODELS[0]]:
            quota = (rows[MODELS[0]][scope]["top10_n"], rows[MODELS[0]][scope]["rankable_dates"])
            if any((rows[m][scope]["top10_n"], rows[m][scope]["rankable_dates"]) != quota for m in MODELS):
                raise ValueError("Paired Top10 quotas differ")
    return {
        "seed": int(seed), "complete_score_rows": len(frames[MODELS[0]]),
        "common_mature_rows": len(common), "score_key_sha256": identities[MODELS[0]],
        "cohort_sha256": extensions[MODELS[0]]["cohort_sha256"],
        "rows": {model: list(row_map[model].values()) for model in MODELS},
        "year_rows": {model: list(year_map[model].values()) for model in MODELS},
    }


def _metrics_per_seed(seed_results: list[dict[str, Any]], *, year: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate paired *differences*, never metrics from incompatible model targets."""
    if not seed_results:
        raise ValueError("No fully verified benchmark seed")
    source = "year_rows" if year else "rows"
    def _index(item, model):
        return {(r["split"], r.get("year") if year else None): r
                for r in item[source][model]}
    reference = _index(seed_results[0], MODELS[0])
    metrics = []
    contrasts = []
    for key in reference:
        scope, yr = key
        for model in MODELS:
            vectors = [_index(run, model)[key] for run in seed_results]
            template = vectors[0]
            if any(int(v["top10_n"]) != int(template["top10_n"]) or
                   int(v["stock_days"]) != int(template["stock_days"]) or
                   int(v["rankable_dates"]) != int(template["rankable_dates"]) for v in vectors):
                raise ValueError(f"Cross-seed sample/Top10 quota mismatch: {key} {model}")
            for metric in VALUE_KEYS:
                values = np.asarray([float(v[metric]) for v in vectors], dtype=np.float64)
                row = {"split": scope, "model": model, "metric": metric,
                       "seed_count": len(seed_results), "top10_n_per_seed": int(template["top10_n"]),
                       "mean_r": float(values.mean()), "sigma_r": float(values.std(ddof=0)),
                       "per_seed_r": [float(v) for v in values]}
                if year:
                    row["year"] = int(yr)
                metrics.append(row)
        for model, control in CONTRASTS:
            model_values = [_index(run, model)[key] for run in seed_results]
            control_values = [_index(run, control)[key] for run in seed_results]
            for metric in VALUE_KEYS:
                differences = np.asarray([
                    float(t[metric]) - float(c[metric])
                    for t, c in zip(model_values, control_values)
                ], dtype=np.float64)
                if not np.isfinite(differences).all():
                    raise ValueError("Non-finite paired economic metric")
                row = {"split": scope, "model": model, "reference": control,
                       "metric": metric, "seed_count": len(seed_results),
                       "delta_mean_r": float(differences.mean()),
                       "delta_sigma_r": float(differences.std(ddof=0)),
                       "positive_seeds": int((differences > 0).sum()),
                       "negative_seeds": int((differences < 0).sum()),
                       "zero_seeds": int((differences == 0).sum()),
                       "paired_delta_by_seed_r": [float(d) for d in differences]}
                if year:
                    row["year"] = int(yr)
                contrasts.append(row)
    return metrics, contrasts


def _make_report(*, seed_results: list[dict[str, Any]], benchmark: Mapping[str, Any],
                 model_manifests: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    seeds = [int(run["seed"]) for run in seed_results]
    if seeds != [int(v) for v in benchmark["resolved_seeds"]]:
        raise ValueError("Audit benchmark seeds do not equal the registered benchmark")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Repeated benchmark seed")
    for field in ("score_key_sha256", "common_mature_rows"):
        if len({run[field] for run in seed_results}) != 1:
            raise ValueError(f"Cross-seed AO/CG/CH eligibility differs: {field}")
    if len({tuple(sorted(run["cohort_sha256"].items())) for run in seed_results}) != 1:
        raise ValueError("Cross-seed cohort SHA mismatch")
    overall_metrics, overall_deltas = _metrics_per_seed(seed_results, year=False)
    annual_metrics, annual_deltas = _metrics_per_seed(seed_results, year=True)
    return {
        "audit_id": AUDIT_ID, "status": "RESULT_AVAILABLE", "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": MODE, "benchmark": dict(benchmark),
        "models": list(MODELS), "source_manifests": dict(model_manifests),
        "scientific_boundary": {
            "truth": "AO canonical 40D-mature full-bar MFE40/D40/R; U40=MFE40-D40",
            "common_stock_days": "strict common ticker/date/group_index/fold/cutoff, all 3 models, all 4 seeds",
            "top10": "same-date top ceil(10% of common eligible), min day pool > 10; ticker tie-break",
            "robustness_sigma": "population standard deviation ddof=0 of paired seed differences",
            "evidence": "iterative OOS; no fitting/selection/promotion",
        },
        "seed_results": seed_results, "overall_mean_sigma": overall_metrics,
        "overall_paired_deltas": overall_deltas,
        "year_mean_sigma": annual_metrics, "year_paired_deltas": annual_deltas,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# AO / CG / CH four-seed Rolling OOS common U40", "",
             f"Benchmark: `{report['benchmark']['benchmark_id']}`; seeds: " +
             ", ".join(str(x) for x in report["benchmark"]["resolved_seeds"]), "",
             "Same 40D mature stock-day/fold/cutoff cohort and same Top10 quota per seed.",
             "All entries below are evaluation-only iterative OOS, not realized portfolio PnL.", ""]
    for scope in (SCOPE_ALL, SCOPE_BREAKOUT):
        lines += [f"## {scope}", "", "| Model | Top10 U40 mean±σ (R) | MFE40 mean (R) | D40 mean (R) |",
                  "|---|---:|---:|---:|"]
        for model in MODELS:
            lookup = {row["metric"]: row for row in report["overall_mean_sigma"]
                      if row["split"] == scope and row["model"] == model}
            lines.append(f"| {model} | {lookup['top10_u40_r']['mean_r']:.4f} ± {lookup['top10_u40_r']['sigma_r']:.4f} | "
                         f"{lookup['top10_mfe40_r']['mean_r']:.4f} | {lookup['top10_d40_r']['mean_r']:.4f} |")
        lines += ["", "| Paired U40 contrast | Mean ΔR | σ(ΔR) | Winning seeds | Seed-wise ΔR |",
                  "|---|---:|---:|---:|---|"]
        for t, b in CONTRASTS:
            row = next(r for r in report["overall_paired_deltas"]
                       if r["split"] == scope and r["model"] == t and r["reference"] == b
                       and r["metric"] == "top10_u40_r")
            sequence = ", ".join(f"{v:+.4f}" for v in row["paired_delta_by_seed_r"])
            lines.append(f"| {t} − {b} | {row['delta_mean_r']:+.4f} | {row['delta_sigma_r']:.4f} | "
                         f"{row['positive_seeds']}/{row['seed_count']} | {sequence} |")
        lines += ["", "### Year × seed paired ΔU40", "",
                  "| Year | CG−AO mean (wins) | CH−AO mean (wins) | CH−CG mean (wins) |",
                  "|---|---:|---:|---:|"]
        years = sorted({r["year"] for r in report["year_paired_deltas"] if r["split"] == scope})
        for year in years:
            cells = []
            for model, ref in CONTRASTS:
                row = next(r for r in report["year_paired_deltas"]
                           if r["split"] == scope and r["year"] == year and r["model"] == model
                           and r["reference"] == ref and r["metric"] == "top10_u40_r")
                cells.append(f"{row['delta_mean_r']:+.4f} ({row['positive_seeds']}/{row['seed_count']})")
            lines.append(f"| {year} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Evidence limits", "",
              "- Four seeds are not four independent calendar regimes; folds and maturities overlap across models.",
              "- This is repeatedly inspected iterative OOS. No statistical significance or strategy promotion is established.",
              "- D40 lower is better, U40 and MFE40 higher are better; these are not executable trade returns.",
              "- Persistent Standard SOP/Robustness comparison schema is untouched.", ""]
    return "\n".join(lines)


def run(*, root: Path = PROJECT_ROOT, write_report: bool = True) -> dict[str, Any]:
    if Path(root).resolve() != PROJECT_ROOT.resolve():
        raise ValueError("One-off audit must operate on this project's canonical root")
    benchmark = get_robustness_benchmark_policy_snapshot()
    seeds = tuple(int(s) for s in resolve_robustness_benchmark_seeds())
    if tuple(benchmark["resolved_seeds"]) != seeds:
        raise ValueError("Robustness benchmark SSOT mismatch")
    settings = _model_settings()
    source = settings[MODELS[0]]
    print("[READ-ONLY] AO/CG/CH four-seed common-U40 audit; no training", flush=True)
    # Canonical target builder reconstructs the shared raw full-horizon truth, not
    # any cross-target rho/SOP surrogate. No OOS value enters model selection.
    truth_bundle = load_daily_universal_ranker_data(
        filter_id=str(source.filter_id), model_architecture=str(source.model_architecture),
        experiment_profile=str(source.experiment_profile),
        preload_feature_bank=False, allow_stale_source=False, project_root=PROJECT_ROOT,
    )
    truth = _canonical_u40_truth(truth_bundle, source_target_id=str(source.continuous_target_id))
    candidates = load_official_breakout_candidate_keys(
        str(source.filter_id), allow_stale_source=False,
    )
    seed_results: list[dict[str, Any]] = []
    model_manifests: dict[int, dict[str, Any]] = {}
    for i, seed in enumerate(seeds, start=1):
        print(f"[CHECK] seed={seed} ({i}/{len(seeds)}): official READY/PIT SHA/folds", flush=True)
        frames: dict[str, pd.DataFrame] = {}
        model_manifests[seed] = {}
        for model in MODELS:
            frame, provenance = _validated_score_for_seed(
                model_id=model, settings=settings[model], seed=seed,
            )
            frames[model] = frame
            model_manifests[seed][model] = provenance
        result = _paired_seed_result(seed=seed, truth=truth, frames=frames, candidates=candidates)
        seed_results.append(result)
        print(f"[DONE] seed={seed}: mature_stock_days={result['common_mature_rows']}", flush=True)
        del frames
    report = _make_report(seed_results=seed_results, benchmark=benchmark,
                          model_manifests=model_manifests)
    markdown = _render_markdown(report)
    if write_report:
        destination = (PROJECT_ROOT / "outputs" / "filters" / "breakout_quality" /
                       str(source.filter_id) / "audit" / "ao_cg_ch_four_seed_common_u40")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (destination / "result.md").write_text(markdown, encoding="utf-8")
        print("[DONE] outputs/filters/breakout_quality/" + str(source.filter_id) +
              "/audit/ao_cg_ch_four_seed_common_u40/result.md")
    print(markdown)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AO/CG/CH 4-seed Rolling U40 read-only audit")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write audit result files")
    args = parser.parse_args(argv)
    run(write_report=not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

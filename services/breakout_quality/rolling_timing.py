"""Isolated Rolling training timing A/B harness.

This module benchmarks the canonical point-in-time trainer without changing its
scientific contract.  The first run stores an immutable baseline.  Later runs use
the current implementation as a candidate and compare wall-clock plus exact result
hashes against that baseline.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS,
    BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS,
    get_breakout_quality_rolling_timing_settings,
)
from core.console_report import (
    console_color_enabled,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from core.file_integrity import compute_file_sha256
from core.report_style import (
    SIGNAL_NEGATIVE,
    SIGNAL_POSITIVE,
    SIGNAL_WARNING,
    markdown_signal,
    terminal_signal,
)
from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME
from filters.breakout_quality.paths import resolve_filter_output_dir
from filters.breakout_quality.workflow_io import write_json
from services.breakout_quality.point_in_time_scores import (
    FOLD_MANIFEST_FILENAME,
    FOLD_SCORE_FILENAME,
    build_selection_point_in_time_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMING_SCHEMA_VERSION = 1
TIMING_KIND = "rolling_training_ab"
_SOURCE_FINGERPRINT_PATHS = (
    "services/breakout_quality/train_continuous_ranker.py",
    "services/breakout_quality/continuous_ranker_pipeline.py",
    "services/breakout_quality/point_in_time_scores.py",
    "filters/breakout_quality/daily_ranker_data.py",
)


def _timing_root(*, filter_id: str, experiment_profile: str, seed: int) -> Path:
    return (
        resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)
        / "timing"
        / "rolling_training"
        / str(experiment_profile)
        / f"seed_{int(seed)}"
    )


def _baseline_summary_path(root: Path) -> Path:
    return root / "baseline.json"


def _candidate_summary_path(root: Path) -> Path:
    return root / "candidate_latest.json"


def _latest_report_path(root: Path) -> Path:
    return root / "timing_report.md"


def resolve_rolling_timing_artifact_paths() -> dict[str, Path]:
    timing = get_breakout_quality_rolling_timing_settings()
    root = _timing_root(
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        experiment_profile=timing.experiment_profile,
        seed=timing.seed,
    )
    return {
        "root": root,
        "baseline": _baseline_summary_path(root),
        "candidate": _candidate_summary_path(root),
        "report": _latest_report_path(root),
    }


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in _SOURCE_FINGERPRINT_PATHS:
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            "Timing JSON根節點必須是object: "
            + project_relative_display_path(path, project_root=PROJECT_ROOT)
        )
    return payload


def _semantic_model_state_sha256(model_path: Path) -> str:
    import torch

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Timing checkpoint格式不合法")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Timing checkpoint缺少model_state_dict")

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not hasattr(tensor, "detach"):
            raise ValueError(f"Timing model_state_dict包含非Tensor: {key}")
        value = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _single_fold_artifacts(point_in_time_dir: Path) -> tuple[Path, Path, Path]:
    fold_root = point_in_time_dir / "folds"
    fold_dirs = sorted(path for path in fold_root.iterdir() if path.is_dir())
    if len(fold_dirs) != 1:
        raise ValueError(
            "Timing Mode每個benchmark year必須只產生一個fold: "
            f"observed={len(fold_dirs)}"
        )
    fold_dir = fold_dirs[0]
    manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
    model_path = fold_dir / DEFAULT_MODEL_FILENAME
    score_path = fold_dir / FOLD_SCORE_FILENAME
    for path in (manifest_path, model_path, score_path):
        if not path.is_file():
            raise FileNotFoundError(
                "Timing Mode缺少benchmark工件: "
                + project_relative_display_path(path, project_root=PROJECT_ROOT)
            )
    return manifest_path, model_path, score_path


def _run_year(*, run_root: Path, year: int, profile: str, seed: int) -> dict[str, Any]:
    year_root = run_root / f"year_{int(year)}"
    if year_root.exists():
        shutil.rmtree(year_root)
    point_in_time_dir = year_root / "point_in_time"
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    code = build_selection_point_in_time_scores(
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        experiment_profile=profile,
        score_start_date=f"{int(year):04d}-01-01",
        score_end_date=f"{int(year):04d}-12-31",
        fold_months=int(BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS),
        inner_validation_months=int(BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS),
        train_window_months=None,
        seed=int(seed),
        resume=False,
        checkpoint_only=False,
        plan_only=False,
        point_in_time_dir_override=str(point_in_time_dir),
    )
    elapsed_wall = time.perf_counter() - started_wall
    elapsed_cpu = time.process_time() - started_cpu
    if int(code) != 0:
        raise RuntimeError(f"Timing Mode benchmark失敗: year={year}, returncode={code}")

    manifest_path, model_path, score_path = _single_fold_artifacts(point_in_time_dir)
    manifest = _load_json(manifest_path)
    epoch_selection = dict(manifest.get("epoch_selection") or {})
    group_counts = dict(manifest.get("group_counts") or {})
    return {
        "year": int(year),
        "fold_id": str(manifest.get("fold_id") or ""),
        "elapsed_wall_sec": float(elapsed_wall),
        "elapsed_cpu_sec": float(elapsed_cpu),
        "selected_epoch": int(manifest.get("selected_epoch") or 0),
        "best_validation_mean_daily_spearman": epoch_selection.get(
            "best_validation_mean_daily_spearman"
        ),
        "contract_fingerprint": str(manifest.get("contract_fingerprint") or ""),
        "model_state_sha256": _semantic_model_state_sha256(model_path),
        "scores_file_sha256": compute_file_sha256(score_path),
        "group_counts": {
            key: int(group_counts.get(key, 0) or 0)
            for key in ("train", "validation", "final_refit", "score")
        },
    }


def _benchmark_payload(timing) -> dict[str, Any]:
    return {
        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        "experiment_profile": str(timing.experiment_profile),
        "seed": int(timing.seed),
        "score_years": [int(year) for year in timing.score_years],
        "fold_months": int(BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS),
        "inner_validation_months": int(
            BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS
        ),
        "train_window_months": None,
        "resume": False,
    }


def _build_run_summary(*, role: str, run_root: Path) -> dict[str, Any]:
    timing = get_breakout_quality_rolling_timing_settings()
    profile = str(timing.experiment_profile)
    seed = int(timing.seed)
    years = tuple(int(year) for year in timing.score_years)
    source_fingerprint = _source_fingerprint()
    rows: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for year in years:
        print(f"\n[Timing] {role} | year={year} | profile={profile} | seed={seed}")
        rows.append(
            _run_year(
                run_root=run_root,
                year=year,
                profile=profile,
                seed=seed,
            )
        )
    measurement_wall_sec = float(time.perf_counter() - total_started)
    benchmark_wall_sec = float(
        sum(float(row.get("elapsed_wall_sec", 0.0) or 0.0) for row in rows)
    )
    return {
        "schema_version": TIMING_SCHEMA_VERSION,
        "kind": TIMING_KIND,
        "role": str(role),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": _benchmark_payload(timing),
        "implementation_fingerprint": source_fingerprint,
        "elapsed_wall_sec": benchmark_wall_sec,
        "measurement_wall_sec": measurement_wall_sec,
        "years": rows,
    }


def _comparison_payload(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_benchmark = dict(baseline.get("benchmark") or {})
    cand_benchmark = dict(candidate.get("benchmark") or {})
    benchmark_equal = base_benchmark == cand_benchmark
    baseline_rows = {
        int(row["year"]): dict(row) for row in list(baseline.get("years") or [])
    }
    candidate_rows = {
        int(row["year"]): dict(row) for row in list(candidate.get("years") or [])
    }
    year_keys_equal = tuple(sorted(baseline_rows)) == tuple(sorted(candidate_rows))
    rows: list[dict[str, Any]] = []
    all_exact = bool(benchmark_equal and year_keys_equal)
    for year in sorted(set(baseline_rows) | set(candidate_rows)):
        base = baseline_rows.get(year)
        cand = candidate_rows.get(year)
        if base is None or cand is None:
            all_exact = False
            rows.append({"year": year, "exact_result": False, "missing_side": True})
            continue
        contract_equal = str(base.get("contract_fingerprint")) == str(
            cand.get("contract_fingerprint")
        )
        epoch_equal = int(base.get("selected_epoch", 0)) == int(
            cand.get("selected_epoch", 0)
        )
        model_equal = str(base.get("model_state_sha256")) == str(
            cand.get("model_state_sha256")
        )
        scores_equal = str(base.get("scores_file_sha256")) == str(
            cand.get("scores_file_sha256")
        )
        counts_equal = dict(base.get("group_counts") or {}) == dict(
            cand.get("group_counts") or {}
        )
        exact = bool(
            contract_equal and epoch_equal and model_equal and scores_equal and counts_equal
        )
        all_exact = bool(all_exact and exact)
        base_sec = float(base.get("elapsed_wall_sec", 0.0) or 0.0)
        cand_sec = float(cand.get("elapsed_wall_sec", 0.0) or 0.0)
        rows.append(
            {
                "year": year,
                "baseline_sec": base_sec,
                "candidate_sec": cand_sec,
                "elapsed_change_sec": cand_sec - base_sec,
                "elapsed_change_pct": (
                    ((cand_sec / base_sec) - 1.0) * 100.0 if base_sec > 0.0 else None
                ),
                "speedup_x": (base_sec / cand_sec if cand_sec > 0.0 else None),
                "contract_equal": contract_equal,
                "selected_epoch_equal": epoch_equal,
                "model_state_equal": model_equal,
                "scores_equal": scores_equal,
                "group_counts_equal": counts_equal,
                "exact_result": exact,
            }
        )
    baseline_total = float(baseline.get("elapsed_wall_sec", 0.0) or 0.0)
    candidate_total = float(candidate.get("elapsed_wall_sec", 0.0) or 0.0)
    return {
        "benchmark_equal": benchmark_equal,
        "year_keys_equal": year_keys_equal,
        "exact_result": all_exact,
        "baseline_total_sec": baseline_total,
        "candidate_total_sec": candidate_total,
        "total_elapsed_change_sec": candidate_total - baseline_total,
        "total_elapsed_change_pct": (
            ((candidate_total / baseline_total) - 1.0) * 100.0
            if baseline_total > 0.0
            else None
        ),
        "total_speedup_x": (
            baseline_total / candidate_total if candidate_total > 0.0 else None
        ),
        "years": rows,
    }


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    seconds = max(0.0, float(value))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60.0
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes:02d}:{secs:04.1f}"


def _render_report_markdown(
    *, baseline: dict[str, Any], candidate: dict[str, Any] | None, comparison: dict[str, Any] | None
) -> str:
    benchmark = dict(baseline.get("benchmark") or {})
    lines = [
        "# Rolling Training Timing Mode",
        "",
        f"- Profile：`{benchmark.get('experiment_profile')}`",
        f"- Seed：`{benchmark.get('seed')}`",
        f"- Score years：`{benchmark.get('score_years')}`",
        f"- Fold months：`{benchmark.get('fold_months')}`",
        f"- Inner validation months：`{benchmark.get('inner_validation_months')}`",
        "- Train window：`Extending / all legal history`",
        "- Resume：`False`（每次從零訓練，避免cache污染timing）",
        "",
        "## Baseline",
        "",
        f"- Implementation fingerprint：`{baseline.get('implementation_fingerprint')}`",
        f"- Total wall-clock：`{_format_seconds(float(baseline.get('elapsed_wall_sec', 0.0) or 0.0))}`",
    ]
    if candidate is None or comparison is None:
        lines.extend(
            [
                "",
                "> Baseline 已建立。程式改善後以相同 Timing Mode 再跑一次，即會產生 candidate A/B 比較。",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "",
            "## Candidate vs Baseline",
            "",
            f"- Candidate implementation fingerprint：`{candidate.get('implementation_fingerprint')}`",
            "- Exact result："
            + markdown_signal(
                "PASS / bitwise exact" if comparison.get("exact_result") else "FAIL / result changed",
                SIGNAL_POSITIVE if comparison.get("exact_result") else SIGNAL_NEGATIVE,
                bold=True,
            ),
            f"- Baseline total：`{_format_seconds(comparison.get('baseline_total_sec'))}`",
            f"- Candidate total：`{_format_seconds(comparison.get('candidate_total_sec'))}`",
            f"- Total speedup：`{float(comparison.get('total_speedup_x') or 0.0):.3f}x`",
            f"- Elapsed change：`{float(comparison.get('total_elapsed_change_pct') or 0.0):+.2f}%`",
            "",
            "| Year | Baseline | Candidate | Speedup | Δ time | Contract | Epoch | Model | Scores | Exact |",
            "|---:|---:|---:|---:|---:|---|---|---|---|---|",
        ]
    )
    for row in list(comparison.get("years") or []):
        if row.get("missing_side"):
            fail = markdown_signal("FAIL", SIGNAL_NEGATIVE, bold=True)
            lines.append(
                f"| {row.get('year')} | - | - | - | - | {fail} | {fail} | {fail} | {fail} | {fail} |"
            )
            continue
        lines.append(
            "| {year} | {base} | {cand} | {speed:.3f}x | {delta:+.2f}% | {contract} | {epoch} | {model} | {scores} | {exact} |".format(
                year=int(row["year"]),
                base=_format_seconds(row.get("baseline_sec")),
                cand=_format_seconds(row.get("candidate_sec")),
                speed=float(row.get("speedup_x") or 0.0),
                delta=float(row.get("elapsed_change_pct") or 0.0),
                contract=markdown_signal(
                    "PASS" if row.get("contract_equal") else "FAIL",
                    SIGNAL_POSITIVE if row.get("contract_equal") else SIGNAL_NEGATIVE,
                ),
                epoch=markdown_signal(
                    "PASS" if row.get("selected_epoch_equal") else "FAIL",
                    SIGNAL_POSITIVE if row.get("selected_epoch_equal") else SIGNAL_NEGATIVE,
                ),
                model=markdown_signal(
                    "PASS" if row.get("model_state_equal") else "FAIL",
                    SIGNAL_POSITIVE if row.get("model_state_equal") else SIGNAL_NEGATIVE,
                ),
                scores=markdown_signal(
                    "PASS" if row.get("scores_equal") else "FAIL",
                    SIGNAL_POSITIVE if row.get("scores_equal") else SIGNAL_NEGATIVE,
                ),
                exact=markdown_signal(
                    "PASS" if row.get("exact_result") else "FAIL",
                    SIGNAL_POSITIVE if row.get("exact_result") else SIGNAL_NEGATIVE,
                    bold=True,
                ),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_console_summary(
    *, baseline: dict[str, Any], candidate: dict[str, Any] | None, comparison: dict[str, Any] | None
) -> None:
    color = console_color_enabled()
    benchmark = dict(baseline.get("benchmark") or {})
    print("\n" + render_title("Rolling Training Timing Mode"))
    print(
        render_key_values(
            (
                ("Profile", benchmark.get("experiment_profile")),
                ("Seed", benchmark.get("seed")),
                ("Score years", benchmark.get("score_years")),
                ("Mode", "Extending / from-scratch / isolated artifacts"),
            )
        )
    )
    if candidate is None or comparison is None:
        print(
            terminal_signal(
                "Baseline 已建立；改善程式後用同一 Timing Mode 再跑即可比較。",
                SIGNAL_WARNING,
                enabled=color,
            )
        )
        return

    rows = []
    for row in list(comparison.get("years") or []):
        rows.append(
            (
                str(row.get("year")),
                _format_seconds(row.get("baseline_sec")),
                _format_seconds(row.get("candidate_sec")),
                "-" if row.get("speedup_x") is None else f"{float(row['speedup_x']):.3f}x",
                "-" if row.get("elapsed_change_pct") is None else f"{float(row['elapsed_change_pct']):+.2f}%",
                terminal_signal(
                    "PASS" if row.get("exact_result") else "FAIL",
                    SIGNAL_POSITIVE if row.get("exact_result") else SIGNAL_NEGATIVE,
                    enabled=color,
                ),
            )
        )
    print(render_section("改善前後"))
    print(
        render_table(
            ("Year", "Baseline", "Candidate", "Speedup", "Δ time", "Exact result"),
            rows,
            alignments=("right", "right", "right", "right", "right", "left"),
        )
    )
    print(
        render_key_values(
            (
                ("Baseline total", _format_seconds(comparison.get("baseline_total_sec"))),
                ("Candidate total", _format_seconds(comparison.get("candidate_total_sec"))),
                ("Total speedup", f"{float(comparison.get('total_speedup_x') or 0.0):.3f}x"),
                ("Elapsed change", f"{float(comparison.get('total_elapsed_change_pct') or 0.0):+.2f}%"),
                (
                    "Result equality",
                    terminal_signal(
                        "PASS / bitwise exact" if comparison.get("exact_result") else "FAIL / candidate不可視為結果不變",
                        SIGNAL_POSITIVE if comparison.get("exact_result") else SIGNAL_NEGATIVE,
                        enabled=color,
                    ),
                ),
            )
        )
    )


def run_timing_comparison() -> int:
    timing = get_breakout_quality_rolling_timing_settings()
    root = _timing_root(
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        experiment_profile=timing.experiment_profile,
        seed=timing.seed,
    )
    root.mkdir(parents=True, exist_ok=True)
    baseline_path = _baseline_summary_path(root)
    report_path = _latest_report_path(root)

    if not baseline_path.is_file():
        baseline_run_root = root / "baseline_run"
        if baseline_run_root.exists():
            shutil.rmtree(baseline_run_root)
        baseline = _build_run_summary(role="baseline", run_root=baseline_run_root)
        write_json(baseline_path, baseline)
        report_path.write_text(
            _render_report_markdown(baseline=baseline, candidate=None, comparison=None),
            encoding="utf-8",
        )
        _render_console_summary(baseline=baseline, candidate=None, comparison=None)
        print(
            f"Timing report：{project_relative_display_path(report_path, project_root=PROJECT_ROOT)}"
        )
        return 0

    baseline = _load_json(baseline_path)
    expected_benchmark = _benchmark_payload(timing)
    if dict(baseline.get("benchmark") or {}) != expected_benchmark:
        raise ValueError(
            "Timing baseline與目前config benchmark不一致；請先在Timing Mode使用「重新建立改善前 Baseline」。"
        )
    candidate_run_root = root / "candidate_run"
    if candidate_run_root.exists():
        shutil.rmtree(candidate_run_root)
    candidate = _build_run_summary(role="candidate", run_root=candidate_run_root)
    comparison = _comparison_payload(baseline, candidate)
    candidate["comparison_to_baseline"] = comparison
    write_json(_candidate_summary_path(root), candidate)
    report_path.write_text(
        _render_report_markdown(
            baseline=baseline,
            candidate=candidate,
            comparison=comparison,
        ),
        encoding="utf-8",
    )
    _render_console_summary(
        baseline=baseline,
        candidate=candidate,
        comparison=comparison,
    )
    print(
        f"Timing report：{project_relative_display_path(report_path, project_root=PROJECT_ROOT)}"
    )
    return 0 if bool(comparison.get("exact_result")) else 2


def show_timing_status() -> int:
    timing = get_breakout_quality_rolling_timing_settings()
    root = _timing_root(
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        experiment_profile=timing.experiment_profile,
        seed=timing.seed,
    )
    baseline_path = _baseline_summary_path(root)
    candidate_path = _candidate_summary_path(root)
    report_path = _latest_report_path(root)
    print("\n" + render_title("Rolling Training Timing Mode 設定"))
    print(
        render_key_values(
            (
                ("Profile", timing.experiment_profile),
                ("Seed", timing.seed),
                ("Score years", list(timing.score_years)),
                ("Train window", "Extending / all legal history"),
                ("Resume", False),
                ("Baseline", "READY" if baseline_path.is_file() else "MISSING"),
                ("Candidate", "READY" if candidate_path.is_file() else "MISSING"),
                (
                    "Report",
                    project_relative_display_path(report_path, project_root=PROJECT_ROOT),
                ),
            )
        )
    )
    return 0


def reset_timing_baseline() -> int:
    timing = get_breakout_quality_rolling_timing_settings()
    root = _timing_root(
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        experiment_profile=timing.experiment_profile,
        seed=timing.seed,
    )
    if root.exists():
        shutil.rmtree(root)
    print("Timing baseline 已重設；下次執行會重新建立改善前 baseline。")
    return 0


def show_latest_timing_report() -> int:
    timing = get_breakout_quality_rolling_timing_settings()
    root = _timing_root(
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        experiment_profile=timing.experiment_profile,
        seed=timing.seed,
    )
    report_path = _latest_report_path(root)
    if not report_path.is_file():
        raise FileNotFoundError("尚未建立 Rolling Timing report")
    print(report_path.read_text(encoding="utf-8"))
    return 0


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    action = str(args[0]).strip().lower() if args else "run"
    if len(args) > 1:
        raise ValueError("timing-rolling-training只接受一個action")
    if action in {"run", "compare"}:
        return run_timing_comparison()
    if action == "status":
        return show_timing_status()
    if action in {"reset", "reset-baseline"}:
        return reset_timing_baseline()
    if action in {"report", "show-report"}:
        return show_latest_timing_report()
    if action in {"-h", "--help", "help"}:
        print("用法: timing-rolling-training [run|status|reset-baseline|report]")
        return 0
    raise ValueError(f"不支援的 Timing action: {action}")


__all__ = [
    "TIMING_KIND",
    "TIMING_SCHEMA_VERSION",
    "_comparison_payload",
    "reset_timing_baseline",
    "resolve_rolling_timing_artifact_paths",
    "run_timing_comparison",
    "show_latest_timing_report",
    "show_timing_status",
]

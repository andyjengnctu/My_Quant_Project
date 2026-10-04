from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.display_policy import scale_system_score_for_display
from core.portfolio_stats import (
    calc_score_min_quarter_return_multiplier,
    calc_score_win_rate_multiplier,
)
from core.training_policy import (
    is_score_min_quarter_return_amp_enabled,
    is_score_monthly_win_rate_amp_enabled,
    resolve_score_min_quarter_return_floor,
    resolve_score_min_quarter_return_target,
    resolve_score_monthly_win_rate_target,
)
from services.optimizer.study_utils import INVALID_TRIAL_VALUE, is_qualified_trial_value


P9_WORK_RELATIVE = Path(
    "outputs/optimizer/strategy_param_schedule/experiments/param_p9_validation_plateau_refit"
)
STUDY_NAME = "portfolio_optimization_overnight"
_YEAR_RE = re.compile(r"_(20\d{2})(?:_[^/]*)?\.db$")
_REFIT_RE = re.compile(r"validation_budget_refit_(\d+)")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _full_refit_db_paths(execution_root: Path) -> list[Path]:
    return sorted(
        path
        for path in execution_root.rglob("*.db")
        if any(_REFIT_RE.fullmatch(part) for part in path.parts)
        and "outer_rolling_oos" in path.parts
        and "db" in path.parts
    )




def _candidate_execution_roots(base: Path) -> list[Path]:
    roots = [base]
    attempts = base / "force_rebuild_attempts"
    if attempts.is_dir():
        roots.extend(path for path in attempts.iterdir() if path.is_dir())
    return roots


def _all_full_refit_db_paths(base: Path) -> list[Path]:
    paths: dict[str, Path] = {}
    for root in _candidate_execution_roots(base):
        for path in _full_refit_db_paths(root):
            paths[str(path.resolve())] = path
    return sorted(paths.values(), key=lambda path: path.as_posix())


def _latest_row_per_year(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_year: dict[int, dict[str, Any]] = {}
    unknown: list[dict[str, Any]] = []
    for row in rows:
        year = row.get("oos_year")
        if year is None:
            unknown.append(row)
            continue
        year_int = int(year)
        current = by_year.get(year_int)
        rank = (int(row.get("_mtime_ns") or 0), str(row.get("db") or ""))
        current_rank = (
            int((current or {}).get("_mtime_ns") or 0),
            str((current or {}).get("db") or ""),
        )
        if current is None or rank > current_rank:
            by_year[year_int] = row
    ordered = [by_year[year] for year in sorted(by_year)]
    ordered.extend(sorted(unknown, key=lambda row: str(row.get("db") or "")))
    return ordered


def _parse_year(path: Path) -> int | None:
    match = _YEAR_RE.search(path.name)
    return int(match.group(1)) if match else None


def _resolve_oos_year(path: Path, attrs: dict[str, Any]) -> int | None:
    parsed = _parse_year(path)
    if parsed is not None:
        return parsed
    search_train_end_year = attrs.get("search_train_end_year")
    try:
        resolved = int(search_train_end_year) + 1
    except (TypeError, ValueError):
        return None
    return resolved if 2000 <= resolved <= 2100 else None


def _parse_selected_trials(path: Path) -> int | None:
    for part in path.parts:
        match = _REFIT_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


def _best_trial(study):
    candidates = []
    for trial in list(getattr(study, "trials", []) or []):
        value = getattr(trial, "value", None)
        if value is None or not is_qualified_trial_value(value):
            continue
        candidates.append(trial)
    if not candidates:
        return None
    return max(candidates, key=lambda trial: (float(trial.value), -int(trial.number)))


def _score_breakdown(attrs: dict[str, Any], stored_score: float) -> dict[str, Any]:
    monthly_win_rate = _finite(attrs.get("m_win_rate"))
    min_quarter_return = _finite(attrs.get("min_quarter_return_pct"))

    monthly_enabled = bool(is_score_monthly_win_rate_amp_enabled())
    monthly_target = float(resolve_score_monthly_win_rate_target())
    monthly_multiplier = (
        calc_score_win_rate_multiplier(monthly_win_rate, monthly_target)
        if monthly_enabled
        else 1.0
    )

    quarter_enabled = bool(is_score_min_quarter_return_amp_enabled())
    quarter_floor = float(resolve_score_min_quarter_return_floor())
    quarter_target = float(resolve_score_min_quarter_return_target())
    quarter_multiplier = (
        calc_score_min_quarter_return_multiplier(
            min_quarter_return,
            quarter_floor,
            quarter_target,
        )
        if quarter_enabled
        else 1.0
    )

    if quarter_enabled and quarter_multiplier == 0.0:
        diagnosis = "QUARTER_MULTIPLIER_ZERO"
    elif monthly_enabled and monthly_multiplier <= 0.01:
        diagnosis = "MONTHLY_MULTIPLIER_HEAVY_COMPRESSION"
    elif abs(float(stored_score)) <= 5e-7:
        diagnosis = "ZERO_OR_NEAR_ZERO_OTHER"
    else:
        diagnosis = "NON_ZERO"

    return {
        "portfolio_return_pct": _finite(attrs.get("pf_return")),
        "portfolio_mdd_pct": _finite(attrs.get("pf_mdd")),
        "monthly_win_rate_pct": monthly_win_rate,
        "monthly_multiplier_enabled": monthly_enabled,
        "monthly_multiplier_target_pct": monthly_target,
        "monthly_multiplier": float(monthly_multiplier),
        "min_quarter_return_pct": min_quarter_return,
        "quarter_multiplier_enabled": quarter_enabled,
        "quarter_multiplier_floor_pct": quarter_floor,
        "quarter_multiplier_target_pct": quarter_target,
        "quarter_multiplier": float(quarter_multiplier),
        "stored_base_score": float(stored_score),
        "display_base_score": float(scale_system_score_for_display(stored_score)),
        "diagnosis": diagnosis,
    }


def inspect_db(path: Path) -> dict[str, Any]:
    import optuna

    storage = f"sqlite:///{path.resolve()}"
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    best = _best_trial(study)
    if best is None:
        return {
            "db": _project_relative(path),
            "oos_year": _parse_year(path),
            "selected_trials": _parse_selected_trials(path),
            "status": "NO_QUALIFIED_TRIAL",
            "_mtime_ns": int(path.stat().st_mtime_ns),
        }

    attrs = dict(getattr(best, "user_attrs", {}) or {})
    stored_score = _finite(attrs.get("base_score", best.value), default=float(INVALID_TRIAL_VALUE))
    return {
        "db": _project_relative(path),
        "oos_year": _resolve_oos_year(path, attrs),
        "selected_trials": _parse_selected_trials(path),
        "status": "OK",
        "best_trial": int(best.number) + 1,
        "_mtime_ns": int(path.stat().st_mtime_ns),
        **_score_breakdown(attrs, stored_score),
    }


def _print_rows(rows: list[dict[str, Any]], work_root: Path, scanned_db_count: int) -> None:
    print("=" * 150)
    print("P9 FULL-REFIT BEST BASE SCORE DIAGNOSTIC (READ-ONLY)")
    print("=" * 150)
    print(f"Work root      : {_project_relative(work_root)}")
    print(f"Studies scanned: {int(scanned_db_count)} full-refit DB(s) across canonical + force_rebuild_attempts")
    print(
        "Quarter rule  : multiplier=0 when min quarter <= "
        f"{resolve_score_min_quarter_return_floor():.2f}% "
        f"(target={resolve_score_min_quarter_return_target():.2f}%)"
    )
    print(
        "Monthly rule  : target="
        f"{resolve_score_monthly_win_rate_target():.2f}% | enabled={is_score_monthly_win_rate_amp_enabled()}"
    )
    print("-" * 150)
    print(
        f"{'Year':>4} {'N*':>4} {'Trial':>5} {'Ret%':>9} {'MDD%':>8} "
        f"{'MonWin%':>9} {'MonMult':>10} {'MinQ%':>9} {'QMult':>10} "
        f"{'Base(raw)':>13} {'Base(display)':>14}  Diagnosis"
    )
    print("-" * 150)
    for row in sorted(rows, key=lambda item: (item.get("oos_year") or 9999, item.get("db") or "")):
        if row.get("status") != "OK":
            print(
                f"{str(row.get('oos_year') or '-'):>4} {str(row.get('selected_trials') or '-'):>4} "
                f"{'-':>5} {'-':>9} {'-':>8} {'-':>9} {'-':>10} {'-':>9} {'-':>10} "
                f"{'-':>13} {'-':>14}  {row.get('status')}"
            )
            continue
        year_text = str(row.get("oos_year") if row.get("oos_year") is not None else "-")
        selected_text = str(row.get("selected_trials") if row.get("selected_trials") is not None else "-")
        trial_text = str(row.get("best_trial") if row.get("best_trial") is not None else "-")
        print(
            f"{year_text:>4} {selected_text:>4} {trial_text:>5} "
            f"{float(row['portfolio_return_pct']):>9.3f} {float(row['portfolio_mdd_pct']):>8.3f} "
            f"{float(row['monthly_win_rate_pct']):>9.3f} {float(row['monthly_multiplier']):>10.6f} "
            f"{float(row['min_quarter_return_pct']):>9.3f} {float(row['quarter_multiplier']):>10.6f} "
            f"{float(row['stored_base_score']):>13.9f} {float(row['display_base_score']):>14.6f}  "
            f"{row['diagnosis']}"
        )
    print("=" * 150)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read existing PARAM-P9 full-refit Optuna SQLite studies and explain why best_base is zero/near-zero. "
            "This tool is read-only and does not retrain or mutate research artifacts."
        )
    )
    parser.add_argument(
        "--work-root",
        default=str(P9_WORK_RELATIVE),
        help="P9 optimizer work root relative to project root (default: canonical PARAM-P9 workspace).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional JSON output path relative to project root.",
    )
    args = parser.parse_args()

    work_root = Path(args.work_root)
    if not work_root.is_absolute():
        work_root = PROJECT_ROOT / work_root
    if not work_root.exists():
        print(f"[錯誤] 找不到P9 workspace: {_project_relative(work_root)}")
        return 2

    db_paths = _all_full_refit_db_paths(work_root)
    if not db_paths:
        print(
            "[錯誤] 找不到PARAM-P9 full-refit SQLite studies。"
            f"檢查位置: {_project_relative(work_root)}"
        )
        return 3

    inspected_rows = [inspect_db(path) for path in db_paths]
    rows = _latest_row_per_year(inspected_rows)
    _print_rows(rows, work_root, len(db_paths))

    if args.json_out:
        output = Path(args.json_out)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now().astimezone().isoformat(),
            "work_root": _project_relative(work_root),
            "studies_scanned": int(len(db_paths)),
            "rows": [{key: value for key, value in row.items() if key != "_mtime_ns"} for row in rows],
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON: {_project_relative(output)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One-off controlled repair of missing FIT provenance, not a model-selection workflow.

Uses the existing read-only audit to inspect every configured model first, then
hands approved work to canonical Rolling and Forward producers through Research.
No legacy checkpoint is relabelled as a newly verified fitted model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.breakout_quality_policy import (
    get_breakout_quality_model_test_settings, get_breakout_quality_workflow_settings,
)
from services.breakout_quality.fit_repair_transaction import recover_incomplete_fit_repair
from services.breakout_quality.fit_origin_repair_plan import plan_fit_origin_repairs
from tools.audit.breakout_quality.fit_lineage_snapshot import _inspect_profile


def collect_repair_plan(root: Path) -> tuple[list[dict], list[dict]]:
    """Inventory the entire shared comparison list before any producer is invoked."""
    model_profiles = get_breakout_quality_model_test_settings().model_profiles
    records = [
        _inspect_profile(root, str(model_id), str(profile), include_torch=False)
        for model_id, profile in model_profiles
    ]
    candidates, _blocked = plan_fit_origin_repairs(
        project_root=root, model_profiles=model_profiles,
    )
    jobs = [{
        "model_id": item.model_id, "profile": item.profile,
        "old_sha256": item.previous_checkpoint_sha256,
    } for item in candidates]
    return records, jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-pass FIT repair plan for all configured model profiles")
    parser.add_argument("--execute", action="store_true", help="run approved canonical Rolling and Forward repairs")
    parser.add_argument("--recover", action="store_true", help="restore an interrupted repair from its journal")
    args = parser.parse_args(argv)
    root = _PROJECT_ROOT.resolve()
    print("\nFitted-model provenance | all configured models | engineering repair")
    if args.recover:
        recovered = 0
        for _model_id, profile in get_breakout_quality_model_test_settings().model_profiles:
            settings = get_breakout_quality_workflow_settings(experiment_profile=str(profile))
            journal = (root / "outputs" / "filters" / "breakout_quality"
                       / str(settings.filter_id) / "audit" / "fit_repair"
                       / (str(profile) + ".in_progress.json"))
            if journal.is_file():
                recover_incomplete_fit_repair(journal_path=journal)
                print(f"Recovered original publications for {_model_id}")
                recovered += 1
        print(f"FIT recovery complete: {recovered} pending transaction(s)")
        return 0
    records, jobs = collect_repair_plan(root)
    for record in records:
        blockers = ", ".join(record["blockers"]) or "no structural evidence gap detected"
        print(f"{record['model_id']}: {record['classification']} | {blockers}")
    known = {job["profile"] for job in jobs}
    unsupported = [record for record in records if record["blockers"] and record["profile"] not in known]
    if unsupported:
        print("[BLOCKED] Some models have unresolved/corrupt evidence outside the controlled rebuild policy.")
        return 2
    if not jobs:
        print("No known missing-origin rebuild is required; readiness still needs canonical FIT validation.")
        return 0
    if not args.execute:
        for job in jobs:
            print(f"REBUILD REQUIRED: {job['model_id']} | old checkpoint={job['old_sha256'][:16]}… | old publications preserved")
        print("Read-only plan complete; use --execute for explicit, rollback-protected canonical rebuilding.")
        return 0

    from services.research.breakout_quality_application import repair_missing_fit_origin_from_canonical_rolling
    for job in jobs:
        print(f"[CANONICAL FIT REPAIR] {job['model_id']} | Rolling FIT then verified Forward reuse")
        result = repair_missing_fit_origin_from_canonical_rolling(
            "apps/research.py model",
            profile_name=job["profile"],
            expected_previous_checkpoint_sha256=job["old_sha256"],
        )
        print(f"[REPAIR VERIFIED] {result['model_id']} | FIT SHA={result['repaired_checkpoint_sha256']}")
    print("Repair committed. Historical model/evaluation publications are retained under historical_fit_repairs.")
    print("Run the standard Research Model Compare menu to independently verify all profiles and publish the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

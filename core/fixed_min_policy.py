"""Single scientific source of truth for the non-optimized Fixed Min treatment.

These are prespecified constants, not parameters selected from 2013-2026
performance. The exact same object feeds the training Target and Portfolio
Replay. No yearly, fold, or replication-seed-dependent parameter values.
"""
from __future__ import annotations

from pathlib import Path

from core.file_integrity import canonical_json_sha256
from core.params_io import params_to_json_dict
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.strategy_params import V16StrategyParams

FIXED_MIN_ID = "fixed_min_constant_v1"
FIXED_MIN_TARGET = "FixedMinTrade_NetPnL"
FIXED_MIN_FIRST_EFFECTIVE_DATE = "1900-01-01"  # timeless, predetermined constants
FIXED_MIN_RESEARCH_END_DATE = "2026-03-02"
FIXED_MIN_PARAM_RELATIVE_PATH = (
    "models/research/breakout_quality/strategy_params/fixed_min_constant_v1/roos_base_best.json"
)
FIXED_MIN_MANIFEST_RELATIVE_PATH = (
    "models/research/breakout_quality/strategy_params/fixed_min_constant_v1/producer_manifest.json"
)


def fixed_min_params() -> V16StrategyParams:
    return V16StrategyParams(
        high_len=200, atr_len=14, atr_buy_tol=1.5,
        atr_times_init=2.0, atr_times_trail=3.5,
        tp_percent=0.0, use_breakout_buy=True,
        use_breakout_ema_filter=False, use_bb=False, use_kc=False,
        use_vol=False, use_breakout_return_filter=False,
        use_breakout_false_filter=False, use_breakout_reclaim_reentry=False,
        use_breakout_quality_filter=False, use_breakout_quality_ranking=False,
        use_history_threshold=False,
    )


def fixed_min_signature() -> str:
    return build_portfolio_params_signature(fixed_min_params())


def fixed_min_scientific_sha256() -> str:
    return canonical_json_sha256({
        "contract": FIXED_MIN_ID,
        "params_signature": fixed_min_signature(),
        "first_effective_date": FIXED_MIN_FIRST_EFFECTIVE_DATE,
        "research_end_date": FIXED_MIN_RESEARCH_END_DATE,
    })



def verify_fixed_min_source(source: object) -> bool:
    """Verify a Fixed Min PIT teacher without importing the service that builds it.

    The canonical source is fixed parameters for all time; only the FIT seed may
    differ. This policy-layer validator must not depend on the PIT source class.
    """
    if source.source_id != FIXED_MIN_ID:
        return False
    if (source.param_sha256 != fixed_min_scientific_sha256()
            or source.publication_dates != (FIXED_MIN_FIRST_EFFECTIVE_DATE,)
            or source.oos_start_date is not None
            or len(source.schedule) != 1):
        return False
    period = source.schedule[0]
    return bool(
        period.source_id == FIXED_MIN_ID
        and str(period.start.date()) == FIXED_MIN_FIRST_EFFECTIVE_DATE
        and period.params_signature == fixed_min_signature()
        and build_portfolio_params_signature(period.params) == fixed_min_signature()
    )


def fixed_min_param_payload() -> dict:
    params = params_to_json_dict(fixed_min_params())
    return {
        "schema_type": "active_param_ensemble",
        "schema_version": 1,
        "mode": "rolling",
        "selector": "base_finalist_best",
        "fixed_min_contract_version": 1,
        "fixed_min_contract_id": FIXED_MIN_ID,
        "random_seed_ensemble": {"seed_count": 1, "min_agree": 1},
        "params_ensemble_by_effective_date": {
            FIXED_MIN_FIRST_EFFECTIVE_DATE: [{"member_index": 1, "params": params}],
        },
        "params_by_effective_date": {FIXED_MIN_FIRST_EFFECTIVE_DATE: params},
        "folds": [{"effective_start": FIXED_MIN_FIRST_EFFECTIVE_DATE,
                   "effective_end": FIXED_MIN_RESEARCH_END_DATE,
                   "oos_start_date": FIXED_MIN_FIRST_EFFECTIVE_DATE,
                   "oos_end_date": FIXED_MIN_RESEARCH_END_DATE}],
        "meta": {"first_oos_date": FIXED_MIN_FIRST_EFFECTIVE_DATE,
                 "last_oos_date": FIXED_MIN_RESEARCH_END_DATE,
                 "fixed_min_scientific_sha256": fixed_min_scientific_sha256()},
        "summary": {"oos_end_date": FIXED_MIN_RESEARCH_END_DATE,
                    "folds": 1,
                    "oos_period": f"{FIXED_MIN_FIRST_EFFECTIVE_DATE}~{FIXED_MIN_RESEARCH_END_DATE}"},
    }


def fixed_min_artifact_path(project_root: Path) -> Path:
    return Path(project_root).resolve() / FIXED_MIN_PARAM_RELATIVE_PATH


def ensure_fixed_min_param_artifact(project_root: Path) -> Path:
    """Publish the deterministic parameter source; never overwrite drifted evidence."""
    import json
    path = fixed_min_artifact_path(project_root)
    canonical = fixed_min_param_payload()
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("Fixed Min parameter artifact cannot be verified") from exc
        if existing != canonical:
            raise ValueError("Fixed Min parameter artifact differs from canonical constants")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing an unrelated artifact even under races.
    try:
        with path.open("x", encoding="utf-8") as f:
            json.dump(canonical, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
    except FileExistsError:
        return ensure_fixed_min_param_artifact(project_root)
    return path



def fixed_min_producer_manifest_path(project_root: Path) -> Path:
    return Path(project_root).resolve() / FIXED_MIN_MANIFEST_RELATIVE_PATH


def _fixed_min_producer_manifest_payload(path: Path) -> dict:
    from core.strategy_param_artifacts import compute_strategy_param_scientific_sha256
    from core.file_integrity import compute_file_sha256

    return {
        "schema_type": "fixed_min_parameter_producer_manifest",
        "schema_version": 1,
        "source_id": FIXED_MIN_ID,
        "parameter_path": FIXED_MIN_PARAM_RELATIVE_PATH,
        "fixed_min_contract_sha256": fixed_min_scientific_sha256(),
        "parameter_scientific_sha256": compute_strategy_param_scientific_sha256(path),
        "parameter_file_sha256": compute_file_sha256(path),
    }


def inspect_fixed_min_robustness_parameter_artifact(project_root: Path) -> dict:
    """Inspect the immutable, seed-independent source without producing work."""
    import json

    path = fixed_min_artifact_path(project_root)
    manifest_path = fixed_min_producer_manifest_path(project_root)
    if not path.exists():
        return {"action": "BLOCKED" if manifest_path.exists() else "BUILD",
                "status": "SOURCE_MISSING_WITH_MANIFEST" if manifest_path.exists()
                          else "FIXED_MIN_SOURCE_MISSING"}
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"action": "BLOCKED", "status": "FIXED_MIN_SOURCE_UNREADABLE"}
    if actual != fixed_min_param_payload():
        return {"action": "BLOCKED", "status": "FIXED_MIN_CONSTANT_DRIFT"}
    if not manifest_path.exists():
        return {"action": "REPAIR", "status": "FIXED_MIN_MANIFEST_MISSING"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"action": "BLOCKED", "status": "FIXED_MIN_MANIFEST_UNREADABLE"}
    if manifest != _fixed_min_producer_manifest_payload(path):
        return {"action": "BLOCKED", "status": "FIXED_MIN_MANIFEST_DRIFT"}
    return {"action": "REUSE", "status": "READY"}


def ensure_fixed_min_robustness_parameter_artifact(project_root: Path) -> dict:
    """Publish metadata for the single fixed source, never seed-specific params."""
    import json

    before = inspect_fixed_min_robustness_parameter_artifact(project_root)
    if before["action"] == "BLOCKED":
        raise ValueError(f"Fixed Min benchmark source drift: {before['status']}")
    path = ensure_fixed_min_param_artifact(project_root)
    manifest_path = fixed_min_producer_manifest_path(project_root)
    expected_manifest = _fixed_min_producer_manifest_payload(path)
    if not manifest_path.exists():
        # Exclusive create prevents replacement of any registered manifest.
        try:
            with manifest_path.open("x", encoding="utf-8") as f:
                json.dump(expected_manifest, f, ensure_ascii=False, sort_keys=True, indent=2)
                f.write("\n")
        except FileExistsError as exc:
            # Another process may have published the same manifest concurrently.
            # Accept only an independently verified exact canonical publication.
            concurrent = inspect_fixed_min_robustness_parameter_artifact(project_root)
            if concurrent["action"] != "REUSE":
                raise ValueError(
                    f"Fixed Min concurrent manifest publication invalid: {concurrent['status']}"
                ) from exc
    state = inspect_fixed_min_robustness_parameter_artifact(project_root)
    if state["action"] != "REUSE":
        raise ValueError(f"Fixed Min benchmark publication invalid: {state['status']}")
    return {
        "path": path,
        "manifest_path": manifest_path,
        "sha256": expected_manifest["parameter_file_sha256"],
        "scientific_sha256": expected_manifest["parameter_scientific_sha256"],
        "action": "REUSE" if before["action"] == "REUSE" else before["action"],
    }

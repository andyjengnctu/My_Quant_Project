"""Canonical Optimizer evidence adapter for formal MinTrade40 FIT labels.

The loader is read-only. Historical source construction is separately guarded
and delegated only to the canonical Optimizer when explicitly authorized.
Neither path can certify a synthetic or retired P2 teacher based on the mere
existence of a plausible parameter JSON.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from core.file_integrity import canonical_json_sha256, load_json_strict
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.strategy_param_artifacts import compute_strategy_param_file_sha256
from filters.breakout_quality.daily_trade40_target import (
    build_formal_min_execution_schedule,
    resolve_formal_min_param_records,
)
from services.breakout_quality.formal_mintrade40_pit_source import (
    FormalMinTrade40PITBlocked,
    FormalMinTrade40PITSource,
    slice_formal_mintrade40_pit_source_for_fold,
)
from services.optimizer.strategy_param_service import _canonical_schedule_build_contract
from services.optimizer.strategy_param_training import (
    MIN_ROOS_SEARCH_FIELDS, _build_min_roos_schedule_contract,
)
from filters.breakout_quality.trade_path_label import (
    TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
    TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
)


# This is a *candidate* historical training-source namespace, not an existing
# PARAM-P16 physical artifact or an authorization to introduce a model/arm.
HISTORICAL_MINTRADE40_TRAINING_SET = "MINTRADE40_P16_HISTORICAL_LABEL_V1"


def formal_mintrade40_history_params_path(project_root: str | Path, seed: int) -> Path:
    """One optimizer-owned publication path for a single historical label seed."""
    from filters.breakout_quality.strategy_compare_sources import (
        PARAM_POLICY_SPECS, PARAM_POLICY_BASE_FINALIST_BEST,
    )

    if type(seed) is not int or seed <= 0:
        raise FormalMinTrade40PITBlocked("Historical MinTrade40 optimizer seed invalid")
    filename = str(PARAM_POLICY_SPECS[PARAM_POLICY_BASE_FINALIST_BEST]["filename"])
    return (Path(project_root).resolve() / "outputs" / "optimizer"
            / "mintrade40_label_history" / f"seed_{seed}" / "p2_dl_off_trained"
            / "active_params" / filename)


def _expected_historical_runtime_identity(project_root: Path) -> str:
    """Canonical optimizer fingerprint with a strictly read-only source inventory.

    The producer uses ``build_source_data_inventory`` to persist its normal
    content witness. A preflight must not perform this write, but still compare
    the identical inventory and all optimizer contract fields.
    """
    from config.execution_policy import (
        DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT,
        DEFAULT_PORTFOLIO_MAX_POSITIONS, DEFAULT_PORTFOLIO_ROTATION,
    )
    from filters.breakout_quality.source_inventory import _build_source_data_inventory
    from filters.breakout_quality.trade_path_label import TRADE_PATH_BASE_FILTER_ID
    from services.optimizer.strategy_param_training import (
        _runtime_contract, build_min_roos_fold_overrides,
    )
    policy = _canonical_schedule_build_contract(family="min")
    reference = _build_min_roos_schedule_contract(
        first_oos_date=TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
        last_oos_date=TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
        train_window_months=int(policy["train_window_months"]),
        oos_months=int(policy["oos_horizon_months"]),
    )
    args = SimpleNamespace(
        dataset=str(policy["dataset"]), filter_id=TRADE_PATH_BASE_FILTER_ID,
        trials_per_fold=int(policy["trials_per_fold"]),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
    )
    expected = _runtime_contract(
        root=project_root, args=args, baseline_contract=reference,
        fold_overrides=build_min_roos_fold_overrides(
            baseline_contract=reference, args=args, training_dl_enabled=False,
        ),
        model_artifact=None, arm_id=HISTORICAL_MINTRADE40_TRAINING_SET,
        training_dl_enabled=False, binary_pit=None,
        dataset_inventory=_build_source_data_inventory(project_root, args.dataset),
    )
    return str(expected["runtime_identity_sha256"])


def inspect_formal_mintrade40_optimizer_teacher(
    *, project_root: str | Path, seed: int,
) -> dict[str, object]:
    """Read-only exact-source REUSE/BUILD/RESUME/BLOCKED for historical Min.

    Completed and interrupted sources must match today's *original* optimizer
    runtime fingerprint, including raw data inventory, search, policy and dates.
    A partial run without a matching immutable preflight is never overwritten.
    The canonical optimizer alone decides which matching folds can resume.
    """
    root = Path(project_root).resolve()
    path = formal_mintrade40_history_params_path(root, seed)
    arm_dir = path.parent.parent
    completed = path.is_file()
    preflight_path = arm_dir / "rolling_preflight.json"
    other_files = any(p.is_file() for p in arm_dir.rglob("*")) if arm_dir.is_dir() else False
    if not completed and not preflight_path.is_file():
        return {"path": path, "action": "BLOCKED" if other_files else "BUILD",
                "reason": (
                    "Interrupted optimizer has files but no original preflight; preserve and inspect"
                    if other_files else "Missing original seed-specific 2014-2020 Optimizer evidence"
                )}
    try:
        prior = load_json_strict(preflight_path)
        if not isinstance(prior, dict):
            raise FormalMinTrade40PITBlocked("Optimizer preflight is not a JSON object")
        original_fingerprint = str(prior.get("runtime_identity_sha256") or "")
        if not original_fingerprint or original_fingerprint != _expected_historical_runtime_identity(root):
            raise FormalMinTrade40PITBlocked(
                "Original optimizer source/parameter/period/runtime identity does not match"
            )
        # A partial preflight is not enough if its fields have been tampered.
        unsigned = {key: value for key, value in prior.items()
                    if key not in {"runtime_identity_sha256", "created_at", "status",
                                   "binary_pit_optimizer_coverage_path"}}
        if canonical_json_sha256(unsigned) != original_fingerprint:
            raise FormalMinTrade40PITBlocked("Optimizer preflight payload fingerprint differs")
        if completed:
            load_formal_mintrade40_optimizer_teacher(path, expected_seed=seed)
            return {"path": path, "action": "REUSE",
                    "reason": "Original same-seed 2014-2020 Optimizer evidence and current source identity verified"}
        summary_path = arm_dir / "rolling_optimizer_summary.json"
        if summary_path.exists():
            summary = load_json_strict(summary_path)
            if not isinstance(summary, dict) or str(summary.get("status")) == "COMPLETED":
                raise FormalMinTrade40PITBlocked("Optimizer summary completed but parameter publication missing")
    except (FormalMinTrade40PITBlocked, OSError, ValueError, KeyError, TypeError) as exc:
        return {"path": path, "action": "BLOCKED",
                "reason": f"Historical optimizer evidence unverified; preserve bytes: {exc}"}
    return {"path": path, "action": "RESUME",
            "reason": "Original runtime identity verified; canonical optimizer resumes only missing trials/folds"}


def load_formal_mintrade40_optimizer_teacher(
    params_path: str | Path,
    *,
    expected_seed: int,
    expected_parameter_set: str = HISTORICAL_MINTRADE40_TRAINING_SET,
) -> FormalMinTrade40PITSource:
    """Verify original per-fold Optimizer evidence and derive as-of dates.

    The three original artifacts are produced by ONE canonical Optimizer run:
    active_params/*.json, rolling_preflight.json and rolling_optimizer_summary.json.
    A model's Label Owner never infers publication dates from a file mtime or
    changes a fold's effective date to fill missing early training history.

    This is source *candidate* verification only. It does not by itself verify
    every historical CSV's price coverage or authorize the downstream model FIT.
    """
    path = Path(params_path).resolve()
    if (not path.is_file() or path.parent.name != "active_params"
            or path.parent.parent.name != "p2_dl_off_trained"
            or path.parent.parent.parent.name != f"seed_{expected_seed}"
            or path.parent.parent.parent.parent.name != "mintrade40_label_history"):
        raise FormalMinTrade40PITBlocked("Canonical historical Min Optimizer source path/seed invalid")
    candidate_dir = path.parent.parent
    try:
        params = load_json_strict(path)
        preflight = load_json_strict(candidate_dir / "rolling_preflight.json")
        summary = load_json_strict(candidate_dir / "rolling_optimizer_summary.json")
    except (OSError, ValueError, TypeError) as exc:
        raise FormalMinTrade40PITBlocked(
            f"Missing or corrupt canonical Optimizer provenance: {type(exc).__name__}: {exc}"
        ) from exc
    if not all(isinstance(item, dict) for item in (params, preflight, summary)):
        raise FormalMinTrade40PITBlocked("Canonical Optimizer provenance must contain objects")
    if type(expected_seed) is not int or expected_seed <= 0:
        raise FormalMinTrade40PITBlocked("Invalid seed-specific Min teacher identity")
    if expected_parameter_set != HISTORICAL_MINTRADE40_TRAINING_SET:
        raise FormalMinTrade40PITBlocked("Retired P2/P16 or unregistered historical teacher rejected")

    training_policy = _canonical_schedule_build_contract(family="min")
    reference = _build_min_roos_schedule_contract(
        first_oos_date=TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
        last_oos_date=TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
        train_window_months=int(training_policy["train_window_months"]),
        oos_months=int(training_policy["oos_horizon_months"]),
    )
    adaptation = dict(params.get("breakout_quality_param_adaptation") or {})
    meta = dict(params.get("meta") or {})
    if any((
        adaptation.get("mode") != "min_roos_training",
        adaptation.get("parameter_set") != expected_parameter_set,
        adaptation.get("training_dl_enabled") is not False,
        adaptation.get("fixed_rule_contract") != "all_rule_filters_off",
        tuple(adaptation.get("search_fields") or ()) != tuple(MIN_ROOS_SEARCH_FIELDS),
        preflight.get("mode") != "binary_dl_filter_min_roos_param_adaptation",
        preflight.get("arm_id") != expected_parameter_set,
        preflight.get("training_dl_enabled") is not False,
        preflight.get("dataset") != "full",
        preflight.get("binary_pit") is not None,
        preflight.get("binary_runtime") is not None,
        str(preflight.get("runtime_identity_sha256") or "") != str(summary.get("runtime_identity_sha256") or ""),
        not str(preflight.get("runtime_identity_sha256") or ""),
        str(summary.get("status") or "") != "COMPLETED",
        str(summary.get("parameter_set") or "") != expected_parameter_set,
        str(meta.get("train_window_months")) != str(training_policy["train_window_months"]),
        str(meta.get("oos_horizon_months")) != str(training_policy["oos_horizon_months"]),
        str(meta.get("trials_per_fold")) != str(training_policy["trials_per_fold"]),
        str(preflight.get("trials_per_fold")) != str(training_policy["trials_per_fold"]),
        tuple(preflight.get("search_fields") or ()) != tuple(MIN_ROOS_SEARCH_FIELDS),
        str((preflight.get("rolling_policy") or {}).get("train_window_months")) != str(training_policy["train_window_months"]),
        str((preflight.get("rolling_policy") or {}).get("oos_horizon_months")) != str(training_policy["oos_horizon_months"]),
    )):
        raise FormalMinTrade40PITBlocked("Optimizer training policy/provenance differs from canonical P16 Min history")
    preflight_contract = {
        key: value for key, value in preflight.items()
        if key not in {
            "runtime_identity_sha256", "created_at", "status",
            "binary_pit_optimizer_coverage_path",
        }
    }
    if canonical_json_sha256(preflight_contract) != str(preflight["runtime_identity_sha256"]):
        raise FormalMinTrade40PITBlocked("Optimizer runtime contract fingerprint mismatch")
    reference_identity = dict(preflight.get("parameter_reference") or {})
    if (reference_identity.get("type") != reference["reference_type"]
            or reference_identity.get("path") is not None
            or reference_identity.get("sha256") != reference["sha256"]):
        raise FormalMinTrade40PITBlocked("Optimizer historical Min reference identity invalid")
    rolling_policy = dict(preflight.get("rolling_policy") or {})
    if any(rolling_policy.get(key) != value for key, value in reference["meta"].items()):
        raise FormalMinTrade40PITBlocked("Optimizer historical rolling policy identity invalid")
    if not isinstance(preflight.get("dataset_identity"), dict):
        raise FormalMinTrade40PITBlocked("Optimizer raw dataset inventory identity missing")
    from core.file_integrity import is_sha256_hex
    if not is_sha256_hex(preflight["dataset_identity"].get("csv_inventory_sha256")):
        raise FormalMinTrade40PITBlocked("Optimizer raw dataset inventory SHA256 invalid")

    try:
        records = resolve_formal_min_param_records("rolling_active_param_ensemble", params)
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalMinTrade40PITBlocked("Canonical Optimizer schedule invalid") from exc
    folds = list(params.get("folds") or ())
    if not records or len(folds) != len(records):
        raise FormalMinTrade40PITBlocked("Optimizer fold provenance incomplete")
    if int(dict(params.get("summary") or {}).get("folds", len(folds))) != len(folds):
        raise FormalMinTrade40PITBlocked("Optimizer fold summary does not match schedule")
    first_period = pd.Timestamp(TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE).to_period("M")
    last_period = pd.Timestamp(TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE).to_period("M")
    horizon_months = int(training_policy["oos_horizon_months"])
    expected_starts = pd.period_range(first_period, last_period, freq="M")[::horizon_months]
    if tuple(pd.Timestamp(row["effective_date"]).to_period("M") for row in records) != tuple(expected_starts):
        raise FormalMinTrade40PITBlocked("Optimizer history does not cover canonical pre-OOS Selection period")
    if (str(meta.get("first_oos_date"))[:7] != str(first_period)
            or str(meta.get("last_oos_date"))[:7] != str(last_period)):
        raise FormalMinTrade40PITBlocked("Optimizer historical coverage metadata mismatch")
    fold_by_effective: dict[pd.Timestamp, dict] = {}
    for fold in folds:
        if not isinstance(fold, dict):
            raise FormalMinTrade40PITBlocked("Malformed Optimizer fold provenance")
        try:
            effective = pd.Timestamp(fold["oos_start_date"]).normalize()
        except (KeyError, TypeError, ValueError) as exc:
            raise FormalMinTrade40PITBlocked("Optimizer fold effective date missing") from exc
        if effective in fold_by_effective:
            raise FormalMinTrade40PITBlocked("Duplicated historical Optimizer fold")
        fold_by_effective[effective] = fold

    publication_dates: list[str] = []
    months = int(training_policy["train_window_months"])
    for row in records:
        effective = pd.Timestamp(row["effective_date"]).normalize()
        fold = fold_by_effective.get(effective)
        if fold is None or len(list(row.get("members") or ())) != 1:
            raise FormalMinTrade40PITBlocked("Optimizer fold or single-seed Min P16 member missing")
        member = row["members"][0]
        if (type(member.get("seed")) is not int or member["seed"] != expected_seed
                or ("optimizer_seed" in member and (
                    type(member["optimizer_seed"]) is not int or member["optimizer_seed"] != expected_seed))
                or ("optimizer_seed" in fold and (
                    type(fold["optimizer_seed"]) is not int or fold["optimizer_seed"] != expected_seed))):
            raise FormalMinTrade40PITBlocked("Seed-specific Optimizer fold/member does not match requested seed")
        if build_portfolio_params_signature(row["params_obj"]) != row["params_signature"]:
            raise FormalMinTrade40PITBlocked("Optimizer member parameter signature invalid")
        try:
            train_start = pd.Timestamp(fold["selection_start_date"]).normalize()
            train_end = pd.Timestamp(fold["selection_end_date"]).normalize()
        except (KeyError, TypeError, ValueError) as exc:
            raise FormalMinTrade40PITBlocked("Optimizer historical training dates missing") from exc
        if (train_start != (effective - pd.DateOffset(months=months)).normalize()
                or train_end != effective - pd.Timedelta(days=1)):
            raise FormalMinTrade40PITBlocked("Optimizer fold train/validation history is not PIT-aligned")
        # Published here means as-of valid, *not* a claim that a backtest was
        # physically published in its historical year.
        publication_dates.append(str(train_end.date()))

    digest = compute_strategy_param_file_sha256(path)
    schedule, _provenance = build_formal_min_execution_schedule(
        records, seed=expected_seed, param_sha256=digest,
    )
    # Verify the same production parameter fields must be DL-off; the normal
    # formal-source validator also verifies signatures and sample coverage.
    if any(p.params.use_breakout_quality_filter or p.params.use_breakout_quality_ranking for p in schedule):
        raise FormalMinTrade40PITBlocked("Historical optimizer must use DL-off Min parameters")
    return FormalMinTrade40PITSource(
        seed=expected_seed,
        param_sha256=digest,
        schedule=schedule,
        publication_dates=tuple(publication_dates),
    )



def load_formal_mintrade40_optimizer_fold_teacher(
    params_path: str | Path,
    *,
    expected_seed: int,
    fit_cutoff: str | pd.Timestamp,
) -> tuple[FormalMinTrade40PITSource, str]:
    """Attest original Optimizer bytes, then derive a FIT-local PIT snapshot.

    The consumer must use this verified loader rather than creating a source
    from parameter dates, file existence, or the later full 2014-2020 schedule.
    This is read-only; no historical parameter optimization is performed here.
    """
    source = load_formal_mintrade40_optimizer_teacher(
        params_path, expected_seed=expected_seed,
    )
    return slice_formal_mintrade40_pit_source_for_fold(
        source, fit_cutoff=fit_cutoff,
    )


def prepare_formal_mintrade40_optimizer_teacher(
    *, project_root: str | Path, seed: int, authorized: bool = False,
) -> FormalMinTrade40PITSource:
    """Build/reuse historical Min labels' P16-like schedule with ONE Optimizer.

    No consumer calls this unless authorized by a future formal MR model work
    plan. Its 2014-2020 historical *training* evidence is separate from the
    immutable 2021+ P16 evaluation schedule, which it never reads or modifies.
    The fixed 120-month training horizon, trial budget and search fields are
    delegated to the existing canonical Optimizer owner.
    """
    if not authorized:
        raise FormalMinTrade40PITBlocked(
            "Historical MinTrade40 optimizer build requires model-work authorization"
        )
    if type(seed) is not int or seed <= 0:
        raise FormalMinTrade40PITBlocked("Historical optimizer seed invalid")
    from core.dataset_profiles import DEFAULT_DATASET_PROFILE
    from config.execution_policy import (
        DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT,
        DEFAULT_PORTFOLIO_MAX_POSITIONS, DEFAULT_PORTFOLIO_ROTATION,
    )
    from services.optimizer.strategy_param_training import prepare_selection_historical_p2_params
    from filters.breakout_quality.strategy_compare_sources import PARAM_POLICY_BASE_FINALIST_BEST

    root = Path(project_root).resolve()
    policy = _canonical_schedule_build_contract(family="min")
    if str(policy["dataset"]) != str(DEFAULT_DATASET_PROFILE):
        raise FormalMinTrade40PITBlocked("Historical Optimizer dataset/profile drift")
    expected = formal_mintrade40_history_params_path(root, seed)
    work = expected.parent.parent.parent
    # If the old source exists but is invalid, preserve it. Never silently
    # recalculate a different identity over an existing scientific artifact.
    status = inspect_formal_mintrade40_optimizer_teacher(project_root=root, seed=seed)
    if status["action"] == "BLOCKED":
        raise FormalMinTrade40PITBlocked(str(status["reason"]))
    if status["action"] == "REUSE":
        return load_formal_mintrade40_optimizer_teacher(expected, expected_seed=seed)
    result = prepare_selection_historical_p2_params(
        project_root=root,
        dataset=str(policy["dataset"]),
        param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
        trials_per_fold=int(policy["trials_per_fold"]),
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        optimizer_seed=seed,
        resume_parameter_training=True,
        quiet=False,
        first_oos_date=TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
        last_oos_date=TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
        train_window_months=int(policy["train_window_months"]),
        oos_months=int(policy["oos_horizon_months"]),
        output_relative_dir=work,
        build_context_label="Historical MinTrade40 PIT label source",
        optimizer_output_title="Historical MinTrade40 Optimizer evidence",
        recover_completed_strategy_compare=False,
        parameter_set_identity=HISTORICAL_MINTRADE40_TRAINING_SET,
    )
    produced = Path(result["params_path"]).resolve()
    if produced != expected.resolve():
        raise FormalMinTrade40PITBlocked("Canonical Optimizer published unexpected teacher path")
    return load_formal_mintrade40_optimizer_teacher(produced, expected_seed=seed)


__all__ = [
    "HISTORICAL_MINTRADE40_TRAINING_SET", "load_formal_mintrade40_optimizer_teacher",
    "load_formal_mintrade40_optimizer_fold_teacher",
    "prepare_formal_mintrade40_optimizer_teacher", "formal_mintrade40_history_params_path",
    "inspect_formal_mintrade40_optimizer_teacher",
]

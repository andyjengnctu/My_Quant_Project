"""Canonical Full/P17 stock-day NetPnL teacher for independent rank and sign DLs.

The separate historical Full Optimizer and current P17 parameter lineage are
mandatory.  Neither a Min/P16 teacher nor hindsight Oracle labels may enter FIT.
Only the canonical Optimizer writes parameters; this adapter verifies and
materializes PIT-matured outcomes through the existing broker-ledger simulator.
"""
from __future__ import annotations

from functools import partial
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from config.execution_policy import (
    DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT,
    DEFAULT_PORTFOLIO_MAX_POSITIONS, DEFAULT_PORTFOLIO_ROTATION,
)
from core.file_integrity import canonical_json_sha256, load_json_strict
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.strategy_param_artifacts import compute_strategy_param_file_sha256
from filters.breakout_quality.daily_trade40_target import (
    build_formal_min_execution_schedule, resolve_formal_min_param_records,
)
from filters.breakout_quality.source_inventory import _build_source_data_inventory
from filters.breakout_quality.strategy_compare_sources import (
    PARAM_POLICY_BASE_FINALIST_BEST, PARAM_POLICY_SPECS,
    _validate_requested_param_policy, load_strategy_param_evaluation_view,
)
from filters.breakout_quality.trade_path_label import (
    TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
    TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
)
from services.breakout_quality.formal_mintrade40_pit_source import (
    FormalMinTrade40PITBlocked, FormalMinTrade40PITSource,
    materialize_formal_mintrade40_pit_labels,
)
from services.breakout_quality.formal_mintrade40_training_source import (
    _current_p16_seed_artifact, inspect_formal_mintrade40_p16_teacher,
)
from services.optimizer.strategy_param_service import _canonical_schedule_build_contract
from services.optimizer.strategy_param_training import (
    FULL_ROOS_SEARCH_FIELDS, _build_selection_full_roos_schedule_contract,
    _selection_full_roos_runtime_contract,
    prepare_selection_historical_full_roos_params,
)

FULL_HISTORY_ID = "FULLTRADE_P17_HISTORICAL_LABEL_V1"
FULL_SOURCE_ID = "formal_full_roos_seed"
FULL_BASELINE_ARM = "C97"


def _full_execution_schedule(records, *, seed: int, param_sha256: str):
    """Adapt verified Full Optimizer records via the one canonical execution mapper.

    The underlying mapper verifies every parameter signature, DL-off state and
    date interval; only the provenance source ID differs from Min/P16. This
    adapter does not change the existing Min owner or its signed Oracle source.
    """
    periods, evidence = build_formal_min_execution_schedule(
        records, seed=seed, param_sha256=param_sha256,
    )
    return (tuple(replace(period, source_id=FULL_SOURCE_ID) for period in periods),
            {**evidence, "origin": "formal_full_roos_rolling_per_seed",
             "sources": [{"source_id": FULL_SOURCE_ID, "sha256": param_sha256}],
             "periods": [
                 {**row, "source_id": FULL_SOURCE_ID}
                 for row in evidence["periods"]
             ]})


def historical_params_path(project_root: str | Path, seed: int) -> Path:
    if type(seed) is not int or seed <= 0:
        raise FormalMinTrade40PITBlocked("Full history needs an explicit positive optimizer seed")
    filename = str(PARAM_POLICY_SPECS[PARAM_POLICY_BASE_FINALIST_BEST]["filename"])
    return (Path(project_root).resolve() / "outputs" / "optimizer"
            / "fulltrade_label_history" / f"seed_{seed}" / "full_dl_off_trained"
            / "active_params" / filename)


def _optimizer_policy(root: Path):
    policy = _canonical_schedule_build_contract(family="full")
    contract = _build_selection_full_roos_schedule_contract(
        first_oos_date=TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
        last_oos_date=TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
        train_window_months=int(policy["train_window_months"]),
        oos_months=int(policy["oos_horizon_months"]),
    )
    args = SimpleNamespace(
        dataset=str(policy["dataset"]),
        trials_per_fold=int(policy["trials_per_fold"]),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
    )
    expected = _selection_full_roos_runtime_contract(
        root=root, args=args, schedule_contract=contract,
        dataset_identity_override=_build_source_data_inventory(root, str(args.dataset)),
    )
    return policy, contract, args, expected


def inspect_full_history(*, project_root: str | Path, seed: int) -> dict[str, object]:
    """Read-only preflight. Incompatible existing bytes are NEVER overwritten."""
    root = Path(project_root).resolve()
    path = historical_params_path(root, seed)
    work = path.parent.parent
    preflight_path = work / "rolling_preflight.json"
    if not path.is_file() and not preflight_path.is_file():
        nonempty = work.is_dir() and any(p.is_file() for p in work.rglob("*"))
        return {"path": path, "action": "BLOCKED" if nonempty else "BUILD",
                "reason": "Interrupted Full Optimizer lacks certified preflight" if nonempty
                          else "Missing independent historical Full/P17 teacher"}
    try:
        _, _, _, expected = _optimizer_policy(root)
        original = load_json_strict(preflight_path)
        if not isinstance(original, dict):
            raise ValueError("Optimizer Full preflight is not an object")
        source_sha = str(original.get("runtime_identity_sha256") or "")
        if source_sha != str(expected["runtime_identity_sha256"]):
            raise ValueError("Full historical source/search/period policy drift")
        unsigned = {k: v for k, v in original.items()
                    if k not in {"runtime_identity_sha256", "created_at", "status"}}
        if canonical_json_sha256(unsigned) != source_sha:
            raise ValueError("Full historical preflight has corrupted fingerprint")
        if path.is_file():
            load_full_history(path, expected_seed=seed, preflight_verified=True)
            return {"path": path, "action": "REUSE", "reason": "Full historical Optimizer verified"}
        # The canonical Full ROOS producer publishes a signed preflight and
        # active_params JSON, not a rolling_optimizer_summary.json. An absent
        # params file is a resumable producer task after verifying preflight.
        return {"path": path, "action": "RESUME", "reason": "Full historical preflight verified"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"path": path, "action": "BLOCKED",
                "reason": f"Full Optimizer history invalid, preserved: {type(exc).__name__}: {exc}"}


def load_full_history(path: Path, *, expected_seed: int,
                      preflight_verified: bool = False) -> FormalMinTrade40PITSource:
    """Read only, verify historical Full provenance, seed, folds and PIT dates."""
    path = Path(path).resolve()
    if (path.parent.name != "active_params" or
            path.parent.parent.name != "full_dl_off_trained" or
            path.parent.parent.parent.name != f"seed_{expected_seed}" or
            path.parent.parent.parent.parent.name != "fulltrade_label_history"):
        raise FormalMinTrade40PITBlocked("Wrong historical Full teacher path/seed")
    root = path.parents[6]
    _, reference, policy_args, expected = _optimizer_policy(root)
    params = load_json_strict(path)
    preflight = load_json_strict(path.parent.parent / "rolling_preflight.json")
    if not all(isinstance(v, dict) for v in (params, preflight)):
        raise FormalMinTrade40PITBlocked("Invalid historical Full source evidence")
    if str(preflight.get("runtime_identity_sha256")) != str(expected["runtime_identity_sha256"]):
        raise FormalMinTrade40PITBlocked("Full optimizer preflight runtime identity invalid")
    unsigned = {key: value for key, value in preflight.items()
                if key not in {"runtime_identity_sha256", "created_at", "status"}}
    if canonical_json_sha256(unsigned) != str(expected["runtime_identity_sha256"]):
        raise FormalMinTrade40PITBlocked("Full historical Optimizer original preflight tampered")
    adaptation = dict(params.get("breakout_quality_param_adaptation") or {})
    if (adaptation.get("mode") != "selection_full_roos_training"
            or adaptation.get("parameter_set") != FULL_HISTORY_ID
            or adaptation.get("training_dl_enabled") is not False
            or tuple(adaptation.get("search_fields") or ()) != tuple(FULL_ROOS_SEARCH_FIELDS)):
        raise FormalMinTrade40PITBlocked("Historical Full teacher source/adaptation mismatch")
    meta = dict(params.get("meta") or {})
    for key, wanted in reference["meta"].items():
        actual = meta.get(key)
        if key in {"first_oos_date", "last_oos_date"}:
            # Canonical Outer Rolling stores fold-start month buckets (e.g.
            # requested 2020-12-31 is published as 2020-12-01). The canonical
            # Full Optimizer validator already compares these two fields by
            # month. Keep the *signed* preflight and exact fold/seed/PIT checks.
            try:
                if pd.Timestamp(str(actual)).to_period("M") == pd.Timestamp(str(wanted)).to_period("M"):
                    continue
            except (TypeError, ValueError, OverflowError) as exc:
                raise FormalMinTrade40PITBlocked(
                    f"Historical Full schedule field differs: {key}"
                ) from exc
        elif str(actual) == str(wanted):
            continue
        raise FormalMinTrade40PITBlocked(f"Historical Full schedule field differs: {key}")
    if str(meta.get("trials_per_fold")) != str(policy_args.trials_per_fold):
        raise FormalMinTrade40PITBlocked("Historical Full trial budget differs")
    records = resolve_formal_min_param_records("rolling_active_param_ensemble", params)
    folds = tuple(dict(f) for f in params.get("folds") or ())
    if not folds or len(records) != len(folds):
        raise FormalMinTrade40PITBlocked("Historical Full fold count incomplete")
    first = pd.Timestamp(TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE).to_period("M")
    last = pd.Timestamp(TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE).to_period("M")
    expected_months = tuple(pd.period_range(first, last, freq="M")
                            [::int(reference["meta"]["oos_horizon_months"])])
    if tuple(pd.Timestamp(r["effective_date"]).to_period("M") for r in records) != expected_months:
        raise FormalMinTrade40PITBlocked("Full historical effective periods incomplete")
    by_fold = {pd.Timestamp(f["oos_start_date"]).normalize(): f for f in folds}
    if len(by_fold) != len(folds):
        raise FormalMinTrade40PITBlocked("Duplicated historical Full fold")
    published = []
    for rec in records:
        day = pd.Timestamp(rec["effective_date"]).normalize()
        fold = by_fold.get(day)
        members = tuple(rec.get("members") or ())
        if fold is None or len(members) != 1:
            raise FormalMinTrade40PITBlocked("Missing single seed historical Full optimizer fold/member")
        member = members[0]
        if (type(member.get("seed")) is not int or member["seed"] != expected_seed
                or ("optimizer_seed" in member and member["optimizer_seed"] != expected_seed)
                or ("optimizer_seed" in fold and fold["optimizer_seed"] != expected_seed)):
            raise FormalMinTrade40PITBlocked("Historical Full optimizer seed differs from DL seed")
        if build_portfolio_params_signature(rec["params_obj"]) != rec["params_signature"]:
            raise FormalMinTrade40PITBlocked("Historical Full execution parameter signature drift")
        training_start = pd.Timestamp(fold["selection_start_date"]).normalize()
        training_end = pd.Timestamp(fold["selection_end_date"]).normalize()
        if (training_end != day - pd.Timedelta(days=1)
                or training_start != (day - pd.DateOffset(months=int(reference["meta"]["train_window_months"]))).normalize()):
            raise FormalMinTrade40PITBlocked("Historical Full optimizer trained with future/invalid window")
        published.append(str(training_end.date()))
    sha = compute_strategy_param_file_sha256(path)
    schedule, _ = _full_execution_schedule(
        records, seed=expected_seed, param_sha256=sha,
    )
    return FormalMinTrade40PITSource(
        seed=expected_seed, source_id=FULL_SOURCE_ID, param_sha256=sha,
        schedule=schedule, publication_dates=tuple(published),
    )


def prepare_full_history(*, project_root: str | Path, seed: int,
                         authorized: bool = False) -> FormalMinTrade40PITSource:
    if not authorized:
        raise FormalMinTrade40PITBlocked("Full historical teacher build requires authorized model work")
    root = Path(project_root).resolve()
    state = inspect_full_history(project_root=root, seed=seed)
    if state["action"] == "BLOCKED":
        raise FormalMinTrade40PITBlocked(str(state["reason"]))
    path = historical_params_path(root, seed)
    if state["action"] == "REUSE":
        return load_full_history(path, expected_seed=seed)
    policy, _, args, _ = _optimizer_policy(root)
    result = prepare_selection_historical_full_roos_params(
        project_root=root, dataset=str(args.dataset),
        param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
        trials_per_fold=int(args.trials_per_fold),
        max_positions=int(args.max_positions), rotation=str(args.rotation),
        fixed_risk=float(args.fixed_risk), max_position_cap_pct=float(args.max_position_cap_pct),
        optimizer_seed=seed, resume_parameter_training=True, quiet=False,
        first_oos_date=TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
        last_oos_date=TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
        train_window_months=int(policy["train_window_months"]),
        oos_months=int(policy["oos_horizon_months"]),
        output_relative_dir=path.parent.parent,
        parameter_set_identity=FULL_HISTORY_ID,
        build_context_label="Historical FullTrade P17 PIT teacher",
        optimizer_output_title="Historical FullTrade Optimizer evidence",
    )
    published = Path(result["params_path"]).resolve()
    if published != path.resolve():
        raise FormalMinTrade40PITBlocked("Canonical Full producer published a different teacher path")
    return load_full_history(published, expected_seed=seed)


def inspect_current_full(*, project_root: str | Path, seed: int):
    return inspect_formal_mintrade40_p16_teacher(
        project_root=project_root, seed=seed, baseline_arm=FULL_BASELINE_ARM,
    )


def execute_current_full(*, project_root: str | Path, seed: int):
    return _current_p16_seed_artifact(
        Path(project_root).resolve(), seed, authorized_build=True,
        baseline_arm=FULL_BASELINE_ARM,
    )


def load_full_training_source(*, project_root: str | Path, seed: int,
                              authorized_build: bool = False) -> FormalMinTrade40PITSource:
    root = Path(project_root).resolve()
    current_path, start, end, policy = _current_p16_seed_artifact(
        root, seed, authorized_build=authorized_build,
        baseline_arm=FULL_BASELINE_ARM,
    )
    if authorized_build:
        history = prepare_full_history(project_root=root, seed=seed, authorized=True)
    else:
        history = load_full_history(historical_params_path(root, seed), expected_seed=seed)
    current_source, digest = load_strategy_param_evaluation_view(
        current_path, evaluation_mode="rolling", start_date=start, end_date=end,
    )
    kind = str(current_source.get("kind"))
    if kind not in {"rolling_active_param_ensemble", "rolling_oos_param_schedule"}:
        raise FormalMinTrade40PITBlocked("Full P17 requires a canonical rolling effective schedule")
    _validate_requested_param_policy(current_source, policy)
    records = resolve_formal_min_param_records(kind, current_source["payload"])
    for record in records:
        if any(m.get("seed") != seed for m in record.get("members") or ()):
            raise FormalMinTrade40PITBlocked("Current Full P17 member seed differs from DL seed")
    current, _ = _full_execution_schedule(
        records, seed=seed, param_sha256=digest,
    )
    if not current or current[0].start != pd.Timestamp(start).normalize():
        raise FormalMinTrade40PITBlocked("Current Full P17 missing first OOS effective period")
    if history.schedule[-1].end is None or history.schedule[-1].end >= current[0].start:
        raise FormalMinTrade40PITBlocked("Historical Full and current P17 overlap")
    folds = tuple(dict(row) for row in current_source["payload"].get("folds") or ())
    by_start = {pd.Timestamp(f.get("effective_start") or f["oos_start_date"]).normalize(): f
                for f in folds}
    if len(by_start) != len(folds):
        raise FormalMinTrade40PITBlocked("Duplicated Full P17 effective fold")
    dates = []
    for period in current:
        fold = by_start.get(period.start)
        if fold is None:
            raise FormalMinTrade40PITBlocked("Full P17 original optimizer fold missing")
        selection_end = pd.Timestamp(fold["selection_end_date"]).normalize()
        selection_start = pd.Timestamp(fold["selection_start_date"]).normalize()
        if selection_start >= selection_end or selection_end != period.start - pd.Timedelta(days=1):
            raise FormalMinTrade40PITBlocked("Full P17 optimizer publication is not PIT")
        origin_seed = fold.get("optimizer_seed")
        if (origin_seed is None and kind != "rolling_active_param_ensemble") or (
                origin_seed is not None and (type(origin_seed) is not int or origin_seed != seed)):
            raise FormalMinTrade40PITBlocked("Full P17 optimizer fold seed mismatch")
        dates.append(str(selection_end.date()))
    schedules = history.schedule + current
    publication = history.publication_dates + tuple(dates)
    if any(a.end is None or a.end + pd.Timedelta(days=1) < b.start
           for a, b in zip(schedules, schedules[1:])):
        raise FormalMinTrade40PITBlocked("Uncovered historical/current Full parameter period")
    return FormalMinTrade40PITSource(
        seed=seed, source_id=FULL_SOURCE_ID,
        param_sha256=canonical_json_sha256({
            "contract": "formal_fulltrade_seed_historical_plus_p17_v1", "seed": seed,
            "history_sha256": history.param_sha256, "p17_runtime_sha256": digest,
        }),
        schedule=schedules, publication_dates=publication,
        oos_start_date=str(current[0].start.date()),
    )


def resolve_fulltrade_loader_kwargs(*, project_root: str | Path, seed: int,
                                    authorized_build: bool = False) -> dict[str, object]:
    source = load_full_training_source(
        project_root=project_root, seed=seed, authorized_build=authorized_build,
    )
    return {
        "target_materializer": partial(
            materialize_formal_mintrade40_pit_labels,
            source=source, expected_seed=seed, horizon_sessions=None,
            outcome_metric="net_pnl", cache_project_root=str(Path(project_root).resolve()),
        ),
        "target_lineage": {
            "contract": "formal_fulltrade_seed_historical_plus_p17_v1",
            "target_kind": "formal_seed_fulltrade_fullcycle_broker_net_pnl",
            "horizon_mode": "full_trade_cycle", "value_field": "raw_net_pnl",
            "maturity_policy": "natural_exit_or_research_terminal_closeout_pit_v1",
            "seed": seed, "source_sha256": source.param_sha256,
            "first_effective_date": str(source.schedule[0].start.date()),
            "last_source_period_start": str(source.schedule[-1].start.date()),
            "oos_start_date": source.oos_start_date,
            "teacher_periods": [
                {"start": str(p.start.date()),
                 "end": None if p.end is None else str(p.end.date()),
                 "params_signature": p.params_signature, "published": published}
                for p, published in zip(source.schedule, source.publication_dates)
            ],
        },
    }

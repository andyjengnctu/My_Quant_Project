"""Seed-scoped PIT MinTrade40 label lineage for the canonical Daily Universal trainer.

The two temporal parameter sources are independent, immutable Optimizer outputs:
2014-2020 Selection historical Min, and the corresponding seed's 2021+ P16.
They are *never* chosen by observed Target/strategy performance. Training uses
only labels whose original execution resolved before the relevant model cutoff.
"""
from __future__ import annotations

from pathlib import Path
from functools import partial

import pandas as pd

from config.research import RESEARCH_SINGLE_SEED
from config.execution_policy import DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT
from core.file_integrity import canonical_json_sha256
from core.strategy_compare_policy import (
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_settings,
)
from filters.breakout_quality.daily_trade40_target import (
    build_formal_min_execution_schedule, resolve_formal_min_param_records,
)
from filters.breakout_quality.strategy_compare_sources import load_strategy_param_evaluation_view
from services.breakout_quality.formal_mintrade40_pit_source import (
    FormalMinTrade40PITBlocked, FormalMinTrade40PITSource,
    materialize_formal_mintrade40_pit_labels,
)
from services.optimizer.formal_mintrade40_teacher_source import (
    formal_mintrade40_history_params_path,
    load_formal_mintrade40_optimizer_teacher,
    prepare_formal_mintrade40_optimizer_teacher,
)
from services.optimizer.strategy_param_service import (
    resolve_fixed_multiseed_robustness_strategy_parameter_artifact,
    ensure_fixed_multiseed_robustness_strategy_parameter_artifact,
    inspect_fixed_multiseed_strategy_parameter_artifact,
    ensure_fixed_multiseed_strategy_parameter_artifact,
)
from services.research.strategy_compare_preparation_status import resolve_param_source_path


# Matches the one existing Broker-Cash / P16 Min baseline consumed by the
# formally defined Oracle; not a model-specific teacher or alternative P16.
_BROKER_CASH_ROLLING_PROFILE = 'broker_cash_dl_conversion_rolling'
_BASELINE_ARM = 'C96'


def _current_p16_policy(seed: int):
    """One canonical P16 reference/producer contract for read and build paths."""
    settings = get_strategy_comparison_settings(_BROKER_CASH_ROLLING_PROFILE)
    arm = settings.arms[_BASELINE_ARM]
    source = settings.parameter_sources[arm.param_source]
    builder = source.builder
    if builder is None or not builder.enabled or builder.builder_type != 'optimizer_fixed_multiseed_strategy_params':
        raise FormalMinTrade40PITBlocked('Canonical Min P16 Optimizer producer missing')
    training_profile = str(builder.options.get('training_profile_id') or '')
    if not training_profile:
        raise FormalMinTrade40PITBlocked('Canonical Min P16 training profile missing')
    benchmark = get_strategy_multi_seed_robustness_settings(settings.profile_id)
    optimizer_args = dict(
        training_profile_id=training_profile,
        policy=str(arm.param_policy),
        comparison_end_date=str(settings.end_date),
        dataset=str(settings.dataset),
        max_positions=int(settings.max_positions),
        rotation=str(settings.rotation),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
    )
    return settings, arm, benchmark, optimizer_args


def inspect_formal_mintrade40_p16_teacher(
    *, project_root: str | Path, seed: int,
) -> dict[str, object]:
    """Read-only same-seed P16 status; never starts the canonical Optimizer."""
    root = Path(project_root).resolve()
    settings, arm, benchmark, optimizer_args = _current_p16_policy(seed)
    if seed == int(RESEARCH_SINGLE_SEED):
        path = resolve_param_source_path(
            root, settings, arm.param_source, param_policy=arm.param_policy,
        ).resolve()
        state = inspect_fixed_multiseed_strategy_parameter_artifact(root, **optimizer_args)
    elif seed in tuple(int(value) for value in benchmark.resolved_seeds):
        state = resolve_fixed_multiseed_robustness_strategy_parameter_artifact(
            root, **optimizer_args,
            benchmark_id=str(benchmark.benchmark_id), replication_seed=seed,
        )
        path = Path(state['path']).resolve()
    else:
        raise FormalMinTrade40PITBlocked('Model seed has no registered same-seed P16 producer')
    if bool(state.get('ready')):
        if Path(state['path']).resolve() != path or not path.is_file():
            return {'path': path, 'action': 'BLOCKED', 'reason': 'Verified P16 publication path mismatch'}
        return {'path': path, 'action': 'REUSE', 'reason': 'Original same-seed canonical P16 verified'}
    candidate_action = str(state.get('action') or '').upper()
    if candidate_action not in {'BUILD', 'REBUILD', 'RESUME'}:
        candidate_action = 'BLOCKED'
    if path.is_file() and candidate_action != 'RESUME':
        candidate_action = 'BLOCKED'
    return {
        'path': path,
        'action': candidate_action,
        'reason': f'Canonical P16 seed={seed}: {state.get("status") or "not READY"}',
    }


def _current_p16_seed_artifact(
    root: Path, seed: int, *, authorized_build: bool = False,
) -> tuple[Path, str, str, str]:
    settings, arm, benchmark, optimizer_args = _current_p16_policy(seed)
    benchmark_seeds = tuple(int(value) for value in benchmark.resolved_seeds)
    if seed == int(RESEARCH_SINGLE_SEED):
        path = resolve_param_source_path(
            root, settings, arm.param_source, param_policy=arm.param_policy,
        ).resolve()
        state = inspect_fixed_multiseed_strategy_parameter_artifact(root, **optimizer_args)
        if not bool(state.get('ready')):
            if not authorized_build or str(state.get('action')) == 'BLOCKED':
                raise FormalMinTrade40PITBlocked(
                    f'Canonical single-seed P16 not verified: {state.get("status")}; '
                    'authorized canonical Optimizer required'
                )
            record = ensure_fixed_multiseed_strategy_parameter_artifact(
                root, **optimizer_args,
            )
        else:
            record = state
        if Path(record['path']).resolve() != path or not path.is_file():
            raise FormalMinTrade40PITBlocked('Canonical P16 publication path differs from registered strategy source')
        verified = inspect_fixed_multiseed_strategy_parameter_artifact(root, **optimizer_args)
        if not verified.get('ready'):
            raise FormalMinTrade40PITBlocked('Canonical single-seed P16 remains unverified after producer')
    elif seed in benchmark_seeds:
        robust_args = dict(
            **optimizer_args,
            benchmark_id=str(benchmark.benchmark_id),
            replication_seed=seed,
        )
        record = resolve_fixed_multiseed_robustness_strategy_parameter_artifact(
            root, **robust_args,
        )
        if not bool(record.get('ready')):
            if not authorized_build or str(record.get('action')) == 'BLOCKED':
                raise FormalMinTrade40PITBlocked(
                    f'Canonical P16 seed={seed} not verified: {record.get("status")}; '
                    'authorized canonical Optimizer required'
                )
            ensure_fixed_multiseed_robustness_strategy_parameter_artifact(
                root, **robust_args,
            )
            record = resolve_fixed_multiseed_robustness_strategy_parameter_artifact(
                root, **robust_args,
            )
            if not bool(record.get('ready')):
                raise FormalMinTrade40PITBlocked('Canonical robustness P16 remains unverified after producer')
        path = Path(record['path'])
    else:
        raise FormalMinTrade40PITBlocked('Model seed has no registered same-seed P16 producer')
    return Path(path).resolve(), str(settings.start_date), str(settings.end_date), str(arm.param_policy)


def load_formal_mintrade40_training_source(
    *, project_root: str | Path, seed: int, allow_optimizer_build: bool = False,
) -> FormalMinTrade40PITSource:
    """Bind original Min label history and current P16 for one exact model seed.

    This function never reads hindsight Oracle truth/caches or independently
    optimizes P16. Only an authorized canonical Optimizer may build missing
    historical teacher evidence; an invalid existing artifact is always blocked.
    """
    if type(seed) is not int or seed <= 0:
        raise FormalMinTrade40PITBlocked('Formal MinTrade40 requires an exact positive model seed')
    root = Path(project_root).resolve()
    # Both dependencies belong to canonical Optimizer; when this model work is
    # authorized a missing legitimate P16 is built by its own producer first.
    # Existing incompatible identities remain blocked by that producer.
    p16_path, start, end, policy = _current_p16_seed_artifact(
        root, seed, authorized_build=allow_optimizer_build,
    )
    if allow_optimizer_build:
        history = prepare_formal_mintrade40_optimizer_teacher(
            project_root=root, seed=seed, authorized=True,
        )
    else:
        path = formal_mintrade40_history_params_path(root, seed)
        history = load_formal_mintrade40_optimizer_teacher(path, expected_seed=seed)

    source, digest = load_strategy_param_evaluation_view(
        p16_path, evaluation_mode='rolling', start_date=start, end_date=end,
    )
    if str(source.get('kind')) not in {'rolling_active_param_ensemble', 'rolling_oos_param_schedule'}:
        raise FormalMinTrade40PITBlocked('Min P16 is not a rolling effective parameter schedule')
    from filters.breakout_quality.strategy_compare_sources import _validate_requested_param_policy
    _validate_requested_param_policy(source, policy)
    records = resolve_formal_min_param_records(str(source['kind']), source['payload'])
    for row in records:
        members = tuple(row.get('members') or ())
        if members and any(
            type(member.get('seed')) is not int or member['seed'] != seed
            for member in members
        ):
            raise FormalMinTrade40PITBlocked('Current P16 member/model Seed mismatch')
    current, _ = build_formal_min_execution_schedule(records, seed=seed, param_sha256=digest)
    if not current or current[0].start != pd.Timestamp(start).normalize():
        raise FormalMinTrade40PITBlocked('Current P16 does not begin at the registered OOS boundary')
    if history.schedule[-1].end is None or history.schedule[-1].end >= current[0].start:
        raise FormalMinTrade40PITBlocked('Historical/P16 Min schedules overlap or have open-ended history')
    # Bind the source's original optimizer training cutoff, not an invented
    # publication date inferred from an effective date or a filesystem mtime.
    folds = tuple(dict(fold) for fold in source['payload'].get('folds') or ())
    by_start: dict[pd.Timestamp, dict] = {}
    for fold in folds:
        try:
            effective = pd.Timestamp(fold.get('effective_start') or fold['oos_start_date']).normalize()
        except (TypeError, ValueError, KeyError) as exc:
            raise FormalMinTrade40PITBlocked('P16 original fold effective date missing') from exc
        if effective in by_start:
            raise FormalMinTrade40PITBlocked('P16 original fold effective dates overlap')
        by_start[effective] = fold
    published = []
    for period in current:
        fold = by_start.get(period.start)
        if fold is None:
            raise FormalMinTrade40PITBlocked('P16 lacks matching original Optimizer fold')
        try:
            selected_end = pd.Timestamp(fold['selection_end_date']).normalize()
            selected_start = pd.Timestamp(fold['selection_start_date']).normalize()
        except (TypeError, ValueError, KeyError) as exc:
            raise FormalMinTrade40PITBlocked('P16 original historical training dates missing') from exc
        if (selected_start >= selected_end
                or selected_end != period.start - pd.Timedelta(days=1)):
            raise FormalMinTrade40PITBlocked('P16 original Optimizer fold selection is not PIT aligned')
        origin_seed = fold.get('optimizer_seed')
        if (origin_seed is None and source['kind'] != 'rolling_active_param_ensemble'):
            raise FormalMinTrade40PITBlocked('P16 single-parameter fold has no canonical optimizer Seed')
        if origin_seed is not None and (type(origin_seed) is not int or origin_seed != seed):
            raise FormalMinTrade40PITBlocked('P16 fold Optimizer Seed differs from model Seed')
        published.append(str(selected_end.date()))
    published = tuple(published)
    schedules = history.schedule + current
    publication_dates = history.publication_dates + published
    if any(a.end is None or a.end + pd.Timedelta(days=1) < b.start
           for a, b in zip(schedules, schedules[1:])):
        raise FormalMinTrade40PITBlocked('Formal Min schedule has an uncovered effective period')
    return FormalMinTrade40PITSource(
        seed=seed,
        param_sha256=canonical_json_sha256({
            'contract': 'formal_mintrade40_seed_historical_plus_p16_v1',
            'seed': seed, 'history_sha256': history.param_sha256,
            'p16_runtime_sha256': digest,
        }),
        schedule=schedules,
        publication_dates=publication_dates,
        oos_start_date=str(current[0].start.date()),
    )


def resolve_formal_mintrade40_loader_kwargs(
    *, project_root: str | Path, seed: int, authorized_build: bool = False,
    horizon_sessions: int | None = 40, outcome_metric: str = "net_r",
) -> dict:
    """Return a seed-bound materializer and immutable source provenance.

    All trainer/PIT/audit consumers call this same owner. Materialization always
    goes through canonical 40-session Min execution, never the Oracle cache.
    """
    if outcome_metric not in {"net_r", "net_pnl"} or (horizon_sessions is not None and
            (type(horizon_sessions) is not int or horizon_sessions <= 0)):
        raise ValueError("Unregistered formal Min target horizon/metric")
    source = load_formal_mintrade40_training_source(
        project_root=project_root, seed=int(seed),
        allow_optimizer_build=bool(authorized_build),
    )
    full_cycle = horizon_sessions is None
    lineage_extension = (
        {'target_kind': f'formal_seed_mintrade_fullcycle_broker_{outcome_metric}',
         'maturity_policy': 'natural_exit_or_research_terminal_closeout_pit_v1',
         'horizon_mode': 'full_trade_cycle',
         **({'value_field': 'raw_net_pnl'} if outcome_metric == 'net_pnl' else {})}
        if full_cycle else {}
    )
    return {
        'target_materializer': partial(
            materialize_formal_mintrade40_pit_labels,
            source=source, expected_seed=int(seed),
            horizon_sessions=horizon_sessions, outcome_metric=outcome_metric,
            cache_project_root=str(Path(project_root).resolve()),
        ),
        'target_lineage': {
            'contract': 'formal_mintrade40_seed_historical_plus_p16_v1',
            **lineage_extension,
            'seed': int(seed),
            'source_sha256': str(source.param_sha256),
            'first_effective_date': str(source.schedule[0].start.date()),
            'last_source_period_start': str(source.schedule[-1].start.date()),
            'oos_start_date': source.oos_start_date,
            'teacher_periods': [
                {'start': str(period.start.date()),
                 'end': None if period.end is None else str(period.end.date()),
                 'params_signature': period.params_signature,
                 'published': published}
                for period, published in zip(source.schedule, source.publication_dates)
            ],
        },
    }


__all__ = ['load_formal_mintrade40_training_source',
           'resolve_formal_mintrade40_loader_kwargs',
           'inspect_formal_mintrade40_p16_teacher']

"""Training-history composition over canonical label and parameter producers."""
from dataclasses import replace
from functools import partial
from pathlib import Path

from config.breakout_quality_history import TRAINING_HISTORY_EXTENSIONS


def _early_inspect(*, contract, project_root, seed):
    from services.optimizer.formal_mintrade40_teacher_source import inspect_formal_mintrade40_optimizer_teacher
    return inspect_formal_mintrade40_optimizer_teacher(
        project_root=project_root, seed=seed, history_contract=contract)


def _early_execute(*, contract, project_root, seed):
    from services.optimizer.formal_mintrade40_teacher_source import prepare_formal_mintrade40_optimizer_teacher
    return prepare_formal_mintrade40_optimizer_teacher(
        project_root=project_root, seed=seed, authorized=True, history_contract=contract)


def _early_loader(*, contract, project_root, seed, authorized_build):
    from services.breakout_quality.formal_mintrade40_training_source import resolve_formal_mintrade40_loader_kwargs
    control = resolve_formal_mintrade40_loader_kwargs(
        project_root=project_root, seed=seed, authorized_build=authorized_build,
        horizon_sessions=None, outcome_metric='net_pnl')
    extended = resolve_formal_mintrade40_loader_kwargs(
        project_root=project_root, seed=seed, authorized_build=authorized_build,
        horizon_sessions=None, outcome_metric='net_pnl', history_extension=contract)
    extended['target_materializer'] = partial(
        _prefix_materializer, control=control['target_materializer'],
        extended=extended['target_materializer'],
        boundary=control['target_lineage']['first_effective_date'])
    extended['target_lineage'] = {
        **extended['target_lineage'],
        'prefix_composition': 'original_information_rows_reuse_original_signal_lifecycle_v1',
        'original_first_effective_date': control['target_lineage']['first_effective_date'],
    }
    return extended


def _prefix_materializer(*, control, extended, boundary, frame, positions, **kwargs):
    """Keep every old label byte-identical, including pending orders at the join.

    The next executable session selects the original or prefix producer. Older
    trades may mature later; neither history is force-closed at the join.
    """
    import numpy as np
    import pandas as pd
    from dataclasses import fields
    from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITLabels
    positions = np.asarray(positions, dtype=np.int64)
    dates = pd.DatetimeIndex(frame.index)
    executable = np.minimum(positions + 1, len(dates) - 1)
    original = np.asarray(dates.take(executable) >= pd.Timestamp(boundary))
    pieces = []
    for mask, producer in ((original, control), (~original, extended)):
        if mask.any():
            pieces.append((mask, producer(frame=frame, positions=positions[mask], **kwargs)))
    if not pieces:
        return extended(frame=frame, positions=positions, **kwargs)
    result = {}
    for field in fields(FormalMinTrade40PITLabels):
        if field.name == 'preflight':
            result[field.name] = {'composition': 'original_plus_early_prefix',
                                  'sources': [part.preflight for _, part in pieces]}
        elif getattr(pieces[0][1], field.name) is None:
            result[field.name] = None
        else:
            value = np.empty(len(positions), dtype=getattr(pieces[0][1], field.name).dtype)
            for mask, part in pieces:
                value[mask] = getattr(part, field.name)
            result[field.name] = value
    return FormalMinTrade40PITLabels(**result)


def extend_source_capability(source, profile):
    extension = TRAINING_HISTORY_EXTENSIONS.get(profile)
    if extension is None:
        return source
    contract = dict(extension)
    early = replace(source.dependencies[0],
        name='early_history', builder_type='optimizer_early_history', priority=50,
        inspect=partial(_early_inspect, contract=contract), execute=partial(_early_execute, contract=contract))
    return replace(source, loader=partial(_early_loader, contract=contract),
                   dependencies=(*source.dependencies, early))


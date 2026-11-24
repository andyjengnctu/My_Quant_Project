"""Daily DL loader adapter; feature-price identity has an independent owner."""
from __future__ import annotations
from filters.breakout_quality.daily_ranker_data import load_daily_universal_ranker_data
from filters.breakout_quality.workflow_io import PROJECT_ROOT
from services.breakout_quality.trade_count_source import resolve_trade_count_feature_source
from services.breakout_quality.feature_price_source import (
    RAW_PRICE_INPUT_CAPABILITY, uses_alternate_price_input,
    resolve_feature_price_source, verified_current_price_source_identity,
)


def load_daily_universal_ranker_data_with_price_source(**kwargs):
    """Drop-in domain adapter for every direct daily-universal producer."""
    root = kwargs.get("project_root", PROJECT_ROOT)
    from services.breakout_quality.trade_path_target_source import loader_kwargs_for_trade_path
    seed = kwargs.pop('seed', None)
    authorized = bool(kwargs.pop('authorized_target_dependency_build', False))
    kwargs.update(loader_kwargs_for_trade_path(
        experiment_profile=str(kwargs['experiment_profile']),
        project_root=root, seed=seed, authorized_build=authorized,
    ))
    alternate = resolve_feature_price_source(
        model_architecture=kwargs["model_architecture"],
        project_root=root,
        create=True,
    )
    activity = resolve_trade_count_feature_source(
        model_architecture=kwargs["model_architecture"], project_root=root,
    )
    return load_daily_universal_ranker_data(
        **kwargs, feature_price_source=alternate, feature_trade_count_source=activity,
    )

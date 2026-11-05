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


def fixed_min_source(seed: int):
    """The same immutable constant params as the strategy (no Optimizer/P16)."""
    if type(seed) is not int or seed <= 0:
        raise ValueError("Fixed Min requires a positive FIT seed")
    import pandas as pd
    from filters.breakout_quality.daily_trade40_target import ExecutableMinPeriod
    from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITSource
    period = ExecutableMinPeriod(
        start=pd.Timestamp(FIXED_MIN_FIRST_EFFECTIVE_DATE), end=None,
        params=fixed_min_params(), source_id=FIXED_MIN_ID,
        params_signature=fixed_min_signature(),
    )
    return FormalMinTrade40PITSource(
        seed=seed, param_sha256=fixed_min_scientific_sha256(),
        schedule=(period,), publication_dates=(FIXED_MIN_FIRST_EFFECTIVE_DATE,),
        source_id=FIXED_MIN_ID, oos_start_date=None,
    )


def verify_fixed_min_source(source) -> bool:
    """Strictly prove one identical constant, without accepting fake Optimizer lineage."""
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

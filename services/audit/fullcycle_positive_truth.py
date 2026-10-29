"""Signed, NON-PIT *raw* Full-Cycle Min NetPnL for one diagnostic gate.

This is a consumer of the existing single-stock Min execution owner.  It does
not compute orders, fees, stops, exits or sizing itself.  It independently
reconstructs the canonical outcome *values* once, and requires an exact SHA
match against the existing formally signed NetPnL truth receipt, including
its daily percentile scores, before publishing the additional sign witness.
The original truth and every existing Oracle Replay receipt are immutable.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import zipfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.file_integrity import (
    canonical_json_sha256, compute_file_sha256,
    atomic_replace_with_retry, exclusive_file_lock,
)
from core.oracle_positive_gate import POSITIVE_FULLCYCLE_NETPNL_GATE
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.daily_trade40_target import (
    collect_canonical_exchange_sessions, compute_counterfactual_trade40_batch,
    rank_trade40_outcome_frame,
)
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame
from services.audit.daily_trade40_oracle import truth_producer_fingerprint

SCHEMA = 1


def _identity(*, seed, price_sha, param_sha, source_evidence, project_root):
    root = Path(__file__).resolve().parents[2]
    return {
        'schema': SCHEMA, 'gate': POSITIVE_FULLCYCLE_NETPNL_GATE,
        'seed': int(seed), 'price_sha256': str(price_sha),
        'param_sha256': str(param_sha),
        'canonical_raw_sha256': str(source_evidence['raw_label_sha256']),
        'canonical_rank_sha256': str(source_evidence['rank_sha256']),
        'canonical_truth_source_sha256': str(source_evidence['truth_source_sha256']),
        'gate_source_sha256': compute_file_sha256(root / 'core/oracle_positive_gate.py'),
        'sign_producer_sha256': compute_file_sha256(Path(__file__)),
    }


def _path(root: Path, subdir: str, identity: Mapping[str, Any]) -> Path:
    from config.audit import AUDIT_OUTPUT_ROOT
    return (root / AUDIT_OUTPUT_ROOT / subdir / 'positive_raw_truth' /
            str(identity['seed']) / f"{canonical_json_sha256(identity)}.npz")


def _load(path: Path, identity: Mapping[str, Any]) -> dict[tuple[str, str], float] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'meta', 'tickers', 'days', 'raw'}:
                raise ValueError('unexpected positive-truth payload fields')
            meta = json.loads(str(archive['meta'].item()))
            tickers, days, values = archive['tickers'], archive['days'], archive['raw']
        if (meta.get('identity') != dict(identity) or
                meta.get('receipt_sha256') != canonical_json_sha256({
                    'identity': dict(identity), 'raw_sha256': meta.get('raw_sha256'),
                    'rows': meta.get('rows'), 'positive': meta.get('positive'),
                })):
            raise ValueError('positive-truth receipt identity or SHA mismatch')
        if (tickers.ndim != 1 or days.ndim != 1 or values.ndim != 1 or
                len(tickers) != len(days) or len(days) != len(values) or
                tickers.dtype.kind != 'U' or days.dtype.kind != 'U' or values.dtype.kind != 'f' or
                len(values) != int(meta['rows'])):
            raise ValueError('positive-truth rows/shape invalid')
        keys = list(zip(tickers.tolist(), days.tolist()))
        if len(set(keys)) != len(keys):
            raise ValueError('positive-truth contains duplicate keys')
        digest = sha256()
        for key, value in zip(keys, values):
            if not np.isfinite(value) and not np.isnan(value):
                raise ValueError('positive-truth has infinite value')
            digest.update(f'{key[0]}|{key[1]}|{float(value):.17g}\n'.encode('ascii'))
        if (digest.hexdigest() != meta.get('raw_sha256') or
                sum(float(x) > 0 for x in values) != int(meta['positive'])):
            raise ValueError('positive-truth values differ from signed receipt')
        return {key: float(value) for key, value in zip(keys, values) if np.isfinite(value)}
    except (OSError, ValueError, TypeError, KeyError, IndexError, zipfile.BadZipFile) as exc:
        raise ValueError(f'Positive-NetPnL truth evidence corrupt: {path.name}: {exc}') from exc


def _save(path: Path, identity: Mapping[str, Any], raw: Mapping[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(raw)
    digest = sha256()
    for ticker, day in keys:
        digest.update(f'{ticker}|{day}|{float(raw[ticker, day]):.17g}\n'.encode('ascii'))
    receipt = {'identity': dict(identity), 'raw_sha256': digest.hexdigest(),
               'rows': len(keys), 'positive': sum(float(x) > 0 for x in raw.values())}
    meta = json.dumps({**receipt, 'receipt_sha256': canonical_json_sha256(receipt)},
                      sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    fd, temporary = tempfile.mkstemp(prefix='.positive.', suffix='.npz', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, meta=np.asarray(meta),
                                tickers=np.asarray([key[0] for key in keys]),
                                days=np.asarray([key[1] for key in keys]),
                                raw=np.asarray([raw[key] for key in keys], dtype=np.float64))
            stream.flush()
            os.fsync(stream.fileno())
        with exclusive_file_lock(path.parent / '.publish.lock'):
            existing = _load(path, identity)
            if existing is None:
                atomic_replace_with_retry(temporary, path)
            else:
                expected = {key: float(value) for key, value in raw.items() if np.isfinite(value)}
                if existing != expected:
                    raise ValueError('same-identity positive truth conflicts with old evidence')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_or_build_positive_truth(
    *, project_root: Path, output_subdir: str, dataset: str, filter_id: str,
    profile_id: str, seed: int, price_sha256: str, param_sha256: str,
    canonical_schedule, research_cutoff: str, score_start: str,
    evaluation_end: str, canonical_scores: Mapping[tuple[str, str], float],
    canonical_evidence: Mapping[str, Any], feature_cohort_cache=None,
    progress=None,
) -> tuple[dict[tuple[str, str], float], Path]:
    """Return exact, finite raw PnLs; absent/NULL keys are rejected by the gate."""
    root = Path(project_root).resolve()
    if (canonical_evidence.get('truth_source_sha256') != truth_producer_fingerprint('net_pnl')
            or canonical_evidence.get('teacher', {}).get('seed') != seed
            or canonical_evidence.get('teacher', {}).get('param_identity_sha256') != param_sha256
            or canonical_evidence.get('horizon_sessions') is not None):
        raise ValueError('positive gate requires current full-cycle signed NetPnL source')
    identity = _identity(seed=seed, price_sha=price_sha256, param_sha=param_sha256,
                         source_evidence=canonical_evidence, project_root=root)
    path = _path(root, output_subdir, identity)
    existing = _load(path, identity)
    if existing is not None:
        if not set(existing).issubset(canonical_scores):
            raise ValueError('positive raw keys differ from original score keys')
        return existing, path
    # Same generic feature cohort; no strategy filter or portfolio-selected
    # sample may enter this table. The canonical producer remains the authority.
    from services.audit.oracle_score_artifacts import oracle_dataset_summary_sha256
    dataset_sha = oracle_dataset_summary_sha256(root=root, filter_id=filter_id)
    key = (str(root), dataset, filter_id, profile_id, str(score_start),
           str(evaluation_end), str(dataset_sha))
    snapshot = (feature_cohort_cache or {}).get(key)
    if snapshot is None:
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        profile = get_breakout_quality_experiment_profile(profile_id)
        bundle = load_profile_continuous_ranker_data(
            filter_id=filter_id, model_architecture=profile.model_architecture,
            experiment_profile=profile_id, preload_feature_bank=False,
            allow_stale_source=False, project_root=root,
        )
        cohort = pd.DataFrame(bundle.group_table).reset_index(drop=True)
        cohort['ticker'] = cohort['ticker'].map(normalize_ticker)
        cohort['date'] = pd.to_datetime(cohort['date']).dt.normalize()
        cohort = cohort.loc[(cohort['date'] >= pd.Timestamp(score_start)) &
                            (cohort['date'] <= pd.Timestamp(evaluation_end))].reset_index(drop=True)
    else:
        cohort = snapshot[0]
    if cohort.duplicated(['ticker', 'date']).any() or len(cohort) != len(canonical_scores):
        raise ValueError('positive gate does not have the same certified score cohort')
    csv_inputs, duplicates = discover_dataset_csv_inputs(root, dataset)
    if duplicates:
        raise ValueError('positive gate discovered duplicate market data sources')
    calendar = collect_canonical_exchange_sessions(
        csv_inputs, benchmark_ticker=str(DEFAULT_LABEL_POLICY.benchmark_ticker),
        horizon_sessions=None,
    )
    paths = {normalize_ticker(ticker): Path(file) for ticker, file in csv_inputs}
    frames = []
    total = cohort['ticker'].nunique()
    for count, (ticker, selected) in enumerate(cohort.groupby('ticker', sort=True), start=1):
        frame = load_dataset_frame(paths[ticker], ticker, min_rows=1)
        positions = pd.DatetimeIndex(frame.index).normalize().get_indexer(
            pd.DatetimeIndex(selected['date']))
        if (positions < 0).any():
            raise ValueError(f'positive gate missing canonical market-day: {ticker}')
        batch = compute_counterfactual_trade40_batch(
            frame, positions, ticker=ticker, schedule=canonical_schedule,
            exchange_calendar=calendar, horizon_sessions=None,
            research_cutoff=research_cutoff, include_net_pnl=True,
        )
        if any(reason == 'no_historical_effective_Min_param' for reason in batch.reasons):
            raise ValueError('positive gate missing formal P16 effective period')
        frames.append(pd.DataFrame({
            'ticker': ticker, 'date': selected['date'].dt.strftime('%Y-%m-%d').to_numpy(),
            'realized_net_pnl': np.asarray(batch.target_raw_pnl, dtype=np.float64),
        }))
        if progress is not None and (count % 200 == 0 or count == total):
            progress(f'正獲利資格 {count}/{total} stocks')
    outcome = pd.concat(frames, ignore_index=True).sort_values(['date', 'ticker'], kind='stable')
    raw_sha = sha256(outcome.to_csv(index=False, na_rep='NULL', float_format='%.17g').encode('utf-8')).hexdigest()
    if raw_sha != canonical_evidence['raw_label_sha256']:
        raise ValueError('positive gate raw PnL differs from canonical signed NetPnL labels')
    ranked = rank_trade40_outcome_frame(outcome, value_column='realized_net_pnl')
    if len(ranked) != len(canonical_scores) or ranked != dict(canonical_scores):
        raise ValueError('positive gate daily percentile does not equal canonical score')
    raw = {(row.ticker, row.date): float(row.realized_net_pnl)
           for row in outcome.itertuples(index=False, name='Row')}
    _save(path, identity, raw)
    return {key: val for key, val in raw.items() if np.isfinite(val)}, path

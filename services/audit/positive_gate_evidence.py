"""Immutable per-seed evidence for the one-time positive-NetPnL Audit.

This is recorded only for the explicitly diagnostic positive-gate Oracle.
Economic observations come from the *actual* Portfolio replay; no audit
trade simulator or second PnL formula supplies the recorded win/loss result.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from core.file_integrity import canonical_json_sha256, exclusive_file_lock, atomic_replace_with_retry
from core.oracle_positive_gate import POSITIVE_FULLCYCLE_NETPNL_GATE
from config.audit import AUDIT_OUTPUT_ROOT

SCHEMA = 1
TARGET = 'DailyUniversal_MinTrade_NetPnL_Positive_Direct'


def evidence_path(root: Path, subdir: str, seed: int, identity: Mapping[str, Any]) -> Path:
    return (Path(root) / AUDIT_OUTPUT_ROOT / subdir / 'positive_gate_replay_evidence' /
            str(int(seed)) / f'{canonical_json_sha256(identity)}.json')


def read_evidence(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        if obj.get('schema') != SCHEMA or obj.get('target') != TARGET or not isinstance(obj.get('rows'), list):
            raise ValueError('positive gate evidence has wrong schema or target')
        sha = obj.get('receipt_sha256')
        if sha != canonical_json_sha256({key: value for key, value in obj.items() if key != 'receipt_sha256'}):
            raise ValueError('positive gate evidence digest mismatch')
        if canonical_json_sha256(obj['replay_identity']) != obj['replay_identity_sha256']:
            raise ValueError('positive gate evidence replay identity SHA mismatch')
        return obj
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(f'Positive gate Audit evidence invalid: {path.name}: {exc}') from exc


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    if hasattr(value, 'item'):
        value = value.item()
    if hasattr(value, 'isoformat'):
        return value.isoformat()[:10]
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite positive gate execution evidence')
        return value
    raise ValueError(f'unsupported positive gate evidence scalar: {type(value).__name__}')


def _pick(row, fields):
    return {key: _scalar(row.get(key)) for key in fields}


def save_positive_replay_evidence(
    *, project_root: Path, output_subdir: str, seed: int,
    replay_identity: Mapping[str, Any], portfolio: Mapping[str, Any],
    execution_rows: list[Mapping[str, Any]],
    raw_net_pnl: Mapping[tuple[str, str], float], summary: Mapping[str, Any],
) -> Path:
    closed = list(portfolio['profile'].get('closed_trade_rows') or [])
    if len(closed) != int(summary['trades']):
        raise ValueError('Positive gate completed Round-Trip count is not canonical')
    filled = []
    for entry in execution_rows:
        if not bool(entry.get('entry_filled')):
            continue
        filled.append(_pick(entry, (
            'ticker', 'entry_source', 'type', 'signal_date', 'candidate_date',
            'trade_date', 'limit_px', 'init_sl', 'chosen_qty', 'filled_qty',
            'candidate_sizing_capital', 'sizing_equity', 'actual_initial_risk_total_milli',
        )))
    rows = []
    outcomes = {'profit': 0, 'loss': 0, 'zero': 0}
    for trade in closed:
        ticker = str(trade.get('ticker') or '')
        day = str(trade.get('breakout_quality_score_date') or '')[:10]
        raw = raw_net_pnl.get((ticker, day))
        # A missing score or nonpositive raw outcome can never be accepted by
        # this arm. If a second entry path evades the Selector, fail the RUN.
        if raw is None or not math.isfinite(raw) or raw <= 0:
            raise ValueError(f'Positive gate accepted a NULL/nonpositive completed trade: {ticker} {day}')
        entry = _pick(trade, (
            'ticker', 'entry_trade_date', 'signal_date', 'candidate_date',
            'breakout_quality_score_date', 'entry_type', 'candidate_type',
            'entry_price', 'exit_price', 'exit_type', 'exit_date', 'pnl', 'r_mult',
        ))
        entry['target_raw_net_pnl'] = float(raw)
        result = float(trade['pnl'])
        outcomes['profit' if result > 0 else 'loss' if result < 0 else 'zero'] += 1
        rows.append(entry)
    total = len(rows)
    expected_win = 0.0 if total == 0 else 100.0 * outcomes['profit'] / total
    if not math.isclose(expected_win, float(summary['win_rate_pct']), abs_tol=1e-8):
        raise ValueError('positive gate evidence win rate differs from formal Portfolio SSOT')
    data = {
        'schema': SCHEMA, 'target': TARGET, 'gate': POSITIVE_FULLCYCLE_NETPNL_GATE,
        'seed': int(seed), 'replay_identity': dict(replay_identity),
        'replay_identity_sha256': canonical_json_sha256(replay_identity),
        'summary': {'trades': total, 'win_rate_pct': float(summary['win_rate_pct']),
                    'romd': float(summary['romd']), 'outcomes': outcomes},
        'rows': rows, 'fills': filled,
    }
    data['receipt_sha256'] = canonical_json_sha256(data)
    path = evidence_path(Path(project_root), output_subdir, seed, replay_identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.positive_replay.', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            stream.flush()
            os.fsync(stream.fileno())
        with exclusive_file_lock(path.parent / '.publish.lock'):
            if path.is_file():
                if read_evidence(path) != data:
                    raise ValueError('positive gate same Replay identity produced conflicting closed trades')
            else:
                atomic_replace_with_retry(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path

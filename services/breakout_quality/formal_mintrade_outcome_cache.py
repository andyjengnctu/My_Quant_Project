"""Disposable, content-verified per-stock memoization of canonical Min executions.

Only the *execution batch* is cached. The original, seed-verified Optimizer
teacher and all fold/PIT maturity and sample gates are checked on every call.
NetR and NetPnL are two values of that one execution, never separate paths.
This cache is not an Oracle publication, a model artifact or a source of truth.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
import os
import sys
from pathlib import Path
from uuid import uuid4
from typing import Callable
from zipfile import BadZipFile
import zlib

import numpy as np
import pandas as pd

from filters.breakout_quality.daily_trade40_target import CounterfactualTrade40Batch

_CACHE_SCHEMA = 1
_CACHE_INPUT_ROOTS = (
    "core", "filters/breakout_quality", "services/breakout_quality",
    "strategies/breakout",
)
_CACHE_EXECUTION_CONFIG = (
    "config/broker_accounting.py", "config/execution_policy.py",
    "config/strategy_params.py", "config/breakout_policy.py",
    "config/training_policy.py",
)
_CACHE_SUBDIR = Path("outputs/filters/breakout_quality/formal_mintrade_outcome_cache")


@lru_cache(maxsize=1)
def _implementation_digest() -> str:
    """Conservatively expire shards after any relevant executable-code change.

    Cover the entire dependency domains rather than relying on a manually
    updated version number or accidentally missing an imported accounting rule.
    """
    code_root = Path(__file__).resolve().parents[2]
    digest = sha256()
    for folder in _CACHE_INPUT_ROOTS:
        for path in sorted((code_root / folder).rglob("*.py")):
            digest.update(path.relative_to(code_root).as_posix().encode("utf-8"))
            digest.update(sha256(path.read_bytes()).digest())
    for name in _CACHE_EXECUTION_CONFIG:
        path = code_root / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(sha256(path.read_bytes()).digest())
    # The native C signal kernels are built outside the source tree. Include
    # their source and any active binary, not the entire transient outputs dir.
    source = code_root / "core/native_optimizer_hotpath.c"
    if source.is_file():
        digest.update(sha256(source.read_bytes()).digest())
    native_root = code_root / "outputs/ml_optimizer/native"
    for suffix in ("*.pyd", "*.so", "*.dll"):
        for path in sorted(native_root.glob(suffix)):
            digest.update(path.name.encode("utf-8"))
            digest.update(sha256(path.read_bytes()).digest())
    # Signal/parameter defaults are imported from the wide model config. Hash
    # only execution-relevant defaults so changing a model comparison list or
    # this CPU worker knob does not invalidate correct broker outcomes.
    from config.breakout_quality import (
        BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    )
    digest.update(json.dumps({
        "default_filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        "default_score_threshold": BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    }, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _cache_identity(
    *, frame: pd.DataFrame, positions: np.ndarray, ticker: str,
    exchange_calendar: pd.DatetimeIndex, source_sha256: str,
    seed: int, schedule_signatures: tuple[str, ...],
    horizon_sessions: int | None, cutoff: pd.Timestamp,
    signal_state_reset_date: str | None,
) -> str:
    digest = sha256()
    header = {
        "schema": _CACHE_SCHEMA,
        "execution_code_sha256": _implementation_digest(),
        "ticker": str(ticker),
        "source_sha256": str(source_sha256),
        "seed": int(seed),
        "schedule_signatures": schedule_signatures,
        "horizon_sessions": horizon_sessions,
        "cutoff": str(cutoff.date()),
        "signal_state_reset_date": signal_state_reset_date,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "python_cache_tag": sys.implementation.cache_tag,
        "frame_columns": list(map(str, frame.columns)),
        "frame_attrs": frame.attrs,
    }
    digest.update(json.dumps(header, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8"))
    # Hash every sanitized source column, including index and unusual optional
    # columns; the simulator may depend on more than the minimum five OHLCV.
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64).tobytes())
    digest.update(np.asarray(positions, dtype=np.int64).tobytes())
    digest.update(pd.DatetimeIndex(exchange_calendar).asi8.tobytes())
    return digest.hexdigest()


def _read_cache(path: Path, key: str, count: int) -> CounterfactualTrade40Batch | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {
                "identity", "net_r", "net_pnl", "valid", "maturity",
                "reasons", "forced", "truncated", "cutoff_closeouts",
            } or str(data["identity"].item()) != key:
                return None
            net_r = np.asarray(data["net_r"], dtype=np.float32)
            net_pnl = np.asarray(data["net_pnl"], dtype=np.float64)
            valid = np.asarray(data["valid"], dtype=bool)
            maturity = np.asarray(data["maturity"], dtype="datetime64[D]")
            reasons = tuple(str(value) for value in data["reasons"])
            forced = np.asarray(data["forced"], dtype=bool)
            truncated = int(data["truncated"].item())
            closeouts = int(data["cutoff_closeouts"].item())
        if not all(arr.ndim == 1 and len(arr) == count for arr in (
            net_r, net_pnl, valid, maturity, forced,
        )) or len(reasons) != count:
            return None
        if (np.any(forced & ~valid) or np.any(~np.isfinite(net_r[valid]))
                or np.any(~np.isfinite(net_pnl[valid]))
                or np.any(np.isfinite(net_r[~valid]))
                or np.any(np.isfinite(net_pnl[~valid]))
                or np.any(np.isnat(maturity[valid]))
                or not 0 <= closeouts <= truncated <= count):
            return None
        return CounterfactualTrade40Batch(
            target_raw_r=net_r, valid_mask=valid, maturity_dates=maturity,
            reasons=reasons, cutoff_truncated_rows=truncated,
            research_cutoff_closeout_rows=closeouts, target_raw_pnl=net_pnl,
            research_cutoff_forced_mask=forced,
        )
    except (OSError, BadZipFile, zlib.error, ValueError, KeyError, TypeError, EOFError, OverflowError):
        return None


def _write_cache(path: Path, key: str, batch: CounterfactualTrade40Batch) -> None:
    if batch.target_raw_pnl is None or batch.research_cutoff_forced_mask is None:
        raise ValueError("Canonical trade-path cache requires both broker outcomes and cutoff evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{os.getpid()}.{uuid4().hex}.npz"
    try:
        np.savez_compressed(
            temporary,
            identity=np.asarray(key),
            net_r=np.asarray(batch.target_raw_r, dtype=np.float32),
            net_pnl=np.asarray(batch.target_raw_pnl, dtype=np.float64),
            valid=np.asarray(batch.valid_mask, dtype=bool),
            maturity=np.asarray(batch.maturity_dates, dtype="datetime64[D]"),
            reasons=np.asarray(batch.reasons, dtype=str),
            forced=np.asarray(batch.research_cutoff_forced_mask, dtype=bool),
            truncated=np.asarray(batch.cutoff_truncated_rows, dtype=np.int64),
            cutoff_closeouts=np.asarray(batch.research_cutoff_closeout_rows, dtype=np.int64),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_compute_formal_mintrade_outcomes(
    *, project_root: Path, frame: pd.DataFrame, positions: np.ndarray,
    ticker: str, exchange_calendar: pd.DatetimeIndex, source_sha256: str,
    seed: int, schedule_signatures: tuple[str, ...],
    horizon_sessions: int | None, research_cutoff: pd.Timestamp,
    signal_state_reset_date: str | None,
    compute: Callable[[], CounterfactualTrade40Batch],
) -> CounterfactualTrade40Batch:
    """Recheck all exact inputs; cache hits skip only the expensive simulator."""
    key = _cache_identity(
        frame=frame, positions=positions, ticker=ticker,
        exchange_calendar=exchange_calendar, source_sha256=source_sha256,
        seed=seed, schedule_signatures=schedule_signatures,
        horizon_sessions=horizon_sessions, cutoff=research_cutoff,
        signal_state_reset_date=signal_state_reset_date,
    )
    # One replaceable shard per ticker/seed/horizon/cutoff. Source/data/code
    # changes replace stale bytes instead of accumulating every research edit.
    # The full content identity stays inside the file and is ALWAYS checked.
    slot = sha256(json.dumps({
        "ticker": str(ticker), "seed": int(seed),
        "horizon_sessions": horizon_sessions,
        "cutoff": str(pd.Timestamp(research_cutoff).date()),
    }, sort_keys=True).encode("utf-8")).hexdigest()
    path = Path(project_root) / _CACHE_SUBDIR / slot[:2] / f"{slot}.npz"
    hit = _read_cache(path, key, len(positions))
    if hit is not None:
        return hit
    batch = compute()
    _write_cache(path, key, batch)
    return batch

"""Canonical complete holding episodes and reusable causal input windows."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

from core.file_integrity import atomic_replace_with_retry, canonical_json_sha256, compute_observed_file_sha256
from filters.breakout_quality.holding_value import HoldingFeatures, build_ticker_holding_rows
from filters.breakout_quality.workflow_io import load_dataset_frame
from services.breakout_quality.holding_value_model import _write_json
from services.breakout_quality.formal_mintrade40_pit_source import slice_formal_mintrade40_pit_source_for_fold

SAMPLE_CONTRACT = "canonical_complete_natural_and_all_discretionary_paths_with_risk_and_dates_v1"
COLUMNS = ["ticker", "information_date", "episode", "entry_date", "maturity_date", "natural_net_pnl",
           "early_net_pnl", "target", "state", "initial_risk_money", "natural_exit_date", "early_exit_date",
           "decision_allowed", "decision_execution_date"]


def _worker(args):
    ticker, path, teacher, cutoff, destination = args
    diagnostics = {}
    records = build_ticker_holding_rows(ticker=ticker, frame=load_dataset_frame(Path(path), ticker, min_rows=1),
        teacher=teacher, cutoff=cutoff, complete_episode=True, diagnostics=diagnostics)
    destination = Path(destination)
    temporary = destination.with_suffix(".tmp.parquet")
    pd.DataFrame(records, columns=COLUMNS).to_parquet(temporary, index=False)
    atomic_replace_with_retry(temporary, destination)
    return {"ticker": ticker, "states": len(records), "sha256": compute_observed_file_sha256(destination),
            "excluded": diagnostics}


def build_episode_rows(root, *, seed, teacher, cutoff, inputs, implementation):
    prefix, prefix_identity = slice_formal_mintrade40_pit_source_for_fold(teacher, fit_cutoff=pd.Timestamp(cutoff))
    files = {ticker: compute_observed_file_sha256(path) for ticker, path in inputs}
    identity = {"contract": SAMPLE_CONTRACT, "seed": int(seed), "cutoff": str(pd.Timestamp(cutoff).date()),
                "teacher_prefix": prefix_identity, "implementation": implementation, "files": files}
    fingerprint = canonical_json_sha256(identity)
    cache = Path(root) / "outputs/research/breakout_quality/holding_episode_samples" / f"seed_{seed}" / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    receipt_path = cache / "receipt.json"
    previous = json.loads(receipt_path.read_text("utf-8")) if receipt_path.exists() else {}
    if previous and previous.get("identity") != identity:
        raise ValueError("Complete episode cache identity collision")
    completed = dict(previous.get("tickers", {}))
    pending = []
    for ticker, path in inputs:
        if ticker == "0050":
            continue
        dest = cache / (ticker + ".parquet")
        record = completed.get(ticker)
        if record and dest.exists() and compute_observed_file_sha256(dest) == record["sha256"]:
            continue
        pending.append((ticker, str(path), prefix, identity["cutoff"], str(dest)))
    if pending:
        with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as pool:
            futures = [pool.submit(_worker, args) for args in pending]
            for number, future in enumerate(as_completed(futures), 1):
                record = future.result()
                completed[record["ticker"]] = record
                _write_json(receipt_path, {"identity": identity, "tickers": completed})
                if number % 20 == 0 or number == len(futures):
                    print(f"[完整交易 seed={seed}] 樣本 {number}/{len(futures)} stocks", flush=True)
    tables = [pd.read_parquet(cache / (ticker + ".parquet")) for ticker, _ in inputs
              if ticker != "0050" and completed[ticker]["states"]]
    if not tables:
        raise ValueError("No complete mature holding episodes")
    rows = pd.concat(tables, ignore_index=True).sort_values(["episode", "information_date"], kind="stable").reset_index(drop=True)
    rows["ticker"] = rows.ticker.astype(str)
    exclusions = {}
    for record in completed.values():
        for key, value in record["excluded"].items():
            exclusions[key] = exclusions.get(key, 0) + value
    evidence = {"identity": identity, "receipt": str(receipt_path),
                "receipt_sha256": compute_observed_file_sha256(receipt_path),
                "episodes": int(rows.episode.nunique()), "states": len(rows), "stocks": int(rows.ticker.nunique()),
                "excluded": exclusions}
    return rows, evidence


class CachedHoldingFeatures:
    """Compute each stock/date window once; never normalize across future dates."""
    def __init__(self, rows, frames):
        self.frames = frames
        source = HoldingFeatures(frames, frames["0050"])
        keys = rows[["ticker", "information_date"]].drop_duplicates()
        eligible = [source.coordinates(ticker, pd.Timestamp(day)) is not None
                    for ticker, day in keys.itertuples(index=False, name=None)]
        keys = keys.loc[eligible].reset_index(drop=True)
        self.windows = np.zeros((len(keys) + 1, 300, 10), dtype=np.float32)
        for offset in range(0, len(keys), 512):
            batch = keys.iloc[offset:offset + 512].assign(state=lambda x: [[0.] * 6] * len(x))
            self.windows[offset + 1:offset + 1 + len(batch)] = source.batch(batch)[0]
        index = {(ticker, day): n + 1 for n, (ticker, day) in enumerate(keys.itertuples(index=False, name=None))}
        rows["feature_index"] = [index.get((ticker, day), 0) for ticker, day in zip(rows.ticker, rows.information_date)]
        rows["decision_allowed"] = rows.decision_allowed.astype(bool) & rows.feature_index.ne(0)
        print(f"[完整交易] 已準備 {len(keys):,} 個唯一價量視窗；保留 {rows.episode.nunique():,} 筆完整路徑", flush=True)

    def batch(self, rows):
        return self.windows[rows.feature_index.to_numpy(dtype=np.int64)], np.asarray(rows.state.tolist(), dtype=np.float32)

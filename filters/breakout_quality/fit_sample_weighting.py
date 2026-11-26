"""Canonical signal episodes, stage-local frequency weights and pair-mass audit.

Episodes are projected from the existing label lifecycle, never reconstructed
from trade outcomes or calendar gaps. Only selected FIT rows enter a count.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from core.breakout_quality_fit_weighting import EPISODE_COLUMN, fit_sample_weight_contract


def attach_fit_episodes(table, *, frames, tickers, target_valid, source,
                       exchange_calendar, policy):
    """Opt-in metadata projection, preserving every original row and its order."""
    if fit_sample_weight_contract(policy) is None:
        return
    from filters.breakout_quality.daily_trade40_target import compute_counterfactual_trade40_batch

    if source is None or type(source.seed) is not int or source.seed <= 0:
        raise ValueError("Signal frequency metadata requires the verified same-seed teacher")
    valid = np.asarray(target_valid, dtype=bool)
    if valid.shape != (len(table),):
        raise ValueError("FIT episode validity mask is not aligned")
    result = np.full(len(table), None, dtype=object)
    legal_ids = np.flatnonzero(valid)
    # Group once; do not scan the entire stock-day table once per ticker.
    groups = table.iloc[legal_ids].groupby("ticker_id", observed=True, sort=False).indices
    for ticker_id, offsets in groups.items():
        ids = legal_ids[offsets]
        positions = table.iloc[ids]["source_pos"].to_numpy(dtype=np.int64)
        ticker = str(tickers[int(ticker_id)])
        frame = frames[int(ticker_id)]
        if not (table.iloc[ids]["ticker"].astype(str) == ticker).all():
            raise ValueError("FIT episode ticker/frame identity mismatch")
        keys = {}
        compute_counterfactual_trade40_batch(
            frame, positions, ticker=ticker, schedule=source.schedule,
            exchange_calendar=exchange_calendar, horizon_sessions=None,
            signal_state_reset_date=source.oos_start_date,
            episode_keys_out=keys, episode_keys_only=True,
        )
        if set(keys) != set(map(int, positions)):
            raise ValueError(f"Missing canonical signal episode for legal FIT rows: {ticker}")
        result[ids] = [f"{source.seed}|{keys[int(pos)]}" for pos in positions]
    table[EPISODE_COLUMN] = result


def _fit_episode_rows(table, fit_ids):
    ids = np.asarray(fit_ids, dtype=np.int64)
    if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= len(table))):
        raise ValueError("Invalid ordered FIT ids for signal frequency")
    if EPISODE_COLUMN not in table:
        raise ValueError("Missing canonical FIT episode metadata")
    rows = table.iloc[ids]
    if rows.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/information-date in signal frequency FIT")
    keys = rows[EPISODE_COLUMN]
    if keys.isna().any() or not keys.map(lambda x: isinstance(x, str)).all():
        raise ValueError("Missing or ambiguous FIT signal episode")
    parts = keys.str.split("|", expand=True)
    if parts.shape[1] != 3 or not parts[0].str.fullmatch(r"[1-9][0-9]*").all():
        raise ValueError("Invalid same-seed signal episode key")
    origins = pd.to_datetime(parts[2], format="%Y-%m-%d", errors="raise")
    dates = pd.to_datetime(rows["date"], errors="raise")
    maturity = pd.to_datetime(rows["label_eval_end_date"], errors="raise")
    if (parts[0].nunique() != 1 or not np.array_equal(parts[1], rows["ticker"].astype(str))
            or dates.isna().any() or origins.isna().any() or maturity.isna().any()
            or (origins.to_numpy() > dates.to_numpy()).any()
            or (maturity.to_numpy() <= dates.to_numpy()).any()):
        raise ValueError("Inconsistent seed/ticker/origin/maturity in FIT episode metadata")
    return ids, rows, keys.to_numpy(dtype=str)


def ordered_fit_episode_sha256(table, fit_ids, policy):
    contract = fit_sample_weight_contract(policy)
    if contract is None:
        return None
    _, _, keys = _fit_episode_rows(table, fit_ids)
    digest = sha256(json.dumps(contract, sort_keys=True).encode("utf-8"))
    for key in keys:
        digest.update((str(key) + "\n").encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FitSampleWeights:
    values: np.ndarray
    episode_codes: np.ndarray
    counts: np.ndarray
    summary: dict


def build_fit_sample_weights(table, fit_ids, *, policy, cutoff, raw_target, expected_seed):
    contract = fit_sample_weight_contract(policy)
    if contract is None:
        return None
    ids, rows, keys = _fit_episode_rows(table, fit_ids)
    if type(expected_seed) is not int or expected_seed <= 0 or any(
            key.split("|", 1)[0] != str(expected_seed) for key in keys):
        raise ValueError("FIT episode seed differs from the fitting seed")
    if cutoff is None or pd.isna(pd.Timestamp(cutoff)):
        raise ValueError("Signal frequency FIT requires its exclusive stage cutoff")
    end = pd.Timestamp(cutoff).normalize()
    if ((pd.to_datetime(rows["date"]) >= end).any()
            or (pd.to_datetime(rows["label_eval_end_date"]) >= end).any()
            or not np.isfinite(np.asarray(raw_target)[ids]).all()):
        raise ValueError("Signal frequency FIT includes immature/NULL/future rows")
    codes, unique = pd.factorize(keys, sort=False)
    counts = np.bincount(codes)
    values = np.full(len(table), np.nan, dtype=np.float32)
    values[ids] = 1.0 / counts[codes]
    episode_codes = np.full(len(table), -1, dtype=np.int64)
    episode_codes[ids] = codes
    for array in (values, episode_codes, counts):
        array.flags.writeable = False
    top = max(1, int(np.ceil(len(counts) * .1)))
    summary = {
        "contract": contract, "exclusive_fit_cutoff": str(end.date()),
        "ordered_episode_sha256": ordered_fit_episode_sha256(table, ids, policy),
        "rows": int(len(ids)), "episodes": int(len(unique)),
        "stocks": int(rows["ticker"].nunique()), "dates": int(rows["date"].nunique()),
        "rows_per_episode_median": float(np.median(counts)),
        "rows_per_episode_max": int(counts.max()),
        "top_10pct_episode_row_share": float(np.sort(counts)[-top:].sum() / len(ids)),
        "base_item_mass_per_episode": 1.0,
    }
    return FitSampleWeights(values, episode_codes, counts, summary)


class PairMassAudit:
    """Mean per-update normalized pair endpoint mass, NOT gradient attribution.

    Before/after use the same current predicted margins. Each pair contributes
    half its normalized weight to each endpoint. Batch weights normalize once
    across all dates, exactly as the loss. No alternative model is evaluated.
    """
    def __init__(self, fit_weights):
        self.fit_weights = fit_weights
        self.before = np.zeros(len(fit_weights.counts), dtype=np.float64)
        self.after = np.zeros_like(self.before)
        self.batches = 0

    def batch(self, ids):
        return _BatchPairMass(self, np.asarray(ids, dtype=np.int64))

    def summary(self):
        def concentration(values):
            if not self.batches:
                return None
            shares = values / self.batches
            top = max(1, int(np.ceil(len(shares) * .1)))
            return {"max_episode_share": float(shares.max()),
                    "top_10pct_episode_share": float(np.sort(shares)[-top:].sum()),
                    "effective_episodes": float(1.0 / np.square(shares).sum())}
        return {"metric": "mean_batch_normalized_pair_endpoint_mass_not_gradient",
                "reference": "original_delta_ndcg_at_same_current_margins",
                "optimizer_batches": self.batches,
                "before": concentration(self.before), "after": concentration(self.after)}


class _BatchPairMass:
    def __init__(self, owner, ids):
        self.owner, self.ids = owner, ids
        self.before = self.after = None

    def observe(self, torch, positions, comparable, before, after):
        if self.before is None:
            self.before = torch.zeros(len(self.ids), dtype=before.dtype, device=before.device)
            self.after = torch.zeros_like(self.before)
        endpoints = torch.nonzero(comparable, as_tuple=False)
        for axis in (0, 1):
            selected = positions.index_select(0, endpoints[:, axis])
            self.before.index_add_(0, selected, before.detach())
            self.after.index_add_(0, selected, after.detach())

    def finish(self):
        if self.before is None:
            return
        codes = self.owner.fit_weights.episode_codes[self.ids]
        if (codes < 0).any():
            raise ValueError("Pair audit encountered rows outside its FIT cohort")
        for values, destination in ((self.before, self.owner.before), (self.after, self.owner.after)):
            shares = (values / values.sum()).detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(shares).all():
                raise ValueError("Invalid normalized pair endpoint mass")
            destination += np.bincount(codes, weights=shares, minlength=len(destination))
        self.owner.batches += 1

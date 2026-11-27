"""PIT fitting of whole-episode stopping policies; no replay or roster ownership."""
from __future__ import annotations

from pathlib import Path
import json
import os
import random
import time
import numpy as np
import pandas as pd

from core.file_integrity import atomic_replace_with_retry, compute_observed_file_sha256
from filters.breakout_quality.holding_value import episode_split
from filters.breakout_quality.holding_policy import (
    CONTRACT, build_policy_network, expected_episode_gain, episode_outcomes, outcome_metrics,
)
from services.breakout_quality.holding_value_model import _write_json


def configure_torch():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_policy(model, rows, features, norm, device):
    import torch
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), 512):
            x, state = features.batch(rows.iloc[start:start + 512])
            logits = model(torch.as_tensor(x, device=device),
                           torch.as_tensor((state - norm[0]) / norm[1], device=device))
            result.extend(logits.cpu().numpy().tolist())
    return np.asarray(result)


def _fit_fold(rows, features, *, start, seed, directory, fit_identity):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "fold.json"
    if path.exists():
        stored = json.loads(path.read_text("utf-8"))
        if stored.get("fit_identity") == fit_identity and all(
                compute_observed_file_sha256(directory / item["weights"]) == item["weights_sha256"]
                for item in stored["models"].values()):
            print(f"[HV2 seed={seed}] REUSE {start.date()}", flush=True)
            return stored
        raise ValueError("Published stopping FIT has conflicting identity or weights")
    train, validation, final = episode_split(rows, cutoff=start, validation_start=start - pd.DateOffset(years=1))
    tr, va, refit = (rows.iloc[indices].reset_index(drop=True) for indices in (train, validation, final))
    if (tr.episode.nunique() < CONTRACT["min_train_episodes"] or
            va.episode.nunique() < CONTRACT["min_validation_episodes"]):
        raise ValueError(f"Insufficient complete stopping episodes before {start.date()}")
    device = configure_torch()

    def norm(data):
        state = np.asarray(data.state.tolist(), dtype=np.float32)
        return state.mean(axis=0), np.maximum(state.std(axis=0), 1e-6)

    def reset(kind):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return build_policy_network(kind).to(device)

    def epoch(model, optimizer, data, normalization, rng):
        model.train()
        groups = list(data.groupby("episode", sort=False).indices.values())
        order = rng.permutation(len(groups))
        total_gain = 0.0
        for offset in range(0, len(order), CONTRACT["episode_batch_size"]):
            group = [groups[i] for i in order[offset:offset + CONTRACT["episode_batch_size"]]]
            ids = np.concatenate(group)
            batch = data.iloc[ids]
            sequence, state = features.batch(batch)
            logits = model(torch.as_tensor(sequence, device=device),
                           torch.as_tensor((state - normalization[0]) / normalization[1], device=device))
            gain = expected_episode_gain(logits,
                torch.as_tensor(-batch.target.to_numpy(dtype=np.float32), device=device),
                torch.as_tensor(batch.decision_allowed.to_numpy(dtype=bool, copy=True), device=device), [len(g) for g in group])
            loss = -gain.sum() / CONTRACT["episode_batch_size"]
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite whole-episode stopping objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONTRACT["gradient_clip"])
            optimizer.step()
            total_gain += float(gain.detach().sum().cpu())
        return total_gain / len(groups)

    models = {}
    fit_norm = norm(tr)
    final_norm = norm(refit)
    for kind in ("dl", "linear"):
        model = reset(kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONTRACT["learning_rate"])
        rng = np.random.default_rng(seed)
        # The original policy is an explicit zero-gain fallback, selected only
        # with FIT validation. It is never mistaken for evidence of useful DL.
        baseline = outcome_metrics(episode_outcomes(va, np.full(len(va), -100.0)))
        best_gain, best_epoch, best_metrics = 0.0, 0, baseline
        curve = []
        for e in range(1, CONTRACT["max_epochs"] + 1):
            clock = time.monotonic()
            training_gain = epoch(model, optimizer, tr, fit_norm, rng)
            prediction = predict_policy(model, va, features, fit_norm, device)
            metrics = outcome_metrics(episode_outcomes(va, prediction))
            curve.append({"epoch": e, "train_expected_gain_r": training_gain, "validation": metrics})
            if metrics["mean_gain_r"] > best_gain:
                best_gain, best_epoch, best_metrics = metrics["mean_gain_r"], e, metrics
            print(f"[HV2 seed={seed}] {start.year} {kind} epoch {e}/{CONTRACT['max_epochs']} "
                  f"val ΔR={metrics['mean_gain_r']:+.6f} exits={metrics['early_exit_fraction']:.2%} "
                  f"{time.monotonic()-clock:.1f}s", flush=True)
        model = reset(kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONTRACT["learning_rate"])
        rng = np.random.default_rng(seed)
        for e in range(1, best_epoch + 1):
            epoch(model, optimizer, refit, final_norm, rng)
            print(f"[HV2 seed={seed}] {start.year} {kind} refit {e}/{best_epoch}", flush=True)
        weights = directory / (kind + ".pt")
        temp = weights.with_suffix(".tmp.pt")
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "state_mean": torch.tensor(final_norm[0]), "state_scale": torch.tensor(final_norm[1]),
                    "seed": int(seed), "fit_identity": fit_identity, "kind": kind}, temp)
        atomic_replace_with_retry(temp, weights)
        models[kind] = {"weights": weights.name, "weights_sha256": compute_observed_file_sha256(weights),
                        "selected_epochs": best_epoch, "selected_policy": "learned" if best_epoch else "always_continue",
                        "validation": best_metrics, "curve": curve}
    result = {"fit_identity": fit_identity, "cutoff": str(start.date()), "seed": int(seed),
              "fit_max_maturity": str(refit.maturity_date.max()),
              "train_episodes": int(tr.episode.nunique()), "validation_episodes": int(va.episode.nunique()),
              "refit_episodes": int(refit.episode.nunique()), "train_states": len(tr),
              "validation_states": len(va), "refit_states": len(refit), "models": models}
    _write_json(path, result)
    return result

"""PIT producer/runtime for the independent position continuation model.

Research owns scheduling, controls, replay and reporting. This producer owns
holding samples, fold fits, provenance and conditional inference only.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import random
import time
import numpy as np
import pandas as pd

from core.dataset_profiles import build_dataset_generation_identity
from core.file_integrity import atomic_replace_with_retry, canonical_json_sha256, compute_observed_file_sha256
from core.position_exit_policy import bind_position_exit_provider, position_exit_spec
from filters.breakout_quality.holding_value import (
    CONTRACT, MODEL_ID, HoldingFeatures, build_network, build_ticker_holding_rows,
    episode_split, position_context,
)
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame
from services.breakout_quality.formal_mintrade40_training_source import load_formal_mintrade40_training_source
from services.breakout_quality.formal_mintrade40_pit_source import slice_formal_mintrade40_pit_source_for_fold


def configuration(settings, arm):
    if position_exit_spec(arm.dl_runtime_options) is None:
        raise ValueError("Arm has no position exit model")
    source = settings.dl_sources[arm.dl_id]
    single = bool(source.point_in_time_single_score_block)
    return {"dataset": settings.dataset, "start": settings.start_date, "end": settings.end_date,
            "mode": "oos" if single else "rolling", "months": int(source.point_in_time_fold_months or 12)}


def model_directory(root, *, seed, config):
    return Path(root) / "models/research/breakout_quality/holding_value" / MODEL_ID / f"seed_{seed}" / config["mode"]


def fold_ranges(config):
    start, end = pd.Timestamp(config["start"]), pd.Timestamp(config["end"])
    result = []
    while start <= end:
        following = end + pd.Timedelta(days=1) if config["mode"] == "oos" else start + pd.DateOffset(months=config["months"])
        result.append((start, min(end, following - pd.Timedelta(days=1))))
        start = following
    return tuple(result)


def implementation_identity(root):
    from core.execution_source_dependencies import execution_source_inventory
    names = (*execution_source_inventory("mintrade_truth", "daily_prices"),
             "core/position_exit_policy.py", "filters/breakout_quality/holding_value.py",
             "services/breakout_quality/holding_value_model.py",
             "services/breakout_quality/formal_mintrade40_training_source.py",
             "services/breakout_quality/formal_mintrade40_pit_source.py")
    return canonical_json_sha256({name: compute_observed_file_sha256(Path(root) / name) for name in sorted(set(names))})


def publication_identity(root, *, seed, config, teacher=None):
    teacher = teacher or load_formal_mintrade40_training_source(project_root=root, seed=int(seed))
    return {"contract": CONTRACT, "seed": int(seed), "configuration": config,
            "implementation": implementation_identity(root),
            "data_generation": build_dataset_generation_identity(root, config["dataset"]),
            "teacher_sha256": teacher.param_sha256}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    atomic_replace_with_retry(temporary, path)


def inspect_holding_model(root, *, seed, config):
    directory = model_directory(root, seed=seed, config=config)
    manifest_path = directory / "manifest.json"
    try:
        expected = publication_identity(root, seed=seed, config=config)
        if not manifest_path.is_file():
            return {"ready": False, "path": str(manifest_path), "reason": "尚未訓練持倉模型"}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != expected or manifest.get("status") != "COMPLETED":
            raise ValueError("持倉模型來源或契約已變更")
        requested = [(str(a.date()), str(b.date())) for a, b in fold_ranges(config)]
        if [(f["start"], f["end"]) for f in manifest["folds"]] != requested:
            raise ValueError("持倉模型年度覆蓋不完整")
        for fold in manifest["folds"]:
            if not (pd.Timestamp(fold["fit_max_maturity"]) < pd.Timestamp(fold["start"])):
                raise ValueError("持倉模型 FIT 成熟時間不合法")
            for key in ("weights", "metrics"):
                path = directory / fold[key]
                if path.resolve().parent != directory.resolve() or compute_observed_file_sha256(path) != fold[key + "_sha256"]:
                    raise ValueError("持倉模型工件校驗失敗")
        return {"ready": True, "path": str(manifest_path), "reason": "PIT 持倉模型可重用",
                "sha256": compute_observed_file_sha256(manifest_path), "manifest": manifest}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"ready": False, "path": str(manifest_path), "reason": str(exc)}


def _ticker_worker(arguments):
    ticker, path, teacher, cutoff, output = arguments
    frame = load_dataset_frame(Path(path), ticker, min_rows=1)
    records = build_ticker_holding_rows(ticker=ticker, frame=frame, teacher=teacher, cutoff=cutoff)
    target = Path(output)
    temporary = target.with_suffix(".tmp.parquet")
    pd.DataFrame(records, columns=["ticker", "information_date", "episode", "entry_date", "maturity_date",
                                   "natural_net_pnl", "early_net_pnl", "target", "state"]).to_parquet(temporary, index=False)
    atomic_replace_with_retry(temporary, target)
    return {"ticker": ticker, "rows": len(records), "sha256": compute_observed_file_sha256(target)}


def _build_rows(root, *, seed, config, teacher, inputs, identity):
    cutoff = fold_ranges(config)[-1][0] - pd.Timedelta(days=1)
    prefix, _ = slice_formal_mintrade40_pit_source_for_fold(teacher, fit_cutoff=cutoff)
    source_files = {ticker: compute_observed_file_sha256(path) for ticker, path in inputs}
    fingerprint = canonical_json_sha256({"identity": identity, "cutoff": str(cutoff.date()), "files": source_files})
    cache = Path(root) / "outputs/research/breakout_quality/holding_value_samples" / f"seed_{seed}" / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    receipt = cache / "receipt.json"
    previous = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
    completed = dict(previous.get("tickers", {}))
    pending = []
    for ticker, path in inputs:
        if ticker == "0050":
            continue
        output = cache / f"{ticker}.parquet"
        record = completed.get(ticker)
        if record and output.is_file() and compute_observed_file_sha256(output) == record["sha256"]:
            continue
        pending.append((ticker, str(path), prefix, str(cutoff.date()), str(output)))
    if pending:
        # Bounded CPU preparation; no nested model/GPU workers.
        with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as pool:
            futures = [pool.submit(_ticker_worker, args) for args in pending]
            for index, future in enumerate(futures):
                record = future.result()
                completed[record["ticker"]] = record
                _write_json(receipt, {"fingerprint": fingerprint, "tickers": completed})
                if index % 20 == 0 or index == len(futures) - 1:
                    print(f"[HV1 seed={seed}] 持倉資料 {index + 1}/{len(futures)} stocks", flush=True)
    frames = [pd.read_parquet(cache / f"{ticker}.parquet") for ticker, _ in inputs
              if ticker != "0050" and completed[ticker]["rows"]]
    if not frames:
        raise ValueError("No mature canonical holding episodes")
    rows = pd.concat(frames, ignore_index=True)
    rows["ticker"] = rows.ticker.astype(str)
    return rows, {"fingerprint": fingerprint, "stocks": int(rows.ticker.nunique()),
                  "episodes": int(rows.episode.nunique()), "states": len(rows),
                  "cutoff": str(cutoff.date()), "receipt": str(receipt)}


def _fit_fold(rows, features, *, start, end, seed, directory, fit_identity):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    train, validation, final = episode_split(rows, cutoff=start, validation_start=start - pd.DateOffset(years=1))
    if len(train) < 256 or len(validation) < 64:
        raise ValueError(f"Insufficient mature holding train/validation states: {len(train)}/{len(validation)}")
    name = "fold_" + start.strftime("%Y%m%d")
    metrics_path, weights_path = directory / (name + ".json"), directory / (name + ".pt")
    if metrics_path.exists() and weights_path.exists():
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        if stored.get("fit_identity") == fit_identity and stored.get("weights_sha256") == compute_observed_file_sha256(weights_path):
            print(f"[HV1 seed={seed}] REUSE {name}", flush=True)
            return stored
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    def reset():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return build_network().to(device)
    contexts = np.asarray(rows.state.tolist(), dtype=np.float32)
    target = rows.target.to_numpy(dtype=np.float32)
    def normalization(indices):
        mean = contexts[indices].mean(axis=0)
        scale = np.maximum(contexts[indices].std(axis=0), 1e-6)
        target_scale = max(float(np.std(target[indices])), 1e-6)
        return mean, scale, target_scale
    def weights(indices):
        counts = rows.iloc[indices].episode.value_counts()
        values = rows.iloc[indices].episode.map(lambda key: 1.0 / counts[key]).to_numpy(dtype=np.float32)
        return values / values.mean()
    def batches(indices, norm, *, rng=None):
        order = np.arange(len(indices))
        if rng is not None:
            rng.shuffle(order)
        w = weights(indices)
        for offset in range(0, len(order), CONTRACT["batch_size"]):
            local = order[offset:offset + CONTRACT["batch_size"]]
            ids = indices[local]
            sequence, state = features.batch(rows.iloc[ids])
            yield (torch.as_tensor(sequence, device=device),
                   torch.as_tensor((state - norm[0]) / norm[1], device=device),
                   torch.as_tensor(target[ids] / norm[2], device=device),
                   torch.as_tensor(w[local], device=device), ids)
    def predict(model, indices, norm):
        model.eval()
        predictions = []
        with torch.inference_mode():
            for sequence, state, _, _, _ in batches(indices, norm):
                predictions.extend((model(sequence, state).cpu().numpy() * norm[2]).tolist())
        return np.asarray(predictions)
    def epoch(model, optimizer, indices, norm, rng):
        model.train()
        # Unbiased SGD for episode-equal MSE; every state remains eligible.
        # Avoid fitting hundreds of correlated closes of a long winner on
        # every epoch, without deleting those states or using future outcomes.
        episode_rows = rows.iloc[indices].groupby("episode", sort=False).indices
        sampled = np.asarray([indices[int(rng.choice(local))] for local in episode_rows.values()])
        for sequence, state, y, w, _ in batches(sampled, norm, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(sequence, state) - y).square() * w).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite holding loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    norm = normalization(train)
    model = reset()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONTRACT["learning_rate"])
    rng = np.random.default_rng(seed)
    best, best_epoch, stale, validation_prediction = float("inf"), 0, 0, None
    curve = []
    val_weights = weights(validation)
    for e in range(1, CONTRACT["max_epochs"] + 1):
        epoch(model, optimizer, train, norm, rng)
        prediction = predict(model, validation, norm)
        mse = float(np.average((prediction - target[validation]) ** 2, weights=val_weights))
        curve.append({"epoch": e, "validation_mse": mse})
        print(f"[HV1 seed={seed}] {name} epoch={e} validation MSE={mse:.5f}", flush=True)
        if mse < best:
            best, best_epoch, stale, validation_prediction = mse, e, 0, prediction
        else:
            stale += 1
        if stale >= CONTRACT["patience"]:
            break
    # Low-cost necessity check on the SAME temporal split; no extra C controls.
    # Linear baseline sees causal price/volume summaries plus identical state.
    def linear_inputs(indices):
        result = []
        for offset in range(0, len(indices), CONTRACT["batch_size"]):
            seq, state = features.batch(rows.iloc[indices[offset:offset + CONTRACT["batch_size"]]])
            result.append(np.concatenate((seq[:, -1], seq.mean(axis=1), seq.std(axis=1), state), axis=1))
        return np.concatenate(result)
    x, xv = linear_inputs(train), linear_inputs(validation)
    xm, xs = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    x = np.column_stack((np.ones(len(x)), (x - xm) / xs))
    xv = np.column_stack((np.ones(len(xv)), (xv - xm) / xs))
    w = weights(train).astype(float)
    penalty = np.eye(x.shape[1]); penalty[0, 0] = 0
    coefficient = np.linalg.solve(x.T @ (x * w[:, None]) + penalty, x.T @ (target[train] * w))
    linear = xv @ coefficient
    def metrics(prediction):
        exit_now = prediction < 0
        return {"mse": float(np.average((prediction - target[validation]) ** 2, weights=val_weights)),
                "early_exit_fraction": float(np.mean(exit_now)),
                "delta_r_vs_hold": float(np.average(-target[validation] * exit_now, weights=val_weights)),
                "lost_continuation_r": float(np.average(np.maximum(target[validation], 0) * exit_now, weights=val_weights))}
    validation_metrics = {"dl": metrics(validation_prediction), "linear": metrics(linear),
                          "always_hold": metrics(np.full(len(validation), max(0.0, float(target[train].mean()))))}
    final_norm = normalization(final)
    model = reset(); optimizer = torch.optim.Adam(model.parameters(), lr=CONTRACT["learning_rate"])
    rng = np.random.default_rng(seed)
    for e in range(1, best_epoch + 1):
        epoch(model, optimizer, final, final_norm, rng)
        print(f"[HV1 seed={seed}] {name} refit {e}/{best_epoch}", flush=True)
    temporary = weights_path.with_suffix(".tmp.pt")
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "state_mean": torch.tensor(final_norm[0]), "state_scale": torch.tensor(final_norm[1]),
                "target_scale": float(final_norm[2]), "seed": seed, "fit_identity": fit_identity}, temporary)
    atomic_replace_with_retry(temporary, weights_path)
    result = {"fit_identity": fit_identity, "start": str(start.date()), "end": str(end.date()),
              "fit_max_maturity": str(rows.iloc[final].maturity_date.max()),
              "train_states": len(train), "validation_states": len(validation), "refit_states": len(final),
              "train_episodes": int(rows.iloc[train].episode.nunique()),
              "validation_episodes": int(rows.iloc[validation].episode.nunique()),
              "selected_epochs": best_epoch, "curve": curve, "validation": validation_metrics,
              "weights": weights_path.name, "weights_sha256": compute_observed_file_sha256(weights_path),
              "metrics": metrics_path.name}
    _write_json(metrics_path, result)
    return result


def ensure_holding_model(root, *, seed, config):
    from core.file_integrity import exclusive_file_lock
    directory = model_directory(root, seed=seed, config=config)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(directory / "writer.lock", conflict_message="此 Seed 的持倉模型正在訓練"):
        ready = inspect_holding_model(root, seed=seed, config=config)
        if ready["ready"]:
            return ready
        teacher = load_formal_mintrade40_training_source(project_root=root, seed=int(seed), allow_optimizer_build=True)
        identity = publication_identity(root, seed=seed, config=config, teacher=teacher)
        inputs, duplicates = discover_dataset_csv_inputs(Path(root), config["dataset"])
        if duplicates:
            raise ValueError("Ambiguous canonical holding price universe")
        rows, sample_evidence = _build_rows(root, seed=seed, config=config, teacher=teacher, inputs=inputs, identity=identity)
        required = set(rows.ticker) | {"0050"}
        frames = {ticker: load_dataset_frame(path, ticker, min_rows=1) for ticker, path in inputs if ticker in required}
        if "0050" not in frames:
            raise ValueError("Missing canonical market benchmark 0050")
        features = HoldingFeatures(frames, frames["0050"])
        eligible = [features.coordinates(str(ticker), pd.Timestamp(day)) is not None
                    for ticker, day in zip(rows.ticker, rows.information_date)]
        rows = rows.loc[eligible].reset_index(drop=True)
        folds = []
        for start, end in fold_ranges(config):
            _, teacher_prefix = slice_formal_mintrade40_pit_source_for_fold(teacher, fit_cutoff=start - pd.Timedelta(days=1))
            mature = rows[pd.to_datetime(rows.maturity_date) < start]
            # Bind the actual causal data prefix; later bars cannot change a prior FIT identity.
            digest = hashlib.sha256()
            for ticker in sorted(set(mature.ticker) | {"0050"}):
                frame = frames[ticker].loc[:start - pd.Timedelta(days=1)]
                digest.update(ticker.encode()); digest.update(frame.index.to_numpy().tobytes())
                digest.update(frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float).tobytes())
            fit_identity = canonical_json_sha256({"contract": CONTRACT, "seed": seed,
                "implementation": identity["implementation"], "teacher_prefix": teacher_prefix,
                "price_prefix": digest.hexdigest(), "rows": mature.to_json(orient="records"),
                "start": str(start.date()), "end": str(end.date())})
            fit = _fit_fold(rows, features, start=start, end=end, seed=seed, directory=directory, fit_identity=fit_identity)
            folds.append({**fit, "metrics_sha256": compute_observed_file_sha256(directory / fit["metrics"])})
        _write_json(directory / "manifest.json", {"identity": identity, "status": "COMPLETED", "samples": sample_evidence,
                    "folds": folds, "evaluation_label": "迭代研究 OOS", "oracle_status": "CONTRACT_UNRESOLVED"})
        result = inspect_holding_model(root, seed=seed, config=config)
        if not result["ready"]:
            raise RuntimeError(result["reason"])
        return result


@contextmanager
def holding_model_runtime(root, *, seed, config, evidence_path=None):
    import torch
    ready = inspect_holding_model(root, seed=seed, config=config)
    if not ready["ready"]:
        raise ValueError("Holding model not READY: " + ready["reason"])
    directory = Path(ready["path"]).parent
    inputs, _ = discover_dataset_csv_inputs(Path(root), config["dataset"])
    paths = dict(inputs)
    market = load_dataset_frame(paths["0050"], "0050", min_rows=1)
    frames, feature_sources, loaded, records = {}, {}, {}, []
    torch.set_num_threads(2)
    def predict(*, ticker, position, information_date, execution_date, completed_atr, completed_high, params):
        if ticker not in paths:
            return None
        if ticker not in frames:
            frames[ticker] = load_dataset_frame(paths[ticker], ticker, min_rows=1)
            feature_sources[ticker] = HoldingFeatures({ticker: frames[ticker]}, market)
        frame = frames[ticker]
        features = feature_sources[ticker]
        coordinates = features.coordinates(ticker, pd.Timestamp(information_date))
        if coordinates is None:
            return None
        candidates = [f for f in ready["manifest"]["folds"] if pd.Timestamp(f["start"]) <= execution_date <= pd.Timestamp(f["end"])]
        if len(candidates) != 1:
            raise ValueError("No unique PIT holding fold for executable session")
        fold = candidates[0]
        # A new year's fit is available before that execution session, using only
        # fully matured prior-year history; no current bar enters its inputs.
        if fold["weights"] not in loaded:
            payload = torch.load(directory / fold["weights"], map_location="cpu", weights_only=True)
            if payload["seed"] != seed or payload["fit_identity"] != fold["fit_identity"]:
                raise ValueError("Holding weight seed/FIT lineage mismatch")
            model = build_network(); model.load_state_dict(payload["state_dict"]); model.eval()
            loaded[fold["weights"]] = model, payload
        model, payload = loaded[fold["weights"]]
        state = position_context(position, frame=frame, information_pos=coordinates[0],
                                 completed_atr=completed_atr, completed_high=completed_high, params=params)
        sequence, context = features.batch(pd.DataFrame([{"ticker": ticker, "information_date": information_date, "state": state}]))
        with torch.inference_mode():
            value = float(model(torch.from_numpy(sequence),
                (torch.from_numpy(context) - payload["state_mean"]) / payload["state_scale"]).item()) * payload["target_scale"]
        records.append({"ticker": ticker, "information_date": str(information_date),
                        "execution_date": str(execution_date), "entry_date": str(position["entry_trade_date"]),
                        "delta_r": value, "exit_requested": value < 0,
                        "fit_identity": fold["fit_identity"]})
        return value
    with bind_position_exit_provider(predict):
        yield ready
    if evidence_path is not None:
        _write_json(evidence_path, {"model_manifest_sha256": ready["sha256"], "seed": seed,
                                   "contract": CONTRACT, "decisions": records})

"""PIT-safe encoder initialization primitives for breakout-quality rankers."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from core.breakout_quality_registry import (
    get_breakout_quality_encoder_pretraining_profile,
)
from filters.breakout_quality.models.active import build_active_model
from filters.breakout_quality.torch_runtime import (
    autocast_context,
    build_grad_scaler,
    seed_torch,
)


def build_ticker_balanced_epoch_ids(
    group_table: pd.DataFrame,
    fit_ids: np.ndarray,
    *,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Select exactly one fitting-scope window per ticker, then deterministically shuffle."""

    ids = np.asarray(fit_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < 2:
        raise ValueError("SCC encoder pretraining fitting ids必須是一維且至少2筆")
    scoped = group_table.iloc[ids]
    if "ticker" not in scoped.columns:
        raise ValueError("SCC encoder pretraining group table缺少ticker")
    ticker_values = scoped["ticker"].astype(str).to_numpy()
    tickers = tuple(sorted(set(ticker_values.tolist())))
    if len(tickers) < 2:
        raise ValueError("SCC encoder pretraining至少需要2個ticker class")

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(epoch), 0x534343])
    )
    chosen: list[int] = []
    for ticker in tickers:
        candidates = ids[ticker_values == ticker]
        chosen.append(int(candidates[int(rng.integers(0, len(candidates)))]))
    selected = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(selected)
    return selected, tickers


def _ticker_targets(
    group_table: pd.DataFrame,
    group_ids: np.ndarray,
    ticker_to_class: dict[str, int],
) -> np.ndarray:
    tickers = group_table.iloc[np.asarray(group_ids, dtype=np.int64)]["ticker"].astype(str)
    return np.asarray([ticker_to_class[value] for value in tickers], dtype=np.int64)


def _pretrain_stock_code_encoder_state(
    torch,
    *,
    feature_bank: np.ndarray,
    group_table: pd.DataFrame,
    fit_ids: np.ndarray,
    feature_count: int,
    context_count: int,
    model_architecture: str,
    pretraining_profile_name: str,
    seed: int,
    plan,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit SCC only on downstream fitting rows and return the shared encoder state."""

    profile = get_breakout_quality_encoder_pretraining_profile(
        pretraining_profile_name
    )
    ids = np.asarray(fit_ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("encoder pretraining fitting ids不得為空")
    if int(feature_bank.shape[2]) != int(feature_count):
        raise ValueError("encoder pretraining feature_count與feature bank不一致")

    scoped_tickers = group_table.iloc[ids]["ticker"].astype(str).to_numpy()
    tickers = tuple(sorted(set(scoped_tickers.tolist())))
    if len(tickers) < 2:
        raise ValueError("SCC encoder pretraining至少需要2個ticker class")
    ticker_to_class = {ticker: index for index, ticker in enumerate(tickers)}

    seed_torch(torch, seed=int(seed), plan=plan)
    model = build_active_model(
        feature_count=int(feature_count),
        context_count=int(context_count),
        architecture=str(model_architecture),
    ).to(plan.device)
    required = (
        "encode",
        "encoder_parameters",
        "export_encoder_state_dict",
        "encoder_embedding_width",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            f"architecture不支援formal encoder pretraining contract: missing={missing}"
        )

    classifier = torch.nn.Linear(
        int(model.encoder_embedding_width), int(len(tickers))
    ).to(plan.device)
    trainable_parameters = list(model.encoder_parameters()) + list(classifier.parameters())
    if not trainable_parameters:
        raise ValueError("SCC encoder pretraining沒有trainable parameters")
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=float(profile.learning_rate),
        weight_decay=float(profile.weight_decay),
    )
    grad_scaler = build_grad_scaler(torch, plan)

    best_accuracy = 0.0
    final_accuracy = 0.0
    final_loss = float("nan")
    total_samples = 0
    model.train()
    classifier.train()
    for epoch in range(1, int(profile.epochs) + 1):
        epoch_ids, epoch_tickers = build_ticker_balanced_epoch_ids(
            group_table,
            ids,
            seed=int(seed),
            epoch=int(epoch),
        )
        if epoch_tickers != tickers:
            raise AssertionError("SCC ticker class universe在epoch內發生漂移")
        epoch_targets = _ticker_targets(group_table, epoch_ids, ticker_to_class)
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_count = 0
        for start in range(0, len(epoch_ids), int(profile.batch_size)):
            stop = min(start + int(profile.batch_size), len(epoch_ids))
            batch_ids = epoch_ids[start:stop]
            batch_targets = epoch_targets[start:stop]
            features = np.ascontiguousarray(
                np.asarray(feature_bank[batch_ids], dtype=np.float32)
            )
            x = torch.from_numpy(features).to(plan.device, non_blocking=True)
            y = torch.from_numpy(batch_targets).to(plan.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(torch, plan):
                encoded = model.encode(x)
                logits = classifier(encoded)
                loss = torch.nn.functional.cross_entropy(logits.float(), y)
            if grad_scaler is None:
                loss.backward()
                if float(profile.gradient_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, float(profile.gradient_clip_norm)
                    )
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                if float(profile.gradient_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, float(profile.gradient_clip_norm)
                    )
                grad_scaler.step(optimizer)
                grad_scaler.update()
            count = int(len(batch_ids))
            epoch_loss_sum += float(loss.detach().float().cpu()) * count
            epoch_correct += int((logits.detach().argmax(dim=1) == y).sum().cpu())
            epoch_count += count

        if epoch_count != len(tickers):
            raise AssertionError("SCC每epoch必須恰好一個sample per ticker")
        final_loss = float(epoch_loss_sum / max(1, epoch_count))
        final_accuracy = float(epoch_correct / max(1, epoch_count))
        best_accuracy = max(best_accuracy, final_accuracy)
        total_samples += int(epoch_count)

    encoder_state = model.export_encoder_state_dict()
    if not encoder_state:
        raise ValueError("SCC encoder pretraining輸出encoder state為空")
    scoped_dates = pd.to_datetime(group_table.iloc[ids]["date"], errors="raise")
    ticker_fingerprint = hashlib.sha256(
        "\n".join(tickers).encode("utf-8")
    ).hexdigest()
    summary = {
        "profile": profile.as_manifest_payload(),
        "stage": str(stage),
        "fitting_group_count": int(len(ids)),
        "fitting_date_start": str(scoped_dates.min().date()),
        "fitting_date_end": str(scoped_dates.max().date()),
        "ticker_count": int(len(tickers)),
        "ticker_class_fingerprint_sha256": ticker_fingerprint,
        "samples_per_epoch": int(len(tickers)),
        "total_pretraining_samples": int(total_samples),
        "best_train_accuracy": float(best_accuracy),
        "final_train_accuracy": float(final_accuracy),
        "final_train_loss": float(final_loss),
        "pretraining_label": "ticker",
        "downstream_target_labels_used": False,
        "downstream_validation_used": False,
        "selection_metric": "none_fixed_epochs",
        "temporary_classification_head_persisted": False,
        "downstream_encoder_frozen": False,
    }

    del optimizer, classifier, model
    if plan.device_type == "cuda":
        torch.cuda.empty_cache()
    return encoder_state, summary


def _pretrain_future_path_encoder_state(
    torch,
    *,
    feature_bank,
    group_table: pd.DataFrame,
    fit_ids: np.ndarray,
    feature_count: int,
    context_count: int,
    model_architecture: str,
    pretraining_profile_name: str,
    seed: int,
    plan,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a temporary future-path regressor using only PIT-matured fitting rows."""

    profile = get_breakout_quality_encoder_pretraining_profile(pretraining_profile_name)
    ids = np.asarray(fit_ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("future-path encoder pretraining fitting ids不得為空")
    if int(feature_bank.shape[2]) != int(feature_count):
        raise ValueError("future-path encoder pretraining feature_count與feature bank不一致")
    required_bank = ("future_path_maturity_mask", "future_path_statistics")
    missing_bank = [name for name in required_bank if not hasattr(feature_bank, name)]
    if missing_bank:
        raise ValueError(
            "future-path encoder pretraining需要canonical lazy OHLCV feature provider: "
            f"missing={missing_bank}"
        )

    scoped_dates = pd.to_datetime(group_table.iloc[ids]["date"], errors="raise")
    fitting_cutoff = pd.Timestamp(scoped_dates.max()).normalize()
    max_horizon_bars = int(profile.future_path_max_horizon_bars or 0)
    if max_horizon_bars <= 0:
        raise ValueError("future-path pretraining缺少合法max horizon contract")
    matured_mask = feature_bank.future_path_maturity_mask(
        ids,
        horizon_bars=max_horizon_bars,
        cutoff_date=fitting_cutoff,
    )
    matured_ids = ids[np.asarray(matured_mask, dtype=bool)]
    if len(matured_ids) < 2:
        raise ValueError("future-path encoder pretraining沒有足夠PIT-matured fitting rows")
    matured_tickers = tuple(
        sorted(set(group_table.iloc[matured_ids]["ticker"].astype(str).tolist()))
    )
    if len(matured_tickers) < 2:
        raise ValueError("future-path encoder pretraining至少需要2個matured ticker")

    epoch_schedules: list[np.ndarray] = []
    for epoch in range(1, int(profile.epochs) + 1):
        epoch_ids, epoch_tickers = build_ticker_balanced_epoch_ids(
            group_table,
            matured_ids,
            seed=int(seed),
            epoch=int(epoch),
        )
        if epoch_tickers != matured_tickers:
            raise AssertionError("future-path ticker universe在epoch內發生漂移")
        epoch_schedules.append(epoch_ids)

    scheduled_ids = np.concatenate(epoch_schedules).astype(np.int64, copy=False)
    unique_ids = np.unique(scheduled_ids).astype(np.int64, copy=False)
    raw_targets, target_names = feature_bank.future_path_statistics(
        unique_ids,
        cutoff_date=fitting_cutoff,
        target_names=tuple(profile.target_names),
        horizon_bars=max_horizon_bars,
    )
    if tuple(target_names) != tuple(profile.target_names):
        raise ValueError("future-path provider target schema與pretraining profile不一致")
    raw_targets = np.asarray(raw_targets, dtype=np.float64)
    if raw_targets.shape != (len(unique_ids), len(profile.target_names)):
        raise ValueError("future-path target matrix shape不符合scientific contract")
    if not bool(np.isfinite(raw_targets).all()):
        raise ValueError("future-path target matrix含non-finite value")

    scheduled_positions = np.searchsorted(unique_ids, scheduled_ids)
    if bool(np.any(scheduled_positions >= len(unique_ids))) or not np.array_equal(
        unique_ids[scheduled_positions], scheduled_ids
    ):
        raise AssertionError("future-path fitting-schedule target lookup失敗")
    scheduled_targets = raw_targets[scheduled_positions]
    target_mean = np.mean(scheduled_targets, axis=0, dtype=np.float64)
    target_std = np.std(scheduled_targets, axis=0, ddof=0, dtype=np.float64)
    if not bool(np.isfinite(target_mean).all() and np.isfinite(target_std).all()):
        raise ValueError("future-path fitting-scope standardization產生non-finite值")
    if bool(np.any(target_std <= np.finfo(np.float32).eps)):
        bad = [
            str(profile.target_names[index])
            for index in np.flatnonzero(target_std <= np.finfo(np.float32).eps)
        ]
        raise ValueError(f"future-path target在fitting scope缺乏變異: {bad}")
    standardized_targets = ((raw_targets - target_mean) / target_std).astype(np.float32)

    seed_torch(torch, seed=int(seed), plan=plan)
    model = build_active_model(
        feature_count=int(feature_count),
        context_count=int(context_count),
        architecture=str(model_architecture),
    ).to(plan.device)
    required = (
        "encode",
        "encoder_parameters",
        "export_encoder_state_dict",
        "encoder_embedding_width",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            f"architecture不支援formal encoder pretraining contract: missing={missing}"
        )

    regressor = torch.nn.Linear(
        int(model.encoder_embedding_width), int(len(profile.target_names))
    ).to(plan.device)
    trainable_parameters = list(model.encoder_parameters()) + list(regressor.parameters())
    if not trainable_parameters:
        raise ValueError("future-path encoder pretraining沒有trainable parameters")
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=float(profile.learning_rate),
        weight_decay=float(profile.weight_decay),
    )
    grad_scaler = build_grad_scaler(torch, plan)

    final_loss = float("nan")
    total_samples = 0
    model.train()
    regressor.train()
    for epoch_ids in epoch_schedules:
        positions = np.searchsorted(unique_ids, epoch_ids)
        if bool(np.any(positions >= len(unique_ids))) or not np.array_equal(
            unique_ids[positions], epoch_ids
        ):
            raise AssertionError("future-path pretraining target lookup失敗")
        epoch_targets = standardized_targets[positions]
        epoch_loss_sum = 0.0
        epoch_count = 0
        for start in range(0, len(epoch_ids), int(profile.batch_size)):
            stop = min(start + int(profile.batch_size), len(epoch_ids))
            batch_ids = epoch_ids[start:stop]
            features = np.ascontiguousarray(
                np.asarray(feature_bank[batch_ids], dtype=np.float32)
            )
            targets = np.ascontiguousarray(epoch_targets[start:stop], dtype=np.float32)
            x = torch.from_numpy(features).to(plan.device, non_blocking=True)
            y = torch.from_numpy(targets).to(plan.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(torch, plan):
                encoded = model.encode(x)
                predictions = regressor(encoded)
                loss = torch.nn.functional.smooth_l1_loss(
                    predictions.float(),
                    y.float(),
                    beta=float(profile.smooth_l1_beta),
                    reduction="mean",
                )
            if grad_scaler is None:
                loss.backward()
                if float(profile.gradient_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, float(profile.gradient_clip_norm)
                    )
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                if float(profile.gradient_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, float(profile.gradient_clip_norm)
                    )
                grad_scaler.step(optimizer)
                grad_scaler.update()
            count = int(len(batch_ids))
            epoch_loss_sum += float(loss.detach().float().cpu()) * count
            epoch_count += count
        if epoch_count != len(matured_tickers):
            raise AssertionError("future-path每epoch必須恰好一個sample per matured ticker")
        final_loss = float(epoch_loss_sum / max(1, epoch_count))
        total_samples += int(epoch_count)

    encoder_state = model.export_encoder_state_dict()
    if not encoder_state:
        raise ValueError("future-path encoder pretraining輸出encoder state為空")
    summary = {
        "profile": profile.as_manifest_payload(),
        "stage": str(stage),
        "fitting_group_count": int(len(ids)),
        "fitting_date_start": str(scoped_dates.min().date()),
        "fitting_date_end": str(scoped_dates.max().date()),
        "future_path_maturity_cutoff": str(fitting_cutoff.date()),
        "matured_fitting_group_count": int(len(matured_ids)),
        "matured_ticker_count": int(len(matured_tickers)),
        "unique_pretraining_window_count": int(len(unique_ids)),
        "samples_per_epoch": int(len(matured_tickers)),
        "total_pretraining_samples": int(total_samples),
        "final_train_loss": float(final_loss),
        "pretraining_label": "future_path_statistics",
        "future_path_target_names": list(profile.target_names),
        "future_path_target_mean": [float(value) for value in target_mean],
        "future_path_target_std": [float(value) for value in target_std],
        "future_path_max_horizon_bars": int(max_horizon_bars),
        "downstream_target_labels_used": False,
        "downstream_validation_used": False,
        "future_price_path_used_for_pretraining": True,
        "selection_metric": "none_fixed_epochs",
        "temporary_regression_head_persisted": False,
        "downstream_encoder_frozen": False,
    }

    del optimizer, regressor, model
    if plan.device_type == "cuda":
        torch.cuda.empty_cache()
    return encoder_state, summary


def pretrain_encoder_state(
    torch,
    *,
    feature_bank,
    group_table: pd.DataFrame,
    fit_ids: np.ndarray,
    feature_count: int,
    context_count: int,
    model_architecture: str,
    pretraining_profile_name: str,
    seed: int,
    plan,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Dispatch a declarative encoder-initialization recipe without experiment identity branches."""

    profile = get_breakout_quality_encoder_pretraining_profile(pretraining_profile_name)
    kwargs = dict(
        torch=torch,
        feature_bank=feature_bank,
        group_table=group_table,
        fit_ids=fit_ids,
        feature_count=feature_count,
        context_count=context_count,
        model_architecture=model_architecture,
        pretraining_profile_name=pretraining_profile_name,
        seed=seed,
        plan=plan,
        stage=stage,
    )
    if profile.task == "stock_code_classification":
        return _pretrain_stock_code_encoder_state(**kwargs)
    if profile.task == "future_path_statistics_regression":
        return _pretrain_future_path_encoder_state(**kwargs)
    raise ValueError(f"不支援的 encoder pretraining task: {profile.task!r}")


__all__ = [
    "build_ticker_balanced_epoch_ids",
    "pretrain_encoder_state",
]

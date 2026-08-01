"""Selection-only TS2Vec self-supervised encoder pretraining."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

from config.breakout_quality import (
    SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
    SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES,
    build_breakout_quality_pretraining_profile_payload,
    get_breakout_quality_pretraining_profile,
    resolve_breakout_quality_random_seed,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_PRETRAINING_PROFILE,
    BREAKOUT_QUALITY_PRETRAINING_BATCH_SIZE,
    BREAKOUT_QUALITY_PRETRAINING_CONTRASTIVE_ALPHA,
    BREAKOUT_QUALITY_PRETRAINING_EPOCHS,
    BREAKOUT_QUALITY_PRETRAINING_FAMILY,
    BREAKOUT_QUALITY_PRETRAINING_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_PRETRAINING_LEARNING_RATE,
    BREAKOUT_QUALITY_PRETRAINING_MASK_PROBABILITY,
    BREAKOUT_QUALITY_PRETRAINING_MIN_CROP_BARS,
    BREAKOUT_QUALITY_PRETRAINING_STRIDE,
    BREAKOUT_QUALITY_PRETRAINING_TEMPORAL_UNIT,
    BREAKOUT_QUALITY_PRETRAINING_WEIGHT_DECAY,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY, FEATURE_COLUMNS
from filters.breakout_quality.models.spec import TS2VEC_FROZEN_LINEAR_V1, get_model_spec
from filters.breakout_quality.models.ts2vec import (
    build_ts2vec_encoder,
    hierarchical_contrastive_loss,
)
from filters.breakout_quality.pretraining_store import (
    build_file_record,
    close_pretraining_windows,
    load_validated_pretraining_dataset,
    resolve_pretrained_encoder_paths,
)
from filters.breakout_quality.splits import resolve_breakout_quality_outer_policy
from filters.breakout_quality.torch_runtime import (
    SUPPORTED_MIXED_PRECISION_DTYPES,
    SUPPORTED_TORCH_DEVICES,
    autocast_context,
    build_grad_scaler,
    resolve_torch_execution_plan,
    seed_torch,
)
from tools.filters.breakout_quality.common import (
    PROJECT_ROOT,
    load_validated_dataset_bundle,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Selection-only TS2Vec encoder pretraining")
    parser.add_argument("--dataset", default="full")
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument(
        "--experiment-profile",
        choices=SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        default=BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        help="下游 supervised experiment profile；決定 pretrained encoder 的正式工件路徑",
    )
    parser.add_argument(
        "--pretraining-profile",
        choices=SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES,
        default=BREAKOUT_QUALITY_PRETRAINING_PROFILE,
    )
    parser.add_argument("--epochs", type=int, default=BREAKOUT_QUALITY_PRETRAINING_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BREAKOUT_QUALITY_PRETRAINING_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=BREAKOUT_QUALITY_PRETRAINING_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=BREAKOUT_QUALITY_PRETRAINING_WEIGHT_DECAY)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=BREAKOUT_QUALITY_PRETRAINING_GRADIENT_CLIP_NORM,
    )
    parser.add_argument("--stride", type=int, default=BREAKOUT_QUALITY_PRETRAINING_STRIDE)
    parser.add_argument("--min-crop-bars", type=int, default=BREAKOUT_QUALITY_PRETRAINING_MIN_CROP_BARS)
    parser.add_argument("--mask-probability", type=float, default=BREAKOUT_QUALITY_PRETRAINING_MASK_PROBABILITY)
    parser.add_argument("--contrastive-alpha", type=float, default=BREAKOUT_QUALITY_PRETRAINING_CONTRASTIVE_ALPHA)
    parser.add_argument("--temporal-unit", type=int, default=BREAKOUT_QUALITY_PRETRAINING_TEMPORAL_UNIT)
    parser.add_argument(
        "--seed",
        type=int,
        default=resolve_breakout_quality_random_seed(),
        help="亂數種子；省略時使用config的BREAKOUT_QUALITY_RANDOM_SEED",
    )
    parser.add_argument("--device", choices=SUPPORTED_TORCH_DEVICES, default=BREAKOUT_QUALITY_TORCH_DEVICE)
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    )
    parser.add_argument(
        "--mixed-precision-dtype",
        choices=SUPPORTED_MIXED_PRECISION_DTYPES,
        default=BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_ALLOW_TF32,
    )
    args = parser.parse_args(argv)
    if int(args.seed) < 0:
        parser.error("--seed 必須 >= 0")
    return args


def validate_args(args) -> None:
    if BREAKOUT_QUALITY_MODEL_ARCHITECTURE != TS2VEC_FROZEN_LINEAR_V1:
        raise ValueError("TS2Vec pretrain 只允許在 active ts2vec_frozen_linear_v1 執行")
    if int(args.epochs) < 1 or int(args.batch_size) < 2:
        raise ValueError("pretraining epochs 必須 >=1 且 batch-size 必須 >=2")
    if (
        float(args.lr) <= 0
        or float(args.weight_decay) < 0
        or float(args.gradient_clip_norm) < 0
    ):
        raise ValueError(
            "pretraining lr 必須 >0，weight-decay與gradient-clip-norm必須 >=0"
        )
    if int(args.stride) < 1 or int(args.min_crop_bars) < 2:
        raise ValueError("pretraining stride 必須 >=1 且 min-crop-bars 必須 >=2")
    if not 0.0 <= float(args.mask_probability) < 1.0:
        raise ValueError("mask-probability 必須介於 0（含）與 1（不含）")
    if not 0.0 <= float(args.contrastive_alpha) <= 1.0:
        raise ValueError("contrastive-alpha 必須介於 0 與 1")
    if int(args.temporal_unit) < 0:
        raise ValueError("temporal-unit 必須 >=0")


def _build_pretraining_optimizer(torch, encoder, *, optimizer_name: str, lr: float, weight_decay: float):
    normalized = str(optimizer_name).strip().lower()
    kwargs = {"lr": float(lr), "weight_decay": float(weight_decay)}
    if normalized == "adam":
        return torch.optim.Adam(encoder.parameters(), **kwargs)
    if normalized == "adamw":
        return torch.optim.AdamW(encoder.parameters(), **kwargs)
    raise ValueError(f"不支援的 TS2Vec pretraining optimizer: {optimizer_name!r}")


def _source_data_end(dataset_summary: dict, events: pd.DataFrame) -> str:
    date_range = dataset_summary.get("source_data_date_range")
    if isinstance(date_range, dict) and str(date_range.get("end") or "").strip():
        return str(date_range["end"])
    return pd.to_datetime(events["label_eval_end_date"], errors="raise").max().strftime("%Y-%m-%d")


def _resolve_pretrained_encoder_output(args):
    experiment_profile = str(args.experiment_profile)
    paths = resolve_pretrained_encoder_paths(
        PROJECT_ROOT,
        args.filter_id,
        model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        experiment_profile=experiment_profile,
    )
    return experiment_profile, paths


def _overlapping_crops(rng: np.random.Generator, sequence_length: int, min_crop: int) -> tuple[int, int, int]:
    crop_length = int(rng.integers(min_crop, sequence_length + 1))
    max_start = sequence_length - crop_length
    if max_start == 0:
        return 0, 0, crop_length
    for _attempt in range(16):
        start1 = int(rng.integers(0, max_start + 1))
        start2 = int(rng.integers(0, max_start + 1))
        overlap = crop_length - abs(start1 - start2)
        if overlap >= min_crop:
            return start1, start2, crop_length
    start = int(rng.integers(0, max_start + 1))
    return start, start, crop_length


def _aligned_views(torch, encoder, xb, *, rng, min_crop, mask_probability):
    sequence_length = int(xb.shape[2])
    start1, start2, crop_length = _overlapping_crops(rng, sequence_length, min_crop)
    crop1 = xb[:, :, start1:start1 + crop_length]
    crop2 = xb[:, :, start2:start2 + crop_length]
    mask1 = torch.rand((xb.shape[0], crop_length), device=xb.device) >= float(mask_probability)
    mask2 = torch.rand((xb.shape[0], crop_length), device=xb.device) >= float(mask_probability)
    z1 = encoder(crop1, temporal_mask=mask1).transpose(1, 2)
    z2 = encoder(crop2, temporal_mask=mask2).transpose(1, 2)
    overlap_start = max(start1, start2)
    overlap_end = min(start1 + crop_length, start2 + crop_length)
    z1_start = overlap_start - start1
    z2_start = overlap_start - start2
    overlap_length = overlap_end - overlap_start
    return (
        z1[:, z1_start:z1_start + overlap_length],
        z2[:, z2_start:z2_start + overlap_length],
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    validate_args(args)
    started = time.perf_counter()
    pretraining_profile = get_breakout_quality_pretraining_profile(
        str(args.pretraining_profile)
    )
    pretraining_profile_payload = build_breakout_quality_pretraining_profile_payload(
        str(args.pretraining_profile),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        gradient_clip_norm=float(args.gradient_clip_norm),
        min_crop_bars=int(args.min_crop_bars),
        mask_probability=float(args.mask_probability),
        contrastive_alpha=float(args.contrastive_alpha),
        temporal_unit=int(args.temporal_unit),
    )
    if pretraining_profile.family != BREAKOUT_QUALITY_PRETRAINING_FAMILY:
        raise ValueError(
            "pretraining profile family 與 active policy 不一致: "
            f"profile={pretraining_profile.family}, policy={BREAKOUT_QUALITY_PRETRAINING_FAMILY}"
        )
    dataset_summary, _X, _C, _y, events = load_validated_dataset_bundle(
        args.filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=True,
    )
    outer_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=_source_data_end(dataset_summary, events),
    )
    source_selection = dataset_summary.get("source_selection")
    if not isinstance(source_selection, dict):
        raise ValueError("supervised dataset summary 缺少 source_selection")
    expected_max_tickers = max(
        0, int(source_selection.get("requested_max_tickers", -1))
    )
    pretrain_summary, windows, _index = load_validated_pretraining_dataset(
        PROJECT_ROOT,
        args.filter_id,
        dataset_profile=str(args.dataset),
        family=BREAKOUT_QUALITY_PRETRAINING_FAMILY,
        stride=int(args.stride),
        expected_selection_start=str(outer_policy["selection_start_date"]),
        expected_selection_end=str(outer_policy["selection_end_date"]),
        expected_window_bars=int(DEFAULT_LABEL_POLICY.feature_window_bars),
        expected_max_tickers=expected_max_tickers,
        require_current_source=True,
    )
    try:
        torch = __import__("torch")
        nn = __import__("torch.nn", fromlist=["nn"])
        execution_plan = resolve_torch_execution_plan(
            torch,
            requested_device=str(args.device),
            mixed_precision=bool(args.mixed_precision),
            mixed_precision_dtype=str(args.mixed_precision_dtype),
            deterministic_algorithms=bool(args.deterministic_algorithms),
            allow_tf32=bool(args.allow_tf32),
        )
        if execution_plan.device_type == "cpu":
            torch.set_num_threads(1)
        seed_torch(torch, seed=int(args.seed), plan=execution_plan)
        spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
        encoder = build_ts2vec_encoder(
            nn,
            torch,
            feature_count=len(FEATURE_COLUMNS),
            spec=spec,
        ).to(execution_plan.device)
        optimizer = _build_pretraining_optimizer(
            torch,
            encoder,
            optimizer_name=pretraining_profile.optimizer_name,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        scaler = build_grad_scaler(torch, execution_plan)
        rng = np.random.default_rng(int(args.seed))
        history = []
        print(
            "TS2Vec pretraining | "
            f"windows={len(windows):,}, selection={outer_policy['selection_start_date']}~{outer_policy['selection_end_date']}, "
            f"profile={pretraining_profile.name}, optimizer={pretraining_profile.optimizer_name}, "
            f"device={execution_plan.device_type}, dtype={execution_plan.autocast_dtype_name}"
        )
        for epoch in range(1, int(args.epochs) + 1):
            epoch_started = time.perf_counter()
            order = rng.permutation(len(windows))
            losses = []
            encoder.train()
            for start in range(0, len(order), int(args.batch_size)):
                indices = order[start:start + int(args.batch_size)]
                if len(indices) < 2:
                    continue
                xb_np = np.asarray(windows[indices], dtype=np.float32)
                xb = torch.from_numpy(xb_np).to(execution_plan.device).transpose(1, 2)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(torch, execution_plan):
                    z1, z2 = _aligned_views(
                        torch,
                        encoder,
                        xb,
                        rng=rng,
                        min_crop=int(args.min_crop_bars),
                        mask_probability=float(args.mask_probability),
                    )
                    loss = hierarchical_contrastive_loss(
                        torch,
                        z1,
                        z2,
                        alpha=float(args.contrastive_alpha),
                        temporal_unit=int(args.temporal_unit),
                    )
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("TS2Vec pretraining loss 非有限值")
                if scaler is None:
                    loss.backward()
                    if float(args.gradient_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            encoder.parameters(), float(args.gradient_clip_norm)
                        )
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if float(args.gradient_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            encoder.parameters(), float(args.gradient_clip_norm)
                        )
                    scaler.step(optimizer)
                    scaler.update()
                losses.append(float(loss.detach().cpu().item()))
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            if not math.isfinite(mean_loss):
                raise FloatingPointError("TS2Vec epoch loss 非有限值")
            elapsed = time.perf_counter() - epoch_started
            history.append({"epoch": epoch, "loss": round(mean_loss, 6), "elapsed_sec": round(elapsed, 3)})
            print(f"  Epoch {epoch:>2}/{int(args.epochs)} | Loss {mean_loss:.6f} | 耗時 {elapsed:.1f}s")

        experiment_profile, paths = _resolve_pretrained_encoder_output(args)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "encoder_state_dict": {
                    key: value.detach().cpu() for key, value in encoder.state_dict().items()
                },
                "feature_count": len(FEATURE_COLUMNS),
                "model_spec": spec.as_manifest_payload(),
                "pretraining_profile": pretraining_profile_payload,
                "pretraining_dataset_fingerprint": pretrain_summary["configuration_fingerprint"],
            },
            paths.encoder,
        )
        manifest = {
            "schema_version": 1,
            "filter_id": str(args.filter_id),
            "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
            "experiment_profile": experiment_profile,
            "pretraining_profile": pretraining_profile_payload,
            "model_spec": spec.as_manifest_payload(),
            "encoder": build_file_record(paths.encoder),
            "pretraining_dataset_fingerprint": pretrain_summary["configuration_fingerprint"],
            "pretraining_dataset": {
                "family": BREAKOUT_QUALITY_PRETRAINING_FAMILY,
                "stride": int(args.stride),
                "window_count": int(len(windows)),
                "selection_start_date": str(outer_policy["selection_start_date"]),
                "selection_end_date": str(outer_policy["selection_end_date"]),
            },
            "training": {
                "optimizer_name": pretraining_profile.optimizer_name,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "gradient_clip_norm": float(args.gradient_clip_norm),
                "min_crop_bars": int(args.min_crop_bars),
                "mask_probability": float(args.mask_probability),
                "contrastive_alpha": float(args.contrastive_alpha),
                "temporal_unit": int(args.temporal_unit),
                "seed": int(args.seed),
                "history": history,
            },
            "torch_execution": execution_plan.as_manifest_payload(),
            "oos_windows_used": False,
            "pass_reject_labels_used": False,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        write_json(paths.manifest, manifest)
        print(f"已輸出 pretrained encoder: {paths.encoder}")
        print(f"已輸出 pretraining manifest: {paths.manifest}")
        return 0
    finally:
        close_pretraining_windows(windows)


if __name__ == "__main__":
    raise SystemExit(main())

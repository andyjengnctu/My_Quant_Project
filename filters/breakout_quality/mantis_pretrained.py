"""Pinned external MantisV2 checkpoint loading for the 9D frozen probe."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Any

from filters.breakout_quality.mantis_contract import (
    MANTIS_PACKAGE_NAME,
    MANTIS_PACKAGE_VERSION,
    MANTIS_V2_CHECKPOINT_FILENAME,
    MANTIS_V2_CHECKPOINT_SHA256,
    MANTIS_V2_CONFIG_FILENAME,
    MANTIS_V2_CONFIG_SHA256,
    MANTIS_V2_REPOSITORY,
    MANTIS_V2_REVISION,
    require_mantis_v2_class,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_download():
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "9D MantisV2 需要 huggingface-hub；請執行 "
            "python -m pip install huggingface-hub safetensors einops"
        ) from exc
    return snapshot_download


def load_mantis_v2_pretrained_encoder_state(*, model_spec) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download, verify, truncate and return the immutable official encoder state."""
    if str(model_spec.family) != "mantis_v2_frozen_linear":
        raise ValueError("MantisV2 checkpoint loader 收到非 MantisV2 model spec")
    snapshot_path = Path(
        _snapshot_download()(
            repo_id=MANTIS_V2_REPOSITORY,
            revision=MANTIS_V2_REVISION,
            allow_patterns=(MANTIS_V2_CONFIG_FILENAME, MANTIS_V2_CHECKPOINT_FILENAME),
        )
    )
    checkpoint_path = snapshot_path / MANTIS_V2_CHECKPOINT_FILENAME
    config_path = snapshot_path / MANTIS_V2_CONFIG_FILENAME
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise ValueError(f"MantisV2 snapshot 缺少必要檔案: {snapshot_path}")
    actual_checkpoint_sha256 = _sha256(checkpoint_path)
    actual_config_sha256 = _sha256(config_path)
    if actual_checkpoint_sha256 != MANTIS_V2_CHECKPOINT_SHA256:
        raise ValueError(
            "MantisV2 checkpoint SHA256 不一致: "
            f"expected={MANTIS_V2_CHECKPOINT_SHA256}, actual={actual_checkpoint_sha256}"
        )
    if actual_config_sha256 != MANTIS_V2_CONFIG_SHA256:
        raise ValueError(
            "MantisV2 config SHA256 不一致: "
            f"expected={MANTIS_V2_CONFIG_SHA256}, actual={actual_config_sha256}"
        )

    MantisV2 = require_mantis_v2_class()
    configured = MantisV2(
        hidden_dim=int(model_spec.mantis_hidden_dim or 0),
        num_patches=int(model_spec.mantis_num_patches or 0),
        kernel_size=int(model_spec.kernel_size),
        scalar_scales=None,
        hidden_dim_scalar_enc=int(model_spec.mantis_scalar_hidden_dim or 0),
        epsilon_scalar_enc=float(model_spec.mantis_scalar_epsilon or 0.0),
        transf_depth=int(model_spec.mantis_transformer_depth or 0),
        transf_num_heads=int(model_spec.mantis_transformer_heads or 0),
        transf_mlp_dim=int(model_spec.mantis_transformer_mlp_dim or 0),
        transf_dim_head=int(model_spec.mantis_transformer_dim_head or 0),
        transf_dropout=float(model_spec.dropout),
        return_transf_layer=int(model_spec.mantis_return_transformer_layer or 0),
        output_token=str(model_spec.mantis_output_token or ""),
        device="cpu",
        pre_training=False,
    )
    encoder = configured.from_pretrained(str(snapshot_path))
    encoder.remove_transf_layers()
    encoder.eval()
    state = {
        key: value.detach().cpu()
        for key, value in encoder.state_dict().items()
    }
    if not state:
        raise ValueError("MantisV2 pretrained encoder state 不得為空")

    resolved_revision = snapshot_path.name
    if resolved_revision != MANTIS_V2_REVISION:
        raise ValueError(
            "MantisV2 snapshot resolved revision 不一致: "
            f"expected={MANTIS_V2_REVISION}, actual={resolved_revision}"
        )
    record = {
        "source_type": "hugging_face_snapshot",
        "repository": MANTIS_V2_REPOSITORY,
        "requested_revision": MANTIS_V2_REVISION,
        "resolved_revision": resolved_revision,
        "checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": actual_checkpoint_sha256,
            "size_bytes": int(checkpoint_path.stat().st_size),
        },
        "config": {
            "filename": config_path.name,
            "sha256": actual_config_sha256,
            "size_bytes": int(config_path.stat().st_size),
        },
        "package": {
            "name": MANTIS_PACKAGE_NAME,
            "version": version(MANTIS_PACKAGE_NAME),
            "expected_version": MANTIS_PACKAGE_VERSION,
        },
        "model_architecture": model_spec.architecture,
        "model_spec": model_spec.as_manifest_payload(),
        "encoder_frozen_downstream": True,
        "project_selection_windows_used_for_encoder_training": False,
        "project_oos_windows_used_for_encoder_training": False,
        "project_pass_reject_labels_used_for_encoder_training": False,
        "project_encoder_fine_tuning_used": False,
        "publisher_pretraining_description": "CauKer-2M synthetic time-series pretraining",
    }
    return state, record


__all__ = [
    "MANTIS_V2_CHECKPOINT_FILENAME",
    "MANTIS_V2_CHECKPOINT_SHA256",
    "MANTIS_V2_CONFIG_FILENAME",
    "MANTIS_V2_CONFIG_SHA256",
    "MANTIS_V2_REPOSITORY",
    "MANTIS_V2_REVISION",
    "load_mantis_v2_pretrained_encoder_state",
]

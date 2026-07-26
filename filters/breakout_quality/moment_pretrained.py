"""Pinned official MOMENT-1-base checkpoint loading for the 9E frozen probe."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from filters.breakout_quality.moment_contract import (
    MOMENT_CHECKPOINT_FILENAME,
    MOMENT_CHECKPOINT_SHA256,
    MOMENT_CHECKPOINT_SIZE_BYTES,
    MOMENT_CONFIG_FILENAME,
    MOMENT_CONFIG_SIZE_BYTES,
    MOMENT_PACKAGE_NAME,
    MOMENT_PACKAGE_VERSION,
    MOMENT_REPOSITORY,
    MOMENT_REVISION,
    MOMENT_TRANSFORMERS_PACKAGE_NAME,
    MOMENT_TRANSFORMERS_VERSION,
    require_moment_pipeline_class,
    validate_moment_config,
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
            "9E MOMENT 需要 huggingface-hub；請先安裝 requirements/requirements-moment.txt"
        ) from exc
    return snapshot_download


def load_moment_pretrained_encoder_state(*, model_spec) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download, verify and return a self-contained frozen MOMENT encoder state."""
    if str(model_spec.family) != "moment_frozen_linear":
        raise ValueError("MOMENT checkpoint loader 收到非 MOMENT model spec")
    snapshot_path = Path(
        _snapshot_download()(
            repo_id=MOMENT_REPOSITORY,
            revision=MOMENT_REVISION,
            allow_patterns=(MOMENT_CONFIG_FILENAME, MOMENT_CHECKPOINT_FILENAME),
        )
    )
    checkpoint_path = snapshot_path / MOMENT_CHECKPOINT_FILENAME
    config_path = snapshot_path / MOMENT_CONFIG_FILENAME
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise ValueError(f"MOMENT snapshot 缺少必要檔案: {snapshot_path}")

    checkpoint_size = int(checkpoint_path.stat().st_size)
    config_size = int(config_path.stat().st_size)
    checkpoint_sha256 = _sha256(checkpoint_path)
    config_sha256 = _sha256(config_path)
    if checkpoint_sha256 != MOMENT_CHECKPOINT_SHA256 or checkpoint_size != MOMENT_CHECKPOINT_SIZE_BYTES:
        raise ValueError(
            "MOMENT checkpoint 契約不一致: "
            f"expected_sha={MOMENT_CHECKPOINT_SHA256}, actual_sha={checkpoint_sha256}, "
            f"expected_size={MOMENT_CHECKPOINT_SIZE_BYTES}, actual_size={checkpoint_size}"
        )
    if config_size != MOMENT_CONFIG_SIZE_BYTES:
        raise ValueError(
            "MOMENT config size 不一致: "
            f"expected={MOMENT_CONFIG_SIZE_BYTES}, actual={config_size}"
        )
    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取 MOMENT config: {config_path}; {type(exc).__name__}: {exc}") from exc
    validate_moment_config(config_payload)

    MOMENTPipeline = require_moment_pipeline_class()
    encoder = MOMENTPipeline.from_pretrained(
        str(snapshot_path),
        model_kwargs={"task_name": "embedding"},
    )
    encoder.init()
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    state = {key: value.detach().cpu() for key, value in encoder.state_dict().items()}
    if not state:
        raise ValueError("MOMENT pretrained encoder state 不得為空")

    resolved_revision = snapshot_path.name
    if resolved_revision != MOMENT_REVISION:
        raise ValueError(
            "MOMENT snapshot resolved revision 不一致: "
            f"expected={MOMENT_REVISION}, actual={resolved_revision}"
        )
    record = {
        "source_type": "hugging_face_snapshot",
        "repository": MOMENT_REPOSITORY,
        "requested_revision": MOMENT_REVISION,
        "resolved_revision": resolved_revision,
        "checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
        },
        "config": {
            "filename": config_path.name,
            "sha256": config_sha256,
            "size_bytes": config_size,
            "semantic_validation": "exact_pinned_config",
        },
        "package": {
            "name": MOMENT_PACKAGE_NAME,
            "version": version(MOMENT_PACKAGE_NAME),
            "expected_version": MOMENT_PACKAGE_VERSION,
        },
        "runtime_dependencies": {
            MOMENT_TRANSFORMERS_PACKAGE_NAME: {
                "version": version(MOMENT_TRANSFORMERS_PACKAGE_NAME),
                "expected_version": MOMENT_TRANSFORMERS_VERSION,
            }
        },
        "model_architecture": model_spec.architecture,
        "model_spec": model_spec.as_manifest_payload(),
        "encoder_frozen_downstream": True,
        "project_selection_windows_used_for_encoder_training": False,
        "project_oos_windows_used_for_encoder_training": False,
        "project_pass_reject_labels_used_for_encoder_training": False,
        "project_encoder_fine_tuning_used": False,
        "publisher_pretraining_description": "Timeseries-PILE masked patch reconstruction pretraining",
    }
    return state, record


__all__ = [
    "MOMENT_CHECKPOINT_FILENAME",
    "MOMENT_CHECKPOINT_SHA256",
    "MOMENT_CONFIG_FILENAME",
    "MOMENT_REPOSITORY",
    "MOMENT_REVISION",
    "load_moment_pretrained_encoder_state",
]

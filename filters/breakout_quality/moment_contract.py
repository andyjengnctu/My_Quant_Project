"""Immutable external dependency and checkpoint contract for MOMENT-1-base 9E."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any


MOMENT_PACKAGE_NAME = "momentfm"
MOMENT_PACKAGE_VERSION = "0.1.4"
MOMENT_TRANSFORMERS_PACKAGE_NAME = "transformers"
MOMENT_TRANSFORMERS_VERSION = "5.5.0"
MOMENT_INSTALL_HINT = (
    "python -m pip install --index-url https://pypi.org/simple "
    '"transformers==5.5.0"；再執行：python -m pip install '
    '--index-url https://pypi.org/simple --no-deps "momentfm==0.1.4"'
)

MOMENT_REPOSITORY = "AutonLab/MOMENT-1-base"
MOMENT_REVISION = "b0ae5751d8ef43d72ad48fb5128e2ddc93c94b53"
MOMENT_CHECKPOINT_FILENAME = "model.safetensors"
MOMENT_CHECKPOINT_SHA256 = "1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825"
MOMENT_CHECKPOINT_SIZE_BYTES = 453_940_120
MOMENT_CONFIG_FILENAME = "config.json"
MOMENT_CONFIG_SIZE_BYTES = 949
MOMENT_INPUT_LENGTH = 512
MOMENT_PATCH_LENGTH = 8
MOMENT_PATCH_STRIDE = 8
MOMENT_EMBEDDING_DIM = 768
MOMENT_TRANSFORMER_LAYERS = 12
MOMENT_TRANSFORMER_HEADS = 12


_MOMENT_MODEL_CONFIG: dict[str, Any] = {
    "task_name": "reconstruction",
    "model_name": "MOMENT",
    "transformer_type": "encoder_only",
    "d_model": None,
    "seq_len": MOMENT_INPUT_LENGTH,
    "patch_len": MOMENT_PATCH_LENGTH,
    "patch_stride_len": MOMENT_PATCH_STRIDE,
    "device": "cpu",
    "transformer_backbone": "google/flan-t5-base",
    "model_kwargs": {},
    "t5_config": {
        "architectures": ["T5ForConditionalGeneration"],
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": MOMENT_EMBEDDING_DIM,
        "decoder_start_token_id": 0,
        "dropout_rate": 0.1,
        "eos_token_id": 1,
        "feed_forward_proj": "gated-gelu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "layer_norm_epsilon": 1e-6,
        "model_type": "t5",
        "n_positions": MOMENT_INPUT_LENGTH,
        "num_decoder_layers": MOMENT_TRANSFORMER_LAYERS,
        "num_heads": MOMENT_TRANSFORMER_HEADS,
        "num_layers": MOMENT_TRANSFORMER_LAYERS,
        "output_past": True,
        "pad_token_id": 0,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 32128,
    },
}


def moment_model_config() -> dict[str, Any]:
    """Return a fresh copy of the pinned official MOMENT-1-base config."""
    import copy

    return copy.deepcopy(_MOMENT_MODEL_CONFIG)


def validate_moment_config(payload: Any) -> None:
    """Reject any checkpoint config that differs from the pinned model semantics."""
    if not isinstance(payload, dict):
        raise ValueError("MOMENT config 根節點必須是 object")
    if payload != _MOMENT_MODEL_CONFIG:
        raise ValueError(
            "MOMENT config 與釘死的 MOMENT-1-base 規格不一致: "
            f"expected={_MOMENT_MODEL_CONFIG}, actual={payload}"
        )


def _require_exact_package_version(package_name: str, expected_version: str) -> str:
    try:
        installed_version = version(package_name)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"9E MOMENT 缺少 {package_name}=={expected_version}；請先執行："
            f"{MOMENT_INSTALL_HINT}"
        ) from exc
    if installed_version != expected_version:
        raise RuntimeError(
            "9E MOMENT 套件版本不一致："
            f"package={package_name}, expected={expected_version}, actual={installed_version}；"
            f"請執行：{MOMENT_INSTALL_HINT}"
        )
    return installed_version


def require_moment_pipeline_class():
    """Load the pinned MOMENT pipeline lazily so normal project imports stay optional."""
    _require_exact_package_version(
        MOMENT_TRANSFORMERS_PACKAGE_NAME,
        MOMENT_TRANSFORMERS_VERSION,
    )
    _require_exact_package_version(MOMENT_PACKAGE_NAME, MOMENT_PACKAGE_VERSION)
    try:
        from momentfm import MOMENTPipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "無法匯入 momentfm.MOMENTPipeline；請確認 transformers 與 momentfm 已完整安裝："
            f"{MOMENT_INSTALL_HINT}"
        ) from exc
    return MOMENTPipeline


__all__ = [
    "MOMENT_CHECKPOINT_FILENAME",
    "MOMENT_CHECKPOINT_SHA256",
    "MOMENT_CHECKPOINT_SIZE_BYTES",
    "MOMENT_CONFIG_FILENAME",
    "MOMENT_CONFIG_SIZE_BYTES",
    "MOMENT_EMBEDDING_DIM",
    "MOMENT_INPUT_LENGTH",
    "MOMENT_INSTALL_HINT",
    "MOMENT_PACKAGE_NAME",
    "MOMENT_PACKAGE_VERSION",
    "MOMENT_PATCH_LENGTH",
    "MOMENT_PATCH_STRIDE",
    "MOMENT_REPOSITORY",
    "MOMENT_REVISION",
    "MOMENT_TRANSFORMER_HEADS",
    "MOMENT_TRANSFORMER_LAYERS",
    "MOMENT_TRANSFORMERS_PACKAGE_NAME",
    "MOMENT_TRANSFORMERS_VERSION",
    "moment_model_config",
    "require_moment_pipeline_class",
    "validate_moment_config",
]

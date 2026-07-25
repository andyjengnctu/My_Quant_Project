"""Immutable external dependency and checkpoint contract for MantisV2 9D."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


MANTIS_PACKAGE_NAME = "mantis-tsfm"
MANTIS_PACKAGE_VERSION = "1.0.0"
MANTIS_INSTALL_HINT = (
    "python -m pip install einops huggingface-hub safetensors && "
    "python -m pip install --no-deps mantis-tsfm==1.0.0"
)

MANTIS_V2_REPOSITORY = "paris-noah/MantisV2"
MANTIS_V2_REVISION = "99fe0f548960e272fbfa4b82fd9b5b5956779dfd"
MANTIS_V2_CHECKPOINT_FILENAME = "model.safetensors"
MANTIS_V2_CHECKPOINT_SHA256 = "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1"
MANTIS_V2_CONFIG_FILENAME = "config.json"
MANTIS_V2_CONFIG_SHA256 = "f28885b7aa662cedcaf5627659fc0fba9cf8ee49d5b991280522e04fa5b1ae80"


def require_mantis_v2_class():
    """Load the pinned external architecture without importing its trainer stack."""
    try:
        installed_version = version(MANTIS_PACKAGE_NAME)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "9D MantisV2 需要 mantis-tsfm==1.0.0；請先執行："
            f"{MANTIS_INSTALL_HINT}"
        ) from exc
    if installed_version != MANTIS_PACKAGE_VERSION:
        raise RuntimeError(
            "9D MantisV2 套件版本不一致："
            f"expected={MANTIS_PACKAGE_VERSION}, actual={installed_version}；"
            f"請執行：{MANTIS_INSTALL_HINT}"
        )
    try:
        from mantis.architecture import MantisV2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "無法匯入 mantis.architecture.MantisV2；請確認相依套件完整："
            f"{MANTIS_INSTALL_HINT}"
        ) from exc
    return MantisV2


__all__ = [
    "MANTIS_INSTALL_HINT",
    "MANTIS_PACKAGE_NAME",
    "MANTIS_PACKAGE_VERSION",
    "MANTIS_V2_CHECKPOINT_FILENAME",
    "MANTIS_V2_CHECKPOINT_SHA256",
    "MANTIS_V2_CONFIG_FILENAME",
    "MANTIS_V2_CONFIG_SHA256",
    "MANTIS_V2_REPOSITORY",
    "MANTIS_V2_REVISION",
    "require_mantis_v2_class",
]

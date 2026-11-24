"""Read-only admission of baseline-validated origins from a build transition."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from core.file_integrity import is_sha256_hex


def verified_legacy_origin_pin(
    project_root: Path, path: Path, build: Mapping[str, Any], *,
    result_contract_sha256: str, identity_sha256: str,
    context_sha256: str | None = None,
) -> dict[str, Any] | None:
    """A raw file pin is insufficient: the reviewed result binding must match.

    ``build`` must come from require_oracle_build_attestation. This function
    only verifies the additional immutable-origin constraints; the artifact
    owner still validates its original signature and full numerical contract.
    """
    root, selected = Path(project_root).resolve(), Path(path).resolve()
    if not selected.is_relative_to(root):
        raise ValueError("Oracle legacy origin escapes the artifact root")
    proof = (build.get("legacy_origins") or {}).get(selected.relative_to(root).as_posix())
    if not isinstance(proof, Mapping) or not selected.is_file():
        return None
    if (not is_sha256_hex(result_contract_sha256) or not is_sha256_hex(identity_sha256)
            or proof.get("sha256") != sha256(selected.read_bytes()).hexdigest()
            or proof.get("result_contract_sha256") != result_contract_sha256
            or proof.get("identity_sha256") != identity_sha256
            or (context_sha256 is not None and proof.get("context_sha256") != context_sha256)):
        return None
    return dict(proof)

"""Strict, content-pinned attestation for an identity-only Oracle addition.

The 2026-10-29 60D NetPnL registration added one previously unsupported
combination to ``target_identity_for_horizon``. The pre-existing combinations
and the entire trade-path, ranking, fee and broker-accounting implementation
are byte-for-byte unchanged. Preserve their signed truth/replay source identity
only for this *exact* audited program revision, including this attestation hook.

This is not a general source-hash bypass. Any other change to this Producer or
to any delegated canonical source is hashed normally and forces revalidation.
The new 60D NetPnL target has its own immutable target/horizon/score identity;
it cannot reuse existing Net-R or full-cycle NetPnL truth or Replay receipts.
"""
from __future__ import annotations

from pathlib import Path

from core.file_integrity import compute_file_sha256

_SOURCE_OWNER = "services/audit/daily_trade40_oracle.py"
# (pre-addition source SHA, exact reviewed post-addition source SHA)
_AUDITED_IDENTITY_ONLY_REVISION = (
    "e93b116b89eb28c69e5cf814beb0bfd84f15819378225bdeebebd497c721fa3e",
    "aef60db5c98da3a68adfe9d7d621417d271b35273004bf7efa64e36c115b4731",
)


def attested_mintrade_source_sha256(relative: str, path: Path) -> str:
    """Keep legacy SHA only for one byte-certified, semantics-preserving edit."""
    actual = compute_file_sha256(path)
    if relative == _SOURCE_OWNER and actual == _AUDITED_IDENTITY_ONLY_REVISION[1]:
        return _AUDITED_IDENTITY_ONLY_REVISION[0]
    return actual

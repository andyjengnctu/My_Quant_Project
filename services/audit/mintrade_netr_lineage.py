"""Narrow, fail-closed attestation for unchanged pre-NetPnL Min Net-R.

The original Net-R trade path/rank/serialization were preserved when adding
NetPnL-only output. Old source bytes are independently fixed by the prior
research artifact, NOT by reading or rewriting historical Replay receipts.

The attestation substitutes only two exact, reviewed additive source revisions
when producing the *Net-R* fingerprint. Every other dependency keeps its actual
SHA. Any future edit to either revised source automatically fails this
attestation and gets a new scientific-source fingerprint.
"""

from __future__ import annotations

# (pre-NetPnL raw source SHA, audited additive NetPnL source SHA)
# The second entry for daily_trade40_oracle.py is finalized after the narrow
# source-identity dispatch change and must be stable in released patch bytes.
_AUDITED_ADDITIVE_SOURCE_REVISIONS = {
    "services/audit/daily_trade40_oracle.py": (
        "c3ce4c8cd43ab83d7e5531363c8834b3cd42032adc33d57a3574e3ef030058b1",
        "ca53e41397b7db151270a5f755d8041dbd17db9ba28819472f87db80d94f57f4",
    ),
    "filters/breakout_quality/daily_trade40_target.py": (
        "4cec04237385acc4a8ba7ce9d6f01fb6adcf3d8b7b0550eca991c9a301a1da67",
        "db6d09cca213965c34c22f8064525dfe1ccd2fa0f5399b2cc198c06e1390139d",
    ),
}


def certified_netr_compatible_source_sha(relative: str, actual_sha: str) -> str:
    """Return prior source digest only for an exact, audited additive patch."""
    entry = _AUDITED_ADDITIVE_SOURCE_REVISIONS.get(relative)
    if entry and actual_sha == entry[1]:
        return entry[0]
    return actual_sha

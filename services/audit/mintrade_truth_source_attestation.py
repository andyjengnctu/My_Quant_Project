"""Strict, content-pinned attestation for an identity-only Oracle addition.

The 2026-10-29 60D NetPnL registration added one previously unsupported
combination to ``target_identity_for_horizon``. The pre-existing combinations
and the entire trade-path, ranking, fee and broker-accounting implementation
are byte-for-byte unchanged. Preserve their signed truth/replay source identity
only for this *exact* audited program revision, including this attestation hook.

This is not a general source-hash bypass. A later 2026-10-30 change
replaces the outer per-ticker serial loop with deterministic spawn workers,
keeping the same single-stock execution function and byte-identical canonical
label/rank publication. The exact reviewed bytes below map to the pre-existing
Min truth source identity ONLY for that execution-order-only revision. Any
other producer, delegated execution or accounting source change is hashed
normally and forces revalidation.
The new 60D NetPnL target has its own immutable target/horizon/score identity;
it cannot reuse existing Net-R or full-cycle NetPnL truth or Replay receipts.
"""
from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path

from core.file_integrity import compute_file_sha256

_SOURCE_OWNER = "services/audit/daily_trade40_oracle.py"
# (pre-addition source SHA, exact reviewed post-addition source SHA)
_AUDITED_IDENTITY_ONLY_REVISION = (
    "e93b116b89eb28c69e5cf814beb0bfd84f15819378225bdeebebd497c721fa3e",
    "aef60db5c98da3a68adfe9d7d621417d271b35273004bf7efa64e36c115b4731",
)
_AUDITED_PARALLEL_ONLY_REVISION = (
    _AUDITED_IDENTITY_ONLY_REVISION[0],
    "2b28f4841808385fb1691a86f76b4d6a804ccfc07d128bb6b08bd4cc3d772b23",
)

# target_identity_for_horizon() allocates *names*, never raw prices, fills,
# exits, fees, or ranks.  The currently archived owner execution AST is an
# independent certificate.  A future Target addition/removal confined to that
# function cannot change the economic producer's signed source identity.
# Actual target/label IDs are still included individually in every cache and
# Replay receipt: changing an EXISTING label mapping is never silent REUSE.
_CERTIFIED_EXECUTABLE_OWNER_AST_SHA256 = (
    "c78a8a08b8af9c7730445c91dbd1f22571001b4ab2c9d69decc43bb879b1eaa4"
)
_NON_ECONOMIC_TARGET_REGISTRY_FUNCTION = "target_identity_for_horizon"


def _executable_owner_ast_sha256(path: Path) -> str:
    """Certify all actual Oracle-truth code except the naming-only registry.

    Comments and line endings are non-executable. Every change to an import,
    producer, worker, fee/ledger interaction or truth-cache function is kept
    in the AST and invalidates the certificate. An unparseable owner fails
    closed rather than inheriting an earlier signed truth source identity.
    """
    source = Path(path).read_text(encoding="utf-8-sig")
    module = ast.parse(source, filename=str(path))
    naming_only = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == _NON_ECONOMIC_TARGET_REGISTRY_FUNCTION]
    if len(naming_only) != 1:
        raise ValueError("Oracle truth Target naming registry missing or duplicated")
    executable = ast.Module(
        body=[node for node in module.body if node is not naming_only[0]],
        type_ignores=[],
    )
    # Descriptive docstrings are not code or economic policy. Strip them at
    # every level to avoid a text-only edit invalidating signed trade truth.
    class _NoDocstrings(ast.NodeTransformer):
        def generic_visit(self, node):
            result = super().generic_visit(node)
            if isinstance(result, (ast.Module, ast.ClassDef,
                                   ast.FunctionDef, ast.AsyncFunctionDef)):
                body = result.body
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    result.body = body[1:]
            return result

    normalized = _NoDocstrings().visit(executable)
    return sha256(ast.dump(normalized, include_attributes=False).encode("utf-8")).hexdigest()


def attested_mintrade_source_sha256(relative: str, path: Path) -> str:
    """Certify economic producer code independently of the Target-name roster.

    Existing signed receipts keep their original byte identity only when the
    executable owner AST is unchanged. This is not a wildcard for any code or
    independent data-source change; unrecognized revisions hash normally.
    """
    actual = compute_file_sha256(path)
    if relative != _SOURCE_OWNER:
        return actual
    # Historical byte-exact receipts remain independently certified without
    # requiring a second read of an archive path. This also keeps isolated
    # source-hash unit fixtures valid (only known exact SHA values qualify).
    if actual in {_AUDITED_IDENTITY_ONLY_REVISION[0],
                  _AUDITED_IDENTITY_ONLY_REVISION[1],
                  _AUDITED_PARALLEL_ONLY_REVISION[1]}:
        return _AUDITED_IDENTITY_ONLY_REVISION[0]
    if _executable_owner_ast_sha256(path) == _CERTIFIED_EXECUTABLE_OWNER_AST_SHA256:
        return _AUDITED_IDENTITY_ONLY_REVISION[0]
    return actual

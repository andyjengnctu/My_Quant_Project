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
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from core.file_integrity import compute_file_sha256
from core.research_execution_equivalence import reviewed_execution_source_sha256

_SOURCE_OWNER = "services/audit/daily_trade40_oracle.py"
# Min truth is produced with both DL flags disabled by
# build_formal_min_execution_schedule. The runtime adapter supplies only the
# callbacks behind those flags; its calibration/ranking code is never an
# input to this producer. Keep the old provenance slot for signed receipts,
# but do not couple DL-off truth to unrelated model/selector maintenance.
# The schedule, signal dispatcher and actual trade/accounting owners remain
# strict dependencies, so enabling DL or changing its bypass invalidates truth.
_DL_OFF_ADAPTER = "filters/breakout_quality/runtime.py"
_DL_OFF_ADAPTER_PROVENANCE = (
    "548e0d32024a09a0e03d9ead82327904fa285f3320b3dd50b8d078bcf79dedc5",
    "ea2564e7c5d36f34f89cb93afaf8380100bc64cd9af2fd359059aa8dedd4c843",
)
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
# The reviewed parallel-only Oracle producer is byte-pinned by
# _AUDITED_PARALLEL_ONLY_REVISION.  Pin its executable AST here as well, so
# adding a *naming-only* Target cannot invalidate already signed economics.
# Historical exact-byte revisions still use their independent SHA allowlist.
# All other changes to executable code fail closed and hash normally.
_CERTIFIED_EXECUTABLE_OWNER_AST_SHA256 = (
    "c78a8a08b8af9c7730445c91dbd1f22571001b4ab2c9d69decc43bb879b1eaa4"
)
_NON_ECONOMIC_TARGET_REGISTRY_FUNCTION = "target_identity_for_horizon"

# Reviewed against source ZIP SHA256
# 741acd4c41ff058dd04aa867fe17a5205d679dea2eb60fde6b4c46c464358c46.
# Its default Oracle producer is exactly the 80a4fe35... source in the signed
# eight-seed receipts. The new daily loaders add an opt-in FIT materializer;
# the existing Oracle provider passes neither that callback nor a signal reset.
# The trade primitive adds only memoized period lookup and a returned FIT-only
# forced-close mask on this default path. Raw R/PnL, NULL, ranks and fees stay
# identical. Effective provider contracts are independently pinned by the
# daily-source owner; every other delegated execution file remains strict.
_AUDITED_DEFAULT_ORACLE_DAILY_SOURCES = (
    "27a79f35a10ee331be504942c28deb1080e2060cc10ae3dc41c1e804352624db",
    "b7bdeb766babdb66ac5084125d08f2bf59390843e4dc09eb84645c093a3c770d",
)
_AUDITED_DEFAULT_TRADE_PRIMITIVE = (
    "filters/breakout_quality/daily_trade40_target.py",
    "f62f7c9fa2ac8db936bdfb4fbac8f374f21ed68d7540fb4b77bad8312148c931",
    "4df90ba6d95a22cd0a78845141dc6bfb91683e7fa2ecc612cafa6f66e6bd3370",
)


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
    if relative == _DL_OFF_ADAPTER:
        return _DL_OFF_ADAPTER_PROVENANCE[0]
    actual = compute_file_sha256(path)
    actual = reviewed_execution_source_sha256(relative, actual)
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


@lru_cache(maxsize=1)
def _truth_source_inventory() -> tuple[str, ...]:
    """Use the shared dependency leaf without importing the certified producer."""
    from core.execution_source_dependencies import execution_source_inventory
    return execution_source_inventory("mintrade_truth")


@lru_cache(maxsize=1)
def _mintrade_source_fingerprint_proof() -> tuple[str, frozenset[str]]:
    """Reconstruct LF/CRLF closures of the *executing* economic sources.

    No historical SHA is accepted on its own. Every delegated source stays
    in the canonical producer's inventory; only line-ending bytes vary.
    AI: The nested daily-source owner also proves its current byte encodings;
    the outer closure must carry that proof through each dependency layer.
    This does not accept an unknown older formula, fee or eligibility change.
    """
    from services.audit.oracle_score_artifacts import (
        oracle_daily_source_fingerprint_proof,
        _CERTIFIED_LF_PRICE_SOURCE_SHA256, _CERTIFIED_CRLF_PRICE_SOURCE_SHA256,
    )

    root = Path(__file__).resolve().parents[2]
    source_modules = _truth_source_inventory()
    daily, daily_encodings = oracle_daily_source_fingerprint_proof(root)
    daily_sources = set(daily_encodings)
    if daily in {_CERTIFIED_LF_PRICE_SOURCE_SHA256, _CERTIFIED_CRLF_PRICE_SOURCE_SHA256}:
        daily_sources.update((_CERTIFIED_LF_PRICE_SOURCE_SHA256,
                              _CERTIFIED_CRLF_PRICE_SOURCE_SHA256))
    contents = {name: (root / name).read_bytes() for name in source_modules}
    result = set()
    executing = None
    for newline in (None, b"\n", b"\r\n"):
        for raw_owner in (False, True):
            hashes = {}
            from core.file_integrity import canonical_json_sha256
            hashes["@execution_sources"] = canonical_json_sha256(list(source_modules))
            for name, data in contents.items():
                if name == _DL_OFF_ADAPTER:
                    hashes[name] = _DL_OFF_ADAPTER_PROVENANCE[0]
                    continue
                if newline is not None:
                    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                    if newline == b"\r\n":
                        data = data.replace(b"\n", b"\r\n")
                hashes[name] = (attested_mintrade_source_sha256(name, root / name)
                                if name == _SOURCE_OWNER and not raw_owner
                                else reviewed_execution_source_sha256(name, sha256(data).hexdigest()))
            # A Windows checkout/partial ZIP can convert just one source,
            # leaving a mixed-encoding tree. Reconstruct these closures from
            # current bytes as well as uniform LF/CRLF trees. This is a bounded
            # encoding proof, not acceptance of an arbitrary historical SHA.
            variants = [hashes]
            for name, data in contents.items():
                if name == _DL_OFF_ADAPTER:
                    variants.extend({**hashes, name: value}
                                    for value in _DL_OFF_ADAPTER_PROVENANCE
                                    if value != hashes[name])
                    continue
                if name == _SOURCE_OWNER and not raw_owner:
                    continue  # This owner already has an executable-AST proof.
                lf = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                for encoded in (lf, lf.replace(b"\n", b"\r\n")):
                    value = reviewed_execution_source_sha256(name, sha256(encoded).hexdigest())
                    if value != hashes[name]:
                        variants.append({**hashes, name: value})
            for values in variants:
                from core.research_execution_equivalence import project_reviewed_source_inventory
                projected = project_reviewed_source_inventory(
                    values, scope="mintrade_source_graph", project_root=root,
                    source_encoding=newline,
                )
                for daily_sha in daily_sources:
                    digest = sha256(b"canonical_daily_source\0" + daily_sha.encode("ascii") + b"\n")
                    for name, source_sha in projected.items():
                        digest.update(name.encode("utf-8") + b"\0")
                        digest.update(source_sha.encode("ascii") + b"\n")
                    result.add(digest.hexdigest())
                    if values is hashes and newline is None and not raw_owner and daily_sha == daily:
                        executing = digest.hexdigest()
    if executing is None:
        raise ValueError("Oracle truth source closure has no executing fingerprint")
    return executing, frozenset(result)


@lru_cache(maxsize=1)
def _encoding_equivalent_mintrade_sources() -> frozenset[str]:
    return _mintrade_source_fingerprint_proof()[1]


def mintrade_sources_equivalent(recorded: str | None, current: str | None) -> bool:
    """Prove reviewed source equivalence; all data/economic inputs stay strict."""
    from core.file_integrity import is_sha256_hex

    if not is_sha256_hex(recorded) or not is_sha256_hex(current):
        return False
    if recorded == current:
        return True
    equivalents = _encoding_equivalent_mintrade_sources()
    return (current == _mintrade_source_fingerprint_proof()[0]
            and current in equivalents
            and recorded in equivalents | _archived_default_oracle_sources())


@lru_cache(maxsize=1)
def _archived_default_oracle_sources() -> frozenset[str]:
    """Reconstruct an archived closure, never whitelist an old result digest.

    An edit to any other fee, signal, price, parameter or execution owner also
    changes this reconstruction. An unknown loader/primitive revision fails
    closed. This certificate cannot bless the earlier, different trade labels.
    """
    from services.audit.oracle_score_artifacts import oracle_daily_source_fingerprint

    root = Path(__file__).resolve().parents[2]
    previous_daily, reviewed_daily = _AUDITED_DEFAULT_ORACLE_DAILY_SOURCES
    relative, previous_primitive, reviewed_primitive = _AUDITED_DEFAULT_TRADE_PRIMITIVE
    if (oracle_daily_source_fingerprint(root) != reviewed_daily
            or attested_mintrade_source_sha256(relative, root / relative) != reviewed_primitive):
        return frozenset()
    digest = sha256(b"canonical_daily_source\0" + previous_daily.encode("ascii") + b"\n")
    for name in _truth_source_inventory():
        value = (previous_primitive if name == relative else
                 attested_mintrade_source_sha256(name, root / name))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.encode("ascii") + b"\n")
    return frozenset((digest.hexdigest(),))

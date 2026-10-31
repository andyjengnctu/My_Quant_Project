"""Durable, content-certified perfect-price Target score panels.

This score producer cache is deliberately independent of portfolio Replay,
selected DailyUniversal horizons and report rendering. Only source-equivalent
score tables can REUSE. Cache absence authorizes a score BUILD, never a
Portfolio Replay; a corrupt cache always BLOCKS rather than rebuilding over it.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Callable

import numpy as np

from core.file_integrity import (
    atomic_replace_with_retry, canonical_json_sha256,
    compute_file_sha256, exclusive_file_lock, is_sha256_hex,
)
from filters.breakout_quality.dataset_store import resolve_dataset_paths
from filters.breakout_quality.paths import resolve_filter_output_dir

_PRICE_SCORE_CACHE_SCHEMA = 1
# Includes pure-price and official daily-DL eligibility capabilities, but not
# portfolio execution, menu, rendering or experiment report source code.
_DAILY_SCORE_SOURCE_MODULES = (
    "filters/breakout_quality/daily_ranker_data.py",
    "filters/breakout_quality/continuous_ranker_data.py",
    "filters/breakout_quality/continuous_target.py",
    "filters/breakout_quality/mfe_safety_geometry.py",
    "filters/breakout_quality/profile_ranker_data.py",
    "filters/breakout_quality/workflow_io.py",
    "filters/breakout_quality/csv_io.py",
    "filters/breakout_quality/splits.py",
    "filters/breakout_quality/dataset_store.py",
    "filters/breakout_quality/dataset_readiness.py",
    "filters/breakout_quality/contract.py",
    "filters/breakout_quality/features.py",
    "filters/breakout_quality/models/architectures.py",
    "filters/breakout_quality/models/spec.py",
    "filters/breakout_quality/models/spec_contract.py",
    "filters/breakout_quality/models/spec_registry.py",
    "filters/breakout_quality/models/spec_builders.py",
    "core/data_utils.py",
    "core/breakout_quality_registry.py",
    "core/breakout_quality_policy.py",
    "core/breakout_quality_runtime.py",
    "core/breakout_quality_runtime_resolver.py",
    "core/dataset_profiles.py",
    "core/walk_forward_policy.py",
    "config/breakout_quality.py",
    "config/research.py",
    "config/training_policy.py",
    "config/market_data.py",
)


# Price-score lineage is a dependency graph, not a hash of all research
# experiments currently registered.  The three declarative model owner files
# can acquire new *unrelated* DL profiles without altering the four existing
# Oracle price/source profiles.  Preserve the earlier signed source identity
# only when those exact effective provider contracts remain byte-equivalent.
#
# The fourth change was an independently reviewed NetPnL-only diagnostic in
# daily_ranker_data. It has a much narrower exception: any subsequent change
# to the shared data loader still invalidates source identity as before.
# All other feature/price/execution sources retain strict byte-level hashes.
_ORACLE_PRICE_PROVIDER_ROLES = (
    "pure_mfe40_provider", "mfe20_d20_provider", "first_breach_provider",
    "daily_trade40_profile",
)
_ORACLE_PRICE_PROFILE_CONTRACT_SHA256 = (
    "31e7937daf822ce4636038f6de639704409a8f6776c76a22e65de211c02a9433"
)
# A current training-model choice or the set of compared DL models cannot
# influence historical price-truth labels. All other project BQ configuration
# fields remain attested; changing a feature/eligibility policy is NOT ignored.
_NON_EXECUTABLE_MODEL_SELECTION_FIELDS = frozenset((
    "BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE",
    "BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE",
    "BREAKOUT_QUALITY_MODEL_TEST_PROFILES",
    "BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES",
    "BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_PROFILES",
))
# Configuration is projected by *ownership*, not by the current finite model
# roster.  Future MR additions/deletions may add constants with the same
# workflow prefixes without silently invalidating unrelated Oracle prices.
# Only declarations used to select/compare models are excluded; the model
# architecture, PIT dates, data eligibility, risk and feature policies remain
# part of the source certificate.
_NON_EXECUTABLE_MODEL_WORKFLOW_PREFIXES = (
    "BREAKOUT_QUALITY_MODEL_RESEARCH_",
    "BREAKOUT_QUALITY_MODEL_TEST_",
    "BREAKOUT_QUALITY_STANDARD_MODEL_COMPARISON_",
    "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_",
    "BREAKOUT_QUALITY_ROLLING_TIMING_",
)
_ORACLE_PRICE_CONFIGURATION_SHA256 = (
    "e82d7ac8fe0d9c9879a0e9afa5991c448b0a0d119a5de2e84493cf3615c8f82f"
)
_LEGACY_PRICE_REGISTRY_SOURCE_SHA256 = {
    "core/breakout_quality_registry.py": "538a9508b0522566e4944f727f09edd6015eaf8cb63a80454355710c8a62d4a3",
    "core/breakout_quality_runtime.py": "84c8bbf9aa3fe35f9a9552e6881fee6b859a60ce5eee71932c536fda44372749",
    "config/breakout_quality.py": "b6799f8409488f91efa3c122635147ff5d07a72efabde2832e0a6464d3728b59",
}
_AUDITED_NETPNL_DIAGNOSTIC_ONLY_REVISION = (
    "5fe6caf6e64fec14486c1b8b2bdc855bef244d2044fa7645f0b3f7b936291116",
    "91254d3ba1d71ef95a1a96dffed068335fa3b239bf6fb42f9baf64406d247906",
)


def _oracle_price_provider_contract_sha256() -> str:
    """Hash *effective* provider policies, not unrelated MR registration rows.

    This is a source-attestation boundary. An Oracle provider edit invalidates
    the legacy equivalence, whereas adding/removing an unrelated DL model does
    not. Runtime resolution and data/feature implementation remain certified
    independently by the source closure below.
    """
    from dataclasses import asdict
    from core.audit_registry import AUDIT_REUSABLE_REPORTS
    from core.breakout_quality_registry import get_breakout_quality_experiment_profile
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe

    registered = AUDIT_REUSABLE_REPORTS["breakout_quality"]["target_oracle_romd"]["source"]
    contracts = {}
    for role in _ORACLE_PRICE_PROVIDER_ROLES:
        profile = str(registered[role])
        contracts[role] = {
            "profile_id": profile,
            "profile": asdict(get_breakout_quality_experiment_profile(profile)),
            "recipe": asdict(get_continuous_ranker_execution_recipe(profile)),
        }
    return canonical_json_sha256(contracts)


def _oracle_price_configuration_sha256() -> str:
    import config.breakout_quality as cfg
    source = {
        name: value for name, value in vars(cfg).items()
        if name.isupper() and not name.startswith("_")
        and name not in _NON_EXECUTABLE_MODEL_SELECTION_FIELDS
        and not name.startswith(_NON_EXECUTABLE_MODEL_WORKFLOW_PREFIXES)
        and not callable(value)
    }
    return canonical_json_sha256(source)



def oracle_daily_source_fingerprint(root: Path | None = None) -> str:
    """Canonical feature/price execution lineage, independent of model inventory.

    The historical source digest can only be retained after proving that the
    *effective* Oracle provider policies equal the archived, independently
    checked source revision. A new/unrelated DL registration does not affect
    that proof. Unknown shared-loader or price-primitive edits always produce
    a new digest, so stale prices cannot be falsely REUSEd.
    """
    code_root = Path(__file__).resolve().parents[2]
    root = Path(root) if root is not None else code_root
    # Synthetic fixtures with a separate source root must keep ordinary strict
    # byte hashing; a project-local attestation cannot certify arbitrary files.
    attested = root.resolve() == code_root.resolve()
    if attested:
        try:
            attested = (
                _oracle_price_provider_contract_sha256()
                == _ORACLE_PRICE_PROFILE_CONTRACT_SHA256
                and _oracle_price_configuration_sha256()
                == _ORACLE_PRICE_CONFIGURATION_SHA256
            )
        except (ValueError, KeyError, TypeError):
            attested = False
    source_digest = sha256()
    for relative in _DAILY_SCORE_SOURCE_MODULES:
        source_file = root / relative
        if not source_file.is_file():
            raise ValueError(f"Oracle price-score source implementation unavailable: {relative}")
        physical_sha = compute_file_sha256(source_file)
        source_sha = physical_sha
        if attested and relative in _LEGACY_PRICE_REGISTRY_SOURCE_SHA256:
            source_sha = _LEGACY_PRICE_REGISTRY_SOURCE_SHA256[relative]
        elif (attested and relative == "filters/breakout_quality/daily_ranker_data.py"
              and physical_sha == _AUDITED_NETPNL_DIAGNOSTIC_ONLY_REVISION[0]):
            source_sha = _AUDITED_NETPNL_DIAGNOSTIC_ONLY_REVISION[1]
        source_digest.update(relative.encode("utf-8") + b"\0")
        source_digest.update(source_sha.encode("ascii") + b"\n")
    return source_digest.hexdigest()



# The 154445 and 170351 historical price producer closures were independently
# checked: their four changed files are *byte-identical after CRLF -> LF*,
# including the actual runtime and target data owner. The source SHA alone
# differs because the 170351 ZIP serialized those same source files as CRLF.
# This is a single historical encoding-equivalence statement, NOT a wildcard
# allowing unknown producer changes or missing receipt evidence.
_CERTIFIED_CRLF_PRICE_SOURCE_SHA256 = (
    "71e08430171303ede7f8868f2b46993bf40941ac602d381da1c49deb632fc881"
)
_CERTIFIED_LF_PRICE_SOURCE_SHA256 = (
    "b7bdeb766babdb66ac5084125d08f2bf59390843e4dc09eb84645c093a3c770d"
)
_CERTIFIED_PRICE_ALGORITHM_SHA256 = (
    "d73b4ce20dd958cb75b094f49d6abac94d31e34ca7222fa5806675fd931964bd"
)
_PRICE_LINEAGE_FIELDS = frozenset((
    "source_code_sha256", "score_algorithm_sha256",
    "dataset_summary_sha256", "label_policy", "truth_source_sha256",
))


def certified_price_lineages_equivalent(
    recorded: Mapping[str, Any], current: Mapping[str, Any],
) -> bool:
    """Prove exactly one signed LF/CRLF-only historical producer equivalence.

    The caller still verifies signed Replay cache bytes, identical actual
    perfect-score SHA, feature-eligible stock-days, seed, P16, price inventory,
    selector, period and full portfolio economics. A changed source formula,
    dataset summary, label policy or any other lineage field fails closed.
    """
    if not isinstance(recorded, Mapping) or not isinstance(current, Mapping):
        return False
    if dict(recorded) == dict(current):
        return True
    # During warm witness discovery the producer returns the base source
    # lineage; saved Replay receipts additionally carry a score recipe/profile
    # and target label.  Their full equality is independently required when
    # building the final expected Replay identity below.
    allowed_replay_annotations = {"profile_id", "score_recipe", "target_id"}
    if (not _PRICE_LINEAGE_FIELDS.issubset(recorded)
            or not _PRICE_LINEAGE_FIELDS.issubset(current)
            or (set(recorded) ^ set(current)) - allowed_replay_annotations
            or any(recorded[key] != current[key] for key in set(recorded) & set(current)
                   if key not in {"source_code_sha256", "truth_source_sha256"})):
        return False
    if (current.get("source_code_sha256") != _CERTIFIED_LF_PRICE_SOURCE_SHA256
            or recorded.get("source_code_sha256") != _CERTIFIED_CRLF_PRICE_SOURCE_SHA256
            or current.get("score_algorithm_sha256") != _CERTIFIED_PRICE_ALGORITHM_SHA256
            or recorded.get("score_algorithm_sha256") != _CERTIFIED_PRICE_ALGORITHM_SHA256):
        return False
    if (recorded.get("dataset_summary_sha256") != current.get("dataset_summary_sha256")
            or recorded.get("label_policy") != current.get("label_policy")):
        return False
    if not all(is_sha256_hex(item.get("truth_source_sha256")) for item in (recorded, current)):
        return False
    return all(
        item["truth_source_sha256"] == canonical_json_sha256({
            key: item[key] for key in (
                "source_code_sha256", "score_algorithm_sha256",
                "dataset_summary_sha256", "label_policy",
            )
        }) for item in (recorded, current)
    )


def oracle_dataset_summary_sha256(*, root: Path, filter_id: str) -> str | None:
    """Shared readonly witness for the metadata used by daily score eligibility."""
    summary = resolve_dataset_paths(resolve_filter_output_dir(Path(root), filter_id)).summary
    return compute_file_sha256(summary) if summary.is_file() else None


def price_score_replay_lineage(
    *, root: Path, filter_id: str, score_algorithm_sha256: str,
    label_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Attest current price/feature production without constructing any Target.

    Daily-universal samples are derived from the separately verified OHLCV
    inventory; the Dataset summary also supplies their source cutoff/policy.
    A historical score SHA alone cannot certify either the current formula or
    that source metadata. Missing summary evidence has a distinct identity;
    an actual cold RUN still requires the canonical provider to validate it.
    """
    if not is_sha256_hex(score_algorithm_sha256):
        raise ValueError("Oracle price-score algorithm SHA required")
    lineage = {
        "source_code_sha256": oracle_daily_source_fingerprint(),
        "score_algorithm_sha256": score_algorithm_sha256,
        "dataset_summary_sha256": oracle_dataset_summary_sha256(root=root, filter_id=filter_id),
        "label_policy": dict(label_policy),
    }
    return {**lineage, "truth_source_sha256": canonical_json_sha256(lineage)}


def price_score_source_identity(
    *, root: Path, dataset: str, filter_id: str, price_sha256: str,
    providers: Mapping[str, str], recipes: Mapping[str, tuple[str, str]],
    score_start: str, cutoff: str, score_algorithm_sha256: str,
    label_policy: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Verify source file bytes and form a cache identity without scoring rows.

    ``None`` is allowed only if the formal source dataset summary is absent:
    the canonical producer must then validate or fail on its own. This keeps
    isolated synthetic fixtures independent of a real historical Dataset.
    Existing but incompatible evidence is always an error, not a cache miss.
    """
    if not is_sha256_hex(price_sha256) or not is_sha256_hex(score_algorithm_sha256):
        raise ValueError("Oracle price-score identity requires verified source SHA")
    root = Path(root).resolve()
    paths = resolve_dataset_paths(resolve_filter_output_dir(root, filter_id))
    if not paths.summary.is_file():
        return None
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("dataset") != dataset or summary.get("filter_id") != filter_id:
        raise ValueError("Oracle price-score source dataset summary does not match canonical scope")
    # Certify every declared binary input, not merely summary metadata.
    # Some officially registered daily feature providers also consume market
    # set artifacts; an unchanged event-bank SHA alone is not proof of reuse.
    for field, paths_by_name in (
        ("dataset_artifacts", paths.artifact_paths()),
        ("market_set_artifacts", paths.market_set_artifact_paths()),
    ):
        records = summary.get(field)
        if records is None and field == "market_set_artifacts":
            continue
        if not isinstance(records, dict) or set(records) != set(paths_by_name):
            raise ValueError(f"Oracle price-score source has incomplete {field}")
        for name, path in paths_by_name.items():
            stored = records[name]
            if (not isinstance(stored, dict)
                    or not is_sha256_hex(stored.get("sha256"))
                    or not path.is_file()
                    or compute_file_sha256(path) != stored["sha256"]):
                raise ValueError(f"Oracle price-score upstream {field} SHA differs: {name}")
    return {
        "schema": _PRICE_SCORE_CACHE_SCHEMA,
        "dataset": dataset, "filter_id": filter_id,
        "price_sha256": price_sha256,
        "dataset_summary_sha256": compute_file_sha256(paths.summary),
        "source_code_sha256": oracle_daily_source_fingerprint(root),
        "score_algorithm_sha256": score_algorithm_sha256,
        "providers": dict(sorted(providers.items())),
        "recipes": {name: list(spec) for name, spec in sorted(recipes.items())},
        "score_start": score_start, "cutoff": cutoff,
        "label_policy": dict(label_policy),
    }


def verified_existing_price_score_witnesses(
    *, cache_root: Path, current_identity: Mapping[str, Any],
    requested_sources: Mapping[str, tuple[str, str]],
    score_sha: Callable[[Mapping[tuple[str, str], float]], str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Certify archived price-score *values* without rerunning a Target producer.

    Historical price source-code revisions are never accepted on a SHA alias
    alone: the complete archived NPZ matrix must pass the canonical score
    digest, its immutable input/source manifest must match the present price,
    Dataset and approved score-algorithm contract, and its actual scored-key
    membership must be verified. The Replay owner independently checks every
    signed observation and all economic/execution inputs before REUSE.

    A changed score algorithm, provider, price generation, label policy, source
    cohort or missing/corrupt NPZ cannot authorize old economic observations.
    A new unrelated Target/Model never changes existing named source witnesses.
    """
    if not isinstance(current_identity, Mapping):
        return {}
    # Historical source-bridge authority exists only for the independently
    # certified, CURRENT executable price/feature closure. A changed feature
    # transform, data loader, provider, price primitive or policy must not be
    # accepted merely because an archived matrix still hashes to its OLD
    # numerical scores. This boundary applies ONLY to archival migration;
    # ordinary exact-source receipts use the original canonical path.
    if (current_identity.get("source_code_sha256") != _CERTIFIED_LF_PRICE_SOURCE_SHA256
            or current_identity.get("score_algorithm_sha256") !=
                _CERTIFIED_PRICE_ALGORITHM_SHA256):
        return {}
    store = OraclePriceScoreStore(cache_root)
    valid: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(cache_root).glob("price_scores_*.npz")):
        # Read metadata alone first; old algorithm generations never force a
        # 630k x 9 array read just to determine they are irrelevant.
        try:
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"].item()))
            old = meta.get("identity")
            if not isinstance(old, dict) or old.get("schema") != _PRICE_SCORE_CACHE_SCHEMA:
                raise ValueError("archived price-score identity malformed")
            if store._path(old) != path:
                raise ValueError("archived price-score filename does not match identity")
            if not is_sha256_hex(current_identity.get("dataset_summary_sha256")):
                continue
            for field in (
                "schema", "dataset", "filter_id", "price_sha256",
                "dataset_summary_sha256", "score_algorithm_sha256",
                "score_start", "cutoff", "label_policy",
            ):
                if old.get(field) != current_identity.get(field):
                    break
            else:
                old_providers = old.get("providers")
                old_recipes = old.get("recipes")
                current_providers = current_identity.get("providers")
                current_recipes = current_identity.get("recipes")
                if (not all(isinstance(x, Mapping) for x in
                            (old_providers, old_recipes, current_providers, current_recipes))
                        or not is_sha256_hex(old.get("source_code_sha256"))):
                    raise ValueError("archived price-score provider identity malformed")
                relevant = {
                    source: (recipe, provider)
                    for source, (recipe, provider) in requested_sources.items()
                    if list(old_recipes.get(source, ())) == [recipe, provider]
                    and list(current_recipes.get(source, ())) == [recipe, provider]
                    and old_providers.get(provider) == current_providers.get(provider)
                }
                if not relevant:
                    continue
                verified = store.read(old, score_sha=score_sha)
                if verified is None:
                    raise ValueError("archived price-score file disappeared")
                panels, cohort = verified
                if (not all(source in panels for source in relevant)
                        or not all(is_sha256_hex(meta["score_sha256"].get(source))
                                   for source in relevant)):
                    raise ValueError("archived price-score target inventory inconsistent")
                old_lineage = {
                    "source_code_sha256": old["source_code_sha256"],
                    "score_algorithm_sha256": old["score_algorithm_sha256"],
                    "dataset_summary_sha256": old["dataset_summary_sha256"],
                    "label_policy": old["label_policy"],
                }
                legacy_truth_sha = canonical_json_sha256(old_lineage)
                for source in relevant:
                    scores = panels[source]
                    keys = sorted(scores)
                    digest = sha256()
                    for ticker, day in keys:
                        digest.update(f"{ticker}|{day}\n".encode("utf-8"))
                    value = {
                        "score_sha256": meta["score_sha256"][source],
                        "eligible_keys_sha256": digest.hexdigest(),
                        "cohort_rows": len(keys),
                        "first_common_day": min(day for _, day in keys),
                        "last_common_day": max(day for _, day in keys),
                        "source_code_sha256": old["source_code_sha256"],
                        "truth_source_sha256": legacy_truth_sha,
                    }
                    # Different archived score generations with the SAME
                    # declared algorithm and full input contract cannot both
                    # be current truth. Never pick a favorable version.
                    previous = valid.setdefault(source, [])
                    numerical = (
                        "score_sha256", "eligible_keys_sha256", "cohort_rows",
                        "first_common_day", "last_common_day",
                    )
                    if any(tuple(old_value[field] for field in numerical) !=
                           tuple(value[field] for field in numerical)
                           for old_value in previous):
                        raise ValueError("conflicting same-contract historical price-score generations")
                    if value not in previous:
                        previous.append(value)
        except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError,
                zipfile.BadZipFile) as exc:
            raise ValueError(f"Oracle archived price-score provenance corrupt: {path.name}: {exc}") from exc
    return {source: tuple(items) for source, items in valid.items()}


class OraclePriceScoreStore:
    """One immutable score-panel result per certified provider/algorithm input."""

    def __init__(self, cache_root: Path):
        self.cache_root = Path(cache_root)

    def _path(self, identity: Mapping[str, Any]) -> Path:
        if identity.get("schema") != _PRICE_SCORE_CACHE_SCHEMA:
            raise ValueError("Oracle price-score cache schema invalid")
        return self.cache_root / f"price_scores_{canonical_json_sha256(identity)}.npz"

    def read(
        self, identity: Mapping[str, Any],
        *, score_sha: Callable[[Mapping[tuple[str, str], float]], str],
    ) -> tuple[dict[str, dict[tuple[str, str], float]], dict[str, Any]] | None:
        path = self._path(identity)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as obj:
                if set(obj.files) != {"metadata", "tickers", "days", "matrix"}:
                    raise ValueError("Oracle score cache unexpected array fields")
                meta = json.loads(str(obj["metadata"].item()))
                tickers, days, matrix = obj["tickers"], obj["days"], obj["matrix"]
            if (meta.get("schema") != _PRICE_SCORE_CACHE_SCHEMA
                    or meta.get("identity") != dict(identity)):
                raise ValueError("Oracle price-score cache identity mismatch")
            receipt = {field: meta[field] for field in
                       ("schema", "identity", "targets", "cohort", "score_sha256")}
            if canonical_json_sha256(receipt) != meta.get("receipt_sha256"):
                raise ValueError("Oracle price-score evidence receipt hash mismatch")
            targets = meta["targets"]
            if (not isinstance(targets, list) or not targets
                    or not all(isinstance(target, str) for target in targets)
                    or len(set(targets)) != len(targets)
                    or tickers.ndim != 1 or days.ndim != 1 or matrix.ndim != 2
                    or len(tickers) == 0 or len(days) != len(tickers)
                    or matrix.shape != (len(tickers), len(targets))
                    or tickers.dtype.kind != "U" or days.dtype.kind != "U"
                    or matrix.dtype.kind != "f" or not np.isfinite(matrix).all()):
                raise ValueError("Oracle price-score data shape/type/value invalid")
            keys = list(zip(tickers.tolist(), days.tolist()))
            if len(set(keys)) != len(keys):
                raise ValueError("Oracle price-score duplicate stock-day")
            scores = {target: dict(zip(keys, matrix[:, col].tolist()))
                      for col, target in enumerate(targets)}
            if set(scores) != set(meta["score_sha256"]):
                raise ValueError("Oracle price-score target inventory differs")
            if any(score_sha(scores[target]) != meta["score_sha256"][target]
                   for target in targets):
                raise ValueError("Oracle price-score matrix SHA mismatch")
            cohort = meta["cohort"]
            if (cohort.get("cohort_rows") != len(keys)
                    or cohort.get("first_common_day") != min(days)
                    or cohort.get("last_common_day") != max(days)):
                raise ValueError("Oracle price-score cohort evidence mismatch")
            return scores, cohort
        except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError,
                zipfile.BadZipFile) as exc:
            raise ValueError(f"Oracle price-score cache is corrupt; BLOCKED: {path.name}: {exc}") from exc

    def write(
        self, identity: Mapping[str, Any],
        scores: Mapping[str, Mapping[tuple[str, str], float]],
        cohort: Mapping[str, Any],
        *, score_sha: Callable[[Mapping[tuple[str, str], float]], str],
    ) -> None:
        targets = sorted(scores)
        if not targets:
            raise ValueError("Oracle price-score cache requires targets")
        keys = sorted(scores[targets[0]])
        if not keys or any(set(scores[target]) != set(keys) for target in targets):
            raise ValueError("Oracle price-score cache requires one common cohort")
        receipt = {
            "schema": _PRICE_SCORE_CACHE_SCHEMA, "identity": dict(identity),
            "targets": targets, "cohort": dict(cohort),
            "score_sha256": {name: score_sha(scores[name]) for name in targets},
        }
        metadata = json.dumps({**receipt, "receipt_sha256": canonical_json_sha256(receipt)},
                              ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        matrix = np.empty((len(keys), len(targets)), dtype=np.float64)
        for col, target in enumerate(targets):
            matrix[:, col] = np.fromiter(
                (float(scores[target][key]) for key in keys),
                dtype=np.float64, count=len(keys),
            )
        self.cache_root.mkdir(parents=True, exist_ok=True)
        path = self._path(identity)
        fd, temp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=self.cache_root)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, metadata=np.asarray(metadata),
                                    tickers=np.asarray([key[0] for key in keys]),
                                    days=np.asarray([key[1] for key in keys]),
                                    matrix=matrix)
                stream.flush()
                os.fsync(stream.fileno())
            with exclusive_file_lock(path.parent / ".price_scores.publish.lock"):
                existing = self.read(identity, score_sha=score_sha)
                if existing is not None:
                    if existing[0] != {name: dict(values) for name, values in scores.items()} or existing[1] != dict(cohort):
                        raise ValueError("Oracle same-identity price-score source has conflicting outcomes")
                else:
                    atomic_replace_with_retry(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

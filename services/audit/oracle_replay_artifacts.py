"""Canonical per-seed Oracle replay evidence cache (hindsight-only, never PIT).

This module owns execution reuse, not Target, selector or accounting semantics.
Full-identity cache entries survive interrupted audits; historical V1 reports may
be read as observations only after an explicit strict compatibility check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from config.audit import AUDIT_OUTPUT_ROOT
from core.file_integrity import (
    compute_file_sha256, is_sha256_hex, exclusive_file_lock, atomic_replace_with_retry,
    canonical_json_sha256,
)
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs

# AI: V1 accidentally hashed Strategy Compare presentation/membership metadata.
# AI: V2 keys only execution inputs; V1 is retained read-only for verified reuse.
_CACHE_SCHEMA = 2
_LEGACY_CACHE_SCHEMA = 1
_REQUIRED_METRICS = (
    "total_return_pct", "annualized_return_pct", "mdd_pct", "romd", "ev_a_r",
    "trades", "exposure_pct",
)


_MISSING = object()


_REPLAY_COHORT_FIELDS = frozenset((
    "cohort_rows", "first_common_day", "last_common_day",
    "research_cutoff", "risk_budget_return",
    "oracle_parity_implementation", "oracle_only",
))


def _execution_cohort(cohort: Mapping[str, Any]) -> dict[str, Any]:
    """Retain replay-affecting cohort boundaries, not report provider inventories.

    Comparison membership changes `providers`, `market_truth` counters and
    diagnostic counts even when the *actual injected* target score table is
    byte-identical. Such source descriptions are evidence in the report, not
    extra portfolio inputs. The score/eligible-key SHA, target profile, price
    inventory SHA, rolling P16 SHA and economic rules are independently
    included in the target-specific identity. Baseline C96 consumes no score.
    """
    return {key: value for key, value in cohort.items()
            if key in _REPLAY_COHORT_FIELDS}


# Research comparison arm IDs are evidence of which equivalent selector was
# inspected, not computational replay inputs. The actual selector is already
# identified by ranking_policy/options, while the injected stock-day values
# and feature eligibility are certified by the score/keys SHA256 fingerprints.
_REFERENCE_ONLY_SCORE_FIELDS = frozenset((
    "reference_arm_id", "reference_score_source_id", "reference_profile_id",
))


def _execution_score_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in contract.items()
            if key not in _REFERENCE_ONLY_SCORE_FIELDS}


def _execution_replay_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(policy)
    # No dependency may infer or reconstruct the selected model from this key;
    # the formal Exact selection mode and options are recorded independently.
    result.pop("exact_arm", None)
    # These counts describe the already executed label producer; they do not
    # affect the submitted score, stock-day eligibility or Broker-Cash Replay.
    # Historical records legitimately omit them. Keep their original bytes,
    # while excluding the non-executable counters from the execution identity.
    daily = result.get("hindsight_target_contract")
    if isinstance(daily, Mapping):
        daily = dict(daily)
        daily.pop("cutoff_truncated_rows", None)
        daily.pop("research_cutoff_closeout_rows", None)
        result["hindsight_target_contract"] = daily
    score_contract = result.get("canonical_dl_score_contract")
    if isinstance(score_contract, Mapping):
        result["canonical_dl_score_contract"] = _execution_score_contract(score_contract)
    return result


def _v2_presentation_identity(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Verify old V2 presentation-only drift without rewriting historical bytes.

    The old snapshot carried the first enabled Exact arm ID and its model
    reference names in the replay identity. Changing which equivalent model
    is enabled must not invalidate the very same executed score/selector.
    Actual target source, score values and stock-day eligibility SHA, dates,
    parameter/price bytes and all other economic inputs remain strict.
    """
    result = dict(stored)
    if result.get("cache_schema") != _CACHE_SCHEMA or expected.get("cache_schema") != _CACHE_SCHEMA:
        return result
    if result.get("target") != expected.get("target"):
        return result
    previous_policy = result.get("replay_policy")
    wanted_policy = expected.get("replay_policy")
    if not isinstance(previous_policy, Mapping) or not isinstance(wanted_policy, Mapping):
        return result
    if result.get("target") != "C96":
        score_sha = wanted_policy.get("perfect_score_sha256")
        previous_contract = previous_policy.get("canonical_dl_score_contract")
        wanted_contract = wanted_policy.get("canonical_dl_score_contract")
        if (not is_sha256_hex(score_sha)
                or previous_policy.get("perfect_score_sha256") != score_sha
                or not isinstance(previous_contract, Mapping)
                or not isinstance(wanted_contract, Mapping)
                or _execution_score_contract(previous_contract) !=
                        _execution_score_contract(wanted_contract)):
            return result
    # This transform may only erase fields already proven non-executable;
    # the complete normalized identity must still match in _inspect_siblings.
    result["replay_policy"] = _execution_replay_policy(previous_policy)
    if isinstance(result.get("target_definition"), str):
        result["target_definition"] = expected.get("target_definition")
    if isinstance(result.get("cohort"), Mapping):
        result["cohort"] = _execution_cohort(result["cohort"])
    return result


def _changed_identity_fields(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> tuple[str, ...]:
    """Explain strict scientific identity inequality without relaxing the match."""
    changes: list[str] = []

    def visit(previous: Any, current: Any, field: str) -> None:
        if isinstance(previous, dict) and isinstance(current, dict):
            for key in sorted(previous.keys() | current.keys(), key=str):
                visit(previous.get(key, _MISSING), current.get(key, _MISSING),
                      f"{field}.{key}" if field else str(key))
        elif previous != current:
            changes.append(field)

    visit(dict(stored), dict(expected), "")
    return tuple(changes)


def _identity_field_value(identity: Mapping[str, Any], field: str) -> str:
    value: Any = identity
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            return "<absent>"
        value = value[key]
    if isinstance(value, str) and is_sha256_hex(value):
        return value[:16] + "..."
    if isinstance(value, (list, dict)):
        return f"<{type(value).__name__}>"
    printable = repr(value)
    return printable if len(printable) <= 100 else printable[:97] + "..."


def price_inventory_sha256(project_root: Path, dataset: str) -> str:
    """Hash real Research OHLCV *bytes* (not timestamps or file existence)."""
    sources, duplicates = discover_dataset_csv_inputs(Path(project_root), str(dataset))
    if duplicates or not sources:
        raise ValueError("Oracle price inventory cannot be certified for reuse")
    digest = hashlib.sha256()
    for ticker, source in sorted(sources, key=lambda entry: (str(entry[0]), str(entry[1]))):
        path = Path(source)
        if not path.is_file():
            raise ValueError("Oracle price inventory file missing")
        digest.update(str(ticker).encode("utf-8") + b"\x00")
        digest.update(compute_file_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def replay_identity(*, dataset: str, dataset_sha256: str, target: str,
                    definition: Mapping[str, Any],
                    start: str, end: str, benchmark: str, param_sha256: str,
                    seed: int, cohort: Mapping[str, Any],
                    ranking_policy: str, ranking_options: Mapping[str, Any],
                    evaluation_mode: str, max_positions: int,
                    replay_policy: Mapping[str, Any]) -> dict[str, Any]:
    """Replay identity is independent of report version / output path / timestamp."""
    if not is_sha256_hex(dataset_sha256) or not is_sha256_hex(param_sha256):
        raise ValueError("Oracle dataset/parameter SHA256 provenance missing")
    if not target or target not in definition:
        raise ValueError("Oracle target identity missing")
    return {
        "cache_schema": _CACHE_SCHEMA,
        "science": "target_ground_truth_exact_broker_cash_v1",
        "target": str(target),
        # The immutable target/arm name, not editable report explanation,
        # identifies the score treatment. Actual perfect-score and source
        # fingerprints are embedded in replay_policy.
        "target_definition": str(target),
        "dataset": str(dataset),
        "dataset_sha256": str(dataset_sha256),
        "benchmark": str(benchmark),
        "param_sha256": str(param_sha256),
        "seed": int(seed),
        "start_date": str(start),
        "end_date": str(end),
        "max_positions": int(max_positions),
        "rotation": "off",
        "evaluation_mode": str(evaluation_mode),
        "ranking_policy": str(ranking_policy),
        "ranking_options": dict(ranking_options),
        "replay_policy": _execution_replay_policy(replay_policy),
        "cohort": _execution_cohort(cohort),
    }


def _validate_observation(observation: Mapping[str, Any], *, target: str, seed: int) -> dict[str, Any]:
    row = dict(observation)
    if row.get("target") != target or int(row.get("seed", -1)) != int(seed):
        raise ValueError("Oracle cached observation has mismatched target / seed")
    for field in _REQUIRED_METRICS:
        value = float(row[field])
        if not math.isfinite(value):
            raise ValueError("Oracle cached observation has nonfinite metric")
    if int(row["trades"]) < 0 or not isinstance(row.get("yearly"), list) or not row["yearly"]:
        raise ValueError("Oracle cached observation lacks completed yearly evidence")
    for item in row["yearly"]:
        if not isinstance(item, dict) or not item.get("year") or not math.isfinite(float(item["strategy_return_pct"])):
            raise ValueError("Oracle cached yearly evidence incomplete")
    if target != "C96" and not math.isfinite(float(row["delta_romd_vs_c96"])):
        raise ValueError("Oracle cached contrast is incomplete")
    return row


def oracle_cohorts_compatible(saved: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Certify an old *superset provider* join was sample-neutral.

    Dropping an unused truth source cannot be proven safe merely because it no
    longer supplies a score. The historical intersection is equal to the
    retained source ONLY if the old market evidence proves that each retained
    source had exactly the same number of rows as the joined cohort. With
    matching prices/profile identities, this is a set-theoretic equivalence
    proof (the historical intersection is a subset of each source).

    Missing row-count evidence fails closed. No old cache is modified.
    """
    if saved == current:
        return True
    if not isinstance(saved, Mapping) or not isinstance(current, Mapping):
        return False
    old_providers, new_providers = saved.get("providers"), current.get("providers")
    old_market, new_market = saved.get("market_truth"), current.get("market_truth")
    old_extra, new_extra = saved.get("rows_outside_dl_training_provider_by_source"), current.get("rows_outside_dl_training_provider_by_source")
    if not all(isinstance(x, Mapping) for x in
               (old_providers, new_providers, old_market, new_market, old_extra, new_extra)):
        return False
    if (not new_providers or not set(new_providers).issubset(old_providers)
            or any(old_providers[key] != value for key, value in new_providers.items())
            or len(new_providers) >= len(old_providers)):
        return False
    for key, value in current.items():
        if key in {"providers", "market_truth", "rows_outside_dl_training_provider_by_source"}:
            continue
        if saved.get(key, _MISSING) != value:
            return False
    if set(saved) - set(current):
        return False
    old_counts = old_market.get("mature_rows_by_provider")
    new_counts = new_market.get("mature_rows_by_provider")
    if not isinstance(old_counts, Mapping) or not isinstance(new_counts, Mapping):
        return False
    if any(old_market.get(key, _MISSING) != value
           for key, value in new_market.items() if key != "mature_rows_by_provider"):
        return False
    if set(old_market) != set(new_market):
        return False
    if not set(new_counts).issubset(old_counts):
        return False
    n = current.get("cohort_rows")
    if not isinstance(n, int) or n <= 0 or saved.get("cohort_rows") != n:
        return False
    if any(old_counts[key] != n or new_counts[key] != n
           for key in new_counts):
        return False
    if (not set(new_extra).issubset(old_extra)
            or any(old_extra[key] != value for key, value in new_extra.items())):
        return False
    return True


@dataclass(frozen=True)
class OracleReplayDecision:
    """One fully certified replay identity; never a file-presence guess."""

    action: str
    identity: Mapping[str, Any]
    observation: Mapping[str, Any] | None
    origin: str
    reason: str = ""


class OracleReplayExecutionPlan:
    """One execution owner: REUSE cannot invoke any scientific producer."""

    def __init__(self, store: "OracleReplayStore", decisions: Mapping[tuple[int, str], OracleReplayDecision]):
        self._store = store
        self._decisions = dict(decisions)

    def action(self, seed: int, target: str) -> str:
        return self._decisions[(int(seed), str(target))].action

    def actions(self) -> dict[tuple[int, str], str]:
        return {key: row.action for key, row in self._decisions.items()}

    def blocked(self) -> dict[tuple[int, str], str]:
        return {key: row.reason for key, row in self._decisions.items()
                if row.action == "BLOCKED"}

    def require_ready(self) -> None:
        if self.blocked():
            cases = [f"{target} seed={seed}: {reason}"
                     for (seed, target), reason in self.blocked().items()]
            raise ValueError(f"Oracle full-identity BLOCKED: {len(cases)} conflict(s); "
                             + " | ".join(cases[:3]))

    def execute(self, seed: int, target: str,
                producer: Callable[[], Mapping[str, Any]]) -> tuple[dict[str, Any], str]:
        """Only an explicit RUN may call producer and persist its result."""
        self.require_ready()
        row = self._decisions[(int(seed), str(target))]
        if row.action == "REUSE":
            if row.observation is None:
                raise ValueError("Oracle REUSE missing completed observation")
            if row.origin.startswith("verified legacy replay cache"):
                # Verify the alias again before publishing the canonical key.
                # This is metadata promotion, NEVER a portfolio computation.
                refreshed, origin = self._store.resolve_completed(row.identity)
                if refreshed != row.observation or origin != row.origin:
                    raise ValueError("Oracle historical replay changed after preflight")
                self._store.write(row.identity, refreshed, origin="verified_legacy_replay_cache")
            return dict(row.observation), row.origin
        if row.action != "RUN" or row.observation is not None:
            raise ValueError("Invalid canonical Replay execution plan")
        result = dict(producer())
        self._store.write(row.identity, result, origin="canonical_replay")
        return result, "newly completed"


class OracleReplayStore:
    """Durable per-arm cache; no producer-side recomputation on a cache hit."""

    def __init__(self, project_root: Path, output_subdir: str):
        root = Path(project_root).resolve()
        self.project_root = root
        self.cache_root = root / AUDIT_OUTPUT_ROOT / Path(output_subdir) / "replay_cache"

    def _path(self, identity: Mapping[str, Any]) -> Path:
        seed = int(identity["seed"])
        target = str(identity["target"])
        # Public Oracle target labels may contain a hyphen (e.g. T40-MFE).
        # Only single filename-safe labels are admitted; path traversal and
        # separators remain invalid, without maintaining a target allowlist.
        if not target.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Oracle invalid cache target")
        return self.cache_root / str(seed) / f"{target}_{canonical_json_sha256(identity)}.json"

    @staticmethod
    def _legacy_semantic_identity(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any] | None:
        """Upgrade *only* the old redundant settings digest, never replay semantics.

        In V1, that SHA included display names, enabled-in-compare flags and
        parameter-source catalog details.  All replay-effective facts were
        also recorded separately (price/param SHA, execution and score policies,
        exact target, seed, dates, cohort and economic defaults).  Require an
        exact V2 field inventory and value match; no historical settings hash
        is treated as authority for a new execution policy.
        """
        if stored.get("cache_schema") != _LEGACY_CACHE_SCHEMA:
            return None
        if (set(stored) != (set(expected) | {"strategy_settings_fingerprint"})
                or not is_sha256_hex(stored.get("strategy_settings_fingerprint"))):
            return None
        normalized = dict(stored)
        normalized.pop("strategy_settings_fingerprint")
        normalized["cache_schema"] = _CACHE_SCHEMA
        return normalized

    def _inspect_siblings(
        self, identity: Mapping[str, Any]
    ) -> tuple[tuple[dict[str, Any], str] | None, str | None]:
        """Single strict resolver for legacy-equivalent and conflicting caches.

        The legacy settings SHA is non-executable metadata; all other fields
        must match exactly.  Never silently pick between inconsistent completed
        observations, and never remove or overwrite historical evidence.
        """
        path = self._path(identity)
        requested_cohort = identity.get("cohort") or {}
        if not isinstance(requested_cohort, dict):
            raise ValueError("Oracle requested replay cohort is malformed")
        conflicts: list[tuple[Path, dict[str, Any]]] = []
        aliases: list[tuple[Path, dict[str, Any]]] = []
        for candidate in sorted(path.parent.glob(f"{identity['target']}_*.json")):
            if candidate == path:
                continue
            # A filename prefix is NOT an Oracle Target identity.  R40_* also
            # matches R40_Direct_<sha>.  Do not open an unrelated file at all:
            # even a corrupt sibling from another arm must not block this arm.
            suffix = candidate.stem[len(str(identity["target"])) + 1:]
            if not is_sha256_hex(suffix):
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                peer = payload["identity"]
                if (not isinstance(peer, dict)
                        or peer.get("cache_schema") not in (_LEGACY_CACHE_SCHEMA, _CACHE_SCHEMA)
                        or payload.get("cache_schema") != peer.get("cache_schema")
                        or int(peer.get("seed", -1)) != int(identity["seed"])
                        or payload.get("identity_sha256") != canonical_json_sha256(peer)
                        or self._path(peer) != candidate):
                    raise ValueError("Oracle sibling replay cache identity malformed")
                # R40_*.json also matches R40_Direct_*.json; they are distinct.
                if peer.get("target") != identity["target"]:
                    continue
                peer_cohort = peer.get("cohort") or {}
                if not isinstance(peer_cohort, dict):
                    raise ValueError("Oracle sibling replay cache cohort malformed")
                record = self.read(peer)
                if record is None:
                    raise ValueError("Oracle sibling replay cache is missing")
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                raise ValueError(
                    f"Oracle sibling replay cache is corrupt; fail closed: "
                    f"{candidate.relative_to(self.project_root).as_posix()}"
                ) from exc
            # An implementation-only Oracle parity bug fix invalidates all
            # earlier hindsight score replays. Keep the old bytes as history,
            # but do not let their incompatible same-window identities block
            # the corrected result from being computed alongside them.
            # This exemption never applies to C96 (the unscored baseline) or
            # to two new-revision observations with differing semantics.
            requested_policy = identity.get("replay_policy") or {}
            peer_policy = peer.get("replay_policy") or {}
            if (isinstance(requested_policy, Mapping)
                    and isinstance(peer_policy, Mapping)
                    and (requested_policy.get("period_stats_implementation")
                         != peer_policy.get("period_stats_implementation")
                         or requested_policy.get("oracle_parity_implementation")
                         != peer_policy.get("oracle_parity_implementation"))):
                continue
            # Independently attested Target-outcome producers can coexist in
            # the same evaluation window. An earlier receipt with no current
            # producer proof (or a different proof) remains intact as history,
            # but cannot block the authorized corrected RUN. This is a generic
            # source capability, NOT a 40D/60D/name-specific exception.
            wanted_truth = requested_policy.get("hindsight_target_contract") if isinstance(
                requested_policy, Mapping) else None
            previous_truth = peer_policy.get("hindsight_target_contract") if isinstance(
                peer_policy, Mapping) else None
            if isinstance(wanted_truth, Mapping) and is_sha256_hex(
                    wanted_truth.get("truth_source_sha256")) and (
                    not isinstance(previous_truth, Mapping) or
                    previous_truth.get("truth_source_sha256") != wanted_truth["truth_source_sha256"]):
                continue
            if (peer.get("dataset") == identity["dataset"]
                    and peer.get("start_date") == identity["start_date"]
                    and peer.get("end_date") == identity["end_date"]):
                # Comparison-inventory providers may differ; strict execution
                # identity below (including actual score SHA) is authoritative.
                candidate_identity = self._legacy_semantic_identity(peer, identity)
                if candidate_identity is None:
                    candidate_identity = dict(peer)
                if peer.get("cache_schema") == _CACHE_SCHEMA:
                    candidate_identity = _v2_presentation_identity(candidate_identity, identity)
                if oracle_cohorts_compatible(candidate_identity.get("cohort") or {}, requested_cohort):
                    candidate_identity = dict(candidate_identity, cohort=dict(requested_cohort))
                if candidate_identity == dict(identity):
                    aliases.append((candidate, record))
                else:
                    conflicts.append((candidate, peer))
        if conflicts:
            examples: dict[str, str] = {}
            for _candidate, peer in conflicts:
                normalized = self._legacy_semantic_identity(peer, identity) or peer
                if peer.get("cache_schema") == _CACHE_SCHEMA:
                    normalized = _v2_presentation_identity(normalized, identity)
                for field in _changed_identity_fields(normalized, identity):
                    examples.setdefault(
                        field, f"{field} [cache={_identity_field_value(normalized, field)}, "
                               f"current={_identity_field_value(identity, field)}]",
                    )
            sample_path = conflicts[0][0].relative_to(self.project_root).as_posix()
            return None, (f"{len(conflicts)} conflicting cache file(s); example={sample_path}; "
                          f"changed fields: {'; '.join(examples[name] for name in sorted(examples))}")
        if aliases:
            if len({canonical_json_sha256(record) for _path, record in aliases}) != 1:
                sample_path = aliases[0][0].relative_to(self.project_root).as_posix()
                raise ValueError(f"Oracle legacy-equivalent replay caches disagree on observations: {sample_path}")
            source = aliases[0][0].relative_to(self.project_root).as_posix()
            return (aliases[0][1], source), None
        return None, None

    def resolve_completed(self, identity: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
        """Canonical current cache, or fully verified legacy semantic alias."""
        current = self.read(identity)
        if current is not None:
            return current, "verified per-arm cache"
        legacy, conflict = self._inspect_siblings(identity)
        if conflict is not None:
            raise ValueError(f"same-window replay_cache has a DIFFERENT identity for "
                             f"{identity['target']} seed={identity['seed']}; {conflict}; "
                             f"replay remains BLOCKED until provenance is verified")
        if legacy is not None:
            record, source = legacy
            return record, f"verified legacy replay cache ({source})"
        return None, "none"

    def plan_execution(self, identities: Mapping[tuple[int, str], Mapping[str, Any]]) -> OracleReplayExecutionPlan:
        """Certify every arm once before running any Replay, with no guesswork."""
        if not identities:
            raise ValueError("Oracle empty replay comparison plan")
        result: dict[tuple[int, str], OracleReplayDecision] = {}
        for (seed, target), identity in identities.items():
            key = (int(seed), str(target))
            if identity.get("seed") != key[0] or identity.get("target") != key[1]:
                raise ValueError("Oracle plan key and scientific identity differ")
            try:
                observation, origin = self.resolve_completed(identity)
                result[key] = OracleReplayDecision(
                    "REUSE" if observation is not None else "RUN",
                    dict(identity), observation, origin,
                )
            except ValueError as exc:
                result[key] = OracleReplayDecision(
                    "BLOCKED", dict(identity), None, "none", str(exc),
                )
        return OracleReplayExecutionPlan(self, result)

    def explain_other_identity(self, identity: Mapping[str, Any]) -> str | None:
        """Explain genuine replay differences; legacy UI-only drift is compatible."""
        _legacy, conflict = self._inspect_siblings(identity)
        return conflict

    def has_other_identity(self, identity: Mapping[str, Any]) -> bool:
        """Compatibility probe; explanation and matching have a single owner."""
        return self.explain_other_identity(identity) is not None

    def preview_candidates(
        self, *, seeds, targets, dataset: str | None = None,
        start_date: str | None = None, requested_end: str | None = None,
        benchmark: str | None = None,
        required_oracle_revision: str | None = None,
    ) -> dict[tuple[int, str], str]:
        """Inventory validated local bytes without certifying scientific REUSE.

        The complete replay identity, including parameter/price/cohort hashes,
        is intentionally checked later by ``resolve_completed``. A corrupt
        candidate is never presented as a reusable artifact.
        """
        result: dict[tuple[int, str], str] = {}
        for seed in seeds:
            for target in targets:
                key = (int(seed), str(target))
                try:
                    identities = self.existing_completed_identities(seed=key[0], target=key[1])
                except (OSError, ValueError, KeyError, TypeError, OverflowError):
                    result[key] = "INVALID"
                    continue
                matching = [identity for identity in identities if (
                    (target == "C96" or required_oracle_revision is None
                     or (identity.get("replay_policy") or {}).get("oracle_parity_implementation")
                        == required_oracle_revision)
                    and (dataset is None or identity.get("dataset") == dataset)
                    and (start_date is None or identity.get("start_date") == start_date)
                    and (requested_end is None or
                         (identity.get("cohort") or {}).get("research_cutoff") == requested_end)
                    and (benchmark is None or identity.get("benchmark") == benchmark)
                )]
                result[key] = (
                    "MULTIPLE" if len(matching) > 1 else
                    "CACHE" if matching else "MISSING"
                )
        return result

    def existing_completed_identities(self, *, seed: int, target: str) -> tuple[dict[str, Any], ...]:
        """Inventory *verified stored bytes*, not current scientific REUSE eligibility.

        A preview may use this to show existing completion candidates without
        materializing expensive future labels. Final RUN/REUSE/BLOCKED decisions
        must still be made by resolve_completed with the FULL replay identity.
        """
        if not target.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Oracle invalid cache target")
        results: list[dict[str, Any]] = []
        for path in sorted((self.cache_root / str(int(seed))).glob(f"{target}_*.json")):
            # A prefix like R40_ also matches R40_Direct_ in a glob; only
            # accept the canonical <target>_<SHA256>.json filename shape.
            if not is_sha256_hex(path.stem[len(target) + 1:]):
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                identity = raw["identity"]
                if not isinstance(identity, dict):
                    raise ValueError("Oracle cached identity is not a mapping")
                # A shorter target prefix may also match its Direct counterpart.
                if identity.get("target") != target:
                    continue
                if self._path(identity) != path or self.read(identity) is None:
                    raise ValueError("Oracle cache name or completed observation is invalid")
            except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Oracle stored replay cache inventory is corrupt: "
                    + path.relative_to(self.project_root).as_posix()
                ) from exc
            results.append(identity)
        return tuple(results)

    def read(self, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._path(identity)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload.get("cache_schema") != identity.get("cache_schema")
                    or identity.get("cache_schema") not in (_LEGACY_CACHE_SCHEMA, _CACHE_SCHEMA)
                    or payload.get("identity") != dict(identity)
                    or payload.get("identity_sha256") != canonical_json_sha256(identity)):
                raise ValueError("Oracle cache identity mismatch")
            observation = _validate_observation(payload["observation"], target=identity["target"], seed=identity["seed"])
            if payload.get("observation_sha256") != canonical_json_sha256(observation):
                raise ValueError("Oracle cached observation SHA256 mismatch")
            return observation
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Oracle existing replay cache is corrupt or conflicting; fail closed") from exc

    def write(self, identity: Mapping[str, Any], observation: Mapping[str, Any], *, origin: str) -> None:
        if identity.get("cache_schema") != _CACHE_SCHEMA:
            raise ValueError("Oracle historical replay cache is read-only")
        record = _validate_observation(observation, target=identity["target"], seed=identity["seed"])
        path = self._path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The same-identity lock and atomic replacement avoid partial reads and
        # concurrent writers fabricating a completed result.
        with exclusive_file_lock(path.with_name("." + path.name + ".lock")):
            existing = self.read(identity)
            if existing is not None:
                if canonical_json_sha256(existing) != canonical_json_sha256(record):
                    raise ValueError("Oracle deterministic replay cache conflict")
                return
            payload = {
                "cache_schema": int(identity["cache_schema"]),
                "identity": dict(identity),
                "identity_sha256": canonical_json_sha256(identity),
                "observation": record,
                "observation_sha256": canonical_json_sha256(record),
                "origin": str(origin),
            }
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                atomic_replace_with_retry(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


__all__ = [
    "OracleReplayStore", "price_inventory_sha256", "replay_identity",
    "oracle_cohorts_compatible", "OracleReplayDecision", "OracleReplayExecutionPlan",
]

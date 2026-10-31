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
import zipfile
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from config.audit import AUDIT_OUTPUT_ROOT
from core.file_integrity import (
    compute_file_sha256, is_sha256_hex, exclusive_file_lock, atomic_replace_with_retry,
    canonical_json_sha256,
)
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs
from core.training_policy import benchmark_seed_execution_equivalent

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
        if isinstance(daily.get("teacher"), Mapping):
            from services.audit.daily_trade40_oracle import truth_producer_fingerprint
            from services.audit.mintrade_truth_source_attestation import mintrade_sources_equivalent
            current = truth_producer_fingerprint()
            if mintrade_sources_equivalent(daily.get("truth_source_sha256"), current):
                daily["truth_source_sha256"] = current
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
    # The only additional permitted transform is the separately attested
    # LF/CRLF-only price producer revision. Signed old cache bytes remain
    # unchanged, and every other producer/execution input stays strict.
    normalized_policy = _execution_replay_policy(previous_policy)
    from services.audit.oracle_score_artifacts import certified_price_lineages_equivalent
    if certified_price_lineages_equivalent(
        previous_policy.get("hindsight_target_contract"),
        wanted_policy.get("hindsight_target_contract"),
    ):
        normalized_policy["hindsight_target_contract"] = dict(
            wanted_policy["hindsight_target_contract"])
    result["replay_policy"] = normalized_policy
    if isinstance(result.get("target_definition"), str):
        result["target_definition"] = expected.get("target_definition")
    if isinstance(result.get("cohort"), Mapping):
        result["cohort"] = _execution_cohort(result["cohort"])
    return result


def _same_producer_different_score(previous: Mapping[str, Any],
                                   expected: Mapping[str, Any]) -> bool:
    """Distinguish impossible same-source score drift from a different RUN identity.

    A score or raw/rank digest changing with *identical* price, P16,
    eligibility, code lineage, Target contract and Replay inputs is a
    determinism conflict. Different executable inputs/revisions are new
    independent cache keys: preserve the old result, compute the new one.
    This capability applies to any scored Target, without arm/period lists.
    """
    prev_policy = previous.get("replay_policy")
    wanted_policy = expected.get("replay_policy")
    if not isinstance(prev_policy, Mapping) or not isinstance(wanted_policy, Mapping):
        return False
    previous_lineage = prev_policy.get("hindsight_target_contract")
    wanted_lineage = wanted_policy.get("hindsight_target_contract")
    signed_scores = (is_sha256_hex(prev_policy.get("perfect_score_sha256")) and
                     is_sha256_hex(wanted_policy.get("perfect_score_sha256")))
    signed_outcome = (isinstance(previous_lineage, Mapping) and
                      isinstance(wanted_lineage, Mapping) and
                      is_sha256_hex(previous_lineage.get("truth_source_sha256")) and
                      previous_lineage.get("truth_source_sha256") ==
                          wanted_lineage.get("truth_source_sha256"))
    if not (signed_scores or signed_outcome):
        # Unscored controls have no Target-result digest to disagree on.
        return False
    def source_only(value: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(value)
        # Old signed V2 receipts also carried presentation descriptions and
        # reference-only model IDs. These are not changes to deterministic
        # score production; normalize *before* testing impossible score drift.
        result["target_definition"] = result.get("target")
        cohort = result.get("cohort")
        if isinstance(cohort, Mapping):
            result["cohort"] = _execution_cohort(cohort)
        policy = _execution_replay_policy(dict(result.get("replay_policy") or {}))
        policy.pop("perfect_score_sha256", None)
        lineage = policy.get("hindsight_target_contract")
        if isinstance(lineage, Mapping):
            lineage = dict(lineage)
            # These are producer *results*, never source inputs. Source-code
            # lineage, horizon, teacher, policy and cutoff must still match.
            for name in ("rank_sha256", "raw_label_sha256"):
                lineage.pop(name, None)
            policy["hindsight_target_contract"] = lineage
        result["replay_policy"] = policy
        return result
    return (source_only(previous) == source_only(expected) and
            dict(previous) != dict(expected))


def _independent_source_revision(previous: Mapping[str, Any],
                                 expected: Mapping[str, Any]) -> bool:
    """Approved implementation/producer corrections retain a separate old result."""
    previous_policy = previous.get("replay_policy") or {}
    wanted_policy = expected.get("replay_policy") or {}
    if not isinstance(previous_policy, Mapping) or not isinstance(wanted_policy, Mapping):
        return False
    if any(wanted_policy.get(field) != previous_policy.get(field) for field in
           ("period_stats_implementation", "oracle_parity_implementation")):
        return True
    wanted_truth = wanted_policy.get("hindsight_target_contract")
    previous_truth = previous_policy.get("hindsight_target_contract")
    from services.audit.oracle_score_artifacts import certified_price_lineages_equivalent
    if certified_price_lineages_equivalent(previous_truth, wanted_truth):
        return False
    if isinstance(previous_truth, Mapping) and isinstance(wanted_truth, Mapping):
        from services.audit.mintrade_truth_source_attestation import mintrade_sources_equivalent
        if (isinstance(wanted_truth.get("teacher"), Mapping) and mintrade_sources_equivalent(
                previous_truth.get("truth_source_sha256"), wanted_truth.get("truth_source_sha256"))):
            return False
    return (isinstance(wanted_truth, Mapping)
            and is_sha256_hex(wanted_truth.get("truth_source_sha256"))
            and (not isinstance(previous_truth, Mapping)
                 or previous_truth.get("truth_source_sha256") != wanted_truth["truth_source_sha256"]))


def _known_input_differences(stored: Mapping[str, Any],
                             expected: Mapping[str, Any]) -> tuple[str, ...]:
    """Compare only verified inputs; absent outcome/key hashes are not invented."""
    differences: list[str] = []

    def visit(previous: Any, current: Any, path: str) -> None:
        if isinstance(current, Mapping) and isinstance(previous, Mapping):
            for key, value in current.items():
                visit(previous.get(key, _MISSING), value,
                      f"{path}.{key}" if path else str(key))
        elif previous != current:
            differences.append(path)

    visit(stored, expected, "")
    return tuple(differences)


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


def verified_daily_truth_witness(
    *, cache_root: Path, expected_inputs: Mapping[str, Any],
    score_sha: Callable[[Mapping[tuple[str, str], float]], str],
) -> dict[str, Any] | None:
    """Read an already published current-source truth; never call its Producer.

    Full NPZ values, key membership and immutable producer receipts are checked
    by the truth owner. This allows a completed current truth cache to certify
    an old Portfolio *before* the REUSE/RUN matrix, without feature loading,
    single-stock simulation, score rebuilding or Portfolio execution.
    """
    import numpy as np
    from services.audit.daily_trade40_oracle import _truth_cache_read
    from services.audit.mintrade_truth_source_attestation import mintrade_sources_equivalent

    target = str(expected_inputs["target_id"])
    matches = []
    for path in sorted((Path(cache_root) / str(expected_inputs["seed"])).glob(f"{target}_*.npz")):
        if not is_sha256_hex(path.stem[len(target) + 1:]):
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                identity = json.loads(str(archive["metadata"].item()))["identity"]
            if (not isinstance(identity, dict) or
                    path.stem != f"{target}_{canonical_json_sha256(identity)}"):
                raise ValueError("daily truth cache filename/identity mismatch")
            source_sha = identity.get("truth_source_sha256")
            current_sha = expected_inputs["truth_source_sha256"]
            if (not mintrade_sources_equivalent(source_sha, current_sha)
                    or any(identity.get(key, _MISSING) != value
                           for key, value in expected_inputs.items()
                           if key != "truth_source_sha256")):
                continue
            if set(identity) != set(expected_inputs) | {"feature_eligible_keys_sha256"}:
                continue
            cached = _truth_cache_read(path, identity)
            if cached is None:
                raise ValueError("daily truth cache disappeared during preflight")
            scores, evidence = cached
            ordered = sorted(scores)
            key_digest = hashlib.sha256()
            group_digest = hashlib.sha256(b"ticker,date\n")
            windows_group_digest = hashlib.sha256(b"ticker,date\r\n")
            for ticker, day in ordered:
                key_digest.update(f"{ticker}|{day}\n".encode("utf-8"))
                group_digest.update(f"{ticker},{day}\n".encode("utf-8"))
                windows_group_digest.update(f"{ticker},{day}\r\n".encode("utf-8"))
            if identity["feature_eligible_keys_sha256"] not in {
                    group_digest.hexdigest(), windows_group_digest.hexdigest()}:
                raise ValueError("daily truth feature-key identity differs from actual NPZ keys")
            days = [day for _, day in ordered]
            if (evidence.get("first_date") != min(days)
                    or evidence.get("last_date") != max(days)):
                raise ValueError("daily truth date bounds differ from actual NPZ keys")
            current_evidence = {**evidence, "truth_source_sha256": current_sha}
            current_evidence.pop("source_reuse_status", None)
            matches.append({
                "evidence": current_evidence,
                "score_sha256": score_sha(scores),
                "keys": (key_digest.hexdigest(), len(scores), min(days), max(days)),
            })
        except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Oracle existing daily truth witness is invalid: {path.name}: {exc}") from exc
    if not matches:
        return None
    if any(item != matches[0] for item in matches[1:]):
        raise ValueError("Oracle current-source truth caches disagree on identical inputs")
    return matches[0]


def price_inventory_sha256(project_root: Path, dataset: str) -> str:
    """Reuse a published content-verified price generation; otherwise hash bytes.

    The canonical source owner can certify one pre-existing Dataset generation
    once when the original Dataset summary/metadata matches. Subsequent Oracle
    plans read only the content-attested source proof and metadata. This does
    not build a Dataset, Target or Portfolio, nor alter signed Replay receipts.
    Source-only fixtures without a canonical Dataset retain the strict legacy
    byte-hashing path.
    """
    from filters.breakout_quality.source_inventory import (
        read_published_price_inventory, certify_existing_dataset_price_generation,
    )
    published = read_published_price_inventory(project_root, dataset)
    if published is not None:
        return published
    bootstrapped = certify_existing_dataset_price_generation(project_root, dataset)
    if bootstrapped is not None:
        return bootstrapped
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
    if "win_rate_pct" in row:
        win_rate = float(row["win_rate_pct"])
        if not math.isfinite(win_rate) or not 0.0 <= win_rate <= 100.0:
            raise ValueError("Oracle cached win rate must be a finite percentage in [0, 100]")
    if target != "C96" and not math.isfinite(float(row["delta_romd_vs_c96"])):
        raise ValueError("Oracle cached contrast is incomplete")
    return row


def _same_replay_observation(saved: Any, rerun: Any) -> bool:
    """An extra metric may not revise any existing historical replay observation."""
    if isinstance(saved, Mapping) and isinstance(rerun, Mapping):
        return (set(saved) == set(rerun) and
                all(_same_replay_observation(saved[key], rerun[key]) for key in saved))
    if isinstance(saved, list) and isinstance(rerun, list):
        return (len(saved) == len(rerun) and
                all(_same_replay_observation(a, b) for a, b in zip(saved, rerun)))
    if isinstance(saved, bool) or isinstance(rerun, bool):
        return type(saved) is type(rerun) and saved == rerun
    if isinstance(saved, (int, float)) and isinstance(rerun, (int, float)):
        return math.isfinite(float(saved)) and math.isfinite(float(rerun)) and math.isclose(
            float(saved), float(rerun), rel_tol=1e-10, abs_tol=1e-10)
    return saved == rerun


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

    def __init__(self, store: "OracleReplayStore", decisions: Mapping[tuple[int, str], OracleReplayDecision],
                 *, allow_historical_execution_input_drift: bool = False):
        self._store = store
        self._decisions = dict(decisions)
        self._allow_historical_execution_input_drift = allow_historical_execution_input_drift

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
                refreshed, origin = self._store.resolve_completed(
                    row.identity,
                    allow_historical_execution_input_drift=self._allow_historical_execution_input_drift,
                )
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

    def _win_rate_path(self, identity: Mapping[str, Any]) -> Path:
        """Independent supplemental metric evidence; the original replay receipt is immutable."""
        replay = self._path(identity)
        return self.cache_root.parent / "win_rate_evidence" / str(identity["seed"]) / replay.name

    def with_win_rate(self, identity: Mapping[str, Any],
                      observation: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return a complete observation or None when an old receipt needs metric replay.

        Missing evidence is not a zero win rate and is never reported as REUSE.
        """
        row = _validate_observation(observation, target=identity["target"], seed=identity["seed"])
        if "win_rate_pct" in row:
            return row
        path = self._win_rate_path(identity)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            data = {key: payload[key] for key in (
                "schema_version", "identity_sha256", "base_observation_sha256", "win_rate_pct")}
            if (data["schema_version"] != 1
                    or data["identity_sha256"] != canonical_json_sha256(identity)
                    or data["base_observation_sha256"] != canonical_json_sha256(row)
                    or payload.get("evidence_sha256") != canonical_json_sha256(data)):
                raise ValueError("metric evidence provenance differs from canonical Replay")
            augmented = {**row, "win_rate_pct": data["win_rate_pct"]}
            return _validate_observation(augmented, target=identity["target"], seed=identity["seed"])
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Oracle win-rate evidence is corrupt or inconsistent; fail closed") from exc

    def write_win_rate_evidence(self, identity: Mapping[str, Any],
                                previous: Mapping[str, Any],
                                replayed: Mapping[str, Any], *,
                                allow_historical_execution_input_drift: bool = False) -> dict[str, Any]:
        """Backfill only an observed metric after verifying identical Replay economics.

        Never rewrite an earlier signed observation or let a changed result
        silently masquerade as a presentation-only field addition.
        """
        old = _validate_observation(previous, target=identity["target"], seed=identity["seed"])
        new = _validate_observation(replayed, target=identity["target"], seed=identity["seed"])
        if "win_rate_pct" in old or "win_rate_pct" not in new:
            raise ValueError("Oracle win-rate supplement requires a missing original metric")
        expected = dict(new)
        expected.pop("win_rate_pct")
        if not _same_replay_observation(old, expected):
            raise ValueError("Oracle win-rate metric replay changes prior economic evidence; BLOCKED")
        # Resolve again after Replay: the signed original must not have changed.
        persisted, _ = self.resolve_completed(
            identity,
            allow_historical_execution_input_drift=allow_historical_execution_input_drift,
        )
        if persisted is None or persisted != old:
            raise ValueError("Oracle original replay receipt changed during win-rate backfill")
        path = self._win_rate_path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "identity_sha256": canonical_json_sha256(identity),
            "base_observation_sha256": canonical_json_sha256(old),
            "win_rate_pct": float(new["win_rate_pct"]),
        }
        payload = {**data, "evidence_sha256": canonical_json_sha256(data)}
        with exclusive_file_lock(path.with_name("." + path.name + ".lock")):
            if path.exists():
                existing = self.with_win_rate(identity, old)
                if existing is None or existing["win_rate_pct"] != new["win_rate_pct"]:
                    raise ValueError("Oracle deterministic win-rate evidence conflict")
            else:
                temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                try:
                    temporary.write_text(json.dumps(payload, ensure_ascii=False,
                        sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    atomic_replace_with_retry(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        return {**old, "win_rate_pct": float(new["win_rate_pct"])}

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
        self, identity: Mapping[str, Any], *,
        allow_historical_execution_input_drift: bool = False,
    ) -> tuple[tuple[dict[str, Any], str] | None, str | None]:
        """Resolve a signed cache against the current execution identity.

        In explicitly authorized interactive plans, old *execution inputs*
        are separate historical results and do not block a current-key RUN.
        Strict read-only/noninteractive callers retain their prior policy.
        Corruption and same-source contradictory observations always block;
        no old cache is overwritten or inferred reusable.
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
                # Historical signed metric supplements belong to their
                # original identity; do not accidentally backfill a valid
                # win-rate observation merely because the aggregate grew.
                complete = self.with_win_rate(peer, record)
                if complete is not None:
                    record = complete
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                raise ValueError(
                    f"Oracle sibling replay cache is corrupt; fail closed: "
                    f"{candidate.relative_to(self.project_root).as_posix()}"
                ) from exc
            # The same revision policy governs both the read-only source plan
            # and final full-identity execution. Old evidence is never rewritten.
            if _independent_source_revision(peer, identity):
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
                if (candidate_identity.get("benchmark") != identity.get("benchmark")
                        and benchmark_seed_execution_equivalent(
                            recorded_benchmark=str(candidate_identity.get("benchmark") or ""),
                            active_benchmark=str(identity.get("benchmark") or ""),
                            seed=int(identity["seed"]),
                        )):
                    # Only the outer aggregate membership differs. Every
                    # executable input, including P16 SHA, must still match
                    # in the full-identity equality below.
                    candidate_identity = dict(candidate_identity, benchmark=identity["benchmark"])
                if oracle_cohorts_compatible(candidate_identity.get("cohort") or {}, requested_cohort):
                    candidate_identity = dict(candidate_identity, cohort=dict(requested_cohort))
                if candidate_identity == dict(identity):
                    aliases.append((candidate, record))
                elif (not allow_historical_execution_input_drift or
                      _same_producer_different_score(candidate_identity, identity)):
                    # The interactive Research work plan can explicitly
                    # authorize a new execution identity while preserving
                    # signed old receipts. In other (read-only) workflows,
                    # source/contract drift still blocks unapproved REPLAY.
                    # A same-producer contradictory score always blocks.
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

    def resolve_completed(
        self, identity: Mapping[str, Any], *,
        allow_historical_execution_input_drift: bool = False,
    ) -> tuple[dict[str, Any] | None, str]:
        """Canonical current cache or fully verified legacy-equivalent alias.

        Authorized interactive execution may coexist with *different* signed
        historical inputs. A source mismatch is never a REUSE: it returns
        missing, so only the requested new identity gets RUN.
        """
        current = self.read(identity)
        if current is not None:
            return current, "verified per-arm cache"
        legacy, conflict = self._inspect_siblings(
            identity,
            allow_historical_execution_input_drift=allow_historical_execution_input_drift,
        )
        if conflict is not None:
            raise ValueError(f"same-window replay_cache has a DIFFERENT identity for "
                             f"{identity['target']} seed={identity['seed']}; {conflict}; "
                             f"replay remains BLOCKED until provenance is verified")
        if legacy is not None:
            record, source = legacy
            return record, f"verified legacy replay cache ({source})"
        return None, "none"

    def plan_execution(
        self, identities: Mapping[tuple[int, str], Mapping[str, Any]], *,
        allow_historical_execution_input_drift: bool = False,
    ) -> OracleReplayExecutionPlan:
        """Certify once. Only an authorized RUN may coexist with old inputs."""
        if not identities:
            raise ValueError("Oracle empty replay comparison plan")
        result: dict[tuple[int, str], OracleReplayDecision] = {}
        for (seed, target), identity in identities.items():
            key = (int(seed), str(target))
            if identity.get("seed") != key[0] or identity.get("target") != key[1]:
                raise ValueError("Oracle plan key and scientific identity differ")
            try:
                observation, origin = self.resolve_completed(
                    identity,
                    allow_historical_execution_input_drift=allow_historical_execution_input_drift,
                )
                result[key] = OracleReplayDecision(
                    "REUSE" if observation is not None else "RUN",
                    dict(identity), observation, origin,
                )
            except ValueError as exc:
                result[key] = OracleReplayDecision(
                    "BLOCKED", dict(identity), None, "none", str(exc),
                )
        return OracleReplayExecutionPlan(
            self, result,
            allow_historical_execution_input_drift=allow_historical_execution_input_drift,
        )

    def existing_completed_identities(self, *, seed: int, target: str) -> tuple[dict[str, Any], ...]:
        """Inventory *verified stored bytes*, not current scientific REUSE eligibility.

        Current-source identity reconciliation may inspect signed completion
        candidates without computing future labels. Only a FULL replay identity
        resolved by the canonical planner may authorize REUSE/RUN/BLOCKED.
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

    def inspect_current_source(
        self, *, seed: int, target: str, expected_inputs: Mapping[str, Any],
        allow_historical_execution_input_drift: bool = False,
        certified_price_lineage: Mapping[str, Any] | None = None,
        diagnostic_out: dict[str, Any] | None = None,
    ) -> tuple[tuple[dict[str, Any], ...], str | None]:
        """Find signed source candidates without producing any missing scores.

        This cannot certify REUSE: the caller must still reconstruct and check
        the complete replay identity. With no current candidate, known input
        drift keeps the strict noninteractive policy; unknown score/key hashes
        remain unknown. Approved producer revisions share the final cache rule.
        """
        candidates = self.existing_completed_identities(seed=seed, target=target)
        expected = dict(expected_inputs)
        if isinstance(expected.get("replay_policy"), Mapping):
            expected["replay_policy"] = _execution_replay_policy(expected["replay_policy"])
        normalized = []
        for candidate in candidates:
            inputs = dict(candidate)
            if (inputs.get("benchmark") != expected.get("benchmark")
                    and benchmark_seed_execution_equivalent(
                        recorded_benchmark=str(inputs.get("benchmark") or ""),
                        active_benchmark=str(expected.get("benchmark") or ""),
                        seed=int(seed),
                    )):
                # Outer benchmark membership is not a different per-seed
                # execution. This is only a source *candidate* probe: exact
                # P16, price, score and Replay identities are still attested
                # below and by the canonical cache owner before REUSE.
                inputs["benchmark"] = expected["benchmark"]
            if isinstance(inputs.get("replay_policy"), Mapping):
                inputs["replay_policy"] = _execution_replay_policy(inputs["replay_policy"])
                wanted_policy = expected.get("replay_policy") or {}
                if isinstance(wanted_policy, Mapping):
                    previous_lineage = inputs["replay_policy"].get("hindsight_target_contract")
                    current_lineage = wanted_policy.get("hindsight_target_contract")
                    from services.audit.oracle_score_artifacts import (
                        certified_price_lineages_equivalent,
                    )
                    if (certified_price_lineage is not None and
                            certified_price_lineages_equivalent(
                                previous_lineage, certified_price_lineage)):
                        # This preliminary probe intentionally knows only the
                        # producer SHA. Complete source/score/economic evidence
                        # is checked later from the original signed identity.
                        inputs["replay_policy"]["hindsight_target_contract"] = {
                            **previous_lineage,
                            "truth_source_sha256": certified_price_lineage["truth_source_sha256"],
                        }
            normalized.append((candidate, inputs))
        current = tuple(candidate for candidate, inputs in normalized
                        if not _known_input_differences(inputs, expected))
        if diagnostic_out is not None:
            # Read-only provenance facts. Never expose local paths, signed
            # values, or score data; show FIELD NAMES only. A diagnostic does
            # not authorize an incomplete/same-window receipt as REUSE.
            comparable = [
                inputs for _candidate, inputs in normalized
                if all(inputs.get(field) == expected.get(field)
                       for field in ("dataset", "start_date", "end_date"))
            ]
            differences = sorted({field for inputs in comparable
                                  for field in _known_input_differences(inputs, expected)})
            diagnostic_out.update(
                signed_receipts=len(candidates),
                same_window_receipts=len(comparable),
                source_candidates=len(current),
                differing_fields=tuple(differences),
            )
        if current or allow_historical_execution_input_drift:
            return current, None
        for candidate, inputs in normalized:
            if _independent_source_revision(candidate, expected):
                continue
            if all(candidate.get(field) == expected.get(field)
                   for field in ("dataset", "start_date", "end_date")):
                changed = _known_input_differences(inputs, expected)
                if changed:
                    return (), ("same-window replay_cache has a DIFFERENT identity for "
                                f"{target} seed={seed}; changed fields: " + "; ".join(changed))
        return (), None

    def resolve_verified_price_score_revision(
        self, *, expected: Mapping[str, Any], saved: Mapping[str, Any],
        witness: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Replay a signed *observation lookup*, not an old numerical producer.

        The caller must have independently hashed the full archived price-score
        array and confirmed its exact per-target value/key digest. This method
        then requires an identical full economic identity after removing ONLY
        the two source-code lineage digests proven by that archived truth
        witness. No old signed receipt or scientific identity is rewritten.
        """
        old_policy = saved.get("replay_policy")
        new_policy = expected.get("replay_policy")
        if not isinstance(old_policy, Mapping) or not isinstance(new_policy, Mapping):
            return None
        old_truth = old_policy.get("hindsight_target_contract")
        new_truth = new_policy.get("hindsight_target_contract")
        if not isinstance(old_truth, Mapping) or not isinstance(new_truth, Mapping):
            return None
        required = (
            "source_code_sha256", "score_algorithm_sha256",
            "dataset_summary_sha256", "label_policy", "truth_source_sha256",
        )
        if not all(field in old_truth and field in new_truth for field in required):
            return None
        old_digest = canonical_json_sha256({key: old_truth[key] for key in required[:-1]})
        new_digest = canonical_json_sha256({key: new_truth[key] for key in required[:-1]})
        if (old_digest != old_truth["truth_source_sha256"]
                or new_digest != new_truth["truth_source_sha256"]
                or old_truth["source_code_sha256"] != witness.get("source_code_sha256")
                or old_truth["truth_source_sha256"] != witness.get("truth_source_sha256")
                or old_policy.get("perfect_score_sha256") != witness.get("score_sha256")
                or new_policy.get("perfect_score_sha256") != witness.get("score_sha256")):
            return None
        for policy in (old_policy, new_policy):
            score_contract = policy.get("canonical_dl_score_contract")
            if (not isinstance(score_contract, Mapping)
                    or score_contract.get("canonical_feature_eligible_keys_sha256") !=
                        witness.get("eligible_keys_sha256")
                    or score_contract.get("oracle_scored_rows") !=
                        witness.get("cohort_rows")):
                return None
        for identity in (saved, expected):
            cohort = identity.get("cohort")
            if (not isinstance(cohort, Mapping)
                    or cohort.get("cohort_rows") != witness.get("cohort_rows")
                    or cohort.get("first_common_day") != witness.get("first_common_day")
                    or cohort.get("last_common_day") != witness.get("last_common_day")):
                return None
        # Every non-source execution input, including exact/direct ranking,
        # P16/seed/fee, full PIT score mapping, time/feature eligibility and
        # portfolio boundary must match. An archived cohort may carry extra
        # report-only provider fields; use the existing V2 normalizer.
        normalized = _v2_presentation_identity(saved, expected)
        normalized = dict(normalized)
        peer_policy = dict(normalized.get("replay_policy") or {})
        peer_truth = dict(peer_policy.get("hindsight_target_contract") or {})
        if (set(peer_truth) != set(new_truth)
                or any(peer_truth.get(key) != value for key, value in new_truth.items()
                       if key not in ("source_code_sha256", "truth_source_sha256"))):
            return None
        peer_truth["source_code_sha256"] = new_truth["source_code_sha256"]
        peer_truth["truth_source_sha256"] = new_truth["truth_source_sha256"]
        peer_policy["hindsight_target_contract"] = peer_truth
        normalized["replay_policy"] = peer_policy
        if normalized.get("benchmark") != expected.get("benchmark"):
            if not benchmark_seed_execution_equivalent(
                    recorded_benchmark=str(normalized.get("benchmark") or ""),
                    active_benchmark=str(expected.get("benchmark") or ""),
                    seed=int(expected["seed"])):
                return None
            normalized["benchmark"] = expected["benchmark"]
        if normalized != dict(expected):
            return None
        obs = self.read(saved)  # original signed identity + observation SHA
        if obs is None:
            return None
        return self.with_win_rate(saved, obs)

    def resolve_verified_daily_truth_revision(
        self, *, expected: Mapping[str, Any], saved: Mapping[str, Any],
        verified_evidence: Mapping[str, Any], score_sha256: str,
    ) -> dict[str, Any] | None:
        """REUSE after the CURRENT MinTrade truth Producer has verified its output.

        Unlike a source-SHA compatibility table, this path cannot certify an
        old score using only historical declarations: the caller must have
        loaded or built current-source truth, checked every stock-day score
        against its actual keys/SHA, and constructed `expected` from that
        evidence. Only then can the old signed Replay be reused without
        re-executing Portfolio. All original cache bytes remain immutable.
        """
        if (not is_sha256_hex(score_sha256)
                or expected.get("target") != saved.get("target")
                or expected.get("seed") != saved.get("seed")):
            return None
        old_policy = saved.get("replay_policy")
        new_policy = expected.get("replay_policy")
        if not isinstance(old_policy, Mapping) or not isinstance(new_policy, Mapping):
            return None
        old_truth = old_policy.get("hindsight_target_contract")
        new_truth = new_policy.get("hindsight_target_contract")
        if not isinstance(old_truth, Mapping) or not isinstance(new_truth, Mapping):
            return None
        if (old_policy.get("perfect_score_sha256") != score_sha256
                or new_policy.get("perfect_score_sha256") != score_sha256
                or not is_sha256_hex(new_truth.get("truth_source_sha256"))
                or verified_evidence.get("truth_source_sha256") !=
                    new_truth["truth_source_sha256"]):
            return None
        # The producer's full current evidence is an independent second
        # witness. Do not let a caller invent a source-equivalent identity
        # from a self-consistent old Replay receipt alone.
        evidence_fields = (
            "target_id", "label_id", "profile_id", "rank_policy",
            "rank_sha256", "raw_label_sha256", "dataset_summary_sha256",
            "calendar_sha256", "teacher", "mature_eligible_rows",
            "research_cutoff", "score_start_date", "censored_rank_last_rows",
        )
        if any(verified_evidence.get(field) != new_truth.get(field)
               for field in evidence_fields):
            return None
        if any(not is_sha256_hex(new_truth.get(field)) for field in
               ("raw_label_sha256", "rank_sha256", "calendar_sha256",
                "dataset_summary_sha256")):
            return None
        teacher = new_truth.get("teacher")
        if (not isinstance(teacher, Mapping)
                or teacher.get("seed") != expected["seed"]
                or teacher.get("param_identity_sha256") != expected.get("param_sha256")):
            return None
        if (set(old_truth) ^ set(new_truth)) - {"dataset_summary_sha256"}:
            return None
        # Preserve the historic science. Current-source NPZ evidence may also
        # supply changed/missing Dataset metadata: identical actual raw labels,
        # ranks, keys, price/P16, period and every economic rule remain strict.
        if any(old_truth.get(field) != value for field, value in new_truth.items()
               if field not in {"truth_source_sha256", "dataset_summary_sha256"}):
            return None
        normalized = _v2_presentation_identity(saved, expected)
        normalized = dict(normalized)
        policy = dict(normalized.get("replay_policy") or {})
        policy_truth = dict(policy.get("hindsight_target_contract") or {})
        if policy_truth != _execution_replay_policy({
            "hindsight_target_contract": old_truth,
        }).get("hindsight_target_contract"):
            return None
        policy_truth["truth_source_sha256"] = new_truth["truth_source_sha256"]
        policy_truth["dataset_summary_sha256"] = new_truth["dataset_summary_sha256"]
        policy["hindsight_target_contract"] = policy_truth
        normalized["replay_policy"] = policy
        if normalized.get("benchmark") != expected.get("benchmark"):
            if not benchmark_seed_execution_equivalent(
                    recorded_benchmark=str(normalized.get("benchmark") or ""),
                    active_benchmark=str(expected.get("benchmark") or ""),
                    seed=int(expected["seed"])):
                return None
            normalized["benchmark"] = expected["benchmark"]
        if normalized != dict(expected):
            return None
        # Requires signed identity/observation and explicit Win Rate evidence.
        observed = self.read(saved)
        if observed is None:
            return None
        return self.with_win_rate(saved, observed)

    def write_verified_daily_truth_revision(
        self, *, expected: Mapping[str, Any], saved: Mapping[str, Any],
        verified_evidence: Mapping[str, Any], score_sha256: str,
    ) -> dict[str, Any] | None:
        """Seal the exact present-day execution identity after one authorized verification.

        The old receipt remains byte-identical.  A future preflight can reuse
        the new certified identity without materializing the daily label table
        again; its certificate always points to the immutable signed origin.
        """
        result = self.resolve_verified_daily_truth_revision(
            expected=expected, saved=saved, verified_evidence=verified_evidence,
            score_sha256=score_sha256,
        )
        if result is None:
            return None
        origin_observation = self.read(saved)
        if origin_observation is None:
            return None
        certificate = {
            "schema_version": 1,
            "origin_identity": dict(saved),
            "origin_observation_sha256": canonical_json_sha256(origin_observation),
            "verified_observation_sha256": canonical_json_sha256(result),
            "full_truth_evidence_sha256": canonical_json_sha256(verified_evidence),
            "score_sha256": score_sha256,
            "truth_source_sha256": expected["replay_policy"]["hindsight_target_contract"]["truth_source_sha256"],
        }
        self.write(expected, result, origin="certified_identical_signed_daily_truth_replay",
                   reuse_certificate=certificate)
        persisted = self.read(expected)
        if persisted != result:
            raise ValueError("Oracle attested source revision observation differs after publication")
        return result

    def read(self, identity: Mapping[str, Any], *,
             _seen: frozenset[str] = frozenset()) -> dict[str, Any] | None:
        path = self._path(identity)
        if not path.is_file():
            return None
        identity_sha = canonical_json_sha256(identity)
        if identity_sha in _seen:
            raise ValueError("Oracle replay reuse certificate contains an identity cycle")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload.get("cache_schema") != identity.get("cache_schema")
                    or identity.get("cache_schema") not in (_LEGACY_CACHE_SCHEMA, _CACHE_SCHEMA)
                    or payload.get("identity") != dict(identity)
                    or payload.get("identity_sha256") != identity_sha):
                raise ValueError("Oracle cache identity mismatch")
            observation = _validate_observation(payload["observation"], target=identity["target"], seed=identity["seed"])
            if payload.get("observation_sha256") != canonical_json_sha256(observation):
                raise ValueError("Oracle cached observation SHA256 mismatch")
            if payload.get("origin") == "certified_identical_signed_daily_truth_replay":
                certificate = payload.get("reuse_certificate")
                if (not isinstance(certificate, dict)
                        or certificate.get("schema_version") != 1
                        or not isinstance(certificate.get("origin_identity"), dict)
                        or not is_sha256_hex(certificate.get("full_truth_evidence_sha256"))
                        or not is_sha256_hex(certificate.get("origin_observation_sha256"))
                        or not is_sha256_hex(certificate.get("verified_observation_sha256"))
                        or not is_sha256_hex(certificate.get("truth_source_sha256"))
                        or payload.get("reuse_certificate_sha256") != canonical_json_sha256(certificate)):
                    raise ValueError("Oracle source-revision certificate malformed")
                origin_identity = certificate["origin_identity"]
                truth = identity.get("replay_policy", {}).get("hindsight_target_contract", {})
                if (certificate["truth_source_sha256"] != truth.get("truth_source_sha256")
                        or certificate["score_sha256"] != identity.get("replay_policy", {}).get("perfect_score_sha256")
                        or certificate["verified_observation_sha256"] != canonical_json_sha256(observation)
                        or origin_identity.get("seed") != identity.get("seed")
                        or origin_identity.get("target") != identity.get("target")
                        or origin_identity == dict(identity)):
                    raise ValueError("Oracle source-revision certificate binding mismatch")
                signed_origin = self.read(origin_identity, _seen=_seen | {identity_sha})
                if (signed_origin is None or canonical_json_sha256(signed_origin) !=
                        certificate["origin_observation_sha256"]):
                    raise ValueError("Oracle attested historical replay receipt is missing or changed")
                verified_origin = self.with_win_rate(origin_identity, signed_origin)
                if verified_origin != observation:
                    raise ValueError("Oracle certified old and new replay observations differ")
            elif "reuse_certificate" in payload or "reuse_certificate_sha256" in payload:
                raise ValueError("Oracle unsupported replay certificate origin")
            return observation
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Oracle existing replay cache is corrupt or conflicting; fail closed") from exc

    def write(self, identity: Mapping[str, Any], observation: Mapping[str, Any], *,
              origin: str, reuse_certificate: Mapping[str, Any] | None = None) -> None:
        if identity.get("cache_schema") != _CACHE_SCHEMA:
            raise ValueError("Oracle historical replay cache is read-only")
        record = _validate_observation(observation, target=identity["target"], seed=identity["seed"])
        if (origin == "certified_identical_signed_daily_truth_replay") != (reuse_certificate is not None):
            raise ValueError("Oracle source-revision receipt requires a verifiable certificate")
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
            if reuse_certificate is not None:
                payload["reuse_certificate"] = dict(reuse_certificate)
                payload["reuse_certificate_sha256"] = canonical_json_sha256(reuse_certificate)
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

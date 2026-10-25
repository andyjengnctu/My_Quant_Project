"""Canonical per-seed Oracle replay evidence cache (hindsight-only, never PIT).

This module owns execution reuse, not Target, selector or accounting semantics.
Full-identity cache entries survive interrupted audits; historical V1 reports may
be read as observations only after an explicit strict compatibility check.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from config.audit import AUDIT_OUTPUT_ROOT
from core.report_contract_versions import load_report_contract_versions
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
        "target_definition": str(definition[target]),
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
        "replay_policy": dict(replay_policy),
        "cohort": dict(cohort),
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
            if (peer.get("dataset") == identity["dataset"]
                    and peer.get("start_date") == identity["start_date"]
                    and peer.get("end_date") == identity["end_date"]
                    and (peer_cohort.get("providers") == requested_cohort.get("providers")
                         or (isinstance(peer_cohort.get("providers"), Mapping)
                             and isinstance(requested_cohort.get("providers"), Mapping)
                             and set(requested_cohort["providers"]).issubset(peer_cohort["providers"])))):
                candidate_identity = self._legacy_semantic_identity(peer, identity)
                if candidate_identity is None:
                    candidate_identity = dict(peer)
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

    def explain_other_identity(self, identity: Mapping[str, Any]) -> str | None:
        """Explain genuine replay differences; legacy UI-only drift is compatible."""
        _legacy, conflict = self._inspect_siblings(identity)
        return conflict

    def has_other_identity(self, identity: Mapping[str, Any]) -> bool:
        """Compatibility probe; explanation and matching have a single owner."""
        return self.explain_other_identity(identity) is not None

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


class HistoricalOracleIndex:
    """Discover locally completed Oracle reports without a target-inventory allowlist.

    Historical reports may predate per-arm input-byte fingerprints. Admit an
    observation only if its *own* target definition, seed, completed metrics,
    research cohort and strategy provenance match the current requested arm.
    Report inventories and presentation versions must not dictate reuse.
    """

    def __init__(self, project_root: Path):
        output_root = Path(project_root).resolve() / AUDIT_OUTPUT_ROOT / "breakout_quality"
        self._output_root = output_root
        self._project_root = Path(project_root).resolve()
        # The append-only report ledger verifies presentation provenance; a
        # future report version must not force identical science to be replayed.
        versions = load_report_contract_versions().get("audit.target_oracle_romd", ())
        self._approved_report_versions = {item["contract_id"]: item["fingerprint"] for item in versions}
        self.snapshots: list[tuple[str, Path, dict[str, Any]]] = []
        # Search the local Audit output tree, rather than an explicit list of
        # past report versions, folders or former Oracle target collections.
        for path in sorted(output_root.rglob("audit.json")):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                if "target_oracle_romd" in path.parts:
                    raise ValueError("Oracle completed report unreadable; do not silently recalculate") from exc
                continue  # An unrelated Audit report is not Oracle provenance.
            if not isinstance(payload, dict) or payload.get("report_type") not in (
                None, "target_ground_truth_strategy_romd",
            ):
                continue
            if payload.get("report_id") == "target_oracle_romd":
                origin = "reusable"
            elif "one_time" in path.parts and payload.get("report_id") in (
                None, "AUD-target-ground-truth-romd",
            ):
                origin = "one_time"
            else:
                continue
            if payload.get("status") == "COMPLETED":
                self.snapshots.append((origin, path, payload))
                # A persistent Audit may contain independently matured horizon
                # panels. They carry the same approved report provenance, but
                # each panel has its own canonical cohort, dates and baseline.
                # A later run recovers its completed per-seed observations even
                # when the strict per-arm replay cache is absent.
                panels = payload.get("additional_panels") or {}
                if not isinstance(panels, dict):
                    raise ValueError("Oracle completed report has invalid additional panels")
                for panel_id, panel in panels.items():
                    if not isinstance(panel_id, str) or not isinstance(panel, dict):
                        raise ValueError("Oracle completed panel provenance is malformed")
                    if not all(field in panel for field in (
                        "start_date", "end_date", "common_cohort", "definitions", "baseline", "oracles",
                        "param_artifacts",
                    )):
                        raise ValueError("Oracle completed panel lacks scientific provenance")
                    # No hardcoded horizon or target allowlist in the index.
                    panel_payload = dict(payload)
                    panel_payload.update(panel)
                    self.snapshots.append((origin, path, panel_payload))

    def _compatibility_issues(self, payload: Mapping[str, Any], *,
                              context: Mapping[str, Any], origin: str) -> tuple[str, ...]:
        """Explain the same validation used for admission; never weaken reuse."""
        issues: list[str] = []
        for key, expected in (
            ("status", "COMPLETED"), ("hindsight_non_pit", True),
            ("read_only", True), ("benchmark_id", context["benchmark"]),
            ("strategy_profile_id", context["strategy_profile_id"]),
            ("dataset", context["dataset"]),
            ("start_date", context["start_date"]),
            ("end_date", context["end_date"]),
        ):
            if payload.get(key) != expected:
                issues.append(key)
        if list(payload.get("seeds") or []) != list(context["seeds"]):
            issues.append("seeds")
        if not isinstance(payload.get("baseline"), dict):
            issues.append("baseline")
        if not isinstance(payload.get("oracles"), list):
            issues.append("oracles")
        if payload.get("report_type") not in (None, "target_ground_truth_strategy_romd"):
            issues.append("report_type")
        if origin == "one_time":
            if payload.get("report_id") not in (None, "AUD-target-ground-truth-romd"):
                issues.append("report_id")
        elif origin == "reusable":
            if payload.get("report_id") != "target_oracle_romd":
                issues.append("report_id")
            if payload.get("report_type") != "target_ground_truth_strategy_romd":
                issues.append("report_type")
            version = payload.get("schema_version")
            if type(version) is not int or version < 2:
                issues.append("schema_version")
            if payload.get("report_contract_fingerprint") != self._approved_report_versions.get(
                payload.get("report_contract_id")
            ):
                issues.append("report_contract_id/fingerprint")
        else:
            issues.append("origin")
        cohort = payload.get("common_cohort")
        if not isinstance(cohort, dict):
            issues.append("common_cohort")
        else:
            wanted = context["cohort"]
            if not oracle_cohorts_compatible(cohort, wanted):
                issues.append("common_cohort.scientific_identity")
        definitions = payload.get("definitions")
        if (not isinstance(definitions, dict) or not definitions
                or any(not isinstance(name, str) or not name
                       or not isinstance(description, str) or not description
                       for name, description in definitions.items())):
            issues.append("definitions")
        param_rows = payload.get("param_artifacts")
        if not isinstance(param_rows, list) or len(param_rows) != len(context["seeds"]):
            issues.append("param_artifacts")
        else:
            try:
                archived = {int(item["seed"]): str(item["param_sha256"])
                            for item in param_rows
                            if isinstance(item, dict) and "seed" in item and "param_sha256" in item}
            except (ValueError, TypeError, OverflowError):
                archived = {}
            if archived != context["param_sha_by_seed"] or not all(
                is_sha256_hex(value) for value in archived.values()
            ):
                issues.append("param_artifacts.param_sha256")
        if isinstance(payload.get("oracles"), list) and isinstance(definitions, dict):
            try:
                targets = {row.get("target") for row in payload["oracles"] if isinstance(row, dict)}
                if len(targets) != len(payload["oracles"]) or targets != set(definitions):
                    issues.append("oracles/definitions inventory")
            except (TypeError, ValueError):
                issues.append("oracles/definitions inventory")
        return tuple(dict.fromkeys(issues))

    def _compatible(self, payload: Mapping[str, Any], *, context: Mapping[str, Any], origin: str) -> bool:
        return not self._compatibility_issues(payload, context=context, origin=origin)

    def diagnose_missing(self, *, target: str, context: Mapping[str, Any]) -> tuple[str, ...]:
        """Explain related completed local results rejected by strict identity.

        Other maturities and reports without this target are not evidence that
        the requested new arm was previously computed.
        """
        explanations: list[str] = []
        for origin, path, payload in self.snapshots:
            if any(payload.get(key) != context[key] for key in ("dataset", "start_date", "end_date")):
                continue
            definitions = payload.get("definitions")
            if target != "C96" and isinstance(definitions, dict) and target not in definitions:
                continue
            problems = list(self._compatibility_issues(payload, context=context, origin=origin))
            if target != "C96" and payload["definitions"].get(target) != context["definitions"].get(target):
                problems.append(f"definitions.{target}")
            if not problems:
                problems.append("missing complete same-seed observations")
            short_path = path.relative_to(self._project_root).as_posix()
            explanations.append(f"{short_path}: {', '.join(problems)}")
        return tuple(explanations)

    def lookup(self, *, target: str, seed: int, context: Mapping[str, Any]) -> dict[str, Any] | None:
        observations = []
        for origin, source, payload in self.snapshots:
            if not self._compatible(payload, context=context, origin=origin):
                continue
            if target != "C96" and payload["definitions"].get(target) != context["definitions"].get(target):
                continue
            rows = [payload["baseline"]] if target == "C96" else [
                row for row in payload["oracles"] if row.get("target") == target
            ]
            for aggregate in rows:
                seed_rows = aggregate.get("seed_observations")
                if not isinstance(seed_rows, list) or len(seed_rows) != len(context["seeds"]):
                    continue
                try:
                    rows_by_seed = {int(record["seed"]): record for record in seed_rows}
                    if sorted(rows_by_seed) != sorted(context["seeds"]):
                        continue
                    for field in _REQUIRED_METRICS:
                        mean = sum(float(item[field]) for item in seed_rows) / len(seed_rows)
                        if not math.isclose(mean, float(aggregate[field]), rel_tol=1e-7, abs_tol=1e-7):
                            raise ValueError("Oracle historical aggregate disagrees with seed rows")
                    record = _validate_observation(rows_by_seed[int(seed)], target=target, seed=seed)
                except (TypeError, KeyError, OverflowError, ValueError) as exc:
                    raise ValueError("Oracle matched historical observation is incomplete / corrupted") from exc
                observations.append(record)
        if not observations:
            return None
        fingerprints = {canonical_json_sha256(row) for row in observations}
        if len(fingerprints) != 1:
            raise ValueError("Oracle historical same-identity observations conflict")
        return observations[0]


__all__ = [
    "OracleReplayStore", "HistoricalOracleIndex", "price_inventory_sha256", "replay_identity",
    "oracle_cohorts_compatible",
]

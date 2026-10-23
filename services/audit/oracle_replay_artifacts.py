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

_CACHE_SCHEMA = 1
_REQUIRED_METRICS = (
    "total_return_pct", "annualized_return_pct", "mdd_pct", "romd", "ev_a_r",
    "trades", "exposure_pct",
)



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
                    definition: Mapping[str, Any], settings_fingerprint: str,
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
        "strategy_settings_fingerprint": str(settings_fingerprint),
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


class OracleReplayStore:
    """Durable per-arm cache; no producer-side recomputation on a cache hit."""

    def __init__(self, project_root: Path, output_subdir: str):
        root = Path(project_root).resolve()
        self.cache_root = root / AUDIT_OUTPUT_ROOT / Path(output_subdir) / "replay_cache"

    def _path(self, identity: Mapping[str, Any]) -> Path:
        seed = int(identity["seed"])
        target = str(identity["target"])
        if not target.replace("_", "").isalnum():
            raise ValueError("Oracle invalid cache target")
        return self.cache_root / str(seed) / f"{target}_{canonical_json_sha256(identity)}.json"

    def has_other_identity(self, identity: Mapping[str, Any]) -> bool:
        """Block weaker legacy fallback after *same-panel* scientific drift.

        Different mature horizon panels legitimately share C96/seed and cache
        directory. Their distinct window/cohort must never disable report-based
        reuse for the other panel. Within the same panel, changed input bytes,
        params or replay policy must still block weaker historical fallback.
        """
        path = self._path(identity)
        requested_cohort = identity.get("cohort") or {}
        for candidate in path.parent.glob(f"{identity['target']}_*.json"):
            if candidate == path:
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                peer = payload["identity"]
                if (payload.get("cache_schema") != _CACHE_SCHEMA
                        or not isinstance(peer, dict)
                        or peer.get("target") != identity["target"]
                        or int(peer.get("seed", -1)) != int(identity["seed"])):
                    raise ValueError("Oracle sibling replay cache identity malformed")
                peer_cohort = peer.get("cohort") or {}
                if not isinstance(peer_cohort, dict):
                    raise ValueError("Oracle sibling replay cache cohort malformed")
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                raise ValueError("Oracle sibling replay cache is corrupt; fail closed") from exc
            if (peer.get("dataset") == identity["dataset"]
                    and peer.get("start_date") == identity["start_date"]
                    and peer.get("end_date") == identity["end_date"]
                    and peer_cohort.get("providers") == requested_cohort.get("providers")
                    and peer_cohort.get("scoring") == requested_cohort.get("scoring")):
                return True
        return False

    def read(self, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._path(identity)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload.get("cache_schema") != _CACHE_SCHEMA
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
                "cache_schema": _CACHE_SCHEMA,
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

    def _compatible(self, payload: Mapping[str, Any], *, context: Mapping[str, Any], origin: str) -> bool:
        if (payload.get("status") != "COMPLETED" or payload.get("hindsight_non_pit") is not True
                or payload.get("read_only") is not True
                or payload.get("benchmark_id") != context["benchmark"]
                or payload.get("strategy_profile_id") != context["strategy_profile_id"]
                or payload.get("dataset") != context["dataset"]
                or payload.get("start_date") != context["start_date"]
                or payload.get("end_date") != context["end_date"]
                or list(payload.get("seeds") or []) != list(context["seeds"])
                or not isinstance(payload.get("baseline"), dict)
                or not isinstance(payload.get("oracles"), list)):
            return False
        if payload.get("report_type") not in (None, "target_ground_truth_strategy_romd"):
            return False
        # Presentation identity is separate from per-target scientific identity.
        # Reusable reports must have a ledger-approved report contract.
        if origin == "one_time":
            if payload.get("report_id") not in (None, "AUD-target-ground-truth-romd"):
                return False
        elif origin == "reusable":
            if (payload.get("report_id") != "target_oracle_romd"
                    or payload.get("report_type") != "target_ground_truth_strategy_romd"
                    or not isinstance(payload.get("schema_version"), int)
                    or payload["schema_version"] < 2
                    or payload.get("report_contract_fingerprint") != self._approved_report_versions.get(
                        payload.get("report_contract_id")
                    )):
                return False
        else:
            return False
        cohort = payload.get("common_cohort")
        if not isinstance(cohort, dict):
            return False
        wanted = context["cohort"]
        for field in ("cohort_rows", "first_common_day", "last_common_day", "research_cutoff",
                      "risk_budget_return", "providers", "scoring", "market_truth",
                      "rows_outside_dl_training_provider_by_source"):
            if cohort.get(field) != wanted.get(field):
                return False
        # Source-defined supplemental evidence is validated when present, but
        # an older report need not possess fields introduced in newer schemas.
        # Unknown source fields cannot silently redefine the current cohort.
        if any(key not in wanted or wanted[key] != value for key, value in cohort.items()):
            return False
        definitions = payload.get("definitions")
        if (not isinstance(definitions, dict) or not definitions
                or any(not isinstance(name, str) or not name
                       or not isinstance(description, str) or not description
                       for name, description in definitions.items())):
            return False
        param_rows = payload.get("param_artifacts") or []
        if not isinstance(param_rows, list) or len(param_rows) != len(context["seeds"]):
            return False
        archived = {int(item["seed"]): str(item["param_sha256"])
                    for item in param_rows
                    if isinstance(item, dict) and "seed" in item and "param_sha256" in item}
        if archived != context["param_sha_by_seed"] or not all(is_sha256_hex(v) for v in archived.values()):
            return False
        targets = {row.get("target") for row in payload["oracles"] if isinstance(row, dict)}
        # A completed report must be internally consistent, regardless of its
        # historical number of targets. Matching the *requested* definition
        # is performed in lookup(), not via a hardcoded report inventory.
        if (len(targets) != len(payload["oracles"])
                or targets != set(definitions)):
            return False
        return True

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
]

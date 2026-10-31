"""Effective Min truth references shared by Exact, Direct and gate consumers.

The original NPZ and its source provenance are immutable. A reference records
which already certified calculation/input contract can consume it. Planning
never opens its numerical arrays; an actual RUN consumer validates those
arrays through the original truth owner before using them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from core.file_integrity import canonical_json_sha256, atomic_replace_with_retry, exclusive_file_lock


def source_context(execution: Mapping[str, Any]) -> dict[str, Any]:
    source = execution["source_contract"]
    if not source or not str(source["recipe"]).startswith("daily_trade_path_"):
        raise ValueError("Min truth reference requires a daily trade-path capability")
    return {
        key: execution[key] for key in (
            "dataset", "price_sha256", "param_sha256", "seed", "start_date", "end_date",
            "research_cutoff", "source_contract", "dataset_inputs", "replay_calculation_revision",
            "replay_implementation_sha256",
        )
    } | {
        "schema": 1,
        "filter_id": execution["replay_policy"]["filter_id"],
        "score_start": execution["score_contract"]["score_start_date"],
        "economics": {key: execution["replay_policy"][key] for key in (
            "fixed_risk", "max_position_cap_pct", "broker_accounting", "execution_defaults")},
    }


def _evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evidence.items()
            if key not in {"source_reuse_status", "effective_source_reference"}}


class OracleTruthReferences:
    def __init__(self, cache_root: Path):
        self.cache_root = Path(cache_root).resolve()
        self.root = self.cache_root.parent / "truth_index"

    def _path(self, context):
        return self.root / str(int(context["seed"])) / f"{canonical_json_sha256(context)}.json"

    @staticmethod
    def _matches(identity, context):
        mapping = {
            "dataset": "dataset", "price_sha256": "price_sha256", "param_sha256": "param_sha256",
            "seed": "seed", "filter_id": "filter_id", "evaluation_start": "start_date",
            "evaluation_end": "end_date", "research_cutoff": "research_cutoff", "score_start": "score_start",
        }
        return (all(identity.get(old) == context[new] for old, new in mapping.items()) and
                identity.get("profile_id") == context["source_contract"]["feature_provider"]["profile_id"] and
                identity.get("horizon_sessions") == context["source_contract"]["horizon_sessions"])

    def read(self, context):
        path = self._path(context)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            payload = {key: raw[key] for key in ("schema", "context", "origin_identity", "evidence_sha256")}
            if (payload["schema"] != 1 or payload["context"] != context
                    or raw.get("reference_sha256") != canonical_json_sha256(payload)
                    or not self._matches(payload["origin_identity"], context)):
                raise ValueError("truth reference contract/SHA mismatch")
            return raw
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Oracle truth reference corrupt: {path.name}: {exc}") from exc

    def origin_path(self, identity):
        name = str(identity["target_id"])
        if not name.replace("_", "").isalnum():
            raise ValueError("Oracle invalid truth artifact name")
        return self.cache_root / str(int(identity["seed"])) / f"{name}_{canonical_json_sha256(identity)}.npz"

    def _publish(self, context, identity, evidence):
        if not self._matches(identity, context):
            raise ValueError("Oracle truth origin differs from effective inputs")
        payload = {"schema": 1, "context": context, "origin_identity": identity,
                   "evidence_sha256": canonical_json_sha256(_evidence(evidence))}
        path = self._path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(path.with_name("." + path.name + ".lock")):
            existing = self.read(context)
            if existing is not None:
                if existing["evidence_sha256"] != payload["evidence_sha256"]:
                    raise ValueError("Oracle same truth contract has conflicting certified evidence")
                return
            temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
            try:
                temporary.write_text(json.dumps({**payload, "reference_sha256": canonical_json_sha256(payload)},
                    sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                atomic_replace_with_retry(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def publish_existing(self, execution, *, replay_identity=None, evidence=None):
        """Import metadata only AFTER canonical Replay/source certification.

        A missing truth NPZ does not invalidate a completed Portfolio. Without
        an NPZ no source reference is published; a future consumer must BUILD.
        This never infers scores from portfolio summary statistics.
        """
        context = source_context(execution)
        if self.read(context) is not None:
            return
        if replay_identity is not None:
            wanted = replay_identity["replay_policy"]["hindsight_target_contract"]
            score_sha = replay_identity["replay_policy"]["perfect_score_sha256"]
        else:
            wanted = _evidence(evidence)
            score_sha = wanted["rank_sha256"]
        target = str(wanted["target_id"])
        for path in sorted((self.cache_root / str(context["seed"])).glob(target + "_*.npz")):
            # Opening metadata.npy does not decompress tickers/days/scores.
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"].item()))
            identity = meta["identity"]
            if not self._matches(identity, context):
                continue
            original = meta["evidence"]
            if any(original.get(key) != wanted.get(key) for key in (
                    "raw_label_sha256", "rank_sha256", "truth_source_sha256", "dataset_summary_sha256",
                    "calendar_sha256", "teacher", "target_id", "label_id", "mature_eligible_rows")):
                continue
            payload = {key: meta.get(key) for key in ("schema", "identity", "evidence", "score_sha256")}
            if (path != self.origin_path(identity) or meta.get("payload_sha256") != canonical_json_sha256(payload)
                    or meta.get("score_sha256") != score_sha):
                raise ValueError("Oracle signed truth metadata/filename is corrupt")
            self._publish(context, identity, original)
            return

    def load(self, execution):
        context = source_context(execution)
        reference = self.read(context)
        if reference is None:
            return None
        from services.audit.daily_trade40_oracle import _truth_cache_read
        loaded = _truth_cache_read(self.origin_path(reference["origin_identity"]), reference["origin_identity"])
        if loaded is None:
            raise ValueError("Oracle referenced truth artifact is missing")
        scores, evidence = loaded
        if canonical_json_sha256(_evidence(evidence)) != reference["evidence_sha256"]:
            raise ValueError("Oracle referenced truth evidence differs from its certified origin")
        # Bind the feature-key identity to the actual consumed arrays, including
        # historical Windows CSV line endings. Portfolio keys are checked again
        # at the score boundary by the canonical execution service.
        from hashlib import sha256
        ordered = sorted(scores)
        key_digests = {sha256(("ticker,date" + newline + "".join(
            f"{ticker},{day}{newline}" for ticker, day in ordered)).encode()).hexdigest()
            for newline in ("\n", "\r\n")}
        if reference["origin_identity"]["feature_eligible_keys_sha256"] not in key_digests:
            raise ValueError("Oracle referenced truth feature membership differs from actual arrays")
        return scores, {**evidence, "effective_source_reference": reference["reference_sha256"]}

    def validate_consumption(self, execution, evidence):
        reference = self.read(source_context(execution))
        if (reference is None or evidence.get("effective_source_reference") != reference["reference_sha256"]
                or canonical_json_sha256(_evidence(evidence)) != reference["evidence_sha256"]):
            raise ValueError("Oracle consumed truth lacks its effective calculation reference")
        return reference["reference_sha256"]


def load_or_build_daily_truth(*, execution, cache_root, producer):
    references = OracleTruthReferences(cache_root)
    loaded = references.load(execution)
    if loaded is not None:
        return loaded
    scores, evidence = producer()
    references.publish_existing(execution, evidence=evidence)
    if references.read(source_context(execution)) is not None:
        reference = references.read(source_context(execution))
        evidence = {**evidence, "effective_source_reference": reference["reference_sha256"]}
        references.validate_consumption(execution, evidence)
    return scores, evidence


def load_or_build_positive_truth(*, execution, cache_root, **kwargs):
    """Consume a certified source without relabelling its immutable provenance.

    An existing raw witness is read with its original source identity. If no
    raw witness exists, the canonical positive producer reconstructs values
    with current code and independently checks EVERY raw/rank value against
    the certified source. Its new witness then correctly names current code.
    """
    from services.audit import fullcycle_positive_truth as positive
    evidence = kwargs["canonical_evidence"]
    if "effective_source_reference" not in evidence:
        return positive.load_or_build_positive_truth(**kwargs)
    OracleTruthReferences(cache_root).validate_consumption(execution, evidence)
    identity = positive._identity(seed=kwargs["seed"], price_sha=kwargs["price_sha256"],
        param_sha=kwargs["param_sha256"], source_evidence=evidence, project_root=kwargs["project_root"])
    path = positive._path(kwargs["project_root"], kwargs["output_subdir"], identity)
    existing = positive._load(path, identity)
    if existing is not None:
        if not set(existing).issubset(kwargs["canonical_scores"]):
            raise ValueError("positive raw witness differs from certified score membership")
        return existing, path
    from services.audit.daily_trade40_oracle import truth_producer_fingerprint
    current_evidence = {**evidence, "truth_source_sha256": truth_producer_fingerprint("net_pnl")}
    return positive.load_or_build_positive_truth(**{**kwargs, "canonical_evidence": current_evidence})

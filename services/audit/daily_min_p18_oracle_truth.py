"""P18-only signed economic truth, delegating the canonical Daily Min engine.

Numerical outcomes and feature eligibility use the original Daily primitive;
P18 parameters, profile, cache namespace, Target and evidence stay independent.
No model FIT or Optimizer is invoked by this hindsight producer.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from core.breakout_quality_runtime import DAILY_MIN_P18_FULLCYCLE_NETPNL_TARGET_ID
from core.file_integrity import canonical_json_sha256, compute_file_sha256
from filters.breakout_quality.daily_min_p18_target import (
    SOURCE_ID, daily_min_p18_execution_sha256, verify_daily_min_p18_params,
)
from services.audit import true_daily_mintrade_oracle as daily_min
from services.audit.daily_trade40_oracle import DailyTrade40OracleBlockedError

TARGET_ID = DAILY_MIN_P18_FULLCYCLE_NETPNL_TARGET_ID
LABEL_ID = f"LABEL-{TARGET_ID}"
ENTRY_POLICY = daily_min.ENTRY_POLICY


@lru_cache(maxsize=1)
def truth_producer_fingerprint():
    return canonical_json_sha256({
        "target_id": TARGET_ID, "entry_policy": ENTRY_POLICY,
        "producer": compute_file_sha256(Path(__file__)),
        "p18_execution": daily_min_p18_execution_sha256(),
        "canonical_daily_min": daily_min.truth_producer_fingerprint(),
    })


def build_daily_min_p18_oracle_scores(*, output_subdir, canonical_schedule, **kwargs):
    if not canonical_schedule or any(p.source_id != SOURCE_ID for p in canonical_schedule):
        raise DailyTrade40OracleBlockedError("P18 Oracle requires its own signed parameter schedule")
    for period in canonical_schedule:
        verify_daily_min_p18_params(period.params)
    # This numerical producer already accepts the date-effective Min schedule.
    # Its private artifacts are isolated by namespace AND the P18 param/profile
    # identities. Preserve its raw witness rather than reconstructing NetPnL.
    scores, parent = daily_min.build_true_daily_universal_oracle_scores(
        output_subdir=f"{output_subdir}/daily_min_p18", canonical_schedule=canonical_schedule,
        **kwargs,
    )
    evidence = {
        **parent, "target_id": TARGET_ID, "label_id": LABEL_ID,
        "truth_source_sha256": truth_producer_fingerprint(),
        "economic_teacher_source_id": SOURCE_ID,
        "canonical_daily_evidence": parent,
    }
    return scores, evidence


def load_positive_truth(*, output_subdir, canonical_evidence, execution=None,
                        cache_root=None, **kwargs):
    effective = canonical_evidence.get("effective_source_reference")
    if effective:
        if execution is None or cache_root is None:
            raise DailyTrade40OracleBlockedError("P18 effective truth requires a signed input context")
        from services.audit.oracle_truth_references import OracleTruthReferences
        OracleTruthReferences(cache_root).validate_consumption(execution, canonical_evidence)
    if (canonical_evidence.get("target_id") != TARGET_ID
            or canonical_evidence.get("label_id") != LABEL_ID
            or canonical_evidence.get("economic_teacher_source_id") != SOURCE_ID
            or canonical_evidence.get("entry_policy") != ENTRY_POLICY
            or (not effective and canonical_evidence.get("truth_source_sha256") != truth_producer_fingerprint())):
        raise DailyTrade40OracleBlockedError("P18 Positive truth provenance mismatch")
    parent = canonical_evidence.get("canonical_daily_evidence")
    if not isinstance(parent, dict) or any(
        parent.get(key) != canonical_evidence.get(key) for key in (
            "profile_id", "entry_policy", "teacher", "rank_sha256", "raw_label_sha256",
            "calendar_sha256", "settled_net_pnl_rows", "dataset_summary_sha256",
        )
    ):
        raise DailyTrade40OracleBlockedError("P18 raw witness does not match signed economic truth")
    return daily_min.load_positive_truth(
        output_subdir=f"{output_subdir}/daily_min_p18", canonical_evidence=parent, **kwargs,
    )

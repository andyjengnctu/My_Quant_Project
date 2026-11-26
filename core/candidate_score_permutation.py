"""Decision-local randomization of observed candidate scores, never of signals.

The opt-in replay scope is deliberately separate from model training and PIT
stores. Original replays pay only a ContextVar lookup and retain their old path.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

POLICY = "same_day_raw_score_permutation_v1"
REPEATS = 99
OPTION = "candidate_score_permutation"
_ACTIVE = ContextVar("candidate_score_permutation", default=None)


def permutation_spec(options):
    spec = dict(options or {}).get(OPTION)
    if spec is None:
        return None
    if (not isinstance(spec, dict) or set(spec) != {"policy", "repeats", "control_arm_id"}
            or spec.get("policy") != POLICY or type(spec.get("repeats")) is not int
            or spec["repeats"] != REPEATS or not isinstance(spec.get("control_arm_id"), str)
            or not spec["control_arm_id"]):
        raise ValueError("Invalid fixed 99-repeat candidate-score permutation contract")
    return dict(spec)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _identity(row):
    ticker = str(row.get("ticker") or "").strip()
    if not ticker:
        raise ValueError("Candidate permutation requires canonical stock identity")
    return ticker


def permute_candidate_scores(rows, *, seed: int, repeat_id: int, decision_date):
    """Uniform Fisher-Yates including fixed points; order/scheduling independent.

    SHA256 counter/rejection sampling makes the protocol independent of Python's
    RNG version. The key contains neither selector, arm, future data nor scores.
    """
    from core.portfolio_entry_selection_common import _candidate_continuous_score
    if type(seed) is not int or type(repeat_id) is not int or not 1 <= repeat_id <= REPEATS:
        raise ValueError("Permutation requires an explicit benchmark seed and repeat 1..99")
    date = str(decision_date)[:10]
    if len(date) != 10:
        raise ValueError("Permutation requires a decision date")
    identities = [_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Canonical orderable candidate identity is ambiguous")
    valid = sorted((identity, index, _candidate_continuous_score(row))
                   for index, (identity, row) in enumerate(zip(identities, rows))
                   if _candidate_continuous_score(row) is not None)
    key = _digest([POLICY, seed, repeat_id, date, [item[0] for item in valid]])
    donors = list(range(len(valid)))
    counter = 0
    for index in range(len(donors) - 1, 0, -1):
        bound = index + 1
        limit = (1 << 256) - (1 << 256) % bound
        while True:
            value = int(hashlib.sha256(f"{key}:{counter}".encode()).hexdigest(), 16)
            counter += 1
            if value < limit:
                break
        other = value % bound
        donors[index], donors[other] = donors[other], donors[index]
    projected = list(rows)
    mapping = []
    for index, (identity, row_index, score) in enumerate(valid):
        donor, _, assigned = valid[donors[index]]
        row = dict(rows[row_index])
        row["breakout_quality_rank"] = dict(row["breakout_quality_rank"])
        row["breakout_quality_score"] = assigned
        if "score" in row["breakout_quality_rank"]:
            row["breakout_quality_rank"]["score"] = assigned
        projected[row_index] = row
        mapping.append({"recipient": identity, "donor": donor, "original_score": score,
                        "assigned_score": assigned})
    evidence = {"decision_date": date, "candidate_count": len(rows),
                "valid_score_count": len(valid), "missing_score_count": len(rows) - len(valid),
                "exchangeable": len({item[2] for item in valid}) > 1,
                "assignment_changed": any(item["original_score"] != item["assigned_score"]
                                          for item in mapping),
                "mapping_sha256": _digest(mapping), "mapping": mapping}
    return projected, evidence


@dataclass
class PermutationReplay:
    seed: int
    repeat_id: int
    evidence_file: Any
    counts: dict = field(default_factory=lambda: dict(
        decision_days=0, candidate_days=0, exchangeable_days=0, assignment_changed_days=0,
        selection_changed_days=0, candidate_count=0, valid_score_count=0, missing_score_count=0))

    def record(self, evidence, original_actions, permuted_actions):
        original = [_identity(row) for row in original_actions]
        permuted = [_identity(row) for row in permuted_actions]
        # Basket membership, not a harmless reordering of the same basket.
        changed = set(original) != set(permuted)
        evidence.update(original_selected=original, permuted_selected=permuted,
                        selection_changed=changed)
        counts = self.counts
        counts["decision_days"] += 1
        for key in ("candidate_count", "valid_score_count", "missing_score_count"):
            counts[key] += evidence[key]
        for key, condition in (("candidate_days", evidence["candidate_count"] > 0),
                               ("exchangeable_days", evidence["exchangeable"]),
                               ("assignment_changed_days", evidence["assignment_changed"]),
                               ("selection_changed_days", changed)):
            counts[key] += int(condition)
        self.evidence_file.write(json.dumps(evidence, ensure_ascii=False, allow_nan=False) + "\n")

    def summary(self):
        return {**self.counts, "selection_change_fraction": (
            self.counts["selection_changed_days"] / self.counts["candidate_days"]
            if self.counts["candidate_days"] else None)}


@contextmanager
def permutation_replay_scope(*, seed, repeat_id, evidence_file):
    state = PermutationReplay(int(seed), int(repeat_id), evidence_file)
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)


def active_permutation(params):
    return _ACTIVE.get() if bool(getattr(params, "use_breakout_quality_ranking", False)) else None


def prepare_permuted_selection(state, *, today, rows, reorder_callback,
                               select_action_callback, selector_kwargs):
    # Same-state shadow selection is diagnostic only: no orders/cash/history
    # are executed. Protect originals even if a future selector mutates its rows.
    original_rows, diagnostic = reorder_callback(deepcopy(rows), **selector_kwargs)
    original_actions = select_action_callback(original_rows, diagnostic)
    projected, evidence = permute_candidate_scores(
        rows, seed=state.seed, repeat_id=state.repeat_id, decision_date=today)
    return projected, evidence, original_actions

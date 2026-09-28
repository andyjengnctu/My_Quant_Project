from __future__ import annotations

import secrets
import random
from typing import Any

SEED_ENSEMBLE_MAX_SEED_VALUE = 2_147_483_647
SEED_ENSEMBLE_MIN_AGREE_AUTO = "auto"


def coerce_seed_ensemble_size(value: Any, *, default: int = 1) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    return max(1, int(resolved))


def resolve_seed_ensemble_min_agree(seed_count: Any, min_agree: Any = SEED_ENSEMBLE_MIN_AGREE_AUTO) -> int:
    n = coerce_seed_ensemble_size(seed_count, default=1)
    text = str(min_agree).strip().lower() if min_agree is not None else SEED_ENSEMBLE_MIN_AGREE_AUTO
    if text in {"", "none", "null", SEED_ENSEMBLE_MIN_AGREE_AUTO, "majority", "half_plus_one"}:
        requested = n // 2 + 1
    else:
        try:
            requested = int(min_agree)
        except (TypeError, ValueError):
            requested = n // 2 + 1
    return min(n, max(1, int(requested)))



def resolve_seed_ensemble_parallel_backend(parallel_backend: Any = "process") -> str:
    text = str(parallel_backend or "process").strip().lower()
    if text in {"process", "processes", "proc", "multiprocess", "multiprocessing"}:
        return "process"
    if text in {"thread", "threads", "threading"}:
        return "thread"
    return "process"


def resolve_seed_ensemble_parallel_workers(seed_count: Any, parallel_workers: Any = 1) -> int:
    n = coerce_seed_ensemble_size(seed_count, default=1)
    text = str(parallel_workers).strip().lower() if parallel_workers is not None else ""
    if text in {"auto", "max", "all", "n", "seed_count", "seeds", "seed"}:
        requested = n
    else:
        try:
            requested = int(parallel_workers)
        except (TypeError, ValueError):
            requested = 1
    return min(n, max(1, int(requested)))


def generate_random_seed_ensemble(seed_count: Any) -> list[int]:
    n = coerce_seed_ensemble_size(seed_count, default=1)
    seeds: set[int] = set()
    while len(seeds) < n:
        seeds.add(1 + secrets.randbelow(SEED_ENSEMBLE_MAX_SEED_VALUE))
    return sorted(int(seed) for seed in seeds)


def generate_deterministic_seed_ensemble(seed_count: Any, *, generator_seed: Any) -> list[int]:
    """Return a reproducible unique seed bundle for controlled Research fitting.

    Trading intentionally uses ``generate_random_seed_ensemble`` on each explicit
    retrain.  Research controlled comparisons may instead pin a generator seed so
    every arm/fold can replay the same stochastic realizations.
    """

    n = coerce_seed_ensemble_size(seed_count, default=1)
    try:
        seed0 = int(generator_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed ensemble generator_seed不合法: {generator_seed!r}") from exc
    if seed0 < 0:
        raise ValueError("seed ensemble generator_seed必須>=0")
    rng = random.Random(seed0)
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < n:
        value = int(rng.randrange(1, SEED_ENSEMBLE_MAX_SEED_VALUE))
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def resolve_seed_ensemble_seeds(policy: Any) -> list[int]:
    """Resolve actual optimizer seeds from one seed-ensemble policy contract.

    A policy with ``resolved_seeds`` is deterministic and must match its declared
    ``seed_count`` exactly.  Otherwise the historical random-per-training behavior
    is preserved.
    """

    payload = dict(policy or {})
    count = coerce_seed_ensemble_size(payload.get("seed_count", 1), default=1)
    raw_resolved = payload.get("resolved_seeds")
    if raw_resolved not in (None, ""):
        if not isinstance(raw_resolved, (list, tuple)):
            raise ValueError("seed ensemble resolved_seeds必須是list/tuple")
        seeds = [int(value) for value in raw_resolved]
        if len(seeds) != count:
            raise ValueError(
                "seed ensemble resolved_seeds數量與seed_count不一致: "
                f"count={count}, resolved={len(seeds)}"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("seed ensemble resolved_seeds不可重複")
        if any(seed < 1 or seed >= SEED_ENSEMBLE_MAX_SEED_VALUE for seed in seeds):
            raise ValueError("seed ensemble resolved_seeds超出合法範圍")
        return seeds
    return generate_random_seed_ensemble(count)


def build_seed_ensemble_policy_snapshot(*, enabled: Any, seed_count: Any, min_agree: Any) -> dict:
    n = coerce_seed_ensemble_size(seed_count, default=1)
    agree = resolve_seed_ensemble_min_agree(n, min_agree)
    return {
        "enabled": bool(enabled),
        "seed_count": int(n),
        "min_agree": int(agree),
        "min_agree_requested": min_agree,
        "min_agree_max": int(n),
        "min_agree_rule": "clamp_1_to_seed_count",
        "seed_mode": "random_n_each_training_run",
        "params_storage": "same_json",
    }


def normalize_seed_ensemble_members(members: Any) -> list[dict]:
    if not isinstance(members, list):
        return []
    normalized = []
    for idx, item in enumerate(members, start=1):
        if not isinstance(item, dict):
            continue
        params = item.get("params")
        if not isinstance(params, dict) or not params:
            continue
        member = dict(item)
        member["member_index"] = int(member.get("member_index") or idx)
        member["params"] = dict(params)
        normalized.append(member)
    return normalized


def renumber_seed_ensemble_members(members: Any) -> list[dict]:
    normalized = normalize_seed_ensemble_members(members)
    for idx, member in enumerate(normalized, start=1):
        member["member_index"] = int(idx)
    return normalized

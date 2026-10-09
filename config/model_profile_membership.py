"""Pure declaration-composition guard for configured (model_id, profile) pairs.

This config-level helper contains no scientific registry lookups or runtime policy;
actual research identity legality remains owned by core.breakout_quality_policy.
"""

from __future__ import annotations

def merge_breakout_quality_model_profiles(
    *groups: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Merge ordered model/profile groups while rejecting identity collisions.

    Exact duplicates are intentionally de-duplicated.  Reusing one MR id for a different
    profile, or one profile under a different MR id, is a configuration error and fails
    before any training/comparison workflow can run.
    """

    rows: list[tuple[str, str]] = []
    by_model_id: dict[str, str] = {}
    by_profile: dict[str, str] = {}
    for group in groups:
        for raw_model_id, raw_profile in group:
            model_id = str(raw_model_id).strip()
            profile = str(raw_profile).strip()
            if not model_id or not profile:
                raise ValueError("模型研究model id/profile不得為空")
            existing_profile = by_model_id.get(model_id)
            existing_model_id = by_profile.get(profile)
            if existing_profile is not None and existing_profile != profile:
                raise ValueError(
                    f"模型研究model id重複綁定不同profile: {model_id} -> "
                    f"{existing_profile!r} / {profile!r}"
                )
            if existing_model_id is not None and existing_model_id != model_id:
                raise ValueError(
                    f"模型研究profile重複綁定不同model id: {profile} -> "
                    f"{existing_model_id!r} / {model_id!r}"
                )
            if existing_profile is not None:
                continue
            by_model_id[model_id] = profile
            by_profile[profile] = model_id
            rows.append((model_id, profile))
    return tuple(rows)

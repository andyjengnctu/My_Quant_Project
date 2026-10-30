"""Pure canonical FIT identity projection for a verified MinTrade40 teacher.

Only a verified canonical Optimizer source may supply period records. This
projection binds what could affect labels at a particular FIT cutoff, while the
original complete-file hashes remain in separate audit provenance.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from core.file_integrity import canonical_json_sha256


def mintrade40_teacher_asof_sha256(
    *, seed: int, records: Sequence[Mapping[str, Any]], fit_cutoff: Any,
    oos_start_date: str | None = None,
) -> str:
    if type(seed) is not int or seed <= 0 or not records:
        raise ValueError("MinTrade40 fold teacher identity requires canonical seed/records")
    cutoff = pd.Timestamp(fit_cutoff).normalize()
    if pd.isna(cutoff):
        raise ValueError("MinTrade40 fold cutoff invalid")
    previous_end = None
    projected = []
    for record in records:
        start = pd.Timestamp(record['start']).normalize()
        end = None if record.get('end') is None else pd.Timestamp(record['end']).normalize()
        pub = pd.Timestamp(record['published']).normalize()
        signature = str(record['params_signature'])
        if (pd.isna(start) or pd.isna(pub) or (end is not None and (pd.isna(end) or end < start))
                or pub > start or not signature or (previous_end is not None and start <= previous_end)):
            raise ValueError("MinTrade40 teacher period lineage invalid")
        if projected and previous_end is None:
            raise ValueError("MinTrade40 teacher has open-ended middle period")
        previous_end = end
        if start > cutoff:
            continue
        projected.append({
            'start': str(start.date()),
            'end': str(min(end, cutoff).date() if end is not None else cutoff.date()),
            'params_signature': signature,
            'published': str(pub.date()),
        })
    if not projected:
        raise ValueError("MinTrade40 FIT cutoff has no verified historical teacher")
    boundary = None if oos_start_date is None else pd.Timestamp(oos_start_date).normalize()
    if boundary is not None and pd.isna(boundary):
        raise ValueError("MinTrade40 OOS state boundary invalid")
    return canonical_json_sha256({
        'type': 'formal_mintrade40_fold_teacher_asof_v1',
        'seed': seed, 'fit_cutoff': str(cutoff.date()),
        'oos_signal_reset': (str(boundary.date()) if boundary is not None and boundary <= cutoff else None),
        'periods': projected,
    })

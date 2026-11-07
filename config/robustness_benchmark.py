"""Immutable benchmark registry with one active four-seed Research cohort.

The registered v1 (4S) and v2 (8S) scientific identities remain immutable.
Oracle and Model/Strategy Robustness both consume v1; v2 remains available
for historical signed per-seed provenance, never for current aggregates.
"""

from config.training_policy import ROBUSTNESS_BENCHMARK_SPECS as _HISTORICAL_SPECS

ROBUSTNESS_BENCHMARK_SPECS = {
    **_HISTORICAL_SPECS,
    "end_to_end_v2": {
        "seed_count": 8,
        "seed_generator_seed": 20260810,
        "identity_fingerprint": "70652f5df5052510e49a9d44233fdc7d220154b0851d9701134a227481f22e6c",
    },
}
# One current benchmark; the signed v2 registry entry is historical only.
ROBUSTNESS_BENCHMARK_ID = "end_to_end_v1"
MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID = ROBUSTNESS_BENCHMARK_ID
TARGET_ORACLE_ROBUSTNESS_BENCHMARK_ID = ROBUSTNESS_BENCHMARK_ID
ROBUSTNESS_BENCHMARK_SEED_COUNT = int(ROBUSTNESS_BENCHMARK_SPECS[ROBUSTNESS_BENCHMARK_ID]["seed_count"])
ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED = int(
    ROBUSTNESS_BENCHMARK_SPECS[ROBUSTNESS_BENCHMARK_ID]["seed_generator_seed"]
)

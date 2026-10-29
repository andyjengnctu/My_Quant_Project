"""Sole active end-to-end benchmark selection and append-only seed registry.

The historical v1 registration is imported (not redefined) from its original
signed source file.  That file is deliberately left unchanged: it belongs to
an already-published Oracle daily-truth source fingerprint.  No runtime
consumer reads its obsolete ``ROBUSTNESS_BENCHMARK_ID`` pointer.

Benchmark N is an aggregation scope; each fitted per-seed parameter result
retains its immutable original artifact provenance when legally reused.
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
ROBUSTNESS_BENCHMARK_ID = "end_to_end_v2"
ROBUSTNESS_BENCHMARK_SEED_COUNT = int(ROBUSTNESS_BENCHMARK_SPECS[ROBUSTNESS_BENCHMARK_ID]["seed_count"])
ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED = int(
    ROBUSTNESS_BENCHMARK_SPECS[ROBUSTNESS_BENCHMARK_ID]["seed_generator_seed"]
)

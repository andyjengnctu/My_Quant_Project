"""One immutable benchmark registry, separate selection per research workflow.

The historical v1 registration is imported unchanged from its original
signed source file. Model/Strategy robustness select v1 (4 seeds) while Target
Oracle selects v2 (8 seeds). Both resolve the same registry; no second seed
sequence or fitting identity is created. ``ROBUSTNESS_BENCHMARK_ID`` remains
v2 for existing Oracle/Optimizer consumers that use its canonical default.
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
# AI: The Oracle's 8-seed aggregate is independent of the original
# Model/Strategy robustness 4-seed aggregate. Both reuse the same immutable
# registered seed identities, never a second seed list or generator.
MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID = "end_to_end_v1"
ROBUSTNESS_BENCHMARK_ID = "end_to_end_v2"  # Oracle/default v2; preserve existing P16 source
TARGET_ORACLE_ROBUSTNESS_BENCHMARK_ID = ROBUSTNESS_BENCHMARK_ID
ROBUSTNESS_BENCHMARK_SEED_COUNT = int(ROBUSTNESS_BENCHMARK_SPECS[ROBUSTNESS_BENCHMARK_ID]["seed_count"])
ROBUSTNESS_BENCHMARK_SEED_GENERATOR_SEED = int(
    ROBUSTNESS_BENCHMARK_SPECS[ROBUSTNESS_BENCHMARK_ID]["seed_generator_seed"]
)

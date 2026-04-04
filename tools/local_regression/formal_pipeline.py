from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FormalStepSpec:
    name: str
    command: str
    summary_file: str
    requires_dataset: bool = True


FORMAL_SINGLE_ENTRY = "apps/test_suite.py"

FORMAL_STEP_SPECS: Tuple[FormalStepSpec, ...] = (
    FormalStepSpec("quick_gate", "tools/local_regression/run_quick_gate.py", "quick_gate_summary.json", True),
    FormalStepSpec("consistency", "tools/validate/cli.py --dataset reduced", "validate_consistency_summary.json", True),
    FormalStepSpec("chain_checks", "tools/local_regression/run_chain_checks.py", "chain_summary.json", True),
    FormalStepSpec("ml_smoke", "tools/local_regression/run_ml_smoke.py", "ml_smoke_summary.json", True),
    FormalStepSpec("meta_quality", "tools/local_regression/run_meta_quality.py", "meta_quality_summary.json", False),
)

FORMAL_STEP_ORDER: Tuple[str, ...] = tuple(spec.name for spec in FORMAL_STEP_SPECS)
FORMAL_COMMAND_ORDER: Tuple[tuple[str, str, str], ...] = tuple(
    (spec.name, spec.command, spec.summary_file) for spec in FORMAL_STEP_SPECS
)
DATASET_REQUIRED_STEPS = {spec.name for spec in FORMAL_STEP_SPECS if spec.requires_dataset}
FORMAL_COMMANDS: Tuple[str, ...] = tuple(spec.command for spec in FORMAL_STEP_SPECS)

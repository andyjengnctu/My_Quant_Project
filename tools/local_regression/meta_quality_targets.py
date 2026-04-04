from __future__ import annotations

CORE_TRADING_COVERAGE_TARGETS = [
    "core/backtest_core.py",
    "core/backtest_finalize.py",
    "core/portfolio_engine.py",
    "core/position_step.py",
    "core/portfolio_entries.py",
    "core/portfolio_exits.py",
    "core/portfolio_ops.py",
    "core/trade_plans.py",
    "core/entry_plans.py",
    "core/portfolio_candidates.py",
    "core/portfolio_fast_data.py",
    "core/extended_signals.py",
    "core/signal_utils.py",
]
ENTRY_PATH_CRITICAL_COVERAGE_TARGETS = [
    "core/portfolio_entries.py",
    "core/entry_plans.py",
]
CRITICAL_COVERAGE_TARGETS = [
    "core/backtest_core.py",
    "core/backtest_finalize.py",
    "core/portfolio_engine.py",
    "core/position_step.py",
    "core/portfolio_exits.py",
    *ENTRY_PATH_CRITICAL_COVERAGE_TARGETS,
]
TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS = [
    "tools/local_regression/common.py",
    "tools/local_regression/formal_pipeline.py",
    "tools/local_regression/meta_quality_targets.py",
    "tools/local_regression/meta_quality_coverage.py",
    "tools/local_regression/run_meta_quality.py",
    "tools/local_regression/run_all.py",
    "tools/validate/preflight_env.py",
    "core/test_suite_reporting.py",
    "apps/test_suite.py",
]
FORMAL_STEP_ENTRY_COVERAGE_TARGETS = [
    "tools/local_regression/run_quick_gate.py",
    "tools/validate/cli.py",
]
FORMAL_STEP_IMPLEMENTATION_COVERAGE_TARGETS = [
    "tools/validate/main.py",
]
COVERAGE_TARGETS = list(dict.fromkeys([
    "tools/validate/synthetic_cases.py",
    "tools/validate/synthetic_meta_cases.py",
    "tools/validate/synthetic_unit_cases.py",
    "tools/validate/synthetic_history_cases.py",
    "tools/validate/synthetic_flow_cases.py",
    "tools/validate/synthetic_take_profit_cases.py",
    "tools/validate/synthetic_contract_cases.py",
    "tools/validate/synthetic_guardrail_cases.py",
    "tools/validate/synthetic_display_cases.py",
    "tools/validate/synthetic_reporting_cases.py",
    "tools/validate/synthetic_error_cases.py",
    "tools/validate/synthetic_data_quality_cases.py",
    "tools/validate/synthetic_cli_cases.py",
    "tools/validate/synthetic_strategy_cases.py",
    "tools/validate/synthetic_regression_cases.py",
    "tools/local_regression/run_chain_checks.py",
    "tools/local_regression/run_ml_smoke.py",
    *FORMAL_STEP_ENTRY_COVERAGE_TARGETS,
    *FORMAL_STEP_IMPLEMENTATION_COVERAGE_TARGETS,
    *TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS,
    "tools/validate/reporting.py",
    "tools/portfolio_sim/reporting.py",
    "core/scanner_display.py",
    "core/strategy_dashboard.py",
    "core/display_common.py",
    "core/price_utils.py",
    "core/history_filters.py",
    "core/portfolio_stats.py",
    *CORE_TRADING_COVERAGE_TARGETS,
]))
COVERAGE_LINE_MIN_FLOOR = 55.0
COVERAGE_BRANCH_MIN_FLOOR = 50.0
CRITICAL_COVERAGE_LINE_MIN_FLOOR = 30.0
CRITICAL_COVERAGE_BRANCH_MIN_FLOOR = 25.0
COVERAGE_MAX_LINE_BRANCH_GAP = 5.0

__all__ = [
    "CORE_TRADING_COVERAGE_TARGETS",
    "ENTRY_PATH_CRITICAL_COVERAGE_TARGETS",
    "CRITICAL_COVERAGE_TARGETS",
    "TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS",
    "FORMAL_STEP_ENTRY_COVERAGE_TARGETS",
    "FORMAL_STEP_IMPLEMENTATION_COVERAGE_TARGETS",
    "COVERAGE_TARGETS",
    "COVERAGE_LINE_MIN_FLOOR",
    "COVERAGE_BRANCH_MIN_FLOOR",
    "CRITICAL_COVERAGE_LINE_MIN_FLOOR",
    "CRITICAL_COVERAGE_BRANCH_MIN_FLOOR",
    "COVERAGE_MAX_LINE_BRANCH_GAP",
]

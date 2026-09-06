from __future__ import annotations

CORE_TRADING_COVERAGE_TARGETS = [
    "core/backtest_core.py",
    "core/backtest_finalize.py",
    "core/portfolio_engine.py",
    "core/portfolio_benchmark.py",
    "core/portfolio_replay_support.py",
    "core/portfolio_levels.py",
    "core/portfolio_ensemble.py",
    "core/position_step.py",
    "core/exit_priority.py",
    "core/event_hash_chain.py",
    "core/trading_identity.py",
    "core/trading_state_paths.py",
    "core/trading_account_state.py",
    "core/trading_order_state.py",
    "core/trading_tp_progress.py",
    "core/trading_stop_exit_progress.py",
    "core/trading_fill_transaction.py",
    "services/trading/account_state.py",
    "services/trading/order_state.py",
    "services/trading/fill_reconciliation.py",
    "services/trading/daily_workflow.py",
    "services/trading/entry_order_submission.py",
    "services/trading/order_planning.py",
    "services/trading/proposed_order_state.py",
    "services/trading/position_market_context.py",
    "services/trading/position_rollforward.py",
    "services/trading/indicator_exit_planning.py",
    "services/trading/indicator_exit_order_submission.py",
    "services/trading/protection_planning.py",
    "services/trading/protection_order_submission.py",
    "services/trading/operations_status.py",
    "services/trading/operational_audit.py",
    "core/portfolio_entries.py",
    "core/portfolio_entry_plans.py",
    "core/portfolio_entry_selection.py",
    "core/portfolio_entry_selection_common.py",
    "core/portfolio_entry_selection_max_dl.py",
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
    "core/portfolio_entry_plans.py",
    "core/portfolio_entry_selection.py",
    "core/portfolio_entry_selection_common.py",
    "core/portfolio_entry_selection_max_dl.py",
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
    "tools/local_regression/checklist_contract.py",
    "tools/local_regression/formal_pipeline.py",
    "tools/local_regression/formal_wall_time.py",
    "tools/local_regression/meta_quality_targets.py",
    "tools/local_regression/meta_quality_coverage.py",
    "tools/local_regression/run_meta_quality.py",
    "tools/local_regression/run_all.py",
    "tools/validate/preflight_env.py",
    "core/test_suite_reporting.py",
    "apps/test_suite.py",
]
POLICY_CONTRACT_COVERAGE_TARGETS = [
    "core/capital_policy.py",
    "core/selection_policy.py",
    "core/strategy_params.py",
    "core/params_io.py",
    "config/execution_policy.py",
    "core/execution_policy.py",
    "config/training_policy.py",
    "core/training_policy.py",
    "config/training_performance_policy.py",
    "core/training_performance.py",
    "config/display_policy.py",
    "core/display_policy.py",
    "config/research.py",
    "core/research_policy.py",
    "config/trading.py",
    "core/trading_policy.py",
    "core/runtime_domains.py",
    "core/dataset_dates.py",
    "core/trading_market_clock.py",
    "core/trading_capabilities.py",
    "services/downloader/application.py",
    "services/downloader/universe.py",
    "services/downloader/sync.py",
    "services/trading/strategy_param_training.py",
    "config/downloader.py",
    "config/runtime.py",
    "config/strategy_compare.py",
    "core/strategy_compare_registry.py",
    "core/strategy_compare_policy.py",
    "config/audit.py",
    "core/audit_registry.py",
    "core/audit_policy.py",
    "config/breakout_policy.py",
    "core/breakout_policy.py",
]
FORMAL_STEP_ENTRY_COVERAGE_TARGETS = [
    "tools/local_regression/run_quick_gate.py",
    "tools/validate/cli.py",
]
FORMAL_STEP_IMPLEMENTATION_COVERAGE_TARGETS = [
    "tools/validate/main.py",
]
# Formal coverage measures production/runtime and formal-runner code exercised by
# synthetic cases.  The validator implementations themselves are correctness
# drivers, not coverage subjects; tracing them adds substantial line-event cost
# without improving production-risk coverage.
COVERAGE_TARGETS = list(dict.fromkeys([
    "core/research_orchestration.py",
    "services/research/artifact_orchestrator.py",
    "tools/local_regression/run_chain_checks.py",
    "tools/local_regression/run_ml_smoke.py",
    *FORMAL_STEP_ENTRY_COVERAGE_TARGETS,
    *FORMAL_STEP_IMPLEMENTATION_COVERAGE_TARGETS,
    *TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS,
    "tools/validate/reporting.py",
    "services/portfolio_sim/reporting.py",
    "core/scanner_display.py",
    "core/strategy_dashboard.py",
    "core/display_common.py",
    "core/price_utils.py",
    "core/history_filters.py",
    "core/portfolio_stats.py",
    *CORE_TRADING_COVERAGE_TARGETS,
    *POLICY_CONTRACT_COVERAGE_TARGETS,
]))
COVERAGE_LINE_MIN_FLOOR = 55.0
COVERAGE_BRANCH_MIN_FLOOR = 50.0
CRITICAL_COVERAGE_LINE_MIN_FLOOR = 30.0
CRITICAL_COVERAGE_BRANCH_MIN_FLOOR = 25.0
COVERAGE_MAX_LINE_BRANCH_GAP = 5.0


def build_coverage_include_paths(project_root) -> list[str]:
    """Return exact absolute files instrumented by the formal coverage run."""

    from pathlib import Path

    root = Path(project_root).resolve()
    return [str((root / rel_path).resolve()) for rel_path in COVERAGE_TARGETS]


__all__ = [
    "CORE_TRADING_COVERAGE_TARGETS",
    "ENTRY_PATH_CRITICAL_COVERAGE_TARGETS",
    "CRITICAL_COVERAGE_TARGETS",
    "TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS",
    "POLICY_CONTRACT_COVERAGE_TARGETS",
    "FORMAL_STEP_ENTRY_COVERAGE_TARGETS",
    "FORMAL_STEP_IMPLEMENTATION_COVERAGE_TARGETS",
    "COVERAGE_TARGETS",
    "COVERAGE_LINE_MIN_FLOOR",
    "COVERAGE_BRANCH_MIN_FLOOR",
    "CRITICAL_COVERAGE_LINE_MIN_FLOOR",
    "CRITICAL_COVERAGE_BRANCH_MIN_FLOOR",
    "COVERAGE_MAX_LINE_BRANCH_GAP",
    "build_coverage_include_paths",
]

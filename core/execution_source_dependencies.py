"""Declarative executable-source dependency graph shared by Replay and Truth.

Scopes are capabilities, never research/model identities. Ordered expansion keeps
existing producer inventories stable; common numerical dependencies have one owner.
Registration-only metadata is projected separately from executable implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import ast
from copy import deepcopy


@dataclass(frozen=True)
class ExecutionSourceCapability:
    owners: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()


_TRADE_NUMERICS = ('core/position_step.py',
 'core/position_management.py',
 'core/exit_priority.py',
 'core/entry_plans.py',
 'core/extended_signals.py',
 'core/order_lot_policy.py',
 'core/exact_accounting.py',
 'core/fee_rebate.py',
 'core/price_utils.py',
 'core/selection_policy.py',
 'core/signal_utils.py',
 'core/breakout_false_filter.py',
 'core/capital_policy.py')

_PORTFOLIO_CANDIDATES = ('core/portfolio_engine.py',
 'core/portfolio_day_candidates.py',
 'core/portfolio_candidate_admission.py',
 'core/portfolio_candidates.py',
 'core/portfolio_day_context.py',
 'core/portfolio_day_events.py',
 'core/portfolio_replay_support.py',
 'core/portfolio_benchmark.py',
 'core/portfolio_day_equity.py',
 'core/portfolio_day_signal_lifecycle.py',
 'core/portfolio_capacity_rows.py',
 'core/portfolio_terminal_equity.py',
 'core/portfolio_terminal_statistics.py',
 'core/portfolio_terminal_reporting.py',
 'core/portfolio_run_result.py',
 'core/portfolio_entries.py',
 'core/portfolio_exits.py',
 'core/portfolio_entry_selection.py',
 'core/portfolio_entry_selection_common.py',
 'core/portfolio_entry_selection_max_dl.py',
 'core/portfolio_ensemble.py',
 'core/portfolio_entry_plans.py',
 'core/portfolio_entry_policy.py',
 'core/portfolio_fast_data.py',
 'core/portfolio_stats.py',
 'core/portfolio_fast_access.py',
 'core/ranking_policy.py',
 'core/portfolio_run_state.py',
 'core/buy_sort.py',
 'core/backtest_core.py',
 'core/backtest_finalize.py',
 'core/history_filters.py',
 'services/portfolio_replay.py',
 'services/portfolio_replay_runtime.py')

_DAILY_PRICES = ('filters/breakout_quality/daily_ranker_data.py',
 'filters/breakout_quality/daily_model_input_context.py',
 'filters/breakout_quality/daily_sample_accumulator.py',
 'filters/breakout_quality/continuous_ranker_data.py',
 'filters/breakout_quality/continuous_target.py',
 'filters/breakout_quality/mfe_safety_geometry.py',
 'filters/breakout_quality/profile_ranker_data.py',
 'filters/breakout_quality/workflow_io.py',
 'filters/breakout_quality/csv_io.py',
 'filters/breakout_quality/splits.py',
 'filters/breakout_quality/dataset_store.py',
 'filters/breakout_quality/dataset_readiness.py',
 'filters/breakout_quality/contract.py',
 'filters/breakout_quality/features.py',
 'filters/breakout_quality/models/architectures.py',
 'filters/breakout_quality/models/spec.py',
 'filters/breakout_quality/models/spec_contract.py',
 'filters/breakout_quality/models/spec_registry.py',
 'filters/breakout_quality/models/spec_builders.py',
 'core/data_utils.py',
 'core/breakout_quality_registry.py',
 'core/breakout_quality_policy.py',
 'core/breakout_quality_runtime.py',
 'core/breakout_quality_runtime_resolver.py',
 'core/dataset_profiles.py',
 'core/walk_forward_policy.py',
 'config/breakout_quality.py',
 'config/research.py',
 'config/training_policy.py',
 'config/market_data.py')

_MINTRADE_TRUTH = ('services/audit/daily_trade40_oracle.py',
 'filters/breakout_quality/daily_trade40_target.py',
 'filters/breakout_quality/trade_path_label.py',
 'filters/breakout_quality/runtime.py',
 'core/backtest_finalize.py',
 'config/trading.py')

_COMMON_RUNTIME = (
    "core/native_optimizer_hotpath.py", "core/native_optimizer_hotpath.c",
    "core/training_performance.py", "config/training_performance_policy.py",
    "config/broker_accounting.py", "config/execution_policy.py",
    "core/feature_bank.py", "core/strategy_params.py", "strategies/breakout/schema.py",
    "core/breakout_quality_runtime_port.py", "core/seed_ensemble_policy.py",
    "config/breakout_policy.py",
)
_PARAMETER_RUNTIME = (
    "core/active_param_ensemble.py", "core/rolling_oos_params.py", "core/params_io.py",
    "core/portfolio_param_runtime.py", "core/param_access.py", "core/raw_universe_contract.py",
    "core/file_integrity.py", "core/scientific_runtime_context.py",
)
_PORTFOLIO_SUPPORT = (
    "core/breakout_reentry.py", "core/breakout_quality_ranking_context.py",
    "core/positive_score_admission.py", "core/positive_netpnl_contract.py",
    "core/config.py", "core/training_policy.py", "config/training_policy.py",
    "core/walk_forward_policy.py", "core/trade_plans.py", "core/data_utils.py",
)
SOURCE_CAPABILITIES = MappingProxyType({
    "trade_numerics": ExecutionSourceCapability(_TRADE_NUMERICS),
    "numerical_runtime": ExecutionSourceCapability(_COMMON_RUNTIME),
    "parameter_runtime": ExecutionSourceCapability(_PARAMETER_RUNTIME),
    "portfolio_support": ExecutionSourceCapability(_PORTFOLIO_SUPPORT),
    "portfolio_runtime": ExecutionSourceCapability(_PORTFOLIO_CANDIDATES),
    "replay": ExecutionSourceCapability(dependencies=("trade_numerics", "numerical_runtime", "parameter_runtime", "portfolio_runtime", "portfolio_support")),
    "portfolio_candidates": ExecutionSourceCapability(dependencies=("portfolio_runtime", "trade_numerics", "numerical_runtime", "parameter_runtime", "portfolio_support")),
    "daily_common_filters": ExecutionSourceCapability(("core/daily_common_entry_filters.py",)),
    "daily_prices": ExecutionSourceCapability(_DAILY_PRICES),
    "mintrade_truth": ExecutionSourceCapability(_MINTRADE_TRUTH, ("trade_numerics", "numerical_runtime", "parameter_runtime")),
    "fulltrade_truth": ExecutionSourceCapability((
        "services/audit/fulltrade_oracle_truth.py",
        "services/breakout_quality/formal_fulltrade_training_source.py",
        "services/optimizer/strategy_param_service.py",
        "services/optimizer/strategy_param_training.py",
        "services/optimizer/strategy_param_contracts.py",
    )),
})


def execution_source_inventory(*capabilities: str) -> tuple[str, ...]:
    """Expand explicit dependencies in stable order, rejecting cycles/unknown scopes."""
    owners = {}
    visited = set()
    pending = set()
    def visit(name):
        if name in pending:
            raise ValueError(f"Cyclic execution source capability: {name}")
        if name in visited:
            return
        spec = SOURCE_CAPABILITIES.get(name)
        if spec is None:
            raise ValueError(f"Unknown execution source capability: {name}")
        pending.add(name)
        for relative in spec.owners:
            if (not isinstance(relative, str) or relative.startswith(("/", "\\"))
                    or ":" in relative or ".." in relative.split("/")):
                raise ValueError("Execution sources must be project-relative paths")
            owners[relative] = None
        for dependency in spec.dependencies:
            visit(dependency)
        pending.remove(name)
        visited.add(name)
    for capability in capabilities:
        visit(capability)
    return tuple(owners)


def project_ranking_policy_source(module: ast.Module, policy_id: str | None) -> ast.Module:
    """Keep executable registry machinery and only the consumed descriptor.

    Resolve literal IDs from the supplied source, never from an already imported
    module. A mutation probe or alternate checkout therefore cannot inherit the
    live process's metadata. Unknown declarative shapes remain strict.
    """
    result = deepcopy(module)
    constants = {}
    declaration = None
    for node in result.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == "RANKING_POLICY_SPECS":
            declaration = node
        try:
            constants[name] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            # Nonliteral bindings are not safe descriptor constants. Retain
            # their source and let descriptor validation fail closed below.
            continue
    if declaration is None or not isinstance(declaration.value, ast.Tuple):
        return result
    entries = []
    identifiers = set()
    def declarative_value(value):
        if isinstance(value, ast.Name):
            return value.id in constants
        if isinstance(value, (ast.Tuple, ast.List)):
            return all(declarative_value(item) for item in value.elts)
        try:
            ast.literal_eval(value)
        except (ValueError, TypeError):
            return False
        return True
    for item in declaration.value.elts:
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name) or item.func.id != "RankingPolicySpec":
            return result
        if (item.args or any(kw.arg is None or not declarative_value(kw.value) for kw in item.keywords)
                or len({kw.arg for kw in item.keywords}) != len(item.keywords)):
            # An unused declaration may still execute import-time code. Only
            # pure data additions qualify for capability-local projection.
            return result
        key = next((kw.value for kw in item.keywords if kw.arg == "policy_id"), None)
        value = constants.get(key.id) if isinstance(key, ast.Name) else (key.value if isinstance(key, ast.Constant) else None)
        if not isinstance(value, str) or value in identifiers:
            return result
        identifiers.add(value)
        if value == policy_id:
            entries.append(item)
    for node in result.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Dict)):
            for key, value_node in zip(node.value.keys, node.value.values):
                value = constants.get(key.id) if isinstance(key, ast.Name) else (key.value if isinstance(key, ast.Constant) else None)
                registry_field = (
                    isinstance(value_node, ast.Attribute) and value_node.attr == "score_order"
                    and isinstance(value_node.value, ast.Subscript)
                    and isinstance(value_node.value.value, ast.Name)
                    and value_node.value.value.id == "_POLICY_BY_ID"
                    and ast.dump(value_node.value.slice) == ast.dump(key)
                )
                if value in identifiers and value != policy_id and not (declarative_value(value_node) or registry_field):
                    # Dropping an unused keyed value must not hide a helper call
                    # that executes while the registry module is imported.
                    # The known score_order property and registry constructor
                    # remain in the projected source, so their edits invalidate.
                    return result
    declaration.value.elts = entries
    kept = []
    for node in result.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = constants.get(node.targets[0].id)
            if isinstance(value, str) and value in identifiers and value != policy_id:
                continue
            if isinstance(node.value, ast.Dict):
                selected = []
                for key, value_node in zip(node.value.keys, node.value.values):
                    value = constants.get(key.id) if isinstance(key, ast.Name) else (key.value if isinstance(key, ast.Constant) else None)
                    if value not in identifiers or value == policy_id:
                        selected.append((key, value_node))
                node.value.keys = [key for key, _ in selected]
                node.value.values = [value for _, value in selected]
        kept.append(node)
    result.body = kept
    return result

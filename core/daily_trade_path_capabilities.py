"""Declarative extensions to the canonical full-cycle Daily truth capability.

Consumers resolve producer, source identity and numerical owners here instead
of introducing experiment-specific dispatch into every Oracle stage.
"""
from core.breakout_quality_runtime import (
    DAILY_MIN_P18_FULLCYCLE_NETPNL_TARGET_ID, DAILY_MIN_P18_SOURCE_ID, DAILY_MIN_P18_TRADE_PATH_KIND,
    DAILY_FULL_P20_FULLCYCLE_NETPNL_TARGET_ID, DAILY_FULL_P20_SOURCE_ID,
    DAILY_FULL_P20_TRADE_PATH_KIND, DAILY_FULL_P20_TEACHER_CONTRACT,
)

DAILY_TRUTH_CAPABILITIES = {
    "daily_trade_path_daily_full_p20_net_pnl": {
        "target_id": DAILY_FULL_P20_FULLCYCLE_NETPNL_TARGET_ID,
        "target_kind": DAILY_FULL_P20_TRADE_PATH_KIND,
        "teacher_contract": DAILY_FULL_P20_TEACHER_CONTRACT,
        "outcome_metric": "net_pnl", "calculation_revision": 3,
        "entry_policy": "next_exchange_session_daily_full_p20_atr_limit_v1",
        "source_id": DAILY_FULL_P20_SOURCE_ID,
        "positive_output": "netpnl_positive_probability_v1",
        "producer_module": "services.audit.daily_full_p20_oracle_truth",
        "builder": "build_daily_full_p20_oracle_scores",
        "record_validator": ("services.breakout_quality.daily_full_p20_training_source",
                             "validate_teacher_records"),
        "calculation_owners": (
            ('daily_full_entry', 'filters/breakout_quality/daily_full_p20_target.py', 'compute_daily_full_p20_batch'),
            ('daily_full_filters', 'core/daily_common_entry_filters.py', 'daily_common_filter_signals'),
            ('stock_outcomes', 'services/audit/daily_full_p20_oracle_truth.py', '_stock_outcome'),
            ('truth_cohort', 'services/audit/daily_full_p20_oracle_truth.py', 'build_daily_full_p20_oracle_scores'),
            ('rank_outcomes', 'filters/breakout_quality/daily_trade40_target.py', 'rank_trade40_outcome_frame'),
            ('schedule', 'filters/breakout_quality/daily_trade40_target.py', 'build_formal_min_execution_schedule'),
            ('active_period', 'filters/breakout_quality/daily_trade40_target.py', 'resolve_historical_min_period'),
            ('broker_path', 'filters/breakout_quality/trade_path_label.py', 'simulate_realized_trade_path_label'),
        ),
    },

    "daily_trade_path_daily_min_p18_net_pnl": {
        "target_id": DAILY_MIN_P18_FULLCYCLE_NETPNL_TARGET_ID,
        "target_kind": DAILY_MIN_P18_TRADE_PATH_KIND,
        "teacher_contract": "daily_min_p18_historical_plus_current_v1",
        "outcome_metric": "net_pnl", "calculation_revision": 1,
        "entry_policy": "next_exchange_session_information_close_limit_v1",
        "source_id": DAILY_MIN_P18_SOURCE_ID,
        "positive_output": "netpnl_positive_probability_v1",
        "producer_module": "services.audit.daily_min_p18_oracle_truth",
        "builder": "build_daily_min_p18_oracle_scores",
        "calculation_owners": (
            ("p18_truth", "services/audit/daily_min_p18_oracle_truth.py", "build_daily_min_p18_oracle_scores"),
            ("p18_parameters", "filters/breakout_quality/daily_min_p18_target.py", "verify_daily_min_p18_params"),
            ("stock_outcomes", "services/audit/true_daily_mintrade_oracle.py", "_stock_outcome"),
            ("truth_cohort", "services/audit/true_daily_mintrade_oracle.py", "build_true_daily_universal_oracle_scores"),
            ("daily_min_entry", "filters/breakout_quality/true_daily_mintrade_target.py", "compute_true_daily_mintrade_batch"),
            ("rank_outcomes", "filters/breakout_quality/daily_trade40_target.py", "rank_trade40_outcome_frame"),
            ("schedule", "filters/breakout_quality/daily_trade40_target.py", "build_formal_min_execution_schedule"),
            ("active_period", "filters/breakout_quality/daily_trade40_target.py", "resolve_historical_min_period"),
            ("broker_path", "filters/breakout_quality/trade_path_label.py", "simulate_realized_trade_path_label"),
        ),
    },
}


def daily_truth_capability(recipe):
    return DAILY_TRUTH_CAPABILITIES.get(str(recipe))


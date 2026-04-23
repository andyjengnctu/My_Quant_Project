import csv
import json
import os
import time

from core.display import C_CYAN, C_GRAY, C_RESET

PROFILE_FIELDS = [
    "trial_number", "objective_mode", "search_train_end_year", "search_train_date_count", "objective_wall_sec", "prep_wall_sec", "prep_worker_total_sum_sec",
    "prep_worker_copy_sum_sec", "prep_worker_generate_signals_sum_sec", "prep_worker_assign_sum_sec",
    "prep_worker_run_backtest_sum_sec", "prep_worker_to_dict_sum_sec", "prep_ok_count",
    "prep_fail_count", "prep_avg_per_ok_sec", "sort_dates_sec", "portfolio_wall_sec",
    "portfolio_total_sec", "portfolio_ticker_dates_sec", "portfolio_build_trade_index_sec",
    "portfolio_day_loop_sec", "portfolio_candidate_scan_sec", "portfolio_rotation_sec",
    "portfolio_settle_sec", "portfolio_buy_sec", "portfolio_equity_mark_sec",
    "portfolio_closeout_sec", "portfolio_curve_stats_sec", "filter_rules_sec",
    "score_calc_sec", "ret_pct", "mdd", "trade_count",
    "annual_return_pct", "annual_trades", "reserved_buy_fill_rate",
    "full_year_count", "min_full_year_return_pct",
    "m_win_rate", "r_squared",
    "base_score", "trial_value", "fail_reason",
]


class OptimizerProfileRecorder:
    def __init__(self, output_dir, session_ts, enabled=True, console_print=False, print_every_n_trials=1):
        self.output_dir = output_dir
        self.session_ts = session_ts
        self.enabled = bool(enabled)
        self.console_print = bool(console_print)
        self.print_every_n_trials = max(1, int(print_every_n_trials))
        self.csv_path = os.path.join(output_dir, f"optimizer_profile_{session_ts}.csv")
        self.summary_path = os.path.join(output_dir, f"optimizer_profile_summary_{session_ts}.json")
        self.rows = []
        self._run_started_perf_counter = None
        self.first_trial_completed_wall_sec = None

    def init_output_files(self):
        if not self.enabled:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS)
            writer.writeheader()

    def mark_run_started(self):
        self._run_started_perf_counter = time.perf_counter()
        self.first_trial_completed_wall_sec = None

    def append_row(self, row):
        if not self.enabled:
            return

        normalized = {field: row.get(field, "") for field in PROFILE_FIELDS}
        self.rows.append(normalized)
        if self._run_started_perf_counter is not None and self.first_trial_completed_wall_sec is None:
            self.first_trial_completed_wall_sec = max(0.0, time.perf_counter() - self._run_started_perf_counter)
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS)
            writer.writerow(normalized)

    def build_summary_payload(self):
        if not self.enabled or not self.rows:
            return {
                "trial_count": len(self.rows),
                "avg": {},
                "first_trial_completed_wall_sec": self.first_trial_completed_wall_sec,
            }

        numeric_fields = [
            "objective_wall_sec", "prep_wall_sec", "prep_worker_total_sum_sec",
            "prep_worker_generate_signals_sum_sec", "prep_worker_run_backtest_sum_sec",
            "prep_worker_to_dict_sum_sec", "prep_avg_per_ok_sec", "sort_dates_sec",
            "portfolio_wall_sec", "portfolio_total_sec", "portfolio_ticker_dates_sec",
            "portfolio_build_trade_index_sec", "portfolio_day_loop_sec",
            "portfolio_candidate_scan_sec", "portfolio_rotation_sec", "portfolio_settle_sec",
            "portfolio_buy_sec", "portfolio_equity_mark_sec", "portfolio_closeout_sec",
            "portfolio_curve_stats_sec", "filter_rules_sec", "score_calc_sec",
        ]
        summary = {
            "trial_count": len(self.rows),
            "avg": {},
            "first_trial_completed_wall_sec": self.first_trial_completed_wall_sec,
        }

        for field in numeric_fields:
            values = []
            for row_idx, row in enumerate(self.rows):
                value = row.get(field, "")
                if isinstance(value, (int, float)):
                    values.append(float(value))
                    continue
                if isinstance(value, str) and value not in ("", None):
                    try:
                        values.append(float(value))
                    except ValueError as exc:
                        raise ValueError(f"PROFILE_ROWS[{row_idx}]['{field}'] 無法轉成 float: {value!r}") from exc
            summary["avg"][field] = (sum(values) / len(values)) if values else 0.0
        return summary

    def print_summary(self):
        if not self.enabled or not self.rows:
            return

        summary = self.build_summary_payload()
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        avg = summary["avg"]
        print(
            f"{C_CYAN}📊 Profiling 平均摘要（{summary['trial_count']} trials）:{C_RESET}\n"
            f"{C_GRAY}   objective={avg.get('objective_wall_sec', 0.0):.3f}s | "
            f"prep_wall={avg.get('prep_wall_sec', 0.0):.3f}s | "
            f"portfolio_wall={avg.get('portfolio_wall_sec', 0.0):.3f}s | "
            f"worker_generate_sum={avg.get('prep_worker_generate_signals_sum_sec', 0.0):.3f}s | "
            f"worker_backtest_sum={avg.get('prep_worker_run_backtest_sum_sec', 0.0):.3f}s | "
            f"to_dict_sum={avg.get('prep_worker_to_dict_sum_sec', 0.0):.3f}s | "
            f"pf_day_loop={avg.get('portfolio_day_loop_sec', 0.0):.3f}s{C_RESET}"
        )
        print(f"{C_GRAY}   CSV: {self.csv_path}{C_RESET}")
        print(f"{C_GRAY}   摘要: {self.summary_path}{C_RESET}")

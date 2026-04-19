import io
from contextlib import redirect_stdout

from core.config import SCORE_CALC_METHOD, SCORE_NUMERATOR_METHOD
from core.display_common import _strip_ansi
from core import display as display_module
from core.scanner_display import print_scanner_header
from core.strategy_dashboard import print_strategy_dashboard
from tools.scanner.reporting import print_scanner_start_banner, print_scanner_summary

from .checks import add_check


def _capture_output(callable_obj):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        callable_obj()
    return _strip_ansi(buffer.getvalue())


def validate_display_reporting_sanity_case(_base_params):
    case_id = "DISPLAY_REPORTING_SANITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    params = {
        "use_bb": True,
        "bb_len": 21,
        "bb_mult": 2.5,
        "use_kc": True,
        "kc_len": 34,
        "kc_mult": 1.8,
        "use_vol": True,
        "vol_short_len": 7,
        "vol_long_len": 21,
        "high_len": 123,
        "atr_len": 17,
        "atr_buy_tol": 1.2,
        "atr_times_init": 2.3,
        "atr_times_trail": 3.4,
        "tp_percent": 0.45,
        "min_history_trades": 11,
        "min_history_win_rate": 0.56,
        "min_history_ev": 0.78,
        "scanner_live_capital": 2_000_000.0,
    }

    scanner_text = _capture_output(lambda: print_scanner_header(params))
    add_check(results, "display_reporting", case_id, "scanner_contains_global_strategy_header", True, "全域戰略: 買入排序" in scanner_text)
    expected_scanner_score_header = f"評分模型 [{SCORE_CALC_METHOD}] | 評分分子 [{SCORE_NUMERATOR_METHOD}]"
    add_check(results, "display_reporting", case_id, "scanner_score_header_separates_model_and_numerator", True, expected_scanner_score_header in scanner_text and " / 分子 " not in scanner_text)
    add_check(results, "display_reporting", case_id, "scanner_contains_training_params", True, "突破 123日 | ATR 17日 | 掛單 +1.2倍 | 初始 -2.3倍 | 追蹤 -3.4倍 | 半倉 45%" in scanner_text)
    add_check(results, "display_reporting", case_id, "scanner_contains_filter_params", True, "布林(BB) 啟用 (長21, 寬2.5x) | 阿肯那(KC) 啟用 (長34, 寬1.8x) | 均量 啟用 (短7>長21)" in scanner_text)
    add_check(results, "display_reporting", case_id, "scanner_contains_history_thresholds", True, "交易 >= 11 次 | 勝率 >= 56% | 期望值 >= 0.78R" in scanner_text)
    add_check(results, "display_reporting", case_id, "scanner_contains_live_capital", True, "Scanner資金: live capital = 2,000,000" in scanner_text)

    dashboard_text = _capture_output(
        lambda: print_strategy_dashboard(
            params,
            title="策略測試儀表板",
            mode_display="投組模式",
            max_pos=5,
            trades=12,
            missed_b=3,
            missed_s=1,
            final_eq=1_234_567,
            avg_exp=62.34,
            sys_ret=18.76,
            bm_ret=10.11,
            sys_mdd=8.9,
            bm_mdd=12.34,
            win_rate=58.33,
            payoff=1.85,
            ev=0.72,
            benchmark_ticker="0050",
            max_exp=88.88,
            r_sq=0.91,
            m_win_rate=66.67,
            bm_r_sq=0.55,
            bm_m_win_rate=41.67,
            normal_trades=9,
            extended_trades=3,
            annual_trades=4.25,
            reserved_buy_fill_rate=83.33,
            annual_return_pct=12.34,
            bm_annual_return_pct=7.89,
            min_full_year_return_pct=5.43,
            bm_min_full_year_return_pct=-2.1,
        )
    )

    banner_text = _capture_output(lambda: print_scanner_start_banner("2026-04-02 12:34:56"))
    add_check(results, "display_reporting", case_id, "scanner_banner_has_title_and_time", True, "啟動【v16 尊爵版】極速平行掃描儀" in banner_text and "2026-04-02 12:34:56" in banner_text)

    scanner_summary_text = _capture_output(
        lambda: print_scanner_summary(
            count_scanned=77,
            elapsed_time=12.34,
            count_history_qualified=5,
            count_skipped_insufficient=2,
            count_sanitized_candidates=1,
            max_workers=6,
            pool_start_method="spawn",
            candidate_rows=[
                {"kind": "extended", "sort_value": 1.2, "ticker": "1101", "text": "1101 | sort=1.2"},
                {"kind": "buy", "sort_value": 3.4, "ticker": "2330", "text": "2330 | sort=3.4"},
            ],
            scanner_issue_log_path="outputs/vip_scanner/scanner_issues_20260402_123456.log",
        )
    )
    add_check(results, "display_reporting", case_id, "scanner_summary_has_counts_and_runtime", True, "共掃描 77 檔標的" in scanner_summary_text and "耗時 12.34 秒" in scanner_summary_text and "max_workers: 6" in scanner_summary_text and "start_method: spawn" in scanner_summary_text)
    add_check(results, "display_reporting", case_id, "scanner_summary_sorts_buy_before_extended", True, scanner_summary_text.find("[新訊號] 2330 | sort=3.4") < scanner_summary_text.find("[延續候選] 1101 | sort=1.2"))
    add_check(results, "display_reporting", case_id, "scanner_summary_has_candidate_stats_and_issue_path", True, "候選統計：新訊號 1 檔 | 延續候選 1 檔" in scanner_summary_text and "outputs/vip_scanner/scanner_issues_20260402_123456.log" in scanner_summary_text)

    reexport_checks = all(
        getattr(display_module, name) is obj
        for name, obj in {
            "print_scanner_header": print_scanner_header,
            "print_strategy_dashboard": print_strategy_dashboard,
            "_strip_ansi": _strip_ansi,
        }.items()
    )
    add_check(results, "display_reporting", case_id, "core_display_reexports_expected_symbols", True, reexport_checks)
    add_check(results, "display_reporting", case_id, "dashboard_contains_title", True, "策略測試儀表板" in dashboard_text)
    expected_dashboard_score_header = f"評分模型 [{SCORE_CALC_METHOD}] | 評分分子 [{SCORE_NUMERATOR_METHOD}] | 系統得分:"
    add_check(results, "display_reporting", case_id, "dashboard_score_header_separates_model_and_numerator", True, expected_dashboard_score_header in dashboard_text and " / 分子 " not in dashboard_text)
    add_check(results, "display_reporting", case_id, "dashboard_contains_mode_and_positions", True, "模式: 投組模式 | 最大持股: 5 檔" in dashboard_text)
    add_check(results, "display_reporting", case_id, "dashboard_contains_trade_split", True, "總交易次數: 12 筆 (正常:9 | 延續:3) | 年化交易次數: 4.25 次/年" in dashboard_text)
    add_check(results, "display_reporting", case_id, "dashboard_contains_missed_counts_and_asset", True, "錯失次數: 買 3 | 賣 1 | 保留後買進成交率: 83.33% | 最終資產: 1,234,567 元" in dashboard_text)
    add_check(results, "display_reporting", case_id, "dashboard_contains_avg_exposure", True, "平均資金水位: 62.34 % (最高 88.88 %)" in dashboard_text)
    add_check(results, "display_reporting", case_id, "dashboard_contains_return_row", True, "總資產報酬率" in dashboard_text and "+18.76%" in dashboard_text and "+10.11%" in dashboard_text)
    add_check(results, "display_reporting", case_id, "dashboard_contains_benchmark_ticker", True, "同期大盤 (0050)" in dashboard_text)
    add_check(results, "display_reporting", case_id, "dashboard_contains_ev_row", True, "實戰期望值(EV)" in dashboard_text and "0.72 R" in dashboard_text)
    add_check(results, "display_reporting", case_id, "dashboard_contains_filter_summary", True, "濾網參數 : 布林(BB) 啟用 (長21, 寬2.5x) | 阿肯那(KC) 啟用 (長34, 寬1.8x) | 均量 啟用 (短7>長21)" in dashboard_text)

    summary["scanner_lines"] = len([line for line in scanner_text.splitlines() if line.strip()])
    summary["scanner_summary_lines"] = len([line for line in scanner_summary_text.splitlines() if line.strip()])
    summary["banner_lines"] = len([line for line in banner_text.splitlines() if line.strip()])
    summary["dashboard_lines"] = len([line for line in dashboard_text.splitlines() if line.strip()])
    return results, summary

import os
import sys
import warnings
import time

from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir, get_dataset_profile_label, resolve_dataset_profile_from_cli_env
from core.display import C_CYAN, C_GREEN, C_GRAY, C_RED, C_RESET, C_YELLOW, print_strategy_dashboard
from core.runtime_utils import has_help_flag, safe_prompt, safe_prompt_choice, safe_prompt_int
from .reporting import export_portfolio_reports, print_yearly_return_report
from .runtime import BEST_PARAMS_PATH, PROJECT_ROOT, ensure_runtime_dirs, load_strict_params, run_portfolio_simulation

warnings.simplefilter("default")
warnings.filterwarnings("once", category=RuntimeWarning)


def main(argv=None, env=None):
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env

    if has_help_flag(argv):
        print("用法: python apps/portfolio_sim.py [--dataset reduced|full]")
        print("說明: 非互動模式會自動套用預設輸入；完整資料集預設使用 /data/tw_stock_data_vip。")
        return 0

    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 投資組合模擬器：機構級實戰期望值 (終極模組化對齊版){C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")

    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(
            argv,
            env,
            default=DEFAULT_DATASET_PROFILE,
        )
        selected_data_dir = get_dataset_dir(PROJECT_ROOT, dataset_profile_key)
        print(
            f"{C_GRAY}📁 使用資料集: {get_dataset_profile_label(dataset_profile_key)} | "
            f"來源: {dataset_source} | 路徑: {selected_data_dir}{C_RESET}"
        )

        rotation_choice = safe_prompt_choice(
            "👉 1. 啟用「汰弱換股」？ (Y/N, 預設 N): ",
            "N",
            ("Y", "N"),
            "汰弱換股選項",
        )
        user_rotation = rotation_choice == 'Y'
        user_max_pos = safe_prompt_int(
            "👉 2. 最大持倉數量 (預設 10): ",
            10,
            "最大持倉數量",
            min_value=1,
        )
        user_start_year = safe_prompt_int(
            "👉 3. 開始回測年份 (預設 2015): ",
            2015,
            "開始回測年份",
            min_value=1900,
        )
        user_benchmark = safe_prompt("👉 4. 大盤比較標的 (預設 0050): ", "0050")
    except ValueError as e:
        print(f"{C_RED}❌ {e}{C_RESET}", file=sys.stderr)
        raise SystemExit(1)

    ensure_runtime_dirs()
    try:
        params = load_strict_params(BEST_PARAMS_PATH)
        print(f"\n{C_GREEN}✅ 成功載入 AI 訓練大腦！{C_RESET}")

        start_time = time.time()
        result = run_portfolio_simulation(
            selected_data_dir,
            params,
            user_max_pos,
            user_rotation,
            user_start_year,
            user_benchmark,
        )
        end_time = time.time()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    (
        df_eq, df_tr, tot_ret, mdd, trade_count, win_rate, pf_ev, pf_payoff,
        final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed,
        total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate,
        normal_trade_count, extended_trade_count, annual_trades,
        reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct, pf_profile,
    ) = result

    mode_display = "開啟 (強勢輪動)" if user_rotation else "關閉 (穩定鎖倉)"
    min_full_year_return_pct = pf_profile.get("min_full_year_return_pct", 0.0)
    bm_min_full_year_return_pct = pf_profile.get("bm_min_full_year_return_pct", 0.0)

    print(f"\n{C_CYAN}================================================================================{C_RESET}")
    print(f"📊 【投資組合實戰模擬報告 (自 {user_start_year} 年起算)】")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"回測總耗時: {end_time - start_time:.2f} 秒")

    print_strategy_dashboard(
        params=params, title="績效與風險對比表", mode_display=mode_display, max_pos=user_max_pos,
        trades=trade_count, missed_b=total_missed, missed_s=total_missed_sells,
        final_eq=final_eq, avg_exp=avg_exp, sys_ret=tot_ret, bm_ret=bm_ret,
        sys_mdd=mdd, bm_mdd=bm_mdd, win_rate=win_rate, payoff=pf_payoff, ev=pf_ev,
        benchmark_ticker=user_benchmark, max_exp=max_exp,
        r_sq=r_sq, m_win_rate=m_win_rate, bm_r_sq=bm_r_sq, bm_m_win_rate=bm_m_win_rate,
        normal_trades=normal_trade_count, extended_trades=extended_trade_count,
        annual_trades=annual_trades, reserved_buy_fill_rate=reserved_buy_fill_rate,
        annual_return_pct=annual_return_pct, bm_annual_return_pct=bm_annual_return_pct,
        min_full_year_return_pct=min_full_year_return_pct, bm_min_full_year_return_pct=bm_min_full_year_return_pct
    )

    df_yearly = print_yearly_return_report(pf_profile.get("yearly_return_rows", []))
    if pf_profile.get("full_year_count", 0) > 0:
        print(
            f"{C_GRAY}完整年度數: {pf_profile.get('full_year_count', 0)} | "
            f"最差完整年度報酬: {pf_profile.get('min_full_year_return_pct', 0.0):.2f}% | "
            f"大盤最差完整年度報酬: {pf_profile.get('bm_min_full_year_return_pct', 0.0):.2f}% | "
            f"年化報酬率: {annual_return_pct:.2f}%{C_RESET}"
        )

    export_portfolio_reports(df_eq, df_tr, df_yearly, user_benchmark, user_start_year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os
import sys
import warnings
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir, get_dataset_profile_label, resolve_dataset_profile_from_cli_env, build_missing_dataset_dir_message, build_empty_dataset_dir_message
from core.model_paths import resolve_named_params_path
from core.display import C_CYAN, C_GREEN, C_GRAY, C_RED, C_RESET, C_YELLOW, print_strategy_dashboard
from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, has_help_flag, resolve_cli_program_name, safe_prompt, safe_prompt_choice, safe_prompt_int, validate_cli_args

warnings.simplefilter("default")
warnings.filterwarnings("once", category=RuntimeWarning)


def main(argv=None, env=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    validate_cli_args(argv, value_options=("--dataset",))

    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/portfolio_sim/main.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print("說明: 非互動模式會自動套用預設輸入；預設資料集為完整；參數來源預設 champion；大盤比較固定使用 0050；開始回測年份預設取自 config/walk_forward_policy.py。")
        return 0

    from core.data_utils import discover_unique_csv_inputs
    from tools.portfolio_sim.reporting import export_portfolio_reports, print_yearly_return_report
    from tools.portfolio_sim.runtime import ensure_runtime_dirs, load_strict_params, run_portfolio_simulation
    from tools.portfolio_sim.simulation_runner import PORTFOLIO_DEFAULT_BENCHMARK_TICKER, resolve_default_portfolio_start_year

    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(
            argv,
            env,
            default=DEFAULT_DATASET_PROFILE,
        )
        selected_data_dir = get_dataset_dir(PROJECT_ROOT, dataset_profile_key)
        if not os.path.isdir(selected_data_dir):
            raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
        csv_inputs, _ = discover_unique_csv_inputs(selected_data_dir)
        if not csv_inputs:
            raise FileNotFoundError(build_empty_dataset_dir_message(dataset_profile_key, selected_data_dir))
    except (ValueError, FileNotFoundError) as e:
        print(f"{C_RED}❌ {e}{C_RESET}", file=sys.stderr)
        return 1

    default_start_year = resolve_default_portfolio_start_year()
    try:
        param_source_choice = safe_prompt_choice(
            "👉 1. 參數來源 (C=champion / R=run_best, 預設 C): ",
            "C",
            ("C", "R"),
            "參數來源",
        )
        param_source = "champion" if param_source_choice == "C" else "run_best"
        params_path = resolve_named_params_path(PROJECT_ROOT, param_source)
        params = load_strict_params(params_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 投資組合模擬器：機構級實戰期望值 (終極模組化對齊版){C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(
        f"{C_GRAY}📁 使用資料集: {get_dataset_profile_label(dataset_profile_key)} | "
        f"來源: {dataset_source} | 路徑: {selected_data_dir}{C_RESET}"
    )
    print(f"\n{C_GREEN}✅ 成功載入 AI 訓練大腦！{C_RESET}")

    try:
        rotation_choice = safe_prompt_choice(
            "👉 1. 啟用「汰弱換股」？ (Y/N, 預設 N): ",
            "N",
            ("Y", "N"),
            "汰弱換股選項",
        )
        user_rotation = rotation_choice == 'Y'
        user_max_pos = safe_prompt_int(
            "👉 3. 最大持倉數量 (預設 10): ",
            10,
            "最大持倉數量",
            min_value=1,
        )
        user_start_year = safe_prompt_int(
            f"👉 4. 開始回測年份 (預設 {default_start_year}): ",
            default_start_year,
            "開始回測年份",
            min_value=1900,
        )
        user_benchmark = PORTFOLIO_DEFAULT_BENCHMARK_TICKER
    except ValueError as e:
        print(f"{C_RED}❌ {e}{C_RESET}", file=sys.stderr)
        return 1

    ensure_runtime_dirs()
    try:
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
    run_cli_entrypoint(main)

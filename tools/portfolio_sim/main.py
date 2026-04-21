import os
import sys
import warnings
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir, get_dataset_profile_label, resolve_dataset_profile_from_cli_env, build_missing_dataset_dir_message, build_empty_dataset_dir_message
from core.model_paths import resolve_candidate_best_params_path, resolve_run_best_params_path
from core.display import C_CYAN, C_GREEN, C_GRAY, C_RED, C_RESET, C_YELLOW, print_strategy_dashboard
from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, has_help_flag, resolve_cli_program_name, safe_prompt, safe_prompt_choice, safe_prompt_int, parse_int_strict, parse_float_strict, validate_cli_args

warnings.simplefilter("default")
warnings.filterwarnings("once", category=RuntimeWarning)


_FIXED_RISK_MENU_DEFAULT = 0.01
_FIXED_RISK_MENU_QUICK_CHOICE_2 = 0.02


def _resolve_portfolio_fixed_risk_input(raw_value):
    text = str(raw_value).strip()
    if text == "":
        return _FIXED_RISK_MENU_DEFAULT
    if text == "2":
        return _FIXED_RISK_MENU_QUICK_CHOICE_2
    return parse_float_strict(text, "固定風險比例", min_value=0.0, max_value=1.0, strict_gt=True)


def main(argv=None, env=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    validate_cli_args(argv, value_options=("--dataset",))

    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/portfolio_sim/main.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print("說明: 非互動模式會自動套用預設輸入；預設資料集為完整；參數來源可選 run_best（預設）或 candidate_best；大盤比較固定使用 0050；開始回測年份預設取自目前資料集的 OOS 起始日期。")
        return 0

    from core.data_utils import normalize_ticker_from_csv_filename
    from core.walk_forward_policy import load_walk_forward_policy

    def _has_any_csv_input(data_dir: str) -> bool:
        try:
            with os.scandir(data_dir) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    if normalize_ticker_from_csv_filename(entry.name):
                        return True
        except OSError:
            return False
        return False

    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(
            argv,
            env,
            default=DEFAULT_DATASET_PROFILE,
        )
        selected_data_dir = get_dataset_dir(PROJECT_ROOT, dataset_profile_key)
        if not os.path.isdir(selected_data_dir):
            raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
        if not _has_any_csv_input(selected_data_dir):
            raise FileNotFoundError(build_empty_dataset_dir_message(dataset_profile_key, selected_data_dir))
    except (ValueError, FileNotFoundError) as e:
        print(f"{C_RED}❌ {e}{C_RESET}", file=sys.stderr)
        return 1

    policy = load_walk_forward_policy(PROJECT_ROOT)
    default_start_year_hint = int(policy["search_train_end_year"]) + 1

    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 投資組合模擬器：機構級實戰期望值 (終極模組化對齊版){C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(
        f"{C_GRAY}📁 使用資料集: {get_dataset_profile_label(dataset_profile_key)} | "
        f"來源: {dataset_source} | 路徑: {selected_data_dir}{C_RESET}"
    )

    try:
        param_source_choice = safe_prompt_choice(
            "👉 參數來源：[Enter] run_best (預設)  [C] candidate_best :  ",
            "R",
            ("R", "C"),
            "參數來源",
        )
        param_source = "candidate_best" if param_source_choice == "C" else "run_best"
        print(f"{C_GRAY}ℹ️ 參數來源: {param_source}{C_RESET}")
        rotation_choice = safe_prompt_choice(
            "👉 汰弱換股：[N] 關閉 (預設)  [Y] 啟用 :  ",
            "N",
            ("N", "Y"),
            "汰弱換股選項",
        )
        user_rotation = rotation_choice == 'Y'
        user_max_pos = safe_prompt_int(
            "👉 最大持倉數量：[ Enter] 10  [數字] 指定數量，:  ",
            10,
            "最大持倉數量",
            min_value=1,
        )
        raw_start_year = safe_prompt(
            f"👉 開始回測年份：[Enter] 測試起始年{default_start_year_hint}  [數字] 指定年份: ",
            "",
        ).strip()
        raw_fixed_risk = safe_prompt(
            "👉 單筆固定風險：[Enter] 0.01  [2] 0.02  [數字] 指定比例: ",
            "",
        ).strip()
        user_fixed_risk = _resolve_portfolio_fixed_risk_input(raw_fixed_risk)
        from tools.portfolio_sim.simulation_runner import PORTFOLIO_DEFAULT_BENCHMARK_TICKER

        if raw_start_year == "":
            user_start_year = int(default_start_year_hint)
            user_benchmark = PORTFOLIO_DEFAULT_BENCHMARK_TICKER
        else:
            user_start_year = parse_int_strict(raw_start_year, "開始回測年份", min_value=1900)
            user_benchmark = PORTFOLIO_DEFAULT_BENCHMARK_TICKER
    except ValueError as e:
        print(f"{C_RED}❌ {e}{C_RESET}", file=sys.stderr)
        return 1

    from tools.portfolio_sim.reporting import export_portfolio_reports, print_yearly_return_report
    from tools.portfolio_sim.runtime import ensure_runtime_dirs, load_strict_params, run_portfolio_simulation

    try:
        if param_source == "candidate_best":
            params_path = resolve_candidate_best_params_path(PROJECT_ROOT)
        else:
            params_path = resolve_run_best_params_path(PROJECT_ROOT)
        params = load_strict_params(params_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    try:
        params.fixed_risk = user_fixed_risk
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    print(f"\n{C_GREEN}✅ 成功載入 AI 訓練大腦！{C_RESET}")
    print(f"{C_GRAY}📦 參數檔: {params_path}{C_RESET}")
    print(f"{C_GRAY}ℹ️ 單筆固定風險: {params.fixed_risk:.4f}{C_RESET}")

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

    df_yearly = print_yearly_return_report(
        pf_profile.get("yearly_return_rows", []),
        benchmark_yearly_return_rows=pf_profile.get("bm_yearly_return_rows", []),
        benchmark_ticker=user_benchmark,
    )
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

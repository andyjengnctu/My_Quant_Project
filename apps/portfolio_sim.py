import os
import sys
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.log_utils import format_exception_summary
from core.runtime_utils import run_cli_entrypoint, has_help_flag, resolve_cli_program_name, should_auto_open_browser, validate_cli_args

HELP_DESCRIPTION = "說明: 非互動模式會自動套用預設輸入；預設資料集為完整，路徑為 /data/tw_stock_data_vip。"
LAZY_EXPORTS = {
    "ensure_runtime_dirs",
    "is_insufficient_data_error",
    "load_strict_params",
    "print_yearly_return_report",
    "run_portfolio_simulation",
}


def main(argv=None, env=None):
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv, value_options=("--dataset",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "apps/portfolio_sim.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print(HELP_DESCRIPTION)
        return 0

    from tools.portfolio_sim import main as portfolio_main

    exit_code = portfolio_main(argv=argv, env=env)
    if exit_code == 0:
        try:
            from tools.portfolio_sim.reporting import DASHBOARD_HTML_PATH

            if os.path.exists(DASHBOARD_HTML_PATH) and should_auto_open_browser(os.environ if env is None else env):
                webbrowser.open('file://' + os.path.realpath(DASHBOARD_HTML_PATH))
        except (ImportError, OSError, ValueError, RuntimeError, webbrowser.Error) as exc:
            print(f"⚠️ 自動開啟投組報表失敗: {format_exception_summary(exc)}", file=sys.stderr)
    return exit_code


def __getattr__(name):
    if name == "main":
        return main
    if name in LAZY_EXPORTS:
        from tools import portfolio_sim as portfolio_module

        value = getattr(portfolio_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ensure_runtime_dirs",
    "is_insufficient_data_error",
    "load_strict_params",
    "print_yearly_return_report",
    "run_portfolio_simulation",
    "main",
]

if __name__ == "__main__":
    run_cli_entrypoint(main)

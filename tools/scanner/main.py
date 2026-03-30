import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import run_cli_entrypoint, has_help_flag, resolve_cli_program_name, validate_cli_args


def main(argv=None, env=None):
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv, value_options=("--dataset",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/scanner/main.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print("說明: 預設資料集為完整；縮減資料集路徑為 <repo>/data/tw_stock_data_vip_reduced。")
        return 0

    from tools.scanner.scan_runner import main as scanner_main

    return scanner_main(argv=argv, env=env)


def run_daily_scanner(*args, **kwargs):
    from tools.scanner.scan_runner import run_daily_scanner as _run_daily_scanner

    return _run_daily_scanner(*args, **kwargs)


__all__ = [
    "run_daily_scanner",
    "main",
]

if __name__ == "__main__":
    run_cli_entrypoint(main)

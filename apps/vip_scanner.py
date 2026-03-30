import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import run_cli_entrypoint, has_help_flag, resolve_cli_program_name, validate_cli_args

HELP_DESCRIPTION = "說明: 預設資料集為完整；縮減資料集路徑為 <repo>/data/tw_stock_data_vip_reduced。"
LAZY_EXPORTS = {
    "ensure_runtime_dirs",
    "is_insufficient_data_error",
    "load_strict_params",
    "process_single_stock",
    "resolve_scanner_max_workers",
    "run_daily_scanner",
}


def main(argv=None, env=None):
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv, value_options=("--dataset",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "apps/vip_scanner.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print(HELP_DESCRIPTION)
        return 0

    from tools.scanner import main as scanner_main

    return scanner_main(argv=argv, env=env)


def __getattr__(name):
    if name == "main":
        return main
    if name in LAZY_EXPORTS:
        from tools import scanner as scanner_module

        value = getattr(scanner_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ensure_runtime_dirs",
    "is_insufficient_data_error",
    "load_strict_params",
    "process_single_stock",
    "resolve_scanner_max_workers",
    "run_daily_scanner",
    "main",
]

if __name__ == "__main__":
    run_cli_entrypoint(main)

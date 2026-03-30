import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import has_help_flag, resolve_cli_program_name, validate_cli_args


def main(argv=None, environ=None):
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv, value_options=("--dataset",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/validate/cli.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print("說明: 預設資料集為縮減；reduced 測試資料路徑為 <repo>/data/tw_stock_data_vip_reduced。")
        return 0

    from tools.validate import main as validate_main

    return validate_main(argv=argv, environ=environ)


__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())

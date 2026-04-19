import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import run_cli_entrypoint, has_help_flag, resolve_cli_program_name, validate_cli_args

HELP_DESCRIPTION = "說明: 預設資料集為完整。可選 --model split|full；split 為固定 pre-deploy+OOS，full 為全資料選參。輸入 0 只匯出 run_best。"


def main(argv=None, environ=None):
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv, value_options=("--dataset", "--model"))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "apps/ml_optimizer.py")
        print(f"用法: python {program_name} [--dataset reduced|full] [--model split|full]")
        print(HELP_DESCRIPTION)
        return 0

    from tools.optimizer import main as optimizer_main

    return optimizer_main(argv=argv, environ=environ)


__all__ = ["main"]

if __name__ == "__main__":
    run_cli_entrypoint(main)

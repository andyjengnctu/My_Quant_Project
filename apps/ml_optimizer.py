import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import enable_line_buffered_stdout, has_help_flag, validate_cli_args


OPTIMIZER_HELP_LINES = (
    "用法: python apps/ml_optimizer.py [--dataset reduced|full]",
    "說明: 預設資料集為完整；非互動模式預設訓練次數為 0；可用環境變數 V16_OPTIMIZER_TRIALS 指定 trial 數。",
)


def main(argv=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    try:
        validate_cli_args(argv, allowed_value_options=("--dataset",))
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    if has_help_flag(argv):
        for line in OPTIMIZER_HELP_LINES:
            print(line)
        return 0
    from tools.optimizer import main as optimizer_main
    return optimizer_main(argv=argv)


__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())

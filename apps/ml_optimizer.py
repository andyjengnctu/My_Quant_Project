import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import run_cli_entrypoint, has_help_flag, resolve_cli_program_name, validate_cli_args

HELP_DESCRIPTION = "說明: 預設資料集為完整；非互動模式預設訓練次數為 0；可用環境變數 V16_OPTIMIZER_TRIALS 指定 trial 數、V16_OPTIMIZER_SEED 指定固定 seed；walk-forward 設定來自 config/walk_forward_policy.json；完成指定訓練次數或輸入 0 匯出時，會更新本輪最佳 run_best_params.json；正式現役版固定使用 champion_params.json；未指定 --promote 或 V16_OPTIMIZER_AUTO_PROMOTE 時，互動模式會詢問是否在 Compare PASS 後自動升版；trial=0 永不升版。"


def main(argv=None, environ=None):
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv, value_options=("--dataset",), flag_options=("--promote",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "apps/ml_optimizer.py")
        print(f"用法: python {program_name} [--dataset reduced|full] [--promote]")
        print(HELP_DESCRIPTION)
        return 0

    from tools.optimizer import main as optimizer_main

    return optimizer_main(argv=argv, environ=environ)


__all__ = ["main"]

if __name__ == "__main__":
    run_cli_entrypoint(main)

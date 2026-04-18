import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import run_cli_entrypoint, has_help_flag, resolve_cli_program_name, validate_cli_args

HELP_DESCRIPTION = "說明: 預設資料集為完整、模型模式預設為 split；可用 --model split|legacy（wf 仍接受但會視為 split）或環境變數 V16_OPTIMIZER_MODEL 切換；split 模式採 train/test 完全分離：訓練只看 train RoMD、測試只看 test RoMD，Champion 永遠保留目前測試 RoMD 最好的策略+參數，不會把測試分數自動回灌到同一次訓練；互動式 Optimizer 選單預設訓練 50000 次，輸入 0 可寫入 best run 並重新測試目前記憶庫最佳；非互動模式預設訓練次數為 0；可用環境變數 V16_OPTIMIZER_TRIALS 指定 trial 數、V16_OPTIMIZER_SEED 指定固定 seed；walk-forward 設定來自 config/walk_forward_policy.py；正式現役版固定使用 champion_params.json。"


def main(argv=None, environ=None):
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv, value_options=("--dataset", "--model"), flag_options=("--promote",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "apps/ml_optimizer.py")
        print(f"用法: python {program_name} [--dataset reduced|full] [--model split|legacy] [--promote]")
        print(HELP_DESCRIPTION)
        return 0

    from tools.optimizer import main as optimizer_main

    return optimizer_main(argv=argv, environ=environ)


__all__ = ["main"]

if __name__ == "__main__":
    run_cli_entrypoint(main)

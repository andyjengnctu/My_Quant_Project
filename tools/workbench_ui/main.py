import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import parse_no_arg_cli


HELP_DESCRIPTION = "啟動股票工具工作台；目前內建單股回測檢視頁籤，後續 GUI 功能統一擴充於此入口。"


def main(argv=None):
    cli_info = parse_no_arg_cli(argv, "apps/workbench.py", description=HELP_DESCRIPTION)
    if cli_info["help"]:
        return 0

    from tools.workbench_ui.workbench import launch_workbench

    launch_workbench()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

import sys

from core.runtime_utils import parse_no_arg_cli

from tools.gui.workbench import launch_workbench


HELP_DESCRIPTION = "啟動股票工具工作台；目前內建單股回測檢視與投組回測檢視頁籤，後續 GUI 功能統一擴充於此入口。"


def main(argv=None):
    cli_info = parse_no_arg_cli(argv, "apps/gui.py", description=HELP_DESCRIPTION)
    if cli_info["help"]:
        return 0

    launch_workbench()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import parse_no_arg_cli


HELP_DESCRIPTION = "啟動股票工具工作台；整合單股回測、投組回測與實際交易帳戶操作。"


def main(argv=None):
    cli_info = parse_no_arg_cli(argv, "apps/workbench.py", description=HELP_DESCRIPTION)
    if cli_info["help"]:
        return 0

    from services.workbench_ui.workbench import launch_workbench

    launch_workbench()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

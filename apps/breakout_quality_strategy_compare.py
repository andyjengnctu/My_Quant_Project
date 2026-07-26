import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import run_cli_entrypoint


def main(argv=None):
    from tools.filters.breakout_quality.strategy_compare import main as compare_main

    return compare_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    run_cli_entrypoint(main)

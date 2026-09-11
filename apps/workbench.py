import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import run_cli_entrypoint
from services.workbench_ui import main


__all__ = ["main"]


if __name__ == "__main__":
    run_cli_entrypoint(main, sys.argv)

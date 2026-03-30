import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import has_help_flag, validate_cli_args

HELP_LINES = (
    "用法: python apps/smart_downloader.py",
    "說明: 下載或更新完整資料集到預設 full dataset 路徑。",
)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv)
    if has_help_flag(argv):
        for line in HELP_LINES:
            print(line)
        return 0

    from tools.downloader import main as downloader_main

    return downloader_main(argv=argv)


def __getattr__(name):
    if name == "main":
        return main
    if name == "smart_download_vip_data":
        from tools import downloader as downloader_module

        value = getattr(downloader_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["smart_download_vip_data", "main"]

if __name__ == "__main__":
    raise SystemExit(main())

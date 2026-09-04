import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.runtime_utils import run_cli_entrypoint, has_help_flag, resolve_cli_program_name, validate_cli_args

HELP_DESCRIPTION = "說明: 下載或更新 Trading 完整資料集；Research 資料不會被修改。"


def main(argv=None):
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv)
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "apps/smart_downloader.py")
        print(f"用法: python {program_name}")
        print(HELP_DESCRIPTION)
        return 0

    from services.downloader import main as downloader_main

    return downloader_main(argv=argv)


def __getattr__(name):
    if name == "main":
        return main
    if name == "smart_download_vip_data":
        from services import downloader as downloader_module

        value = getattr(downloader_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["smart_download_vip_data", "main"]

if __name__ == "__main__":
    run_cli_entrypoint(main)

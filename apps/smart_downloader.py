import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.downloader import main, smart_download_vip_data

__all__ = ["smart_download_vip_data", "main"]

if __name__ == "__main__":
    raise SystemExit(main())

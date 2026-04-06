import os
import time
import requests
import pandas as pd
import sys
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dataset_profiles import DATASET_PROFILE_FULL, get_dataset_dir
from core.log_utils import append_issue_log, build_timestamped_log_path
from core.runtime_utils import get_taipei_now, get_taipei_file_mtime
from core.output_paths import build_output_dir

API_TOKEN = os.getenv("FINMIND_API_TOKEN", "")
BASE_DIR = PROJECT_ROOT
SAVE_DIR = get_dataset_dir(BASE_DIR, DATASET_PROFILE_FULL)


# # (AI註: 單一真理來源 - universe 名單路徑必須即時依 SAVE_DIR 推導，避免目錄重導後仍寫回舊路徑)
def get_universe_list_file_path():
    return os.path.join(SAVE_DIR, "universe_list.txt")


MIN_VOLUME = 1_000_000
MIN_MARKET_CAP = 10_000_000_000
RESCAN_DAYS = 7
REQUEST_TIMEOUT_SEC = 10
YF_SCREEN_SLEEP_SEC = 0.01
FINMIND_DOWNLOAD_SLEEP_SEC = 0.5
FINMIND_PRICE_DATASET = 'TaiwanStockPriceAdj'
OUTPUT_DIR = build_output_dir(BASE_DIR, 'smart_downloader')

# # (AI註: 大量批次時避免逐筆錯誤洗板；詳細清單仍保留在摘要與 log)
def _get_optional_curl_request_exceptions():
    try:
        from curl_cffi.requests.exceptions import RequestException as CurlRequestException
    except (ImportError, ModuleNotFoundError) as exc:
        return (), f"{type(exc).__name__}: {exc}"
    return (CurlRequestException,), ""


OPTIONAL_CURL_REQUEST_EXCEPTIONS, OPTIONAL_CURL_REQUEST_EXCEPTIONS_IMPORT_ERROR = _get_optional_curl_request_exceptions()


EXPECTED_MARKET_DATE_EXCEPTIONS = (
    requests.RequestException,
    ValueError,
    KeyError,
    IndexError,
    TypeError,
    ImportError,
    ModuleNotFoundError,
) + OPTIONAL_CURL_REQUEST_EXCEPTIONS

EXPECTED_UNIVERSE_FETCH_EXCEPTIONS = (
    requests.RequestException,
    ValueError,
    KeyError,
    IndexError,
    pd.errors.EmptyDataError,
) + OPTIONAL_CURL_REQUEST_EXCEPTIONS

EXPECTED_SCREENING_EXCEPTIONS = (
    requests.RequestException,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
) + OPTIONAL_CURL_REQUEST_EXCEPTIONS

EXPECTED_LAST_DATE_CHECK_EXCEPTIONS = (
    OSError,
    ValueError,
    KeyError,
    IndexError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
) + OPTIONAL_CURL_REQUEST_EXCEPTIONS

EXPECTED_DOWNLOAD_EXCEPTIONS = (
    requests.RequestException,
    ValueError,
    KeyError,
    TypeError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    OSError,
    ImportError,
    ModuleNotFoundError,
) + OPTIONAL_CURL_REQUEST_EXCEPTIONS

# # (AI註: 大量批次時預設不逐筆洗板；需要時再手動切成 True)
VERBOSE_UNIVERSE_FETCH_ERRORS = False
VERBOSE_LAST_DATE_CHECK_ERRORS = False
VERBOSE_DOWNLOAD_ERRORS = False

dl = None


def get_yfinance_module():
    import yfinance as yf
    return yf


def get_finmind_dataloader_class():
    from FinMind.data import DataLoader
    return DataLoader


def get_finmind_loader():
    global dl
    if dl is None:
        DataLoader = get_finmind_dataloader_class()
        dl = DataLoader()
        if API_TOKEN:
            dl.login_by_token(api_token=API_TOKEN)
    return dl


# # (AI註: 將執行期目錄建立延後到實際執行，避免被 import 時產生副作用)
def ensure_runtime_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# # (AI註: 防錯透明化 - 將錯誤摘要落檔，避免長時間批次執行後 console 訊息遺失)
DOWNLOADER_ISSUE_LOG_PATH = None


# # (AI註: session log path 改為 lazy init；只有真的要寫 log 時才建立輸出目錄)
def get_downloader_issue_log_path():
    global DOWNLOADER_ISSUE_LOG_PATH
    if DOWNLOADER_ISSUE_LOG_PATH is None:
        ensure_runtime_dirs()
        DOWNLOADER_ISSUE_LOG_PATH = build_timestamped_log_path(
            "downloader_issues",
            log_dir=OUTPUT_DIR,
            timestamp=get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")
        )
    return DOWNLOADER_ISSUE_LOG_PATH


# # (AI註: 將下載器的非致命問題統一寫入單一 session log，避免每種類別各自爆檔)
def append_downloader_issues(section, lines):
    if not lines:
        return

    append_issue_log(
        get_downloader_issue_log_path(),
        [f"[{section}] {line}" for line in lines]
    )

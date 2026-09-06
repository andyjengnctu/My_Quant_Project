import os
import re
import time
import requests
import pandas as pd
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.downloader import (
    DOWNLOADER_FINMIND_DOWNLOAD_SLEEP_SEC as FINMIND_DOWNLOAD_SLEEP_SEC,
    DOWNLOADER_MIN_MARKET_CAP as MIN_MARKET_CAP,
    DOWNLOADER_MIN_VOLUME as MIN_VOLUME,
    DOWNLOADER_REQUEST_TIMEOUT_SEC as REQUEST_TIMEOUT_SEC,
    DOWNLOADER_RESCAN_DAYS as RESCAN_DAYS,
    DOWNLOADER_VERBOSE_DOWNLOAD_ERRORS as VERBOSE_DOWNLOAD_ERRORS,
    DOWNLOADER_VERBOSE_LAST_DATE_CHECK_ERRORS as VERBOSE_LAST_DATE_CHECK_ERRORS,
    DOWNLOADER_VERBOSE_UNIVERSE_FETCH_ERRORS as VERBOSE_UNIVERSE_FETCH_ERRORS,
)
from core.log_utils import append_issue_log, build_timestamped_log_path
from core.runtime_utils import get_taipei_now, get_taipei_file_mtime
from core.runtime_domains import (
    RUNTIME_DOMAIN_TRADING,
    assert_runtime_write_path_is_not_research_dataset,
    resolve_runtime_domain_paths,
    resolve_runtime_output_dir,
)

FINMIND_API_TOKEN_ENV_VAR = "FINMIND_API_TOKEN"
FINMIND_API_TOKEN_DOC_RELATIVE_PATH = Path("doc") / "FINMIND_API_TOKEN.md"


def resolve_finmind_api_token_path(project_root=None) -> Path:
    root = Path(PROJECT_ROOT if project_root is None else project_root).resolve()
    return root / FINMIND_API_TOKEN_DOC_RELATIVE_PATH


def _extract_finmind_api_token_from_markdown(text: str) -> str:
    lines = str(text or "").splitlines()
    in_token_section = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if re.fullmatch(r"#{1,6}\s*API_TOKEN\s*:?[\s]*", stripped, flags=re.IGNORECASE):
            in_token_section = True
            continue
        if not in_token_section:
            continue
        if stripped.startswith("#"):
            break
        if not stripped or stripped.startswith("```"):
            continue
        labelled = re.fullmatch(
            r"(?:FINMIND_)?API_TOKEN\s*[:=]\s*(.+)",
            stripped,
            flags=re.IGNORECASE,
        )
        value = labelled.group(1).strip() if labelled else stripped
        if len(value) >= 2 and value[0] == value[-1] == "`":
            value = value[1:-1].strip()
        return value
    return ""


def resolve_finmind_api_token(*, project_root=None, environ=None) -> str:
    env = os.environ if environ is None else environ
    env_token = str(env.get(FINMIND_API_TOKEN_ENV_VAR, "") or "").strip()
    if env_token:
        return env_token
    token_path = resolve_finmind_api_token_path(project_root)
    if not token_path.is_file():
        return ""
    return _extract_finmind_api_token_from_markdown(token_path.read_text(encoding="utf-8"))


# Backward-compatible startup snapshot; loader initialization resolves again so the
# canonical environment/file precedence remains owned by resolve_finmind_api_token().
API_TOKEN = resolve_finmind_api_token()
BASE_DIR = PROJECT_ROOT
RUNTIME_DOMAIN = RUNTIME_DOMAIN_TRADING
_DOMAIN_PATHS = resolve_runtime_domain_paths(BASE_DIR, domain=RUNTIME_DOMAIN)
SAVE_DIR = _DOMAIN_PATHS.data_dir


# # (AI註: 單一真理來源 - universe 名單路徑必須即時依 SAVE_DIR 推導，避免目錄重導後仍寫回舊路徑)
def get_universe_list_file_path():
    return os.path.join(SAVE_DIR, "universe_cache_v3.json")


FINMIND_PRICE_DATASET = 'TaiwanStockPriceAdj'
FINMIND_UNIVERSE_VOLUME_DATASET = 'TaiwanStockPriceAdj'
FINMIND_UNIVERSE_MARKET_VALUE_DATASET = 'TaiwanStockMarketValue'
OUTPUT_DIR = resolve_runtime_output_dir(BASE_DIR, domain=RUNTIME_DOMAIN, category='smart_downloader')

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
    IndexError,
    TypeError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    pd.errors.EmptyDataError,
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
        token = resolve_finmind_api_token()
        if token:
            dl.login_by_token(api_token=token)
    return dl


# # (AI註: 將執行期目錄建立延後到實際執行，避免被 import 時產生副作用)
def ensure_runtime_dirs():
    assert_runtime_write_path_is_not_research_dataset(PROJECT_ROOT, SAVE_DIR)
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

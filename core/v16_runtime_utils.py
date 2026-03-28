import inspect
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import get_context
from zoneinfo import ZoneInfo


TAIPEI_TIMEZONE_NAME = "Asia/Taipei"
TAIPEI_TIMEZONE = ZoneInfo(TAIPEI_TIMEZONE_NAME)
VALID_PROCESS_START_METHODS = ("spawn", "forkserver", "fork")
DEFAULT_PROCESS_START_METHODS_POSIX = ("forkserver", "spawn")
DEFAULT_PROCESS_START_METHODS_WINDOWS = ("spawn",)


# # (AI註: 台灣盤前/交易日推導統一使用 Asia/Taipei，避免容器或異地主機時區漂移)
def get_taipei_now():
    return datetime.now(TAIPEI_TIMEZONE)


# # (AI註: 檔案快取時間也統一轉成 Asia/Taipei aware datetime，避免 now 與 mtime 比較時口徑不一致)
def get_taipei_file_mtime(path):
    return datetime.fromtimestamp(os.path.getmtime(path), tz=TAIPEI_TIMEZONE)


# # (AI註: 互動式 prompt 與 headless 判斷集中管理，避免各工具腳本各自分叉)
def is_interactive_stdin():
    return bool(sys.stdin and sys.stdin.isatty())


# # (AI註: 非互動/CI/headless 執行時直接回傳預設值，避免 input() 在自動化環境拋 EOFError)
def safe_prompt(prompt_text, default_value):
    if not is_interactive_stdin():
        return default_value

    try:
        raw = input(prompt_text).strip()
    except EOFError:
        return default_value
    return raw if raw != "" else default_value


# # (AI註: 自動開瀏覽器只在明確可互動桌面環境啟用；容器/CI/無 GUI 時跳過，避免殘留 subprocess warning)
def should_auto_open_browser(environ=None):
    env = os.environ if environ is None else environ
    forced_flag = str(env.get("V16_AUTO_OPEN_BROWSER", "")).strip().lower()
    if forced_flag in {"0", "false", "n", "no", "off"}:
        return False
    if forced_flag in {"1", "true", "y", "yes", "on"}:
        return True

    if not is_interactive_stdin():
        return False

    if os.name == "nt":
        return True

    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


# # (AI註: ProcessPool 啟動方法集中管理；預設避開 fork，必要時可用環境變數 V16_PROCESS_START_METHOD 覆寫)
def resolve_process_start_method():
    requested = os.getenv("V16_PROCESS_START_METHOD", "").strip().lower()
    if requested:
        candidate_methods = (requested,)
    elif os.name == "nt":
        candidate_methods = DEFAULT_PROCESS_START_METHODS_WINDOWS
    else:
        candidate_methods = DEFAULT_PROCESS_START_METHODS_POSIX

    for method in candidate_methods:
        if method not in VALID_PROCESS_START_METHODS:
            continue
        try:
            get_context(method)
            return method
        except ValueError:
            continue

    return "spawn"


# # (AI註: 只有支援 mp_context 的 Python 版本才注入；保持相容性)
def get_process_pool_executor_kwargs():
    if "mp_context" not in inspect.signature(ProcessPoolExecutor).parameters:
        return {}, None

    start_method = resolve_process_start_method()
    return {"mp_context": get_context(start_method)}, start_method

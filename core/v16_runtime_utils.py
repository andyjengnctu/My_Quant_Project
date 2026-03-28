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


# # (AI註: 將互動式輸入判斷集中管理；非互動/EOF 時回退預設值，但保留管線餵值能力)
def is_interactive_stdin():
    stdin = getattr(sys, "stdin", None)
    if stdin is None:
        return False

    try:
        return bool(stdin.isatty())
    except (AttributeError, OSError, ValueError):
        return False


# # (AI註: input 例外與空字串預設值集中處理；不要在各工具各自維護一份 prompt 邏輯)
def safe_prompt(prompt_text, default_value):
    stdin = getattr(sys, "stdin", None)
    if stdin is None:
        return default_value

    try:
        raw = input(prompt_text).strip()
    except EOFError:
        return default_value
    return raw if raw != "" else default_value


# # (AI註: 整數型 CLI/ENV/UI 參數共用同一套驗證，避免各工具各自 int() 後噴 traceback)
def parse_int_strict(raw_value, *, field_name, min_value=None):
    text = str(raw_value).strip()
    if text == "":
        raise ValueError(f"{field_name} 不可空白。")

    try:
        value = int(text)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} 必須是整數，收到: {raw_value!r}") from e

    if min_value is not None and value < min_value:
        raise ValueError(f"{field_name} 不可小於 {min_value}，收到: {value}")

    return value


# # (AI註: 整數 prompt 直接共用 safe_prompt + parse_int_strict，避免重複維護輸入驗證)
def safe_prompt_int(prompt_text, default_value, *, field_name, min_value=None):
    raw = safe_prompt(prompt_text, str(default_value))
    return parse_int_strict(raw, field_name=field_name, min_value=min_value)


# # (AI註: 選單型 prompt 共用同一套 allowed-values 驗證，避免錯輸入被靜默吞掉)
def safe_prompt_choice(prompt_text, default_value, allowed_values, *, field_name, case_sensitive=False):
    normalized_allowed = [str(value).strip() for value in allowed_values]
    if not normalized_allowed:
        raise ValueError(f"{field_name} 的 allowed_values 不可為空。")

    raw = str(safe_prompt(prompt_text, str(default_value))).strip()
    if case_sensitive:
        if raw in normalized_allowed:
            return raw
    else:
        normalized_map = {value.lower(): value for value in normalized_allowed}
        normalized_raw = raw.lower()
        if normalized_raw in normalized_map:
            return normalized_map[normalized_raw]

    allowed_text = ", ".join(normalized_allowed)
    raise ValueError(f"{field_name} 只接受 {allowed_text}，收到: {raw!r}")


# # (AI註: 自動開瀏覽器只在互動且具圖形環境時啟用；可由環境變數 V16_AUTO_OPEN_BROWSER 強制覆寫)
def should_auto_open_browser(environ=None):
    env = os.environ if environ is None else environ
    override = str(env.get("V16_AUTO_OPEN_BROWSER", "")).strip().lower()
    if override in {"0", "false", "n", "no", "off"}:
        return False
    if override in {"1", "true", "y", "yes", "on"}:
        return True

    if not is_interactive_stdin():
        return False

    if os.name == "nt" or sys.platform == "darwin":
        return True

    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))

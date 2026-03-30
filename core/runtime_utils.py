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


def enable_line_buffered_stdout(stream=None):
    target = sys.stdout if stream is None else stream
    reconfigure = getattr(target, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(line_buffering=True, errors="replace")
        except TypeError:
            try:
                reconfigure(line_buffering=True)
            except (OSError, ValueError):
                pass
        except (OSError, ValueError):
            pass
    return target


def has_help_flag(argv):
    args = [] if argv is None else list(argv)
    return any(str(arg).strip() in {"-h", "--help"} for arg in args[1:])


def resolve_cli_program_name(argv, default_program):
    args = [] if argv is None else list(argv)
    if args:
        raw_program = str(args[0]).strip()
        if raw_program:
            return raw_program.replace("\\", "/")
    return default_program


def validate_cli_args(argv, *, value_options=(), flag_options=()):
    args = [] if argv is None else list(argv)
    allowed_value_options = {str(option).strip() for option in value_options}
    allowed_flag_options = {str(option).strip() for option in flag_options}
    allowed_flag_options.update({"-h", "--help"})

    idx = 1
    while idx < len(args):
        raw_arg = str(args[idx]).strip()
        if raw_arg == "":
            raise ValueError("CLI 參數不能為空字串。")

        option_name, has_inline_value, inline_value = raw_arg.partition("=")

        if option_name in allowed_value_options:
            if has_inline_value:
                if inline_value.strip() == "":
                    raise ValueError(f"{option_name}= 不能為空，請提供有效值。")
                idx += 1
                continue

            if idx + 1 >= len(args):
                raise ValueError(f"{option_name} 缺少值，請提供有效值。")

            next_value = str(args[idx + 1]).strip()
            if next_value == "" or next_value.startswith("-"):
                raise ValueError(f"{option_name} 缺少值，請提供有效值。")

            idx += 2
            continue

        if option_name in allowed_flag_options:
            if has_inline_value:
                raise ValueError(f"{option_name} 不接受值，收到: {inline_value}")
            idx += 1
            continue

        if raw_arg.startswith("-"):
            raise ValueError(f"不支援的參數: {raw_arg}")

        raise ValueError(f"不支援的位置參數: {raw_arg}")


def run_cli_entrypoint(main_func, *args, handled_exceptions=(FileNotFoundError, ValueError, RuntimeError), error_formatter=None, **kwargs):
    try:
        raise SystemExit(main_func(*args, **kwargs))
    except handled_exceptions as exc:
        formatter = error_formatter
        message = formatter(exc) if callable(formatter) else f"❌ {exc}"
        print(message, file=sys.stderr)
        raise SystemExit(1)


# # (AI註: input 例外與空字串預設值集中處理；不要在各工具各自維護一份 prompt 邏輯)
def safe_prompt(prompt_text, default_value):
    stdin = getattr(sys, "stdin", None)
    if stdin is None:
        return default_value

    try:
        if is_interactive_stdin():
            raw = input(prompt_text).strip()
        else:
            raw = stdin.readline()
            if raw == "":
                return default_value
            raw = raw.strip()
    except EOFError:
        return default_value
    except (OSError, ValueError):
        return default_value
    return raw if raw != "" else default_value


# # (AI註: 整數輸入驗證集中處理，避免各工具直接 int(...) 導致 traceback 或口徑分叉)
def parse_int_strict(raw_value, field_name, *, min_value=None, max_value=None):
    text = str(raw_value).strip()
    try:
        value = int(text)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} 必須是整數，收到: {raw_value}") from None

    if min_value is not None and value < min_value:
        raise ValueError(f"{field_name} 必須 >= {min_value}，收到: {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{field_name} 必須 <= {max_value}，收到: {value}")
    return value


# # (AI註: prompt + 整數解析共用單一入口；互動 / ENV / pipe 的錯誤訊息保持一致)
def safe_prompt_int(prompt_text, default_value, field_name, *, min_value=None, max_value=None):
    raw_value = safe_prompt(prompt_text, str(default_value))
    return parse_int_strict(raw_value, field_name, min_value=min_value, max_value=max_value)


# # (AI註: 選單輸入集中驗證，避免各工具把無效選項靜默吞掉)
def safe_prompt_choice(prompt_text, default_value, valid_choices, field_name):
    normalized_map = {str(choice).strip().upper(): str(choice).strip() for choice in valid_choices}
    raw_value = safe_prompt(prompt_text, str(default_value)).strip()
    normalized = raw_value.upper()
    if normalized not in normalized_map:
        valid_text = ", ".join(str(choice) for choice in valid_choices)
        raise ValueError(f"{field_name} 只接受 {valid_text}，收到: {raw_value}")
    return normalized_map[normalized]


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

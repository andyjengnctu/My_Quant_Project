import os
import re
import shutil
import sys
import unicodedata

C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_BLUE = '\033[94m'
C_GREEN = '\033[92m'
C_GRAY = '\033[90m'
C_RESET = '\033[0m'


def console_color_enabled(stream=None):
    """Return whether ANSI color is appropriate for the active console."""

    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    target = stream if stream is not None else getattr(sys, "stdout", None)
    return bool(target is not None and hasattr(target, "isatty") and target.isatty())


def format_elapsed(seconds):
    total_tenths = int(max(0.0, float(seconds)) * 10.0 + 0.5)
    hours, remainder = divmod(total_tenths, 36000)
    minutes, remainder = divmod(remainder, 600)
    seconds_value = remainder / 10.0
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_value:04.1f}"
    return f"{minutes:02d}:{seconds_value:04.1f}"


def render_elapsed(seconds, *, color=False):
    value = format_elapsed(seconds)
    return f"{C_CYAN}{value}{C_RESET}" if bool(color) else value


ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
HTML_SPAN_RE = re.compile(r'</?span\b[^>]*>', re.IGNORECASE)


def _strip_ansi(s):
    return ANSI_RE.sub('', str(s))


def _strip_display_markup(s):
    return HTML_SPAN_RE.sub('', _strip_ansi(s))


def _display_width(s):
    text = _strip_display_markup(s)
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
    return width


class InlineProgress:
    """Refresh one bounded terminal line; redirected output receives only the final state."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.inline = bool(getattr(self.stream, "isatty", lambda: False)())
        self._last_width = 0
        self._last_text = ""
        self._active = False

    def update(self, text):
        value = str(text).replace("\r", " ").replace("\n", " ")
        self._last_text = value
        if not self.inline:
            return
        columns = _terminal_content_width()
        fitted = _truncate_display_width(value, columns)
        visible_width = _display_width(fitted)
        padding = " " * max(0, min(self._last_width, columns) - visible_width)
        self.stream.write(f"\r{fitted}{padding}")
        self.stream.flush()
        self._last_width = visible_width
        self._active = True

    def print_line(self, text):
        if self.inline and self._active:
            clear_width = min(self._last_width, _terminal_content_width())
            self.stream.write("\r" + (" " * clear_width) + "\r")
        print(str(text), file=self.stream, flush=True)
        self._last_width = 0
        self._active = False

    def finish(self, text=None):
        if self.inline:
            if text is not None:
                if self._active:
                    clear_width = min(self._last_width, _terminal_content_width())
                    self.stream.write("\r" + (" " * clear_width) + "\r")
                value = str(text).replace("\r", " ").replace("\n", " ")
                print(value, file=self.stream, flush=True)
            elif self._active:
                self.stream.write("\n")
                self.stream.flush()
        else:
            if text is not None:
                self._last_text = str(text).replace("\r", " ").replace("\n", " ")
            if self._last_text:
                print(self._last_text, file=self.stream, flush=True)
        self._last_width = 0
        self._active = False


def _terminal_content_width():
    return max(1, int(shutil.get_terminal_size(fallback=(100, 24)).columns) - 1)


def _truncate_display_width(text, max_width):
    raw = str(text)
    limit = max(1, int(max_width))
    if _display_width(raw) <= limit:
        return raw

    target_width = max(0, limit - 1)
    output = []
    visible_width = 0
    index = 0
    contains_ansi = False
    while index < len(raw):
        ansi_match = ANSI_RE.match(raw, index)
        if ansi_match is not None:
            output.append(ansi_match.group(0))
            contains_ansi = True
            index = ansi_match.end()
            continue
        char = raw[index]
        char_width = _display_width(char)
        if visible_width + char_width > target_width:
            break
        output.append(char)
        visible_width += char_width
        index += 1
    output.append("…")
    if contains_ansi:
        output.append(C_RESET)
    return "".join(output)


def _pad_display(s, width, align='left'):
    s = str(s)
    visible = _display_width(s)
    pad = max(0, width - visible)
    if align == 'right':
        return ' ' * pad + s
    return s + ' ' * pad


def _table_row(c1, c2, c3, c4, w1=16, w=16):
    return (
        f"| {_pad_display(c1, w1)} "
        f"| {_pad_display(c2, w)} "
        f"| {_pad_display(c3, w)} "
        f"| {_pad_display(c4, w)} |"
    )


def get_p(params, key, default=None):
    if isinstance(params, dict):
        return params.get(key, default)
    return getattr(params, key, default)

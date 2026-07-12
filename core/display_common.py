import re
import sys
import unicodedata

C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_BLUE = '\033[94m'
C_GREEN = '\033[92m'
C_GRAY = '\033[90m'
C_RESET = '\033[0m'


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


def _strip_ansi(s):
    return ANSI_RE.sub('', str(s))


def _display_width(s):
    text = _strip_ansi(s)
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
    return width


class InlineProgress:
    """Refresh one terminal line; redirected output receives only the final state."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.inline = bool(getattr(self.stream, "isatty", lambda: False)())
        self._last_width = 0
        self._last_text = ""
        self._active = False

    def update(self, text):
        value = str(text)
        self._last_text = value
        if not self.inline:
            return
        visible_width = _display_width(value)
        padding = " " * max(0, self._last_width - visible_width)
        self.stream.write(f"\r{value}{padding}")
        self.stream.flush()
        self._last_width = visible_width
        self._active = True

    def print_line(self, text):
        if self.inline and self._active:
            self.stream.write("\r" + (" " * self._last_width) + "\r")
        print(str(text), file=self.stream, flush=True)
        self._last_width = 0
        self._active = False

    def finish(self, text=None):
        if text is not None:
            self.update(text)
        if self.inline:
            if self._active:
                self.stream.write("\n")
                self.stream.flush()
        elif self._last_text:
            print(self._last_text, file=self.stream, flush=True)
        self._last_width = 0
        self._active = False


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

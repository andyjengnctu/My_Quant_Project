import re
import unicodedata

C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_GRAY = '\033[90m'
C_RESET = '\033[0m'

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

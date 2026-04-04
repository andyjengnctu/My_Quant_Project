import io
from contextlib import redirect_stderr, redirect_stdout


def suppress_tool_output(func, *args, **kwargs):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        return func(*args, **kwargs)


def resolve_source_date_column(source_df, ticker):
    if "Time" in source_df.columns:
        return "Time"
    if "Date" in source_df.columns:
        return "Date"
    raise KeyError(f"{ticker}: 找不到 Date/Time 欄位")

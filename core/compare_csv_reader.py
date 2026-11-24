"""Open resolved comparison details under the cleaner's publication lock."""
from contextlib import contextmanager, nullcontext
from pathlib import Path

from core.compare_csv_storage import DETAIL_NAMES, compare_root, detail_read_path, storage_lock


@contextmanager
def open_detail(path: str | Path):
    """Pin the resolved file descriptor before cleanup can unlink its name.

    Parsing occurs after releasing the short publication lock. An open file
    remains readable after unlink on POSIX; Windows may refuse the cleaner's
    unlink while the descriptor is in use, in which case cleanup skips it.
    """
    path = Path(path)
    root = compare_root(path) if path.name in DETAIL_NAMES else None
    stream = None
    try:
        with storage_lock(root) if root is not None else nullcontext():
            stream = detail_read_path(path).open("rb")
        yield stream
    finally:
        if stream is not None:
            stream.close()


def read_detail_csv(path: str | Path, **options):
    """Eager pandas CSV read with the original parser options unchanged."""
    import pandas as pd
    if options.get("iterator") or options.get("chunksize") is not None:
        raise ValueError("Chunked comparison reads must keep open_detail() active")
    with open_detail(path) as stream:
        return pd.read_csv(stream, **options)

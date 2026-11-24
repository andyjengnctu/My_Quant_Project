"""Storage operations for immutable, published Strategy Compare detail CSVs."""
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

from core.storage_paths import clean_path, plain
from core.compare_csv_lineage import DETAIL_NAMES, compare_root, reused_detail_source
from core.reused_copy_storage import (
    detail_exists, detail_read_path, detail_sha256, inherited_detail_files,
)

_LOG = logging.getLogger(__name__)
RETIRED_PAIR_EXPORTS = ("strategy_comparison.html",)


@contextmanager
def storage_lock(root):
    """Serialize only short publication/replacement operations, never data hashing."""
    path = root / "outputs/project_cleanup/compare_csv.lock"
    if not clean_path(path) or (path.exists() and path.stat().st_nlink != 1):
        raise OSError("清理鎖路徑異常")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if not os.fstat(stream.fileno()).st_size:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            until = time.monotonic() + 30
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= until:
                        raise
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield  # Closing the descriptor releases the OS lock, including after a crash.


@contextmanager
def comparison_publication_lock(path):
    """Serialize owned pair/run JSON commits with byte-alias cleanup commits."""
    path = Path(path).absolute()
    root = compare_root(path)
    if root is None and path.name in {"strategy_comparison.json", "manifest.json"} and path.parent.parent.name == "runs":
        root = compare_root(path.parent / "pairs/_publication/no_filter_orderable_candidates.csv")
    if root is None:
        yield
    else:
        with storage_lock(root):
            yield


def publish_detail_csv(frame, path, **options):
    """A producer must never truncate an inode shared by prior results."""
    path = Path(path).absolute()
    root = compare_root(path)
    if root is None or path.name not in DETAIL_NAMES:
        return frame.to_csv(path, **options)
    from core.file_integrity import atomic_replace_with_retry
    with tempfile.TemporaryDirectory(prefix=".csv-write-", dir=path.parent) as work:
        tmp = Path(work) / path.name
        frame.to_csv(tmp, **options)
        with storage_lock(root):
            atomic_replace_with_retry(tmp, path)


def file_metadata(path, info):
    """Skip unhandled streams/file attributes. Preserve ACLs, mode and timestamps."""
    if not plain(info) or not info.st_ino or not clean_path(path):
        raise ValueError("非一般實體檔案")
    if os.name == "nt":
        excluded = (stat.FILE_ATTRIBUTE_ENCRYPTED | stat.FILE_ATTRIBUTE_OFFLINE
                    | stat.FILE_ATTRIBUTE_SYSTEM | stat.FILE_ATTRIBUTE_READONLY)
        if info.st_file_attributes & excluded:
            raise ValueError("加密、離線、系統或唯讀檔案")
        from core.ntfs_cleanup_info import security_key
        return (info.st_file_attributes, security_key(path))
    xattrs = tuple((key, os.getxattr(path, key)) for key in sorted(os.listxattr(path)))
    return (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode), xattrs)


def disk_bytes(path, info):
    if os.name == "nt":
        from core.ntfs_cleanup_info import allocation
        return allocation(path)
    return info.st_blocks * 512


def copy_reused_detail(src, dst):
    """copytree calls this with a freshly created destination. Fall back to copy2."""
    source, dest = Path(src), Path(dst)
    root = compare_root(source)
    if root and root == compare_root(dest) and source.name == dest.name and source.name in DETAIL_NAMES:
        try:
            if not clean_path(source) or not clean_path(dest) or dest.exists():
                raise ValueError("non-fresh destination")
            from core.file_integrity import file_change_signature
            source_revision = file_change_signature(source)
            # A new file in this directory supplies the intended inherited ACL.
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as stream:
                probe = Path(stream.name)
            try:
                shutil.copystat(source, probe)
                if file_metadata(source, source.lstat()) != file_metadata(probe, probe.lstat()):
                    raise ValueError("不同檔案安全屬性")
            finally:
                if os.name == "nt":
                    os.chmod(probe, stat.S_IWRITE)
                probe.unlink(missing_ok=True)
            with storage_lock(root):
                if file_change_signature(source) != source_revision:
                    raise ValueError("來源已變更")
                os.link(source, dest)
            return str(dest)
        except (OSError, ValueError) as exc:
            _LOG.debug(
                "策略比較 CSV 改用複製：%s → %s (%s: %s)",
                source.absolute().relative_to(root).as_posix(),
                dest.absolute().relative_to(root).as_posix(),
                type(exc).__name__, str(exc).replace(str(root), "."),
            )
    if dest.exists() and dest.name in DETAIL_NAMES and compare_root(dest):
        with tempfile.TemporaryDirectory(dir=dest.parent) as work:
            tmp = Path(work) / dest.name
            shutil.copy2(src, tmp)
            with storage_lock(compare_root(dest)):
                os.replace(tmp, dest)
        return str(dest)
    return shutil.copy2(src, dst)

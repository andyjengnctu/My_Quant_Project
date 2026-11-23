"""Storage operations for immutable, published Strategy Compare detail CSVs."""
from contextlib import contextmanager
import logging
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

_LOG = logging.getLogger(__name__)
RETIRED_PAIR_EXPORTS = ("strategy_comparison.html",)

DETAIL_NAMES = frozenset(
    f"{side}_{detail}.csv" for side in ("no_filter", "score_ranking")
    for detail in ("orderable_candidates", "orderable_target_diagnostics")
)


def compare_root(path):
    """Only the canonical .../runs/<run>/pairs/<pair>/<file> output namespace."""
    p = Path(path).absolute()
    if len(p.parents) >= 7 and p.parents[1].name == "pairs" and p.parents[3].name == "runs":
        # Current and historical profile roots have different depths. The run
        # and pair publication boundaries, not a fixed depth, establish ownership.
        for parent in p.parents[4:]:
            if parent.name == "strategy_compare" and parent.parent.name == "outputs":
                return parent.parent.parent
    return None


def plain(info):
    return (stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
            and not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def clean_path(path):
    return Path(path).absolute() == Path(path).resolve(strict=False)


def reused_detail_source(path, *, revisions=None):
    """Follow the producer's existing REUSE provenance; create no new storage format."""
    path = Path(path).absolute()
    root = compare_root(path)
    if root is None or path.name not in DETAIL_NAMES:
        return None
    seen = {path}
    while True:
        marker = path.parent / "strategy_comparison.json"
        if not clean_path(marker) or not marker.is_file():
            return None
        from core.file_integrity import file_change_signature
        before = file_change_signature(marker)
        payload = json.loads(marker.read_text(encoding="utf-8-sig"))
        if file_change_signature(marker) != before:
            raise ValueError("讀取 REUSE 來源紀錄時內容變動")
        if revisions is not None:
            revisions[marker] = before
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        source = metadata.get("cache_reused_from") if isinstance(metadata, dict) else None
        if not isinstance(source, str) or not source.strip():
            return None
        parent = Path(source)
        if parent.is_absolute():
            raise ValueError("舊 REUSE 紀錄使用絕對來源路徑；保留副本，避免專案搬移後失去來源")
        path = (root / parent / path.name).absolute()
        if (path in seen or compare_root(path) != root or not clean_path(path)):
            raise ValueError("REUSE 明細來源循環或離開正式專案路徑")
        seen.add(path)
        if path.is_file():
            if not plain(path.lstat()):
                raise ValueError("REUSE 明細來源不是一般檔案")
            return path


def detail_read_path(path):
    path = Path(path)
    if path.is_file():
        return path
    source = reused_detail_source(path)
    if source is None:
        return path
    # For pinned publications the inherited bytes must still match THIS pair's
    # existing receipt, even when an audit reads the CSV directly.
    from core.file_integrity import compute_observed_file_sha256, load_json_object_or_none
    payload = load_json_object_or_none(path.parents[2] / "strategy_comparison.json")
    record = payload.get("pair_file_integrity") if isinstance(payload, dict) else None
    if record is not None:
        try:
            expected = record["pairs"][path.parent.name][path.name]
        except (KeyError, TypeError) as exc:
            raise ValueError("REUSE 明細缺少原有的完整性收據") from exc
        if compute_observed_file_sha256(source) != expected:
            raise ValueError("REUSE 明細來源與發布時內容不同")
    return source


def detail_exists(path):
    try:
        return detail_read_path(path).is_file()
    except (OSError, ValueError):
        return False


def detail_sha256(path, *, file_hasher=None):
    from core.file_integrity import compute_file_sha256
    return (file_hasher or compute_file_sha256)(detail_read_path(path))


def inherited_detail_files(pair_dir):
    """Only existing REUSE links may supply missing published detail filenames."""
    return {name: Path(pair_dir) / name for name in DETAIL_NAMES
            if detail_exists(Path(pair_dir) / name)}


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

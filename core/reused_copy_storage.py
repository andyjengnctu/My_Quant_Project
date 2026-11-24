"""Logical reads and byte-preserving proofs for producer-owned CSV aliases.

This is not a scientific completion or reuse record. The producer's original
pair JSON proves the copy operation; the separate cleanup receipt pins the
original CSV bytes even when its completed source predates scientific hashes.
"""
from copy import deepcopy
from contextvars import ContextVar
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from core.storage_paths import clean_path, plain, relative_path
from core.compare_csv_lineage import (
    DETAIL_NAMES, _recorded_detail_source, compare_root, reused_detail_source,
)
from core.strategy_compare_storage_evidence import (
    completed_detail_storage_evidence as _completed_detail_storage_evidence,
)
from core.file_integrity import (
    atomic_write_json, compute_file_sha256, compute_observed_file_sha256, file_change_signature,
    is_sha256_hex, load_json_strict, load_json_object_or_none,
)

_SCHEMA = "unfinished_reused_copy_storage_v1"
_BASELINE_NAME = "no_filter_orderable_candidates.csv"
_LOCATION_FIELDS = ("output_scope", "output_dir", "cache_reused_from")
_READING = ContextVar("reused_copy_storage_reading", default=frozenset())


def root_for(path):
    path = Path(path).absolute()
    if (path.name not in DETAIL_NAMES or len(path.parents) < 5
            or path.parents[1].name != "pairs" or path.parents[3].name != "runs"):
        return None
    return compare_root(path)


def receipt_path(path):
    path = Path(path).absolute()
    root = root_for(path)
    if root is None:
        raise ValueError("不是正式 Strategy Compare 的逐檔副本位置")
    identity = sha256(path.relative_to(root).as_posix().encode("utf-8")).hexdigest()
    return root / "outputs/project_cleanup/reused_copies" / (identity + ".json")


def _remember(path, revisions):
    if not clean_path(path):
        raise ValueError("副本證據路徑經過連結")
    current = file_change_signature(path)
    if revisions.setdefault(path, current) != current:
        raise ValueError("副本證據在驗證期間變動")
    return current


def _read(path, revisions):
    before = _remember(path, revisions)
    if not plain(path.lstat()):
        raise ValueError("副本證據不是一般實體 JSON")
    payload = load_json_strict(path)
    if not isinstance(payload, dict) or file_change_signature(path) != before:
        raise ValueError("副本證據格式錯誤或讀取期間變動")
    return payload


def _hash(path, revisions, file_hasher):
    before = _remember(path, revisions)
    if not plain(path.lstat()):
        raise ValueError("明細不是一般實體檔案")
    digest = file_hasher(path)
    if not is_sha256_hex(digest) or file_change_signature(path) != before:
        raise ValueError("明細內容在驗證期間變動")
    return digest


def _metadata(path, revisions):
    root = root_for(path)
    if root is None:
        raise ValueError("不是正式 Strategy Compare 的逐檔副本位置")
    pair = _read(path.parent / "strategy_comparison.json", revisions)
    metadata = pair.get("metadata")
    if (not isinstance(metadata, dict)
            or metadata.get("output_dir") != path.parent.relative_to(root).as_posix()):
        raise ValueError("pair 尚未完成 producer 的 REUSE 副本發布")
    # A whole-pair copy and an independently executed arm with a reused
    # baseline are different producer contracts. Only the former can stand
    # in for all four details; baseline provenance covers candidate rows only.
    source_value = metadata.get("cache_reused_from")
    if source_value is not None:
        if metadata.get("output_scope") != "reused_pair_cache":
            raise ValueError("整份 REUSE 副本的 producer scope 不一致")
    elif (path.name == _BASELINE_NAME and metadata.get("baseline_reused") is True
            and metadata.get("output_scope") == "caller_override"
            and metadata.get("comparison_mode") == "score-ranking"):
        source_value = metadata.get("baseline_reuse_source")
    else:
        raise ValueError("pair 缺少適用於此檔案的明確 REUSE 副本證據")
    source = root / relative_path(source_value) / path.name
    if (root_for(source) != root or source == path or source.parents[2] == path.parents[2]):
        raise ValueError("REUSE 副本來源不是另一個正式 run")
    _remember(source, revisions)
    return pair, source


def _completed(path, revisions, file_hasher):
    run = path.parents[2]
    _read(run / "manifest.json", revisions)
    payload = _read(run / "strategy_comparison.json", revisions)
    pair = _read(path.parent / "strategy_comparison.json", revisions)
    completed_detail_storage_evidence(
        path=path, run_payload=payload, project_root=root_for(path), file_hasher=file_hasher,
    )
    return pair


def _same_copied_payload(target, source):
    copies = [deepcopy(value) for value in (target, source)]
    for value in copies:
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            return False
        for field in _LOCATION_FIELDS:
            metadata.pop(field, None)
    return copies[0] == copies[1]


def _physical_source(path, revisions, file_hasher):
    """Validate every completed publication encountered in recorded lineage."""
    if path.is_file():
        return path
    observed = {}
    source = reused_detail_source(path, revisions=observed)
    if source is None:
        raise ValueError("完整發布來源沒有可讀的原始明細")
    for marker, stamp in observed.items():
        if revisions.setdefault(marker, stamp) != stamp:
            raise ValueError("REUSE 來源鏈在驗證期間變動")
        candidate = marker.parent / path.name
        if root_for(candidate) != root_for(path):
            raise ValueError("REUSE 來源鏈離開正式比較範圍")
        _remember(candidate, revisions)
        _completed(candidate, revisions, file_hasher)
    _completed(source, revisions, file_hasher)
    return source


def _receipt(path, revisions):
    """Validate receipt identity/provenance without granting scientific status."""
    receipt = receipt_path(path)
    revision = _remember(receipt, revisions)
    if revision == ("missing",):
        return None
    payload = _read(receipt, revisions)
    root = root_for(path)
    pair, direct = _metadata(path, revisions)
    if (set(payload) != {"schema", "target", "source", "copied_from", "sha256"}
            or payload.get("schema") != _SCHEMA
            or payload.get("target") != path.relative_to(root).as_posix()
            or payload.get("copied_from") != direct.parent.relative_to(root).as_posix()
            or not is_sha256_hex(payload.get("sha256"))):
        raise ValueError("未完成副本的儲存收據與原始位置／來源不同")
    source = root / relative_path(payload.get("source"))
    if root_for(source) != root or source == path or source.name != path.name:
        raise ValueError("未完成副本的儲存收據來源無效")
    return payload, source


def _unchanged(revisions):
    if any(not clean_path(path) or file_change_signature(path) != stamp
           for path, stamp in revisions.items()):
        raise ValueError("副本、來源或發布紀錄在驗證後變動")


def publication_evidence(path, *, file_hasher=compute_observed_file_sha256):
    """Return revisions and original SHA; never accept a partial/invalid run.

    Revisions include the two absent publication markers. The cleaner checks
    them again under storage_lock before persisting a receipt and unlinking.
    """
    path = Path(path).absolute()
    if root_for(path) is None:
        raise ValueError("不是正式 Strategy Compare 的逐檔副本位置")
    revisions = {}
    for marker in (path.parents[2] / "manifest.json", path.parents[2] / "strategy_comparison.json"):
        if _remember(marker, revisions) != ("missing",):
            raise ValueError("run 已有部分或完整發布紀錄，不適用未完成副本證明")
    pair, direct = _metadata(path, revisions)
    source_pair = _completed(direct, revisions, file_hasher)
    if (pair["metadata"].get("cache_reused_from") is not None
            and not _same_copied_payload(pair, source_pair)):
        raise ValueError("pair 內容超出 producer 三個位置欄位的變更，不能視為原始整份副本")
    source = _physical_source(direct, revisions, file_hasher)
    digest = _hash(path, revisions, file_hasher)
    if _hash(source, revisions, file_hasher) != digest:
        raise ValueError("未完成副本與完整發布來源不是相同 bytes")
    prior = _receipt(path, revisions)
    if prior is not None and prior[0]["sha256"] != digest:
        raise ValueError("未完成副本已偏離原有儲存 SHA256 收據")
    _unchanged(revisions)
    return revisions, digest


def record_alias(path, source, digest, *, file_hasher=compute_observed_file_sha256):
    """Pin bytes before unlink; caller holds storage_lock and may cache hashes.

    The caller's revision-keyed file_hasher avoids rereading large CSVs inside
    the short commit lock after publication_evidence checked their full bytes.
    """
    path, source = Path(path).absolute(), Path(source).absolute()
    root = root_for(path)
    if root is None or root_for(source) != root or source.name != path.name or source == path:
        raise ValueError("未完成副本的保留來源位置無效")
    revisions, original = publication_evidence(path, file_hasher=file_hasher)
    if digest != original:
        raise ValueError("未完成副本的 SHA256 與原始 bytes 不同")
    # The cleaner may choose an older identical, physical ancestor. Require it
    # to occur in the producer lineage, not merely have coincidentally equal CSV.
    current, seen = path, {path}
    while current != source:
        current = reused_detail_source(current, revisions=revisions)
        if current is None or current in seen:
            raise ValueError("保留來源不在 producer 的 REUSE 來源鏈")
        seen.add(current)
        _completed(current, revisions, file_hasher)
        if _hash(current, revisions, file_hasher) != digest:
            raise ValueError("REUSE 來源鏈有不同 bytes")
    receipt = receipt_path(path)
    if not clean_path(receipt) or (receipt.exists() and (not plain(receipt.lstat()) or receipt.stat().st_nlink != 1)):
        raise ValueError("未完成副本儲存收據路徑不安全")
    _, direct = _metadata(path, revisions)
    _unchanged(revisions)
    payload = {"schema": _SCHEMA, "target": path.relative_to(root).as_posix(),
               "source": source.relative_to(root).as_posix(),
               "copied_from": direct.parent.relative_to(root).as_posix(), "sha256": digest}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(receipt, payload)


def _receipt_detail_source(path):
    """Resolve an existing storage receipt, rejecting missing/changed bytes."""
    path = Path(path).absolute()
    if root_for(path) is None:
        return None
    revisions = {}
    record = _receipt(path, revisions)
    if record is None:
        return None
    payload, source = record
    _completed(source, revisions, compute_observed_file_sha256)
    actual = _physical_source(source, revisions, compute_observed_file_sha256)
    if _hash(actual, revisions, compute_observed_file_sha256) != payload["sha256"]:
        raise ValueError("REUSE 保留來源與副本原有的儲存 SHA256 不同")
    _unchanged(revisions)
    return actual


def receipt_detail_source(path):
    """Resolve receipts without allowing recursive or cyclic storage aliases."""
    path = Path(path).absolute()
    reading = _READING.get()
    if path in reading:
        raise ValueError("未完成副本的儲存收據來源循環")
    token = _READING.set(reading | {path})
    try:
        return _receipt_detail_source(path)
    finally:
        _READING.reset(token)


class _UnavailableRunReceipt(Exception):
    """A failed read is absent for this call, never a reusable observation."""


@lru_cache(maxsize=16)
def _run_pair_integrity_record(path: Path, revision: tuple):
    """Retain only the small receipt, never the run's large result tables."""
    payload = load_json_object_or_none(path)
    if file_change_signature(path) != revision:
        raise ValueError("REUSE 完整性收據在讀取期間變動")
    if not isinstance(payload, dict):
        raise _UnavailableRunReceipt
    return payload.get("pair_file_integrity")


def _current_run_pair_integrity_record(path: Path):
    # AI: Cache parsing, not authorization. Re-observe the physical revision on
    # every read, including Windows ChangeTime for restored-mtime rewrites.
    revision = file_change_signature(path)
    try:
        record = _run_pair_integrity_record(path, revision)
    except _UnavailableRunReceipt:
        return None
    if file_change_signature(path) != revision:
        raise ValueError("REUSE 完整性收據在驗證期間變動")
    return record


def detail_read_path(path):
    path = Path(path)
    if path.is_file():
        return path
    source = receipt_detail_source(path)
    if source is None:
        source = reused_detail_source(path)
    if source is None:
        return path
    # For pinned publications the inherited bytes must still match THIS pair's
    # existing receipt, even when an audit reads the CSV directly.
    from core.file_integrity import compute_observed_file_sha256
    from core import caller_baseline_storage as caller
    if caller.root_for(path) is not None:
        if compute_observed_file_sha256(source) != caller.expected_sha256(path):
            raise ValueError("caller baseline 來源與原有儲存收據不同")
        return source
    record = _current_run_pair_integrity_record(path.parents[2] / "strategy_comparison.json")
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


def inherited_detail_files(pair_dir, *, present_names):
    """Resolve only names absent from the caller's fresh directory inventory."""
    pair_dir = Path(pair_dir)
    missing = DETAIL_NAMES.difference(present_names)
    if not missing or compare_root(pair_dir / "strategy_comparison.json") is None:
        return {}
    # An original pair may legitimately omit optional diagnostics. Without
    # recorded REUSE provenance there is nothing to inherit. Do not re-stat
    # files whose presence/type the caller already obtained from scandir.
    try:
        payload = json.loads((pair_dir / "strategy_comparison.json").read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    return {name: pair_dir / name for name in missing
            if _recorded_detail_source(metadata, name) is not None and detail_exists(pair_dir / name)}


def completed_detail_storage_evidence(
    *, path: Path, run_payload: Mapping[str, Any], project_root: Path,
    file_hasher: Callable[[Path], str] = compute_file_sha256,
) -> str | None:
    """Compose publication evidence with the same alias-aware reader as consumers."""
    return _completed_detail_storage_evidence(
        path=path, run_payload=run_payload, project_root=project_root,
        detail_hasher=lambda logical_path: detail_sha256(logical_path, file_hasher=file_hasher),
    )

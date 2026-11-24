"""Strategy Compare publication boundaries and per-file storage evidence.

This shared storage owner verifies existing publication receipts. Scientific
whole-run reuse and pair-file inventory remain in the research service, which
imports these same checks rather than maintaining a second completion rule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from core.compare_csv_storage import DETAIL_NAMES, clean_path, compare_root, detail_sha256, plain
from core.file_integrity import compute_file_sha256, is_sha256_hex, load_json_object_or_none

PAIR_FILE_INTEGRITY_FIELD = "pair_file_integrity"
PAIR_FILE_INTEGRITY_SCHEMA_VERSION = 1
_PRIMARY_PAIR_JSON = "strategy_comparison.json"


def _completed_run_manifest(run_dir: Path, run_payload: Mapping[str, Any], project_root: Path) -> dict:
    """Shared commit-boundary check; callers choose run or storage-file scope."""
    directory = Path(run_dir)
    manifest_path = directory / "manifest.json"
    if not clean_path(manifest_path) or not manifest_path.is_file() or not plain(manifest_path.lstat()):
        raise ValueError("manifest 缺少或不是專案內的一般檔案")
    manifest = load_json_object_or_none(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(run_payload, Mapping):
        raise ValueError("manifest／run 結果不是有效 JSON object")
    if str(run_payload.get("status") or "") != "COMPLETED":
        raise ValueError("run.status 不是 COMPLETED")
    fingerprint = str(run_payload.get("config_fingerprint") or "")
    if len(fingerprint) != 12 or manifest.get("config_fingerprint") != fingerprint:
        raise ValueError("manifest 與 run 的 config_fingerprint 缺少或不同")
    for key in (
        "report_id", "report_version", "report_contract_fingerprint",
        "pair_file_integrity", "pair_execution", "comparison_period", "artifact_identities",
    ):
        if key in run_payload or key in manifest:
            if run_payload.get(key) != manifest.get(key):
                raise ValueError(f"manifest 與 run 的 {key} 缺少或不同")
    recorded_run_dir = str(manifest.get("run_dir") or "").strip()
    if recorded_run_dir:
        recorded_path = Path(recorded_run_dir)
        if not recorded_path.is_absolute():
            recorded_path = Path(project_root) / recorded_path
        if recorded_path.resolve() != directory.resolve():
            raise ValueError("manifest.run_dir 指向其他位置")
    return manifest


def completed_detail_storage_evidence(
    *, path: Path, run_payload: Mapping[str, Any], project_root: Path,
    file_hasher: Callable[[Path], str] = compute_file_sha256,
) -> str | None:
    """Prove this optional detail was committed, without accepting a whole run.

    Only byte-identical duplicates with an existing reader fallback may be
    removed by the caller. Unrelated pair/report failures remain failures for
    scientific REUSE and are not rewritten or repaired by this storage check.
    The return value is its existing pinned SHA, or None for legacy unpinned.
    """
    path = Path(path)
    root = Path(project_root).resolve()
    if path.name not in DETAIL_NAMES or compare_root(path) != root or not clean_path(path):
        raise ValueError("不是正式比較範圍內可追溯來源的明細")
    run_dir, pair_id = path.parents[2], path.parent.name
    _completed_run_manifest(run_dir, run_payload, root)
    pairs = run_payload.get("pairs")
    if not isinstance(pairs, dict) or not isinstance(pairs.get(pair_id), dict):
        raise ValueError("run.pairs 缺少此 pair 的正式結果")
    pair_json = path.parent / _PRIMARY_PAIR_JSON
    if (not clean_path(pair_json) or not pair_json.is_file() or not plain(pair_json.lstat())
            or load_json_object_or_none(pair_json) != pairs[pair_id]):
        raise ValueError("pair JSON 缺少或與 run.pairs 不同")
    record = run_payload.get(PAIR_FILE_INTEGRITY_FIELD)
    if record is None:
        return None
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != PAIR_FILE_INTEGRITY_SCHEMA_VERSION):
        raise ValueError("pair_file_integrity schema 無效")
    pinned_pairs = record.get("pairs")
    if not isinstance(pinned_pairs, dict) or set(pinned_pairs) != set(pairs):
        raise ValueError("pair_file_integrity 的 pair 清單不完整")
    digests = pinned_pairs.get(pair_id)
    digest = digests.get(path.name) if isinstance(digests, dict) else None
    if not is_sha256_hex(digest):
        raise ValueError("此 CSV 缺少有效的發布 SHA256")
    expected = str(digest).strip().lower()
    if detail_sha256(path, file_hasher=file_hasher) != expected:
        raise ValueError("此 CSV 內容與已發布 SHA256 不符")
    return expected



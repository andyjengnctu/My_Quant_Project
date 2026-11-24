"""Storage proofs for completed, frozen threshold-selection replay baselines.

The existing selection receipt is read-only scientific evidence. A separate
small storage receipt pins bytes before a duplicate baseline CSV is removed.
Neither receipt grants scientific REUSE or changes the original research state.
"""
from pathlib import Path
import re

from core.storage_paths import clean_path, plain, relative_path
from core.file_integrity import (
    atomic_write_text, canonical_json_sha256, compute_observed_file_sha256,
    file_change_signature, load_json_strict,
)

BASELINE_NAME = "no_filter_orderable_candidates.csv"
STORAGE_NAME = "baseline_storage.json"
_SCHEMA = "caller_baseline_storage_v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY = re.compile(r"[0-9a-f]{16}\Z")
_MASK = re.compile(r"mask_([0-9]+)\Z")
_SEED = re.compile(r"seed_([0-9]+)\Z")


def root_for(path: Path) -> Path | None:
    """Recognize only the bounded namespace with an existing frozen contract."""
    path = Path(path).absolute()
    if path.name not in {BASELINE_NAME, "strategy_comparison.json"} or len(path.parents) < 10:
        return None
    p = path.parents
    if (p[0].name != "replay" or not _MASK.fullmatch(p[1].name)
            or not _IDENTITY.fullmatch(p[2].name) or p[3].name != "selection"
            or not _SEED.fullmatch(p[4].name)
            or (p[6].name, p[7].name, p[8].name) != ("positive_threshold", "optimizer", "outputs")):
        return None
    return p[9]


def _read(path: Path, revisions: dict) -> dict:
    if not clean_path(path) or not plain(path.lstat()):
        raise ValueError("caller baseline 收據不是一般實體檔案")
    before = file_change_signature(path)
    payload = load_json_strict(path)
    if not isinstance(payload, dict) or file_change_signature(path) != before:
        raise ValueError("caller baseline 收據格式錯誤或讀取時變動")
    previous = revisions.setdefault(path, before)
    if previous != before:
        raise ValueError("caller baseline 收據在來源驗證期間變動")
    return payload


def _publication(path: Path, revisions: dict) -> dict:
    root = root_for(path)
    if root is None or path.name != BASELINE_NAME:
        raise ValueError("不是受支援的 caller baseline 儲存位置")
    if not clean_path(path) or not path.resolve().is_relative_to(root):
        raise ValueError("caller baseline 路徑經過連結或離開專案")
    selection = path.parents[2]
    frozen = _read(selection / "frozen_threshold.json", revisions)
    receipt_sha = frozen.get("receipt_sha256")
    receipt_body = {key: value for key, value in frozen.items() if key != "receipt_sha256"}
    identity = frozen.get("selection_identity")
    if (frozen.get("status") != "FROZEN" or not isinstance(receipt_sha, str)
            or not _HEX.fullmatch(receipt_sha)
            or canonical_json_sha256(receipt_body) != receipt_sha
            or not isinstance(identity, dict)
            or canonical_json_sha256(identity, length=16) != selection.name):
        raise ValueError("門檻選擇缺少一致的既有 FROZEN 收據")
    spec = identity.get("spec")
    if (not isinstance(spec, dict)
            or spec.get("output_subdir") != path.parents[5].relative_to(root).as_posix()
            or type(identity.get("seed")) is not int
            or identity["seed"] != int(_SEED.fullmatch(path.parents[4].name)[1])):
        raise ValueError("門檻收據與工作目錄／Seed 不一致")
    payload = _read(path.parent / "strategy_comparison.json", revisions)
    # This caller persisted the engine's returned payload after its output
    # publication completed. Require exact equality, not just a JSON filename.
    if _read(path.parents[1] / "result.json", revisions) != payload:
        raise ValueError("caller 完成結果與 replay 發布內容不同")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("caller replay 缺少 metadata")
    options = metadata.get("score_ranking_options")
    if not isinstance(options, dict):
        raise ValueError("caller replay 缺少原有選擇設定")
    mask = int(_MASK.fullmatch(path.parents[1].name)[1])
    candidates = frozen.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("FROZEN 收據缺少已完成候選結果")
    matches = [item for item in candidates if isinstance(item, dict)
               and item.get("admission_mask_key") == mask
               and item.get("threshold") == options.get("positive_probability_threshold")]
    if (metadata.get("output_scope") != "caller_override"
            or metadata.get("output_dir") != path.parent.relative_to(root).as_posix()
            or metadata.get("comparison_mode") != "score-ranking"
            or metadata.get("comparison_period") != {"start": identity.get("start"), "end": identity.get("end")}
            or not identity.get("start") or not identity.get("end")
            or metadata.get("params_source_file_sha256") != identity.get("params_sha256")
            or not isinstance(identity.get("params_sha256"), str)
            or not _HEX.fullmatch(identity["params_sha256"])
            or not isinstance(payload.get("no_filter"), dict) or not payload["no_filter"]
            or not isinstance(payload.get("score_ranking"), dict) or not payload["score_ranking"]
            or not isinstance(payload.get("yearly"), list) or not payload["yearly"]
            or len(matches) != 1 or matches[0].get("metrics") != payload["score_ranking"]
            or matches[0].get("yearly") != payload["yearly"]):
        raise ValueError("caller replay 與 FROZEN 候選／來源／期別證據不一致")
    if frozen.get("selected_threshold") not in [row.get("threshold") for row in candidates if isinstance(row, dict)]:
        raise ValueError("FROZEN 收據沒有對應的已選結果")
    return payload


def _storage_digest(path: Path, revisions: dict) -> str | None:
    receipt = path.parent / STORAGE_NAME
    if not receipt.exists():
        return None
    root = root_for(path)
    payload = _read(receipt, revisions)
    digest = payload.get("sha256")
    if (payload.get("schema") != _SCHEMA or payload.get("filename") != BASELINE_NAME
            or not isinstance(digest, str) or not _HEX.fullmatch(digest)
            or payload.get("pair_sha256") != compute_observed_file_sha256(path.parent / "strategy_comparison.json")):
        raise ValueError("caller baseline 儲存收據與原發布內容不同")
    source = root / relative_path(payload.get("retained_source"))
    if source == path or root_for(source) != root or source.parents[2] != path.parents[2]:
        raise ValueError("caller baseline 儲存收據的來源範圍無效")
    return digest


def publication_evidence(path: Path) -> tuple[dict, str | None]:
    path = Path(path).absolute()
    revisions = {}
    _publication(path, revisions)
    digest = _storage_digest(path, revisions)
    return revisions, digest


def recorded_source(path: Path, *, revisions=None) -> Path | None:
    path = Path(path).absolute()
    observed = {} if revisions is None else revisions
    payload = _publication(path, observed)
    metadata = payload["metadata"]
    if metadata.get("baseline_reused") is not True:
        return None
    root = root_for(path)
    source = root / relative_path(metadata.get("baseline_reuse_source")) / BASELINE_NAME
    if root_for(source) != root or source.parents[2] != path.parents[2] or source == path:
        raise ValueError("caller baseline REUSE 來源離開已完成的同一選擇工作")
    # After cleanup a missing filename must carry its own byte pin. A source
    # path in metadata alone cannot silently substitute later changed bytes.
    if not path.exists() and _storage_digest(path, observed) is None:
        raise ValueError("缺少 caller baseline 原有內容的儲存收據")
    return source


def expected_sha256(path: Path) -> str:
    revisions, digest = publication_evidence(path)
    if digest is None:
        raise ValueError("缺少 caller baseline 原有內容的儲存收據")
    return digest


def record_alias(path: Path, source: Path, digest: str) -> None:
    """Pin verified original bytes before unlink; caller holds storage_lock."""
    import json
    path, source = Path(path).absolute(), Path(source).absolute()
    root = root_for(path)
    if (root is None or root_for(source) != root or source.parents[2] != path.parents[2]
            or source == path or not isinstance(digest, str) or not _HEX.fullmatch(digest)):
        raise ValueError("caller baseline 儲存來源或 SHA 無效")
    observed, prior = publication_evidence(path)
    source_evidence, source_digest = publication_evidence(source)
    observed.update(source_evidence)
    if source_digest is not None and source_digest != digest:
        raise ValueError("caller baseline 保留來源偏離自身的儲存收據")
    for file in (path, source):
        if not clean_path(file) or not plain(file.lstat()):
            raise ValueError("caller baseline 儲存來源不是一般實體檔案")
        before = file_change_signature(file)
        if compute_observed_file_sha256(file) != digest:
            raise ValueError("caller baseline 副本與保留來源並非相同 bytes")
        observed[file] = before
    if prior is not None and prior != digest:
        raise ValueError("caller baseline 內容偏離既有儲存收據")
    if any(not clean_path(file) or file_change_signature(file) != stamp for file, stamp in observed.items()):
        raise ValueError("caller baseline 收據或內容在儲存驗證後變動")
    receipt = path.parent / STORAGE_NAME
    if not clean_path(receipt) or (receipt.exists() and (not plain(receipt.lstat()) or receipt.stat().st_nlink != 1)):
        raise ValueError("caller baseline 儲存收據路徑不安全")
    payload = {"schema": _SCHEMA, "filename": BASELINE_NAME, "sha256": digest,
               "retained_source": source.relative_to(root).as_posix(),
               "pair_sha256": compute_observed_file_sha256(path.parent / "strategy_comparison.json")}
    atomic_write_text(receipt, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

"""Recorded Strategy Compare detail provenance, independent of byte aliases."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path, PureWindowsPath

from core.file_integrity import file_change_signature
from core.storage_paths import clean_path, plain

DETAIL_NAMES = frozenset(
    f"{side}_{detail}.csv" for side in ("no_filter", "score_ranking")
    for detail in ("orderable_candidates", "orderable_target_diagnostics")
)
_READ_SCOPE = ContextVar("compare_detail_lineage_read_scope", default=None)


@contextmanager
def detail_lineage_read_scope():
    """Share parsed provenance within a read-only plan, then verify every marker.

    Long REUSE chains are shared by many files and comparison arms. This scope
    retains source strings and path facts, never artifact-readiness decisions.
    No observation survives the operation; changed or redirected markers reject
    the entire plan before its caller receives it.
    """
    if _READ_SCOPE.get() is not None:
        yield
        return
    observed, paths = {}, {}
    token = _READ_SCOPE.set((observed, paths))
    try:
        yield
    finally:
        _READ_SCOPE.reset(token)
    for marker, (revision, _sources) in observed.items():
        if not clean_path(marker) or file_change_signature(marker) != revision:
            raise ValueError("REUSE 來源紀錄在檢查期間變動；請重新載入")
    for path, (state, revision) in paths.items():
        if _source_path_state(path) != state or file_change_signature(path) != revision:
            raise ValueError("REUSE 明細路徑在檢查期間變動；請重新載入")


def _read_recorded_sources(marker):
    scope = _READ_SCOPE.get()
    observed = scope[0] if scope is not None else None
    if observed is not None and marker in observed:
        return observed[marker]
    if not clean_path(marker) or not marker.is_file():
        return None
    before = file_change_signature(marker)
    payload = json.loads(marker.read_text(encoding="utf-8-sig"))
    if file_change_signature(marker) != before:
        raise ValueError("讀取 REUSE 來源紀錄時內容變動")
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    result = (before, {name: _recorded_detail_source(metadata, name) for name in DETAIL_NAMES})
    if observed is not None:
        observed[marker] = result
    return result


def _source_path_state(path):
    scope = _READ_SCOPE.get()
    if scope is not None and path in scope[1]:
        return scope[1][path][0]
    if not clean_path(path):
        return False, False, False
    exists = path.is_file()
    state = (True, exists, plain(path.lstat()) if exists else False)
    if scope is not None:
        scope[1][path] = (state, file_change_signature(path))
    return state


def compare_root(path):
    """Recognize owned comparison outputs with a verifiable storage contract."""
    p = Path(path).absolute()
    if len(p.parents) >= 7 and p.parents[1].name == "pairs" and p.parents[3].name == "runs":
        # Current and historical profile roots have different depths. The run
        # and pair publication boundaries, not a fixed depth, establish ownership.
        for parent in p.parents[4:]:
            if parent.name == "strategy_compare" and parent.parent.name == "outputs":
                return parent.parent.parent
    from core.caller_baseline_storage import root_for
    return root_for(p)


def _recorded_detail_source(metadata, filename):
    if not isinstance(metadata, dict):
        return None
    source = metadata.get("cache_reused_from")
    if source is not None:
        return source if isinstance(source, str) and source.strip() else None
    # The baseline producer reuses the candidate rows. Target diagnostics are
    # joined again against the current model and have no baseline-file lineage.
    if filename == "no_filter_orderable_candidates.csv" and metadata.get("baseline_reused") is True:
        source = metadata.get("baseline_reuse_source")
        if isinstance(source, str) and source.strip():
            return source
    return None


def reused_detail_source(path, *, revisions=None):
    """Follow the producer's existing REUSE provenance; create no new storage format."""
    path = Path(path).absolute()
    root = compare_root(path)
    if root is None or path.name not in DETAIL_NAMES:
        return None
    from core import caller_baseline_storage as caller
    seen = {path}
    while True:
        if caller.root_for(path) is not None:
            source = caller.recorded_source(path, revisions=revisions)
            if source is None:
                return None
            path = source.absolute()
        else:
            marker = path.parent / "strategy_comparison.json"
            recorded = _read_recorded_sources(marker)
            if recorded is None:
                return None
            before, sources = recorded
            if revisions is not None:
                revisions[marker] = before
            source = sources[path.name]
            if source is None:
                return None
            parent = Path(source)
            if parent.is_absolute() or PureWindowsPath(source).anchor:
                raise ValueError("舊 REUSE 紀錄使用絕對來源路徑；保留副本，避免專案搬移後失去來源")
            path = (root / parent / path.name).absolute()
        if path in seen or compare_root(path) != root:
            raise ValueError("REUSE 明細來源循環或離開正式專案路徑")
        clean, exists, regular = _source_path_state(path)
        if not clean:
            raise ValueError("REUSE 明細來源循環或離開正式專案路徑")
        seen.add(path)
        if exists:
            if not regular:
                raise ValueError("REUSE 明細來源不是一般檔案")
            return path

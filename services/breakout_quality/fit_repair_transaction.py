"""Fail-closed, profile-scoped publication transaction for explicit FIT repairs.

A repair delegates training/evaluation to the canonical producers. It never infers
missing historical FIT provenance from matching context or checkpoint bytes.
Existing model/evaluation publications are moved aside on the same volume before
any build and retained as historical evidence on successful validation. No
research identity or FitKey is changed by this engineering transaction.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Callable, Iterator

from core.file_integrity import exclusive_file_lock, sibling_publication_lock_path


def selection_pit_build_lock_path(point_in_time_dir: Path) -> Path:
    """Stable per-PIT-root lock even while FIT Repair renames its parent profile.

    The profile's architecture-level parent is never one of the standard FIT
    Repair publication roots. Do not put this lock inside the movable PIT tree.
    """
    pit = Path(point_in_time_dir).resolve()
    return pit.parent.parent / f".{pit.parent.name}.{pit.name}.selection_pit_build.lock"


@contextmanager
def _fit_repair_lock(journal_path: Path) -> Iterator[None]:
    """Per-journal OS lock; released by the kernel even after process termination.

    The lock file itself remains on disk to avoid races caused by deleting an
    inode while a second process is waiting on it. No extra user CLI is needed.
    """
    with exclusive_file_lock(
        journal_path.with_suffix(".lock"),
        conflict_message="Another FIT repair is still active for this profile",
    ):
        yield


def fit_repair_journal_path(*, project_root: Path, filter_id: str, profile: str) -> Path:
    """Single owner for the legacy FIT repair transaction namespace."""
    if not str(filter_id).strip() or not str(profile).strip():
        raise ValueError("FIT repair identity must not be empty")
    if any(char in str(profile) for char in ("/", "\\", "..")):
        raise ValueError("FIT repair profile is not a path component")
    return (Path(project_root) / "outputs" / "filters" / "breakout_quality"
            / str(filter_id) / "audit" / "fit_repair"
            / (str(profile) + ".in_progress.json"))


def _under_root(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _safe_roots(project_root: Path, publication_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    root = Path(project_root).resolve(strict=True)
    paths: list[Path] = []
    for raw in publication_roots:
        path = Path(raw).absolute()
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise RuntimeError(f"FIT repair publication root is not a plain directory: {path}")
        canonical = path.resolve()
        if canonical == root or not _under_root(root, canonical):
            raise RuntimeError(f"FIT repair publication is outside project root: {canonical}")
        if canonical not in paths:
            paths.append(canonical)
    # A model's Rolling publication may be nested under the Forward model dir.
    return tuple(path for path in paths if not any(
        other != path and other in path.parents for other in paths
    ))


def archived_rolling_fit_fold_candidates(
    *, point_in_time_dir: Path, fold_id: str, project_root: Path,
) -> tuple[Path, ...]:
    """Locate old Rolling folds displaced by a committed FIT repair.

    Repairs can archive an entire model directory containing ``point_in_time``.
    Search only the exact former ancestors of this PIT directory, within the
    same project and ``models``/``outputs`` domain. An archive is a candidate,
    never a REUSE decision; the PIT FIT owner must validate all source bytes,
    training identity and historical context binding before materializing it.
    """
    root = Path(project_root).resolve(strict=True)
    pit = Path(point_in_time_dir).absolute()
    name = str(fold_id).strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Archived FIT fold ID must be a single safe path component")
    if pit.is_symlink() or not _under_root(root, pit.resolve()):
        raise RuntimeError("Archived FIT lookup requires a non-symlink project PIT directory")
    relative = pit.relative_to(root)
    if not relative.parts or relative.parts[0] not in {"models", "outputs"}:
        return ()
    domain_root = root / relative.parts[0]
    candidates: list[Path] = []
    ancestor = pit
    while ancestor != domain_root:
        if ancestor.is_symlink():
            raise RuntimeError("Archived FIT lookup refuses symlink ancestor")
        archive_base = ancestor.parent / "historical_fit_repairs" / ancestor.name
        if archive_base.is_symlink():
            raise RuntimeError("Archived FIT lookup refuses symlink archive root")
        if archive_base.is_dir():
            suffix = pit.relative_to(ancestor)
            for stamp in sorted(archive_base.iterdir(), reverse=True):
                if stamp.is_symlink() or not stamp.is_dir():
                    raise RuntimeError("Archived FIT lookup refuses unsafe repair archive")
                original = stamp / "original"
                candidate = original / suffix / "folds" / name
                if any(part.is_symlink() for part in (original, candidate.parent, candidate)):
                    raise RuntimeError("Archived FIT lookup refuses symlink source fold")
                if ((candidate / "manifest.json").is_file()
                    or (candidate / "model.pt").is_file()):
                    if ((candidate / "manifest.json").is_symlink()
                        or (candidate / "model.pt").is_symlink()):
                        raise RuntimeError("Archived FIT lookup refuses symlink FIT artifacts")
                    candidates.append(candidate)
        ancestor = ancestor.parent
    return tuple(dict.fromkeys(candidates))


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _drop_new_publication(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Refusing to remove symlink during FIT repair rollback: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        raise RuntimeError(f"FIT repair rollback encountered non-directory: {path}")


def _restore(sources: list[dict]) -> None:
    """Restore original directories from staging or partially archived locations."""
    for row in reversed(sources):
        source, hold, archive = (Path(row[key]) for key in ("source", "hold", "archive"))
        if not row["existed"]:
            _drop_new_publication(source)
            continue
        existing_backup = hold if hold.exists() else archive
        if not existing_backup.is_dir():
            # An exception while moving earlier roots can leave later roots
            # untouched. Do not delete that untouched historical publication.
            if source.is_dir():
                continue
            raise RuntimeError(
                f"Original FIT repair publication needs manual recovery: {source}; "
                f"hold={hold}; archive={archive}"
            )
        _drop_new_publication(source)
        existing_backup.rename(source)


def _recover_incomplete_fit_repair_unlocked(*, journal_path: Path, project_root: Path | None = None) -> str:
    """Restore only a verified, uncommitted journal; do not infer FIT identity."""
    journal_path = Path(journal_path).resolve(strict=True)
    if journal_path.stat().st_size == 0:
        # The initial O_EXCL marker precedes every publication move; an empty
        # journal can only be created before staging started.
        journal_path.unlink()
        return "EMPTY_JOURNAL_CLEARED"
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    if (payload.get("schema") != "fit_repair_transaction_v1"
        or payload.get("state") not in {"in_progress", "committed"}
        or not payload.get("sources")):
        raise RuntimeError("FIT repair journal schema/state invalid; manual inspection required")
    # Refuse journal path injection: only same-parent unpublished holding dirs
    # and managed history directories may be restored. Resolve before mutation.
    for row in payload["sources"]:
        source = Path(row["source"])
        hold = Path(row["hold"])
        archive = Path(row["archive"])
        if (source.is_symlink() or hold.is_symlink() or archive.is_symlink()):
            raise RuntimeError("FIT repair recovery refuses symlink publication paths")
        if (not source.is_absolute() or not hold.is_absolute() or not archive.is_absolute()
            or source.parent != hold.parent
            or not hold.name.startswith("." + source.name + ".fit_repair_pending_")
            or archive.parent.parent.parent != source.parent / "historical_fit_repairs"
            or archive.parent.parent.name != source.name or archive.name != "original"):
            raise RuntimeError("FIT repair journal contains unsupported path; manual inspection required")
    if project_root is not None:
        project_root = Path(project_root).resolve(strict=True)
        if not _under_root(project_root, journal_path):
            raise RuntimeError("FIT recovery journal is outside project root")
        _safe_roots(project_root, tuple(Path(row["source"]) for row in payload["sources"]))
    if payload["state"] == "committed":
        # A crash after atomic commit but before journal cleanup is not a
        # failed repair. Never roll the new model back after a recorded commit.
        # Require all declared old publications to be archived first.
        if any(row["existed"] and (
            not Path(row["archive"]).is_dir() or Path(row["hold"]).exists()
        ) for row in payload["sources"]):
            raise RuntimeError("Committed FIT repair has incomplete historical archives")
        journal_path.unlink()
        return "COMMITTED_CLEANED"
    _restore(list(payload["sources"]))
    journal_path.unlink()
    return "ROLLED_BACK"


def recover_incomplete_fit_repair(
    *, journal_path: Path, project_root: Path | None = None,
    point_in_time_dir: Path | None = None, shared_cache_root: Path | None = None,
) -> str:
    """Auto-safe restart: recovery cannot overlap a PIT writer or FIT cache writer."""
    journal_path = Path(journal_path).absolute()
    with _fit_repair_lock(journal_path):
        if not journal_path.is_file():
            return "NO_JOURNAL"
        with ExitStack() as locks:
            if point_in_time_dir is not None:
                locks.enter_context(exclusive_file_lock(
                    selection_pit_build_lock_path(point_in_time_dir),
                    conflict_message="Another Selection PIT build or FIT repair is active",
                ))
            if shared_cache_root is not None and journal_path.stat().st_size:
                # Journal source paths are not trusted merely because they are
                # named here. The existing strict recovery validator runs before
                # any mutation; this lookup ONLY selects a coordination lock.
                payload = json.loads(journal_path.read_text(encoding="utf-8"))
                cache_parent = Path(shared_cache_root).resolve()
                sources = payload.get("sources") if isinstance(payload, dict) else None
                if not isinstance(sources, list):
                    raise RuntimeError("FIT recovery journal sources invalid")
                entries = [Path(row["source"]).absolute() for row in sources
                           if isinstance(row, dict) and isinstance(row.get("source"), str)
                           and Path(row["source"]).absolute().parent == cache_parent]
                if len(entries) > 1:
                    raise RuntimeError("FIT recovery journal has multiple shared FIT cache entries")
                if entries:
                    locks.enter_context(exclusive_file_lock(
                        sibling_publication_lock_path(entries[0]),
                        conflict_message="Another shared FIT checkpoint publication is active",
                    ))
            return _recover_incomplete_fit_repair_unlocked(
                journal_path=journal_path, project_root=project_root,
            )


@contextmanager
def preserved_fit_publications(
    *, project_root: Path, publication_roots: tuple[Path, ...],
    journal_path: Path, repair_identity: str,
    point_in_time_dir: Path | None = None,
    shared_cache_entry: Path | None = None,
    locked_preflight: Callable[[], None] | None = None,
) -> Iterator[tuple[Path, ...]]:
    """Guard the repair and overlapping PIT/FitKey mutations with stable OS locks."""
    with _fit_repair_lock(Path(journal_path).absolute()):
        with ExitStack() as locks:
            if point_in_time_dir is not None:
                locks.enter_context(exclusive_file_lock(
                    selection_pit_build_lock_path(point_in_time_dir),
                    conflict_message="Another Selection PIT build or FIT repair is active",
                ))
            if shared_cache_entry is not None:
                locks.enter_context(exclusive_file_lock(
                    sibling_publication_lock_path(shared_cache_entry),
                    conflict_message="Another shared FIT checkpoint publication is active",
                ))
            if locked_preflight is not None:
                locked_preflight()
            with _preserved_fit_publications_unlocked(
                project_root=project_root, publication_roots=publication_roots,
                journal_path=journal_path, repair_identity=repair_identity,
            ) as paths:
                yield paths


@contextmanager
def _preserved_fit_publications_unlocked(
    *, project_root: Path, publication_roots: tuple[Path, ...],
    journal_path: Path, repair_identity: str,
) -> Iterator[tuple[Path, ...]]:
    """Move old publications aside, then retain them in history or roll back.

    Successful commit requires caller's canonical FIT and report checks before
    this context exits. Shared input, context and checkpoint caches are excluded.
    """
    paths = _safe_roots(project_root, publication_roots)
    if not paths:
        raise ValueError("No publication roots supplied for FIT repair")
    journal_path = Path(journal_path).absolute()
    project_root = Path(project_root).resolve(strict=True)
    if not _under_root(project_root, journal_path.resolve()) or journal_path.suffix != ".json":
        raise ValueError("FIT repair journal must be a JSON path inside project root")
    if not str(repair_identity).strip():
        raise ValueError("FIT repair identity must not be empty")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    sources: list[dict] = []
    for path in paths:
        hold = path.with_name(f".{path.name}.fit_repair_pending_{stamp}")
        archive = path.parent / "historical_fit_repairs" / path.name / stamp / "original"
        if hold.exists() or archive.exists():
            raise RuntimeError("FIT repair staging/archive path collision; refusing to overwrite")
        sources.append({
            "source": str(path), "hold": str(hold), "archive": str(archive),
            "existed": path.exists(),
        })
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL prevents concurrent repairs of the same logical profile.
    fd = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    _atomic_journal_written = False
    try:
        _atomic_json(journal_path, {
            "schema": "fit_repair_transaction_v1", "state": "in_progress",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "repair_identity": str(repair_identity), "sources": sources,
        })
        _atomic_journal_written = True
        for row in sources:
            if row["existed"]:
                Path(row["source"]).rename(Path(row["hold"]))
        yield tuple(paths)
        # Caller's validations have passed. Archive the historical truth and
        # leave the newly built publication at the canonical paths.
        for row in sources:
            if not row["existed"]:
                continue
            archive = Path(row["archive"])
            archive.parent.mkdir(parents=True, exist_ok=False)
            Path(row["hold"]).rename(archive)
        _atomic_json(journal_path, {
            "schema": "fit_repair_transaction_v1", "state": "committed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "repair_identity": str(repair_identity), "sources": sources,
        })
    except BaseException:
        _restore(sources)
        raise
    finally:
        # If rollback failed, retain journal for an explicit recover call.
        if _atomic_journal_written and journal_path.exists():
            try:
                payload = json.loads(journal_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                payload = {}
            if payload.get("state") == "committed":
                journal_path.unlink()
            elif all(not Path(row["hold"]).exists() for row in sources):
                # An error that already rolled back has no unresolved staging.
                # Missing original sources remain visible for manual audit if
                # the rollback raised before restoring every archive.
                if all(not Path(row["archive"]).exists() for row in sources):
                    journal_path.unlink()
        elif not _atomic_journal_written and journal_path.exists():
            journal_path.unlink()

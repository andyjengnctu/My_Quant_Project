"""Fail-closed, profile-scoped publication transaction for explicit FIT repairs.

A repair delegates training/evaluation to the canonical producers. It never infers
missing historical FIT provenance from matching context or checkpoint bytes.
Existing model/evaluation publications are moved aside on the same volume before
any build and retained as historical evidence on successful validation. No
research identity or FitKey is changed by this engineering transaction.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Iterator


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


def recover_incomplete_fit_repair(*, journal_path: Path) -> None:
    """Explicit recovery after an interrupted Python/process/OS session.

    Never auto-recover on a new run: a human must choose recovery instead of
    silently discarding newly produced but not yet committed evidence.
    """
    journal_path = Path(journal_path).resolve(strict=True)
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
        if (not source.is_absolute() or not hold.is_absolute() or not archive.is_absolute()
            or source.parent != hold.parent
            or not hold.name.startswith("." + source.name + ".fit_repair_pending_")
            or archive.parent.parent.parent != source.parent / "historical_fit_repairs"
            or archive.parent.parent.name != source.name or archive.name != "original"):
            raise RuntimeError("FIT repair journal contains unsupported path; manual inspection required")
    if payload["state"] == "committed":
        # A crash after atomic commit but before journal cleanup is not a
        # failed repair. Never roll the new model back after a recorded commit.
        # Require all declared old publications to be archived first.
        if any(row["existed"] and (
            not Path(row["archive"]).is_dir() or Path(row["hold"]).exists()
        ) for row in payload["sources"]):
            raise RuntimeError("Committed FIT repair has incomplete historical archives")
        journal_path.unlink()
        return
    _restore(list(payload["sources"]))
    journal_path.unlink()


@contextmanager
def preserved_fit_publications(
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

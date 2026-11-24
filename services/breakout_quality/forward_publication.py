"""Forward producer transactions: independent FIT and evaluation commits.

Training writes private, uniquely named directories with unchanged basenames.
Only validated bytes enter the published namespace. A durable, bounded journal
finishes interrupted multi-file commits; readers continue to check manifest SHA.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from copy import copy
from functools import wraps
import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import get_ident
from uuid import uuid4

from core.file_integrity import (
    atomic_replace_with_retry, atomic_write_json, compute_file_sha256,
    exclusive_file_lock,
)

from services.breakout_quality.ranker_output_paths import resolve_ranker_training_output_paths

_ACTIVE_PUBLICATION = ContextVar("forward_training_publication", default=None)
_HELD_LEASES = ContextVar("forward_publication_leases", default=frozenset())
_ACTIVE_NAMESPACES = ContextVar("forward_active_namespaces", default=frozenset())


@contextmanager
def forward_publication_leases(model_dir, output_dir):
    """Allow an explicit repair to delegate to its in-process trainer safely."""
    held = _HELD_LEASES.get()
    keys = tuple((get_ident(), str(Path(root).resolve()), phase)
                 for root, phase in ((model_dir, "fit"), (output_dir, "evaluation")))
    with ExitStack() as stack:
        for key in keys:
            if key not in held:
                root, phase = Path(key[1]), key[2]
                stack.enter_context(exclusive_file_lock(
                    root.with_name(f".{root.name}.forward-{phase}.lock"),
                    conflict_message=f"Forward {phase} producer already active",
                ))
        token = _HELD_LEASES.set(held.union(keys))
        try:
            yield
        finally:
            _HELD_LEASES.reset(token)


class ForwardPublication:
    """One producer's leased namespaces and recoverable publication commits."""

    def __init__(self, model_dir: Path, output_dir: Path):
        self.model_dir = Path(model_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        if self.model_dir == self.output_dir or (
            self.model_dir in self.output_dir.parents or self.output_dir in self.model_dir.parents
        ):
            raise ValueError("Forward model and evaluation namespaces must be disjoint")
        self.fit_committed = False
        self.stack = ExitStack()

    def _journal(self, phase):
        return self.model_dir.with_name(f".{self.model_dir.name}.forward-{phase}.json")

    def __enter__(self):
        try:
            active = _ACTIVE_NAMESPACES.get()
            keys = {(get_ident(), str(self.model_dir)), (get_ident(), str(self.output_dir))}
            if active.intersection(keys):
                raise RuntimeError("Forward producer already active in this execution context")
            token = _ACTIVE_NAMESPACES.set(active.union(keys))
            self.stack.callback(_ACTIVE_NAMESPACES.reset, token)
            self.stack.enter_context(forward_publication_leases(self.model_dir, self.output_dir))
            self.recover("fit")
            self.recover("evaluation")
            # Finish the previous bounded force-rebuild recovery protocol before
            # entering the new producer transaction. Never guess between two roots.
            for root in (self.model_dir, self.output_dir):
                backup = root.with_name(f".{root.name}.superseded_fit_backup")
                if backup.exists():
                    if backup.is_symlink() or not backup.is_dir() or root.exists():
                        raise ValueError("Forward superseded FIT recovery is ambiguous")
                    atomic_replace_with_retry(backup, root)
            self.model_stage = Path(self.stack.enter_context(tempfile.TemporaryDirectory(
                prefix=f".{self.model_dir.name}.forward-stage-", dir=self.model_dir.parent)))
            self.output_stage = Path(self.stack.enter_context(tempfile.TemporaryDirectory(
                prefix=f".{self.output_dir.name}.forward-stage-", dir=self.output_dir.parent)))
            for source, stage in ((self.model_dir, self.model_stage), (self.output_dir, self.output_stage)):
                if source.is_symlink():
                    raise ValueError("Forward publication namespace cannot be a symlink")
                if source.is_dir():
                    for path in source.iterdir():
                        if path.is_file() and not path.name.startswith("."):
                            if path.is_symlink():
                                raise ValueError("Forward publication artifact cannot be a symlink")
                            shutil.copy2(path, stage / path.name)
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def _checked_entry(self, entry, phase):
        target = Path(entry["target"]).resolve()
        pending = Path(entry["pending"]).resolve()
        if not any(target.is_relative_to(root) for root in (self.model_dir, self.output_dir)):
            raise ValueError("Forward recovery target escaped its publication namespace")
        fit_names = {"model.pt", "fitted_model_manifest.json"}
        fit_target = (target.parent == self.model_dir and target.name in fit_names) or (
            target.is_relative_to(self.model_dir / "historical_forward_fits"))
        evaluation_target = (target.parent == self.output_dir) or (
            target.parent == self.model_dir and target.name not in fit_names)
        if (phase == "fit" and not fit_target) or (phase == "evaluation" and not evaluation_target):
            raise ValueError("Forward recovery target does not belong to its phase")
        if pending.parent != target.parent or not pending.name.startswith(".forward-commit-"):
            raise ValueError("Forward recovery staging path is invalid")
        return target, pending

    def _validated_commit_entries(self, phase):
        journal = self._journal(phase)
        if not journal.exists():
            return None
        payload = json.loads(journal.read_text(encoding="utf-8"))
        if (payload.get("version") != 1 or payload.get("phase") != phase
                or payload.get("state") != "VALIDATED" or not isinstance(payload.get("entries"), list)):
            raise ValueError("Forward recovery journal contract mismatch")
        entries = list(payload["entries"])
        targets, staging = set(), set()
        # Validate the complete recovery set before replacing another file.
        for entry in entries:
            target, pending = self._checked_entry(entry, phase)
            if target in targets or pending in staging:
                raise ValueError("Forward recovery journal contains duplicate entries")
            targets.add(target)
            staging.add(pending)
            source = pending if pending.is_file() else target
            if not source.is_file() or compute_file_sha256(source) != entry["sha256"]:
                raise ValueError("Forward recovery bytes do not match validated publication")
        return entries

    def recover(self, phase):
        entries = self._validated_commit_entries(phase)
        if entries is None:
            return
        for entry in entries:
            target, pending = self._checked_entry(entry, phase)
            if pending.exists():
                atomic_replace_with_retry(pending, target)
        self._journal(phase).unlink()

    def discard_pending_commit(self, phase):
        """Called only before an enclosing explicit repair restores old roots."""
        entries = self._validated_commit_entries(phase)
        if entries is None:
            return
        for entry in entries:
            _target, pending = self._checked_entry(entry, phase)
            pending.unlink(missing_ok=True)
        self._journal(phase).unlink()

    def _commit(self, phase, pairs):
        entries = []
        try:
            for source, target in pairs:
                source, target = Path(source), Path(target)
                digest = compute_file_sha256(source)
                if target.is_file() and compute_file_sha256(target) == digest:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_name(f".forward-commit-{uuid4().hex}-{target.name}")
                entries.append({"target": str(target), "pending": str(pending), "sha256": digest})
                shutil.copyfile(source, pending)
                with pending.open("r+b") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
                if compute_file_sha256(pending) != digest:
                    raise ValueError("Forward artifact changed during staging")
            atomic_write_json(self._journal(phase), {
                "version": 1, "phase": phase, "state": "VALIDATED", "entries": entries,
            })
        except BaseException:
            # Once a journal exists it owns recovery and its pending files.
            if not self._journal(phase).exists():
                for entry in entries:
                    Path(entry["pending"]).unlink(missing_ok=True)
            raise
        self.recover(phase)

    def commit_fit(self, *, checkpoint_validator):
        """Publish checkpoint files admitted by the trainer's canonical FIT owner."""
        admitted = checkpoint_validator(self.model_stage / "model.pt")
        names = list(admitted or ("model.pt",))
        if not names or names[0] != "model.pt" or any(
            name not in {"model.pt", "fitted_model_manifest.json"} for name in names
        ) or len(set(names)) != len(names):
            raise ValueError("Forward FIT validator returned invalid checkpoint files")
        pairs = [(self.model_stage / name, self.model_dir / name) for name in names]
        # Preserve the existing Rolling-import historical evidence, if created.
        archive = self.model_stage / "historical_forward_fits"
        for source in sorted(archive.rglob("*")) if archive.exists() else ():
            if source.is_file():
                target = self.model_dir / source.relative_to(self.model_stage)
                if target.exists() and compute_file_sha256(target) != compute_file_sha256(source):
                    raise ValueError("Forward historical FIT publication conflict")
                pairs.insert(0, (source, target))
        self._commit("fit", pairs)
        self.fit_records = tuple((name, compute_file_sha256(self.model_dir / name)) for name in names)
        self.fit_committed = True

    def commit_evaluation(self, validator):
        if not self.fit_committed:
            raise ValueError("Forward evaluation cannot precede validated FIT publication")
        def verify_fit():
            if any(compute_file_sha256(root / name) != digest
                   for name, digest in self.fit_records for root in (self.model_dir, self.model_stage)):
                raise ValueError("Forward FIT changed before evaluation publication")
        verify_fit()
        validator(self.model_stage, self.output_stage)
        verify_fit()
        pairs = []
        for stage, root in ((self.output_stage, self.output_dir), (self.model_stage, self.model_dir)):
            for path in sorted(stage.iterdir()):
                if path.is_file() and path.name not in {"model.pt", "fitted_model_manifest.json", "manifest.json"}:
                    pairs.append((path, root / path.name))
        # The canonical model manifest is the final publication fence.
        pairs.append((self.model_stage / "manifest.json", self.model_dir / "manifest.json"))
        self._commit("evaluation", pairs)


def commit_forward_fit(**kwargs):
    publication = _ACTIVE_PUBLICATION.get()
    if publication is None:
        raise RuntimeError("Forward FIT write requires the producer publication scope")
    publication.commit_fit(**kwargs)


def recover_forward_publication(model_dir, output_dir):
    """Finish prior validated commits before repair archives their namespace."""
    with forward_publication_leases(model_dir, output_dir):
        publication = ForwardPublication(model_dir, output_dir)
        publication.recover("fit")
        publication.recover("evaluation")


def discard_repair_forward_publication(model_dir, output_dir):
    """Cancel this repair's incomplete commits before its old roots return.

    The repair must first recover earlier Forward transactions under the same
    leases. Thus these bounded journals belong exclusively to the current repair.
    """
    with forward_publication_leases(model_dir, output_dir):
        publication = ForwardPublication(model_dir, output_dir)
        publication.discard_pending_commit("evaluation")
        publication.discard_pending_commit("fit")


def forward_training_publication(function):
    """Cover direct trainer, CLI, BUILD, REFRESH and force entry points equally."""
    @wraps(function)
    def run(args):
        from filters.breakout_quality.ranking_score_store import (
            clear_continuous_ranker_oos_score_caches, load_continuous_ranker_oos_contract_at_paths,
        )
        root = function.__globals__["PROJECT_ROOT"]
        paths, output = resolve_ranker_training_output_paths(args, project_root=root)
        with ForwardPublication(paths.model_dir, output) as publication:
            staged_args = copy(args)
            staged_args.model_output_dir = str(publication.model_stage)
            staged_args.research_output_dir = str(publication.output_stage)
            token = _ACTIVE_PUBLICATION.set(publication)
            try:
                result = function(staged_args)
                if int(result) != 0:
                    return result
                def validate(model_dir, output_dir):
                    contract = load_continuous_ranker_oos_contract_at_paths(
                        str(root), str(args.filter_id), str(args.model_architecture),
                        str(args.experiment_profile), model_dir=model_dir,
                        research_output_dir=output_dir, require_scores=True,
                    )
                    if int(contract.seed) != int(args.seed):
                        raise ValueError("Forward evaluation seed mismatch")
                publication.commit_evaluation(validate)
                return result
            finally:
                _ACTIVE_PUBLICATION.reset(token)
                clear_continuous_ranker_oos_score_caches()
    return run


def published_forward_path(path):
    """Display the stable destination while producer writes private staging."""
    publication = _ACTIVE_PUBLICATION.get()
    path = Path(path)
    if publication is not None:
        for stage, root in ((publication.model_stage, publication.model_dir),
                            (publication.output_stage, publication.output_dir)):
            if path.is_relative_to(stage):
                return root / path.relative_to(stage)
    return path

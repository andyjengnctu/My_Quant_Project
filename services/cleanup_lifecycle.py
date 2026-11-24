"""Resolve producer-declared work lifecycles for the existing cleaner.

This module plans proofs only. The cleaner retains ownership of unlink, Git,
NTFS, per-file revalidation, progress, and the existing recovery receipt.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import re
import stat
from zipfile import ZipFile

from core.artifact_lifecycle import (
    MANIFEST_NAME, RETAINED_ROLES, lifecycle_access, read_lifecycle,
    retained_closure, process_is_absent,
)
from core.compare_csv_storage import clean_path, plain
from core.file_integrity import file_change_signature


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _reference_strings(value, owning_fields):
    """Exclude only an owner's verified field value, never an entire JSON."""
    if not owning_fields:
        yield from _strings(value)
    elif isinstance(value, str):
        if owning_fields.get(()) != value:
            yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield key
            nested = {path[1:]: expected for path, expected in owning_fields.items()
                      if path and path[0] == key}
            yield from _reference_strings(child, nested)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            nested = {path[1:]: expected for path, expected in owning_fields.items()
                      if path and path[0] == index}
            yield from _reference_strings(child, nested)


def _purpose_payload(payload):
    if isinstance(payload, dict):
        if payload.keys() & {"contract", "producer", "subject", "active_subjects",
                             "model_artifact_fingerprint", "model_fingerprint"}:
            return True
        return any(_purpose_payload(item) for item in payload.values())
    if isinstance(payload, list):
        return any(_purpose_payload(item) for item in payload)
    return False


def _purpose_record(relative, payload):
    """Unknown producer/identity declarations require a fresh owner proof."""
    return (relative.parts[0] in {"models", "config"}
            or relative.name in {MANIFEST_NAME, "manifest.json", "strategy_comparison.json"}
            or _purpose_payload(payload))


class LifecycleCleanup:
    def __init__(self, root: Path, progress=None):
        self.root = root
        self.progress = progress or (lambda message: None)
        self.files = {}
        self.directories = set()
        self.directory_revisions = {}
        self.records = {}
        self.artifacts = {}
        self.keep = set()
        self.errors = {}
        self.revisions = {}
        self.reference_revisions = {}
        self.pending_reference_paths = set()
        self.strict_reference_paths = set()
        self.deleted_paths = set()
        self.bundles = set()
        self.bundle_rule = None
        self.bundle_config = None
        self.active = set()
        self.external_refs = set()
        self.verified_records = set()
        self.bundle_directory_revision = None
        self.bundle_kept_revisions = {}
        self.deletion_evidence = {}
        self.extra_targets = set()
        self.extra_roots = set()
        self.ownership_references = {}

    def observe_directory(self, relative):
        self.directories.add(relative)
        if ((relative == Path(".") or relative.parts[0] in {"outputs", "models", "state", "config", "data"})
                and not relative.is_relative_to("outputs/project_cleanup")):
            self.directory_revisions[relative] = file_change_signature(self.root / relative)

    def observe(self, relative: Path, size: int):
        self.files[relative] = size

    def prepare(self, active_subjects):
        self.active = set(active_subjects)
        # Configuration can activate a subject by identity rather than a path.
        # Include non-JSON sources, too; reference parsing is not authority here.
        self.revisions.update((path, file_change_signature(self.root / path))
                              for path in self.files if path.parts[0] == "config"
                              and path.parent.name != "__pycache__")
        for marker in self.files:
            if marker.name != MANIFEST_NAME or marker.parts[0] != "outputs":
                continue
            try:
                record = read_lifecycle(self.root, marker)
                self.records[marker] = record
                self.revisions[marker] = record.revision
                for artifact in record.artifacts:
                    if artifact.path in self.artifacts:
                        raise ValueError("工件出現在多個生命週期紀錄中")
                    self.artifacts[artifact.path] = (record, artifact)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.errors[marker] = str(exc)
        self._prepare_bundles()
        if self.records or self.bundles or self.extra_targets:
            self.external_refs = self._external_references()
            self._refresh_closure()
            self.bundles = {path for path in self.bundles if not self.protected(path)}

    def protected(self, relative):
        return relative in self.keep or any(parent in self.keep for parent in relative.parents)

    def _reference_targets(self):
        # Broad output-root settings are destinations, not dependencies on every
        # past artifact. An explicit lifecycle retain/requires can still pin one.
        targets = set(self.bundles) | set(self.records) | self.extra_targets
        for path, (record, _) in self.artifacts.items():
            targets.add(path)
            targets.update(parent for parent in path.parents
                           if parent.is_relative_to(record.workspace))
        return targets

    def _refresh_closure(self):
        refs = set(self.external_refs) | self.extra_roots
        known = self._reference_targets()
        for value in self.active:
            if not isinstance(value, str) or not value or '\x00' in value:
                continue
            path = Path(os.path.normpath(value.replace('\\', '/')))
            if path.is_absolute():
                try:
                    path = path.relative_to(self.root)
                except ValueError:
                    continue
            if path in known:
                refs.add(path)
        self.keep = retained_closure(self.records.values(), external_roots=refs,
                                     active_subjects=self.active, existing_paths=self.files)

    def _external_references(self, paths=None, *, incremental=False):
        """Protect references from existing result/config/state JSON as well.

        References are a veto, never deletion authority. Unknown/malformed
        evidence blocks lifecycle deletions; it is not rewritten or fabricated.
        """
        ancestors = self._reference_targets()
        # JSON results contain millions of ordinary labels/metrics. A path
        # resolving to a known target must mention its basename, unless a dot
        # component (or a Windows drive-relative path) reaches it relative to this
        # JSON's directory. This filter
        # only avoids constructing Paths; malformed JSON still fails closed.
        names = {path.name for path in ancestors if path.name}
        components = "|".join(re.escape(name) for name in names)
        possible_path = re.compile(
            r"(?:^|[/\\])(?:" + components + r")(?:[/\\]|$)"
            + r"|(?:^|[/\\])\.\.?(?:[/\\]|$)|^[A-Za-z]:", re.IGNORECASE,
        )
        refs = set()
        inspected = self.files if paths is None else paths
        for index, relative in enumerate(inspected, 1):
            if relative.suffix.lower() != ".json" or relative in self.records:
                continue
            self.progress(f"核對保留依賴 {index:,}/{len(self.files):,} 個檔案")
            if relative.parts[0] not in {"outputs", "models", "state", "config", "data"}:
                continue
            if relative.is_relative_to("outputs/project_cleanup") or relative.is_relative_to("outputs/local_regression/_staging"):
                continue
            owned = self.artifacts.get(relative)
            if owned:
                # Registered producers declare the dependency edges, including
                # those of result/evidence files. Do not mistake their output
                # directory metadata for a consumer dependency on every file.
                continue
            file = self.root / relative
            try:
                revision = file_change_signature(file)
                if incremental and revision == ("missing",) and clean_path(file) and not file.is_symlink():
                    # Removing an external reference never releases its old
                    # dependencies during this cleanup plan.
                    self.reference_revisions[relative] = revision
                    continue
                if not clean_path(file) or not plain(file.lstat()):
                    raise ValueError("引用紀錄不是一般檔案")
                payload = json.loads(file.read_text(encoding="utf-8-sig"))
                if file_change_signature(file) != revision:
                    raise ValueError("引用紀錄在讀取期間變動")
                if _purpose_record(relative, payload):
                    if incremental:
                        raise ValueError("新增／變更產生者用途紀錄，必須重新盤點")
                    self.strict_reference_paths.add(relative)
                self.reference_revisions[relative] = revision
                for value in _reference_strings(payload, self.ownership_references.get(relative, {})):
                    # Not every JSON string is a path. Only strings resolving
                    # to a declared file or its ancestor can protect artifacts.
                    if not value or '\x00' in value or not possible_path.search(value):
                        continue
                    text = value.replace('\\', '/')
                    path = Path(text)
                    bases = (self.root,) if path.is_absolute() else (self.root, file.parent)
                    for base in bases:
                        absolute = Path(os.path.normpath(base / path))
                        try:
                            target = absolute.relative_to(self.root)
                        except ValueError:
                            continue
                        if target in ancestors:
                            refs.add(target)
            except (OSError, ValueError, TypeError, RecursionError) as exc:
                self.errors[relative] = "無法驗證外部引用：" + str(exc)
        return refs

    def _authority_snapshot(self, worker):
        """Keep deletion authority separate from refreshable external references."""
        revisions = dict(self.revisions)
        scopes = {Path("config"), *(record.workspace for record in self.records.values())}
        for relative, stamp in self.reference_revisions.items():
            if (relative in self.strict_reference_paths or relative in self.ownership_references
                    or any(relative == path for record in self.records.values()
                           for path, _ in record.publication)):
                revisions[relative] = stamp
        for adapter in getattr(worker, "storage_adapters", {}).values():
            scopes.update(Path(p).relative_to(self.root) if Path(p).is_absolute() else Path(p)
                          for p in getattr(adapter, "proof_scopes", ()))
            for path, stamp in getattr(adapter, "proof_revisions", {}).items():
                relative = Path(path).relative_to(self.root) if Path(path).is_absolute() else Path(path)
                if revisions.setdefault(relative, stamp) != stamp:
                    raise ValueError("用途與來源證據的版本互相衝突")
        for path in (*revisions, *scopes):
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("用途與來源證據離開專案")
        return revisions, scopes

    def _require_authority(self, revisions):
        for relative, stamp in revisions.items():
            path = self.root / relative
            current = file_change_signature(path)
            if relative in self.deleted_paths and current == ("missing",):
                continue  # Only this cleaner's successful unlinks get a tombstone.
            if not clean_path(path) or current != stamp:
                raise ValueError("用途、來源或發布證據在盤點後變動，請重新盤點")

    def _refresh_reference_inventory(self, scopes):
        """Inspect changed directories; additions can only add reference vetoes."""
        revisions = dict(self.directory_revisions)
        pending = list(revisions)
        visited, new_json, new_directories = set(), set(), set()
        new_files = {}
        while pending:
            relative = pending.pop()
            if relative in visited:
                continue
            visited.add(relative)
            directory = self.root / relative
            before = file_change_signature(directory)
            if before == revisions.get(relative):
                continue
            if any(relative.is_relative_to(scope) for scope in scopes):
                raise ValueError("產生者用途／科學工作目錄在盤點後變動，請重新盤點")
            if not clean_path(directory):
                raise ValueError("新增引用目錄經過連結")
            if before == ("missing",):
                revisions.pop(relative, None)
                continue  # Prior references stay retained even after removal.
            info = directory.lstat()
            if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise ValueError("引用目錄不是一般實體目錄")
            additions, children = {}, []
            with os.scandir(directory) as entries:
                for entry in entries:
                    child = relative / entry.name
                    if (child.parts[0] not in {"outputs", "models", "state", "config", "data"}
                            or child.is_relative_to("outputs/project_cleanup")):
                        continue
                    child_path = self.root / child
                    info = entry.stat(follow_symlinks=False)
                    if (not clean_path(child_path) or stat.S_ISLNK(info.st_mode)
                            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                        raise ValueError("新增／變更檔案含連結或重新解析點")
                    if stat.S_ISDIR(info.st_mode):
                        if child not in revisions:
                            if (child in self.files or child in new_files or any(child.is_relative_to(scope)
                                    or scope.is_relative_to(child) for scope in scopes)):
                                raise ValueError("新增產生者工作目錄或檔案型別已改變")
                            children.append(child)
                    elif plain(info):
                        if child in self.directories or child in new_directories:
                            raise ValueError("引用目錄的檔案型別已改變")
                        if child not in self.files and child not in new_files:
                            if child.name == MANIFEST_NAME:
                                raise ValueError("新增生命週期紀錄，必須重新盤點")
                            additions[child] = info.st_size
                    else:
                        raise ValueError("新增引用包含未知檔案型別")
            if file_change_signature(directory) != before:
                raise ValueError("新增檔案清單在讀取期間變動")
            revisions[relative] = before
            new_directories.add(relative)
            new_files.update(additions)
            new_json.update(path for path in additions if path.suffix.lower() == ".json")
            pending.extend(children)
        # Commit together: a failed traversal cannot hide a previously unseen
        # reference JSON from the next cleanup phase.
        self.directory_revisions = revisions
        self.directories.update(new_directories)
        self.files.update(new_files)
        self.pending_reference_paths.update(new_json)
        return new_json

    def _refresh_reference_snapshot(self, scopes):
        self._refresh_reference_inventory(scopes)
        # A later stat/parse can fail after inventory commits. Until the whole
        # refresh succeeds, new JSON must remain pending for the next phase.
        changed = set(self.pending_reference_paths)
        for relative, stamp in self.reference_revisions.items():
            if file_change_signature(self.root / relative) != stamp:
                if any(relative.is_relative_to(scope) for scope in scopes):
                    raise ValueError("用途／科學範圍內的引用紀錄在盤點後變動")
                changed.add(relative)
        previous = set(self.external_refs)
        self.external_refs.update(self._external_references(changed, incremental=True))
        self._refresh_closure()
        if self.errors:
            first = next(iter(self.errors))
            raise ValueError(f"無法重驗新增引用：{first.as_posix()}：{self.errors[first]}")
        # Reading B can race with a writer changing already-read A or adding C.
        # Require one stable inventory after the entire refresh, not just each
        # JSON's own read window. This check creates no new deletion authority.
        for relative, stamp in (*self.reference_revisions.items(), *self.directory_revisions.items()):
            path = self.root / relative
            if not clean_path(path) or file_change_signature(path) != stamp:
                raise ValueError("外部引用或目錄在整批重驗期間再次變動")
        self.pending_reference_paths.difference_update(changed)
        if self.external_refs - previous:
            raise ValueError("新增保留引用，已保護相關工件；請重新盤點")

    def _prepare_bundles(self):
        from core.output_retention import expired_paths, local_regression_bundle_rule
        config = self.root / "tools/local_regression/manifest.json"
        if not config.is_file():
            return
        try:
            if not clean_path(config) or not plain(config.lstat()):
                raise ValueError("診斷保留設定不是一般檔案")
            revision = file_change_signature(config)
            payload = json.loads(config.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict) or payload.get("retention_enabled", True) is not True:
                return
            self.bundle_rule = local_regression_bundle_rule(self.root, payload)
            directory_revision = file_change_signature(self.bundle_rule.target_dir)
            catalog = {p.relative_to(self.root): file_change_signature(p)
                       for p in self.bundle_rule.target_dir.glob("to_chatgpt_bundle_*.zip")
                       if clean_path(p) and plain(p.lstat())}
            self.bundles = {p.relative_to(self.root) for p in expired_paths(self.bundle_rule)}
            self.bundles.difference_update(self.keep)
            if (file_change_signature(config) != revision
                    or file_change_signature(self.bundle_rule.target_dir) != directory_revision
                    or any(file_change_signature(self.root / p) != stamp for p, stamp in catalog.items())):
                raise ValueError("保留設定在讀取期間變動")
            self.bundle_config = (config, revision)
            self.bundle_directory_revision = directory_revision
            self.revisions.update(catalog)
            self.bundle_kept_revisions = {p: stamp for p, stamp in catalog.items() if p not in self.bundles}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.errors[Path("tools/local_regression/manifest.json")] = str(exc)
            self.bundles.clear()

    def candidate(self, relative: Path):
        if relative in self.bundles:
            return not self.errors and not self.protected(relative)
        pair = self.artifacts.get(relative)
        if not pair:
            return False
        record, artifact = pair
        return (not self.errors and not self.protected(relative)
                and artifact.role not in RETAINED_ROLES
                and record.state not in {"RUNNING", "RESUMABLE"}
                and (artifact.role != "cache" or record.state in {"SUPERSEDED", "ABANDONED"})
                and (record.state != "SUPERSEDED" or bool(record.subject)))

    def retained_reason(self, relative: Path):
        if relative.name == MANIFEST_NAME:
            return "保留用途與生命週期紀錄"
        if relative in self.artifacts:
            record, artifact = self.artifacts[relative]
            if self.errors:
                first = next(iter(self.errors))
                return f"清理依賴證據不完整：{first.as_posix()}：{self.errors[first]}"
            if relative in self.keep:
                return "正式成果、續跑、現行使用或被保留工件直接／間接引用"
            return f"工作尚未放行此類工件：state={record.state}, role={artifact.role}"
        return None

    def _verify_publication(self, worker, record):
        if file_change_signature(self.root / record.marker) != record.revision:
            raise ValueError("工作用途／狀態在盤點後變動")
        if record.marker not in self.verified_records:
            if read_lifecycle(self.root, record.marker) != record:
                raise ValueError("工作用途／狀態在盤點後變動")
            self.verified_records.add(record.marker)
        if record.active_pid is not None and not process_is_absent(record.active_pid):
            raise ValueError("產生此工作檔的程序仍在執行或無法確認已結束")
        if record.state in {"COMPLETED", "SUPERSEDED"} and not record.publication:
            raise ValueError("缺少已完成或替代成果的發布證據")
        if record.state == "SUPERSEDED" and record.subject in self.active:
            raise ValueError("被取代工作仍由現行設定引用")
        evidence = {record.marker: record.revision}
        for path, digest in record.publication:
            file = self.root / path
            evidence[path] = file_change_signature(file)
            if not clean_path(file) or not plain(file.lstat()) or worker.hash_file(file) != digest:
                raise ValueError("正式／替代成果缺少或內容已變更")
        return evidence

    def proof(self, worker, relative: Path):
        self.deletion_evidence.pop(relative, None)
        if relative in self.bundles:
            if not self.candidate(relative):
                return ""
            config, revision = self.bundle_config
            if file_change_signature(config) != revision:
                raise ValueError("診斷保留設定已變動")
            # Rank was established once for this snapshot. Our own deletions
            # only remove expired items; an external directory change invalidates it.
            if file_change_signature(self.bundle_rule.target_dir) != self.bundle_directory_revision:
                raise ValueError("診斷封包清單在盤點後變動，請重新盤點")
            for path, stamp in {relative: self.revisions[relative], **self.bundle_kept_revisions}.items():
                if file_change_signature(self.root / path) != stamp:
                    raise ValueError("診斷封包內容或保留排序已變動")
            with ZipFile(self.root / relative) as archive:
                if "master_summary.json" not in archive.namelist():
                    raise ValueError("不是可驗證的測試診斷封包，保留")
                summary = json.loads(archive.read("master_summary.json"))
                if not isinstance(summary, dict) or not summary:
                    raise ValueError("診斷封包缺少結果紀錄")
            self.deletion_evidence[relative] = {
                relative: self.revisions[relative], **self.bundle_kept_revisions,
                config.relative_to(self.root): revision,
                self.bundle_rule.target_dir.relative_to(self.root): self.bundle_directory_revision,
            }
            return "超過設定保留範圍的已發布測試診斷封包"
        record, artifact = self.artifacts[relative]
        if not self.candidate(relative):
            return ""
        evidence = self._verify_publication(worker, record)
        if worker.hash_file(self.root / relative) != artifact.sha256:
            raise ValueError("工作檔內容與產生者的用途紀錄不同")
        if artifact.role == "replica":
            if artifact.replacement is None or artifact.replacement == relative:
                raise ValueError("副本缺少可保留的替代來源")
            replacement = self.root / artifact.replacement
            evidence.setdefault(artifact.replacement, file_change_signature(replacement))
            if (not clean_path(replacement) or not plain(replacement.lstat())
                    or worker.hash_file(replacement) != artifact.sha256):
                raise ValueError("副本與替代來源內容不一致")
        self.deletion_evidence[relative] = evidence
        return {"COMPLETED": "已完成工作且沒有保留依賴的中間檔／副本",
                "SUPERSEDED": "產生者確認已取代且沒有保留依賴的工作檔／快取",
                "ABANDONED": "產生者確認已結案且不需續跑的工作殘留"}[record.state]

    def verify_before_delete(self, worker, relative: Path):
        """Recheck this deletion's evidence at the unlink boundary, under lease."""
        evidence = self.deletion_evidence.get(relative)
        if not evidence:
            raise ValueError("缺少本次刪除的用途驗證")
        for path, revision in evidence.items():
            file = self.root / path
            if not clean_path(file) or file_change_signature(file) != revision:
                raise ValueError("用途、發布成果或替代來源在驗證後變動")

    @contextmanager
    def session(self, worker):
        with lifecycle_access(self.root) if worker.apply else nullcontext():
            if self.errors and (self.records or self.extra_targets or any(p.name == MANIFEST_NAME for p in self.errors)):
                first = next(iter(self.errors))
                raise ValueError(f"依賴證據不完整：{first.as_posix()}：{self.errors[first]}")
            authority, scopes = self._authority_snapshot(worker)
            self._require_authority(authority)
            if self.records or self.bundles or self.extra_targets:
                self._refresh_reference_snapshot(scopes)
            self._require_authority(authority)
            self._refresh_closure()
            yield

    def removed(self, relative):
        self.deletion_evidence.pop(relative, None)
        self.files.pop(relative, None)
        self.revisions.pop(relative, None)
        self.reference_revisions.pop(relative, None)
        self.pending_reference_paths.discard(relative)
        self.directory_revisions.pop(relative, None)
        if file_change_signature(self.root / relative) == ("missing",):
            self.deleted_paths.add(relative)
        parent = relative.parent
        if parent in self.directory_revisions:
            self.directory_revisions[parent] = file_change_signature(self.root / parent)
        if self.bundle_rule and self.root / parent == self.bundle_rule.target_dir:
            self.bundle_directory_revision = file_change_signature(self.bundle_rule.target_dir)

    def clear(self, worker, candidates):
        removed = set()
        for relative, size in candidates:
            worker.remove_one("lifecycle", relative, size, self)
            if worker.apply and not (self.root / relative).exists():
                removed.add(relative)
        if worker.apply:
            self._empty_work_directories(worker, removed)

    def _empty_work_directories(self, worker, removed):
        directories = set()
        for path in removed:
            pair = self.artifacts.get(path)
            if not pair:
                continue
            workspace = pair[0].workspace
            directories.update(parent for parent in path.parents
                               if parent != workspace and parent.is_relative_to(workspace))
        for relative in sorted(directories, key=lambda p: (-len(p.parts), p.as_posix())):
            try:
                path = worker.deletion_path(relative)
                if (relative in self.directories and clean_path(path)
                        and path.is_dir() and not worker.protected_by_git(relative)
                        and not self.protected(relative) and not any(path.iterdir())):
                    path.rmdir()
                    self.removed(relative)
                    worker.note("RMDIR", relative, 0, "已完成清理的空工作目錄")
            except (OSError, ValueError) as exc:
                worker.note("SKIP", relative, 0, exc)

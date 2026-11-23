"""Resolve producer-declared work lifecycles for the existing cleaner.

This module plans proofs only. The cleaner retains ownership of unlink, Git,
NTFS, per-file revalidation, progress, and the existing recovery receipt.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
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
        self.bundles = set()
        self.bundle_rule = None
        self.bundle_config = None
        self.active = set()
        self.external_refs = set()
        self.verified_records = set()
        self.bundle_directory_revision = None
        self.bundle_kept_revisions = {}

    def observe_directory(self, relative):
        self.directories.add(relative)
        if ((relative == Path(".") or relative.parts[0] in {"outputs", "models", "state", "config", "data"})
                and not relative.is_relative_to("outputs/project_cleanup")):
            self.directory_revisions[relative] = file_change_signature(self.root / relative)

    def observe(self, relative: Path, size: int):
        self.files[relative] = size

    def prepare(self, active_subjects):
        self.active = set(active_subjects)
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
        if self.records or self.bundles:
            self.external_refs = self._external_references()
            self._refresh_closure()
            self.bundles = {path for path in self.bundles if not self.protected(path)}

    def protected(self, relative):
        return relative in self.keep or any(parent in self.keep for parent in relative.parents)

    def _reference_targets(self):
        # Broad output-root settings are destinations, not dependencies on every
        # past artifact. An explicit lifecycle retain/requires can still pin one.
        targets = set(self.bundles) | set(self.records)
        for path, (record, _) in self.artifacts.items():
            targets.add(path)
            targets.update(parent for parent in path.parents
                           if parent.is_relative_to(record.workspace))
        return targets

    def _refresh_closure(self):
        refs = set(self.external_refs)
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

    def _external_references(self):
        """Protect references from existing result/config/state JSON as well.

        References are a veto, never deletion authority. Unknown/malformed
        evidence blocks lifecycle deletions; it is not rewritten or fabricated.
        """
        ancestors = self._reference_targets()
        refs = set()
        for index, relative in enumerate(self.files, 1):
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
                if not clean_path(file) or not plain(file.lstat()):
                    raise ValueError("引用紀錄不是一般檔案")
                payload = json.loads(file.read_text(encoding="utf-8-sig"))
                if file_change_signature(file) != revision:
                    raise ValueError("引用紀錄在讀取期間變動")
                self.revisions[relative] = revision
                for value in _strings(payload):
                    # Not every JSON string is a path. Only strings resolving
                    # to a declared file or its ancestor can protect artifacts.
                    if not value or '\x00' in value:
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
        for path, digest in record.publication:
            file = self.root / path
            if not clean_path(file) or not plain(file.lstat()) or worker.hash_file(file) != digest:
                raise ValueError("正式／替代成果缺少或內容已變更")

    def proof(self, worker, relative: Path):
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
            return "超過設定保留範圍的已發布測試診斷封包"
        record, artifact = self.artifacts[relative]
        if not self.candidate(relative):
            return ""
        self._verify_publication(worker, record)
        if worker.hash_file(self.root / relative) != artifact.sha256:
            raise ValueError("工作檔內容與產生者的用途紀錄不同")
        if artifact.role == "replica":
            if artifact.replacement is None or artifact.replacement == relative:
                raise ValueError("副本缺少可保留的替代來源")
            replacement = self.root / artifact.replacement
            if (not clean_path(replacement) or not plain(replacement.lstat())
                    or worker.hash_file(replacement) != artifact.sha256):
                raise ValueError("副本與替代來源內容不一致")
        return {"COMPLETED": "已完成工作且沒有保留依賴的中間檔／副本",
                "SUPERSEDED": "產生者確認已取代且沒有保留依賴的工作檔／快取",
                "ABANDONED": "產生者確認已結案且不需續跑的工作殘留"}[record.state]

    @contextmanager
    def session(self, worker):
        with lifecycle_access(self.root) if worker.apply else nullcontext():
            if self.errors and (self.records or any(p.name == MANIFEST_NAME for p in self.errors)):
                first = next(iter(self.errors))
                raise ValueError(f"依賴證據不完整：{first.as_posix()}：{self.errors[first]}")
            self._refresh_closure()
            # Revalidate the entire evidence snapshot before any unlink. A
            # prior preview never authorizes a later apply.
            for index, (path, revision) in enumerate(self.revisions.items(), 1):
                self.progress(f"再次核對用途與引用 {index:,}/{len(self.revisions):,}")
                if file_change_signature(self.root / path) != revision:
                    raise ValueError("用途或引用在盤點後變動，請重新盤點")
            if self.records or self.bundles:
                for index, (path, revision) in enumerate(self.directory_revisions.items(), 1):
                    self.progress(f"核對盤點後的目錄變動 {index:,}/{len(self.directory_revisions):,}")
                    if file_change_signature(self.root / path) != revision:
                        raise ValueError("檔案清單在盤點後變動，可能新增引用，請重新盤點")
            yield

    def removed(self, relative):
        self.files.pop(relative, None)
        self.revisions.pop(relative, None)
        self.directory_revisions.pop(relative, None)
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
            path = self.root / relative
            try:
                if (relative in self.directories and clean_path(path)
                        and path.is_dir() and not worker.protected_by_git(relative)
                        and not self.protected(relative) and not any(path.iterdir())):
                    path.rmdir()
                    self.removed(relative)
                    worker.note("RMDIR", relative, 0, "已完成清理的空工作目錄")
            except OSError as exc:
                worker.note("SKIP", relative, 0, exc)

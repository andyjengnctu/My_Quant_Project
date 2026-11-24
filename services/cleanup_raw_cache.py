"""Deduplicate producer-identified execution caches without loading pickle."""
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path

from core.compare_csv_storage import clean_path, file_metadata
from core.file_integrity import canonical_json_sha256, file_change_signature
from services.optimizer.raw_cache import (
    inspect_named_raw_cache_metadata, raw_cache_writer_lease,
)



def _payload_location(relative):
    relative = Path(relative)
    return (not relative.is_absolute() and ".." not in relative.parts
            and len(relative.parts) >= 4 and relative.parts[0] == "outputs"
            and relative.parent.name == "raw_cache" and relative.suffix == ".pkl")


def _retained_namespace(relative):
    parts = tuple(part.casefold() for part in relative.parts)
    return "trading" in parts[:3] or "promotion_releases" in parts


def _metadata(root, relative):
    result = inspect_named_raw_cache_metadata(root / relative)
    if not result.ready:
        raise ValueError(result.reason)
    return result.value


class RawCacheCleanup:
    def __init__(self, root):
        self.root = Path(root)
        self.records = {}
        self.sources = {}
        self.errors = {}
        self.evidence = {}
        self.leases = set()

    def prepare(self, inventory):
        self.records.clear()
        self.sources.clear()
        self.errors.clear()
        self.evidence.clear()
        groups = defaultdict(list)
        for relative in inventory:
            if not _payload_location(relative):
                continue
            try:
                record = _metadata(self.root, relative)
                self.records[relative] = record
                groups[canonical_json_sha256(record[0])].append(relative)
            except (OSError, ValueError, TypeError) as exc:
                self.errors[relative] = str(exc)
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda path: (not _retained_namespace(path), path.as_posix()))
            retained = members[0]
            for relative in members[1:]:
                if not _retained_namespace(relative):
                    self.sources[relative] = retained

    def candidate(self, relative):
        return relative in self.sources

    @property
    def reference_targets(self):
        return {path for relative in self.sources for path in (relative, relative.parent)}

    @property
    def proof_revisions(self):
        """The prepare-time metadata authority; never refresh it on access.

        Payload equality is established separately under both writer leases in
        proof(), and remains subject to verify_before_delete().
        """
        used = set(self.sources) | set(self.sources.values())
        return {(self.root / relative).with_suffix(".json"): self.records[relative][2]
                for relative in used}

    @property
    def proof_scopes(self):
        """Both exact cache workspaces must remain stable between phases."""
        used = set(self.sources) | set(self.sources.values())
        return {self.records[relative][1]["cache_dir"].relative_to(self.root)
                for relative in used}

    @contextmanager
    def deletion_lease(self, worker, relative):
        source = self.sources[relative]
        locks = [self.records[path][1] for path in sorted((relative, source), key=lambda p: p.as_posix())]
        acquired = []
        try:
            with ExitStack() as stack:
                for paths in locks:
                    if not clean_path(paths["lock_dir"]) or not clean_path(paths["cache_dir"]):
                        raise ValueError("raw cache writer 鎖路徑不安全")
                    if worker.apply:
                        stack.enter_context(raw_cache_writer_lease(paths, blocking=False))
                        acquired.append(paths["lock_dir"])
                    elif paths["lock_dir"].exists():
                        raise RuntimeError("raw cache writer 鎖存在，略過預估")
                self.leases.add(relative)
                try:
                    yield
                finally:
                    self.leases.discard(relative)
        finally:
            for path in acquired:
                worker.lifecycle.removed(path.relative_to(self.root))

    def proof(self, worker, relative):
        self.evidence.pop(relative, None)
        if relative not in self.sources or relative not in self.leases:
            raise ValueError("raw cache 尚未取得兩端 writer lease")
        source = self.sources[relative]
        evidence = {}
        for item in (relative, source):
            metadata, _, revision = _metadata(self.root, item)
            original, _, original_revision = self.records[item]
            if metadata != original or revision != original_revision:
                raise ValueError("raw cache 用途或來源在盤點後變動")
            evidence[(self.root / item).with_suffix(".json")] = revision
            evidence[self.root / item] = file_change_signature(self.root / item)
        path, retained = self.root / relative, self.root / source
        if (path.stat().st_size != retained.stat().st_size
                or file_metadata(path, path.lstat()) != file_metadata(retained, retained.lstat())
                or worker.hash_file(path) != worker.hash_file(retained)):
            return ""
        self.evidence[relative] = evidence
        self.verify_before_delete(worker, relative)
        return "產生者來源身分與內容均相同的 execution raw cache 副本；完整保留來源：" + source.as_posix()

    def verify_before_delete(self, worker, relative):
        if relative not in self.leases or relative not in self.evidence:
            raise ValueError("缺少本次 raw cache 刪除的 writer lease 或內容證明")
        for path, revision in self.evidence[relative].items():
            if not clean_path(path) or file_change_signature(path) != revision:
                raise ValueError("raw cache 副本、保留來源或用途在驗證後變動")

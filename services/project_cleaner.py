"""One pass over the live project. Only producer-proven disposables can be removed."""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import time
from zipfile import ZipFile, BadZipFile
import json

from core.compare_csv_storage import (DETAIL_NAMES, RETIRED_PAIR_EXPORTS, clean_path, compare_root,
                                      disk_bytes, file_metadata, plain, storage_lock, reused_detail_source)
from core.file_integrity import (compute_observed_file_sha256, exclusive_file_lock, file_change_signature,
                                 load_json_object_or_none)
from services import cleanup_rules as rules
from services.cleanup_report import LABEL, receipt, recover, report_path, show_summary
from services.cleanup_lifecycle import LifecycleCleanup

PROTECTED = {
    "apps": "專案程式", "services": "專案程式", "core": "專案程式",
    "filters": "專案程式", "strategies": "專案程式", "tools": "專案工具與測試",
    "requirements": "專案相依設定", ".git": "Git 工作紀錄", "data": "原始／正式資料", "state": "狀態與可接續工作",
    "doc": "文件與研究證據", "config": "專案設定",
}
SCIENTIFIC = {"runs", "rolling_oos", "point_in_time", "fitting_checkpoints", "strategy_params",
              "shared_fitting_checkpoints", "model_robustness", "strategy_param_schedule",
              "standard_model_comparison", "robustness", "training", "optimizer_runtime"}
FAILURES = (OSError, ValueError, RuntimeError, ImportError, sqlite3.Error, BadZipFile)


def walk(root, note, visit=None):
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            info = folder.lstat()
            if (not clean_path(folder) or not stat.S_ISDIR(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                note("SKIP", folder.relative_to(root), 0, "連結／重新解析點")
                continue
            with os.scandir(folder) as entries:
                if visit is not None:
                    visit(folder.relative_to(root))
                for entry in entries:
                    path = Path(entry.path)
                    try:
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                            note("SKIP", path.relative_to(root), 0, "連結／重新解析點")
                        elif stat.S_ISDIR(info.st_mode):
                            stack.append(path)
                        elif plain(info):
                            yield path.relative_to(root), info.st_size
                        else:
                            note("SKIP", path.relative_to(root), 0, "特殊檔案")
                    except OSError as exc:
                        note("ERROR", path.relative_to(root), 0, exc)
        except OSError as exc:
            note("ERROR", folder.relative_to(root), 0, exc)


def regular(path):
    if not clean_path(path):
        raise ValueError("路徑經過連結")
    info = path.lstat()
    if not plain(info) or not info.st_ino:
        raise ValueError("缺少一般檔案身分")
    return info


def retained(rel, live):
    parts = rel.parts
    if parts[0] in PROTECTED:
        return "KEEP", PROTECTED[parts[0]]
    if "trading" in parts[:3]:
        return "KEEP", "Trading 帳務／正式工件"
    if len(parts) >= 2 and parts[:2] == ("outputs", "project_cleanup"):
        return "KEEP", "清理服務紀錄"
    if parts[0] in {"models", "outputs"}:
        if live.intersection(parts):
            return "KEEP", "現行研究設定所用模型目錄"
        if SCIENTIFIC.intersection(parts) or rel.name in {"manifest.json", "model.pt", "scores.csv"}:
            return "KEEP", "正式研究、歷史證據或可續跑工件"
        return "REVIEW", "尚缺產生者的逐檔用途、完成／取代證據，保留"
    return "REVIEW", "非清理規則管理的本機檔案"


def pair_location(root, rel):
    p = root / rel
    return compare_root(p) == root


class Cleaner:
    def __init__(self, root, log, apply):
        self.root, self.note, self.apply = root, log.row, apply
        self.last_progress = time.monotonic()
        self.units = {}
        self.publication_revisions = {}
        self.pinned_hashes = {}
        self.hashes = {}
        self.groups = defaultdict(list)
        self.deletes = defaultdict(list)
        self.work_dirs = defaultdict(set)
        self.git_ok = True
        self.lifecycle = LifecycleCleanup(root, self.progress)
        self.lifecycle_candidates = []
        self.refresh_tracking()
        self.legacy = self.lookup(rules.legacy_parameters, Path("models"))
        self.retired = self.lookup(rules.retired_diagnostics, rules.DIAGNOSTICS)
        from core.research_summary import get_research_summary_model_profiles
        self.live = {p for _, p in get_research_summary_model_profiles()}

    def refresh_tracking(self):
        try:
            self.tracked, self.index_path, self.index_revision = rules.tracking_snapshot(self.root)
        except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            self.tracked = set()
            self.git_ok = False
            self.note("SKIP", Path("."), 0, f"Git 索引無法核對；略過移除操作：{exc}")

    def protected_by_git(self, rel):
        if self.git_ok and file_change_signature(self.index_path) != self.index_revision:
            self.refresh_tracking()
        return not self.git_ok or rel in self.tracked

    def lookup(self, fn, rel):
        try:
            return fn(self.root)
        except FAILURES as exc:
            self.note("SKIP", rel, 0, f"無法取得此類別的保留／清理依據：{exc}")
            return {}

    def progress(self, message):
        if time.monotonic() - self.last_progress >= 4:
            print(message, flush=True)
            self.last_progress = time.monotonic()

    def hash_file(self, path):
        """Share byte verification within this scan, never cache publication readiness."""
        path = Path(path)
        revision = file_change_signature(path)
        cached = self.hashes.get(revision)
        if cached:
            return cached
        self.progress(f"正在驗證內容 {path.relative_to(self.root).as_posix()}")
        digest = compute_observed_file_sha256(path)
        if file_change_signature(path) != revision:
            raise ValueError("驗證期間檔案變動")
        self.hashes[revision] = digest
        return digest

    def published_pairs(self, run_rel):
        from services.research.strategy_compare_pair_integrity import completed_strategy_run_publication_matches
        if run_rel not in self.units:
            run = self.root / run_rel
            names = set()
            markers = (run / "manifest.json", run / "strategy_comparison.json")
            revisions = {p: file_change_signature(p) for p in markers}
            if all(clean_path(run / n) and (run / n).is_file() for n in ("manifest.json", "strategy_comparison.json")):
                payload = load_json_object_or_none(run / "strategy_comparison.json")
                pairs = payload.get("pairs") if isinstance(payload, dict) else None
                if isinstance(pairs, dict) and completed_strategy_run_publication_matches(
                        run_dir=run, run_payload=payload, project_root=self.root, file_hasher=self.hash_file):
                    for name, pair in pairs.items():
                        if (not isinstance(name, str) or not name or name in {".", ".."}
                                or Path(name).name != name or not isinstance(pair, dict)):
                            continue
                        file = run / "pairs" / name / "strategy_comparison.json"
                        pair_revision = file_change_signature(file)
                        if (clean_path(file) and load_json_object_or_none(file) == pair
                                and pair_revision == file_change_signature(file)
                                and all(file_change_signature(p) == v for p, v in revisions.items())):
                            names.add(name)
                            self.publication_revisions[file.parent] = {**revisions, file: pair_revision}
                            record = payload.get("pair_file_integrity")
                            if isinstance(record, dict):
                                for filename, digest in record["pairs"][name].items():
                                    self.pinned_hashes[file.parent / filename] = str(digest).strip().lower()
            self.units[run_rel] = names
        return self.units[run_rel]

    def published(self, rel):
        if rel.parent.name not in self.published_pairs(rel.parents[2]):
            return False
        return all(clean_path(p) and file_change_signature(p) == revision
                   for p, revision in self.publication_revisions[self.root / rel.parent].items())

    def kind(self, rel):
        if rel in self.legacy:
            return "legacy", None
        if rel in self.retired:
            return "retired_cache", None
        if rel.suffix == ".pyc" and rel.parent.name == "__pycache__" and rel.parts[0] not in {"models","outputs","state","data",".git"}:
            return "pyc", None
        if rel.is_relative_to(rules.STAGING):
            local = rel.relative_to(rules.STAGING)
            if len(local.parts) >= 2 and rules.PROBE.fullmatch(local.parts[0]):
                return "fixture", rules.STAGING
        unit = rules.robust_unit(rel)
        if unit and len(rel.relative_to(unit).parts) >= 3:
            return "robust", unit
        if rel.name.startswith(".") and rel.suffix == ".npz" and rel.parts[:3] == ("outputs","filters","breakout_quality"):
            return "abandoned", None
        if rel.name in RETIRED_PAIR_EXPORTS and pair_location(self.root, rel):
            return "old_export", None
        return None, None

    def proof(self, kind, rel, scope):
        path = self.root / rel
        if kind == "pyc" and rules.generated_bytecode(self.root, rel, self.tracked):
            return "可由現存原始碼重建的 Python 快取"
        if kind == "legacy" and rel in self.legacy:
            evidence = rules.legacy_parameters(self.root).get(rel)
            if evidence:
                return "Optimizer 已驗證遷移：" + evidence
        if kind == "retired_cache" and rel in self.retired and rules.is_projection(path):
            return "已離開現行清單的診斷投影快取：" + self.retired[rel]
        if kind == "fixture":
            return rules.fixture_reason(self.root, rel)
        if kind == "abandoned" and rules.abandoned_pid(path, rel):
            return "已結束程序留下的診斷快取發布暫存"
        if kind == "robust" and rel.relative_to(rules.robust_unit(rel)).parts[1] in scope:
            return "Robustness 已驗證正式成果，其保留規則捨棄此工作副本"
        if kind == "old_export" and self.published(rel):
            if (path.parent / "strategy_comparison.md").is_file():
                return "已發布比較的退役 HTML 匯出，正式報表已存在"
        if kind == "lifecycle":
            return scope.proof(self, rel)
        return ""

    def scan(self):
        print("盤點專案並核對保留依據…", flush=True)
        deferred = []
        def visit(rel):
            self.lifecycle.observe_directory(rel)
            unit = rules.robust_unit(rel)
            if unit and self.git_ok:
                self.work_dirs[unit].add(rel)
                self.deletes.setdefault(unit, [])
        for n, (rel, size) in enumerate(walk(self.root, self.note, visit), 1):
            self.lifecycle.observe(rel, size)
            self.progress(f"已盤點 {n:,} 個檔案，目前 {rel.parts[0]}/")
            if rel in self.tracked:
                self.note("KEEP", rel, size, "Git 追蹤檔")
                continue
            kind, lease = self.kind(rel)
            if kind and self.git_ok:
                self.deletes[lease].append((kind, rel, size))
                continue
            if self.git_ok and rel.name in DETAIL_NAMES and pair_location(self.root, rel) and size:
                path = self.root / rel
                try:
                    info = regular(path)
                    key = (rel.parts[2:4], rel.parent.name, rel.name, info.st_size, info.st_mtime_ns, info.st_dev)
                    self.groups[key].append(rel)
                except FAILURES as exc:
                    self.note("SKIP", rel, size, exc)
                continue
            deferred.append((rel, size))
        try:
            self.lifecycle.prepare(rules.active_artifact_subjects(self.root))
        except FAILURES as exc:
            self.lifecycle.errors[Path("config")] = f"現行依賴解析失敗：{exc}"
        for rel, size in deferred:
            if self.git_ok and self.lifecycle.candidate(rel):
                self.lifecycle_candidates.append((rel, size))
            else:
                reason = self.lifecycle.retained_reason(rel)
                action, fallback = retained(rel, self.live)
                self.note("KEEP" if reason else action, rel, size, reason or fallback)
        for rel, reason in self.lifecycle.errors.items():
            self.note("SKIP", rel, 0, reason)

    def clear_unused(self):
        if self.apply and self.git_ok:
            # Never consume an earlier preview as authority.
            self.refresh_tracking()
            if not self.git_ok:
                return
            self.legacy = self.lookup(rules.legacy_parameters, Path("models"))
            self.retired = self.lookup(rules.retired_diagnostics, rules.DIAGNOSTICS)
        try:
            if self.git_ok:
                self.lifecycle.active = rules.active_artifact_subjects(self.root)
                with self.lifecycle.session(self):
                    self._clear_unused()
        except FAILURES as exc:
            rows = list(self.lifecycle_candidates)
            rows.extend((rel, size) for entries in self.deletes.values() for _, rel, size in entries)
            for rel, size in rows:
                self.note("SKIP", rel, size, f"用途或保留依賴無法驗證：{exc}")

    def _clear_unused(self):
        self.lifecycle.clear(self, self.lifecycle_candidates)
        # Also consider known empty error-probe folders. Empty arbitrary folders are kept.
        staging = self.root / rules.STAGING
        if self.git_ok and clean_path(staging) and staging.is_dir():
            self.deletes.setdefault(rules.STAGING, [])
        for lease, rows in self.deletes.items():
            try:
                with rules.deletion_lease(self.root, lease, preview=not self.apply):
                    scope = rules.robust_disposable(self.root / lease) if lease and lease != rules.STAGING else set()
                    for kind, rel, size in rows:
                        self.remove_one(kind, rel, size, scope)
                    if lease == rules.STAGING and self.apply:
                        self.empty_fixtures(staging)
                    elif self.apply and scope:
                        self.empty_robust_work(lease, scope)
            except FAILURES as exc:
                for _, rel, size in rows:
                    self.note("SKIP", rel, size, f"工作仍在使用或無法驗證：{exc}")
                if not rows:
                    self.note("SKIP", lease, 0, f"工作仍在使用或無法驗證：{exc}")

    def empty_robust_work(self, unit, scope):
        # The same completed-run proof and writer lease cover files and empty
        # folders. Use only directories seen by the scan; never traverse anew
        # through a replaced link or remove an unrecognized work scope.
        for rel in sorted(self.work_dirs[unit], key=lambda p: (-len(p.parts), p)):
            parts = rel.relative_to(unit / "work").parts
            if parts and parts[0] not in scope:
                continue
            path = self.root / rel
            try:
                info = path.lstat()
                if (not self.protected_by_git(rel) and clean_path(path)
                        and stat.S_ISDIR(info.st_mode)
                        and not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                        and not self.lifecycle.protected(rel)
                        and not any(path.iterdir())):
                    path.rmdir()
                    self.lifecycle.removed(rel)
                    self.note("RMDIR", rel, 0, "已完成 Robustness 的空工作資料夾")
            except OSError as exc:
                self.note("SKIP", rel, 0, exc)

    def empty_fixtures(self, staging):
        for child in staging.iterdir():
            if not rules.PROBE.fullmatch(child.name) or not clean_path(child) or not child.is_dir():
                continue
            # Only remove the producer's known directory layout, never arbitrary subdirectories.
            for sub in ("models", "broken_optimizer_db/ml_optimizer/study_db",
                        "export_only_empty_db/ml_optimizer/study_db", "export_only_missing_db/ml_optimizer/study_db"):
                p = child / sub
                while p != child:
                    if (clean_path(p) and p.is_dir() and not self.lifecycle.protected(p.relative_to(self.root))
                            and not any(p.iterdir())):
                        p.rmdir()
                        self.lifecycle.removed(p.relative_to(self.root))
                        self.note("RMDIR", p.relative_to(self.root), 0, "已結束測試的空資料夾")
                    p = p.parent
            if not self.lifecycle.protected(child.relative_to(self.root)) and not any(child.iterdir()):
                child.rmdir()
                self.lifecycle.removed(child.relative_to(self.root))
                self.note("RMDIR", child.relative_to(self.root), 0, "已結束測試的空資料夾")

    def remove_one(self, kind, rel, size, scope):
        path = self.root / rel
        try:
            if self.lifecycle.protected(rel):
                self.note("KEEP", rel, size, "被正式成果、續跑或其他保留工件引用")
                return
            info = regular(path)
            revision = file_change_signature(path)
            reason = self.proof(kind, rel, scope)
            if not reason or self.protected_by_git(rel):
                self.note("KEEP", rel, size, "現行、工作尚未完成或刪除證據不足")
                return
            if info.st_nlink != 1 or info.st_size != size:
                self.note("SKIP", rel, size, "多重連結或尺寸已變動")
                return
            if os.name == "nt":
                file_metadata(path, info)  # Preserve files with additional NTFS data streams.
            space = disk_bytes(path, info)
            if self.apply:
                if (self.protected_by_git(rel) or not clean_path(path)
                        or file_change_signature(path) != revision):
                    raise ValueError("檔案已變動")
                self.note("INTENT", rel, size, reason)
                path.unlink()
                self.lifecycle.removed(rel)
            self.note("DELETE" if self.apply else "PLAN_DELETE", rel, size, reason, space)
        except FAILURES as exc:
            self.note("SKIP", rel, size, exc)

    def clear_reused_details(self):
        try:
            with self.lifecycle.session(self):
                self._clear_reused_details()
        except FAILURES as exc:
            for rows in self.groups.values():
                for rel in rows:
                    self.note("SKIP", rel, self.lifecycle.files.get(rel, 0), f"保留依賴無法驗證：{exc}")

    def _clear_reused_details(self):
        print("核對 REUSE 來源，刪除已有完整原始資料的重複明細…", flush=True)
        candidates = sorted(rel for paths in self.groups.values() for rel in paths)
        for index, rel in enumerate(candidates, 1):
            self.progress(f"已核對 REUSE 明細 {index:,}/{len(candidates):,}")
            path = self.root / rel
            size = 0
            try:
                if self.lifecycle.protected(rel):
                    self.note("KEEP", rel, self.lifecycle.files.get(rel, 0), "被正式成果、續跑或其他保留工件引用")
                    continue
                info = regular(path)
                size = info.st_size
                revision = file_change_signature(path)
                lineage_revisions = {}
                source = reused_detail_source(path, revisions=lineage_revisions)
                if source is None:
                    self.note("KEEP", rel, size, "原始比較明細，沒有可取代的 REUSE 來源")
                    continue
                if not self.published(rel):
                    raise ValueError("副本尚未完整發布或發布內容無法驗證")
                digest = self.hash_file(path)
                access = file_metadata(path, info)
                # Follow only byte-identical, committed producer provenance. An
                # intervening copy with changed diagnostics remains an original.
                seen = {path}
                retained = None
                while source is not None:
                    if source in seen:
                        raise ValueError("REUSE 來源循環")
                    seen.add(source)
                    source_rel = source.relative_to(self.root)
                    source_info = regular(source)
                    if (source_info.st_size != size or not self.published(source_rel)
                            or file_metadata(source, source_info) != access
                            or self.hash_file(source) != digest):
                        break
                    retained = source
                    source = reused_detail_source(source, revisions=lineage_revisions)
                if retained is None:
                    self.note("KEEP", rel, size, "REUSE 來源不存在、內容不同或未通過完整性驗證")
                    continue
                source = retained
                source_rel = source.relative_to(self.root)
                source_revision = file_change_signature(source)
                if (self.pinned_hashes.get(path, digest) != digest
                        or self.pinned_hashes.get(source, digest) != digest):
                    raise ValueError("內容已偏離已驗證的發布紀錄")
                reason = "已有完整且內容一致的 REUSE 原始來源，移除重複副本"
                with storage_lock(self.root) if self.apply else nullcontext():
                    if (self.protected_by_git(rel) or not clean_path(path) or not clean_path(source)
                            or file_change_signature(path) != revision
                            or file_change_signature(source) != source_revision
                            or any(not clean_path(p) or file_change_signature(p) != stamp
                                   for p, stamp in lineage_revisions.items())
                            or not self.published(rel) or not self.published(source_rel)):
                        raise ValueError("檔案或發布紀錄在核對後變動")
                    # Removing another name for an already shared inode does
                    # not free those data blocks a second time.
                    space = disk_bytes(path, regular(path)) if regular(path).st_nlink == 1 else 0
                    if self.apply:
                        self.note("INTENT", rel, size, reason, source=source_rel)
                        path.unlink()
                        self.lifecycle.removed(rel)
                        # Our unlink changes the shared inode's change time,
                        # not its bytes. Reuse that verified digest while the
                        # producer's publication lock is still held.
                        after = file_change_signature(source)
                        if revision[:2] == source_revision[:2] and after[:4] == source_revision[:4]:
                            self.hashes[after] = digest
                self.note("DELETE" if self.apply else "PLAN_DELETE", rel, size, reason, space, source_rel)
            except FAILURES as exc:
                self.note("SKIP", rel, size, exc)


@contextmanager
def session(root):
    lock = report_path(root, "cleaner.lock")
    with exclusive_file_lock(lock, conflict_message="另一個清理程式正在執行"):
        stage = root / "outputs/project_cleanup/pending.link"
        if stage.exists() or stage.is_symlink():
            # Only retire the link in our interrupted transaction. A matching
            # name and link count alone cannot establish ownership.
            import csv
            pending = report_path(root, "apply.pending.csv")
            source = None
            if pending.is_file():
                with pending.open(encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        if row.get("action") == "INTENT" and row.get("retained_copy"):
                            source = (root / row["retained_copy"]).absolute()
            if (source is None or not clean_path(source) or not source.resolve().is_relative_to(root)
                    or not clean_path(stage) or regular(stage).st_nlink < 2 or not source.is_file()
                    or not stage.samefile(source)):
                raise ValueError("暫存位置有無法核對來源的檔案，已保留。")
            stage.unlink()
        recover(root)
        yield


def run(root, apply=False):
    root = Path(root).resolve(strict=True)
    with session(root), receipt(root, "apply" if apply else "preview") as log:
        cleaner = Cleaner(root, log, apply)
        cleaner.scan()
        print(f"進入用途驗證 {sum(map(len, cleaner.deletes.values())) + len(cleaner.lifecycle_candidates):,} 個檔案；"
              f"核對 REUSE 副本 {sum(map(len, cleaner.groups.values())):,} 個明細。", flush=True)
        cleaner.clear_unused()
        cleaner.clear_reused_details()


def show_history(root):
    with session(root):
        report = report_path(root, "last_apply.zip")
        if not report.is_file():
            print("尚無清理紀錄。")
            return
        with ZipFile(report) as z:
            if not {"summary.json", "decisions.csv"}.issubset(z.namelist()):
                raise ValueError("清理紀錄缺少摘要或明細")
            show_summary(json.loads(z.read("summary.json")))
            print("完整逐檔紀錄：outputs/project_cleanup/last_apply.zip 內的 decisions.csv")
            query = input("輸入檔名或部分路徑查看明細（Enter 返回）：").strip().replace(chr(92), "/").casefold()
            if query:
                import csv
                import io
                matches = 0
                with z.open("decisions.csv") as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                    for row in csv.DictReader(text):
                        if row["action"] != "INTENT" and query in row["path"].casefold():
                            matches += 1
                            if matches <= 30:
                                print(f"{LABEL.get(row['action'], row['action'])}  {row['path']}  {row['reason']}")
                print(f"共 {matches:,} 筆，最多顯示前 30 筆。")

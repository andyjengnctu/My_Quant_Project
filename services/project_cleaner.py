"""One pass over the live project. Only producer-proven disposables can be removed."""
from collections import defaultdict
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import time
from zipfile import ZipFile
import json

from core.compare_csv_storage import (DETAIL_NAMES, RETIRED_PAIR_EXPORTS, clean_path, compare_root,
                                      disk_bytes, file_metadata, plain, storage_lock)
from core.file_integrity import (compute_observed_file_sha256, exclusive_file_lock, file_change_signature,
                                 load_json_object_or_none)
from services import cleanup_rules as rules
from services.cleanup_report import LABEL, receipt, recover, report_path, show_summary

PROTECTED = {
    "app": "專案程式", "apps": "專案程式", "services": "專案程式", "core": "專案程式",
    "filters": "專案程式", "strategies": "專案程式", "tools": "專案工具與測試",
    "requirements": "專案相依設定", ".git": "Git 工作紀錄", "data": "原始／正式資料", "state": "狀態與可接續工作",
    "doc": "文件與研究證據", "config": "專案設定",
}
SCIENTIFIC = {"runs", "rolling_oos", "point_in_time", "fitting_checkpoints", "strategy_params",
              "shared_fitting_checkpoints", "model_robustness", "strategy_param_schedule",
              "standard_model_comparison", "robustness", "training", "optimizer_runtime"}
FAILURES = (OSError, ValueError, RuntimeError, ImportError, sqlite3.Error)


def walk(root, note):
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            with os.scandir(folder) as entries:
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
        return "REVIEW", "產生程式尚未證明已失去用途"
    return "REVIEW", "非清理規則管理的本機檔案"


def pair_location(root, rel):
    p = root / rel
    return compare_root(p) == root


class Cleaner:
    def __init__(self, root, log, apply):
        self.root, self.note, self.apply = root, log.row, apply
        self.last_progress = time.monotonic()
        self.units = {}
        self.groups = defaultdict(list)
        self.deletes = defaultdict(list)
        self.git_ok = True
        try:
            self.tracked = rules.tracked_paths(root)
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            self.tracked = set()
            self.git_ok = False
            self.note("SKIP", Path("."), 0, f"Git 索引無法核對；略過移除與共用操作：{exc}")
        self.legacy = self.lookup(rules.legacy_parameters, Path("models"))
        self.retired = self.lookup(rules.retired_diagnostics, rules.DIAGNOSTICS)
        from core.research_summary import get_research_summary_model_profiles
        self.live = {p for _, p in get_research_summary_model_profiles()}

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

    def published_pairs(self, run_rel):
        from services.research.strategy_compare_pair_integrity import completed_strategy_run_publication_matches
        if run_rel not in self.units:
            run = self.root / run_rel
            names = set()
            if all(clean_path(run / n) and (run / n).is_file() for n in ("manifest.json", "strategy_comparison.json")):
                payload = load_json_object_or_none(run / "strategy_comparison.json")
                if isinstance(payload, dict) and completed_strategy_run_publication_matches(
                        run_dir=run, run_payload=payload, project_root=self.root):
                    for name, pair in (payload.get("pairs") or {}).items():
                        if (not isinstance(name, str) or not name or name in {".", ".."}
                                or Path(name).name != name or not isinstance(pair, dict)):
                            continue
                        file = run / "pairs" / name / "strategy_comparison.json"
                        if clean_path(file) and load_json_object_or_none(file) == pair:
                            names.add(name)
            self.units[run_rel] = names
        return self.units[run_rel]

    def kind(self, rel):
        if rel in self.legacy:
            return "legacy", None
        if rel in self.retired:
            return "retired_cache", None
        if rel.suffix == ".pyc" and rel.parent.name == "__pycache__" and rel.parts[0] not in {"models","outputs","state","data",".git"}:
            return "pyc", None
        if rel.is_relative_to(rules.STAGING) and len(rel.relative_to(rules.STAGING).parts) >= 2:
            return "fixture", rules.STAGING
        unit = rules.robust_unit(rel)
        if unit:
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
            return "Optimizer 已驗證遷移：" + self.legacy[rel]
        if kind == "retired_cache" and rel in self.retired and rules.is_projection(path):
            return "已離開現行清單的診斷投影快取：" + self.retired[rel]
        if kind == "fixture":
            return rules.fixture_reason(self.root, rel)
        if kind == "abandoned" and rules.abandoned_pid(path, rel):
            return "已結束程序留下的診斷快取發布暫存"
        if kind == "robust" and rel.relative_to(rules.robust_unit(rel)).parts[1] in scope:
            return "Robustness 已驗證正式成果，其保留規則捨棄此工作副本"
        if kind == "old_export" and rel.parent.name in self.published_pairs(rel.parents[2]):
            if (path.parent / "strategy_comparison.md").is_file():
                return "已發布比較的退役 HTML 匯出，正式報表已存在"
        return ""

    def scan(self):
        print("盤點專案並核對保留依據…", flush=True)
        for n, (rel, size) in enumerate(walk(self.root, self.note), 1):
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
            action, reason = retained(rel, self.live)
            self.note(action, rel, size, reason)

    def clear_unused(self):
        if self.apply and self.git_ok:
            # Never consume an earlier preview as authority.
            self.tracked = rules.tracked_paths(self.root)
            self.legacy = self.lookup(rules.legacy_parameters, Path("models"))
            self.retired = self.lookup(rules.retired_diagnostics, rules.DIAGNOSTICS)
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
            except FAILURES as exc:
                for _, rel, size in rows:
                    self.note("SKIP", rel, size, f"工作仍在使用或無法驗證：{exc}")

    def empty_fixtures(self, staging):
        for child in staging.iterdir():
            if not rules.PROBE.fullmatch(child.name) or not clean_path(child) or not child.is_dir():
                continue
            # Only remove the producer's known directory layout, never arbitrary subdirectories.
            for sub in ("models", "broken_optimizer_db/ml_optimizer/study_db",
                        "export_only_empty_db/ml_optimizer/study_db", "export_only_missing_db/ml_optimizer/study_db"):
                p = child / sub
                while p != child:
                    if clean_path(p) and p.is_dir() and not any(p.iterdir()):
                        p.rmdir()
                        self.note("RMDIR", p.relative_to(self.root), 0, "已結束測試的空資料夾")
                    p = p.parent
            if not any(child.iterdir()):
                child.rmdir()
                self.note("RMDIR", child.relative_to(self.root), 0, "已結束測試的空資料夾")

    def remove_one(self, kind, rel, size, scope):
        path = self.root / rel
        try:
            info = regular(path)
            revision = file_change_signature(path)
            reason = self.proof(kind, rel, scope)
            if not reason or rel in self.tracked:
                self.note("KEEP", rel, size, "現行、工作尚未完成或刪除證據不足")
                return
            if info.st_nlink != 1 or info.st_size != size:
                self.note("SKIP", rel, size, "多重連結或尺寸已變動")
                return
            if os.name == "nt":
                file_metadata(path, info)  # Preserve files with additional NTFS data streams.
            space = disk_bytes(path, info)
            if self.apply:
                if not clean_path(path) or file_change_signature(path) != revision:
                    raise ValueError("檔案已變動")
                self.note("INTENT", rel, size, reason)
                path.unlink()
            self.note("DELETE" if self.apply else "PLAN_DELETE", rel, size, reason, space)
        except FAILURES as exc:
            self.note("SKIP", rel, size, exc)

    def share(self):
        print("驗證已發布的比較結果，核對重複 CSV 的實際內容…", flush=True)
        for i, (key, paths) in enumerate(self.groups.items(), 1):
            self.progress(f"已處理 CSV 候選群組 {i:,}/{len(self.groups):,}")
            try:
                self.share_group(paths, key[3], key[4])
            except FAILURES as exc:
                self.note("ERROR", paths[0], 0, str(exc))

    def share_group(self, paths, size, stamp):
        physical = defaultdict(list)
        for rel in sorted(paths):
            st = regular(self.root / rel)
            if st.st_size != size or st.st_mtime_ns != stamp:
                self.note("SKIP", rel, size, "盤點後尺寸或修改時間已變動")
                continue
            physical[(st.st_dev, st.st_ino)].append(rel)
        if len(physical) == 1:
            for rel in next(iter(physical.values())):
                self.note("KEEP", rel, size, "已共用同一實體或無其他同條件副本")
            return
        masters = {}
        # Reuse existing verified link sets first; unknown external links never become masters.
        for members in sorted(physical.values(), key=lambda v: (-len(v), v[0])):
            path = self.root / members[0]
            published = all(rel.parent.name in self.published_pairs(rel.parents[2]) for rel in members)
            if not published or regular(path).st_nlink != len(members):
                for name in members:
                    self.note("KEEP", name, size, "尚未完成發布、完整性無法驗證或有其他連結")
                continue
            try:
                before = file_change_signature(path)
                info = regular(path)
                if info.st_size != size or info.st_mtime_ns != stamp:
                    raise ValueError("盤點後尺寸或修改時間已變動")
                access = file_metadata(path, info)
                digest = compute_observed_file_sha256(path)
                if before != file_change_signature(path):
                    raise ValueError("比對期間檔案變動")
                key = (digest, access)
                if key not in masters:
                    masters[key] = [path, before]
                    for name in members:
                        self.note("KEEP", name, size, "保留已驗證實體資料")
                    continue
                if len(members) != 1:
                    for name in members:
                        self.note("KEEP", name, size, "既有獨立共用群組")
                    continue
                self.link_one(masters[key], members[0], size, before, info)
            except FAILURES as exc:
                for name in members:
                    self.note("SKIP", name, size, exc)

    def link_one(self, master, rel, size, expected, info):
        path = self.root / rel
        source, source_revision = master
        source_rel = source.relative_to(self.root)
        space = disk_bytes(path, info)
        reason = "已發布且內容、修改時間、存取權限一致"
        if self.apply:
            with storage_lock(self.root):
                if (not clean_path(source) or not clean_path(path)
                        or file_change_signature(source) != source_revision
                        or file_change_signature(path) != expected or regular(path).st_nlink != 1):
                    raise ValueError("檔案在盤點後變動")
                stage = self.root / "outputs/project_cleanup/pending.link"
                if stage.exists() or stage.is_symlink():
                    raise ValueError("清理暫存連結已被使用")
                self.note("INTENT", rel, size, reason, source=source_rel)
                os.link(source, stage)
                try:
                    os.replace(stage, path)
                finally:
                    stage.unlink(missing_ok=True)
                master[1] = file_change_signature(source)
        self.note("LINK" if self.apply else "PLAN_LINK", rel, size, reason, space, source_rel)


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
        print(f"進入用途驗證 {sum(map(len, cleaner.deletes.values())):,} 個檔案；"
              f"進入重複比對 {sum(len(paths) for paths in cleaner.groups.values() if len(paths) > 1):,} 個明細。", flush=True)
        cleaner.clear_unused()
        cleaner.share()


def show_history(root):
    with session(root):
        report = report_path(root, "last_apply.zip")
        if not report.is_file():
            print("尚無清理紀錄。")
            return
        with ZipFile(report) as z:
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

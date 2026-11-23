"""Inventory the local Stock project using file metadata. Run with standard Python."""

import csv
import heapq
import io
import os
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def tracked_files(root):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, check=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout

    try:
        git_root = os.fsdecode(git("rev-parse", "--show-toplevel").rstrip(b"\r\n"))
        if Path(git_root).resolve() != root:
            return None, "專案不是 Git 工作樹根目錄"
        paths = git("ls-files", "--cached", "-z").split(b"\0")
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, f"無法查詢 Git 索引（{type(exc).__name__}）"
    return {os.path.normcase(os.fsdecode(p)) for p in paths if p}, "已讀取 Git 索引"


def inventory(root, bundle, writing, tracked, git_status):
    # Each value starts with direct files only. Roll up once, after traversal.
    folders = {".": [0, 0]}
    pending = ["."]
    largest = []
    skipped = []
    count = 0
    issues = 0
    next_update = time.monotonic() + 5

    def directory_entries(folder):
        nonlocal issues
        try:
            with os.scandir(root / folder) as entries:
                yield from entries
        except OSError as exc:
            skipped.append((folder, type(exc).__name__, exc.errno))
            issues += 1

    with bundle.open("files.csv", "w", force_zip64=True) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", errors="backslashreplace", newline="") as stream:
            rows = csv.writer(stream)
            rows.writerow(("path", "bytes", "modified_utc", "git_index"))
            while pending:
                folder = pending.pop()
                for entry in directory_entries(folder):
                    relative = entry.name if folder == "." else f"{folder}/{entry.name}"
                    if relative == writing:
                        continue
                    if time.monotonic() >= next_update:
                        print(f"已盤點 {count:,} 個檔案，目前目錄 {folder}", flush=True)
                        next_update = time.monotonic() + 5
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        skipped.append((relative, type(exc).__name__, exc.errno))
                        issues += 1
                        continue
                    mode = info.st_mode
                    reparse = getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                    if stat.S_ISLNK(mode) or reparse:
                        skipped.append((relative, "link_or_reparse_point", ""))
                    elif stat.S_ISDIR(mode):
                        folders[relative] = [0, 0]
                        pending.append(relative)
                    elif stat.S_ISREG(mode):
                        status = "unknown" if tracked is None else (
                            "listed" if os.path.normcase(relative) in tracked else "unlisted"
                        )
                        try:
                            modified = datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat()
                        except (OSError, OverflowError, ValueError) as exc:
                            modified = ""
                            skipped.append((relative, "mtime_" + type(exc).__name__, ""))
                            issues += 1
                        rows.writerow(("./" + relative, info.st_size, modified, status))
                        count += 1
                        folders[folder][0] += 1
                        folders[folder][1] += info.st_size
                        item = (info.st_size, relative)
                        if len(largest) < 50:
                            heapq.heappush(largest, item)
                        else:
                            heapq.heappushpop(largest, item)
                    else:
                        skipped.append((relative, "special_file", ""))

    root_count, root_size = folders["."]
    for folder in reversed(folders):
        if folder != ".":
            parent = folders[folder.rpartition("/")[0] or "."]
            parent[0] += folders[folder][0]
            parent[1] += folders[folder][1]
    ordered = sorted(folders, key=lambda key: (-folders[key][1], key))

    def write_csv(name, header, records):
        with bundle.open(name, "w", force_zip64=True) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8-sig", errors="backslashreplace", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(records)

    write_csv("folders.csv", ("folder", "recursive_files", "recursive_bytes"),
              (("./" + key if key != "." else ".", *folders[key]) for key in ordered))
    write_csv("scan_notes.csv", ("path", "reason", "errno"),
              (("./" + key, reason, code) for key, reason, code in skipped))
    notes = [
        f"檔案數 {count:,} ｜ 邏輯大小 {folders['.'][1]:,} bytes（{folders['.'][1] / 2**30:.3f} GiB）",
        f"略過或異常紀錄 {len(skipped):,} 筆，其中讀取錯誤 {issues:,} 筆。{git_status}。",
        "容量按檔案路徑計算，硬連結各計一次。目錄已含子目錄，各層不可相加。",
        "", "第一層用量（檔案數 / GiB / 路徑）",
    ]
    top = [(root_size, root_count, "（根目錄直屬檔案）")]
    top.extend((folders[k][1], folders[k][0], k + "/") for k in folders if k != "." and "/" not in k)
    notes.extend(f"{n:>10,}  {size / 2**30:>10.3f}  {key}" for size, n, key in sorted(top, reverse=True))
    notes.extend(["", "最大 50 個檔案（bytes / 路徑）"])
    notes.extend(f"{size:>16,}  {name}" for size, name in sorted(largest, reverse=True))
    bundle.writestr("summary.txt", ("\n".join(notes) + "\n").encode("utf-8-sig", errors="backslashreplace"))
    return notes[:2]


def main():
    root = Path(__file__).resolve().parents[1]
    if not (root / "doc/PROJECT_SETTINGS.md").is_file():
        raise SystemExit("請將工具解壓到 Stock 專案根目錄，保留 tools/ 路徑。")
    target = root / "outputs/project_file_inventory"
    if target.resolve() != target:
        raise SystemExit("報告目錄指向其他路徑，請檢查 outputs/project_file_inventory。")
    target.mkdir(parents=True, exist_ok=True)
    print("開始盤點專案檔案大小與修改時間。", flush=True)
    tracked, git_status = tracked_files(root)
    prefix = datetime.now().strftime("inventory_%Y%m%d_%H%M%S_")
    with tempfile.NamedTemporaryFile(dir=target, prefix=prefix, suffix=".partial", delete=False) as temp:
        stage = Path(temp.name)
    try:
        with ZipFile(stage, "w", compression=ZIP_DEFLATED, compresslevel=3) as bundle:
            summary = inventory(root, bundle, stage.relative_to(root).as_posix(), tracked, git_status)
        report = stage.with_suffix(".zip")
        stage.rename(report)
    finally:
        stage.unlink(missing_ok=True)
    print(*summary, f"盤點報告 {report.relative_to(root).as_posix()}", sep="\n")


if __name__ == "__main__":
    main()

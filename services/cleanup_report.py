"""Keep one review and one cleanup receipt, including an interrupted receipt."""
from collections import Counter
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from core.compare_csv_storage import clean_path
from core.file_integrity import atomic_replace_with_retry

FIELDS = ("action", "path", "logical_bytes", "saved_bytes", "reason", "retained_copy")
LABEL = {
    "KEEP": "保留", "REVIEW": "待辨識／保留", "SKIP": "略過", "ERROR": "錯誤",
    "DELETE": "已刪除檔案", "LINK": "已共用內容", "PLAN_DELETE": "可刪檔案",
    "PLAN_LINK": "可共用內容", "RMDIR": "已刪空資料夾",
}


def report_path(root, name):
    p = root / "outputs/project_cleanup" / name
    if not clean_path(p) or (p.exists() and (not p.is_file() or p.stat().st_nlink != 1)):
        raise OSError("清理紀錄路徑異常")
    return p


def finish(root, mode, status, pending):
    counts, saved, logical = Counter(), Counter(), Counter()
    reasons = {}
    with pending.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            action = row["action"]
            if action not in LABEL:
                continue
            try:
                size, delta = int(row["logical_bytes"]), int(row["saved_bytes"])
            except (TypeError, ValueError):
                status = "中斷，最後一筆紀錄可能不完整"
                continue
            counts[action] += 1
            saved[action] += delta
            logical[action] += size
            reason = (row.get("reason") or "").split("：", 1)[0]
            entry = reasons.setdefault(action, {}).setdefault(reason, [0, 0])
            entry[0] += 1
            entry[1] += size
    info = dict(mode=mode, status=status, counts=counts, saved=saved, logical=logical, reasons=reasons)
    if status == "已完成" and (counts["SKIP"] or counts["ERROR"]):
        info["status"] = "已完成，部分項目略過或失敗，詳見明細"
    target = report_path(root, f"last_{mode}.zip")
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with ZipFile(tmp_path, "w", ZIP_DEFLATED, compresslevel=2) as z:
            z.write(pending, "decisions.csv")
            z.writestr("summary.json", json.dumps(info, ensure_ascii=False, indent=2))
        atomic_replace_with_retry(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    # Preserve the intent until a leftover link has been recovered. A failed
    # unlink or interrupted link syscall must not strand the next invocation.
    stage = root / "outputs/project_cleanup/pending.link"
    if not (mode == "apply" and (stage.exists() or stage.is_symlink())):
        pending.unlink()
    return info


def recover(root):
    for mode in ("apply", "preview"):
        pending = report_path(root, f"{mode}.pending.csv")
        if pending.is_file():
            finish(root, mode, "上次中斷；僅統計已記錄完成的操作", pending)


class Receipt:
    def __init__(self, root, mode, stream):
        self.root, self.mode, self.stream = root, mode, stream
        self.writer = csv.writer(stream)
        self.writer.writerow(FIELDS)
        stream.flush()

    def row(self, action, path, size, reason, saved=0, source=None):
        reason = str(reason).replace(str(self.root), ".")
        self.writer.writerow((action, "./" + path.as_posix(), size, saved, reason,
                              "./" + source.as_posix() if source else ""))
        if action in {"INTENT", "DELETE", "LINK", "ERROR"}:
            self.stream.flush()


@contextmanager
def receipt(root, mode):
    pending = report_path(root, f"{mode}.pending.csv")
    status = "已中斷；僅統計已記錄完成的操作"
    try:
        with pending.open("x", encoding="utf-8-sig", newline="") as stream:
            item = Receipt(root, mode, stream)
            yield item
            status = "已完成"
    finally:
        summary = finish(root, mode, status, pending)
        show_summary(summary)
        print(f"逐檔明細：outputs/project_cleanup/last_{mode}.zip（decisions.csv）")


def show_summary(summary):
    if (not isinstance(summary, dict) or summary.get("mode") not in {"apply", "preview"}
            or not isinstance(summary.get("status"), str)):
        raise ValueError("清理摘要格式不完整")
    for field in ("counts", "saved", "logical"):
        values = summary.get(field)
        if (not isinstance(values, dict)
                or any(not isinstance(v, int) or v < 0 for v in values.values())):
            raise ValueError("清理摘要統計格式不完整")
    print("\n" + summary["status"])
    for action, label in LABEL.items():
        count = summary["counts"].get(action, 0)
        if count:
            saved = summary["saved"].get(action, 0)
            print(f"  {label}  {count:,} 份，容量 {summary['logical'].get(action, 0)/2**30:.3f} GiB"
                  + (f"，可回收／已回收 {saved:,} bytes" if saved else ""))
            if action in {"KEEP", "REVIEW", "DELETE", "PLAN_DELETE", "SKIP", "ERROR"}:
                details = sorted(summary.get("reasons", {}).get(action, {}).items(), key=lambda item: -item[1][0])
                for reason, (n, size) in details[:8]:
                    print(f"    {reason}  {n:,} 份／{size/2**30:.3f} GiB")
    total = sum(summary["saved"].values())
    prefix = "預估可釋放" if summary["mode"] == "preview" else "依已完成檔案的磁碟配置量計算，共釋放"
    print(f"{prefix} {total/2**30:.3f} GiB（{total:,} bytes）")

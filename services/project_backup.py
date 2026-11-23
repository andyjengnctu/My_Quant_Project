"""Whole-project Windows backup with Robocopy's read-only difference check."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


DEFAULT_BACKUP_DRIVE = "D:/"
BACKUP_PREFIX = "My_Quant_Project"


def _robocopy(source, destination, log, *, verify=False):
    options = ["/E", "/COPY:DAT", "/DCOPY:DAT"]
    options += ["/L", "/R:0", "/W:0"] if verify else ["/Z", "/MT:8", "/R:2", "/W:2"]
    # Robocopy supports long paths by default; /256 would disable that support.
    # No /MIR, /PURGE or /MOVE: destination-only files and source files survive.
    command = ["robocopy.exe", str(source), str(destination), *options,
               "/NP", "/TEE", f"/UNILOG:{log}"]
    stage = "唯讀比對" if verify else "複製"
    try:
        # Keep native console output live, without shell quoting or decoding it.
        code = subprocess.run(command, check=False).returncode
    except KeyboardInterrupt as exc:
        raise RuntimeError(
            f"{stage}已中止，備份尚未驗證完成，先不要清理。\n紀錄：{log.as_posix()}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"無法執行 Robocopy {stage}，備份尚未驗證完成，先不要清理。"
            f"\n原因：{exc}\n紀錄位置（若已建立）：{log.as_posix()}"
        ) from exc
    if verify and code not in (0, 2):
        raise RuntimeError(
            f"仍有待複製項目、差異或讀取錯誤（Robocopy 代碼 {code}），先不要清理。"
            f"\n請查看：{log.as_posix()}"
        )
    if not verify and not 0 <= code < 8:
        raise RuntimeError(
            f"複製有失敗或中止（Robocopy 代碼 {code}），先不要清理原專案。"
            f"\n請查看：{log.as_posix()}"
        )


def backup_project(root, *, backup_drive=DEFAULT_BACKUP_DRIVE):
    """Copy the running project, then verify; never dispatch cleanup.

    The drive override is for service reuse/tests. The cleaner menu uses
    D: and creates an independent, identified backup each time. Verification uses
    Robocopy metadata comparison, not a byte-for-byte hash or live snapshot.
    """
    if sys.platform != "win32":
        raise RuntimeError("備份到 D 槽功能需要在 Windows 執行（使用 robocopy.exe）。")
    source = Path(root)
    if not source.is_dir():
        raise ValueError("找不到來源專案資料夾。")
    source = source.resolve(strict=True)
    drive = Path(backup_drive)
    if not drive.is_absolute():
        raise ValueError("備份磁碟必須是絕對路徑。")
    if not drive.is_dir():
        raise ValueError(f"找不到備份磁碟或資料夾：{drive.as_posix()}")
    backup_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    target = drive / f"{BACKUP_PREFIX}_{backup_id}"
    resolved_target = target.resolve()
    if source.is_relative_to(resolved_target) or resolved_target.is_relative_to(source):
        raise ValueError("來源與備份位置不能相同，也不能互相包含；請從原專案執行。")

    # Exclusive creation prevents a collision from merging or overwriting runs.
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("備份辨識碼已存在，未覆蓋舊備份；請重新選擇備份。") from exc
    # Both logs share the backup ID and stay outside the copied tree.
    copy_log = drive / f"{BACKUP_PREFIX}_backup_{backup_id}.log"
    check_log = drive / f"{BACKUP_PREFIX}_verify_{backup_id}.log"
    print(f"\n備份辨識碼：{backup_id}\n備份來源：目前專案根目錄（.）"
          f"\n備份位置：{target.as_posix()}", flush=True)
    print("本次使用獨立資料夾，完整保存一份專案。", flush=True)
    print("備份期間請勿修改專案檔案。", flush=True)
    print(f"複製紀錄：{copy_log.as_posix()}\n比對紀錄：{check_log.as_posix()}", flush=True)
    print("\n[1/2] 開始複製整個專案…", flush=True)
    _robocopy(source, target, copy_log)
    print("\n[2/2] 開始唯讀比對，不修改專案與備份檔案…", flush=True)
    _robocopy(source, target, check_log, verify=True)
    print(f"\n備份與唯讀比對完成：{target.as_posix()}", flush=True)
    print(f"複製紀錄：{copy_log.as_posix()}\n比對紀錄：{check_log.as_posix()}", flush=True)
    return target


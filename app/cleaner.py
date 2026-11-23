"""Interactive, standalone Stock project cleaner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.project_cleaner import FAILURES, run, show_history


def main():
    if not (ROOT / "doc/PROJECT_SETTINGS.md").is_file():
        raise SystemExit("請將程式放在專案的 app/ 目錄。")
    while True:
        print("\n專案檔案清理\n[1] 分析檔案與預估可釋放空間（預設）"
              "\n[2] 一鍵清理\n[3] 查看最近一次清理紀錄\n[0] 離開")
        try:
            answer = input("請選擇：").strip() or "1"
            if answer == "0":
                return
            if answer == "3":
                show_history(ROOT)
            elif answer in {"1", "2"}:
                run(ROOT, apply=answer == "2")
            else:
                print("請輸入 0～3。")
        except EOFError:
            return
        except KeyboardInterrupt:
            print("\n操作已中止；已完成的操作已記錄。")
        except FAILURES as exc:
            print(f"本次操作未完成：{str(exc).replace(str(ROOT), '.')}")


if __name__ == "__main__":
    main()

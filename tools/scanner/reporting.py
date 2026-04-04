from core.buy_sort import get_buy_sort_title
from core.config import get_buy_sort_method
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW


def print_scanner_start_banner(now_label):
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"{C_CYAN}🚀 啟動【v16 尊爵版】極速平行掃描儀 | 時間: {now_label}{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")


def print_scanner_summary(*, count_scanned, elapsed_time, count_history_qualified, count_skipped_insufficient, count_sanitized_candidates, max_workers, pool_start_method, candidate_rows, scanner_issue_log_path):
    candidate_rows.sort(key=lambda x: (x['sort_value'], x['ticker']), reverse=True)
    sort_title = get_buy_sort_title(get_buy_sort_method())

    print(" " * 160, end="\r")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(
        f"⚡ 掃描完畢！共掃描 {count_scanned} 檔標的，耗時 {elapsed_time:.2f} 秒。"
        f"歷史及格候選: {count_history_qualified} 檔 | 資料不足跳過: {count_skipped_insufficient} 檔 | "
        f"候選清洗: {count_sanitized_candidates} 檔 | max_workers: {max_workers}"
        f" | start_method: {pool_start_method or 'default'}"
    )

    if candidate_rows:
        new_count = sum(1 for x in candidate_rows if x['kind'] == 'buy')
        extended_count = sum(1 for x in candidate_rows if x['kind'] == 'extended')
        print(f"\n{C_RED}🔥 【明日候選清單：新訊號 + 延續候選同池排序】 {sort_title} 🔥{C_RESET}")
        print(f"{C_GRAY}   顏色區分：{C_RED}紅色=新訊號{C_GRAY} | {C_YELLOW}黃色=延續候選{C_RESET}")
        print(f"{C_GRAY}   候選統計：新訊號 {new_count} 檔 | 延續候選 {extended_count} 檔{C_RESET}")
        for item in candidate_rows:
            prefix = "[新訊號]" if item['kind'] == 'buy' else "[延續候選]"
            color = C_RED if item['kind'] == 'buy' else C_YELLOW
            print(f"   {color}➤ {prefix} {item['text']}{C_RESET}")
    else:
        print(f"\n{C_GREEN}💤 今日無符合實戰買點的標的，保留現金，明日再戰！{C_RESET}")

    if scanner_issue_log_path:
        print(
            f"\n{C_YELLOW}⚠️ 清洗摘要已寫入: {scanner_issue_log_path} "
            f"(候選清洗 {count_sanitized_candidates} 檔){C_RESET}"
        )

    print(f"{C_CYAN}================================================================================{C_RESET}")

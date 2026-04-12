# GPT 交付前檢查表

用途：assistant 每輪交付前操作檢查表；不作本地端 formal test 主表、不記 `B` / `T` / `G` / `E` 狀態，也不取代 `PROJECT_SETTINGS.md` 與 `TEST_SUITE_CHECKLIST.md`。

文件分工：`PROJECT_SETTINGS.md` 管原則與邊界；`TEST_SUITE_CHECKLIST.md` 管本地端 formal test suite 收斂與維護；本檔只管 GPT 交付前操作檢查。

## A. 開始前

1. 確認本輪最新基準，並回報 ZIP 檔名、SHA256、全新解壓目錄與已讀文件。
2. 先檢查是否存在尚未列入 `doc/TEST_SUITE_CHECKLIST.md`、但應由正式入口涵蓋的缺口。

## B. Formal test chain 變更時

1. 對同一 target 逐項自檢 definition、import、registry、checklist、parser、guard、正式入口摘要、help 與對應 meta guard 是否已實際同步。
2. 追蹤 ID 或 canonical 主題更名時，須同步檢查所有引用層，不得只改局部別名、摘要或相鄰文件。
3. 若修改 checklist 主表、`T`、`G`、`E` 等機械排序區塊，交付前必須整段重排並對照既有排序 guard。

## C. Bundle 修復時

1. 逐條對照 bundle 原始失敗項完成閉環；不得以相鄰文件、註解、help 或 checklist 已同步視為修復完成。
2. 交付前必須逐條確認 bundle 原始失敗項已消失。

## D. 交付前

1. 依本輪最嚴格檢查結果，一次找出並修正所有目前可發現問題；不得將同源、同鏈或同契約的已知相鄰問題拆成多輪逐步釋出。
2. 若仍有無法同輪清除的阻塞，必須明確揭露；不得以「先修這部分」視為完成。
3. 僅在確認無已知問題後交付 patch / ZIP，並列出本輪修改檔案與必要替換方式。

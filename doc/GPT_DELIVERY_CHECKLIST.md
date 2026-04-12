# GPT 交付前檢查表

用途：assistant 每輪交付前操作檢查表；不作本地端 formal test 主表、不記 `B` / `T` / `G` / `E` 狀態，也不取代 `PROJECT_SETTINGS.md` 與 `TEST_SUITE_CHECKLIST.md`。

文件分工：`PROJECT_SETTINGS.md` 管原則與邊界；`TEST_SUITE_CHECKLIST.md` 管本地端 formal test suite 收斂與維護；本檔只管 GPT 交付前操作檢查。

## A. 開始前

1. 確認本輪最新基準，並回報當前工作基準、ZIP 檔名、SHA256、全新解壓目錄與已讀文件。
2. 先檢查是否存在尚未列入 `doc/TEST_SUITE_CHECKLIST.md`、但應由正式入口涵蓋的缺口。
3. 若確認沒有新增 formal coverage gap，交付時明確回報正式入口已涵蓋目前需求。

## B. Formal test chain 變更時

1. 對同一 target 逐項自檢 definition、import、registry、`doc/TEST_SUITE_CHECKLIST.md`、parser、guard、正式入口摘要、help 與對應 meta guard 是否已實際同步。
2. 追蹤 ID 或 canonical 主題更名時，須同步檢查所有引用層，不得只改局部別名、摘要或相鄰文件。
3. 若修改 `doc/TEST_SUITE_CHECKLIST.md` 主表、`T`、`G`、`E` 等機械排序區塊，交付前必須整段重排並對照既有排序 guard。
4. 若本輪新增 `G` 紀錄，交付前必須重新抽出整個對應日期區塊，依 tracking ID 穩定排序後整段覆寫回原位；不得將新列直接追加在同日區塊尾端、留在空白行之後，或只改局部單列後跳過整段重排。

## C. Bundle 修復時

1. 逐條對照 bundle 原始失敗項完成閉環；不得以相鄰文件、註解、help 或 `doc/TEST_SUITE_CHECKLIST.md` 已同步視為修復完成。
2. 交付前必須逐條確認 bundle 原始失敗項已消失。
3. 若前一輪修改在本輪仍被 bundle 或再檢查證明有錯，除修正原始失敗外，必須同步更新本檔，補上可直接防止同類錯誤再犯的檢查條款。

## D. 交付前

1. 依本輪最嚴格檢查結果，一次找出並修正所有目前可發現問題；不得將同源、同鏈或同契約的已知相鄰問題拆成多輪逐步釋出。
2. 若仍有無法同輪清除的阻塞，必須明確揭露；不得以「先修這部分」視為完成。
3. 完成 GPT 端最嚴格檢查後，僅在確認無已知問題時交付 patch / ZIP，並列出本輪修改檔案與必要替換方式。

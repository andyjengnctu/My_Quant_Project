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
4. 若本輪新增 `G` 紀錄，交付前必須重新抽出整個對應日期區塊，依 formal tracking ID sort key（prefix / numeric / suffix）穩定排序後整段覆寫回原位；不得以人工目測、字典序、尾端追加或局部插入取代正式排序。不得將新列直接追加在同日區塊尾端、留在空白行之後，或只改局部單列後跳過整段重排。
5. 只要本輪動到 `G` 區，交付前除對應日期區塊重排外，還必須再對整個 `G` 表執行一次由上到下的日期 / tracking ID 全表 guard 檢查，並確認同日區塊內 `B` / `T` 等追蹤列皆符合 formal tracking ID sort key；不得只檢當前日期區塊就交付。
6. 若本輪修改 `doc/PROJECT_SETTINGS.md`、`doc/ARCHITECTURE.md`、`doc/CMD.md` 等雙 checklist 分工文件，交付前必須逐行檢查是否殘留未指明檔名的裸 `checklist` 用詞；不得以語意接近或既有段落已同步視為完成。
7. 若本輪新增或調整 validator / Txx / Bxx，而 `apps/test_suite.py` 仍保留人工維護的 coverage 摘要註解或 `--help` 長說明，交付前必須全文搜尋並同步更新相關 Txx / contract 主題；不得只更新 registry、checklist 或 meta guard。
8. 若本輪修改任何會被 formal contract / parser / meta guard 逐字比對的 literal，或其對應的 canonical 名稱、追蹤 ID、正式入口摘要 / help 關鍵字，交付前必須從對應 validator、meta guard、parser 的 expected literal、禁止字串與比對條件反查，逐項核對所有正向與反向 literal；不得只憑語意相近、單一例句或局部全文搜尋視為完成。

## C. Bundle 修復時

1. 逐條對照 bundle 原始失敗項完成閉環；不得以相鄰文件、註解、help 或 `doc/TEST_SUITE_CHECKLIST.md` 已同步視為修復完成。
2. 交付前必須逐條確認 bundle 原始失敗項已消失。
3. 若前一輪修改在本輪仍被 bundle 或再檢查證明有錯，除修正原始失敗外，必須同步更新本檔，將防再犯要求上提為可泛化、可操作的交付前檢查；不得只補單一案例、單一字串或局部實作特例。若失敗來自 exact-string contract，新增或修改條款時還必須逐字核對 formal expected literal，不得只做 markdown 格式化、反引號包裝、全半形替換、標點微調或語意接近改寫。

## D. 交付前

1. 依本輪最嚴格檢查結果，一次找出並修正所有目前可發現問題；不得將同源、同鏈或同契約的已知相鄰問題拆成多輪逐步釋出。
2. 若仍有無法同輪清除的阻塞，必須明確揭露；不得以「先修這部分」視為完成。
3. 完成 GPT 端最嚴格檢查後，僅在確認無已知問題時交付 patch / ZIP，並列出本輪修改檔案與必要替換方式。

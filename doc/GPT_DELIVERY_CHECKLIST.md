# GPT 交付前檢查表

用途：assistant 每輪交付前操作檢查表；不作本地端 formal test 主表、不記 `B` / `T` / `G` / `E` 狀態，也不取代 `PROJECT_SETTINGS.md` 與 `TEST_SUITE_CHECKLIST.md`；本地 formal test、`apps/test_suite.py`、synthetic registry 與 bundle 檢查均不驗證本檔內容。

文件分工：`PROJECT_SETTINGS.md` 管原則與邊界；`TEST_SUITE_CHECKLIST.md` 管本地端 formal test suite 收斂與維護；`GPT_DELIVERY_CHECKLIST.md` 只管 GPT 交付前操作檢查。

原則：本檔只保留高層、可泛化、可操作的交付 guard；不得長期累積只對單次事故、單一句子、單一 wording 或局部實作特例有效的補丁型條款。

## A. 開始前

1. 確認本輪最新基準，並回報當前工作基準、ZIP 檔名、SHA256、全新解壓目錄與已讀文件。
2. 先檢查是否存在尚未列入 `doc/TEST_SUITE_CHECKLIST.md`、但應由正式入口涵蓋的 formal coverage gap。
3. 若確認沒有新增 formal coverage gap，交付時明確回報正式入口已涵蓋目前需求。

## B. 正式鏈、文件與契約

1. 若本輪修改 formal test chain，同一 target 必須整鏈核對 definition、import、registry、`doc/TEST_SUITE_CHECKLIST.md`、parser、guard、正式入口摘要與 help；任一層未同步，不得交付。
2. 若本輪修改 canonical 名稱、追蹤 ID、validator 名稱、正式入口名稱、共享 schema，或將既有 contract / case 改指向另一份文件或模組，必須同步反查所有引用層；不得只改函式本體，卻保留舊 case_id、T 項摘要、registry 描述或 help 主題。若 checklist section token 或摘要表 canonical 名稱已更名，也不得在 result name、case_id、validator 名稱或 registry 描述殘留舊 token。
3. 若新舊 formal 規則只是同一 shipped 契約下的子面向擴充，優先併入既有 `Bxx` / `Txx`；不要把已確認可合併的同族 CLI、文件、coverage completeness、per-file minimum，或 checklist 結構子契約（如首行標題、摘要表排序）再拆成平行主表 ID。若 repo 內既有獨立主表尚未實際退役，不得先在 GPT checklist 宣稱已一律合併。
4. 負向檢查與 contract 同步只針對 shipped 正式輸出、`--help`、schema、檔案樹與正式 payload 本體；不要再為 validator / synthetic / oracle 的內部 import、helper 選擇、AST 掃描策略、樣本 payload 字面值、sub-check 名稱、source literal 或 help wording / bare-term hygiene 追加新的 formal 規則。
5. 若本輪修改 `doc/TEST_SUITE_CHECKLIST.md` 的主表、`T`、`G`、`E` 等機械真理區，交付前必須整表核對排序、摘要、最新狀態與 transition 連續性；若插回既有日期區塊或補寫 `DONE -> N/A` / `DONE -> PARTIAL`，必須重排整個受影響同日區塊，不得只修新插入列附近幾行。
6. 若本輪修改 `doc/TEST_SUITE_CHECKLIST.md` 的 markdown table 列內容，交付前必須逐列核對欄數仍符合該表 header；退役 / 併入既有主表時，不得把 `B2/B3` 的「類別 / 項目」欄位擠成一格，也不得在表格 cell 內保留未轉寫的裸 `|` 字元。
7. formal-facing 文件、`doc/CMD.md`、`doc/ARCHITECTURE.md`、`apps/test_suite.py --help`、report 與 retention 說明只保留穩定、對使用者有意義的正式資訊；不得回指 `PROJECT_SETTINGS.md` / `GPT_DELIVERY_CHECKLIST.md`，也不得把 helper 長清單、暫時事故修補、內部 meta guard 名稱或非 canonical 路徑升格為正式契約。
8. 若本輪將既有 formal contract / case / `Bxx` / `Txx` 改列 `N/A`、compatibility stub 或其他非正式長期路徑，交付前必須同步清理 registry、parser、`doc/TEST_SUITE_CHECKLIST.md` 摘要與歷史狀態；不要再把已降級的治理 hygiene 升回 formal blocker。

## C. Bundle 與收斂

1. 若使用者提供 bundle，必須逐條對照 bundle 原始失敗項完成閉環；不得以相鄰文件、註解、help 或 checklist 已同步視為修復完成。
2. 若問題屬既有 Bxx / Txx / validator contract 鏈，交付前必須建立同源 / 同鏈 / 同契約收斂清單；至少列出鏈根、掃描範圍、逐項結果與未清阻塞，並一次清空相鄰缺口。
3. 若前一輪修改在本輪仍被 bundle 或再檢查證明有錯，除修正原始失敗外，必須同步把防再犯要求提升為可泛化、可操作的交付前規則；不得再補單一案例型條款。

## D. 交付前

1. 依本輪最嚴格檢查結果，一次找出並修正所有目前可發現問題；不得將同源、同鏈或同契約的已知相鄰問題拆成多輪逐步釋出。
2. 若仍有無法同輪清除的阻塞，必須明確揭露；不得以局部修補視為完成。
3. 完成交付前自檢時，只核對正式輸出、正式契約、必要文件同步與本輪修改涉及的同鏈項；不要再把 validator 寫法 hygiene、單次 wording 或事故級 patch 規則疊成第二層檢查。
4. 僅在確認無已知問題時交付 patch / ZIP，並列出本輪修改檔案與必要替換方式。

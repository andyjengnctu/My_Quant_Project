# 專案設定

1. 每輪開始前必須先讀取並遵守 `/doc/PROJECT_SETTINGS.md` 與 `/doc/TEST_SUITE_CHECKLIST.md`；其中 `PROJECT_SETTINGS.md` 為最高優先規則，`TEST_SUITE_CHECKLIST.md` 為 test suite 收斂與維護清單；兩者均不得忽略、弱化、選擇性遵守、以慣例覆蓋、或自行推定例外。

## A. 工作基準與執行紀律

1. 本輪基準為使用者最新提供的程式、ZIP、檔案，或本輪最新 assistant 交付之程式碼、patch、修補 ZIP；後出現者即為當前基準。
2. 每次開始前，必須先回報當前工作基準，並明確回報已讀 `/doc/PROJECT_SETTINGS.md` 與 `/doc/TEST_SUITE_CHECKLIST.md`；若當前工作基準為 ZIP，另須回報 ZIP 檔名、SHA256 與全新解壓目錄。

## B. 標準測試流程

1. `apps/test_suite.py` 為所有已實作測試的單一正式入口，僅限在本地端執行；GPT 端不得重覆執行 `apps/test_suite.py` 已涵蓋項目、不得執行任何動態測試，且不得繞過正式入口直接執行其涵蓋的 formal step、validator、腳本或函式。
2. `tools/local_regression/formal_pipeline.py` 為單一真理來源。
3. 每輪檢查開始前，GPT 端必須先檢查目前未列在 `doc/TEST_SUITE_CHECKLIST.md`、但必須被`apps/test_suite.py`涵蓋的缺口。
4. 缺口包含已定義在專案設定中規則、也包含未定義在專案設定中但應考慮的規則。
5. 如發現缺口，必須更新 `doc/TEST_SUITE_CHECKLIST.md`，並先提供 `apps/test_suite.py` 補全。
6. 如果確認沒有再發現額外缺口，也須回報`apps/test_suite.py`已涵蓋完整專案測試需求。
7. 檢查到問題就直接在本輪提供修改。
8. 修改問題後，也提出避免 GPT再發生類似問題的作法，透過 `/doc/PROJECT_SETTINGS.md` 或 `/doc/TEST_SUITE_CHECKLIST.md`來強制。
9. 每輪更新完，要同步更新 `doc/TEST_SUITE_CHECKLIST.md` 狀態；主表狀態為唯一真理來源，`E` 僅作同輪無法一次清空時的未完成暫存索引，`T` / `G` 僅為同步索引；主表追蹤 ID 與建議測試項追蹤 ID 的實際格式與欄位限制一律定義於 `doc/TEST_SUITE_CHECKLIST.md`。
10. 文件分工固定：PROJECT_SETTINGS.md 管原則與權責，TEST_SUITE_CHECKLIST.md 管測試清單、狀態、映射、收斂紀錄與欄位細則。除必要短引用外，同一要求只能保留一份原文。
11. checklist 只保留可機械比對的必要資訊；摘要區只保留最小必要索引，時間軸與完成日期只記於收斂紀錄，不得保留無關敘事。追蹤 ID 必須穩定；若更名，必須同輪同步更新 checklist、parser、guard 與正式入口摘要。
12. 提交前必須將checklist主表與G表依日期、ID重排。
13. 違反本節任一前置順序、禁止事項、文件分工或邊界限制，均視為違規。

## C. 回覆、交付與輸出

1. 只提供客觀分析與建議，不奉承；回答必須明確、精簡、避免重複。
2. AI 註解與使用者註解必須明確區分；只能刪 AI 註解，不可刪使用者註解，不可擅改或刪除原本正確的程式碼。
3. 提供程式碼片段時，必須先給可直接搜尋定位的完整舊片段，再給可直接貼上的新片段。
4. 提供修補 ZIP 時，只能包含有修改的檔案，並維持目錄架構，讓使用者可以直接在root貼上取代舊檔，並列出修改了哪些檔案。
5. 預設以提供 ZIP為主，除非使用者要求提供程式碼片段，或只需要改一小段。
6. 如架構調整需刪檔，須提供可執行的 command，避免使用者手動刪錯。
7. `outputs/` 根目錄只放工具分類資料夾；各工具輸出必須落到各自資料夾，禁止再把檔案散落到 `outputs/` 根目錄。
8. 修改 `/doc/PROJECT_SETTINGS.md` 時，必須以 assistant 可明確遵守、使用者易讀易維護、兼顧後續泛用性為原則；應保持文字精簡、嚴禁重複條文。

## D. Coding 與架構原則

1. 單一真理來源：相同邏輯不得重複實作。
2. 統計口徑必須完全一致：成交、未成交、miss buy、EV、勝率、Round-Trip PnL 不得分叉。
3. 避免 magic number；公式與參數必須可解釋。
4. 禁止裸 except；所有被捕捉且非純 control-flow / feature probe 的異常都必須可追蹤，連 optional dependency / GUI fallback 也不得 silent swallow。
5. 架構調整不得明顯犧牲效率；若提高未來策略修改或 ML / DRL / LLM 升級複雜度，必須先明確說明。
6. 正式入口集中於 `apps/`；`core/` 只放核心規則與共用計算；`tools/` 只放驗證、除錯與開發輔助工具。
7. 拆分、合併、移動、重新命名檔案時，必須遵守：單一職責、上層呼叫下層、禁止反向依賴、禁止循環依賴、禁止規則分叉、禁止重複實作。
8. 架構或模組責任有變動時，必須同步更新 `doc/ARCHITECTURE.md` 與 `doc/CMD.md`。
9. 凡新增、刪除、調整 test suite 項目、優先級、狀態，或變更測試分層與維護原則時，必須同步更新 `doc/TEST_SUITE_CHECKLIST.md`；若影響模組責任或測試入口，再同步更新 `doc/ARCHITECTURE.md` 與 `doc/CMD.md`。
10. 正式帳務中的 `cash`、`pnl`、`equity`、`reserved_cost`、`risk` 必須以整筆總額 ledger 的單一真理來源計算；不得以含費每股價或 per-share × qty 回推正式總額。
11. 凡屬買入、停損、停利、漲跌停與一字鎖死等交易狀態判斷，必須經共享價格正規化 helper；不得以 raw float equality 或未正規化浮點值直接判定。
12. 凡變更 `run_v16_backtest`、scanner reference 或其他跨工具共用 public payload / stats key 時，必須同輪檢索全部 consumer 與 formal contract，維持既有欄位相容或同步改完所有 caller；不得只改 producer。
13. 單股 backtest 與 debug 路徑中的 `currentCapital` 一律代表可用現金；任何半倉、全倉或期末結算只能加回實際 `net sell total`，不得只加 `realized pnl`；持倉中的 `currentEquity` 一律以 `cash + 當前可變現淨值` 計。
14. debug / GUI / 單股模擬凡重播 entry/exit 現金路徑時，買進當下也必須同步扣除實際 `net buy total`；不得只在賣出時更新現金，否則 sizing、trade log 與 completed-trade 重建都會分叉。
15. 凡屬可見 trade log / history row 的金額欄位，若後續工具會以其重建 completed trades 或彙總損益，必須統一使用共享 money-rounding helper；不得在各模組散落內建 `round(..., 2)` 形成 0.01 級顯示口徑分叉。
16. 凡 validator / consumer 以可見 completed-trade 或 standalone log 的 rounded 金額作 expected oracle 時，也必須使用共享 money-rounding helper；不得混用內建 `round(..., 2)` 與 HALF_UP，否則會把顯示 rounding 口徑誤判成核心不一致。
17. 凡以可見 trade log / history row 重建 completed trades 的 helper / consumer，也必須以共享 money-rounding helper 正規化逐列金額並累加；不得在重建路徑另用內建 `round(..., 2)`，否則會把顯示值重建成不同於核心 completed trades 的序列。
18. debug / GUI / history 的期末強制結算若需合成整筆 completed-trade `total_pnl`，必須以 `realized_pnl_milli + final_leg_pnl_milli` 走整數 ledger 路徑後再轉顯示；不得以 float 將既有已實現損益與尾倉損益直接相加。
19. `tools/validate/synthetic_meta_cases.py` 內任何 validator 若以 `summary["source_path"]` 或類似欄位回報來源檔，所使用的 path 變數必須在該函式內明確宣告；不得引用其他 validator 的局部變數名稱，避免 synthetic suite 因 `NameError` 中斷。
20. 凡 unit / synthetic validator 需要比對顯示金額、rounded leg pnl 或 completed-trade 顯示總損益時，也必須使用共享 money-rounding helper；不得在 validator 內另用內建 `round(..., 2)` 產生與正式顯示口徑不同的 oracle。
21. debug / GUI / history 的可見交易明細或 marker 若要顯示 `buy_capital`、`sell_capital`、`gross_amount`、`total_return_pct` 等正式交易總額或報酬率，必須以 exact ledger total（如 `net_buy_total_milli`、`net_total_milli`）推導；不得再以含費每股顯示價或 `per-share × qty` 回推可見總額。
22. debug / GUI / history 若已有 `entry_capital_total`、`net_buy_total_milli` 等既存總額欄位可用，計算 `total_return_pct` 或其他以 full-entry capital 為分母的可見報酬率時，必須優先使用該總額欄位；不得跳過既有 total 而直接退回 `entry * qty` 等 per-share fallback。
23. debug / GUI / history 的買進可見欄位或 marker 若顯示 `buy_capital`、買進 `gross_amount` 或其他 entry total，必須優先使用 `net_buy_total_milli`、`entry_cost` 或共享 exact-total helper；不得以 `entry_price * qty`、`entry * qty` 等 per-share fallback 回推買進總額。
24. debug / GUI / history 若顯示半倉停利或其他單腿 exit 的可見 `pnl_pct` / 報酬率，必須優先以 exact ledger 的 `allocated_cost_milli` 與該腿 `pnl_milli` 計算；不得以 `(net_price - entry) / entry` 等 per-share 浮點差價公式回推單腿報酬率。

## E. 交易與策略原則

1. 杜絕未來函數：不可偷看未來資料。
2. 同 K 棒同時觸發停損 / 停利時，一律以最壞停損情況計算。
3. 資金與權益曲線一律以扣除手續費、稅金後的淨值為準。
4. 半倉停利僅視為現金回收，尾倉結算才算完整 Round-Trip PnL。
5. 除「只能盤前掛單」與「持倉後才能設定 / 執行停損停利」這兩項最小物理限制外，不得另加非最小、非物理且非必要的交易限制；若既有規則、延伸規格或測試契約與此原則衝突，一律以口徑一致、符合物理意義與真實可執行性為最高優先考量。
6. 只能盤前掛單；盤中不得新增、改單、換股。
7. 不得當沖；買入當日不可賣出；當日賣出回收資金不得於當日再投入。
8. 若某檔當日未成交，不得於同一交易日盤中自動改掛其他股票。
9. `L` 只作盤前最高可接受買入價、資金占用估算與最壞情境風險 sizing 上界；不得再兼作首個可執行停損 / 停利基準。
10. 首個可執行停損 / 停利，只能在 `t+1` 實際成交後、於 `t+1` 收盤後 frozen；其基準為 `P_fill` 與盤前已知波動尺度，預設使用 `ATR_t`，並自 `t+2` 起生效。
11. `t+1` 買入當日雖不得賣出，仍必須判斷 stop / tp 是否已被觸發；觸發與執行必須分離，`t+1` 只記 trigger，`t+2` 起於第一個可執行時點強制執行，不得要求再次碰價才執行。
12. `t+1` 觸發停損者，`t+2` 起應優先出清；`t+1` 觸發停利者，`t+2` 起應依既定股數減倉。觸發價、觸發日、執行價、執行日必須分開記錄，不得把 `t+1` 觸發價當成交價。
13. 單筆理論風險上限若需嚴格限制，盤前股數必須以 `L` 對應之最壞可成交情境、含費稅淨損失與 `floor` sizing 計算；該 sizing 基準只用於股數與風險上限控制，不得衍生 `P_fill <= candidate_plan.init_sl` 等成交否決下限。實際成交後之 stop / tp 可依 `P_fill` 定義，但不得回頭放寬事前風險上限。
14. 未成交延續候選不得預先具有可執行停損 / 停利；只有當首個待掛單日曾進入可買區時，才可用當日反事實進場價 `P' = min(Open, L)` 建立固定的失效 / 達標 barrier。該 barrier 只用於延續候選存續判斷，不得隨日漂移，也不得冒充已持倉 stop / tp。
15. 凡屬買入、停損、停利、失效或達標線之 hit 判斷，長倉一律採 inclusive 口徑：`Low <= line`、`High >= line`；`>` / `<` 僅用於 breakout、cross 或啟動條件，不得與 hit 判斷混用。
16. `trailing_stop` 只屬持倉後動態保護，不得回寫或覆蓋首個可執行停損；候選資格、是否掛單、是否成交、是否 miss buy、stop / tp 觸發、實際執行與歷史績效統計，必須分層定義，不得混用。



## F. 專案特例

1. `apps/portfolio_sim.py` 自動開瀏覽器暫時允許。
2. 暫時只使用還原價，不考慮 raw。
3. `doc/ToDo.md` 是使用者自已看的備忘錄，檢查時不要用考慮。
4. 使用者註解不需要檢查。
# 專案設定

1. 每輪開始前必須先讀取並遵守 `/doc/PROJECT_SETTINGS.md`、`/doc/TEST_SUITE_CHECKLIST.md` 與 `/doc/GPT_DELIVERY_CHECKLIST.md`；其中 `PROJECT_SETTINGS.md` 為最高優先規則，`TEST_SUITE_CHECKLIST.md` 為本地端 formal test suite 收斂與維護清單，`GPT_DELIVERY_CHECKLIST.md` 為 GPT 交付前操作檢查表；三者均不得忽略、弱化、選擇性遵守、以慣例覆蓋或自行推定例外。
2. 文件治理與承載邊界依 A3–A4 執行；formal 流程與同步要求依 B 節執行。

## A. 工作基準與文件分工

1. 本輪基準為使用者最新提供的程式、ZIP、檔案，或本輪最新 assistant 交付之程式碼、patch、修補 ZIP；後出現者即為當前基準。
2. 每次開始前，必須先回報當前工作基準與已讀文件；具體回報欄位依 `doc/GPT_DELIVERY_CHECKLIST.md` 執行。
3. 文件分工固定：`PROJECT_SETTINGS.md` 只保留上位原則、模組責任、邊界與資料流；`TEST_SUITE_CHECKLIST.md` 只保留 formal 主表、狀態、測試入口與收斂索引；`GPT_DELIVERY_CHECKLIST.md` 只保留 GPT 交付前操作檢查，不作 formal 主表、狀態真理來源或 validator 被測內容。
4. 文件承載邊界固定：`PROJECT_SETTINGS.md` 只保留長期穩定、可泛化原則；高波動 wording、暫時事故補丁、validator 內部命名、單次案例 hygiene 與細部驗證技巧，不得上升到 `PROJECT_SETTINGS.md` 或 formal 主表。formal test suite 只驗 shipped 正式介面、schema、檔案樹、正式入口與行為；GPT 操作習慣只留在 `GPT_DELIVERY_CHECKLIST.md`。

## B. 標準測試流程

1. `apps/test_suite.py` 為所有已實作測試的單一正式入口，僅限在本地端執行；GPT 端不得重覆執行其已涵蓋項目、不得執行任何動態測試，也不得繞過正式入口直接執行其涵蓋的 formal step、validator、腳本或函式。
2. `tools/local_regression/formal_pipeline.py` 為單一真理來源。
3. 每輪開始前，必須先檢查目前是否存在尚未列入 `doc/TEST_SUITE_CHECKLIST.md`、但應由正式入口涵蓋的缺口；若有缺口，先更新 `doc/TEST_SUITE_CHECKLIST.md` 與正式入口，再處理其他問題。
4. 若使用者未提供 bundle，視為已在本地完成 `apps/test_suite.py` 且結果全過；若提供 bundle，必須逐條對照 bundle 原始失敗項完成閉環修正；未消除原始失敗項前，不得以相鄰文件、註解、help 或 checklist 已同步視為修復完成。
5. 檢查到問題就直接在本輪提供修改；若變更 formal chain，必須同輪完成 definition、import、registry、`doc/TEST_SUITE_CHECKLIST.md`、parser、guard、正式入口、help 與對應 meta guard 的全鏈同步。
6. 若前一輪修改在本輪仍被 bundle 或再檢查證明有錯，除修正原始失敗外，必須同步更新 `doc/GPT_DELIVERY_CHECKLIST.md`，把防再犯要求提升為可泛化、可操作的交付前檢查；不得只補單一案例、單一字串或局部實作特例。
7. 一般註解（含 AI 註解、程式內摘要註解與未被正式介面直接讀取的 docstring）預設不納入 GPT 最嚴格檢查、本地 formal test suite、bundle 與 `apps/test_suite.py` 驗證範圍；但若該文字會被 parser、`--help`、UI、report、export、bundle 或 formal contract 直接讀取／輸出，則視為正式介面。
8. 凡修改 `doc/TEST_SUITE_CHECKLIST.md` 的主表、`T`、`G`、`E` 等機械排序區塊，必須維持既有排序 guard 可通過；具體交付前重排與核對步驟依 `doc/GPT_DELIVERY_CHECKLIST.md` 執行。
9. 執行最嚴格檢查或再檢查時，必須以同輪一次找出並修正所有目前可發現的問題為原則；不得將同源、同鏈或同契約的已知相鄰問題拆成多輪逐步釋出。若存在本輪無法清除的阻塞，不得將局部修補視為完成。

## C. 回覆、交付與輸出

1. 只提供客觀分析與建議，不奉承；回答必須明確、精簡、避免重複。
2. AI 註解與使用者註解必須明確區分；只能刪 AI 註解，不可刪使用者註解，不可擅改或刪除原本正確的程式碼。
3. 提供程式碼片段時，必須先給可直接搜尋定位的完整舊片段，再給可直接貼上的新片段。
4. 提供修補 ZIP 時，只能包含有修改的檔案，並維持目錄架構，讓使用者可以直接在 root 貼上取代舊檔，並列出修改了哪些檔案。
5. 預設以提供 ZIP 為主，除非使用者要求提供程式碼片段，或只需要改一小段。
6. 如架構調整需刪檔，須提供可執行的 command，避免使用者手動刪錯。
7. `outputs/` 根目錄只放工具分類資料夾；各工具輸出必須落到各自資料夾，禁止再把檔案散落到 `outputs/` 根目錄。
8. 交付前必須完成 GPT 端自檢；具體檢查與交付條件依 `doc/GPT_DELIVERY_CHECKLIST.md` 執行。

## D. Coding 與架構原則

1. 單一真理來源：相同邏輯不得重複實作，核心規則、帳務、價格與統計的衍生結果不得分叉。
2. 統計口徑必須完全一致：成交、未成交、miss buy、EV、勝率與 Round-Trip PnL 不得因路徑、顯示或用途不同而改變定義。
3. 避免 magic number；公式、參數與比較規則必須可解釋。
4. 禁止裸 except；所有被捕捉且非純 control-flow / feature probe 的異常都必須可追蹤。
5. 架構調整不得明顯犧牲效率；若提高未來策略修改或 ML / DRL / LLM 升級複雜度，必須先明確說明。
6. 正式入口集中於 `apps/`；`core/` 只放核心規則與共用計算；`tools/` 只放驗證、除錯與開發輔助工具。
7. 拆分、合併、移動或重新命名檔案時，必須遵守單一職責、分層呼叫、禁止反向依賴、禁止循環依賴、禁止規則分叉與禁止重複實作。
8. 架構、模組責任、正式入口、共享 schema、canonical 名稱或共享資料流有變動時，實作、文件、registry、validator 與正式輸出必須同輪同步；顯示、除錯、fallback、preview、reporting 等輔助路徑不得重建、猜測、省略、改名或覆寫正式共享上下文。
9. validator、oracle 與 meta guard 只驗 invariant、契約、角色分離與同步完整性，不重寫分叉實作；rename 後一律以 canonical 名稱為準，legacy alias 僅作相容邊界。

## E. 交易與策略原則

1. 杜絕未來函數：任何候選、掛單、成交、停損、停利、延續判斷與統計，都不得偷看當下尚未知的未來資料。
2. 同一事件的判斷、觸發、執行與統計口徑必須一致；不確定時一律採最保守、最不利於績效的可執行解讀。
3. 資金、權益、PnL、報酬率、勝率、EV 與 Round-Trip 定義，必須以扣除手續費、稅金後的淨值為準，且不得因半倉、顯示或報表需求分叉。
4. 交易限制只保留真實可執行、具物理意義且口徑一致的最小必要約束；不得任意疊加與實盤不相稱的額外限制。
5. 掛單、成交、觸發、執行、失效、達標與結算必須分層定義；價格、日期、股數、現金回收與完整交易結果不得混用或提前認列。
6. 風險 sizing、停損/停利、trailing stop、延續候選與 barrier 的關係，必須遵守單一真理來源與角色分離；候選規則不得冒充已持倉規則，保護性機制不得回寫成交前基準。
7. hit 判斷、觸發紀錄、執行條件與統計口徑必須在全專案保持一致；不得依工具、顯示路徑或資料流向切換比較符號或語意。
8. 單筆風險上限、候選存續、miss buy、rotation、forced exit 與同日資金可用性，必須統一服從相同的交易日序、可執行邊界與資金回收規則。

## F. 專案特例

1. `apps/portfolio_sim.py` 自動開瀏覽器暫時允許。
2. 暫時只使用還原價，不考慮 raw。
3. `doc/ToDo.md` 與一般使用者註解屬使用者自有備忘／說明，不納入 formal / GPT 最嚴格檢查；但若其文字被正式介面直接讀取，仍視為正式輸出。

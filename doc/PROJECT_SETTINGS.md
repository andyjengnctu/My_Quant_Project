# 專案設定

1. 每輪開始前必須先讀取並遵守 `/doc/PROJECT_SETTINGS.md` 與 `/doc/TEST_SUITE_CHECKLIST.md`；其中 `PROJECT_SETTINGS.md` 為最高優先規則，`TEST_SUITE_CHECKLIST.md` 為 test suite 收斂與維護清單；兩者均不得忽略、弱化、選擇性遵守、以慣例覆蓋、或自行推定例外。
2.  `PROJECT_SETTINGS.md` 與`TEST_SUITE_CHECKLIST.md`更新原則: `PROJECT_SETTINGS.md` 只保留原則性條款; `/doc/TEST_SUITE_CHECKLIST.md` 只保留可機械比對的必要資訊; 兩者不可違反單一真理原則; 各文件內也不可有任何重覆性描述，保持文件至最精簡。


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
11. 提交前必須將checklist主表與G表依日期、ID重排; 摘要區只保留最小必要索引，時間軸與完成日期只記於收斂紀錄，不得保留無關敘事。追蹤 ID 必須穩定；若更名，必須同輪同步更新 checklist、parser、guard 與正式入口摘要。
12. 如果使用者沒有提供 boundle結果，代表已在本地端執行過`apps/test_suite.py`，並且結果為All Passed; 如果使用者提供bBoudle結果，GPT 必須修正錯誤。
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
9. 提供patch前必需在GPT端做過最嚴格檢查，如確認沒有問題才提供。

## D. Coding 與架構原則

D 節只保留跨模組、長期穩定的原則；會隨實作演進而調整的細部 contract、欄位、helper、signature、rounding、fallback 與 formal guard，一律下沉到 `doc/TEST_SUITE_CHECKLIST.md` 維護，不在本檔重複展開。

1. 單一真理來源：相同邏輯不得重複實作。
2. 統計口徑必須完全一致：成交、未成交、miss buy、EV、勝率、Round-Trip PnL 不得分叉。
3. 避免 magic number；公式與參數必須可解釋。
4. 禁止裸 except；所有被捕捉且非純 control-flow / feature probe 的異常都必須可追蹤。
5. 架構調整不得明顯犧牲效率；若提高未來策略修改或 ML / DRL / LLM 升級複雜度，必須先明確說明。
6. 正式入口集中於 `apps/`；`core/` 只放核心規則與共用計算；`tools/` 只放驗證、除錯與開發輔助工具。
7. 拆分、合併、移動、重新命名檔案時，必須遵守：單一職責、上層呼叫下層、禁止反向依賴、禁止循環依賴、禁止規則分叉、禁止重複實作。
8. 架構或模組責任有變動時，必須同步更新 `doc/ARCHITECTURE.md` 與 `doc/CMD.md`。
9. 凡新增、刪除、調整 test suite 項目、優先級、狀態，或變更測試分層與維護原則時，必須同步更新 `doc/TEST_SUITE_CHECKLIST.md`；若影響模組責任或測試入口，再同步更新 `doc/ARCHITECTURE.md` 與 `doc/CMD.md`。
10. 正式帳務、價格正規化、費稅、sizing 與可見統計，必須共享同一套核心規則與資料來源；任何顯示、重建、排序、比較或 validator oracle 都不得另建分叉公式。
11. 涉及商品別、tick、漲跌停、費稅、日期、ticker、security profile、position 等上下文的跨模組規則，必須由共享 helper 與明確參數傳遞維持一致；不得依賴自由變數、隱式 fallback 或模組私有特例。
12. 凡 shared helper、public payload、stats schema、keyword 參數或跨工具契約有變動，必須同輪同步更新 producer、caller、consumer、formal contract 與相關文件；不得只改其中一側。
13. debug / GUI / history / reporting / scanner / validator / local regression 等非核心路徑，仍必須服從與正式交易邏輯相同的單一真理來源；不得因 fallback、顯示需求或 optional dependency 而繞開核心規則。
14. validator 與 oracle 必須優先驗證 invariant、契約與單一真理來源，不得重寫一份分叉實作；若需依賴可變狀態，必須先 snapshot 再比對。

## E. 交易與策略原則

E 節只保留交易決策、執行邊界與統計口徑的長期原則；會隨策略與 formal contract 演進而調整的細部交易規則、日序、barrier、hit 判斷、延續候選、價格基準、觸發/執行欄位與驗證細節，一律下沉到 `doc/TEST_SUITE_CHECKLIST.md` 維護，不在本檔重複展開。

1. 杜絕未來函數：任何候選、掛單、成交、停損、停利、延續判斷與統計，都不得偷看當下尚未知的未來資料。
2. 同一事件的判斷與執行口徑必須一致，且在不確定時一律採最保守、最不利於績效的可執行解讀。
3. 資金、權益、PnL、報酬率、勝率、EV 與 Round-Trip 定義，必須以扣除手續費、稅金後的淨值為準，且不得因半倉、顯示或報表需求分叉。
4. 交易限制只保留真實可執行、具物理意義且口徑一致的最小必要約束；不得任意疊加與實盤不相稱的額外限制。
5. 掛單、成交、觸發、執行、失效、達標與結算必須分層定義；價格、日期、股數、現金回收與完整交易結果不得混用或提前認列。
6. 風險 sizing、停損/停利、trailing stop、延續候選與 barrier 的關係，必須遵守單一真理來源與角色分離；候選規則不得冒充已持倉規則，保護性機制不得回寫成交前基準。
7. hit 判斷、觸發紀錄、執行條件與統計口徑必須在全專案保持一致；不得依工具、顯示路徑或資料流向切換比較符號或語意。
8. 單筆風險上限、候選存續、miss buy、rotation、forced exit 與同日資金可用性，必須統一服從相同的交易日序、可執行邊界與資金回收規則。
9. checklist 負責維護 E 節細部交易 contract、formal guard、狀態與時間軸；本檔只保留原則，不再承載會頻繁變動的細項清單。


## F. 專案特例

1. `apps/portfolio_sim.py` 自動開瀏覽器暫時允許。
2. 暫時只使用還原價，不考慮 raw。
3. `doc/ToDo.md` 是使用者自已看的備忘錄，檢查時不要用考慮。
4. 使用者註解不需要檢查。

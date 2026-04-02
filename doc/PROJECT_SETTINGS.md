# 專案設定

1. 每輪開始前必須先讀取並遵守 `/doc/PROJECT_SETTINGS.md` 與 `/doc/TEST_SUITE_CHECKLIST.md`；其中 `PROJECT_SETTINGS.md` 為最高優先規則，`TEST_SUITE_CHECKLIST.md` 為 test suite 收斂與維護清單；兩者均不得忽略、弱化、選擇性遵守、以慣例覆蓋、或自行推定例外。

## A. 工作基準與執行紀律

1. 本輪基準為使用者最新提供的程式、ZIP、檔案，或本輪最新 assistant 交付之程式碼、patch、修補 ZIP；後出現者即為當前基準。
2. 每次開始前，必須先回報當前工作基準，並明確回報已讀 `/doc/PROJECT_SETTINGS.md` 與 `/doc/TEST_SUITE_CHECKLIST.md`；若當前工作基準為 ZIP，另須回報 ZIP 檔名、SHA256 與全新解壓目錄。

## B. 標準測試流程

1. 使用者要求「完整檢查」時，GPT 端必須先做 test suite / checklist sufficiency review；不得只檢查 checklist 是否有非 `DONE` 項目，還必須主動檢查 test suite 是否仍有缺口但尚未列入 `doc/TEST_SUITE_CHECKLIST.md`。
2. sufficiency review 的檢查範圍，至少包含正式入口、步驟註冊、自我驗證、coverage guard、artifact / summary contract、文件同步、收斂紀錄同步，以及 checklist 自身是否有缺口、遺漏回寫、狀態過舊或摘要失同步。
3. 若發現 test suite 有未列入 checklist 的缺口，或 checklist 自身有缺口、遺漏回寫、狀態過舊、摘要失同步，須先明確指出，並先更新 `doc/TEST_SUITE_CHECKLIST.md`，再進入後續驗證或修改。
4. 在 test suite 完全收斂前，GPT 端的首要任務是補齊 test suite 與 checklist，使本地端正式驗證可獨立完成；GPT 端僅可在收斂過程中做必要的定向補驗，用於辨識未覆蓋缺口、驗證收斂修補方向，不得把 GPT 補驗當成常態驗證機制。
5. 已標記為 `DONE` 的項目，原則上以 test suite 為主，GPT 端不重複完整執行；只有在本輪改動直接影響相關模組、測試入口、輸出契約、架構責任，或 sufficiency review 發現可疑缺口 / 症狀時，才做定向複核。
6. 只有在前述判定確實需要執行正式步驟時，才套用以下執行規則：
   - 若本輪基準為 ZIP，先做環境 bootstrap 與 preflight。
   - 測試資料使用 `data/tw_stock_data_vip_reduced`。
   - `apps/test_suite.py` 為所有已實作測試的單一正式入口。
   - reduced local regression 的正式組成步驟定義，以 `tools/local_regression/formal_pipeline.py` 為單一真理來源。
   - GPT 端可依本輪影響面拆開執行正式組成步驟，但不得與 `tools/local_regression/formal_pipeline.py` 與單一正式入口定義失同步。

## C. 回覆、交付與輸出

1. 只提供客觀分析與建議，不奉承；回答必須明確、精簡、避免重複。
2. AI 註解與使用者註解必須明確區分；只能刪 AI 註解，不可刪使用者註解，不可擅改或刪除原本正確的程式碼。
3. 提供程式碼片段時，必須先給可直接搜尋定位的完整舊片段，再給可直接貼上的新片段。
4. 提供修補 ZIP 時，只能包含有修改的檔案，並維持目錄架構，讓使用者可以直接在root貼上取代舊檔，並列出修改了哪些檔案。
5. 如架構調整需刪檔，須提供可執行的 command，避免使用者手動刪錯。
6. `outputs/` 根目錄只放工具分類資料夾；各工具輸出必須落到各自資料夾，禁止再把檔案散落到 `outputs/` 根目錄。
7. 修改 `/doc/PROJECT_SETTINGS.md` 時，必須以 assistant 可明確遵守、使用者易讀易維護、兼顧後續泛用性為原則；應保持文字精簡、嚴禁重複條文。

## D. Coding 與架構原則

1. 單一真理來源：相同邏輯不得重複實作。
2. 統計口徑必須完全一致：成交、未成交、miss buy、EV、勝率、Round-Trip PnL 不得分叉。
3. 避免 magic number；公式與參數必須可解釋。
4. 禁止裸 except；所有異常必須可追蹤。
5. 架構調整不得明顯犧牲效率；若提高未來策略修改或 ML / DRL / LLM 升級複雜度，必須先明確說明。
6. 正式入口集中於 `apps/`；`core/` 只放核心規則與共用計算；`tools/` 只放驗證、除錯與開發輔助工具。
7. 拆分、合併、移動、重新命名檔案時，必須遵守：單一職責、上層呼叫下層、禁止反向依賴、禁止循環依賴、禁止規則分叉、禁止重複實作。
8. 架構或模組責任有變動時，必須同步更新 `doc/ARCHITECTURE.md` 與 `doc/CMD.md`。
9. 凡新增、刪除、調整 test suite 項目、優先級、狀態，或變更測試分層與維護原則時，必須同步更新 `doc/TEST_SUITE_CHECKLIST.md`；若影響模組責任或測試入口，再同步更新 `doc/ARCHITECTURE.md` 與 `doc/CMD.md`。

## E. 交易與策略原則

1. 杜絕未來函數：不可偷看未來資料。
2. 同 K 棒同時觸發停損 / 停利時，一律以最壞停損情況計算。
3. 資金與權益曲線一律以扣除手續費、稅金後的淨值為準。
4. 半倉停利僅視為現金回收，尾倉結算才算完整 Round-Trip PnL。
5. 只能盤前掛單；盤中不得新增、改單、換股。
6. 不得當沖；買入當日不可賣出；當日賣出回收資金不得於當日再投入。
7. 若某檔當日未成交，不得於同一交易日盤中自動改掛其他股票。
8. 停利 / 停損只能對已持有部位預先設定。
9. 候選資格、是否掛單、是否成交、是否 miss buy、歷史績效統計，必須分層定義，不得混用。
10. 單股回測不得使用該檔自身歷史績效 filter 作為買入閘門；歷史績效 filter 只能用於投組層與scanner。

## F. 專案特例

1. `apps/portfolio_sim.py` 自動開瀏覽器暫時允許。
2. 暫時只使用還原價，不考慮 raw。
# 專案設定

1. 每輪開始前必須先讀取並遵守 `/doc/PROJECT_SETTINGS.md`； 為全專案的最上層原則，視為憲法; 條款以長期穩定、可泛化為主; 
2. 使用者要求「完整檢查」，GPT 即強制進入固定檢查矩陣模式，必須以表格逐項列出 (1) 本文件 A 至 E 全部條款、(2)專案內未列出但本輪應被檢查之全部項目，並以「獨立檢查」方式確認所有項目; 不得以任何理由採用非「獨立檢以」的驗證方式; 也不得以任何理由省略、延後、縮減或跳過任一項目的獨立檢查; 在未確認所有項目都完成獨立檢查前，本輪檢查不得停止。



## A. 標準測試流程

1. 本輪基準為使用者最新提供的程式、ZIP、檔案，或本輪最新 GPT 交付之程式碼、patched ZIP；後出現者即為當前基準。
2. 每次開始前，必須先回報當前工作基準與已讀文件；若當前基準為 ZIP，另須回報 ZIP 檔名、SHA256 與全新解壓目錄。
3. GPT 每輪都必須對當前基準做全專案完整檢查，並於同一輪內盡可能找出所有問題、直接提供修正，不得以多輪零碎修補取代完整檢查。
4. `apps/run_bundle.py` 為本地正式 double check 與交付打包的單一使用者入口；預設流程固定為 stage 變更 → commit 當前 snapshot → package ZIP → 執行 `apps/test_suite.py`。formal test 即使 FAIL 仍必須保留先前已建立的 commit 與 ZIP，讓失敗版本可被完整交付與閉環修正；`apps/test_suite.py` 是 `run_bundle.py` 內部正式測試執行器，不再作為後續要求使用者直接執行的主要入口。正式 double check 不取代 GPT 的全專案完整檢查。
5. GPT 不得執行 `apps/run_bundle.py` 或 `apps/test_suite.py`，也不得以重建 formal suite 或直接執行其涵蓋 step 的方式替代本地正式執行；但仍必須以非 formal-suite 的方式獨立檢查整個專案，並檢查 `apps/run_bundle.py`、`apps/test_suite.py` 本身是否可信，以及 `doc/TEST_SUITE_CHECKLIST.md` 是否涵蓋必要測項。
6. 若使用者提供由 `apps/run_bundle.py`／`apps/test_suite.py` 產生的 bundle 測試結果，GPT 必須完成閉環修正；若未提供 bundle，僅代表本地端尚未提供 formal double check 證據。後續正式本機整合測試一律以 `apps/run_bundle.py` 預設流程執行，不使用 `--no-commit`，除非使用者當輪明確要求。
7. `/doc/PROJECT_SETTINGS.md` 不得被 `apps/run_bundle.py`／`apps/test_suite.py` 反向檢查。
8. 每輪檢查結束後，如有發現問題，就在本輪提供修改，不用再詢問使用者。


## B. 回覆、交付與輸出

1. 只提供客觀分析與建議，不奉承；
2. 回覆必須講重點、明確、精準，不得將一樣的概念分不同層次重複說明。
3. AI 註解與使用者註解必須明確區分；只能刪 AI 註解，不可刪使用者註解，不可擅改或刪除原本正確的程式碼。
4. 提供程式碼片段時，必須先給可直接搜尋定位的完整舊片段，再給可直接貼上的新片段。
5. 提供修補 ZIP 時，只能包含有修改的檔案，並維持目錄架構，讓使用者可以直接在 root 貼上取代舊檔，並列出修改了哪些檔案。
6. 預設以提供 ZIP 為主，除非使用者要求提供程式碼片段，或只需要改一小段。
7. 如架構調整需刪檔，須提供可執行的 command，避免使用者手動刪錯。
8. `outputs/` 根目錄只放工具分類資料夾；各工具輸出必須落到各自資料夾，禁止再把檔案散落到 `outputs/` 根目錄。
9. 交付前必須完成 GPT 端自檢；只核對 blocker、正式輸出、正式契約、必要文件同步與本輪修改涉及的同鏈項；不得把 validator 寫法 hygiene、單次 wording 或事故級 patch 規則再疊成第二層檢查。
10. 凡修改 `doc/TEST_SUITE_CHECKLIST.md` 的主表、`T`、`G`、`E` 等機械排序區塊，必須維持既有排序 guard 可通過；交付前須整表核對排序、摘要、最新狀態與 transition 連續性；若插回既有日期區塊或補寫 `DONE -> N/A` / `DONE -> PARTIAL`，必須重排整個受影響同日區塊；若修改 markdown table 列內容，須逐列核對欄數符合 header，且表格 cell 內不得保留未轉寫的裸 `|` 字元。
11. 對研究、改善與除錯問題，回覆與實驗計畫必須以可立即執行的分析、實作或測試為主；除非使用者明確要求，不得把等待未來資料、等待市場自然累積、凍結研究或暫停行動當成主要下一步。資料有限時，應優先在既有資料下設計無前視、Train／Validation 隔離、消融、Rolling、錯誤歸因或其他可執行方案。
12. 所有使用者可見的檔案／資料夾路徑，預設必須以專案根目錄為基準顯示穩定相對路徑，並統一使用 `/` 分隔；不得在一般 console、摘要或工件清單中輸出本機絕對路徑。Manifest、hash identity、runtime 讀取與外部明確指定路徑可保留 canonical path，但不得改變使用者可見顯示規則。
13. GPT回報執行方式時，若正式互動選單已有對應入口，必須以選單操作步驟為主；只有尚未納入正式選單的臨時診斷或研究CLI，才提供直接指令。
14. 正式功能已有互動選單時，正式操作流程不得要求使用者先執行可由程式確定完成的零散CLI；實驗對象、開關、工件來源、前置建立政策與訓練參數必須集中於`config/`，方便使用者檢視與自行調整，不得硬編碼於選單。
15. 正式App執行前必須建立工件依賴計畫；對於可由既有真理工件確定性產生、或已有正式builder的缺少／過期工件，應依config自動重用、建立、重建或接續。只有涉及新Label、新模型訓練、研究選擇或缺少上游真理工件時，才能停止並導向對應正式入口。
16. Breakout Quality 的策略層正式結果輸出（包含 Strategy Compare、策略參數適應與 strategy gate 類流程）只要產生持久工件，就必須同時提供可直接閱讀的簡易報表：互動執行時輸出console摘要，並保存Markdown報表；RUN與REUSE／cache重用必須使用同一份canonical結果與renderer，不能因重用而只剩JSON／CSV。JSON／CSV／manifest等詳細工件由同一份結果補充即可，不要求每個支援檔再複製一份簡易報表。
17. `apps/research.py`的模型訓練工作類型及其model provider正式研究／驗證輸出必須有App層簡易報表：成功完成的子命令在既有詳細console／JSON／CSV／Markdown工件之外，固定再輸出短版console摘要並保存`outputs/filters/breakout_quality/<filter_id>/simple_reports/<command>.md`；摘要只讀取既有canonical工件補充核心指標，不得另算第二套模型／PIT指標。完整workflow必須另有最終總結；Audit選單執行完成後必須直接顯示最近Audit摘要。
18. 所有互動選單的項目文字、可用項目與顯示順序不得硬編碼特定實驗、模型、候選、策略arm、ID或目前設定值。選單只能以泛化工作類型作固定文字；若需顯示目前比較／執行對象，必須由`config/`、Registry或active settings動態產生。可選對象、reference、比較組合與摘要對象亦必須由設定驅動；validator不得把當前config值或特定ID字串固定成唯一合法選單輸出。


## C. Coding 與架構原則

1. 單一真理來源：相同邏輯不得重複實作，核心規則、帳務、價格與統計的衍生結果不得分叉。
2. 統計口徑必須完全一致：成交、未成交、miss buy、EV、勝率與 Round-Trip PnL 不得因路徑、顯示或用途不同而改變定義。
3. 避免 magic number；公式、參數與比較規則必須可解釋。
4. 禁止裸 except；所有被捕捉且非純 control-flow / feature probe 的異常都必須可追蹤。
5. 架構調整不得明顯犧牲效率；若提高未來策略修改或 ML / DRL / LLM 升級複雜度，必須先明確說明。
6. 正式入口集中於 `apps/`；`core/` 只放核心規則與共用計算；`tools/` 只放驗證、除錯與開發輔助工具。
7. 拆分、合併、移動或重新命名檔案時，必須遵守單一職責、分層呼叫、禁止反向依賴、禁止循環依賴、禁止規則分叉與禁止重複實作。
8. `config/` 下的正式參數均屬使用者可自行調整的設定；GPT、validator 與 formal suite 不得把任何目前值、預設值或特定值硬編碼成唯一合法答案，也不得為了讓測試通過而擅自修改、覆寫、還原或限制使用者設定。測試僅可驗證欄位存在、型別、合法範圍、跨欄一致性、衍生結果與 runtime 是否忠實採用當前設定；若測試需要固定案例，必須在測試內使用隔離 override，不得限制實際 `config/`。此規則包含但不限於 `training_policy`、`search_space` 與 `display_policy`。
9. 研究工作可由單一`apps/research.py`作為使用者正式入口，但模型訓練、策略參數最佳化與策略組合比較必須維持獨立工作類型與application/service責任分離；一般策略組合比較不得自行建立Label、選擇模型或訓練模型權重，但可依config透過正式共用服務匯出既有模型的推論工件，以及建立比較所需的策略參數工件。唯一明確例外為`Multiple-seed robustness`：因其scientific variable本身就是模型訓練seed、最終判讀對象是策略績效，可在`[3] 策略組合比較`下作獨立robustness orchestrator，且只能呼叫canonical model-training service建立隔離、暫存的per-seed權重／score後立即進同一策略replay；不得建立新Label、改target／architecture／loss／hyperparameter、覆寫canonical模型、挑best seed、做seed ensemble或把seed工件升級成正式模型。各工作類型不得複製核心訓練邏輯；特定model、實驗標的、比較arm與Audit module不得由選單選擇，必須由`config/`指定。
10. 前置工件自動建立必須先顯示可稽核計畫並只確認一次；任一步驟失敗時立即停止後續回放、保留可接續工件、回報失敗步驟與相對路徑，不得產生不完整的正式比較報表。
11. 臨時性研究／Audit／診斷／synthetic test 程式屬可拋棄工程資產；一旦其待決策問題已有結論、對應實驗已記錄於Registry／Log，且目前正式runtime、active Audit或必要compatibility不再依賴，就必須刪除implementation、config／catalog registration與只服務該程式的tests／helpers，不得以永久`enabled=False`、historical CLI或只為coverage保留的方式累積。
12. `meta quality`必須執行輕量瘦身掃描，主動列出：(a) disabled formal Audit、(b) 無法由目前正式runtime或active formal Audit import-reachability到達的`tools/audit`／compatibility模組、(c) 明顯過大的Audit-specific synthetic test。此掃描預設只提供`CLEAN／REVIEW`維護訊號與候選清單，不得單獨造成formal FAIL，避免維護建議反過來形成第二套阻擋型Audit。
13. 自動瘦身判定只能提出候選，不得因檔名、年齡或行數直接刪除；若歷史compatibility仍被目前正式runtime用於舊工件解讀／重現，或模組仍是目前target／PIT／strategy compare的必要依賴，就必須保留。實際刪除前須確認引用鏈，並優先移除已完成的一次性研究程式與其專屬測試。


## D. 交易與策略原則

1. 杜絕未來函數：任何候選、掛單、成交、停損、停利、延續判斷與統計，都不得偷看當下尚未知的未來資料。
2. 不確定時一律採最保守、最不利於績效的可執行解讀。
3. 受限使用者盤中無法操作，必須盤前完成掛單 (包含買入標的、買入股數、買價上限)，並且無法在同天買入又賣出同一隻股票、也禁止當沖交易。
4. 由於D3的限制，交易當日資金運用在必須在盤前就鎖定，就算沒買到，盤中也無法換股操作。
5. 受限卷商無法賣出未持倉股票，必須在買入後才能依成交價設定停損、停利。
6. 資金、權益、PnL、報酬率、勝率、EV 與 Round-Trip 定義，必須以扣除手續費、稅金後的淨值為準，且不得因半倉、顯示或報表需求分叉。
7. 執行條件與統計口徑必須全專案保持一致，包含 風險 sizing、停損/停利、trailing stop、延續候選、失效、hit 判斷、觸發紀錄等。
8. 掛單、成交、觸發、執行、失效、達標與結算必須分層定義；


## E. 專案特例

1. `apps/portfolio_sim.py` 自動開瀏覽器暫時允許。
2. `doc/ToDo.md` 與一般使用者註解屬使用者自有備忘／說明，不納入 formal / GPT 最嚴格檢查；但若其文字被正式介面直接讀取，仍視為正式輸出。
3. 暫時只使用還原價，不考慮 raw。
4. `doc/FINMIND_API_TOKEN.md` 為使用者本機私有憑證文件；其內容與是否被 `apps/package_zip.py` 收錄，暫時排除於 GPT 與 formal 最嚴格檢查及修正範圍之外。除非使用者另行要求，不得主動修改、移除、遮罩、加入 `.gitignore` 或調整打包器排除規則。
5. 凡使用者要求分析、改善、修改、比較或測試 `breakout_quality filter`，開始任何設計、命名或程式修改前，必須依序讀取 `/doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md` 與 `/doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md`。Registry 是實驗 ID／namespace／命名／版本語意／目前基準與 ID 佔用狀態的單一真理來源；Experiment Log 是已測矩陣、實驗證據、結果、排除方向與後續順序的單一真理來源。新增任何模型研究、DL source、策略 runtime arm 或 Audit 前必須先查 Registry，不得憑對話記憶編號、不得重用已淘汰／停止／legacy ID。每次實驗完成後，必須在同一輪同步 Registry 的 identity／status，並將程式基準、唯一變更、固定條件、Dataset／Label 重建需求、Selection／OOS 主要結果、與基準差異、採用判定及下一步回寫 Experiment Log。尚未取得結果的實作只能標記為 `IMPLEMENTED` 或 `PLANNED`，不得預先寫成有效或無效；若新結果與舊紀錄衝突，須以可追溯的最新結果更新並保留差異說明。
6. `breakout_quality filter` 的 model architecture 版本只可表示神經網路結構或輸入表示的差異；optimizer、learning-rate schedule、augmentation、loss weighting 等訓練方法必須以命名 experiment profile 管理，不得為了隔離實驗輸出而新增模型版本。正常新訓練只可使用 active architecture；已淘汰架構應保留為 legacy read-only compatibility，供舊 checkpoint／manifest 重建與歷史重現，不得再次出現在正式新實驗入口。
7. `breakout_quality filter` 的 OOS 可持續用於實驗結果比較、錯誤歸因、年度／regime 診斷、策略經濟效果評估，以及形成下一個實驗假設；不得因 OOS 已被查看而禁止後續研究。每個實驗的 Train／Validation、loss、gradient、early stopping、epoch 選擇、threshold／calibration 擬合、normalization、feature／label 建立、sample weighting 與 hyperparameter optimization，均不得讀取或使用 OOS rows、labels、scores 或其統計量。模型是否優於基準可由固定 OOS 結果判定；若同一 OOS 被反覆使用，僅需明確標記為「迭代研究 OOS 證據」，不得宣稱是 untouched holdout，但不得把等待新資料當成唯一或主要下一步。

8. `breakout_quality filter` 的 identifier namespace 必須分層：模型／學習研究使用 Registry 的 `MR-*`；model architecture 使用 `ARCH-*`；runtime DL source 使用 `DL-*`；策略使用方式／portfolio runtime arm 使用 `SR-C*`；策略參數 stage 使用 `PARAM-P*`；Audit 使用 `AUD-*`。既有程式 alias（例如 `A9`、`C12`）可為相容性保留，但文件在可能混淆時必須使用 canonical prefix。單純改變既有 DL 的 score refresh timing、gate、ranking 或 allocation 屬 `SR-C*`，不得誤編成新 model experiment；只有模型權重／訓練目標／architecture／training-data semantics 的受控研究變更才可取得新 `MR-*`。所有已使用 ID 永久保留，不得回收。

9. `breakout_quality filter` 的研究 Audit 必須遵守「最小必要證據」原則。開始 Audit 前必須先定義待決策問題、目前真正阻擋決策的關鍵不確定性，以及可停止分析並做出 `GO`、`REJECT` 或 `NEXT EXPERIMENT` 的條件；只有當既有證據尚不足以做出該決策，而且新的 Audit 結果可能實質改變決策時，才新增 Audit。Audit 的範圍必須縮到足以回答該關鍵不確定性的最小集合，優先重用既有正式工件與 read-only 診斷，不得為了 attribution 完整度、可解釋性、好奇心或「還能再拆更細」而擴大分析。Audit 次數不設硬性上限；若前一輪後仍存在會實質改變決策的關鍵不確定性，可以再做下一個必要 Audit，但每一輪都必須重新說明缺少哪一項決策證據，以及該結果如何影響下一步。一旦證據已足以支持決策，就必須停止 Audit 鏈並直接進入決策或下一個受控實驗。

# 專案設定

1. 每輪開始專案修改或檢查前，必須先讀取並遵守 `doc/PROJECT_SETTINGS.md`。本文件為全專案最高層治理原則，只保存長期穩定、跨模組且可泛化的專案級規則；可變設定、current identity、實驗狀態、工件實作細節與歷史結果，應由各自的 `config/`、Registry、Architecture、Research Queue 或 Experiment Log 持有，不得在本文件建立第二份真理來源。
2. 使用者明確要求「完整檢查」時，GPT 必須進入全專案檢查模式，以表格逐項檢查：(1) 本文件 A 至 E 的全部適用條款；(2) 雖未列於本文件但依本輪基準與架構應檢查的其他項目。各項必須有獨立檢查結論，不得以 formal suite PASS 或其他項目結果代替；所有適用項目完成前不得宣稱完整檢查完成。


## A. 標準測試流程

1. 本輪工作基準為使用者最新提供的程式、ZIP、檔案，或本輪最新 GPT 交付之程式碼／patched ZIP；後出現者取代先前版本並成為當前基準。
2. 每次開始專案修改或檢查前，必須確認並回報當前工作基準與本輪必要已讀文件；若基準為 ZIP，另須回報 ZIP 檔名、SHA256 與本輪使用的全新解壓目錄。
3. GPT 每輪修改程式時，必須完整檢查本輪修改範圍、直接依賴鏈、正式契約及可能受影響的同鏈項目，並在同輪修正已可確定的同鏈問題；使用者明確要求「完整檢查」時，另依本文件頂層第 2 條執行全專案檢查。
4. `apps/run_bundle.py` 為本機正式 double check 與交付打包的單一使用者入口；預設流程為 stage 變更 → commit 當前 snapshot → package ZIP → 執行 `apps/test_suite.py`。formal test 即使 FAIL，既有 commit 與 ZIP 仍須保留，以供完整交付與閉環修正。`apps/test_suite.py` 僅為內部正式 runner，不作一般使用者主要入口。formal double check 不取代 GPT 依 A3 或頂層第 2 條應完成的檢查。
5. GPT 不得執行 `apps/run_bundle.py` 或 `apps/test_suite.py`，亦不得重建整套 formal suite 冒充本機正式測試。GPT 仍須以獨立於 formal suite 的方式完成本輪應有檢查；本輪涉及正式測試鏈、使用者要求完整檢查，或已有證據顯示 formal gate 本身可能有問題時，才須進一步檢查 `apps/run_bundle.py`、`apps/test_suite.py` 與 `doc/TEST_SUITE_CHECKLIST.md` 的可信度與必要覆蓋。
6. 若使用者提供正式測試結果，GPT 必須依失敗證據完成閉環分析與修正；未提供時只能視為尚無本機 formal double check 證據，不得自行假定 PASS。正式本機整合測試一律使用 `apps/run_bundle.py` 預設流程，不使用 `--no-commit`，除非使用者當輪明確要求。
7. `doc/PROJECT_SETTINGS.md` 不得由 formal suite 或 validator 以硬編碼目前條款內容的方式反向定義其合法性；formal tests 只能驗證程式遵守已落地的正式契約，不得成為 PROJECT_SETTINGS 的上位真理。
8. 發現屬本輪修改範圍或直接依賴鏈、且修正方向可由既有契約唯一確定的問題時，應同輪直接修正；涉及新的 research decision、scientific semantics、使用者設定取捨或不可唯一確定的產品決策時，不得擅自代替使用者決定。


## B. 回覆、交付與輸出

1. 只提供客觀分析與建議，不奉承。
2. 回覆必須講重點、明確、精準，不得將相同概念分不同層次重複說明。
3. AI 註解與使用者註解必須明確區分；不得刪除使用者註解，也不得在無必要理由下修改或刪除原本正確的程式碼。
4. 提供程式碼片段時，必須先給可直接搜尋定位的完整舊片段，再給可直接貼上的新片段。
5. 提供修補 ZIP 時，只能包含有修改的檔案，並維持目錄架構，讓使用者可直接在 root 貼上取代舊檔，並列出修改檔案。
6. 預設以提供 ZIP 為主，除非使用者要求程式碼片段，或只需要修改很小範圍。
7. 如架構調整需刪檔，須提供可執行的 command，避免使用者手動刪錯。
8. `outputs/` 根目錄只作正式輸出 domain 的分類入口；持久工件必須落入其所屬工具／domain 子目錄，不得無治理地散落於根目錄。
9. 交付前必須完成 GPT 端自檢；範圍以 blocker、正式輸出、正式契約、必要文件同步及本輪修改同鏈項為準，不得把單次事故 patch、validator wording 或 hygiene 再提升成第二套永久 gate。
10. 修改 `doc/TEST_SUITE_CHECKLIST.md` 時，必須遵守該文件自身定義的排序、schema、transition 與 Markdown table 機械契約；PROJECT_SETTINGS 不另保存其細部格式規格。
11. 對研究、改善與除錯問題，應優先提出可立即執行且無前視的分析、實驗或修正；除非使用者明確要求，不得把等待未來資料或暫停研究作為主要下一步。
12. 所有使用者可見的檔案／資料夾路徑，預設以專案根目錄為基準顯示穩定相對路徑，並統一使用 `/` 分隔；一般 console、摘要或工件清單不得輸出本機絕對路徑。Manifest、hash identity、runtime 讀取與外部明確指定路徑可保留 canonical path，但不得改變使用者可見顯示規則。
13. GPT 回報正式操作方式時，若已有對應互動入口，必須以正式選單操作為主；只有尚未納入正式流程的臨時診斷／研究操作才提供直接 CLI。
14. 正式互動流程不得要求使用者先執行可由程式確定完成的零散 CLI；可調設定、執行對象、工件來源與前置建立政策須由正式設定／Registry 驅動，不得硬編於選單。
15. 正式 App 執行前必須建立可稽核的 dependency plan；可由既有真理工件或 canonical producer 確定性補建的依賴，應自動 REUSE／BUILD／REBUILD／RESUME。只有需要新的研究選擇或缺少不可確定建立的上游真理時才能 BLOCKED。
16. 所有正式持久研究／策略結果都必須提供可直接閱讀的人讀摘要；RUN／REUSE／cache 必須重用同一 canonical result 與 renderer，不得因重用方式不同而改變報表口徑。JSON／CSV／manifest 等詳細工件僅作補充。
17. 所有正式互動選單只可硬編穩定的工作類型；current 實驗、model、arm、reference、ID、顯示順序與可選集合必須由設定／Registry／current state 驅動，validator 不得硬編目前值為唯一合法答案。
18. 所有人讀報表的方向判讀與色彩語意必須重用 `core/report_style.py`；不得以 emoji／燈號建立第二套判讀規則，也不得直接以數值正負代替 metric direction。不同 renderer 必須使用同一語意契約；具體 palette、aggregate 上色與顯示實作由 report style／Architecture 持有。
19. 同一研究／策略結果的 metric 定義、label、單位、precision 與方向性必須來自共用 metric registry 與同一 canonical result；renderer 只負責組裝與顯示，不得由 raw rows 另算第二套同名指標。
20. 長流程互動執行預設只顯示必要的 RUN／REUSE／DONE 進度與最終核心摘要；pair-level 詳細報表、solver states、repair/ascent、stale guard、timing 等過程資訊應保存於工件供追查，不得在 aggregate 執行時重複洗版。
21. 正式共同報表的核心 metric 集合、欄位順序與表格結構必須由 `core/report_metrics.py` 的 canonical registry／report contract 唯一持有並由 renderer 重用，不得在 PROJECT_SETTINGS 或個別報表保存第二份欄位清單。
22. R 預測／轉化等診斷 metric 的分群、欄位與順序由 canonical metric/report registry 唯一持有；RCE 必須由 canonical diagnostics contract 唯一定義，所有使用者共用相同 trade partition、Target join、coverage 與 numerator／denominator semantics，不得在 renderer 或其他模組建立 proxy／第二套公式，也不得增加未經正式 contract 定義的 magic threshold。
23. 不同性質的診斷 metric 不得互相替代或以 proxy 偽裝成同一指標；報表需要的既有模型驗證 metric 必須直接重用 canonical validation artifact，不得為報表重算第二套。


## C. Coding 與架構原則

1. 單一真理來源：相同邏輯不得重複實作，核心規則、帳務、價格與統計的衍生結果不得分叉。
2. 統計口徑必須完全一致：成交、未成交、miss buy、EV、勝率與 Round-Trip PnL 不得因路徑、顯示或用途不同而改變定義。
3. 避免 magic number；公式、參數與比較規則必須可解釋。
4. 禁止裸 except；所有被捕捉且非純 control-flow／feature probe 的異常都必須可追蹤。
5. 架構調整不得明顯犧牲效率；若提高未來策略修改或 ML／DRL／LLM 升級複雜度，必須先明確說明。
6. 正式 application entry 位於 `apps/`；可重用的正式 domain／service／core logic 不得放入單一 App；`tools/` 只承擔驗證、診斷與開發輔助，不得成為正式 runtime 的第二份規則來源。
7. 拆分、合併、移動或重新命名檔案時，必須遵守單一職責、分層呼叫、禁止反向依賴、禁止循環依賴、禁止規則分叉與禁止重複實作。
8. `config/` 的正式可調參數不得被 GPT、validator 或 formal suite 以目前值／預設值硬編成唯一合法答案；測試只能驗證 schema、合法性、跨欄契約及 runtime 是否忠實採用設定。固定測試案例必須使用隔離 override，不得修改或限制實際 config。
9. `apps/research.py` 可作統一使用者入口，但模型訓練、Optimizer、Strategy Compare 與 Audit 必須維持獨立 application／service 責任與 canonical producer ownership。Consumer／orchestrator 不得建立新的 scientific identity、改變 target／architecture／loss／hyperparameter、挑選模型／seed，或複製 canonical producer 邏輯；可依 dependency plan 委派 canonical producer 執行確定性的 REUSE／BUILD／REBUILD／RESUME，完成後重新 plan，但 producer ownership 不因此轉移。需要新的研究選擇或缺少不可確定建立的上游真理時才可 BLOCKED。
10. Strategy parameter truth 只能由 canonical Optimizer producer 產生；其他工作類型只能解析、驗證、委派及消費，不得建立第二套 search／stitch／freeze／seed／trial semantics。
11. 研究品質 gate 與最終策略績效 gate 必須分層；若 current research contract 明確允許受控 FAIL／WARN 繼續驗證，其行為由正式 research spec 定義，不得由 consumer 自行決定，也不得因通過診斷流程自動 promotion。
12. Benchmark／robustness 可建立隔離的 per-seed transient artifacts，但只能透過 canonical producer 建立，不得覆寫 production truth、挑 best seed、做隱含 ensemble 或自動 promotion。
13. 前置工件自動建立必須先顯示可稽核計畫並只確認一次；任一步驟失敗時立即停止後續回放、保留可接續工件、回報失敗步驟與相對路徑，不得產生不完整的正式比較報表。
14. 臨時研究、Audit、診斷與 dedicated synthetic 屬可拋棄工程資產；待決策問題結案且正式 runtime／必要 compatibility 已不再依賴時，應連同專屬 registration、helper 與 tests 一併退役，不得以永久 disabled／historical CLI 累積。
15. Meta-quality 可執行輕量的 dead-code／research-debt 候選掃描，但其詳細掃描集合由正式 test contract 持有；此類維護訊號預設不得單獨形成 formal FAIL。
16. 自動瘦身判定只能提出候選，不得因檔名、年齡或行數直接刪除；實際刪除前須確認 dependency／compatibility 引用鏈，並優先移除已完成的一次性研究程式與其專屬測試。
17. 被採納為 current workflow 的研究成果必須將仍有效的定義與 producer／service 移入正式 domain，再退役原 experiment／Audit implementation；除非另有研究決策，此遷移必須 behavior-preserving。


## D. 交易與策略原則

1. 杜絕未來函數：任何候選、掛單、成交、停損、停利、延續判斷與統計，都不得偷看當下尚未知的未來資料。
2. 不確定時一律採最保守、最不利於績效的可執行解讀。
3. 受限使用者盤中無法操作，必須盤前完成掛單（包含買入標的、買入股數、買價上限），並且無法在同天買入又賣出同一隻股票，也禁止當沖交易。
4. 由於 D3 的限制，交易當日資金運用必須在盤前鎖定；即使掛單未成交，盤中亦不得將該資金重新分配至其他股票。
5. 受限券商無法賣出未持倉股票，停損／停利掛單只能在買入成交後依實際成交價建立。
6. 資金、權益、PnL、報酬率、勝率、EV 與 Round-Trip 定義，必須以扣除手續費、稅金後的淨值為準，且不得因半倉、顯示或報表需求分叉。
7. 執行條件與統計口徑必須全專案保持一致，包含風險 sizing、停損／停利、trailing stop、延續候選、失效、hit 判斷、觸發紀錄等。
8. 掛單、成交、觸發、執行、失效、達標與結算必須分層定義。


## E. 專案特例

1. `apps/portfolio_sim.py` 自動開瀏覽器暫時允許。
2. `doc/ToDo.md` 是使用者自行維護的私人工作筆記，不是專案 research backlog、current status、決策依據或 GPT 待辦來源。除非使用者當輪明確要求讀取／整理 `doc/ToDo.md`，GPT 不得主動讀取、引用、依賴或用其內容推導下一步；formal／GPT 最嚴格檢查亦排除該檔。一般使用者註解同屬使用者自有備忘／說明；若其文字被正式介面直接讀取，仍視為正式輸出。
3. 暫時只使用還原價，不考慮 raw。
4. `doc/FINMIND_API_TOKEN.md` 為使用者本機私有憑證文件；其內容與是否被 `apps/package_zip.py` 收錄，暫時排除於 GPT 與 formal 最嚴格檢查及修正範圍之外。除非使用者另行要求，不得主動修改、移除、遮罩、加入 `.gitignore` 或調整打包器排除規則。
5. 凡分析、修改或測試 `breakout_quality`，開始新的 scientific identity、設計或程式修改前必須先讀取 `doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md` 與 `doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md`。Registry 是 identity／namespace／current status 的唯一真理；Experiment Log 是已執行 evidence／result／decision 的唯一真理。實驗開始與完成時須同步更新其應負責的狀態，未取得結果不得預先標記成功或失敗。
6. `breakout_quality` architecture identity 只表示模型結構或輸入表示；training recipe、loss、optimizer、augmentation 等實驗差異由 experiment profile／research identity 管理。已淘汰 architecture 只可作 historical compatibility，不得成為新的 current experiment identity。
7. `breakout_quality` 的 OOS 可持續用於實驗結果比較、錯誤歸因、年度／regime 診斷、策略經濟效果評估及形成下一個實驗假設；不得因 OOS 已被查看而禁止後續研究。但任何 Train／Validation、loss、gradient、early stopping、epoch 選擇、threshold／calibration、normalization、feature／label、sample weighting 或 hyperparameter optimization 均不得讀取或使用 OOS rows、labels、scores 或其統計量。若同一 OOS 被反覆使用，應標記為「迭代研究 OOS 證據」，不得宣稱是 untouched holdout。
8. `breakout_quality` 的 canonical identity namespace、prefix 與 allocation 規則由 Experiment Registry 唯一持有；程式 alias 僅作 compatibility。已使用的 scientific identity 永久保留，不得回收或憑記憶重用。
9. `breakout_quality` 的研究 Audit 必須遵守「最小必要證據」原則。開始 Audit 前必須先定義待決策問題、目前真正阻擋決策的關鍵不確定性，以及可停止分析並做出 GO／REJECT／NEXT EXPERIMENT 的條件；只有既有證據不足且新的 Audit 可能實質改變決策時才新增 Audit。Audit 次數不設硬性上限；一旦證據足以支持決策，就必須停止 Audit 鏈並直接進入決策或下一個受控實驗。
10. Daily-universal model research 必須維持 strategy-agnostic stock-day universe；breakout qualification、candidate membership、portfolio cash／holdings／selector state 等策略執行資訊，不得在沒有明確新 research decision 下滲入通用模型 Target／Input。若通用模型需要 risk／economic geometry，只能使用與模型目標必要且 decision-time 合法的資訊；strategy execution-specific geometry 不得無理由滲入 universal target。模型若使用 risk／cost／accounting semantics，必須直接重用 canonical risk/accounting SSOT，不得在 DL 模組建立第二套公式。歷史參數或其他 input 缺失不得以未來資訊回填；research artifact 不得因完成訓練或產生結果而自動取代 production identity。
11. `doc/BREAKOUT_QUALITY_RESEARCH_QUEUE.md` 只管理尚未完成的研究問題、優先順序、前置條件與停止條件；Registry 管 identity／current state，Experiment Log 管已完成 evidence。推進研究前須依 E5 讀完 Registry 與 Experiment Log，再讀 Research Queue；開始、完成、改變優先順序或結案時同步更新。未開始實作的項目不得預占 scientific identity。
12. `breakout_quality` 的時間驗證、Strategy Compare、Robustness 與其工件治理遵守以下長期規則；current mode、日期、ID、seed count、worker、路徑與實驗狀態由各自的 `config/`、Registry、Architecture、Research Queue 或 Experiment Log 持有：
   - **時間合法性**：任一 prediction／fold 的 training、target maturity、validation、normalization、sample weighting、parameter fitting 與其他自動決策，都必須在該 prediction／fold 的 information cutoff 前合法可得；evaluation 結果不得回頭影響當下模型或參數。
   - **Data-driven history**：Model training universe 與會影響 fitting 的歷史邊界必須由自身 data／feature／target legality 決定，不得被其他 workflow 的 historical cutoff 無意截斷；相關邊界屬 artifact reuse identity。
   - **Historical evidence**：Historical scientific identity、原始結果、controlled contrast 與當時語意不得因 current framework 改版而回頭改寫、刪除或偽裝成 current evidence。
   - **Strategy parameter SSOT**：Strategy parameter truth 只能由 canonical Optimizer producer 建立；同一 scientific parameter truth 不得因 OOS／Rolling 或其他 evaluation mode 建立第二份 physical truth。Consumer／orchestrator 只能解析、驗證、委派及消費。
   - **Artifact contract / isolation**：Artifact reuse 必須以完整 canonical contract／identity 驗證，不得僅因檔案存在即視為 READY。不同 scientific identity、evaluation identity、benchmark scope 或 semantic purpose 的 artifacts 必須隔離，不得互相覆寫或錯誤 REUSE；composite／multi-source arm 的所有依賴必須解析至一致且相容的 evaluation identity。
   - **Research / Production**：Research、benchmark 或 candidate artifact 不得因建立完成自動成為 production／runtime truth；promotion 必須有明確 decision 與所需 evidence。
   - **Compare Suite SSOT**：同一 comparison framework 的 arms、contrasts 與 policies 必須由單一 canonical Compare Suite 定義；single／multi、OOS／Rolling 不得各自維護第二份 scientific matrix。Multi-seed robustness 必須是 canonical single-seed suite 的 end-to-end 多 seed 延伸；除 seed 集合與跨-seed aggregate 外，不得私自增加、刪除或改變 scientific arms／contrasts。
   - **Robustness seed semantics**：Robustness benchmark seeds 必須由單一 config-driven benchmark specification 可重現地產生；同一 benchmark seed 必須一致傳遞至所有 declared seed-sensitive components，不得讓不同 component 使用不同 benchmark seed、挑 best seed、fallback production seed／其他 seed，或把 benchmark sampling 與 production ensemble／consensus sampling 混用。
   - **Producer / dependency semantics**：Single-seed 與 multi-seed 必須共用同一 canonical dependency planner、producer／service、artifact resolver 與 scientific contract；producer 只能建立其所擁有的 domain，不得以 side effect 越權建立下游 scientific artifact。單一 research execution 若要求 end-to-end same-seed，seed identity 必須有單一 config SSOT，各 component 不得持有獨立數值或隱含 fallback。
   - **Checkpoint / score reuse**：Model checkpoint reuse 必須依 scientific fitting identity；純 evaluation horizon／mode label 不得使相同 fit 成為不同 scientific model。Checkpoint 可依 fitting identity 共用，但 score 仍須依各自 score universe／evaluation contract 產生，不得因共用 checkpoint 而跨 evaluation identity 誤用 score。完全相同且宣告 deterministic 的 fitting identity 若產生不相容 artifact，必須 fail-fast。
   - **Authorization / identity**：Scientific workflow 的執行資格必須由 canonical research identity／authorization contract 明確決定；UI 可顯示未授權工作類型及 BLOCKED 原因，但 builder／consumer 不得繞過底層 authorization。Logical evaluation source identity 可以表達不同 consumption semantics，但不得因此複製相同 physical truth。
   - **Execution vs science**：Prefetch、parallelism、cache 及其他 performance optimization 只能改變 execution strategy，不得改變 sample membership／order、seed、optimizer updates、loss、determinism 或其他 scientific semantics；具體 execution knob 由 config 持有，不屬 scientific identity。Performance benchmark 結果只記錄於 Experiment Log，不進 PROJECT_SETTINGS。

# 專案設定

1. 每輪開始專案修改或檢查前，必須先讀取並遵守 `doc/PROJECT_SETTINGS.md`。本文件為全專案最高層治理原則；除 E 章明確列出的專案例外外，只保存長期穩定、跨模組且可泛化的專案級規則。可變設定、current identity、實驗狀態、工件實作細節與歷史結果，應由各自的 `config/`、Registry、Architecture 或 Experiment Log 持有，不得在本文件建立第二份真理來源。
2. 使用者明確要求「完整檢查」時，GPT 必須進入全專案檢查模式，以表格逐項檢查：(1) 本文件 A 至 F 的全部適用條款；(2) 雖未列於本文件但依本輪基準與架構應檢查的其他項目。各項必須有獨立檢查結論，不得以 formal suite PASS 或其他項目結果代替；所有適用項目完成前不得宣稱完整檢查完成。


## A. 標準測試流程

1. 本輪工作基準為使用者最新提供的程式、ZIP、檔案，或本輪最新 GPT 交付之程式碼／patched ZIP；後出現者取代先前版本並成為當前基準。
2. 每次開始專案修改或檢查前，必須確認並回報當前工作基準與本輪必要已讀文件；若基準為 ZIP，另須回報 ZIP 檔名、SHA256 與本輪使用的全新解壓目錄。
3. GPT 每輪修改程式時，必須完整檢查本輪修改範圍、直接依賴鏈、正式契約及可能受影響的同鏈項目，並在同輪修正可由既有契約唯一確定的問題；涉及新的 research decision、scientific semantics、使用者設定取捨或不可唯一確定的產品決策時不得擅自決定。使用者明確要求「完整檢查」時，另依本文件頂層第 2 條執行全專案檢查。
4. `apps/run_bundle.py` 為本機正式 double check 與交付打包的單一使用者入口；預設流程為 stage 變更 → commit 當前 snapshot → package ZIP → 執行 `apps/test_suite.py`。formal test 即使 FAIL，既有 commit 與 ZIP 仍須保留。`apps/test_suite.py` 僅為內部正式 runner，不作一般使用者主要入口。若使用者提供正式測試結果，GPT 必須依失敗證據完成閉環分析與修正；未提供時不得自行假定 PASS。正式本機整合測試不使用 `--no-commit`，除非使用者當輪明確要求。
5. GPT 不得執行 `apps/run_bundle.py` 或 `apps/test_suite.py`，亦不得重建整套 formal suite 冒充本機正式測試；仍須以獨立方式完成本輪應有檢查。本輪涉及正式測試鏈、使用者要求完整檢查，或已有證據顯示 formal gate 本身可能有問題時，才須進一步檢查 `run_bundle.py`、`test_suite.py` 與 `doc/TEST_SUITE_CHECKLIST.md` 的可信度與必要覆蓋。
6. `doc/PROJECT_SETTINGS.md` 不得由 formal suite 或 validator 以硬編碼目前條款內容的方式反向定義其合法性；formal tests 只能驗證程式遵守已落地的正式契約，不得成為 PROJECT_SETTINGS 的上位真理。


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
13. 正式互動流程應以既有正式選單為主要操作入口，不得要求使用者先執行可由程式確定完成的零散 CLI；只有尚未納入正式流程的臨時診斷／研究操作才提供直接 CLI。**Continuous DL 模型研究的七個工作類型 [1]～[7]、文案、順序與選號是跨正式研究模型設定的固定 UI 契約，隔離實驗不可另生選單或scope切換，不得由當前演算法、Registry 可執行清單、rolling_authorized、是否已有工件或實驗狀態決定隱藏、改號或改操作流程。** Registry／config 只決定模型與研究執行契約，**標準 Research 服務**須具備七項既定工作且不能向實驗 Producer 查詢選單能力；缺少可確定建置的依賴由原入口自動委派。標準工作缺失是整合錯誤，應由通用契約測試 fail-fast，**不得改測試期待隱藏選項使之 PASS**。**研究選單及工作流程依 2026-09-04 已有介面維持原樣**：不得因某一 MR identity 加入額外 scope 切換、[8] 工作鍵、模型專用說明、確認選單或不同 dispatch；正式比較對象只能取自既有 Model Compare/Test config，不能被特定實驗臨時改寫。模型特有診斷只能隔離為一次性專題 Audit，不得替換常駐模型比較及 Standard SOP。其他選單只可硬編穩定工作類型，current 實驗、model、arm、reference、ID 與前置政策仍由設定／Registry／current state 驅動，validator 不得把目前 scientific identity 當成唯一合法值。
14. 〔歷史引用索引〕原 B14 正式人讀報表與 RUN／REUSE 一致性規則，統一以 F2 為準。
15. 〔歷史引用索引〕原 B15 指標／顯示 SSOT 規則，統一以 F3、F4 為準。
16. 〔歷史引用索引〕原 B16 診斷指標來源與口徑規則，統一以 F3 為準。
17. 〔歷史引用索引〕原 B17 長流程進度／摘要規則，統一以 F9 為準。
18. 〔歷史引用索引〕原 B18 正式報表改版授權／版本保護規則，統一以 F4～F7 為準。
19. 〔歷史引用索引〕原 B19 Machine-readable Contract、Mode 與 Audit 規則，統一以 F4、F8 為準。


## C. Coding 與架構原則

1. 單一真理來源：相同邏輯不得重複實作，核心規則、帳務、價格與統計的衍生結果不得分叉。
2. 統計口徑必須完全一致：成交、未成交、miss buy、EV、勝率與 Round-Trip PnL 不得因路徑、顯示或用途不同而改變定義。
3. 避免 magic number；公式、參數與比較規則必須可解釋。
4. 禁止裸 except；所有被捕捉且非純 control-flow／feature probe 的異常都必須可追蹤。
5. 架構調整不得無必要地明顯犧牲效率；correctness、scientific semantics、PIT legality 與 SSOT 優先於效能。若必要的正確實作會造成明顯效能或未來維護成本，必須明確說明 trade-off，再於不改變上述語意的前提下優化。
6. 正式 application entry 位於 `apps/`；可重用的正式 domain／service／core logic 不得放入單一 App；`tools/` 只承擔驗證、診斷與開發輔助，不得成為正式 runtime 的第二份規則來源。
7. 拆分、合併、移動或重新命名檔案時，必須遵守單一職責、分層呼叫、禁止反向依賴、禁止循環依賴、禁止規則分叉與禁止重複實作。
8. `config/` 的正式可調參數不得被 GPT、validator 或 formal suite 以目前值／預設值硬編成唯一合法答案；測試只能驗證 schema、合法性、跨欄契約及 runtime 是否忠實採用設定。固定測試案例必須使用隔離 override，不得修改或限制實際 config。
9. `apps/research.py` 可作統一使用者入口，但模型訓練、Optimizer、Strategy Compare、Audit 等 domain 必須維持獨立 application／service 責任與 canonical producer ownership。正式流程須先建立可稽核 dependency plan；**authorization 與 scientific contract legality 優先於 dependency completeness**。只有在 current authorization 與 scientific contract 允許的前提下，可確定的缺失依賴才由 canonical producer 執行 REUSE／BUILD／REBUILD／RESUME 並重新 plan；未授權 workflow 不得因依賴技術上可建立而繞過 BLOCKED。consumer／orchestrator 只能解析、驗證、委派與消費，不得建立新的 scientific identity、改變 target／architecture／loss／hyperparameter、挑選模型／seed、複製 producer 邏輯，或以 side effect 越權建立下游 artifact。需要新的研究選擇、未取得必要 authorization，或缺少不可確定建立的上游真理時才可 BLOCKED。前置建立只確認一次；任一步驟失敗即停止下游正式回放／報表並保留可接續工件。
10. Strategy parameter truth 只能由 canonical Optimizer producer 產生；其他工作類型不得建立第二套 search／stitch／freeze／seed／trial semantics。同一 scientific parameter truth 不得因 OOS／Rolling 或其他 evaluation mode 建立第二份 physical truth，只能依各 mode 的合法時間語意解析與消費同一 canonical lineage。
11. Artifact reuse 必須依完整 canonical contract／identity 驗證，不得僅因檔案存在即視為 READY；不同 scientific identity、evaluation identity、benchmark scope 或 semantic purpose 的 artifacts 必須隔離，不得互相覆寫或錯誤 REUSE。Composite／multi-source 工作的各依賴也必須解析至彼此相容的 identity。
12. 研究品質 gate 與最終策略績效 gate 必須分層；受控 FAIL／WARN 是否允許繼續由正式 research spec 決定。Research、benchmark、candidate 或 per-seed transient artifact 不得因建立完成、通過診斷流程或取得較佳結果而自動覆寫 production truth、挑 best seed、形成隱含 ensemble 或自動 promotion；promotion 必須有明確 decision 與所需 evidence。
13. 臨時研究、Audit、診斷與 dedicated synthetic 的 **implementation 資產**屬可拋棄工程資產。被採納的研究成果必須將仍有效的定義與 producer／service 移入正式 domain，再退役原 experiment／Audit implementation；未採納或已結案的 implementation 則在確認 runtime／compatibility dependency 後，連同專屬 registration、helper 與 tests 一併退役。退役 implementation 不得刪除或改寫 Registry／Experiment Log 中的 scientific identity、decision、原始結果與維持歷史可追溯性所必要的 evidence。自動瘦身只能提出候選，不得因檔名、年齡或行數直接刪除；除非另有研究決策，promotion／遷移必須 behavior-preserving。
14. Meta-quality 可執行輕量的 dead-code／research-debt 候選掃描，但其詳細掃描集合由正式 test contract 持有；此類維護訊號預設不得單獨形成 formal FAIL。
15. Prefetch、parallelism、cache 或其他 performance optimization 只能改變 execution strategy，不得改變 sample membership／order、seed、optimizer updates、loss、determinism 或其他 scientific semantics；具體 execution knob 由 config 持有，不屬 scientific identity，performance benchmark 結果只記錄於對應實驗紀錄。

16. **Declarative Experiment / Generic Engine**：研究 identity、顯示名稱與歷史敘述不得滲入正式 execution engine。Experiment／profile 必須先解析成單一 canonical runtime contract（例如 target、context、objective／loss、pair-weight、dependency、model-output schema、metric bundle、authorization），trainer、data provider、artifact planner、inference、menu 與 renderer 只能消費該 contract 或其 reusable capability，不得各自以 `MR-*`、profile name、target alias 或其他 identity 重猜相同語意。若新研究只是重新組合既有 capability，production engine 原則上只新增／修改 declarative spec；若仍需在多個 generic consumer 增加 experiment-specific `if/elif`，視為架構缺口，必須先抽成 reusable policy／capability，而不是擴散特例。
17. **能力單一 owner 與修改半徑**：新的 target、context、objective、pair-weight、model-output、artifact dependency、metric 或 authorization semantics 必須各有唯一 owner／registry／contract；不得同時在 config、trainer、data、dependency planner、report、menu 與 tests 維護多份對照表。跨層 consumer 需要同一資訊時，必須從 owner 派生或透過 runtime contract 傳遞。若一個局部需求造成多個同層／跨層檔案同步認 identity，應優先視為 shotgun-surgery 訊號並收斂 abstraction。
18. **避免 God module 與中央 switch 持續膨脹**：不得因新增 experiment／architecture／objective 而持續把 identity-specific branch 堆入大型 trainer、factory、application 或 config helper。拆分必須依責任與穩定 contract 邊界，不得只為降低 LOC 把一個大型模組切成多個高度互相依賴的小檔；重構的成功判準包含後續需求的修改檔案數與需理解的責任範圍下降，而不只是單檔變小。
19. **Capability-oriented tests**：generic capability 應由參數化／registry-driven contract tests 自動覆蓋；dedicated synthetic 只用於真正新增 primitive、特殊 invariant 或歷史 bug regression。不得為每個只是既有 capability 組合的新 experiment，重複維護 `impacted_modules`、coverage target、檔名清單或第二份 experiment matrix；檔案位置重構不應被測試系統誤視為 scientific semantic 變更。
20. **重構不得改 scientific/artifact identity**：純 architecture／code-organization refactor 必須保持既有 runtime payload、manifest／fingerprint、artifact reuse、PIT legality、seed、模型輸出與報表 contract 語意不變；若無法保持，必須把會改 identity 的部分拆成另一次明確 research／product decision，不得混在 behavior-preserving cleanup 中。
21. **研究變更隔離與修改半徑驗收**：新增 scientific identity 或 reusable capability 時，不得在同一輪混入與該研究必要實作無關的 wording cleanup、dead-code cleanup、檔案搬遷或一般重構；若另有價值，必須拆成獨立 engineering change。只重新組合既有 capability 的研究，production code 原則上只修改 declarative spec；新增真正 primitive 時，合理 code 修改面原則上限於 declarative spec、該 capability 的唯一 owner／registry 與其 generic primitive invariant test。若 trainer、data provider、artifact planner、inference、renderer、menu、synthetic registry 等 generic consumer 仍需同步辨認新 identity，視為 modification-radius leakage，必須先收斂 reusable contract，而非接受 shotgun surgery。治理文件依 E5 另行同步，不計入 code modification radius；每輪交付須區分必要 code changes 與 incidental changes，不得以後者掩蓋架構驗收。Formal／synthetic guard 應驗證 owner、dependency 與 identity leakage 等泛化 invariant，不得硬編目前 MR 名稱或固定檔案數作為唯一合法答案。

22. **Model topology／multi-head composition 單一註冊原則**：model architecture identity、family、head／pooling topology、context requirement、active eligibility 與 model-spec builder routing 必須由 canonical architecture descriptor／registry 派生；`active` dispatcher、spec resolver、model factory、trainer、renderer 與 tests 不得各自維護同一 architecture-ID 清單或為新 variant 同步增加 identity branch。可由 model spec 表達的 head topology 必須 declarative 化，model implementation 依 head contract／family 建構。多 head training 必須以 reusable composition contract 宣告 combined output、head widths、target mapping、每 head pair-weight、由哪一 head 產生 detached pair context、loss weights 與 semantics metadata；generic trainer／training contract／renderer只消費該 composition，不得為每個 training objective 新增專屬 loss handler／schema／report branch。若新增一個同 family architecture 或 multi-head objective 仍需同步修改 architecture list、active dispatcher、spec registry、trainer、training contract、renderer或 synthetic registration 多處，視為 modification-radius leakage，必須先擴充 canonical descriptor／composition seam；generic synthetic 必須自動遍歷 registry 並檢查 architecture identity 不重複滲入 consumer。

## D. 交易與策略原則

1. 杜絕未來函數：任何候選、掛單、成交、停損、停利、延續判斷與統計，都不得偷看當下尚未知的未來資料。
2. 不確定時一律採最保守、最不利於績效的可執行解讀。
3. 受限使用者盤中無法進行新的人工交易決策，必須盤前完成買入掛單決策（包含買入標的、買入股數、買價上限）；盤中不得依新資訊重新選股、改配資金或進行 discretionary 操作，且無法在同天買入又賣出同一隻股票，也禁止當沖交易。成交後依盤前既定規則機械建立的保護單依 D5 處理，不視為新的盤中交易決策。
4. 由於 D3 的限制，交易當日資金運用必須在盤前鎖定；即使掛單未成交，盤中亦不得將該資金重新分配至其他股票。
5. 受限券商無法賣出未持倉股票，停損／停利掛單只能在買入成交後，依盤前既定規則與實際成交價機械建立；不得藉成交後資訊新增 discretionary 判斷或改變原策略規則。
6. 資金、權益、PnL、報酬率、勝率、EV 與 Round-Trip 定義，必須以扣除手續費、稅金後的淨值為準，且不得因半倉、顯示或報表需求分叉。
7. 執行條件與統計口徑必須全專案保持一致，包含風險 sizing、停損／停利、trailing stop、延續候選、失效、hit 判斷、觸發紀錄等。
8. 掛單、成交、觸發、執行、失效、達標與結算必須分層定義。


## E. 專案特例

1. `apps/portfolio_sim.py` 自動開瀏覽器暫時允許。
2. `doc/ToDo.md` 是使用者自行維護的私人工作筆記，不是專案 research backlog、current status、決策依據或 GPT 待辦來源。除非使用者當輪明確要求讀取／整理 `doc/ToDo.md`，GPT 不得主動讀取、引用、依賴或用其內容推導下一步；formal／GPT 最嚴格檢查亦排除該檔。一般使用者註解同屬使用者自有備忘／說明；若其文字被正式介面直接讀取，仍視為正式輸出。
3. 模型與策略的價格 input 現階段只使用 FinMind 提供的 canonical 還原價；專案不得自行維護第二套台股還原價計算 engine。`TaiwanStockPrice` raw price 允許完整保存於 Market Data archive，僅作原始市場 evidence、稽核、資料修復／比對與未來另行授權研究用途，保存 raw 不代表目前模型／策略可直接消費 raw。任何使用 current-vintage 還原價的歷史 feature／target／candidate 語意都必須符合 D1；若 representation 對供應商因後續公司行動造成的 retrospective adjustment 不具 invariance，必須另行完成 PIT legality 設計與驗證後才能使用，不得以目前 FinMind 還原價直接回填形成未來函數。
4. `doc/FINMIND_API_TOKEN.md` 為使用者本機私有憑證文件；其內容與是否被 `apps/package_zip.py` 收錄，暫時排除於 GPT 與 formal 最嚴格檢查及修正範圍之外。除非使用者另行要求，不得主動修改、移除、遮罩、加入 `.gitignore` 或調整打包器排除規則。
5. 凡分析、修改或測試 `breakout_quality`，開始新的 scientific identity、設計或程式修改前，必須依序讀取 `doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md` 與 `doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md`，並檢查現行 `config/`、runtime contract、已取得的正式工件與使用者最新研究決策。Registry 是 identity／namespace／current state 的唯一真理；Experiment Log 是已完成 evidence／result／decision 的唯一真理。**`doc/BREAKOUT_QUALITY_RESEARCH_QUEUE.md` 已由使用者決定永久退役**：不得重建、不得以歷史待辦或舊研究順序決定下一個方向；建議必須從現行科學問題、當前證據與必要性重新評估，未嘗試的構想不預占 scientific identity，亦不因討論而自動加入常駐文件。開始、取得結果或結案時，只更新各自 canonical owner；不得把討論中的猜想當成已執行 evidence。
6. `breakout_quality` architecture identity 只表示模型結構或輸入表示；training recipe、loss、optimizer、augmentation 等實驗差異由 experiment profile／research identity 管理。已淘汰的 historical architecture identity 不得直接復活為新的 current identity；若未來重新研究相同或近似結構，必須建立新的 research identity，並依 Registry 規則判定是否需要新的 architecture identity。
7. `breakout_quality` 的 OOS 可持續用於實驗結果比較、錯誤歸因、年度／regime 診斷、策略經濟效果評估及形成下一個實驗假設；人類研究決策可以受已觀察的迭代 OOS evidence 啟發，但不得把 OOS 資料或其統計直接輸入下一輪 computational fitting／selection pipeline。任何 Train／Validation、loss、gradient、early stopping、epoch 選擇、threshold／calibration、normalization、feature／label、sample weighting 或 hyperparameter optimization 均不得讀取或使用 OOS rows、labels、scores 或其統計量。若同一 OOS 被反覆使用，應標記為「迭代研究 OOS 證據」，不得宣稱是 untouched holdout。
8. `breakout_quality` 的 canonical identity namespace、prefix 與 allocation 規則由 Experiment Registry 唯一持有；程式 alias 僅作 compatibility。已使用的 scientific identity 永久保留，不得回收或憑記憶重用。
9. `breakout_quality` 的**研究假說／evidence Audit**必須遵守「最小必要證據」原則；本條不限制頂層第 2 條的 engineering full-project inspection。開始 research Audit 前必須先定義待決策問題、真正阻擋決策的關鍵不確定性，以及可停止分析並做出 GO／REJECT／NEXT EXPERIMENT 的條件；只有既有證據不足且新的 Audit 可能實質改變決策時才新增 Audit。Audit 次數不設硬性上限；一旦證據足以支持決策，就必須停止 Audit 鏈並直接進入決策或下一個受控實驗。
10. Daily-universal model research 必須維持 strategy-agnostic stock-day universe；breakout qualification、candidate membership、portfolio cash／holdings／selector state 等策略執行資訊，不得在沒有明確新 research decision 下滲入通用模型 Target／Input。若通用模型需要 risk／economic geometry，只能使用與模型目標必要且 decision-time 合法的資訊；strategy execution-specific geometry 不得無理由滲入 universal target。模型若使用 risk／cost／accounting semantics，必須直接重用 canonical risk/accounting SSOT，不得在 DL 模組建立第二套公式；歷史參數或其他 input 缺失不得以未來資訊回填。
11. `doc/CMD.md` 是使用者自行維護的個人常用指令筆記，**不是正式 CLI、目前設定或 Research/Trading 操作流程的 SSOT**。除非使用者當輪明確要求修改該檔，GPT 不得增刪、改寫、重新排序或自動同步其中任何指令與說明；不能為了讓文件與新程式一致而擅自改動。正式使用者入口仍以既有 application menu、source code 與 canonical config／contract 判定；必要時只能提示使用者該筆記可能過期，由使用者自行決定是否更新。
11. `breakout_quality` 的時間驗證、Strategy Compare 與 Robustness 額外遵守以下 scientific invariants：
   - **時間合法性**：任一 prediction／fold 的 training、target maturity、validation、normalization、sample weighting、parameter fitting 與其他自動決策，都必須在該 prediction／fold 的 information cutoff 前合法可得；evaluation 結果不得回頭影響當下模型或參數。
   - **Data-driven history**：Model training universe 與會影響 fitting 的歷史邊界必須由自身 data／feature／target legality 決定，不得被其他 workflow 的 historical cutoff 無意截斷；相關邊界屬 artifact reuse identity。
   - **Historical evidence**：Historical scientific identity、原始結果、controlled contrast 與當時語意不得因 current framework 改版而回頭改寫、刪除或偽裝成 current evidence。
   - **Compare Suite SSOT**：同一 comparison framework 的 arms、contrasts 與 policies 必須由單一 canonical Compare Suite 定義；single／multi、OOS／Rolling 不得各自維護第二份 scientific matrix。Multi-seed robustness 必須是 canonical single-seed suite 的 end-to-end 多 seed 延伸；除 seed 集合與跨-seed aggregate 外，不得私自改變 scientific arms／contrasts。
   - **Robustness seed semantics**：Benchmark seeds 必須由單一 config-driven specification 可重現地產生；同一 benchmark seed 必須一致傳遞至所有 declared seed-sensitive components，各 component 不得持有獨立值或隱含 fallback，也不得挑 best seed、fallback production／其他 seed，或把 benchmark sampling 與 production ensemble／consensus sampling 混用。
   - **Checkpoint / score reuse**：Model checkpoint reuse 必須依 scientific fitting identity；純 evaluation horizon／mode label 不得使相同 fit 成為不同 scientific model。Checkpoint 可依 fitting identity 共用，但 score 仍須依各自 score universe／evaluation contract 產生，不得跨 evaluation identity 誤用。完全相同且宣告 deterministic 的 fitting identity 若產生不相容 artifact，必須 fail-fast。
   - **Authorization / identity**：Scientific workflow 的執行資格必須由 canonical research identity／authorization contract 明確決定，且 authorization 優先於 dependency completeness。固定七項研究選單的顯示不得被授權、工件有無或模型名稱過濾；執行中的科學合法性檢查仍須保留，**不可把缺少可由合法 canonical Producer 建立的工件誤判為未授權而 BLOCKED**，也不得因前置依賴可建而繞過真正的 PIT／資料／Trading 授權。Logical evaluation source identity 可以表達不同 consumption semantics，但不得因此複製相同 physical truth。

12. **Research／Trading EV(R) 特例（使用者明確授權維持）**：Research 策略層 EV(A)／初始風險 R，與 Trading 帳務中心「實績 EV(R)」（平均單筆淨損益 ÷ 平均實際虧損金額）是兩個有意不同的統計口徑。各自原有公式、欄位名稱、標籤、畫面、報表與計算方式必須維持不變；不得因一般 SSOT 收斂而合併、互相替代、重新命名、新增並列 EV 欄位或回推缺少的初始風險資料。未經使用者另行授權，不得將此項列為需要修正的 SSOT-05。

## F. Scientific Contract／SOP／指標與報表治理

1. **研究設計與報表職責分離**：先核准 Model／Strategy 的共通 SOP 科學評估目的、必要指標、算法與適用範圍，才允許發布正式報表。模型自己的 Target／Score、獨立的共同 Outcome 評估、策略 Replay 的實際成交與現金績效不可混為一談；模型特定 Target、Head、策略特定 K／風控等不得暗中重定義共通 SOP。Global／Breakout 等應用範圍是否適用，由已核准的 evaluation contract 決定；相同範圍使用相同計算規則。
2. **正式人讀報表必備且唯一來源**：所有持久 Research／Strategy 結果均必須提供可直接閱讀的人讀摘要；RUN／REUSE／cache 共用同一 canonical result、metric 定義與 renderer，JSON／CSV／manifest 僅作詳細 evidence；不得因 work mode、model/profile、是否已產生工件而另建第二套同名報表或以缺值宣稱完成。
3. **指標單一 owner**：key、公式、來源、numerator／denominator、cohort、trade partition、Target join、時間／PIT、單位、精度、標籤、偏好方向與顏色語意由 canonical metric／diagnostics owner（包括 `core/report_metrics.py`、`core/report_style.py`）定義；報表 renderer 只消費 canonical result，不得重算 raw rows、偷用 proxy 替換 RCE 等診斷指標，或以正負號／emoji／燈號再定義好壞。模型驗證值須直接 REUSE 原 canonical validation evidence。
4. **Machine-readable Standard SOP Contract**：所有常駐／標準／可重複使用報表必須由一份 machine-readable report contract 固定 report ID、使用入口、正式章節／表格／欄位身分及順序、label、precision、unit、direction、適用範圍與必要 evidence；Console／Markdown／RUN／REUSE 等 renderer 只能讀它。單模型／多模型、Forward／Rolling／Robustness 的共通指標 schema 必須同源；各 mode 要求的必要 fold／seed 等完整性不可因精簡人讀版面而免除。
5. **正式改版需先授權且不可原地改寫**：新增、刪除、合併、拆分或重排任何常駐章節／欄位，或更改公式、metric semantics、報表分工、正式操作入口、格式、顏色、輸出位置，均須先取得使用者明確授權並發行新的 `report_id@vN`；既有版本與歷史 evidence 永久保留，不得用新 schema 或改寫舊 fingerprint 冒充舊版結果。純顯示改版亦走報表版本治理，但不得無故更動 fitted-model identity 或強迫重訓。
6. **版本紀錄與事前發布檢查**：`doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json` 為 append-only report version／fingerprint 真理，經批准的獨立版本定義文件及其 Hash 同步記錄；每次執行與正式發布前，比對指定 report ID、版本、目前 schema、相應定義文件及完整 evaluation/scientific identity，禁止只有測試完才發現未授權 drift。禁止藉修改程式 schema、Golden Hash、validator 使新版本自我核准；跨版本修改必須由使用者審查明確差異。**不要求新增本機 Git tag、第二套私人 CLI 或額外 Git 操作。** 同一可修改專案內的 ledger／Hash 僅能偵測正常流程漂移，不能宣稱提供獨立防篡改的密碼學授權。
7. **SOP 必測與發佈資格**：完整報表必須證明所有已核准必測指標、合法 PIT／maturity／score／price／seed／fold／replay／fee／accounting 證據齊備；合法評估因資料缺少／未成熟／樣本不相容無法完成時，必須 `INCOMPLETE/BLOCKED`，不得以 `N/A`、`-`、0、其他指標、歷史平均或 Extension 偽裝成已完成。品質差或 negative result 與證據未完成須分開狀態，不能因模型表現不佳將 SOP 本身標為不適用。模式間不同的必要證據由事前核准的 evaluation contract 規定，不能臨時修改。
8. **Audit 與 mode-specific evidence 分層**：模型／策略特定假說、Head、Label、K／Gate、First-Passage、Selection Quality 等深入比較，除非已正式核准為跨同類研究共通的必測 SOP，不得常駐加入 Standard SOP；一次性專題 Audit 自身版型／內容變更不需事前授權，但不得重定義正式 SOP 的 metric、流程、選單或 historical truth，且必須遵守 read-only／PIT／SSOT。結案後檢查依賴再退役專屬 implementation／registration／tests，保留 Registry、Log、完整成果與可重現歷史 evidence。**Rolling／Robustness 必要的跨 Fold／跨 Seed 分布及 same-seed 對照仍是正式必測，不是可刪的模型專屬 Extension。**
9. **報表輸出與執行進度分離**：長流程預設只顯示必要的 RUN／REUSE／DONE 進度及核心結果；pair-level、solver／repair、stale guard、timing 等詳細資訊保留在工件。Execution Summary 的耗時不視為科學績效指標，也不得決定策略比較名次。
10. **尊重既有共通與例外口徑**：正式選單／使用方式遵守 B13，research artifact／fitting identity 遵守 C9～C12，Trading／Research 交易與統計口徑遵守 D 章；Research EV(A) 與 Trading EV(R) 的有意不同定義依 E12 保留，不得因共通報表收斂而修改。已授權的 Model／Strategy SOP 現行與歷史章節、Golden Fingerprint 與具體指標由各版本 contract／Registry 持有，不在 PROJECT_SETTINGS 建立另一份 current identity。
11. **正式 SOP 只能依獨立研究證據改版**：先使用已核准 SOP，再以事前定義、OOS／seed／fold-aware、受控條件一致的 Read-only Audit 評估候選新指標與下游策略表現的穩健關係；需控制多重比較與迭代 OOS 限制，不得將這些 Audit 結果直接回饋訓練／選模，亦不得未經使用者授權就把相關性最高的指標加為永久 SOP。

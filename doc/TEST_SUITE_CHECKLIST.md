# Test Suite 收斂清單

目的：整理 `apps/test_suite.py` 與其本地動態測試組成步驟的覆蓋範圍、缺口、優先順序與建議落點，供後續逐項收斂。

文件分工：
- 本檔只維護測試清單、覆蓋狀態、測試入口映射與收斂紀錄。
- 原則、權責、前置順序與禁止事項一律引用 `doc/PROJECT_SETTINGS.md`；本檔不重寫該等規則。

範圍：
- 納入 `PROJECT_SETTINGS.md` 中的長期規則。
- 納入未明列於專案設定、但對正確性、穩定性、可重現性與可維護性必要的測試項目。
- 不納入暫時專案特例：`apps/portfolio_sim.py` 自動開瀏覽器、只使用還原價不考慮 raw。

狀態定義：
- `DONE`：已確認有明確測試覆蓋。
- `PARTIAL`：已有部分覆蓋，但仍有缺口。
- `TODO`：目前未見足夠覆蓋，應補。
- `N/A`：明確不納入正式長期 test suite。

優先級：
- `P0`：直接影響交易正確性、統計口徑或未來函數。
- `P1`：高價值補強，避免回歸或誤判。
- `P2`：品質與工具鏈補強。

索引管理：
- `Bxx` 為主表穩定追蹤 ID。
- `Txx` 為建議測試項目的 stable tracking ID namespace；出現結構調整時，仍以穩定追蹤與同輪同步更新為原則。

收斂原則：
1. 本檔主責維護主表、覆蓋狀態、對應測試入口與收斂紀錄；原則性規則不在本檔重寫。
2. 先補長期固定測試，再補可隨策略升級調整的測試；優先補 synthetic / unit / contract test，避免讓 GPT 端重跑本地完整動態流程。
3. test suite 應優先驗證規格、契約與 invariant，避免綁死當前 ML / DRL / LLM 策略實作細節。
4. 每完成一項，需同步更新本表狀態、對應測試入口與結果摘要；若新增測試導致模組責任改變，再更新 `doc/ARCHITECTURE.md` 與 `doc/CMD.md`。
5. 主表狀態為唯一真理來源；同步順序固定為先改主表，再同步 `T` / `G`；若同輪存在未完成缺口，再同步 `E`。任何 `Bxx` / `Txx` / 狀態變更，必須同一次 patch 更新完畢。
6. 摘要表只保留最小必要欄位：`T` 不重複抄寫完成日期；完成日期與狀態時間軸一律只記於 `G`。
7. `T` 每列只記一個 `Txx` 與一個測試入口；不得在同列混寫多個 validator 或 script。各摘要表固定依 ID 升冪排序；`G` 僅記錄實際狀態變更，且固定依日期升冪、同日再依 tracking ID 排序鍵整理：先比字首 namespace（如 `B` 在 `T` 前），再比數字段遞增，最後才比尾碼；新增或改寫 `G` 列時，不得直接 append 到當日區塊尾端，必須先抽出整個同日區塊、完成區塊內重排、以整段覆寫後再交付；若同日已有任何 `B` / `T` 混排列，禁止只插單列後假設局部正確，必須對整個同日區塊重新排序並逐列核對前後相鄰排序鍵；若新增列的排序鍵小於當前同日區塊尾列，禁止保留原尾端位置，必須插回正確位置後再檢查前一列 ≤ 當前列 ≤ 後一列；任何 `G` 列只能寫在 `## G. 逐項收斂紀錄` 表格內，檔案開頭第一個非空行固定為 `# Test Suite 收斂清單`；`G` 備註欄最多只能保留一個 code/path/test entry，若同輪涉及多個檔案或測試，僅保留單一代表 entry，其餘改寫為一般文字；純補充說明改寫為表格外文字，不得再寫 `DONE -> DONE`、`PARTIAL -> PARTIAL` 等無狀態變更列；同一 tracking ID 的 `NEW -> *` 只能出現在該 ID 首次收斂紀錄，且 `G` 中任何 backticked `validate_*` 名稱都必須是目前仍存在的有效 validator；交付前必須再做一次機械核對：確認該日期區塊所有列已依上述排序鍵單調不下降，否則不得交付。

## A. 分層原則

### A1. 長期固定測試
這類測試不應因策略從規則式升級到 ML / DRL / LLM 而改變；主要驗證不可變規格、跨工具契約與基礎品質屬性。

應優先納入：
- 交易規則 invariant。
- 統計口徑一致性。
- 費稅後淨值與 Round-Trip 定義。
- 候選 / 掛單 / 成交 / miss buy 分層。
- 禁止未來函數。
- fail-fast、壞輸入、資料韌性。
- CLI / artifact / schema 契約。
- coverage baseline、重跑一致性、效能 baseline。

### A2. 可隨策略升級調整的測試
這類測試可隨策略演進調整，但仍應保留，以避免實驗型改動破壞對外介面與最基本可用性。

只建議驗證：
- 模型輸入 / 輸出 schema。
- 同 seed 的可重現性。
- ranking / scoring 輸出的範圍、型別、排序穩定性。
- optimizer / scanner / reporting 的最低可用性。

避免寫成：
- 某模型分數必須等於固定小數。
- 某版本一定挑到某一檔股票。
- 驗證大量內部中間變數與執行順序。
- 在測試中重寫一份策略邏輯。

## B. 長期固定測試清單

### B1. 專案設定對應清單（不含暫時特例）

| ID | 優先級 | 項目 | 目前判定 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|---|
| B01 | P0 | 杜絕未來函數 | DONE | 已以 prev-day-only PIT case 與 setup-index prev-day-only case 共同釘死盤前排程只能讀前一日訊號，並驗證 setup index 不得偷看當日資料 | `tools/validate/synthetic_history_cases.py` |
| B02 | P1 | 同 K 棒停利/停損取最壞停損 | DONE | 已有明確 synthetic case | 既有 synthetic case |
| B03 | P0 | 權益曲線、資金、PnL 一律為扣費扣稅後淨值 | DONE | 已新增直接手算對帳案例，逐欄位檢查 entry cash / entry equity / exit pnl / final equity / total return | `tools/validate/synthetic_take_profit_cases.py` |
| B04 | P1 | 半倉停利只算現金回收，尾倉才算完整 Round-Trip | DONE | 已新增直接案例，斷言半倉列不得提前帶完整 `該筆總損益`，完整 Round-Trip 僅在尾倉結算列完成 | `tools/validate/synthetic_take_profit_cases.py` |
| B05 | P0 | 只能盤前掛單；盤中不得新增/改單/換股 | DONE | 已新增直接禁止盤中改單 case，並以 failed fill / same-day sell / rotation case 共同覆蓋盤中新增與換股 | `tools/validate/synthetic_flow_cases.py` |
| B06 | P0 | 不得當沖；買入當日不可賣出；當日賣出回收資金不得當日再投入 | DONE | 已新增買入當日不可賣出的直接 case；同日賣出後不得再投入則由既有 same-day sell block / rotation T+1 case 覆蓋 | `tools/validate/synthetic_flow_cases.py` |
| B07 | P0 | 未成交不得同日盤中自動改掛其他股票 | DONE | 已新增直接 failed fill 後不得同日換股 case | `tools/validate/synthetic_flow_cases.py` |
| B08 | P0 | 停利/停損只能對已持有部位預先設定 | DONE | 已新增 zero-qty position direct assertion，確認無持倉時 stop/tp / indicator sell 不得產生任何 exit event | `tools/validate/synthetic_take_profit_cases.py` |
| B09 | P1 | 候選、掛單、成交、miss buy、歷史績效統計必須分層定義 | DONE | 已新增 candidate / filled / missed-buy 三層直接案例，並補 non-candidate setup 不得 seed / revive extended candidate，釘死狀態不得混用 | `tools/validate/synthetic_flow_cases.py` |
| B10 | P1 | 單股回測不得用自身歷史績效 filter 作為買入閘門；history filter 僅用於投組層/scanner | DONE | 已新增 cross-tool case，直接對照單股回測仍成交、scanner 端仍拒絕非 candidate | `tools/validate/synthetic_history_cases.py` |
| B54 | P1 | 單股回測與投組回測必須一致使用複利資金口徑，不得保留 hidden fixed-cap sizing 分支 | DONE | 已新增 direct synthetic case，直接釘死即使經過連續獲利，單股回測後續 sizing、`asset_growth` 與 `score` 都必須反映複利資金；不得再以固定 `initial_capital` 壓回單股 sizing | `tools/validate/synthetic_history_cases.py`, `core/capital_policy.py`, `core/backtest_core.py`, `core/backtest_finalize.py`, `tools/debug/backtest.py` |

### B2. 未明列於專案設定，但正式 test suite 應納入

| ID | 優先級 | 類別 | 項目 | 目前判定 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|---|---|
| B11 | P1 | 契約 | 跨工具 schema / 欄位語意一致 | DONE | 已補 missed sell / trade log / stats 一致性，以及 validate / issue report / optimizer profile / local regression summaries（preflight / dataset prepare / chain / ml smoke / meta quality / master summary）的 CSV / XLSX / JSON contract；另已釘死 early-failure `master_summary.json` 的 `payload_failures` 必須維持與正常路徑一致的語意，且不得把合法 FAIL payload 誤標成 `summary_unreadable`，主要跨工具 schema 與欄位語意已收斂 | contract tests under `tools/validate/` |
| B12 | P1 | 決定性 | 同資料、同參數、同 seed 結果可重現 | DONE | 已補 `run_ml_smoke.py` fixed-seed 雙跑、`run_chain_checks.py` scanner reduced snapshot 雙跑 digest、`validate_scanner_worker_repeatability_case` 與 `validate_scan_runner_repeatability_case`，正式入口與 scanner 入口重跑一致性已收斂 | `tools/local_regression/`, `tools/validate/synthetic_regression_cases.py` |
| B13 | P1 | 邊界值 | 數值穩定性、rounding、tick、odd lot | DONE | 已新增 `price_utils` / `history_filters` / `portfolio_stats` unit-like 邊界案例，覆蓋 tick、稅費、sizing、全贏/全輸與空序列 | `tools/validate/synthetic_unit_cases.py` |
| B14 | P1 | 韌性 | 髒資料、缺欄位、NaN、日期亂序、OHLC 異常 | DONE | 已新增資料清洗 expected behavior / fail-fast / `load_clean_df` 整合案例，直接釘死髒資料修正、欄位缺失、NaN、日期亂序、OHLC 異常與清洗後列數行為 | `tools/validate/synthetic_data_quality_cases.py`, `core/data_utils.py`, `tools/validate/real_case_io.py` |
| B15 | P1 | 錯誤處理 | 壞 JSON、缺參數、缺檔、匯入失敗、API 失敗時訊息可定位 | DONE | 已補 `params_io` / `module_loader` / `preflight_env` 的 module 級錯誤路徑，並補 downloader universe fetch 全失敗與 screening 初始化失敗的 fatal error path，錯誤訊息與 issue log 已可定位 | `core/params_io.py`, `tools/validate/preflight_env.py`, `tools/validate/module_loader.py`, `tools/validate/synthetic_error_cases.py` |
| B16 | P2 | CLI | 互斥參數、預設值、help 與實作一致 | DONE | 已補 dataset wrapper、local regression / no-arg CLI 與剩餘直接入口 CLI 契約，覆蓋 help、預設 passthrough、`--only` / `--steps` 正規化、未知參數、缺值、空值、位置參數拒絕，以及 `run_all.py` 參數錯誤 stderr usage 必須同步列出 `meta_quality` | `tools/validate/synthetic_cli_cases.py`, `apps/*.py`, `core/runtime_utils.py` |
| B17 | P2 | I/O | 輸出工件、bundle、retention、rerun 覆寫行為 | DONE | 已補 validate summary / optimizer profile / issue report 的 output contract，以及 bundle/archive/root-copy/retention lifecycle、PASS/FAIL bundle selection、artifacts manifest、rerun 覆寫內容契約；另補 quick gate 發生 runtime error 時仍必須落出正式 FAIL summary / console artifact 契約 | `core/output_paths.py`, `core/output_retention.py`, `tools/validate/synthetic_contract_cases.py`, `tools/local_regression/run_quick_gate.py` |
| B18 | P1 | 回歸 | 重跑一致性、狀態汙染、cache 汙染 | DONE | 已補 `run_chain_checks.py` 雙跑 digest、`run_ml_smoke.py` fixed-seed 雙跑、`validate_optimizer_raw_cache_rerun_consistency_case` 與 `validate_run_all_repeatability_case`，正式入口與 cache 汙染隔離已收斂 | `tools/local_regression/`, `tools/optimizer/raw_cache.py`, `tools/validate/synthetic_regression_cases.py` |
| B19 | P2 | 效能 | reduced dataset 時間基線、optimizer 每 trial 上限、記憶體回歸 | DONE | 已將 quick gate / consistency / chain checks / ml smoke / meta quality / total suite duration、optimizer 平均 trial wall time 與各步驟 / meta quality traced peak memory 全數納入 `run_meta_quality.py` 正式 gating，並同步回寫 step summaries / `meta_quality_summary.json` / `apps/test_suite.py` 摘要；step summary 的 duration 欄位已補齊，consistency / chain / total budget 依實測 reduced baseline 校正；另已把 chain checks 改為同輪共用 prepared market context、prepared scanner snapshot、cached single-backtest stats，並將 replay counts 直接併入第一次 timeline 主流程；`portfolio_sim` 驗證也改直接共用 `validate_one_ticker()` 已產生的 prepared df / standalone logs，不再重讀 CSV、重新 sanitize / prepare，且單檔 portfolio context 的 `fast_data / sorted_dates / start_year` 改為共用，不再在 real-case 與 tool check 各自重建；`validate_consistency` 執行 synthetic suite 時已同步寫出 coverage artifacts，`run_meta_quality.py` 可直接重用同輪 artifacts，不再重跑一次 synthetic coverage suite | `tools/local_regression/`, `core/runtime_utils.py`, `apps/test_suite.py` |
| B20 | P2 | 文件 | `doc/CMD.md` 指令與實作一致 | DONE | 已新增 CMD Python 指令契約案例，校驗腳本存在、`--dataset` / `--only` / `--steps` 參數值合法，並確認文件中的專案腳本已納入 quick gate help 檢查 | `tools/validate/synthetic_meta_cases.py` |
| B21 | P2 | 顯示 | 報表欄位、排序、百分比格式與來源一致 | DONE | 已補 scanner header / start banner / summary、strategy dashboard、validate console summary、issue Excel report schema、portfolio yearly/export report，以及 `apps/test_suite.py` 在 PASS / FAIL / manifest-blocked / partial-selected-steps / preflight-failed / dataset-prepare-failed / summary-unreadable 下的人類可讀摘要契約；另補 score header 顯示契約，釘死 `評分模型` 與 `評分分子` 必須分欄顯示、不得再以 `/ 分子` 混入同一括號；並補 checklist status vocabulary sync 與 meta quality coverage line/branch/min-threshold/missing-zero-target guard 摘要顯示，且以 `run_all.py` contract 釘死 preflight 早退時 dataset_prepare 仍需標記為 `NOT_RUN`，避免 real path 誤落成 `missing_summary_file` | `tools/validate/synthetic_display_cases.py`, `tools/validate/synthetic_reporting_cases.py`, `tools/validate/synthetic_contract_cases.py`, `core/display.py`, `tools/scanner/reporting.py`, `tools/portfolio_sim/reporting.py`, `apps/test_suite.py`, `tools/local_regression/run_all.py` |
| B22 | P2 | 覆蓋率 | line / branch coverage 報表 | DONE | 已將 `run_meta_quality.py` 的 synthetic coverage suite、formal helper probe、key target presence/hit 與 manifest 化 line / branch minimum threshold gate 收斂為正式路徑，並同步回寫 `meta_quality_summary.json` / `apps/test_suite.py` 摘要顯示 | `tools/local_regression/run_meta_quality.py`, `tools/local_regression/common.py`, `apps/test_suite.py` |
| B23 | P1 | Meta | checklist / 測試註冊 / 正式入口一致性 | DONE | 已補 synthetic 主入口遺漏註冊案例，並新增 imported / defined `validate_*` case、formal pipeline registry / formal-entry / run_all / preflight / test_suite 一致性 formal guard，以及 `core/` / `tools/` 不得反向 import `apps/` 的分層 guard；正式步驟單一真理來源已收斂到 `tools/local_regression/formal_pipeline.py` | `tools/validate/synthetic_meta_cases.py`, `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_cases.py`, `tools/local_regression/formal_pipeline.py` |
| B24 | P1 | Meta | known-bad fault injection：關鍵規則故意破壞後測試必須 fail | DONE | 已新增 meta fault-injection case，直接對 same-day sell、same-bar stop priority、fee/tax、history filter misuse 注入 known-bad 行為，並驗證既有測試會產生 FAIL | `tools/validate/synthetic_meta_cases.py` |
| B25 | P1 | Meta | independent oracle / golden cases：高風險數值規則不可只與 production 共用同邏輯 | DONE | 已新增獨立 oracle golden case，對 net sell、position size、history EV、annual return / sim years 以手算或獨立公式對照 production | `tools/validate/synthetic_unit_cases.py` |
| B26 | P1 | Meta | checklist 是否已足夠覆蓋完整性（包含 test suite 本身） | DONE | 已補主表 / `T` / `G` 收斂紀錄完整同步 formal guard，並阻擋 convergence 紀錄失同步、`T` 以 `+`、`/`、`,` 或多個 code reference 混寫多個測試入口、`G` 備註欄混寫多個測試入口、`G` transition 缺少合法狀態轉移格式、同一 tracking ID 在已有歷史列後重複寫 `NEW -> *`、`G` 備註欄殘留已退役 validator 名稱、檔案開頭第一個非空行漂移，以及 legacy `D` / `F1` 區不得回流；checklist 自身完整性已納入正式 gate | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/meta_contracts.py`, `doc/TEST_SUITE_CHECKLIST.md` |
| B27 | P1 | Meta | 禁止循環依賴（模組層級 import cycle） | DONE | 已補 project import graph cycle guard，直接阻擋 `apps/` / `core/` / `tools/` 間的模組層級循環依賴（含函式內 import） | `tools/validate/synthetic_meta_cases.py`, `tools/validate/meta_contracts.py` |
| B28 | P1 | 覆蓋率 | key coverage targets 應包含核心交易模組 | DONE | 已將 `core/backtest_core.py`、`core/backtest_finalize.py`、`core/portfolio_engine.py`、`core/position_step.py`、`core/portfolio_entries.py`、`core/portfolio_exits.py`、`core/portfolio_ops.py`、`core/trade_plans.py`、`core/entry_plans.py`，以及直接承接候選分層 / PIT 歷史績效 / 延續訊號規則的 `core/portfolio_candidates.py`、`core/portfolio_fast_data.py`、`core/extended_signals.py`、`core/signal_utils.py` 納入 `COVERAGE_TARGETS`，並新增 completeness guard，直接阻擋核心交易模組未入列 | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py` |
| B29 | P1 | 覆蓋率 | critical files 應具備 per-file line / branch minimum gate | DONE | 已對 `core/backtest_core.py`、`core/portfolio_engine.py`、`core/position_step.py`、`core/portfolio_exits.py` 建立 per-file line / branch minimum coverage guard，直接阻擋 overall coverage 過關但核心檔仍偏薄 | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py` |
| B30 | P1 | 覆蓋率 | overall coverage minimum threshold 應逐步提高，branch 優先 | DONE | 已將正式 coverage 基線提高為 `line 55% / branch 50%`，並新增 threshold floor guard，阻擋門檻回退到舊的 `50 / 45`；branch 與 line 的 gap 也已納入 formal policy 檢查 | `tools/local_regression/common.py`, `tools/local_regression/manifest.json`, `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py` |
| B31 | P1 | 覆蓋率 | entry path 關鍵模組應納入 critical file per-file coverage gate | DONE | 已將 `core/portfolio_entries.py` 與 `core/entry_plans.py` 納入 `CRITICAL_COVERAGE_TARGETS`，並新增 entry-path completeness / importability guard，避免 only-exit / engine critical gate 漏掉實際高風險進場邏輯 | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_cases.py` |
| B32 | P1 | 覆蓋率 | critical file per-file minimum threshold 應具備 stage-2 floor guard，branch 優先 | DONE | 已將 `coverage_critical_line_min_percent` / `coverage_critical_branch_min_percent` 正式基線提高為 `30% / 25%`，並新增 critical threshold floor guard，阻擋 critical per-file 門檻回退到舊的 `25 / 20`；critical branch 與 line 的 gap 亦已納入 formal policy 檢查 | `tools/local_regression/common.py`, `tools/local_regression/manifest.json`, `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py` |
| B33 | P1 | I/O | reduced dataset 應具備 member / content fingerprint gate | DONE | 已為 reduced dataset 補上 `csv_members_sha256`、`csv_content_sha256`、`csv_total_bytes` 與 `fingerprint_algorithm`，並將 fingerprint 同步寫入 dataset prepare summary；另補 `run_all.main()` dataset prepare PASS 主路徑 contract，避免新欄位或常數接線缺漏時在正式入口退化成 NameError | `tools/local_regression/common.py`, `tools/local_regression/run_all.py`, `tools/validate/synthetic_contract_cases.py` |
| B34 | P1 | I/O | summary / manifest / artifact 寫檔應採 atomic write，避免 partial overwrite | DONE | 已將 `write_json` / `write_text` / `write_csv` 收斂為同一 atomic replace helper，並補 replace-failure recovery、transient retry、cleanup-failure root-exception preservation，以及 `write_local_regression_summary()` 正式路徑的 atomic-write contract，要求舊內容不得被半寫覆蓋、temp 檔必須清乾淨；若 temp cleanup 失敗也不得覆蓋原始寫檔例外，且 Windows 暫時性 share violation 不得直接把 step 誤判成 FAIL | `tools/local_regression/common.py`, `tools/validate/reporting.py`, `tools/validate/synthetic_contract_cases.py` |
| B35 | P1 | 覆蓋率 | test suite orchestrator modules 應納入 coverage targets | DONE | 已將 `tools/local_regression/common.py`、`formal_pipeline.py`、`meta_quality_targets.py`、`meta_quality_coverage.py`、`run_meta_quality.py`、`run_all.py`、`tools/validate/preflight_env.py`、`core/test_suite_reporting.py`、`apps/test_suite.py` 納入 `TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS`，並新增 completeness / importability guard，避免測試編排層、formal preflight 檢查層與 coverage 治理層退化卻仍以 coverage 過關 | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py` |
| B36 | P1 | I/O | artifacts manifest 應具備 sha256，不可只靠 size_bytes | DONE | 已為每個 artifact manifest entry 補上 `sha256`，並新增 contract 對照實際檔案 hash，避免同大小內容漂移被 size 假象掩蓋 | `tools/local_regression/common.py`, `tools/validate/synthetic_contract_cases.py` |
| B37 | P2 | Meta | synthetic registry 應具備 metadata contract（layer / cost / impacted modules） | DONE | 已將 synthetic registry 升級為 metadata registry，補上 layer / cost class / impacted modules，並新增 metadata contract；同時保留 `get_synthetic_validators()` 相容 façade，避免 formal 入口、coverage 與 checklist guard 斷裂 | `tools/validate/synthetic_cases.py`, `tools/validate/synthetic_meta_cases.py` |
| B38 | P1 | 覆蓋率 | formal pipeline step entry wrappers 應納入 coverage targets | DONE | 已將 `tools/local_regression/run_quick_gate.py` 與 `tools/validate/cli.py` 納入 `COVERAGE_TARGETS`，並新增 formal-step completeness / importability guard，避免正式步驟 wrapper 退化卻 coverage baseline 未偵測 | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py` |
| B39 | P1 | 覆蓋率 | split formal-step implementation modules 應納入 coverage targets | DONE | 已將 `tools/validate/main.py` 納入 `COVERAGE_TARGETS`，並新增 formal-step implementation completeness / importability guard，避免 `tools/validate/cli.py` 仍在、但 consistency 真正執行本體退化卻 coverage baseline 未偵測 | `tools/local_regression/meta_quality_targets.py`, `tools/validate/synthetic_meta_cases.py` |
| B40 | P1 | Meta | `PeakTracedMemoryTracker` lifecycle 必須用 context manager 統一管理 | DONE | 已將 `run_chain_checks.py`、`run_meta_quality.py`、`run_ml_smoke.py`、`run_quick_gate.py`、`tools/validate/main.py` 收斂為 `with PeakTracedMemoryTracker() as tracker:`，並新增 static guard，直接阻擋手動 `__enter__` / `__exit__` 導致 early return / runtime error 路徑漏掉 `__exit__` | `core/runtime_utils.py`, `tools/local_regression/*.py`, `tools/validate/main.py`, `tools/validate/synthetic_meta_cases.py` |
| B41 | P1 | 文件 | `doc/CMD.md` 與 `doc/ARCHITECTURE.md` 不得殘留已移除的 app 測試入口與手動刪檔指引 | DONE | 已移除 `apps/local_regression.py` / `apps/validate_consistency.py` 的殘留文件指引，並新增 formal guard，直接阻擋文件再出現已移除入口或「手動刪除」式 app 測試入口清理說明 | `doc/CMD.md`, `doc/ARCHITECTURE.md`, `tools/validate/meta_contracts.py`, `tools/validate/synthetic_meta_cases.py` |
| B42 | P1 | Meta | app thin wrapper 的 lazy public exports 不得發生 `LAZY_EXPORTS` / `__all__` 漏同步 | DONE | 已補 thin wrapper export contract，直接阻擋 `apps/portfolio_sim.py`、`apps/vip_scanner.py` 出現 lazy export 重複、`__all__` 漏列或 lazy symbol 無法解析，避免對外 façade 可 `getattr` 但 public export contract 漏同步 | `apps/portfolio_sim.py`, `apps/vip_scanner.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_cases.py` |
| B43 | P1 | I/O | `apps/package_zip.py` 正式入口必須驗證 root 舊 package ZIP 歸檔、保留 root bundle copy，且輸出 ZIP 不得夾帶 Python 快取 | DONE | 已補 `package_zip` runtime contract，直接釘死 root 既有非 bundle 舊 ZIP 必須移入 `arch`、`to_chatgpt_bundle_*.zip` 不得被移入 `arch`，且新 ZIP 僅可包含 tracked/untracked 非忽略檔，不得夾帶 `__pycache__/` 或 `*.pyc` | `apps/package_zip.py`, `tools/validate/synthetic_cli_cases.py`, `tools/validate/synthetic_cases.py` |
| B44 | P1 | Meta | `quick_gate` 不得移除裸 `except` static guard | DONE | 已補 `quick_gate` bare-except guard contract，直接釘死 `run_static_checks()` 必須保留 `bare_except_scan`，且遇到裸 `except` 時必須回報 FAIL 與命中文件 | `tools/local_regression/run_quick_gate.py`, `tools/validate/synthetic_contract_cases.py`, `tools/validate/synthetic_cases.py` |
| B45 | P1 | I/O | `quick_gate` 不得移除 output path / outputs root / log path guard | DONE | 已補 `quick_gate` output-path guard contract，直接釘死正式入口 summary 必須保留 `output_path_contract`、`outputs_root_layout`、`log_path_contract` 關鍵步驟，且 guard FAIL 時必須正確傳遞到 `failed_steps`；另已補 `resolve_log_dir` / `build_timestamped_log_path` / `write_issue_log` 的 outputs-root create-path guard，並釘死 `resolve_log_dir("outputs")` 不得先落入 generic root-dir 錯誤，必須維持 outputs-root 專屬拒絕語意 | `tools/local_regression/run_quick_gate.py`, `core/output_paths.py`, `core/log_utils.py`, `tools/validate/synthetic_contract_cases.py`, `tools/validate/synthetic_cases.py` |
| B46 | P1 | 錯誤處理 | formal pipeline 關鍵 fallback / console tail 不得靜默吞掉非 cleanup I/O 例外 | DONE | 已補 dataset prepare fallback summary write traceability 與 console tail read-error traceability contract，直接釘死 `run_all.py` fallback 寫檔失敗必須回寫 `fallback_write_errors` / stderr，且 `gather_recent_console_tail()` 讀檔失敗不得靜默略過 | `tools/local_regression/run_all.py`, `tools/local_regression/common.py`, `tools/validate/synthetic_contract_cases.py`, `tools/validate/synthetic_cases.py` |
| B55 | P1 | 契約 | consistency / 單檔 parity 驗證路徑必須維持複利資金，單股、單檔投組與 portfolio_sim 不得再強制切回 fixed-capital | DONE | 已補 direct synthetic contract，直接釘死 consistency parity params 必須保留 compounding，且 single backtest / 單檔 portfolio timeline / portfolio_sim prepared 在虧損後與獲利後都必須一致反映複利資金，candidate sizing 與實際 entry budget 兩條路徑不得分叉 | `tools/validate/synthetic_contract_cases.py`, `tools/validate/real_case_runners.py`, `tools/validate/scanner_expectations.py`, `core/capital_policy.py`, `core/portfolio_entries.py` |
| B56 | P1 | 契約 | scanner 參考投入 / 掛單股數必須使用 `scanner_live_capital`，不得再回退到 `initial_capital` 或回測複利資金 | DONE | 已補 direct synthetic contract，直接釘死 scanner projected qty / proj_cost / 顯示訊息都必須跟 `scanner_live_capital` 一致，且 `initial_capital` 與 scanner 資金來源不得混用 | `tools/validate/synthetic_contract_cases.py`, `core/capital_policy.py`, `core/price_utils.py`, `tools/scanner/stock_processor.py` |
| B57 | P1 | 契約 | 系統評分分子必須可在總報酬率與年化報酬率之間切換，且 score 公式不得把分子 / 分母切換混成不同 mode | DONE | 已補 direct synthetic contract，直接釘死 `SCORE_NUMERATOR_METHOD` 可在 `ANNUAL_RETURN` / `TOTAL_RETURN` 間切換，兩者都必須共用同一個 `|MDD| + 0.0001` 分母，且 `LOG_R2` 品質單調性不得被新分子選項破壞 | `tools/validate/synthetic_strategy_cases.py`, `config/training_policy.py`, `core/portfolio_stats.py` |
| B58 | P1 | 契約 | 保留中的單一模式參數不得接受不支援值後靜默忽略；目前 `use_compounding` 必須 fail-fast 拒絕 `False` | DONE | 已補 direct guardrail case，直接釘死 `use_compounding=False` 無論 JSON 載入、dataclass 建立或 setattr 都必須報錯，避免保留參數變成靜默 no-op | `tools/validate/synthetic_guardrail_cases.py`, `core/strategy_params.py` |
| B59 | P1 | Meta | 關鍵費稅 / sizing / 資金來源 helper 必須維持單一真理來源，不得在其他模組重複定義 | DONE | 已補 static contract，直接釘死 `calc_entry_price`、`calc_net_sell_price`、`calc_position_size`、`calc_initial_risk_total`、`resolve_single_backtest_sizing_capital`、`resolve_portfolio_sizing_equity`、`resolve_portfolio_entry_budget`、`resolve_scanner_live_capital` 只能定義在各自 canonical 模組，避免費稅 / sizing / 資金來源規則分叉 | `tools/validate/meta_contracts.py`, `tools/validate/synthetic_meta_cases.py`, `core/price_utils.py`, `core/capital_policy.py` |
| B60 | P1 | Meta | `PROJECT_SETTINGS.md` 必須明確禁止 GPT 端重跑任何動態測試，且不得繞過 `apps/test_suite.py` 直接執行 formal step / validator / 腳本 / 函式 | DONE | 已補 explicit boundary wording 與 static contract，直接釘死文件必須同時宣告『不得重覆執行 `apps/test_suite.py` 已涵蓋項目 / 不得執行任何動態測試』與『不得繞過正式入口直接執行其涵蓋的 formal step、validator、腳本或函式』 | `doc/PROJECT_SETTINGS.md`, `tools/validate/meta_contracts.py`, `tools/validate/synthetic_meta_cases.py` |
| B61 | P1 | Meta | 關鍵資金 / 參數 / 政策契約模組必須納入 coverage targets，避免 helper / guardrail 雖有測項但 coverage 退化時無法被 meta quality 擋下 | DONE | 已補 static coverage-target contract，直接釘死 `core/capital_policy.py`、`core/strategy_params.py`、`core/params_io.py`、`config/execution_policy.py`、`config/training_policy.py` 必須列入 coverage targets，且關鍵公開符號需可匯入 | `tools/local_regression/meta_quality_targets.py`, `tools/validate/synthetic_meta_cases.py` |
| B62 | P1 | Meta | `doc/TEST_SUITE_CHECKLIST.md` 檔案開頭第一個非空行必須固定為 `# Test Suite 收斂清單`，避免 checklist parser / guard 前提漂移 | DONE | 已補 direct checklist contract，直接釘死檔案前置非空內容不得漂移；若有人在標題前插入額外說明、註記或其他文字，formal guard 必須 fail | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `doc/TEST_SUITE_CHECKLIST.md` |
| B63 | P1 | Meta | synthetic meta validators 讀取 `summarize_result` 額外欄位時，必須相容 flattened top-level payload；不得直接依賴巢狀 `result["extra"]` schema | DONE | 已補 shared summary-value accessor 與 static contract，直接釘死 `tools/validate/synthetic_meta_cases.py` 不得再用直接 `.get("extra", {}).get(...)` 讀取 meta summary 欄位，避免 formal synthetic suite 因 payload schema 讀法錯誤出現假失敗 | `tools/local_regression/common.py`, `tools/validate/synthetic_meta_cases.py` |
| B64 | P1 | Meta | checklist 各摘要表（`E1` / `E2` / `E3` / `T`）必須固定依 tracking ID 升冪排序，避免同步表雖內容正確但順序漂移而破壞機械比對 | DONE | 已補 direct checklist contract，直接釘死 `E1` / `E2` / `E3` / `T` 不能只做集合相等；若任何摘要表 row order 逆序，formal guard 必須 fail 並指出對應表名與前後 ID | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `doc/TEST_SUITE_CHECKLIST.md` |
| B65 | P1 | CLI | `apps/package_zip.py` 必須支援外部參數一鍵執行 commit → zip → test_suite，且 ZIP 檔名必須反映 commit 後 HEAD | DONE | 已補 orchestration contract，直接釘死 `--commit-message` 會先 `git add -A` + `git commit -m ...`、zip 產物必須使用 commit 後 HEAD short sha，且 `--run-test-suite` 必須在打包後才執行 `apps/test_suite.py` | `apps/package_zip.py`, `tools/validate/synthetic_cli_cases.py`, `tools/validate/synthetic_cases.py` |
| B66 | P1 | GUI | `apps/gui.py` 必須作為單一 GUI 啟用入口，且 workbench panel registry / 單股回測後端 / Excel+HTML artifact 契約必須穩定 | DONE | 已補 GUI workbench contract，直接釘死 `apps/gui.py` 必須對應到 `tools.gui.main`、workbench 必須註冊單股回測檢視 panel，且 debug backend 必須同時產生 `Debug_TradeLog_<ticker>.xlsx` 與 `Debug_TradeChart_<ticker>.html` | `apps/gui.py`, `tools/gui/workbench.py`, `tools/validate/synthetic_contract_cases.py` |
| B67 | P1 | GUI | 單股 debug chart hooks 在 `export_chart=False` 時必須完全 no-op；`chart_context=None` 不得造成 runtime side effect 或中斷單股 / chain / synthetic suite | DONE | 已補 direct contract，直接釘死 `run_debug_backtest()` 走無圖模式時仍必須正常產生交易明細；chart marker / active level hooks 必須接受 `chart_context=None` | `tools/debug/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B68 | P1 | 契約 | validation / tool-check payload 的 `module_path` 必須統一為 repo-relative、forward-slash 穩定路徑；不得回傳機器相依絕對路徑 | DONE | 已補 direct contract 與 shared normalizer，直接釘死 absolute / backslash path 都必須正規化為穩定 repo-relative path，避免 bundle 與 synthetic 在不同 OS/工作目錄出現假失敗 | `tools/validate/module_loader.py`, `tools/validate/synthetic_contract_cases.py` |
| B69 | P1 | 契約 | shared path normalizer 必須接受 `pathlib.Path` 等 path-like 輸入；不得假設 `PROJECT_ROOT` 或 path payload 一定是 str 才能正規化 | DONE | 已補 direct contract，直接釘死 `normalize_project_relative_path()` 對 `Path` 輸入也必須穩定回傳 repo-relative、forward-slash 路徑，避免 synthetic / local regression helper 再因字串 API 假設發生 runtime regression | `tools/validate/module_loader.py`, `tools/validate/synthetic_contract_cases.py` |
| B70 | P1 | 契約 | shared module-loader path helpers 必須接受被 patch 成字串的 `PROJECT_ROOT`；不得假設 module-level root 永遠是 `Path`，否則 synthetic / error-path 測試環境會產生 helper 自身回歸 | DONE | 已補 direct contract，直接釘死 `build_project_absolute_path()` 與 `normalize_project_relative_path()` 在 `PROJECT_ROOT` 被 patch 成字串時仍必須正常組路徑並回傳穩定 repo-relative path，避免 shared helper 與既有 `patch.object(module_loader, "PROJECT_ROOT", str(...))` 測試環境互撞 | `tools/validate/module_loader.py`, `tools/validate/synthetic_contract_cases.py` |
| B71 | P1 | GUI | GUI 單股回測檢視必須內嵌大型 K 線圖，且初始視窗與縮放後 Y 軸比例都必須依可視 X 範圍自動重算；不得因全歷史資料或離屏極值導致圖形失真 | DONE | 已補 direct contract，直接釘死 workbench panel 必須宣告 inline chart backend、debug backend 必須回傳 chart payload，且 chart helper 的預設視窗與可視區間價量範圍計算必須忽略離屏極值並可建立 2 軸內嵌 figure | `tools/debug/charting.py`, `tools/gui/single_stock_inspector.py`, `tools/validate/synthetic_contract_cases.py` |
| B72 | P2 | Meta | synthetic validator 若使用 `np.` 等外部 alias，必須顯式宣告對應 import，不得依賴 transitive import 或未定義名稱，避免 coverage / consistency synthetic suite 因 `NameError` 假失敗 | DONE | 已補 static contract，直接掃描 `tools/validate/synthetic*_cases.py` 中使用 `np.` 的模組，釘死必須顯式 `import numpy as np`；並修正 `synthetic_contract_cases.py` 缺失 import 所造成的 coverage synthetic suite runtime regression | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B73 | P2 | Meta | 掃描 synthetic validator alias 使用情形時，必須以 AST 實際語意判定，不得用原始字串搜尋 `"np."` 誤判字串常值、註解或 validator 自述文字為真實 alias 使用，避免 meta validator 自己製造假失敗 | DONE | 已補 direct contract，直接釘死 numpy alias 使用掃描必須忽略僅出現在字串常值中的 `np.`，並仍可正確抓到實際 `np.array(...)` AST 使用；同步把 alias-import validator 改為 AST 判定，避免 `synthetic_meta_cases.py` 因自述字串誤被列為缺 import 模組 | `tools/validate/synthetic_meta_cases.py` |
| B74 | P1 | GUI | GUI 單股回測工作台必須以 K 線圖分頁作為主檢視；成交量預設隱藏，切換後須以前景疊加方式共用同一圖面且高度低於 1/4；摘要/明細須獨立分頁；GUI 開啟時預設最大化，且內嵌 K 線必須保留完整歷史可平移/縮放，不得因 GUI render slicing 截斷左右歷史資料 | DONE | 已補 direct contract，直接釘死 GUI panel 必須使用 notebook 分頁承接 K 線圖/摘要/明細、成交量 toggle 預設關閉、workbench 預設 maximized，且 chart helper 必須宣告 full-history navigation 與 volume overlay ratio ≤ 1/4；overlay axis 存在性改以 shared chart contract 驗證，不得綁死 `figure.axes` 長度，避免 matplotlib backend / inset axes 註冊差異造成假失敗 | `tools/gui/single_stock_inspector.py`, `tools/gui/workbench.py`, `tools/debug/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B75 | P1 | GUI | GUI 單股回測內嵌 K 線圖必須支援 toolbar-free 滑鼠互動：滾輪直接縮放、左鍵直接拖曳平移時間軸，且重新渲染成交量時不得殘留已銷毀 toolbar/widget 造成 Tk runtime error | DONE | 已補 direct contract，直接釘死 GUI panel 不得再依賴 `NavigationToolbar2Tk`、必須綁定 shared mouse navigation binder，並要求 chart helper 宣告 wheel-zoom / left-drag-pan / no-toolbar contract，避免縮放平移體驗分叉與 volume toggle 時殘留失效 widget | `tools/gui/single_stock_inspector.py`, `tools/debug/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B76 | P2 | Meta | synthetic validator 直接引用外部 chart/navigation helper 名稱時，必須在同檔顯式 import；不得依賴遺漏名稱在 formal suite 執行時才以 `NameError` 暴露，避免 coverage / consistency synthetic suite 假失敗 | DONE | 已補 static contract，直接釘死 `synthetic_contract_cases.py` 只要實際使用 `bind_matplotlib_chart_navigation`，就必須顯式自 `tools.debug.charting` import 該名稱；並修正缺失 import 造成的 synthetic suite runtime regression | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B77 | P1 | 契約 | debug 單股回測 entry marker 在 entry plan 不可掛單或為 `None` 時，必須 no-op；不得在 GUI / chart export 路徑中因直接存取 `entry_plan["limit_price"]` 而對特定股票炸出 `'NoneType' object is not subscriptable` | DONE | 已補 direct contract，直接以 `entry_plan=None` 呼叫 debug entry marker helper，釘死必須不新增 marker 且不得拋例外；同步修正 `_record_entry_plan_marker()` 的空 plan guard，避免高價股或 sizing 為 0 的股票在 GUI 執行回測時失敗 | `tools/debug/entry_flow.py`, `tools/validate/synthetic_contract_cases.py` |
| B78 | P2 | 效能 | 內嵌 GUI K 線相關 synthetic contract 若只需驗證 chart payload / figure 契約，不得強制走 HTML export；正式 debug analysis 應支援在 `export_chart=False` 時直接回傳 chart payload，避免 consistency synthetic suite 因額外載入 plotly / HTML artifact 路徑造成記憶體回歸 | DONE | 已補 direct contract，直接釘死 `run_debug_analysis(..., export_chart=False, return_chart_payload=True)` 必須可回傳 `chart_payload` 且 `chart_path` 保持 `None`；同步讓 reporting / backtest / trade_log wrapper 支援此 lightweight path，避免 GUI chart synthetic validator 為了拿 payload 而額外走 HTML export | `tools/debug/reporting.py`, `tools/debug/backtest.py`, `tools/debug/trade_log.py`, `tools/validate/synthetic_contract_cases.py` |

### B3. 可隨策略升級調整的測試

| ID | 優先級 | 類別 | 項目 | 目前判定 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|---|---|
| B47 | P1 | 模型介面 | model feature schema / prediction schema 穩定 | DONE | 已新增 model I/O schema case，直接驗輸入欄位、輸出欄位、型別與缺值處理 | `tools/validate/synthetic_strategy_cases.py`, `tools/optimizer/`, `tools/scanner/` |
| B48 | P1 | 重現性 | 同 seed 下 optimizer / model inference 可重現 | DONE | 已新增 strategy repeatability case，直接雙跑 scanner inference 與 optimizer objective，驗證輸出 payload、trial params 與 profile_row 在固定 seed / 固定輸入下可重現 | `tools/validate/synthetic_strategy_cases.py`, `tools/local_regression/`, `tools/optimizer/` |
| B49 | P1 | 排序輸出 | ranking / scoring 輸出可排序、可比較、無 NaN | DONE | 已新增 ranking / scoring sanity case，直接驗排序值可用、方向一致與型別正確 | `tools/validate/synthetic_strategy_cases.py`, `core/buy_sort.py`, `tools/scanner/` |
| B50 | P2 | 最低可用性 | 模型升級後 scanner / optimizer / reporting 仍可跑通 | DONE | 已新增 strategy minimum viability case，直接驗 scanner、optimizer、strategy dashboard、scanner summary 與 yearly return report 在策略輸入下可正常執行 | `tools/validate/synthetic_strategy_cases.py`, `apps/ml_optimizer.py`, `apps/vip_scanner.py` |
| B51 | P2 | 報表相容 | 新策略輸出仍符合既有 artifact / reporting schema | DONE | 已新增 strategy reporting schema compatibility case，直接驗 best_params export payload keys、scanner normalized payload keys 與 yearly return report columns 維持既有 schema | `tools/validate/synthetic_strategy_cases.py`, `tools/portfolio_sim/`, `tools/scanner/reporting.py` |
| B52 | P1 | Optimizer 契約 | objective 淘汰值 / fail_reason / profile_row / best_params export 穩定 | DONE | 已新增 optimizer objective / export contract case，直接驗 `INVALID_TRIAL_VALUE`、fail_reason、profile_row、`tp_percent` 還原優先序、export 成敗，以及訓練中斷且未達指定 trial 數時不得自動覆寫 `best_params.json`；僅完成指定訓練次數或輸入 0 走 export-only 模式時才可更新 | `tools/validate/synthetic_strategy_cases.py`, `tools/optimizer/main.py`, `tools/optimizer/objective_runner.py`, `tools/optimizer/runtime.py`, `tools/optimizer/study_utils.py`, `strategies/breakout/search_space.py`, `strategies/breakout/adapter.py`, `config/training_policy.py`, `config/execution_policy.py` |
| B53 | P1 | I/O | reduced dataset 契約必須依目前目錄快照動態推導，不得綁死固定成員或固定筆數 | DONE | 已將 reduced dataset contract 改為直接根據目前資料夾中的 CSV members / content 動態計算 `csv_count` 與 fingerprint；formal guard 只要求資料夾非空且 members 不重複，避免之後調整 reduced dataset 又必須回頭改程式常數 | `tools/local_regression/common.py`, `tools/validate/synthetic_contract_cases.py`, `data/tw_stock_data_vip_reduced/` |

## E. 未完成缺口摘要

使用方式：本節僅在同輪無法一次清空時暫存未完成缺口；若本輪已清空，維持空表即可。主維護來源仍是主表；`E3` 僅在確實存在未完成 `Txx` 時填寫，平時保持空表。

### E1. 目前所有 `PARTIAL` 的主表項目摘要

| 類型 | ID | 項目 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|

### E2. 目前所有 `TODO` 的主表項目摘要

| 類型 | ID | 項目 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|

### E3. 目前所有未完成的建議測試項目摘要

| ID | 建議測試名稱 | 目前狀態 | 對應主表項目 |
|---|---|---|---|

## T. 已完成建議測試映射

使用方式：本節只保留 `DONE` 的建議測試項目最小必要索引；不重複抄寫主表的建議落點，也不重複記錄完成日期。主表狀態、測試入口細節與缺口摘要仍以主表為準，時間軸僅寫在 `G`。

維護規則：`T` 固定只留「ID / 建議測試名稱 / 對應主表項目」，並依 ID 升冪排序。交付前至少核對一次「所有已註冊 validator / script 類型的 `Txx` 已同步列入 `T`，且 `T` 與 `G` 的最新狀態一致」。

### T. 目前所有 `DONE` 的建議測試項目摘要

| ID | 建議測試名稱 | 對應主表項目 |
|---|---|---|
| T01 | `validate_synthetic_same_day_buy_sell_forbidden_case` | B06 |
| T02 | `validate_synthetic_intraday_reprice_forbidden_case` | B05 |
| T03 | `validate_synthetic_no_intraday_switch_after_failed_fill_case` | B07 |
| T04 | `validate_synthetic_exit_orders_only_for_held_positions_case` | B08 |
| T05 | `validate_synthetic_fee_tax_net_equity_case` | B03 |
| T06 | `validate_synthetic_round_trip_pnl_only_on_tail_exit_case` | B04 |
| T07 | `validate_synthetic_missed_sell_accounting_case` | B11 |
| T08 | `validate_synthetic_candidate_order_fill_layer_separation_case` | B09 |
| T09 | `validate_synthetic_portfolio_history_filter_only_case` | B10 |
| T10 | `validate_synthetic_lookahead_prev_day_only_case` | B01 |
| T11 | `validate_price_utils_unit_case` | B13 |
| T12 | `validate_history_filters_unit_case` | B13 |
| T13 | `validate_portfolio_stats_unit_case` | B13 |
| T14 | `validate_model_io_schema_case` | B47 |
| T15 | `tools/local_regression/run_ml_smoke.py` | B12 |
| T16 | `validate_ranking_scoring_sanity_case` | B49 |
| T17 | `tools/validate/synthetic_reporting_cases.py` | B21 |
| T18 | `validate_output_contract_case` | B11 / B17 |
| T19 | `tools/local_regression/run_chain_checks.py` | B18 |
| T20 | `tools/local_regression/run_meta_quality.py` | B22 |
| T21 | `core/runtime_utils.py` | B19 |
| T22 | `validate_registry_checklist_entry_consistency_case` | B23 |
| T23 | `validate_known_bad_fault_injection_case` | B24 |
| T24 | `validate_independent_oracle_golden_case` | B25 |
| T25 | `tools/validate/meta_contracts.py` | B26 |
| T26 | `validate_cmd_document_contract_case` | B20 |
| T27 | `validate_display_reporting_sanity_case` | B21 |
| T28 | `validate_artifact_lifecycle_contract_case` | B17 |
| T29 | `tools/local_regression/formal_pipeline.py` | B23 |
| T30 | `validate_params_io_error_path_case` | B15 |
| T31 | `validate_module_loader_error_path_case` | B15 |
| T32 | `validate_preflight_error_path_case` | B15 |
| T33 | `validate_sanitize_ohlcv_expected_behavior_case` | B14 |
| T34 | `validate_sanitize_ohlcv_failfast_case` | B14 |
| T35 | `validate_load_clean_df_data_quality_case` | B14 |
| T36 | `validate_dataset_cli_contract_case` | B16 |
| T37 | `validate_local_regression_cli_contract_case` | B16 |
| T38 | `tools/scanner/scan_runner.py` | B12 / B18 |
| T39 | `validate_issue_excel_report_schema_case` | B21 |
| T40 | `validate_portfolio_export_report_artifacts_case` | B21 |
| T41 | `validate_extended_tool_cli_contract_case` | B16 |
| T42 | `validate_downloader_market_date_fallback_case` | B15 |
| T43 | `validate_downloader_sync_error_path_case` | B15 |
| T44 | `validate_downloader_main_error_path_case` | B15 |
| T45 | `validate_local_regression_summary_contract_case` | B11 |
| T46 | `validate_test_suite_summary_failure_reporting_case` | B21 |
| T47 | `validate_test_suite_summary_manifest_failure_reporting_case` | B21 |
| T48 | `validate_test_suite_summary_optional_dataset_skip_case` | B21 |
| T49 | `validate_test_suite_summary_preflight_failure_reporting_case` | B21 |
| T50 | `validate_test_suite_summary_dataset_prepare_failure_reporting_case` | B21 |
| T51 | `validate_test_suite_summary_unreadable_payload_reporting_case` | B21 |
| T52 | `validate_run_all_preflight_early_failure_dataset_contract_case` | B11 / B17 |
| T53 | `validate_test_suite_summary_checklist_status_sync_case` | B21 |
| T54 | `validate_test_suite_summary_meta_quality_guardrail_reporting_case` | B21 / B22 |
| T55 | `validate_single_formal_test_entry_contract_case` | B26 |
| T56 | `validate_synthetic_setup_index_prev_day_only_case` | B01 |
| T57 | `validate_downloader_universe_fetch_error_path_case` | B15 |
| T58 | `validate_downloader_universe_screening_init_error_path_case` | B15 |
| T59 | `validate_meta_quality_performance_memory_contract_case` | B19 |
| T60 | `validate_test_suite_summary_meta_quality_memory_reporting_case` | B19 / B21 |
| T61 | `validate_portfolio_sim_prepared_tool_contract_case` | B11 / B19 |
| T62 | `validate_scanner_prepared_tool_contract_case` | B19 |
| T63 | `validate_debug_trade_log_prepared_tool_contract_case` | B19 |
| T64 | `validate_scanner_reference_clean_df_contract_case` | B19 |
| T65 | `validate_meta_quality_reuses_existing_coverage_artifacts_case` | B19 / B22 |
| T66 | `tools/validate/synthetic_cases.py` | B23 |
| T67 | `tools/validate/synthetic_meta_cases.py` | B26 |
| T68 | `apps/test_suite.py` | B26 |
| T69 | `validate_no_reverse_app_layer_dependencies_case` | B23 |
| T70 | `validate_run_all_manifest_failure_master_summary_contract_case` | B17 |
| T71 | `validate_synthetic_same_bar_stop_priority_case` | B02 |
| T72 | `validate_synthetic_half_tp_full_year_case` | B04 / B21 |
| T73 | `validate_synthetic_extended_miss_buy_case` | B09 |
| T74 | `validate_synthetic_competing_candidates_case` | B09 |
| T75 | `validate_synthetic_same_day_sell_block_case` | B06 |
| T76 | `validate_synthetic_rotation_t_plus_one_case` | B05 / B06 |
| T77 | `validate_synthetic_missed_buy_no_replacement_case` | B07 |
| T78 | `validate_synthetic_unexecutable_half_tp_case` | B04 |
| T79 | `validate_synthetic_history_ev_threshold_case` | B10 |
| T80 | `validate_synthetic_single_backtest_not_gated_by_own_history_case` | B10 |
| T81 | `validate_synthetic_pit_same_day_exit_excluded_case` | B01 |
| T82 | `validate_synthetic_pit_multiple_same_day_exits_case` | B01 |
| T83 | `validate_synthetic_proj_cost_cash_capped_case` | B09 |
| T84 | `validate_synthetic_param_guardrail_case` | B15 |
| T85 | `validate_validate_console_summary_reporting_case` | B21 |
| T86 | `validate_portfolio_yearly_report_schema_case` | B21 |
| T87 | `validate_test_suite_summary_reporting_case` | B21 |
| T88 | `validate_scanner_worker_repeatability_case` | B12 |
| T89 | `validate_scan_runner_repeatability_case` | B12 |
| T90 | `validate_optimizer_raw_cache_rerun_consistency_case` | B18 |
| T91 | `validate_run_all_repeatability_case` | B18 |
| T92 | `validate_no_top_level_import_cycles_case` | B27 |
| T93 | `validate_core_trading_modules_in_coverage_targets_case` | B28 |
| T94 | `validate_critical_file_coverage_minimum_gate_case` | B29 |
| T95 | `validate_coverage_threshold_floor_case` | B30 |
| T96 | `validate_entry_path_critical_coverage_gate_case` | B31 |
| T97 | `validate_critical_coverage_threshold_floor_case` | B32 |
| T98 | `validate_dataset_fingerprint_contract_case` | B33 |
| T99 | `validate_atomic_write_contract_case` | B34 |
| T100 | `tools/local_regression/common.py` | B36 |
| T101 | `validate_test_suite_orchestrator_coverage_targets_case` | B35 |
| T102 | `validate_atomic_write_retry_contract_case` | B34 |
| T103 | `validate_run_all_dataset_prepare_pass_main_contract_case` | B33 |
| T104 | `validate_synthetic_registry_metadata_contract_case` | B37 |
| T105 | `validate_optimizer_objective_export_contract_case` | B52 |
| T106 | `validate_formal_step_entry_coverage_targets_case` | B38 |
| T107 | `validate_checklist_g_single_note_entry_delimiter_case` | B26 |
| T108 | `validate_checklist_f2_single_entry_delimiter_case` | B26 |
| T109 | `validate_checklist_g_transition_format_case` | B26 |
| T110 | `validate_checklist_no_legacy_d_section_case` | B26 |
| T111 | `validate_formal_step_implementation_coverage_targets_case` | B39 |
| T112 | `validate_peak_traced_memory_tracker_context_management_case` | B40 |
| T113 | `validate_run_all_cli_error_usage_contract_case` | B16 |
| T114 | `validate_no_legacy_app_entry_doc_references_case` | B41 |
| T115 | `validate_app_thin_wrapper_export_contract_case` | B42 |
| T116 | `validate_package_zip_runtime_contract_case` | B43 |
| T117 | `validate_synthetic_non_candidate_setup_does_not_seed_extended_signal_case` | B09 |
| T118 | `validate_atomic_write_cleanup_error_preserves_root_exception_case` | B34 |
| T119 | `validate_validate_summary_atomic_write_contract_case` | B34 |
| T120 | `validate_quick_gate_bare_except_guard_contract_case` | B44 |
| T121 | `validate_quick_gate_output_path_guard_contract_case` | B45 |
| T122 | `validate_dataset_prepare_fallback_write_traceability_case` | B46 |
| T123 | `validate_console_tail_read_error_traceability_case` | B46 |
| T124 | `validate_checklist_g_ordering_case` | B26 |
| T125 | `validate_checklist_no_legacy_f1_section_case` | B26 |
| T126 | `validate_strategy_repeatability_case` | B48 |
| T127 | `validate_strategy_minimum_viability_case` | B50 |
| T128 | `validate_strategy_reporting_schema_compatibility_case` | B51 |
| T129 | `validate_reduced_dataset_dynamic_contract_case` | B53 |
| T130 | `validate_synthetic_single_backtest_uses_compounding_capital_case` | B54 |
| T131 | `validate_single_ticker_compounding_parity_contract_case` | B55 |
| T132 | `validate_scanner_live_capital_contract_case` | B56 |
| T133 | `validate_optimizer_interrupt_export_contract_case` | B52 |
| T134 | `validate_score_numerator_option_case` | B57 |
| T135 | `validate_use_compounding_failfast_guardrail_case` | B58 |
| T136 | `validate_critical_helper_single_source_contract_case` | B59 |
| T137 | `validate_project_settings_dynamic_test_boundary_case` | B60 |
| T138 | `validate_policy_contract_modules_in_coverage_targets_case` | B61 |
| T139 | `validate_checklist_g_new_transition_first_occurrence_case` | B26 |
| T140 | `validate_checklist_g_note_validate_reference_exists_case` | B26 |
| T141 | `validate_checklist_first_nonempty_line_case` | B62 |
| T142 | `validate_synthetic_meta_cases_summary_value_accessor_contract_case` | B63 |
| T143 | `validate_checklist_summary_tables_sorted_by_id_case` | B64 |
| T144 | `validate_package_zip_commit_test_suite_orchestration_case` | B65 |
| T145 | `validate_gui_workbench_contract_case` | B66 |
| T146 | `validate_debug_trade_log_chart_context_optional_case` | B67 |
| T147 | `validate_tool_module_path_normalization_case` | B68 |
| T148 | `validate_module_path_normalizer_accepts_path_objects_case` | B69 |
| T149 | `validate_module_loader_project_root_string_patch_case` | B70 |
| T150 | `validate_gui_embedded_chart_contract_case` | B71 |
| T151 | `validate_synthetic_case_numpy_alias_import_contract_case` | B72 |
| T152 | `validate_synthetic_case_numpy_alias_scan_ignores_string_literals_contract_case` | B73 |
| T153 | `validate_gui_chart_workspace_contract_case` | B74 |
| T154 | `validate_gui_mouse_navigation_contract_case` | B75 |
| T155 | `validate_synthetic_case_chart_navigation_binder_import_contract_case` | B76 |
| T156 | `validate_debug_entry_plan_marker_optional_contract_case` | B77 |
| T157 | `validate_debug_chart_payload_without_html_export_contract_case` | B78 |

## G. 逐項收斂紀錄

使用方式：每次只挑少數高優先項目處理，完成後更新本節，不要重開一份新清單。編輯本節時，先依日期定位到對應區塊，再抽出整個同日區塊依排序鍵重排後整段覆寫回原位；禁止把新列直接追加到該日期區塊尾端，也禁止只改局部單列後跳過同日區塊總排序檢查；若新增列排序鍵小於當前尾列，必須回插到正確位置，不得留在尾端。交付前至少再做一次同日區塊機械核對：由上到下檢查 namespace、數字段、尾碼三層排序鍵皆未逆序，且新增列同時滿足前一列 ≤ 當前列 ≤ 後一列；備註欄若需要引用檔案或測試名稱，只能保留一個代表 entry。

| 日期 | 項目 ID | 動作 | 狀態變更 | 備註 |
|---|---|---|---|---|
| 2026-04-01 | T01 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_same_day_buy_sell_forbidden_case |
| 2026-04-01 | T02 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_intraday_reprice_forbidden_case |
| 2026-04-01 | T03 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_no_intraday_switch_after_failed_fill_case |
| 2026-04-01 | T04 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_exit_orders_only_for_held_positions_case |
| 2026-04-01 | T05 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_fee_tax_net_equity_case |
| 2026-04-01 | T06 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_round_trip_pnl_only_on_tail_exit_case |
| 2026-04-01 | T07 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_missed_sell_accounting_case |
| 2026-04-01 | T08 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_candidate_order_fill_layer_separation_case |
| 2026-04-01 | T09 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_portfolio_history_filter_only_case |
| 2026-04-01 | T10 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_lookahead_prev_day_only_case |
| 2026-04-01 | T11 | 新增 unit-like 邊界案例並驗證 | TODO -> DONE | validate_price_utils_unit_case |
| 2026-04-01 | T12 | 新增 unit-like 邊界案例並驗證 | TODO -> DONE | validate_history_filters_unit_case |
| 2026-04-01 | T13 | 新增 unit-like 邊界案例並驗證 | TODO -> DONE | validate_portfolio_stats_unit_case |
| 2026-04-01 | T15 | 新增 optimizer fixed-seed 雙跑一致性檢查 | TODO -> PARTIAL | `run_ml_smoke.py` 已比較雙跑 trial / best_params digest；scanner 尚未補 |
| 2026-04-01 | T18 | 新增 CSV / XLSX / JSON output contract case 並驗證 | TODO -> PARTIAL | validate_output_contract_case |
| 2026-04-01 | T19 | 新增 chain checks 雙跑 digest 對比與 optimizer 雙跑 | TODO -> PARTIAL | 已補雙跑流程，但 scanner 入口尚未收斂。 |
| 2026-04-01 | T20 | 新增 `run_meta_quality.py` 產出 coverage baseline | TODO -> PARTIAL | 目前已覆蓋 synthetic coverage suite 與 key target coverage，並納入正式入口摘要。 |
| 2026-04-01 | T21 | 新增 `run_meta_quality.py` performance baseline gating | TODO -> PARTIAL | 已正式檢查 reduced suite 各步驟 / total duration 與 optimizer 平均 trial wall time；記憶體回歸仍未納入 |
| 2026-04-01 | T22 | 新增 meta registry case 並驗證 | TODO -> DONE | validate_registry_checklist_entry_consistency_case |
| 2026-04-01 | T23 | 新增 meta fault-injection case 並驗證 | TODO -> DONE | validate_known_bad_fault_injection_case |
| 2026-04-01 | T24 | 新增 independent oracle golden case 並驗證 | TODO -> DONE | validate_independent_oracle_golden_case |
| 2026-04-01 | T26 | 新增 CMD 指令契約案例並驗證 | TODO -> DONE | validate_cmd_document_contract_case |
| 2026-04-02 | B01 | 補 setup index prev-day-only invariant 後主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_history_cases.py` |
| 2026-04-02 | B11 | 跨工具 schema / 欄位語意補齊，主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-02 | B12 | 決定性主表收斂為 DONE | PARTIAL -> DONE | `run_ml_smoke.py` |
| 2026-04-02 | B15 | 補 downloader 外部 API fatal error path 後主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_error_cases.py` |
| 2026-04-02 | B16 | CLI 契約涵蓋補齊，主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_cli_cases.py` |
| 2026-04-02 | B18 | 重跑一致性 / 狀態汙染主表收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_chain_checks.py` |
| 2026-04-02 | B19 | 將 traced peak memory 納入正式 gate，主表升為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | B22 | 將 coverage report baseline 收斂為正式 gate，主表升為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | B23 | 檢出 synthetic 主入口漏註冊既有 `validate_*` case，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_cases.py` 尚未完整覆蓋 imported validate cases |
| 2026-04-02 | B23 | 補齊 synthetic 主入口遺漏註冊與 registry completeness guard 後收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_cases.py` |
| 2026-04-02 | B26 | checklist / test suite 自身完整性收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | B26 | 檢出完成摘要索引仍有漏同步風險，主表改回 PARTIAL | DONE -> PARTIAL | checklist 自身仍有回寫 / 摘要失同步缺口 |
| 2026-04-02 | B26 | 補齊 checklist main / `T` / `G` sync blocker 後收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | T14 | 新增 model I/O schema 案例並驗證 | TODO -> DONE | `validate_model_io_schema_case` |
| 2026-04-02 | T15 | 補 scanner worker / `scan_runner` 入口重跑一致性後收斂完成 | PARTIAL -> DONE | `validate_scanner_worker_repeatability_case` |
| 2026-04-02 | T16 | 新增 ranking / scoring sanity 案例並驗證 | TODO -> DONE | `validate_ranking_scoring_sanity_case` |
| 2026-04-02 | T17 | reporting schema compatibility checks 收斂完成，並新增輸出檔 schema 補強 | TODO -> DONE | `validate_issue_excel_report_schema_case` |
| 2026-04-02 | T18 | 擴充 local regression summary contract 並收斂完成 | PARTIAL -> DONE | `validate_output_contract_case` |
| 2026-04-02 | T19 | 補 `run_all.py` 同 run dir rerun summary / bundle repeatability 後收斂完成 | PARTIAL -> DONE | `validate_optimizer_raw_cache_rerun_consistency_case` |
| 2026-04-02 | T20 | 補 manifest 化 line / branch threshold gate 與 summary sync | PARTIAL -> DONE | `run_meta_quality.py` |
| 2026-04-02 | T21 | 補 traced peak memory regression gate 後 performance baseline 收斂完成 | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | T25 | 擴充 checklist sufficiency formal check 到單一正式入口與 legacy entry 檢查後收斂完成 | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | T27 | 擴充 scanner summary / banner 與 display re-export 後收斂完成 | PARTIAL -> DONE | validate_display_reporting_sanity_case |
| 2026-04-02 | T28 | 擴充 artifact lifecycle contract 並驗證 | PARTIAL -> DONE | validate_artifact_lifecycle_contract_case |
| 2026-04-02 | T29 | 新增 formal-entry consistency checks 並驗證 | TODO -> DONE | `run_meta_quality.py` formal-entry consistency checks |
| 2026-04-02 | T30 | 新增 params_io 錯誤路徑案例並驗證 | TODO -> DONE | `validate_params_io_error_path_case` |
| 2026-04-02 | T31 | 新增 module_loader 錯誤路徑案例並驗證 | TODO -> DONE | `validate_module_loader_error_path_case` |
| 2026-04-02 | T32 | 新增 preflight 錯誤路徑案例並驗證 | TODO -> DONE | `validate_preflight_error_path_case` |
| 2026-04-02 | T33 | 新增資料清洗 expected behavior 案例並驗證 | TODO -> DONE | `validate_sanitize_ohlcv_expected_behavior_case` |
| 2026-04-02 | T34 | 新增資料清洗 fail-fast 案例並驗證 | TODO -> DONE | `validate_sanitize_ohlcv_failfast_case` |
| 2026-04-02 | T35 | 新增 `load_clean_df` 資料品質整合案例並驗證 | TODO -> DONE | `validate_load_clean_df_data_quality_case` |
| 2026-04-02 | T36 | 新增 dataset wrapper CLI 契約案例並驗證 | TODO -> DONE | `validate_dataset_cli_contract_case` |
| 2026-04-02 | T37 | 新增 local regression / no-arg CLI 契約案例並驗證 | TODO -> DONE | `validate_local_regression_cli_contract_case` |
| 2026-04-02 | T38 | 新增 scanner reduced snapshot 雙跑 digest 並驗證 | TODO -> DONE | `run_chain_checks.py` 已將 scanner 候選 / 狀態 / issue line 納入 rerun consistency payload |
| 2026-04-02 | T39 | 新增 issue Excel report schema 案例並驗證 | NEW -> DONE | `validate_issue_excel_report_schema_case` |
| 2026-04-02 | T40 | 新增 portfolio export report artifacts 案例並驗證 | NEW -> DONE | `validate_portfolio_export_report_artifacts_case` |
| 2026-04-02 | T41 | 補齊剩餘直接入口 CLI 契約並收斂 B16 | TODO -> DONE | `validate_extended_tool_cli_contract_case` |
| 2026-04-02 | T42 | 新增 downloader market-date fallback 案例並驗證 | TODO -> DONE | `validate_downloader_market_date_fallback_case` |
| 2026-04-02 | T43 | 新增 downloader sync 錯誤聚合案例並驗證 | TODO -> DONE | `validate_downloader_sync_error_path_case` |
| 2026-04-02 | T44 | 新增 downloader main 失敗摘要案例並驗證 | TODO -> DONE | `validate_downloader_main_error_path_case` |
| 2026-04-02 | T45 | 新增 local regression summary contract case 並驗證 | TODO -> DONE | `validate_local_regression_summary_contract_case` |
| 2026-04-02 | T46 | 新增腳本失敗摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_failure_reporting_case` |
| 2026-04-02 | T47 | 新增 manifest blocked 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_manifest_failure_reporting_case` |
| 2026-04-02 | T48 | 新增 partial selected-steps 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_optional_dataset_skip_case` |
| 2026-04-02 | T49 | 新增 preflight fail 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_preflight_failure_reporting_case` |
| 2026-04-02 | T50 | 新增 dataset prepare fail 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_dataset_prepare_failure_reporting_case` |
| 2026-04-02 | T51 | 新增 summary unreadable 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_unreadable_payload_reporting_case` |
| 2026-04-02 | T52 | 補 `run_all.py` preflight 早退 dataset not-run contract 並驗證 | TODO -> DONE | `validate_run_all_preflight_early_failure_dataset_contract_case` |
| 2026-04-02 | T53 | 補 checklist status vocabulary sync 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_checklist_status_sync_case` |
| 2026-04-02 | T54 | 補 meta quality coverage guardrail 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_meta_quality_guardrail_reporting_case` |
| 2026-04-02 | T55 | 新增單一正式測試入口契約案例並驗證 | NEW -> DONE | `validate_single_formal_test_entry_contract_case` |
| 2026-04-02 | T56 | 新增 setup-index prev-day-only synthetic case 並驗證 | NEW -> DONE | `validate_synthetic_setup_index_prev_day_only_case` |
| 2026-04-02 | T57 | 新增 downloader universe fetch fatal error case 並驗證 | NEW -> DONE | `validate_downloader_universe_fetch_error_path_case` |
| 2026-04-02 | T58 | 新增 downloader screening init fatal error case 並驗證 | NEW -> DONE | `validate_downloader_universe_screening_init_error_path_case` |
| 2026-04-02 | T59 | 新增 meta quality performance memory contract case 並驗證 | NEW -> DONE | `validate_meta_quality_performance_memory_contract_case` |
| 2026-04-02 | T60 | 新增 test suite meta quality memory reporting case 並驗證 | NEW -> DONE | `validate_test_suite_summary_meta_quality_memory_reporting_case` |
| 2026-04-02 | T61 | 新增 portfolio_sim prepared tool contract case 並驗證 | NEW -> DONE | `validate_portfolio_sim_prepared_tool_contract_case` |
| 2026-04-02 | T62 | 新增 scanner prepared tool contract case 並驗證 | NEW -> DONE | `validate_scanner_prepared_tool_contract_case` |
| 2026-04-02 | T63 | 新增 debug trade log prepared tool contract case 並驗證 | NEW -> DONE | `validate_debug_trade_log_prepared_tool_contract_case` |
| 2026-04-02 | T64 | 新增 scanner reference clean-df contract case 並驗證 | NEW -> DONE | `validate_scanner_reference_clean_df_contract_case` |
| 2026-04-02 | T65 | 新增 meta quality coverage artifact reuse contract case 並驗證 | NEW -> DONE | `validate_meta_quality_reuses_existing_coverage_artifacts_case` |
| 2026-04-02 | T66 | 新增 imported validate cases vs synthetic registry formal guard 缺口 | NEW -> TODO | 需補正式註冊完整性檢查。 |
| 2026-04-02 | T66 | 補 imported / defined validate cases 與 synthetic registry 完整一致 formal guard | TODO -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-02 | T67 | 新增主表 / `T` / `G` 完整同步 formal guard 缺口 | NEW -> TODO | 需補 checklist 自身同步性檢查。 |
| 2026-04-02 | T67 | 補主表 / `T` / `G` 收斂紀錄完整同步 formal guard | TODO -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | T68 | 新增 checklist 完成映射同步缺口 | NEW -> TODO | 需阻擋已完成 T 項遺漏於 `T` 仍被判定為已收斂 |
| 2026-04-02 | T68 | 補 checklist `DONE` 摘要缺漏自動偵測與阻擋 | TODO -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B27 | 補 top-level import cycle formal guard 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-03 | B28 | 補入核心交易模組 coverage target completeness 主表項目 | NEW -> PARTIAL | `run_meta_quality.py` 已有 key target hit guard，但尚未明確要求核心交易模組入列 |
| 2026-04-03 | B28 | 核心交易模組已納入 `COVERAGE_TARGETS`，主表收斂為 DONE | PARTIAL -> DONE | `run_meta_quality.py` |
| 2026-04-03 | B29 | 補入 critical file per-file coverage minimum gate 主表項目 | NEW -> TODO | 目前僅有 overall coverage gate，尚未建立核心檔 per-file minimum guard |
| 2026-04-03 | B29 | 核心檔 per-file coverage minimum guard 已建立，主表收斂為 DONE | TODO -> DONE | `run_meta_quality.py` 已正式檢查 critical file line / branch minimum coverage |
| 2026-04-03 | B30 | 補入 coverage threshold gradual uplift 主表項目 | NEW -> PARTIAL | 已有 minimum threshold gate，但正式基線仍為 `line 50% / branch 45%` |
| 2026-04-03 | B30 | 正式 coverage 基線已提高並補 floor guard，主表收斂為 DONE | PARTIAL -> DONE | 基線已提升，並由 `run_meta_quality.py` 阻擋回退。 |
| 2026-04-03 | B31 | 進場關鍵模組已納入 critical file coverage gate，主表收斂為 DONE | NEW -> DONE | `run_meta_quality.py` |
| 2026-04-03 | B32 | critical per-file threshold 已提升到 stage-2 正式基線，主表收斂為 DONE | NEW -> DONE | 基線已提升，並由 `run_meta_quality.py` 阻擋回退。 |
| 2026-04-03 | B33 | 補上 reduced dataset member/content fingerprint gate，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/common.py` |
| 2026-04-03 | B34 | 補上 atomic write 與 replace-failure recovery contract，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/common.py` |
| 2026-04-03 | B35 | 將 test suite orchestrator modules 納入 coverage targets，主表收斂為 DONE | NEW -> DONE | `run_meta_quality.py` |
| 2026-04-03 | B36 | 補上 artifacts manifest sha256 contract，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/common.py` |
| 2026-04-03 | B37 | 補 synthetic registry metadata contract 後主表納入 DONE | NEW -> DONE | `tools/validate/synthetic_cases.py` |
| 2026-04-03 | B38 | 將 formal pipeline step entry wrappers 納入 coverage targets，主表收斂為 DONE | NEW -> DONE | `run_meta_quality.py` |
| 2026-04-03 | B39 | 將 split formal-step implementation modules 納入 coverage targets，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/meta_quality_targets.py` |
| 2026-04-03 | B40 | 補上 `PeakTracedMemoryTracker` context-manager lifecycle contract，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-03 | B52 | 補 optimizer objective / export contract 最低維護線 | NEW -> DONE | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-03 | T69 | 新增 `core/` / `tools/` 不得反向 import `apps/` 的分層 guard | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-03 | T70 | 新增 manifest failure master summary schema contract 並驗證 | NEW -> DONE | `validate_run_all_manifest_failure_master_summary_contract_case` |
| 2026-04-03 | T71 | 將既有同棒停損優先 synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_same_bar_stop_priority_case` |
| 2026-04-03 | T72 | 將既有半倉停利 full-year synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_half_tp_full_year_case` |
| 2026-04-03 | T73 | 將既有 extended miss buy synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_extended_miss_buy_case` |
| 2026-04-03 | T74 | 將既有 competing candidates synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_competing_candidates_case` |
| 2026-04-03 | T75 | 將既有 same-day sell block synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_same_day_sell_block_case` |
| 2026-04-03 | T76 | 將既有 rotation T+1 synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_rotation_t_plus_one_case` |
| 2026-04-03 | T77 | 將既有 missed buy no replacement synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_missed_buy_no_replacement_case` |
| 2026-04-03 | T78 | 將既有 unexecutable half TP synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_unexecutable_half_tp_case` |
| 2026-04-03 | T79 | 將既有 history EV threshold synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_history_ev_threshold_case` |
| 2026-04-03 | T80 | 將既有 single-backtest own-history guard synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_single_backtest_not_gated_by_own_history_case` |
| 2026-04-03 | T81 | 將既有 PIT same-day exit excluded synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_pit_same_day_exit_excluded_case` |
| 2026-04-03 | T82 | 將既有 PIT multiple same-day exits synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_pit_multiple_same_day_exits_case` |
| 2026-04-03 | T83 | 將既有 projected-cost cash-capped synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_proj_cost_cash_capped_case` |
| 2026-04-03 | T84 | 將既有 param guardrail synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_param_guardrail_case` |
| 2026-04-03 | T85 | 將既有 validate console summary reporting contract 回寫 checklist | NEW -> DONE | `validate_validate_console_summary_reporting_case` |
| 2026-04-03 | T86 | 將既有 portfolio yearly report schema contract 回寫 checklist | NEW -> DONE | `validate_portfolio_yearly_report_schema_case` |
| 2026-04-03 | T87 | 將既有 test suite PASS summary reporting contract 回寫 checklist | NEW -> DONE | `validate_test_suite_summary_reporting_case` |
| 2026-04-03 | T88 | 將既有 scanner worker repeatability synthetic case 回寫 checklist | NEW -> DONE | `validate_scanner_worker_repeatability_case` |
| 2026-04-03 | T89 | 將既有 scan runner repeatability synthetic case 回寫 checklist | NEW -> DONE | `validate_scan_runner_repeatability_case` |
| 2026-04-03 | T90 | 將既有 optimizer raw cache rerun consistency case 回寫 checklist | NEW -> DONE | `validate_optimizer_raw_cache_rerun_consistency_case` |
| 2026-04-03 | T91 | 將既有 run_all repeatability case 回寫 checklist | NEW -> DONE | `validate_run_all_repeatability_case` |
| 2026-04-03 | T92 | 新增 top-level import cycle guard 並驗證 | NEW -> DONE | `validate_no_top_level_import_cycles_case` |
| 2026-04-03 | T93 | 新增核心交易模組 coverage target completeness 建議測試項目 | NEW -> TODO | `validate_core_trading_modules_in_coverage_targets_case` |
| 2026-04-03 | T93 | 補上核心交易模組 coverage target completeness guard 並驗證 | TODO -> DONE | `validate_core_trading_modules_in_coverage_targets_case` |
| 2026-04-03 | T94 | 新增 critical file per-file coverage minimum guard 建議測試項目 | NEW -> TODO | `validate_critical_file_coverage_minimum_gate_case` |
| 2026-04-03 | T94 | 補上 critical file per-file coverage minimum guard 並驗證 | TODO -> DONE | `validate_critical_file_coverage_minimum_gate_case` |
| 2026-04-03 | T95 | 新增 coverage threshold floor 建議測試項目 | NEW -> TODO | `validate_coverage_threshold_floor_case` |
| 2026-04-03 | T95 | 補上 coverage threshold floor guard 並驗證 | TODO -> DONE | `validate_coverage_threshold_floor_case` |
| 2026-04-03 | T96 | 新增 entry path critical coverage gate 建議測試並驗證 | NEW -> DONE | `validate_entry_path_critical_coverage_gate_case` |
| 2026-04-03 | T97 | 新增 critical per-file threshold stage-2 floor 建議測試並驗證 | NEW -> DONE | `validate_critical_coverage_threshold_floor_case` |
| 2026-04-03 | T98 | 新增 reduced dataset fingerprint contract 並驗證 | NEW -> DONE | `validate_dataset_fingerprint_contract_case` |
| 2026-04-03 | T99 | 新增 atomic write replace-failure recovery contract 並驗證 | NEW -> DONE | `validate_atomic_write_contract_case` |
| 2026-04-03 | T100 | 擴充 artifact manifest sha256 生成邏輯並由 contract 驗證對照 | NEW -> DONE | `tools/local_regression/common.py` |
| 2026-04-03 | T101 | 新增 test suite orchestrator coverage target completeness guard 並驗證 | NEW -> DONE | `validate_test_suite_orchestrator_coverage_targets_case` |
| 2026-04-03 | T102 | 新增 atomic write transient retry contract 並驗證 | NEW -> DONE | `validate_atomic_write_retry_contract_case` |
| 2026-04-03 | T103 | 新增 run_all dataset prepare PASS 主路徑 contract 並驗證 | NEW -> DONE | `validate_run_all_dataset_prepare_pass_main_contract_case` |
| 2026-04-03 | T104 | 新增 synthetic registry metadata contract case 並驗證 | NEW -> DONE | `validate_synthetic_registry_metadata_contract_case` |
| 2026-04-03 | T105 | 新增 optimizer objective / export contract case 並驗證 | NEW -> DONE | `validate_optimizer_objective_export_contract_case` |
| 2026-04-03 | T106 | 新增 formal step entry wrappers coverage target completeness 建議測試並驗證 | NEW -> DONE | `validate_formal_step_entry_coverage_targets_case` |
| 2026-04-03 | T107 | 新增 `G` 備註欄 delimiter-agnostic single-entry guard 並驗證 | NEW -> DONE | `validate_checklist_g_single_note_entry_delimiter_case` |
| 2026-04-03 | T108 | 新增 `T` 測試入口 delimiter-agnostic single-entry guard 並驗證 | NEW -> DONE | `validate_checklist_f2_single_entry_delimiter_case` |
| 2026-04-03 | T109 | 新增 `G` transition format guard 並驗證 | NEW -> DONE | `validate_checklist_g_transition_format_case` |
| 2026-04-03 | T110 | 新增 checklist legacy `D` 區移除 guard 並驗證 | NEW -> DONE | `validate_checklist_no_legacy_d_section_case` |
| 2026-04-03 | T111 | 新增 split formal-step implementation coverage target completeness guard 並驗證 | NEW -> DONE | `validate_formal_step_implementation_coverage_targets_case` |
| 2026-04-03 | T112 | 新增 memory tracker lifecycle contract 並驗證 | NEW -> DONE | `validate_peak_traced_memory_tracker_context_management_case` |
| 2026-04-04 | B41 | 移除 legacy app 測試入口文件殘留與手動刪檔指引後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-04 | B42 | 補上 app thin wrapper public export contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-04 | B43 | 補上 package_zip 正式入口 runtime contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_cli_cases.py` |
| 2026-04-04 | B44 | 補上 quick_gate bare-except static guard contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B45 | 補上 quick_gate output path / outputs root / log path guard contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B46 | 補上 formal pipeline fallback / console tail traceability contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B48 | 補 strategy-layer repeatability formal case 後收斂為 DONE | TODO -> DONE | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-04 | B50 | 補 strategy minimum viability formal smoke 後收斂為 DONE | TODO -> DONE | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-04 | B51 | 補 strategy reporting / artifact schema compatibility formal case 後收斂為 DONE | TODO -> DONE | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-04 | B53 | 將 reduced dataset contract 改為目錄快照動態推導，移除固定成員 / 固定筆數依賴後收斂為 DONE | NEW -> DONE | `tools/local_regression/common.py` |
| 2026-04-04 | B54 | 專案資金規則改為全系統複利後，原單股 fixed-cap 規格退役，主表先改回 PARTIAL | DONE -> PARTIAL | 待改為單股複利資金 contract |
| 2026-04-04 | B54 | 改為單股複利資金 contract 並收斂完成 | PARTIAL -> DONE | `tools/validate/synthetic_history_cases.py` |
| 2026-04-04 | B55 | 專案資金規則改為全系統複利後，原 execution-only fixed-cap 規格退役，主表先改回 PARTIAL | DONE -> PARTIAL | 待改為單檔複利 parity contract |
| 2026-04-04 | B55 | 改為單檔複利 parity contract 並收斂完成 | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B55 | 檢出單檔複利 parity contract 初版未覆蓋獲利後 entry budget，主表改回 PARTIAL | DONE -> PARTIAL | portfolio 實際下單仍可能以 available_cash 恢復複利 |
| 2026-04-04 | B55 | 補齊 gain-side entry budget contract 與 portfolio 單檔 parity 路徑後收斂為 DONE | PARTIAL -> DONE | `core/portfolio_entries.py` |
| 2026-04-04 | B56 | 新增 scanner live capital contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B57 | 新增 score numerator option contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-04 | T113 | 新增 run_all CLI error usage contract 並驗證 | NEW -> DONE | `validate_run_all_cli_error_usage_contract_case` |
| 2026-04-04 | T114 | 新增 legacy app 測試入口文件殘留 guard 並驗證 | NEW -> DONE | `validate_no_legacy_app_entry_doc_references_case` |
| 2026-04-04 | T115 | 新增 app thin wrapper lazy export contract 並驗證 | NEW -> DONE | `validate_app_thin_wrapper_export_contract_case` |
| 2026-04-04 | T116 | 新增 package_zip runtime contract 並驗證 | NEW -> DONE | `validate_package_zip_runtime_contract_case` |
| 2026-04-04 | T117 | 新增 non-candidate setup 不得 seed / revive extended candidate synthetic case 並驗證 | NEW -> DONE | `validate_synthetic_non_candidate_setup_does_not_seed_extended_signal_case` |
| 2026-04-04 | T118 | 新增 atomic write cleanup failure contract 並驗證 | NEW -> DONE | `validate_atomic_write_cleanup_error_preserves_root_exception_case` |
| 2026-04-04 | T119 | 新增 validate summary 正式路徑 atomic write contract 並驗證 | NEW -> DONE | `validate_validate_summary_atomic_write_contract_case` |
| 2026-04-04 | T120 | 新增 quick_gate bare-except guard contract 並驗證 | NEW -> DONE | `validate_quick_gate_bare_except_guard_contract_case` |
| 2026-04-04 | T121 | 新增 quick_gate output path / outputs root / log path guard contract 並驗證 | NEW -> DONE | `validate_quick_gate_output_path_guard_contract_case` |
| 2026-04-04 | T122 | 新增 dataset prepare fallback write traceability contract 並驗證 | NEW -> DONE | `validate_dataset_prepare_fallback_write_traceability_case` |
| 2026-04-04 | T123 | 新增 console tail read-error traceability contract 並驗證 | NEW -> DONE | `validate_console_tail_read_error_traceability_case` |
| 2026-04-04 | T124 | 新增 checklist `G` 日期 / ID 排序 guard 並驗證 | NEW -> DONE | `validate_checklist_g_ordering_case` |
| 2026-04-04 | T125 | 新增 legacy F1 回流 guard 並驗證 | NEW -> DONE | `validate_checklist_no_legacy_f1_section_case` |
| 2026-04-04 | T126 | 新增 strategy-layer repeatability formal case 並驗證 | NEW -> DONE | `validate_strategy_repeatability_case` |
| 2026-04-04 | T127 | 新增 strategy minimum viability formal smoke 並驗證 | NEW -> DONE | `validate_strategy_minimum_viability_case` |
| 2026-04-04 | T128 | 新增 strategy reporting / artifact schema compatibility formal case 並驗證 | NEW -> DONE | `validate_strategy_reporting_schema_compatibility_case` |
| 2026-04-04 | T129 | 將 reduced dataset contract 改為動態快照驗證並完成同步 | NEW -> DONE | `validate_reduced_dataset_dynamic_contract_case` |
| 2026-04-04 | T130 | 專案資金規則改為全系統複利後，原單股 fixed-cap synthetic case 退役，先改回 PARTIAL | DONE -> PARTIAL | 待改為單股複利 synthetic case |
| 2026-04-04 | T130 | 改為單股複利 synthetic case 並驗證 | PARTIAL -> DONE | `validate_synthetic_single_backtest_uses_compounding_capital_case` |
| 2026-04-04 | T131 | 專案資金規則改為全系統複利後，原 execution-only fixed-cap parity contract 退役，先改回 PARTIAL | DONE -> PARTIAL | 待改為單檔複利 parity contract |
| 2026-04-04 | T131 | 改為單檔複利 parity contract 並驗證 | PARTIAL -> DONE | `validate_single_ticker_compounding_parity_contract_case` |
| 2026-04-04 | T131 | 檢出單檔複利 parity contract 初版僅覆蓋虧損側，改回 PARTIAL | DONE -> PARTIAL | 尚未釘死獲利後不得再放大倉位 |
| 2026-04-04 | T131 | 擴充獲利側 entry budget 的單檔複利 parity contract 並收斂完成 | PARTIAL -> DONE | `validate_single_ticker_compounding_parity_contract_case` |
| 2026-04-04 | T132 | 新增 scanner live capital contract 並驗證 | NEW -> DONE | `validate_scanner_live_capital_contract_case` |
| 2026-04-04 | T133 | 新增 optimizer interrupt export contract 並驗證 | NEW -> DONE | `validate_optimizer_interrupt_export_contract_case` |
| 2026-04-04 | T134 | 新增 score numerator option contract 並驗證 | NEW -> DONE | `validate_score_numerator_option_case` |
| 2026-04-05 | B21 | 檢出 score header 將評分模型與分子混寫在同一括號，主表先改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_display_cases.py` |
| 2026-04-05 | B21 | 將 score header 改為模型/分子分欄顯示並補契約後收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_display_cases.py` |
| 2026-04-05 | B26 | 檢出 `G` 仍可殘留退役 validator 名稱與重複 `NEW -> *`，主表改回 PARTIAL | DONE -> PARTIAL | checklist 自身完整性仍有歷史回寫缺口 |
| 2026-04-05 | B26 | 補上 `G` 的 `NEW` 首次出現約束與有效 validator reference guard 後收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B26 | 檢出 checklist 首行固定標題仍缺 formal guard，主表改回 PARTIAL | DONE -> PARTIAL | checklist 自身完整性仍有檔首契約缺口 |
| 2026-04-05 | B26 | 補上 checklist 首行固定標題 guard 後重新收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B26 | 檢出 checklist 摘要表固定升冪排序仍缺 formal guard，主表改回 PARTIAL | DONE -> PARTIAL | checklist 自身完整性仍有摘要表排序契約缺口 |
| 2026-04-05 | B26 | 補上 checklist 摘要表固定升冪排序 guard 後重新收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B26 | 檢出 `G` 同日區塊新增列後未整段重排，造成排序 guard 再次被真實 bundle 擊中，主表改回 PARTIAL | DONE -> PARTIAL | checklist 同日區塊回寫仍有排序失誤 |
| 2026-04-05 | B26 | 修正 2026-04-05 同日區塊排序並補強前後鄰列核對要求後重新收斂為 DONE | PARTIAL -> DONE | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-05 | B43 | 依新規格調整 package_zip：root bundle 不得移入 arch，主表先改回 PARTIAL | DONE -> PARTIAL | root `to_chatgpt_bundle_*.zip` 應保留於 root |
| 2026-04-05 | B43 | 改為只歸檔非 bundle 舊 ZIP 並保留 root bundle copy 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_cli_cases.py` |
| 2026-04-05 | B45 | 檢出 `write_issue_log` / `build_timestamped_log_path` 仍可接受 outputs 根目錄，主表改回 PARTIAL | DONE -> PARTIAL | log path create-path outputs-root guard 仍有缺口 |
| 2026-04-05 | B45 | 補上 outputs-root create-path guard 後重新收斂為 DONE | PARTIAL -> DONE | `core/log_utils.py` |
| 2026-04-05 | B45 | 檢出 `resolve_log_dir("outputs")` 先命中 generic root-dir 錯誤，未維持 outputs-root 專屬拒絕語意，主表改回 PARTIAL | DONE -> PARTIAL | outputs-root guard 判斷順序錯誤 |
| 2026-04-05 | B45 | 調整 `resolve_log_dir` 判斷順序後重新收斂為 DONE | PARTIAL -> DONE | `core/log_utils.py` |
| 2026-04-05 | B58 | 補 `use_compounding` unsupported-value fail-fast guard 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_guardrail_cases.py` |
| 2026-04-05 | B59 | 補關鍵 helper single-source-of-truth static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B60 | 補 `PROJECT_SETTINGS.md` dynamic-test / formal-step bypass boundary contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B61 | 補 policy/config coverage-target static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B62 | 新增 checklist 首行固定標題 contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B62 | 檢出 T141 直接依賴巢狀 `result["extra"]` 讀取 summary 欄位，造成首行 guard synthetic validator 假失敗，主表改回 PARTIAL | DONE -> PARTIAL | checklist 首行契約的 synthetic validator payload 讀法仍綁定舊 schema |
| 2026-04-05 | B62 | 改以 shared summary-value accessor 相容 flattened payload，並補 static contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B63 | 新增 synthetic meta summary-value accessor contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B64 | 新增 checklist 摘要表固定升冪排序 contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B65 | 新增 package_zip commit → zip → test_suite orchestration contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_cli_cases.py` |
| 2026-04-05 | B66 | 新增 GUI workbench / debug artifact contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B67 | 修正單股 debug chart hooks 在無圖模式仍存取 `chart_context` 的 runtime regression，並補 no-op contract 後收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B68 | 新增 module_path repo-relative normalization contract，並以 shared normalizer 收斂跨 OS 路徑穩定性 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B69 | 新增 path-like module path normalization contract，釘死 `Path` 輸入也必須維持穩定 repo-relative 輸出 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B70 | 補 shared module-loader path helper 在 `PROJECT_ROOT` 被 patch 成字串時的相容 contract，避免 synthetic error-path 測試環境再次觸發 helper 自身回歸 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B71 | 新增 GUI 內嵌 K 線圖 viewport / autoscale contract，釘死大型內嵌檢視與可視區間縮放比例 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B72 | 新增 synthetic validator 外部 alias 顯式 import contract，並修正 `synthetic_contract_cases.py` 缺失 `numpy as np` 導致的 coverage synthetic suite 假失敗 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B73 | 新增 synthetic alias 掃描 AST contract，避免以字串搜尋 `np.` 誤判 validator 自述內容為實際 numpy alias 使用 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B74 | 檢出 GUI chart workspace contract 以 `figure.axes` 長度硬編碼 overlay volume axis，跨 matplotlib backend / inset axes 註冊差異造成 synthetic 假失敗，主表先改回 PARTIAL | DONE -> PARTIAL | overlay axis presence contract 寫法過度依賴 backend 細節 |
| 2026-04-05 | B74 | 改以 shared chart contract 驗證 overlay axis 存在性並同步壓低 chart payload 記憶體占用後重新收斂為 DONE | PARTIAL -> DONE | `tools/debug/charting.py` |
| 2026-04-05 | B75 | 新增 GUI toolbar-free mouse navigation contract，釘死滾輪縮放、左鍵拖曳平移與 volume toggle 不得依賴已銷毀 toolbar widget | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B76 | 新增 synthetic validator chart-navigation helper 顯式 import contract，避免遺漏 `bind_matplotlib_chart_navigation` 在 formal suite 才以 `NameError` 造成假失敗 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B77 | 新增 debug entry marker optional-plan contract，釘死 entry_plan=None 時必須 no-op 不得炸出 NoneType 下標錯誤 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B78 | 新增 debug chart payload 無需 HTML export contract，釘死 `export_chart=False` 也可回傳 payload 以避免 synthetic suite 額外載入 plotly 造成記憶體回歸 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | T27 | 補 scanner / dashboard score header 顯示契約後收斂完成 | DONE -> PARTIAL | display header contract 缺口待補。 |
| 2026-04-05 | T27 | 補 scanner / dashboard score header 顯示契約並驗證 | PARTIAL -> DONE | validate_display_reporting_sanity_case |
| 2026-04-05 | T116 | 依新規格調整 package_zip runtime contract：root bundle 不得移入 arch，建議測試先改回 PARTIAL | DONE -> PARTIAL | root `to_chatgpt_bundle_*.zip` 應保留於 root |
| 2026-04-05 | T116 | 改為只歸檔非 bundle 舊 ZIP 並保留 root bundle copy 後重新收斂 | PARTIAL -> DONE | `validate_package_zip_runtime_contract_case` |
| 2026-04-05 | T121 | 檢出 quick_gate log-path contract 尚未覆蓋 outputs-root create path，建議測試先改回 PARTIAL | DONE -> PARTIAL | 尚未釘死拒絕 outputs 根目錄寫入 |
| 2026-04-05 | T121 | 補上 outputs-root create-path guard 並重新收斂 | PARTIAL -> DONE | `validate_quick_gate_output_path_guard_contract_case` |
| 2026-04-05 | T121 | 檢出 outputs-root create-path guard 錯誤語意被 generic root-dir 檢查覆蓋，建議測試先改回 PARTIAL | DONE -> PARTIAL | outputs-root 錯誤語意未命中 |
| 2026-04-05 | T121 | 修正 outputs-root 專屬拒絕語意後重新收斂 | PARTIAL -> DONE | `validate_quick_gate_output_path_guard_contract_case` |
| 2026-04-05 | T124 | 檢出 `G` 同日區塊新增列後未整段重排，排序 guard 再次被真實 bundle 擊中 | DONE -> PARTIAL | `validate_checklist_g_ordering_case` |
| 2026-04-05 | T124 | 修正同日區塊排序並補強前後鄰列核對流程後重新收斂 | PARTIAL -> DONE | `validate_checklist_g_ordering_case` |
| 2026-04-05 | T135 | 新增 `use_compounding=False` fail-fast guardrail 並驗證 | NEW -> DONE | `validate_use_compounding_failfast_guardrail_case` |
| 2026-04-05 | T136 | 新增關鍵 helper single-source-of-truth contract 並驗證 | NEW -> DONE | `validate_critical_helper_single_source_contract_case` |
| 2026-04-05 | T137 | 新增 GPT 端 dynamic-test / formal-step bypass boundary contract 並驗證 | NEW -> DONE | `validate_project_settings_dynamic_test_boundary_case` |
| 2026-04-05 | T138 | 新增 policy/config coverage-target contract 並驗證 | NEW -> DONE | `validate_policy_contract_modules_in_coverage_targets_case` |
| 2026-04-05 | T139 | 新增 `G` 的 `NEW` 只能出現在首次收斂紀錄的 guard 並驗證 | NEW -> DONE | `validate_checklist_g_new_transition_first_occurrence_case` |
| 2026-04-05 | T139 | 檢出 synthetic mutation target 指到首筆合法歷史列，guard 驗證暫時失真 | DONE -> PARTIAL | `validate_checklist_g_new_transition_first_occurrence_case` |
| 2026-04-05 | T139 | 改為鎖定同 ID 非首次出現列後，guard synthetic validator 重新收斂 | PARTIAL -> DONE | `validate_checklist_g_new_transition_first_occurrence_case` |
| 2026-04-05 | T140 | 新增 `G` 備註欄 validator reference existence guard 並驗證 | NEW -> DONE | `validate_checklist_g_note_validate_reference_exists_case` |
| 2026-04-05 | T141 | 新增 checklist 首行固定標題 guard 並驗證 | NEW -> DONE | `validate_checklist_first_nonempty_line_case` |
| 2026-04-05 | T141 | 檢出 synthetic case 直接依賴巢狀 `result["extra"]` 導致 formal suite 假失敗 | DONE -> PARTIAL | `validate_checklist_first_nonempty_line_case` |
| 2026-04-05 | T141 | 改以 shared summary-value accessor 讀取 flattened payload 後重新收斂 | PARTIAL -> DONE | `validate_checklist_first_nonempty_line_case` |
| 2026-04-05 | T142 | 新增 synthetic meta summary-value accessor static contract 並驗證 | NEW -> DONE | `validate_synthetic_meta_cases_summary_value_accessor_contract_case` |
| 2026-04-05 | T143 | 新增 checklist 摘要表固定升冪排序 guard 並驗證 | NEW -> DONE | `validate_checklist_summary_tables_sorted_by_id_case` |
| 2026-04-05 | T144 | 新增 package_zip commit → zip → test_suite orchestration contract 並驗證 | NEW -> DONE | `validate_package_zip_commit_test_suite_orchestration_case` |
| 2026-04-05 | T145 | 新增 GUI workbench / debug artifact contract 並驗證 | NEW -> DONE | `validate_gui_workbench_contract_case` |
| 2026-04-05 | T146 | 新增 debug chart hooks 無圖 no-op contract 並驗證 | NEW -> DONE | `validate_debug_trade_log_chart_context_optional_case` |
| 2026-04-05 | T147 | 新增 module_path repo-relative normalization contract 並驗證 | NEW -> DONE | `validate_tool_module_path_normalization_case` |
| 2026-04-05 | T148 | 新增 path-like module path normalization contract 並驗證 | NEW -> DONE | `validate_module_path_normalizer_accepts_path_objects_case` |
| 2026-04-05 | T149 | 新增 `PROJECT_ROOT` string-patch 相容 contract 並驗證 | NEW -> DONE | `validate_module_loader_project_root_string_patch_case` |
| 2026-04-05 | T150 | 新增 GUI 內嵌 K 線圖 viewport / autoscale contract 並驗證 | NEW -> DONE | `validate_gui_embedded_chart_contract_case` |
| 2026-04-05 | T151 | 新增 synthetic validator 外部 alias 顯式 import contract 並驗證 | NEW -> DONE | `validate_synthetic_case_numpy_alias_import_contract_case` |
| 2026-04-05 | T152 | 新增 synthetic alias 掃描忽略字串常值 contract 並驗證 | NEW -> DONE | `validate_synthetic_case_numpy_alias_scan_ignores_string_literals_contract_case` |
| 2026-04-05 | T153 | 檢出 GUI chart workspace contract 將 overlay axes count 寫死為 2，跨 backend 造成 synthetic 假失敗 | DONE -> PARTIAL | `validate_gui_chart_workspace_contract_case` |
| 2026-04-05 | T153 | 改以 shared chart contract 驗證 overlay axis 存在性並重新收斂 | PARTIAL -> DONE | `validate_gui_chart_workspace_contract_case` |
| 2026-04-05 | T154 | 新增 GUI toolbar-free mouse navigation / no-toolbar volume toggle contract 並驗證 | NEW -> DONE | `validate_gui_mouse_navigation_contract_case` |
| 2026-04-05 | T155 | 新增 chart-navigation binder 顯式 import static contract 並驗證 | NEW -> DONE | `validate_synthetic_case_chart_navigation_binder_import_contract_case` |
| 2026-04-05 | T156 | 新增 debug entry marker optional-plan contract 並驗證 | NEW -> DONE | `validate_debug_entry_plan_marker_optional_contract_case` |
| 2026-04-05 | T157 | 新增 debug chart payload 無需 HTML export contract 並驗證 | NEW -> DONE | `validate_debug_chart_payload_without_html_export_contract_case` |

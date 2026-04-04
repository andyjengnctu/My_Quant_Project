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
- `Bxx` / `Cxx` 為主表穩定追蹤 ID。
- `Dxx` 為建議測試項目的 stable tracking ID namespace；即使舊 `D` 區已刪除，既有 `Dxx` 仍持續沿用，不重編、不回收。

收斂原則：
1. 本檔主責維護主表、覆蓋狀態、對應測試入口與收斂紀錄；原則性規則不在本檔重寫。
2. 先補長期固定測試，再補可隨策略升級調整的測試；優先補 synthetic / unit / contract test，避免讓 GPT 端重跑本地完整動態流程。
3. test suite 應優先驗證規格、契約與 invariant，避免綁死當前 ML / DRL / LLM 策略實作細節。
4. 每完成一項，需同步更新本表狀態、對應測試入口與結果摘要；若新增測試導致模組責任改變，再更新 `doc/ARCHITECTURE.md` 與 `doc/CMD.md`。
5. 主表狀態為唯一真理來源；同步順序固定為先改主表，再同步 `F` / `G`；若同輪存在未完成缺口，再同步 `E`。任何 `Bxx` / `Dxx` / 狀態變更，必須同一次 patch 更新完畢。
6. 摘要表只保留最小必要欄位：`E` 固定只留一張未完成暫存表；`F` 不重複抄寫完成日期；完成日期與狀態時間軸一律只記於 `G`。
7. `F` 每列只記一個 `Dxx` 與一個測試入口；不得在同列混寫多個 validator 或 script。`E` / `F` 固定依 ID 升冪排序；`G` 僅記錄實際狀態變更，且固定依日期升冪、同日再依 ID 升冪整理；純補充說明改寫為表格外文字，不得再寫 `DONE -> DONE`、`PARTIAL -> PARTIAL` 等無狀態變更列。

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
| B21 | P2 | 顯示 | 報表欄位、排序、百分比格式與來源一致 | DONE | 已補 scanner header / start banner / summary、strategy dashboard、validate console summary、issue Excel report schema、portfolio yearly/export report，以及 `apps/test_suite.py` 在 PASS / FAIL / manifest-blocked / partial-selected-steps / preflight-failed / dataset-prepare-failed / summary-unreadable 下的人類可讀摘要契約；另補 checklist status vocabulary sync 與 meta quality coverage line/branch/min-threshold/missing-zero-target guard 摘要顯示，並以 `run_all.py` contract 釘死 preflight 早退時 dataset_prepare 仍需標記為 `NOT_RUN`，避免 real path 誤落成 `missing_summary_file` | `tools/validate/synthetic_display_cases.py`, `tools/validate/synthetic_reporting_cases.py`, `tools/validate/synthetic_contract_cases.py`, `core/display.py`, `tools/scanner/reporting.py`, `tools/portfolio_sim/reporting.py`, `apps/test_suite.py`, `tools/local_regression/run_all.py` |
| B22 | P2 | 覆蓋率 | line / branch coverage 報表 | DONE | 已將 `run_meta_quality.py` 的 synthetic coverage suite、formal helper probe、key target presence/hit 與 manifest 化 line / branch minimum threshold gate 收斂為正式路徑，並同步回寫 `meta_quality_summary.json` / `apps/test_suite.py` 摘要顯示 | `tools/local_regression/run_meta_quality.py`, `tools/local_regression/common.py`, `apps/test_suite.py` |
| B23 | P1 | Meta | checklist / 測試註冊 / 正式入口一致性 | DONE | 已補 synthetic 主入口遺漏註冊案例，並新增 imported / defined `validate_*` case、formal pipeline registry / formal-entry / run_all / preflight / test_suite 一致性 formal guard，以及 `core/` / `tools/` 不得反向 import `apps/` 的分層 guard；正式步驟單一真理來源已收斂到 `tools/local_regression/formal_pipeline.py` | `tools/validate/synthetic_meta_cases.py`, `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_cases.py`, `tools/local_regression/formal_pipeline.py` |
| B24 | P1 | Meta | known-bad fault injection：關鍵規則故意破壞後測試必須 fail | DONE | 已新增 meta fault-injection case，直接對 same-day sell、same-bar stop priority、fee/tax、history filter misuse 注入 known-bad 行為，並驗證既有測試會產生 FAIL | `tools/validate/synthetic_meta_cases.py` |
| B25 | P1 | Meta | independent oracle / golden cases：高風險數值規則不可只與 production 共用同邏輯 | DONE | 已新增獨立 oracle golden case，對 net sell、position size、history EV、annual return / sim years 以手算或獨立公式對照 production | `tools/validate/synthetic_unit_cases.py` |
| B26 | P1 | Meta | checklist 是否已足夠覆蓋完整性（包含 test suite 本身） | DONE | 已補主表 / `E` / `F` / `G` 收斂同步 formal guard，並阻擋 convergence 紀錄失同步、`F` 以 `+`、`/`、`,` 或多個 code reference 混寫多個測試入口、`G` 備註欄混寫多個測試入口、`G` transition 缺少合法狀態轉移格式，以及 legacy `D` / `E1-E3` / `F1` 區不得回流；checklist 自身完整性已納入正式 gate | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/meta_contracts.py`, `doc/TEST_SUITE_CHECKLIST.md` |
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
| B43 | P1 | I/O | `apps/package_zip.py` 正式入口必須驗證舊 ZIP 全數歸檔，且輸出 ZIP 不得夾帶 Python 快取 | DONE | 已補 `package_zip` runtime contract，直接釘死 root 既有舊 ZIP 不分 branch label 都必須移入 `arch/`，且新 ZIP 僅可包含 tracked/untracked 非忽略檔，不得夾帶 `__pycache__/` 或 `*.pyc` | `apps/package_zip.py`, `tools/validate/synthetic_cli_cases.py`, `tools/validate/synthetic_cases.py` |
| B44 | P1 | Meta | `quick_gate` 不得移除裸 `except` static guard | DONE | 已補 `quick_gate` bare-except guard contract，直接釘死 `run_static_checks()` 必須保留 `bare_except_scan`，且遇到裸 `except` 時必須回報 FAIL 與命中文件 | `tools/local_regression/run_quick_gate.py`, `tools/validate/synthetic_contract_cases.py`, `tools/validate/synthetic_cases.py` |
| B45 | P1 | I/O | `quick_gate` 不得移除 output path / outputs root / log path guard | DONE | 已補 `quick_gate` output-path guard contract，直接釘死正式入口 summary 必須保留 `output_path_contract`、`outputs_root_layout`、`log_path_contract` 關鍵步驟，且 guard FAIL 時必須正確傳遞到 `failed_steps` | `tools/local_regression/run_quick_gate.py`, `core/output_paths.py`, `core/log_utils.py`, `tools/validate/synthetic_contract_cases.py`, `tools/validate/synthetic_cases.py` |
| B46 | P1 | 錯誤處理 | formal pipeline 關鍵 fallback / console tail 不得靜默吞掉非 cleanup I/O 例外 | DONE | 已補 dataset prepare fallback summary write traceability 與 console tail read-error traceability contract，直接釘死 `run_all.py` fallback 寫檔失敗必須回寫 `fallback_write_errors` / stderr，且 `gather_recent_console_tail()` 讀檔失敗不得靜默略過 | `tools/local_regression/run_all.py`, `tools/local_regression/common.py`, `tools/validate/synthetic_contract_cases.py`, `tools/validate/synthetic_cases.py` |

### B3. 可隨策略升級調整的測試

| ID | 優先級 | 類別 | 項目 | 原則 | 建議落點 |
|---|---|---|---|---|---|
| C01 | P1 | 模型介面 | model feature schema / prediction schema 穩定 | 驗輸入欄位、輸出欄位、型別、缺值處理；不驗固定分數 | `tools/optimizer/`, `tools/scanner/` |
| C02 | P1 | 重現性 | 同 seed 下 optimizer / model inference 可重現 | 驗結果穩定在可接受範圍；不綁死搜尋路徑細節 | `tools/local_regression/`, `tools/optimizer/` |
| C03 | P1 | 排序輸出 | ranking / scoring 輸出可排序、可比較、無 NaN | 驗排序值可用、方向一致、型別正確；不驗固定名次 | `core/buy_sort.py`, `tools/scanner/` |
| C04 | P2 | 最低可用性 | 模型升級後 scanner / optimizer / reporting 仍可跑通 | 驗 CLI 與輸出仍可用；不驗內部中間流程 | `apps/ml_optimizer.py`, `apps/vip_scanner.py` |
| C05 | P2 | 報表相容 | 新策略輸出仍符合既有 artifact / reporting schema | 驗欄位存在與語意；不驗具體績效數值 | `tools/portfolio_sim/`, `tools/scanner/reporting.py` |
| C06 | P1 | Optimizer 契約 | objective 淘汰值 / fail_reason / profile_row / best_params export 穩定 | 驗 `INVALID_TRIAL_VALUE`、fail_reason、profile_row、`tp_percent` 還原優先序與 export 成敗；不驗固定分數 | `tools/optimizer/objective_runner.py`, `tools/optimizer/runtime.py`, `tools/optimizer/study_utils.py` |

## E. 未完成缺口摘要

使用方式：本節僅在同輪無法一次清空時暫存未完成缺口；若本輪已清空，維持空表即可。主維護來源仍是主表；本節固定只保留一張未完成暫存表，避免 `E1` / `E2` / `E3` 分拆維護。

| ID | 目前狀態 | 項目 | 缺口摘要 / 對應主表項目 | 建議落點 |
|---|---|---|---|---|

## F. 已完成建議測試映射

使用方式：本節只保留 `DONE` 的建議測試項目最小必要索引；不重複抄寫主表的建議落點，也不重複記錄完成日期。主表狀態、測試入口細節與缺口摘要仍以主表為準，時間軸僅寫在 `G`。

維護規則：`F` 固定只留「ID / 建議測試名稱 / 對應主表項目」，並依 ID 升冪排序。交付前至少核對一次「所有已註冊 validator / script 類型的 `Dxx` 已同步列入 `F`，且 `F` 與 `G` 的最新狀態一致」。

| ID | 建議測試名稱 | 對應主表項目 |
|---|---|---|
| D01 | `validate_synthetic_same_day_buy_sell_forbidden_case` | B06 |
| D02 | `validate_synthetic_intraday_reprice_forbidden_case` | B05 |
| D03 | `validate_synthetic_no_intraday_switch_after_failed_fill_case` | B07 |
| D04 | `validate_synthetic_exit_orders_only_for_held_positions_case` | B08 |
| D05 | `validate_synthetic_fee_tax_net_equity_case` | B03 |
| D06 | `validate_synthetic_round_trip_pnl_only_on_tail_exit_case` | B04 |
| D07 | `validate_synthetic_missed_sell_accounting_case` | B11 |
| D08 | `validate_synthetic_candidate_order_fill_layer_separation_case` | B09 |
| D09 | `validate_synthetic_portfolio_history_filter_only_case` | B10 |
| D10 | `validate_synthetic_lookahead_prev_day_only_case` | B01 |
| D11 | `validate_price_utils_unit_case` | B13 |
| D12 | `validate_history_filters_unit_case` | B13 |
| D13 | `validate_portfolio_stats_unit_case` | B13 |
| D14 | `validate_model_io_schema_case` | C01 |
| D15 | `tools/local_regression/run_ml_smoke.py` | B12 |
| D16 | `validate_ranking_scoring_sanity_case` | C03 |
| D17 | `tools/validate/synthetic_reporting_cases.py` | B21 |
| D18 | `validate_output_contract_case` | B11 / B17 |
| D19 | `tools/local_regression/run_chain_checks.py` | B18 |
| D20 | `tools/local_regression/run_meta_quality.py` | B22 |
| D21 | `core/runtime_utils.py` | B19 |
| D22 | `validate_registry_checklist_entry_consistency_case` | B23 |
| D23 | `validate_known_bad_fault_injection_case` | B24 |
| D24 | `validate_independent_oracle_golden_case` | B25 |
| D25 | `tools/validate/meta_contracts.py` | B26 |
| D26 | `validate_cmd_document_contract_case` | B20 |
| D27 | `validate_display_reporting_sanity_case` | B21 |
| D28 | `validate_artifact_lifecycle_contract_case` | B17 |
| D29 | `tools/local_regression/formal_pipeline.py` | B23 |
| D30 | `validate_params_io_error_path_case` | B15 |
| D31 | `validate_module_loader_error_path_case` | B15 |
| D32 | `validate_preflight_error_path_case` | B15 |
| D33 | `validate_sanitize_ohlcv_expected_behavior_case` | B14 |
| D34 | `validate_sanitize_ohlcv_failfast_case` | B14 |
| D35 | `validate_load_clean_df_data_quality_case` | B14 |
| D36 | `validate_dataset_cli_contract_case` | B16 |
| D37 | `validate_local_regression_cli_contract_case` | B16 |
| D41 | `tools/scanner/scan_runner.py` | B12 / B18 |
| D42 | `validate_issue_excel_report_schema_case` | B21 |
| D43 | `validate_portfolio_export_report_artifacts_case` | B21 |
| D45 | `validate_extended_tool_cli_contract_case` | B16 |
| D46 | `validate_downloader_market_date_fallback_case` | B15 |
| D47 | `validate_downloader_sync_error_path_case` | B15 |
| D48 | `validate_downloader_main_error_path_case` | B15 |
| D49 | `validate_local_regression_summary_contract_case` | B11 |
| D50 | `validate_test_suite_summary_failure_reporting_case` | B21 |
| D51 | `validate_test_suite_summary_manifest_failure_reporting_case` | B21 |
| D52 | `validate_test_suite_summary_optional_dataset_skip_case` | B21 |
| D53 | `validate_test_suite_summary_preflight_failure_reporting_case` | B21 |
| D54 | `validate_test_suite_summary_dataset_prepare_failure_reporting_case` | B21 |
| D55 | `validate_test_suite_summary_unreadable_payload_reporting_case` | B21 |
| D56 | `validate_run_all_preflight_early_failure_dataset_contract_case` | B11 / B17 |
| D57 | `validate_test_suite_summary_checklist_status_sync_case` | B21 |
| D58 | `validate_test_suite_summary_meta_quality_guardrail_reporting_case` | B21 / B22 |
| D59 | `validate_single_formal_test_entry_contract_case` | B26 |
| D60 | `validate_synthetic_setup_index_prev_day_only_case` | B01 |
| D61 | `validate_downloader_universe_fetch_error_path_case` | B15 |
| D62 | `validate_downloader_universe_screening_init_error_path_case` | B15 |
| D63 | `validate_meta_quality_performance_memory_contract_case` | B19 |
| D64 | `validate_test_suite_summary_meta_quality_memory_reporting_case` | B19 / B21 |
| D65 | `validate_portfolio_sim_prepared_tool_contract_case` | B11 / B19 |
| D66 | `validate_scanner_prepared_tool_contract_case` | B19 |
| D67 | `validate_debug_trade_log_prepared_tool_contract_case` | B19 |
| D68 | `validate_scanner_reference_clean_df_contract_case` | B19 |
| D69 | `validate_meta_quality_reuses_existing_coverage_artifacts_case` | B19 / B22 |
| D70 | `tools/validate/synthetic_cases.py` | B23 |
| D71 | `tools/validate/synthetic_meta_cases.py` | B26 |
| D72 | `apps/test_suite.py` | B26 |
| D73 | `validate_no_reverse_app_layer_dependencies_case` | B23 |
| D74 | `validate_run_all_manifest_failure_master_summary_contract_case` | B17 |
| D75 | `validate_synthetic_same_bar_stop_priority_case` | B02 |
| D76 | `validate_synthetic_half_tp_full_year_case` | B04 / B21 |
| D77 | `validate_synthetic_extended_miss_buy_case` | B09 |
| D78 | `validate_synthetic_competing_candidates_case` | B09 |
| D79 | `validate_synthetic_same_day_sell_block_case` | B06 |
| D80 | `validate_synthetic_rotation_t_plus_one_case` | B05 / B06 |
| D81 | `validate_synthetic_missed_buy_no_replacement_case` | B07 |
| D82 | `validate_synthetic_unexecutable_half_tp_case` | B04 |
| D83 | `validate_synthetic_history_ev_threshold_case` | B10 |
| D84 | `validate_synthetic_single_backtest_not_gated_by_own_history_case` | B10 |
| D85 | `validate_synthetic_pit_same_day_exit_excluded_case` | B01 |
| D86 | `validate_synthetic_pit_multiple_same_day_exits_case` | B01 |
| D87 | `validate_synthetic_proj_cost_cash_capped_case` | B09 |
| D88 | `validate_synthetic_param_guardrail_case` | B15 |
| D89 | `validate_validate_console_summary_reporting_case` | B21 |
| D90 | `validate_portfolio_yearly_report_schema_case` | B21 |
| D91 | `validate_test_suite_summary_reporting_case` | B21 |
| D92 | `validate_scanner_worker_repeatability_case` | B12 |
| D93 | `validate_scan_runner_repeatability_case` | B12 |
| D94 | `validate_optimizer_raw_cache_rerun_consistency_case` | B18 |
| D95 | `validate_run_all_repeatability_case` | B18 |
| D96 | `validate_no_top_level_import_cycles_case` | B27 |
| D97 | `validate_core_trading_modules_in_coverage_targets_case` | B28 |
| D98 | `validate_critical_file_coverage_minimum_gate_case` | B29 |
| D99 | `validate_coverage_threshold_floor_case` | B30 |
| D100 | `validate_entry_path_critical_coverage_gate_case` | B31 |
| D101 | `validate_critical_coverage_threshold_floor_case` | B32 |
| D102 | `validate_dataset_fingerprint_contract_case` | B33 |
| D103 | `validate_atomic_write_contract_case` | B34 |
| D104 | `tools/local_regression/common.py` | B36 |
| D105 | `validate_test_suite_orchestrator_coverage_targets_case` | B35 |
| D106 | `validate_atomic_write_retry_contract_case` | B34 |
| D107 | `validate_run_all_dataset_prepare_pass_main_contract_case` | B33 |
| D108 | `validate_synthetic_registry_metadata_contract_case` | B37 |
| D109 | `validate_optimizer_objective_export_contract_case` | C06 |
| D110 | `validate_formal_step_entry_coverage_targets_case` | B38 |
| D111 | `validate_checklist_g_single_note_entry_delimiter_case` | B26 |
| D112 | `validate_checklist_f2_single_entry_delimiter_case` | B26 |
| D113 | `validate_checklist_g_transition_format_case` | B26 |
| D114 | `validate_checklist_no_legacy_d_section_case` | B26 |
| D115 | `validate_formal_step_implementation_coverage_targets_case` | B39 |
| D116 | `validate_peak_traced_memory_tracker_context_management_case` | B40 |
| D117 | `validate_run_all_cli_error_usage_contract_case` | B16 |
| D118 | `validate_no_legacy_app_entry_doc_references_case` | B41 |
| D119 | `validate_app_thin_wrapper_export_contract_case` | B42 |
| D120 | `validate_package_zip_runtime_contract_case` | B43 |
| D121 | `validate_synthetic_non_candidate_setup_does_not_seed_extended_signal_case` | B09 |
| D122 | `validate_atomic_write_cleanup_error_preserves_root_exception_case` | B34 |
| D123 | `validate_validate_summary_atomic_write_contract_case` | B34 |
| D124 | `validate_quick_gate_bare_except_guard_contract_case` | B44 |
| D125 | `validate_quick_gate_output_path_guard_contract_case` | B45 |
| D126 | `validate_dataset_prepare_fallback_write_traceability_case` | B46 |
| D127 | `validate_console_tail_read_error_traceability_case` | B46 |
| D128 | `validate_checklist_g_ordering_case` | B26 |
| D129 | `validate_checklist_no_legacy_f1_section_case` | B26 |

## G. 逐項收斂紀錄

使用方式：每次只挑少數高優先項目處理，完成後更新本節，不要重開一份新清單。

| 日期 | 項目 ID | 動作 | 狀態變更 | 備註 |
|---|---|---|---|---|
| 2026-04-01 | D01 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_same_day_buy_sell_forbidden_case |
| 2026-04-01 | D02 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_intraday_reprice_forbidden_case |
| 2026-04-01 | D03 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_no_intraday_switch_after_failed_fill_case |
| 2026-04-01 | D04 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_exit_orders_only_for_held_positions_case |
| 2026-04-01 | D05 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_fee_tax_net_equity_case |
| 2026-04-01 | D06 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_round_trip_pnl_only_on_tail_exit_case |
| 2026-04-01 | D07 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_missed_sell_accounting_case |
| 2026-04-01 | D08 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_candidate_order_fill_layer_separation_case |
| 2026-04-01 | D09 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_portfolio_history_filter_only_case |
| 2026-04-01 | D10 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_lookahead_prev_day_only_case |
| 2026-04-01 | D11 | 新增 unit-like 邊界案例並驗證 | TODO -> DONE | validate_price_utils_unit_case |
| 2026-04-01 | D12 | 新增 unit-like 邊界案例並驗證 | TODO -> DONE | validate_history_filters_unit_case |
| 2026-04-01 | D13 | 新增 unit-like 邊界案例並驗證 | TODO -> DONE | validate_portfolio_stats_unit_case |
| 2026-04-01 | D15 | 新增 optimizer fixed-seed 雙跑一致性檢查 | TODO -> PARTIAL | `run_ml_smoke.py` 已比較雙跑 trial / best_params digest；scanner 尚未補 |
| 2026-04-01 | D18 | 新增 CSV / XLSX / JSON output contract case 並驗證 | TODO -> PARTIAL | validate_output_contract_case |
| 2026-04-01 | D19 | 新增 chain checks 雙跑 digest 對比與 optimizer 雙跑 | TODO -> PARTIAL | 已補雙跑流程，但 scanner 入口尚未收斂。 |
| 2026-04-01 | D20 | 新增 `run_meta_quality.py` 產出 coverage baseline | TODO -> PARTIAL | 目前已覆蓋 synthetic coverage suite 與 key target coverage，並納入正式入口摘要。 |
| 2026-04-01 | D21 | 新增 `run_meta_quality.py` performance baseline gating | TODO -> PARTIAL | 已正式檢查 reduced suite 各步驟 / total duration 與 optimizer 平均 trial wall time；記憶體回歸仍未納入 |
| 2026-04-01 | D22 | 新增 meta registry case 並驗證 | TODO -> DONE | validate_registry_checklist_entry_consistency_case |
| 2026-04-01 | D23 | 新增 meta fault-injection case 並驗證 | TODO -> DONE | validate_known_bad_fault_injection_case |
| 2026-04-01 | D24 | 新增 independent oracle golden case 並驗證 | TODO -> DONE | validate_independent_oracle_golden_case |
| 2026-04-01 | D26 | 新增 CMD 指令契約案例並驗證 | TODO -> DONE | validate_cmd_document_contract_case |
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
| 2026-04-02 | B26 | 補齊 checklist main / `F` / `G` sync blocker 後收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | D14 | 新增 model I/O schema 案例並驗證 | TODO -> DONE | `validate_model_io_schema_case` |
| 2026-04-02 | D15 | 補 scanner worker / `scan_runner` 入口重跑一致性後收斂完成 | PARTIAL -> DONE | `validate_scanner_worker_repeatability_case` |
| 2026-04-02 | D16 | 新增 ranking / scoring sanity 案例並驗證 | TODO -> DONE | `validate_ranking_scoring_sanity_case` |
| 2026-04-02 | D17 | reporting schema compatibility checks 收斂完成，並新增輸出檔 schema 補強 | TODO -> DONE | `validate_issue_excel_report_schema_case` |
| 2026-04-02 | D18 | 擴充 local regression summary contract 並收斂完成 | PARTIAL -> DONE | `validate_output_contract_case` |
| 2026-04-02 | D19 | 補 `run_all.py` 同 run dir rerun summary / bundle repeatability 後收斂完成 | PARTIAL -> DONE | `validate_optimizer_raw_cache_rerun_consistency_case` |
| 2026-04-02 | D20 | 補 manifest 化 line / branch threshold gate 與 summary sync | PARTIAL -> DONE | `run_meta_quality.py` |
| 2026-04-02 | D21 | 補 traced peak memory regression gate 後 performance baseline 收斂完成 | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | D25 | 擴充 checklist sufficiency formal check 到單一正式入口與 legacy entry 檢查後收斂完成 | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | D27 | 擴充 scanner summary / banner 與 display re-export 後收斂完成 | PARTIAL -> DONE | validate_display_reporting_sanity_case |
| 2026-04-02 | D28 | 擴充 artifact lifecycle contract 並驗證 | PARTIAL -> DONE | validate_artifact_lifecycle_contract_case |
| 2026-04-02 | D29 | 新增 formal-entry consistency checks 並驗證 | TODO -> DONE | `run_meta_quality.py` formal-entry consistency checks |
| 2026-04-02 | D30 | 新增 params_io 錯誤路徑案例並驗證 | TODO -> DONE | `validate_params_io_error_path_case` |
| 2026-04-02 | D31 | 新增 module_loader 錯誤路徑案例並驗證 | TODO -> DONE | `validate_module_loader_error_path_case` |
| 2026-04-02 | D32 | 新增 preflight 錯誤路徑案例並驗證 | TODO -> DONE | `validate_preflight_error_path_case` |
| 2026-04-02 | D33 | 新增資料清洗 expected behavior 案例並驗證 | TODO -> DONE | `validate_sanitize_ohlcv_expected_behavior_case` |
| 2026-04-02 | D34 | 新增資料清洗 fail-fast 案例並驗證 | TODO -> DONE | `validate_sanitize_ohlcv_failfast_case` |
| 2026-04-02 | D35 | 新增 `load_clean_df` 資料品質整合案例並驗證 | TODO -> DONE | `validate_load_clean_df_data_quality_case` |
| 2026-04-02 | D36 | 新增 dataset wrapper CLI 契約案例並驗證 | TODO -> DONE | `validate_dataset_cli_contract_case` |
| 2026-04-02 | D37 | 新增 local regression / no-arg CLI 契約案例並驗證 | TODO -> DONE | `validate_local_regression_cli_contract_case` |
| 2026-04-02 | D41 | 新增 scanner reduced snapshot 雙跑 digest 並驗證 | TODO -> DONE | `run_chain_checks.py` 已將 scanner 候選 / 狀態 / issue line 納入 rerun consistency payload |
| 2026-04-02 | D42 | 新增 issue Excel report schema 案例並驗證 | NEW -> DONE | `validate_issue_excel_report_schema_case` |
| 2026-04-02 | D43 | 新增 portfolio export report artifacts 案例並驗證 | NEW -> DONE | `validate_portfolio_export_report_artifacts_case` |
| 2026-04-02 | D45 | 補齊剩餘直接入口 CLI 契約並收斂 B16 | TODO -> DONE | `validate_extended_tool_cli_contract_case` |
| 2026-04-02 | D46 | 新增 downloader market-date fallback 案例並驗證 | TODO -> DONE | `validate_downloader_market_date_fallback_case` |
| 2026-04-02 | D47 | 新增 downloader sync 錯誤聚合案例並驗證 | TODO -> DONE | `validate_downloader_sync_error_path_case` |
| 2026-04-02 | D48 | 新增 downloader main 失敗摘要案例並驗證 | TODO -> DONE | `validate_downloader_main_error_path_case` |
| 2026-04-02 | D49 | 新增 local regression summary contract case 並驗證 | TODO -> DONE | `validate_local_regression_summary_contract_case` |
| 2026-04-02 | D50 | 新增腳本失敗摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_failure_reporting_case` |
| 2026-04-02 | D51 | 新增 manifest blocked 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_manifest_failure_reporting_case` |
| 2026-04-02 | D52 | 新增 partial selected-steps 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_optional_dataset_skip_case` |
| 2026-04-02 | D53 | 新增 preflight fail 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_preflight_failure_reporting_case` |
| 2026-04-02 | D54 | 新增 dataset prepare fail 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_dataset_prepare_failure_reporting_case` |
| 2026-04-02 | D55 | 新增 summary unreadable 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_unreadable_payload_reporting_case` |
| 2026-04-02 | D56 | 補 `run_all.py` preflight 早退 dataset not-run contract 並驗證 | TODO -> DONE | `validate_run_all_preflight_early_failure_dataset_contract_case` |
| 2026-04-02 | D57 | 補 checklist status vocabulary sync 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_checklist_status_sync_case` |
| 2026-04-02 | D58 | 補 meta quality coverage guardrail 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_meta_quality_guardrail_reporting_case` |
| 2026-04-02 | D59 | 新增單一正式測試入口契約案例並驗證 | NEW -> DONE | `validate_single_formal_test_entry_contract_case` |
| 2026-04-02 | D60 | 新增 setup-index prev-day-only synthetic case 並驗證 | NEW -> DONE | `validate_synthetic_setup_index_prev_day_only_case` |
| 2026-04-02 | D61 | 新增 downloader universe fetch fatal error case 並驗證 | NEW -> DONE | `validate_downloader_universe_fetch_error_path_case` |
| 2026-04-02 | D62 | 新增 downloader screening init fatal error case 並驗證 | NEW -> DONE | `validate_downloader_universe_screening_init_error_path_case` |
| 2026-04-02 | D63 | 新增 meta quality performance memory contract case 並驗證 | NEW -> DONE | `validate_meta_quality_performance_memory_contract_case` |
| 2026-04-02 | D64 | 新增 test suite meta quality memory reporting case 並驗證 | NEW -> DONE | `validate_test_suite_summary_meta_quality_memory_reporting_case` |
| 2026-04-02 | D65 | 新增 portfolio_sim prepared tool contract case 並驗證 | NEW -> DONE | `validate_portfolio_sim_prepared_tool_contract_case` |
| 2026-04-02 | D66 | 新增 scanner prepared tool contract case 並驗證 | NEW -> DONE | `validate_scanner_prepared_tool_contract_case` |
| 2026-04-02 | D67 | 新增 debug trade log prepared tool contract case 並驗證 | NEW -> DONE | `validate_debug_trade_log_prepared_tool_contract_case` |
| 2026-04-02 | D68 | 新增 scanner reference clean-df contract case 並驗證 | NEW -> DONE | `validate_scanner_reference_clean_df_contract_case` |
| 2026-04-02 | D69 | 新增 meta quality coverage artifact reuse contract case 並驗證 | NEW -> DONE | `validate_meta_quality_reuses_existing_coverage_artifacts_case` |
| 2026-04-02 | D70 | 新增 imported validate cases vs synthetic registry formal guard 缺口 | NEW -> TODO | 需補正式註冊完整性檢查。 |
| 2026-04-02 | D70 | 補 imported / defined validate cases 與 synthetic registry 完整一致 formal guard | TODO -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-02 | D71 | 新增主表 / `F` / `G` 完整同步 formal guard 缺口 | NEW -> TODO | 需補 checklist 自身同步性檢查。 |
| 2026-04-02 | D71 | 補主表 / `F` / `G` 收斂紀錄完整同步 formal guard | TODO -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | D72 | 新增 checklist 完成映射同步缺口 | NEW -> TODO | 需阻擋已完成 D 項遺漏於 `F` 仍被判定為已收斂 |
| 2026-04-02 | D72 | 補 checklist `DONE` 摘要缺漏自動偵測與阻擋 | TODO -> DONE | `tools/local_regression/run_meta_quality.py` |
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
| 2026-04-03 | C06 | 補 optimizer objective / export contract 最低維護線 | NEW -> DONE | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-03 | D73 | 新增 `core/` / `tools/` 不得反向 import `apps/` 的分層 guard | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-03 | D74 | 新增 manifest failure master summary schema contract 並驗證 | NEW -> DONE | `validate_run_all_manifest_failure_master_summary_contract_case` |
| 2026-04-03 | D75 | 將既有同棒停損優先 synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_same_bar_stop_priority_case` |
| 2026-04-03 | D76 | 將既有半倉停利 full-year synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_half_tp_full_year_case` |
| 2026-04-03 | D77 | 將既有 extended miss buy synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_extended_miss_buy_case` |
| 2026-04-03 | D78 | 將既有 competing candidates synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_competing_candidates_case` |
| 2026-04-03 | D79 | 將既有 same-day sell block synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_same_day_sell_block_case` |
| 2026-04-03 | D80 | 將既有 rotation T+1 synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_rotation_t_plus_one_case` |
| 2026-04-03 | D81 | 將既有 missed buy no replacement synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_missed_buy_no_replacement_case` |
| 2026-04-03 | D82 | 將既有 unexecutable half TP synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_unexecutable_half_tp_case` |
| 2026-04-03 | D83 | 將既有 history EV threshold synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_history_ev_threshold_case` |
| 2026-04-03 | D84 | 將既有 single-backtest own-history guard synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_single_backtest_not_gated_by_own_history_case` |
| 2026-04-03 | D85 | 將既有 PIT same-day exit excluded synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_pit_same_day_exit_excluded_case` |
| 2026-04-03 | D86 | 將既有 PIT multiple same-day exits synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_pit_multiple_same_day_exits_case` |
| 2026-04-03 | D87 | 將既有 projected-cost cash-capped synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_proj_cost_cash_capped_case` |
| 2026-04-03 | D88 | 將既有 param guardrail synthetic case 回寫 checklist | NEW -> DONE | `validate_synthetic_param_guardrail_case` |
| 2026-04-03 | D89 | 將既有 validate console summary reporting contract 回寫 checklist | NEW -> DONE | `validate_validate_console_summary_reporting_case` |
| 2026-04-03 | D90 | 將既有 portfolio yearly report schema contract 回寫 checklist | NEW -> DONE | `validate_portfolio_yearly_report_schema_case` |
| 2026-04-03 | D91 | 將既有 test suite PASS summary reporting contract 回寫 checklist | NEW -> DONE | `validate_test_suite_summary_reporting_case` |
| 2026-04-03 | D92 | 將既有 scanner worker repeatability synthetic case 回寫 checklist | NEW -> DONE | `validate_scanner_worker_repeatability_case` |
| 2026-04-03 | D93 | 將既有 scan runner repeatability synthetic case 回寫 checklist | NEW -> DONE | `validate_scan_runner_repeatability_case` |
| 2026-04-03 | D94 | 將既有 optimizer raw cache rerun consistency case 回寫 checklist | NEW -> DONE | `validate_optimizer_raw_cache_rerun_consistency_case` |
| 2026-04-03 | D95 | 將既有 run_all repeatability case 回寫 checklist | NEW -> DONE | `validate_run_all_repeatability_case` |
| 2026-04-03 | D96 | 新增 top-level import cycle guard 並驗證 | NEW -> DONE | `validate_no_top_level_import_cycles_case` |
| 2026-04-03 | D97 | 新增核心交易模組 coverage target completeness 建議測試項目 | NEW -> TODO | `validate_core_trading_modules_in_coverage_targets_case` |
| 2026-04-03 | D97 | 補上核心交易模組 coverage target completeness guard 並驗證 | TODO -> DONE | `validate_core_trading_modules_in_coverage_targets_case` |
| 2026-04-03 | D98 | 新增 critical file per-file coverage minimum guard 建議測試項目 | NEW -> TODO | `validate_critical_file_coverage_minimum_gate_case` |
| 2026-04-03 | D98 | 補上 critical file per-file coverage minimum guard 並驗證 | TODO -> DONE | `validate_critical_file_coverage_minimum_gate_case` |
| 2026-04-03 | D99 | 新增 coverage threshold floor 建議測試項目 | NEW -> TODO | `validate_coverage_threshold_floor_case` |
| 2026-04-03 | D99 | 補上 coverage threshold floor guard 並驗證 | TODO -> DONE | `validate_coverage_threshold_floor_case` |
| 2026-04-03 | D100 | 新增 entry path critical coverage gate 建議測試並驗證 | NEW -> DONE | `validate_entry_path_critical_coverage_gate_case` |
| 2026-04-03 | D101 | 新增 critical per-file threshold stage-2 floor 建議測試並驗證 | NEW -> DONE | `validate_critical_coverage_threshold_floor_case` |
| 2026-04-03 | D102 | 新增 reduced dataset fingerprint contract 並驗證 | NEW -> DONE | `validate_dataset_fingerprint_contract_case` |
| 2026-04-03 | D103 | 新增 atomic write replace-failure recovery contract 並驗證 | NEW -> DONE | `validate_atomic_write_contract_case` |
| 2026-04-03 | D104 | 擴充 artifact manifest sha256 生成邏輯並由 contract 驗證對照 | NEW -> DONE | `tools/local_regression/common.py` |
| 2026-04-03 | D105 | 新增 test suite orchestrator coverage target completeness guard 並驗證 | NEW -> DONE | `validate_test_suite_orchestrator_coverage_targets_case` |
| 2026-04-03 | D106 | 新增 atomic write transient retry contract 並驗證 | NEW -> DONE | `validate_atomic_write_retry_contract_case` |
| 2026-04-03 | D107 | 新增 run_all dataset prepare PASS 主路徑 contract 並驗證 | NEW -> DONE | `validate_run_all_dataset_prepare_pass_main_contract_case` |
| 2026-04-03 | D108 | 新增 synthetic registry metadata contract case 並驗證 | NEW -> DONE | `validate_synthetic_registry_metadata_contract_case` |
| 2026-04-03 | D109 | 新增 optimizer objective / export contract case 並驗證 | NEW -> DONE | `validate_optimizer_objective_export_contract_case` |
| 2026-04-03 | D110 | 新增 formal step entry wrappers coverage target completeness 建議測試並驗證 | NEW -> DONE | `validate_formal_step_entry_coverage_targets_case` |
| 2026-04-03 | D111 | 新增 `G` 備註欄 delimiter-agnostic single-entry guard 並驗證 | NEW -> DONE | `validate_checklist_g_single_note_entry_delimiter_case` |
| 2026-04-03 | D112 | 新增 `F` 測試入口 delimiter-agnostic single-entry guard 並驗證 | NEW -> DONE | `validate_checklist_f2_single_entry_delimiter_case` |
| 2026-04-03 | D113 | 新增 `G` transition format guard 並驗證 | NEW -> DONE | `validate_checklist_g_transition_format_case` |
| 2026-04-03 | D114 | 新增 checklist legacy `D` 區移除 guard 並驗證 | NEW -> DONE | `validate_checklist_no_legacy_d_section_case` |
| 2026-04-03 | D115 | 新增 split formal-step implementation coverage target completeness guard 並驗證 | NEW -> DONE | `validate_formal_step_implementation_coverage_targets_case` |
| 2026-04-03 | D116 | 新增 memory tracker lifecycle contract 並驗證 | NEW -> DONE | `validate_peak_traced_memory_tracker_context_management_case` |
| 2026-04-04 | B41 | 移除 legacy app 測試入口文件殘留與手動刪檔指引後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-04 | B42 | 補上 app thin wrapper public export contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-04 | B43 | 補上 package_zip 正式入口 runtime contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_cli_cases.py` |
| 2026-04-04 | B44 | 補上 quick_gate bare-except static guard contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B45 | 補上 quick_gate output path / outputs root / log path guard contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B46 | 補上 formal pipeline fallback / console tail traceability contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | D117 | 新增 run_all CLI error usage contract 並驗證 | NEW -> DONE | `validate_run_all_cli_error_usage_contract_case` |
| 2026-04-04 | D118 | 新增 legacy app 測試入口文件殘留 guard 並驗證 | NEW -> DONE | `validate_no_legacy_app_entry_doc_references_case` |
| 2026-04-04 | D119 | 新增 app thin wrapper lazy export contract 並驗證 | NEW -> DONE | `validate_app_thin_wrapper_export_contract_case` |
| 2026-04-04 | D120 | 新增 package_zip runtime contract 並驗證 | NEW -> DONE | `validate_package_zip_runtime_contract_case` |
| 2026-04-04 | D121 | 新增 non-candidate setup 不得 seed / revive extended candidate synthetic case 並驗證 | NEW -> DONE | `validate_synthetic_non_candidate_setup_does_not_seed_extended_signal_case` |
| 2026-04-04 | D122 | 新增 atomic write cleanup failure contract 並驗證 | NEW -> DONE | `validate_atomic_write_cleanup_error_preserves_root_exception_case` |
| 2026-04-04 | D123 | 新增 validate summary 正式路徑 atomic write contract 並驗證 | NEW -> DONE | `validate_validate_summary_atomic_write_contract_case` |
| 2026-04-04 | D124 | 新增 quick_gate bare-except guard contract 並驗證 | NEW -> DONE | `validate_quick_gate_bare_except_guard_contract_case` |
| 2026-04-04 | D125 | 新增 quick_gate output path / outputs root / log path guard contract 並驗證 | NEW -> DONE | `validate_quick_gate_output_path_guard_contract_case` |
| 2026-04-04 | D126 | 新增 dataset prepare fallback write traceability contract 並驗證 | NEW -> DONE | `validate_dataset_prepare_fallback_write_traceability_case` |
| 2026-04-04 | D127 | 新增 console tail read-error traceability contract 並驗證 | NEW -> DONE | `validate_console_tail_read_error_traceability_case` |
| 2026-04-04 | D128 | 新增 checklist `G` 日期 / ID 排序 guard 並驗證 | NEW -> DONE | `validate_checklist_g_ordering_case` |

| 2026-04-04 | D129 | 新增 legacy F1 回流 guard 並驗證 | NEW -> DONE | validate_checklist_no_legacy_f1_section_case |

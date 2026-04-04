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
7. `T` 每列只記一個 `Txx` 與一個測試入口；不得在同列混寫多個 validator 或 script。各摘要表固定依 ID 升冪排序；`G` 僅記錄實際狀態變更，且固定依日期升冪、同日再依 ID 升冪整理；新增或改寫 `G` 列時，不得直接 append 到當日區塊尾端，必須先完成同日區塊內依 ID 重排後再交付；任何 `G` 列只能寫在 `## G. 逐項收斂紀錄` 表格內，檔案開頭第一個非空行固定為 `# Test Suite 收斂清單`；`G` 備註欄最多只能保留一個 code/path/test entry，若同輪涉及多個檔案或測試，僅保留單一代表 entry，其餘改寫為一般文字；純補充說明改寫為表格外文字，不得再寫 `DONE -> DONE`、`PARTIAL -> PARTIAL` 等無狀態變更列。

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
| B21 | P2 | 顯示 | 報表欄位、排序、百分比格式與來源一致 | DONE | 已補 scanner header / start banner / summary、strategy dashboard、validate console summary、issue Excel report schema、portfolio yearly/export report，以及 `apps/test_suite.py` 在 PASS / FAIL / manifest-blocked / partial-selected-steps / preflight-failed / dataset-prepare-failed / summary-unreadable 下的人類可讀摘要契約；另補 checklist status vocabulary sync 與 meta quality coverage line/branch/min-threshold/missing-zero-target guard 摘要顯示，並以 `run_all.py` contract 釘死 preflight 早退時 dataset_prepare 仍需標記為 `NOT_RUN`，避免 real path 誤落成 `missing_summary_file` | `tools/validate/synthetic_display_cases.py`, `tools/validate/synthetic_reporting_cases.py`, `tools/validate/synthetic_contract_cases.py`, `core/display.py`, `tools/scanner/reporting.py`, `tools/portfolio_sim/reporting.py`, `apps/test_suite.py`, `tools/local_regression/run_all.py` |
| B22 | P2 | 覆蓋率 | line / branch coverage 報表 | DONE | 已將 `run_meta_quality.py` 的 synthetic coverage suite、formal helper probe、key target presence/hit 與 manifest 化 line / branch minimum threshold gate 收斂為正式路徑，並同步回寫 `meta_quality_summary.json` / `apps/test_suite.py` 摘要顯示 | `tools/local_regression/run_meta_quality.py`, `tools/local_regression/common.py`, `apps/test_suite.py` |
| B23 | P1 | Meta | checklist / 測試註冊 / 正式入口一致性 | DONE | 已補 synthetic 主入口遺漏註冊案例，並新增 imported / defined `validate_*` case、formal pipeline registry / formal-entry / run_all / preflight / test_suite 一致性 formal guard，以及 `core/` / `tools/` 不得反向 import `apps/` 的分層 guard；正式步驟單一真理來源已收斂到 `tools/local_regression/formal_pipeline.py` | `tools/validate/synthetic_meta_cases.py`, `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_cases.py`, `tools/local_regression/formal_pipeline.py` |
| B24 | P1 | Meta | known-bad fault injection：關鍵規則故意破壞後測試必須 fail | DONE | 已新增 meta fault-injection case，直接對 same-day sell、same-bar stop priority、fee/tax、history filter misuse 注入 known-bad 行為，並驗證既有測試會產生 FAIL | `tools/validate/synthetic_meta_cases.py` |
| B25 | P1 | Meta | independent oracle / golden cases：高風險數值規則不可只與 production 共用同邏輯 | DONE | 已新增獨立 oracle golden case，對 net sell、position size、history EV、annual return / sim years 以手算或獨立公式對照 production | `tools/validate/synthetic_unit_cases.py` |
| B26 | P1 | Meta | checklist 是否已足夠覆蓋完整性（包含 test suite 本身） | DONE | 已補主表 / `T` / `G` 收斂紀錄完整同步 formal guard，並阻擋 convergence 紀錄失同步、`T` 以 `+`、`/`、`,` 或多個 code reference 混寫多個測試入口、`G` 備註欄混寫多個測試入口、`G` transition 缺少合法狀態轉移格式，以及 legacy `D` / `F1` 區不得回流；checklist 自身完整性已納入正式 gate | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/meta_contracts.py`, `doc/TEST_SUITE_CHECKLIST.md` |
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
| B55 | P1 | 契約 | consistency / 單檔 parity 驗證路徑必須維持複利資金，單股、單檔投組與 portfolio_sim 不得再強制切回 fixed-capital | DONE | 已補 direct synthetic contract，直接釘死 consistency parity params 必須保留 compounding，且 single backtest / 單檔 portfolio timeline / portfolio_sim prepared 在虧損後與獲利後都必須一致反映複利資金，candidate sizing 與實際 entry budget 兩條路徑不得分叉 | `tools/validate/synthetic_contract_cases.py`, `tools/validate/real_case_runners.py`, `tools/validate/scanner_expectations.py`, `core/capital_policy.py`, `core/portfolio_entries.py` |
| B56 | P1 | 契約 | scanner 參考投入 / 掛單股數必須使用 `scanner_live_capital`，不得再回退到 `initial_capital` 或回測複利資金 | DONE | 已補 direct synthetic contract，直接釘死 scanner projected qty / proj_cost / 顯示訊息都必須跟 `scanner_live_capital` 一致，且 `initial_capital` 與 scanner 資金來源不得混用 | `tools/validate/synthetic_contract_cases.py`, `core/capital_policy.py`, `core/price_utils.py`, `tools/scanner/stock_processor.py` |
| B57 | P1 | 契約 | 系統評分分子必須可在總報酬率與年化報酬率之間切換，且 score 公式不得把分子 / 分母切換混成不同 mode | DONE | 已補 direct synthetic contract，直接釘死 `SCORE_NUMERATOR_METHOD` 可在 `ANNUAL_RETURN` / `TOTAL_RETURN` 間切換，兩者都必須共用同一個 `|MDD| + 0.0001` 分母，且 `LOG_R2` 品質單調性不得被新分子選項破壞 | `tools/validate/synthetic_strategy_cases.py`, `config/training_policy.py`, `core/portfolio_stats.py` |

### B3. 可隨策略升級調整的測試

| ID | 優先級 | 類別 | 項目 | 目前判定 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|---|---|
| B47 | P1 | 模型介面 | model feature schema / prediction schema 穩定 | DONE | 已新增 model I/O schema case，直接驗輸入欄位、輸出欄位、型別與缺值處理 | `tools/validate/synthetic_strategy_cases.py`, `tools/optimizer/`, `tools/scanner/` |
| B48 | P1 | 重現性 | 同 seed 下 optimizer / model inference 可重現 | DONE | 已新增 strategy repeatability case，直接雙跑 scanner inference 與 optimizer objective，驗證輸出 payload、trial params 與 profile_row 在固定 seed / 固定輸入下可重現 | `tools/validate/synthetic_strategy_cases.py`, `tools/local_regression/`, `tools/optimizer/` |
| B49 | P1 | 排序輸出 | ranking / scoring 輸出可排序、可比較、無 NaN | DONE | 已新增 ranking / scoring sanity case，直接驗排序值可用、方向一致與型別正確 | `tools/validate/synthetic_strategy_cases.py`, `core/buy_sort.py`, `tools/scanner/` |
| B50 | P2 | 最低可用性 | 模型升級後 scanner / optimizer / reporting 仍可跑通 | DONE | 已新增 strategy minimum viability case，直接驗 scanner、optimizer、strategy dashboard、scanner summary 與 yearly return report 在策略輸入下可正常執行 | `tools/validate/synthetic_strategy_cases.py`, `apps/ml_optimizer.py`, `apps/vip_scanner.py` |
| B51 | P2 | 報表相容 | 新策略輸出仍符合既有 artifact / reporting schema | DONE | 已新增 strategy reporting schema compatibility case，直接驗 best_params export payload keys、scanner normalized payload keys 與 yearly return report columns 維持既有 schema | `tools/validate/synthetic_strategy_cases.py`, `tools/portfolio_sim/`, `tools/scanner/reporting.py` |
| B52 | P1 | Optimizer 契約 | objective 淘汰值 / fail_reason / profile_row / best_params export 穩定 | DONE | 已新增 optimizer objective / export contract case，直接驗 `INVALID_TRIAL_VALUE`、fail_reason、profile_row、`tp_percent` 還原優先序、export 成敗，以及訓練中斷且未達指定 trial 數時不得自動覆寫 `best_params.json`；僅完成指定訓練次數或輸入 0 走 export-only 模式時才可更新 | `tools/validate/synthetic_strategy_cases.py`, `tools/optimizer/main.py`, `tools/optimizer/objective_runner.py`, `tools/optimizer/runtime.py`, `tools/optimizer/study_utils.py`, `strategies/breakout/search_space.py`, `strategies/breakout/adapter.py`, `config/training_policy.py`, `config/execution_policy.py`, `config/selection_policy.py` |
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

## G. 逐項收斂紀錄

使用方式：每次只挑少數高優先項目處理，完成後更新本節，不要重開一份新清單。編輯本節時，先依日期定位到對應區塊，再在同日區塊內依項目 ID 升冪插入或重排，禁止把新列直接追加到該日期區塊尾端；備註欄若需要引用檔案或測試名稱，只能保留一個代表 entry。

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
| 2026-04-04 | B54 | 補上單股固定資金 contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_history_cases.py` |
| 2026-04-04 | B55 | 專案資金規則改為全系統複利後，原 execution-only fixed-cap 規格退役，主表先改回 PARTIAL | DONE -> PARTIAL | 待改為單檔複利 parity contract |
| 2026-04-04 | B55 | 改為單檔複利 parity contract 並收斂完成 | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B55 | 補 execution-only fixed-capital parity contract 後，主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B55 | 檢出 execution-only fixed-capital contract 未覆蓋獲利後 entry budget，主表改回 PARTIAL | DONE -> PARTIAL | portfolio 實際下單仍可能以 available_cash 恢復複利 |
| 2026-04-04 | B55 | 補齊 gain-side entry budget contract 與 portfolio fixed-cap 路徑後收斂為 DONE | PARTIAL -> DONE | `core/portfolio_entries.py` |
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
| 2026-04-04 | T130 | 新增單股固定資金 synthetic case 並驗證 | NEW -> DONE | `validate_synthetic_single_backtest_uses_fixed_initial_capital_case` |
| 2026-04-04 | T131 | 專案資金規則改為全系統複利後，原 execution-only fixed-cap parity contract 退役，先改回 PARTIAL | DONE -> PARTIAL | 待改為單檔複利 parity contract |
| 2026-04-04 | T131 | 改為單檔複利 parity contract 並驗證 | PARTIAL -> DONE | `validate_single_ticker_compounding_parity_contract_case` |
| 2026-04-04 | T131 | 新增 execution-only fixed-capital parity contract 並驗證 | NEW -> DONE | `validate_execution_only_fixed_capital_contract_case` |
| 2026-04-04 | T131 | 檢出 execution-only fixed-capital parity contract 僅覆蓋虧損側，改回 PARTIAL | DONE -> PARTIAL | 尚未釘死獲利後不得再放大倉位 |
| 2026-04-04 | T131 | 擴充獲利後不得再放大倉位的 execution-only contract 並收斂完成 | PARTIAL -> DONE | `validate_execution_only_fixed_capital_contract_case` |
| 2026-04-04 | T132 | 新增 scanner live capital contract 並驗證 | NEW -> DONE | `validate_scanner_live_capital_contract_case` |
| 2026-04-04 | T133 | 新增 optimizer interrupt export contract 並驗證 | NEW -> DONE | `validate_optimizer_interrupt_export_contract_case` |
| 2026-04-04 | T134 | 新增 score numerator option contract 並驗證 | NEW -> DONE | `validate_score_numerator_option_case` |

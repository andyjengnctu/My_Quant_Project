# Test Suite 收斂清單

用途：正式 test suite 維護清單；主表為唯一真理來源。

文件分工：`PROJECT_SETTINGS.md` 管原則；本檔管主表、狀態、測試入口與收斂索引。

範圍：納入長期規則與必要 formal contract；不納入暫時特例：`apps/portfolio_sim.py` 自動開瀏覽器、只使用還原價不考慮 raw。

狀態：`DONE` 已覆蓋；`PARTIAL` 仍有缺口；`TODO` 待補；`N/A` 不納入正式長期 test suite。

優先級：`P0` 交易正確性/統計口徑/未來函數；`P1` 高價值補強；`P2` 品質與工具鏈。

索引：`Bxx` 主表 ID；`Txx` 建議測試 ID。

維護規則：
1. 同步順序固定為主表 → `T` / `G` → `E`。
2. `T` 只留最小索引；每列一個 `Txx` 與一個測試入口，依 ID 升冪排序。
3. `G` 只記錄實際狀態變更；依日期升冪、同日再依 tracking ID 排序；`NEW -> *` 只能出現在首筆，且不得出現 no-op transition。
4. `G` 備註欄只留單一代表 entry；日期只記於 `G`。
5. 其餘文字只保留最小必要；會隨實作變動的細節由主表、formal contract 與收斂紀錄承接。

## A. 分層原則

- A1. 長期固定測試：驗 invariant、契約與品質基線。
- A2. 可調整測試：驗 schema、可重現性與最低可用性，不綁死策略結果。

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
| B54 | P1 | 單股回測與投組回測必須一致使用複利資金口徑，不得保留 hidden fixed-cap sizing 分支 | DONE | 已新增 direct synthetic case，直接釘死即使經過連續獲利，單股回測後續 sizing、`asset_growth` 與 `score` 都必須反映複利資金；不得再以固定 `initial_capital` 壓回單股 sizing | `tools/validate/synthetic_history_cases.py`, `core/capital_policy.py`, `core/backtest_core.py`, `core/backtest_finalize.py`, `tools/trade_analysis/backtest.py` |

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
| B21 | P2 | 顯示 | 報表欄位、排序、百分比格式與來源一致 | DONE | 已補 scanner header / start banner / summary、strategy dashboard、validate console summary、issue Excel report schema、portfolio yearly/export report，以及 `apps/test_suite.py` 在 PASS / FAIL / manifest-blocked / partial-selected-steps / preflight-failed / dataset-prepare-failed / summary-unreadable 下的人類可讀摘要契約；另補 score header 顯示契約，釘死 `評分模型` 與 `評分分子` 必須分欄顯示、不得再以 `/ 分子` 混入同一括號；並補 checklist status vocabulary sync 與 meta quality coverage line/branch/min-threshold/missing-zero-target guard 摘要顯示，且以 `run_all.py` contract 釘死 preflight 早退時 dataset_prepare 仍需標記為 `NOT_RUN`，避免 real path 誤落成 `missing_summary_file`；另補 portfolio export 的 Plotly optional dependency fallback 契約，要求主要 Excel artifact 仍須成功匯出、HTML 可略過但必須輸出可追蹤錯誤摘要，且 `except (...)` tuple 不得引用未匯入模組 | `tools/validate/synthetic_display_cases.py`, `tools/validate/synthetic_reporting_cases.py`, `tools/validate/synthetic_contract_cases.py`, `core/display.py`, `tools/scanner/reporting.py`, `tools/portfolio_sim/reporting.py`, `apps/test_suite.py`, `tools/local_regression/run_all.py` |
| B22 | P2 | 覆蓋率 | line / branch coverage 報表 | DONE | 已將 `run_meta_quality.py` 的 synthetic coverage suite、formal helper probe、key target presence/hit 與 manifest 化 line / branch minimum threshold gate 收斂為正式路徑，並同步回寫 `meta_quality_summary.json` / `apps/test_suite.py` 摘要顯示 | `tools/local_regression/run_meta_quality.py`, `tools/local_regression/common.py`, `apps/test_suite.py` |
| B23 | P1 | Meta | checklist / 測試註冊 / 正式入口一致性 | DONE | 已補 synthetic 主入口遺漏註冊案例，並新增 imported / defined `validate_*` case、formal pipeline registry / formal-entry / run_all / preflight / test_suite 一致性 formal guard；`T` 摘要維持單一測試入口欄位，但 formal step 類 script/CLI 必須以 `formal_pipeline.py` 註冊的完整 command string 原樣記錄，仍視為單一入口，不得裁成裸 script path；另補 `core/` / `tools/` 不得反向 import `apps/` 的分層 guard，正式步驟單一真理來源已收斂到 `tools/local_regression/formal_pipeline.py` | `tools/validate/synthetic_meta_cases.py`, `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_cases.py`, `tools/local_regression/formal_pipeline.py` |
| B24 | P1 | Meta | known-bad fault injection：關鍵規則故意破壞後測試必須 fail | DONE | 已新增 meta fault-injection case，直接對 same-day sell、same-bar stop priority、fee/tax、history filter misuse 注入 known-bad 行為，並驗證既有測試會產生 FAIL | `tools/validate/synthetic_meta_cases.py` |
| B25 | P1 | Meta | independent oracle / golden cases：高風險數值規則不可只與 production 共用同邏輯 | DONE | 已新增獨立 oracle golden case，對 net sell、position size、history EV、annual return / sim years 以手算或獨立公式對照 production | `tools/validate/synthetic_unit_cases.py` |
| B26 | P1 | Meta | checklist 是否已足夠覆蓋完整性（包含 test suite 本身） | DONE | 已補主表 / `T` / `G` 收斂紀錄完整同步 formal guard，並阻擋 convergence 紀錄失同步、`T` 以 `+`、`/`、`,` 或多個 code reference 混寫多個測試入口、formal command string 在 `T` 必須仍被視為單一測試入口、`G` 備註欄混寫多個測試入口、`G` transition 缺少合法狀態轉移格式、同一 tracking ID 在已有歷史列後重複寫 `NEW -> *`、`G` 備註欄殘留已退役 validator 名稱、檔案開頭第一個非空行漂移，以及 legacy `D` / `F1` 區不得回流；checklist 自身完整性已納入正式 gate | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/meta_contracts.py`, `doc/TEST_SUITE_CHECKLIST.md` |
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
| B66 | P1 | GUI | `apps/workbench.py` 必須作為單一 GUI 啟用入口，且 workbench panel registry / 單股回測後端 / Excel artifact 與 inline chart payload 契約必須穩定 | DONE | 已補 GUI workbench contract，直接釘死 `apps/workbench.py` 必須對應到 `tools.workbench_ui.main`、workbench 必須註冊單股回測檢視 panel，且 GUI 單股後端必須產生 `Debug_TradeLog_<ticker>.xlsx` 並回傳 inline `chart_payload`，不得再把 HTML artifact 當成 GUI 必要輸出 | `apps/workbench.py`, `tools/workbench_ui/workbench.py`, `tools/validate/synthetic_contract_cases.py` |
| B67 | P1 | GUI | 單股 debug chart hooks 在 `export_chart=False` 時必須完全 no-op；`chart_context=None` 不得造成 runtime side effect 或中斷單股 / chain / synthetic suite | DONE | 已補 direct contract，直接釘死 `run_debug_backtest()` 走無圖模式時仍必須正常產生交易明細；chart marker / active level hooks 必須接受 `chart_context=None` | `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B68 | P1 | 契約 | validation / tool-check payload 的 `module_path` 必須統一為 repo-relative、forward-slash 穩定路徑；不得回傳機器相依絕對路徑 | DONE | 已補 direct contract 與 shared normalizer，直接釘死 absolute / backslash path 都必須正規化為穩定 repo-relative path，避免 bundle 與 synthetic 在不同 OS/工作目錄出現假失敗 | `tools/validate/module_loader.py`, `tools/validate/synthetic_contract_cases.py` |
| B69 | P1 | 契約 | shared path normalizer 必須接受 `pathlib.Path` 等 path-like 輸入；不得假設 `PROJECT_ROOT` 或 path payload 一定是 str 才能正規化 | DONE | 已補 direct contract，直接釘死 `normalize_project_relative_path()` 對 `Path` 輸入也必須穩定回傳 repo-relative、forward-slash 路徑，避免 synthetic / local regression helper 再因字串 API 假設發生 runtime regression | `tools/validate/module_loader.py`, `tools/validate/synthetic_contract_cases.py` |
| B70 | P1 | 契約 | shared module-loader path helpers 必須接受被 patch 成字串的 `PROJECT_ROOT`；不得假設 module-level root 永遠是 `Path`，否則 synthetic / error-path 測試環境會產生 helper 自身回歸 | DONE | 已補 direct contract，直接釘死 `build_project_absolute_path()` 與 `normalize_project_relative_path()` 在 `PROJECT_ROOT` 被 patch 成字串時仍必須正常組路徑並回傳穩定 repo-relative path，避免 shared helper 與既有 `patch.object(module_loader, "PROJECT_ROOT", str(...))` 測試環境互撞 | `tools/validate/module_loader.py`, `tools/validate/synthetic_contract_cases.py` |
| B71 | P1 | GUI | GUI 單股回測檢視必須內嵌大型 K 線圖，且初始視窗與縮放後 Y 軸比例都必須依可視 X 範圍自動重算；不得因全歷史資料或離屏極值導致圖形失真 | DONE | 已補 direct contract，直接釘死 workbench panel 必須宣告 inline chart backend、debug backend 必須回傳 chart payload，且 chart helper 的預設視窗與可視區間價量範圍計算必須忽略離屏極值並可建立 2 軸內嵌 figure | `tools/trade_analysis/charting.py`, `tools/workbench_ui/single_stock_inspector.py`, `tools/validate/synthetic_contract_cases.py` |
| B72 | P2 | Meta | synthetic validator 若使用 `np.` 等外部 alias，必須顯式宣告對應 import，不得依賴 transitive import 或未定義名稱，避免 coverage / consistency synthetic suite 因 `NameError` 假失敗 | DONE | 已補 static contract，直接掃描 `tools/validate/synthetic*_cases.py` 中使用 `np.` 的模組，釘死必須顯式 `import numpy as np`；並修正 `synthetic_contract_cases.py` 缺失 import 所造成的 coverage synthetic suite runtime regression | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B73 | P2 | Meta | 掃描 synthetic validator alias 使用情形時，必須以 AST 實際語意判定，不得用原始字串搜尋 `"np."` 誤判字串常值、註解或 validator 自述文字為真實 alias 使用，避免 meta validator 自己製造假失敗 | DONE | 已補 direct contract，直接釘死 numpy alias 使用掃描必須忽略僅出現在字串常值中的 `np.`，並仍可正確抓到實際 `np.array(...)` AST 使用；同步把 alias-import validator 改為 AST 判定，避免 `synthetic_meta_cases.py` 因自述字串誤被列為缺 import 模組 | `tools/validate/synthetic_meta_cases.py` |
| B74 | P1 | GUI | GUI 單股回測工作台必須以 K 線圖分頁作為主檢視；成交量預設隱藏，切換後須以前景疊加方式共用同一圖面且高度低於 1/4；僅保留交易明細與 Console 分頁，不得再保留執行摘要分頁；GUI 開啟時預設最大化，且內嵌 K 線必須保留完整歷史可平移/縮放，不得因 GUI render slicing 截斷左右歷史資料 | DONE | 已補 direct contract，直接釘死 GUI panel 必須使用 notebook 分頁承接 K 線圖/交易明細/Console、不得再暴露執行摘要分頁、成交量 toggle 預設關閉、workbench 預設 maximized，且 chart helper 必須宣告 full-history navigation 與 volume overlay ratio ≤ 1/4；overlay axis 存在性改以 shared chart contract 驗證，不得綁死 `figure.axes` 長度，避免 matplotlib backend / inset axes 註冊差異造成假失敗 | `tools/workbench_ui/single_stock_inspector.py`, `tools/workbench_ui/workbench.py`, `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B75 | P1 | GUI | GUI 單股回測內嵌 K 線圖必須支援 toolbar-free 滑鼠互動：滾輪直接縮放、左鍵直接拖曳平移時間軸，且重新渲染成交量時不得殘留已銷毀 toolbar/widget 造成 Tk runtime error | DONE | 已補 direct contract，直接釘死 GUI panel 不得再依賴 `NavigationToolbar2Tk`、必須綁定 shared mouse navigation binder，並要求 chart helper 宣告 wheel-zoom / left-drag-pan / no-toolbar contract，避免縮放平移體驗分叉與 volume toggle 時殘留失效 widget | `tools/workbench_ui/single_stock_inspector.py`, `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B76 | P2 | Meta | synthetic validator 直接引用外部 chart/navigation helper 名稱時，必須在同檔顯式 import；不得依賴遺漏名稱在 formal suite 執行時才以 `NameError` 暴露，避免 coverage / consistency synthetic suite 假失敗 | DONE | 已補 static contract，直接釘死 `synthetic_contract_cases.py` 只要實際使用 `bind_matplotlib_chart_navigation`，就必須顯式自 `tools.trade_analysis.charting` import 該名稱；並修正缺失 import 造成的 synthetic suite runtime regression | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B77 | P1 | 契約 | debug 單股回測 entry marker 在 entry plan 不可掛單或為 `None` 時，必須 no-op；不得在 GUI / chart export 路徑中因直接存取 `entry_plan["limit_price"]` 而對特定股票炸出 `'NoneType' object is not subscriptable` | DONE | 已補 direct contract，直接以 `entry_plan=None` 呼叫 debug entry marker helper，釘死必須不新增 marker 且不得拋例外；同步修正 `_record_entry_plan_marker()` 的空 plan guard，避免高價股或 sizing 為 0 的股票在 GUI 執行回測時失敗 | `tools/trade_analysis/entry_flow.py`, `tools/validate/synthetic_contract_cases.py` |
| B78 | P2 | 效能 | 內嵌 GUI K 線相關 synthetic contract 若只需驗證 chart payload / figure 契約，不得強制走 HTML export；正式 debug analysis 應支援在 `export_chart=False` 時直接回傳 chart payload，避免 consistency synthetic suite 因額外載入 plotly / HTML artifact 路徑造成記憶體回歸 | DONE | 已補 direct contract，直接釘死 `run_debug_analysis(..., export_chart=False, return_chart_payload=True)` 必須可回傳 `chart_payload` 且 `chart_path` 保持 `None`；同步讓 reporting / backtest / trade_log wrapper 支援此 lightweight path，避免 GUI chart synthetic validator 為了拿 payload 而額外走 HTML export | `tools/trade_analysis/reporting.py`, `tools/trade_analysis/backtest.py`, `tools/trade_analysis/trade_log.py`, `tools/validate/synthetic_contract_cases.py` |
| B79 | P2 | Meta | `tools/validate/synthetic_cases.py` 的 `_entry(validate_...)` registry symbol 必須能在檔內解析到對應 import 或本地定義；不得等 consistency import 階段才以 `NameError` 爆炸，否則 quick gate 無法在 formal suite 前置攔截 synthetic 主入口回歸 | DONE | 已補 quick gate static contract，直接以 AST 掃描 `synthetic_cases.py` 的 registry entries、imported validate symbols 與本地定義，釘死 `_entry(...)` 內的 `validate_*` symbol 不得有 unresolved/duplicate；後續凡新增 `synthetic_contract_cases.py` validator 註冊到 registry，同輪必須同步補齊 `synthetic_cases.py` import list，避免 `validate_gui_scanner_console_and_latest_contract_case` / `validate_gui_sidebar_latest_preview_contract_case` 這類 GUI validator 再次因漏 import 造成 synthetic suite runtime regression | `tools/local_regression/run_quick_gate.py`, `tools/validate/synthetic_cases.py` |
| B80 | P1 | GUI | GUI 單股回測 K 線圖必須預設最近 18 個月視窗、採台股紅漲綠跌與深色背景，並在同一 shared chart contract 下提供即時 hover 值、買訊/賣訊半透明註記、停損/停利/限價/成交線、右側全歷史摘要、右下狀態框，且信號標記時間點必須符合盤前掛單 / 次日成交交易原則 | DONE | 已補 direct contract，直接釘死 GUI inline path 預設不得預先輸出 HTML、chart contract 必須宣告最近 18 個月預設視窗 / hover 值顯示 / 台股色系 / 摘要框 / 狀態框 / signal annotation，並以小型 synthetic frame 驗證買訊與賣訊標記發生在訊號日、實際買進與賣出 marker 發生在次一根 K 棒，避免圖面提示時間點與正式交易規則分叉 | `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/workbench_ui/single_stock_inspector.py`, `tools/validate/synthetic_contract_cases.py` |
| B81 | P1 | 契約 | chart helper 必須接受缺少 `limit_line` / `entry_line` / `tp_line` / `stop_line` / `summary_box` / `status_box` 等 optional overlay keys 的最小 chart payload，並在建立 GUI figure / hover / autoscale 前自動補齊預設值；不得因 synthetic / legacy payload 缺少可選欄位而在 consistency runtime 以 `KeyError` 假失敗 | DONE | 已補 direct contract，直接以缺少 optional overlay keys 的最小 payload 呼叫 shared chart normalizer 與 matplotlib figure builder，釘死 helper 必須自動補齊 overlay 線陣列與摘要欄位預設值，避免 synthetic validator 或 legacy caller 因 payload schema 省略可選欄位而在 chart 路徑炸出 `KeyError` | `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B82 | P1 | GUI | GUI 工作台必須整體套用 deep-dark 佈景，並以 shared chart contract 支援左右鍵逐根移動時間軸、拖曳平移時避免價格軸上下晃動、成交量 overlay 保持可互動效能、左下狀態 chip、以及 zoom 後 K 棒與量柱寬度等比例調整 | DONE | 已補 direct contract，直接釘死 workbench spec 必須宣告 `ui_theme=deep_dark`、chart helper 必須提供 keyboard pan / left-bottom status chips / dynamic candle width contract，並以 stub canvas 綁定 navigation binder 驗證左右鍵移動與 toolbar-free 互動旗標 | `tools/workbench_ui/workbench.py`, `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B83 | P2 | 契約 | synthetic GUI chart validator 使用的 matplotlib canvas stub 必須滿足 figure cleanup 所需的最小 mouse-grab 介面；至少需提供 `grab_mouse()` / `release_mouse()`，不得在 formal suite runtime 因 stub 缺口而於 `figure.clear()` / artist cleanup 階段以 `AttributeError` 假失敗 | DONE | 已補 direct contract，直接釘死 shared `_build_matplotlib_canvas_stub()` 必須宣告 `grab_mouse()` / `release_mouse()` 並在 release 後清空 grabbed axis；同步讓 GUI keyboard-pan validator 共用該 stub，避免 synthetic contract 自己的 canvas 測試替身比 matplotlib figure cleanup 契約更弱而造成 consistency 假失敗 | `tools/validate/synthetic_contract_cases.py` |
| B84 | P2 | 契約 | synthetic GUI chart validator 使用的 matplotlib canvas stub 必須同時滿足 figure cleanup 對 `canvas.toolbar` 的最低相容契約；至少需宣告 `toolbar` attribute，且預設為 `None`，不得在 formal suite runtime 因 stub 缺少 toolbar 屬性而於 `figure.clear()` / artist cleanup 階段以 `AttributeError` 假失敗 | DONE | 已補 direct contract，直接釘死 shared `_build_matplotlib_canvas_stub()` 必須宣告 `toolbar=None`，避免 matplotlib cleanup 讀取 `self.canvas.toolbar` 時因 synthetic 替身比實際 canvas 介面更弱而造成 consistency 假失敗 | `tools/validate/synthetic_contract_cases.py` |
| B85 | P1 | 契約 | shared chart signal annotation helper 必須接受可選 `meta` keyword，並將其穩定寫入 chart context / chart payload；不得在 GUI 或 synthetic path 因 sell-signal annotation 帶入績效 metadata 而以 `TypeError` runtime 失敗 | DONE | 已補 direct contract，直接釘死 `record_signal_annotation()` 必須接受 `meta` keyword，且 `profit_pct` 等 metadata 必須在 chart context 與 normalized chart payload 中保留，避免 GUI chart / synthetic coverage path 對賣訊績效色彩與訊號框資訊的 shared helper 契約再分叉 | `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B86 | P1 | GUI | GUI 單股 K 線圖必須消除重複買訊資訊框、將買訊/歷績門檻狀態晶片固定於右下角、滑鼠拖曳平移採 pixel-anchor 避免左右平移時上下跳動，並以 explicit dark widget styles 真正套用 deep-dark 佈景；同時背景網格需維持低對比、賣出虧損資訊框必須半透明 | DONE | 已補 direct contract，直接釘死 panel 必須顯式套用 `Workbench.*` dark styles、mouse pan 必須使用 pixel-anchor、status chip layout 必須是 `right_bottom`、買進 trade label 不得再與買訊框重複渲染，並用 synthetic figure 驗證僅保留賣出績效框與淡化 grid alpha；dark-style contract 只驗顯式 `Workbench.*` 行為，不得再綁死已移除容器類型如 `Workbench.TLabelframe` | `tools/workbench_ui/workbench.py`, `tools/workbench_ui/single_stock_inspector.py`, `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B87 | P1 | GUI | GUI workbench deep-dark theme 供 palette/style 使用的 theme token 必須完整宣告；`configure_workbench_theme()` 不得引用未定義 accent 常數而在啟動入口直接 NameError | DONE | 已補 direct contract，直接釘死 `WORKBENCH_ACCENT` 必須存在且為 hex 色碼，避免 `apps/workbench.py` 於 theme 初始化階段即失敗 | `tools/workbench_ui/workbench.py`, `tools/validate/synthetic_contract_cases.py` |
| B88 | P1 | GUI | GUI 單股工作台必須固定使用完整資料集、同時提供『計算候選股』與『計算歷史績效股』兩個獨立掃描鍵、兩個下拉選單 / Console 分頁 / 一鍵回到最新 K 線，且 scanner 每檔輸出與歷史績效股下拉選單都必須顯示資產成長與當前排序探針，latest-bar 買訊預覽、右側狀態晶片與滑鼠互動不得與單股圖表契約分叉 | DONE | 已補 direct contract，直接釘死 panel 不得再顯示資料集選項，必須提供候選股與歷史績效股兩個 scanner button、candidate/history combobox、Console tab、latest button；scanner runner 必須同時維持 `candidate_rows` 與 `history_qualified_rows` payload 契約，scanner row text 與 history dropdown 必須共用 sort probe 顯示資產成長，chart helper 必須提供 `scroll_chart_to_latest()` 與右側留白契約 | `tools/workbench_ui/single_stock_inspector.py`, `tools/scanner/scan_runner.py`, `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B89 | P1 | GUI | GUI 單股工作台必須以右側獨立 sidebar 呈現買入訊號 / 符合歷史績效 / 歷史績效表 / 選取日線值 / 回到最新 K 線，並支援 Enter 直接回測、候選股選取即回測、最新 K 線後右側留白約 1/6 版面，以及 latest-bar 買訊的隔日預掛線預覽不得畫在訊號當日 | DONE | 已補 direct contract，直接釘死 panel 不得保留執行回測按鈕、ticker entry 必須綁定 Enter、candidate select 必須直接觸發回測；chart helper 必須宣告 future_preview 與動態 right padding，且最新 K 線後的右側留白需壓到約 1/6 版面，backtest latest signal preview 必須走 next-day future preview path | `tools/workbench_ui/single_stock_inspector.py`, `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/validate/synthetic_contract_cases.py` |
| B94 | P1 | GUI | GUI 單股回測檢視必須使用純黑 K 線底色、固定文案的右側狀態晶片、右側 OHLCV / 線值單一資訊來源、移除 chart hint footer 以擴大主圖高度，且買訊預掛線必須自訊號次日開始預覽；賣出/停損資訊框需包含最大回撤 | DONE | 已補 refined visual contract，直接釘死 pure-black chart background、固定「出現買入訊號」/「符合歷史績效」晶片文案、右側 sidebar 必須承接 OHLCV 與線值、footer 不得再保留 chart hint 空間、賣出框需列出最大回撤，且 entry preview lines 必須在次日預先畫出 | `tools/workbench_ui/single_stock_inspector.py`, `tools/workbench_ui/workbench.py`, `tools/trade_analysis/charting.py`, `tools/trade_analysis/entry_flow.py`, `tools/validate/synthetic_contract_cases.py` |
| B95 | P1 | GUI | GUI 單股回測狀態晶片在 runtime 必須保持固定文案、僅以底色切換狀態；延續候選預掛線必須依候選資格跨多日連續顯示，但未成交前只得預覽限價，不得預先畫出 stop / tp | DONE | 已補 direct contract，直接釘死右側「出現買入訊號」/「符合歷史績效」晶片不得被 runtime 文案覆寫，且 entry preview 必須走 candidate layer；即使 entry plan 因不可掛單而為 `None`，延續候選有效期間仍須連續顯示 limit 預掛線，未成交前不得預畫 stop / tp；延續候選 preview 的 counterfactual builder 呼叫也必須同步傳遞 `ticker`、`security_profile` 與 `trade_date`，避免 contract 仍比對舊 signature | `tools/workbench_ui/single_stock_inspector.py`, `tools/trade_analysis/entry_flow.py`, `tools/validate/synthetic_contract_cases.py` |
| B96 | P1 | GUI | GUI 單股回測圖例必須貼近 K 線圖分頁上緣但仍保留小幅左上 inset gap、價格軸數字不得因邊界裁切而缺字，且主圖左/下邊界需盡量收小以保留更多 K 線可視區；延續候選若一路持續到最新一根，最新實際 K 棒與 next-day future preview 都必須維持限價預掛線一致顯示，未成交前 stop / tp 必須保持缺席 | DONE | 已補 direct contract，直接釘死 matplotlib 左/下 margin 收小、legend 需保留小幅 top-left inset gap 且不得裁切價格軸；並補 latest extended candidate preview contract，釘死延續候選走到最新一根時，最後一根實際 K 棒與 next-day future preview 都必須保留 limit 預掛線，未成交前 stop / tp 保持缺席 | `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/validate/synthetic_contract_cases.py` |
| B97 | P1 | GUI | GUI 單股 latest raw-signal 買訊預覽路徑必須可直接執行；`tools/trade_analysis/backtest.py` 若引用 `build_normal_entry_plan` / `build_normal_candidate_plan`，必須同步自 `core.entry_plans` 明確 import，避免 GUI/coverage runtime 因未定義 helper 而在單股圖表路徑 NameError | DONE | 已補 direct contract，直接以 AST 掃描 `tools/trade_analysis/backtest.py`，釘死 latest raw-signal 預覽使用的 `build_normal_entry_plan` 與 tail preview 使用的 `build_normal_candidate_plan` 都必須被引用且已自 `core.entry_plans` 匯入，避免再次只改呼叫點卻漏同步 import | `tools/trade_analysis/backtest.py`, `tools/validate/synthetic_contract_cases.py` |
| B98 | P0 | 交易規格 | 延續候選不得再以 signal day frozen `L/S/T` 模型存續；未成交前不得預先具有可執行 stop / tp，僅當首個待掛單日曾進入可買區時，才可用固定反事實 `P' = min(Open, L)` 建立不隨日漂移的失效 / 達標 barrier；`signal_valid` 與 `today_orderable` 必須分層，固定 `L` 若低於今日跌停價不得進 `orderable_candidates_today` | DONE | 已補 direct synthetic case，直接釘死延續候選不得再以 signal day frozen `L/S/T` 或每日漂移 `reference_price` 模型存續；只有首個待掛單日曾進入可買區時，才可建立固定反事實 `P'` barrier，且 valid signal 即使當日價格帶不可達也只能留在 candidate layer，不得擠進 orderable list | `tools/validate/synthetic_flow_cases.py` |
| B99 | P1 | Meta | 非 error-path 的 synthetic validator 不得寫入違反 `strategy_params` 驗證契約的常值；`initial_capital` 等 strict-gt 參數若需測錯，只能留在 explicit error-path case，避免 coverage synthetic suite 因測試自身非法參數而假失敗 | DONE | 已補 meta contract，直接 AST 掃描非 `synthetic_error_cases.py` 的 synthetic validators，不得再寫入 `initial_capital<=0` 這類非法常值；並同步修正 GUI continuity contract 改以極小正資金維持 qty=0 測意 | `tools/validate/synthetic_meta_cases.py` |
| B100 | P1 | Meta | `tools/validate/synthetic_meta_cases.py` 若引用 shared path helper `build_project_absolute_path`，必須顯式自 `.module_loader` import，避免 synthetic coverage suite 因 helper 未定義而在 runtime 才 NameError | DONE | 已補 meta contract，直接 AST 掃描 `synthetic_meta_cases.py`：只要使用 `build_project_absolute_path`，就必須有顯式 from-import；避免 coverage synthetic suite 再因 shared path helper 漏 import 而假失敗 | `tools/validate/synthetic_meta_cases.py` |
| B101 | P1 | GUI | GUI 單股回測賣訊註記必須僅表達 signal day 資訊，不得冒充已成交賣出；future preview 的限價線若超出當前 K 棒高低範圍仍必須納入可視價格範圍；期末強制結算圖示需以黃色區分 | DONE | 已補 direct contract，直接釘死賣訊框只能顯示訊號日收盤與未成交提示，不得再混入股數 / 金額 / 損益；並以 chart range case 驗證 future preview 的限價線會參與 autoscale，另釘死期末強制結算 marker color 為黃色 | `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/validate/synthetic_contract_cases.py` |
| B102 | P1 | GUI | GUI 單股回測賣訊註記必須只顯示「賣訊」，指標賣出 marker 必須為綠色橫線，停損/賣出/結算資訊框需顯示交易次數，且買訊框中的限價價格即使未畫出 future preview 線也必須納入可視價格範圍；停利線需以黃色顯示 | DONE | 已補 direct contract，直接釘死賣訊註記不得再附任何 detail 文案、指標賣出 marker 改為綠色水平線、停損/賣出資訊框加入交易次數，並以 signal annotation meta 驗證買訊框內的限價價格會參與 autoscale；另釘死停利線顏色為黃色 | `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/trade_analysis/exit_flow.py`, `tools/validate/synthetic_contract_cases.py` |
| B103 | P1 | GUI | GUI 買入資訊框應貼近買進 K 線位置，且停損/賣出/結算資訊框中的交易次數必須置於最後一行，並對齊最新完成 round-trip 交易次數，不得與右側歷史績效表不一致 | DONE | 已補 direct contract，直接釘死買入資訊框應留在 K 線 overlay 並貼近對應 K 線下方、不得搬到右側 sidebar；若同一 K 線或相鄰少數 K 線已有其他下方資訊框，必須以水平錯位與額外垂直分層避免重疊。另停損/賣出/結算資訊框的交易次數必須是最後一行，最終出場事件需以 completed round-trip 口徑顯示最新交易次數，避免與右側歷史績效表分叉 | `tools/workbench_ui/single_stock_inspector.py`, `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/trade_analysis/exit_flow.py`, `tools/validate/synthetic_contract_cases.py` |
| B104 | P0 | 交易規格 | 細部交易契約必須由 checklist / formal contract 明確承接：`L` 只作進場上限 / 最壞風險 sizing 上界，首個可執行 stop / tp 只能於 `t+1` 成交後以 `P_fill + ATR_t` 在 `t+1` 收盤後 frozen，`t+1` 觸發與 `t+2` 執行必須分離，且長倉 hit 採 `Low <= line` / `High >= line` | DONE | 已補 meta contract，直接釘死 `PROJECT_SETTINGS.md` 只保留最小必要限制與細節下沉原則；上述 `L` 角色、`P_fill + ATR_t` 首個可執行 stop / tp、固定反事實 barrier 與 inclusive hit 語意則由 checklist 與 formal contract 承接，避免再次回退到舊 `L / init_sl / T` frozen 口徑 | `doc/TEST_SUITE_CHECKLIST.md`, `tools/validate/synthetic_meta_cases.py` |
| B105 | P0 | 交易規格 | runtime 必須分離盤前 sizing 風險與成交後首個可執行 stop / tp：candidate plan 仍可用 `L` 推導最壞情境 sizing，但 filled position 的 `initial_stop` / `tp_half` / entry-day pending stop-tp trigger 必須改以實際成交價建構，且 `t+1` hit stop/tp 後 `t+2` 必須以開盤第一個可執行價強制執行，不得要求再次碰價 | DONE | 已補 direct synthetic case，直接釘死 candidate plan 保留 `L` 基準 sizing、filled position 改用 `P_fill + ATR_t` 建構首個可執行 stop / tp，且 entry day 觸發 stop/tp 必須排程到次日開盤執行；停利若已於 `t+1` 觸發，`t+2` 就算未再碰價也必須依 queued action 減倉或出清 | `tools/validate/synthetic_flow_cases.py`, `core/entry_plans.py`, `core/position_step.py`, `core/portfolio_exits.py` |
| B106 | P1 | GUI / Debug 契約 | 只要 debug analysis 要求 `return_chart_payload=True` 或 `export_chart=True`，即使輸入 `price_df` 為空，也必須回傳可正規化的 placeholder chart payload，不得把空 payload 延後到 GUI figure / normalize 路徑才以 runtime 方式爆炸 | DONE | 已補 direct contract，直接釘死空 `price_df` 路徑仍須回傳單根 placeholder payload，避免 synthetic GUI / chart coverage 因空 payload 在 figure / normalize 階段 runtime 失敗而掩蓋真正規格檢查 | `tools/trade_analysis/reporting.py`, `tools/validate/synthetic_contract_cases.py` |
| B107 | P1 | Meta / GUI 契約 | synthetic contract 若直接以 dict literal 呼叫 `normalize_chart_payload_contract(...)`，payload 必須明確提供 `x` 軸欄位；不得再以缺少 `x` 的假 payload 讓 coverage synthetic suite 在 contract 自身 runtime 失敗 | DONE | 已補 meta contract，直接 AST 掃描 `synthetic_contract_cases.py` 中所有對 `normalize_chart_payload_contract(...)` 的 dict-literal 呼叫，逐一要求 payload 含 `x` 欄位；並同步修正兩個 GUI visual contract 的合成 payload | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B108 | P1 | 核心韌性 / 空輸入契約 | 單股回測核心在空 `price_df` 下必須直接回傳零交易、零 miss buy/sell、`is_candidate=False` 的穩定 stats，不得在尾端索引 `C[-1]` / `Dates[-1]` / `buyCondition[-1]` 才 runtime 失敗 | DONE | 已補 direct synthetic case，直接釘死空 `price_df` 進入 `run_v16_backtest()` 時必須 early-return 穩定 stats 與空 logs；避免 GUI / debug 空資料 placeholder 路徑再次在 backtest core 尾端索引炸出 `IndexError` | `tools/validate/synthetic_flow_cases.py`, `core/backtest_core.py` |
| B109 | P1 | Meta / Registry 契約 | `tools/validate/synthetic_cases.py` 對各 `synthetic_*` 模組的 from-import 必須指向實際存在該 validator symbol 的模組；不得把 validator 匯錯模組，導致 coverage synthetic suite 在 import 時直接 `ImportError` | DONE | 已補 meta contract，直接 AST 掃描 `synthetic_cases.py` 對各 `synthetic_*` 模組的 from-import，逐一驗證被匯入 symbol 確實存在於目標模組；避免再發生 validator 實作位於 `synthetic_flow_cases.py` 卻誤從 `synthetic_portfolio_cases.py` 匯入的 registry import 回歸 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_cases.py` |
| B110 | P1 | 錯誤處理 / Meta 契約 | 專案內廣義例外處理（`except Exception` / `except BaseException`）必須綁定例外物件並可追蹤；若非直接 re-raise，handler 內必須使用綁定的例外，禁止 silent swallow | DONE | 已補 meta contract，直接 AST 掃描 `apps/`、`config/`、`core/`、`strategies/`、`tools/` 的 broad exception handler；逐一要求 handler 必須綁定例外名稱，且若非 re-raise 就必須在 body 內實際使用該例外，避免再次出現 GUI / chart runtime 失敗被靜默吞掉而無法定位 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/charting.py`, `tools/workbench_ui/single_stock_inspector.py` |
| B111 | P1 | 錯誤處理 / Meta 契約 | optional dependency fallback（如 `ImportError` / `ModuleNotFoundError`）不得 silent swallow；若選擇降級或略過，必須綁定例外並保留可追蹤 detail | DONE | 已補 meta contract，直接 AST 掃描 GUI / debug / validate / downloader 的 optional dependency fallback handler；逐一要求 handler 必須綁定例外名稱，且若非 re-raise 就必須在 body 內實際使用該例外，避免 TkAgg / coverage / curl_cffi 缺失時只默默降級而無法定位 | `tools/validate/synthetic_meta_cases.py`, `tools/workbench_ui/single_stock_inspector.py`, `tools/validate/main.py`, `tools/downloader/runtime.py` |
| B112 | P1 | 錯誤處理 / GUI 契約 | GUI TclError fallback 不得 silent swallow；若因 theme / palette / maximize 相容性而降級，必須綁定例外並保留可追蹤 detail | DONE | 已補 meta contract，直接 AST 掃描 `tools/workbench_ui/*.py` 的 `TclError` fallback handler；逐一要求 handler 必須綁定例外名稱，且若非 re-raise 就必須在 body 內實際使用該例外，避免 GUI 啟動時 theme / zoom fallback 靜默吞錯而無法定位 | `tools/validate/synthetic_meta_cases.py`, `tools/workbench_ui/workbench.py` |
| B113 | P1 | 錯誤處理 / Meta 契約 | 非 broad / optional-import / GUI TclError 的 specific exception 若採 pass-only silent fallback，必須被 formal suite 直接禁止；僅允許明確列為 control-flow 的例外（目前 `FileNotFoundError` cleanup）保留 pass-only | DONE | 已補 meta contract，直接 AST 掃描 `apps/`、`config/`、`core/`、`strategies/`、`tools/` 的 specific pass-only exception handler，排除 synthetic 測試檔與允許的 `FileNotFoundError` cleanup；逐一禁止 `ValueError` / `OSError` 等非 control-flow 例外以 `pass` 靜默吞掉，避免 shared runtime / chart helper 降級路徑失去可追蹤性 | `tools/validate/synthetic_meta_cases.py`, `core/runtime_utils.py`, `tools/trade_analysis/charting.py` |
| B114 | P1 | GUI / Debug 顯示契約 | 單股 GUI / debug 的買訊、買入、停利、賣出 / 停損 / 結算資訊框，必須顯示同一口徑的預留 / 實支、腿損益 / 整筆總損益與 completed-trade 統計；debug view 的起始資金基準需與 scanner `scanner_live_capital` 對齊，避免買訊預留與 GUI round-trip 顯示再次分叉 | DONE | 已補 output contract，直接驗證 debug view params 需以 `scanner_live_capital` 正規化起始資金、買訊框需顯示資金 / 預留、買入框需顯示停利 / 限價 / 成交 / 停損，且買入資訊框必須像買訊資訊框一樣貼在對應 K 線下方，不得缺席也不得搬到右側 sidebar；若買訊框與買入框落在同一 K 線或相鄰少數 K 線，必須以水平錯位與額外垂直分層避免重疊。單股 inspector 右側 sidebar 右下區只顯示交易資訊（停利 / 限價 / 成交 / 停損），不得改放買訊資訊，停利框需顯示金額 / 損益、賣出 / 停損 / 結算框需顯示資金 / 損益 / 總損益 / 報酬率且交易次數必須使用 completed snapshot；避免 GUI 再把尾倉單腿損益誤當整筆結果或把 pre-exit / post-exit 統計混寫 | `tools/validate/synthetic_contract_cases.py`, `tools/trade_analysis/trade_log.py`, `tools/trade_analysis/exit_flow.py` |
| B115 | P1 | Meta / GUI 契約 | synthetic GUI / debug contract 若需 patch `tools.trade_analysis.backtest` 的 PIT history snapshot seam，`tools/trade_analysis/backtest.py` 必須穩定暴露 `_build_pit_history_snapshot`，且 `run_debug_analysis()` 必須經由該 seam 取用；不得只保留直接 import helper，否則 formal synthetic suite 會在 patch 階段提早 `AttributeError`，並連帶造成 coverage target 缺漏假失敗 | DONE | 已補 meta contract，直接 AST 掃描 `tools/trade_analysis/backtest.py`，釘死模組必須暴露 `_build_pit_history_snapshot` 且 `run_debug_analysis()` 的 signal-day / latest-day snapshot 都必須經由該 seam；並同步在 backtest 模組恢復 stable alias，避免 helper 重構時再次把 synthetic coverage suite 在 patch 階段炸掉 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/backtest.py`, `tools/validate/synthetic_contract_cases.py` |
| B116 | P1 | GUI / Visual 契約 | 單股 GUI / debug 的買訊藍色 annotation 箭頭，其尖點必須錨定訊號 K 棒低點；實際買進三角箭頭則必須錨定成交價，兩者不得混用成同一價位 | DONE | 已補 output contract，直接驗證 `_record_buy_signal_annotation(...)` 即使存在 entry plan 仍必須以 `signal_low` 作為 `anchor_price`，並驗證買進 trade marker 必須使用 `buy_price`；避免 GUI 把買訊位置與實際成交位置畫成同一點而失真 | `tools/validate/synthetic_contract_cases.py`, `tools/trade_analysis/backtest.py` |
| B117 | P1 | Meta / Fixture Schema 契約 | `tools/validate/synthetic_contract_cases.py` 對 multi-ticker synthetic case 不得再直接讀取過時的 `case["price_df"]`；必須改由 `case["frames"][case["primary_ticker"]]` 取得主 frame，避免 synthetic coverage suite 因 fixture schema 漂移在 contract 自身 runtime 提早 `KeyError` | DONE | 已補 meta contract，直接 AST 掃描 `synthetic_contract_cases.py`，禁止 `case["price_df"]` 舊存取；並同步修正 GUI visual contract 改從 `frames[primary_ticker]` 取主 frame，避免 coverage synthetic suite 再因 validator 自己的 fixture schema 假設過時而提早中斷 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B118 | P1 | Meta / Helper Import 契約 | `tools/validate/synthetic_contract_cases.py` 若直接呼叫 `tools.trade_analysis.backtest._record_buy_signal_annotation`，必須顯式 import 該 helper；不得引用未定義名稱，否則 synthetic coverage suite 會先以 `NameError` 中斷，並連帶造成 coverage target 缺漏假失敗 | DONE | 已補 meta contract，直接 AST 掃描 `synthetic_contract_cases.py`，釘死檔案若使用 `_record_buy_signal_annotation` 符號就必須顯式自 `tools.trade_analysis.backtest` import；並同步補上 validator 實際 import，避免 GUI visual contract 再因未定義名稱提早炸掉 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B119 | P1 | Meta / GUI 契約 | `validate_gui_trade_count_and_sidebar_sync_contract_case` 必須以強制結算行為 probe 驗證 completed round-trip 交易次數，不得再把過時的 `history_snapshot=latest_history_snapshot` / `_resolve_completed_trade_count(history_snapshot, include_current_round_trip=True)` 字串當成唯一正確實作，否則 synthetic suite 會把已正確的 GUI 交易次數語意誤判成 FAIL | DONE | 已補 meta contract，直接掃描 `synthetic_contract_cases.py` 的 GUI trade-count validator，禁止舊 exit snippet literal，並強制 validator 必須使用 `append_debug_forced_closeout(...)` 與 `build_trade_stats_index(...)` 做行為驗證；避免 contract 再因綁死過時實作細節而製造假失敗 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B120 | P0 | 交易規格 / 一致性 | 投組 cash-capped entry 路徑不得丟失 candidate plan 的 `entry_atr` / `target_price` 等成交後 first-actionable stop/tp 所需欄位；否則單股與投組會在新 `P_fill + ATR_t` 規格下分叉 | DONE | 已補 direct synthetic case，直接釘死 `build_daily_candidates()` 產生的 orderable candidate 必須保留 `target_price` / `entry_atr`，且 `execute_reserved_entries_for_day()` 經 cash-capped entry 後建立的 filled position 仍須以實際成交價建構 `initial_stop` / `trailing_stop` / `tp_half` 與 entry-day queued action；避免投組在資金重算路徑悄悄退回舊 `L / init_sl / T` 語意 | `tools/validate/synthetic_flow_cases.py`, `core/portfolio_candidates.py`, `core/portfolio_entries.py` |
| B121 | P0 | 交易規格 / 一致性 | 只要 `t+1` 已觸及 `L`、`qty > 0` 且資金條件成立，就不得僅因 `P_fill <= candidate_plan.init_sl` 而拒絕成交；`candidate_plan.init_sl` 只能作盤前 worst-case sizing，不得作成交否決下限 | DONE | 已補 direct synthetic case，直接釘死當實際成交價低於 candidate plan 的 limit-based sizing stop 時，`execute_pre_market_entry_plan()` 仍必須成交，且 filled position 之 `initial_stop` / `trailing_stop` / `tp_half` 必須改用實際成交價與 `ATR_t` 建立；避免 runtime 殘留舊 `先達停損放棄進場` 語意 | `tools/validate/synthetic_flow_cases.py`, `core/entry_plans.py` |
| B122 | P1 | Meta / Quick Gate 契約 | `quick_gate` 必須在 CLI / formal 匯入前，以靜態方式驗證 `tools/validate/synthetic_cases.py` 的 from-import target 是否指向實際宣告該 validator symbol 的模組；不得等 runtime import `synthetic_cases` 時才以 `ImportError` 失敗，並連帶遮蔽原本應回報的 CLI / dataset guard | DONE | 已補 quick gate static contract，直接共用 synthetic import-target resolution helper，於 `run_static_checks()` 新增 `synthetic_registry_import_targets`；即使 `synthetic_cases.py` 尚未可 import，仍能在 quick gate 前置攔截匯錯模組的 validator import，避免同一個 wiring 錯誤同時拖垮 quick_gate / consistency / meta_quality | `tools/local_regression/run_quick_gate.py`, `tools/validate/meta_contracts.py` |
| B123 | P0 | 契約 / 一致性 | 正式帳務中的 `cash` / `pnl` / `equity` / `reserved_cost` / `risk`、partial-exit cost-basis allocation、商品別驅動的賣出交易稅與 tick/limit hit 判斷，必須收斂到整數 exact-accounting 單一真理來源；directional tick rounding、tick band lookup、商品 profile 自動辨識、漲跌停 raw-limit 對齊與賣出稅率，也必須先依 raw price、ticker/metadata 與交易日期決定方向、區間、商品別與稅率，再轉為合法 tick 價與 net sell total；不得再以含費每股價或 per-share × qty 回推正式總額，也不得先做 0.001 量化後再決定 up/down、band，或把 ETF / ETN / REIT / 債券 ETF 類誤套股票 tick ladder / 單一 stock-only 交易稅 | DONE | 已補 exact-accounting unit-like contract，直接釘死 ledger 守恆、partial-exit cost-basis 回沖、integer tick/limit hit、cash/risk boundary、單股/投組 closeout parity、display-derived 欄位，以及 raw-price directional tick rounding / band lookup 邊界、跨級距漲跌停 raw-limit tick-band 對齊、ETF / ETN / REIT 類商品 profile 自動辨識後的兩級 tick 規則，與依商品別 / 交易日期決定的賣出交易稅（股票 0.3%、ETF / ETN 0.1%、REIT 免稅、債券 ETF 免徵期限內 0 稅率）；並同步把正式核心 cash/pnl/equity/reserved_cost 與商品別驅動的 raw-price tick 正規化 / limit-price 對齊 / sell-tax 路徑收斂到 `core/exact_accounting.py` | `tools/validate/synthetic_unit_cases.py`, `core/exact_accounting.py` |
| B124 | P1 | Meta / Schema 契約 | `build_backtest_stats()` / `run_v16_backtest()` 的 public stats payload 必須維持既有 snake_case 相容 key（如 `trade_count` / `expected_value` / `asset_growth` / `max_drawdown` / `missed_buys` / `is_setup_today` / `extended_candidate_today` / `current_position` / `score`）；若 stats producer 進一步把 `final_date` / `trade_date` 等日期上下文往下傳給 extended preview builder，或顯示 `buyPrice` / `sellPrice` / `tpPrice` 等可見 preview 價格，也必須同步更新函式簽名、全部 caller 與 static contract，並重用共享 stop / target helper；不得只在函式內引用未宣告日期名稱，或以手寫浮點公式推導 preview 價格，導致 GUI / scanner / consistency runtime `NameError` 或可見價格與正式交易規則分叉 | DONE | 已補 static meta contract，直接 AST 掃描 `core/backtest_finalize.py` 的 `build_backtest_stats()` dict key，並釘死 `final_date` 簽名、empty/final caller 傳遞、extended candidate 必須走 threaded `trade_date=final_date`，且 `sellPrice` / `tpPrice` preview 必須分別走共享 initial-stop / target helper、不得保留手寫 `tpPrice` 浮點公式；避免 refactor 後只改 producer 內部實作就讓 GUI / scanner / consistency runtime 因缺 key、`NameError` 或 preview 價格口徑分叉而中斷 | `tools/validate/synthetic_meta_cases.py`, `core/backtest_finalize.py`, `core/backtest_core.py` |
| B125 | P0 | 契約 / 一致性 | 單股 backtest / debug 的 `currentCapital` 必須固定代表可用現金；任何半倉、全倉與期末結算都只能加回 `net sell total`，`currentEquity` 必須以 `cash + 當前可變現淨值` 計；不得再沿用 `+ realized pnl` 或 `cash + floating pnl` 的舊路徑，否則會讓單股與投組 trade_count / asset_growth / completed-trade PnL 分叉 | DONE | 已補 static meta contract，直接掃描 `core/backtest_core.py`、`core/backtest_finalize.py`、`tools/trade_analysis/backtest.py` 與 `tools/trade_analysis/exit_flow.py`，釘死單股 / debug 現金更新必須使用 freed cash / net sell total，mark-to-market equity 必須使用 net liquidation value；避免 exact-accounting 遷移後把 `currentCapital` 誤當 accumulated pnl | `tools/validate/synthetic_meta_cases.py`, `core/backtest_core.py` |
| B126 | P1 | Meta / 契約 | debug backtest / GUI 重播現金路徑時，買進當下也必須扣除實際 `net buy total`，且凡 entry flow 新增 `ticker` / `security_profile` / `trade_date` 等商品 profile / 日期感知 keyword 參數時，必須同步更新 callee 函式簽名與 caller 傳遞；不得只在賣出時更新 `current_capital`，也不得只改 caller 造成 `unexpected keyword argument` 或商品 profile / 交易日期傳遞中斷，否則會讓 debug trade log sizing、completed-trade PnL sequence 與核心單股回測分叉 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/backtest.py` 與 `tools/trade_analysis/entry_flow.py`，釘死 debug entry flow 必須回傳 `spent_cash` 並由主流程扣減 `current_capital`，且當 caller 傳入 `ticker` / `security_profile` / `trade_date` 時，callee 簽名與 normal / extended entry builder 呼叫也必須同步接收並傳遞；避免重播路徑只補賣出現金、漏扣買進現金或發生 `unexpected keyword argument` | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/backtest.py`, `tools/trade_analysis/entry_flow.py` |
| B127 | P1 | 契約 / 顯示一致性 | 使用者可見的 debug / portfolio trade log 若以 rounded 單筆損益重建 completed trades，則半倉停利 + 尾倉賣出後的逐列可見值加總必須精準回到 core completed-trade 總損益；不得只保證內部 exact pnl 一致，放任可見明細因 rounding residual 漂移 0.01 | DONE | 已補 exact-accounting display reconciliation unit case，並把 debug / portfolio full-exit 顯示損益改為吸收前序 rounded leg residual；避免 completed-trade sequence 與可見 trade log 因 0.01 rounding 差異而分叉 | `tools/validate/synthetic_unit_cases.py`, `core/portfolio_exits.py`, `tools/trade_analysis/exit_flow.py` |
| B128 | P1 | Meta / 顯示一致性 契約 | 凡屬會被下游以可見金額欄位重建 completed trades 或彙總損益的 debug / portfolio trade log，必須統一使用共享 money-rounding helper；不得在各模組散落內建 `round(..., 2)` 造成 0.01 級可見口徑分叉 | DONE | 已補 static meta contract，直接掃描 `core/exact_accounting.py`、`core/portfolio_exits.py` 與 `tools/trade_analysis/log_rows.py`，釘死 shared display rounding 必須使用 Decimal HALF_UP，且 history/log row helper 必須委派共享 rounding helper；避免之後又在單一模組偷回內建 `round(..., 2)` 造成 completed-trade 顯示重建漂移 | `tools/validate/synthetic_meta_cases.py`, `core/exact_accounting.py`, `tools/trade_analysis/log_rows.py` |
| B129 | P1 | Meta / 驗證 Oracle 契約 | 凡 validator / consumer 以可見 completed-trade 或 standalone log 的 rounded 金額作 expected oracle 時，必須與共享 money-rounding helper 使用同一 HALF_UP 口徑；不得混用內建 `round(..., 2)` 將顯示 rounding 誤判成核心差異 | DONE | 已補 static meta contract，直接掃描 `tools/validate/real_case_assertions.py`，釘死 real-case completed-trade expected oracle 必須委派共享 rounding helper，且不得再用內建 `round(float(...), 2)` / `round(sum(...), 2)`；避免 consistency 因 validation consumer 與顯示層 rounding 規則分叉而虛假 FAIL | `tools/validate/synthetic_meta_cases.py`, `tools/validate/real_case_assertions.py` |
| B130 | P1 | Meta / 重建契約 | 凡以可見 trade log / history row 重建 completed trades 的 helper / consumer，必須以共享 money-rounding helper 正規化逐列金額並累加；不得在重建路徑另用內建 `round(..., 2)`，否則會把同一批可見 row 重建成不同於核心 completed trades 的序列與總損益 | DONE | 已補 static meta contract，直接掃描 `tools/validate/trade_rebuild.py`，釘死 trade rebuild helper 必須匯入共享 rounding helper、逐列 `pnl` 先正規化後再累加 completed-trade total，且不得殘留內建 `round(..., 2)`；避免 debug / portfolio completed-trade sequence 因重建 helper 使用不同 rounding 規則而再度漂移 0.01 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/trade_rebuild.py` |
| B131 | P1 | Meta / 顯示一致性契約 | debug / GUI / history 的期末強制結算若需合成整筆 completed-trade `total_pnl`，必須以 `realized_pnl_milli + final_leg_pnl_milli` 走整數 ledger 路徑後再轉顯示；不得以 float 將既有已實現損益與尾倉損益直接相加，否則會在最後一筆 forced closeout 殘留 0.01 級漂移 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/exit_flow.py` 的 `append_debug_forced_closeout()`，釘死整筆 `total_pnl` 必須先組 `total_pnl_milli` 再 `milli_to_money(...)`，且不得殘留舊 `float(realized_pnl + final_leg_actual_pnl)` 路徑；避免 debug completed-trade sequence / realized-pnl-sum 只在期末結算尾筆分叉 0.01 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/exit_flow.py` |
| B132 | P1 | Meta / suite 可執行性契約 | `tools/validate/synthetic_meta_cases.py` 內任何 validator 若回填 `summary["source_path"]` 等來源路徑欄位，引用的 path 變數必須在該函式內明確宣告；不得誤引用其他 validator 的局部變數名稱，避免 synthetic suite runtime 因 `NameError` 中斷 | DONE | 已補 static meta contract，直接掃描 `tools/validate/synthetic_meta_cases.py` 內各 `validate_*` 函式的 `summary["source_path"]` 指派，釘死 `.relative_to(PROJECT_ROOT)` 所用 path 名稱必須在該函式內有宣告；避免 copy-paste 後把 `source_path` 誤寫成其他 validator 的局部 path 變數，導致 synthetic suite 於 coverage / consistency 階段 runtime 中斷 | `tools/validate/synthetic_meta_cases.py` |
| B133 | P1 | Meta / 驗證 Oracle 契約 | unit / synthetic validator 若比對可見 rounded 金額、display leg pnl 或 completed-trade 顯示總損益，必須與正式顯示層共用同一 money-rounding helper；不得在 validator 內另用內建 `round(..., 2)` 產生不同 rounding oracle | DONE | 已補 static meta contract，直接掃描 `tools/validate/synthetic_unit_cases.py` 的 `validate_exact_accounting_display_leg_reconciliation_case`，釘死 display-leg reconciliation validator 必須委派 `round_money_for_display(...)`，且不得殘留內建 `round(..., 2)`；避免 unit validator 自己用不同 rounding 規則把顯示口徑誤判成核心差異 | `tools/validate/synthetic_meta_cases.py` |
| B134 | P1 | Meta / 顯示總額契約 | debug / GUI / history 的可見交易明細或 marker 若顯示 `buy_capital`、`sell_capital`、`gross_amount`、`total_return_pct` 等正式交易總額或報酬率，必須由 exact ledger total（如 `net_buy_total_milli`、`net_total_milli`）推導；不得再以含費每股顯示價或 `per-share × qty` 回推可見總額，避免把可見值重新拉回浮點攤提路徑 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/exit_flow.py`，釘死 debug exit flow 的 full-entry capital、half/full exit `gross_amount` 與 marker `sell_capital` 都必須優先使用 ledger total，且 display sell-total helper 必須由簽名顯式接受 `position` / `current_date` 上下文、不得殘留 `sell_net_price * qty` 舊路徑或自由變數 fallback；避免 debug trade row / marker 顯示金額與正式帳務總額再度分叉 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/exit_flow.py` |
| B135 | P1 | Meta / 顯示報酬率分母契約 | debug / GUI / history 若已有 `entry_capital_total`、`net_buy_total_milli` 等既存 full-entry capital total，計算 `total_return_pct` 或其他以 full-entry capital 為分母的可見報酬率時，必須優先使用該總額欄位；若必須 fallback，也必須走共享 exact-total / average-price total helper；不得跳過既有 total 或退回 `entry * qty` 等 raw per-share 浮點公式，避免可見報酬率在 partial-exit / rounded entry 顯示下再次偏離 exact ledger | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/exit_flow.py` 的 `_resolve_full_entry_capital_milli()`，釘死 helper 必須先看 `net_buy_total_milli`，再看 `entry_capital_total`，最後 fallback 也必須走 average-price total helper；不得殘留 `entry * qty` 原始 per-share 回推或把已含費平均價誤餵給 gross-price helper 再次加費 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/exit_flow.py` |
| B136 | P1 | Meta / 買進顯示總額契約 | debug / GUI / history 的買進可見欄位或 marker 若顯示 `buy_capital`、買進 `gross_amount` 或其他 entry total，必須優先使用 `net_buy_total_milli`、`entry_cost` 或共享 exact-total helper；不得以 `entry_price * qty`、`entry * qty` 等 per-share fallback 回推買進總額，避免買進可見資本再次偏離 exact ledger | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/entry_flow.py`，釘死 debug entry flow 的買進 `gross_amount` 與 marker `buy_capital` 都必須優先使用 `net_buy_total_milli` / `entry_cost` / exact-total helper，且不得殘留 `entry_result['entry_price'] * entry_plan['qty']` 舊 fallback；避免 debug 買進可見金額再度分叉 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/entry_flow.py` |
| B137 | P1 | Meta / 半倉單腿報酬率契約 | debug / GUI / history 若顯示半倉停利或其他單腿 exit 的可見 `pnl_pct` / 報酬率，必須優先以 exact ledger 的 `allocated_cost_milli` 與該腿 `pnl_milli` 計算；若必須 fallback，也必須走共享 exact-total / exact-sell-total helper；不得以 `(net_price - entry) / entry` 等 per-share 浮點差價公式回推單腿報酬率，避免單腿可見報酬率與 exact ledger 單腿損益分叉 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/exit_flow.py`，釘死半倉停利 marker 的 `pnl_pct` 必須走 `_resolve_display_leg_return_pct()`，且 helper 必須優先使用 `allocated_cost_milli` / `pnl_milli` 並直接以 integer totals 計算報酬率，fallback 也必須走 exact total helper；不得殘留 per-share 浮點差價公式或 `milli_to_money(...)` ratio 舊路徑 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/exit_flow.py` |
| B138 | P1 | Meta / 賣訊持倉報酬率契約 | debug / GUI / history 若顯示賣訊 annotation 的可見 `profit_pct`、盈虧顏色或其他 signal-day 持倉報酬率，必須優先以 exact ledger 的 full-entry capital、已實現損益與剩餘部位 mark-to-market 淨值計算；若必須 fallback，也必須走共享 average-price total helper；不得以 `(close - entry) / entry` 等 raw close 與 per-share 成本的浮點差價公式回推，避免賣訊顏色與持倉淨報酬率在接近損益兩平時分叉 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/backtest.py`，釘死賣訊 `profit_pct` 必須走 `_resolve_sell_signal_profit_pct()`，且 helper 必須優先使用 `net_buy_total_milli`、`realized_pnl_milli` 與 `remaining_cost_basis_milli` 做 exact mark-to-market，fallback 也必須走 average-price total helper；不得殘留 raw close 差價或 `entry_price * qty` 舊 per-share 路徑 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/backtest.py` |
| B139 | P1 | Meta / debug fallback exact-helper 契約 | debug / GUI / history 的 fallback helper 若在缺少預先儲存 total / allocated-cost 時仍需回推可見資本、單腿報酬率或持倉報酬率，必須優先使用共享 exact-total / exact-ledger helper（如 `calc_total_from_average_price(...)`、`net_total_milli - pnl_milli`）；不得再以 `entry * qty`、`entry_price * qty`、`(net_price - entry) / entry` 等 raw per-share 浮點公式作最後 fallback，避免可見值在邊界與四捨五入情境再次分叉 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/backtest.py` 與 `tools/trade_analysis/exit_flow.py`，釘死 debug fallback helper 必須走共享 exact-total / exact-ledger helper，且不得殘留 `entry * qty`、`entry_price * qty` 或 per-share 浮點差價公式；同時已收斂到 average-price total helper，避免把 `entry` 這類已含費平均價誤餵給 gross-price helper 再次加費 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/backtest.py` |
| B140 | P1 | Meta / average-price total 契約 | 凡 helper 只有 `entry`、`net_stop_price` 或其他由既有總額反推的已正規化每股平均價格時，若需重建 total，不得再以 `money_to_milli(price * qty)` 走浮點乘法，也不得把該平均價當原始買價餵給 `calc_entry_total_cost(...)` 重新加費；必須使用共享 average-price total helper，以避免 risk / fallback total double-count 費用或引入浮點乘法尾差 | DONE | 已補 static meta contract，直接掃描 `core/exact_accounting.py` 與 `core/price_utils.py`，釘死 average-price total helper 必須存在，且 `calc_initial_risk_total()` 必須走該 helper，不得殘留 `money_to_milli(entry_price * qty)` 舊路徑；同時 debug fallback 已改用同一 helper，避免把已含費平均價錯當原始買價再度加費 | `tools/validate/synthetic_meta_cases.py`, `core/exact_accounting.py` |

| B141 | P1 | Meta / 投組 rotation 報酬率契約 | 投組 rotation、汰弱賣出候選比較或其他持倉優劣排序，若需以持倉報酬率做比較，必須優先以 exact ledger 的 full-entry capital、已實現損益與剩餘部位 mark-to-market 淨值計算；不得以 `(close - entry) / entry` 等 raw close 與 per-share 成本的浮點差價公式回推，避免 rotation 決策與正式帳務口徑分叉 | DONE | 已補 static meta contract，直接掃描 `core/portfolio_exits.py`，釘死 rotation 的 `ret` 必須走 exact mark-to-market helper，且不得殘留 `ret = (pt_y_close - pos['entry']) / pos['entry']` 舊公式；避免投組 rotation 將 raw close 與 per-share 平均成本當作正式持倉報酬率 | `tools/validate/synthetic_meta_cases.py`, `core/portfolio_exits.py` |
| B142 | P1 | Meta / validator exact-ledger oracle 契約 | 凡 unit / synthetic validator、oracle、expected 值建構若需比較正式帳務 total、risk budget、freed cash、realized pnl、entry cash after buy 或 scanner projected qty，必須使用共享 exact ledger / integer budget helper，且商品別與日期相關 oracle 必須同步傳遞 `ticker` / `trade_date`；不得以 `price * qty`、`net_price * qty`、`capital * risk_fraction` 等 per-share float 公式或 stock-only 預設自行重建 oracle，避免 validator 自己與正式帳務口徑分叉 | DONE | 已補 static meta contract，直接掃描 `tools/validate/synthetic_unit_cases.py`、`tools/validate/synthetic_take_profit_cases.py`、`tools/validate/scanner_expectations.py` 與 `tools/validate/synthetic_contract_cases.py`，釘死 oracle 必須使用 buy/sell ledger、integer risk budget 與共享 projected-qty helper，且 scanner/reference oracle 必須同步傳遞 `ticker` / `trade_date`；不得殘留 `gross = float(price) * int(qty)`、`risk_budget = capital * risk_fraction`、未帶商品別/日期的 `calc_reference_candidate_qty(...)` 或其他 stock-only 預設 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/scanner_expectations.py` |
| B143 | P1 | Meta / checklist DONE 測試摘要表結構契約 | `T. 目前所有 DONE 的建議測試項目摘要` 必須維持合法 markdown table 結構，包含 header separator row，且資料列 ID 必須是 `Txx`、對應主表項必須是 `Bxx`；不得讓 parser 把表頭列當成資料列，避免 checklist parser / meta-registry 產生假性 FAIL | DONE | 已補 static meta contract，直接檢查 `doc/TEST_SUITE_CHECKLIST.md` 的 `T` 摘要表是否保留 header separator row，並驗證 `_load_done_test_rows()` 解析出的 `id` / `b_id` 皆符合 `Txx` / `Bxx`；避免表格排版小失誤直接污染 meta-registry 載入結果 | `tools/validate/synthetic_meta_cases.py`, `doc/TEST_SUITE_CHECKLIST.md` |
| B144 | P1 | Meta / mutating validator oracle snapshot 契約 | 凡 validator / synthetic case 若呼叫會原地修改 `position` 或其他狀態的 producer，且 expected 值依賴呼叫前成本基礎或持倉狀態，必須先 snapshot 呼叫前狀態；不得在 producer 執行後再讀取已被修改的物件作 oracle，避免 validator 自己把 mutated state 誤當 expected | DONE | 已補 static meta contract，直接掃描 `tools/validate/synthetic_take_profit_cases.py` 的 same-bar stop-priority case，釘死 expected pnl 必須使用 `original_cost_basis_milli` snapshot，而不得在 `execute_bar_step(...)` 後再讀 `position["remaining_cost_basis_milli"]`；避免 in-place mutation 把 oracle 算錯 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_take_profit_cases.py` |
| B145 | P1 | Meta / shared helper import 契約 | 凡核心或 producer 以共享 helper 取代舊邏輯時，必須同輪同步補齊 import；不得只改函式呼叫而漏 import，避免靜態 compile 未必能捕捉、runtime 才因 `NameError` 失敗。尤其 `price_utils` 這類核心 helper 聚合模組，若引用 `calc_total_from_average_price_milli` 等 exact-accounting helper，必須明確從單一來源匯入並由 static meta contract 釘死 | DONE | 已補 static meta contract，直接掃描 `core/price_utils.py`，釘死 `calc_initial_risk_total()` 在使用 `calc_total_from_average_price_milli(...)` 時，該 helper 必須已於 import 區塊明確匯入；避免 helper call 已替換但 import 遺漏，形成 runtime `NameError` | `tools/validate/synthetic_meta_cases.py`, `core/price_utils.py` |
| B146 | P1 | Meta / array 價格正規化契約 | 任何 scalar / vectorized / array 版價格正規化 helper（如批次 tick 對齊、批次買入上限正規化）也必須委派共享 raw-price tick helper，且 shared caller 必須把 ticker / security_profile / trade_date 正確往下傳；不得另寫 threshold/tick ladder、`valid_prices / ticks`、`np.ceil` / `np.floor` 的獨立浮點取整實作，也不得先以 `price_to_milli(...)` 預量化 raw price 後再決定 tick band / direction，或在 candidate-plan resize / scanner projected qty 路徑遺漏商品 profile 與日期上下文，導致 ETF / ETN / REIT / 債券 ETF 仍走 stock-only 稅率與 ladder；prepared scanner frame 的最新交易日也必須由共享 helper 統一解析，並同時支援欄位與 index 日期來源，避免 scanner runtime 與 validator/oracle 因資料形態差異分叉；rotation / exit caller 也不得引用未宣告的 `ticker` 自由變數 | DONE | 已補 static meta contract，直接掃描 `core/price_utils.py`、`core/signal_utils.py`、`core/backtest_core.py`、`core/portfolio_entries.py`、`core/position_step.py`、`core/portfolio_exits.py`、`core/entry_plans.py` 與 `tools/scanner/stock_processor.py`，釘死 scalar / array 版 `get_tick_size*()` 與 `round_to_tick*()` 必須委派 `get_tick_milli_from_price(...)` 與 `round_price_to_tick_milli(...)`，且 shared caller 必須以 ticker / security_profile / trade_date 將商品 profile、日期上下文與 prepared frame 最新交易日傳遞到買入上限、candidate-plan resize、scanner projected qty、漲跌停、rotation 賣出與出場路徑；不得殘留 `get_tick_milli(price_to_milli(price))`、`round_price_milli_to_tick(price_to_milli(price), ...)`、`ratios = valid_prices / ticks`、`np.ceil` / `np.floor` 的舊路徑、stock-only tick / 稅率假設、僅依 `df["Date"]` 取最新日期或 `adjust_long_sell_fill_price(w_open, ticker=ticker)` 這類未宣告變數引用 | `tools/validate/synthetic_meta_cases.py`, `core/price_utils.py`, `core/entry_plans.py`, `tools/scanner/stock_processor.py`, `core/portfolio_exits.py` |
| B147 | P1 | Meta / 正式入口摘要同步契約 | `apps/test_suite.py` 若保留頂部 coverage / contract 摘要註解並列舉 `Txx`，則該列舉必須與實際已掛入 synthetic registry 的同主題 contract ID 同步；不得讓正式入口摘要註解漏列最新 `Txx`，避免維護者誤判 formal 覆蓋範圍 | DONE | 已補 static meta contract，直接切出 `apps/test_suite.py` 頂部 exact-contract 摘要註解 block，釘死該摘要註解必須包含 `T225/T226/T229/T230/T231/T232/T233/T234/T235/T236`，且必須明確提到 `T234`、`T235` 與 `T236`、不得保留漏列新 ID 的舊註解；對應 validator 也必須同步掛入 `tools/validate/synthetic_cases.py` registry，避免 checklist 已列 DONE 但 formal suite 未實際執行 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_cases.py`, `apps/test_suite.py` |
| B148 | P1 | Meta / exact-ledger ratio path 契約 | 凡 debug / GUI / history / rotation 以 exact ledger 或 integer total 計算報酬率、持倉優劣比較或 leg return 時，分子分母必須直接使用 milli / integer total 相除；不得先各自 `milli_to_money(...)` 轉回 float 金額後再相除，避免浮點偏移在可見報酬率與 rotation 排序產生微小分叉 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/exit_flow.py`、`tools/trade_analysis/backtest.py` 與 `core/portfolio_exits.py`，釘死 total/leg/sell-signal/rotation return 必須直接以 integer total 計算，且不得保留 `milli_to_money(...)/milli_to_money(...)` 舊路徑 | `doc/PROJECT_SETTINGS.md`, `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/exit_flow.py`, `tools/trade_analysis/backtest.py`, `core/portfolio_exits.py` |
| B149 | P1 | Meta / debug exit milli binding 契約 | 凡 debug / GUI / history 的 final-exit `total_return_pct` 等可見公式改為引用 `total_pnl_milli`、`full_entry_capital_milli` 等整數 ledger 變數時，必須先在同一路徑明確綁定該 `*_milli` 變數，再由其推導顯示值；不得殘留未定義局部變數或混用舊 float `total_pnl` 路徑，避免 consistency / chain checks 因 runtime `NameError` 全面中斷 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/exit_flow.py` 的 `process_debug_position_step()` 與 `append_debug_forced_closeout()`，釘死 final-exit `total_return_pct` 必須先綁定對應 `total_pnl_milli`，再以 `milli_to_money(total_pnl_milli)` 推導顯示 `total_pnl`；不得保留 `float(position.get('realized_pnl', pnl_realized))` 或 `float(position.get('realized_pnl', 0.0) + final_leg_actual_pnl)` 等舊 float 路徑 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/exit_flow.py` |
| B150 | P1 | Meta / core R-multiple exact-ledger 契約 | 核心回測與投組統計若需累計 `R_Multiple` / `r_mult`、勝負 R 與 EV，必須以 `*_pnl_milli` 與 `initial_risk_total_milli` 直接走整數 ledger ratio helper；不得回退成 `float total_pnl / float initial_risk_total`，避免 closed-trades stats、單股與投組統計在邊界值因浮點路徑再度分叉 | DONE | 已補 static meta contract 與共享 ratio helper，直接掃描 `core/exact_accounting.py`、`core/backtest_core.py`、`core/backtest_finalize.py` 與 `core/portfolio_exits.py`，釘死核心 `trade_r_mult` / `total_r` 必須委派 `calc_ratio_from_milli(...)`，且不得殘留 `total_pnl / position['initial_risk_total']` 或 `total_pnl / pos['initial_risk_total']` 舊浮點公式；避免單股回測、期末結算、rotation 與投組 closeout 的 `R_Multiple` / EV 統計只在核心路徑分叉 | `tools/validate/synthetic_meta_cases.py`, `core/exact_accounting.py`, `core/backtest_core.py`, `core/backtest_finalize.py`, `core/portfolio_exits.py` |

### B3. 可隨策略升級調整的測試

| ID | 優先級 | 類別 | 項目 | 目前判定 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|---|---|
| B47 | P1 | 模型介面 | model feature schema / prediction schema 穩定 | DONE | 已新增 model I/O schema case，直接驗輸入欄位、輸出欄位、型別與缺值處理 | `tools/validate/synthetic_strategy_cases.py`, `tools/optimizer/`, `tools/scanner/` |
| B48 | P1 | 重現性 | 同 seed 下 optimizer / model inference 可重現 | DONE | 已新增 strategy repeatability case，直接雙跑 scanner inference 與 optimizer objective，驗證輸出 payload、trial params 與 profile_row 在固定 seed / 固定輸入下可重現 | `tools/validate/synthetic_strategy_cases.py`, `tools/local_regression/`, `tools/optimizer/` |
| B49 | P1 | 排序輸出 | ranking / scoring 輸出可排序、可比較、無 NaN | DONE | 已新增 ranking / scoring sanity case，直接驗 EV / PROJ_COST / HIST_WIN_X_TRADES / ASSET_GROWTH 排序值可用、方向一致與型別正確 | `tools/validate/synthetic_strategy_cases.py`, `core/buy_sort.py`, `tools/scanner/` |
| B50 | P2 | 最低可用性 | 模型升級後 scanner / optimizer / reporting 仍可跑通 | DONE | 已新增 strategy minimum viability case，直接驗 scanner、optimizer、strategy dashboard、scanner summary 與 yearly return report 在策略輸入下可正常執行 | `tools/validate/synthetic_strategy_cases.py`, `apps/ml_optimizer.py`, `apps/vip_scanner.py` |
| B51 | P2 | 報表相容 | 新策略輸出仍符合既有 artifact / reporting schema | DONE | 已新增 strategy reporting schema compatibility case，直接驗 best_params export payload keys、scanner normalized payload keys 與 yearly return report columns 維持既有 schema | `tools/validate/synthetic_strategy_cases.py`, `tools/portfolio_sim/`, `tools/scanner/reporting.py` |
| B52 | P1 | Optimizer 契約 | objective 淘汰值 / fail_reason / profile_row / best_params export 穩定 | DONE | 已新增 optimizer objective / export contract case，直接驗 `INVALID_TRIAL_VALUE`、fail_reason、profile_row、`tp_percent` 還原優先序、export 成敗，以及訓練中斷且未達指定 trial 數時不得自動覆寫 `best_params.json`；僅完成指定訓練次數或輸入 0 走 export-only 模式時才可更新 | `tools/validate/synthetic_strategy_cases.py`, `tools/optimizer/main.py`, `tools/optimizer/objective_runner.py`, `tools/optimizer/runtime.py`, `tools/optimizer/study_utils.py`, `strategies/breakout/search_space.py`, `strategies/breakout/adapter.py`, `config/training_policy.py`, `config/execution_policy.py` |
| B53 | P1 | I/O | reduced dataset 契約必須依目前目錄快照動態推導，不得綁死固定成員或固定筆數 | DONE | 已將 reduced dataset contract 改為直接根據目前資料夾中的 CSV members / content 動態計算 `csv_count` 與 fingerprint；formal guard 只要求資料夾非空且 members 不重複，避免之後調整 reduced dataset 又必須回頭改程式常數 | `tools/local_regression/common.py`, `tools/validate/synthetic_contract_cases.py`, `data/tw_stock_data_vip_reduced/` |

## E. 未完成缺口摘要

使用方式：僅在存在未完成項時填寫；平時維持空表。

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

使用方式：只保留 `DONE` 項的最小索引；詳情仍以主表與 `G` 為準。

維護規則：`T` 只留「ID / 建議測試名稱 / 對應主表項目」，並依 ID 升冪排序。

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
| T18 | `validate_output_contract_case` | B11 |
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
| T38 | `tools/scanner/scan_runner.py` | B12 |
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
| T52 | `validate_run_all_preflight_early_failure_dataset_contract_case` | B11 |
| T53 | `validate_test_suite_summary_checklist_status_sync_case` | B21 |
| T54 | `validate_test_suite_summary_meta_quality_guardrail_reporting_case` | B21 |
| T55 | `validate_single_formal_test_entry_contract_case` | B26 |
| T56 | `validate_synthetic_setup_index_prev_day_only_case` | B01 |
| T57 | `validate_downloader_universe_fetch_error_path_case` | B15 |
| T58 | `validate_downloader_universe_screening_init_error_path_case` | B15 |
| T59 | `validate_meta_quality_performance_memory_contract_case` | B19 |
| T60 | `validate_test_suite_summary_meta_quality_memory_reporting_case` | B19 |
| T61 | `validate_portfolio_sim_prepared_tool_contract_case` | B11 |
| T62 | `validate_scanner_prepared_tool_contract_case` | B19 |
| T63 | `validate_debug_trade_log_prepared_tool_contract_case` | B19 |
| T64 | `validate_scanner_reference_clean_df_contract_case` | B19 |
| T65 | `validate_meta_quality_reuses_existing_coverage_artifacts_case` | B19 |
| T66 | `tools/validate/synthetic_cases.py` | B23 |
| T67 | `tools/validate/synthetic_meta_cases.py` | B26 |
| T68 | `apps/test_suite.py` | B26 |
| T69 | `validate_no_reverse_app_layer_dependencies_case` | B23 |
| T70 | `validate_run_all_manifest_failure_master_summary_contract_case` | B17 |
| T71 | `validate_synthetic_same_bar_stop_priority_case` | B02 |
| T72 | `validate_synthetic_half_tp_full_year_case` | B04 |
| T73 | `validate_synthetic_extended_miss_buy_case` | B09 |
| T74 | `validate_synthetic_competing_candidates_case` | B09 |
| T75 | `validate_synthetic_same_day_sell_block_case` | B06 |
| T76 | `validate_synthetic_rotation_t_plus_one_case` | B05 |
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
| T158 | `tools/local_regression/run_quick_gate.py` | B79 |
| T159 | `validate_gui_chart_recent_view_signal_overlay_contract_case` | B80 |
| T160 | `validate_chart_payload_optional_overlay_keys_contract_case` | B81 |
| T161 | `validate_gui_dark_theme_and_keyboard_pan_contract_case` | B82 |
| T162 | `validate_gui_navigation_canvas_stub_cleanup_contract_case` | B83 |
| T163 | `validate_gui_navigation_canvas_stub_toolbar_contract_case` | B84 |
| T164 | `validate_record_signal_annotation_meta_contract_case` | B85 |
| T165 | `validate_gui_chart_overlay_layout_and_pan_contract_case` | B86 |
| T166 | `validate_workbench_theme_accent_symbol_contract_case` | B87 |
| T167 | `validate_gui_scanner_console_and_latest_contract_case` | B88 |
| T168 | `validate_gui_sidebar_latest_preview_contract_case` | B89 |
| T173 | `validate_gui_single_stock_refined_visual_contract_case` | B94 |
| T174 | `validate_gui_extended_preview_continuity_contract_case` | B95 |
| T175 | `validate_gui_chart_margin_and_latest_extended_preview_contract_case` | B96 |
| T176 | `validate_gui_latest_raw_signal_preview_helper_contract_case` | B97 |
| T177 | `validate_synthetic_extended_signal_a2_frozen_plan_case` | B98 |
| T178 | `validate_synthetic_case_non_error_initial_capital_contract_case` | B99 |
| T179 | `validate_synthetic_meta_cases_build_project_absolute_path_import_contract_case` | B100 |
| T180 | `validate_gui_signal_annotation_and_forced_close_visual_contract_case` | B101 |
| T181 | `validate_gui_trade_marker_and_tp_visual_contract_case` | B102 |
| T182 | `validate_gui_trade_count_and_sidebar_sync_contract_case` | B103 |
| T183 | `validate_project_settings_init_sl_frozen_plan_principle_case` | B104 |
| T184 | `validate_synthetic_init_sl_single_source_runtime_case` | B105 |
| T185 | `validate_debug_empty_price_df_chart_payload_contract_case` | B106 |
| T186 | `validate_synthetic_case_normalize_chart_payload_literal_x_contract_case` | B107 |
| T187 | `validate_synthetic_empty_backtest_df_contract_case` | B108 |
| T188 | `validate_synthetic_cases_import_target_resolution_contract_case` | B109 |
| T189 | `validate_broad_exception_traceability_contract_case` | B110 |
| T190 | `validate_optional_dependency_fallback_traceability_contract_case` | B111 |
| T191 | `validate_gui_tcl_fallback_traceability_contract_case` | B112 |
| T192 | `validate_specific_pass_only_exception_traceability_contract_case` | B113 |
| T193 | `validate_gui_trade_box_capital_and_round_trip_contract_case` | B114 |
| T194 | `validate_debug_backtest_history_snapshot_patch_seam_contract_case` | B115 |
| T195 | `validate_gui_buy_signal_annotation_anchor_price_contract_case` | B116 |
| T196 | `validate_synthetic_contract_cases_no_legacy_price_df_case_key_contract_case` | B117 |
| T197 | `validate_gui_buy_signal_annotation_helper_import_contract_case` | B118 |
| T198 | `validate_gui_trade_count_contract_no_legacy_exit_snippet_case` | B119 |
| T199 | `validate_synthetic_portfolio_entry_preserves_fill_based_first_actionable_case` | B120 |
| T200 | `validate_synthetic_fill_below_limit_based_sizing_stop_still_enters_case` | B121 |
| T201 | `validate_quick_gate_synthetic_registry_import_targets_contract_case` | B122 |
| T202 | `tools/validate/cli.py --dataset reduced` | B23 |
| T203 | `validate_checklist_f2_formal_command_single_entry_case` | B26 |
| T204 | `validate_exact_accounting_ledger_conservation_case` | B123 |
| T205 | `validate_exact_accounting_cost_basis_allocation_case` | B123 |
| T206 | `validate_exact_accounting_tick_limit_integer_case` | B123 |
| T207 | `validate_exact_accounting_cash_risk_boundary_case` | B123 |
| T208 | `validate_exact_accounting_single_vs_portfolio_parity_case` | B123 |
| T209 | `validate_exact_accounting_display_derived_case` | B123 |
| T210 | `validate_single_backtest_stats_legacy_schema_contract_case` | B124 |
| T211 | `validate_single_backtest_exact_cash_path_contract_case` | B125 |
| T212 | `validate_debug_backtest_entry_cash_path_contract_case` | B126 |
| T213 | `validate_exact_accounting_display_leg_reconciliation_case` | B127 |
| T214 | `validate_display_money_rounding_helper_contract_case` | B128 |
| T215 | `validate_real_case_completed_trade_rounding_oracle_contract_case` | B129 |
| T216 | `validate_trade_rebuild_rounding_helper_contract_case` | B130 |
| T217 | `validate_debug_forced_closeout_exact_total_pnl_contract_case` | B131 |
| T218 | `validate_synthetic_meta_source_path_binding_contract_case` | B132 |
| T219 | `validate_unit_display_rounding_helper_contract_case` | B133 |
| T220 | `validate_debug_exit_display_capital_uses_ledger_totals_contract_case` | B134 |
| T221 | `validate_debug_exit_entry_capital_fallback_contract_case` | B135 |
| T222 | `validate_debug_entry_display_capital_uses_exact_total_contract_case` | B136 |
| T223 | `validate_debug_half_exit_leg_return_pct_uses_allocated_cost_contract_case` | B137 |
| T224 | `validate_debug_sell_signal_profit_pct_uses_exact_mark_to_market_contract_case` | B138 |
| T225 | `validate_debug_exact_fallback_helpers_contract_case` | B139 |
| T226 | `validate_average_price_total_helper_contract_case` | B140 |
| T227 | `validate_portfolio_rotation_mark_to_market_return_contract_case` | B141 |
| T228 | `validate_validator_oracles_use_exact_ledger_totals_contract_case` | B142 |
| T229 | `validate_checklist_done_test_summary_markdown_structure_case` | B143 |
| T230 | `validate_same_bar_stop_priority_oracle_snapshots_pre_exit_cost_basis_contract_case` | B144 |
| T231 | `validate_price_utils_average_price_total_import_contract_case` | B145 |
| T232 | `validate_price_utils_array_tick_normalization_contract_case` | B146 |
| T233 | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` | B147 |
| T234 | `validate_exact_ledger_return_ratio_no_money_float_division_contract_case` | B148 |
| T235 | `validate_debug_exit_total_return_milli_binding_contract_case` | B149 |
| T236 | `validate_core_r_multiple_exact_ledger_contract_case` | B150 |
## G. 逐項收斂紀錄

使用方式：每次只挑少數高優先項目處理，完成後更新本節，不要重開一份新清單。編輯本節時，先依日期定位到對應區塊，再抽出整個同日區塊依排序鍵重排後整段覆寫回原位；禁止把新列直接追加到該日期區塊尾端，也禁止只改局部單列後跳過同日區塊總排序檢查；若新增列排序鍵小於當前尾列，必須回插到正確位置，不得留在尾端。G 只記錄實際狀態變更；不得寫 `DONE -> DONE`、`PARTIAL -> PARTIAL`、`TODO -> TODO` 等 no-op transition。同日同 ID 若有多筆狀態變更，必須依實際演進排序；`NEW -> *` 只能出現在該 ID 首筆，且 `NEW -> PARTIAL` / `NEW -> DONE` 必須排在後續 `PARTIAL -> DONE` 或 `DONE -> PARTIAL` 之前。交付前至少再做一次同日區塊機械核對：由上到下檢查 namespace、數字段、尾碼三層排序鍵皆未逆序，且新增列同時滿足前一列 ≤ 當前列 ≤ 後一列；備註欄若需要引用檔案或測試名稱，只能保留一個代表 entry。

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
| 2026-04-02 | B26 | 檢出完成摘要索引仍有漏同步風險，主表改回 PARTIAL | DONE -> PARTIAL | checklist 自身仍有回寫 / 摘要失同步缺口 |
| 2026-04-02 | B26 | checklist / test suite 自身完整性收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
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
| 2026-04-04 | B55 | 檢出單檔複利 parity contract 初版未覆蓋獲利後 entry budget，主表改回 PARTIAL | DONE -> PARTIAL | portfolio 實際下單仍可能以 available_cash 恢復複利 |
| 2026-04-04 | B55 | 改為單檔複利 parity contract 並收斂完成 | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
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
| 2026-04-04 | T131 | 檢出單檔複利 parity contract 初版僅覆蓋虧損側，改回 PARTIAL | DONE -> PARTIAL | 尚未釘死獲利後不得再放大倉位 |
| 2026-04-04 | T131 | 擴充獲利側 entry budget 的單檔複利 parity contract 並收斂完成 | PARTIAL -> DONE | `validate_single_ticker_compounding_parity_contract_case` |
| 2026-04-04 | T131 | 改為單檔複利 parity contract 並驗證 | PARTIAL -> DONE | `validate_single_ticker_compounding_parity_contract_case` |
| 2026-04-04 | T132 | 新增 scanner live capital contract 並驗證 | NEW -> DONE | `validate_scanner_live_capital_contract_case` |
| 2026-04-04 | T133 | 新增 optimizer interrupt export contract 並驗證 | NEW -> DONE | `validate_optimizer_interrupt_export_contract_case` |
| 2026-04-04 | T134 | 新增 score numerator option contract 並驗證 | NEW -> DONE | `validate_score_numerator_option_case` |
| 2026-04-05 | B21 | 檢出 score header 將評分模型與分子混寫在同一括號，主表先改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_display_cases.py` |
| 2026-04-05 | B21 | 將 score header 改為模型/分子分欄顯示並補契約後收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_display_cases.py` |
| 2026-04-05 | B26 | 檢出 `G` 仍可殘留退役 validator 名稱與重複 `NEW -> *`，主表改回 PARTIAL | DONE -> PARTIAL | checklist 自身完整性仍有歷史回寫缺口 |
| 2026-04-05 | B26 | 檢出 `G` 同日區塊新增列後未整段重排，造成排序 guard 再次被真實 bundle 擊中，主表改回 PARTIAL | DONE -> PARTIAL | checklist 同日區塊回寫仍有排序失誤 |
| 2026-04-05 | B26 | 檢出 checklist 摘要表固定升冪排序仍缺 formal guard，主表改回 PARTIAL | DONE -> PARTIAL | checklist 自身完整性仍有摘要表排序契約缺口 |
| 2026-04-05 | B26 | 檢出 checklist 首行固定標題仍缺 formal guard，主表改回 PARTIAL | DONE -> PARTIAL | checklist 自身完整性仍有檔首契約缺口 |
| 2026-04-05 | B26 | 修正 2026-04-05 同日區塊排序並補強前後鄰列核對要求後重新收斂為 DONE | PARTIAL -> DONE | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-05 | B26 | 補上 `G` 的 `NEW` 首次出現約束與有效 validator reference guard 後收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B26 | 補上 checklist 摘要表固定升冪排序 guard 後重新收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B26 | 補上 checklist 首行固定標題 guard 後重新收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B43 | 依新規格調整 package_zip：root bundle 不得移入 arch，主表先改回 PARTIAL | DONE -> PARTIAL | root `to_chatgpt_bundle_*.zip` 應保留於 root |
| 2026-04-05 | B43 | 改為只歸檔非 bundle 舊 ZIP 並保留 root bundle copy 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_cli_cases.py` |
| 2026-04-05 | B45 | 檢出 `resolve_log_dir("outputs")` 先命中 generic root-dir 錯誤，未維持 outputs-root 專屬拒絕語意，主表改回 PARTIAL | DONE -> PARTIAL | outputs-root guard 判斷順序錯誤 |
| 2026-04-05 | B45 | 檢出 `write_issue_log` / `build_timestamped_log_path` 仍可接受 outputs 根目錄，主表改回 PARTIAL | DONE -> PARTIAL | log path create-path outputs-root guard 仍有缺口 |
| 2026-04-05 | B45 | 補上 outputs-root create-path guard 後重新收斂為 DONE | PARTIAL -> DONE | `core/log_utils.py` |
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
| 2026-04-05 | B74 | 改以 shared chart contract 驗證 overlay axis 存在性並同步壓低 chart payload 記憶體占用後重新收斂為 DONE | PARTIAL -> DONE | `tools/trade_analysis/charting.py` |
| 2026-04-05 | B75 | 新增 GUI toolbar-free mouse navigation contract，釘死滾輪縮放、左鍵拖曳平移與 volume toggle 不得依賴已銷毀 toolbar widget | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B76 | 新增 synthetic validator chart-navigation helper 顯式 import contract，避免遺漏 `bind_matplotlib_chart_navigation` 在 formal suite 才以 `NameError` 造成假失敗 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B77 | 新增 debug entry marker optional-plan contract，釘死 entry_plan=None 時必須 no-op 不得炸出 NoneType 下標錯誤 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B78 | 新增 debug chart payload 無需 HTML export contract，釘死 `export_chart=False` 也可回傳 payload 以避免 synthetic suite 額外載入 plotly 造成記憶體回歸 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B79 | 新增 quick gate synthetic registry symbol-resolution static contract，釘死 `_entry(validate_...)` 不得引用未 import / 未定義 symbol，避免 consistency import 階段才以 `NameError` 爆炸 | NEW -> DONE | `tools/local_regression/run_quick_gate.py` |
| 2026-04-05 | B80 | 新增 GUI 近期視窗 / 台股色系 / hover 值 / signal overlay / timing contract，釘死圖面提示與盤前掛單規則不得分叉 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B83 | 新增 synthetic GUI canvas stub cleanup contract，釘死 stub 必須提供 `grab_mouse()` / `release_mouse()`，避免 figure cleanup 於 formal suite runtime 因替身介面不足假失敗 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B84 | 新增 synthetic GUI canvas stub toolbar contract，釘死 stub 必須宣告 `toolbar=None`，避免 figure cleanup 讀取 `self.canvas.toolbar` 時因替身介面不足假失敗 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B85 | 新增 signal annotation meta contract，釘死 shared helper 必須接受 `meta` keyword 並保留至 chart payload，避免賣訊績效 metadata 在 GUI / synthetic path runtime 分叉 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B86 | 新增 GUI overlay layout / dark widget styles / pixel-anchor mouse pan contract，釘死右下 status chip、買訊不重複框、淡化 grid 與滑鼠拖曳不跳動 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B87 | 新增 GUI workbench theme accent token contract，釘死 dark theme palette/style 不得引用未宣告 accent 常數而在 GUI 啟動入口直接 NameError | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B88 | 新增 GUI scanner / Console / latest-view contract，釘死完整資料集固定化、候選股下拉與 latest-bar 預覽不得分叉 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | B89 | 新增 GUI 右側 sidebar / Enter 回測 / latest next-day preview contract，釘死狀態摘要與預掛線預覽不得再分叉 | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-05 | T27 | 補 scanner / dashboard score header 顯示契約後收斂完成 | DONE -> PARTIAL | display header contract 缺口待補。 |
| 2026-04-05 | T27 | 補 scanner / dashboard score header 顯示契約並驗證 | PARTIAL -> DONE | validate_display_reporting_sanity_case |
| 2026-04-05 | T116 | 依新規格調整 package_zip runtime contract：root bundle 不得移入 arch，建議測試先改回 PARTIAL | DONE -> PARTIAL | root `to_chatgpt_bundle_*.zip` 應保留於 root |
| 2026-04-05 | T116 | 改為只歸檔非 bundle 舊 ZIP 並保留 root bundle copy 後重新收斂 | PARTIAL -> DONE | `validate_package_zip_runtime_contract_case` |
| 2026-04-05 | T121 | 檢出 outputs-root create-path guard 錯誤語意被 generic root-dir 檢查覆蓋，建議測試先改回 PARTIAL | DONE -> PARTIAL | outputs-root 錯誤語意未命中 |
| 2026-04-05 | T121 | 檢出 quick_gate log-path contract 尚未覆蓋 outputs-root create path，建議測試先改回 PARTIAL | DONE -> PARTIAL | 尚未釘死拒絕 outputs 根目錄寫入 |
| 2026-04-05 | T121 | 修正 outputs-root 專屬拒絕語意後重新收斂 | PARTIAL -> DONE | `validate_quick_gate_output_path_guard_contract_case` |
| 2026-04-05 | T121 | 補上 outputs-root create-path guard 並重新收斂 | PARTIAL -> DONE | `validate_quick_gate_output_path_guard_contract_case` |
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
| 2026-04-05 | T158 | 新增 quick gate synthetic registry symbol-resolution static contract 並驗證 | NEW -> DONE | `tools/local_regression/run_quick_gate.py` |
| 2026-04-05 | T159 | 新增 GUI 近期視窗 / signal overlay / timing contract 並驗證 | NEW -> DONE | `validate_gui_chart_recent_view_signal_overlay_contract_case` |
| 2026-04-05 | T160 | 新增 chart payload optional overlay keys contract 並驗證 | NEW -> DONE | `validate_chart_payload_optional_overlay_keys_contract_case` |
| 2026-04-05 | T161 | 新增 GUI deep-dark theme / keyboard pan / dynamic candle width contract 並驗證 | NEW -> DONE | `validate_gui_dark_theme_and_keyboard_pan_contract_case` |
| 2026-04-05 | T162 | 新增 synthetic GUI canvas stub cleanup contract 並驗證 | NEW -> DONE | `validate_gui_navigation_canvas_stub_cleanup_contract_case` |
| 2026-04-05 | T163 | 新增 synthetic GUI canvas stub toolbar contract 並驗證 | NEW -> DONE | `validate_gui_navigation_canvas_stub_toolbar_contract_case` |
| 2026-04-05 | T164 | 新增 signal annotation meta contract 並驗證 | NEW -> DONE | `validate_record_signal_annotation_meta_contract_case` |
| 2026-04-05 | T165 | 新增 GUI overlay layout / dark widget styles / pixel-anchor mouse pan contract 並驗證 | NEW -> DONE | `validate_gui_chart_overlay_layout_and_pan_contract_case` |
| 2026-04-05 | T166 | 新增 GUI workbench theme accent token contract 並驗證 | NEW -> DONE | `validate_workbench_theme_accent_symbol_contract_case` |
| 2026-04-05 | T167 | 新增 GUI scanner / Console / latest-view contract 並驗證 | NEW -> DONE | `validate_gui_scanner_console_and_latest_contract_case` |
| 2026-04-05 | T168 | 新增 GUI 右側 sidebar / latest next-day preview contract 並驗證 | NEW -> DONE | `validate_gui_sidebar_latest_preview_contract_case` |
| 2026-04-06 | B94 | 新增單股 refined visual / next-day preview contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-06 | B95 | 新增 GUI runtime 固定文案晶片 / 延續候選多日預掛線連續性 contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-06 | B96 | 新增 GUI 圖例頂部貼齊 / 價格軸防裁切 / latest extended preview contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-06 | B97 | 新增 latest raw-signal preview helper import contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-06 | B98 | 新增延續候選固定反事實 barrier / today-orderable 分層 contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_flow_cases.py` |
| 2026-04-06 | B99 | 新增 non-error synthetic validator 非法初始資金常值 guard，釘死 `initial_capital<=0` 只能留在 explicit error-path case | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-06 | B100 | 新增 synthetic meta shared path helper 顯式 import guard，避免 `build_project_absolute_path` 漏 import 直到 coverage suite runtime 才 NameError | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-06 | B101 | 新增 GUI 賣訊註記限縮 / future preview autoscale / 黃色強制結算圖示 contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-06 | B102 | 新增 GUI 賣訊 title-only / 綠色指標賣出 marker / 停損賣出框交易次數 / 買訊 frozen 價格 autoscale / 黃色停利線 contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-06 | B103 | 新增 GUI 側欄停利圖示同步 / 交易次數排序與 completed round-trip 一致性 contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-06 | T173 | 新增 GUI 單股 refined visual contract 並驗證 | NEW -> DONE | `validate_gui_single_stock_refined_visual_contract_case` |
| 2026-04-06 | T174 | 新增 GUI 延續候選多日預掛線連續性與固定晶片文案 contract 並驗證 | NEW -> DONE | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-06 | T175 | 新增 GUI 圖例頂部貼齊 / 價格軸防裁切 / latest extended preview contract 並驗證 | NEW -> DONE | `validate_gui_chart_margin_and_latest_extended_preview_contract_case` |
| 2026-04-06 | T176 | 新增 latest raw-signal preview helper import contract 並驗證 | NEW -> DONE | `validate_gui_latest_raw_signal_preview_helper_contract_case` |
| 2026-04-06 | T177 | 新增延續候選固定反事實 barrier / today-orderable 分層 contract 並驗證 | NEW -> DONE | `validate_synthetic_extended_signal_a2_frozen_plan_case` |
| 2026-04-06 | T178 | 新增 non-error synthetic validator 非法初始資金常值 contract 並驗證 | NEW -> DONE | `validate_synthetic_case_non_error_initial_capital_contract_case` |
| 2026-04-06 | T179 | 新增 synthetic meta shared path helper 顯式 import contract 並驗證 | NEW -> DONE | `validate_synthetic_meta_cases_build_project_absolute_path_import_contract_case` |
| 2026-04-06 | T180 | 新增 GUI 賣訊註記限縮 / future preview 限價 autoscale / 黃色強制結算圖示 contract 並驗證 | NEW -> DONE | `validate_gui_signal_annotation_and_forced_close_visual_contract_case` |
| 2026-04-06 | T181 | 新增 GUI 賣訊 title-only / 綠色指標賣出 marker / 停損賣出框交易次數 / 買訊限價 autoscale / 黃色停利線 contract 並驗證 | NEW -> DONE | `validate_gui_trade_marker_and_tp_visual_contract_case` |
| 2026-04-06 | T182 | 新增 GUI 側欄停利圖示同步 / 交易次數排序與 completed round-trip 一致性 contract 並驗證 | NEW -> DONE | `validate_gui_trade_count_and_sidebar_sync_contract_case` |
| 2026-04-07 | B104 | 新增最小物理限制優先 / `L` 僅作進場與 sizing / `P_fill` 首個可執行 stop-tp / 延續候選固定反事實 barrier 的專案設定原則後主表收斂為 DONE | NEW -> DONE | `doc/PROJECT_SETTINGS.md` |
| 2026-04-07 | B105 | 新增 runtime `init_sl` 單一真理來源 / trailing 與初始停損脫鉤 / `T` 不得依實際成交價重算的交易規格後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_flow_cases.py` |
| 2026-04-07 | B106 | 新增空 `price_df` 仍需回傳可正規化 placeholder chart payload 的 GUI / debug 契約後主表收斂為 DONE | NEW -> DONE | `tools/trade_analysis/reporting.py` |
| 2026-04-07 | B107 | 新增 normalize chart payload dict-literal 必須顯式提供 `x` 欄位的 meta / GUI 契約後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-07 | B108 | 新增空 `price_df` 單股回測核心必須 early-return 穩定 stats 的韌性契約後主表收斂為 DONE | NEW -> DONE | `core/backtest_core.py` |
| 2026-04-07 | B109 | 新增 synthetic registry import 目標模組必須實際存在對應 validator symbol 的 meta 契約後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-07 | B110 | 新增 broad exception handler 綁定例外且可追蹤的錯誤處理 / meta 契約後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-07 | B111 | 新增 optional dependency fallback traceability contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-07 | B112 | 新增 GUI TclError fallback traceability contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-07 | B113 | 新增 specific pass-only exception traceability contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-07 | B114 | 新增 GUI / debug trade-box 預留 / 實支 / round-trip total-pnl / scanner-capital-basis contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-07 | B115 | 新增 debug backtest PIT history snapshot patch seam 穩定契約後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-07 | B116 | 新增 GUI 買訊 annotation 錨定訊號低點、買進三角錨定成交價的 visual contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-07 | B117 | 新增 synthetic contract multi-ticker fixture 不得再讀取舊 `case["price_df"]` key 的 meta contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-07 | B118 | 新增 GUI 買訊 annotation helper 顯式 import meta contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-07 | B119 | 新增 GUI trade-count validator 禁止綁死過時 exit snippet 的 meta contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-07 | T183 | 新增專案設定最小物理限制 / `P_fill` 首個可執行 stop-tp / 延續候選固定反事實 barrier 原則 meta contract 並驗證 | NEW -> DONE | `validate_project_settings_init_sl_frozen_plan_principle_case` |
| 2026-04-07 | T184 | 新增 `init_sl` 單一真理來源 runtime synthetic case 並驗證 | NEW -> DONE | `validate_synthetic_init_sl_single_source_runtime_case` |
| 2026-04-07 | T185 | 新增空 `price_df` chart payload placeholder contract 並驗證 | NEW -> DONE | `validate_debug_empty_price_df_chart_payload_contract_case` |
| 2026-04-07 | T186 | 新增 normalize chart payload dict-literal `x` 欄位 meta contract 並驗證 | NEW -> DONE | `validate_synthetic_case_normalize_chart_payload_literal_x_contract_case` |
| 2026-04-07 | T187 | 新增空 `price_df` 單股回測核心 direct synthetic contract 並驗證 | NEW -> DONE | `validate_synthetic_empty_backtest_df_contract_case` |
| 2026-04-07 | T188 | 新增 synthetic registry import 目標解析 meta contract 並驗證 | NEW -> DONE | `validate_synthetic_cases_import_target_resolution_contract_case` |
| 2026-04-07 | T189 | 新增 broad exception traceability meta contract 並驗證 | NEW -> DONE | `validate_broad_exception_traceability_contract_case` |
| 2026-04-07 | T190 | 新增 optional dependency fallback traceability meta contract 並驗證 | NEW -> DONE | `validate_optional_dependency_fallback_traceability_contract_case` |
| 2026-04-07 | T191 | 新增 GUI TclError fallback traceability meta contract 並驗證 | NEW -> DONE | `validate_gui_tcl_fallback_traceability_contract_case` |
| 2026-04-07 | T192 | 新增 specific pass-only exception traceability meta contract 並驗證 | NEW -> DONE | `validate_specific_pass_only_exception_traceability_contract_case` |
| 2026-04-07 | T193 | 新增 GUI trade-box 預留 / 實支 / round-trip total-pnl / scanner-capital-basis contract 並驗證 | NEW -> DONE | `validate_gui_trade_box_capital_and_round_trip_contract_case` |
| 2026-04-07 | T194 | 新增 debug backtest PIT history snapshot patch seam meta contract 並驗證 | NEW -> DONE | `validate_debug_backtest_history_snapshot_patch_seam_contract_case` |
| 2026-04-07 | T195 | 新增 GUI 買訊 annotation 錨定訊號低點且買進三角錨定成交價 contract 並驗證 | NEW -> DONE | `validate_gui_buy_signal_annotation_anchor_price_contract_case` |
| 2026-04-07 | T196 | 新增 synthetic contract 禁止舊 `case["price_df"]` key 存取的 meta contract 並驗證 | NEW -> DONE | `validate_synthetic_contract_cases_no_legacy_price_df_case_key_contract_case` |
| 2026-04-07 | T197 | 新增 GUI 買訊 annotation helper 顯式 import meta contract 並驗證 | NEW -> DONE | `validate_gui_buy_signal_annotation_helper_import_contract_case` |
| 2026-04-07 | T198 | 新增 GUI trade-count validator 禁止綁死過時 exit snippet 的 meta contract 並驗證 | NEW -> DONE | `validate_gui_trade_count_contract_no_legacy_exit_snippet_case` |
| 2026-04-08 | B120 | 新增投組 cash-capped entry 不得丟失 fill-based first-actionable stop/tp 所需欄位的一致性規格後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_flow_cases.py` |
| 2026-04-08 | B121 | 新增 candidate-plan limit-based sizing stop 不得回頭否決成交的一致性規格後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_flow_cases.py` |
| 2026-04-08 | B122 | 新增 quick gate 前置攔截 synthetic registry import target wiring 錯誤的契約後主表收斂為 DONE | NEW -> DONE | `tools/local_regression/run_quick_gate.py` |
| 2026-04-08 | T199 | 新增投組 cash-capped entry 保留 fill-based first-actionable 欄位 synthetic case 並驗證 | NEW -> DONE | `validate_synthetic_portfolio_entry_preserves_fill_based_first_actionable_case` |
| 2026-04-08 | T200 | 新增實際成交價低於 candidate limit-based sizing stop 仍須成交 synthetic case 並驗證 | NEW -> DONE | `validate_synthetic_fill_below_limit_based_sizing_stop_still_enters_case` |
| 2026-04-08 | T201 | 新增 quick gate synthetic registry import target static check 並驗證 | NEW -> DONE | `validate_quick_gate_synthetic_registry_import_targets_contract_case` |
| 2026-04-09 | B26 | 檢出 checklist `T` 單列單入口 parser 未把 formal command string 視為單一測試入口，主表改回 PARTIAL | DONE -> PARTIAL | `tools/local_regression/run_meta_quality.py` |
| 2026-04-09 | B26 | 補上 formal command string 單列單入口 guard 並重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-09 | B88 | 檢出 GUI scanner 每檔輸出與歷史績效股下拉未顯示資產成長 / 排序探針，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-09 | B88 | 補上 scanner row / history dropdown 資產成長 sort probe contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-09 | B123 | 新增正式帳務必須以整數 exact-accounting ledger / cost-basis / tick-limit 單一真理來源收斂的契約後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_unit_cases.py` |
| 2026-04-09 | B124 | 新增單股 backtest public stats legacy schema static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-09 | B125 | 新增單股 backtest / debug exact cash / equity path static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-09 | B126 | 新增 debug backtest 買進現金扣減 static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-09 | T167 | 檢出 GUI scanner / history dropdown contract 未覆蓋資產成長 sort probe 顯示 | DONE -> PARTIAL | `validate_gui_scanner_console_and_latest_contract_case` |
| 2026-04-09 | T167 | 補上 GUI scanner / history dropdown 資產成長 sort probe contract 並驗證 | PARTIAL -> DONE | `validate_gui_scanner_console_and_latest_contract_case` |
| 2026-04-09 | T202 | 新增 formal consistency step 完整 command string 必須列入 checklist `T` 摘要的 meta contract 並同步補齊映射 | NEW -> DONE | `tools/validate/cli.py --dataset reduced` |
| 2026-04-09 | T203 | 新增 checklist `T` formal command string 單列單入口 contract 並驗證 | NEW -> DONE | `validate_checklist_f2_formal_command_single_entry_case` |
| 2026-04-09 | T204 | 新增 exact-accounting ledger conservation contract 並驗證 | NEW -> DONE | `validate_exact_accounting_ledger_conservation_case` |
| 2026-04-09 | T205 | 新增 exact-accounting partial-exit cost-basis allocation contract 並驗證 | NEW -> DONE | `validate_exact_accounting_cost_basis_allocation_case` |
| 2026-04-09 | T206 | 新增 exact-accounting integer tick / limit hit contract 並驗證 | NEW -> DONE | `validate_exact_accounting_tick_limit_integer_case` |
| 2026-04-09 | T207 | 新增 exact-accounting cash / risk boundary contract 並驗證 | NEW -> DONE | `validate_exact_accounting_cash_risk_boundary_case` |
| 2026-04-09 | T208 | 新增 exact-accounting single-stock / portfolio parity contract 並驗證 | NEW -> DONE | `validate_exact_accounting_single_vs_portfolio_parity_case` |
| 2026-04-09 | T209 | 新增 exact-accounting display-derived field contract 並驗證 | NEW -> DONE | `validate_exact_accounting_display_derived_case` |
| 2026-04-09 | T210 | 新增單股 backtest public stats legacy schema static contract 並驗證 | NEW -> DONE | `validate_single_backtest_stats_legacy_schema_contract_case` |
| 2026-04-09 | T211 | 新增單股 backtest / debug exact cash / equity path static contract 並驗證 | NEW -> DONE | `validate_single_backtest_exact_cash_path_contract_case` |
| 2026-04-09 | T212 | 新增 debug backtest 買進現金扣減 static contract 並驗證 | NEW -> DONE | `validate_debug_backtest_entry_cash_path_contract_case` |
| 2026-04-09 | T213 | 補 completed-trade 可見 rounded leg reconciliation contract，避免 debug / portfolio trade log 與 core completed trades 因 0.01 殘差分叉 | NEW -> DONE | `validate_exact_accounting_display_leg_reconciliation_case` |
| 2026-04-10 | B128 | 新增 shared display money-rounding helper static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-10 | B129 | 新增 real-case completed-trade rounding oracle static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/real_case_assertions.py` |
| 2026-04-10 | B130 | 新增 trade-rebuild shared rounding helper static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/trade_rebuild.py` |
| 2026-04-10 | B131 | 新增 debug forced-closeout exact total-pnl static contract 後主表收斂為 DONE | NEW -> DONE | `tools/trade_analysis/exit_flow.py` |
| 2026-04-10 | B132 | 新增 synthetic-meta source-path binding static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-10 | B133 | 新增 unit-display rounding-helper static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-10 | B134 | 新增 debug exit display-capital exact-ledger static contract 後主表收斂為 DONE | NEW -> DONE | `tools/trade_analysis/exit_flow.py` |
| 2026-04-10 | B135 | 新增 debug exit entry-capital fallback exact-ledger static contract 後主表收斂為 DONE | NEW -> DONE | `tools/trade_analysis/exit_flow.py` |
| 2026-04-10 | B136 | 新增 debug entry display-capital exact-total static contract 後主表收斂為 DONE | NEW -> DONE | `tools/trade_analysis/entry_flow.py` |
| 2026-04-10 | B137 | 新增 debug half-exit leg-return exact-allocated-cost static contract 後主表收斂為 DONE | NEW -> DONE | `tools/trade_analysis/exit_flow.py` |
| 2026-04-10 | B138 | 新增 debug sell-signal profit-pct exact mark-to-market static contract 後主表收斂為 DONE | NEW -> DONE | `tools/trade_analysis/backtest.py` |
| 2026-04-10 | B139 | 新增 debug fallback exact-helper static contract 後主表收斂為 DONE | NEW -> DONE | `tools/trade_analysis/backtest.py` |
| 2026-04-10 | B140 | 新增 average-price total helper static contract 後主表收斂為 DONE | NEW -> DONE | `core/exact_accounting.py` |
| 2026-04-10 | B141 | 新增投組 rotation exact mark-to-market return static contract 後主表收斂為 DONE | NEW -> DONE | `core/portfolio_exits.py` |
| 2026-04-10 | B142 | 新增 validator/oracle exact-ledger total static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_unit_cases.py` |
| 2026-04-10 | B143 | 新增 checklist DONE 測試摘要表結構 static contract 後主表收斂為 DONE | NEW -> DONE | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-10 | B144 | 新增 mutating validator oracle snapshot static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_take_profit_cases.py` |
| 2026-04-10 | B145 | 新增 shared helper import static contract 後主表收斂為 DONE | NEW -> DONE | `core/price_utils.py` |
| 2026-04-10 | T214 | 新增 shared display money-rounding helper static contract 並驗證 | NEW -> DONE | `validate_display_money_rounding_helper_contract_case` |
| 2026-04-10 | T215 | 新增 real-case completed-trade rounding oracle static contract 並驗證 | NEW -> DONE | `validate_real_case_completed_trade_rounding_oracle_contract_case` |
| 2026-04-10 | T216 | 新增 trade-rebuild shared rounding helper static contract 並驗證 | NEW -> DONE | `validate_trade_rebuild_rounding_helper_contract_case` |
| 2026-04-10 | T217 | 新增 debug forced-closeout exact total-pnl static contract 並驗證 | NEW -> DONE | `validate_debug_forced_closeout_exact_total_pnl_contract_case` |
| 2026-04-10 | T218 | 新增 synthetic-meta source-path binding static contract 並驗證 | NEW -> DONE | `validate_synthetic_meta_source_path_binding_contract_case` |
| 2026-04-10 | T219 | 新增 unit-display rounding-helper static contract 並驗證 | NEW -> DONE | `validate_unit_display_rounding_helper_contract_case` |
| 2026-04-10 | T220 | 新增 debug exit display-capital exact-ledger static contract 並驗證 | NEW -> DONE | `validate_debug_exit_display_capital_uses_ledger_totals_contract_case` |
| 2026-04-10 | T221 | 新增 debug exit entry-capital fallback exact-ledger static contract 並驗證 | NEW -> DONE | `validate_debug_exit_entry_capital_fallback_contract_case` |
| 2026-04-10 | T222 | 新增 debug entry display-capital exact-total static contract 並驗證 | NEW -> DONE | `validate_debug_entry_display_capital_uses_exact_total_contract_case` |
| 2026-04-10 | T223 | 新增 debug half-exit leg-return exact-allocated-cost static contract 並驗證 | NEW -> DONE | `validate_debug_half_exit_leg_return_pct_uses_allocated_cost_contract_case` |
| 2026-04-10 | T224 | 新增 debug sell-signal profit-pct exact mark-to-market static contract 並驗證 | NEW -> DONE | `validate_debug_sell_signal_profit_pct_uses_exact_mark_to_market_contract_case` |
| 2026-04-10 | T225 | 新增 debug fallback exact-helper static contract 並驗證 | NEW -> DONE | `validate_debug_exact_fallback_helpers_contract_case` |
| 2026-04-10 | T226 | 新增 average-price total helper static contract 並驗證 | NEW -> DONE | `validate_average_price_total_helper_contract_case` |
| 2026-04-10 | T227 | 新增投組 rotation exact mark-to-market return static contract 並驗證 | NEW -> DONE | `validate_portfolio_rotation_mark_to_market_return_contract_case` |
| 2026-04-10 | T228 | 新增 validator/oracle exact-ledger total static contract 並驗證 | NEW -> DONE | `validate_validator_oracles_use_exact_ledger_totals_contract_case` |
| 2026-04-10 | T229 | 新增 checklist DONE 測試摘要表結構 static contract 並驗證 | NEW -> DONE | `validate_checklist_done_test_summary_markdown_structure_case` |
| 2026-04-10 | T230 | 新增 same-bar stop-priority oracle snapshot static contract 並驗證 | NEW -> DONE | `validate_same_bar_stop_priority_oracle_snapshots_pre_exit_cost_basis_contract_case` |
| 2026-04-10 | T231 | 新增 price_utils average-price total import static contract 並驗證 | NEW -> DONE | `validate_price_utils_average_price_total_import_contract_case` |
| 2026-04-10 | T232 | 新增 price_utils array tick-normalization shared-helper static contract 並驗證 | NEW -> DONE | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | B21 | 以 full dataset 額外審計檢出 portfolio export 的 Plotly fallback `except (...)` tuple 引用未匯入 `webbrowser`，主表改回 PARTIAL | DONE -> PARTIAL | `tools/portfolio_sim/reporting.py` |
| 2026-04-11 | B21 | 補齊 portfolio export Plotly fallback 匯入與 fallback reporting contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_reporting_cases.py` |
| 2026-04-11 | B26 | 檢出 `G` 區 `B124` 歷史收斂列 note 欄混寫兩個檔案 reference，違反單 note entry 規則，主表改回 PARTIAL | DONE -> PARTIAL | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-11 | B26 | 將 `B124` 歷史收斂列 note 欄改回單一 code reference 後重新收斂為 DONE | PARTIAL -> DONE | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-11 | B26 | 依 bundle 實際失敗檢出 `G` 區同日追蹤列未依 tracking ID 機械排序，主表改回 PARTIAL | DONE -> PARTIAL | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-11 | B26 | 將 `G` 區收斂紀錄改為依日期、namespace、數字尾碼穩定排序後重新收斂為 DONE | PARTIAL -> DONE | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-11 | B95 | 檢出 GUI extended preview continuity contract 仍比對舊 `build_extended_candidate_plan_from_signal(..., ticker=ticker)` signature，未同步 `security_profile` 傳遞，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-11 | B95 | 將 GUI extended preview continuity contract 同步到 `ticker` + `security_profile` counterfactual signature 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-11 | B95 | 檢出 GUI extended preview continuity contract 尚未同步 `trade_date` counterfactual signature，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-11 | B95 | 將 GUI extended preview continuity contract 同步到 `ticker` + `security_profile` + `trade_date` counterfactual signature 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-11 | B95 | 檢出 GUI extended preview continuity contract 仍比對 `trade_date=current_date`，未同步 debug entry flow 的 `effective_trade_date` fallback signature，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-11 | B95 | 將 GUI extended preview continuity contract 同步到 `trade_date=effective_trade_date` counterfactual signature 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-11 | B123 | 以 reduced dataset 實際比對檢出跨 tick band 漲跌停價仍沿用基準價 band，主表改回 PARTIAL | DONE -> PARTIAL | `core/exact_accounting.py` |
| 2026-04-11 | B123 | 檢出 ETF / ETN / REIT 類商品仍沿用股票 tick ladder，主表改回 PARTIAL | DONE -> PARTIAL | `core/exact_accounting.py` |
| 2026-04-11 | B123 | 修正漲跌停價改依 raw-limit 價本身決定 tick band 後重新收斂為 DONE | PARTIAL -> DONE | `core/exact_accounting.py` |
| 2026-04-11 | B123 | 補齊 ticker / metadata 商品 profile 自動辨識與兩級 tick 路由後重新收斂為 DONE | PARTIAL -> DONE | `core/exact_accounting.py` |
| 2026-04-11 | B123 | 以 reduced dataset 實際比對檢出 ETF / ETN / REIT / 債券 ETF 仍共用 stock-only 賣出交易稅，主表改回 PARTIAL | DONE -> PARTIAL | `core/exact_accounting.py` |
| 2026-04-11 | B123 | 補齊商品別 / 交易日期驅動的賣出交易稅路由後重新收斂為 DONE | PARTIAL -> DONE | `core/exact_accounting.py` |
| 2026-04-11 | B124 | 檢出 `build_backtest_stats()` 內部已使用 `trade_date=final_date` 但函式簽名與 caller 尚未同步日期上下文，改回 PARTIAL | DONE -> PARTIAL | `core/backtest_finalize.py` |
| 2026-04-11 | B124 | 補齊 `build_backtest_stats()` 的 `final_date` 簽名、empty/final caller 傳遞與 static contract 後重新收斂為 DONE | PARTIAL -> DONE | `core/backtest_finalize.py` |
| 2026-04-11 | B124 | 檢出 `build_backtest_stats()` 的 `tpPrice` preview 仍用手寫浮點公式，未重用共享 target helper，改回 PARTIAL | DONE -> PARTIAL | `core/backtest_finalize.py` |
| 2026-04-11 | B124 | 補齊 `build_backtest_stats()` 的 `buyPrice` / `sellPrice` / `tpPrice` preview helper contract，改用共享 stop / target helper 後重新收斂為 DONE | PARTIAL -> DONE | `core/backtest_finalize.py` |
| 2026-04-11 | B126 | 檢出 debug entry flow caller 已傳 `ticker` 但 callee 簽名未同步更新，主表改回 PARTIAL | DONE -> PARTIAL | `tools/trade_analysis/entry_flow.py` |
| 2026-04-11 | B126 | 將 debug entry flow 簽名、商品 profile 傳遞與 static contract 一併補齊後重新收斂為 DONE | PARTIAL -> DONE | `tools/trade_analysis/entry_flow.py` |
| 2026-04-11 | B126 | 檢出 debug entry flow / backtest caller-callee 尚未同步 `trade_date` keyword 與 builder 傳遞，改回 PARTIAL | DONE -> PARTIAL | `tools/trade_analysis/entry_flow.py` |
| 2026-04-11 | B126 | 補齊 debug entry flow / backtest 的 `trade_date` 簽名、caller 傳遞與 static contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/trade_analysis/entry_flow.py` |
| 2026-04-11 | B126 | 檢出 debug entry flow 實作仍未真正接受與傳遞 `trade_date`，主表改回 PARTIAL | DONE -> PARTIAL | `tools/trade_analysis/entry_flow.py` |
| 2026-04-11 | B126 | 將 debug entry flow 的 `trade_date` 簽名、caller 傳遞與 static contract 真正落地後重新收斂為 DONE | PARTIAL -> DONE | `tools/trade_analysis/entry_flow.py` |
| 2026-04-11 | B134 | 以 full dataset 額外審計檢出 debug exit display-total helper 仍引用自由變數 `position` / `current_date`，主表改回 PARTIAL | DONE -> PARTIAL | `tools/trade_analysis/exit_flow.py` |
| 2026-04-11 | B134 | 將 debug exit display-total helper 改為顯式接受 `position` / `current_date` 並擴充 static contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B135 | 檢出 debug exit entry-capital fallback validator 讀錯 helper，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B135 | 將 debug exit entry-capital fallback validator 改為分別檢查 `_resolve_full_entry_capital_milli()` 與 `_resolve_display_sell_total_milli()` 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B137 | 檢出 debug half-exit leg-return static validator 仍要求舊 milli_to_money ratio，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B137 | 將 debug half-exit leg-return static validator 同步到 integer-total ratio 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B138 | 檢出 debug sell-signal profit-pct static contract 仍比對舊 helper / helper-call signature，未同步 `signal_date` keyword，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B138 | 將 debug sell-signal profit-pct static contract 同步到 `signal_date` helper / helper-call signature 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B141 | 檢出投組 rotation exact mark-to-market return static contract 仍比對舊 helper-call signature，未同步 `trade_date=today`，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B141 | 將投組 rotation exact mark-to-market return static contract 同步到 `trade_date=today` helper-call signature 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B142 | 檢出 scanner/reference oracle 的 projected-qty 仍未同步傳遞 `ticker` / `trade_date`，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/scanner_expectations.py` |
| 2026-04-11 | B142 | 補齊 scanner/reference oracle 的商品別與日期上下文傳遞 contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/scanner_expectations.py` |
| 2026-04-11 | B146 | 檢出 portfolio rotation 賣出路徑仍引用未宣告 `ticker` 自由變數，主表改回 PARTIAL | DONE -> PARTIAL | `core/portfolio_exits.py` |
| 2026-04-11 | B146 | 檢出 scalar / array 價格正規化 caller 尚未一路傳遞 ticker / security_profile，主表改回 PARTIAL | DONE -> PARTIAL | `core/price_utils.py` |
| 2026-04-11 | B146 | 補齊 portfolio rotation 賣出路徑改用 `weakest_ticker` 並擴充 static contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B146 | 補齊 signal / backtest / portfolio / position 路徑的 ticker / security_profile 傳遞後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B146 | 檢出 candidate-plan resize / scanner projected qty 仍未將 `ticker` / `trade_date` 傳入 sizing helper，改回 PARTIAL | DONE -> PARTIAL | `core/entry_plans.py` |
| 2026-04-11 | B146 | 補齊 candidate-plan resize / scanner projected qty 的商品別與日期上下文傳遞 contract 後重新收斂為 DONE | PARTIAL -> DONE | `core/entry_plans.py` |
| 2026-04-11 | B146 | 檢出 prepared scanner frame 最新交易日解析尚未覆蓋 attrs / MultiIndex fallback，主表改回 PARTIAL | DONE -> PARTIAL | `tools/scanner/stock_processor.py` |
| 2026-04-11 | B146 | 補齊 prepared scanner frame 最新交易日 attrs / MultiIndex fallback 與 static contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/scanner/stock_processor.py` |
| 2026-04-11 | B146 | 檢出 prepared scanner frame 最新交易日仍只接受 `df["Date"]` 欄位、未支援 index 日期來源，改回 PARTIAL | DONE -> PARTIAL | `tools/scanner/stock_processor.py` |
| 2026-04-11 | B146 | 補齊 prepared scanner frame 最新交易日欄位 / index 雙來源解析與 static contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/scanner/stock_processor.py` |
| 2026-04-11 | B146 | 檢出 scanner runtime 與 validator/oracle 各自維護最新交易日解析 helper、違反單一真理來源，主表改回 PARTIAL | DONE -> PARTIAL | `core/data_utils.py` |
| 2026-04-11 | B146 | 以共享 frame 最新交易日 helper 收斂 scanner runtime 與 validator/oracle 後重新收斂為 DONE | PARTIAL -> DONE | `core/data_utils.py` |
| 2026-04-11 | B147 | 新增正式入口摘要同步契約，要求 apps/test_suite.py 的 Txx 註解列舉與實際 synthetic registry 同步 | NEW -> DONE | `apps/test_suite.py` |
| 2026-04-11 | B147 | 檢出正式入口摘要註解新增 exact-contract 後仍漏列 T234/T235，主表改回 PARTIAL | DONE -> PARTIAL | `apps/test_suite.py` |
| 2026-04-11 | B147 | 補齊正式入口摘要註解與 summary meta contract 對 T234/T235 的同步後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B147 | 檢出正式入口摘要註解與 summary meta contract 尚未同步新增的 T236，主表改回 PARTIAL | DONE -> PARTIAL | `apps/test_suite.py` |
| 2026-04-11 | B147 | 將正式入口摘要註解與 summary meta contract 同步擴充到 T236 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B147 | 檢出 summary comment contract 尚未明確釘死 T235 明示檢查，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B147 | 補齊 T235 明示檢查與正式入口摘要同步後重新收斂為 DONE | PARTIAL -> DONE | `apps/test_suite.py` |
| 2026-04-11 | B147 | 檢出 summary comment coverage contract 仍掃描整份 source_text 而非頂部摘要註解 block，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B147 | 將 summary comment coverage contract 收斂到頂部摘要註解 block 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B148 | 新增 exact-ledger ratio path 契約，釘死 return / rotation ratio 不得先轉 float money 再相除 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B149 | 新增 debug exit milli binding 契約，釘死 final-exit total_return_pct 不得引用未定義 total_pnl_milli | NEW -> DONE | `tools/trade_analysis/exit_flow.py` |
| 2026-04-11 | B149 | 檢出 milli-binding contract 的 forced-closeout 負向守衛仍比對錯誤舊字串，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B149 | 補齊 forced-closeout 舊 float total-pnl 路徑負向守衛後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B150 | 新增 core R-multiple exact-ledger 契約，釘死核心回測 / 投組統計不得以 float `total_pnl / initial_risk_total` 累計 `r_mult` | NEW -> DONE | `core/backtest_core.py` |
| 2026-04-11 | T40 | 以 full dataset 額外審計檢出 portfolio export reporting synthetic case 尚未覆蓋 Plotly import failure fallback，改回 PARTIAL | DONE -> PARTIAL | `validate_portfolio_export_report_artifacts_case` |
| 2026-04-11 | T40 | 擴充 portfolio export reporting synthetic case 納入 Plotly import failure fallback artifact / traceability 後重新驗證 | PARTIAL -> DONE | `validate_portfolio_export_report_artifacts_case` |
| 2026-04-11 | T174 | 檢出 GUI extended preview continuity static contract 仍比對舊 counterfactual signature，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T174 | 將 GUI extended preview continuity static contract 同步到 `ticker` + `security_profile` signature 後重新驗證 | PARTIAL -> DONE | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T174 | 檢出 GUI extended preview continuity static contract 尚未同步 `trade_date` signature，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T174 | 將 GUI extended preview continuity static contract 同步到 `ticker` + `security_profile` + `trade_date` signature 後重新驗證 | PARTIAL -> DONE | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T174 | 檢出 GUI extended preview continuity static contract 仍比對 `trade_date=current_date`，未同步 `effective_trade_date` fallback signature，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T174 | 將 GUI extended preview continuity static contract 同步到 `trade_date=effective_trade_date` signature 後重新驗證 | PARTIAL -> DONE | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T206 | 以 reduced dataset 實際比對檢出 exact-accounting tick/limit unit contract 尚未覆蓋跨 tick band 漲跌停價案例，改回 PARTIAL | DONE -> PARTIAL | `validate_exact_accounting_tick_limit_integer_case` |
| 2026-04-11 | T206 | 檢出 exact-accounting tick/limit unit contract 尚未覆蓋 ETF / ETN / REIT 類商品 profile 與兩級 tick 案例，改回 PARTIAL | DONE -> PARTIAL | `validate_exact_accounting_tick_limit_integer_case` |
| 2026-04-11 | T206 | 擴充 exact-accounting tick/limit unit contract 納入 ticker 自動辨識與 ETF / ETN / REIT 兩級 tick 案例後重新驗證 | PARTIAL -> DONE | `validate_exact_accounting_tick_limit_integer_case` |
| 2026-04-11 | T206 | 擴充 exact-accounting tick/limit unit contract 納入跨 tick band 漲跌停價案例後重新驗證 | PARTIAL -> DONE | `validate_exact_accounting_tick_limit_integer_case` |
| 2026-04-11 | T206 | 以 reduced dataset 實際比對檢出 exact-accounting tick/limit unit contract 尚未覆蓋 ETF / ETN / REIT / 債券 ETF 商品別賣出交易稅，改回 PARTIAL | DONE -> PARTIAL | `validate_exact_accounting_tick_limit_integer_case` |
| 2026-04-11 | T206 | 擴充 exact-accounting tick/limit unit contract 納入商品別 / 交易日期驅動的賣出交易稅案例後重新驗證 | PARTIAL -> DONE | `validate_exact_accounting_tick_limit_integer_case` |
| 2026-04-11 | T210 | 檢出 `build_backtest_stats()` preview 價格 contract 未釘住共享 stop / target helper、改回 PARTIAL | DONE -> PARTIAL | `validate_single_backtest_stats_legacy_schema_contract_case` |
| 2026-04-11 | T210 | 補齊 `build_backtest_stats()` stop / tp preview helper static contract 後重新收斂為 DONE | PARTIAL -> DONE | `validate_single_backtest_stats_legacy_schema_contract_case` |
| 2026-04-11 | T212 | 檢出 debug backtest entry cash-path static contract 尚未釘死 `ticker` / `security_profile` caller-callee 相容，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_backtest_entry_cash_path_contract_case` |
| 2026-04-11 | T212 | 擴充 debug backtest entry cash-path static contract 納入 entry flow 簽名與商品 profile 傳遞後重新驗證 | PARTIAL -> DONE | `validate_debug_backtest_entry_cash_path_contract_case` |
| 2026-04-11 | T212 | 檢出 debug backtest entry cash-path static contract 尚未釘死 `trade_date` caller-callee 相容與 builder 傳遞，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_backtest_entry_cash_path_contract_case` |
| 2026-04-11 | T212 | 擴充 debug backtest entry cash-path static contract 納入 `trade_date` 簽名、caller 與 builder 傳遞後重新驗證 | PARTIAL -> DONE | `validate_debug_backtest_entry_cash_path_contract_case` |
| 2026-04-11 | T212 | 檢出 debug backtest entry cash-path static contract 雖已要求 `trade_date`，但實作仍未真正接受與傳遞，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_backtest_entry_cash_path_contract_case` |
| 2026-04-11 | T212 | 將 debug backtest entry cash-path static contract 與實作同步到 `trade_date` 簽名、caller 傳遞後重新驗證 | PARTIAL -> DONE | `validate_debug_backtest_entry_cash_path_contract_case` |
| 2026-04-11 | T220 | 以 full dataset 額外審計檢出 debug exit display-total helper static contract 尚未釘死顯式 `position` / `current_date` 上下文，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_exit_display_capital_uses_ledger_totals_contract_case` |
| 2026-04-11 | T220 | 擴充 debug exit display-total helper static contract 納入顯式 `position` / `current_date` 簽名與 caller 傳遞後重新驗證 | PARTIAL -> DONE | `validate_debug_exit_display_capital_uses_ledger_totals_contract_case` |
| 2026-04-11 | T221 | 檢出 debug exit entry-capital fallback contract 讀錯 helper，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_exit_entry_capital_fallback_contract_case` |
| 2026-04-11 | T221 | 將 debug exit entry-capital fallback contract 改為分別檢查 entry-capital helper 與 sell-total helper 後重新驗證 | PARTIAL -> DONE | `validate_debug_exit_entry_capital_fallback_contract_case` |
| 2026-04-11 | T223 | 檢出 debug half-exit leg-return static contract 仍要求舊 milli_to_money ratio，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_half_exit_leg_return_pct_uses_allocated_cost_contract_case` |
| 2026-04-11 | T223 | 將 debug half-exit leg-return static contract 同步到 integer-total ratio 後重新驗證 | PARTIAL -> DONE | `validate_debug_half_exit_leg_return_pct_uses_allocated_cost_contract_case` |
| 2026-04-11 | T224 | 檢出 debug sell-signal profit-pct static contract 仍比對舊 helper / helper-call signature，未同步 `signal_date` keyword，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_sell_signal_profit_pct_uses_exact_mark_to_market_contract_case` |
| 2026-04-11 | T224 | 將 debug sell-signal profit-pct static contract 同步到 `signal_date` helper / helper-call signature 後重新驗證 | PARTIAL -> DONE | `validate_debug_sell_signal_profit_pct_uses_exact_mark_to_market_contract_case` |
| 2026-04-11 | T227 | 檢出投組 rotation exact mark-to-market return static contract 仍比對舊 helper-call signature，未同步 `trade_date=today`，改回 PARTIAL | DONE -> PARTIAL | `validate_portfolio_rotation_mark_to_market_return_contract_case` |
| 2026-04-11 | T227 | 將投組 rotation exact mark-to-market return static contract 同步到 `trade_date=today` helper-call signature 後重新驗證 | PARTIAL -> DONE | `validate_portfolio_rotation_mark_to_market_return_contract_case` |
| 2026-04-11 | T228 | 檢出 validator exact-ledger oracle static contract 尚未覆蓋 scanner/reference projected-qty 的 `ticker` / `trade_date` 傳遞，改回 PARTIAL | DONE -> PARTIAL | `validate_validator_oracles_use_exact_ledger_totals_contract_case` |
| 2026-04-11 | T228 | 擴充 validator exact-ledger oracle static contract 納入 scanner/reference projected-qty 的商品別與日期上下文後重新驗證 | PARTIAL -> DONE | `validate_validator_oracles_use_exact_ledger_totals_contract_case` |
| 2026-04-11 | T232 | 檢出 array tick-normalization static contract 尚未覆蓋 portfolio rotation 賣出路徑的未宣告 `ticker` 引用，改回 PARTIAL | DONE -> PARTIAL | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 檢出 array tick-normalization static contract 尚未覆蓋 ticker / security_profile 傳遞路徑，改回 PARTIAL | DONE -> PARTIAL | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 擴充 array tick-normalization static contract 納入 portfolio rotation 賣出路徑 ticker 來源契約後重新驗證 | PARTIAL -> DONE | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 擴充 array tick-normalization static contract 納入商品 profile 傳遞與 stock-only ladder 禁止案例後重新驗證 | PARTIAL -> DONE | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 檢出 candidate-plan resize / scanner projected qty 的 caller-threading contract 缺口，改回 PARTIAL | DONE -> PARTIAL | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 補齊 candidate-plan resize / scanner projected qty 的 caller-threading static contract 後重新收斂為 DONE | PARTIAL -> DONE | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 檢出 array tick-normalization static contract 尚未覆蓋 prepared scanner frame 最新交易日 attrs / MultiIndex fallback，改回 PARTIAL | DONE -> PARTIAL | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 擴充 array tick-normalization static contract 納入 prepared scanner frame 最新交易日 attrs / MultiIndex fallback 後重新收斂為 DONE | PARTIAL -> DONE | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 檢出 array tick-normalization static contract 尚未覆蓋 prepared scanner frame 最新交易日的 index fallback，改回 PARTIAL | DONE -> PARTIAL | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 擴充 array tick-normalization static contract 納入 prepared scanner frame 最新交易日欄位 / index 雙來源解析後重新收斂為 DONE | PARTIAL -> DONE | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 檢出 array/scanner static contract 尚未釘死 validator/oracle 重用共享最新交易日 helper，改回 PARTIAL | DONE -> PARTIAL | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 擴充 array/scanner static contract 納入最新交易日共享 helper 後重新收斂為 DONE | PARTIAL -> DONE | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T233 | 新增 test_suite summary comment coverage static contract 並驗證 | NEW -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T233 | 檢出 summary comment coverage contract 仍只覆蓋到 T233，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T233 | 擴充 summary comment coverage contract 納入 T234/T235 並重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T233 | 檢出 summary comment coverage contract 尚未同步新增的 T236，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T233 | 擴充 summary comment coverage contract 納入 T236 並重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T233 | 檢出 summary comment coverage contract 尚未明確釘死 T235，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T233 | 擴充 summary comment coverage contract 納入 T235 明示檢查並重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T233 | 檢出 summary comment coverage contract 仍掃描整份 source_text、未限縮到頂部摘要註解 block，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T233 | 將 summary comment coverage contract 收斂到頂部摘要註解 block 後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T234 | 新增 exact-ledger return ratio no-money-float-division static contract 並驗證 | NEW -> DONE | `validate_exact_ledger_return_ratio_no_money_float_division_contract_case` |
| 2026-04-11 | T235 | 新增 debug exit total-return milli binding static contract 並驗證 | NEW -> DONE | `validate_debug_exit_total_return_milli_binding_contract_case` |
| 2026-04-11 | T235 | 檢出 milli-binding contract 尚未覆蓋 forced-closeout 路徑，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_exit_total_return_milli_binding_contract_case` |
| 2026-04-11 | T235 | 擴充 milli-binding contract 納入 forced-closeout 路徑後重新驗證 | PARTIAL -> DONE | `validate_debug_exit_total_return_milli_binding_contract_case` |
| 2026-04-11 | T235 | 檢出 forced-closeout milli-binding contract 的舊 float total-pnl 負向守衛仍比對錯誤字串，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_exit_total_return_milli_binding_contract_case` |
| 2026-04-11 | T235 | 補齊 forced-closeout 舊 float total-pnl 負向守衛後重新驗證 | PARTIAL -> DONE | `validate_debug_exit_total_return_milli_binding_contract_case` |
| 2026-04-11 | T236 | 新增 core R-multiple exact-ledger static contract 並驗證 | NEW -> DONE | `validate_core_r_multiple_exact_ledger_contract_case` |

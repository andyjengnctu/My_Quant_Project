# Test Suite 收斂清單

用途：正式 test suite 維護清單；主表為唯一真理來源。

文件分工：`TEST_SUITE_CHECKLIST.md` 只管主表、狀態、測試入口與收斂索引。

範圍：納入長期規則與必要 formal contract；不納入暫時特例：`apps/portfolio_sim.py` 自動開瀏覽器、只使用還原價不考慮 raw。

狀態：`DONE` 已覆蓋；`PARTIAL` 仍有缺口；`TODO` 待補；`N/A` 不納入正式長期 test suite。

優先級：`P0` 交易正確性/統計口徑/未來函數；`P1` 高價值補強；`P2` 品質與工具鏈。

索引：`Bxx` 主表 ID；`Txx` 建議測試 ID。

維護規則：
1. 同步順序固定為主表 → `T` / `G` → `E`。
2. `T` 只留最小索引；每列一個 `Txx` 與一個測試入口，依 ID 升冪排序。
3. `G` 只記錄實際狀態變更；依日期升冪、同日再依 tracking ID 排序；`NEW -> *` 只能出現在首筆，且不得出現 no-op transition。
4. `G` 備註欄只作最小必要的收斂索引與人工說明；formal blocker 不檢查其文字 hygiene，日期只記於 `G`。
5. 其餘文字只保留最小必要；會隨實作變動的細節由主表、formal contract 與收斂紀錄承接。

## A. 分層原則

- A1. 長期固定測試：驗 invariant、契約與品質基線。
- A2. 可調整測試：驗 schema、可重現性與最低可用性，不綁死策略結果。

## B. 長期固定測試清單

### B1. 長期固定核心規則（不含暫時特例）

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

### B2. 長期固定補充契約

| ID | 優先級 | 類別 | 項目 | 目前判定 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|---|---|
| B11 | P1 | 契約 | 跨工具 schema / 欄位語意一致 | DONE | 已補 missed sell / trade log / stats 一致性，以及 validate / issue report / optimizer profile / local regression summaries（preflight / dataset prepare / chain / ml smoke / meta quality / master summary）的 CSV / XLSX / JSON contract；另已釘死 early-failure `master_summary.json` 的 `payload_failures` 必須維持與正常路徑一致的語意，且不得把合法 FAIL payload 誤標成 `summary_unreadable`；並補 `meta_quality_summary.json` 的 `formal_entry` nested schema contract，要求 `registry_steps / registry_commands / run_all_steps / preflight_steps / test_suite_steps` 完整存在，且 stale-key 排除檢查必須直接禁止已退役的 legacy `steps` 舊鍵 | `tools/validate/synthetic_contract_cases.py` |
| B12 | P1 | 決定性 | 同資料、同參數、同 seed 結果可重現 | DONE | 已補 `run_ml_smoke.py` fixed-seed 雙跑、`run_chain_checks.py` scanner reduced snapshot 雙跑 digest、`validate_scanner_worker_repeatability_case` 與 `validate_scan_runner_repeatability_case`，正式入口與 scanner 入口重跑一致性已收斂 | `tools/local_regression/`, `tools/validate/synthetic_regression_cases.py` |
| B13 | P1 | 邊界值 | 數值穩定性、rounding、tick、odd lot | DONE | 已新增 `price_utils` / `history_filters` / `portfolio_stats` unit-like 邊界案例，覆蓋 tick、稅費、sizing、全贏/全輸與空序列 | `tools/validate/synthetic_unit_cases.py` |
| B14 | P1 | 韌性 | 髒資料、缺欄位、NaN、日期亂序、OHLC 異常 | DONE | 已新增資料清洗 expected behavior / fail-fast / `load_clean_df` 整合案例，直接釘死髒資料修正、欄位缺失、NaN、日期亂序、OHLC 異常與清洗後列數行為 | `tools/validate/synthetic_data_quality_cases.py`, `core/data_utils.py`, `tools/validate/real_case_io.py` |
| B15 | P1 | 錯誤處理 | 壞 JSON、缺參數、缺檔、匯入失敗、API 失敗時訊息可定位 | DONE | 已補 `params_io` / `module_loader` / `preflight_env` 的 module 級錯誤路徑，並補 downloader universe fetch 全失敗與 screening 初始化失敗的 fatal error path，錯誤訊息與 issue log 已可定位 | `core/params_io.py`, `tools/validate/preflight_env.py`, `tools/validate/module_loader.py`, `tools/validate/synthetic_error_cases.py` |
| B16 | P2 | CLI | 互斥參數、預設值、help 與 shipped 指令文件一致 | DONE | 已補 dataset wrapper、local regression / no-arg CLI 與剩餘直接入口 CLI 契約，覆蓋 help、預設 passthrough、`--only` / `--steps` 正規化、未知參數、缺值、空值、位置參數拒絕；`apps/workbench.py` 也已納入 no-arg CLI formal 邊界，不再只驗 help。另 `run_all.py` 參數錯誤 stderr usage 必須同步列出 `meta_quality`；`doc/CMD.md` 的 shipped Python 指令與主要參數也併入同一 formal 邊界 | `tools/validate/synthetic_cli_cases.py`, `tools/validate/synthetic_meta_cases.py`, `apps/*.py`, `core/runtime_utils.py`, `doc/CMD.md` |
| B17 | P2 | I/O | 輸出工件、bundle、retention、rerun 覆寫行為 | DONE | 已補 validate summary / optimizer profile / issue report 的 output contract，以及 bundle/archive/root-copy/retention lifecycle、PASS/FAIL bundle selection、artifacts manifest、rerun 覆寫內容契約；另補 quick gate 發生 runtime error 時仍必須落出正式 FAIL summary / console artifact 契約 | `core/output_paths.py`, `core/output_retention.py`, `tools/validate/synthetic_contract_cases.py`, `tools/local_regression/run_quick_gate.py` |
| B18 | P1 | 回歸 | 重跑一致性、狀態汙染、cache 汙染 | DONE | 已補 `run_chain_checks.py` 雙跑 digest、`run_ml_smoke.py` fixed-seed 雙跑、`validate_optimizer_raw_cache_rerun_consistency_case` 與 `validate_run_all_repeatability_case`，正式入口與 cache 汙染隔離已收斂 | `tools/local_regression/`, `tools/optimizer/raw_cache.py`, `tools/validate/synthetic_regression_cases.py` |
| B19 | P2 | 效能 | reduced dataset 時間基線、optimizer 每 trial 上限、記憶體回歸 | DONE | 已將 quick gate / consistency / chain checks / ml smoke / meta quality / total suite duration、optimizer 平均 trial wall time 與各步驟 / meta quality traced peak memory 全數納入 `run_meta_quality.py` 正式 gating，並同步回寫 step summaries / `meta_quality_summary.json` / `apps/test_suite.py` 摘要；step summary 的 duration 欄位已補齊，consistency / chain / total budget 依實測 reduced baseline 校正；另已把 chain checks 改為同輪共用 prepared market context、prepared scanner snapshot、cached single-backtest stats，並將 replay counts 直接併入第一次 timeline 主流程；`portfolio_sim` 驗證也改直接共用 `validate_one_ticker()` 已產生的 prepared df / standalone logs，不再重讀 CSV、重新 sanitize / prepare，且單檔 portfolio context 的 `fast_data / sorted_dates / start_year` 改為共用，不再在 real-case 與 tool check 各自重建；`validate_consistency` 執行 synthetic suite 時已同步寫出 coverage artifacts，`run_meta_quality.py` 可直接重用同輪 artifacts，不再重跑一次 synthetic coverage suite | `tools/local_regression/`, `core/runtime_utils.py`, `apps/test_suite.py` |
| B20 | P2 | 文件 | `doc/CMD.md` 指令與實作一致（已併入 B16） | N/A | 已併入 B16 的 CLI / help / `doc/CMD.md` shipped 指令契約；保留 ID 只作歷史索引，不再作獨立 formal blocker | `validate_cmd_document_contract_case`, `doc/CMD.md` |
| B21 | P2 | 顯示 | 報表欄位、排序、百分比格式與來源一致 | DONE | 已補 scanner header / start banner / summary、strategy dashboard、validate console summary、issue Excel report schema、portfolio yearly/export report，以及 `apps/test_suite.py` 在 PASS / FAIL / manifest-blocked / partial-selected-steps / preflight-failed / dataset-prepare-failed / summary-unreadable 下的人類可讀摘要契約；另補 score header 顯示契約，釘死 `評分模型` 與 `評分分子` 必須分欄顯示、不得再以 `/ 分子` 混入同一括號；並補 checklist status vocabulary sync 與 meta quality coverage line/branch/min-threshold/missing-zero-target guard 摘要顯示，且以 `run_all.py` contract 釘死 preflight 早退時 dataset_prepare 仍需標記為 `NOT_RUN`，避免 real path 誤落成 `missing_summary_file`；另補 portfolio export 的 Plotly optional dependency fallback 契約，要求主要 Excel artifact 仍須成功匯出、HTML 可略過但必須輸出可追蹤錯誤摘要，且 `except (...)` tuple 不得引用未匯入模組 | `tools/validate/synthetic_display_cases.py`, `tools/validate/synthetic_reporting_cases.py`, `tools/validate/synthetic_contract_cases.py`, `core/display.py`, `tools/scanner/reporting.py`, `tools/portfolio_sim/reporting.py`, `apps/test_suite.py`, `tools/local_regression/run_all.py` |
| B22 | P2 | 覆蓋率 | line / branch coverage 報表與核心覆蓋門檻 | DONE | 已將 `run_meta_quality.py` 的 synthetic coverage suite、formal helper probe、manifest 化 line / branch minimum threshold gate、key coverage targets completeness，以及 critical files per-file line / branch minimum gate 收斂為同一正式 coverage 路徑，並同步回寫 `meta_quality_summary.json` / `apps/test_suite.py` 摘要顯示 | `tools/local_regression/run_meta_quality.py`, `tools/local_regression/common.py`, `apps/test_suite.py` |
| B23 | P1 | Meta | checklist / 測試註冊 / 正式入口一致性 | DONE | 已補 synthetic 主入口遺漏註冊案例，並新增 imported / defined `validate_*` case、formal pipeline registry / formal-entry / run_all / preflight / test_suite 一致性 formal guard；`T` 摘要只要求每列維持單一 shipped formal entry，formal step 單一真理來源維持在 `tools/local_regression/formal_pipeline.py`，不再把 command string 的逐字記錄形式上升為獨立 release blocker；另補 `core/` / `tools/` 不得反向 import `apps/` 的分層 guard | `tools/validate/synthetic_meta_cases.py`, `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_cases.py`, `tools/local_regression/formal_pipeline.py` |
| B24 | P1 | Meta | known-bad fault injection：關鍵規則故意破壞後測試必須 fail | DONE | 已新增 meta fault-injection case，直接對 same-day sell、same-bar stop priority、fee/tax、history filter misuse 注入 known-bad 行為，並驗證既有測試會產生 FAIL | `tools/validate/synthetic_meta_cases.py` |
| B25 | P1 | Meta | independent oracle / golden cases：高風險數值規則不可只與 production 共用同邏輯 | DONE | 已新增獨立 oracle golden case，對 net sell、position size、history EV、annual return / sim years 以手算或獨立公式對照 production | `tools/validate/synthetic_unit_cases.py` |
| B26 | P1 | Meta | checklist 是否已足夠覆蓋完整性（包含 test suite 本身） | DONE | 已補主表 / `T` / `G` 收斂紀錄完整同步 formal guard；正式 blocker 只保留 checklist 結構可解析、主表 / `E` / `T` 摘要同步、`T` 每列單一 shipped formal entry，以及 `G` 合法狀態轉移 / 首次 `NEW` / 狀態鏈連續 / 非 no-op / 日期與 tracking ID 排序；`G` 備註欄只作治理索引與人工說明，不再承擔 release-blocking formal contract | `tools/local_regression/run_meta_quality.py`, `tools/validate/meta_contracts.py`, `doc/TEST_SUITE_CHECKLIST.md` |
| B27 | P1 | Meta | 禁止循環依賴（模組層級 import cycle） | DONE | 已補 project import graph cycle guard，直接阻擋 `apps/` / `core/` / `tools/` 間的模組層級循環依賴（含函式內 import） | `tools/validate/synthetic_meta_cases.py`, `tools/validate/meta_contracts.py` |
| B28 | P1 | 覆蓋率 | key coverage targets 應包含核心交易模組（已併入 B22） | N/A | 已併入 B22 的正式 coverage 邊界；保留 ID 只作歷史索引，不再作獨立 formal blocker | `validate_core_trading_modules_in_coverage_targets_case`, `tools/local_regression/run_meta_quality.py` |
| B29 | P1 | 覆蓋率 | critical files 應具備 per-file line / branch minimum gate（已併入 B22） | N/A | 已併入 B22 的正式 coverage 邊界；保留 ID 只作歷史索引，不再作獨立 formal blocker | `validate_critical_file_coverage_minimum_gate_case`, `tools/local_regression/run_meta_quality.py` |
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
| B57 | P1 | 契約 | 系統評分分子必須可在總報酬率與年化報酬率之間切換，且 score 公式不得把分子 / 分母切換混成不同 mode | DONE | 已補 direct synthetic contract，直接釘死 `SCORE_NUMERATOR_METHOD` 可在 `ANNUAL_RETURN` / `TOTAL_RETURN` 間切換，兩者都必須共用同一個 `abs(MDD) + 0.0001` 分母，且 `LOG_R2` 品質單調性不得被新分子選項破壞 | `tools/validate/synthetic_strategy_cases.py`, `config/training_policy.py`, `core/portfolio_stats.py` |
| B58 | P1 | 契約 | 保留中的單一模式參數不得接受不支援值後靜默忽略；目前 `use_compounding` 必須 fail-fast 拒絕 `False` | DONE | 已補 direct guardrail case，直接釘死 `use_compounding=False` 無論 JSON 載入、dataclass 建立或 setattr 都必須報錯，避免保留參數變成靜默 no-op | `tools/validate/synthetic_guardrail_cases.py`, `core/strategy_params.py` |
| B59 | P1 | Meta | 關鍵費稅 / sizing / 資金來源 helper 必須維持單一真理來源，不得在其他模組重複定義 | DONE | 已補 static contract，直接釘死 `calc_entry_price`、`calc_net_sell_price`、`calc_position_size`、`calc_initial_risk_total`、`resolve_single_backtest_sizing_capital`、`resolve_portfolio_sizing_equity`、`resolve_portfolio_entry_budget`、`resolve_scanner_live_capital` 只能定義在各自 canonical 模組，避免費稅 / sizing / 資金來源規則分叉 | `tools/validate/meta_contracts.py`, `tools/validate/synthetic_meta_cases.py`, `core/price_utils.py`, `core/capital_policy.py` |
| B61 | P1 | Meta | 關鍵資金 / 參數 / 政策契約模組必須納入 coverage targets，避免 helper / guardrail 雖有測項但 coverage 退化時無法被 meta quality 擋下 | DONE | 已補 static coverage-target contract，直接釘死 `core/capital_policy.py`、`core/strategy_params.py`、`core/params_io.py`、`config/execution_policy.py`、`config/training_policy.py` 必須列入 coverage targets，且關鍵公開符號需可匯入 | `tools/local_regression/meta_quality_targets.py`, `tools/validate/synthetic_meta_cases.py` |
| B62 | P1 | Meta | checklist 首行固定標題 exact-string guard（已退役；不再作獨立 formal blocker） | N/A | 已降級為非必要 wording hygiene；保留 ID 只作歷史索引，不再要求 `doc/TEST_SUITE_CHECKLIST.md` 的第一個非空行必須逐字固定 | `doc/PROJECT_SETTINGS.md` |
| B63 | P1 | Meta | synthetic meta validator flattened-payload 讀取 hygiene（已退役；不再作 formal blocker） | N/A | 屬 validator 內部 payload 讀法與相容細節；保留 ID 只作歷史索引，正式長期 contract 不再要求 synthetic meta validators 的欄位讀取方式 | `doc/PROJECT_SETTINGS.md` |
| B64 | P1 | Meta | checklist 摘要表排序獨立主表契約（已併入 B26） | N/A | 已併入 B26 的 checklist 機械排序邊界；保留 ID 只作歷史索引，不再作獨立 formal blocker | `validate_checklist_summary_tables_sorted_by_id_case`, `tools/local_regression/run_meta_quality.py` |
| B65 | P1 | CLI | `apps/package_zip.py` 必須支援外部參數一鍵執行 commit → zip → test_suite，且 ZIP 檔名必須反映 commit 後 HEAD | DONE | 已補 orchestration contract，直接釘死 `--commit-message` 會先 `git add -A` + `git commit -m ...`、zip 產物必須使用 commit 後 HEAD short sha，且 `--run-test-suite` 必須在打包後才執行 `apps/test_suite.py` | `apps/package_zip.py`, `tools/validate/synthetic_cli_cases.py`, `tools/validate/synthetic_cases.py` |
| B66 | P1 | GUI | `apps/workbench.py` 必須作為單一 GUI 啟用入口，且 workbench panel registry / 單股回測後端 / Excel artifact 與 inline chart payload 契約必須穩定 | DONE | 已補 GUI workbench contract，直接釘死 `apps/workbench.py` 必須對應到 `tools.workbench_ui.main`、workbench 必須註冊單股回測檢視 panel，且 GUI 單股後端必須產生 `Debug_TradeLog_<ticker>.xlsx` 並回傳 inline `chart_payload`，不得再把 HTML artifact 當成 GUI 必要輸出；workbench panel registry 與 inspector 內部整合必須優先使用 canonical `run_ticker_analysis` / `resolve_trade_analysis_data_dir` / `create_matplotlib_trade_chart_figure` aliases，不得再把 legacy `run_debug_ticker_analysis` / `resolve_debug_data_dir` / `create_matplotlib_debug_chart_figure` 當成 workbench 內部正式介面 | `apps/workbench.py`, `tools/workbench_ui/workbench.py`, `tools/workbench_ui/single_stock_inspector.py`, `tools/trade_analysis/trade_log.py`, `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B67 | P1 | GUI | 單股 debug chart hooks 在 `export_chart=False` 時必須完全 no-op；`chart_context=None` 不得造成 runtime side effect 或中斷單股 / chain / synthetic suite | DONE | 已補 direct contract，直接釘死 `run_debug_backtest()` 走無圖模式時仍必須正常產生交易明細；chart marker / active level hooks 必須接受 `chart_context=None` | `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B68 | P1 | 契約 | validation / tool-check payload 的 `module_path` 必須統一為 repo-relative、forward-slash 穩定路徑；不得回傳機器相依絕對路徑 | DONE | 已補 direct contract 與 shared normalizer，直接釘死 absolute / backslash path 都必須正規化為穩定 repo-relative path，避免 bundle 與 synthetic 在不同 OS/工作目錄出現假失敗 | `tools/validate/module_loader.py`, `tools/validate/synthetic_contract_cases.py` |
| B69 | P1 | 契約 | shared path normalizer 必須接受 `pathlib.Path` 等 path-like 輸入；不得假設 `PROJECT_ROOT` 或 path payload 一定是 str 才能正規化 | DONE | 已補 direct contract，直接釘死 `normalize_project_relative_path()` 對 `Path` 輸入也必須穩定回傳 repo-relative、forward-slash 路徑，避免 synthetic / local regression helper 再因字串 API 假設發生 runtime regression | `tools/validate/module_loader.py`, `tools/validate/synthetic_contract_cases.py` |
| B70 | P1 | 契約 | shared module-loader path helpers 必須接受被 patch 成字串的 `PROJECT_ROOT`；不得假設 module-level root 永遠是 `Path`，否則 synthetic / error-path 測試環境會產生 helper 自身回歸 | DONE | 已補 direct contract，直接釘死 `build_project_absolute_path()` 與 `normalize_project_relative_path()` 在 `PROJECT_ROOT` 被 patch 成字串時仍必須正常組路徑並回傳穩定 repo-relative path，避免 shared helper 與既有 `patch.object(module_loader, "PROJECT_ROOT", str(...))` 測試環境互撞 | `tools/validate/module_loader.py`, `tools/validate/synthetic_contract_cases.py` |
| B71 | P1 | GUI | GUI 單股回測檢視必須內嵌大型 K 線圖，且初始視窗與縮放後 Y 軸比例都必須依可視 X 範圍自動重算；不得因全歷史資料或離屏極值導致圖形失真 | DONE | 已補 direct contract，直接釘死 workbench panel 必須宣告 inline chart backend 並使用 canonical `create_matplotlib_trade_chart_figure` alias，trade-analysis backend 必須回傳 chart payload，且 chart helper 的預設視窗與可視區間價量範圍計算必須忽略離屏極值並可建立 2 軸內嵌 figure | `tools/trade_analysis/charting.py`, `tools/workbench_ui/single_stock_inspector.py`, `tools/validate/synthetic_contract_cases.py` |
| B72 | P2 | Meta | synthetic validator 外部 alias import hygiene（已退役；不再作 formal blocker） | N/A | 屬 validator 自身 import hygiene；保留 ID 只作歷史索引，不再將 synthetic validator 的 alias import 細節上升為 formal blocker | `doc/PROJECT_SETTINGS.md` |
| B73 | P2 | Meta | synthetic validator alias 掃描 AST implementation hygiene（已退役；不再作 formal blocker） | N/A | 屬 validator 自身掃描實作細節；保留 ID 只作歷史索引，不再將 alias 掃描的 AST / 字串判定策略上升為 formal blocker | `doc/PROJECT_SETTINGS.md` |
| B74 | P1 | GUI | GUI 單股回測工作台必須以 K 線圖分頁作為主檢視；成交量預設隱藏，切換後須以前景疊加方式共用同一圖面且高度低於 1/4；僅保留交易明細與 Console 分頁，不得再保留執行摘要分頁；GUI 開啟時預設最大化，且內嵌 K 線必須保留完整歷史可平移/縮放，不得因 GUI render slicing 截斷左右歷史資料 | DONE | 已補 direct contract，直接釘死 GUI panel 必須使用 notebook 分頁承接 K 線圖/交易明細/Console、不得再暴露執行摘要分頁、成交量 toggle 預設關閉、workbench 預設 maximized，且 chart helper 必須宣告 full-history navigation 與 volume overlay ratio ≤ 1/4；overlay axis 存在性改以 shared chart contract 驗證，不得綁死 `figure.axes` 長度，避免 matplotlib backend / inset axes 註冊差異造成假失敗 | `tools/workbench_ui/single_stock_inspector.py`, `tools/workbench_ui/workbench.py`, `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B75 | P1 | GUI | GUI 單股回測內嵌 K 線圖必須支援 toolbar-free 滑鼠互動：滾輪直接縮放、左鍵直接拖曳平移時間軸，且重新渲染成交量時不得殘留已銷毀 toolbar/widget 造成 Tk runtime error | DONE | 已補 direct contract，直接釘死 GUI panel 不得再依賴 `NavigationToolbar2Tk`、必須綁定 shared mouse navigation binder，並要求 chart helper 宣告 wheel-zoom / left-drag-pan / no-toolbar contract，避免縮放平移體驗分叉與 volume toggle 時殘留失效 widget | `tools/workbench_ui/single_stock_inspector.py`, `tools/trade_analysis/charting.py`, `tools/validate/synthetic_contract_cases.py` |
| B76 | P2 | Meta | synthetic validator chart/navigation helper import hygiene（已退役；不再作 formal blocker） | N/A | 屬 validator 自身 import hygiene；保留 ID 只作歷史索引，不再將 synthetic validator 的 helper import 細節上升為 formal blocker | `doc/PROJECT_SETTINGS.md` |
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
| B89 | P1 | GUI | GUI 單股工作台必須以右側獨立 sidebar 呈現買入訊號 / 符合歷史績效 / 單股歷史績效表 / 選取日線值 / 回到最新 K 線，並支援 Enter 直接回測、候選股選取即回測、最新 K 線後右側留白約 1/6 版面，以及 latest-bar 買訊的隔日預掛線預覽不得畫在訊號當日 | DONE | 已補 direct contract，直接釘死 panel 不得保留執行回測按鈕、ticker entry 必須綁定 Enter、candidate select 必須直接觸發回測；chart helper 必須宣告 future_preview 與動態 right padding，且最新 K 線後的右側留白需壓到約 1/6 版面，backtest latest signal preview 必須走 next-day future preview path | `tools/workbench_ui/single_stock_inspector.py`, `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/validate/synthetic_contract_cases.py` |
| B94 | P1 | GUI | GUI 單股回測檢視必須使用純黑 K 線底色、固定文案的右側狀態晶片、右側 OHLCV / 線值單一資訊來源、移除 chart hint footer 以擴大主圖高度，且買訊預掛線必須自訊號次日開始預覽；賣出/停損資訊框需包含最大回撤 | DONE | 已補 refined visual contract，直接釘死 pure-black chart background、固定「出現買入訊號」/「符合歷史績效」晶片文案、右側 sidebar 必須承接 OHLCV 與線值、footer 不得再保留 chart hint 空間、賣出框需列出最大回撤，且 entry preview lines 必須在次日預先畫出 | `tools/workbench_ui/single_stock_inspector.py`, `tools/workbench_ui/workbench.py`, `tools/trade_analysis/charting.py`, `tools/trade_analysis/entry_flow.py`, `tools/validate/synthetic_contract_cases.py` |
| B95 | P1 | GUI | GUI 單股回測狀態晶片在 runtime 必須保持固定文案、僅以底色切換狀態；延續候選預掛線必須依候選資格跨多日連續顯示；未建立隔日 shadow 前只預覽限價，shadow 建立後停損 / 停利須沿用隔日反事實狀態 | DONE | 已補 direct contract，直接釘死右側「出現買入訊號」/「符合歷史績效」晶片不得被 runtime 文案覆寫，且 entry preview 必須走 candidate layer；即使 entry plan 因不可掛單而為 `None`，延續候選有效期間仍須連續顯示 limit 預掛線；shadow 建立後停損 / 停利不得以後續成交日重算；延續候選 preview 的 counterfactual builder 呼叫也必須同步傳遞 `ticker`、`security_profile` 與 `trade_date`，避免 contract 仍比對舊 signature | `tools/workbench_ui/single_stock_inspector.py`, `tools/trade_analysis/entry_flow.py`, `tools/validate/synthetic_contract_cases.py` |
| B96 | P1 | GUI | GUI 單股回測圖例必須貼近 K 線圖分頁上緣但仍保留小幅左上 inset gap、價格軸數字不得因邊界裁切而缺字，且主圖左/下邊界需盡量收小以保留更多 K 線可視區；延續候選若一路持續到最新一根，最新實際 K 棒與 next-day future preview 都必須維持隔日錨定限價一致顯示；shadow 建立前 stop / tp 缺席，建立後使用 shadow 狀態 | DONE | 已補 direct contract，直接釘死 matplotlib 左/下 margin 收小、legend 需保留小幅 top-left inset gap 且不得裁切價格軸；並補 latest extended candidate preview contract，釘死延續候選走到最新一根時，最後一根實際 K 棒與 next-day future preview 都必須保留隔日錨定 limit 預掛線；shadow 建立後 stop / tp 改由 shadow 狀態供應 | `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/validate/synthetic_contract_cases.py` |
| B97 | P1 | GUI | GUI 單股 latest raw-signal 買訊預覽路徑必須可直接執行；`tools/trade_analysis/backtest.py` 若引用 `build_normal_entry_plan` / `build_normal_candidate_plan`，必須同步自 `core.entry_plans` 明確 import，避免 GUI/coverage runtime 因未定義 helper 而在單股圖表路徑 NameError | DONE | 已補 direct contract，直接以 AST 掃描 `tools/trade_analysis/backtest.py`，釘死 latest raw-signal 預覽使用的 `build_normal_entry_plan` 與 tail preview 使用的 `build_normal_candidate_plan` 都必須被引用且已自 `core.entry_plans` 匯入，避免再次只改呼叫點卻漏同步 import | `tools/trade_analysis/backtest.py`, `tools/validate/synthetic_contract_cases.py` |
| B98 | P0 | 交易規格 | 延續候選不得再以 signal day frozen `L/S/T` 模型存續；所有延續追蹤必須在首個待掛單日用固定反事實 `P' = min(Open, L)` 建立不隨日漂移的 shadow entry / 失效 / 達標 barrier，但後續實際預掛 `limit_price` 仍必須沿用正常買訊隔日的原始掛單限價 `L`，不得把 shadow `entry_fill_price` 誤當限價；即使該日 `Low > L` 也只代表價格因素未成交；`signal_valid` 與 `today_orderable` 必須分層，固定掛單價若低於今日跌停價不得進 `orderable_candidates_today` | DONE | 已補 direct synthetic case，直接釘死延續候選不得再以 signal day frozen `L/S/T` 或每日漂移 `reference_price` 模型存續；首個待掛單日一律建立固定反事實 `P'` barrier，若該日 `Low > L` 則 `P'` 固定為 `L`，且延續候選後續 order limit 必須保持原始 `L`、shadow `P'` 只作管理狀態基準；valid signal 即使當日價格帶不可達也只能留在 candidate layer，不得擠進 orderable list | `tools/validate/synthetic_flow_cases.py` |
| B99 | P1 | Meta | synthetic validator 非 error-path 參數字面值 hygiene（已退役；不再作 formal blocker） | N/A | 屬測試資料與 validator 寫法 hygiene；保留 ID 只作歷史索引，不再將 synthetic validator 自身常值選擇上升為 formal blocker | `doc/PROJECT_SETTINGS.md` |
| B100 | P1 | Meta | synthetic_meta_cases shared path helper import hygiene（已退役；不再作 formal blocker） | N/A | 屬 validator 自身 import hygiene；保留 ID 只作歷史索引，不再將 `synthetic_meta_cases.py` 的 helper import 細節上升為 formal blocker | `doc/PROJECT_SETTINGS.md` |
| B101 | P1 | GUI | GUI 單股回測賣訊註記必須僅表達 signal day 資訊，不得冒充已成交賣出；future preview 的限價線若超出當前 K 棒高低範圍仍必須納入可視價格範圍；期末強制結算圖示需以黃色區分 | DONE | 已補 direct contract，直接釘死賣訊框只能顯示訊號日收盤與未成交提示，不得再混入股數 / 金額 / 損益；並以 chart range case 驗證 future preview 的限價線會參與 autoscale，另釘死期末強制結算 marker color 為黃色 | `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/validate/synthetic_contract_cases.py` |
| B102 | P1 | GUI | GUI 單股回測賣訊註記必須以賣訊 schema 呈現訊號日資訊，指標賣出 marker 必須為綠色橫線，停損 / 指標賣出 / 期末結算資訊框需顯示交易次數，且買訊框中的限價價格即使未畫出 future preview 線也必須納入可視價格範圍；停利線需以黃色顯示 | DONE | 已補 direct contract，直接釘死賣訊註記由 shared schema 顯示資金 / 股數 / 參考收，不得混入成交金額 / 損益 / 報酬率等已成交欄位；指標賣出 marker 改為綠色水平線，停損 / 指標賣出資訊框加入交易次數，並以 signal annotation meta 驗證買訊框內的限價價格會參與 autoscale；另釘死停利線顏色為黃色 | `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/trade_analysis/exit_flow.py`, `tools/validate/synthetic_contract_cases.py` |
| B103 | P1 | GUI | GUI 買入資訊框應貼近買進 K 線位置，單股與投組 K 線圖例須固定完整列出；買訊、買進、賣訊、停利、停損、指標賣出、期末結算資訊框必須由同一 schema 產生；停損 / 指標賣出 / 期末結算需列出最大回撤，且停損資訊框不得同時列出損益與總損益 | DONE | 已補 direct contract，直接釘死買入資訊框應留在 K 線 overlay 並貼近對應 K 線下方、不得搬到右側 sidebar；若同一 K 線或相鄰少數 K 線已有其他下方資訊框，必須以水平錯位與額外垂直分層避免重疊。另交易資訊框欄位順序改由 chart schema 控制：買訊為資金 / 股數 / 限價 / 預留，買進為資金 / 股數 / 停利 / 限價 / 成交 / 停損 / 實支 / 進場類型 / 結果，停利為資金 / 股數 / 成交 / 金額 / 損益 / 報酬率，停損與指標賣出為資金 / 股數 / 成交 / 金額 / 損益 / 報酬率 / 勝率 / 最大回撤 / 交易次數 / 結果，期末結算同出場欄位但不顯示結果；停損資訊框只保留該次交易損益，不另列總損益，避免圖面語意分叉 | `tools/workbench_ui/single_stock_inspector.py`, `tools/trade_analysis/charting.py`, `tools/trade_analysis/backtest.py`, `tools/trade_analysis/exit_flow.py`, `tools/validate/synthetic_contract_cases.py` |
| B104 | P0 | 交易規格 | 細部交易契約必須由 checklist / formal contract 明確承接：`L` 只作進場上限 / 最壞風險 sizing 上界，首個可執行 stop / tp 只能於 `t+1` 成交後以 `P_fill + ATR_t` 在 `t+1` 收盤後 frozen，`t+1` 觸發與 `t+2` 執行必須分離，且長倉 hit 採 `Low <= line` / `High >= line` | DONE | 已補 meta contract，直接釘死上述 `L` 角色、`P_fill + ATR_t` 首個可執行 stop / tp、固定反事實 barrier 與 inclusive hit 語意必須由 checklist 與 formal contract 承接，避免再次回退到舊 `L / init_sl / T` frozen 口徑 | `doc/TEST_SUITE_CHECKLIST.md`, `tools/validate/synthetic_meta_cases.py` |
| B105 | P0 | 交易規格 | runtime 必須分離盤前 sizing 風險與成交後首個可執行 stop / tp：candidate plan 仍可用 `L` 推導最壞情境 sizing，但 filled position 的 `initial_stop` / `tp_half` / entry-day pending stop-tp trigger 必須改以實際成交價建構，且 `t+1` hit stop/tp 後 `t+2` 必須以開盤第一個可執行價強制執行，不得要求再次碰價 | DONE | 已補 direct synthetic case，直接釘死 candidate plan 保留 `L` 基準 sizing、filled position 改用 `P_fill + ATR_t` 建構首個可執行 stop / tp，且 entry day 觸發 stop/tp 必須排程到次日開盤執行；停利若已於 `t+1` 觸發，`t+2` 就算未再碰價也必須依 queued action 減倉或出清 | `tools/validate/synthetic_flow_cases.py`, `core/entry_plans.py`, `core/position_step.py`, `core/portfolio_exits.py` |
| B106 | P1 | GUI / Debug 契約 | 只要 debug analysis 要求 `return_chart_payload=True` 或 `export_chart=True`，即使輸入 `price_df` 為空，也必須回傳可正規化的 placeholder chart payload，不得把空 payload 延後到 GUI figure / normalize 路徑才以 runtime 方式爆炸 | DONE | 已補 direct contract，直接釘死空 `price_df` 路徑仍須回傳單根 placeholder payload，避免 synthetic GUI / chart coverage 因空 payload 在 figure / normalize 階段 runtime 失敗而掩蓋真正規格檢查 | `tools/trade_analysis/reporting.py`, `tools/validate/synthetic_contract_cases.py` |
| B107 | P1 | Meta / GUI 契約 | synthetic contract literal chart payload `x` 欄位 hygiene（已退役；不再作 formal blocker） | N/A | 屬 synthetic contract 自身樣本 payload hygiene；保留 ID 只作歷史索引，不再將 dict literal 的欄位補齊細節上升為 formal blocker | `doc/PROJECT_SETTINGS.md` |
| B108 | P1 | 核心韌性 / 空輸入契約 | 單股回測核心在空 `price_df` 下必須直接回傳零交易、零 miss buy/sell、`is_candidate=False` 的穩定 stats，不得在尾端索引 `C[-1]` / `Dates[-1]` / `buyCondition[-1]` 才 runtime 失敗 | DONE | 已補 direct synthetic case，直接釘死空 `price_df` 進入 `run_v16_backtest()` 時必須 early-return 穩定 stats 與空 logs；避免 GUI / debug 空資料 placeholder 路徑再次在 backtest core 尾端索引炸出 `IndexError` | `tools/validate/synthetic_flow_cases.py`, `core/backtest_core.py` |
| B109 | P1 | Meta / Registry 契約 | `tools/validate/synthetic_cases.py` 對各 `synthetic_*` 模組的 from-import 必須指向實際存在該 validator symbol 的模組；不得把 validator 匯錯模組，導致 coverage synthetic suite 在 import 時直接 `ImportError` | DONE | 已補 meta contract，直接 AST 掃描 `synthetic_cases.py` 對各 `synthetic_*` 模組的 from-import，逐一驗證被匯入 symbol 確實存在於目標模組；避免再發生 validator 實作位於 `synthetic_flow_cases.py` 卻誤從 `synthetic_portfolio_cases.py` 匯入的 registry import 回歸 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_cases.py` |
| B110 | P1 | 錯誤處理 / Meta 契約 | 專案內廣義例外處理（`except Exception` / `except BaseException`）必須綁定例外物件並可追蹤；若非直接 re-raise，handler 內必須使用綁定的例外，禁止 silent swallow | DONE | 已補 meta contract，直接 AST 掃描 `apps/`、`config/`、`core/`、`strategies/`、`tools/` 的 broad exception handler；逐一要求 handler 必須綁定例外名稱，且若非 re-raise 就必須在 body 內實際使用該例外，避免再次出現 GUI / chart runtime 失敗被靜默吞掉而無法定位 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/charting.py`, `tools/workbench_ui/single_stock_inspector.py` |
| B111 | P1 | 錯誤處理 / Meta 契約 | optional dependency fallback（如 `ImportError` / `ModuleNotFoundError`）不得 silent swallow；若選擇降級或略過，必須綁定例外並保留可追蹤 detail | DONE | 已補 meta contract，直接 AST 掃描 GUI / debug / validate / downloader 的 optional dependency fallback handler；逐一要求 handler 必須綁定例外名稱，且若非 re-raise 就必須在 body 內實際使用該例外，避免 TkAgg / coverage / curl_cffi 缺失時只默默降級而無法定位 | `tools/validate/synthetic_meta_cases.py`, `tools/workbench_ui/single_stock_inspector.py`, `tools/validate/main.py`, `tools/downloader/runtime.py` |
| B112 | P1 | 錯誤處理 / GUI 契約 | GUI TclError fallback 不得 silent swallow；若因 theme / palette / maximize 相容性而降級，必須綁定例外並保留可追蹤 detail | DONE | 已補 meta contract，直接 AST 掃描 `tools/workbench_ui/*.py` 的 `TclError` fallback handler，且掃描目標不得為空集合；逐一要求 handler 必須綁定例外名稱，且若非 re-raise 就必須在 body 內實際使用該例外，避免 GUI 啟動時 theme / zoom fallback 靜默吞錯或因掃描路徑漂移而 vacuous pass | `tools/validate/synthetic_meta_cases.py`, `tools/workbench_ui/workbench.py` |
| B113 | P1 | 錯誤處理 / Meta 契約 | 非 broad / optional-import / GUI TclError 的 specific exception 若採 pass-only silent fallback，必須被 formal suite 直接禁止；僅允許明確列為 control-flow 的例外（目前 `FileNotFoundError` cleanup）保留 pass-only | DONE | 已補 meta contract，直接 AST 掃描 `apps/`、`config/`、`core/`、`strategies/`、`tools/` 的 specific pass-only exception handler，排除 synthetic 測試檔與允許的 `FileNotFoundError` cleanup；逐一禁止 `ValueError` / `OSError` 等非 control-flow 例外以 `pass` 靜默吞掉，避免 shared runtime / chart helper 降級路徑失去可追蹤性 | `tools/validate/synthetic_meta_cases.py`, `core/runtime_utils.py`, `tools/trade_analysis/charting.py` |
| B114 | P1 | GUI / Debug 顯示契約 | 單股與投組 GUI / debug 的買訊、買進、賣訊、停利、停損、指標賣出、期末結算資訊框，必須使用同一欄位 schema 與同一 formatter；投組右側單股歷史績效必須使用本次實際回測 params 與 fixed_risk，成交 marker 不得用 OHLC fallback 假補成交價，active stop / tp / limit 線必須由 portfolio engine 輸出而非 GUI 重演交易邏輯；投組單檔視角必須可追查成交與錯失事件，missed-only ticker 不得從下拉與 K 線消失 | DONE | 已補 output contract，直接驗證 debug view params 需以 `scanner_live_capital` 正規化起始資金、買訊框需依 schema 顯示資金 / 股數 / 限價 / 預留，買進框需依 schema 顯示資金 / 股數 / 停利 / 限價 / 成交 / 停損 / 實支 / 進場類型 / 結果；買入資訊框必須像買訊資訊框一樣貼在對應 K 線下方，不得缺席也不得搬到右側 sidebar；若買訊框與買入框落在同一 K 線或相鄰少數 K 線，必須以水平錯位與額外垂直分層避免重疊。單股與投組 inspector 右側 sidebar 右下區顯示交易資訊（停利 / 限價 / 成交 / 停損 / 預留 / 實支），不得把買訊資訊混成交易資訊；hover snapshot 必須分別綁定 `reserved_capital` 與 `buy_capital`，停利框需依序顯示成交 / 金額 / 損益 / 報酬率，停損 / 指標賣出 / 期末結算框需顯示資金 / 股數 / 成交 / 金額 / 損益 / 報酬率 / 勝率 / 最大回撤 / 交易次數，且不得出現總損益標籤；投組顯示契約另釘死 fixed_risk cache key、filled-buy/missed-buy 成交率、engine 輸出 active levels、不使用 OHLC fallback、missed-only ticker 下拉追查、錯失買進 / 錯失賣出 K 線 marker、錯失列預留 / 實支分流與單檔 missed / 資金效率摘要，避免單股 / 投組顯示口徑再分叉 | `tools/validate/synthetic_contract_cases.py`, `tools/trade_analysis/trade_log.py`, `tools/trade_analysis/entry_flow.py`, `tools/trade_analysis/exit_flow.py`, `tools/trade_analysis/charting.py`, `tools/workbench_ui/single_stock_inspector.py`, `tools/workbench_ui/portfolio_backtest_inspector.py`, `core/portfolio_engine.py`, `core/portfolio_entries.py`, `core/portfolio_exits.py` |
| B115 | P1 | Meta / GUI 契約 | synthetic GUI / debug contract 若需 patch `tools.trade_analysis.backtest` 的 PIT history snapshot seam，`tools/trade_analysis/backtest.py` 必須穩定暴露 `_build_pit_history_snapshot`，且 `run_debug_analysis()` 必須經由該 seam 取用；不得只保留直接 import helper，否則 formal synthetic suite 會在 patch 階段提早 `AttributeError`，並連帶造成 coverage target 缺漏假失敗 | DONE | 已補 meta contract，直接 AST 掃描 `tools/trade_analysis/backtest.py`，釘死模組必須暴露 `_build_pit_history_snapshot` 且 `run_debug_analysis()` 的 signal-day / latest-day snapshot 都必須經由該 seam；並同步在 backtest 模組恢復 stable alias，避免 helper 重構時再次把 synthetic coverage suite 在 patch 階段炸掉 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/backtest.py`, `tools/validate/synthetic_contract_cases.py` |
| B116 | P1 | GUI / Visual 契約 | 單股 GUI / debug 的買訊藍色 annotation 箭頭，其尖點必須錨定訊號 K 棒低點；實際買進三角箭頭則必須錨定成交價，兩者不得混用成同一價位 | DONE | 已補 output contract，直接驗證 `_record_buy_signal_annotation(...)` 即使存在 entry plan 仍必須以 `signal_low` 作為 `anchor_price`，並驗證買進 trade marker 必須使用 `buy_price`；避免 GUI 把買訊位置與實際成交位置畫成同一點而失真 | `tools/validate/synthetic_contract_cases.py`, `tools/trade_analysis/backtest.py` |
| B117 | P1 | Meta / Fixture Schema 契約 | `tools/validate/synthetic_contract_cases.py` 對 multi-ticker synthetic case 不得再直接讀取過時的 `case["price_df"]`；必須改由 `case["frames"][case["primary_ticker"]]` 取得主 frame，避免 synthetic coverage suite 因 fixture schema 漂移在 contract 自身 runtime 提早 `KeyError` | DONE | 已補 meta contract，直接 AST 掃描 `synthetic_contract_cases.py`，禁止 `case["price_df"]` 舊存取；並同步修正 GUI visual contract 改從 `frames[primary_ticker]` 取主 frame，避免 coverage synthetic suite 再因 validator 自己的 fixture schema 假設過時而提早中斷 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B118 | P1 | Meta / Helper Import 契約 | `tools/validate/synthetic_contract_cases.py` 若直接呼叫 `tools.trade_analysis.backtest._record_buy_signal_annotation`，必須顯式 import 該 helper；不得引用未定義名稱，否則 synthetic coverage suite 會先以 `NameError` 中斷，並連帶造成 coverage target 缺漏假失敗 | DONE | 已補 meta contract，直接 AST 掃描 `synthetic_contract_cases.py`，釘死檔案若使用 `_record_buy_signal_annotation` 符號就必須顯式自 `tools.trade_analysis.backtest` import；並同步補上 validator 實際 import，避免 GUI visual contract 再因未定義名稱提早炸掉 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B119 | P1 | Meta / GUI 契約 | `validate_gui_trade_count_and_sidebar_sync_contract_case` 必須以強制結算行為 probe 驗證 completed round-trip 交易次數，不得再把過時的 `history_snapshot=latest_history_snapshot` / `_resolve_completed_trade_count(history_snapshot, include_current_round_trip=True)` 字串當成唯一正確實作，否則 synthetic suite 會把已正確的 GUI 交易次數語意誤判成 FAIL | DONE | 已補 meta contract，直接掃描 `synthetic_contract_cases.py` 的 GUI trade-count validator，禁止舊 exit snippet literal，並強制 validator 必須使用 `append_debug_forced_closeout(...)` 與 `build_trade_stats_index(...)` 做行為驗證；避免 contract 再因綁死過時實作細節而製造假失敗 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_contract_cases.py` |
| B120 | P0 | 交易規格 / 一致性 | 投組 cash-capped entry 路徑不得丟失 candidate plan 的 `entry_atr` / `target_price` 等成交後 first-actionable stop/tp 所需欄位；否則單股與投組會在新 `P_fill + ATR_t` 規格下分叉 | DONE | 已補 direct synthetic case，直接釘死 `build_daily_candidates()` 產生的 orderable candidate 必須保留 `target_price` / `entry_atr`，且 `execute_reserved_entries_for_day()` 經 cash-capped entry 後建立的 filled position 仍須以實際成交價建構 `initial_stop` / `trailing_stop` / `tp_half` 與 entry-day queued action；避免投組在資金重算路徑悄悄退回舊 `L / init_sl / T` 語意 | `tools/validate/synthetic_flow_cases.py`, `core/portfolio_candidates.py`, `core/portfolio_entries.py` |
| B121 | P0 | 交易規格 / 一致性 | 只要 `t+1` 已觸及 `L`、`qty > 0` 且資金條件成立，就不得僅因 `P_fill <= candidate_plan.init_sl` 而拒絕成交；`candidate_plan.init_sl` 只能作盤前 worst-case sizing，不得作成交否決下限 | DONE | 已補 direct synthetic case，直接釘死當實際成交價低於 candidate plan 的 limit-based sizing stop 時，`execute_pre_market_entry_plan()` 仍必須成交，且 filled position 之 `initial_stop` / `trailing_stop` / `tp_half` 必須改用實際成交價與 `ATR_t` 建立；避免 runtime 殘留舊 `先達停損放棄進場` 語意 | `tools/validate/synthetic_flow_cases.py`, `core/entry_plans.py` |
| B122 | P1 | Meta / Quick Gate 契約 | `quick_gate` 必須在 CLI / formal 匯入前，以靜態方式驗證 `tools/validate/synthetic_cases.py` 的 from-import target 是否指向實際宣告該 validator symbol 的模組；不得等 runtime import `synthetic_cases` 時才以 `ImportError` 失敗，並連帶遮蔽原本應回報的 CLI / dataset guard | DONE | 已補 quick gate static contract，直接共用 synthetic import-target resolution helper，於 `run_static_checks()` 新增 `synthetic_registry_import_targets`；即使 `synthetic_cases.py` 尚未可 import，仍能在 quick gate 前置攔截匯錯模組的 validator import，避免同一個 wiring 錯誤同時拖垮 quick_gate / consistency / meta_quality | `tools/local_regression/run_quick_gate.py`, `tools/validate/meta_contracts.py` |
| B123 | P0 | 契約 / 一致性 | 正式帳務中的 `cash` / `pnl` / `equity` / `reserved_cost` / `risk`、partial-exit cost-basis allocation、商品別驅動的賣出交易稅與 tick/limit hit 判斷，必須收斂到整數 exact-accounting 單一真理來源；directional tick rounding、tick band lookup、商品 profile 自動辨識、漲跌停 raw-limit 對齊與賣出稅率，也必須先依 raw price、ticker/metadata 與交易日期決定方向、區間、商品別與稅率，再轉為合法 tick 價與 net sell total；不得再以含費每股價或 per-share × qty 回推正式總額，也不得先做 0.001 量化後再決定 up/down、band，或把 ETF / ETN / REIT / 債券 ETF 類誤套股票 tick ladder / 單一 stock-only 交易稅 | DONE | 已補 exact-accounting unit-like contract，直接釘死 ledger 守恆、partial-exit cost-basis 回沖、integer tick/limit hit、cash/risk boundary、單股/投組 closeout parity、display-derived 欄位，以及 raw-price directional tick rounding / band lookup 邊界、跨級距漲跌停 raw-limit tick-band 對齊、ETF / ETN / REIT 類商品 profile 自動辨識後的兩級 tick 規則，與依商品別 / 交易日期決定的賣出交易稅（股票 0.3%、ETF / ETN 0.1%、REIT 免稅、債券 ETF 免徵期限內 0 稅率）；並同步把正式核心 cash/pnl/equity/reserved_cost 與商品別驅動的 raw-price tick 正規化 / limit-price 對齊 / sell-tax 路徑收斂到 `core/exact_accounting.py` | `tools/validate/synthetic_unit_cases.py`, `core/exact_accounting.py` |
| B124 | P1 | Meta / Schema 契約 | `build_backtest_stats()` / `run_v16_backtest()` 的 public stats payload 必須維持既有 snake_case 相容 key（如 `trade_count` / `expected_value` / `asset_growth` / `max_drawdown` / `missed_buys` / `is_setup_today` / `extended_candidate_today` / `current_position` / `score`）；若 stats producer 進一步把 `final_date` / `trade_date` / `security_profile` 等日期或商品上下文往下傳給 preview builder，或顯示 `buyPrice` / `sellPrice` / `tpPrice` 等可見 preview 價格，也必須同步更新函式簽名、全部 caller 與 static contract，並重用共享 stop / target helper；不得只在函式內引用未宣告日期名稱，或讓正常 / 延續預覽漏傳商品 profile，導致 GUI / scanner / consistency runtime `NameError` 或可見價格與正式交易規則分叉 | DONE | 已補 static meta contract，直接 AST 掃描 `core/backtest_finalize.py` 的 `build_backtest_stats()` dict key，並釘死 `final_date` / `security_profile` 簽名、empty/final caller 傳遞、normal / extended preview 必須走 threaded `security_profile` 與 `trade_date=final_date`，且 `sellPrice` / `tpPrice` preview 必須分別走共享 initial-stop / target helper、不得保留手寫 `tpPrice` 浮點公式；避免 refactor 後只改 producer 內部實作就讓 GUI / scanner / consistency runtime 因缺 key、`NameError` 或商品 profile 漏傳而中斷 | `tools/validate/synthetic_meta_cases.py`, `core/backtest_finalize.py`, `core/backtest_core.py` |
| B125 | P0 | 契約 / 一致性 | 單股 backtest / debug 的 `currentCapital` 必須固定代表可用現金；任何半倉、全倉與期末結算都只能加回 `net sell total`，`currentEquity` 必須以 `cash + 當前可變現淨值` 計；不得再沿用 `+ realized pnl` 或 `cash + floating pnl` 的舊路徑，否則會讓單股與投組 trade_count / asset_growth / completed-trade PnL 分叉 | DONE | 已補 static meta contract，直接掃描 `core/backtest_core.py`、`core/backtest_finalize.py`、`tools/trade_analysis/backtest.py` 與 `tools/trade_analysis/exit_flow.py`，釘死單股 / debug 現金更新必須使用 freed cash / net sell total，mark-to-market equity 必須使用 net liquidation value；避免 exact-accounting 遷移後把 `currentCapital` 誤當 accumulated pnl | `tools/validate/synthetic_meta_cases.py`, `core/backtest_core.py` |
| B126 | P1 | Meta / 契約 | debug backtest / GUI 重播現金路徑時，買進當下也必須扣除實際 `net buy total`，且凡 entry flow 或 preview builder 新增 `ticker` / `security_profile` / `trade_date` 等商品 profile / 日期感知 keyword 參數時，必須同步更新 callee 函式簽名與 caller 傳遞；不得只在賣出時更新 `current_capital`，也不得只改 caller 造成 `unexpected keyword argument` 或商品 profile / 交易日期傳遞中斷，否則會讓 debug trade log sizing、completed-trade PnL sequence 與核心單股回測分叉 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/backtest.py` 與 `tools/trade_analysis/entry_flow.py`，釘死 debug entry flow 必須回傳 `spent_cash` 並由主流程扣減 `current_capital`，且當 caller 傳入 `ticker` / `security_profile` / `trade_date` 時，callee 簽名與 normal / extended entry builder、買訊預覽與 latest raw / extended preview 呼叫也必須同步接收並傳遞；避免重播路徑只補賣出現金、漏扣買進現金或讓 GUI 預覽仍走缺 profile 的舊路徑 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/backtest.py`, `tools/trade_analysis/entry_flow.py` |
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
| B146 | P1 | Meta / array 價格正規化契約 | 任何 scalar / vectorized / array 版價格正規化 helper（如批次 tick 對齊、批次買入上限正規化）也必須委派共享 raw-price tick helper，且 shared caller 必須把 ticker / security_profile / trade_date 正確往下傳；不得另寫 threshold/tick ladder、`valid_prices / ticks`、`np.ceil` / `np.floor` 的獨立浮點取整實作，也不得先以 `price_to_milli(...)` 預量化 raw price 後再決定 tick band / direction，或在 backtest / candidate / scanner projected qty 路徑遺漏商品 profile 與日期上下文，導致 ETF / ETN / REIT / 債券 ETF 仍走 stock-only 稅率與 ladder；prepared fast-data / scanner frame 的商品 profile 與最新交易日也必須由共享欄位或 helper 統一傳遞與解析，避免 scanner runtime 與 validator/oracle 因資料形態差異分叉；rotation / exit caller 也不得引用未宣告的 `ticker` 自由變數 | DONE | 已補 static meta contract，直接掃描 `core/price_utils.py`、`core/signal_utils.py`、`core/backtest_core.py`、`core/portfolio_fast_data.py`、`core/portfolio_candidates.py`、`core/portfolio_entries.py`、`core/position_step.py`、`core/portfolio_exits.py`、`core/entry_plans.py` 與 `tools/scanner/stock_processor.py`，釘死 scalar / array 版 `get_tick_size*()` 與 `round_to_tick*()` 必須委派 `get_tick_milli_from_price(...)` 與 `round_price_to_tick_milli(...)`，且 shared caller 必須以 ticker / security_profile / trade_date 將商品 profile、日期上下文與 prepared frame 最新交易日傳遞到單股 backtest、normal/extended candidate、買入上限、candidate-plan resize、scanner projected qty、漲跌停、rotation 賣出與出場路徑；不得殘留 `get_tick_milli(price_to_milli(price))`、`round_price_milli_to_tick(price_to_milli(price), ...)`、`ratios = valid_prices / ticks`、`np.ceil` / `np.floor` 的舊路徑、stock-only tick / 稅率假設、prepared fast-data 丟失 `security_profile`、僅依 `df["Date"]` 取最新日期或 `adjust_long_sell_fill_price(w_open, ticker=ticker)` 這類未宣告變數引用 | `tools/validate/synthetic_meta_cases.py`, `core/price_utils.py`, `core/portfolio_fast_data.py`, `core/portfolio_candidates.py`, `core/entry_plans.py`, `tools/scanner/stock_processor.py`, `core/portfolio_exits.py` |
| B147 | P1 | Meta / 正式入口摘要註解排除契約 | `apps/test_suite.py` 的頂部摘要註解若保留，僅供閱讀，不納入本地 formal test suite 驗證；正式入口同步以 synthetic registry、`--help`、文件與 checklist 為準，避免註解字串成為無止境的 mechanical contract 維護點 | N/A | 已自正式範圍移除；歷史 `G` 備註中的舊 validator 名稱僅保留文字紀錄，不再要求以 compatibility stub 或 registry 例外維持可解析，正式同步改以 registry / `--help` / 文件 / checklist 為準 | `doc/TEST_SUITE_CHECKLIST.md`, `apps/test_suite.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_cases.py` |
| B148 | P1 | Meta / exact-ledger ratio path 契約 | 凡 debug / GUI / history / rotation 以 exact ledger 或 integer total 計算報酬率、持倉優劣比較或 leg return 時，分子分母必須直接使用 milli / integer total 相除；不得先各自 `milli_to_money(...)` 轉回 float 金額後再相除，避免浮點偏移在可見報酬率與 rotation 排序產生微小分叉 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/exit_flow.py`、`tools/trade_analysis/backtest.py` 與 `core/portfolio_exits.py`，釘死 total/leg/sell-signal/rotation return 必須直接以 integer total 計算，且不得保留 `milli_to_money(...)/milli_to_money(...)` 舊路徑 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/exit_flow.py`, `tools/trade_analysis/backtest.py`, `core/portfolio_exits.py` |
| B149 | P1 | Meta / debug exit milli binding 契約 | 凡 debug / GUI / history 的 final-exit `total_return_pct` 等可見公式改為引用 `total_pnl_milli`、`full_entry_capital_milli` 等整數 ledger 變數時，必須先在同一路徑明確綁定該 `*_milli` 變數，再由其推導顯示值；不得殘留未定義局部變數或混用舊 float `total_pnl` 路徑，避免 consistency / chain checks 因 runtime `NameError` 全面中斷 | DONE | 已補 static meta contract，直接掃描 `tools/trade_analysis/exit_flow.py` 的 `process_debug_position_step()` 與 `append_debug_forced_closeout()`，釘死 final-exit `total_return_pct` 必須先綁定對應 `total_pnl_milli`，再以 `milli_to_money(total_pnl_milli)` 推導顯示 `total_pnl`；不得保留 `float(position.get('realized_pnl', pnl_realized))` 或 `float(position.get('realized_pnl', 0.0) + final_leg_actual_pnl)` 等舊 float 路徑 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/exit_flow.py` |
| B150 | P1 | Meta / core R-multiple exact-ledger 契約 | 核心回測與投組統計若需累計 `R_Multiple` / `r_mult`、勝負 R 與 EV，必須以 `*_pnl_milli` 與 `initial_risk_total_milli` 直接走整數 ledger ratio helper；不得回退成 `float total_pnl / float initial_risk_total`，避免 closed-trades stats、單股與投組統計在邊界值因浮點路徑再度分叉 | DONE | 已補 static meta contract 與共享 ratio helper，直接掃描 `core/exact_accounting.py`、`core/backtest_core.py`、`core/backtest_finalize.py` 與 `core/portfolio_exits.py`，釘死核心 `trade_r_mult` / `total_r` 必須委派 `calc_ratio_from_milli(...)`，且不得殘留 `total_pnl / position['initial_risk_total']` 或 `total_pnl / pos['initial_risk_total']` 舊浮點公式；避免單股回測、期末結算、rotation 與投組 closeout 的 `R_Multiple` / EV 統計只在核心路徑分叉 | `tools/validate/synthetic_meta_cases.py`, `core/exact_accounting.py`, `core/backtest_core.py`, `core/backtest_finalize.py`, `core/portfolio_exits.py` |
| B151 | P2 | 文件 / GUI workbench 文件同步契約 | `doc/CMD.md` 與 `doc/ARCHITECTURE.md` 的 GUI workbench 頁籤敘述必須與實作一致；在已移除執行摘要分頁後，文件不得再殘留 summary-tab 描述 | DONE | 已補 static document-sync contract，直接比對 `single_stock_inspector.py` 的 notebook tabs 與兩份文件中的 workbench 描述；釘死文件必須描述為 `K 線圖 / 交易明細 / Console` 三分頁，且不得再殘留執行摘要分頁敘述 | `tools/validate/synthetic_meta_cases.py`, `doc/CMD.md`, `doc/ARCHITECTURE.md`, `tools/workbench_ui/single_stock_inspector.py` |
| B152 | P2 | 文件 / trade_analysis rename 相容契約 | `tools/trade_analysis/` 完成 rename 後，若對外 API、validator label 或輸出資料夾仍為相容性暫保留 `run_debug_*` / `debug_trade_log` 舊名，文件必須明確標示為 legacy 相容名；單股 trade-analysis 的正式使用者入口必須維持 `apps/workbench.py`，`tools/trade_analysis/trade_log.py` 只能描述為共用 backend / 開發輔助 CLI；helper CLI 可保留協助性 help / prompt，但不得再被 formal quick-gate / extended dataset CLI surface 視為 release-blocking 正式使用者入口，避免維護者誤判模組責任、入口層級與輸出分類 | DONE | 已補 static document-sync / helper-CLI contract，直接比對 `doc/CMD.md`、`doc/ARCHITECTURE.md` 與 `tools/trade_analysis/trade_log.py`；釘死文件必須將 `apps/workbench.py` 描述為單股 trade-analysis 單一使用者入口、將 `trade_log.py` 描述為共用 backend / 開發輔助 CLI、明示 legacy `run_debug_*` / `debug_trade_log` 相容名，並在輸出與 retention 區段一律使用完整 `outputs/debug_trade_log/` canonical 路徑標示 trade_analysis legacy output dir；formal CLI surface 也已同步移出 `trade_log.py` 的 quick-gate help / dataset-error 契約，避免 helper CLI 被誤當正式入口。不得再殘留 `trade_log.py` 正式入口、`交易除錯子系統`、`請輸入要除錯的股票代號`、`交易明細除錯工具` 或裸 `debug_trade_log` 目錄名 | `tools/validate/synthetic_cli_cases.py`, `tools/validate/synthetic_meta_cases.py`, `tools/local_regression/run_quick_gate.py`, `doc/CMD.md`, `doc/ARCHITECTURE.md`, `tools/trade_analysis/trade_log.py` |
| B153 | P2 | 契約 / trade_analysis canonical alias export 契約 | `tools/trade_analysis/` rename 後，package `tools.trade_analysis` 與 `tools/trade_analysis/trade_log.py` 必須同步提供 canonical `run_trade_analysis` / `run_trade_backtest` / `run_prepared_trade_backtest` / `run_ticker_analysis` aliases，並保留 legacy `run_debug_*` 相容入口；不得只剩 legacy debug 命名，避免模組名稱已切換但公開 API 仍被舊語意綁死 | DONE | 已補 static export contract，直接比對 `tools/trade_analysis/__init__.py`、`tools/trade_analysis/trade_log.py` 與文件；釘死 package 與 trade_log 都必須同時暴露 canonical aliases 與 legacy `run_debug_*` 相容 aliases，並在 `doc/CMD.md` / `doc/ARCHITECTURE.md` 明示 canonical alias 名稱 | `tools/validate/synthetic_meta_cases.py`, `tools/trade_analysis/__init__.py`, `tools/trade_analysis/trade_log.py`, `doc/CMD.md`, `doc/ARCHITECTURE.md` |
| B154 | P1 | Meta | synthetic_contract_cases shared path helper 使用 hygiene（已退役；不再作 formal blocker） | N/A | 屬 validator 自身 path helper 使用細節；保留 ID 只作歷史索引，不再將 `synthetic_contract_cases.py` 的 helper 選擇上升為 formal blocker | `doc/PROJECT_SETTINGS.md` |
| B155 | P1 | Meta / checklist G 連續狀態鏈契約 | `doc/TEST_SUITE_CHECKLIST.md` 的 `G. 逐項收斂紀錄` 同日同 ID 若有多筆狀態變更，前一筆的 `to` 狀態必須等於下一筆的 `from` 狀態；不得出現 `PARTIAL -> DONE` 後又直接接 `DONE -> PARTIAL` 之前未先回到 `DONE` 的斷鏈順序，避免 done/unfinished 摘要與主表狀態同步被歷史紀錄順序誤導 | DONE | 已補 checklist G transition-chain contract，直接檢查同 ID 歷史收斂列的前後狀態是否連續，並以 mutation case 釘死若將同 ID 的 DONE/PARTIAL 收斂列寫反順序，`run_meta_quality.py` 必須回報 FAIL；同輪也已重排歷史斷鏈列，避免 convergence 最新狀態被錯序列覆蓋 | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `doc/TEST_SUITE_CHECKLIST.md` |
| B156 | P1 | Meta / public stats 盈虧口徑契約 | 單股 backtest public stats payload 的 `totalNetProfit` / `totalNetProfitPct` 必須與 `currentEquity` 承接同一淨值口徑；只要期末仍有持倉，legacy camelCase `totalNetProfit` 也不得退回可用現金 `currentCapital` 基準，避免同一 payload 內部現金/權益語意分叉 | DONE | 已補 static meta contract，直接掃描 `core/backtest_finalize.py`，釘死 `total_net_profit` 必須以 `current_equity - initial_capital` 計，且 public payload 的 `totalNetProfit` / `totalNetProfitPct` 必須共用同一 profit/equity 基準；不得殘留 `current_capital - initial_capital` 舊路徑 | `tools/validate/synthetic_meta_cases.py`, `core/backtest_finalize.py` |
| B157 | P1 | Meta / 正式入口 help 穩定主題摘要契約 | `apps/test_suite.py --help` 只保留穩定、跨模組、對使用者仍有意義的主題摘要；若需描述已實作測試覆蓋範圍，必須用穩定 theme token，而不得逐項列舉高波動 exact contract 名稱，避免正式入口 help 再度變成高維護同步面 | DONE | 已補 static meta contract，直接擷取 `apps/test_suite.py --help` 輸出並檢查說明列，釘死其必須保留交易口徑一致性、保守出場解讀與 `debug-backtest` 現金路徑等對使用者仍有意義的穩定主題摘要；避免將 checklist heading、治理 guard 或其他內部 meta 細節重新灌回正式入口 help | `tools/validate/synthetic_meta_cases.py`, `apps/test_suite.py` |
| B158 | P1 | Meta | 正式入口 help 舊 wording / 裸用詞排除契約（已退役；改留穩定主題正向契約） | N/A | 屬 help wording / bare-term hygiene；保留 ID 只作歷史索引，正式長期 contract 改由 B157 的穩定主題正向契約承接，不再以負向 wording 排除作 formal blocker | `apps/test_suite.py`, `doc/PROJECT_SETTINGS.md` |
| B161 | P0 | Meta / 同一事件保守可執行解讀契約 | 同一事件的判斷與執行口徑必須一致，且在不確定時一律採最保守、最不利於績效的可執行解讀；同棒停損 / 停利衝突不得先取較佳結果，前日已決定的停損也不得要求隔日再觸價一次才執行 | DONE | 已補 direct synthetic case，明確釘死同棒 stop/tp 歧義必須先取 STOP、若開盤已落到更差可成交價則以該更差 open 成交、前一日已決定的 deferred stop 必須於次一日開盤直接執行且不得要求再跌破一次 | `tools/validate/synthetic_take_profit_cases.py`, `core/position_step.py`, `core/entry_plans.py` |
| B162 | P2 | 文件 / ARCHITECTURE apps exact file-tree 契約 | `doc/ARCHITECTURE.md` 不再以 apps exact file-tree 作為主要承載面；架構文件只保留正式入口與分層責任，不再逐列維護 GUI 入口的 file-tree 細節 | N/A | 已自 long-term formal scope 移除；`apps/workbench.py` 仍於正式入口段落承接，但不再由 exact file-tree validator 逐條釘死 | `doc/ARCHITECTURE.md` |
| B163 | P2 | 文件 / ARCHITECTURE models exact file-tree 契約 | `doc/ARCHITECTURE.md` 不再以 models exact file-tree 與 shipped best_params 檔名作為主要承載面；架構文件只保留 models 類別責任，不再逐檔維護工件清單 | N/A | 已自 long-term formal scope 移除；best_params 類工件仍由 repo 本體與正式流程承接，不再要求 `doc/ARCHITECTURE.md` 逐檔列舉 | `doc/ARCHITECTURE.md` |
| B164 | P2 | 文件 / ARCHITECTURE internal helper exact file-tree 契約 | `doc/ARCHITECTURE.md` 不再把 internal helper 模組 exact file-tree 列舉作為長期 formal contract 主體；架構文件只保留穩定子系統、正式入口與少數關鍵 shipped 工件索引，避免高波動 helper 清單反客為主 | N/A | 已將 internal helper exact file-tree 列舉退出 long-term formal scope；是否保留個別 helper 說明改由 `doc/ARCHITECTURE.md` 自行維護，不再由 formal validator 逐條釘死 | `doc/ARCHITECTURE.md` |
| B165 | P2 | 文件 / ARCHITECTURE internal support exact file-tree 契約 | `doc/ARCHITECTURE.md` 不再把 internal support 模組 exact file-tree 列舉作為長期 formal contract 主體；架構文件只保留穩定子系統、正式入口與少數關鍵 shipped 工件索引，避免高波動 support 清單成為主要同步面 | N/A | 已將 internal support exact file-tree 列舉退出 long-term formal scope；是否保留個別 support 模組說明改由 `doc/ARCHITECTURE.md` 自行維護，不再由 formal validator 逐條釘死 | `doc/ARCHITECTURE.md` |
| B166 | P2 | 文件 / ARCHITECTURE local_regression meta_quality 檔案樹同步契約 | `doc/ARCHITECTURE.md` 的 Local Regression 檔案樹必須以可機械比對的乾淨 tree entry 列出 `tools/local_regression/run_meta_quality.py`；不得把說明直接拼進檔名字串，避免 shipped file path 與文件樹條目分叉 | DONE | 已補 static document-sync contract，直接釘死 Local Regression 檔案樹需列出乾淨的 `run_meta_quality.py` tree entry、排除把 helper 說明拼進檔名的 malformed line，並保留下方職責段落承接 `meta quality` 說明 | `tools/validate/synthetic_meta_cases.py`, `doc/ARCHITECTURE.md`, `tools/local_regression/run_meta_quality.py` |
| B167 | P2 | 輸出 / validate runtime 暫存 staging 契約 | validate / synthetic error-path / regression 暫存工件若需落到 repo `outputs/`，必須收斂到既有 `outputs/local_regression/_staging/` 內部 staging；不得另建 `outputs/validate/` 根分類，避免輸出分類、retention 與文件同步再度分叉 | DONE | 已補 static output-path contract，直接比對 `tools/validate/synthetic_error_cases.py`、`tools/validate/synthetic_regression_cases.py`、`tools/local_regression/run_all.py`、`doc/CMD.md` 與 `doc/ARCHITECTURE.md`；釘死 validate runtime 暫存必須走 `outputs/local_regression/_staging/validate_runtime/`，且 staging 清理仍由既有 `local_regression_staging` retention 規則承接；不得回流 `outputs/validate/` 根分類 | `tools/validate/synthetic_meta_cases.py`, `tools/validate/synthetic_error_cases.py`, `tools/validate/synthetic_regression_cases.py`, `tools/local_regression/run_all.py`, `doc/CMD.md`, `doc/ARCHITECTURE.md` |

### B3. 可隨策略升級調整的測試

| ID | 優先級 | 類別 | 項目 | 目前判定 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|---|---|
| B47 | P1 | 模型介面 | model feature schema / prediction schema 穩定 | DONE | 已新增 model I/O schema case，直接驗輸入欄位、輸出欄位、型別與缺值處理，並釘死 repo 內 `models/*best_params*.json` shipped 工件不得對 float-schema 欄位輸出 `int` 型別 | `tools/validate/synthetic_strategy_cases.py`, `tools/optimizer/`, `tools/scanner/`, `models/candidate_best_params.json`, `models/run_best_params.json` |
| B48 | P1 | 重現性 | 同 seed 下 optimizer / model inference 可重現 | DONE | 已新增 strategy repeatability case，直接雙跑 scanner inference 與 optimizer objective，驗證輸出 payload、trial params 與 profile_row 在固定 seed / 固定輸入下可重現 | `tools/validate/synthetic_strategy_cases.py`, `tools/local_regression/`, `tools/optimizer/` |
| B49 | P1 | 排序輸出 | ranking / scoring 輸出可排序、可比較、無 NaN | DONE | 已新增 ranking / scoring sanity case，直接驗 EV / PROJ_COST / HIST_WIN_X_TRADES / ASSET_GROWTH 排序值可用、方向一致與型別正確 | `tools/validate/synthetic_strategy_cases.py`, `core/buy_sort.py`, `tools/scanner/` |
| B50 | P2 | 最低可用性 | 模型升級後 scanner / optimizer / reporting 仍可跑通 | DONE | 已新增 strategy minimum viability case，直接驗 scanner、optimizer、strategy dashboard、scanner summary 與 yearly return report 在策略輸入下可正常執行，並釘死 scanner summary 的 issue-log path 必須維持 canonical `outputs/vip_scanner/`，不得回流退役 `outputs/scanner/` 類別 | `tools/validate/synthetic_strategy_cases.py`, `apps/ml_optimizer.py`, `apps/vip_scanner.py` |
| B51 | P2 | 報表相容 | 新策略輸出仍符合既有 artifact / reporting schema | DONE | 已新增 strategy reporting schema compatibility case，直接驗 candidate_best / run_best export payload keys、scanner normalized payload keys 與 yearly return report columns 維持既有 schema | `tools/validate/synthetic_strategy_cases.py`, `tools/portfolio_sim/`, `tools/scanner/reporting.py` |
| B52 | P1 | Optimizer 契約 | objective 淘汰值 / fail_reason / profile_row / candidate_best/run_best 參數工件 export 穩定 | DONE | 已新增 optimizer objective / export contract case，直接驗 `INVALID_TRIAL_VALUE`、fail_reason、profile_row、`tp_percent` 還原優先序、export 成敗、`atr_buy_tol` / `min_history_ev` / `tp_percent` step-float canonicalization、預設費率 decimal canonical 輸出、repo 內 `models/candidate_best_params.json` 與 `models/run_best_params.json` 參數工件不得殘留浮點尾差，以及訓練中斷且未達指定 trial 數時不得自動覆寫 `models/run_best_params.json`；僅完成指定訓練次數或輸入 0 走 export-only 模式時才可更新。另已補 training split policy loader / `.py` / `.json` override / `None -> auto-derive search_train_end_year` / legacy walk-forward policy symbol reject contract，並釘死 optimizer callbacks 的 train 篩選必須重用 core 單一來源、KC 顯示名稱須與 dashboard 一致 | `tools/validate/synthetic_strategy_cases.py`, `tools/validate/synthetic_cases.py`, `core/walk_forward_policy.py`, `config/training_policy.py`, `tools/optimizer/callbacks.py`, `tools/optimizer/main.py`, `tools/optimizer/objective_runner.py`, `tools/optimizer/runtime.py`, `tools/optimizer/study_utils.py`, `strategies/breakout/search_space.py`, `strategies/breakout/adapter.py`, `config/execution_policy.py`, `models/candidate_best_params.json`, `models/run_best_params.json` |
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
| T26 | `validate_cmd_document_contract_case` | B16 |
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
| T93 | `validate_core_trading_modules_in_coverage_targets_case` | B22 |
| T94 | `validate_critical_file_coverage_minimum_gate_case` | B22 |
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
| T108 | `validate_checklist_t_single_entry_delimiter_case` | B26 |
| T109 | `validate_checklist_g_transition_format_case` | B26 |
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
| T138 | `validate_policy_contract_modules_in_coverage_targets_case` | B61 |
| T139 | `validate_checklist_g_new_transition_first_occurrence_case` | B26 |
| T143 | `validate_checklist_summary_tables_sorted_by_id_case` | B26 |
| T144 | `validate_package_zip_commit_test_suite_orchestration_case` | B65 |
| T145 | `validate_gui_workbench_contract_case` | B66 |
| T146 | `validate_debug_trade_log_chart_context_optional_case` | B67 |
| T147 | `validate_tool_module_path_normalization_case` | B68 |
| T148 | `validate_module_path_normalizer_accepts_path_objects_case` | B69 |
| T149 | `validate_module_loader_project_root_string_patch_case` | B70 |
| T150 | `validate_gui_embedded_chart_contract_case` | B71 |
| T153 | `validate_gui_chart_workspace_contract_case` | B74 |
| T154 | `validate_gui_mouse_navigation_contract_case` | B75 |
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
| T180 | `validate_gui_signal_annotation_and_forced_close_visual_contract_case` | B101 |
| T181 | `validate_gui_trade_marker_and_tp_visual_contract_case` | B102 |
| T182 | `validate_gui_trade_count_and_sidebar_sync_contract_case` | B103 |
| T183 | `validate_checklist_physical_trading_principles_contract_case` | B104 |
| T184 | `validate_synthetic_init_sl_single_source_runtime_case` | B105 |
| T185 | `validate_debug_empty_price_df_chart_payload_contract_case` | B106 |
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
| T203 | `validate_checklist_t_formal_command_single_entry_case` | B26 |
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
| T234 | `validate_exact_ledger_return_ratio_no_money_float_division_contract_case` | B148 |
| T235 | `validate_debug_exit_total_return_milli_binding_contract_case` | B149 |
| T236 | `validate_core_r_multiple_exact_ledger_contract_case` | B150 |
| T237 | `validate_gui_workbench_documentation_sync_case` | B151 |
| T238 | `validate_trade_analysis_legacy_naming_documentation_contract_case` | B152 |
| T239 | `validate_trade_analysis_canonical_alias_export_contract_case` | B153 |
| T241 | `validate_checklist_g_transition_sequence_case` | B155 |
| T242 | `validate_single_backtest_public_profit_equity_consistency_contract_case` | B156 |
| T243 | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` | B157 |
| T248 | `validate_synthetic_conservative_executable_exit_interpretation_case` | B161 |
| T249 | `validate_architecture_workbench_entry_file_tree_sync_case` | B162 |
| T250 | `validate_architecture_models_champion_params_file_tree_sync_case` | B163 |
| T253 | `validate_architecture_local_regression_meta_quality_file_tree_sync_case` | B166 |
| T256 | `validate_validate_runtime_tmp_output_staging_contract_case` | B167 |
| T257 | `validate_optimizer_walk_forward_policy_contract_case` | B52 |
| T258 | `validate_optimizer_session_milestone_cache_case` | B52 |

## G. 逐項收斂紀錄

使用方式：每次只挑少數高優先項目處理，完成後更新本節，不要重開一份新清單。編輯本節時，先依日期定位到對應區塊，再抽出整個同日區塊依排序鍵重排後整段覆寫回原位；禁止把新列直接追加到該日期區塊尾端，也禁止只改局部單列後跳過同日區塊總排序檢查；若新增列排序鍵小於當前尾列，必須回插到正確位置，不得留在尾端。G 只記錄實際狀態變更；不得寫 `DONE -> DONE`、`PARTIAL -> PARTIAL`、`TODO -> TODO` 等 no-op transition。同日同 ID 若有多筆狀態變更，必須依實際演進排序；`NEW -> *` 只能出現在該 ID 首筆，且 `NEW -> PARTIAL` / `NEW -> DONE` 必須排在後續 `PARTIAL -> DONE` 或 `DONE -> PARTIAL` 之前。交付前至少再做一次同日區塊機械核對：由上到下檢查 namespace、數字段、尾碼三層排序鍵皆未逆序，且新增列同時滿足前一列 ≤ 當前列 ≤ 後一列；備註欄僅作最小必要的治理索引與人工說明。

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
| 2026-04-01 | T15 | 新增 optimizer fixed-seed 雙跑一致性檢查 | TODO -> PARTIAL | `tools/local_regression/run_ml_smoke.py` |
| 2026-04-01 | T18 | 新增 CSV / XLSX / JSON output contract case 並驗證 | TODO -> PARTIAL | validate_output_contract_case |
| 2026-04-01 | T19 | 新增 chain checks 雙跑 digest 對比與 optimizer 雙跑 | TODO -> PARTIAL | `tools/local_regression/run_chain_checks.py` |
| 2026-04-01 | T20 | 新增 `run_meta_quality.py` 產出 coverage baseline | TODO -> PARTIAL | `tools/local_regression/run_meta_quality.py` |
| 2026-04-01 | T21 | 新增 `run_meta_quality.py` performance baseline gating | TODO -> PARTIAL | `tools/local_regression/run_meta_quality.py` |
| 2026-04-01 | T22 | 新增 meta registry case 並驗證 | TODO -> DONE | validate_registry_checklist_entry_consistency_case |
| 2026-04-01 | T23 | 新增 meta fault-injection case 並驗證 | TODO -> DONE | validate_known_bad_fault_injection_case |
| 2026-04-01 | T24 | 新增 independent oracle golden case 並驗證 | TODO -> DONE | validate_independent_oracle_golden_case |
| 2026-04-01 | T26 | 新增 CMD 指令契約案例並驗證 | TODO -> DONE | validate_cmd_document_contract_case |
| 2026-04-02 | B01 | 補 setup index prev-day-only invariant 後主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_history_cases.py` |
| 2026-04-02 | B11 | 跨工具 schema / 欄位語意補齊，主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-02 | B12 | 決定性主表收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_ml_smoke.py` |
| 2026-04-02 | B15 | 補 downloader 外部 API fatal error path 後主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_error_cases.py` |
| 2026-04-02 | B16 | CLI 契約涵蓋補齊，主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_cli_cases.py` |
| 2026-04-02 | B18 | 重跑一致性 / 狀態汙染主表收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_chain_checks.py` |
| 2026-04-02 | B19 | 將 traced peak memory 納入正式 gate，主表升為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | B22 | 將 coverage report baseline 收斂為正式 gate，主表升為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | B23 | 檢出 synthetic 主入口漏註冊既有 `validate_*` case，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_cases.py` 尚未完整覆蓋 imported validate cases |
| 2026-04-02 | B23 | 補齊 synthetic 主入口遺漏註冊與 registry completeness guard 後收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_cases.py` |
| 2026-04-02 | B26 | 檢出完成摘要索引仍有漏同步風險，主表改回 PARTIAL | DONE -> PARTIAL | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | B26 | 補齊 checklist main / `T` / `G` sync blocker 後收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | T14 | 新增 model I/O schema 案例並驗證 | TODO -> DONE | `validate_model_io_schema_case` |
| 2026-04-02 | T15 | 補 scanner worker / `scan_runner` 入口重跑一致性後收斂完成 | PARTIAL -> DONE | `validate_scanner_worker_repeatability_case` |
| 2026-04-02 | T16 | 新增 ranking / scoring sanity 案例並驗證 | TODO -> DONE | `validate_ranking_scoring_sanity_case` |
| 2026-04-02 | T17 | reporting schema compatibility checks 收斂完成，並新增輸出檔 schema 補強 | TODO -> DONE | `validate_issue_excel_report_schema_case` |
| 2026-04-02 | T18 | 擴充 local regression summary contract 並收斂完成 | PARTIAL -> DONE | `validate_output_contract_case` |
| 2026-04-02 | T19 | 補 `run_all.py` 同 run dir rerun summary / bundle repeatability 後收斂完成 | PARTIAL -> DONE | `validate_optimizer_raw_cache_rerun_consistency_case` |
| 2026-04-02 | T20 | 補 manifest 化 line / branch threshold gate 與 summary sync | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | T21 | 補 traced peak memory regression gate 後 performance baseline 收斂完成 | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | T25 | 擴充 checklist sufficiency formal check 到單一正式入口與 legacy entry 檢查後收斂完成 | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | T27 | 擴充 scanner summary / banner 與 display re-export 後收斂完成 | PARTIAL -> DONE | `validate_display_reporting_sanity_case` |
| 2026-04-02 | T28 | 擴充 artifact lifecycle contract 並驗證 | PARTIAL -> DONE | validate_artifact_lifecycle_contract_case |
| 2026-04-02 | T29 | 新增 formal-entry consistency checks 並驗證 | TODO -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | T30 | 新增 params_io 錯誤路徑案例並驗證 | TODO -> DONE | `validate_params_io_error_path_case` |
| 2026-04-02 | T31 | 新增 module_loader 錯誤路徑案例並驗證 | TODO -> DONE | `validate_module_loader_error_path_case` |
| 2026-04-02 | T32 | 新增 preflight 錯誤路徑案例並驗證 | TODO -> DONE | `validate_preflight_error_path_case` |
| 2026-04-02 | T33 | 新增資料清洗 expected behavior 案例並驗證 | TODO -> DONE | `validate_sanitize_ohlcv_expected_behavior_case` |
| 2026-04-02 | T34 | 新增資料清洗 fail-fast 案例並驗證 | TODO -> DONE | `validate_sanitize_ohlcv_failfast_case` |
| 2026-04-02 | T35 | 新增 `load_clean_df` 資料品質整合案例並驗證 | TODO -> DONE | `validate_load_clean_df_data_quality_case` |
| 2026-04-02 | T36 | 新增 dataset wrapper CLI 契約案例並驗證 | TODO -> DONE | `validate_dataset_cli_contract_case` |
| 2026-04-02 | T37 | 新增 local regression / no-arg CLI 契約案例並驗證 | TODO -> DONE | `validate_local_regression_cli_contract_case` |
| 2026-04-02 | T38 | 新增 scanner reduced snapshot 雙跑 digest 並驗證 | TODO -> DONE | `tools/local_regression/run_chain_checks.py` |
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
| 2026-04-02 | T66 | 新增 imported validate cases vs synthetic registry formal guard 缺口 | NEW -> TODO | `tools/validate/synthetic_cases.py` |
| 2026-04-02 | T66 | 補 imported / defined validate cases 與 synthetic registry 完整一致 formal guard | TODO -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-02 | T67 | 新增主表 / `T` / `G` 完整同步 formal guard 缺口 | NEW -> TODO | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-02 | T67 | 補主表 / `T` / `G` 收斂紀錄完整同步 formal guard | TODO -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | T68 | 新增 checklist 完成映射同步缺口 | NEW -> TODO | `apps/test_suite.py` |
| 2026-04-02 | T68 | 補 checklist `DONE` 摘要缺漏自動偵測與阻擋 | TODO -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B27 | 補 top-level import cycle formal guard 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-03 | B28 | 補入核心交易模組 coverage target completeness 主表項目 | NEW -> PARTIAL | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B28 | 核心交易模組已納入 `COVERAGE_TARGETS`，主表收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B29 | 補入 critical file per-file coverage minimum gate 主表項目 | NEW -> TODO | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B29 | 核心檔 per-file coverage minimum guard 已建立，主表收斂為 DONE | TODO -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B30 | 補入 coverage threshold gradual uplift 主表項目 | NEW -> PARTIAL | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B30 | 正式 coverage 基線已提高並補 floor guard，主表收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B31 | 進場關鍵模組已納入 critical file coverage gate，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B32 | critical per-file threshold 已提升到 stage-2 正式基線，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B33 | 補上 reduced dataset member/content fingerprint gate，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/common.py` |
| 2026-04-03 | B34 | 補上 atomic write 與 replace-failure recovery contract，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/common.py` |
| 2026-04-03 | B35 | 將 test suite orchestrator modules 納入 coverage targets，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-03 | B36 | 補上 artifacts manifest sha256 contract，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/common.py` |
| 2026-04-03 | B37 | 補 synthetic registry metadata contract 後主表納入 DONE | NEW -> DONE | `tools/validate/synthetic_cases.py` |
| 2026-04-03 | B38 | 將 formal pipeline step entry wrappers 納入 coverage targets，主表收斂為 DONE | NEW -> DONE | `tools/local_regression/run_meta_quality.py` |
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
| 2026-04-03 | T108 | 新增 `T` 測試入口 delimiter-agnostic single-entry guard 並驗證 | NEW -> DONE | `validate_checklist_t_single_entry_delimiter_case` |
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
| 2026-04-04 | B54 | 專案資金規則改為全系統複利後，原單股 fixed-cap 規格退役，主表先改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_history_cases.py` |
| 2026-04-04 | B54 | 改為單股複利資金 contract 並收斂完成 | PARTIAL -> DONE | `tools/validate/synthetic_history_cases.py` |
| 2026-04-04 | B55 | 專案資金規則改為全系統複利後，原 execution-only fixed-cap 規格退役，主表先改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B55 | 改為單檔複利 parity contract 並收斂完成 | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-04 | B55 | 檢出單檔複利 parity contract 初版未覆蓋獲利後 entry budget，主表改回 PARTIAL | DONE -> PARTIAL | `core/portfolio_entries.py` |
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
| 2026-04-04 | T130 | 專案資金規則改為全系統複利後，原單股 fixed-cap synthetic case 退役，先改回 PARTIAL | DONE -> PARTIAL | `validate_synthetic_single_backtest_uses_compounding_capital_case` |
| 2026-04-04 | T130 | 改為單股複利 synthetic case 並驗證 | PARTIAL -> DONE | `validate_synthetic_single_backtest_uses_compounding_capital_case` |
| 2026-04-04 | T131 | 專案資金規則改為全系統複利後，原 execution-only fixed-cap parity contract 退役，先改回 PARTIAL | DONE -> PARTIAL | `validate_single_ticker_compounding_parity_contract_case` |
| 2026-04-04 | T131 | 改為單檔複利 parity contract 並驗證 | PARTIAL -> DONE | `validate_single_ticker_compounding_parity_contract_case` |
| 2026-04-04 | T131 | 檢出單檔複利 parity contract 初版僅覆蓋虧損側，改回 PARTIAL | DONE -> PARTIAL | `validate_single_ticker_compounding_parity_contract_case` |
| 2026-04-04 | T131 | 擴充獲利側 entry budget 的單檔複利 parity contract 並收斂完成 | PARTIAL -> DONE | `validate_single_ticker_compounding_parity_contract_case` |
| 2026-04-04 | T132 | 新增 scanner live capital contract 並驗證 | NEW -> DONE | `validate_scanner_live_capital_contract_case` |
| 2026-04-04 | T133 | 新增 optimizer interrupt export contract 並驗證 | NEW -> DONE | `validate_optimizer_interrupt_export_contract_case` |
| 2026-04-04 | T134 | 新增 score numerator option contract 並驗證 | NEW -> DONE | `validate_score_numerator_option_case` |
| 2026-04-05 | B21 | 檢出 score header 將評分模型與分子混寫在同一括號，主表先改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_display_cases.py` |
| 2026-04-05 | B21 | 將 score header 改為模型/分子分欄顯示並補契約後收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_display_cases.py` |
| 2026-04-05 | B26 | 檢出 `G` 仍可殘留退役 validator 名稱與重複 `NEW -> *`，主表改回 PARTIAL | DONE -> PARTIAL | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B26 | 補上 `G` 的 `NEW` 首次出現約束與有效 validator reference guard 後收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B26 | 檢出 `G` 同日區塊新增列後未整段重排，造成排序 guard 再次被真實 bundle 擊中，主表改回 PARTIAL | DONE -> PARTIAL | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-05 | B26 | 修正 2026-04-05 同日區塊排序並補強前後鄰列核對要求後重新收斂為 DONE | PARTIAL -> DONE | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-05 | B26 | 檢出 checklist 摘要表固定升冪排序仍缺 formal guard，主表改回 PARTIAL | DONE -> PARTIAL | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B26 | 補上 checklist 摘要表固定升冪排序 guard 後重新收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B26 | 檢出 checklist 首行固定標題仍缺 formal guard，主表改回 PARTIAL | DONE -> PARTIAL | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B26 | 補上 checklist 首行固定標題 guard 後重新收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-05 | B43 | 依新規格調整 package_zip：root bundle 不得移入 arch，主表先改回 PARTIAL | DONE -> PARTIAL | `apps/package_zip.py` |
| 2026-04-05 | B43 | 改為只歸檔非 bundle 舊 ZIP 並保留 root bundle copy 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_cli_cases.py` |
| 2026-04-05 | B45 | 檢出 `resolve_log_dir("outputs")` 先命中 generic root-dir 錯誤，未維持 outputs-root 專屬拒絕語意，主表改回 PARTIAL | DONE -> PARTIAL | `tools/local_regression/run_quick_gate.py` |
| 2026-04-05 | B45 | 調整 `resolve_log_dir` 判斷順序後重新收斂為 DONE | PARTIAL -> DONE | `core/log_utils.py` |
| 2026-04-05 | B45 | 檢出 `write_issue_log` / `build_timestamped_log_path` 仍可接受 outputs 根目錄，主表改回 PARTIAL | DONE -> PARTIAL | `core/log_utils.py` |
| 2026-04-05 | B45 | 補上 outputs-root create-path guard 後重新收斂為 DONE | PARTIAL -> DONE | `core/log_utils.py` |
| 2026-04-05 | B58 | 補 `use_compounding` unsupported-value fail-fast guard 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_guardrail_cases.py` |
| 2026-04-05 | B59 | 補關鍵 helper single-source-of-truth static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B61 | 補 policy/config coverage-target static contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B62 | 新增 checklist 首行固定標題 contract 後主表收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-05 | B62 | 檢出 T141 直接依賴巢狀 `result["extra"]` 讀取 summary 欄位，造成首行 guard synthetic validator 假失敗，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
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
| 2026-04-05 | B74 | 檢出 GUI chart workspace contract 以 `figure.axes` 長度硬編碼 overlay volume axis，跨 matplotlib backend / inset axes 註冊差異造成 synthetic 假失敗，主表先改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
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
| 2026-04-05 | T27 | 補 scanner / dashboard score header 顯示契約後收斂完成 | DONE -> PARTIAL | `validate_display_reporting_sanity_case` |
| 2026-04-05 | T27 | 補 scanner / dashboard score header 顯示契約並驗證 | PARTIAL -> DONE | validate_display_reporting_sanity_case |
| 2026-04-05 | T116 | 依新規格調整 package_zip runtime contract：root bundle 不得移入 arch，建議測試先改回 PARTIAL | DONE -> PARTIAL | `validate_package_zip_runtime_contract_case` |
| 2026-04-05 | T116 | 改為只歸檔非 bundle 舊 ZIP 並保留 root bundle copy 後重新收斂 | PARTIAL -> DONE | `validate_package_zip_runtime_contract_case` |
| 2026-04-05 | T121 | 檢出 quick_gate output-path guard 錯誤語意被 generic root-dir 檢查覆蓋，建議測試先改回 PARTIAL | DONE -> PARTIAL | `validate_quick_gate_output_path_guard_contract_case` |
| 2026-04-05 | T121 | 修正 outputs-root 專屬拒絕語意後重新收斂 | PARTIAL -> DONE | `validate_quick_gate_output_path_guard_contract_case` |
| 2026-04-05 | T121 | 檢出 quick_gate log-path contract 尚未覆蓋 outputs-root create path，建議測試先改回 PARTIAL | DONE -> PARTIAL | `validate_quick_gate_output_path_guard_contract_case` |
| 2026-04-05 | T121 | 補上 outputs-root create-path guard 並重新收斂 | PARTIAL -> DONE | `validate_quick_gate_output_path_guard_contract_case` |
| 2026-04-05 | T124 | 檢出 `G` 同日區塊新增列後未整段重排，排序 guard 再次被真實 bundle 擊中 | DONE -> PARTIAL | `validate_checklist_g_ordering_case` |
| 2026-04-05 | T124 | 修正同日區塊排序並補強前後鄰列核對流程後重新收斂 | PARTIAL -> DONE | `validate_checklist_g_ordering_case` |
| 2026-04-05 | T135 | 新增 `use_compounding=False` fail-fast guardrail 並驗證 | NEW -> DONE | `validate_use_compounding_failfast_guardrail_case` |
| 2026-04-05 | T136 | 新增關鍵 helper single-source-of-truth contract 並驗證 | NEW -> DONE | `validate_critical_helper_single_source_contract_case` |
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
| 2026-04-07 | B104 | 新增 `L` 僅作進場與 sizing / `P_fill` 首個可執行 stop-tp / 延續候選固定反事實 barrier 的細部交易契約承接後主表收斂為 DONE | NEW -> DONE | `doc/TEST_SUITE_CHECKLIST.md` |
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
| 2026-04-07 | T183 | 新增細部交易契約承接 meta contract 並驗證 `L` / `P_fill` / 延續候選 barrier / inclusive hit 語意 | NEW -> DONE | `validate_checklist_physical_trading_principles_contract_case` |
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
| 2026-04-09 | T202 | 新增 formal consistency step 完整 command string 必須列入 checklist `T` 摘要的 meta contract 並同步補齊映射 | NEW -> DONE | `tools/validate/cli.py` |
| 2026-04-09 | T203 | 新增 checklist `T` formal command string 單列單入口 contract 並驗證 | NEW -> DONE | `validate_checklist_t_formal_command_single_entry_case` |
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
| 2026-04-11 | B123 | 修正漲跌停價改依 raw-limit 價本身決定 tick band 後重新收斂為 DONE | PARTIAL -> DONE | `core/exact_accounting.py` |
| 2026-04-11 | B123 | 檢出 ETF / ETN / REIT 類商品仍沿用股票 tick ladder，主表改回 PARTIAL | DONE -> PARTIAL | `core/exact_accounting.py` |
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
| 2026-04-11 | B146 | 補齊 portfolio rotation 賣出路徑改用 `weakest_ticker` 並擴充 static contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B146 | 檢出 scalar / array 價格正規化 caller 尚未一路傳遞 ticker / security_profile，主表改回 PARTIAL | DONE -> PARTIAL | `core/price_utils.py` |
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
| 2026-04-11 | B147 | 檢出 summary comment coverage contract 仍把摘要註解開頭 wording 寫死、未依頂部 block 與 Txx 列舉穩健比對，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B147 | 將 summary comment coverage contract 改為依頂部摘要註解 block 與 T225~T237 列舉穩健比對後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B148 | 新增 exact-ledger ratio path 契約，釘死 return / rotation ratio 不得先轉 float money 再相除 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B149 | 新增 debug exit milli binding 契約，釘死 final-exit total_return_pct 不得引用未定義 total_pnl_milli | NEW -> DONE | `tools/trade_analysis/exit_flow.py` |
| 2026-04-11 | B149 | 檢出 milli-binding contract 的 forced-closeout 負向守衛仍比對錯誤舊字串，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B149 | 補齊 forced-closeout 舊 float total-pnl 路徑負向守衛後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | B150 | 新增 core R-multiple exact-ledger 契約，釘死核心回測 / 投組統計不得以 float `total_pnl / initial_risk_total` 累計 `r_mult` | NEW -> DONE | `core/backtest_core.py` |
| 2026-04-11 | B151 | 新增 GUI workbench 文件同步 static contract，並修正文檔殘留的執行摘要分頁敘述後收斂為 DONE | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-11 | T40 | 以 full dataset 額外審計檢出 portfolio export reporting synthetic case 尚未覆蓋 Plotly import failure fallback，改回 PARTIAL | DONE -> PARTIAL | `validate_portfolio_export_report_artifacts_case` |
| 2026-04-11 | T40 | 擴充 portfolio export reporting synthetic case 納入 Plotly import failure fallback artifact / traceability 後重新驗證 | PARTIAL -> DONE | `validate_portfolio_export_report_artifacts_case` |
| 2026-04-11 | T174 | 檢出 GUI extended preview continuity static contract 仍比對舊 counterfactual signature，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T174 | 將 GUI extended preview continuity static contract 同步到 `ticker` + `security_profile` signature 後重新驗證 | PARTIAL -> DONE | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T174 | 檢出 GUI extended preview continuity static contract 尚未同步 `trade_date` signature，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T174 | 將 GUI extended preview continuity static contract 同步到 `ticker` + `security_profile` + `trade_date` signature 後重新驗證 | PARTIAL -> DONE | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T174 | 檢出 GUI extended preview continuity static contract 仍比對 `trade_date=current_date`，未同步 `effective_trade_date` fallback signature，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T174 | 將 GUI extended preview continuity static contract 同步到 `trade_date=effective_trade_date` signature 後重新驗證 | PARTIAL -> DONE | `validate_gui_extended_preview_continuity_contract_case` |
| 2026-04-11 | T206 | 以 reduced dataset 實際比對檢出 exact-accounting tick/limit unit contract 尚未覆蓋跨 tick band 漲跌停價案例，改回 PARTIAL | DONE -> PARTIAL | `validate_exact_accounting_tick_limit_integer_case` |
| 2026-04-11 | T206 | 擴充 exact-accounting tick/limit unit contract 納入跨 tick band 漲跌停價案例後重新驗證 | PARTIAL -> DONE | `validate_exact_accounting_tick_limit_integer_case` |
| 2026-04-11 | T206 | 檢出 exact-accounting tick/limit unit contract 尚未覆蓋 ETF / ETN / REIT 類商品 profile 與兩級 tick 案例，改回 PARTIAL | DONE -> PARTIAL | `validate_exact_accounting_tick_limit_integer_case` |
| 2026-04-11 | T206 | 擴充 exact-accounting tick/limit unit contract 納入 ticker 自動辨識與 ETF / ETN / REIT 兩級 tick 案例後重新驗證 | PARTIAL -> DONE | `validate_exact_accounting_tick_limit_integer_case` |
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
| 2026-04-11 | T232 | 擴充 array tick-normalization static contract 納入 portfolio rotation 賣出路徑 ticker 來源契約後重新驗證 | PARTIAL -> DONE | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-11 | T232 | 檢出 array tick-normalization static contract 尚未覆蓋 ticker / security_profile 傳遞路徑，改回 PARTIAL | DONE -> PARTIAL | `validate_price_utils_array_tick_normalization_contract_case` |
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
| 2026-04-11 | T233 | 檢出 summary comment coverage contract 仍把摘要註解開頭 wording 寫死、未依頂部 block 與 T225~T237 列舉穩健比對，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T233 | 將 summary comment coverage contract 改為依頂部摘要註解 block 與 T225~T237 列舉穩健比對後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-11 | T234 | 新增 exact-ledger return ratio no-money-float-division static contract 並驗證 | NEW -> DONE | `validate_exact_ledger_return_ratio_no_money_float_division_contract_case` |
| 2026-04-11 | T235 | 新增 debug exit total-return milli binding static contract 並驗證 | NEW -> DONE | `validate_debug_exit_total_return_milli_binding_contract_case` |
| 2026-04-11 | T235 | 檢出 milli-binding contract 尚未覆蓋 forced-closeout 路徑，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_exit_total_return_milli_binding_contract_case` |
| 2026-04-11 | T235 | 擴充 milli-binding contract 納入 forced-closeout 路徑後重新驗證 | PARTIAL -> DONE | `validate_debug_exit_total_return_milli_binding_contract_case` |
| 2026-04-11 | T235 | 檢出 forced-closeout milli-binding contract 的舊 float total-pnl 負向守衛仍比對錯誤字串，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_exit_total_return_milli_binding_contract_case` |
| 2026-04-11 | T235 | 補齊 forced-closeout 舊 float total-pnl 負向守衛後重新驗證 | PARTIAL -> DONE | `validate_debug_exit_total_return_milli_binding_contract_case` |
| 2026-04-11 | T236 | 新增 core R-multiple exact-ledger static contract 並驗證 | NEW -> DONE | `validate_core_r_multiple_exact_ledger_contract_case` |
| 2026-04-11 | T237 | 新增 GUI workbench 文件同步 static contract 並驗證 | NEW -> DONE | `validate_gui_workbench_documentation_sync_case` |
| 2026-04-12 | B26 | 依 bundle 再次檢出 `G` 同日追蹤列回寫後未整段重排，排序 guard 再被真實失敗擊中，主表改回 PARTIAL | DONE -> PARTIAL | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-12 | B26 | 補上 E/T 摘要區與子標題唯一性 guard、移除重複區塊並重新驗證 | PARTIAL -> DONE | `validate_checklist_summary_section_headings_unique_case` |
| 2026-04-12 | B47 | 檢出 repo 內 `models/*best_params*.json` shipped 工件仍允許 float-schema 欄位寫成 `int`，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-12 | B47 | 擴充 model I/O schema contract 納入 shipped `champion/run_best params json` 型別一致性並同步修正 `models/champion_params.json` / `models/run_best_params.json` 後重新收斂為 DONE | PARTIAL -> DONE | `validate_model_io_schema_case` |
| 2026-04-12 | B52 | 檢出 best_params export contract 尚未釘死 search-step float canonicalization 與預設費率 decimal canonical 輸出，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-12 | B52 | 補齊 best_params export canonicalization contract 與匯出鏈後重新收斂為 DONE | PARTIAL -> DONE | `tools/optimizer/study_utils.py` |
| 2026-04-12 | B52 | 檢出 repo 內 `models/run_best_params.json` / `models/champion_params.json` 參數工件仍殘留浮點尾差，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-12 | B52 | 補齊 shipped optimizer artifacts canonical decimal contract 並同步清理 `models/run_best_params.json` / `models/champion_params.json` 後重新收斂為 DONE | PARTIAL -> DONE | `validate_optimizer_objective_export_contract_case` |
| 2026-04-12 | B52 | 檢出 shipped optimizer artifacts canonical decimal contract 尚未覆蓋 `models/champion_params.json`，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-12 | B52 | 擴充 shipped optimizer artifacts canonical decimal contract 納入 `models/champion_params.json` 後重新收斂為 DONE | PARTIAL -> DONE | `validate_optimizer_objective_export_contract_case` |
| 2026-04-12 | B66 | 檢出 workbench panel registry / inspector 仍綁定 legacy debug aliases，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-12 | B66 | 將 workbench panel registry / inspector 改為優先使用 canonical trade_analysis aliases 並補齊 formal guard 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-12 | B71 | 檢出 GUI embedded chart contract 仍比對舊 debug chart alias，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-12 | B71 | 將 GUI embedded chart contract 與 checklist 摘要同步到 canonical trade chart alias 後重新驗證 | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-12 | B112 | 檢出 GUI TclError fallback meta contract 掃描路徑誤指 `tools/gui`，guard 在空集合上 vacuous pass，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B112 | 修正為掃描 `tools/workbench_ui/*.py` 並要求目標檔案非空後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B114 | 檢出 GUI 買入資訊框缺少實支且 T193 contract 誤寫成禁止實支，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-12 | B114 | 補上買入資訊框實支最後一行與 sidebar 同步 contract 後主表恢復 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-12 | B114 | 檢出 single_stock_inspector sidebar 交易資訊仍缺少實支欄位綁定，改回 PARTIAL | DONE -> PARTIAL | `tools/workbench_ui/single_stock_inspector.py` |
| 2026-04-12 | B114 | 補上 sidebar 實支欄位與 hover snapshot 綁定 contract 後主表恢復 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-05-02 | B114 | 補上投組 missed-only ticker 下拉追查、錯失事件 K 線 marker、錯失列預留 / 實支分流、單檔 missed / 資金效率摘要與單股 fixed_risk UI contract 後主表維持 DONE | DONE -> DONE | `tools/validate/synthetic_contract_cases.py`, `tools/workbench_ui/portfolio_backtest_inspector.py`, `tools/workbench_ui/single_stock_inspector.py`, `core/portfolio_entries.py`, `core/portfolio_exits.py` |
| 2026-04-12 | B124 | 檢出 `build_backtest_stats()` 的 normal preview 仍只靠 active signal 解析 `security_profile`，empty/final caller 也未明確傳入商品 profile，改回 PARTIAL | DONE -> PARTIAL | `core/backtest_finalize.py` |
| 2026-04-12 | B124 | 補齊 `build_backtest_stats()` 的 `security_profile` 簽名、empty/final caller 傳遞與 normal / extended preview 商品 profile contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B126 | 檢出 debug backtest 的買訊預覽、latest raw preview 與 latest extended preview 尚未同步傳遞 `security_profile`，改回 PARTIAL | DONE -> PARTIAL | `tools/trade_analysis/backtest.py` |
| 2026-04-12 | B126 | 補齊 debug backtest entry flow 與全部 preview 路徑的 `security_profile` 傳遞 contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B146 | 檢出單股 backtest normal signal / entry 與 portfolio candidate builder 尚未一路傳遞 `security_profile`，改回 PARTIAL | DONE -> PARTIAL | `core/backtest_core.py` |
| 2026-04-12 | B146 | 補齊 backtest / packed fast-data / portfolio candidate 的 `security_profile` 傳遞與 static contract 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 最嚴格檢查檢出正式入口摘要註解尚未同步新增 T248 與 `conservative-executable-exit interpretation` 主題，改回 PARTIAL | DONE -> PARTIAL | `apps/test_suite.py` |
| 2026-04-12 | B147 | 補齊正式入口摘要註解與 summary meta contract 對 T248 / 新主題的同步後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 檢出 summary comment coverage contract 尚未同步新增的 T238，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 擴充 summary comment coverage contract 納入 T238 後重新驗證 | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 檢出 summary comment coverage contract 尚未同步新增的 T239，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 擴充 summary comment coverage contract 納入 T239 後重新驗證 | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 檢出 summary comment coverage contract 尚未同步新增的 T240，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 擴充 summary comment coverage contract 納入 T240 後重新驗證 | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 檢出 summary comment coverage contract 尚未同步新增的 T241，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 擴充 summary comment coverage contract 納入 T241 後重新驗證 | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 檢出 summary comment coverage contract 與正式入口摘要尚未同步新增的 T242，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 將 summary comment coverage contract、正式入口摘要與 checklist 映射同步納入 T242 後重新驗證 | PARTIAL -> DONE | `apps/test_suite.py` |
| 2026-04-12 | B147 | 檢出 summary comment coverage contract 尚未同步新增的 T243，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 擴充 summary comment coverage contract 納入 T243 後重新驗證 | PARTIAL -> DONE | `apps/test_suite.py` |
| 2026-04-12 | B147 | 檢出 summary comment coverage contract 尚未同步新增的 T244，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 擴充 summary comment coverage contract 納入 T244 後重新驗證 | PARTIAL -> DONE | `apps/test_suite.py` |
| 2026-04-12 | B147 | 最嚴格檢查檢出正式入口摘要註解尚未同步新增的 T247 與 `checklist-summary-heading-uniqueness` 主題，改回 PARTIAL | DONE -> PARTIAL | `apps/test_suite.py` |
| 2026-04-12 | B147 | 擴充正式入口摘要註解與 summary meta contract 納入 T247 與 checklist 摘要標題唯一性主題後重新驗證 | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 最嚴格檢查檢出正式入口摘要註解雖持續補尾端新增 `Txx`，但仍漏列同主題既有 `T227/T228` 與 `portfolio-rotation-return` / `validator-oracle exact-ledger` 主題，改回 PARTIAL | DONE -> PARTIAL | `apps/test_suite.py` |
| 2026-04-12 | B147 | 將正式入口摘要註解與 summary meta contract 回補 `T227/T228` 與對應主題後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 依使用者要求將 `apps/test_suite.py` 頂部摘要註解排除出本地 formal 驗證範圍，主表改為 N/A | DONE -> N/A | `apps/test_suite.py` |
| 2026-04-12 | B147 | bundle 檢出 summary comment compatibility stub 雖已改列 N/A，但 meta-registry completeness guard 仍將其視為必須註冊 validator，主表改回 PARTIAL | N/A -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B147 | 將 summary comment compatibility stub 同步排除於 meta-registry completeness guard 後重新回復 N/A | PARTIAL -> N/A | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B152 | 新增 trade_analysis rename 相容文件契約並驗證 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B153 | 新增 trade_analysis canonical alias export 契約並驗證 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B154 | 新增 synthetic_contract_cases shared path helper 契約並驗證 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B155 | 新增 checklist G transition-chain contract 並驗證 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B156 | 新增單股 backtest public stats 盈虧口徑一致性契約並驗證 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B157 | 新增正式入口 help 摘要同步契約並驗證 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B157 | 最嚴格檢查檢出正式入口 `--help` 長說明尚未同步新增的 checklist 摘要標題唯一性主題，改回 PARTIAL | DONE -> PARTIAL | `apps/test_suite.py` |
| 2026-04-12 | B157 | 擴充正式入口 help 摘要與對應 meta contract 納入 `summary-section-heading-uniqueness` 後重新驗證 | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B157 | 最嚴格檢查檢出正式入口 `--help` 長說明尚未同步新增 `conservative-executable-exit interpretation contract` 主題，改回 PARTIAL | DONE -> PARTIAL | `apps/test_suite.py` |
| 2026-04-12 | B157 | 補齊正式入口 `--help` 長說明與 help meta contract 對新主題的同步後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B158 | 新增正式入口 help 舊 wording / 裸用詞排除契約並驗證 | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-12 | B161 | 新增同一事件保守可執行解讀契約，補 formal case 明確覆蓋 stop/tp 歧義、gap-to-open 與 deferred stop 次日開盤執行 | NEW -> DONE | `validate_synthetic_conservative_executable_exit_interpretation_case` |
| 2026-04-12 | B162 | 最嚴格檢查檢出 `doc/ARCHITECTURE.md` apps 檔案樹漏列 `apps/workbench.py`，補上 file-tree sync contract 與文件列後收斂為 DONE | NEW -> DONE | `validate_architecture_workbench_entry_file_tree_sync_case` |
| 2026-04-12 | B163 | 新增 ARCHITECTURE models 檔案樹 shipped champion 工件同步契約並驗證 | NEW -> DONE | `validate_architecture_models_champion_params_file_tree_sync_case` |
| 2026-04-12 | B164 | 新增 ARCHITECTURE shipped helper modules 檔案樹同步契約並驗證 | NEW -> DONE | `doc/ARCHITECTURE.md` |
| 2026-04-12 | B165 | 新增 ARCHITECTURE shipped support modules 檔案樹同步契約並驗證 | NEW -> DONE | `doc/ARCHITECTURE.md` |
| 2026-04-12 | B166 | 新增 ARCHITECTURE Local Regression `run_meta_quality.py` 檔案樹同步契約並驗證 | NEW -> DONE | `validate_architecture_local_regression_meta_quality_file_tree_sync_case` |
| 2026-04-12 | T14 | 檢出 model I/O schema contract 尚未覆蓋 repo shipped `champion/run_best params json` 的 float-schema 型別一致性，改回 PARTIAL | DONE -> PARTIAL | `validate_model_io_schema_case` |
| 2026-04-12 | T14 | 擴充 model I/O schema contract 納入 shipped `champion/run_best params json` 型別檢查並重新驗證 | PARTIAL -> DONE | `validate_model_io_schema_case` |
| 2026-04-12 | T105 | 檢出 optimizer objective / export contract case 尚未釘死 search-step float canonicalization 與預設費率 decimal canonical 輸出，改回 PARTIAL | DONE -> PARTIAL | `validate_optimizer_objective_export_contract_case` |
| 2026-04-12 | T105 | 擴充 optimizer objective / export contract case 納入 canonicalization 檢查後重新驗證 | PARTIAL -> DONE | `validate_optimizer_objective_export_contract_case` |
| 2026-04-12 | T105 | 檢出 optimizer objective / export contract case 尚未覆蓋 repo 內 `models/run_best_params.json` / `models/champion_params.json` 參數工件 canonical decimal，改回 PARTIAL | DONE -> PARTIAL | `validate_optimizer_objective_export_contract_case` |
| 2026-04-12 | T105 | 擴充 optimizer objective / export contract case 納入 shipped optimizer artifacts canonical decimal 檢查後重新驗證 | PARTIAL -> DONE | `validate_optimizer_objective_export_contract_case` |
| 2026-04-12 | T105 | 檢出 optimizer objective / export contract case 尚未覆蓋 `models/champion_params.json` shipped artifact canonical decimal，改回 PARTIAL | DONE -> PARTIAL | `validate_optimizer_objective_export_contract_case` |
| 2026-04-12 | T105 | 擴充 optimizer objective / export contract case 納入 `models/champion_params.json` canonical decimal 檢查後重新驗證 | PARTIAL -> DONE | `validate_optimizer_objective_export_contract_case` |
| 2026-04-12 | T124 | 依 bundle 再次檢出 `G` 同日追蹤列回寫後未整段重排，排序 guard 再被真實失敗擊中 | DONE -> PARTIAL | `validate_checklist_g_ordering_case` |
| 2026-04-12 | T145 | 檢出 GUI workbench contract 尚未禁止 panel registry / inspector 使用 legacy debug aliases，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_workbench_contract_case` |
| 2026-04-12 | T145 | 擴充 GUI workbench contract 納入 canonical trade_analysis alias 偏好後重新驗證 | PARTIAL -> DONE | `validate_gui_workbench_contract_case` |
| 2026-04-12 | T150 | 檢出 GUI embedded chart contract 仍比對舊 debug chart alias，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_embedded_chart_contract_case` |
| 2026-04-12 | T150 | 將 GUI embedded chart contract 同步到 canonical trade chart alias 後重新驗證 | PARTIAL -> DONE | `validate_gui_embedded_chart_contract_case` |
| 2026-04-12 | T191 | 檢出 GUI TclError fallback validator 掃描舊 `tools/gui` 路徑，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_tcl_fallback_traceability_contract_case` |
| 2026-04-12 | T191 | 修正為掃描 `tools/workbench_ui/*.py` 並要求目標檔案非空後重新驗證 | PARTIAL -> DONE | `validate_gui_tcl_fallback_traceability_contract_case` |
| 2026-04-12 | T193 | 檢出 GUI trade-box contract 誤把買入框實支視為禁止項，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_trade_box_capital_and_round_trip_contract_case` |
| 2026-04-12 | T193 | 補上買入資訊框實支最後一行 contract 並重新驗證 | PARTIAL -> DONE | `validate_gui_trade_box_capital_and_round_trip_contract_case` |
| 2026-04-12 | T193 | 檢出 GUI trade-box contract 尚未釘死 sidebar 實支欄位與 hover snapshot 綁定，改回 PARTIAL | DONE -> PARTIAL | `validate_gui_trade_box_capital_and_round_trip_contract_case` |
| 2026-04-12 | T193 | 補上 sidebar 實支欄位與 hover snapshot 綁定 contract 並重新驗證 | PARTIAL -> DONE | `validate_gui_trade_box_capital_and_round_trip_contract_case` |
| 2026-04-12 | T210 | 檢出單股 backtest stats legacy schema static contract 尚未覆蓋 `security_profile` 簽名與 normal preview caller-threading，改回 PARTIAL | DONE -> PARTIAL | `validate_single_backtest_stats_legacy_schema_contract_case` |
| 2026-04-12 | T210 | 擴充單股 backtest stats legacy schema static contract 納入 `security_profile` 簽名、caller 與 preview threading 後重新驗證 | PARTIAL -> DONE | `validate_single_backtest_stats_legacy_schema_contract_case` |
| 2026-04-12 | T212 | 檢出 debug backtest entry cash-path static contract 尚未覆蓋買訊預覽與 latest preview 的 `security_profile` 傳遞，改回 PARTIAL | DONE -> PARTIAL | `validate_debug_backtest_entry_cash_path_contract_case` |
| 2026-04-12 | T212 | 擴充 debug backtest entry cash-path static contract 納入 buy-signal / latest raw / latest extended preview 的 `security_profile` threading 後重新驗證 | PARTIAL -> DONE | `validate_debug_backtest_entry_cash_path_contract_case` |
| 2026-04-12 | T232 | 檢出 array tick-normalization static contract 尚未覆蓋 backtest / packed fast-data / portfolio candidate 的 `security_profile` 傳遞，改回 PARTIAL | DONE -> PARTIAL | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-12 | T232 | 擴充 array tick-normalization static contract 納入 backtest / packed fast-data / portfolio candidate 的 `security_profile` threading 後重新驗證 | PARTIAL -> DONE | `validate_price_utils_array_tick_normalization_contract_case` |
| 2026-04-12 | T233 | 檢出 summary comment coverage contract 尚未同步新增的 T238，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 擴充 summary comment coverage contract 納入 T238 後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 檢出 summary comment coverage contract 尚未同步新增的 T239，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 擴充 summary comment coverage contract 納入 T239 後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 檢出 summary comment coverage contract 尚未同步新增的 T240，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 擴充 summary comment coverage contract 納入 T240 後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 檢出 summary comment coverage contract 尚未同步新增的 T241，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 擴充 summary comment coverage contract 納入 T241 後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 檢出 summary comment coverage contract 尚未同步新增的 T242，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 擴充 summary comment coverage contract 納入 T242 後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 檢出 summary comment coverage contract 尚未同步新增的 T243，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 擴充 summary comment coverage contract 納入 T243 後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 檢出 summary comment coverage contract 尚未同步新增的 T244，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 擴充 summary comment coverage contract 納入 T244 後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 檢出 summary comment coverage contract 尚未同步新增的 T247 與 checklist 摘要標題唯一性主題，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 擴充 summary comment coverage contract 納入 T247 與 checklist 摘要標題唯一性主題後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 最嚴格檢查檢出 summary comment coverage contract 尚未同步新增 T248 與 `conservative-executable-exit interpretation` 主題，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 擴充 summary comment coverage contract 納入 T248 與新主題後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 最嚴格檢查檢出 summary comment coverage contract 雖持續補尾端新增 `Txx`，但仍漏列同主題既有 `T227/T228` 與 `portfolio-rotation-return` / `validator-oracle exact-ledger` 主題，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 擴充 summary comment coverage contract 回補 `T227/T228` 與對應主題後重新驗證 | PARTIAL -> DONE | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 依使用者要求將 test_suite 註解排除出本地 formal 驗證範圍，建議測試改為 N/A | DONE -> N/A | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | bundle 檢出 summary comment compatibility stub 雖已改列 N/A，但 meta-registry completeness guard 仍將其視為必須註冊 validator，改回 PARTIAL | N/A -> PARTIAL | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T233 | 將 summary comment compatibility stub 同步排除於 meta-registry completeness guard 後重新回復 N/A | PARTIAL -> N/A | `validate_test_suite_summary_comment_covers_latest_exact_contract_ids_case` |
| 2026-04-12 | T238 | 新增 trade_analysis rename 相容文件契約並驗證 | NEW -> DONE | `validate_trade_analysis_legacy_naming_documentation_contract_case` |
| 2026-04-12 | T239 | 新增 trade_analysis canonical alias export 契約並驗證 | NEW -> DONE | `validate_trade_analysis_canonical_alias_export_contract_case` |
| 2026-04-12 | T240 | 新增 synthetic_contract_cases shared path helper 契約並驗證 | NEW -> DONE | `validate_synthetic_contract_cases_project_root_path_helper_contract_case` |
| 2026-04-12 | T241 | 新增 checklist G transition-chain contract 並驗證 | NEW -> DONE | `validate_checklist_g_transition_sequence_case` |
| 2026-04-12 | T242 | 新增單股 backtest public stats 盈虧口徑一致性契約並驗證 | NEW -> DONE | `validate_single_backtest_public_profit_equity_consistency_contract_case` |
| 2026-04-12 | T243 | 新增正式入口 help 摘要同步契約並驗證 | NEW -> DONE | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-12 | T243 | 最嚴格檢查檢出正式入口 help 穩定主題 token 契約尚未覆蓋摘要標題唯一性主題，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-12 | T243 | 擴充正式入口 help 穩定主題 token 契約納入摘要標題唯一性主題後重新驗證 | PARTIAL -> DONE | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-12 | T243 | 最嚴格檢查檢出 help 穩定主題 token 契約尚未同步新增「保守可執行出場解讀」主題，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-12 | T243 | 擴充 help 穩定主題 token 契約納入新主題後重新驗證 | PARTIAL -> DONE | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-12 | T244 | 新增正式入口 help 舊 wording / 裸用詞排除契約並驗證 | NEW -> DONE | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-12 | T247 | 新增 checklist E/T 摘要區唯一性契約並驗證 | NEW -> DONE | `validate_checklist_summary_section_headings_unique_case` |
| 2026-04-12 | T248 | 新增同一事件保守可執行解讀 synthetic case，明確驗證同棒 stop/tp 歧義、gap-to-open 與 deferred stop 次日開盤執行 | NEW -> DONE | `validate_synthetic_conservative_executable_exit_interpretation_case` |
| 2026-04-12 | T249 | 新增 ARCHITECTURE apps 檔案樹 workbench 入口同步契約並驗證 | NEW -> DONE | `validate_architecture_workbench_entry_file_tree_sync_case` |
| 2026-04-12 | T250 | 新增 ARCHITECTURE models 檔案樹 shipped champion 工件同步契約並驗證 | NEW -> DONE | `validate_architecture_models_champion_params_file_tree_sync_case` |
| 2026-04-12 | T251 | 新增 ARCHITECTURE shipped helper modules 檔案樹同步契約並驗證 | NEW -> DONE | `doc/ARCHITECTURE.md` |
| 2026-04-12 | T252 | 新增 ARCHITECTURE shipped support modules 檔案樹同步契約並驗證 | NEW -> DONE | `doc/ARCHITECTURE.md` |
| 2026-04-12 | T253 | 新增 ARCHITECTURE Local Regression `run_meta_quality.py` 檔案樹同步契約並驗證 | NEW -> DONE | `validate_architecture_local_regression_meta_quality_file_tree_sync_case` |
| 2026-04-13 | B11 | 最嚴格檢查檢出 `meta_quality_summary.json` 的 `formal_entry` nested schema contract 尚未同步目前 `run_meta_quality.py` 輸出鍵，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-13 | B11 | 補齊 `registry_steps / registry_commands / run_all_steps / preflight_steps / test_suite_steps` 並排除已退役的 legacy `steps` 舊鍵後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-13 | B11 | 最嚴格檢查檢出 output contract 仍以退役欄位名直掛 active sub-check 與 checklist 摘要，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-13 | B11 | 將 active sub-check 與 checklist 摘要改為中性「退役舊鍵排除」語意後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-13 | B11 | 最嚴格檢查檢出 `formal_entry` stale-key 排除檢查實際誤檢 `project_settings_steps`、未直接禁止 legacy `steps` 舊鍵，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-13 | B11 | 將 `formal_entry` stale-key 排除檢查改為直接禁止 legacy `steps` 舊鍵後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-13 | B20 | 將 `doc/CMD.md` shipped 指令契約併入 B16，B20 改列歷史索引 | DONE -> N/A | `validate_cmd_document_contract_case` |
| 2026-04-13 | B26 | 最嚴格檢查檢出 `G` 同日區塊未重排導致 `B158` 落在 `T124` 後方，主表改回 PARTIAL | DONE -> PARTIAL | `tools/local_regression/run_meta_quality.py` |
| 2026-04-13 | B26 | 依日期與 tracking ID 重新排序 `G` 同日區塊並補強交付前整段重排 guard 後重新收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-13 | B26 | 依 bundle 實際失敗再次檢出 `G` 同日區塊新增較小 tracking ID 後未整段重排，排序 guard 再被真實失敗擊中，主表改回 PARTIAL | DONE -> PARTIAL | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-13 | B26 | 將 2026-04-13 同日區塊抽出後依日期與 tracking ID 穩定重排並重新收斂為 DONE | PARTIAL -> DONE | `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-13 | B28 | 將 key coverage targets completeness 併入 B22，B28 改列歷史索引 | DONE -> N/A | `validate_core_trading_modules_in_coverage_targets_case` |
| 2026-04-13 | B29 | 將 critical files per-file coverage minimum gate 併入 B22，B29 改列歷史索引 | DONE -> N/A | `validate_critical_file_coverage_minimum_gate_case` |
| 2026-04-13 | B50 | 最嚴格檢查檢出 strategy minimum viability smoke 尚未釘死 scanner summary canonical issue-log path，仍可回流退役 `outputs/scanner/` 類別，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-13 | B50 | 將 scanner summary smoke contract 收斂為 canonical `outputs/vip_scanner/` issue-log path 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-13 | B62 | 將 checklist 首行固定標題 exact-string guard 降級為非必要 wording hygiene，改列 N/A | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | B63 | 將 synthetic meta validator flattened-payload 讀取 hygiene 退出 formal blocker | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | B64 | 將 checklist 摘要表排序獨立主表契約併入 B26，改列歷史索引 | DONE -> N/A | `validate_checklist_summary_tables_sorted_by_id_case` |
| 2026-04-13 | B72 | 將 synthetic validator alias import hygiene 退出 formal blocker | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | B73 | 將 synthetic validator alias 掃描 implementation hygiene 退出 formal blocker | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | B76 | 將 synthetic validator chart/navigation helper import hygiene 退出 formal blocker | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | B99 | 將 synthetic validator 非 error-path 參數字面值 hygiene 退出 formal blocker | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | B100 | 將 synthetic_meta_cases shared path helper import hygiene 退出 formal blocker | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | B107 | 將 synthetic contract literal chart payload `x` 欄位 hygiene 退出 formal blocker | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | B152 | 最嚴格檢查檢出 trade_analysis legacy output dir 文件契約在 `doc/CMD.md` 與 `doc/ARCHITECTURE.md` 的 retention / output 區仍殘留裸 `debug_trade_log` 目錄名，與 canonical `outputs/debug_trade_log/` 路徑分叉，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B152 | 將 trade_analysis legacy output dir 文件契約與對應 validator 收斂為一律使用完整 `outputs/debug_trade_log/` canonical 路徑後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B154 | 將 synthetic_contract_cases shared path helper 使用 hygiene 退出 formal blocker | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | B157 | 最嚴格檢查檢出正式入口 `--help` 穩定主題摘要鏈雖宣稱已納入 `summary-section-heading-uniqueness`，但 help 與對應 meta contract 皆未實際覆蓋，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B157 | 將正式入口 help 摘要與對應 meta contract 同步補上 `summary-section-heading-uniqueness` theme token 後重新收斂為 DONE | PARTIAL -> DONE | `apps/test_suite.py` |
| 2026-04-13 | B157 | 最嚴格檢查檢出正式入口 help 穩定主題契約名義上驗 `--help`、實際卻掃 source 內單行 `print` 字串，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B157 | 將正式入口 help 穩定主題契約改為直接擷取 `apps/test_suite.py --help` 輸出後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B157 | 最嚴格檢查檢出正式入口 `--help` 穩定主題摘要仍將 checklist heading / 治理 guard 等內部 meta 細節列為必備 token，超出對使用者仍有意義的正式入口邊界，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B157 | 將正式入口 help 穩定主題摘要收斂為交易口徑一致性、保守出場解讀與 `debug-backtest` 現金路徑等使用者可理解的穩定主題後重新收斂為 DONE | PARTIAL -> DONE | `apps/test_suite.py` |
| 2026-04-13 | B158 | 最嚴格檢查檢出 help 舊 wording / 裸用詞排除契約仍以 `exact-contract` 直掛 active sub-check 與主表摘要，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 將 active sub-check 與主表摘要改為中性「舊契約 wording」語意後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 最嚴格檢查檢出 stale-wording 排除檢查仍殘留較長完整舊短語，未收斂到較短已足夠覆蓋的最小必要片段，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 將 stale-wording 排除檢查收斂為較短已足夠覆蓋的最小必要片段後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 最嚴格檢查檢出 help 舊 wording 排除檢查仍在同一 validator 內保留被 `checklist-sort-guard` 已覆蓋的較長舊短語，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 將 help 舊 wording 排除檢查去除重疊長舊短語、收斂為不重複的最小必要片段後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 最嚴格檢查檢出 bare `checklist` 排除契約實際只硬掛 `checklist-sort-guard` 舊短語，未在允許 canonical `TEST_SUITE_CHECKLIST` 的前提下直接檢查其餘 bare term，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 將 bare `checklist` 排除檢查改為先白名單 canonical `TEST_SUITE_CHECKLIST`、再直接檢查剩餘 bare term 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 最嚴格檢查檢出 help 舊 wording / 裸用詞排除契約的 active sub-check 名稱仍直接掛被排除的完整 legacy fragment，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 將 help 舊 wording / 裸用詞排除契約的 active sub-check 名稱收斂為中性 stale-wording 類型後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 最嚴格檢查檢出 help 舊 wording / 裸用詞排除契約對舊 contract-listing 排除僅覆蓋單一 exemplar、未同步覆蓋同鏈已退役的 `conservative-executable-exit interpretation contract`，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 將 help 舊 wording / 裸用詞排除契約擴充為逐一覆蓋同鏈已退役 exact-contract wording 後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 最嚴格檢查檢出 help 排除契約已擴張到 validator 命名與補丁式 wording hygiene，超出正式入口 help 成品層邊界，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 將正式範圍收斂為僅驗 `apps/test_suite.py --help` 的穩定主題與 bare `checklist` 排除後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 最嚴格檢查檢出 help 舊 wording / 裸用詞排除契約名義上驗 `--help`、實際卻掃 source 內單行 `print` 字串，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 將 help 舊 wording / 裸用詞排除契約改為直接擷取 `apps/test_suite.py --help` 輸出後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 最嚴格檢查檢出 help 裸用詞排除契約雖已移除 `TEST_SUITE_CHECKLIST` token，卻仍保留 canonical 白名單特例說明，與正式入口成品層需求不再一致，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 將 help 裸用詞排除契約收斂為直接禁止 bare `checklist` 與 `contract` 長列舉後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-13 | B158 | 將 help 舊 wording / 裸用詞排除契約退出 formal blocker，改由 B157 穩定主題正向契約承接 | DONE -> N/A | `apps/test_suite.py` |
| 2026-04-13 | B164 | 最嚴格檢查檢出 `doc/ARCHITECTURE.md` 仍把 internal helper 模組 exact file-tree 列舉升格為長期 formal contract，與穩定子系統 / 正式入口邊界不一致，改列 N/A | DONE -> N/A | `doc/ARCHITECTURE.md` |
| 2026-04-13 | B165 | 最嚴格檢查檢出 `doc/ARCHITECTURE.md` 仍把 internal support 模組 exact file-tree 列舉升格為長期 formal contract，與穩定子系統 / 正式入口邊界不一致，改列 N/A | DONE -> N/A | `doc/ARCHITECTURE.md` |
| 2026-04-13 | B166 | 依文件瘦身方向將 ARCHITECTURE local_regression exact file-tree 同步契約退出 long-term formal scope，改列 N/A | DONE -> N/A | `doc/ARCHITECTURE.md` |
| 2026-04-13 | T18 | 最嚴格檢查檢出 output contract case 尚未覆蓋 `meta_quality_summary.json` `formal_entry` nested schema required keys 與 stale-key 排除檢查，改回 PARTIAL | DONE -> PARTIAL | `validate_output_contract_case` |
| 2026-04-13 | T18 | 擴充 output contract case，補齊 `meta_quality_summary.json` `formal_entry` nested schema required keys 與 stale-key 排除檢查後重新收斂為 DONE | PARTIAL -> DONE | `validate_output_contract_case` |
| 2026-04-13 | T18 | 最嚴格檢查檢出 stale-key 排除檢查實際誤檢 `project_settings_steps`、未直接覆蓋 legacy `steps` 舊鍵，改回 PARTIAL | DONE -> PARTIAL | `validate_output_contract_case` |
| 2026-04-13 | T18 | 將 stale-key 排除檢查改為直接覆蓋 legacy `steps` 舊鍵後重新收斂為 DONE | PARTIAL -> DONE | `validate_output_contract_case` |
| 2026-04-13 | T107 | 依 formal 瘦身將 `G` 備註欄 single-entry delimiter hygiene guard 退出正式長期 test suite，改列 N/A | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | T110 | 將 legacy `D` 區回流 guard 自 formal blocker 退役 | DONE -> N/A | `validate_checklist_no_legacy_d_section_case` |
| 2026-04-13 | T124 | 補齊 checklist `G` 最新狀態與 `T` DONE 摘要的重新收斂紀錄，避免 done/unfinished 摘要與 convergence 狀態分叉 | PARTIAL -> DONE | `validate_checklist_g_ordering_case` |
| 2026-04-13 | T124 | 最嚴格檢查檢出 `G` 同日區塊排序斷裂並命中 `checklist_g_rows_sorted_by_date_then_id`，改回 PARTIAL | DONE -> PARTIAL | `validate_checklist_g_ordering_case` |
| 2026-04-13 | T124 | 將 2026-04-13 同日區塊依日期與 tracking ID 重排後重新收斂為 DONE | PARTIAL -> DONE | `validate_checklist_g_ordering_case` |
| 2026-04-13 | T124 | 依 bundle 實際失敗再次檢出 `G` 同日區塊新增較小 tracking ID 後未整段重排，排序 guard 再被真實失敗擊中 | DONE -> PARTIAL | `validate_checklist_g_ordering_case` |
| 2026-04-13 | T124 | 將 2026-04-13 同日區塊抽出後依日期與 tracking ID 穩定重排並重新收斂 | PARTIAL -> DONE | `validate_checklist_g_ordering_case` |
| 2026-04-13 | T125 | 將 legacy `F1` 區回流 guard 自 formal blocker 退役 | DONE -> N/A | `validate_checklist_no_legacy_f1_section_case` |
| 2026-04-13 | T127 | 最嚴格檢查檢出 strategy minimum viability case 尚未直接覆蓋 scanner summary canonical issue-log path，改回 PARTIAL | DONE -> PARTIAL | `validate_strategy_minimum_viability_case` |
| 2026-04-13 | T127 | 擴充 scanner summary smoke contract，釘死 canonical `outputs/vip_scanner/` issue-log path 後重新收斂為 DONE | PARTIAL -> DONE | `validate_strategy_minimum_viability_case` |
| 2026-04-13 | T140 | 依 formal 瘦身將 `G` 備註欄 validator reference existence hygiene guard 退出正式長期 test suite，改列 N/A | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | T141 | 將 checklist 首行固定標題 exact-string guard 自 formal blocker 退役 | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | T142 | 將 synthetic meta summary-value accessor hygiene guard 自 formal blocker 退役 | DONE -> N/A | `validate_synthetic_meta_cases_summary_value_accessor_contract_case` |
| 2026-04-13 | T151 | 將 synthetic validator alias import hygiene guard 自 formal blocker 退役 | DONE -> N/A | `validate_synthetic_case_numpy_alias_import_contract_case` |
| 2026-04-13 | T152 | 將 synthetic validator alias 掃描 implementation hygiene guard 自 formal blocker 退役 | DONE -> N/A | `validate_synthetic_case_numpy_alias_scan_ignores_string_literals_contract_case` |
| 2026-04-13 | T155 | 將 synthetic validator chart/navigation helper import hygiene guard 自 formal blocker 退役 | DONE -> N/A | `validate_synthetic_case_chart_navigation_binder_import_contract_case` |
| 2026-04-13 | T178 | 將 synthetic validator 非 error-path 參數字面值 hygiene guard 自 formal blocker 退役 | DONE -> N/A | `validate_synthetic_case_non_error_initial_capital_contract_case` |
| 2026-04-13 | T179 | 將 synthetic_meta_cases shared path helper import hygiene guard 自 formal blocker 退役 | DONE -> N/A | `validate_synthetic_meta_cases_build_project_absolute_path_import_contract_case` |
| 2026-04-13 | T186 | 將 synthetic contract literal chart payload `x` 欄位 hygiene guard 自 formal blocker 退役 | DONE -> N/A | `validate_synthetic_case_normalize_chart_payload_literal_x_contract_case` |
| 2026-04-13 | T238 | 最嚴格檢查檢出 trade_analysis legacy naming 文件契約在 retention / output 區仍以裸 `debug_trade_log` 目錄名承接相容輸出，與 canonical `outputs/debug_trade_log/` 路徑分叉，改回 PARTIAL | DONE -> PARTIAL | `validate_trade_analysis_legacy_naming_documentation_contract_case` |
| 2026-04-13 | T238 | 將 trade_analysis legacy naming 文件契約與 `doc/CMD.md` / `doc/ARCHITECTURE.md` 一起收斂為完整 `outputs/debug_trade_log/` canonical 路徑後重新收斂為 DONE | PARTIAL -> DONE | `validate_trade_analysis_legacy_naming_documentation_contract_case` |
| 2026-04-13 | T240 | 將 synthetic_contract_cases shared path helper 使用 hygiene guard 自 formal blocker 退役 | DONE -> N/A | `validate_synthetic_contract_cases_project_root_path_helper_contract_case` |
| 2026-04-13 | T243 | 最嚴格檢查檢出正式入口 help 穩定主題 token 契約雖宣稱已納入 `summary-section-heading-uniqueness`，但 active validator 與 help 皆未實際覆蓋，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-13 | T243 | 將正式入口 help 穩定主題 token 契約與 help 說明同步補上 `summary-section-heading-uniqueness` 後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-13 | T243 | 最嚴格檢查檢出穩定主題 token validator 名義上驗 `--help`、實際卻掃 source 內單行 `print` 字串，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-13 | T243 | 將穩定主題 token validator 改為直接擷取 `apps/test_suite.py --help` 輸出後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-13 | T243 | 最嚴格檢查檢出正式入口 help 穩定主題 token 仍把 checklist heading / 治理 guard 等內部 meta 細節當成必備輸出，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-13 | T243 | 將正式入口 help 穩定主題 token 收斂為交易口徑一致性、保守出場解讀與 `debug-backtest` 現金路徑後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_mentions_stable_theme_tokens_case` |
| 2026-04-13 | T244 | 最嚴格檢查檢出 help 舊 wording / 裸用詞排除契約的 active sub-check 名稱仍殘留 `exact-contract` 舊語意，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 將 active sub-check 名稱改為中性「舊契約 wording 列舉」語意後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 最嚴格檢查檢出 help 舊 wording / 裸用詞排除契約的負向檢查仍硬掛較長完整舊短語，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 將負向檢查收斂為較短已足夠覆蓋的最小必要片段後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 最嚴格檢查檢出 help 舊 wording / 裸用詞排除契約仍在 sibling check 內保留被 `checklist-sort-guard` 已覆蓋的較長舊短語，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 將 help 舊 wording / 裸用詞排除契約去除重疊長舊短語、收斂為不重複的最小必要片段後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 最嚴格檢查檢出 bare `checklist` 排除契約實際只硬掛 `checklist-sort-guard` 舊短語，未在允許 canonical `TEST_SUITE_CHECKLIST` 的前提下直接檢查其餘 bare term，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 將 bare `checklist` 排除契約改為先白名單 canonical `TEST_SUITE_CHECKLIST`、再直接檢查剩餘 bare term 後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 最嚴格檢查檢出 help 舊 wording / 裸用詞排除契約的 active sub-check 名稱仍直接掛被排除的完整 legacy fragment，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 將 help 舊 wording / 裸用詞排除契約的 active sub-check 名稱收斂為中性 stale-wording 類型後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 最嚴格檢查檢出 help 舊 wording / 裸用詞排除契約對舊 contract-listing 排除僅覆蓋單一 exemplar、未同步覆蓋同鏈已退役的 `conservative-executable-exit interpretation contract`，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 將 help 舊 wording / 裸用詞排除契約擴充為逐一覆蓋同鏈已退役 exact-contract wording 後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 最嚴格檢查檢出 validator 已擴張到檢查自身命名與補丁式 wording hygiene，超出 `--help` 成品層正式範圍，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 將 validator 收斂為只驗 `apps/test_suite.py --help` 的穩定主題與 bare `checklist` 排除後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 最嚴格檢查檢出 help 舊 wording / 裸用詞排除 validator 名義上驗 `--help`、實際卻掃 source 內單行 `print` 字串，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 將 help 舊 wording / 裸用詞排除 validator 改為直接擷取 `apps/test_suite.py --help` 輸出後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 最嚴格檢查檢出 help 裸用詞排除 validator 仍以 `TEST_SUITE_CHECKLIST` 白名單承接已不應出現在正式 help 的 canonical token，改回 PARTIAL | DONE -> PARTIAL | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 將 help 裸用詞排除 validator 收斂為直接禁止 bare `checklist` 與 `contract` 長列舉後重新收斂為 DONE | PARTIAL -> DONE | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T244 | 將 help 舊 wording / 裸用詞排除 guard 自 formal blocker 退役 | DONE -> N/A | `validate_test_suite_help_text_has_no_stale_wording_or_bare_term_case` |
| 2026-04-13 | T247 | 將 checklist 摘要區 heading 唯一性 guard 自 formal blocker 退役 | DONE -> N/A | `validate_checklist_summary_section_headings_unique_case` |
| 2026-04-13 | T251 | 最嚴格檢查檢出 internal helper exact file-tree validator 將高波動 helper 清單升格為長期 formal contract，與 `doc/ARCHITECTURE.md` 穩定邊界不一致，改列 N/A | DONE -> N/A | `doc/ARCHITECTURE.md` |
| 2026-04-13 | T252 | 最嚴格檢查檢出 internal support exact file-tree validator 將高波動 support 清單升格為長期 formal contract，與 `doc/ARCHITECTURE.md` 穩定邊界不一致，改列 N/A | DONE -> N/A | `doc/ARCHITECTURE.md` |
| 2026-04-13 | T254 | 依 formal 瘦身將 `G` 備註欄單一可解析代表 entry hygiene guard 退出正式長期 test suite，改列 N/A | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-13 | T255 | 依 formal 瘦身將 `G` 備註欄 canonical path existence hygiene guard 退出正式長期 test suite，改列 N/A | DONE -> N/A | `doc/PROJECT_SETTINGS.md` |
| 2026-04-14 | B152 | 最嚴格檢查檢出 trade_analysis 文件與 helper CLI 仍把 `trade_log.py` 誤綁為正式入口並保留 debug-only prompt / banner，改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-14 | B152 | 將 trade_analysis 正式使用者入口收斂為 `apps/workbench.py`，並同步修正文檔與 helper CLI 語意後重新收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_meta_cases.py` |
| 2026-04-14 | B162 | 依文件瘦身方向將 ARCHITECTURE apps exact file-tree 同步契約退出 long-term formal scope，補齊最新狀態為 N/A | DONE -> N/A | `doc/ARCHITECTURE.md` |
| 2026-04-14 | B163 | 依文件瘦身方向將 ARCHITECTURE models exact file-tree 同步契約退出 long-term formal scope，補齊最新狀態為 N/A | DONE -> N/A | `doc/ARCHITECTURE.md` |
| 2026-04-14 | B166 | 目前主表仍保留 Local Regression `run_meta_quality.py` 檔案樹同步契約為正式 DONE，補齊最新狀態鏈 | N/A -> DONE | `validate_architecture_local_regression_meta_quality_file_tree_sync_case` |
| 2026-04-14 | B167 | 新增 validate runtime 暫存 staging 契約並驗證 | NEW -> DONE | `validate_validate_runtime_tmp_output_staging_contract_case` |
| 2026-04-14 | T238 | 最嚴格檢查檢出 trade_analysis legacy naming 文件契約仍將 `trade_log.py` 綁為正式入口且 helper CLI 殘留 debug-only prompt / banner，改回 PARTIAL | DONE -> PARTIAL | `validate_trade_analysis_legacy_naming_documentation_contract_case` |
| 2026-04-14 | T238 | 將 trade_analysis legacy naming 文件契約與文件 / helper CLI 一起收斂為 `apps/workbench.py` 單一使用者入口後重新收斂為 DONE | PARTIAL -> DONE | `validate_trade_analysis_legacy_naming_documentation_contract_case` |
| 2026-04-14 | T256 | 新增 validate runtime 暫存 staging 契約並驗證 | NEW -> DONE | `validate_validate_runtime_tmp_output_staging_contract_case` |
| 2026-04-18 | B52 | 檢出 walk-forward policy config loader 與 optimizer callbacks 顯示鏈缺少 formal contract，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_strategy_cases.py` |
| 2026-04-18 | B52 | 補齊 walk-forward policy config loader / callbacks contract 後重新收斂為 DONE | PARTIAL -> DONE | `validate_optimizer_walk_forward_policy_contract_case` |
| 2026-04-18 | T257 | 新增 optimizer walk-forward policy / callbacks contract 並驗證 | NEW -> DONE | `validate_optimizer_walk_forward_policy_contract_case` |
| 2026-04-24 | T258 | 新增 optimizer milestone cache contract 並驗證 | NEW -> DONE | `validate_optimizer_session_milestone_cache_case` |

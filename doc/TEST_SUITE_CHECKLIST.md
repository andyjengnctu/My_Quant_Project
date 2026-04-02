# Test Suite 收斂清單

目的：整理 `apps/test_suite.py` 與其本地動態測試組成步驟的覆蓋範圍、缺口、優先順序與建議落點，供後續逐項收斂。

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

收斂原則：
1. 先補長期固定測試，再補可隨策略升級調整的測試。
2. 每完成一項，需同步更新本表狀態、對應測試入口與結果摘要。
3. 若新增測試導致模組責任改變，再更新 `doc/ARCHITECTURE.md` 與 `doc/CMD.md`。
4. 優先補 synthetic / unit / contract test；避免讓 GPT 端重跑本地完整動態流程。
5. test suite 應優先驗證規格、契約與 invariant，避免綁死當前 ML / DRL / LLM 策略實作細節。
6. 在 test suite 完全收斂前，`TODO` 與 `PARTIAL` 項目仍由 GPT 端補驗；`DONE` 項目原則上不重複完整執行，但若本輪改動直接影響其模組、測試入口、輸出契約、架構責任，或出現可疑症狀，GPT 端仍需做定向複核。
7. GPT 端補驗應採差異化驗證：只補缺口與高風險影響面，不重跑已穩定覆蓋且與本輪改動無關的完整動態流程。
8. 每輪開始時，GPT 也必須檢查本清單是否仍足夠支撐本輪完整性判斷；檢查範圍不只包含正式邏輯與跨工具契約，也包含 test suite 本身的註冊完整性、自我驗證、coverage 與收斂缺口。若不足，須先把缺口回寫到本清單，再執行後續驗證或修改。

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
| B01 | P0 | 杜絕未來函數 | PARTIAL | 已新增 prev-day-only PIT case 補強盤前只能讀前一日資料，但仍不足以證明全域無 lookahead | `tools/validate/synthetic_history_cases.py` |
| B02 | P1 | 同 K 棒停利/停損取最壞停損 | DONE | 已有明確 synthetic case | 既有 synthetic case |
| B03 | P0 | 權益曲線、資金、PnL 一律為扣費扣稅後淨值 | DONE | 已新增直接手算對帳案例，逐欄位檢查 entry cash / entry equity / exit pnl / final equity / total return | `tools/validate/synthetic_take_profit_cases.py` |
| B04 | P1 | 半倉停利只算現金回收，尾倉才算完整 Round-Trip | DONE | 已新增直接案例，斷言半倉列不得提前帶完整 `該筆總損益`，完整 Round-Trip 僅在尾倉結算列完成 | `tools/validate/synthetic_take_profit_cases.py` |
| B05 | P0 | 只能盤前掛單；盤中不得新增/改單/換股 | DONE | 已新增直接禁止盤中改單 case，並以 failed fill / same-day sell / rotation case 共同覆蓋盤中新增與換股 | `tools/validate/synthetic_flow_cases.py` |
| B06 | P0 | 不得當沖；買入當日不可賣出；當日賣出回收資金不得當日再投入 | DONE | 已新增買入當日不可賣出的直接 case；同日賣出後不得再投入則由既有 same-day sell block / rotation T+1 case 覆蓋 | `tools/validate/synthetic_flow_cases.py` |
| B07 | P0 | 未成交不得同日盤中自動改掛其他股票 | DONE | 已新增直接 failed fill 後不得同日換股 case | `tools/validate/synthetic_flow_cases.py` |
| B08 | P0 | 停利/停損只能對已持有部位預先設定 | DONE | 已新增 zero-qty position direct assertion，確認無持倉時 stop/tp / indicator sell 不得產生任何 exit event | `tools/validate/synthetic_take_profit_cases.py` |
| B09 | P1 | 候選、掛單、成交、miss buy、歷史績效統計必須分層定義 | DONE | 已新增 candidate / filled / missed-buy 三層直接案例，釘死狀態不得混用 | `tools/validate/synthetic_flow_cases.py` |
| B10 | P1 | 單股回測不得用自身歷史績效 filter 作為買入閘門；history filter 僅用於投組層/scanner | DONE | 已新增 cross-tool case，直接對照單股回測仍成交、scanner 端仍拒絕非 candidate | `tools/validate/synthetic_history_cases.py` |

### B2. 未明列於專案設定，但正式 test suite 應納入

| ID | 優先級 | 類別 | 項目 | 目前判定 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|---|---|
| B11 | P1 | 契約 | 跨工具 schema / 欄位語意一致 | DONE | 已補 missed sell / trade log / stats 一致性，以及 validate / issue report / optimizer profile / local regression summaries（preflight / dataset prepare / chain / ml smoke / meta quality / master summary）的 CSV / XLSX / JSON contract，主要跨工具 schema 與欄位語意已收斂 | contract tests under `tools/validate/` |
| B12 | P1 | 決定性 | 同資料、同參數、同 seed 結果可重現 | DONE | 已補 `run_ml_smoke.py` fixed-seed 雙跑、`run_chain_checks.py` scanner reduced snapshot 雙跑 digest、`validate_scanner_worker_repeatability_case` 與 `validate_scan_runner_repeatability_case`，正式入口與 scanner 入口重跑一致性已收斂 | `tools/local_regression/`, `tools/validate/synthetic_regression_cases.py` |
| B13 | P1 | 邊界值 | 數值穩定性、rounding、tick、odd lot | DONE | 已新增 `price_utils` / `history_filters` / `portfolio_stats` unit-like 邊界案例，覆蓋 tick、稅費、sizing、全贏/全輸與空序列 | `tools/validate/synthetic_unit_cases.py` |
| B14 | P1 | 韌性 | 髒資料、缺欄位、NaN、日期亂序、OHLC 異常 | DONE | 已新增資料清洗 expected behavior / fail-fast / `load_clean_df` 整合案例，直接釘死髒資料修正、欄位缺失、NaN、日期亂序、OHLC 異常與清洗後列數行為 | `tools/validate/synthetic_data_quality_cases.py`, `core/data_utils.py`, `tools/validate/real_case_io.py` |
| B15 | P1 | 錯誤處理 | 壞 JSON、缺參數、缺檔、匯入失敗、API 失敗時訊息可定位 | PARTIAL | 已補 `params_io` / `module_loader` / `preflight_env` 的 module 級錯誤路徑，並新增 downloader market-date fallback / sync aggregation / main summary fail-fast；其他外部 API 與更多 module error path 仍未全面補齊 | `core/params_io.py`, `tools/validate/preflight_env.py`, `tools/validate/module_loader.py` |
| B16 | P2 | CLI | 互斥參數、預設值、help 與實作一致 | DONE | 已補 dataset wrapper、local regression / no-arg CLI 與剩餘直接入口 CLI 契約，覆蓋 help、預設 passthrough、`--only` / `--steps` 正規化、未知參數、缺值、空值與位置參數拒絕 | `tools/validate/synthetic_cli_cases.py`, `apps/*.py`, `core/runtime_utils.py` |
| B17 | P2 | I/O | 輸出工件、bundle、retention、rerun 覆寫行為 | DONE | 已補 validate summary / optimizer profile / issue report 的 output contract，以及 bundle/archive/root-copy/retention lifecycle、PASS/FAIL bundle selection、artifacts manifest 與 rerun 覆寫內容契約 | `core/output_paths.py`, `core/output_retention.py`, `tools/validate/synthetic_contract_cases.py` |
| B18 | P1 | 回歸 | 重跑一致性、狀態汙染、cache 汙染 | DONE | 已補 `run_chain_checks.py` 雙跑 digest、`run_ml_smoke.py` fixed-seed 雙跑、`validate_optimizer_raw_cache_rerun_consistency_case` 與 `validate_run_all_repeatability_case`，正式入口與 cache 汙染隔離已收斂 | `tools/local_regression/`, `tools/optimizer/raw_cache.py`, `tools/validate/synthetic_regression_cases.py` |
| B19 | P2 | 效能 | reduced dataset 時間基線、optimizer 每 trial 上限、記憶體回歸 | PARTIAL | 已將 quick gate / consistency / chain checks / ml smoke / meta quality / total suite duration 與 optimizer 平均 trial wall time 納入 `run_meta_quality.py` 正式 gating；記憶體回歸仍未納入 | `tools/local_regression/` |
| B20 | P2 | 文件 | `doc/CMD.md` 指令與實作一致 | DONE | 已新增 CMD Python 指令契約案例，校驗腳本存在、`--dataset` / `--only` / `--steps` 參數值合法，並確認文件中的專案腳本已納入 quick gate help 檢查 | `tools/validate/synthetic_meta_cases.py` |
| B21 | P2 | 顯示 | 報表欄位、排序、百分比格式與來源一致 | DONE | 已補 scanner header / start banner / summary、strategy dashboard、validate console summary、issue Excel report schema、portfolio yearly/export report，以及 `apps/test_suite.py` 在 PASS / FAIL / manifest-blocked / partial-selected-steps / preflight-failed / dataset-prepare-failed / summary-unreadable 下的人類可讀摘要契約；另補 checklist status vocabulary sync 與 meta quality coverage line/branch/min-threshold/missing-zero-target guard 摘要顯示，並以 `run_all.py` contract 釘死 preflight 早退時 dataset_prepare 仍需標記為 `NOT_RUN`，避免 real path 誤落成 `missing_summary_file` | `tools/validate/synthetic_display_cases.py`, `tools/validate/synthetic_reporting_cases.py`, `tools/validate/synthetic_contract_cases.py`, `core/display.py`, `tools/scanner/reporting.py`, `tools/portfolio_sim/reporting.py`, `apps/test_suite.py`, `tools/local_regression/run_all.py` |
| B22 | P2 | 覆蓋率 | line / branch coverage 報表 | DONE | 已將 `run_meta_quality.py` 的 synthetic coverage suite、formal helper probe、key target presence/hit 與 manifest 化 line / branch minimum threshold gate 收斂為正式路徑，並同步回寫 `meta_quality_summary.json` / `apps/test_suite.py` 摘要顯示 | `tools/local_regression/run_meta_quality.py`, `tools/local_regression/common.py`, `apps/test_suite.py` |
| B23 | P1 | Meta | checklist / 測試註冊 / 正式入口一致性 | DONE | 已新增 meta registry case 與 `run_meta_quality.py` formal-entry 檢查；同時校驗 `DONE` 摘要、對應 test function、synthetic 主入口、`run_all.py` / `preflight_env.py` / `apps/test_suite.py` / `PROJECT_SETTINGS.md` 的正式步驟一致 | `tools/validate/synthetic_meta_cases.py`, `tools/local_regression/run_meta_quality.py` |
| B24 | P1 | Meta | known-bad fault injection：關鍵規則故意破壞後測試必須 fail | DONE | 已新增 meta fault-injection case，直接對 same-day sell、same-bar stop priority、fee/tax、history filter misuse 注入 known-bad 行為，並驗證既有測試會產生 FAIL | `tools/validate/synthetic_meta_cases.py` |
| B25 | P1 | Meta | independent oracle / golden cases：高風險數值規則不可只與 production 共用同邏輯 | DONE | 已新增獨立 oracle golden case，對 net sell、position size、history EV、annual return / sim years 以手算或獨立公式對照 production | `tools/validate/synthetic_unit_cases.py` |
| B26 | P1 | Meta | checklist 是否已足夠覆蓋完整性（包含 test suite 本身） | DONE | 已將 `run_meta_quality.py` 的 formal check 擴充到主表 / E/F 摘要一致性、`apps/test_suite.py` 單一正式入口、`doc/CMD.md` / `doc/ARCHITECTURE.md` 單一入口宣告、legacy app test entry 缺失與可疑替代測試入口檔名檢查；test suite 自身完整性已納入正式 gate | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/meta_contracts.py` |

## C. 可隨策略升級調整的測試清單

| ID | 優先級 | 類別 | 項目 | 原則 | 建議落點 |
|---|---|---|---|---|---|
| C01 | P1 | 模型介面 | model feature schema / prediction schema 穩定 | 驗輸入欄位、輸出欄位、型別、缺值處理；不驗固定分數 | `tools/optimizer/`, `tools/scanner/` |
| C02 | P1 | 重現性 | 同 seed 下 optimizer / model inference 可重現 | 驗結果穩定在可接受範圍；不綁死搜尋路徑細節 | `tools/local_regression/`, `tools/optimizer/` |
| C03 | P1 | 排序輸出 | ranking / scoring 輸出可排序、可比較、無 NaN | 驗排序值可用、方向一致、型別正確；不驗固定名次 | `core/buy_sort.py`, `tools/scanner/` |
| C04 | P2 | 最低可用性 | 模型升級後 scanner / optimizer / reporting 仍可跑通 | 驗 CLI 與輸出仍可用；不驗內部中間流程 | `apps/ml_optimizer.py`, `apps/vip_scanner.py` |
| C05 | P2 | 報表相容 | 新策略輸出仍符合既有 artifact / reporting schema | 驗欄位存在與語意；不驗具體績效數值 | `tools/portfolio_sim/`, `tools/scanner/reporting.py` |

## D. 建議先補的測試項目

### D1. 長期固定測試：先收斂

| ID | 建議測試名稱 | 目標 |
|---|---|---|
| D01 | `validate_synthetic_same_day_buy_sell_forbidden_case` | 已完成；直接釘死不得當沖、買入當日不可賣出 |
| D02 | `validate_synthetic_intraday_reprice_forbidden_case` | 已完成；直接釘死盤中不得改單 |
| D03 | `validate_synthetic_no_intraday_switch_after_failed_fill_case` | 已完成；直接釘死未成交不得同日換股 |
| D04 | `validate_synthetic_exit_orders_only_for_held_positions_case` | 已完成；直接釘死 stop/tp 只能作用於已持有部位 |
| D05 | `validate_synthetic_fee_tax_net_equity_case` | 已完成；將 cash / final_equity / equity_curve / trade pnl 逐欄位對帳 |

### D2. 長期固定測試：接續補強

| ID | 建議測試名稱 | 目標 |
|---|---|---|
| D06 | `validate_synthetic_round_trip_pnl_only_on_tail_exit_case` | 已完成；補強半倉與 completed trade 口徑 |
| D07 | `validate_synthetic_missed_sell_accounting_case` | 已完成；補強 missed sell、trade log、stats 一致 |
| D08 | `validate_synthetic_candidate_order_fill_layer_separation_case` | 已完成；補強候選 / 掛單 / 成交 / miss buy 分層 |
| D09 | `validate_synthetic_portfolio_history_filter_only_case` | 已完成；補強 history filter 僅用於投組層 / scanner |
| D10 | `validate_synthetic_lookahead_prev_day_only_case` | 已完成；補強盤前只能讀前一日資料 |
| D11 | `tests_unit_price_utils.py` | 已完成；釘死 tick、費率、稅金、sizing、half sell qty 邊界 |
| D12 | `tests_unit_history_filters.py` | 已完成；釘死 EV、win rate、trade count 邊界 |
| D13 | `tests_unit_portfolio_stats.py` | 已完成；釘死年化、MDD、R²、空序列邊界 |

### D3. 可隨策略升級調整：最低維護線

| ID | 建議項目 | 目標 |
|---|---|---|
| D14 | model input / output schema checks | 已完成；釘死 optimizer best_params / scanner result 的輸入輸出 schema、型別與缺值處理 |
| D15 | deterministic regression for optimizer/scanner | 已完成；已補 optimizer fixed-seed 雙跑、scanner worker repeatability 與 `scan_runner` 入口重跑一致性 |
| D16 | ranking / scoring output sanity checks | 已完成；釘死 buy_sort / portfolio score 單調性、有限值與 scanner sort_value 可比較性 |
| D17 | reporting schema compatibility checks | 已完成；已補 console/Excel/portfolio export/test_suite 摘要的 reporting schema 相容性，並補齊 FAIL / manifest-blocked / partial-selected-steps 摘要路徑 |

### D4. 品質補強

| ID | 建議項目 | 目標 |
|---|---|---|
| D18 | contract tests for CSV / XLSX / JSON outputs | 已補 validate summary / optimizer profile / issue report 的 schema contract；更多工具輸出仍可補 |
| D28 | `validate_artifact_lifecycle_contract_case` | 已補 bundle/archive/root-copy/retention lifecycle、PASS/FAIL bundle selection、artifacts manifest 與 rerun 覆寫內容契約 |
| D30 | `validate_params_io_error_path_case` | 已補壞 JSON、缺必要欄位、未知欄位、缺檔時的 fail-fast 與錯誤訊息定位 |
| D31 | `validate_module_loader_error_path_case` | 已補 syntax error、缺必要屬性與 checked path/reason 彙整的錯誤路徑 |
| D32 | `validate_preflight_error_path_case` | 已補 requirements 檔缺失、非法 steps、import failure detail 的錯誤路徑 |
| D46 | `validate_downloader_market_date_fallback_case` | 已補 FinMind / YFinance 皆失敗時的 market-date fallback、issue log 與 weekday 回退口徑 |
| D47 | `validate_downloader_sync_error_path_case` | 已補 downloader sync 對空資料 / RequestException 的錯誤聚合、ticker 可定位與 issue log path 回傳 |
| D48 | `validate_downloader_main_error_path_case` | 已補 downloader main 在全失敗時的 RuntimeError 摘要、計數與 issue log path 輸出 |
| D33 | `validate_sanitize_ohlcv_expected_behavior_case` | 已補髒資料清洗 expected behavior：負成交量修正、零量保留、重複日期去重、亂序排序與 OHLC/NaN 壞列清除 |
| D34 | `validate_sanitize_ohlcv_failfast_case` | 已補缺少日期欄、缺必要欄位、全列無效、清洗後列數不足的 fail-fast |
| D35 | `validate_load_clean_df_data_quality_case` | 已補 `real_case_io.load_clean_df()` 與資料清洗整合案例，確認 sanitize stats 與清洗後排序/列數一致 |
| D36 | `validate_dataset_cli_contract_case` | 已補 dataset wrapper CLI 契約：help、預設 passthrough、`--dataset` 值傳遞、未知參數與位置參數拒絕 |
| D37 | `validate_local_regression_cli_contract_case` | 已補 run_all / preflight / no-arg CLI 契約：`--only` / `--steps` 正規化、help 與未知參數 / 位置參數拒絕 |
| D45 | `validate_extended_tool_cli_contract_case` | 已補剩餘直接入口 CLI：`tools/optimizer/main.py`、`tools/portfolio_sim/main.py`、`tools/scanner/scan_runner.py`、`tools/validate/main.py`、`tools/debug/trade_log.py`、`apps/test_suite.py` 的 help 與參數拒絕契約 |
| D19 | rerun / cache pollution checks | 已完成；已補 raw cache rerun / mutation isolation、`run_all.py` 同 run dir rerun summary / bundle repeatability 與 chain snapshot 雙跑 digest |
| D41 | `tools/local_regression/run_chain_checks.py` scanner reduced snapshot rerun digest | 已補 scanner reduced snapshot 雙跑 digest，確認 scanner 候選 / 狀態 / issue line 在同資料同參數下穩定一致 |
| D20 | coverage report baseline | 已補 `run_meta_quality.py` 產出 synthetic coverage suite 的 line / branch baseline、key target presence/hit 與 manifest 化 minimum threshold gate，並已同步到 `apps/test_suite.py` 摘要顯示 |
| D21 | performance baseline checks | 已補 `run_meta_quality.py` 讀取同輪 step summaries 與 optimizer profile summary，正式檢查 reduced suite duration / total duration / optimizer 平均 trial wall time；記憶體回歸仍未納入 |
| D22 | registry / checklist / main-entry consistency checks | 已完成；確認 `DONE` 項目皆已映射到實際 test function 與 synthetic 主入口 |
| D29 | formal non-synthetic entry consistency checks | 已完成；確認 `run_all.py` / `preflight_env.py` / `apps/test_suite.py` / `PROJECT_SETTINGS.md` 的正式步驟一致 |
| D26 | `validate_cmd_document_contract_case` | 釘死 `doc/CMD.md` 的 Python 指令、步驟名與腳本存在性契約 |
| D27 | `validate_display_reporting_sanity_case` | 已補 scanner header / start banner / summary、strategy dashboard 與 `core/display.py` re-export 顯示契約 |
| D42 | `validate_issue_excel_report_schema_case` | 已補 validate issue Excel 輸出檔的 sheet / header / ticker text format schema |
| D43 | `validate_portfolio_export_report_artifacts_case` | 已補 portfolio report Excel / HTML 輸出檔 schema 與 artifact 路徑契約 |
| D50 | `validate_test_suite_summary_failure_reporting_case` | 已補 `apps/test_suite.py` 腳本失敗時的 failure reason 與 rerun command 摘要契約 |
| D51 | `validate_test_suite_summary_manifest_failure_reporting_case` | 已補 manifest failed 時 preflight / dataset / regression steps blocked 摘要契約 |
| D52 | `validate_test_suite_summary_optional_dataset_skip_case` | 已補 partial selected-steps 結果下 dataset prepare not_required 與 not_selected 摘要契約 |
| D53 | `validate_test_suite_summary_preflight_failure_reporting_case` | 已補 preflight fail 時 blocked steps、bundle 空值與步驟名稱顯示摘要契約 |
| D54 | `validate_test_suite_summary_dataset_prepare_failure_reporting_case` | 已補 dataset prepare fail 時 blocked regression steps 與錯誤細節摘要契約 |
| D55 | `validate_test_suite_summary_unreadable_payload_reporting_case` | 已補 step summary unreadable 時 `summary_unreadable` / `error_type` 顯示摘要契約 |
| D56 | `validate_run_all_preflight_early_failure_dataset_contract_case` | 已補 `run_all.py` preflight 早退且 dataset step 應存在時，`dataset_prepare` 必須列入 `not_run_step_names`，避免 `apps/test_suite.py` 將其誤顯示成 `missing_summary_file` |
| D57 | `validate_test_suite_summary_checklist_status_sync_case` | 已補 `apps/test_suite.py` meta quality 摘要需直接使用 checklist 的 `DONE / PARTIAL / TODO / N/A` 狀態語彙，並顯示 partial / todo / done ID 預覽 |
| D58 | `validate_test_suite_summary_meta_quality_guardrail_reporting_case` | 已補 `apps/test_suite.py` meta quality 摘要顯示 coverage line / branch、minimum threshold、missing / zero-covered targets 與 checklist guard 狀態 |
| D23 | known-bad fault injection checks | 已完成；對 same-day sell、same-bar stop priority、fee/tax、history filter misuse 注入 known-bad 行為並驗證既有測試會 fail |
| D24 | independent oracle / golden numeric cases | 已完成；以獨立 oracle 對照 net sell、position size、history EV、annual return / sim years |
| D25 | checklist sufficiency review | 已補 `run_meta_quality.py` formal check，並擴充到 `apps/test_suite.py` 單一正式入口、CMD / ARCHITECTURE 宣告與 legacy / 可疑替代入口檢查 |
| D59 | `validate_single_formal_test_entry_contract_case` | 已補 `apps/test_suite.py` 單一正式入口契約，確認無 legacy app test entry 與可疑替代測試入口檔名 |

## E. 未完成缺口摘要

說明：本節只作為目前所有未完成項目的快速索引，方便優先查看 `PARTIAL` 與 `TODO`；主維護來源仍是本檔前文各表格，不另作第二份主清單。

### E1. 目前所有 `PARTIAL` 的主表項目摘要

| 類型 | ID | 項目 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|
| 規則 | B01 | 杜絕未來函數 | 已新增 prev-day-only PIT case 補強盤前只能讀前一日資料，但仍不足以證明全域無 lookahead | `tools/validate/synthetic_history_cases.py` |
| 品質 | B15 | 壞 JSON、缺參數、缺檔、匯入失敗、API 失敗時訊息可定位 | 已補 `params_io` / `module_loader` / `preflight_env` 的 module 級錯誤路徑，並新增 downloader market-date fallback / sync aggregation / main summary fail-fast；其他外部 API 與更多 module error path 仍未全面補齊 | `core/params_io.py`, `tools/validate/preflight_env.py`, `tools/validate/module_loader.py` |
| 效能 | B19 | reduced dataset 時間基線、optimizer 每 trial 上限、記憶體回歸 | 已將 quick gate / consistency / chain checks / ml smoke / meta quality / total suite duration 與 optimizer 平均 trial wall time 納入 `run_meta_quality.py` 正式 gating；記憶體回歸仍未納入 | `tools/local_regression/` |

### E2. 目前所有 `TODO` 的主表項目摘要

| 類型 | ID | 項目 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|

### E3. 目前所有未完成的建議測試項目摘要

| ID | 建議測試名稱 / 項目 | 目前狀態 | 對應主表項目 |
|---|---|---|---|
| D18 | contract tests for CSV / XLSX / JSON outputs | DONE | B11 / B17 |
| D21 | performance baseline checks | PARTIAL | B19 |

## F. 已完成覆蓋摘要

說明：本節只作為目前所有 `DONE` 項目的快速索引；不區分原本既有或本輪新增。主維護來源仍是本檔前文各表格與收斂紀錄，不另作第二份主清單。

### F1. 目前所有 `DONE` 的主表項目摘要

| 類型 | ID | 項目 | 對應測試入口 | 完成日期 |
|---|---|---|---|---|
| 規則 | B02 | 同 K 棒停利/停損取最壞停損 | 既有 synthetic case | 既有 |
| 品質 | B13 | 數值穩定性、rounding、tick、odd lot | `tools/validate/synthetic_unit_cases.py` | 2026-04-01 |
| 品質 | B14 | 髒資料、缺欄位、NaN、日期亂序、OHLC 異常 | `tools/validate/synthetic_data_quality_cases.py` | 2026-04-02 |
| 品質 | B16 | 互斥參數、預設值、help 與實作一致 | `tools/validate/synthetic_cli_cases.py`, `apps/*.py`, `core/runtime_utils.py` | 2026-04-02 |
| 規則 | B03 | 權益曲線、資金、PnL 一律為扣費扣稅後淨值 | `tools/validate/synthetic_take_profit_cases.py` | 2026-04-01 |
| 規則 | B04 | 半倉停利只算現金回收，尾倉才算完整 Round-Trip | `tools/validate/synthetic_take_profit_cases.py` | 2026-04-01 |
| 規則 | B05 | 只能盤前掛單；盤中不得新增/改單/換股 | `tools/validate/synthetic_flow_cases.py` | 2026-04-01 |
| 規則 | B06 | 不得當沖；買入當日不可賣出；當日賣出回收資金不得當日再投入 | `tools/validate/synthetic_flow_cases.py` | 2026-04-01 |
| 規則 | B07 | 未成交不得同日盤中自動改掛其他股票 | `tools/validate/synthetic_flow_cases.py` | 2026-04-01 |
| 規則 | B08 | 停利/停損只能對已持有部位預先設定 | `tools/validate/synthetic_take_profit_cases.py` | 2026-04-01 |
| 規則 | B09 | 候選、掛單、成交、miss buy、歷史績效統計必須分層定義 | `tools/validate/synthetic_flow_cases.py` | 2026-04-01 |
| 規則 | B10 | 單股回測不得用自身歷史績效 filter 作為買入閘門；history filter 僅用於投組層/scanner | `tools/validate/synthetic_history_cases.py` | 2026-04-01 |
| 契約 | B11 | 跨工具 schema / 欄位語意一致 | `tools/validate/synthetic_contract_cases.py`, `tools/validate/synthetic_portfolio_cases.py` | 2026-04-02 |
| 決定性 | B12 | 同資料、同參數、同 seed 結果可重現 | `tools/local_regression/run_ml_smoke.py`, `tools/local_regression/run_chain_checks.py`, `tools/validate/synthetic_regression_cases.py` | 2026-04-02 |
| 回歸 | B18 | 重跑一致性、狀態汙染、cache 汙染 | `tools/local_regression/run_chain_checks.py`, `tools/local_regression/run_ml_smoke.py`, `tools/validate/synthetic_regression_cases.py` | 2026-04-02 |
| I/O | B17 | 輸出工件、bundle、retention、rerun 覆寫行為 | `tools/validate/synthetic_contract_cases.py` | 2026-04-02 |
| 文件 | B20 | `doc/CMD.md` 指令與實作一致 | `tools/validate/synthetic_meta_cases.py` | 2026-04-01 |
| 顯示 | B21 | 報表欄位、排序、百分比格式與來源一致 | `tools/validate/synthetic_display_cases.py`, `tools/validate/synthetic_reporting_cases.py`, `core/display.py`, `tools/scanner/reporting.py`, `tools/portfolio_sim/reporting.py`, `apps/test_suite.py` | 2026-04-02 |
| Meta | B22 | line / branch coverage 報表 | `tools/local_regression/run_meta_quality.py`, `tools/local_regression/common.py`, `apps/test_suite.py` | 2026-04-02 |
| Meta | B23 | checklist / 測試註冊 / 正式入口一致性 | `tools/validate/synthetic_meta_cases.py`, `tools/local_regression/run_meta_quality.py` | 2026-04-02 |
| Meta | B24 | known-bad fault injection：關鍵規則故意破壞後測試必須 fail | `tools/validate/synthetic_meta_cases.py` | 2026-04-01 |
| Meta | B25 | independent oracle / golden cases：高風險數值規則不可只與 production 共用同邏輯 | `tools/validate/synthetic_unit_cases.py` | 2026-04-01 |
| Meta | B26 | checklist 是否已足夠覆蓋完整性（包含 test suite 本身） | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/meta_contracts.py` | 2026-04-02 |

### F2. 目前所有 `DONE` 的建議測試項目摘要

| ID | 建議測試名稱 | 對應主表項目 | 完成日期 |
|---|---|---|---|
| D01 | `validate_synthetic_same_day_buy_sell_forbidden_case` | B06 | 2026-04-01 |
| D02 | `validate_synthetic_intraday_reprice_forbidden_case` | B05 | 2026-04-01 |
| D03 | `validate_synthetic_no_intraday_switch_after_failed_fill_case` | B07 | 2026-04-01 |
| D04 | `validate_synthetic_exit_orders_only_for_held_positions_case` | B08 | 2026-04-01 |
| D05 | `validate_synthetic_fee_tax_net_equity_case` | B03 | 2026-04-01 |
| D06 | `validate_synthetic_round_trip_pnl_only_on_tail_exit_case` | B04 | 2026-04-01 |
| D07 | `validate_synthetic_missed_sell_accounting_case` | B11 | 2026-04-01 |
| D08 | `validate_synthetic_candidate_order_fill_layer_separation_case` | B09 | 2026-04-01 |
| D09 | `validate_synthetic_portfolio_history_filter_only_case` | B10 | 2026-04-01 |
| D10 | `validate_synthetic_lookahead_prev_day_only_case` | B01 | 2026-04-01 |
| D11 | `validate_price_utils_unit_case` | B13 | 2026-04-01 |
| D12 | `validate_history_filters_unit_case` | B13 | 2026-04-01 |
| D13 | `validate_portfolio_stats_unit_case` | B13 | 2026-04-01 |
| D22 | `validate_registry_checklist_entry_consistency_case` | B23 | 2026-04-01 |
| D29 | `run_meta_quality.py` formal-entry consistency checks | B23 | 2026-04-02 |
| D18 | `validate_output_contract_case` | B11 / B17 | 2026-04-02 |
| D49 | `validate_local_regression_summary_contract_case` | B11 | 2026-04-02 |
| D28 | `validate_artifact_lifecycle_contract_case` | B17 | 2026-04-02 |
| D26 | `validate_cmd_document_contract_case` | B20 | 2026-04-01 |
| D25 | run_meta_quality.py checklist sufficiency review | B26 | 2026-04-02 |
| D59 | `validate_single_formal_test_entry_contract_case` | B26 | 2026-04-02 |
| D27 | `validate_display_reporting_sanity_case` | B21 | 2026-04-02 |
| D53 | `validate_test_suite_summary_preflight_failure_reporting_case` | B21 | 2026-04-02 |
| D54 | `validate_test_suite_summary_dataset_prepare_failure_reporting_case` | B21 | 2026-04-02 |
| D55 | `validate_test_suite_summary_unreadable_payload_reporting_case` | B21 | 2026-04-02 |
| D56 | `validate_run_all_preflight_early_failure_dataset_contract_case` | B21 | 2026-04-02 |
| D57 | `validate_test_suite_summary_checklist_status_sync_case` | B21 | 2026-04-02 |
| D58 | `validate_test_suite_summary_meta_quality_guardrail_reporting_case` | B21 / B22 | 2026-04-02 |
| D20 | `run_meta_quality.py` coverage report baseline | B22 | 2026-04-02 |
| D23 | `validate_known_bad_fault_injection_case` | B24 | 2026-04-01 |
| D24 | `validate_independent_oracle_golden_case` | B25 | 2026-04-01 |
| D30 | `validate_params_io_error_path_case` | B15 | 2026-04-02 |
| D31 | `validate_module_loader_error_path_case` | B15 | 2026-04-02 |
| D32 | `validate_preflight_error_path_case` | B15 | 2026-04-02 |
| D46 | `validate_downloader_market_date_fallback_case` | B15 | 2026-04-02 |
| D47 | `validate_downloader_sync_error_path_case` | B15 | 2026-04-02 |
| D48 | `validate_downloader_main_error_path_case` | B15 | 2026-04-02 |
| D33 | `validate_sanitize_ohlcv_expected_behavior_case` | B14 | 2026-04-02 |
| D34 | `validate_sanitize_ohlcv_failfast_case` | B14 | 2026-04-02 |
| D35 | `validate_load_clean_df_data_quality_case` | B14 | 2026-04-02 |
| D36 | `validate_dataset_cli_contract_case` | B16 | 2026-04-02 |
| D37 | `validate_local_regression_cli_contract_case` | B16 | 2026-04-02 |
| D45 | `validate_extended_tool_cli_contract_case` | B16 | 2026-04-02 |
| D41 | `tools/local_regression/run_chain_checks.py` scanner reduced snapshot rerun digest | B12 / B18 | 2026-04-02 |
| D15 | `validate_scan_runner_repeatability_case` | B12 | 2026-04-02 |
| D19 | `validate_run_all_repeatability_case` | B18 | 2026-04-02 |
| D14 | `validate_model_io_schema_case` | C01 | 2026-04-02 |
| D16 | `validate_ranking_scoring_sanity_case` | C03 | 2026-04-02 |

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
| 2026-04-01 | D08 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_candidate_order_fill_layer_separation_case |
| 2026-04-01 | D09 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_portfolio_history_filter_only_case |
| 2026-04-01 | D10 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_lookahead_prev_day_only_case |
| 2026-04-01 | D07 | 新增 synthetic case 並驗證 | TODO -> DONE | validate_synthetic_missed_sell_accounting_case |
| 2026-04-01 | D11 | 新增 unit-like 邊界案例並驗證 | TODO -> DONE | validate_price_utils_unit_case |
| 2026-04-01 | D18 | 新增 CSV / XLSX / JSON output contract case 並驗證 | TODO -> PARTIAL | validate_output_contract_case |
| 2026-04-02 | D18 | 擴充 local regression summary contract 並收斂完成 | PARTIAL -> DONE | validate_output_contract_case + validate_local_regression_summary_contract_case |
| 2026-04-02 | D49 | 新增 local regression summary contract case 並驗證 | TODO -> DONE | `validate_local_regression_summary_contract_case` |
| 2026-04-02 | B11 | 跨工具 schema / 欄位語意補齊，主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-01 | D12 | 新增 unit-like 邊界案例並驗證 | TODO -> DONE | validate_history_filters_unit_case |
| 2026-04-01 | D13 | 新增 unit-like 邊界案例並驗證 | TODO -> DONE | validate_portfolio_stats_unit_case |
| 2026-04-01 | D22 | 新增 meta registry case 並驗證 | TODO -> DONE | validate_registry_checklist_entry_consistency_case |
| 2026-04-01 | D23 | 新增 meta fault-injection case 並驗證 | TODO -> DONE | validate_known_bad_fault_injection_case |
| 2026-04-01 | D24 | 新增 independent oracle golden case 並驗證 | TODO -> DONE | validate_independent_oracle_golden_case |
| 2026-04-01 | D15 | 新增 optimizer fixed-seed 雙跑一致性檢查 | TODO -> PARTIAL | `run_ml_smoke.py` 已比較雙跑 trial / best_params digest；scanner 尚未補 |
| 2026-04-01 | D19 | 新增 chain checks 雙跑 digest 對比與 optimizer 雙跑 | TODO -> PARTIAL | `run_chain_checks.py` / `run_ml_smoke.py` |
| 2026-04-01 | D20 | 新增 `run_meta_quality.py` 產出 coverage baseline | TODO -> PARTIAL | 目前已覆蓋 synthetic coverage suite 與 key target coverage，並納入 `apps/test_suite.py` / `run_all.py` |
| 2026-04-02 | D20 | 補 manifest 化 line / branch threshold gate 與 summary sync | PARTIAL -> DONE | `run_meta_quality.py` + `apps/test_suite.py` |
| 2026-04-01 | D25 | 新增 `run_meta_quality.py` formal check | PARTIAL -> PARTIAL | 已可執行校驗主表 / 未完成摘要 / 已完成摘要一致性，並納入 `apps/test_suite.py` / `run_all.py` |
| 2026-04-02 | D25 | 擴充 checklist sufficiency formal check 到單一正式入口與 legacy entry 檢查後收斂完成 | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` + `tools/validate/meta_contracts.py` |
| 2026-04-02 | D59 | 新增單一正式測試入口契約案例並驗證 | NEW -> DONE | `validate_single_formal_test_entry_contract_case` |
| 2026-04-02 | B26 | checklist / test suite 自身完整性收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` + `tools/validate/synthetic_meta_cases.py` + `tools/validate/meta_contracts.py` |
| 2026-04-01 | D21 | 新增 `run_meta_quality.py` performance baseline gating | TODO -> PARTIAL | 已正式檢查 reduced suite 各步驟 / total duration 與 optimizer 平均 trial wall time；記憶體回歸仍未納入 |
| 2026-04-01 | D26 | 新增 CMD 指令契約案例並驗證 | TODO -> DONE | validate_cmd_document_contract_case |
| 2026-04-02 | D27 | 擴充 scanner summary / banner 與 display re-export 後收斂完成 | PARTIAL -> DONE | validate_display_reporting_sanity_case |
| 2026-04-02 | D50 | 新增腳本失敗摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_failure_reporting_case` |
| 2026-04-02 | D51 | 新增 manifest blocked 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_manifest_failure_reporting_case` |
| 2026-04-02 | D52 | 新增 partial selected-steps 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_optional_dataset_skip_case` |
| 2026-04-02 | D53 | 新增 preflight fail 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_preflight_failure_reporting_case` |
| 2026-04-02 | D54 | 新增 dataset prepare fail 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_dataset_prepare_failure_reporting_case` |
| 2026-04-02 | D55 | 新增 summary unreadable 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_unreadable_payload_reporting_case` |
| 2026-04-02 | D56 | 補 `run_all.py` preflight 早退 dataset not-run contract 並驗證 | TODO -> DONE | `validate_run_all_preflight_early_failure_dataset_contract_case` |
| 2026-04-02 | D57 | 補 checklist status vocabulary sync 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_checklist_status_sync_case` |
| 2026-04-02 | D58 | 補 meta quality coverage guardrail 摘要案例並驗證 | TODO -> DONE | `validate_test_suite_summary_meta_quality_guardrail_reporting_case` |
| 2026-04-02 | B22 | 將 coverage report baseline 收斂為正式 gate，主表升為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py`, `tools/local_regression/common.py`, `apps/test_suite.py` |
| 2026-04-02 | D28 | 擴充 artifact lifecycle contract 並驗證 | PARTIAL -> DONE | validate_artifact_lifecycle_contract_case |
| 2026-04-02 | D29 | 新增 formal-entry consistency checks 並驗證 | TODO -> DONE | `run_meta_quality.py` formal-entry consistency checks |
| 2026-04-02 | D30 | 新增 params_io 錯誤路徑案例並驗證 | TODO -> DONE | `validate_params_io_error_path_case` |
| 2026-04-02 | D31 | 新增 module_loader 錯誤路徑案例並驗證 | TODO -> DONE | `validate_module_loader_error_path_case` |
| 2026-04-02 | D32 | 新增 preflight 錯誤路徑案例並驗證 | TODO -> DONE | `validate_preflight_error_path_case` |
| 2026-04-02 | D46 | 新增 downloader market-date fallback 案例並驗證 | TODO -> DONE | `validate_downloader_market_date_fallback_case` |
| 2026-04-02 | D47 | 新增 downloader sync 錯誤聚合案例並驗證 | TODO -> DONE | `validate_downloader_sync_error_path_case` |
| 2026-04-02 | D48 | 新增 downloader main 失敗摘要案例並驗證 | TODO -> DONE | `validate_downloader_main_error_path_case` |
| 2026-04-02 | D33 | 新增資料清洗 expected behavior 案例並驗證 | TODO -> DONE | `validate_sanitize_ohlcv_expected_behavior_case` |
| 2026-04-02 | D34 | 新增資料清洗 fail-fast 案例並驗證 | TODO -> DONE | `validate_sanitize_ohlcv_failfast_case` |
| 2026-04-02 | D35 | 新增 `load_clean_df` 資料品質整合案例並驗證 | TODO -> DONE | `validate_load_clean_df_data_quality_case` |
| 2026-04-02 | D36 | 新增 dataset wrapper CLI 契約案例並驗證 | TODO -> DONE | `validate_dataset_cli_contract_case` |
| 2026-04-02 | D37 | 新增 local regression / no-arg CLI 契約案例並驗證 | TODO -> DONE | `validate_local_regression_cli_contract_case` |
| 2026-04-02 | D45 | 補齊剩餘直接入口 CLI 契約並收斂 B16 | TODO -> DONE | `validate_extended_tool_cli_contract_case` |
| 2026-04-02 | B16 | CLI 契約涵蓋補齊，主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_cli_cases.py` |
| 2026-04-02 | D41 | 新增 scanner reduced snapshot 雙跑 digest 並驗證 | TODO -> DONE | `run_chain_checks.py` 已將 scanner 候選 / 狀態 / issue line 納入 rerun consistency payload |
| 2026-04-02 | D14 | 新增 model I/O schema 案例並驗證 | TODO -> DONE | `validate_model_io_schema_case` |
| 2026-04-02 | D16 | 新增 ranking / scoring sanity 案例並驗證 | TODO -> DONE | `validate_ranking_scoring_sanity_case` |

| 2026-04-02 | D20 | 擴充 `run_meta_quality.py` coverage probe 到 formal helper path | PARTIAL -> PARTIAL | 已補 chain checks / ml smoke / display / test_suite summary path，仍未形成更完整的正式路徑 coverage gate |
| 2026-04-02 | D20 | 再擴充 `run_meta_quality.py` coverage probe 到 `run_all.py` helper path | PARTIAL -> PARTIAL | 已補 `_safe_format_preflight_summary` / `_write_dataset_prepare_summary` / `_compute_not_run_step_names` / `_build_bundle_entries` |

- 2026-04-02：`D17` 收斂為已完成；新增 `D42`（issue Excel report schema）與 `D43`（portfolio export report artifacts），將 `B21` 的 reporting schema compatibility 從 console/yearly/test-suite summary 補到輸出檔 schema。
- 2026-04-02：補入 `D50`、`D51`、`D52`，將 `apps/test_suite.py` 摘要契約從 PASS 顯示補到腳本失敗、manifest blocked 與 partial selected-steps 路徑。
- 2026-04-02：再補 `D53`、`D54`、`D55`，把 `apps/test_suite.py` 摘要契約延伸到 preflight fail、dataset prepare fail 與 summary unreadable 路徑，並補步驟名稱與空 bundle path 的顯示穩定性。
- 2026-04-02：補 `D56`，把 `run_all.py` real preflight early-failure 路徑納入 contract，要求 dataset step 雖未產生 payload，仍必須在 `not_run_step_names` 中標示為 `dataset_prepare`。
- 2026-04-02：補 `D57`、`D58`，把 `apps/test_suite.py` meta quality 摘要延伸到 checklist status vocabulary sync 與 coverage guardrail 顯示；同時將 `D20` / `B22` 從 `PARTIAL` 收斂為 `DONE`。

| 2026-04-02 | D15 | 補 scanner worker / `scan_runner` 入口重跑一致性後收斂完成 | PARTIAL -> DONE | `validate_scanner_worker_repeatability_case` + `validate_scan_runner_repeatability_case` + `run_ml_smoke.py` fixed-seed dual-run |
| 2026-04-02 | D19 | 補 `run_all.py` 同 run dir rerun summary / bundle repeatability 後收斂完成 | PARTIAL -> DONE | `validate_optimizer_raw_cache_rerun_consistency_case` + `validate_run_all_repeatability_case` |
| 2026-04-02 | B12 | 決定性主表收斂為 DONE | PARTIAL -> DONE | `run_ml_smoke.py` + `run_chain_checks.py` + `tools/validate/synthetic_regression_cases.py` |
| 2026-04-02 | B18 | 重跑一致性 / 狀態汙染主表收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_chain_checks.py` + `tools/local_regression/run_ml_smoke.py` + `tools/validate/synthetic_regression_cases.py` |

## H. 完成判準

可視為 test suite 已明顯收斂的最低條件：
1. `B1` 區 `P0` 項目全數至少達到 `DONE`。
2. `B1` 區其餘項目不得低於 `PARTIAL`，且缺口需可明確說明。
3. `B2` 區 `B11`、`B12`、`B13`、`B14`、`B18`、`B22` 至少達到 `PARTIAL`。
4. `B23`、`B24`、`B25`、`B26` 至少達到 `PARTIAL`，以證明 checklist 與 test suite 本身正確性已被納入收斂。
5. 有固定的 coverage baseline 與 reduced dataset regression baseline。
6. 可隨策略升級調整之測試，至少要覆蓋 model/schema/seed/reporting 四類介面契約。
7. 每輪開始時，需先判斷目前 checklist 是否仍足夠支撐本輪完整性判斷；若不足，須先更新 checklist 再進行後續驗證或修改。
8. 新增測試後，不得造成規則分叉、模組責任混亂或明顯效能退化。
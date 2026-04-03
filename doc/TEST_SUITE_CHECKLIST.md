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
6. GPT 端完整檢查流程、sufficiency review、`TODO` / `PARTIAL` / `DONE` 的處理原則，以及正式步驟執行條件，一律以 `doc/PROJECT_SETTINGS.md` 的「## B. 標準測試流程」為單一真理來源；本清單不再重複改寫同一套規則，以避免不同步。

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
| B09 | P1 | 候選、掛單、成交、miss buy、歷史績效統計必須分層定義 | DONE | 已新增 candidate / filled / missed-buy 三層直接案例，釘死狀態不得混用 | `tools/validate/synthetic_flow_cases.py` |
| B10 | P1 | 單股回測不得用自身歷史績效 filter 作為買入閘門；history filter 僅用於投組層/scanner | DONE | 已新增 cross-tool case，直接對照單股回測仍成交、scanner 端仍拒絕非 candidate | `tools/validate/synthetic_history_cases.py` |

### B2. 未明列於專案設定，但正式 test suite 應納入

| ID | 優先級 | 類別 | 項目 | 目前判定 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|---|---|
| B11 | P1 | 契約 | 跨工具 schema / 欄位語意一致 | DONE | 已補 missed sell / trade log / stats 一致性，以及 validate / issue report / optimizer profile / local regression summaries（preflight / dataset prepare / chain / ml smoke / meta quality / master summary）的 CSV / XLSX / JSON contract；另已釘死 early-failure `master_summary.json` 的 `payload_failures` 必須維持與正常路徑一致的語意，且不得把合法 FAIL payload 誤標成 `summary_unreadable`，主要跨工具 schema 與欄位語意已收斂 | contract tests under `tools/validate/` |
| B12 | P1 | 決定性 | 同資料、同參數、同 seed 結果可重現 | DONE | 已補 `run_ml_smoke.py` fixed-seed 雙跑、`run_chain_checks.py` scanner reduced snapshot 雙跑 digest、`validate_scanner_worker_repeatability_case` 與 `validate_scan_runner_repeatability_case`，正式入口與 scanner 入口重跑一致性已收斂 | `tools/local_regression/`, `tools/validate/synthetic_regression_cases.py` |
| B13 | P1 | 邊界值 | 數值穩定性、rounding、tick、odd lot | DONE | 已新增 `price_utils` / `history_filters` / `portfolio_stats` unit-like 邊界案例，覆蓋 tick、稅費、sizing、全贏/全輸與空序列 | `tools/validate/synthetic_unit_cases.py` |
| B14 | P1 | 韌性 | 髒資料、缺欄位、NaN、日期亂序、OHLC 異常 | DONE | 已新增資料清洗 expected behavior / fail-fast / `load_clean_df` 整合案例，直接釘死髒資料修正、欄位缺失、NaN、日期亂序、OHLC 異常與清洗後列數行為 | `tools/validate/synthetic_data_quality_cases.py`, `core/data_utils.py`, `tools/validate/real_case_io.py` |
| B15 | P1 | 錯誤處理 | 壞 JSON、缺參數、缺檔、匯入失敗、API 失敗時訊息可定位 | DONE | 已補 `params_io` / `module_loader` / `preflight_env` 的 module 級錯誤路徑，並補 downloader universe fetch 全失敗與 screening 初始化失敗的 fatal error path，錯誤訊息與 issue log 已可定位 | `core/params_io.py`, `tools/validate/preflight_env.py`, `tools/validate/module_loader.py`, `tools/validate/synthetic_error_cases.py` |
| B16 | P2 | CLI | 互斥參數、預設值、help 與實作一致 | DONE | 已補 dataset wrapper、local regression / no-arg CLI 與剩餘直接入口 CLI 契約，覆蓋 help、預設 passthrough、`--only` / `--steps` 正規化、未知參數、缺值、空值與位置參數拒絕 | `tools/validate/synthetic_cli_cases.py`, `apps/*.py`, `core/runtime_utils.py` |
| B17 | P2 | I/O | 輸出工件、bundle、retention、rerun 覆寫行為 | DONE | 已補 validate summary / optimizer profile / issue report 的 output contract，以及 bundle/archive/root-copy/retention lifecycle、PASS/FAIL bundle selection、artifacts manifest、rerun 覆寫內容契約；另補 quick gate 發生 runtime error 時仍必須落出正式 FAIL summary / console artifact 契約 | `core/output_paths.py`, `core/output_retention.py`, `tools/validate/synthetic_contract_cases.py`, `tools/local_regression/run_quick_gate.py` |
| B18 | P1 | 回歸 | 重跑一致性、狀態汙染、cache 汙染 | DONE | 已補 `run_chain_checks.py` 雙跑 digest、`run_ml_smoke.py` fixed-seed 雙跑、`validate_optimizer_raw_cache_rerun_consistency_case` 與 `validate_run_all_repeatability_case`，正式入口與 cache 汙染隔離已收斂 | `tools/local_regression/`, `tools/optimizer/raw_cache.py`, `tools/validate/synthetic_regression_cases.py` |
| B19 | P2 | 效能 | reduced dataset 時間基線、optimizer 每 trial 上限、記憶體回歸 | DONE | 已將 quick gate / consistency / chain checks / ml smoke / meta quality / total suite duration、optimizer 平均 trial wall time 與各步驟 / meta quality traced peak memory 全數納入 `run_meta_quality.py` 正式 gating，並同步回寫 step summaries / `meta_quality_summary.json` / `apps/test_suite.py` 摘要；step summary 的 duration 欄位已補齊，consistency / chain / total budget 依實測 reduced baseline 校正；另已把 chain checks 改為同輪共用 prepared market context、prepared scanner snapshot、cached single-backtest stats，並將 replay counts 直接併入第一次 timeline 主流程；`portfolio_sim` 驗證也改直接共用 `validate_one_ticker()` 已產生的 prepared df / standalone logs，不再重讀 CSV、重新 sanitize / prepare，且單檔 portfolio context 的 `fast_data / sorted_dates / start_year` 改為共用，不再在 real-case 與 tool check 各自重建；`validate_consistency` 執行 synthetic suite 時已同步寫出 coverage artifacts，`run_meta_quality.py` 可直接重用同輪 artifacts，不再重跑一次 synthetic coverage suite | `tools/local_regression/`, `core/runtime_utils.py`, `apps/test_suite.py` |
| B20 | P2 | 文件 | `doc/CMD.md` 指令與實作一致 | DONE | 已新增 CMD Python 指令契約案例，校驗腳本存在、`--dataset` / `--only` / `--steps` 參數值合法，並確認文件中的專案腳本已納入 quick gate help 檢查 | `tools/validate/synthetic_meta_cases.py` |
| B21 | P2 | 顯示 | 報表欄位、排序、百分比格式與來源一致 | DONE | 已補 scanner header / start banner / summary、strategy dashboard、validate console summary、issue Excel report schema、portfolio yearly/export report，以及 `apps/test_suite.py` 在 PASS / FAIL / manifest-blocked / partial-selected-steps / preflight-failed / dataset-prepare-failed / summary-unreadable 下的人類可讀摘要契約；另補 checklist status vocabulary sync 與 meta quality coverage line/branch/min-threshold/missing-zero-target guard 摘要顯示，並以 `run_all.py` contract 釘死 preflight 早退時 dataset_prepare 仍需標記為 `NOT_RUN`，避免 real path 誤落成 `missing_summary_file` | `tools/validate/synthetic_display_cases.py`, `tools/validate/synthetic_reporting_cases.py`, `tools/validate/synthetic_contract_cases.py`, `core/display.py`, `tools/scanner/reporting.py`, `tools/portfolio_sim/reporting.py`, `apps/test_suite.py`, `tools/local_regression/run_all.py` |
| B22 | P2 | 覆蓋率 | line / branch coverage 報表 | DONE | 已將 `run_meta_quality.py` 的 synthetic coverage suite、formal helper probe、key target presence/hit 與 manifest 化 line / branch minimum threshold gate 收斂為正式路徑，並同步回寫 `meta_quality_summary.json` / `apps/test_suite.py` 摘要顯示 | `tools/local_regression/run_meta_quality.py`, `tools/local_regression/common.py`, `apps/test_suite.py` |
| B23 | P1 | Meta | checklist / 測試註冊 / 正式入口一致性 | DONE | 已補 synthetic 主入口遺漏註冊案例，並新增 imported / defined `validate_*` case、formal pipeline registry / formal-entry / run_all / preflight / test_suite 一致性 formal guard，以及 `core/` / `tools/` 不得反向 import `apps/` 的分層 guard；正式步驟單一真理來源已收斂到 `tools/local_regression/formal_pipeline.py` | `tools/validate/synthetic_meta_cases.py`, `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_cases.py`, `tools/local_regression/formal_pipeline.py` |
| B24 | P1 | Meta | known-bad fault injection：關鍵規則故意破壞後測試必須 fail | DONE | 已新增 meta fault-injection case，直接對 same-day sell、same-bar stop priority、fee/tax、history filter misuse 注入 known-bad 行為，並驗證既有測試會產生 FAIL | `tools/validate/synthetic_meta_cases.py` |
| B25 | P1 | Meta | independent oracle / golden cases：高風險數值規則不可只與 production 共用同邏輯 | DONE | 已新增獨立 oracle golden case，對 net sell、position size、history EV、annual return / sim years 以手算或獨立公式對照 production | `tools/validate/synthetic_unit_cases.py` |
| B26 | P1 | Meta | checklist 是否已足夠覆蓋完整性（包含 test suite 本身） | DONE | 已補主表 / `F2` / `G` 收斂紀錄完整同步 formal guard，並阻擋 `DONE` 摘要缺漏與 convergence 紀錄失同步；checklist 自身完整性已納入正式 gate | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/meta_contracts.py`, `doc/TEST_SUITE_CHECKLIST.md` |

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
| D17 | `tools/validate/synthetic_reporting_cases.py` | 已完成；已補 console/Excel/portfolio export/test_suite 摘要的 reporting schema 相容性，並補齊 FAIL / manifest-blocked / partial-selected-steps 摘要路徑 |

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
| D49 | `validate_local_regression_summary_contract_case` | 已補 local regression summaries（含 quick gate readonly compile、runtime FAIL summary 與 console artifact）的 schema / summary contract |
| D60 | `validate_synthetic_setup_index_prev_day_only_case` | 已補 setup index prev-day-only synthetic case，釘死 setup index 不得偷看當日資料 |
| D61 | `validate_downloader_universe_fetch_error_path_case` | 已補 universe fetch 全失敗 fatal error path、錯誤摘要與 issue log 可定位性 |
| D62 | `validate_downloader_universe_screening_init_error_path_case` | 已補 universe screening 初始化失敗 fatal error path 與錯誤訊息定位 |
| D33 | `validate_sanitize_ohlcv_expected_behavior_case` | 已補髒資料清洗 expected behavior：負成交量修正、零量保留、重複日期去重、亂序排序與 OHLC/NaN 壞列清除 |
| D34 | `validate_sanitize_ohlcv_failfast_case` | 已補缺少日期欄、缺必要欄位、全列無效、清洗後列數不足的 fail-fast |
| D35 | `validate_load_clean_df_data_quality_case` | 已補 `real_case_io.load_clean_df()` 與資料清洗整合案例，確認 sanitize stats 與清洗後排序/列數一致 |
| D36 | `validate_dataset_cli_contract_case` | 已補 dataset wrapper CLI 契約：help、預設 passthrough、`--dataset` 值傳遞、未知參數與位置參數拒絕 |
| D37 | `validate_local_regression_cli_contract_case` | 已補 run_all / preflight / no-arg CLI 契約：`--only` / `--steps` 正規化、help 與未知參數 / 位置參數拒絕 |
| D45 | `validate_extended_tool_cli_contract_case` | 已補剩餘直接入口 CLI：`tools/optimizer/main.py`、`tools/portfolio_sim/main.py`、`tools/scanner/scan_runner.py`、`tools/validate/main.py`、`tools/debug/trade_log.py`、`apps/test_suite.py` 的 help 與參數拒絕契約 |
| D19 | rerun / cache pollution checks | 已完成；已補 raw cache rerun / mutation isolation、`run_all.py` 同 run dir rerun summary / bundle repeatability 與 chain snapshot 雙跑 digest |
| D41 | `tools/local_regression/run_chain_checks.py` scanner reduced snapshot rerun digest | 已補 scanner reduced snapshot 雙跑 digest，確認 scanner 候選 / 狀態 / issue line 在同資料同參數下穩定一致 |
| D20 | coverage report baseline | 已補 `run_meta_quality.py` 產出 synthetic coverage suite 的 line / branch baseline、key target presence/hit 與 manifest 化 minimum threshold gate，並已同步到 `apps/test_suite.py` 摘要顯示 |
| D21 | performance baseline checks | 已補 `run_meta_quality.py` 讀取同輪 step summaries 與 optimizer profile summary，正式檢查 reduced suite duration / total duration / optimizer 平均 trial wall time，並新增 traced peak memory regression gate；各 step summary 的 `duration_sec` 已補齊，budget 依實測 reduced baseline 校正；chain checks 同輪 prepared context / scanner snapshot / cached single-backtest stats 已去重，replay counts 併入第一次 timeline 主流程；`portfolio_sim` 驗證改直接共用 `validate_one_ticker()` 已產生的 prepared df / standalone logs，不再重讀 CSV、重新 sanitize / prepare；單檔 portfolio context 也改共用 `fast_data / sorted_dates / start_year`；`scanner` / `debug_trade_log` 驗證也已補 prepared-context 等價契約；real-case `scanner_ref_stats` 改直接吃 `clean_df` fast path；同輪 `validate_consistency` 已同步寫出 synthetic coverage artifacts，`run_meta_quality.py` 可直接重用，避免再跑一次 synthetic suite |
| D63 | `validate_meta_quality_performance_memory_contract_case` | 已補 meta quality performance memory contract，直接釘死 step peak memory / max peak memory / meta quality peak memory 欄位與 budget gate |
| D64 | `validate_test_suite_summary_meta_quality_memory_reporting_case` | 已補 `apps/test_suite.py` meta quality traced peak memory 摘要顯示契約 |
| D65 | `validate_portfolio_sim_prepared_tool_contract_case` | 已補 `portfolio_sim` 單檔工具驗證 prepared-context 路徑與既有 temp-dir 路徑等價契約，避免加速後統計口徑漂移 |
| D66 | `validate_scanner_prepared_tool_contract_case` | 已補 `vip_scanner` prepared-context / precomputed-stats 路徑與既有 file-path 路徑等價契約，避免加速後候選狀態與排序口徑漂移 |
| D67 | `validate_debug_trade_log_prepared_tool_contract_case` | 已補 `debug_trade_log` prepared-context 路徑與既有 raw-df 路徑等價契約，避免加速後明細列內容與逐筆損益 sequence 漂移 |
| D68 | `validate_scanner_reference_clean_df_contract_case` | 已補 real-case `scanner_ref_stats` clean-df fast path 與既有 file-path 路徑等價契約，避免加速後 scanner 參考 stats 口徑漂移 |
| D69 | `validate_meta_quality_reuses_existing_coverage_artifacts_case` | 已補 `run_meta_quality.py` 重用同輪 `validate_consistency` coverage artifacts 契約，避免同一輪再重跑一次 synthetic coverage suite |
| D70 | `validate_registry_checklist_entry_consistency_case` imported / defined validator completeness guard | 已補 imported / defined `validate_*` cases 與 synthetic registry 完整一致 formal guard |
| D71 | `run_meta_quality.py` checklist main / `F2` / `G` sync guard | 已補主表 / `F2` / `G` 收斂紀錄完整同步 formal guard |
| D72 | `run_meta_quality.py` checklist `DONE` summary omission blocker | 已補 checklist `DONE` 摘要缺漏自動偵測與阻擋 |
| D73 | `validate_no_reverse_app_layer_dependencies_case` | 已補 `core/` / `tools/` 不得反向 import `apps/` 的分層 guard，避免 formal helper / synthetic reporting 再度耦合 app 入口 |
| D22 | registry / checklist / main-entry consistency checks | 已完成；已補 imported / defined validate case、synthetic 正式註冊清單完整一致，以及 registry 反向對照 `F2` DONE validator 摘要完整性的 formal guard，並保留單一正式入口與 checklist 對照驗證 |
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
| D56 | `validate_run_all_preflight_early_failure_dataset_contract_case` | 已補 `run_all.py` preflight / dataset_prepare 早退時的 contract：除 `not_run_step_names` 與正式 schema 外，`master_summary.json` 的 `payload_failures` 也必須與正常路徑維持一致語意，且合法 FAIL payload 不得被誤標為 `summary_unreadable`，避免早退 bundle 只補殼不補失敗細節 |
| D74 | `validate_run_all_manifest_failure_master_summary_contract_case` | 已補 `run_all.py` manifest failed 時 `master_summary.json` 仍須維持正式 schema，並以 blocked placeholder 補齊 preflight / dataset_prepare / payload_failures 契約 |
| D57 | `validate_test_suite_summary_checklist_status_sync_case` | 已補 `apps/test_suite.py` meta quality 摘要需直接使用 checklist 的 `DONE / PARTIAL / TODO / N/A` 狀態語彙，並顯示 partial / todo / done ID 預覽 |
| D58 | `validate_test_suite_summary_meta_quality_guardrail_reporting_case` | 已補 `apps/test_suite.py` meta quality 摘要顯示 coverage line / branch、minimum threshold、missing / zero-covered targets 與 checklist guard 狀態 |
| D23 | known-bad fault injection checks | 已完成；對 same-day sell、same-bar stop priority、fee/tax、history filter misuse 注入 known-bad 行為並驗證既有測試會 fail |
| D24 | independent oracle / golden numeric cases | 已完成；以獨立 oracle 對照 net sell、position size、history EV、annual return / sim years |
| D25 | checklist sufficiency review | 已完成；已補主表 / `F2` / `G` 收斂紀錄完整同步 formal guard，並阻擋 `DONE` 摘要缺漏與 convergence 狀態失同步 |
| D59 | `validate_single_formal_test_entry_contract_case` | 已補 `apps/test_suite.py` 單一正式入口契約，確認無 legacy app test entry 與可疑替代測試入口檔名 |
| D75 | `validate_synthetic_same_bar_stop_priority_case` | 已補同 K 棒停利/停損取最壞停損 synthetic case，直接釘死同棒雙觸發時必須以停損結算 |
| D76 | `validate_synthetic_half_tp_full_year_case` | 已補半倉停利與 full-year yearly return synthetic case，確認半倉列與年度報酬列同步存在 |
| D77 | `validate_synthetic_extended_miss_buy_case` | 已補 extended miss buy synthetic case，確認 extended candidate / missed buy / scanner status 一致 |
| D78 | `validate_synthetic_competing_candidates_case` | 已補同日競爭候選排序 synthetic case，確認 tie-break 後只允許單一中選標的成交 |
| D79 | `validate_synthetic_same_day_sell_block_case` | 已補當日賣出後不得同日再買 synthetic case，確認再投入必須延後到次日 |
| D80 | `validate_synthetic_rotation_t_plus_one_case` | 已補 rotation T+1 synthetic case，確認汰弱賣出後僅能於次日再評估買進 |
| D81 | `validate_synthetic_missed_buy_no_replacement_case` | 已補 missed buy 後不得盤中改掛替代標的 synthetic case |
| D82 | `validate_synthetic_unexecutable_half_tp_case` | 已補不可執行半倉停利 synthetic case，確認 trade log / portfolio rows 均不得產生假半倉列 |
| D83 | `validate_synthetic_history_ev_threshold_case` | 已補 history EV threshold equality synthetic case，確認 EV 等於門檻時仍依規格視為合格 |
| D84 | `validate_synthetic_single_backtest_not_gated_by_own_history_case` | 已補單股回測不得被自身 history filter 擋下的 direct synthetic case |
| D85 | `validate_synthetic_pit_same_day_exit_excluded_case` | 已補 PIT same-day exit excluded synthetic case，確認 exit_date 當天不得偷看同日剛結束交易 |
| D86 | `validate_synthetic_pit_multiple_same_day_exits_case` | 已補 PIT 多筆同日 exit synthetic case，確認同日所有已平倉交易皆不得納入當日 PIT 統計 |
| D87 | `validate_synthetic_proj_cost_cash_capped_case` | 已補 projected-cost / cash-capped order synthetic case，確認排序估計成本不得繞過實際可用現金上限 |
| D88 | `validate_synthetic_param_guardrail_case` | 已補 strategy param guardrail synthetic case，確認非法參數值 fail-fast，且 runtime worker 參數受界限約束 |
| D89 | `validate_validate_console_summary_reporting_case` | 已補 validate console summary reporting contract，確認 counts / path / fail preview 顯示穩定 |
| D90 | `validate_portfolio_yearly_report_schema_case` | 已補 portfolio yearly report schema contract，確認年度報酬表欄位與空資料 schema 穩定 |
| D91 | `validate_test_suite_summary_reporting_case` | 已補 test suite PASS summary reporting contract，確認 bundle / 步驟摘要 / 重點結果 / retention 顯示穩定 |

## E. 未完成缺口摘要

說明：本節只作為目前所有未完成項目的快速索引，方便優先查看 `PARTIAL` 與 `TODO`；主維護來源仍是本檔前文各表格，不另作第二份主清單。

### E1. 目前所有 `PARTIAL` 的主表項目摘要

| 類型 | ID | 項目 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|

### E2. 目前所有 `TODO` 的主表項目摘要

| 類型 | ID | 項目 | 缺口摘要 | 建議落點 |
|---|---|---|---|---|

### E3. 目前所有未完成的建議測試項目摘要

| ID | 建議測試名稱 / 項目 | 目前狀態 | 對應主表項目 |
|---|---|---|---|

## F. 已完成覆蓋摘要

說明：本節只作為目前所有 `DONE` 項目的快速索引；不區分原本既有或本輪新增。主維護來源仍是本檔前文各表格與收斂紀錄，不另作第二份主清單。

### F1. 目前所有 `DONE` 的主表項目摘要

| 類型 | ID | 項目 | 對應測試入口 | 完成日期 |
|---|---|---|---|---|
| 規則 | B02 | 同 K 棒停利/停損取最壞停損 | 既有 synthetic case | 既有 |
| 規則 | B01 | 杜絕未來函數 | `tools/validate/synthetic_history_cases.py` | 2026-04-02 |
| 品質 | B13 | 數值穩定性、rounding、tick、odd lot | `tools/validate/synthetic_unit_cases.py` | 2026-04-01 |
| 品質 | B15 | 壞 JSON、缺參數、缺檔、匯入失敗、API 失敗時訊息可定位 | `core/params_io.py`, `tools/validate/preflight_env.py`, `tools/validate/module_loader.py`, `tools/validate/synthetic_error_cases.py` | 2026-04-02 |
| 效能 | B19 | reduced dataset 時間基線、optimizer 每 trial 上限、記憶體回歸 | `tools/local_regression/run_meta_quality.py`, `core/runtime_utils.py`, `apps/test_suite.py` | 2026-04-02 |
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
| Meta | B23 | checklist / 測試註冊 / 正式入口一致性 | `tools/validate/synthetic_meta_cases.py`, `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_cases.py`, `tools/local_regression/formal_pipeline.py` | 2026-04-03 |
| Meta | B24 | known-bad fault injection：關鍵規則故意破壞後測試必須 fail | `tools/validate/synthetic_meta_cases.py` | 2026-04-01 |
| Meta | B26 | checklist 是否已足夠覆蓋完整性（包含 test suite 本身） | `tools/local_regression/run_meta_quality.py`, `tools/validate/synthetic_meta_cases.py`, `tools/validate/meta_contracts.py`, `doc/TEST_SUITE_CHECKLIST.md` | 2026-04-02 |
| Meta | B25 | independent oracle / golden cases：高風險數值規則不可只與 production 共用同邏輯 | `tools/validate/synthetic_unit_cases.py` | 2026-04-01 |

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
| D60 | `validate_synthetic_setup_index_prev_day_only_case` | B01 | 2026-04-02 |
| D11 | `validate_price_utils_unit_case` | B13 | 2026-04-01 |
| D12 | `validate_history_filters_unit_case` | B13 | 2026-04-01 |
| D13 | `validate_portfolio_stats_unit_case` | B13 | 2026-04-01 |
| D22 | `validate_registry_checklist_entry_consistency_case` | B23 | 2026-04-01 |
| D29 | `run_meta_quality.py` formal-entry consistency checks | B23 | 2026-04-02 |
| D70 | `validate_registry_checklist_entry_consistency_case` imported / defined validator completeness guard | B23 | 2026-04-02 |
| D18 | `validate_output_contract_case` | B11 / B17 | 2026-04-02 |
| D49 | `validate_local_regression_summary_contract_case` | B11 | 2026-04-02 |
| D28 | `validate_artifact_lifecycle_contract_case` | B17 | 2026-04-02 |
| D26 | `validate_cmd_document_contract_case` | B20 | 2026-04-01 |
| D25 | run_meta_quality.py checklist sufficiency review | B26 | 2026-04-02 |
| D59 | `validate_single_formal_test_entry_contract_case` | B26 | 2026-04-02 |
| D71 | `run_meta_quality.py` checklist main / `F2` / `G` sync guard | B26 | 2026-04-02 |
| D72 | `run_meta_quality.py` checklist `DONE` summary omission blocker | B26 | 2026-04-02 |
| D73 | `validate_no_reverse_app_layer_dependencies_case` | B23 | 2026-04-03 |
| D27 | `validate_display_reporting_sanity_case` | B21 | 2026-04-02 |
| D53 | `validate_test_suite_summary_preflight_failure_reporting_case` | B21 | 2026-04-02 |
| D54 | `validate_test_suite_summary_dataset_prepare_failure_reporting_case` | B21 | 2026-04-02 |
| D55 | `validate_test_suite_summary_unreadable_payload_reporting_case` | B21 | 2026-04-02 |
| D56 | `validate_run_all_preflight_early_failure_dataset_contract_case` | B11 / B17 | 2026-04-03 |
| D74 | `validate_run_all_manifest_failure_master_summary_contract_case` | B17 | 2026-04-03 |
| D57 | `validate_test_suite_summary_checklist_status_sync_case` | B21 | 2026-04-02 |
| D58 | `validate_test_suite_summary_meta_quality_guardrail_reporting_case` | B21 / B22 | 2026-04-02 |
| D64 | `validate_test_suite_summary_meta_quality_memory_reporting_case` | B19 / B21 | 2026-04-02 |
| D65 | `validate_portfolio_sim_prepared_tool_contract_case` | B11 / B19 | 2026-04-02 |
| D20 | `run_meta_quality.py` coverage report baseline | B22 | 2026-04-02 |
| D21 | run_meta_quality.py performance baseline checks | B19 | 2026-04-02 |
| D63 | `validate_meta_quality_performance_memory_contract_case` | B19 | 2026-04-02 |
| D23 | `validate_known_bad_fault_injection_case` | B24 | 2026-04-01 |
| D24 | `validate_independent_oracle_golden_case` | B25 | 2026-04-01 |
| D30 | `validate_params_io_error_path_case` | B15 | 2026-04-02 |
| D31 | `validate_module_loader_error_path_case` | B15 | 2026-04-02 |
| D32 | `validate_preflight_error_path_case` | B15 | 2026-04-02 |
| D46 | `validate_downloader_market_date_fallback_case` | B15 | 2026-04-02 |
| D47 | `validate_downloader_sync_error_path_case` | B15 | 2026-04-02 |
| D48 | `validate_downloader_main_error_path_case` | B15 | 2026-04-02 |
| D61 | `validate_downloader_universe_fetch_error_path_case` | B15 | 2026-04-02 |
| D62 | `validate_downloader_universe_screening_init_error_path_case` | B15 | 2026-04-02 |
| D33 | `validate_sanitize_ohlcv_expected_behavior_case` | B14 | 2026-04-02 |
| D34 | `validate_sanitize_ohlcv_failfast_case` | B14 | 2026-04-02 |
| D35 | `validate_load_clean_df_data_quality_case` | B14 | 2026-04-02 |
| D36 | `validate_dataset_cli_contract_case` | B16 | 2026-04-02 |
| D37 | `validate_local_regression_cli_contract_case` | B16 | 2026-04-02 |
| D45 | `validate_extended_tool_cli_contract_case` | B16 | 2026-04-02 |
| D41 | `tools/local_regression/run_chain_checks.py` scanner reduced snapshot rerun digest | B12 / B18 | 2026-04-02 |
| D15 | `validate_scan_runner_repeatability_case` | B12 | 2026-04-02 |
| D15 | `validate_scanner_worker_repeatability_case` | B12 | 2026-04-02 |
| D19 | `validate_run_all_repeatability_case` | B18 | 2026-04-02 |
| D19 | `validate_optimizer_raw_cache_rerun_consistency_case` | B18 | 2026-04-02 |
| D14 | `validate_model_io_schema_case` | C01 | 2026-04-02 |
| D16 | `validate_ranking_scoring_sanity_case` | C03 | 2026-04-02 |
| D17 | `tools/validate/synthetic_reporting_cases.py` | B21 | 2026-04-02 |
| D42 | `validate_issue_excel_report_schema_case` | B21 | 2026-04-02 |
| D43 | `validate_portfolio_export_report_artifacts_case` | B21 | 2026-04-02 |
| D50 | `validate_test_suite_summary_failure_reporting_case` | B21 | 2026-04-02 |
| D51 | `validate_test_suite_summary_manifest_failure_reporting_case` | B21 | 2026-04-02 |
| D52 | `validate_test_suite_summary_optional_dataset_skip_case` | B21 | 2026-04-02 |
| D66 | `validate_scanner_prepared_tool_contract_case` | B19 | 2026-04-02 |
| D67 | `validate_debug_trade_log_prepared_tool_contract_case` | B19 | 2026-04-02 |
| D68 | `validate_scanner_reference_clean_df_contract_case` | B19 | 2026-04-02 |
| D69 | `validate_meta_quality_reuses_existing_coverage_artifacts_case` | B19 / B22 | 2026-04-02 |
| D75 | `validate_synthetic_same_bar_stop_priority_case` | B02 | 2026-04-01 |
| D76 | `validate_synthetic_half_tp_full_year_case` | B04 / B21 | 2026-04-01 |
| D77 | `validate_synthetic_extended_miss_buy_case` | B09 | 2026-04-01 |
| D78 | `validate_synthetic_competing_candidates_case` | B09 | 2026-04-01 |
| D79 | `validate_synthetic_same_day_sell_block_case` | B06 | 2026-04-01 |
| D80 | `validate_synthetic_rotation_t_plus_one_case` | B05 / B06 | 2026-04-01 |
| D81 | `validate_synthetic_missed_buy_no_replacement_case` | B07 | 2026-04-01 |
| D82 | `validate_synthetic_unexecutable_half_tp_case` | B04 | 2026-04-01 |
| D83 | `validate_synthetic_history_ev_threshold_case` | B10 | 2026-04-01 |
| D84 | `validate_synthetic_single_backtest_not_gated_by_own_history_case` | B10 | 2026-04-01 |
| D85 | `validate_synthetic_pit_same_day_exit_excluded_case` | B01 | 2026-04-01 |
| D86 | `validate_synthetic_pit_multiple_same_day_exits_case` | B01 | 2026-04-02 |
| D87 | `validate_synthetic_proj_cost_cash_capped_case` | B09 | 2026-04-01 |
| D88 | `validate_synthetic_param_guardrail_case` | B15 | 2026-04-02 |
| D89 | `validate_validate_console_summary_reporting_case` | B21 | 2026-04-02 |
| D90 | `validate_portfolio_yearly_report_schema_case` | B21 | 2026-04-02 |
| D91 | `validate_test_suite_summary_reporting_case` | B21 | 2026-04-02 |

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
| 2026-04-02 | D49 | 擴充 quick gate readonly compile 與 runtime failure summary contract | DONE -> DONE | `validate_local_regression_summary_contract_case` 補 cfile staging 與 runtime FAIL summary / console artifact 檢查 |
| 2026-04-02 | B11 | 跨工具 schema / 欄位語意補齊，主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_contract_cases.py` |
| 2026-04-03 | B23 | 將正式步驟單一真理來源收斂到 `tools/local_regression/formal_pipeline.py` | DONE -> DONE | `run_meta_quality.py` 改驗證 code registry / run_all / preflight / test_suite 一致性；`PROJECT_SETTINGS.md` 改回僅保留上層原則 |
| 2026-04-03 | D73 | 新增 `core/` / `tools/` 不得反向 import `apps/` 的分層 guard | NEW -> DONE | `tools/validate/synthetic_meta_cases.py` + `tools/validate/meta_contracts.py` |
| 2026-04-03 | B23 | 補 `core/` / `tools/` 不得反向 import `apps/` guard，維持正式入口分層收斂 | DONE -> DONE | `core/test_suite_reporting.py` + `apps/test_suite.py` + `tools/validate/synthetic_reporting_cases.py` + `tools/local_regression/run_meta_quality.py` |
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
| 2026-04-02 | B23 | 檢出 synthetic 主入口漏註冊既有 `validate_*` case，主表改回 PARTIAL | DONE -> PARTIAL | `tools/validate/synthetic_cases.py` 尚未完整覆蓋 imported validate cases |
| 2026-04-02 | B26 | 檢出 `F2` `DONE` 摘要漏列既有完成項目，主表改回 PARTIAL | DONE -> PARTIAL | checklist 自身仍有回寫 / 摘要失同步缺口 |
| 2026-04-02 | D70 | 新增 imported validate cases vs synthetic registry formal guard 缺口 | NEW -> TODO | 需補 `tools/validate/synthetic_meta_cases.py` / `run_meta_quality.py` 檢查正式註冊完整性 |
| 2026-04-02 | D71 | 新增主表 / `F2` / `G` 完整同步 formal guard 缺口 | NEW -> TODO | 需補 `run_meta_quality.py` / `meta_contracts.py` 檢查 checklist 自身同步性 |
| 2026-04-02 | D72 | 新增 checklist `DONE` 摘要缺漏自動偵測缺口 | NEW -> TODO | 需阻擋 `F2` 遺漏已完成 D 項仍被判定為已收斂 |
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
| 2026-04-03 | D56 | 擴充 preflight / dataset_prepare 早退 contract，補 `payload_failures` 語意一致性與 `summary_unreadable` 誤判防呆並驗證 | DONE -> DONE | `validate_run_all_preflight_early_failure_dataset_contract_case` |
| 2026-04-03 | D74 | 新增 manifest failure master summary schema contract 並驗證 | NEW -> DONE | `validate_run_all_manifest_failure_master_summary_contract_case` |
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

- 2026-04-02：D20 擴充 `run_meta_quality.py` coverage probe 到 formal helper path；已補 chain checks / ml smoke / display / test_suite summary path，此補充已併入同日 `PARTIAL -> DONE` 收斂，不另新增狀態列。
- 2026-04-02：D20 再擴充 `run_meta_quality.py` coverage probe 到 `run_all.py` helper path；已補 `_safe_format_preflight_summary` / `_write_dataset_prepare_summary` / `_compute_not_run_step_names` / `_build_bundle_entries`，此補充已併入同日 `PARTIAL -> DONE` 收斂，不另新增狀態列。

| 2026-04-02 | D17 | reporting schema compatibility checks 收斂完成，並新增輸出檔 schema 補強 | TODO -> DONE | `validate_issue_excel_report_schema_case` + `validate_portfolio_export_report_artifacts_case`；將 `B21` 的 reporting schema compatibility 從 console/yearly/test-suite summary 補到輸出檔 schema |
- 2026-04-02：補入 `D50`、`D51`、`D52`，將 `apps/test_suite.py` 摘要契約從 PASS 顯示補到腳本失敗、manifest blocked 與 partial selected-steps 路徑。
- 2026-04-02：再補 `D53`、`D54`、`D55`，把 `apps/test_suite.py` 摘要契約延伸到 preflight fail、dataset prepare fail 與 summary unreadable 路徑，並補步驟名稱與空 bundle path 的顯示穩定性。
- 2026-04-02：補 `D56`，把 `run_all.py` real preflight early-failure 路徑納入 contract，要求 dataset step 雖未產生 payload，仍必須在 `not_run_step_names` 中標示為 `dataset_prepare`。
- 2026-04-03：擴充 `D56` 並新增 `D74`，補齊 `run_all.py` 在 preflight / dataset_prepare / manifest 早退路徑下 `master_summary.json` 除正式 schema 外，`payload_failures` 也必須維持與正常路徑一致的語意，且合法 FAIL payload 不得誤判為 `summary_unreadable`，避免 bundle 結構與失敗細節分叉。
- 2026-04-02：補 `D57`、`D58`，把 `apps/test_suite.py` meta quality 摘要延伸到 checklist status vocabulary sync 與 coverage guardrail 顯示；同時將 `D20` / `B22` 從 `PARTIAL` 收斂為 `DONE`。

| 2026-04-02 | D15 | 補 scanner worker / `scan_runner` 入口重跑一致性後收斂完成 | PARTIAL -> DONE | `validate_scanner_worker_repeatability_case` + `validate_scan_runner_repeatability_case` + `run_ml_smoke.py` fixed-seed dual-run |
| 2026-04-02 | D19 | 補 `run_all.py` 同 run dir rerun summary / bundle repeatability 後收斂完成 | PARTIAL -> DONE | `validate_optimizer_raw_cache_rerun_consistency_case` + `validate_run_all_repeatability_case` |
| 2026-04-02 | B12 | 決定性主表收斂為 DONE | PARTIAL -> DONE | `run_ml_smoke.py` + `run_chain_checks.py` + `tools/validate/synthetic_regression_cases.py` |
| 2026-04-02 | B18 | 重跑一致性 / 狀態汙染主表收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_chain_checks.py` + `tools/local_regression/run_ml_smoke.py` + `tools/validate/synthetic_regression_cases.py` |

| 2026-04-02 | D60 | 新增 setup-index prev-day-only synthetic case 並驗證 | NEW -> DONE | `validate_synthetic_setup_index_prev_day_only_case` |
| 2026-04-02 | B01 | 補 setup index prev-day-only invariant 後主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_history_cases.py` |
| 2026-04-02 | D61 | 新增 downloader universe fetch fatal error case 並驗證 | NEW -> DONE | `validate_downloader_universe_fetch_error_path_case` |
| 2026-04-02 | D62 | 新增 downloader screening init fatal error case 並驗證 | NEW -> DONE | `validate_downloader_universe_screening_init_error_path_case` |
| 2026-04-02 | B15 | 補 downloader 外部 API fatal error path 後主表收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_error_cases.py` + `tools/downloader/universe.py` |
| 2026-04-02 | D63 | 新增 meta quality performance memory contract case 並驗證 | NEW -> DONE | `validate_meta_quality_performance_memory_contract_case` |
| 2026-04-02 | D64 | 新增 test suite meta quality memory reporting case 並驗證 | NEW -> DONE | `validate_test_suite_summary_meta_quality_memory_reporting_case` |
| 2026-04-02 | D21 | 補 traced peak memory regression gate 後 performance baseline 收斂完成 | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` + `core/runtime_utils.py` + `apps/test_suite.py` |
| 2026-04-02 | B19 | 將 traced peak memory 納入正式 gate，主表升為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py`, `core/runtime_utils.py`, `apps/test_suite.py` |

| 2026-04-02 | D42 | 新增 issue Excel report schema 案例並驗證 | NEW -> DONE | `validate_issue_excel_report_schema_case` |
| 2026-04-02 | D43 | 新增 portfolio export report artifacts 案例並驗證 | NEW -> DONE | `validate_portfolio_export_report_artifacts_case` |
| 2026-04-02 | D65 | 新增 portfolio_sim prepared tool contract case 並驗證 | NEW -> DONE | `validate_portfolio_sim_prepared_tool_contract_case` |
| 2026-04-02 | D66 | 新增 scanner prepared tool contract case 並驗證 | NEW -> DONE | `validate_scanner_prepared_tool_contract_case` |
| 2026-04-02 | D67 | 新增 debug trade log prepared tool contract case 並驗證 | NEW -> DONE | `validate_debug_trade_log_prepared_tool_contract_case` |
| 2026-04-02 | D68 | 新增 scanner reference clean-df contract case 並驗證 | NEW -> DONE | `validate_scanner_reference_clean_df_contract_case` |
| 2026-04-02 | D69 | 新增 meta quality coverage artifact reuse contract case 並驗證 | NEW -> DONE | `validate_meta_quality_reuses_existing_coverage_artifacts_case` |
| 2026-04-02 | D70 | 補 imported / defined validate cases 與 synthetic registry 完整一致 formal guard | TODO -> DONE | `tools/validate/synthetic_meta_cases.py` + `tools/validate/synthetic_cases.py` |
| 2026-04-02 | D71 | 補主表 / `F2` / `G` 收斂紀錄完整同步 formal guard | TODO -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | D72 | 補 checklist `DONE` 摘要缺漏自動偵測與阻擋 | TODO -> DONE | `tools/local_regression/run_meta_quality.py` |
| 2026-04-02 | B23 | 補齊 synthetic 主入口遺漏註冊與 registry completeness guard 後收斂為 DONE | PARTIAL -> DONE | `tools/validate/synthetic_cases.py` + `tools/validate/synthetic_meta_cases.py` |
| 2026-04-02 | B26 | 補齊 checklist main / `F2` / `G` sync 與 `DONE` 摘要缺漏 blocker 後收斂為 DONE | PARTIAL -> DONE | `tools/local_regression/run_meta_quality.py` + `doc/TEST_SUITE_CHECKLIST.md` |
| 2026-04-03 | D22 | 補 registry 反向對照 `F2` DONE validator 摘要完整性 guard，避免已註冊 validator 漏記於 checklist | DONE -> DONE | `tools/validate/synthetic_meta_cases.py` + `doc/TEST_SUITE_CHECKLIST.md` |
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

## H. 完成判準

可視為 test suite 已明顯收斂的最低條件：
1. `B1` 區 `P0` 項目全數至少達到 `DONE`。
2. `B1` 區其餘項目不得低於 `PARTIAL`，且缺口需可明確說明。
3. `B2` 區 `B11`、`B12`、`B13`、`B14`、`B18`、`B22` 至少達到 `PARTIAL`。
4. `B23`、`B24`、`B25`、`B26` 至少達到 `PARTIAL`，以證明 checklist 與 test suite 本身正確性已被納入收斂。
5. 有固定的 coverage baseline 與 reduced dataset regression baseline。
6. 可隨策略升級調整之測試，至少要覆蓋 model/schema/seed/reporting 四類介面契約。
7. 每輪開始時，需先判斷目前 checklist 是否仍足夠支撐本輪完整性判斷；此判斷也包含 checklist 自身是否仍有缺口、摘要失同步、狀態過舊或未回寫項目；若不足，須先更新 checklist 再進行後續驗證或修改。
8. 新增測試後，不得造成規則分叉、模組責任混亂或明顯效能退化。

## I. 收斂結案註記

截至 2026-04-02 最近兩輪 reduced suite 實測：
- `quick gate` 約 `10.22s`
- `consistency` 約 `53.91s ~ 59.52s`
- `chain checks` 約 `29.66s ~ 29.86s`
- `ml smoke` 約 `3.81s ~ 4.01s`
- `meta quality` 約 `1.41s ~ 9.82s`，其中同輪 coverage artifact reuse 生效後已降至約 `1.41s`
- `total` 約 `104.82s ~ 107.62s`

- 2026-04-02：`B19 / D21` 補記效能優化脈絡：去除 chain checks / consistency 內部重工、將 replay counts 併入第一次 timeline、移除 portfolio_sim 驗證 temp CSV 二次載入，並將 scanner / debug trade log 驗證改為共用 prepared context 與 precomputed stats。
- 2026-04-02：補記 reduced suite 最近兩輪實測基線約 `107.62s` 與 `104.82s`；`meta quality` 因 coverage artifact reuse 已降至約 `1.41s`。後續若要再動正式流程，需先以 profiling 證明存在明確固定重工或高 fanout 熱點。

目前正式 test suite 的 checklist / registry / formal-entry 自我驗證缺口已補齊：synthetic 主入口已補回遺漏註冊案例，`run_meta_quality.py` 與 meta registry case 也已正式阻擋主表 / `F2` / `G` 失同步與 `DONE` 摘要缺漏。後續若仍要以「大幅縮短整體測試時間」為目標，不應再直接沿著小型 adapter 細修；需先用 profiling 明確拆出 `consistency` 與 `chain checks` 的熱點，再決定是否值得動正式流程。

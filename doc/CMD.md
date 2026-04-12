
# 重現環境
python -m pip install -r requirements/requirements-lock.txt

# 打包
python apps/package_zip.py
python apps/package_zip.py --commit-message "feat: snapshot" --run-test-suite

# 資料集切換
python tools/validate/cli.py --dataset reduced
python tools/validate/cli.py --dataset full
python apps/portfolio_sim.py --dataset full
python apps/portfolio_sim.py --dataset reduced
python apps/vip_scanner.py --dataset full
python apps/vip_scanner.py --dataset reduced
python apps/ml_optimizer.py --dataset full
python apps/ml_optimizer.py --dataset reduced

# 環境變數切換
# validate 專用
set V16_VALIDATE_DATASET=reduced
# 主工具共用
set V16_DATASET_PROFILE=full
# optimizer 可重現
set V16_OPTIMIZER_SEED=20260401

# package_zip 可選擇先 git add -A + commit，再打包；`--run-test-suite` 會在打包後執行 `python apps/test_suite.py`
# package_zip 只會把 root 的非 bundle ZIP 移入 arch；`to_chatgpt_bundle_*.zip` 會留在 root，由 test_suite 自己維護最新 copy

# optimizer 架構
python apps/ml_optimizer.py --dataset full            # 正式入口
# apps/ml_optimizer.py 為薄入口，從 tools.optimizer 套件 façade 匯入 main
# tools/optimizer/__init__.py 統一匯出 optimizer 公開介面；main.py 負責 CLI/啟動，session.py 提供 session façade，prep.py / objective.py 為 façade，實作分散於 raw_cache.py / trial_inputs.py / objective_profiles.py / objective_filters.py / objective_runner.py / callbacks.py / runtime.py；其中 breakout 策略專屬參數契約、轉接層與搜尋空間已分別移至 strategies/breakout/schema.py、strategies/breakout/adapter.py、strategies/breakout/search_space.py
- `apps/ml_optimizer.py` 僅在完成指定訓練次數，或輸入 0 走提取匯出模式時，才會更新 `models/best_params.json`；若使用者中斷且未達指定次數，禁止自動覆寫。
- 系統評分由 `core/portfolio_stats.py::calc_portfolio_score()` 統一計算；`SCORE_CALC_METHOD` 控制主體算法，`SCORE_NUMERATOR_METHOD` 可在 `ANNUAL_RETURN` 與 `TOTAL_RETURN` 間切換分子，分母固定為 `|MDD| + 0.0001`。
- 若要固定 optimizer 搜尋路徑，可設 `V16_OPTIMIZER_SEED=<seed>`。
- 若要固定或關閉 optimizer 停利搜尋，請改 `config/training_policy.py` 的 `OPTIMIZER_FIXED_TP_PERCENT`；`None` 代表搜尋、`0.0` 代表固定關閉。

# 設定 / 參數架構
# config/training_policy.py：selection gate、EV/買入排序/score 類全域訓練政策、optimizer 固定停利比例與硬門檻
# config/execution_policy.py：共用資金、費用、複利、scanner/live capital 與 runtime 執行預設
# strategies/breakout/schema.py：breakout 策略專屬參數契約與 guardrail
# strategies/breakout/adapter.py：breakout 參數分層轉接與 section split
# strategies/breakout/search_space.py：breakout optimizer 搜尋空間
# core/strategy_params.py：聚合 breakout + selection + execution 的共用參數契約
# core/capital_policy.py：單股/投組/scanner 共用資金與 sizing 規則
# core/exact_accounting.py：正式整數 ledger / cost-basis allocation / tick-limit 正規化單一真理來源
# core/config.py：相容 façade；保留既有匯入路徑

# validate 架構
python tools/validate/cli.py --dataset reduced        # validate standalone CLI
# tools/validate/cli.py 為薄入口；總控在 tools/validate/main.py
# tools/validate/check_result_utils.py / portfolio_payloads.py / scanner_expectations.py / reporting.py 分別負責檢查結果記錄、投組 payload/年度欄位摘要、scanner 預期 payload/reference check、validate console summary / issue Excel / local regression summary JSON
# tools/validate/module_loader.py / tool_check_common.py / portfolio_tool_checks.py / external_tool_checks.py 分別負責模組動態載入、smoke check 共用工具、portfolio_sim smoke checks、scanner/downloader/debug smoke checks；checks.py / tool_adapters.py / tool_checks.py 僅保留 façade
# tools/validate/real_case_io.py / real_case_runners.py / real_case_assertions.py 分別負責真實 ticker 的 CSV/清洗、執行/掃描協調、cross-check 規則；real_cases.py 僅保留 façade
# synthetic_cases.py 負責 suite 入口、validator registry 與相容 façade；synthetic_portfolio_common.py / synthetic_take_profit_cases.py / synthetic_flow_cases.py / synthetic_portfolio_cases.py 分別負責 synthetic 投組共用 helper、半倉停利案例、流程/rotation 案例與 façade；synthetic_unit_cases.py 負責 price_utils / history_filters / portfolio_stats 邊界案例，以及 exact-accounting ledger / cost-basis allocation / integer tick-limit / cash-risk boundary / single-vs-portfolio parity 契約；synthetic_meta_cases.py 負責 TEST_SUITE_CHECKLIST / registry / synthetic 主入口一致性（GPT checklist 不列入本地 formal 驗證）、project import graph cycle guard 與 `doc/CMD.md` 指令契約案例；synthetic_error_cases.py 負責 params_io / module_loader / preflight / downloader 的 fail-fast 錯誤路徑；synthetic_data_quality_cases.py 負責髒資料清洗 expected behavior / fail-fast / `load_clean_df` 整合；synthetic_cli_cases.py 負責 dataset wrapper / local regression / no-arg CLI / 剩餘直接入口 CLI 契約；synthetic_display_cases.py 負責 scanner header / start banner / summary、strategy dashboard 與 core.display re-export output sanity；synthetic_contract_cases.py 負責 validate summary / optimizer profile / issue report 的 CSV/XLSX/JSON contract；synthetic_strategy_cases.py 另補 model I/O schema、ranking / scoring sanity、optimizer objective / export contract、strategy repeatability、minimum viability 與 reporting schema compatibility；synthetic_history_cases.py / synthetic_guardrail_cases.py / synthetic_param_cases.py 分別負責歷史門檻案例、guardrail 案例與 façade


## 資料下載

```bash
python apps/smart_downloader.py
```

下載正式入口為 `apps/smart_downloader.py`；入口層從 `tools.downloader` 套件 façade 匯入 `main` 與 `smart_download_vip_data`。`tools/downloader/main.py` 只負責總流程協調，`runtime.py` 管理共用設定 / lazy loader / issue log，`universe.py` 負責市場日期與海選，`sync.py` 負責 VIP 資料下載與最新日期跳過。


## 單股分析（trade_analysis）

```bash
python tools/trade_analysis/trade_log.py
```

`tools/trade_analysis/trade_log.py` 為單股 trade-analysis 正式入口；package `tools.trade_analysis` 與 `trade_log.py` 現已同步提供 canonical `run_trade_analysis` / `run_trade_backtest` / `run_prepared_trade_backtest` / `run_ticker_analysis` aliases，並為維持既有工具鏈相容保留 legacy `run_debug_*` / `debug_trade_log` 命名。`tools/workbench_ui/` 內部整合則優先使用 canonical `run_ticker_analysis` / `resolve_trade_analysis_data_dir` / `create_matplotlib_trade_chart_figure` aliases，不再把 legacy debug 命名當成 workbench 內部正式介面。`tools/trade_analysis/backtest.py` 只保留主控 façade，進場流程在 `tools/trade_analysis/entry_flow.py`，出場 / 錯失賣出 / 期末結算在 `tools/trade_analysis/exit_flow.py`，明細列建構 helper 在 `tools/trade_analysis/log_rows.py`，K 線 payload / HTML 輸出 / GUI 內嵌 matplotlib figure 在 `tools/trade_analysis/charting.py`，輸出摘要在 `tools/trade_analysis/reporting.py`。

## GUI 工作台

```bash
python apps/workbench.py
```

`apps/workbench.py` 是 GUI 單一啟用入口；GUI 工作台與頁籤 registry 在 `tools/workbench_ui/workbench.py`，目前內建 `單股回測檢視` 頁籤：K 線圖分頁預設佔主要版面、GUI 開啟即預設最大化，內嵌圖預設顯示最近 18 個月且可平移/縮放到完整歷史，色系採台股紅漲綠跌與深色背景，左上 hover 區會即時顯示滑鼠所在 K 棒的 OHLCV/限價/成交/停損/停利；買訊/賣訊以半透明註記顯示掛單或歷績資訊，並改由右側獨立 sidebar 呈現買入訊號 / 歷史績效符合 / 全歷史摘要 / 選取日線值與「回到最新 K 線」操作；成交量預設隱藏且可切換為疊加於同一圖面的下緣 overlay（高度低於 1/4），交易明細與 Console 為獨立分頁，控制列固定使用完整資料集，提供候選股掃描下拉選單，股票代號輸入後按 Enter 即執行回測、候選股下拉選取即直接回測；互動採 toolbar-free：滑鼠滾輪直接縮放、左鍵直接拖曳時間軸、左右鍵逐根移動。GUI 預設不預先輸出 HTML，點擊按鈕時才 lazy export HTML artifact，仍保留 Excel 與 HTML 輸出。


# portfolio engine 架構
# run_portfolio_timeline() 正式總控與最終整合仍在 core/portfolio_engine.py
# 當日候選池掃描、normal/extended 候選規格與排序在 core/portfolio_candidates.py
# 快取市場資料/PIT 索引在 core/portfolio_fast_data.py
# 日內操作 façade 在 core/portfolio_ops.py
# 盤前買進執行/延續訊號清理在 core/portfolio_entries.py
# 汰弱換股/持倉結算/期末結算在 core/portfolio_exits.py
# 曲線/年度/年化統計與分數在 core/portfolio_stats.py


# single-stock core 架構
# core/backtest_core.py：單股 K 棒推進與回測總控
# core/exact_accounting.py：正式整數 ledger / cost-basis allocation / tick-limit 正規化單一真理來源
# core/price_utils.py：跳價/成交價/股數/成本/漲跌停與賣出阻塞判斷 façade
# core/signal_utils.py：技術指標與訊號生成
# core/trade_plans.py：候選/掛單/延續訊號 façade
# core/history_filters.py：歷史績效候選門檻
# core/entry_plans.py：候選規格、盤前掛單規格、成交後 first-actionable stop/tp 與 entry-day trigger queue 建立
# core/extended_signals.py：延續候選固定反事實 entry-ref / invalidation-completion barrier / today-orderable 規格

- validate 子模組已再拆分：`synthetic_history_cases.py`、`synthetic_guardrail_cases.py`、`synthetic_frame_utils.py`、`synthetic_case_builders.py`。


# apps 入口層
# apps/portfolio_sim.py 為薄入口，從 tools.portfolio_sim 套件 façade 匯入公開介面；main.py 負責 CLI/互動流程，runtime.py 為 façade，runtime_common.py 負責共用路徑/參數載入/runtime 目錄/不足資料判定，simulation_runner.py 負責預載入與 timeline 執行，reporting.py 負責年度報酬 / Excel / Plotly 輸出
# apps/vip_scanner.py 為薄入口，從 tools.scanner 套件 façade 匯入公開介面；main.py 為 façade，scan_runner.py 負責 CLI/平行掃描，worker.py 為 façade，runtime_common.py 負責共用路徑/runtime 目錄/參數載入/worker 數判定，stock_processor.py 負責單股掃描 worker，reporting.py 負責啟動/摘要/候選清單輸出
# apps/workbench.py 為單一 GUI 啟用入口，從 tools.workbench_ui 套件 façade 匯入 main；workbench / panel registry 在 tools/workbench_ui/workbench.py，單股回測檢視頁籤在 tools/workbench_ui/single_stock_inspector.py

# display 架構
# core/display.py 為 façade；display_common.py 負責 ANSI 色彩/表格與共用 helper，scanner_display.py 負責 scanner header，strategy_dashboard.py 負責策略 dashboard 與對比表


- 單股核心已拆分為 `core/position_step.py` 與 `core/backtest_finalize.py`，對外仍由 `core/backtest_core.py` 提供 façade。

# 命名 / 結構收尾原則
# apps/* 僅從對應子系統套件 façade 匯入公開介面，避免入口層直接依賴更深子模組
# tools/*/__init__.py 與 façade 檔維持穩定公開介面；子模組可繼續細拆，但外部匯入路徑應盡量不變
# 模組檔名已完成版本前綴移除；既有函式／類別識別字若仍含版本字樣，屬另案處理範圍


## 一鍵測試（reduced）

日常一鍵測試固定只用 reduced；正式對外入口為 `apps/test_suite.py`，正式組成步驟定義以 `tools/local_regression/formal_pipeline.py` 為準。

### 一鍵執行

```bash
python apps/test_suite.py
```

Windows：

```bat
python apps\test_suite.py
```

若只想單獨執行底層 orchestrator：

```bash
python tools/local_regression/run_all.py
```

若完整入口已找出失敗步驟，可只重跑指定步驟：

```bash
python tools/local_regression/run_all.py --only quick_gate
python tools/local_regression/run_all.py --only consistency,ml_smoke
```

若只想先檢查目前 Python 環境是否已具備 `requirements/requirements.txt` 所需套件：

```bash
python tools/validate/preflight_env.py
```

- `preflight_env.py` 只做檢查，不自動安裝依賴。
- `python tools/validate/preflight_env.py --steps quick_gate,consistency` 可只檢查指定步驟所需套件；但 `quick_gate` 內含 optimizer export-only 錯誤路徑，所以仍需要 `optuna` 與 `SQLAlchemy`。
- `python apps/test_suite.py` 與 `python tools/local_regression/run_all.py` 都會先執行這個 preflight；若缺件會先 fail-fast，不進入後續 reduced 測試。兩者都會串接所有已實作測試，包含 `meta_quality`。`run_chain_checks.py` 目前也會把 scanner reduced snapshot 納入雙跑 digest；`run_ml_smoke.py` 會以固定 seed 做雙跑；`synthetic_regression_cases.py` 另外補齊 `scan_runner` 入口重跑一致性、optimizer raw cache rerun / mutation isolation 與 `run_all.py` 同 run dir rerun summary / bundle repeatability。`meta_quality_targets.py` 集中 coverage target / floor 常數，`meta_quality_coverage.py` 集中 coverage summary / reuse / formal helper probe，共用給 `run_meta_quality.py` 與 synthetic meta cases；`meta_quality` 會讀取同輪 step summary 與 optimizer profile summary，檢查 reduced performance baseline，並校驗 `tools/local_regression/formal_pipeline.py` / `run_all.py` / `preflight_env.py` / `apps/test_suite.py` 的正式步驟一致性；`apps/test_suite.py` 結果摘要格式則統一收斂於 `core/test_suite_reporting.py`，避免 `tools/` 反向依賴 `apps/`；文件分工固定為 `PROJECT_SETTINGS.md` 管上層原則、`TEST_SUITE_CHECKLIST.md` 管本地端 formal test suite、`GPT_DELIVERY_CHECKLIST.md` 管 GPT 交付前操作檢查表。
- 日常流程先跑完整入口；只有完整入口已找出 FAIL 步驟時，才使用 `python tools/local_regression/run_all.py --only ...` 重跑指定步驟。`meta_quality` 也已納入完整入口。

### 輸出位置

`apps/test_suite.py` 執行完成後：

```text
outputs/local_regression/
  to_chatgpt_bundle_<timestamp>_<uniqueid>.zip   # 歷史 bundle

<repo>/
  to_chatgpt_bundle_<timestamp>_<uniqueid>.zip   # 根目錄最新 copy
```

- `outputs/local_regression/` 保留歷史 bundle。
- 專案根目錄只保留最新一份同名 copy，方便直接上傳給 ChatGPT。
- `apps/test_suite.py` 結束時會在主控台印出整體 PASS/FAIL、五個步驟摘要（quick gate / consistency / chain checks / ml smoke / meta quality）、全量 ticker 摘要，以及兩個 bundle 路徑。
- `tools/local_regression/manifest.json` 內的 `performance_*_max_sec` 與 `performance_optimizer_trial_avg_max_sec` 可調整 reduced performance baseline 門檻。

### reduced 資料集

local regression 與 validate 的 reduced 測試資料來源固定為：
- `<repo>/data/tw_stock_data_vip_reduced`

reduced dataset 直接以 `<repo>/data/tw_stock_data_vip_reduced` 目前實際存在的 CSV 快照為準；formal pipeline 只要求資料夾非空，並對當前 members / content 產生 fingerprint，不再把成員名單或筆數寫死在程式中。

不再支援 `data.zip`、`V16_PROJECT_DATA_ZIP` 或 `/mnt/data/data.zip` 回退來源。
打包給 ChatGPT 前，請先確認該資料夾已隨專案一併納入 ZIP。


### apps 入口

正式測試入口已收斂為 `apps/test_suite.py`；日常 GUI 單一啟用入口為 `apps/workbench.py`；`apps/` 不再保留舊的 regression / consistency 測試入口。


## Test suite bundle

`python apps/test_suite.py` 會先在 staging 組裝結果，再打成單一 bundle。

- PASS：bundle 只含 minimum set 摘要檔。
- FAIL：bundle 自動擴充為 debug bundle，納入失敗步驟所需除錯材料。
- 歷史 bundle 保留在 `outputs/local_regression/`；根目錄只保留最新一份同名 copy。
- 內部 staging 目錄打包完成後自動刪除，不保留散開 json、log、latest 或 runs 供日常查看。

## 其它工具輸出分類

- `outputs/validate_consistency/`：standalone consistency 報表。
- `outputs/ml_optimizer/`：optimizer profiling / 載入摘要。
- `outputs/portfolio_sim/`：投組報表與載入摘要。
- `outputs/vip_scanner/`：scanner issue log。
- `outputs/debug_trade_log/`：`trade_analysis` 單股分析輸出；為維持既有工具鏈相容，暫沿用 legacy 目錄名 `debug_trade_log`，內含 Excel 明細與 HTML K 線檢視。
- `outputs/smart_downloader/`：下載器 issue log。


## Output retention

`python apps/test_suite.py` 結束後會自動執行 output retention，不需額外手動清理。
預設規則：
- `outputs/local_regression/` 歷史 bundle：保留最近 20 份，刪除超過 30 天
- `validate_consistency`、`portfolio_sim`：保留最近 10 份，刪除超過 30 天
- `ml_optimizer`、`vip_scanner`、`smart_downloader`、`debug_trade_log`（trade_analysis legacy output dir）：保留最近 5 份，刪除超過 14 天


- `tools/validate/synthetic_contract_cases.py` 亦負責 artifact lifecycle contract：bundle、archive、root copy、retention。


- `tools/validate/synthetic_reporting_cases.py`：檢查 validate console summary、issue Excel report、portfolio yearly/export report、`core/test_suite_reporting.py` / `apps/test_suite.py` 結果摘要的 reporting schema / 格式相容性。

# 專案架構說明

本文件說明目前專案檔案樹、各檔案用途與依賴原則。

## 檔案樹

```text
project/
├─ .gitignore                         # Git 忽略規則
├─ apps/
│  ├─ __init__.py                     # apps 套件 façade；統一匯出正式入口
│  ├─ ml_optimizer.py                 # 參數最佳化正式入口（薄入口）
│  ├─ portfolio_sim.py                # 投組模擬正式入口（薄入口）
│  ├─ smart_downloader.py             # 資料下載正式入口（薄入口）
│  ├─ package_zip.py                  # 專案打包正式入口
│  ├─ test_suite.py                   # 一鍵測試正式入口（reduced）
│  ├─ vip_scanner.py                  # 掃描器正式入口（薄入口）
│  └─ workbench.py                    # GUI 工作台正式入口（薄入口）
├─ config/
│  ├─ __init__.py                     # 純設定資料套件
│  ├─ training_policy.py              # selection gate、EV/買入排序/score、optimizer 固定停利比例與硬門檻
│  └─ execution_policy.py             # 共用資金、費用、複利與 runtime 執行預設
├─ core/
│  ├─ __init__.py                     # 套件初始化檔
│  ├─ buy_sort.py                 # 買入候選排序邏輯
│  ├─ config.py                   # 相容 façade；穩定匯出設定常數、訓練政策、參數契約與資金規則
│  ├─ strategy_params.py          # 共用聚合參數契約（breakout + training gate + execution）
│  ├─ capital_policy.py           # 單股/投組/scanner 的資金與 sizing 規則
│  ├─ backtest_core.py                     # 單股策略核心：K棒推進與單股回測總控
│  ├─ exact_accounting.py         # 正式帳務精確整數 ledger、cost-basis allocation、tick/limit 正規化單一真理來源
│  ├─ price_utils.py              # 跳價/成交價/成本/股數/漲跌停與賣出阻塞判斷單一口徑
│  ├─ signal_utils.py             # 單股技術指標與訊號生成
│  ├─ trade_plans.py              # 單股候選/掛單/延續訊號 façade（保留穩定匯入路徑）
│  ├─ history_filters.py          # 歷史績效候選門檻
│  ├─ entry_plans.py              # 候選規格、盤前掛單規格、成交後 first-actionable stop/tp 與 entry-day trigger queue 建立
│  ├─ extended_signals.py         # 延續候選固定反事實 entry-ref / barrier expiry / today-orderable 規格
│  ├─ data_utils.py               # OHLCV 清理與共用資料工具
│  ├─ dataset_profiles.py         # 資料集模式/CLI/ENV 解析與路徑切換單一入口
│  ├─ model_paths.py              # models 目錄與 best_params 路徑解析 helper
│  ├─ output_paths.py             # outputs/<category> 目錄正規化與建立 helper
│  ├─ display.py                  # 顯示 façade：統一匯出色彩/表格/scanner header/strategy dashboard
│  ├─ display_common.py           # ANSI 色彩、顯示寬度、表格 row 與通用參數讀取 helper
│  ├─ scanner_display.py          # scanner header 與訓練/濾網/硬門檻輸出
│  ├─ strategy_dashboard.py       # 策略 dashboard、對比表與硬門檻輸出
│  ├─ log_utils.py                # logging 與輸出輔助工具
│  ├─ params_io.py                # 參數讀寫、json 載入/匯出
│  ├─ portfolio_engine.py         # 投組核心 timeline 總控與最終整合
│  ├─ portfolio_candidates.py     # 投組候選池掃描、normal/extended 候選規格與排序
│  ├─ portfolio_fast_data.py      # 投組快取市場資料、mark-to-market 與歷史 PIT 統計索引單一口徑
│  ├─ portfolio_entries.py        # 投組進場流程：盤前買進執行、延續訊號清理
│  ├─ portfolio_exits.py          # 投組出場流程：汰弱換股、持倉結算、期末結算
│  ├─ portfolio_ops.py            # 投組日內操作 façade：統一匯出 entries / exits 介面
│  ├─ portfolio_stats.py          # 投組曲線/年度/年化統計與分數計算單一口徑
│  ├─ position_step.py            # 單股持倉 K 棒步進與出場事件處理
│  ├─ backtest_finalize.py        # 單股回測期末結算與統計彙整
│  ├─ runtime_utils.py            # 執行期共用工具：ProcessPool 啟動方法、Asia/Taipei 時間工具
│  └─ test_suite_reporting.py     # test suite 結果摘要格式與共用顯示 helper
├─ doc/
│  ├─ ARCHITECTURE.md                 # 本檔；檔案樹、用途與依賴原則說明
│  ├─ CMD.md                          # 常用指令與操作說明
│  ├─ FINMIND_API_TOKEN.md            # API token 說明
│  ├─ GPT_DELIVERY_CHECKLIST.md       # GPT 交付前操作檢查表（不納入本地 formal 驗證）
│  ├─ PROJECT_SETTINGS.md             # 專案最高優先規則文件
│  ├─ TEST_SUITE_CHECKLIST.md         # test suite 收斂清單；區分長期固定測試與可隨策略升級調整之測試
│  └─ ToDo.md                         # 待辦事項與後續整理筆記
├─ data/
│  ├─ tw_stock_data_vip/              # 專案內保留的資料佔位與名單檔
│  └─ tw_stock_data_vip_reduced/      # reduced 測試資料快照；local regression / validate 直接使用，members / fingerprint 由目錄現況動態決定
├─ models/
│  ├─ all_best_params_1.json         # 特定評分口徑下的最佳參數紀錄
│  ├─ all_best_params_2.json         # 特定評分口徑下的最佳參數紀錄
│  ├─ all_best_params_3.json         # 歷史最佳參數或不同批次最佳化輸出
│  └─ best_params.json               # 目前主要使用的最佳參數檔
├─ strategies/
│  ├─ __init__.py                     # 策略命名空間
│  └─ breakout/
│     ├─ __init__.py                  # breakout 策略 façade
│     ├─ schema.py                    # breakout 策略參數契約、defaults、guardrail
│     ├─ adapter.py                   # breakout 參數分層轉接與 section split
│     └─ search_space.py              # breakout optimizer 搜尋空間
├─ requirements/
│  ├─ export_requirements_lock.py     # 輸出 requirements lock 的輔助腳本
│  ├─ requirements-lock.txt           # 鎖版本套件清單
│  └─ requirements.txt                # 主要相依套件清單
└─ tools/
   ├─ __init__.py                     # tools 套件初始化檔
   ├─ downloader/
   │  ├─ __init__.py                  # downloader 子套件初始化檔
   │  ├─ main.py                      # 下載正式主控；CLI 與總流程協調
   │  ├─ runtime.py                   # downloader 共用設定、lazy loader、log 與執行期狀態
   │  ├─ universe.py                  # 最新交易日判定、universe 快取與海選
   │  └─ sync.py                      # VIP 資料下載與最新日期跳過邏輯
   ├─ portfolio_sim/
   │  ├─ __init__.py                  # portfolio_sim 子套件 façade；統一匯出模擬公開介面
   │  ├─ main.py                      # portfolio_sim CLI 主控與互動流程
   │  ├─ runtime.py                   # portfolio_sim runtime façade
   │  ├─ runtime_common.py            # portfolio_sim 共用路徑/參數/目錄與不足資料判定
   │  ├─ simulation_runner.py         # portfolio_sim 預載入與 timeline 執行
   │  └─ reporting.py                 # portfolio_sim 年度報酬、Excel 與 Plotly 輸出
   ├─ scanner/
   │  ├─ __init__.py                  # scanner 子套件 façade；統一匯出掃描公開介面
   │  ├─ main.py                      # scanner CLI façade
   │  ├─ scan_runner.py               # scanner CLI 主控與平行掃描流程
   │  ├─ worker.py                    # scanner worker façade
   │  ├─ runtime_common.py            # scanner 共用路徑/參數/目錄與 worker 數判定
   │  ├─ stock_processor.py           # 單股掃描 worker、候選排序值與參考投入計算
   │  └─ reporting.py                 # scanner 啟動/摘要/候選清單輸出
   ├─ trade_analysis/
   │  ├─ __init__.py                  # trade_analysis 子套件 façade；統一匯出單股分析公開介面
   │  ├─ backtest.py                  # 交易回放與明細列建構
   │  ├─ charting.py                  # 單股分析 K 線 payload、HTML 輸出與 GUI 內嵌 matplotlib figure
   │  ├─ entry_flow.py                # 單股分析買進 / 延續候選進場流程
   │  ├─ exit_flow.py                 # 單股分析出場 / 錯失賣出 / 期末結算流程
   │  ├─ history_snapshot.py          # 單股分析歷史績效 snapshot / payoff / asset-growth helper
   │  ├─ log_rows.py                  # 單股分析明細列建構與半倉停利價 helper
   │  ├─ reporting.py                 # 單股分析報表輸出與虧損摘要
   │  └─ trade_log.py                 # 單股分析入口、資料集解析與對外包裝
   ├─ workbench_ui/
   │  ├─ __init__.py                  # workbench_ui 子套件 façade；統一匯出工作台入口
   │  ├─ main.py                      # 工作台啟動 façade
   │  ├─ workbench.py                 # 單一視窗 workbench 與 panel registry
   │  └─ single_stock_inspector.py    # 單股回測檢視頁籤；內嵌 K 線圖與交易明細表
   ├─ optimizer/
   │  ├─ __init__.py                  # optimizer 子套件 façade；統一匯出 optimizer 公開介面
   │  ├─ main.py                      # 最佳化主流程、study 控制與歷史最佳還原
   │  ├─ prep.py                      # optimizer 預處理 façade：統一匯出原始資料快取與 trial 輸入準備
   │  ├─ raw_cache.py                 # optimizer 原始資料快取、資料清洗與載入摘要
   │  ├─ trial_inputs.py              # optimizer worker 預處理、平行/回退流程與 trial 輸入整合
   │  ├─ objective.py                 # optimizer objective façade：統一匯出 trial 參數/profile/filter/objective runner
   │  ├─ objective_profiles.py        # optimizer façade：策略 trial 參數抽樣 + 初始 profile row 建構
   │  ├─ objective_filters.py         # optimizer filter rules
   │  ├─ objective_runner.py          # trial 級評分流程、portfolio timeline 執行與 user_attrs 寫回
   │  ├─ callbacks.py                 # monitoring callback、profiling console print 與破紀錄展示
   │  ├─ profile.py                   # optimizer profiling CSV/JSON 輸出與摘要
   │  └─ study_utils.py               # trial / study / 參數還原共用工具
   └─ validate/
      ├─ __init__.py                  # validate 子套件 façade；統一匯出 validate 公開介面
      ├─ check_result_utils.py        # validate 檢查結果記錄、ticker 正規化與共用失敗/跳過判定
      ├─ checks.py                    # validate checks façade：統一匯出結果/統計/scanner 預期 helpers
      ├─ cli.py                       # validate standalone CLI 薄入口
      ├─ main.py                      # 一致性驗證總控；資料集解析、真實掃描協調與結果彙整
      ├─ meta_contracts.py            # checklist / 文件 / registry 的 markdown 與 AST contract helper
      ├─ real_case_assertions.py      # 真實 ticker 驗證的 cross-check 規則與比對項目
      ├─ real_case_io.py              # 真實 ticker 驗證的 CSV 路徑解析與資料清洗載入
      ├─ real_case_runners.py         # 真實 ticker 驗證執行、單股/單檔投組檢查與掃描協調
      ├─ real_cases.py                # 真實 ticker 驗證 façade：統一匯出 io / runners 介面
      ├─ module_loader.py            # validate 模組動態載入、可恢復例外與模組快取
      ├─ portfolio_payloads.py        # validate 投組 payload、年度欄位與 completed trade 摘要 helper
      ├─ reporting.py                 # validate 報表輸出、console summary 與 local regression summary JSON
      ├─ scanner_expectations.py      # scanner 預期 payload / reference check helper
      ├─ synthetic_cases.py           # synthetic suite 入口、validator registry 與相容 façade
      ├─ synthetic_fixtures.py        # synthetic 測試資料與案例生成
      ├─ synthetic_param_cases.py     # synthetic 參數 guardrail / 排序與歷史門檻案例
      ├─ synthetic_flow_cases.py       # synthetic 延續候選/競爭候選/同日賣出封鎖/rotation 案例
      ├─ synthetic_portfolio_cases.py # synthetic 投組案例 façade：統一匯出各 case validator
      ├─ synthetic_take_profit_cases.py # synthetic 半倉停利/不可執行半倉停利案例
      ├─ synthetic_unit_cases.py        # synthetic unit-like 邊界案例：price_utils / history_filters / portfolio_stats / exact-accounting ledger-cost-basis-tick-limit parity
      ├─ synthetic_meta_cases.py        # synthetic meta 案例：TEST_SUITE_CHECKLIST / registry / synthetic 主入口一致性（`doc/GPT_DELIVERY_CHECKLIST.md` 不納入本地 formal 驗證）
      ├─ synthetic_error_cases.py       # synthetic 錯誤路徑案例：params_io / module_loader / preflight / downloader fail-fast
      ├─ synthetic_data_quality_cases.py# synthetic 資料品質案例：髒資料清洗 expected behavior / fail-fast / load_clean_df 整合
      ├─ synthetic_display_cases.py     # synthetic 顯示契約案例：scanner header / dashboard / display re-export output sanity
      ├─ synthetic_cli_cases.py         # synthetic CLI 契約案例：dataset wrapper / local regression / no-arg CLI
      ├─ synthetic_reporting_cases.py   # synthetic 報表契約案例：validate / portfolio / test suite summary schema
      ├─ synthetic_strategy_cases.py    # synthetic 可隨策略升級調整案例：model I/O schema / ranking & scoring sanity / optimizer objective-export contract / strategy repeatability / minimum viability / reporting schema compatibility
      ├─ tool_adapters.py             # validate 對 apps / trade_analysis 工具的動態載入 façade
      ├─ tool_check_common.py         # smoke check 共用輸出抑制與日期欄位解析
      ├─ tool_checks.py               # smoke check façade
      ├─ portfolio_tool_checks.py     # portfolio_sim smoke checks
      ├─ external_tool_checks.py      # scanner/downloader/debug smoke checks
      ├─ preflight_env.py             # requirements 依賴 preflight 檢查
      └─ trade_rebuild.py             # trade log / completed trade 重建工具
```

## 分層原則

- `apps/`：正式執行入口，只負責 CLI、流程組裝與執行期 bootstrap，不得在入口層重寫核心交易規則；`apps/ml_optimizer.py` 現為薄入口，最佳化主流程已拆成 `tools/optimizer/main.py`（CLI/啟動）、`session.py`（session 狀態 façade）、`prep.py` / `raw_cache.py` / `trial_inputs.py`（原始資料快取、worker 預處理與 trial 輸入整合）、`objective.py` / `objective_profiles.py` / `objective_filters.py` / `objective_runner.py`（optimizer façade / 初始 profile / filter rules / objective runner），其中 breakout 策略專屬參數契約、轉接層與搜尋空間已分別移至 `strategies/breakout/schema.py`、`strategies/breakout/adapter.py` 與 `strategies/breakout/search_space.py`，投組層 / scanner 歷史績效門檻已併入 `config/training_policy.py`，共用資金 / 費用 / 複利設定集中於 `config/execution_policy.py`，全域 score / threshold 與 `OPTIMIZER_FIXED_TP_PERCENT` 則集中於 `config/training_policy.py`；`callbacks.py`（monitoring / 破紀錄展示）與 `runtime.py`（記憶庫流程 / 歷史最佳還原 / 匯出控制）；`apps/smart_downloader.py` 現為薄入口，下載流程已拆成 `tools/downloader/main.py`（總控）、`runtime.py`（共用設定 / lazy loader / issue log）、`universe.py`（市場日期與海選）與 `sync.py`（VIP 資料下載與更新跳過）；`apps/portfolio_sim.py` 現為薄入口，模擬流程已拆成 `tools/portfolio_sim/main.py`（CLI/互動流程）、`runtime.py` façade、`runtime_common.py`（共用路徑 / 參數載入 / runtime 目錄 / 不足資料判定）、`simulation_runner.py`（預載入與 timeline 執行）與 `reporting.py`（年度報酬 / Excel / Plotly 輸出）；`apps/vip_scanner.py` 現為薄入口，掃描流程已拆成 `tools/scanner/main.py` façade、`scan_runner.py`（CLI/平行掃描）、`worker.py` façade、`runtime_common.py`（共用路徑 / runtime 目錄 / 參數載入 / worker 數判定）、`stock_processor.py`（單股掃描 worker）與 `reporting.py`（啟動/摘要/候選清單輸出）；`apps/workbench.py` 為單一 GUI 啟用入口，僅匯入 `tools.workbench_ui.main`；GUI 工作台與後續頁籤擴充集中在 `tools/workbench_ui/workbench.py` 的 panel registry，單股回測檢視頁籤在 `tools/workbench_ui/single_stock_inspector.py`，K 線圖分頁預設佔主要版面且 GUI 開啟即預設最大化，成交量預設隱藏並於需要時以同圖 overlay 呈現，交易明細與 Console 改以獨立分頁承接；workbench panel registry 與 inspector 內部整合優先使用 canonical `run_ticker_analysis` / `resolve_trade_analysis_data_dir` / `create_matplotlib_trade_chart_figure` aliases，legacy `run_debug_*` / `create_matplotlib_debug_chart_figure` 僅保留相容用途；`apps/package_zip.py` 為專案打包正式入口，會清除 Python 快取、將 root 的非 bundle 舊 ZIP 移到 `arch/`、保留 `to_chatgpt_bundle_*.zip` 於 root，並以目前 working tree 的 tracked/untracked 非忽略檔建立乾淨 ZIP，強制排除 `__pycache__/` 與 `*.pyc`；若提供 `--commit-message`，會先 `git add -A` + `git commit -m ...`，若再加 `--run-test-suite`，則於打包後執行 `apps/test_suite.py`。
- `core/exact_accounting.py`：正式 `cash/pnl/equity/reserved_cost/risk` 與 partial-exit cost-basis allocation 的精確整數帳務單一真理來源；`core/price_utils.py` 僅保留價格/tick/limit façade 與顯示相容 helper；`core/model_paths.py` / `core/output_paths.py` 分別集中 models/best_params 路徑解析與 `outputs/<category>` 目錄正規化。
- `tools/validate/`：一致性驗證子系統，已拆成 `check_result_utils.py`（檢查結果記錄 / ticker 正規化 / 可恢復錯誤判定）、`portfolio_payloads.py`（投組 payload / 年度欄位 / completed trade 摘要）、`scanner_expectations.py`（scanner 預期 payload / reference check）、`meta_contracts.py`（checklist / 文件 / registry 的 markdown 與 AST contract helper）、`module_loader.py`（模組動態載入與快取）、`tool_check_common.py`（smoke check 共用輸出抑制與日期欄位解析）、`portfolio_tool_checks.py`（portfolio_sim smoke checks）、`external_tool_checks.py`（scanner/downloader/debug smoke checks）、`tool_checks.py`（smoke check façade）、`checks.py` / `tool_adapters.py` façade、`synthetic_cases.py`（含 synthetic validator metadata registry、suite 入口與相容 façade）、`synthetic_portfolio_common.py`、`synthetic_take_profit_cases.py`、`synthetic_flow_cases.py`、`synthetic_portfolio_cases.py` façade、`synthetic_unit_cases.py`（price_utils / history_filters / portfolio_stats 邊界案例）、`synthetic_meta_cases.py`（TEST_SUITE_CHECKLIST / registry / synthetic 主入口一致性；`doc/GPT_DELIVERY_CHECKLIST.md` 不納入本地 formal 驗證）、`synthetic_error_cases.py`（params_io / module_loader / preflight / downloader fail-fast 錯誤路徑）、`synthetic_data_quality_cases.py`（髒資料清洗 expected behavior / fail-fast / `load_clean_df` 整合）、`synthetic_display_cases.py`（scanner header / start banner / summary、strategy dashboard 與 display re-export output sanity）、`synthetic_cli_cases.py`（dataset wrapper / local regression / no-arg CLI / 剩餘直接入口 CLI 契約）、`synthetic_reporting_cases.py`（validate / portfolio / test suite summary reporting schema）、`synthetic_strategy_cases.py`（model I/O schema / ranking & scoring sanity / optimizer objective-export contract / strategy repeatability / minimum viability / reporting schema compatibility）、`synthetic_history_cases.py`、`synthetic_guardrail_cases.py`、`synthetic_param_cases.py` façade、`synthetic_frame_utils.py`、`synthetic_case_builders.py`、`synthetic_fixtures.py` façade、`trade_rebuild.py`、`reporting.py`（validate console summary / issue Excel / local regression summary JSON）、`real_case_assertions.py`、`real_case_io.py`、`real_case_runners.py`、`real_cases.py` façade；`main.py` 僅保留資料集解析、真實掃描協調、synthetic suite 觸發與結果彙整，真實 ticker 驗證已再拆成 `real_case_io.py`（CSV 路徑解析 / 資料清洗）與 `real_case_runners.py`（單股 / 單檔投組 / 真實掃描協調），cross-check 規則集中到 `real_case_assertions.py`；synthetic 投組案例已再拆成 `synthetic_take_profit_cases.py`（半倉停利相關）與 `synthetic_flow_cases.py`（延續候選 / 競爭候選 / 同日賣出封鎖 / rotation），`synthetic_portfolio_cases.py` 僅保留 façade。
- `tools/trade_analysis/`：單股 trade-analysis 子系統；package 與 `trade_log.py` 對外已同步提供 canonical `run_trade_analysis` / `run_trade_backtest` / `run_prepared_trade_backtest` / `run_ticker_analysis` aliases，並為維持相容性保留 legacy `run_debug_*` / `debug_trade_log` 命名。`trade_log.py` 保留 CLI、資料集解析與 GUI 共用後端封裝，`backtest.py` 只保留單股分析主控 façade，買進 / 延續候選進場已拆到 `entry_flow.py`，出場 / 錯失賣出 / 期末結算已拆到 `exit_flow.py`，歷史績效 snapshot / payoff / asset-growth helper 集中到 `history_snapshot.py`，明細列建構與半倉停利價 helper 集中到 `log_rows.py`，`charting.py` 專責 K 線 payload / HTML 輸出 / GUI 內嵌 figure、完整歷史視窗導航、預設最近 18 個月視窗、台股紅漲綠跌色系、可視區間 autoscale、滑鼠滾輪縮放 / 左鍵拖曳平移、左上即時 hover OHLCV 顯示、買訊/賣訊半透明註記、停損/停利/限價/成交線（未成交前僅預覽限價）與右側摘要/右下狀態框、同圖成交量 overlay 與中文字型 fallback；GUI 預設 path 改為只回傳 chart payload、不預先輸出 HTML，以降低互動記憶體與載入成本，`reporting.py` 專責 Excel / HTML artifact 與虧損摘要。
- `tools/workbench_ui/`：GUI 工作台子系統；`main.py` 只負責啟動，`workbench.py` 維護單一視窗與頁籤 registry，`single_stock_inspector.py` 承接單股回測檢視頁籤與內嵌圖表 canvas / 右側 sidebar / 候選股掃描下拉選單 / Console 分頁；圖表分頁預設優先顯示且 GUI 啟動即最大化，K 線預設顯示最近 18 個月並可滑動/縮放到完整歷史，右側 sidebar 統一承接買入訊號、歷史績效符合、全歷史摘要、選取日線值與回到最新 K 線操作，股票代號輸入後按 Enter 即執行回測、候選股下拉選取即直接回測；成交量可切換為同圖 overlay、交易明細與 Console 以獨立分頁承接，HTML K 線圖改為按鈕點擊時才 lazy export，並採 toolbar-free 互動：滑鼠滾輪直接縮放、左鍵直接拖曳時間軸、左右鍵逐根移動，避免 GUI 與交易邏輯耦合。
- `core/`：核心規則與共用計算，應作為單一真理來源；目前 `portfolio_engine.py` 已只保留 `run_portfolio_timeline()` 總控與最終整合，當日候選池掃描 / normal / extended 候選規格與排序已抽至 `portfolio_candidates.py`，快取市場資料/PIT 統計索引抽至 `portfolio_fast_data.py`，日內操作 façade 保留於 `portfolio_ops.py`，其中盤前買進執行/延續訊號清理抽至 `portfolio_entries.py`，汰弱換股/持倉結算/期末結算抽至 `portfolio_exits.py`，曲線/年度/年化統計與分數計算抽至 `portfolio_stats.py`；全域訓練政策改由 `config/training_policy.py` 集中提供，training policy / execution policy 分別由 `config/training_policy.py`、`config/execution_policy.py` 集中提供，其中 training policy 也包含投組層 / scanner 歷史績效門檻，`core/strategy_params.py` 聚合 breakout + training gate + execution 的共用參數契約，`core/config.py` 只保留相容 façade。`backtest_core.py` 已縮為單股 K 棒推進與回測總控；單棒持倉步進抽至 `position_step.py`，回測收尾與統計彙整抽至 `backtest_finalize.py`；跳價/成本/股數/漲跌停口徑抽至 `price_utils.py`，技術指標與訊號生成抽至 `signal_utils.py`，`trade_plans.py` 已縮為 façade，歷史績效候選門檻抽至 `history_filters.py`，候選/掛單/成交後部位建立與 entry-day stop/tp 觸發排程抽至 `entry_plans.py`，延續候選固定反事實 entry-ref / invalidation-completion barrier / today-orderable 規格抽至 `extended_signals.py`。`display.py` 已縮為 façade，ANSI 色彩/表格與共用參數 helper 抽至 `display_common.py`，scanner header 輸出抽至 `scanner_display.py`，strategy dashboard 與對比表輸出抽至 `strategy_dashboard.py`。
- `tools/`：除錯、驗證與開發輔助工具；可呼叫核心邏輯，但不得成為正式交易規則唯一來源。
- `doc/`：文件與規則說明，以 `PROJECT_SETTINGS.md` 為最高優先規則文件；`TEST_SUITE_CHECKLIST.md` 為本地端 formal test suite 收斂主清單，維護長期固定測試、可隨策略升級調整之測試、優先級與狀態；`GPT_DELIVERY_CHECKLIST.md` 為 assistant 交付前操作檢查表，不作 formal test 主表或狀態真理來源，亦不納入 `apps/test_suite.py`、本地端 formal validator、synthetic registry 或 bundle 檢查。
- `models/`：參數結果與最佳化輸出，不放正式交易邏輯。
- 執行期資料集預設優先使用 `/data/`；若執行環境不存在 `/data/`，則自動退回 `project/data/`。因此 Linux 可直接使用 `/data/...`，Windows 等無 `/data` 的環境可直接使用專案根目錄下的 `data/...`。
- `strategies/`：策略專屬參數契約、搜尋空間與未來多策略擴充插槽；目前 breakout 已先落地，後續 ML 等策略應比照同層擴充。
- `requirements/`：環境相依與版本鎖定，不放商業邏輯。

## 依賴方向

- `core/` 不得依賴 `tools/`、`apps/` 或純顯示用途程式。
- `apps/` 與 `tools/` 可依賴 `core/`，但不得在外層重寫核心規則。
- 顯示、輸出、CLI、下載流程不得反向影響核心交易規則。
- 參數驗證、交易規則、統計口徑應集中管理，不得在多處重複實作。

## 命名與 façade 規則

- `apps/` 只保留正式入口檔，檔名以功能語意為主；入口層應優先從對應子系統套件 `__init__.py` 匯入，而不是直接依賴更深子模組。
- `tools/*/__init__.py` 與必要 façade 檔負責提供穩定公開介面；子模組可繼續細拆，但對外匯入路徑應盡量維持穩定。
- 核心與工具模組命名已統一改為職責語意命名，不再使用版本前綴；若未來要處理歷史函式／類別識別字中的版本字樣，須另案評估。
- 當 façade 或套件公開介面調整時，應同步更新 `doc/CMD.md` 與本文件中的檔案樹與說明。

## 維護要求

- 新增、刪除、移動、拆分、合併檔案，或調整模組責任與依賴方向時，必須同步更新本文件與 `doc/CMD.md`。
- 若本文件與實際程式不一致，應優先修正文件，不得放任過期。
- 若測試入口、測試分層、主要測試責任或 test suite 維護方式有變動，必須同步更新 `doc/TEST_SUITE_CHECKLIST.md` 與本文件；若連帶影響操作方式，再同步更新 `doc/CMD.md`。



## Local Regression（reduced only）

新增 `tools/local_regression/`，作為本地 reduced 一鍵測試的 orchestrator，只做薄封裝，不重寫交易規則；並由 `apps/test_suite.py` 提供使用者直接執行的正式入口與主控台進度條 / 簡易結果整理。

```text
tools/local_regression/
├── __init__.py
├── common.py
├── manifest.json
├── run_all.py
├── run_all.bat
├── run_chain_checks.py
├── run_ml_smoke.py
├── run_meta_quality.py（含 `run_all.py` helper path coverage probe）
├── meta_quality_coverage.py
├── meta_quality_targets.py
└── run_quick_gate.py

tools/validate/
└── preflight_env.py
```

### 職責
- `run_quick_gate.py`：靜態檢查、CLI 錯誤路徑、缺參數 / 壞參數 / optimizer export-only 壞 DB fail-fast。
- `run_chain_checks.py`：對 reduced 內實際 discover 到的全部 ticker 執行單股 → PIT → 候選 → 可掛單 → 成交 / miss buy 全鏈路對帳，並額外做 chain payload 與 scanner reduced snapshot 的雙跑 digest 對比，檢查重跑一致性；輸出全量 chain summary / chain details。
- `run_ml_smoke.py`：reduced + 少量 trial 的 optimizer smoke，並以固定 seed 做雙跑一致性檢查，確認 optimizer 結果可重現。
- `synthetic_regression_cases.py`：補齊 scanner worker / `tools/scanner/scan_runner.py` 入口重跑一致性、optimizer raw cache rerun / mutation isolation，以及 `run_all.py` 同 run dir rerun summary / bundle repeatability。
- `run_meta_quality.py`：meta quality 工具；對 synthetic coverage suite 產出 line / branch coverage baseline，正式檢查 `doc/TEST_SUITE_CHECKLIST.md` 主表、未完成摘要、已完成摘要的一致性，並校驗 `tools/local_regression/formal_pipeline.py` / `run_all.py` / `preflight_env.py` / `apps/test_suite.py` 的正式步驟一致性，以及文件是否正確宣告單一入口與 registry 原則；同時讀取同輪 local regression step summary 與 optimizer profile summary 做 reduced performance baseline gating；已納入 `apps/test_suite.py` / `run_all.py` 單一入口。
- `formal_pipeline.py`：local regression 正式組成步驟的單一真理來源；定義 step name、command、summary file 與 dataset requirement，供 `run_all.py` / `apps/test_suite.py` / `preflight_env.py` / `run_meta_quality.py` 共用。
- `run_all.py`：先執行 `tools/validate/preflight_env.py` 等價檢查；通過後才依 `formal_pipeline.py` 一鍵串接 quick gate / consistency / chain checks / ml smoke / meta quality，並輸出 `master_summary.json`、`artifacts_manifest.json`、`to_chatgpt_bundle.zip`；另支援 `--only ...`，讓完整入口失敗後只重跑指定步驟；對外提供 apps 使用的 progress callback。
- `common.py`：manifest、輸出目錄、JSON/CSV、reduced dataset 存在性檢查、bundle 打包。
- `preflight_env.py`：根據 `requirements/requirements.txt` 檢查目前 Python 環境是否已具備必要套件；若由 local regression 指定步驟呼叫，則只檢查該步驟實際需要的套件；`quick_gate` 因含 optimizer export-only 錯誤路徑，仍需 `optuna` 與 `SQLAlchemy`；只檢查、不自動安裝。

### 設計原則
- 固定使用 reduced，避免把 full dataset 變成日常 gate。
- 測試只做 orchestration 與 summary，不複製核心交易規則。
- 結果先在 staging 目錄整理後打包；歷史 bundle 歸檔到 `outputs/local_regression/`，專案根目錄只保留最新一份同名 copy。


### 測試入口收斂
- `apps/test_suite.py` 是日常唯一建議使用的一鍵測試入口，且必須串接所有已實作測試；正式步驟順序由 `tools/local_regression/formal_pipeline.py` 提供；先跑完整 regression，再依失敗摘要決定是否用 `run_all.py --only ...` 展開。
- `tools/validate/cli.py` 保留 validate standalone CLI；一致性驗證不再需要佔用 `apps/` 入口位置。
- `core/test_suite_reporting.py` 集中 `apps/test_suite.py` 結果摘要格式與 step label 單一口徑，供 formal entry、meta quality coverage probe 與 synthetic reporting contract 共用，避免 `tools/` 反向依賴 `apps/`。
- 正式測試入口已收斂為 `apps/test_suite.py`；`apps/` 不再保留舊的 regression / consistency 測試入口。


## Test suite bundle

`python apps/test_suite.py` 預設只保留專案根目錄最新唯一 `to_chatgpt_bundle_<timestamp>_<id>.zip`。

- PASS：bundle 只含 minimum set 摘要檔。
- FAIL：bundle 自動擴充為 debug bundle，納入失敗步驟所需除錯材料。
- 內部 staging 目錄打包完成後自動刪除；不再保留 `runs/`、`latest/` 或散開 json。


### bundle 歸檔規則
- `test_suite` / `local_regression` 先在 staging 目錄整理檔案。
- 打包後，歷史 bundle 保留在 `outputs/local_regression/`。
- 專案根目錄只保留最新一份同名 copy，作為直接上傳用。
- staging 目錄會在打包完成後清除。


## Output 分類原則

- `outputs/` 根目錄只放工具分類資料夾，不再讓工具直接把檔案散落到根目錄。
- standalone 工具輸出固定分類到各自資料夾：`validate_consistency`、`ml_optimizer`、`portfolio_sim`、`vip_scanner`、`debug_trade_log`（trade_analysis legacy output dir）、`smart_downloader`。
- `apps/test_suite.py` 不論 PASS / FAIL 都先在 staging 組裝結果，再打成單一 bundle；歷史 bundle 保留於 `outputs/local_regression/`，根目錄只留最新一份 copy。
- 預設輸出採 minimum set；詳細除錯材料只在 FAIL 或指定 debug 模式時放入 bundle。


## Output retention

- `core/output_retention.py`：集中管理 output retention，提供雙門檻（最近 N 份 + 最多 D 天）清理。
- 由 `tools/local_regression/run_all.py` 在 `apps/test_suite.py` 結束後自動觸發，不新增獨立 cleanup app。


- `tools/validate/synthetic_contract_cases.py` 亦負責 artifact lifecycle contract：bundle、archive、root copy、retention。

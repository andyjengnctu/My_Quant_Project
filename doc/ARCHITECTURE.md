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
│  ├─ validate_consistency.py         # 一致性驗證正式入口
│  └─ vip_scanner.py                  # 掃描器正式入口（薄入口）
├─ core/
│  ├─ __init__.py                     # 套件初始化檔
│  ├─ v16_buy_sort.py                 # 買入候選排序邏輯
│  ├─ v16_config.py                   # dataclass、參數預設、參數驗證、共用設定
│  ├─ v16_core.py                     # 單股策略核心：K棒推進與單股回測總控
│  ├─ v16_price_utils.py              # 跳價/成交價/成本/股數/漲跌停與賣出阻塞判斷單一口徑
│  ├─ v16_signal_utils.py             # 單股技術指標與訊號生成
│  ├─ v16_trade_plans.py              # 單股候選/掛單/延續訊號 façade（保留穩定匯入路徑）
│  ├─ v16_history_filters.py          # 歷史績效候選門檻
│  ├─ v16_entry_plans.py              # 候選規格、盤前掛單規格、成交後部位建立
│  ├─ v16_extended_signals.py         # 延續訊號狀態、延續候選與延續掛單規格
│  ├─ v16_data_utils.py               # OHLCV 清理與共用資料工具
│  ├─ v16_dataset_profiles.py         # 資料集模式/CLI/ENV 解析與路徑切換單一入口
│  ├─ v16_display.py                  # 顯示 façade：統一匯出色彩/表格/scanner header/strategy dashboard
│  ├─ v16_display_common.py           # ANSI 色彩、顯示寬度、表格 row 與通用參數讀取 helper
│  ├─ v16_scanner_display.py          # scanner header 與訓練/濾網/硬門檻輸出
│  ├─ v16_strategy_dashboard.py       # 策略 dashboard、對比表與硬門檻輸出
│  ├─ v16_log_utils.py                # logging 與輸出輔助工具
│  ├─ v16_params_io.py                # 參數讀寫、json 載入/匯出
│  ├─ v16_portfolio_engine.py         # 投組核心 timeline 總控與最終整合
│  ├─ v16_portfolio_candidates.py     # 投組候選池掃描、normal/extended 候選規格與排序
│  ├─ v16_portfolio_fast_data.py      # 投組快取市場資料、mark-to-market 與歷史 PIT 統計索引單一口徑
│  ├─ v16_portfolio_entries.py        # 投組進場流程：盤前買進執行、延續訊號清理
│  ├─ v16_portfolio_exits.py          # 投組出場流程：汰弱換股、持倉結算、期末結算
│  ├─ v16_portfolio_ops.py            # 投組日內操作 façade：統一匯出 entries / exits 介面
│  ├─ v16_portfolio_stats.py          # 投組曲線/年度/年化統計與分數計算單一口徑
│  └─ v16_runtime_utils.py            # 執行期共用工具：ProcessPool 啟動方法、Asia/Taipei 時間工具
├─ doc/
│  ├─ ARCHITECTURE.md                 # 本檔；檔案樹、用途與依賴原則說明
│  ├─ CMD.md                          # 常用指令與操作說明
│  ├─ FINMIND_API_TOKEN.md            # API token 說明
│  ├─ PROJECT_SETTINGS.md             # 專案最高優先規則文件
│  └─ ToDo.md                         # 待辦事項與後續整理筆記
├─ data/
│  ├─ tw_stock_data_vip/              # 專案內保留的資料佔位與名單檔
│  └─ tw_stock_data_vip_reduced/      # 專案內保留的資料佔位
├─ models/
│  ├─ v16_all_best_params (LOG_R2).json  # 特定評分口徑下的最佳參數紀錄
│  ├─ v16_all_best_params (RoMD).json    # 特定評分口徑下的最佳參數紀錄
│  ├─ v16_all_best_params_3.json         # 歷史最佳參數或不同批次最佳化輸出
│  └─ v16_best_params.json               # 目前主要使用的最佳參數檔
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
   ├─ debug/
   │  ├─ __init__.py                  # debug 子套件 façade；統一匯出 debug 公開介面
   │  ├─ backtest.py                  # 交易回放與明細列建構
   │  ├─ reporting.py                 # debug 報表輸出與虧損摘要
   │  └─ trade_log.py                 # 交易除錯入口、資料集解析與對外包裝
   ├─ optimizer/
   │  ├─ __init__.py                  # optimizer 子套件 façade；統一匯出 optimizer 公開介面
   │  ├─ main.py                      # 最佳化主流程、study 控制與歷史最佳還原
   │  ├─ prep.py                      # optimizer 預處理 façade：統一匯出原始資料快取與 trial 輸入準備
   │  ├─ raw_cache.py                 # optimizer 原始資料快取、資料清洗與載入摘要
   │  ├─ trial_inputs.py              # optimizer worker 預處理、平行/回退流程與 trial 輸入整合
   │  ├─ objective.py                 # optimizer objective façade：統一匯出 trial 參數/profile/filter/objective runner
   │  ├─ objective_profiles.py        # trial 參數抽樣與初始 profile row 建構
   │  ├─ objective_filters.py         # optimizer filter rules
   │  ├─ objective_runner.py          # trial 級評分流程、portfolio timeline 執行與 user_attrs 寫回
   │  ├─ callbacks.py                 # monitoring callback、profiling console print 與破紀錄展示
   │  ├─ profile.py                   # optimizer profiling CSV/JSON 輸出與摘要
   │  └─ study_utils.py               # trial / study / 參數還原共用工具
   └─ validate/
      ├─ __init__.py                  # validate 子套件 façade；統一匯出 validate 公開介面
      ├─ check_result_utils.py        # validate 檢查結果記錄、ticker 正規化與共用失敗/跳過判定
      ├─ checks.py                    # validate checks façade：統一匯出結果/統計/scanner 預期 helpers
      ├─ main.py                      # 一致性驗證總控；資料集解析、真實掃描協調與結果彙整
      ├─ real_case_assertions.py      # 真實 ticker 驗證的 cross-check 規則與比對項目
      ├─ real_case_io.py              # 真實 ticker 驗證的 CSV 路徑解析與資料清洗載入
      ├─ real_case_runners.py         # 真實 ticker 驗證執行、單股/單檔投組檢查與掃描協調
      ├─ real_cases.py                # 真實 ticker 驗證 façade：統一匯出 io / runners 介面
      ├─ module_loader.py            # validate 模組動態載入、可恢復例外與模組快取
      ├─ portfolio_payloads.py        # validate 投組 payload、年度欄位與 completed trade 摘要 helper
      ├─ reporting.py                 # validate 報表輸出與 console summary
      ├─ synthetic_cases.py           # synthetic suite 入口與 validator 編排
      ├─ synthetic_fixtures.py        # synthetic 測試資料與案例生成
      ├─ synthetic_param_cases.py     # synthetic 參數 guardrail / 排序與歷史門檻案例
      ├─ synthetic_flow_cases.py       # synthetic 延續候選/競爭候選/同日賣出封鎖/rotation 案例
      ├─ synthetic_portfolio_cases.py # synthetic 投組案例 façade：統一匯出各 case validator
      ├─ synthetic_take_profit_cases.py # synthetic 半倉停利/不可執行半倉停利案例
      ├─ tool_adapters.py             # validate 對 apps/debug 工具的動態載入 façade
      ├─ tool_check_common.py         # smoke check 共用輸出抑制與日期欄位解析
      ├─ portfolio_tool_checks.py     # portfolio_sim smoke checks
      ├─ external_tool_checks.py      # scanner/downloader/debug smoke checks
      └─ trade_rebuild.py             # trade log / completed trade 重建工具
```

## 分層原則

- `apps/`：正式執行入口，只負責 CLI、流程組裝與執行期 bootstrap，不得在入口層重寫核心交易規則；`apps/ml_optimizer.py` 現為薄入口，最佳化主流程已拆成 `tools/optimizer/main.py`（CLI/啟動）、`session.py`（session 狀態 façade）、`prep.py` / `raw_cache.py` / `trial_inputs.py`（原始資料快取、worker 預處理與 trial 輸入整合）、`objective.py` / `objective_profiles.py` / `objective_filters.py` / `objective_runner.py`（trial 參數 / 初始 profile / filter rules / objective runner）、`callbacks.py`（monitoring / 破紀錄展示）與 `runtime.py`（記憶庫流程 / 歷史最佳還原 / 匯出控制）；`apps/smart_downloader.py` 現為薄入口，下載流程已拆成 `tools/downloader/main.py`（總控）、`runtime.py`（共用設定 / lazy loader / issue log）、`universe.py`（市場日期與海選）與 `sync.py`（VIP 資料下載與更新跳過）；`apps/portfolio_sim.py` 現為薄入口，模擬流程已拆成 `tools/portfolio_sim/main.py`（CLI/互動流程）、`runtime.py` façade、`runtime_common.py`（共用路徑 / 參數載入 / runtime 目錄 / 不足資料判定）、`simulation_runner.py`（預載入與 timeline 執行）與 `reporting.py`（年度報酬 / Excel / Plotly 輸出）；`apps/vip_scanner.py` 現為薄入口，掃描流程已拆成 `tools/scanner/main.py` façade、`scan_runner.py`（CLI/平行掃描）、`worker.py` façade、`runtime_common.py`（共用路徑 / runtime 目錄 / 參數載入 / worker 數判定）、`stock_processor.py`（單股掃描 worker）與 `reporting.py`（啟動/摘要/候選清單輸出）。
- `tools/validate/`：一致性驗證子系統，已拆成 `check_result_utils.py`（檢查結果記錄 / ticker 正規化 / 可恢復錯誤判定）、`portfolio_payloads.py`（投組 payload / 年度欄位 / completed trade 摘要）、`scanner_expectations.py`（scanner 預期 payload / reference check）、`module_loader.py`（模組動態載入與快取）、`tool_check_common.py`（smoke check 共用輸出抑制與日期欄位解析）、`portfolio_tool_checks.py`（portfolio_sim smoke checks）、`external_tool_checks.py`（scanner/downloader/debug smoke checks）、`tool_checks.py`（smoke check façade）、`checks.py` / `tool_adapters.py` façade、`synthetic_cases.py`、`synthetic_portfolio_common.py`、`synthetic_take_profit_cases.py`、`synthetic_flow_cases.py`、`synthetic_portfolio_cases.py` façade、`synthetic_history_cases.py`、`synthetic_guardrail_cases.py`、`synthetic_param_cases.py` façade、`synthetic_frame_utils.py`、`synthetic_case_builders.py`、`synthetic_fixtures.py` façade、`trade_rebuild.py`、`reporting.py`、`real_case_assertions.py`、`real_case_io.py`、`real_case_runners.py`、`real_cases.py` façade；`main.py` 僅保留資料集解析、真實掃描協調、synthetic suite 觸發與結果彙整，真實 ticker 驗證已再拆成 `real_case_io.py`（CSV 路徑解析 / 資料清洗）與 `real_case_runners.py`（單股 / 單檔投組 / 真實掃描協調），cross-check 規則集中到 `real_case_assertions.py`；synthetic 投組案例已再拆成 `synthetic_take_profit_cases.py`（半倉停利相關）與 `synthetic_flow_cases.py`（延續候選 / 競爭候選 / 同日賣出封鎖 / rotation），`synthetic_portfolio_cases.py` 僅保留 façade。
- `tools/debug/`：交易除錯子系統；`trade_log.py` 保留 CLI 與資料集解析，`backtest.py` 只保留 debug 回測主控 façade，買進 / 延續候選進場已拆到 `entry_flow.py`，出場 / 錯失賣出 / 期末結算已拆到 `exit_flow.py`，明細列建構與半倉停利價 helper 集中到 `log_rows.py`，`reporting.py` 專責 Excel 匯出與虧損摘要。
- `core/`：核心規則與共用計算，應作為單一真理來源；目前 `v16_portfolio_engine.py` 已只保留 `run_portfolio_timeline()` 總控與最終整合，當日候選池掃描 / normal / extended 候選規格與排序已抽至 `v16_portfolio_candidates.py`，快取市場資料/PIT 統計索引抽至 `v16_portfolio_fast_data.py`，日內操作 façade 保留於 `v16_portfolio_ops.py`，其中盤前買進執行/延續訊號清理抽至 `v16_portfolio_entries.py`，汰弱換股/持倉結算/期末結算抽至 `v16_portfolio_exits.py`，曲線/年度/年化統計與分數計算抽至 `v16_portfolio_stats.py`。`v16_core.py` 已縮為單股 K 棒推進與回測總控；單棒持倉步進抽至 `v16_position_step.py`，回測收尾與統計彙整抽至 `v16_backtest_finalize.py`；跳價/成本/股數/漲跌停口徑抽至 `v16_price_utils.py`，技術指標與訊號生成抽至 `v16_signal_utils.py`，`v16_trade_plans.py` 已縮為 façade，歷史績效候選門檻抽至 `v16_history_filters.py`，候選/掛單/成交後部位建立抽至 `v16_entry_plans.py`，延續訊號狀態、延續候選與延續掛單規格抽至 `v16_extended_signals.py`。`v16_display.py` 已縮為 façade，ANSI 色彩/表格與共用參數 helper 抽至 `v16_display_common.py`，scanner header 輸出抽至 `v16_scanner_display.py`，strategy dashboard 與對比表輸出抽至 `v16_strategy_dashboard.py`。
- `tools/`：除錯、驗證與開發輔助工具；可呼叫核心邏輯，但不得成為正式交易規則唯一來源。
- `doc/`：文件與規則說明，以 `PROJECT_SETTINGS.md` 為最高優先規則文件。
- `models/`：參數結果與最佳化輸出，不放正式交易邏輯。
- 執行期資料集預設優先使用 `/data/`；若執行環境不存在 `/data/`，則自動退回 `project/data/`。因此 Linux 可直接使用 `/data/...`，Windows 等無 `/data` 的環境可直接使用專案根目錄下的 `data/...`。
- `requirements/`：環境相依與版本鎖定，不放商業邏輯。

## 依賴方向

- `core/` 不得依賴 `tools/`、`apps/` 或純顯示用途程式。
- `apps/` 與 `tools/` 可依賴 `core/`，但不得在外層重寫核心規則。
- 顯示、輸出、CLI、下載流程不得反向影響核心交易規則。
- 參數驗證、交易規則、統計口徑應集中管理，不得在多處重複實作。

## 命名與 façade 規則

- `apps/` 只保留正式入口檔，檔名以功能語意為主；入口層應優先從對應子系統套件 `__init__.py` 匯入，而不是直接依賴更深子模組。
- `tools/*/__init__.py` 與必要 façade 檔負責提供穩定公開介面；子模組可繼續細拆，但對外匯入路徑應盡量維持穩定。
- `core/` 內既有 `v16_*` 命名目前視為歷史核心相容名稱；新拆出的非必要模組應優先用職責語意命名，不再新增新的版本前綴。
- 當 façade 或套件公開介面調整時，應同步更新 `doc/CMD.md` 與本文件中的檔案樹與說明。

## 維護要求

- 新增、刪除、移動、拆分、合併檔案，或調整模組責任與依賴方向時，必須同步更新本文件與 `doc/CMD.md`。
- 若本文件與實際程式不一致，應優先修正文件，不得放任過期。

- `core/v16_position_step.py`: 單股持倉 K 棒步進與出場事件處理。
- `core/v16_backtest_finalize.py`: 單股回測期末結算與統計彙整。

# 打包
$branch="test-branch-1"; $ts=Get-Date -Format "yyyyMMdd_HHmmss"; $sha=(git rev-parse --short $branch).Trim(); git archive --format=zip -o "${branch}_${ts}_${sha}.zip" $branch

# 重現環境
python -m pip install -r requirements/requirements-lock.txt

# 資料集切換
python apps/validate_consistency.py --dataset reduced
python apps/validate_consistency.py --dataset full
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

# optimizer 架構
python apps/ml_optimizer.py --dataset full            # 正式入口
# apps/ml_optimizer.py 為薄入口，從 tools.optimizer 套件 façade 匯入 main
# tools/optimizer/__init__.py 統一匯出 optimizer 公開介面；main.py 負責 CLI/啟動，session.py 提供 session façade，prep.py / objective.py 為 façade，實作分散於 raw_cache.py / trial_inputs.py / objective_profiles.py / objective_filters.py / objective_runner.py / callbacks.py / runtime.py

# validate 架構
python apps/validate_consistency.py --dataset reduced    # 正式入口
# apps/validate_consistency.py 為薄入口；總控在 tools/validate/main.py
# tools/validate/check_result_utils.py / portfolio_payloads.py / scanner_expectations.py 分別負責檢查結果記錄、投組 payload/年度欄位摘要、scanner 預期 payload/reference check
# tools/validate/module_loader.py / tool_check_common.py / portfolio_tool_checks.py / external_tool_checks.py 分別負責模組動態載入、smoke check 共用工具、portfolio_sim smoke checks、scanner/downloader/debug smoke checks；checks.py / tool_adapters.py / tool_checks.py 僅保留 façade
# tools/validate/real_case_io.py / real_case_runners.py / real_case_assertions.py 分別負責真實 ticker 的 CSV/清洗、執行/掃描協調、cross-check 規則；real_cases.py 僅保留 façade
# synthetic_cases.py 負責 suite 入口；synthetic_portfolio_common.py / synthetic_take_profit_cases.py / synthetic_flow_cases.py / synthetic_portfolio_cases.py 分別負責 synthetic 投組共用 helper、半倉停利案例、流程/rotation 案例與 façade；synthetic_history_cases.py / synthetic_guardrail_cases.py / synthetic_param_cases.py 分別負責歷史門檻案例、guardrail 案例與 façade


## 資料下載

```bash
python apps/smart_downloader.py
```

下載正式入口為 `apps/smart_downloader.py`；入口層從 `tools.downloader` 套件 façade 匯入 `main` 與 `smart_download_vip_data`。`tools/downloader/main.py` 只負責總流程協調，`runtime.py` 管理共用設定 / lazy loader / issue log，`universe.py` 負責市場日期與海選，`sync.py` 負責 VIP 資料下載與最新日期跳過。


## 交易除錯

```bash
python tools/debug/trade_log.py
```

`tools/debug/trade_log.py` 為 debug 正式入口；`tools/debug/backtest.py` 只保留主控 façade，進場流程在 `tools/debug/entry_flow.py`，出場 / 錯失賣出 / 期末結算在 `tools/debug/exit_flow.py`，明細列建構 helper 在 `tools/debug/log_rows.py`，輸出摘要在 `tools/debug/reporting.py`。


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
# core/price_utils.py：跳價/成交價/股數/成本/漲跌停與賣出阻塞判斷
# core/signal_utils.py：技術指標與訊號生成
# core/trade_plans.py：候選/掛單/延續訊號 façade
# core/history_filters.py：歷史績效候選門檻
# core/entry_plans.py：候選規格、盤前掛單規格、成交後部位建立
# core/extended_signals.py：延續訊號狀態、延續候選與延續掛單規格

- validate 子模組已再拆分：`synthetic_history_cases.py`、`synthetic_guardrail_cases.py`、`synthetic_frame_utils.py`、`synthetic_case_builders.py`。


# apps 入口層
# apps/portfolio_sim.py 為薄入口，從 tools.portfolio_sim 套件 façade 匯入公開介面；main.py 負責 CLI/互動流程，runtime.py 為 façade，runtime_common.py 負責共用路徑/參數載入/runtime 目錄/不足資料判定，simulation_runner.py 負責預載入與 timeline 執行，reporting.py 負責年度報酬 / Excel / Plotly 輸出
# apps/vip_scanner.py 為薄入口，從 tools.scanner 套件 façade 匯入公開介面；main.py 為 façade，scan_runner.py 負責 CLI/平行掃描，worker.py 為 façade，runtime_common.py 負責共用路徑/runtime 目錄/參數載入/worker 數判定，stock_processor.py 負責單股掃描 worker，reporting.py 負責啟動/摘要/候選清單輸出

# display 架構
# core/display.py 為 façade；display_common.py 負責 ANSI 色彩/表格與共用 helper，scanner_display.py 負責 scanner header，strategy_dashboard.py 負責策略 dashboard 與對比表


- 單股核心已拆分為 `core/position_step.py` 與 `core/backtest_finalize.py`，對外仍由 `core/backtest_core.py` 提供 façade。

# 命名 / 結構收尾原則
# apps/* 僅從對應子系統套件 façade 匯入公開介面，避免入口層直接依賴更深子模組
# tools/*/__init__.py 與 façade 檔維持穩定公開介面；子模組可繼續細拆，但外部匯入路徑應盡量不變
# 模組檔名已完成版本前綴移除；既有函式／類別識別字若仍含版本字樣，屬另案處理範圍


## 本地一鍵回歸（reduced）

本地最小必要回歸固定只用 reduced。

### 一鍵執行

```bash
python tools/local_regression/run_all.py
```

或直接使用 `apps/` 使用者入口（會顯示進度條與簡易結果整理）：

```bash
python apps/local_regression.py
```

Windows：

```bat
tools\local_regression\run_all.bat
```

### 輸出位置

執行完成後，請查看：

```text
outputs/local_regression/latest/
```

主要檔案：
- `master_summary.json`
- `quick_gate_summary.json`
- `chain_summary.csv`
- `chain_summary.json`
- `ml_smoke_summary.json`
- `to_chatgpt_bundle.zip`

另外，執行完成後也會在專案根目錄額外保留一份最新 bundle：
- `to_chatgpt_bundle_<timestamp>_<uniqueid>.zip`

`apps/local_regression.py` 結束時也會在主控台印出整體 PASS/FAIL、三個步驟摘要、以及最新 bundle 位置。

根目錄舊的 `to_chatgpt_bundle*.zip` 會自動刪除，只保留最新一份，避免還要一層層進 `outputs/` 尋找。

### reduced 資料集

若專案根目錄尚未有 `data/tw_stock_data_vip_reduced`，local regression 會優先嘗試：
1. `<repo>/data.zip`
2. 環境變數 `V16_PROJECT_DATA_ZIP`
3. `/mnt/data/data.zip`

自動解壓其中的 `tw_stock_data_vip_reduced` 到 `<repo>/data/`。

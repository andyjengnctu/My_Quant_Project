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
# apps/ml_optimizer.py 為薄入口；tools/optimizer/main.py 負責 CLI/啟動，session.py 負責 session 狀態 façade，prep.py / raw_cache.py / trial_inputs.py 負責原始資料快取、worker 預處理與 trial 輸入整合，objective.py / objective_profiles.py / objective_filters.py / objective_runner.py 負責 trial 參數、初始 profile、filter rules 與 objective runner，callbacks.py 負責 monitoring / 展示，runtime.py 負責記憶庫流程/匯出

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

下載正式入口為 `apps/smart_downloader.py`；`tools/downloader/main.py` 只負責總流程協調，`runtime.py` 管理共用設定 / lazy loader / issue log，`universe.py` 負責市場日期與海選，`sync.py` 負責 VIP 資料下載與最新日期跳過。


## 交易除錯

```bash
python tools/debug/trade_log.py
```

`tools/debug/trade_log.py` 為 debug 正式入口；交易回放主邏輯在 `tools/debug/backtest.py`，輸出摘要在 `tools/debug/reporting.py`。


# portfolio engine 架構
# run_portfolio_timeline() 正式總控、候選池掃描與最終整合仍在 core/v16_portfolio_engine.py
# 快取市場資料/PIT 索引在 core/v16_portfolio_fast_data.py
# 日內操作 façade 在 core/v16_portfolio_ops.py
# 盤前買進執行/延續訊號清理在 core/v16_portfolio_entries.py
# 汰弱換股/持倉結算/期末結算在 core/v16_portfolio_exits.py
# 曲線/年度/年化統計與分數在 core/v16_portfolio_stats.py


# single-stock core 架構
# core/v16_core.py：單股 K 棒推進與回測總控
# core/v16_price_utils.py：跳價/成交價/股數/成本/漲跌停與賣出阻塞判斷
# core/v16_signal_utils.py：技術指標與訊號生成
# core/v16_trade_plans.py：候選規格、盤前掛單規格、延續訊號狀態與進場成交判定

- validate 子模組已再拆分：`synthetic_history_cases.py`、`synthetic_guardrail_cases.py`、`synthetic_frame_utils.py`、`synthetic_case_builders.py`。

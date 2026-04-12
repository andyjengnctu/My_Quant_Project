# 重現環境

```bash
python requirements/export_requirements_lock.py
```

# 資料集切換

```bash
python apps/test_suite.py
python tools/local_regression/run_all.py --only quick_gate
python tools/validate/preflight_env.py
```

- 日常 reduced 一鍵入口：`python apps/test_suite.py`
- 只有完整入口已指出失敗步驟時，才用 `python tools/local_regression/run_all.py --only ...` 重跑指定步驟。
- `tools/local_regression/formal_pipeline.py` 為正式步驟單一真理來源。
- `python tools/validate/preflight_env.py` 只檢查環境，不自動安裝依賴。

# 打包

```bash
python apps/package_zip.py
python apps/package_zip.py --run-test-suite
```

- `--run-test-suite` 會在打包後執行 `python apps/test_suite.py`。
- `package_zip` 只會把 root 的非 bundle ZIP 移入 archive；`to_chatgpt_bundle_*.zip` 由 test suite 維護最新 root copy。

# 主工具入口

```bash
python apps/ml_optimizer.py
python apps/portfolio_sim.py
python apps/smart_downloader.py
python apps/vip_scanner.py
python apps/workbench.py
```

- `apps/` 只作正式入口；模組責任與依賴方向以 `doc/ARCHITECTURE.md` 為準。

# Workbench

- `apps/workbench.py` 為 GUI 正式入口。
- K 線檢視中，交易明細與 Console 為獨立分頁。
- 日常 GUI 問題先檢查 `tools/workbench_ui/single_stock_inspector.py`，再看 `tools/workbench_ui/workbench.py`。

# Trade analysis

- `tools/trade_analysis/trade_log.py` 為單股 trade-analysis 正式入口。
- 為維持相容性，保留 legacy `run_debug_*` API 名稱。
- 對外建議使用 canonical `run_trade_analysis` / `run_trade_backtest` / `run_prepared_trade_backtest` / `run_ticker_analysis` aliases。

# 輸出分類

- `outputs/local_regression/`：test suite 歷史 bundle。
- `outputs/validate_consistency/`：standalone consistency 報表。
- `outputs/ml_optimizer/`：optimizer profiling / 載入摘要。
- `outputs/portfolio_sim/`：投組報表與載入摘要。
- `outputs/vip_scanner/`：scanner issue log。
- `outputs/smart_downloader/`：下載器 issue log。
- `outputs/debug_trade_log/`：`trade_analysis` 單股分析輸出；為維持既有工具鏈相容，暫沿用 legacy 目錄名 `debug_trade_log`。

# Output retention

- `outputs/local_regression/`：保留最近 20 份，刪除超過 30 天。
- `outputs/validate_consistency/`、`outputs/portfolio_sim/`：保留最近 10 份，刪除超過 30 天。
- `outputs/ml_optimizer/`、`outputs/vip_scanner/`、`outputs/smart_downloader/`、`debug_trade_log`（trade_analysis legacy output dir）：保留最近 5 份，刪除超過 14 天。

# 文件分工

- `PROJECT_SETTINGS.md`：上層原則、模組責任、邊界與資料流。
- `TEST_SUITE_CHECKLIST.md`：本地 formal test suite 主表、狀態與收斂索引。
- `GPT_DELIVERY_CHECKLIST.md`：GPT 交付前操作檢查；不納入本地 formal 驗證。

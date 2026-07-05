# 常用指令

python apps/ml_optimizer.py --dataset full --timing --trials 10 `效能驗證`
python apps\ml_optimizer.py --dataset full --outer-oos --timing --trials 10 --outer-first-oos-date 2021-01-01 --outer-last-oos-date 2026-01-01 --outer-window-mode fixed --outer-train-window-months 60 --outer-oos-months 12 --yes `rolling效能驗證`

## 環境 / 測試

```bash
python requirements/export_requirements_lock.py
python apps/test_suite.py
python tools/local_regression/run_all.py --only quick_gate
python tools/validate/preflight_env.py
```

- 日常一鍵入口：`python apps/test_suite.py`
- 正式對外入口為 `apps/test_suite.py`。
- 只有正式入口已指出失敗步驟時，才用 `python tools/local_regression/run_all.py --only ...` 重跑指定步驟。
- `python tools/validate/preflight_env.py` 只檢查環境，不自動安裝依賴。

## 打包

```bash
python apps/package_zip.py
python apps/package_zip.py --run-test-suite
python apps/package_zip.py --commit-message "chore: package before delivery" --run-test-suite
```

## 主工具入口

```bash
python apps/ml_optimizer.py
python apps/portfolio_sim.py
python apps/smart_downloader.py
python apps/vip_scanner.py
python apps/workbench.py
```

- `apps/` 只作正式入口；模組責任與依賴方向以 `doc/ARCHITECTURE.md` 為準。

# Workbench

- `apps/workbench.py` 為 GUI 正式入口，也是單股 trade-analysis 的單一使用者入口。
- Workbench 上方控制列提供股票代號輸入、常用股票下拉、候選股掃描與歷史績效股掃描。
- K 線檢視中，交易明細與 Console 為獨立分頁。
- 日常 GUI 問題先檢查 `tools/workbench_ui/single_stock_inspector.py`，再看 `tools/workbench_ui/workbench.py`。

# Trade analysis

- `tools/trade_analysis/trade_log.py` 提供單股 trade-analysis 共用 backend / 開發輔助 CLI；正式使用者入口仍為 `apps/workbench.py`。
- 為維持相容性，保留 legacy `run_debug_*` API 名稱。
- 對外建議使用 canonical `run_trade_analysis` / `run_trade_backtest` / `run_prepared_trade_backtest` / `run_ticker_analysis` aliases。


# Breakout quality filter

## 研究資料、訓練與評估

```bash
python tools/filters/breakout_quality/build_dataset.py --dataset full --filter-id breakout_quality_v1
python tools/filters/breakout_quality/train.py --filter-id breakout_quality_v1 --inner-validation-ratio 0.20
python tools/filters/breakout_quality/export_scores.py --filter-id breakout_quality_v1 --scope research
# threshold 只能用 Selection 內的 inner validation 選擇
python tools/filters/breakout_quality/evaluate.py --filter-id breakout_quality_v1 --split validation --threshold 0.50
# threshold 鎖定後，外層 OOS 只做最終評估
python tools/filters/breakout_quality/evaluate.py --filter-id breakout_quality_v1 --split oos --threshold 0.50
```

- Dataset/tool 輸出固定在 `outputs/filters/breakout_quality/<filter_id>/`。
- Model、manifest 與 `split_assignments.csv` 固定在 `models/filters/breakout_quality/<filter_id>/`。
- 外層正式期間只有 `selection / oos`，日期直接讀 `core.walk_forward_policy`；Selection 內才依 `BREAKOUT_QUALITY_INNER_VALIDATION_RATIO` 切 `train / validation / embargo`。
- Rolling OOS fold 必須沿用既有 `V16_WF_SELECTION_START_DATE`、`V16_WF_SEARCH_TRAIN_END_DATE`、`V16_WF_OOS_START_DATE`、`V16_WF_OOS_END_DATE` policy override；不得另傳一套 breakout-quality 專用日期。
- `research` 分數固定寫到 `outputs/filters/breakout_quality/<filter_id>/research_scores.csv`，不會改動正式 `scores.csv` 或 model manifest。
- 正式 `models/.../scores.csv` 只能由明確的 `--scope forward_oos` 建立。

## 建立正式 forward-OOS score table

1. 先以完整研究資料執行 `build_dataset.py` 與 `train.py`；`train.py` 依既有 walk-forward policy 只用 outer Selection 的 inner train/validation，保留 `model.pt`、`split_assignments.csv`、`manifest.json` 與 `model_information_cutoff`。
2. 若目前 dataset 已包含 outer OOS，可直接匯出；若需延伸到更新資料，只重新執行 `build_dataset.py`，不可重新 train 同一模型。
3. 執行：

```bash
python tools/filters/breakout_quality/export_scores.py --filter-id breakout_quality_v1 --scope forward_oos
```

- `forward_oos` 只匯出落在同一份 walk-forward OOS window、且事件日嚴格晚於 `model_information_cutoff` 的分數；若沒有符合資料會直接失敗。
- `available_from` 之前視為模型尚未啟用；`available_through` 之後若出現候選事件則 fail-fast，沒有候選事件時不要求不存在的分數。
- 正式 runtime 只讀 `models/filters/breakout_quality/<filter_id>/scores.csv`，並驗證 model/score SHA256、schema、high_len coverage、OOS eligibility 與可用日期。
- 現有舊版 `breakout_quality_v1` manifest 不符合新版 artifact contract 時，必須依上述流程重建，不得由 runtime 猜測或自動相容。

# 輸出分類

- `outputs/local_regression/`：test suite 歷史 bundle。
- `outputs/local_regression/_staging/`：formal / validate 暫存 staging；屬 `local_regression` 內部子目錄，會由 retention 自動清理。
- `outputs/validate_consistency/`：standalone consistency 報表。
- `outputs/ml_optimizer/`：optimizer profiling / 載入摘要。
- `outputs/portfolio_sim/`：投組報表與載入摘要。
- `outputs/vip_scanner/`：scanner issue log。
- `outputs/smart_downloader/`：下載器 issue log。
- `outputs/debug_trade_log/`：`trade_analysis` 單股分析輸出；為維持既有工具鏈相容，暫沿用 legacy 目錄名 `debug_trade_log`。
- `outputs/debug_trade_log/`（trade_analysis legacy output dir）屬既有工具鏈相容邊界。
- `outputs/workbench_ui/`：Workbench GUI runtime 快取；目前用於常用股票中文名稱快取；若 reduced 代碼組變動或缺名，Workbench 會優先查官方 CSV / ISIN 名錄並於必要時做 SSL 容錯與 HTTP fallback。

## 其他文件

- `ARCHITECTURE.md`：分層、正式入口、依賴方向與共享邊界。
- `TEST_SUITE_CHECKLIST.md`：formal test suite 主表、狀態與收斂索引。

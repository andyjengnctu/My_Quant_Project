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

正式操作統一由 `apps/breakout_quality.py` 進入；`tools/filters/breakout_quality/` 的直接 CLI 僅保留開發與相容用途。

互動式 PowerShell／Terminal 直接執行下列指令會開啟選單；選單提供完整研究流程、單步操作、正式 forward-OOS 匯出與工件狀態檢查。

```bash
python apps/breakout_quality.py
```

完整研究流程會依序執行：必要時建立 dataset → train → export research scores → 產生易讀研究報表；可選擇是否把 OOS 納入報表。報表開頭將 Filter ID、統計口徑、Selection/OOS 日期與固定訓練參數合併顯示；第 1 區呈現 Epoch 選擇與完整重訓結果，第 2 區比較各資料區段。資料區段與日期分欄，最終部署判定只保留 OOS 依據、部署決策與不可回調同一 OOS 的限制。未納入 OOS 時，報表會明確說明因此不輸出 OOS Confusion Matrix 與 Selection/OOS 差異。終端顯示表格化比較、條列式 Epoch／部署判定與整合比例的 Confusion Matrix；互動式終端會以淡藍、綠、黃、紅標示重點，重新導向或測試輸出不插入 ANSI 色碼。Markdown 以相同語意顏色呈現，完整 metrics JSON 會寫入 `outputs/filters/breakout_quality/<filter_id>/reports/`。批次或需要可重現命令時使用 `workflow`：

```bash
python apps/breakout_quality.py workflow --filter-id breakout_quality_v1 --dataset full --epochs 20 --batch-size 256 --lr 0.001 --seed 42 --fixed-threshold 0.50 --use-inner-validation --inner-validation-months 24
```

- `workflow` 只有在 dataset 工件完整、profile、ticker coverage、feature/label policy、欄位契約與來源 CSV inventory 全部一致時才跳過重建；full/reduced 原始 CSV 的檔案成員、大小或修改時間有變化時會自動重建。單獨執行 `train` 時若偵測到來源已更新，會 fail-fast 並要求先重建，避免靜默使用過期 dataset。需要無條件重建時加 `--rebuild-dataset`。
- 尚未準備最終 OOS 評估時，可加 `--no-evaluate-oos`；報表只包含 Selection 內診斷，並明確標示不能作為正式泛化結論。
- 選單只是正式 UI orchestration；dataset、split、training、export 與 evaluation 規則仍只實作在既有子系統，不在 app 複製。
- `epochs`、`batch size`、`learning rate`、`random seed`、最少 train/validation rows、threshold 與 inner-validation 預設均集中於 `config/breakout_quality_policy.py`；CLI／選單可單次覆蓋，但不保存第二份預設值。

也可逐步執行：

```bash
python apps/breakout_quality.py build-dataset --dataset full --filter-id breakout_quality_v1
# 預設關閉 inner validation：epochs 是完整 Selection 的正式固定訓練次數
python apps/breakout_quality.py train --filter-id breakout_quality_v1 --epochs 20 --lr 0.001 --seed 42 --fixed-threshold 0.50 --no-use-inner-validation
# 開啟時：epochs 是搜尋上限；以 Selection 尾端 N 個月選 best epoch，之後完整 Selection 重訓
python apps/breakout_quality.py train --filter-id breakout_quality_v1 --epochs 20 --lr 0.001 --seed 42 --fixed-threshold 0.50 --use-inner-validation --inner-validation-months 24
python apps/breakout_quality.py export-scores --filter-id breakout_quality_v1 --scope research
# 建議日常使用：終端表格報表 + Markdown 解釋報表 + 完整 metrics JSON
python apps/breakout_quality.py report --filter-id breakout_quality_v1 --no-include-oos
# 參數與模型已鎖定後，才把最終 OOS 納入報表
python apps/breakout_quality.py report --filter-id breakout_quality_v1 --include-oos
# 需要稽核單一 split 的完整原始 JSON 時才使用 evaluate
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --split train
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --split validation
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --split selection
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --split oos
```

- Dataset/tool 輸出固定在 `outputs/filters/breakout_quality/<filter_id>/`；易讀報表固定為 `reports/evaluation_report.md`，完整報表數據固定為 `reports/evaluation_metrics.json`。
- Model、manifest 與 `split_assignments.csv` 固定在 `models/filters/breakout_quality/<filter_id>/`；固定 threshold 也寫入 manifest，OOS 評估不得改用其他值；正式啟用時 active `breakout_quality_score_threshold` 應與該固定值一致。
- 外層正式期間只有 `selection / oos`，日期直接讀 `core.walk_forward_policy`。
- `BREAKOUT_QUALITY_USE_INNER_VALIDATION=False` 時，全部 eligible Selection 固定 epochs 訓練；開啟時，Selection 尾端 `BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS` 個月用來選 epoch，之後重新初始化模型，以全部 eligible Selection（含 validation 與 inner-embargo rows）按選定 epoch 重訓。
- inner validation 只可選 epoch；fixed threshold 仍在 OOS 前鎖定，不可由 validation 或 OOS 自動最佳化。
- Rolling OOS fold 必須沿用既有 `V16_WF_SELECTION_START_DATE`、`V16_WF_SEARCH_TRAIN_END_DATE`、`V16_WF_OOS_START_DATE`、`V16_WF_OOS_END_DATE` policy override；不得另傳一套 breakout-quality 專用日期。
- `research` 分數固定寫到 `outputs/filters/breakout_quality/<filter_id>/research_scores.csv`，不會改動正式 `scores.csv` 或 model manifest。
- 正式 `models/.../scores.csv` 只能由明確的 `--scope forward_oos` 建立。

## 建立正式 forward-OOS score table

1. 先以正式 app 的 `build-dataset` 與 `train` 子命令建立完整研究資料並訓練；`train` 依既有 walk-forward policy 執行固定 epoch 模式，或以 Selection 內 validation 選 epoch 後完整重訓，並保留 `model.pt`、`split_assignments.csv`、`manifest.json` 與 `model_information_cutoff`。
2. 若目前 dataset 已包含 outer OOS，可直接匯出；若需延伸到更新資料，只重新執行 `build-dataset`，不可重新 train 同一模型。
3. 執行：

```bash
python apps/breakout_quality.py export-scores --filter-id breakout_quality_v1 --scope forward_oos
```

- `forward_oos` 只匯出落在同一份 walk-forward OOS window、且事件日嚴格晚於 `model_information_cutoff` 的固定規格模型分數；若沒有符合資料會直接失敗。
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

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

互動式 PowerShell／Terminal 直接執行下列指令會開啟選單；選單提供完整研究流程、單步操作、正式 forward-OOS 匯出與工件狀態檢查。第一層選單確認操作類型後，所有已由 `config/breakout_quality_policy.py` 定義的設定都直接採用 policy，不再重複詢問，包括 Filter ID、model architecture、epochs、training batch size、evaluation batch size、evaluation workers、parallel split evaluation、feature-bank preload、training prefetch、learning rate、weight decay、gradient clipping、final refit mode、class weight mode、time weight mode、random seed、threshold、inner validation、validation 月數與 early stopping。互動式完整研究流程固定使用 Full dataset、全部股票並執行 OOS，不再詢問 dataset 類型、最多股票數或是否執行 OOS；開始確認預設為 Y；互動式單獨建立 dataset 也固定使用 Full dataset 與全部股票。dataset 是否需要建立／重建由 workflow 自動偵測，偵測到過期或不一致時直接重建；只有 dataset 已最新時才詢問是否強制重建，預設 N。需要 reduced、限制股票數或略過 OOS 的開發／單次執行時，改用對應 CLI 參數。

```bash
python apps/breakout_quality.py
```

完整研究流程會依序執行：必要時建立 dataset → train → export research scores → 產生易讀研究報表；報表預設納入 OOS。互動式「產生易讀研究報表」固定讀取最終 OOS 並納入報表，不再詢問；讀取後不得依同一段 OOS 回頭調整 threshold、epochs、learning rate、feature、label 或模型。報表開頭將 Filter ID、統計口徑、Selection/OOS 日期與固定訓練參數合併顯示；後續依序呈現 Epoch 選擇、Selection Confusion Matrix、OOS Confusion Matrix、各資料區段比較與 OOS 綜合判定。OOS 綜合判定合併原本的 Selection/OOS 差異與部署判定，依「主要成效、過度篩選防線、輔助診斷」三類編排，並新增逐項判讀欄。資料區段與日期分欄；第 4 區固定精簡為「原始 PASS、模型 PASS、PASS Precision、Precision 絕對、PASS Recall、平均 Score」，依此順序呈現。REJECT Specificity、REJECT NPV、Accuracy 與 Precision 相對僅保留在 Confusion Matrix 下方或 OOS 綜合判定。Confusion Matrix 中央只保留 TP／FN／FP／TN；右側依序顯示「原始PASS → TP + FN」與「原始REJECT → FP + TN」，底部依序顯示「TP + FP → 模型PASS」與「FN + TN → 模型REJECT」。分類品質另以「指標、公式、結果、解釋」表呈現；Precision 絕對／相對另以「指標、公式、結果」表呈現。Confusion Matrix 前不再重複顯示統計口徑或列／欄說明。終端會以淡藍、綠、黃、紅標示重點；Confusion Matrix 僅以綠色標示 TP／TN、紅色標示 FP／FN，原始／模型類別與合計維持中性色，且每一行獨立重設 ANSI 色碼，避免跨格污染。重新導向或測試輸出不插入 ANSI 色碼。Markdown 以相同語意顏色呈現，完整 metrics JSON 會寫入 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/reports/`。批次或需要可重現命令時使用 `workflow`：

```bash
python apps/breakout_quality.py workflow --filter-id breakout_quality_v1 --dataset full --epochs 20 --batch-size 128 --evaluation-batch-size 4096 --evaluation-workers 4 --no-parallel-split-evaluation --train-prefetch-batches 0 --preload-feature-bank --lr 0.0003 --weight-decay 0.0001 --gradient-clip-norm 1.0 --final-refit-mode selected_epochs --class-weight-mode none --time-weight-mode none --seed 42 --fixed-threshold 0.50 --use-inner-validation --inner-validation-months 24
```

模型架構只由 `config/breakout_quality_policy.py` 的下列設定切換：

```python
BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "multiscale_cnn_v3"
# 或
BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "multiscale_cnn_v2"
# 或
BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "multiscale_cnn_v1"
# 或
BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "tiny_cnn_v1"
# 或
BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "residual_tcn_v1"
```

切換架構後可直接重跑 `workflow`。現有 Dataset、feature bank、future-path cache 與 labels 會沿用，不需重建；模型必須重新訓練。五種架構的 model、manifest、scores、research scores 與 reports 分開存放，因此不會互相覆蓋。`multiscale_cnn_v1`、`multiscale_cnn_v2` 與 `multiscale_cnn_v3` 使用相同 23,522 個參數、三分支容量、GroupNorm、約 244 bars receptive field、20／60／120／300 bars window-average 與 last pooling。v1 三個 branch 全部使用 level；v2 只把短／中期 branch 轉成 previous-close price return／range 與 normalized log-volume delta；v3 再只把短／中期的個股 O/H/L/C 變化改為「個股變化減同欄 0050 變化」，個股量能 delta、0050 五個 delta channels 與長期 level branch 維持 v2 定義。v2／v3 第一個 bar 因視窗外沒有前值而固定為零。`residual_tcn_v1` 使用 6 個 residual dilated blocks、dilation 1/2/4/8/16/32、約 253 bars receptive field，以及 last/average/max pooling；`tiny_cnn_v1` 保留舊兩層 Conv1D baseline。

- `workflow` 會自動分成三種處理：工件、profile、ticker coverage、feature/high_len/benchmark/path-cache、欄位契約或來源 CSV inventory 改變時完整重建；只有 label horizon/PASS/REJECT 改變且 horizon 未超過 future path cache 時執行快速 relabel；全部一致時跳過。 完整重建採用 `ticker/date` feature bank 去重、逐檔 CSV 讀取與 per-ticker chunk 合併，正式陣列可 mmap 載入。互動選單只有在判定不需更新時，才詢問「是否強制完整重建 dataset」，預設 N；選 Y 等同 `--rebuild-dataset`。單獨執行 `train` 時若偵測到來源已更新，會 fail-fast 並要求先重建，避免靜默使用過期 dataset。 Dataset 完整重建的逐股票進度固定在同一行刷新，避免大量輸出洗版；重新導向輸出時只保留最終進度摘要與必要的 skip 訊息。
- 尚未準備最終 OOS 評估時，可加 `--no-evaluate-oos`；報表只包含 Selection 內診斷，並明確標示不能作為正式泛化結論。
- 選單只是正式 UI orchestration；dataset、split、training、export 與 evaluation 規則仍只實作在既有子系統，不在 app 複製。
- `BREAKOUT_QUALITY_MODEL_ARCHITECTURE`、epochs、training batch size、evaluation batch size、evaluation workers、parallel split evaluation、training prefetch、feature-bank preload、`learning rate`、`weight decay`、`gradient clip norm`、final refit mode、class weight mode、time weight mode、`random seed`、最少 train/validation rows、threshold 與 inner-validation 預設均集中於 `config/breakout_quality_policy.py`；互動選單直接採用 policy，不再逐項詢問，CLI 可單次覆蓋且不回寫 policy。Inner Validation 開啟時，final refit 依 policy 選擇：`selected_epochs` 以相同 epoch 數在完整 eligible Selection 重訓；`matched_optimizer_steps` 則把 best epoch 轉成 Inner Train 的 optimizer updates，再於完整 Selection 重訓相同更新量，且最低完整走過 Selection 一次。`class_weight_mode=none` 不對約 55/45 的 Label 額外做 inverse-frequency balancing；`time_weight_mode=none` 是第一階段基準，後續可單獨改為 `year_balanced_sqrt`，以溫和平方根權重降低事件密集年份對 loss 的支配。`evaluation batch size` 與 `evaluation workers` 控制完整 Train／Validation／Selection 及 score export 的 read-only 推論：原 batch boundaries、全部 requested rows、輸出列序與最終 reduction 順序都不變。開啟 `parallel split evaluation` 時，Inner Train 與 Validation 的完整評估同時執行，峰值最多使用 `2 × evaluation workers`，但各自仍使用原本的資料列、batch 與模型快照。訓練仍固定單執行緒；`zero_grad(set_to_none=True)`、policy 指定的 Adam weight decay、gradient clipping、RAM preload 與可選的 batch prefetch 均忠實套用並寫入 manifest；除已明確設定的 regularization 外，不改訓練 rows、shuffle 或 batch 邊界。Final refit 最後一輪已完成的完整指標會直接沿用，避免完全重複推論。Research／forward-OOS score export 也使用固定 batch 的平行 model replicas，最後依原列序寫回相同 scores。GPU 或多執行緒 training 可能改變浮點結果，因此不在 strict-result 預設模式啟用。`train` 終端的 Epoch 選擇每輪顯示完整 Inner Train Loss、Validation Loss、新最佳標記與該 Epoch 總耗時；完整 Selection 重訓沒有獨立 Validation，因此每輪顯示 Train Loss 與耗時。完整研究流程會在每一階段結束後顯示階段耗時，並在最後顯示 workflow 總耗時；互動終端中的時間值使用淡藍色，重新導向或測試輸出不插入 ANSI 色碼。關鍵資料區段與工件路徑保留在終端，完整 history、split policy、overlap、counts 與 Epoch `elapsed_sec` 仍保留於 `manifest.json`，不再將整包 Python dict 印到終端。

也可逐步執行：

```bash
python apps/breakout_quality.py build-dataset --dataset full --filter-id breakout_quality_v1

# 只更新 label；通常由 workflow 自動偵測，不需手動執行
python apps/breakout_quality.py build-dataset --dataset full --filter-id breakout_quality_v1 --relabel-only
# 預設關閉 inner validation：epochs 是完整 Selection 的正式固定訓練次數
python apps/breakout_quality.py train --filter-id breakout_quality_v1 --epochs 20 --lr 0.001 --seed 42 --fixed-threshold 0.50 --no-use-inner-validation
# 開啟時：epochs 是搜尋上限；以 Selection 尾端 N 個月選 best epoch，之後按 matched optimizer steps 完整 Selection 重訓
python apps/breakout_quality.py train --filter-id breakout_quality_v1 --epochs 20 --lr 0.0003 --weight-decay 0.0001 --gradient-clip-norm 1.0 --final-refit-mode matched_optimizer_steps --class-weight-mode none --time-weight-mode none --seed 42 --fixed-threshold 0.50 --use-inner-validation --inner-validation-months 24
python apps/breakout_quality.py export-scores --filter-id breakout_quality_v1 --scope research --inference-batch-size 4096 --inference-workers 4 --preload-feature-bank
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

- Dataset 與 future-path cache 固定在 `outputs/filters/breakout_quality/<filter_id>/`；research scores 與易讀報表依架構放在 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/`，報表固定為 `reports/evaluation_report.md`，完整報表數據固定為 `reports/evaluation_metrics.json`。
- Model、manifest 與 `split_assignments.csv` 依架構固定在 `models/filters/breakout_quality/<filter_id>/<model_architecture>/`；固定 threshold 也寫入 manifest，OOS 評估不得改用其他值；正式啟用時 active `breakout_quality_score_threshold` 應與該固定值一致。
- 外層正式期間只有 `selection / oos`，日期直接讀 `core.walk_forward_policy`。
- `BREAKOUT_QUALITY_USE_INNER_VALIDATION=False` 時，全部 eligible Selection 固定 epochs 訓練；開啟時，Selection 尾端 `BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS` 個月用來選 epoch，之後重新初始化模型，以全部 eligible Selection（含 validation 與 inner-embargo rows）重訓。預設 `BREAKOUT_QUALITY_FINAL_REFIT_MODE="matched_optimizer_steps"` 會匹配 Inner Train best epoch 的 optimizer updates，並保證完整 Selection 至少遍歷一次；舊式相同 epoch 數只保留為 `selected_epochs` 對照模式。
- inner validation 只可選 epoch；fixed threshold 仍在 OOS 前鎖定，不可由 validation 或 OOS 自動最佳化。
- Rolling OOS fold 必須沿用既有 `V16_WF_SELECTION_START_DATE`、`V16_WF_SEARCH_TRAIN_END_DATE`、`V16_WF_OOS_START_DATE`、`V16_WF_OOS_END_DATE` policy override；不得另傳一套 breakout-quality 專用日期。
- `research` 分數固定寫到 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/research_scores.csv`，不會改動正式 `scores.csv` 或 model manifest。
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
- 正式 runtime 只讀目前 policy 架構的 `models/filters/breakout_quality/<filter_id>/<model_architecture>/scores.csv`，並驗證 model/score SHA256、schema、high_len coverage、OOS eligibility 與可用日期。
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

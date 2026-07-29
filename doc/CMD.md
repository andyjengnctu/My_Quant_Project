# 常用指令

python apps/ml_optimizer.py --dataset full --timing --trials 10 `效能驗證`
python apps\ml_optimizer.py --dataset full --outer-oos --timing --trials 10 --outer-first-oos-date 2021-01-01 --outer-last-oos-date 2026-01-01 --outer-window-mode fixed --outer-train-window-months 120 --outer-oos-months 12 --yes `rolling效能驗證`

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

互動式 PowerShell／Terminal 直接執行下列指令會開啟選單；選單提供完整研究流程、單步操作、正式 forward-OOS 匯出與工件狀態檢查。第一層選單確認操作類型後，所有已由 `config/breakout_quality_policy.py` 定義的設定都直接採用 policy，不再重複詢問，包括 Filter ID、model architecture、experiment profile、epochs、training batch size、evaluation batch size、evaluation workers、parallel split evaluation、feature-bank preload、training prefetch、Torch device、mixed precision dtype、determinism、TF32、learning rate、weight decay、gradient clipping、final refit mode、class weight mode、time weight mode、random seed、threshold、inner validation、validation 月數與 early stopping。互動式完整研究流程固定使用 Full dataset、全部股票並執行 OOS，不再詢問 dataset 類型、最多股票數或是否執行 OOS；開始確認預設為 Y；互動式單獨建立 dataset 也固定使用 Full dataset 與全部股票。dataset 是否需要建立／重建由 workflow 自動偵測，偵測到過期或不一致時直接重建；只有 dataset 已最新時才詢問是否強制重建，預設 N。需要 reduced、限制股票數或略過 OOS 的開發／單次執行時，改用對應 CLI 參數。

```bash
python apps/breakout_quality.py
```

只有需要重建9D MantisV2 legacy工件時才需安裝固定相依套件。官方 `mantis-tsfm==1.0.0` 宣告 `pandas<3.0`，而本專案鎖定 pandas 3.x，因此必須先安裝相容依賴，再以 `--no-deps` 安裝 Mantis，避免 pip 降級既有資料鏈：

```bash
python -m pip install -r requirements/requirements-mantis-v2.txt
python -m pip install --no-deps mantis-tsfm==1.0.0
python -c "from importlib.metadata import version; from mantis.architecture import MantisV2; print('mantis-tsfm', version('mantis-tsfm'))"
```


只有需要重建9E MOMENT-1-base legacy工件時才需安裝官方選配套件。`momentfm==0.1.4` 的 PyPI metadata 釘死舊版 NumPy、Hugging Face Hub 與 Transformers；本專案不得因此降級既有資料鏈，所以先安裝專案驗證過的 Transformers runtime，再以 `--no-deps` 安裝 MOMENT。使用者目前 Python 3.14 環境仍須由本地正式測試確認相容性：

```bash
python -m pip install --index-url https://pypi.org/simple -r requirements/requirements-moment.txt
python -m pip install --index-url https://pypi.org/simple --no-deps momentfm==0.1.4
python -c "from importlib.metadata import version; from momentfm import MOMENTPipeline; print('momentfm', version('momentfm')); print('transformers', version('transformers')); print('MOMENTPipeline import: OK')"
```

9E legacy runtime 契約固定為 `momentfm==0.1.4` 與 `transformers==5.5.0`。不要直接執行 `pip install momentfm==0.1.4`，否則 pip 可能嘗試把本專案的 NumPy／Hub／Transformers 降到該套件 metadata 所列的舊版本。由於採刻意隔離安裝，`pip check` 仍會依舊 metadata 報告版本不相容，不能用它取代上方版本檢查與正式 `apps/test_suite.py`。

完整研究流程會依active architecture分流：必要時建立 supervised dataset；目前policy已退回9A InceptionTime，依序執行train → export research scores → 產生易讀研究報表；9A維持已接受排序／高品質基準，8F sequence-only保留高覆蓋基準。9C TS2Vec、9D MantisV2、9E MOMENT與9F Patch Transformer已轉為legacy read-only：Selection-only pretraining chain與外部checkpoint下載／驗證只供歷史工件重建，不再由正式新實驗workflow啟動；報表預設納入 OOS。互動式「產生易讀研究報表」固定讀取最終 OOS 並納入報表，不再詢問；讀取後不得依同一段 OOS 回頭調整 threshold、epochs、learning rate、feature、label 或模型。報表開頭將 Filter ID、統計口徑、Selection/OOS 日期與固定訓練參數合併顯示；後續依序呈現 Epoch 選擇、Selection Confusion Matrix、OOS Confusion Matrix、各資料區段比較、排序與校準診斷、OOS 年度診斷及 OOS 綜合判定。排序診斷固定包含PR-AUC、Precision@50/60/70% coverage、Recall@60% Precision、Brier與ECE，診斷threshold不得用於回頭調整OOS。OOS 綜合判定合併原本的 Selection/OOS 差異與部署判定，依「主要成效、過度篩選防線、輔助診斷」三類編排，並新增逐項判讀欄。資料區段與日期分欄；第 4 區固定精簡為「原始 PASS、模型 PASS、PASS Precision、Precision 絕對、PASS Recall、平均 Score」，依此順序呈現。REJECT Specificity、REJECT NPV、Accuracy 與 Precision 相對僅保留在 Confusion Matrix 下方或 OOS 綜合判定。Confusion Matrix 中央只保留 TP／FN／FP／TN；右側依序顯示「原始PASS → TP + FN」與「原始REJECT → FP + TN」，底部依序顯示「TP + FP → 模型PASS」與「FN + TN → 模型REJECT」。分類品質另以「指標、公式、結果、解釋」表呈現；Precision 絕對／相對另以「指標、公式、結果」表呈現。Confusion Matrix 前不再重複顯示統計口徑或列／欄說明。終端會以淡藍、綠、黃、紅標示重點；Confusion Matrix 僅以綠色標示 TP／TN、紅色標示 FP／FN，原始／模型類別與合計維持中性色，且每一行獨立重設 ANSI 色碼，避免跨格污染。重新導向或測試輸出不插入 ANSI 色碼。Markdown 以相同語意顏色呈現，完整 metrics JSON 會寫入 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/reports/`。批次或需要可重現命令時使用 `workflow`：

```bash
python apps/breakout_quality.py workflow --filter-id breakout_quality_v1 --dataset full --experiment-profile unique_group_sampling --epochs 200 --batch-size 128 --evaluation-batch-size 4096 --evaluation-workers 4 --no-parallel-split-evaluation --train-prefetch-batches 0 --preload-feature-bank --device auto --mixed-precision --mixed-precision-dtype auto --deterministic-algorithms --no-allow-tf32 --lr 0.0003 --weight-decay 0.0001 --gradient-clip-norm 1.0 --final-refit-mode selected_epochs --class-weight-mode none --time-weight-mode none --seed 42 --fixed-threshold 0.50 --use-inner-validation --inner-validation-months 24
```

模型架構與訓練實驗分開管理：

```python
# config/breakout_quality_policy.py
BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "inception_time_v1"
BREAKOUT_QUALITY_EXPERIMENT_PROFILE = "unique_group_sampling"
BREAKOUT_QUALITY_PRETRAINING_PROFILE = "ts2vec_selection_only"
```

- `BREAKOUT_QUALITY_MODEL_ARCHITECTURE` 只描述網路與輸入結構。目前policy預設為已接受排序／高品質基準9A-BN `inception_time_v1`；8F `multiscale_cnn_sequence_only_v1`保留高覆蓋基準；`inception_time_market_set_v1`為尚待完整Selection／OOS驗證的active research architecture，Stage 0／1只允許研究鏈，不具正式forward runtime資格。9F `patch_transformer_v1`完整OOS排序低於9A，已轉為legacy read-only；其10-bar非重疊patch、128維embedding、3層／4-head Transformer、256維MLP、sinusoidal position與patch-mean pooling規格只供舊工件重建。9E MOMENT、9D MantisV2、9C TS2Vec、9B ModernTCN、9A-GN及8P亦為legacy read-only。
- `BREAKOUT_QUALITY_EXPERIMENT_PROFILE` 描述下游supervised optimizer、LR schedule、augmentation與training sampling unit；`BREAKOUT_QUALITY_PRETRAINING_PROFILE`只保留9C legacy TS2Vec encoder的optimizer、epochs、batch、LR、weight decay、gradient clip、crop、mask與contrastive loss；active 9A／8F及legacy 9D／9E／9F均不讀取此profile。profile定義集中在`config/breakout_quality_experiments.py`：已接受的9A-BN與高覆蓋基準8F均沿用`unique_group_sampling / time_weight=none`；9B ModernTCN雖沿用相同profile，但完整OOS固定coverage排序全面低於9A，已轉為legacy。8K `unique_group_date_balanced` 已由完整 OOS 淘汰，只保留歷史重現；8J direct best inner checkpoint 同樣只供歷史重現。`baseline`、`adamw_only`、`adam_warmup_cosine`、`history_masking_only` 保留為歷史 profile。
- `multiscale_cnn_sequence_only_dual_path_v1`、`multiscale_cnn_regime_context_v1`、`multiscale_cnn_v2～v8`、`tiny_cnn_v1` 與 `residual_tcn_v1` 保留為 legacy architecture，只供讀取舊 checkpoint、重現既有實驗與稽核歷史 manifest；正常 workflow 不再用它們建立新實驗。
- AdamW、scheduler、augmentation與sampling等訓練方法不建立假模型版本。9A-GN屬正規化結構變更，因此使用獨立architecture `inception_time_group_norm_v1`；唯一差異是8個`BatchNorm1d(128)`改為`GroupNorm(8, 128)`。完整OOS顯示固定coverage排序明顯低於9A-BN，因此已轉為legacy，不再允許正式新訓練；9A-BN與8F保留為accepted比較基準。
- 既有supervised Dataset、feature bank、future-path cache、4維event context arrays與labels可直接沿用，不需重建或relabel。9D legacy工件固定把每個300-bar feature channel獨立線性插值至512，送入釘死 `paris-noah/MantisV2` revision的frozen encoder，取第3層（index 2）CLS＋mean combined embedding，再串接10個channel embedding並只訓練單一linear head；encoder內部以固定chunk切分，避免evaluation batch 4096一次展開40,960條單變量序列造成GPU記憶體尖峰。9C獨立Selection-only rolling-window dataset與pretrained encoder工件同樣只保留歷史重建；9E MOMENT checkpoint下載、凍結encoder與linear head鏈也只供legacy重建。正式9F workflow不執行project pretraining或任何外部checkpoint下載；既有supervised Dataset不需重建。9C、9D與9E鏈只供legacy重建。MantisV2、TS2Vec frozen probe與sequence-only models在forward時明確不讀取event context。8F 在 Inner Train／Final Refit 只保留每個 `ticker/date` 的最小原始 row index，batch size 128 因而代表 128 個 unique groups；early-stopping patience 固定為 1。Validation、Selection、OOS 與報表仍使用完整 rows及既有 `1/group_size` 口徑。每個 architecture/profile 使用獨立工件路徑。

- `multiscale_cnn_sequence_only_v1` 的 trainable parameters 比 v1 少 `4 × 32 = 128`，差異只來自 head 第一層不再接收 `high_len_norm`、`breakout_level_to_close`、`close_to_breakout_level`、`high_to_breakout_level`。


### Stage 0／1 Learned Global Market Set 研究流程

此實驗不改Label、候選300-bar window、RF229或9A候選encoder，只新增全市場point-in-time learned set branch。先使用獨立filter id並修改：

```python
# config/breakout_quality_policy.py
BREAKOUT_QUALITY_DEFAULT_FILTER_ID = "breakout_quality_v1_market_set_v1"
BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "inception_time_market_set_v1"
BREAKOUT_QUALITY_EXPERIMENT_PROFILE = "unique_group_sampling"
```

執行完整研究流程：

```bash
python apps/breakout_quality.py workflow --filter-id breakout_quality_v1_market_set_v1 --dataset full --experiment-profile unique_group_sampling --evaluate-oos
```

- 切換此architecture後必須完整重建Dataset；除了既有candidate feature bank，還會建立共用的`market_daily_features.npy`、`market_daily_valid_mask.npy`、`market_date_ordinals.npy`、`market_tickers.csv`與`group_market_date_index.npy`。
- 市場股票每日日輸入只含close-to-close、overnight、intraday、high-low range與log-volume change，加上明確history／stock mask；不預先加入MA50、MA200、RSI或ticker identity。
- 同一天多個breakout只計算一次市場表示。`BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH`預設4，限制每個實體訓練／推論batch最多展開4個不同市場日期；這是資源邊界，不是Label或sampling權重。
- 第一版只有4個Global Learned Queries；沒有candidate-conditioned attention、sector tokens或learned lag。這些功能必須等Stage 1顯示穩定增量後再獨立導入。
- 此架構目前只能匯出`research` scores。`export-scores --scope forward_oos`會明確拒絕，直到正式scanner具備每日point-in-time market bank、版本hash與coverage契約。


工件隔離方式：

```text
models/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/
outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/
```

舊版 v1 baseline 工件若仍在沒有 `baseline/` 子目錄的歷史路徑，讀取端會以唯讀相容方式載入；該 fallback 不可作為正式 `forward_oos` 的寫入目標。要更新正式分數，必須先以 `baseline` profile 重新訓練到 canonical profile 子目錄；所有新工件一律寫入 profile 子目錄。

- `workflow` 會自動分成三種處理：工件、profile、ticker coverage、feature/high_len/benchmark/path-cache、欄位契約或來源 CSV inventory 改變時完整重建；只有 label horizon/PASS/REJECT 改變且 horizon 未超過 future path cache 時執行快速 relabel；全部一致時跳過。 完整重建採用 `ticker/date` feature bank 去重、逐檔 CSV 讀取與 per-ticker chunk 合併，正式陣列可 mmap 載入。互動選單只有在判定不需更新時，才詢問「是否強制完整重建 dataset」，預設 N；選 Y 等同 `--rebuild-dataset`。單獨執行 `train` 時若偵測到來源已更新，會 fail-fast 並要求先重建，避免靜默使用過期 dataset。 Dataset 完整重建的逐股票進度固定在同一行刷新，避免大量輸出洗版；重新導向輸出時只保留最終進度摘要與必要的 skip 訊息。
- 尚未準備最終 OOS 評估時，可加 `--no-evaluate-oos`；報表只包含 Selection 內診斷，並明確標示不能作為正式泛化結論。
- 選單只是正式 UI orchestration；dataset、split、training、export 與 evaluation 規則仍只實作在既有子系統，不在 app 複製。
- `BREAKOUT_QUALITY_MODEL_ARCHITECTURE`、`BREAKOUT_QUALITY_EXPERIMENT_PROFILE`、`BREAKOUT_QUALITY_PRETRAINING_PROFILE`、epochs、training batch size、evaluation batch size、evaluation workers、parallel split evaluation、training prefetch、feature-bank preload、`learning rate`、`weight decay`、`gradient clip norm`、final refit mode、class weight mode、time weight mode、`random seed`、最少 train/validation rows、threshold 與 inner-validation 預設均集中於 `config/breakout_quality_policy.py`；互動選單直接採用 policy，不再逐項詢問，CLI 可單次覆蓋且不回寫 policy。Inner Validation 開啟時，正式模型模式依 policy 選擇：`selected_epochs` 在完整 eligible Selection重訓相同 epoch；`matched_optimizer_steps` 匹配更新量；`best_inner_checkpoint` 則直接恢復 best epoch state，不重新初始化、不執行 Final Refit。`class_weight_mode=none` 不對約 55/45 的 Label 額外做 inverse-frequency balancing；`time_weight_mode=none` 是第一階段基準，後續可單獨改為 `year_balanced_sqrt`，以溫和平方根權重降低事件密集年份對 loss 的支配。`evaluation batch size` 與 `evaluation workers` 控制 read-only 推論。完整 Train／Validation／Selection 評估仍以 event rows 為單位，維持原 batch boundaries、全部 requested rows、輸出列序與最終 reduction 順序；score export 對 `use_dataset_context=false` 的模型則以 unique ticker/date feature groups 定義固定 batches，每個 group 只推論一次並廣播至原 event-row 輸出列序。開啟 `parallel split evaluation` 時，Inner Train 與 Validation 的完整評估同時執行，峰值最多使用 `2 × evaluation workers`，但各自仍使用原本的資料列、batch 與模型快照。訓練仍固定單執行緒；`zero_grad(set_to_none=True)`、experiment profile 指定的 Adam／AdamW、weight decay、gradient clipping、RAM preload 與可選的 batch prefetch 均忠實套用並寫入 manifest；除已明確設定的 regularization 外，不改訓練 rows、shuffle 或 batch 邊界。Final refit 模式最後一輪的完整指標會直接沿用；`best_inner_checkpoint` 會在恢復 checkpoint 後對完整 Selection 做一次完整 rows、既有 `1/group_size` 的 group-weighted 研究評估。Research／forward-OOS score export 對 `use_dataset_context=false` 的 sequence-only active model 改為每個 unique ticker/date feature group 只推論一次，再精確廣播到全部 high_len event rows；使用 Dataset context 的 legacy model 才維持逐 event-row 固定 batch 推論。兩條路徑都保留原輸出列序。9A起GPU是正式research路徑；目前policy architecture為`inception_time_v1`，9A與8F分別保留排序／高品質及高覆蓋比較基準，9C `ts2vec_frozen_linear_v1`、9D `mantis_v2_frozen_linear_v1`、9E `moment_1_base_frozen_linear_v1`與9B `modern_tcn_v1`只保留歷史重建：`device=auto`優先CUDA，mixed precision自動優先BF16、否則FP16；deterministic algorithms預設開啟、TF32預設關閉。training execution會寫入checkpoint/manifest；CPU fallback仍保留，但不同device/dtype結果須視為不同execution contract。`train` 終端的 Epoch 選擇每輪顯示完整 Inner Train Loss、Validation Loss、新最佳標記與該 Epoch 總耗時；完整 Selection 重訓沒有獨立 Validation，因此每輪顯示 Train Loss 與耗時。完整研究流程會在每一階段結束後顯示階段耗時，並在最後顯示 workflow 總耗時；互動終端中的時間值使用淡藍色，重新導向或測試輸出不插入 ANSI 色碼。關鍵資料區段與工件路徑保留在終端，完整 history、split policy、overlap、counts 與 Epoch `elapsed_sec` 仍保留於 `manifest.json`，不再將整包 Python dict 印到終端。

也可逐步執行：

```bash
python apps/breakout_quality.py build-dataset --dataset full --filter-id breakout_quality_v1

# 9C TS2Vec已淘汰；build-pretrain-dataset／pretrain只保留歷史工件重建，active 9A workflow不執行。

# 只更新 label；通常由 workflow 自動偵測，不需手動執行
python apps/breakout_quality.py build-dataset --dataset full --filter-id breakout_quality_v1 --relabel-only
# 預設關閉 inner validation：epochs 是完整 Selection 的正式固定訓練次數
python apps/breakout_quality.py train --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --epochs 20 --lr 0.001 --time-weight-mode none --seed 42 --fixed-threshold 0.50 --no-use-inner-validation
# 開啟時：epochs 是搜尋上限；以 Selection 尾端 N 個月選 best epoch，之後依 final-refit-mode 直接採用 best checkpoint 或進行完整 Selection 重訓
python apps/breakout_quality.py train --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --epochs 200 --device auto --mixed-precision --mixed-precision-dtype auto --deterministic-algorithms --no-allow-tf32 --lr 0.0003 --weight-decay 0.0001 --gradient-clip-norm 1.0 --final-refit-mode selected_epochs --class-weight-mode none --time-weight-mode none --seed 42 --fixed-threshold 0.50 --use-inner-validation --inner-validation-months 24
python apps/breakout_quality.py export-scores --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --scope research --device auto --mixed-precision --mixed-precision-dtype auto --deterministic-algorithms --no-allow-tf32 --inference-batch-size 4096 --inference-workers 4 --preload-feature-bank
# 建議日常使用：終端表格報表 + Markdown 解釋報表 + 完整 metrics JSON
python apps/breakout_quality.py report --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --no-include-oos
# 參數與模型已鎖定後，才把最終 OOS 納入報表
python apps/breakout_quality.py report --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --include-oos
# 需要稽核單一 split 的完整原始 JSON 時才使用 evaluate
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --split train
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --split validation
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --split selection
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --split oos
# 稽核 Selection 是否涵蓋 OOS 的市場狀態，並歸因指定年度；不重建 Dataset、不重訓、不改 Label
python apps/breakout_quality.py regime-audit --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --focus-year 2022 --min-selection-groups 100 --min-support-share-ratio 0.5
```

`regime-audit` 也可由 `python apps/breakout_quality.py` 的互動選單執行。它以 canonical 300-bar feature bank 中的0050序列，在每個 `ticker/date` breakout group只計算一次事件日可觀測市場狀態。Trend使用0050相對200日均線與60日報酬；Drawdown使用距252日高點；Volatility只以Selection的20日年化波動率三分位數定義low／medium／high，OOS不得參與分箱。輸出固定包含年度覆蓋、各regime的Selection/OOS event share、PASS率、Precision、Recall、PR-AUC、combined-regime支撐數與低代表性標記；`--focus-year` 另輸出該年度各combined regime的TP／FP／TN／FN、年度錯誤貢獻，以及描述性排除low-support事件後的指標。排除比較只作歸因，不得回頭建立年度／regime gate或調整threshold。

- Dataset 與 future-path cache 固定在 `outputs/filters/breakout_quality/<filter_id>/`；research scores 與易讀報表依架構及實驗放在 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/`。一般評估固定輸出 `reports/evaluation_report.md` 與 `reports/evaluation_metrics.json`；regime稽核固定輸出 `reports/regime_coverage_audit.md`、`reports/regime_coverage_audit.json`、`reports/regime_coverage_cells.csv`、`reports/regime_focus_year_cells.csv` 與 `reports/regime_event_groups.csv`。
- Model、manifest 與 `split_assignments.csv` 依架構及實驗固定在 `models/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/`；固定 threshold 也寫入 manifest，OOS 評估不得改用其他值；正式啟用時 active `breakout_quality_score_threshold` 應與該固定值一致。
- 外層正式期間只有 `selection / oos`，日期直接讀 `core.walk_forward_policy`。
- `BREAKOUT_QUALITY_USE_INNER_VALIDATION=False` 時，全部 eligible Selection 固定 epochs 訓練；開啟時，Selection 尾端 `BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS` 個月用來選 epoch。正式預設為 8F `BREAKOUT_QUALITY_FINAL_REFIT_MODE="selected_epochs"`；8I matched optimizer steps與 8J best inner checkpoint 均只保留歷史重現。
- inner validation 只可選 epoch；fixed threshold 仍在 OOS 前鎖定，不可由 validation 或 OOS 自動最佳化。
- Rolling OOS fold 必須沿用既有 `V16_WF_SELECTION_START_DATE`、`V16_WF_SEARCH_TRAIN_END_DATE`、`V16_WF_OOS_START_DATE`、`V16_WF_OOS_END_DATE` policy override；不得另傳一套 breakout-quality 專用日期。
- `research` 分數固定寫到 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/research_scores.csv`，不會改動正式 `scores.csv` 或 model manifest。
- 正式 `models/.../scores.csv` 只能由明確的 `--scope forward_oos` 建立。

## 建立正式 forward-OOS score table

1. 先以正式 app 的 `build-dataset` 與 `train` 子命令建立完整研究資料並訓練；`train` 依既有 walk-forward policy 執行固定 epoch 模式，或以 Selection 內 validation 選 epoch 後完整重訓，並保留 `model.pt`、`split_assignments.csv`、`manifest.json` 與 `model_information_cutoff`。
2. 若目前 dataset 已包含 outer OOS，可直接匯出；若需延伸到更新資料，只重新執行 `build-dataset`，不可重新 train 同一模型。
3. 執行：

```bash
python apps/breakout_quality.py export-scores --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --scope forward_oos
```

- `forward_oos` 的策略執行期仍使用同一份 walk-forward OOS window；但盤前下單使用前一交易日訊號，因此 Score 匯出會從 `model_information_cutoff` 當日開始建立訊號 coverage，包含 OOS 首個執行日前必要的 signal anchor。這些前置 Score 只供首日盤前排序／過濾，不會把策略回放或 OOS 評估提前。
- Manifest 會分別記錄 `required_signal_start`、`execution_start`、實際第一個 Score 事件日 `available_from` 與 `available_through`。`required_signal_start` 等於 `model_information_cutoff`；cutoff 當日 Score 只可供下一交易日盤前訂單使用，早於該日不得使用模型；`available_through` 後若出現候選事件則 fail-fast。
- 正式 runtime 只讀目前 policy 架構與 experiment profile 的 `models/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/scores.csv`，並驗證 model/score SHA256、schema、high_len coverage、OOS eligibility 與可用日期。
- `forward_oos` 會以目前原始資料重新建立 runtime 突破候選全集，不再只沿用訓練 Dataset 內可建立特徵的事件。可評分事件使用模型 score；因 benchmark 日期缺失、300-bar feature history 不足或特徵無效而不可評分的正式候選，會寫入同目錄 `unavailable_scores.csv`，並在 `scores.csv` 固定記為 `0.0`（保守 REJECT）。未知缺分仍會 fail-fast。
- 每次原始 CSV inventory 改變或套用本契約修正後，都必須重新執行本命令；不需重訓模型或重跑 Rolling optimizer。
- 現有舊版 `breakout_quality_v1` manifest 不符合新版 artifact contract 時，必須依上述流程重建，不得由 runtime 猜測或自動相容。

## 固定 9A 策略層經濟效果對照

### 正式無前視 OOS 對照

正式比較不能拿「今天才訓練完成」的單一 `models/run_best_params.json` 回放整段 2021～2025；該參數已看過後期資料，不符合歷史交易日使用當時已生效 active param 的原則。應先由 Rolling OOS Optimizer 產生按生效日切換的參數組。不要加 `--timing`，因 timing mode 不寫出正式 `roos_*.json`：

```bash
python apps/ml_optimizer.py --dataset full --outer-oos --trials 10 --outer-first-oos-date 2021-01-01 --outer-last-oos-date 2026-01-01 --outer-window-mode fixed --outer-train-window-months 120 --outer-oos-months 12 --yes
```

目前 Trade selector 對應的正式參數組為：

```text
models/roos_base_finalists_agree.json
```

完成上方 `forward_oos` score 匯出與 Rolling OOS 參數組後執行：

```bash
python apps/breakout_quality_strategy_compare.py --dataset full --params models/roos_base_finalists_agree.json --max-positions 10 --rotation off
```

- 比較期間固定為 runtime manifest 的 `execution_start` ～ `available_through`；Score table 可從更早的 `required_signal_start` 開始，只用來供應首個執行日前的原始 breakout signal Score。Rolling active-param 生效期間只需完整覆蓋實際策略執行期，不需覆蓋前置 Score anchor。
- 每個歷史交易日都使用 Rolling OOS 檔內當日已生效的單一參數或 seed ensemble；兩組交易日期與 0050 benchmark 必須完全一致。
- 工具與 Portfolio Simulator 共用 `core.portfolio_engine`、正式 signal generation、成交／費用／資金／持股延續及 `core.portfolio_stats`；Optimizer 也共用相同核心，但 Optimizer 是參數搜尋流程，策略對照是固定參數 OOS replay，兩者不是相同工作流。
- 兩組完整參數／每個 ensemble member 只能有 `use_breakout_quality_filter=False/True` 一項差異；filter ID 與 threshold 均固定為 active policy／manifest 值，不重新最佳化任何策略參數。
- `Candidate_Supply_Gap` 是「盤前可用持股格數 − 當日可掛單候選數」的非負值，只表示候選供給是否足夠；`End_Position_Gap` 才是成交執行後仍未滿倉的格數。
- 輸出固定在 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/strategy_compare/`，包含 Markdown／JSON 主報表、兩組 equity／trade／daily-capacity CSV、年度報酬比較，以及一筆一列的 round-trip 交易歸因報表。
- 正常策略比較完成後會自動產生 `trade_attribution.md/.json`、`trade_attribution_trades.csv`、`trade_attribution_yearly.csv` 與兩組 `*_round_trips.csv`。配對鍵固定為 ticker＋實際進場日＋進場類型；R、PnL、費稅與結算沿用 Portfolio Engine 的 closed-trade 真理來源。
- 若策略比較已經跑完，只需重建歸因與修正部分年度標記，不必再次執行 replay：

```bash
python apps/breakout_quality_strategy_compare.py --attribution-only
```

- `--attribution-only` 只讀取既有 `strategy_comparison.json`、`no_filter_trades.csv`、`quality_filter_trades.csv` 與正式 runtime score；它會把只到 2026-03-02 的 2026 年標為非完整年度，再輸出交易歸因。
- 此對照只判斷固定 9A 是否改善淨報酬、回撤、穩定性及資金使用；不得依結果回頭調整 threshold、epochs、feature、Label 或模型。

### Breakout Quality Score 候選排序探索性比較

固定 threshold 0.5 hard filter 已由策略 OOS 淘汰；若要測試 9A 的相對排序能力，只使用下列隔離模式：

以 `base_finalist_best` 單一 runtime member 做較純的 Score Ranking ablation：

```bash
python apps/breakout_quality_strategy_compare.py --comparison-mode score-ranking --param-policy base-finalist-best --dataset full --max-positions 10 --rotation off
```

工具會自動使用 `models/roos_base_best.json`，並驗證每個生效日恰有 1 個 member、`min_agree=1`。Baseline 與 score-ranking 兩組都固定 `use_breakout_quality_filter=False`；唯一差異為 `use_breakout_quality_ranking=False/True`。因所有候選票數皆為 1，實際有效排序為「Quality Score 由高到低 → 既有買入排序 → deterministic ticker」。輸出位於 `strategy_compare_score_ranking_base_finalist_best/`。

保留 `base_finalists_agree` 的既有探索性比較時使用：

```bash
python apps/breakout_quality_strategy_compare.py --comparison-mode score-ranking --param-policy base-finalists-agree --dataset full --max-positions 10 --rotation off
```

此模式自動使用 `models/roos_base_finalists_agree.json`；候選先通過 `min_agree`，再依「finalist同意數由高到低 → 同票Quality Score由高到低 → 既有買入排序 → deterministic ticker」，輸出位於 `strategy_compare_score_ranking_base_finalists_agree/`。

- `--param-policy` 與參數檔內 `selector` 不一致時直接拒絕；`base-finalist-best` 另要求每期 `1 member / min_agree=1`。
- 可正常評分但低 Score 的候選仍保留，只是順位靠後；`unavailable_scores.csv` 中不可評分候選維持保守排除。Continuation 與 STOP 後 Re-entry 沿用原始 breakout Score。
- Optimizer search space 固定 ranking=`False`，不得把此機制放入參數搜尋。
- 此實驗是在已查看舊 OOS 後進行的探索性機制比較；即使改善，也必須由全新 forward period 驗證後才可考慮部署。

### `run_best_params.json` 的用途與產生方式

Trade Mode 會先輸出 `models/candidate_best_params.json`；目前 `TRADE_MODE_AUTO_PROMOTE_RUN_BEST=True`，候選通過正式 promotion 契約時才建立或更新 `models/run_best_params.json`：

```bash
python apps/ml_optimizer.py --dataset full --model trade --trials 10
```

目前 random-seed ensemble 已啟用，因此 `run_best_params.json` 可能是 static active-param ensemble，而不是單一參數 JSON；策略對照工具支援此格式，但僅可明確標記為非 OOS 敏感度診斷：

```bash
python apps/breakout_quality_strategy_compare.py --dataset full --params models/run_best_params.json --allow-static-diagnostic --max-positions 10 --rotation off
```

不指定 `--params` 時，也只有加上 `--allow-static-diagnostic` 才會使用正式 primary param source。此結果不可作為 2021～2025 無前視 OOS 部署證據。

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

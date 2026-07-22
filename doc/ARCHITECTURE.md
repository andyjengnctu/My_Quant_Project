# 架構概覽

本文件只保留穩定分層、正式入口、依賴方向與共享邊界。操作步驟看 `doc/CMD.md`；formal contract 與狀態看 `doc/TEST_SUITE_CHECKLIST.md`。

## 穩定檔案樹

```text
project/
├─ apps/
│  ├─ breakout_quality.py             # Breakout quality 正式入口（選單／workflow／子命令分派）
│  ├─ ml_optimizer.py                 # 參數最佳化正式入口（薄入口）
│  ├─ portfolio_sim.py                # 投組模擬正式入口（薄入口）
│  ├─ smart_downloader.py             # 資料下載正式入口（薄入口）
│  ├─ package_zip.py                  # 專案打包正式入口
│  ├─ test_suite.py                   # 一鍵測試正式入口（reduced）
│  ├─ vip_scanner.py                  # 掃描器正式入口（薄入口）
│  └─ workbench.py                    # GUI 工作台正式入口（薄入口）
├─ config/
│  ├─ breakout_policy.py              # breakout 策略預設與 optimizer high_len 範圍
│  ├─ breakout_quality_policy.py      # breakout quality active model/profile 與 feature/label/training 政策
│  ├─ breakout_quality_experiments.py # 命名訓練實驗 profile（optimizer/schedule/augmentation）
│  ├─ training_policy.py              # 訓練政策與 selection gate
│  ├─ display_policy.py               # console/report 顯示政策
│  └─ execution_policy.py             # 資金、費用與 runtime 執行預設
├─ core/
│  ├─ config.py                       # 相容 façade；穩定匯出設定常數與參數契約
│  ├─ strategy_params.py              # breakout + training gate + execution 聚合參數契約
│  ├─ capital_policy.py               # 單股/投組/scanner 共用資金與 sizing 規則
│  ├─ exact_accounting.py             # 正式整數 ledger / cost-basis allocation / tick 正規化單一真理來源
│  ├─ backtest_core.py                # 單股回測總控 façade
│  ├─ portfolio_engine.py             # 投組 timeline 總控 façade
│  ├─ model_paths.py              # models 目錄與預設參數來源解析
│  ├─ output_paths.py             # outputs/<category> 目錄正規化與建立 helper
│  ├─ output_retention.py         # outputs retention 雙門檻清理 helper
│  └─ display.py                      # 顯示 façade
├─ doc/
│  ├─ TEST_SUITE_CHECKLIST.md         # formal test suite 主表與索引
│  ├─ ARCHITECTURE.md                 # 本檔
│  └─ CMD.md                          # 常用指令與操作說明
├─ filters/
│  └─ breakout_quality/               # quality feature、shared split、artifact contract、model factory、正式 runtime score lookup
├─ models/
│  ├─ filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/
│  │  ├─ model.pt                     # architecture/profile-scoped canonical model artifact
│  │  ├─ split_assignments.csv        # outer Selection/OOS + Selection train/embargo assignment
│  │  ├─ manifest.json                # model spec/profile/split/score/OOS eligibility 契約
│  │  └─ scores.csv                   # architecture/profile-scoped canonical event score table
│  ├─ full_base_best.json            # Full finalist best base policy 參數檔
│  ├─ full_base_finalists_agree.json # Full finalist agree base policy 參數檔
│  ├─ full_ensemble_base.json        # Full seed ensemble base policy 參數檔
│  ├─ oos_ensemble_base.json         # OOS seed ensemble base policy 參數檔
│  └─ roos_ensemble_base.json        # ROOS seed ensemble base policy 參數檔
└─ tools/
   ├─ downloader/                     # 資料下載子系統
   ├─ filters/breakout_quality/        # quality dataset/train/export/evaluate 子系統實作與開發相容入口
   ├─ optimizer/                      # 參數最佳化子系統
   ├─ portfolio_sim/                  # 投組模擬子系統
   ├─ scanner/                        # 掃描器子系統
   ├─ trade_analysis/                 # 單股 trade-analysis 子系統
   ├─ validate/                       # validate / synthetic / real-case 驗證子系統
   ├─ local_regression/               # reduced formal orchestrator
   └─ workbench_ui/                   # GUI 子系統
```

## 關鍵 shipped 模組索引

### `tools/optimizer/`

```text
   │  ├─ runtime.py                   # optimizer 執行期狀態、匯出控制與歷史最佳還原
   │  ├─ session.py                   # optimizer session 狀態 façade
```

- `tools/optimizer/`：參數最佳化子系統；由 `apps/ml_optimizer.py` 進入。

### `tools/trade_analysis/`

```text
   │  ├─ history_snapshot.py          # 單股分析歷史績效 snapshot / payoff / asset-growth helper
```

- `tools/trade_analysis/`：單股 trade-analysis 子系統；由 `apps/workbench.py` 經 `tools/workbench_ui/` 觸發，`tools/trade_analysis/trade_log.py` 提供共用 backend / 開發輔助 CLI。
- 為維持相容性，保留 legacy `run_debug_*` API 名稱，同時提供 canonical `run_trade_analysis` / `run_trade_backtest` / `run_prepared_trade_backtest` / `run_ticker_analysis` aliases。

### `apps/breakout_quality.py`、`filters/breakout_quality/` 與 `tools/filters/breakout_quality/`

- `apps/breakout_quality.py` 是 dataset、training、score export、易讀 report 與詳細 evaluation 的單一正式使用者入口；負責互動選單、可重現 workflow orchestration、來源 CSV inventory freshness 判斷與子命令分派，但不複製 dataset、模型、split、export、metrics 或 report 規則。互動選單在第一層操作選定後，Filter ID 與全部訓練設定直接讀取 `config/breakout_quality_policy.py`，不再建立第二份互動預設或逐項重問。互動式完整研究流程固定使用 Full dataset、全部股票並執行 OOS；互動式單獨建立 dataset 也固定使用 Full dataset 與全部股票，因此不再詢問 dataset profile、ticker coverage 或 workflow OOS。dataset 是否需要處理由 workflow 自動偵測：來源 CSV、feature/high_len/benchmark/path-cache 或儲存契約改變時完整重建；只有 label horizon/PASS/REJECT 改變且仍落在 future path cache 範圍內時，只快速 relabel，不重算特徵；只有判定完全不需更新時才詢問是否強制完整重建，預設 N。evaluation split 與正式匯出仍保留必要確認；完整研究流程的開始確認預設為 Y；互動式易讀報表固定納入 OOS，不再詢問。CLI 仍可作 reduced、max-tickers、略過 OOS 或其他單次 override，且不回寫 policy。
- `filters/breakout_quality/` 承接 feature/label、indexed feature-bank dataset storage、source-data inventory fingerprint、artifact contract、canonical path、外層 Selection/OOS split 與正式 runtime lookup。模型架構與訓練實驗分離：`multiscale_cnn_v1` 是正式 baseline；`multiscale_cnn_regime_context_v1` 是目前結構性實驗架構，保留相同三分支 Level CNN 與原 4 個 context，只從既有 300×10 sequence 即時計算 0050 20／60 日 log return、annualized volatility 與個股相對強弱，透過零初始化的 6→32 projection 注入 head。它不改 Dataset storage contract，也不使用未來資料。`multiscale_cnn_v2～v8`、`tiny_cnn_v1` 與 `residual_tcn_v1` 保留為 legacy architecture，只供舊 checkpoint／manifest 重建與歷史重現。AdamW、LR schedule、augmentation 等訓練方法由 `config/breakout_quality_experiments.py` 的命名 profile 管理；6A、6B、7A 已淘汰，本輪 profile 回到 `baseline`。checkpoint、manifest 與 canonical path 同時釘死 architecture 及 experiment profile；正式 runtime 不執行 CNN，只讀目前 policy architecture/profile 的 canonical `scores.csv`。
- `tools/filters/breakout_quality/` 承接 dataset、training、score export、研究 metrics 與 report rendering 子系統實作；dataset 以 `ticker/date` 為 group，只保存一份由 `BREAKOUT_QUALITY_FEATURE_WINDOW_BARS` 決定長度的 sequence feature，事件列以 `group_index` 加上 high_len context 取用；未來 K 線路徑另以可 mmap 的 `.npy` cache 保存，調整 label 門檻或 cache 範圍內的 horizon 時只重貼 label。建立時逐檔讀取股票 CSV，先寫入每檔暫存 chunk，再合併成可 mmap 的正式 `.npy`，不再一次把全部股票、事件表或特徵常駐 RAM，也不再使用大型壓縮 `dataset.npz`。Label 以突破訊號日收盤價為基準，逐日累積 MFE 與達成前 MAE；在尚未觸及最大不利跌幅前，MFE 嚴格大於最低漲幅且 MFE／MAE 嚴格大於最低比率才是 PASS，零 MAE 視為比率成立，同日同時達成 PASS 與風險上限時保守判為 REJECT；其餘完整有效路徑均為 REJECT。資料不足或 K 線無效者只標記為內部 INVALID 並排除，不是第三種 Label；不模擬成交、ATR、停損、停利或任何策略參數。訓練權重先以 `ticker/date` group 去除 high_len 重複放大，再依 policy 選擇 class weight 與 time weight；預設不做 inverse-frequency class balancing，年度平衡則保留 `year_balanced_sqrt` 溫和模式作獨立實驗。Inner Validation 選出的 best epoch 會記錄實際 optimizer updates；final refit 由 policy 的 `BREAKOUT_QUALITY_FINAL_REFIT_MODE` 決定，目前預設 `selected_epochs` 會以相同 epoch 數在完整 eligible Selection 重訓，另可切換為 `matched_optimizer_steps` 匹配 Inner Train 更新量並至少完整遍歷 Selection 一次。`evaluate.py` 是完整 JSON 稽核介面，`report.py` 只重用同一份 metrics，輸出表格化終端報表、Markdown 與完整報表 JSON，不另算第二套指標。報表預設納入 OOS，並依序呈現 Epoch 選擇、Selection／OOS Confusion Matrix、各資料區段比較與 OOS 綜合判定；固定 epoch 模式與 inner validation 模式使用不同的最終模型說明。OOS 綜合判定將 OOS 整體與各年度放在欄位，每個 OOS 儲存格同時顯示實際結果與相較 Selection 的差異。training 會驗證 source-data inventory，來源已更新時拒絕沿用過期 dataset。直接 CLI 僅保留開發與既有指令相容，不再作為文件建議的正式入口。
- Dataset 與 future-path cache 固定共用 `outputs/filters/breakout_quality/<filter_id>/`；正式模型工件依架構與訓練實驗分開存放於 `models/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/`。`BREAKOUT_QUALITY_MODEL_ARCHITECTURE` 與 `BREAKOUT_QUALITY_EXPERIMENT_PROFILE` 都不進入 Dataset fingerprint，也不觸發 feature bank 重建；舊 v1 baseline 無 profile 子目錄的工件僅提供唯讀相容 fallback；不得在該舊路徑更新 manifest 或寫入正式 forward-OOS scores，所有新訓練與正式輸出一律寫入 profile 子目錄。`split_assignments.csv` 的 outer `selection/oos` 日期直接來自 `core.walk_forward_policy`。
- inner validation 為可配置模式：關閉時完整 Selection 固定 epochs；開啟時只在 Selection 內選 epoch，之後以全部 eligible Selection 重訓。Final refit 忠實採用 `BREAKOUT_QUALITY_FINAL_REFIT_MODE`；目前預設 `selected_epochs`，也可切換為 matched optimizer steps 並保證至少一個完整 Selection pass。manifest 同步保存 selected/target/actual steps、等效 epochs、class/time weight mode 與權重摘要。OOS 永不參與 epoch、threshold 或模型選擇。
- 訓練效能優化以結果一致為硬性契約：每個 epoch 的訓練 rows、shuffle、batch size、seed、optimizer 更新次數與訓練單執行緒設定均不得改變。訓練使用 `zero_grad(set_to_none=True)`；optimizer、LR schedule 與 augmentation 由 experiment profile 決定；augmentation 僅允許作用於 training input，不可污染 Validation／OOS，通用 learning rate、weight decay 與 gradient clipping 仍由 policy 控制；模型每個參數在每個 batch 都有 gradient，可省去清零寫入而不改 optimizer 更新；可選擇將去重 feature bank 與小型事件陣列預載至 RAM，但資料值與列順序不變。完整 Train／Validation／Selection 評估與 score export 共用 `filters/breakout_quality/inference.py` 的 strict-result inference：維持相同 batch boundaries、全部 requested rows 與原輸出列序，只把彼此獨立的 batches 分派給 `BREAKOUT_QUALITY_EVALUATION_WORKERS` 個單執行緒 model replicas，再依原列序與原 reduction 順序彙總。開啟 `BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION` 時，Inner Train 與 Validation 使用相同 epoch model 的獨立 read-only 快照同時評估，不改 optimizer、random state 或 best-epoch 判定。Final refit 最後一輪已產生的完整 Selection metrics 直接作為 final metrics，不再做一次完全重複的推論。GPU 與多執行緒 training 不屬於 strict-result 模式，因其浮點 reduction 可能改變模型權重與 threshold 邊界判定。
- Dataset 工件只放 `outputs/filters/breakout_quality/<filter_id>/`；research scores 與 report 依架構及實驗放在 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/`，報表固定於該目錄下的 `reports/evaluation_report.md` 與 `reports/evaluation_metrics.json`，不得覆蓋正式 `scores.csv`。
- `dl_quality_score >= active breakout_quality_score_threshold` 是唯一通過判斷；manifest 只宣告契約與 OOS 可用日期，不預先固化另一份 `dl_pass`。
- 正式 runtime 僅在 manifest 宣告的有效期間套用模型；有效期後只要出現未覆蓋候選事件即 fail-fast。

### `tools/validate/`

- `tools/validate/`：正式 invariant、contract、schema 與 real-case 驗證子系統；正式細目與狀態以 `doc/TEST_SUITE_CHECKLIST.md` 為準。

### `tools/local_regression/`

```text
├── formal_pipeline.py
├── meta_quality_coverage.py
├── meta_quality_targets.py
├── run_meta_quality.py
```

- `tools/local_regression/`：reduced formal orchestrator；`formal_pipeline.py` 為正式步驟單一真理來源。
- `run_meta_quality.py`：meta quality 工具；負責 coverage / summary / baseline 與 formal step 對照。

## 子系統責任

- `apps/test_suite.py` 是日常唯一建議使用的一鍵測試入口。

- `apps/`：正式入口層，只從對應子系統 façade 匯入公開介面。
- `core/`：核心規則、帳務、價格、統計、path 與共用 helper；不得放 UI orchestration 或 validate 腳本。
- `tools/`：下載、最佳化、單股分析、validate、local regression 與 GUI 子系統；workbench 的交易明細與 Console 改以獨立分頁承接。
- `config/`：共用政策與執行預設。
- `models/`：最佳參數檔與模型相關輸入。
- `doc/`：架構、常用指令與 formal checklist 文件。

## 正式入口

- `apps/breakout_quality.py`：Breakout quality 互動選單、完整 research workflow、dataset、training、score export、易讀 report 與詳細 evaluation 單一正式入口。
- `apps/test_suite.py`：日常一鍵測試正式入口。
- `apps/ml_optimizer.py`：optimizer 正式入口。
- `apps/package_zip.py`：打包正式入口。
- `apps/portfolio_sim.py`：投組模擬正式入口。
- `apps/smart_downloader.py`：下載器正式入口。
- `apps/vip_scanner.py`：scanner 正式入口。
- `apps/workbench.py`：GUI / workbench 正式入口；也是單股 trade-analysis 的單一使用者入口。

## 正式單一真理來源 / 開發輔助

- `tools/local_regression/formal_pipeline.py`：formal 步驟單一真理來源，供正式入口與 local regression 內部編排使用；不是使用者正式入口。
- `tools/trade_analysis/trade_log.py`：單股 trade-analysis 共用 backend / 開發輔助 CLI；不是正式使用者入口。

## 子系統責任

- `optimizer`：參數搜尋、最佳化輸出與結果整理。
- `portfolio_sim`：投組模擬、統計與報表。
- `scanner`：候選掃描、排序與 issue log。
- `trade_analysis`：單股分析、圖表與交易明細輸出。
- `validate`：formal contract、schema、synthetic 與 real-case 驗證。
- `local_regression`：reduced formal orchestrator 與 bundle 產出。
- `workbench_ui`：GUI 主視窗與單股檢視頁面；上方控制列提供股票代號、常用股票、候選股與歷史績效股操作，K 線圖主檢視下交易明細與 Console 以獨立分頁承接。

## 依賴方向

- 分層呼叫固定為 `apps -> tools -> core`。
- `core/` 不反向依賴 `tools/` 或 `apps/`。
- 正式 test chain 只由 `apps/test_suite.py` 與 `tools/local_regression/formal_pipeline.py` 收斂。

## 共享邊界

- 所有工具輸出皆落在 `outputs/<category>/`；輸出位置與 retention 規則以 `core/output_paths.py`、`core/output_retention.py` 與 `doc/CMD.md` 為準。
- `outputs/local_regression/_staging/` 為 local regression / validate 共用暫存 staging 子目錄；不新增 `outputs/validate/` 根分類。
- `outputs/debug_trade_log/` 為 `trade_analysis` 相容輸出目錄；為維持相容性，暫沿用 `debug_trade_log` 這個 legacy 名稱。
- `outputs/debug_trade_log/`（trade_analysis legacy output dir）屬既有工具鏈相容邊界，不代表子系統角色仍是 debug-only。
- `outputs/workbench_ui/` 為 GUI runtime 快取分類；目前承接常用股票中文名稱快取。

## 維護原則

- 本檔只承接穩定子系統、正式入口、依賴方向與共享邊界。
- 高波動操作細節移至 `doc/CMD.md`；formal 細部契約移至 `doc/TEST_SUITE_CHECKLIST.md`。
- 不以 exact file-tree、helper 長清單、局部 alias 說明或暫時演進敘事作為本檔主要承載面。

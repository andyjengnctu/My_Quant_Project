# 架構概覽

本文件只保留穩定分層、正式入口、依賴方向與共享邊界。操作步驟看 `doc/CMD.md`；formal contract 與狀態看 `doc/TEST_SUITE_CHECKLIST.md`。

## 穩定檔案樹

```text
project/
├─ apps/
│  ├─ breakout_quality.py             # Breakout quality 正式入口（選單／workflow／子命令分派）
│  ├─ breakout_quality_strategy_compare.py # 固定參數 no-filter vs active quality filter 對照入口
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
│  └─ <optimizer parameter artifacts>.json # runtime 產生或使用者保留的可選參數工件；檔名依 optimizer mode／selector 而定
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
- `filters/breakout_quality/` 承接 feature/label、indexed feature-bank dataset storage、source-data inventory fingerprint、artifact contract、canonical path、外層 Selection/OOS split 與正式 runtime lookup。模型架構與訓練實驗分離：目前policy已退回9A `inception_time_v1`排序／高品質基準，8F `multiscale_cnn_sequence_only_v1`保留高覆蓋基準；10A `inception_time_market_set_candidate_v1`完整OOS排序低於9A，已轉為legacy read-only。`inception_time_market_set_v1`修正logical-batch契約後完整OOS仍低於9A，已轉為legacy read-only。9F `patch_transformer_v1`完整OOS排序低於9A，已轉為legacy read-only；其300×10 sequence依10 bars切成30個非重疊patch、128維embedding、3層／4-head Transformer、MLP 256、sinusoidal position與patch mean pooling規格只供舊checkpoint／manifest重建。9E `moment_1_base_frozen_linear_v1`完整OOS排序低於9A與8F，亦為legacy read-only。9E釘死官方`AutonLab/MOMENT-1-base`revision與checkpoint SHA256，runtime固定為`momentfm==0.1.4 / transformers==5.5.0`且以`--no-deps`隔離安裝避免降級主環境，300×10輸入固定插值至512，使用embedding mode保留10個channel，各channel對patch取mean後串接為7680維，只訓練linear head，不使用Dataset context、project pretraining、OOS或PASS／REJECT labels訓練encoder。9D `mantis_v2_frozen_linear_v1`完整OOS固定coverage排序低於9A，已轉為legacy read-only。9D釘死官方Hugging Face repository、commit revision、config與checkpoint SHA256；300×10輸入逐channel固定插值至512，以官方建議的第3層（index 2）combined token輸出，各channel獨立編碼後串接，只訓練linear head且不建立project pretraining dataset。9C `ts2vec_frozen_linear_v1`以Selection-only未標記rolling windows做TS2Vec-style hierarchical contrastive pretraining後凍結encoder，但完整OOS排序明顯低於9A，已轉為legacy read-only。9B `modern_tcn_v1`使用6個96-channel residual blocks、kernel 51 depthwise temporal convolution、4× pointwise expansion與global-average pooling；雖與9A容量近似，但完整OOS固定coverage排序、Accuracy與校準全面惡化，已轉為legacy read-only。9A-GN `inception_time_group_norm_v1`只把6個Inception module與2個residual projection的`BatchNorm1d(128)`改為`GroupNorm(8, 128)`，但完整OOS固定coverage排序顯著下降，已轉為legacy read-only。`inception_time_v1`維持排序／高品質實證基準；`multiscale_cnn_sequence_only_v1` 保留為高覆蓋實證基準；它保留 `multiscale_cnn_v1` 的 300×10 Level sequence、三分支 CNN、pooling、dropout 與 32 維 head，只讓 head 不再拼接 Dataset 既有的 4 維 event context，因此 trainable parameters 僅減少 128，Dataset storage contract、feature bank、context arrays 與 labels 都不需重建。8P `multiscale_cnn_sequence_only_dual_path_v1` 的完整OOS只讓Precision增加0.06 pp，卻降低Recall、Accuracy與Score，已轉為legacy read-only；`multiscale_cnn_v1` 則保留為使用原4維event context的歷史比較架構。`multiscale_cnn_sequence_only_dual_path_v1`、`multiscale_cnn_regime_context_v1`、`multiscale_cnn_v2～v8`、`tiny_cnn_v1` 與 `residual_tcn_v1` 保留為 legacy architecture，只供舊 checkpoint／manifest 重建與歷史重現。AdamW、LR schedule、augmentation、time weighting 與 sampling unit 等supervised訓練方法，以及9C TS2Vec的optimizer／epochs／batch／LR／crop／mask／contrastive loss設定，均由 `config/breakout_quality_experiments.py` 的命名 profile 管理；8B `recent_decay_60m` 已被完整 OOS 淘汰。8F 使用 `unique_group_sampling / batch_size=128 / time_weight=none / patience=1`，只將 sequence-only training unit 改為 unique `ticker/date` group；完整 OOS 已證明 Recall、Accuracy、Score 與 drift 大幅改善，因此 8F 保留為高覆蓋研究基準。9A InceptionTime 在固定coverage下的Precision跨Selection／OOS幾乎不下降，升為排序／高品質模型基準。Active `inception_time_v1` 的 depth、目標 receptive field 與 residual interval 由 `config/breakout_quality_policy.py` 單一設定；kernel sizes 依目標視野自動生成為三個近似 1x／1/2x／1/4x 的正奇數尺度，實際 receptive field 與 kernels 寫入 model manifest。預設 minimum target 228 bars 會精確還原 9A 的 depth 6、kernels 39／19／9與實際 receptive field 229 bars；legacy `inception_time_group_norm_v1` 維持原始固定結構，不受 active 設定影響。8G patience 5 已淘汰，因其重新造成 Selection 過擬合與 OOS score drift；8H batch size 64 使 optimizer updates 加倍但 OOS Precision、Accuracy 與 Score 均低於 8F；8I matched optimizer steps 雖提高 Precision，但 Recall、Accuracy、Score與泛化落差低於 8F，因此 8F 實證基準維持 batch 128、patience 1、`selected_epochs`。8J `unique_group_best_inner_checkpoint` 已完成但被 OOS 淘汰：Precision幾乎不變，Recall、Accuracy 與 Score明顯低於 8F；該 profile只供歷史重現。8K `unique_group_date_balanced` 已由完整 OOS淘汰：threshold 0.5下幾乎全部判PASS，Precision Lift只剩+0.45 pp；該profile只供歷史重現。現有multiscale CNN細調、supervised ModernTCN、三個frozen probe（TS2Vec、MantisV2、MOMENT）、9F supervised Patch Transformer與10A Candidate-conditioned Query均停止；9A與8F分別保留排序／高品質及高覆蓋比較基準。Stage 1 `inception_time_market_set_v1`與9A-GN、9B、9C、9D、9E、9F完整OOS均已淘汰，只供舊checkpoint／manifest重建。現有300×10單模型architecture與Market Set橫向搜尋停止；下一步改做11A連續策略對齊target的資料與可學性稽核，不先調整模型容量、threshold或Market Query。Patch Transformer、ModernTCN與InceptionTime訓練均支援`device=auto/cpu/cuda`、CUDA mixed precision（auto優先BF16，否則FP16）、deterministic algorithms與TF32顯式契約；checkpoint/manifest保存training execution，score export可使用獨立推論device。checkpoint、manifest 與 canonical path 同時釘死 architecture 及 experiment profile，model spec 另以 `use_dataset_context=false` 明確記錄 sequence-only 契約；正式 runtime 不執行 CNN，只讀目前 policy architecture/profile 的 canonical `scores.csv`。

- Legacy `inception_time_market_set_v1` 曾在9A候選分支之外增加Stage 0／1全市場表示：Dataset以benchmark交易日為calendar，保存point-in-time `date × ticker × 5`基礎OHLCV變化與valid mask，同日breakout共用同一market date index；Shared Stock Temporal Encoder（GroupNorm，避免不同market batch／無效股票比例污染其他股票的正規化統計）以同一套權重把每檔股票300日壓成32維表示，4個Global Learned Queries以排列不變attention pooling形成128維市場表示，再與9A候選128維表示融合。第一版不含ticker identity、candidate-conditioned query、sector token、learned lag或股票兩兩self-attention；實體batch依market date分塊並限制不同日期數，避免重複展開全市場tensor。此architecture已由修正後完整OOS淘汰，不再允許正式新訓練；舊research checkpoint／manifest仍可嚴格重建，`forward_oos`與scanner runtime維持fail-fast。

- Active `inception_time_market_set_candidate_v1` 沿用既有point-in-time Market Set Bank與Shared Stock Temporal Encoder，但不再使用與候選無關的Global Learned Queries。每個候選128維embedding會投影成可設定數量的candidate-conditioned query（目前1個），對該事件日期的全市場32維stock embeddings做masked multi-head cross-attention，再形成128維candidate-specific market embedding與候選表示融合。Market Bank仍依日期microbatch物化，候選encoder仍對完整logical batch只forward一次；因此`max_dates_per_batch`只影響記憶體，不改batch=128、optimizer step、loss denominator或epoch語意。此架構目前只允許research score export，`forward_oos`與scanner仍fail-fast。
- 正式 `forward_oos` score export 不得只重播訓練 Dataset 內「成功建立特徵」的事件；它必須重新使用目前 canonical OHLCV 清洗與突破 crossover 規則建立當前 runtime 候選全集。每一個 `ticker/date/high_len` 候選必須二擇一：可建立完整模型輸入者寫入模型 probability；因 benchmark 日期缺失、歷史窗不足或非有限特徵而不可評分者，明確寫入同目錄 `unavailable_scores.csv`，並在 canonical `scores.csv` 以固定 `0.0` 保守映射為 REJECT。manifest 必須保存 current source CSV inventory、候選總數、模型評分數、保守拒絕數及原因統計；runtime 必須驗證 audit rows 與 `scores.csv` 的 0.0 一致。只有已被正式記錄的不可評分事件可保守拒絕，任何未記錄缺分仍須 fail-fast。
- `apps/breakout_quality_strategy_compare.py` 是固定 active breakout-quality 策略經濟效果驗證的薄入口；`tools/filters/breakout_quality/strategy_compare.py` 負責載入 active forward-OOS runtime contract與外部 active-param source，建立完全相同的 no-filter／quality-filter 參數後只切換 `use_breakout_quality_filter`，並透過 Portfolio Simulator 的 canonical runners 執行同期間回放。正式歷史 OOS 模式只接受 Rolling OOS 單一參數 schedule 或 Rolling active-param seed ensemble，且其生效期間必須完整覆蓋 filter forward-OOS；每個交易日由既有 active-param resolver 使用該日已生效參數，符合任何交易日所有決策使用當日 active param 的唯一真理。單一參數或 static `run_best_params.json` 僅在顯式 `--allow-static-diagnostic` 時允許，報表必須標為非無前視敏感度診斷。工具不得改寫 optimizer search space、不得重選策略參數，也不得使用 research-only score 取代 canonical `scores.csv`。Optimizer、Portfolio Simulator與本工具共用 `core.portfolio_engine` 的成交／費用／帳務／資金與持股延續語意及 `core.portfolio_stats`；差異僅在 Optimizer 執行搜尋與訓練區段，本工具執行固定參數 OOS replay。`core/portfolio_engine.py` 只在非訓練 profile 中附加每日 `portfolio_capacity_rows` 與一筆一列 `closed_trade_rows` 診斷：前者保留候選供給／持股缺口，後者直接沿用扣費後 closed-trade PnL與R，不另算第二套round-trip。`tools/filters/breakout_quality/trade_attribution.py` 將兩組closed trades以ticker＋實際進場日＋進場類型精確配對，分成共同、no-filter only與filter only，並拆分直接門檻拒絕與投組路徑擠出。`--attribution-only` 可從既有transaction CSV重建相同round-trip，不重新跑portfolio replay。
- Breakout-quality candidate ranking 是 hard filter 之外的獨立策略機制。`use_breakout_quality_ranking=True` 時不得同時啟用 `use_breakout_quality_filter`；signal generation 不以 threshold 刪除可評分候選，`core/portfolio_candidates.py` 只在候選形成後讀取原始 breakout signal date 的 canonical runtime score。Continuation 與 STOP 後 Re-entry 沿用同一 setup 的原始 breakout Score payload；Re-entry 的重新站回確認日只作新的交易訊號日，不得拿確認日重新查 score table。Ensemble 成交持倉必須逐 member 保存原始 Score payload，STOP 後各 member 的 watch state 與再次形成的 Re-entry 共識都沿用各自原始 `score_date`；`unavailable_scores.csv` 記錄的不可評分事件維持保守排除，未知缺分仍 fail-fast。單一參數候選排序為 Quality Score 降冪後接既有 buy-sort；active-param ensemble 則先以 `min_agree` 決定資格，再固定依 vote count 降冪、同票候選的 median Quality Score 降冪、既有 buy-sort、ticker deterministic 排序。Optimizer search space 將 ranking 固定為 False；只有 `apps/breakout_quality_strategy_compare.py --comparison-mode score-ranking` 可建立 baseline/ranking 隔離對照，兩組 hard filter 皆為 False。`--param-policy base-finalist-best` 會自動解析 `roos_base_best.json`、驗證每期 `1 member / min_agree=1`，此時所有票數相同，Quality Score 是第一個有效排序欄位，輸出置於 `strategy_compare_score_ranking_base_finalist_best/`；`--param-policy base-finalists-agree` 則保留 finalist 同意數第一、Score只重排同票候選，輸出置於 `strategy_compare_score_ranking_base_finalists_agree/`。參數 selector 或 member/min_agree 契約不一致時必須 fail-fast。該比較是在已查看舊 OOS 後的探索性機制診斷，不得直接作部署證據。 正式 Score 工件的日期語意分成兩層：`required_signal_start` 等於 `model_information_cutoff`，代表模型可於 cutoff 當日收盤後用同日訊號建立下一交易日盤前訂單，必須涵蓋 OOS 首個執行日前一交易日的盤前訊號 anchor；`execution_start` 則固定等於 `outer_oos_policy.oos_start_date`。策略回放不得因補齊 signal score 而提前，亦不得改用執行日收盤 Score。
- `tools/filters/breakout_quality/` 承接 supervised dataset、score export、研究 metrics 與 report rendering；9E legacy鏈下載並驗證釘死的MOMENT-1-base snapshot，完整frozen encoder state與來源契約內嵌正式checkpoint，後續舊工件export／report不需重新下載；9D legacy鏈下載並驗證釘死的外部MantisV2 snapshot，checkpoint與manifest保存完整frozen encoder、來源revision/hash、套件版本及project data isolation flags，score export與runtime只從正式checkpoint重建，不在推論時重新下載；Selection-only pretraining dataset、TS2Vec encoder pretraining與frozen-head training子系統保留為9C legacy工件重建；9C、9D、9E與9F均不由active 9A workflow啟動；dataset 以 `ticker/date` 為 group，只保存一份由 `BREAKOUT_QUALITY_FEATURE_WINDOW_BARS` 決定長度的 sequence feature，事件列以 `group_index` 加上 high_len context 取用；未來 K 線路徑另以可 mmap 的 `.npy` cache 保存，調整 label 門檻或 cache 範圍內的 horizon 時只重貼 label。建立時逐檔讀取股票 CSV，先寫入每檔暫存 chunk，再合併成可 mmap 的正式 `.npy`，不再一次把全部股票、事件表或特徵常駐 RAM，也不再使用大型壓縮 `dataset.npz`。Label 以突破訊號日收盤價為基準，逐日累積 MFE 與達成前 MAE；在尚未觸及最大不利跌幅前，MFE 嚴格大於最低漲幅且 MFE／MAE 嚴格大於最低比率才是 PASS，零 MAE 視為比率成立，同日同時達成 PASS 與風險上限時保守判為 REJECT；其餘完整有效路徑均為 REJECT。資料不足或 K 線無效者只標記為內部 INVALID 並排除，不是第三種 Label；不模擬成交、ATR、停損、停利或任何策略參數。訓練資料先依 experiment profile 決定 sampling unit：baseline 使用全部 event rows 加 `1/group_size`，8F `unique_group_sampling` 則在 shuffle 前只保留每個 `ticker/date` 的最小原始 row index，使每個 group 每 epoch 只參與一次 optimizer sampling；之後再依 policy 選擇 class weight 與 training-only weight。8K `unique_group_date_balanced` 歷史 profile以各 training phase 內同日 eligible unique groups數 `n_d` 設定 raw weight=`1/n_d`，並使用固定 configured batch-size denominator；此方法已由完整 OOS淘汰；active 9A及8F比較基準均不套用日期平衡；9B ModernTCN歷史結果同樣未使用日期平衡。Validation／Selection／OOS始終使用完整 rows與 `1/group_size` 評估。預設不做 inverse-frequency class balancing，年度平衡則保留 `year_balanced_sqrt` 歷史模式。Inner Validation 選出的 best epoch 會記錄實際 optimizer updates；final refit 由 policy 的 `BREAKOUT_QUALITY_FINAL_REFIT_MODE` 決定，8F 正式基準的 `selected_epochs` 會以相同 epoch 數在完整 eligible Selection 重訓；`matched_optimizer_steps` 已由 8I 淘汰，只保留歷史重現。8J 的 `best_inner_checkpoint` 可供歷史重現，但已被完整 OOS 淘汰；active 9A及8F比較基準均維持 `selected_epochs`；9B ModernTCN歷史實驗亦使用相同refit。`evaluate.py` 是完整 JSON 稽核介面，`report.py` 只重用同一份 metrics，輸出表格化終端報表、Markdown 與完整報表 JSON，不另算第二套指標。報表預設納入 OOS，並依序呈現 Epoch 選擇、Selection／OOS Confusion Matrix、各資料區段比較與 OOS 綜合判定；固定 epoch 模式與 inner validation 模式使用不同的最終模型說明。報表另以同一份固定 OOS score 產生逐年度分類、固定 coverage 排序與校準診斷表，部分年度會明確標記；年度切片不得用於回頭選 threshold、epochs 或模型。OOS 綜合判定仍只比較 Selection 與完整 OOS。training 會驗證 source-data inventory，來源已更新時拒絕沿用過期 dataset。直接 CLI 僅保留開發與既有指令相容，不再作為文件建議的正式入口。
- 9C自監督legacy工件仍採獨立契約：rolling windows位於`outputs/filters/breakout_quality/<filter_id>/pretraining/ts2vec_v1/stride_<N>/`，內含`windows.npy`、`window_index.csv`與summary；pretrained encoder位於正式architecture/profile模型目錄下的`pretraining/`；workflow與直接pretrain CLI都必須把同一個supervised `experiment_profile`傳入encoder path與manifest，禁止回退到policy預設而寫錯profile目錄。Dataset summary鎖定Selection日期上限、stride、300×10 schema、outer-policy fingerprint與CSV inventory；encoder manifest鎖定dataset fingerprint、encoder hash與`ts2vec_selection_only`命名pretraining profile完整payload，並要求`oos_windows_used=false`與`pass_reject_labels_used=false`；profile payload與active config不一致時下游訓練及runtime artifact validation均fail-fast。正式下游checkpoint保存完整encoder＋head，但optimizer只接收642個linear-head參數，831,168個encoder參數保持凍結且eval。
- 固定策略對照輸出位於 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/strategy_compare/`；主檔為 `strategy_comparison.md`／`.json`，並保留兩組 equity、trades、daily capacity、年度報酬，以及 `trade_attribution.md/.json`、交易配對、年度R歸因與兩組round-trip CSV。年度完整性必須同時通過既有市場邊界檢查且涵蓋一月至十二月；被回測終點截短的2026年不得標為完整年度。此目錄只存研究比較結果，不是正式 runtime score 來源。
- Dataset 與 future-path cache 固定共用 `outputs/filters/breakout_quality/<filter_id>/`；正式模型工件依架構與訓練實驗分開存放於 `models/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/`。`BREAKOUT_QUALITY_MODEL_ARCHITECTURE` 與 `BREAKOUT_QUALITY_EXPERIMENT_PROFILE` 都不進入 Dataset fingerprint，也不觸發 feature bank 重建；舊 v1 baseline 無 profile 子目錄的工件僅提供唯讀相容 fallback；不得在該舊路徑更新 manifest 或寫入正式 forward-OOS scores，所有新訓練與正式輸出一律寫入 profile 子目錄。`split_assignments.csv` 的 outer `selection/oos` 日期直接來自 `core.walk_forward_policy`。
- inner validation 為可配置模式：關閉時完整 Selection 固定 epochs；開啟時只在 Selection 內選 epoch，正式模型再依 policy 選擇直接採用 best checkpoint 或以 eligible Selection 重訓。Final refit 忠實採用 `BREAKOUT_QUALITY_FINAL_REFIT_MODE`；目前 active 9A-BN與比較基準8F均採 `selected_epochs`；9B ModernTCN歷史實驗亦採相同模式；matched optimizer steps 與 best inner checkpoint 分別由 8I、8J 淘汰。manifest 對 8J 明確保存 best checkpoint source、Final Refit `performed=false` 與 steps=0；對 refit 模式則保存 selected/target/actual steps、等效 epochs、class/time weight mode 與權重摘要。OOS 永不參與 epoch、threshold 或模型選擇。
- 訓練效能優化以結果一致為硬性契約：同一 experiment profile 內每個 epoch 的 sampling units、shuffle、batch size、seed、optimizer 更新次數與訓練單執行緒設定均不得漂移；只有命名 profile 可明確改變 sampling unit。8F／8K 的 batch size 以 unique `ticker/date` groups 計，代表列固定為最小原始 event row index，且不得改變 Validation／OOS 的完整 rows口徑；active 9A及8F比較基準的early-stopping patience均為1、time weight均為none；9B ModernTCN歷史實驗亦使用相同設定。8K date-balanced profile已淘汰，只保留歷史契約。訓練使用 `zero_grad(set_to_none=True)`；optimizer、LR schedule 與 augmentation 由 experiment profile 決定；augmentation 僅允許作用於 training input，不可污染 Validation／OOS，通用 learning rate、weight decay 與 gradient clipping 仍由 policy 控制；模型每個參數在每個 batch 都有 gradient，可省去清零寫入而不改 optimizer 更新；可選擇將去重 feature bank 與小型事件陣列預載至 RAM，但資料值與列順序不變。完整 Train／Validation／Selection 評估共用 `filters/breakout_quality/inference.py` 的 strict-result event-row inference：維持相同 batch boundaries、全部 requested rows 與原輸出列序，只把彼此獨立的 batches 分派給 `BREAKOUT_QUALITY_EVALUATION_WORKERS` 個單執行緒 model replicas，再依原列序與原 reduction 順序彙總。score export 若 model spec 為 `use_dataset_context=false`，則依 canonical `event_group_index` 對每個 unique ticker/date feature group 只推論一次、先計算一次 probability，再精確 broadcast 至全部 event rows，確保同 group 分數 bit-identical；只有使用 Dataset context 的 legacy model 保留逐 event-row inference。開啟 `BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION` 時，Inner Train 與 Validation 使用相同 epoch model 的獨立 read-only 快照同時評估，不改 optimizer、random state 或 best-epoch 判定。Final refit 模式最後一輪已產生的完整 Selection metrics 直接作為 final metrics；8J 則在恢復 best checkpoint 後對完整 Selection 做一次完整 rows、既有 `1/group_size` 的 group-weighted 評估，並明確標記為 evaluation 而非 refit。GPU 與多執行緒 training 不屬於 strict-result 模式，因其浮點 reduction 可能改變模型權重與 threshold 邊界判定。
- Dataset 工件只放 `outputs/filters/breakout_quality/<filter_id>/`；research scores 與 report 依架構及實驗放在 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/`，報表固定於該目錄下的 `reports/evaluation_report.md` 與 `reports/evaluation_metrics.json`，不得覆蓋正式 `scores.csv`。
- `dl_quality_score >= active breakout_quality_score_threshold` 是唯一通過判斷；manifest 只宣告契約與 OOS 可用日期，不預先固化另一份 `dl_pass`。
- Sequence-only（`use_dataset_context=false`）正式 score export 已宣告 `shared_group_score_broadcast=true`；runtime 因此以 `ticker/date` 作為唯一 score lookup key，`high_len` 只用於驗證 artifact coverage。`score_store` 載入時必須確認同一 `ticker/date` 的所有 event-row scores 完全一致，再去重建唯一 date-level index；使用 Dataset context 的 legacy model 仍維持使用 `ticker/date/high_len` 精確 key。
- 正式 runtime 僅在 manifest 宣告的有效期間套用模型；有效期後只要出現未覆蓋候選事件即 fail-fast。

### `tools/validate/`

### 11A Strategy-aligned Continuous Target audit

11A第一階段是獨立research-target子系統，不是新的model architecture。`filters/breakout_quality/continuous_target.py`以既有canonical event anchor與future high／low path cache建立group-level `strategy_aligned_opportunity_r_v1`；`tools/filters/breakout_quality/audit_continuous_target.py`負責固定split分布、同日排序可學性及可選Round-trip R方向診斷，正式入口為：

```bash
python apps/breakout_quality.py audit-continuous-target
```

target固定使用40-bar horizon與10% risk budget：首次風險觸發前最大有利漲幅R，扣除到達高點前最大不利跌幅R及最多0.5R的時間懲罰。同日High／Low歧義採adverse-first，風險觸發日High不計；首日觸發輸出−1R。公式不讀取split統計、OOS、模型score或actual trade R，不做normalization／clipping。實際R診斷在active 9A正式模型輸出樹依語意優先序搜尋hard-filter與score-ranking的`strategy_compare*`工件；有metadata時只接受目前filter／architecture／profile及`historical_active_param_oos`結果。每個目錄優先讀`no_filter_round_trips.csv`，若只有交易歷史則重用`trade_attribution.reconstruct_round_trips`從`no_filter_trades.csv`在記憶體重建；亦可用`--round-trips`或`--trade-history`明確指定，禁止另寫第二套Round-trip口徑。

工件位於`outputs/filters/breakout_quality/<filter_id>/continuous_targets/strategy_aligned_opportunity_r_v1/`，與architecture／experiment profile工件隔離；只沿用feature-group index及future-path cache，不改Dataset fingerprint、不重建feature bank、不relabel。manifest明確保存`training_performed=false`與`runtime_eligible=false`。此階段只產生arrays與audit，不授權regression training、score export或scanner runtime。


- `tools/validate/`：正式 invariant、contract、schema 與 real-case 驗證子系統；正式細目與狀態以 `doc/TEST_SUITE_CHECKLIST.md` 為準。

### `tools/local_regression/`

```text
├── formal_pipeline.py
├── meta_quality_coverage.py
├── meta_quality_targets.py
├── run_meta_quality.py
```

- `tools/local_regression/`：reduced formal orchestrator；`formal_pipeline.py` 為正式步驟單一真理來源。
- `run_all.py` 在每次 formal staging run 以目前 `config/` defaults 生成隔離的 `formal_primary_params.json`，並透過 `V16_RUN_BEST_PARAMS_PATH` runtime override 傳給 dataset prep、consistency、chain、quick gate、ML smoke 與 meta quality；formal regression 不依賴、也不覆寫 `models/run_best_params.json`，且不要求 repository 或交付 ZIP 內建 `models/run_best_params.json`。
- `run_meta_quality.py`：meta quality 工具；負責 coverage / summary / baseline 與 formal step 對照。

## 子系統責任

- `apps/test_suite.py` 是日常唯一建議使用的一鍵測試入口。

- `apps/`：正式入口層，只從對應子系統 façade 匯入公開介面。
- `core/`：核心規則、帳務、價格、統計、path 與共用 helper；不得放 UI orchestration 或 validate 腳本。
- `tools/`：下載、最佳化、單股分析、validate、local regression 與 GUI 子系統；workbench 的交易明細與 Console 改以獨立分頁承接。
- `config/`：共用政策與執行預設。
- `models/`：模型工件與 runtime 產生或使用者保留的可選最佳參數輸入；沒有 path override 時，預設參數 fallback 仍解析到 `models/run_best_params.json`，但 repository／交付 ZIP 不必預先包含該可變動工件。
- `doc/`：架構、常用指令與 formal checklist 文件。

## 正式入口

- `apps/breakout_quality.py`：Breakout quality 互動選單、完整 research workflow、dataset、training、score export、易讀 report 與詳細 evaluation 單一正式入口。
- `apps/breakout_quality_strategy_compare.py`：固定參數 no-filter 與 active breakout-quality filter 的策略經濟效果隔離比較正式入口；只允許 `use_breakout_quality_filter` 不同。
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

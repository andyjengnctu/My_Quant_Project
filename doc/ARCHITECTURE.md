# 架構概覽

本文件只保留穩定分層、正式入口、依賴方向與共享邊界。操作步驟看 `doc/CMD.md`；formal contract 與狀態看 `doc/TEST_SUITE_CHECKLIST.md`。

## 穩定檔案樹

```text
project/
├─ apps/
│  ├─ research.py                     # 研究單一正式入口：模型訓練／策略參數最佳化／策略組合比較／Audit
│  ├─ portfolio_sim.py                # 投組模擬正式入口（薄入口）
│  ├─ smart_downloader.py             # 資料下載正式入口（薄入口）
│  ├─ run_bundle.py                   # 本機 double check／commit／package／formal test 單一使用者入口
│  ├─ package_zip.py                  # run_bundle 使用的專案打包 helper；亦保留直接 snapshot 用途
│  ├─ test_suite.py                   # run_bundle 內部 formal test runner
│  ├─ vip_scanner.py                  # 掃描器正式入口（薄入口）
│  └─ workbench.py                    # GUI 工作台正式入口（薄入口）
├─ config/
│  ├─ breakout_policy.py              # breakout 策略預設與 optimizer high_len 範圍
│  ├─ research.py                     # active model與model provider設定
│  ├─ breakout_quality.py             # 唯一可編輯設定：模型／Label／training／profiles／PIT／策略workflow
│  ├─ audit.py                        # active Audit module與各Audit對象／來源／維度／輸出政策
│  ├─ training_policy.py              # 訓練政策與 selection gate
│  ├─ display_policy.py               # console/report 顯示政策
│  └─ execution_policy.py             # 資金、費用與 runtime 執行預設
├─ core/
│  ├─ config.py                       # 相容 façade；穩定匯出設定常數與參數契約
│  ├─ strategy_params.py              # breakout + training gate + execution 聚合參數契約
│  ├─ capital_policy.py               # 單股/投組/scanner 共用資金與 sizing 規則
│  ├─ exact_accounting.py             # 正式整數 ledger / cost-basis allocation / tick 正規化單一真理來源
│  ├─ backtest_core.py                # 單股回測總控 façade
│  ├─ portfolio_engine.py             # 投組 timeline 總控 façade；只保留run_portfolio_timeline orchestration
│  ├─ portfolio_benchmark.py          # benchmark period統計與bounded cache
│  ├─ portfolio_replay_support.py     # replay phase錯誤上下文與candidate diagnostic snapshot
│  ├─ portfolio_levels.py             # held／shadow active-level rows
│  ├─ portfolio_ensemble.py           # seed-ensemble candidate aggregation與signal orchestration
│  ├─ portfolio_entry_plans.py        # reserved/cash-capped entry plan primitives
│  ├─ portfolio_entry_selection.py    # resource-aware selector thin router
│  ├─ portfolio_entry_selection_common.py # Binary／Continuous共用resource-aware search
│  ├─ portfolio_entry_selection_max_dl.py # Max-DL／feasible-ascent search
│  ├─ portfolio_entries.py            # reserved order fill／missed-buy／extended-signal cleanup state transition
│  ├─ console_report.py               # 全專案簡易console／相對工件路徑格式SSOT
│  ├─ file_integrity.py               # file／canonical-JSON hash SSOT
│  ├─ path_utils.py                   # 跨平台path判定／split SSOT
│  ├─ serialization_utils.py          # 共用output text／JSON-native serialization SSOT
│  ├─ runtime_utils.py                # 共用runtime／environment flag helper
│  ├─ model_paths.py                  # models 目錄與預設參數來源解析
│  ├─ output_paths.py                 # outputs/<category> 目錄正規化與建立 helper
│  ├─ output_retention.py         # outputs retention 雙門檻清理 helper
│  └─ display.py                      # 顯示 façade
├─ doc/
│  ├─ TEST_SUITE_CHECKLIST.md         # formal test suite 主表與索引
│  ├─ ARCHITECTURE.md                 # 本檔
│  └─ CMD.md                          # 常用指令與操作說明
├─ filters/
│  └─ breakout_quality/               # quality feature/label、shared split、artifact contract、model factory、正式 runtime score lookup
├─ services/
│  ├─ portfolio_replay.py             # canonical portfolio replay／market-context application service
│  └─ optimizer/                      # 正式optimizer primitives；目前含raw cache／trial inputs／walk-forward
├─ models/
│  ├─ filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/
│  │  ├─ model.pt                     # architecture/profile-scoped canonical model artifact
│  │  ├─ split_assignments.csv        # outer Selection/OOS + Selection train/embargo assignment
│  │  ├─ manifest.json                # model spec/profile/split/score/OOS eligibility 契約
│  │  ├─ scores.csv                   # architecture/profile-scoped canonical event score table
│  │  └─ point_in_time/               # Selection rolling/cross-fitted Score與fold工件
│  └─ <optimizer parameter artifacts>.json # runtime 產生或使用者保留的可選參數工件；檔名依 optimizer mode／selector 而定
└─ tools/
   ├─ audit/                          # 全專案Audit catalog／runner／domain implementations
   ├─ downloader/                     # 資料下載子系統
   ├─ filters/breakout_quality/        # quality dataset/train/export/evaluate 子系統實作與開發研究入口
   ├─ optimizer/                      # optimizer互動／orchestration與尚待搬遷相容層
   ├─ portfolio_sim/                  # 投組模擬CLI／報表與legacy import相容層
   ├─ scanner/                        # 掃描器子系統
   ├─ trade_analysis/                 # 單股 trade-analysis 子系統
   ├─ validate/                       # validate / synthetic / real-case 驗證子系統
   ├─ local_regression/               # reduced formal orchestrator
   └─ workbench_ui/                   # GUI 子系統
```

## 關鍵 shipped 模組索引

### Portfolio Core

- `core/portfolio_engine.py`只保留`run_portfolio_timeline()`日序 orchestration；benchmark cache、replay diagnostics、active-level rows與seed-ensemble aggregation分別由`portfolio_benchmark.py`、`portfolio_replay_support.py`、`portfolio_levels.py`、`portfolio_ensemble.py`持有canonical implementation。engine只以同function-object import保留歷史private helper相容名稱，不保存第二套實作。
- Entry path採三層責任：`portfolio_entry_plans.py`只建立reserved／cash-capped pre-market entry plan；`portfolio_entry_selection.py`是resource-aware policy thin router，Binary／Continuous共用搜尋在`portfolio_entry_selection_common.py`，Max-DL／feasible-ascent在`portfolio_entry_selection_max_dl.py`；`portfolio_entries.py`只保留reserved order fill、missed-buy記錄與extended-signal cleanup state transition。selection不得反向import`portfolio_entries.py`，所有實際成交仍由同一`execute_pre_market_entry_plan()`與exact-accounting執行。
- Portfolio Core split是infrastructure refactor，不改candidate validity、盤前reservation、fill、fee/tax、TP/SL、rotation、Round-Trip、resource-aware K/R0／stale-score semantics或Strategy Compare identity。Formal impact registry與coverage targets必須指向canonical split modules，compatibility façade coverage不得取代implementation coverage。

### `services/portfolio_replay.py` 與 `services/optimizer/`

- `services/portfolio_replay.py`是canonical portfolio replay application service；Strategy Compare、portfolio CLI與Workbench均直接引用此正式服務。舊`tools/portfolio_sim/simulation_runner.py`只保留module alias相容層，不保存第二套replay實作。
- `services/optimizer/raw_cache.py`、`trial_inputs.py`、`walk_forward.py`是Portfolio Replay共用的正式optimizer primitives；無current consumer的`tools/optimizer/` compatibility aliases已移除，正式validator直接import service owner。仍被目前CLI/orchestration實際使用的tools modules才保留，不以「可能相容」為理由永久保留空alias。

### `tools/optimizer/`

```text
   │  ├─ runtime.py                   # optimizer 執行期狀態、匯出控制與歷史最佳還原
   │  ├─ session.py                   # optimizer session 狀態 façade
```

- `tools/optimizer/`目前保留策略參數最佳化互動／orchestration與legacy import façade；由 `apps/research.py` 的「策略參數最佳化」工作類型進入。正式可共用primitive逐步移入`services/optimizer/`，不得再由`filters/`新增對`tools.optimizer`的反向依賴。

### `tools/trade_analysis/`

```text
   │  ├─ history_snapshot.py          # 單股分析歷史績效 snapshot / payoff / asset-growth helper
```

- `tools/trade_analysis/`：單股 trade-analysis 子系統；由 `apps/workbench.py` 經 `tools/workbench_ui/` 觸發，`tools/trade_analysis/trade_log.py` 提供共用 backend / 開發輔助 CLI。
- 為維持相容性，保留 legacy `run_debug_*` API 名稱，同時提供 canonical `run_trade_analysis` / `run_trade_backtest` / `run_prepared_trade_backtest` / `run_ticker_analysis` aliases。

### `apps/research.py`、`tools/filters/breakout_quality/application.py` 與 Breakout Quality domain

- `apps/research.py`是研究單一正式入口；Audit工作類型由`config/audit.py`的active module指定，`tools/audit/catalog.py`只登記**目前仍支援**的Audit／research utility，`tools/audit/runner.py`依兩者派送。Formal handler必須`mode=formal`且`read_only=true`，缺工件即BLOCKED，不得重跑策略、建立Label、訓練模型或修改runtime。已完成且不再被runtime／active Audit／必要compatibility引用的一次性Audit，不以`enabled=False`或historical CLI永久保留；研究結論留在Experiment Registry／Log，implementation、catalog registration與專屬synthetic/helper一併退役。`core/`與`filters/`不得反向import `tools/audit/`；runtime與Audit若共用計算，純計算真理必須留在正式domain／core。`tools/local_regression/run_meta_quality.py`另執行advisory slimming scan，主動列出disabled formal Audit、無current reachability的Audit／compatibility模組與過大的Audit-specific synthetic tests；只回報`CLEAN／REVIEW`，不單獨造成formal FAIL。

- Candidate生命週期採單一責任契約：**Strategy owns validity / DL owns quality / Portfolio selector owns allocation**。只有原策略可決定candidate建立、continuation、Re-entry與失效；DL PASS／REJECT只代表quality，不得刪除仍屬策略VALID的candidate；selector只在既有合法candidate pool中依當下資源配置。Audit可用Future Label／MFE／MAE／Realized R做事後診斷，但不得回流當日runtime或產生第二套candidate-invalid語意。
- `filters/breakout_quality/trade_path_label.py` 與 `tools/filters/breakout_quality/build_trade_path_labels.py` 組成A2 realized trade-path Label鏈。Label identity與9A MFE／MAE契約隔離；Builder hardlink／copy既有300×10 feature bank，只重建event labels與events metadata。Teacher params由2014～2020 Selection Min ROOS no-DL schedule與2021～2026既有P2 schedule合併，每日只解析當時已生效參數。模擬器直接重用`generate_signals`、normal／extended pre-market entry plan、shadow continuation cleanup、`execute_bar_step`及exact-accounting；初次miss buy只維持pending，同一原始event後續成交只產生一個終局Label，已成交淨Realized R正值為PASS、非正為REJECT；永未成交、shadow終止、新setup覆蓋或資料結尾仍未成交均為EXCLUDED並排除訓練。已成交但資料結尾仍持倉時，直接重用單股正式最後交易日強制結算，形成PASS或REJECT。`train.py`、Binary PIT builder與artifact validator依filter_id解析Label policy，禁止新模型與9A Dataset／manifest混接。
- Realized trade-path Label／模型流程與策略比較維持分離：Label/model/forward-score由正式model workflow產生，策略經濟比較由config-driven Strategy Compare執行。歷史Old/New專用Trade-path Gate已於Legacy Cleanup Batch 7退役；既有結果只讀保留於Registry／Experiment Log。
- `filters/breakout_quality/` 承接 feature/label、indexed feature-bank dataset storage、source-data inventory fingerprint、artifact contract、canonical path、外層 Selection/OOS split 與正式 runtime lookup。模型架構與訓練實驗分離：目前policy已退回9A `inception_time_v1`排序／高品質基準，8F `multiscale_cnn_sequence_only_v1`保留高覆蓋基準；10A `inception_time_market_set_candidate_v1`完整OOS排序低於9A，已轉為legacy read-only。`inception_time_market_set_v1`修正logical-batch契約後完整OOS仍低於9A，已轉為legacy read-only。9F `patch_transformer_v1`完整OOS排序低於9A，已轉為legacy read-only；其300×10 sequence依10 bars切成30個非重疊patch、128維embedding、3層／4-head Transformer、MLP 256、sinusoidal position與patch mean pooling規格只供舊checkpoint／manifest重建。9E `moment_1_base_frozen_linear_v1`完整OOS排序低於9A與8F，亦為legacy read-only。9E釘死官方`AutonLab/MOMENT-1-base`revision與checkpoint SHA256，runtime固定為`momentfm==0.1.4 / transformers==5.5.0`且以`--no-deps`隔離安裝避免降級主環境，300×10輸入固定插值至512，使用embedding mode保留10個channel，各channel對patch取mean後串接為7680維，只訓練linear head，不使用Dataset context、project pretraining、OOS或PASS／REJECT labels訓練encoder。9D `mantis_v2_frozen_linear_v1`完整OOS固定coverage排序低於9A，已轉為legacy read-only。9D釘死官方Hugging Face repository、commit revision、config與checkpoint SHA256；300×10輸入逐channel固定插值至512，以官方建議的第3層（index 2）combined token輸出，各channel獨立編碼後串接，只訓練linear head且不建立project pretraining dataset。9C `ts2vec_frozen_linear_v1`以Selection-only未標記rolling windows做TS2Vec-style hierarchical contrastive pretraining後凍結encoder，但完整OOS排序明顯低於9A，已轉為legacy read-only。9B `modern_tcn_v1`使用6個96-channel residual blocks、kernel 51 depthwise temporal convolution、4× pointwise expansion與global-average pooling；雖與9A容量近似，但完整OOS固定coverage排序、Accuracy與校準全面惡化，已轉為legacy read-only。9A-GN `inception_time_group_norm_v1`只把6個Inception module與2個residual projection的`BatchNorm1d(128)`改為`GroupNorm(8, 128)`，但完整OOS固定coverage排序顯著下降，已轉為legacy read-only。`inception_time_v1`維持排序／高品質實證基準；`multiscale_cnn_sequence_only_v1` 保留為高覆蓋實證基準；它保留 `multiscale_cnn_v1` 的 300×10 Level sequence、三分支 CNN、pooling、dropout 與 32 維 head，只讓 head 不再拼接 Dataset 既有的 4 維 event context，因此 trainable parameters 僅減少 128，Dataset storage contract、feature bank、context arrays 與 labels 都不需重建。8P `multiscale_cnn_sequence_only_dual_path_v1` 的完整OOS只讓Precision增加0.06 pp，卻降低Recall、Accuracy與Score，已轉為legacy read-only；`multiscale_cnn_v1` 則保留為使用原4維event context的歷史比較架構。`multiscale_cnn_sequence_only_dual_path_v1`、`multiscale_cnn_regime_context_v1`、`multiscale_cnn_v2～v8`、`tiny_cnn_v1` 與 `residual_tcn_v1` 保留為 legacy architecture，只供舊 checkpoint／manifest 重建與歷史重現。AdamW、LR schedule、augmentation、time weighting 與 sampling unit 等supervised訓練方法，以及9C TS2Vec的optimizer／epochs／batch／LR／crop／mask／contrastive loss設定，均由 `config/breakout_quality.py` 的命名 profile 管理；8B `recent_decay_60m` 已被完整 OOS 淘汰。8F 使用 `unique_group_sampling / batch_size=128 / time_weight=none / patience=1`，只將 sequence-only training unit 改為 unique `ticker/date` group；完整 OOS 已證明 Recall、Accuracy、Score 與 drift 大幅改善，因此 8F 保留為高覆蓋研究基準。9A InceptionTime 在固定coverage下的Precision跨Selection／OOS幾乎不下降，升為排序／高品質模型基準。Active `inception_time_v1` 的 depth、目標 receptive field 與 residual interval 由 `config/breakout_quality.py` 單一設定；kernel sizes 依目標視野自動生成為三個近似 1x／1/2x／1/4x 的正奇數尺度，實際 receptive field 與 kernels 寫入 model manifest。預設 minimum target 228 bars 會精確還原 9A 的 depth 6、kernels 39／19／9與實際 receptive field 229 bars；legacy `inception_time_group_norm_v1` 維持原始固定結構，不受 active 設定影響。8G patience 5 已淘汰，因其重新造成 Selection 過擬合與 OOS score drift；8H batch size 64 使 optimizer updates 加倍但 OOS Precision、Accuracy 與 Score 均低於 8F；8I matched optimizer steps 雖提高 Precision，但 Recall、Accuracy、Score與泛化落差低於 8F，因此 8F 實證基準維持 batch 128、patience 1、`selected_epochs`。8J `unique_group_best_inner_checkpoint` 已完成但被 OOS 淘汰：Precision幾乎不變，Recall、Accuracy 與 Score明顯低於 8F；該 profile只供歷史重現。8K `unique_group_date_balanced` 已由完整 OOS淘汰：threshold 0.5下幾乎全部判PASS，Precision Lift只剩+0.45 pp；該profile只供歷史重現。現有multiscale CNN細調、supervised ModernTCN、三個frozen probe（TS2Vec、MantisV2、MOMENT）、9F supervised Patch Transformer與10A Candidate-conditioned Query均停止；9A與8F分別保留排序／高品質及高覆蓋比較基準。Stage 1 `inception_time_market_set_v1`與9A-GN、9B、9C、9D、9E、9F完整OOS均已淘汰，只供舊checkpoint／manifest重建。現有300×10單模型architecture與Market Set橫向搜尋停止；下一步改做11A連續策略對齊target的資料與可學性稽核，不先調整模型容量、threshold或Market Query。Patch Transformer、ModernTCN與InceptionTime訓練均支援`device=auto/cpu/cuda`、CUDA mixed precision（auto優先BF16，否則FP16）、deterministic algorithms與TF32顯式契約；checkpoint/manifest保存training execution，score export可使用獨立推論device。checkpoint、manifest 與 canonical path 同時釘死 architecture 及 experiment profile，model spec 另以 `use_dataset_context=false` 明確記錄 sequence-only 契約；正式 runtime 不執行 CNN，只讀目前 policy architecture/profile 的 canonical `scores.csv`。

- Legacy `inception_time_market_set_v1` 曾在9A候選分支之外增加Stage 0／1全市場表示：Dataset以benchmark交易日為calendar，保存point-in-time `date × ticker × 5`基礎OHLCV變化與valid mask，同日breakout共用同一market date index；Shared Stock Temporal Encoder（GroupNorm，避免不同market batch／無效股票比例污染其他股票的正規化統計）以同一套權重把每檔股票300日壓成32維表示，4個Global Learned Queries以排列不變attention pooling形成128維市場表示，再與9A候選128維表示融合。第一版不含ticker identity、candidate-conditioned query、sector token、learned lag或股票兩兩self-attention；實體batch依market date分塊並限制不同日期數，避免重複展開全市場tensor。此architecture已由修正後完整OOS淘汰，不再允許正式新訓練；舊research checkpoint／manifest仍可嚴格重建，`forward_oos`與scanner runtime維持fail-fast。

- Active `inception_time_market_set_candidate_v1` 沿用既有point-in-time Market Set Bank與Shared Stock Temporal Encoder，但不再使用與候選無關的Global Learned Queries。每個候選128維embedding會投影成可設定數量的candidate-conditioned query（目前1個），對該事件日期的全市場32維stock embeddings做masked multi-head cross-attention，再形成128維candidate-specific market embedding與候選表示融合。Market Bank仍依日期microbatch物化，候選encoder仍對完整logical batch只forward一次；因此`max_dates_per_batch`只影響記憶體，不改batch=128、optimizer step、loss denominator或epoch語意。此架構目前只允許research score export，`forward_oos`與scanner仍fail-fast。
- 正式 `forward_oos` score export 不得只重播訓練 Dataset 內「成功建立特徵」的事件；它必須重新使用目前 canonical OHLCV 清洗與突破 crossover 規則建立當前 runtime 候選全集。每一個 `ticker/date/high_len` 候選必須二擇一：可建立完整模型輸入者寫入模型 probability；因 benchmark 日期缺失、歷史窗不足或非有限特徵而不可評分者，明確寫入同目錄 `unavailable_scores.csv`，並在 canonical `scores.csv` 以固定 `0.0` 保守映射為 REJECT。manifest 必須保存 current source CSV inventory、候選總數、模型評分數、保守拒絕數及原因統計；runtime 必須驗證 audit rows 與 `scores.csv` 的 0.0 一致。只有已被正式記錄的不可評分事件可保守拒絕，任何未記錄缺分仍須 fail-fast。
- Continuous／Daily ranker正式CLI dispatch由`services/breakout_quality/ranker_cli.py`單向解析profile並呼叫對應trainer；`train_continuous_ranker.py`只承接event-group ranker，`train_daily_ranker.py`只承接daily-universal ranker，兩者不得互相import。共用訓練primitive仍只由`services/breakout_quality/ranker_training.py`與`continuous_ranker_pipeline.py`提供，避免形成trainer↔trainer↔shared API循環。正式`tools/filters/breakout_quality/application.py`的`train-continuous-ranker` command固定指向dispatcher，而不是依靠tools compatibility alias或跨trainer lazy import。
- Selection point-in-time Score 子系統由 `services/breakout_quality/continuous_ranker_pipeline.py`、`point_in_time_scores.py` 與 `point_in_time_audit.py` 組成。Continuous／Daily／PIT／multi-seed共用 `services/breakout_quality/ranker_training.py` public API 取得percentile target、Validation epoch selection、final refit、inference與rank metrics；consumer不得跨模組呼叫trainer `_private` helper或注入整個trainer module，因此loss／epoch-selection／refit／inference語意維持單一實作。Builder 使用 expanding-window folds；train／validation／refit 資料除事件日期早於 score period 外，還必須滿足 `label_eval_end_date < score_start`，每個 score group只能由一個尚未看過該事件的凍結模型評分。PIT起始日預設為`auto`：依實際Dataset、Target、label completion、24個月Validation與最小train／validation／score groups契約逐月尋找最早合法日期；`--plan-only`可先輸出解析日期與fold計畫。Fold identity固定為`fold_YYYYMMDD_YYYYMMDD`，向前延伸歷史不會改變既有期間的ID；若舊`fold_000`類工件的日期、模型、資料、訓練與來源契約完全相同，builder會驗證checkpoint／Score hash後遷移為穩定日期ID並直接重用。每 fold 保存日期、rows/groups、selected epoch、checkpoint hash及Score hash；串接後 fail-fast 檢查 coverage、重複、缺失、cutoff、identity與有限值。正式串接 Score CSV不含 Future Target，初始 manifest 明確 `eligible=false`、只可做Selection模型驗證；audit才離線 join continuous target計算Spearman、daily Spearman、年度與decile spread、fold drift、PASS分類重疊及orderable coverage。Audit以單一payload同時產生表格化終端摘要、`selection_point_in_time_audit.md`易讀報表與完整JSON；JSON以SHA256綁定產生它的PIT manifest、Scores、coverage與Continuous Target manifest，策略gate只接受完全相同來源工件；報表固定揭露設定、Score coverage、逐年與逐fold結果、研究邊界及工件路徑，不另算第二套指標。主選單狀態頁另列Target與PIT Markdown報表。PIT工件不是 forward-OOS runtime score，不得自動進入scanner或策略排序。
- `apps/research.py`的「策略組合比較」是正式策略比較入口；第一層選單由`config/strategy_compare.py`的profiles動態產生，永久區分`Selection PIT 策略比較`與`Forward-OOS 策略比較`，另提供跨profile狀態檢視；每個profile內才提供「執行目前比較設定／查看設定、工件與預計動作」。選單不得硬編特定arm ID、TP1、A9或其他實驗版本名稱。Selection與Forward共用同一engine，但各自有獨立period、enabled arms/contrasts與`outputs/strategy_compare/<profile>/`命名空間，禁止靠覆寫單一active matrix切換研究階段；profile可由config宣告`reuse_output_roots`唯讀掃描舊run cache以平滑遷移，任何新run／latest仍只能寫入目前profile root。`config/strategy_compare.py`只保存current profile dependency closure所需parameter sources、DL sources、arms、contrasts與preparation policy；arm／contrast是否啟用只由各profile的`arm_ids`／`contrast_ids` membership決定，歷史唯讀定義隔離於`config/compatibility/strategy_compare_history.py`。同一param source／rule policy只定義一個共用DL-off基準，可掛一個或多個DL-on模型；各DL-on arm可獨立開關，runtime必須逐一與同一DL-off基準形成controlled pair，並驗證重複回放的基準摘要與年度報酬完全一致。`core/strategy_comparison.py`只提供泛用設定、驗證、依賴計畫與fingerprint；`filters/breakout_quality/strategy_compare_preparation_status.py`是artifact readiness／period／parameter coverage與`StrategyPreparationPlan`建構的單一owner；`filters/breakout_quality/strategy_compare_preparation.py`只保留builder執行與dependency-wave runner façade，並透過`filters/breakout_quality/export_scores.py`、`strategy_optimizer_policy.py`與`strategy_param_training.py`正式共用服務補建既有模型的forward-OOS scores或比較所需策略參數；不建立Label、不選模型、不訓練模型權重。`filters/breakout_quality/strategy_comparison.py`完成一次確認後的前置編排；Strategy Compare runtime依責任拆分為`strategy_compare_engine.py`的pair orchestration、`strategy_compare_contracts.py`的mode/schema contract、`strategy_compare_sources.py`的參數來源與controlled-pair resolution、`strategy_compare_replay.py`的canonical scenario replay／standalone baseline cache、`strategy_compare_diagnostics.py`的replay後selection diagnostics、`strategy_compare_reporting.py`的pair summary/yearly/readable-report renderer。Strategy Compare的人讀aggregate輸出由`strategy_comparison.py`依序組裝：(1)核心策略結果，固定依`core/report_metrics.py`的`CORE_STRATEGY_RESULT_METRICS`順序輸出`報酬 → MDD → RoMD → 年化 → 最差完整年度 → Log R² → 月勝率 → 勝率 → Payoff → EV → 交易數 → 平均曝險`；(2)R預測／轉化，固定合併成單一「arm放列、指標放欄」表，表頭第一層用三個分群超欄：實際交易=`平均R／中位R／Coverage／DL選擇R`、模型預測=`Dailyρ／Globalρ／Pair一致／Top-R／Bottom-R／Top-BottomR`、選股轉換=`RCE／Target mean R／Target %ile／Top-K／Opp gap`；`RCE=Target-covered exclusive completed-trade realized mean-R edge / Future Target mean-R edge`，兩個edge使用完全相同的canonical closed-trade exclusive且Target可觀測subset；Future Target只由既有post-replay selected-target sidecar join，少量Target缺值只從兩個edge共同排除並保留coverage diagnostics，不要求100% coverage、不設最低coverage magic threshold；REUSE舊pair時只讀既有trade/target sidecar backfill、不觸發market replay；`DL選擇R`則保留為exclusive total realized-R attribution，不與RCE混用；Top/Bottom-R直接讀既有模型驗證工件的最高／最低Score十分位Target R，不另算第二套；`定義`與`理想方向`移到表格下方註解，欄位顏色採每個metric在同表arm之間「綠＝最佳、紅＝最差、白＝其餘」；(3)跨allocator共同的精簡資金／執行；(4)年度結果。Strategy Compare aggregate四區全部使用同一best/worst色彩契約；除年度結果外，每張表前兩欄固定為`編號／比較對象`，年度結果維持`年度`放列、arm放欄；沒有通用higher/lower方向的metric維持白色。`core/report_metrics.py`是label／單位／小數位／方向性的共用metric registry，`core/report_style.py`是綠／紅／黃／灰判讀語意與Console ANSI／Markdown inline-HTML文字色SSOT；所有判讀只對文字本身上色，禁止交通燈／圓點emoji marker；pair、aggregate、PIT audit、模型報表與optimizer／strategy dashboard不得各自重定義同名metric或顏色方向；domain-specific threshold可保留在其正式policy，但threshold判定後的positive／negative／warning／neutral顏色映射必須回到共用style。`strategy_compare_diagnostics.py`除保留replay後selection diagnostics owner外，也只從既有PIT／continuous驗證工件及pair canonical payload聚合`strategy_diagnostics.md`與aggregate共用R分析payload，不得從raw market／trade rows重算第二套同名指標。solver states、repair/ascent、stale guard與selector timing等演算法專屬debug只保留於JSON／sidecar。portfolio replay底層仍只透過`services/portfolio_replay.py`取得；上述formal modules不得反向import`tools/portfolio_sim`，且同一責任不得在engine再保留第二套實作。任一前置步驟失敗即停止回放、保留可接續工件並回報步驟與相對路徑。正式回放前須先由全部啟用DL runtime工件解析共同可比較期間，並驗證每個rolling active-param來源完整覆蓋該期間；歷史Label teacher params不得冒充forward績效比較參數。Min ROOS使用forward P2 DL-off-trained工件，Min-DL ROOS使用forward P3 DL-on-trained工件；缺少或identity／coverage不符時依config自動建立或接續。前置允許分波重新規劃，例如先建立forward scores取得正式期間，再建立因此顯露為缺少／過期的參數工件；所有來源READY後才開始第一個pair replay。目前正式profile的比較對象與差異組合完全由`config/strategy_compare.py`的current profile membership驅動；退役TP1/A9及其他historical arms只存在compatibility catalog，不得因歷史定義存在而自動進入current profile。Full DL-off可作standalone comparator，不要求同group另啟用DL-on arm。歷史`trained_with_dl_id`參數仍維持配對限制：任何非空identity只允許搭配訓練時相同DL runtime，舊TP1／A9 P3工件亦繼續隔離保存，不得混入current Min/Full continuous matrix。`direct_selection_delta_r`是相對同一param source／rule policy之DL-off基準的attribution，只能在相同參數與規則宇宙內比較；跨參數或跨rule-policy contrast不得把兩個不同baseline attribution相減後顯示為直接選擇效果。每個profile的輸出以啟用arm IDs與config fingerprint建立`outputs/strategy_compare/<profile>/runs/<timestamp>_<arms>_<fingerprint>/`並更新該profile自己的`latest/`，manifest保存設定snapshot、共同比較期間、前置計畫與工件SHA256。Strategy Compare另以pair-level replay fingerprint重用跨run已完成結果；fingerprint只包含會改變replay的dataset／period／param policy／max positions／rotation／param source／DL source／runtime mode、engine schema與對應param/model/manifest/forward-score SHA256，不包含contrast或報表文字，因此新增比較項不會使既有arm失效。同一run若仍需建立新DL arm，`reuse_shared_baseline`會把同一param source／rule policy的DL-off replay視為shared baseline：歷史compatible pair可直接提供baseline，否則第一個新pair只算一次，後續pair只執行DL-on path；baseline重用前仍須逐項驗證dataset、param SHA、param policy、rules、shared overrides、max positions、rotation與period完全一致。
- Resource-aware quality是candidate ranking的盤前overlay，不是第三套signal filter。所有resource-aware policy都保留完整setup lifecycle與Min ROOS初始buy-sort；`core/portfolio_entry_selection_common.py`／`core/portfolio_entry_selection_max_dl.py`在正式reserve前經`portfolio_entry_selection.py` router重用同一`build_cash_capped_entry_plan()`／exact-accounting建立Min ROOS盤前baseline。C11 `resource-aware-binary`沿Min ROOS順位接受第一個維持cash-binding且提高PASS reserved capital的promotion；C12 `resource-aware-binary-basket`每輪評估全部尚未promotion的A9 PASS候選，以exact cash-cap結果做best-improvement；C14/C15 `resource-aware-continuous`仍只在Min ROOS cash-binding日讓frozen continuous score介入，slot-binding日完全保留Min ROOS。C16 `resource-aware-continuous-capital-preserving`改以basket-level resource floor取代cash-binding feasibility：cash或slot-binding日都可評估MR-12A score，但任何接受basket都必須同時滿足`selected_count >= Min ROOS baseline`與`exact reserved capital >= Min ROOS baseline`；完整score order不合法時只接受符合雙resource floor的deterministic best-improvement promotions。C17 `resource-aware-continuous-max-dl`再把研究變數收斂成stock membership：Min ROOS只固定每日預留單數K與reserved-capital floor R0，DL score先取Top-K；不合法時最多K步minimum-repair，每一步只替換一個原Top-K成員並重跑canonical exact reservation；已選basket內的執行順序仍沿用Min ROOS rank，正式action prefix限制為K筆，因此不把stock selection與allocation priority混成同一變數。C12/C16/C17都刻意不做指數級全子集合窮舉，避免明顯犧牲正式replay效率；所有resource-aware policy都不得新增利用率百分比、距限價bucket、score cutoff、Min ROOS／DL權重、Future Target或candidate-day重新打分。
- Current exact No-R0 research亦由`portfolio_entry_selection_max_dl.py`單一owner承接。SR-C48不把capital乘進candidate score，也不使用baseline R0；在同一K/canonical-cash universe中先exact求最大Score coverage下的Score endpoint與reserved-capital endpoint，再以兩端min-max normalization exact最大化basket-level `Q_norm × C_norm`。三個search pass都重用canonical cash-capped entry simulation與deterministic baseline execution order；branch-and-bound只能用admissible上界做pruning，不得beam/truncate/time-limit，Selection implementation不自動建立Forward、robustness或production runtime。

- Selection Full ROOS使用獨立`PARAM-P4` historical rolling active params：2014～2020、120m train／12m OOS、canonical Full optimizer search space、TP/DL/History threshold依current optimizer policy固定OFF；不得以2021+ `full_roos` forward active params倒灌Selection。Selection Full historical row為C32/C33/C34，P4與三個scientific IDs永久保留；current Selection核心比較只啟用C32 Full DL-off baseline、C23 Min baseline、C25 Min+MR-12B與C28 Min+MR-13A。C33/C34保留歷史重現但inactive；engine支援standalone DL-off comparator，因此C32不必掛隱藏Full+DL arm。
- Selection與Forward-OOS由兩個永久共存的Strategy Compare profiles隔離；current核心比較均縮為四arm：Selection=`C32/C23/C25/C28`，Forward=`C1/C3/C20/C29`。Full只作DL-off完整策略體系baseline，MR-13A改善聚焦Min universe下與MR-12B的純source對照；C33/C34/C30/C31僅保留歷史重現。
- Breakout-quality candidate ranking 是 hard filter 之外的獨立策略機制。`use_breakout_quality_ranking=True` 時不得同時啟用 `use_breakout_quality_filter`；signal generation 不以 threshold 刪除可評分候選，`core/portfolio_candidates.py` 只在候選形成後讀取原始 breakout signal date 的 canonical runtime score。Continuation 與 STOP 後 Re-entry 沿用同一 setup 的原始 breakout Score payload；Re-entry 的重新站回確認日只作新的交易訊號日，不得拿確認日重新查 score table。Ensemble 成交持倉必須逐 member 保存原始 Score payload，STOP 後各 member 的 watch state 與再次形成的 Re-entry 共識都沿用各自原始 `score_date`；Selection PIT或canonical ranking分數缺失時不得刪除候選、不得填0；候選保留並回退原buy-sort。Continuation／Re-entry必須沿用相同score source，source identity不一致時fail-fast。單一參數候選的正式原始排序為 Quality Score 降冪後接既有 buy-sort；active-param ensemble 則先以 `min_agree` 決定資格，再固定依 vote count 降冪、同票候選的 median Quality Score 降冪、既有 buy-sort、ticker deterministic 排序。`legacy strategy-compare --ranking-policy`另提供兩個CLI-only、輸出隔離的capital-aware研究政策，不改正式策略預設：R2 `capital-adjusted-score`使用正式盤前exact-accounting後的`proj_cost`與該候選`sizing_capital / max_position_cap_pct`計算部署率，以`Score × deployment_rate`排序；R3 `capital-bucket-then-score`只對同日有效Score候選依部署率橫斷面三分位分成高／中／低桶，再於桶內按Score排序。兩者在ensemble中仍維持vote count第一，部署率取共識members有限值的中位數；缺分候選保持原buy-sort fallback。這些政策不得另算第二套sizing、不得使用Future Target、不得成為optimizer trial維度，也不加入互動選單。一般Optimizer search space仍將ranking固定為False，不允許把ranking開關當trial維度；`strategy-compare --comparison-mode score-ranking`建立舊ROOS下的Baseline／Sort Only隔離對照。歷史Selection ranking×parameter adaptation CLI已於Legacy Cleanup Batch 7退役；其coverage與2×2研究證據只讀保留，不再作正式optimizer或menu入口。現行參數準備統一由`strategy_param_training`／Strategy Compare preparation負責。`--param-policy base-finalist-best` 會自動解析 `roos_base_best.json`、驗證每期 `1 member / min_agree=1`，此時所有票數相同，Quality Score 是第一個有效排序欄位，輸出置於 `strategy_compare_score_ranking_base_finalist_best/`；`--param-policy base-finalists-agree` 則保留 finalist 同意數第一、Score只重排同票候選，輸出置於 `strategy_compare_score_ranking_base_finalists_agree/`。參數 selector 或 member/min_agree 契約不一致時必須 fail-fast。該比較是在已查看舊 OOS 後的探索性機制診斷，不得直接作部署證據。 正式 Score 工件的日期語意分成兩層：`required_signal_start` 等於 `model_information_cutoff`，代表模型可於 cutoff 當日收盤後用同日訊號建立下一交易日盤前訂單，必須涵蓋 OOS 首個執行日前一交易日的盤前訊號 anchor；`execution_start` 則固定等於 `outer_oos_policy.oos_start_date`。策略回放不得因補齊 signal score 而提前，亦不得改用執行日收盤 Score。
- `tools/filters/breakout_quality/` 承接 supervised dataset、training、evaluation、研究 metrics 與 report rendering；正式 score export 位於`filters/breakout_quality/export_scores.py`，不再保留tools alias；9E legacy鏈下載並驗證釘死的MOMENT-1-base snapshot，完整frozen encoder state與來源契約內嵌正式checkpoint，後續舊工件export／report不需重新下載；9D legacy鏈下載並驗證釘死的外部MantisV2 snapshot，checkpoint與manifest保存完整frozen encoder、來源revision/hash、套件版本及project data isolation flags，score export與runtime只從正式checkpoint重建，不在推論時重新下載；Selection-only pretraining dataset、TS2Vec encoder pretraining與frozen-head training子系統保留為9C legacy工件重建；9C、9D、9E與9F均不由active 9A workflow啟動；dataset 以 `ticker/date` 為 group，只保存一份由 `BREAKOUT_QUALITY_FEATURE_WINDOW_BARS` 決定長度的 sequence feature，事件列以 `group_index` 加上 high_len context 取用；未來 K 線路徑另以可 mmap 的 `.npy` cache 保存，調整 label 門檻或 cache 範圍內的 horizon 時只重貼 label。建立時逐檔讀取股票 CSV，先寫入每檔暫存 chunk，再合併成可 mmap 的正式 `.npy`，不再一次把全部股票、事件表或特徵常駐 RAM，也不再使用大型壓縮 `dataset.npz`。Label 以突破訊號日收盤價為基準，逐日累積 MFE 與達成前 MAE；在尚未觸及最大不利跌幅前，MFE 嚴格大於最低漲幅且 MFE／MAE 嚴格大於最低比率才是 PASS，零 MAE 視為比率成立，同日同時達成 PASS 與風險上限時保守判為 REJECT；其餘完整有效路徑均為 REJECT。資料不足或 K 線無效者只標記為內部 INVALID 並排除，不是第三種 Label；不模擬成交、ATR、停損、停利或任何策略參數。訓練資料先依 experiment profile 決定 sampling unit：baseline 使用全部 event rows 加 `1/group_size`，8F `unique_group_sampling` 則在 shuffle 前只保留每個 `ticker/date` 的最小原始 row index，使每個 group 每 epoch 只參與一次 optimizer sampling；之後再依 policy 選擇 class weight 與 training-only weight。8K `unique_group_date_balanced` 歷史 profile以各 training phase 內同日 eligible unique groups數 `n_d` 設定 raw weight=`1/n_d`，並使用固定 configured batch-size denominator；此方法已由完整 OOS淘汰；active 9A及8F比較基準均不套用日期平衡；9B ModernTCN歷史結果同樣未使用日期平衡。Validation／Selection／OOS始終使用完整 rows與 `1/group_size` 評估。預設不做 inverse-frequency class balancing，年度平衡則保留 `year_balanced_sqrt` 歷史模式。Inner Validation 選出的 best epoch 會記錄實際 optimizer updates；final refit 由 policy 的 `BREAKOUT_QUALITY_FINAL_REFIT_MODE` 決定，8F 正式基準的 `selected_epochs` 會以相同 epoch 數在完整 eligible Selection 重訓；`matched_optimizer_steps` 已由 8I 淘汰，只保留歷史重現。8J 的 `best_inner_checkpoint` 可供歷史重現，但已被完整 OOS 淘汰；active 9A及8F比較基準均維持 `selected_epochs`；9B ModernTCN歷史實驗亦使用相同refit。`evaluate.py` 是完整 JSON 稽核介面，`report.py` 只重用同一份 metrics，輸出表格化終端報表、Markdown 與完整報表 JSON，不另算第二套指標。報表預設納入 OOS，並依序呈現 Epoch 選擇、Selection／OOS Confusion Matrix、各資料區段比較與 OOS 綜合判定；固定 epoch 模式與 inner validation 模式使用不同的最終模型說明。報表另以同一份固定 OOS score 產生逐年度分類、固定 coverage 排序與校準診斷表，部分年度會明確標記；年度切片不得用於回頭選 threshold、epochs 或模型。OOS 綜合判定仍只比較 Selection 與完整 OOS。training 會驗證 source-data inventory，來源已更新時拒絕沿用過期 dataset。直接 CLI 僅保留開發與既有指令相容，不再作為文件建議的正式入口。
- 9C自監督legacy工件仍採獨立契約：rolling windows位於`outputs/filters/breakout_quality/<filter_id>/pretraining/ts2vec_v1/stride_<N>/`，內含`windows.npy`、`window_index.csv`與summary；pretrained encoder位於正式architecture/profile模型目錄下的`pretraining/`；workflow與直接pretrain CLI都必須把同一個supervised `experiment_profile`傳入encoder path與manifest，禁止回退到policy預設而寫錯profile目錄。Dataset summary鎖定Selection日期上限、stride、300×10 schema、outer-policy fingerprint與CSV inventory；encoder manifest鎖定dataset fingerprint、encoder hash與`ts2vec_selection_only`命名pretraining profile完整payload，並要求`oos_windows_used=false`與`pass_reject_labels_used=false`；profile payload與active config不一致時下游訓練及runtime artifact validation均fail-fast。正式下游checkpoint保存完整encoder＋head，但optimizer只接收642個linear-head參數，831,168個encoder參數保持凍結且eval。
- Current／historical model boundary：正式新訓練只可經`filters/breakout_quality/models/active.py`的`get_active_model_spec()`／`build_active_model()`；binary trainer與continuous/daily ranker不得import compatibility factory。`filters/breakout_quality/models/factory.py`只保留artifact reconstruction／inference相容入口，遇到legacy architecture才lazy delegate至`models/legacy_compatibility.py`；MOMENT／Mantis／TS2Vec builders與external contract亦只在historical branch實際需要時載入。`get_model_spec()`／`model_spec_from_manifest()`仍能嚴格解析全部歷史model spec，因此archived checkpoint／manifest identity不變，但legacy architecture不得透過formal new-training CLI重新產生權重。 `apps/research.py model`正式command registry亦不再公開`build-pretrain-dataset`／`pretrain`；TS2Vec historical dataset／encoder modules只保留底層compatibility用途，不屬於current Research command surface。零consumer的舊`mantis_pretrained.py`／`moment_pretrained.py` standalone training/download loaders已移除；historical Mantis／MOMENT checkpoint reconstruction仍由model spec、legacy builder與artifact contract維持，不以死loader檔案作相容層。
- canonical策略比較主工件為 `strategy_comparison.md`／`.json`及pair-level equity、trades、daily-capacity、selected diagnostics、年度報酬與round-trip attribution。`core/console_report.py`是全專案簡易報表格式SSOT；RUN與REUSE固定由canonical pair JSON payload呼叫同一renderer。`filters/breakout_quality/strategy_compare_engine.py`只保留pair orchestration／CLI façade，canonical replay、reporting與diagnostics分別位於`strategy_compare_replay.py`、`strategy_compare_reporting.py`、`strategy_compare_diagnostics.py`。Strategy Compare不得import Audit；一次性score-ranking capture／portfolio attribution Audit已完成並退役，歷史研究證據只保留於Experiment Registry／Log與既有輸出工件。
- Dataset 與 future-path cache 固定共用 `outputs/filters/breakout_quality/<filter_id>/`；正式模型工件依架構與訓練實驗分開存放於 `models/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/`。`BREAKOUT_QUALITY_MODEL_ARCHITECTURE` 與 `BREAKOUT_QUALITY_EXPERIMENT_PROFILE` 都不進入 Dataset fingerprint，也不觸發 feature bank 重建；舊 v1 baseline 無 profile 子目錄的工件僅提供唯讀相容 fallback；不得在該舊路徑更新 manifest 或寫入正式 forward-OOS scores，所有新訓練與正式輸出一律寫入 profile 子目錄。Continuous Forward-OOS score artifact由profile sample scope決定：event-group ranker使用`continuous_ranker_scores.csv`並以`split=oos`取runtime rows；daily-universal ranker使用`daily_ranker_oos_scores.csv.gz`，該檔整體即為OOS daily stock-day universe，不存在event-only `split`欄；所有consumer必須共用`ranking_score_store.resolve_continuous_ranker_oos_score_path()`，不得自行拼檔名。`split_assignments.csv` 的 outer `selection/oos` 日期直接來自 `core.walk_forward_policy`。
- inner validation 為可配置模式：關閉時完整 Selection 固定 epochs；開啟時只在 Selection 內選 epoch，正式模型再依 policy 選擇直接採用 best checkpoint 或以 eligible Selection 重訓。Final refit 忠實採用 `BREAKOUT_QUALITY_FINAL_REFIT_MODE`；目前 active 9A-BN與比較基準8F均採 `selected_epochs`；9B ModernTCN歷史實驗亦採相同模式；matched optimizer steps 與 best inner checkpoint 分別由 8I、8J 淘汰。manifest 對 8J 明確保存 best checkpoint source、Final Refit `performed=false` 與 steps=0；對 refit 模式則保存 selected/target/actual steps、等效 epochs、class/time weight mode 與權重摘要。OOS 永不參與 epoch、threshold 或模型選擇。
- Binary、canonical、pretraining、continuous-ranker與Selection PIT workflow共用唯一的`BREAKOUT_QUALITY_RANDOM_SEED`；目前值為42。PIT manifest、fold fingerprint與策略score contract仍鎖定實際Seed，防止不同Seed工件resume或混接。CLI `--seed`只作單次執行覆寫，不建立第二個正式設定。
- 訓練效能優化以結果一致為硬性契約：同一 experiment profile 內每個 epoch 的 sampling units、shuffle、batch size、seed、optimizer 更新次數與訓練單執行緒設定均不得漂移；只有命名 profile 可明確改變 sampling unit。8F／8K 的 batch size 以 unique `ticker/date` groups 計，代表列固定為最小原始 event row index，且不得改變 Validation／OOS 的完整 rows口徑；active 9A及8F比較基準的early-stopping patience均為1、time weight均為none；9B ModernTCN歷史實驗亦使用相同設定。8K date-balanced profile已淘汰，只保留歷史契約。訓練使用 `zero_grad(set_to_none=True)`；optimizer、LR schedule 與 augmentation 由 experiment profile 決定；augmentation 僅允許作用於 training input，不可污染 Validation／OOS，通用 learning rate、weight decay 與 gradient clipping 仍由 policy 控制；模型每個參數在每個 batch 都有 gradient，可省去清零寫入而不改 optimizer 更新；可選擇將去重 feature bank 與小型事件陣列預載至 RAM，但資料值與列順序不變。完整 Train／Validation／Selection 評估共用 `filters/breakout_quality/inference.py` 的 strict-result event-row inference：維持相同 batch boundaries、全部 requested rows 與原輸出列序，只把彼此獨立的 batches 分派給 `BREAKOUT_QUALITY_EVALUATION_WORKERS` 個單執行緒 model replicas，再依原列序與原 reduction 順序彙總。score export 若 model spec 為 `use_dataset_context=false`，則依 canonical `event_group_index` 對每個 unique ticker/date feature group 只推論一次、先計算一次 probability，再精確 broadcast 至全部 event rows，確保同 group 分數 bit-identical；只有使用 Dataset context 的 legacy model 保留逐 event-row inference。開啟 `BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION` 時，Inner Train 與 Validation 使用相同 epoch model 的獨立 read-only 快照同時評估，不改 optimizer、random state 或 best-epoch 判定。Final refit 模式最後一輪已產生的完整 Selection metrics 直接作為 final metrics；8J 則在恢復 best checkpoint 後對完整 Selection 做一次完整 rows、既有 `1/group_size` 的 group-weighted 評估，並明確標記為 evaluation 而非 refit。GPU 與多執行緒 training 不屬於 strict-result 模式，因其浮點 reduction 可能改變模型權重與 threshold 邊界判定。
- Dataset 工件只放 `outputs/filters/breakout_quality/<filter_id>/`；research scores 與 report 依架構及實驗放在 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/`，報表固定於該目錄下的 `reports/evaluation_report.md` 與 `reports/evaluation_metrics.json`，不得覆蓋正式 `scores.csv`。
- `dl_quality_score >= active breakout_quality_score_threshold` 是唯一通過判斷；manifest 只宣告契約與 OOS 可用日期，不預先固化另一份 `dl_pass`。
- Sequence-only（`use_dataset_context=false`）正式 score export 已宣告 `shared_group_score_broadcast=true`；runtime 因此以 `ticker/date` 作為唯一 score lookup key，`high_len` 只用於驗證 artifact coverage。`score_store` 載入時必須確認同一 `ticker/date` 的所有 event-row scores 完全一致，再去重建唯一 date-level index；使用 Dataset context 的 legacy model 仍維持使用 `ticker/date/high_len` 精確 key。
- 正式 runtime 僅在 manifest 宣告的有效期間套用模型；有效期後只要出現未覆蓋候選事件即 fail-fast。

### `tools/validate/`

Breakout Quality synthetic validators依測試責任拆成policy、artifact、model、audit、PIT、strategy與strategy-app七個case modules；`synthetic_breakout_quality_support.py`只提供共享fixture imports／helpers。舊`synthetic_breakout_quality_cases.py` compatibility façade已因無current consumer而移除；正式synthetic registry `synthetic_cases.py`直接import各domain owner，meta registry contract驗證每個Breakout Quality validator只有一個domain owner、retired façade不得復活，並將support與所有domain implementation modules納入key coverage targets。 Source-level contracts透過`tools/validate/source_index.py`共用process-local source text／AST cache；cache以檔案mtime_ns＋size失效，僅消除同一輪synthetic suite重複I/O／parse，不快取validator結果、不跳過contract，也不跨process持久化。

### Continuous Target canonical build service

Continuous Target已由一次性研究Audit升級為正式continuous workflow的canonical資料建置層。Target數學與strict component loader由`filters/breakout_quality/continuous_target.py`單一承接；versioned artifact建置由`services/breakout_quality/continuous_target_builder.py`承接；描述性分布／rankability metrics由`services/breakout_quality/continuous_target_metrics.py`提供。正式模型流程只經`tools/filters/breakout_quality/prepare_continuous_target.py`呼叫service，不得反向依賴`tools/audit/`、Strategy Compare或歷史研究report。

目前支援`strategy_aligned_opportunity_r_v1`與`strategy_aligned_opportunity_no_time_r_v1`。兩者都沿用canonical event anchor、future high／low path cache、valid mask、adverse-first與固定risk-budget語意；No-time版本固定移除time-penalty項，不以Selection／Validation／OOS、actual trade R或舊Audit結論擬合係數。Target工件仍使用既有version-scoped路徑與artifact filenames以維持consumer相容，但其角色是build diagnostics，不是需要永久保留的research Audit gate。

歷史11A～11F的一次性歸因／消融CLI與專屬synthetic contracts已在研究結論寫入Experiment Registry／Log後退役；若未來Target研究結果成為正式workflow，必須同樣抽出穩定formula/service後刪除歷史gate與臨時implementation。

### 11B Strategy-aligned Daily Percentile Ranker

11B是獨立experiment profile `strategy_aligned_daily_percentile_mse`，不是新model architecture。它保留active 9A `inception_time_v1`的300×10輸入、RF229與原2-logit head，將`softmax(logits)[:, PASS]`視為0～1排序分數，對Selection內每個日期的`strategy_aligned_opportunity_r_v1` raw target percentile使用MSE。日期內採average rank，轉換為`(rank−1)/(n−1)`；同值共享平均rank，singleton固定0.5。不同日期互不共享位置、尺度或統計量，因此不建立跨年度normalization。

正式入口為：

```bash
python apps/research.py model train-continuous-ranker --filter-id breakout_quality_v1
```

Inner Train只負責gradient更新，Validation以mean daily Spearman最大化選epoch、相同Spearman時才比較較低MSE；完整Selection依`selected_epochs`重新初始化重訓。checkpoint寫入前只建立Selection percentile target，OOS percentile、OOS metrics與actual-R診斷均在模型凍結後執行。11B工件寫入`inception_time_v1/strategy_aligned_daily_percentile_mse/`獨立profile路徑，research scores每個group只保留唯一一列並以`selection_role`標示Inner Train／Validation；manifest固定`runtime_eligible=false`；binary runtime loader、classification workflow與forward-OOS score export均拒絕此profile，不覆蓋9A `unique_group_sampling`正式模型。11B是已淘汰的research-only實驗，只保留`train-continuous-ranker` CLI子命令供歷史重現；臨時研究不加入互動選單，也不另行複製訓練邏輯。

### Historical Target attribution diagnostics（retired）

原11C～11E的candidate coverage、Target component attribution與time-penalty ablation均屬已完成的一次性研究診斷；研究證據只讀保留於Experiment Registry／Log與既有outputs，不再保留runtime CLI、replay helper或專屬synthetic contract。正式Strategy Compare不得為重現這些歷史診斷保留第二套candidate replay。

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

- `apps/run_bundle.py` 是日常唯一建議使用的本機 double check 與交付打包入口。

- `apps/`：正式入口層，只從對應 application/service façade 匯入公開介面。
- `services/`：正式 application/service orchestration；可組合`core/`、`filters/`與其他正式service，不得反向依賴`tools/`。
- `core/`：核心規則、帳務、價格、統計、path 與共用 helper；不得放 UI orchestration 或 validate 腳本。
- `tools/`：Audit、CLI／GUI、下載、validate、local regression與legacy import compatibility wrapper；canonical portfolio replay、optimizer library與Breakout Quality training／PIT application service均位於`services/`，正式domain與services不得反向依賴`tools/`。
- `config/`：共用政策與執行預設。
- `models/`：模型工件與 runtime 產生或使用者保留的可選最佳參數輸入；沒有 path override 時，預設參數 fallback 仍解析到 `models/run_best_params.json`，但 repository／交付 ZIP 不必預先包含該可變動工件。
- `doc/`：架構、常用指令與 formal checklist 文件。

## 正式入口

- `apps/research.py`：研究單一正式入口；主選單只選工作類型。模型訓練、策略參數最佳化、策略組合比較與Audit維持獨立責任；Strategy Compare透過`services/portfolio_replay.py`重用canonical replay，模型／PIT producer由`services/breakout_quality/`承接。Multiple-seed robustness可保存經驗證的compact attribution source供必要的read-only診斷，但已完成的attribution Audit implementation不因此永久保留。目前formal Audit只保留`mr13e-minimum-repair-mechanism`；完成該決策後也應依disposable lifecycle退役。
- `tools/filters/breakout_quality/application.py`：Breakout Quality model provider，承接原完整model workflow、dataset、training、score export、易讀report與詳細evaluation；不是使用者直接入口。
- `apps/run_bundle.py`：日常本機 double check 與修改交付的單一使用者入口；固定順序為stage → commit current snapshot → package ZIP → formal test。formal test失敗時保留commit與ZIP，讓該失敗版本可被完整重現與交付。
- `apps/test_suite.py`：`run_bundle.py`內部formal test runner；不作為一般日常使用者入口。
- `apps/package_zip.py`：打包正式入口；直接呼叫只負責snapshot/package，不取代`run_bundle.py`的整合流程。
- `apps/portfolio_sim.py`：投組模擬正式入口。
- `apps/smart_downloader.py`：下載器正式入口。
- `apps/vip_scanner.py`：scanner 正式入口。
- `apps/workbench.py`：GUI / workbench 正式入口；也是單股 trade-analysis 的單一使用者入口；K 線檢視中的交易明細與 Console 改以獨立分頁承接。

## 正式單一真理來源 / 開發輔助

- `core/file_integrity.py`／`core/path_utils.py`／`core/serialization_utils.py`／`core/runtime_utils.py`：跨workflow共用的hash、cross-platform path判定、output serialization與environment flag解析SSOT；consumer只import/re-export，不得複製同一實作。
- `services/optimizer/dependency_stats.py`：optimizer local-min dependency統計初始payload SSOT；seed-ensemble policy snapshot仍由`core/seed_ensemble_policy.py`建立，outer rolling只保留單一wrapper owner。
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

- 依賴方向以正式domain/service為中心：`apps -> services -> filters/core`，或薄入口直接`apps -> filters/core`；CLI／GUI／Audit可為`apps -> tools -> services/filters/core`。
- `services/`、`core/`與`filters/`不得反向依賴`tools/`或`apps/`；跨層共用application orchestration放`services/`，純計算真理放正式domain／core。
- 正式 test chain 只由 `apps/test_suite.py` 與 `tools/local_regression/formal_pipeline.py` 收斂；修改交付使用`apps/run_bundle.py`，先固化commit與package，再執行formal test，FAIL時保留該snapshot供閉環修正。

## 共享邊界

- 所有工具輸出皆落在 `outputs/<category>/`；輸出位置與 retention 規則以 `core/output_paths.py`、`core/output_retention.py` 與 `doc/CMD.md` 為準。
- `core/file_integrity.py`是generic file SHA256與canonical JSON SHA256的單一真理來源；Breakout Quality artifacts、Strategy Compare identity、trade-path label identity與local-regression manifest不得各自重寫hash serialization。
- `outputs/local_regression/_staging/` 為 local regression / validate 共用暫存 staging 子目錄；不新增 `outputs/validate/` 根分類。
- `outputs/debug_trade_log/` 為 `trade_analysis` 相容輸出目錄；為維持相容性，暫沿用 `debug_trade_log` 這個 legacy 名稱。
- `outputs/debug_trade_log/`（trade_analysis legacy output dir）屬既有工具鏈相容邊界，不代表子系統角色仍是 debug-only。
- `outputs/workbench_ui/` 為 GUI runtime 快取分類；目前承接常用股票中文名稱快取。
- `outputs/strategy_compare/robustness/<fingerprint>/`只永久保存multi-seed aggregate manifest／seed metrics／summary／Markdown；每seed model、full scores與replay detail預設只在`models/research/breakout_quality/strategy_compare/multi_seed_robustness/`及run `work/`暫存，完成seed observation後依config retention policy清除。

## 維護原則

- 本檔只承接穩定子系統、正式入口、依賴方向與共享邊界。
- 高波動操作細節移至 `doc/CMD.md`；formal 細部契約移至 `doc/TEST_SUITE_CHECKLIST.md`。
- 不以 exact file-tree、helper 長清單、局部 alias 說明或暫時演進敘事作為本檔主要承載面。

### Continuous Target preparation boundary

`tools/filters/breakout_quality/prepare_continuous_target.py`是continuous workflow的泛用前置層。它不定義Target公式，只依active profile的`continuous_target_id`驗證或呼叫`services/breakout_quality/continuous_target_builder.py`。Target manifest除Dataset policy與group count外，必須綁定產生它的Dataset artifact SHA256；衍生Target另綁定來源Target manifest SHA256。`tools/filters/breakout_quality/application.py`的continuous完整流程固定為Dataset preparation → Continuous Target preparation → PIT Score build → PIT model audit。

No-time Target固定為：

```text
strategy_aligned_opportunity_no_time_r_v1
target_raw_r = favorable_return / risk_budget - adverse_return_to_peak / risk_budget
```

它重用base Target的strict component arrays、valid mask、opportunity bar、risk-breach bar與adverse-first語意；不得要求歷史11E report、approval flag或其他一次性Audit artifact才允許build，也不得以OOS fitted coefficient改寫公式。PIT Audit則直接由`services/breakout_quality/point_in_time_audit.py`承接，不再經`tools/audit/` compatibility wrapper。

### 11G PASS-conditional No-time Magnitude Ranker

既有No-time Target研究顯示Selection內可排序，但Binary AUC約0.99，因此此歷史PASS-conditional實驗不得再以全Label objective重跑11B。其experiment profile為：

```text
strategy_aligned_no_time_pass_magnitude_mse
```

仍重用active `inception_time_v1`與既有2-logit head；模型結構與checkpoint shape不變。`services/breakout_quality/train_continuous_ranker.py`依profile的`training_label_scope=pass_only`，只在同日PASS groups內建立No-time Target percentile，並只用PASS groups更新gradient、選epoch與完整Selection refit。

Checkpoint寫入前不得建立OOS percentile；模型凍結後才輸出OOS PASS-only主要指標、all-label次要診斷，以及actual PASS／REJECT round-trip分層結果。11G為research-only、CLI-only，不加入互動選單，不建立threshold、runtime combination或forward-OOS正式scores。

### 11H～11K 歷史診斷（已完成並退役）

11H PASS realization-gap、11I Selection strategy-realization、11J per-candidate counterfactual與11K portfolio selection-pressure已完成其研究決策用途。其結論與實驗身份保留於`doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md`／`doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md`及既有輸出工件；對應一次性Audit implementation、CLI registration與專屬synthetic tests已從current tree移除，不再列為可執行正式命令。若未來出現新的決策不確定性，應依新的Audit identity與最小必要證據重新設計，不得復活舊臨時入口。

### Binary DL 4×2 Min ROOS rolling parameter adaptation boundary

`filters/breakout_quality/strategy_param_training.py`是CLI-only研究編排層，固定比較四種參數來源：P0原ROOS＋原正式規則、P1原ROOS＋rules全關、P2在rules全關／DL關環境訓練、P3在rules全關／DL開環境訓練。每套參數各以Binary DL關／開回放一次，形成A0／B0至A3／B3八個操作點。P2與P3都只搜尋`high_len`、`atr_len`、`atr_buy_tol`、`atr_times_init`與`atr_times_trail`；其餘optimizer維度依canonical config/schema固定，不再先訓練另一輪完整ROOS來提供`high_len`或非Min欄位。Outer Rolling trials的未覆寫預設只讀`config.training_policy.OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT`，不得另設200／300等第二套預設；不複製objective、portfolio replay、active-param export或統計。

Min ROOS每個fold使用同一組canonical非Min固定值；`tools/optimizer/outer_rolling_oos.py`以process-safe fold-specific fixed overrides在session建立前依OOS起日套用，使objective、local-min review、OOS diagnostics與active-param輸出共用同一effective contract。`strategies/breakout/search_space.py`的數值與categorical解析器必須優先採用session fixed overrides，避免固定欄位繼續消耗trial維度。

P3不得使用final 9A score回灌歷史optimizer。`tools/filters/breakout_quality/build_binary_point_in_time_scores.py`以expanding-window folds建立Binary PIT模型與scores；每個fold的Inner Train、Validation與完整refit都只包含score period開始日前且Label已完成的groups，score period不參與訓練或epoch選擇。`filters/breakout_quality/binary_pit_score_store.py`驗證schema、coverage、唯一key、0～1分數及`model_information_cutoff < score date`。`filters/breakout_quality/runtime.py`以scoped filter source context切換Binary PIT；optimizer主process與平行workers都必須驗證相同manifest／scores identity，缺失或不一致時fail-fast，禁止回退canonical final／research／Selection in-sample scores。Binary PIT的最早合法日期由無前視訓練量決定，不要求倒推覆蓋整段120個月Selection；PIT開始日前固定pass-through，語意等同DL-off，PIT期間內缺少候選分數採保守REJECT，PIT尾端早於optimizer最新Selection則fail-fast。P3 preflight必須逐fold輸出`bootstrap_fallback_only／partial_score_history／full_score_history`與calendar coverage，並把完整coverage policy納入runtime identity。

最終報表只保留八操作點矩陣、各參數下DL增量、參數適應比較與風險參數差異。正式採用主比較為`B3−A2`；interaction=`(B3−A3)−(B2−A2)`只能判斷DL-aware參數是否改善DL增量，不可取代絕對績效。

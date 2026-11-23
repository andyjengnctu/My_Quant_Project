# 專案檔案樹

本檔只用來定位**目前的主要檔案、目錄、正式入口及穩定職責**。這是架構示意樹，不列出所有內部 helper 或執行時動態產生的工件；實際目錄與有效設定以程式、`config/` 和各 domain 的契約為準。

**不在本檔記錄**：日期／Phase 修補歷程、性能測試、REUSE／RUN 事故、研究結果、舊版本 SHA、已退役功能回顧、待辦。研究數據與決策由 Experiment Log／Registry 持有；本檔不另建第二份紀錄，也不新增封存。

## 穩定檔案樹

```text
project/
├─ apps/
│  ├─ research.py                     # 研究單一正式入口：模型訓練／策略參數最佳化／策略組合比較／Audit
│  ├─ portfolio_sim.py                # 投組模擬正式入口（薄入口）
│  ├─ smart_downloader.py             # Market Data V2 Daily Update + Preflight/Bootstrap 正式互動入口（薄入口）
│  ├─ run_bundle.py                   # 本機 double check／commit／package／formal test 單一使用者入口
│  ├─ package_zip.py                  # run_bundle 使用的專案打包 helper；亦保留直接 snapshot 用途
│  ├─ test_suite.py                   # run_bundle 內部 formal test runner
│  ├─ vip_scanner.py                  # 掃描器正式入口（薄入口）
│  └─ workbench.py                    # GUI 工作台正式入口（薄入口）
├─ config/
│  ├─ breakout_policy.py              # breakout 策略預設與 optimizer high_len 可調範圍（declarative）
│  ├─ research.py                     # active model／provider spec／Research orchestration／active cutoff compatibility alias（declarative）
│  ├─ trading.py                      # Trading active strategy／param selector／DL policy（declarative）
│  ├─ downloader.py                   # downloader篩選、timeout、sleep與verbosity設定（declarative）
│  ├─ market_data.py                  # Research V1/V2與Trading Market Data lifecycle設定（declarative）
│  ├─ runtime.py                      # Scanner／Portfolio／Optimizer／Research process runtime與console設定（declarative）
│  ├─ breakout_quality.py             # BQ 使用者／專案可調設定與current workflow membership（declarative）
│  ├─ strategy_compare.py             # Strategy Compare執行／diagnostic／promotion設定（declarative）
│  ├─ audit.py                        # Audit active module與output root（declarative）
│  ├─ training_policy.py              # Strategy Optimizer generic policy／execution defaults（declarative）
│  ├─ strategy_params.py              # Research [2] current strategy-parameter target + validation protocol config
│  ├─ training_performance_policy.py  # optimizer execution/performance／timing benchmark knobs（declarative）
│  ├─ display_policy.py               # console/report 顯示設定值（declarative）
│  ├─ broker_accounting.py            # 券商原始手續費／稅率／整元捨去 settlement SSOT
│  └─ execution_policy.py             # Research資金／與Trading一致的券商現金費稅預設、凍結研究費率常數與runtime parameter schema（declarative）
├─ core/
│  ├─ config.py                       # 相容 façade；穩定匯出設定常數與參數契約
│  ├─ execution_policy.py             # execution config snapshot runtime helper
│  ├─ display_policy.py               # display coercion／formatter／snapshot runtime helper
│  ├─ research_summary.py             # Summary 共用研究清單投影與 model/strategy binding 驗證
│  ├─ research_policy.py              # Research provider／artifact policy typed resolver
│  ├─ trading_policy.py               # Trading strategy config validation／resolved snapshot
│  ├─ trading_data_dependencies.py    # Trading strategy/model required market-data dependency registry + fingerprint SSOT
│  ├─ runtime_domains.py              # Research/Trading data/models/outputs/state path SSOT與write guard
│  ├─ market_data_contract.py         # Market Data provider/price source與Research/Trading lifecycle typed contract
│  ├─ market_data_dataset_registry.py # FinMind V2 archive roster/query strategy/PIT-review classification registry
│  ├─ market_data_integrity_contract.py # Full Database Integrity date-semantic audit contract；不參與 Provider Snapshot fingerprint
│  ├─ market_data_instrument_universe.py # FinMind StockInfo/delisting → historical TWSE/TPEX stock+ETF universe filter
│  ├─ market_data_bootstrap_planner.py # Preflight evidence → exact logical bootstrap request plan
│  ├─ market_data_bootstrap_requests.py # Registry/evidence → deterministic bootstrap HTTP request manifest/identity
│  ├─ market_data_bootstrap_progress.py # Bootstrap ETA/duration lightweight SSOT；不依賴interactive menu/application
│  ├─ market_data_execution_policy.py # Market Data V2 quota/retry/lease execution policy resolver
│  ├─ market_data_storage_policy.py   # Market Data V2 Parquet/ZSTD/disk-headroom execution policy resolver
│  ├─ market_data_storage_contract.py # Market Data V2 bootstrap archive layout/metadata/commit receipt contract
│  ├─ market_data_trading_sync_policy.py # Trading V2 archive sidecar repair cadence/non-blocking policy resolver
│  ├─ market_data_trading_sync.py     # Provider snapshot → deterministic Trading daily sync request manifest
│  ├─ market_data_freshness_contract.py # 51-dataset cadence/publication/freshness/completeness operational SSOT
│  ├─ market_data_due_planner.py       # local-only target/publication due planner；NO_DUE 不觸碰 provider
│  ├─ market_data_trading_storage_contract.py # Trading V2 overlay/state/batch physical layout contract
│  ├─ trading_market_clock.py          # Trading完整日K safety cutoff／information-date seal
│  ├─ trading_capabilities.py          # Pre-live implementation capability facts／LIVE blocker SSOT
│  ├─ trading_account_state.py        # Trading cash/positions atomic state + hash-chain mutation contract
│  ├─ trading_order_state.py          # Trading broker-order ORDERED/PARTIAL/FILLED/CANCELLED state + hash-chain contract
│  ├─ trading_fill_transaction.py     # Trading confirmed-fill two-state write-ahead transaction contract
│  ├─ training_performance.py         # optimizer performance config resolver／env override／snapshot
│  ├─ research_execution_resources.py # Research seed／ticker 共用 CPU 資源預算
│  ├─ research_execution_equivalence.py # 已驗證執行加速的來源／計算等價證據
│  ├─ training_policy.py              # Strategy Optimizer training policy validation／derived snapshot／runtime owner
│  ├─ strategy_param_scientific_contract.py # Strategy Parameter resolved scientific-contract/fingerprint boundary
│  ├─ selection_policy.py             # history-selection 策略參數 schema／default snapshot owner
│  ├─ breakout_policy.py              # breakout high_len range validation／derived values helper
│  ├─ strategy_params.py              # breakout + training gate + execution 聚合參數契約
│  ├─ strategy_compare_registry.py    # Compare Suite／profile／arm／contrast catalog SSOT
│  ├─ strategy_compare_policy.py      # Strategy Compare resolver／validation／authorization
│  ├─ audit_registry.py               # formal/reusable Audit definitions／source／dimensions SSOT
│  ├─ audit_policy.py                 # Audit typed definition／validation／runtime resolver
│  ├─ breakout_quality_registry.py    # BQ scientific/profile/research-spec registry SSOT
│  ├─ breakout_quality_fit_contract_version.py # FIT 現行版本／已發布工件版本驗證的共用唯讀 owner
│  ├─ breakout_quality_policy.py      # BQ derived/current workflow／comparison／Rolling／PIT resolver
│  ├─ breakout_quality_runtime.py     # Continuous-ranker generic execution capability／composition owner
│  ├─ breakout_quality_runtime_resolver.py # scientific profile → runtime recipe 單向 resolver gateway
│  ├─ capital_policy.py               # 單股/投組/scanner 共用資金與 sizing 規則
│  ├─ exact_accounting.py             # 正式整數 ledger / cost-basis allocation / tick 正規化單一真理來源
│  ├─ backtest_core.py                # 單股回測總控 façade
│  ├─ portfolio_engine.py             # 投組 timeline 總控 façade；只保留run_portfolio_timeline orchestration
│  ├─ portfolio_day_candidates.py    # 每日候選 gate/producer 委派與盤前資源選擇協調
│  ├─ portfolio_day_signal_lifecycle.py # Re-entry／滿倉setup追蹤／延續訊號清理的single/multi seed路由
│  ├─ portfolio_replay_diagnostics.py # Replay candidate/selector/order/交易診斷輸出組裝
│  ├─ portfolio_terminal_equity.py    # 期末結算後權益/退款應收/月季年帳面修正
│  ├─ portfolio_terminal_statistics.py # trade metrics協調與canonical period-return報表彙整
│  ├─ portfolio_terminal_reporting.py # optional profile sidecar與原23/25欄結果contract組裝
│  ├─ portfolio_benchmark.py          # benchmark period統計與bounded cache
│  ├─ portfolio_replay_support.py     # replay phase錯誤上下文與candidate diagnostic snapshot
│  ├─ portfolio_levels.py             # held／shadow active-level rows
│  ├─ portfolio_ensemble.py           # seed-ensemble candidate aggregation與signal orchestration
│  ├─ portfolio_entry_plans.py        # reserved/cash-capped entry plan primitives
│  ├─ portfolio_entry_selection.py    # resource-aware selector thin router
│  ├─ portfolio_entry_selection_common.py # Binary／Continuous共用resource-aware search
│  ├─ portfolio_entry_selection_max_dl.py # Max-DL／feasible-ascent search
│  ├─ portfolio_entries.py            # reserved order fill／missed-buy／extended-signal cleanup state transition
│  ├─ console_report.py               # 全專案簡易console格式SSOT
│  ├─ path_utils.py                   # 跨平台path判定／project-relative顯示path SSOT
│  ├─ file_integrity.py               # file／canonical-JSON hash SSOT
│  ├─ process_lock.py                 # local process-owned SQLite mutex；跨 thread/process 的 non-blocking critical section
│  ├─ serialization_utils.py          # 共用output text／JSON-native serialization SSOT
│  ├─ runtime_utils.py                # 共用runtime／environment flag helper
│  ├─ model_paths.py                  # models 目錄與預設參數來源解析
│  ├─ output_paths.py                 # outputs/<category> 目錄正規化與建立 helper
│  ├─ output_retention.py         # outputs retention 雙門檻清理 helper
│  └─ display.py                      # 顯示 façade
├─ data/                             # Research／Trading market data，實際路徑與分區由 runtime_domains 定義
├─ doc/
│  ├─ PROJECT_SETTINGS.md             # 跨模組專案治理規則
│  ├─ BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md # 研究身分、決策與來源
│  ├─ BREAKOUT_QUALITY_EXPERIMENT_LOG.md # 實際研究結果與數據
│  ├─ TEST_SUITE_CHECKLIST.md         # formal test suite 主表與索引
│  ├─ ARCHITECTURE.md                 # 本檔
│  ├─ CMD.md                          # 使用者自行維護的常用指令筆記（GPT不得自動增改）
│  └─ scientific_contracts/           # machine-readable 報表契約版本台帳
├─ filters/
│  └─ breakout_quality/               # quality feature/label、shared split、artifact contract、model factory、正式 runtime score lookup
├─ services/
│  ├─ portfolio_replay.py             # canonical portfolio replay／market-context application service
│  ├─ portfolio_sim/                  # portfolio CLI/report/runtime service owner
│  ├─ downloader/                     # 正式資料下載 owner；main只負責composition/render/dispatch；menu_contract持有輕量選單契約，另含Preflight、Bootstrap、ledger、quota executor與Parquet storage
│  ├─ scanner/                        # 正式掃描 runtime service owner
│  ├─ workbench_ui/                   # GUI application service/UI owner；含回測檢視、交易中心、帳務中心與資料中心頁
│  ├─ trade_analysis/                 # 單股 trade-analysis backend/service owner
│  ├─ breakout_quality/               # fitted_model_artifacts.py：canonical FIT 身分／權重等價；fitting_checkpoint_store.py：Shared FIT cache 唯一發布 owner；point_in_time_scores.py：PIT 評分 producer
│  ├─ audit/                          # reusable read-only Audit；dl_fit_sample_coverage.py核對全DL Rolling FIT Fold／來源簽章／年度覆蓋，不建置模型或Target
│  ├─ research/                       # Research application boundaries／artifact orchestration；target_oracle_application.py 僅在取得授權後委派既有 Optimizer P16，Oracle Audit 仍只讀
│  │  ├─ strategy_robustness_units.py # 逐策略/Seed不可變結果索引、單元身分、凍結差額工作計畫
│  │  ├─ strategy_multi_seed_robustness.py # 先解析完成證據，只委派差額依賴/FIT/Replay，組裝正式多Seed報表
│  │  └─ strategy_compare_engine.py # 共用交易/指標owner；支援不重跑舊基準的單arm暫存觀察輸出
│  ├─ trading/                        # Trading orchestration；market_data_update canonical update、data_readiness dependency gate、market_data_dataset_state state owner、strategy-param/account/order/fill/protection/daily workflow
│  └─ optimizer/                      # 正式optimizer primitives；目前含raw cache／trial inputs／walk-forward
├─ models/
│  ├─ filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/
│  │  ├─ model.pt                     # architecture/profile-scoped canonical model artifact
│  │  ├─ split_assignments.csv        # outer Selection/OOS + Selection train/embargo assignment
│  │  ├─ manifest.json                # model spec/profile/split/score/OOS eligibility 契約
│  │  ├─ scores.csv                   # architecture/profile-scoped canonical event score table
│  │  └─ point_in_time/               # Selection rolling/cross-fitted Score與fold工件
│  ├─ strategy_params/                # Research Optimizer-owned Strategy Parameter SSOT；models root不再放current策略JSON
│  │  ├─ canonical/                   # Research current策略；family/policy以檔名區隔，一檔保存完整跨年schedule
│  │  │  ├─ full_base_best.json / full_*_finalists_agree.json / full_ensemble_*.json
│  │  │  ├─ min_base_best.json / min_*_finalists_agree.json / min_ensemble_*.json
│  │  │  ├─ full_manifest.json / min_manifest.json
│  │  │  └─ run_best_params.json / candidate_*_params.json
│  │  └─ benchmark/<benchmark_id>/    # 隔離robustness題庫；每個獨立seed以檔名區隔；只放正式truth
│  │     ├─ full_base_best_seed_<seed>.json
│  │     ├─ min_base_best_seed_<seed>.json
│  │     └─ <family>_seed_<seed>_manifest.json
│  └─ trading/strategy_params/        # Trading Optimizer-owned Strategy Parameter SSOT；與Research physical truth隔離
│     └─ canonical/                   # trade latest-data/fixed-window multi-seed policy artifacts + manifest
├─ outputs/optimizer/strategy_param_schedule/  # Optimizer resume/intermediate workspace；永不作current truth
│  ├─ canonical/<family>/schedule_2021_forward/
│  └─ benchmark/<benchmark_id>/seed_<seed>/<family>/schedule_2021_forward/
├─ state/                            # Trading mutable 帳戶／訂單持久狀態（與 Research 隔離）
└─ tools/
   ├─ filters/breakout_quality/       # 開發／historical research CLI，不保存正式 runtime 第二份規則
   ├─ optimizer/                      # optimizer互動／尚待搬遷相容層
   ├─ validate/                       # validate / synthetic / real-case 驗證子系統
   └─ local_regression/               # reduced formal orchestrator
```

## 主要分層與正式入口

- `apps/`：正式 CLI／GUI 入口；主要流程委派給 `services/`，不在入口複製業務邏輯。
- `services/`：研究、Trading、下載、投組模擬、Optimizer、Workbench 的正式組合與工作流程。
- `core/`：交易、風控、費稅、資金、統計、來源及共用契約的唯一規則 owner。
- `filters/breakout_quality/`：Breakout Quality 的特徵、Target、模型及正式分數消費元件。
- `config/`：宣告式設定；`tools/`：隔離診斷及驗證，不是第二套正式交易引擎。
- 依賴原則為 `apps → services → core/filters`，不得由正式 domain 反向依賴 `apps/`、`tools/`。
- `apps/run_bundle.py` 是日常唯一建議使用的本機 double check 與交付打包入口；`apps/test_suite.py` 是其內部 formal runner。

## 正式入口

- `apps/research.py`：Research 單一選單；`[1] Summary`、`[2] Target 驗證`、`[3] 模型訓練／驗證`、`[4] 策略參數最佳化`、`[5] 策略組合比較`，其餘工作依序延後。
- `config/research_summary.py`：目前研究清單唯一來源；每列宣告既有 Oracle recipe、模型／Positive 模型與策略 binding，或無 DL 對照。`core/research_summary.py` 推導三個工作入口的 Target、DL 與策略集合；各 scientific Registry 仍持有算法、身分與歷史定義。Summary 唯讀消費 canonical evidence，缺工件仍保留列；沒有匹配 Oracle 的模型／策略只顯示實際結果，Oracle 欄位留白。
- `[2] Target 驗證`：正式授權矩陣由 `services/audit/target_oracle_romd.py` 唯一核對；只有註冊的科學 Target × Strategy treatment 進入 Target 驗證畫面、Target RUN／REUSE 計數與人讀 Target 表。No-DL 策略不具 Target，留在 Summary／Strategy Compare；已核准的同 Seed RoMD 差值仍需同經濟家族的 No-DL 控制，該控制以明列且受授權的「非 Target 策略基準依賴」參與內部 Replay 與簽章，但不混入 Target 工作清單。Target 清單與內部控制依賴都從同一 Oracle Registry 投影，簽章 receipt key 不改。Replay 僅按當次已授權 `(seed, target)` RUN 格子動態分派，每個活躍 Seed 一個獨立程序，Seed 內維持原本 Target 順序及同 Seed 控制配對。共用 price-score 真值僅由主程序預先建立，Seed-specific 真值／Portfolio Replay 工件維持各自 Seed 路徑；完整 REUSE 不啟動子程序。並行上限由 `config/audit.py` 的 `TARGET_ORACLE_REPLAY_SEED_WORKERS` 控制（資源設定，不影響科學／工件身分）；不可跨 Seed 重用未驗證工件。
- Target 工件契約：`services/research/oracle_artifact_contract.py` 持有有效計算／資料／P16／交易契約與完成結果索引；`services/audit/oracle_truth_references.py` 持有 Min 真值的共用消費參照。Oracle 完美分數已替代排名與 Positive 預測，對照模型／策略編號只保留為來源證據；有效契約仍逐項保存與核對評分資格、時間、Selector、Positive 門檻、入場、參數與費稅。同一契約投影供新工件、舊簽章驗證、REUSE 計畫與Summary 共用；Summary 優先從同一計畫讀取已驗證且全部 Seed 齊備的結果，不依賴聚合報表重新發布。日常矩陣只核對契約與原始簽章收據，舊格式的完整來源核對限首次匯入；數值真值只由 RUN 消費者讀取並驗證。RUN worker 必須攜帶其 Seed／Target 的有效契約；REUSE 由結果索引參照已認證的原始收據，不另複製 Replay 或產生 source-revision 別名收據，歷史證書僅保留讀取驗證。原始 Replay／NPZ 保留其物理來源 SHA，不把模型清單、報表或 Dataset 發布 metadata 當成執行輸入。計算 revision 是工程失效身分：影響 eligibility、Oracle 截斷／組合、共用 Replay 或未納入數值 owner 投影的執行修復，須調升唯一 owner 中對應 revision；不另立 scientific identity，也不追加舊 producer SHA 通行名單。
- Oracle 來源換行等價：`mintrade_truth_source_attestation.py` 從當前完整來源 bytes 重建全 LF／CRLF 與各單檔換行變體；`oracle_artifact_contract.py::oracle_source_contract` 將此證明組合進 capability 宣告的 delegated closure，計算 owner、資料與經濟契約仍須完全相同。Replay 索引、遺失索引時的原始 signed-context SHA 重建及 truth reference 共用此投影；原始收據／來源 SHA 不改寫，不讀取數值陣列，不接受未能從執行來源重建的歷史 digest。
- Daily trade-path capability：`core/daily_trade_path_capabilities.py` 宣告 Target、teacher、source、數值 primitive 與 producer；runtime policy、模型前置依賴及 Target 顯示從該宣告派生。`daily_full_p20_training_source.py` 的 `DailyTradePathTeacher` 是逐 seed 歷史／當期 Optimizer 解析與驗證的共用 owner；保留原模組路徑及 P20 API。`daily_full_p20_pit_source.py` 的共用 materializer 持有 cutoff、maturity 與 outcome cache；各能力只提供入場 primitive 與獨立 cache namespace。`daily_filtered_mintrade_target.py` 組合正式 Daily admission 與既有 close-limit Broker-Cash primitive；`daily_filtered_trade_path_truth.py` 持有該能力的 signed cohort／outcome／witness 流程。既有 P20 Oracle producer 保留其原計算與有效工件契約。
- Teacher 依賴：`trade_path_target_source.py` 從 capability 產生歷史／當期依賴，標準模型與 Strategy Compare 前置圖消費同一 `TradePathSource.dependencies`。`strategy_compare_preparation_status.py` 在 service layer 投影依賴；`strategy_compare_dl_artifacts.py` 只驗證工件；`strategy_compare_sources.py` 持有參數路徑解析。Consumer 不隱形訓練 teacher，也不以其他 seed／家族替代。
- Oracle 執行依賴隔離：P16 Breakout／既有 Daily P18 的 Replay 數值 owner 與 effective-context payload 維持既有簽章，不因新增 P19／P20 候選功能而增添欄位；不可透過舊 SHA 白名單或不經驗證的歷史數值補位。新 Daily 參數家族的執行契約由 canonical Strategy Parameter Profile 宣告；`services/research/oracle_artifact_contract.py` 的 `attest_daily_candidate_execution` 對該執行契約計算唯一數值能力指紋，**同時**寫入實際 Replay 簽章政策與 Oracle effective context，兩者不一致即 fail-closed。P20 的 Daily Full 候選濾網由 `core/daily_common_entry_filters.py` 持有，經 `core/portfolio_fast_data.py` 同一正式資料預處理入口注入，不擴充舊的 `core/signal_utils.py`；其本身的修改只失效使用 P20 的認證，P19／P18 與 Breakout 不應連帶失效。所有策略共享的價格限價、ATR、成交與持倉引擎如實際變更仍須令受影響的 Replay 失效；依真實 source closure 判斷，不靠模型或 C 編號推測。測試須覆蓋舊原指紋穩定、新能力獨立失效、Replay／Context 簽章一致、各 active No-DL 對照及多個參數家族動態增退；未有四 Seed 正式收據不得顯示 Oracle RoMD。
- Optimizer Daily trial-preparation contract: `services/optimizer/session.py` 的 `OptimizerSession.optimizer_runtime_context()` 是資料準備、Trial／Local-Min／Finalist、OOS diagnostics 與 Replay 的共用 execution scope；`prepare_trial_inputs` 不得在未綁定該 scope 的 Optuna thread 上準備 Daily 候選，且 worker 必須接收同一明確 `daily_execution_contract`。P20 的 `is_setup` 候選與正式 Portfolio Engine 必須採同一 Daily Full 濾網口徑。修正會使既有 Optuna Trial 失效的資料準備 Bug 時，由 `strategies/daily/search_space.py` 所屬 Profile 提供工程 runtime revision，`services/optimizer/strategy_param_training.py` 僅把它納入適用 Profile 的 optimizer runtime／Study 身分，避免錯誤 Trial 假 REUSE；不得變動 P18／P19 科學契約、原有參數上下限或已發布 Strategy Replay 結果。
- Positive 模型／策略：Rank 與 Positive classifier 是獨立 FIT，使用相同 seed、cutoff 與成熟 raw NetPnL lineage。Classifier 為真正二類 Cross-Entropy，提供 `positive_p_positive`；`core/positive_netpnl_contract.py` 在 slot／cash 處理前套用嚴格 `P(positive)>0.5`。`strategy_compare_dl_artifacts.py` 驗證兩份依賴，`strategy_score_projection.py` 僅簽署各模型實際提供的欄位。現行模型、策略與版本由 Registry／config 持有。
- Positive 自學決策能力：`core/breakout_quality_runtime.py` 的 `PayoffCategoricalTrainingPolicy` 是權重、驗證與 action 欄位語意 owner；繼承既有二類訓練 policy，舊 policy payload 不增加欄位。`filters/breakout_quality/categorical_decision.py` 持有 FIT-only 權重、payoff CE、weighted Validation CE/Brier 與 strict-logit action 的唯一實作。Generic trainer 按 capability 委派，`continuous_ranker_pipeline.py` 由同一模型 logits 輸出連續 payoff support 與二元 action；Forward／PIT 的既有 score contract 自動保存／驗證欄位。模型不以其他 DL 分數為 input，不新增 architecture、資料 loader、fold planner 或 FIT owner。
- 模型 action admission：`core/positive_score_admission.py` 的 `model_decision` source mode 只消費同資訊日的獨立第二模型 action=1；拒絕 probability threshold 設定與缺分數，不影響 Rank、cash／slot 規則。`core/strategy_comparison.py` 依第二模型 policy 驗證實際輸出欄位。新比較仍有且只有兩個 DL；既有 probability admission owner 與 Oracle 證書維持原契約。
- Target 顯示：`core/research_summary.py::research_summary_target_label` 是交易路徑 Target `(Pxx)` 後綴的唯一 formatter；Oracle 與尚無 Oracle 的 Summary 策略列皆由實際 Optimizer parameter-set 取得版本後共用，模型專屬列不憑名稱猜參數。顯示字串不進入任何 Target／FIT／Replay／Oracle 簽章。
- 未決 Oracle：Summary 可對新 model-output composition 明列 `CONTRACT_UNRESOLVED` 與原因，由 `core/research_summary.py` 驗證；Oracle coverage 與 Target×DL binding 共同消費此宣告。僅豁免該尚無核准 perfect-score mapping 的比較，不能借用相同 Rank／Target 但不同 admission 的 Oracle，亦不能造成既有策略 binding 不唯一。正式模型與策略比較仍可執行；原報表欄位與選單不變。
- Research input validation：`core/file_integrity.py` 的 `compute_observed_file_sha256` 只記憶已讀取 bytes 的 SHA，依檔案路徑、裝置／inode、size、mtime 與 change time 失效；Windows 使用 `FileBasicInfo.ChangeTime`，不能以 creation time 當作改寫見證。來源清單仍逐次比對 canonical inventory 與內容收據，不快取 READY。`workflow_io.py` 依 Dataset summary／完整 backing-file revision 共用一次 SHA、shape、row 與 future-path 驗證，呼叫端 policy／來源資格逐次驗證；僅保留一份 event table 並回傳獨立副本，不持有 mmap handle，以免阻擋 Windows 更新資料。跨程序重新驗證；新發布、原地改寫、刪檔或驗證中異動不能沿用先前結果。執行等價只由 `research_execution_equivalence.py` 的精確程式對照認證，不改 Model FIT、Strategy Replay、Target 或來源資料身分。
- Layered Oracle reuse: Oracle receives **perfect independent NetPnL ranks and NetPnL-positive flags**, independent of the real predictor topology, and retains signed per-seed receipts only under verified effective inputs, P16, cash, selector, cutoff, score-membership, and a reviewed identical positive-admission producer. Legacy receipt versions are never directly reused for revised *actual* DL strategy runs. Model PIT manifests and score columns must include the independent classifier output; when either score is missing the shared selector fails closed. Source numerical/input contracts are not a target identity rename. Signed perfect-score Oracle and actual DL results have separate evidence and REUSE checks.
- `apps/workbench.py`：GUI / workbench 正式入口，`apps/workbench.py` 為單一 GUI 啟用入口；單股檢視為 K 線圖、交易明細、Console 三個分頁。
- `apps/smart_downloader.py`：Market Data 更新、Preflight／Bootstrap 入口。
- `apps/cleaner.py`：專案檔案清理的獨立選單，可預覽、執行清理及查閱最近一次紀錄。
  - 遷移完成的舊參數仍須核對現行路徑覆寫及正式 preparation 的完整依賴解析，包含 builder 間接使用的上游參數及各來源政策；仍被使用的舊路徑保留。診斷快取的現行模型判定亦涵蓋這些間接參數的 teacher 與其他正式模型依賴。
  - Robustness 的可拋棄工作範圍由原 producer 的 retention policy 提供。清理先驗證完成工件，並保留舊報表補齊所需的 baseline 及來源收據引用的工作檔；未完成工作、正式模型／分數和原始研究證據不以退役或檔案大小判定刪除。
  - Robustness 工作區辨識不固定套用單一目錄深度，舊版輸出位置也須通過相同完成驗證。可清範圍內的空目錄與檔案共用工作鎖及刪除依據；預覽不移除目錄，未知範圍及連結保留。
  - 已發布比較 CSV 以內容、權限、修改時間及完整性證據核對後共用實體，保留全部原路徑。既有共用群組亦可合併，釋放空間只在最後一個舊實體連結替換成功時入帳；中斷可從原選單接續。
- `apps/portfolio_sim.py`：投組模擬入口；`apps/vip_scanner.py`：掃描器入口。

## 主要資料夾、持久來源與相容邊界

- Strategy Robustness：比較清單是結果視圖，不是完成單元的身分。設定的 Robustness output root 下 `scientific_units/` 原子保存逐策略/Seed 收據與原始交易證據參照；舊格式結果經原契約驗證後匯入，不改寫原始績效或分數證據。計畫與執行共用一份凍結的 REUSE/Replay 集合，只對待執行單元解析模型依賴；每單元完成即提交，不依賴整份報表完成。來源證據損毀或同身分矛盾不得當成快取未命中而重算覆寫。
- Robustness 依賴未就緒時的唯讀 REUSE 預檢：若共用 Strategy preparation 已由已驗證的 PIT 或 Dataset 合法解析比較期間，單一新 Target 的歷史 Optimizer teacher `BUILD/RESUME` 不得使其餘舊 arm/seed 的比較期間消失。參數、Dataset 可驗證的舊單元仍依其原始 scientific-unit signature 匯入既有 signed observation；缺參數 SHA 的新單元不得 REUSE。這只影響 `PREPARABLE` 計畫及原有結果的唯讀辨識，不允許在 Target teacher 尚未 READY 時執行、越過 Optimizer 的 No Finalist gate，或冒充整份 Robustness READY。
- Robustness 模型工作區：在原設定 model work root 下依單一來源契約/Seed定位，不受其他模型或策略清單影響。來源寫入鎖涵蓋生成、回放與清理；未完成 Fold 仍由 canonical FIT producer 判定 RESUME，consumer 不刪除既有工件以強制重訓。正式報表、模型 FIT identity 與統計仍由原有 owner 持有。
- `models/strategy_params/canonical/run_best_params.json`：研究 canonical 策略參數來源；models root不再放current策略JSON。正式測試使用獨立的 `formal_primary_params.json` 和 `V16_RUN_BEST_PARAMS_PATH` override，不覆寫研究正式參數。
- `core/runtime_domains.py` 管理 Research／Trading 的 data／models／outputs／state 實體分區；`core/output_paths.py`、`core/output_retention.py` 管理輸出與保留規則。
- `services/trade_analysis/`：單股 trade-analysis 子系統；由 `apps/workbench.py` 經 `services/workbench_ui/` 觸發。`trade_log.py` 提供共用 backend / 開發輔助 CLI，為維持相容性仍暴露 legacy `run_debug_*` / `debug_trade_log`，不作正式使用者入口。
- `outputs/debug_trade_log/`（trade_analysis legacy output dir）屬相容路徑，名稱沿用 `debug_trade_log`；不是新的 debug-only 交易規則來源。
- `outputs/local_regression/_staging/` 為 local regression / validate 共用暫存 staging 子目錄；不新增 `outputs/validate/` 根分類。
- `doc/PROJECT_SETTINGS.md` 持有治理原則；`doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md` 持有研究身分／決策；`doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md` 保存研究數據；`doc/CMD.md` 由使用者自行維護。

## 本機檢查工具位置

`tools/local_regression/`：

```text
├── formal_pipeline.py
├── run_meta_quality.py
└── run_all.py
```

- `run_meta_quality.py`：meta quality 工具；負責 formal contract、coverage、摘要與結構一致性檢查。
- Formal suite 的 `tools/local_regression/manifest.json:test_cpu_threads` 統一限制測試子程序的 native CPU pools（預設 1）；synthetic runner 同時限制已載入的 PyTorch，結束／失敗均恢復原執行緒數。只影響隔離測試，不改 Research／Trading 訓練設定。Quick Gate、Consistency、Chain Checks、ML Smoke 依 CPU 數有界並行，Meta Quality 等待全部 producer 完成後消費本輪 coverage；不重用前輪 PASS、不減少 validator、coverage targets 或門檻。
- Consistency 的 synthetic execution 由 `tools/validate/validator_execution.py` 負責；`manifest.json:synthetic_workers` 是子程序數上限（預設 4，1 為原循序路徑），依 CPU／native threads 與其他 formal producers 保留容量。以 registry 的實作 module 分組，保留同 owner 的快取共用與組內順序，再依原 registry 順序彙整全部結果。每個 validator 完成原始斷言後，立即將診斷結果快照為可傳輸資料：保留純資料、數值與 PASS／FAIL／SKIP，函式／module 等 process-local 物件只保留人讀文字，不重算斷言；循序與並行共用此邊界，後續模組重載或可變診斷更新不得破壞先前 receipt。每次使用全新的 worker coverage 目錄，各程序獨立收集 branch coverage，父程序逐份驗證後合併，只產生一次 JSON；Meta Quality 沿用同輪成果。缺少／重複／錯誤身分的 receipt、程序異常、硬逾時或 coverage 缺損皆不得 PASS。此分工讓大型靜態 AST 快取留在個別 worker，避免拖累父程序的 coverage 報告輸出；不改 Research／Trading／FIT／Replay 身分。
- `source_index.py` 依當前檔案簽章共用單一 process-local 結構索引，包含 imports／exception handlers／top-level bindings、宣告位置與必要時的 branch literal facts。僅需要宣告的 registry／maintenance consumer 不保留函式本文；維護掃描的 imports、宣告、source-shape 與執行分支共用一次解析所得資料，完整 AST 在提取後釋放。循環垃圾回收僅在單檔解析／索引建構期間暫緩，成功或例外退出皆恢復呼叫端設定；明確要求完整 AST 的 consumer 仍可使用原有 AST cache。分支掃描的文字預篩僅排除不可能含目標 literal 的檔案，跳脫字元、隱式字串串接與其前綴／註解一律保留完整 AST 檢查。檔案變更或刪除仍重新驗證，不快取 PASS／FAIL；索引失效測試使用隔離 cache，不清空所在 worker 已建立的正式索引。維護掃描維持 advisory，檢查集合、候選判定、coverage targets 與門檻不因加速改變。
- Strategy Compare 的純計畫顯示／score-reuse synthetic 使用隔離的缺少 teacher 工件狀態，保留 registry dependencies 及真實逐 seed planner，避免每個顯示案例重查全部 Optimizer publication。Teacher inspection／build／PIT、授權與 seed 去重仍由原有獨立行為案例驗證。


### 真實路徑模型的決策條件輸入

- `trade_decision_context.py` 持有決策時可知的 numeric input schema／PIT publication 及還原價縮放不變規則，消費 Target 所綁的 canonical same-seed teacher，不產生第二套參數、訊號或交易結果。
- Architecture descriptor 的 `trade_decision_context` capability 委派通用 data provider；條件缺值採 mask，不改原模型的股票日／Target membership。InceptionTime 的雙線性交互基底讓同日共同政策可以調整不同個股 latent 的排序權重。
- `FitDataDependencies` 在新 capability 出現時才增加 context contract 與 ordered FIT values 證據；原模型的 source／FIT payload 不變。Scientific identities、狀態與實驗範圍由 Experiment Registry 持有，績效只由 Experiment Log 持有。

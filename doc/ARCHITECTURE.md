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
│  ├─ research/                       # Research application boundaries／artifact orchestration／Strategy Compare orchestration
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

- `apps/research.py`：Research 的 Target 驗證、模型、Optimizer、比較及 Audit 單一選單入口。
- `apps/workbench.py`：GUI / workbench 正式入口，`apps/workbench.py` 為單一 GUI 啟用入口；單股檢視為 K 線圖、交易明細、Console 三個分頁。
- `apps/smart_downloader.py`：Market Data 更新、Preflight／Bootstrap 入口。
- `apps/portfolio_sim.py`：投組模擬入口；`apps/vip_scanner.py`：掃描器入口。

## 主要資料夾、持久來源與相容邊界

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

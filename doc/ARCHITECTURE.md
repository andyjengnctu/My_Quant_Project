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
│  ├─ portfolio_run_state.py          # Run-local 持倉／現金／訊號狀態與盤前資金、候選預留
│  ├─ ranking_policy.py               # 排序能力純資料規格；selector、option、dependency role與報表投影同源
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
│  ├─ output_retention.py         # outputs retention 規則與唯讀候選計畫
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
│  │  ├─ oracle_result_contract.py    # Target Truth／Score／Replay／Observation 有效結果身分；來源實作另作 provenance
│  │  ├─ oracle_build_attestation.py  # 當前 Target 計算實作驗證紀錄的唯讀核對；未知實作 BLOCKED
│  │  ├─ oracle_numerical_sources.py  # 實際數值 callable／import／native 閉包投影；支援凍結來源驗證
│  │  ├─ oracle_legacy_admission.py   # 限定原始工件與語意 key 的歷史接納 pins 驗證
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
- Optimizer schedule／adaptation 由 `services/optimizer/strategy_param_contracts.py` 統一建構；參數驗證只回傳正規化候選，寫入與 metadata migration 由 producer 明確發布。Research／teacher／Cleaner 透過公開 inspection 取得工件事實；`core/artifact_inspection.py` 的狀態與原因只用於工程診斷，不進入科學身分。
- `services/optimizer/execution_settings.py` 持有不可變執行設定與有界環境變數 adapter；session 消費同一快照，Windows spawn 與既有環境變數 worker 的相容轉接須在取消或例外後復原，不改 seed／Trial／科學設定。
- Optimizer model mode 由已解析的 run/session policy 明確傳給 fingerprint、policy selection 與結果投影；`core/training_policy.py` 的 ContextVar 只保留有界相容介面，早退、取消、巢狀操作均復原，不能決定另一個 session 的有效契約。
- `core/portfolio_entry_policy.py` 的不可變 `PortfolioExecutionContract` 由入口解析，經 preparation、cache、thread/spawn worker 傳至 Portfolio engine；Daily teacher source 屬 run-local 依賴，Breakout 不繼承外層 Daily mode。
- `core/portfolio_benchmark.py` 的有界 cache 使用完整有效 calendar、起始日前錨點及實際收盤價作 key，計算消費同一輸入快照，回傳結果與 cache 分離；Python 物件 ID 重用、來源修改或 caller 修改回傳 rows 不得改變下一次數值。
- `core/execution_source_dependencies.py` 持有 Replay／Truth 的能力來源依賴圖，包含實際執行的 Python、native C 與設定 owner。排序來源只投影所選規格與相依 primitive；未使用的新規格不改既有計算身分。來源等價證據必須驗證完整依賴閉包，不能以固定舊 hash 代替現碼核對。
- Daily 資料 loader 委派 `filters/breakout_quality/daily_sample_accumulator.py` 保留 target-valid 在前、inference-only 在後的 ticker/date 順序；teacher 以明確 reference 傳遞，context metadata 由單一投影產生。資料排除原因屬執行診斷，不改樣本及 FIT manifest。
- `services/trading/workbench_snapshot.py` 統一 Trading 頁面與帳務中心的快照服務：先執行既有 reconcile／recovery，再以短鎖擷取 account/order/pending，鎖外投影同一 revisions；衍生報表發布不得覆寫較新 revision。`state_snapshot.py` 隔離 worker 讀取且禁止 projection 寫入；UI 使用 `services/workbench_ui/background_tasks.py` 的背景工作與過期結果防護，不持有交易規則。
- `services/trading/state_lock.py` 集中未完成交易 journal 的寫入閘門；所有 public account/order/pending mutation 必須在寫入前檢查，與 active-order 業務政策分離。內部 pending transfer/recovery 僅在已持有相同 lock 且 journal hash 相符的 scope 中提交。
- `services/trading/market_data_consumer.py` 在短 thread/process lock 內核對 expected consumer fingerprint 後發布；日期及同日 overlay lineage 不得倒退。promotion 發生衝突時重新擷取來源與規劃，不能用舊 read model 盲目重試。
- Forward READY 由 `services/breakout_quality/forward_model_source.py` 統一驗證；canonical 與 isolated Strategy Compare／Robustness 以可重驗的 binding 消費相同工件契約。`services/breakout_quality/forward_publication.py` 持有 producer lease、唯一 staging 與可恢復提交，接收 canonical FIT owner 的驗證 callback；FIT 與 evaluation 分別發布，report 失敗不撤銷已驗證的 FIT。
- Ranker 路徑解析由 `services/breakout_quality/ranker_output_paths.py` 持有，feature price 身分由 `services/breakout_quality/feature_price_source.py` 持有，SOP completeness 純驗證由 `filters/breakout_quality/standard_model_sop_contract.py` 持有；既有 producer 模組保留公開轉接介面。Publication 與 inspection 不透過 trainer 取得這些共用契約，避免 FIT、repair 與研究執行服務形成循環引用。

## 正式入口

- `apps/research.py`：Research 單一選單；`[1] Summary`、`[2] Target 驗證`、`[3] 模型訓練／驗證`、`[4] 策略參數最佳化`、`[5] 策略組合比較`，其餘工作依序延後。
- Summary 的 `[V] 一鍵驗證並補齊結果` 依序委派模型 Rolling、策略 Rolling、模型 Rolling Robustness、策略 Rolling Robustness、Target 驗證，再重新讀取 Summary。每步使用目前共用清單與原有 producer 的完整身分驗證／補件規則，精簡顯示執行階段並在完成時輸出原正式結果報表，失敗即停止後續工作。批次入口 `summary verify` 使用相同流程。
- Research 互動確認：模型比較／模型 Robustness 與策略 Robustness 先完成 canonical 前置檢查並列出 REUSE／BUILD 狀態，再提供 Enter 執行、0／EOF 返回。確認後重新解析當下依賴，讀取範圍及來源驗證結果不跨使用者輸入持有；相同計畫不重複列印。Summary 一鍵驗證與非互動批次入口維持既有流程。
- Research 執行狀態：`core/research_status_display.py` 共用 Target 風格的 `S1=實際 Seed` 對照與逐 Seed 矩陣。策略只列 C 編號，模型只列 MR 編號；每格直接消費 canonical Strategy Result／Model Readiness，不從整列摘要或模型分數推斷策略 REUSE。已重用的前置工件合併計數，共通原因合併顯示，待處理原因完整換行保留；此為確認前的操作資訊，不改 Standard SOP、工作計畫或科學身分。
- Research 唯讀載入：Summary／Target preview 在同一次操作內共用 Oracle build 檢查，進入及返回前仍完整驗證來源與證書；Strategy plan 共用歷史 REUSE 鏈的解析結果與路徑觀察，返回前重驗每份來源紀錄及明細路徑的實體位置、存在狀態與 change time。模型比較、模型 Robustness 與策略 Robustness 的完整前置計畫，各自共用一次行情 inventory 讀取範圍，由 `source_inventory.py` 在操作兩端完整核對來源清單與內容見證，發布 price manifest 不使用此讀取快取。這些觀察在完成／失敗後即清空。Pair 與 PIT Fold 檔案 SHA 依既有 `compute_observed_file_sha256` 重用相同實體 revision 的已驗證 digest，工件清單、canonical manifest 與 RUN／REUSE 判定每次重新核對；不跨使用者確認或 producer 執行持有 preview 狀態。
- `config/research_summary.py`：目前研究清單唯一來源；每列宣告既有 Oracle recipe、模型／Positive 模型與策略 binding，或無 DL 對照。`core/research_summary.py` 推導三個工作入口的 Target、DL 與策略集合；各 scientific Registry 仍持有算法、身分與歷史定義。Summary 唯讀消費 canonical evidence，缺工件仍保留列；沒有匹配 Oracle 的模型／策略只顯示實際結果，Oracle 欄位留白。
- `[2] Target 驗證`：正式授權矩陣由 `services/audit/target_oracle_romd.py` 唯一核對；只有註冊的科學 Target × Strategy treatment 進入 Target 驗證畫面、Target RUN／REUSE 計數與人讀 Target 表。No-DL 策略不具 Target，留在 Summary／Strategy Compare；已核准的同 Seed RoMD 差值仍需同經濟家族的 No-DL 控制，該控制以明列且受授權的「非 Target 策略基準依賴」參與內部 Replay 與簽章，但不混入 Target 工作清單。Target 清單與內部控制依賴都從同一 Oracle Registry 投影，簽章 receipt key 不改。Replay 僅按當次已授權 `(seed, target)` RUN 格子動態分派，每個活躍 Seed 一個獨立程序，Seed 內維持原本 Target 順序及同 Seed 控制配對。共用 price-score 真值僅由主程序預先建立，Seed-specific 真值／Portfolio Replay 工件維持各自 Seed 路徑；完整 REUSE 不啟動子程序。並行上限由 `config/audit.py` 的 `TARGET_ORACLE_REPLAY_SEED_WORKERS` 控制（資源設定，不影響科學／工件身分）；不可跨 Seed 重用未驗證工件。
- Target 工件契約：`services/research/oracle_result_contract.py` 分別投影 Truth、Score／cohort、Replay 與完整 Observation 的有效輸入與既有 calculation revisions；來源程式 SHA、commit 與檔案位置屬 producer provenance，不參與有效結果 key。`services/research/oracle_artifact_contract.py` 持有簽章驗證與完成結果索引，`services/audit/oracle_truth_references.py` 持有交易路徑真值的共用消費參照。Oracle 完美分數已替代排名與 Positive 預測，對照模型／策略編號只保留為來源證據；評分資格、時間、Selector、Positive 門檻、入場、參數與費稅仍逐項核對。`services/audit/oracle_observation.py` 只投影 canonical Portfolio 結果，包含 Win Rate 與完整 yearly evidence，不重算帳務。Summary 與 REUSE 共用同一唯讀 inspection；舊格式先驗原始簽章、完整 observation 與限定來源證明，再由 canonical producer 原子發布不可變接納 reference，提交前重驗來源 pins。原始 Replay／NPZ／收據不改寫；缺可重建 Truth NPZ 不撤銷已完成 Replay，也不能宣稱 Truth READY。數值真值只由 RUN 消費者載入。真正計算修正只更新唯一 owner 的受影響 calculation revision，失效沿實際依賴傳遞，不另立 scientific identity。
- Target 實作驗證：`services/research/oracle_build_attestation.py` 核對當前數值來源閉包與其計算版本。來源未經驗證時回報 `IMPLEMENTATION_UNVERIFIED`／BLOCKED，列出來源或計算版本差異，不把全表降成 RUN。`apps/run_bundle.py` 維持 stage → commit → 凍結 ZIP → formal test，再以結構等價或獨立 baseline/current parity 證據完成驗證紀錄發布；formal PASS 本身不代表同版本計算等價。現行紀錄位於 `outputs/local_regression/oracle_contract/current.json`，不進 Git，與交付 ZIP 內紀錄 bytes 相同；封包 finalization 只加入工程證據並核對凍結來源，失敗保留原 commit／ZIP 且回報失敗。主交付 ZIP 保留現行紀錄與工程證據 manifest；review／parity／legacy audit 與當次 formal 的完整原始 bytes，合併為同名且有內容雜湊的標準 tar.xz 證據附件，逐件核對 SHA，歸檔時與主 ZIP 一起保留。一般 runtime 唯讀 REUSE 只需主 ZIP；在全新解包專案進行開發再認證前，依 manifest 將完整證據附件原樣解至專案根目錄，再使用既有 run_bundle 入口。完整基準來源 ZIP 在建立證明時驗證，保留為外部稽核原件及 SHA，不重複嵌入交付包，也不累加歷次 formal ZIP。解包後的 runtime 唯讀檢查不依賴外部基準 ZIP。歷史證據與原始收據保留，不在正式 runtime 持續累積來源 SHA 通行名單。
- Oracle 來源等價以已簽署的策略績效、逐年結果與交易證據為範圍，同時驗證 producer 及 consumer 閉包。Benchmark cache 僅影響未被 Oracle 消費的報表欄位時，不使該 Oracle 收據失效；此證明不適用於完整 Portfolio／benchmark 報表。新的來源或消費語意必須保留實際來源身分，不能沿用舊證書；重用檢查不修改原收據，也不觸發 Portfolio 重算。
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
- REUSE 狀態載入：PIT Fold 雜湊檢查沿用 Research CPU 預算並行讀取獨立檔案，錯誤仍按 manifest 順序回報；score／Audit 重用相同實體 revision 的已驗證 SHA。`strategy_score_projection.py` 的本機衍生快取只保存各 score 欄位投影雜湊，讀取仍核對實際 CSV SHA、完整本機 runtime 程式內容與 Python／pandas／NumPy 版本；損壞、缺失或版本不同即由原資料 loader 重算，不保存 READY 判定。快取位於 filter output 下的 `score_projection_cache/`，不是模型／Replay 真理或新的 scientific identity。Strategy plan 只有在輸入改變時才重做同次執行的第二遍 pair 搜尋。Summary 只搜尋畫面需要的策略，取得各項最新合法結果後停止讀取更舊報表；多 scope Oracle preview 共用行情內容證據並在返回前重驗。Oracle 語法投影只複製移除 docstring 所需的 AST 節點，完整來源閉包與投影 bytes 維持原契約。
- Strategy 歷史報表搜尋：`strategy_compare_reuse.py` 在單次 canonical plan 內共用唯讀 JSON 解析結果，每份報表於讀取前後及計畫返回前核對實體 revision；缺檔亦納入觀察。解析結果在操作完成／失敗時釋放，不跨互動確認持有，也不保存 REUSE 判定；Forward 缺少現行模型時掃描歷史來源不再逐個策略重讀相同報表。CSV 邏輯讀取沿用實體 revision 綁定的小型完整性收據快取，上限 256 份，以免跨歷史 run 搜尋時反覆淘汰重讀；每次讀取仍重新確認檔案 revision。
- Layered Oracle reuse: Oracle receives **perfect independent NetPnL ranks and NetPnL-positive flags**, independent of the real predictor topology, and retains signed per-seed receipts only under verified effective inputs, P16, cash, selector, cutoff, score-membership, and a reviewed identical positive-admission producer. Legacy receipt versions are never directly reused for revised *actual* DL strategy runs. Model PIT manifests and score columns must include the independent classifier output; when either score is missing the shared selector fails closed. Source numerical/input contracts are not a target identity rename. Signed perfect-score Oracle and actual DL results have separate evidence and REUSE checks.
- `apps/workbench.py`：GUI / workbench 正式入口，`apps/workbench.py` 為單一 GUI 啟用入口；單股檢視為 K 線圖、交易明細、Console 三個分頁。
- `apps/smart_downloader.py`：Market Data 更新、Preflight／Bootstrap 入口。
- `apps/cleaner.py`：專案檔案清理的獨立選單，可預覽、執行清理、查閱最近一次紀錄，並以 [4] 備份整個專案至 D 槽及驗證。
  - `services/project_backup.py` 持有整案備份流程：以執行中專案根目錄為來源，每次建立獨立的 `D:/My_Quant_Project_<辨識碼>`。辨識碼為本機日期時間與 8 位隨機碼，資料夾以獨占建立避免覆蓋前次備份；每次獨立保存整份專案。Windows Robocopy 使用 `/E /COPY:DAT /DCOPY:DAT /Z /MT:8 /R:2 /W:2 /NP /TEE /UNILOG`；不使用 `/256`，保留長路徑支援。複製回傳碼 0～7 才進入 `/E /L /COPY:DAT /DCOPY:DAT /R:0 /W:0 /NP /TEE /UNILOG` 的第二次唯讀比對，且只有 0、2 視為通過；失敗或中止不顯示完成、不接續清理。流程不使用 `/MIR`、`/PURGE`、`/MOVE`，保留原專案與既有備份；拒絕來源與目的地相同或互相包含。兩份 Unicode 紀錄放在 D 槽根目錄，檔名為 `My_Quant_Project_backup_<辨識碼>.log`／`My_Quant_Project_verify_<辨識碼>.log`，避免紀錄本身改變比對結果。此驗證沿用 Robocopy 的檔案差異判斷，並非內容雜湊驗證或執行中資料的快照；備份期間應停止修改專案。
  - `core/artifact_lifecycle.py` 持有共用儲存用途契約；`services/cleanup_lifecycle.py` 解析用途、工作狀態、現行設定與跨工作依賴，實際刪除仍由 `services/project_cleaner.py` 統一執行。規則不列舉目前策略、模型或實驗 ID，也不從年齡、檔名或未出現在選單推定研究工件可刪。
  - 儲存依賴採單向分層：`core/storage_paths.py` 持有實體／相對路徑檢查；`core/compare_csv_lineage.py` 解析原 producer 的位置與 REUSE 來源；`core/strategy_compare_storage_evidence.py` 驗證發布紀錄，透過明確注入的 logical-path hasher 核對內容。`core/reused_copy_storage.py` 組合來源讀取、別名收據與發布驗證，保留來源鏈的循環防護；`core/compare_csv_storage.py` 持有寫入／鎖並 re-export 既有讀取 API，底層規則不反向依賴此介面。此分層不改儲存格式、SHA、研究身分或刪除條件。
  - 所有清理目標在刪除邊界再次限制為專案內的相對實體路徑，拒絕絕對路徑、父目錄跳脫、NTFS stream 與連結；保護目錄名稱不分大小寫。通用生命週期的用途紀錄、發布成果與副本替代來源，在內容驗證前固定版本，並在刪除前再次核對，來源異動即保留。
  - Python 快取須有現存的一般原始碼及有效 bytecode header，且最後修改／檔案變更時間均已超過 24 小時；近期建立或重建的快取先保留，避免與執行中的程式反覆刪除／重建。年齡只作冷卻條件，不取代用途驗證；未來時間或無法驗證的時間亦保留。原始碼尚未納入 Git 亦可重建，但 Git 追蹤的快取、缺少原始碼、連結、未知格式及資料／模型／狀態領域仍保留。僅對本次已刪快取所留下的空 `__pycache__` 使用非遞迴 `rmdir`，不清空任意目錄；設定目錄的引用與版本保護仍維持。
  - Producer 在獨立 outputs 工作目錄發布 `artifact_lifecycle.json`，逐檔宣告 `result/evidence/resume/work/cache/replica`、SHA256、`requires` 與保留根；這是工程儲存證據，不能取代或改寫 FIT／Replay／Oracle 的 scientific manifest。`result/evidence/resume` 永遠保留；保留工件的直接、間接及目錄依賴，包含跨工作依賴，也保護原有清理規則的候選。
  - `RUNNING/RESUMABLE` 工作保留。`COMPLETED` 必須驗證已發布成果，才移除無保留依賴的 work／replica；`SUPERSEDED` 另須有替代成果、非空工作身分且未被現行設定引用，才放行舊工作檔或 cache；`ABANDONED` 只能由 producer 明確結案，不能由 cleaner 依年齡猜測。replica 還要驗證保留來源內容相同；有程序身分時，無法確認程序已結束即保留。
  - Producer 與 cleaner 共用生命週期鎖；發布用途或新增依賴必須經此鎖，使用工作區前先發布 RUNNING／RESUMABLE，結案後才發布終態。刪除前重查 Git、SHA、用途紀錄、既有 JSON 引用與目錄快照。設定、用途宣告、來源證據或 adapter 所有工作區變動須重新盤點；一般外部狀態 JSON 與新增目錄可增量重讀，舊引用持續保留、新引用立即加入保護，不能藉刷新取得新的刪除授權。讀取期間變動、無法解析、連結及非預期用途均停止該輪刪除並記錄原因。
  - 診斷封包的 producer 與 cleaner 共用 `core/output_retention.py` 的規則。份數、天數及 `local_regression_retention_mode` 由 `tools/local_regression/manifest.json` 提供：count 依份數、age 依天數、both 須同時超過、either 任一超過。診斷預設使用 count；其它既有 RetentionRule 仍預設 both。只有可驗證的已發布診斷 ZIP 納入排名，仍被保留工件引用者另行保護。
  - `tools/local_regression/common.py` 在診斷封包發布成功後，使用原有 `artifacts_manifest.json` 的逐檔內容證據登記暫存用途，並保留發布程序身分；未列入清單、內容改變或清單不符者不授權清理。所有可拋棄工作檔清除後，其輕量用途紀錄保留供查核，不再永久釘住診斷封包。舊工作沒有足夠證據時不補造完成紀錄；其他 producer 可接入相同契約，不需再修改 cleaner 的 ID 清單。
  - 遷移完成的舊參數仍須核對現行路徑覆寫及正式 preparation 的完整依賴解析，包含 builder 間接使用的上游參數及各來源政策；仍被使用的舊路徑保留。
  - 可重建的 real-path diagnostic NPZ 與 Formal Min outcome NPZ 由各自 producer 驗證 schema、陣列、數值與內容摘要後放行，規則不依模型／實驗 ID 或年齡。讀取、發布與清理共用短期鎖；cache miss 仍由原 producer 計算相同結果，可能增加下一次執行時間。原始 Dataset、FIT、score 與研究結果不適用此規則；僅以非遞迴 rmdir 清除本次刪除後的空 shard 目錄。
  - `services/cleanup_raw_cache.py` 只移除 producer metadata 與內容、存取屬性均相同的 execution raw cache 副本，不載入 pickle。兩端共用原 writer lease，保留來源納入跨規則依賴保護；Trading／promotion 來源與沒有可驗證替代來源的歷史快取保留。raw writer 鎖以永久的小型 OS guard 序列化建立／回收／釋放，只有確認舊程序已結束才回收，釋放時亦核對 owner token，避免等待者誤刪新持有者的鎖。
  - Robustness 的可拋棄工作範圍由原 producer 的 retention policy 提供。清理先驗證完成工件，並保留舊報表補齊所需的 baseline 及來源收據引用的工作檔；未完成工作、正式模型／分數和原始研究證據不以退役或檔案大小判定刪除。
  - Robustness 工作區辨識不固定套用單一目錄深度，舊版輸出位置也須通過相同完成驗證。可清範圍內的空目錄與檔案共用工作鎖及刪除依據；預覽不移除目錄，未知範圍及連結保留。
  - 舊 `FAILED/resumable` attempt 另由 `services/cleanup_superseded_robustness.py` 判斷是否仍有續跑用途：全部 arm／seed 的 scientific-unit 身分及既有／晚到觀察，須已被通過原 owner 完整驗證的正式成果覆蓋，且原 retention 明確允許，才能移除已核對用途的 baseline 工作 CSV／Markdown。舊 FAILED 狀態、科學 JSON、seed／yearly 與 attribution 原件保留。model work 僅允許與通過 canonical owner 驗證的來源身分、收據與 SHA 均相同的 score CSV；所有權重、唯一分數、其他 run 共用的工作區均保留，並取得原 run／model／PIT writer lease。年齡只提示重評，不單獨授權刪除。
  - `services/cleanup_optimizer_scratch.py` 對 Optuna 舊工作庫使用原 producer 的 preflight／fold identity 規則：同工作空間已發布新身分且新庫有完成試驗、舊庫僅有未完成抽樣而無 trial value／intermediate／heartbeat／任何終態結果，才可在兩端原 writer lease 下移除。current identity、validation 派生身分無法確定者、存在 journal 或引用者保留；具有獨有成功或失敗試驗的歷史庫不因久未使用而刪除。
  - 儲存 adapter 共用 cleaner 的刪除邊界、Git 保護、引用閉包與來源保護。只有經該 owner 完整驗證的特定 JSON 欄位及其原值可視為位置宣告；同一路徑出現在其他欄位或其他紀錄仍是保留依賴，不能整份忽略 manifest。取代成果、舊觀察、來源內容、設定及引用版本均須在驗證前固定並在 unlink 前重查。
  - 已發布比較的重複明細，依 producer 原有 `metadata.cache_reused_from` 追溯來源；`baseline_reused`／`baseline_reuse_source` 僅適用 producer 實際複製的 `no_filter_orderable_candidates.csv`，不套用到重新計算的 diagnostics。核對來源及副本的完成發布、逐 bytes 內容、存取屬性、Git 保護與來源鏈未變動後，實際刪除副本。沒有來源紀錄、內容不同、來源不完整或循環時保留，不能只因 CSV 同名或大小相同而刪除。原始 CSV 不壓縮；唯一來源、原始 JSON／Markdown／manifest 與既有研究證據保留。
  - `core/caller_baseline_storage.py` 驗證 positive-threshold selection 的 frozen receipt、selection identity、mask result 與 replay pair 一致後，才允許同一 selection 的 baseline 副本清理；evaluation 不套用。新增 `baseline_storage.json` 只固定被刪副本的 SHA、保留來源與既有結果身分，不改寫科學結果。
  - `core/reused_copy_storage.py` 處理 run manifest 與 run result 都尚未發布的 producer 副本：整 pair REUSE 必須只在 producer 的三個位置欄位與完整發布來源不同；caller-override 的 score-ranking baseline 則須有明確 baseline provenance，且僅適用 `no_filter_orderable_candidates.csv`。兩者都須驗證完整發布來源與相同 CSV bytes，先在 `outputs/project_cleanup/reused_copies/` 原子寫入 target／source／copied_from／SHA 儲存收據，再刪副本；不補造 completed manifest，也不阻止原 producer 正常補齊報表與完成工作。保留來源異動、收據無效或來源鏈循環即拒絕回溯。
  - 重複 CSV 的儲存驗證以該 CSV／pair 為範圍：`core/strategy_compare_storage_evidence.py` 持有共用的 manifest-last 完成邊界，核對本 pair 與 run 結果一致及該 CSV 的既有 SHA，再驗證保留來源；research service 引用同一實作。其他 pair 或其他報表異常不應阻擋這份已驗證副本的移除；完整 run／pair 的 scientific REUSE 與 Runtime Gate 仍執行原有完整性檢查，清理不能將失敗結果改為 READY。來源 revision 必須固定在內容驗證前，不能在 hash 後才取得版本並誤認變動來源仍是已驗證內容。
  - `last_apply.zip`／`last_preview.zip` 除逐檔決策外，可附 `evidence/`：失敗發布驗證的 manifest／run／pair JSON，以及待辨識工作區既有的 manifest／結果摘要。待辨識證據依其工作區占用量優先、使用獨立的 16 MiB 診斷資料預算；未收錄者列於 `review_context.json`。僅擷取既有 JSON，不匯出模型或大型 CSV、不補造完成紀錄，也不將診斷資料的大小／順位用作刪除判斷。略過原因須保留具體失敗欄位。
  - `core/compare_csv_storage.py` 統一解析明細的既有 REUSE 來源鏈及上述儲存收據；Compare 完整性驗證、baseline／PIT／Audit 讀取與後續 REUSE 複製共用此解析。`core/compare_csv_reader.py` 在共用儲存鎖內完成解析與開檔後才交由 pandas 讀取，避免解析後被 cleaner 刪除；JSON 發布亦共用同一鎖。仍依原 SHA 驗證，不新增 scientific identity 或替代研究 manifest。正式與歷史 profile 根目錄深度可不同，均依相同 run／pair 發布邊界辨識。
  - 清理摘要分列刪除檔案的邏輯大小與實際磁碟配置量。原已共用 inode 的副本被刪除時，只減少資料夾的檔名與邏輯大小，不重複計入已釋放磁碟容量；舊版中斷的暫存連結仍可復原。清理與 CSV producer 共用短期發布鎖，中斷後由原選單接續。
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
- Synthetic worker 逐案例傳回完成回執；後續崩潰、硬退出或取消時，`coverage_artifacts/synthetic_completed_receipts.json` 僅保存本輪已完成的診斷。整套仍為 incomplete／FAIL，部分 coverage 不具有發布資格；重試先清除先前診斷。架構呼叫／import guard 以 AST 驗證實際依賴，獨立 fixture 保留違反契約的 mutation 覆蓋。
- Formal suite 的 `tools/local_regression/manifest.json:test_cpu_threads` 統一限制測試子程序的 native CPU pools（預設 1）；synthetic runner 同時限制已載入的 PyTorch，結束／失敗均恢復原執行緒數。只影響隔離測試，不改 Research／Trading 訓練設定。Quick Gate、Consistency、Chain Checks、ML Smoke 依 CPU 數有界並行，Meta Quality 等待全部 producer 完成後消費本輪 coverage；不重用前輪 PASS、不減少 validator、coverage targets 或門檻。
- Consistency 的 synthetic execution 由 `tools/validate/validator_execution.py` 負責；`manifest.json:synthetic_workers` 是子程序數上限（預設 4，1 為原循序路徑），依 CPU／native threads 與其他 formal producers 保留容量。以 registry 的實作 module 分組，保留同 owner 的快取共用與組內順序，再依原 registry 順序彙整全部結果。每個 validator 完成原始斷言後，立即將診斷結果快照為可傳輸資料：保留純資料、數值與 PASS／FAIL／SKIP，函式／module 等 process-local 物件只保留人讀文字，不重算斷言；循序與並行共用此邊界，後續模組重載或可變診斷更新不得破壞先前 receipt。每次使用全新的 worker coverage 目錄，各程序獨立收集 branch coverage，父程序逐份驗證後合併，只產生一次 JSON；Meta Quality 沿用同輪成果。缺少／重複／錯誤身分的 receipt、程序異常、硬逾時或 coverage 缺損皆不得 PASS。此分工讓大型靜態 AST 快取留在個別 worker，避免拖累父程序的 coverage 報告輸出；不改 Research／Trading／FIT／Replay 身分。
- `source_index.py` 依當前檔案簽章共用單一 process-local 結構索引，包含 imports／exception handlers／top-level bindings、宣告位置與必要時的 branch literal facts。僅需要宣告的 registry／maintenance consumer 不保留函式本文；維護掃描的 imports、宣告、source-shape 與執行分支共用一次解析所得資料，完整 AST 在提取後釋放。循環垃圾回收僅在單檔解析／索引建構期間暫緩，成功或例外退出皆恢復呼叫端設定；明確要求完整 AST 的 consumer 仍可使用原有 AST cache。分支掃描的文字預篩僅排除不可能含目標 literal 的檔案，跳脫字元、隱式字串串接與其前綴／註解一律保留完整 AST 檢查。檔案變更或刪除仍重新驗證，不快取 PASS／FAIL；索引失效測試使用隔離 cache，不清空所在 worker 已建立的正式索引。維護掃描維持 advisory，檢查集合、候選判定、coverage targets 與門檻不因加速改變。
- Strategy Compare 的純計畫顯示／score-reuse synthetic 使用隔離的缺少 teacher 工件狀態，保留 registry dependencies 及真實逐 seed planner，避免每個顯示案例重查全部 Optimizer publication。Teacher inspection／build／PIT、授權與 seed 去重仍由原有獨立行為案例驗證。


### 真實路徑模型的決策條件輸入

- `trade_decision_context.py` 持有決策時可知的 numeric input schema／PIT publication 及還原價縮放不變規則，消費 Target 所綁的 canonical same-seed teacher，不產生第二套參數、訊號或交易結果。
- Architecture descriptor 的 `trade_decision_context` capability 委派通用 data provider；條件缺值採 mask，不改原模型的股票日／Target membership。InceptionTime 的雙線性交互基底讓同日共同政策可以調整不同個股 latent 的排序權重。
- `FitDataDependencies` 在新 capability 出現時才增加 context contract 與 ordered FIT values 證據；原模型的 source／FIT payload 不變。Scientific identities、狀態與實驗範圍由 Experiment Registry 持有，績效只由 Experiment Log 持有。


### Controlled learning studies

`config/breakout_quality_history.py` declares model training-history extensions. `services/breakout_quality/training_history.py` composes canonical early-history and unchanged original-label materializers; parameter preparation remains a registered dependency of the existing Research artifact planner. The current model/strategy bindings live in `config/research_summary.py`, which projects membership into the existing model comparison, Strategy Compare, Robustness, Target and Summary workflows. No separate learning-study menu, orchestration engine or report schema is used. New history bounds and ordered FIT truth participate in model identity; unchanged controls retain their existing FIT/Replay contracts and canonical validation/reuse rules.


### Research 同日百分位分數組合

`core/rank_score_composition.py` 持有未擬合的 frozen score composition 契約；`filters/breakout_quality/rank_score_composition.py` 透過原 PIT／Forward admission owner 驗證全部來源、同 Seed 及資訊 cutoff，在共同 feature-eligible 股票日母體上呼叫既有百分位 primitive。原 `strategy_compare_execution` 組裝 mode／seed 路徑，`strategy_compare_runtime_contract` 每次 Replay 準備一次表，原 runtime score lookup 在 unchanged information-date／selector 邊界消費；不另建 Replay、FIT 或資料真理。零權重控制仍保留全部共同母體依賴。來源列舉與 score projection／signature 同步納入所有組成模型，因此任何來源變更會使相關 Replay 失效。`config/research_summary.py` 的 `rank_models` binding 表示既有模型組成，原 Summary 欄位顯示權重，沒有獨立模型 ρ；未核准 perfect-score 合成的 Oracle 明列未解，不借用其他 Target 結果。

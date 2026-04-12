# 專案架構說明

本文件只記錄穩定子系統、正式入口、依賴方向與必要 shipped 模組索引；操作步驟看 `doc/CMD.md`，formal contract 與狀態看 `doc/TEST_SUITE_CHECKLIST.md`。

## 穩定檔案樹

```text
project/
├─ apps/
│  ├─ ml_optimizer.py                 # 參數最佳化正式入口（薄入口）
│  ├─ portfolio_sim.py                # 投組模擬正式入口（薄入口）
│  ├─ smart_downloader.py             # 資料下載正式入口（薄入口）
│  ├─ package_zip.py                  # 專案打包正式入口
│  ├─ test_suite.py                   # 一鍵測試正式入口（reduced）
│  ├─ vip_scanner.py                  # 掃描器正式入口（薄入口）
│  └─ workbench.py                    # GUI 工作台正式入口（薄入口）
├─ config/
│  ├─ training_policy.py              # 訓練政策與 selection gate
│  └─ execution_policy.py             # 資金、費用與 runtime 執行預設
├─ core/
│  ├─ config.py                       # 相容 façade；穩定匯出設定常數與參數契約
│  ├─ strategy_params.py              # breakout + training gate + execution 聚合參數契約
│  ├─ capital_policy.py               # 單股/投組/scanner 共用資金與 sizing 規則
│  ├─ exact_accounting.py             # 正式整數 ledger / cost-basis allocation / tick 正規化單一真理來源
│  ├─ backtest_core.py                # 單股回測總控 façade
│  ├─ portfolio_engine.py             # 投組 timeline 總控 façade
│  ├─ model_paths.py              # models 目錄與 best_params 路徑解析 helper
│  ├─ output_paths.py             # outputs/<category> 目錄正規化與建立 helper
│  ├─ output_retention.py         # outputs retention 雙門檻清理 helper
│  └─ display.py                      # 顯示 façade
├─ doc/
│  ├─ PROJECT_SETTINGS.md             # 專案最高優先規則
│  ├─ TEST_SUITE_CHECKLIST.md         # formal test suite 主表與索引
│  ├─ GPT_DELIVERY_CHECKLIST.md       # GPT 交付前操作檢查表
│  ├─ ARCHITECTURE.md                 # 本檔
│  └─ CMD.md                          # 常用指令與操作說明
├─ models/
│  ├─ all_best_params_1.json         # 特定評分口徑下的最佳參數紀錄
│  ├─ all_best_params_2.json         # 特定評分口徑下的最佳參數紀錄
│  ├─ all_best_params_3.json         # 歷史最佳參數或不同批次最佳化輸出
│  └─ best_params.json               # 目前主要使用的最佳參數檔
└─ tools/
   ├─ downloader/                     # 資料下載子系統
   ├─ optimizer/                      # 參數最佳化子系統
   ├─ portfolio_sim/                  # 投組模擬子系統
   ├─ scanner/                        # 掃描器子系統
   ├─ trade_analysis/                 # 單股 trade-analysis 子系統
   ├─ validate/                       # validate / synthetic / real-case 驗證子系統
   ├─ local_regression/               # reduced formal orchestrator
   └─ workbench_ui/                   # GUI 子系統
```

## 關鍵 shipped 模組索引

### `tools/trade_analysis/`

```text
   │  ├─ history_snapshot.py          # 單股分析歷史績效 snapshot / payoff / asset-growth helper
```

- `tools/trade_analysis/`：單股 trade-analysis 子系統；`tools/trade_analysis/trade_log.py` 為單股 trade-analysis 正式入口。
- 為維持相容性，保留 legacy `run_debug_*` API 名稱，同時提供 canonical `run_trade_analysis` / `run_trade_backtest` / `run_prepared_trade_backtest` / `run_ticker_analysis` aliases。

### `tools/validate/`

```text
      ├─ meta_contracts.py            # TEST_SUITE_CHECKLIST / 文件 / registry 的 markdown 與 AST contract helper
      ├─ synthetic_contract_cases.py    # synthetic 工件 lifecycle / GUI / shared helper 契約案例
      ├─ synthetic_history_cases.py     # synthetic PIT / history filter / compounding-capital 案例
      ├─ synthetic_guardrail_cases.py   # synthetic project-settings / config / exception / fallback guardrail 案例
      ├─ synthetic_regression_cases.py  # synthetic rerun-repeatability / cache-isolation / bundle-repeatability 案例
      ├─ synthetic_portfolio_common.py  # synthetic 投組案例共用 builder / helper
      ├─ synthetic_frame_utils.py       # synthetic DataFrame / row / assertion helper
      ├─ synthetic_case_builders.py     # synthetic chart / GUI / payload case builder helper
      ├─ synthetic_display_cases.py     # synthetic 顯示契約案例：scanner header / dashboard / display re-export output sanity
      ├─ synthetic_reporting_cases.py   # synthetic 報表契約案例：validate / portfolio / test suite summary schema
```

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

- `apps/`：正式入口層，只從對應子系統 façade 匯入公開介面。
- `core/`：核心規則、帳務、價格、統計、path 與共用 helper；不得放 UI orchestration 或 validate 腳本。
- `tools/`：下載、最佳化、單股分析、validate、local regression 與 GUI 子系統。
- `config/`：共用政策與執行預設。
- `models/`：最佳參數檔與模型相關輸入。
- `doc/`：治理文件；上層原則在 `PROJECT_SETTINGS.md`，formal 主表在 `TEST_SUITE_CHECKLIST.md`，操作步驟在 `CMD.md`。

## GUI / workbench

- `apps/workbench.py` 為單一 GUI 啟用入口。
- `tools/workbench_ui/workbench.py` 負責主視窗與 panel registry；`single_stock_inspector.py` 負責單股回測檢視頁籤。
- `tools/workbench_ui/single_stock_inspector.py` 的 K 線檢視中，交易明細與 Console 改以獨立分頁承接。
- GUI 檢視層中，交易明細與 Console 以獨立分頁承接。

## 依賴方向

- 上層呼叫下層：`apps/ -> tools/ -> core/`。
- `core/` 不反向依賴 `tools/` 或 `apps/`。
- `tools/*/__init__.py` 與 façade 檔維持穩定公開介面；子模組可再細拆，但外部匯入路徑應盡量不變。
- formal test chain 只由 `apps/test_suite.py` 與 `tools/local_regression/formal_pipeline.py` 收斂；`doc/GPT_DELIVERY_CHECKLIST.md` 不納入本地 formal 驗證。

## 輸出與相容邊界

- 所有工具輸出皆落在 `outputs/<category>/`；輸出位置與 retention 規則以 `core/output_paths.py`、`core/output_retention.py` 與 `doc/CMD.md` 為準。
- `outputs/debug_trade_log/` 為 `trade_analysis` 相容輸出目錄；為維持相容性，暫沿用 `debug_trade_log` 這個 legacy 名稱。
- `debug_trade_log`（trade_analysis legacy output dir）屬既有工具鏈相容邊界，不代表子系統角色仍是 debug-only。

## 維護原則

- 架構文件只保留穩定子系統、正式入口、依賴方向與必要 shipped 模組索引；高波動操作細節移至 `doc/CMD.md`，formal 細部契約移至 `doc/TEST_SUITE_CHECKLIST.md`。
- 需逐字比對的文字只保留 canonical 名稱、正式入口、section heading 與最小必要 fragment；避免把高波動敘述做成 exact-string contract。
- 若模組責任、正式入口、共享資料流或 shipped 關鍵模組有變動，必須同步更新本檔、`doc/CMD.md` 與相關 formal contract。

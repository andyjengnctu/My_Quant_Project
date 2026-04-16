# 架構概覽

本文件只保留穩定分層、正式入口、依賴方向與共享邊界。操作步驟看 `doc/CMD.md`；formal contract 與狀態看 `doc/TEST_SUITE_CHECKLIST.md`。

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
│  ├─ model_paths.py              # models 目錄與 champion/run_best 路徑解析 helper
│  ├─ output_paths.py             # outputs/<category> 目錄正規化與建立 helper
│  ├─ output_retention.py         # outputs retention 雙門檻清理 helper
│  └─ display.py                      # 顯示 façade
├─ doc/
│  ├─ TEST_SUITE_CHECKLIST.md         # formal test suite 主表與索引
│  ├─ ARCHITECTURE.md                 # 本檔
│  └─ CMD.md                          # 常用指令與操作說明
├─ models/
│  ├─ all_best_params_1.json         # 特定評分口徑下的最佳參數紀錄
│  ├─ all_best_params_2.json         # 特定評分口徑下的最佳參數紀錄
│  ├─ all_best_params_3.json         # 歷史最佳參數或不同批次最佳化輸出
│  └─ champion_params.json               # 目前主要使用的最佳參數檔
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

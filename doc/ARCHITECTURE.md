# 架構概覽

本文件只保留穩定分層、正式入口、依賴方向與共享邊界。操作步驟看 `doc/CMD.md`；formal contract 與狀態看 `doc/TEST_SUITE_CHECKLIST.md`。

## 分層

- `apps/`：正式入口層。
- `core/`：核心規則、帳務、價格、統計、共享 path 與共用 helper。
- `tools/`：下載、最佳化、投組模擬、掃描、單股分析、validate、local regression 與 GUI 子系統。
- `config/`：共用政策與執行預設。
- `models/`：最佳參數檔與模型相關輸入。
- `doc/`：架構、常用指令與 formal checklist 文件。

## 正式入口

- `apps/test_suite.py`：日常一鍵測試正式入口。
- `tools/local_regression/formal_pipeline.py`：formal 步驟單一真理來源。
- `apps/ml_optimizer.py`：optimizer 正式入口。
- `apps/portfolio_sim.py`：投組模擬正式入口。
- `apps/smart_downloader.py`：下載器正式入口。
- `apps/vip_scanner.py`：scanner 正式入口。
- `apps/workbench.py`：GUI / workbench 正式入口。
- `tools/trade_analysis/trade_log.py`：單股 trade-analysis 正式入口。

## 子系統責任

- `optimizer`：參數搜尋、最佳化輸出與結果整理。
- `portfolio_sim`：投組模擬、統計與報表。
- `scanner`：候選掃描、排序與 issue log。
- `trade_analysis`：單股分析、圖表與交易明細輸出。
- `validate`：formal contract、schema、synthetic 與 real-case 驗證。
- `local_regression`：reduced formal orchestrator 與 bundle 產出。
- `workbench_ui`：GUI 主視窗與單股檢視頁面。

## 依賴方向

- 分層呼叫固定為 `apps -> tools -> core`。
- `core/` 不反向依賴 `tools/` 或 `apps/`。
- 正式 test chain 只由 `apps/test_suite.py` 與 `tools/local_regression/formal_pipeline.py` 收斂。

## 共享邊界

- 所有工具輸出皆落在 `outputs/<category>/`。
- 輸出位置與 retention 規則由 `core/output_paths.py`、`core/output_retention.py` 管理。
- 共享欄位、canonical 名稱與正式 schema 變動時，producer、consumer、顯示、驗證與文件必須同輪同步。

## 維護原則

- 本檔只承接穩定子系統、正式入口、依賴方向與共享邊界。
- 高波動操作細節移至 `doc/CMD.md`；formal 細部契約移至 `doc/TEST_SUITE_CHECKLIST.md`。
- 不以 exact file-tree、helper 長清單、局部 alias 說明或暫時演進敘事作為本檔主要承載面。

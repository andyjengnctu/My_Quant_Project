# 專案架構說明

本文件說明目前專案檔案樹、各檔案用途與依賴原則。

## 檔案樹

```text
project/
├─ .gitignore                         # Git 忽略規則
├─ apps/
│  ├─ __init__.py                     # apps 套件初始化檔
│  ├─ ml_optimizer.py                 # 參數最佳化正式入口（薄入口）
│  ├─ portfolio_sim.py                # 投組模擬正式入口
│  ├─ smart_downloader.py             # 資料下載正式入口（薄入口）
│  ├─ validate_consistency.py         # 一致性驗證正式入口
│  └─ vip_scanner.py                  # 掃描器正式入口
├─ core/
│  ├─ __init__.py                     # 套件初始化檔
│  ├─ v16_buy_sort.py                 # 買入候選排序邏輯
│  ├─ v16_config.py                   # dataclass、參數預設、參數驗證、共用設定
│  ├─ v16_core.py                     # 單股策略核心邏輯、訊號、進出場與回測主流程
│  ├─ v16_data_utils.py               # OHLCV 清理與共用資料工具
│  ├─ v16_dataset_profiles.py         # 資料集模式/CLI/ENV 解析與路徑切換單一入口
│  ├─ v16_display.py                  # 顯示與輸出格式整理
│  ├─ v16_log_utils.py                # logging 與輸出輔助工具
│  ├─ v16_params_io.py                # 參數讀寫、json 載入/匯出
│  ├─ v16_portfolio_engine.py         # 投組核心 timeline、候選池、資金/名額保留與最終整合
│  ├─ v16_portfolio_fast_data.py      # 投組快取市場資料、mark-to-market 與歷史 PIT 統計索引單一口徑
│  ├─ v16_portfolio_stats.py          # 投組曲線/年度/年化統計與分數計算單一口徑
│  └─ v16_runtime_utils.py            # 執行期共用工具：ProcessPool 啟動方法、Asia/Taipei 時間工具
├─ doc/
│  ├─ ARCHITECTURE.md                 # 本檔；檔案樹、用途與依賴原則說明
│  ├─ CMD.md                          # 常用指令與操作說明
│  ├─ FINMIND_API_TOKEN.md            # API token 說明
│  ├─ PROJECT_SETTINGS.md             # 專案最高優先規則文件
│  └─ ToDo.md                         # 待辦事項與後續整理筆記
├─ data/
│  ├─ tw_stock_data_vip/              # 專案內保留的資料佔位與名單檔
│  └─ tw_stock_data_vip_reduced/      # 專案內保留的資料佔位
├─ models/
│  ├─ v16_all_best_params (LOG_R2).json  # 特定評分口徑下的最佳參數紀錄
│  ├─ v16_all_best_params (RoMD).json    # 特定評分口徑下的最佳參數紀錄
│  ├─ v16_all_best_params_3.json         # 歷史最佳參數或不同批次最佳化輸出
│  └─ v16_best_params.json               # 目前主要使用的最佳參數檔
├─ requirements/
│  ├─ export_requirements_lock.py     # 輸出 requirements lock 的輔助腳本
│  ├─ requirements-lock.txt           # 鎖版本套件清單
│  └─ requirements.txt                # 主要相依套件清單
└─ tools/
   ├─ __init__.py                     # tools 套件初始化檔
   ├─ downloader/
   │  ├─ __init__.py                  # downloader 子套件初始化檔
   │  └─ main.py                      # 下載主流程、名單快篩與資料輸出
   ├─ debug/
   │  ├─ __init__.py                  # debug 子套件初始化檔
   │  ├─ backtest.py                  # 交易回放與明細列建構
   │  ├─ reporting.py                 # debug 報表輸出與虧損摘要
   │  └─ trade_log.py                 # 交易除錯入口、資料集解析與對外包裝
   ├─ optimizer/
   │  ├─ __init__.py                  # optimizer 子套件初始化檔
   │  ├─ main.py                      # 最佳化主流程、study 控制與歷史最佳還原
   │  ├─ prep.py                      # optimizer 原始資料快取、worker 預處理與平行/回退流程
   │  ├─ profile.py                   # optimizer profiling CSV/JSON 輸出與摘要
   │  └─ study_utils.py               # trial / study / 參數還原共用工具
   └─ validate/
      ├─ __init__.py                  # validate 子套件初始化檔
      ├─ checks.py                    # validate 共用檢查結果、統計比對與 payload 建構
      ├─ main.py                      # 一致性驗證主模組；資料集解析、真實 ticker 編排與總控入口
      ├─ reporting.py                 # validate 報表輸出與 console summary
      ├─ synthetic_cases.py           # synthetic case 驗證編排與 suite 入口
      ├─ synthetic_fixtures.py        # synthetic 測試資料與案例生成
      ├─ tool_adapters.py             # validate 對 apps/debug 工具的動態載入與 smoke 檢查
      └─ trade_rebuild.py             # trade log / completed trade 重建工具
```

## 分層原則

- `apps/`：正式執行入口，只負責 CLI、流程組裝與執行期 bootstrap，不得在入口層重寫核心交易規則；`apps/ml_optimizer.py` 現為薄入口，最佳化主流程移至 `tools/optimizer/`。
- `tools/validate/`：一致性驗證子系統，已拆成 `checks.py`、`tool_adapters.py`、`synthetic_cases.py`、`synthetic_fixtures.py`、`trade_rebuild.py`、`reporting.py`；`main.py` 僅保留資料集解析、真實 ticker 驗證編排與總控流程。
- `tools/debug/`：交易除錯子系統；`trade_log.py` 保留 CLI 與資料集解析，`backtest.py` 專責正式核心邏輯回放與明細列建構，`reporting.py` 專責 Excel 匯出與虧損摘要。
- `core/`：核心規則與共用計算，應作為單一真理來源；本輪先把 `v16_portfolio_engine.py` 的快取市場資料/PIT 統計索引與曲線/年度/年化統計函式抽到 `v16_portfolio_fast_data.py`、`v16_portfolio_stats.py`，保留 `run_portfolio_timeline()` 作為投組總控。
- `tools/`：除錯、驗證與開發輔助工具；可呼叫核心邏輯，但不得成為正式交易規則唯一來源。
- `doc/`：文件與規則說明，以 `PROJECT_SETTINGS.md` 為最高優先規則文件。
- `models/`：參數結果與最佳化輸出，不放正式交易邏輯。
- 執行期資料集預設優先使用 `/data/`；若執行環境不存在 `/data/`，則自動退回 `project/data/`。因此 Linux 可直接使用 `/data/...`，Windows 等無 `/data` 的環境可直接使用專案根目錄下的 `data/...`。
- `requirements/`：環境相依與版本鎖定，不放商業邏輯。

## 依賴方向

- `core/` 不得依賴 `tools/`、`apps/` 或純顯示用途程式。
- `apps/` 與 `tools/` 可依賴 `core/`，但不得在外層重寫核心規則。
- 顯示、輸出、CLI、下載流程不得反向影響核心交易規則。
- 參數驗證、交易規則、統計口徑應集中管理，不得在多處重複實作。

## 維護要求

- 新增、刪除、移動、拆分、合併檔案，或調整模組責任與依賴方向時，必須同步更新本文件與 `doc/CMD.md`。
- 若本文件與實際程式不一致，應優先修正文件，不得放任過期。

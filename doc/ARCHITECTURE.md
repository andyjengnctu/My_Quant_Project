# 專案架構說明

本文件說明目前專案檔案樹、各檔案用途與依賴原則。

## 檔案樹

```text
project/
├─ .gitignore                         # Git 忽略規則
├─ v16_ml_optimizer2.py               # 參數最佳化主程式
├─ v16_portfolio_sim.py               # 投組模擬主程式
├─ v16_vip_scanner.py                 # 掃描器主程式
├─ vip_smart_downloader.py            # 資料下載主程式
├─ core/
│  ├─ __init__.py                     # 套件初始化檔
│  ├─ v16_buy_sort.py                 # 買入候選排序邏輯
│  ├─ v16_config.py                   # dataclass、參數預設、參數驗證、共用設定
│  ├─ v16_core.py                     # 單股策略核心邏輯、訊號、進出場與回測主流程
│  ├─ v16_data_utils.py               # OHLCV 清理與共用資料工具
│  ├─ v16_dataset_profiles.py         # 測試/驗證資料集模式與路徑切換單一入口
│  ├─ v16_display.py                  # 顯示與輸出格式整理
│  ├─ v16_log_utils.py                # logging 與輸出輔助工具
│  ├─ v16_params_io.py                # 參數讀寫、json 載入/匯出
│  ├─ v16_portfolio_engine.py         # 投組核心流程、候選池、資金/名額保留、timeline 與統計口徑
│  └─ v16_runtime_utils.py            # 執行期共用工具：ProcessPool 啟動方法、Asia/Taipei 時間工具
├─ doc/
│  ├─ ARCHITECTURE.md                 # 本檔；檔案樹、用途與依賴原則說明
│  ├─ CMD.md                          # 常用指令與操作說明
│  ├─ FINMIND_API_TOKEN.md            # API token 說明
│  ├─ PROJECT_SETTINGS.md             # 專案最高優先規則文件
│  └─ ToDo.md                         # 待辦事項與後續整理筆記
├─ tw_stock_data_vip/                 # 完整版測試資料集
├─ tw_stock_data_vip_reduced/         # 縮減版測試資料集（供 validate 預設使用）
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
   ├─ debug_trade_log.py              # 交易除錯工具
   └─ validate_v16_consistency.py     # 一致性驗證工具與 synthetic case 驗證主程式
```

## 分層原則

- `core/`：核心規則與共用計算，應作為單一真理來源。
- 根目錄主程式：負責流程組裝與執行入口，不得重寫核心交易規則。
- `tools/`：負責除錯、驗證、一致性檢查，可呼叫核心邏輯，但不得成為正式邏輯唯一來源。
- `doc/`：文件與規則說明，以 `PROJECT_SETTINGS.md` 為最高優先規則文件。
- `models/`：參數結果與最佳化輸出，不放正式交易邏輯。
- `requirements/`：環境相依與版本鎖定，不放商業邏輯。

## 依賴方向

- `core/` 不得依賴 `tools/`、主程式入口或純顯示用途程式。
- 主程式與工具可依賴 `core/`，但不得在外層重寫核心規則。
- 顯示、輸出、CLI、下載流程不得反向影響核心交易規則。
- 參數驗證、交易規則、統計口徑應集中管理，不得在多處重複實作。

## 維護要求

- 新增、刪除、移動、拆分、合併檔案，或調整模組責任與依賴方向時，必須同步更新本文件。
- 若本文件與實際程式不一致，應優先修正文件，不得放任過期。

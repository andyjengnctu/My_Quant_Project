# Optimizer 升級暫行規劃（Temporary）

> 目的：先以最小可落地版補上「避免 overfit 的獨立驗證鏈」，不一次大改 optimizer 主目標函數。

## 0. 本輪原則
- 先做最小可落地版（MVP），避免一次改太大導致搜尋口徑劇烈漂移。
- 先補「獨立 OOS 驗證報表」，再決定是否把 OOS 指標正式納入 objective / 升版門檻。
- 暫不直接改動 best trial 的主搜尋分數公式；先增加獨立 walk-forward 驗證輸出。

## 1. 四個面向

### 1) 訓練 / 測試取樣方式
- [x] 收斂：MVP 採 expanding walk-forward。
- [x] 初版設定：最小訓練窗 8 年、OOS 視窗 6 個月、逐窗往前滾。
- [x] 主設定已調整為 `train_start_year = 2012`、`min_train_years = 8`（優先補 2020 shock / 2022 bear regime 覆蓋）。
- [ ] 後續再評估是否改為 rolling walk-forward。

### 2) 驗證指標
- [x] 收斂：MVP 先看下列核心指標。
  - OOS 視窗報酬率
  - OOS 視窗分數（沿用既有 calc_portfolio_score）
  - OOS 視窗 MDD
  - OOS 年化交易次數
  - OOS 保留後買進成交率
- [x] 收斂：總結先用
  - 視窗分數中位數
  - 視窗報酬率中位數
  - 最差視窗報酬率
  - 最大視窗 MDD
- [ ] 後續再決定是否把 worst-window / median-score 正式納入 objective。
- [x] 先新增「報表級升版門檻」：僅輸出 pass/watch/fail，不阻擋匯出。

### 3) 驗證報表
- [x] 收斂：MVP 先輸出
  - summary JSON
  - 視窗明細 CSV
  - markdown 報表
- [x] 收斂：MVP regime 先用 benchmark(0050) 半年報酬粗分
  - up: >= +8%
  - flat: (-8%, +8%)
  - down: <= -8%
- [ ] 後續再細分 high-vol / event-driven regime。

### 4) 升版規則
- [x] 收斂：本輪先不自動用 walk-forward 結果決定 best_params 是否匯出。
- [x] 收斂：本輪先把 walk-forward 結果當成獨立驗證報表。
- [x] 報表級 MVP 升版門檻（僅判讀，不阻擋匯出）
  - median_window_score > 0
  - worst_ret_pct >= -8%
  - flat median_score >= 0（若 flat 視窗存在）
  - down 視窗數 >= 1 否則只可視為 regime 覆蓋不足
- [ ] 下一階段再考慮是否將門檻接入正式升版流程

## 2. 本輪 MVP 實作項目
- [x] 新增暫時規劃檔（本檔）。
- [x] 新增 walk-forward 評估模組。
- [x] 在 optimizer 匯出 best_params 後，自動產生 walk-forward 驗證報表。
- [x] 報表落點遵守 outputs/ml_optimizer/。
- [x] GPT 端自檢：語法、匯入、輸出鏈、正式入口鏈。

## 3. 下一階段候選
1. 把 walk-forward summary 指標納入 study callback 顯示。
2. 把 median / worst-window 指標納入升版門檻。
3. 再評估是否要把 objective 從純 in-sample 改為 in-sample + OOS 混合分數。


### 目前主設定
- [x] Walk-Forward 主設定：2012 + 8（train_start_year=2012, min_train_years=8）。
- [x] 單版報表：輸出 challenger 的 WF 報表與 gate。
- [x] 升級 MVP：輸出 Champion / Challenger 比較報表（先不自動升版）。
- [ ] 後續：將 compare gate 接成正式升版阻擋規則。

# Optimizer 升級暫行規劃（Temporary）

> 目的：以最小但正確的主線升級 optimizer，先把「訓練區間切乾淨 + OOS/WF 主導 trial 評分」落地；先不處理定期重訓流程與 validator。顯示層則在 P0 落地後，優先以 workbench 正規呈現來檢視升級效果。

---

## 0. 本輪收斂共識

### 0.1 目前策略本質
- 目前 optimizer 產出的是固定策略參數 `.json`，不是另有需重訓權重的第二層模型。
- 因此現階段主問題不是「定版後如何重訓權重」，而是「如何用乾淨的訓練 / 測試流程選出一組固定 `.json`」。

### 0.2 現階段先不處理的範圍
- [ ] 先不把「定期重訓 / 每 K 天重選參」納入這輪主線。
- [ ] 先不新增第二種 optimizer。
- [ ] 先不做 validator。
- [ ] 先不把 console 顯示做得更複雜；顯示層優先方向是 P0 後用 workbench 正規呈現。

### 0.3 現階段主線定義
- [x] 現階段主線是：**先訓練出一組固定參數，再用多個 OOS windows 驗這一組參數的穩定性**。
- [x] 這個穩定性分數必須對應到同一組固定參數；因此本輪不採「每個 window 再訓一組新參數」的流程。
- [x] 若 trial 都會跑 OOS/WF 穩定性評分，則 `base_score` 不再適合作為主 objective，只保留 sanity gate / debug / tie-break 意義。

---

## 1. 升級主線（P0）

### 1) 訓練 / 測試取樣方式
- [x] MVP 已採 expanding walk-forward。
- [x] 已有主設定：`train_start_year = 2012`、`min_train_years = 8`、`test_window_months = 6`。
- [ ] **P0：把主搜尋資料從目前的全吃改成明確 train 區間。**
  - 目標：`base_score` 主回測不再使用 `2012 ~ 最新日` 全資料。
  - 先收斂為固定切法：
    - search train：`2012 ~ search_train_end`
    - WF/OOS：`wf_start ~ wf_end`
  - 初步目標口徑：
    - search train：`2012 ~ 2019`
    - WF/OOS：`2020 ~ 最新已納入驗證區間`

### 2) objective / trial 評分
- [x] 目前主 objective 仍以 `base_score` 為主，WF 主要用在報表 / gate / compare。
- [ ] **P0：在現有 optimizer 內加入 `objective_mode`，不新增第二種 optimizer。**
  - `legacy_base_score`
  - `wf_gate_median`
- [ ] **P0：每個 trial 都跑固定參數版的 WF 穩定性評分。**
- [ ] **P0：trial 級 hard gate 採現有 WF quality gate。**
  - `median_window_score >= gate_min_median_score`
  - `worst_ret_pct >= gate_min_worst_ret_pct`
  - `flat_median_score >= gate_min_flat_median_score`（若 flat 視窗存在）
- [ ] **P0：`quality_gate fail` 直接淘汰 trial。**
- [ ] **P0：`final_score = median_window_score`。**
- [ ] **P0：`base_score` 降級為 sanity gate / debug / tie-break，不再主導排名。**
- [ ] 先不做「混權重大公式」：
  - 不先做 `base_score + worst-window + median + mdd penalty` 這種加權混分。
  - 避免新增 magic number 與 objective overfit。

### 3) best trial / run_best 決策
- [ ] **P0：新增 mode-aware best-trial resolver，不再直接依賴 `study.best_trial`。**
- [ ] `wf_gate_median` 模式下的排序收斂：
  1. `wf_quality_gate_status == pass`
  2. `wf_median_window_score` 由大到小
  3. `wf_worst_ret_pct` 由大到小
  4. `wf_flat_median_score` 由大到小
  5. `base_score` 由大到小
  6. `trial.number` 由小到大
- [ ] **P0：`run_best_params.json` 改由新 resolver 產出。**
- [x] compare gate / promote gate 仍維持單一路徑，不另分叉一套 OOS optimizer 流程。

### 4) trial metadata / 後續顯示共用欄位
- [ ] **P0：每個 trial 都寫入完整 WF attrs，供 resolver / callback / workbench 共用。**
  - `base_score`
  - `wf_window_count`
  - `wf_median_window_score`
  - `wf_worst_ret_pct`
  - `wf_flat_median_score`
  - `wf_max_mdd`
  - `wf_upgrade_status`
  - `wf_quality_gate_status`
  - `wf_coverage_gate_status`

---

## 2. 驗證指標（沿用 / 收斂）

### 1) OOS 視窗指標
- [x] OOS 視窗報酬率
- [x] OOS 視窗分數（沿用既有 `calc_portfolio_score`）
- [x] OOS 視窗 MDD
- [x] OOS 年化交易次數
- [x] OOS 保留後買進成交率

### 2) summary 指標
- [x] 視窗分數中位數
- [x] 視窗報酬率中位數
- [x] 最差視窗報酬率
- [x] 最大視窗 MDD
- [x] flat regime median score

### 3) regime 粗分（MVP）
- [x] benchmark(0050) 半年報酬粗分
  - up: `>= +8%`
  - flat: `(-8%, +8%)`
  - down: `<= -8%`
- [ ] 後續再細分 `high-vol / event-driven regime`。

---

## 3. 顯示與輸出

### 1) 已有輸出
- [x] summary JSON
- [x] 視窗明細 CSV
- [x] markdown 報表
- [x] 報表落點遵守 `outputs/ml_optimizer/`

### 2) 顯示層收斂
- [ ] **P0 完成後，優先進行 workbench 升級，用來正規顯示 WF / compare / promotion 摘要。**
- [ ] **workbench 需能讓升級過程中同步檢視 objective / WF / compare 的實際效果。**
- [ ] **console 只保留必要訊息，不優先複雜化。**
- [ ] callback 顯示也不是優先主線，先以 workbench 為主。

---

## 4. 升版規則

### 1) 已收斂 / 已接入
- [x] 候選版自身 `upgrade gate` + `compare gate` 共同決定是否可 promote。
- [x] compare assessment 已收斂成單一真理來源。
- [x] compare tolerance 已外部化到 `config/walk_forward_policy.json`。
- [x] Promote 已改成明確開關：`--promote` 或 `V16_OPTIMIZER_AUTO_PROMOTE=1`。
- [x] `trial=0` 永不 promote；只做 `run_best` 匯出與報表。

### 2) 後續主線
- [ ] 在 `wf_gate_median` objective 落地後，再確認 compare / promote 口徑是否仍完全一致。
- [ ] 先不再額外引入第二層加權升版公式。

---

## 5. 程式修改主清單（下一輪直接動工）

### P0 必做
- [ ] `config/walk_forward_policy.json`
  - 新增 `objective_mode`
  - 新增 / 明確化主搜尋 train 區間設定（例如 `search_train_end_year` 或同等欄位）
- [ ] `tools/optimizer/objective_runner.py`
  - 主回測改吃固定 train 區間，不再全吃
  - 加入 trial 級 WF 評分與 quality gate
  - `final_score = median_window_score`
  - `base_score` 改成 attrs / tie-break
- [ ] `tools/optimizer/study_utils.py`
  - 新增 mode-aware resolver
- [ ] `tools/optimizer/main.py`
  - `run_best` 改由新 resolver 決定
- [ ] `doc/TEMP_OPTIMIZER_UPGRADE_PLAN.md`
  - 同步更新本檔狀態

### P0 後優先進行
- [ ] workbench 正規顯示 WF / compare / promotion 摘要
- [ ] workbench 顯示 trial / run_best / champion 關鍵欄位
- [ ] workbench 支援檢視 objective 切換後的實際效果

### P1 後做
- [ ] regime 再細分
- [ ] validator

---

## 6. 命名收斂（沿用）
- [x] 正式現役版：`models/champion_params.json`
- [x] 本輪訓練最佳：`models/run_best_params.json`
- [x] Compare 報表已補 `OOS 總績效比較（Champion vs Challenger vs 0050）`

---

## 7. 關鍵設計結論（避免後續再混淆）
- [x] 這輪主線不是「每個 window 再訓一組參數」。
- [x] 這輪主線是「先選出一組固定參數，再對多個 OOS windows 驗穩定性」。
- [x] 因此這輪穩定性分數是對應單一固定參數，而不是對應定期重訓流程。
- [x] rolling walk-forward 不屬於本輪主線；若未來要做「每 K 天重訓 / spec 穩定性」再另案評估 expanding vs rolling train window。

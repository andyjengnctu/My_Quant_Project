# Optimizer 升級暫行規劃（Temporary）

> 目的：以最小但正確的主線升級 optimizer，先把「訓練區間切乾淨 + OOS/WF 主導 trial 評分」落地；先不處理定期重訓流程與 validator，P0 後優先進行 workbench 升級以檢視效果。

---

## 0. 本輪收斂共識

### 0.1 目前策略本質
- 目前 optimizer 產出的是固定策略參數 `.json`，不是另有需重訓權重的第二層模型。
- 因此現階段主問題不是「定版後如何重訓權重」，而是「如何用乾淨的訓練 / 測試流程選出一組固定 `.json`」。

### 0.2 現階段先不處理的範圍
- [x] 先不把「定期重訓 / 每 K 天重選參」納入這輪主線。
- [x] 先不新增第二種 optimizer。
- [x] 先不做 validator。
- [x] 先不把 console 顯示做得更複雜；顯示層較偏向未來用 workbench 正規呈現。

### 0.3 現階段主線定義
- [x] 現階段主線是：**先訓練出一組固定參數，再用多個 OOS windows 驗這一組參數的穩定性**。
- [x] 這個穩定性分數必須對應到同一組固定參數；因此本輪不採「每個 window 再訓一組新參數」的流程。
- [x] 若 trial 都會跑 OOS/WF 穩定性評分，則 `base_score` 不再適合作為主 objective，只保留 sanity gate / debug / tie-break 意義。
- [x] `legacy` 模式保留原本口徑：主搜尋吃到**最新可得日期**；`wf` 模式才使用固定 train 區間。

---

## 1. 升級主線（P0）

### 1) 訓練 / 測試取樣方式
- [x] MVP 已採 expanding walk-forward。
- [x] 已有主設定：`train_start_year = 2012`、`min_train_years = 8`、`test_window_months = 6`。
- [x] **P0：把主搜尋資料從目前的全吃改成明確 train 區間。**
  - `wf` 模式：
    - search train：`2012 ~ search_train_end_year`
    - WF/OOS：`wf_start ~ wf_end`
  - 目前預設口徑：
    - search train：`2012 ~ 2019`
    - WF/OOS：`2020 ~ 最新已納入驗證區間`
  - `legacy` 模式保留原本行為：主搜尋吃到最新可得日期。
- [x] 本輪主線**不做** rolling walk-forward；若未來改成「每 K 天重訓 / spec 穩定性」再另案評估 expanding vs rolling。

### 2) objective / trial 評分
- [x] 已在現有 optimizer 內加入 `objective_mode`，不新增第二種 optimizer。
  - `legacy_base_score`
  - `wf_gate_median`
- [x] **每個 trial 都跑固定參數版的 WF 穩定性評分**（`wf_gate_median` 模式下）。
- [x] **trial 級 hard gate 採現有 WF quality gate。**
  - `median_window_score >= gate_min_median_score`
  - `worst_ret_pct >= gate_min_worst_ret_pct`
  - `flat_median_score >= gate_min_flat_median_score`（若 flat 視窗存在）
- [x] **`quality_gate fail` 直接淘汰 trial。**
- [x] **`final_score = median_window_score`**（`wf_gate_median` 模式下）。
- [x] **`base_score` 降級為 sanity gate / debug / tie-break，不再主導排名**（`wf_gate_median` 模式下）。
- [x] 先不做「混權重大公式」：
  - 不先做 `base_score + worst-window + median + mdd penalty` 這種加權混分。
  - 避免新增 magic number 與 objective overfit。

### 3) best trial / run_best 決策
- [x] **已新增 mode-aware best-trial resolver，不再直接依賴 `study.best_trial`。**
- [x] `wf_gate_median` 模式下的排序收斂：
  1. `wf_quality_gate_status == pass`
  2. `wf_median_window_score` 由大到小
  3. `wf_worst_ret_pct` 由大到小
  4. `wf_flat_median_score` 由大到小
  5. `base_score` 由大到小
  6. `trial.number` 由小到大
- [x] **`run_best_params.json` 改由新 resolver 產出。**
- [x] compare gate / promote gate 仍維持單一路徑，不另分叉一套 OOS optimizer 流程。

### 4) trial metadata / 後續顯示共用欄位
- [x] **每個 trial 都寫入完整 WF attrs，供 resolver / callback / workbench 共用。**
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
- [x] 不優先把 console 越做越複雜。
- [x] 比起 callback / console 複雜化，較偏向用 workbench 正規顯示 WF / compare / promotion 摘要。
- [ ] **P0 後優先進行 workbench 升級**，用來檢視 objective / WF / compare / promotion 效果。
- [ ] console 只保留必要訊息；callback 顯示不是優先主線。

---

## 4. 升版規則

### 1) 已收斂 / 已接入
- [x] 候選版自身 `upgrade gate` + `compare gate` 共同決定是否可 promote。
- [x] compare assessment 已收斂成單一真理來源。
- [x] compare tolerance 已外部化到 `config/walk_forward_policy.py`。
- [x] Promote 已改成明確開關：`--promote` 或 `V16_OPTIMIZER_AUTO_PROMOTE=1`。
- [x] `trial=0` 永不 promote；只做 `run_best` 匯出與報表。

### 2) 後續主線
- [ ] 在 `wf_gate_median` objective 落地後，再確認 compare / promote 口徑是否仍完全一致。
- [ ] 先不再額外引入第二層加權升版公式。

---

## 5. 程式修改主清單

### P0 已完成
- [x] `config/walk_forward_policy.py`
  - 新增 `objective_mode`
  - 新增 / 明確化主搜尋 train 區間設定（如 `search_train_end_year`）
- [x] `tools/optimizer/objective_runner.py`
  - `wf` 模式下主回測改吃固定 train 區間，不再全吃
  - 加入 trial 級 WF 評分與 quality gate
  - `final_score = median_window_score`
  - `base_score` 改成 attrs / tie-break
- [x] `tools/optimizer/study_utils.py`
  - 新增 mode-aware resolver
- [x] `tools/optimizer/main.py`
  - `run_best` 改由新 resolver 決定
- [x] `doc/TEMP_OPTIMIZER_UPGRADE_PLAN.md`
  - 同步更新本檔狀態

### P0 後優先
- [ ] workbench 正規顯示 WF / compare / promotion 摘要
- [ ] workbench 顯示 trial / run_best / champion 關鍵欄位

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
- [x] `wf` 模式是新主線；`legacy` 模式只為保留原本用法與回退對照。
- [x] 等這條主線穩定後，若未來要做「每 K 天重訓 / spec 穩定性」再另開下一層規劃，不與本輪混在一起。

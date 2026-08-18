# Breakout Quality Research Queue

## 1. 用途與權責

本文件只管理 `breakout_quality` **尚待嘗試、正在執行或具條件式前置的研究項目**，讓後續實驗有明確順序、待決策問題與停止條件。

- **Registry**：`doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md`，唯一管理 identity／namespace／目前正式 status。
- **Experiment Log**：`doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md`，唯一保存已執行證據、數值結果、採用／淘汰理由。
- **Research Queue（本文件）**：只管理下一步研究順序、前置條件與 stop rule；不得覆寫 Registry／Log。
- **`doc/ToDo.md`**：使用者私人筆記；GPT 不主動讀取、引用或用來決定研究順序，除非使用者當輪明確要求。

狀態只使用：`ACTIVE`、`PLANNED`、`CONDITIONAL`、`BLOCKED`、`CLOSED`。尚未真正開始實作的項目不得預占科學 identity。

---

## 2. Current Queue

| 優先序 | 方案／待決策問題 | 狀態 | 固定條件 | 成功／停止條件 | 下一步 |
|---:|---|---|---|---|---|
| 1 | **Pre-Test Gate**：新模型／新策略調整先用原本單模型2021+ OOS與對應Strategy Pre-Test快速淘汰明顯無效假說；PASS才投入10-fold Rolling。 | **ACTIVE / USER_PRIORITY** | Pre-Test不可作promotion；模型與策略共用同一單模型OOS score source，Strategy Pre-Test不得重訓模型。 | 只回答「值得不值得跑Rolling」；若快速模型／策略轉化已明顯失敗則STOP。 | 先完成active candidate Pre-Test，再決定是否跑完整Rolling。 |
| 2 | **Evaluation Framework Migration / Extending-Window Rolling**：建立2016→2025單一expanding、annual PIT-safe refit chain，取代active Selection/Frozen二分。 | **ACTIVE / USER_PRIORITY** | 舊PIT folds保留historical evidence；因解除2011 sample cutoff已改變training-data semantics，新Extending-Window從2016起依完整合法歷史重建，之後逐年延伸。Extending-Window strategy params只stitch既有合法P2 schedules，不重做跨期optimizer。Current arms=`C58/C59/C60`，production仍C42/C44。 | Seed42 Extending-Window artifacts與C58/C59/C60能完整跑通且每foldinformation cutoff、parameter schedule、primary/secondary source均PIT-safe；若contract不相容則fail-fast，不得用legacy Frozen score補洞。 | 先準備`CONT13E_ROLL / CONT13K_ROLL / CONT13M_ROLL`與`P2_EXTENDING`，再跑Extending-Window Strategy Compare。 |
| 3 | **Fixed-Window Rolling**：控制歷史長度後檢查MR-13E/K/M跨年代learnability是否穩定。 | **PLANNED / AFTER_OPERATIONAL** | Fixed train window=`120M`、score fold=`12M`、inner validation=`24M`、Seed42；輸出與Extending-Window隔離。報表必須保留train span/groups/tickers與score groups。 | 若固定window後各年代ranking/strategy方向仍有顯著結構差異，才判定可能有regime learnability；若差異大幅縮小，則舊Selection/Frozen差異主要是data maturity/policy confounder。 | Extending-Window Seed42完成後執行model Stability；第一版不做multi-seed。 |
| 4 | **Extending-Window Rolling C60 Multi-seed robustness**。 | **CONDITIONAL** | fixed=`C58`、stochastic=`C60`；每seed同時建立13K/13M Extending-Window sources；`seed_count=4`、generator=`20260810`。不跑Legacy C56/C57 Frozen/Selection robustness。 | 只有Seed42 Extending-Window結果仍足以改變B2 GO/REJECT或promotion判斷才執行；否則停止節省算力。 | 依項目1結果決定。 |
| 5 | **Plan C-M：13K PIT-safe upside context → Conditional Low-Adverse model**。 | **PLANNED / AFTER_FRAMEWORK_MIGRATION** | Stage-1 13K context必須rolling cross-fitted/PIT-safe，同日percentile；Stage-2 Target固定MR-13M low-adverse。不得把full-fit score回灌training rows。 | Conditional adverse ranking需明顯優於MR-13M，且Extending-Window strategy conversion保留13K upside；否則停止stacking。 | 新evaluation framework完成最小驗證後才開始並分配新MR identity。 |
| 6 | **Plan A：MR-13H economic Target + parameter-free path-dynamics representation**。 | **PLANNED** | Target固定MR-13H、InceptionTime family、Full-list pairwise、Seed42；只改causal path primitives。 | Extending-Window model metrics同方向明顯勝reference才進strategy；否則停止。 | Plan C-M之後。 |
| 7 | **Plan B3：13K confidence boundary + 13M tie-break**。 | **CONDITIONAL** | 不用magic score-gap；優先用Extending-Window multi-seed rank stability。 | 只有能建立無調參confidence contract才GO。 | 後做。 |
| 8 | **Plan C-K：13M PIT-safe safety context → Conditional MFE model**。 | **CONDITIONAL** | Stage-1 13M context須rolling PIT-safe；Stage-2 Target固定Pure MFE。 | 只有C-M未解決且仍有conditional-stacking價值時做。 | C-M後。 |
| 9 | **13M sizing / regime gating / cross-sectional representation**。 | **CONDITIONAL** | 保持一次只改一個主要研究維度。 | 只有前述較小變更不足且新證據可能改變決策時才展開。 | 最後。 |

### 2.1 Framework migration對既有研究的處理

- 2014～2020既有Selection PIT annual folds不丟棄，重新分類為historical expanding annual-refit evidence；contract相容時直接成為legacy historical evidence。
- 2021～2026既有Frozen Forward Cxx/MRxx結果、reports、manifests與ID永久保留為`LEGACY_FROZEN_FORWARD / HISTORICAL_EVIDENCE`，但不再是current Gate或robustness。
- C57 Seed42 historical PIT與C56 Seed42 Legacy Frozen結果都保留；原本「先完成C56/C57兩stage robustness」的Queue項目因evaluation-policy migration而**CLOSED / SUPERSEDED**，不是因策略結果被否定。
- 同stage controlled contrasts仍可引用；跨Selection/Frozen的直接差值不得再單獨解讀為temporal stability/learnability。

## 3. 已停止的相鄰方向（不得重新包裝成新項目）

- MR-13L：MFE/adverse raw-R dual regression後直接相減。
- MR-13N：single-model 50/50 equal-rank composite。
- MR-13K + MR-13M frozen 50/50 score fusion；兩expert score近鏡像而互相抵消。
- MR-13O：Pareto-dominance pairwise；Forward Pareto僅弱於可進PIT的程度。
- SR-C55：raw MR-13M baseline-relative basket safety floor；雖降MDD與提高勝率，但Return/Payoff/RoMD轉換不佳，已被B2 residual safety取代。
- 不掃 `MFE - λ × adverse`、30/70～70/30 score權重、Pareto threshold、loss小改來追OOS。

本節只摘要 stop direction；正式數值與結案理由仍以 Registry／Experiment Log 為準。

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
| 1 | **Extending-Window Test / OOS Test**：固定2020年底information cutoff，只訓練一次並score 2021～最新，快速判斷active模型／策略是否值得投入Rolling Test。 | **ACTIVE / USER_PRIORITY** | score=`2021-01-01→auto latest`、single score block；train／validation／refit只用2021前合法成熟歷史。Current strategy matrix=`C61/C58/C59/C60`。 | model Gate或策略轉化明顯失敗即可STOP；仍有決策價值才進Rolling Test。 | 先以OOS完成current candidate最小必要時間外證據。 |
| 2 | **Extending-Window Test / Rolling Test**：完整12M annual refit evidence。 | **CONDITIONAL / AFTER_OOS** | period=`2016～2025`、10 folds、expanding history、12M cadence；既有合法annual folds/results沿原canonical paths REUSE。P2/P4 Extending schedules保持PIT-safe，不因mode重做optimizer。 | 只有OOS通過且完整年度穩定性仍可能改變GO/REJECT/promotion時執行；最終promotion只看Rolling。 | OOS GO後補足10-fold evidence。 |
| 3 | **Fixed-Window Stability Test**：控制歷史長度後檢查MR-13E/K/M learnability。 | **PLANNED / AFTER_EXTENDING_OOS** | fixed train window=`120M`、inner validation=`24M`；OOS只訓練一次並score 2021→最新，必要時才做Rolling 12M十fold；輸出與Extending隔離。 | 若OOS已足以顯示固定window下結構差異，再決定是否需要Rolling；不作production performance Truth。 | Extending OOS之後按最小必要證據執行。 |
| 4 | **Extending-Window C60 Multi-seed Robustness Test**。 | **CONDITIONAL** | fixed=`C58`、stochastic=`C60`、seed_count=`4`、generator=`20260810`；OOS先用single 2021→latest fold×4 seeds，Rolling才是10 annual folds×4 seeds。OOS/Rolling使用隔離的robustness/model-work roots；legacy robustness不補跑。 | 只有Seed42與OOS strategy evidence仍足以改變B2 GO/REJECT或promotion判斷才執行；OOS robustness若已失敗就不跑Rolling。 | 先OOS robustness；只有結果仍需要完整seed evidence才補Rolling。 |
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

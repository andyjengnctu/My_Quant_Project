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
| 1 | **Extending-Window Test / OOS Test**：固定2020年底information cutoff，只訓練一次並score 2021～最新，快速判斷active模型／策略是否值得投入Rolling Test。 | **ACTIVE / USER_PRIORITY / RERUN_AFTER_CONTRACT_FIX** | score=`2021-01-01→auto latest`、single score block；train／validation／refit只用2021前合法成熟歷史；C61/C62使用Optimizer canonical Full schedule，C58/C63/C59/C60使用Min schedule；OOS只在讀取時freeze同一跨年JSON的2021合法member，Research不得另建OOS參數工件。C60 K/M同一OOS namespace。Current strategy matrix=`C61/C62/C58/C63/C59/C60`；C62/C63只換`base-finalists-agree` parameter policy。 | model Gate或策略轉化明顯失敗即可STOP；仍有決策價值才進Rolling Test。 | 先以OOS完成current candidate最小必要時間外證據。 |
| 2 | **Extending-Window Test / Rolling Test**：完整12M annual refit evidence。 | **CONDITIONAL / AFTER_OOS** | period=`2021→latest`、expanding history、12M cadence；既有2021+合法annual folds沿canonical PIT fold store REUSE，latest partial fold按需補建；策略結果使用新的rolling_2021_forward namespace；C61/C62與OOS共讀同一Optimizer canonical Full JSON，C58/C63/C59/C60共讀同一Min JSON；Rolling按effective date使用跨年schedule；C62/C63固定`base-finalists-agree`。所有`base_best/finalists_agree/ensemble_*`與seed/trials由同一Strategy Parameter SSOT管理，Research不另建P2/P4 schedule。 | 只有OOS通過且完整年度穩定性仍可能改變GO/REJECT/promotion時執行；最終promotion只看Rolling。 | OOS GO後補足2021→latest的12M rolling evidence。 |
| 3 | **Fixed-Window Stability Test**：控制歷史長度後檢查MR-13E/K/M learnability。 | **PLANNED / AFTER_EXTENDING_OOS** | fixed train window=`120M`、inner validation=`24M`；OOS只訓練一次並score 2021→最新，必要時才做Rolling 12M年度fold（2021→latest動態決定）；輸出與Extending隔離。 | 若OOS已足以顯示固定window下結構差異，再決定是否需要Rolling；不作production performance Truth。 | Extending OOS之後按最小必要證據執行。 |
| 4 | **Extending-Window Multi-seed Robustness Test：end-to-end benchmark alignment**。 | **ACTIVE / RESEARCH_ORCHESTRATION_FIX_IMPLEMENTED / FORMAL_RERUN_PENDING / EVIDENCE_PENDING** | 與single-seed共用`extending_current`六arm Compare Suite。唯一benchmark題庫由`config/training_policy.py`管理：`benchmark_id=end_to_end_v1`、2 seeds、generator=`20260810`、resolved=`693545351/2014432738`；題庫只擁有seed identity，strategy trials/window/cadence/parallel全部直接繼承canonical Optimizer；`trials/fold`只由`OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT`即時解析，Research Queue不保存目前數值。C61/C58/C59/C60全部使用per-seed benchmark strategy params；C59/C60再以同一seed訓MR-13E或K+M；C62/C63只作production finalists-agree consensus reference。每seed/family只建立一份`2021→latest`完整benchmark strategy schedule JSON；OOS/Rolling共讀同一實體工件，OOS只在replay讀取時freeze 2021合法member，Rolling按effective date使用完整schedule，禁止2021-only參數訓練或2022+ tail stitch。模型端同樣共用benchmark-only 2021 initial checkpoint cache：E/K/M同seed且fitting contract一致時OOS與Rolling必須使用同一checkpoint SHA，各mode只依自己的score horizon重評score；Rolling自2022起才補annual model folds。per-mode checkpoint/scores可照retention清除，但shared initial cache保留；仍禁止fallback回Seed42 production params。 | Runtime只有在benchmark strategy artifacts與same-seed model artifacts全部identity對齊後才允許該seed replay；每seed先算C61-C58、C59-C58、C59-C61、C60-C58、C60-C61、C60-C59 paired deltas，再跨2 seeds aggregate。strategy schedule一次交給canonical Rolling Optimizer建立全部年度fold並沿用其parallel policy；shared model cache若fitting identity不符就不得重用，identity相同但checkpoint SHA不同直接FAIL。OOS已足以否決則不跑Rolling。 | **先執行OOS benchmark長跑取得2-seed evidence；若結果仍可能改變決策，再執行Rolling。Rolling應REUSE同一完整strategy schedule與同seed E/K/M initial checkpoint，只新增必要的2022+ model annual folds。** |
| 5 | **Plan C-M：13K PIT-safe upside context → Conditional Low-Adverse model**。 | **PLANNED / AFTER_FRAMEWORK_MIGRATION** | Stage-1 13K context必須rolling cross-fitted/PIT-safe，同日percentile；Stage-2 Target固定MR-13M low-adverse。不得把full-fit score回灌training rows。 | Conditional adverse ranking需明顯優於MR-13M，且Extending-Window strategy conversion保留13K upside；否則停止stacking。 | 新evaluation framework完成最小驗證後才開始並分配新MR identity。 |
| 6 | **Plan A：MR-13H economic Target + parameter-free path-dynamics representation**。 | **PLANNED** | Target固定MR-13H、InceptionTime family、Full-list pairwise、Seed42；只改causal path primitives。 | Extending-Window model metrics同方向明顯勝reference才進strategy；否則停止。 | Plan C-M之後。 |
| 7 | **Plan B3：13K confidence boundary + 13M tie-break**。 | **CONDITIONAL** | 不用magic score-gap；優先用Extending-Window multi-seed rank stability。 | 只有能建立無調參confidence contract才GO。 | 後做。 |
| 8 | **Plan C-K：13M PIT-safe safety context → Conditional MFE model**。 | **CONDITIONAL** | Stage-1 13M context須rolling PIT-safe；Stage-2 Target固定Pure MFE。 | 只有C-M未解決且仍有conditional-stacking價值時做。 | C-M後。 |
| 9 | **13M sizing / regime gating / cross-sectional representation**。 | **CONDITIONAL** | 保持一次只改一個主要研究維度。 | 只有前述較小變更不足且新證據可能改變決策時才展開。 | 最後。 |

### 2.1 Framework migration對既有研究的處理

- 2026-08-21：Research execution前置已收斂為共用artifact graph／orchestrator；Dataset/Target/model/PIT/Optimizer params/Audit source依同一REUSE/BUILD/REBUILD/RESUME/BLOCKED語意處理，且每個producer後強制re-plan。這是framework工程前置，不改Queue研究優先順序；formal double check通過後仍先取得Extending OOS／必要Robustness evidence。
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

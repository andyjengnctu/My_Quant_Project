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
| 1 | **SR-C68/C69 Matched Feasible-Ascent K-Flex Test**：在global exact K-Flex計算不可實用後，以同一deterministic local-search family隔離solver effect與K effect。 | **ACTIVE / IMPLEMENTED / RESULT_PENDING** | C68與C59完全同K/R0/model/params/execution，只把exact改為C68/C69專用deterministic matched feasible-ascent（不改歷史feasible-ascent）；C69與C68完全同solver/R0，只把count由`==C58 K`放寬到`C58 K<=count<=physical free slots`。不得訓練新模型、改R0、target、threshold、sizing、cash或entry。 | 先跑single-seed OOS + Rolling。先看`C68-C59`是否顯示local solver本身造成大幅漂移，再以`C69-C68`判斷純K-Flex是否增加count、保留R0/資金部署並改善Raw→Planned High-MFE與Realized EV/Return/RoMD。若C69不能相對C68形成有意義改善，停止K-Flex，不進robustness。 | 執行current Compare Suite C68/C69；結果後才決定是否把相同K-Flex local contract移植到C65/C66。 |
| 2 | **Primary-preserving MR-13P runtime condition**：C64 hard safety floor已REJECT；只有Stage Attribution顯示主要瓶頸是resource conversion且MR-13P mechanism仍相關時才回頭測minimum-primary-regret secondary application。 | **CONDITIONAL / AFTER_C68_C69_RESULT** | 固定MR-13P模型與dual-head PIT，不改Target/architecture/loss；不做score fusion、固定權重、OOS-derived threshold或basket-wide safety floor。 | 只有C68/C69結果仍顯示resource conversion已改善但Safety/upside conversion問題未解、且MR-13P mechanism仍可能改變決策時才GO；仍以`+2R前初始Stop↓`與Full-MFE preservation為主。 | C68/C69結果後再決定，不平行執行。 |
| 3 | **Conditional model formulation後續**：只有Stage Attribution顯示Raw Top-K本身已明顯缺乏High-MFE，才設計下一個單一conditional變體。 | **CONDITIONAL / AFTER_C68_C69_RESULT** | 保持daily-universal strategy-agnostic、一次只改一個conditional mechanism；禁止掃權重／threshold或回到raw scalar MFE-adverse composite。 | 必須由C68/C69結果顯示解除K限制後Raw→Planned問題仍不足以解釋conversion，且既有Raw Top-K evidence不再足以支持runtime主因；否則停止擴張。 | C68/C69後做。 |
| 4 | **Plan C-M：PIT-safe predicted-upside context → Conditional Low-Adverse model**。 | **CONDITIONAL / LATER** | Stage-1 upside context必須rolling cross-fitted/PIT-safe同日percentile；Stage-2不得讀full-fit score。 | 只有現有single-model conditional家族仍有明確未解問題時才GO。 | 後做。 |
| 5 | **Survivable Pure-MFE**。 | **CONDITIONAL / FALLBACK_AFTER_CONDITIONAL_LINE** | Daily-universal、Full-list pairwise、canonical fixed-risk geometry、Seed42；只改Target，不引入strategy state。 | Conditional路線已有足夠否定證據才GO。 | Conditional主線後。 |
| 6 | **First-Passage / competing-risk Target**。 | **CONDITIONAL / AFTER_SURVIVABLE_TARGET** | 保持 daily-universal、PIT-safe、strategy-agnostic；避免OOS-derived magic threshold。 | survivable-MFE仍不足才做。 | 後做。 |
| 7 | **Extending-Window Multi-seed Robustness Test**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | 與single-seed共用`extending_current` Compare Suite與same-seed SSOT。 | 只有C65或C66（或後續candidate）single-seed已明顯改善C59才投入。 | 延後。 |
| 8 | **Fixed-Window Stability Test**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | fixed train window=`120M`、inner validation=`24M`。 | 只有candidate已改善conversion且history length仍可能改變決策才做。 | 延後。 |
| 9 | **Plan B3：confidence boundary + conditional tie-break**。 | **CONDITIONAL** | 只在可建立parameter-free confidence contract時考慮。 | Conditional主線仍需要更弱secondary作用才GO。 | 後做。 |
| 10 | **sizing / regime gating / cross-sectional representation**。 | **CONDITIONAL** | 一次只改一個主要研究維度。 | 前述較直接方案不足且新證據可能改變決策時才展開。 | 最後。 |

### 2.1 Framework migration對既有研究的處理

- 2026-08-25：`AUD-mfe-safety-target-geometry`與`AUD-selection-k-r0-attribution`均已取得足夠證據並退役：Raw Top-K High-MFE約72–79%，但Planned僅約36–44%，direct-infeasible約92–98%，K headroom約60–72%。原`SR-C67 K-Flex/R0-preserved global exact`雖完成contract與安全效能剪枝，但兩次實際OOS replay仍在large-K exact score certification長時間無結果，故標記`COMPUTE_BLOCKED / NO_STRATEGY_RESULT`並退出current suite；不是策略REJECT。current主線改為matched local pair：C68=dedicated matched fixed-K feasible-ascent solver control、C69=同一專用solver + K-Flex/R0-preserved treatment；`C69-C68`才是純K效果。Multi-seed與Fixed-Window持續延後。
- 2026-08-25：使用者明確要求Single-head與Duo-head兩種Reverse-Conditional MFE都實作並比較可學性／獲利轉換率。正式配置MR-13Q/R與C65/C66；兩者共用完全相同`J=U-E(U\|S)` truth與strategy contract，唯一模型差異為MR-13R是否顯式學Raw Safety condition。原minimum-primary-regret MR-13P runtime實驗降為A/B之後的fallback。
- 2026-08-24：C64 single-seed OOS+Rolling conversion evidence已取得。Conditional Safety確實把Adverse壓低，但basket-wide hard floor在兩mode都未改善`+2R前初始Stop`且Full-MFE明顯下降；使用者明確決定**持續改善Conditional路線**，因此C64 hard-floor formulation結案但MR-13P signal不結案。current主線先做primary-preserving/minimum-regret runtime condition；若仍不足，再做唯一一個conditional model formulation v2。Survivable Pure-MFE與First-Passage降為更後續fallback。
- 2026-08-23：current single-seed OOS／Rolling與First-Passage evidence已取得；使用者將研究主線改為**Conversion-first**。在找到相對C59更好的candidate前，Extending Multi-seed Robustness與Fixed-Window Stability均降為後續validation gate。MFE／MAE component learnability均已明確，`MR-13P` Seed42 Forward Model Gate已GO，current主線進入C64 single-seed conversion Gate。
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

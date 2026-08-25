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
| 1 | **SR-C70 MR-13R No-K / No-R0 Joint Resource Ablation**：直接驗證C66的高learnability Conditional-MFE ranking是否主要被C58-derived K/R0聯合contract改寫。 | **ACTIVE / IMPLEMENTED / RESULT_PENDING** | C70與C66必須共用MR-13R `CONT13R_ROLL`、final model_score、Min base-finalist-best、all-off、canonical sizing/cash/orderability/execution與max positions=10；唯一變更是完全移除baseline K count restriction與R0 reserved-capital floor。不得新增Raw Safety gate、threshold、model retraining、capital objective或其他selector規則。 | 先跑single-seed OOS + Rolling。Mechanism先看Filled四象限：High-MFE是否回升、HM/LS/LM/HS如何移動；再看Full-MFE、Adverse、Realized EV、平均曝險/position gap。若High-MFE與deployment回升且EV/conversion改善，支持K×R0為主要bottleneck；若High-MFE回升但HM/LS/adverse惡化且EV不改善，則K/R0具有quality-proxy保護作用，下一步才有理由測MR-13R Raw Safety head。 | Strategy Compare current suite只保留C61/C58/C59/C64/C66/C70；Primary=`C70-C66`。不進multi-seed，直到OOS+Rolling已足以決策。 |
| 2 | **Primary-preserving MR-13P runtime condition**：C64 hard safety floor已REJECT；只有Stage Attribution顯示主要瓶頸是resource conversion且MR-13P mechanism仍相關時才回頭測minimum-primary-regret secondary application。 | **CONDITIONAL / AFTER_C70_RESULT** | 固定MR-13P模型與dual-head PIT，不改Target/architecture/loss；不做score fusion、固定權重、OOS-derived threshold或basket-wide safety floor。 | 只有C70在移除K/R0後仍呈現High-MFE但HM/LS/adverse偏高、且MR-13P mechanism仍可能改變決策時才GO；仍以`+2R前初始Stop↓`與Full-MFE preservation為主。 | C70 OOS+Rolling後再決定，不平行執行。 |
| 3 | **Conditional model formulation後續**：只有Stage Attribution顯示Raw Top-K本身已明顯缺乏High-MFE，才設計下一個單一conditional變體。 | **CONDITIONAL / AFTER_C70_RESULT** | 保持daily-universal strategy-agnostic、一次只改一個conditional mechanism；禁止掃權重／threshold或回到raw scalar MFE-adverse composite。 | 必須由C70顯示joint K/R0 restriction解除後，既有MR-13R final score仍無法把upside轉成安全EV，才允許新增conditional model formulation；否則停止擴張。 | C70 OOS+Rolling後做。 |
| 4 | **Plan C-M：PIT-safe predicted-upside context → Conditional Low-Adverse model**。 | **CONDITIONAL / LATER** | Stage-1 upside context必須rolling cross-fitted/PIT-safe同日percentile；Stage-2不得讀full-fit score。 | 只有現有single-model conditional家族仍有明確未解問題時才GO。 | 後做。 |
| 5 | **Survivable Pure-MFE**。 | **CONDITIONAL / FALLBACK_AFTER_CONDITIONAL_LINE** | Daily-universal、Full-list pairwise、canonical fixed-risk geometry、Seed42；只改Target，不引入strategy state。 | Conditional路線已有足夠否定證據才GO。 | Conditional主線後。 |
| 6 | **First-Passage / competing-risk Target**。 | **CONDITIONAL / AFTER_SURVIVABLE_TARGET** | 保持 daily-universal、PIT-safe、strategy-agnostic；避免OOS-derived magic threshold。 | survivable-MFE仍不足才做。 | 後做。 |
| 7 | **Extending-Window Multi-seed Robustness Test**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | 與single-seed共用`extending_current` Compare Suite與same-seed SSOT。 | 只有C70（或後續candidate）完成single-seed OOS+Rolling且仍值得驗證穩健性時才投入。 | 延後。 |
| 8 | **Fixed-Window Stability Test**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | fixed train window=`120M`、inner validation=`24M`。 | 只有candidate已改善conversion且history length仍可能改變決策才做。 | 延後。 |
| 9 | **Plan B3：confidence boundary + conditional tie-break**。 | **CONDITIONAL** | 只在可建立parameter-free confidence contract時考慮。 | Conditional主線仍需要更弱secondary作用才GO。 | 後做。 |
| 10 | **sizing / regime gating / cross-sectional representation**。 | **CONDITIONAL** | 一次只改一個主要研究維度。 | 前述較直接方案不足且新證據可能改變決策時才展開。 | 最後。 |

### 2.1 Framework migration對既有研究的處理

- 2026-08-25：`AUD-mfe-safety-target-geometry`已結案；`AUD-selection-k-r0-attribution`雖已有result，但使用者明確要求在K/R0影響解決前**保留formal implementation與可重跑入口**，因此改列ACTIVE_RETAINED，不退役。C68/C69 OOS+Rolling結果亦已取得：C69 position-gap顯著下降，但RoMD/EV跨兩mode惡化，且Full-MFE略升同時Adverse升高；當時下一個最小證據轉為`AUD-c69-marginal-position-attribution`，該Audit後續已完成並導向目前`SR-C70`。Multi-seed與Fixed-Window持續延後。
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
- 2026-08-25：C69 marginal Audit已取得結果：OOS/ Rolling K+1+ High-MFE=`72.41/57.14%`、HM/LS=`55.17/40.00%`、Realized EV=`-0.04/-0.05R`；因此MR-13E unrestricted K-Flex不promotion。使用者下一步決定只在目前learnability較強的MR-13R/C66上共同移除K與R0，其餘不動，建立`SR-C70`。Current Compare Suite聚焦為C61/C58/C59/C64/C66/C70；舊K/R0 Audit與C69 marginal Audit結果/implementation保留。

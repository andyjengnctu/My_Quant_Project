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
| 1 | **AUD-mr13r-joint-capital-drawdown**：C71-C73的Safety gate與C74 joint product都未把HM/HS/RoMD推離既有trade-off；先判斷MR-13R兩head是否已有joint HM/HS information、Safety為何提高Exposure、single-stock adverse改善為何沒有等比例轉成portfolio MDD。 | **ACTIVE / IMPLEMENTED / RESULT_PENDING** | 只讀C71-C74已完成OOS/ Rolling pair sidecars與canonical MFE×Safety future truth；pin fingerprints OOS=`b12582a9de23`、Rolling=`3bca1e932f2e`。不重跑strategy、不訓練模型、不調threshold/weight。 | 若predicted Safety×MFE右上角與高Safety cohort內Conditional-MFE仍有明顯HM/HS/MFE enrichment，優先修selector conversion；若joint signal本身弱，回model joint objective；若MDD主要由同期loss clustering/portfolio concentration造成，轉portfolio construction。取得足以做NEXT決策即停止Audit。 | `Research → Audit／診斷 → 策略 Pair／Portfolio Attribution`執行正式Audit並審閱joint-signal / capital-conversion / drawdown三層結果。 |
| 2 | **Conditional model formulation後續**：只有Audit顯示MR-13R現有兩head在高Safety cohort內缺joint MFE signal，才設計下一個單一joint/conditional target。 | **CONDITIONAL / AFTER_AUDIT_RESULT** | Daily-universal、strategy-agnostic；一次只改一個objective mechanism；不掃權重、threshold或回到OOS-tuned scalar composite。 | Audit若顯示joint signal已存在則不重訓；只有model-level缺口足以改變決策才GO。 | Audit後決定。 |
| 3 | **Portfolio construction / drawdown mechanism**：只有Audit顯示single-stock Safety已改善但MDD主要由同期持倉/entry clustering主導時才展開。 | **CONDITIONAL / AFTER_AUDIT_RESULT** | 不把future truth放入runtime；一次只改一個portfolio-level mechanism，與single-stock model objective分離。 | Top drawdown episodes若顯示明確共振/集中結構且不是個股adverse本身即可解釋，才GO。 | Audit後決定。 |
| 4 | **Primary-preserving MR-13P runtime condition**。 | **CONDITIONAL / AFTER_AUDIT_RESULT** | 固定MR-13P模型與dual-head PIT；不做score fusion、固定權重、OOS-derived threshold或basket-wide safety floor。 | 只有Audit仍支持secondary safety signal可改善conversion且不需新model objective時才GO。 | 後做。 |
| 5 | **Plan C-M：PIT-safe predicted-upside context → Conditional Low-Adverse model**。 | **CONDITIONAL / LATER** | Stage-1 upside context必須rolling cross-fitted/PIT-safe同日percentile；Stage-2不得讀full-fit score。 | 只有現有single-model conditional家族仍有明確未解model問題時才GO。 | 後做。 |
| 6 | **Survivable Pure-MFE**。 | **CONDITIONAL / FALLBACK_AFTER_CONDITIONAL_LINE** | Daily-universal、Full-list pairwise、canonical fixed-risk geometry、Seed42；只改Target，不引入strategy state。 | Conditional路線已有足夠否定證據才GO。 | Conditional主線後。 |
| 7 | **First-Passage / competing-risk Target**。 | **CONDITIONAL / AFTER_SURVIVABLE_TARGET** | 保持 daily-universal、PIT-safe、strategy-agnostic；避免OOS-derived magic threshold。 | survivable-MFE仍不足才做。 | 後做。 |
| 8 | **Extending-Window Multi-seed Robustness Test**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | 與single-seed共用`extending_current` Compare Suite與same-seed SSOT。 | 只有Audit後產生真正有決策價值的candidate才投入；C71-C74目前都不進robustness。 | 延後。 |
| 9 | **Fixed-Window Stability Test**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | fixed train window=`120M`、inner validation=`24M`。 | 只有candidate已改善conversion且history length仍可能改變決策才做。 | 延後。 |
| 10 | **sizing / regime gating / cross-sectional representation**。 | **CONDITIONAL** | 一次只改一個主要研究維度。 | Audit/conditional/portfolio較直接方案不足且新證據可能改變決策時才展開。 | 最後。 |

### 2.1 Framework migration對既有研究的處理

- 2026-08-26：C74 parameter-free `Safety_pct × ConditionalMFE_pct` OOS/ Rolling已完成但未提升HM/HS/RoMD：OOS Return/MDD/RoMD/Exposure=`61.71%/20.16%/3.06/72.08%`、HM/HS=`21.84%`；Rolling=`108.52%/15.03%/7.22/67.13%`、HM/HS=`22.85%`。因此停止selector arithmetic；current priority切到`AUD-mr13r-joint-capital-drawdown`，只讀C71-C74 completed sidecars回答joint signal、Safety→capital conversion與portfolio drawdown clustering。
- 2026-08-26：C71-C73 single-seed OOS/ Rolling已完成。提高Raw Safety cutoff會單調降低HM/LS、提高High-Safety與平均曝險，但主要增加LM/HS，HM/HS只小幅改善，High-MFE/Full-MFE與RoMD則走弱；使用者明確停止threshold tuning，避免把迭代OOS變成threshold lookahead。下一步先做C74 parameter-free `Safety_pct × ConditionalMFE_pct` joint selector；若仍無法提升HM/HS，才進Joint-Signal/Capital/Drawdown Audit。C70仍只保留historical evidence。
- 2026-08-25：`AUD-mfe-safety-target-geometry`已結案；`AUD-selection-k-r0-attribution`雖已有result，但使用者明確要求在K/R0影響解決前**保留formal implementation與可重跑入口**，因此改列ACTIVE_RETAINED，不退役。C69 marginal Audit導向SR-C70 joint No-K/No-R0；C70 OOS/ Rolling現已完成：High-MFE約`62.09/60.26%`與Full-MFE=`1.93/1.97R`大幅提高，但HM/LS=`42.58/40.26%`、High-Safety=`35.44/36.88%`、EV=`0.25/0.36R`，顯示主要問題轉成absolute Safety。故current最小change為`SR-C71=C70+MR-13R Raw Safety gate`；Multi-seed與Fixed-Window持續延後。
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
- 2026-08-25：C69 marginal Audit已取得結果：OOS/ Rolling K+1+ High-MFE=`72.41/57.14%`、HM/LS=`55.17/40.00%`、Realized EV=`-0.04/-0.05R`；因此MR-13E unrestricted K-Flex不promotion。使用者下一步決定只在目前learnability較強的MR-13R/C66上共同移除K與R0，其餘不動，建立`SR-C70`。當時Compare Suite聚焦為C61/C58/C59/C64/C66/C70；C70結果完成後current再加入C71 Raw-Safety-only treatment；舊K/R0 Audit與C69 marginal Audit結果/implementation保留。

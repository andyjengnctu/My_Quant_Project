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
| 1 | **MR-13S Safety→Raw-MFE Duo-head Forward Model Gate**：直接檢驗MR-13R joint collapse是否來自residual `J=U-E(U|S)`失去absolute MFE×Safety座標。 | **ACTIVE / IMPLEMENTED / MODEL_GATE_RESULT_PENDING** | 與MR-13R共用`inception_time_safety_conditional_mfe_v1`、Raw Safety S、stop-gradient Safety context、full-list Delta-NDCG、兩head固定等權loss、daily-universal universe與Seed42；唯一scientific change是final target=`absolute Pure-MFE percentile U`。epoch selection只用Validation Raw-MFE Dailyρ；Forward OOS只作frozen model Gate，不得fit threshold/weight/calibration。 | Raw Safety與Raw-MFE marginal signal必須保留；5×5不應再呈MR-13R極端反對角，S5×M5需有實質sample support，且S4/S5×M4/M5應相對population穩定enrich HM/HS；Safety高cohort內Raw-MFE→actual MFE rho不可崩潰。若joint geometry仍collapse，停止conversion並回model representation/joint formulation；若GO才授權下一個唯一strategy conversion。 | `Research → 模型訓練 → 訓練目前模型 → forward-OOS模型報表`。先審閱MR-13S model-only Gate；本項不建立PIT/C75。 |
| 2 | **MR-13S strategy conversion**：只在model Gate GO後設計一個最小conversion。 | **CONDITIONAL / AFTER_MR13S_MODEL_GATE_GO** | 不掃threshold、lambda或selector arithmetic；先固定model/checkpoint/params/rules與No-K/No-R0研究基準，只允許一個事前定義的joint conversion。 | 只有model Gate證明absolute Safety×Raw-MFE有可用joint upper-right geometry才GO；否則不占`SR-C*` identity。 | Model Gate後決定，不預占C75。 |
| 3 | **Joint representation / target formulation後續**。 | **CONDITIONAL / ONLY_IF_MR13S_MODEL_GATE_FAILS** | Daily-universal、strategy-agnostic；一次只改一個model mechanism，不使用OOS結果fit scalar weights、threshold或calibration。 | 若MR-13S兩個absolute heads仍各自可學但joint geometry再次collapse，才進representation/joint objective；若MR-13S joint Gate GO則本項停止。 | MR-13S Gate後決定。 |
| 4 | **Portfolio construction / drawdown mechanism**。 | **CONDITIONAL / DEFERRED** | 不把future truth放入runtime；一次只改一個portfolio-level mechanism，與single-stock model objective分離。 | Schema-v2 Audit顯示top drawdown主要由Low-MFE positions貢獻，未支持把entry clustering/concentration列為current blocker；只有後續更好的single-stock joint conversion仍被portfolio covariance主導才GO。 | 延後。 |
| 5 | **Primary-preserving MR-13P runtime condition**。 | **CONDITIONAL / FALLBACK** | 固定MR-13P模型與dual-head PIT；不做score fusion、固定權重、OOS-derived threshold或basket-wide safety floor。 | 只有MR-13S/joint formulation線停止後，且既有secondary safety signal仍可能改變conversion決策才GO。 | 後做。 |
| 6 | **Plan C-M：PIT-safe predicted-upside context → Conditional Low-Adverse model**。 | **CONDITIONAL / LATER** | Stage-1 upside context必須rolling cross-fitted/PIT-safe同日percentile；Stage-2不得讀full-fit score。 | 只有現有single-model conditional家族仍有明確未解model問題時才GO。 | 後做。 |
| 7 | **Survivable Pure-MFE / First-Passage target family**。 | **CONDITIONAL / FALLBACK_AFTER_CONDITIONAL_LINE** | Daily-universal、PIT-safe、strategy-agnostic；一次只改一個Target，不引入strategy state。 | MR-13S與後續joint formulation已有足夠否定證據才GO。 | Conditional主線後。 |
| 8 | **Extending-Window Multi-seed Robustness Test**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | 與single-seed共用`extending_current` Compare Suite與same-seed SSOT。 | 只有MR-13S Gate後產生真正有決策價值的strategy candidate才投入；C71-C74不進robustness。 | 延後。 |
| 9 | **Fixed-Window Stability Test**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | fixed train window=`120M`、inner validation=`24M`。 | 只有candidate已改善conversion且history length仍可能改變決策才做。 | 延後。 |
| 10 | **sizing / regime gating / cross-sectional representation**。 | **CONDITIONAL** | 一次只改一個主要研究維度。 | 更直接的MR-13S/joint/portfolio證據不足且新證據可能改變決策時才展開。 | 最後。 |

### 2.1 Framework migration對既有研究的處理

- 2026-08-26：`AUD-mr13r-joint-capital-drawdown` schema-v2已取得決策充分結果並停止：OOS/Rolling Conditional-MFE→actual MFE Dailyρ=`0.342/0.376`、Raw Safety→actual Safety=`0.339/0.364`，但joint-product→HM/HS僅=`0.035/0.058`，predicted S5×M5兩mode皆只有`N=1`；S5 cohort MFE rho仍=`0.151/0.169`，因此不是兩head marginal learnability消失，而是global joint geometry近乎反對角。True-MTM drawdown全部reconcile canonical Equity Δ，且最大回撤主要由Low-MFE quadrants貢獻（例如Rolling C73 LM/HS+LM/LS=`-14.84%Peak`對總DD=`-16.11%`），不支持先做portfolio concentration repair。Decision=`MARGINAL_SIGNAL_PRESENT / JOINT_GEOMETRY_COLLAPSED`；Audit保留read-only evidence，NEXT改為MR-13S model-only Safety→absolute Raw-MFE controlled contrast，不新增C75。
- 2026-08-26：`AUD-mr13r-joint-capital-drawdown`初版正式輸出暴露兩個reporting/accounting缺口：terminal未完整顯示既有5×5/Safety cohort evidence，且drawdown `Overlap ΣR`把peak→trough期間重疊交易的**最終完整 realized R**誤當MDD contribution（已觀察到OOS C72 `+8.29R`、Rolling C73 `+43.07R`仍對應負MDD，足以否定該欄位語意）。同一Audit identity已升schema v2：Joint顯示每格`N / HM/HS%`、upper-right N、Safety cohort rho/High-MFE/HM/HS與joint-product rho；drawdown以封存pair fee/tax、實際交易cashflow與canonical Close重建peak-EOD→trough-EOD position MTM Δ，硬性reconcile Equity Δ並依HM/HS四象限及持倉生命週期分組。正式結果仍`RESULT_PENDING`；不新增C75、不重訓模型、不恢復K/R0。
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

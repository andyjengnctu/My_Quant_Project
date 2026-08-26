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
| 1 | **MR-13T Raw-data Direct HM/HS Tri-head Model Gate**：在改representation前，先測raw 300×10是否已有可直接辨識HM/HS的資訊，只是MR-13S marginal-only supervision沒有抽出joint intersection。 | **ACTIVE / IMPLEMENTED / RESULT_PENDING** | MR-13S為strict control；固定raw 300×10、S/U targets、shared InceptionTime trunk超參數、Raw Safety與Safety-conditioned Raw-MFE heads、daily-universal universe、Seed42、split、optimizer、full-list Delta-NDCG及Raw-MFE epoch selection。唯一scientific change為shared latent新增Direct HM/HS head，H=`1[S>=0.5 AND U>=0.5]`，三head固定等權loss；Joint head不讀marginal predicted scores，不加人工input，不使用breakout/strategy state或OOS fitting。 | Primary Gate比較Direct Joint vs same-model `Safety×RawMFE` product：Forward OOS direct Pair/PR-AUC/Top10 HM/HS enrichment需明確優於product，Breakout不得方向反轉；同時Raw Safety/Raw-MFE marginal learnability不得出現實質崩潰。若GO，證明raw latent可承載joint info並優先研究joint supervision/representation讀出；若FAIL，才進raw-data joint architecture/representation，handcrafted path-state更後。 | 執行`[1] 訓練目前模型 → Forward-OOS標準Model SOP`；只做Seed42 frozen model Gate，不建PIT/Strategy arm。 |
| 2 | **MR-13S strategy conversion**。 | **CLOSED / MODEL_GATE_FAIL** | MR-13S marginal heads雖強，但predicted S5×M5=`0`、joint-product→HM/HS Dailyρ=`0.0380`。 | Model Gate未達事前joint-support條件，因此不得建立C75或其他conversion。Truth Geometry也已證明不是Breakout universe缺乏upper-right。 | 關閉；只有未來新的model identity先通過joint Gate才重新考慮strategy conversion。 |
| 3 | **Portfolio construction / drawdown mechanism**。 | **CONDITIONAL / DEFERRED** | 不把future truth放入runtime；一次只改一個portfolio-level mechanism，與single-stock model objective分離。 | Schema-v2 Audit顯示top drawdown主要由Low-MFE positions貢獻，且MR-13S truth control確認Breakout本身存在joint support；只有後續更好的single-stock joint conversion仍被portfolio covariance主導才GO。 | 延後。 |
| 4 | **Primary-preserving MR-13P runtime condition**。 | **CONDITIONAL / FALLBACK** | 固定MR-13P模型與dual-head PIT；不做score fusion、固定權重、OOS-derived threshold或basket-wide safety floor。 | 只有representation/joint formulation線停止後，且既有secondary safety signal仍可能改變conversion決策才GO。 | 後做。 |
| 5 | **Plan C-M：PIT-safe predicted-upside context → Conditional Low-Adverse model**。 | **CONDITIONAL / LATER** | Stage-1 upside context必須rolling cross-fitted/PIT-safe同日percentile；Stage-2不得讀full-fit score。 | 只有representation-level方向無足夠增量且conditional architecture仍有未解model問題時才GO。 | 後做。 |
| 6 | **Survivable Pure-MFE / First-Passage target family**。 | **CONDITIONAL / FALLBACK_AFTER_REPRESENTATION_LINE** | Daily-universal、PIT-safe、strategy-agnostic；一次只改一個Target，不引入strategy state。 | Representation與conditional formulation已有足夠否定證據才GO。 | 更後。 |
| 7 | **Extending-Window Multi-seed Robustness Test**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | 與single-seed共用`extending_current` Compare Suite與same-seed SSOT。 | 只有未來新model通過joint Gate並產生真正有決策價值的strategy candidate才投入；C71-C74/MR-13S不進robustness。 | 延後。 |
| 8 | **Fixed-Window Stability Test**。 | **CONDITIONAL / AFTER_MODEL_OR_CONVERSION_CANDIDATE** | fixed train window=`120M`、inner validation=`24M`。 | 只有candidate已通過primary Forward Gate且history length仍可能改變決策才做。 | 延後。 |
| 9 | **sizing / regime gating / cross-sectional representation**。 | **CONDITIONAL** | 一次只改一個主要研究維度。 | 更直接的representation/joint/portfolio證據不足且新證據可能改變決策時才展開。 | 最後。 |

### 2.1 Framework migration對既有研究的處理

- 2026-08-27：MR-13S之後的下一步由「直接改representation」修正為先做`MR-13T` direct HM/HS supervision control。理由：MR-13S只以S/U兩個marginal loss訓練，HM/HS僅是OOS diagnostic；在尚未直接要求模型辨識upper-right前，不能把joint collapse單獨歸因input representation。MR-13T保持raw 300×10與MR-13S trunk/marginal heads不變，只新增shared-latent H head；若仍FAIL才啟動raw-data architecture/representation研究。
- 2026-08-27：Reusable Audit Planned membership dependency已修正：generic Opportunity/Trade reports以canonical execution sidecar＋orderable join為SSOT，不再要求Max-DL專屬`feasible_ascent_final` selector trace；No-K/No-R0 selectors合法無該stage不得BLOCK。Resource Constraint one-time Audit仍保留selector trace/daily-capacity依賴。純engineering fix，不改Queue scientific priority。
- 2026-08-26：MR-13S Actual Truth Geometry已完成：Daily actual S↔MFE rho=`-0.1261`、S5×M5=`23,242 / 3.82% / 0.92×`、S4+×M4+=`85,047 / 13.98% / 0.87×`；Breakout actual rho=`-0.0547`、S5×M5=`986 / 5.68% / 1.09×`、S4+×M4+=`2,725 / 15.71% / 0.93×`。相對predicted Daily/Breakout relation=`-0.9373/-0.9029`且S5×M5=`0`，因此Decision=`ACTUAL_JOINT_SUPPORT_CONFIRMED / BREAKOUT_STRUCTURAL_TRADEOFF_REJECTED / PREDICTED_COMPONENT_COLLAPSE_CONFIRMED`。Truth Geometry不再是待辦；下一個scientific問題限定representation-level controlled experiment。
- 2026-08-27：Research report architecture完成第二階段收斂：Model/Strategy一鍵SOP維持；Reusable Audit不再用四個混合family，而固定為`Opportunity／Selection → Trade Outcome／Path → Portfolio／Drawdown`三個原因層。Daily/Breakout actual 5×5由canonical geometry SSOT共用；Breakout只filter daily percentile。Pair/OOS-Rolling/year/regime/seed改為lens；K/R0/cash/slot binding移出常駐Selection report，保留`AUD-selection-k-r0-attribution` one-time retained。MR-13R/C69可泛化logic已遷移，one-time identity轉Historical/disabled。此工程不改research priority、K/R0或scientific semantics。
- 2026-08-26：MR-13S Forward Model Gate已完成：Forward OOS Raw Safety/Raw-MFE Dailyρ=`0.3556/0.3948`、Breakout=`0.3165/0.3567`，但joint-product→HM/HS僅=`0.0380`、predicted S5×M5=`N=0`，故strategy conversion Gate FAIL且不建立C75。因S5內Raw-MFE→actual MFE仍=`0.2587`，在直接進representation experiment前先補同一identity的read-only Actual Truth Geometry Control；Breakout只filter daily-universal percentile，不subset rerank。
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

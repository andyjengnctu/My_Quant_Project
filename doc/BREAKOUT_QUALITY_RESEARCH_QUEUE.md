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
| 1 | **MR-13P / C64 Single-model Conditional MFE–Safety Conversion Gate**：Model Gate已確認Primary MFE保留且Conditional Safety可Forward learn；現在只回答model-level conditional safety能否改善實際portfolio conversion。 | **ACTIVE / MODEL_GATE_GO / SINGLE_SEED_CONVERSION_GATE** | C64與C58共用Min `base-finalist-best` params、all-off、K/R0、canonical sizing/cash/execution；Primary=`CONT13P_ROLL` Pure-MFE，secondary=同一dual-head PIT artifact的`conditional_safety_score`，直接套baseline-relative coverage+score-sum hard floor。因target已conditionalize，runtime不得再做`same_day_rank_ols_v1`、score fusion或threshold sweep。 | 主Gate=`+2R前初始Stop↓`；同時要求`+1R/+3R`方向一致、Full-MFE不實質下降、Adverse可接受、Realized EV↑，並由Return/RoMD確認經濟效果。先比C58/C59/C60；若不能明顯改善C59即停止MR-13P conversion，不進robustness/stability。 | 先跑single-seed OOS Strategy Compare取得C64 conversion evidence；只有OOS仍不足以做GO/REJECT且Rolling可能改變決策時才補single-seed Rolling。找到明顯優於C59的candidate後才進Multi-seed robustness／Fixed-Window。 |
| 2 | **Survivable Pure-MFE**：若MR-13P conditional formulation無法改善conversion，再隔離hard risk-truncation本身是否比continuous adverse condition更有效。 | **CONDITIONAL / FALLBACK_AFTER_MR13P** | Daily-universal strategy-agnostic universe、InceptionTime family、Full-list pairwise、現有canonical fixed-risk geometry、Seed42；只改Target為first canonical risk breach前的Pure-MFE，不引入cash／holdings／candidate state，不用OOS統計做fitting。未真正開始前不再預占新MR identity。 | 只有MR-13P Model/Conversion Gate不足以支持下一步時才GO；single-seed OOS不優於C59即停止，不先做robustness。 | MR-13P結果後再決定。 |
| 3 | **First-Passage / competing-risk Target**：若 continuous survivable-MFE 仍無法把「先達 upside vs 先碰 risk」直接轉成可實現交易，研究 event-order learning。 | **CONDITIONAL / AFTER_SURVIVABLE_TARGET** | 保持 daily-universal、PIT-safe、strategy-agnostic；優先 parameter-free／continuous survival formulation，避免由 OOS 挑 `+1R/+2R/+3R` magic threshold。 | 只有 survivable-MFE 已證明 risk-order 是關鍵但 continuous Target 仍轉化不足時才做；不得因單一 OOS 門檻最好看就把該門檻回灌 fitting。 | 與 path-dynamics 依前一實驗結果二選一，不平行擴張。 |
| 4 | **Plan C-M：13K PIT-safe upside context → Conditional Low-Adverse model**。 | **CONDITIONAL / DEPRIORITIZED_BY_FIRST_PASSAGE** | Stage-1 13K context必須rolling cross-fitted/PIT-safe，同日percentile；Stage-2 Target固定MR-13M low-adverse。不得把full-fit score回灌training rows。 | 只有新的 conversion evidence 顯示 adverse 在 survivability 之外仍有穩定 incremental information 時才GO；Conditional adverse ranking需明顯優於MR-13M且strategy conversion保留upside，否則停止stacking。 | 不再作 conversion 主線第一選擇。 |
| 5 | **Extending-Window Multi-seed Robustness Test：end-to-end benchmark alignment**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | 與single-seed共用`extending_current` Compare Suite，不另列arm matrix；benchmark seed identity、Optimizer strategy params與E/K/M/P模型仍遵守既有same-seed SSOT。 | 只有新 candidate 在 single-seed OOS、必要的Rolling中相對 C59 有足以改變決策的 conversion／strategy improvement 才投入；OOS robustness 已足以否決就不跑 Rolling robustness。 | 找到更好 candidate 後再長跑，不作目前開發前置。 |
| 6 | **Fixed-Window Stability Test**：控制歷史長度後檢查新 candidate 與reference的時間敏感性。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | fixed train window=`120M`、inner validation=`24M`；與Extending輸出隔離，不作production performance Truth。 | 只有候選已改善 conversion，且歷史長度／time-regime stability 仍可能改變GO/REJECT時才執行。 | robustness／stability 都延後至 candidate development 後。 |
| 7 | **Plan B3：13K confidence boundary + 13M tie-break**。 | **CONDITIONAL** | 不用magic score-gap；若未來執行 robustness，可優先利用cross-seed rank stability建立confidence contract。 | 只有能建立無調參confidence contract且conversion主線仍需要secondary safety tie-break才GO。 | 後做。 |
| 8 | **Plan C-K：13M PIT-safe safety context → Conditional MFE model**。 | **CONDITIONAL** | Stage-1 13M context須rolling PIT-safe；Stage-2 Target固定Pure MFE。 | 只有C-M仍具研究價值且未解決問題時才做。 | C-M後。 |
| 9 | **13M sizing / regime gating / cross-sectional representation**。 | **CONDITIONAL** | 保持一次只改一個主要研究維度。 | 只有前述較直接的conversion方案不足且新證據可能改變決策時才展開。 | 最後。 |

### 2.1 Framework migration對既有研究的處理

- 2026-08-23：current single-seed OOS／Rolling與First-Passage evidence已取得；使用者將研究主線改為**Conversion-first**。在找到相對C59更好的candidate前，Extending Multi-seed Robustness與Fixed-Window Stability均降為後續validation gate。MFE／MAE component learnability均已明確，`MR-13P` Seed42 Forward Model Gate已GO，current主線進入C64 single-seed conversion Gate；Survivable Pure-MFE仍為C64失敗後的controlled fallback。
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

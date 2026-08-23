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
| 1 | **MR-13P Single-model Conditional MFE–Safety**：利用Pure-MFE與Low-Adverse各自已證明的learnability，在單一shared encoder中以MFE為primary axis，secondary只學「同等MFE下異常低adverse」的conditional residual safety，避免兩個獨立expert於runtime互相抵消。 | **ACTIVE / FORWARD_RESULT_AVAILABLE_PENDING_CONDITIONAL_REVIEW / CONVERSION_FIRST** | Daily-universal strategy-agnostic universe、300×10、40D full horizon、Seed42、Full-list Delta-NDCG pairwise；architecture=`inception_time_conditional_mfe_safety_v1`。Primary target=同日Pure-MFE percentile；secondary target=同日Low-Adverse percentile對true MFE percentile含intercept OLS residual後再同日percentile化。Conditional head接收shared latent+stop-gradient primary prediction；不使用OOS統計、strategy state、score fusion weight或magic threshold。 | 先做Forward model Gate：Conditional Safety需有穩定正向Daily/Pair learnability，且MFE head不能相對既有13K evidence出現明顯崩壞；Model Gate通過才建立最小single-seed conversion strategy contrast。最終主Gate仍是`+2R前初始Stop↓`且Full-MFE／Realized EV／Return/RoMD不反向惡化。失敗即停止，不先robustness。 | Seed42 Forward training已完成；下一步不重訓，直接由模型選單「查看目前Workflow、工件與模型報表」重畫既有canonical report並審閱Conditional Safety Forward-OOS／breakout metrics。只有Model Gate審閱通過才建立最小single-seed conversion strategy contrast；審閱前不建PIT/runtime source/strategy arm。 |
| 2 | **Survivable Pure-MFE**：若MR-13P conditional formulation無法改善conversion，再隔離hard risk-truncation本身是否比continuous adverse condition更有效。 | **CONDITIONAL / FALLBACK_AFTER_MR13P** | Daily-universal strategy-agnostic universe、InceptionTime family、Full-list pairwise、現有canonical fixed-risk geometry、Seed42；只改Target為first canonical risk breach前的Pure-MFE，不引入cash／holdings／candidate state，不用OOS統計做fitting。未真正開始前不再預占新MR identity。 | 只有MR-13P Model/Conversion Gate不足以支持下一步時才GO；single-seed OOS不優於C59即停止，不先做robustness。 | MR-13P結果後再決定。 |
| 3 | **First-Passage / competing-risk Target**：若 continuous survivable-MFE 仍無法把「先達 upside vs 先碰 risk」直接轉成可實現交易，研究 event-order learning。 | **CONDITIONAL / AFTER_SURVIVABLE_TARGET** | 保持 daily-universal、PIT-safe、strategy-agnostic；優先 parameter-free／continuous survival formulation，避免由 OOS 挑 `+1R/+2R/+3R` magic threshold。 | 只有 survivable-MFE 已證明 risk-order 是關鍵但 continuous Target 仍轉化不足時才做；不得因單一 OOS 門檻最好看就把該門檻回灌 fitting。 | 與 path-dynamics 依前一實驗結果二選一，不平行擴張。 |
| 4 | **Plan C-M：13K PIT-safe upside context → Conditional Low-Adverse model**。 | **CONDITIONAL / DEPRIORITIZED_BY_FIRST_PASSAGE** | Stage-1 13K context必須rolling cross-fitted/PIT-safe，同日percentile；Stage-2 Target固定MR-13M low-adverse。不得把full-fit score回灌training rows。 | 只有新的 conversion evidence 顯示 adverse 在 survivability 之外仍有穩定 incremental information 時才GO；Conditional adverse ranking需明顯優於MR-13M且strategy conversion保留upside，否則停止stacking。 | 不再作 conversion 主線第一選擇。 |
| 5 | **Extending-Window Multi-seed Robustness Test：end-to-end benchmark alignment**。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | 與single-seed共用`extending_current`六arm Compare Suite；benchmark seed identity、Optimizer strategy params與E/K/M模型仍遵守既有same-seed SSOT。 | 只有新 candidate 在 single-seed OOS、必要的Rolling中相對 C59 有足以改變決策的 conversion／strategy improvement 才投入；OOS robustness 已足以否決就不跑 Rolling robustness。 | 找到更好 candidate 後再長跑，不作目前開發前置。 |
| 6 | **Fixed-Window Stability Test**：控制歷史長度後檢查新 candidate 與reference的時間敏感性。 | **CONDITIONAL / AFTER_BETTER_CONVERSION_CANDIDATE** | fixed train window=`120M`、inner validation=`24M`；與Extending輸出隔離，不作production performance Truth。 | 只有候選已改善 conversion，且歷史長度／time-regime stability 仍可能改變GO/REJECT時才執行。 | robustness／stability 都延後至 candidate development 後。 |
| 7 | **Plan B3：13K confidence boundary + 13M tie-break**。 | **CONDITIONAL** | 不用magic score-gap；若未來執行 robustness，可優先利用cross-seed rank stability建立confidence contract。 | 只有能建立無調參confidence contract且conversion主線仍需要secondary safety tie-break才GO。 | 後做。 |
| 8 | **Plan C-K：13M PIT-safe safety context → Conditional MFE model**。 | **CONDITIONAL** | Stage-1 13M context須rolling PIT-safe；Stage-2 Target固定Pure MFE。 | 只有C-M仍具研究價值且未解決問題時才做。 | C-M後。 |
| 9 | **13M sizing / regime gating / cross-sectional representation**。 | **CONDITIONAL** | 保持一次只改一個主要研究維度。 | 只有前述較直接的conversion方案不足且新證據可能改變決策時才展開。 | 最後。 |

### 2.1 Framework migration對既有研究的處理

- 2026-08-23：current single-seed OOS／Rolling與First-Passage evidence已取得；使用者將研究主線改為**Conversion-first**。在找到相對C59更好的candidate前，Extending Multi-seed Robustness與Fixed-Window Stability均降為後續validation gate。MFE／MAE component learnability均已明確，最新決策先做`MR-13P`單模型Conditional MFE–Safety，Survivable Pure-MFE改為其失敗後的controlled fallback。
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

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
| 1 | **Plan C-M：13K PIT-safe upside context → Conditional Low-Adverse model**。問題：已知upside condition後，第二模型能否學出「同樣高upside中誰較不會先跌」？ | **PLANNED / NEXT** | Stage-1 13K context必須cross-fitted/PIT-safe，建議同日percentile；Stage-2仍看原sequence，Target固定MR-13M low-adverse，與MR-13M source-only比較。不得把full-fit 13K score回灌training rows。 | Conditional adverse ranking需明顯優於MR-13M，且後續strategy conversion能保留13K upside；否則停止stacking。 | 下一個主線；開始實作時才建立新的model/profile identity。 |
| 2 | **Plan A：MR-13H economic Target + parameter-free path-dynamics input representation**。問題：原300×10是否缺少joint MFE/path-risk表示？ | **PLANNED** | Target固定MR-13H `MFE-adverse`、InceptionTime family、Full-list pairwise、Seed42；只新增無可調窗口的causal path primitives。 | Forward rho/Pair/Top-K與breakout slice同方向明顯勝MR-13H才進PIT；近似或更差即停止簡單path primitives。 | Plan C-M之後；開始時才建立新ARCH/MR identity。 |
| 3 | **Plan B3：13K confidence boundary + 13M tie-break**。問題：13M只在13K自身不確定時介入，能否保留明確winner？ | **CONDITIONAL** | 不使用固定score-gap magic threshold；優先以既有multi-seed rank stability/confidence定義邊界。 | 只有能建立無調參confidence contract才GO。 | 需先完成13K robustness證據。 |
| 4 | **Plan C-K：13M PIT-safe safety context → Conditional MFE model**。 | **CONDITIONAL** | Stage-1 13M context須PIT-safe；Stage-2 Target固定Pure MFE，與MR-13K source-only比較。 | 只有C-M未解決且仍有conditional-stacking價值時才做。 | 排在C-M後。 |
| 5 | **13M → Position Sizing**。13K決定買誰，13M只決定risk allocation。 | **CONDITIONAL** | 不改membership；會改canonical 1% sizing，因此屬較大策略變因。 | 只有selection-level整合不足且仍需利用13M降MDD時才做。 | 後做。 |
| 6 | **Regime-dependent 13K / 13M gating**。 | **CONDITIONAL** | 需先有causal、非績效調參的market-state gate。 | 只有前述方法不足時才考慮。 | 後做。 |
| 7 | **Cross-sectional / market-state representation**。 | **CONDITIONAL** | 不重跑已淘汰的單純architecture橫向搜尋。 | 只有Plan A證明簡單path representation不足時才GO。 | 最後的representation升級方向。 |
| — | **MR-13K vs MR-13E exact 8-seed robustness**：Selection `C42/C53`、Forward `C44/C54`。 | **ACTIVE / independent** | `seed_count=8`、generator=`20260810`；不挑best、不ensemble；production仍C42/C44。 | 完成paired robustness後結案13K source-only路線。 | 可與Plan C-M獨立執行，C55/C56不得加入robustness matrix。 |

### 2.1 最近完成且有正向證據

- **Plan B2 / `SR-C56`：CLOSED / FORWARD GO / NOT PROMOTED**。Seed42 Forward C56=`148.34% Return / 12.86% MDD / 11.53 RoMD / 47.90% Win Rate / 1.14R EV`；相對C54保留約`87.9%` Return，同時MDD `-7.09pp`、RoMD `+3.08`、Win Rate `+5.35pp`、EV `+0.34R`，並明顯優於C55 raw-safety。B2已證明Residual Safety具有portfolio conversion價值；但仍低於production C44的Return/RoMD且只有Seed42 Forward證據，因此不promotion、不加入current robustness，研究主線轉Plan C-M。

## 3. 已停止的相鄰方向（不得重新包裝成新項目）

- MR-13L：MFE/adverse raw-R dual regression後直接相減。
- MR-13N：single-model 50/50 equal-rank composite。
- MR-13K + MR-13M frozen 50/50 score fusion；兩expert score近鏡像而互相抵消。
- MR-13O：Pareto-dominance pairwise；Forward Pareto僅弱於可進PIT的程度。
- SR-C55：raw MR-13M baseline-relative basket safety floor；雖降MDD與提高勝率，但Return/Payoff/RoMD轉換不佳，已被B2 residual safety取代。
- 不掃 `MFE - λ × adverse`、30/70～70/30 score權重、Pareto threshold、loss小改來追OOS。

本節只摘要 stop direction；正式數值與結案理由仍以 Registry／Experiment Log 為準。

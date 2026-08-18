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
| 1 | **方案 B：MR-13K upside objective + MR-13M path-safety hard constraint**。問題：13K 能抓大 winner 但策略轉換較差，13M 是否可作 portfolio guardrail，在不把兩模型 score 加權的情況下提高勝率／EV／RoMD？ | **ACTIVE / `SR-C55` Forward-only** | Min ROOS、all-off、K、R0、canonical sizing/cash/orderability/execution、`exact_branch_and_bound_v1`、MR-13K primary score全部沿用C54；MR-13M只要求選中basket的 safety score coverage 與 score-sum不得低於同日DL-off Min ROOS baseline；無lambda、無固定threshold、無score fusion。 | **GO**：C55相對C54出現有意義的portfolio translation改善，且safety floor無違規，尤其關注Win Rate／EV／RoMD、MDD與Payoff是否保留；**STOP**：只換來安全但Return/EV/RoMD明顯惡化，或幾乎無法改善C54。 | 先只跑Seed42 Forward。GO才建立MR-13M Selection PIT並設計Selection版dual-model arm；STOP則不建立13M PIT。 |
| 2 | **方案 A：MR-13H economic Target + parameter-free path-dynamics input representation**。問題：現有300×10 OHLCV表示是否不足以辨識「高MFE且低adverse」的joint residual？ | **PLANNED** | Target固定MR-13H `MFE - adverse-to-peak`、同一daily universe、InceptionTime family、Full-list pairwise、Seed42；只改input representation。第一版只加入無可調窗口的一階path primitives，如close return、overnight gap、intraday return、high-low range、close location、volume change及stock-vs-benchmark relative moves。 | **GO**：Forward Daily rho／Pair與breakout slice同方向明顯優於MR-13H，且Top-K品質改善；**STOP**：與MR-13H近似或更差，表示簡單path primitives沒有新增可用joint information。 | 方案B完成決策後實作；開始實作時才依Registry建立新的 `ARCH-*` / `MR-*` identity。 |
| 3 | **Cross-sectional / market-state representation**。問題：若方案A失敗，joint quality是否需要同日市場橫截面或更結構性的market-state資訊？ | **CONDITIONAL** | 不重跑已淘汰的ModernTCN／Patch Transformer等單純architecture橫向搜尋；Target仍以已確認的economic objective為準。 | 只有方案A證明簡單path representation不足時才GO；否則保持BLOCKED。 | 先定義最小cross-sectional causal representation，再單變因驗證。 |
| 4 | **MR-13K vs MR-13E exact 8-seed robustness**（Selection `C42/C53`、Forward `C44/C54`）。問題：13K single-seed strategy translation較差是否跨seed一致？ | **ACTIVE / independent** | `seed_count=8`、`seed_generator_seed=20260810`；不挑best seed、不ensemble；production仍C42/C44。 | 完成兩stage paired robustness後整體結案MR-13K source-only路線。 | 可與方案B工程驗證獨立執行；不得因C55加入而改 robustness matrix。 |

---

## 3. 已停止的相鄰方向（不得重新包裝成新項目）

- MR-13L：MFE/adverse raw-R dual regression後直接相減。
- MR-13N：single-model 50/50 equal-rank composite。
- MR-13K + MR-13M frozen 50/50 score fusion；兩expert score近鏡像而互相抵消。
- MR-13O：Pareto-dominance pairwise；Forward Pareto僅弱於可進PIT的程度。
- 不掃 `MFE - λ × adverse`、30/70～70/30 score權重、Pareto threshold、loss小改來追OOS。

本節只摘要 stop direction；正式數值與結案理由仍以 Registry／Experiment Log 為準。

# Breakout Quality Filter 實驗紀錄與後續計畫

## 1. 用途與更新規則

本文件是 `breakout_quality filter` 改善工作的單一實驗紀錄。凡要分析、修改、比較或測試此 Filter，開始前必須先讀本文件，避免重複測試已淘汰方向或遺失目前基準。

每次完成一個實驗後，必須在同一輪更新本文件，至少記錄：

1. 日期與實驗狀態：`PLANNED`、`IMPLEMENTED`、`RESULT_AVAILABLE`、`ACCEPTED` 或 `REJECTED`。
2. 程式基準：ZIP／commit／patch 名稱與 SHA256；未知時明確標示未知。
3. 唯一變更、所有固定條件，以及是否需要重建 Dataset／重新 Label。
4. Selection 與 OOS 的主要結果，至少包含原始 PASS、模型 PASS、PASS Precision、Precision Lift、PASS Recall、Accuracy、平均 Score。
5. 與當前正式基準的差異、結論、是否採用，以及下一個單一變更。
6. 尚未取得結果的實作只能標記為 `IMPLEMENTED`，不得先寫成有效或無效。

### OOS 使用規則（2026-07-29 起）

- OOS 可持續用於模型／架構結果比較、錯誤歸因、年度與 regime 診斷、策略經濟效果評估，以及決定下一個單一變更。
- Train／Validation、loss、gradient、early stopping、epoch 選擇、threshold／calibration 擬合、normalization、feature／label 建立、sample weighting 與 hyperparameter optimization，不得讀取或使用 OOS rows、labels、scores 或其統計量。
- 每個新實驗仍以 Selection 內的 Inner Train／Validation 完成訓練與選 epoch；完整 OOS 只在模型凍結後執行，並可用來接受、淘汰或形成下一個實驗。
- 同一 OOS 經多次比較後，結果標記為「迭代研究 OOS 證據」，不宣稱是 untouched holdout；但這不構成停止研究或等待新資料的理由。
- 後續建議必須優先提供可立即執行的實驗、實作、診斷或修正；不得把等待新的 forward labeled period 當成主要下一步。
- 本規則取代文件中所有「因 OOS 已查看而不得再研究」或「只能等待新資料」的概括性限制；個別已淘汰方向仍維持淘汰，除非提出本質不同的新機制。

本文件只記錄已知事實。歷史結果若缺少完整報表，會標記「精確值未保留」，不得自行補值。歷史資料整理截止日為 **2026-08-02**。

---

## 2. 目前基準

### 2.1 程式基準

| 項目 | 目前狀態 |
|---|---|
| 基準 ZIP | 本輪來源 `test-branch-1_20260802_034615_439e600(3).zip`，SHA256 `268eed2ee4312891166197edb7257827d94752f09ebdd8a71b2ba3b0b58085b1`；沿用2014-01-01～2020-12-31 Selection PIT Baseline／Score Sort結果、read-only capture attribution與共用console報表實作；本輪再統一`[1/Enter] 模型研究與驗證`的狀態文字／PIT模型報表色彩，並將Dataset建立進度限制在單一終端列，避免長行自動換行洗版 |
| SHA256 | 來源ZIP SHA256 `268eed2ee4312891166197edb7257827d94752f09ebdd8a71b2ba3b0b58085b1`。既有3.78結果維持：Baseline淨總報酬182.62%、MDD 13.18%、RoMD 13.86；Score Sort淨總報酬144.80%、MDD 21.53%、RoMD 6.73。模型層通過、Sort Only拒絕；本輪只調整使用者可見報表與進度顯示，不改寫或重算上述績效 |
| 程式版本範圍 | Active architectures為9A `inception_time_v1`排序／高品質基準與8F `multiscale_cnn_sequence_only_v1`高覆蓋基準；10A `inception_time_market_set_candidate_v1`與Global Stage 1 `inception_time_market_set_v1`均維持legacy read-only；9A-GN、9B、9C、9D、9E與9F同樣只供舊工件重建 |
| Policy 預設 | architecture=`inception_time_v1`、filter id=`breakout_quality_v1`；depth=`6`、kernels=`39/19/9`、RF=`229 bars`；experiment profile=`unique_group_sampling`；batch=`128 groups`、patience=`1`、final refit=`selected_epochs`；device=`auto`、mixed precision=`true/auto dtype`、deterministic=`true`、TF32=`false` |
| 當前最佳實證模型 | 9A `inception_time_v1 / unique_group_sampling / threshold 0.5` 為新的排序／高品質模型基準；8F `multiscale_cnn_sequence_only_v1` 保留為高覆蓋基準 |
| Dataset | 正式policy退回既有 `breakout_quality_v1` 300×10 feature bank與固定百分比Label；不需重建Dataset、relabel、9A checkpoint或9A scores。10A Market Bank、checkpoint、manifest與research scores保留於獨立legacy路徑供歷史重現 |

使用者所稱「退回 v8 版本」是退回**尚未加入 v9 auxiliary head 的程式版本**，不是把 policy 預設改成 `multiscale_cnn_v8`。目前active architectures為已接受的9A `inception_time_v1`排序／高品質基準與8F `multiscale_cnn_sequence_only_v1`高覆蓋基準；10A Candidate-conditioned Market Set與Stage 1 Global Market Set均已淘汰並轉為legacy read-only；9A-GN、9B ModernTCN、9C TS2Vec、9D MantisV2、9E MOMENT與9F Patch Transformer只保留legacy read-only compatibility。

### 2.2 固定 Label 與訓練條件

| 項目 | 固定值 |
|---|---:|
| Feature Window | 300 bars |
| Label Horizon | 40 bars |
| PASS 最低 MFE | 嚴格大於 5% |
| PASS 最低 MFE／MAE | 嚴格大於 2.0 |
| 最大不利跌幅 | 觸及 −10% 即 REJECT |
| Epoch 上限 | 200 |
| Early-stopping patience | **1**；8G patience 5 已淘汰 |
| Batch Size | **128** unique `ticker/date` groups；8H batch size 64 已淘汰 |
| Optimizer | `adam`（6A AdamW 與 6B schedule 已淘汰） |
| LR Schedule | `none` |
| Augmentation | `none`；7A masking 已淘汰 |
| Learning Rate | 0.0003 |
| Weight Decay | 0.0001 |
| Gradient Clip | 1.0 |
| Threshold | 0.5 |
| Seed | `BREAKOUT_QUALITY_RANDOM_SEED=42`；binary、continuous、pretraining與Selection PIT workflow共用 |
| Final Model Mode | **`selected_epochs`**；8I matched optimizer steps 與 8J direct best checkpoint 均已淘汰 |
| Class Weight | `none` |
| Time Weight | `none`；8K `date_balanced` 已淘汰，只保留歷史重現 |
| Training Sampling | `unique_ticker_date`；每個 group 每個 epoch 只進入一次 optimizer sampling |

### 2.3 正式研究基準：8F `multiscale_cnn_sequence_only_v1 / unique_group_sampling`

| OOS 指標 | 8F 基準 |
|---|---:|
| 原始 PASS | 55.63% |
| PASS Precision | 58.46% |
| Precision Lift | +2.83 pp |
| PASS Recall | 76.96% |
| 模型 PASS | 73.24% |
| Accuracy | 56.76% |
| 平均 Score | 0.5497 |
| Selection→OOS Precision 差 | **+1.56 pp** |
| Selection→OOS Accuracy 差 | −0.70 pp |
| Selection→OOS Score 差 | −0.0287 |

8A `baseline` 仍保留為較高 Precision 的歷史參考（OOS Precision 59.30%、Lift +3.66 pp），但其 Recall、Accuracy、Score 與 Selection→OOS drift 明顯較差。後續實驗以 8F 為比較基準，不能用 Selection Precision 上升掩蓋完整 OOS Recall、Accuracy、Score 或泛化落差惡化。


### 2.4 新排序／高品質基準：9A `inception_time_v1 / unique_group_sampling`

| OOS 指標 | 9A InceptionTime |
|---|---:|
| 原始 PASS | 55.63% |
| PASS Precision | **62.99%** |
| Precision Lift | **+7.35 pp** |
| PASS Recall | 51.44% |
| 模型 PASS | 45.43% |
| Accuracy | 56.17% |
| 平均 Score | 0.4842 |
| PR-AUC | 0.6257 |
| Precision@50% coverage | 62.77% |
| Precision@60% coverage | 61.62% |
| Precision@70% coverage | 60.25% |
| Recall@60% Precision | 77.27% |
| Brier | 0.2453 |
| ECE | 0.0722 |

9A 在 threshold 0.5 下相較 8F 提高 OOS Precision 4.53 pp，但 Recall 下降 25.52 pp，屬較高品質、較低 coverage 的分類操作點。Selection→OOS 的固定 coverage Precision 幾乎沒有下降：P@50%、P@60%、P@70% 分別變化 +0.07、+0.28、+0.25 pp，因此仍保留為研究排序基準；但固定 threshold 0.5 的無前視策略對照已證明淨報酬、MDD、RoMD、Payoff、EV與曝險全面惡化，故不得作正式 runtime gate。8F只保留歷史高覆蓋分類基準，不因9A策略失敗而自動升格部署。 交易歸因進一步確認總 R 差異為 −46.62R，其中共同交易只貢獻 +0.15R，獨有交易選擇效果為 −46.77R；No-filter only 的 286 筆交易合計 187.49R、平均 0.66R、Payoff 3.21，Filter only 的 276 筆替代交易僅 140.72R、平均 0.51R、Payoff 2.76。失效主因是篩掉與路徑擠出的大贏家高於所避開的輸家，並非共同交易執行或帳務差異。 後續Score Ranking也未形成可部署證據：`base_finalists_agree`只在同票時使用Score，總報酬雖+4.88pp但只有2023改善；更純的`base_finalist_best`全域Score第一排序使總報酬129.08%降至96.96%（−32.12pp）、RoMD 7.42降至5.32，且除2023外所有年度均落後。故9A Score不得作全域第一順位，也不得再用既有OOS調整Score／票數混合權重。

---

## 3. 已完成實驗

### 3.1 架構基線

| 日期 | 實驗 | 唯一主要變更 | 結果摘要 | 判定 |
|---|---|---|---|---|
| 2026-07-12～13 | `tiny_cnn_v1` | 小型兩層 CNN | Train loss 約只由 0.67 降至 0.65，Selection 本身即不足 | `REJECTED`；不要再退回 Tiny |
| 2026-07-13 | `residual_tcn_v1` | 6 個 residual dilated blocks | Train／Selection 學得更強，但 OOS 過度擬合明顯 | `REJECTED` |
| 2026-07-13～14 | `multiscale_cnn_v1` | Short／Medium／Long 三分支 Level input | 相較 Tiny 可學到有效 Selection 訊號；後續所有消融的正式基準 | `ACCEPTED` |

### 3.2 Refit、類別權重與時間權重

| 日期 | 實驗 | 唯一變更 | OOS Precision Lift | OOS Recall | 模型 PASS | 判定 |
|---|---|---|---:|---:|---:|---|
| 2026-07-14 | Matched optimizer steps only | `selected_epochs` → `matched_optimizer_steps` | +4.55 pp | 31.49% | 29.11% | `REJECTED`；過度保守 |
| 2026-07-14 | No class weight only | `inverse_frequency` → `none` | **+3.66 pp** | **51.49%** | **48.32%** | `ACCEPTED`；目前 v1 基準 |
| 2026-07-14 | Year balancing only | `none` → `year_balanced_sqrt` | 精確值未保留 | 精確值未保留 | 精確值未保留 | `REJECTED`；整體較差 |
| 2026-07-14 | Matched steps + no class weight | 同時使用 matched steps 與 no class weight | +4.14 pp | 41.09% | 38.25% | `REJECTED`；仍過度篩選 |

### 3.3 Branch input representation

| 日期 | 版本 | 唯一變更 | OOS Precision | Lift | Recall | 模型 PASS | Accuracy | Score | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-07-14 | v1 | Short／Medium／Long 全部 Level | **59.29%** | **+3.66 pp** | 51.49% | 48.32% | **53.34%** | 0.4791 | `ACCEPTED` |
| 2026-07-14 | v2 | Short＋Medium 改 Return／Delta | 57.35% | +1.72 pp | **56.44%** | **54.74%** | 52.42% | **0.5118** | `REJECTED`；提高 Score／Recall、降低辨識精度 |
| 2026-07-14 | v3 | v2 的個股 O/H/L/C 再減 0050 同欄 Return | 57.59% | +1.96 pp | 54.77% | 52.91% | 52.40% | 0.4995 | `REJECTED`；與 v2 資訊等價，無實質改善 |
| 2026-07-14 | v7 | 只有 Short 改 Return／Delta | 58.34% | +2.71 pp | 53.16% | 50.69% | 52.83% | 0.4870 | `REJECTED` |
| 2026-07-15 | v8 | 只有 Medium 改 Return／Delta | 57.70% | +2.07 pp | 53.91% | 51.97% | 52.38% | 0.4905 | `REJECTED` |

共同結論：Return／Delta 使用越多，通常 Score、模型 PASS 與 Recall 越高，但 Precision Lift 與 Accuracy 越低；主要是改變固定 threshold 0.5 的操作點，未提升整體可分性。不要再測 Long Return 或更多 Return／Level 組合。

### 3.4 Long Branch 容量與 regularization

| 日期 | 版本 | 唯一變更 | OOS Precision | Lift | Recall | 模型 PASS | Accuracy | Score | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-07-14 | v4 | Long channels 16 → 8 | 58.91% | +3.27 pp | 52.88% | 49.94% | 53.26% | **0.4806** | `REJECTED`；接近 v1，但未超越 |
| 2026-07-14 | v5 | Long channels 16 → 12 | 58.00% | +2.37 pp | 53.69% | 51.50% | 52.61% | 0.4779 | `REJECTED` |
| 2026-07-14 | v6 | Long dropout 0.25 → 0.40 | 58.53% | +2.90 pp | 49.70% | 47.24% | 52.43% | 0.4662 | `REJECTED`；主要指標一起退步 |

共同結論：縮減 channels 或提高 Long Branch dropout 沒有提升真正 OOS 辨識力。不要再測 Long channels 10／14、dropout 0.30／0.35／0.50 或其他 Long Branch 微調。

### 3.5 連續輔助目標：v9（2026-07-21）

v9 在 v1 shared representation 後增加 3-output auxiliary head，目標為：

- `mfe_progress`
- `mae_risk_fraction`
- `reward_risk_progress`

Loss 為 Cross-Entropy 加上權重 0.2 的 Smooth L1 auxiliary loss。此實驗已完成，之後程式退回 v9 前版本。

| 指標 | v1 基準 | v9 | v9 − v1 |
|---|---:|---:|---:|
| OOS PASS Precision | 59.29% | 58.39% | −0.90 pp |
| OOS Precision Lift | +3.66 pp | +2.76 pp | −0.90 pp |
| OOS PASS Recall | 51.49% | 47.26% | −4.23 pp |
| OOS 模型 PASS | 48.32% | 45.02% | −3.30 pp |
| OOS Accuracy | 53.34% | 51.93% | −1.41 pp |
| OOS 平均 Score | 0.4791 | 0.4445 | −0.0346 |
| Selection→OOS Precision 差 | −2.85 pp | −4.21 pp | 惡化 1.36 pp |
| Selection→OOS Score 差 | −0.1129 | −0.1388 | 惡化 0.0259 |

Selection Precision 升至 62.61%、Lift 升至 +7.80 pp，但 OOS 全面退步，顯示 auxiliary targets 加強了 Selection 期間未來路徑幅度的擬合，沒有提升跨時期泛化。

判定：`REJECTED`。目前不要依同一段 OOS 繼續微調 auxiliary weight、target 子集合或 clip。

### 3.6 OOS horizon／更新頻率

| 日期 | 實驗 | 結果 | 判定 |
|---|---|---|---|
| 2026-07-21 | 只預測下一年 OOS | 使用者已實測，沒有比目前整段 OOS 更好；精確報表值未保留 | `REJECTED`；不要把縮短 OOS horizon 或年度 refit 當成近期優先方向 |
| 2026-07-21 | Multi-fold validation | 尚未作為模型改善手段；它只提高選模可靠度，不會自行改善單一 fold 的模型能力 | 不列為目前「提升整體 OOS」的直接實驗 |


### 3.7 Optimizer 6A：AdamW only（2026-07-22）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED` |
| 程式基準 | `test-branch-1_20260722_131354_dc805ba.zip`；SHA256 `cf5187c01421f20c0efc48f63b48016413efc628baf2c0187b5daf669f29b0ed` |
| 唯一學習變更 | `multiscale_cnn_v1 / baseline` 的 `torch.optim.Adam` → `multiscale_cnn_v1 / adamw_only` 的 `torch.optim.AdamW` |
| 固定條件 | LR 0.0003、weight decay 0.0001、scheduler=`none`、augmentation=`none`、seed 42、threshold 0.5、`selected_epochs`、class/time weight=`none` |
| Dataset／Label | 不重建、不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 75.69%、PASS Precision 62.27%、Lift +7.46 pp、Recall 86.00%、Accuracy 63.77%、Score 0.5824 |
| OOS | 原始 PASS 55.63%、模型 PASS 45.80%、PASS Precision 59.02%、Lift +3.39 pp、Recall 48.60%、Accuracy 52.63%、Score 0.4693 |
| 相較 v1 baseline | OOS Precision −0.27 pp、Lift −0.27 pp、Recall −2.89 pp、模型 PASS −2.52 pp、Accuracy −0.71 pp、Score −0.0098；Precision gap 由 −2.85 pp 惡化為 −3.25 pp；Score gap 約 −0.1130，未改善 |
| 判定 | AdamW 沒有提升 OOS，且主要指標多數退步；退回 Adam |
| 下一步 | 6B 使用 Adam，只加入 step-based warmup＋cosine schedule |

Formal bundle 閉環紀錄：2026-07-22 本地正式測試的 consistency 僅失敗 1 項：互動式 report 已正確傳遞 `--experiment-profile`，但 synthetic CLI contract 仍使用重構前的舊 expected argv。已將測試 fixture 隔離覆寫為 `baseline`，並把 expected argv 同步為 `--filter-id synthetic_quality --experiment-profile baseline --include-oos`。這是 validator 契約同步修正，不改 runtime、模型、Dataset 或 6A 學習條件；其後 6A Full OOS 已完成並判定 `REJECTED`。

### 3.8 LR Schedule 6B：Adam + step-based warmup／cosine（2026-07-22）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED` |
| 程式基準 | `test-branch-1_20260722_152735_403de13.zip`；SHA256 `0160f86de40465e547b3dd4dfa63ad673680242b9928da4e119f82759bfb9986` |
| 唯一學習變更 | `baseline` 的固定 LR → `adam_warmup_cosine` 的 step-based LR schedule；optimizer 維持 Adam |
| Experiment profile | optimizer=`adam`、schedule=`linear_warmup_cosine`、augmentation=`none` |
| Schedule | 每個 phase 各自重建；前 5% optimizer updates 線性 warmup 至 0.0003，後 95% cosine decay 至 0.00003；每次 optimizer update 前套用 |
| 固定條件 | v1、weight decay 0.0001、seed 42、threshold 0.5、`selected_epochs`、class/time weight=`none` |
| Dataset／Label | 不重建、不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 68.62%、PASS Precision 63.07%、Lift +8.26 pp、Recall 78.97%、Accuracy 63.13%、Score 0.5594 |
| OOS | 原始 PASS 55.63%、模型 PASS 37.87%、PASS Precision 58.41%、Lift +2.78 pp、Recall 39.76%、Accuracy 50.74%、Score 0.4459 |
| 相較 v1 baseline | OOS Precision／Lift 各 −0.88 pp、Recall −11.73 pp、模型 PASS −10.45 pp、Accuracy −2.60 pp、Score −0.0332；Precision gap 由 −2.85 pp 惡化為 −4.66 pp，Score gap由 −0.1129 略惡化為 −0.1135 |
| 判定 | Schedule 使模型顯著更保守，Selection Precision 上升但 OOS Precision、Recall、Accuracy 全面下降；淘汰並退回 Adam 固定 LR |
| 下一步 | 7A 只加入舊歷史 contiguous masking augmentation |

### 3.9 Augmentation 7A：舊歷史 contiguous masking（2026-07-22）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED` |
| 程式基準 | `test-branch-1_20260722_174947_5fd19d8.zip`；SHA256 `c3ebda41cceffdc5da5b88be1098aaf01f7a023ea605ecd8f6618b4a0d40fec4` |
| 唯一學習變更 | `baseline` → `history_masking_only`；只在 training batch 即時套用一段舊歷史 masking |
| Experiment profile | optimizer=`adam`、schedule=`none`、augmentation=`old_history_contiguous_mask` |
| Masking | 每筆 training sample 50% 機率；300 bars 中最近 60 bars 完全保護；舊歷史隨機遮蔽 10～30 bars；10 channels 共用同一時間區段，以左右邊界逐 channel 線性插值 |
| Validation／OOS | 完全不套用 augmentation；評估與 score export 使用原始 features |
| 固定條件 | v1、LR 0.0003、weight decay 0.0001、seed 42、threshold 0.5、`selected_epochs`、class/time weight=`none` |
| Dataset／Label | 不重建、不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 76.33%、PASS Precision 62.00%、Lift +7.19 pp、Recall 86.35%、Accuracy 63.51%、Score 0.5803 |
| OOS | 原始 PASS 55.63%、模型 PASS 45.71%、PASS Precision 58.90%、Lift +3.26 pp、Recall 48.39%、Accuracy 52.50%、Score 0.4698 |
| 相較 v1 baseline | OOS Precision／Lift 各 −0.39／−0.40 pp、Recall −3.10 pp、模型 PASS −2.61 pp、Accuracy −0.84 pp、Score −0.0093；Precision gap 由 −2.85 pp 惡化為 −3.11 pp，只有 Score gap 由 −0.1129 微幅改善為 −0.1105 |
| 判定 | Precision Lift、Recall、Accuracy 均下降，不符合 7B 啟動條件；停止輕量 augmentation 路線 |
| 下一步 | 結構性方向：新增由既有序列即時計算的低維市場 regime context |


### 3.10 結構性實驗：低維市場 Regime Context（2026-07-22）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED` |
| 程式基準 | `test-branch-1_20260722_184343_a759ee8(6).zip`；SHA256 `de8594e6b66a216867ce4b4310e31b7ec00c5047428facb2500b3c5d8b5a4a2e` |
| 唯一模型變更 | `multiscale_cnn_v1` → `multiscale_cnn_regime_context_v1`；三個 Level branches 與原 4 個 context 完全保留，只新增 6 個 deterministic regime context 經零初始化 projection 注入 head |
| Regime context | 0050 20／60 日 log return、0050 20／60 日 annualized close volatility、個股減 0050 的 20／60 日 log return |
| 資訊時點 | 全部由既有 300×10 sequence 截至事件日即時計算，不使用未來資料 |
| 固定條件 | experiment profile=`baseline`、Adam、固定 LR 0.0003、weight decay 0.0001、augmentation=`none`、seed 42、threshold 0.5、`selected_epochs`、class/time weight=`none` |
| Dataset／Label | 不重建、不 relabel；維持固定百分比 Label |
| Selection | 原始 PASS 54.81%、模型 PASS 76.38%、PASS Precision 61.92%、Lift +7.11 pp、Recall 86.30%、Accuracy 63.40%、Score 0.5902 |
| OOS | 原始 PASS 55.63%、模型 PASS 48.95%、PASS Precision 58.70%、Lift +3.07 pp、Recall 51.66%、Accuracy 52.89%、Score 0.4637 |
| 相較 v1 baseline | OOS Precision／Lift 各 −0.59 pp、Recall +0.17 pp、模型 PASS +0.63 pp、Accuracy −0.45 pp、Score −0.0154；Precision gap 由 −2.85 pp 惡化為 −3.22 pp；Score gap 由 −0.1129 惡化為 −0.1265 |
| 判定 | Regime context 只改善 Selection 內 validation loss／accuracy，沒有延續至 OOS；主要泛化指標低於 v1，因此淘汰並退回 `multiscale_cnn_v1 / baseline` |
| 下一步 | 不再把分年／分 Fold／threshold 診斷列為模型改善優先；ATR Label 已由使用者確認實測無改善。下一個直接改善整體 OOS 的單一實驗改為移除原 4 維 handcrafted context，只保留 300×10 sequence |

### 3.11 ATR／波動率尺度 Label（使用者最新結果，2026-07-22）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED` |
| 程式基準 | 使用者已實測；本輪提供的 bundle 未保留該次完整程式 ZIP、SHA256 與精確報表 |
| 唯一主要變更 | 固定百分比 MFE／adverse barrier 改為事件日可觀測 ATR／波動率尺度；其他詳細倍數以當次實驗為準，本文件不自行補值 |
| Dataset／Label | 需要 relabel；是否重建 feature bank 由當次 Dataset policy 決定 |
| Selection／OOS 結果 | 使用者確認整體 OOS 沒有比固定百分比 Label 更好；精確指標未保留 |
| 判定 | ATR 尺度沒有解決目前整體 OOS 泛化問題；退回固定百分比 Label，不再列為近期方向 |
| 下一步 | 改測輸入與目標學習機制，不再調 Label 尺度 |

### 3.12 Architecture／Experiment Profile 管理規則（2026-07-22）

- `multiscale_cnn_sequence_only_v1` 已取得完整 OOS 結果並升為目前 accepted architecture 與正式研究基準。
- `multiscale_cnn_v1` 保留為 active historical comparator；其完整 OOS 證據仍有效，但新實驗預設固定從 sequence-only 基準延伸，除非實驗目的明確要求比較 context 有無。
- `multiscale_cnn_regime_context_v1`、`multiscale_cnn_v2～v8`、`tiny_cnn_v1`、`residual_tcn_v1` 均為 legacy read-only compatibility；保留程式碼只供舊 checkpoint／manifest 重建與歷史重現。
- optimizer、LR schedule、augmentation、loss weighting 等訓練差異只可新增 experiment profile，不可再建立 v10、v11 等假模型版本。
- `baseline`、`adamw_only`、`adam_warmup_cosine`、`history_masking_only` 與 `recent_decay_60m` 是訓練 profile；工件依 architecture／profile 子目錄隔離。`recent_decay_60m` 已被 8B 完整 OOS 淘汰，不得再作正式新實驗入口。
- 舊 v1 baseline 工件的無 profile 歷史路徑只提供唯讀 fallback；不得用它覆寫 manifest 或匯出正式 forward-OOS scores。新訓練與正式輸出一律寫入 `<architecture>/<experiment_profile>/`。

### 3.13 8A Sequence-only context ablation（2026-07-22）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `ACCEPTED`；升為目前完整 OOS 比較基準 |
| 程式基準 | 結果 ZIP `test-branch-1_20260722_231639_449a31e.zip`；SHA256 `afe427a2eeecc9565857f15362dfc1e82118f52d2eb92b65968c00e1f66aa569` |
| Architecture | `multiscale_cnn_sequence_only_v1`；experiment profile=`baseline` |
| 唯一模型變更 | 保留 v1 的 300×10 Level sequence、Short／Medium／Long branches、channels 16／16／16、kernel、pooling、dropout 與 32 維 head；head 第一層不再拼接 Dataset 的 4 維 event context |
| 移除輸入 | `high_len_norm`、`breakout_level_to_close`、`close_to_breakout_level`、`high_to_breakout_level` |
| 固定條件 | 固定百分比 Label、Adam、LR 0.0003、weight decay 0.0001、gradient clip 1.0、augmentation=`none`、seed 42、threshold 0.5、`selected_epochs`、class/time weight=`none` |
| Dataset／Label | 不重建、不 relabel；現有 context arrays 保留於 Dataset 以維持 storage contract，但模型 forward 明確忽略其數值 |
| Selection | 原始 PASS 54.81%、模型 PASS 76.03%、PASS Precision 62.07%、Lift +7.26 pp、Recall 86.11%、Accuracy 63.55%、Score 0.5864 |
| OOS | 原始 PASS 55.63%、模型 PASS 50.17%、PASS Precision 59.30%、Lift +3.66 pp、Recall 53.47%、Accuracy 53.70%、Score 0.4886 |
| 相較 v1 baseline | OOS Precision 約 +0.01 pp、Lift 持平、Recall +1.98 pp、模型 PASS +1.85 pp、Accuracy +0.36 pp、Score +0.0095；Precision gap 由 −2.85 pp 縮至 −2.77 pp，Score gap 由 −0.1129 縮至 −0.0978 |
| 判定 | Precision 未顯著提高，但在 Precision 不退步下，Recall、模型 PASS、Accuracy、Score 與兩項泛化落差一致改善；採用為較簡單的新基準，仍未解決整體 OOS drift |
| 下一步 | 8B 只測 recent-decay time weighting；其結果已取得並淘汰，詳見 3.14 |

### 3.14 8B Recent-decay time weighting（2026-07-22）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；使用者已退回 8A sequence-only／baseline |
| 程式基準 | 8B patch `breakout_quality_recent_decay_60m_patch_20260722.zip`；使用者結果完成後退回 ZIP `test-branch-1_20260722_231639_449a31e(1).zip`，SHA256 `afe427a2eeecc9565857f15362dfc1e82118f52d2eb92b65968c00e1f66aa569` |
| 唯一學習變更 | `multiscale_cnn_sequence_only_v1 / baseline` → 同架構 `recent_decay_60m`；每個 training phase 依 group age 使用 `0.5 ** (age_months / 60)`，再正規化為總 group weight 不變 |
| 固定條件 | 固定百分比 Label、Adam、LR 0.0003、weight decay 0.0001、gradient clip 1.0、augmentation=`none`、seed 42、threshold 0.5、`selected_epochs`、class weight=`none` |
| Dataset／Label | 不重建、不 relabel；Validation、Selection 報表與 OOS 評估維持未加權 |
| Selection | 原始 PASS 54.81%、模型 PASS 73.16%、PASS Precision 62.94%、Lift +8.14 pp、Recall 84.03%、Accuracy 64.13%、Score 0.5830 |
| OOS | 原始 PASS 55.63%、模型 PASS 47.04%、PASS Precision 58.64%、Lift +3.01 pp、Recall 49.59%、Accuracy 52.50%、Score 0.4627 |
| 相較 8A 基準 | OOS Precision −0.66 pp、Lift −0.65 pp、Recall −3.88 pp、模型 PASS −3.13 pp、Accuracy −1.20 pp、Score −0.0259；Precision gap 由 −2.77 pp 惡化為 −4.30 pp，Score gap 由 −0.0978 惡化為 −0.1203 |
| 判定 | Selection Precision／Lift 上升，但完整 OOS 所有主要指標及泛化落差均惡化；較近期 Selection weighting 加深對 Selection 末期型態的擬合，未提升跨 2021–2026 的泛化 |
| 下一步 | 維持 8A 基準，改測 8C same-day cross-sectional rank context；不再調 recent-decay 半衰期 |

### 3.15 8C Same-day cross-sectional rank context（2026-07-23）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；使用者已退回 8A sequence-only／baseline |
| 程式基準 | 8C patch `breakout_quality_cross_sectional_rank_v1_patch_20260723.zip`，SHA256 `85ec1dae421a289ec1152cfca53df8af4a3ba91575f3e61c186bc7f8aec2f0a4`；結果由使用者提供，完整結果工件未保留；退回 ZIP `test-branch-1_20260723_002452_4931f08(1).zip`，SHA256 `ee90f0712b77d2b1de6b2f97d83fa8c6b7b1550187e4907ef96385d4229d5f0c` |
| 唯一模型／輸入變更 | 在 8A sequence-only CNN head 加入由事件日及以前資料計算的 20／60／120 日個股報酬同日 percentile；同一 `ticker/date` 的多個 `high_len` 共用相同 rank context |
| 固定條件 | 固定百分比 Label、Adam、LR 0.0003、weight decay 0.0001、gradient clip 1.0、augmentation=`none`、seed 42、threshold 0.5、`selected_epochs`、class/time weight=`none` |
| Dataset／Label | 只新增 rank-context sidecar；300×10 feature bank 與 Label 不變，不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 73.24%、PASS Precision 62.97%、Lift +8.17 pp、Recall 84.16%、Accuracy 64.20%、Score 0.5774 |
| OOS | 原始 PASS 55.63%、模型 PASS 47.92%、PASS Precision 58.76%、Lift +3.13 pp、Recall 50.61%、Accuracy 52.76%、Score 0.4714 |
| 相較 8A 基準 | OOS Precision −0.54 pp、Lift −0.53 pp、Recall −2.86 pp、模型 PASS −2.25 pp、Accuracy −0.94 pp、Score −0.0172；Precision gap 由 −2.77 pp 惡化為 −4.22 pp，Score gap 由 −0.0978 惡化為 −0.1061 |
| 判定 | 同日 percentile 使 Selection Precision／Lift 上升，但完整 OOS Precision、Recall、模型 PASS、Accuracy、Score 與兩項泛化落差全部低於 8A；新增相對強弱 context 仍形成 Selection 捷徑，未提升跨時期泛化 |
| 下一步 | 維持 8A sequence-only／baseline，停止增加 context；下一個單一實驗改為 8D hybrid BCE + same-day pairwise ranking loss，直接改變學習目標 |

### 3.16 8D Hybrid BCE + same-day pairwise ranking loss（2026-07-23）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；使用者已退回 8A sequence-only／baseline |
| 程式基準 | 8D patch `breakout_quality_same_day_pairwise_rank_patch_20260723.zip`，SHA256 `1ffa0c5a028ed962ceb377e631272b152f363061e0c5809b9ae08b8c5e5c4a2b`；結果完成後退回 ZIP `test-branch-1_20260723_082925_cc41468(1).zip`，SHA256 `0a13a83aeb93572c9eda96999cf2079db920c0b78ecfcf2046e87c1189e32f68` |
| 唯一學習變更 | 8A binary cross-entropy 加上權重 0.20 的同日 unique `ticker/date` PASS／REJECT pairwise logistic ranking loss；模型輸入與架構不變 |
| 固定條件 | 固定百分比 Label、Adam、LR 0.0003、weight decay 0.0001、gradient clip 1.0、augmentation=`none`、seed 42、threshold 0.5、`selected_epochs`、class/time weight=`none` |
| Dataset／Label | 不重建、不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 71.31%、PASS Precision 62.37%、Lift +7.56 pp、Recall 81.15%、Accuracy 62.83%、Score 0.5713 |
| OOS | 原始 PASS 55.63%、模型 PASS 46.49%、PASS Precision 59.11%、Lift +3.48 pp、Recall 49.40%、Accuracy 52.84%、Score 0.4744 |
| 相較 8A 基準 | OOS Precision −0.19 pp、Lift −0.18 pp、Recall −4.07 pp、模型 PASS −3.68 pp、Accuracy −0.86 pp、Score −0.0142；Precision gap 由 −2.77 pp 惡化為 −3.25 pp，Score gap 只由 −0.0978 微幅縮至 −0.0969 |
| 判定 | Ranking loss 使模型更保守，Recall 與可用候選數下降，但 OOS Precision 未提高；微小 Score gap 改善不足以抵銷所有主要 OOS 指標退步 |
| 下一步 | 停止微調 ranking weight、margin 與 pair sampling；改測固定 multi-seed probability ensemble |

### 3.17 8E Fixed 8-seed probability ensemble（2026-07-23）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；使用者已退回 8A sequence-only／baseline |
| 程式基準 | 8E patch `breakout_quality_fixed_8seed_ensemble_parallel_patch_20260723.zip`，SHA256 `1cfac3a5260df77b35f0f322ce073fa25554a1534598d73f5389e06213c41843`；結果完成後退回 ZIP `test-branch-1_20260723_082925_cc41468(2).zip`，SHA256 `0a13a83aeb93572c9eda96999cf2079db920c0b78ecfcf2046e87c1189e32f68` |
| 唯一學習變更 | 固定訓練 seeds 42～49 的八個 8A sequence-only members，最終使用八個 PASS probabilities 的算術平均；不挑 seed、不投票、threshold 維持 0.5 |
| 執行方式 | 最多兩個 seed processes 平行；每個 member 使用獨立工件與固定訓練契約 |
| 固定條件 | 固定百分比 Label、Adam、LR 0.0003、weight decay 0.0001、gradient clip 1.0、augmentation=`none`、threshold 0.5、`selected_epochs`、class/time weight=`none` |
| Dataset／Label | 不重建、不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 70.92%、PASS Precision 63.29%、Lift +8.48 pp、Recall 81.89%、Accuracy 64.04%、Score 0.5564 |
| OOS | 原始 PASS 55.63%、模型 PASS 39.68%、PASS Precision 58.23%、Lift +2.60 pp、Recall 41.53%、Accuracy 50.90%、Score 0.4476 |
| 相較 8A 基準 | OOS Precision −1.07 pp、Lift −1.06 pp、Recall −11.94 pp、模型 PASS −10.49 pp、Accuracy −2.80 pp、Score −0.0410；Precision gap 由 −2.77 pp 惡化為 −5.06 pp，Score gap 由 −0.0978 惡化為 −0.1088 |
| 判定 | 八員平均降低了 seed 方差，卻無法修復所有 members 共有的 OOS score drift；固定 threshold 下模型顯著更保守，完整 OOS 全面低於 8A |
| 下一步 | 停止 ensemble seed 數、聚合方式與投票規則的細調；回到單模型，先修正 sequence-only 訓練仍以重複 event rows 作為 optimizer sampling unit 的問題 |

### 3.18 8F Unique ticker/date group training（2026-07-23）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `ACCEPTED`；目前正式研究基準 |
| 程式基準 | 來源 ZIP `test-branch-1_20260723_222424_33327b5.zip`，SHA256 `37afba30e6df53d3ae8bdfd06740cd754d396453a2238910ed9fb8de775b5c92`；交付 patch `breakout_quality_unique_group_sampling_patch_20260723.zip`，SHA256 `5c78c1fc78f5cac1fb3a56c050c568314f15eca048acbcefcdb4bb7a691ea8bc` |
| Architecture／Profile | `multiscale_cnn_sequence_only_v1 / unique_group_sampling`；model spec、參數量與輸入完全沿用 8A |
| 唯一訓練變更 | Inner Train 與 Final Refit 在 shuffle／batching 前，依 `ticker/date` 壓成一筆 deterministic representative；規則固定為最小原始 event row index |
| 固定條件 | Patience 1、固定百分比 Label、Adam、LR 0.0003、weight decay 0.0001、gradient clip 1.0、seed 42、threshold 0.5、`selected_epochs`、augmentation/class/time weight=`none` |
| Dataset／Label | 不重建 feature bank、不 relabel；Validation／Selection／OOS 繼續以完整 rows及 `1/group_size` 評估 |
| Training 壓縮 | Inner Train 538,887 rows → 16,832 groups；Final Refit 729,654 rows → 23,072 groups；Best Epoch 2；Final Refit 362 optimizer steps |
| Selection | 原始 PASS 54.81%、模型 PASS 88.90%、PASS Precision 56.90%、Lift +2.09 pp、Recall 92.29%、Accuracy 57.46%、Score 0.5784 |
| OOS | 原始 PASS 55.63%、模型 PASS 73.24%、PASS Precision 58.46%、Lift +2.83 pp、Recall 76.96%、Accuracy 56.76%、Score 0.5497 |
| 相較 8A | OOS Precision −0.84 pp、Lift −0.83 pp，但 Recall +23.49 pp、模型 PASS +23.07 pp、Accuracy +3.06 pp、Score +0.0611；Precision gap 由 −2.77 pp 改善為 +1.56 pp，Score gap 由 −0.0978 改善為 −0.0287 |
| 判定 | 移除同事件多個 `high_len` rows 對不同 Adam updates 的重複影響後，Selection 指標下降但完整 OOS 延續性大幅改善；依「整體 OOS 優先」採用為新研究基準 |
| 下一步 | 先測 8G：只把 early-stopping patience 1 改為 5，確認 unique-group training 是否只是訓練不足 |

### 3.19 8G Unique-group sampling + patience 5（2026-07-24）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；policy 已退回 8F patience 1 |
| 程式基準 | 使用者 ZIP `test-branch-1_20260724_002430_8f4f510.zip`，SHA256 `7c127ead794f0388386fad11e83485c66bb2735d5838783e302ffd257c1a0b87`；結果由使用者提供，ZIP 內 policy 為 patience 5，但內附 accepted 8F model manifest 仍為 patience 1，因此本輪以使用者貼出的報表作為 8G 結果來源並將程式設定退回 1 |
| 唯一訓練變更 | 8F 的 `BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE` 由 1 改為 5；architecture、sampling、Dataset、Label、optimizer、LR、batch size、threshold、seed 全部不變 |
| Selection | 原始 PASS 54.81%、模型 PASS 75.20%、PASS Precision 61.42%、Lift +6.61 pp、Recall 84.27%、Accuracy 62.37%、Score 0.5799 |
| OOS | 原始 PASS 55.63%、模型 PASS 42.12%、PASS Precision 57.55%、Lift +1.92 pp、Recall 43.58%、Accuracy 50.73%、Score 0.4789 |
| 相較 8F | OOS Precision −0.91 pp、Lift −0.91 pp、Recall −33.38 pp、模型 PASS −31.12 pp、Accuracy −6.03 pp、Score −0.0708；Precision gap 由 +1.56 pp 惡化為 −3.87 pp，Score gap 由 −0.0287 惡化為 −0.1009 |
| 判定 | Patience 增加讓 Selection Precision 顯著上升，但完整 OOS 恢復嚴重 score drift與過度保守；8F 不是單純訓練不足，Epoch 2 附近的早停本身就是重要 regularization |
| 下一步 | 不再增加 patience 或直接延長 unique-group epochs。下一個單一實驗規劃為 8H：維持 patience 1，只把 batch size 128 groups 改為 64 groups，增加每個 epoch 的 optimizer updates但不重複走訪同一 group |

### 3.20 8H Unique-group sampling + batch size 64（2026-07-24）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；policy 已退回 8F batch size 128 |
| 程式基準 | 使用者 ZIP `test-branch-1_20260724_005131_551b46d.zip`，SHA256 `082f2066f757d1fc8608c31d2fa4954484102433a3d356b3c36cda0e11b87ae3` |
| 唯一訓練變更 | 8F 的 batch size 由 128 unique groups 改為 64；patience 1、architecture、sampling、Dataset、Label、optimizer、LR、threshold、seed 與 final refit mode 全部不變 |
| Training | Best Epoch 仍為 2；Final Refit 每 epoch 361 steps，總計 722 optimizer steps，約為 8F 的 2 倍 |
| Selection | 原始 PASS 54.81%、模型 PASS 86.59%、PASS Precision 57.22%、Lift +2.41 pp、Recall 90.41%、Accuracy 57.70%、Score 0.5667 |
| OOS | 原始 PASS 55.63%、模型 PASS 73.90%、PASS Precision 58.12%、Lift +2.49 pp、Recall 77.20%、Accuracy 56.37%、Score 0.5440 |
| 相較 8F | OOS Precision −0.34 pp、Lift −0.34 pp、Recall +0.24 pp、模型 PASS +0.66 pp、Accuracy −0.39 pp、Score −0.0057；Precision gap 由 +1.56 pp 縮為 +0.90 pp，Accuracy gap 由 −0.70 pp 惡化為 −1.33 pp；Score gap由 −0.0287 微幅改善為 −0.0227 |
| 判定 | 將 optimizer updates 加倍沒有提高 Precision／Lift，Accuracy 與平均 Score 也下降；僅 Recall 與 Score gap 有極小改善，不足以取代 8F。停止 batch size 64，退回 128 |
| 下一步 | 不再做 32／64／96 等小 batch 細調。下一個單一實驗規劃為 8I：維持 8F batch 128 與 patience 1，只把 Final Refit 從 `selected_epochs` 改為 `matched_optimizer_steps`，隔離完整 Selection 比 Inner Train 多 37% optimizer updates 的 exposure mismatch |

### 3.21 8I Unique-group sampling + matched optimizer steps refit（2026-07-24）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；policy 已退回 8F `selected_epochs` |
| 程式基準 | 使用者 ZIP `test-branch-1_20260724_010852_7df842d.zip`，SHA256 `4486c7ce9721d07ccb73da2ae4cc2510110190a54cd8eabc0c596481fb1b8af4` |
| 唯一訓練變更 | 8F 的 Final Refit `selected_epochs` → `matched_optimizer_steps`；Inner Validation、Best Epoch 2、batch 128 groups、patience 1、模型、Dataset、Label、optimizer、LR、threshold與 seed 全部不變 |
| Final Refit exposure | Inner Train Best Epoch 2 = 264 optimizer steps；完整 Selection Final Refit 也固定為 264 steps，相當於 1.459 個 Selection epochs；8F `selected_epochs` 則為 362 steps／2 epochs |
| Selection | 原始 PASS 54.81%、模型 PASS 77.18%、PASS Precision 58.05%、Lift +3.24 pp、Recall 81.75%、Accuracy 57.62%、Score 0.5629 |
| OOS | 原始 PASS 55.63%、模型 PASS 54.93%、PASS Precision 59.11%、Lift +3.48 pp、Recall 58.36%、Accuracy 54.38%、Score 0.5278 |
| 相較 8F | OOS Precision／Lift各 +0.65 pp，但 Recall −18.60 pp、模型 PASS −18.31 pp、Accuracy −2.38 pp、Score −0.0219；Precision gap 由 +1.56 pp 縮為 +1.06 pp，Accuracy gap 由 −0.70 pp 惡化為 −3.24 pp，Score gap由 −0.0287 惡化為 −0.0351 |
| 補充比較 | 相較 8A，8I Precision僅低 0.19 pp，但 Recall +4.89 pp、Accuracy +0.68 pp、Score +0.0392；可視為較 8A 健康的高篩選歷史參考，但仍不符合目前「整體 OOS 優先」的正式基準條件 |
| 判定 | Matched steps 減少 Final Refit exposure 後確實提高 Precision，但代價是大幅降低 Recall、候選覆蓋、Accuracy 與 Score；不能取代 8F。停止 matched-steps 細調並退回 `selected_epochs` |
| 下一步 | 8J 只測「直接採用 Inner Validation 選出的 best checkpoint，不做重新初始化 Final Refit」，隔離問題是否來自 refit 本身，而非 exposure steps |


### 3.22 8J Direct best inner-validation checkpoint（2026-07-24）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；policy 已退回 8F `unique_group_sampling / selected_epochs` |
| 程式基準 | 使用者 ZIP `test-branch-1_20260724_011917_d1bc1cc.zip`，SHA256 `89183e6e5b3df7d8f2b345ce19417970fc7cf27cc065febd9d4f2586c160c713`，疊加 `breakout_quality_best_inner_checkpoint_8j_patch_20260724.zip` |
| 結果來源 | 使用者於 2026-07-24 提供 Selection／OOS 綜合判定表；本輪未提供含 8J model／manifest 的結果 ZIP，因此工件 SHA256 未知，結果數值依使用者報表記錄 |
| 唯一訓練變更 | 8F 的 Inner Validation 流程選出 Best Epoch 後，直接恢復並保存該 checkpoint；不重新初始化、不在完整 Selection 執行 Final Refit |
| 固定條件 | architecture、unique-group sampling、batch 128 groups、patience 1、Adam、LR 0.0003、weight decay 0.0001、gradient clip 1.0、seed 42、threshold 0.5、固定百分比 Label、augmentation/class/time weight=`none` 均不變 |
| Dataset／Label | 不重建 feature bank、不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 73.82%、PASS Precision 58.22%、Lift +3.41 pp、Recall 78.41%、Accuracy 57.32%、Score 0.5460 |
| OOS | 原始 PASS 55.63%、模型 PASS 61.69%、PASS Precision 58.50%、Lift +2.87 pp、Recall 64.87%、Accuracy 54.85%、Score 0.5226 |
| Selection→OOS | Precision +0.28 pp、Recall −13.54 pp、模型 PASS −12.13 pp、Accuracy −2.47 pp、Score −0.0234 |
| 相較 8F | OOS Precision／Lift 各僅 +0.04 pp；Recall −12.09 pp、模型 PASS −11.55 pp、Accuracy −1.91 pp、Score −0.0271；Precision gap由 +1.56 pp 縮為 +0.28 pp，Accuracy gap由 −0.70 pp 惡化為 −2.47 pp；只有 Score gap由 −0.0287 微幅改善為 −0.0234 |
| 判定 | 直接使用 best inner checkpoint 沒有提高整體 OOS。它以幾乎不變的 Precision 換取明顯較低 Recall、候選覆蓋、Accuracy 與 Score；不符合採用條件，正式基準維持 8F selected-epochs refit |
| 下一步 | 停止 Final Refit mode 微調；8K 改測 unique-group training 的**日期密度平衡**，降低突破群聚日期對 optimizer 的支配 |

### 3.23 8K Unique-group date-density balancing（2026-07-24）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；active policy 已退回 8F `unique_group_sampling / time_weight=none` |
| 程式基準 | 使用者 ZIP `test-branch-1_20260724_093559_485edba.zip`，SHA256 `8c73380d3a3803346a6c2c55c2fa2a9e2f32e2342a988cd76854c27c3f3103ef`；結果表由使用者提供的 `貼上的 Markdown (1).md` 記錄 |
| 正式比較口徑 | 依事前鎖定設定：threshold 0.5、patience 1、Best Epoch 2。ZIP 內現存 8K manifest 為後續 patience 5／threshold 0.55／Best Epoch 10 工件，不能代替正式 8K 單一變更結果 |
| 唯一訓練變更 | 8F unique-group sampling 保留全部 eligible groups；同一 training phase 內，同日每個 group raw weight=`1 / 當日 eligible unique group 數`，再正規化為平均 group weight 1；training loss 使用 `fixed_batch_size` denominator |
| 固定條件 | architecture=`multiscale_cnn_sequence_only_v1`、batch 128 groups、patience 1、selected-epochs refit、Adam、LR 0.0003、weight decay 0.0001、gradient clip 1.0、seed 42、threshold 0.5、固定百分比 Label、augmentation/class weight=`none` 均不變 |
| Dataset／Label | 不重建 feature bank、不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 99.21%、PASS Precision 54.93%、Lift +0.13 pp、Recall 99.44%、Accuracy 54.98%、Score 0.5768 |
| OOS | 原始 PASS 55.63%、模型 PASS 97.64%、PASS Precision 56.08%、Lift +0.45 pp、Recall 98.42%、Accuracy 56.24%、Score 0.5602 |
| Selection→OOS | Precision +1.15 pp、Recall −1.01 pp、模型 PASS −1.57 pp、Accuracy +1.26 pp、Score −0.0166 |
| 相較 8F | OOS Precision／Lift各 −2.38 pp，Recall +21.46 pp、模型 PASS +24.40 pp、Accuracy −0.52 pp、Score +0.0105；Precision gap由 +1.56 pp 降為 +1.15 pp，Accuracy gap改善 1.96 pp，Score gap改善 0.0121 |
| 判定 | 日期完全等權使模型在 threshold 0.5 幾乎全部判 PASS，OOS 模型 PASS 97.64%，Precision Lift只剩 +0.45 pp；雖 score drift縮小，但已失去實際篩選能力，且 Precision／Accuracy／Score三項採用指標只有 Score改善，不符合事前採用條件 |
| Threshold 0.55 診斷 | 同一 patience 1 模型把門檻改為 0.55後，OOS Precision 60.80%、Lift +5.17 pp、Recall 58.69%、模型 PASS 53.71%、Accuracy 55.97%；與 8F 同樣使用 0.55 時，8K 的 Precision、Recall、Accuracy 均較高，因此保留為固定 0.55 的 forward-validation 候選；但目前 OOS 已參與門檻判讀，不再是未污染 final OOS |
| 過度訓練診斷 | Patience 5／Best Epoch 10／threshold 0.55：OOS Precision 61.43%、Recall 34.27%、Accuracy 51.46%、Score 0.5144；固定 200 epochs／threshold 0.5：OOS Precision 56.85%、Recall 28.55%、Accuracy 48.20%、Score 0.3604。兩者均再次證明延長訓練造成嚴重 OOS 過擬合 |
| 下一步 | 不再使用完整日期等權或依本次 OOS調 threshold。8L 規劃改測 date-diverse batch scheduling：保留 8F 的無日期權重目標與 threshold 0.5，只改 batch 內日期組成，降低同日相關樣本集中於同一 optimizer update |


### 3.24 8L Date-diverse batch scheduling（2026-07-24）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；使用者已退回 8F random batch scheduling |
| 程式基準 | 使用者 ZIP `test-branch-1_20260724_100522_938ba56(1).zip`，SHA256 `ce70a6db63f83d189664b5b0ed01df4d0fe85d9bacc7b27f6b7c031e63d5a21c`，疊加 8L date-diverse batch patch；結果數值由使用者報表提供 |
| 唯一變更 | 每個 epoch 的 unique groups、sample weights、batch size 與 optimizer steps均不變，只以 date-interleaved順序讓同一 batch 優先包含不同交易日 |
| Dataset／Label | 不重建 feature bank、不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 86.86%、PASS Precision 57.28%、Lift +2.47 pp、Recall 90.78%、Accuracy 57.84%、Score 0.5655 |
| OOS | 原始 PASS 55.63%、模型 PASS 71.24%、PASS Precision 58.41%、Lift +2.78 pp、Recall 74.80%、Accuracy 56.35%、Score 0.5409 |
| Selection→OOS | Precision +1.13 pp、Recall −15.98 pp、模型 PASS −15.62 pp、Accuracy −1.49 pp、Score −0.0246 |
| 相較 8F | OOS Precision／Lift各 −0.05 pp，Recall −2.16 pp、模型 PASS −2.00 pp、Accuracy −0.41 pp、Score −0.0088；只有 Score gap改善 0.0041 |
| 判定 | 主要絕對指標均未超越 8F，同日樣本集中於 batch並非主要 OOS瓶頸；停止 date／ticker interleaved scheduling細調 |

### 3.25 8M Long Branch last-state pooling（2026-07-24）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；使用者已退回 8F architecture |
| 程式基準 | 使用者 ZIP `test-branch-1_20260724_100522_938ba56(2).zip`，SHA256 `ce70a6db63f83d189664b5b0ed01df4d0fe85d9bacc7b27f6b7c031e63d5a21c`，疊加 `breakout_quality_long_branch_last_pooling_8m_patch_20260724.zip`；結果數值由使用者報表提供 |
| 唯一變更 | Long branch summary由 120-bar average＋300-bar average改為 last state＋120-bar average；參數量、head寬度與其他 branches不變 |
| Dataset／Label | 不重建 feature bank、不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 84.65%、PASS Precision 57.75%、Lift +2.94 pp、Recall 89.20%、Accuracy 58.31%、Score 0.5728 |
| OOS | 原始 PASS 55.63%、模型 PASS 66.10%、PASS Precision 57.58%、Lift +1.94 pp、Recall 68.40%、Accuracy 54.38%、Score 0.5370 |
| Selection→OOS | Precision −0.17 pp、Recall −20.79 pp、模型 PASS −18.56 pp、Accuracy −3.93 pp、Score −0.0358 |
| 相較 8F | OOS Precision −0.88 pp、Lift −0.89 pp、Recall −8.56 pp、模型 PASS −7.14 pp、Accuracy −2.38 pp、Score −0.0127；Precision、Accuracy、Score gaps均惡化 |
| 判定 | 移除 Long 300-bar average並未改善泛化，反而降低所有主要 OOS指標；Long branch last-state pooling路線停止 |

### 3.26 8O Zero-initialized gated temporal pooling（2026-07-24）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；active policy 已退回 8F `multiscale_cnn_sequence_only_v1 / unique_group_sampling` |
| 程式基準 | 8O 結果 ZIP `test-branch-1_20260724_135343_4cead4c.zip`，SHA256 `a03340f1ed2abf56e4bb20dbbec378a649a19477d0f14f7c2f0d71bf2ac928b9`；使用者已退回 8F policy |
| Architecture | `multiscale_cnn_sequence_only_gated_pool_v1`；目前退回 ZIP 未保留該架構程式，僅由本紀錄保存歷史規格與結果；正式入口為 `multiscale_cnn_sequence_only_v1` |
| 唯一變更 | 每個 branch新增一個零初始化 1×1 scalar temporal gate；20／60／120／300-bar fixed mean pooling改為同一 branch gate的 softmax weighted pooling，last-state summaries不變 |
| Dataset／Label | 不重建 feature bank、不 relabel |
| Selection | 原始 PASS 54.81%、模型 PASS 88.90%、PASS Precision 56.73%、Lift +1.92 pp、Recall 92.02%、Accuracy 57.16%、Score 0.5844 |
| OOS | 原始 PASS 55.63%、模型 PASS 71.24%、PASS Precision 58.33%、Lift +2.70 pp、Recall 74.70%、Accuracy 56.24%、Score 0.5508 |
| Selection→OOS | Precision +1.61 pp、Recall −17.32 pp、模型 PASS −17.66 pp、Accuracy −0.91 pp、Score −0.0336 |
| 相較 8F | OOS Precision／Lift各 −0.13 pp、Recall −2.26 pp、模型 PASS −2.00 pp、Accuracy −0.52 pp、Score +0.0011；Precision gap改善 0.05 pp，但 Recall、Accuracy與 Score gaps分別惡化 1.99、0.21、0.0049 |
| 判定 | 只有平均 Score極小幅增加，Precision、Recall、Accuracy與主要泛化 gaps均未形成 Pareto improvement；learnable pooling未突破現有 fixed pooling前緣，不再細調 gate hidden width、temperature或初始化 |
| 下一步 | 進入現有 multiscale CNN 最後一項 8P raw＋window-normalized dual-path；若仍無固定 coverage排序改善，停止本模型家族微調並轉向 InceptionTime／ModernTCN／自監督 encoder |

### 3.27 8P Raw＋window-normalized dual-path（2026-07-24）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；未形成 Pareto improvement，已退回 8F；dual-path architecture 轉為 legacy read-only |
| 程式基準 | `test-branch-1_20260724_140019_f3aa36d.zip`，SHA256 `556dea38770f6f3cae5e11fd9136d3d7a0a2cfc538606e4d944176fffc98a7f3` |
| Architecture | `multiscale_cnn_sequence_only_dual_path_v1`；與 8F 使用相同 `unique_group_sampling` profile，工件依 architecture 隔離 |
| 唯一變更 | 保留原 10 維 raw／level 三分支 CNN，新增同一 300-bar 視窗逐 sample／逐 channel z-score 的第二組三分支 CNN；每個 branch 的 raw／normalized summaries 經獨立 linear fusion後再送入原 32 維 head |
| Normalization | 只使用該事件已知的 300 bars；`mean` 與 population standard deviation 逐 sample／channel計算，epsilon=`1e-5`，常數 channel輸出全零，不讀取事件日之後資料 |
| 受控初始化 | Raw branches與 head依相同 seed保持 8F初始權重；三個 fusion初始化為左側 identity、normalized側全零及 bias全零，因此 eval-mode初始 logits精確等同8F；第一個 backward normalized fusion即可收到梯度，融合開啟後 normalized branches可學習 |
| Parameters | 8F 23,394 → 8P 49,858；增加來自第二組三分支 CNN與三個 branch-level fusion，receptive field仍為244 bars，head input/output寬度不變 |
| 固定條件 | unique `ticker/date` sampling、batch 128、patience 1、`selected_epochs`、Adam、LR 0.0003、weight decay 0.0001、threshold 0.5、seed 42、class/time weight=`none` |
| Dataset／Label | 不重建 feature bank、不新增 sidecar、不 relabel；normalized path於 model forward由既有 feature-bank batch即時計算 |
| Selection | Precision 57.32%、Lift +2.52 pp、Recall 89.65%、模型 PASS 85.71%、Accuracy 57.75%、Score 0.5770 |
| OOS | Precision 58.52%、Lift +2.89 pp、Recall 68.48%、模型 PASS 65.10%、Accuracy 55.46%、Score 0.5354 |
| 與 8F 差異 | OOS Precision +0.06 pp、Lift +0.06 pp，但 Recall −8.48 pp、模型 PASS −8.14 pp、Accuracy −1.30 pp、Score −0.0143；Precision gap由 +1.56 pp縮為 +1.20 pp，Recall／Accuracy／Score gaps均惡化 |
| 判定 | Precision增幅僅0.06 pp，不足以抵銷Recall、Accuracy與Score的明顯下降；較大容量與normalized path未提高整體可分性 |
| 下一步 | 停止現有 multiscale CNN家族微調；進入9A單一InceptionTime，沿用8F unique-group sampling與完整評估契約，不以ensemble或threshold調整混入首輪比較 |


### 3.28 9A 單一 InceptionTime（2026-07-24）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `ACCEPTED`；升為新的排序／高品質模型基準；8F保留為高覆蓋基準 |
| 程式基準 | 結果 ZIP `test-branch-1_20260724_173238_f8387e3.zip`，SHA256 `34c8583d4ecfe3d9d79264170832bfb08764fbeb187b78ebeb8d5cb3efac7624`；結果文字 SHA256 `e78623c1791fdcf647e604ff6f8ebe5277190551d2349115235975f4b81734ab` |
| Architecture | `inception_time_v1`；depth 6、filters 32、bottleneck 32、kernels 39／19／9、每3 modules residual、global average pooling、473,218 parameters |
| 固定條件 | unique-group sampling、batch 128、patience 1、selected-epochs、Adam、LR 0.0003、threshold 0.5、seed 42、class/time weight=`none`、CUDA BF16、deterministic algorithms、TF32 off |
| Dataset／Label | 沿用300×10 feature bank與固定百分比Label；不重建、不 relabel |
| Selection | 模型 PASS 64.02%、Precision 60.95%、Lift +6.14 pp、Recall 71.19%、Accuracy 59.21%、Score 0.5184、PR-AUC 0.6443 |
| OOS | 模型 PASS 45.43%、Precision 62.99%、Lift +7.35 pp、Recall 51.44%、Accuracy 56.17%、Score 0.4842、PR-AUC 0.6257 |
| 固定 coverage | OOS P@50／60／70% coverage = 62.77／61.62／60.25%；Selection→OOS 分別 +0.07／+0.28／+0.25 pp，排序泛化穩定 |
| 校準 | Score gap −0.0342；Brier 0.2394→0.2453，ECE 0.0308→0.0722。主要剩餘問題是 OOS calibration／coverage drift，而非排序能力失效 |
| 相較 8F | OOS Precision +4.53 pp、Lift +4.52 pp、Specificity +30.67 pp；Recall −25.52 pp、模型 PASS −27.81 pp、Accuracy −0.59 pp、Score −0.0655。近似相同 coverage 下，9A P@70%=60.25% 高於 8F 在73.24% coverage的58.46%，顯示真正排序改善 |
| 判定 | 接受為新的高品質／排序模型基準；不依OOS回調threshold。8F保留供高覆蓋用途與策略層比較 |
| 下一步 | 先做 `9A-GN`：只將InceptionTime的BatchNorm改為GroupNorm，測試能否改善ECE、Brier、Score／coverage drift並保留固定coverage Precision；若無改善，再進入9B ModernTCN |

### 3.29 9A-GN InceptionTime GroupNorm ablation（2026-07-25）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；policy已退回9A-BN `inception_time_v1`，9A-GN轉為legacy read-only |
| 程式基準 | 來源 ZIP `test-branch-1_20260724_204125_8561814.zip`，SHA256 `23ba431eaf7c914bc2de5465f72bd74054bfb71ffa95fc9c8c0ea1983149c58c`；套用 `breakout_quality_inceptiontime_groupnorm_9a_gn_patch_20260724.zip` 後執行；本輪未提供post-run ZIP，結果以使用者提供的完整報表數值為準 |
| Architecture | `inception_time_group_norm_v1`；只將9A的8個`BatchNorm1d(128)`改為`GroupNorm(8,128)`，其他架構、參數量與訓練條件不變 |
| Dataset／Label | 沿用300×10 feature bank、相同outer split與固定百分比Label；未重建、未relabel |
| Selection | 模型PASS 86.35%、Precision 57.58%、Lift +2.77 pp、Recall 90.72%、Accuracy 58.28%、Score 0.6029、PR-AUC 0.6394、P@50／60／70%=62.59／61.33／59.98%、Brier 0.2414、ECE 0.0549 |
| OOS | 模型PASS 80.39%、Precision 57.08%、Lift +1.45 pp、Recall 82.48%、Accuracy 55.75%、Score 0.5757、PR-AUC 0.5650、P@50／60／70%=59.46／58.60／57.89%、R@P60%=0.50%、Brier 0.2497、ECE 0.0356 |
| Selection→OOS | PR-AUC −0.0744；P@50／60／70%分別 −3.13／−2.73／−2.09 pp；Precision −0.50 pp、Recall −8.24 pp、模型PASS −5.96 pp、Accuracy −2.53 pp、Score −0.0272；Brier惡化0.0083，ECE反而改善0.0193 |
| 相較9A-BN | OOS PR-AUC −0.0607；P@50／60／70%分別 −3.31／−3.02／−2.36 pp；R@P60% −76.77 pp；Precision −5.91 pp、Accuracy −0.42 pp、Brier惡化0.0044。Recall +31.04 pp與模型PASS +34.96 pp主要來自score整體上移與coverage放寬，不是排序能力提升；ECE改善0.0366、Score gap改善0.0070不足以抵銷排序崩落 |
| 指標檢查 | `R@P60%=0.50%`並非報表計算錯誤：其定義是在所有precision≥60%的tie-safe cut中取最大recall；9A-GN在50% coverage時Precision已只有59.46%，因此只有極小的最前端區段能維持60% Precision |
| 判定 | GroupNorm改善了ECE與平均Score drift，但顯著破壞跨時期排序；未達任何固定coverage採用門檻，不再搜尋GroupNorm group數或其他normalization變體 |
| 下一步 | 進入9B ModernTCN；9A-BN維持排序／高品質正式基準，8F維持高覆蓋基準 |



### 3.30 9B ModernTCN（2026-07-25）

| 項目 | 內容 |
|---|---|
| 狀態 | `REJECTED`；policy已退回9A `inception_time_v1`，9B `modern_tcn_v1`轉為legacy read-only |
| 程式基準 | 實作來源`test-branch-1_20260725_085352_917ceb0.zip`，SHA256 `3f2ef72e52614bc82cc935c6fb6f833b390068fca40ec1a44bafbd692c16bbaa`；post-run ZIP `test-branch-1_20260725_093450_7b92818(1).zip`，SHA256 `97967b2be899c5be1312bb7f600ffdbcfce439a53a32fef2aee978f4d2c45aab` |
| Architecture | `modern_tcn_v1`；6個kernel-51 large-kernel depthwise residual blocks、96 channels、4× pointwise expansion、BatchNorm、global-average pooling |
| 唯一變更 | 以容量近似的ModernTCN取代9A InceptionTime；資料、Label、split、sampling、optimizer與GPU execution均不變 |
| 固定條件 | 300×10 raw-level sequence、Dataset context disabled、unique-group sampling、batch 128、patience 1、selected_epochs、Adam、LR 0.0003、weight decay 0.0001、threshold 0.5、seed 42、CUDA BF16、deterministic、TF32 off |
| 模型規模 | 475,394 trainable parameters；9A為473,218；receptive field 301 bars |
| Dataset／Label | 沿用既有feature bank與固定百分比Label；未重建Dataset、未relabel |
| Selection | 模型PASS 63.96%、Precision 62.33%、Lift +7.52 pp、Recall 72.74%、Accuracy 60.97%、Score 0.5347、PR-AUC 0.6671、P@50／60／70%=64.05／62.95／61.32%、R@P60%=84.36%、Brier 0.2333、ECE 0.0202 |
| OOS | 模型PASS 38.93%、Precision 57.94%、Lift +2.31 pp、Recall 40.55%、Accuracy 50.55%、Score 0.4505、PR-AUC 0.5799、P@50／60／70%=58.17／57.91／57.87%、R@P60%=15.06%、Brier 0.2664、ECE 0.1211 |
| Selection→OOS | PR-AUC −0.0872；P@50／60／70%分別 −5.88／−5.04／−3.45 pp；R@P60% −69.30 pp；Precision −4.39 pp、Recall −32.19 pp、模型PASS −25.03 pp、Accuracy −10.42 pp、Score −0.0842；Brier惡化0.0331、ECE惡化0.1009 |
| 相較9A | OOS PR-AUC −0.0458；P@50／60／70%分別 −4.60／−3.71／−2.38 pp；R@P60% −62.21 pp；Precision −5.05 pp、Recall −10.89 pp、模型PASS −6.50 pp、Accuracy −5.62 pp、Score −0.0337；Brier惡化0.0211、ECE惡化0.0489 |
| 判定 | Selection全面優於9A但OOS排序、分類與校準全面崩落，屬嚴重跨時期過擬合；不是threshold或單純score calibration問題。停止supervised CNN／TCN架構橫向搜尋，不再微調ModernTCN depth、kernel、channels或dropout |
| 下一步 | 9C Selection-only TS2Vec自監督預訓練；OOS未標記資料不得參與pretraining，先以frozen linear probe檢查representation是否跨時期穩定 |
| 退回驗證 | `test-branch-1_20260725_100932_a492e13.zip`的policy、active architecture與文件已正確退回9A；但ZIP遺漏`modern_tcn.py`，且factory／InceptionTime implementation無法重建`modern_tcn_v1`與`inception_time_group_norm_v1` legacy工件，與本紀錄及runtime契約不一致。本輪恢復兩個legacy架構的唯讀strict reconstruction、ModernTCN manifest execution驗證、歷史報表顯示與formal impacted-module registry；active policy、9A權重、Dataset、Label、split與研究結果均未改變 |

### 3.31 9C TS2Vec Selection-only自監督預訓練（2026-07-25）

| 項目 | 結果 |
|---|---|
| 狀態 | `REJECTED`；active policy退回9A `inception_time_v1`，9C `ts2vec_frozen_linear_v1`轉為legacy read-only；不啟動9C2 fine-tuning |
| 程式基準 | 結果 ZIP `test-branch-1_20260725_175013_818dfa5.zip`，SHA256 `c7b5b7bb8b6ccf3073a0a64944ba74969d7a70f528d3c26ce6ce48c8ac753ee1`；結果文字 SHA256 `a5f5d0f3cc6d2c41aaeac9272e874a837003c32142aa7cf76b33c19ea4758339` |
| Architecture | `ts2vec_frozen_linear_v1`；8層dilated residual encoder、hidden 128、representation 320、global-max pooling、320→2 linear head；total 831,810、frozen 831,168、trainable 642 parameters |
| 唯一研究變更 | 以Selection-only TS2Vec-style hierarchical contrastive pretraining取代純supervised representation；下游encoder完全凍結，只訓練linear head。資料、Label、split、unique-group sampling與評估契約不變 |
| Pretraining | 194,173個Selection rolling windows，2011-01-01～2020-12-31，stride 5；10 epochs、batch 128、AdamW、BF16 CUDA；loss由1.828744降至0.068249，未使用OOS windows或PASS／REJECT labels |
| 下游訓練 | Best epoch 2，最低Validation loss 0.689281；完整Selection重訓2 epochs，final loss 0.674765。loss接近隨機二元分類的0.693，顯示frozen representation對目前Label只有弱線性可分性 |
| Dataset／Label | 新增獨立pretraining dataset與encoder工件；既有supervised feature bank、event rows、labels與future-path cache未重建、未relabel |
| Selection | 原始PASS 54.81%、模型PASS 81.77%、Precision 57.88%、Lift +3.07 pp、Recall 86.35%、Accuracy 58.07%、Score 0.5710、PR-AUC 0.6225、P@50／60／70%=61.55／60.40／59.29%、R@P60%=70.19%、Brier 0.2409、ECE 0.0230 |
| OOS | 原始PASS 55.63%、模型PASS 88.78%、Precision 56.11%、Lift +0.48 pp、Recall 89.53%、Accuracy 55.21%、Score 0.6149、PR-AUC 0.5653、P@50／60／70%=58.16／57.85／57.33%、R@P60%=0.10%、Brier 0.2541、ECE 0.0746 |
| Selection→OOS | PR-AUC −0.0572；P@50／60／70%分別 −3.39／−2.55／−1.96 pp；R@P60% −70.09 pp；Precision −1.77 pp、模型PASS +7.01 pp、Score +0.0439；Brier惡化0.0132、ECE惡化0.0516 |
| 相較9A | OOS PR-AUC −0.0604；P@50／60／70%分別 −4.61／−3.77／−2.92 pp；R@P60% −77.17 pp；Precision −6.88 pp、Lift −6.87 pp、Accuracy −0.96 pp；Recall +38.09 pp與模型PASS +43.35 pp來自score整體上移及幾乎全面放行，不是排序改善；Brier惡化0.0088、ECE惡化0.0024 |
| 指標判讀 | `R@P60%=0.10%`與P@50%=58.16%一致：模型在50% coverage時已無法維持60% Precision，只有極小最前端tie-safe區段達標；不是靠threshold 0.5即可修正的問題 |
| 判定 | Frozen probe未接近9A，且Selection→OOS排序與校準明顯崩落；不符合預先設定的9C2啟動條件。停止TS2Vec epochs、stride、pooling、MLP head、threshold與fine-tuning細調，避免利用同一OOS救援失敗表示 |
| 下一步 | 9D MantisV2 frozen encoder＋linear probe；仍固定Label、split、unique-group sampling與評估口徑。先測外部預訓練表示，不同輪混入fine-tuning、ensemble或threshold調整 |

### 3.32 9D MantisV2 frozen encoder＋linear probe（2026-07-26）

| 項目 | 內容 |
|---|---|
| 狀態 | `REJECTED`；未達9A frozen-probe比較門檻，policy退回9A，MantisV2轉為legacy read-only；不啟動encoder fine-tuning |
| 程式基準 | 結果ZIP `test-branch-1_20260726_012004_f22e85c.zip`，SHA256 `27daea835892c0637927ef3f98e252a9ec5a39da418a7f7b97fadfc9c7b96f70`；結果文字SHA256 `7a63a463136400bd0419acdc500e3d35bc2d90d681a2c1c2b7d0aeed2a7819cd` |
| Architecture | `mantis_v2_frozen_linear_v1`；官方MantisV2 encoder、輸入長度512、32 patches、取Transformer第3層（index 2）的CLS＋mean combined 512維embedding；10個feature channels獨立編碼後串接，單一5120→2 linear head；frozen 2,214,144、trainable 10,242 parameters |
| 唯一研究變更 | 以釘死外部預訓練MantisV2 frozen encoder取代9A supervised encoder與9C自家TS2Vec encoder；encoder完全凍結且維持eval，只訓練linear head |
| 外部來源契約 | repository=`paris-noah/MantisV2`；revision=`99fe0f548960e272fbfa4b82fd9b5b5956779dfd`；checkpoint／config驗證SHA256；套件=`mantis-tsfm==1.0.0`；未使用project pretraining、OOS windows或PASS／REJECT labels訓練encoder |
| 下游訓練 | Best epoch 1，最低Validation loss 0.678870；完整Selection重訓1 epoch，final loss 0.675498 |
| Dataset／Label | 沿用既有300×10 supervised feature bank、固定百分比Label、Selection／OOS split與unique-group sampling；不重建、不relabel |
| Selection | 原始PASS 54.81%、模型PASS 87.48%、Precision 57.53%、Lift +2.72 pp、Recall 91.82%、Accuracy 58.36%、Score 0.6033、PR-AUC 0.6348、P@50／60／70%=62.48／61.44／60.11%、R@P60%=77.51%、Brier 0.2412、ECE 0.0553 |
| OOS | 原始PASS 55.63%、模型PASS 86.11%、Precision 58.13%、Lift +2.50 pp、Recall 89.98%、Accuracy 58.37%、Score 0.6043、PR-AUC 0.5933、P@50／60／70%=60.05／59.39／58.70%、R@P60%=55.15%、Brier 0.2460、ECE 0.0480 |
| Selection→OOS | PR-AUC −0.0415；P@50／60／70%分別 −2.43／−2.05／−1.41 pp；R@P60% −22.36 pp；threshold 0.5 Precision +0.60 pp、Recall −1.84 pp、模型PASS −1.37 pp、Accuracy +0.01 pp、Score +0.0010；Brier惡化0.0048、ECE改善0.0073 |
| 相較9A | OOS PR-AUC −0.0324；P@50／60／70%分別 −2.72／−2.23／−1.55 pp；R@P60% −22.12 pp；threshold 0.5 Precision −4.86 pp、Recall +38.54 pp、模型PASS +40.68 pp、Accuracy +2.20 pp、Score +0.1201；Brier惡化0.0007、ECE改善0.0242 |
| 相較8F | OOS Precision −0.33 pp，但Recall +13.02 pp、模型PASS +12.87 pp、Accuracy +1.61 pp、Score +0.0546；9D較像更寬鬆的高coverage模型，未形成比8F更高Precision或比9A更強排序的明確新定位 |
| 判定 | 9D明顯優於9C且threshold 0.5的coverage／score跨期穩定，但主要預設判定是固定coverage排序；五項排序指標全部低於9A，R@P60%落後22.12 pp，因此不符合controlled fine-tuning啟動條件。停止9D threshold、adapter、head、channel aggregation、output layer/token與encoder fine-tuning細調，避免用同一OOS救援 |
| 下一步 | 進入9E MOMENT frozen encoder＋linear probe；仍固定Label、split、unique-group sampling與評估口徑，不同輪混入fine-tuning、ensemble或threshold調整 |


### 3.33 9E MOMENT-1-base frozen encoder＋linear probe（2026-07-26）

| 項目 | 內容 |
|---|---|
| 狀態 | `REJECTED`；未達9A frozen-probe比較門檻，policy退回9A，MOMENT轉為legacy read-only；不啟動encoder fine-tuning |
| 程式基準 | 結果ZIP `test-branch-1_20260726_083239_00c2f57.zip`，SHA256 `1a27923eb85ebc014b9616fba4962428b248c162754cf1a90f2af9cb33fd424e`；結果文字SHA256 `fc874856bef08998ee37ca02bb4902db2df465fb847cc42bbd239b8a38638122` |
| Architecture | `moment_1_base_frozen_linear_v1`；官方MOMENT-1-base encoder、300→512 bars、patch／stride 8、12層T5-base、每channel對64 patches取mean後串接為7680維，單一7680→2 linear head；frozen 109,635,456、trainable 15,362 parameters |
| 唯一研究變更 | 以釘死外部預訓練MOMENT frozen encoder取代9A supervised encoder與9D MantisV2 encoder；patch embedder與encoder完全凍結、固定eval，只訓練linear head |
| 外部來源契約 | repository=`AutonLab/MOMENT-1-base`；revision=`b0ae5751d8ef43d72ad48fb5128e2ddc93c94b53`；checkpoint SHA256=`1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825`；runtime=`momentfm==0.1.4 / transformers==5.5.0`；未使用project pretraining、OOS windows或PASS／REJECT labels訓練encoder |
| 下游訓練 | Best epoch 2，最低Validation loss 0.684556；完整Selection重訓2 epochs，final loss 0.676872；train耗時4:04:24.7、score export耗時1:56:42.4、總耗時6:01:21.5 |
| Dataset／Label | 沿用既有300×10 supervised feature bank、固定百分比Label、Selection／OOS split與unique-group sampling；不重建、不relabel |
| Selection | 原始PASS 54.81%、模型PASS 70.18%、Precision 59.46%、Lift +4.65 pp、Recall 76.14%、Accuracy 58.47%、Score 0.5258、PR-AUC 0.6301、P@50／60／70%=61.58／60.58／59.49%、R@P60%=71.87%、Brier 0.2419、ECE 0.0379 |
| OOS | 原始PASS 55.63%、模型PASS 75.31%、Precision 56.40%、Lift +0.77 pp、Recall 76.35%、Accuracy 54.01%、Score 0.5378、PR-AUC 0.5857、P@50／60／70%=58.04／57.50／56.77%、R@P60%=23.07%、Brier 0.2476、ECE 0.0271 |
| Selection→OOS | PR-AUC −0.0444；P@50／60／70%分別 −3.54／−3.08／−2.72 pp；R@P60% −48.80 pp；threshold 0.5 Precision −3.06 pp、Recall +0.21 pp、模型PASS +5.13 pp、Accuracy −4.46 pp、Score +0.0120；Brier惡化0.0057、ECE改善0.0108 |
| 相較9A | OOS PR-AUC −0.0400；P@50／60／70%分別 −4.73／−4.12／−3.48 pp；R@P60% −54.20 pp；threshold 0.5 Precision −6.59 pp、Lift −6.58 pp、Recall +24.91 pp、模型PASS +29.88 pp、Accuracy −2.16 pp、Score +0.0536；Brier惡化0.0023、ECE改善0.0451 |
| 相較8F | OOS Precision／Lift各 −2.06 pp、Recall −0.61 pp、模型PASS +2.07 pp、Accuracy −2.75 pp、Score −0.0119；連高覆蓋基準也未超越 |
| 相較9D | OOS PR-AUC −0.0076；P@50／60／70%分別 −2.01／−1.89／−1.93 pp；R@P60% −32.08 pp；Precision −1.73 pp、Recall −13.63 pp、Accuracy −4.36 pp、Score −0.0665；只有ECE改善0.0209，且計算成本大幅增加 |
| 判定 | 9E的低ECE只代表分數尺度較穩，無法抵消OOS ranking全面退步；固定coverage排序與R@P60%皆遠低於9A，Selection→OOS drift也明顯。Frozen representation不具足夠跨期線性可分性，不啟動MOMENT fine-tuning，停止threshold、output reduction、patch／stride、pooling、MLP head、adapter與同checkpoint細調 |
| 下一步 | 9F小型Patch Transformer已實作；執行本地formal suite與完整Selection／OOS。仍固定Label、split、unique-group sampling、optimizer與評估口徑，不混入self-supervised pretraining、ensemble或threshold調整 |

---

### 3.34 9F：小型 Patch Transformer supervised architecture（2026-07-26）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED`；未接近9A排序基準，policy退回9A，`patch_transformer_v1`轉為legacy read-only；不啟動9F-B masked pretraining |
| 程式基準 | 結果 ZIP `test-branch-1_20260726_092733_cdea14d.zip`，SHA256 `2da379dce8536d83b165d499a48b6e7d38d588bfc6755bcc416e3ff9431da8a6`；結果文字SHA256 `f1e0661077ab38c07f3ae70257e60a2699af46d711c68c69c622e52bd4cebcba` |
| Architecture | `patch_transformer_v1`；300×10 sequence以10 bars為一個非重疊patch，30個patch tokens；128維embedding、3層Transformer、4 heads、MLP 256、sinusoidal position、patch mean pooling，共411,138個可訓練參數 |
| 唯一研究變更 | 以專案內從零監督式訓練的Patch Transformer取代9A InceptionTime；沒有外部預訓練、masked pretraining、Dataset context、ensemble或threshold調整 |
| 固定條件 | 固定百分比Label、Selection／OOS split、`unique_group_sampling`、Adam、LR 0.0003、batch 128、patience 1、`selected_epochs`、threshold 0.5、seed 42及全部ranking／calibration口徑不變 |
| Dataset／Label | 沿用既有300×10 supervised feature bank與labels；沒有重建或relabel；OOS未參與訓練 |
| Epoch | Best Epoch 1；最低Validation Loss 0.684602；完整Selection重訓1 epoch，Final Loss 0.684105 |
| Selection | 原始PASS 54.81%、模型PASS 98.13%、Precision 55.18%、Lift +0.38 pp、Recall 98.81%、Accuracy 55.37%、Score 0.6006；PR-AUC 0.6119、P@50／60／70%=60.88／59.81／59.18%、R@P60%=63.86%、Brier 0.2455、ECE 0.0525 |
| OOS | 原始PASS 55.63%、模型PASS 82.62%、Precision 57.73%、Lift +2.10 pp、Recall 85.74%、Accuracy 57.15%、Score 0.5692；PR-AUC 0.5707、P@50／60／70%=58.50／58.60／58.33%、R@P60%=0.17%、Brier 0.2468、ECE 0.0358 |
| 相較9A | OOS PR-AUC −0.0550；P@50／60／70%分別 −4.27／−3.02／−1.92 pp；R@P60% −77.10 pp；threshold 0.5 Precision −5.26 pp、Lift −5.25 pp、Recall +34.30 pp、模型PASS +37.19 pp、Accuracy +0.98 pp、Score +0.0850；Brier惡化0.0015、ECE改善0.0364 |
| 相較8F | OOS Precision／Lift各 −0.73 pp，但Recall +8.78 pp、模型PASS +9.38 pp、Accuracy +0.39 pp、Score +0.0195；更寬鬆卻沒有更高Precision，未形成新的高coverage定位 |
| 泛化判讀 | Selection→OOS PR-AUC −0.0412，R@P60%由63.86%崩落至0.17%；threshold Precision雖上升2.55 pp，但來自score／coverage大幅漂移，不能掩蓋高分排序幾乎失效 |
| 判定 | 9F五項主要排序指標全面低於9A，且OOS PR-AUC只比原始PASS率高1.44 pp；高Recall主要來自放行82.62%候選，不是更強篩選。停止Patch size、depth、heads、embedding、pooling、position、threshold與masked-pretraining細調 |
| 下一步 | 停止現有300×10單模型architecture橫向搜尋。下一階段先固定9A分數與threshold 0.5，進行策略層no-filter vs 9A經濟效果驗證；不重新訓練、不依同一OOS回調門檻，主要檢查淨報酬、最大回撤、報酬／最大回撤、年化報酬、Log R²、月勝率、交易數與持股缺口 |


### 策略層驗證前置修正：unique-group score 單一真理與逐年 OOS 診斷

| 項目 | 內容 |
|---|---|
| 狀態 | `IMPLEMENTED`；尚未重新匯出9A分數或取得新的策略結果 |
| 程式基準 | 來源`test-branch-1_20260726_123557_2b7b48f.zip`，SHA256 `48029b98e0bc53ae03920a6b6fc07405462b001b9d8f3633f43d9c7aa908893f`；active仍為`inception_time_v1 / unique_group_sampling / threshold 0.5`；沒有新增model architecture或experiment profile |
| 唯一行為修正 | 對`use_dataset_context=false`模型，每個canonical ticker/date feature group只推論一次並在probability層精確broadcast至全部high_len event rows；evaluate對此類模型要求同group score完全相同，不再以`5e-4`容忍差異 |
| Legacy相容 | 使用Dataset event context的legacy architecture仍保留逐event-row inference與既有bounded-noise診斷，不改舊checkpoint／manifest重建語意 |
| Dataset／Label | 不需重建Dataset或relabel；沿用既有`feature_bank.npy`、`event_group_index.npy`與固定百分比Label；新增ticker/date↔group_index一對一完整覆蓋驗證 |
| 報表 | 完整OOS新增逐年度分類、PR-AUC、P@50／60／70%、R@P60%、Brier與ECE；部分calendar year明確標記。年度表只作同一份固定OOS score診斷，不允許調參 |
| 既有結果 | 9A／8F與9B～9F歷史Selection／OOS數值不因本次實作預先改寫；必須重新匯出9A research scores後才產生新報表 |
| 歷史契約銜接 | 2026-07-24曾為保留舊event-row batch下的同checkpoint末位數值而退回unique-group export；本次在策略層驗證前正式改以「同ticker/date只有一個canonical score」為優先契約。重新匯出的9A score可能因推論batch單位改變而與舊表有微小差異，屬明確execution contract變更，不可混用新舊score或要求末位數值完全相同 |
| 下一步 | 重新匯出修正後9A score並確認同ticker/date完全一致，之後以固定策略參數執行no-filter vs 9A threshold 0.5經濟效果比較 |

Formal bundle閉環（2026-07-26 13:26）：來源程式ZIP `test-branch-1_20260726_132542_f054eb2.zip`，SHA256 `59be5c15356b681461ca78cd0fb5164d4d9d232b8cf816ca6ede13ac13c92b8c`；bundle `to_chatgpt_bundle_20260726_132656_a0fd8473.zip`，SHA256 `5f3431e922780847ccecdeb68cb90f2237da38f052e27404ef265bfc851176f7`。quick gate、chain checks與ML smoke均PASS；consistency僅`default_primary_param_fallback_filename`失敗，meta quality也只因同一synthetic case連帶FAIL。根因是`resolve_default_primary_param_source_record()`在缺少optional `models/run_best_params.json`時，錯把第一個現存的`base_best.json`當成預設runtime來源，違反既有ARCHITECTURE與B163契約。已修正為：只有`V16_RUN_BEST_PARAMS_PATH`可覆寫；沒有override時永遠解析至`models/run_best_params.json`，其他現存工件只可由`discover_model_param_sources()`供互動選擇，不得靜默改變預設來源。此閉環不改9A模型、score inference、Dataset、Label、threshold、年度指標或策略邏輯。


### 3.35 Unique-group重跑確認與固定9A策略對照入口（2026-07-26）

| 項目 | 紀錄 |
|---|---|
| 狀態 | unique-group score與逐年OOS：`RESULT_AVAILABLE`；固定9A策略對照工具：`IMPLEMENTED`，尚未取得真實投組結果 |
| 程式基準 | `test-branch-1_20260726_133801_f785df1.zip`；SHA256 `aecf6f4e5f19fd6785f209339be388fb2632b0a7ca9b11bdb4fb2c08af9dfcd6` |
| 結果來源 | `已貼上文字 (1)(14).txt`；SHA256 `b6278b87f7ac045010d9799b4cab63d301be61ea4d5a0e99b84e4e6ae63983eb` |
| 重跑確認 | active 9A `inception_time_v1 / unique_group_sampling / threshold 0.5` 正常完成訓練、unique-group research score export與年度報表；OOS Precision 62.99%、Lift +7.35 pp、Recall 51.44%、模型PASS 45.43%、PR-AUC 0.6257，與第2.4節正式值一致 |
| 年度診斷 | 2021／2023／2024／2025 的 PR-AUC 分別為0.6158／0.6545／0.6612／0.6878；2022為0.4582且P@50%只有42.96%，顯示該年度排序失效。2024 threshold 0.5模型PASS僅22.55%，但P@50%仍66.46%，顯示主要是score尺度／coverage收縮而非排序崩落 |
| 唯一實作變更 | 新增固定參數策略比較入口；同一參數檔、資金、持股上限、rotation、回測日期與0050基準下，只允許切換`use_breakout_quality_filter=False/True`。另在非訓練portfolio profile附加每日候選供給與持股缺口診斷，不改交易執行 |
| 固定條件 | 不重新訓練、不重算Label、不調threshold、不改search space、不加入8F／ensemble；策略比較必須先建立active 9A canonical `forward_oos` scores並使用其available期間 |
| 輸出 | `strategy_comparison.md/.json`、no-filter與quality-filter的equity／trades／daily-capacity CSV、年度報酬比較CSV |
| Dataset／Label | 不需重建或relabel；若目前只有research scores，需額外執行既有`export-scores --scope forward_oos`，不是重訓 |
| 判定 | 年度表證明單一threshold coverage具有明顯regime差異，因此不得只靠分類表決定部署；下一步直接執行固定9A策略經濟效果對照 |
| 下一步 | 取得真實`strategy_comparison.md/.json`後比較淨總報酬、最大回撤、RoMD、年化、Log R²、月勝率、交易數、候選／持股缺口與年度穩定性；結果只決定是否部署／是否再比較8F，不得回頭調9A |


### 3.60 Selection Point-in-time Score Workflow 第一階段（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE`；已完成程式與獨立合成契約驗證，尚未在完整專案資料上訓練 |
| 程式基準 | 來源 `test-branch-1_20260801_144606_024e49d(2).zip`，SHA256 `762eac6a328c318efa9ecbd1922e29607e06b8f9053a80e1322fff770ccbb6c6` |
| 單一入口 | `apps/breakout_quality.py`主選單改為模型研究、策略驗證、設定／工件狀態三項；舊`apps/breakout_quality_strategy_compare.py`只轉送至新`strategy-compare`子命令 |
| 泛用config | 新增`config/breakout_quality_workflow.py`，集中filter、architecture、experiment profile、seed、PIT日期／fold及策略mode／Score source／buy-sort；選單與工件名稱不寫死9A／11G／11K。後續3.63補正為依profile training objective自動派送binary或continuous流程 |
| 共用pipeline | 新增`continuous_ranker_pipeline.py`重用既有資料載入、percentile target、Validation選epoch、final refit、checkpoint與inference；沒有複製loss或建立第二套訓練語意 |
| PIT builder | 新增expanding-window builder。Train／Validation／final refit均要求事件早於score period且`label_eval_end_date < score_start`；每個group只能由一個未見該事件的fold模型評分 |
| 工件防錯 | 每fold保存日期範圍、row/group coverage、selected epoch、checkpoint與Score hash；串接時檢查重複、缺失、cutoff、identity、有限值與完整coverage。正式Score CSV不含Future Target |
| 模型audit | 新增PIT audit，離線join Target後輸出global／daily Spearman、年度與decile spread、fold分布／drift、PASS分類重疊與orderable coverage；已支援既有orderable工件的`target_date`日期欄位 |
| Runtime邊界 | Builder manifest固定`eligible=false`與`selection_model_validation_only`；不作forward-OOS runtime、scanner、buy-sort或optimizer輸入。策略選單在PIT score-store與泛用Score排序尚未實作時明確阻擋 |
| 驗證 | 已通過modified-files編譯、CLI help、舊入口轉接、AST無循環／無apps反向依賴、無bare except，以及獨立合成資料的fold embargo、coverage、duplicate rejection與orderable `target_date` coverage；依專案規範未執行`apps/test_suite.py` |
| 尚未完成 | 未建立真實`selection_point_in_time_scores.csv`、未取得模型Spearman結果、未修改buy-sort、未跑策略optimizer或OOS績效比較 |


### 3.61 Formal quick-gate 舊 Strategy Compare Help 閉環（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING`；已完成根因確認、程式修正與獨立CLI重現，待使用者重新執行正式suite確認 |
| 程式基準 | 來源`test-branch-1_20260801_170534_3cc7f52.zip`，SHA256 `881a4670875a0e633d8ae74eff90f6b96b33e2df9e11b4b05717f64d94d3a003`；bundle `to_chatgpt_bundle_20260801_170703_78c6748d.zip`，SHA256 `45021dfceb2c12241d66669f13922dfc18f54e1782fa45d09160bf73564ca8d5` |
| Formal結果 | quick gate唯一FAIL為`help::breakout_quality_strategy_compare.py`；consistency、chain checks、ML smoke與meta quality均PASS。Bundle manifest共58個工件，逐檔size與SHA256驗證一致 |
| 根因 | 舊`apps/breakout_quality_strategy_compare.py`雖正常以exit code 0轉呼叫統一入口，但轉送時把program name硬設為`apps/breakout_quality.py`，使help首行顯示`breakout_quality.py strategy-compare`；quick gate依舊CLI契約要求看到`breakout_quality_strategy_compare.py`，因此判FAIL |
| 唯一修正 | 舊adapter轉送時保留自己的program name；統一入口新增泛用legacy command-entrypoint alias解析，只有alias已代表該子命令時不再把command附加到`sys.argv[0]`。舊入口仍不建立argparse或第二套strategy compare邏輯 |
| 獨立驗證 | `python apps/breakout_quality_strategy_compare.py --help`回傳0並顯示`usage: breakout_quality_strategy_compare.py ...`；`python apps/breakout_quality.py strategy-compare --help`回傳0並維持`usage: breakout_quality.py strategy-compare ...`。全專案251個Python檔編譯／AST通過、無bare except、無top-level import cycle、core／config／filters／strategies無反向依賴apps |
| Dataset／模型 | 不需重建Dataset或relabel；不改continuous-ranker、PIT folds、checkpoint、Score、9A、Target、optimizer、buy-sort、portfolio replay或正式策略參數 |
| 下一步 | 使用者套用修補後重跑formal suite；若quick gate通過，即繼續建立Selection point-in-time Scores，不新增其他模型或排序變更 |


### 3.62 Selection PIT Scores完成與Orderable Coverage欄位碰撞修正（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `SCORES_BUILT / AUDIT_FIX_IMPLEMENTED / AUDIT_RESULT_PENDING`；真實Selection PIT Scores已完成，模型audit因orderable候選既有Score欄位碰撞中止，本輪已修正程式但尚未取得重跑結果 |
| 程式基準 | 使用者結果基準`test-branch-1_20260801_173038_874fe01.zip`，SHA256 `e0271e5d47c31176956a51057b0f1eabdb3f03fbb054b4022426488a7bd30914` |
| PIT建立結果 | 7個年度expanding folds，評分期間2014-01-01～2020-12-31；串接18,247 groups，coverage=1.0000。各fold score groups依序為2,077、1,803、2,521、3,068、1,853、3,507、3,418 |
| Epoch選擇 | 各fold selected epoch依序為3、4、2、2、1、2、1；對應Validation mean daily Spearman為0.2707、0.2446、0.1536、0.2293、0.2693、0.2513、0.2516。這些是fold內Validation選epoch數值，不是PIT評分期間的模型audit結果 |
| 已建立工件 | `selection_point_in_time_scores.csv`、combined manifest與coverage，以及7個fold checkpoint／scores／manifest；使用者執行輸出顯示combined coverage完整 |
| Audit失敗根因 | 既有`selection_orderable_candidates.csv`含舊`breakout_quality_score`；audit再merge PIT lookup同名欄位後，pandas改名為`breakout_quality_score_x/y`，後續讀取原欄名發生`KeyError`。此錯誤發生在Scores完成後的離線coverage audit，不影響fold訓練、checkpoint或已輸出的PIT Score值 |
| 唯一修正 | PIT lookup在merge前改用audit保留欄名；coverage只讀該保留欄位。候選工件原有Score即使存在且皆為有限值，也不得被誤算為PIT coverage；另回報候選工件是否帶有舊Score與實際coverage score source |
| 防錯驗證 | 合成orderable工件刻意帶入同名舊Score：一筆有PIT match、一筆無PIT match但舊Score有效；修正後coverage仍正確為1／2=0.5，證明未誤用舊Score。保留欄位若被外部工件占用則fail-fast |
| Dataset／模型 | 不重建Dataset、不relabel、不重訓fold、不改continuous-ranker、selected epoch、checkpoint、Score CSV、Target、seed、buy-sort或策略參數 |
| 下一步 | 直接重跑`python apps/breakout_quality.py audit-point-in-time-scores`；取得global／daily Spearman、年度spread、fold drift及orderable coverage後，才判定是否進入泛用Score buy-sort。不得因本次audit程式錯誤重跑7個fold |


### 3.63 泛用Profile Workflow Router修正（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE`；完成程式、設定與獨立路由驗證，未重新訓練模型或執行策略回放 |
| 程式基準 | 來源`test-branch-1_20260801_173825_934eddc.zip`，SHA256 `f4a75d6a5d5d74b3c5579b28148684bcb155f2a35b412cd73622b6498fb1de35` |
| 問題 | 第一階段主選單雖使用泛用名稱，但`get_breakout_quality_workflow_settings()`、狀態頁、`[Enter]`與`[1]`均硬綁continuous ranker／PIT流程；把profile改成9A `unique_group_sampling`會直接報錯，違反config驅動設計 |
| 唯一修正 | Experiment profile既有`training_objective`成為流程派送單一來源。Binary classification走既有完整classification research workflow；continuous ranker維持Selection PIT builder＋audit。狀態頁只顯示當前objective相關工件 |
| 策略auto契約 | `BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE／SCORE_SOURCE／BUY_SORT`新增`auto`：binary解析為`hard-filter／canonical_runtime／original`；continuous解析為`score-ranking／selection_point_in_time／breakout_quality_score_desc`。明確覆寫仍需通過跨欄一致性檢查 |
| 參數政策 | `base-finalist-best`／`base-finalists-agree`不再只限score-ranking；hard-filter也可由統一選單忠實傳入並以不同輸出目錄隔離。Controlled pair仍只允許切換相應filter或ranking開關；`attribution-only`與11C預設改由同一canonical directory discovery辨識policy-specific及舊目錄 |
| 切換方式 | 原9A與continuous皆只切換workflow profile；Seed統一由`BREAKOUT_QUALITY_RANDOM_SEED`控制，預設42，不再依profile切換。此列原先的continuous seed=1規則已由3.67取代 |
| 固定條件 | 不改active architecture、Dataset、Label、threshold、continuous Target、PIT folds／checkpoint／Scores、portfolio accounting、候選生成、成交、出場或正式runtime開關 |
| Dataset／Label | 不需重建Dataset或relabel；只有使用者在binary模型流程確認開始訓練時，才依既有workflow規則檢查並更新工件 |
| 驗證 | 已獨立驗證binary／continuous settings解析、binary模型route、binary hard-filter策略route、hard-filter參數政策輸出隔離與下游目錄發現、CLI help、全專案編譯／AST與依賴方向；依專案規範未執行`apps/test_suite.py` |
| 結果邊界 | 本輪只修正操作與配置契約；不得據此宣稱9A或continuous模型、排序或策略績效改善 |


### 3.64 Formal Consistency Canonical Strategy-Compare Fixture閉環（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING`；根因已由formal bundle確認，validator fixture已同步，待使用者重跑正式suite確認 |
| 程式基準 | 來源`test-branch-1_20260801_175951_e781354.zip`，SHA256 `f2d8b1dcc1d79d848013995f68fc80697039776c7837ce010cf19df966e63a8c`；bundle `to_chatgpt_bundle_20260801_180117_5ac62bbf.zip`，SHA256 `e33c8a5ace14bd0e2fbef31a30d97d06550d04d5473dd18393e495a7e6566d5d` |
| Formal結果 | quick gate PASS；consistency為4,999 PASS／30 SKIP／1 FAIL；chain checks與ML smoke PASS；meta quality唯一FAIL為`coverage_synthetic_suite_runs_successfully`。Consistency唯一失敗metric為`continuous_target_round_trip_auto_path_uses_output_tree` |
| 根因 | Router修正後hard-filter strategy compare正式優先目錄為`strategy_compare_base_finalist_best`，舊`strategy_compare`只作相容fallback。Runtime正確把舊目錄回報為`active_9a_strategy_compare_discovery`；synthetic fixture仍把檔案建立在舊目錄，卻期待`active_9a_standard_path`，形成validator自相矛盾 |
| 唯一修正 | Synthetic fixture不再硬編碼`strategy_compare`；改呼叫`canonical_strategy_compare_output_dir_names(COMPARISON_MODE_HARD_FILTER)[0]`建立目前正式優先目錄，並保留`active_9a_standard_path`預期。Runtime discovery、目錄優先序與相容fallback均不修改 |
| Meta quality閉環 | Coverage比例、critical targets與checklist本身均已通過；meta quality失敗只因synthetic fail count=1。修正同一fixture後，該衍生FAIL應同步消失，但正式結果仍須由本機重跑確認 |
| Dataset／模型 | 不重建Dataset、不relabel、不重訓9A或continuous folds；不改checkpoint、PIT Scores、Target、threshold、seed、optimizer、buy-sort、portfolio replay或策略參數 |
| 獨立驗證 | 已獨立驗證canonical優先目錄回報`active_9a_standard_path`、舊目錄回報`active_9a_strategy_compare_discovery`、全專案AST／compile、bare-except、依賴方向、import cycle、Markdown table與formal registry／checklist一致性；依規範未執行`apps/test_suite.py`或其正式step |
| 下一步 | 使用者覆蓋修補後重跑正式suite；預期consistency與meta quality同時恢復PASS。若仍有新FAIL，再依新bundle閉環，不修改runtime以迎合舊fixture |


### 3.65 Breakout-quality Config單一來源整併（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE`；完成設定架構整併與獨立相容驗證，未重新訓練模型或執行策略回放 |
| 程式基準 | 來源`test-branch-1_20260801_180913_c77a7cc(1).zip`，SHA256 `90691c277723f2846e94f56c4bc4f17f9c852112d65805b46ecee9a75088b101` |
| 問題 | `breakout_quality_policy.py`、`breakout_quality_experiments.py`與`breakout_quality_workflow.py`分散模型、profile與workflow設定；使用者切換9A／continuous流程時需理解三個檔案，且容易誤改非作用中的設定 |
| 唯一修正 | 新增`config/breakout_quality.py`作唯一可編輯設定來源。檔案最上方先放主選單workflow profile切換，再依模型identity、Dataset／Label、architecture、training、validation、execution、PIT與策略分類排列；profile類別／registry、驗證、衍生值與helper全部集中下半部 |
| 最終檔案契約 | 依使用者要求只保留必要檔案，三個舊設定檔已刪除；專案內部與外部腳本都必須直接import`config.breakout_quality`，不提供會隱藏殘留依賴的alias |
| 行為一致性 | 合併前後132個公開名稱完整保留；所有可序列化設定、active profile payload、workflow manifest payload、Inception kernels與receptive field逐項一致。後續刪除alias檔只移除舊import入口，不改canonical設定值、衍生結果或runtime行為 |
| 固定條件 | 不改active architecture、binary／continuous profile定義、optimizer、Label、threshold、seed目前值、PIT fold／checkpoint／Scores、strategy mode解析、portfolio accounting、候選生成、成交或出場 |
| Dataset／Label | 不需重建Dataset或relabel；設定檔整併不改任何artifact identity或hash契約 |
| 結果邊界 | 本輪只改善設定可維護性與單一真理來源；不得據此宣稱9A、continuous ranker或策略績效改善 |


### 3.66 單一Config刪檔後Formal閉環（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING`；已修正bundle確認的validator／文件殘留，待使用者重跑正式suite確認 |
| 程式基準 | 來源`test-branch-1_20260801_183429_c2f3b2a.zip`，SHA256 `e5cf9d58b8b3f5165cff24a2e17889247e3ce6add46656aaf9a83cd1e1f8faa6`；bundle`to_chatgpt_bundle_20260801_183502_5e1d75ef.zip`，SHA256 `a4b1867dab4e3c7ed3c07573ec13d363be622b4aec82d1113ccfbf59e68c1f47` |
| Formal結果 | quick gate PASS；consistency 956 PASS／30 SKIP／1 FAIL；chain checks與ML smoke PASS；meta quality FAIL 4。Consistency唯一FAIL為synthetic suite啟動時`ModuleNotFoundError: config.breakout_quality_policy`；coverage run info顯示returncode=1、synthetic_case_count=0 |
| 根因 | Runtime與專案內部imports已正確使用`config.breakout_quality`，但`validate_breakout_quality_policy_single_source_case`仍把三個舊alias檔存在與可import當成成功條件。使用者刪除舊檔後，該case在產生check row前就拋例外，連帶使coverage line／branch／key-target gates因0個synthetic cases而失敗 |
| 唯一修正 | Single-source validator改為要求canonical檔存在、舊三檔全部不存在，並繼續AST／source掃描runtime是否殘留舊import；同步canonical docstring、CMD、Architecture與checklist。沒有恢復alias檔來迎合舊測試 |
| Meta quality閉環 | 四個meta failures均是synthetic suite未執行的衍生結果；修正後suite可進入原coverage cases。實際coverage百分比仍以本機正式重跑為準，不預先寫成PASS |
| Dataset／模型 | 不重建Dataset、不relabel、不重訓9A或continuous folds；不改checkpoint、PIT Scores、Target、threshold、seed、optimizer、buy-sort、portfolio replay或策略參數 |
| 獨立驗證 | 已獨立執行changed-case、全專案compile／AST、舊import AST掃描、bare-except、依賴方向、import cycle、CLI、binary／continuous workflow解析、Markdown table與checklist transition檢查；依規範未執行`apps/test_suite.py`或其正式steps |
| 下一步 | 使用者覆蓋修補後重跑正式suite；預期consistency與四個coverage衍生FAIL同步消失。如coverage仍有獨立FAIL，再以新bundle追查，不恢復舊config檔 |


## 4. 已排除或暫停的方向

下列方向已有足夠證據，不應在沒有新機制或新資料證據時重複測試：

1. 退回 Tiny CNN。
2. Residual TCN 加深／加大容量。
3. Short／Medium／Long 的更多 Level／Return 組合。
4. 相對 0050 Return 的同資訊線性重組。
5. Long Branch channels 8～16 間的細部搜尋。
6. Long Branch dropout 的細部搜尋。
7. Matched optimizer steps final refit；重複 event-row sampling 與 unique-group sampling（8I）都已證明會用較高 Precision 換取過大的 Recall、Accuracy 與 Score 損失，不再細調 step ratio。
8. Inverse-frequency class weight 或 year-balanced time weight。
9. 三個連續 auxiliary targets 同時加入。
10. 只把 OOS 縮短成下一年，期待模型自然改善。
11. 把 multi-fold validation 誤當成會直接提高 OOS 的模型改動。
12. 舊歷史 contiguous masking 與其條件式 noise 延伸。
13. 目前的 6 維低維市場 regime context；Selection 內 validation 有改善，但 OOS Precision、Accuracy 與 Score 低於 v1。
14. ATR／波動率尺度 Label；使用者已實測整體 OOS 沒有優於固定百分比 Label。
15. 把分年、分季度、分 Fold 或 threshold 診斷當成會直接提升模型能力的實驗；這些只能解釋問題，不列為目前改善優先。
16. Recent-decay time weighting；60 個月半衰期使 Selection 指標上升，但完整 OOS Precision、Recall、Accuracy、Score 與泛化落差全部惡化。不得再細調 36／48／72／84 個月半衰期。
17. Same-day 20／60／120 日報酬 percentile context；Selection Precision／Lift 上升，但完整 OOS 主要指標與泛化落差全部低於 8A。停止增加 handcrafted／rank／regime context。
18. Same-day pairwise ranking loss；使模型更保守，OOS Precision 未提高且 Recall、Accuracy、Score 下降。不得再調 ranking weight、margin 或 pair sampling。
19. 固定 8-seed probability ensemble；OOS Precision、Recall、模型 PASS、Accuracy、Score 與泛化落差全面低於 8A。不得再細調 seed 數、平均 logits／probabilities、投票或 min-agree。
20. Unique-group training 增加 early-stopping patience／直接延長 epochs；patience 5 使 Selection 變好但完整 OOS Precision、Recall、Accuracy、Score 與 drift 全面惡化。
21. Unique-group batch size 由 128 降至 64，或繼續細調 32／64／96 等較小 batch；8H 已證明 optimizer updates 加倍仍未提高 OOS Precision／Lift。
22. Unique-group matched optimizer steps final refit；8I 雖提高 OOS Precision 0.65 pp，但 Recall、Accuracy、Score與主要泛化落差均明顯低於 8F。
23. 直接採用 best inner-validation checkpoint；8J 的 OOS Precision僅比 8F 高 0.04 pp，但 Recall、模型 PASS、Accuracy 與 Score 明顯下降。停止繼續變形 Final Refit mode。
24. 完整日期等權 `1 / 當日 group 數` 作為 threshold 0.5正式模型；8K 在此操作點幾乎全部判 PASS。8K@0.55只保留固定門檻的 forward-validation候選，不再用既有 OOS細調門檻。
25. Date-diverse／ticker-diverse batch scheduling；8L 所有主要 OOS絕對指標均未超越 8F。
26. Long branch last-state pooling；8M 的 OOS Precision、Recall、Accuracy、Score及泛化 gaps均低於 8F。
27. Zero-initialized gated temporal pooling；8O 的 OOS Precision、Recall、Accuracy均低於 8F，只有平均 Score增加 0.0011，主要 gaps未改善。停止 gate width、temperature、初始化與 attention pooling細調。
28. Raw＋window-normalized dual-path；8P只讓OOS Precision增加0.06 pp，卻使Recall下降8.48 pp、Accuracy下降1.30 pp、Score下降0.0143。停止現有multiscale CNN家族微調。
29. InceptionTime GroupNorm；9A-GN雖改善ECE與Score gap，但OOS PR-AUC下降0.0607、P@50／60／70%下降3.31／3.02／2.36 pp，固定coverage排序明顯崩落。不再搜尋GroupNorm group數、LayerNorm或其他只換normalization的細調。
30. ModernTCN supervised architecture search；9B Selection全面變強但OOS PR-AUC、固定coverage Precision、Accuracy、Brier與ECE全面惡化。停止ModernTCN depth、kernel、channels、expansion、dropout與其他supervised CNN／TCN橫向調整。
31. TS2Vec Selection-only frozen probe；9C OOS PR-AUC較9A低0.0604，P@50／60／70%低4.61／3.77／2.92 pp，模型PASS升至88.78%但Lift只剩+0.48 pp。Frozen representation不具穩定線性可分性，不啟動9C2，不再調pretraining epochs、stride、crop、mask、pooling、head、fine-tuning或threshold。
32. MantisV2 frozen probe；9D雖有穩定的threshold 0.5 coverage與較高Recall，但OOS PR-AUC較9A低0.0324，P@50／60／70%低2.72／2.23／1.55 pp，R@P60%低22.12 pp。未達fine-tuning啟動條件，不再調threshold、adapter、head、channel aggregation、output layer/token或encoder fine-tuning。
33. MOMENT-1-base frozen probe；9E OOS PR-AUC較9A低0.0400，P@50／60／70%低4.73／4.12／3.48 pp，R@P60%低54.20 pp，且連8F高覆蓋基準也未超越。低ECE不是排序改善，不啟動fine-tuning，不再調threshold、output reduction、patch／stride、pooling、head、adapter或同checkpoint變體。
34. 小型supervised Patch Transformer；9F OOS PR-AUC較9A低0.0550，P@50／60／70%低4.27／3.02／1.92 pp，R@P60%只剩0.17%。不啟動9F-B masked pretraining，不再調patch size、embedding、depth、heads、MLP、position、pooling或threshold；停止現有300×10單模型architecture橫向搜尋。
35. 9A固定threshold 0.5 runtime gate；無前視Rolling active-param OOS中淨總報酬147.71%降至121.14%、MDD由16.42%惡化至20.75%、RoMD由9.00降至5.84，Payoff、EV、曝險與最差完整年度亦全面惡化。不得依同一OOS調threshold、年度門檻、calibration或regime開關；9A只保留研究排序基準。
36. 9A交易層歸因；共同交易R差異只有+0.15R，全部−46.62R落差幾乎全由獨有交易選擇效果−46.77R造成。被排除贏家326.25R大於避開輸家138.76R，替代交易平均R與Payoff亦低於被取代交易；停止以現有固定百分比Label／score作硬式進場gate，也不得依既有OOS設年度或regime開關。
37. 9A Quality Score作候選全域第一排序；`base_finalist_best`單一member隔離比較使淨總報酬下降32.12pp、RoMD下降2.10、平均曝險下降27.96pp，且只有2023改善。不得再調Score權重、票數／Score混合公式、top-k、年度門檻或regime開關；`base_finalists_agree`同票Score tie-break的+4.88pp也只屬單一regime探索結果，不得部署。
38. 為了讓Score實驗較單純而切換正式selector至`base_finalist_best`；其no-filter baseline相較`base_finalists_agree` baseline的總報酬低18.63pp、MDD較高0.99pp、RoMD低1.58。此跨selector比較只作描述性診斷，但已沒有支持正式切換的經濟證據。
39. 把2022排序失效主要歸因於combined-regime low-support，或只靠oversampling／重複少量deep-drawdown事件修復；排除36.01%的low-support事件後PR-AUC只由0.4582升至0.4722，已見regime仍廣泛失效。
40. Candidate-conditioned Market Set與其learned-lag延伸；10A OOS PR-AUC較9A低0.0248，P@50／60／70%低0.75／0.41／0.47 pp，R@P60%低2.69 pp。Recall增加7.30 pp只因模型PASS增加7.34 pp，不是排序改善；不啟動10B learned lag、query數／heads／embedding或Market Set微調。
41. 11B同日percentile MSE與其直接微調；OOS mean daily Spearman僅0.1327、PR-AUC 0.5959，actual trade R Spearman −0.0003，Score前10%平均0.6500R反而低於後10%的2.0251R。不得再調MSE／Huber、epoch、patience、LR、batch、percentile公式、直接pairwise loss或同一全事件訓練母體。
42. 直接建立qualified-candidate-only sampling profile；11C顯示Score↔Target由all OOS 0.1677升至qualified 0.1918、orderable 0.1891，actual trades更達0.2459，沒有母體崩落證據。不得以候選母體不一致為理由直接重訓。
43. 直接以全Label重新訓練No-time percentile ranker，或搜尋time penalty正負號／係數；11F Binary AUC約0.99，證明全Label loss會再次被PASS／REJECT分離支配。11G已完成唯一允許的PASS-only測試並淘汰，不得回頭加入BCE、pairwise、qualified sampling、time權重搜尋或9A Score blending。
44. 微調11G PASS-only magnitude ranker；雖OOS PASS-only Score↔Target達0.3464、mean daily Spearman 0.2944，但actual PASS Score↔R為−0.1066，Score top decile僅0.6117R、bottom decile3.5519R。不得再調MSE／Huber、epochs、patience、LR、batch、percentile、head或與9A融合。
45. 直接以11I actual portfolio round trips建立完整strategy-realization target；11I僅415筆Target matched trades，actual trade coverage只占qualified 21.77%。未成交或因capacity／cash competition未入選的候選不得填0R，也不得把portfolio selection偏差當成全部候選Target。
46. 繼續修補11J counterfactual replay一致性；六次本機執行仍無法重現11I的2,003筆，且曾出現MemoryError。11J正式停止，不再改core／observer／sidecar／日期或identity；後續只使用11I已凍結工件做read-only歸因。
47. 將Future Target直接作runtime buy-sort、以OOS搜尋Score權重、重試11G loss／epoch／LR／batch／sampling、把完整Selection 9A重訓當成本輪前置條件、或新增11L版本名稱。後續只先驗證既有PASS-only No-time continuous ranker的Selection point-in-time預測能力，再決定是否進入泛用Score排序與策略參數適應。

---

## 5. 接下來要嘗試的列表

所有實驗一次只改一項。既有 OOS 可持續作為固定比較集；每次模型的訓練、Validation、early stopping 與 epoch 選擇必須完全限制在 Selection 內，完整 OOS 只能在模型凍結後執行。OOS 結果可以用來接受、淘汰或形成下一個實驗，不再以「OOS 已被查看」作為停止研究的理由。正式 runtime 仍維持 `base_finalists_agree` 既有排序且 Quality Ranking 關閉，除非新實驗同時通過模型指標與策略經濟效果。

### 目前新增優先：Selection Point-in-time Continuous-ranker Workflow

| 項目 | 設計 |
|---|---|
| 狀態 | `MODEL_DIRECTION_PASS / STRATEGY_RESULT_PENDING`；2026-08-01使用唯一正式Seed 42完成7-fold／18,247 groups，模型方向已通過；不需因Seed重建 |
| 研究依據 | 既有11G能預測部分PASS-only No-time Target，但完整Selection refit Score不具備策略optimizer所需的未見資料性質；actual trades又受到舊排序、持倉與資金限制，因此先建立與正式OOS相同語意的PIT Score |
| 唯一變更 | 使用expanding-window folds；每fold只用score period以前、且`label_eval_end_date < score_start`的歷史資料完成Inner Validation、epoch selection及final refit，再只評分下一段未見資料 |
| 固定模型 | architecture／experiment profile／target由`config/breakout_quality.py`指定；binary、continuous、pretraining與PIT全部共用`BREAKOUT_QUALITY_RANDOM_SEED=42`。目前仍為PASS-only No-time magnitude continuous ranker，不重試MSE／Huber／epoch／LR／batch／sampling |
| 正式工件 | `selection_point_in_time_scores.csv`、combined manifest／coverage，以及每fold checkpoint、scores、manifest與SHA256 |
| 模型audit | 先計算Score↔Target Spearman、mean daily Spearman、年度與decile spread、fold drift、PASS分類重疊及orderable coverage；Future Target只在audit離線join，不寫入Score CSV |
| 策略邊界 | Builder manifest仍固定`eligible=false`；策略入口另以audit gate、Score hash、identity與Seed驗證授權Selection replay。已接入泛用Score buy-sort，但尚未執行策略結果 |
| 下一步 | 直接執行Baseline／Sort Only策略比較；既有Seed 42 PIT Scores與audit可作正式前置，不需重建 |
| Dataset／training | 不重建binary Dataset、不重訓完整Selection 9A；只沿用既有continuous-ranker training pipeline建立歷史fold模型 |
| 執行入口 | `python apps/breakout_quality.py build-point-in-time-scores`；完成後執行`audit-point-in-time-scores` |
| UI／runtime | 新主選單不含9A／11G／11K名稱；experiment profile的training objective自動派送binary classification或continuous PIT流程；research-only低階功能維持CLI-only；PIT Score不是scanner或forward-OOS runtime工件 |
| 啟動錯誤閉環 | `IMPLEMENTED / RESULT_NOT_AVAILABLE`：完整Dataset尾端Label horizon未完成group可合法使整組`label_eval_end_date`皆空；group一致性檢查改以`nunique(dropna=False)`把「全空」視為單一一致狀態，同時仍拒絕同group混用空值與完成日期或多個完成日期。這些group因`target_valid=false`及日期比較為False，不會進入train／validation／final refit；本輪只解除誤判，不放寬前視隔離 |

### 優先 6A：AdamW only

| 項目 | 設計 |
|---|---|
| 狀態 | `REJECTED`；AdamW OOS 低於 baseline |
| 唯一變更 | experiment profile `baseline` → `adamw_only`；實際學習差異只有 `Adam` → `AdamW` |
| Learning Rate | 維持 0.0003 |
| Weight Decay | 維持 0.0001，不同時改 regularization 強度 |
| LR Schedule | `none` |
| Augmentation | `none` |
| Dataset rebuild | 不需要 |
| 目的 | 隔離檢查 decoupled weight decay 是否得到較平滑、較能泛化的解 |

判定條件：OOS Precision Lift 不低於 +3.66 pp、Accuracy 不低於 53.34%，Recall 不應明顯低於 51.49%，且至少一項 Selection→OOS 落差縮小。

### 優先 6B：Step-based learning-rate schedule

| 項目 | 設計 |
|---|---|
| 狀態 | `REJECTED`；OOS Precision、Recall、Accuracy 均低於 baseline |
| 唯一變更 | 退回 baseline Adam，只加入 step-based schedule |
| Warmup | 前 5% optimizer updates 線性 warmup |
| Decay | 後 95% 使用 cosine decay |
| Minimum LR | 初始 LR 的 10%，即 0.00003 |
| 更新單位 | 每次 optimizer update，不是每個 epoch |
| Refit | Inner training 與完整 Selection refit 各自依實際 total steps 重建 scheduler |
| Dataset rebuild | 不需要 |
| 目的 | 降低後期 overshoot，觀察是否改善 OOS 而非只降低 Selection loss |

6A 已被拒絕，因此 6B 固定使用 Adam，只加入 scheduler，確保仍是單一變更。

### 優先 7A：輕量舊歷史 contiguous masking augmentation

| 項目 | 設計 |
|---|---|
| 狀態 | `REJECTED`；主要 OOS 指標低於 baseline |
| 唯一變更 | experiment profile `baseline` → `history_masking_only`；Training batches 加入一段舊歷史 masking |
| 適用區段 | 300 bars 中前 240 bars |
| 保護區段 | 最近 60 bars 完全不動 |
| Mask 長度 | 每次隨機 10～30 bars |
| 套用機率 | 每筆 training sample 50% |
| 填補方式 | 使用遮蔽區段左右邊界線性插值，不直接填 0 |
| Channels | 個股與 0050 的 10 個 channels 使用同一時間區段同步處理 |
| Validation／OOS | 完全不套用 augmentation |
| Dataset rebuild | 不需要；在 training batch 即時產生 |
| 目的 | 降低模型依賴 300 日視窗中某一段 Selection 特有歷史形狀或 regime |

第一個 augmentation 只做 masking，不同時加入 noise、scaling、time warping 或 channel dropout。

### 優先 7B：小幅連續值 noise（條件式）

`CANCELLED`。7A 已明確降低 Precision Lift、Recall 與 Accuracy，不符合啟動條件；不再堆疊 noise、scaling、time warping 或其他 augmentation。

### 後續直接改善整體 OOS 的順序

前述 6A、6B、7A、regime context 與 ATR Label 都已失敗。後續只安排會直接改變模型輸入、訓練分布或學習目標的實驗；分年／分 Fold 報表不再插入主線。

#### 優先 8A：Sequence-only context ablation

| 項目 | 設計 |
|---|---|
| 狀態 | `ACCEPTED`；歷史 architecture 基準，後由 8F unique-group training 取代為正式研究基準 |
| Architecture | `multiscale_cnn_sequence_only_v1` |
| 唯一變更 | 保留 v1 三個 Level branches、pooling 與 head；head 不再拼接原 4 維 handcrafted context，只使用 300×10 sequence summary |
| 完整 OOS | Precision 59.30%、Lift +3.66 pp、Recall 53.47%、模型 PASS 50.17%、Accuracy 53.70%、Score 0.4886 |
| 判定 | Precision 持平，但 Recall、Accuracy、Score 與 Selection→OOS 落差一致改善；採用為新基準 |

#### 優先 8B：Recent-decay time weighting

`REJECTED`。60 個月半衰期使 Selection Precision／Lift 上升，但完整 OOS Precision、Recall、Accuracy、Score 與兩項泛化落差全部惡化。已退回 `multiscale_cnn_sequence_only_v1 / baseline / time_weight=none`，不再細調半衰期。

#### 優先 8C：Same-day cross-sectional rank context

`REJECTED`。相較 8A，完整 OOS Precision −0.54 pp、Lift −0.53 pp、Recall −2.86 pp、模型 PASS −2.25 pp、Accuracy −0.94 pp、Score −0.0172，Precision／Score gap 也都惡化。已退回 8A；不再增加 return、volume、ATR、breakout magnitude 或其他 handcrafted／rank context。

#### 優先 8D：Hybrid classification + same-day ranking loss

`REJECTED`。相較 8A，完整 OOS Precision −0.19 pp、Lift −0.18 pp、Recall −4.07 pp、模型 PASS −3.68 pp、Accuracy −0.86 pp、Score −0.0142；模型更保守但 Precision 未提高。停止 ranking loss 路線。

#### 優先 8E：Fixed 8-seed probability ensemble

`REJECTED`。相較 8A，完整 OOS Precision −1.07 pp、Lift −1.06 pp、Recall −11.94 pp、模型 PASS −10.49 pp、Accuracy −2.80 pp、Score −0.0410，Precision／Score gap 均惡化。Seed variance 不是目前主要瓶頸，不再調 ensemble 聚合。

#### 優先 8F：Unique ticker/date group training

`ACCEPTED`。相較 8A，OOS Precision／Lift 分別低 0.84／0.83 pp，但 Recall +23.49 pp、模型 PASS +23.07 pp、Accuracy +3.06 pp、Score +0.0611；Precision gap 由 −2.77 pp 改善為 +1.56 pp，Score gap 由 −0.0978 改善為 −0.0287。依整體 OOS 泛化優先，升為目前正式研究基準。

#### 優先 8G：Unique-group sampling + patience 5

`REJECTED`。相較 8F，OOS Precision −0.91 pp、Lift −0.91 pp、Recall −33.38 pp、模型 PASS −31.12 pp、Accuracy −6.03 pp、Score −0.0708；Selection Precision 上升但完整 OOS 恢復嚴重過度擬合。已退回 patience 1，不再細調 patience 2～10 或直接增加 group epochs。

#### 優先 8H：Unique-group sampling + batch size 64

`REJECTED`。相較 8F，OOS Precision／Lift 各 −0.34 pp，Recall +0.24 pp、模型 PASS +0.66 pp，但 Accuracy −0.39 pp、Score −0.0057；Precision gap 由 +1.56 pp 縮為 +0.90 pp，只有 Score gap 微幅改善。Optimizer updates 加倍未提高真正篩選力，已退回 batch size 128，不再細調更小 batch。

#### 優先 8I：Unique-group sampling + matched optimizer steps refit

`REJECTED`。相較 8F，OOS Precision／Lift各 +0.65 pp，但 Recall −18.60 pp、模型 PASS −18.31 pp、Accuracy −2.38 pp、Score −0.0219；Accuracy／Score gap 也惡化。已退回 `selected_epochs`，不再細調 matched step ratio。

#### 優先 8J：直接採用 best inner-validation checkpoint

`REJECTED`。相較 8F，OOS Precision／Lift各只增加 0.04 pp，但 Recall −12.09 pp、模型 PASS −11.55 pp、Accuracy −1.91 pp、Score −0.0271；Precision與 Accuracy gap也惡化。已退回 `unique_group_sampling / selected_epochs`，不再繼續變形 Final Refit mode。

#### 優先 8K：Unique-group date-density balancing

`REJECTED`。正式 threshold 0.5／patience 1 下，OOS 模型 PASS 97.64%、Precision 56.08%、Lift僅 +0.45 pp；相較 8F，Precision／Lift各下降 2.38 pp且 Accuracy下降 0.52 pp。Threshold 0.55結果只屬已查看 OOS後的操作點診斷，不得作為模型改善或正式門檻選擇。停止完整日期等權與延長訓練。

#### 優先 8L：Date-diverse batch scheduling

`REJECTED`。相較 8F，OOS Precision／Lift各 −0.05 pp、Recall −2.16 pp、Accuracy −0.41 pp、Score −0.0088；停止 batch scheduling細調。

#### 優先 8M：Long Branch last-state pooling

`REJECTED`。相較 8F，OOS Precision −0.88 pp、Recall −8.56 pp、Accuracy −2.38 pp、Score −0.0127；移除 Long 300-bar average無助於泛化。

#### 優先 8O：Zero-initialized gated temporal pooling

`REJECTED`。相較 8F，OOS Precision／Lift各 −0.13 pp、Recall −2.26 pp、模型 PASS −2.00 pp、Accuracy −0.52 pp，只有平均 Score +0.0011；Precision gap僅改善 0.05 pp，但 Recall、Accuracy與 Score gaps均惡化。停止 gated／attention pooling細調。

#### 後續 8P：Raw＋window-normalized dual-path

`REJECTED`。相較8F，OOS Precision／Lift各只增加0.06 pp，但Recall −8.48 pp、模型 PASS −8.14 pp、Accuracy −1.30 pp、Score −0.0143，且主要Selection→OOS gaps未改善。dual-path轉為legacy read-only；現有multiscale CNN家族微調正式停止。

#### 9A：單一 InceptionTime

`ACCEPTED`。OOS Precision 62.99%、Lift +7.35 pp；固定coverage P@50／60／70%分別為62.77／61.62／60.25%，且相較Selection沒有下降，證明排序能力跨時期穩定。threshold 0.5下Recall為51.44%、模型PASS為45.43%，因此定位為高品質／較低coverage模型；8F保留為高coverage基準。

#### 9A-GN：InceptionTime GroupNorm ablation

`REJECTED`。相較9A-BN，OOS PR-AUC −0.0607，P@50／60／70%分別 −3.31／−3.02／−2.36 pp，R@P60%由77.27%降至0.50%；雖然ECE與Score gap改善，但排序能力明顯崩落。已退回`inception_time_v1`，GroupNorm版本轉為legacy read-only，不再細調normalization。

#### 9C：TS2Vec Selection-only自監督預訓練

`REJECTED`。OOS PR-AUC 0.5653，P@50／60／70%=58.16／57.85／57.33%，均顯著低於9A；模型PASS升至88.78%，但Precision只比原始PASS高0.48 pp。Frozen probe未達9C2啟動條件，已退回9A並停止TS2Vec細調。

Formal bundle閉環（2026-07-25 23:13）：quick gate、chain checks與ML smoke均PASS；consistency有4項FAIL，meta quality僅因`coverage_synthetic_suite_runs_successfully`連帶FAIL。4項皆為9C轉legacy後的validator fixture同步問題：legacy expected set漏列`ts2vec_frozen_linear_v1`；active InceptionTime報表fixture仍混入並要求TS2Vec pretraining欄位；pretraining tamper案例也錯用active InceptionTime manifest。修正後active fixture只驗證9A欄位，另建立獨立legacy TS2Vec report／artifact fixture，持續驗證舊工件可讀且pretraining profile竄改必須fail-fast。此閉環不改9C結果、9A正式基準、Dataset、Label、threshold或任何模型訓練行為。

Formal bundle閉環（2026-07-25 23:30）：quick gate、chain checks與ML smoke均PASS；consistency只剩1項FAIL。獨立legacy TS2Vec fixture將split檔改成1筆train＋1筆validation後，split manifest仍複製較早的2筆train統計，正式artifact validator因`selection_role_counts`不一致而正確fail-fast。修正為沿用同一份`validation_split_record`後再更新檔案hash／size，使split內容與metadata回到單一真理來源；meta quality的synthetic suite與coverage四項FAIL均屬此錯誤的連鎖結果。

#### 9D：MantisV2 frozen encoder＋linear probe

`REJECTED`。相較9A，OOS PR-AUC −0.0324、P@50／60／70%分別 −2.72／−2.23／−1.55 pp、R@P60% −22.12 pp。threshold 0.5下Recall與Accuracy較高，但模型PASS達86.11%，屬更寬鬆的高coverage操作點，未形成比9A更強排序或比8F更高Precision的新定位。MantisV2轉為legacy read-only，不啟動fine-tuning或同家族細調。

Formal bundle閉環（2026-07-26 01:37）：quick gate、chain checks與ML smoke均PASS；consistency僅1項FAIL，meta quality也只因同一項synthetic failure連帶FAIL。9D退回9A時，runtime與model spec已正確將`mantis_v2_frozen_linear_v1`轉為legacy read-only，但`validate_breakout_quality_policy_single_source_case`的顯式legacy expected tuple漏列MantisV2，導致actual比expected多一個正確的legacy architecture。已同步fixture；此修正不改runtime、9D結果、9A／8F active集合、Dataset、Label、threshold或訓練行為。

#### 9E：MOMENT-1-base frozen encoder＋linear probe

`REJECTED`。OOS PR-AUC 0.5857，P@50／60／70%=58.04／57.50／56.77%，R@P60%=23.07%，相較9A全面下降；threshold 0.5下模型PASS 75.31%、Recall 76.35%，但Precision只比原始PASS高0.77 pp。ECE較低不代表排序改善。MOMENT轉為legacy read-only，不啟動fine-tuning或同checkpoint細調。

#### 9F：小型Patch Transformer supervised architecture

`REJECTED`。OOS PR-AUC 0.5707，P@50／60／70%=58.50／58.60／58.33%，R@P60%=0.17%；相較9A分別低0.0550、4.27／3.02／1.92 pp與77.10 pp。threshold 0.5下Recall 85.74%、模型PASS 82.62%，但Precision只有57.73%，屬大量放行而非排序改善。Patch Transformer轉為legacy read-only，不啟動9F-B或同架構細調。

#### 下一階段：固定9A的策略層經濟效果驗證

| 項目 | 固定設計 |
|---|---|
| 狀態 | `IMPLEMENTED`；非新模型訓練，待本地真實策略結果 |
| 第一個單一比較 | no-filter基準 vs 固定9A `inception_time_v1 / unique_group_sampling / threshold 0.5` |
| 固定條件 | 不重新訓練、不重算Label、不調threshold、不加入8F；正式歷史OOS使用同一份Rolling OOS active-param schedule／seed ensemble，逐交易日套用當日已生效參數，交易邏輯、資金、持股延續與0050基準全部不變；兩組及每個ensemble member只切換filter開關 |
| 主要指標 | 淨總報酬、最大回撤、報酬／最大回撤、年化報酬、Log R²、月勝率、交易數、候選／持股缺口與年度穩定性 |
| 判定用途 | 驗證9A分類排序改善是否轉化成策略經濟效果；不得再以同一OOS結果回頭修改模型或門檻 |
| 參數來源契約 | 正式OOS指定`models/roos_base_finalists_agree.json`或其他完整覆蓋期間的rolling paramset；最新`models/run_best_params.json`可能是static seed ensemble且看過後期資料，只能加`--allow-static-diagnostic`作非OOS敏感度診斷 |
| 後續 | 只有9A相較no-filter形成清楚的經濟改善，才另立8F高coverage策略比較；否則先凍結模型研究並累積新的forward labeled期間 |

實作契約修正（2026-07-26）：檢查正式呼叫鏈時發現第一版工具預設以`models/run_best_params.json`作歷史比較，一方面無法讀取目前Trade Mode可能產生的static active-param ensemble格式，另一方面最新run_best若回放2021～2025會把後期資訊帶回較早年度，不符合歷史active param無前視原則。工具已改為：正式模式要求明確指定Rolling OOS paramset並驗證完整覆蓋filter forward-OOS；支援rolling單一schedule及rolling seed ensemble，逐日沿用Portfolio Simulator既有active-param resolver；single／static run_best只在`--allow-static-diagnostic`下允許並明確標記。此修正只處理策略比較參數來源與研究設計，不改9A模型、Dataset、Label、score、threshold、optimizer objective、portfolio成交或帳務語意，真實投組結果仍為待執行。

| 追溯項目 | 本輪內容 |
|---|---|
| 程式基準 | 使用者ZIP `test-branch-1_20260726_133801_f785df1.zip`，SHA256 `aecf6f4e5f19fd6785f209339be388fb2632b0a7ca9b11bdb4fb2c08af9dfcd6`，再套用前一輪策略比較patch，SHA256 `b0485da5be9ff788ef04779386243c1c01fb83b8d24dffa97b2ec4b8ecb4ba02` |
| 唯一變更 | 策略比較的active-param來源契約：新增single／static ensemble／rolling schedule／rolling ensemble辨識，正式OOS強制rolling來源與期間覆蓋，static來源改為顯式diagnostic |
| 固定條件 | 9A architecture/profile、threshold 0.5、forward-OOS scores、Dataset、Label、交易核心、費稅、資金、持股、rotation與0050 benchmark均不變 |
| 重建需求 | Dataset、Label、feature bank、9A model與research scores皆不需重建；正式比較需先有canonical forward-OOS `scores.csv`與Rolling OOS `roos_*.json` |
| Selection／OOS結果 | 無新模型或分類結果；上一輪9A結果維持，真實no-filter vs quality-filter投組結果尚未執行 |
| 判定 | `IMPLEMENTED`；修正研究設計與格式相容性，不宣告策略有效或無效 |
| 下一步 | 本機先產生`models/roos_base_finalists_agree.json`，再以同一歷史daily active-param ensemble執行固定threshold策略比較 |

Formal bundle閉環（2026-07-26 15:13）：quick gate、chain checks與ML smoke均PASS；consistency回報2項FAIL，但兩列其實是同一份`doc/CMD.md`內兩種策略比較命令重複觸發相同契約：新正式入口`apps/breakout_quality_strategy_compare.py`已存在且`--help`正常，卻未加入quick-gate的集中`HELP_TARGETS`，所以文件契約判為「CMD有正式指令、help探針未覆蓋」。meta quality僅因synthetic suite非零退出而連帶FAIL，coverage本身68.42%並非本輪根因。已將該入口加入`HELP_TARGETS`與`INLINE_CLI_TARGETS`，使文件、正式CLI與快速help探針回到單一真理來源；同時將script存在／help覆蓋改為每個script只產生一組檢查，保留每條命令各自的參數契約，避免同一根因膨脹成多個FAIL。此修正不改策略比較邏輯、Rolling active-param契約、9A模型、Dataset、Label、score、threshold、optimizer、portfolio成交或帳務。

| 追溯項目 | Formal bundle閉環內容 |
|---|---|
| 程式基準 | 使用者ZIP `test-branch-1_20260726_151145_4f4d491.zip`，SHA256 `04407165f9c025a03fcc13b1cea6279267e72b5b63cf04daa63088fed4008a3d` |
| Bundle | `to_chatgpt_bundle_20260726_151302_4314b7e2.zip`，SHA256 `087f27e62bfb2e15615b1f896e54ec03d6c4d9aeb612e27636f2dc12e43491e2` |
| 唯一變更 | quick-gate help registry加入`apps/breakout_quality_strategy_compare.py`並納入inline CLI probe；CMD文件契約的script-level存在／help檢查改為每個script去重一次 |
| 固定條件 | 9A architecture/profile、threshold 0.5、Rolling參數來源、forward-OOS score、交易與統計口徑全部不變 |
| 重建需求 | 不需重建Dataset、Label、feature bank、模型、score或Rolling參數；只需重跑本地formal suite確認閉環 |
| Selection／OOS結果 | 無新模型、分類或投組結果 |
| 判定 | `IMPLEMENTED`；修正formal help coverage漏登，非模型或策略實驗 |
| 下一步 | 套用修補後重跑`python apps/test_suite.py`；通過後再進行Rolling OOS參數產生與固定9A策略比較 |

策略比較 runtime score lookup 閉環（2026-07-26 19:07）：正式 `forward_oos` score 已成功輸出640,981列，日期2021-01-04～2026-03-02；no-filter Rolling ensemble replay可完成，但quality-filter在建立2021～2026 active-param快取時，於ticker=2412、high_len=160、2025-06-12因精確`ticker/date/high_len`列不存在而中止。根因是9A `use_dataset_context=false` score export已按unique `ticker/date`只推論一次並精確broadcast到event rows，manifest也宣告`shared_group_score_broadcast=true`，但runtime仍沿用legacy event-context模型的三欄lookup key；因此score產生與score消費沒有完成同一group語意。已將runtime contract讀取該宣告：sequence-only工件先驗證同一`ticker/date`所有event-row分數完全一致，再建立唯一`ticker/date` index；active `high_len`仍必須落在artifact宣告coverage內，但不再要求該日期必須剛好保留同一high_len event row。使用Dataset context的legacy模型仍維持精確三欄lookup並fail-fast。此修正不改score值、threshold、模型、Dataset、Label、Rolling參數、候選定義、成交或帳務；既有`scores.csv`與manifest可直接重用。

| 追溯項目 | Runtime score lookup閉環內容 |
|---|---|
| 程式基準 | 使用者ZIP `test-branch-1_20260726_190737_6b7c70d.zip`，SHA256 `179a8dc5ae2a5636faea27c22f25e11b1437ba268595148210a7663fdd8a0cca` |
| 唯一變更 | sequence-only正式runtime lookup由`ticker/date/high_len`改為manifest宣告的`ticker/date` shared-group key；legacy context-dependent工件不變 |
| 固定條件 | 9A architecture/profile、threshold 0.5、forward-OOS score values、Rolling active-param ensemble、portfolio交易與統計口徑全部不變 |
| 重建需求 | 不需重建Dataset、Label、feature bank、模型、score或Rolling參數；套用patch後直接重跑strategy compare |
| Selection／OOS結果 | 無新分類或投組結果；本輪只完成runtime契約閉環，策略比較仍待完成 |
| 判定 | `IMPLEMENTED`；修正key語意分叉，不預判9A策略有效或無效 |
| 下一步 | 直接重跑`python apps/breakout_quality_strategy_compare.py --dataset full --params models/roos_base_finalists_agree.json --max-positions 10 --rotation off` |


正式 runtime 候選全集閉環（2026-07-26 19:26）：套用 shared `ticker/date` lookup 後，quality-filter 仍在 `2412 / 2025-06-12 / high_len=160` 報告整個 ticker/date 不存在，證明舊 `scores.csv` 並非只有 high_len key 分叉，而是 score producer 只輸出訓練 Dataset 中成功建立 300-bar 個股＋0050 sequence 的事件；Portfolio runtime 則對所有價格突破候選要求分數。reduced 資料可重現同型案例：`2330 / 2025-06-11 / high_len=70` 為 canonical crossover，但0050缺少同日資料，舊 Dataset直接略過。正式 forward export 已改為重新掃描目前 canonical source CSV，以與 `core.signal_utils` 相同的 crossover公式建立候選全集；可評分事件走固定9A模型，不可建立模型輸入者寫入`unavailable_scores.csv`並在canonical `scores.csv`固定為0.0保守REJECT，manifest保存來源inventory、候選／評分／保守拒絕數與原因。runtime驗證audit row必須存在於score table且精確為0.0；未被audit記錄的任何缺分仍fail-fast。此修正只閉合正式score覆蓋契約，不改9A模型、threshold、Dataset／Label、Rolling active params、候選 crossover、成交或帳務。

| 追溯項目 | Runtime候選全集閉環內容 |
|---|---|
| 程式基準 | 使用者ZIP `test-branch-1_20260726_192609_c92225a.zip`，SHA256 `9cfca2af009a6c9b2ad2a20e63fbd4fdb1c6d18131fe58b9dd14b075d90bc885` |
| 唯一變更 | `forward_oos`由stored labeled Dataset event subset改為current canonical runtime crossover universe；不可評分候選明確audit並固定0.0保守REJECT |
| 固定條件 | 9A architecture/profile、checkpoint、threshold 0.5、Label、research報表、Rolling OOS參數、策略candidate公式、Portfolio成交與統計口徑均不變 |
| 重建需求 | 不需重建Dataset、Label、feature bank、9A model或Rolling參數；**必須重新匯出forward-OOS scores與manifest** |
| Selection／OOS結果 | 無新模型或分類結果；本輪為runtime coverage契約修正，策略比較仍待完成 |
| 判定 | `IMPLEMENTED`；不可評分事件依專案保守原則固定REJECT，不宣告9A策略有效或無效 |
| 下一步 | 重跑`export-scores --scope forward_oos`，確認輸出`unavailable_scores.csv`及coverage counts，再重跑固定no-filter vs 9A策略比較 |


### 3.36 固定9A策略經濟效果結果與交易歸因工具（2026-07-26）

| 項目 | 紀錄 |
|---|---|
| 狀態 | 固定9A策略操作點：`RESULT_AVAILABLE / REJECTED_FOR_RUNTIME_DEPLOYMENT`；交易歸因：`RESULT_AVAILABLE` |
| 程式基準 | `test-branch-1_20260726_201327_1836e7d.zip`，SHA256 `6f4491d1ae2e20aab2ee4fa39d7e77ae14ef1f741e9afc4ef10f47cf39abc335` |
| 結果來源 | `strategy_comparison.md`，SHA256 `85550be389c7a33463192b49c3311ba35f4ff796083c845cec7adc83a8e9669e`；`trade_attribution.md`，SHA256 `a524719c726bf4f746febe4ae0f88efdb9c0730ae8deb6dd7215252ae8ca2967` |
| 比較設計 | 2021-01-04～2026-03-02；同一`roos_base_finalists_agree.json` Rolling active-param ensemble、相同資金／持股／rotation／0050／交易日期，只切換`use_breakout_quality_filter`；lookahead-safe=`True` |
| No-filter | 淨總報酬147.71%、MDD16.42%、RoMD9.00、年化19.23%、Log R²0.9180、月勝率63.49%、交易353、Payoff3.38、EV0.78R、平均曝險81.24% |
| 9A threshold 0.5 | 淨總報酬121.14%、MDD20.75%、RoMD5.84、年化16.63%、Log R²0.8474、月勝率66.67%、交易343、Payoff2.92、EV0.67R、平均曝險70.35% |
| 與基準差異 | 淨總報酬−26.56pp、MDD惡化+4.33pp、RoMD−3.16、年化−2.59pp、Payoff−0.45、EV−0.11R、平均曝險−10.89pp；候選供給不足日+128、持股缺口+280格日 |
| 年度結果 | 2021 −7.98pp、2022 −10.58pp、2023 +19.90pp、2024 +3.94pp、2025 −9.16pp、2026截至03-02 −0.89pp；2026為partial year，不得標示完整年度 |
| 交易歸因 | 共同交易67筆、No-filter only 286筆、Filter only 276筆；No-filter總R 275.42、Filter總R 228.79、差異−46.62R。共同交易R差異僅+0.15R，獨有交易選擇效果−46.77R；被排除贏家326.25R、避開輸家138.76R、替代贏家281.00R、替代輸家140.28R |
| 歸因判讀 | 9A失效主要來自交易選擇與投組路徑：No-filter only平均0.66R／Payoff3.21，Filter only平均0.51R／Payoff2.76，勝率與持有日幾乎相同，表示filter未改善命中率，反而削弱右尾大贏家。211筆No-filter only為threshold直接拒絕、75筆為路徑擠出，無score漏配為0 |
| 採用判定 | 9A分類Precision提升未轉化為策略效益；報酬、回撤、Payoff、EV、曝險、最差完整年度與曲線穩定性均惡化。保留9A作研究排序基準，但`threshold=0.5`不得啟用runtime gate |
| 本輪唯一實作變更 | 新增`trade_attribution.py`與既有比較入口的`--attribution-only`；以ticker＋實際進場日＋進場類型精確配對共同／no-filter only／filter only round trips，拆分被排除贏家、避開輸家、替代交易、直接門檻拒絕與投組路徑擠出；同時修正被回測終點截短的2026年度完整性 |
| 固定條件 | 不重訓、不relabel、不調threshold、不改9A checkpoint／score／Rolling參數／候選／成交／費稅／帳務；R與PnL沿用Portfolio Engine closed-trade真理來源 |
| 重建需求 | 不需重建Dataset、Label、feature bank、model、score或Rolling params；既有策略比較輸出仍在時只執行`python apps/breakout_quality_strategy_compare.py --attribution-only` |
| 下一步 | 歸因已完成；不得依同一OOS調門檻、calibration或regime開關，也不另跑8F策略比較。模型研究正式凍結；保留9A作排序研究基準，等待2026-03-03之後累積足夠且完成40交易日Label horizon的新forward labeled期間 |


### 3.37 9A Quality Score 候選排序探索性機制（2026-07-26）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED`；尚無真實策略結果，不得預先判定有效或無效 |
| 程式基準 | `test-branch-1_20260726_214442_75770f0.zip`，SHA256 `28fb45e6400a0241e6ee3e229f896c4541e203e96dcd9e5e8cc645f971812fcd` |
| 背景 | 9A threshold 0.5 hard gate 已由Rolling OOS策略結果淘汰，但9A固定coverage排序跨Selection／OOS相對穩定；使用者要求測試Score只作候選優先順序、不作生殺門檻 |
| 唯一變更 | Baseline固定`use_breakout_quality_filter=False / use_breakout_quality_ranking=False`；探索組固定hard filter=False，只將`use_breakout_quality_ranking=True`。Optimizer search space將ranking固定False，不參與參數搜尋 |
| 排序契約 | 先以既有`min_agree`決定候選資格；active-param ensemble排序固定為vote count由高到低、同票候選的median Quality Score由高到低、既有buy-sort、ticker deterministic。Score不得讓較少票候選超越較多票候選 |
| Score語意 | 正常、Continuation及STOP後Re-entry均使用原始breakout signal date的canonical runtime Score；正常可評分低分候選仍保留並往後排；`unavailable_scores.csv`中的不可評分事件維持保守排除，未知缺分fail-fast |
| 固定條件 | 9A checkpoint／scores、threshold欄位、Rolling active params、members／min_agree、資金、持股、rotation、交易成本、成交、停損／停利／trailing、0050與比較期間全部不變；threshold不作gate |
| 重建需求 | 不需重訓、relabel、重建Dataset／feature bank、重新匯出score或重跑optimizer；套用程式後直接執行隔離比較 |
| 輸出 | `strategy_compare_score_ranking/strategy_comparison.md/.json`及兩組equity／trades／daily-capacity／年度報酬；成交紀錄另保存進場Quality Score、Score日期及ensemble同意數供歸因 |
| 研究限制 | 此機制是在已查看2021～2026舊OOS後提出，只能作探索性診斷；即使改善，也不得直接作部署證據，必須等待全新forward period驗證 |
| 下一步 | 重新匯出含pre-execution signal anchor的正式scores後，再執行`python apps/breakout_quality_strategy_compare.py --comparison-mode score-ranking --dataset full --params models/roos_base_finalists_agree.json --max-positions 10 --rotation off`，取得真實結果後再將本節更新為`RESULT_AVAILABLE` |

Score-ranking OOS邊界閉環（2026-07-26 22:45；23:13更正）：第一次執行時，策略回放正確從2021-01-04開始，但候選`00633L`使用前一交易日2020-12-31的原始breakout signal Score；舊forward-OOS工件的`available_from=2021-01-04`，將Score事件涵蓋期誤與策略執行期綁定，因而fail-fast。22:45版曾將正式Score起點設為`model_information_cutoff`下一日，但`model_information_cutoff`實際是訓練標籤資訊最後使用到的交易日；模型可於該日收盤後完成，並使用同日收盤訊號建立下一交易日盤前訂單，因此下一日規則仍會漏掉2020-12-31。23:13版更正為`required_signal_start = model_information_cutoff`，正式Score包含cutoff當日事件，但strategy compare仍只從`execution_start`回放，不會把策略執行提前，也不得用執行日收盤資料替代前一日訊號。此修正不改模型、Label、Score公式、候選、排序、Rolling active params、成交或帳務；狀態為`IMPLEMENTED`，必須重新匯出forward-OOS scores，無須重訓或重跑optimizer。

| 追溯項目 | Score-ranking OOS邊界閉環內容 |
|---|---|
| 程式基準 | 使用者ZIP `test-branch-1_20260726_224528_e4be18f.zip`，SHA256 `6f4c066003ef6a7d0d44e00767c323fcb56d90045f01fc936c82ae1fcbd26d9b` |
| 唯一變更 | forward score signal coverage最終更正為由model cutoff當日開始；strategy compare仍固定由outer OOS execution start開始 |
| 固定條件 | 9A checkpoint／score定義、hard filter關閉、同票Score排序、Rolling ensemble、資金、持股、交易成本與0050不變 |
| 重建需求 | 不需重建Dataset、Label、feature bank、model或Rolling params；必須重新執行`export-scores --scope forward_oos` |
| Selection／OOS結果 | 無新結果；本輪只完成日期邊界契約閉環 |
| 判定 | `IMPLEMENTED`；不得預判Score ranking有效或無效 |
| 下一步 | 重匯scores後重跑score-ranking策略比較 |

### 3.38 Score-ranking Re-entry 原始事件 Score 繼承閉環（2026-07-27）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED`；修正 runtime 狀態傳遞，尚無新的策略比較結果 |
| 程式基準 | `test-branch-1_20260727_003139_6a2bb9b.zip`，SHA256 `ebae1e008354e2757291f30665644d519e896ae4bd9bf6eaf3689cf8be73197b` |
| 觸發案例 | Score-ranking replay 於 `2022-01-18` 建立候選時，對 `3706 / 2022-01-17 / high_len=150` 報「score table 缺少正式候選事件」 |
| 根因 | 該日期是STOP後Re-entry的重新站回確認日，不是原始breakout event。舊流程在建立Re-entry extended candidate時使用確認日 `signal_date`重新查正式score table，違反3.37既定的「Re-entry沿用原始breakout Score」契約；正式score table正確地不包含非breakout確認日 |
| 唯一變更 | 正常breakout查到的canonical rank payload會寫入continuation state與成交持倉；ensemble逐member保存各自原始rank；STOP後各member watch state、Re-entry signal state與再次聚合候選完整繼承原始`score/score_date/filter_id`。Re-entry確認日仍保存為新的交易`signal_date`，但ranking不得再用它查score table |
| 固定條件 | 9A model／Score公式與既有scores、hard filter關閉、vote count第一、同票Score排序、Rolling active params、members／min_agree、候選資格、成交、停損／停利／trailing、費稅、帳務與0050全部不變 |
| Dataset／Label／工件需求 | 不需重建Dataset、Label、feature bank、model、scores或Rolling params；套用程式後直接重新執行score-ranking策略比較 |
| 驗證 | Direct synthetic case證明Re-entry交易訊號日可為`2022-01-17`，但rank仍保留原始`score_date=2021-12-30`；並以mock禁止runtime lookup，確認Re-entry不再查確認日。另驗證ensemble candidate逐member保存原始rank mapping |
| Selection／OOS結果 | 無新模型或策略結果；本輪只修正已確認的runtime語意分叉 |
| 判定 | `IMPLEMENTED`；不得預判Score ranking有效或無效 |
| 下一步 | 不需重匯scores；直接重跑`python apps/breakout_quality_strategy_compare.py --comparison-mode score-ranking --dataset full --params models/roos_base_finalists_agree.json --max-positions 10 --rotation off` |

### 3.39 `base_finalists_agree` Quality Score 排序結果（2026-07-27）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / EXPLORATORY_POSITIVE / NOT_ACCEPTED_FOR_DEPLOYMENT` |
| 結果來源 | 使用者提供策略比較報表；期間2021-01-01～2026-03-02，參數為`roos_base_finalists_agree.json`，歷史active-param無前視=True |
| 唯一差異 | 兩組hard filter皆False，只切換`use_breakout_quality_ranking=False/True`；排序為finalist同意數第一、同票Score第二 |
| 主要結果 | 淨總報酬147.71%→152.58%（+4.88pp）、MDD16.42%→16.61%（惡化0.19pp）、RoMD9.00→9.18、年化19.23%→19.68%；但Log R²−0.0537、月勝率−4.76pp、勝率−2.31pp、EV−0.13R、平均曝險−23.90pp、最差完整年度惡化7.47pp |
| 年度結果 | 2021 −10.19pp、2022 −7.47pp、2023 +42.51pp、2024 −4.12pp、2025 −3.59pp、2026 partial −1.25pp；只有2023改善 |
| 判定 | 總報酬小幅正向但高度集中單一regime，跨年穩定性與一般交易品質惡化；不得部署，也不得依舊OOS繼續調整票數／Score混合權重 |
| 下一步 | 僅作一次較純的`base_finalist_best`單一member Score Ranking ablation，以移除finalist票數排序干擾；結果仍只屬探索性診斷 |

### 3.40 `base_finalist_best` 單一 member Score Ranking 隔離比較（2026-07-27）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / REJECTED / NOT_ACCEPTED_FOR_DEPLOYMENT` |
| 程式基準 | 結果基準`test-branch-1_20260727_104312_94c6a3a.zip`，SHA256 `be17cf6e15e6fecaaffb4d3fd96ae8d7f2c85bc02bec4858d4b1479920247937`；比較契約最初實作於`test-branch-1_20260727_081157_61ad01f(1).zip` |
| 研究目的 | 移除`finalists_agree`的同意數第一順位，直接驗證9A Score是否能在同一套單一Rolling active param下改善候選全域排序 |
| 唯一變更 | 同一`roos_base_best.json`建立baseline與ranking兩組；hard filter皆False，只切換`use_breakout_quality_ranking=False/True`；selector=`base_finalist_best`、每期1 member、`min_agree=1` |
| 主要結果 | 淨總報酬129.08%→96.96%（−32.12pp）、MDD17.41%→18.23%（惡化0.83pp）、RoMD7.42→5.32、年化17.43%→14.04%、Log R² 0.9436→0.8283、月勝率63.49%→58.73%、勝率41.52%→37.91%、Payoff2.95→2.86；EV雖0.53R→0.64R，但平均曝險84.02%→56.07%（−27.96pp） |
| 年度結果 | 2021 −19.37pp、2022 −10.13pp、2023 +28.88pp、2024 −0.34pp、2025 −6.36pp、2026 partial −1.11pp；只有2023改善。移除2023後，以年度報酬描述性連乘，baseline約+70.71%，ranking約+20.79% |
| 容量診斷 | 平均每日可掛單候選50.55→46.39、候選供給不足日+26；但期末未滿倉日745→445、持股缺口−643格日，同時平均曝險大降。持股檔數與資金曝險口徑不同：ranking更常持有較多檔，但依既有風險sizing配置的單檔金額較小。每日可掛單候選也會受既有持倉、當日賣出、Continuation／Re-entry狀態影響，屬投組路徑結果，不是純原始signal數 |
| 與3.39合併判讀 | Score作`finalists_agree`同票tie-break僅呈現2023集中型小幅正報酬；當移除票數保護、改成所有候選全域Score第一時，經濟效果顯著惡化。這表示9A分類Score不能直接解讀為最終Round-trip R或資本效率的單調排序；3.39的+4.88pp不能歸因為穩定Score優勢 |
| 採用判定 | `REJECTED`。不得切換正式selector至`base_finalist_best`，不得啟用Quality Score全域第一排序，也不得依既有OOS繼續調整Score權重、票數／Score混合公式、年度或regime開關。正式runtime維持`base_finalists_agree`既有排序且`use_breakout_quality_ranking=False` |
| Dataset／Label／工件需求 | 不需重建Dataset、Label、feature bank、model、scores或optimizer；本結果只更新實驗紀錄 |
| 下一步 | 舊OOS的hard gate、同票Score tie-break與全域Score第一排序研究全部凍結；等待2026-03-03之後累積足夠且完成40交易日Label horizon的新forward labeled期間再作一次預先固定設計的驗證 |

### 3.41 Breakout Event Regime Coverage Audit（2026-07-28）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE`；2022同時存在深度回撤regime支撐缺口與已見regime下的關係漂移，只作根因診斷，不形成runtime gate |
| 程式基準 | 工具基準`test-branch-1_20260728_232004_77fa495.zip`，SHA256 `ee3c13f0c5cb238f836ab90dded37a405343d25be8a16de7bdb6d842f1a9e296`；結果由使用者在完整本機Dataset執行提供 |
| 研究目的 | 驗證Selection中的熊市／深回撤breakout事件是否不足，並區分2022的排序失效是regime支撐缺口或相同regime下的關係漂移 |
| 固定條件 | 9A `inception_time_v1 / unique_group_sampling`、固定threshold 0.5、既有research score、Dataset、Label、split與模型全部不變；regime只使用事件日收盤及以前的0050 300-bar sequence |
| 年度結果 | 2022 Groups 2,016、原始PASS 40.13%、模型PASS 55.31%、Precision 43.68%、Recall 60.20%、PR-AUC 0.4582；Bear事件63.74%、深回撤事件36.01%、low-support 36.01%、Selection低代表性53.52% |
| 支撐缺口 | Deep drawdown在Selection僅24 groups、OOS有906 groups，Selection/OOS share ratio 0.019；Bear／deep-drawdown／high-volatility在Selection僅24 groups、OOS有618 groups，OOS PASS 38.67%、Precision 40.62%、PR-AUC 0.4755 |
| 關係漂移 | 一般Bear事件在Selection/OOS占比相近（9.16%／9.37%），但PASS由62.52%降至39.94%、OOS PR-AUC僅0.4986；Bear／correction／high-volatility在Selection已有1,467 groups，OOS仍只有40.73% PASS與0.5287 PR-AUC，證明不能只歸因於樣本數不足 |
| 高波動判讀 | High-volatility占比由Selection 33.32%升至OOS 64.63%，但整體OOS PR-AUC仍有0.6130；真正嚴重的是高波動與Bear／deep-drawdown條件疊加，而非高波動單獨必然失效 |
| 判定 | 支持「2022部分失效源於深度回撤Breakout訓練支撐不足」；同時存在conditional／concept shift。不得把所有Bear樣本粗略加權、依regime調threshold或直接建立熊市關閉開關 |
| 重建需求 | 無；此結果只更新研究判讀，不重建Dataset／Label／feature bank、不重訓或重匯score |
| 下一步 | 對2022輸出combined-regime的TP／FP／TN／FN與年度錯誤貢獻，並描述性比較排除low-support後的PR-AUC；若仍低，確認2022失效不只由支撐缺口造成 |

### 3.42 Focus-year Regime Attribution（2026-07-28）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE`；2022排序失效不能主要歸因於low-support，已確認更廣泛的conditional／concept shift |
| 程式基準 | `test-branch-1_20260728_235533_ee053dd.zip`，SHA256 `c2ec6c713ab4693cd54a1d26497aa01cb11f48b58f807a7b38b56d1dfe2cf29b`，套用 `breakout_quality_focus_year_regime_attribution_patch_20260728.zip`，SHA256 `5fe3ca1e0db8c21df5575e3dcd635739b3de7fa8edd8a5a5509099459b05c53f`；結果由使用者本機完整Dataset執行提供 |
| 唯一變更 | `regime-audit`新增`--focus-year`；輸出指定年度每個combined regime的Groups、Selection支撐、年內占比、TP／FP／TN／FN、年度FP／FN貢獻，以及完整年度、low-support only、排除low-support後三組描述性比較 |
| 介面 | 互動選單新增「市場狀態覆蓋與年度歸因稽核」；預設focus year為2022、min-selection-groups為100、min-support-share-ratio為0.5 |
| 固定條件 | 9A architecture/profile、checkpoint、research score、threshold 0.5、Dataset、Label、split、optimizer、runtime候選與交易帳務全部不變；排除比較只作歸因，不形成runtime gate |
| 完整2022 | Groups 2,016、原始PASS 40.13%、模型PASS 55.31%、Precision 43.68%、Recall 60.20%、Accuracy 52.88%、平均Score 0.4992、PR-AUC 0.4582；TP／FP／TN／FN＝487／628／579／322 |
| Low-support only | Groups 726（36.01%）、原始PASS 37.88%、模型PASS 64.74%、Precision 39.79%、Recall 68.00%、平均Score 0.5241、PR-AUC 0.4626；TP／FP／TN／FN＝187／283／168／88。此區只占36.01%事件，卻貢獻45.06%年度FP，屬明顯高風險區 |
| 排除Low-support後 | Groups 1,290、原始PASS 41.40%、模型PASS 50.00%、Precision 46.51%、Recall 56.18%、Accuracy 55.12%、平均Score 0.4852、PR-AUC 0.4722；相較完整年度Precision僅+2.83 pp、Recall−4.02 pp、PR-AUC僅+0.0140，排序仍接近失效 |
| 主要FP來源 | `bear / deep_drawdown / high`占29.07%事件、34.71% FP；`bear / correction / high`占26.44%事件、25.96% FP；`transition / deep_drawdown / high`占6.94%事件、10.35% FP。三者合計62.45%事件、71.02%年度FP |
| 已見regime失效 | `bear / correction / high`在Selection已有1,467 groups，2022仍只有38.09%原始PASS、42.40% Precision與0.4962 PR-AUC；`transition / correction / high`有535 Selection groups但PR-AUC 0.4267；`bear / near_high / high`有146 Selection groups但PR-AUC 0.4097。證明問題不只絕對樣本不足 |
| 判定 | Deep-drawdown low-support會放大2022錯誤放行，但不是主因；排除後PR-AUC仍只有0.4722，確認同一粗略regime內的K線結構→Label關係在2022發生廣泛漂移。不得以oversampling、年度／regime threshold、熊市關閉開關或只補少量deep-drawdown案例作為下一步 |
| 重建需求 | 無；此結果只更新研究判讀，不重建Dataset／Label／feature bank、不重訓或重匯score |
| 下一步 | 先做只讀的Market Breadth／Breakout Density Coverage Audit：在事件日以前計算全市場站上50／200日均線比例、20／60日正報酬比例、新高／breakout密度與產業廣度，優先檢查`bear / correction / high`等已有Selection支撐卻失效的regime。若同一0050 regime可被breadth再分出穩定Label差異，才進入單一breadth-context模型實驗；否則維持模型凍結並等待新forward period |

### 3.43 Active InceptionTime 可設定 Receptive Field（2026-07-29）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / REJECTED`；使用者回報600-window／RF600與300-window／RF251效果均不好，精確Selection／OOS數值未提供；policy退回300-window／RF229 |
| 程式基準 | `test-branch-1_20260729_002155_31e9709.zip`，SHA256 `eeceee667c98932ab4b0b1ca0e1e4da9734a2bff3d061c5eca0d513ed686500f`，再套用可設定receptive-field patch；結果為使用者定性回報，完整報表未提供 |
| 研究目的 | 讓active `inception_time_v1`可在較長feature window內建立足以覆蓋長期趨勢的有效receptive field，不再固定為229 bars |
| 唯一程式變更 | `config/breakout_quality_policy.py`新增`BREAKOUT_QUALITY_INCEPTION_DEPTH`、`BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS`、`BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY`；依target與depth自動生成三尺度正奇數kernels，model manifest保存實際kernels與receptive field |
| 預設相容性 | 預設depth=6、minimum target=228、residual_every=3，精確還原kernels 39／19／9與實際RF 229，故未調設定時模型結構不變 |
| Legacy相容性 | `inception_time_group_norm_v1`維持固定6層、39／19／9與RF 229，不受active設定影響，確保舊checkpoint／manifest重建 |
| 合法性限制 | target不得大於`BREAKOUT_QUALITY_FEATURE_WINDOW_BARS`；depth必須為正整數；residual interval必須整除depth；實際RF因奇數kernel向上取整可略高於target |
| 600-bars候選 | 設定`FEATURE_WINDOW_BARS=600`、`INCEPTION_TARGET_RECEPTIVE_FIELD_BARS=600`、depth=6時，自動產生kernels 101／51／25與實際RF 601 |
| 固定條件 | Label、horizon、optimizer、experiment profile、batch、patience、threshold、seed、normalization、filters、bottleneck、pooling全部不變 |
| 重建需求 | 只改target／depth但feature window不變：Dataset與Label可沿用，但必須重新訓練、重新匯出score與報表；若feature window由300改為600：必須完整重建feature bank／Dataset並重新訓練與匯出score |
| Selection／OOS結果 | 使用者回報RF600與RF251均不佳；因未提供完整報表，Precision／Recall／PR-AUC等精確值不得補寫 |
| 下一步 | 停止RF微調並維持300-window／RF229；下一個單一架構實驗改為加入全市場point-in-time learned set representation，不加入人工MA breadth、候選專屬query、sector或lag |


### 3.44 Stage 0／1 Learned Global Market Set Architecture（2026-07-29）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED`；已完成資料、模型、訓練、research inference與工件契約，尚未取得Selection／OOS結果 |
| 程式基準 | 初始來源ZIP `test-branch-1_20260729_155326_44f3825.zip`，SHA256 `090c08a219177188c462a60ef562fba5184d326949cd46eddb3efea8dcd44ac9`；使用者回報首輪交付只加入架構但policy仍指向9A，實際完整流程重現原9A結果。policy啟用後最新來源為 `test-branch-1_20260729_174337_e9a9834(1).zip`，SHA256 `00a09176e52ae108d86d3925a2f3f7b4c566d657f4d8919a158ea7f4f20a98ae`；本輪再修正 active market-set policy 下 synthetic CLI Dataset fixture 未建立 market-set artifacts 的 formal 測試契約問題 |
| 研究目的 | 在不使用MA50／MA200等人工breadth週期的前提下，讓模型由全市場股票過去300日的基礎OHLCV變化自行學習市場廣度、領漲群、弱勢尾部與分化，補足只看突破股＋0050無法觀察的橫斷面資訊 |
| Stage 0資料契約 | 新增共用point-in-time market bank：`date × ticker × 5 base features`與明確valid mask；特徵為close-to-close、overnight、intraday、high-low range與log-volume change，只使用當日及以前資料；同一天全部breakout事件共用同一份市場資料，不在每個event重複儲存 |
| Stage 1模型 | 新architecture `inception_time_market_set_v1`：候選主分支完整沿用9A 300-window／RF229 InceptionTime encoder；每檔市場股票通過同一套小型Shared Stock Temporal Encoder（GroupNorm，避免不同market batch／無效股票比例污染其他股票的正規化統計）形成32維embedding；4個Global Learned Queries以4-head attention做排列不變的set pooling，形成128維market embedding，再與候選128維表示融合分類 |
| 資源控制 | Training／evaluation／research export按unique market date重用market representation；每個實體batch最多包含`BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH=4`個不同日期，避免隨機event batch一次展開數百個全市場300日tensor |
| 未加入項目 | 不加入candidate-conditioned attention、sector tokens、ticker identity、learned lag relation、股票兩兩self-attention或人工MA／RSI／MACD；這些只能在Stage 1確認有穩定增量後逐項研究 |
| 固定條件 | Label、300-bar候選window、RF229、9A候選encoder、unique-group sampling、optimizer、LR、batch policy、patience、final refit、threshold 0.5與seed全部不變；唯一新增資訊為全市場learned set branch |
| Dataset／工件需求 | 切換architecture後必須完整重建Dataset以建立market-set artifacts，之後重新訓練、匯出research scores與報表；checkpoint、manifest與split assignment保存market-set contract與artifact hashes，避免沿用不一致工件 |
| Runtime邊界 | 目前只允許research workflow；`--scope forward_oos`對此architecture明確fail-fast，因Stage 0／1尚未建立scanner每日market-bank建置、版本與coverage契約，不得誤宣稱可部署 |
| Formal double-check閉環 | 使用者bundle顯示quick gate、chain checks、ML smoke均PASS；consistency兩項FAIL均來自generic Dataset refresh synthetic fixture仍只建立舊式Dataset，active architecture要求market-set artifacts後使unchanged／fast-relabel案例被誤判為rebuild，並連帶造成meta quality的`coverage_synthetic_suite_runs_successfully`失敗。正式runtime rebuild／relabel邏輯未發現分叉；fixture已改為依active model spec動態建立必要market-set contract與artifacts，不硬編碼policy值 |
| Selection／OOS結果 | 尚無有效market-set結果；使用者先前得到與9A完全相同的數值，是因policy未啟用新架構，該次結果只證明9A deterministic重現，不列為Stage 1結果 |
| 下一步 | 直接以目前policy執行完整Full workflow；啟動前必須看到filter id=`breakout_quality_v1_market_set_v1`、architecture=`inception_time_market_set_v1`與Market Set架構摘要。完成後比較9A與Stage 1的Selection／OOS PR-AUC、P@50／60／70、Recall、模型PASS與年度穩定性；未確認Global Market Set有一致增量前，不進入candidate query、sector或lag |


### 3.45 Stage 1 首輪結果與 Market Microbatch／Optimizer Batch 契約修正（2026-07-29）

| 項目 | 紀錄 |
|---|---|
| 狀態 | 首輪結果 `RESULT_AVAILABLE / INVALIDATED_AS_SINGLE_CHANGE`；batch契約修正 `IMPLEMENTED`，待重新訓練 |
| 程式基準 | 使用者ZIP `test-branch-1_20260729_175535_5404f83.zip`，SHA256 `d5d4c995dbfad5a774c5c55aa6aebbc5e1dd515afa7398780319820f1e677117`；結果由使用者本機完整Full workflow提供 |
| 首輪架構 | `inception_time_market_set_v1 / unique_group_sampling`；候選9A分支＋全市場Shared Stock Encoder＋4個Global Learned Queries；Dataset、Label、threshold與OOS split不變 |
| 首輪訓練 | 使用者執行log顯示patience=3（9A固定基準為1）；Best Epoch 1、最低Validation Loss 0.689330；完整Selection重訓1 epoch，但Final Refit實際為573 optimizer steps。9A基準為每epoch 181 steps、Best Epoch 2、Final Refit 362 steps。因此首輪同時混入patience與optimizer-step語意差異 |
| 首輪Selection | 原始PASS 54.81%、模型PASS 92.36%、PASS Precision 54.94%、Lift +0.13 pp、Recall 92.58%、Accuracy 54.32%、平均Score 0.5155、PR-AUC 0.5487 |
| 首輪OOS | 原始PASS 55.63%、模型PASS 88.26%、PASS Precision 55.01%、Lift −0.62 pp、Recall 87.27%、Accuracy 53.22%、平均Score 0.5137、PR-AUC 0.5077；固定coverage P@50／60／70為50.12／51.67／53.10% |
| 年度結果 | PR-AUC：2021 0.4752、2022 0.3579、2023 0.5413、2024 0.5379、2025 0.5731；沒有改善2022，且所有年度均低於9A |
| 直接判讀 | Score集中在約0.51、模型幾乎全部判PASS，Selection與OOS排序接近隨機；此首輪模型不可採用，也不得進入candidate query／sector／lag |
| 發現的契約問題 | `BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH=4`原意只限制全市場tensor的記憶體切片，但舊實作把每個date-limited實體batch直接當成一次optimizer update，使Final每epoch由181膨脹為573 steps；同時9A BatchNorm每次只看到少數日期的高度相關事件。此變更違反「batch=128 unique groups、optimizer與sampling固定」的單一實驗設計 |
| 修正 | 訓練現在先形成最多4個日期的market microbatches，再將多個microbatches封裝為最多128 events的一個optimizer logical batch；候選分支對完整logical batch一次forward，使BatchNorm看到跨microbatch事件；market分支逐microbatch forward後再融合；loss denominator、gradient clip、LR schedule與optimizer.step均以logical batch為單位 |
| 修正後預期契約 | Inner每epoch optimizer steps應回到`ceil(16,832/128)=132`；Final每epoch應回到`ceil(23,072/128)=181`。`max_dates_per_batch`只能改記憶體峰值與速度，不得改optimizer updates、sample weight或epoch語意 |
| 重建需求 | Market Set Dataset／Label／feature bank可沿用；必須刪除或覆蓋首輪`inception_time_market_set_v1` checkpoint、重新訓練、重新匯出research scores與報表 |
| 下一步 | 只重跑修正後的同一Stage 1實驗。啟動後先核對Final Refit每epoch為181 steps；若修正後Selection／Validation PR-AUC仍明顯低於9A，直接淘汰Global Market Set v1，不再調LR、query數或attention heads |


### 3.46 Stage 1 Logical-batch修正後完整結果（2026-07-29）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / REJECTED`；Global Market Set v1不採用；candidate-conditioned query後續改列為本質不同的獨立實驗，sector tokens與learned lag仍不進入 |
| 程式基準 | 使用者ZIP `test-branch-1_20260729_185613_cddc3ec.zip`，SHA256 `54eec16aaa01f2683069577e148e6edc0c8f0cb20d887f07ab12330509e251ce`；結果來源為使用者本機完整workflow log `已貼上文字 (1)(20).txt` |
| 契約確認 | policy正確啟用`inception_time_market_set_v1 / unique_group_sampling`；patience=1；Best Epoch=2；Final Refit=362 steps，精確等於2 × ceil(23,072/128)，證明market microbatch只控制記憶體，optimizer logical batch已恢復128 groups |
| 唯一變更 | 相對9A只新增point-in-time全市場Shared Stock Encoder、4個Global Learned Queries與candidate／market fusion；Label、300-window、RF229、候選InceptionTime、optimizer、LR、batch、patience、final refit、threshold與seed固定 |
| 修正後Selection | 原始PASS 54.81%、模型PASS 70.38%、PASS Precision 59.19%、Lift +4.38 pp、Recall 76.01%、Accuracy 58.13%、平均Score 0.5318、PR-AUC 0.6087；相較9A Selection PR-AUC 0.6443下降0.0356 |
| 修正後OOS | 原始PASS 55.63%、模型PASS 58.80%、PASS Precision 59.88%、Lift +4.25 pp、Recall 63.30%、Accuracy 55.99%、平均Score 0.5202、PR-AUC 0.6092；P@50／60／70=60.30／59.81／59.33%，R@P60=60.24% |
| 與9A差異 | OOS PR-AUC −0.0165；P@50／60／70分別 −2.47／−1.81／−0.92 pp；R@P60 −17.03 pp；PASS Precision −3.11 pp；Lift −3.10 pp。Recall +11.86 pp與模型PASS +13.37 pp只代表較寬鬆coverage，沒有形成更強排序 |
| 校準 | Brier由9A 0.2453微降至0.2447，ECE由0.0722降至0.0374；但校準改善不能補償固定coverage排序與Precision全面下降 |
| 年度結果 | PR-AUC：2021 0.6151、2022 0.4526、2023 0.6515、2024 0.6506、2025 0.6578；相較9A每一年均下降，差異依序−0.0007、−0.0056、−0.0030、−0.0106、−0.0300。2022未改善 |
| 判定 | 修正後單一變更仍明顯低於9A Selection／Validation／OOS排序；全域市場摘要沒有提供可泛化增量，且較高Recall來自放行更多候選。依預先固定規則直接淘汰，不調LR、query數、heads、embedding或fusion容量 |
| 採用／退回 | policy退回`inception_time_v1 / breakout_quality_v1`；`inception_time_market_set_v1`移至legacy read-only，只供舊Dataset、checkpoint、manifest與research result重現；不刪除既有market-set工件 |
| Formal double-check閉環 | 使用者於退回9A／Market Set轉legacy後執行formal suite：quick gate、chain checks與ML smoke均PASS；consistency唯一FAIL為synthetic policy SSOT的預期legacy集合漏列`inception_time_market_set_v1`，正式`ACTIVE_MODEL_ARCHITECTURES`／`LEGACY_MODEL_ARCHITECTURES`實作正確。測試fixture已補入該legacy architecture，並連帶解除meta quality的`coverage_synthetic_suite_runs_successfully`失敗 |
| 下一步 | Global Market Set v1維持淘汰，但Candidate-conditioned Query可視為本質不同的獨立架構實驗；若執行，Train／Validation與epoch選擇只用Selection，模型凍結後再用既有2021～2025 OOS與9A固定比較。不得因OOS已被查看而停止研究，也不得把等待新forward資料列為主要下一步 |


### 3.47 OOS研究治理與行動原則更新（2026-07-29）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `ACCEPTED`；專案長期研究規則更新 |
| 程式基準 | `test-branch-1_20260729_201632_cad0d3b.zip`，SHA256 `7c0e1b0926b52bccd7dd75da81d5f4e42888903516f31853ad3e9553d5a2c7ba` |
| 唯一變更 | 只更新 `doc/PROJECT_SETTINGS.md` 與本實驗紀錄，不改模型、Dataset、Label、score、runtime或測試程式 |
| OOS規則 | OOS可持續用於比較、診斷、接受／淘汰與形成下一個實驗；Train／Validation、loss、gradient、early stopping、epoch、threshold／calibration、normalization、feature／label、sample weighting與hyperparameter optimization不得使用OOS |
| 證據標記 | 同一OOS多次使用時標記為「迭代研究OOS證據」，不宣稱untouched holdout，但不因此停止研究 |
| 行動規則 | 後續必須提出可立即執行的實驗、實作、診斷或修正；不得再以等待新forward資料、凍結研究或無所作為作為主要建議 |
| Dataset／Label／重訓 | 不需要 |
| 下一步 | Candidate-conditioned Query列為下一個獨立單一變更實驗；只用Selection完成訓練／Validation，模型凍結後用固定OOS比較9A |


### 3.48 10A Candidate-conditioned Query 完整結果（2026-07-29）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / REJECTED`；policy已退回9A `inception_time_v1`，10A轉為legacy read-only，不啟動10B learned lag |
| 程式基準 | 結果ZIP `test-branch-1_20260729_211214_690584f.zip`，SHA256 `639bcd20d1cab58015cabd5fe92ed9cd180b65528a281f24763cfb057233fab9`；完整workflow結果 `已貼上文字 (1)(21).txt`；本輪closure patch為`patch_10a_result_revert_9a_20260729.zip`，因ZIP內文件不可自指涉自身穩定SHA256，交付SHA256於回覆列示 |
| 唯一模型變更 | 保留9A Candidate InceptionTime與Stage 1 Shared Stock Encoder；由每個candidate embedding產生1個query，對同日全市場stock embeddings做masked cross-attention，形成candidate-specific market embedding |
| 固定條件 | 300 bars、RF229、固定百分比Label、Selection內Inner Train／Validation、Adam 0.0003、batch 128 unique groups、patience 1、selected_epochs、threshold 0.5、seed 42、class/time weight none；OOS未參與訓練或epoch選擇 |
| 訓練 | Best Epoch 3，最低Validation loss 0.681544；完整Selection重訓3 epochs，Final loss 0.678506；527,074個可訓練參數 |
| Selection | 原始PASS 54.81%、模型PASS 60.22%、Precision 61.08%、Lift +6.27 pp、Recall 67.12%、Accuracy 58.54%、Score 0.5092、PR-AUC 0.6191；P@50／60／70%=62.11／61.12／59.81%，R@P60%=75.17% |
| OOS | 原始PASS 55.63%、模型PASS 52.77%、Precision 61.92%、Lift +6.29 pp、Recall 58.74%、Accuracy 56.95%、Score 0.4974、PR-AUC 0.6009；P@50／60／70%=62.02／61.21／59.78%，R@P60%=74.58%、Brier 0.2458、ECE 0.0622 |
| 相較9A | OOS PR-AUC −0.0248；P@50／60／70%分別 −0.75／−0.41／−0.47 pp；R@P60% −2.69 pp；threshold 0.5 Precision −1.07 pp、Lift −1.06 pp，但Recall +7.30 pp、模型PASS +7.34 pp、Accuracy +0.78 pp、Score +0.0132。較高coverage沒有換得更強排序 |
| 年度診斷 | 2022 PR-AUC 0.4531、Precision 43.05%，仍未修復既有熊市／深回撤concept shift；2023～2025較高Precision不能抵銷完整OOS排序低於9A |
| 校準判讀 | ECE由9A 0.0722改善至0.0622，但Brier略惡化0.0005，且PR-AUC與全部固定coverage Precision下降；校準改善不足以構成採用理由 |
| 採用判定 | `REJECTED`。Candidate-conditioned Query未證明全市場股票關係提供增量；不得以Recall上升解讀為成功，不做策略回測，不調query數、heads、embedding、LR或Market Set pooling |
| Dataset／Label／工件 | 不需重建或刪除任何Dataset、Label、Market Bank或10A工件；它們保留於`breakout_quality_v1_market_set_v1/inception_time_market_set_candidate_v1/unique_group_sampling`供歷史重現。正式policy退回既有9A，不需重訓或重匯9A scores |
| 下一步 | 不啟動10B Candidate-conditioned Learned Lag；進入11A Strategy-aligned Continuous Outcome Target，先以既有future-path cache建立連續target分布與同日排序可學性稽核，再決定正式訓練contract |


### 3.49 11A Strategy-aligned Continuous Outcome Target 可學性稽核（2026-07-29）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / DISTRIBUTION_PASS / TRADE_R_PRIMARY_PASS`；完整Dataset target arrays、同日排序與實際Round-trip R方向診斷均已完成。11A資料目標正式結案，可進入獨立11B研究訓練；此狀態不代表11B模型已有效 |
| 程式基準 | 11A最終結果以本輪唯一來源ZIP `test-branch-1_20260730_134655_6e073b2.zip`，SHA256 `ccd1083e852cf8a085f749dd4c1be61c006ecb1bf7f3a1c2990b8558ab6342d4`及使用者提供的`continuous_target_audit(2).md`為準；正式policy維持9A。完整audit命令為`python apps/breakout_quality.py audit-continuous-target` |
| 研究目的 | 9A binary PASS／REJECT score無法單調對應最終Round-trip R或資本效率；11A先測試連續future outcome是否具有足夠分布、同日非tie排序與實際R方向一致性，再決定是否值得建立regression／ranking模型 |
| 固定target | `strategy_aligned_opportunity_r_v1`：首次觸及−10%風險障礙前的最大有利漲幅÷10%風險預算，減去到達該高點前的最大不利跌幅÷10%，再減`0.5R × (opportunity_bar−1)/(40−1)`；最大高點取最早出現日 |
| 保守路徑語意 | 同日High／Low先後未知採adverse-first；若Low先觸及−10%障礙，該日High不列入可用機會。首日即觸及風險障礙時target固定為−1R。未滿40根或K線無效標記invalid，不建立第三種label |
| 無前視／防調參 | target公式只讀既有event anchor與固定future high／low cache；risk budget、horizon與time penalty由目前固定Label policy直接推導。不得讀取Selection／Validation／OOS分布、實際交易R、threshold或模型分數；不做normalization與clipping |
| 稽核內容 | 對Inner Train、Validation、Selection、OOS固定切分輸出percentiles、正值率、unique率、極端正值集中度、binary-label AUC、與MFE／MAE關係；逐日計算可排序日期比例、pairwise non-tie率、候選數與target spread，以及同日PASS對REJECT concordance |
| 實際R診斷 | 在active 9A正式模型輸出樹依序搜尋hard-filter `strategy_compare`、`base_finalists_agree`／`base_finalist_best` score-ranking與其他`strategy_compare*`目錄；有metadata時只接受目前filter／architecture／profile及`historical_active_param_oos`工件。每個目錄先讀`no_filter_round_trips.csv`，若只有`no_filter_trades.csv`則以既有canonical `reconstruct_round_trips`在記憶體重建；亦可顯式使用`--round-trips`或`--trade-history`。再描述性計算target與realized R Spearman、target top／bottom decile平均R與≥2R贏家保留率；實際R不得回頭改公式、normalization、clip或任何參數 |
| 工件 | `outputs/filters/breakout_quality/<filter_id>/continuous_targets/strategy_aligned_opportunity_r_v1/`；包含6個group arrays、`manifest.json`、`continuous_target_audit.json/.md`、`continuous_target_daily_rankability.csv`與可選trade matches CSV |
| Dataset／重建 | 不重建feature bank、不重算candidate features、不relabel；直接沿用canonical `group_anchor_prices`、future high／low path cache、available bars、event group index與既有split policy。target contract與工件獨立於model architecture／experiment profile |
| Formal契約 | B171／T268 direct synthetic contract共16項：除固定公式、adverse-first、首日−1R、時間懲罰、invalid future、deterministic group arrays、strict JSON、同日rankability、audit-only與CLI註冊外，驗證hard-filter標準Round-trip路徑、canonical `no_filter_trades.csv`重建、跨正式score-ranking輸出發現且跳過static diagnostic，以及顯式`--trade-history` override |
| Formal double-check閉環 | 使用者於2026-07-30以結果ZIP `test-branch-1_20260730_125841_23286ca.zip`（SHA256 `6f7cdcd443966148d16dfca495036bfe9583c5684ca6e39b19832920d6facc59`）執行正式suite；quick gate／chain checks／ML smoke PASS，consistency有3項FAIL：legacy預期集合漏列10A Candidate Query、11A synthetic由`tools/`反向import `apps.breakout_quality`、registry layer誤寫未允許的`research_contract`；meta quality只由synthetic suite失敗連帶觸發。已補legacy fixture、改以AST靜態解析CLI registry、將11A validator歸入`core_invariant`，並將10A set-invariance probe改為確定性candidate embedding差異與跨裝置浮點容差；不改target公式、Dataset、模型或runtime |
| 完整資料結果 | groups=55,509、valid=54,419（98.04%）。Inner／Validation／Selection／OOS mean=0.7394／0.9111／0.7839／0.9815；P50=0.3111／0.3319／0.3167／0.3201；P99=6.5133／8.3370／6.8589／9.9519；正值率69.49%／70.95%／69.89%／68.89%；Unique率98.73%／99.55%／98.73%／99.29%；所有區段同日可排序日與pair非Tie率均100%。Top 1%正target貢獻Selection 11.64%、OOS 10.22%，沒有單一極端尾端主導。Binary AUC約0.9705～0.9718、同日PASS／REJECT concordance約0.9697～0.9724，表示target高度接近既有二元Label的連續化；但actual-R診斷證明連續大小仍含有交易價值排序訊息 |
| 實際Round-trip R結果 | no-filter round trips共435筆，成功配對424筆，coverage 97.47%；Spearman(target, realized R)=0.4044。Target top decile實際平均2.2060R，bottom decile 0.0942R；≥2R大贏家有78.85%位於target上半部。這些統計只作方向驗證，不參與target公式、normalization、clipping、loss或任何參數擬合 |
| 下一步 | 進入11B `strategy_aligned_daily_percentile_mse`：保留9A網路與2-logit checkpoint結構，以PASS softmax probability回歸Selection內每個日期的11A target percentile；只用Validation mean daily Spearman選epoch，OOS在完整Selection重訓與checkpoint寫入後才評估 |


### 3.50 11B Strategy-aligned Daily Percentile Regression（2026-07-30）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / REJECTED`；模型對11A target有弱排序能力，但未轉化為actual Round-trip R，禁止進入runtime或策略Score ranking |
| 程式基準 | 結果ZIP `test-branch-1_20260730_173148_433ed48.zip`，SHA256 `5cfed17414beb326b4fa172a32cd851404511a3fcac8661d68ad335547cd78eb`；完整結果來源為`continuous_ranker_report.md/.json` |
| 固定條件 | 9A `inception_time_v1`、300×10、RF229、2-logit head、unique ticker/date sampling、Adam 0.0003、batch128、patience1、selected_epochs、seed42；Selection內Inner Train／Validation選epoch，完整Selection重訓後才評估OOS |
| 訓練結果 | Epoch 2以Validation mean daily Spearman 0.1934入選；完整Selection重訓2 epochs。Final refit後Inner／Validation／Selection／OOS mean daily Spearman為0.2074／0.1704／0.1963／0.1327，global raw-target Spearman為0.2237／0.1747／0.2101／0.1677 |
| OOS target排序 | Pair concordance 0.5387；Score top decile的11A raw target平均1.5822，bottom decile 0.2872，證明模型學到部分target方向，但強度有限 |
| 相較9A分類排序 | OOS PR-AUC 0.5959，較9A 0.6257低0.0298；P@50／60／70%=60.41／59.49／58.66%，較9A低2.36／2.13／1.59 pp。沒有保留9A的固定coverage排序優勢 |
| Actual Round-trip R | 435筆交易中422筆配對，coverage 97.01%；Spearman(model score, realized R)=−0.0003。Score top decile平均0.6500R，bottom decile2.0251R；≥2R大贏家只有55.77%位於score上半部，接近隨機且decile方向反轉 |
| 與11A target差異 | 11A target本身在近似同一交易集合的Spearman為0.4044、top／bottom decile為2.2060R／0.0942R、≥2R贏家上半部保留78.85%；11B學習後幾乎完全遺失這個經濟排序訊號 |
| 判定 | `REJECTED`。不能因OOS target Spearman為正就忽略actual R失效；不做策略回測、不匯出runtime scores、不調MSE／Huber、epoch、patience、LR、batch、percentile轉換或再加入pairwise loss |
| Dataset／工件 | 不重建Dataset、不relabel、不刪除11A arrays或11B checkpoint／scores／report；11B維持research-only歷史重現，正式9A工件與forward-OOS scores不變 |
| 下一步 | 先做11C Qualified Candidate-set Coverage Audit，確認失效是否來自全事件訓練母體與策略實際候選集合不一致；在該audit完成前不建立11C模型 |


### 3.51 11C Qualified Candidate-set Coverage Audit（2026-07-30）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / QUALIFIED_SAMPLING_NOT_SUPPORTED`；母體與交易子集只讀歸因完成，11B失效不能歸因為qualified／orderable候選層的Score↔Target崩落 |
| 程式基準 | 結果ZIP `test-branch-1_20260730_183212_cf1f2cd.zip`，SHA256 `f915767cd5c49397987d411c3b3bcbd1d1c1b6a8d1a96004f5c4ee74a80fd5ab`；正式結果來源為`qualified_candidate_set_audit.json`，執行時間180.98秒 |
| 母體覆蓋 | 全OOS 17,346 unique groups；qualified 4,971（占全OOS 28.66%）；orderable 4,957，為qualified的99.03%；actual trades成功配對422／435，且422筆全部屬於qualified及orderable集合 |
| Score↔Target | Global Spearman由all OOS 0.1677升至qualified 0.1918、orderable 0.1891；pair concordance由0.5387升至0.5574／0.5573，Score top-bottom target spread由1.2950R擴至1.6516R／1.6305R。Mean daily Spearman雖由0.1327降至0.1096／0.1077，但沒有母體崩落 |
| Actual trades | Target↔realized R=0.4015，Target top／bottom decile實際R=2.1737R／0.0942R，≥2R贏家78.85%位於Target上半部；Score↔Target=0.2459，但Score↔realized R仍為−0.0003，Score top／bottom decile實際R仍反轉為0.6500R／2.0251R |
| 判定 | 不建立`qualified_candidate_set_sampling` profile。模型進入qualified／orderable甚至actual-trade集合後仍保留或提高Score↔Target，因此全事件訓練母體不是11B失效主因；問題是Score捕捉到的Target變異並非與realized R相關的Target變異 |
| OOS邊界 | 本結果只作凍結11B模型的迭代研究歸因；不改target、loss、sampling、weight、epoch、threshold、normalization或runtime contract |
| Dataset／工件 | 不重建Dataset、不relabel、不重訓9A／11B；保留11C四層candidate與actual trade工件供後續成分歸因 |
| 下一步 | 進入11D Label-conditional Target Component Attribution Audit：將11A拆為favorable R、adverse R與time penalty R，並在qualified與actual trades分別計算Score及realized R關係；同時依PASS／REJECT分層，判定11A的0.4015是否只來自二元Label分離 |

### 3.52 11D Label-conditional Target Component Attribution Audit（2026-07-30）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / TIME_PENALTY_DIRECTION_MISMATCH`；PASS內連續Target仍含actual-R排序資訊，11A經濟關係不是只由PASS／REJECT分離造成；但原time penalty與actual R方向相反，禁止直接建立conditional magnitude模型 |
| 程式基準 | 結果ZIP `test-branch-1_20260730_185755_152e019.zip`，SHA256 `c1a2b1285f3fc8999b6969a40b453f3303306d240ebb0224a9eb1ed89ba678cc`；正式結果來源為`target_component_attribution_audit.md`，由使用者本機既有11A／11B／11C工件執行 |
| Qualified成分 | 4,971 rows；Score↔Target 0.1918、Score↔favorable R 0.1854、Score↔adverse R 0.1109、Score↔time penalty R −0.1610。模型同時偏好較高favorable與較快opportunity，但對adverse幅度的方向並不理想 |
| Actual trades | 422 rows；Target↔R 0.4015、Score↔R −0.0003。Favorable R↔R 0.4998為最強固定成分，time penalty R↔R亦為+0.4454；adverse R↔R僅−0.0515，接近無關 |
| PASS條件 | 214 rows；Target↔R 0.3600、Score↔Target 0.3500，但Score↔R −0.1011。Favorable↔R 0.4470、time penalty↔R 0.4734、adverse↔R 0.0313。證明PASS內仍有連續幅度資訊，但11B學到的Target方向沒有轉化為交易R |
| REJECT條件 | 208 rows；Target↔R −0.1915、Score↔Target 0.1269、Score↔R −0.0252；favorable↔R 0.1595、time penalty↔R 0.3432、adverse↔R 0.0858。原Target在REJECT內方向不穩，不支持全Label共用magnitude head |
| 主要錯位 | 11A公式固定扣除time penalty，但actual trades中較大的time penalty與較高R正相關；同時11B Score在qualified中與time penalty為負相關。也就是模型偏好較快機會，而實際策略大贏家反而常需要較長時間發展，形成Target學習與交易R的方向抵銷 |
| 判定 | 不直接建立PASS-conditional magnitude head，不調time penalty係數、不反向加分、不重跑11B。下一步只允許單一固定消融：移除time penalty，保持favorable與adverse原單位，先驗證No-time Target的actual-R方向與PASS條件穩定性 |
| Dataset／runtime | 不重建Dataset、不relabel、不重訓9A／11B；11D工件保留research-only，不建立runtime score或策略回測 |
| 下一步 | 進入11E Fixed Time-penalty Ablation Audit：固定`target_no_time_r=favorable_r-adverse_r`，比較原Target與No-time Target在qualified、actual trades及PASS／REJECT條件下的Spearman與decile spread；本輪仍不建立新target version或模型 |

### 3.53 11E Fixed Time-penalty Ablation Audit（2026-07-30）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / PASSED_FIXED_ABLATION`；overall、PASS與decile spread三個固定門檻均通過，只授權建立新version arrays與Selection-only稽核，不授權模型訓練 |
| 程式基準 | 結果由使用者在`test-branch-1_20260730_192246_43be284.zip`前一版本執行`audit-target-time-ablation`取得；本輪最新程式基準SHA256為`ef88d0461f56322a2c6f5eb6243e46ae1dafe644ae9196398d16b0523461a7b8` |
| 唯一變更 | 固定`target_no_time_r=favorable_r-adverse_r`；只移除原11A time penalty，不反向加分、不搜尋係數、不改risk budget、horizon、favorable或adverse定義 |
| Qualified結果 | 4,971 rows；Score↔原Target 0.1918、Score↔No-time Target 0.1466。既有11B Score較不貼近新Target，不能沿用11B模型作有效性證據 |
| Actual結果 | 422 trades；Target↔R由0.4015升至0.4875，Δρ=+0.0859；top-bottom decile由2.0795R擴大至2.6824R，spread增加0.6029R |
| PASS／REJECT | PASS內由0.3600升至0.4542（+0.0942）；REJECT由−0.1915升至0.0675。改善不只出現在REJECT，符合固定放行條件 |
| 判定 | 固定移除time penalty有效；不搜尋反向係數或其他權重。下一步只建立`strategy_aligned_opportunity_no_time_r_v1` arrays並做Selection-only可學性稽核 |
| OOS／runtime邊界 | 本結果是迭代研究OOS證據，只形成下一個固定假設；不建立loss、sampling、epoch、threshold、normalization、runtime score或策略回測 |
| 執行 | `python apps/breakout_quality.py audit-target-time-ablation --filter-id breakout_quality_v1` |

### 3.54 11F No-time Target Arrays＋Selection-only Learnability Audit（2026-07-30）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / SELECTION_LEARNABILITY_PASS`；新Target在Inner Train／Validation／Selection均維持高unique、同日rankability與非tie，允許進入固定PASS-conditional訓練；本輪本身未評估OOS或訓練模型 |
| 程式基準 | 結果ZIP `test-branch-1_20260730_194008_2053d47.zip`，SHA256 `d0e8e93fbf11e115fac111ce3ef6c5cee0e1161ade963681b8240100205596f6`；正式結果來源為`continuous_target_audit(4).md` |
| Target版本 | `strategy_aligned_opportunity_no_time_r_v1`；固定`target_raw_r=favorable_return/risk_budget-adverse_return/risk_budget`，沿用11A component arrays、valid mask、peak、risk及adverse-first語意 |
| 分布 | Inner Train／Validation／Selection mean為0.9440／1.1389／0.9945R，P50為0.4900／0.5652／0.5080R，P99為6.8683／8.8051／7.3047R；Validation右尾較高但與11A同方向，沒有分布失控 |
| 可排序性 | Unique率97.56%／98.90%／97.34%，同日rankable date均100%，pair非tie均99.99%；Selection內具備連續排序Target的技術條件 |
| 與二元Label | Binary AUC 0.9906／0.9920／0.9910，同日PASS／REJECT concordance約0.991，顯示全Label訓練會幾乎只學二元分離；不得直接重跑11B全Label percentile regression |
| 與11A關係 | 與11A Target Spearman 0.9707／0.9724／0.9711；固定移除time後仍保留主要排序結構，但11E已證明actual-R方向顯著改善 |
| 判定 | 允許11G，但training label scope必須固定為PASS-only，避免0.99 Binary AUC支配loss；Validation只以PASS-only mean daily Spearman選epoch，OOS在checkpoint寫入後才評估 |
| OOS／runtime | 11F沒有建立OOS指標、不讀actual R或11B score；新Target仍為research-only，不覆蓋11A或9A |

### 3.55 11G PASS-conditional No-time Magnitude Ranker（2026-07-30）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / REJECTED`；PASS-only No-time Target在OOS可學，但actual PASS realized R反向，模型不得採用或微調 |
| 程式基準 | 結果ZIP `test-branch-1_20260730_202010_4cc9d53.zip`，SHA256 `89769903bfe6abe41af8dead26c926a5e2123825115c39f035e71e3364dc55bf`；正式結果來源為`continuous_ranker_report(1).md` |
| Experiment profile | `strategy_aligned_no_time_pass_magnitude_mse`；active 9A `inception_time_v1`與既有2-logit head，training label scope=`pass_only` |
| Epoch選擇 | Best epoch 2；選epoch當下Validation PASS-only mean daily Spearman 0.2485。完整Selection PASS groups依2 epochs重新初始化重訓 |
| OOS PASS-only | 9,650 groups；global Spearman 0.3464、mean daily Spearman 0.2944、pair concordance 0.6350；Score top／bottom decile No-time Target為3.7433／1.1282，模型確實學到Target排序 |
| Actual PASS trades | 214筆；Score↔Target 0.3476、Target↔R 0.4542，但Score↔R −0.1066；Score top／bottom decile實際R為0.6117／3.5519，方向明顯反轉 |
| Overall／REJECT | Overall 422筆Score↔R −0.0012、top／bottom decile 0.2756／1.4585R；REJECT Score↔R −0.0475，只作診斷 |
| 判定 | 淘汰11G。失敗不是模型無法學No-time Target，而是模型學到的Target成分沒有轉化成策略可實現R；不得調超參數、loss、head或融合 |
| 下一步 | 進入11H PASS-only Realization-gap Attribution，固定檢查Score↔Favorable／Adverse、`Target−R` gap、`R÷Favorable` capture ratio與控制Target後partial Score↔R |
| Runtime | research-only工件保留供重現；不設定threshold、不匯出forward-OOS runtime scores、不覆蓋9A |

### 3.56 11H PASS-only Realization-gap Attribution Audit（2026-07-30）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / STRATEGY_REALIZATION_GAP_CONFIRMED`；11G高Score確實偏向較大未實現機會與較低capture ratio，且控制Target後Score↔R仍明顯為負 |
| 程式基準 | 以`test-branch-1_20260730_202010_4cc9d53.zip`、SHA256 `89769903bfe6abe41af8dead26c926a5e2123825115c39f035e71e3364dc55bf`為來源基準 |
| 唯一變更 | 不改模型或Target；只讀11G scores、11F No-time component arrays與11A canonical actual trade matches，聚焦原始Label=PASS |
| 固定指標 | `realization_gap_r=target_raw_r-r_multiple`；`favorable_capture_ratio=r_multiple/favorable_r`；另計算Score↔Favorable／Adverse、Score↔gap／capture及控制Target或兩成分後的partial Spearman |
| Decile歸因 | 分別依Score與Target切top／bottom 10%，輸出Target、Favorable、Adverse、realized R、gap與capture ratio，確認高Score是否只是更大未實現機會 |
| 防錯 | strict驗證11G report／scores、11A trade matches SHA256、11F No-time arrays及逐筆`target=favorable-adverse`；actual PASS配對數必須與11G report一致 |
| Validator閉環 | 本輪獨立直跑另發現11D component-loader synthetic fixture的`first_risk_breach_bar` tuple誤混入常數，已移除並重新驗證11D contract；不改正式11D／11H計算語意 |
| 正式結果 | OOS PASS 9,650 groups；Score↔Target 0.3464、Score↔Favorable 0.3980。Actual PASS 214筆；Score↔R −0.1066、Score↔gap 0.4051、Score↔capture −0.2023，控制Target後partial Score↔R −0.3167、控制Favorable＋Adverse後−0.3459 |
| 判定 | 11G失敗主因是策略實現率，不再微調MFE型Target或11G模型。下一步只能建立Selection內nested historical OOS strategy-realization coverage audit；不得用2021～2026正式OOS直接建訓練Target |
| Runtime | research-only、CLI-only、不訓練、不建立profile／checkpoint／threshold／策略回測，不加入互動選單 |
| 執行 | `python apps/breakout_quality.py audit-pass-realization-gap --filter-id breakout_quality_v1` |


### 3.57 11I Nested Selection Strategy-realization Coverage Audit（2026-07-30）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / TARGET_DIRECTION_PASS_COVERAGE_INSUFFICIENT`；Selection nested-OOS Target方向明確成立，但actual portfolio coverage不足以建立完整候選Target |
| 程式基準 | `test-branch-1_20260730_204217_d8ab94e.zip`，SHA256 `5c8361ca04ad4af1a3ea98212ab684c486975534d100c022ba8f41f1a28a56bc` |
| 必要前置 | 正式rolling params只涵蓋2021～2026，不可倒灌Selection。11I以2014-01-01～2020-12-31、120個月fixed train window、12個月OOS建立nested lookahead-safe params；策略replay只到11F Selection可評分事件上限2020-11-05，並以manifest fail-fast，避免Target使用2021價格。2014以前因資料不足不納入策略實現稽核 |
| 隔離輸出 | 透過`V16_MODELS_DIR=models/research/breakout_quality/selection_strategy_realization`輸出research `roos_*.json`；修正outer rolling writer忠實採用該環境路徑，正式`models/roos_*.json`不受影響 |
| Trial單一真理 | `--optimizer-trials`未指定時直接引用`config.training_policy.OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT`；不再硬編碼1000。明確CLI值仍可單次覆蓋。既有`prepare_selection_nested_roos.ps1`為靜態工件，config變更後必須重新執行`--prepare-only` |
| Replay | 使用canonical no-filter active-param replay、candidate capture與round-trip reconstruction，比較2014-01-01～2020-11-05 qualified、orderable、actual trades及11F No-time Target↔strategy R；不得超出11F `final_refit_date_range` |
| 正式結果 | Qualified 2,003、orderable 1,993；No-time Target match約95.16%。Actual round trips 459筆、Target matched 415筆，coverage vs qualified 21.77%；Target↔strategy R=0.5644、PASS內0.5182，top／bottom decile R=2.6354／−0.6363 |
| Target邊界 | Target方向通過，但actual trades受max positions、資金鎖定與候選競爭選擇，只能做coverage audit；未交易qualified candidates不得標0R，不授權直接建立Target arrays或訓練 |
| 判定 | 進入11J canonical per-candidate counterfactual execution；只移除capacity／cash competition，保留正式成交與出場規則。若仍未成交則保持缺值 |
| Runtime／UI | research-only、CLI-only，不加入互動選單，不改9A、scanner、optimizer正式參數或runtime |
| 執行 | 先`audit-selection-strategy-realization --prepare-only`產生PowerShell腳本；完成nested optimizer後再執行`audit-selection-strategy-realization --quiet` |

### 3.58 11J Canonical Per-candidate Counterfactual Execution Audit（2026-08-01停止）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `STOPPED / REPLAY_NONREPRODUCIBLE`；不再要求本機執行，不再修改core、observer、sidecar、日期或identity |
| 最終基準 | `test-branch-1_20260801_143545_4d07fe5.zip`，SHA256 `953b03570c4df22c4e0389f6031f6b90a70530171f76e1f6d5d8a60f52160ab7` |
| 最終失敗 | plain `dict replay_counts`＋獨立execution sidecar仍得到1,969／2,003，與11I凍結候選母體不一致；先前另曾得到1,978／2,003及於2014-05-15發生MemoryError |
| 結論 | 11J已消耗過多除錯成本，且問題集中在歷史replay重現而非Target經濟假設。不得放寬2,003 guard、補值、把缺少候選視為未成交或繼續修改正式交易核心 |
| 保留邊界 | 歷史程式與文件保留供追溯；任何11J輸出都不得作Target、模型、ranking或runtime依據 |
| 下一步 | 跳過counterfactual，進入11K只讀portfolio selection-pressure attribution |

### 3.59 11K Portfolio Selection-pressure Attribution Audit（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE`；只讀11I凍結工件，不重播市場、不建立counterfactual、不訓練 |
| 程式基準 | `test-branch-1_20260801_143545_4d07fe5.zip`，SHA256 `953b03570c4df22c4e0389f6031f6b90a70530171f76e1f6d5d8a60f52160ab7` |
| 研究目的 | 判斷portfolio capacity／cash competition發生時，actual trades是否已偏向同日較高No-time Target候選，或現有buy-sort系統性排除較高Target候選 |
| 固定輸入 | strict驗證11I JSON與orderable／trade-matches SHA256；actual trades必須是orderable ticker＋entry-date完整子集，且matched count必須等於11I 415 |
| 核心指標 | 同日Target percentile、top-half／top-quartile rate、按每日實際買入數k的Target top-k retention、Target opportunity gap、Target↔selected indicator，以及壓力分桶結果 |
| R邊界 | realized R只用actual selected trades；未選候選保持NaN，不填0R，不推估反事實R。另描述性計算selected percentile↔R及selected top／bottom quartile平均R |
| Runtime／UI | research-only、CLI-only，不加入互動選單，不修改9A、optimizer、threshold、Dataset、Target或core交易規則 |
| 執行 | `python apps/breakout_quality.py audit-selection-pressure --filter-id breakout_quality_v1` |




---


### 3.67 Breakout Quality固定單一Seed 42（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING / MODEL_RESULT_NOT_AVAILABLE` |
| 程式基準 | `test-branch-1_20260801_190545_31efeae.zip`；SHA256 `4feec395fd20ca42a9aee1024c4ef8fd813abd36eca7bbbace1c6ee071e3ed71` |
| 問題 | `BREAKOUT_QUALITY_RANDOM_SEED=None`仍隱含依profile選binary=42／continuous=1；使用者切換profile時Seed會暗中改變，仍不是單一真理來源 |
| 唯一修正 | `BREAKOUT_QUALITY_RANDOM_SEED`改為必填非負整數，預設42。Binary、continuous、pretraining、PIT與主選單workflow全部使用同一值；experiment profile不再影響Seed。CLI `--seed`只作單次覆寫 |
| Dataset／Label | 不需重建Dataset或relabel |
| 工件影響 | 9A既有Seed 42 checkpoint／scores可沿用。Continuous既有Seed 1 checkpoint與7-fold PIT Scores保留為歷史工件，但不得以`--resume`接成Seed 42；當前continuous workflow須重建Seed 42 checkpoint／PIT Scores |
| 固定條件 | 不改architecture、Label、continuous Target、loss、epoch、batch、sampling、PIT fold日期、buy-sort、策略參數、成交與帳務 |
| 驗證邊界 | 本輪只完成設定與路由契約；尚未產生Seed 42 continuous模型或PIT結果，不得宣稱預測或策略效果 |
| 下一步 | 使用者本機重跑formal suite；若繼續continuous主線，先移除或改名舊`point_in_time`目錄，再以Seed 42完整建立PIT Scores與audit |

### 3.68 Continuous PIT Workflow自動準備Dataset（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING / MODEL_RESULT_NOT_AVAILABLE` |
| 程式基準 | `test-branch-1_20260801_191450_5f7702c.zip`；SHA256 `7c5b5f5029c6303642be11430d090ebebea6a05be41e36da626c3f75550cb5f7` |
| 問題 | 主選單continuous objective直接呼叫PIT builder，未先執行binary workflow已有的Dataset refresh contract；首次執行在`dataset_summary.json`缺少時立即`FileNotFoundError` |
| 唯一修正 | 抽出共用`_dataset_refresh_step()`；binary與continuous共用相同Full dataset／全部股票檢查。Continuous在缺少、storage／source inventory／ticker coverage／feature policy不一致時先執行`build-dataset`完整重建；只有Label policy改變時使用`--relabel-only`；Dataset就緒後才執行PIT builder與audit |
| Dataset／Label | 本輪程式修正本身不改Dataset或Label契約。使用者本機目前缺少Dataset工件，因此下次主選單執行會建立正式Full dataset；若已有有效工件則跳過，若只有Label policy改變則快速relabel |
| 固定條件 | architecture、experiment profile、continuous Target、Seed 42、loss、epoch、batch、sampling、PIT fold／validation日期、Score、buy-sort、策略參數、成交與帳務均不變 |
| 驗證邊界 | Synthetic CLI已驗證缺少Dataset時命令順序必須為`build-dataset → build-point-in-time-scores → audit-point-in-time-scores`，Dataset已就緒時仍只執行後兩步。尚未在本環境建立完整台股Dataset或訓練PIT folds，不得宣稱模型結果 |
| 下一步 | 套用修補後重新由主選單按Enter；程式會先建立Full dataset，再開始Seed 42 PIT folds。正式測試結果以使用者本機`python apps/test_suite.py`為準 |

### 3.69 Continuous Target自動準備與Dataset identity綁定（2026-08-01）

| 項目 | 結果 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE` |
| 問題 | Full Dataset已成功建立，但continuous workflow直接進PIT builder；configured No-time Target manifest缺少時發生`FileNotFoundError`。既有Target loader另只比對group count與policy，Dataset重新掃描但數量相同時仍可能誤用舊group排列的Target arrays |
| 唯一修正 | 新增泛用`prepare-continuous-target`前置命令。主選單continuous命令順序改為`Dataset preparation → Continuous Target preparation → PIT Score build → PIT audit`。Target已存在時需通過group count、Dataset policy、Dataset artifact SHA256；衍生No-time Target另驗證來源Target manifest SHA256。缺少或stale時自動重建。Active workflow已採用的固定No-time公式允許直接重建既定arrays，不要求每次重跑歷史11E研究gate；原research-only CLI未帶approved flag時仍維持既有gate |
| Dataset／Label | 使用者本機Full Dataset已建立，無需再次掃描；Target arrays與manifest需建立。Label公式、Dataset內容與group contract不變 |
| 固定條件 | architecture=`inception_time_v1`、profile=`strategy_aligned_no_time_pass_magnitude_mse`、Target公式、Seed=42、PIT 12／24 months、loss、epoch、batch、sampling、buy-sort、策略參數與交易帳務均不變 |
| 驗證邊界 | 本輪只完成自動準備、manifest identity與synthetic契約；尚未產生Seed 42 PIT Scores或模型audit，不得宣稱預測能力或策略改善 |
| 下一步 | 套用修補後重新按Enter；Dataset應顯示skip，程式會先建立基礎component Target與No-time Target，再自動進入PIT folds |


### 3.70 Continuous PIT易讀模型評估報表（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE` |
| 程式基準 | `test-branch-1_20260801_194645_7b02dab.zip`；SHA256 `8d121ac8b579f2c3a0bf9d0316f208c32573705612d6513ad561edbbbcf754d8` |
| 問題 | Continuous workflow雖已有Target Markdown與PIT audit Markdown，但終端只輸出單行指標，PIT Markdown也缺少完整設定、Score coverage、逐fold分布、年度方向摘要、研究邊界與工件索引，使用者難以像既有binary workflow一樣直接閱讀 |
| 唯一修正 | `audit-point-in-time-scores`改以單一payload同步輸出表格化終端摘要、`selection_point_in_time_audit.md`完整易讀報表與JSON。固定章節包含執行設定、核心排序、年度穩定性、fold drift、PASS／REJECT重疊、orderable coverage、研究邊界及工件路徑。主選單狀態頁新增Target audit Markdown；Target current／重建完成時直接顯示易讀報表路徑 |
| 指標單一來源 | Console、Markdown與JSON共用同一payload，不重算另一套Spearman、decile、drift或coverage |
| Dataset／Label | 不需重建Dataset或relabel；既有Target與PIT模型工件均不因報表格式改變而失效 |
| 固定條件 | architecture、profile、Target、Seed=42、PIT 12／24 months、loss、epoch、batch、sampling、Score、buy-sort、策略參數與交易帳務均不變 |
| 驗證邊界 | Direct synthetic只驗證報表章節、數值來源與「策略optimizer未執行／Future Target未作runtime sort」邊界；尚未取得本機完整PIT模型結果，不得預先判定模型通過 |
| 下一步 | 使用者由主選單完成Target→PIT→audit後，直接審閱終端摘要或`selection_point_in_time_audit.md`；只有多數年度排序方向穩定為正時才進策略績效驗證 |


## 6. 實驗執行順序

```text
6A AdamW only（REJECTED）
→ 6B step-based LR schedule（REJECTED）
→ 7A old-history masking（REJECTED）
→ 低維市場 regime context（REJECTED）
→ ATR／波動率尺度 Label（REJECTED）
→ 退回 multiscale_cnn_v1 / baseline／固定百分比 Label
→ 8A sequence-only context ablation（ACCEPTED；歷史架構基準，後由 8F 取代）
→ 8B recent-decay time weighting（REJECTED；已退回 8A baseline）
→ 8C same-day cross-sectional rank context（REJECTED；已退回 8A baseline）
→ 8D classification + same-day ranking loss（REJECTED；已退回 8A baseline）
→ 8E fixed 8-seed probability ensemble（REJECTED；已退回 8A baseline）
→ 8F unique ticker/date group training（ACCEPTED；目前研究基準）
→ 8G unique-group patience 5（REJECTED；已退回 patience 1）
→ 8H unique-group batch size 64（REJECTED；已退回 batch size 128）
→ 8I unique-group matched optimizer steps refit（REJECTED；已退回 selected_epochs）
→ 8J direct best inner-validation checkpoint（REJECTED；已退回 selected_epochs）
→ 8K unique-group date-density balancing（threshold 0.5 REJECTED；0.55保留 forward-validation候選）
→ 8L date-diverse batch scheduling（REJECTED）
→ 8M Long Branch last-state pooling（REJECTED）
→ 8O zero-initialized gated temporal pooling（REJECTED）
→ 8P raw＋window-normalized dual-path（REJECTED；停止multiscale CNN家族微調）
→ 9A 單一InceptionTime（ACCEPTED；新的排序／高品質模型基準）
→ 9A-GN InceptionTime GroupNorm ablation（REJECTED；已退回9A-BN）
→ 9B ModernTCN（REJECTED；已退回9A並轉為legacy）
→ 9C TS2Vec Selection-only自監督預訓練＋frozen linear probe（REJECTED；已退回9A並轉為legacy）
→ 9D MantisV2 frozen encoder＋linear probe（REJECTED；已退回9A並轉為legacy）
→ 9E MOMENT-1-base frozen encoder＋linear probe（REJECTED；已退回9A並轉為legacy）
→ 9F 小型Patch Transformer supervised architecture（REJECTED；已退回9A並轉為legacy）
→ unique-group score單一真理＋逐年OOS診斷（RESULT_AVAILABLE；9A重跑確認）
→ 固定9A threshold 0.5的策略層no-filter對照（REJECTED_FOR_RUNTIME_DEPLOYMENT）
→ 既有OOS交易層歸因與partial-year修正（RESULT_AVAILABLE；獨有交易選擇效果−46.77R）
→ 9A Quality Score只作finalists-agree同票候選排序（RESULT_AVAILABLE；總報酬小幅正向但僅2023改善，不部署）
→ base-finalist-best單一member Score Ranking隔離比較（REJECTED；全域Score第一使總報酬−32.12pp，僅2023改善）
→ Breakout Event Regime Coverage Audit（RESULT_AVAILABLE；確認深度回撤支撐缺口＋熊市關係漂移）
→ 2022 Focus-year Regime Attribution（RESULT_AVAILABLE；排除low-support後PR-AUC仍僅0.4722，確認廣泛concept shift）
→ Active InceptionTime可設定Receptive Field（REJECTED；RF600與RF251定性結果均不好，退回RF229）
→ Stage 0 point-in-time Market Set Bank＋Stage 1 Global Market Set Encoder（REJECTED；logical-batch修正後OOS PR-AUC 0.6092，低於9A 0.6257；已轉legacy）
→ 10A Candidate-conditioned Query（REJECTED；OOS PR-AUC 0.6009，低於9A；不啟動learned lag）
→ 11A Strategy-aligned Continuous Target Audit（RESULT_AVAILABLE；DISTRIBUTION_PASS／TRADE_R_PRIMARY_PASS）
→ 11B Strategy-aligned Daily Percentile Regression（REJECTED；OOS target弱正向但actual R Spearman −0.0003、decile反轉）
→ 11C Qualified Candidate-set Coverage Audit（RESULT_AVAILABLE；qualified／orderable Score↔Target未崩落，不支持qualified sampling）
→ 11D Label-conditional Target Component Attribution Audit（RESULT_AVAILABLE；PASS內連續Target有效，但time penalty方向與actual R錯位）
→ 11E Fixed Time-penalty Ablation Audit（RESULT_AVAILABLE；overall／PASS／decile spread均改善，固定消融通過）
→ 11F No-time Target Arrays＋Selection-only Learnability Audit（RESULT_AVAILABLE；SELECTION_LEARNABILITY_PASS）
→ 11G PASS-conditional No-time Magnitude Ranker（REJECTED；OOS PASS Target排序成立但actual PASS Score↔R −0.1066、decile反轉）
→ 11H PASS-only Realization-gap Attribution Audit（RESULT_AVAILABLE；STRATEGY_REALIZATION_GAP_CONFIRMED）
→ 11I Nested Selection Strategy-realization Coverage Audit（RESULT_AVAILABLE；Target方向通過，但actual coverage 21.77%不足）
→ 11J Canonical Per-candidate Counterfactual Execution Audit（STOPPED；replay無法重現2,003筆，不再修補）
→ 11K Portfolio Selection-pressure Attribution Audit（IMPLEMENTED；歷史read-only工件保留，不再作主線）
→ Selection Point-in-time Score Workflow（SCORES_BUILT；7 folds／18,247 groups完成，待模型audit重跑）
→ 模型audit通過後才新增泛用Score buy-sort與Baseline／Sort Only比較
→ 有合理改善跡象後才讓主策略參數適應Score排序，最後再作正式OOS比較
```

任何新結果都必須追加至第 3 節，並同步更新第 2 節目前基準、第 4 節排除方向與第 5～6 節待辦順序。


### 3.66 單一Seed與刪除舊Strategy Compare入口閉環（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING`；已完成程式、文件與獨立契約修正，待使用者重跑正式suite |
| 程式基準 | `test-branch-1_20260801_185047_5a647b5.zip`；SHA256 `2eb142dcbd23d9508fc5e1aed3cdb2dcb49eac35e48733b3dc57fb9e64fa51dd`；bundle `to_chatgpt_bundle_20260801_185213_8c8baa64.zip`；SHA256 `acb26b34e76dc5ecd4731c5f676437543c9ee39c78ece6ec1f8e61f98fd1cdbd` |
| Formal結果 | quick gate只有`help::breakout_quality_strategy_compare.py`失敗；consistency兩列為CMD仍要求舊script存在及B170仍宣告舊entry；meta quality只因synthetic suite非零退出連帶失敗 |
| Seed根因 | `BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED=42`控制低階train CLI，`BREAKOUT_QUALITY_WORKFLOW_SEED=1`另控制主選單／PIT workflow，形成兩套使用者設定 |
| Seed修正 | 只保留`BREAKOUT_QUALITY_RANDOM_SEED`。預設`None`時依當前profile objective自動解析：binary classification為42、continuous ranker為1；使用者只切換`BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE`。需特殊重現時可在config填非負整數，單次CLI `--seed`優先 |
| Seed簡化補正 | `IMPLEMENTED / RESULT_NOT_AVAILABLE`：`BREAKOUT_QUALITY_RANDOM_SEED`改為必填非負整數，預設42；binary、continuous、pretraining與PIT皆使用同一值。移除`None`與依profile自動選Seed邏輯。Dataset／Label不需重建；若continuous既有checkpoint／PIT folds為Seed 1，改用42後須重建相應模型工件，不得續接成同一實驗 |
| 入口修正 | 已刪除的`apps/breakout_quality_strategy_compare.py`不再是相容或正式入口；quick gate、CMD、Architecture、B170／B182、synthetic registry全部改以`apps/breakout_quality.py strategy-compare`為唯一入口 |
| 固定條件 | 不改architecture、Dataset、Label、Target、loss、epoch、PIT fold邊界、threshold、buy-sort、optimizer search space、portfolio成交或帳務；Seed由profile-dependent 1／42改為單一42，因此continuous checkpoint與PIT Score需重建 |
| Dataset／Label | 不需重建Dataset或relabel；9A既有Seed 42模型可沿用。Continuous既有Seed 1 checkpoint／PIT folds不可視為Seed 42工件，必須重建 |
| 下一步 | 套用修補後重跑`python apps/test_suite.py`；正式結果以使用者本機輸出為準 |


### 3.71 Selection PIT Seed 42探索性模型結果與Seed契約更正（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / MODEL_DIRECTION_PASS`；後續已確認Seed 42與PIT方法合法性無衝突，且統一為唯一正式Seed |
| 結果來源 | 使用者本機輸出；程式ZIP `test-branch-1_20260801_200203_fe190a4.zip`，SHA256 `3d7daa9333481667c282845636720118b4c724af66dbf86eec07fd51446b7dee` |
| 工件identity | `inception_time_v1 / strategy_aligned_no_time_pass_magnitude_mse / strategy_aligned_opportunity_no_time_r_v1`；Seed 42；2014-01-01～2020-12-31；7 folds；18,247 groups；coverage 100% |
| PASS-only排序 | groups 10,160；global Spearman 0.3074；mean daily rho 0.2370；top／bottom decile Target 3.3883／1.0428R，spread 2.3455R |
| 年度穩定性 | 2014～2020共7/7年度Spearman為正，7/7年度spread為正；年度Spearman 0.2367～0.4034 |
| Drift／分類重疊 | drift flag=False，最大相鄰mean shift 0.8832 pooled SD；Score vs PASS AUC 0.5620，證明不是單純重複PASS／REJECT |
| Orderable coverage | 舊`selection_orderable_candidates.csv`不存在，因此模型audit未提供；改由下一階段兩組canonical strategy replay現場產生 |
| 方向判定 | 模型層方向通過原定gate，可支持Score Sort假設；Seed 42是在結果前既有的固定正式設定，不構成PIT前視或OOS洩漏，可直接作Selection策略比較前置工件 |
| 修正 | 取消獨立workflow Seed；binary、continuous、pretraining與PIT全部共用`BREAKOUT_QUALITY_RANDOM_SEED=42`。PIT fold fingerprint與策略入口仍檢查實際Seed，僅禁止不同Seed工件混接 |

### 3.72 Selection PIT Score Sort與策略比較接線（2026-08-01）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / STRATEGY_RESULT_NOT_AVAILABLE`；Seed 42 PIT模型gate已通過，不需因Seed重建 |
| 程式基準 | 修改基準`test-branch-1_20260801_200203_fe190a4.zip`，SHA256 `3d7daa9333481667c282845636720118b4c724af66dbf86eec07fd51446b7dee`；本輪修補ZIP hash列於交付回覆 |
| 唯一變更 | 在已完成的PIT模型層之後接上泛用Score buy-sort與Controlled Baseline／Sort Only Selection replay；不改continuous模型、Target、loss、epoch、LR、batch、sampling或主策略參數 |
| 固定條件 | `seed=42`、`base-finalist-best`、單一member／min_agree=1、max positions 10、rotation off；候選生成、進場、持倉、資金、停損停利及歷史active-param schedule固定 |
| Dataset／Label | 不需重建Dataset、Label、Continuous Target或既有Seed 42 PIT checkpoints／Scores／audit；目前正式單一Seed即為42 |
| Score source | 新增`filters/breakout_quality/ranking_score_store.py`，strict驗證PIT score／manifest／audit identity、日期、coverage、模型gate，並以SHA256綁定audit所使用的PIT manifest／Scores／coverage／Continuous Target manifest；`runtime.py`以scoped context在canonical與Selection PIT來源間派送，不改scanner hard-filter runtime |
| 排序語意 | 有效Score由高到低；同分使用原buy-sort；缺分不排除、不填0，排在有分候選後並完整回退原buy-sort；最後ticker deterministic。Continuation／Re-entry沿用原setup的source與Score payload |
| 參數來源 | `selection_point_in_time + base-finalist-best`自動解析`models/research/breakout_quality/selection_strategy_realization/roos_base_best.json`，要求rolling active params完整覆蓋PIT期間且每期1 member／min_agree=1 |
| Controlled comparison | Baseline與Score Sort只切換`use_breakout_quality_ranking`；候選生成、進場、持倉、資金、停損停利、rotation與策略參數相同 |
| 離線診斷 | 兩組replay後才joinPIT target，輸出orderable Score coverage、selected Target percentile、Target top-k retention、opportunity gap及selected Target mean；Future Target不進runtime sort |
| 工件 | `strategy_compare_score_ranking_base_finalist_best_selection_point_in_time/`下輸出comparison Markdown／JSON、equity、trades、daily capacity、yearly、兩組orderable／selected buys與Target診斷CSV |
| UI／入口 | 主選單`[1]`已接Selection PIT；唯一CLI為`apps/breakout_quality.py strategy-compare`，舊`apps/breakout_quality_strategy_compare.py`已刪除且不得恢復。`final_selection_model_oos`仍明確未開放 |
| Selection／OOS結果 | Seed 42 Selection PIT模型結果已通過模型gate；Baseline／Sort Only策略結果尚未執行，OOS亦未執行。模型結果不得補寫成策略績效 |
| 與基準差異／採用 | 程式層採用Score Sort接線；模型gate已由Seed 42 PIT結果通過，策略結果尚未採用。只有Selection策略比較出現合理選股或績效改善，才允許進主策略參數重新優化 |
| 驗證 | 獨立synthetic已驗證Score desc、同分／缺分fallback、source context還原、audit來源hash、模型gate不讀策略結果、Future Target post-replay join與單一Seed契約；正式完整replay尚未執行 |
| 下一步 | 直接由主選單獨立選`[1]`執行Baseline／Sort Only；不需重建既有Seed 42 PIT folds，也不自動串接optimizer |


### 3.73 單一Seed與唯一Strategy Compare入口回歸修正（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / STRATEGY_RESULT_NOT_AVAILABLE` |
| 程式基準 | `test-branch-1_20260802_010530_611f6d3.zip`；SHA256 `2faa38e8013dd1080e01ed7173ea43fab94b7886bee209a20f5bf5cf866614c2` |
| 問題 | 基準重新出現`BREAKOUT_QUALITY_WORKFLOW_RANDOM_SEED=1`與已刪除的`apps/breakout_quality_strategy_compare.py`，違反先前已完成的單一設定及單一正式入口契約 |
| 根因 | 後續Score Sort修補依較早的「保留相容轉接」計畫重新加入舊app，並把交接文件中的seed=1誤當成必須獨立於既有Seed 42的技術契約 |
| 修正 | 移除`BREAKOUT_QUALITY_WORKFLOW_RANDOM_SEED`，workflow直接呼叫`resolve_breakout_quality_random_seed()`；刪除舊app，只保留`python apps/breakout_quality.py strategy-compare` |
| 固定條件 | `BREAKOUT_QUALITY_RANDOM_SEED=42`；不改architecture、profile、Target、PIT fold、loss、epoch、LR、batch、sampling、Score Sort、策略參數、成交或帳務 |
| Dataset／Label／模型工件 | 不需重建Dataset、Label、Continuous Target或現有Seed 42 PIT工件；若日後使用CLI覆寫不同Seed，原有fingerprint與strategy gate仍會阻止混接 |
| 結果 | 既有Seed 42 PIT模型結果維持`MODEL_DIRECTION_PASS`；本輪未執行Baseline／Score Sort策略replay，策略結果仍為`RESULT_NOT_AVAILABLE` |
| 測試 | 新增synthetic契約：canonical config只能有一個Seed設定、binary與continuous workflow對同一override解析相同Seed、舊strategy compare app必須不存在 |
| 下一步 | 刪除本機舊app後，直接執行`python apps/breakout_quality.py strategy-compare`或從主選單選`[1]` |

### 3.74 單一入口與Checklist Formal Bundle閉環（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING`；已依使用者提供bundle完成失敗閉環，待本機重新執行正式suite確認 |
| 程式基準 | `test-branch-1_20260802_011924_a6141cf.zip`；SHA256 `61036fe4c869a496081103e67c74f235c620b3e1c5ce9b391ba44d817ad7a8a6` |
| Formal bundle | `to_chatgpt_bundle_20260802_012046_117ce083.zip`；SHA256 `a8d7f45c641cf81ba0b3333c1643138022c3fad1b6505f46684ebe12d8e7e976` |
| 原始結果 | quick gate、chain checks、ml smoke通過；consistency只有`breakout_quality_strategy_compare_uses_only_canonical_app_entry`失敗；meta quality四項失敗中coverage為同一synthetic失敗連帶，另三項為B183／B184主表及T280 DONE摘要漏同步 |
| 根因 | 前一修補ZIP只能覆蓋檔案，未實際刪除本機舊`apps/breakout_quality_strategy_compare.py`；同輪新增G收斂紀錄B183、B184與T280，但未同步主表與DONE測試映射 |
| 修正 | 從本輪實際程式基準刪除舊strategy compare app；補上B183 Selection PIT Score排序／策略比較主表、B184單一Seed／唯一入口主表及T280對B183映射；新增B26與B184的PARTIAL到DONE閉環紀錄 |
| 固定條件 | 不改Dataset、Label、Continuous Target、PIT Scores、模型、Seed 42、Score Sort、策略參數、交易規則、帳務或正式績效結果 |
| 結果邊界 | 本輪只修正式入口與測試治理同步；Baseline／Score Sort策略結果仍為`RESULT_NOT_AVAILABLE`，不得由測試通過推論策略有效 |
| 下一步 | 套用修補並確實刪除舊app後，重新執行`python apps/test_suite.py`；預期consistency單一失敗與meta四項連帶失敗消失，正式結果仍以本機輸出為準 |



### 3.75 Checklist Registry唯一映射閉環（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING`；已依第二次formal bundle修正兩個consistency failure，尚未宣稱本機formal suite通過 |
| 程式基準 | `test-branch-1_20260802_013153_8239ebe.zip`；SHA256 `0590fd64c93afa09aa35b1a54840fea05ab4c159be293c3d227a1823a092a644` |
| Formal bundle | `to_chatgpt_bundle_20260802_013316_c2844f24.zip`；SHA256 `31c0a5e0a2127427d930ea5d194aa0d55ae1329605b14c97cc501026d772e3cd` |
| 原始結果 | quick gate、chain checks、ml smoke通過；consistency只剩`B184_done_summary_has_done_test_mapping`與`done_test_names_unique`兩項；meta quality僅因synthetic suite非零退出連帶失敗 |
| 根因 | T279與T280同時映射`validate_breakout_quality_point_in_time_score_builder_contract_case`，違反DONE test name唯一性；B184雖為DONE但沒有自己的T映射 |
| 唯一修正 | 將B183的Score Sort／來源context／audit hash／模型gate／post-replay Target診斷拆為`validate_breakout_quality_selection_point_in_time_score_sort_contract_case`；新增`validate_breakout_quality_single_seed_single_entry_contract_case`承接B184；T280改指B183獨立validator並新增T281映射B184 |
| 固定條件 | 不改Dataset、Label、Continuous Target、PIT Scores、模型、Seed 42、Score Sort runtime、策略參數、交易規則、帳務或績效結果 |
| 結果邊界 | 本輪只修validator registry與checklist機械同步；Selection Baseline／Score Sort策略結果仍為`RESULT_NOT_AVAILABLE` |
| 下一步 | 套用修補後重新執行`python apps/test_suite.py`；正式結果以使用者本機輸出為準 |

### 3.76 Selection PIT缺分候選Runtime閉環（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / STRATEGY_RERUN_PENDING`；已重現並修正Score Sort第一日runtime錯誤，尚未取得完整Baseline／Score Sort比較結果 |
| 程式基準 | `test-branch-1_20260802_013952_744919d.zip`；SHA256 `6683b981a6ca4e9719a41bf2518462cb6851c9d7e168656dade9a4608b21f886` |
| 使用者本機結果 | Baseline `no_filter`已完成2014-01-01～2020-12-31 replay，終值2,840,064；Score Sort在2014-01-02 `build_daily_candidates`以`TypeError: float() argument must be a string or a real number, not 'NoneType'`中止，因此本輪沒有策略績效差異可判定 |
| 根因 | Selection PIT lookup對score table缺少ticker／signal-date事件時正確回傳`available=false`、`score=None`、`unavailable_reason=missing_ticker_date_score`；但`attach_breakout_quality_rank()`仍沿用舊hard-filter假設，先執行`float(score)`且禁止不可評分事件進入continuation／re-entry state，與B183「缺分不得排除或填0、須回退原buy-sort」契約衝突 |
| 唯一修正 | 新增單一`normalize_breakout_quality_rank_payload()`；可評分payload仍嚴格要求0～1有限值，不可評分payload保存`score=None`、原因、原始score date、score source及PIT identity。Candidate row、continuation與re-entry共用同一正規化，不重新查詢原事件Score |
| 排序語意 | 缺分事件不是REJECT；normal／continuation／re-entry候選均保留`use_breakout_quality_ranking=true`與`available=false`，`breakout_quality_score=None`，由既有buy-sort邏輯排在有效Score後並完整回退原排序 |
| 額外修正 | 原attach流程會丟失`score_source`，使PIT continuation下一日被誤判為canonical source；新版保存全部來源metadata。舊持倉fallback也不再把`score=None`硬標成`available=true` |
| 文件契約同步 | B170舊文字的「不可評分事件保守排除」改為分流：hard-filter維持保守REJECT；score-ranking缺分不得排除或填0，須保存payload並回退原buy-sort，與B183及runtime一致 |
| 固定條件 | 不改Dataset、Label、Continuous Target、Seed 42、PIT checkpoints／Scores／audit、模型gate、active params、候選生成、成交、持倉、資金、停損停利、buy-sort優先規則或Future Target邊界 |
| Dataset／Label／模型工件 | 不需重建；只需重新執行主選單`[2] 策略績效驗證` |
| 獨立驗證 | T280新增不可評分PIT payload保存、PIT source繼承、不重新lookup、candidate row不執行`float(None)`及Score維持None的direct synthetic案例；全專案靜態與依賴檢查另由本輪交付列示 |
| 結果邊界 | Baseline終值只證明基準路徑可完成；Score Sort尚未完成，因此不得比較報酬、MDD、選股Target或宣稱排序有效／無效 |
| 下一步 | 套用修補後直接重跑`python apps/breakout_quality.py strategy-compare`或主選單`[1]`；不需重跑PIT模型或audit |

### 3.77 Selection PIT Score Event Date診斷閉環（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / STRATEGY_RERUN_PENDING`；runtime replay不再因假性Score identity mismatch中止，正式策略比較結果待使用者本機重跑 |
| 程式基準 | `test-branch-1_20260802_015757_25ace3c.zip`；SHA256 `f0591239b0caa737252b83e98a02d204a7d72eb668b58bb942daf6341f446cc9` |
| 使用者結果 | Baseline完成2014-01-01～2020-12-31，終值2,840,064；Score Sort已通過先前`float(None)`位置，但在post-replay策略診斷拋出「策略replay使用的Breakout Quality Score與PIT score table不一致」，未產生正式比較報表 |
| 根因 | continuation／re-entry的交易`signal_date`可以晚於原始breakout事件；runtime已正確保存並沿用原始`breakout_quality_score_date`，但`_strategy_selection_diagnostics()`仍以新的交易`signal_date`對回PIT table，將合法沿用的原始Score誤判為不一致，Future Target也會對錯事件 |
| 修正 | 新增`score_event_date`診斷鍵：優先使用`breakout_quality_score_date`，只有未提供時才回退candidate `signal_date`。PIT Score identity、coverage及Future Target事後join均使用原始Score事件日期；交易選取與成交追蹤仍使用當前`trade_date／signal_date`，兩種日期語意不再混用 |
| Fail-fast | 真正Score數值不一致仍會拒絕，錯誤訊息新增ticker、trade date、交易signal date、score event date、runtime score及PIT score，避免再次只得到無法定位的總括錯誤 |
| 固定條件 | Seed 42、PIT Score工件、continuous模型、Target、buy-sort、歷史active params、候選生成、成交、持倉、資金、停損停利及帳務完全不變；Future Target仍只在replay後離線使用 |
| Dataset／模型工件 | 不需重建Dataset、Label、Continuous Target、PIT folds、checkpoint、Scores或模型audit；只需重新執行Selection策略比較 |
| 獨立驗證 | T280新增continuation／re-entry交易signal date晚於原始score date的案例，確認Score identity與Target均對回原事件；另驗證真正0.7對0.8的Score差異仍fail-fast且錯誤包含完整identity |
| 下一步 | 執行`python apps/breakout_quality.py strategy-compare`；只有完整產出Baseline／Score Sort報表後，才能判斷是否進入主策略參數適應階段 |

### 3.78 Selection PIT Score Sort經濟效果結果（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `RESULT_AVAILABLE / MODEL_DIRECTION_PASS / SORT_ONLY_REJECTED / PARAM_ADAPTATION_DIAGNOSTIC_NEXT` |
| 程式基準 | `test-branch-1_20260802_020859_8ae2f29.zip`；SHA256 `2c51f5e0ae36204205aac25b81452935e99a1ac2c862329deeff33ec65c4e0f4` |
| 比較設計 | 2014-01-01～2020-12-31；Selection point-in-time Scores；歷史active params無前視；`base_finalist_best`；single member／min_agree=1；Baseline與Score Sort唯一差異為`use_breakout_quality_ranking=False/True`，hard filter兩組皆False |
| 固定模型 | `breakout_quality_v1 / inception_time_v1 / strategy_aligned_no_time_pass_magnitude_mse / strategy_aligned_opportunity_no_time_r_v1 / seed 42`；不改loss、epoch、LR、batch、sampling、threshold或PIT folds |
| Baseline結果 | 淨總報酬182.62%、MDD 13.18%、RoMD 13.86、年化16.00%、Log R² 0.9349、月勝率61.90%、527 trades、勝率38.14%、Payoff 2.92、EV 0.28R、平均曝險77.33% |
| Score Sort結果 | 淨總報酬144.80%、MDD 21.53%、RoMD 6.73、年化13.65%、Log R² 0.8599、月勝率63.10%、502 trades、勝率35.26%、Payoff 3.07、EV 0.35R、平均曝險54.20% |
| 主要差異 | Score Sort淨總報酬−37.82pp、MDD +8.35pp、RoMD −7.13、年化−2.36pp、交易數−25、勝率−2.88pp、平均曝險−23.13pp；Payoff +0.15、EV +0.07R、月勝率+1.19pp |
| 年度穩定性 | Score Sort於2014、2016、2017優於Baseline；2015、2018、2019、2020較差。最大負貢獻為2018 −12.08pp，其次2015 −9.43pp；不是單一年份造成，4/7完整年度落後 |
| 候選供給 | 平均每日可掛單候選19.78→19.66，供給幾乎不變；候選供給不足日329→338只增9日。期末未滿倉日755→630、持股缺口格日2336→2121反而改善，因此報酬惡化不能歸因為候選數不足或持倉slot不足 |
| Selection Target診斷 | Orderable Score coverage 0.9808→0.9942；selected Target percentile 0.6964→0.7170；top-k retention 0.3679→0.4694；opportunity gap 1.9209R→1.3577R；selected Target mean 1.0868R→1.2079R。模型確實把實際買入候選往較高No-time Target移動 |
| 核心判讀 | 模型排序與Target選股層通過，但No-time Target不是實際策略的資本效率或可捕捉R。Score Sort選到的候選具有較高Target與較高單筆EV／Payoff，卻在既有sizing、停損停利、partial exit、持有期與資金配置下造成平均曝險大幅下降、勝率與路徑穩定性惡化，最終總報酬及MDD皆變差 |
| 採用判定 | `Sort Only`不得成為正式排序；正式runtime維持Baseline。Continuous ranker本身不淘汰，因PIT模型gate及實際selected Target改善均成立，符合「先確認模型有效，再判斷參數是否需適應」的研究假設 |
| 下一個單一變更 | 先新增read-only `score-ranking capture attribution audit`，使用既有兩組replay工件分解初始投入比例、stop distance／ATR risk、fill rate、持有期、半倉後殘餘slot-days、exit reason、realized R／Target capture ratio、產業／日期集中度及2018差異。Future Target仍只作post-replay join，不進runtime或optimizer。只有歸因顯示問題可由既有主策略參數空間調整，才固定Score契約後執行Selection內參數適應 |
| OOS邊界 | 本結果僅為Selection PIT比較；未執行Adapted策略或正式OOS。不得使用OOS調Score權重、排序規則或策略參數 |

### 3.79 Score-ranking Capture Attribution與彩色易讀報表（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / CAPTURE_AUDIT_RESULT_NOT_AVAILABLE`；程式與報表契約完成，實際歸因結果待使用者從既有replay工件執行 |
| 程式基準 | `test-branch-1_20260802_022643_9c4f83a.zip`；SHA256 `fd8e4d0e3b04e03fc47ce610d869f24499cb937a1b38c4f98dfd057e2ff1b032` |
| 前置結果 | 3.78已確認PIT模型與selected Target方向通過，但Sort Only總報酬、MDD、RoMD、年化與平均曝險惡化；正式runtime維持Baseline，下一步只允許read-only capture attribution判斷是否有參數適應的機械瓶頸 |
| 新增audit | `tools/filters/breakout_quality/audit_score_ranking_capture.py`只讀既有Baseline／Score Sort transaction history、策略summary及post-replay selected-target diagnostics，逐筆重建entry到full exit lifecycle；不重播portfolio、不改candidate、Score、params、成交、帳務或optimizer |
| 歸因指標 | 平均預留與實際投入、投入／預留比例、初始stop distance、保留買單fill rate、持有日、首次partial時間、partial到full exit日曆日、依daily-capacity交易日曆計算的尾倉slot-days、partial比例、entry-date／月份集中度、可用時的產業集中度、exit reason、Realized R、Target R、Target capture ratio、realization gap、capital return及年度差異；交易列沒有canonical產業欄位時產業指標為N/A，不自行推測 |
| 決策邊界 | 只有Target選擇改善、經濟效果失敗且audit確認曝險／sizing／capture／turnover／fill至少一項可觀測瓶頸，才標記`ADAPTATION_DIAGNOSTIC_SUPPORTED`；否則Sort Only維持淘汰且不得直接啟動optimizer。這只是Selection內是否值得做參數適應的診斷，不是OOS採用證據 |
| Future Target | 只讀兩組replay完成後輸出的selected-target diagnostics；不得進runtime排序、資金配置、成交或optimizer，payload明確保存`future_target_used_for_runtime=false` |
| 易讀報表 | 原`strategy_comparison`及新`score_ranking_capture_audit`都直接在console完整顯示；TTY用綠／紅／黃／灰，非TTY退回純文字。正式檔案保留Markdown、JSON與必要CSV，停止產生HTML，舊HTML於重建時清除；顏色只呈現已計算差異，不建立第二套指標 |
| 正式工件 | `strategy_comparison.md/.json`；`score_ranking_capture_audit.md/.json`；`no_filter_capture_lifecycle.csv`、`score_ranking_capture_lifecycle.csv`、`score_ranking_capture_yearly.csv`、`score_ranking_capture_scenarios.csv`；console只顯示專案根目錄相對路徑 |
| 重用流程 | `[2] 策略績效驗證`完整replay後自動在console輸出兩份易讀報表；已有3.78 replay工件時可用`strategy-compare --comparison-mode score-ranking --score-source selection_point_in_time --param-policy base-finalist-best --capture-audit-only`只重建報表與audit，不重跑portfolio |
| 固定條件 | 不改Dataset、Label、Continuous Target、PIT folds／Scores／audit、Seed 42、Score Sort、歷史active params、候選生成、成交、資金、停損停利、帳務或3.78既有績效結果 |
| 驗證 | 新增獨立synthetic覆蓋trade lifecycle、投入／stop、partial日曆天數與daily-capacity交易slot-days、entry-date／月份集中度、產業欄位缺失N/A、Target capture、decision gate、Future Target runtime隔離、console完整表格與ANSI色彩、Markdown色彩語意、HTML不產生且舊檔清除、所有正式輸出工件及`--capture-audit-only`只讀重建且禁止portfolio replay；完整本機歸因數值尚未執行，不可預寫結論 |
| 下一步 | 先執行`--capture-audit-only`取得3.78既有replay的capture結果；只有狀態為`ADAPTATION_DIAGNOSTIC_SUPPORTED`，才提出固定Score契約下的Selection策略參數適應範圍 |

### 3.80 Breakout Quality主選單編號一致化（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_APPLICABLE`；只調整互動選單編號與對應測試／文件，不改模型、Score、策略或報表計算 |
| 程式基準 | `test-branch-1_20260802_022643_9c4f83a.zip`；SHA256 `fd8e4d0e3b04e03fc47ce610d869f24499cb937a1b38c4f98dfd057e2ff1b032` |
| 新選單 | `[1/Enter] 模型研究與驗證`、`[2] 策略績效驗證`、`[3] 查看目前設定與工件狀態`、`[0] 離開` |
| 輸入契約 | Enter與數字1皆進模型研究；數字2進策略驗證；數字3顯示狀態；無效提示範圍同步為0～3 |
| 固定條件 | 不改Dataset、Label、Continuous Target、PIT Scores、Seed 42、Score Sort、歷史active params、交易引擎、capture audit或任何績效結果 |
| 驗證 | synthetic CLI需分別驗證Enter／1／2／3路由與畫面文字，避免只改顯示而未改執行行為 |

### 3.81 Console易讀報表、相對工件路徑與HTML移除（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_APPLICABLE`；只調整breakout-quality使用者可見輸出，不改模型、Score、策略、Target或績效計算 |
| 程式基準 | `test-branch-1_20260802_031330_46cfab1.zip`；SHA256 `15b26d3fc08ae0b80fd003cde47ba98e2935e665f4e200e2ccc206b1b8455697` |
| Console SSOT | 新增`filters/breakout_quality/console_report.py`，統一標題、段落、key-value、表格、狀態、ANSI-safe欄寬與工件清單；`apps/breakout_quality.py`狀態頁及主要build／train／audit／strategy報表共用，不再各自拼接不同格式 |
| 易讀報表 | 策略比較與capture audit的完整易讀內容直接寫入console；TTY使用綠／紅／黃／灰，重新導向或非TTY自動退回純文字。Markdown／JSON／CSV仍作正式持久工件，不新增HTML |
| HTML處理 | 停止產生`strategy_comparison.html`與`score_ranking_capture_audit.html`；重建主報表或capture audit時會刪除同目錄舊版HTML，避免使用者誤讀過期檔案 |
| 路徑規則 | 所有breakout-quality console工件／狀態路徑以專案根目錄為基準顯示`/`分隔相對路徑；manifest、hash identity與runtime canonical path不變。此通用顯示規則已寫入`doc/PROJECT_SETTINGS.md` |
| 固定條件 | Dataset、Label、Continuous Target、Seed 42、PIT folds／Scores／audit、Score Sort、歷史active params、候選生成、成交、資金、停損停利、帳務與3.78績效結果全部不變 |
| 驗證邊界 | Direct synthetic驗證console完整章節、強制色彩與非HTML工件、舊HTML清除、`--capture-audit-only`不重跑portfolio、相對路徑及共用格式；本輪沒有新的capture或策略數值 |

### 3.82 模型研究簡易報表配色與Dataset建立去洗版（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_APPLICABLE`；只改善`[1/Enter] 模型研究與驗證`的console可讀性與Dataset建立進度更新，不改任何研究或策略結果 |
| 程式基準 | 來源`test-branch-1_20260802_034615_439e600(3).zip`；SHA256 `268eed2ee4312891166197edb7257827d94752f09ebdd8a71b2ba3b0b58085b1`；本輪修補ZIP只包含修改檔，SHA256列於交付回覆 |
| 共用色彩 | workflow標題／段落、Dataset／Continuous Target準備狀態、PIT fold plan、fold執行狀態與PIT模型評估報表改用既有`console_report`色彩語意：青色為標題／主要證據、綠色為完成／正向／完整coverage、黃色為待執行／部分coverage／需注意、紅色為負向／drift、灰色為中性或不適用；非TTY與`NO_COLOR`仍輸出純文字 |
| Dataset去洗版 | 共用`InlineProgress`先依終端顯示寬度截斷含ANSI／中日文寬字元的進度文字，再以carriage return覆寫同一列；較短新內容會補空白清除殘字，skip訊息會先清除進度列後獨立輸出，最後完成摘要只輸出一次完整行。進度文字縮短為`ticker／events／groups／elapsed`，避免窄視窗自動換行累積 |
| PIT報表 | 核心Spearman、年度正向率、fold PASS rho、drift、AUC、Top-decile PASS改善、Score coverage與研究邊界使用同一套狀態色；純文字輸出與彩色輸出移除ANSI後內容完全一致，不建立第二套指標或判定 |
| 固定條件 | 不改Dataset schema、feature、Label、Continuous Target公式、Seed 42、training scope、fold日期、checkpoint、PIT Scores、策略排序、active params、候選生成、交易、資金、停損停利、帳務或3.78績效結果 |
| Dataset／Label／模型工件 | 不需重建；既有工件可直接使用。只有使用者原本就缺少Dataset／Target／PIT工件時，互動流程才會照既有契約建立 |
| 獨立驗證 | 全專案Python語法解析、修改模組import、ANSI／純文字內容同一性、TTY長行截斷與單列覆寫、非TTY只輸出最終狀態、Markdown表格欄數、修改檔路徑與ZIP內容均由本輪獨立檢查；正式`apps/test_suite.py`依專案規則留待使用者本機執行 |
| 結果邊界 | 本輪沒有Selection／OOS／策略績效實驗，故主要結果與基準差異均為N/A；不得由顯示改善推論模型效果改變 |


### 3.83 模型研究流程跳過訊息彙總與工件路徑降噪（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_APPLICABLE`；只調整`[1/Enter] 模型研究與驗證`的使用者可見console，不改Dataset、模型、Score或策略結果 |
| 程式基準 | 來源`test-branch-1_20260802_160127_20d8492(1).zip`；SHA256 `b5096f340f71c67d6031d2ef7086dc080204095ab5fdc129e3d74e32650385f4` |
| 問題 | Dataset完整建立仍逐筆列出所有「有效資料不足」ticker，雖然進度本身覆寫單列，數十筆skip訊息仍造成洗版；模型流程亦反覆輸出使用者通常不會開啟的內部工件路徑與storage資訊 |
| Dataset輸出 | 取消逐ticker`[略過]`行；依原因累計並合併至唯一完成列，例如`跳過=59（有效資料不足=59）`。動態列數不再造成不同原因字串；compact模式下重複來源檔也只顯示一筆彙總；完整skip統計仍保存於dataset summary的`source_selection`供追蹤 |
| Compact console | 互動式模型研究流程啟用暫時compact console scope；所有共用`print_artifact_paths()`在此scope內不輸出，target／PIT console內另有的診斷路徑亦隱藏；scope結束後恢復原環境，直接CLI仍保留原工件路徑行為 |
| Workflow狀態 | Continuous-ranker狀態頁不再逐一列出8個檔案與路徑；改為`Dataset`、`Continuous Target`、`PIT Scores`、`PIT 模型驗證`四個聚合狀態，顯示`完整／不完整／缺少` |
| 重覆資訊 | Dataset compact流程不再於完成列後重覆輸出`storage=events/groups/dedup`；Dataset重建技術原因合併成單行使用者分類；已符合的Continuous Target不再重覆宣告；PIT已重用fold不逐筆列出，只於完成摘要顯示`重用／新建`；最終Dataset完成列已包含股票數、跳過彙總、events、groups與耗時 |
| 固定條件 | 不改Dataset schema既有欄位、feature、Label、Continuous Target、Seed 42、PIT fold、checkpoint、Score、模型gate、策略排序、交易或帳務；dataset summary只新增skip診斷欄位 |
| Dataset／Label／模型工件 | 不需因本修正重建；若使用者原本正在建立Dataset，套用新版後重新執行即可取得降噪輸出 |
| 驗證 | 新增skip原因正規化、彙總順序、rebuild reason單行化、current target靜默與compact工件路徑抑制契約；另獨立檢查compact scope恢復、狀態頁無路徑、PIT unavailable diagnostics無路徑、全專案語法／import／裸except／依賴循環及正式測試入口與checklist可信度 |
| 結果邊界 | 本輪沒有Selection、OOS或策略績效結果，顯示改善不得解讀為模型效果改變 |

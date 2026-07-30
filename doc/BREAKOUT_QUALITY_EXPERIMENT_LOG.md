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

本文件只記錄已知事實。歷史結果若缺少完整報表，會標記「精確值未保留」，不得自行補值。歷史資料整理截止日為 **2026-07-30**。

---

## 2. 目前基準

### 2.1 程式基準

| 項目 | 目前狀態 |
|---|---|
| 基準 ZIP | 本輪唯一來源ZIP `test-branch-1_20260730_171844_f457748.zip`，SHA256 `c93e7897a5c17f6fb3d8a4d0dd39a7a0da86a1f30f7ac1fc1f95a10ab62b1850`；11A完整target與actual Round-trip R audit已通過主要方向門檻，11B同日percentile regression及互動選單`[10]`已實作但尚未取得訓練結果；正式policy仍為9A `inception_time_v1`與filter id `breakout_quality_v1` |
| SHA256 | 本輪來源 ZIP：`c93e7897a5c17f6fb3d8a4d0dd39a7a0da86a1f30f7ac1fc1f95a10ab62b1850`；本地formal bundle `to_chatgpt_bundle_20260730_172006_97718afc.zip`：`63d850c44f9ce8e011c87f07bbf755f0eb8048422b44374c14a68c6bd6a4a684`；11A完整結果文件為使用者提供的`continuous_target_audit(2).md`；10A完整workflow結果來源為 `已貼上文字 (1)(21).txt`；9A分類結果來源仍為 `b6278b87f7ac045010d9799b4cab63d301be61ea4d5a0e99b84e4e6ae63983eb` |
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
| Seed | 42 |
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

---

## 5. 接下來要嘗試的列表

所有實驗一次只改一項。既有 OOS 可持續作為固定比較集；每次模型的訓練、Validation、early stopping 與 epoch 選擇必須完全限制在 Selection 內，完整 OOS 只能在模型凍結後執行。OOS 結果可以用來接受、淘汰或形成下一個實驗，不再以「OOS 已被查看」作為停止研究的理由。正式 runtime 仍維持 `base_finalists_agree` 既有排序且 Quality Ranking 關閉，除非新實驗同時通過模型指標與策略經濟效果。

### 目前新增優先：11B Strategy-aligned Daily Percentile Regression

| 項目 | 設計 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE`；研究訓練、報表與formal contract已實作，尚未執行完整模型訓練，不得預判有效 |
| 研究依據 | 11A target有效率98.04%、同日可排序與pair非Tie均100%；與actual realized R的Spearman為0.4044，Top target decile實際平均2.2060R、Bottom decile 0.0942R，且78.85%的≥2R贏家位於target上半部 |
| 唯一變更 | 網路仍為9A `inception_time_v1`與原2-logit head；將PASS softmax probability作0～1排序分數，以同日11A target percentile為監督目標，loss固定MSE |
| Profile | `strategy_aligned_daily_percentile_mse`；objective屬experiment profile，不新增architecture名稱，不改active 9A policy |
| Epoch選擇 | 只在Selection內，以Validation mean daily Spearman最大化選epoch；同分時才選較低Validation MSE。完整Selection依`selected_epochs`重新初始化重訓 |
| 防長尾／regime尺度漂移 | 不直接回歸raw R；每個日期各自以average rank轉成0～1 percentile，singleton固定0.5，不做跨split或OOS normalization |
| OOS邊界 | checkpoint寫入前不得建立或讀取OOS percentile target；OOS只在模型凍結後計算研究指標與actual-R方向診斷 |
| Runtime邊界 | research-only、沒有threshold、不覆蓋9A模型、不允許`export-scores --scope forward_oos`，也不進scanner或portfolio runtime |
| Dataset rebuild | 不重建feature bank、不relabel；嚴格讀取既有11A versioned arrays與manifest，hash或group contract不符即fail-fast |
| 判定 | 先看OOS mean daily/global Spearman、pair concordance、P@50／60／70與target top／bottom decile；再看model score對realized R Spearman、前後decile平均R及≥2R贏家保留率。通過後才做固定策略Score ranking比較 |

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
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE`；程式、CLI、研究工件與synthetic contract已完成，尚未執行完整訓練與OOS評估 |
| 程式基準 | 唯一來源ZIP `test-branch-1_20260730_171844_f457748.zip`，SHA256 `c93e7897a5c17f6fb3d8a4d0dd39a7a0da86a1f30f7ac1fc1f95a10ab62b1850`；此版已包含11B與互動選單`[10]`，本輪只修正式Checklist治理紀錄，不改模型、Dataset、target或runtime |
| Experiment profile | `strategy_aligned_daily_percentile_mse`；objective=`daily_percentile_regression`、continuous target=`strategy_aligned_opportunity_r_v1`、loss=`mse`、epoch metric=`mean_daily_spearman` |
| Architecture | 維持9A `inception_time_v1`、300×10 input、RF229與原2-logit head；score定義為`softmax(logits)[:, PASS]`，不新增architecture版本、不改checkpoint parameter shapes |
| 監督目標 | 對每個日期內的有效11A raw target採average rank並轉為`(rank−1)/(n−1)`；同值使用平均rank、單一候選日固定0.5。此轉換只依該日期事件，不使用其他日期或split統計 |
| 訓練與epoch | unique ticker/date sampling、Adam、MSE；Inner Train更新gradient，Validation只計算metrics。以Validation mean daily Spearman最大化選epoch，tie-break較低Validation MSE；之後重新初始化並以完整Selection重訓selected epochs |
| OOS防前視 | checkpoint完成寫入前只建立Selection percentile target；OOS percentile、OOS model metrics與actual-R診斷均在checkpoint凍結後才建立與執行，不參與loss、gradient、early stopping或epoch選擇 |
| 工件 | `models/.../inception_time_v1/strategy_aligned_daily_percentile_mse/model.pt`與manifest／split；`outputs/.../inception_time_v1/strategy_aligned_daily_percentile_mse/continuous_ranker_scores.csv`、report JSON／Markdown及group percentile array。Scores每個group唯一一列，Selection內以`selection_role`標示Inner Train／Validation |
| Runtime | `runtime_eligible=false`；不設threshold，不允許binary runtime artifact loader或forward-OOS export，不覆蓋9A `unique_group_sampling`工件 |
| Dataset／Label | 不重建Dataset、不relabel；嚴格驗證11A target manifest、檔案hash、group count與dataset policy。11A invalid groups不進loss或評估 |
| Formal契約 | B172／T269驗證profile與architecture分離、classification workflow拒絕research profile、同日percentile/tie/singleton、跨日期隔離、target hash fail-fast、2-logit checkpoint shape、research-only工件、OOS post-checkpoint順序、CLI註冊，以及互動選單`[10]`只路由至同一`train-continuous-ranker` command module |
| Formal double-check閉環 | 使用者於2026-07-30執行正式suite：quick gate、consistency、chain checks與ML smoke均PASS；meta quality只有`checklist_g_rows_require_actual_status_change`與`checklist_g_rows_sorted_by_date_then_id`兩項FAIL。根因為B172選單紀錄誤寫`DONE -> DONE`，且2026-07-30同日G區塊未依B／T namespace及數字段排序；已將選單新增與驗證拆成`DONE -> PARTIAL -> DONE`，並整段重排同日G列。此閉環只修改文件治理紀錄，不改11B程式或研究契約 |
| 下一步 | 可執行`python apps/breakout_quality.py`後選`[10]`，或直接執行`python apps/breakout_quality.py train-continuous-ranker --filter-id breakout_quality_v1`，取得完整Selection/OOS與realized-R結果；尚未取得結果前不得標記ACCEPTED或REJECTED |


---

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
→ 11B Strategy-aligned Daily Percentile Regression（IMPLEMENTED；RESULT_NOT_AVAILABLE）
```

任何新結果都必須追加至第 3 節，並同步更新第 2 節目前基準、第 4 節排除方向與第 5～6 節待辦順序。

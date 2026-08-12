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

本文件只記錄已知事實。歷史結果若缺少完整報表，會標記「精確值未保留」，不得自行補值。歷史資料整理截止日為 **2026-08-04**。

---

## 2. 目前基準

### 2.1 程式基準

| 項目 | 目前狀態 |
|---|---|
| 基準 ZIP | 本輪輸入基準為`test-branch-1_20260806_095542_dec6ccd.zip`；SHA256 `8327673830c90179f3639226729ef2b3b67110c3053ba7a5c39f998bd9bb6027`。Formal bundle為`to_chatgpt_bundle_20260806_095639_662f5f81.zip`；SHA256 `5f203a2a8d7fc96221b14807c823cc16e8e538f380c78f3ce1411105c53095a1`。本輪閉環修正forward score canonical write-path的architecture參數傳遞、continuous-ranker validator的canonical來源定位，以及coverage對Torch動態`_remote_module_*`來源的排除；Dataset、Label、模型、threshold、ROOS、策略執行與正式策略結果均未改變 |
| SHA256／最新結果 | 最新Binary OOS：PASS Precision 62.99%、原始PASS 55.63%、Precision Lift +7.35pp，但PASS Recall 51.44%、模型PASS 45.43%。A／B／C／F顯示只移除optional filters的C相較A總報酬+9.51pp、MDD−4.05pp、RoMD+2.96；F相較C總報酬+7.34pp，但MDD+4.95pp、RoMD−2.41、EV−0.11R、直接交易選擇差異−21.83R，且改善集中2023。現有9A hard filter不升級runtime；下一步在相同模型與threshold下逐步關閉歷史門檻、Re-entry及KC，分離DL本身與規則／出場交互作用。歷史Selection PIT與R2／R3結果保留供研究重現 |
| 程式版本範圍 | Active architectures為9A `inception_time_v1`排序／高品質基準與8F `multiscale_cnn_sequence_only_v1`高覆蓋基準；10A `inception_time_market_set_candidate_v1`與Global Stage 1 `inception_time_market_set_v1`均維持legacy read-only；9A-GN、9B、9C、9D、9E與9F同樣只供舊工件重建 |
| Policy 預設 | workflow architecture=`inception_time_v1`、filter id=`breakout_quality_v1`、experiment profile=`unique_group_sampling`、objective=`binary_classification`、scope=`all_labels`、threshold 0.5、Seed 42；正式策略mode自動解析為`hard-filter / canonical_runtime / original buy-sort`。Selection PIT continuous-ranker設定與工件保留，但只由其CLI研究入口使用；底層9A結構維持depth 6、kernels 39／19／9、RF 229 bars |
| 當前最佳實證模型 | 9A `inception_time_v1 / unique_group_sampling / threshold 0.5` 為新的排序／高品質模型基準；8F `multiscale_cnn_sequence_only_v1` 保留為高覆蓋基準 |
| Dataset | 沿用既有`breakout_quality_v1` 300×10 feature bank與固定百分比Label。使用者已於2026-08-04重新訓練`inception_time_v1 / unique_group_sampling`，產生新的model／split／manifest；research report、forward-OOS scores與正式策略比較仍待選單流程執行。10A Market Bank與其他legacy工件保留於獨立路徑供歷史重現 |

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
| 新增audit | `tools/audit/portfolio/score_ranking_capture.py`只讀既有Baseline／Score Sort transaction history、策略summary及post-replay selected-target diagnostics，逐筆重建entry到full exit lifecycle；不重播portfolio、不改candidate、Score、params、成交、帳務或optimizer |
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
| Console SSOT | 新增`core/console_report.py`，統一標題、段落、key-value、表格、狀態、ANSI-safe欄寬與工件清單；`apps/breakout_quality.py`狀態頁及主要build／train／audit／strategy報表共用，不再各自拼接不同格式 |
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

### 3.84 模型研究互動輸出第二次收斂（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING`；只調整互動式模型研究console與相應validator，完整Markdown／JSON及模型計算不變 |
| 程式基準 | `test-branch-1_20260802_163620_82699ea.zip`；SHA256 `e970a5ab1cb45ce2c1db837f864bdcf82bc7761b194012fff1b6252ddca09246` |
| Formal bundle | `to_chatgpt_bundle_20260802_163751_43e54221.zip`；SHA256 `48264151d484df2ed83cf26b7640d62f012b7d8a4f6f37e0c72f5a0c41ccdfa3` |
| 使用者實際輸出 | Dataset skip已正確彙總為61筆，但基礎11A target audit仍列4個split、No-time target再列3個split；7個PIT folds逐Epoch與refit共輸出數十行；模型audit又重覆Workflow identity並顯示年度、fold、分類、coverage及狀態六段 |
| Formal結果 | quick gate、chain checks、ml smoke通過；consistency只有`breakout_quality_model_research_menu_route`失敗；meta quality的唯一實質失敗為同一synthetic failure連帶 |
| Formal根因 | validator在`_dataset_refresh_step`回傳不需重建時仍要求console必須出現一個`[Dataset]`，與「沒有動作就不輸出」的compact契約衝突；程式路由本身仍正確執行Target→PIT→audit |
| Target輸出 | compact流程不再顯示基礎11A split audit；最終No-time Target只輸出valid coverage、Selection mean、AUC、source rho及OOS未評估的一行摘要 |
| Fold輸出 | compact流程保留一行PIT plan及一行執行環境；epoch selection與歷史refit不逐Epoch列印，每個新建fold只輸出best epoch、Validation rho、score groups及耗時一行；重用fold只在最終摘要計數 |
| 模型報表 | compact流程不再重覆Filter／Architecture／Profile／Seed等Workflow設定，也不顯示完整年度與fold分布表；只保留期間／coverage、PASS-only核心排序、年度正向比例與drift、分類重疊、orderable coverage、正式model gate及下一步。直接CLI、Markdown與JSON仍保留完整報表 |
| 固定條件 | 不改Dataset、Label、Continuous Target公式、PIT fold邊界、Seed 42、architecture、loss、epoch selection、refit、Score、模型gate、策略參數、交易或帳務 |
| Dataset／Label／模型工件 | 不需重建；本輪只改顯示。既有完整工件可直接重跑選單並由resume重用PIT folds |
| Selection結果 | 使用者輸出仍為PASS-only global rho 0.3074、daily rho 0.2370、spread 2.3455R、7/7年度rho與spread為正、drift=False、Score vs PASS AUC 0.5620；與3.71相同，無新模型實驗差異 |
| 採用判定 | 採用compact互動輸出；完整診斷移至既有Markdown／JSON，避免為可讀性刪除研究證據 |
| 下一步 | 套用後重跑`python apps/test_suite.py`作本機formal double check；策略層仍依既定順序執行`[2] 策略績效驗證` |

### 3.85 Selection Score-ranking策略參數適應與三組比較接線（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE / FORMAL_RERUN_PENDING`；完成Selection參數適應與Baseline／Sort Only／Adapted合併流程，尚未執行完整optimizer或宣稱Adapted有效 |
| 程式基準 | `test-branch-1_20260802_170248_5bafacd.zip`；SHA256 `d995acfd7b0396501f232107a259c3724562877f489072f39c997d6b325c9932` |
| Capture結果同步 | 既有3.78工件的正式read-only歸因為`ADAPTATION_DIAGNOSTIC_SUPPORTED`：平均曝險77.33%→54.20%、平均預留174,795.96→118,710.30、平均投入164,607.74→107,804.68、投入／預留93.36%→89.44%、初始停損6.39%→9.21%、持有41.94→44.66日、Realized R 0.28→0.35、Target R 0.99→1.15、投入資金報酬3.00%→3.87%。主要瓶頸為較寬停損使固定風險sizing縮小投入與曝險，符合啟動Selection參數適應的前置條件 |
| 入口 | `python apps/breakout_quality.py strategy-adapt`與主選單`[2] 策略績效驗證 → [2] 策略參數適應與績效比較`共用同一實作；原`[1/Enter] 比較目前策略`仍可獨立執行Baseline／Sort Only，不新增第三個比較選項 |
| 合併流程 | 先以相同固定風險與部位限制重建Baseline／Sort Only及capture gate，再固定Score Sort執行optimizer、選出`base-finalist-best`、回放Adapted，最後一次輸出三組績效、年度、Selection Target、capture與參數差異 |
| Search space | 沿用目前正式`BREAKOUT_OPTIMIZER_SEARCH_SPACE`及既有objective／inner validation／local-min finalist流程，不新增新策略參數或改range。`use_breakout_quality_ranking=True`與`use_breakout_quality_filter=False`由adaptation session固定，不能成為Optuna trial；Score權重、threshold、混合比例、模型超參數、PIT fold、Future Target及OOS均不可搜尋 |
| TP契約 | 完全沿用目前`OPTIMIZER_FIXED_TP_PERCENT=0.0`；第一輪固定關閉停利，不改為`None`，因此`tp_percent`不進trial search space |
| 固定執行條件 | 預設由config解析：trials 1000、fixed risk 0.01、max position cap 0.30、max positions 10、rotation off、Seed 42；這些值在單次adaptation內固定且不可由optimizer搜尋，但仍遵守PROJECT_SETTINGS的config可調整性 |
| PIT／optimizer identity | 首個trial前嚴格載入Selection PIT Score contract，並核對目前workflow凍結的filter／architecture／profile／continuous target／Seed／score start／PIT fold／inner validation；Score CSV、PIT manifest、PIT audit、Dataset、Selection期間、模型identity、search-space fingerprint與optimizer effective-policy fingerprint共同形成runtime identity。該identity寫入Optuna study user attrs、prep cache、full-evaluation cache、manifest與summary；已有trial但identity缺失或不一致時一律fail-fast，禁止續接舊study或跨PIT工件共用cache |
| Capture口徑 | 主報表與adaptation gate改為`Σ Realized R / Σ Target R`、逐筆ratio中位數與Target≥0.5R aggregate capture；原arithmetic mean只保留為JSON欄位`raw_mean_target_capture_ratio`，不得作主判定 |
| Adapted輸出 | `models/research/breakout_quality/score_ranking_adaptation/`輸出`adapted_best_params.json`、`adapted_manifest.json`、`optimizer_summary.json`、三組Markdown／JSON／CSV與Adapted replay／capture工件；前置Baseline／Sort Only隔離在`baseline_sort_only/`，不覆寫既有正式比較，只有runtime identity、相對路徑及全部必要工件SHA256一致時才沿用，任一檔案異動即重跑。相同identity重跑只補足尚缺trial，不重複追加完整trial數。Adapted Selection結果固定標記`FITTED_SELECTION_DIAGNOSTIC`，不代表泛化或正式採用 |
| OOS邊界 | 本輪只建立Selection fitting流程；Future Target只可在replay後離線join。只有Adapted params與Score模型、Score工件、排序規則、search space、fixed risk及部位限制全部凍結後，才可執行正式OOS；OOS不得回頭調整任何項目 |
| 驗證 | 已新增direct synthetic覆蓋固定ranking／filter不進trial、現行TP=0.0、目前模型／target／Seed／PIT fold凍結、PIT cache／study identity、缺少identity拒絕續跑、config固定值、前置工件隔離及identity／全工件hash重用、trial補足語意、Future Target／OOS隔離、合併選單／CLI、capture新口徑及`FITTED_SELECTION_DIAGNOSTIC`。正式`apps/test_suite.py`依專案規則未在本輪執行，須由使用者本機重跑 |
| 下一步 | 套用patch後先執行`python apps/test_suite.py`；通過後執行主選單策略驗證的`[2]`或CLI `python apps/breakout_quality.py strategy-adapt`取得第一輪Selection結果 |

### 3.86 Strategy Adaptation Formal Synthetic Session契約閉環（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING`；已修正使用者本機formal bundle檢出的synthetic session介面落差，尚待使用者重跑`apps/test_suite.py`確認正式雙重檢查 |
| 程式基準 | `test-branch-1_20260802_181238_754f580.zip`；SHA256 `0bd8af98f12ee5f23480771e8fae2b9c2e32c0e9a2c95f1bd4114f2cbd285e32` |
| Formal bundle | `to_chatgpt_bundle_20260802_181408_21f259c0.zip`；SHA256 `02d7523c26298cab2f95a4681707569dd4420c4290c56282ea5d3d5f4cf3ac72` |
| Formal結果 | quick gate、chain checks、ml smoke通過；consistency只有synthetic suite runtime失敗；meta quality的`coverage_synthetic_suite_runs_successfully`與`coverage_key_targets_hit`為同一中斷造成的連帶失敗 |
| 根因 | 3.85為正式`OptimizerSession`新增`apply_fixed_strategy_param_overrides()`、`optimizer_runtime_context()`與`runtime_cache_identity`，但`tools/validate/synthetic_strategy_cases.py`的`_FakeOptimizerSession`仍停留在舊介面，objective synthetic於首個override呼叫即`AttributeError`；不是optimizer objective、PIT identity、Score ranking、交易或帳務邏輯失敗 |
| 修正 | synthetic session同步正式最小介面：支援固定策略參數覆寫、runtime context及runtime cache identity；新增direct contract釘死三個hook，避免後續正式session擴充後測試替身再次落後 |
| 獨立驗證 | 直接受影響的optimizer synthetic共51項0失敗；完整synthetic consistency共4,126項、242 cases、0失敗、0 skip。另完成全專案AST／compile、import cycle、下層反向依賴`apps/`、bare except、正式入口及checklist可信度獨立檢查；依規定未執行`apps/test_suite.py` |
| 固定條件 | 不改strategy adaptation search space、objective、PIT Score／manifest／audit、Seed 42、TP=0.0、fixed risk、position cap、max positions、rotation、Score Sort、Future Target／OOS邊界、交易或帳務 |
| Dataset／Label／模型工件 | 不需重建Dataset、Label、Continuous Target、PIT Scores、模型checkpoint或策略比較工件；本輪只修正formal synthetic test double |
| Selection／OOS結果 | 無新增結果；Adapted仍為`RESULT_NOT_AVAILABLE`，不得由本輪測試修正推論有效性 |
| 下一步 | 套用patch後重跑`python apps/test_suite.py`；五步全PASS後再執行`python apps/breakout_quality.py strategy-adapt` |

### 3.87 策略績效驗證互動報表依決策層分區（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_APPLICABLE / FORMAL_RERUN_PENDING`；只重組`[2] 策略績效驗證`的互動式簡易報表，不改策略比較、capture、optimizer或任何績效數值 |
| 程式基準 | `test-branch-1_20260802_182132_7ce4eb8.zip`；SHA256 `e59e3e79fbaa9a9b306637595702ed4bb929bb4ec38803aaa9d2c8230c2d12ba` |
| 問題 | 原主表把投組報酬、單筆交易、資金曝險、候選供給與持倉缺口混在同一張表；capture audit又把部位、周轉、Target轉換、集中度與exit mix混為一表，使用者難以由結果建立因果鏈 |
| 互動入口 | `[2] 策略績效驗證`下的目前策略比較與策略參數適應均啟用既有compact console scope；直接CLI仍保留完整技術context及完整表格，Markdown／JSON／CSV內容與schema不變 |
| 策略比較簡表 | 先顯示四層綜合判定，再依`投組報酬與風險`、`單筆交易品質`、`資金使用與持倉容量`、`年度報酬`、`模型選股方向`及`判讀限制`分區；每區增加單行讀法，將「持倉格較滿但每格投入較小」與「模型Target排序改善但投組失敗」直接分層呈現 |
| Capture簡表 | 依`部位與資金配置`、`單筆交易品質與Target轉換`、`成交與資金周轉`、`進場集中度`、`最終出場結構`及`依進場年度`分區；年度寬表拆成交易品質與投入規模兩張窄表；無canonical產業資料時只顯示一行略過，不再列多個N/A欄位 |
| Adaptation簡表 | Baseline／Sort Only／Adapted三組比較同樣拆為投組、單筆、資金配置、Target轉換、年度、模型選股及參數差異，不再將全部指標塞入單一三組表 |
| 顯示降噪 | 互動簡表不顯示參數檔路徑、param selector、runtime members、ranking key等除錯context；保留期間、比較組別、Score source與無前視狀態。直接CLI仍可查看完整context |
| 固定條件 | 不改Dataset、Label、Continuous Target、PIT Score、Seed 42、active params、候選、排序、成交、資金、停損停利、accounting、capture公式、decision gate、optimizer search space或3.78既有結果 |
| Dataset／Label／模型工件 | 不需重建；只需套用程式後重新進入`[2] 策略績效驗證`即可看到新分區報表 |
| 驗證 | direct synthetic新增互動compact scope、分區標題、內部context抑制及完整CLI不受影響契約；相關capture與CLI validators獨立通過。正式`apps/test_suite.py`依專案規則留待使用者本機執行 |
| 結果邊界 | 本輪沒有新的Selection／OOS／Adapted績效結果，不得由報表重組推論策略效果改變 |

### 3.88 策略比較與Capture合併為單一一階報表（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_APPLICABLE / FORMAL_RERUN_PENDING`；只重構`[2] 策略績效驗證 → [1/Enter] 比較目前策略`的互動式compact console，不改策略、模型、Score、capture或任何績效數值 |
| 程式基準 | 來源`test-branch-1_20260802_191150_34070ba(1).zip`；SHA256 `517d655859da3a4043702edc999c91b3fb602a49dfe3bf3bd40c51e8618a95c7`；本輪修補ZIP只包含修改檔，SHA256列於交付回覆 |
| 問題 | 3.87雖已將主策略表與capture表各自分區，但互動流程仍連續輸出兩份報表，造成主報表讀完後又出現第二套`1～8`編號；相同指標亦可能跨兩份報表重覆出現，整體仍過於混雜 |
| 新資訊架構 | 互動Score-ranking比較只輸出一份報表，且只使用一階`1～10`：`投組報酬與風險`、`單筆交易結果`、`資金投入與部位大小`、`候選供給與持倉容量`、`模型選股能力`、`Target到實際報酬的轉換`、`資金周轉與進場集中`、`出場結構`、`年度結果與年度歸因`、`綜合判定、限制與下一步`；結論固定放最後 |
| 數據完整性 | 保留原策略主表、Selection選股診斷、capture主指標、成交／半倉／集中度、四類exit reason、完整年度報酬、依進場年度R／aggregate capture／投入金額及所有判讀；重覆指標只顯示一次。年度章節明確區分「權益曲線年度」與「交易進場年度」兩種統計口徑 |
| 顯示行為 | compact模式下不再追加第二份`Score Sort資金配置與Target Capture診斷`；`--capture-audit-only`在compact scope亦輸出同一份合併摘要。非compact直接CLI仍保留原兩份完整技術報表，Markdown／JSON／CSV schema與內容不變 |
| 指標說明 | 每個一階區塊先以單行定義其指標口徑；EV與平均Realized R因目前採相同R口徑仍保留，以對應兩份正式工件，但加註其數值通常相同；Future Target持續明示只在回放完成後join，未參與runtime或optimizer |
| 固定條件 | 不改Dataset、Label、Continuous Target、PIT Score、Seed 42、active params、ranking、候選生成、成交、風險sizing、停損停利、accounting、capture公式、decision gate、adaptation search space或3.78既有結果 |
| Dataset／Label／模型工件 | 不需重建；套用程式後重新進入策略比較即可看到新版單一報表 |
| 驗證 | direct synthetic新增10個一階章節、結論位於第10節、第二份capture標題不得出現在合併報表、內部參數檔路徑不顯示等契約；完整CLI與capture正式工件契約維持不變。正式`apps/test_suite.py`依專案規則留待使用者本機執行 |
| 結果邊界 | 本輪沒有新的Selection／OOS／Adapted結果；報表合併不得解讀為策略效果改變 |


### 3.89 Selection Rolling Score-ranking參數適應驗證改版（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE / FORMAL_RERUN_PENDING`；完成rolling adaptation程式與驗證契約，尚未執行完整optimizer或產生Adapted Rolling績效 |
| 程式基準 | `test-branch-1_20260802_200805_eb6d123.zip`；SHA256 `bb5f321a1eb3899d7588949d30f3e764824dc1cd8854efc9e990b4f4d17a7497` |
| 改版原因 | 3.85初版把Adapted設計為完整Selection單次1000 trials，而正式Baseline active params使用7個rolling folds、每fold `OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT=100`。兩者訓練預算、參數生效方式及擬合語意不對稱，Selection三組結果不能乾淨歸因於Score-ranking參數適應 |
| 新入口 | `python apps/breakout_quality.py strategy-adapt`與主選單`[2] 策略績效驗證 → [2] 驗證策略參數適應`共用同一rolling validation實作；`[1/Enter] 比較目前策略`維持獨立Baseline／Sort Only比較，不新增第三個手動比較選項 |
| Rolling對稱契約 | Adapted直接讀取正式Baseline `roos_base_best.json`，要求同一first／last OOS、120個月fixed train window、12個月OOS horizon、fold數、`base-finalist-best`及100 trials／fold；CLI、config與Baseline工件任一trial數不一致即fail-fast。第一階段不再引用`OPTIMIZER_SINGLE_FOLD_TRIALS_DEFAULT=1000`，也不產生完整Selection final-refit參數 |
| 唯一策略差異 | Adapted optimizer session固定`use_breakout_quality_ranking=True`、`use_breakout_quality_filter=False`與`selection_point_in_time` Score source；正式search space、objective、inner validation、local-min／finalist流程、TP=0.0、fixed risk、position cap、max positions與rotation沿用目前設定。Ranking、filter、Score threshold／權重、模型超參數、PIT fold、Future Target與OOS均不得成為trial維度 |
| Process-safe runtime | `run_outer_rolling_oos()`新增可選、JSON可序列化的`optimizer_session_spec`；主程序、fold worker及seed-ensemble worker都由同一spec重建固定策略覆寫、PIT runtime context與runtime cache identity。一般optimizer未傳spec時仍走原路徑，預設行為不變 |
| PIT／cache identity | Dataset、Baseline active params SHA256、rolling policy、search space、optimizer effective policy、filter／architecture／profile／target／Seed、PIT Score CSV／manifest／audit SHA256及固定runtime共同形成identity，並切分prep／full-evaluation cache；Adapted active params完成後再次驗證fold、trial與每個member的固定ranking／filter／risk／cap／TP契約 |
| Training Score coverage | 正式PIT Scores期間為2014-01-01～2020-12-31，而Baseline最早training window為2004-01-01～2013-12-31。新版不隱藏此差異：每fold輸出`rolling_training_score_coverage.csv`；PIT開始日前缺分依既有正式契約回退原buy-sort。以本輪正式Baseline 7 folds預檢：首個2014 fold為`bootstrap_fallback_only`，後6 folds皆為`partial_score_history`，訓練期間calendar coverage依序約10%～60%，沒有full-history fold。OOS必須完整位於PIT期間，PIT期間內若出現非預期Score coverage缺口則fail-fast |
| 三組結果 | 同一流程輸出Baseline（原排序＋原rolling params）、Sort Only（Score Sort＋原rolling params）及Adapted Rolling（Score Sort＋新rolling params），並產生績效、年度、capital／capture、Selection Target與參數分布比較；Adapted只可標記`ROLLING_SELECTION_DIAGNOSTIC`，不得宣稱正式泛化 |
| 正式工件 | `models/research/breakout_quality/score_ranking_adaptation/rolling_validation/`下輸出`rolling_preflight.json`、`rolling_training_score_coverage.csv`、`adapted_active_params/roos_base_best.json`、`rolling_optimizer_summary.json`、`rolling_adaptation_manifest.json`及三組Markdown／JSON／CSV／replay／capture工件；Baseline／Sort Only隔離在`baseline_sort_only/`，不覆寫既有正式比較 |
| 不包含項目 | 本輪不執行完整Selection單次1000-trial final refit、不產生`adapted_best_params.json`、不執行正式OOS、不改Score模型／PIT folds／search space／TP／fixed risk或排序規則。只有rolling診斷支持適應後，才另行設計final refit與凍結流程 |
| 獨立驗證 | 完整synthetic consistency共4,134 checks／242 cases／0失敗；另完成253個Python source的AST／compileall、7個修改模組import、import cycle、下層反向依賴`apps/`、bare except、唯一入口、config可調整性、PIT identity、Future Target隔離、Checklist B／T／G／E與`apps/test_suite.py`靜態可信度檢查。Direct contract覆蓋outer rolling trials來源、Baseline fold／trial mismatch拒絕、固定ranking／filter不進trial、一般search space與TP不變、process-safe session spec、PIT cache identity、bootstrap／partial Score-history coverage、PIT內gap拒絕、Adapted active-param固定契約、前置工件hash重用、合併menu／CLI及`ROLLING_SELECTION_DIAGNOSTIC`邊界；正式`apps/test_suite.py`依專案規則未在本輪執行 |
| 下一步 | 套用patch後先由使用者本機執行`python apps/test_suite.py`；五步全PASS後執行主選單`[2] → [2] 驗證策略參數適應`。本輪未取得實際rolling結果，不得預寫Adapted有效 |

### 3.90 Formal Meta Quality Checklist G排序閉環（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / FORMAL_RERUN_PENDING`；四個功能步驟已由使用者本機確認PASS，本輪只修正meta quality的Checklist G機械排序，正式suite待使用者重跑 |
| 程式基準 | `test-branch-1_20260802_204537_76279dd.zip`；SHA256 `fd03e8ac95378a2cdf0080d67754822f436ec9cb28132048b1f93fa9194279f0` |
| Formal結果 | quick gate、consistency、chain checks與ml smoke皆PASS；meta quality唯一失敗為`checklist_g_rows_sorted_by_date_then_id` |
| 根因 | `2026-08-02`同日區塊先列出T283，再補寫B186的`DONE -> PARTIAL -> DONE`狀態變更，違反日期升冪且同日依tracking ID排序的既有契約；主表、T摘要、transition內容及rolling adaptation程式本身均未失敗 |
| 修正 | 重排完整`2026-08-02` G區塊為B26、B170、B182、B183、B184、B185、B186、T280、T281、T282、T283；同ID多筆列維持原實際演進次序，並在B26補記本次治理契約退回與重新收斂 |
| 固定條件 | 不改Dataset、Label、Continuous Target、Seed 42、PIT Scores、rolling folds、100 trials／fold、search space、optimizer、Score Sort、候選、成交、資金、停損停利、capture、Selection／OOS邊界或任何績效數值 |
| Dataset／模型工件 | 不需重建；本輪只修改`doc/TEST_SUITE_CHECKLIST.md`與本實驗紀錄 |
| 獨立驗證 | 以獨立parser檢查G全表日期／natural tracking ID排序、同ID transition chain、NEW首筆、no-op、欄數與裸pipe；另核對主表／T／G最新狀態、摘要、唯一入口、AST／compileall、import cycle、反向依賴與bare except。正式`apps/test_suite.py`未由GPT執行 |
| 下一步 | 套用patch後重跑`python apps/test_suite.py`；預期meta quality不再回報Checklist G排序失敗。本輪沒有新的Adapted Rolling結果 |

### 3.91 Adapted Rolling固定參數跨Objective／OOS／Active-param輸出閉環（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / ADAPTED_RESULT_RERUN_PENDING / OPTIMIZER_SEARCH_REUSE_SUPPORTED`；7-fold optimizer搜尋已完成，但原逐fold OOS／OOS_CHAIN與active-param輸出遺失固定Score-ranking契約，因此不得採用；修正後若原工件仍在，可沿用已選trial而不重跑100 trials／fold |
| 程式基準 | `test-branch-1_20260802_205227_8906865.zip`；SHA256 `3e1c6f1e69d847eec0b92228056471e6417542302cf45ef9b3404e877b76ac73` |
| 使用者本機結果 | Baseline／Sort Only完整重播與3.78一致；Adapted Rolling共7 folds、每fold 100 trials，optimizer與local-min均完成，總時間約36分57秒；產生`adapted_active_params/roos_base_best.json`後，validator在2014-01-01 member讀到`use_breakout_quality_ranking=False`而中止 |
| 根因 | Optimizer objective先以`session.apply_fixed_strategy_param_overrides()`正確套用`ranking=True / filter=False`，所以trial評分與被選參數是在Score Sort下完成；但outer rolling的逐fold OOS diagnostics、finalist ensemble replay、policy schedule與active-param export再次只從原始`trial.params`重建完整參數，未套用session固定覆寫，缺少的欄位遂回到`V16StrategyParams`預設`ranking=False`。原validator最後正確擋下不一致，但擋得太晚 |
| 結果邊界 | 使用者輸出中的Adapted逐foldOOS、OOS_AVG、OOS_CHAIN與active-param replay均未使用固定Score Sort，不能作參數適應結論；已選trial的可搜尋策略欄位仍可保留，因其training objective確實在固定Score Sort context下評分。Baseline／Sort Only既有結果不受影響 |
| 修正 | `build_best_params_payload_from_trial()`支援持久化與明確傳入fixed overrides；objective把固定契約寫入trial user attrs。Outer rolling新增單一effective-param materialization，逐fold單trial OOS、finalist ensemble OOS、policy schedules、best-finalist params及所有active-param JSON均共用同一契約，不再由raw trial params分叉 |
| 既有搜尋復原 | `strategy-adapt`先核對上一版`rolling_preflight.json` runtime identity、rolling meta、fold數、trials與`roos_base_best.json`完整性；一致時直接將固定契約materialize到既有active-param工件並進行三組正式replay，不再追加另一輪100 trials／fold。Identity或工件不一致時才重新執行optimizer |
| 固定條件 | 不改Dataset、Label、Continuous Target、PIT Scores、Seed 42、7 folds、100 trials／fold、search space、objective、TP=0.0、fixed risk、position cap、max positions、rotation、Score Sort規則、Future Target或Selection／OOS邊界 |
| Dataset／Label／模型工件 | 不需重建；若本機仍保留本次`rolling_validation/rolling_preflight.json`與`adapted_active_params/roos_base_best.json`，套用patch後重跑`[2] 驗證策略參數適應`即可復原並完成三組比較 |
| 驗證 | T283新增effective trial params、policy schedule與active-param payload三層直接案例，確認`ranking=True / filter=False`跨objective、OOS與export一致；另保留最終active-param fail-fast。正式`apps/test_suite.py`依專案規則由使用者本機執行 |
| 下一步 | 套用patch後先執行`python apps/test_suite.py`；通過後重新執行主選單`[2] → [2] 驗證策略參數適應`。預期沿用既有7-fold搜尋，只重做固定契約materialization與Baseline／Sort Only／Adapted Rolling三組replay；在新結果產生前不得宣稱Adapted有效 |

### 3.92 策略比較與Rolling Adaptation互動簡表統一（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_APPLICABLE / FORMAL_RERUN_PENDING`；只統一互動式compact console資訊架構，不改optimizer、active params、Score、交易或任何績效數值 |
| 程式基準 | 來源`test-branch-1_20260802_221006_5d5a1ca.zip`；SHA256 `59d0dfcc9913dacc32ce4ebae20e6d084afc129f98db607f7bfeed736974ff8f`；本輪patch SHA256列於交付回覆 |
| 使用者要求 | `[1/Enter] 比較目前策略`與`[2] 驗證策略參數適應`的簡易報表須採一致格式；`[2]`獨有的Rolling訓練／Score coverage與active-param差異仍需保留 |
| 共通版型 | Adaptation compact改用與策略比較完全相同的前九個一階區塊與定義：投組報酬與風險、單筆交易結果、資金投入與部位大小、候選供給與持倉容量、模型選股能力、Target到實際報酬的轉換、資金周轉與進場集中、出場結構、年度結果與年度歸因 |
| 三組欄位 | 每個共通表格固定顯示Baseline、Sort Only、Adapted Rolling、`Adapted Rolling − Sort Only`適應差異與判讀；年度權益、進場年度R、aggregate capture及投入規模採同一欄位順序 |
| Adaptation專屬區塊 | 第10區保留Rolling folds、trials／fold、train window、OOS horizon、搜尋重用狀態及逐fold PIT Score coverage；第11區保留Baseline／Adapted active-param中位數與範圍；第12區才輸出綜合判定、限制與下一步，確保結論永遠位於最後 |
| 顯示降噪 | 互動compact模式不再在三組摘要後追加第二份`Score Sort資金配置與Target Capture診斷`；直接CLI非compact模式仍保留完整技術報表及既有Markdown／JSON／CSV工件 |
| 固定條件 | 不改Dataset、Label、Continuous Target、PIT Scores、Seed 42、7 folds、100 trials／fold、search space、objective、TP=0.0、fixed risk、position cap、max positions、rotation、Score排序、Future Target邊界或任何Selection結果 |
| Dataset／Label／模型工件 | 不需重建；若既有rolling結果工件完整，只需重新執行選單`[2] 驗證策略參數適應`即可看到統一後簡表 |
| 驗證 | T283改為直接驗證前九區標題及順序、共同定義、適應差異欄、Bootstrap coverage、參數差異、結論置末與第二份capture標題排除；正式`apps/test_suite.py`依專案規則留待使用者本機執行 |
| 結果邊界 | 本輪沒有重跑optimizer或portfolio replay，不得由顯示重構推論Adapted Rolling效果改變 |

### 3.93 Rolling Adaptation trials來源解耦Baseline歷史工件（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE / FORMAL_RERUN_PENDING`；移除Baseline歷史active-param工件trial數對目前Adapted Rolling設定的硬性限制，尚未重跑optimizer或產生新Adapted結果 |
| 程式基準 | `test-branch-1_20260802_224235_c249226.zip`；SHA256 `d0172528bd9423d1d5ea49cd50a878e0f598ec5367fdd31a6b2e4df5ab9b7123` |
| 問題 | 使用者將`config/training_policy.py`的`OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT`由100調為200後，`strategy-adapt`仍要求requested／config值等於既有Baseline `roos_base_best.json`保存的100，導致合法config調整被`ValueError`阻擋，違反config可調整性與「歷史工件不得鎖死目前設定」原則 |
| 新契約 | Adapted Rolling的`trials_per_fold`預設直接讀取目前`OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT`；明確CLI值仍可單次覆蓋。Baseline active-param工件的`meta.trials_per_fold`只保留為來源追溯與報表診斷，不再參與相等性fail-fast |
| 保留一致性 | Baseline工件仍提供first／last OOS、fixed train window、OOS horizon、fold schedule、active-param selector與歷史參數；Adapted工件必須忠實採用本次requested trials，既有Adapted搜尋只有runtime identity與本次trials相同才可重用，否則重新執行optimizer |
| Identity與輸出 | `rolling_preflight.json`、runtime identity、optimizer summary、三組比較payload與compact console同時保存本次Adapted trials、trial source及Baseline歷史artifact trials；`trial_count_match_required=false`，避免隱藏兩者差異但不再阻擋執行 |
| 固定條件 | 不改rolling fold日期、train window、OOS horizon、search space、objective、Seed 42、Score Sort、PIT identity、TP、fixed risk、position cap、max positions、rotation、Future Target或Selection／OOS邊界 |
| 結果邊界 | 若目前policy為200且既有Adapted搜尋為100，runtime identity不同，應重跑7 folds × 200 trials；Baseline／Sort Only仍使用既有歷史active params。此結果可比較策略效果，但報表會明確顯示Baseline工件與Adapted的trial預算不同，不再宣稱trial數相等 |
| 驗證 | T283使用Baseline artifact trial數故意不同於目前training policy的synthetic案例，確認Baseline contract可載入、outer rolling argv採目前requested值、runtime保存來源與舊artifact值、Adapted active-param只接受本次requested trials，且compact report顯示兩者而不作硬性gate。正式`apps/test_suite.py`依專案規則由使用者本機執行 |
| 下一步 | 套用patch後先執行`python apps/test_suite.py`；通過後執行主選單`[2] → [2] 驗證策略參數適應`。目前training policy為200時會執行200 trials／fold，不再因Baseline歷史工件為100而中止 |

### 3.94 Ranking × Parameter 2×2 Rolling適應驗證（2026-08-02）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `IMPLEMENTED / RESULT_NOT_AVAILABLE / FORMAL_RERUN_PENDING`；完成2×2對稱rolling實作，尚未執行兩套optimizer與四組Selection replay，不得預判哪種ranking較好 |
| 程式基準 | `test-branch-1_20260802_225828_9750ad1(1).zip`；SHA256 `c9acc89e37cfbed4fbc8c5c16d27656e8d69ecd47fd3c682b870bf66a7099e27` |
| 問題 | 原三組Baseline／Sort Only／Adapted Rolling中，Adapted Rolling同時改變ranking與active params；`Adapted − Baseline`及`Adapted − Sort Only`均無法乾淨判斷ranking優劣，因改善／惡化可能只是重新最佳化造成 |
| 新比較矩陣 | Baseline＝原ranking＋原歷史params；Sort Only＝Score ranking＋同一原歷史params；Baseline Adapted＝原ranking＋重新rolling最佳化params；Score Adapted＝Score ranking＋重新rolling最佳化params |
| Ranking-only判定 | 固定原歷史params，只比較`Sort Only − Baseline`；此差異仍是ranking本身的純控制實驗 |
| Optimized-system判定 | Baseline Adapted與Score Adapted各自執行optimizer，但必須共用相同fold schedule、trials／fold來源與值、search space、objective、Seed、sampler、TP、fixed risk、position cap、max positions及rotation；唯一固定差異為ranking False／True。以`Score Adapted − Baseline Adapted`判斷兩套完整最佳化系統 |
| 其他差異用途 | `Baseline Adapted − Baseline`只代表原ranking refit效果；`Score Adapted − Sort Only`只代表Score ranking條件下的參數重調效果；兩者不得代替ranking判定 |
| 工件隔離 | 新增`baseline_adapted/`與`score_adapted/`兩個獨立arm，各自保存preflight、runtime identity、optimizer runtime、active params與summary；prep／evaluation cache及study不得跨arm共用。Baseline／Sort Only read-only工件仍隔離在`baseline_sort_only/` |
| 四組回放 | 兩套重調active params完成後，分別以原ranking與Score ranking回放，輸出四組績效、年度、Selection Target、capital／capture、active-param分布與兩種對稱差異；optimized capture audit固定比較Baseline Adapted與Score Adapted |
| Console | 沿用既有前九個共同區塊；每區先顯示「固定原參數：純Ranking效果」，再顯示「各自Rolling重調：完整系統效果」。第10區保留fold／trials／PIT Score coverage與兩臂search reuse，第11區比較兩個重調arm的參數中位數／範圍，第12區分別輸出Ranking-only與Optimized-system判定 |
| Trials契約 | 兩個重調arm的trials／fold都從目前`OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT`或同一CLI override取得，彼此必須相同；Baseline歷史工件舊trial數仍只作追溯，不作hard gate |
| 固定邊界 | 不改Dataset、Label、continuous ranker、PIT Scores／folds、Seed 42、search space、objective、TP、risk／position限制、Score Sort規則、Future Target post-replay only或Selection／OOS邊界；不執行完整Selection final refit或正式OOS |
| 驗證 | T283改為驗證兩個相反ranking固定arm、共同非ranking搜尋維度與預算、runtime/cache identity分離、兩套active-param固定契約、四組報表、兩種對稱判讀與舊Baseline／Sort工件hash重用。正式`apps/test_suite.py`依規定由使用者本機執行 |
| 下一步 | 套用patch後先跑`python apps/test_suite.py`；通過後執行主選單`[2] → [2] 驗證策略參數適應`。首次執行需完成Baseline Adapted與Score Adapted兩套rolling optimizer；結果產生前維持`RESULT_NOT_AVAILABLE` |
## 2026-08-03 — Ranking×Parameter 2×2改為單一Score Adapted訓練，並修正diagnostics runtime context

### 狀態

`IMPLEMENTED / RESULT_NOT_AVAILABLE`

本輪只完成程式、契約與獨立synthetic驗證；未執行正式rolling optimizer、Selection四組績效或正式OOS，不得預先宣稱Adapted有效。

### 問題

1. 前版把2×2誤實作為`Baseline Adapted`與`Score Adapted`兩套rolling optimizer。既有正式ROOS本身已是舊ranking的rolling結果，再訓練一次舊ranking既浪費時間，也把正式基準換成另一批受trial budget／sampler影響的參數。
2. Score Adapted的optimizer objective使用正式Selection PIT context，但fold trial完成後的OOS diagnostics未進入同一runtime context，process內設定退回`unique_group_sampling`並尋找錯誤manifest。通用sequential fallback無法修正deterministic identity錯誤，反而再次浪費執行時間。
3. Outer rolling預設memory study；程序失敗後已完成trials不具跨程序恢復能力。

### 修正後設計

只訓練一套新參數：

| 組別 | Ranking | Active params | 新訓練 |
|---|---|---|---:|
| Baseline | 舊ranking | 既有正式ROOS | 否 |
| Sort Only | Score ranking | 同一套既有正式ROOS | 否 |
| Param Only | 舊ranking | Score Adapted新參數 | 否，只回放 |
| Adapted | Score ranking | 同一套Score Adapted新參數 | 是，唯一optimizer |

正式歸因：

- `Sort Only − Baseline`：舊參數下純ranking效果。
- `Adapted − Param Only`：同一套新參數下純ranking效果。
- `Param Only − Baseline`：舊ranking下參數替換效果。
- `Adapted − Sort Only`：Score ranking下參數適應效果。
- `Adapted − Baseline`：候選新系統相對正式Baseline的整體效果。

### 流程修正

- `[2] 驗證策略參數適應`只載入並驗證`[1] 比較目前策略`已保存的Baseline／Sort Only正式工件；缺少或identity過期時要求先執行`[1]`，不在`[2]`暗中重跑舊參數replay。
- 移除正式流程對`baseline_adapted/`的依賴；既有目錄可保留為research-only歷史工件，但不再讀取或重跑。
- Score Adapted只有一個runtime identity、一套optimizer輸出與一套active params；Param Only與Adapted直接共用該active-param schedule，只切換ranking。

### Runtime context與錯誤分類修正

- 在rolling trials開始前，以正式`filter_id`、architecture與experiment profile解析並驗證model manifest；不存在、無法解析或profile不符時立即fail-fast。
- 平行worker建立session後先驗證序列化runtime context可重建。
- fold OOS diagnostics及OOS_CHAIN replay全部包在同一`session.optimizer_runtime_context()`。
- adaptation專用環境設定同步寫入父process環境，確保Windows spawn worker繼承正式`V16_MODELS_DIR`、study storage及resume設定。
- runtime identity／manifest錯誤標記為`NON_RETRYABLE_RUNTIME_IDENTITY_ERROR`，直接中止；不再啟動無效sequential fallback。

### Study持久化與續跑

- Score Adapted強制使用SQLite study。
- DB名稱由runtime identity、OOS fold與seed member決定，同一identity重跑會接續同一study。
- 只補足`requested trials − existing trials`。
- study保存並檢查effective policy、runtime cache identity與固定策略覆寫；不一致時禁止續跑。
- 先前已失敗且使用memory storage的程序，若已結束，已完成trials無法由本修正事後恢復；本機下一次須重新建立Score Adapted搜尋。此限制不影響修正後未來中斷的SQLite續跑。

### 結果邊界

- 完整Selection final refit：未執行。
- 正式OOS：未執行。
- Future Target：仍只允許post-replay join，不得進runtime、optimizer或cache key以外的決策路徑。
- `baseline_adapted`既有結果：不納入正式2×2比較。


## 2026-08-03 — Score Adapted preflight改驗證Selection PIT工件

### 狀態

`IMPLEMENTED / RESULT_NOT_AVAILABLE / FORMAL_RERUN_PENDING`

本輪只修正rolling trials開始前的工件preflight；未執行optimizer、四組Selection replay或正式OOS，不得預判Adapted有效。

### 問題

前版為避免diagnostics退回`unique_group_sampling`，新增了full-model profile根目錄`manifest.json`檢查。但`selection_point_in_time` ranking runtime不載入full-model checkpoint／manifest，只讀取Selection PIT score table。正式continuous-ranker profile可具有完整PIT Scores、PIT manifest及PIT模型驗證，而不需要存在：

`models/filters/breakout_quality/<filter_id>/<architecture>/<experiment_profile>/manifest.json`

因此使用者工件狀態全部完整，仍在任何rolling trial開始前被錯誤`FileNotFoundError`阻擋。

### 根因

`strategy_adapt._validate_formal_model_manifest()`錯用了binary／canonical filter runtime的`resolve_filter_artifact_paths()`。真正Selection PIT ranking依賴已由`load_selection_point_in_time_ranking_contract()`驗證：

- `selection_point_in_time_scores.csv`
- `selection_point_in_time_manifest.json`
- `selection_point_in_time_audit.json`
- coverage工件與Continuous Target綁定
- filter／architecture／experiment profile identity
- score檔SHA256與size
- PIT模型驗證gate

### 修正

- 移除full-model profile根目錄manifest要求。
- 新增Selection PIT runtime preflight，只接受已載入且identity一致、模型驗證為PASS的PIT contract。
- `rolling_preflight.json`與最終adaptation manifest改保存PIT score／manifest／audit相對路徑及SHA256。
- 缺少PIT工件、PIT identity不一致或PIT模型驗證未通過仍在trials前fail-fast。
- diagnostics／OOS_CHAIN runtime context、SQLite resume、不可重試錯誤分類及單一Score Adapted訓練四組回放契約不變。

### 固定條件與結果邊界

不改Dataset、Label、Continuous Target、PIT Scores、模型、Seed 42、fold schedule、trials、search space、objective、TP、fixed risk、position cap、max positions、rotation、ranking規則、交易／帳務、Future Target post-replay only或Selection／OOS邊界。Dataset／Label／PIT工件不需重建。

### 驗證

T283改為直接建立三個PIT runtime工件，驗證正常contract可通過、缺少score檔與錯誤profile會於rolling前拒絕；不再建立或要求full-model`manifest.json`。正式`apps/test_suite.py`依專案規定由使用者本機執行。

## 2026-08-03 — Baseline／Sort Only正式比較工件identity相容修正

### 狀態

`IMPLEMENTED / RESULT_NOT_AVAILABLE / FORMAL_RERUN_PENDING`

本輪只修正`[1] 比較目前策略`與`[2] 驗證策略參數適應`之間的工件identity與重用契約；未執行Score Adapted optimizer、四組Selection replay或正式OOS。

### 程式基準

- ZIP：`test-branch-1_20260803_014347_0235c8f.zip`
- SHA256：`0c05fbb9dcd3508648a3a280824c2ee8334e15f34e0dba2aafd20bcffd3e6f3c`
- 全新解壓：本輪隔離暫存檢查目錄（不納入正式工件identity）

### 問題

使用者已完成`[1] 比較目前策略`，但`[2]`仍回報找不到identity一致的正式比較工件。根因是互動式`[1]`沒有明確傳入第一輪adaptation固定的`fixed_risk=0.01`與`max_position_cap_pct=0.30`，因此strategy comparison metadata的override欄位為`null`；`[2]`只用override欄位與目前固定值做嚴格比對，未檢查舊ROOS回放實際active params已是相同值。舊`adaptation_pair_manifest`亦被錯當成來源真理，可能在`[1]`重跑後反向阻擋新正式工件。

### 修正

- 互動式`[1]`現在明確傳入與`[2]`相同的fixed risk及position cap，後續新工件metadata直接一致。
- 既有`[1]`工件若override欄位為`null`，改由Baseline與Sort Only兩組實際參數payload逐項確認`fixed_risk`及`max_position_cap_pct`全都等於目前固定值；符合時直接沿用，不要求重跑。
- 若任一effective param不同或缺少可驗證值，仍拒絕載入。
- `adaptation_pair_manifest.json`降為`[2]`可重建的衍生索引；正式comparison payload、PIT identity與必要CSV／JSON／Markdown工件均通過後，manifest會依目前identity與實際SHA256重新寫入。
- identity錯誤訊息改列出具體不一致欄位，避免只有泛化的「請先執行[1]」。

### 固定條件與結果邊界

不改Dataset、Label、Continuous Target、PIT Scores、模型、Seed 42、rolling folds、trials、search space、objective、ranking、TP、risk值、position cap值、max positions、rotation、交易／帳務、Future Target post-replay only或Selection／OOS邊界。既有`[1]`工件若effective params符合條件即可直接重用。

### 驗證

T283新增：舊metadata未明確override但effective params一致時可載入；任一實際risk錯值時拒絕；stale adaptation manifest在正式工件通過後可刷新；互動式`[1]`argv明確攜帶共同risk／cap。正式`apps/test_suite.py`依專案規定由使用者本機執行。

## 2026-08-03 — Score Adapted PIT models root與active-param輸出目錄分離

### 狀態

`IMPLEMENTED / RESULT_NOT_AVAILABLE / FORMAL_RERUN_PENDING`

本輪只修正Score Adapted rolling的路徑解析與不可重試分類；未執行optimizer、四組Selection replay或正式OOS，不得預判Adapted有效。

### 程式基準

- ZIP：`test-branch-1_20260803_015534_cc59543.zip`
- SHA256：`488db8f18f43b322564cbee0810c17fdf033303e1b3dcbe31267c6d659965ca0`
- 全新解壓：本輪隔離暫存檢查目錄（不納入正式工件identity）

### 問題

Score Adapted為了把`roos_*.json`隔離到`rolling_validation/score_adapted/active_params/`，將通用`V16_MODELS_DIR`改指向該輸出目錄。但Breakout Quality Selection PIT路徑解析同樣使用`V16_MODELS_DIR`作正式模型根目錄，導致worker錯誤尋找：

`models/research/.../score_adapted/active_params/filters/breakout_quality/.../point_in_time/selection_point_in_time_scores.csv`

而不是正式：

`models/filters/breakout_quality/.../point_in_time/selection_point_in_time_scores.csv`

因此7個fold均在trial 0附近失敗；原不可重試分類亦未涵蓋`找不到Selection PIT score`，通用流程又進入不可能成功的sequential fallback。

### 修正

- 由preflight已驗證的Selection PIT score路徑反推出正式models根目錄，Windows spawn worker的`V16_MODELS_DIR`只指向該正式根目錄。
- `run_outer_rolling_oos()`新增可選`paramset_models_dir`，只負責最終`roos_*.json`輸出；Score Adapted明確傳入arm的`active_params/`，不再借用`V16_MODELS_DIR`。
- Objective、fold diagnostics與OOS_CHAIN仍透過同一Selection PIT runtime context取得architecture／profile／score source。
- `找不到Selection PIT score／manifest／audit`及PIT runtime identity不一致加入不可重試分類，直接中止，不再進sequential fallback。
- SQLite study identity、單一Score Adapted訓練與Baseline／Sort Only／Param Only／Adapted四組回放契約不變。

### 既有執行工件

本次錯誤發生時各fold顯示trial 0／200，沒有可用完成trial。套用修正後可直接重新執行；若SQLite DB內已有合法同identity trials，既有resume契約仍只補足剩餘trials。舊`baseline_adapted/`工件仍不納入正式流程。

### 固定條件與結果邊界

不改Dataset、Label、Continuous Target、PIT Scores、模型、Seed 42、fold schedule、trials、search space、objective、TP、fixed risk、position cap、max positions、rotation、ranking規則、交易／帳務、Future Target post-replay only或Selection／OOS邊界。Dataset／Label／PIT工件不需重建。

### 驗證

T283新增正式PIT models root解析、active-param獨立輸出、Selection PIT缺檔不可重試及source wiring案例。正式`apps/test_suite.py`依專案規定由使用者本機執行。


## 2026-08-03 — Ensemble partial Score availability保守fallback修正

### 狀態

`IMPLEMENTED / RESULT_NOT_AVAILABLE / FORMAL_RERUN_PENDING`

本輪只修正Score Adapted fold diagnostics的ensemble Score聚合與錯誤重試分類；未完成7-fold rolling、四組Selection replay或正式OOS，不得預判Adapted有效。

### 程式基準

- ZIP：`test-branch-1_20260803_021601_c5bd4b2.zip`
- SHA256：`cbef176743f6351bc6fa388ad6b1000cff37924861244d41b662a096c60c1998`
- 全新解壓：本輪隔離暫存檢查目錄（不納入正式工件identity）

### 問題

使用者本機Score Adapted執行至fold diagnostics時，ticker `00655L`的ensemble共識members出現部分有Selection PIT Score、部分缺分，舊`_aggregate_ensemble_candidate_rows()`要求所有member availability完全一致，因此在`build_daily_ensemble_candidates`中拋出`同一 ticker 的 ensemble members Score availability不一致`。平行流程又把此deterministic契約錯誤當一般process失敗，進入必然同樣失敗的sequential fallback。

### 根因

ensemble以ticker聚合，但不同參數member可能在同一交易日承接不同原始normal／continuation／re-entry訊號，因此`score_date`可以不同。Selection PIT Score依`ticker／原始score_date`查詢；不同原始事件日期一邊有分、一邊缺分是合法partial availability，不代表同一PIT key回傳矛盾。

### 修正

- 只有全部共識members均有有效Score時，聚合候選才使用member Score中位數參與同票ranking。
- 若不同原始score date造成partial availability，保留候選、保存完整`ensemble_member_quality_rank_by_key`，聚合rank標記`available=false`與`partial_ensemble_member_score_availability`，回退既有buy-sort；不得填0或排除候選。
- 若全部members均缺分，維持既有缺分fallback。
- 若同ticker且同一score date仍同時出現available與unavailable，視為真正資料／runtime契約分叉並fail-fast。
- 聚合結果新增available／unavailable member數量、keys及`all_available / partial_available_fallback / none_available`診斷欄位。
- deterministic ensemble ranking／Score source／score payload契約錯誤加入不可重試分類，不再啟動sequential fallback；一般process pool失敗仍保留fallback。

### 既有執行與續跑

本次已有4個fold完成diagnostics，其餘fold的SQLite studies已持久化接近或完成200 trials。套用修正後以相同runtime identity重新執行，應沿用既有studies、只補足缺少trials並重做未完成diagnostics；目前中斷輸出的OOS_AVG與部分fold表不得作正式結論。

### 固定條件與結果邊界

不改Dataset、Label、Continuous Target、PIT Scores、模型、Seed 42、fold schedule、trials、search space、objective、TP、fixed risk、position cap、max positions、rotation、Score排序鍵、交易／帳務、Future Target post-replay only或Selection／OOS邊界。單一Score Adapted訓練與Baseline／Sort Only／Param Only／Adapted四組回放契約不變。

### 驗證

T267新增不同score date partial availability保留候選、回退原排序、逐member rank保存及同score date分叉fail-fast案例；T283新增deterministic ensemble契約錯誤不進sequential fallback案例。正式`apps/test_suite.py`依專案規定由使用者本機執行。

## 2026-08-03 — Ranking × Parameter正式2×2結果與capital-aware ranking下一步

### 狀態

`RESULT_AVAILABLE / SCORE_ADAPTED_PARAMS_REJECTED / ADAPTED_SYSTEM_REJECTED`

本節依使用者提供的完整正式輸出摘要回寫。原始`rolling_validation/`結果工件與完整console輸出未包含於本ZIP，故數值視為已提供的正式結果紀錄，本輪未重新執行optimizer或portfolio replay。

### 程式基準

- ZIP：`test-branch-1_20260803_122159_172cc87(1).zip`
- SHA256：`5c8653168fe3e42d532275466e18e6e0fce839ae08adf35ecd09c18a518f9f5a`
- 比較期間：2014-01-01～2020-12-31 Selection PIT
- 正式2×2：Baseline＝舊ranking＋舊ROOS；Sort Only＝Score ranking＋舊ROOS；Param Only＝舊ranking＋同一套Score Adapted params；Adapted＝Score ranking＋同一套Score Adapted params

### 固定條件

Dataset、Continuous Target、Selection PIT Scores、PIT模型驗證、Score來源、候選filters、entry／stop／exit、fixed risk、position cap、max positions、rotation、交易帳務與Future Target post-replay-only契約不變。新參數只由Score-ranking rolling optimizer訓練一次；Param Only與Adapted使用完全相同的active-param schedule。

### 主要結果

| 指標 | Baseline | Sort Only | Param Only | Adapted |
|---|---:|---:|---:|---:|
| 淨總報酬 | 182.62% | 144.80% | 65.46% | 59.71% |
| 最大回撤 | 13.18% | 21.53% | 14.57% | 12.64% |
| Return／MDD | 13.86 | 6.73 | 4.49 | 4.72 |
| 平均曝險 | 77.33% | 54.20% | 73.21% | 50.69% |
| 平均初始停損距離 | 6.39% | 9.21% | 6.36% | 10.46% |
| 平均實際投入金額 | 164,608 | 107,805 | 121,640 | 65,584 |
| 平均Realized R | 0.28R | 0.35R | 0.34R | 0.38R |
| 選中候選Target mean | 1.0868R | 1.2079R | 0.7571R | 1.2178R |

補充：新參數使交易數由Baseline 527筆降至Param Only 355筆，平均持有日由41.94日升至59.25日。新參數本身即顯著劣於舊正式ROOS。

### 歸因與判定

- `Sort Only − Baseline`與`Adapted − Param Only`均顯示Score ranking提高Future Target與單筆Realized R，但降低資金投入、平均曝險及總報酬。
- 共同機械鏈為：Score ranking偏好較寬初始停損候選；fixed-risk sizing下每股風險提高，買入股數與每slot投入金額隨之下降，平均曝險約降低22～23個百分點。
- 現行No-time Target描述未來價格機會，未直接納入實際初始停損距離、固定風險後部位金額、剩餘現金或正式entry／stop／exit capture效率。因此「排序能力PASS」不等於「投組經濟效果PASS」。
- PIT模型排序能力與較高Future Target選擇維持`PASS`；舊參數Score ranking、Score Adapted params與完整Adapted系統均為`REJECTED`；正式策略維持Baseline。

### 下一個單一研究方向

優先建立capital-aware ranking／停損距離消融的正式實驗契約，先固定舊正式ROOS與全部交易規則，只改候選排序：原ranking、原始Score ranking、Score加停損距離約束、停損距離分桶後桶內Score ranking。第一階段只做程式契約與可回放診斷，不重訓模型、不重新執行rolling optimizer，也不直接提高fixed risk、position cap或max positions。

後續第二順位為Filter × Score分組消融；第三順位才是Selection PIT Score向前回補可行性。現有7 folds training calendar coverage為0%、10%、20%、30%、40%、50%、60%，目前不足以單獨證明coverage是Adapted失敗主因。



## 2026-08-03 — Selection PIT向前延伸與coverage提升後共同folds 2×2

### 狀態

`RESULT_AVAILABLE / COVERAGE_IMPROVEMENT_CONFIRMED / COVERAGE_NOT_PRIMARY_FAILURE / SCORE_ADAPTED_PARAMS_REJECTED / ADAPTED_SYSTEM_REJECTED`

使用者已在本機完整資料完成Selection PIT向前延伸、PIT audit、Score Adapted rolling optimizer及四組2×2 replay。實際PIT起點由`2014-01-01`向前延伸至`2011-01-01`；7個Baseline共同fold全部保留，weighted training calendar coverage由30.0%提高至60.0%，每fold均提高30.0pp。coverage提升目標已達成，但Score ranking在舊參數與新Adapted參數下仍明顯降低投組報酬，因此coverage不足不再列為主要失敗原因。

### 程式基準

- 輸入ZIP：`test-branch-1_20260803_184631_c665cb2(1).zip`
- SHA256：`e75d9295d50b117f71ac6541a75e26937db822ba9f886cd33f82e65fe7474e9e`
- 本輪修改只以該ZIP全新解壓內容為基準；交付patch ZIP的SHA256以本輪回覆列示。

### 唯一主要變更

PIT builder既有`auto`最早合法日期、日期型fold identity與舊fold安全重用維持不變。Strategy Adapt的coverage門檻由「只保留100% coverage尾段folds」改為「保留全部原Baseline共同folds，延伸後training coverage必須相較舊正式`2014-01-01`參考起點提升」。此變更只調整rolling比較樣本與coverage判定，不改模型架構、Target公式、loss、Seed、fold長度或策略交易規則。

### 實作內容

1. PIT fold ID維持穩定日期鍵`fold_YYYYMMDD_YYYYMMDD`；向前新增fold不會改變既有期間identity。
2. Builder resume仍只在日期、information cutoff、資料來源、模型規格、training settings、group／event counts、lookahead契約與checkpoint／Score SHA256完全一致時，遷移或重用舊fold。
3. `score-start-date=auto`仍由Dataset、Continuous Target、PASS-only scope、`label_eval_end_date`、24個月Validation與最小group門檻解析最早合法月份；`plan-only`只輸出日期與fold計畫，不訓練或寫入正式Score。
4. 正式build完成後重建合併Scores、coverage CSV、manifest與PIT audit；既有相容fold可重用，新增較早fold必須實際訓練。
5. 新增`BREAKOUT_QUALITY_POINT_IN_TIME_COVERAGE_REFERENCE_START_DATE="2014-01-01"`作為舊正式coverage參考，不限制實際PIT起點。每個Baseline training window同時計算reference與actual calendar coverage、差值及狀態。
6. 正式2×2保留全部Baseline rolling folds與原OOS期間。actual coverage必須在所有fold不低於reference，至少一個fold嚴格改善，且加權總coverage提高；實際PIT起點之前允許回退原buy-sort，實際PIT期間內缺口仍禁止。Baseline、Sort Only、Param Only與Adapted使用相同fold schedule、期間、PIT identity、risk、position cap、max positions、rotation及帳務契約。

### Dataset／Target／模型工件需求

- Dataset：不因程式變更重建；本機必須存在目前完整`breakout_quality_v1` Dataset及與其SHA256一致的Continuous Target。
- Label／Continuous Target：公式與arrays不變，不relabel；若本機Target因Dataset更新而stale，仍由既有workflow先重建。
- PIT模型：最早合法日期到既有2014年前的fold需要新訓練；2014後舊fold只有在完整契約與hash完全一致時才遷移重用。
- PIT audit：合併Scores／manifest改變後必須重新執行，舊audit因來源SHA256不同不得沿用。
- Strategy optimizer：只在PIT audit通過、全部OOS位於PIT期間、逐fold coverage未退步且整體coverage提升後執行；不再因未達100%而排除fold或中止。

### 固定條件與結果邊界

不改9A架構、No-time Target、daily percentile regression、PASS-only scope、Seed 42、12個月PIT fold、24個月inner validation、optimizer search space、trials來源、fixed risk、position cap、max positions、rotation、原始Score ranking、entry／stop／exit、portfolio accounting或Future Target post-replay-only契約。此輪不判定新ranking有效或無效；先前2014～2020 Baseline／Sort Only與原混合coverage 2×2結果仍保留為歷史證據，新的coverage提升版2×2必須待本機完整執行後另行回寫。

### 本機正式執行順序

以正式選單操作：

1. 執行`apps/breakout_quality.py`，主選單按`Enter`進入「模型研究與驗證」，確認建立或更新Dataset、Continuous Target、向前延伸PIT Scores並完成PIT模型驗證。
2. 返回主選單選`[2] 策略績效驗證`。
3. 在策略績效驗證選`[2] 驗證策略參數適應（PIT coverage提升即可）`，由流程自動完成共同fold coverage preflight、Baseline／Sort Only重建或重用、Score Adapted rolling optimizer及四組2×2 replay。

### 獨立驗證

- PIT builder既有日期型fold identity、auto最早合法日期、舊fold遷移與resume契約維持，未修改其模型／資料邏輯。
- Strategy Adapt新增獨立synthetic契約：actual起點2007相較reference起點2014時，四個Baseline folds全部保留、逐foldcoverage均改善、共同比較期間維持2014～2018；actual與reference相同時必須拒絕。
- Coverage報表同步輸出reference、actual、提升pp、加權coverage gain與各fold狀態；runtime identity明確記錄pre-PIT fallback允許、100%非必要及PIT期間gap禁止。
- 上述程式契約已由使用者本機完整資料執行並取得以下正式結果；正式`apps/test_suite.py`是否已於本機通過，使用者本輪未另提供bundle。

### 本機正式結果（coverage提升後）

- 程式基準：`test-branch-1_20260803_210453_6f421c1(1).zip`
- SHA256：`38d4f9799b89ebcaa4720d8ef5ae033917d5a68100c5723712502cd0a965a5a1`
- 比較期間：2014-01-01～2020-12-31
- PIT actual start：2011-01-01
- Coverage reference start：2014-01-01
- Actual／reference weighted coverage：60.0%／30.0%，提升30.0pp
- Rolling folds：7；每fold coverage均提升30.0pp
- Adapted trials／fold：300；Baseline歷史工件為100 trials／fold
- Future Target：只在portfolio replay完成後join作診斷，未進入runtime、optimizer或交易決策

| 指標 | Baseline | Sort Only | Param Only | Adapted |
|---|---:|---:|---:|---:|
| 淨總報酬 | 182.62% | 147.57% | 145.84% | 75.70% |
| 最大回撤 | 13.18% | 21.56% | 15.79% | 18.40% |
| Return／MDD | 13.86 | 6.85 | 9.24 | 4.11 |
| 平均曝險 | 77.33% | 55.02% | 81.00% | 54.28% |
| 平均初始停損距離 | 6.39% | 9.23% | 5.58% | 8.84% |
| 平均實際投入金額 | 164,607.74 | 107,910.18 | 147,685.60 | 71,804.77 |
| 平均Realized R | 0.28R | 0.30R | 0.53R | 0.37R |
| 選中候選Target mean | 1.0912R | 1.2201R | 0.8217R | 1.0756R |
| Aggregate Target capture | 0.30 | 0.26 | 0.77 | 0.35 |

### 與原30% coverage結果比較

舊結果為Baseline 182.62%、Sort Only 144.80%、Param Only 65.46%、Adapted 59.71%；本次分別為182.62%、147.57%、145.84%、75.70%。Param Only與Adapted數值明顯提高，但本次Adapted trials／fold為300，而舊執行紀錄使用200 trials／fold，因此兩次結果不是只改coverage的單一控制實驗，不能把全部差異歸因於coverage。可確定的正式結論應以本次同一2×2內的對稱比較為準：

- 舊ROOS固定時，`Sort Only − Baseline = −35.05pp`，MDD增加8.38pp，Return／MDD下降7.01。
- 新Adapted params固定時，`Adapted − Param Only = −70.14pp`，EV由0.53R降至0.37R，Aggregate capture由0.77降至0.35。
- Score Adapted params本身亦未打敗舊ROOS：`Param Only − Baseline = −36.78pp`；在Score ranking下，`Adapted − Sort Only = −71.87pp`。
- 因此coverage提升改善了optimizer可見的新ranking歷史比例，但沒有使Score ranking或Score Adapted params通過正式投組門檻。

### 歸因與採用判定

1. Score排序能力仍有效：兩套參數下，Target percentile、top-k retention、Target opportunity gap與Target mean均改善，表示模型確實挑到事後價格機會較高的候選。
2. 固定舊參數時，Score ranking的平均Realized R由0.28R升至0.30R、平均投入資金報酬由3.00%升至3.85%，但初始停損距離由6.39%擴大至9.23%，平均投入由164,608降至107,910，平均曝險由77.33%降至55.02%。
3. 同一套新Adapted params下，初始停損距離由5.58%擴大至8.84%，平均投入由147,686降至71,805，平均曝險由81.00%降至54.28%；同時Realized R、Payoff、勝率與capture亦下降，顯示不只資金利用率，策略capture契約也與Score排序不相容。
4. Score ranking反而減少未滿倉日與持股缺口，但美元曝險仍大幅下降，證明主因不是候選供給或持倉格填不滿，而是每個slot因寬停損及fixed-risk sizing而配置過小。
5. 正式策略維持Baseline；原始Score ranking、Score Adapted params及完整Adapted系統均`REJECTED`。Selection PIT向前延伸工件保留，供後續同一PIT source的研究重用。

### 下一個單一研究方向

下一步固定舊正式ROOS、fixed risk、position cap、max positions、entry／stop／exit、rotation與帳務，只做capital-aware ranking／停損距離消融，不重訓模型、不重新執行rolling optimizer。第一輪比較：

1. 原正式buy-sort。
2. 原始Score第一排序。
3. `Score × projected capital fraction`，其中projected capital fraction必須沿用正式盤前sizing單一真理結果，不另寫近似公式。
4. 依正式預估投入比例或初始停損距離作固定分桶，再於同桶內依Score排序；分桶邊界須由Selection train-only分布或固定可解釋契約決定，不得由2014～2020回放績效調參。

第一階段只判斷能否在維持Target mean／Realized R優勢下恢復平均投入、曝險、總報酬與Return／MDD。Filter × Score分組消融維持第二順位；不再重跑相同原始Score Adapted optimizer，也不直接提高fixed risk、position cap或max positions。

## 2026-08-03 — Capital-aware Ranking R2／R3 CLI-only消融

### 狀態

`RESULT_AVAILABLE / R2_REJECTED / R3_DIRECTION_PASS_FORMAL_POLICY_NOT_ACCEPTED`

### 程式基準

- 輸入ZIP：`test-branch-1_20260803_211701_c0cb6c3.zip`
- SHA256：`1c8aa4144af00f7ae7696c90dbe649932650e89f16881da7a62b9d3382f6cc9e`
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

### 實驗目的

Coverage提升後的正式2×2已確認原始Score ranking在舊ROOS與新Adapted params下都降低投組報酬；共同機械鏈是Score偏好較寬初始停損候選，fixed-risk sizing使每slot投入與平均曝險下降。下一個單一變更固定舊正式ROOS與全部交易規則，只調整Score候選排序，使模型品質與正式可部署部位共同進入ranking。

### 唯一變更

新增`strategy-compare --ranking-policy`兩個CLI-only研究政策；正式預設仍為`score`，互動選單、optimizer、模型、Target與正式Baseline不變。

1. R2 `capital-adjusted-score`：沿用正式候選已完成exact-accounting後的`proj_cost`，計算`projected_capital_fraction = proj_cost / sizing_capital`及`deployment_rate = min(1, projected_capital_fraction / max_position_cap_pct)`；有效Score候選依`Score × deployment_rate`降冪，再沿用既有buy-sort。
2. R3 `capital-bucket-then-score`：只在當日有效PIT Score候選中，依正式`deployment_rate`的當日橫斷面1/3與2/3分位分成高／中／低三桶；先按桶別，再於桶內按Score，最後沿用既有buy-sort。相同部署率可落在同桶，不使用回放績效或Future Target決定邊界。
3. Active-param ensemble仍先按vote count；同票候選使用指定policy。部署率使用member有限值中位數。PIT缺分候選不排除、不填0，並完整回退原buy-sort；全缺分日期不得因capital-aware欄位改變pre-PIT排序。
4. 非原始policy輸出使用獨立目錄，避免覆蓋既有R1正式診斷。

### 固定條件

- 參數：既有`base_finalist_best` rolling active params。
- Score：目前Selection PIT工件，實際起點`2011-01-01`。
- 比較期間：預定固定`2014-01-01～2020-12-31`。
- fixed risk、position cap、max positions、entry／stop／trail／exit、rotation、候選filters、交易成本、帳務與0050 benchmark全部不變。
- 不重建Dataset、不relabel、不重建Continuous Target、不重訓模型、不執行rolling optimizer。
- Future Target只可在portfolio replay完成後join作read-only診斷。

### 實作與驗證

- `core/buy_sort.py`集中管理三種ranking policy及正式部署率衍生公式，避免R2與R3各自重算sizing。
- `core/portfolio_candidates.py`在候選形成後保存正式`projected_capital_fraction`、`deployment_rate`與policy；不修改股數或價格。
- `core/portfolio_engine.py`將相同policy套用於active-param ensemble，同時保存候選replay診斷欄位。
- `filters/breakout_quality/runtime.py`以ContextVar傳遞本次research ranking policy，預設仍為原始`score`。
- `strategy_compare.py`新增`--ranking-policy`、policy-specific排序metadata與隔離輸出目錄。
- T267既有direct synthetic case已擴充：驗證部署率公式、R2乘積排序、R3三分桶、全缺分fallback及ensemble vote優先；本輪不執行正式`apps/test_suite.py`。

### Dataset／Label／模型工件需求

本次程式變更不需要重建Dataset、Label、Continuous Target、PIT Scores或checkpoint。正式本機執行只需要既有完整市場資料、Selection PIT manifest／audit／scores及`roos_base_best.json`。

### 真實 Selection replay 結果

固定Baseline為既有`base_finalist_best` rolling active params，期間`2014-01-01～2020-12-31`；R0 Baseline總報酬182.62%、MDD 13.18%、RoMD 13.86、平均曝險77.33%、平均實際投入164,607.74、平均初始停損距離6.39%、平均Realized R 0.28R、平均Target R 1.00R、Aggregate capture 0.30。

| 指標 | R2 `capital-adjusted-score` | R3 `capital-bucket-then-score` | R2判讀 | R3判讀 |
|---|---:|---:|---|---|
| 淨總報酬 | 176.18% | 184.12% | 低於Baseline 6.44pp | 高於Baseline 1.51pp |
| 最大回撤 | 14.15% | 19.20% | 輕微惡化0.98pp | 明顯惡化6.02pp |
| Return／MDD | 12.45 | 9.59 | 低於Baseline 1.41 | 低於Baseline 4.27 |
| Log R² | 0.9266 | 0.9101 | 退步 | 明顯退步 |
| 月勝率 | 63.10% | 59.52% | 改善1.19pp | 退步2.38pp |
| 平均曝險 | 77.44% | 70.68% | 已恢復至Baseline附近 | 仍低6.65pp |
| 保留買單成交率 | 95.85% | 99.61% | 明顯下降3.77pp | 幾乎不變 |
| 平均實際投入 | 158,831.07 | 142,637.68 | 低3.51% | 低13.35% |
| 平均初始停損距離 | 6.58% | 7.62% | 接近Baseline | 仍偏寬 |
| 平均Realized R | 0.29R | 0.34R | 微幅改善 | 明顯改善 |
| 平均Target R | 0.98R | 1.05R | 低於Baseline | 高於Baseline |
| Aggregate capture | 0.30 | 0.33 | 持平 | 改善 |
| 平均投入資金報酬 | 2.79% | 3.74% | 退步 | 明顯改善 |

### 歸因與採用判定

1. **R2拒絕目前公式。** `Score × deployment_rate`成功把曝險由原始R1約55%恢復至77.44%，證明資金部署確實是原始Score ranking的重要瓶頸；但R2同時使保留買單成交率降至95.85%、Target mean降至1.0450R、平均投入資金報酬降至2.79%，最終總報酬仍低於Baseline且MDD略高。連續乘積把Score百分位當成可線性縮放的經濟價值，並把既有執行排序壓到後面，未形成可採用證據。
2. **R3方向保留，但目前policy不採用。** 三分桶後桶內Score使Target percentile、top-k retention、Target mean、Realized R、Aggregate capture與平均投入資金報酬全部改善，總報酬184.12%亦略高於Baseline；這證明「先限制極小部位候選壟斷，再保留Score排序」是有效研究方向。
3. R3仍未通過正式風險邊界：MDD由13.18%升至19.20%、RoMD由13.86降至9.59、Log R²由0.9349降至0.9101、月勝率由61.90%降至59.52%、最差年度由-0.35%惡化至-3.93%。R3較低勝率、較高Payoff／EV、較長持有期與較寬停損，形成較不平滑且更集中於少數贏家的報酬路徑；不能只因總報酬略高即升格正式policy。
4. R2與R3都比原始R1總報酬147.57%大幅改善，確認capital-aware ranking修正方向成立；但正式策略仍維持R0 Baseline，R2標記`REJECTED`，R3標記`RESULT_AVAILABLE / RESEARCH_DIRECTION_PASS / FORMAL_POLICY_NOT_ACCEPTED`。
5. 本次不重建Dataset、Label、Continuous Target、PIT Scores或checkpoint，也不執行optimizer；Future Target只於replay完成後join。

### 下一步

先執行R3最大回撤read-only attribution，不改ranking、不重訓模型、不跑optimizer：定位最大回撤起迄日期，逐日／逐交易比較Baseline與R3的獨有持倉、初始停損距離、deployment bucket、Score、Target、Realized R、持有期、損失聚集與產業／月份集中。先確認MDD惡化是由特定regime、跨桶硬排序、寬停損或損失時間聚集造成，再決定下一個單一變更。未完成此歸因前，不直接調整桶數、桶邊界、fixed risk、position cap或max positions；Filter × Score消融維持後續順位。

## 2026-08-03 — R3 Capital-bucket Ranking Rolling Parameter Adaptation

### 狀態

`RESULT_AVAILABLE / R3_RANKING_CONFIRMED / R3_ADAPTED_PARAMS_REJECTED / FORMAL_BASELINE_UNCHANGED`

使用者判定R3已在總報酬、EV、Target選擇、Target capture與平均投入資金報酬形成足夠的多項改善，不先投入專門最大回撤歸因；下一個單一實驗改為固定R3 ranking契約，執行與既有Score Adapted相同的rolling optimizer及四組2×2。此決定取代前一節「先做R3最大回撤read-only attribution」的下一步，但不改變R3目前尚未升格正式policy的判定。

### 程式基準

- 輸入ZIP：`test-branch-1_20260803_221024_d9ba32f.zip`
- SHA256：`273d6257fcc42b403ec1c588b67a3f227bd5c02102576da256e0663c26dd99bc`
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

### 唯一主要變更

`tools/filters/breakout_quality/strategy_adapt.py`新增CLI-only `--ranking-policy capital-bucket-then-score`。未指定時仍維持既有原始`score`流程與互動選單行為；R3不加入主選單、不改正式策略預設，也不把ranking policy、桶數或桶邊界放入optimizer搜尋。

R3流程固定：

1. Baseline：原buy-sort＋舊正式ROOS。
2. R3 Sort Only：`capital-bucket-then-score`＋舊正式ROOS。
3. Param Only：原buy-sort＋同一套R3 Adapted active params。
4. R3 Adapted：`capital-bucket-then-score`＋同一套R3 Adapted active params。

只訓練一套R3 Adapted rolling optimizer；Param Only不是第二套optimizer，而是把同一套新參數切回原ranking的反事實回放。

### Runtime與工件隔離

- R3 ranking policy進入optimizer runtime context、runtime cache identity、persistent study identity、Sort Only／Adapted replay context、preflight、summary與manifest。
- 原始Score Adapted沿用`models/research/breakout_quality/score_ranking_adaptation/rolling_validation/`。
- R3 Adapted使用獨立路徑`models/research/breakout_quality/score_ranking_adaptation/capital_bucket_then_score/rolling_validation/`。
- 兩種policy不得互相resume、覆蓋active params或重用Baseline／Sort Only pair manifest。
- Selection PIT正式models root與Adapted active-param輸出仍保持分離。
- active params內固定`use_breakout_quality_ranking=True`、hard filter=False、fixed risk與position cap；實際R3 policy由同一Selection PIT runtime context提供，不成為策略參數trial。

### 固定條件

Dataset、Continuous Target、PIT Scores、PIT audit、9A模型、No-time Target、Seed 42、rolling fold schedule、120個月training window、12個月OOS horizon、300 trials／fold、search space、objective、sampler、TP、fixed risk、position cap、max positions、rotation、entry／stop／exit、交易成本、portfolio accounting與Future Target post-replay-only契約全部不變。

Coverage契約維持：

- actual PIT start=`2011-01-01`
- reference start=`2014-01-01`
- weighted coverage=`60.0%` vs `30.0%`
- 7 folds全部保留
- actual逐fold不得低於reference，至少一fold及加權總coverage必須提高
- actual PIT起點以前允許原buy-sort fallback
- PIT期間內缺分禁止
- 全部OOS replay必須位於PIT期間

### Dataset／Label／模型工件需求

不重建Dataset、不relabel、不重建Continuous Target、不重訓breakout-quality模型、不重建PIT Scores。需要本機既有完整市場資料、Selection PIT score／manifest／audit、正式`roos_base_best.json`與optimizer runtime依賴。首次R3執行會建立全新的R3 studies及R3 Adapted active params；不得沿用原始Score Adapted的完成study。

### 本機執行

R3為CLI-only：

```bash
python apps/breakout_quality.py strategy-adapt --dataset full --param-policy base-finalist-best --ranking-policy capital-bucket-then-score
```

預期輸出：

- R3 Baseline／Sort Only pair
- R3 rolling preflight與training coverage
- R3 Adapted active params
- rolling optimizer summary與manifest
- Baseline／R3 Sort Only／Param Only／R3 Adapted四組比較
- adapted-params ranking capture audit
- 年度報酬與參數差異

### 原定採用判定

本輪程式接線時預先設定以下判定維度；實際結果與正式結論記錄於後續「本機正式結果」與「對稱比較與判定」：

1. R3 Adapted相較R3 Sort Only是否提高總報酬或Return／MDD。
2. R3 Adapted相較Param Only的ranking效果是否仍為正。
3. 最大回撤是否低於R3 Sort Only的19.20%，Return／MDD是否高於9.59。
4. EV、Realized R、Target mean、三種capture與平均投入資金報酬是否保留R3優勢。
5. Param Only是否顯示R3 Adapted params本身破壞原策略。
6. 改善是否跨年度，而非只來自單一年份。

### 原定本機執行

R3 CLI已由使用者本機完成；本機正式結果見下節。正式`apps/test_suite.py`仍由使用者本地執行，GPT未執行。

### 本機正式結果

- 結果程式基準：`test-branch-1_20260803_235424_5af0f20.zip`
- SHA256：`2b5e23c783c506adc27660a7e5d7c48b077737fcadb3166f43aa7fe5a35542cd`
- 比較期間：`2014-01-01～2020-12-31`
- Ranking policy：`capital-bucket-then-score`
- Rolling folds：7；train window 120個月；OOS horizon 12個月；300 trials／fold
- PIT actual／reference start：`2011-01-01`／`2014-01-01`
- Actual／reference weighted coverage：60.0%／30.0%；7 folds均提升30.0pp
- Adapted search reused：False；新參數optimizer數：1
- Future Target：只在portfolio replay完成後join，未進入optimizer或交易runtime

| 指標 | Baseline | R3 Sort Only | Param Only | R3 Adapted |
|---|---:|---:|---:|---:|
| 淨總報酬 | 182.62% | 184.12% | 123.77% | 147.09% |
| 最大回撤 | 13.18% | 19.20% | 14.22% | 12.92% |
| Return／MDD | 13.86 | 9.59 | 8.71 | 11.38 |
| 年化報酬 | 16.00% | 16.09% | 12.20% | 13.80% |
| Log R² | 0.9349 | 0.9101 | 0.9418 | 0.9439 |
| 月勝率 | 61.90% | 59.52% | 69.05% | 60.71% |
| 平均曝險 | 77.33% | 70.68% | 82.61% | 74.96% |
| 平均初始停損距離 | 6.39% | 7.62% | 5.36% | 6.85% |
| 平均Realized R | 0.28R | 0.34R | 0.49R | 0.52R |
| 平均投入資金報酬 | 3.00% | 3.74% | 1.96% | 2.20% |
| 選中候選Target mean | 1.0912R | 1.1080R | 0.7013R | 0.9356R |
| Aggregate Target capture | 0.30 | 0.33 | 0.69 | 0.60 |

### 對稱比較與判定

1. **R3 ranking效果再次成立。** 舊ROOS固定時，`R3 Sort Only − Baseline = +1.51pp`總報酬，但MDD增加6.02pp、Return／MDD下降4.27；同一套新Adapted params固定時，`R3 Adapted − Param Only = +23.33pp`總報酬、MDD降低1.30pp、Return／MDD提高2.68。R3在兩套參數下都提高總報酬，且在新參數下連風險調整績效也同步改善，因此ranking方向不是舊參數偶然。
2. **R3 Adapted params拒絕。** 不使用R3時，`Param Only − Baseline = -58.85pp`；使用R3時，`R3 Adapted − R3 Sort Only = -37.03pp`。新參數雖把R3 MDD由19.20%降至12.92%、Return／MDD由9.59提高至11.38，但犧牲37.03pp總報酬，且仍未打敗Baseline的Return／MDD 13.86，不能升格正式參數。
3. 新參數建立的候選母體品質較差：Param Only／R3 Adapted的Target mean為0.7013R／0.9356R，均低於舊ROOS的1.0912R／1.1080R。R3可在新候選池內把Target mean提高0.2343R，但無法恢復舊ROOS候選池品質；這比單純增加trials更值得關注。
4. R3 Adapted的EV與Realized R為0.52R、MDD為12.92%，但平均投入資金報酬只有2.20%，低於Baseline 3.00%與R3 Sort Only 3.74%。新參數改善單筆R與風險路徑，卻沒有形成足夠的資本效率與總報酬。
5. 正式策略與正式ROOS維持Baseline。R3保留`RESEARCH_DIRECTION_PASS / FORMAL_POLICY_NOT_ACCEPTED`；R3 Adapted params標記`REJECTED`。不再增加相同optimizer trials，也不再以同一search space重跑R3 adaptation。

### 下一個單一研究方向

直接測試先前保留的`R3 ranking × optional entry filters`交互作用，不先做最大回撤逐筆歸因。第一階段採一次粗粒度2×2 gate，固定舊正式ROOS、PIT Scores、R3三分桶、fixed risk、position cap、max positions、ATR entry／stop／trail、exit、rotation、交易成本與帳務：

1. Current filters＋原buy-sort。
2. Current filters＋R3。
3. Optional entry filters全部關閉＋原buy-sort。
4. Optional entry filters全部關閉＋R3。

第一階段只強制關閉primary entry qualification中的`use_breakout_ema_filter`、`use_bb`、`use_vol`、`use_breakout_return_filter`與`use_breakout_false_filter`；不關閉`high_len`事件定義、ATR buy／stop／trail、`use_kc` exit、reclaim re-entry、fixed risk或position cap。正式交互作用比較為`(4−3) − (2−1)`。若全部關閉後R3 ranking效果明顯改善，再逐一拆解五個filter；若沒有改善，停止filter-conflict假設，下一步才考慮重做strategy／capital-aligned Target。此gate不重訓模型、不重建PIT Scores、不執行optimizer。


## 2026-08-04 — Optional Entry Filters × Ranking A～E Gate

### 狀態

`IMPLEMENTED / RESULT_NOT_AVAILABLE / LOCAL_FULL_DATA_EXECUTION_REQUIRED`

### 程式基準

- 輸入ZIP：`test-branch-1_20260804_000824_eb63f9d.zip`
- SHA256：`eaeb64f05a319e489b2fa263ca52b5753e195671dbe783999b62e97a2b6adcdf`
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

### 實驗目的

R3 ranking在舊ROOS與R3 Adapted params下都形成正向ranking效果，但R3 Adapted params整體弱於舊ROOS；下一步不再增加optimizer trials，而是固定舊正式ROOS，直接測試既有optional entry filters是否刪除或扭曲Score／R3較擅長的候選母體。使用者要求在原A～D粗粒度gate之外增加E「Optional entry filters全關＋原始Score sort」，用以判斷filters全關後原始Score是否恢復，以及R3資金分桶是否仍有必要。

### 唯一主要變更

新增CLI-only `strategy-filter-gate`，固定五組：

1. A：目前active params filters＋原buy-sort。
2. B：目前active params filters＋R3 `capital-bucket-then-score`。
3. C：五個optional entry filters全部關閉＋原buy-sort。
4. D：五個optional entry filters全部關閉＋R3。
5. E：五個optional entry filters全部關閉＋原始`score` ranking。

全關欄位只包含：

- `use_breakout_ema_filter`
- `use_bb`
- `use_vol`
- `use_breakout_return_filter`
- `use_breakout_false_filter`

不關閉`high_len`事件定義、ATR buy／initial stop／trail、`use_kc` exit、reclaim re-entry、fixed risk、position cap或max positions。

### Runtime與報表契約

- Gate重用`strategy_compare.run_comparison`三次，不複製portfolio engine、sizing、成交、費用或統計：AB=current filters × R3、CD=filters all-off × R3、CE=filters all-off × raw Score。
- C會在CD與CE各執行一次；兩次`no_filter_equity.csv`、`no_filter_trades.csv`與`no_filter_daily_capacity.csv`的SHA256必須完全相同，否則fail-fast。
- 合併報表輸出A～E相同口徑的投組、資金、Target、capture與模型選股表，並固定列出：`B−A`、`D−C`、`E−C`、`C−A`、`D−B`、`D−E`及R3 filter interaction `(D−C)−(B−A)`。
- E只回答filters全關後原始Score ranking效果；`D−E`回答在相同候選池下R3資金分桶相較原始Score是否仍有增益。
- `strategy-compare --optional-entry-filters all-off`會對pair兩側同時固定五個filters為False，pair內仍只允許`use_breakout_quality_ranking`不同；all-off輸出identity與既有current-filter比較隔離。
- Gate不加入互動選單，不修改正式Baseline與正式ranking policy。

### 固定條件

- 參數：舊正式`base_finalist_best` rolling active params。
- Score：Selection PIT `selection_point_in_time`。
- 預定期間：`2014-01-01～2020-12-31`。
- R3三分桶、Score、fixed risk、position cap、max positions、ATR entry／stop／trail、exit、rotation、交易成本、portfolio accounting與0050 benchmark不變。
- 不重建Dataset、不relabel、不重建Continuous Target、不重訓模型、不重建PIT Scores、不執行optimizer。
- Future Target只在portfolio replay完成後離線join，未進入候選、排序、資金配置或成交決策。

### Dataset／Label／模型工件需求

本次程式變更不需要重建Dataset、Label、Continuous Target、PIT Scores或checkpoint。本機執行需要既有完整市場資料、Selection PIT score／manifest／audit與`models/research/breakout_quality/selection_strategy_realization/roos_base_best.json`。

### 本機執行

本Gate為臨時研究CLI，尚未納入正式選單：

```bash
python apps/breakout_quality.py strategy-filter-gate --dataset full --param-policy base-finalist-best --start-date 2014-01-01 --end-date 2020-12-31 --max-positions 10 --rotation off
```

主要輸出位於：

`outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/strategy_filter_gate_base_finalist_best_selection_point_in_time/`

包含三個pair子目錄與合併`strategy_filter_gate.md`／`strategy_filter_gate.json`。

### 採用判定

尚未取得真實Selection replay結果，不預判有效。結果取得後依以下順序判讀：

1. `D−C`是否明顯優於`B−A`；若是，表示filters全關後R3相對效果增加。
2. `E−C`是否由既有raw Score負面結果轉正或大幅改善；若是，表示entry filters可能是原始Score失敗的重要交互因素。
3. `D−E`是否仍為正；若是，R3資金分桶在filters全關候選池仍有額外價值；若接近零或為負，可能只需原始Score而不需R3。
4. 同時檢查Baseline絕對效果：C、D、E不能只因相對差異改善就忽略總報酬、MDD、Return／MDD、資本效率與Target capture。
5. 只有粗粒度全關gate為正，才逐一拆解五個filter；若D與E都沒有改善，停止filter-conflict假設，下一步轉向strategy／capital-aligned Target。

## 2026-08-04 — A～E Gate Selection PIT profile預設修正

### 狀態

`IMPLEMENTED / RESULT_NOT_AVAILABLE / LOCAL_FULL_DATA_EXECUTION_REQUIRED`

### 程式基準

- 輸入ZIP：`test-branch-1_20260804_003430_2259c60.zip`
- SHA256：`c39b72722685988a420a8429ef01bb31d1eb506d5c46b0bd6717e6585f5ec0fd`

### 問題與唯一變更

使用者本機執行`strategy-filter-gate`時，Gate雖固定使用Selection PIT score source，CLI預設卻取用binary classification常數`BREAKOUT_QUALITY_EXPERIMENT_PROFILE=unique_group_sampling`，因此錯誤尋找：

`models/filters/breakout_quality/breakout_quality_v1/inception_time_v1/unique_group_sampling/point_in_time/selection_point_in_time_scores.csv`

實際Selection PIT continuous-ranker工件位於workflow profile`strategy_aligned_no_time_pass_magnitude_mse`。本輪只把Gate的filter id、model architecture、experiment profile、dataset、max positions與rotation預設改為直接讀取`get_breakout_quality_workflow_settings()`；A～E情境、舊ROOS、ranking、filters、PIT score內容、策略執行與報表口徑均未改變。

### Dataset／Label／模型工件

不重建Dataset、不relabel、不重訓模型、不重建Continuous Target或PIT Scores。修正後直接讀取既有workflow Selection PIT工件。

### 驗證契約

`validate_breakout_quality_strategy_comparison_contract_case`新增動態設定案例，確認Gate預設identity始終等於目前workflow設定；不得把任何當前profile值硬編碼為唯一合法答案。

### 結果與判定

尚未取得本機A～E實際績效。此修正只排除錯誤profile路徑，狀態維持`IMPLEMENTED`；使用者須重新執行同一條`strategy-filter-gate`命令。


## 2026-08-04 — Optional Entry Filters × Ranking A～E Gate 本機正式結果

### 狀態

`RESULT_AVAILABLE / FILTER_CONFLICT_HYPOTHESIS_REJECTED / OPTIONAL_FILTERS_REQUIRED / FORMAL_BASELINE_UNCHANGED`

### 結果程式基準

- ZIP：`test-branch-1_20260804_005030_da9891c.zip`
- SHA256：`3b0d7bbdec8045c271cb635cee04e615d783e073442f3a5131736c9f59793cd7`
- 比較期間：`2014-01-01～2020-12-31`
- 參數：舊正式`base_finalist_best` rolling active params
- Score source：Selection PIT `selection_point_in_time`
- Future Target：只在portfolio replay完成後離線join，未進入候選、排序、資金配置或成交決策

### A～E正式結果

| 指標 | A 目前filters＋原排序 | B 目前filters＋R3 | C filters全關＋原排序 | D filters全關＋R3 | E filters全關＋raw Score |
|---|---:|---:|---:|---:|---:|
| 淨總報酬 | 182.62% | 184.12% | 92.73% | 104.51% | 71.20% |
| 最大回撤 | 13.18% | 19.20% | 28.17% | 28.81% | 25.96% |
| Return／MDD | 13.86 | 9.59 | 3.29 | 3.63 | 2.74 |
| EV／平均Realized R | 0.28R | 0.34R | 0.43R | 0.41R | 0.19R |
| 平均曝險 | 77.33% | 70.68% | 89.64% | 81.22% | 61.91% |
| 平均實際投入 | 164,608 | 142,638 | 153,156 | 117,273 | 73,216 |
| 平均初始停損距離 | 6.39% | 7.62% | 4.91% | 6.89% | 9.67% |
| 選中候選Target mean | 1.09R | 1.11R | 0.60R | 0.98R | 1.15R |
| Aggregate Target capture | 0.30 | 0.33 | 0.81 | 0.45 | 0.19 |
| 平均投入資金報酬 | 3.00% | 3.74% | 1.89% | 2.50% | 1.88% |

主要對稱差異：

- `B−A`：總報酬`+1.51pp`、MDD`+6.02pp`、Return／MDD`−4.27`、Realized R`+0.07R`、Target mean`+0.02R`、capture`+0.03`。
- `D−C`：總報酬`+11.78pp`、MDD`+0.64pp`、Return／MDD`+0.34`、Realized R`−0.02R`、Target mean`+0.38R`、capture`−0.35`。
- `E−C`：總報酬`−21.53pp`、MDD`−2.21pp`、Return／MDD`−0.55`、Realized R`−0.24R`、Target mean`+0.55R`、capture`−0.61`。
- `C−A`：總報酬`−89.89pp`、MDD`+14.99pp`、Return／MDD`−10.57`。
- `D−B`：總報酬`−79.62pp`、MDD`+9.61pp`、Return／MDD`−5.96`。
- `D−E`：總報酬`+33.31pp`、MDD`+2.84pp`、Return／MDD`+0.89`、Realized R`+0.22R`、Target mean`−0.17R`、capture`+0.26`。
- R3 interaction `(D−C)−(B−A)`：總報酬`+10.27pp`，但不得脫離C／D的絕對績效解讀。

### 正式判定

1. **Optional entry filters是必要的候選品質層，不是壓制Score的主要衝突源。** 全關後原排序總報酬由182.62%降至92.73%、MDD由13.18%升至28.17%；R3由184.12%降至104.51%、MDD由19.20%升至28.81%。三種ranking在全關候選池的絕對績效都遠低於目前filters情境。
2. **正向R3 interaction只表示R3對劣化候選池具有較強的相對補救能力，不代表關閉filters有利。** D相較C增加11.78pp，但D仍比B少79.62pp總報酬，且Return／MDD只有3.63。
3. **原始Score在filters全關後沒有恢復，反而更差。** E雖把Target mean提高至1.15R，但平均投入只剩73,216、停損距離擴至9.67%、Realized R降至0.19R、capture降至0.19，總報酬只剩71.20%。這再次確認raw Score偏向高Future Target但低部署／低可實現性候選。
4. **R3資金分桶機制仍有獨立價值。** 在相同filters全關候選池，D相較E增加33.31pp總報酬、0.22R Realized R與0.26 capture；R3以略低Target mean換取更高部署與可實現性。但此價值不足以補救filters全關造成的候選品質崩落。
5. C的capture 0.81不可解讀為策略更佳；其平均Target只有0.58R／Target mean 0.60R，分母較低且平均投入資金報酬只有1.89%。採用判定仍以絕對報酬、MDD、Return／MDD與資本效率為主。
6. **Filter-conflict粗粒度假設拒絕。** 不進入逐一關閉EMA、BB、Volume、Return與False-breakout filters的拆解；正式filters、正式ROOS與正式Baseline均維持不變。R3仍保留`RESEARCH_DIRECTION_PASS / FORMAL_POLICY_NOT_ACCEPTED`。

### 下一個單一研究方向

轉向`Capital-aligned Target feasibility audit`，不先重訓模型。固定目前optional filters、舊正式ROOS、Selection PIT Scores、R3與portfolio規則，使用既有orderable candidate／Future Target與正式盤前sizing欄位，建立只供離線稽核的：

`capital_opportunity_proxy = no_time_target_R × deployment_rate`

其中`deployment_rate`必須重用正式`proj_cost / sizing_capital / max_position_cap_pct`並截斷於1，不另寫近似sizing。先比較raw No-time Target與capital proxy對下列結果的日內排序能力：實際投入資金報酬、Realized R、`Realized R × deployment_rate`、top-k retention及年度穩定性。此audit不得進runtime、不得使用OOS結果調參、不得建立checkpoint或PIT Scores。

只有capital proxy在多數年度與主要投組貢獻指標上穩定優於現行No-time Target，才建立新的versioned Continuous Target與PIT ranker；若未通過，停止目前continuous-target家族，不以不同乘方、權重或分桶數繼續Selection調參。

### Dataset／Label／模型工件需求

本次結果回寫不修改程式，不重建Dataset、Label、Continuous Target、PIT Scores或checkpoint。下一個feasibility audit可直接使用既有Selection策略比較候選工件、Future Target與正式sizing payload；只有audit通過後才評估新Target arrays與模型重訓。

## 2026-08-04 — Binary DL Filter Replacement A／B／C／F Gate

### 狀態

`IMPLEMENTED / RESULT_NOT_AVAILABLE / LOCAL_FULL_DATA_EXECUTION_REQUIRED`

### 程式基準

- 輸入ZIP：`test-branch-1_20260804_010014_afebdd6.zip`
- SHA256：`ce9e5b11fb2df28dde5c2410f12fcb71b303c9fc3c69ea067819596f59d15e70`
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

### 實驗動機

A～E Gate已證明直接關閉optional entry filters而沒有替代品質Gate會使候選池與絕對投組績效崩落；但該Gate只測原buy-sort、continuous raw Score與R3，沒有測「規則品質filters全關後，由既有9A Binary DL Filter作唯一買入品質確認」。本輪依使用者決定，不再要求DL Score取代position-aware ranking，而是保留原buy-sort、sizing、entry／stop／exit與portfolio accounting，只讓Binary DL Filter取代EMA、BB、Volume、breakout return及false-breakout五個rule-based品質filters。

### 唯一主要變更

新增CLI-only `strategy-dl-filter-gate`，固定四組：

1. A：目前optional entry filters＋Binary DL filter關閉＋原buy-sort。
2. B：目前optional entry filters＋Binary DL filter開啟＋原buy-sort。
3. C：五個optional entry filters全關＋Binary DL filter關閉＋原buy-sort。
4. F：五個optional entry filters全關＋Binary DL filter開啟＋原buy-sort。

全關欄位只包含：

- `use_breakout_ema_filter`
- `use_bb`
- `use_vol`
- `use_breakout_return_filter`
- `use_breakout_false_filter`

不關閉`high_len`突破事件定義、ATR buy／initial stop／trail、`use_kc` exit、reclaim re-entry、fixed risk、position cap、max positions或原position-aware buy-sort。四組均固定`use_breakout_quality_ranking=False`，不使用continuous Score或R3。

### Runtime與報表契約

- Gate重用`strategy_compare.run_comparison`兩次，不複製portfolio engine、成交、費用、統計或trade attribution：AB=current optional filters × binary hard filter；CF=all-off optional filters × binary hard filter。
- Binary identity固定由`BREAKOUT_QUALITY_DEFAULT_FILTER_ID / BREAKOUT_QUALITY_MODEL_ARCHITECTURE / BREAKOUT_QUALITY_EXPERIMENT_PROFILE`解析，目前為`breakout_quality_v1 / inception_time_v1 / unique_group_sampling`；threshold固定使用OOS前鎖定的`BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD`，目前0.5，不提供Gate內threshold調整。
- Score source固定`canonical_runtime`；未指定日期時使用runtime manifest宣告的`execution_start～available_through`，不得誤用continuous Selection PIT的2014～2020期間。
- AB與CF必須共用相同runtime manifest／score、threshold、active-param來源與hash、rolling selector、member count、min_agree、期間、benchmark、max positions及rotation；唯一跨pair差異是五個optional filters是否全關。
- 合併報表固定輸出A／B／C／F投組指標、年度報酬、`B−A`、`F−C`、`C−A`、`F−A`及replacement interaction `(F−C)−(B−A)`，並列AB與CF的trade attribution：被排除贏家R、避開輸家|R|、替代贏家／輸家R、直接DL拒絕與portfolio path displacement。
- `F−C`回答Binary DL在沒有規則品質filters時能否單獨提供有效品質Gate；`B−A`回答疊加既有filters是否有效；`F−A`才是DL-only replacement相較目前正式策略的採用比較。正interaction只代表替代優於疊加，不等於F的絕對績效通過。
- 輸出隔離於binary model output下`strategy_dl_filter_gate_<param_policy>_canonical_runtime/`，包含`pair_ab/`、`pair_cf/`、`strategy_dl_filter_gate.md`與`strategy_dl_filter_gate.json`。

### 固定條件

- 不重建Dataset、不relabel、不重訓9A、不重新匯出score、不調threshold、不執行optimizer。
- 使用正式rolling active params、原position-aware buy-sort、相同fixed risk、position cap、max positions、rotation、交易成本、成交與portfolio accounting。
- Future Target與continuous PIT Score完全不進入runtime或報表計算。
- Hard-filter不可評分事件沿用既有保守REJECT契約；不得填0、改用PIT score或回退成ranking。

### 本機執行

此Gate為研究CLI，不加入互動選單：

```bash
python apps/breakout_quality.py strategy-dl-filter-gate --dataset full --param-policy base-finalist-best --max-positions 10 --rotation off
```

若需縮短期間，`--start-date`與`--end-date`必須同時指定，且完整落在canonical runtime score coverage內。

### 採用判定

尚未取得真實策略結果，不預判有效。正式判讀順序：

1. `F−C`是否顯示DL能從全突破候選中移除較多虧損、保留足夠大贏家，並改善總報酬、MDD、Return／MDD、EV與資金使用。
2. `F−A`是否至少打平目前rule-based filters策略的總報酬與Return／MDD，且MDD、曝險、候選供給與年度穩定性沒有明顯惡化。
3. AB與CF trade attribution中，被避免輸家與替代贏家的R總和是否大於被排除贏家與替代輸家的R總和；改善不得只來自共同交易微小執行差異。
4. interaction若為正，只能證明DL作替代者比疊加者更合適；若F仍顯著低於A，replacement假設仍拒絕。
5. 只有現有9A的F相較A形成合理改善，才進一步研究新的binary Label、architecture或固定Validation threshold；否則不得用同一OOS調threshold救援。

### Dataset／Label／模型工件需求

本次程式變更不需重建任何Dataset、Label、checkpoint或score。使用者本機需已有9A canonical `forward_oos` runtime manifest／scores與正式rolling active-param工件；缺少或identity不一致時fail-fast。

## 2026-08-04 — Binary DL Filter 正式選單閉環

### 狀態

`IMPLEMENTED / MODEL_TRAINED / REPORT_AND_STRATEGY_RESULT_NOT_AVAILABLE`

### 程式基準

- 輸入ZIP：`test-branch-1_20260804_015125_1e01efe.zip`
- SHA256：`36a4cbe8e08dd81c62d45abb062493e4b60cb17ec8aab0b4505dfddc54948064`
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

### 使用者已完成訓練

- Profile：`inception_time_v1 / unique_group_sampling`
- Device：CUDA；mixed precision BF16；deterministic=True；TF32=False。
- Epoch selection：Validation Loss；Epoch 2為最佳，最低Validation Loss 0.708591；Epoch 3 early stop。
- Final Selection refit：2 epochs；Final Loss 0.671645。
- Inner Train：538,887 rows／16,832 groups，2011-01-03～2018-11-05。
- Validation：187,316 rows／6,065 groups，2019-01-02～2020-11-05。
- Final Refit：729,654 rows／23,072 groups，2011-01-03～2020-11-05。
- OOS未參與訓練：591,679 rows／17,346 groups，2021-01-04～2025-12-22。
- 已產生`model.pt`、`split_assignments.csv`與`manifest.json`；尚未取得本次research report、forward-OOS score與策略績效結果。

### 唯一主要變更

1. 正式workflow profile切回`unique_group_sampling`，主選單自動解析為Binary classification與hard-filter strategy comparison。
2. Binary「模型研究與驗證」新增兩個正式選項：
   - 重新訓練後，依序顯示OOS簡易模型報表、匯出forward-OOS runtime scores、與Baseline比較實際投組績效。
   - 使用既有模型，不重新訓練；先自動更新research scores，再顯示報表並比較Baseline。
3. 模型報表必須先於策略比較顯示；策略比較重用唯一`strategy-compare` hard-filter鏈，明確傳入filter／architecture／profile／canonical runtime score／fixed risk／position cap。
4. A／B／C／F、Optional filters A～E及其他臨時研究仍維持CLI-only，不加入正式選單。
5. 因主workflow切回Binary，CLI-only A～E continuous Ranking Gate的預設identity改為固定continuous-ranker profile，不再錯誤跟隨主workflow。

### 固定條件

- Dataset、Label、9A architecture、training profile、threshold 0.5、正式rolling params、原position-aware buy-sort、fixed risk、position cap、交易成本與portfolio accounting不變。
- 不新增threshold調整、optimizer、Score ranking、R3或新的模型實驗。
- OOS仍只在模型凍結後用於簡易報表與策略經濟效果比較。

### 採用判定

本輪只完成正式操作閉環，不預判新模型有效。使用者應從主選單`[1] 模型研究與驗證`選擇「使用既有模型」，先取得OOS簡易模型報表，再取得hard-filter相對Baseline的實際績效；結果取得後回寫Selection／OOS分類指標、策略總報酬、MDD、Return／MDD、EV、曝險與採用判定。

## 2026-08-04 — Binary正式策略比較 console capture_result 修正

### 狀態

`IMPLEMENTED / MODEL_RESULT_AVAILABLE / STRATEGY_REPLAY_RERUN_REQUIRED_FOR_FINAL_REPORT`

### 程式基準

- 輸入ZIP：`test-branch-1_20260804_140456_3d88824.zip`
- SHA256：`79a7e287aae8edc8a807e8556bbedf49eefab8307e1d74a1ee10a4f001950aee`
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

### 本機模型結果

- Profile：`inception_time_v1 / unique_group_sampling`；threshold固定`0.5`。
- Epoch 2為最低Validation Loss `0.708591`；Epoch 3 early stop；完整Selection依2 epochs重訓。
- Selection：PASS Precision `60.95%`、PASS Recall `71.19%`、模型PASS `64.02%`。
- OOS：PASS Precision `62.99%`，相較原始PASS `55.63%`提升`+7.35pp`；PASS Recall `51.44%`、模型PASS `45.43%`、Accuracy `56.17%`、PR-AUC `0.6257`、ECE `0.0722`。
- 模型報表判定維持：OOS precision提升，但Recall與模型PASS下降，正式部署須由策略經濟效果決定；不得用此OOS回調threshold或訓練設定。

### 問題與唯一變更

Binary hard-filter replay已完成Baseline與quality-filter兩組，但`strategy_compare.run_comparison`只在score-ranking分支建立`capture_result`；共用console renderer在hard-filter分支仍傳入該區域變數，因而於報表輸出階段發生`UnboundLocalError`。本輪只在分支前將`capture_result`初始化為`None`；score-ranking capture audit、hard-filter trade attribution、portfolio replay、策略參數、Scores、threshold與所有績效口徑均未改變。

### 工件與重跑需求

- 不需重新訓練模型、不需重新匯出research或forward-OOS Scores。
- 錯誤發生前`strategy_comparison.json`、Markdown、equity／trades與trade attribution已進入寫出流程，但為取得完整console與確認工件閉環，套用patch後由正式選單重新執行「使用既有模型」策略驗證。
- 重新執行會重做inference／replay，但不會重訓；最終策略採用判定待完整策略報表取得後回寫。

### 驗證契約

`validate_breakout_quality_strategy_comparison_contract_case`新增直接hard-filter `run_comparison`案例，mock正式runtime、參數與replay payload，確認共用console收到`capture_result=None`且hard-filter JSON正常回傳；不得以只測score-ranking或靜態字串搜尋取代。

### 同輪額外檢查修正

獨立執行`validate_breakout_quality_strategy_comparison_contract_case`時，既有mocked A／B／C／F Gate只替換`run_comparison`，未隔離canonical model preflight，會因GPT檢查環境沒有本機9A checkpoint而失敗。此為測試自足性問題，不影響正式Gate。測試現同步mock `_validate_binary_runtime_preflight`，仍保留另有專門案例驗證缺模型／只缺Scores的正式preflight分流。

## 2026-08-04 — Formal consistency／coverage synthetic 閉環修正

### 狀態

`IMPLEMENTED / INDEPENDENT_SYNTHETIC_PASS / LOCAL_FORMAL_RERUN_REQUIRED`

### 程式與測試基準

- 輸入程式ZIP：`test-branch-1_20260804_143233_f1d6807.zip`
- 程式SHA256：`04e6bae036562c471b45d0ecbf00dd47f9ada3fec089b8b29fa0870e9d0db28b`
- Formal bundle：`to_chatgpt_bundle_20260804_143322_bf24a8be.zip`
- Bundle SHA256：`6bc34702d640a9d5503e190628268d7e925ccdec4b41f7656f08193ec327d893`
- Bundle結果：quick gate PASS、chain checks PASS、ML smoke PASS；consistency 1項FAIL；meta quality 4項coverage FAIL。

### 根因

Consistency唯一失敗為synthetic suite在`validate_breakout_quality_strategy_adaptation_contract_case`中直接讀取目前正式Binary workflow；`strategy_adapt._validate_fixed_contract`正確拒絕Binary hard-filter identity，導致整個coverage synthetic suite提前中止。Meta quality的`coverage_synthetic_suite_runs_successfully`、line／branch minimum與key targets四項失敗均為同一中止事件的連鎖結果，不是實際coverage退步。

完整registry往下執行另發現三類被首個例外遮蔽的測試問題：

1. Continuous模型研究與strategy adaptation案例未使用隔離workflow override，會受目前Binary正式設定影響。
2. Binary主選單新增子選單與訓練後正式策略驗證後，CLI fixture仍使用舊輸入序列、舊route、缺少`experiment_profile`及舊hard-filter argv。
3. `strategy_filter_gate.py`與`strategy_dl_filter_gate.py`的broad exception handler雖原樣重拋，但未綁定例外名稱，不符合正式traceability meta contract。

### 唯一變更

- `validate_breakout_quality_strategy_adaptation_contract_case`在測試內隔離覆寫continuous profile及strategy auto欄位，直接驗證score-ranking／Selection PIT identity、CLI fixed risk與position cap；測試結束後正式Binary config保持原值。
- `validate_dataset_cli_contract_case`將continuous流程改為隔離profile案例，並同步Binary子選單route、相對工件路徑、完整hard-filter identity及訓練後post-validation fixture。
- 兩個研究Gate的`except Exception`改為綁定例外名稱，仍先輸出已捕捉console後原樣`raise`；錯誤處理語意不變。

### 固定條件與工件需求

- 正式workflow仍為`unique_group_sampling / binary_classification / hard-filter / canonical_runtime / original buy-sort`。
- Continuous PIT ranking與strategy adaptation仍為CLI-only研究流程；runtime fixed contract沒有放寬。
- 不修改Dataset、Label、model checkpoint、research／runtime Scores、threshold、rolling params、交易引擎或策略績效口徑。
- 不需重訓模型、不需重新匯出Scores。

### 獨立驗證

- `validate_breakout_quality_strategy_adaptation_contract_case`：19／19 PASS。
- 完整synthetic registry：242 validators、4,169 checks、0 FAIL。
- 正式`apps/test_suite.py`依專案規定未由GPT執行；使用者套用patch後只需重跑本地formal suite，預期consistency與coverage meta quality恢復。

## 2026-08-04 — 9A Binary DL Hard Filter 正式 OOS 策略結果

### 狀態

`RESULT_AVAILABLE / CURRENT_FILTERS_PLUS_DL_REJECTED / DL_REPLACEMENT_GATE_PENDING / FORMAL_BASELINE_UNCHANGED`

### 程式與模型基準

- 程式ZIP：`test-branch-1_20260804_145148_ef17394.zip`
- SHA256：`00e6b0029b958bd3807868b772d65f0a1cf702277f781602e9e8bd437d15dd44`
- 模型：`breakout_quality_v1 / inception_time_v1 / unique_group_sampling`
- Threshold：固定`0.5`；不得依本次OOS或策略結果回調。
- 模型訓練：Epoch 2為最低Validation Loss `0.708591`；Epoch 3 early stop；完整Selection依2 epochs重訓。
- Runtime score：`forward_oos`；candidate rows `665,528`，model-scored `646,382`，conservative reject `19,146`。

### 固定條件與唯一差異

- 比較期間：`2021-01-01～2026-03-02`。
- 使用相同`base_finalist_best` rolling active params、原position-aware buy-sort、fixed risk、position cap、max positions、rotation、交易成本、成交與portfolio accounting。
- Baseline固定`use_breakout_quality_filter=False`；Active quality filter固定`use_breakout_quality_filter=True`。
- 現有EMA、BB、Volume、breakout return及false-breakout optional entry filters在兩組均依當期active params維持原設定。
- 不使用continuous Score ranking、R2、R3、Future Target或optimizer。

### 模型 OOS 結果

| 指標 | Selection | OOS | 判讀 |
|---|---:|---:|---|
| 原始PASS | 54.81% | 55.63% | OOS基準略高0.83pp |
| 模型PASS | 64.02% | 45.43% | OOS操作點明顯更保守 |
| PASS Precision | 60.95% | 62.99% | OOS高於原始PASS `+7.35pp` |
| PASS Recall | 71.19% | 51.44% | 錯殺48.56%的原始PASS |
| Accuracy | 59.21% | 56.17% | 較Selection下降3.04pp |
| PR-AUC | 0.6443 | 0.6257 | 仍有排序訊號但不足以保證策略效益 |
| 平均Score | 0.5184 | 0.4842 | OOS分布向下漂移 |
| ECE | 0.0308 | 0.0722 | OOS calibration惡化 |

### 正式策略結果：A目前filters＋無DL vs B目前filters＋Binary DL

| 指標 | A Baseline | B Binary DL | B−A |
|---|---:|---:|---:|
| 淨總報酬 | 129.08% | 85.79% | −43.29pp |
| 最大回撤 | 17.41% | 26.34% | +8.93pp |
| Return／MDD | 7.42 | 3.26 | −4.16 |
| 年化報酬 | 17.43% | 12.76% | −4.67pp |
| Log R² | 0.9436 | 0.7870 | −0.1567 |
| 月勝率 | 63.49% | 53.97% | −9.52pp |
| 交易數 | 407 | 396 | −11 |
| 勝率 | 41.52% | 37.37% | −4.15pp |
| Payoff | 2.95 | 2.80 | −0.16 |
| EV | 0.53R | 0.36R | −0.18R |
| 平均曝險 | 84.02% | 68.22% | −15.80pp |
| 平均每日可掛單候選 | 50.55 | 20.98 | −29.57 |
| 候選供給不足日 | 163日 | 383日 | +220日 |
| 每日結束未滿倉日 | 745日 | 661日 | −84日 |
| 每日結束持股缺口 | 1,871格日 | 2,360格日 | +489格日 |

年度報酬只有2023與2026改善；2021、2022、2024及2025均落後，且2022由`+7.90%`降至`−6.86%`。結果不是單一年份噪音，而是多數年度及報酬、回撤、交易品質、候選供給與資金使用同步惡化。

### 判定

1. **9A在事件Label上具有OOS分類訊號，但目前filters下作第二層hard filter正式失敗。** Precision提升不能補償Recall、候選供給、曝險、勝率、Payoff與EV下降。
2. 模型PASS由Selection 64.02%降至OOS 45.43%，而策略每日可掛單候選由50.55降至20.98；quality gate縮減候選池後造成更深的持股缺口與較低資金部署。
3. 交易數只減少11筆，但候選池與曝險大幅下降，表示主要傷害包含portfolio path displacement與替代交易品質下降，不是單純少交易。
4. **不得依本次OOS調threshold、epochs、learning rate、Label或模型。** Current-filters-plus-DL路徑標記`REJECTED`，正式runtime仍關閉Binary DL filter。
5. 此結果只完成A／B，不能用來否定使用者提出的「移除optional filters、Binary DL作唯一品質Gate」。真正replacement假設仍必須看F相較A的絕對績效。

### 下一步：執行 A／B／C／F Replacement Gate

直接使用既有model、manifest與forward-OOS scores，不需重新訓練或重新export：

```bash
python apps/breakout_quality.py strategy-dl-filter-gate --dataset full --param-policy base-finalist-best --max-positions 10 --rotation off
```

正式判讀順序：

1. `F−C`：Binary DL在optional filters全關後是否能單獨改善完全不篩選的突破候選池。
2. `F−A`：DL-only replacement能否至少打平目前正式rule-based filters策略；這是採用主判定。
3. `(F−C)−(B−A)`只用來判斷DL作替代者是否比疊加者更有效；正值不代表F絕對績效通過。
4. 若F仍顯著低於A，停止目前9A Binary DL Filter正式整合，不以同一OOS調threshold救援；下一個模型實驗必須提出本質不同的Label或representation機制。

### Dataset／Label／模型工件需求

本次結果回寫不修改程式、Dataset、Label、checkpoint或Scores。A／B／C／F Gate可直接使用目前完整工件執行；不需重訓、relabel、optimizer或PIT重建。

## 2026-08-04 — Binary DL Rule Ablation A0／B0 ～ A4／B4 Matrix

### 狀態

`IMPLEMENTED / LOCAL_FULL_DATA_EXECUTION_REQUIRED / FORMAL_BASELINE_UNCHANGED`

### 程式基準

- 輸入ZIP：`test-branch-1_20260804_182739_d9ddf41(2).zip`
- SHA256：`bc1f9aa5409141ea527342ebe422fa36c222e5d3014e488e85b1337bb8353f01`
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

### 前一輪 A／B／C／F 結果

固定期間`2021-01-01～2026-03-02`、`base-finalist-best` rolling active params、9A `inception_time_v1 / unique_group_sampling / threshold 0.5`、canonical runtime score及原position-aware buy-sort：

| 組別 | Optional filters | Binary DL | 總報酬 | MDD | RoMD | EV | 平均曝險 |
|---|---|---|---:|---:|---:|---:|---:|
| A | 目前 | 關 | 129.08% | 17.41% | 7.42 | 0.53R | 84.02% |
| B | 目前 | 開 | 85.79% | 26.34% | 3.26 | 0.36R | 68.22% |
| C | 全關 | 關 | 138.58% | 13.35% | 10.38 | 0.47R | 92.02% |
| F | 全關 | 開 | 145.92% | 18.31% | 7.97 | 0.35R | 79.04% |

判定：B明確拒絕；C為本輪風險調整後最佳研究情境；F相較C雖總報酬+7.34pp，但MDD+4.95pp、RoMD−2.41、EV−0.11R、直接交易選擇差異−21.83R，且年度改善主要集中2023，因此現有9A DL-only gate不採用。A仍為正式基準；C只保留研究候選，不直接升級正式策略。

### 唯一主要變更

將原A／B／C／F重新命名並擴充為單一五層矩陣：

| 層級 | A組 | B組 | Optional filters | 歷史門檻 | Re-entry | KC出場 |
|---:|---|---|---|---|---|---|
| 0 | A0：DL關 | B0：DL開 | 依原active params | 依原active params | 依原active params | 依原active params |
| 1 | A1：DL關 | B1：DL開 | 五項全關 | 依原active params | 依原active params | 依原active params |
| 2 | A2：DL關 | B2：DL開 | 五項全關 | 關 | 依原active params | 依原active params |
| 3 | A3：DL關 | B3：DL開 | 五項全關 | 關 | 關 | 依原active params |
| 4 | A4：DL關 | B4：DL開 | 五項全關 | 關 | 關 | 關 |

使用者原訊息最後一列寫為「A3 B4」，依層級與成對比較契約正規化為`A4／B4`。A4／B4只關閉KC，不關閉半倉停利、ATR初始停損、trailing、fixed-risk sizing、position cap、成交／費稅或原buy-sort。

### 架構與報表

- 五層均重用`strategy_compare.run_comparison`與正式portfolio engine／trade attribution，不複製成交、帳務或統計。
- `strategy_compare.run_comparison`新增內部`shared_param_overrides`，同時套用至同層A與B；同層仍只允許`use_breakout_quality_filter=False/True`一項差異。該參數不暴露為一般CLI，避免使用者任意建立未登錄組合。
- 合併報表不再列大量跨組pair；固定輸出：情境矩陣、每層A／B與`Bn−An`投組增量、交易品質／資金使用、A系列相對A0的規則消融、五層trade attribution及年度`Bn−An`矩陣。
- 新輸出目錄：`strategy_dl_filter_rule_ablation_gate_<param_policy>_canonical_runtime/`，內含五個pair目錄及單一Markdown／JSON摘要。

### 固定條件

- 不重建Dataset、不relabel、不重訓模型、不重新選epoch、不調threshold、不執行optimizer。
- 模型、runtime scores、active-param來源、比較期間、benchmark、max positions、rotation、fixed risk、position cap、買入排序、交易成本及portfolio accounting全部固定。
- 不使用continuous Score、R2、R3、Future Target或Selection PIT ranking。

### 採用判讀

1. 每層只以同層`Bn−An`判斷DL增量，避免將規則消融效果誤認為DL效果。
2. A系列`A1−A0`至`A4−A0`只判斷逐步移除規則對基礎策略的影響。
3. DL有效至少需同時改善Return／MDD、EV、直接交易選擇R及年度穩定性；總報酬單獨上升不足以通過。
4. 若只有關閉規則的A系列改善，而各層`Bn−An`持續惡化，結論應是規則本身需簡化，而不是9A DL有效。
5. 真實結果取得前，本實驗只標記`IMPLEMENTED`，不得預判A2／A3／A4或B2／B3／B4有效。

### Dataset／Label／模型工件需求

無需重建任何Dataset、Label、checkpoint或Score；直接沿用目前9A canonical forward-OOS工件與rolling active-param JSON執行CLI即可。


## 2026-08-04 — Binary DL Risk-only Parameter Adaptation A5／B5 Stage Gate

### 狀態

`IMPLEMENTED / LOCAL_FULL_DATA_EXECUTION_REQUIRED / A6_B6_BINARY_PIT_REQUIRED / FORMAL_BASELINE_UNCHANGED`

### 程式基準

- 輸入ZIP：`test-branch-1_20260804_201536_24f4bed(1).zip`
- SHA256：`18c91a1e36374212c3745fbb7dda1ce21900ac88caa9456201972848436a5c0e`
- 使用者另提供目前`roos_base_best.json`內容供契約核對；2021～2026六個effective dates的`use_history_threshold`皆為False，Re-entry僅2026為True，KC僅2021為True。
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

### 研究問題

前一輪A0／B0～A4／B4顯示，歷史門檻本來全期關閉；再關Re-entry與KC的績效差異接近零。後續不再維護逐層規則消融，報表只比較：

1. 原正式規則全套。
2. Rule-based filters全關。

使用者提出的下一個問題是：9A Binary DL失敗，是否因正式ROOS風險／執行參數是在DL關閉環境下訓練，未適應DL篩選後的候選與持倉分布。

### A5／B5 唯一主要變更

- 訓練規則固定Rule-based filters全關：五個optional entry filters、歷史門檻、Re-entry及KC均關閉。
- optimizer只搜尋四個風險／執行參數：`atr_len`、`atr_buy_tol`、`atr_times_init`、`atr_times_trail`。
- `high_len`、其他rule參數、TP、fixed risk、position cap、max positions、rotation、費稅與原position-aware buy-sort全部固定；非風險值依Baseline各effective date凍結。
- A5為DL關閉環境訓練出的risk-only rolling active params，回放時DL關閉。
- B5沿用A5完全相同的active-param schedule，回放時只開啟Binary DL hard filter。
- 完成A5後，另將同一套新風險值合併回原Baseline rule設定，報表同時輸出「原正式規則全套」與「Rule-based filters全關」兩列；不再逐項比較History、Re-entry或KC。

### 架構契約

- `strategies/breakout/search_space.py`的int／float解析新增session fixed override支援，使凍結欄位不再消耗trial維度。
- `tools/optimizer/outer_rolling_oos.py`新增fold-specific fixed override mapping；每個fold依OOS effective date建立session，objective、local-min、OOS diagnostics與active-param export共用同一固定契約。
- 新CLI：`strategy-dl-filter-param-adapt-gate`，research-only、CLI-only，不加入正式選單。
- 新輸出：`models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/`。

### A6／B6 無前視限制

A6／B6代表optimizer訓練期間就開啟Binary DL，再用同一套新參數分別回放DL關／開。第一個rolling fold的訓練期包含2011～2020；目前9A canonical forward-OOS score只從2020-12-31後開始，不能回灌歷史Selection訓練。

因此A6／B6必須先建立每個optimizer歷史日期可用、模型只讀取當時以前資料的Binary PIT scores，並將該PIT source接入optimizer runtime。以下均禁止作替代：

- 最終9A canonical forward-OOS scores。
- `research_scores.csv`。
- 最終模型Selection in-sample scores。

本輪只實作合法preflight與`BINARY_PIT_REQUIRED`狀態，不以缺分fallback或final model score偷跑A6／B6。只有A5／B5結果支持DL與風險參數可能存在協同作用時，才進入Binary PIT builder與A6／B6 runtime wiring。

### 固定條件

不重建Dataset、不relabel、不重訓9A模型、不調threshold 0.5、不改原buy-sort、不改fixed risk／position cap值、不改max positions、rotation、成交／費稅或portfolio accounting。A5／B5直接使用既有9A canonical runtime manifest／scores作OOS回放；optimizer訓練本身固定DL關閉，不讀該OOS score。

### 結果與採用判定

目前只有實作與契約驗證，尚無本機完整rolling optimizer及replay結果，不得判定A5／B5有效或無效。正式策略仍維持A0與既有正式ROOS。

A5／B5至少需同時檢查Return／MDD、EV、直接交易選擇R與年度穩定性；總報酬單獨改善不足以通過。A6／B6尚未執行，且不得在Binary PIT缺失時開始。

## 2026-08-04 — Binary DL 4 Parameters × 2 DL States Risk Adaptation Gate

### 狀態

`IMPLEMENTED / LOCAL_BINARY_PIT_AND_ROLLING_EXECUTION_REQUIRED / FORMAL_BASELINE_UNCHANGED`

### 程式基準

- 輸入ZIP：`test-branch-1_20260804_213150_b044668(1).zip`
- SHA256：`d6cb8e3e2fa4f47d6b56ae3d239ee80b867bfa54a71688b28444a1b9299cc866`
- 本節取代前一節只做到A5／B5且將A6／B6停在`BINARY_PIT_REQUIRED`的暫時編排；歷史紀錄保留，但後續命名統一為P0～P3、A0／B0～A3／B3。

### 最終比較矩陣

| 參數組 | 參數來源 | Rule-based規則 | DL關 | DL開 |
|---|---|---|---|---|
| P0 | 原`base-finalist-best` ROOS | 原正式active-param設定 | A0 | B0 |
| P1 | 原`base-finalist-best` ROOS | filters全關 | A1 | B1 |
| P2 | filters全關、optimizer訓練時DL關 | filters全關 | A2 | B2 |
| P3 | filters全關、optimizer訓練時DL開 | filters全關 | A3 | B3 |

P2與P3只搜尋`atr_len`、`atr_buy_tol`、`atr_times_init`及`atr_times_trail`。`high_len`、TP、fixed risk、position cap、max positions、rotation、費稅、買入排序與其他非風險參數依Baseline各effective date固定，不得成為trial維度。

### Binary PIT無前視契約

新增`build-binary-point-in-time-scores` CLI與`binary_point_in_time` runtime source。P3每個optimizer歷史日期只能讀取expanding-window PIT Scores；每個fold的Train／Validation／完整refit均只包含score start前且`label_eval_end_date`已完成的groups，score period不得參與訓練或epoch selection。每列分數保存`fold_id`及`model_information_cutoff`，且必須滿足cutoff早於score date。

PIT source同時傳入optimizer主process、fold session及平行workers，並在search前驗證manifest／scores identity。以下來源一律禁止替代：

- 最終9A canonical forward-OOS scores。
- `research_scores.csv`。
- 最終模型Selection in-sample scores。

缺少或identity不一致時fail-fast，不得靜默回退。

### 報表與主要判讀

單一報表固定包含：

1. A0／B0至A3／B3八操作點。
2. `B0−A0`至`B3−A3`的DL增量。
3. `A1−A0`、`A2−A1`、`B2−B1`、`A3−A1`及`B3−B1`的規則／參數效果。
4. 最終公平比較`B3−A2`。
5. interaction=`(B3−A3)−(B2−A2)`，只判斷DL-aware參數是否改善DL增量。
6. P0/P1、P2與P3逐年風險參數差異。

`B3−A2`至少需同時改善Return／MDD、EV、直接交易選擇R及年度穩定性；總報酬或interaction單獨為正不足以採用。

### 固定邊界

本輪不重建Dataset、不relabel、不調9A threshold 0.5、不改原position-aware buy-sort、不改fixed risk／position cap值、不改max positions、rotation、成交／費稅或portfolio accounting。Binary PIT會逐fold訓練歷史模型，但不覆蓋正式9A checkpoint／manifest／scores。正式策略仍維持原ROOS且Binary DL runtime關閉，直到本機完整4×2結果通過。

## 2026-08-05 — 4×2 Binary DL Parameter Adaptation Formal Meta Quality 閉環

### 狀態

`IMPLEMENTED / FORMAL_QUICK_CONSISTENCY_CHAIN_ML_PASS / META_CHECKLIST_FIXED / LOCAL_FORMAL_RERUN_REQUIRED`

### 程式與測試基準

- 輸入程式ZIP：`test-branch-1_20260804_235803_1be64ec.zip`
- 程式SHA256：`09142fb9c119bd3a59c7a86573956db8a9a5fecda9623152e15d633651a1c859`
- Formal bundle：`to_chatgpt_bundle_20260804_235944_65c06eef.zip`
- Bundle SHA256：`f4d7fea784bb08d47b581107554d0e36d5204ea135423363a2f9d0867787e52f`
- Bundle結果：quick gate PASS、consistency PASS、chain checks PASS、ML smoke PASS；meta quality僅Checklist G兩項FAIL。

### 根因與唯一變更

- `B187`最新兩筆4×2 contract transition被附加在同日`T284`後方，違反Checklist G依日期再依ID排序。
- `T284`另新增`DONE -> DONE`列，沒有實際狀態變更，違反收斂紀錄只能記錄真實transition的契約。
- 本輪只將兩筆`B187` transition移回同日既有`B187`區塊，並移除`T284 DONE -> DONE` no-op列；不修改程式、Binary PIT、optimizer、Dataset、Label、model、Scores、策略參數或4×2報表。

### 固定條件與結果邊界

- 4種參數×DL關／開八操作點與P2／P3 optimizer實作維持不變。
- `T284`最新有效狀態仍由既有`PARTIAL -> DONE`列表示；擴充coverage屬同一DONE契約的內容更新，不另創造狀態transition。
- 本次只修正機械治理文件；不需重訓模型、重建Binary PIT或重跑4×2研究。

### 驗證與下一步

- GPT端獨立檢查須確認Checklist G排序、status chain、no-op guard、摘要映射與Markdown欄數全部通過。
- 正式`apps/test_suite.py`依專案規定不由GPT執行；使用者套用patch後重跑本地formal suite，預期meta quality恢復PASS。

## 2026-08-05 — 4×2 Binary DL Result Source-Identity Correction

### 狀態

`RESULT_RECEIVED / B_ARMS_INVALID_FOR_FINAL_4X2 / REPLAY_SOURCE_FIXED / LOCAL_REPLAY_RERUN_REQUIRED / FORMAL_BASELINE_UNCHANGED`

### 程式與結果基準

- 使用者結果ZIP：`test-branch-1_20260805_063715_38f58a7.zip`
- SHA256：`c07d449a8cdad96e39e2d87c60edb5613aeaf900df954d20742e2a1d28d75d9d`
- 使用者提供完整Binary PIT建置、P3 DL-on rolling optimizer及A0／B0至A3／B3輸出。
- Binary PIT期間：`2006-12-01～2026-03-02`；21 folds；P3 optimizer為200 trials／fold。

### 已確認結果

不使用DL的A系列不依賴Score source，因此下列結果有效：

| 組別 | 參數 | 規則 | 報酬 | MDD | RoMD | EV | 曝險 |
|---|---|---|---:|---:|---:|---:|---:|
| A0 | P0原ROOS | 原正式規則 | 129.08% | 17.41% | 7.42 | 0.53R | 84.02% |
| A1 | P1原ROOS | filters全關 | 138.09% | 13.35% | 10.34 | 0.47R | 92.01% |
| A2 | P2 DL-off-trained | filters全關 | 166.69% | 15.41% | 10.82 | 0.73R | 92.13% |
| A3 | P3 DL-on-trained | filters全關、replay時DL關 | 119.77% | 11.83% | 10.12 | 1.16R | 92.06% |

A2相較A1：報酬`+28.60pp`、MDD`+2.06pp`、RoMD`+0.48`、EV`+0.27R`；顯示在DL關閉環境下重訓四個ATR風險／執行參數具有研究價值。A3具有最高EV與最低MDD，但總報酬及RoMD均低於A2。

### 發現的source identity錯誤

P3 optimizer訓練與rolling OOS診斷使用`binary_point_in_time` Scores；但後續四個strategy comparison pair仍由`strategy_compare`預設讀取`canonical_runtime`。因此目前B0／B1／B2／B3是「固定final 9A model」診斷，不是原4×2設計要求的同一Binary PIT DL state。

尤其目前報表的`B3−A2 = -88.01pp`不可作最終公平比較，因B3並非使用P3 optimizer所看到的DL runtime。P3 optimizer的PIT OOS_CHAIN與canonical B3 replay之間的差異，不能解讀為參數失敗或模型失敗，必須先統一source後重跑。

另4×2總表讀取不存在的`trades`欄位，八組交易數全部顯示0；個別strategy comparison中的`trade_count`仍正確，屬合併報表顯示錯誤。

### 唯一修正

- `strategy_compare.run_comparison`新增內部research-only `hard_filter_source`，可將Binary PIT manifest／scores透過正式`breakout_quality_filter_source_context`傳入portfolio replay；一般正式hard-filter CLI仍維持canonical runtime契約。
- 4×2 Gate的四個pair均固定使用同一份Binary PIT source，DL關閉A臂雖不讀Score，DL開啟B臂則與P3 optimizer完全同源。
- replay期間固定為Baseline first OOS date `2021-01-01`至Binary PIT `available-through`，不回放PIT早期但無active params的區間。
- 4×2總表交易數改讀正式`trade_count`。

### 固定條件

不重建Dataset、不relabel、不調threshold、不重建已完成的Binary PIT、不重跑P2／P3 optimizer。只需在修正版上重跑4×2 CLI；既有PIT與P2／P3 active-param identity相同時應直接重用，重新執行四個pair replay與合併報表。

### 採用判定

目前只可判定A2是無DL研究候選；Binary DL是否在PIT-consistent 4×2下有效仍未取得合法結果。正式策略維持A0、Binary DL runtime關閉。修正版結果取得前，不以目前B0～B3或`B3−A2`作模型／參數採用結論。

## 2026-08-05 — 4×2 Binary PIT Process-worker Source Propagation Correction

### 狀態

`RESULT_RERUN_RECEIVED / B_ARMS_INVALIDATED_AGAIN / WORKER_SOURCE_PROPAGATION_FIXED / LOCAL_REPLAY_RERUN_REQUIRED / FORMAL_BASELINE_UNCHANGED`

### 程式與結果基準

- 使用者結果ZIP：`test-branch-1_20260805_171121_f19c053.zip`
- SHA256：`836bb3eb84a882e5f1b68dd670736235b438dcc56858946ddc4fc4046136ff50`
- 使用者依前一版source修正重新執行4×2 Gate；報表已顯示`Score source=binary_point_in_time`，且八操作點交易數由錯誤的0修正為407／396、384／446、336／374、275／278。

### 重新追查結果

四個B臂的報酬、MDD、RoMD、EV、曝險、交易數及年度結果仍與前一版canonical replay逐項完全相同。這不是可直接接受的「PIT與canonical恰好一致」證據；程式追查確認實際source仍未傳入建立訊號的Windows process workers：

1. `strategy_compare._run_scenario`只在主程序進入`breakout_quality_filter_source_context`的ContextVar。
2. active-param replay透過`prepare_trial_inputs`建立`ProcessPoolExecutor`，Windows使用spawn；子程序不繼承主程序ContextVar。
3. `filters/breakout_quality/runtime.py`雖已有process environment fallback，但strategy replay沒有設定`BREAKOUT_QUALITY_FILTER_SCORE_SOURCE`、`BREAKOUT_QUALITY_BINARY_PIT_MANIFEST`及`BREAKOUT_QUALITY_BINARY_PIT_SCORES`。
4. 因此子程序在`generate_signals`時仍使用預設`canonical_runtime`；報表metadata只反映主程序指定來源，沒有證明worker實際使用來源。

所以本次B0／B1／B2／B3及`B3−A2 = -88.01pp`再次失效，不得作最終4×2採用判定。A0～A3不啟用DL、不讀Score，結果仍有效；P3 optimizer原本已用受控environment傳遞Binary PIT，P3參數工件本身不需重訓。

### 唯一修正

- `filters/breakout_quality/runtime.py`新增統一execution context：同時設定主程序ContextVar與可由spawned workers繼承的三項受控environment，scenario結束後逐項還原原值。
- `strategy_compare._run_scenario`改用統一execution context，確保portfolio signal-prep process workers與主程序讀取同一Binary PIT manifest／scores。
- strategy comparison metadata新增`hard_filter_source_transport=contextvar_and_process_environment`；4×2報表明示`process workers已傳遞`。
- Synthetic contract新增真實Python子程序探針，不再只mock函式參數；直接驗證worker看到`binary_point_in_time`及完全相同manifest／scores路徑，並驗證離開scenario後主程序environment完整還原。

### 固定條件與重跑需求

不重建Dataset、不relabel、不重建Binary PIT、不重跑P2／P3 rolling optimizer、不調threshold 0.5、不改風險搜尋欄位、原buy-sort、資金帳務或正式runtime。套用修正後只重新執行4×2 CLI；既有Binary PIT與P2／P3參數identity相同時應直接重用，只重跑四個strategy comparison pairs與合併報表。

### 採用判定

目前仍只接受A2為無DL研究候選；正式策略維持A0、Binary DL runtime關閉。取得process-worker一致的重新回放結果前，不採用任何B臂或`B3−A2`結論。

## 2026-08-05 — 4×2 Binary PIT Process-worker Consistent Final Result

### 狀態

`RESULT_VALID / PROCESS_WORKER_BINARY_PIT_CONFIRMED / CURRENT_9A_BINARY_DL_REJECTED / A2_NO_DL_RESEARCH_CANDIDATE / FORMAL_BASELINE_UNCHANGED`

### 程式與結果基準

- 使用者結果ZIP：`test-branch-1_20260805_175702_4d89c30.zip`
- SHA256：`d2089bb2e4cda0ddd539d2876d48db984e90f9154abc58c9061fe620945b19ce`
- 使用者依process-worker source修正版重新執行`strategy-dl-filter-param-adapt-gate`。
- 報表明示`DL replay source=binary_point_in_time（八操作點一致；process workers已傳遞）`。
- A0～A3與前次完全一致；B0～B3相較前次canonical誤用結果明顯改變，確認Binary PIT source已實際進入建立訊號的spawned workers，而不只是metadata顯示改變。
- 比較期間：`2021-01-01～2026-03-02`；Binary threshold固定`0.5`；原position-aware buy-sort、max positions 10、rotation off、資金帳務與交易成本固定。

### 八操作點主要結果

| 組別 | 參數／規則 | DL | 報酬 | MDD | RoMD | EV | 曝險 | 交易數 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A0 | P0原ROOS／原正式規則 | 關 | 129.08% | 17.41% | 7.42 | 0.53R | 84.02% | 407 |
| B0 | P0原ROOS／原正式規則 | 開 | 143.73% | 25.96% | 5.54 | 0.44R | 66.52% | 399 |
| A1 | P1原ROOS／rules全關 | 關 | 138.09% | 13.35% | 10.34 | 0.47R | 92.01% | 384 |
| B1 | P1原ROOS／rules全關 | 開 | 184.32% | 19.22% | 9.59 | 0.54R | 76.00% | 454 |
| A2 | P2 DL-off-trained／rules全關 | 關 | 166.69% | 15.41% | 10.82 | 0.73R | 92.13% | 336 |
| B2 | P2 DL-off-trained／rules全關 | 開 | 150.25% | 27.03% | 5.56 | 0.65R | 77.98% | 385 |
| A3 | P3 DL-on-trained／rules全關 | 關 | 119.77% | 11.83% | 10.12 | 1.16R | 92.06% | 275 |
| B3 | P3 DL-on-trained／rules全關 | 開 | 99.78% | 18.27% | 5.46 | 0.59R | 76.16% | 289 |

### Binary DL增量

| 比較 | Δ報酬 | ΔMDD | ΔRoMD | ΔEV | Δ曝險 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| B0−A0 | +14.65pp | +8.55pp | -1.88 | -0.09R | -17.50pp | 拒絕；總報酬增加不足以覆蓋MDD、RoMD、EV與年度穩定性惡化 |
| B1−A1 | +46.23pp | +5.86pp | -0.75 | +0.07R | -16.01pp | 不採用；報酬與EV改善，但RoMD下降、MDD提高，且收益高度集中2023、2022為-12.16% |
| B2−A2 | -16.44pp | +11.63pp | -5.26 | -0.08R | -14.16pp | 明確拒絕 |
| B3−A3 | -20.00pp | +6.44pp | -4.66 | -0.57R | -15.91pp | 明確拒絕 |

最終公平比較`B3−A2`為：報酬`-66.91pp`、MDD`+2.86pp`、RoMD`-5.36`、EV`-0.14R`、曝險`-15.98pp`。interaction=`(B3−A3)−(B2−A2)=-3.56pp`，表示在DL開啟環境重訓風險參數沒有形成正協同，反而使DL增量略為更差。

### 採用判定

1. 現有9A Binary DL模型、目前Label及threshold 0.5在PIT-consistent 4×2下正式拒絕；不得promote至正式runtime。
2. B1雖為八組最高總報酬，但未通過既定採用契約：MDD及RoMD惡化、年度不穩定，且目前使用者貼出的總表未包含直接交易選擇R，不能以單一總報酬宣稱DL有效。
3. A2是目前最有價值的無DL研究候選：相較A1報酬`+28.60pp`、RoMD`+0.48`、EV`+0.27R`，代價是MDD`+2.06pp`；仍屬研究候選，不直接取代正式A0。
4. A3雖有最低MDD與最高EV，但總報酬及RoMD均低於A2；不作首選。
5. 正式策略維持A0與原ROOS，Binary DL runtime維持關閉。

### 後續順序

1. 無DL策略線：先對A2／P2做參數選擇穩定性、fold／年度依賴與base-best／agree／ensemble一致性驗證，再決定是否進入正式promotion程序。
2. DL研究線：停止對目前9A Label繼續調threshold或重訓風險參數；下一個模型改測chronological first-touch Binary Label。
3. 新Label先固定使用A2／P2參數做`DL關 vs DL開`Gate；只有同時改善RoMD、EV、直接交易選擇R及年度穩定性，才投入Binary PIT與DL-on optimizer。不得一開始就重做P3型昂貴參數適應。
4. 若要解釋B1的高總報酬來源，應讀取其`trade_attribution.json`／`trade_attribution.md`；該歸因未包含在使用者本次貼出的console，因此本節不推測獨有交易選擇R。

## 2026-08-05 — A2 Realized Trade-path Label Model Workflow Implementation

### 狀態

`IMPLEMENTED / MODEL_RESULT_NOT_AVAILABLE / STRATEGY_GATE_NOT_RUN / FORMAL_BASELINE_UNCHANGED`

### 研究目的

現有9A Binary Label只描述固定40 bars的MFE／MAE價格機會，未完整對齊A2正式買入限價、初次miss buy後延續候選、initial stop、trailing、indicator exit與淨額帳務。使用者要求先由互動選單確認新Label模型Prediction效果，再以CLI和A2 no-DL／舊Label DL比較策略績效。

### 新Label契約

- Label ID：`a2_realized_trade_path_v1`
- 獨立filter ID：`breakout_quality_a2_trade_path_v1`，不得覆蓋9A `breakout_quality_v1`。
- Event scope：一個原始breakout setup的完整生命週期；Feature snapshot固定原始signal date。
- 初次miss buy只維持pending／continuation，不標REJECT；後續回到原始limit成交後沿用同一event。
- 新setup會依正式策略覆蓋舊延續訊號；shadow completion／invalidation與永未成交均標EXCLUDED並排除Binary訓練。
- 已成交交易直接重用正式entry plan、extended shadow state、`execute_bar_step`與exact accounting；淨`realized_net_r > 0`標PASS，其餘完整已成交交易標REJECT。若資料結尾仍持有已成交部位，必須沿用單股正式回測的最後交易日強制結算，不得另標未結算排除。
- 同一ticker/date group只有當日A2 active `high_len`事件取得有效Label，其餘high_len rows維持INVALID。

### Teacher與資料鏈

- 2014～2020：以既有Selection rolling baseline建立rules全關／DL關、只搜尋`atr_len / atr_buy_tol / atr_times_init / atr_times_trail`的歷史P2 teacher schedule。
- 2021～2026：重用既有P2 DL-off-trained active params。
- 每個事件只使用當日已生效teacher params；兩段schedule不得effective-date重複。
- 衍生Dataset沿用9A 300×10 feature bank與group arrays，只重建event labels、events metadata與summary；source inventory與artifact SHA256仍須一致。

### 選單與CLI

Binary模型研究子選單改為：

```text
[1/Enter] 建立新Label → 重新訓練 → 模型預測報表
[2] 使用既有模型 → 更新Scores → 模型預測報表
[3] 查看Label與事件生命週期摘要
```

兩條模型路徑都在Selection／OOS Prediction報表後停止，不匯出runtime scores、不執行策略回放。

策略比較改由CLI-only `strategy-trade-path-label-gate`執行；只使用既有凍結模型，更新forward-OOS scores後固定比較：A2 Base（DL關）、Old Label 9A（DL開）、New Trade-path Label（DL開）。主判定為`New−Base`，輔助為`New−Old`，兩個pair的A2 no-DL base必須逐指標一致。

### 採用限制

本次只完成資料、模型與Gate流程，尚無新Label模型Prediction或策略結果。正式策略維持A0；A2維持無DL研究候選；9A Binary DL runtime維持關閉。不得依未執行結果宣稱新Label有效。

## 2026-08-05 — A2 Trade-path Historical Teacher OOS Boundary Fix

### 狀態

`IMPLEMENTED / LABEL_BUILD_RETRY_REQUIRED / MODEL_RESULT_NOT_AVAILABLE / FORMAL_BASELINE_UNCHANGED`

### 問題

使用者由模型研究選單執行`建立新Label → 重新訓練 → 模型預測報表`時，歷史teacher前置檢查將Selection rolling baseline的`meta.last_oos_date=2020-12-01`誤當成最後active-param生效日，硬性要求`2020-01-01`，因此在建立Label前錯誤中止。rolling工件中的`last_oos_date`代表最後OOS可覆蓋月份；年度active params的最後生效日才是`2020-01-01`，兩者語意不同。

### 唯一修正

- 歷史teacher基準仍要求`meta.first_oos_date=2014-01-01`，且最後OOS邊界必須位於2020年並涵蓋`2020-01-01`。
- 不再把`meta.last_oos_date`硬編碼為`2020-01-01`。
- 改為直接驗證`params_ensemble_by_effective_date`必須恰好完整包含`2014-01-01`至`2020-01-01`七個年度生效日；缺少、增加或日期不合法均fail-fast。
- 新增direct synthetic regression，確認`last_oos_date=2020-12-01`合法，且缺少2020年度active params仍會被拒絕。

### Dataset／結果

本修正只改teacher工件期間驗證，不改Dataset、Label公式、Feature、模型、threshold、策略參數或帳務。先前失敗發生在Label建置開始前，因此沒有可沿用的新Label模型結果；使用者需由原選單重新執行。正式策略維持A0，A2仍為無DL研究候選。

## 2026-08-05 — A2 Trade-path Single-stock Parity Contract v2

### 狀態

`IMPLEMENTED / DATASET_REBUILD_REQUIRED / MODEL_RETRAIN_REQUIRED / RESULT_NOT_AVAILABLE / FORMAL_BASELINE_UNCHANGED`

### 程式基準

- 使用者ZIP：`test-branch-1_20260805_191236_8de0a17(1).zip`
- SHA256：`409785c2860ba8301ef06063f493177726fbdb85c7c2e9c8799dd8d60453f1a3`
- 本輪以全新解壓基準修改，交付只包含異動檔案。

### 問題與契約澄清

同一股票、同一原始signal date、相同OHLCV、相同策略參數及相同明示single-stock sizing capital下，TP1與單股正式回測必須取得完全相同的掛單、miss buy、pending／continuation、成交、停損／停利／trailing／indicator exit、最後交易日結算、進出價格、淨PnL與Realized R。差異只可存在於Dataset後處理：未成交終局不應被當作REJECT，而應以EXCLUDED排除Binary訓練。

舊實作雖重用正式entry與bar-step核心，但仍有三項不足：

1. TP1對已成交但資料結尾仍持倉的事件標記`insufficient_future_after_fill`，單股正式回測則以最後交易日收盤價強制結算，造成終局分叉。
2. Dataset以`INVALID=-1`混合表示合法排除、資料錯誤與右設限，對外無清楚PASS／REJECT／EXCLUDED狀態契約。
3. 正式成交紀錄未完整保存signal、entry與terminal execution context，無法對TP1逐筆比對entry date／price、exit date／price／reason、PnL及R。

### 唯一變更

- Label contract升級為version 2，正式狀態固定為`PASS=1`、`REJECT=0`、`EXCLUDED=-1`；既有數值`-1`保留相容，但使用者可見語意不再稱INVALID。
- PASS只由完整交易的`realized_net_r > 0`產生；REJECT由完整交易的`realized_net_r <= 0`產生。首次miss buy本身永不直接產生REJECT。
- 未成交事件被新setup覆蓋、shadow終止、資料結尾仍未成交，以及teacher／active high_len／source-date等不可形成合法Label的事件，均以具體`label_reason`寫入EXCLUDED。
- 已成交但資料結尾仍持倉時，TP1直接重用正式`finalize_open_position_at_end()`，以與單股回測相同的最後交易日收盤價及exact accounting形成PASS或REJECT，並記錄`FORCED_CLOSEOUT`。
- TP1與單股正式交易紀錄均保存signal date、entry type／date／price、exit date／price／reason、sizing capital、淨PnL與R，供逐事件parity檢查。
- Dataset summary與loader新增status／reason count、contract version及完整欄位一致性驗證；未知reason、status與數值Label不一致、PASS／REJECT缺少進出路徑或EXCLUDED帶有realized結果均fail-fast。

### 固定條件

不改Feature、原始signal-date snapshot、teacher active-param schedule、模型architecture、experiment profile、threshold、PIT隔離、策略參數、正常進出規則、exact-accounting公式或正式策略基準。TP1 sizing使用同一筆事件明示的single-stock sizing capital；本輪不引入投組資金、持股上限或候選競爭語意。

### Dataset／模型影響

Label schema與已成交資料尾端終局已變更，既有trade-path Dataset及其模型不可直接視為contract v2工件。必須重新建立Label Dataset並重新訓練；尚未取得Selection／OOS Prediction或策略Gate結果，不得預判有效。正式策略仍維持A0，A2維持無DL研究候選，9A Binary DL runtime維持關閉。

### 獨立驗證

- Initial fill後STOP：TP1與單股正式回測逐項一致。
- Initial miss後continuation fill再IND_SELL：entry／exit date與price、reason、淨PnL及R逐項一致。
- Initial fill後先執行半倉停利，再由IND_SELL結束剩餘部位：最終淨PnL及R與單股正式結果一致。
- 完整虧損交易：標REJECT，且與單股正式結果一致。
- 首次miss後始終未成交：標EXCLUDED，不得標REJECT。
- 已成交後資料結尾仍持倉：兩邊均走正式`FORCED_CLOSEOUT`並逐項一致。
- 11種`label_reason`均映射到唯一PASS／REJECT／EXCLUDED狀態。

### 下一步

使用正式互動選單：`模型研究與驗證` → `Binary模型研究` → `建立新Label → 重新訓練 → 模型預測報表`。完成後先檢查Label狀態／原因摘要與Selection／OOS Prediction；策略比較仍只由既有CLI-only Gate進行。

## 2026-08-05 — A2 Trade-path Label Model Prediction Result

### 狀態

`RESULT_VALID / MODEL_OOS_WEAK_AND_UNSTABLE / STRATEGY_GATE_PENDING / FORMAL_BASELINE_UNCHANGED`

### 程式與結果基準

- 使用者結果ZIP：`test-branch-1_20260805_230725_ff04a05.zip`
- SHA256：`8a616fed6646360219c7acba214fe5753a9d043cbf09a9a8be3317e9713d22a0`
- 使用者依trade-path single-stock parity contract v2重新建立Label、重新訓練`inception_time_v1 / unique_group_sampling`，並完成Selection／OOS Prediction報表。
- Dataset、Label、模型、threshold及split均由本次console結果確認；策略Gate尚未執行。

### P2 historical teacher穩定性資訊

建立2014～2020 historical teacher schedule時，7-fold rolling optimizer同時輸出三種參數選擇：

| Policy | Fold OOS AVG RoMD | 0050 | OOS_CHAIN RoMD | 0050 |
|---|---:|---:|---:|---:|
| base-finalist-best | 1.78 | 1.78 | 5.01 | 5.60 |
| base-finalists-agree | 2.04 | 1.78 | 6.84 | 5.60 |
| seed ensemble | 1.78 | 1.78 | 5.01 | 5.60 |

`base-finalists-agree`在本段歷史teacher OOS_CHAIN較佳；但本次TP1 Label實驗的teacher identity已預先固定為`base-finalist-best`，不得在看到模型OOS後改換teacher並混稱同一實驗。這項結果只作A2／P2參數穩定性證據，不改變本次Label工件。

### Dataset與訓練結果

- Events：`1,793,028`
- PASS：`7,573`
- REJECT：`9,952`
- EXCLUDED：`1,775,503`
- Binary eligible：`17,525`；整體PASS率約`43.21%`。
- Inner Train：`5,847` groups；Validation：`1,730` groups；Final Selection Refit：`7,628` groups；固定OOS：`9,263` groups。
- Inner Validation最佳Epoch：`2`；最低Validation Loss：`0.648880`；完整Selection依selected epochs重訓2 epochs，Final Loss：`0.633495`。

### Selection／OOS主要結果

| 指標 | Selection | OOS | OOS判讀 |
|---|---:|---:|---|
| 原始PASS | 39.09% | 44.64% | OOS基準較高5.55pp |
| 模型PASS | 20.02% | 11.93% | OOS通過量顯著下降 |
| PASS Precision | 56.32% | 48.05% | OOS僅高於原始PASS 3.41pp |
| PASS Recall | 28.84% | 12.84% | OOS錯殺87.16%的真PASS |
| PR-AUC | 0.5240 | 0.4796 | 僅略高於OOS prevalence 0.4464，排序能力弱 |
| Brier | 0.2220 | 0.3006 | OOS惡化 |
| ECE | 0.0070 | 0.2082 | 明顯校準漂移 |
| 平均Score | 0.3914 | 0.2613 | OOS分數整體下移 |

年度模型PASS比例高度不穩定：2021 `0.16%`、2022 `53.73%`、2023 `13.57%`、2024 `0.20%`、2025 `21.65%`、2026 partial `0.00%`。這不是單純threshold略偏，而是明顯跨年度score distribution／calibration shift；不得使用同一OOS回頭調threshold、epochs、feature、Label或訓練設定。

### 採用判定

1. 模型層只取得微弱OOS Precision增益，且Recall、PR-AUC、校準與年度穩定性不足；不得直接promote。
2. 仍依預先定義流程執行一次策略層Gate，確認硬篩選後是否意外形成RoMD、EV、直接交易選擇R與年度穩定性的共同改善。
3. 若`New−Base`未同時通過上述四項，停止此trade-path Label模型線，不調OOS threshold，也不投入Binary PIT與DL-on optimizer。
4. 正式策略維持A0；A2仍為無DL研究候選；現有9A Binary DL runtime仍關閉。

## 2026-08-05 — Trade-path Label Strategy Gate Decision-contract Completion

### 狀態

`IMPLEMENTED / STRATEGY_GATE_RERUN_REQUIRED / DATASET_AND_MODEL_REUSE / FORMAL_BASELINE_UNCHANGED`

### 問題

預定採用契約要求`New−Base`同時改善RoMD、EV、直接交易選擇R及年度穩定性，但既有`strategy-trade-path-label-gate`合併報表未讀取pair內已產生的`trade_attribution.json`，因此沒有顯示`exclusive_selection_delta_r`。此外，兩個pair的A2 no-DL base只比較七個摘要數值，未驗證equity、trade history及daily capacity是否逐工件完全一致。直接執行舊Gate會得到不足以做最終判定的報表。

### 唯一修正

- Gate schema升級為version 2，讀取Old／New pair各自的`trade_attribution.json`並fail-fast驗證有限`exclusive_selection_delta_r`。
- 三組策略總表新增「直接選擇R」；差異表固定輸出`New−Base`、`New−Old`與`Old−Base`的直接選擇R。
- 判讀契約明示：只有`New−Base`同時改善RoMD、EV、直接交易選擇R及年度穩定性，才可進入Binary PIT與DL-on optimizer。
- 兩個pair除既有base摘要一致性外，新增`no_filter_equity.csv`、`no_filter_trades.csv`與`no_filter_daily_capacity.csv`逐檔SHA256完全一致檢查；任何差異立即中止。
- 合併JSON保存Old／New完整trade attribution及三個base identity SHA256；正式工件清單同時列出兩份歸因Markdown。

### 固定條件與工件影響

不改Dataset、Label、模型checkpoint、research scores、threshold 0.5、A2／P2 active params、rules全關、原buy-sort、max positions、rotation、risk、position cap、費稅或策略帳務。既有Dataset與模型可直接沿用；只需在修正版執行一次CLI-only strategy Gate。

### 獨立驗證

- Direct synthetic確認總表與差異表顯示Old `-2.50R`、New `+3.25R`及`New−Old=+5.75R`。
- Direct synthetic確認三個duplicate base工件完全相同時接受，任一檔案內容不同即fail-fast。
- 原trade-path single-stock parity 15項加上本次2項Gate contract，共17項全部通過。

### 下一步

套用本修正後執行既有CLI-only `strategy-trade-path-label-gate`；不需重新建立Label或重新訓練。取得Gate結果前，不判定新Label有效。



## 2026-08-06 — Config-driven獨立策略績效比較App

### 狀態

`IMPLEMENTED / PERFORMANCE_RESULT_PENDING / DATASET_LABEL_MODEL_REUSE / FORMAL_BASELINE_UNCHANGED`

### 程式基準

- 輸入：`test-branch-1_20260805_232145_20f7566.zip`
- SHA256：`a0db0a6924664d20d85b0f8d6a89bd2516cf79a69872a1db9a0cc5e4394bf22f`
- 唯一變更：模型訓練、策略參數訓練與策略績效比較入口分離；未修改TP1 Dataset、Label、模型架構、模型權重、threshold、P2／P3參數或策略交易語意。

### 實作契約

- `apps/breakout_quality.py`只負責Breakout Quality Dataset／Label／模型訓練與模型評估；模型流程完成research／forward-OOS score工件後停止，不執行portfolio replay。
- 新增獨立正式入口`apps/strategy_compare.py`；選單只顯示「執行目前比較設定」與「查看目前比較設定與工件狀態」，不顯示C1～C6、TP1或其他特定版本名稱。
- `apps/strategy_compare.py`已納入quick-gate的help與inline CLI正式入口registry，避免新增App未被入口檢查覆蓋。
- `config/strategy_compare.py`逐項條列parameter sources、DL sources、arms及contrasts，每項以`enabled`獨立開關；已移除代表整套實驗的active comparison ID。
- `core/strategy_comparison.py`提供泛用schema、跨欄驗證與config fingerprint；Breakout Quality orchestration與canonical engine位於`filters/breakout_quality/`。舊`filters/breakout_quality/strategy_compare_engine.py`只保留相容別名。
- 比較流程只讀取既有模型、Scores與ROOS工件；缺件時於replay前fail-fast，不建立Label、不訓練模型、不匯出缺少Scores，也不執行optimizer。
- 輸出自動使用enabled arm IDs與config fingerprint建立`outputs/strategy_compare/runs/`工件，並更新`outputs/strategy_compare/latest/`；manifest保存設定snapshot與輸入工件SHA256。

### Dataset／Label／模型影響

不需重建Dataset、不需重新Label、不需重新訓練TP1。此次只有App、config、模型score交付邊界與比較 orchestration 架構調整；不改Label、模型權重或策略交易語意。策略績效數值尚未執行，因此不得標記為有效或無效。

### 下一步

先由`apps/strategy_compare.py`查看目前設定與工件狀態；若目前config啟用的參數或DL工件缺失，回到各自的模型／optimizer入口建立，不由比較App代辦。工件完整後執行目前比較設定，取得實際績效結果，再依預先定義contrast回頭判斷TP1是否需要調整。

### 2026-08-06｜獨立策略比較 App formal bundle 閉環（ACCEPTED）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `ACCEPTED`；只修正 synthetic validator 對 canonical engine 的來源定位，不改 Dataset、Label、模型、Scores、threshold、ROOS、策略執行或報表數值 |
| 程式基準 | 使用者本輪輸入 `test-branch-1_20260806_011335_c378133.zip`；formal bundle 為 `to_chatgpt_bundle_20260806_011510_48e506c0.zip` |
| Formal 結果 | quick gate PASS、chain checks PASS、ml smoke PASS；consistency 5,165 PASS／30 SKIP／2 FAIL。兩個 FAIL 均因 validator 仍從 legacy alias `filters/breakout_quality/strategy_compare_engine.py` 搜尋 runtime token；canonical engine 已依新分層移至 `filters/breakout_quality/strategy_compare_engine.py`。Meta quality 的 coverage 行79.11%、分支61.11%均達標，唯一 FAIL 是 synthetic suite 連帶失敗 |
| 唯一變更 | `qualified_candidate_audit_is_cli_only_and_reuses_canonical_replay` 與 `candidate_counterfactual_cli_only_and_sidecar_is_not_replay_counts` 改讀 canonical engine；synthetic registry 的相關 `impacted_modules` 也改指向 canonical engine，確保後續該檔異動會觸發既有策略契約。legacy alias 本身仍由獨立 config-driven App contract 驗證為相容轉接，不要求複製 runtime 實作 token |
| Dataset／Label | 不重建、不 relabel |
| Selection／OOS | 未重訓、未重跑模型或策略，無新 Selection／OOS 數值 |
| 採用判定 | 修正 validator 架構定位；維持 `apps/strategy_compare.py`、`config/strategy_compare.py`、`filters/breakout_quality/strategy_compare_engine.py` 為正式分層 |

## 2026-08-06 — Strategy Compare選單前置工件閉環

### 狀態

`IMPLEMENTED / PERFORMANCE_RESULT_PENDING / DATASET_LABEL_MODEL_REUSE / FORMAL_BASELINE_UNCHANGED`

### 程式基準

- 輸入：`test-branch-1_20260806_012720_453c286(2).zip`
- SHA256：`723851ac35ba95688a845ee13f2c4a5575724fcb03ccb108e97ab866fac7ae08`
- 唯一變更：策略比較App依`config/strategy_compare.py`建立可稽核的前置依賴計畫，對可由既有正式工件確定產生的缺件自動重用、建立、重建或接續；不修改TP1 Dataset、Label、模型架構、模型權重、threshold、交易語意或既有ROOS。

### 實作契約

- `apps/strategy_compare.py`維持泛化常駐選單：「執行目前比較設定／查看設定、工件與預計動作」。一般操作不再要求先輸入零散CLI。
- `config/strategy_compare.py`集中arms、contrasts、parameter／DL sources、forward score推論設定、Min-DL rolling trials與`auto_prepare／reuse／rebuild／resume／confirmation`政策，方便使用者直接檢視與調整。
- 狀態分為`READY／PREPARABLE／BLOCKED`；執行前完整顯示`REUSE／BUILD／REBUILD／BLOCKED／RUN／REPORT`計畫並只確認一次。
- 已有模型但缺少或過期正式forward-OOS scores時，透過正式`filters/breakout_quality/export_scores.py`共用服務依config補匯出，不重新訓練模型。
- 比較所需Min-DL ROOS缺少或identity過期時，透過正式`filters/breakout_quality/strategy_param_training.py`共用服務建立或接續指定parameter set；不順帶執行舊4×2 replay。
- 缺少模型checkpoint、模型identity不相容或缺少其他不可自行推導的上游真理工件時標記`BLOCKED`，不建立Label、不選模型、不訓練模型權重。
- 任一前置步驟失敗即停止後續portfolio replay，錯誤回報包含artifact key、動作與專案相對路徑；已完成optimizer工件保留供下次接續，不產生不完整正式比較報表。
- 正式比較JSON與manifest升級保存preparation policy／plan、config snapshot、啟用arms／contrasts與輸入工件SHA256。
- `doc/PROJECT_SETTINGS.md`新增選單優先、config集中、前置工件閉環、入口分離及一次確認等長期泛化條款。

### Dataset／Label／模型影響

不需重建Dataset、不需重新Label、不需重新訓練TP1。若既有TP1 checkpoint完整，缺少forward-OOS scores只做推論匯出；Min-DL參數僅在目前config啟用且工件缺少／過期時執行策略參數optimizer。

### 採用判定與下一步

本輪只完成操作與工件依賴閉環，尚未取得新策略績效，不能判定TP1有效或無效。套用後由`python apps/strategy_compare.py`進入選單，先查看預計動作，再執行目前設定；程式自動補齊可準備工件並完成實際比較，取得結果後再依config中的contrasts判讀TP1。

## 2026-08-06 — Strategy Compare前置閉環 Formal Bundle修正

### 狀態

`ACCEPTED / FORMAL_FAILURE_ROOT_CAUSES_CLOSED / DATASET_LABEL_MODEL_REUSE / PERFORMANCE_RESULT_UNCHANGED`

### 程式與Formal基準

- 使用者ZIP：`test-branch-1_20260806_095542_dec6ccd.zip`
- ZIP SHA256：`8327673830c90179f3639226729ef2b3b67110c3053ba7a5c39f998bd9bb6027`
- Formal bundle：`to_chatgpt_bundle_20260806_095639_662f5f81.zip`
- Bundle SHA256：`5f203a2a8d7fc96221b14807c823cc16e8e538f380c78f3ce1411105c53095a1`
- 使用者本地結果：quick gate PASS、chain checks PASS、ml smoke PASS；consistency因synthetic suite TypeError失敗1項；meta quality的4項coverage失敗皆由同一synthetic suite提前中止連帶造成。

### 根因與唯一修正

1. `filters/breakout_quality/export_scores.py`的`_resolve_forward_export_write_paths()`新增必填`model_architecture`後，正式forward-OOS匯出呼叫與兩個runtime-artifact synthetic fixture仍沿用舊呼叫契約。已在三個呼叫點明確傳入architecture，且helper使用傳入值解析canonical writable path。
2. continuous-ranker contract仍從legacy alias `tools/filters/breakout_quality/export_scores.py`搜尋binary-profile阻擋token；正式實作已移至`filters/breakout_quality/export_scores.py`。validator改讀canonical正式模組，不要求legacy alias複製runtime實作。
3. 完整synthetic suite跑通後，Torch會在專案root產生後刪除動態JIT來源`_remote_module_non_scriptable`；coverage JSON生成會因已刪除來源拋出`NoSource`。consistency與meta-quality coverage均新增`_remote_module_*`omit契約，並以synthetic regression固定此行為。

### 獨立驗證

- Runtime artifact contract：65／65 PASS。
- Continuous ranker contract：15／15 PASS。
- 全部synthetic registry：245個validators、4,210個checks，0 FAIL。
- Coverage target scope：line 78.68%（門檻55%）、branch 61.22%（門檻50%）；所有key targets均存在且有命中，critical files line／branch門檻全部通過。
- 本輪未執行`apps/test_suite.py`或正式五步流程；使用者需在本機重新執行正式測試確認double check。

### Dataset／模型／策略影響

不重建Dataset、不重新Label、不重新訓練模型、不匯出新正式Scores、不改threshold、不改P2／P3 ROOS、不改交易或帳務語意，也沒有新Selection／OOS或策略績效結果。本輪只修正正式匯出參數契約與測試／coverage基礎設施。


## 2026-08-06 — TP1 Binary PIT mixed-label前置失敗修正

### 狀態

`IMPLEMENTED / P3_PREPARATION_RERUN_REQUIRED / DATASET_LABEL_MODEL_REUSE / PERFORMANCE_RESULT_PENDING`

### 程式與錯誤基準

- 使用者ZIP：`test-branch-1_20260806_175431_9167541.zip`
- ZIP SHA256：`8db2f114c86c96623d9492413c31a3c006806ba9e3cec81189061059f83925aa`
- 使用者已由`apps/strategy_compare.py`成功自動匯出TP1 forward-OOS scores；後續重建`Min-DL-TP1 ROOS`時，Binary PIT前置於`continuous ranker發現同group混合binary label`停止，因此尚未執行C1～C6績效比較。

### 根因

`tools/filters/breakout_quality/build_binary_point_in_time_scores.py`錯誤重用continuous-ranker的`_group_table()`。該helper要求同一feature group的所有event rows具有完全相同Label；但TP1 trade-path Dataset刻意保留同一`ticker/date`下的一筆teacher-active PASS／REJECT，以及其他`inactive_high_len_for_teacher` EXCLUDED rows。EXCLUDED並非binary target，不應被判定為PASS／REJECT衝突。

### 唯一修正

- Binary PIT改用專用group representative契約，不再依賴continuous-ranker helper。
- 每個feature group優先選擇第一筆eligible PASS／REJECT作為訓練代表；若沒有eligible binary row，使用第一筆event row並標為非target。
- EXCLUDED rows不再造成mixed-label失敗；若同group真正同時存在互相衝突的eligible PASS與REJECT，仍立即fail-fast。
- Binary PIT schema升級為version 2，fold fingerprint與manifest記錄新的group representative contract，避免重用舊語意工件。
- Binary DL參數適應synthetic contract新增TP1 PASS／REJECT＋EXCLUDED group案例，並將Binary PIT builder納入該validator的impacted modules。

### Dataset／Label／模型影響

不重建Dataset、不重新Label、不重新訓練TP1模型、不改threshold、forward-OOS scores或交易語意。使用者剛匯出的正式forward scores可直接沿用；只需重新由策略比較選單執行目前設定，程式會重建Binary PIT、接續P3 optimizer並在工件完整後執行C1～C6。

### 採用判定與下一步

本輪只關閉前置工件builder錯誤，尚無新策略績效結果，不得據此判定TP1有效。套用修正後由`python apps/strategy_compare.py`選擇`[1/Enter] 執行目前比較設定`；既有forward scores應顯示REUSE，Binary PIT與Min-DL參數依config自動建立／接續，完成後再產生正式比較報表。

## 2026-08-06 — TP1 Binary PIT historical Selection coverage contract修正

### 狀態

`IMPLEMENTED / P3_PREPARATION_RERUN_REQUIRED / DATASET_LABEL_MODEL_REUSE / PERFORMANCE_RESULT_PENDING`

### 程式與錯誤基準

- 使用者ZIP：`test-branch-1_20260806_180934_7c85422.zip`
- ZIP SHA256：`a20fb6f2bc38d0edb90f1cd7f33dc09f3c86a8b90242d19e0be84cb7e5e8d0e4`
- 使用者已成功建立Binary PIT v2，合法score期間為`2016-03-01～2026-03-02`；P3前置隨後因validator要求完整覆蓋optimizer歷史Selection `2011-01-01～2025-12-31`而停止，尚未執行P3 optimizer或C1～C6績效比較。

### 根因

正式Binary PIT runtime原本就把score開始日前候選全部pass-through，語意等同DL-off；score期間內才依PIT score啟用hard filter，score尾端之後若出現候選則fail-fast。舊P3 preflight卻額外要求Binary PIT從120個月Selection最早日開始100%覆蓋，與runtime SSOT矛盾，也要求模型在尚無足夠Inner Train／Validation資料時產生不可能的歷史分數。

### 唯一修正

- P3不再要求不可能的完整歷史coverage；Binary PIT必須與optimizer Selection有實際重疊，且score尾端必須至少覆蓋最新Selection結束日。
- PIT開始日前固定採`pass_through_dl_off`；PIT期間內無對應候選分數維持既有`conservative_reject`；PIT尾端過期維持`fail_on_candidate`。
- 逐rolling fold輸出`bootstrap_fallback_only／partial_score_history／full_score_history`、Selection期間、PIT重疊期間與calendar coverage；本次`2016-03-01`起點相對2021～2026六個120月Selection folds皆為partial history，coverage約48.4%逐年提高至98.4%，加權約73.4%。
- coverage policy、逐fold audit與score identity一併進入P3 runtime identity；P3目錄新增`binary_pit_optimizer_coverage.csv`，`rolling_preflight.json`與`rolling_optimizer_summary.json`保存同一coverage契約。
- Binary PIT／TP1 Dataset、Label、模型權重、threshold、forward scores、交易規則與Full／Min ROOS均不修改。

### 採用判定與下一步

此修正只關閉前置契約矛盾，不能視為TP1有效。套用後由`apps/strategy_compare.py`選擇`[1/Enter] 執行目前比較設定`；既有Binary PIT應直接重用，P3 optimizer依config建立或接續，完成後才執行C1～C6並取得正式績效結果。

## 2026-08-06 — Strategy Compare前置完成後status contract修正

### 狀態

`IMPLEMENTED / C1_C6_REPLAY_RERUN_REQUIRED / P3_ARTIFACT_REUSE / PERFORMANCE_RESULT_PENDING`

### 程式與錯誤基準

- 使用者ZIP：`test-branch-1_20260806_183741_032ae37.zip`
- ZIP SHA256：`2824d9d5d48422a81f9c0f8c88e71a3fbd80b872c0d96166e85e1882933239f4`
- 使用者已成功完成Binary PIT與Min-DL參數前置，並輸出P3相關計畫／報表；其後正式C1～C6 replay尚未開始，即於`filters/breakout_quality/strategy_comparison.py`建立run directory前發生`KeyError: config_fingerprint`。

### 根因

`run_strategy_comparison()`最初取得的是正式orchestration status，包含`config_fingerprint`、`artifact_identities`與`resolved_parameter_paths`。當前置builder完成後，程式卻直接採用`strategy_compare_preparation.collect_artifact_status()`回傳的低階readiness payload；該payload刻意不負責config fingerprint，因此覆蓋正式status contract，直到後續讀取`status["config_fingerprint"]`才失敗。

### 唯一修正

- 前置builder完成後，不再把低階preparation payload直接當成正式orchestration status。
- 新增正式post-preparation refresh：透過`filters/breakout_quality/strategy_comparison.collect_artifact_status()`重新計算目前工件identity與最終config fingerprint，並恢復resolved parameter paths。
- refresh後必須同時具備`config_fingerprint`、`artifact_identities`、`resolved_parameter_paths`與`preparation_plan`，且整體狀態必須為`READY`；缺欄位或仍非READY時在portfolio replay前以可追蹤RuntimeError停止，不再出現延遲KeyError。
- 保留首次執行計畫與requested fingerprint，正式輸出仍同時記錄requested／final preparation plan及前後fingerprint。
- `validate_strategy_compare_config_driven_app_contract_case`新增直接post-preparation狀態交接案例，覆蓋完整欄位恢復與缺少fingerprint拒絕。

### Dataset／Label／模型／參數影響

不重建Dataset、不重新Label、不重新訓練TP1模型、不改threshold、Binary PIT、forward scores、交易規則或帳務。使用者本輪已完成的P3／Min-DL工件可直接重用；本次只修正前置完成後的狀態交接與輸出identity閉環。

### 採用判定與下一步

尚未取得C1～C6實際績效，不得判定TP1有效或無效。套用修正後由`python apps/strategy_compare.py`進入選單並執行目前比較設定；若P3工件identity完整，計畫應顯示REUSE並直接進入C1～C6 replay與正式報表。

## 2026-08-06 — Strategy Compare execution-pair runtime contract修正

### 狀態

`IMPLEMENTED / C1_C6_REPLAY_RERUN_REQUIRED / ALL_PREREQUISITE_ARTIFACTS_REUSE / PERFORMANCE_RESULT_PENDING`

### 程式與錯誤基準

- 使用者ZIP：`test-branch-1_20260806_195324_e5bfab8.zip`
- ZIP SHA256：`0752a7163153bd168d7cea473409a787d53f02704d0596c485604735bf8baf88`
- 使用者狀態頁與執行計畫均為`READY`，TP1 model／manifest／forward scores、Full ROOS、Min ROOS與Min-DL-TP1 ROOS皆可重用；按下執行後，在第一個正式pair replay前發生`NameError: _execution_pairs is not defined`，因此尚未產生C1～C6績效結果。

### 根因

config-driven策略比較重構後，`run_strategy_comparison()`保留對`_execution_pairs(settings)`的呼叫，但正式orchestration模組遺漏該helper本體。既有validator只驗證config、前置計畫、status refresh與fingerprint，沒有實際建立execution pairs或走完整mocked replay orchestration，因此compile／import與前置契約均通過，直到真實READY路徑才發生NameError。

### 唯一修正

- 在`filters/breakout_quality/strategy_comparison.py`新增通用`_execution_pairs()`，只依目前啟用arms，以config順序建立每個`param_source／rule_policy`的canonical DL-off／DL-on pair。
- runtime再次檢查同group不得重複DL狀態、不得缺少off／on任一側，且DL-on arm必須有`dl_id`；停用arm不會被隱性執行。
- `validate_strategy_compare_config_driven_app_contract_case`新增兩層直接回歸：一是驗證目前config產生三個正確pair；二是mock正式`run_comparison()`走完整READY orchestration，確認每個pair各執行一次、六個enabled arms皆進入正式payload並成功寫出報表／JSON／manifest。

### Dataset／Label／模型／參數影響

不重建Dataset、不重新Label、不重新訓練TP1模型、不改threshold、Binary PIT、forward scores、交易規則、帳務、Full ROOS、Min ROOS或Min-DL-TP1 ROOS。使用者已完成的所有前置工件可直接重用；本輪只修正正式比較pair建立與測試覆蓋。

### 採用判定與下一步

尚未取得C1～C6實際績效，不得判定TP1有效或無效。套用修正後由`python apps/strategy_compare.py`進入選單並執行目前比較設定；計畫應維持全部`REUSE`，隨後直接進入三個canonical pair replay與正式報表。

## 2026-08-06 — Strategy Compare Min ROOS forward-source與期間coverage preflight修正

### 狀態

`IMPLEMENTED / C1_C6_REPLAY_RERUN_REQUIRED / FULL_PAIR_PRIOR_OUTPUT_DIAGNOSTIC_ONLY / DATASET_LABEL_MODEL_REUSE / PERFORMANCE_RESULT_PENDING`

### 程式與錯誤基準

- 使用者ZIP：`test-branch-1_20260806_200539_c64241b.zip`
- ZIP SHA256：`8101c7d857ec0ad49b810b6d47a29bf4262a552c5dad1bd158bf1d53659cb6db`
- 使用者所有前置工件顯示READY後開始C1～C6；第一組Full ROOS／formal pair已完成並輸出pair-level工件，第二組Min ROOS／all-off在正式replay前因active params只涵蓋`2014-01-01～2020-12-31`、比較期間為`2021-01-01～2026-03-02`而停止，尚未產生完整C1～C6頂層報表。

### 根因

`config/strategy_compare.py`的`min_roos`誤指向`trade_path_label/a2_teacher_params/p2_dl_off_trained`。該工件是TP1 Label建立使用的Selection歷史teacher，只涵蓋2014～2020；正式forward績效比較應使用`binary_dl_filter_param_adaptation/risk_only_rolling/p2_dl_off_trained`，其rolling schedule與正式Baseline同為2021～2026。原狀態頁只驗證JSON種類與selector，沒有在第一個pair前驗證所有參數來源是否完整覆蓋共同comparison period，因此錯誤被延遲到第二組replay才暴露。

### 唯一修正

- `min_roos`改指向forward P2 DL-off-trained active params與`rolling_preflight.json`，並在config配置P2正式builder；缺少、identity不符或期間不足時由策略比較選單自動建立／接續，不再要求臨時CLI。
- 狀態服務由全部啟用DL runtime工件解析共同可比較期間；若config明確指定日期則驗證其位於所有runtime工件範圍內，否則使用共同交集。本次固定為`2021-01-01～2026-03-02`。
- 每個rolling active-param來源在產生執行計畫時即驗證coverage；不完整者標為`PARAM_PERIOD_MISMATCH`並依builder轉為BUILD／REBUILD或BLOCKED。所有來源READY後才建立run directory並開始第一個pair replay。
- 前置服務改為多波依賴重新規劃：例如先建立forward scores取得正式期間，再自動發現並建立因此顯露為缺少／過期的P2／P3，直到READY或明確無進展／BLOCKED。
- P2-only參數建立不再把Binary PIT或模型checkpoint視為訓練必要條件；P3契約、Binary PIT、threshold與既有P3工件不變。

### Dataset／Label／模型／參數影響

不重建Dataset、不重新Label、不重新訓練TP1模型、不調threshold、不修改forward scores、Binary PIT、交易規則、帳務、Full ROOS或P3。若本機forward P2工件已存在且identity／coverage正確，重跑時直接REUSE；否則依config自動建立或接續P2。先前失敗run中的Full pair輸出只屬可追蹤pair診斷，不是完整C1～C6正式結果，不作採用判定。

### 採用判定與下一步

尚未取得完整C1～C6，不得判定TP1有效或無效。套用修正後由`python apps/strategy_compare.py`選擇`[2] 查看設定、工件與預計動作`，確認Min ROOS路徑位於`binary_dl_filter_param_adaptation/risk_only_rolling/p2_dl_off_trained`且整體為READY或PREPARABLE；再選`[1/Enter] 執行目前比較設定`。全部前置完成後重跑三個canonical pairs並產生頂層正式報表。


## 2026-08-06 — TP1 C1～C6正式績效結果與A9 runtime比較擴充

### 狀態

`C1_C6_ACCEPTED / TP1_HARD_FILTER_REJECTED / MIN_DL_TP1_NOT_PROMOTED / A9_RUNTIME_COMPARISON_IMPLEMENTED / A9_RESULT_PENDING`

### 程式與結果基準

- 使用者ZIP：`test-branch-1_20260806_202837_283d61c(1).zip`
- ZIP SHA256：`4ca1949169372cdab3bd3a55d0ec0813cd3406651ae4b399e85b67ba065944a9`
- 正式比較期間：`2021-01-01～2026-03-02`
- Dataset：`full`
- Param policy：`base-finalist-best`
- Max positions：10
- Rotation：off
- 固定條件：同一參數組內只切hard filter；原position-aware buy-sort、費稅、帳務與active-param無前視契約不變。

### C1～C6主要結果

| Arm | 組合 | 報酬 | MDD | RoMD | EV | 曝險 | 交易 |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 | Full ROOS × DL-off | 129.08% | 17.41% | 7.42 | 0.53R | 84.02% | 407 |
| C2 | Full ROOS × DL-TP1 | 55.04% | 10.93% | 5.04 | 0.87R | 38.19% | 158 |
| C3 | Min ROOS × DL-off | 166.69% | 15.41% | 10.82 | 0.73R | 92.13% | 336 |
| C4 | Min ROOS × DL-TP1 | 88.85% | 16.58% | 5.36 | 0.96R | 46.52% | 163 |
| C5 | Min-DL-TP1 ROOS × DL-off | 129.11% | 12.96% | 9.96 | 1.35R | 91.96% | 278 |
| C6 | Min-DL-TP1 ROOS × DL-TP1 | 21.42% | 10.54% | 2.03 | 0.65R | 51.20% | 146 |

- TP1同參數直接效果全部惡化：`C2−C1=-74.03pp / -2.38 RoMD / -79.90R直接選擇R`；`C4−C3=-77.84pp / -5.46 / -89.76R`；`C6−C5=-107.69pp / -7.93 / -280.31R`。
- TP1造成候選、曝險與交易數大幅下降；C5→C6平均曝險`91.96%→51.20%`、交易`278→146`，直接選擇R為`-280.31R`，證明不是單純交易變少，而是排除方向本身為負。
- C5相對C3降低MDD並提高單筆EV，但報酬`-37.58pp`、RoMD`-0.86`，因此Min-DL-TP1參數不晉升。
- 目前研究基準改為`C3 = Min ROOS × DL-off`；TP1 Binary hard filter不部署，且不再用同一OOS回頭調threshold或訓練條件。

### A9加入比較的唯一變更

- `config/strategy_compare.py`新增第二個DL source：`A9 = breakout_quality_v1 / inception_time_v1 / unique_group_sampling / threshold 0.5`。
- 新增三個runtime arms：`C7 Full ROOS × DL-A9`、`C8 Min ROOS × DL-A9`、`C9 Min-DL-TP1 ROOS × DL-A9`。
- 新增A9相對同參數DL-off contrasts：`C7−C1`、`C8−C3`、`C9−C5`；以及TP1相對A9 contrasts：`C2−C7`、`C4−C8`、`C6−C9`。
- 比較引擎由「每組只能一個DL-on」泛化為「一個共用DL-off基準可掛多個DL-on模型」；每個DL-on仍獨立執行controlled pair，重複基準的摘要與年度報酬必須完全一致，否則fail-fast。
- A9只作runtime hard-filter比較；本輪不建立A9-aware Min-DL參數、不覆蓋TP1 P3、不重新訓練A9或TP1模型。

### Dataset／Label／模型影響

不重建Dataset、不重新Label、不重新訓練TP1或A9模型、不改threshold、不改P2／P3 ROOS、交易規則或帳務。若A9正式forward-OOS scores缺少或過期，`apps/strategy_compare.py`依config使用既有A9 checkpoint自動補匯出。

### 下一步

由`python apps/strategy_compare.py`選擇`[2] 查看設定、工件與預計動作`，確認A9 model／manifest／forward scores為READY或PREPARABLE；再選`[1/Enter] 執行目前比較設定`。取得C7～C9後，以同參數A9效果及TP1−A9 contrasts判定TP1問題是Label／模型特有，或Binary hard-filter共同結構問題。

## 2026-08-07｜Min-DL identity與顯示名稱收斂（IMPLEMENTED）

### 狀態

`IMPLEMENTED / RESULT_PENDING`

### 唯一變更

- 使用者要求移除沒有物理意義的跨版本組合：DL-aware參數若在某一DL版本開啟環境下訓練，runtime DL-on時必須使用同一版本。
- 使用者可見名稱簡化為：`Min ROOS`、`Min ROOS: TP1-on`、`Min ROOS: A9-on`、`Min-TP1 ROOS`、`Min-TP1 ROOS: DL-on`、`Min-A9 ROOS`、`Min-A9 ROOS: DL-on`。
- `Full ROOS`同樣採`Full ROOS`、`Full ROOS: TP1-on`、`Full ROOS: A9-on`顯示。
- 移除原本`Min-DL-TP1 ROOS × DL-A9`這類`trained_with_dl_id != runtime dl_id` arm及其contrasts。
- 新增A9專屬`Min-A9 ROOS` P3來源；A9 P3工件獨立存於`models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/p3_dl_on_trained/A9/`，既有TP1 P3路徑不變。
- `core/strategy_comparison.py`新增硬性identity guard，任何DL-aware參數與不同DL runtime配對會在replay前fail-fast。

### 固定條件

不重建Dataset、不重新Label、不重新訓練TP1或A9模型、不改threshold、交易規則、帳務、position-aware buy-sort或既有TP1 P3。A9若缺少專屬Min-A9 P3，`apps/strategy_compare.py`依config透過既有正式策略參數builder建立／接續，不由策略比較App訓練模型權重。

### 新正式比較矩陣

| DL-off基準 | DL-on比較 | 物理意義 |
|---|---|---|
| Full ROOS | Full ROOS: TP1-on | 同Full參數，只切TP1 runtime |
| Full ROOS | Full ROOS: A9-on | 同Full參數，只切A9 runtime |
| Min ROOS | Min ROOS: TP1-on | 同Min參數，只切TP1 runtime |
| Min ROOS | Min ROOS: A9-on | 同Min參數，只切A9 runtime |
| Min-TP1 ROOS | Min-TP1 ROOS: DL-on | TP1-trained參數只配TP1 |
| Min-A9 ROOS | Min-A9 ROOS: DL-on | A9-trained參數只配A9 |

### Dataset／Label／模型重建需求

- Dataset：不需。
- Label：不需。
- TP1／A9模型：不需。
- A9 forward-OOS scores：缺少或過期時依既有模型自動補匯出。
- Min-A9 P3：缺少或identity／coverage不符時依config建立或接續。

### 結果與採用判定

本輪只有程式與契約修改，尚未取得新的策略績效，因此不得預先判定A9-aware參數有效或無效。既有C1～C6結果不變。

### 下一步

由`python apps/strategy_compare.py`進入正式選單，先選`[2] 查看設定、工件與預計動作`確認Min-A9 P3為READY或PREPARABLE，再以`[1/Enter] 執行目前比較設定`取得六個合法controlled pairs。


## 2026-08-07 — C1～C10正式結果：TP1／A9 hard filter與matched Min-DL參數適應結案

### 狀態

`C1_C10_ACCEPTED / TP1_HARD_FILTER_REJECTED / A9_HARD_FILTER_REJECTED / MIN_TP1_NOT_PROMOTED / MIN_A9_NOT_PROMOTED / MIN_ROOS_REMAINS_RESEARCH_BASELINE / NEXT_ROBUSTNESS_AND_CAPACITY_PRESERVING_A9`

### 程式與結果基準

- 使用者ZIP：`test-branch-1_20260807_140716_a6358d3.zip`
- ZIP SHA256：`de9916b323d141422a9a695a7fc2fcdc46853317d692fc2cfdd5fd2ad26e2fa0`
- 正式比較期間：`2021-01-01～2026-03-02`
- Dataset：`full`
- Param policy：`base-finalist-best`
- Max positions：10
- Rotation：off
- Config fingerprint：`62b4dfabe42a`
- 固定條件：同一參數組內只切指定DL runtime；原position-aware buy-sort、費稅、帳務、active-param無前視與盤前資金鎖定契約不變。

### Dataset／Label／模型／參數重建需求

- Dataset：不重建。
- Label：不重建；A9沿用既有MFE／MAE Label，TP1沿用既有realized trade-path Label。
- A9／TP1模型權重與threshold：不重訓、不調整；固定既有forward-OOS scores與threshold `0.5`。
- Min ROOS、Min-TP1 ROOS、Min-A9 ROOS：重用本次正式比較已建立且identity／coverage通過的rolling active-param工件；本節只記錄實際forward-OOS策略結果與採用判定。

### C1～C10主要結果

| Arm | 組合 | 報酬 | MDD | RoMD | 年化 | EV | 曝險 | 交易 | 同參數DL選擇R |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | Full ROOS | 129.08% | 17.41% | 7.42 | 17.43% | 0.53R | 84.02% | 407 | 0.00R |
| C2 | Full ROOS: TP1-on | 55.04% | 10.93% | 5.04 | 8.87% | 0.87R | 38.19% | 158 | -79.90R |
| C3 | Min ROOS | 166.69% | 15.41% | 10.82 | 20.95% | 0.73R | 92.13% | 336 | 0.00R |
| C4 | Min ROOS: TP1-on | 88.85% | 16.58% | 5.36 | 13.12% | 0.96R | 46.52% | 163 | -89.76R |
| C5 | Min-TP1 ROOS | 129.11% | 12.96% | 9.96 | 17.44% | 1.35R | 91.96% | 278 | 0.00R |
| C6 | Min-TP1 ROOS: DL-on | 21.42% | 10.54% | 2.03 | 3.83% | 0.65R | 51.20% | 146 | -280.31R |
| C7 | Full ROOS: A9-on | 85.79% | 26.34% | 3.26 | 12.76% | 0.36R | 68.22% | 396 | -76.47R |
| C8 | Min ROOS: A9-on | 142.70% | 21.75% | 6.56 | 18.76% | 0.64R | 78.24% | 374 | -6.12R |
| C9 | Min-A9 ROOS | 119.77% | 11.83% | 10.12 | 16.49% | 1.16R | 92.06% | 275 | 0.00R |
| C10 | Min-A9 ROOS: DL-on | 78.67% | 21.53% | 3.65 | 11.91% | 0.55R | 76.79% | 278 | -166.38R |

### Controlled判定

- TP1 hard filter正式否決：`C2−C1=-74.03pp / -2.38 RoMD / -79.90R`、`C4−C3=-77.84pp / -5.46 / -89.76R`、`C6−C5=-107.69pp / -7.93 / -280.31R`。三套參數下直接選擇R皆大幅為負，不再做TP1 threshold、TP1-aware optimizer或TP1 capacity-preserving延伸。
- A9 hard filter同樣不得部署：`C7−C1=-43.29pp / -4.16 RoMD / -76.47R`；`C8−C3=-23.99pp / -4.26 / -6.12R`；`C10−C9=-41.10pp / -6.47 / -166.38R`。matched A9-aware參數未能修復hard-filter結構，C10相對C8報酬再差`64.03pp`、RoMD再差`2.91`，因此不再做A9-aware risk optimizer。
- `C3 = Min ROOS`仍是全期最高報酬與最高RoMD arm，且相對C1為`+37.61pp報酬 / -2.00pp MDD / +3.40 RoMD`；但年度優勢高度集中於2023：C3在2023為`80.22%`，其餘年度為`23.08% / 0.84% / 11.21% / 2.25% / 4.85%`。作為只讀診斷，排除2023後逐年報酬鏈結約為C3 `47.98%`、C1 `70.71%`、C9 `81.53%`，因此C3尚不得直接晉升正式策略，下一步先做selector／fold／年度集中度穩定性Gate。
- C9雖總報酬低於C3，但MDD `11.83%`、RoMD `10.12`且六個年度皆為正；這只視為參數穩定性線索，不視為Min-A9部署證據，因其參數是在A9-on訓練環境下選得而runtime為DL-off。
- A9唯一保留的研究理由位於Min ROOS：C8相對C3的同參數直接選擇R只有`-6.12R`，但曝險下降`13.90pp`且總報酬下降`23.99pp`，顯示主要損失可能來自hard filter刪除候選後的資金／持股路徑，而非強烈反向的單筆選擇。後續若繼續DL，只測A9的capacity-preserving使用方式；不再測TP1。

### 報表語意修正

本次正式報表暴露一個attribution顯示問題：`direct_selection_delta_r`只在相同`param_source`與`rule_policy`的DL-off基準／DL-on arms之間有共同物理基準。舊報表對`C6-C4`、`C10-C8`、`C6-C1`、`C10-C1`等跨參數contrast仍直接相減兩個不同attribution，數值雖可算但沒有controlled direct-selection意義。正式報表已改為只對同參數／同rule-policy contrasts顯示`Δ同參數DL選擇R`；跨參數contrast固定顯示`-`。原C1～C10策略績效本身不受此修正影響。

### 下一步順序

1. 先做`Min ROOS` promotion robustness gate：重用既有rolling optimizer工件，比較正式selector／seed ensemble／fold與年度集中度，不重新調參；若優勢只由2023或單一selector造成，維持C1正式基準。
2. 只有A9保留DL研究：以C3為固定基準，實作capacity-preserving A9。A9不得再作eligibility hard reject；只在盤前候選數超過可掛單容量時影響候選優先序，必須保持正式slot／資金／盤前掛單與盤中不可換股契約，不得以OOS回調threshold。
3. 若capacity-preserving A9的同參數直接選擇效果仍不為正，Binary DL策略線停止；若轉正但總績效仍差，再做exact-accounting資金路徑歸因，不直接進新Label／新架構。

## 2026-08-07 — A9 Resource-aware Binary：盤前資源瓶頸介入實作

### 狀態

`IMPLEMENTED / RESULT_PENDING / C3_C8_C11_FOCUSED_GATE`

### 程式基準

- 使用者ZIP：`test-branch-1_20260807_153958_8da250a.zip`
- ZIP SHA256：`56667bca105a0772973f5a2697007ec048995cafbf45ef94c3508c6de190157d`
- 固定研究基準：`C3 = Min ROOS`
- 既有hard-filter對照：`C8 = Min ROOS: A9-on`
- 新arm：`C11 = Min ROOS: A9 resource-aware`

### 研究動機

C8相對C3的交易數由336增加至374，但平均曝險由92.13%降至78.24%，顯示A9低曝險不能只用「補回被hard filter刪掉的候選」解釋。使用者要求Binary DL先維持與Min ROOS相同的盤前資訊哲學：只使用當下持股、free slots、可用現金、正式cash-capped sizing、原Min ROOS順序與既有A9 Binary判定，不預測成交率、持有期或未來曝險，也不新增距限價bucket、資金利用率百分比或加權係數。

### Resource-aware正式契約

1. A9不再作setup eligibility hard reject；normal／continuation／合法Re-entry候選生命週期保留，沿用原始breakout日canonical A9 score payload。
2. 候選建立與初始順序保持Min ROOS原position-aware buy-sort；`resource-aware-binary`不得在candidate construction階段先按Score重排。
3. 每日盤前在真正reserve前，以正式`build_cash_capped_entry_plan()`與exact accounting模擬Min ROOS順序；1% risk sizing是最大部位，不是最低部位，剩餘現金不足時允許正式縮單。
4. 若Min ROOS模擬會先用滿free slots，或沒有仍可競爭的未選候選，判定為`capital-utilization`：DL完全不介入。
5. 只有Min ROOS在free slots尚未用滿時就因現金／正式最低下單契約無法再建立任何剩餘候選單，才判定為`dl-selection`；此時cash是盤前binding resource。
6. DL overlay由Min ROOS基準開始，依原Min ROOS順位逐一嘗試提前尚未選中的A9 PASS候選；每次trial都重新走相同cash-capped exact-accounting模擬。
7. trial只有在cash仍為binding resource，且「實際可預留在PASS候選上的資金」嚴格增加時才接受；不再要求總預留資金必須大於等於Min ROOS的精確金額，避免把DL空間鎖死。
8. 同等可行改善依Min ROOS原順位決定；找不到改善即回退Min ROOS。不得使用Future Target、當日尚未完成OHLCV或任何新增數值Threshold。

### 新盤前診斷

正式daily-capacity新增Resource-aware mode、baseline／selected掛單數與PASS數、總預留資金、PASS預留資金及promotion狀態。策略比較報表新增：DL選股日、資金利用優先日、實際改單日、新增PASS單、PASS預留資金增量與總預留資金增量。

### 目前比較設定

只啟用C3、C8、C11，避免重跑已結案的C1～C10其他arms：

| Arm | Runtime | 目的 |
|---|---|---|
| C3 | Min ROOS | 原資金利用基準 |
| C8 | Min ROOS: A9-on | 既有A9 hard-filter失敗對照 |
| C11 | Min ROOS: A9 resource-aware | 驗證只在cash先成瓶頸時介入是否能保住資金利用並改善選股 |

啟用contrasts固定為`C8-C3`、`C11-C3`、`C11-C8`。

### Dataset／Label／模型重建需求

- Dataset：不重建。
- Label：不重建。
- A9模型權重：不重訓。
- A9 threshold：不調整，沿用既有正式threshold。
- forward-OOS scores：沿用既有A9 canonical runtime工件；缺少或過期才由正式共用服務補匯出。
- Min ROOS active params：沿用既有P2工件，不重新optimizer。

### 結果採用規則

本節只有runtime實作，尚無C11正式績效。先看C11相對C3是否恢復接近Min ROOS的曝險與RoMD，再看同參數DL選擇R、EV及年度穩定性；不得依C11結果回頭新增資金利用Threshold或調A9 threshold。

### 下一步

`python apps/strategy_compare.py` → `[2] 查看設定、工件與預計動作` → `[1/Enter] 執行目前比較設定`，取得C3／C8／C11固定forward-OOS結果。

## 2026-08-07 — C11 Resource-aware正式結果與C12 best-improvement basket實作

### 狀態

`C11_ACCEPTED_AS_DL_RUNTIME_RESEARCH_BASELINE / LOW_EXPOSURE_PROBLEM_RESOLVED / A9_SELECTION_SIGNAL_POSITIVE / C12_IMPLEMENTED_RESULT_PENDING`

### C11正式結果基準

- 使用者正式輸出期間：`2021-01-01～2026-03-02`
- Dataset：`full`
- Param policy：`base-finalist-best`
- Max positions：10
- Rotation：off
- Config fingerprint：`bdf346da83e7`
- 固定模型：A9 `breakout_quality_v1 / inception_time_v1 / unique_group_sampling`；不重訓、不調threshold。
- 固定策略參數：`C3 = Min ROOS`，同一P2 rolling active params；C11只改盤前resource-aware runtime。

### C11主要結果

| Arm | 報酬 | MDD | RoMD | 年化 | EV | 曝險 | 交易 | 同參數DL選擇R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 Min ROOS | 166.69% | 15.41% | 10.82 | 20.95% | 0.73R | 92.13% | 336 | 0.00R |
| C8 Min ROOS: A9-on hard filter | 142.70% | 21.75% | 6.56 | 18.76% | 0.64R | 78.24% | 374 | -6.12R |
| C11 Min ROOS: A9 resource-aware | 184.69% | 15.67% | 11.79 | 22.49% | 0.87R | 92.26% | 373 | +79.95R |

- `C11−C3 = +18.01pp報酬 / +0.26pp MDD / +0.97 RoMD / +0.14R EV / +0.13pp曝險 / +79.95R同參數DL選擇R`。
- C11把C8的`-13.90pp`曝險缺口完全消除；平均曝險由C3 `92.13%`微升至`92.26%`，證明A9先前低資金運用的主要問題是hard-filter部署，而不是必然需要預測成交率或持有期。
- Resource-aware盤前診斷：`dl-selection=131日`、`capital-utilization=563日`、`實際改單=92日`、`新增PASS單=111`、`PASS預留資金增量=16,324,574`、`總預留資金增量=-44,794`。A9幾乎不改變總盤前資金配置，就把更多資金重新分配到PASS候選。
- 年度C11為`2021 18.80% / 2022 5.00% / 2023 83.87% / 2024 18.37% / 2025 -3.73% / 2026 8.92%`；2021與2025低於C3，最差完整年度由C3 `0.84%`變為`-3.73%`，因此C11仍是研究基準而非正式部署晉升。
- 只讀集中度診斷：排除2023後逐年鏈結約為C3 `47.98%`、C11 `54.83%`，C11仍約領先`6.85pp`；改善並非只由2023單一年份造成。

### C12唯一變更

C11在`dl-selection` mode內依Min ROOS原順位掃描PASS候選，遇到第一個能提高PASS reserved capital的promotion就立即接受，因此結果可能受first-improvement路徑影響。C12新增`resource-aware-binary-basket` runtime，但保持以下全部不變：

- Min ROOS cash-binding Gate不變；slot先binding時DL完全不介入。
- A9模型、threshold、score日期語意、normal／continuation／Re-entry lifecycle不變。
- exact accounting、cash-capped sizing、盤前鎖定、max positions、rotation與Min ROOS初始buy-sort不變。
- 不新增距限價Threshold、利用率百分比、DL權重、Future Target、fill prediction或holding prediction。

C12只把`dl-selection`內的promotion選擇由first-improvement改為best-improvement：每一輪對所有尚未promotion的A9 PASS候選各自做完整exact cash-cap trial，只保留仍為cash-binding且使`PASS reserved capital`增加，或在PASS reserved相同時使PASS數增加的trial；當輪選擇PASS reserved最高、其次PASS數較多、再依Min ROOS順位決勝的trial，接受後重新評估下一輪，直到沒有改善。

精確窮舉所有PASS子集合雖可作理論global search，但候選數與free slots增加時呈指數複雜度，與專案「架構調整不得明顯犧牲效率」原則衝突，因此不納入正式runtime。C12是零新增Threshold、每輪全候選best-improvement的可執行改進，不宣稱數學全域最優。

### 新比較設定

只啟用：

| Arm | Runtime | 目的 |
|---|---|---|
| C3 | Min ROOS | 同參數資金利用基準 |
| C11 | A9 resource-aware first-improvement | 已證明有效的新DL runtime基準 |
| C12 | A9 resource-aware best-improvement basket | 驗證去除first-candidate順序偏誤是否仍有額外價值 |

啟用contrasts固定為`C11-C3`、`C12-C3`、`C12-C11`；C8 hard-filter停用，不需重跑已結案失敗對照。

### Dataset／Label／模型重建需求

Dataset不重建、Label不重建、A9不重訓、threshold不調整、forward-OOS scores重用、Min ROOS P2重用。C12只是策略runtime排序變更。

### 採用判定與下一步

C11結果已接受為新的DL runtime研究基準；C12尚未取得正式forward-OOS結果，只標記`IMPLEMENTED`。下一步由`python apps/strategy_compare.py`進入選單，先`[2] 查看設定、工件與預計動作`確認只含C3／C11／C12，再`[1/Enter] 執行目前比較設定`。C12只有在相對C11維持接近相同曝險、且RoMD／EV／同參數DL選擇R至少一項有實質增量且年度穩定性未明顯惡化時才保留；否則維持較簡單的C11。

## 2026-08-07 — C12正式結果、最大化PASS研究方向與正式PASS Quality Audit框架

### 狀態

`C12_RESULT_AVAILABLE / MAX_PASS_USAGE_DIRECTION_RETAINED / PASS_QUALITY_AUDIT_IMPLEMENTED_RESULT_PENDING`

### C12正式結果

正式期間`2021-01-01～2026-03-02`、Dataset=`full`、Param policy=`base-finalist-best`、Max positions=10、Rotation=off；A9模型、threshold、Min ROOS P2與forward-OOS scores均不變。

| Arm | 報酬 | MDD | RoMD | 年化 | EV | Payoff | 曝險 | 交易 | 同參數DL選擇R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 Min ROOS | 166.69% | 15.41% | 10.82 | 20.95% | 0.73R | 3.62 | 92.13% | 336 | 0.00R |
| C11 A9 resource-aware first-improvement | 184.69% | 15.67% | 11.79 | 22.49% | 0.87R | 3.26 | 92.26% | 373 | +79.95R |
| C12 A9 resource-aware best-improvement | 172.59% | 16.47% | 10.48 | 21.46% | 0.88R | 3.25 | 92.49% | 388 | +94.59R |

`C12−C11 = -12.11pp報酬 / +0.80pp MDD / -1.31 RoMD / -1.03pp年化 / 約0.00R EV / +0.22pp曝險 / +15交易 / +14.64R同參數DL選擇R`。

Resource-aware盤前診斷：C11為`DL選股131日 / 資金利用優先563日 / 實際改單92日 / 新增PASS 111 / PASS預留資金+16,324,574 / 總預留-44,794`；C12為`146 / 601 / 119 / 143 / +22,169,986 / +493,750`。C12確實提高PASS使用與同參數DL選擇R，但策略報酬／RoMD未同步改善。

### 使用者研究決策

不把C12較差的策略績效解讀成「應降低PASS使用」；後續固定研究原則為：**在既有資源契約下最大化PASS使用，改善PASS本身的品質**。不得新增Min ROOS sorting與PASS sorting的混合權重、比例或依OOS結果調整的折衷參數，避免多出一層selector比例最佳化問題。

### Candidate validity／DL quality／allocation單一責任契約

1. Strategy唯一擁有candidate validity：建立、normal／continuation／Re-entry、expiry／invalidation全部沿用原策略SSOT。
2. DL只擁有quality：PASS／REJECT描述當下或既定snapshot的品質，不得使仍屬策略VALID的candidate失效。
3. Portfolio selector只擁有allocation：只在策略已判VALID的candidate pool內依盤前cash／slots／正式cash-capped sizing分配資源。
4. 因此未來若做candidate-day DL，語意只能是每日更新quality；不得建立第二套DL candidate expiry。Future Label／MFE／MAE／Realized R只能事後Audit，不得回流當日runtime。

### 正式Audit framework

本輪以使用者ZIP`test-branch-1_20260807_194217_8fbe28b(1).zip`為基準，SHA256=`d47b5209b477b3c739479c5a0851ea9a81d3d70bf7b9241016a52df4de32c34a`，實作config-driven正式Audit框架：

- 新增`config/audit.py`，各模組Audit對象、source、dimensions、outcomes與output policy集中管理；正式App不得硬編碼C11／C12等研究名稱。
- `apps/breakout_quality.py`正式主選單新增`Audit／診斷`，子選單固定為執行目前設定、查看設定／工件／預計動作、查看最近結果。
- 第一個正式Audit為`a9_pass_quality`，預設只讀最新strategy-compare的C12 arm，分析A9 PASS內部的raw score、candidate age、candidate type、既有Label／MFE／MAE與實際selected trades Realized R。
- Audit只讀既有正式工件；缺件顯示BLOCKED，不重跑策略、不建立Label、不train模型、不改runtime。
- Score／age分組以config指定quantile groups產生，只供read-only診斷；不得直接變成runtime threshold、age cutoff、Min ROOS／DL混合比例或其他OOS調參。
- 輸出固定於`outputs/audit/<module>/<audit>/runs/<timestamp>/`並維護`latest/`，包含Markdown、JSON、PASS candidate明細與各分組CSV。

### Dataset／Label／模型重建需求

- Dataset：不重建。
- Label：不重建。
- A9模型／threshold：不修改、不重訓。
- Strategy replay：Audit不重跑；只讀既有正式strategy-compare工件。
- 本節Audit僅為`IMPLEMENTED`，尚未取得正式真實資料Audit結果，不得預判score／age／candidate type哪一項是PASS品質主因。

### 下一步

由`python apps/breakout_quality.py`進入`[2] Audit／診斷`，先`[2] 查看 Audit 設定、工件與預計動作`，確認`a9_pass_quality`來源READY，再`[1/Enter] 執行目前 Audit 設定`。依正式Audit結果判斷PASS品質改善方向；在結果前不新增candidate-day失效規則、不新增score threshold或排序混合權重。

## 2026-08-07 — Audit framework formal double-check例外契約修正

### 狀態

`AUDIT_FRAMEWORK_FORMAL_BLOCKER_FIXED / RESULT_PENDING`

### Formal bundle根因

使用者在`test-branch-1_20260807_205701_5d16129.zip`執行正式suite後，quick gate／chain checks／ml smoke通過；consistency唯一FAIL為`META_SPECIFIC_PASS_ONLY_EXCEPTION_TRACEABILITY_CONTRACT`，指出`tools/audit/breakout_quality/pass_quality.py`的`_json_native()`使用`except (TypeError, ValueError): pass`。meta quality的`coverage_synthetic_suite_runs_successfully`亦僅因同一synthetic FAIL連帶失敗；coverage本身不是不足。

### 唯一修正

`_json_native()`在`pd.isna(value)`無法判定特殊物件時，明確`return value`，保留原本「無法判空值就交給後續序列化」的control-flow語意，不再使用pass-only specific exception handler。Audit資料來源、PASS判定、candidate validity／DL quality／allocation契約、分組、Label／Realized R口徑、strategy runtime與模型全部不變。

### Dataset／Label／模型重建需求

- Dataset：不重建。
- Label：不重建。
- A9模型／threshold：不修改、不重訓。
- Strategy replay：不重跑。
- Audit真實資料結果：仍為`RESULT_PENDING`。

### 下一步

套用修正後以本地正式`apps/test_suite.py`做double check；預期consistency的specific pass-only exception synthetic與meta quality的coverage synthetic-run gate恢復PASS。之後才執行`[2] Audit／診斷`取得A9 PASS Quality正式結果。


## 2026-08-07 — A9 PASS Quality正式Audit結果：Score整體單調性弱、Age cutoff不成立、下一步檢查PASS persistence amplification

### 狀態

`RESULT_AVAILABLE / SCORE_GLOBAL_RANK_NOT_SUPPORTED / AGE_CUTOFF_NOT_SUPPORTED / PASS_PERSISTENCE_AUDIT_NEXT`

### 結果來源與固定條件

- 使用者以正式`[2] Audit／診斷`執行`a9_pass_quality`；來源arm=`C12`、DL=`A9`、runtime=`resource-aware-binary-basket`、threshold=`0.5`。
- Audit只讀既有strategy-compare、A9 PASS candidates、Dataset event Label與已發生selected trades；沒有重播策略、建立Label、重訓模型、改threshold或改candidate lifecycle。
- Strategy validity／DL quality／Selector allocation三層契約維持不變；本節所有Label／MFE／MAE／Realized R只供事後研究診斷。

### PASS總覽

- Orderable candidates：`102,859`
- A9 PASS candidate-days：`51,885`，PASS share=`50.44%`
- Selected PASS：`223`；Selected PASS mean Realized R=`0.89R`
- 原Event Label coverage=`98.17%`；candidate-day weighted原Event Label PASS rate=`51.73%`

### Score診斷

| Score quantile | PASS candidate-days | Selected | 原Event Label PASS | Selected mean Realized R |
|---|---:|---:|---:|---:|
| Q1 0.500～0.518 | 10,830 | 75 | 47.77% | 0.57R |
| Q2 0.518～0.533 | 10,272 | 46 | 50.60% | 0.52R |
| Q3 0.534～0.551 | 10,208 | 31 | 56.09% | 0.69R |
| Q4 0.551～0.576 | 10,373 | 41 | 46.27% | 1.03R |
| Q5 0.577～0.703 | 10,202 | 30 | 58.27% | 2.28R |

- `Score ↔ 原Event Label Spearman = 0.049`：A9 raw score在全部PASS candidate-days內沒有足夠整體單調排序力；Q4 Label rate回落亦違反簡單單調排序假設。
- `Score ↔ selected Realized R Spearman = 0.096`：只有弱正相關；Q5 mean Realized R雖明顯較高，但只有30筆selected，且selected trades存在portfolio selection bias，不能據此直接把raw score變成全域PASS ranking。
- 因此目前不啟動「C12 + raw score全域排序」；也不依Q5邊界建立新threshold。

### Candidate age／type診斷

| Age quantile | PASS candidate-days | Selected | 原Event Label PASS | MFE | MAE | Selected mean Realized R |
|---|---:|---:|---:|---:|---:|---:|
| A1，平均2.7日 | 12,202 | 84 | 61.35% | 7.16% | 4.77% | 0.26R |
| A2，平均7.6日 | 9,567 | 41 | 55.74% | 6.92% | 5.29% | 0.28R |
| A3，平均14.1日 | 9,593 | 37 | 51.11% | 6.62% | 5.73% | 2.03R |
| A4，平均24.9日 | 10,252 | 39 | 47.43% | 6.19% | 5.90% | -0.35R |
| A5，平均56.2日 | 10,271 | 22 | 41.93% | 5.52% | 6.11% | 4.73R |

- `Age ↔ 原Event Label Spearman = -0.113`，且candidate-day weighted Label rate由A1 `61.35%`單調降至A5 `41.93%`，MFE同步下降、MAE同步上升；這表示較老的可掛單PASS pool含較高比例的原Event false-positive events。
- 但`Age ↔ selected Realized R Spearman = -0.067`，實際selected mean R高度非單調，A5甚至為`4.73R`且只有22筆。因此不能把上述組成差異解讀成「candidate越老就一定越差」，也不得建立age cutoff或DL第二套expiry。
- Candidate type：extended=`47,268 / 51,885 = 91.1%`的PASS candidate-days，原Event Label PASS=`50.81%`、selected mean R=`1.23R`；normal=`4,617`、Label PASS=`61.38%`、selected mean R=`-0.05R`。這同樣不支持直接淘汰extended candidate。

### 目前最重要的新假設

A9是在原signal event評分一次，但同一策略VALID event可在後續多個candidate days重複出現。Audit顯示extended佔PASS candidate-days約91.1%，而candidate age越高時原Event Label PASS composition越低。下一步先驗證是否存在**PASS persistence amplification**：A9 false-positive PASS events是否平均比true-positive PASS events存活更久，因而在可掛單pool中被重複放大。這是candidate-pool weighting問題，不是另一套candidate invalidation規則。

下一個只讀Audit應比較：

1. unique PASS event-level Label precision vs candidate-day weighted Label precision；
2. 原Event Label PASS／REJECT各自的candidate-days per event、max candidate age、extended-day數；
3. false-positive events對全部PASS candidate-days與C12 selected PASS的占比放大倍率；
4. 不使用Future Target改runtime，不建立age threshold；只有確認persistence amplification後，才設計「DL quality每日更新或加入合法current-state資訊」的模型實驗。

### Audit讀檔警告修正

正式執行出現兩個`pandas DtypeWarning`，分別來自orderable candidates中本Audit不使用的mixed-type欄位，以及Dataset events的ticker型別推斷。本輪只把Audit CSV讀取改為讀取所需欄位、ticker／candidate_type明確string dtype與`low_memory=False`；不改任何Audit數值口徑、策略、Dataset、Label、模型或runtime。

## 2026-08-07 — A9 PASS Persistence Audit實作：驗證false-positive candidate-day amplification

### 狀態

`IMPLEMENTED / RESULT_PENDING / NO_RUNTIME_CHANGE`

### 程式基準

- 使用者ZIP：`test-branch-1_20260807_212852_3d5d0bc.zip`
- SHA256：`5c12b8302a318eb15c254864a60fa970d34eb40c6418dc5e54581d2b585469e3`
- 本輪只擴充正式Audit framework；C3／C11／C12策略runtime、A9模型、threshold、Min ROOS參數、candidate lifecycle與正式策略結果全部不變。

### 唯一變更

在既有`config/audit.py`中保留已完成的`a9_pass_quality` profile但預設`enabled=False`，新增並預設啟用`a9_pass_persistence / audit_type=pass_persistence`。正式Audit選單不新增專用項目，仍只執行config中enabled的Audit。Persistence與PASS Quality共用同一正式strategy-compare來源解析、A9 PASS threshold、selected buys、trades與Dataset event Label讀取鏈，避免第二套資料口徑。

Persistence以`ticker / signal_date / high_len`作原breakout event identity，只讀A9已判PASS且策略仍屬orderable的candidate-days，輸出：

1. unique PASS event-level Label precision；
2. candidate-day weighted Label precision與相對event-level precision差；
3. 原Event Label PASS／REJECT各自的candidate-days per event、max candidate age、extended candidate-days及selected Realized R；
4. A9 false-positive（模型PASS但原Event Label REJECT）在unique-event、candidate-day及C12 selected層的share；
5. candidate-day false-positive amplification ratio、selected amplification ratio，以及REJECT／PASS candidate-days per event ratio。

### 固定語意與限制

- Strategy唯一擁有candidate validity；Persistence只量化仍屬策略VALID／orderable pool的重複權重。
- `DL REJECT`仍不等於candidate invalid；Persistence結果不得建立第二套DL expiry。
- 不新增age cutoff、score threshold、Min ROOS／DL混合比例或其他runtime參數。
- 原Event Label／MFE／MAE／Realized R只供事後read-only Audit，不得回流當日runtime。
- selected amplification存在portfolio selection bias，只用來判斷C12 allocation有沒有進一步放大／抑制原Event false positives，不當作所有未選candidate的反事實。

### Dataset／Label／模型重建需求

- Dataset：不重建。
- Label：不重建。
- A9模型：不重訓。
- A9 threshold：不調整。
- Strategy replay：Audit不重跑；只讀既有正式strategy-compare C12工件。

### GPT獨立固定案例

以隔離臨時工件建立4個A9 PASS unique events：3個原Event Label PASS各只出現1個candidate-day，1個原Event Label REJECT連續出現3個candidate-days。Audit得到unique-event Label PASS=`75%`、candidate-day weighted Label PASS=`50%`、false-positive event share=`25%`、candidate-day false-positive share=`50%`、candidate-day amplification=`2.0x`、REJECT／PASS candidate-days per event=`3.0x`；同時`persistence_does_not_define_candidate_expiry=True`。此案例只驗證計算與語意，不是正式研究結果。

### 採用判定與下一步

本輪只標記`IMPLEMENTED`，不得預先判定真實A9是否存在persistence amplification。下一步由`python apps/breakout_quality.py`進入`[2] Audit／診斷`，先`[2] 查看 Audit 設定、工件與預計動作`確認`a9_pass_persistence`為READY，再`[1/Enter] 執行目前 Audit 設定`。只有正式結果顯示false-positive candidate-days／event明顯高於true-positive且candidate-day precision相對unique-event顯著被稀釋，才把candidate-state／daily quality更新列為下一個策略使用方式實驗候選；若未來需要重訓模型，再依Experiment Registry另行配置新的`MR-*`。否則改查其他PASS品質來源，不建立age-based invalidation。

## 2026-08-07 — A9 PASS Persistence正式Audit結果：false-positive persistence amplification成立；下一步改為C13策略使用語意

### 狀態

`RESULT_AVAILABLE / FALSE_PASS_PERSISTENCE_AMPLIFICATION_CONFIRMED / SELECTOR_NOT_PRIMARY_CAUSE / SR-C13_CANDIDATE_DAY_RESCORE_PLANNED`

### 結果來源與固定條件

- 使用者以正式`[2] Audit／診斷`執行`a9_pass_persistence`；來源arm=`SR-C12`、DL=`DL-A9`、runtime=`resource-aware-binary-basket`、threshold=`0.5`。
- Event identity=`ticker / signal_date / high_len`；Audit只讀A9已判PASS且策略仍屬VALID／orderable的candidate-days、原Event Label與selected trades。
- Strategy validity／DL quality／Selector allocation三層契約不變；本Audit不建立expiry、不改candidate lifecycle、不改C12、不重播策略、不重訓模型。

### 正式結果

| 指標 | 結果 |
|---|---:|
| PASS candidate-days | 51,885 |
| Unique PASS events | 4,706 |
| Event Label coverage | 94.35% |
| Candidate-day Label coverage | 96.40% |
| Unique-event Label PASS | 65.27% |
| Candidate-day weighted Label PASS | 53.96% |
| Candidate-day − Event precision | -11.31pp |
| Selected known-label PASS | 56.02% |
| False-positive unique-event share | 34.73% |
| False-positive candidate-day share | 46.04% |
| Candidate-day amplification | 1.33x |
| False-positive selected share | 43.98% |
| Selected amplification | 1.27x |
| True PASS days/event | 9.3 |
| False PASS days/event | 14.9 |
| False / True persistence | 1.60x |

原Event Label PASS events為`2,898`個、`26,990` candidate-days、平均`9.3 days/event`、max age平均`17.0日`、extended days/event=`8.3`、selected=`121`、selected mean Realized R=`1.43R`。原Event Label REJECT但A9判PASS的false-positive events為`1,542`個、`23,028` candidate-days、平均`14.9 days/event`、max age平均`28.5日`、extended days/event=`14.0`、selected=`95`、selected mean Realized R=`0.24R`。

### 判讀

1. Persistence amplification正式成立：false PASS比true PASS平均多存活`5.6` candidate-days，即約`+60%`；false-positive share由unique-event層`34.73%`放大到candidate-day層`46.04%`，使Label precision由`65.27%`稀釋至`53.96%`，下降`11.31pp`。
2. C12不是主要放大來源：selected false-positive share=`43.98%`低於candidate-day pool的`46.04%`，selected precision=`56.02%`亦比candidate-day precision高`2.06pp`。因此Resource-aware basket selector略為抑制而非進一步放大false positives；主要結構問題位於`DL-A9` breakout-event單次quality被後續candidate-days重複沿用。
3. false PASS被選後的mean Realized R=`0.24R`，顯著低於true PASS selected的`1.43R`，差`1.19R`；因此false-positive persistence不只是Label組成差異，亦與實際selected經濟品質一致。
4. 本結果不得解讀為candidate age本身就是失效條件；candidate validity仍完全由原策略SSOT決定。Persistence只證明「一次性event quality在daily candidate pool的權重會失真」。不建立age cutoff或第二套DL expiry。

### 編號／研究層級更正

- 先前草案將下一步稱為`A10 Candidate-State Quality`是錯誤分類，現已取消。
- `MR-10A`早已由2026-07-29的Candidate-conditioned Query使用並永久占用；不得重用或以`A10`作模糊別名。
- 本次要測的是**既有DL-A9在策略中的score refresh timing**，不是模型訓練／architecture／Label升級，因此應進入策略runtime namespace，正式規劃為`SR-C13`。
- `MR-9A`（模型研究實驗）與`DL-A9`（runtime DL source）為不同namespace；完整定義以`doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md`為準。

### 下一個策略runtime實驗：SR-C13 A9 Candidate-day Re-score

`SR-C13`保持`SR-C12`「在既有資源契約下最大化PASS使用」的allocation原則，唯一研究變更是：**同一個策略VALID candidate在每個decision day重新用既有DL-A9模型與最新合法盤前snapshot計算quality，而不是整段lifecycle永久沿用原breakout日score。**

固定設計原則：

1. **模型不升級**：沿用同一`DL-A9 = MR-9A / ARCH-inception_time_v1 / PROFILE-unique_group_sampling`checkpoint、權重與threshold `0.5`；不重新訓練、不新建model experiment ID。
2. **策略validity不變**：normal／continuation／Re-entry／expiry仍由原策略唯一決定；每日A9 REJECT只代表當日quality，不得刪除仍屬策略VALID的candidate。
3. **盤前無前視**：decision day D 的candidate-day re-score只可使用D盤前合法可知資料；若特徵定義依收盤bar，最晚為D-1 close。
4. **allocation不變**：沿用`SR-C12` resource-aware basket／最大化PASS資源契約；唯一差異是PASS／REJECT使用「當日重新計算的DL-A9 quality」而非原event靜態quality。
5. **不新增混合比例**：不加入Min ROOS／DL weight、不加入age cutoff、不調A9 threshold、不依OOS建立score bucket。
6. **這是runtime distribution-shift實驗**：A9原本由breakout-event snapshot訓練，直接套到candidate-day state可能失效；是否能降低persistence amplification正是SR-C13要驗證的策略使用問題。若SR-C13顯示既有A9無法泛化到candidate-day snapshot，之後才另立真正的model-training experiment，並依Registry重新分配新的`MR-*` ID，不能預先占用或重用`MR-10A`。

### Dataset／Label／模型重建需求

- Dataset／Label：本階段不重建、不relabel；SR-C13不進行模型訓練。
- A9模型／threshold：不重訓、不調整。
- Candidate-day inference artifact：需要新增或擴充正式推論工件，使每個策略VALID candidate decision day可取得合法snapshot下的DL-A9 score；它是推論／runtime工件，不是新Dataset Label或新model version。
- Strategy compare：待SR-C13實作後，以`SR-C12`為主要controlled comparator，另保留`SR-C3`作絕對Min ROOS基準；不得把SR-C11／SR-C12混合比例變成新參數。

### 採用判定與下一步

下一步先實作`SR-C13` candidate-day re-score runtime與其必要推論工件，不修改A9 training pipeline。核心問題是：在固定C12最大化PASS資源契約下，**只改score refresh timing**是否能降低candidate-day false-positive persistence amplification，並把更多PASS使用轉化為更好的selection R／EV／RoMD，而不重新出現資金利用率下降。若SR-C13失敗，再根據結果判斷是否需要真正的candidate-state模型訓練；該模型屆時另行取得新的`MR-*` ID。


## 2026-08-07 — AUD-a9-selection-confidence 實作：只驗證DL Selection Mode內原Event confidence排序力

### 狀態

`IMPLEMENTED / RESULT_PENDING / READ_ONLY_AUDIT / NO_RUNTIME_CHANGE`

### 程式基準

- 使用者ZIP：`test-branch-1_20260807_221137_3b983d3(1).zip`
- SHA256：`79ddc403b353c3a6580b3a408fa5d5656517f637338aa2b92a930d59d9c361b8`
- 本輪依`PROJECT_SETTINGS → Experiment Registry → Experiment Log`完成identity reconciliation後實作；不修改`SR-C12`、`DL-A9`、threshold、Min ROOS、candidate lifecycle或任何模型權重。

### 前一個SR-C13規劃的更正

`SR-C13`先前正式規劃為「同一DL-A9對每個candidate-day最新snapshot重新評分」。進一步核對A9 training semantics後，這個方向在實作前取消：A9是breakout-event classifier，而extended candidate-day通常不是breakout線型；直接把非breakout state送回A9會產生輸入distribution與score語意改變。依Registry已使用ID永久保留規則，`SR-C13`標記`CANCELLED_BEFORE_IMPLEMENTATION`且不得重用；未來下一個strategy runtime ID從`SR-C14`開始，但目前不預先分配其identity。

### 本輪唯一研究問題

使用者方向固定為「在資源契約下最大化PASS使用，再提高PASS品質」，不最佳化Min ROOS／DL混合比例。現有PASS Quality Audit顯示全體selected PASS中`Score ↔ Realized R Spearman=0.096`且Q5 mean R較高，但該結果包含沒有真正PASS競爭的日期，不能直接回答confidence能否用於DL Selection Mode排序。

因此新增`AUD-a9-selection-confidence / audit_type=selection_confidence`，來源固定為`SR-C12 / DL-A9`既有正式strategy-compare工件，只取：

1. `Resource_Aware_Mode = dl-selection`；
2. 同一trade date至少有config設定的`minimum_competing_pass_candidates`個A9 PASS（目前config為2；validator只檢查>=2，不把目前值硬編碼成唯一合法值）；
3. A9 confidence固定沿用原breakout event score，不對extended candidate當日線型重新推論。

### 診斷口徑

- Competition candidate-day `Score ↔ 原Event Label Spearman`。
- 同一批曾參與競爭的unique breakout events之`Score ↔ Event Label Spearman`，用來辨識candidate-day persistence weighting是否改變整體關係。
- 實際selected PASS中`Score ↔ Realized R Spearman`；未成交PASS不填補或估計反事實R。
- 同日PASS pairwise Event Label concordance：只比較Label不同的candidate pair，confidence較高者若為Event Label PASS則concordant；score tie按0.5計入。
- 同日已買PASS pairwise Realized R concordance：只比較同日兩筆都有有限Realized R且R不同的pair；confidence較高者R較高則concordant。
- Competition PASS的config-driven score quantile只做read-only分層，輸出候選數、unique event數、selected數、Label PASS rate、selected mean／median R；不得直接轉成runtime threshold。

### 固定限制

- Strategy唯一擁有candidate validity；本Audit不建立DL expiry或age cutoff。
- DL-A9仍只代表原breakout event quality；extended candidate不是新的breakout event，不做candidate-day A9 re-score。
- Realized R只存在於實際成交者，故R相關與pairwise結果存在portfolio selection bias；不得宣稱是所有未選PASS的counterfactual。
- Audit結果若支持confidence priority，才在下一輪依Registry建立新的`SR-C14` controlled arm；不得把`SR-C13`改名重用，也不得加入Min ROOS／DL比例權重。

### Dataset／Label／模型重建需求

- Dataset：不重建。
- Label：不重建。
- DL-A9：不重訓。
- threshold：不調整。
- Strategy replay：Audit不重跑；只讀既有C12 orderable／selected／trades／daily-capacity與Dataset events工件。

### GPT獨立固定案例

以隔離strategy-compare工件建立兩個`dl-selection`日，每日3個A9 PASS，其中低confidence event為Label REJECT、高confidence events為Label PASS；同日各選2個高confidence PASS並給定較高score對較高Realized R。另加入一個`capital-utilization`日作排除案例。Audit得到competition days=2、competition PASS=6、selected PASS=4、Event Label pairwise comparable pairs=4且concordance=100%、selected Realized R pairwise comparable pairs=2且concordance=100%，並確認capital-utilization日不進competition scope、`extended_candidate_is_not_rescored_as_breakout=True`。

### 下一步

本輪只能標記`IMPLEMENTED / RESULT_PENDING`。正式下一步由`python apps/breakout_quality.py → [2] Audit／診斷`執行目前config。只有正式結果在真正DL-selection competition場景顯示A9 confidence對Event Label及／或selected Realized R具有足夠且方向一致的排序證據，才定義`SR-C14 = C12 max-PASS + confidence priority`候選設計；若證據不足，保留C12，不新增confidence sorting。

## 2026-08-07 — AUD-a9-selection-confidence正式結果：A9 confidence不適合作為DL Selection Mode主排序

### 狀態

`RESULT_AVAILABLE / WEAK_RANKING_SIGNAL / CONFIDENCE_PRIORITY_NOT_ADOPTED`

### 正式結果

使用者以正式Audit選單執行`AUD-a9-selection-confidence`，來源=`SR-C12 / DL-A9`、runtime=`resource-aware-binary-basket`、threshold=`0.5`。Competition限定`Resource_Aware_Mode=dl-selection`且同日至少2個A9 PASS，extended沿用原breakout event A9 score，不做candidate-day re-score。

| 指標 | 結果 |
|---|---:|
| DL Selection days | 146 |
| PASS competition days | 142 |
| Competition PASS candidate-days | 6,619 |
| Competition unique events | 2,111 |
| Selected PASS with Realized R | 161 |
| Candidate-day Score ↔ Event Label | 0.071 |
| Unique-event Score ↔ Event Label | 0.102 |
| Selected Score ↔ Realized R | 0.082 |
| Label pair-weighted concordance | 53.66% |
| Label daily mean concordance | 49.18% |
| Realized R pair-weighted concordance | 55.17% |
| Realized R daily mean concordance | 58.49% |
| Realized R comparable pairs / days | 58 / 29 |

Score Q5（`0.574～0.691`）的11筆實際買入mean R=`7.16R`、median=`1.00R`，但Q1～Q5的Event Label PASS rate並非單調，且Q5 selected樣本僅11筆。不得依此建立新的confidence cutoff。

### 判讀與採用

1. A9 confidence在真正多PASS競爭日的整體排序力弱；尤其每日平均Event Label concordance=`49.18%`，不支持把A9 confidence作為PASS主排序。
2. Q5尾端可能存在訊號，但樣本太少且整體不單調；依既有OOS結果設定score threshold會引入新的result-driven tuning，不採用。
3. 因此不建立`SR-C14 = A9 confidence priority`。`SR-C14`仍可依Registry分配給其他新的策略runtime identity，但不得重用`SR-C13`。


## 2026-08-07 — SR-C14實作：Capital-utilization first + Frozen Continuous Rank

### 狀態

`IMPLEMENTED / RESULT_PENDING / STRATEGY_RUNTIME_ONLY / NO_MODEL_RETRAIN`

### 程式基準

- 使用者ZIP：`test-branch-1_20260807_224112_5781bc1.zip`
- SHA256：`718cd8432ec87b24340d971291f2118248656e4308a8cfa43b3e74cbe9e153c7`
- 本輪依`PROJECT_SETTINGS → Experiment Registry → Experiment Log`確認`MR-11G`與既有continuous score歷史後實作。

### 研究假說

歷史continuous ranker不能因舊raw Score Sort策略績效差就直接判定模型本身無效。舊策略把continuous score直接放在全域buy-sort前方，曾明顯降低相對Min ROOS的資金利用／曝險；而目前resource-aware runtime已能先由Min ROOS exact cash-capped replay判定當日真正binding resource。因此本輪只隔離「continuous score在capital-utilization-first使用方式下是否有經濟價值」。

### Canonical identity

- Model research來源：`MR-11G`（歷史status仍為REJECTED；本輪不翻案、不重訓）。
- Runtime score source：新增`DL-CONT11G`，程式alias=`CONT11G`。
- Backing：`ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_pass_magnitude_mse`。
- Score source：`continuous_ranker_oos`，只讀既有MR-11G frozen OOS `continuous_ranker_scores.csv`的`split=oos` rows。
- Strategy arm：`SR-C14 = Min ROOS: Continuous resource-aware`。

### 唯一策略使用變更

`SR-C14`完全不使用`DL-A9`的PASS／REJECT，也沒有binary threshold。每天盤前先以原Min ROOS排序與正式cash-capped sizing建立baseline：

1. 若Min ROOS需要用滿free slots，或資金利用／position slots先成為限制，mode=`capital-utilization`：**完全維持Min ROOS排序，continuous score不介入**。
2. 若Min ROOS在free slots尚未用滿前即到達cash-binding狀態，mode=`dl-selection`：才使用`DL-CONT11G` frozen event-level continuous score作quality ordering。
3. 先嘗試`continuous_score desc`完整排序；若exact cash-capped replay仍維持cash-binding，直接採用。
4. 若完整score order會破壞cash-binding資源契約，回到Min ROOS baseline，按score由高到低嘗試promotion；只接受promotion後仍cash-binding、被提升candidate實際進入selected basket，且selected continuous-score lexicographic quality改善的變更。
5. 缺score candidate不排除，保留Min ROOS fallback。Continuation／re-entry沿用原breakout event score；不把extended當日線型重新送進MR-11G。

### 不變條件

- `PARAM-P2 / Min ROOS`、all-off rules、max positions、cash-cap sizing與全部candidate lifecycle不變。
- 不調MR-11G權重、target、training profile、seed或任何模型超參數。
- 不新增continuous score threshold。
- 不新增Min ROOS／DL blending weight。
- 不建立PASS／REJECT；continuous score只在DL-selection mode提供完整quality ranking。
- Strategy comparison不得自行重訓MR-11G。若model／manifest／report／OOS score缺少或hash／identity不一致，preparation直接`BLOCKED`並導向模型研究入口。

### Research score-source contract

新增`continuous_ranker_oos` read-only contract，要求：

- model／manifest／`continuous_ranker_report.json`／`continuous_ranker_scores.csv`全部存在；
- filter／architecture／profile identity一致；
- objective=`daily_percentile_regression`；training label scope=`pass_only`；
- model與scores hash／size符合manifest／report；
- model information cutoff早於OOS execution start；
- 只載入`split=oos`，`ticker/date`唯一，score為0～1有限值；
- comparison period只能落在frozen OOS score可用期間。

這個contract只允許controlled strategy research replay，不把MR-11G歷史research-only工件升格成正式scanner/filter runtime。

另需明確保留一項deployment風險：MR-11G的training label scope為`pass_only`，但其OOS score工件可對OOS breakout events輸出model score；SR-C14刻意在全部orderable breakout events上評估這個frozen score。這不是既有模型已證明的all-event ranking能力，而是本次controlled strategy deployment hypothesis的一部分。若C14失敗，不得用結果回頭改score threshold或宣稱資本契約失效；需分開判讀「pass-only trained score對all-event extrapolation」與portfolio allocation效果。

### 正式比較矩陣

本輪config聚焦：

- `SR-C3`：Min ROOS。
- `SR-C12`：A9 max-PASS resource-aware basket，保留作目前最大化PASS comparator。
- `SR-C14`：Continuous resource-aware。

Contrasts：`C12-C3`、`C14-C3`、`C14-C12`。

### 診斷

Resource-aware每日診斷除既有mode／改單／reserved-capital外，新增：

- selected continuous score count／sum／mean及相對Min ROOS baseline差；
- continuous新選入單數；
- direct score order仍符合cash-binding的天數。

不得只用score改善判定成功；正式採用仍以報酬、MDD、RoMD、EV、曝險、資金使用與同參數direct-selection R綜合判讀。

### Dataset／Label／模型重建需求

- Dataset：不重建。
- Label／continuous target：不重建。
- MR-11G：不重訓。
- Strategy params：重用既有`PARAM-P2 / Min ROOS`。
- Continuous scores：只重用既有frozen OOS research工件；策略比較沒有auto builder。

### 下一步

本輪只能標記`IMPLEMENTED / RESULT_PENDING`。使用正式`apps/strategy_compare.py`選單先查看工件計畫；若`DL-CONT11G`既有MR-11G OOS工件READY，再執行`SR-C3 / SR-C12 / SR-C14`比較。核心判定是：capital-utilization-first是否能消除舊continuous raw sort的資金利用缺陷，使continuous ranking的模型層排序訊號轉化為更好的portfolio RoMD／EV／direct-selection R；不得依本次結果回頭調continuous score cutoff或混合權重。


## 2026-08-07 — SR-C14 formal double-check：config-driven validator 硬編碼修正

### 狀態

`IMPLEMENTED`（測試契約／文件閉環；SR-C14 runtime、DL-CONT11G、策略參數與比較設定均未改變）。

### Formal bundle

- Project ZIP：`test-branch-1_20260807_234329_7f2698a.zip`
- Project SHA256：`b39875038ed055988530a7aaa27bbf410cc7e502867bfa28b8d8661fc6a6fe5d`
- Formal bundle：`to_chatgpt_bundle_20260807_234520_14c1f84e.zip`
- Bundle SHA256：`07afaa632a12be203f62ca1c1ff08a7768abcc63f3296d0b31f34db5360b7f85`

### Formal 結果與 root cause

使用者本機正式 suite：

- quick gate：PASS
- consistency：FAIL，2 checks
- chain checks：PASS
- ml smoke：PASS
- meta quality：FAIL；唯一 failure=`coverage_synthetic_suite_runs_successfully`

兩個 consistency FAIL 均位於 `STRATEGY_COMPARE_CONFIG_DRIVEN_APP`：

1. `enabled_arms_build_canonical_execution_pairs_before_replay` 的 expected 仍寫死舊 focused matrix `C3/C11/C12`，實際 config 已合法為 `C3/C12/C14`。
2. `individual_dl_arm_and_contrast_switches_are_runtime_effective` 在停用 C12 後仍固定期待 `C3/C11`，實際 current config 正確得到 `C3/C14`。

因此 root cause 是 **validator 把可由使用者調整的 `config/strategy_compare.py` 目前值硬編碼成唯一合法 expected**，違反 `PROJECT_SETTINGS` C8；不是 SR-C14 runtime、continuous score、cash-binding 邏輯或 coverage 百分比失敗。Meta quality 的 coverage synthetic failure 只是上述 2 個 synthetic FAIL 的連帶結果；正式 coverage 本身 Line 79.28% / Branch 61.47% 已高於門檻。

### 修正

`validate_strategy_compare_config_driven_app_contract_case` 改為：

- execution-pair expected 依當前 `settings.enabled_arms`、`param_source`、`rule_policy` 動態推導；
- individual arm switch 從目前 enabled DL arms 中隔離選一個 target，停用其相關 contrasts 後，expected enabled set 由原 settings 動態計算；
- fingerprint 測試動態尋找一個 enabled definition 與一個 disabled definition，不再釘死 `C3`／`C12`；
- CONT11G／C14 的**功能 identity 與受控 runtime semantics**仍可直接驗證，但不得把目前 enabled matrix 當成固定合法答案。

### 獨立驗證

不執行 `apps/test_suite.py`。GPT 以 direct synthetic 執行 `validate_strategy_compare_config_driven_app_contract_case`，目前 config 下 23 checks 全部通過；另以隔離 override 的不同 enabled-arm matrix 再驗證 config-driven 行為，確保測試不依賴目前 `C3/C12/C14` 選擇。

本修正不改 SR-C14 scientific variable；SR-C14 仍維持 `IMPLEMENTED / RESULT_PENDING`。

## 2026-08-08 — SR-C14 runtime blocker：resource-aware-continuous pre-sort dispatch 補登錄

### 狀態

`IMPLEMENTED / RESULT_PENDING`（SR-C14 scientific variable 不變；僅修正候選建立階段 policy dispatch 漏登錄）。

### 使用者實際錯誤

正式 `apps/strategy_compare.py` 在 `score_ranking` replay 的第一個交易日 `2021-01-04`，於 `phase=build_daily_candidates` 發生：

`ValueError: 不支援的 breakout-quality ranking policy: 'resource-aware-continuous'`

### Root cause

`core/buy_sort.py` 的 `SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES` 已包含 `resource-aware-continuous`，且 `core/portfolio_entries.py` 已完成 C14 的盤前 resource-aware continuous selector；但候選建立階段的 `build_breakout_quality_ranking_prefixes()` 與 `sort_candidate_rows()` 只把 `resource-aware-binary`／`resource-aware-binary-basket` 視為「盤前 resource gate 前不得 quality-sort」的特殊 policy。

因此 `resource-aware-continuous` 被誤送進一般 quality-prefix dispatch，最終落入 unsupported-policy exception。這是 runtime dispatch 漏登錄，不是 MR-11G score、工件 identity、cash-binding 或 C14 selector 邏輯錯誤。

### 修正

- 在 `core/buy_sort.py` 建立單一 `RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES` 真理集合，統一包含 Binary、Binary Basket 與 Continuous 三種 resource-aware policy。
- `build_breakout_quality_ranking_prefixes()` 對全部 resource-aware policy 回傳空 prefix，候選建立階段完整保留原 Min ROOS sort semantics。
- `sort_candidate_rows()` 同樣以該集合判定 resource-aware，禁止在 resource bottleneck 尚未決定前預先套 Continuous score。
- `core/portfolio_entries.py` 的 resource-aware policy acceptance 改共用同一集合，避免 supported list 與盤前 selector 再次分叉。
- synthetic contract 新增 `resource_aware_continuous_presort_preserves_min_roos_before_resource_gate`，直接覆蓋本次正式 runtime 漏洞。

### 不變條件

- `DL-CONT11G / MR-11G` model、target、score、threshold semantics均未改。
- `SR-C14` 仍只在 Min ROOS exact cash-capped replay 已判定 cash-binding 後才使用 continuous ranking。
- slot／capital-utilization mode 仍完全保留 Min ROOS。
- 不新增 score threshold、Min ROOS／Continuous blending weight 或 candidate invalidation 規則。

### 下一步

套用本修正後重跑正式 `apps/strategy_compare.py`；SR-C14 仍維持 `RESULT_PENDING`，不得在取得 C3／C12／C14 正式結果前預先判定有效或無效。


## 2026-08-08 — SR-C14正式結果：Capital-utilization first消除macro曝險問題，但CONT11G排序仍不採用

### 狀態

`RESULT_AVAILABLE / CAPITAL_UTILIZATION_HYPOTHESIS_PARTIALLY_CONFIRMED / CONT11G_DEPLOYMENT_NOT_ADOPTED`

### 正式結果基準

- 使用者結果ZIP：`test-branch-1_20260808_002357_b220cbb.zip`；SHA256=`eb872749778055d480fa4b3ba9038feacedcce92b4e941d859443f5e2aa7c3be`。
- 正式比較期間：`2021-01-01～2025-12-22`。結束日受`DL-CONT11G / MR-11G` frozen OOS score coverage限制，因此不得直接把本輪C3／C12數值與先前延伸到2026-03-02的C11／C12絕對值混比；本輪C3／C12／C14三者同期間、同Min ROOS P2 active params，可作受控contrast。
- Dataset=`full`、param policy=`base-finalist-best`、max positions=10、rotation=off、config fingerprint=`902c90b40dc2`。
- C14唯一變更維持既定契約：Min ROOS exact cash-cap先判resource mode；capital-utilization mode完全維持Min ROOS；只有cash-binding的DL Selection Mode才使用frozen `DL-CONT11G` continuous score。A9 PASS／REJECT不用於C14。

### 主要結果

| Arm | 報酬 | MDD | RoMD | 年化 | EV | 曝險 | 交易 | 同參數DL選擇R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SR-C3 Min ROOS | 156.34% | 15.41% | 10.15 | 20.87% | 0.65R | 91.99% | 329 | 0.00R |
| SR-C12 A9 max-PASS resource-aware | 162.59% | 16.47% | 9.87 | 21.46% | 0.82R | 92.38% | 375 | +94.60R |
| SR-C14 Continuous resource-aware | 157.07% | 17.17% | 9.15 | 20.94% | 0.58R | 91.74% | 395 | +14.62R |

- `C14−C3 = +0.73pp報酬 / +1.76pp MDD / -1.00 RoMD / +0.07pp年化 / -0.07R EV / -0.25pp曝險 / +66交易 / +14.62R同參數DL選擇R`。
- `C14−C12 = -5.52pp報酬 / +0.70pp MDD / -0.73 RoMD / -0.52pp年化 / -0.24R EV / -0.64pp曝險 / +20交易 / -79.99R同參數DL選擇R`。
- 年度C14為`2021 18.85% / 2022 -6.85% / 2023 79.66% / 2024 26.96% / 2025 1.80%`；只有2024明顯優於C3，年度方向不穩定。

### Capital-utilization判讀

1. 使用者假說「舊continuous直接排序失敗可能主要因破壞資金利用率」獲得**部分確認**。C14平均曝險`91.74%`只比C3 `91.99%`低`0.25pp`，與舊raw Score Sort的大幅曝險下降不同，證明capital-utilization-first確實幾乎消除了macro exposure問題。
2. 但micro sizing結構仍改變：平均預留`201,347.89→159,732.00`、平均實際投入`194,644.57→153,067.87`，平均初始停損距離`4.44%→6.36%`；C14實際改單116日、continuous新選入147單、selected score sum增加41.748，但總預留資金增量仍為`-1,861,230`。因此resource gate能守住總曝險，不能保證continuous選到的個別position具有與Min ROOS相同的capital geometry。
3. 更重要的是經濟排序本身不足：C14 EV降至`0.58R`、勝率降至`39.24%`、同參數DL選擇R只有`+14.62R`；相較C12的`+94.60R`明顯較弱。故在macro資金利用問題被大幅控制後，現有`MR-11G` score仍未形成足夠的portfolio selection value。

### 模型語意限制

- `MR-11G`歷史training scope=`PASS-only`。SR-C14把其frozen OOS continuous score用於全部orderable breakout events，是刻意的controlled deployment hypothesis，不代表模型原本具有all-event ranking語意。
- 因此本結果只能否決「直接把現有PASS-only MR-11G接到capital-first即可」；**不能否決「真正以all-event continuous quality為training semantics的模型 + capital-utilization first」這個較一般的方向**。
- 歷史11F已證明`strategy_aligned_opportunity_no_time_r_v1`在all-label上與binary label高度可分（AUC約0.99），且11E actual Target↔R約0.4875；若後續建立新的all-event continuous ranker，必須將「全事件排序」明確定義成新的model research semantics，不可把MR-11G改名或重用其identity。

### 採用判定

- `SR-C14 / DL-CONT11G`目前deployment：`NOT_ADOPTED`。
- `SR-C12`仍是「最大化PASS使用」的active research base；`SR-C11`仍保留較佳已知resource-aware經濟結果的歷史地位。
- 不再對MR-11G做runtime sorting微調、score threshold或Min ROOS／continuous混合權重。

### 下一步

若繼續continuous方向，下一步應是**新的model research**，而不是再改SR-C14：建立真正以`all-events`為training scope的continuous breakout-event quality ranker，再用相同capital-utilization-first runtime作受控比較。開始實作前須重新查Registry分配新的`MR-*`／`DL-*`／`SR-C*` identity；本輪不預先占用新ID，也不依本次OOS結果調任何target係數或runtime權重。

## 2026-08-08 — MR-12A / DL-CONT12A / SR-C15：No-time All-event Continuous Ranker + Capital-utilization first

### 狀態

`IMPLEMENTED / ARTIFACT_PENDING / RESULT_PENDING`

### 程式基準

- 使用者ZIP：`test-branch-1_20260808_003228_5f69112.zip`
- SHA256：`056229e0608bdff039a61d14154459ba5ea7f2352e5dbb79755cdee965d73a68`
- 本輪開始前已依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 實驗 identity

- Model research：`MR-12A`
- Architecture：`ARCH-inception_time_v1`
- Training profile：`PROFILE-strategy_aligned_no_time_all_event_mse`
- Continuous Target：`strategy_aligned_opportunity_no_time_r_v1`
- Training label scope：`all_labels`
- Runtime research DL source：`DL-CONT12A`（alias=`CONT12A`）
- Strategy arm：`SR-C15 / C15 Min ROOS: All-event Continuous resource-aware`

### 研究假說與唯一模型變更

SR-C14已正式證明capital-utilization-first可把舊raw continuous score sort的macro曝險問題大幅消除，但`MR-11G`的PASS-only score在all-event deployment仍未勝過C12。MR-12A因此只測一個新的模型語意：

`MR-11G training scope=pass_only → MR-12A training scope=all_labels`

其餘固定：

- Dataset仍為`DATA-breakout_quality_v1`，不重建資料語意。
- Target仍為既有`strategy_aligned_opportunity_no_time_r_v1`，不修改公式、不依C14 OOS結果調Target係數。
- Architecture仍為`inception_time_v1`。
- Objective仍為`daily_percentile_regression`、loss=`mse`、epoch selection=`mean_daily_spearman`。
- Sampling仍為`unique_ticker_date`，同一event不因事件列數取得額外訓練權重。
- Seed沿用config正式Seed；CLI可顯式`--seed 42`作本次重現。
- OOS不參與gradient、epoch selection或Target percentile擬合；OOS percentile與推論仍只在checkpoint寫入後建立。

MR-12A的同日percentile以**全部有效breakout events**的No-time Target共同排名，因此模型正式學習的是all-event continuous quality ordering，而不是先假定Binary PASS後再排Magnitude。

### Strategy runtime：SR-C15

SR-C15完全重用SR-C14已固定的capital-utilization-first契約；沒有新的portfolio tuning：

1. 所有策略VALID candidates先維持Min ROOS原排序。
2. 以正式exact cash-capped sizing判斷binding resource。
3. position／free slots先成瓶頸時，quality model完全不介入。
4. 只有cash在free slots尚未用滿前先成瓶頸時進入DL-selection。
5. DL-selection使用`DL-CONT12A` frozen OOS continuous score高→低排序；若完整score order破壞cash-binding，沿用既有cash-binding constrained promotion fallback。
6. 不使用A9 PASS／REJECT、不新增score threshold、不新增Min ROOS／Continuous blending weight、不改candidate validity或extended lifecycle。
7. continuation／re-entry只沿用原breakout event的continuous score，不把extended當日非breakout線型重新送入模型。

因此`SR-C15 − SR-C14`的模型層唯一差異是`PASS-only trained score → all-event trained score`；`SR-C15 − SR-C12`則比較完整continuous all-event ranking與A9 max-PASS allocation。

### 工件依賴與建立政策

- `strategy_aligned_opportunity_no_time_r_v1` Target：已有且current時REUSE；缺少／stale才由正式`prepare-continuous-target`重建。
- MR-12A model／manifest／report／`continuous_ranker_scores.csv`：屬新模型研究工件，**策略比較不得自動訓練**。
- `DL-CONT12A` score-source contract沿用泛用`continuous_ranker_oos` loader，但現在由experiment profile本身驗證`continuous_target_id / training_objective / training_label_scope`，不再把`pass_only`硬編碼成唯一合法scope；因此既有`DL-CONT11G`與新`DL-CONT12A`仍共用同一工件驗證單一真理。
- Strategy params：重用`PARAM-P2 / Min ROOS`。

### 正式比較矩陣

目前`config/strategy_compare.py`聚焦：

- `SR-C3`：Min ROOS
- `SR-C12`：A9 max-PASS resource-aware basket
- `SR-C15`：MR-12A all-event continuous resource-aware

Contrasts：`C12-C3`、`C15-C3`、`C15-C12`。`SR-C14`保留歷史結果但本輪disabled，避免重跑已完成的PASS-only deployment。

### 執行順序

MR-12A為模型研究CLI-only，先建立／確認Target，再訓練：

`python apps/breakout_quality.py prepare-continuous-target --filter-id breakout_quality_v1 --target-id strategy_aligned_opportunity_no_time_r_v1`

`python apps/breakout_quality.py train-continuous-ranker --filter-id breakout_quality_v1 --model-architecture inception_time_v1 --experiment-profile strategy_aligned_no_time_all_event_mse --seed 42`

完成後回到正式`apps/strategy_compare.py`選單；狀態頁必須顯示`DL-CONT12A`四個research工件READY，才可執行C3/C12/C15 replay。

### 採用判定

本輪尚未訓練／尚未取得OOS與策略結果，只能標記`IMPLEMENTED / RESULT_PENDING`。不得預先宣稱MR-12A或SR-C15有效；後續先看MR-12A OOS all-event daily/global Spearman、pair concordance與actual Round-trip R，再看SR-C15相對C3/C12的RoMD、EV、曝險與同參數DL選擇R。不得依結果回頭新增score cutoff或blending weight。

## 2026-08-08 — MR-12A formal synthetic isolation blocker修正

### 狀態

`MR-12A / DL-CONT12A / SR-C15 scientific semantics unchanged`；本輪只修正式synthetic validator污染。

### 正式bundle

- 使用者程式ZIP：`test-branch-1_20260808_010913_f46cb91.zip`；SHA256=`0b1b6d4bc2d5947247d954ed78197aeb6da057e47fe6c0b2eda6a3648657eeb2`。
- Formal bundle：`to_chatgpt_bundle_20260808_011057_1aa1a900.zip`；SHA256=`c1890dea909fed18c4b2c7bd24408c105260c389d9cc12f996a8097a2c5d29da`。
- consistency：2 FAIL；meta quality唯一FAIL=`coverage_synthetic_suite_runs_successfully`，由相同2個synthetic FAIL連帶造成。Coverage本身Line=`79.32%`、Branch=`61.54%`，均高於正式門檻。

### Root cause

MR-12A新增`STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE`時，`tools/validate/synthetic_breakout_quality_cases.py`有7處把該常數誤插在既有`STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE`之後。其中5處位於`patch.object(...)`，成為第4個positional `spec` argument；另外2處污染expected tuple。因`patch.object`實際仍只把workflow profile設為PASS-only，兩個expected卻多期待all-event profile，正式suite因此得到2個FAIL：

- `strategy_filter_gate_keeps_continuous_ranker_identity_when_main_workflow_is_binary`
- `strategy_adaptation_uses_isolated_continuous_workflow_without_mutating_config`

這是synthetic test patch污染；MR-12A profile、CONT12A artifact contract、C15 runtime及strategy config本身沒有變更。

### 修正

- 移除全部7處相鄰誤插的all-event profile常數。
- 既有PASS-only isolated workflow tests恢復單一`strategy_aligned_no_time_pass_magnitude_mse` override；MR-12A all-event測試仍由專用`validate_breakout_quality_all_event_no_time_ranker_contract_case`獨立覆蓋。
- 獨立AST檢查確認該synthetic檔內`patch.object`不存在超過3個positional arguments。
- 直接regression確認strategy filter gate預設identity仍為PASS-only continuous profile，strategy adaptation隔離override仍解析`score-ranking / selection_point_in_time`且離開context後不修改正式config。

### 研究語意不變

`MR-12A`仍為`IMPLEMENTED / ARTIFACT_PENDING`；`DL-CONT12A`仍為`ARTIFACT_PENDING`；`SR-C15`仍為`IMPLEMENTED / RESULT_PENDING`。本輪不訓練模型、不改Target、scope、architecture、resource gate或comparison matrix。



## 2026-08-08 — MR-12A / SR-C15 正式結果：All-event Continuous在相同capital-first runtime下明顯優於PASS-only，但尚不直接promote

### 狀態

`RESULT_AVAILABLE / ALL_EVENT_TRAINING_SUPPORTED / STRATEGY_PROMISING_NOT_PROMOTED`

### 正式結果範圍

- 期間：`2021-01-01～2025-12-22`。
- Dataset=`full`、Param=`PARAM-P2 / Min ROOS`、param policy=`base-finalist-best`、max positions=10、rotation=off。
- 正式比較：`SR-C3 / SR-C12 / SR-C15`；config fingerprint=`4da3217c83bd`。
- `SR-C15`完全沿用`SR-C14`的capital-utilization-first、exact cash-cap、cash-binding與fallback契約；唯一模型差異是`DL-CONT11G / MR-11G pass_only → DL-CONT12A / MR-12A all_labels`。
- 本輪使用既有forward/OOS score；Future Target未進runtime。

### 正式策略結果

| Arm | Return | MDD | RoMD | Annual | EV | Exposure | Trades | Same-param DL selection R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SR-C3` | 156.34% | 15.41% | 10.15 | 20.87% | 0.65R | 91.99% | 329 | 0.00R |
| `SR-C12` | 162.59% | 16.47% | 9.87 | 21.46% | 0.82R | 92.38% | 375 | +94.60R |
| `SR-C15` | **168.69%** | **14.81%** | **11.39** | **22.02%** | 0.64R | 91.71% | 379 | +26.78R |

- `C15−C3 = +12.35pp Return / -0.60pp MDD / +1.24 RoMD / +1.15pp annual / -0.02R EV / -0.28pp exposure / +50 trades / +26.78R same-param DL selection R`。
- `C15−C12 = +6.10pp Return / -1.66pp MDD / +1.51 RoMD / +0.56pp annual / -0.19R EV / -0.67pp exposure / +4 trades / -67.82R same-param DL selection R`。
- C15 Log R²=`0.9039`、月勝率=`71.67%`，均優於C3；勝率=`42.48%`、Payoff=`3.21`、EV=`0.64R`則未改善。

### All-event training的乾淨受控證據：C15 vs C14

SR-C14與SR-C15使用同一期間、同一Min ROOS參數與同一capital-utilization-first runtime；核心差異是PASS-only trained score改成all-event trained score。歷史SR-C14為Return=`157.07%`、MDD=`17.17%`、RoMD=`9.15`、EV=`0.58R`、Exposure=`91.74%`、Trades=`395`、same-param selection R=`+14.62R`。因此：

- `C15−C14 = +11.62pp Return / -2.36pp MDD / +2.24 RoMD / +0.06R EV / -0.03pp Exposure / -16 Trades / +12.16R same-param selection R`。
- 曝險幾乎完全相同，故這個改善不能再用macro capital utilization差異解釋；它支持`all_labels` training semantics相對MR-11G PASS-only extrapolation更適合all-event deployment。

### 年度與集中度

C15年度為：`2021 22.70% / 2022 -3.59% / 2023 75.93% / 2024 31.33% / 2025 -1.71%`。

- 相對C3只有2024明顯勝出；2021、2022、2023、2025均略差或明顯較差。
- 以年度報酬鏈結做read-only concentration check，排除2024後：C3約`130.50%`、C12約`122.83%`、C15約`104.56%`；因此C15相對C3/C12的全期報酬優勢高度依賴2024。此計算只用已完成年度結果做歸因，不回流任何runtime或training。
- 相對C14則年度改善較分散：2021 +3.85pp、2022 +3.26pp、2023 -3.73pp、2024 +4.37pp、2025 -3.51pp；5年中3年改善，故「all-event優於PASS-only deployment」的證據比「C15已穩定勝過C3/C12」更強。

### Capital geometry / per-trade品質

- C15平均曝險`91.71%`，與C3 `91.99%`接近；capital-utilization-first仍成功守住macro exposure。
- 平均實際投入由C3 `194,644.57`降至C15 `172,080.46`，平均初始停損距離由`4.44%`升至`5.68%`，但平均投入資金報酬由`2.22%`升至`3.16%`。
- 期末未滿倉日`890→590`、持股缺口總和`2499→872`，顯示C15雖單筆部位較小，卻能更常填滿portfolio slots／降低持股缺口。
- 同時EV只`0.64R`、same-param DL selection R只有`+26.78R`，遠低於C12 `+94.60R`；因此C15的portfolio優勢不能簡化成「每筆交易R更好」，更可能同時包含position geometry、slot utilization、timing／diversification與compounding效果。

### 採用判定

1. **模型研究層**：`MR-12A all-event training semantics`獲得正面支持；在完全相同capital-first runtime下明顯優於`MR-11G pass_only` deployment。
2. **策略層**：`SR-C15`目前標記`PROMISING_NOT_PROMOTED`，不立即取代C3/C12。理由不是全期績效不足，而是年度集中度高，且EV／same-param selection R與portfolio總報酬給出不同訊號。
3. 不依本輪OOS結果新增score threshold、blending weight、年份/regime條件或Target係數。

### 下一步

先做read-only attribution，而不是再訓練或調runtime。下一個診斷應回答：

- C15對C3/C12的超額PnL是否集中在少數月份／少數entry cohorts／少數交易；
- 2024的+20.12pp相對C3改善由哪些已成交selection changes、position sizing與持有重疊造成；
- C15為何在EV與same-param selection R較弱時仍能得到較高portfolio Return與較低MDD；
- 將selection improvement拆成`per-trade R`、`capital return`、`position size / stop distance`、`slot occupancy`與`compounding path`，但全部保持read-only，不新增OOS調參。

取得上述歸因後，再決定是否把SR-C15升格為新的continuous research baseline。

## 2026-08-08 — Project-wide Audit framework收斂＋AUD-c15-strategy-attribution實作

### 狀態

`INFRASTRUCTURE_IMPLEMENTED / AUD-c15-strategy-attribution IMPLEMENTED / FORMAL_RESULT_PENDING`

### 程式基準

- 使用者ZIP：`test-branch-1_20260808_111855_e2b93d1.zip`
- SHA256：`19a571ba739f071214f9bfd649931142738c00a85f9b946764f263ed617c717c`
- 本輪開始前已依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### Audit framework架構變更

使用者確認後續Audit不只服務Breakout Quality，也可能涵蓋其他filter、portfolio、optimizer、data或非filter模組，因此Audit由Breakout-Quality局部工具提升為project-wide subsystem：

- `tools/audit/catalog.py`：所有Audit implementation／domain／mode／read-only／CLI metadata的單一inventory。
- `config/audit.py`：只保存目前active formal Audit policy，不再兼任全Audit inventory。
- `tools/audit/runner.py`：只依config＋catalog派送formal read-only Audit。
- `apps/audit.py`：全專案正式Audit入口；`apps/breakout_quality.py`的Audit選單只是相同backend的Breakout Quality facade。
- 原`tools/filters/breakout_quality/audit_*.py`與`regime_audit.py`已實體搬至`tools/audit/breakout_quality/`；portfolio-generic score-ranking capture移至`tools/audit/portfolio/`。
- 共用console/path renderer由`filters/breakout_quality/console_report.py`提升為`core/console_report.py`，避免project-wide Audit依賴Breakout Quality namespace。
- 不保留舊Audit相容wrapper；原`tools/filters/breakout_quality/strategy_compare.py` legacy alias亦刪除，研究工具直接import canonical `filters.breakout_quality.strategy_compare_engine`。
- 本輪再清除其餘Breakout Quality純相容alias：`tools/filters/breakout_quality/common.py`、`export_scores.py`與`strategy_dl_filter_param_adapt_gate.py`；呼叫端分別直接使用`filters.breakout_quality.workflow_io`、`filters.breakout_quality.export_scores`與`filters.breakout_quality.strategy_param_training`，不留轉接殼。
- 正式strategy comparison共用的trade attribution與判讀style由tools提升到`filters/breakout_quality/trade_attribution.py`及`filters/breakout_quality/strategy_report_style.py`；另新增`filters/breakout_quality/continuous_ranker_data.py`，讓正式strategy diagnostics讀continuous dataset／Target時不需要import training orchestration tool。

Formal runtime依賴方向同步收斂：`core/`與`filters/`不得反向import `tools/audit/`。原score-ranking capture不再由`strategy_compare_engine`在replay/report流程內自動執行；`tools/audit/portfolio/score_ranking_capture.py`只可由既有completed pair工件read-only materialize。需要該歷史研究診斷的Gate／adaptation工具在pair完成或重用後顯式呼叫，不改正式strategy-comparison JSON，也不重跑portfolio。

### AUD-c15-strategy-attribution

Registry identity：`AUD-c15-strategy-attribution`；config ID=`c15-strategy-attribution`。

目的只回答SR-C15既有OOS結果的歸因問題，不建立新strategy arm、不訓練模型、不改selector：

1. `C15 vs C3`及`C15 vs C12`的超額wealth是否集中於少數月份／2024；
2. 哪些trade dates真正改變selected basket；
3. common／candidate-only／comparator-only trades的PnL／R診斷；
4. reserved capital、invested capital、initial stop distance、holding days、capital return等capital geometry；
5. underfilled days、end-position gap slot-days、平均持股與changed-day capacity；
6. 為何C15即使EV與same-param selection R不突出，仍可能得到更好的portfolio Return／MDD。

正式portfolio歸因採每日exact log-wealth差：

`Δ_t = log(1+r_C15,t) - log(1+r_comparator,t)`

並以正式summary total return校正首日純輸出邊界，使`sum(Δ_t)`精確等於：

`log((1 + Return_C15) / (1 + Return_comparator))`

因此monthly／yearly contributions可加總回完整relative wealth path。Trade PnL／R只作mechanism diagnosis，不宣稱其加總等於最終portfolio return，因position sizing、cash timing、持有重疊與compounding均會改變portfolio path。

Trade identity在cross-arm attribution使用`ticker + entry_date + entry_type + signal_date + occurrence`，避免同ticker／同進場日但不同原始breakout event被誤配為common trade。Selection-day identity則使用`ticker + signal_date`。

### 資料與輸出契約

Audit只讀`outputs/strategy_compare/latest/manifest.json`指向的completed正式run，依arm metadata解析C15、C3、C12各自pair與scenario，不重新執行strategy replay。必要來源包含comparison summary、trades、equity、daily capacity及selected buys；缺任一必要工件即`BLOCKED`。

輸出固定於：

`outputs/audit/breakout_quality/c15_strategy_attribution/runs/<timestamp>/`

並更新：

`outputs/audit/breakout_quality/c15_strategy_attribution/latest/`

主工件為`audit.md`與`audit.json`；每個comparator另保存daily/monthly/yearly log wealth、selection days、trade contributions、capacity days及兩臂trade lifecycle CSV。

metadata固定標示：`read_only=true`、`portfolio_replay_executed=false`、`training_performed=false`、`future_target_used_for_runtime=false`。任何結果不得回流runtime threshold、年份/regime gate、Target係數、training scope或selector policy。

### 本輪驗證邊界

本輪沒有執行`apps/test_suite.py`。另以兩層獨立測試驗證實作：

- focused trade-identity case：同ticker／同entry date／同entry type但不同`signal_date`，必須拆成1筆candidate-only＋1筆comparator-only，不得誤配common；
- end-to-end臨時C3／C12／C15 completed strategy-compare fixture：從latest manifest、pair resolver、trades／equity／capacity／selected一路產生`audit.md/json`與16個pair CSV；兩個contrast的`sum(Δ log wealth)`皆在`1e-12`內等於正式total-return導出的relative log wealth，metadata確認read-only且未replay／training。

全專案獨立靜態檢查另確認282個Python檔AST可解析、`compileall`通過、無bare except、無pass-only except handler、無internal import cycle，且`core/`／`filters/`沒有`tools.audit`反向依賴。新`apps/audit.py`已直接執行`--help`與status smoke；本ZIP未包含`outputs/strategy_compare/latest/manifest.json`，因此正式C15 Audit status正確顯示`BLOCKED`。上述synthetic只證明實作契約，**不是SR-C15正式Audit結果**。

因此`AUD-c15-strategy-attribution`目前仍為`IMPLEMENTED / RESULT_PENDING`。下一步由使用者在含正式`outputs/strategy_compare`工件的專案以`python apps/audit.py`進入正式選單執行Audit，再據此決定是否需要capital-preserving basket selector研究；不得在結果前先建立新的C16或回頭修改MR-12A。

## 2026-08-08 — Project-wide Audit migration formal-regression修正

### 狀態

`INFRASTRUCTURE_FIX / AUD-c15-strategy-attribution RESULT_PENDING`。

### 程式基準

- 使用者ZIP：`test-branch-1_20260808_115630_c38367d.zip`
- SHA256：`360e9acfb464b7f8b14ba86a2fe8983605887bd15a04c3de69c885207781d1f7`
- 本輪開始前已依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 本機formal結果與根因

使用者回報quick gate只有`dataset_cli::cli.py::--dataset bad`失敗，consistency與meta quality則因`missing_summary_file`失敗。獨立重現確認`tools/validate/cli.py --dataset bad`尚未進入dataset profile validation，就在lazy載入consistency synthetic時碰到Audit migration後未同步的legacy import：`tools.filters.breakout_quality.common`。同一synthetic檔另殘留已刪除的`export_scores`與`strategy_compare`舊入口，因此合法consistency執行也會在建立summary前中止；meta quality後續載入同一synthetic registry亦受影響。

### 修正

1. `tools/validate/synthetic_breakout_quality_cases.py`三個legacy引用改為canonical正式來源：`filters.breakout_quality.workflow_io`、`filters.breakout_quality.export_scores`、`filters.breakout_quality.strategy_compare_engine`；不恢復任何compatibility wrapper。
2. `tools/validate/cli.py`把`--dataset`值的合法性驗證移至lazy import consistency implementation之前；非法dataset、缺值與空值皆在CLI boundary fail-fast，不再受synthetic/import狀態干擾。
3. `AUD-c15-strategy-attribution`identity、runtime、模型、策略結果與`IMPLEMENTED / RESULT_PENDING`狀態均未改變；本輪沒有產生新的研究結果。

### 獨立驗證

- `--dataset bad`回傳1並輸出`不支援的資料集模式`；`--dataset`缺值與`--dataset=`空值亦分別以預期訊息fail-fast。
- `tools.validate.synthetic_cases`完整import成功；所有本輪已刪除Breakout Quality legacy module的AST import引用=0。
- 全專案282個Python檔AST parse與`compileall`通過；bare except=0、pass-only exception handler=0、internal import cycle=0、`core/filters → tools.audit`=0、legacy wrapper=0。
- `apps/test_suite.py`僅做靜態可信度檢查，未由GPT執行；formal結果仍須由使用者本機重新執行確認。

## 2026-08-08 — Project-wide Audit migration consistency／coverage synthetic閉環修正

### 狀態

`INFRASTRUCTURE_FIX / AUD-c15-strategy-attribution RESULT_PENDING`。

### 程式基準

- 使用者ZIP：`test-branch-1_20260808_121538_c482ac9.zip`
- SHA256：`fc72fc2655a2d29ddd58e85ccbe17848273bb81901b0c8ab4169e8bdde3b0490`
- Formal bundle：`to_chatgpt_bundle_20260808_121631_e2812b00.zip`
- 本輪開始前已依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 使用者本機formal結果

- quick gate：PASS。
- consistency：FAIL 1；唯一失敗為`SYNTHETIC_SUITE`，`FileNotFoundError`仍讀取已搬移的`tools/filters/breakout_quality/audit_pass_realization_gap.py`。
- chain checks：PASS。
- ml smoke：PASS。
- meta quality：FAIL 4，皆為coverage synthetic相關。

`coverage_artifacts/coverage_run_info.json`確認coverage subprocess `returncode=1`，stderr為同一個`audit_pass_realization_gap.py` FileNotFoundError，且`synthetic_case_count=0`。因此當輪line coverage=`22.71%`、branch coverage=`22.98%`只是synthetic suite在執行任何case前中止造成的連鎖結果，不代表正式coverage能力退化；不得調低coverage threshold來消除FAIL。

### 根因

Audit migration後，production imports已改到`tools/audit/`，但`tools/validate/synthetic_breakout_quality_cases.py`仍殘留三類舊契約：

1. 多個contract test直接以舊實體路徑`read_text()` Audit原始碼；第一個撞到的是11H pass-realization-gap，但後面另有11C／11D／11E／11F／11I／11J／11K與PIT相關舊路徑。
2. PIT builder synthetic仍從`tools.filters.breakout_quality.continuous_ranker_pipeline` import已移至`filters.breakout_quality.continuous_ranker_data`的`_validate_group_consistency`。
3. Audit CLI已改為`tools/audit/catalog.py`動態merge至`apps/breakout_quality.py::COMMAND_MODULES`，但多個synthetic仍只AST讀取靜態`COMMAND_MODULES = {...}` literal，因此會把合法Audit command誤判成未註冊。

另發現score-ranking capture compact console正式標題已為`Score Sort 資金配置與 Target Capture 診斷`，舊synthetic仍要求非compact長標題；此為stale output assertion，不應反向修改正式報表。

### 修正

只修改validator與本Experiment Log，不改production strategy／model／config：

- 所有已搬移Audit的source-reading contract改讀`tools/audit/breakout_quality/`正式位置；不恢復任何legacy檔。
- qualified-candidate audit source亦改讀project-wide Audit位置。
- PIT group consistency helper改從canonical `filters.breakout_quality.continuous_ranker_data` import。
- 新增catalog-aware Audit CLI module lookup；11A／11C～11K等Audit command registration檢查直接以`tools.audit.catalog.get_domain_cli_commands("breakout_quality")`為SSOT，不再把動態Audit誤當靜態dict內容。
- score-ranking capture compact console assertion同步目前正式compact title；非compact正式title與report內容均未修改。

### 獨立驗證

未執行`apps/test_suite.py`，亦未直接重跑formal consistency／meta-quality step。

逐一直接驗證本次migration受影響的contract case：continuous target、qualified candidate set、target component attribution、time-penalty ablation、no-time target、pass realization gap、selection strategy realization、candidate counterfactual、portfolio selection pressure、PIT score builder、score-ranking capture與project-wide Audit framework，全部`failed=0`。

全專案獨立靜態檢查另確認：282個Python檔AST可解析、`compileall`通過、bare except=0、pass-only except=0、missing internal import=0、internal import cycle=0、layer violation=0；所有16個Audit catalog modules可import，`core/filters → tools.audit`仍為0，已刪除Audit舊檔名／舊module import在current Python source中為0。`apps/test_suite.py`只做靜態可信度檢查，仍含quick gate／consistency／chain checks／ml smoke／meta quality五段且不讀`PROJECT_SETTINGS.md`；Checklist markdown欄位、T排序、G日期／ID排序亦通過獨立檢查。

交易正式層未修改；另獨立確認盤前reserve在intraday fill判斷前扣除、限價成交使用`min(open, limit)`且low未到limit則miss、entry-day stop／TP只建立next-open pending action、sell cash扣除fee與tax、extended正式限價仍取`orig_limit`。

### 研究狀態

`MR-12A`、`DL-CONT12A`、`SR-C15`與`AUD-c15-strategy-attribution`的identity／策略結果／研究判定均不變；`AUD-c15-strategy-attribution`仍為`IMPLEMENTED / RESULT_PENDING`。本輪只修formal validation migration regression，不新增C16、不訓練模型、不修改Target或runtime selector。

## 2026-08-08 — Formal synthetic metadata／Workbench documentation同步修正

### 狀態

`INFRASTRUCTURE_FIX / AUD-c15-strategy-attribution RESULT_PENDING`。

### 程式基準

- 使用者ZIP：`test-branch-1_20260808_122638_d04e5a1.zip`
- SHA256：`7cdc074ea0f577d5ec8f71450a87cb9f45675fe55816775056bf3bc0db0d4507`
- Formal bundle：`to_chatgpt_bundle_20260808_122824_661b56ee.zip`
- 本輪開始前已依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 使用者本機formal結果

- quick gate：PASS。
- consistency：FAIL 2；真實股票FAIL=0，兩個FAIL皆為synthetic/meta contract。
- chain checks：PASS。
- ml smoke：PASS。
- meta quality：只剩`coverage_synthetic_suite_runs_successfully` FAIL；line=`79.32%`、branch=`61.44%`與key targets皆已通過。

`coverage_run_info.json`顯示coverage synthetic已完整執行247個case，但`synthetic_fail_count=2`，與consistency兩個meta FAIL完全一致。因此本輪不修改coverage threshold或production runtime。

### 根因與修正

1. `META_GUI_WORKBENCH_DOCUMENTATION_SYNC`：Workbench實作與`doc/CMD.md`均已明確使用`交易明細`、`Console`兩個獨立tab；`doc/ARCHITECTURE.md`的`tools/workbench_ui`段亦已有相同描述，但`正式入口`的`apps/workbench.py`列未同步，造成跨文件契約FAIL。修正為在正式入口列明確補上`K 線檢視中的交易明細與 Console 改以獨立分頁承接`；不放寬validator。
2. `META_SYNTHETIC_REGISTRY_METADATA`：`validate_breakout_quality_binary_dl_param_adaptation_contract_case`的`impacted_modules`將canonical `filters/breakout_quality/strategy_param_training.py`重複登記兩次。只移除重複metadata entry，不改validator與production code。

### 研究狀態

`MR-12A`、`DL-CONT12A`、`SR-C15`與`AUD-c15-strategy-attribution`的identity、策略結果與研究判定均不變；`AUD-c15-strategy-attribution`仍為`IMPLEMENTED / RESULT_PENDING`。本輪只修formal synthetic metadata與文件同步問題。

## 2026-08-08 — AUD-c15-strategy-attribution正式結果：C15全期優勢由2024 over-compensate非2024劣勢，機制以portfolio geometry為主

### 狀態

`AUD-c15-strategy-attribution RESULT_AVAILABLE / SR-C15 PROMISING_NOT_PROMOTED`

### 程式基準與正式來源

- 使用者ZIP：`test-branch-1_20260808_123516_f128346.zip`
- SHA256：`13904dc779a52ccd259df1614a716d1198f0725dcf67c00d99f8a6134de31d03`
- Audit正式讀取：`outputs/strategy_compare/runs/20260808_011957_C3-C12-C15_4da3217c83bd`
- 比較期間：`2021-01-01～2025-12-22`
- Audit契約：只讀既有strategy replay；未重跑portfolio、未修改score／selector／training，Future Target未進runtime。

### C15 vs C3

| 指標 | 結果 |
|---|---:|
| Total Return | `156.34% → 168.69%`（`+12.35pp`） |
| MDD | `15.41% → 14.81%`（`-0.60pp`） |
| RoMD | `10.15 → 11.39`（`+1.24`） |
| EV | `0.65R → 0.64R`（`-0.02R`） |
| Average Exposure | `91.99% → 91.71%`（`-0.28pp`） |
| Final relative wealth advantage | `+4.82%` |
| 2024 share of net Δlog wealth | `353.54%` |
| 2024 relative wealth effect | 約`+18.10%` |
| 非2024 relative wealth effect | 約`-11.25%` |

Selection／trade：選股不同日`273`；C15-only/C3-only選入單=`249/199`；exclusive selection `ΔR=+26.78R`。交易層common trades `ΔPnL=-206,198.31`，all-trade `ΔPnL=+123,465.99`，因此exclusive selection交易層`ΔPnL=+329,664.30`。Trade PnL只作mechanism diagnosis，不等同portfolio wealth差。

Capital geometry：平均投入`194,645 → 172,080`、平均預留`201,348 → 179,205`、平均停損距離`4.44% → 5.68%`、平均Capital Return`2.22% → 3.16%`、平均Realized R`0.65R → 0.64R`。Underfilled end days `890 → 590`、position-gap slot-days `2499 → 872`；selection-changed days平均持股`+1.48`，成交買單總差`+50`且missed-buy差`0`。

### C15 vs C12

| 指標 | 結果 |
|---|---:|
| Total Return | `162.59% → 168.69%`（`+6.10pp`） |
| MDD | `16.47% → 14.81%`（`-1.66pp`） |
| RoMD | `9.87 → 11.39`（`+1.51`） |
| EV | `0.82R → 0.64R`（`-0.19R`） |
| Average Exposure | `92.38% → 91.71%`（`-0.67pp`） |
| Final relative wealth advantage | `+2.32%` |
| 2024 share of net Δlog wealth | `472.28%` |
| 2024 relative wealth effect | 約`+11.46%` |
| 非2024 relative wealth effect | 約`-8.19%` |

Selection／trade：選股不同日`293`；C15-only/C12-only選入單=`260/256`；exclusive selection `ΔR=-67.83R`。common trades `ΔPnL=-73,701.70`，all-trade `ΔPnL=+60,984.48`，因此exclusive selection交易層`ΔPnL=+134,686.18`。C15雖exclusive summed R大幅較弱，exclusive dollar PnL仍較高，證明R總和不能代表cash-capped portfolio的經濟貢獻。

Capital geometry：平均投入`160,998 → 172,080`、平均預留`168,163 → 179,205`、平均停損距離`5.49% → 5.68%`、平均Capital Return`1.98% → 3.16%`、平均Realized R`0.82R → 0.64R`。Underfilled end days `744 → 590`、position-gap slot-days `1249 → 872`；selection-changed days平均持股`+0.26`、成交買單總差`+4`、missed-buy差`-1`。

### 歸因判定

1. **SR-C15不升格正式基準。** 2024對C3/C12的Δlog wealth分別占全期淨差異`353.54% / 472.28%`；換成更直觀的exact log-wealth分解，2024單獨約帶來`+18.10% / +11.46%`相對wealth effect，而其餘年份合計約為`-11.25% / -8.19%`。因此不是「2024只是主要貢獻」，而是**2024超額績效必須抵銷其他年份整體落後後，C15才留下全期優勢**。
2. **相對C3，主要改善不是平均R。** C15平均R略低，但單筆占用資金較少、停損距離較寬、slot occupancy明顯提高，且changed days多成交50筆；全期優勢更符合「風險sizing／資金占用／可同時持有數／fill path／compounding」共同作用。
3. **相對C12，交易數差異已很小，仍不能用平均R或selection R解釋。** C15只多4筆成交，exclusive `ΔR=-67.83R`，但exclusive dollar PnL約`+134.7k`且capital return更高；主要差別進一步落在實際risk-dollar、position sizing、交易時點與持有重疊。
4. 既有`SR-C15 vs SR-C14`仍是判斷`MR-12A all_labels`相對`MR-11G pass_only`最乾淨的受控比較，因兩者runtime與macro exposure幾乎相同。現有AUD-c15只比較C3/C12，不能用來取代C14/C15的model-source attribution。

### Audit輸出補強

同一`AUD-c15-strategy-attribution`升級結果schema至v2，不改任何策略或研究變數：

- 集中度表直接新增focus-year與non-focus的relative wealth effect，避免`>100%` share難以直觀解讀。
- `Selection／trade`直接顯示`Exclusive selection ΔPnL`，並把原誤導性的`All matched trade ΔPnL`更名為`All trade ΔPnL`。
- 新增`SR-C15 selector自身盤前診斷`：直接從既有daily-capacity工件顯示DL-selection／capital-utilization days、selector changed days、相對同日Min ROOS的planned selected-count差、reserved-cash差、score gain與direct-score-order feasibility；仍只讀既有工件、不重跑portfolio。
- Formal synthetic B190/T287同步覆蓋schema v2、non-focus wealth分解與新增selector diagnostics。

### 下一步

不先建立新的SR-C16，也不回頭調MR-12A模型。下一個最高資訊量工作是**同一capital-utilization-first runtime下的SR-C15 vs SR-C14 read-only attribution**，用來把`all_labels` score source的改善與portfolio geometry分離。完成後若確認MR-12A在相同geometry下仍有穩定經濟貢獻，再把「增加DL實際決策日、同時不惡化盤前資源利用」設計為新的selector研究；selector variant應先在pre-2021 Selection/PIT strategy replay中決定，再回到既有2021+迭代研究OOS評估。


## 2026-08-08 — AUD-c15-source-attribution實作：跨run重用C14/C15正式工件隔離all-label vs PASS-only score source

### 狀態

`IMPLEMENTED / RESULT_PENDING`

### 程式基準

- 使用者ZIP：`test-branch-1_20260808_125606_32f0bea.zip`
- SHA256：`b1581a8d30e9c8ed215e077985c678abb0b05f80fc8914ee7d2cf6ec9a9c0f9c`
- 本輪開始前已依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 既有正式證據

`AUD-c15-strategy-attribution`已確認：C15相對C3/C12的全期優勢高度依賴2024，且平均Realized R沒有同步提高；C15自身盤前resource-aware診斷為DL-selection days=114、selector changed days=94、相對Min ROOS預計選入單數差=+24、累計預留資金差約-1.064M、direct score-order feasible days=32。這正式證明目前cash-binding契約可守住macro exposure，但不等同逐日／逐basket資金幾何完全不變。

同時既有C14正式結果已保存於config fingerprint=`902c90b40dc2`，C15正式結果保存於`4da3217c83bd`；兩者期間皆為`2021-01-01～2025-12-22`，均使用`PARAM-P2 / Min ROOS`、`base-finalist-best`、max positions=10、rotation=off與`resource-aware-continuous` runtime。為避免為read-only歸因重跑portfolio，本輪新增跨run正式工件解析。

### 唯一變更

新增`AUD-c15-source-attribution`，仍重用既有`strategy_attribution` handler，不建立新策略arm或模型：

- Candidate：`SR-C15 / DL-CONT12A / MR-12A all_labels`，依正式config fingerprint=`4da3217c83bd`解析已完成run。
- Comparator：`SR-C14 / DL-CONT11G / MR-11G pass_only`，依正式config fingerprint=`902c90b40dc2`解析已完成run。
- 跨run只在comparison period、dataset、param policy、max positions、rotation、param source、rule policy、runtime mode與`param:min_roos`正式SHA全部一致時允許；任一不一致即BLOCKED。
- Audit結果schema升至v3，metadata明確保存每個arm的source run／fingerprint與`cross_run`狀態。
- Selector資源診斷由只顯示C15，擴充成pair-level顯示C14與C15各自相對同日Min ROOS的DL-selection days、selector changed days、planned selected-count delta、reserved-cash delta、promoted score orders與direct score-order feasible days；不同模型score數值本身不直接互比。

### 研究邊界

本輪只新增read-only Audit source與診斷；不重跑C14/C15、不重訓MR-11G/MR-12A、不改Target、score、resource gate、selector或strategy parameters。正式結果取得前`AUD-c15-source-attribution`只能標記`IMPLEMENTED / RESULT_PENDING`。

### 獨立驗證

- direct synthetic已覆蓋同run既有C15/C3/C12 attribution與跨run C15/C14 fingerprint resolver；跨run案例確認`portfolio_replay_executed=false`、exact `sum(Δlog wealth)=target relative wealth`、param SHA相同才READY，且pair-level comparator resource diagnostics存在。
- GPT未執行`apps/test_suite.py`；正式double check仍由使用者本機執行。

### 下一步

套用後直接由`apps/audit.py`正式選單執行目前Audit設定即可；若本機保留上述兩個strategy-compare正式runs，應直接READY並產生C15 vs C14結果，不需要重新執行`apps/strategy_compare.py`。取得結果後再判斷是否建立新的capital-preserving selector研究；本輪不預先分配`SR-C16`。

## 2026-08-08 — AUD-c15-source-attribution正式結果：MR-12A all-label source成立，SR-C15仍不promotion

### 狀態

`AUD-c15-source-attribution RESULT_AVAILABLE / MR-12A SOURCE_SUPPORTED / DL-CONT12A SOURCE_SUPPORTED / SR-C15 PROMISING_NOT_PROMOTED`

### 正式來源

- Candidate run：`outputs/strategy_compare/runs/20260808_011957_C3-C12-C15_4da3217c83bd`
- Comparator run：`outputs/strategy_compare/runs/20260808_001744_C3-C12-C14_902c90b40dc2`
- 比較期間：`2021-01-01～2025-12-22`
- Candidate：`SR-C15 / DL-CONT12A / MR-12A all_labels`
- Comparator：`SR-C14 / DL-CONT11G / MR-11G pass_only`
- 共用契約：`PARAM-P2 / Min ROOS`、all-off rules、max positions=10、rotation=off、`resource-aware-continuous`，跨run已驗證Min ROOS參數SHA一致。
- Audit只讀既有replay；不重跑、不改score／selector／training。

### 正式結果

| 指標 | C14 | C15 | C15 - C14 |
|---|---:|---:|---:|
| Total Return | 157.07% | 168.69% | +11.61pp |
| MDD | 17.17% | 14.81% | -2.36pp |
| RoMD | 9.15 | 11.39 | +2.24 |
| EV | 0.58R | 0.64R | +0.06R |
| Average Exposure | 91.74% | 91.71% | -0.03pp |
| Final relative wealth | - | - | +4.52% |

Wealth concentration：2024單獨relative wealth effect約`+3.44%`，非2024期間仍為`+1.04%`；2024占全期淨Δlog wealth `76.58%`，但不像C15 vs C3/C12需要由2024補回其他年份整體負貢獻。Top正貢獻月份跨2021、2022、2023、2024。

Selection／trade：選股不同日`300`；C15-only/C14-only=`239/255`；exclusive selection `ΔR=+12.16R`、`ΔPnL=+196,527.93`；common trades `ΔPnL=-80,379.56`，all-trade `ΔPnL=+116,148.37`。

Capital geometry：平均實際投入`153,068 → 172,080`、平均預留`159,732 → 179,205`、平均停損距離`6.36% → 5.68%`、平均Capital Return`2.63% → 3.16%`、平均Realized R`0.58R → 0.64R`。Underfilled end days`633 → 590`，position-gap slot-days`955 → 872`。

Selector自身盤前診斷：C14/C15 DL-selection days=`126/114`、selector changed days=`116/94`、相對各自Min ROOS planned selected-count delta=`+27/+24`、reserved capital delta=`-1,861,230/-1,063,551`、promoted score orders=`147/119`、direct score-order feasible days=`35/32`。

### 判定

1. 相同Min ROOS、相同runtime與幾乎相同macro exposure下，MR-12A all-label source在Return／MDD／RoMD／EV／exclusive R／exclusive dollar PnL／Capital Return均同向優於MR-11G PASS-only source，且非2024仍為正，因此`MR-12A / DL-CONT12A`升為`SOURCE_SUPPORTED`。
2. Source支持不等於策略promotion。SR-C15相對C3/C12仍高度依賴2024，故維持`PROMISING_NOT_PROMOTED`。
3. C15相對Min ROOS出現planned selected-count `+24`但reserved capital `-1.064M`，正式證明「仍為cash-binding」不能保證basket-level資金幾何不退化；後續不再調模型，優先研究selector resource floor。

## 2026-08-08 — SR-C16實作：All-event Continuous basket-level capital-preserving selector

### 狀態

`SR-C16 IMPLEMENTED / RESULT_PENDING`

### 程式基準

- 使用者指定ZIP：`test-branch-1_20260808_131423_1149509(1).zip`
- SHA256：`368572aa5f8b85179ebc1379dbfd1ccb5d3827257d3a4b963e15c17c7bf558ae`
- 本輪開始前已依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 唯一研究變更

`SR-C16`固定沿用：

- `PARAM-P2 / Min ROOS`
- rules=`all_off`
- `DL-CONT12A / MR-12A`
- frozen OOS continuous score
- max positions=10
- rotation=off
- 原策略sizing／exact accounting／成交語意

唯一新變數為selector feasibility contract；不修改模型權重、Target、training scope、score cutoff、blend weight、年份／regime gate或strategy params。

新runtime mode：`resource-aware-continuous-capital-preserving`。

每日盤前先用原Min ROOS順序與正式cash-capped exact reservation建立baseline。任何DL重排只有同時滿足下列兩個invariant才可接受：

`selected_count >= baseline_selected_count`

`reserved_cost_milli >= baseline_reserved_cost_milli`

因此C16不再把「cash仍為binding resource」當作可行條件；cash-binding與slot-binding日皆可讓MR-12A score參與，但不得以減少預計選入數或降低盤前reserved capital交換品質分數。所有條件只使用盤前可知資料，未使用隔日成交或Future Target。

### Selector演算法

1. 候選建立與pre-sort完全維持Min ROOS原排序。
2. 先計算Min ROOS exact reservation baseline。
3. 若沒有可競爭的未選候選或沒有有效continuous score，完整回退Min ROOS。
4. 先嘗試純score-descending order；只有雙resource floor成立且selected quality嚴格改善才接受。
5. 若純score order不可行，從Min ROOS開始做deterministic best-improvement score promotions；每輪評估所有尚未promotion的scored candidate，exact replay後只保留雙resource floor成立且quality嚴格改善的trial，再採當輪最佳trial。
6. 不新增任何resource tolerance／threshold；exact milli accounting直接比較，不用近似比例。

### 正式比較矩陣

`config/strategy_compare.py`目前只啟用：

- `C3 = SR-C3 / Min ROOS`
- `C15 = SR-C15 / MR-12A cash-binding resource-aware continuous`
- `C16 = SR-C16 / MR-12A capital-preserving continuous`

只啟用兩個核心contrast：

- `C16-C15`：同一MR-12A source下，只隔離selector resource contract。
- `C16-C3`：檢查新selector相對Min ROOS基準的完整經濟效果。

不再把C12/C14塞入本輪主矩陣。

### 診斷與驗證

Daily capacity新增：`Resource_Aware_Preservation_Required`、`Resource_Aware_Selected_Count_Preserved`、`Resource_Aware_Reserved_Capital_Preserved`。Strategy summary新增planned selected-count delta與resource-preservation violation days；C16正式結果的violation days必須為0，否則視為runtime contract failure，不得解讀績效。

獨立direct synthetic已新增兩個固定案例：

- slot-binding baseline下，高score候選若selected count不降且reserved capital提高，C16允許換股；舊C15仍不介入該類日。
- 高score候選若使reserved capital低於Min ROOS baseline，即使score較高也必須拒絕並保留原順序。

`validate_strategy_compare_config_driven_app_contract_case`直接執行`48/48 PASS`。GPT未執行`apps/test_suite.py`。

### 下一步

使用正式`apps/strategy_compare.py`選單執行目前設定。結果取得前`SR-C16`只能標記`IMPLEMENTED / RESULT_PENDING`；不得依2021+ OOS結果回頭調resource tolerance、score threshold或其他numeric gate。正式報表先驗證resource-preservation violation days=`0`，再比較C16-C15與C16-C3的Return／MDD／RoMD／EV、exposure、selected-count delta、reserved-capital delta與DL selection coverage。


## 2026-08-08 — SR-C16正式結果＋SR-C17 Max-DL constrained basket實作

### SR-C16 RESULT_AVAILABLE

程式／回放工件由使用者本地正式`apps/strategy_compare.py`產生；期間`2021-01-01 ～ 2025-12-22`，固定`PARAM-P2 / Min ROOS`、`DL-CONT12A / MR-12A`、all-off rules、max positions=10、rotation=off。

主要結果：C16 Return=175.15%、MDD=15.86%、RoMD=11.05、Annual Return=22.60%、Log R²=0.8773、月勝率=63.33%、EV=0.48R、Exposure=91.83%。相對C15：Return +6.46pp、MDD +1.04pp、RoMD -0.34、Annual Return +0.59pp、EV -0.15R、Exposure +0.11pp；相對C3：Return +18.81pp、MDD +0.45pp、RoMD +0.90、EV -0.17R。

資源契約：DL-selection days=270、selector changed days=179、相對Min ROOS預計選入單數差=+70、reserved-capital delta=+1,361,933、resource-preservation violation days=0。結論為`RESOURCE_CONTRACT_PASSED / NOT_PROMOTED`：C16證明可大幅提高DL介入率且不降低盤前selected-count／reserved-capital floor，但風險品質與EV相對C15退步，不把最高Return直接視為selector最終解。

### SR-C17 IMPLEMENTED

程式基準：`test-branch-1_20260808_134906_908f3b4(1).zip`，SHA256 `703650d1947d8b619d4e4739ebf35049a852bed1872a7eb7f93fa10cb920c5e0`。

唯一變更為selector；Dataset、Label、MR-12A模型權重、continuous OOS scores、策略參數、sizing、accounting、execution、max positions與rotation均不變，不重建Dataset／Label、不重新訓練模型。

SR-C17每日先以Min ROOS exact reservation建立：
- `K = baseline selected_count`
- `R0 = baseline reserved_cost`

之後basket membership的唯一objective為frozen MR-12A continuous score。先取純DL Top-K；若以正式cash-capped reservation檢查後不滿足`selected_count == K`或`reserved_cost >= R0`，才從原Top-K做deterministic minimum-repair：每一步只替換一個尚未替換的Top-K成員，所有可能single-swap均以正式exact reservation重算；只接受hard-resource deficit嚴格改善的trial，並在可改善trial中保留DL quality最高者，最多K步。若仍無法取得合法basket，回退同日Min ROOS baseline basket。

為避免「DL選誰」與「DL決定誰先吃資金」混為同一研究變數，basket membership由DL score決定，但已選basket內的盤前執行順序固定沿用原Min ROOS rank。Runtime保留完整orderable-candidate universe供候選統計／diagnostics，只有action prefix限制為K筆，避免第K+1候選在正式reservation繼續建立掛單。

時間複雜度：純Top-K排序`O(N log N)`；repair最多K輪，每輪評估至多`K(N-K)`個single swaps，每個trial exact simulation最多K筆，故worst-case約`O(N log N + K^3 N)`；本專案`K<=10`時屬小常數乘線性候選規模。此快速selector不宣稱組合數學全域最優；小N exhaustive oracle只作工程驗證，不回流策略參數。

GPT獨立小型oracle驗證（非正式策略績效）：995個可行random small-N cases中resource violations=0，快速selector與exhaustive global best basket一致990例（99.50%）；12例回退Min ROOS。此數據只驗證approximation行為與硬契約，不作SR-C17績效採用證據。正式策略結果仍為`RESULT_PENDING`。

正式比較設定收斂為`C3 / C16 / C17`，只開`C17-C16`與`C17-C3`。取得forward-OOS結果後可判斷max-DL selector是否值得凍結；不得依該結果新增repair tolerance、blend、score cutoff或capital/DL比例。

## 2026-08-08 — SR-C17正式結果、Exact Max-DL計算時間評估與SR-C18 feasible-ascent

### 狀態

`SR-C17 RESULT_AVAILABLE / SELECTOR_NOT_FROZEN`

`SR-C18 IMPLEMENTED / RESULT_PENDING`

### 程式基準

- 使用者指定ZIP：`test-branch-1_20260808_144639_67acaf9(1).zip`
- SHA256：`45a8c140e8e15fa9164452b4505bc011788b2633888ac36f26882e1ce8847190`
- 本輪開始前已依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。
- Dataset、Label、`MR-12A / DL-CONT12A`模型權重、forward OOS scores、`PARAM-P2 / Min ROOS`、sizing、accounting、execution、max positions=10、rotation=off皆固定；本輪不重訓模型、不修改Target、不使用forward-OOS績效調score／threshold／blend／resource tolerance。

### SR-C17正式forward結果

使用者本地正式`apps/strategy_compare.py`工件：config fingerprint=`7a0888323f3b`，期間`2021-01-01 ～ 2025-12-22`。

C17：Return=164.13%、MDD=14.38%、RoMD=11.42、Annual Return=21.60%、Log R²=0.8865、月勝率=68.33%、EV=0.72R、Exposure=92.04%、trades=328、same-param DL selection R=+20.31R。

相對C3：Return +7.79pp、MDD -1.03pp、RoMD +1.27、Annual Return +0.73pp、EV +0.06R、Exposure +0.05pp、trades -1、same-param DL selection R +20.31R。

相對C16：Return -11.02pp、MDD -1.48pp、RoMD +0.37、Annual Return -1.00pp、EV +0.23R、Exposure +0.21pp、trades -80、same-param DL selection R +37.85R。

年度報酬：2021=19.18%、2022=-2.80%、2023=76.00%、2024=32.68%、2025=-2.36%。本輪不依年度／headline return回頭調selector；C17後續變更理由只來自搜尋完整性與計算時間。

Resource diagnostics：Max-DL eligible=223日、repair=209日、final fallback=26日、K violation=0、resource-preservation violation=0、planned selected-count delta=0、reserved-capital delta=+906,375。Min ROOS baseline本身必為合法K／R0 basket，因此26個fallback表示C17 local minimum-repair沒有找到某些已知存在的合法／更佳路徑；這是selector search completeness問題，而不是模型或resource contract問題。

### Exact global Max-DL搜尋評估：不採用production

本輪先實作工程用exact global search原型，目標與C17完全相同：固定K、R0與basket內Min ROOS execution order，只求hard-feasible basket中`sum(DL score)`全域最大。先測naive DFS Branch-and-Bound，再測以C17合法解作incumbent的best-first K-combination enumeration；兩者都沒有node cap／timeout近似作為正式語意，工程壓力測試只用來判斷可否實際部署。

代表性single-case selector CPU benchmark（K=10）：

- N=12：C17約3.5ms；exact約8.4ms。
- N=14：C17約4.8ms；exact約72ms。
- N=20：C17約16ms；exact約1.33s。
- N=22：C17約17ms；exact約0.67s。
- N=30：C17約22ms；exact超過10s仍未完成。
- N=80：C17約107ms；exact超過10s仍未完成。

真實strategy candidate-day平均候選規模約80，因此global exact的worst／difficult-case成本會直接妨礙後續MR-12B／MR-12C反覆strategy replay。結論：**global exact search只保留工程oracle用途，不建立正式runtime mode、不分配獨立SR ID、不進production。**

### SR-C18：Max-DL feasible-ascent

為消除C17「repair失敗即停止／fallback」而不引入global exact爆炸成本，分配`SR-C18`。C18與C17使用完全相同：

- `PARAM-P2 / Min ROOS`
- `DL-CONT12A / MR-12A`
- `K = Min ROOS baseline selected_count`
- `R0 = Min ROOS baseline reserved capital`
- basket內正式執行順序固定Min ROOS rank
- full orderable universe與K筆action prefix分離

唯一差異是搜尋完整度：先執行C17取得合法seed；即使C17原本fallback Min ROOS，該baseline也只視為合法seed，不是終點。其後每輪枚舉目前K-basket內每個selected與外部每個unselected的single swap；每個trial以canonical exact reservation重算，只接受`selected_count == K`、`reserved_cost >= R0`且DL quality嚴格提高的方案，並選當輪DL quality最高的feasible improvement。重複直到不存在任何improving feasible single swap，因此輸出保證為deterministic **1-swap local optimum**，但不宣稱global optimum。Capital全程只作hard feasibility，不是objective。

新增runtime policy／mode：`resource-aware-continuous-max-dl-feasible-ascent`。正式比較設定改為`C3 / C17 / C18`，只開`C18-C17`與`C18-C3`。

### 計算時間與工程驗證

300組random exact-reservation selector benchmark（N=8～30）：

- eligible cases=300
- C17 seed fallback=51
- C18相對C17 DL score improvement cases=61
- resource violations=0
- DL score regressions=0
- 最終非1-swap local optimum=0
- C17 median=2.484ms；C18 median=3.720ms，median slowdown=1.497×
- C17 total=1,263.35ms；C18 total=1,864.79ms，total slowdown=1.476×

代表性N=80、K=10、50組random cases：

- C17 median=77.35ms、p95=172.63ms、max=215.14ms、total=4,361.50ms
- C18 median=108.26ms、p95=295.70ms、max=336.11ms、total=6,828.64ms
- slowdown：median=1.40×、p95 ratio=1.71×、total=1.57×
- C17 final fallback=17/50；C18 seed fallback=17/50但**final fallback=0**
- C18 ascent days=39/50、ascent steps=105、ascent exact evaluations=76,694

最終封裝前另以獨立seed重跑N=80、K=10、50-case verification：C17 median=38.76ms／p95=79.49ms／total=1,990.35ms；C18 median=56.73ms／p95=155.47ms／total=3,524.29ms；median slowdown=1.46×、total slowdown=1.77×，C17 fallback=14而C18 final fallback=0、resource violation=0。兩批N=80 benchmark的絕對毫秒受候選成本分布與CPU狀態影響，但一致顯示C18約1.4～1.5× median、約1.6～1.8× total selector成本，仍遠低於global exact difficult-case的10秒級。

N=20與N=40代表性50-case benchmark的total slowdown分別約1.26×與1.42×。相較global exact在N=30／80超過10秒，C18維持百毫秒級selector成本，適合後續頻繁DL模型策略驗證。

Strategy Compare summary新增selector CPU timing：calls、total ms、median ms、p95 ms、max ms；並新增C18 seed-fallback、feasible-ascent days／steps／evaluations、1-swap local-optimum days。正式本機replay後以同一run內C17／C18 timing作最終真實成本比較。

### Direct synthetic / 判定

`validate_strategy_compare_config_driven_app_contract_case`新增C18固定案例：

1. C17 minimum-repair得到次佳合法basket時，C18會在相同K／R0下找到更高DL quality的1-swap feasible improvement。
2. C17 seed原本fallback Min ROOS時，C18仍從該合法seed繼續DL ascent，不把fallback當終點。

目前direct synthetic共32項全部PASS。GPT未執行`apps/test_suite.py`。

C18狀態固定`IMPLEMENTED / RESULT_PENDING`。取得正式forward結果前不得把engineering timing、random benchmark或C17 headline績效當成C18採用證據；也不得依forward結果調single-swap規則、score cutoff、resource floor或任何numeric selector參數。

## 2026-08-08 — SR-C18 pair-level結果與多DL共用baseline timing聚合修正

### 狀態

`SR-C18 PAIR_RESULT_OBSERVED / FINAL_REPORT_RERUN_REQUIRED`

### 程式／正式執行證據

- 使用者本地`apps/strategy_compare.py`：config fingerprint=`4621173de22d`，期間`2021-01-01 ～ 2025-12-22`。
- 固定`PARAM-P2 / Min ROOS`、`DL-CONT12A / MR-12A`、max positions=10、rotation=off；本輪不修改selector、score、Target、resource floor或任何numeric gate。
- C17 pair-level重現：Return=164.13%、MDD=14.38%、RoMD=11.42、EV=0.72R、Exposure=92.04%。
- C18 pair-level：Return=140.56%、MDD=14.73%、RoMD=9.54、Annual Return=19.33%、Log R²=0.8810、月勝率=60.00%、EV=0.57R、Exposure=91.98%、trades=333。相對C3：Return -15.78pp、MDD -0.68pp、RoMD -0.61、EV -0.09R。此pair-level結果只作迭代研究OOS證據，不回流修改C18搜尋規則。
- 最終多arm summary尚未完成，因聚合C17／C18兩個pair的重複C3 baseline時拋出：`ValueError: 多DL比較的共用基準不一致: arm=C3, key=resource_aware_selector_timing_max_ms, first=0.088, repeated=0.0352`。因此正式C17／C18 selector timing table尚未取得，不把pair console的整體elapsed秒數當selector CPU比較。

### 根因

`filters/breakout_quality/strategy_comparison.py::_assert_same_shared_baseline()`原本把所有summary key都當成deterministic baseline identity；SR-C18加入`resource_aware_selector_timing_total/median/p95/max_ms`後，C3在不同controlled pair被重播時會產生正常CPU jitter，因此錯把非決定性runtime measurement當策略結果差異。另`core/portfolio_entries.py`即使`policy is None`也量到wrapper執行時間，使DL-off C3帶有沒有語意的selector timing。

### 修正

1. `policy is None`時直接回傳canonical default diagnostics，`selector_elapsed_ns=0`；DL-off baseline不再宣稱執行Max-DL selector。
2. shared-baseline equality明確排除全部`resource_aware_selector_timing_*`；Return／MDD／EV／trade count／resource deterministic diagnostics仍維持exact一致性檢查。
3. direct synthetic新增timing-only mismatch可聚合案例，並保留`total_return_pct`不同必須拒絕的原案例；目前`validate_strategy_compare_config_driven_app_contract_case`為33/33 PASS。

### 下一步

套用修正後以相同`config/strategy_compare.py`正式選單重跑，不改C18 selector。重跑目的只完成final summary與同run C17／C18 selector total／median／p95／max CPU time比較；在正式summary完成前Registry維持`FINAL_REPORT_RERUN_REQUIRED`。


## 2026-08-08 — SR-C18正式closure：Selector freeze，研究主線轉回DL品質

### 正式結果

使用者本地`apps/strategy_compare.py`以config fingerprint=`4621173de22d`完成`C3 / C17 / C18`正式summary；期間`2021-01-01 ～ 2025-12-22`，固定`PARAM-P2 / Min ROOS`、`DL-CONT12A / MR-12A`、max positions=10、rotation=off。

C18：Return=140.56%、MDD=14.73%、RoMD=9.54、Annual Return=19.33%、Log R²=0.8810、月勝率=60.00%、EV=0.57R、Exposure=91.98%、trades=333、same-param DL selection R=-26.25R。相對C17：Return -23.57pp、MDD +0.36pp、RoMD -1.88、Annual Return -2.27pp、EV -0.15R、Exposure -0.05pp、same-param DL selection R -46.57R。相對C3：Return -15.78pp、MDD -0.68pp、RoMD -0.61、EV -0.09R。

### Max-DL selector完整度

C18 Max-DL eligible=242日、repair=232日、seed原為fallback=34日、final fallback=0、feasible-ascent improvement=23日、1-swap local optimum=242日、K violation=0、resource-preservation violation=0。相對Min ROOS planned selected-count delta=0、reserved-capital delta=+1,019,846。

因此C18已完成本研究階段要求：所有可介入日皆取得合法K/R0 basket，且搜尋停止點皆為deterministic 1-swap local optimum；不再因C18當前MR-12A績效不佳而修改capital floor、K、repair/ascent規則、score cutoff、blend或其他selector語意。Global exact search先前已因N≈30～80 difficult cases超過10秒而排除production。

### 正式計算時間

同一正式run：
- C17 selector total=1306.82ms、median=0.660ms、p95=3.431ms、max=15.414ms、repair eval=22,605。
- C18 selector total=1668.23ms、median=0.829ms、p95=4.065ms、max=29.897ms、repair eval=23,456、feasible-ascent eval=25,803。
- C18/C17 slowdown：total=1.277×、median=1.256×、p95=1.185×、max=1.940×；絕對增量total僅361.41ms。整段score-ranking replay console仍約12秒，因此selector額外成本不構成後續反覆模型策略驗證的主要瓶頸。

### 判定

`SR-C18 RESULT_AVAILABLE / SELECTOR_FROZEN_FOR_MODEL_RESEARCH / NOT_PROMOTED_AS_STRATEGY`。

C18績效顯著低於C17，同時C18把DL objective推得更完整（selected score gain=+14.441 vs C17 +11.826，final fallback=0），卻使same-param DL selection R由C17 +20.31R降至C18 -26.25R。此證據定位為**現有MR-12A ranking品質不足以支撐更強DL主導**，而不是再調selector的理由。後續所有新continuous DL source固定使用SR-C18作策略驗證harness，讓模型差異直接暴露在相同K/R0、execution order、sizing與accounting下。

另核對目前程式：`MR-12A / PROFILE-strategy_aligned_no_time_all_event_mse`本身已使用`daily_percentile_regression`，Target為同日all-event `strategy_aligned_opportunity_no_time_r_v1` percentile，loss=MSE、epoch selection=`mean_daily_spearman`。因此下一個模型研究不得把「daily percentile regression」當新變數；較乾淨的下一步是固定同一Target／all-label scope／ARCH-inception_time_v1，改研究pairwise或listwise ranking loss。

## 2026-08-08 — MR-12B：All-event within-day Pairwise Ranker + C17/C18雙selector驗證

### 狀態

`MR-12B / DL-CONT12B / SR-C19 / SR-C20 IMPLEMENTED / RESULT_PENDING`

### 基準與研究動機

- 唯一程式基準：`test-branch-1_20260808_160555_0f9f831.zip`，SHA256=`2f17172fed82222cb7df72ec265c13905a8fc1bc1d49ee7d34cbe67706bba79a`。
- C18已完成selector freeze，但MR-12A在較完整Max-DL objective下績效與same-param DL selection R惡化，研究瓶頸定位為DL ranking品質。
- 使用者要求C17與C18都保留作固定validation harness，用來驗證新DL改善是否在較Max-DL selector上轉化最好；本輪不再修改K、R0、repair/ascent、execution order或任何capital/DL比例。
- MR-12A本身已是all-event daily-percentile regression，因此MR-12B不重做percentile target，而直接改learning objective/loss。

### MR-12B固定項與唯一scientific change

固定：`strategy_aligned_opportunity_no_time_r_v1`、all-label scope、`ARCH-inception_time_v1`、兩logit head、Adam、LR/weight decay/grad clip、Selection/Validation/OOS切分、OOS隔離、epoch selection=`mean_daily_spearman`、runtime score=`softmax_pass_probability`。

唯一scientific change：`daily_percentile_regression + MSE` → `daily_pairwise_ranking + RankNet-style pairwise logistic`。同一天Target不相等的pair才參與，tie忽略、pair等權；margin固定`PASS logit - REJECT logit`，loss為`softplus(-sign(target_i-target_j) * (margin_i-margin_j))`。為讓同日pair完整可見，training改為whole-date coherent packing：同一date不得拆到不同mini-batches；group每epoch仍只曝光一次，但optimizer step數可因date packing改變，這是pairwise objective的必要batch semantics，不是額外調參。

Manifest/report明確保存pair scope、pair weighting、margin、whole-date batching與runtime score contract；`ranking_score_store`依profile嚴格驗證objective/loss/target/scope與pairwise artifact semantics。Strategy Compare不得自動訓練MR-12B。

### 固定策略驗證矩陣

- `C17 = MR-12A + max-DL minimum-repair`（anchor）
- `C18 = MR-12A + max-DL feasible-ascent`（anchor）
- `C19 = MR-12B + 完全相同C17 selector`
- `C20 = MR-12B + 完全相同C18 selector`

核心contrast：`C19-C17`（C17下純model gain）、`C20-C18`（C18下純model gain）、`C20-C19`（同MR-12B在較Max selector上的轉化）；另保留C19/C20相對C3作最終策略經濟參考。不得用這些forward/OOS結果回頭調pair sampling、loss、K/R0或selector。

### GPT獨立直接驗證

- MR-12B profile/CLI identity、all-label/no-time target、pairwise loss與mean-daily-Spearman epoch metric。
- Whole-date batch不切分同日competition set。
- 正序margin的pairwise loss低於反序且gradient有限。
- C17/C18×MR-12A/MR-12B雙selector矩陣與model-only/selector-conversion contrasts。
- MR-12A、PIT builder與config-driven strategy compare相關direct regression持續通過。
- GPT未執行`apps/test_suite.py`，亦未執行正式模型訓練；正式結果待使用者本機執行。


## 2026-08-08 — MR-12B formal consistency CLI regression closure

### 狀態

`MR-12B IMPLEMENTED / RESULT_PENDING` 不變；本輪只修formal synthetic CLI contract，未修改pairwise loss、whole-date batching、artifact semantics、C17/C18 selector或C19/C20策略矩陣。

### 使用者正式bundle

- 程式基準：`test-branch-1_20260808_162301_1defb21.zip`，SHA256=`d53c626ff7f8a34e921ed8eb3a7c8dba42ea4980726846b9a65c104fb878257b`。
- Debug bundle：`to_chatgpt_bundle_20260808_162439_7a2ae34f.zip`，SHA256=`439a04c6af18538a8901b6e3428189e8deb73c5c815810f98847f2ae93797f4e`。
- formal摘要：quick gate PASS、chain checks PASS、ml smoke PASS；consistency執行約91秒後因未產生summary被wrapper標記`missing_summary_file`；meta quality FAIL為`coverage_synthetic_suite_runs_successfully`、`coverage_key_targets_hit`與`performance_required_step_summaries_present`三項。

### 根因

`consistency.log`顯示真正第一個例外發生於`validate_dataset_cli_contract_case`：MR-12B實作後Continuous模型子選單正式將`[1/Enter]`定義為「訓練目前模型 → forward-OOS模型報表」，原synthetic仍使用歷史輸入`[Enter main, Enter submenu, 0]`並期待PIT route。第二個`0`實際被full-train確認提示消耗，返回主選單後mock input耗盡而拋`StopIteration`，因此`tools/validate/cli.py`未完成summary輸出。另同一validator的Workflow status expected仍硬編`unique_group_sampling / binary_classification`與Binary artifact表，與目前config-driven MR-12B continuous workflow不符。

### 修正

1. PIT route synthetic改為明確輸入`2`，確認後再返回主選單；不再依子選單預設項的歷史位置碰巧路由。
2. 新增獨立案例直接驗證Continuous模型子選單`Enter`會進`_interactive_continuous_full_train`，因此full-train與PIT兩條正式入口都有直接contract。
3. Missing-dataset PIT preparation案例同樣改為明確`2` route，保留`build-dataset → prepare-continuous-target → build-point-in-time-scores → audit-point-in-time-scores`原契約。
4. Workflow status測項改由當前`get_breakout_quality_workflow_settings()`派生profile／training objective／scope；Binary與Continuous各驗自己的正式status surface，兩者共同禁止使用者本機絕對路徑。validator不再把目前config某個歷史值硬編成唯一合法答案。

### 獨立驗證

- `validate_dataset_cli_contract_case`：171/171 PASS。
- MR-12B pairwise direct contract：7/7 PASS。
- MR-12A regression：4/4 PASS。
- PIT builder：22/22 PASS。
- config-driven Strategy Compare：33/33 PASS。
- 本輪未執行`apps/test_suite.py`或formal consistency/meta-quality step；正式閉環待使用者套patch後以單一正式入口重跑。

## 2026-08-08 — Strategy Compare completed-pair／shared-baseline reuse infrastructure

### 狀態

Infrastructure completed；`MR-12B / DL-CONT12B / SR-C19 / SR-C20`仍維持`IMPLEMENTED / RESULT_PENDING`。本輪不修改MR-12B pairwise loss／batching／artifact semantics，也不修改SR-C17／SR-C18 selector。

### 程式基準

- 使用者指定ZIP：`test-branch-1_20260808_163726_2c8a0b5.zip`
- SHA256：`d43b07a46472b574f4125b95e01f3b3f700308c119c2528ea39d7ba8d3195ea8`
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 唯一變更

`config/strategy_compare.py`新增兩個正式execution policy：

- `reuse_completed_results=True`：跨run搜尋已完成pair，只有pair replay fingerprint完全一致才可直接重用。
- `reuse_shared_baseline=True`：同一`param_source / rule_policy`下，DL-off baseline最多執行一次；若已有compatible歷史pair可直接從該pair重建baseline，否則本次run第一個新pair建立baseline後供後續新pair共用。

Pair fingerprint不使用整份config fingerprint，而只包含會影響replay的dataset、comparison period、param policy、max positions、rotation、param-source contract、DL source identity、off/on runtime contract、strategy engine schema與對應`param / model / manifest / forward_scores` SHA256。Contrast、arm顯示名稱、description與report-only設定不參與pair identity；因此新增MR-12B contrasts不會讓C17/C18歷史結果失效。任何param／model／manifest／forward-score SHA或period/runtime contract變更都必須cache miss並重新replay。

歷史pair命中後會複製完整pair正式工件到本次run並在top-level payload／manifest記錄`pair_execution=REUSE`與來源；新pair記錄`RUN`。Shared baseline只支援score-ranking path，重用前engine再次驗證dataset、param SHA、param policy、optional-filter policy、shared overrides、max positions、rotation與comparison period一致，不只依orchestrator cache判斷。

### MR-12B預期執行矩陣

若C17/C18既有正式工件與目前CONT12A／PARAM-P2 identity仍一致，MR-12B模型工件完成後正式計畫應為：

- `REUSE C3`
- `REUSE C17`
- `REUSE C18`
- `RUN C19`
- `RUN C20`

C19與C20都重用同一C3 baseline；因此本輪只需執行兩條新的MR-12B DL-on replay，不再重算既有MR-12A arms。

### GPT獨立驗證

`validate_strategy_compare_config_driven_app_contract_case`新增completed-pair cache、score SHA invalidation、report-only config independence、same-run shared-baseline及baseline contract rehydration／mismatch rejection案例；direct contract共37項全部PASS。另以完整orchestration mock確認C17/C18命中cache而C19/C20未命中時，canonical engine實際只呼叫2次，兩次皆收到`baseline_reuse_dir`。GPT未執行`apps/test_suite.py`。

## 2026-08-08 — MR-12B formal consistency：4×2 validator config-sensitive profile closure

### 狀態

`MR-12B / DL-CONT12B / SR-C19 / SR-C20 IMPLEMENTED / RESULT_PENDING`不變；本輪只修formal synthetic expected，不修改pairwise loss、whole-date batching、artifact semantics、Strategy Compare cache、C17/C18 selector或C19/C20策略矩陣。

### 使用者正式bundle

- 程式基準：`test-branch-1_20260808_170549_2bd495e.zip`，SHA256=`9ec5af4554fd986fcefd9591c2687665d393c6f0c82edafaf767cf52468f233b`。
- Debug bundle：`to_chatgpt_bundle_20260808_170738_8d95d831.zip`，SHA256=`3fd1670b7cf405ad481c4974cd36b518433a6cdae700af195ec087c888b2e1ef`。
- formal摘要：quick gate PASS、chain checks PASS、ml smoke PASS；consistency FAIL僅1項；meta quality只剩`coverage_synthetic_suite_runs_successfully` FAIL。

### 根因

`consistency_failures_20260808_170733.csv`唯一失敗為`BREAKOUT_QUALITY_BINARY_DL_PARAM_ADAPTATION / binary_dl_four_by_two_risk_only_fold_contract`：validator expected tuple最後一欄固定為歷史`unique_group_sampling`，但`strategy_param_training._parse_args([])`依法讀取目前workflow設定，MR-12B active profile為`strategy_aligned_no_time_all_event_pairwise`。Runtime／parser行為正確，錯誤在synthetic把`config/`目前值硬編為唯一合法答案，違反`PROJECT_SETTINGS C8`。meta-quality coverage failure只是同一synthetic suite returncode=1的下游結果。

### 修正

4×2 risk-only synthetic先獨立讀取`get_breakout_quality_workflow_settings()`，expected profile改用`workflow_settings.experiment_profile`，再與`parse_dl_param_adapt_args([]).experiment_profile`比較；因此仍可攔截parser default未忠實跟隨config，但不限制使用者目前必須使用特定profile。`RISK_SEARCH_FIELDS`、P2/P3 DL off/on、optional filters all-off、all-rule filters all-off與fold runtime override等其餘contract維持exact比較。

### GPT獨立驗證

`validate_breakout_quality_binary_dl_param_adaptation_contract_case`直接執行9/9 PASS。另掃描`tools/validate/`內其餘`unique_group_sampling`引用，保留明確隔離的歷史／固定profile案例，不把合法固定fixture誤改為動態config。GPT未執行`apps/test_suite.py`或formal consistency/meta-quality step；正式閉環待使用者套patch後以單一正式入口重跑。


## 2026-08-08 — Breakout Quality策略層正式輸出簡易報表契約

### 狀態

Infrastructure completed；不占用新的`MR-*`／`SR-C*`／`AUD-*` identity。`MR-12B / DL-CONT12B / SR-C19 / SR-C20`狀態與研究語意不變。

### 程式基準

- 使用者指定ZIP：`test-branch-1_20260808_171335_08a20ea.zip`
- SHA256：`560d1ac9c1907c027f86048a22b2df3fc75f7c4ff17e7042677e98a486d3cc36`
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 使用者要求與唯一變更

使用者要求Breakout Quality策略層輸出一律具有簡易報表。本輪將此要求提升為全域正式輸出契約：Strategy Compare、策略參數適應與strategy gate等頂層正式策略結果只要產生持久工件，就必須同時有console易讀摘要與Markdown；JSON／CSV／manifest只作同一結果的詳細工件，不要求每個支援檔重複一份報表。

現有非cache策略流程本來已各自具有console renderer與Markdown；真正缺口出現在新增completed-pair cache後的Strategy Compare REUSE：舊pair目錄雖被整體複製，但orchestrator只顯示`REUSE`路徑，不重新顯示該pair簡報，且複製的Markdown仍是舊run metadata。修正後canonical engine可直接由`strategy_comparison.json` payload解析metadata／baseline／active arm／delta／yearly／selection diagnostics，RUN與REUSE共同使用同一套Markdown與console renderer；REUSE更新本次run metadata後會重新materialize`strategy_comparison.md`並顯示簡報。Pair cache required-files新增Markdown，缺簡易報表的歷史pair不視為完整cache。

### 固定研究／runtime條件

本輪不修改MR-12B pairwise loss、whole-date batching、CONT12B artifact semantics、C17/C18 selector、C19/C20矩陣、策略sizing／accounting／execution、pair fingerprint或shared-baseline cache identity。報表只重用canonical pair JSON既有數值，不另算第二套策略指標。

### GPT獨立驗證

- config-driven Strategy Compare direct contract新增pair payload同時產生Markdown＋console簡報、cache required-files包含Markdown、C17/C18 REUSE重新materialize本次Markdown並在互動執行顯示相同pair簡報；相關contract直接執行39項全部PASS。
- 新增`validate_breakout_quality_strategy_readable_report_contract_case`，集中盤點Strategy Compare pair／multi-arm、Binary參數適應、Selection策略適應、optional-filter gate、Binary DL rule gate與trade-path label gate共七個正式策略結果入口，要求persistent Markdown與`core.console_report`-based console renderer。
- GPT未執行`apps/test_suite.py`；正式double check仍由使用者本機單一正式入口執行。


## 2026-08-08 — `apps/breakout_quality.py`正式輸出簡易報表

### 狀態

Infrastructure completed；不占用新的`MR-*`／`SR-C*`／`AUD-*` identity。MR-12B、DL-CONT12B、C17/C18/C19/C20研究語意與策略結果均不變。

### 程式基準

- 使用者指定ZIP：`test-branch-1_20260808_172555_4bbab0b.zip`
- SHA256：`ef689ab698a18f61e0c1a0b3eb5a4b0bedddf39b1441fe04b1f19a72a0434aad`
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 使用者要求與唯一變更

使用者澄清前一輪所稱`strategy_quality`實際指`apps/breakout_quality.py`，要求該App本身的正式輸出都具備簡易報表。本輪在App command dispatcher增加統一summary：成功subcommand保留原本詳細console／JSON／CSV／Markdown，最後再輸出`Breakout Quality 簡易報表`並保存`outputs/filters/breakout_quality/<filter_id>/simple_reports/<command>.md`。摘要固定顯示command、status、filter、architecture、profile、objective與elapsed；Dataset、Binary report、Continuous ranker、PIT audit與Continuous Target只讀既有canonical工件補核心數字，不建立第二套模型統計。完整workflow另輸出最終summary；Audit互動執行後直接顯示最近Audit摘要。

### 固定研究／runtime條件

不修改Dataset／Label、MR-12B pairwise loss與whole-date batching、模型權重、PIT score語意、C17/C18 selector、C19/C20、Strategy Compare cache／fingerprint、策略sizing／accounting／execution。

### GPT獨立驗證

`validate_breakout_quality_app_simple_report_contract_case`直接6/6 PASS，驗證console、persistent Markdown、專案相對路徑、active identity與subcommand／workflow掛載；既有`validate_dataset_cli_contract_case` 174項PASS。GPT未執行`apps/test_suite.py`。

## 2026-08-08 — MR-12B正式策略驗證：Pairwise改善成立，較Max-DL harness放大模型品質差異

### 狀態

`MR-12B / DL-CONT12B = RESULT_AVAILABLE / MODEL_ECONOMIC_IMPROVEMENT_SUPPORTED`；`SR-C19 / SR-C20 = RESULT_AVAILABLE / MODEL_VALIDATION_ARM`。C17/C18 selector語意維持凍結，不依本次forward/OOS結果調整。

### 正式controlled replay

- Strategy Compare fingerprint：`a530e2b6f5f4`
- 期間：2021-01-01～2025-12-22
- Dataset：full
- Params：`PARAM-P2 / Min ROOS`
- C17/C19共用minimum-repair selector；C18/C20共用feasible-ascent selector。
- C19唯一模型差異：`DL-CONT12A / MR-12A MSE → DL-CONT12B / MR-12B pairwise`。
- C20唯一模型差異同上。

### 主要結果

- C19：Return 181.75%、MDD 16.19%、RoMD 11.23、Annual 23.19%、EV 0.70R、Exposure 92.14%、same-param DL selection R +25.05R。
- C20：Return 179.54%、MDD 14.48%、RoMD 12.40、Annual 23.00%、EV 0.84R、Exposure 92.21%、same-param DL selection R +55.31R。
- C19-C17：Return +17.62pp、MDD +1.81pp、RoMD -0.19、EV -0.02R、selection R +4.74R。
- C20-C18：Return +38.98pp、MDD -0.25pp、RoMD +2.86、EV +0.27R、selection R +81.57R。
- C20-C19：Return -2.22pp，但MDD -1.70pp、RoMD +1.17、EV +0.14R、selection R +30.27R。

### 判定

MR-12B pairwise objective相對MR-12A MSE的模型經濟改善成立。更重要的是，C18-style較Max-DL selector在MR-12A弱ranking時放大錯誤，但在MR-12B改善後同時把MDD、RoMD、EV與selection R顯著拉高，證明C17/C18雙harness具有辨識模型品質與selector轉化差異的價值。後續新模型必須同時跑兩者；不得因C19 raw Return略高或C20風險調整較佳而刪除任一harness。


## 2026-08-08 — MR-12C實作：All-event ListNet Top-one Listwise Ranker

### 狀態

`MR-12C / DL-CONT12C / SR-C21 / SR-C22 = IMPLEMENTED / RESULT_PENDING`。本輪未執行正式模型訓練、forward-OOS推論或策略replay。

### 程式基準

- 使用者指定ZIP：`test-branch-1_20260808_174356_04c0505(2).zip`
- SHA256：`6144ba09731c6e75da5d49334d746c3aced6b1d35e85d06a9f4ade7895730104`
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 唯一scientific change

固定MR-12B的：

- `strategy_aligned_opportunity_no_time_r_v1`
- `all_labels`
- `ARCH-inception_time_v1`
- optimizer／LR／weight decay／gradient clip
- Selection／Validation／OOS split
- epoch selection=`mean_daily_spearman`
- runtime score=`softmax PASS probability`
- C17/C18 K/R0／minimum-repair／feasible-ascent／execution-order語意

只將training objective由`daily_pairwise_ranking / pairwise_logistic`改為`daily_listwise_ranking / listnet_top_one_cross_entropy`。

### Listwise契約

每一交易日完整candidate list不得跨mini-batch切分。模型排序score使用`PASS logit - REJECT logit`；真實daily-percentile target由高到低形成tie groups。每個rankable date把daily-percentile target經softmax轉成target top-one distribution，模型`PASS logit - REJECT logit`經softmax轉成prediction distribution，再計算整日cross-entropy；相同target自然得到相同target mass，不建立任意tie順序。各rankable date等權；不新增temperature、Top-K weight、cutoff或其他可依OOS調整的超參數。

### 策略驗證矩陣

- `C21 = MR-12C + 完全相同C17 minimum-repair selector`
- `C22 = MR-12C + 完全相同C18 feasible-ascent selector`
- 核心contrast：`C21-C19`（固定C17-style下Listwise vs Pairwise）、`C22-C20`（固定C18-style下Listwise vs Pairwise）、`C22-C21`（同MR-12C在較Max selector上的轉化）。
- C17/C18/C19/C20持續保留，completed-pair cache可重用；Strategy Compare不得自動訓練CONT12C。

### GPT獨立直接驗證

- listwise profile／CLI identity、Target／scope／architecture family與runtime score契約。
- whole-date batching不切分同日candidate list。
- ListNet top-one cross-entropy對正確完整排序給出較低loss、gradient有限，且交換同target tie成員不改loss。
- 真optimizer step可完成且model parameters確實更新。
- C19/C20與C21/C22雙selector矩陣及三個核心contrast。
- MR-12B pairwise direct contract持續通過。
- GPT未執行`apps/test_suite.py`；正式結果待使用者本機執行。


## 2026-08-08 — Research單一正式入口與config-driven dispatch

### 狀態

Infrastructure completed；不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity。既有MR-12C／DL-CONT12C與C17～C22研究語意、模型工件、selector與策略結果狀態均不變。

### 程式基準

- 使用者指定ZIP：`test-branch-1_20260808_182645_c736a49(2).zip`
- SHA256：`9ed8580d4ad2697cc36417086f08d595e72b327cdc4fec3c82696cb3f9f9c8b2`
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 使用者要求與唯一變更

研究功能改由單一`apps/research.py`作為使用者正式入口，主選單只選工作類型：`模型訓練／策略參數最佳化／策略組合比較／Audit／診斷／查看目前設定與工件狀態`。特定model、實驗標的、比較arm與Audit module不得成為選單項目；active model/provider由`config/research.py`指定，Audit module由`config/audit.py`指定，Strategy Compare仍由`config/strategy_compare.py`指定。

`[2] 策略參數最佳化`直接dispatch至既有`tools.optimizer`，保留原optimizer互動選單與runtime policy，不建立第二套optimizer UI。原`apps/breakout_quality.py`的完整model workflow移至既有`tools/filters/breakout_quality/application.py`作為Breakout Quality model provider；Strategy Compare與Audit繼續重用既有正式service。`apps/research.py`只負責menu、config resolution與dispatch，不承擔模型／策略計算。

舊研究wrapper `apps/breakout_quality.py`、`apps/ml_optimizer.py`、`apps/strategy_compare.py`、`apps/audit.py`退出正式入口；因patch ZIP無法表達刪除，套用patch後需依交付命令移除。`PROJECT_SETTINGS C9`同步明確化為「單一physical Research入口、工作類型與application/service責任仍分離」。

### 固定研究／runtime條件

本輪不修改Dataset／Label、MR-12B/MR-12C objective、模型權重、PIT score、DL source identity、C17/C18 selector、C19～C22 validation arms、策略參數搜尋空間、replay、sizing、accounting、execution、Strategy Compare fingerprint/cache或Audit算法。

### GPT獨立驗證

- `apps/research.py --help`、`model --help`、`optimizer --help`、`compare --help`、`audit --help`均可直接執行；主選單文字與`[2] → 原optimizer backend` dispatch另以mock直接驗證。
- 受入口遷移影響的CLI、local-regression、architecture與Breakout Quality／Strategy Compare／Audit synthetic contracts合計480項直接檢查全部PASS；其中Dataset／CLI 162項、17組Breakout Quality source/runtime contract 252項均為0 failure。
- 全專案281個Python檔`py_compile` PASS；裸`except:`=0，`core/`／`filters/`反向import `apps`或`tools.audit`=0，current code／operational docs舊Research entry引用=0。
- GPT未執行`apps/test_suite.py`或formal pipeline；正式double check仍由使用者本機單一正式入口執行。

## 2026-08-08 — Research單一入口 formal consistency 閉環修正

### 狀態

Infrastructure fix completed；不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity，Registry scientific identity與目前model／selector判定不變。

### 程式與formal bundle基準

- 使用者正式suite後ZIP：`test-branch-1_20260808_191745_a497f1c.zip`
- ZIP SHA256：`1bc4f037f27f31aae40fb7121fc956c5ea1777165a97c4454d7f5dbb4d6050d5`
- Formal bundle：`to_chatgpt_bundle_20260808_191933_7a62f740.zip`
- Bundle SHA256：`ba4f4cbd39891e43aa33e2c7d37b0b3d102940f8b2961e187a3223012cc86257`

### Formal結果與根因

使用者本機formal double check：quick gate PASS、chain checks PASS、ml smoke PASS；consistency只有7個FAIL，全部來自`META_REGISTRY_CHECKLIST_ENTRY`的`B170/B182/B184/B188/B189/B190/B194_declared_entry_file_exists`。原因是Research單一入口遷移後，`doc/TEST_SUITE_CHECKLIST.md`現行B2主表仍把已刪除的`apps/breakout_quality.py`／`apps/strategy_compare.py`／`apps/audit.py`列為建議落點。Runtime、策略計算與模型流程沒有失敗。

另發現meta checker原本每個DONE B-row只驗證第一個`apps/core/tools`路徑，因此B171等列後段殘留的舊Research wrapper未被本次7個FAIL揭露。

### 修正

- 現行B2主表所有Research舊wrapper引用改為`apps/research.py`，並同步更新B182/B184/B186/B188/B189/B190/B194的單一入口工作類型語意；歷史G紀錄與Experiment Log舊指令保留作歷史證據，不改寫。
- `validate_registry_checklist_entry_consistency_case`改為逐列驗證建議落點中所有`apps/config/core/filters/tools/*.py`宣告，不再只取第一個path；失敗時note直接列出missing paths。
- B23 checklist／registry／正式入口一致性契約同步補上「DONE主表列所有宣告Python路徑必須存在」。

### 固定研究／runtime條件

不修改Dataset／Label、MR-12B／MR-12C訓練目標、模型權重、PIT scores、Strategy Compare arms／fingerprint／cache、C17/C18 selector、策略參數、replay、sizing、accounting、execution或Audit算法。

### 獨立驗證

- `validate_registry_checklist_entry_consistency_case`：1217／1217 PASS；強化後同時檢出並修正原本被第一路徑遮蔽的B50／B172／B177現行stale declaration。
- 受影響直接契約：Dataset／CLI 162／162 PASS、Strategy Compare config-driven app 39／39 PASS、Audit framework 8／8 PASS、Breakout Quality簡易報表6／6 PASS；合計1432項direct checks、0 failure。
- `apps/research.py`、`model`、`optimizer`、`compare`、`audit` help route全部可正常解析；`optimizer`仍原樣轉交既有optimizer service。
- 全專案282個Python檔AST parse／compile PASS；裸`except:`=0、pass-only exception handler=0、local import cycle=0、`core/`／`filters/`反向import `apps`或`tools.audit`=0；現行DONE checklist宣告Python path缺件=0，`PROJECT_SETTINGS.md`／`ARCHITECTURE.md`／`CMD.md`舊Research wrapper引用=0。
- `G`表1461筆狀態鏈、日期／ID排序、重複`NEW`、no-op transition均為0異常，B23最新狀態為DONE；既有歷史列未改寫，只新增本輪B23兩筆收斂紀錄。
- GPT未執行`apps/test_suite.py`或formal pipeline；formal suite結果須由使用者套用patch後於本機正式入口重跑確認。

## 2026-08-08 — Continuous ranker簡易報表P1口徑修正：Competition-day Top-K與Validation標示

### 狀態

Infrastructure/reporting fix completed；不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity，`MR-12B / DL-CONT12B`研究身份、模型objective、權重、selector與既有策略判定均不變。

### 程式基準

- 當前GPT交付基準：`p1_topk_console_report_patch_20260808.zip`
- SHA256：`591bcf33890aa97dda13efdb5617a3782b784ca2423f716f775d6c860a26bd10`
- 全專案檢查使用使用者最新版`test-branch-1_20260808_195246_d2960ce.zip`解壓後覆蓋上述P1 patch的完整工作樹。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 使用者實跑揭露與修正

1. 原P1的`NDCG@K / Top-K Target / Lift / Oracle overlap`把`candidate_count <= K`的交易日也納入；這些日子全部候選本來就會進Top-K，排序不影響實際選股，且Oracle overlap／NDCG會被結構性灌高。修正後Top-K主指標固定只統計`candidate_count > K`的competition days；JSON新增`all_date_count / competition_date_count / excluded_non_competition_date_count / competition_rule / top_k_scope`，K-boundary仍在同一competition scope計算。
2. 使用者同次執行看到epoch selection的Validation Daily Spearman `0.1169`，以及完整Selection refit checkpoint對原Validation rows再評估的`0.1516`。後者的rows已納入final refit，不能再被解讀成選模Validation。Console／Markdown改為明確分開`選模 Validation rho`與`重訓後原 Validation rho`，並標示checkpoint後split metrics不再參與選模。
3. `trade R: not found:`原本直接輸出本機絕對路徑；改以`project_relative_display_path`顯示專案相對路徑，canonical path仍可留在JSON metadata。

### 固定研究／runtime條件

不修改Dataset／Target、MR-12B pairwise logistic、whole-date batching、optimizer／LR、epoch selection規則、模型權重語意、PIT scores、C17/C18 selector、C19/C20策略arm、sizing／accounting／execution。Top-K／K-boundary仍只屬checkpoint後描述性評估，不進loss、gradient、epoch selection或selector。

### GPT獨立驗證

- `validate_breakout_quality_pairwise_ranker_contract_case`：9項、0 failure；新增非競爭日排除與缺trade工件相對路徑契約。
- `validate_breakout_quality_app_simple_report_contract_case`：6項、0 failure；`validate_dataset_cli_contract_case`：163項、0 failure，確認既有簡易報表仍保留並加入competition-day Top-K／K-boundary與清楚Validation標示。
- `apps/research.py --help`與`apps/research.py model --help`可正常解析；全專案282個Python檔AST parse／compile PASS，裸`except:`=0、pass-only handler=0、簡單local import cycle=0、`core/filters`反向依賴`apps`或`tools.audit`=0；現行DONE checklist建議落點147個具體Python path缺件=0。
- GPT未執行`apps/test_suite.py`或formal pipeline；正式double check仍由使用者本機單一正式入口執行。

## 2026-08-08 — Continuous ranker P2：MR-12A/B/C paired、Random baseline、Dynamic-K

### 狀態

Infrastructure/diagnostic implementation completed；實際研究結果待使用者本機既有工件執行。此輪不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；MR-12A/B/C frozen model identity、C17/C18 selector與C17～C22既有策略結果均不變。

### 程式基準

- 使用者指定最新版ZIP：`test-branch-1_20260808_202244_4926c2c(1).zip`
- SHA256：`0305b59118a41c81df2b3364e380476bfe763ec800af9fc7655f315690b6c8af`
- 全新解壓工作目錄：`/mnt/data/p2_work`
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`，並另檢視`TEST_SUITE_CHECKLIST.md`與`apps/test_suite.py`覆蓋邊界；未執行formal suite。

### P2目的與固定比較

P1已證明MR-12B在fixed K=10 competition days存在Top-K lift，但K-boundary concordance只小幅高於50%。P2不開新模型、不重訓、不重跑Strategy Compare，而是只讀MR-12A／MR-12B／MR-12C既有canonical frozen score，補齊三種描述性比較：

1. **Fixed-K paired**：Selection與Forward OOS都要求三模型`ticker/date/group_index/split`候選row identity完全一致，且raw／daily-percentile target bit-semantic一致後才允許同日比較；K沿用正式策略max positions設定。
2. **Exact random baseline**：不使用Monte Carlo。每交易日解析計算uniform random ordering的期望NDCG、Oracle Top-K overlap=`K/N`、Top-K lift=`0`、Boundary concordance=`0.5`與Boundary gap=`0`，再以交易日等權聚合。
3. **Dynamic-K paired**：以目前C17 frozen constrained selector既有Strategy Compare pair作reference；只讀`score_ranking_daily_capacity.csv`與`score_ranking_orderable_candidates.csv`。K固定取C17 `Resource_Aware_Max_DL_Eligible`日的`Resource_Aware_Pre_Market_Order_Limit`，候選集合取當日實際orderable candidates，再以`ticker + signal_date`對回MR-12A/B/C OOS frozen score。只有三模型score與target完整覆蓋的日期才進paired品質統計，另明列eligible／orderable／partial／missing／full-score coverage。

同日paired contrast固定至少輸出`MR-12B−MR-12A`與`MR-12C−MR-12B`的平均Δ、median Δ、左側較佳日比例與同日數；指標包含NDCG@K、Top-K Target/Lift、Oracle overlap、K-boundary concordance與Boundary raw-target gap。

### 正式入口與輸出

Research model menu新增：`[4] 比較 MR-12A/B/C → paired／random／Dynamic-K`。正式路徑仍為`apps/research.py → 模型訓練`，比較命令為read-only provider command `compare-continuous-rankers`。輸出集中於：

- `outputs/filters/breakout_quality/<filter_id>/continuous_ranker_comparison/continuous_ranker_comparison.json`
- `outputs/filters/breakout_quality/<filter_id>/continuous_ranker_comparison/continuous_ranker_comparison.md`
- 既有`simple_reports/compare-continuous-rankers.md`持續提供console簡易摘要；Profile明列`MR-12A / MR-12B / MR-12C`，Objective明列read-only paired quality comparison，避免誤標成目前active單一training profile。

### 固定研究／runtime條件

不修改Dataset／Continuous Target、MR-12A MSE／MR-12B pairwise logistic／MR-12C ListNet、training sampling、optimizer、epoch selection、checkpoint、forward score、PIT score、C17/C18 selector、Strategy Compare fingerprint/cache、策略參數、replay、sizing、accounting或execution。P2的Selection/OOS Target只在checkpoint後作描述性診斷，不進loss、gradient、epoch selection、模型選擇或selector調整。

### GPT獨立驗證

- MR-12A all-event direct contract：4/4 PASS；MR-12B/P2 extended contract：13/13 PASS；MR-12C contract：8/8 PASS。
- Dataset／Research CLI contract：165/165 PASS；Breakout Quality簡易報表contract：7/7 PASS；P2 option 4 route、custom comparison identity與相對路徑均已直接驗證。
- Synthetic registry metadata：8/8 PASS；Registry／Checklist一致性：1217/1217 PASS。上述直接檢查合計1422項、0 failure。
- 全專案284個Python檔`py_compile`／AST parse PASS；裸`except:`=0、pass-only exception handler=0、`core/filters`反向依賴`apps`或`tools.audit`=0、簡單local import cycle=0。
- GPT未執行`apps/test_suite.py`或formal pipeline；P2實際A/B/C比較結果與formal double check待使用者本機執行。

## 2026-08-09 — P2 config-driven menu／comparison identity infrastructure fix

### 狀態

Infrastructure/UI contract fix completed；不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity，MR-12A/B/C frozen model identity、C17/C18 selector與P2診斷算法不變。

### 程式基準

- 使用者最新版完整ZIP：`test-branch-1_20260808_202244_4926c2c(1).zip`
- ZIP SHA256：`0305b59118a41c81df2b3364e380476bfe763ec800af9fc7655f315690b6c8af`
- 上輪P2 patch：`p2_paired_random_dynamic_k_patch_20260808.zip`
- P2 patch SHA256：`199e575b1d7576c673378e24fe75994efea2d97713db5e017cdde3db0cf7ba39`
- 本輪工作樹：以上完整ZIP全新解壓後覆蓋P2 patch，再進行config-driven修正。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 問題與修正

P2初版正式model menu直接顯示`MR-12A/B/C`，App簡易摘要與比較renderer亦固定假設MR-12B為摘要模型、MR-12A為baseline、C17為Dynamic-K reference。這違反既有config-driven原則，也會讓後續只改config時UI／摘要與實際比較對象分叉。

本輪修正：

1. `doc/PROJECT_SETTINGS.md`新增永久條款：互動選單不得硬編碼特定experiment／model／arm／ID或目前設定值；可用項目、顯示label、reference、比較組合與摘要對象必須由`config/`／Registry／active settings驅動，validator不得鎖死目前config值。
2. `config/breakout_quality.py`把continuous-ranker comparison完整移到USER SETTINGS：`enabled`、`menu label`、`model_profiles`、`reference_arm`、`summary_pair`，並提供集中驗證／解析getter。
3. `tools/filters/breakout_quality/application.py`的`[4]`選單只顯示config提供的泛化label；enabled=false時不顯示。比較簡易報表的Profile、摘要模型與Δ比較名稱均由canonical comparison JSON/config動態生成，不再固定MR-12B／MR-12A。
4. `compare_continuous_rankers.py`的model specs、Dynamic-K reference、console／Markdown標題與契約文字全部由config設定與canonical payload生成；不再硬編碼MR-12A/B/C或C17。
5. synthetic validator改用隔離config override驗證runtime忠實採用設定，不再把目前MR/C ID或menu字串當唯一合法答案。

### 固定研究／runtime條件

不修改Dataset／Continuous Target、MR-12A MSE、MR-12B pairwise logistic、MR-12C ListNet、模型checkpoint／scores、PIT、Top-K／K-boundary公式、exact random baseline、Dynamic-K join語意、C17/C18 selector、Strategy Compare replay／fingerprint、策略參數、sizing、accounting或execution。

### GPT獨立驗證

- 全專案284個Python檔`py_compile`／AST parse PASS；裸`except:`=0、pass-only exception handler=0、`core/filters`反向依賴`apps`或`tools.audit`=0、local import cycle=0。
- 全專案互動選單literal掃描：含`[n]`之menu項目中硬編`MR-*`／`DL-*`／`SR-C*`／`Cxx`／`Axx`數量=0。
- 隔離config override直接驗證：menu label可改為任意泛化設定文字、comparison model IDs/order／reference arm／summary pair皆由`config/breakout_quality.py`解析；`enabled=False`時正式model menu不顯示比較項目；renderer亦只讀canonical payload identity。
- `doc/TEST_SUITE_CHECKLIST.md` B16已補config-driven menu contract；主表B16狀態仍為DONE且與最近`PARTIAL -> DONE` transition一致，該列Markdown欄數合法。
- formal synthetic registry仍註冊`validate_dataset_cli_contract_case`與`validate_breakout_quality_pairwise_ranker_contract_case`，可在使用者本機正式suite覆蓋本輪UI/config與P2比較契約。
- GPT未執行`apps/test_suite.py`或formal pipeline；正式double check仍由使用者本機正式入口執行。

## 2026-08-09 — Config-driven UI follow-up：移除非選單使用者可見的研究 ID 殘留

狀態：Infrastructure/UI consistency fix completed；不占用新的 `MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity，既有 MR-12A/B/C、C17～C22、P1/P2 診斷算法與研究結果均不變。

程式基準：使用者 ZIP `test-branch-1_20260809_020312_f38483f.zip`，SHA256 `be668fcd1262e296db4d377be6406191a534169ce8542e2ea10bfee0b92e72aa`。

本輪只修使用者可見文字的一致性：

1. `train_continuous_ranker.py` 完成訊息不再輸出硬編碼 phase（例如 `12B continuous ranker完成`），統一為泛化 `Continuous ranker完成`；正式 profile／experiment identity 仍由 canonical report/manifest 保存，不改模型語意。
2. `compare_continuous_rankers.py` 找不到 Dynamic-K reference pair 時，不再列出固定 `C17/C18/C19/C20/C21/C22`；錯誤訊息改為顯示目前 config 的 `reference_arm_id`，並泛化指向 `outputs/strategy_compare/runs/`。
3. `PROJECT_SETTINGS.md` B18 已是正式 UI 契約，本輪不重複新增條款；互動選單仍由 `comparison_settings.menu_label` 與 config-driven enable/profile/reference/summary settings 驅動。

不修改 Dataset／Target、training objective、optimizer、epoch selection、checkpoint、frozen score、Top-K／K-boundary、random baseline、Dynamic-K join、selector、Strategy Compare、sizing／accounting／execution。

## 2026-08-09 — P2 Dynamic-K repeated-signal occurrence merge fix

### 狀態

Infrastructure/diagnostic bug fix completed；不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity。MR-12A/B/C frozen score、C17/C18 selector、Top-K／K-boundary公式與既有策略結果均不變。

### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_021303_cfb6d8e.zip`
- SHA256：`de4f6714d95742e2aca0060942be543b14b8b7b479210224fec0287e9fbd35ce`
- 全新解壓工作目錄：`/mnt/data/p2_merge_fix`
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。

### 使用者實跑揭露與根因

`apps/research.py → 模型訓練 → 比較設定中的 Continuous Rankers`進入Dynamic-K時，`score_ranking_orderable_candidates.csv`可合法包含同一`(ticker, signal_date)`在不同`trade_date`持續orderable的occurrence；P2初版卻以`validate="one_to_one"`把orderable occurrence對 frozen score，因而對正常continuation/pending資料拋出`MergeError: Merge keys are not unique in left dataset`。

### 修正

1. Dynamic-K join改為明確的`many_to_one`：左側允許同一signal跨不同trade date重複成為orderable occurrence；右側每個模型的frozen score仍強制`(ticker, signal_date)`唯一，若右側重複則fail-fast並輸出sample，禁止many-to-many。
2. `score_ranking_orderable_candidates.csv`只讀Dynamic-K需要的`ticker / trade_date / signal_date`三欄，並改用既有`read_breakout_quality_csv`，同時避免未使用mixed-type欄造成`DtypeWarning`，並保留`0050 / 00643`等前導零ticker。
3. synthetic regression加入「相同signal在兩個後續trade dates持續orderable」案例，驗證兩天occurrence都保留、同一frozen score可合法重用且full-score coverage正確。

### 固定研究／runtime條件

不修改Dataset／Continuous Target、模型訓練、checkpoint、frozen score值、PIT、Strategy Compare replay、C17/C18 selector、Dynamic-K的每日K來源、候選membership、random baseline、Top-K／boundary公式、策略參數、sizing、accounting或execution。


## 2026-08-09 — P2首輪實跑：Fixed-K結果成立；Dynamic-K score-event-date join bug

### 狀態

`P2 FIXED_K_RESULT_AVAILABLE / DYNAMIC_K_INVALIDATED_PENDING_RERUN`。不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；MR-12B仍為current model research anchor，MR-12C仍維持REJECTED。此輪只修read-only診斷join，不修改任何模型、score、selector或策略replay。

### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_022059_051c97b(2).zip`
- SHA256：`afb1556a6374db8e5f5f0f08e982896cddabce4f9ae4e119ad59ea5797e05178`
- 全新解壓工作目錄：`/mnt/data/stock_p2_review`
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`或formal pipeline。

### 使用者本機P2結果

Fixed K=10的Forward OOS共有603個competition days。MR-12B相對MR-12A：NDCG `0.6905 vs 0.6615`（paired mean Δ `+0.0290`）、Top-K raw-target lift `0.4089R vs 0.2488R`（Δ `+0.1601R`）、Oracle overlap `59.80% vs 57.20%`（Δ `+2.60pp`）、Boundary concordance `52.98% vs 49.96%`（Δ `+3.02pp`）、Boundary gap `+0.2121R vs -0.0489R`（Δ `+0.2610R`）。此結果與C19/C20相對C17/C18的controlled replay方向一致，支持MR-12B broad OOS ranking quality改善。

Selection fixed-K則MR-12B略弱於MR-12A，而MR-12C在Selection描述性指標較高但正式策略經濟結果已被淘汰；Selection rows屬final-refit已見資料，僅保留描述性用途，不作新模型採用依據。

首輪Dynamic-K顯示C17 eligible=223日、三模型完整score僅94日、coverage `42.15%`、K=1～8且mean `1.34`；在該94日子集MR-12B Boundary=`43.94%`、相對MR-12A `-12.53pp`。此數字後續查明受diagnostic join bug污染，不得作MR-12D設計或MR-12B否定證據。

### 根因

正式Strategy Compare對continuation／re-entry已有canonical契約：交易候選的`signal_date`可能晚於最初breakout事件；runtime score與Future Target都必須優先使用candidate保存的`breakout_quality_score_date`（canonical `score_event_date`）對回模型／PIT工件，只有該欄缺少時才fallback `signal_date`。`strategy_compare_engine._strategy_selection_diagnostics()`已正確實作此契約。

P2 `_dynamic_orderable_frame()`卻只讀`ticker / trade_date / signal_date`，並以`ticker + signal_date`對MR-12A/B/C frozen OOS score。對沿用原始breakout score的continuation／re-entry occurrence會錯失合法score，造成大量partial-score days與錯誤的Dynamic-K paired樣本。

### 修正

1. P2仍以reference arm既有`score_ranking_orderable_candidates.csv`作共同候選集合，但額外讀取`breakout_quality_score_date`；若非空則解析成`score_event_date`，缺少時才fallback `signal_date`，非法日期fail-fast。
2. MR-12A/B/C frozen score lookup改為`ticker + score_event_date` many-to-one；右側每個模型仍強制該key唯一，禁止many-to-many。
3. coverage新增runtime score-event-date列數、fallback signal-date列數、以及`score_event_date != signal_date`列數，讓後續可直接確認continuation/re-entry映射是否被正確使用。
4. synthetic contract補入`signal_date`晚於原始`breakout_quality_score_date`的later occurrence案例，確保仍能取得原始frozen score。
5. 同輪移除`tools/validate/synthetic_breakout_quality_cases.py`直接讀`doc/PROJECT_SETTINGS.md`文字的反向政策測試，遵守`PROJECT_SETTINGS A7`；保留對實際strategy readable-report程式行為的contract檢查。

### 下一步

先用同一正式選單重新執行`apps/research.py → 模型訓練 → 比較設定中的 Continuous Rankers`。在修正後Dynamic-K coverage與結果取得前，不新增MR-12D、不改Pairwise loss、不改C17/C18 selector。若修正後coverage顯著提高且MR-12B在Dynamic-K仍弱於MR-12A，再做K-stratified（尤其K=1/2/3）與C18 feasible-swap attribution，判斷是否為top-prefix極前段排序問題；只有歸因成立後才設計新的model experiment。


## 2026-08-09 — P2 Dynamic-K重跑：score-date假設排除；complete-case raw-score限制確認

狀態：`P2_FIXED_K_RESULT_AVAILABLE / DYNAMIC_K_LIMITED_DIAGNOSTIC`。不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；`MR-12B / DL-CONT12B`仍維持current model research anchor，`MR-12C`仍維持REJECTED。

### 本輪基準與唯一診斷變化

- 最新程式ZIP：`test-branch-1_20260809_031311_2ee25ae.zip`；SHA256：`a140e71c420f05b14d61118c764c543012c763096ae0b39403ff8b0a6ec87c1b`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_031311`；開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。
- 使用者本機以score-event-date修正版重新執行`apps/research.py → 模型訓練 → 比較設定中的 Continuous Rankers`。
- Fixed K=10 Selection／OOS數字與前輪完全相同，表示score-event-date修正只影響Dynamic-K join路徑。
- Dynamic-K仍為C17 max-DL eligible `223`日，其中三模型score+target全候選共同完整日`94`、非共同完整日`129`、day coverage `42.15%`；K=`1～8`、mean=`1.34`。
- `沿用runtime score date列=17,744`，但`score date != signal date列=0`。因此上一輪「continuation/re-entry score-event-date錯接造成42.15% coverage」假設被實跑否定；不得再把該假設當作Dynamic-K弱勢解釋。

### 仍成立的Fixed-K OOS證據

MR-12B相對MR-12A在603個共同OOS competition days：NDCG `+0.0290`、Top-K Lift `+0.1601R`、Oracle overlap `+2.60pp`、Boundary `+3.02pp`、Boundary gap `+0.2610R`。此結果與C19-C17及C20-C18 controlled strategy replay的模型經濟改善方向一致。

### Dynamic-K common-complete raw-score結果

在94個C17-reference common-complete competition days：

- MR-12A：NDCG `0.5819`、Top-K Lift `0.5308R`、Boundary `56.47%`、Boundary gap `+0.1133R`。
- MR-12B：NDCG `0.5162`、Top-K Lift `0.1752R`、Boundary `43.94%`、Boundary gap `-0.1365R`。
- MR-12B − MR-12A：NDCG `-0.0657`、Top-K Lift `-0.3556R`、Boundary `-12.53pp`、Boundary gap `-0.2498R`。
- MR-12C同樣弱於MR-12A；相對MR-12B的Boundary僅`+2.51pp`，不足以推翻正式C21/C22已淘汰ListNet的經濟證據。

### 新確認的診斷語意限制

程式核對確認，P2 Dynamic-K目前不是「實際C17 selector basket品質」：

1. P2只有當日**全部orderable candidates × 全部比較模型的score與target都完整**時才保留整天；正式`continuous-score-max-dl`只要求當天至少一個candidate有continuous score即可進DL selection，缺score candidate會排在已評分候選之後，因此`42.15%`是P2 complete-case day coverage，不是runtime DL可用率。
2. P2對保留日直接依各模型raw score做Top-K／boundary；正式C17還會以Min ROOS固定`K/R0`後做exact-reservation minimum-repair，C18再做feasible-ascent single swaps。因此P2沒有量到resource-feasible最終basket。
3. Dynamic-K的候選集合、K與portfolio state固定取C17 reference replay；C19/C20正式策略各自重播後會因先前選股、持倉、現金與continuation path不同而形成不同後續state。P2是同一reference path上的controlled static diagnostic，不等同各模型native strategy path。

所以目前能下的結論是：**MR-12B在Fixed-K broad OOS ranking較MR-12A好，但在C17-reference、低K、common-complete raw-score子集上較弱；尚不能判定弱點是K=1/2極前段排序、partial-score日、resource repair，或path-dependent selector interaction。**

### 本輪P2診斷補強

不修改模型／score／selector／strategy replay，只補read-only P2：

- Dynamic-K標題改為`reference path / common-complete raw-score diagnostic`，禁止再標示成「實際selector決策邊界」。
- coverage新增common-complete candidate row rate、reference runtime scored-candidate rate、各模型candidate score coverage與runtime unavailable reason counts。
- 新增依Dynamic K分層的paired表，直接輸出K=1/2/3/...的NDCG、Boundary與Top-K Lift，先判斷MR-12B弱勢是否集中於極低K。
- App簡易摘要同步標示`Dynamic raw-score Boundary`，避免和C17/C18正式selector action品質混淆。

### 下一步

1. 先原樣重跑P2取得新增candidate-level coverage與K-stratified表，不重訓、不重跑Strategy Compare。
2. 若MR-12B弱勢主要集中K=1/2，進一步做top-prefix decision attribution；若各K都弱，再查complete-case selection bias與score-unavailable候選。
3. 無論K-stratified結果如何，下一個真正對應經濟結果的診斷應以C17/C18各arm的**盤前planned basket / repair / feasible-swap action**為單位，而不是再用generic raw-score Top-K代替selector。若現有持久工件不足以重建planned basket，應先補保存selector action membership/rank的canonical diagnostic欄位，再由既有replay產物或下一次正式replay產生，不得用selected fills冒充盤前planned orders。
4. 在上述歸因完成前，不建立MR-12D、不依94日Dynamic-K子集改Pairwise loss。


## 2026-08-09 — P2 Dynamic-K K分層console renderer API閉環

狀態：`IMPLEMENTED / LOCAL_RERUN_REQUIRED`。此為P2 read-only診斷輸出bug fix，不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity，不修改Dataset、Label、模型權重、score、selector或strategy replay。

### 程式基準與本機失敗

- 輸入ZIP：`test-branch-1_20260809_034619_538614e.zip`；SHA256：`805cf0b8c7bf55a81c13145413ff51d2b84b8132221232700f75b385285f8a0d`。
- 使用者由`apps/research.py → 模型訓練 → 比較設定中的 Continuous Rankers`執行P2時，計算已完成至console rendering，但新增`_render_dynamic_k_strata_table()`呼叫共用`core.console_report.render_table()`時誤用私有舊renderer的keyword `aligns=`，造成`TypeError: render_table() got an unexpected keyword argument 'aligns'`，因此本輪沒有產生可採用的新K-stratified結果。

### 根因與修正

- `core.console_report.render_table()`正式keyword為`alignments=`；`aligns=`只屬`tools/filters/breakout_quality/report.py`內部`_render_ascii_table()`，兩者API不同。
- 將P2 K分層renderer改為`alignments=`，不改任何metric計算、candidate coverage、Dynamic-K sample或既有工件語意。
- 在既有continuous-ranker synthetic contract中直接呼叫`_render_dynamic_k_strata_table()`並驗證標題欄位，避免未來只測metric payload而漏掉console renderer的API mismatch。

### 下一步

原樣重跑正式選單P2即可；不需重訓模型、不需重跑Strategy Compare。取得K=1/2/3/...分層與candidate coverage後，再依前節既定順序判斷top-prefix弱點或complete-case selection bias；在歸因完成前仍不建立MR-12D。


## 2026-08-09 — P2 Dynamic-K K分層結果：弱勢集中K=1；新增完整OOS prefix sweep

### 狀態

`P2_K_STRATIFIED_RESULT_AVAILABLE / FIXED_OOS_PREFIX_SWEEP_IMPLEMENTED`。此為read-only模型品質診斷，不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；`MR-12B / DL-CONT12B`維持current model research anchor，`MR-12C`維持REJECTED。歸因完成前不修改Pairwise loss、不建立新MR。

### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_035134_5989e50.zip`。
- SHA256：`54beca984605fee7618a80098e2a0b0b2a5ee1f0773bfe4d8062609616d0260d`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_035134`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### 使用者本機P2結果

Fixed K=10 OOS維持603個competition days；MR-12B相對MR-12A：NDCG `+0.0290`、Top-K Lift `+0.1601R`、Oracle overlap `+2.60pp`、Boundary `+3.02pp`、Boundary gap `+0.2610R`，broad Top-10 forward-OOS優勢仍成立。

Dynamic-K reference仍為C17：eligible `223`日、common-complete `94`日、day coverage `42.15%`；但common-complete candidate coverage與C17 runtime scored-candidate coverage均為`97.52%`。因此day coverage低不是大量候選缺分，而是少量缺分散落在多個日期造成strict all-candidate complete-day淘汰。

K分布顯示C17 reference path大多數決策是極低K：eligible `K=1:179, K=2:30, K=3:8, K=4:3, K=7:1, K=8:2`，即K=1佔`179/223 = 80.27%`；common-complete中K=1亦佔`71/94 = 75.53%`。

K=1的MR-12B弱勢明確：NDCG `0.4997 vs 0.5883`（B-A `-0.0886`）、Boundary `38.03% vs 57.75%`（`-19.72pp`）、Top-1 raw-target lift `0.0425R vs 0.4903R`（`-0.4478R`）。因K=1時boundary width自動縮成1，Boundary即比較score rank #1與#2的raw target，故`38.03%`代表MR-12B排第一的候選其target高於第二名的比例低於50%。

K=2與K=3的NDCG／Boundary反而偏向MR-12B：K=2 NDCG Δ `+0.0588`、Boundary Δ `+15.38pp`；K=3 NDCG Δ `+0.0089`、Boundary Δ `+4.44pp`。但樣本僅13日與5日，不能作採用結論。K>=4樣本更少，不具判定力。故目前Dynamic-K aggregate B<A主要由K=1主導，不是已證明所有Dynamic-K皆較差。

### 下一個可執行診斷

為區分「MR-12B一般性的Top-1弱點」與「只在C17 reference path／其日期子集出現」，P2新增config-driven Forward-OOS Fixed-K prefix sweep。預設K=`1,2,3,5,10`，全部使用完整paired OOS event-group候選宇宙，只改K；不重訓模型、不strategy replay、不使用selector state。若完整OOS K=1仍B<A，才支持top-prefix objective mismatch假說；若完整OOS K=1為B>=A，則弱點較可能來自C17 path／regime／candidate subset，下一步應做native selector/action attribution而不是改loss。

### 實作

1. `config/breakout_quality.py`新增`BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_FIXED_K_VALUES`，並由comparison settings驗證正整數、唯一、遞增；值屬使用者config，不由validator硬編唯一合法答案。
2. `compare_continuous_rankers.py`新增`_evaluate_fixed_k_sweep()`，在同一完整OOS paired frame上逐K呼叫canonical `_evaluate_paired_frame()`；JSON schema升為2，新增`fixed_k_prefix_sweep`，console／Markdown新增Forward-OOS prefix表。
3. `application.py`短版簡易報表新增最小configured K的OOS NDCG Δ與Boundary Δ，直接讀canonical P2 JSON，不重算第二套指標。
4. 既有Pairwise synthetic contract加入isolated config override與fixed-K sweep直接行為檢查。

### 科學約束

此prefix sweep只作已查看OOS上的迭代研究歸因，符合`PROJECT_SETTINGS E7`；不得進loss、gradient、epoch selection、normalization、sample weighting或hyperparameter optimization。只有先確認弱點的範圍與selector interaction後，才允許提出新的model experiment。


## 2026-08-09 — P2完整Forward-OOS prefix sweep：一般Top-1弱點排除；reference-subset attribution實作

### 狀態

`P2_PREFIX_SWEEP_RESULT_AVAILABLE / REFERENCE_SUBSET_ATTRIBUTION_IMPLEMENTED`。此為read-only模型品質歸因，不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；`MR-12B / DL-CONT12B`維持current model research anchor，`MR-12C`維持REJECTED。

### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_040310_a2c253e.zip`。
- SHA256：`f07c982725c40f517e8b76986544f8ab4f24c51bcba87b3ce5f1723925c11552`。
- GPT fresh extract：`/mnt/data/stock_review_040310`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### 使用者本機P2結果

完整Forward-OOS共同候選只改K的prefix sweep：

- K=1：Days=`1133`；MR-12B vs MR-12A NDCG `0.5779 vs 0.5302`（Δ `+0.0477`）、Boundary `52.69% vs 47.48%`（Δ `+5.21pp`）、Top-1 Lift `0.8244R vs 0.5069R`（Δ `+0.3175R`）。
- K=2：NDCG Δ `+0.0327`、Boundary Δ `-0.14pp`、Lift `0.6915R vs 0.4404R`。
- K=3：NDCG Δ `+0.0309`、Boundary Δ `+2.27pp`、Lift `0.6168R vs 0.4124R`。
- K=5：NDCG Δ `+0.0281`、Boundary Δ `-0.09pp`、Lift `0.5233R vs 0.3567R`。
- K=10：NDCG Δ `+0.0290`、Boundary Δ `+3.02pp`、Lift `0.4089R vs 0.2488R`。

因此MR-12B在完整Forward-OOS從Top-1到Top-10的NDCG與raw-target lift皆優於MR-12A，K=1本身亦有明顯Boundary優勢；上一輪「Pairwise可能改善broad ranking但犧牲extreme Top-1」假說被本輪完整OOS結果否定，不得據此設計top-1-aware loss。

### 與C17-reference K=1子集的矛盾

C17 reference path仍有eligible K=1 `179/223`日；common-complete K=1為`71/94`日。該71日orderable raw-score診斷中MR-12B相對MR-12A NDCG `-0.0886`、Boundary `-19.72pp`、Top-1 Lift `-0.4478R`，方向與完整OOS K=1相反。candidate-level score coverage仍為`97.52%`，故此矛盾不能由「一般Top-1能力不足」或大量缺分直接解釋。

目前最小可辨識的兩個來源為：

1. **reference-date / regime conditioning**：C17 max-DL eligible、尤其K=1日期本身是否是一群MR-12B相對弱勢的市場／訊號日期。
2. **orderable candidate / portfolio-state conditioning**：即使固定同一天，經正式策略候選形成、持倉／現金／continuation狀態後留下的orderable universe，是否把MR-12B優勢反轉。

### 本輪新增read-only attribution

P2新增`reference_subset_attribution`，對每個實際Dynamic-K值固定**同一批common-complete reference dates與同一個K**，同時計算：

- `event_universe`：完整Forward-OOS canonical same-day event-group共同候選；
- `orderable_universe`：既有reference arm盤前orderable candidates。

console／Markdown直接列出MR-12B−MR-12A的NDCG、Boundary與Top-K Lift差異。判讀契約：若同一批K=1 reference dates的Event Δ仍為正，但Orderable Δ轉負，弱勢主要來自candidate/path conditioning；若Event Δ也已轉負，則reference-date/regime conditioning已足以解釋至少部分反轉。此診斷仍不等同resource-feasible最終basket，不重訓模型、不重跑Strategy Compare、不修改selector。

### 下一步

1. 原樣執行`apps/research.py → 模型訓練 → 比較設定中的 Continuous Rankers`取得新的Reference-path子集歸因表。
2. 若K=1 Event Δ為正而Orderable Δ為負，下一步補C17/C19與C18/C20盤前planned basket／repair／feasible-ascent action attribution；現有持久工件若不足，先在canonical replay diagnostics保存planned action rank/membership，再由正式Strategy Compare產生。
3. 若K=1 Event Δ已為負，先做reference dates的year/regime／candidate-set特徵歸因，再決定是否需要PIT fold stability；不得直接修改loss。
4. 在上述歸因完成前，不建立MR-12D。

## 2026-08-09 — P2 Reference-subset attribution結果：反轉由Orderable universe引入；score-date comparability診斷實作

### 狀態

`P2_REFERENCE_SUBSET_RESULT_AVAILABLE / SCORE_DATE_COMPARABILITY_IMPLEMENTED`。read-only模型品質歸因，不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；`MR-12B / DL-CONT12B`維持current model research anchor，`MR-12C`維持REJECTED。

### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_041427_90499bb.zip`。
- SHA256：`ddf8d4d2666b16871ad2c00c794861b7b4511230683c58570104ba9248cd6304`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_041427`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### 使用者本機Reference-subset attribution結果

對C17 common-complete K=1的同一71個reference dates固定K=1：

- 完整Forward-OOS `event_universe`：有效competition days=`64`；MR-12B−MR-12A NDCG Δ=`+0.0532`、Boundary Δ=`+4.69pp`、Top-1 Lift Δ=`-0.0052R`。
- C17 `orderable_universe`：competition days=`71`；NDCG Δ=`-0.0886`、Boundary Δ=`-19.72pp`、Top-1 Lift Δ=`-0.4478R`。

因此「reference-date / regime conditioning本身使MR-12B Top-1變差」不成立：同一批日期回到canonical same-day event universe後，MR-12B的NDCG與Boundary仍優於MR-12A；負向反轉是在正式C17 orderable candidate universe形成後才出現。K=2 Event與Orderable NDCG Δ皆為正（`+0.0797/+0.0588`），其餘K樣本過少不作採用判定。

### 新的最小機制假說

MR-12B training objective為**within-day RankNet**：pair只在相同event date內建立，loss只約束同日margin ordering。正式C17 orderable pool則可在同一trade date同時包含新訊號與延續中的歷史訊號；這些候選沿用各自原始`breakout_quality_score_date`的frozen score，因此runtime會比較不同`score_event_date` cohort的score。

這形成可直接驗證的objective/runtime mismatch假說：

- `same score-event-date pairs`接近MR-12B實際training pair scope；
- `cross score-event-date pairs`從未被MR-12B pairwise loss直接約束其absolute score scale；
- MR-12A的daily-percentile MSE雖也是same-day target，但逐row MSE對[0,1] percentile提供absolute score anchor，理論上較容易保留跨date cohort可比性。

此機制目前僅為待驗證假說；不得在取得pair-scope診斷前建立MR-12D或修改loss。

### 本輪新增read-only診斷

`compare-continuous-rankers`新增`score_event_date_comparability`：

1. 對common-complete C17 orderable rows計算`trade_date - score_event_date`，輸出score-age>0候選比例、age bucket、mixed-score-date trade days，以及K=1 mixed-score-date比例。
2. 在每個trade date內以`target_raw_r`作方向真值，分別計算：
   - all pairs；
   - same-score-event-date pairs；
   - cross-score-event-date pairs；
   - K=1 same-score-event-date pairs；
   - K=1 cross-score-event-date pairs。
3. 每個scope同時輸出pair-weighted concordance與mean-daily concordance，並直接比較config summary pair（目前MR-12B−MR-12A）。
4. 此診斷只讀既有frozen score與C17 orderable artifacts；不重訓、不strategy replay、不修改selector，不把OOS資料送回loss／gradient／epoch selection。

### 判讀規則與下一步

- 若MR-12B在same-score-date pairs仍優於／不弱於MR-12A，但cross-score-date pairs明顯轉負，則支持「跨score-date cohort可比性」是主要runtime落差；下一個model research才可考慮以Selection-only資料做`within-day pairwise + absolute percentile anchor`之單一objective變更。
- 若same-score-date與cross-score-date皆轉負，則orderable universe還有其他candidate-lifecycle／portfolio-state conditioning，下一步拆score age、candidate type、fresh/continuation與C17/C19 native planned basket action。
- 在上述診斷完成前，不建立MR-12D、不修改C17/C18。

## 2026-08-09 — P2 score-date comparability結果：跨date尺度失配排除；orderable event-target語意邊界確認

### 狀態

`P2_SCORE_DATE_COMPARABILITY_RESULT_AVAILABLE / CROSS_DATE_MISMATCH_REJECTED`。此為read-only模型品質／候選語意歸因，不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；`MR-12B / DL-CONT12B`維持current model research anchor，`MR-12C`維持REJECTED。不得依本輪orderable原始event-target診斷修改Pairwise loss。

### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_043045_8dbc562.zip`。
- SHA256：`2e4383cb7eef5f9a83e27b18a891476df63291e47f51e037ff674e7e4e02861e`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_043045`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### 使用者本機結果

在C17 reference common-complete orderable universe：

- score-age>0候選=`3246 / 3246 = 100%`；score age mean=`20.22`天、median=`14`天、max=`288`天。
- mixed score-date日=`92/94 = 97.87%`；K=1 mixed score-date日=`70/71 = 98.59%`。
- All pairs：MR-12B Pair=`54.89%` vs MR-12A=`52.50%`，Δ=`+2.39pp`；mean-daily Δ=`+1.38pp`。
- Same score-date：Pair Δ=`+0.67pp`；mean-daily Δ=`+2.79pp`。
- Cross score-date：Pair Δ=`+2.49pp`；mean-daily Δ=`+1.38pp`。
- K=1 same score-date：Pair Δ=`+0.45pp`；mean-daily Δ=`+2.46pp`。
- K=1 cross score-date：Pair Δ=`+2.14pp`；mean-daily Δ=`+1.48pp`。

因此上一輪「MR-12B within-day RankNet缺乏跨score-event-date absolute comparability，導致C17 orderable K=1反轉」假說被直接否定：MR-12B在same-date與cross-date pair ordering都不弱於MR-12A，且cross-date優勢更大。

### 新確認的語意邊界

P2 Dynamic-K / score-date comparability目前使用的方向真值仍是每個候選**原始score-event-date**對應的`strategy_aligned_opportunity_no_time_r_v1 target_raw_r`。該Target由原始事件日後固定future horizon建立；當候選在較晚trade date仍orderable時，P2並沒有重新錨定到當下trade date，也沒有重建剩餘可實現strategy R。

本輪orderable候選全部`score_age>0`，且median 14天、mean 20.22天，故：

1. P2仍可回答「frozen score對原始事件Target ordering保留多少」；
2. 但Dynamic-K Top-K Lift／Boundary不得解讀成「在當下trade date買入後的counterfactual realized R」；
3. C17 K=1的Event `B>A`、Orderable原始event-target `B<A`與正式C19/C20經濟結果`B>A`並不形成模型採用矛盾；目前缺的是trade-date action outcome attribution，而不是更多原始event-target切片。

### 本輪報表契約修正

- P2 JSON schema提升為5；Dynamic-K與score-date comparability明確保存`target_anchor=original_score_event_date_target`與`trade_date_remaining_opportunity_evaluated=false`。
- console／Markdown新增score-age bucket顯示，並明確註記原始event-target不是trade-date remaining opportunity / counterfactual realized R。
- 不修改任何score、Target建置、模型、selector、Strategy Compare、accounting或execution。

### 下一步

1. **模型研究主線回到MR-12B Selection PIT**：由正式選單建立／更新Selection PIT Scores並做PIT模型驗證，確認Pairwise ranking優勢是否跨歷史fold穩定；PIT只使用其合法Selection／Validation時間窗，不讀Forward-OOS作訓練或選模。
2. P2到此不再因C17 K=1原始event-target反轉設計新loss；MR-12D暫不建立。
3. 若仍要解釋C17/C19、C18/C20的portfolio action差異，下一個診斷必須以**orderable trade date的canonical counterfactual／realized strategy outcome**為真值，並直接對planned basket／repair／feasible-ascent action做歸因；不得再用原始score-event-date Target冒充當下剩餘機會。



## 2026-08-09 — MR-12B Selection PIT模型驗證：10/10年度rho為正，Gate PASS

### 狀態

`MR-12B / DL-CONT12B`更新為`PIT_MODEL_VALIDATION_PASS / MODEL_ECONOMIC_IMPROVEMENT_SUPPORTED`。本輪只取得既有MR-12B的Selection point-in-time模型驗證結果，不建立新`MR-*`、`DL-*`、`SR-C*`或`AUD-*` identity。

### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_045404_66521e1.zip`。
- SHA256：`710539a6fc783608572c753a050e2a83237d6500919fbebfddd1e63690abff5a`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_045404_710539a6`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### 使用者本機PIT結果

- Profile：`strategy_aligned_no_time_all_event_pairwise`；objective=`daily_pairwise_ranking`；scope=`all_labels`；seed=42。
- PIT period：`2011-01-01～2020-12-31`；12-month score folds／24-month inner validation；folds=`10`。
- PIT scores：`23,932 / 23,932` groups，coverage=`100.00%`；本輪10 folds全部重用既有合法工件，未重訓。
- PASS-only target ordering：global rho=`0.2336`；mean daily rho=`0.1859`；top-bottom spread=`1.1386R`。
- 年度穩定：rho>0=`10/10`；spread>0=`9/10`。
- 分類重疊：AUC=`0.5670`；Top decile PASS=`63.07%`；Overall PASS=`55.46%`。
- Gate=`PASS`。Gate固定只使用PASS-only global/daily Spearman與多數年度rho/spread方向，不使用策略績效。
- `drift=True`：audit定義為至少一組相鄰fold的score平均值位移`>=1.0 pooled score SD`。此旗標描述score level／calibration drift，不代表年度ranking方向失敗，也不是目前predeclared Gate veto。
- Orderable coverage尚未提供，依正式語意於後續Selection strategy replay建立。

### 判定

MR-12B的Selection PIT模型層證據通過：10個年度的PASS-only rho全部為正、9/10年度top-bottom spread為正，且全期global／daily rho均為正。這補上Forward-OOS與C19/C20 controlled replay之外的歷史時間穩定性證據。`drift=True`需在策略層留意跨fold score level，但目前沒有證據支持因該warning修改Pairwise loss或拒絕MR-12B。 此audit只評估MR-12B自身是否具有正向且跨年穩定的PIT排序能力，**沒有與MR-12A做same-fold paired comparison**，因此不可把Gate PASS寫成「MR-12B在歷史每個fold都優於MR-12A」。若要補齊learning-objective歷史比較，應以相同PIT日期／候選做MR-12B−MR-12A paired PIT read-only comparison。

### 下一步

1. 進入正式Selection PIT策略績效驗證，讓歷史每個交易日只使用當時合法PIT score，並建立orderable score coverage；比較時維持既有策略參數／selector contract，禁止依PIT結果回頭修改loss、epoch或score normalization。
2. 策略層優先看Return、MDD、RoMD、EV、Exposure／資金利用率與年度穩定，確認PIT模型排序能力是否能轉成無前視portfolio economics。
3. 若策略結果因fold邊界或跨foldscore level出現異常，再做read-only cross-fold/action attribution；`drift=True`本身不足以建立新model experiment。

## 2026-08-09 — Selection PIT 3-arm策略經濟驗證入口實作（SR-C23／C24／C25）

### 狀態

`IMPLEMENTED / STRATEGY_RESULT_NOT_AVAILABLE`。本輪只建立正式config-driven Strategy Compare入口與工件契約，不執行strategy replay、不重訓MR-12B、不重新最佳化策略參數。

### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_050516_cc11b79.zip`。
- SHA256：`7adc1e247972afa409c391b1e46f830c016d7ff692845b0ef1055acaf5fcb8b0`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_050516`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### 固定研究設計

- 比較期間固定`2014-01-01～2020-12-31`。
- 共用策略參數來源：歷史A2 teacher P2 Min ROOS active params：`models/research/breakout_quality/trade_path_label/a2_teacher_params/p2_dl_off_trained/active_params/roos_base_best.json`。此工件由Selection nested baseline衍生，rules固定all-off、training DL off、只搜尋既定risk fields；Strategy Compare僅重用，不自動執行rolling optimizer。
- 參數工件額外驗證`breakout_quality_param_adaptation.mode=risk_only_training`、`parameter_set=P2_HISTORY`、`fixed_rule_contract=all_rule_filters_off`、`training_dl_enabled=false`，避免誤接forward 2021～2026 P2或其他參數來源。
- 新runtime source：`DL-CONT12B-PIT`（config alias=`CONT12B_PIT`），identity固定MR-12B／InceptionTime／`strategy_aligned_no_time_all_event_pairwise`；score source=`selection_point_in_time`。前置檢查要求Selection PIT score／manifest／audit完整且model Gate=`PASS`；Strategy Compare不得build/rebuild PIT model或score。
- `SR-C23`：historical P2 Min ROOS、rules all-off、DL off baseline。
- `SR-C24`：與C23相同params，使用MR-12B PIT score，完全沿用C17 `resource-aware-continuous-max-dl` minimum-repair selector。
- `SR-C25`：與C23相同params，使用MR-12B PIT score，完全沿用C18 `resource-aware-continuous-max-dl-feasible-ascent` selector。
- 正式 contrasts：`C24-C23`、`C25-C23`、`C25-C24`。
- 不改max positions=10、rotation=off、param policy=`base-finalist-best`、accounting、execution、candidate lifecycle或selector語意。

### 實作邊界

1. `core.strategy_comparison`正式允許read-only `selection_point_in_time` DL source，且禁止binary threshold與forward-score builder。
2. `strategy_compare_preparation`直接使用`load_selection_point_in_time_ranking_contract`驗證PIT工件與Gate，並把score／manifest／audit SHA綁入artifact identity；缺工件時BLOCK，不自動訓練。
3. Strategy pair cache對`selection_point_in_time` source額外納入PIT audit identity；既有canonical／continuous-OOS sources的fingerprint欄位與schema維持不變，避免無關地失效歷史cache，同時防止PIT audit變更後誤重用舊結果。
4. Engine既有Selection PIT lookup／post-replay target diagnostics沿用；PIT continuous source在metadata中threshold固定為None。
5. Config切換為C23／C24／C25三arm；舊C17～C22保留Registry／config歷史重現但本輪disabled。

### 下一步

使用正式選單`apps/research.py → [3] 策略組合比較 → [2] 查看設定、工件與預計動作`。先確認期間2014～2020、arms=C23/C24/C25、PIT工件與historical P2 params皆READY；若READY，再`[1/Enter] 執行目前比較設定`。結果優先比較Return、MDD、RoMD、EV、Exposure／資金利用率、orderable PIT-score coverage及年度穩定性；不得依此結果回頭調MR-12B loss／epoch／score normalization。



## 2026-08-09 — MR-12B Selection PIT C23/C24/C25策略經濟驗證結果

### 狀態

`RESULT_AVAILABLE / PIT_MODEL_GATE_PASS / PIT_STRATEGY_NOT_PROMOTED / REALIZATION_ATTRIBUTION_NEXT`

### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_055527_b0e14c8.zip`。
- SHA256：`0b3702630954ad39a254de2805f30b5cb6239c7df43f500630824e93bdda88ba`。
- GPT fresh extract：`/mnt/data/stock_review_055527`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### 固定條件

2014-01-01～2020-12-31；historical P2 Min ROOS active params；`base_finalist_best`；max positions=10；rotation=off；optional entry filters=all-off；hard filter off；Selection PIT source=`DL-CONT12B-PIT`；Dataset、Continuous Target、PIT folds／scores／audit、entry／stop／exit、accounting與portfolio規則固定。C23為DL-off baseline；C24只加入MR-12B PIT + C17 minimum-repair；C25只加入同一PIT source + C18 feasible-ascent。Future Target只於replay完成後join，不進runtime。

### 主要結果

| Arm | Return | MDD | RoMD | Annual | EV | Exposure | Trades | same-param DL selection R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C23 | 127.45% | 25.45% | 5.01 | 12.46% | 0.62R | 87.68% | 379 | 0.00R |
| C24 | 108.05% | 24.51% | 4.41 | 11.04% | 0.53R | 87.40% | 374 | -35.78R |
| C25 | 115.42% | 26.36% | 4.38 | 11.59% | 0.59R | 87.65% | 366 | -19.52R |

Controlled deltas：

- C24-C23：Return `-19.40pp`、MDD `-0.94pp`、RoMD `-0.60`、Annual `-1.42pp`、EV `-0.09R`、Exposure `-0.29pp`、selection R `-35.78R`。
- C25-C23：Return `-12.03pp`、MDD `+0.91pp`、RoMD `-0.63`、Annual `-0.87pp`、EV `-0.03R`、Exposure `-0.03pp`、selection R `-19.52R`。
- C25-C24：Return `+7.37pp`、MDD `+1.85pp`、RoMD `-0.03`、Annual `+0.55pp`、EV `+0.06R`、Exposure `+0.25pp`、selection R `+16.26R`。Feasible-ascent改善minimum-repair的經濟轉化，但仍不足以超越C23。

### Selection／resource診斷

- C24相對C23：selected Target percentile `+0.0164`、Target mean `+0.0476R`、opportunity gap改善`0.1373R`，但selection R `-35.78R`、期末未滿倉日`+318`、持股缺口`+297格日`。
- C25相對C23：selected Target percentile `+0.0178`、Target mean `+0.0306R`、opportunity gap改善`0.0607R`，但selection R `-19.52R`、期末未滿倉日`+259`、持股缺口`+274格日`。
- Candidate supply幾乎不變：平均每日可掛單候選C23/C24/C25=`65.33/63.25/65.10`，供給不足日=`101/103/102`；因此策略惡化不能主要歸因於候選供給不足。
- C24/C25平均曝險仍接近C23，代表「更多未滿倉日」不等於總資金曝險同比例下降；需要拆解position sizing、fill、holding/partial-tail slot occupancy與capital deployment。
- C24的Future Target mean／opportunity gap優於C25，但C25的Return與selection R反而較好；這是直接證據顯示`strategy_aligned_opportunity_no_time_r_v1`的原始event Target改善不等於portfolio可實現R改善。

### 判定

1. MR-12B的Selection PIT模型Gate仍維持`PASS`；本次不否定Pairwise模型研究結論。
2. `DL-CONT12B-PIT`直接套用historical P2 Min ROOS的C24/C25皆`NOT_ADOPTED`；不得升格正式策略source。
3. C25相對C24的改善證明selector搜尋完整度有影響，但不是根因；較Max selector仍無法使PIT source超越baseline。
4. 本次最強訊號為Target→realized economics realization gap：post-replay Future Target改善與same-param selection R／總報酬方向相反。不得直接依Selection結果修改MR-12B loss、architecture或OOS selector。

### 下一步

下一個最高資訊量工作是read-only **PIT strategy realization/capture attribution**，直接重用C23/C24/C25既有replay工件，不重跑portfolio、不重訓模型、不跑optimizer。至少拆解：exclusive trade realized R／PnL、planned/reserved→fill、position sizing與stop distance、holding與partial-tail slot-days、capital deployment、Target capture ratio／realization gap、以及年度貢獻；優先比較C24-C23與C25-C23，再用C25-C24隔離selector轉化。只有歸因確認有明確可由既有策略參數空間修正的機械瓶頸，才考慮Selection內參數適應；否則維持C23 baseline並把PIT ranking的直接策略部署淘汰。

## 2026-08-09 — AUD-c23-c25-pit-realization 實作：Selection PIT策略實現／Capture只讀歸因

### 狀態

`IMPLEMENTED / RESULT_PENDING / READ_ONLY_EXISTING_REPLAY_ONLY`

### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_063143_d77bb26.zip`。
- SHA256：`bd5fef06ca9e26a984187820b0d7f91ca78d9a9f0bb5afad46a5824ea78f56d5`。
- GPT fresh extract：`/mnt/data/stock_review_063143`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### Audit identity／固定來源

- 新增`AUD-c23-c25-pit-realization`（config id=`c23-c25-pit-realization`），不建立新`MR-*`、`DL-*`、`SR-C*`或參數stage。
- 正式來源由`config/audit.py`驅動：baseline=`SR-C23`；candidate arms=`SR-C24 / SR-C25`；預設讀取最新已完成Strategy Compare run。handler不硬編arm ID。
- Audit只接受既有completed score-ranking pair工件；需要`trades/equity/daily_capacity/selected_buys/selected_target_diagnostics/strategy_comparison.json`完整存在。缺工件即BLOCKED，不自動replay或重建。

### 實作與單一真理

1. `tools/audit/breakout_quality/strategy_realization_capture.py`只做orchestration與報表；不重寫交易／Target公式。
2. Wealth path、selection difference、trade contribution、capital geometry與slot occupancy共用既有`c15_strategy_attribution` pair primitive；該primitive正式公開為`build_strategy_attribution_pair_payload`。
3. Fill、reserved→invested、stop distance、holding、partial-tail slot-days與Target→Realized capture共用`tools/audit/portfolio/score_ranking_capture.py`既有`build_score_ranking_capture_audit`。Future Target仍只在replay後離線join。
4. 新Audit另外把exclusive trades拆成baseline-only winner／loser與candidate-only winner／loser，並保存每pair詳細CSV；此拆解只來自canonical completed round trips，不估未成交候選R。
5. `tools/audit/sources/strategy_compare.py`補齊已正式存在的`resource-aware-continuous-max-dl`與`resource-aware-continuous-max-dl-feasible-ascent` artifact prefix resolver；不改runtime selector。
6. Formal Audit runner仍由`apps/research.py → Audit／診斷`共用config/catalog backend；本輪把舊`c15-source-attribution`切為OFF、新Audit切為ON。

### 獨立synthetic結果

以隔離completed strategy-pair fixture直接驗證兩個candidate arms：Target mean刻意高於baseline、Realized R刻意較低、exclusive selection R為負且slot gap增加。新Audit status=`READY`，兩pair均辨識`target_to_realized_divergence=True`，並產生`audit.md/.json`與trade/capture lifecycle CSV；metadata固定`portfolio_replay_executed=false`、`training_performed=false`。同一獨立Audit framework synthetic共9項、0失敗。

### 下一步

使用正式選單`apps/research.py → [4] Audit／診斷`先查看設定／工件狀態；READY後執行目前Audit。結果先比較C24-C23與C25-C23的exclusive winner/loser R、fill、平均投入、holding／partial-tail、underfilled slot-days及aggregate Target capture，再用兩者差異判斷是否存在可由既有Selection策略參數空間檢驗的mechanical bottleneck。Audit結果取得前不得進參數適應或建立新MR。



## 2026-08-09 — AUD-c23-c25-pit-realization結果：winner capture不足；修正capture-gap不等於參數機械瓶頸

### 狀態

`RESULT_AVAILABLE / REALIZATION_GAP_CONFIRMED / PARAM_ADAPTATION_NOT_SUPPORTED`。本輪取得使用者本機只讀Audit結果，並修正既有capture decision將「capture惡化」本身誤當成可由策略參數適應檢驗之機械瓶頸的過度判定。未重跑portfolio、未重訓模型、未跑optimizer、未建立新`MR-*`／`SR-C*`。

### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_065116_e3092c2.zip`。
- SHA256：`c714add288f842d6b7383f5a577bf38220785beffc55e841af69bebf3c269729`。
- GPT fresh extract：`/mnt/data/stock_review_065116`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### 使用者本機Audit結果

#### C24 vs C23

- Return `127.45% → 108.05%`（`-19.40pp`）；RoMD `5.01 → 4.41`；EV `0.62R → 0.53R`；exclusive selection R=`-35.78R`。
- 平均Target R `0.46R → 0.50R`（`+0.04R`），但平均Realized R `0.62R → 0.53R`（`-0.09R`）；aggregate capture `1.44 → 1.03`（`-0.41`）。
- Fill `-0.80pp`、平均投入`-3,832`（約`-2.46%`）、投入／預留比`-0.58pp`、平均持有`-0.28日`、平均曝險`-0.29pp`。這些均未跨既定mechanical-gap門檻（fill `-2pp`、sizing `-10%`、holding `+3日`、exposure `-5pp`）。
- Exclusive trades：C23-only winners `69 / +203.30R`、losers `88 / -73.68R`；C24-only winners `57 / +173.78R`、losers `95 / -79.94R`。
- R分解：winner contribution約`-29.52R`；loser contribution約`-6.26R`；合計`-35.78R`。主要driver=`winner_capture`。Exclusive decisive win rate約`43.95% → 37.50%`（`-6.45pp`）。

#### C25 vs C23

- Return `127.45% → 115.42%`（`-12.03pp`）；RoMD `5.01 → 4.38`；EV `0.62R → 0.59R`；exclusive selection R=`-19.52R`。
- 平均Target R `0.46R → 0.52R`（`+0.06R`），但平均Realized R `0.62R → 0.59R`（`-0.03R`）；aggregate capture `1.44 → 1.18`（`-0.26`）。
- Fill `-1.61pp`、平均投入`+8,288`、投入／預留比`-0.57pp`、平均持有`+1.17日`、平均曝險`-0.03pp`；同樣未跨既定mechanical-gap門檻。
- Exclusive trades：C23-only winners `73 / +162.85R`、losers `99 / -86.87R`；C25-only winners `61 / +140.82R`、losers `98 / -84.36R`。
- R分解：winner contribution約`-22.03R`；loser contribution約`+2.51R`（candidate反而少承擔loser R）；合計`-19.52R`。主要driver=`winner_capture`。Exclusive decisive win rate約`42.44% → 38.36%`（`-4.08pp`）。

### 判定修正

舊`score_ranking_capture._decision`把`capture_gap`與fill／sizing／deployment／holding並列，只要Target改善且經濟失敗，capture gap單獨成立就會輸出`ADAPTATION_DIAGNOSTIC_SUPPORTED`。這會把**結果層的Target→Realized mismatch**誤稱為**可由既有策略參數空間修正的機械瓶頸**。本輪改為：

1. `capture_gap`仍保存為`realization_gap_detected`，但不單獨觸發parameter adaptation。
2. 只有預先定義的fill／sizing／deployment／holding mechanical gap至少一項成立，才可輸出`ADAPTATION_DIAGNOSTIC_SUPPORTED`。
3. 若Target改善、經濟失敗、capture gap成立，但沒有mechanical gap，狀態改為`SORT_ONLY_REJECTED_REALIZATION_GAP_NO_MECHANICAL_BOTTLENECK`，`parameter_adaptation_candidate=false`。
4. `AUD-c23-c25-pit-realization`新增exclusive winner/loser R contribution、exclusive win rate與implied selection R閉環；報表明確區分winner capture與loser avoidance。

### 科學判定

- C24/C25的直接PIT部署仍`NOT_ADOPTED`。
- 本Audit不否定MR-12B PIT模型Gate；它證明的是**高Target selection沒有被既有策略路徑轉成較高realized R**。
- 目前損失主要來自「少捕捉winner」，不是loser severity、fill、sizing、holding或平均exposure已被證實惡化到足以構成參數適應mechanical hypothesis。
- 因此**現在不進Selection參數適應**，也不從本Audit直接修改MR-12B loss／threshold／selector。

### 下一步

下一個最高資訊量read-only診斷優先檢查**PIT-specific fold score drift / mixed-fold runtime conditioning**：PIT audit已有`drift=True`，而Strategy replay的orderable候選可跨年度fold保留frozen score。應直接量測trade date是否混用不同PIT fold model scores、fold boundary附近的exclusive winner capture／selection R是否異常，以及負向selection R是否集中mixed-fold日期。這是PIT-specific問題，先於新Target／新MR。若fold conditioning不能解釋loss，再回到Target與realized trade-path語意研究；不得先跑optimizer。

## 2026-08-09 — AUD-c23-c25-pit-fold-runtime 實作：PIT fold drift／mixed-fold winner capture只讀歸因

### 狀態

`IMPLEMENTED / RESULT_PENDING / READ_ONLY_EXISTING_REPLAY_AND_PIT_ONLY`。本輪只新增正式Audit與獨立synthetic contract，不執行portfolio replay、不重訓／校正PIT模型、不跑optimizer。

### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_073059_dca5f98.zip`。
- SHA256：`7eed43b7ae194e0e42f68a87c2a96754f4fe6d5cbb4603823a1e28d3f0a999f3`。
- GPT fresh extract：`/mnt/data/stock_review_073059`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### Audit identity／固定來源

- 新增`AUD-c23-c25-pit-fold-runtime`（config id=`c23-c25-pit-fold-runtime`）；不新增`MR-*`、`DL-*`、`SR-C*`或`PARAM-*`。
- source由`config/audit.py`驅動：baseline與candidate arms均不硬編於handler；目前設定為C23 baseline、C24/C25 candidates，讀最新已完成Strategy Compare。
- candidate arm必須綁定`score_source=selection_point_in_time`；Audit再由Strategy Compare保存的DL identity解析合法PIT score／manifest／audit，沿用正式hash/Gate驗證。
- fold boundary window預設`30` calendar days，為config可調研究顯示參數，不進runtime。

### 實作內容

1. 每個orderable candidate以`breakout_quality_score_date`優先、否則`signal_date`對回PIT score table的`fold_id`；保存score age與PIT score coverage。
2. 每個trade date計算fold count、mixed-fold flag、scored pairs與cross-fold pair share，直接量化runtime是否把不同fold模型分數混在同一候選池。
3. 共用`build_strategy_attribution_pair_payload`的canonical round-trip／exclusive trade結果；該primitive新增只讀`signal_date`欄位，以`entry_date+ticker+signal_date`對回candidate occurrence，不改既有R／PnL口徑。
4. Exclusive selection R另拆成single-fold vs mixed-fold days，以及距PIT fold transition是否落於±config window；同時拆winner R contribution與loser R contribution。
5. 報表同步顯示既有PIT audit的`fold_drift`與fold score分布，但不重算另一套Gate，也不把共現解讀成因果。
6. 若mixed-fold／fold-boundary不能解釋C24/C25 winner capture loss，下一步回到Target／realized outcome語意；不得因本Audit直接校正score或修改MR-12B loss／OOS selector。

### 獨立synthetic

隔離fixture建立F1/F2兩fold、1個mixed-fold trade day與1個single-fold trade day；mixed day固定exclusive selection R=`-3R`（winner contribution=`-2R`），single-fold day=`+2R`，fold boundary ±30天亦固定`-3R`。`validate_breakout_quality_audit_framework_contract_case`直接驗證mixed-fold day count、cross-fold pair share、兩scope R分解與read-only attribution primitive，結果0失敗。

### 下一步

使用正式選單`apps/research.py → [4] Audit／診斷 → [2] 查看Audit設定、工件與預計動作`；READY後`[1/Enter]`執行。先看C24/C25的mixed-fold day比例、cross-fold pair share、mixed vs single exclusive selection R，以及fold-boundary vs outside winner contribution。只有損失明顯集中才考慮Selection-only score-normalization／runtime假說。


## 2026-08-09 — AUD-c23-c25-pit-fold-runtime結果：fold drift／mixed-fold不是PIT直接部署失敗主因

### 狀態

`RESULT_AVAILABLE / FOLD_DRIFT_NOT_PRIMARY_CAUSE / CROSS_FOLD_NORMALIZATION_NOT_SUPPORTED`。本輪取得使用者本機只讀Audit結果；未重跑portfolio、未重訓／校正PIT模型、未跑optimizer。

### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_082042_f0e60a4.zip`。
- SHA256：`33eedd1798f6a4ecfacc85c65405a64dafb9eda8c7c8da7763fbcded281931fb`。
- GPT fresh extract：`/mnt/data/stock_review_082042`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

### 使用者本機Audit結果

#### C24 vs C23

- PIT drift=`True`，max adjacent mean shift=`1.03 pooled SD`，flagged fold=`fold_20170101_20171231`。
- Mixed-fold days=`353/1633 (21.62%)`，cross-fold pair share=`5.12%`，score age mean/median=`26.16/17.00日`。
- Exclusive selection R=`-35.78R`，但mixed-fold=`+13.91R`；single-fold=`-46.05R`；no-orderable=`-3.64R`。
- Fold boundary ±30日內=`+2.17R`；outside=`-37.95R`。
- 因此負R不集中mixed-fold或fold transition，反而主要在正常single-fold／boundary outside。

#### C25 vs C23

- Mixed-fold days=`357/1635 (21.83%)`，cross-fold pair share=`5.55%`，score age mean/median=`26.22/17.00日`。
- Exclusive selection R=`-19.52R`；mixed-fold=`-4.25R`、single-fold=`-10.65R`、no-orderable=`-4.62R`。
- Fold boundary ±30日內=`+4.93R`；outside=`-24.45R`。
- mixed-fold有部分winner capture損失，但無法解釋主要總損失；fold boundary更不是負R集中區。

### 判定

1. PIT audit的`drift=True`保留為score-level warning，但本次實際策略損失不支持其為C24/C25 winner-capture失敗主因。
2. 不建立Selection-only cross-fold normalization／calibration runtime；不得因本Audit修改MR-12B loss、epoch、threshold或selector。
3. 下一步回到Target／realized outcome語意：直接使用已成交exclusive trades，檢查PIT score與原始event Target對realized R的對齊，並按signal→entry age拆分，以區分event Target老化與Target公式本身失配。
4. 既有11J per-candidate counterfactual已正式STOPPED，不能為此重啟；未成交／未選候選realized R仍保持未知。

## 2026-08-09 — AUD-c23-c25-pit-target-realization實作：exclusive trade Target／Score→Realized R與Score-age只讀歸因

### 狀態

`IMPLEMENTED / RESULT_PENDING / READ_ONLY_ACTUAL_TRADES_ONLY`。本輪只新增正式Audit與獨立synthetic contract；不執行portfolio replay、不建立counterfactual、不重訓／校正PIT模型、不跑optimizer。

### Audit identity／固定來源

- 新增`AUD-c23-c25-pit-target-realization`（config id=`c23-c25-pit-target-realization`）；不新增`MR-*`、`DL-*`、`SR-C*`或`PARAM-*`。
- source由`config/audit.py`驅動：baseline／candidate arms均不硬編於handler；目前設定讀C23 baseline與C24/C25 candidates的latest completed Strategy Compare。
- `score_age_quantile_groups=4`為config可調Audit維度，只影響診斷分層，不進runtime。

### 實作內容

1. 共用`build_strategy_attribution_pair_payload`的canonical exclusive trade結果；每筆trade保留`ticker + signal_date + entry_date`與actual `r_multiple`，不重算另一套PnL／R。
2. candidate arm解析validated Selection PIT score／manifest／audit identity；以`ticker + signal_date`對齊PIT `model_score`與`group_index`，再透過validated Continuous Target arrays取得原始`target_raw_r`。
3. Baseline-only與candidate-only actual exclusive trades分別輸出平均Target、平均Realized R、win rate、Score↔Target、Target↔Realized、Score↔Realized、Age↔Realized與Age↔(Target−R) Spearman。
4. 以所有covered exclusive trades的signal→entry calendar age共同做config-driven quantile切分；每bucket同時輸出baseline/candidate Target、Realized、win rate與Selection ΔR。
5. 判讀只允許兩個受控方向：若負Selection R主要集中高age bucket，形成event Target老化／延續候選語意失真假說；若各age bucket皆負，優先視為Target公式與正式trade-path R失配。結果不直接授權新MR、參數適應或OOS修改。
6. 未成交／未選候選沒有counterfactual realized R；Audit metadata固定`counterfactual_performed=false`，不得填0或推論其績效。

### 下一步

使用正式選單`apps/research.py → [4] Audit／診斷 → [2] 查看Audit設定、工件與預計動作`；READY後`[1/Enter]`執行。先比較C24/C25各age quantile的Selection ΔR、Target與Realized方向，再決定後續是研究candidate aging／refresh語意，或回到Target label semantics。

## 2026-08-09 — AUD-c23-c25-pit-target-realization結果：C25負Selection R集中old-signal tail

### 狀態

`RESULT_AVAILABLE / C25_OLD_SIGNAL_TAIL_SUPPORTED / C24_NON_MONOTONIC / NO_MODEL_OR_TARGET_CHANGE`。

### 程式基準

- ZIP：`test-branch-1_20260809_123909_0537dbe.zip`
- SHA256：`d62779ff43005ba1d6eb75f6bb9ab7cddf79bab6156c605c1a0439c79226455c`
- Audit：`AUD-c23-c25-pit-target-realization`
- 期間：2014-01-01～2020-12-31；只讀既有`SR-C23/C24/C25` completed replay、validated Selection PIT score與原始Continuous Target。
- 不重跑portfolio、不建立未成交counterfactual、不重訓／校正模型、不改selector／params／Target。

### 結果

#### C24 vs C23

- Exclusive selection R=`-35.78R`；Target/score coverage=`93.85%`。
- C24-only相對C23-only：平均Target `+0.10R`，平均Realized `-0.38R`，win rate `-9.32pp`。
- Target↔Realized rho：C23-only=`0.34`、C24-only=`0.23`；Score↔Realized rho：`-0.04/-0.15`。
- Age buckets Selection ΔR：Q1=`+31.73R`、Q2=`-50.95R`、Q3=`-9.41R`、Q4=`-21.61R`。
- 負R不是單調隨age增加；因此C24不支持簡單全域age cutoff。

#### C25 vs C23

- Exclusive selection R=`-19.52R`；Target/score coverage=`93.66%`。
- C25-only相對C23-only：平均Target `+0.14R`，平均Realized `-0.11R`，win rate `-4.62pp`。
- Target↔Realized rho：C23-only=`0.36`、C25-only=`0.35`；Score↔Realized rho：`-0.02/-0.06`。
- Age buckets Selection ΔR：Q1=`+5.67R`、Q2=`+13.31R`、Q3=`+2.76R`、Q4=`-40.18R`。
- Covered trades的Q1～Q3合計=`+21.74R`；Q4(age `23～288` calendar days，median=`38.5`)單獨反轉為`-40.18R`。C25-only Q4平均Realized=`0.42R`，C23-only=`1.50R`；Q4 Target則`0.23R vs 0.25R`，已不存在前3個bucket的Target優勢。
- Age↔Realized與Age↔(Target−R) Spearman接近0，表示不是平滑單調age效應，而是尾端／threshold-like現象。

### 判定

1. `MR-12B`與原始Continuous Target目前均**不因本Audit被淘汰或修改**；C25的Target↔Realized仍約`0.35`，不支持「Target全面失效」。
2. `SR-C25` feasible-ascent是下一個runtime研究基礎；C24因age pattern非單調，不作age-guard基礎。
3. 下一個Selection-only受控假說應是**stale-score membership guard**：過舊PIT score不得驅動DL造成basket membership change，但candidate本身不得被拒絕／過期，需完整保留Min ROOS fallback與K/R0資源契約。
4. 若實作固定score-age cutoff，該值只能由本Selection evidence預先固定並在OOS前凍結；OOS結果不得再用來調cutoff。未成交／未選候選仍不得建立counterfactual R。
5. 在這個runtime guard完成Selection驗證前，不建立新`MR-*`、不重做Target label、不進參數適應。

### 下一步

先以`SR-C25`為控制基準實作單一runtime變更：stale-score只禁止DL membership change，不刪候選、不改score、不改feasible-ascent的K/R0 hard feasibility。Selection內與`SR-C23`及`SR-C25`比較Return、RoMD、EV、same-param DL selection R、underfilled slot-days與guard觸發日；若Selection改善再凍結同一規則進Forward-OOS驗證。


## 2026-08-09 — SR-C26實作：Selection PIT feasible-ascent stale-score membership guard

### 狀態

`IMPLEMENTED / RESULT_PENDING / SELECTION_ONLY / CUTOFF_FROZEN_22D / NO_MODEL_OR_TARGET_CHANGE`。

### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_144248_97c69f4.zip`。
- SHA256：`86e8d0293fff2b126cf4b934a3488e5d7d814909e156370c7d448f6f99319d43`。
- GPT fresh extract：`/mnt/data/stock_review_144248`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；Registry確認`SR-C26`尚未占用。
- 本輪不執行`apps/test_suite.py`，只做獨立synthetic／AST／compile／CLI／依賴檢查。

### Selection evidence與固定假說

`AUD-c23-c25-pit-target-realization`顯示SR-C25 actual exclusive trades：Q1～Q3(age 1～22 calendar days) Selection ΔR合計`+21.74R`，Q4(age 23～288日、median 38.5日)單獨`-40.18R`。因此在任何Forward-OOS驗證前預先固定`22 calendar days`為唯一age門檻；此值之後不得依OOS結果改成20／25／30日或做sweep。

### SR-C26唯一scientific change

1. Baseline固定SR-C25：historical P2 Min ROOS active params、rules all-off、`DL-CONT12B-PIT / MR-12B`、C18 feasible-ascent、K/R0、entry／exit／accounting、max positions=10、rotation off全部不變。
2. 新runtime mode：`resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard`。
3. `config/strategy_compare.py`唯一新增runtime option：`stale_score_membership_guard_max_age_days=22`。
4. Score age固定以`trade_date - breakout_quality_score_date` calendar days計算；缺score date才fallback原`signal_date`。有有效score但日期不可稽核時採保守guard；未評分candidate不因本規則被視為stale。
5. Candidate本身**不得刪除、過期或hard reject**。若C25/C17 seed相對Min ROOS的membership變更牽涉stale scored candidate，C26先回到Min ROOS合法seed；之後feasible-ascent只接受不牽涉stale scored candidate、且canonical exact reservation滿足K/R0 hard floor的score-improving single swap。
6. Fresh候選全部存在時，C26必須與C25 membership一致；stale候選仍完整保留於orderable universe與fallback排序。

### Strategy Compare active matrix

- `SR-C23`：Selection Min ROOS PIT baseline（DL off）。
- `SR-C25`：MR-12B PIT feasible-ascent control。
- `SR-C26`：MR-12B PIT feasible-ascent + 22-day stale-score membership guard。
- Enabled contrasts：`C25-C23`、`C26-C25`、`C26-C23`。
- `SR-C24`與歷史C17～C22 identity保留但disabled，不得因本輪改寫歷史結果。

### 診斷與cache契約

- 新增daily diagnostics：guard enabled／max age、stale candidate count、guard triggered、seed blocked與blocked feasible score-improving swaps。
- Strategy summary直接聚合guard觸發日、stale候選數與blocked swaps，並在Resource-aware盤前診斷顯示22日門檻。
- C26 runtime options進入pair replay fingerprint；改門檻必須使C26 cache失效。C25 arm contract沒有runtime options且engine schema保持既有版本，因此既有C25 completed pair仍可正常REUSE，不因新增C26或報表欄位無關失效。

### 獨立synthetic結果

固定同一個C25 feasible-ascent盤前fixture：

- 全部score新鮮：C25 action=`F2,F3`，C26 action同為`F2,F3`。
- 只將`F3` score date改為超過22日：C26 action回到`F1,F2`；`F3`仍存在orderable output，guard triggered且blocked feasible score-improving swap>0；K與reserved-capital floor均維持。
- 核心selector不含`C26` scientific arm ID，只讀generic runtime policy與ranking options。
- Focused synthetic：既有config-driven Strategy Compare contract `41/41 PASS`；新增stale-score guard contract `4/4 PASS`。
- 以原始ZIP與修改後程式對同一dummy artifact identity實算SR-C25 pair fingerprint，兩邊皆為`079ff59ca62a3bf3`，確認C25既有cache identity不因C26新增而改變。

### 下一步

套用程式後先執行`apps/research.py → [3] 策略組合比較 → [2] 查看設定、工件與預計動作`。若Selection PIT與historical P2 artifacts READY，正式執行C23/C25/C26；預期既有C23/C25可依fingerprint重用，C26為新RUN。先看`C26-C25`是否改善same-param DL selection R／Return／RoMD／EV且不惡化underfilled slot-days，再看`C26-C23`是否真正超越baseline。只有Selection結果支持C26，才可把**完全相同22日規則**凍結搬到Forward-OOS；OOS不得再調門檻。

## 2026-08-09 — SR-C26 Selection結果：stale guard修復selection R，但portfolio gate仍未通過

### 狀態

`RESULT_AVAILABLE / STALE_GUARD_SUPPORTED / NOT_PROMOTED / FORWARD_OOS_NOT_AUTHORIZED`。

### 程式基準與正式結果

- 使用者本機執行基準：`test-branch-1_20260809_150343_3b98cc3.zip`。
- SHA256：`fe7d7aed0ec5ccbc126ff0f45b9c542cca82b1f1dc1923ebdda060b21bde9a48`。
- Strategy Compare run：`outputs/strategy_compare/runs/20260809_150908_C23-C25-C26_6018e78bd663`。
- 期間：2014-01-01～2020-12-31；historical P2 Min ROOS active params；MR-12B Selection PIT；C26唯一scientific change仍為預先凍結的22-calendar-day stale-score membership guard。

### C26相對C25：guard純runtime效果

- Return `115.42% → 119.74%`（`+4.32pp`）。
- MDD `26.36% → 27.11%`（`+0.75pp`）；RoMD `4.38 → 4.42`（`+0.04`）。
- EV `0.59R → 0.67R`（`+0.08R`）。
- same-param DL selection R `-19.52R → +7.99R`（`+27.51R`）。
- underfilled end days `963 → 923`（`-40`）；position-gap slot-days `1927 → 1781`（`-146`）。
- 實際改單日 `94 → 75`；Continuous新選入單 `97 → 79`；guard觸發日=`43`、blocked feasible score-improving swaps=`378`。
- 判定：22日stale-score guard確實修復C25 old-signal tail造成的selection loss；Selection evidence支持guard機制本身。

### C26相對C23：Selection portfolio gate

- Return `127.45% → 119.74%`（`-7.71pp`）。
- MDD `25.45% → 27.11%`（`+1.66pp`）；RoMD `5.01 → 4.42`（`-0.59`）；年化 `-0.55pp`。
- EV `0.62R → 0.67R`（`+0.05R`），same-param DL selection R=`+7.99R`。
- 平均曝險只`-0.20pp`，但underfilled end days仍`+219`、position-gap slot-days仍`+128`。
- 選中候選Target percentile `+0.0237`、Target mean `+0.0680R`；Target方向仍改善。
- 年度Return僅2014、2015、2019優於C23；2016、2017、2018、2020較弱。

### 科學判定

1. C26把C25的負selection R修成相對C23 `+7.99R`，故stale-score guard假說不是無效；不得把本結果解讀成回到C25。
2. 但主要策略Gate仍以Return／MDD／RoMD等portfolio結果為準；C26尚未超越C23，因此**不得進Forward-OOS**。
3. `+7.99R selection R`與`-7.71pp Return`形成新的R→portfolio translation gap。不能據此再調22日cutoff、增加第二個age threshold或修改MR-12B／Target。
4. 下一步只允許read-only attribution：拆exclusive selection ΔPnL、common-trade ΔPnL、position sizing／capital return、slot occupancy與wealth/compounding path，判斷positive R為何沒有轉成positive portfolio wealth。

## 2026-08-09 — AUD-c23-c26-pit-portfolio-translation實作：C26正Selection R到負Portfolio Return的只讀歸因

### 狀態

`IMPLEMENTED / RESULT_PENDING / READ_ONLY_EXISTING_REPLAY_ONLY`。

### Audit identity／固定來源

- 新增`AUD-c23-c26-pit-portfolio-translation`（config id=`c23-c26-pit-portfolio-translation`）；不新增`MR-*`、`DL-*`、`SR-C*`或`PARAM-*`。
- 目前candidate=`SR-C26`，comparators=`SR-C23 / SR-C25`，source=`latest` completed Strategy Compare。
- 共用既有`strategy_attribution` primitive；不建立第二套R／PnL／wealth公式，不重跑portfolio。

### 唯一目的

直接拆解：
- Exclusive selection ΔR 與 Exclusive selection ΔPnL；
- Common trades ΔPnL 與 All trade ΔPnL；
- position sizing／reserved／invested、capital return、holding；
- underfilled days／position-gap slot-days；
- exact log-wealth path與年度／月份compounding concentration。

若C26相對C23出現`Exclusive selection ΔR > 0`但總Return < 0，報表必須明確標示R→portfolio wealth translation gap，不能把positive R直接當作策略promotion證據。

### 使用限制／下一步

Audit只用既有Selection replay做解釋，不授權調22日cutoff、不授權新age gate、不授權參數適應或模型／Target修改。若負wealth主要可由既有capital/path機械項清楚解釋，才形成下一個單一Selection受控假說；若無明確可泛化機械原因，`SR-C26`維持NOT_PROMOTED並停止這條runtime微調鏈。

## 2026-08-09 — AUD-c23-c26-pit-portfolio-translation結果：正Selection R未轉成dollar PnL／wealth

### 狀態

`RESULT_AVAILABLE / R_TO_DOLLAR_TRANSLATION_GAP_CONFIRMED / C26_NOT_PROMOTED`。

### 程式基準與來源

- 使用者本機執行基準：`test-branch-1_20260809_152631_768fd77.zip`。
- SHA256：`28c35d36c73ca4a6278bbd2f026d9a1b43c793ecac82e841fb65658198f28156`。
- Audit：`AUD-c23-c26-pit-portfolio-translation`；source=`outputs/strategy_compare/runs/20260809_150908_C23-C25-C26_6018e78bd663`。
- 期間2014-01-01～2020-12-31；只讀既有replay，不重跑portfolio、不改score／selector／training。

### C26 vs C23

- Return=`119.74% vs 127.45%`（`-7.71pp`）；MDD=`+1.66pp`、RoMD=`-0.59`、EV=`+0.05R`。
- Exclusive selection ΔR=`+7.99R`，但Exclusive selection ΔPnL=`-35,524.03`。
- Common trades ΔPnL=`-41,604.23`；All trade ΔPnL=`-77,128.26`。
- Final relative wealth advantage=`-3.39%`；2020只占全期淨差異`12.61%`，非2020 relative wealth effect=`-2.97%`，故不是單一年份集中。
- 平均實際投入`+2,053`、平均預留`+2,607`、平均停損距離`+0.58pp`、平均Capital Return約持平、平均Realized R=`+0.05R`、平均持有日`+2.05`。
- Underfilled end days=`+219`、position-gap slot-days=`+128`；changed days成交買單差=`-15`、missed buy差=`+2`。

### C26 vs C25

- Return=`+4.32pp`、RoMD=`+0.04`、EV=`+0.08R`。
- Exclusive selection ΔR=`+27.51R`、Exclusive selection ΔPnL=`+57,383.45`、Common trades ΔPnL=`-14,203.16`、All trade ΔPnL=`+43,180.29`。
- Final relative wealth advantage=`+2.00%`；underfilled end days=`-40`、position-gap slot-days=`-146`。
- 這證明22日stale-score guard不只改善未加權R，也改善actual dollar trade PnL；不得回退C25或重新調22日門檻。

### 判定

1. C26相對C23的portfolio失敗由兩層共同構成：exclusive trades雖ΣR較高但dollar PnL較低；common trades亦因後續portfolio path出現負PnL差。
2. `Common trades ΔPnL`可能是前段wealth差異造成後續position sizing縮放的下游放大，不能與exclusive selection loss直接視為兩個獨立根因。
3. 現有平均invested／stop distance／capital return不足以解釋`+7.99R → -35.5k`；下一步只能在同一read-only Audit內拆每筆initial-risk dollars與共同交易R×Risk分解，不得新增cutoff、age gate、參數適應或Forward-OOS。

## 2026-08-09 — AUD-c23-c26-pit-portfolio-translation擴充：risk-dollar／common sizing分解

### 狀態

`IMPLEMENTED / SAME_AUDIT_ID / READ_ONLY / EXTENDED_DIAGNOSTIC_PENDING_RERUN`。

### 實作

1. 不新增`AUD-*`；維持`AUD-c23-c26-pit-portfolio-translation`，因問題仍是同一個R→PnL→wealth translation。
2. 每筆已結算trade在`R != 0`時只以canonical `PnL / R_Multiple`反推`implied initial risk`；`R=0`無法識別，明確列為未覆蓋，不由stop distance或未成交counterfactual補值。
3. Exclusive candidate-only／comparator-only新增：ΣR、ΣPnL、risk coverage、Σ／平均implied initial risk、risk-weighted R，以及winner／loser各自平均risk dollars。
4. Common matched trades新增精確對稱分解：`Δ(R×Risk) = ΔR × average(Risk) + ΔRisk × average(R)`；另列risk-size/path effect、R-difference effect、uncovered ΔPnL與decomposition residual。
5. 此分解只讀既有trade CSV；不改accounting、replay、selector、22日guard、模型或Target。
6. 同時修正generic strategy-attribution報表仍殘留「只比較C15 selector」的硬編文字，改由candidate arm identity動態顯示；不改計算。

### 下一步

重跑同一Audit即可。若C26-C23 exclusive端顯示winner risk dollars較小／loser risk dollars較大，則未加權R的正值主要被risk allocation/timing抵消；若common `Risk-size/path effect`幾乎解釋`-41.6k`且`R-difference effect`接近0，則common loss只是前段wealth差造成的下游position-size放大，不應另建selector假說。只有出現清楚、可泛化且Selection內可事前觀測的mechanism，才允許建立下一個SR runtime；否則C26維持NOT_PROMOTED並停止微調鏈。

## 2026-08-09 — AUD-c23-c26-pit-portfolio-translation risk-dollar重跑結果：C26 runtime微調鏈結案

### 狀態

`RESULT_AVAILABLE / RISK_DOLLAR_PATH_DECOMPOSITION_CONFIRMED / C26_NOT_PROMOTED / RUNTIME_MICROTUNING_STOPPED`。

### 程式基準與來源

- 使用者本機執行基準：`test-branch-1_20260809_153811_a2377c1.zip`。
- SHA256：`775a804fc47b4ab5684a602ebab4e8e2a9f966b533873f9f3e20c2313669ff35`。
- Audit：`AUD-c23-c26-pit-portfolio-translation`；source=`outputs/strategy_compare/runs/20260809_150908_C23-C25-C26_6018e78bd663`。
- 期間2014-01-01～2020-12-31；只讀既有replay，不重跑portfolio、不改score／selector／training。

### C26 vs C23：exclusive risk-dollar translation

- Exclusive selection ΔR=`+7.99R`，但exclusive ΔPnL=`-35,524.03`。
- C23-only：173 trades，ΣR=`+76.19R`、ΣPnL=`+231,681.38`、Σ implied initial risk=`854,162`、平均risk=`4,937`、risk-weighted R=`0.27R`。
- C26-only：158 trades，ΣR=`+84.18R`、ΣPnL=`+196,157.35`、Σ implied initial risk=`924,372`、平均risk=`5,850`、risk-weighted R=`0.21R`。
- Winner／loser平均risk：C23=`4,718 / 5,094`；C26=`5,674 / 5,967`。C26並非簡單整體de-risk；真正差異是實際risk dollars對不同R magnitude／交易時點的權重分布，使equal-weighted ΣR優勢反轉為較低的risk-weighted R與dollar PnL。

### C26 vs C23：common matched trades

- Common trades=`206`，risk coverage=`100%`。
- Common ΔR約=`0.00R`，Common ΔPnL=`-41,604.23`。
- C23/C26平均implied initial risk=`9,049 / 9,117`；C26平均risk並未較低。
- 對稱分解：risk-size/path effect=`-41,612.25`；R-difference effect=`+8.02`；uncovered=`0`；residual約`0`。
- 因共同交易R幾乎完全相同，這`-41.6k`不是「同一交易被C26做壞」，而是前面portfolio path、當時equity／cash／position state與canonical sizing共同形成的trade-specific risk-dollar差。它是既有selection path的下游放大，不應另建selector或依事後R修改fixed-risk sizing。

### C26 vs C25

- Exclusive ΔR=`+27.51R`、Exclusive ΔPnL=`+57,383.45`、All trade ΔPnL=`+43,180.29`，再次確認22日stale-score guard本身在equal-weighted R與actual dollar PnL兩層均有效。
- C25-only/C26-only risk-weighted R=`0.06R/0.13R`；guard後實際risk-dollar品質亦改善。
- Common ΔPnL=`-14,203.16`幾乎完全由risk-size/path effect=`-14,217.77`解釋；R-difference effect僅`+14.61`。

### 最終判定

1. `SR-C26`保留為`STALE_GUARD_SUPPORTED / NOT_PROMOTED`歷史研究結果；22-calendar-day guard不得再依Selection／OOS結果調整，也不回退SR-C25。
2. C26相對C23的`+7.99R`只代表equal-weighted exclusive R改善，不能取代portfolio Return／MDD／RoMD gate；actual exclusive risk-weighted R反而`0.21R < 0.27R`，且dollar PnL為負。
3. Common trade loss已被risk-size/path effect完整解釋；因平均risk並未下降，不能簡化為「C26整體position size較小」，而是portfolio path造成的trade-specific sizing分布。這是下游結果，不構成新的可事前觀測selector假說。
4. 目前沒有清楚、可泛化、Selection盤前可觀測且不依賴未來realized outcome的mechanism支持再改fixed risk、stop distance、selector、22日guard或資金配置。為避免Selection過度微調，**C26 runtime微調鏈在此停止；不進Forward-OOS、不進參數適應、不新增SR-C27。**
5. `same-param DL selection R`後續只保留為診斷，不得單獨作promotion gate；portfolio gate仍以Return／MDD／RoMD／EV與正式wealth path為主。

### 下一步

離開C26 runtime微調，回到尚未補齊的模型層證據：先做`MR-12B vs MR-12A` **same-fold paired Selection PIT read-only comparison**，固定相同PIT日期／共同候選／Continuous Target，直接比較每fold daily rho、pair concordance、Top-K／boundary與年度穩定性；只讀既有PIT工件，若MR-12A工件不存在才由正式模型入口依既有profile建立，不用C23/C26策略結果調模型。這一步先判定Pairwise objective在歷史PIT是否真的穩定優於MSE，再決定下一個模型研究方向；不得重啟11J counterfactual。


## 2026-08-09 — MR-13A Daily Universal No-time Pairwise Ranker：stage 1實作

### 狀態

`IMPLEMENTED / RESULT_PENDING / FORWARD_OOS_MODEL_GATE_ONLY / PIT_NOT_IMPLEMENTED / STRATEGY_NOT_CONNECTED`。

### 使用者決策與程式基準

- 使用者明確選擇離開C26 runtime微調鏈，並將下一個模型方向改為「每天對所有合法股票直接預測quality score」，不再先以breakout event決定模型sample。
- 因此上一版Log排隊的`MR-12B vs MR-12A same-fold paired Selection PIT read-only comparison`暫不執行；這是使用者新的研究優先順序，不代表舊比較假說被結果淘汰。
- 本輪來源ZIP：`test-branch-1_20260809_162849_0734449(1).zip`。
- SHA256：`e7b71905c7c469b0f66758e207c4f80f0515ca101a79d20674b4b90c7c0dac5f`。
- 新Model Research ID：`MR-13A`；profile=`daily_universal_no_time_pairwise`；target=`daily_opportunity_no_time_r_v1`。
- Active identity分離：模型研究入口由`BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE`指向MR-13A；既有strategy workflow的`BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE`維持MR-12B。MR-13A stage 1不得因切換模型研究profile而改變Selection PIT／strategy compare的預設模型身份。

### Scientific contract

1. Sample universe由`breakout_event_groups`改為`daily_eligible_stock_days`：每支有足夠300-bar歷史、benchmark同日可用、且40-bar future target完整的股票／交易日最多一筆sample。
2. Candidate membership、`high_len`、breakout level與任何strategy order／fill state都不參與training sample selection或模型input；breakout candidates只在checkpoint寫入後作OOS diagnostic slice。
3. Input architecture不變：沿用既有`300×10` stock+0050 normalized OHLCV sequence與`ARCH-inception_time_v1`；不新增手工MA／breakout context。
4. Target沿用MR-12 no-time target的既有40-day adverse-first future-path核心：在首次10% risk barrier touch前取最早最大safe high，adverse excursion量到該peak；公式仍為`favorable/risk_budget - adverse/risk_budget`。新identity只表示它可在任意合法stock-day計算，不依賴breakout-event語意。
5. Objective／loss／epoch Gate不變：`daily_pairwise_ranking / pairwise_logistic / mean_daily_spearman`；whole-date coherent batch不得拆日，也不新增pair sampling或固定pair數magic number。
6. OOS在checkpoint寫入後才推論；不得參與loss、gradient、epoch selection或target normalization。

### Storage／performance contract

- 不建立`stock-day × 300 × 10` expanded feature bank；daily sample只持久化compact identity／target／score類工件。
- 訓練時以canonical sanitized ticker OHLCV + benchmark作lazy sequence materialization，重用既有`build_breakout_quality_sequence_feature`，避免第二套feature公式。
- Daily 40-bar target改為每ticker向量化計算；以109個synthetic stock-days逐筆對照canonical scalar target，最大絕對誤差=`3.0131e-08`，語意一致。
- Daily universe的epoch selection略過每epoch完整inner-train inference metrics，因該數值不參與選模；完整Validation仍每epoch評估，選模規則不變。舊MR-12 path預設行為不變。

### Stage 1驗證輸出

正式訓練完成後，報表必須同時提供：

- `All eligible stock-days` forward OOS：Mean Daily Spearman、Global Spearman、Pair Concordance、Top/Bottom 10% Target、NDCG@K、Top-K Lift、Oracle overlap、Boundary concordance／gap。
- `Breakout candidate slice` forward OOS：以同一MR-13A checkpoint／scores切出官方breakout ticker/date後計算同組排序品質；candidate membership只作post-checkpoint診斷。
- Stage 1不建立Selection PIT scores、不建立strategy ranking source、不跑C17/C18／ROOS，也不根據strategy結果調模型。

### 下一步Gate

先由使用者在正式`apps/research.py → 模型訓練`執行MR-13A full forward-OOS。只有全市場與breakout-candidate slice模型結果支持daily-universal方向，才進MR-13A stage 2建立Selection PIT daily scores；若forward-OOS本身不支持，直接停在模型層，不消耗PIT／strategy replay成本。

## 2026-08-09 — MR-13A Stage 1：daily lazy feature feeding效能修正

### 狀態

`IMPLEMENTED / PERFORMANCE_ONLY / SCIENTIFIC_CONTRACT_UNCHANGED / RESULT_PENDING`。

### 程式基準與問題定位

- 使用者本輪來源ZIP：`test-branch-1_20260809_182833_19839e6.zip`。
- SHA256：`30c30900ddce00674ce8c853ef1bda76e50ffec56af84795c3eaca1d281fcad9`。
- 使用者觀察：MR-13A full training耗時長，GPU utilization約20%。
- 原MR-13A為避免建立巨大的`stock-day × 300 × 10` expanded feature artifact，採`LazyDailyFeatureBank`；但每個training／evaluation batch會在單一Python執行緒逐stock-day呼叫DataFrame版canonical feature builder。每筆均重做stock／0050 300-bar slicing、benchmark date lookup與OHLCV normalization；同一交易日的0050 window亦會被每支股票重算。GPU因此大量時間等待CPU feature materialization。
- `ARCH-inception_time_v1`僅約473k trainable parameters；在RTX 5080等高階GPU上，此feeding bottleneck會比模型算力更早成為限制。

### Performance-only實作

1. `filters/breakout_quality/features.py`新增canonical `normalize_ohlcv_array_window()`，DataFrame builder與MR-13A NumPy lazy path共用同一數值實作，避免feature公式分叉。
2. `LazyDailyFeatureBank`建立時只把canonical sanitized OHLCV轉為RAM中的NumPy arrays；不建立每個stock-day的expanded sequence bank。
3. 每個stock-day只以NumPy slice materialize其300-bar stock window，不再逐sample執行Pandas `iloc`與benchmark index lookup。
4. dataset建索引時同步保存每個sample的benchmark position；同一benchmark position的normalized 300×5 window在RAM cache只計算一次並跨batch／epoch重用。cache規模只隨benchmark交易日數成長，不隨ticker×date sample數成長。
5. Continuous ranker新增config-driven `BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES=2`；單一背景CPU worker只依既定batch順序預先materialize後續features，GPU仍按原順序做完全相同optimizer steps。設為0可關閉。
6. 不修改MR-13A sample universe、40-day target、daily percentile、whole-date batch ordering、pairwise loss、optimizer、LR、mixed precision、epoch selection、checkpoint、OOS Gate或持久工件內容。

### 獨立等價／效能檢查

- Synthetic 40 tickers × 3 dates，共120個300×10 sequence：新NumPy lazy path與原canonical `build_breakout_quality_sequence_feature()`逐值`array_equal=True`，最大絕對誤差=`0`。
- Synthetic單日550 stocks materialization：原版約`0.22～0.32s/batch`（約`1.7k～2.5k stock-days/s`）；修正後約`0.05～0.06s/batch`（約`9.3k～10.8k stock-days/s`），此環境約4～5倍feeding加速。此benchmark只量feature materialization，不宣稱等比例縮短整體training wall time。
- Prefetch pipeline synthetic 10-batch檢查（每批CPU materialization與GPU stand-in各30ms）：prefetch=0約`0.604s`、prefetch=2約`0.335s`，batch order完全一致；此測試只驗證pipeline overlap與順序契約，不代表正式GPU wall-time倍率。

### 判定

此修正不占用新的`MR-*` identity，因沒有改變模型權重語意、target、loss、architecture或training-data semantics；MR-13A仍為`RESULT_PENDING`。正式full forward-OOS需以修正版重新執行後，才能記錄實際epoch時間、GPU utilization與模型Gate結果。


## 2026-08-09 — MR-13A Stage 1結果：Forward-OOS Gate通過，授權Stage 2 Selection PIT

### 狀態

`RESULT_AVAILABLE / FORWARD_OOS_MODEL_GATE_PASS / NOT_PROMOTED_OVER_MR12B / STAGE2_AUTHORIZED`。

### 程式基準與固定條件

- 使用者本機結果來源：`test-branch-1_20260809_184850_7662026(1).zip`；SHA256=`c66977555f0f404c36f29f46bbb38a81f4bc17de0cf8607efde3768e973f7e26`。
- Filter=`breakout_quality_v1`；Architecture=`inception_time_v1`；Profile=`daily_universal_no_time_pairwise`；Seed=42。
- Sample scope=`daily_eligible_stock_days`；Target=`daily_opportunity_no_time_r_v1`；Objective=`daily_pairwise_ranking / pairwise_logistic`；epoch metric=`mean_daily_spearman`。
- Daily universe=`1,578,349` samples／`558` tickers；feature storage仍為lazy canonical 300×10 sequence，不建立expanded daily feature artifact。
- Epoch 1為Selection內Inner Validation選模最佳：選模Validation daily rho=`0.0999`；Epoch 2=`0.0436`，故selected epoch=`1`。完整Selection refit固定1 epoch。
- 重訓後原Validation daily rho=`0.2631`只作描述；因final refit已包含原Validation rows，**不得**當作獨立選模／泛化證據。

### Forward-OOS結果

- All eligible stock-days OOS：`608,204` groups；daily rho=`0.1755`、global rho=`0.1761`、pair concordance=`56.12%`；Top 10% Target=`1.8261R`、Bottom 10%=`0.4846R`，spread=`+1.3415R`。
- All-stock K=10：NDCG=`0.6045`、Top-K Target=`2.0375R`、Lift=`+1.0357R`、Oracle overlap=`9.07%`、Boundary=`50.99%`、Boundary gap=`+0.1035R`，competition days=`1,203`。
- Breakout-candidate OOS：`17,346` groups；daily rho=`0.1490`、global rho=`0.1894`、pair=`56.91%`；Top 10% Target=`2.3302R`、Bottom 10%=`0.5250R`。
- Breakout K=10：NDCG=`0.7201`、Top-K Target=`1.6198R`、Lift=`+0.4056R`、Oracle overlap=`61.04%`、Boundary=`54.02%`、Boundary gap=`+0.2158R`，competition days=`603`。

### 與MR-12B anchor的同口徑判讀

- 既有MR-12B breakout Forward-OOS anchor：daily rho=`0.1568`、pair=`56.40%`、K=10 Top-K Target=`1.6231R`、Lift=`+0.4089R`、Oracle overlap=`59.80%`、Boundary=`52.98%`、Boundary gap=`+0.2121R`，competition days同為`603`。
- MR-13A相對MR-12B：daily rho=`-0.0078`；pair=`+0.51pp`；Top-K Target=`-0.0033R`；Lift=`-0.0033R`；Oracle overlap=`+1.24pp`；Boundary=`+1.04pp`；Boundary gap=`+0.0037R`。
- 因此MR-13A不是全面優於MR-12B；較合理結論是breakout實用raw Top-K品質近乎持平、pair／boundary略升，同時成功取得全市場每日排序能力。
- **NDCG比較限制**：MR-13A Stage 1先在full daily universe建立same-day percentile relevance再切breakout candidate；MR-12B在event universe建立percentile relevance。因此`0.7201 vs 0.6905`只作描述，不納入嚴格head-to-head promotion證據。Raw Target／Lift／Pair／Boundary gap／Oracle overlap不依賴此percentile normalization差異，可直接判讀。

### 判定與下一步

1. Stage 1達成預先設定的模型層方向Gate：全市場OOS與breakout slice都呈正向排序能力，因此授權同一`MR-13A`進Stage 2 Selection PIT。
2. 不建立新MR identity；Stage 2不改target、loss、architecture或training-data semantics，只把既有PIT infrastructure泛化到daily sample provider。
3. `DL-CONT12B / MR-12B`仍為current validated continuous anchor；MR-13A尚未建立strategy runtime source，也不得因Stage 1結果切換strategy workflow。
4. Stage 2先看historical PIT all-stock primary ordering與breakout-candidate diagnostic；只有PIT Gate通過後才討論Stage 3 strategy runtime／same-parameter comparison。

## 2026-08-09 — MR-13A Stage 2：Daily Selection PIT與模型Audit實作

### 狀態

`IMPLEMENTED / RESULT_PENDING / SAME_MR13A_ID / PIT_MODEL_GATE_ONLY / STRATEGY_NOT_CONNECTED`。

### 程式基準與唯一變更

- 本輪來源ZIP仍為`test-branch-1_20260809_184850_7662026(1).zip`；SHA256=`c66977555f0f404c36f29f46bbb38a81f4bc17de0cf8607efde3768e973f7e26`。
- 不改MR-13A scientific condition；唯一功能性擴充是讓既有Selection PIT pipeline依`training_sample_scope`選擇canonical event或daily provider，並讓PIT audit primary scope由profile語意決定。
- `BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE`仍維持MR-12B；`BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE`維持MR-13A。Stage 2不得暗中改策略預設。

### PIT／no-lookahead contract

1. Event ranker繼續使用原`load_continuous_ranker_data`；daily ranker由同一public pipeline路由到`load_daily_universal_ranker_data`，不複製第二套PIT trainer。
2. Daily training eligibility不再依賴PASS／REJECT label（daily rows的label固定為-1）；只依daily target validity。Legacy event profile的PASS-only／all-label mask完全保留。
3. 每fold仍固定：inner train與final refit都必須`label_eval_end_date <`對應validation／score cutoff；score rows不進該fold training、epoch selection或target percentile建立。
4. Daily fold沿用Selection內Validation選epoch與完整歷史refit；為避免大量無用推論，daily epoch selection不計算不參與選模的inner-train完整metrics，Validation Gate不變。
5. PIT builder補齊config-driven `train_prefetch_batches`參數；這也修正performance prefetch加入後legacy PIT重建可能缺少arg的infrastructure regression。
6. PIT builder與audit改由實際`--experiment-profile`解析settings，不再錯讀strategy-workflow active profile；因此MR-13A model research與MR-12B strategy workflow真正隔離。
7. PIT audit新增`training_sample_scope`工件身份fail-fast；新工件必須與目前profile完全一致，legacy manifest缺欄位時只按歷史契約解讀為`breakout_event_groups`，不得把event PIT誤接daily provider。

### Audit contract

- Legacy event PIT primary evidence維持`PASS-only target`，舊audit若沒有新欄位仍預設此語意，歷史Gate相容。
- MR-13A daily PIT primary evidence=`all_valid_target / All eligible stock-days`；年度方向Gate仍要求global rho>0、mean daily rho>0、過半年度rho>0且過半年度Top-bottom spread>0。
- Daily audit另以官方breakout dataset ticker/date切出`breakout_candidate_target` diagnostic；candidate membership只在score生成後作audit，不進daily training sample selection。
- Pair concordance與K=10 Top-K／boundary使用現有continuous-ranker canonical quality計算；audit不另寫第二套排名公式。Candidate audit的percentile relevance在candidate scope內重新建立，因此其NDCG口徑對event candidate universe更直接。
- Daily target沒有獨立event-style target manifest；audit source record改以PIT manifest內嵌target contract作可追溯來源，不假造不存在的persisted target artifact。
- PIT builder的`runtime_eligibility.eligible`仍為False；Stage 2 audit結果即使PASS也不等於forward-OOS runtime資格，strategy source／ROOS仍未接入。

### Dataset／Label需求

- 不重建既有`DATA-breakout_quality_v1` feature bank identity，不新增binary Label。
- Daily stock-day index／40-day target依canonical sanitized OHLCV在PIT執行時確定性重建；feature仍lazy materialization，不產生expanded 300×10 daily feature artifact。
- 持久PIT score只保存ticker/date/group index/score/fold/information cutoff等compact欄位；檔案會隨stock-day數成長，但不是feature tensor級膨脹。

### 下一步

由正式`apps/research.py → 模型訓練 → 建立／更新 Selection PIT Scores → PIT模型驗證`執行MR-13A Stage 2。先審查all-stock歷史PIT、年度穩定性與breakout-candidate diagnostic；未取得結果前不得把MR-13A寫成strategy DL source，也不得跑ROOS promotion。

## 2026-08-09 — MR-13A Stage 2結果：Selection PIT模型Gate通過

### 狀態

`RESULT_AVAILABLE / SELECTION_PIT_MODEL_GATE_PASS / STAGE3_STRATEGY_TRANSLATION_AUTHORIZED / NOT_PROMOTED_OVER_MR12B`。

### 程式基準與固定條件

- 使用者提供結果後的最新程式ZIP：`test-branch-1_20260809_193058_e59550d.zip`；SHA256=`ef437ca2f128321243af230743794e73d5204ca41fb186ec6929868c60fe44b8`。
- Filter=`breakout_quality_v1`；Architecture=`inception_time_v1`；Profile=`daily_universal_no_time_pairwise`；Seed=42。
- Objective=`daily_pairwise_ranking / pairwise_logistic`；Target=`daily_opportunity_no_time_r_v1`；PIT fold/validation=`12/24 months`。
- PIT score period=`2013-04-01～2020-12-31`；8 folds；本次model-audit target-valid groups=`778,532`，coverage=`100.00%`。
- 各fold selected epoch依序=`3,1,1,1,1,1,1,1`；validation daily rho依序=`0.0580,0.1125,0.1138,0.1311,0.1065,0.1167,0.1366,0.0567`。

### PIT模型結果

- All eligible target-valid stock-days：global rho=`0.0783`、mean daily rho=`0.1111`、pair concordance=`53.84%`、Top-bottom spread=`+0.3319R`。
- 年度方向穩定：rho>0=`8/8`、spread>0=`8/8`；score-level drift=`True`只作warning，不否決方向Gate。
- Breakout-candidate slice：global rho=`0.0580`、daily rho=`0.0784`、pair=`53.98%`、Top-K Lift=`+0.1102R`、Boundary=`51.49%`。
- Daily sample沒有PASS/REJECT label，因此binary classification overlap不適用；orderable coverage留到strategy replay建立。

### 判定

1. Stage 2 PIT模型Gate PASS：全期與每年度排序方向一致為正，且breakout candidate slice亦維持正向訊號，因此同一`MR-13A`授權進Stage 3 Selection strategy translation。
2. PIT訊號強度明顯低於Stage 1 Forward-OOS，尤其breakout slice daily rho由`0.1490`降至`0.0784`、pair由`56.91%`降至`53.98%`；因此Stage 3必須視為經濟轉化驗證，不得預設daily model已優於MR-12B。
3. `DL-CONT12B / MR-12B`仍為current validated continuous anchor；MR-13A在取得同參數Selection strategy結果前不得promotion或進ROOS。

### Stage 3 preflight發現

在將這份PIT score接入strategy runtime前，另行檢查出兩個不影響上述model-audit數值、但會使strategy replay不合格的契約問題：

1. Stage 2 daily index把「future 40-bar target完整」同時當成score row存在條件。這對train/audit合法，但若策略把score absence當資訊，就會讓是否有score間接受未來資料完整性／停止交易影響。
2. 既有Selection PIT runtime lookup固定使用breakout `signal_date`。對MR-12B event score正確，但對MR-13A daily model會退化為「事件日算一次後沿用」，沒有真正測每日refresh語意。

因此本次`778,532`-row PIT table保留為**model-audit evidence**，但不得直接作Stage 3 strategy runtime source。Stage 3必須先完成future-independent score universe與daily information-date lookup後重建PIT Scores＋Audit。


## 2026-08-09 — MR-13A Stage 3：future-independent daily PIT runtime與C27/C28實作

### 狀態

`IMPLEMENTED / SAME_MR13A_ID / PIT_RUNTIME_READY / SELECTION_STRATEGY_RESULT_PENDING / NOT_PROMOTED`。

### 程式基準

- 本輪來源ZIP：`test-branch-1_20260809_193058_e59550d.zip`。
- SHA256：`ef437ca2f128321243af230743794e73d5204ca41fb186ec6929868c60fe44b8`。
- 不修改MR-13A target、architecture、loss、optimizer、epoch Gate或selector scientific rules；因此不建立新的`MR-*`。

### Daily score eligibility修正

1. Daily universe拆成兩個獨立資格：
   - `feature-eligible`：截至該information date已有完整300-bar stock/benchmark歷史，可安全做inference；**必須有PIT score**。
   - `target-valid`：未來40 bars完整，可計算`daily_opportunity_no_time_r_v1`；只允許進train、validation、final refit與model-quality audit。
2. `target_valid=False`的feature-eligible row會保留ticker/date/group identity並輸出score，但target=`NaN`、`label_eval_end_date=NaT`，不得進loss、percentile target、epoch selection或audit target metrics。
3. 為維持Stage 2 scientific training universe，原target-valid rows保持原ticker/date/group順序與group index；inference-only rows只追加在其後，不重寫既有target-valid identity。
4. 不建立expanded `stock-day × 300 × 10` feature artifact；仍由canonical sanitized OHLCV lazy materialization。

### PIT artifact／fail-fast contract

- Daily PIT fold與top manifest新增`score_eligibility_contract`：`eligibility_basis=feature_history_only`、`future_target_required_for_score=False`、`target_valid_required_for_training=True`。
- 此欄位只加入daily profile；legacy MR-12 event PIT的schema version、fold payload與相容migration欄位維持原狀，避免無關歷史fold失效。
- Strategy loader對MR-13A daily source強制驗證該contract；舊Stage 2 manifest缺欄位時直接要求「重新建立PIT Scores並重新執行PIT audit」，不得silent fallback。

### Daily runtime information-date contract

1. MR-12B event PIT維持以原breakout `signal_date`查score，歷史語意不變。
2. MR-13A daily PIT不使用原signal date；每個盤前decision/candidate refresh查**最新已完成交易日**的score。
3. Extended/continuation candidate若source為daily profile不得沿用先前繼承score；每個decision day重新依目前information date查PIT score。
4. 不使用當日尚未完成OHLCV，不以fill date、future target或成交後狀態決定score date。

### Stage 3 Strategy arms

- 新source：`DL-CONT13A-PIT / CONT13A_PIT`，backing identity=`MR-13A / daily_universal_no_time_pairwise`。
- C26結案時的「不新增SR-C27」限定於**不再延伸C26 runtime微調鏈**；本輪是在使用者另行啟動MR-13A模型研究後，使用下一個未占用strategy ID建立新的model-translation branch，不回收或延續C26假說。
- `SR-C27`：完全沿用`SR-C24` historical P2 params、K/R0、minimum-repair selector與execution order；唯一DL差異=`DL-CONT12B-PIT → DL-CONT13A-PIT`。
- `SR-C28`：完全沿用`SR-C25` historical P2 params、K/R0、feasible-ascent selector與execution order；唯一DL差異=`DL-CONT12B-PIT → DL-CONT13A-PIT`。
- **不繼承SR-C26的22-calendar-day stale-score guard**：該guard是MR-12B event-score aging假說；MR-13A本身每天refresh，帶入22日門檻會混入無關runtime變因。
- Active controlled contrasts固定為：`C27-C24`、`C28-C25`、`C27-C23`、`C28-C23`、`C28-C27`。
  - `C27-C24`、`C28-C25`：隔離純model/source差異。
  - `C27/C28-C23`：檢查daily PIT是否真正轉成portfolio economics。
  - `C28-C27`：檢查相同MR-13A source下feasible-ascent是否比minimum-repair有更好的轉化。

### PIT重建效能契約

- Stage 3改變的是daily PIT **score universe**，不是fold的模型訓練條件。為避免8個既有fold無意義重訓，builder新增daily-only checkpoint rescore路徑。
- 只有舊manifest與checkpoint hash合法，且`inner_train / validation / final_refit`的observed period、group/event-row counts、model information cutoff、model spec、experiment/training settings、source/lookahead contract、seed與selected epoch全部一致時，才允許重用既有權重。
- `score`側group/event-row counts與`score_eligibility_contract`可因feature-eligible universe擴充而改變；重用後會重新推論完整新score period、重寫score artifact／fold manifest與checkpoint內fold contract。
- 任一training-side identity不同時不做近似或部分重用，直接回到既有完整fold training。此路徑只省計算時間，不改MR-13A scientific condition。

### Validation contract

- 新synthetic contract覆蓋：target-invalid row不進train但仍進score period；event profile查signal date、daily profile查information date；daily continuation每日refresh；舊daily PIT score-eligibility manifest被拒；C27/C28只能換source、不准變selector/params；training pipeline與strategy diagnostic共用domain-layer profile sample provider。
- `doc/TEST_SUITE_CHECKLIST.md`新增B197／T294，列為P0正式contract coverage。
- 本輪仍依專案規則不由ChatGPT執行`apps/test_suite.py`；正式double check由使用者本機執行。

### 下一步

1. 先重新建立MR-13A Selection PIT Scores並重跑PIT Audit。score row數**可能高於、也可能等於**Stage 2的`778,532`；是否增加取決於Selection期間是否實際存在feature-eligible但target-invalid rows。正式Gate只驗`feature_history_only` score-eligibility contract與daily information-date lookup，**不得把row數增加當成正確性或模型改善條件**。
2. Audit model metrics仍只計target-valid rows，因此若scientific condition與資料未變，應與Stage 2結果在合理數值誤差內一致；若顯著改變，先停止strategy replay並查dataset identity／split。
3. 新PIT manifest通過score-eligibility contract後，才執行Strategy Compare `C23/C24/C25/C27/C28`。取得Selection結果前不得進ROOS、不得切換strategy workflow active profile、不得宣告MR-13A取代MR-12B。


## 2026-08-09 — MR-13A Stage 3 PIT runtime rebuild結果：contract生效，允許Selection strategy compare

### 執行結果

- Profile=`daily_universal_no_time_pairwise`；PIT period=`2013-04-01～2020-12-31`；8 folds。
- Builder：`checkpoint重評=8`、`新建=0`、完整fold retraining=0；總耗時=`01:14.3`。
- PIT score groups=`778,532`、coverage=`100.00%`。
- Audit all-stock：global rho=`0.0783`、daily rho=`0.1111`、pair=`53.84%`、spread=`+0.3319R`。
- Breakout slice：global rho=`0.0580`、daily rho=`0.0784`、pair=`53.98%`、Top-K Lift=`+0.1102R`、Boundary=`51.49%`。
- Audit Gate=`PASS`；target-valid模型品質與Stage 2結果一致。

### 重要更正：score groups不必增加

Stage 3實作前曾預期future-independent score universe會使row數高於`778,532`。實際重建後row數相同。重新核對canonical builder與PIT split後確認：

1. daily provider已把score eligibility與future target validity分離；`score_ids`只依score日期範圍，不套`target_valid`。
2. train／validation／final-refit仍必須`target_valid=True`且遵守label-end embargo。
3. 因此row數是否增加只取決於2013-04-01～2020-12-31期間是否真的存在feature-eligible但target-invalid stock-days；本次正式資料中該差集沒有造成額外Selection PIT rows。
4. **row count不是Stage 3 contract Gate**。真正Gate是新PIT manifest必須宣告`score_eligibility_contract.eligibility_basis=feature_history_only`，且策略runtime對MR-13A使用最新已完成交易日score，而不是breakout signal-date score。

### 判定

- `DL-CONT13A-PIT`：`STRATEGY_RUNTIME_READY`。
- `SR-C27 / SR-C28`：`READY_FOR_SELECTION_COMPARE / RESULT_PENDING`。
- 下一步直接執行Selection Strategy Compare的C23/C24/C25/C27/C28固定比較；優先看C27-C24與C28-C25的source-only差異。
- 在C27/C28 Selection結果取得前，MR-13A仍不得取代MR-12B，也不得進ROOS。


## 2026-08-09 — Stage 3 Strategy Compare前置依賴自動化修正

### 狀態

`INFRASTRUCTURE_FIX / NO_SCIENTIFIC_CONDITION_CHANGE / SELECTION_COMPARE_STILL_PENDING`。

### 使用者觀察

在C23/C24/C25/C27/C28 Strategy Compare狀態頁，`CONT13A_PIT`已READY，但`CONT12B_PIT`三個top-level工件與`selection_min_roos`缺少時整體直接BLOCKED。使用者指出既有Strategy Compare原已實作「正式入口先規劃依賴，能由既有真理工件確定完成的前置自動建立／接續」契約。

### 根因

1. Selection歷史P2來源`selection_min_roos`在C23～C28研究設計時刻意設定`builder=None`，因此即使它屬策略參數工件、且已有正式rolling optimizer service，planner仍只能標BLOCKED。這與`PROJECT_SETTINGS`目前「策略比較可建立比較所需策略參數」的全專案契約不一致。
2. Selection PIT source一律禁止builder，將「不得由Strategy Compare訓練PIT模型」錯誤擴張為「連既有fold score/checkpoint的純推論工件重建也禁止」。實際上專案契約允許策略比較匯出既有模型推論工件。
3. PIT CLI argparse原先在解析`--experiment-profile`前就以目前model-research Active Profile填入fold／validation／seed等預設值；不同PIT source設定若未來分叉，checkpoint identity可能錯用另一profile的預設。

### 修正

- `selection_min_roos`新增config-driven `selection_historical_p2` builder；重用既有Selection baseline truth，只建立／接續2014～2020 P2_HISTORY risk-only active params，不建立Label、不啟用DL。
- `CONT12B_PIT`與`CONT13A_PIT`新增`selection_pit_from_existing_folds` builder。Strategy Compare若PIT top-level工件缺少／無效，可自動重組既有fold scores，或從training identity完全一致的既有checkpoint重新推論，再執行PIT Audit。
- PIT builder新增`--checkpoint-only`硬契約：任何fold無法由既有score/checkpoint合法重用時，在進入`_train_fold`前立即失敗，Strategy Compare不得因此訓練模型。
- PIT parser先解析`--experiment-profile`，再由該profile取得PIT defaults，消除Active Profile隱性耦合。
- 前置計畫對一個PIT source只建立一個bundle BUILD action，避免manifest／audit／scores三個檔各自重複執行builder；狀態頁仍逐檔顯示預計動作。

### Scientific identity／判定

MR-12B、MR-13A、DL-CONT12B-PIT、DL-CONT13A-PIT、SR-C23～C28的target、model weights、selector、策略參數語意與比較期間均未修改；本輪只修正前置工件 orchestration，不新增MR/SR ID。若既有PIT fold/checkpoint也不存在或identity不相容，仍必須BLOCK並由模型研究入口重建，不能為了自動化跨越「策略比較不得訓練模型」邊界。

## 2026-08-09 — Stage 3 Strategy Compare archived comparator自動重用閉環

### 狀態

`INFRASTRUCTURE_FIX / NO_SCIENTIFIC_CONDITION_CHANGE / C24_C25_ARCHIVED_PAIR_REUSE_SUPPORTED / SELECTION_COMPARE_STILL_PENDING`。

### 程式基準與使用者錯誤

- 本輪來源ZIP：`test-branch-1_20260809_211141_5a2c093.zip`；SHA256=`08172f5433fc76cd2f0fa404cf1e03c71a8ebde1408f0b300ed9f538687175af`。
- Strategy Compare已進入`PREPARABLE`並嘗試checkpoint-only重建`DL-CONT12B-PIT`，但在真正fold reuse前被缺少`outputs/filters/breakout_quality/breakout_quality_v1/continuous_targets/strategy_aligned_opportunity_no_time_r_v1/manifest.json`阻擋。
- 該Continuous Target屬MR-12B模型研究上游；依專案契約，Strategy Compare不得為了補歷史comparator自行建立Target／Label。

### 根因

先前將「Selection PIT top-level工件缺失可由既有fold/checkpoint重建」泛化過度。MR-12B event provider在載入fold plan時仍需要Continuous Target以重建train／validation／refit identity，而PIT Audit也必須用Target重新計算Spearman／spread並驗證Target manifest hash。因此當Target artifact本身已被刪除時，不能合法地只靠checkpoint-only重建完整model-audit source。

然而本輪C24/C25並不是待重跑的新arm：它們已有2014～2020正式completed Strategy Compare pair，且本次用途只是作C27/C28的固定歷史comparator。要求恢復已刪MR-12B Target只為重新產生不會被replay的C24/C25，是不必要的反向依賴。

### 修正

1. Strategy Compare新增狹義`archived_completed_pair` cache路徑，只允許Selection PIT類歷史source缺失時使用；它不把缺少的PIT source標為READY，也不重建Target／Audit。
2. archived pair必須同時滿足：
   - current historical P2 params已存在，且SHA256與completed run記錄完全一致；
   - dataset、param policy、max positions、rotation、comparison period、param source identity、DL filter／architecture／profile／score source與off/on arm runtime contract全部一致；
   - completed run當時記錄的PIT manifest／audit／forward-score identities皆有非空SHA；
   - completed pair的正式JSON／Markdown／yearly／equity／trades／capacity／orderable／selected-buys等必要工件仍完整；
   - pair engine schema仍與目前一致。
3. 任一條件不符即不得使用archived reuse；若current param SHA不同，必須拒絕而不是把歷史結果搬到不同參數宇宙。
4. 前置executor改為每次只完成一個BUILD／REBUILD後立即重新規劃，並優先建立`param:*`。因此本案例會先建立／接續`selection_min_roos`；若其SHA證明C24/C25 completed pairs可重用，下一波直接取消`CONT12B_PIT`重建依賴。若無可用completed pair，原checkpoint-only／BLOCKED安全邊界仍保留。
5. 新run的pair execution manifest會標記`source_artifact_mode=archived_completed_pair`，保留「結果來自已完成歷史pair、不是目前source重新READY」的可稽核差異。

### Scientific identity

MR-12B、MR-13A、C23～C28的模型權重、Target、selector、historical P2語意、comparison period與策略會計均未修改；本輪只修正dependency orchestration與completed-result cache provenance，不新增MR／SR ID。C27/C28 Selection結果仍待正式執行。


## 2026-08-09 — Stage 3 Strategy Compare historical P2 prerequisite閉環

### 狀態

`INFRASTRUCTURE_FIX / NO_SCIENTIFIC_CONDITION_CHANGE / AUTO_PREPARATION_CHAIN_COMPLETED / SELECTION_COMPARE_STILL_PENDING`。

### 程式基準與使用者錯誤

- 本輪來源ZIP：`test-branch-1_20260809_212916_6c122dc.zip`；SHA256=`7324bc96a9f6150b3aa30af03dc5f474f3dd306da767f182c0b4cd0c5b048ed5`。
- Strategy Compare前置已正確把`param:selection_min_roos`排在舊`CONT12B_PIT`之前，但正式執行時因缺少`models/research/breakout_quality/selection_strategy_realization/roos_base_best.json`而停止。
- 該檔不是外部不可重建資料；11I歷史契約本來就是以2014-01-01～2020-12-31、120個月fixed train window、12個月OOS，將canonical outer-rolling optimizer隔離輸出到`models/research/breakout_quality/selection_strategy_realization/`。

### 根因

Stage 3新增的`selection_historical_p2` builder只共用了P2 risk-only訓練服務，仍把11I historical rolling baseline當成「必須預先存在」的上游真理工件。這與`PROJECT_SETTINGS`的正式App依賴規則不一致：策略參數工件若可由正式optimizer service確定建立，Strategy Compare應在同一次auto-preparation內自動建立／接續，不應要求使用者先執行零散optimizer CLI。

### 修正

1. `selection_historical_p2` builder先搜尋既有completed Strategy Compare runs；若run的dataset、period、param policy、max positions、rotation均一致，且pair保存完整`no_filter_params`與原`params_file_sha256`，則以canonical P2 JSON writer重新序列化candidate。只有candidate SHA與舊run記錄**完全一致**且P2_HISTORY／rolling single-member／2014～2020 coverage contract全部通過時，才原樣恢復`selection_min_roos`。這條路徑不重新最佳化，也不從報表文字推測參數。
2. 若沒有任何可驗證completed pair可恢復，builder自動呼叫canonical`run_outer_rolling_oos`建立／接續historical baseline；輸出固定隔離到`models/research/breakout_quality/selection_strategy_realization/`，不得污染正式`models/roos_*.json`。
3. baseline optimizer的first/last OOS date由目前Strategy Compare期間傳入；train window、OOS horizon與baseline trials由`config/strategy_compare.py` builder options驅動，其中baseline trials直接引用`config.training_policy.OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT`，不新增trial magic number。
4. historical baseline的2014～2020、120m／12m canonical defaults抽到`filters/breakout_quality/trade_path_label.py`，11I audit與historical P2 service共用；Strategy Compare仍可由config顯式覆蓋，避免UI或validator硬編目前設定。
5. completed-pair recovery失敗時才進baseline＋P2 optimizer fallback；任何hash、period、policy或member contract不符均不得採用近似恢復。

### Scientific identity

MR-12B、MR-13A、DL-CONT12B-PIT、DL-CONT13A-PIT、SR-C23～C28的模型、Target、selector、historical P2定義、risk search fields、comparison period與策略會計均未改變；本輪只完成原有Strategy Compare auto-preparation dependency chain，不新增MR／SR ID。C27/C28 Selection結果仍待正式執行。


## 2026-08-09 — Stage 3 Strategy Compare Selection baseline programmatic argv 修正

### 狀態

`INFRASTRUCTURE_FIX / NO_SCIENTIFIC_CONDITION_CHANGE / FORMAL_SUITE_PRE_FIX_PASS / SELECTION_COMPARE_STILL_PENDING`。

### 程式基準與使用者錯誤

- 本輪來源ZIP：`test-branch-1_20260809_215948_7b98b09.zip`；SHA256=`b8b121d4e79d832375293e4ef8cba5a81ce6420cb595210aa1c581989b28eb97`。
- 使用者在 Strategy Compare 自動 BUILD `param:selection_min_roos` 時，console 已正確顯示 `2014-01-01~2020-12-31 / train=120m / oos=12m / trials=300/fold`，但 canonical outer-rolling service 隨即拋出 `ValueError: first OOS date 不可晚於 last OOS date`。
- 同一來源基準的正式 local regression bundle `to_chatgpt_bundle_20260809_220143_9cdf2b12.zip` 已全部 PASS（quick gate / consistency / chain checks / ml smoke / meta quality），表示此錯誤位於未被既有 formal case 覆蓋的 programmatic outer-rolling argv 邊界。

### 根因

`run_outer_rolling_oos()` 同時支援真實CLI `sys.argv` 與內部service呼叫。既有 `_extract_cli_value()` / `_has_cli_flag()` 固定從 `argv[1]` 掃描，假設 `argv[0]` 一定是program path；但 `strategy_param_training.py`、`strategy_optimizer_policy.py` 與既有 strategy adaptation service 都以 option-only list 呼叫，第一個token即 `--outer-first-oos-date`。因此 Selection baseline 的 first-OOS option 被忽略，first date回退到較晚的 base policy，而 last date仍讀到2020，形成反向區間。

### 修正

1. outer-rolling argv parser依第一個token是否為option判定掃描起點：option-only programmatic argv從index 0掃描，真實CLI argv仍從index 1掃描。
2. `_extract_cli_value()` 與 `_has_cli_flag()` 共用相同起點語意，避免value option與flag option再度分叉。
3. B179 synthetic contract新增雙路徑回歸：同一2014～2020 schedule分別以option-only argv與含program path的CLI argv解析，兩者必須得到相同 first/last OOS、120m train、12m OOS與trial數；fallback policy故意設為2021，確保第一個option若再被漏讀會立即失敗。
4. 未修改MR-13A模型、PIT、C27/C28 selector、historical P2定義、optimizer objective、trial數、strategy accounting或comparison period。

### Scientific identity

MR-12B、MR-13A、DL-CONT12B-PIT、DL-CONT13A-PIT、SR-C23～C28均不變；C27/C28 Selection Strategy Compare仍待正式執行。


## 2026-08-09 — Min ROOS五欄單階段Rolling與trial單一真理修正

### 狀態

`PARAMETER_SEMANTICS_CORRECTION / OLD_FOUR_ATR_P2_SUPERSEDED / SELECTION_COMPARE_REQUIRES_RERUN`。

### 使用者確認的canonical Min ROOS定義

- Min ROOS每fold只最佳化`high_len`、`atr_len`、`atr_buy_tol`、`atr_times_init`、`atr_times_trail`五個欄位。
- Rule-based entry filters固定全關，DL-off的P2／Selection baseline不使用DL作訓練決策；其餘optimizer維度由canonical config/schema固定，不再從另一輪完整ROOS結果繼承。
- Outer Rolling未顯式CLI override時，trials/fold唯一來源為`config.training_policy.OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT`；`config/strategy_compare.py`不得另設200等第二套預設。

### 根因

舊P2實作沿用早期risk-only研究契約：先建立historical full baseline，再把`high_len`與所有非ATR欄位凍結，只重新搜尋4個ATR欄位；Strategy Compare又另外硬編`trials_per_fold=200`，因此Selection auto-preparation實際出現一次300-trial完整Outer Rolling，再出現一次200-trial risk-only Outer Rolling。這同時違反目前Min ROOS定義與training-policy單一真理。

### 修正

1. `MIN_ROOS_SEARCH_FIELDS`固定為`high_len`＋4個ATR欄位；fold fixed overrides只固定其餘canonical optimizer維度。
2. Selection Min ROOS移除完整historical baseline前置依賴；2014～2020直接執行一次120m train／12m OOS的Outer Rolling建立active schedule。
3. `min_roos`、`min_dl_tp1_roos`、`min_dl_a9_roos`與`selection_min_roos`全部直接引用`OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT`；移除Strategy Compare的200 magic number與`baseline_trials_per_fold`雙重policy。
4. artifact mode改為`min_roos_training`並保存五欄`search_fields`；parameter-training schema與Strategy Compare schema同步升版。舊4-ATR／`high_len`凍結工件及其completed-pair cache不得恢復成current Min ROOS。
5. B187／B189 synthetic contract同步驗證五欄search、single-stage Selection preparation、training-policy trial SSOT及舊artifact rejection。

### Scientific consequence

既有使用舊4-ATR P2的C23／C24／C25 Selection策略結果保留為歷史read-only，但不再是current五欄Min ROOS universe的合法comparator；C27／C28與MR-12B comparator必須在新Min ROOS schedule下同批重跑後才能形成新的Selection策略結論。模型權重、MR-12B／MR-13A Target、PIT score與selector本身未因本輪修改而改變。


## 2026-08-10 — Selection Min ROOS post-run OOS month-boundary與completed-artifact resume修正

### 現象

使用者以current五欄、單階段`PARAM-P2 / Min ROOS`完成Selection 2014-01-01～2020-12-31七個Outer Rolling folds後，正式輸出已寫入`models/research/breakout_quality/trade_path_label/a2_teacher_params/p2_dl_off_trained/active_params/roos_base_best.json`，但前置在post-run validation拋出`P2_HISTORY參數fold schedule不一致: last_oos_date`。訓練效能摘要顯示7/7 completed、total 01:27:32，故不得把問題誤判為optimizer失敗或要求重訓。

### Root cause

- Strategy Compare／Min ROOS request使用完整期間結束日`2020-12-31`；Outer Rolling本身是month-bucket engine，`_parse_oos_boundary()`會將同一月份canonicalize為`2020-12-01`並寫入active-param meta。
- `_validate_min_roos_params()`近期五欄單階段重構後以字串完全相等比較request meta與optimizer output，將同一2020-12月份誤判為不同fold schedule。
- `build_rolling_base_policy()`又直接把requested月底做`+ MonthEnd(1)`，會把2020-12-31錯推到2021-01-31；雖CLI last-OOS仍限制實際fold，但base policy日期語意已分叉。
- post-run validator在寫入`breakout_quality_param_adaptation=min_roos_training`前失敗，原resume判斷要求該stamp已存在，因此即使preflight identity與7-fold params都已完成，下次仍會再次啟動昂貴optimizer。

### 修正

1. Min ROOS與historical teacher的first／last OOS boundary統一按calendar month比較；train-window與OOS-horizon months仍精確比較，年度effective-date schedule另行完整驗證。
2. `build_rolling_base_policy()`先把first／last OOS正規化為month-start，再由last month取得同月MonthEnd，2014-01-01～2020-12-31 request正確維持2020-12-31，不再延伸至2021-01-31。
3. 新增completed-artifact resume：只有既有`rolling_preflight.json`的`runtime_identity_sha256`等於current contract時，才嘗試對既有`roos_base_best.json`執行完整schedule／trials／fixed overrides／五個search fields／舊adaptation排除驗證；全部通過即補寫current`min_roos_training` stamp並直接REUSE，不重新跑Outer Rolling。任何舊4-ATR／risk-only或其他不相容artifact仍會落回正式重訓。
4. T284新增post-run failure resume固定案例；T285重新以canonical end date動態轉month-start作actual fixture，避免expected／actual共用同一月底表示而漏測。

### 判定

這是`PARAM-P2 / Min ROOS`工程契約修正，不新增MR／DL／SR ID，不改五欄search space、training-policy trials、模型、PIT source、selector、交易會計或Strategy Compare scientific matrix。Registry status維持ACTIVE；已完成的七fold current artifact在套用修正後應直接重新驗證接續，而非再執行1:27:32的rolling optimization。

## 2026-08-10 — Strategy Compare PIT model-upstream ownership與預先阻擋修正

### 現象

current五欄Selection Min ROOS已完成並成功接續後，Strategy Compare重新規劃`DL-CONT12B-PIT`與`DL-CONT13A-PIT` top-level工件；planner仍標記checkpoint-only `BUILD`。正式執行`CONT12B_PIT`時，PIT builder在載入event-ranker data bundle前即因缺少`outputs/filters/breakout_quality/breakout_quality_v1/dataset_summary.json`拋出`FileNotFoundError`。同一Dataset summary也是MR-13A daily sample provider的source-selection／inventory真理，因此下一個`CONT13A_PIT`亦會遭遇同一上游缺件。

### Root cause

Strategy Compare先前只檢查PIT top-level score／manifest／audit是否缺失，以及是否配置`selection_pit_from_existing_folds` builder；沒有在計畫階段驗證checkpoint-only builder仍需要的model-work upstream。MR-12B event PIT重建需要canonical Dataset與Continuous Target以重建fold identity／score universe並重跑model Audit；MR-13A daily PIT雖不需要event Continuous Target artifact，仍需要canonical Dataset summary作source-selection／inventory真理。依`PROJECT_SETTINGS`，Strategy Compare可匯出既有模型推論工件但不得自行建立Dataset／Label／Target，因此原`BUILD → runtime FileNotFoundError`計畫不合法。

### 修正

1. `strategy_compare_preparation`在把Selection PIT列為checkpoint-only `BUILD`前先檢查model upstream：所有PIT source都要求canonical Dataset summary及核心Dataset artifacts；event-group profile另外要求對應Continuous Target manifest。缺任一項即在計畫階段標`BLOCKED`，並顯示專案相對缺件路徑，不再等執行後失敗。
2. 模型訓練工作類型新增泛化選單「準備策略比較所需模型工件」。來源集合只由`config/strategy_compare.py`目前enabled arms解析，不提供model ID手動選單、不硬編MR/C編號。
3. 該模型工作流程先以既有canonical Dataset refresh／Continuous Target preparation服務建立model upstream，再對每個設定中的Selection PIT source執行`--checkpoint-only --resume`重建與PIT Audit；因此可建立Label／Target，但任何fold若缺少相容checkpoint仍立即停止，絕不因策略比較需求訓練模型權重。
4. Strategy Compare本身仍不建立Dataset、Label、Target或模型權重；完成模型工作類型準備後再回Strategy Compare即可依原dependency planner繼續。

### Scientific identity

MR-12B、MR-13A、DL-CONT12B-PIT、DL-CONT13A-PIT、五欄Min ROOS、C23～C28 selector／comparison matrix、Target公式、model weights與交易會計均未修改。本輪只修正跨工作類型的dependency ownership、preflight與既有checkpoint推論工件恢復路徑。

## 2026-08-10 — Strategy Compare required PIT missing-fold model-work ownership修正

### 現象

Strategy Compare在model upstream已重建後，以checkpoint-only方式重建`DL-CONT12B-PIT`；`fold_20110101_20111231`沒有可重用score／checkpoint或identity不相容，因此依安全契約停止。錯誤訊息要求改由模型研究入口重建Selection PIT，但前一輪新增的模型訓練「準備策略比較所需模型工件」入口本身仍固定傳入`--checkpoint-only`，而active model-research profile為MR-13A，一般PIT選項只會處理active profile，造成MR-12B缺失fold沒有正式UI可補齊。

### Root cause

跨工作類型ownership只完成了一半：Strategy Compare禁止train是正確的，但config-driven model-prerequisite workflow被錯誤沿用了相同checkpoint-only限制。Selection PIT fold checkpoint本身就是PIT no-lookahead模型生命周期的一部分；在模型訓練工作類型中，對缺失／不相容fold進行訓練是合法且必要的model work，並不等於Strategy Compare自行訓練，也不等於重訓forward-OOS anchor。

### 修正

1. Strategy Compare的`selection_pit_from_existing_folds` builder維持`--checkpoint-only --resume`；任何fold需要training仍立即停止，不跨工作類型。
2. 模型訓練的config-driven「準備策略比較所需模型工件」改為`--resume`而不帶`--checkpoint-only`。每個required PIT source仍由`config/strategy_compare.py` enabled arms自動解析，不提供MR／DL手動選單。
3. PIT builder既有resume順序不變：先重用完整fold，再用相容checkpoint重評score universe，再遷移compatible legacy fold；三者皆不可用時才進`_train_fold()`。因此本案例只會補訓真正缺失／不相容的MR-12B Selection PIT fold，其他相容fold保持reuse。
4. 該模型工作流程仍先由model-work服務準備Dataset／Target，PIT完成後重跑PIT Audit；不因Strategy Compare需求重訓forward-OOS anchor。
5. checkpoint-only失敗訊息改為指向正式模型訓練的泛化準備入口，避免要求使用者切換active profile或執行零散CLI。

### Scientific identity

MR-12B、MR-13A、DL-CONT12B-PIT、DL-CONT13A-PIT、五欄Min ROOS、C23～C28 selector／comparison matrix、Target、loss、PIT fold no-lookahead規則及策略會計均未改變。本輪只修正缺失PIT fold的正式工作類型ownership與可達性，不新增MR／DL／SR ID。


## 2026-08-10 — MR-13A daily PIT Selection strategy translation結果：C28通過Forward-OOS Gate

### 狀態

`RESULT_AVAILABLE / SELECTION_STRATEGY_GATE_PASS / C27_NOT_PROMOTED / C28_FORWARD_OOS_AUTHORIZED / NO_SELECTION_RETUNING`。

### 程式基準與比較契約

- 程式基準：`test-branch-1_20260810_100406_9cdd5aa.zip`；SHA256=`6b6a9a122569ee994ff26ee3134147e221c4f020673725f5686765583aaebd62`。
- Strategy Compare run：`outputs/strategy_compare/runs/20260810_123402_C23-C24-C25-C27-C28_ffefa86f0a89`；期間=`2014-01-01～2020-12-31`、Dataset=`full`、param policy=`base-finalist-best`、max positions=`10`、rotation=`off`。
- 全部arms共用current五欄`PARAM-P2 / Min ROOS` historical active params；五欄為`high_len + atr_len + atr_buy_tol + atr_times_init + atr_times_trail`。Rules全關，C23為DL-off baseline。
- C24/C27固定minimum-repair selector；C25/C28固定feasible-ascent selector。C27/C28唯一model/source差異是`DL-CONT12B-PIT → DL-CONT13A-PIT`；C28不繼承C26 stale-score guard，也沒有新增capital objective、threshold或OOS調參。

### 主要結果

| Arm | Return | MDD | RoMD | Annual | EV | Trades | Same-param selection R |
|---|---:|---:|---:|---:|---:|---:|---:|
| C23 Min ROOS baseline | 107.98% | 20.57% | 5.25 | 11.03% | 0.42R | 475 | 0.00R |
| C24 MR-12B minimum-repair | 82.98% | 18.99% | 4.37 | 9.02% | 0.50R | 420 | +12.19R |
| C25 MR-12B feasible-ascent | 101.55% | 19.66% | 5.17 | 10.53% | 0.52R | 432 | +26.19R |
| C27 MR-13A daily minimum-repair | 102.26% | 21.57% | 4.74 | 10.59% | 0.50R | 431 | +18.41R |
| C28 MR-13A daily feasible-ascent | 151.88% | 18.86% | 8.05 | 14.11% | 0.71R | 435 | +109.91R |

Controlled contrasts：

- `C27-C24`（固定minimum-repair、只換DL source）：Return `+19.27pp`、MDD `+2.57pp`、RoMD `+0.37`、Annual `+1.57pp`、EV `0.00R`、same-param selection R `+6.22R`。MR-13A source較MR-12B改善，但C27本身仍低於C23，因此minimum-repair不採用。
- `C28-C25`（固定feasible-ascent、只換DL source）：Return `+50.33pp`、MDD `-0.80pp`、RoMD `+2.89`、Annual `+3.58pp`、EV `+0.19R`、same-param selection R `+83.72R`。這是支持MR-13A daily source的主要model/source controlled evidence。
- `C28-C23`：Return `+43.90pp`、MDD `-1.71pp`、RoMD `+2.80`、Annual `+3.08pp`、EV `+0.29R`、selection R `+109.91R`。年度Return相對C23在2014/2015/2016/2017/2018/2020改善，6/7正向；只有2019低`-1.49pp`。
- `C28-C27`（相同daily source，只換selector）：Return `+49.62pp`、MDD `-2.71pp`、RoMD `+3.31`、Annual `+3.52pp`、EV `+0.21R`、selection R `+91.50R`。Feasible-ascent對daily source的portfolio translation非常重要，minimum-repair不足以通過baseline gate。

### Selection診斷與判讀

- C28的selected Target mean相對C23由`0.4769R → 0.6832R`（`+0.2063R`），但同日Target percentile由`0.5925 → 0.5260`、opportunity gap由`3.4957R → 4.1697R`惡化。這不構成矛盾：報表先在每日候選集合內計算percentile／top-k，再做daily mean；absolute Target R與within-day percentile衡量不同面向。C28不是在每一天都更接近future Target oracle，而是在不使用Future Target的前提下，選到的絕對opportunity-R與後續實際portfolio economics更好。
- `same-param selection R`是DL-on與同param/rule baseline之exclusive trades R差；C28的`+109.91R`本身不能代替portfolio PnL。但本次與舊C26不同：C28同時有Return、MDD、RoMD、Annual、EV的實際portfolio改善，因此Selection R優勢確實有轉成策略經濟效果。
- C28只有`108`個實際改單日、`12`個feasible-ascent改善日，卻產生大幅wealth-path差異；這是portfolio path敏感性的證據，不是可用來繼續調selector的理由。現有hard feasibility仍維持K/R0、reserved-capital floor，Max-DL K violation=`0`。

### 判定與下一步

1. `SR-C27`：`NOT_PROMOTED`；不進Forward-OOS。
2. `SR-C28`：`SELECTION_GATE_PASS / FORWARD_OOS_AUTHORIZED / FROZEN_SELECTOR`。
3. `MR-13A`：Selection strategy translation已PASS，但尚不得宣告正式取代`MR-12B / DL-CONT12B`；formal anchor promotion必須等Forward-OOS strategy Gate。
4. 保留`DL-CONT13A`作MR-13A frozen Forward-OOS daily score source，保留`SR-C29`作C28語意凍結後的Forward-OOS candidate。Forward對照應只保留current五欄Min ROOS DL-off baseline與相同feasible-ascent selector的MR-12B arm，用來分離portfolio baseline與純DL source效果。
5. 不再於Selection調feasible-ascent、stale cutoff、score cutoff、capital objective、fixed risk或模型；不得把C26的22-day guard帶進C29。下一個可執行工作是2021+ Forward-OOS controlled replay。

## 2026-08-10 — MR-13A Forward-OOS strategy Gate：Min/Full × DL-off/MR-12B/MR-13A 六arm矩陣實作

### 狀態

`IMPLEMENTED / RESULT_PENDING / NO_FORWARD_OOS_RETUNING`。

### 程式基準與研究動機

- 程式基準：`test-branch-1_20260810_124927_509c6ca(2).zip`；SHA256=`a67d850fe07bda0037b33c61b330de1482c790a8229b638dbd063c9089e08fe9`。
- Selection已證明`SR-C28 = current五欄Min ROOS + DL-CONT13A-PIT + feasible-ascent`通過策略Gate；同時current Min ROOS語意已修正為每fold只搜尋`high_len + atr_len + atr_buy_tol + atr_times_init + atr_times_trail`，不再是舊4-ATR／high_len凍結P2。
- 使用者要求下一階段不再帶minimum-repair，改以已多次較佳且Selection已凍結的feasible-ascent，同時加入Full ROOS，用2×3矩陣驗證MR-13A優勢是否只存在Min ROOS，或能泛化到完整策略體系。

### 六個正式arms

| Arm | 參數／rules | DL source | Selector | 狀態 |
|---|---|---|---|---|
| `SR-C3` | current五欄Min ROOS／all-off | off | baseline | enabled baseline |
| `SR-C1` | Full ROOS／formal | off | baseline | enabled baseline |
| `SR-C20` | current五欄Min ROOS／all-off | `DL-CONT12B` | frozen feasible-ascent | current retest |
| `SR-C29` | current五欄Min ROOS／all-off | `DL-CONT13A` | frozen feasible-ascent | new Forward-OOS candidate |
| `SR-C30` | Full ROOS／formal | `DL-CONT12B` | frozen feasible-ascent | new Full comparator |
| `SR-C31` | Full ROOS／formal | `DL-CONT13A` | frozen feasible-ascent | new Full candidate |

`DL-CONT13A`直接重用已完成`MR-13A / PROFILE-daily_universal_no_time_pairwise` frozen Forward-OOS score；Strategy Compare不選模、不訓練模型，也不因本次矩陣改Target／loss／architecture／weights。

### Resource contract泛化

既有max-DL／feasible-ascent runtime演算法本身每次都先以當前pair的DL-off rows與params執行canonical exact-reservation baseline，但manifest／error wording仍稱為`Min ROOS baseline`。為避免Full arms形成「Full params + Min資源底線」的錯誤解讀，本輪把正式契約明確泛化為same-param baseline：

- 每個`param_source/rule_policy` group只建立一個DL-off baseline。
- `K = same-param baseline selected_count`。
- `R0 = same-param baseline exact reserved capital`。
- DL Top-K／minimum-repair seed／feasible-ascent只能在`selected_count == K`且`reserved_cost >= R0`的hard-feasible集合內改善continuous score。
- basket內execution order沿用該same-param DL-off baseline rank；capital只作feasibility，不進DL objective。
- Min arms因此使用Min ROOS自己的K/R0；Full arms使用Full ROOS自己的K/R0，禁止跨param source借用Min baseline。

對應manifest文字改為`same_param_exact_resource_baseline`、`canonical_same_param_exact_cash_cap_baseline`等same-param語意；演算法、sizing、accounting與exact reservation公式不變。

### 比較期間與controlled contrasts

`STRATEGY_COMPARE_START_DATE/END_DATE`皆留空，由啟用的`DL-CONT12B`與`DL-CONT13A`正式Forward-OOS runtime coverage交集自動決定共同期間；不得把任一source較長尾端單獨算進source-only contrast。

啟用contrasts：

1. `C20-C3`：Min下MR-12B相對DL-off。
2. `C29-C3`：Min下MR-13A相對DL-off。
3. `C29-C20`：固定Min＋feasible-ascent的純DL source效果。
4. `C30-C1`：Full下MR-12B相對DL-off。
5. `C31-C1`：Full下MR-13A相對DL-off。
6. `C31-C30`：固定Full＋feasible-ascent的純DL source效果。
7. `C1-C3`：Full相對Min完整策略體系差異；不是單一參數／rule效果。
8. `C31-C29`：固定MR-13A＋feasible-ascent下Full vs Min完整策略體系interaction；`direct_selection_delta_r`依既有contract不得跨param/rule universe相減。

### 判定規則

- 主要model/source Gate看`C29-C20`與`C31-C30`是否方向一致，並同時檢查Return、MDD、RoMD、Annual、EV、年度穩定性與same-param selection R。
- `C29-C3`與`C31-C1`回答DL + selector是否能勝過同一策略體系DL-off baseline。
- `C1-C3`與`C31-C29`只回答Min/Full完整體系interaction，不可解讀成Full optimizer或rule filters的純因果效果。
- 本輪結果取得前不得依Forward-OOS調feasible-ascent、score cutoff、stale guard、capital objective、Min/Full search space或模型；`SR-C27` minimum-repair不回到本矩陣。

### GPT獨立focused驗證

- `validate_strategy_compare_config_driven_app_contract_case`：47 checks / 0 fail；新增same-param baseline與Full-row feasible-ascent contract。
- `validate_breakout_quality_daily_pit_strategy_runtime_contract_case`：8 checks / 0 fail；保留Selection C27/C28 frozen identity，並驗證`DL-CONT13A` Forward-OOS source與六arm／八contrast矩陣。
- 正式`apps/test_suite.py`未由GPT執行；仍須由使用者本地formal double check。

## 2026-08-10 — Forward-OOS Strategy Compare模型前置自動準備路由補全

### 狀態

`INFRASTRUCTURE_FIX / CONFIG_DRIVEN_MODEL_PREREQUISITE_ROUTING_COMPLETE / SCIENTIFIC_IDENTITY_UNCHANGED`。

### 現象

Min/Full × DL-off/MR-12B/MR-13A六arm Forward-OOS矩陣切換到`DL-CONT12B`與`DL-CONT13A`後，Strategy Compare狀態頁正確檢查出`model.pt`／manifest／continuous ranker report／forward scores缺失並標為`BLOCKED`；`param:min_roos`仍可依config自動`BUILD`。然而模型訓練工作類型既有的「準備策略比較所需模型工件」只解析`selection_point_in_time` sources，沒有解析`continuous_ranker_oos` sources，造成新的Forward-OOS矩陣雖有正式模型準備入口名稱，實際卻無法由該入口補齊MR-12B／MR-13A主模型工件。

### Root cause

跨工作類型自動前置的source resolver仍使用先前Selection PIT階段留下的`_strategy_compare_required_selection_pit_sources`，只挑Selection PIT score source。六arm矩陣改用Forward-OOS continuous sources後，enabled arms需要的`DL-CONT12B`／`DL-CONT13A`沒有進入model-work preparation queue。Strategy Compare本身禁止train模型權重是正確契約，錯誤在於模型工作類型的泛化準備入口沒有涵蓋新的score-source種類。

### 修正

1. 模型訓練工作類型改以`_strategy_compare_required_model_sources`解析目前enabled arms的所有正式model sources，目前涵蓋`selection_point_in_time`與`continuous_ranker_oos`，不硬編MR／DL ID。
2. `continuous_ranker_oos`若完整model／manifest／report／scores identity一致，直接`REUSE`；缺少或不相容時，由**模型訓練工作類型**先準備Dataset／Continuous Target（daily profile依既有契約不建立event target），再執行該profile的canonical `train-continuous-ranker`，完成後重新驗證Forward-OOS contract。
3. Selection PIT仍維持既有`--resume` lifecycle：相容fold重用，缺失／不相容fold只在模型工作類型內補訓。
4. Strategy Compare本身仍不執行`train-continuous-ranker`，缺模型權重時維持`BLOCKED`；錯誤／狀態文字改為明確指向`[模型訓練] → 準備策略比較所需模型工件`。因此「自動前置=on」只代表同工作類型可安全確定建立的工件會自動補，不代表Strategy Compare可跨權限訓練模型。
5. `PARAM-P2 / Min ROOS` builder、五欄search、300 trials SSOT、六arm定義、feasible-ascent、Target、loss、architecture、weights與策略accounting均未改變。

### 程式基準

- 輸入基準：`test-branch-1_20260810_132241_b8c1f5e.zip`；SHA256=`b82535e73d58eddb8eba23c0696eb96ffd50dce2c9ba13d2a97d6981a5ceb242`。
- 本輪為workflow／ownership修正，不新增`MR-*`、`DL-*`或`SR-C*` scientific ID。

## 2026-08-10 — MR-13A Forward-OOS daily score artifact contract修正

### 狀態

`INFRASTRUCTURE_FIX / PROFILE_AWARE_FORWARD_SCORE_CONTRACT / EXISTING_MR13A_MODEL_REUSABLE / SCIENTIFIC_IDENTITY_UNCHANGED`。

### 現象與程式基準

- 輸入基準：`test-branch-1_20260810_141107_cb605c7.zip`；SHA256=`8217707a996c542f248418a5d32ad37b1c55b5af6c2c741a707a6b59cce4751f`。
- 使用者由`[模型訓練] → [5] 準備策略比較所需模型工件`成功補建`CONT12B`與`CONT13A`。MR-12B event ranker輸出`continuous_ranker_scores.csv`；MR-13A daily ranker訓練亦PASS並正式輸出`daily_ranker_oos_scores.csv.gz`。
- MR-13A訓練完成後的再次Forward-OOS contract驗證仍固定尋找`continuous_ranker_scores.csv`，因此拋出`FileNotFoundError`。模型、report與daily OOS scores本身已完成，失敗發生於下游artifact resolver／validator。

### Root cause

`load_continuous_ranker_oos_contract()`與`load_continuous_ranker_oos_score_table()`仍把所有`continuous_ranker_oos` source視為event-ranker schema：固定檔名`continuous_ranker_scores.csv`、固定要求`split`欄再過濾`split=oos`，且只認manifest/report的`scores`記錄。MR-13A trainer從設計起即使用daily專屬canonical artifact：`daily_ranker_oos_scores.csv.gz`，整份檔案就是OOS daily stock-day universe，manifest/report key為`oos_scores_gzip`，沒有event-only `split`欄。

### 修正

1. 新增唯一`resolve_continuous_ranker_oos_score_path()`：依experiment profile的`training_sample_scope`解析Forward-OOS score真理路徑；event維持`continuous_ranker_scores.csv`，daily使用`daily_ranker_oos_scores.csv.gz`。模型狀態頁、Strategy Compare preflight與runtime loader共用此resolver。
2. `load_continuous_ranker_oos_contract()`改為profile-aware驗證manifest/report：daily使用`oos_scores_gzip` file identity與現有daily report `sample_scope`／pairwise contract；event原有training semantics、label scope與`scores` identity驗證不放寬。
3. `load_continuous_ranker_oos_score_table()`對event仍要求`split`且只取`oos`；daily不要求`split`，直接驗證整份gzip中的`ticker/date/group_index/model_score`、0～1有限分數與ticker/date唯一性。Runtime仍依daily profile使用最新已完成交易日`information_date`查分，不改無前視語意。
4. 為未來新daily artifact補上`training_label_scope`與`training_semantics`共通metadata；loader仍接受本次已訓練、尚未包含這兩個新欄位的既有MR-13A artifact，因此**不需重新訓練CONT13A**。

### Scientific identity

MR-13A architecture、daily sample universe、target、RankNet loss、selected epoch、模型權重、Forward-OOS scores內容、Selection PIT、feasible-ascent、Min/Full 2×3矩陣、ROOS參數與策略accounting全部不變。本輪只修正trainer與consumer之間的Forward-OOS artifact schema／path契約，不新增`MR-*`、`DL-*`或`SR-C*`。

### GPT focused驗證

- synthetic MR-13A trainer-format artifact（gzip、無`split`、`oos_scores_gzip` identity）可被Forward-OOS contract直接載入，2 rows完整保留。
- `validate_breakout_quality_daily_pit_strategy_runtime_contract_case`：9 checks / 0 fail，新增daily Forward-OOS gzip contract直接驗證。
- `validate_strategy_compare_config_driven_app_contract_case`：48 checks / 0 fail。
- `validate_breakout_quality_listwise_ranker_contract_case`：8 checks / 0 fail，確認event continuous/listwise原有CSV＋split契約未被daily分支破壞。
- 正式`apps/test_suite.py`未由GPT執行；仍由使用者本地formal double check。


## 2026-08-10 — Selection Full row與Strategy Compare雙階段profile固化

- 狀態：`IMPLEMENTED / SELECTION_FULL_ROW_RESULT_PENDING / FORWARD_PROFILE_PRESERVED`。
- 工作基準：`test-branch-1_20260810_153408_87b3b7c.zip`；SHA256=`518c43fcc0be91ff079e0354a60116555b0c9e37aa8a7d7e39f6601d7c1d757d`。
- 使用者要求在Full ROOS加入最終Forward-OOS矩陣後，先補對稱的Selection策略驗證；同時為避免日後再把Selection／Forward enabled arms互相覆寫，`apps/research.py → [3] 策略組合比較`改為兩個永久共存、config-driven profiles：`selection_pit`與`forward_oos`。兩者共用同一strategy comparison engine，但period、enabled arms/contrasts與output namespace隔離。
- 新增`PARAM-P4`：2014-01-01～2020-12-31 historical Full ROOS rolling active params，120m train／12m OOS，trials直接讀`config/training_policy.py`；使用canonical Full optimizer search space。TP、DL hard filter/ranking及History threshold維持current optimizer policy固定OFF，不因Selection結果新增trial維度。參數工件位於`models/research/breakout_quality/strategy_compare/selection_full_roos/`，不得用2021+ `full_roos` forward params倒灌Selection。
- 新增Selection Full row：`SR-C32`=Full DL-off baseline；`SR-C33`=`PARAM-P4 + DL-CONT12B-PIT + frozen feasible-ascent`；`SR-C34`=`PARAM-P4 + DL-CONT13A-PIT + frozen feasible-ascent`。C33/C34的K、R0與reserved-capital floor只取同參數／formal-rules的C32 baseline，不借用Min ROOS。Min row重用`C23/C25/C28`。
- Selection profile contrasts固定包含Min row既有`C25-C23/C28-C23/C28-C25`與Full row`C33-C32/C34-C32/C34-C33`，另保留`C32-C23`與`C34-C28`作完整策略體系interaction；核心新Gate為`C34-C33`純DL source與`C34-C32`Full baseline economics。不得依Selection結果再調feasible-ascent、score cutoff、stale guard、capital objective或模型。
- Forward profile完整保留既有2×3：`C1/C3/C20/C29/C30/C31`及其八個contrasts，不因補Selection Full row而重寫；Forward-OOS仍待Selection Full row結果後執行。
- 模型前置resolver改為跨所有Strategy Compare profiles去重解析依賴，因此模型工作類型可一次準備Selection PIT與Forward-OOS所需sources；Strategy Compare本身仍不得訓練模型權重。
- 為避免profile拆分後遺失既有C23/C25/C28與其他已完成pair cache，兩profile由config宣告唯讀`reuse_output_roots=("outputs/strategy_compare",)`；pair cache與historical P2 recovery可掃描legacy runs，但新run／latest只寫入各自profile root，因此舊工件可重用而未來Selection/Forward結果不再互相覆寫。
- 直接isolated builder驗證：`PARAM-P4`第一次建立會執行一次7-fold Full outer rolling；相同identity第二次直接REUSE，effective dates=`2014-01-01 ... 2020-01-01`，artifact stamp=`selection_full_roos_training / P4_HISTORY`。

## 2026-08-10 — Selection／Forward結果反轉後縮減為四arm核心比較

### 狀態

`CONFIG_REDUCTION / STANDALONE_FULL_BASELINE_SUPPORTED / MR13A_ITERATIVE_RESEARCH_FOCUS / HISTORICAL_FULL_DL_ARMS_PRESERVED`。

### 程式基準與研究結果

- 輸入基準：`test-branch-1_20260810_170110_ba0a90b.zip`；SHA256=`7464b0e2d63d80df597c75f3887931b01b9f336e4682eeb80b5c65d17dbc491d`。
- Selection 2014-01-01～2020-12-31：C23 Min baseline Return=107.98%、RoMD=5.25；C25 Min+MR-12B=101.55%/5.17；C28 Min+MR-13A=151.88%/8.05；C32 Full baseline=132.25%/8.34。歷史Full+DL結果C33=146.25%/7.73、C34=167.12%/9.06永久保留。
- Forward-OOS 2021-01-01～2025-12-22：C1 Full=123.26%/RoMD7.08；C3 Min=86.21%/6.57；C20 Min+MR-12B=172.40%/9.20；C29 Min+MR-13A=71.86%/4.57。C29-C20純DL source Return=-100.54pp、RoMD=-4.62、EV=-0.62R、same-param selection R=-204.13R。歷史Full+DL結果C30=133.21%/7.43、C31=122.00%/7.53永久保留。
- Selection支持MR-13A而Forward-OOS明顯反轉，因此MR-13A不升級；使用者仍要持續改善MR-13A，current comparison改聚焦同一Min universe下MR-12B vs MR-13A的純source差異，Full只保留完整策略體系DL-off baseline。

### Current Strategy Compare profiles

Selection PIT active arms縮減為：

1. `C32` Selection Full ROOS DL-off baseline。
2. `C23` Selection Min ROOS DL-off baseline。
3. `C25` Selection Min + MR-12B feasible-ascent。
4. `C28` Selection Min + MR-13A feasible-ascent。

Selection active contrasts固定為`C32-C23 / C25-C23 / C28-C23 / C28-C25`。

Forward-OOS active arms縮減為：

1. `C1` Full ROOS DL-off baseline。
2. `C3` Min ROOS DL-off baseline。
3. `C20` Min + MR-12B feasible-ascent。
4. `C29` Min + MR-13A feasible-ascent。

Forward active contrasts固定為`C1-C3 / C20-C3 / C29-C3 / C29-C20`。

`C33/C34/C30/C31`不刪除、不改ID、不覆寫結果；只從current profile移除，仍可由Registry與legacy cache重現歷史Full+DL結果。

### Standalone baseline infrastructure

舊Strategy Compare engine假設每個啟用的DL-off `param_source/rule_policy` group都至少要有一個DL-on arm，因此單獨保留Full ROOS baseline會被config validator拒絕。此假設不是科學契約，只是舊controlled-pair orchestration限制。

本輪泛化為：

- DL-on arm仍必須存在同group DL-off baseline。
- DL-off arm可以作standalone comparator，不要求為了engine schema而暗中啟用DL-on arm。
- standalone baseline仍使用同一canonical replay、param/rule policy、accounting與baseline artifact schema；可從profile或legacy compatible pair重用已驗證的`no_filter` baseline，否則只執行DL-off replay一次。
- standalone baseline輸出仍包含`strategy_comparison.md/json`、年度報酬、equity、trades、capacity、orderable與selected artifacts，可直接進四arm彙總報表。
- 此改動只放寬orchestration topology，不改Full/Min params、MR-12B/MR-13A score、feasible-ascent、K/R0、Target、loss或策略accounting，不新增scientific ID。

### 研究邊界

2021～2025 Forward-OOS已被用來判斷MR-13A失敗並形成後續改善假說，因此後續MR-13*可將它當`iterative research OOS evidence`比較泛化，但loss weight、sampling比例、epoch selection或其他超參數仍只能由Train/Validation/Selection決定；不得用該Forward期間直接挑數值。Current四arm設計的目的就是降低比較維度，持續保留Full/Min策略基準，同時把MR-13A改善的主要source Gate固定在`C28-C25`與`C29-C20`。

## 2026-08-10 — Strategy Compare核心四arm顯示名稱統一

- 狀態：`IMPLEMENTED / DISPLAY_ONLY / NO_REPLAY_SEMANTICS_CHANGE`
- 程式基準：`test-branch-1_20260810_193803_f49d873.zip`；SHA256 `c90dca664e3049c3db551a71717f62bd39f3f8506a70c1f37d90201ca3fa95a3`。
- 唯一變更：Selection PIT與Forward-OOS兩個Strategy Compare profile共用同一套核心顯示名稱：`Full ROOS`、`Min ROOS`、`Min MR-12B`、`Min MR-13A`。研究階段由profile頁首標示；MR-12B／MR-13A的`feasible-ascent`與MR-13A daily score語意保留於arm description／runtime contract，不再放入比較對象名稱。
- 固定條件：C/SR ID、param source、DL source、Selection PIT／Forward-OOS period、Min ROOS五欄、Full ROOS、feasible-ascent、K/R0、Dataset、Target、模型、策略執行與所有既有結果均不變。
- Cache：pair cache identity本來只包含replay contract，不包含`name`／`description`；本輪顯示改名不得使既有C25/C28/C20/C29 pair失效。
- Dataset／Label：不重建、不relabel。
- 驗證：`validate_strategy_compare_config_driven_app_contract_case`新增不硬編arm ID／名稱的跨profile顯示一致性guard；目前51項0 fail。pair cache fingerprint對display-only `name`／`description`變更保持不變。
- 結果：本輪不產生新的Selection／Forward-OOS科學結果；既有數值維持原紀錄。

## 2026-08-10 — Multiple-seed strategy robustness infrastructure

- 狀態：`IMPLEMENTED / RESULT_PENDING`；尚未產生任何multi-seed實驗結論，不建立新`MR-*`／`DL-*`／`SR-C*`。
- 正式入口：`apps/research.py` → `策略組合比較` → `Multiple-seed robustness`；設定整合於`config/strategy_compare.py`。
- 比較對象由目前Strategy Compare arm的`robustness_role`驅動，不另寫active C-ID清單：Full ROOS／Min ROOS為`fixed_baseline`，Min MR-12B／Min MR-13A為`stochastic`。
- Seed policy：只設定`seed_count`與`seed_generator_seed`，由deterministic generator產生可重現的unique seeds；報表不指定seed 42、不挑best seed、不做seed ensemble。
- 執行：單一GPU training queue與CPU strategy replay queue重疊；console持續顯示seed序號、比較對象序號、training/replay與總耗時。模型訓練仍呼叫canonical continuous-ranker trainer，但以隔離`model-output-dir`／`research-output-dir`工作目錄避免覆寫正式模型。
- 報表：表一同列Full／Min fixed baselines與stochastic arms各項final-strategy metric Mean（Return、MDD、RoMD、Annual、EV、Payoff、Exposure、Trades、Win Rate、Monthly Win Rate、Log R²、同參數DL選擇R）；表二只列RoMD完整分布（Mean、Median、Std、CV、Min、P25、P75、Max、勝Min／Full counts及cross-seed distribution comparison）。
- Retention：永久輸出固定為`outputs/strategy_compare/robustness/<fingerprint>/`下`manifest.json`、`seed_results.csv`、`robustness_summary.json`、`robustness_report.md`；checkpoint／full scores／replay details預設清除，可由config retention flags顯式保留。fingerprint包含實際共同OOS期間、dataset inventory、策略參數artifact identities、experiment profile semantics、有效training defaults、seed policy與runtime selector contract。


## 2026-08-10 — Selection PIT / Forward-OOS完整語意稽核：Forward score universe future-target依賴修正

### 工作基準

- 使用者指定最新版：`test-branch-1_20260810_212258_a5289fd.zip`
- SHA256：`72c474dbb2d7477c9303490970c0d025b4bbb669609fbe92423e2fdcbf15eda4`
- 本輪以fresh-unzip為唯一修改基準，保留其後新增的Selection／Forward雙profile、current四arm矩陣、daily Forward artifact schema與Multiple-seed robustness isolation；不得用較舊patch整檔覆蓋。

### 稽核結論

Selection PIT本身維持正確：PIT score eligibility只由當時可得feature/benchmark history決定，future Target完整性只控制train／validation／final-refit與事後audit；event profile仍查breakout signal date，daily profile仍於每個盤前decision查最新已完成交易日information-date score。

Forward-OOS模型訓練／model Gate亦沒有把OOS帶入gradient、epoch selection或final refit；但MR-12B event與MR-13A daily trainer在輸出Forward runtime score時仍直接使用target-valid OOS IDs，導致「某日是否有score」取決於future 40-bar Target是否已完整。舊Forward共同期間因此在2025-12-22附近被Target horizon截尾；這是runtime score coverage的no-lookahead違規，不是frozen checkpoint本身失效。

### 修正

1. `ranker_sample_contract.py`新增共用Forward SSOT：prediction-time inference universe只依OOS日期與當時feature history；target-evaluable universe為其`target_valid` subset。
2. event／daily trainer的OOS model metrics仍只使用target-evaluable subset；runtime score artifact改涵蓋完整inference universe。Target未完成rows仍輸出`model_score`，但`target_raw_r`／`target_daily_percentile`固定為NaN。
3. Manifest／report新增`score_eligibility_contract`與`forward_score_coverage`，明確保存inference groups、target-evaluable groups與`future_target_required_for_score=False`。Forward loader要求完整契約；舊target-complete-only score artifact不得直接作current strategy source。
4. 新增`rebuild_forward_oos_scores.py` frozen-checkpoint score-only服務：驗證model/profile/objective/spec/input shape/selected epoch/Dataset/Target identity後，不做fit，只重建Forward scores與report/manifest。模型工作類型先走此路徑，只有checkpoint identity不安全才退回canonical full training；Strategy Compare本身仍禁止train。
5. `compare_continuous_rankers.py`改為允許runtime score table尾端Target=NaN，模型品質比較只在target-evaluable subset計算，避免為了診斷再次截短runtime universe。
6. 保留最新版Multiple-seed robustness的`--model-output-dir`／`--research-output-dir` isolated training與`score_path_override`；本輪沒有改seed生成、final-strategy統計或profile matrix。

### 研究狀態影響

- `DL-CONT12B-PIT`／`DL-CONT13A-PIT`與Selection C23/C25/C28結果維持current，不需因本修正重跑Selection PIT。
- `DL-CONT12B`／`DL-CONT13A` frozen model checkpoints與既有Forward model Gate仍有效，不因score coverage修正重訓權重。
- 舊C20/C29以及舊2021-01-01～2025-12-22 Forward contrasts降級為`PRE_FIX_RESULT_ARCHIVED`，不得再用來宣告MR-13A Forward fail或MR-12B current strategy anchor。
- 下一個正式動作是由模型工作類型以frozen checkpoint重建future-independent Forward scores，再以同一五欄Min ROOS＋frozen feasible-ascent重跑C20/C29 controlled replay；新結果出來前MR-13A維持`PROMOTION_PENDING_RECHECK`。

### Scientific identity

本輪不新增MR／DL／SR ID，不改Target、loss、architecture、selected epoch、Min/Full參數、selector、K/R0、feasible-ascent、交易會計、Strategy Compare雙profile或Multiple-seed robustness scientific design；只修正Forward score availability的時間因果契約與score-only artifact repair path。

## 2026-08-10 — MR-13A daily Forward strategy replay score lookup效能修正

### 現象

- 使用者執行`Forward-OOS策略比較`；C1 Full baseline、C3 Min baseline與C20 MR-12B replay可在數秒至數十秒完成，但最後C29／MR-13A daily `score_ranking`在`2021-02-01`、僅第21/1248日時已耗時`00:01:51`。
- 這不是模型重新inference、GPU訓練或feasible-ascent本身耗時。C29 runtime依daily information-date契約會對候選每日刷新score，因此lookup次數本來就高於event source；真正異常來自每次lookup又對完整daily score MultiIndex執行`get_level_values("date").min()/max()`。

### Root cause

`lookup_continuous_ranker_oos_candidate_score()`雖已透過LRU cache重用整張score DataFrame，但仍在candidate內層迴圈每次重新掃描所有score rows取得`available_from/available_through`。MR-13A daily Forward table約為all-stock × all-OOS-days量級，故單次lookup從indexed key lookup退化成O(score rows)；daily refresh再把此成本乘上每天數十至上百候選。隔離同規模約69萬列MultiIndex重現：單次date min/max掃描約0.19秒，而直接indexed `(ticker,date)` lookup約0.00024秒，與實際21天約111秒的console進度一致。

### 修正

1. `load_continuous_ranker_oos_score_table_from_path()`在CSV/GZIP載入與date normalization時只計算一次`available_from/available_through`，保存於cached DataFrame attrs。
2. canonical normal route直接使用既有`ContinuousRankerOOSContract.available_from/available_through`；`score_path_override`／Multiple-seed robustness isolated score route使用相同cached table attrs，不再重新掃描MultiIndex。
3. key lookup由`key in index`再`loc`的雙查詢改為單次`table.loc[(ticker,date)]`並以`KeyError`表示missing score；score availability、daily information-date refresh、no-lookahead、selector與strategy accounting完全不變。
4. T294新增warm-cache performance-contract fixture：在禁止`pd.MultiIndex.get_level_values()`的情況下仍必須可由override score table成功查值，並驗證預先保存的score period metadata。

### Scientific identity

本輪只修Forward runtime score lookup複雜度，不改MR-12B/MR-13A模型、Forward score values、daily refresh timing、Min/Full參數、feasible-ascent、K/R0、交易會計、Selection PIT、Forward score-universe contract或Multiple-seed robustness scientific design；不新增任何MR/DL/SR ID，也不使既有pair result因identity改變而失效。重新執行C29應得到相同策略結果，只縮短wall time。

## 2026-08-10 — Formal consistency synthetic caller與Forward loader一次性I/O修正

### Formal bundle現象

- `quick gate`、`chain checks`、`ml smoke`皆PASS；`consistency`只有1個FAIL。
- `SYNTHETIC_SUITE`在啟動coverage synthetic時拋出`TypeError: _validate_audit_source_artifact() missing 1 required keyword-only argument: project_root`，因此synthetic case count為0。
- `meta quality`的coverage line／branch／key-target／critical-file六個FAIL均為同一次synthetic early-abort的連帶結果，不代表新增六個runtime regression。

### Root cause與修正

1. `ranking_score_store._validate_audit_source_artifact()`為符合使用者可見相對路徑契約已要求`project_root`；T280的Selection PIT audit source hash synthetic仍沿用舊signature。修正為使用該fixture自己的temporary artifact root，既驗證exact hash binding，也驗證修改後source必須被SHA mismatch拒絕；runtime validator參數與fail-fast語意不放寬。
2. 同輪全專案檢查發現`load_continuous_ranker_oos_score_table_from_path()`首次載入同一Forward score CSV/GZIP時重複呼叫reader兩次；另`load_continuous_ranker_oos_contract()`在table已保存`available_from/available_through`後仍再次掃描MultiIndex date level。兩者移除後只降低一次性I/O/O(N)成本，不改score values、daily information-date refresh、selector、pair identity或策略結果。

### 驗證

- `validate_breakout_quality_selection_point_in_time_score_sort_contract_case`：10 checks／0 fail。
- T294 daily Forward warm-cache效能契約維持：candidate lookup不重新掃描完整score index。
- 本輪不改MR／DL／SR ID、不改Selection PIT／Forward score-universe scientific semantics、不產生新研究結果。


## 2026-08-10 — Post-fix Forward-OOS controlled replay完成：MR-12B維持anchor、MR-13A未泛化

### 工作基準與結果來源

- 程式基準：`test-branch-1_20260810_221241_c0b9a19.zip`；SHA256 `d27645f9c3a9604d56aa32ff49734d55da2836859b243e99e1b03727ea89f4dc`。
- Selection current run：`outputs/strategy_compare/selection_pit/runs/20260810_221455_C32-C23-C25-C28_2e37f985168a/`。
- Forward current run：`outputs/strategy_compare/forward_oos/runs/20260810_221923_C1-C3-C20-C29_b0c6ee3b0b32/`。
- Forward score已先依前一輪修正重建為`feature_history_only` inference universe；本次共同策略期間完整延伸至`2026-03-02`，不再由future 40-bar Target完整性控制score presence。因此本次C20/C29結果是post-fix current evidence，舊2021～2025-12-22結果只保留PRE_FIX歷史。

### Selection PIT固定結果

- C32 Full ROOS：Return `132.25%`、MDD `15.85%`、RoMD `8.34`、Annual `12.80%`、EV `0.71R`。
- C23 Min ROOS：`107.98% / 20.57% / 5.25 / 11.03% / 0.42R`。
- C25 Min MR-12B：`101.55% / 19.66% / 5.17 / 10.53% / 0.52R`，same-param selection R `+26.19R`。
- C28 Min MR-13A：`151.88% / 18.86% / 8.05 / 14.11% / 0.71R`，same-param selection R `+109.91R`。
- C28-C25純DL source：Return `+50.33pp`、MDD `-0.80pp`、RoMD `+2.89`、Annual `+3.58pp`、EV `+0.19R`、selection R `+83.72R`。Selection仍明確支持MR-13A daily source。

### Forward-OOS post-fix結果

共同期間`2021-01-01～2026-03-02`：

- C1 Full ROOS：Return `129.08%`、MDD `17.41%`、RoMD `7.42`、Annual `17.43%`、EV `0.53R`。
- C3 Min ROOS：`101.60% / 13.12% / 7.74 / 14.56% / 1.02R`。
- C20 Min MR-12B：`190.26% / 18.75% / 10.15 / 22.95% / 1.17R`，same-param selection R `+35.93R`。
- C29 Min MR-13A：`83.81% / 15.71% / 5.34 / 12.53% / 0.52R`，same-param selection R `-183.42R`。
- C20-C3：Return `+88.66pp`、MDD `+5.62pp`、RoMD `+2.41`、Annual `+8.39pp`、EV `+0.15R`、selection R `+35.93R`。
- C29-C3：Return `-17.79pp`、MDD `+2.58pp`、RoMD `-2.41`、Annual `-2.03pp`、EV `-0.51R`、selection R `-183.42R`。
- C29-C20純DL source：Return `-106.45pp`、MDD `-3.04pp`、RoMD `-4.81`、Annual `-10.42pp`、EV `-0.65R`、selection R `-219.35R`。

### 判讀

1. MR-13A的Selection優勢在post-fix Forward策略層完全反轉；因score universe已修正並跑到2026-03-02，此反轉不得再歸因於舊future-target coverage bug。
2. C20與C29使用相同current Min ROOS、all-off rules、K/R0與frozen feasible-ascent。C29 selector診斷仍達`230`個Max-DL介入日、`217`個repair日、`20`個feasible-ascent改善日、`230`個1-swap local optimum日且resource/K violation為0；因此目前主要失敗點是**MR-13A score對實際策略邊界選擇的Forward泛化**，不是selector hard-feasibility或資源契約。
3. C29相對C3的勝率由`40.70%`升至`43.87%`，但Payoff由`3.17`降至`2.45`、EV由`1.02R`降至`0.52R`，且same-param selection R為`-183.42R`。這支持「daily score能提高部分命中率，但把basket membership推向較差的R/payoff tail」的診斷方向。
4. MR-12B/C20在相同post-fix期間仍有正selection R與更高RoMD，因此維持current Forward strategy anchor；MR-13A標記`FORWARD_OOS_STRATEGY_FAIL / NOT_PROMOTED`，但daily-universal方向本身仍可作後續研究，不回退或覆寫MR-13A identity。
5. 下一個立即可執行步驟先跑既有`Multiple-seed robustness`，判斷C20/C29差異是systematic model-semantic差異或seed variance；不得挑best seed或做ensemble。若MR-13A跨seed仍普遍負selection R／低RoMD，再做changed-order／swap-boundary attribution與Selection→Forward分布漂移診斷，才建立下一個新的`MR-*`假設。

### 本輪完整檢查追加修正：Robustness摘要reference改為config-driven

- 發現`strategy_multi_seed_robustness.py`計算RoMD「勝Min／勝Full」時以顯示名稱字串`Min ROOS`／`Full ROOS`尋找fixed baseline。這違反專案「摘要對象由config／registry／active settings驅動」契約；改名或切換profile時可能得到錯誤reference。
- `config/strategy_compare.py`的`STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS`新增`romd_reference_baselines`，以`param_source + rule_policy`描述Min／Full語意reference，不另列active C-ID，也不依賴display name。
- `StrategyMultiSeedRobustnessSettings`新增typed mapping與唯一匹配validator；robustness contract保存resolved reference arm identity，summary只依contract的arm_id取baseline。Robustness schema bump `1 → 2`，只影響尚未產生結果的robustness fingerprint，不影響Selection／Forward pair cache或策略回放語意。
- synthetic contract同步改為讀取目前config的profile、seed count／generator seed、arm names與reference semantics，不再把`forward_oos`、8 seeds或目前四個顯示名稱當成唯一合法設定。
- Scientific identity：不新增MR／DL／SR ID，不改model、Target、loss、score、selector、K/R0、交易會計或本次C20/C29結果。

## 2026-08-10 — Multiple-seed robustness isolated Forward execution-start 邊界修正

### 工作基準與實際失敗

- 程式基準：`test-branch-1_20260810_223706_50b9b04.zip`；SHA256 `3f7bc38923639c405f1a044ddb75e55c9ad3ef344dd2258e4409445e3d8e7b67`。
- Forward-OOS Multiple-seed robustness fingerprint維持`bb741ba0b0c3d680`；8 seeds × 2 stochastic arms，共16 work units。
- 第一個`Min MR-12B` seed完成canonical isolated training並queue CPU replay後，下一個`Min MR-13A` training開始；前一個replay隨即拋出：requested `2021-01-01～2026-03-02`、score available `2021-01-04～2026-03-02`。
- `2021-01-01`是Forward calendar OOS start，當天非實際score交易日；isolated score第一列合法從`2021-01-04`開始。這不是score coverage缺口，也不是前一輪future-target tail bug復發。

### Root cause

Canonical continuous-ranker OOS contract本來分開`execution_start`與score table的`available_from`：前者由model manifest `outer_oos_policy.oos_start_date`定義策略可執行期間，後者是實際score table第一個交易日。Multiple-seed的`continuous_score_path_override`支線先前沒有isolated manifest contract，只把score table第一列日期同時當execution start，因而用`2021-01-04`去拒絕合法的calendar request start `2021-01-01`。

### 修正

1. `strategy_compare_engine.run_comparison()`的isolated continuous-score override新增明確`continuous_score_execution_start_override`；score path與execution start必須成對提供。
2. override period resolver保留三個不同語意：`execution_start`、`available_from`、`available_through`。execution start可以早於第一個score交易日，但不得晚於它。
3. `strategy_multi_seed_robustness._validate_training_artifacts()`從每個isolated model manifest讀取`outer_oos_policy.oos_start_date`，並與score table attrs的`available_from/available_through`交叉驗證；replay job顯式傳遞該execution start。
4. CPU replay future若在下一個GPU trainer執行期間失敗，parent會terminate trainer；5秒內未退出則kill，再重新拋出原錯誤。run manifest同步寫`FAILED`、error type/message與`resumable=true`，不產生不完整正式summary/report。

### Resume與scientific identity

- 不改seed list、generator seed、模型profile、training defaults、Target/loss/architecture、Min/Full參數、selector、K/R0、feasible-ascent、交易會計或final-strategy統計。
- 不新增MR／DL／SR ID，不挑best seed、不做seed ensemble。
- 不升robustness schema、不改scientific fingerprint；`bb741ba0b0c3d680`保持有效。修正後由相同Research → Strategy Compare → Multiple-seed robustness入口重新執行，已完整且identity相符的per-seed isolated training artifact可直接`TRAIN REUSE`接續；未完整的training則依既有validator重訓。
- 本輪沒有新的Multiple-seed策略結果，研究狀態仍是`IMPLEMENTED／RESULT_PENDING`。

## 2026-08-10 — Multiple-seed robustness 完整檢查：正式前置、期間與resume完整性修正

### 工作基準與範圍

- 程式基準：`test-branch-1_20260810_225901_d85d909.zip`；SHA256 `b08243dff884aabbedfd887bc75dca75f2af390cc0f731343c322f5135d6cf2d`。
- 本輪是程式／契約完整檢查，沒有新的Multiple-seed策略結果；研究狀態維持`IMPLEMENTED／RESULT_PENDING`，不新增MR／DL／SR identity。
- 固定scientific variables不變：目前config的robustness profile、deterministic generated seeds、Target／architecture／loss／training defaults、Min／Full參數語意、selector、K/R0與feasible-ascent均未因結果調整。

### 發現與修正

1. Multiple-seed原本在required strategy parameter缺件時直接要求使用者先跑另一個Strategy Compare，且status頁沒有只屬於robustness的可稽核前置計畫。新增parameter-only canonical preparation service：normal Strategy Compare的DL-score BLOCKED action不會阻擋isolated-seed workflow；本次需要的param source若已有正式builder則在同一次確認後自動BUILD／REBUILD／REUSE，無builder或缺model upstream truth才事前BLOCKED。
2. Forward robustness period原本可透過normal Strategy Compare status間接依賴目前canonical score coverage。改為在未設定explicit period時直接由canonical Dataset `source_data_date_range.end`與正式walk-forward policy推導共同OOS start/end，因此舊score tail例如`2025-12-22`不得把新isolated training的策略期間截短。
3. isolated continuous override雖已分離calendar `execution_start`與score `available_from`，但輸出的`score_signal_coverage.required_start`仍誤寫成`available_from`。修正為required start=`execution_start`、first scored event=`available_from`，只修metadata，不改候選／score／交易結果。
4. `seed_results.csv` resume新增exact contract integrity：arm/seed observation不得重複、不得出現目前stochastic arms／resolved seeds之外的row，arm_order／seed_order也須一致；partial expected rows仍可合法接續。
5. fixed baseline／resume preflight與summary/report finalization原本可能在失敗後留下`RUNNING` manifest。現在與training/replay相同，均寫`FAILED`、`resumable=true`、`failed_stage`與錯誤內容；正式summary/report只在完整seed observation通過後產生。
6. 執行順序改為：顯示robustness專屬依賴計畫 → 一次確認 → parameter-only自動前置 → 重新收集identity／解析共同期間 → 建立fingerprint-scoped run目錄 → fixed baseline → isolated train/replay → aggregate report。取消操作不再先建立`PLANNED` run artifact。

### 驗證與scientific identity

- GPT獨立custom target checks驗證parameter-only preparation可忽略normal DL blocker、duplicate seed result會被拒絕、`2021-01-01` execution start與`2021-01-04` first score day仍分離。
- 全專案Python compile、AST bare-except、App menu hard-coded experiment ID、import SCC、formal output-root literal掃描均無新增問題；`apps/test_suite.py`未由GPT執行。
- `validate_strategy_compare_config_driven_app_contract_case`已補direct synthetic coverage，但formal double check仍應由使用者本地`apps/test_suite.py`執行。
- 不改既有Selection／Forward策略結果、不挑best seed、不做seed ensemble；尚未產生新的robustness數值。


## 2026-08-10 — Formal double-check閉環：Strategy Compare model-upstream synthetic caller同步

### 工作基準與formal bundle

- 程式基準：`test-branch-1_20260810_234837_b825a83.zip`；SHA256 `94adde8910a8d8eca346b2cd25c0d598b391e6a27d951851210ee28474477717`。
- Formal bundle：`to_chatgpt_bundle_20260810_234950_e975a4ff.zip`；SHA256 `b2d0be5aac68d79d75426006dfb5161c66f9460030e590dfac2736d229688b6b`。
- 使用者本機結果：quick gate／chain checks／ml smoke均PASS；consistency唯一FAIL為synthetic suite ImportError；meta quality六個coverage FAIL均由同一次synthetic early-abort連帶造成。

### Root cause與修正

1. 前一輪將Strategy Compare與Multiple-seed共用的模型上游前置檢查泛化為公開`model_upstream_prerequisite_blockers()`後，`validate_strategy_compare_config_driven_app_contract_case`仍在function-local import舊private helper `_selection_pit_checkpoint_rebuild_blockers`，formal synthetic進入該case即ImportError；production runtime已使用新helper，並非策略／訓練回歸。
2. Synthetic caller改為直接使用canonical `model_upstream_prerequisite_blockers()`，不把舊private helper加回production，避免形成第二套前置邏輯。
3. 新helper對daily-universal profile除Dataset summary外也明確要求market-set artifacts，因此fixture預期由舊的1個blocker同步為2個：Dataset summary + market-set；event-group profile仍為Dataset summary + Continuous Target。
4. 全專案靜態local-import symbol解析確認修正後沒有其他不存在的本地import；舊helper名稱已無殘留。

### Scientific identity與後續

- 不改`MR-*`／`DL-*`／`SR-C*` identity，不改Target、architecture、loss、seed policy、score、selector、K/R0、交易會計或Selection／Forward既有結果。
- Multiple-seed robustness狀態仍為`IMPLEMENTED／RESULT_PENDING`；本輪沒有產生新策略結果。
- Formal coverage的六個meta-quality FAIL應在synthetic suite可完整執行後重新計算；本輪不以GPT端執行formal suite替代使用者本機double check。

## 2026-08-11 — Multiple-seed Forward robustness首批結果：單一seed結論重開、報表matched-pair契約補齊

### 結果來源與固定條件

- 程式基準：`test-branch-1_20260811_000339_d277205.zip`；SHA256 `e49854513aa4b2607a6f8c49a112d1aa1892953ba899ca8693a98bb5347916cc`。
- 使用者提供首批Forward-OOS Multiple-seed console aggregate；目前config為8個deterministic generated seeds、profile=`forward_oos`、同一Min ROOS／all-off rules／frozen feasible-ascent、MR-12B與MR-13A各8個isolated training→strategy replay observations；Full／Min為固定baseline。
- Canonical一般模型流程仍使用`BREAKOUT_QUALITY_RANDOM_SEED=42`；本次multi-seed generator產生的8個seed不包含42，因此本結果是獨立的seed robustness evidence，不是把seed 42重複納入平均。

### 首批平均策略績效

- Full ROOS：Return `129.08%`、MDD `17.41%`、RoMD `7.42`、Annual `17.43%`、EV `0.53R`。
- Min ROOS：`101.60% / 13.12% / 7.74 / 14.56% / 1.02R`。
- Min MR-12B（N=8）：Return Mean `129.01%`、MDD `15.87%`、RoMD `8.21`、Annual `17.15%`、EV `0.77R`、Payoff `3.12`、Exposure `92.76%`、Trades `318.6`、Win `42.61%`、Monthly Win `61.31%`、Log R² `0.8772`。
- Min MR-13A（N=8）：Return Mean `148.18%`、MDD `15.47%`、RoMD `9.97`、Annual `19.02%`、EV `0.94R`、Payoff `3.19`、Exposure `92.83%`、Trades `316.0`、Win `43.66%`、Monthly Win `59.33%`、Log R² `0.8830`。
- MR-13A − MR-12B的平均差：Return `+19.17pp`、MDD `-0.40pp`、RoMD `+1.76`、Annual `+1.87pp`、EV `+0.17R`、Payoff `+0.07`、Win `+1.05pp`；Monthly Win反而`-1.98pp`。

### RoMD分布與研究判讀

- MR-12B：Mean `8.21`、Median `7.70`、Std `2.63`、CV `0.32`、Min/P25/P75/Max=`3.94/6.76/10.77/11.51`、勝Min=`4/8`、勝Full=`4/8`。
- MR-13A：Mean `9.97`、Median `10.87`、Std `3.68`、CV `0.37`、Min/P25/P75/Max=`4.38/6.84/12.92/14.49`、勝Min=`5/8`、勝Full=`5/8`。
- MR-13A的Mean與Median都高於MR-12B，因此不是單一極佳seed單獨把平均拉高；但Std/CV亦更高，顯示daily-universal ranker對training seed更敏感，lower-tail仍明顯存在。
- 這與canonical seed 42的post-fix Forward結果（MR-12B RoMD `10.15`、MR-13A `5.34`）方向相反。結論因此由「MR-13A Forward模型語意失敗」修正為：**seed 42失敗是有效單一run證據，但不足以代表跨seed模型優劣；seed variance本身已是主要研究變數之一。**
- MR-13A仍不升格。下一個判讀必須優先看相同seed下的MR-13A−MR-12B matched-pair RoMD，而不是只比較兩個獨立分布平均或任意cross-seed pairs；不得挑best seed、做seed ensemble或用Forward結果調training hyperparameter。

### 首批報表缺陷與修正

1. `DL選擇R Mean`在兩個stochastic arms顯示`-`。Root cause是robustness worker只讀`trade_attribution.json`，但score-ranking策略不產生該hard-filter attribution檔；正常Strategy Compare其實已有從`no_filter_trades.csv`與active trades重建`exclusive_selection_delta_r`的canonical SSOT。
2. robustness worker改為直接重用Strategy Compare既有`_load_direct_selection_r()`，因此score-ranking與hard-filter均使用同一直接選擇R定義，不新增第二套統計口徑。
3. 首批報表只有RoMD勝Min／Full與Markdown內任意cross-seed distribution probability，沒有回答「同一training seed下MR-13A是否優於MR-12B」；schema `2 → 3`新增`romd_same_seed_comparison`，輸出matched N、雙方勝數、tie與ΔRoMD Mean/Median/Std/Min/P25/P75/Max，console與Markdown都顯示。
4. schema bump只為完成robustness持久結果／報表契約；不改seed generator、Target、architecture、loss、training defaults、Forward period、selector、K/R0、策略accounting或任何scientific variable。因v2首批seed_results沒有可回復的直接選擇R、且暫存replay/training預設已清除，完整v3 `DL選擇R Mean`需要由同一正式入口重新跑8×2 isolated observations；不得由現有aggregate猜值。

### 狀態

- Multiple-seed robustness：`RESULT_AVAILABLE / REPORT_SCHEMA_V3_IMPLEMENTED / MATCHED_PAIR_RECHECK_REQUIRED`。
- MR-12B：維持canonical seed42 runtime anchor，但`MULTI_SEED_SUPERIORITY_NOT_ESTABLISHED`。
- MR-13A：`SINGLE_SEED_FORWARD_FAIL / MULTI_SEED_REOPENED / NOT_PROMOTED`。

## 2026-08-11 — Multiple-seed robustness前置修正：MR-13A不依賴legacy Market Set Bank

### 工作基準與實際BLOCKED

- 程式基準：`test-branch-1_20260811_002332_9edfd4b(1).zip`；SHA256 `0f73da9b9ebb89a5eee796f308bf071057c1e3eefdbabfb28b63b1b36ed6e4c8`。
- 使用者由`apps/research.py` → Strategy Compare → Multiple-seed robustness查看／執行Forward-OOS計畫時，`daily_universal_no_time_pairwise`被`Dataset market-set工件缺少`標成BLOCKED，缺少項目為`market_daily_features.npy`、`market_daily_valid_mask.npy`、`market_date_ordinals.npy`等legacy Market Set Bank sidecars。
- 這個BLOCKED不是資料真的缺少MR-13A所需上游真理，而是前一輪formal synthetic caller閉環時把daily-universal prerequisite錯誤同步成Dataset + market-set，將legacy architecture需求誤套到MR-13A sequence-only ranker。

### Root cause與修正

1. MR-13A `daily_universal_no_time_pairwise`固定使用`inception_time_v1` sequence-only architecture；`load_daily_universal_ranker_data()`明確拒絕`requires_market_set`／dataset context／derived context architecture。
2. MR-13A不讀Market Set Bank。Canonical trainer依既有Dataset summary／source inventory確認資料來源後，直接從canonical OHLCV在batch需要時lazy產生每個stock-day的300×10 window；`daily_opportunity_no_time_r_v1`亦依固定contract由同一OHLCV即時計算，沒有persistent expanded daily feature bank或market-set sidecar prerequisite。
3. `model_upstream_prerequisite_blockers()`移除daily-universal的market-set artifact blocker。Event-group ranker仍維持Dataset + persistent Continuous Target manifest兩項上游契約；daily-universal只要求canonical Dataset truth，後續來源inventory/current-source一致性仍由canonical daily trainer驗證。
4. Multiple-seed execution plan的REUSE說明改為依training sample scope顯示：event-group重用Dataset／Continuous Target truth；daily-universal重用Dataset／source OHLCV truth並由canonical trainer即時計算daily windows與固定target，明確說明不需要legacy market-set。
5. `validate_strategy_compare_config_driven_app_contract_case`同步改回正確契約：空白root下event-group應有Dataset + Continuous Target兩個blocker；daily-universal只有Dataset summary一個blocker，且不得出現market-set blocker。

### Scientific identity與狀態

- 不新增或修改`MR-*`／`DL-*`／`SR-C*` identity，不改MR-12B／MR-13A architecture、Target、loss、training sample definition、seed generator、training defaults、Forward period、selector、K/R0、策略accounting或既有Multiple-seed結果。
- 不建立legacy Market Set Bank，也不把已淘汰的`inception_time_market_set_*` architecture重新帶回正式MR-13A流程。
- Multiple-seed robustness既有結果與schema v3判讀維持有效；本輪只是修正前置依賴分類，讓同一正式入口可直接進入既定isolated train + replay流程。
- GPT未執行`apps/test_suite.py`；formal double check仍由使用者本機正式入口執行。

## 2026-08-11 — Strategy Compare robustness workflow收斂：Selection PIT multi-seed、年度aggregate、console降噪與SSOT

### 工作基準

- 程式基準：`test-branch-1_20260811_003951_11855b1(1).zip`；SHA256 `17637b30a6372992c7dd4aedc91ea8a6603520d2506e2f5c6a254aac496f4b93`。
- 本輪只修改research/robustness infrastructure與report contract；沒有產生新的模型／策略結果，不改既有MR-12B／MR-13A scientific interpretation。

### 實作內容

1. Strategy Compare正式選單由config動態新增兩個獨立robustness工作類型：`Forward-OOS Multi-seed robustness`與`Selection PIT Multi-seed robustness`；App不硬編MR/C/model名稱。
2. Selection PIT robustness改用canonical PIT builder的isolated output override。每個seed只建策略比較期間`2014-01-01～2020-12-31`所需fold；現行12 months/fold因此是7 folds/model/seed，8 seeds×2 stochastic models預計112 fold trainings。Canonical MR-12B 10-fold與MR-13A 8-fold歷史工件不被覆寫，也不是本工作量。
3. `run_comparison(..., quiet=True)`完成真正silent worker contract：market/signal cache、replay狀態與完整pair report都不再由inner worker印到console；robustness orchestrator統一管理進度。TTY使用bounded inline progress，redirected output只保留完成狀態；DONE/REUSE/FAILED與績效delta沿用共用console palette及`strategy_report_style`方向性。
4. 新增`seed_yearly_returns.csv`永久raw aggregate；無論年度表顯示開或關都保存raw yearly。robustness summary/Markdown/console可加入年度Mean/Median/Std/Min/P25/P75/Max與兩個stochastic arms的same-seed年度差值／勝數；每個observation的年度資料在刪除暫存replay前抽取，後續只改report renderer或切換年度顯示可直接重建，不需重訓。
5. `DL選擇R`仍沿用Strategy Compare既有`_load_direct_selection_r()` SSOT；RoMD same-seed matched-pair與cross-seed distribution維持分開。
6. `config/strategy_compare.py`移除重複的Dataset／param policy／max positions／rotation magic values；Strategy Compare直接讀`get_breakout_quality_workflow_settings()`。檔案頂端只集中Strategy Compare自己擁有且可調的常用knobs（seed count/generator、CPU replay workers、console mode、年度報表、retention等）；單GPU training queue屬目前orchestrator能力契約，不偽裝成可調config knob。
7. Robustness schema升至v4，contract拆成scientific identity與execution/report options。Scientific fingerprint不再包含worker數、console mode、progress interval、yearly renderer、report schema、retention、arm顯示名稱與DL description；Selection PIT scientific identity另明確保存fold months／inner validation／minimum group policy，因此只有真正training/runtime semantics改變才換fingerprint。
8. 成功完成仍預設清除isolated checkpoints/Scores/replay details；失敗或中斷保留resumable工件。同fingerprint重跑可重用完成seed；同一v4 scientific fingerprint的run若缺年度raw，該observation會視為尚未完整並補跑一次，之後report-only變更不再需要重訓。Selection PIT即使checkpoint與score共置同一isolated root，`keep_checkpoints`／`keep_scores`仍分別生效。
9. 年度aggregate明確區分結果side：Full/Min fixed baseline一律讀`no_filter_return_pct`，stochastic DL ranking一律讀`score_ranking_return_pct`；即使fixed baseline由既有controlled-pair cache重用，也不得把該pair的DL-on年度報酬誤當baseline。

### Scientific identity與狀態

- 不新增／重用任何`MR-*`、`DL-*`、`SR-C*` ID；不改Target、architecture、loss、selector、K/R0、交易會計或既有Selection/Forward period。
- Selection PIT canonical fold數差異仍是sample-universe最早合法score start造成：MR-12B canonical為2011～2020共10 folds，MR-13A canonical為2013-04～2020共8 folds；本次Selection robustness因策略期間固定2014～2020而雙方一律只需7 folds/seed。
- 狀態：`IMPLEMENTED / FORWARD_COMPATIBLE / SELECTION_ROBUSTNESS_RESULT_PENDING`。
- 依PROJECT_SETTINGS，本輪不執行`apps/test_suite.py`；formal double check由使用者本機正式入口完成。

## 2026-08-11 — Formal bundle 閉環：Strategy Compare 動態選單 synthetic fixture 修正

- 程式基準：`test-branch-1_20260811_014905_2714e23.zip`；SHA256 `001bc5fa9edaa97e1c719f94159681b1e7f32a8766d1c73a112d673375a852d6`。
- Formal bundle：`to_chatgpt_bundle_20260811_015127_83b610ce.zip`；SHA256 `bec9eee24124559c12e1597c31cbdee1938cba87aadeed8dc7a41a87f1ec9b37`。
- 本地 formal summary：quick gate PASS、chain checks PASS、ml smoke PASS；consistency 因 `validate_dataset_cli_contract_case` 在 Strategy Compare 互動選單使用舊的固定輸入`4`，新增第二個robustness profile後`4`已改為Selection PIT robustness，mock input `["4", "0"]`於返回外層時耗盡並拋`StopIteration`，因此 consistency summary 未生成。
- `meta quality`的三個失敗均為上述中止的次生結果：coverage synthetic run info記錄`synthetic_case_count=0`／returncode=1，導致`coverage_synthetic_suite_runs_successfully`與`coverage_key_targets_hit`失敗；performance則因缺少consistency step summary而使`performance_required_step_summaries_present`失敗。既有coverage line/branch門檻本身仍高於要求，quick/chain/ml smoke沒有發現新的runtime失敗。
- 修正只更新synthetic CLI fixture：狀態頁選項改由目前`get_strategy_comparison_profiles()`與`get_strategy_multi_seed_robustness_profiles()`數量動態推導，並同時驗證normal profile與robustness status renderer呼叫數；獨立檢查另發現同case後段`compare robustness` mocks仍是新增`robustness_id=`前的舊signature，已同步改為接受並驗證config解析出的default robustness ID，避免修掉`StopIteration`後下一個formal run再於TypeError中止。
- 不修改Strategy Compare runtime、Selection PIT／Forward-OOS、多seed scientific identity、seed generator、Target、architecture、loss、training defaults、策略參數、selector、交易會計、報表統計或工件格式；不新增MR／DL／SR ID。
- 本輪依`doc/PROJECT_SETTINGS.md`不執行`apps/test_suite.py`或其formal steps；交付前以獨立靜態／結構檢查閉環程式與checklist契約。

## 2026-08-11 — Formal bundle 閉環：silent worker 與 shared console synthetic 契約分離

- 程式基準：`test-branch-1_20260811_020957_219acdb.zip`；SHA256 `abe203462895bc3cfd4b66fb196d3ba845203c1ec72535e896b24b267c49277d`。
- Formal bundle：`to_chatgpt_bundle_20260811_021151_ae41e284.zip`；SHA256 `97f5bcc1a4d30e173bc05bae3a257a9cce1948ca9ba564ff35ab226031950aa1`。
- 本地 formal summary：quick gate PASS、chain checks PASS、ml smoke PASS；consistency已正常產生summary，253個synthetic cases中只剩1個FAIL：`hard_filter_strategy_compare_passes_none_capture_audit_to_shared_console`。meta quality亦只剩`coverage_synthetic_suite_runs_successfully`，是同一個synthetic FAIL的次生結果；coverage line約`71.13%`，沒有跌破既有門檻。
- Root cause不是Strategy Compare runtime regression，而是前一輪完成silent worker後，`run_comparison(..., quiet=True)`依新契約不再呼叫console renderer；舊hard-filter direct synthetic卻仍在`quiet=True`下期待`_render_strategy_console_report()`被呼叫並收到`color=None`，把兩個互斥契約混在同一fixture。
- 修正：hard-filter shared-console direct case改用`quiet=False`，但仍在fixture內`redirect_stdout()`，因此只驗證一般互動Strategy Compare的canonical shared renderer會收到`color=None`且不造成formal log洗版；Multiple-seed robustness direct check則明確驗證orchestrator source以`quiet=True`呼叫inner `run_comparison()`，保留silent worker契約。
- 不修改Strategy Compare runtime、MR-12B／MR-13A、Target、architecture、loss、training defaults、seed generator、Selection／Forward期間、selector、策略參數、交易會計、scientific fingerprint或既有robustness結果；不新增／修改MR／DL／SR identity。
- `doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md`無需變更，因本輪只修validator契約，沒有任何research identity或狀態改變。
- 本輪依`doc/PROJECT_SETTINGS.md`不執行`apps/test_suite.py`或formal consistency step；交付前只做獨立靜態／結構／依賴與validator contract檢查。

## 2026-08-11 — 正式選單 Enter 顯示格式統一與 Strategy Compare robustness 排序

- 程式基準：`test-branch-1_20260811_022602_d75dc04.zip`；SHA256 `efe4eae6e10d9f361c6bab59f4a526847b589774e659f3a1a37cfc3bf38f2cb1`。
- 使用者要求所有「按 Enter 等價於數字 1」的正式互動選單，第一項統一顯示為`[1 ] 項目  (Enter)`；其餘項目統一顯示`[2]  項目`。本輪新增`core.console_report.render_menu_item()`作project-wide顯示SSOT，Research主選單、Strategy Compare階段／robustness子選單、Audit、Binary／Continuous模型研究與optimizer Portfolio記憶庫的numeric-1 default prompt均改用同一renderer。Enter本來不是數字1的Optimizer Mode／Study Mode選項不改語意。
- Strategy Compare正式選單顯示順序改為Selection PIT strategy、Forward-OOS strategy、Selection PIT Multi-seed robustness、Forward-OOS Multi-seed robustness、全部狀態；robustness順序直接由`config/strategy_compare.py`profile insertion order驅動，CLI default robustness仍維持既有`STRATEGY_COMPARE_DEFAULT_ROBUSTNESS_PROFILE`，不因顯示排序改變。
- Validator不硬編目前robustness ID或數量；只驗證共用menu renderer輸出與robustness profile順序跟Strategy profile順序一致。`doc/CMD.md`與Checklist同步更新目前正式UI。
- 本輪只改UI／validator／文件，不改Dataset、Label、MR-12B／MR-13A、Target、architecture、loss、seed、Selection／Forward period、strategy accounting、scientific fingerprint或既有robustness結果；不新增／修改任何MR／DL／SR identity。
- 依`doc/PROJECT_SETTINGS.md`，GPT不執行`apps/test_suite.py`；formal double check由使用者本機正式入口完成。

## 2026-08-11 — Formal bundle 閉環：menu renderer 後的 Audit／Trade-path source-string fixture 修正

- 本地 formal summary：quick gate PASS、chain checks PASS、ml smoke PASS；consistency `5378` checks中只剩2個synthetic FAIL，分別為`project_audit_entry_and_breakout_quality_facade_share_one_config_driven_backend`與`trade_path_menu_completes_model_artifacts_and_strategy_comparison_stays_separate`。Meta quality只剩`coverage_synthetic_suite_runs_successfully`，屬這兩個consistency FAIL的次生結果；coverage line／branch與performance summary本身仍通過。
- Root cause均為上一輪正式選單改用project-wide `render_menu_item()`後的stale source-string fixture：Audit case仍搜尋舊`[4]       Audit／診斷`固定spacing；Trade-path case仍在`application.py` source內搜尋渲染後的`[1 ] ... (Enter)`／`[2]`／`[3]`字串。Production正式入口、工作類型、config-driven backend與模型／策略分離語意沒有回歸。
- 修正新增共用AST source helper，直接驗證`render_menu_item(index, label, default)`呼叫語意；Trade-path仍要求三個模型研究工作類型、builder command、模型流程不得執行策略績效比較且legacy strategy gate不得進正式模型選單；Audit仍要求Research主選單保留Audit入口、模型research facade不混入Audit、active module與enabled audit backend共用同一config-driven服務。Renderer實際spacing／Enter字樣只由`core/console_report.py`與其專屬CLI contract驗證，不再由不相干synthetic重複硬編。
- 本輪只修validator與必要文件，不修改Dataset、Label、MR-12B／MR-13A、Target、architecture、loss、seed、Selection／Forward期間、Strategy Compare、Audit backend、trade-path runtime、scientific fingerprint或既有robustness結果；不新增／修改任何MR／DL／SR identity。

## 2026-08-11 — Enter預設列格式最終統一為 `[1]  ...  (Enter)`

- 使用者進一步修正上一輪的預設列格式：所有「Enter 等價於數字1」的正式互動選單，最終統一顯示為`[1]  項目  (Enter)`；不再使用`[1 ] 項目  (Enter)`。
- 實作只修改project-wide `core.console_report.render_menu_item()`的default renderer，因此Research、Strategy Compare、Audit、Binary／Continuous模型研究與已接共用renderer的optimizer提示會同源同步；非default列仍維持`[2]  項目`。
- `doc/CMD.md`、trade-path前置提示與direct renderer synthetic expectation同步更新；Enter本來不等於數字1的Optimizer Mode／Study Mode仍不改語意。
- 本輪不修改Dataset、Label、MR-12B／MR-13A、Target、architecture、loss、seed、Selection／Forward期間、strategy accounting、scientific fingerprint或既有robustness結果；不新增／修改MR／DL／SR identity。
- 依`doc/PROJECT_SETTINGS.md`，GPT不執行`apps/test_suite.py`；formal double check由使用者本機正式入口完成。



## 2026-08-11 — Multiple-seed robustness performance path：2×GPU queue、daily batch向量化與prepared-data cache

### 工作基準與瓶頸

- 程式基準：`test-branch-1_20260811_025425_0f1a7c6.zip`；SHA256 `a4acff5b03779eecfe7d43af3f2372e41bdb7098507eac77003d1790eb5f7947`。
- 使用者實跑Forward multi-seed時CPU約`10～20%`、GPU約`25～60%`、RAM約`15/32GB`；MR-12B training約`30～55s/seed`，MR-13A約`5.5～6.2min/seed`，strategy replay約`55～80s/observation`。因此wall time主要由8個MR-13A GPU trainings串行主導，而不是CPU replay。
- 本輪只改execution/performance path，不新增MR/DL/SR identity，不改Target、architecture、batch identity/order、loss、optimizer update、epoch selection、final refit、seed generator、Forward/Selection period、selector或strategy accounting。

### 實作

1. `StrategyMultiSeedRobustnessSettings.gpu_train_workers`能力契約由固定1改為`1～2`，`config/strategy_compare.py`常用knobs目前預設GPU train workers=`2`、CPU replay workers=`2`。Orchestrator使用獨立training/replay executors；工作排序依training sample scope泛化，先啟動一個daily與一個non-daily工作，後續優先保持長daily trainings滿載，不硬編MR/C ID。
2. MR-13A `LazyDailyFeatureBank`把各ticker canonical OHLCV pack成單一contiguous array＋ticker offsets；每個batch一次gather全部300-bar stock windows與benchmark windows，再向量化price/volume normalization。Synthetic direct reference確認batch與scalar canonical feature逐元素`array_equal=True`；本輪額外128 samples×300 bars microbenchmark約`10.66×` CPU materialization speedup，僅屬工程microbenchmark，不是模型／策略結果。
3. Continuous ranker新增CPU `train_prefetch_workers`（目前2），prefetch futures雖可同時物化多batch，但仍依原submission順序yield，因此batch identity/order與optimizer updates不變。Pairwise loss對整batch同一日期走等價RankNet fast path；direct check確認loss與pair count和原generic branch完全相同。
4. MR-13A robustness使用run-scoped `BREAKOUT_QUALITY_DAILY_PREPARED_CACHE_PATH`。第一個daily trainer仍以canonical Dataset/source OHLCV完整建立`ContinuousRankerDataBundle`，之後同fingerprint processes只pickle/load相同canonical sanitized arrays、indexes、targets與metadata；cache不保存`N×300×10` expanded stock-day tensors，run成功後隨model scratch清除。Lockfile＋atomic replace避免兩個trainer同時建立；失敗時orchestrator先終止active trainers，再移除本run的prepared-cache lock與中斷寫入的tmp，避免resume卡在已死亡owner或殘留大型partial cache。
5. Forward robustness trainer加入hidden `--strategy-scores-only` execution mode。Inner Train/Validation選epoch、完整Selection refit與frozen checkpoint完全照canonical training執行；checkpoint後只做strategy replay真正需要的Forward score與minimal report/manifest，略過描述性full split metrics、candidate diagnostics與Markdown。MR-12B event ranker與MR-13A daily ranker都支援此模式。
6. 每個stochastic observation永久aggregate新增training phase秒數（Data／Epoch selection／Final refit／Forward score／Export）以及model/score SHA256；robustness summary/console/Markdown新增Training phase Mean。SHA用於parallel execution可稽核，不把worker數或cache/prefetch/report模式放進scientific fingerprint。
7. Selection PIT robustness同樣可用2個isolated trainer processes；daily PIT builder可共享相同run-scoped prepared bundle，但仍只建立2014～2020需要的7 folds/model/seed，PIT scientific policy不變。

### 驗證與狀態

- `LazyDailyFeatureBank` vectorized batch vs scalar reference：逐元素完全相同；另直接驗證packed storage對第二ticker的不足history source position會拒絕，不可跨ticker讀到前一檔尾端；pairwise single-date fast path vs generic formula：loss/count完全相同；2-worker prefetch：batch IDs與materialized values維持原順序。
- Prepared cache以synthetic `ContinuousRankerDataBundle`做pickle round-trip，restore後feature output完全相同。
- 此環境沒有使用者RTX 5080，因此不宣稱實際53分鐘已降至特定wall time；正式GPU utilization、VRAM與wall-time改善必須由使用者本機下一次multi-seed run的新增phase timings驗證。若2 concurrent trainers觸發VRAM/driver限制，可只把`STRATEGY_COMPARE_ROBUSTNESS_GPU_TRAIN_WORKERS`調回1，不改scientific fingerprint。
- 狀態：`IMPLEMENTED / PERFORMANCE_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。
- 依`PROJECT_SETTINGS.md`，GPT未執行`apps/test_suite.py`或formal step；正式double check由使用者本機入口執行。

## 2026-08-11 — Multiple-seed robustness 效能隔離重測：只保留 2×GPU training concurrency

- 使用者回報上一輪合併效能版實跑：`GPU train workers=2 / CPU replay workers=2`，MR-13A單一seed training仍約`5.5～6.3min`，GPU使用率約100%、CPU約20%；16 observations總wall time為`29:51`。相較先前單GPU serial約`53min`，整體wall time實際已接近減半，但無法由合併版判定是2×GPU concurrency、daily vectorization、prepared-data cache、prefetch或minimal export中的哪一項造成。
- 使用者已要求回到一次只測一個execution變因。當前隔離重測只保留`gpu_train_workers=2`；`cpu_replay_workers=1`，daily feature materialization、prefetch、prepared-data cache、trainer完整report/score export全部回到2026-08-11效能合併版之前的canonical路徑。
- 2×GPU僅改orchestrator排程：每個seed/model仍呼叫相同canonical trainer subprocess、相同Dataset/Target/architecture/batch/loss/optimizer/epoch selection/final refit與score export；per-seed model/research/replay目錄維持隔離，失敗時必須終止所有active trainer subprocess並保留resumable manifest。`gpu_train_workers`仍屬execution option，不進scientific fingerprint。
- 本輪目的只量測wall-time因果，不新增MR/DL/SR identity、不改任何模型或策略scientific condition。正式判讀只比較相同8 seeds/2 arms下的16-observation總耗時與per-seed training時間；若總時間顯著下降而單seed training不變，即證明收益來自training overlap。
- 狀態：`IMPLEMENTED / ISOLATED_PERFORMANCE_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。

## 2026-08-11 — Formal bundle 閉環：2×GPU isolated robustness core capability validator 同步

- 程式基準：`test-branch-1_20260811_171754_178cc64.zip`；SHA256 `fb811dc5a269dcb1b88c8d7918943547ebfcdae500c5126c72182388d5b22efa`。
- Formal bundle：`to_chatgpt_bundle_20260811_171856_a29bd126.zip`；SHA256 `b5874f39bf752bcd20aa7dfadb3f53d256b90d5e4f51eaf40192a8a3be4db4c8`。
- Formal consistency共987 checks，真實股票檢查皆正常，唯一FAIL為synthetic suite啟動時`validate_strategy_multi_seed_robustness_settings()`仍要求`gpu_train_workers == 1`，但目前config與orchestrator已正式支援`1～2`且隔離重測設定為2，因此直接拋`ValueError: 目前multi-seed robustness採單一GPU training queue`。
- Root cause是上一輪GPU=2-only覆蓋式patch漏帶`core/strategy_comparison.py`：`config/strategy_compare.py`與`strategy_multi_seed_robustness.py`已支援2個isolated canonical trainers，direct synthetic亦按`1～2`驗證，但core capability contract仍停留舊單GPU限制。這是infrastructure contract不一致，不是模型／策略runtime結果回歸。
- Core validator改為明確能力範圍`MULTI_SEED_GPU_TRAIN_WORKERS_MIN=1`、`MAX=2`；只驗證config值落在合法範圍，不把目前值2硬編成唯一答案。CPU replay、Dataset/Target、architecture、batch/loss/optimizer、epoch selection、final refit、score export、scientific fingerprint與策略accounting均不改。
- Bundle中的六個meta-quality coverage FAIL均為synthetic suite在coverage收集起點即被上述ValueError中止的連鎖結果：`synthetic_case_count=0`，target line/branch只剩27.82%/23.63%，不能解讀成實際coverage退化；修正後需由本機正式入口重新生成coverage。
- 不新增／修改任何`MR-*`、`DL-*`、`SR-C*` identity；2×GPU仍只屬execution option。狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。

## 2026-08-11 — Infrastructure SSOT Batch 1：Strategy Compare前置dependency runner收斂

- 程式基準：`test-branch-1_20260811_221534_5ab88e2.zip`；SHA256 `7cd88f5d7f7ffb35c3407803bdb17e5e02494f22a66120a9df74e338007fc4aa`。
- 本輪屬infrastructure refactor，不新增／修改任何`MR-*`、`DL-*`、`SR-C*`、Target、architecture、loss、seed、Selection／Forward期間、策略參數搜尋語意或交易會計。
- `StrategyPreparationAction`新增`dependencies`、`producer_work_type`與`execution_priority`；`StrategyPreparationPlan.from_actions()`統一計算READY／PREPARABLE／BLOCKED，並拒絕重複artifact key、未知dependency與dependency cycle。
- Strategy Compare全量前置與Multiple-seed robustness的parameter-only前置改共用單一dependency-aware wave runner；parameter-only只選取requested parameter artifacts與其dependency closure，可繼續忽略不相關canonical DL blocker。
- 保留既有重要執行語意：可建立的strategy parameter action以較高priority先執行並立即re-plan，避免先重建可能因新param SHA而可由completed pair免除的歷史DL工件；deterministic score／PIT checkpoint rebuild排在其後。
- archived completed-pair dependency waiver會保留原action dependencies／priority並把producer標成existing artifact，避免cache reuse路徑丟失plan metadata。
- Direct regression以隔離synthetic plan確認：(1) parameter-first + immediate re-plan只執行parameter、後續score轉REUSE；(2) robustness parameter-only可在全plan含無關BLOCKED DL action時仍只建立requested parameter；另新增正式synthetic contract覆蓋dependency metadata與cycle rejection。
- 本輪依`doc/PROJECT_SETTINGS.md`不執行`apps/test_suite.py`或formal suite step；交付前僅做獨立compile、AST/import/static contract與修改同鏈檢查。
- 狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。
## 2026-08-12 — Infrastructure SSOT Batch 1-B：Artifact Registry／Dataset-Target readiness收斂

- 程式基準：`test-branch-1_20260811_235533_5cb18fa.zip`；SHA256 `330d430d38aea9b61a48eb955d38711b9ca253403f205c66c60d4d10089ced5d`。
- 本輪屬infrastructure refactor，不新增／修改任何`MR-*`、`DL-*`、`SR-C*`、Target identity、architecture、loss、seed、Selection／Forward期間、策略參數搜尋語意或交易會計。
- 新增`filters/breakout_quality/artifact_dependency_registry.py`，正式定義Dataset Core、Continuous Target、Model Checkpoint、Forward Score、Selection PIT Score與PIT Audit的semantic dependency與producer work type；Strategy Compare的preparation plan把canonical Dataset／Target顯式列成upstream nodes，後續score／PIT actions直接依賴這些nodes。
- 新增`filters/breakout_quality/dataset_readiness.py`作metadata-only Dataset readiness SSOT；由Model Research既有refresh判斷抽出storage schema／format、artifact metadata、dataset profile、ticker coverage、feature/context contract、label policy、source CSV inventory與architecture-specific Market Set sidecar檢查。Model Research facade與Strategy Compare／robustness現在使用同一結果，避免只看檔案存在就把stale Dataset判READY。
- `filters/breakout_quality/continuous_target.py`抽出`load_validated_continuous_target_manifest()`，與原array loader共用相同schema／identity／dataset artifact binding／target artifact size+SHA驗證；preflight不需載入完整target arrays即可嚴格確認Target是否可REUSE。
- Selection PIT checkpoint-only deterministic rebuild與PIT Audit不再由Strategy Compare組argv呼叫兩個CLI `main()`；`build_selection_point_in_time_scores()`與`audit_selection_point_in_time_scores()`提供公開programmatic service facade，CLI入口只負責parse後委派。Model Research準備Strategy Compare工件同樣改用公開service facade並維持原simple-report輸出契約。
- Direct targeted regression：B189 config-driven／preparation contract `72 checks / 0 fail`、Dataset CLI contract `170 / 0`、Breakout Quality app simple-report `7 / 0`、Continuous Target contract `16 / 0`；全專案compile／AST／import-cycle／bare-except另於交付前獨立檢查。
- `tools/` formal reverse dependencies尚未在本批完全消除；本批只先移除CLI `main(argv)`耦合並建立可搬遷的public service boundary，後續Service Boundary batch再把producer實作移出`tools/`。
- 狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。依`doc/PROJECT_SETTINGS.md`，GPT不執行`apps/test_suite.py`。


## 2026-08-12 — Config SSOT Batch 2-A：execution defaults／optimizer seed／Strategy Compare activation收斂

- 程式基準：`test-branch-1_20260812_004224_2b988ae.zip`；SHA256 `2904b2409463b81a9bdf4f52e8298ba521147835bb7215b9956d52b06c9c6986`。
- 本輪屬config/infrastructure refactor，不新增／修改任何`MR-*`、`DL-*`、`SR-C*`、Target、architecture、loss、Selection／Forward期間、策略參數數值或交易會計；目前數值保持完全相同，只收斂owner。
- `config/execution_policy.py`新增canonical `DEFAULT_PORTFOLIO_MAX_POSITIONS`、`DEFAULT_PORTFOLIO_ROTATION`、`DEFAULT_FIXED_RISK`、`DEFAULT_MAX_POSITION_CAP_PCT`；Breakout Quality workflow、optimizer session/main/rolling fallback、Strategy Compare builder fallback、Audit與Workbench/local-regression預設均改引用同一owner，不再各自硬編`10 / off / 0.01 / 0.30`。
- `config/training_policy.py`新增`OPTIMIZER_RANDOM_SEED_DEFAULT`作optimizer stochastic seed唯一owner；Selection historical parameter builders與optimizer timing fallback改引用該設定。模型training seed仍由`config/breakout_quality.py`獨立擁有，robustness generator seed仍由`config/strategy_compare.py`獨立擁有，避免不同seed語意被錯誤合併。
- `STRATEGY_COMPARE_ARMS`與`STRATEGY_COMPARE_CONTRASTS`移除無效的個別`enabled`欄位；active狀態只由`STRATEGY_COMPARE_PROFILES[*].arm_ids / contrast_ids`決定。Runtime dataclass仍保留衍生`enabled`布林值供既有consumer使用，因此不改正式介面與結果。
- `validate_strategy_compare_config_driven_app_contract_case`新增profile-membership單一啟用來源與execution/optimizer defaults SSOT guard；targeted contract由72增至74項且0 fail。歷史arm／contrast定義本輪仍留在同一config，後續Batch 2-B再做active/historical物理隔離，避免同批同時改identity與設定owner。
- 狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。依`doc/PROJECT_SETTINGS.md`，GPT不執行`apps/test_suite.py`。

## 2026-08-12 — Config SSOT Batch 2-B：Active／Historical Strategy Compare catalog物理隔離

- 程式基準：`test-branch-1_20260812_005720_6f848eb.zip`；SHA256 `294fa86afaa87b8ca44a96b89d0eba89fece87034fa0b28d301a31560ca69317`。
- 本輪屬config/infrastructure refactor，不新增／修改任何`MR-*`、`DL-*`、`SR-C*`、Target、architecture、loss、Selection／Forward期間、策略參數數值、robustness seed或交易會計。
- `config/strategy_compare.py`的user-maintained catalog只保留目前`selection_pit`／`forward_oos` profiles dependency closure：4個parameter sources、4個DL sources、8個arms、8個contrasts；退役的2個parameter sources、5個DL sources、25個arms、51個contrasts移至`config/compatibility/strategy_compare_history.py`，標記為歷史唯讀相容定義。
- `get_strategy_comparison_settings()`在runtime仍把active與historical compatibility catalog無重疊合併，因此既有C17 Dynamic-K reference、歷史arm／contrast synthetic、archived Strategy Compare result decoding與pair replay compatibility均維持；current profile的啟用集合仍只由`STRATEGY_COMPARE_PROFILES[*].arm_ids/contrast_ids`決定。
- 新增active/historical physical-separation direct contract：active arms／contrasts必須精確等於current profiles聯集，active param／DL sources必須精確等於current arms dependency closure，四類catalog皆禁止active/history ID重疊；歷史catalog同樣不得重新出現第二份`enabled`。
- 修改前後`StrategyComparisonSettings.as_dict()`逐profile完全相同；無artifact identity的Selection／Forward config fingerprint維持`49f4b71d293b`／`ce8813716cf0`，因此本輪沒有造成scientific/cache identity變更。targeted Strategy Compare contract由74項增加為75項且0 fail。
- 狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。依`doc/PROJECT_SETTINGS.md`，GPT不執行`apps/test_suite.py`。

## 2026-08-12 — Service Boundary Batch 3-A：Portfolio Replay與共用optimizer primitives移出tools

- 程式基準：`test-branch-1_20260812_012242_fd6022b.zip`；SHA256 `b353fa0f58b90d1f0123e2fd8ea12ba3b32cdbe534abdf422f0ec65201fc1456`。
- 本輪屬architecture/infrastructure refactor，不新增／修改任何`MR-*`、`DL-*`、`SR-C*`、Target、architecture、loss、seed、Selection／Forward期間、strategy parameter、portfolio accounting或replay execution semantics。
- canonical portfolio replay實作由`tools/portfolio_sim/simulation_runner.py`移至`services/portfolio_replay.py`；`filters/breakout_quality/strategy_compare_engine.py`、portfolio CLI與Workbench直接依賴正式service，不再以tools module作runtime implementation owner。
- Portfolio Replay依賴的optimizer raw-data cache、trial-input preparation與walk-forward primitives同步移至`services/optimizer/raw_cache.py`、`trial_inputs.py`、`walk_forward.py`；service modules不得import `tools.*`。其他optimizer orchestration仍暫留`tools/optimizer/`，由Batch 3-B續做正式service搬遷。
- 舊`tools/portfolio_sim/simulation_runner.py`、`runtime_common.py`與`tools/optimizer/{raw_cache,trial_inputs,walk_forward}.py`保留為`sys.modules` module alias compatibility façade；legacy import取得與canonical service完全相同的module object，既有private helper import、monkeypatch target與ProcessPool pickle module identity可延續，且沒有第二套實作。
- Direct regression：Strategy Compare config/service contract`76/76`、strategy comparison`48/48`、Qualified Candidate Audit`9/9`、Candidate Counterfactual Audit`15/15`、optimizer raw-cache`6/6`、raw-universe replay`7/7`、prepared portfolio tool`11/11`，合計172 checks / 0 fail。另以baseline ZIP與新service AST比較canonical replay及三個optimizer primitive的函式／class body，除import owner遷移外行為定義保持一致。
- 本輪同步更新`doc/ARCHITECTURE.md`的`apps -> services -> filters/core`依賴方向與tools compatibility責任；不修改Experiment Registry identity。
- 狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。依`doc/PROJECT_SETTINGS.md`，GPT不執行`apps/test_suite.py`。


## 2026-08-12 — Service Boundary Batch 3-B：PIT／Optimizer canonical library搬出tools

- 使用者本輪最新提供基準：`test-branch-1_20260812_012242_fd6022b(1).zip`；SHA256 `b353fa0f58b90d1f0123e2fd8ea12ba3b32cdbe534abdf422f0ec65201fc1456`。該ZIP與Batch 3-A修改前基準相同，因此本輪先重疊已驗證的3-A patch，再進行3-B；交付patch為相對此最新ZIP的3-A＋3-B累積修改，避免遺漏前批service migration。
- 本輪屬architecture/infrastructure refactor，不新增／修改任何`MR-*`、`DL-*`、`SR-C*`、Target identity、architecture、loss、seed、Selection／Forward期間、策略參數搜尋語意或交易會計。
- canonical Breakout Quality application service移至`services/breakout_quality/`：binary train、continuous ranker、daily ranker、continuous ranker pipeline、Selection PIT build、Binary PIT build與PIT audit均由正式service承接；舊`tools/filters/breakout_quality/*`及`tools/audit/breakout_quality/point_in_time_scores.py`改為direct-script delegate＋import-time `sys.modules`同module compatibility alias，既有CLI與private patch/import contract仍可使用且不複製第二套implementation。
- Strategy Parameter Training所需optimizer dependency closure移至`services/optimizer/`：callbacks、objective、objective runner／filters／profiles、outer rolling OOS、param cache、prep、profile、robustness、runtime、score display、session／factory、study utils；3-A既有raw cache／trial inputs／walk-forward繼續作同一canonical package。搬移service內部只依賴`services/`、`core/`、`filters/`與`config/`，不再反向import`tools/`；舊`tools/optimizer/*` library modules改同module compatibility alias。
- `filters/breakout_quality/strategy_compare_preparation.py`改直接呼叫`services.breakout_quality` PIT build/audit；`strategy_param_training.py`改直接呼叫`services.breakout_quality.binary_point_in_time_scores`與`services.optimizer`，因此正式`services/`、`core/`、`filters/`對`tools/`的AST import邊歸零。
- 為避免搬移時改變實作，21個搬移module以3-A baseline作AST等價比對完全PASS；`services/breakout_quality/train.py`唯一必要差異是搬移後目錄少一層，`PROJECT_ROOT`由`parents[3]`調整為`parents[2]`並新增實際root identity guard；targeted direct regression：Strategy Compare 78項、Binary Param Adapt 10項、PIT builder 29項、Optimizer objective 44項、interrupt 12項、session milestone 3項、walk-forward 48項、raw cache 6項、raw-universe replay 7項、strategy adaptation 19項、Selection strategy realization 12項、strategy comparison 48項均0 fail。
- 狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。依`doc/PROJECT_SETTINGS.md`，GPT不執行`apps/test_suite.py`；本批另交付PowerShell整合測試腳本供使用者本機formal double check。

## 2026-08-12 — Service Boundary Batch 3-B formal recheck：synthetic source-contract改讀canonical service

- 使用者本機formal bundle：`to_chatgpt_bundle_20260812_021528_736f3771.zip`。`quick gate`、`chain checks`、`ml smoke`均PASS；consistency唯一FAIL為synthetic suite runtime `ValueError: substring not found`，meta quality的6個coverage failures均由同一次synthetic coverage提前中止連鎖造成，真實股票一致性沒有新增failure。
- Root cause：Batch 3-B已把continuous ranker實作搬到`services/breakout_quality/train_continuous_ranker.py`，但`validate_breakout_quality_continuous_ranker_contract_case`與`validate_breakout_quality_pass_conditional_ranker_contract_case`仍讀legacy `tools/filters/breakout_quality/train_continuous_ranker.py` wrapper原始碼，再用`ranker_source.index("torch.save(")`驗證checkpoint-before-OOS ordering。Compatibility wrapper刻意不複製implementation，因此該source-level assumption已失效。
- 修正只調整validator的implementation source owner：continuous ranker與binary train source-contract改讀`services/breakout_quality/` canonical service；`COMMAND_MODULES`與CLI相容性仍繼續驗證`tools.filters.breakout_quality.*` wrapper，不放寬CLI compatibility。模型、Target、checkpoint、score、strategy replay與artifact identity均未修改。
- Direct regression：`validate_breakout_quality_continuous_ranker_contract_case` 15/15 PASS；`validate_breakout_quality_pass_conditional_ranker_contract_case` 8/8 PASS。另同步修正`tools/validate/synthetic_cases.py`的service-migration impacted-module mapping，使未來canonical service變更能觸發對應validator；ranker／registry／Checklist相關direct checks合計33項0 fail，且全專案靜態檢查確認不存在其他synthetic direct source path仍指向legacy continuous-ranker wrapper。
- 狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。依`doc/PROJECT_SETTINGS.md`，GPT不執行`apps/test_suite.py`；請使用者本機重跑正式入口完成double check。

## 2026-08-12 — Service Boundary Batch 3-B formal recheck 2：Daily PIT shared sample-provider source-contract同步

- 使用者最新程式基準：`test-branch-1_20260812_022640_253c7ae(1).zip`；SHA256 `78bd0f294ab020207f6c3b2da4e59ff68787c35d986bcc2bc21f0983f13068e0`。Formal bundle：`to_chatgpt_bundle_20260812_022601_5cca91cc(1).zip`；SHA256 `a57816f28b7eedce1e51ea86ae8ed6815ea80b347dacccc80221998ffa185fbc`。
- Formal結果：`quick gate`、`chain checks`、`ml smoke`正常；coverage已恢復至target line約80.27%、branch約62.18%。consistency唯一FAIL為synthetic case `BREAKOUT_QUALITY_DAILY_PIT_STRATEGY_RUNTIME` 的 `training_and_strategy_diagnostics_share_domain_layer_profile_sample_provider`，真實股票failure為0。meta quality唯一FAIL亦只是synthetic suite return code連鎖。
- Root cause：Batch 3-B已將canonical continuous-ranker pipeline搬至`services/breakout_quality/continuous_ranker_pipeline.py`，該service與`filters/breakout_quality/strategy_compare_engine.py`都實際引用`filters/breakout_quality/profile_ranker_data.py::load_profile_continuous_ranker_data`；但validator仍讀legacy `tools/filters/breakout_quality/continuous_ranker_pipeline.py` compatibility wrapper原始碼。Wrapper只做module alias、不複製implementation，因此source-string檢查得到False。
- 修正只把`validate_breakout_quality_daily_pit_strategy_runtime_contract_case`的pipeline source owner改成canonical `services/breakout_quality/continuous_ranker_pipeline.py`。不修改Dataset、Target、sample provider、training、PIT score、checkpoint、Strategy Compare、selector、portfolio replay、artifact schema或任何scientific identity。
- 狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。依`doc/PROJECT_SETTINGS.md`，GPT不執行`apps/test_suite.py`；由使用者本機正式入口完成double check。

## 2026-08-12 — Ranker Training API Batch 4：MR-12B／MR-13A／PIT／robustness共用public service boundary

- 使用者最新程式基準：`test-branch-1_20260812_023758_640750c.zip`；SHA256 `78d7e5f3b11fa61b4dd53f1115868aa8f6d69f1bcf4e55f8eb2910322a52d0c6`。本輪屬architecture/infrastructure refactor，不新增／修改任何`MR-*`、`DL-*`、`SR-C*`、Target、architecture、loss、seed、epoch-selection metric、Selection／Forward期間、checkpoint schema、score schema或策略語意。
- 盤點確認`services/breakout_quality/continuous_ranker_pipeline.py`仍呼叫canonical trainer的`_daily_rank_metrics/_daily_top_k_metrics/_spearman/_select_epoch/_fit_final/_predict_scores`，`train_daily_ranker.py`亦直接呼叫多個private helper且以`ranker_impl=`注入整個trainer module；新增`services/breakout_quality/ranker_training.py`作穩定public API，統一輸出percentile target、rank metrics、epoch selection、final refit、prediction、training semantics與output path resolution。
- `train_continuous_ranker.py`將上述canonical implementation提升為public names，內部caller同步使用public names；舊`_select_epoch/_fit_final/_predict_scores/_split_metrics/_training_semantics/_training_output_paths/_daily_rank_metrics/_daily_top_k_metrics/_spearman`只保留同function-object compatibility alias，沒有第二套實作。Daily Universal trainer移除`ranker_impl=`module injection，continuous PIT pipeline與daily trainer均只依賴`ranker_training` public API。
- 新增direct architecture contract：public API必須暴露固定能力集合、pipeline/daily consumer不得出現`ranker_impl._`或module injection、legacy private names必須與public API為同一function object；synthetic impact registry同步把`ranker_training.py`列為continuous／pairwise／listwise／PIT／daily runtime相關測項的canonical impacted module。
- 狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。依`doc/PROJECT_SETTINGS.md`，GPT不執行`apps/test_suite.py`；由使用者本機正式入口完成double check。

## 2026-08-12 — Strategy Compare Batch 5：God-module依source／replay／diagnostics／reporting責任拆分

- 使用者最新程式基準：`test-branch-1_20260812_075907_fa73493.zip`；SHA256 `080efb7a3056f400ade238c72c1c41de707555d447326f6ae76a5dfd4aa19f8c`。本輪屬architecture/infrastructure refactor，不新增／修改任何`MR-*`、`DL-*`、`SR-C*`、Target、architecture、loss、seed、Selection／Forward期間、strategy parameter、ranking policy、portfolio accounting或replay scientific semantics。
- 盤點時`filters/breakout_quality/strategy_compare_engine.py`為3,748行，param/source resolution、controlled-pair construction、portfolio scenario replay與standalone baseline cache、Selection PIT post-replay diagnostics、summary/yearly normalization、console/Markdown renderer與約900行`run_comparison()` orchestration混在同一module。
- 新增`strategy_compare_contracts.py`作comparison mode與schema contract SSOT、`strategy_compare_sources.py`承接parameter/source resolution與controlled-pair construction、`strategy_compare_replay.py`承接canonical scenario replay與reusable standalone baseline、`strategy_compare_diagnostics.py`承接replay後candidate/selection diagnostics、`strategy_compare_reporting.py`承接result normalization/yearly metrics與human-readable renderer。`strategy_compare_engine.py`降至1,367行，只保留CLI、pair orchestration、attribution compatibility與legacy import façade；formal`strategy_comparison.py`、`strategy_compare_preparation.py`、`strategy_param_training.py`與multi-seed baseline改直接引用對應專責owner。
- 行為保護：53個搬移函式逐一與本輪baseline作AST body等價比對全部PASS；`run_comparison()`、`run_existing_attribution()`、`_resolve_continuous_score_override_period()`、`main()`與`_parse_args()`body亦完全等價。既有engine private/public imports仍以alias/façade保留供歷史tools與validator相容，但不保存第二套implementation。
- Direct regression：Strategy Compare config-driven 79項、strategy comparison 48項、readable report 3項、Daily PIT runtime 9項、Qualified Candidate Audit 9項、Candidate Counterfactual 15項、Selection Strategy Realization 12項、stale-score membership guard 4項、strategy adaptation 19項、strategy reporting schema 5項，合計203項0 fail。
- Experiment Registry未修改；本輪沒有scientific identity變更。狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。依`doc/PROJECT_SETTINGS.md`，GPT不執行formal double check；使用者本機整合入口改由`apps/run_bundle.py --no-commit`執行。


## 2026-08-12 — Strategy Compare Batch 5 formal recheck：replay monkeypatch owner同步

- 使用者本機`apps/run_bundle.py --no-commit`結果：Batch 5 targeted contracts 203/203 PASS、Strategy Compare architecture PASS、Selection PIT與Forward-OOS均READY；formal suite只有consistency 1個synthetic runtime failure，meta quality 6個coverage failures均由synthetic suite提前中止連鎖造成，真實股票檢查沒有新增failure。
- Root cause：`validate_breakout_quality_binary_dl_param_adaptation_contract_case`直接取得已搬到`filters/breakout_quality/strategy_compare_replay.py`的`_run_scenario` function object，但兩個fixture仍patch legacy façade `strategy_compare_engine._run_scenario_inside_source_context`。Python function globals指向canonical replay module，因此patch沒有命中，fixture的專案外`TemporaryDirectory()`真的進入portfolio replay；issue-log安全契約正確拒絕把`log_dir`寫到project root之外。
- 修正只把兩個scenario-context monkeypatch owner改為`filters.breakout_quality.strategy_compare_replay._run_scenario_inside_source_context`。不修改Strategy Compare production code、portfolio replay、log path guard、Dataset、Target、score、strategy parameter、fingerprint或任何scientific identity。
- Direct regression：binary-DL param-adaptation contract 10/10 PASS；另逐一執行synthetic registry全部253個validator，均0 fail／0 exception，確認沒有第二個Batch 5 stale monkeypatch owner。
- 狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。GPT不執行formal `apps/test_suite.py`；使用者本機以`apps/run_bundle.py --no-commit`完成final double check。

## 2026-08-12 — Audit Batch 6：common primitives同源化與by-case interpretation邊界

- 使用者最新程式基準：`test-branch-1_20260812_084620_fa73493.zip`；SHA256 `427a98ba245bc8c57f19ca4ad3f5cbb0315bbf107909db320a12110d25cf2783`。本輪屬architecture/infrastructure refactor，不新增／修改任何`AUD-*`、`MR-*`、`DL-*`、`SR-C*`、Dataset、Target、score、策略參數、replay或scientific identity。
- 盤點確認`tools/audit/`約13.5k行，存在多組跨Audit private helper引用：No-time Target直接取Continuous Target的分布／rankability／Spearman；Candidate Counterfactual直接取Selection Strategy Realization的target attach／param coverage／output-dir；PIT target realization直接取PIT fold runtime的identity helper；11E／11F直接取11D的ranker path／JSON／SHA；另多個Audit仍引用legacy continuous-ranker與Strategy Compare engine private façade。
- 新增`tools/audit/primitives.py`作通用artifact SHA owner；新增`tools/audit/breakout_quality/artifact_primitives.py`、`target_statistics.py`、`selection_replay_primitives.py`、`pit_primitives.py`，分別承接ranker artifact path／strict JSON、Continuous Target描述統計、Selection target join＋lookahead-safe param coverage、Selection PIT DL identity。22個搬移helper與本輪baseline做AST正規化等價比對22/22 PASS；原Audit保留private compatibility alias供既有validator/歷史caller，不保存第二套implementation。
- Audit對continuous ranker改直接使用`services/breakout_quality/ranker_training.py` public API；Selection／Candidate／Qualified Candidate改直接使用Batch 5的`strategy_compare_sources/replay/diagnostics/reporting` public aliases，不再依賴`strategy_compare_engine` private façade。全`tools/audit/**/*.py`跨Audit `_private` import=0、legacy trainer import=0、Strategy Compare engine private import=0。
- 明確保留by-case責任：各Audit的hypothesis、cohort、contrast、Future Label／R使用邊界、attribution interpretation與Markdown／console renderer不抽成通用God-audit；目前仍重名的`_fmt`／`_render_report`／CLI `parse_args/main`因語意與缺值／單位／section policy不同，刻意留在各Audit。
- Direct regression：Qualified Candidate、Target Component、Time-penalty Ablation、No-time Target、PASS Realization Gap、Selection Strategy Realization、Candidate Counterfactual、Portfolio Selection Pressure、Score Ranking Capture與Audit Framework合計99項0 fail；Audit Framework新增private-boundary與shared-identity guards。狀態：`IMPLEMENTED / FORMAL_RECHECK_REQUIRED / SCIENTIFIC_CONDITION_UNCHANGED`。GPT不執行formal suite；使用者本機以`apps/run_bundle.py --no-commit`完成final double check。


## 2026-08-12 — Legacy Cleanup Batch 7：退役Strategy Gate／Adapt研究工具與temp輸出

### 基準

- Baseline：`test-branch-1_20260812_093336_f0627a6(1).zip`
- Baseline SHA256：`02836c0e775a155445b0dde12d26772a2a007d1410c02cfbdc58da067be0a711`
- 本批只做current code/test/document maintenance surface縮減；不修改Dataset、Target、model、score、active params、Strategy Compare fingerprint或portfolio semantics。

### 安全刪除證明

刪除前逐項以三層依賴檢查：

1. AST/import與全專案source reference：四個research CLI只有synthetic validator inbound import，沒有production/app/service/domain import。
2. Formal menu/config/registry：`tools/filters/breakout_quality/application.py`與`apps/research.py`沒有四個CLI的current command routing；current Strategy Compare、parameter training與trade-path model workflow皆已有正式owner。
3. Historical reconstruction：Legacy architecture/pretraining/checkpoint reconstruction仍可能需要TS2Vec／Mantis／MOMENT等相容code，本批明確保留，不因檔名舊而刪除。

### 退役並移除

- `tools/filters/breakout_quality/strategy_adapt.py` — 已完成Selection ranking×parameter adaptation研究；current參數準備由`filters/breakout_quality/strategy_param_training.py`與Strategy Compare preparation承接。
- `tools/filters/breakout_quality/strategy_filter_gate.py` — 歷史Optional-entry-filter A～E research Gate；current策略經濟比較統一由config-driven Strategy Compare承接。
- `tools/filters/breakout_quality/strategy_dl_filter_gate.py` — 歷史Binary DL rule-ablation Gate；其科學結果已記錄，current比較不再維護獨立orchestration。
- `tools/filters/breakout_quality/strategy_trade_path_label_gate.py` — 歷史Old/New trade-path Label策略Gate；Label/model/forward-score仍由formal model workflow產生，策略比較由Strategy Compare承接。
- `doc/result_tmp.md` — 無任何reference的臨時console/result dump。

四個Python CLI原始碼合計5,122行；另同步移除只為上述retired implementation服務的synthetic test surface與current docs CLI說明。

設定命名同步清理：`BreakoutQualityWorkflowSettings`中的`strategy_adapt_trials_per_fold / strategy_adapt_fixed_risk / strategy_adapt_max_position_cap_pct`改為中性`strategy_trials_per_fold / strategy_fixed_risk / strategy_max_position_cap_pct`；移除只等於execution-policy default的`BREAKOUT_QUALITY_STRATEGY_ADAPT_FIXED_RISK / MAX_POSITION_CAP_PCT` alias。既有`as_manifest_payload()`的`strategy.adaptation` payload前後逐值完全相同，不改fingerprint／artifact identity。

### 保留的current與historical compatibility

- current：`filters/breakout_quality/strategy_comparison.py`
- current：`filters/breakout_quality/strategy_param_training.py`
- current：`tools/filters/breakout_quality/build_trade_path_labels.py`
- historical reconstruction：`tools/filters/breakout_quality/build_pretraining_dataset.py`、`pretrain.py`與TS2Vec／Mantis／MOMENT model compatibility。

### Test contract同步

- 移除`validate_breakout_quality_strategy_adaptation_contract_case`；B186轉`N/A`，歷史證據由本Log保留。
- `validate_breakout_quality_strategy_comparison_contract_case`移除已退役A～E／Binary-DL Gate專屬測試，只保留canonical all-off／ranking與正式Strategy Compare contract。
- `validate_breakout_quality_trade_path_label_contract_case`保留Label/model/PIT/formal-menu契約，移除已退役Old/New strategy Gate renderer／artifact identity測試。
- `validate_breakout_quality_strategy_readable_report_contract_case`只盤點三個current persistent result owners。
- 新增`validate_breakout_quality_legacy_research_cleanup_contract_case`，釘死retired paths不存在、current replacements存在、historical checkpoint reconstruction compatibility保留、正式menu/docs不重新宣告舊CLI。

### 狀態

- Infrastructure status：`IMPLEMENTED`
- Formal recheck：`REQUIRED_ON_USER_MACHINE`
- Scientific condition：`UNCHANGED`
- Experiment Registry identity：`UNCHANGED`

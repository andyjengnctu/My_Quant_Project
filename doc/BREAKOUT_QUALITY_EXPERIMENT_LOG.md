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

本文件只記錄已知事實。歷史結果若缺少完整報表，會標記「精確值未保留」，不得自行補值。歷史資料整理截止日為 **2026-07-24**。

---

## 2. 目前基準

### 2.1 程式基準

| 項目 | 目前狀態 |
|---|---|
| 基準 ZIP | `test-branch-1_20260724_031323_dce947f.zip`，SHA256 `9e6d608964ded310cd504ce01fa3bb2ff676e6b40195389c3fb356a4b74c6efd`，加 `breakout_quality_unique_group_date_balanced_8k_patch_20260724.zip` |
| SHA256 | 來源 ZIP：`9e6d608964ded310cd504ce01fa3bb2ff676e6b40195389c3fb356a4b74c6efd` |
| 程式版本範圍 | 實證最佳仍為 8F unique ticker/date group training；active policy 已切到 8K date-density balancing。8G、8H、8I、8J 均已淘汰；8K 尚未取得完整 OOS，只能標記 `IMPLEMENTED` |
| Policy 預設 | architecture=`multiscale_cnn_sequence_only_v1`；experiment profile=`unique_group_date_balanced`；training sampling=`unique_ticker_date`；batch size=`128 groups`；early-stopping patience=`1`；final model mode=`selected_epochs`；time weight=`date_balanced`；training weight reduction=`fixed_batch_size` |
| 當前最佳研究模型 | 仍為 8F `multiscale_cnn_sequence_only_v1 / unique_group_sampling / patience 1`。8K active policy 尚未產生結果，不得預先取代 8F |
| Dataset | Full；維持固定百分比 Label；沿用既有 feature bank、4 維 context arrays、`event_group_index` 與 labels，不需重建或 relabel；Training 只使用 deterministic unique-group representatives，Validation／Selection／OOS 仍使用完整 rows |

使用者所稱「退回 v8 版本」是退回**尚未加入 v9 auxiliary head 的程式版本**，不是把 policy 預設改成 `multiscale_cnn_v8`。目前 architecture 仍為 sequence-only；正式研究基準已由 8A 更新為 8F unique-group sampling。

### 2.2 固定 Label 與訓練條件

| 項目 | 固定值 |
|---|---:|
| Feature Window | 300 bars |
| Label Horizon | 40 bars |
| PASS 最低 MFE | 嚴格大於 5% |
| PASS 最低 MFE／MAE | 嚴格大於 2.0 |
| 最大不利跌幅 | 觸及 −10% 即 REJECT |
| Epoch 上限 | 100 |
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
| Time Weight | 8F 基準=`none`；8K active 實驗=`date_balanced`，只作用於 training loss |
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
| 狀態 | `IMPLEMENTED`；完整 Selection／OOS 結果尚未取得 |
| 程式基準 | 使用者 ZIP `test-branch-1_20260724_031323_dce947f.zip`，SHA256 `9e6d608964ded310cd504ce01fa3bb2ff676e6b40195389c3fb356a4b74c6efd`，疊加 `breakout_quality_unique_group_date_balanced_8k_patch_20260724.zip` |
| 唯一訓練變更 | 8F unique-group sampling 保留全部 eligible groups；同一 training phase 內，同日每個 group raw weight=`1 / 當日 eligible unique group 數`，再正規化為平均 group weight 1 |
| Loss reduction | Training 使用 `fixed_batch_size` denominator；不再以每個隨機 batch 的 weight sum 重新正規化，避免 batch composition 抵銷日期等權語意 |
| 固定條件 | architecture=`multiscale_cnn_sequence_only_v1`、unique-group representative rule、batch 128 groups、patience 1、selected-epochs refit、Adam、LR 0.0003、weight decay 0.0001、gradient clip 1.0、seed 42、threshold 0.5、固定百分比 Label、augmentation/class weight=`none` 均不變 |
| Validation／OOS | 完整 rows 與既有 `1/group_size` 正式 group-weighted 評估；不套用日期平衡 |
| Dataset／Label | 不重建 feature bank、不 relabel；由既有 event date 與 split 即時計算 |
| 工件隔離 | 新 profile=`unique_group_date_balanced`，與 8F `unique_group_sampling` 分開存放 |
| Selection／OOS | 尚未取得 |
| 判定 | 尚不得判定有效或無效 |
| 下一步 | 執行完整 workflow，直接與 8F 比較 OOS Precision、Recall、Accuracy、Score及三項 Selection→OOS gap |

---

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

---

## 5. 接下來要嘗試的列表

所有實驗一次只改一項。後續訓練方法與輸入實驗預設固定使用目前 accepted baseline architecture `multiscale_cnn_sequence_only_v1`；只有明確重現歷史對照時才使用 `multiscale_cnn_v1`。Seed 42、threshold 0.5、no class weight與 no time weight 固定不變；Final Refit 正式基準維持 `selected_epochs`。8J 只作為獨立實驗測試直接使用 best inner checkpoint。

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

| 項目 | 固定設計 |
|---|---|
| 狀態 | `IMPLEMENTED`；完整 OOS 尚未取得 |
| 基準 | 8F `multiscale_cnn_sequence_only_v1 / unique_group_sampling / batch 128 / patience 1 / selected_epochs` |
| 唯一變更 | Training 仍保留全部 unique `ticker/date` groups，但將每個交易日的總 loss contribution 平衡為相同量級；同日每個 group 的 raw weight=`1 / 當日 eligible group 數`，再於 training split 內正規化為平均權重 1 |
| 目的 | 8F 已消除同一事件多個 `high_len` rows 的重複；8K 進一步降低市場全面突破時，單一事件密集日期因 group 數多而支配 optimizer，測試日期群聚是否是剩餘跨時期偏移來源 |
| Loss 契約 | 使用 globally normalized date weights，loss 以固定 batch denominator 聚合，不以各 batch 的 weight sum重新正規化，避免隨機 batch composition 破壞每日期等權語意 |
| 固定條件 | Architecture、unique-group representative rule、batch 128 groups、patience 1、selected-epochs refit、Adam、LR 0.0003、weight decay 0.0001、gradient clip 1.0、seed 42、threshold 0.5、固定百分比 Label及 augmentation/class weight=`none` 均不變；唯一新的 training weight mode 為 `date_balanced` |
| Validation／OOS | 維持完整 rows 與既有 `1/group_size` 的正式 group-weighted 未加日期權重評估；日期平衡只作用於 training loss |
| Dataset／Label | 不需重建 feature bank、不需 relabel；由既有 event date 與 split 即時計算 training weights |
| 採用條件 | 相較 8F，OOS Precision／Accuracy／Score至少兩項改善，Recall不得下降超過 3 pp，且 Precision、Accuracy、Score主要 Selection→OOS gap不得整體惡化 |


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
→ 8K unique-group date-density balancing（IMPLEMENTED；待完整 OOS）
```

任何新結果都必須追加至第 3 節，並同步更新第 2 節目前基準、第 4 節排除方向與第 5～6 節待辦順序。

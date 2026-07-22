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

本文件只記錄已知事實。歷史結果若缺少完整報表，會標記「精確值未保留」，不得自行補值。歷史資料整理截止日為 **2026-07-22**。

---

## 2. 目前基準

### 2.1 程式基準

| 項目 | 目前狀態 |
|---|---|
| 基準 ZIP | `test-branch-1_20260722_130420_bc854a4.zip` |
| SHA256 | `f9b10c512168c5dae250a1c745a7a4fbe3d5132ec804c93ce32f37d48e6753ad` |
| 程式版本範圍 | v9 連續輔助目標加入前；本輪將 model architecture 與 training experiment profile 分離，取消假版本 v10 |
| Policy 預設 | architecture=`multiscale_cnn_v1`；experiment profile=`adamw_only` |
| 當前最佳研究模型 | `multiscale_cnn_v1` |
| Dataset | Full；6A optimizer 變更不需重建 |

使用者所稱「退回 v8 版本」是退回**尚未加入 v9 auxiliary head 的程式版本**；目前正式研究基準模型仍是結果最佳的 `multiscale_cnn_v1`，不是把 policy 預設改成 `multiscale_cnn_v8`。

### 2.2 固定 Label 與訓練條件

| 項目 | 固定值 |
|---|---:|
| Feature Window | 300 bars |
| Label Horizon | 40 bars |
| PASS 最低 MFE | 嚴格大於 5% |
| PASS 最低 MFE／MAE | 嚴格大於 2.0 |
| 最大不利跌幅 | 觸及 −10% 即 REJECT |
| Epoch 上限 | 100 |
| Batch Size | 128 |
| Optimizer | `adamw`（由 `adamw_only` profile 決定；6A 接受與否待 OOS 結果） |
| LR Schedule | `none` |
| Augmentation | `none` |
| Learning Rate | 0.0003 |
| Weight Decay | 0.0001 |
| Gradient Clip | 1.0 |
| Threshold | 0.5 |
| Seed | 42 |
| Final Refit | `selected_epochs` |
| Class Weight | `none` |
| Time Weight | `none` |

### 2.3 正式比較基準：`multiscale_cnn_v1`

| OOS 指標 | v1 基準 |
|---|---:|
| 原始 PASS | 55.63% |
| PASS Precision | 59.29% |
| Precision Lift | +3.66 pp |
| PASS Recall | 51.49% |
| 模型 PASS | 48.32% |
| Accuracy | 53.34% |
| 平均 Score | 0.4791 |
| Selection→OOS Precision 差 | −2.85 pp |
| Selection→OOS Score 差 | −0.1129 |

成功判定不能只看平均 Score 或 Recall。新實驗至少應同時檢查 Precision Lift、Recall、Accuracy、模型 PASS，以及 Selection→OOS 落差。

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
| 狀態 | `IMPLEMENTED`；等待使用者完成 Full workflow 並提供 Selection／OOS 報表 |
| 程式基準 | `test-branch-1_20260722_115701_9cd60c8.zip`；SHA256 `366b0f1a15f6e77cbbee32f94b9275fe2a94676fb747e6fd0e4b817b06e2059c` |
| 架構整理 | 原 6A 以 `multiscale_cnn_v10` 隔離 optimizer 的做法已撤回；v10 未取得結果並被本輪實作取代 |
| 唯一學習變更 | `multiscale_cnn_v1 / baseline` 的 `torch.optim.Adam` → `multiscale_cnn_v1 / adamw_only` 的 `torch.optim.AdamW` |
| Experiment profile | `adamw_only`：optimizer=`adamw`、LR schedule=`none`、augmentation=`none` |
| 工件隔離 | `models/.../multiscale_cnn_v1/adamw_only/` 與 `outputs/.../multiscale_cnn_v1/adamw_only/`；不再新增模型版本 |
| Active／Legacy | 正常新訓練只允許 v1；v2～v8、Tiny、Residual 僅保留舊工件載入與歷史重現 |
| 固定條件 | LR 0.0003、weight decay 0.0001、seed 42、threshold 0.5、`selected_epochs`、class/time weight 均為 `none` |
| Dataset／Label | 不重建、不 relabel |
| Selection／OOS 結果 | 尚未取得 |
| 與 v1 baseline 差異 | 尚未取得 |
| 判定 | 尚不可接受或淘汰；取得結果前維持 `IMPLEMENTED` |
| 下一步 | 先取得 6A 結果，再決定 6B 使用 AdamW 或退回 Adam |

Formal bundle 閉環紀錄：2026-07-22 本地正式測試的 consistency 僅失敗 1 項：互動式 report 已正確傳遞 `--experiment-profile`，但 synthetic CLI contract 仍使用重構前的舊 expected argv。已將測試 fixture 隔離覆寫為 `baseline`，並把 expected argv 同步為 `--filter-id synthetic_quality --experiment-profile baseline --include-oos`。這是 validator 契約同步修正，不改 runtime、模型、Dataset 或 6A 學習條件；6A 狀態仍為 `IMPLEMENTED`。

### 3.8 Architecture／Experiment Profile 管理規則（2026-07-22）

- `multiscale_cnn_v1` 是目前唯一 active architecture。
- `multiscale_cnn_v2～v8`、`tiny_cnn_v1`、`residual_tcn_v1` 均為 legacy read-only compatibility；保留程式碼不代表仍是正式候選。
- optimizer、LR schedule、augmentation、loss weighting 等訓練差異只可新增 experiment profile，不可再建立 v10、v11 等假模型版本。
- `baseline` 與 `adamw_only` 使用相同 v1 模型圖與初始化；工件依 profile 子目錄隔離。
- 舊 v1 baseline 工件的無 profile 歷史路徑只提供唯讀 fallback；不得用它覆寫 manifest 或匯出正式 forward-OOS scores。新訓練與正式輸出一律寫入 `<architecture>/<experiment_profile>/`。

---

## 4. 已排除或暫停的方向

下列方向已有足夠證據，不應在沒有新機制或新資料證據時重複測試：

1. 退回 Tiny CNN。
2. Residual TCN 加深／加大容量。
3. Short／Medium／Long 的更多 Level／Return 組合。
4. 相對 0050 Return 的同資訊線性重組。
5. Long Branch channels 8～16 間的細部搜尋。
6. Long Branch dropout 的細部搜尋。
7. Matched optimizer steps 作為正式 refit。
8. Inverse-frequency class weight 或 year-balanced time weight。
9. 三個連續 auxiliary targets 同時加入。
10. 只把 OOS 縮短成下一年，期待模型自然改善。
11. 把 multi-fold validation 誤當成會直接提高 OOS 的模型改動。

---

## 5. 接下來要嘗試的列表

所有實驗一次只改一項，固定使用 active architecture `multiscale_cnn_v1`、seed 42、threshold 0.5、no class weight、no time weight、`selected_epochs`。每個實驗都不因訓練策略變更而重建 Dataset。

### 優先 6A：AdamW only

| 項目 | 設計 |
|---|---|
| 狀態 | `IMPLEMENTED`；等待結果 |
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
| 狀態 | `PLANNED` |
| 唯一變更 | 在 6A 決定後保留的正式 optimizer 上加入 step-based schedule |
| Warmup | 前 5% optimizer updates 線性 warmup |
| Decay | 後 95% 使用 cosine decay |
| Minimum LR | 初始 LR 的 10%，即 0.00003 |
| 更新單位 | 每次 optimizer update，不是每個 epoch |
| Refit | Inner training 與完整 Selection refit 各自依實際 total steps 重建 scheduler |
| Dataset rebuild | 不需要 |
| 目的 | 降低後期 overshoot，觀察是否改善 OOS 而非只降低 Selection loss |

若 6A 被接受，6B 以 AdamW 為基準；若 6A 被拒絕，6B 回到 Adam 後只加入 scheduler，確保仍是單一變更。

### 優先 7A：輕量舊歷史 contiguous masking augmentation

| 項目 | 設計 |
|---|---|
| 狀態 | `PLANNED` |
| 唯一變更 | Training batches 加入一段舊歷史 masking |
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

只有 7A 至少改善一項 OOS 泛化指標且未明顯傷害 Precision Lift 時才進行。第一版只對 training input 加極小幅、依 channel 尺度限制的零均值 noise；不得與 masking 在第一次 noise 實驗同時啟用。若 7A 明確失敗，先停止 augmentation 路線，不直接堆疊更多擾動。

### 後續結構性方向

只有 6A、6B、7A 都未改善時，才依序評估：

1. **明確市場 regime context**：加入 0050 的 20／60 日報酬、波動率，以及個股相對強弱等低維 context；這不同於 v3 對既有 Return channels 做線性重組。Feature contract 改變時需要重建 feature bank／Dataset。
2. **ATR／波動率尺度 Label**：把固定 5% MFE 與 −10% adverse barrier 改為當時可觀測的 ATR／volatility 倍數，降低不同市場 regime 的 Label 尺度漂移。應優先使用既有 future-path cache 快速 relabel，是否重建由 Dataset policy 自動判定。
3. **基於 Selection 決定的 rank／percentile gate**：只在模型可分性尚可、但固定 0.5 明顯受 score drift 影響時測試；不得使用 OOS 回頭選 percentile。

---

## 6. 實驗執行順序

```text
6A AdamW only
→ 依 6A 結果決定保留 Adam 或 AdamW
→ 6B 只加入 step-based LR schedule
→ 固定最佳 optimizer / schedule
→ 7A 只加入舊歷史 contiguous masking
→ 視 7A 結果決定是否進入 7B
→ 若仍無改善，再進入 regime context 或 volatility-scaled Label
```

任何新結果都必須追加至第 3 節，並同步更新第 2 節目前基準、第 4 節排除方向與第 5～6 節待辦順序。

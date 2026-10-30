# Breakout Quality — 已完成實驗結果與研究決策

## 1. 用途與唯一日誌規則

本文件是 `breakout_quality` **唯一的實驗結果與研究決策日誌**。只保留已測量的控制實驗（包括未通過／負面結果）、對照條件、定量指標、有效性更正、採納／淘汰原因及研究證據位置。**不建立第二份 Markdown、gzip／zip 封存日誌或工程修補年表。** 舊文中不符合這些規則的過程性內容已刪除。

- 科學 identity、歷史 ID 保留與 current eligibility 的 SSOT：`doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md`；目前參數、候選、模型與比較名單由 canonical `config/` 與正式簽章工件決定，**不能把歷史紀錄中的「當前／下一步」當成現行設定**。
- 每個受控研究至少留下：研究 ID、唯一變數／共同條件、評估期間或樣本、Seed、與控制組比較的**實際數值**、結果來源與限制、接受／拒絕／未決決策；沒有實際數據不以 synthetic PASS 冒充科學結果。
- 同一 OOS 是**反覆使用的迭代研究 OOS 證據**，不是 untouched holdout；OOS 數值不可回饋訓練／調參／閾值擬合。NON-PIT Oracle 僅是 hindsight 反事實，不是可部署模型。
- 已知錯誤導致的歷史數值失效，須保留有用的**原數值及其失效標記**，並記下修正後的結果與合法比較範圍；不可把原結果默默替換成 current truth。
- 純工程修補、Test Suite／synthetic、REUSE／RUN 事故、匯入路徑、相容層、選單格式、commit、ZIP／SHA 清單及過時研究待辦，**不得新增至本文件**。若 Bug 改變研究結果的有效性，只在受影響研究旁記一次有效性更正。
- 後續每個研究只增補一筆短的結果與決策；完整巨大表格若已有正式 `result.md`／signed manifest 可只保留關鍵指標及正式工件定位，不重貼逐輪 Console。

## 2. 2026-07 基準、消融與負面實驗（歷史）

以下設定及各項結果是**當時的研究基準**，不是現在的正式 `config` 預設，也不授權復活已淘汰的科學 identity。

### 歷史訓練條件

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

### 正式研究基準：MR-8F（8F）`multiscale_cnn_sequence_only_v1 / unique_group_sampling`

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


### 新排序／高品質基準：9A `inception_time_v1 / unique_group_sampling`

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

### 2026-07 · 已完成實驗

#### 3.1 架構基線

| 日期 | 實驗 | 唯一主要變更 | 結果摘要 | 判定 |
|---|---|---|---|---|
| 2026-07-12～13 | `tiny_cnn_v1` | 小型兩層 CNN | Train loss 約只由 0.67 降至 0.65，Selection 本身即不足 | `REJECTED`；不要再退回 Tiny |
| 2026-07-13 | `residual_tcn_v1` | 6 個 residual dilated blocks | Train／Selection 學得更強，但 OOS 過度擬合明顯 | `REJECTED` |
| 2026-07-13～14 | `multiscale_cnn_v1` | Short／Medium／Long 三分支 Level input | 相較 Tiny 可學到有效 Selection 訊號；後續所有消融的正式基準 | `ACCEPTED` |

#### 3.2 Refit、類別權重與時間權重

| 日期 | 實驗 | 唯一變更 | OOS Precision Lift | OOS Recall | 模型 PASS | 判定 |
|---|---|---|---:|---:|---:|---|
| 2026-07-14 | Matched optimizer steps only | `selected_epochs` → `matched_optimizer_steps` | +4.55 pp | 31.49% | 29.11% | `REJECTED`；過度保守 |
| 2026-07-14 | No class weight only | `inverse_frequency` → `none` | **+3.66 pp** | **51.49%** | **48.32%** | `ACCEPTED`；目前 v1 基準 |
| 2026-07-14 | Year balancing only | `none` → `year_balanced_sqrt` | 精確值未保留 | 精確值未保留 | 精確值未保留 | `REJECTED`；整體較差 |
| 2026-07-14 | Matched steps + no class weight | 同時使用 matched steps 與 no class weight | +4.14 pp | 41.09% | 38.25% | `REJECTED`；仍過度篩選 |

#### 3.3 Branch input representation

| 日期 | 版本 | 唯一變更 | OOS Precision | Lift | Recall | 模型 PASS | Accuracy | Score | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-07-14 | v1 | Short／Medium／Long 全部 Level | **59.29%** | **+3.66 pp** | 51.49% | 48.32% | **53.34%** | 0.4791 | `ACCEPTED` |
| 2026-07-14 | v2 | Short＋Medium 改 Return／Delta | 57.35% | +1.72 pp | **56.44%** | **54.74%** | 52.42% | **0.5118** | `REJECTED`；提高 Score／Recall、降低辨識精度 |
| 2026-07-14 | v3 | v2 的個股 O/H/L/C 再減 0050 同欄 Return | 57.59% | +1.96 pp | 54.77% | 52.91% | 52.40% | 0.4995 | `REJECTED`；與 v2 資訊等價，無實質改善 |
| 2026-07-14 | v7 | 只有 Short 改 Return／Delta | 58.34% | +2.71 pp | 53.16% | 50.69% | 52.83% | 0.4870 | `REJECTED` |
| 2026-07-15 | v8 | 只有 Medium 改 Return／Delta | 57.70% | +2.07 pp | 53.91% | 51.97% | 52.38% | 0.4905 | `REJECTED` |

共同結論：Return／Delta 使用越多，通常 Score、模型 PASS 與 Recall 越高，但 Precision Lift 與 Accuracy 越低；主要是改變固定 threshold 0.5 的操作點，未提升整體可分性。不要再測 Long Return 或更多 Return／Level 組合。

#### 3.4 Long Branch 容量與 regularization

| 日期 | 版本 | 唯一變更 | OOS Precision | Lift | Recall | 模型 PASS | Accuracy | Score | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-07-14 | v4 | Long channels 16 → 8 | 58.91% | +3.27 pp | 52.88% | 49.94% | 53.26% | **0.4806** | `REJECTED`；接近 v1，但未超越 |
| 2026-07-14 | v5 | Long channels 16 → 12 | 58.00% | +2.37 pp | 53.69% | 51.50% | 52.61% | 0.4779 | `REJECTED` |
| 2026-07-14 | v6 | Long dropout 0.25 → 0.40 | 58.53% | +2.90 pp | 49.70% | 47.24% | 52.43% | 0.4662 | `REJECTED`；主要指標一起退步 |

共同結論：縮減 channels 或提高 Long Branch dropout 沒有提升真正 OOS 辨識力。不要再測 Long channels 10／14、dropout 0.30／0.35／0.50 或其他 Long Branch 微調。

#### 3.5 連續輔助目標：v9（2026-07-21）

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

#### 3.6 OOS horizon／更新頻率

| 日期 | 實驗 | 結果 | 判定 |
|---|---|---|---|
| 2026-07-21 | 只預測下一年 OOS | 使用者已實測，沒有比目前整段 OOS 更好；精確報表值未保留 | `REJECTED`；不要把縮短 OOS horizon 或年度 refit 當成近期優先方向 |
| 2026-07-21 | Multi-fold validation | 尚未作為模型改善手段；它只提高選模可靠度，不會自行改善單一 fold 的模型能力 | 不列為目前「提升整體 OOS」的直接實驗 |


#### 3.7 Optimizer 6A：AdamW only（2026-07-22）

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

#### 3.8 LR Schedule 6B：Adam + step-based warmup／cosine（2026-07-22）

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

#### 3.9 Augmentation 7A：舊歷史 contiguous masking（2026-07-22）

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


#### 3.10 結構性實驗：低維市場 Regime Context（2026-07-22）

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

#### 3.11 ATR／波動率尺度 Label（使用者最新結果，2026-07-22）

| 項目 | 紀錄 |
|---|---|
| 狀態 | `REJECTED` |
| 程式基準 | 使用者已實測；本輪提供的 bundle 未保留該次完整程式 ZIP、SHA256 與精確報表 |
| 唯一主要變更 | 固定百分比 MFE／adverse barrier 改為事件日可觀測 ATR／波動率尺度；其他詳細倍數以當次實驗為準，本文件不自行補值 |
| Dataset／Label | 需要 relabel；是否重建 feature bank 由當次 Dataset policy 決定 |
| Selection／OOS 結果 | 使用者確認整體 OOS 沒有比固定百分比 Label 更好；精確指標未保留 |
| 判定 | ATR 尺度沒有解決目前整體 OOS 泛化問題；退回固定百分比 Label，不再列為近期方向 |
| 下一步 | 改測輸入與目標學習機制，不再調 Label 尺度 |

#### 3.12 Architecture／Experiment Profile 管理規則（2026-07-22）

- `multiscale_cnn_sequence_only_v1` 已取得完整 OOS 結果並升為目前 accepted architecture 與正式研究基準。
- `multiscale_cnn_v1` 保留為 active historical comparator；其完整 OOS 證據仍有效，但新實驗預設固定從 sequence-only 基準延伸，除非實驗目的明確要求比較 context 有無。
- `multiscale_cnn_regime_context_v1`、`multiscale_cnn_v2～v8`、`tiny_cnn_v1`、`residual_tcn_v1` 均為 legacy read-only compatibility；保留程式碼只供舊 checkpoint／manifest 重建與歷史重現。
- optimizer、LR schedule、augmentation、loss weighting 等訓練差異只可新增 experiment profile，不可再建立 v10、v11 等假模型版本。
- `baseline`、`adamw_only`、`adam_warmup_cosine`、`history_masking_only` 與 `recent_decay_60m` 是訓練 profile；工件依 architecture／profile 子目錄隔離。`recent_decay_60m` 已被 8B 完整 OOS 淘汰，不得再作正式新實驗入口。
- 舊 v1 baseline 工件的無 profile 歷史路徑只提供唯讀 fallback；不得用它覆寫 manifest 或匯出正式 forward-OOS scores。新訓練與正式輸出一律寫入 `<architecture>/<experiment_profile>/`。

#### 3.13 8A Sequence-only context ablation（2026-07-22）

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

#### 3.14 8B Recent-decay time weighting（2026-07-22）

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

#### 3.15 8C Same-day cross-sectional rank context（2026-07-23）

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

#### 3.16 8D Hybrid BCE + same-day pairwise ranking loss（2026-07-23）

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

#### 3.17 8E Fixed 8-seed probability ensemble（2026-07-23）

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

#### 3.18 8F Unique ticker/date group training（2026-07-23）

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

#### 3.19 8G Unique-group sampling + patience 5（2026-07-24）

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

#### 3.20 8H Unique-group sampling + batch size 64（2026-07-24）

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

#### 3.21 8I Unique-group sampling + matched optimizer steps refit（2026-07-24）

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


#### 3.22 8J Direct best inner-validation checkpoint（2026-07-24）

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

#### 3.23 8K Unique-group date-density balancing（2026-07-24）

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


#### 3.24 8L Date-diverse batch scheduling（2026-07-24）

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

#### 3.25 8M Long Branch last-state pooling（2026-07-24）

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

#### 3.26 8O Zero-initialized gated temporal pooling（2026-07-24）

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

#### 3.27 8P Raw＋window-normalized dual-path（2026-07-24）

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


#### 3.28 9A 單一 InceptionTime（2026-07-24）

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

#### 3.29 9A-GN InceptionTime GroupNorm ablation（2026-07-25）

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



#### 3.30 9B ModernTCN（2026-07-25）

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

#### 3.31 9C TS2Vec Selection-only自監督預訓練（2026-07-25）

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

#### 3.32 9D MantisV2 frozen encoder＋linear probe（2026-07-26）

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


#### 3.33 9E MOMENT-1-base frozen encoder＋linear probe（2026-07-26）

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

#### 3.34 9F：小型 Patch Transformer supervised architecture（2026-07-26）

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


#### 3.35 Unique-group重跑確認與固定9A策略對照入口（2026-07-26）

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


#### 3.60 Selection Point-in-time Score Workflow 第一階段（2026-08-01）

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


#### 3.61 Formal quick-gate 舊 Strategy Compare Help 閉環（2026-08-01）

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


#### 3.62 Selection PIT Scores完成與Orderable Coverage欄位碰撞修正（2026-08-01）

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


#### 3.63 泛用Profile Workflow Router修正（2026-08-01）

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


#### 3.64 Formal Consistency Canonical Strategy-Compare Fixture閉環（2026-08-01）

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


#### 3.65 Breakout-quality Config單一來源整併（2026-08-01）

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


#### 3.66 單一Config刪檔後Formal閉環（2026-08-01）

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


### 2026-07 · 已排除或暫停的方向

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

## 3. 2026-08～10 已完成研究結果與有效性更正

依研究紀錄原有日期排列；每節是當時的已執行證據與判斷，不代表現在可用的同名實驗或當前名單。已移除原本「沒有測試結果的實作日誌」與過時的研究 Queue。

### 2026-08-03 — Ranking × Parameter正式2×2結果與capital-aware ranking下一步

#### 狀態

`RESULT_AVAILABLE / SCORE_ADAPTED_PARAMS_REJECTED / ADAPTED_SYSTEM_REJECTED`

本節依使用者提供的完整正式輸出摘要回寫。原始`rolling_validation/`結果工件與完整console輸出未包含於本ZIP，故數值視為已提供的正式結果紀錄，本輪未重新執行optimizer或portfolio replay。

#### 程式基準

- ZIP：`test-branch-1_20260803_122159_172cc87(1).zip`
- SHA256：`5c8653168fe3e42d532275466e18e6e0fce839ae08adf35ecd09c18a518f9f5a`
- 比較期間：2014-01-01～2020-12-31 Selection PIT
- 正式2×2：Baseline＝舊ranking＋舊ROOS；Sort Only＝Score ranking＋舊ROOS；Param Only＝舊ranking＋同一套Score Adapted params；Adapted＝Score ranking＋同一套Score Adapted params

#### 固定條件

Dataset、Continuous Target、Selection PIT Scores、PIT模型驗證、Score來源、候選filters、entry／stop／exit、fixed risk、position cap、max positions、rotation、交易帳務與Future Target post-replay-only契約不變。新參數只由Score-ranking rolling optimizer訓練一次；Param Only與Adapted使用完全相同的active-param schedule。

#### 主要結果

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

#### 歸因與判定

- `Sort Only − Baseline`與`Adapted − Param Only`均顯示Score ranking提高Future Target與單筆Realized R，但降低資金投入、平均曝險及總報酬。
- 共同機械鏈為：Score ranking偏好較寬初始停損候選；fixed-risk sizing下每股風險提高，買入股數與每slot投入金額隨之下降，平均曝險約降低22～23個百分點。
- 現行No-time Target描述未來價格機會，未直接納入實際初始停損距離、固定風險後部位金額、剩餘現金或正式entry／stop／exit capture效率。因此「排序能力PASS」不等於「投組經濟效果PASS」。
- PIT模型排序能力與較高Future Target選擇維持`PASS`；舊參數Score ranking、Score Adapted params與完整Adapted系統均為`REJECTED`；正式策略維持Baseline。

#### 下一個單一研究方向

優先建立capital-aware ranking／停損距離消融的正式實驗契約，先固定舊正式ROOS與全部交易規則，只改候選排序：原ranking、原始Score ranking、Score加停損距離約束、停損距離分桶後桶內Score ranking。第一階段只做程式契約與可回放診斷，不重訓模型、不重新執行rolling optimizer，也不直接提高fixed risk、position cap或max positions。

後續第二順位為Filter × Score分組消融；第三順位才是Selection PIT Score向前回補可行性。現有7 folds training calendar coverage為0%、10%、20%、30%、40%、50%、60%，目前不足以單獨證明coverage是Adapted失敗主因。

### 2026-08-03 — Selection PIT向前延伸與coverage提升後共同folds 2×2

#### 狀態

`RESULT_AVAILABLE / COVERAGE_IMPROVEMENT_CONFIRMED / COVERAGE_NOT_PRIMARY_FAILURE / SCORE_ADAPTED_PARAMS_REJECTED / ADAPTED_SYSTEM_REJECTED`

使用者已在本機完整資料完成Selection PIT向前延伸、PIT audit、Score Adapted rolling optimizer及四組2×2 replay。實際PIT起點由`2014-01-01`向前延伸至`2011-01-01`；7個Baseline共同fold全部保留，weighted training calendar coverage由30.0%提高至60.0%，每fold均提高30.0pp。coverage提升目標已達成，但Score ranking在舊參數與新Adapted參數下仍明顯降低投組報酬，因此coverage不足不再列為主要失敗原因。

#### 程式基準

- 輸入ZIP：`test-branch-1_20260803_184631_c665cb2(1).zip`
- SHA256：`e75d9295d50b117f71ac6541a75e26937db822ba9f886cd33f82e65fe7474e9e`
- 本輪修改只以該ZIP全新解壓內容為基準；交付patch ZIP的SHA256以本輪回覆列示。

#### 唯一主要變更

PIT builder既有`auto`最早合法日期、日期型fold identity與舊fold安全重用維持不變。Strategy Adapt的coverage門檻由「只保留100% coverage尾段folds」改為「保留全部原Baseline共同folds，延伸後training coverage必須相較舊正式`2014-01-01`參考起點提升」。此變更只調整rolling比較樣本與coverage判定，不改模型架構、Target公式、loss、Seed、fold長度或策略交易規則。

#### 實作內容

1. PIT fold ID維持穩定日期鍵`fold_YYYYMMDD_YYYYMMDD`；向前新增fold不會改變既有期間identity。
2. Builder resume仍只在日期、information cutoff、資料來源、模型規格、training settings、group／event counts、lookahead契約與checkpoint／Score SHA256完全一致時，遷移或重用舊fold。
3. `score-start-date=auto`仍由Dataset、Continuous Target、PASS-only scope、`label_eval_end_date`、24個月Validation與最小group門檻解析最早合法月份；`plan-only`只輸出日期與fold計畫，不訓練或寫入正式Score。
4. 正式build完成後重建合併Scores、coverage CSV、manifest與PIT audit；既有相容fold可重用，新增較早fold必須實際訓練。
5. 新增`BREAKOUT_QUALITY_POINT_IN_TIME_COVERAGE_REFERENCE_START_DATE="2014-01-01"`作為舊正式coverage參考，不限制實際PIT起點。每個Baseline training window同時計算reference與actual calendar coverage、差值及狀態。
6. 正式2×2保留全部Baseline rolling folds與原OOS期間。actual coverage必須在所有fold不低於reference，至少一個fold嚴格改善，且加權總coverage提高；實際PIT起點之前允許回退原buy-sort，實際PIT期間內缺口仍禁止。Baseline、Sort Only、Param Only與Adapted使用相同fold schedule、期間、PIT identity、risk、position cap、max positions、rotation及帳務契約。

#### Dataset／Target／模型工件需求

- Dataset：不因程式變更重建；本機必須存在目前完整`breakout_quality_v1` Dataset及與其SHA256一致的Continuous Target。
- Label／Continuous Target：公式與arrays不變，不relabel；若本機Target因Dataset更新而stale，仍由既有workflow先重建。
- PIT模型：最早合法日期到既有2014年前的fold需要新訓練；2014後舊fold只有在完整契約與hash完全一致時才遷移重用。
- PIT audit：合併Scores／manifest改變後必須重新執行，舊audit因來源SHA256不同不得沿用。
- Strategy optimizer：只在PIT audit通過、全部OOS位於PIT期間、逐fold coverage未退步且整體coverage提升後執行；不再因未達100%而排除fold或中止。

#### 固定條件與結果邊界

不改9A架構、No-time Target、daily percentile regression、PASS-only scope、Seed 42、12個月PIT fold、24個月inner validation、optimizer search space、trials來源、fixed risk、position cap、max positions、rotation、原始Score ranking、entry／stop／exit、portfolio accounting或Future Target post-replay-only契約。此輪不判定新ranking有效或無效；先前2014～2020 Baseline／Sort Only與原混合coverage 2×2結果仍保留為歷史證據，新的coverage提升版2×2必須待本機完整執行後另行回寫。

#### 本機正式執行順序

以正式選單操作：

1. 執行`apps/breakout_quality.py`，主選單按`Enter`進入「模型研究與驗證」，確認建立或更新Dataset、Continuous Target、向前延伸PIT Scores並完成PIT模型驗證。
2. 返回主選單選`[2] 策略績效驗證`。
3. 在策略績效驗證選`[2] 驗證策略參數適應（PIT coverage提升即可）`，由流程自動完成共同fold coverage preflight、Baseline／Sort Only重建或重用、Score Adapted rolling optimizer及四組2×2 replay。

#### 獨立驗證

- PIT builder既有日期型fold identity、auto最早合法日期、舊fold遷移與resume契約維持，未修改其模型／資料邏輯。
- Strategy Adapt新增獨立synthetic契約：actual起點2007相較reference起點2014時，四個Baseline folds全部保留、逐foldcoverage均改善、共同比較期間維持2014～2018；actual與reference相同時必須拒絕。
- Coverage報表同步輸出reference、actual、提升pp、加權coverage gain與各fold狀態；runtime identity明確記錄pre-PIT fallback允許、100%非必要及PIT期間gap禁止。
- 上述程式契約已由使用者本機完整資料執行並取得以下正式結果；正式`apps/test_suite.py`是否已於本機通過，使用者本輪未另提供bundle。

#### 本機正式結果（coverage提升後）

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

#### 與原30% coverage結果比較

舊結果為Baseline 182.62%、Sort Only 144.80%、Param Only 65.46%、Adapted 59.71%；本次分別為182.62%、147.57%、145.84%、75.70%。Param Only與Adapted數值明顯提高，但本次Adapted trials／fold為300，而舊執行紀錄使用200 trials／fold，因此兩次結果不是只改coverage的單一控制實驗，不能把全部差異歸因於coverage。可確定的正式結論應以本次同一2×2內的對稱比較為準：

- 舊ROOS固定時，`Sort Only − Baseline = −35.05pp`，MDD增加8.38pp，Return／MDD下降7.01。
- 新Adapted params固定時，`Adapted − Param Only = −70.14pp`，EV由0.53R降至0.37R，Aggregate capture由0.77降至0.35。
- Score Adapted params本身亦未打敗舊ROOS：`Param Only − Baseline = −36.78pp`；在Score ranking下，`Adapted − Sort Only = −71.87pp`。
- 因此coverage提升改善了optimizer可見的新ranking歷史比例，但沒有使Score ranking或Score Adapted params通過正式投組門檻。

#### 歸因與採用判定

1. Score排序能力仍有效：兩套參數下，Target percentile、top-k retention、Target opportunity gap與Target mean均改善，表示模型確實挑到事後價格機會較高的候選。
2. 固定舊參數時，Score ranking的平均Realized R由0.28R升至0.30R、平均投入資金報酬由3.00%升至3.85%，但初始停損距離由6.39%擴大至9.23%，平均投入由164,608降至107,910，平均曝險由77.33%降至55.02%。
3. 同一套新Adapted params下，初始停損距離由5.58%擴大至8.84%，平均投入由147,686降至71,805，平均曝險由81.00%降至54.28%；同時Realized R、Payoff、勝率與capture亦下降，顯示不只資金利用率，策略capture契約也與Score排序不相容。
4. Score ranking反而減少未滿倉日與持股缺口，但美元曝險仍大幅下降，證明主因不是候選供給或持倉格填不滿，而是每個slot因寬停損及fixed-risk sizing而配置過小。
5. 正式策略維持Baseline；原始Score ranking、Score Adapted params及完整Adapted系統均`REJECTED`。Selection PIT向前延伸工件保留，供後續同一PIT source的研究重用。

#### 下一個單一研究方向

下一步固定舊正式ROOS、fixed risk、position cap、max positions、entry／stop／exit、rotation與帳務，只做capital-aware ranking／停損距離消融，不重訓模型、不重新執行rolling optimizer。第一輪比較：

1. 原正式buy-sort。
2. 原始Score第一排序。
3. `Score × projected capital fraction`，其中projected capital fraction必須沿用正式盤前sizing單一真理結果，不另寫近似公式。
4. 依正式預估投入比例或初始停損距離作固定分桶，再於同桶內依Score排序；分桶邊界須由Selection train-only分布或固定可解釋契約決定，不得由2014～2020回放績效調參。

第一階段只判斷能否在維持Target mean／Realized R優勢下恢復平均投入、曝險、總報酬與Return／MDD。Filter × Score分組消融維持第二順位；不再重跑相同原始Score Adapted optimizer，也不直接提高fixed risk、position cap或max positions。

### 2026-08-03 — Capital-aware Ranking R2／R3 CLI-only消融

#### 狀態

`RESULT_AVAILABLE / R2_REJECTED / R3_DIRECTION_PASS_FORMAL_POLICY_NOT_ACCEPTED`

#### 程式基準

- 輸入ZIP：`test-branch-1_20260803_211701_c0cb6c3.zip`
- SHA256：`1c8aa4144af00f7ae7696c90dbe649932650e89f16881da7a62b9d3382f6cc9e`
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

#### 實驗目的

Coverage提升後的正式2×2已確認原始Score ranking在舊ROOS與新Adapted params下都降低投組報酬；共同機械鏈是Score偏好較寬初始停損候選，fixed-risk sizing使每slot投入與平均曝險下降。下一個單一變更固定舊正式ROOS與全部交易規則，只調整Score候選排序，使模型品質與正式可部署部位共同進入ranking。

#### 唯一變更

新增`strategy-compare --ranking-policy`兩個CLI-only研究政策；正式預設仍為`score`，互動選單、optimizer、模型、Target與正式Baseline不變。

1. R2 `capital-adjusted-score`：沿用正式候選已完成exact-accounting後的`proj_cost`，計算`projected_capital_fraction = proj_cost / sizing_capital`及`deployment_rate = min(1, projected_capital_fraction / max_position_cap_pct)`；有效Score候選依`Score × deployment_rate`降冪，再沿用既有buy-sort。
2. R3 `capital-bucket-then-score`：只在當日有效PIT Score候選中，依正式`deployment_rate`的當日橫斷面1/3與2/3分位分成高／中／低三桶；先按桶別，再於桶內按Score，最後沿用既有buy-sort。相同部署率可落在同桶，不使用回放績效或Future Target決定邊界。
3. Active-param ensemble仍先按vote count；同票候選使用指定policy。部署率使用member有限值中位數。PIT缺分候選不排除、不填0，並完整回退原buy-sort；全缺分日期不得因capital-aware欄位改變pre-PIT排序。
4. 非原始policy輸出使用獨立目錄，避免覆蓋既有R1正式診斷。

#### 固定條件

- 參數：既有`base_finalist_best` rolling active params。
- Score：目前Selection PIT工件，實際起點`2011-01-01`。
- 比較期間：預定固定`2014-01-01～2020-12-31`。
- fixed risk、position cap、max positions、entry／stop／trail／exit、rotation、候選filters、交易成本、帳務與0050 benchmark全部不變。
- 不重建Dataset、不relabel、不重建Continuous Target、不重訓模型、不執行rolling optimizer。
- Future Target只可在portfolio replay完成後join作read-only診斷。

#### 實作與驗證

- `core/buy_sort.py`集中管理三種ranking policy及正式部署率衍生公式，避免R2與R3各自重算sizing。
- `core/portfolio_candidates.py`在候選形成後保存正式`projected_capital_fraction`、`deployment_rate`與policy；不修改股數或價格。
- `core/portfolio_engine.py`將相同policy套用於active-param ensemble，同時保存候選replay診斷欄位。
- `filters/breakout_quality/runtime.py`以ContextVar傳遞本次research ranking policy，預設仍為原始`score`。
- `strategy_compare.py`新增`--ranking-policy`、policy-specific排序metadata與隔離輸出目錄。
- T267既有direct synthetic case已擴充：驗證部署率公式、R2乘積排序、R3三分桶、全缺分fallback及ensemble vote優先；本輪不執行正式`apps/test_suite.py`。

#### Dataset／Label／模型工件需求

本次程式變更不需要重建Dataset、Label、Continuous Target、PIT Scores或checkpoint。正式本機執行只需要既有完整市場資料、Selection PIT manifest／audit／scores及`roos_base_best.json`。

#### 真實 Selection replay 結果

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

#### 歸因與採用判定

1. **R2拒絕目前公式。** `Score × deployment_rate`成功把曝險由原始R1約55%恢復至77.44%，證明資金部署確實是原始Score ranking的重要瓶頸；但R2同時使保留買單成交率降至95.85%、Target mean降至1.0450R、平均投入資金報酬降至2.79%，最終總報酬仍低於Baseline且MDD略高。連續乘積把Score百分位當成可線性縮放的經濟價值，並把既有執行排序壓到後面，未形成可採用證據。
2. **R3方向保留，但目前policy不採用。** 三分桶後桶內Score使Target percentile、top-k retention、Target mean、Realized R、Aggregate capture與平均投入資金報酬全部改善，總報酬184.12%亦略高於Baseline；這證明「先限制極小部位候選壟斷，再保留Score排序」是有效研究方向。
3. R3仍未通過正式風險邊界：MDD由13.18%升至19.20%、RoMD由13.86降至9.59、Log R²由0.9349降至0.9101、月勝率由61.90%降至59.52%、最差年度由-0.35%惡化至-3.93%。R3較低勝率、較高Payoff／EV、較長持有期與較寬停損，形成較不平滑且更集中於少數贏家的報酬路徑；不能只因總報酬略高即升格正式policy。
4. R2與R3都比原始R1總報酬147.57%大幅改善，確認capital-aware ranking修正方向成立；但正式策略仍維持R0 Baseline，R2標記`REJECTED`，R3標記`RESULT_AVAILABLE / RESEARCH_DIRECTION_PASS / FORMAL_POLICY_NOT_ACCEPTED`。
5. 本次不重建Dataset、Label、Continuous Target、PIT Scores或checkpoint，也不執行optimizer；Future Target只於replay完成後join。

#### 下一步

先執行R3最大回撤read-only attribution，不改ranking、不重訓模型、不跑optimizer：定位最大回撤起迄日期，逐日／逐交易比較Baseline與R3的獨有持倉、初始停損距離、deployment bucket、Score、Target、Realized R、持有期、損失聚集與產業／月份集中。先確認MDD惡化是由特定regime、跨桶硬排序、寬停損或損失時間聚集造成，再決定下一個單一變更。未完成此歸因前，不直接調整桶數、桶邊界、fixed risk、position cap或max positions；Filter × Score消融維持後續順位。

### 2026-08-03 — R3 Capital-bucket Ranking Rolling Parameter Adaptation

#### 狀態

`RESULT_AVAILABLE / R3_RANKING_CONFIRMED / R3_ADAPTED_PARAMS_REJECTED / FORMAL_BASELINE_UNCHANGED`

使用者判定R3已在總報酬、EV、Target選擇、Target capture與平均投入資金報酬形成足夠的多項改善，不先投入專門最大回撤歸因；下一個單一實驗改為固定R3 ranking契約，執行與既有Score Adapted相同的rolling optimizer及四組2×2。此決定取代前一節「先做R3最大回撤read-only attribution」的下一步，但不改變R3目前尚未升格正式policy的判定。

#### 程式基準

- 輸入ZIP：`test-branch-1_20260803_221024_d9ba32f.zip`
- SHA256：`273d6257fcc42b403ec1c588b67a3f227bd5c02102576da256e0663c26dd99bc`
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

#### 唯一主要變更

`tools/filters/breakout_quality/strategy_adapt.py`新增CLI-only `--ranking-policy capital-bucket-then-score`。未指定時仍維持既有原始`score`流程與互動選單行為；R3不加入主選單、不改正式策略預設，也不把ranking policy、桶數或桶邊界放入optimizer搜尋。

R3流程固定：

1. Baseline：原buy-sort＋舊正式ROOS。
2. R3 Sort Only：`capital-bucket-then-score`＋舊正式ROOS。
3. Param Only：原buy-sort＋同一套R3 Adapted active params。
4. R3 Adapted：`capital-bucket-then-score`＋同一套R3 Adapted active params。

只訓練一套R3 Adapted rolling optimizer；Param Only不是第二套optimizer，而是把同一套新參數切回原ranking的反事實回放。

#### Runtime與工件隔離

- R3 ranking policy進入optimizer runtime context、runtime cache identity、persistent study identity、Sort Only／Adapted replay context、preflight、summary與manifest。
- 原始Score Adapted沿用`models/research/breakout_quality/score_ranking_adaptation/rolling_validation/`。
- R3 Adapted使用獨立路徑`models/research/breakout_quality/score_ranking_adaptation/capital_bucket_then_score/rolling_validation/`。
- 兩種policy不得互相resume、覆蓋active params或重用Baseline／Sort Only pair manifest。
- Selection PIT正式models root與Adapted active-param輸出仍保持分離。
- active params內固定`use_breakout_quality_ranking=True`、hard filter=False、fixed risk與position cap；實際R3 policy由同一Selection PIT runtime context提供，不成為策略參數trial。

#### 固定條件

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

#### Dataset／Label／模型工件需求

不重建Dataset、不relabel、不重建Continuous Target、不重訓breakout-quality模型、不重建PIT Scores。需要本機既有完整市場資料、Selection PIT score／manifest／audit、正式`roos_base_best.json`與optimizer runtime依賴。首次R3執行會建立全新的R3 studies及R3 Adapted active params；不得沿用原始Score Adapted的完成study。

#### 本機執行

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

#### 原定採用判定

本輪程式接線時預先設定以下判定維度；實際結果與正式結論記錄於後續「本機正式結果」與「對稱比較與判定」：

1. R3 Adapted相較R3 Sort Only是否提高總報酬或Return／MDD。
2. R3 Adapted相較Param Only的ranking效果是否仍為正。
3. 最大回撤是否低於R3 Sort Only的19.20%，Return／MDD是否高於9.59。
4. EV、Realized R、Target mean、三種capture與平均投入資金報酬是否保留R3優勢。
5. Param Only是否顯示R3 Adapted params本身破壞原策略。
6. 改善是否跨年度，而非只來自單一年份。

#### 原定本機執行

R3 CLI已由使用者本機完成；本機正式結果見下節。正式`apps/test_suite.py`仍由使用者本地執行，GPT未執行。

#### 本機正式結果

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

#### 對稱比較與判定

1. **R3 ranking效果再次成立。** 舊ROOS固定時，`R3 Sort Only − Baseline = +1.51pp`總報酬，但MDD增加6.02pp、Return／MDD下降4.27；同一套新Adapted params固定時，`R3 Adapted − Param Only = +23.33pp`總報酬、MDD降低1.30pp、Return／MDD提高2.68。R3在兩套參數下都提高總報酬，且在新參數下連風險調整績效也同步改善，因此ranking方向不是舊參數偶然。
2. **R3 Adapted params拒絕。** 不使用R3時，`Param Only − Baseline = -58.85pp`；使用R3時，`R3 Adapted − R3 Sort Only = -37.03pp`。新參數雖把R3 MDD由19.20%降至12.92%、Return／MDD由9.59提高至11.38，但犧牲37.03pp總報酬，且仍未打敗Baseline的Return／MDD 13.86，不能升格正式參數。
3. 新參數建立的候選母體品質較差：Param Only／R3 Adapted的Target mean為0.7013R／0.9356R，均低於舊ROOS的1.0912R／1.1080R。R3可在新候選池內把Target mean提高0.2343R，但無法恢復舊ROOS候選池品質；這比單純增加trials更值得關注。
4. R3 Adapted的EV與Realized R為0.52R、MDD為12.92%，但平均投入資金報酬只有2.20%，低於Baseline 3.00%與R3 Sort Only 3.74%。新參數改善單筆R與風險路徑，卻沒有形成足夠的資本效率與總報酬。
5. 正式策略與正式ROOS維持Baseline。R3保留`RESEARCH_DIRECTION_PASS / FORMAL_POLICY_NOT_ACCEPTED`；R3 Adapted params標記`REJECTED`。不再增加相同optimizer trials，也不再以同一search space重跑R3 adaptation。

#### 下一個單一研究方向

直接測試先前保留的`R3 ranking × optional entry filters`交互作用，不先做最大回撤逐筆歸因。第一階段採一次粗粒度2×2 gate，固定舊正式ROOS、PIT Scores、R3三分桶、fixed risk、position cap、max positions、ATR entry／stop／trail、exit、rotation、交易成本與帳務：

1. Current filters＋原buy-sort。
2. Current filters＋R3。
3. Optional entry filters全部關閉＋原buy-sort。
4. Optional entry filters全部關閉＋R3。

第一階段只強制關閉primary entry qualification中的`use_breakout_ema_filter`、`use_bb`、`use_vol`、`use_breakout_return_filter`與`use_breakout_false_filter`；不關閉`high_len`事件定義、ATR buy／stop／trail、`use_kc` exit、reclaim re-entry、fixed risk或position cap。正式交互作用比較為`(4−3) − (2−1)`。若全部關閉後R3 ranking效果明顯改善，再逐一拆解五個filter；若沒有改善，停止filter-conflict假設，下一步才考慮重做strategy／capital-aligned Target。此gate不重訓模型、不重建PIT Scores、不執行optimizer。

### 2026-08-04 — Optional Entry Filters × Ranking A～E Gate 本機正式結果

#### 狀態

`RESULT_AVAILABLE / FILTER_CONFLICT_HYPOTHESIS_REJECTED / OPTIONAL_FILTERS_REQUIRED / FORMAL_BASELINE_UNCHANGED`

#### 結果程式基準

- ZIP：`test-branch-1_20260804_005030_da9891c.zip`
- SHA256：`3b0d7bbdec8045c271cb635cee04e615d783e073442f3a5131736c9f59793cd7`
- 比較期間：`2014-01-01～2020-12-31`
- 參數：舊正式`base_finalist_best` rolling active params
- Score source：Selection PIT `selection_point_in_time`
- Future Target：只在portfolio replay完成後離線join，未進入候選、排序、資金配置或成交決策

#### A～E正式結果

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

#### 正式判定

1. **Optional entry filters是必要的候選品質層，不是壓制Score的主要衝突源。** 全關後原排序總報酬由182.62%降至92.73%、MDD由13.18%升至28.17%；R3由184.12%降至104.51%、MDD由19.20%升至28.81%。三種ranking在全關候選池的絕對績效都遠低於目前filters情境。
2. **正向R3 interaction只表示R3對劣化候選池具有較強的相對補救能力，不代表關閉filters有利。** D相較C增加11.78pp，但D仍比B少79.62pp總報酬，且Return／MDD只有3.63。
3. **原始Score在filters全關後沒有恢復，反而更差。** E雖把Target mean提高至1.15R，但平均投入只剩73,216、停損距離擴至9.67%、Realized R降至0.19R、capture降至0.19，總報酬只剩71.20%。這再次確認raw Score偏向高Future Target但低部署／低可實現性候選。
4. **R3資金分桶機制仍有獨立價值。** 在相同filters全關候選池，D相較E增加33.31pp總報酬、0.22R Realized R與0.26 capture；R3以略低Target mean換取更高部署與可實現性。但此價值不足以補救filters全關造成的候選品質崩落。
5. C的capture 0.81不可解讀為策略更佳；其平均Target只有0.58R／Target mean 0.60R，分母較低且平均投入資金報酬只有1.89%。採用判定仍以絕對報酬、MDD、Return／MDD與資本效率為主。
6. **Filter-conflict粗粒度假設拒絕。** 不進入逐一關閉EMA、BB、Volume、Return與False-breakout filters的拆解；正式filters、正式ROOS與正式Baseline均維持不變。R3仍保留`RESEARCH_DIRECTION_PASS / FORMAL_POLICY_NOT_ACCEPTED`。

#### 下一個單一研究方向

轉向`Capital-aligned Target feasibility audit`，不先重訓模型。固定目前optional filters、舊正式ROOS、Selection PIT Scores、R3與portfolio規則，使用既有orderable candidate／Future Target與正式盤前sizing欄位，建立只供離線稽核的：

`capital_opportunity_proxy = no_time_target_R × deployment_rate`

其中`deployment_rate`必須重用正式`proj_cost / sizing_capital / max_position_cap_pct`並截斷於1，不另寫近似sizing。先比較raw No-time Target與capital proxy對下列結果的日內排序能力：實際投入資金報酬、Realized R、`Realized R × deployment_rate`、top-k retention及年度穩定性。此audit不得進runtime、不得使用OOS結果調參、不得建立checkpoint或PIT Scores。

只有capital proxy在多數年度與主要投組貢獻指標上穩定優於現行No-time Target，才建立新的versioned Continuous Target與PIT ranker；若未通過，停止目前continuous-target家族，不以不同乘方、權重或分桶數繼續Selection調參。

#### Dataset／Label／模型工件需求

本次結果回寫不修改程式，不重建Dataset、Label、Continuous Target、PIT Scores或checkpoint。下一個feasibility audit可直接使用既有Selection策略比較候選工件、Future Target與正式sizing payload；只有audit通過後才評估新Target arrays與模型重訓。

### 2026-08-04 — 9A Binary DL Hard Filter 正式 OOS 策略結果

#### 狀態

`RESULT_AVAILABLE / CURRENT_FILTERS_PLUS_DL_REJECTED / DL_REPLACEMENT_GATE_PENDING / FORMAL_BASELINE_UNCHANGED`

#### 程式與模型基準

- 程式ZIP：`test-branch-1_20260804_145148_ef17394.zip`
- SHA256：`00e6b0029b958bd3807868b772d65f0a1cf702277f781602e9e8bd437d15dd44`
- 模型：`breakout_quality_v1 / inception_time_v1 / unique_group_sampling`
- Threshold：固定`0.5`；不得依本次OOS或策略結果回調。
- 模型訓練：Epoch 2為最低Validation Loss `0.708591`；Epoch 3 early stop；完整Selection依2 epochs重訓。
- Runtime score：`forward_oos`；candidate rows `665,528`，model-scored `646,382`，conservative reject `19,146`。

#### 固定條件與唯一差異

- 比較期間：`2021-01-01～2026-03-02`。
- 使用相同`base_finalist_best` rolling active params、原position-aware buy-sort、fixed risk、position cap、max positions、rotation、交易成本、成交與portfolio accounting。
- Baseline固定`use_breakout_quality_filter=False`；Active quality filter固定`use_breakout_quality_filter=True`。
- 現有EMA、BB、Volume、breakout return及false-breakout optional entry filters在兩組均依當期active params維持原設定。
- 不使用continuous Score ranking、R2、R3、Future Target或optimizer。

#### 模型 OOS 結果

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

#### 正式策略結果：A目前filters＋無DL vs B目前filters＋Binary DL

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

#### 判定

1. **9A在事件Label上具有OOS分類訊號，但目前filters下作第二層hard filter正式失敗。** Precision提升不能補償Recall、候選供給、曝險、勝率、Payoff與EV下降。
2. 模型PASS由Selection 64.02%降至OOS 45.43%，而策略每日可掛單候選由50.55降至20.98；quality gate縮減候選池後造成更深的持股缺口與較低資金部署。
3. 交易數只減少11筆，但候選池與曝險大幅下降，表示主要傷害包含portfolio path displacement與替代交易品質下降，不是單純少交易。
4. **不得依本次OOS調threshold、epochs、learning rate、Label或模型。** Current-filters-plus-DL路徑標記`REJECTED`，正式runtime仍關閉Binary DL filter。
5. 此結果只完成A／B，不能用來否定使用者提出的「移除optional filters、Binary DL作唯一品質Gate」。真正replacement假設仍必須看F相較A的絕對績效。

#### 下一步：執行 A／B／C／F Replacement Gate

直接使用既有model、manifest與forward-OOS scores，不需重新訓練或重新export：

```bash
python apps/breakout_quality.py strategy-dl-filter-gate --dataset full --param-policy base-finalist-best --max-positions 10 --rotation off
```

正式判讀順序：

1. `F−C`：Binary DL在optional filters全關後是否能單獨改善完全不篩選的突破候選池。
2. `F−A`：DL-only replacement能否至少打平目前正式rule-based filters策略；這是採用主判定。
3. `(F−C)−(B−A)`只用來判斷DL作替代者是否比疊加者更有效；正值不代表F絕對績效通過。
4. 若F仍顯著低於A，停止目前9A Binary DL Filter正式整合，不以同一OOS調threshold救援；下一個模型實驗必須提出本質不同的Label或representation機制。

#### Dataset／Label／模型工件需求

本次結果回寫不修改程式、Dataset、Label、checkpoint或Scores。A／B／C／F Gate可直接使用目前完整工件執行；不需重訓、relabel、optimizer或PIT重建。

### 2026-08-04 — Binary DL Rule Ablation A0／B0 ～ A4／B4 Matrix

#### 狀態

`IMPLEMENTED / LOCAL_FULL_DATA_EXECUTION_REQUIRED / FORMAL_BASELINE_UNCHANGED`

#### 程式基準

- 輸入ZIP：`test-branch-1_20260804_182739_d9ddf41(2).zip`
- SHA256：`bc1f9aa5409141ea527342ebe422fa36c222e5d3014e488e85b1337bb8353f01`
- 本輪patch ZIP名稱與SHA256以交付回覆為準。

#### 前一輪 A／B／C／F 結果

固定期間`2021-01-01～2026-03-02`、`base-finalist-best` rolling active params、9A `inception_time_v1 / unique_group_sampling / threshold 0.5`、canonical runtime score及原position-aware buy-sort：

| 組別 | Optional filters | Binary DL | 總報酬 | MDD | RoMD | EV | 平均曝險 |
|---|---|---|---:|---:|---:|---:|---:|
| A | 目前 | 關 | 129.08% | 17.41% | 7.42 | 0.53R | 84.02% |
| B | 目前 | 開 | 85.79% | 26.34% | 3.26 | 0.36R | 68.22% |
| C | 全關 | 關 | 138.58% | 13.35% | 10.38 | 0.47R | 92.02% |
| F | 全關 | 開 | 145.92% | 18.31% | 7.97 | 0.35R | 79.04% |

判定：B明確拒絕；C為本輪風險調整後最佳研究情境；F相較C雖總報酬+7.34pp，但MDD+4.95pp、RoMD−2.41、EV−0.11R、直接交易選擇差異−21.83R，且年度改善主要集中2023，因此現有9A DL-only gate不採用。A仍為正式基準；C只保留研究候選，不直接升級正式策略。

#### 唯一主要變更

將原A／B／C／F重新命名並擴充為單一五層矩陣：

| 層級 | A組 | B組 | Optional filters | 歷史門檻 | Re-entry | KC出場 |
|---:|---|---|---|---|---|---|
| 0 | A0：DL關 | B0：DL開 | 依原active params | 依原active params | 依原active params | 依原active params |
| 1 | A1：DL關 | B1：DL開 | 五項全關 | 依原active params | 依原active params | 依原active params |
| 2 | A2：DL關 | B2：DL開 | 五項全關 | 關 | 依原active params | 依原active params |
| 3 | A3：DL關 | B3：DL開 | 五項全關 | 關 | 關 | 依原active params |
| 4 | A4：DL關 | B4：DL開 | 五項全關 | 關 | 關 | 關 |

使用者原訊息最後一列寫為「A3 B4」，依層級與成對比較契約正規化為`A4／B4`。A4／B4只關閉KC，不關閉半倉停利、ATR初始停損、trailing、fixed-risk sizing、position cap、成交／費稅或原buy-sort。

#### 架構與報表

- 五層均重用`strategy_compare.run_comparison`與正式portfolio engine／trade attribution，不複製成交、帳務或統計。
- `strategy_compare.run_comparison`新增內部`shared_param_overrides`，同時套用至同層A與B；同層仍只允許`use_breakout_quality_filter=False/True`一項差異。該參數不暴露為一般CLI，避免使用者任意建立未登錄組合。
- 合併報表不再列大量跨組pair；固定輸出：情境矩陣、每層A／B與`Bn−An`投組增量、交易品質／資金使用、A系列相對A0的規則消融、五層trade attribution及年度`Bn−An`矩陣。
- 新輸出目錄：`strategy_dl_filter_rule_ablation_gate_<param_policy>_canonical_runtime/`，內含五個pair目錄及單一Markdown／JSON摘要。

#### 固定條件

- 不重建Dataset、不relabel、不重訓模型、不重新選epoch、不調threshold、不執行optimizer。
- 模型、runtime scores、active-param來源、比較期間、benchmark、max positions、rotation、fixed risk、position cap、買入排序、交易成本及portfolio accounting全部固定。
- 不使用continuous Score、R2、R3、Future Target或Selection PIT ranking。

#### 採用判讀

1. 每層只以同層`Bn−An`判斷DL增量，避免將規則消融效果誤認為DL效果。
2. A系列`A1−A0`至`A4−A0`只判斷逐步移除規則對基礎策略的影響。
3. DL有效至少需同時改善Return／MDD、EV、直接交易選擇R及年度穩定性；總報酬單獨上升不足以通過。
4. 若只有關閉規則的A系列改善，而各層`Bn−An`持續惡化，結論應是規則本身需簡化，而不是9A DL有效。
5. 真實結果取得前，本實驗只標記`IMPLEMENTED`，不得預判A2／A3／A4或B2／B3／B4有效。

#### Dataset／Label／模型工件需求

無需重建任何Dataset、Label、checkpoint或Score；直接沿用目前9A canonical forward-OOS工件與rolling active-param JSON執行CLI即可。

### 2026-08-05 — 4×2 Binary PIT Process-worker Consistent Final Result

#### 狀態

`RESULT_VALID / PROCESS_WORKER_BINARY_PIT_CONFIRMED / CURRENT_9A_BINARY_DL_REJECTED / A2_NO_DL_RESEARCH_CANDIDATE / FORMAL_BASELINE_UNCHANGED`

#### 程式與結果基準

- 使用者結果ZIP：`test-branch-1_20260805_175702_4d89c30.zip`
- SHA256：`d2089bb2e4cda0ddd539d2876d48db984e90f9154abc58c9061fe620945b19ce`
- 使用者依process-worker source修正版重新執行`strategy-dl-filter-param-adapt-gate`。
- 報表明示`DL replay source=binary_point_in_time（八操作點一致；process workers已傳遞）`。
- A0～A3與前次完全一致；B0～B3相較前次canonical誤用結果明顯改變，確認Binary PIT source已實際進入建立訊號的spawned workers，而不只是metadata顯示改變。
- 比較期間：`2021-01-01～2026-03-02`；Binary threshold固定`0.5`；原position-aware buy-sort、max positions 10、rotation off、資金帳務與交易成本固定。

#### 八操作點主要結果

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

#### Binary DL增量

| 比較 | Δ報酬 | ΔMDD | ΔRoMD | ΔEV | Δ曝險 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| B0−A0 | +14.65pp | +8.55pp | -1.88 | -0.09R | -17.50pp | 拒絕；總報酬增加不足以覆蓋MDD、RoMD、EV與年度穩定性惡化 |
| B1−A1 | +46.23pp | +5.86pp | -0.75 | +0.07R | -16.01pp | 不採用；報酬與EV改善，但RoMD下降、MDD提高，且收益高度集中2023、2022為-12.16% |
| B2−A2 | -16.44pp | +11.63pp | -5.26 | -0.08R | -14.16pp | 明確拒絕 |
| B3−A3 | -20.00pp | +6.44pp | -4.66 | -0.57R | -15.91pp | 明確拒絕 |

最終公平比較`B3−A2`為：報酬`-66.91pp`、MDD`+2.86pp`、RoMD`-5.36`、EV`-0.14R`、曝險`-15.98pp`。interaction=`(B3−A3)−(B2−A2)=-3.56pp`，表示在DL開啟環境重訓風險參數沒有形成正協同，反而使DL增量略為更差。

#### 採用判定

1. 現有9A Binary DL模型、目前Label及threshold 0.5在PIT-consistent 4×2下正式拒絕；不得promote至正式runtime。
2. B1雖為八組最高總報酬，但未通過既定採用契約：MDD及RoMD惡化、年度不穩定，且目前使用者貼出的總表未包含直接交易選擇R，不能以單一總報酬宣稱DL有效。
3. A2是目前最有價值的無DL研究候選：相較A1報酬`+28.60pp`、RoMD`+0.48`、EV`+0.27R`，代價是MDD`+2.06pp`；仍屬研究候選，不直接取代正式A0。
4. A3雖有最低MDD與最高EV，但總報酬及RoMD均低於A2；不作首選。
5. 正式策略維持A0與原ROOS，Binary DL runtime維持關閉。

#### 後續順序

1. 無DL策略線：先對A2／P2做參數選擇穩定性、fold／年度依賴與base-best／agree／ensemble一致性驗證，再決定是否進入正式promotion程序。
2. DL研究線：停止對目前9A Label繼續調threshold或重訓風險參數；下一個模型改測chronological first-touch Binary Label。
3. 新Label先固定使用A2／P2參數做`DL關 vs DL開`Gate；只有同時改善RoMD、EV、直接交易選擇R及年度穩定性，才投入Binary PIT與DL-on optimizer。不得一開始就重做P3型昂貴參數適應。
4. 若要解釋B1的高總報酬來源，應讀取其`trade_attribution.json`／`trade_attribution.md`；該歸因未包含在使用者本次貼出的console，因此本節不推測獨有交易選擇R。

### 2026-08-05 — A2 Trade-path Label Model Prediction Result

#### 狀態

`RESULT_VALID / MODEL_OOS_WEAK_AND_UNSTABLE / STRATEGY_GATE_PENDING / FORMAL_BASELINE_UNCHANGED`

#### 程式與結果基準

- 使用者結果ZIP：`test-branch-1_20260805_230725_ff04a05.zip`
- SHA256：`8a616fed6646360219c7acba214fe5753a9d043cbf09a9a8be3317e9713d22a0`
- 使用者依trade-path single-stock parity contract v2重新建立Label、重新訓練`inception_time_v1 / unique_group_sampling`，並完成Selection／OOS Prediction報表。
- Dataset、Label、模型、threshold及split均由本次console結果確認；策略Gate尚未執行。

#### P2 historical teacher穩定性資訊

建立2014～2020 historical teacher schedule時，7-fold rolling optimizer同時輸出三種參數選擇：

| Policy | Fold OOS AVG RoMD | 0050 | OOS_CHAIN RoMD | 0050 |
|---|---:|---:|---:|---:|
| base-finalist-best | 1.78 | 1.78 | 5.01 | 5.60 |
| base-finalists-agree | 2.04 | 1.78 | 6.84 | 5.60 |
| seed ensemble | 1.78 | 1.78 | 5.01 | 5.60 |

`base-finalists-agree`在本段歷史teacher OOS_CHAIN較佳；但本次TP1 Label實驗的teacher identity已預先固定為`base-finalist-best`，不得在看到模型OOS後改換teacher並混稱同一實驗。這項結果只作A2／P2參數穩定性證據，不改變本次Label工件。

#### Dataset與訓練結果

- Events：`1,793,028`
- PASS：`7,573`
- REJECT：`9,952`
- EXCLUDED：`1,775,503`
- Binary eligible：`17,525`；整體PASS率約`43.21%`。
- Inner Train：`5,847` groups；Validation：`1,730` groups；Final Selection Refit：`7,628` groups；固定OOS：`9,263` groups。
- Inner Validation最佳Epoch：`2`；最低Validation Loss：`0.648880`；完整Selection依selected epochs重訓2 epochs，Final Loss：`0.633495`。

#### Selection／OOS主要結果

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

#### 採用判定

1. 模型層只取得微弱OOS Precision增益，且Recall、PR-AUC、校準與年度穩定性不足；不得直接promote。
2. 仍依預先定義流程執行一次策略層Gate，確認硬篩選後是否意外形成RoMD、EV、直接交易選擇R與年度穩定性的共同改善。
3. 若`New−Base`未同時通過上述四項，停止此trade-path Label模型線，不調OOS threshold，也不投入Binary PIT與DL-on optimizer。
4. 正式策略維持A0；A2仍為無DL研究候選；現有9A Binary DL runtime仍關閉。

### 2026-08-06 — TP1 C1～C6正式績效結果與A9 runtime比較擴充

#### 狀態

`C1_C6_ACCEPTED / TP1_HARD_FILTER_REJECTED / MIN_DL_TP1_NOT_PROMOTED / A9_RUNTIME_COMPARISON_IMPLEMENTED / A9_RESULT_PENDING`

#### 程式與結果基準

- 使用者ZIP：`test-branch-1_20260806_202837_283d61c(1).zip`
- ZIP SHA256：`4ca1949169372cdab3bd3a55d0ec0813cd3406651ae4b399e85b67ba065944a9`
- 正式比較期間：`2021-01-01～2026-03-02`
- Dataset：`full`
- Param policy：`base-finalist-best`
- Max positions：10
- Rotation：off
- 固定條件：同一參數組內只切hard filter；原position-aware buy-sort、費稅、帳務與active-param無前視契約不變。

#### C1～C6主要結果

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

#### A9加入比較的唯一變更

- `config/strategy_compare.py`新增第二個DL source：`A9 = breakout_quality_v1 / inception_time_v1 / unique_group_sampling / threshold 0.5`。
- 新增三個runtime arms：`C7 Full ROOS × DL-A9`、`C8 Min ROOS × DL-A9`、`C9 Min-DL-TP1 ROOS × DL-A9`。
- 新增A9相對同參數DL-off contrasts：`C7−C1`、`C8−C3`、`C9−C5`；以及TP1相對A9 contrasts：`C2−C7`、`C4−C8`、`C6−C9`。
- 比較引擎由「每組只能一個DL-on」泛化為「一個共用DL-off基準可掛多個DL-on模型」；每個DL-on仍獨立執行controlled pair，重複基準的摘要與年度報酬必須完全一致，否則fail-fast。
- A9只作runtime hard-filter比較；本輪不建立A9-aware Min-DL參數、不覆蓋TP1 P3、不重新訓練A9或TP1模型。

#### Dataset／Label／模型影響

不重建Dataset、不重新Label、不重新訓練TP1或A9模型、不改threshold、不改P2／P3 ROOS、交易規則或帳務。若A9正式forward-OOS scores缺少或過期，`apps/strategy_compare.py`依config使用既有A9 checkpoint自動補匯出。

#### 下一步

由`python apps/strategy_compare.py`選擇`[2] 查看設定、工件與預計動作`，確認A9 model／manifest／forward scores為READY或PREPARABLE；再選`[1/Enter] 執行目前比較設定`。取得C7～C9後，以同參數A9效果及TP1−A9 contrasts判定TP1問題是Label／模型特有，或Binary hard-filter共同結構問題。

### 2026-08-07 — C1～C10正式結果：TP1／A9 hard filter與matched Min-DL參數適應結案

#### 狀態

`C1_C10_ACCEPTED / TP1_HARD_FILTER_REJECTED / A9_HARD_FILTER_REJECTED / MIN_TP1_NOT_PROMOTED / MIN_A9_NOT_PROMOTED / MIN_ROOS_REMAINS_RESEARCH_BASELINE / NEXT_ROBUSTNESS_AND_CAPACITY_PRESERVING_A9`

#### 程式與結果基準

- 使用者ZIP：`test-branch-1_20260807_140716_a6358d3.zip`
- ZIP SHA256：`de9916b323d141422a9a695a7fc2fcdc46853317d692fc2cfdd5fd2ad26e2fa0`
- 正式比較期間：`2021-01-01～2026-03-02`
- Dataset：`full`
- Param policy：`base-finalist-best`
- Max positions：10
- Rotation：off
- Config fingerprint：`62b4dfabe42a`
- 固定條件：同一參數組內只切指定DL runtime；原position-aware buy-sort、費稅、帳務、active-param無前視與盤前資金鎖定契約不變。

#### Dataset／Label／模型／參數重建需求

- Dataset：不重建。
- Label：不重建；A9沿用既有MFE／MAE Label，TP1沿用既有realized trade-path Label。
- A9／TP1模型權重與threshold：不重訓、不調整；固定既有forward-OOS scores與threshold `0.5`。
- Min ROOS、Min-TP1 ROOS、Min-A9 ROOS：重用本次正式比較已建立且identity／coverage通過的rolling active-param工件；本節只記錄實際forward-OOS策略結果與採用判定。

#### C1～C10主要結果

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

#### Controlled判定

- TP1 hard filter正式否決：`C2−C1=-74.03pp / -2.38 RoMD / -79.90R`、`C4−C3=-77.84pp / -5.46 / -89.76R`、`C6−C5=-107.69pp / -7.93 / -280.31R`。三套參數下直接選擇R皆大幅為負，不再做TP1 threshold、TP1-aware optimizer或TP1 capacity-preserving延伸。
- A9 hard filter同樣不得部署：`C7−C1=-43.29pp / -4.16 RoMD / -76.47R`；`C8−C3=-23.99pp / -4.26 / -6.12R`；`C10−C9=-41.10pp / -6.47 / -166.38R`。matched A9-aware參數未能修復hard-filter結構，C10相對C8報酬再差`64.03pp`、RoMD再差`2.91`，因此不再做A9-aware risk optimizer。
- `C3 = Min ROOS`仍是全期最高報酬與最高RoMD arm，且相對C1為`+37.61pp報酬 / -2.00pp MDD / +3.40 RoMD`；但年度優勢高度集中於2023：C3在2023為`80.22%`，其餘年度為`23.08% / 0.84% / 11.21% / 2.25% / 4.85%`。作為只讀診斷，排除2023後逐年報酬鏈結約為C3 `47.98%`、C1 `70.71%`、C9 `81.53%`，因此C3尚不得直接晉升正式策略，下一步先做selector／fold／年度集中度穩定性Gate。
- C9雖總報酬低於C3，但MDD `11.83%`、RoMD `10.12`且六個年度皆為正；這只視為參數穩定性線索，不視為Min-A9部署證據，因其參數是在A9-on訓練環境下選得而runtime為DL-off。
- A9唯一保留的研究理由位於Min ROOS：C8相對C3的同參數直接選擇R只有`-6.12R`，但曝險下降`13.90pp`且總報酬下降`23.99pp`，顯示主要損失可能來自hard filter刪除候選後的資金／持股路徑，而非強烈反向的單筆選擇。後續若繼續DL，只測A9的capacity-preserving使用方式；不再測TP1。

#### 報表語意修正

本次正式報表暴露一個attribution顯示問題：`direct_selection_delta_r`只在相同`param_source`與`rule_policy`的DL-off基準／DL-on arms之間有共同物理基準。舊報表對`C6-C4`、`C10-C8`、`C6-C1`、`C10-C1`等跨參數contrast仍直接相減兩個不同attribution，數值雖可算但沒有controlled direct-selection意義。正式報表已改為只對同參數／同rule-policy contrasts顯示`Δ同參數DL選擇R`；跨參數contrast固定顯示`-`。原C1～C10策略績效本身不受此修正影響。

#### 下一步順序

1. 先做`Min ROOS` promotion robustness gate：重用既有rolling optimizer工件，比較正式selector／seed ensemble／fold與年度集中度，不重新調參；若優勢只由2023或單一selector造成，維持C1正式基準。
2. 只有A9保留DL研究：以C3為固定基準，實作capacity-preserving A9。A9不得再作eligibility hard reject；只在盤前候選數超過可掛單容量時影響候選優先序，必須保持正式slot／資金／盤前掛單與盤中不可換股契約，不得以OOS回調threshold。
3. 若capacity-preserving A9的同參數直接選擇效果仍不為正，Binary DL策略線停止；若轉正但總績效仍差，再做exact-accounting資金路徑歸因，不直接進新Label／新架構。

### 2026-08-07 — C11 Resource-aware正式結果與C12 best-improvement basket實作

#### 狀態

`C11_ACCEPTED_AS_DL_RUNTIME_RESEARCH_BASELINE / LOW_EXPOSURE_PROBLEM_RESOLVED / A9_SELECTION_SIGNAL_POSITIVE / C12_IMPLEMENTED_RESULT_PENDING`

#### C11正式結果基準

- 使用者正式輸出期間：`2021-01-01～2026-03-02`
- Dataset：`full`
- Param policy：`base-finalist-best`
- Max positions：10
- Rotation：off
- Config fingerprint：`bdf346da83e7`
- 固定模型：A9 `breakout_quality_v1 / inception_time_v1 / unique_group_sampling`；不重訓、不調threshold。
- 固定策略參數：`C3 = Min ROOS`，同一P2 rolling active params；C11只改盤前resource-aware runtime。

#### C11主要結果

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

#### C12唯一變更

C11在`dl-selection` mode內依Min ROOS原順位掃描PASS候選，遇到第一個能提高PASS reserved capital的promotion就立即接受，因此結果可能受first-improvement路徑影響。C12新增`resource-aware-binary-basket` runtime，但保持以下全部不變：

- Min ROOS cash-binding Gate不變；slot先binding時DL完全不介入。
- A9模型、threshold、score日期語意、normal／continuation／Re-entry lifecycle不變。
- exact accounting、cash-capped sizing、盤前鎖定、max positions、rotation與Min ROOS初始buy-sort不變。
- 不新增距限價Threshold、利用率百分比、DL權重、Future Target、fill prediction或holding prediction。

C12只把`dl-selection`內的promotion選擇由first-improvement改為best-improvement：每一輪對所有尚未promotion的A9 PASS候選各自做完整exact cash-cap trial，只保留仍為cash-binding且使`PASS reserved capital`增加，或在PASS reserved相同時使PASS數增加的trial；當輪選擇PASS reserved最高、其次PASS數較多、再依Min ROOS順位決勝的trial，接受後重新評估下一輪，直到沒有改善。

精確窮舉所有PASS子集合雖可作理論global search，但候選數與free slots增加時呈指數複雜度，與專案「架構調整不得明顯犧牲效率」原則衝突，因此不納入正式runtime。C12是零新增Threshold、每輪全候選best-improvement的可執行改進，不宣稱數學全域最優。

#### 新比較設定

只啟用：

| Arm | Runtime | 目的 |
|---|---|---|
| C3 | Min ROOS | 同參數資金利用基準 |
| C11 | A9 resource-aware first-improvement | 已證明有效的新DL runtime基準 |
| C12 | A9 resource-aware best-improvement basket | 驗證去除first-candidate順序偏誤是否仍有額外價值 |

啟用contrasts固定為`C11-C3`、`C12-C3`、`C12-C11`；C8 hard-filter停用，不需重跑已結案失敗對照。

#### Dataset／Label／模型重建需求

Dataset不重建、Label不重建、A9不重訓、threshold不調整、forward-OOS scores重用、Min ROOS P2重用。C12只是策略runtime排序變更。

#### 採用判定與下一步

C11結果已接受為新的DL runtime研究基準；C12尚未取得正式forward-OOS結果，只標記`IMPLEMENTED`。下一步由`python apps/strategy_compare.py`進入選單，先`[2] 查看設定、工件與預計動作`確認只含C3／C11／C12，再`[1/Enter] 執行目前比較設定`。C12只有在相對C11維持接近相同曝險、且RoMD／EV／同參數DL選擇R至少一項有實質增量且年度穩定性未明顯惡化時才保留；否則維持較簡單的C11。

### 2026-08-07 — C12正式結果、最大化PASS研究方向與正式PASS Quality Audit框架

#### 狀態

`C12_RESULT_AVAILABLE / MAX_PASS_USAGE_DIRECTION_RETAINED / PASS_QUALITY_AUDIT_IMPLEMENTED_RESULT_PENDING`

#### C12正式結果

正式期間`2021-01-01～2026-03-02`、Dataset=`full`、Param policy=`base-finalist-best`、Max positions=10、Rotation=off；A9模型、threshold、Min ROOS P2與forward-OOS scores均不變。

| Arm | 報酬 | MDD | RoMD | 年化 | EV | Payoff | 曝險 | 交易 | 同參數DL選擇R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 Min ROOS | 166.69% | 15.41% | 10.82 | 20.95% | 0.73R | 3.62 | 92.13% | 336 | 0.00R |
| C11 A9 resource-aware first-improvement | 184.69% | 15.67% | 11.79 | 22.49% | 0.87R | 3.26 | 92.26% | 373 | +79.95R |
| C12 A9 resource-aware best-improvement | 172.59% | 16.47% | 10.48 | 21.46% | 0.88R | 3.25 | 92.49% | 388 | +94.59R |

`C12−C11 = -12.11pp報酬 / +0.80pp MDD / -1.31 RoMD / -1.03pp年化 / 約0.00R EV / +0.22pp曝險 / +15交易 / +14.64R同參數DL選擇R`。

Resource-aware盤前診斷：C11為`DL選股131日 / 資金利用優先563日 / 實際改單92日 / 新增PASS 111 / PASS預留資金+16,324,574 / 總預留-44,794`；C12為`146 / 601 / 119 / 143 / +22,169,986 / +493,750`。C12確實提高PASS使用與同參數DL選擇R，但策略報酬／RoMD未同步改善。

#### 使用者研究決策

不把C12較差的策略績效解讀成「應降低PASS使用」；後續固定研究原則為：**在既有資源契約下最大化PASS使用，改善PASS本身的品質**。不得新增Min ROOS sorting與PASS sorting的混合權重、比例或依OOS結果調整的折衷參數，避免多出一層selector比例最佳化問題。

#### Candidate validity／DL quality／allocation單一責任契約

1. Strategy唯一擁有candidate validity：建立、normal／continuation／Re-entry、expiry／invalidation全部沿用原策略SSOT。
2. DL只擁有quality：PASS／REJECT描述當下或既定snapshot的品質，不得使仍屬策略VALID的candidate失效。
3. Portfolio selector只擁有allocation：只在策略已判VALID的candidate pool內依盤前cash／slots／正式cash-capped sizing分配資源。
4. 因此未來若做candidate-day DL，語意只能是每日更新quality；不得建立第二套DL candidate expiry。Future Label／MFE／MAE／Realized R只能事後Audit，不得回流當日runtime。

#### 正式Audit framework

本輪以使用者ZIP`test-branch-1_20260807_194217_8fbe28b(1).zip`為基準，SHA256=`d47b5209b477b3c739479c5a0851ea9a81d3d70bf7b9241016a52df4de32c34a`，實作config-driven正式Audit框架：

- 新增`config/audit.py`，各模組Audit對象、source、dimensions、outcomes與output policy集中管理；正式App不得硬編碼C11／C12等研究名稱。
- `apps/breakout_quality.py`正式主選單新增`Audit／診斷`，子選單固定為執行目前設定、查看設定／工件／預計動作、查看最近結果。
- 第一個正式Audit為`a9_pass_quality`，預設只讀最新strategy-compare的C12 arm，分析A9 PASS內部的raw score、candidate age、candidate type、既有Label／MFE／MAE與實際selected trades Realized R。
- Audit只讀既有正式工件；缺件顯示BLOCKED，不重跑策略、不建立Label、不train模型、不改runtime。
- Score／age分組以config指定quantile groups產生，只供read-only診斷；不得直接變成runtime threshold、age cutoff、Min ROOS／DL混合比例或其他OOS調參。
- 輸出固定於`outputs/audit/<module>/<audit>/runs/<timestamp>/`並維護`latest/`，包含Markdown、JSON、PASS candidate明細與各分組CSV。

#### Dataset／Label／模型重建需求

- Dataset：不重建。
- Label：不重建。
- A9模型／threshold：不修改、不重訓。
- Strategy replay：Audit不重跑；只讀既有正式strategy-compare工件。
- 本節Audit僅為`IMPLEMENTED`，尚未取得正式真實資料Audit結果，不得預判score／age／candidate type哪一項是PASS品質主因。

#### 下一步

由`python apps/breakout_quality.py`進入`[2] Audit／診斷`，先`[2] 查看 Audit 設定、工件與預計動作`，確認`a9_pass_quality`來源READY，再`[1/Enter] 執行目前 Audit 設定`。依正式Audit結果判斷PASS品質改善方向；在結果前不新增candidate-day失效規則、不新增score threshold或排序混合權重。

### 2026-08-07 — A9 PASS Quality正式Audit結果：Score整體單調性弱、Age cutoff不成立、下一步檢查PASS persistence amplification

#### 狀態

`RESULT_AVAILABLE / SCORE_GLOBAL_RANK_NOT_SUPPORTED / AGE_CUTOFF_NOT_SUPPORTED / PASS_PERSISTENCE_AUDIT_NEXT`

#### 結果來源與固定條件

- 使用者以正式`[2] Audit／診斷`執行`a9_pass_quality`；來源arm=`C12`、DL=`A9`、runtime=`resource-aware-binary-basket`、threshold=`0.5`。
- Audit只讀既有strategy-compare、A9 PASS candidates、Dataset event Label與已發生selected trades；沒有重播策略、建立Label、重訓模型、改threshold或改candidate lifecycle。
- Strategy validity／DL quality／Selector allocation三層契約維持不變；本節所有Label／MFE／MAE／Realized R只供事後研究診斷。

#### PASS總覽

- Orderable candidates：`102,859`
- A9 PASS candidate-days：`51,885`，PASS share=`50.44%`
- Selected PASS：`223`；Selected PASS mean Realized R=`0.89R`
- 原Event Label coverage=`98.17%`；candidate-day weighted原Event Label PASS rate=`51.73%`

#### Score診斷

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

#### Candidate age／type診斷

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

#### 目前最重要的新假設

A9是在原signal event評分一次，但同一策略VALID event可在後續多個candidate days重複出現。Audit顯示extended佔PASS candidate-days約91.1%，而candidate age越高時原Event Label PASS composition越低。下一步先驗證是否存在**PASS persistence amplification**：A9 false-positive PASS events是否平均比true-positive PASS events存活更久，因而在可掛單pool中被重複放大。這是candidate-pool weighting問題，不是另一套candidate invalidation規則。

下一個只讀Audit應比較：

1. unique PASS event-level Label precision vs candidate-day weighted Label precision；
2. 原Event Label PASS／REJECT各自的candidate-days per event、max candidate age、extended-day數；
3. false-positive events對全部PASS candidate-days與C12 selected PASS的占比放大倍率；
4. 不使用Future Target改runtime，不建立age threshold；只有確認persistence amplification後，才設計「DL quality每日更新或加入合法current-state資訊」的模型實驗。

#### Audit讀檔警告修正

正式執行出現兩個`pandas DtypeWarning`，分別來自orderable candidates中本Audit不使用的mixed-type欄位，以及Dataset events的ticker型別推斷。本輪只把Audit CSV讀取改為讀取所需欄位、ticker／candidate_type明確string dtype與`low_memory=False`；不改任何Audit數值口徑、策略、Dataset、Label、模型或runtime。

### 2026-08-07 — A9 PASS Persistence正式Audit結果：false-positive persistence amplification成立；下一步改為C13策略使用語意

#### 狀態

`RESULT_AVAILABLE / FALSE_PASS_PERSISTENCE_AMPLIFICATION_CONFIRMED / SELECTOR_NOT_PRIMARY_CAUSE / SR-C13_CANDIDATE_DAY_RESCORE_PLANNED`

#### 結果來源與固定條件

- 使用者以正式`[2] Audit／診斷`執行`a9_pass_persistence`；來源arm=`SR-C12`、DL=`DL-A9`、runtime=`resource-aware-binary-basket`、threshold=`0.5`。
- Event identity=`ticker / signal_date / high_len`；Audit只讀A9已判PASS且策略仍屬VALID／orderable的candidate-days、原Event Label與selected trades。
- Strategy validity／DL quality／Selector allocation三層契約不變；本Audit不建立expiry、不改candidate lifecycle、不改C12、不重播策略、不重訓模型。

#### 正式結果

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

#### 判讀

1. Persistence amplification正式成立：false PASS比true PASS平均多存活`5.6` candidate-days，即約`+60%`；false-positive share由unique-event層`34.73%`放大到candidate-day層`46.04%`，使Label precision由`65.27%`稀釋至`53.96%`，下降`11.31pp`。
2. C12不是主要放大來源：selected false-positive share=`43.98%`低於candidate-day pool的`46.04%`，selected precision=`56.02%`亦比candidate-day precision高`2.06pp`。因此Resource-aware basket selector略為抑制而非進一步放大false positives；主要結構問題位於`DL-A9` breakout-event單次quality被後續candidate-days重複沿用。
3. false PASS被選後的mean Realized R=`0.24R`，顯著低於true PASS selected的`1.43R`，差`1.19R`；因此false-positive persistence不只是Label組成差異，亦與實際selected經濟品質一致。
4. 本結果不得解讀為candidate age本身就是失效條件；candidate validity仍完全由原策略SSOT決定。Persistence只證明「一次性event quality在daily candidate pool的權重會失真」。不建立age cutoff或第二套DL expiry。

#### 編號／研究層級更正

- 先前草案將下一步稱為`A10 Candidate-State Quality`是錯誤分類，現已取消。
- `MR-10A`早已由2026-07-29的Candidate-conditioned Query使用並永久占用；不得重用或以`A10`作模糊別名。
- 本次要測的是**既有DL-A9在策略中的score refresh timing**，不是模型訓練／architecture／Label升級，因此應進入策略runtime namespace，正式規劃為`SR-C13`。
- `MR-9A`（模型研究實驗）與`DL-A9`（runtime DL source）為不同namespace；完整定義以`doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md`為準。

#### 下一個策略runtime實驗：SR-C13 A9 Candidate-day Re-score

`SR-C13`保持`SR-C12`「在既有資源契約下最大化PASS使用」的allocation原則，唯一研究變更是：**同一個策略VALID candidate在每個decision day重新用既有DL-A9模型與最新合法盤前snapshot計算quality，而不是整段lifecycle永久沿用原breakout日score。**

固定設計原則：

1. **模型不升級**：沿用同一`DL-A9 = MR-9A / ARCH-inception_time_v1 / PROFILE-unique_group_sampling`checkpoint、權重與threshold `0.5`；不重新訓練、不新建model experiment ID。
2. **策略validity不變**：normal／continuation／Re-entry／expiry仍由原策略唯一決定；每日A9 REJECT只代表當日quality，不得刪除仍屬策略VALID的candidate。
3. **盤前無前視**：decision day D 的candidate-day re-score只可使用D盤前合法可知資料；若特徵定義依收盤bar，最晚為D-1 close。
4. **allocation不變**：沿用`SR-C12` resource-aware basket／最大化PASS資源契約；唯一差異是PASS／REJECT使用「當日重新計算的DL-A9 quality」而非原event靜態quality。
5. **不新增混合比例**：不加入Min ROOS／DL weight、不加入age cutoff、不調A9 threshold、不依OOS建立score bucket。
6. **這是runtime distribution-shift實驗**：A9原本由breakout-event snapshot訓練，直接套到candidate-day state可能失效；是否能降低persistence amplification正是SR-C13要驗證的策略使用問題。若SR-C13顯示既有A9無法泛化到candidate-day snapshot，之後才另立真正的model-training experiment，並依Registry重新分配新的`MR-*` ID，不能預先占用或重用`MR-10A`。

#### Dataset／Label／模型重建需求

- Dataset／Label：本階段不重建、不relabel；SR-C13不進行模型訓練。
- A9模型／threshold：不重訓、不調整。
- Candidate-day inference artifact：需要新增或擴充正式推論工件，使每個策略VALID candidate decision day可取得合法snapshot下的DL-A9 score；它是推論／runtime工件，不是新Dataset Label或新model version。
- Strategy compare：待SR-C13實作後，以`SR-C12`為主要controlled comparator，另保留`SR-C3`作絕對Min ROOS基準；不得把SR-C11／SR-C12混合比例變成新參數。

#### 採用判定與下一步

下一步先實作`SR-C13` candidate-day re-score runtime與其必要推論工件，不修改A9 training pipeline。核心問題是：在固定C12最大化PASS資源契約下，**只改score refresh timing**是否能降低candidate-day false-positive persistence amplification，並把更多PASS使用轉化為更好的selection R／EV／RoMD，而不重新出現資金利用率下降。若SR-C13失敗，再根據結果判斷是否需要真正的candidate-state模型訓練；該模型屆時另行取得新的`MR-*` ID。

### 2026-08-07 — AUD-a9-selection-confidence正式結果：A9 confidence不適合作為DL Selection Mode主排序

#### 狀態

`RESULT_AVAILABLE / WEAK_RANKING_SIGNAL / CONFIDENCE_PRIORITY_NOT_ADOPTED`

#### 正式結果

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

#### 判讀與採用

1. A9 confidence在真正多PASS競爭日的整體排序力弱；尤其每日平均Event Label concordance=`49.18%`，不支持把A9 confidence作為PASS主排序。
2. Q5尾端可能存在訊號，但樣本太少且整體不單調；依既有OOS結果設定score threshold會引入新的result-driven tuning，不採用。
3. 因此不建立`SR-C14 = A9 confidence priority`。`SR-C14`仍可依Registry分配給其他新的策略runtime identity，但不得重用`SR-C13`。

### 2026-08-08 — SR-C14正式結果：Capital-utilization first消除macro曝險問題，但CONT11G排序仍不採用

#### 狀態

`RESULT_AVAILABLE / CAPITAL_UTILIZATION_HYPOTHESIS_PARTIALLY_CONFIRMED / CONT11G_DEPLOYMENT_NOT_ADOPTED`

#### 正式結果基準

- 使用者結果ZIP：`test-branch-1_20260808_002357_b220cbb.zip`；SHA256=`eb872749778055d480fa4b3ba9038feacedcce92b4e941d859443f5e2aa7c3be`。
- 正式比較期間：`2021-01-01～2025-12-22`。結束日受`DL-CONT11G / MR-11G` frozen OOS score coverage限制，因此不得直接把本輪C3／C12數值與先前延伸到2026-03-02的C11／C12絕對值混比；本輪C3／C12／C14三者同期間、同Min ROOS P2 active params，可作受控contrast。
- Dataset=`full`、param policy=`base-finalist-best`、max positions=10、rotation=off、config fingerprint=`902c90b40dc2`。
- C14唯一變更維持既定契約：Min ROOS exact cash-cap先判resource mode；capital-utilization mode完全維持Min ROOS；只有cash-binding的DL Selection Mode才使用frozen `DL-CONT11G` continuous score。A9 PASS／REJECT不用於C14。

#### 主要結果

| Arm | 報酬 | MDD | RoMD | 年化 | EV | 曝險 | 交易 | 同參數DL選擇R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SR-C3 Min ROOS | 156.34% | 15.41% | 10.15 | 20.87% | 0.65R | 91.99% | 329 | 0.00R |
| SR-C12 A9 max-PASS resource-aware | 162.59% | 16.47% | 9.87 | 21.46% | 0.82R | 92.38% | 375 | +94.60R |
| SR-C14 Continuous resource-aware | 157.07% | 17.17% | 9.15 | 20.94% | 0.58R | 91.74% | 395 | +14.62R |

- `C14−C3 = +0.73pp報酬 / +1.76pp MDD / -1.00 RoMD / +0.07pp年化 / -0.07R EV / -0.25pp曝險 / +66交易 / +14.62R同參數DL選擇R`。
- `C14−C12 = -5.52pp報酬 / +0.70pp MDD / -0.73 RoMD / -0.52pp年化 / -0.24R EV / -0.64pp曝險 / +20交易 / -79.99R同參數DL選擇R`。
- 年度C14為`2021 18.85% / 2022 -6.85% / 2023 79.66% / 2024 26.96% / 2025 1.80%`；只有2024明顯優於C3，年度方向不穩定。

#### Capital-utilization判讀

1. 使用者假說「舊continuous直接排序失敗可能主要因破壞資金利用率」獲得**部分確認**。C14平均曝險`91.74%`只比C3 `91.99%`低`0.25pp`，與舊raw Score Sort的大幅曝險下降不同，證明capital-utilization-first確實幾乎消除了macro exposure問題。
2. 但micro sizing結構仍改變：平均預留`201,347.89→159,732.00`、平均實際投入`194,644.57→153,067.87`，平均初始停損距離`4.44%→6.36%`；C14實際改單116日、continuous新選入147單、selected score sum增加41.748，但總預留資金增量仍為`-1,861,230`。因此resource gate能守住總曝險，不能保證continuous選到的個別position具有與Min ROOS相同的capital geometry。
3. 更重要的是經濟排序本身不足：C14 EV降至`0.58R`、勝率降至`39.24%`、同參數DL選擇R只有`+14.62R`；相較C12的`+94.60R`明顯較弱。故在macro資金利用問題被大幅控制後，現有`MR-11G` score仍未形成足夠的portfolio selection value。

#### 模型語意限制

- `MR-11G`歷史training scope=`PASS-only`。SR-C14把其frozen OOS continuous score用於全部orderable breakout events，是刻意的controlled deployment hypothesis，不代表模型原本具有all-event ranking語意。
- 因此本結果只能否決「直接把現有PASS-only MR-11G接到capital-first即可」；**不能否決「真正以all-event continuous quality為training semantics的模型 + capital-utilization first」這個較一般的方向**。
- 歷史11F已證明`strategy_aligned_opportunity_no_time_r_v1`在all-label上與binary label高度可分（AUC約0.99），且11E actual Target↔R約0.4875；若後續建立新的all-event continuous ranker，必須將「全事件排序」明確定義成新的model research semantics，不可把MR-11G改名或重用其identity。

#### 採用判定

- `SR-C14 / DL-CONT11G`目前deployment：`NOT_ADOPTED`。
- `SR-C12`仍是「最大化PASS使用」的active research base；`SR-C11`仍保留較佳已知resource-aware經濟結果的歷史地位。
- 不再對MR-11G做runtime sorting微調、score threshold或Min ROOS／continuous混合權重。

#### 下一步

若繼續continuous方向，下一步應是**新的model research**，而不是再改SR-C14：建立真正以`all-events`為training scope的continuous breakout-event quality ranker，再用相同capital-utilization-first runtime作受控比較。開始實作前須重新查Registry分配新的`MR-*`／`DL-*`／`SR-C*` identity；本輪不預先占用新ID，也不依本次OOS結果調任何target係數或runtime權重。

### 2026-08-08 — MR-12A / SR-C15 正式結果：All-event Continuous在相同capital-first runtime下明顯優於PASS-only，但尚不直接promote

#### 狀態

`RESULT_AVAILABLE / ALL_EVENT_TRAINING_SUPPORTED / STRATEGY_PROMISING_NOT_PROMOTED`

#### 正式結果範圍

- 期間：`2021-01-01～2025-12-22`。
- Dataset=`full`、Param=`PARAM-P2 / Min ROOS`、param policy=`base-finalist-best`、max positions=10、rotation=off。
- 正式比較：`SR-C3 / SR-C12 / SR-C15`；config fingerprint=`4da3217c83bd`。
- `SR-C15`完全沿用`SR-C14`的capital-utilization-first、exact cash-cap、cash-binding與fallback契約；唯一模型差異是`DL-CONT11G / MR-11G pass_only → DL-CONT12A / MR-12A all_labels`。
- 本輪使用既有forward/OOS score；Future Target未進runtime。

#### 正式策略結果

| Arm | Return | MDD | RoMD | Annual | EV | Exposure | Trades | Same-param DL selection R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SR-C3` | 156.34% | 15.41% | 10.15 | 20.87% | 0.65R | 91.99% | 329 | 0.00R |
| `SR-C12` | 162.59% | 16.47% | 9.87 | 21.46% | 0.82R | 92.38% | 375 | +94.60R |
| `SR-C15` | **168.69%** | **14.81%** | **11.39** | **22.02%** | 0.64R | 91.71% | 379 | +26.78R |

- `C15−C3 = +12.35pp Return / -0.60pp MDD / +1.24 RoMD / +1.15pp annual / -0.02R EV / -0.28pp exposure / +50 trades / +26.78R same-param DL selection R`。
- `C15−C12 = +6.10pp Return / -1.66pp MDD / +1.51 RoMD / +0.56pp annual / -0.19R EV / -0.67pp exposure / +4 trades / -67.82R same-param DL selection R`。
- C15 Log R²=`0.9039`、月勝率=`71.67%`，均優於C3；勝率=`42.48%`、Payoff=`3.21`、EV=`0.64R`則未改善。

#### All-event training的乾淨受控證據：C15 vs C14

SR-C14與SR-C15使用同一期間、同一Min ROOS參數與同一capital-utilization-first runtime；核心差異是PASS-only trained score改成all-event trained score。歷史SR-C14為Return=`157.07%`、MDD=`17.17%`、RoMD=`9.15`、EV=`0.58R`、Exposure=`91.74%`、Trades=`395`、same-param selection R=`+14.62R`。因此：

- `C15−C14 = +11.62pp Return / -2.36pp MDD / +2.24 RoMD / +0.06R EV / -0.03pp Exposure / -16 Trades / +12.16R same-param selection R`。
- 曝險幾乎完全相同，故這個改善不能再用macro capital utilization差異解釋；它支持`all_labels` training semantics相對MR-11G PASS-only extrapolation更適合all-event deployment。

#### 年度與集中度

C15年度為：`2021 22.70% / 2022 -3.59% / 2023 75.93% / 2024 31.33% / 2025 -1.71%`。

- 相對C3只有2024明顯勝出；2021、2022、2023、2025均略差或明顯較差。
- 以年度報酬鏈結做read-only concentration check，排除2024後：C3約`130.50%`、C12約`122.83%`、C15約`104.56%`；因此C15相對C3/C12的全期報酬優勢高度依賴2024。此計算只用已完成年度結果做歸因，不回流任何runtime或training。
- 相對C14則年度改善較分散：2021 +3.85pp、2022 +3.26pp、2023 -3.73pp、2024 +4.37pp、2025 -3.51pp；5年中3年改善，故「all-event優於PASS-only deployment」的證據比「C15已穩定勝過C3/C12」更強。

#### Capital geometry / per-trade品質

- C15平均曝險`91.71%`，與C3 `91.99%`接近；capital-utilization-first仍成功守住macro exposure。
- 平均實際投入由C3 `194,644.57`降至C15 `172,080.46`，平均初始停損距離由`4.44%`升至`5.68%`，但平均投入資金報酬由`2.22%`升至`3.16%`。
- 期末未滿倉日`890→590`、持股缺口總和`2499→872`，顯示C15雖單筆部位較小，卻能更常填滿portfolio slots／降低持股缺口。
- 同時EV只`0.64R`、same-param DL selection R只有`+26.78R`，遠低於C12 `+94.60R`；因此C15的portfolio優勢不能簡化成「每筆交易R更好」，更可能同時包含position geometry、slot utilization、timing／diversification與compounding效果。

#### 採用判定

1. **模型研究層**：`MR-12A all-event training semantics`獲得正面支持；在完全相同capital-first runtime下明顯優於`MR-11G pass_only` deployment。
2. **策略層**：`SR-C15`目前標記`PROMISING_NOT_PROMOTED`，不立即取代C3/C12。理由不是全期績效不足，而是年度集中度高，且EV／same-param selection R與portfolio總報酬給出不同訊號。
3. 不依本輪OOS結果新增score threshold、blending weight、年份/regime條件或Target係數。

#### 下一步

先做read-only attribution，而不是再訓練或調runtime。下一個診斷應回答：

- C15對C3/C12的超額PnL是否集中在少數月份／少數entry cohorts／少數交易；
- 2024的+20.12pp相對C3改善由哪些已成交selection changes、position sizing與持有重疊造成；
- C15為何在EV與same-param selection R較弱時仍能得到較高portfolio Return與較低MDD；
- 將selection improvement拆成`per-trade R`、`capital return`、`position size / stop distance`、`slot occupancy`與`compounding path`，但全部保持read-only，不新增OOS調參。

取得上述歸因後，再決定是否把SR-C15升格為新的continuous research baseline。

### 2026-08-08 — AUD-c15-strategy-attribution正式結果：C15全期優勢由2024 over-compensate非2024劣勢，機制以portfolio geometry為主

#### 狀態

`AUD-c15-strategy-attribution RESULT_AVAILABLE / SR-C15 PROMISING_NOT_PROMOTED`

#### 程式基準與正式來源

- 使用者ZIP：`test-branch-1_20260808_123516_f128346.zip`
- SHA256：`13904dc779a52ccd259df1614a716d1198f0725dcf67c00d99f8a6134de31d03`
- Audit正式讀取：`outputs/strategy_compare/runs/20260808_011957_C3-C12-C15_4da3217c83bd`
- 比較期間：`2021-01-01～2025-12-22`
- Audit契約：只讀既有strategy replay；未重跑portfolio、未修改score／selector／training，Future Target未進runtime。

#### C15 vs C3

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

#### C15 vs C12

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

#### 歸因判定

1. **SR-C15不升格正式基準。** 2024對C3/C12的Δlog wealth分別占全期淨差異`353.54% / 472.28%`；換成更直觀的exact log-wealth分解，2024單獨約帶來`+18.10% / +11.46%`相對wealth effect，而其餘年份合計約為`-11.25% / -8.19%`。因此不是「2024只是主要貢獻」，而是**2024超額績效必須抵銷其他年份整體落後後，C15才留下全期優勢**。
2. **相對C3，主要改善不是平均R。** C15平均R略低，但單筆占用資金較少、停損距離較寬、slot occupancy明顯提高，且changed days多成交50筆；全期優勢更符合「風險sizing／資金占用／可同時持有數／fill path／compounding」共同作用。
3. **相對C12，交易數差異已很小，仍不能用平均R或selection R解釋。** C15只多4筆成交，exclusive `ΔR=-67.83R`，但exclusive dollar PnL約`+134.7k`且capital return更高；主要差別進一步落在實際risk-dollar、position sizing、交易時點與持有重疊。
4. 既有`SR-C15 vs SR-C14`仍是判斷`MR-12A all_labels`相對`MR-11G pass_only`最乾淨的受控比較，因兩者runtime與macro exposure幾乎相同。現有AUD-c15只比較C3/C12，不能用來取代C14/C15的model-source attribution。

#### Audit輸出補強

同一`AUD-c15-strategy-attribution`升級結果schema至v2，不改任何策略或研究變數：

- 集中度表直接新增focus-year與non-focus的relative wealth effect，避免`>100%` share難以直觀解讀。
- `Selection／trade`直接顯示`Exclusive selection ΔPnL`，並把原誤導性的`All matched trade ΔPnL`更名為`All trade ΔPnL`。
- 新增`SR-C15 selector自身盤前診斷`：直接從既有daily-capacity工件顯示DL-selection／capital-utilization days、selector changed days、相對同日Min ROOS的planned selected-count差、reserved-cash差、score gain與direct-score-order feasibility；仍只讀既有工件、不重跑portfolio。
- Formal synthetic B190/T287同步覆蓋schema v2、non-focus wealth分解與新增selector diagnostics。

#### 下一步

不先建立新的SR-C16，也不回頭調MR-12A模型。下一個最高資訊量工作是**同一capital-utilization-first runtime下的SR-C15 vs SR-C14 read-only attribution**，用來把`all_labels` score source的改善與portfolio geometry分離。完成後若確認MR-12A在相同geometry下仍有穩定經濟貢獻，再把「增加DL實際決策日、同時不惡化盤前資源利用」設計為新的selector研究；selector variant應先在pre-2021 Selection/PIT strategy replay中決定，再回到既有2021+迭代研究OOS評估。

### 2026-08-08 — AUD-c15-source-attribution正式結果：MR-12A all-label source成立，SR-C15仍不promotion

#### 狀態

`AUD-c15-source-attribution RESULT_AVAILABLE / MR-12A SOURCE_SUPPORTED / DL-CONT12A SOURCE_SUPPORTED / SR-C15 PROMISING_NOT_PROMOTED`

#### 正式來源

- Candidate run：`outputs/strategy_compare/runs/20260808_011957_C3-C12-C15_4da3217c83bd`
- Comparator run：`outputs/strategy_compare/runs/20260808_001744_C3-C12-C14_902c90b40dc2`
- 比較期間：`2021-01-01～2025-12-22`
- Candidate：`SR-C15 / DL-CONT12A / MR-12A all_labels`
- Comparator：`SR-C14 / DL-CONT11G / MR-11G pass_only`
- 共用契約：`PARAM-P2 / Min ROOS`、all-off rules、max positions=10、rotation=off、`resource-aware-continuous`，跨run已驗證Min ROOS參數SHA一致。
- Audit只讀既有replay；不重跑、不改score／selector／training。

#### 正式結果

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

#### 判定

1. 相同Min ROOS、相同runtime與幾乎相同macro exposure下，MR-12A all-label source在Return／MDD／RoMD／EV／exclusive R／exclusive dollar PnL／Capital Return均同向優於MR-11G PASS-only source，且非2024仍為正，因此`MR-12A / DL-CONT12A`升為`SOURCE_SUPPORTED`。
2. Source支持不等於策略promotion。SR-C15相對C3/C12仍高度依賴2024，故維持`PROMISING_NOT_PROMOTED`。
3. C15相對Min ROOS出現planned selected-count `+24`但reserved capital `-1.064M`，正式證明「仍為cash-binding」不能保證basket-level資金幾何不退化；後續不再調模型，優先研究selector resource floor。

### 2026-08-08 — SR-C16正式結果＋SR-C17 Max-DL constrained basket實作

#### SR-C16 RESULT_AVAILABLE

程式／回放工件由使用者本地正式`apps/strategy_compare.py`產生；期間`2021-01-01 ～ 2025-12-22`，固定`PARAM-P2 / Min ROOS`、`DL-CONT12A / MR-12A`、all-off rules、max positions=10、rotation=off。

主要結果：C16 Return=175.15%、MDD=15.86%、RoMD=11.05、Annual Return=22.60%、Log R²=0.8773、月勝率=63.33%、EV=0.48R、Exposure=91.83%。相對C15：Return +6.46pp、MDD +1.04pp、RoMD -0.34、Annual Return +0.59pp、EV -0.15R、Exposure +0.11pp；相對C3：Return +18.81pp、MDD +0.45pp、RoMD +0.90、EV -0.17R。

資源契約：DL-selection days=270、selector changed days=179、相對Min ROOS預計選入單數差=+70、reserved-capital delta=+1,361,933、resource-preservation violation days=0。結論為`RESOURCE_CONTRACT_PASSED / NOT_PROMOTED`：C16證明可大幅提高DL介入率且不降低盤前selected-count／reserved-capital floor，但風險品質與EV相對C15退步，不把最高Return直接視為selector最終解。

#### SR-C17 IMPLEMENTED

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

### 2026-08-08 — SR-C17正式結果、Exact Max-DL計算時間評估與SR-C18 feasible-ascent

#### 狀態

`SR-C17 RESULT_AVAILABLE / SELECTOR_NOT_FROZEN`

`SR-C18 IMPLEMENTED / RESULT_PENDING`

#### 程式基準

- 使用者指定ZIP：`test-branch-1_20260808_144639_67acaf9(1).zip`
- SHA256：`45a8c140e8e15fa9164452b4505bc011788b2633888ac36f26882e1ce8847190`
- 本輪開始前已依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`。
- Dataset、Label、`MR-12A / DL-CONT12A`模型權重、forward OOS scores、`PARAM-P2 / Min ROOS`、sizing、accounting、execution、max positions=10、rotation=off皆固定；本輪不重訓模型、不修改Target、不使用forward-OOS績效調score／threshold／blend／resource tolerance。

#### SR-C17正式forward結果

使用者本地正式`apps/strategy_compare.py`工件：config fingerprint=`7a0888323f3b`，期間`2021-01-01 ～ 2025-12-22`。

C17：Return=164.13%、MDD=14.38%、RoMD=11.42、Annual Return=21.60%、Log R²=0.8865、月勝率=68.33%、EV=0.72R、Exposure=92.04%、trades=328、same-param DL selection R=+20.31R。

相對C3：Return +7.79pp、MDD -1.03pp、RoMD +1.27、Annual Return +0.73pp、EV +0.06R、Exposure +0.05pp、trades -1、same-param DL selection R +20.31R。

相對C16：Return -11.02pp、MDD -1.48pp、RoMD +0.37、Annual Return -1.00pp、EV +0.23R、Exposure +0.21pp、trades -80、same-param DL selection R +37.85R。

年度報酬：2021=19.18%、2022=-2.80%、2023=76.00%、2024=32.68%、2025=-2.36%。本輪不依年度／headline return回頭調selector；C17後續變更理由只來自搜尋完整性與計算時間。

Resource diagnostics：Max-DL eligible=223日、repair=209日、final fallback=26日、K violation=0、resource-preservation violation=0、planned selected-count delta=0、reserved-capital delta=+906,375。Min ROOS baseline本身必為合法K／R0 basket，因此26個fallback表示C17 local minimum-repair沒有找到某些已知存在的合法／更佳路徑；這是selector search completeness問題，而不是模型或resource contract問題。

#### Exact global Max-DL搜尋評估：不採用production

本輪先實作工程用exact global search原型，目標與C17完全相同：固定K、R0與basket內Min ROOS execution order，只求hard-feasible basket中`sum(DL score)`全域最大。先測naive DFS Branch-and-Bound，再測以C17合法解作incumbent的best-first K-combination enumeration；兩者都沒有node cap／timeout近似作為正式語意，工程壓力測試只用來判斷可否實際部署。

代表性single-case selector CPU benchmark（K=10）：

- N=12：C17約3.5ms；exact約8.4ms。
- N=14：C17約4.8ms；exact約72ms。
- N=20：C17約16ms；exact約1.33s。
- N=22：C17約17ms；exact約0.67s。
- N=30：C17約22ms；exact超過10s仍未完成。
- N=80：C17約107ms；exact超過10s仍未完成。

真實strategy candidate-day平均候選規模約80，因此global exact的worst／difficult-case成本會直接妨礙後續MR-12B／MR-12C反覆strategy replay。結論：**global exact search只保留工程oracle用途，不建立正式runtime mode、不分配獨立SR ID、不進production。**

#### SR-C18：Max-DL feasible-ascent

為消除C17「repair失敗即停止／fallback」而不引入global exact爆炸成本，分配`SR-C18`。C18與C17使用完全相同：

- `PARAM-P2 / Min ROOS`
- `DL-CONT12A / MR-12A`
- `K = Min ROOS baseline selected_count`
- `R0 = Min ROOS baseline reserved capital`
- basket內正式執行順序固定Min ROOS rank
- full orderable universe與K筆action prefix分離

唯一差異是搜尋完整度：先執行C17取得合法seed；即使C17原本fallback Min ROOS，該baseline也只視為合法seed，不是終點。其後每輪枚舉目前K-basket內每個selected與外部每個unselected的single swap；每個trial以canonical exact reservation重算，只接受`selected_count == K`、`reserved_cost >= R0`且DL quality嚴格提高的方案，並選當輪DL quality最高的feasible improvement。重複直到不存在任何improving feasible single swap，因此輸出保證為deterministic **1-swap local optimum**，但不宣稱global optimum。Capital全程只作hard feasibility，不是objective。

新增runtime policy／mode：`resource-aware-continuous-max-dl-feasible-ascent`。正式比較設定改為`C3 / C17 / C18`，只開`C18-C17`與`C18-C3`。

#### 計算時間與工程驗證

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

#### Direct synthetic / 判定

`validate_strategy_compare_config_driven_app_contract_case`新增C18固定案例：

1. C17 minimum-repair得到次佳合法basket時，C18會在相同K／R0下找到更高DL quality的1-swap feasible improvement。
2. C17 seed原本fallback Min ROOS時，C18仍從該合法seed繼續DL ascent，不把fallback當終點。

目前direct synthetic共32項全部PASS。GPT未執行`apps/test_suite.py`。

C18狀態固定`IMPLEMENTED / RESULT_PENDING`。取得正式forward結果前不得把engineering timing、random benchmark或C17 headline績效當成C18採用證據；也不得依forward結果調single-swap規則、score cutoff、resource floor或任何numeric selector參數。

### 2026-08-08 — SR-C18 pair-level結果與多DL共用baseline timing聚合修正

#### 狀態

`SR-C18 PAIR_RESULT_OBSERVED / FINAL_REPORT_RERUN_REQUIRED`

#### 程式／正式執行證據

- 使用者本地`apps/strategy_compare.py`：config fingerprint=`4621173de22d`，期間`2021-01-01 ～ 2025-12-22`。
- 固定`PARAM-P2 / Min ROOS`、`DL-CONT12A / MR-12A`、max positions=10、rotation=off；本輪不修改selector、score、Target、resource floor或任何numeric gate。
- C17 pair-level重現：Return=164.13%、MDD=14.38%、RoMD=11.42、EV=0.72R、Exposure=92.04%。
- C18 pair-level：Return=140.56%、MDD=14.73%、RoMD=9.54、Annual Return=19.33%、Log R²=0.8810、月勝率=60.00%、EV=0.57R、Exposure=91.98%、trades=333。相對C3：Return -15.78pp、MDD -0.68pp、RoMD -0.61、EV -0.09R。此pair-level結果只作迭代研究OOS證據，不回流修改C18搜尋規則。
- 最終多arm summary尚未完成，因聚合C17／C18兩個pair的重複C3 baseline時拋出：`ValueError: 多DL比較的共用基準不一致: arm=C3, key=resource_aware_selector_timing_max_ms, first=0.088, repeated=0.0352`。因此正式C17／C18 selector timing table尚未取得，不把pair console的整體elapsed秒數當selector CPU比較。

#### 根因

`filters/breakout_quality/strategy_comparison.py::_assert_same_shared_baseline()`原本把所有summary key都當成deterministic baseline identity；SR-C18加入`resource_aware_selector_timing_total/median/p95/max_ms`後，C3在不同controlled pair被重播時會產生正常CPU jitter，因此錯把非決定性runtime measurement當策略結果差異。另`core/portfolio_entries.py`即使`policy is None`也量到wrapper執行時間，使DL-off C3帶有沒有語意的selector timing。

#### 修正

1. `policy is None`時直接回傳canonical default diagnostics，`selector_elapsed_ns=0`；DL-off baseline不再宣稱執行Max-DL selector。
2. shared-baseline equality明確排除全部`resource_aware_selector_timing_*`；Return／MDD／EV／trade count／resource deterministic diagnostics仍維持exact一致性檢查。
3. direct synthetic新增timing-only mismatch可聚合案例，並保留`total_return_pct`不同必須拒絕的原案例；目前`validate_strategy_compare_config_driven_app_contract_case`為33/33 PASS。

#### 下一步

套用修正後以相同`config/strategy_compare.py`正式選單重跑，不改C18 selector。重跑目的只完成final summary與同run C17／C18 selector total／median／p95／max CPU time比較；在正式summary完成前Registry維持`FINAL_REPORT_RERUN_REQUIRED`。

### 2026-08-08 — SR-C18正式closure：Selector freeze，研究主線轉回DL品質

#### 正式結果

使用者本地`apps/strategy_compare.py`以config fingerprint=`4621173de22d`完成`C3 / C17 / C18`正式summary；期間`2021-01-01 ～ 2025-12-22`，固定`PARAM-P2 / Min ROOS`、`DL-CONT12A / MR-12A`、max positions=10、rotation=off。

C18：Return=140.56%、MDD=14.73%、RoMD=9.54、Annual Return=19.33%、Log R²=0.8810、月勝率=60.00%、EV=0.57R、Exposure=91.98%、trades=333、same-param DL selection R=-26.25R。相對C17：Return -23.57pp、MDD +0.36pp、RoMD -1.88、Annual Return -2.27pp、EV -0.15R、Exposure -0.05pp、same-param DL selection R -46.57R。相對C3：Return -15.78pp、MDD -0.68pp、RoMD -0.61、EV -0.09R。

#### Max-DL selector完整度

C18 Max-DL eligible=242日、repair=232日、seed原為fallback=34日、final fallback=0、feasible-ascent improvement=23日、1-swap local optimum=242日、K violation=0、resource-preservation violation=0。相對Min ROOS planned selected-count delta=0、reserved-capital delta=+1,019,846。

因此C18已完成本研究階段要求：所有可介入日皆取得合法K/R0 basket，且搜尋停止點皆為deterministic 1-swap local optimum；不再因C18當前MR-12A績效不佳而修改capital floor、K、repair/ascent規則、score cutoff、blend或其他selector語意。Global exact search先前已因N≈30～80 difficult cases超過10秒而排除production。

#### 正式計算時間

同一正式run：
- C17 selector total=1306.82ms、median=0.660ms、p95=3.431ms、max=15.414ms、repair eval=22,605。
- C18 selector total=1668.23ms、median=0.829ms、p95=4.065ms、max=29.897ms、repair eval=23,456、feasible-ascent eval=25,803。
- C18/C17 slowdown：total=1.277×、median=1.256×、p95=1.185×、max=1.940×；絕對增量total僅361.41ms。整段score-ranking replay console仍約12秒，因此selector額外成本不構成後續反覆模型策略驗證的主要瓶頸。

#### 判定

`SR-C18 RESULT_AVAILABLE / SELECTOR_FROZEN_FOR_MODEL_RESEARCH / NOT_PROMOTED_AS_STRATEGY`。

C18績效顯著低於C17，同時C18把DL objective推得更完整（selected score gain=+14.441 vs C17 +11.826，final fallback=0），卻使same-param DL selection R由C17 +20.31R降至C18 -26.25R。此證據定位為**現有MR-12A ranking品質不足以支撐更強DL主導**，而不是再調selector的理由。後續所有新continuous DL source固定使用SR-C18作策略驗證harness，讓模型差異直接暴露在相同K/R0、execution order、sizing與accounting下。

另核對目前程式：`MR-12A / PROFILE-strategy_aligned_no_time_all_event_mse`本身已使用`daily_percentile_regression`，Target為同日all-event `strategy_aligned_opportunity_no_time_r_v1` percentile，loss=MSE、epoch selection=`mean_daily_spearman`。因此下一個模型研究不得把「daily percentile regression」當新變數；較乾淨的下一步是固定同一Target／all-label scope／ARCH-inception_time_v1，改研究pairwise或listwise ranking loss。

### 2026-08-08 — MR-12B正式策略驗證：Pairwise改善成立，較Max-DL harness放大模型品質差異

#### 狀態

`MR-12B / DL-CONT12B = RESULT_AVAILABLE / MODEL_ECONOMIC_IMPROVEMENT_SUPPORTED`；`SR-C19 / SR-C20 = RESULT_AVAILABLE / MODEL_VALIDATION_ARM`。C17/C18 selector語意維持凍結，不依本次forward/OOS結果調整。

#### 正式controlled replay

- Strategy Compare fingerprint：`a530e2b6f5f4`
- 期間：2021-01-01～2025-12-22
- Dataset：full
- Params：`PARAM-P2 / Min ROOS`
- C17/C19共用minimum-repair selector；C18/C20共用feasible-ascent selector。
- C19唯一模型差異：`DL-CONT12A / MR-12A MSE → DL-CONT12B / MR-12B pairwise`。
- C20唯一模型差異同上。

#### 主要結果

- C19：Return 181.75%、MDD 16.19%、RoMD 11.23、Annual 23.19%、EV 0.70R、Exposure 92.14%、same-param DL selection R +25.05R。
- C20：Return 179.54%、MDD 14.48%、RoMD 12.40、Annual 23.00%、EV 0.84R、Exposure 92.21%、same-param DL selection R +55.31R。
- C19-C17：Return +17.62pp、MDD +1.81pp、RoMD -0.19、EV -0.02R、selection R +4.74R。
- C20-C18：Return +38.98pp、MDD -0.25pp、RoMD +2.86、EV +0.27R、selection R +81.57R。
- C20-C19：Return -2.22pp，但MDD -1.70pp、RoMD +1.17、EV +0.14R、selection R +30.27R。

#### 判定

MR-12B pairwise objective相對MR-12A MSE的模型經濟改善成立。更重要的是，C18-style較Max-DL selector在MR-12A弱ranking時放大錯誤，但在MR-12B改善後同時把MDD、RoMD、EV與selection R顯著拉高，證明C17/C18雙harness具有辨識模型品質與selector轉化差異的價值。後續新模型必須同時跑兩者；不得因C19 raw Return略高或C20風險調整較佳而刪除任一harness。

### 2026-08-09 — P2首輪實跑：Fixed-K結果成立；Dynamic-K score-event-date join bug

#### 狀態

`P2 FIXED_K_RESULT_AVAILABLE / DYNAMIC_K_INVALIDATED_PENDING_RERUN`。不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；MR-12B仍為current model research anchor，MR-12C仍維持REJECTED。此輪只修read-only診斷join，不修改任何模型、score、selector或策略replay。

#### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_022059_051c97b(2).zip`
- SHA256：`afb1556a6374db8e5f5f0f08e982896cddabce4f9ae4e119ad59ea5797e05178`
- 全新解壓工作目錄：`/mnt/data/stock_p2_review`
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`或formal pipeline。

#### 使用者本機P2結果

Fixed K=10的Forward OOS共有603個competition days。MR-12B相對MR-12A：NDCG `0.6905 vs 0.6615`（paired mean Δ `+0.0290`）、Top-K raw-target lift `0.4089R vs 0.2488R`（Δ `+0.1601R`）、Oracle overlap `59.80% vs 57.20%`（Δ `+2.60pp`）、Boundary concordance `52.98% vs 49.96%`（Δ `+3.02pp`）、Boundary gap `+0.2121R vs -0.0489R`（Δ `+0.2610R`）。此結果與C19/C20相對C17/C18的controlled replay方向一致，支持MR-12B broad OOS ranking quality改善。

Selection fixed-K則MR-12B略弱於MR-12A，而MR-12C在Selection描述性指標較高但正式策略經濟結果已被淘汰；Selection rows屬final-refit已見資料，僅保留描述性用途，不作新模型採用依據。

首輪Dynamic-K顯示C17 eligible=223日、三模型完整score僅94日、coverage `42.15%`、K=1～8且mean `1.34`；在該94日子集MR-12B Boundary=`43.94%`、相對MR-12A `-12.53pp`。此數字後續查明受diagnostic join bug污染，不得作MR-12D設計或MR-12B否定證據。

#### 根因

正式Strategy Compare對continuation／re-entry已有canonical契約：交易候選的`signal_date`可能晚於最初breakout事件；runtime score與Future Target都必須優先使用candidate保存的`breakout_quality_score_date`（canonical `score_event_date`）對回模型／PIT工件，只有該欄缺少時才fallback `signal_date`。`strategy_compare_engine._strategy_selection_diagnostics()`已正確實作此契約。

P2 `_dynamic_orderable_frame()`卻只讀`ticker / trade_date / signal_date`，並以`ticker + signal_date`對MR-12A/B/C frozen OOS score。對沿用原始breakout score的continuation／re-entry occurrence會錯失合法score，造成大量partial-score days與錯誤的Dynamic-K paired樣本。

#### 修正

1. P2仍以reference arm既有`score_ranking_orderable_candidates.csv`作共同候選集合，但額外讀取`breakout_quality_score_date`；若非空則解析成`score_event_date`，缺少時才fallback `signal_date`，非法日期fail-fast。
2. MR-12A/B/C frozen score lookup改為`ticker + score_event_date` many-to-one；右側每個模型仍強制該key唯一，禁止many-to-many。
3. coverage新增runtime score-event-date列數、fallback signal-date列數、以及`score_event_date != signal_date`列數，讓後續可直接確認continuation/re-entry映射是否被正確使用。
4. synthetic contract補入`signal_date`晚於原始`breakout_quality_score_date`的later occurrence案例，確保仍能取得原始frozen score。
5. 同輪移除`tools/validate/synthetic_breakout_quality_cases.py`直接讀`doc/PROJECT_SETTINGS.md`文字的反向政策測試，遵守`PROJECT_SETTINGS A7`；保留對實際strategy readable-report程式行為的contract檢查。

#### 下一步

先用同一正式選單重新執行`apps/research.py → 模型訓練 → 比較設定中的 Continuous Rankers`。在修正後Dynamic-K coverage與結果取得前，不新增MR-12D、不改Pairwise loss、不改C17/C18 selector。若修正後coverage顯著提高且MR-12B在Dynamic-K仍弱於MR-12A，再做K-stratified（尤其K=1/2/3）與C18 feasible-swap attribution，判斷是否為top-prefix極前段排序問題；只有歸因成立後才設計新的model experiment。

### 2026-08-09 — P2 Dynamic-K重跑：score-date假設排除；complete-case raw-score限制確認

狀態：`P2_FIXED_K_RESULT_AVAILABLE / DYNAMIC_K_LIMITED_DIAGNOSTIC`。不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；`MR-12B / DL-CONT12B`仍維持current model research anchor，`MR-12C`仍維持REJECTED。

#### 本輪基準與唯一診斷變化

- 最新程式ZIP：`test-branch-1_20260809_031311_2ee25ae.zip`；SHA256：`a140e71c420f05b14d61118c764c543012c763096ae0b39403ff8b0a6ec87c1b`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_031311`；開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。
- 使用者本機以score-event-date修正版重新執行`apps/research.py → 模型訓練 → 比較設定中的 Continuous Rankers`。
- Fixed K=10 Selection／OOS數字與前輪完全相同，表示score-event-date修正只影響Dynamic-K join路徑。
- Dynamic-K仍為C17 max-DL eligible `223`日，其中三模型score+target全候選共同完整日`94`、非共同完整日`129`、day coverage `42.15%`；K=`1～8`、mean=`1.34`。
- `沿用runtime score date列=17,744`，但`score date != signal date列=0`。因此上一輪「continuation/re-entry score-event-date錯接造成42.15% coverage」假設被實跑否定；不得再把該假設當作Dynamic-K弱勢解釋。

#### 仍成立的Fixed-K OOS證據

MR-12B相對MR-12A在603個共同OOS competition days：NDCG `+0.0290`、Top-K Lift `+0.1601R`、Oracle overlap `+2.60pp`、Boundary `+3.02pp`、Boundary gap `+0.2610R`。此結果與C19-C17及C20-C18 controlled strategy replay的模型經濟改善方向一致。

#### Dynamic-K common-complete raw-score結果

在94個C17-reference common-complete competition days：

- MR-12A：NDCG `0.5819`、Top-K Lift `0.5308R`、Boundary `56.47%`、Boundary gap `+0.1133R`。
- MR-12B：NDCG `0.5162`、Top-K Lift `0.1752R`、Boundary `43.94%`、Boundary gap `-0.1365R`。
- MR-12B − MR-12A：NDCG `-0.0657`、Top-K Lift `-0.3556R`、Boundary `-12.53pp`、Boundary gap `-0.2498R`。
- MR-12C同樣弱於MR-12A；相對MR-12B的Boundary僅`+2.51pp`，不足以推翻正式C21/C22已淘汰ListNet的經濟證據。

#### 新確認的診斷語意限制

程式核對確認，P2 Dynamic-K目前不是「實際C17 selector basket品質」：

1. P2只有當日**全部orderable candidates × 全部比較模型的score與target都完整**時才保留整天；正式`continuous-score-max-dl`只要求當天至少一個candidate有continuous score即可進DL selection，缺score candidate會排在已評分候選之後，因此`42.15%`是P2 complete-case day coverage，不是runtime DL可用率。
2. P2對保留日直接依各模型raw score做Top-K／boundary；正式C17還會以Min ROOS固定`K/R0`後做exact-reservation minimum-repair，C18再做feasible-ascent single swaps。因此P2沒有量到resource-feasible最終basket。
3. Dynamic-K的候選集合、K與portfolio state固定取C17 reference replay；C19/C20正式策略各自重播後會因先前選股、持倉、現金與continuation path不同而形成不同後續state。P2是同一reference path上的controlled static diagnostic，不等同各模型native strategy path。

所以目前能下的結論是：**MR-12B在Fixed-K broad OOS ranking較MR-12A好，但在C17-reference、低K、common-complete raw-score子集上較弱；尚不能判定弱點是K=1/2極前段排序、partial-score日、resource repair，或path-dependent selector interaction。**

#### 本輪P2診斷補強

不修改模型／score／selector／strategy replay，只補read-only P2：

- Dynamic-K標題改為`reference path / common-complete raw-score diagnostic`，禁止再標示成「實際selector決策邊界」。
- coverage新增common-complete candidate row rate、reference runtime scored-candidate rate、各模型candidate score coverage與runtime unavailable reason counts。
- 新增依Dynamic K分層的paired表，直接輸出K=1/2/3/...的NDCG、Boundary與Top-K Lift，先判斷MR-12B弱勢是否集中於極低K。
- App簡易摘要同步標示`Dynamic raw-score Boundary`，避免和C17/C18正式selector action品質混淆。

#### 下一步

1. 先原樣重跑P2取得新增candidate-level coverage與K-stratified表，不重訓、不重跑Strategy Compare。
2. 若MR-12B弱勢主要集中K=1/2，進一步做top-prefix decision attribution；若各K都弱，再查complete-case selection bias與score-unavailable候選。
3. 無論K-stratified結果如何，下一個真正對應經濟結果的診斷應以C17/C18各arm的**盤前planned basket / repair / feasible-swap action**為單位，而不是再用generic raw-score Top-K代替selector。若現有持久工件不足以重建planned basket，應先補保存selector action membership/rank的canonical diagnostic欄位，再由既有replay產物或下一次正式replay產生，不得用selected fills冒充盤前planned orders。
4. 在上述歸因完成前，不建立MR-12D、不依94日Dynamic-K子集改Pairwise loss。

### 2026-08-09 — P2 Dynamic-K K分層結果：弱勢集中K=1；新增完整OOS prefix sweep

#### 狀態

`P2_K_STRATIFIED_RESULT_AVAILABLE / FIXED_OOS_PREFIX_SWEEP_IMPLEMENTED`。此為read-only模型品質診斷，不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；`MR-12B / DL-CONT12B`維持current model research anchor，`MR-12C`維持REJECTED。歸因完成前不修改Pairwise loss、不建立新MR。

#### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_035134_5989e50.zip`。
- SHA256：`54beca984605fee7618a80098e2a0b0b2a5ee1f0773bfe4d8062609616d0260d`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_035134`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

#### 使用者本機P2結果

Fixed K=10 OOS維持603個competition days；MR-12B相對MR-12A：NDCG `+0.0290`、Top-K Lift `+0.1601R`、Oracle overlap `+2.60pp`、Boundary `+3.02pp`、Boundary gap `+0.2610R`，broad Top-10 forward-OOS優勢仍成立。

Dynamic-K reference仍為C17：eligible `223`日、common-complete `94`日、day coverage `42.15%`；但common-complete candidate coverage與C17 runtime scored-candidate coverage均為`97.52%`。因此day coverage低不是大量候選缺分，而是少量缺分散落在多個日期造成strict all-candidate complete-day淘汰。

K分布顯示C17 reference path大多數決策是極低K：eligible `K=1:179, K=2:30, K=3:8, K=4:3, K=7:1, K=8:2`，即K=1佔`179/223 = 80.27%`；common-complete中K=1亦佔`71/94 = 75.53%`。

K=1的MR-12B弱勢明確：NDCG `0.4997 vs 0.5883`（B-A `-0.0886`）、Boundary `38.03% vs 57.75%`（`-19.72pp`）、Top-1 raw-target lift `0.0425R vs 0.4903R`（`-0.4478R`）。因K=1時boundary width自動縮成1，Boundary即比較score rank #1與#2的raw target，故`38.03%`代表MR-12B排第一的候選其target高於第二名的比例低於50%。

K=2與K=3的NDCG／Boundary反而偏向MR-12B：K=2 NDCG Δ `+0.0588`、Boundary Δ `+15.38pp`；K=3 NDCG Δ `+0.0089`、Boundary Δ `+4.44pp`。但樣本僅13日與5日，不能作採用結論。K>=4樣本更少，不具判定力。故目前Dynamic-K aggregate B<A主要由K=1主導，不是已證明所有Dynamic-K皆較差。

#### 下一個可執行診斷

為區分「MR-12B一般性的Top-1弱點」與「只在C17 reference path／其日期子集出現」，P2新增config-driven Forward-OOS Fixed-K prefix sweep。預設K=`1,2,3,5,10`，全部使用完整paired OOS event-group候選宇宙，只改K；不重訓模型、不strategy replay、不使用selector state。若完整OOS K=1仍B<A，才支持top-prefix objective mismatch假說；若完整OOS K=1為B>=A，則弱點較可能來自C17 path／regime／candidate subset，下一步應做native selector/action attribution而不是改loss。

#### 實作

1. `config/breakout_quality.py`新增`BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_FIXED_K_VALUES`，並由comparison settings驗證正整數、唯一、遞增；值屬使用者config，不由validator硬編唯一合法答案。
2. `compare_continuous_rankers.py`新增`_evaluate_fixed_k_sweep()`，在同一完整OOS paired frame上逐K呼叫canonical `_evaluate_paired_frame()`；JSON schema升為2，新增`fixed_k_prefix_sweep`，console／Markdown新增Forward-OOS prefix表。
3. `application.py`短版簡易報表新增最小configured K的OOS NDCG Δ與Boundary Δ，直接讀canonical P2 JSON，不重算第二套指標。
4. 既有Pairwise synthetic contract加入isolated config override與fixed-K sweep直接行為檢查。

#### 科學約束

此prefix sweep只作已查看OOS上的迭代研究歸因，符合`PROJECT_SETTINGS E7`；不得進loss、gradient、epoch selection、normalization、sample weighting或hyperparameter optimization。只有先確認弱點的範圍與selector interaction後，才允許提出新的model experiment。

### 2026-08-09 — P2完整Forward-OOS prefix sweep：一般Top-1弱點排除；reference-subset attribution實作

#### 狀態

`P2_PREFIX_SWEEP_RESULT_AVAILABLE / REFERENCE_SUBSET_ATTRIBUTION_IMPLEMENTED`。此為read-only模型品質歸因，不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；`MR-12B / DL-CONT12B`維持current model research anchor，`MR-12C`維持REJECTED。

#### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_040310_a2c253e.zip`。
- SHA256：`f07c982725c40f517e8b76986544f8ab4f24c51bcba87b3ce5f1723925c11552`。
- GPT fresh extract：`/mnt/data/stock_review_040310`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

#### 使用者本機P2結果

完整Forward-OOS共同候選只改K的prefix sweep：

- K=1：Days=`1133`；MR-12B vs MR-12A NDCG `0.5779 vs 0.5302`（Δ `+0.0477`）、Boundary `52.69% vs 47.48%`（Δ `+5.21pp`）、Top-1 Lift `0.8244R vs 0.5069R`（Δ `+0.3175R`）。
- K=2：NDCG Δ `+0.0327`、Boundary Δ `-0.14pp`、Lift `0.6915R vs 0.4404R`。
- K=3：NDCG Δ `+0.0309`、Boundary Δ `+2.27pp`、Lift `0.6168R vs 0.4124R`。
- K=5：NDCG Δ `+0.0281`、Boundary Δ `-0.09pp`、Lift `0.5233R vs 0.3567R`。
- K=10：NDCG Δ `+0.0290`、Boundary Δ `+3.02pp`、Lift `0.4089R vs 0.2488R`。

因此MR-12B在完整Forward-OOS從Top-1到Top-10的NDCG與raw-target lift皆優於MR-12A，K=1本身亦有明顯Boundary優勢；上一輪「Pairwise可能改善broad ranking但犧牲extreme Top-1」假說被本輪完整OOS結果否定，不得據此設計top-1-aware loss。

#### 與C17-reference K=1子集的矛盾

C17 reference path仍有eligible K=1 `179/223`日；common-complete K=1為`71/94`日。該71日orderable raw-score診斷中MR-12B相對MR-12A NDCG `-0.0886`、Boundary `-19.72pp`、Top-1 Lift `-0.4478R`，方向與完整OOS K=1相反。candidate-level score coverage仍為`97.52%`，故此矛盾不能由「一般Top-1能力不足」或大量缺分直接解釋。

目前最小可辨識的兩個來源為：

1. **reference-date / regime conditioning**：C17 max-DL eligible、尤其K=1日期本身是否是一群MR-12B相對弱勢的市場／訊號日期。
2. **orderable candidate / portfolio-state conditioning**：即使固定同一天，經正式策略候選形成、持倉／現金／continuation狀態後留下的orderable universe，是否把MR-12B優勢反轉。

#### 本輪新增read-only attribution

P2新增`reference_subset_attribution`，對每個實際Dynamic-K值固定**同一批common-complete reference dates與同一個K**，同時計算：

- `event_universe`：完整Forward-OOS canonical same-day event-group共同候選；
- `orderable_universe`：既有reference arm盤前orderable candidates。

console／Markdown直接列出MR-12B−MR-12A的NDCG、Boundary與Top-K Lift差異。判讀契約：若同一批K=1 reference dates的Event Δ仍為正，但Orderable Δ轉負，弱勢主要來自candidate/path conditioning；若Event Δ也已轉負，則reference-date/regime conditioning已足以解釋至少部分反轉。此診斷仍不等同resource-feasible最終basket，不重訓模型、不重跑Strategy Compare、不修改selector。

#### 下一步

1. 原樣執行`apps/research.py → 模型訓練 → 比較設定中的 Continuous Rankers`取得新的Reference-path子集歸因表。
2. 若K=1 Event Δ為正而Orderable Δ為負，下一步補C17/C19與C18/C20盤前planned basket／repair／feasible-ascent action attribution；現有持久工件若不足，先在canonical replay diagnostics保存planned action rank/membership，再由正式Strategy Compare產生。
3. 若K=1 Event Δ已為負，先做reference dates的year/regime／candidate-set特徵歸因，再決定是否需要PIT fold stability；不得直接修改loss。
4. 在上述歸因完成前，不建立MR-12D。

### 2026-08-09 — P2 Reference-subset attribution結果：反轉由Orderable universe引入；score-date comparability診斷實作

#### 狀態

`P2_REFERENCE_SUBSET_RESULT_AVAILABLE / SCORE_DATE_COMPARABILITY_IMPLEMENTED`。read-only模型品質歸因，不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；`MR-12B / DL-CONT12B`維持current model research anchor，`MR-12C`維持REJECTED。

#### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_041427_90499bb.zip`。
- SHA256：`ddf8d4d2666b16871ad2c00c794861b7b4511230683c58570104ba9248cd6304`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_041427`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

#### 使用者本機Reference-subset attribution結果

對C17 common-complete K=1的同一71個reference dates固定K=1：

- 完整Forward-OOS `event_universe`：有效competition days=`64`；MR-12B−MR-12A NDCG Δ=`+0.0532`、Boundary Δ=`+4.69pp`、Top-1 Lift Δ=`-0.0052R`。
- C17 `orderable_universe`：competition days=`71`；NDCG Δ=`-0.0886`、Boundary Δ=`-19.72pp`、Top-1 Lift Δ=`-0.4478R`。

因此「reference-date / regime conditioning本身使MR-12B Top-1變差」不成立：同一批日期回到canonical same-day event universe後，MR-12B的NDCG與Boundary仍優於MR-12A；負向反轉是在正式C17 orderable candidate universe形成後才出現。K=2 Event與Orderable NDCG Δ皆為正（`+0.0797/+0.0588`），其餘K樣本過少不作採用判定。

#### 新的最小機制假說

MR-12B training objective為**within-day RankNet**：pair只在相同event date內建立，loss只約束同日margin ordering。正式C17 orderable pool則可在同一trade date同時包含新訊號與延續中的歷史訊號；這些候選沿用各自原始`breakout_quality_score_date`的frozen score，因此runtime會比較不同`score_event_date` cohort的score。

這形成可直接驗證的objective/runtime mismatch假說：

- `same score-event-date pairs`接近MR-12B實際training pair scope；
- `cross score-event-date pairs`從未被MR-12B pairwise loss直接約束其absolute score scale；
- MR-12A的daily-percentile MSE雖也是same-day target，但逐row MSE對[0,1] percentile提供absolute score anchor，理論上較容易保留跨date cohort可比性。

此機制目前僅為待驗證假說；不得在取得pair-scope診斷前建立MR-12D或修改loss。

#### 本輪新增read-only診斷

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

#### 判讀規則與下一步

- 若MR-12B在same-score-date pairs仍優於／不弱於MR-12A，但cross-score-date pairs明顯轉負，則支持「跨score-date cohort可比性」是主要runtime落差；下一個model research才可考慮以Selection-only資料做`within-day pairwise + absolute percentile anchor`之單一objective變更。
- 若same-score-date與cross-score-date皆轉負，則orderable universe還有其他candidate-lifecycle／portfolio-state conditioning，下一步拆score age、candidate type、fresh/continuation與C17/C19 native planned basket action。
- 在上述診斷完成前，不建立MR-12D、不修改C17/C18。

### 2026-08-09 — P2 score-date comparability結果：跨date尺度失配排除；orderable event-target語意邊界確認

#### 狀態

`P2_SCORE_DATE_COMPARABILITY_RESULT_AVAILABLE / CROSS_DATE_MISMATCH_REJECTED`。此為read-only模型品質／候選語意歸因，不占用新的`MR-*`／`DL-*`／`SR-C*`／`AUD-*` identity；`MR-12B / DL-CONT12B`維持current model research anchor，`MR-12C`維持REJECTED。不得依本輪orderable原始event-target診斷修改Pairwise loss。

#### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_043045_8dbc562.zip`。
- SHA256：`2e4383cb7eef5f9a83e27b18a891476df63291e47f51e037ff674e7e4e02861e`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_043045`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

#### 使用者本機結果

在C17 reference common-complete orderable universe：

- score-age>0候選=`3246 / 3246 = 100%`；score age mean=`20.22`天、median=`14`天、max=`288`天。
- mixed score-date日=`92/94 = 97.87%`；K=1 mixed score-date日=`70/71 = 98.59%`。
- All pairs：MR-12B Pair=`54.89%` vs MR-12A=`52.50%`，Δ=`+2.39pp`；mean-daily Δ=`+1.38pp`。
- Same score-date：Pair Δ=`+0.67pp`；mean-daily Δ=`+2.79pp`。
- Cross score-date：Pair Δ=`+2.49pp`；mean-daily Δ=`+1.38pp`。
- K=1 same score-date：Pair Δ=`+0.45pp`；mean-daily Δ=`+2.46pp`。
- K=1 cross score-date：Pair Δ=`+2.14pp`；mean-daily Δ=`+1.48pp`。

因此上一輪「MR-12B within-day RankNet缺乏跨score-event-date absolute comparability，導致C17 orderable K=1反轉」假說被直接否定：MR-12B在same-date與cross-date pair ordering都不弱於MR-12A，且cross-date優勢更大。

#### 新確認的語意邊界

P2 Dynamic-K / score-date comparability目前使用的方向真值仍是每個候選**原始score-event-date**對應的`strategy_aligned_opportunity_no_time_r_v1 target_raw_r`。該Target由原始事件日後固定future horizon建立；當候選在較晚trade date仍orderable時，P2並沒有重新錨定到當下trade date，也沒有重建剩餘可實現strategy R。

本輪orderable候選全部`score_age>0`，且median 14天、mean 20.22天，故：

1. P2仍可回答「frozen score對原始事件Target ordering保留多少」；
2. 但Dynamic-K Top-K Lift／Boundary不得解讀成「在當下trade date買入後的counterfactual realized R」；
3. C17 K=1的Event `B>A`、Orderable原始event-target `B<A`與正式C19/C20經濟結果`B>A`並不形成模型採用矛盾；目前缺的是trade-date action outcome attribution，而不是更多原始event-target切片。

#### 本輪報表契約修正

- P2 JSON schema提升為5；Dynamic-K與score-date comparability明確保存`target_anchor=original_score_event_date_target`與`trade_date_remaining_opportunity_evaluated=false`。
- console／Markdown新增score-age bucket顯示，並明確註記原始event-target不是trade-date remaining opportunity / counterfactual realized R。
- 不修改任何score、Target建置、模型、selector、Strategy Compare、accounting或execution。

#### 下一步

1. **模型研究主線回到MR-12B Selection PIT**：由正式選單建立／更新Selection PIT Scores並做PIT模型驗證，確認Pairwise ranking優勢是否跨歷史fold穩定；PIT只使用其合法Selection／Validation時間窗，不讀Forward-OOS作訓練或選模。
2. P2到此不再因C17 K=1原始event-target反轉設計新loss；MR-12D暫不建立。
3. 若仍要解釋C17/C19、C18/C20的portfolio action差異，下一個診斷必須以**orderable trade date的canonical counterfactual／realized strategy outcome**為真值，並直接對planned basket／repair／feasible-ascent action做歸因；不得再用原始score-event-date Target冒充當下剩餘機會。

### 2026-08-09 — MR-12B Selection PIT模型驗證：10/10年度rho為正，Gate PASS

#### 狀態

`MR-12B / DL-CONT12B`更新為`PIT_MODEL_VALIDATION_PASS / MODEL_ECONOMIC_IMPROVEMENT_SUPPORTED`。本輪只取得既有MR-12B的Selection point-in-time模型驗證結果，不建立新`MR-*`、`DL-*`、`SR-C*`或`AUD-*` identity。

#### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_045404_66521e1.zip`。
- SHA256：`710539a6fc783608572c753a050e2a83237d6500919fbebfddd1e63690abff5a`。
- GPT fresh extract：`/mnt/data/stock_review_20260809_045404_710539a6`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

#### 使用者本機PIT結果

- Profile：`strategy_aligned_no_time_all_event_pairwise`；objective=`daily_pairwise_ranking`；scope=`all_labels`；seed=42。
- PIT period：`2011-01-01～2020-12-31`；12-month score folds／24-month inner validation；folds=`10`。
- PIT scores：`23,932 / 23,932` groups，coverage=`100.00%`；本輪10 folds全部重用既有合法工件，未重訓。
- PASS-only target ordering：global rho=`0.2336`；mean daily rho=`0.1859`；top-bottom spread=`1.1386R`。
- 年度穩定：rho>0=`10/10`；spread>0=`9/10`。
- 分類重疊：AUC=`0.5670`；Top decile PASS=`63.07%`；Overall PASS=`55.46%`。
- Gate=`PASS`。Gate固定只使用PASS-only global/daily Spearman與多數年度rho/spread方向，不使用策略績效。
- `drift=True`：audit定義為至少一組相鄰fold的score平均值位移`>=1.0 pooled score SD`。此旗標描述score level／calibration drift，不代表年度ranking方向失敗，也不是目前predeclared Gate veto。
- Orderable coverage尚未提供，依正式語意於後續Selection strategy replay建立。

#### 判定

MR-12B的Selection PIT模型層證據通過：10個年度的PASS-only rho全部為正、9/10年度top-bottom spread為正，且全期global／daily rho均為正。這補上Forward-OOS與C19/C20 controlled replay之外的歷史時間穩定性證據。`drift=True`需在策略層留意跨fold score level，但目前沒有證據支持因該warning修改Pairwise loss或拒絕MR-12B。 此audit只評估MR-12B自身是否具有正向且跨年穩定的PIT排序能力，**沒有與MR-12A做same-fold paired comparison**，因此不可把Gate PASS寫成「MR-12B在歷史每個fold都優於MR-12A」。若要補齊learning-objective歷史比較，應以相同PIT日期／候選做MR-12B−MR-12A paired PIT read-only comparison。

#### 下一步

1. 進入正式Selection PIT策略績效驗證，讓歷史每個交易日只使用當時合法PIT score，並建立orderable score coverage；比較時維持既有策略參數／selector contract，禁止依PIT結果回頭修改loss、epoch或score normalization。
2. 策略層優先看Return、MDD、RoMD、EV、Exposure／資金利用率與年度穩定，確認PIT模型排序能力是否能轉成無前視portfolio economics。
3. 若策略結果因fold邊界或跨foldscore level出現異常，再做read-only cross-fold/action attribution；`drift=True`本身不足以建立新model experiment。

### 2026-08-09 — MR-12B Selection PIT C23/C24/C25策略經濟驗證結果

#### 狀態

`RESULT_AVAILABLE / PIT_MODEL_GATE_PASS / PIT_STRATEGY_NOT_PROMOTED / REALIZATION_ATTRIBUTION_NEXT`

#### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_055527_b0e14c8.zip`。
- SHA256：`0b3702630954ad39a254de2805f30b5cb6239c7df43f500630824e93bdda88ba`。
- GPT fresh extract：`/mnt/data/stock_review_055527`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

#### 固定條件

2014-01-01～2020-12-31；historical P2 Min ROOS active params；`base_finalist_best`；max positions=10；rotation=off；optional entry filters=all-off；hard filter off；Selection PIT source=`DL-CONT12B-PIT`；Dataset、Continuous Target、PIT folds／scores／audit、entry／stop／exit、accounting與portfolio規則固定。C23為DL-off baseline；C24只加入MR-12B PIT + C17 minimum-repair；C25只加入同一PIT source + C18 feasible-ascent。Future Target只於replay完成後join，不進runtime。

#### 主要結果

| Arm | Return | MDD | RoMD | Annual | EV | Exposure | Trades | same-param DL selection R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C23 | 127.45% | 25.45% | 5.01 | 12.46% | 0.62R | 87.68% | 379 | 0.00R |
| C24 | 108.05% | 24.51% | 4.41 | 11.04% | 0.53R | 87.40% | 374 | -35.78R |
| C25 | 115.42% | 26.36% | 4.38 | 11.59% | 0.59R | 87.65% | 366 | -19.52R |

Controlled deltas：

- C24-C23：Return `-19.40pp`、MDD `-0.94pp`、RoMD `-0.60`、Annual `-1.42pp`、EV `-0.09R`、Exposure `-0.29pp`、selection R `-35.78R`。
- C25-C23：Return `-12.03pp`、MDD `+0.91pp`、RoMD `-0.63`、Annual `-0.87pp`、EV `-0.03R`、Exposure `-0.03pp`、selection R `-19.52R`。
- C25-C24：Return `+7.37pp`、MDD `+1.85pp`、RoMD `-0.03`、Annual `+0.55pp`、EV `+0.06R`、Exposure `+0.25pp`、selection R `+16.26R`。Feasible-ascent改善minimum-repair的經濟轉化，但仍不足以超越C23。

#### Selection／resource診斷

- C24相對C23：selected Target percentile `+0.0164`、Target mean `+0.0476R`、opportunity gap改善`0.1373R`，但selection R `-35.78R`、期末未滿倉日`+318`、持股缺口`+297格日`。
- C25相對C23：selected Target percentile `+0.0178`、Target mean `+0.0306R`、opportunity gap改善`0.0607R`，但selection R `-19.52R`、期末未滿倉日`+259`、持股缺口`+274格日`。
- Candidate supply幾乎不變：平均每日可掛單候選C23/C24/C25=`65.33/63.25/65.10`，供給不足日=`101/103/102`；因此策略惡化不能主要歸因於候選供給不足。
- C24/C25平均曝險仍接近C23，代表「更多未滿倉日」不等於總資金曝險同比例下降；需要拆解position sizing、fill、holding/partial-tail slot occupancy與capital deployment。
- C24的Future Target mean／opportunity gap優於C25，但C25的Return與selection R反而較好；這是直接證據顯示`strategy_aligned_opportunity_no_time_r_v1`的原始event Target改善不等於portfolio可實現R改善。

#### 判定

1. MR-12B的Selection PIT模型Gate仍維持`PASS`；本次不否定Pairwise模型研究結論。
2. `DL-CONT12B-PIT`直接套用historical P2 Min ROOS的C24/C25皆`NOT_ADOPTED`；不得升格正式策略source。
3. C25相對C24的改善證明selector搜尋完整度有影響，但不是根因；較Max selector仍無法使PIT source超越baseline。
4. 本次最強訊號為Target→realized economics realization gap：post-replay Future Target改善與same-param selection R／總報酬方向相反。不得直接依Selection結果修改MR-12B loss、architecture或OOS selector。

#### 下一步

下一個最高資訊量工作是read-only **PIT strategy realization/capture attribution**，直接重用C23/C24/C25既有replay工件，不重跑portfolio、不重訓模型、不跑optimizer。至少拆解：exclusive trade realized R／PnL、planned/reserved→fill、position sizing與stop distance、holding與partial-tail slot-days、capital deployment、Target capture ratio／realization gap、以及年度貢獻；優先比較C24-C23與C25-C23，再用C25-C24隔離selector轉化。只有歸因確認有明確可由既有策略參數空間修正的機械瓶頸，才考慮Selection內參數適應；否則維持C23 baseline並把PIT ranking的直接策略部署淘汰。

### 2026-08-09 — AUD-c23-c25-pit-realization結果：winner capture不足；修正capture-gap不等於參數機械瓶頸

#### 狀態

`RESULT_AVAILABLE / REALIZATION_GAP_CONFIRMED / PARAM_ADAPTATION_NOT_SUPPORTED`。本輪取得使用者本機只讀Audit結果，並修正既有capture decision將「capture惡化」本身誤當成可由策略參數適應檢驗之機械瓶頸的過度判定。未重跑portfolio、未重訓模型、未跑optimizer、未建立新`MR-*`／`SR-C*`。

#### 本輪基準

- 使用者最新版ZIP：`test-branch-1_20260809_065116_e3092c2.zip`。
- SHA256：`c714add288f842d6b7383f5a577bf38220785beffc55e841af69bebf3c269729`。
- GPT fresh extract：`/mnt/data/stock_review_065116`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

#### 使用者本機Audit結果

##### C24 vs C23

- Return `127.45% → 108.05%`（`-19.40pp`）；RoMD `5.01 → 4.41`；EV `0.62R → 0.53R`；exclusive selection R=`-35.78R`。
- 平均Target R `0.46R → 0.50R`（`+0.04R`），但平均Realized R `0.62R → 0.53R`（`-0.09R`）；aggregate capture `1.44 → 1.03`（`-0.41`）。
- Fill `-0.80pp`、平均投入`-3,832`（約`-2.46%`）、投入／預留比`-0.58pp`、平均持有`-0.28日`、平均曝險`-0.29pp`。這些均未跨既定mechanical-gap門檻（fill `-2pp`、sizing `-10%`、holding `+3日`、exposure `-5pp`）。
- Exclusive trades：C23-only winners `69 / +203.30R`、losers `88 / -73.68R`；C24-only winners `57 / +173.78R`、losers `95 / -79.94R`。
- R分解：winner contribution約`-29.52R`；loser contribution約`-6.26R`；合計`-35.78R`。主要driver=`winner_capture`。Exclusive decisive win rate約`43.95% → 37.50%`（`-6.45pp`）。

##### C25 vs C23

- Return `127.45% → 115.42%`（`-12.03pp`）；RoMD `5.01 → 4.38`；EV `0.62R → 0.59R`；exclusive selection R=`-19.52R`。
- 平均Target R `0.46R → 0.52R`（`+0.06R`），但平均Realized R `0.62R → 0.59R`（`-0.03R`）；aggregate capture `1.44 → 1.18`（`-0.26`）。
- Fill `-1.61pp`、平均投入`+8,288`、投入／預留比`-0.57pp`、平均持有`+1.17日`、平均曝險`-0.03pp`；同樣未跨既定mechanical-gap門檻。
- Exclusive trades：C23-only winners `73 / +162.85R`、losers `99 / -86.87R`；C25-only winners `61 / +140.82R`、losers `98 / -84.36R`。
- R分解：winner contribution約`-22.03R`；loser contribution約`+2.51R`（candidate反而少承擔loser R）；合計`-19.52R`。主要driver=`winner_capture`。Exclusive decisive win rate約`42.44% → 38.36%`（`-4.08pp`）。

#### 判定修正

舊`score_ranking_capture._decision`把`capture_gap`與fill／sizing／deployment／holding並列，只要Target改善且經濟失敗，capture gap單獨成立就會輸出`ADAPTATION_DIAGNOSTIC_SUPPORTED`。這會把**結果層的Target→Realized mismatch**誤稱為**可由既有策略參數空間修正的機械瓶頸**。本輪改為：

1. `capture_gap`仍保存為`realization_gap_detected`，但不單獨觸發parameter adaptation。
2. 只有預先定義的fill／sizing／deployment／holding mechanical gap至少一項成立，才可輸出`ADAPTATION_DIAGNOSTIC_SUPPORTED`。
3. 若Target改善、經濟失敗、capture gap成立，但沒有mechanical gap，狀態改為`SORT_ONLY_REJECTED_REALIZATION_GAP_NO_MECHANICAL_BOTTLENECK`，`parameter_adaptation_candidate=false`。
4. `AUD-c23-c25-pit-realization`新增exclusive winner/loser R contribution、exclusive win rate與implied selection R閉環；報表明確區分winner capture與loser avoidance。

#### 科學判定

- C24/C25的直接PIT部署仍`NOT_ADOPTED`。
- 本Audit不否定MR-12B PIT模型Gate；它證明的是**高Target selection沒有被既有策略路徑轉成較高realized R**。
- 目前損失主要來自「少捕捉winner」，不是loser severity、fill、sizing、holding或平均exposure已被證實惡化到足以構成參數適應mechanical hypothesis。
- 因此**現在不進Selection參數適應**，也不從本Audit直接修改MR-12B loss／threshold／selector。

#### 下一步

下一個最高資訊量read-only診斷優先檢查**PIT-specific fold score drift / mixed-fold runtime conditioning**：PIT audit已有`drift=True`，而Strategy replay的orderable候選可跨年度fold保留frozen score。應直接量測trade date是否混用不同PIT fold model scores、fold boundary附近的exclusive winner capture／selection R是否異常，以及負向selection R是否集中mixed-fold日期。這是PIT-specific問題，先於新Target／新MR。若fold conditioning不能解釋loss，再回到Target與realized trade-path語意研究；不得先跑optimizer。

### 2026-08-09 — AUD-c23-c25-pit-fold-runtime結果：fold drift／mixed-fold不是PIT直接部署失敗主因

#### 狀態

`RESULT_AVAILABLE / FOLD_DRIFT_NOT_PRIMARY_CAUSE / CROSS_FOLD_NORMALIZATION_NOT_SUPPORTED`。本輪取得使用者本機只讀Audit結果；未重跑portfolio、未重訓／校正PIT模型、未跑optimizer。

#### 程式基準

- 使用者最新版ZIP：`test-branch-1_20260809_082042_f0e60a4.zip`。
- SHA256：`33eedd1798f6a4ecfacc85c65405a64dafb9eda8c7c8da7763fbcded281931fb`。
- GPT fresh extract：`/mnt/data/stock_review_082042`。
- 開始前依序讀取`PROJECT_SETTINGS → BREAKOUT_QUALITY_EXPERIMENT_REGISTRY → BREAKOUT_QUALITY_EXPERIMENT_LOG`；未執行`apps/test_suite.py`。

#### 使用者本機Audit結果

##### C24 vs C23

- PIT drift=`True`，max adjacent mean shift=`1.03 pooled SD`，flagged fold=`fold_20170101_20171231`。
- Mixed-fold days=`353/1633 (21.62%)`，cross-fold pair share=`5.12%`，score age mean/median=`26.16/17.00日`。
- Exclusive selection R=`-35.78R`，但mixed-fold=`+13.91R`；single-fold=`-46.05R`；no-orderable=`-3.64R`。
- Fold boundary ±30日內=`+2.17R`；outside=`-37.95R`。
- 因此負R不集中mixed-fold或fold transition，反而主要在正常single-fold／boundary outside。

##### C25 vs C23

- Mixed-fold days=`357/1635 (21.83%)`，cross-fold pair share=`5.55%`，score age mean/median=`26.22/17.00日`。
- Exclusive selection R=`-19.52R`；mixed-fold=`-4.25R`、single-fold=`-10.65R`、no-orderable=`-4.62R`。
- Fold boundary ±30日內=`+4.93R`；outside=`-24.45R`。
- mixed-fold有部分winner capture損失，但無法解釋主要總損失；fold boundary更不是負R集中區。

#### 判定

1. PIT audit的`drift=True`保留為score-level warning，但本次實際策略損失不支持其為C24/C25 winner-capture失敗主因。
2. 不建立Selection-only cross-fold normalization／calibration runtime；不得因本Audit修改MR-12B loss、epoch、threshold或selector。
3. 下一步回到Target／realized outcome語意：直接使用已成交exclusive trades，檢查PIT score與原始event Target對realized R的對齊，並按signal→entry age拆分，以區分event Target老化與Target公式本身失配。
4. 既有11J per-candidate counterfactual已正式STOPPED，不能為此重啟；未成交／未選候選realized R仍保持未知。

### 2026-08-09 — AUD-c23-c25-pit-target-realization結果：C25負Selection R集中old-signal tail

#### 狀態

`RESULT_AVAILABLE / C25_OLD_SIGNAL_TAIL_SUPPORTED / C24_NON_MONOTONIC / NO_MODEL_OR_TARGET_CHANGE`。

#### 程式基準

- ZIP：`test-branch-1_20260809_123909_0537dbe.zip`
- SHA256：`d62779ff43005ba1d6eb75f6bb9ab7cddf79bab6156c605c1a0439c79226455c`
- Audit：`AUD-c23-c25-pit-target-realization`
- 期間：2014-01-01～2020-12-31；只讀既有`SR-C23/C24/C25` completed replay、validated Selection PIT score與原始Continuous Target。
- 不重跑portfolio、不建立未成交counterfactual、不重訓／校正模型、不改selector／params／Target。

#### 結果

##### C24 vs C23

- Exclusive selection R=`-35.78R`；Target/score coverage=`93.85%`。
- C24-only相對C23-only：平均Target `+0.10R`，平均Realized `-0.38R`，win rate `-9.32pp`。
- Target↔Realized rho：C23-only=`0.34`、C24-only=`0.23`；Score↔Realized rho：`-0.04/-0.15`。
- Age buckets Selection ΔR：Q1=`+31.73R`、Q2=`-50.95R`、Q3=`-9.41R`、Q4=`-21.61R`。
- 負R不是單調隨age增加；因此C24不支持簡單全域age cutoff。

##### C25 vs C23

- Exclusive selection R=`-19.52R`；Target/score coverage=`93.66%`。
- C25-only相對C23-only：平均Target `+0.14R`，平均Realized `-0.11R`，win rate `-4.62pp`。
- Target↔Realized rho：C23-only=`0.36`、C25-only=`0.35`；Score↔Realized rho：`-0.02/-0.06`。
- Age buckets Selection ΔR：Q1=`+5.67R`、Q2=`+13.31R`、Q3=`+2.76R`、Q4=`-40.18R`。
- Covered trades的Q1～Q3合計=`+21.74R`；Q4(age `23～288` calendar days，median=`38.5`)單獨反轉為`-40.18R`。C25-only Q4平均Realized=`0.42R`，C23-only=`1.50R`；Q4 Target則`0.23R vs 0.25R`，已不存在前3個bucket的Target優勢。
- Age↔Realized與Age↔(Target−R) Spearman接近0，表示不是平滑單調age效應，而是尾端／threshold-like現象。

#### 判定

1. `MR-12B`與原始Continuous Target目前均**不因本Audit被淘汰或修改**；C25的Target↔Realized仍約`0.35`，不支持「Target全面失效」。
2. `SR-C25` feasible-ascent是下一個runtime研究基礎；C24因age pattern非單調，不作age-guard基礎。
3. 下一個Selection-only受控假說應是**stale-score membership guard**：過舊PIT score不得驅動DL造成basket membership change，但candidate本身不得被拒絕／過期，需完整保留Min ROOS fallback與K/R0資源契約。
4. 若實作固定score-age cutoff，該值只能由本Selection evidence預先固定並在OOS前凍結；OOS結果不得再用來調cutoff。未成交／未選候選仍不得建立counterfactual R。
5. 在這個runtime guard完成Selection驗證前，不建立新`MR-*`、不重做Target label、不進參數適應。

#### 下一步

先以`SR-C25`為控制基準實作單一runtime變更：stale-score只禁止DL membership change，不刪候選、不改score、不改feasible-ascent的K/R0 hard feasibility。Selection內與`SR-C23`及`SR-C25`比較Return、RoMD、EV、same-param DL selection R、underfilled slot-days與guard觸發日；若Selection改善再凍結同一規則進Forward-OOS驗證。

### 2026-08-09 — SR-C26 Selection結果：stale guard修復selection R，但portfolio gate仍未通過

#### 狀態

`RESULT_AVAILABLE / STALE_GUARD_SUPPORTED / NOT_PROMOTED / FORWARD_OOS_NOT_AUTHORIZED`。

#### 程式基準與正式結果

- 使用者本機執行基準：`test-branch-1_20260809_150343_3b98cc3.zip`。
- SHA256：`fe7d7aed0ec5ccbc126ff0f45b9c542cca82b1f1dc1923ebdda060b21bde9a48`。
- Strategy Compare run：`outputs/strategy_compare/runs/20260809_150908_C23-C25-C26_6018e78bd663`。
- 期間：2014-01-01～2020-12-31；historical P2 Min ROOS active params；MR-12B Selection PIT；C26唯一scientific change仍為預先凍結的22-calendar-day stale-score membership guard。

#### C26相對C25：guard純runtime效果

- Return `115.42% → 119.74%`（`+4.32pp`）。
- MDD `26.36% → 27.11%`（`+0.75pp`）；RoMD `4.38 → 4.42`（`+0.04`）。
- EV `0.59R → 0.67R`（`+0.08R`）。
- same-param DL selection R `-19.52R → +7.99R`（`+27.51R`）。
- underfilled end days `963 → 923`（`-40`）；position-gap slot-days `1927 → 1781`（`-146`）。
- 實際改單日 `94 → 75`；Continuous新選入單 `97 → 79`；guard觸發日=`43`、blocked feasible score-improving swaps=`378`。
- 判定：22日stale-score guard確實修復C25 old-signal tail造成的selection loss；Selection evidence支持guard機制本身。

#### C26相對C23：Selection portfolio gate

- Return `127.45% → 119.74%`（`-7.71pp`）。
- MDD `25.45% → 27.11%`（`+1.66pp`）；RoMD `5.01 → 4.42`（`-0.59`）；年化 `-0.55pp`。
- EV `0.62R → 0.67R`（`+0.05R`），same-param DL selection R=`+7.99R`。
- 平均曝險只`-0.20pp`，但underfilled end days仍`+219`、position-gap slot-days仍`+128`。
- 選中候選Target percentile `+0.0237`、Target mean `+0.0680R`；Target方向仍改善。
- 年度Return僅2014、2015、2019優於C23；2016、2017、2018、2020較弱。

#### 科學判定

1. C26把C25的負selection R修成相對C23 `+7.99R`，故stale-score guard假說不是無效；不得把本結果解讀成回到C25。
2. 但主要策略Gate仍以Return／MDD／RoMD等portfolio結果為準；C26尚未超越C23，因此**不得進Forward-OOS**。
3. `+7.99R selection R`與`-7.71pp Return`形成新的R→portfolio translation gap。不能據此再調22日cutoff、增加第二個age threshold或修改MR-12B／Target。
4. 下一步只允許read-only attribution：拆exclusive selection ΔPnL、common-trade ΔPnL、position sizing／capital return、slot occupancy與wealth/compounding path，判斷positive R為何沒有轉成positive portfolio wealth。

### 2026-08-09 — AUD-c23-c26-pit-portfolio-translation結果：正Selection R未轉成dollar PnL／wealth

#### 狀態

`RESULT_AVAILABLE / R_TO_DOLLAR_TRANSLATION_GAP_CONFIRMED / C26_NOT_PROMOTED`。

#### 程式基準與來源

- 使用者本機執行基準：`test-branch-1_20260809_152631_768fd77.zip`。
- SHA256：`28c35d36c73ca4a6278bbd2f026d9a1b43c793ecac82e841fb65658198f28156`。
- Audit：`AUD-c23-c26-pit-portfolio-translation`；source=`outputs/strategy_compare/runs/20260809_150908_C23-C25-C26_6018e78bd663`。
- 期間2014-01-01～2020-12-31；只讀既有replay，不重跑portfolio、不改score／selector／training。

#### C26 vs C23

- Return=`119.74% vs 127.45%`（`-7.71pp`）；MDD=`+1.66pp`、RoMD=`-0.59`、EV=`+0.05R`。
- Exclusive selection ΔR=`+7.99R`，但Exclusive selection ΔPnL=`-35,524.03`。
- Common trades ΔPnL=`-41,604.23`；All trade ΔPnL=`-77,128.26`。
- Final relative wealth advantage=`-3.39%`；2020只占全期淨差異`12.61%`，非2020 relative wealth effect=`-2.97%`，故不是單一年份集中。
- 平均實際投入`+2,053`、平均預留`+2,607`、平均停損距離`+0.58pp`、平均Capital Return約持平、平均Realized R=`+0.05R`、平均持有日`+2.05`。
- Underfilled end days=`+219`、position-gap slot-days=`+128`；changed days成交買單差=`-15`、missed buy差=`+2`。

#### C26 vs C25

- Return=`+4.32pp`、RoMD=`+0.04`、EV=`+0.08R`。
- Exclusive selection ΔR=`+27.51R`、Exclusive selection ΔPnL=`+57,383.45`、Common trades ΔPnL=`-14,203.16`、All trade ΔPnL=`+43,180.29`。
- Final relative wealth advantage=`+2.00%`；underfilled end days=`-40`、position-gap slot-days=`-146`。
- 這證明22日stale-score guard不只改善未加權R，也改善actual dollar trade PnL；不得回退C25或重新調22日門檻。

#### 判定

1. C26相對C23的portfolio失敗由兩層共同構成：exclusive trades雖ΣR較高但dollar PnL較低；common trades亦因後續portfolio path出現負PnL差。
2. `Common trades ΔPnL`可能是前段wealth差異造成後續position sizing縮放的下游放大，不能與exclusive selection loss直接視為兩個獨立根因。
3. 現有平均invested／stop distance／capital return不足以解釋`+7.99R → -35.5k`；下一步只能在同一read-only Audit內拆每筆initial-risk dollars與共同交易R×Risk分解，不得新增cutoff、age gate、參數適應或Forward-OOS。

### 2026-08-09 — AUD-c23-c26-pit-portfolio-translation risk-dollar重跑結果：C26 runtime微調鏈結案

#### 狀態

`RESULT_AVAILABLE / RISK_DOLLAR_PATH_DECOMPOSITION_CONFIRMED / C26_NOT_PROMOTED / RUNTIME_MICROTUNING_STOPPED`。

#### 程式基準與來源

- 使用者本機執行基準：`test-branch-1_20260809_153811_a2377c1.zip`。
- SHA256：`775a804fc47b4ab5684a602ebab4e8e2a9f966b533873f9f3e20c2313669ff35`。
- Audit：`AUD-c23-c26-pit-portfolio-translation`；source=`outputs/strategy_compare/runs/20260809_150908_C23-C25-C26_6018e78bd663`。
- 期間2014-01-01～2020-12-31；只讀既有replay，不重跑portfolio、不改score／selector／training。

#### C26 vs C23：exclusive risk-dollar translation

- Exclusive selection ΔR=`+7.99R`，但exclusive ΔPnL=`-35,524.03`。
- C23-only：173 trades，ΣR=`+76.19R`、ΣPnL=`+231,681.38`、Σ implied initial risk=`854,162`、平均risk=`4,937`、risk-weighted R=`0.27R`。
- C26-only：158 trades，ΣR=`+84.18R`、ΣPnL=`+196,157.35`、Σ implied initial risk=`924,372`、平均risk=`5,850`、risk-weighted R=`0.21R`。
- Winner／loser平均risk：C23=`4,718 / 5,094`；C26=`5,674 / 5,967`。C26並非簡單整體de-risk；真正差異是實際risk dollars對不同R magnitude／交易時點的權重分布，使equal-weighted ΣR優勢反轉為較低的risk-weighted R與dollar PnL。

#### C26 vs C23：common matched trades

- Common trades=`206`，risk coverage=`100%`。
- Common ΔR約=`0.00R`，Common ΔPnL=`-41,604.23`。
- C23/C26平均implied initial risk=`9,049 / 9,117`；C26平均risk並未較低。
- 對稱分解：risk-size/path effect=`-41,612.25`；R-difference effect=`+8.02`；uncovered=`0`；residual約`0`。
- 因共同交易R幾乎完全相同，這`-41.6k`不是「同一交易被C26做壞」，而是前面portfolio path、當時equity／cash／position state與canonical sizing共同形成的trade-specific risk-dollar差。它是既有selection path的下游放大，不應另建selector或依事後R修改fixed-risk sizing。

#### C26 vs C25

- Exclusive ΔR=`+27.51R`、Exclusive ΔPnL=`+57,383.45`、All trade ΔPnL=`+43,180.29`，再次確認22日stale-score guard本身在equal-weighted R與actual dollar PnL兩層均有效。
- C25-only/C26-only risk-weighted R=`0.06R/0.13R`；guard後實際risk-dollar品質亦改善。
- Common ΔPnL=`-14,203.16`幾乎完全由risk-size/path effect=`-14,217.77`解釋；R-difference effect僅`+14.61`。

#### 最終判定

1. `SR-C26`保留為`STALE_GUARD_SUPPORTED / NOT_PROMOTED`歷史研究結果；22-calendar-day guard不得再依Selection／OOS結果調整，也不回退SR-C25。
2. C26相對C23的`+7.99R`只代表equal-weighted exclusive R改善，不能取代portfolio Return／MDD／RoMD gate；actual exclusive risk-weighted R反而`0.21R < 0.27R`，且dollar PnL為負。
3. Common trade loss已被risk-size/path effect完整解釋；因平均risk並未下降，不能簡化為「C26整體position size較小」，而是portfolio path造成的trade-specific sizing分布。這是下游結果，不構成新的可事前觀測selector假說。
4. 目前沒有清楚、可泛化、Selection盤前可觀測且不依賴未來realized outcome的mechanism支持再改fixed risk、stop distance、selector、22日guard或資金配置。為避免Selection過度微調，**C26 runtime微調鏈在此停止；不進Forward-OOS、不進參數適應、不新增SR-C27。**
5. `same-param DL selection R`後續只保留為診斷，不得單獨作promotion gate；portfolio gate仍以Return／MDD／RoMD／EV與正式wealth path為主。

#### 下一步

離開C26 runtime微調，回到尚未補齊的模型層證據：先做`MR-12B vs MR-12A` **same-fold paired Selection PIT read-only comparison**，固定相同PIT日期／共同候選／Continuous Target，直接比較每fold daily rho、pair concordance、Top-K／boundary與年度穩定性；只讀既有PIT工件，若MR-12A工件不存在才由正式模型入口依既有profile建立，不用C23/C26策略結果調模型。這一步先判定Pairwise objective在歷史PIT是否真的穩定優於MSE，再決定下一個模型研究方向；不得重啟11J counterfactual。

### 2026-08-09 — MR-13A Stage 1結果：Forward-OOS Gate通過，授權Stage 2 Selection PIT

#### 狀態

`RESULT_AVAILABLE / FORWARD_OOS_MODEL_GATE_PASS / NOT_PROMOTED_OVER_MR12B / STAGE2_AUTHORIZED`。

#### 程式基準與固定條件

- 使用者本機結果來源：`test-branch-1_20260809_184850_7662026(1).zip`；SHA256=`c66977555f0f404c36f29f46bbb38a81f4bc17de0cf8607efde3768e973f7e26`。
- Filter=`breakout_quality_v1`；Architecture=`inception_time_v1`；Profile=`daily_universal_no_time_pairwise`；Seed=42。
- Sample scope=`daily_eligible_stock_days`；Target=`daily_opportunity_no_time_r_v1`；Objective=`daily_pairwise_ranking / pairwise_logistic`；epoch metric=`mean_daily_spearman`。
- Daily universe=`1,578,349` samples／`558` tickers；feature storage仍為lazy canonical 300×10 sequence，不建立expanded daily feature artifact。
- Epoch 1為Selection內Inner Validation選模最佳：選模Validation daily rho=`0.0999`；Epoch 2=`0.0436`，故selected epoch=`1`。完整Selection refit固定1 epoch。
- 重訓後原Validation daily rho=`0.2631`只作描述；因final refit已包含原Validation rows，**不得**當作獨立選模／泛化證據。

#### Forward-OOS結果

- All eligible stock-days OOS：`608,204` groups；daily rho=`0.1755`、global rho=`0.1761`、pair concordance=`56.12%`；Top 10% Target=`1.8261R`、Bottom 10%=`0.4846R`，spread=`+1.3415R`。
- All-stock K=10：NDCG=`0.6045`、Top-K Target=`2.0375R`、Lift=`+1.0357R`、Oracle overlap=`9.07%`、Boundary=`50.99%`、Boundary gap=`+0.1035R`，competition days=`1,203`。
- Breakout-candidate OOS：`17,346` groups；daily rho=`0.1490`、global rho=`0.1894`、pair=`56.91%`；Top 10% Target=`2.3302R`、Bottom 10%=`0.5250R`。
- Breakout K=10：NDCG=`0.7201`、Top-K Target=`1.6198R`、Lift=`+0.4056R`、Oracle overlap=`61.04%`、Boundary=`54.02%`、Boundary gap=`+0.2158R`，competition days=`603`。

#### 與MR-12B anchor的同口徑判讀

- 既有MR-12B breakout Forward-OOS anchor：daily rho=`0.1568`、pair=`56.40%`、K=10 Top-K Target=`1.6231R`、Lift=`+0.4089R`、Oracle overlap=`59.80%`、Boundary=`52.98%`、Boundary gap=`+0.2121R`，competition days同為`603`。
- MR-13A相對MR-12B：daily rho=`-0.0078`；pair=`+0.51pp`；Top-K Target=`-0.0033R`；Lift=`-0.0033R`；Oracle overlap=`+1.24pp`；Boundary=`+1.04pp`；Boundary gap=`+0.0037R`。
- 因此MR-13A不是全面優於MR-12B；較合理結論是breakout實用raw Top-K品質近乎持平、pair／boundary略升，同時成功取得全市場每日排序能力。
- **NDCG比較限制**：MR-13A Stage 1先在full daily universe建立same-day percentile relevance再切breakout candidate；MR-12B在event universe建立percentile relevance。因此`0.7201 vs 0.6905`只作描述，不納入嚴格head-to-head promotion證據。Raw Target／Lift／Pair／Boundary gap／Oracle overlap不依賴此percentile normalization差異，可直接判讀。

#### 判定與下一步

1. Stage 1達成預先設定的模型層方向Gate：全市場OOS與breakout slice都呈正向排序能力，因此授權同一`MR-13A`進Stage 2 Selection PIT。
2. 不建立新MR identity；Stage 2不改target、loss、architecture或training-data semantics，只把既有PIT infrastructure泛化到daily sample provider。
3. `DL-CONT12B / MR-12B`仍為current validated continuous anchor；MR-13A尚未建立strategy runtime source，也不得因Stage 1結果切換strategy workflow。
4. Stage 2先看historical PIT all-stock primary ordering與breakout-candidate diagnostic；只有PIT Gate通過後才討論Stage 3 strategy runtime／same-parameter comparison。

### 2026-08-09 — MR-13A Stage 2結果：Selection PIT模型Gate通過

#### 狀態

`RESULT_AVAILABLE / SELECTION_PIT_MODEL_GATE_PASS / STAGE3_STRATEGY_TRANSLATION_AUTHORIZED / NOT_PROMOTED_OVER_MR12B`。

#### 程式基準與固定條件

- 使用者提供結果後的最新程式ZIP：`test-branch-1_20260809_193058_e59550d.zip`；SHA256=`ef437ca2f128321243af230743794e73d5204ca41fb186ec6929868c60fe44b8`。
- Filter=`breakout_quality_v1`；Architecture=`inception_time_v1`；Profile=`daily_universal_no_time_pairwise`；Seed=42。
- Objective=`daily_pairwise_ranking / pairwise_logistic`；Target=`daily_opportunity_no_time_r_v1`；PIT fold/validation=`12/24 months`。
- PIT score period=`2013-04-01～2020-12-31`；8 folds；本次model-audit target-valid groups=`778,532`，coverage=`100.00%`。
- 各fold selected epoch依序=`3,1,1,1,1,1,1,1`；validation daily rho依序=`0.0580,0.1125,0.1138,0.1311,0.1065,0.1167,0.1366,0.0567`。

#### PIT模型結果

- All eligible target-valid stock-days：global rho=`0.0783`、mean daily rho=`0.1111`、pair concordance=`53.84%`、Top-bottom spread=`+0.3319R`。
- 年度方向穩定：rho>0=`8/8`、spread>0=`8/8`；score-level drift=`True`只作warning，不否決方向Gate。
- Breakout-candidate slice：global rho=`0.0580`、daily rho=`0.0784`、pair=`53.98%`、Top-K Lift=`+0.1102R`、Boundary=`51.49%`。
- Daily sample沒有PASS/REJECT label，因此binary classification overlap不適用；orderable coverage留到strategy replay建立。

#### 判定

1. Stage 2 PIT模型Gate PASS：全期與每年度排序方向一致為正，且breakout candidate slice亦維持正向訊號，因此同一`MR-13A`授權進Stage 3 Selection strategy translation。
2. PIT訊號強度明顯低於Stage 1 Forward-OOS，尤其breakout slice daily rho由`0.1490`降至`0.0784`、pair由`56.91%`降至`53.98%`；因此Stage 3必須視為經濟轉化驗證，不得預設daily model已優於MR-12B。
3. `DL-CONT12B / MR-12B`仍為current validated continuous anchor；MR-13A在取得同參數Selection strategy結果前不得promotion或進ROOS。

#### Stage 3 preflight發現

在將這份PIT score接入strategy runtime前，另行檢查出兩個不影響上述model-audit數值、但會使strategy replay不合格的契約問題：

1. Stage 2 daily index把「future 40-bar target完整」同時當成score row存在條件。這對train/audit合法，但若策略把score absence當資訊，就會讓是否有score間接受未來資料完整性／停止交易影響。
2. 既有Selection PIT runtime lookup固定使用breakout `signal_date`。對MR-12B event score正確，但對MR-13A daily model會退化為「事件日算一次後沿用」，沒有真正測每日refresh語意。

因此本次`778,532`-row PIT table保留為**model-audit evidence**，但不得直接作Stage 3 strategy runtime source。Stage 3必須先完成future-independent score universe與daily information-date lookup後重建PIT Scores＋Audit。

### 2026-08-10 — MR-13A daily PIT Selection strategy translation結果：C28通過Forward-OOS Gate

#### 狀態

`RESULT_AVAILABLE / SELECTION_STRATEGY_GATE_PASS / C27_NOT_PROMOTED / C28_FORWARD_OOS_AUTHORIZED / NO_SELECTION_RETUNING`。

#### 程式基準與比較契約

- 程式基準：`test-branch-1_20260810_100406_9cdd5aa.zip`；SHA256=`6b6a9a122569ee994ff26ee3134147e221c4f020673725f5686765583aaebd62`。
- Strategy Compare run：`outputs/strategy_compare/runs/20260810_123402_C23-C24-C25-C27-C28_ffefa86f0a89`；期間=`2014-01-01～2020-12-31`、Dataset=`full`、param policy=`base-finalist-best`、max positions=`10`、rotation=`off`。
- 全部arms共用current五欄`PARAM-P2 / Min ROOS` historical active params；五欄為`high_len + atr_len + atr_buy_tol + atr_times_init + atr_times_trail`。Rules全關，C23為DL-off baseline。
- C24/C27固定minimum-repair selector；C25/C28固定feasible-ascent selector。C27/C28唯一model/source差異是`DL-CONT12B-PIT → DL-CONT13A-PIT`；C28不繼承C26 stale-score guard，也沒有新增capital objective、threshold或OOS調參。

#### 主要結果

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

#### Selection診斷與判讀

- C28的selected Target mean相對C23由`0.4769R → 0.6832R`（`+0.2063R`），但同日Target percentile由`0.5925 → 0.5260`、opportunity gap由`3.4957R → 4.1697R`惡化。這不構成矛盾：報表先在每日候選集合內計算percentile／top-k，再做daily mean；absolute Target R與within-day percentile衡量不同面向。C28不是在每一天都更接近future Target oracle，而是在不使用Future Target的前提下，選到的絕對opportunity-R與後續實際portfolio economics更好。
- `same-param selection R`是DL-on與同param/rule baseline之exclusive trades R差；C28的`+109.91R`本身不能代替portfolio PnL。但本次與舊C26不同：C28同時有Return、MDD、RoMD、Annual、EV的實際portfolio改善，因此Selection R優勢確實有轉成策略經濟效果。
- C28只有`108`個實際改單日、`12`個feasible-ascent改善日，卻產生大幅wealth-path差異；這是portfolio path敏感性的證據，不是可用來繼續調selector的理由。現有hard feasibility仍維持K/R0、reserved-capital floor，Max-DL K violation=`0`。

#### 判定與下一步

1. `SR-C27`：`NOT_PROMOTED`；不進Forward-OOS。
2. `SR-C28`：`SELECTION_GATE_PASS / FORWARD_OOS_AUTHORIZED / FROZEN_SELECTOR`。
3. `MR-13A`：Selection strategy translation已PASS，但尚不得宣告正式取代`MR-12B / DL-CONT12B`；formal anchor promotion必須等Forward-OOS strategy Gate。
4. 保留`DL-CONT13A`作MR-13A frozen Forward-OOS daily score source，保留`SR-C29`作C28語意凍結後的Forward-OOS candidate。Forward對照應只保留current五欄Min ROOS DL-off baseline與相同feasible-ascent selector的MR-12B arm，用來分離portfolio baseline與純DL source效果。
5. 不再於Selection調feasible-ascent、stale cutoff、score cutoff、capital objective、fixed risk或模型；不得把C26的22-day guard帶進C29。下一個可執行工作是2021+ Forward-OOS controlled replay。

### 2026-08-10 — Selection／Forward結果反轉後縮減為四arm核心比較

#### 狀態

`CONFIG_REDUCTION / STANDALONE_FULL_BASELINE_SUPPORTED / MR13A_ITERATIVE_RESEARCH_FOCUS / HISTORICAL_FULL_DL_ARMS_PRESERVED`。

#### 程式基準與研究結果

- 輸入基準：`test-branch-1_20260810_170110_ba0a90b.zip`；SHA256=`7464b0e2d63d80df597c75f3887931b01b9f336e4682eeb80b5c65d17dbc491d`。
- Selection 2014-01-01～2020-12-31：C23 Min baseline Return=107.98%、RoMD=5.25；C25 Min+MR-12B=101.55%/5.17；C28 Min+MR-13A=151.88%/8.05；C32 Full baseline=132.25%/8.34。歷史Full+DL結果C33=146.25%/7.73、C34=167.12%/9.06永久保留。
- Forward-OOS 2021-01-01～2025-12-22：C1 Full=123.26%/RoMD7.08；C3 Min=86.21%/6.57；C20 Min+MR-12B=172.40%/9.20；C29 Min+MR-13A=71.86%/4.57。C29-C20純DL source Return=-100.54pp、RoMD=-4.62、EV=-0.62R、same-param selection R=-204.13R。歷史Full+DL結果C30=133.21%/7.43、C31=122.00%/7.53永久保留。
- Selection支持MR-13A而Forward-OOS明顯反轉，因此MR-13A不升級；使用者仍要持續改善MR-13A，current comparison改聚焦同一Min universe下MR-12B vs MR-13A的純source差異，Full只保留完整策略體系DL-off baseline。

#### Current Strategy Compare profiles

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

#### Standalone baseline infrastructure

舊Strategy Compare engine假設每個啟用的DL-off `param_source/rule_policy` group都至少要有一個DL-on arm，因此單獨保留Full ROOS baseline會被config validator拒絕。此假設不是科學契約，只是舊controlled-pair orchestration限制。

本輪泛化為：

- DL-on arm仍必須存在同group DL-off baseline。
- DL-off arm可以作standalone comparator，不要求為了engine schema而暗中啟用DL-on arm。
- standalone baseline仍使用同一canonical replay、param/rule policy、accounting與baseline artifact schema；可從profile或legacy compatible pair重用已驗證的`no_filter` baseline，否則只執行DL-off replay一次。
- standalone baseline輸出仍包含`strategy_comparison.md/json`、年度報酬、equity、trades、capacity、orderable與selected artifacts，可直接進四arm彙總報表。
- 此改動只放寬orchestration topology，不改Full/Min params、MR-12B/MR-13A score、feasible-ascent、K/R0、Target、loss或策略accounting，不新增scientific ID。

#### 研究邊界

2021～2025 Forward-OOS已被用來判斷MR-13A失敗並形成後續改善假說，因此後續MR-13*可將它當`iterative research OOS evidence`比較泛化，但loss weight、sampling比例、epoch selection或其他超參數仍只能由Train/Validation/Selection決定；不得用該Forward期間直接挑數值。Current四arm設計的目的就是降低比較維度，持續保留Full/Min策略基準，同時把MR-13A改善的主要source Gate固定在`C28-C25`與`C29-C20`。

### 2026-08-10 — Post-fix Forward-OOS controlled replay完成：MR-12B維持anchor、MR-13A未泛化

#### 工作基準與結果來源

- 程式基準：`test-branch-1_20260810_221241_c0b9a19.zip`；SHA256 `d27645f9c3a9604d56aa32ff49734d55da2836859b243e99e1b03727ea89f4dc`。
- Selection current run：`outputs/strategy_compare/selection_pit/runs/20260810_221455_C32-C23-C25-C28_2e37f985168a/`。
- Forward current run：`outputs/strategy_compare/forward_oos/runs/20260810_221923_C1-C3-C20-C29_b0c6ee3b0b32/`。
- Forward score已先依前一輪修正重建為`feature_history_only` inference universe；本次共同策略期間完整延伸至`2026-03-02`，不再由future 40-bar Target完整性控制score presence。因此本次C20/C29結果是post-fix current evidence，舊2021～2025-12-22結果只保留PRE_FIX歷史。

#### Selection PIT固定結果

- C32 Full ROOS：Return `132.25%`、MDD `15.85%`、RoMD `8.34`、Annual `12.80%`、EV `0.71R`。
- C23 Min ROOS：`107.98% / 20.57% / 5.25 / 11.03% / 0.42R`。
- C25 Min MR-12B：`101.55% / 19.66% / 5.17 / 10.53% / 0.52R`，same-param selection R `+26.19R`。
- C28 Min MR-13A：`151.88% / 18.86% / 8.05 / 14.11% / 0.71R`，same-param selection R `+109.91R`。
- C28-C25純DL source：Return `+50.33pp`、MDD `-0.80pp`、RoMD `+2.89`、Annual `+3.58pp`、EV `+0.19R`、selection R `+83.72R`。Selection仍明確支持MR-13A daily source。

#### Forward-OOS post-fix結果

共同期間`2021-01-01～2026-03-02`：

- C1 Full ROOS：Return `129.08%`、MDD `17.41%`、RoMD `7.42`、Annual `17.43%`、EV `0.53R`。
- C3 Min ROOS：`101.60% / 13.12% / 7.74 / 14.56% / 1.02R`。
- C20 Min MR-12B：`190.26% / 18.75% / 10.15 / 22.95% / 1.17R`，same-param selection R `+35.93R`。
- C29 Min MR-13A：`83.81% / 15.71% / 5.34 / 12.53% / 0.52R`，same-param selection R `-183.42R`。
- C20-C3：Return `+88.66pp`、MDD `+5.62pp`、RoMD `+2.41`、Annual `+8.39pp`、EV `+0.15R`、selection R `+35.93R`。
- C29-C3：Return `-17.79pp`、MDD `+2.58pp`、RoMD `-2.41`、Annual `-2.03pp`、EV `-0.51R`、selection R `-183.42R`。
- C29-C20純DL source：Return `-106.45pp`、MDD `-3.04pp`、RoMD `-4.81`、Annual `-10.42pp`、EV `-0.65R`、selection R `-219.35R`。

#### 判讀

1. MR-13A的Selection優勢在post-fix Forward策略層完全反轉；因score universe已修正並跑到2026-03-02，此反轉不得再歸因於舊future-target coverage bug。
2. C20與C29使用相同current Min ROOS、all-off rules、K/R0與frozen feasible-ascent。C29 selector診斷仍達`230`個Max-DL介入日、`217`個repair日、`20`個feasible-ascent改善日、`230`個1-swap local optimum日且resource/K violation為0；因此目前主要失敗點是**MR-13A score對實際策略邊界選擇的Forward泛化**，不是selector hard-feasibility或資源契約。
3. C29相對C3的勝率由`40.70%`升至`43.87%`，但Payoff由`3.17`降至`2.45`、EV由`1.02R`降至`0.52R`，且same-param selection R為`-183.42R`。這支持「daily score能提高部分命中率，但把basket membership推向較差的R/payoff tail」的診斷方向。
4. MR-12B/C20在相同post-fix期間仍有正selection R與更高RoMD，因此維持current Forward strategy anchor；MR-13A標記`FORWARD_OOS_STRATEGY_FAIL / NOT_PROMOTED`，但daily-universal方向本身仍可作後續研究，不回退或覆寫MR-13A identity。
5. 下一個立即可執行步驟先跑既有`Multiple-seed robustness`，判斷C20/C29差異是systematic model-semantic差異或seed variance；不得挑best seed或做ensemble。若MR-13A跨seed仍普遍負selection R／低RoMD，再做changed-order／swap-boundary attribution與Selection→Forward分布漂移診斷，才建立下一個新的`MR-*`假設。

#### 本輪完整檢查追加修正：Robustness摘要reference改為config-driven

- 發現`strategy_multi_seed_robustness.py`計算RoMD「勝Min／勝Full」時以顯示名稱字串`Min ROOS`／`Full ROOS`尋找fixed baseline。這違反專案「摘要對象由config／registry／active settings驅動」契約；改名或切換profile時可能得到錯誤reference。
- `config/strategy_compare.py`的`STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS`新增`romd_reference_baselines`，以`param_source + rule_policy`描述Min／Full語意reference，不另列active C-ID，也不依賴display name。
- `StrategyMultiSeedRobustnessSettings`新增typed mapping與唯一匹配validator；robustness contract保存resolved reference arm identity，summary只依contract的arm_id取baseline。Robustness schema bump `1 → 2`，只影響尚未產生結果的robustness fingerprint，不影響Selection／Forward pair cache或策略回放語意。
- synthetic contract同步改為讀取目前config的profile、seed count／generator seed、arm names與reference semantics，不再把`forward_oos`、8 seeds或目前四個顯示名稱當成唯一合法設定。
- Scientific identity：不新增MR／DL／SR ID，不改model、Target、loss、score、selector、K/R0、交易會計或本次C20/C29結果。

### 2026-08-11 — Multiple-seed Forward robustness首批結果：單一seed結論重開、報表matched-pair契約補齊

#### 結果來源與固定條件

- 程式基準：`test-branch-1_20260811_000339_d277205.zip`；SHA256 `e49854513aa4b2607a6f8c49a112d1aa1892953ba899ca8693a98bb5347916cc`。
- 使用者提供首批Forward-OOS Multiple-seed console aggregate；目前config為8個deterministic generated seeds、profile=`forward_oos`、同一Min ROOS／all-off rules／frozen feasible-ascent、MR-12B與MR-13A各8個isolated training→strategy replay observations；Full／Min為固定baseline。
- Canonical一般模型流程仍使用`BREAKOUT_QUALITY_RANDOM_SEED=42`；本次multi-seed generator產生的8個seed不包含42，因此本結果是獨立的seed robustness evidence，不是把seed 42重複納入平均。

#### 首批平均策略績效

- Full ROOS：Return `129.08%`、MDD `17.41%`、RoMD `7.42`、Annual `17.43%`、EV `0.53R`。
- Min ROOS：`101.60% / 13.12% / 7.74 / 14.56% / 1.02R`。
- Min MR-12B（N=8）：Return Mean `129.01%`、MDD `15.87%`、RoMD `8.21`、Annual `17.15%`、EV `0.77R`、Payoff `3.12`、Exposure `92.76%`、Trades `318.6`、Win `42.61%`、Monthly Win `61.31%`、Log R² `0.8772`。
- Min MR-13A（N=8）：Return Mean `148.18%`、MDD `15.47%`、RoMD `9.97`、Annual `19.02%`、EV `0.94R`、Payoff `3.19`、Exposure `92.83%`、Trades `316.0`、Win `43.66%`、Monthly Win `59.33%`、Log R² `0.8830`。
- MR-13A − MR-12B的平均差：Return `+19.17pp`、MDD `-0.40pp`、RoMD `+1.76`、Annual `+1.87pp`、EV `+0.17R`、Payoff `+0.07`、Win `+1.05pp`；Monthly Win反而`-1.98pp`。

#### RoMD分布與研究判讀

- MR-12B：Mean `8.21`、Median `7.70`、Std `2.63`、CV `0.32`、Min/P25/P75/Max=`3.94/6.76/10.77/11.51`、勝Min=`4/8`、勝Full=`4/8`。
- MR-13A：Mean `9.97`、Median `10.87`、Std `3.68`、CV `0.37`、Min/P25/P75/Max=`4.38/6.84/12.92/14.49`、勝Min=`5/8`、勝Full=`5/8`。
- MR-13A的Mean與Median都高於MR-12B，因此不是單一極佳seed單獨把平均拉高；但Std/CV亦更高，顯示daily-universal ranker對training seed更敏感，lower-tail仍明顯存在。
- 這與canonical seed 42的post-fix Forward結果（MR-12B RoMD `10.15`、MR-13A `5.34`）方向相反。結論因此由「MR-13A Forward模型語意失敗」修正為：**seed 42失敗是有效單一run證據，但不足以代表跨seed模型優劣；seed variance本身已是主要研究變數之一。**
- MR-13A仍不升格。下一個判讀必須優先看相同seed下的MR-13A−MR-12B matched-pair RoMD，而不是只比較兩個獨立分布平均或任意cross-seed pairs；不得挑best seed、做seed ensemble或用Forward結果調training hyperparameter。

#### 首批報表缺陷與修正

1. `DL選擇R Mean`在兩個stochastic arms顯示`-`。Root cause是robustness worker只讀`trade_attribution.json`，但score-ranking策略不產生該hard-filter attribution檔；正常Strategy Compare其實已有從`no_filter_trades.csv`與active trades重建`exclusive_selection_delta_r`的canonical SSOT。
2. robustness worker改為直接重用Strategy Compare既有`_load_direct_selection_r()`，因此score-ranking與hard-filter均使用同一直接選擇R定義，不新增第二套統計口徑。
3. 首批報表只有RoMD勝Min／Full與Markdown內任意cross-seed distribution probability，沒有回答「同一training seed下MR-13A是否優於MR-12B」；schema `2 → 3`新增`romd_same_seed_comparison`，輸出matched N、雙方勝數、tie與ΔRoMD Mean/Median/Std/Min/P25/P75/Max，console與Markdown都顯示。
4. schema bump只為完成robustness持久結果／報表契約；不改seed generator、Target、architecture、loss、training defaults、Forward period、selector、K/R0、策略accounting或任何scientific variable。因v2首批seed_results沒有可回復的直接選擇R、且暫存replay/training預設已清除，完整v3 `DL選擇R Mean`需要由同一正式入口重新跑8×2 isolated observations；不得由現有aggregate猜值。

#### 狀態

- Multiple-seed robustness：`RESULT_AVAILABLE / REPORT_SCHEMA_V3_IMPLEMENTED / MATCHED_PAIR_RECHECK_REQUIRED`。
- MR-12B：維持canonical seed42 runtime anchor，但`MULTI_SEED_SUPERIORITY_NOT_ESTABLISHED`。
- MR-13A：`SINGLE_SEED_FORWARD_FAIL / MULTI_SEED_REOPENED / NOT_PROMOTED`。

### 2026-08-12 — Forward-OOS 8-seed robustness完成：MR-13A平均優勢成立但matched-pair仍不足，補同seedΔDL選擇R報表

#### 結果來源與固定條件

- 程式基準：`test-branch-1_20260812_164416_3e2e5e0.zip`；SHA256 `0bcc8e29fa2fee65ef7afb07c662073cae5055e347e70e23f65b553d4cbbd064`。
- 使用者由`apps/research.py` → Strategy Compare → Forward-OOS Multi-seed robustness正式完成：`seed_count=8`、`seed_generator_seed=20260810`、GPU training workers=2、CPU strategy replay workers=1；共同期間`2021-01-01～2026-03-02`。
- Full／Min baseline直接重用既有current Forward-OOS正式工件；MR-12B／MR-13A各8個deterministic generated seeds均使用isolated canonical trainer + same frozen feasible-ascent strategy replay。
- Dataset、Label、Target、architecture、loss、training sample definition、epoch-selection、Min ROOS參數、selector、K/R0、sizing、accounting、max positions=10、rotation=off皆固定；本輪不挑best seed、不做ensemble、不依Forward結果調hyperparameter。

#### 8-seed正式aggregate

- Full ROOS：Return `129.08%`、MDD `17.41%`、RoMD `7.42`、Annual `17.43%`、EV `0.53R`。
- Min ROOS：Return `101.60%`、MDD `13.12%`、RoMD `7.74`、Annual `14.56%`、EV `1.02R`。
- Min MR-12B：Return Mean `129.01%`、MDD Mean `15.87%`、RoMD Mean/Median `8.21/7.70`、Annual Mean `17.15%`、EV Mean `0.77R`、Payoff `3.12`、Win `42.61%`、Monthly Win `61.31%`、Log R² `0.8772`、DL選擇R Mean `-105.34R`。
- Min MR-13A：Return Mean `148.18%`、MDD Mean `15.47%`、RoMD Mean/Median `9.97/10.87`、Annual Mean `19.02%`、EV Mean `0.94R`、Payoff `3.19`、Win `43.66%`、Monthly Win `59.33%`、Log R² `0.8830`、DL選擇R Mean `-54.70R`。
- MR-13A − MR-12B的aggregate mean差為Return `+19.17pp`、MDD `-0.40pp`、RoMD `+1.76`、Annual `+1.87pp`、EV `+0.17R`、Payoff `+0.07`、Win `+1.05pp`、DL選擇R Mean差 `+50.64R`；Monthly Win為`-1.98pp`。

#### 同seed RoMD與年度穩定性

- MR-12B RoMD完整分布：Mean `8.21`、Median `7.70`、Std `2.63`、CV `0.32`、Min/P25/P75/Max=`3.94/6.76/10.77/11.51`，勝Min/Full=`4/8`、`4/8`。
- MR-13A：Mean `9.97`、Median `10.87`、Std `3.68`、CV `0.37`、Min/P25/P75/Max=`4.38/6.84/12.92/14.49`，勝Min/Full=`5/8`、`5/8`。
- 同seed MR-13A − MR-12B RoMD：4勝4敗、0 tie；ΔRoMD Mean/Median/Std=`+1.76/+1.48/4.47`。因此MR-13A的平均與中位數優勢不是單一best seed造成，但優勢仍未形成一致seed方向，且seed dispersion高於MR-12B。
- 年度MR-13A相對MR-12B的Mean ΔReturn：2021 `+1.86pp`、2022 `+0.12pp`、2023 `+4.27pp`、2024 `+3.82pp`、2025 `+0.88pp`、2026 YTD `-0.61pp`；同seed右勝左依序`6/8, 5/8, 5/8, 7/8, 5/8, 3/8`。改善不是只集中單一年份，但2026 YTD已反轉。

#### 判定

1. canonical seed 42的MR-13A Forward失敗仍是有效單一run證據，但本次8-seed結果再次確認不能把seed 42外推成模型普遍失敗。
2. MR-13A在Return／MDD／RoMD／Annual／EV／Payoff／Win Rate與DL選擇R的cross-seed Mean均優於MR-12B，但RoMD同seed只有4勝4敗，故目前只可判定`AVERAGE_EDGE / MATCHED_ROMD_INCONCLUSIVE`，不得升格current anchor。
3. 兩個stochastic arms的DL選擇R Mean都仍為負值（MR-12B `-105.34R`、MR-13A `-54.70R`）；MR-13A相對MR-12B的Mean改善`+50.64R`尚不能回答是否在相同training seed下穩定改善。這是下一個判讀所缺的唯一主要統計，不應再啟動新的模型訓練或擴seed。
4. 若同seedΔDL選擇R多數為正、但ΔRoMD仍4/8附近，下一步應做portfolio translation attribution；若同seedΔDL選擇R本身亦正負混雜，則問題主要仍在training-seed ranking stability，才設計下一個MR假設。

#### Report schema v5實作

- `filters/breakout_quality/strategy_multi_seed_robustness.py`新增泛化same-seed metric comparison helper；RoMD與DL選擇R共用同一pivot／distribution邏輯，不複製統計實作。
- `direct_selection_r`只有兩個stochastic arms的`param_source`與`rule_policy`完全相同時才允許matched comparison，遵守既有same-param attribution universe契約。
- `robustness_summary.json`新增`direct_selection_r_same_seed_comparison`；console與Markdown新增「DL選擇R同seed配對比較」，輸出雙方勝數、tie與Δ Mean/Median/Std/Min/P25/P75/Max。
- report schema `4 → 5`；scientific fingerprint不包含report schema，因此不觸發模型重訓或strategy replay。
- 「查看最新報表」若偵測既有summary為舊schema且同run `seed_results.csv`仍在，會只用永久raw seed aggregate重建衍生matched-pair欄位與Markdown；不需要重跑8×2 observations，也不讀已清除checkpoint／score／replay details。
- direct synthetic contract同步覆蓋DL選擇R matched-pair；formal double check仍由使用者本機`apps/run_bundle.py`完成，GPT不執行formal suite。

#### 狀態與下一步

- `MR-13A`：`RESULT_AVAILABLE / MULTI_SEED_REOPENED / AVERAGE_EDGE / MATCHED_ROMD_INCONCLUSIVE / MATCHED_SELECTION_R_PENDING / NOT_PROMOTED`。
- `MR-12B / DL-CONT12B`：維持current canonical runtime anchor，`MULTI_SEED_SUPERIORITY_NOT_ESTABLISHED`。
- 下一個正式操作不需再訓練：套用schema v5後由`Research → 策略組合比較 → Forward-OOS Multi-seed robustness → 查看最新報表`，讓既有`seed_results.csv`直接補出同seedΔDL選擇R；取得數值後再決定portfolio translation Audit或下一個MR。

### 2026-08-12 — Forward robustness schema v6：ranking edge只部分轉成portfolio結果

#### 結果來源與固定條件

- 最新程式基準：`test-branch-1_20260812_175546_c5b1e25.zip`；SHA256 `17bd88c37e775c53cf59601f10f49eda5a0e27095f19d62757606055272093a2`。
- 使用者由既有Forward robustness scientific fingerprint=`2f70dbe73dcf3fed`、8 deterministic generated seeds（generator_seed=`20260810`）透過report schema v6查看同seedranking→strategy轉化；本輪報表只讀既有`seed_results.csv`，沒有model training或strategy replay。
- Dataset、Target、architecture、loss、epoch-selection、Min ROOS參數、feasible-ascent selector、K/R0、max positions、rotation、sizing與accounting均未改變；不得依本結果挑best seed、做ensemble或回頭調training semantics。

#### schema v6同seed轉化結果

- `ΔDL選擇R = MR-13A − MR-12B`為正的seeds=`6/8`；其中`ΔRoMD>0`只有`3/6 (50.0%)`。
- ranking與RoMD方向一致=`4/8`，方向相反=`4/8`。描述性Spearman：`rho(ΔDL選擇R, ΔRoMD)=0.476`、`rho(ΔDL選擇R, ΔReturn)=0.333`。
- Discordant rows：S1 `+11.59R / ΔReturn -11.58pp / ΔMDD -0.95pp / ΔRoMD -0.31`；S7 `+35.01R / +24.21pp / +2.59pp / -0.33`；S8 `+74.70R / -50.34pp / -4.40pp / -0.21`，三者皆ranking↑但RoMD↓。S6則`-41.02R / +53.14pp / -1.22pp / +4.66`，為ranking↓但RoMD↑。
- 因此不能再把問題簡化為ranking training-seed stability：MR-13A ranking edge在多數seed成立，但equal-weighted selection R只部分映射到最終Return/MDD/RoMD；S1/S8顯示主要Return path反向，S7顯示MDD path足以吃掉正Return差。

#### 判定與下一步

1. `MR-13A`更新為`MATCHED_SELECTION_R_EDGE / MATCHED_ROMD_INCONCLUSIVE / RANKING_TO_PORTFOLIO_PARTIAL_TRANSLATION / TRADE_PATH_ATTRIBUTION_REQUIRED / NOT_PROMOTED`。
2. `MR-12B / DL-CONT12B`繼續作current canonical runtime anchor；schema v6的Spearman與seed quadrant只是歸因證據，不是promotion gate。
3. 下一步新增planned read-only `AUD-forward-robustness-portfolio-translation`，必須涵蓋全部8個matched seeds，逐seed與aggregate拆common/exclusive active trades、realized R、dollar PnL、risk-dollar sizing、slot/capital geometry、equity/MDD wealth path；不得只挑discordant或best/worst seeds。
4. 目前robustness成功run依retention policy已清除per-seed replay details，因此`seed_results.csv`不足以推導交易層歸因；後續若實作Audit，應由同一scientific contract重建**compact attribution source**並在數值落盤後清除完整checkpoint/score/replay detail，避免為Audit永久保留整棵暫存工件。
5. 在portfolio attribution完成前，不擴到16 seeds、不跑Selection PIT multi-seed、不開新MR、不調MR-13A hyperparameter／loss／selector。

### 2026-08-12 — Forward robustness portfolio Audit完成：平均translation為正但S1/S8存在R→Dollar反轉

- 最新程式基準：`test-branch-1_20260812_205417_4f33303.zip`；SHA256 `123445f876f307e7d942ca3afa7c7cbab9d54a25a7219387e63b0a0008c57874`。使用者直接重跑既有`AUD-forward-robustness-portfolio-translation`；scientific fingerprint=`2f70dbe73dcf3fed`、8 deterministic generated seeds、generator_seed=`20260810`與16/16 VERIFIED compact source不變；Audit只讀既有工件，不train、不replay。
- 全8-seed結果：Ranking↑=`6/8`；Ranking↑且RoMD↑=`3/6`；ranking/RoMD同方向=`4/8`。Direct-pair Exclusive ΔR Mean=`+50.64R`、Selection basis gap Mean=`+0.00R`、Exclusive ΔPnL Mean=`+173,208.15`、Common risk-size effect Mean=`+18,544.82`、ΔGap slot-days Mean=`-99.00`。
- Driver分布：`TRANSLATION_SUCCESS=3`（S2/S3/S4）、`TRADE_SET_DOLLAR_TRANSLATION=2`（S1/S8）、`MDD_WEALTH_PATH=1`（S7）、`PORTFOLIO_OVERRIDE=1`（S6）、`ALIGNED_DECLINE=1`（S5）。
- Discordant evidence：S1 direct-pair Exclusive ΔR=`+11.59R`但Exclusive ΔPnL=`-103,422`、Common sizing ΔPnL=`-12,360`、ΔGap slot-days=`+101`；S8=`+74.70R/-434,639/-68,710/+210`。S7雖Exclusive ΔPnL=`+164,262`、ΔReturn=`+24.21pp`，但ΔMDD=`+2.59pp`使ΔRoMD=`-0.33`；S6則ranking=`-41.02R`卻ΔReturn=`+53.14pp`、ΔRoMD=`+4.66`。
- 判定：MR-13A的ranking／direct-pair trade-set edge在平均上已取得經濟支持，且平均exclusive PnL、common sizing與slot-gap均偏向MR-13A；但seed-level portfolio translation仍高度heterogeneous，因此`MR-13A`維持`NOT_PROMOTED`，`MR-12B / DL-CONT12B`繼續作current runtime anchor。現階段不得再改MR-13A target／loss／architecture／hyperparameter，也不擴seed、不跑Selection PIT multi-seed。
- 下一步不建立新Audit或新模型：現有canonical attribution primitive已計算candidate-only／comparator-only的`risk_weighted_r`、`total/avg implied initial risk`、`winner/loser avg implied initial risk`，只是schema v2報表未顯示。Audit report schema v3只將這些既有欄位拉到`Exclusive trade R→Dollar bridge`，並保存`exclusive_risk_weighted_r_gap`與`exclusive_total_implied_risk_delta`；scientific fingerprint不變。套用後只需重跑同一read-only Audit，直接判定S1/S8是risk weighting、總risk budget scale或trade-set quality造成，不需任何training/replay。

### 2026-08-12 — Forward robustness portfolio Audit schema v3：R→Dollar bridge確認為seed-specific risk allocation heterogeneity

- 最新程式基準：`test-branch-1_20260812_211440_34a43ba.zip`；SHA256 `4602f49a7546f307ed3b4150c6c2011fedb498f27d60118300f75a0ede8b7c5f`。使用者重跑既有`AUD-forward-robustness-portfolio-translation` schema v3；scientific fingerprint=`2f70dbe73dcf3fed`、seed_count=8、generator_seed=`20260810`、16/16 VERIFIED compact source不變；Audit只讀既有工件，不train、不replay。
- 既有aggregate維持：Ranking↑=`6/8`、Ranking↑且RoMD↑=`3/6`、Direct-pair Exclusive ΔR Mean=`+50.64R`、Exclusive ΔPnL Mean=`+173,208.15`、Common risk-size effect Mean=`+18,544.82`、ΔGap slot-days Mean=`-99.00`。
- schema v3新增的exclusive risk bridge顯示：S1 MR-13A-only／MR-12B-only `ΣR=143.33/131.74R`，但`RW-R=0.476/0.547R`、Avg risk=`6,684/6,792`，故Exclusive ΔPnL=`-103,422`；S8 `ΣR=286.63/211.94R`，但`RW-R=0.821/0.900R`、Avg risk=`6,290/7,699`，Exclusive ΔPnL=`-434,639`。反例S6雖`ΣR=187.79/228.80R`，但`RW-R=0.832/0.655R`、Avg risk=`8,152/6,066`，Exclusive ΔPnL=`+520,269`。
- 8-seed aggregate `Exclusive RW-R gap Mean=+0.121R`，因此MR-13A並不存在「平均risk weighting必然較差」的證據；問題是seed-specific R×risk配置異質性。現階段不得直接新增score-weighted sizing、risk multiplier或其他SR規則，否則會把diagnostic observation直接變成OOS-tuned策略。
- 下一步沿用同一`AUD-forward-robustness-portfolio-translation`，report schema v4只做read-only exact decomposition，不新增Audit／SR／MR identity：對每seed的exclusive trades，以兩arm所有可識別initial-risk交易的pooled mean risk作reference，將covered Exclusive ΔPnL精確拆成 `(1) equal-risk selection effect = q_ref × ΔΣR`、`(2) average-risk scale effect`、`(3) within-set R×risk weighting effect`；risk不可由PnL/R識別的交易另列`uncovered ΔPnL`。四項加總必須閉合actual Exclusive ΔPnL，residual fail-fast／報表顯示。
- 判讀順序預先固定：若S1/S8主要負在average-risk scale，才研究capital/risk-budget路徑；若主要負在within-set weighting，才研究為何canonical ATR/cash sizing與realized R呈不利配置；若equal-risk selection本身轉負，則回到trade-set economics。schema v4仍不得作promotion gate或回流training semantics，且不需重跑robustness。

### 2026-08-12 — Forward robustness portfolio Audit schema v4：within-set risk weighting為主要edge稀釋來源；實作entry-execution binding attribution

- 最新程式基準：`test-branch-1_20260812_213111_826092c.zip`；SHA256 `e17d9e35648a67bfcb15bbfe417dd0302ab92c564d928f8a6c5568dd80719798`。使用者重跑既有`AUD-forward-robustness-portfolio-translation` schema v4；scientific fingerprint=`2f70dbe73dcf3fed`、seed_count=8、generator_seed=`20260810`與原16/16 scientific observations不變；Audit只讀既有compact source，不train、不replay。
- schema v4 exact decomposition完全閉合：8-seed Exclusive ΔPnL Mean=`+173,208.15` = equal-risk selection effect Mean=`+355,214.76` + average-risk scale effect Mean=`+14,995.80` + within-set R×risk weighting effect Mean=`-197,002.42` + uncovered=`0`；residual Mean=`0.000000`。因此MR-13A的equal-risk trade-set edge在平均上明確為正，但約55%的selection effect被within-set risk weighting吃掉，主未解問題已從模型ranking移到實際risk-dollar配置。
- seed-level機制：S1 equal-risk selection=`+78,069`、scale=`-14,774`、within-set=`-166,717`，actual Exclusive ΔPnL=`-103,422`；S8=`+523,425/-352,399/-605,665`→`-434,639`；S3雖selection=`+1,528,579`且最終PnL=`+733,325`，within-set仍=`-1,012,222`；S6 selection=`-289,839`，卻由scale=`+432,811`與within-set=`+377,298`反轉為`+520,269`。這些結果不支持直接調高／調低全域fixed risk，也不支持把Forward diagnostic直接變成新的SR規則。
- 下一步固定為同一Audit的Risk Utilization / Binding Attribution，不新增MR／SR／Audit ID。為取得正式entry execution evidence，`core/portfolio_entries.py`只在呼叫者提供sidecar時記錄每筆entry execution的candidate/chosen/filled qty、sizing capital、effective entry budget、available cash、max_qty、limit/stop、實際initial risk與execution order；不改候選排序、qty決策、成交、帳務或正式runtime semantics。
- `strategy_compare_diagnostics._flatten_entry_execution_rows()`只用canonical `calc_position_size()`做counterfactual：原始risk＋position cap、移除risk cap、移除position cap、同時移除兩者，再依正式MaxQty與board-lot policy標記`RISK_CAP / POSITION_CAP / RISK_POSITION_TIE / CAPITAL / MAX_QTY / LOT_ROUNDING / ENTRY_BUDGET_CASH`；risk budget固定由該筆`candidate_sizing_capital × fixed_risk`，actual utilization使用正式position保存的`initial_risk_total_milli`，不得用Audit另造近似風險公式。
- Multiple-seed compact attribution schema `1 → 2`：永久source在既有`trades/equity/daily_capacity/selected_buys`外增加`execution.csv.gz`；仍不保存full scores、orderable universe、checkpoint或完整replay tree。舊schema-v1 16 units因此視為Audit evidence缺件，必須由同一Forward robustness入口以原fingerprint／原8 seeds重建；原`seed_results.csv` scientific observations仍是truth，rebuild後照既有selected epoch／fold count／策略metrics／年度報酬／可得hash驗證，只有完全一致才標`VERIFIED`。
- Audit result schema `4 → 5`：全seed exclusive closed trades依project-wide canonical actual-trade key join到filled entry execution sidecar，輸出pre-cash candidate risk utilization→actual risk utilization、winner-minus-loser utilization，以及cash／risk cap／position cap／capital／MaxQty／lot binding counts與binding signatures；缺任何exclusive trade execution attribution直接fail-fast。execution sidecar另保留`execution_order`，canonical occurrence identity依正式執行順序而非signal date或報表排序建立。
- Scientific status不變：MR-13A維持model-research edge supported但`NOT_PROMOTED`；MR-12B／DL-CONT12B繼續作current runtime anchor。只有schema v5 binding結果能決定是否存在可泛化的portfolio allocation mechanism；結果前不得直接新增score-weighted sizing、risk multiplier、position-cap調整或其他SR-C變更。
- 使用者操作：套用本批後先由`Research → 策略組合比較 → Forward-OOS Multi-seed robustness → 執行`重建compact schema-v2的16個execution evidence units；完成且16/16 VERIFIED後，才由`Research → Audit／診斷 → 執行目前 Audit 設定`讀取schema v5。這次重建不是新scientific experiment，且不得刪除原`outputs/strategy_compare/robustness/2f70dbe73dcf3fed/`。

### 2026-08-12 — Forward robustness portfolio Audit schema v5結果：未找到可泛化的MR-13A-specific sizing／binding機制

- 最新程式基準：`test-branch-1_20260812_222249_c1cf30b.zip`；SHA256 `b6588961ca026f2e476b19f9cfe40133f9b878f78ff06893308d63796f08c1d8`。使用者以原Forward robustness scientific fingerprint=`2f70dbe73dcf3fed`、seed_count=8、generator_seed=`20260810`重建compact attribution schema v2並執行既有`AUD-forward-robustness-portfolio-translation` schema v5；16/16 scientific observations與既有結果一致，Audit只讀compact source，不train、不replay。
- 既有model/trade-set證據不變：MR-13A vs MR-12B同seedΔDL選擇R=`6/8`勝、Mean=`+50.64R`；RoMD=`4/8`勝、Mean=`+1.76`；Direct-pair Exclusive ΔR Mean=`+50.64R`；Exclusive ΔPnL Mean=`+173,208.15`。schema v4 exact bridge仍為equal-risk selection effect Mean=`+355,214.76`、average-risk scale=`+14,995.80`、within-set weighting=`-197,002.42`、residual=`0`。
- schema v5 entry-execution結果：Actual risk-util gap Mean（MR-13A-only − MR-12B-only）=`-1.41pp`；MR-13A-only Winner−Loser actual utilization Mean=`+0.03pp`，MR-12B-only=`-0.64pp`。Binding totals（13A/12B）：Cash=`851/856`、Position cap=`286/284`、Risk cap=`1360/1383`、Risk/Position tie=`0/0`、Capital=`0/0`、MaxQty=`0/0`、Lot rounding=`1360/1328`。
- Seed-level actual utilization仍異質：S1=`47.8%/45.7%`、S2=`42.1%/47.4%`、S3=`50.5%/47.3%`、S4=`45.8%/47.7%`、S5=`44.7%/48.2%`、S6=`52.0%/43.3%`、S7=`44.1%/51.2%`、S8=`41.8%/49.2%`（13A-only/12B-only）。S1/S8的negative dollar translation沒有對應到跨seed一致的cash/risk/position/lot binding差異；S6則以較高actual utilization伴隨positive portfolio override，證明risk allocation能影響結果，但沒有單一constraint可從8-seed evidence直接泛化成策略規則。
- 判定：`AUD-forward-robustness-portfolio-translation = RESULT_AVAILABLE / NO_GENERALIZABLE_BINDING_DIFFERENTIAL`。MR-13A的model/selection edge維持supported，但portfolio RoMD robustness仍不一致，因此`MR-13A = NOT_PROMOTED`、`DL-CONT12B / MR-12B = CURRENT_RUNTIME_ANCHOR`。不得從本Audit直接新增score-weighted sizing、risk multiplier、position-cap或cash-allocation SR；否則等同把Forward OOS attribution結果回寫成tuned策略。
- 下一步解除先前「暫不跑Selection PIT multi-seed」限制：使用既有`Selection PIT Multi-seed robustness`正式workflow，固定預設`seed_count=8`、`generator_seed=20260810`，相同scientific condition下比較MR-12B／MR-13A。目的不是修Forward portfolio，而是確認MR-13A的model/selection edge是否跨Selection與Forward兩個時段都具seed robustness。結果前不擴16 seeds、不開MR-14、不新增SR-C35，也不改MR-13A training semantics。

### 2026-08-13 — Selection PIT 8-seed robustness完成：MR-13A relative edge跨時段反轉，維持MR-12B runtime anchor

- 最新使用者程式基準：`test-branch-1_20260813_085538_7794659(2).zip`；SHA256 `f10206e125b6e1eb1ddeafff04366ea0a878365878ce708cd335cd338d941cf0`。Selection PIT robustness scientific condition固定為2014-01-01～2020-12-31、`seed_count=8`、`generator_seed=20260810`、MR-12B／MR-13A各7 PIT folds/seed、同一Min feasible-ascent strategy replay。使用者先前完成的16個training artifacts全部成功REUSE；isolated PIT score cache修正後MR-13A replay降至約01:42～01:50，完整16/16 observations在約20分鐘replay階段完成。
- Selection PIT 8-seed aggregate：MR-12B Return/MDD/RoMD=`117.08%/19.85%/5.96`、DL選擇R Mean=`64.08R`；MR-13A=`102.98%/20.51%/5.03`、DL選擇R Mean=`43.32R`。Matched MR-13A−MR-12B：ΔRoMD=`-0.93`、Median=`-0.96`、Std=`1.94`，MR-13A勝`2/8`；ΔDL選擇R=`-20.76R`、Median=`-16.59R`、Std=`35.37R`，MR-13A勝`3/8`。Ranking/RoMD方向一致=`7/8`，Spearman(ΔDL選擇R,ΔRoMD)=`0.381`、Spearman(ΔDL選擇R,ΔReturn)=`0.452`。
- 年度strategy return差異（MR-13A − MR-12B）：2014 `+0.06pp`、2015 `-2.41pp`、2016 `-0.27pp`、2017 `-7.48pp`（0/8 seeds勝）、2018 `+2.28pp`（7/8勝）、2019 `-0.25pp`、2020 `+0.18pp`。目前最明顯負向regime集中於2017，但年度Return本身不足以證明ranking edge翻轉原因，不能直接依2017調training或strategy。
- 與既有Forward 8-seed形成正式cross-period reversal：Forward 2021-01-01～2026-03-02中MR-13A−MR-12B ΔDL選擇R Mean=`+50.64R`、6勝2敗，ΔRoMD Mean=`+1.76`、4勝4敗；Selection PIT則分別=`-20.76R`／3勝5敗與`-0.93`／2勝6敗。故MR-13A的daily-universal relative ranking/selection edge沒有跨Selection與Forward兩時段穩定成立。
- 機制判定也與Forward不同：Forward主要是ranking edge平均為正、但portfolio translation heterogeneous；Selection PIT則ranking/RoMD方向一致7/8，且ranking edge本身平均為負。因此Selection時段不應再優先研究sizing／capital translation；核心問題回到daily-universal vs event-based training universe在不同歷史regime下的relative ranking泛化。
- Scientific status：`MR-13A = RESULT_AVAILABLE / CROSS_PERIOD_MULTI_SEED_EDGE_NOT_STABLE / NOT_PROMOTED`；`DL-CONT12B / MR-12B = CURRENT_RUNTIME_ANCHOR`。既有MR-13A單模型Gate與單seedC28/C29結果保留為有效歷史觀察，但不得再外推為跨seed／跨period superiority。不得挑best seed、不得以單一年度或Forward Audit直接調loss／Target／architecture／hyperparameter，也不新增sizing SR。
- 下一步先使用兩邊既有robustness permanent artifacts做read-only cross-period／year-regime attribution：以同一8 generated seeds比較每period的matched ΔDL選擇R、ΔReturn、ΔMDD、ΔRoMD，並由compact attribution source拆出年度／regime trade-set realized-R差異；目的只定位edge flip是否集中於特定年份／市場狀態。只有在可重現的regime mechanism被確認後，才評估是否建立新的`MR-*` training-semantics實驗；目前不直接開MR-14。

### 2026-08-13 — Cross-period / Year-regime Audit完成：Selection負edge廣泛分布，下一步改測MR-12B daily-inference bridge

- 程式基準：`test-branch-1_20260813_123854_7984fda.zip`；SHA256 `1cf5e4eb06b08833b0d578a51ba7dfb4bcada9b99c1de49e080331d78ae4e4bf`。正式`AUD-cross-period-year-regime-attribution`使用Selection fingerprint=`bc496e102dd5348e`、Forward fingerprint=`2f70dbe73dcf3fed`、同一8 generated seeds與`seed_generator_seed=20260810`，全程read-only，不train／score／replay。
- Period aggregate再次確認relative ranking edge方向反轉：Selection PIT MR-13A−MR-12B `ΔDL選擇R Mean=-20.76R`、`ΔRoMD Mean=-0.93`；Forward-OOS為`+50.64R`與`+1.76`。Classification=`RANKING_EDGE_DIRECTION_REVERSAL`。
- 逐年度direct-pair trade-set證據排除「單一2017 regime造成全部失敗」：Selection 2014～2020 Exclusive ΔR符號為5負2正；2014=`-13.96R`、2015=`+1.72R`、2016=`-1.53R`、2017=`-7.71R`、2018=`+10.51R`、2019=`-1.85R`、2020=`-7.93R`。Worst year改為2014，且只占Selection負年度R的`42.3%`；Forward 2021～2026則為5正1負。故Selection弱勢是廣泛period-level instability，不應建立year gate或只針對2017調模型。
- 程式語意核對：MR-12B與MR-13A同用`ARCH-inception_time_v1`、300×10 stock+0050 normalized OHLCV、40-bar adverse-first no-time opportunity公式、within-day RankNet pairwise logistic loss與`mean_daily_spearman` epoch selection；`continuous_target.py`明確定義daily target future-path mechanics與MR-12 no-time target一致。MR-13A相對MR-12B的核心scientific變更是training sample universe由`breakout_event_groups`擴為`daily_eligible_stock_days`，而策略端同時取得daily score refresh。
- 因此下一個最乾淨的causal test不直接建立新`MR-*`：先用**既有MR-12B權重做daily-universal inference bridge**，令training semantics完全維持MR-12B，只把score universe／refresh timing改成與MR-13A相同的feature-eligible daily stock-days。這屬新的`DL-*` score source／`SR-C*` runtime usage研究，而不是新模型。
- 預定判讀：若MR-12B daily-inference bridge在Selection與Forward都維持MR-12B的relative robustness，則daily scoring本身可行，MR-13A問題主要來自universal training-pair dilution；若bridge本身也顯著惡化，才表示event-trained模型對off-event daily inference不具穩定可移植性，之後才值得建立新的candidate-aware daily training `MR-*`。在bridge前不改Target／loss／architecture／hyperparameter，不新增year/regime gate或sizing SR。

### 2026-08-13 — MR-13B Forward Model Gate：Target-gap weighting明確失敗，停止於模型層

- 本機正式模型結果來源：`test-branch-1_20260813_170038_f694db6.zip`；SHA256 `ec3405bbe3ab896f90a0eb050446027ca9687223ab5aa75cd4571491588709b3`
- Profile=`daily_universal_no_time_pairwise_gap_weighted`；MR=`MR-13B`；sample scope=`daily_eligible_stock_days`；Target=`daily_opportunity_no_time_r_v1`；Architecture=`inception_time_v1`；Seed=`42`。MR-13A的Dataset／Target／300×10 input／optimizer／inner-validation／selected-epoch final refit與`mean_daily_spearman`選模全部固定，唯一scientific change仍是同日RankNet pair依daily percentile target gap加權。
- Epoch選擇：Validation daily rho由epoch1=`0.0744`升到epoch2=`0.0771`後，epoch3降到`0.0364`；selected epoch=`2`。完整Selection refit固定2 epochs。重訓後原Validation rho=`0.2840`因原Validation rows已進final refit，只作描述，不是獨立泛化證據。
- Forward all-stock OOS：groups=`608,204`；daily rho=`0.0303`、global rho=`-0.0089`、pair=`51.02%`；Top10% Target=`1.0155R`、Bottom10%=`1.0074R`，spread只剩`+0.0081R`；NDCG@10=`0.5293`、Top-K Target=`1.0386R`、Lift=`+0.0368R`、Oracle overlap=`1.89%`、Boundary=`50.08%`、Boundary gap=`-0.0039R`。
- Breakout-candidate OOS：groups=`17,346`；daily rho=`0.0091`、global rho=`-0.0160`、pair=`51.34%`；Top10% Target=`1.1692R`、Bottom10%=`1.2242R`，spread反轉為`-0.0550R`；NDCG@10=`0.6568`、Top-K Target=`1.2616R`、Lift=`+0.0474R`、Oracle overlap=`54.31%`、Boundary=`51.18%`、Boundary gap=`+0.0491R`。
- 與MR-13A同口徑Forward模型結果比較：MR-13A all-stock daily/global rho=`0.1755/0.1761`、pair=`56.12%`、Top-bottom spread=`+1.3415R`、Top-K Lift=`+1.0357R`；breakout daily/global rho=`0.1490/0.1894`、pair=`56.91%`、Top-bottom spread=`+1.8052R`、Top-K Lift=`+0.4056R`。MR-13B在主要排序證據全面大幅退步，因此不是「略遜」而是Forward model Gate明確失敗。
- 判定：`MR-13B = RESULT_AVAILABLE / REJECTED_AT_FORWARD_MODEL_GATE / NO_PIT / NO_RUNTIME_DL_SOURCE`。不得建立Selection PIT scores、`DL-CONT13B`或策略arms，也不做gap threshold／gap exponent／weight floor等同機制微調，避免沿已失敗的target-gap weighting方向追參數。
- 機制解讀：weighted RankNet讓訓練loss快速下降，但沒有把跨期排序帶到Forward；這與「大量近似pair稀釋 supervision」假說不一致。更合理的是extreme target-distance pairs提高了period-specific／tail ordering的擬合，造成Validation弱、OOS接近隨機且candidate top-bottom反轉。
- 下一個單一變更候選：保留Daily Universal sample、同一strategy-agnostic target與architecture，改為**直接回歸同日daily target percentile的MSE**，也就是Daily Universal Percentile Regression。此objective已存在continuous-ranker canonical training API，可避免O(n²) pair weighting與pair-gap設計，並讓model score直接逼近0～1 cross-sectional opportunity percentile。若正式開始，應建立新的獨立`MR-13C`；在使用者確認前不先建立profile或runtime source。

### 2026-08-13 — MR-13C Forward Model Gate：Daily Spearman改善、Top-tail集中度下降，授權Selection PIT跨期驗證

- 本機模型結果來源：`test-branch-1_20260813_181447_e5e7812.zip`；SHA256 `c67d92341eedc0f4e98eca8dc97191df3fb63c33fb5b85c5f4fdcb3dfb063ec1`。本次Forward結果屬迭代研究OOS證據，不宣稱untouched holdout。
- Profile=`daily_universal_no_time_percentile_mse`；MR=`MR-13C`；sample scope=`daily_eligible_stock_days`；Target=`daily_opportunity_no_time_r_v1`；Architecture=`inception_time_v1`；Seed=`42`。相較MR-13A唯一scientific change仍是training objective由same-date RankNet pairwise logistic改為直接回歸same-date target percentile的MSE。
- Epoch選擇：epoch1 Validation daily rho=`0.0492`為最佳；epoch2=`-0.0341`，故selected epoch=`1`。完整Selection refit固定1 epoch。重訓後原Validation daily rho=`0.2687`因原Validation rows已進final refit，只作描述，不是獨立泛化證據。
- Forward all-stock OOS：groups=`608,204`；daily rho=`0.1877`、global rho=`0.1239`、pair=`56.38%`；Top10% Target=`1.2072R`、Bottom10%=`0.5186R`，spread=`+0.6886R`；NDCG@10=`0.5569`、Top-K Target=`1.3888R`、Lift=`+0.3870R`、Oracle overlap=`3.60%`、Boundary=`48.28%`、Boundary gap=`-0.0746R`。
- Breakout-candidate OOS：groups=`17,346`；daily rho=`0.1565`、global rho=`0.1509`、pair=`56.69%`；Top10% Target=`1.8536R`、Bottom10%=`0.4386R`，spread=`+1.4150R`；NDCG@10=`0.7067`、Top-K Target=`1.5877R`、Lift=`+0.3735R`、Oracle overlap=`59.95%`、Boundary=`52.69%`、Boundary gap=`+0.2172R`。
- 與MR-13A同口徑Forward比較：all-stock daily rho `+0.0122`、pair `+0.26pp`，但global rho `-0.0522`、Top-bottom spread `-0.6529R`、Top-K Lift `-0.6487R`、Boundary gap `-0.1781R`；breakout daily rho `+0.0075`，但global rho `-0.0385`、pair `-0.22pp`、Top-bottom spread `-0.3902R`、Top-K Lift `-0.0321R`，Boundary gap則`+0.0014R`近乎持平。
- 判定：`MR-13C = RESULT_AVAILABLE / FORWARD_OOS_MODEL_GATE_PASS / SELECTION_PIT_MODEL_GATE_AUTHORIZED / NO_RUNTIME_DL_SOURCE`。這不是MR-13C全面勝過MR-13A；較準確結論是MSE讓平均daily ordering略改善、但top-tail concentration明顯變弱。由於MR-13A真正的主要缺陷是Selection 2014-2020與Forward 2021-2026跨期方向反轉，MR-13C現在最有資訊價值的下一步是**Selection PIT模型Gate**，確認percentile regression是否改善跨期穩定性。
- 下一步只執行`Research → 模型訓練 → 建立／更新 Selection PIT Scores → PIT模型驗證`；不建立`DL-CONT13C`、策略arm或multi-seed robustness。若Selection PIT仍出現廣泛弱排序，MR-13C停止於model layer；若PIT跨期明顯改善，再討論runtime source與後續8-seed robustness。

### 2026-08-13 — MR-13D Forward Model Gate：全市場Top-tail回升、整體rho略退，授權Selection PIT

- 本機模型結果來源：`test-branch-1_20260813_190826_32aa156.zip`；SHA256 `dfced3322b0a1d41d315949e4562ba2be6bfad4ebf2e2451e3131f7501e98708`。本次Forward結果屬迭代研究OOS證據。
- Profile=`daily_universal_no_time_upper_tail_pairwise`；MR=`MR-13D`；sample scope=`daily_eligible_stock_days`；Target=`daily_opportunity_no_time_r_v1`；Architecture=`inception_time_v1`；Seed=`42`。相較MR-13A唯一scientific change仍為同日RankNet pair以兩端daily target percentile算術平均作upper-tail relevance權重。
- Epoch選擇：epoch1 Validation daily rho=`0.1184`為最佳；epoch2=`0.0877`，故selected epoch=`1`。完整Selection refit固定1 epoch。重訓後原Validation daily rho=`0.2805`只作描述，不是獨立泛化證據。
- Forward all-stock OOS：groups=`608,204`；daily/global rho=`0.1690/0.1453`、pair=`55.81%`；Top10% Target=`1.6701R`、Bottom10%=`0.5016R`，spread=`+1.1685R`；NDCG@10=`0.5862`、Top-K Target=`1.8004R`、Lift=`+0.7986R`、Oracle overlap=`7.46%`、Boundary=`49.30%`、Boundary gap=`+0.0283R`。
- Breakout-candidate OOS：groups=`17,346`；daily/global rho=`0.1529/0.1576`、pair=`55.69%`；Top10% Target=`1.8422R`、Bottom10%=`0.5372R`，spread=`+1.3050R`；NDCG@10=`0.7086`、Top-K Target=`1.5440R`、Lift=`+0.3298R`、Oracle overlap=`58.84%`、Boundary=`50.74%`、Boundary gap=`+0.0307R`。
- 與MR-13C比較：all-stock daily rho `-0.0187`、global rho `+0.0214`、pair `-0.57pp`、Top-bottom spread `+0.4799R`、Top-K Lift `+0.4116R`、Boundary gap `+0.1029R`。Breakout daily rho `-0.0036`、global rho `+0.0067`、pair `-1.00pp`、Top-bottom spread `-0.1100R`、Top-K Lift `-0.0437R`、Boundary gap `-0.1865R`。
- 判定：`MR-13D = RESULT_AVAILABLE / FORWARD_OOS_MODEL_GATE_PASS / SELECTION_PIT_MODEL_GATE_AUTHORIZED / NO_RUNTIME_DL_SOURCE`。13D沒有像13B失效，且明確修復MR-13C部分all-stock Top-tail弱點，但不是全面優於13C；兩者形成「整體daily ordering vs all-stock top-tail」互補trade-off，因此不在Forward先二選一。
- 使用者進一步指出實際buy list只會使用榜首少數股票，整體rank好不代表前段rank好。故在進Selection PIT前再做最後一個MR-13E，直接研究**目前預測榜單位置對錯排重要性**，而不是再做target relevance或一般full-list平均排序。

### 2026-08-13 — MR-13E Forward Model Gate結果與13C/13D/13E Selection PIT batch

- 本輪結果基準：`test-branch-1_20260813_200804_5faabbe.zip`；SHA256 `3f8072aacae27a4bbe17da286bb995aec67d61f99807551781c2b8ae830eac6a`
- MR-13E Seed42完成Forward Model Gate：selected epoch=`1`；all-stock OOS daily/global rho=`0.2151/0.1042`、pair=`57.42%`、Top10%-Bottom10% target spread=`+0.8234R`、Top-K Lift=`+0.9951R`、Boundary gap=`+0.0376R`；breakout-candidate OOS daily/global rho=`0.1721/0.1535`、pair=`57.43%`、Top10%-Bottom10% target spread=`+1.4294R`、Top-K Lift=`+0.4489R`、Boundary gap=`+0.1961R`。
- 相較MR-13C，13E all-stock daily rho `+0.0274`、Top-K Lift `+0.6081R`；相較MR-13D，daily rho `+0.0461`、Top-K Lift `+0.1965R`。相較MR-13A，daily rho `+0.0396`、pair `+1.30pp`、breakout daily rho `+0.0231`、breakout Top-K Lift `+0.0433R`，但all-stock global rho與Boundary gap仍較低。因此13E不是Forward全面dominance，但為目前13系列最均衡候選。
- 判定：`RESULT_AVAILABLE / FORWARD_OOS_MODEL_GATE_PASS / SELECTION_PIT_MODEL_GATE_AUTHORIZED / NO_RUNTIME_DL_SOURCE`。不得因單一Seed42 Forward結果建立`DL-CONT13E`或策略arm；下一步與已通過Forward Gate的MR-13C、MR-13D一起做Selection PIT Model Gate，直接檢查MR-13A曾出現的Selection↔Forward方向反轉是否改善。
- 新增config-driven PIT Gate batch：`MR-13C / MR-13D / MR-13E`必須使用同一filter、architecture、daily Target、sample scope、Seed、PIT start/end、fold months與inner-validation months。選單一次確認後依序呼叫canonical `build-point-in-time-scores`與`audit-point-in-time-scores`；三模型各自獨立model/PIT工件，不共享weights。
- Batch結果只並列canonical audit的All Daily rho／Global rho／Pair／Top-K Lift／Boundary gap與breakout diagnostic；不建立人工加權總分、不依結果挑seed、不回流training semantics。PIT結果前runtime anchor仍為`MR-12B / DL-CONT12B`。

### 2026-08-13 — MR-13C／13D／13E Selection PIT Model Gate結果：MR-13E取得跨期單seed模型層優先候選

- 結果程式基準：`test-branch-1_20260813_202833_b72d444.zip`；SHA256 `6891174c512f7579ebf25cf4c01992f10e6c5bc560ce1a1446e4e00d0cbc3676`。三模型使用完全相同的filter=`breakout_quality_v1`、architecture=`inception_time_v1`、Target=`daily_opportunity_no_time_r_v1`、sample scope=`daily_eligible_stock_days`、Seed=`42`、PIT period=`2013-04-01～2020-12-31`、fold=`12` months、inner validation=`24` months；各自獨立訓練與PIT工件。
- Coverage：三者皆`778,532/778,532 = 100%`、8 folds。MR-13C PIT all-stock daily/global rho=`0.0841/0.0463`、pair=`53.15%`、Top-K Lift=`+0.1550R`、Boundary gap=`+0.0858R`；年度rho/spread=`7/8, 7/8`正；breakout daily rho=`0.0773`、Top-K Lift=`+0.1139R`。Gate PASS。
- MR-13D PIT all-stock daily/global rho=`0.0842/0.0630`、pair=`52.85%`、Top-K Lift=`+0.1776R`、Boundary gap=`-0.0093R`；年度rho/spread=`7/8, 6/8`正；breakout daily rho=`0.0233`、Top-K Lift=`+0.0657R`。Gate PASS。
- MR-13E PIT all-stock daily/global rho=`0.1627/0.0533`、pair=`55.75%`、Top-bottom spread=`+0.2813R`、Top-K Lift=`+0.4456R`、Boundary gap=`-0.0043R`；年度rho/spread皆`8/8`正。Breakout global/daily rho=`0.0833/0.1019`、pair=`54.77%`、Top-K Lift=`+0.1513R`。Gate PASS。
- Shared PIT比較中，13E的all-stock Daily rho比13C／13D高約`+0.0786/+0.0785`，Pair高`+2.60pp/+2.90pp`，Top-K Lift高`+0.2906R/+0.2680R`；breakout Daily rho也最高。13D的all-stock global rho略高，13C的Boundary gap較佳，但兩者沒有抵銷13E在主要daily ranking、pair、top-tail與breakout slice的整體優勢。
- Cross-period：13E Forward all-stock daily rho=`0.2151`、Top-K Lift=`+0.9951R`，Selection PIT則為`0.1627`、`+0.4456R`；breakout daily rho由Selection `0.1019`到Forward `0.1721`，兩時段皆維持正方向。相較MR-13A過去Selection relative edge為負、Forward為正的方向反轉，13E在模型層沒有重現方向翻轉。這仍是Seed42 model-level evidence，不代表已通過multi-seed strategy robustness。
- `drift=True`不是PIT Gate veto：canonical gate只要求primary global rho>0、daily rho>0、過半年度rho>0、過半年度Top-bottom spread>0；drift只是相鄰fold score mean shift的診斷warning。13E因8/8年度rho與spread皆正，Gate依正式contract為PASS。
- 判定：`MR-13E = RESULT_AVAILABLE / FORWARD_OOS_MODEL_GATE_PASS / SELECTION_PIT_MODEL_GATE_PASS / CROSS_PERIOD_SINGLE_SEED_MODEL_STABILITY_SUPPORTED / STRATEGY_GATE_AUTHORIZED / NO_RUNTIME_DL_SOURCE`。MR-13C與MR-13D也通過絕對PIT Gate，但本輪不各自建立runtime source；下一步只對13E建立Selection PIT與Forward runtime source identity，與MR-12B在相同selector／策略參數下做source-only controlled strategy compare。策略結果前不得promotion，runtime anchor仍為`MR-12B / DL-CONT12B`。

### 2026-08-13 — MR-13E single-seed strategy Gate結果與orderable／feasible alignment Audit

- Audit實作程式基準：`test-branch-1_20260813_230500_c0bbfde.zip`；SHA256 `95cfe895c1d57f61716bd05f02b7b4774e6585fb562703bee52f85b2fa56862b`。本輪只新增read-only Audit／source resolver／synthetic contract與結果文件同步，不改trainer、score source、selector、portfolio engine、sizing或execution。
- 策略Gate結果來自同一source-only controlled compare：Selection PIT固定historical Min params／all-off／feasible-ascent，Forward固定current Min params／all-off／feasible-ascent；MR-12B、MR-13A、MR-13E唯一差異為DL score source。`MR-12B / DL-CONT12B`仍為CURRENT_RUNTIME_ANCHOR。
- Selection 2014-01-01～2020-12-31：C25 MR-12B Return/MDD/RoMD/EV/DL選擇R=`101.55%/19.66%/5.17/0.52R/+26.19R`；C28 MR-13A=`151.88%/18.86%/8.05/0.71R/+109.91R`；C35 MR-13E=`153.96%/19.88%/7.74/0.54R/+35.96R`。13E相對12B Return=`+52.41pp`、RoMD=`+2.58`、DL選擇R=`+9.77R`，Selection Gate PASS；但相對13A Return僅`+2.08pp`，RoMD=`-0.31`、EV=`-0.17R`、DL選擇R=`-73.96R`，不能宣稱13E全面優於13A。
- Forward 2021-01-01～2026-03-02：C20 MR-12B=`190.26%/18.75%/10.15/1.17R/+35.93R`；C29 MR-13A=`83.81%/15.71%/5.34/0.52R/-183.42R`；C36 MR-13E=`128.77%/17.18%/7.49/0.87R/-66.22R`。13E相對13A Return/RoMD/EV/DL選擇R=`+44.96pp/+2.16/+0.35R/+117.20R`，但相對12B=`-61.49pp/-2.65/-0.30R/-102.16R`，故Forward Gate FAIL vs MR-12B，13E不promotion且不直接加入8-seed robustness。
- Resource-aware diagnostics顯示raw score basket通常不能直接落地：Selection C35 DL days=`274`、direct-score-feasible=`13`、repair=`261`、feasible-ascent改善=`17`；Forward C36 DL days=`242`、direct=`16`、repair=`226`、ascent改善=`21`。12B/13A也使用完全相同selector，因此上述現象本身不證明selector造成13E失敗，但顯示全市場model Top-K不能直接代表實際buy-list competition quality。
- 下一步新增正式read-only `AUD-mr13e-orderable-feasible-alignment`，不train、不score、不replay。Selection讀C25/C28/C35、Forward讀C20/C29/C36完成的Strategy Compare `orderable_candidates / daily_capacity / execution / strategy_comparison`工件；共同Future Target固定用`daily_opportunity_no_time_r_v1`，以trade_date之前最新已完成交易日join，且只在replay完成後加入Audit，Future Target不得進runtime decision。
- Ranking層每天只看**實際orderable competition set**，計算Daily Spearman、pair accuracy、Top-N target lift、oracle retention與top-vs-rest pair accuracy；N直接取當日`Resource_Aware_Pre_Market_Order_Limit`，缺時fallback baseline selected count，不寫固定K=10。分層固定為all／direct-feasible／repair／feasible-ascent。
- Translation層以`score_ranking_execution.csv`的`chosen_qty>0`當實際action candidates；每天用實際action count作N，比較raw score Top-N與actual action set的overlap、Action−Raw target R與Action−Raw score，並同樣拆direct／repair／ascent。既有sidecar未永久保存minimum-repair seed與每一步swap membership，因此Audit只定位raw orderable ranking vs resource/action translation，不宣稱逐swap counterfactual identity。
- 自動classification只用方向比較，不建立人工加權總分：若13E Forward all/repair orderable Top-N lift都低於12B則`ORDERABLE_RANKING_ALIGNMENT_GAP`；若orderable ranking不弱但repair Action−Raw target較差則`RESOURCE_TRANSLATION_GAP`；其他為`MIXED_ORDERABLE_AND_TRANSLATION`。

### 2026-08-14 — MR-13E orderable／feasible alignment結果：Forward raw edge存在，斷點定位為resource translation

- 使用者先依正式Strategy Compare artifact contract refresh C25/C28/C35與C20/C29/C36，使canonical `score_ranking_execution.csv`齊備；`AUD-mr13e-orderable-feasible-alignment`之後由BLOCKED轉READY並完整執行。Audit全程只讀completed Strategy Compare artifacts，不train、不score、不replay；共同Future Target固定`daily_opportunity_no_time_r_v1`，只在replay後依trade_date之前一已完成交易日join。
- Selection PIT orderable set：C25/MR-12B Daily rho/Pair/Top-N Lift=`0.0994/53.41%/+0.5826R`，repair rho/Lift=`0.1182/+0.6423R`；C28/MR-13A=`0.0636/52.07%/+0.2661R`，repair=`0.0679/+0.2899R`；C35/MR-13E=`0.1261/54.39%/+0.2989R`，repair=`0.1246/+0.2710R`。因此13E在Selection的broad ranking勝12B，但真正最前端Top-N與repair-day lift仍低於12B，Selection仍有top-tail cross-period stability問題。
- Forward-OOS orderable set：C20/MR-12B Daily rho/Pair/Top-N Lift=`0.1586/55.52%/+0.8223R`，repair rho/Lift=`0.1596/+0.7975R`；C29/MR-13A=`0.1351/54.83%/+1.0243R`，repair=`0.1639/+1.0734R`；C36/MR-13E=`0.1698/55.93%/+1.0996R`，repair=`0.1699/+1.1401R`。13E相對12B Forward Top-N Lift=`+0.2773R`、repair-day Lift=`+0.3426R`，因此「13E只會排全市場、進真正breakout/orderable pool就失效」假說被否定。
- 真正Forward斷點出現在raw→action translation：Raw→Action overlap C20/C29/C36=`0.1742/0.1493/0.1309`；Action−Raw Target=`-0.8091R/-1.2096R/-1.1685R`；repair-day Action−Raw=`-0.9459R/-1.3807R/-1.3132R`。13E相對12B在repair-day多損失`-0.3673R`，即使raw orderable ranking更強，轉成實際entry action時保留的Future Target更少。
- 正式classification=`RESOURCE_TRANSLATION_GAP`。但既有工件只保存raw orderable、aggregate repair/ascent diagnostics與final execution action，沒有永久保存`minimum-repair seed`與每一步accepted swap membership，因此不能把整段translation gap直接歸罪於feasible-ascent；目前只可定位為raw ranking到resource/action basket之間的缺口。
- Research policy維持：MR-13E不promotion、不進8-seed、不開始13F；下一步先精確拆selector transformation stage。只有定位minimum-repair／ascent／action／fill何者造成13E相對12B額外損失後，才討論是否需要改selector或model training semantics。

### 2026-08-14 — MR-13E selector-stage Audit結果：主要額外損失精確定位Raw→minimum-repair

- 使用者完成Selection／Forward Strategy Compare replay refresh並執行`AUD-mr13e-selector-stage-translation`。本次結果對照的策略績效與既有C25/C28/C35、C20/C29/C36完全一致，確認新增trace沒有改變selector／sizing／execution決策。
- Selection repair days：C25/MR-12B Raw→Repair=`-0.9358R`、Repair→Ascent=`+0.0230R`；C28/MR-13A=`-0.6188R/+0.0093R`；C35/MR-13E=`-0.6420R/+0.0316R`。13E Selection Raw→Repair反而比12B少損失`+0.2938R`，因此不能把minimum-repair一般化判定為對13E永久不利。
- Forward repair days：C20/MR-12B Raw→Repair=`-0.9292R`、Repair→Ascent=`+0.0250R`、Ascent→Action=`0`、Action→Fill=`+0.0103R`；C29/MR-13A=`-1.4038R/+0.0206R/0/-0.0022R`；C36/MR-13E=`-1.3763R/+0.0073R/0/+0.0085R`。
- 13E−12B Forward stage差：Raw→Repair=`-0.4471R`、Repair→Ascent=`-0.0177R`、Ascent→Action=`0`、Action→Fill=`-0.0018R`，正式classification=`DOMINANT_EXTRA_LOSS_AT_RAW_TO_REPAIR`。主要extra translation loss在minimum-repair之前／之中完成；feasible-ascent、entry action與actual fill不是主要斷點。
- Membership亦支持同一判讀：Forward Raw/Repair overlap C20/C29/C36=`3.36%/2.40%/2.46%`，Repair/Ascent overlap=`94.84%/95.72%/96.20%`，Ascent/Action=`100%`。Raw Top-N會被minimum-repair大幅重組，而其後ascent只微調、entry action不再改membership。
- Research policy：不直接拿掉feasible-ascent，也不因Forward結果立刻改minimum-repair。Selection同stage方向相反，下一步必須先回答13E Forward `-1.3763R` Raw→Repair損失是K/R0 resource constraints不可避免，還是greedy repair path／one-swap local optimum未找到更高frozen-score合法basket。

### 2026-08-14 — MR-13F Forward Model Gate result + Direct-R Calibration Audit

- 結果基準：`test-branch-1_20260814_223551_df27d70.zip`；SHA256 `3ba0ee2d93997c7a012ccca4fbc1e517bea6a01b7b8c2770fe35166a3015df0b`。MR-13F Seed42 Forward training已完成；沒有建立Selection PIT、runtime DL source或Strategy arm。
- Forward結果：selected epoch=`2`；OOS Huber=`0.7888`、MAE=`1.1547R`、bias=`-0.6505R`；Daily Spearman=`0.1611`、Global Spearman=`0.1310`、Pair=`55.52%`；Top 10% Target=`1.2593R`、Bottom 10%=`0.6711R`，Top-K Lift=`+0.7392R`。Breakout candidate slice Daily rho=`0.0919`、Pair=`54.80%`、Top-K Lift=`+0.2472R`、boundary gap=`-0.0258R`。
- 判讀：MR-13F不是無訊號模型，但pure ranking弱於MR-13E；更重要的是direct-R用途需要輸出數值本身具有R magnitude語意，而目前`-0.6505R` bias與`1.1547R` MAE尚不足以證明。研究狀態因此為`HOLD_FOR_DIRECT_R_CALIBRATION_AUDIT`，不是PASS/REJECT；不先建PIT。
- 新`AUD-mr13f-direct-r-calibration`是最小必要read-only證據。Source固定讀canonical MR-13F Forward `continuous_ranker_report.json`與`daily_ranker_oos_scores.csv.gz`，驗證report/profile/research ID、OOS-unused-for-training及score artifact SHA；不train、不rescore、不建PIT、不replay。
- Baseline不偷看OOS：使用report已保存的Selection-inner Validation `target_r_mean`作frozen constant R prediction；OOS只post-hoc比較MR-13F與constant的Huber/MAE/RMSE。另輸出`Actual R = intercept + slope × Predicted R`的descriptive OLS、Predicted-R equal-count quantile buckets的Actual mean／positive rate與bucket Spearman、以及`Predicted R > 0`/`<= 0`兩組的Actual mean R／positive rate。
- Audit不設定`abs(slope-1)<X`、bucket lift門檻或其他magic threshold；classification只依model vs frozen constant的Huber/MAE方向與slope/bucket Spearman正負提供描述性方向。OOS fitted intercept/slope明確標為`posthoc_fitted_calibration_runtime_eligible=false`，不得直接用作後續策略calibration；若Audit支持再另做Selection/PIT-only calibration controlled experiment。
- 舊`AUD-mr13e-minimum-repair-mechanism`在結果取得前已被C37 controlled decision supersede。C37 Selection明確REJECT後，repair mechanism attribution不再會改變current MR-13F下一步，依最小必要證據與disposable lifecycle標記`NOT_EXECUTED / SUPERSEDED_BY_C37_DECISION`並退役formal handler/dedicated synthetic；production minimum-repair、repair-search certificate sidecar與generic Strategy Compare contracts不刪。

### 2026-08-15 — MR-13F Calibration Audit result → MR-13G Direct-R Mean Regression

- 決策證據：`AUD-mr13f-direct-r-calibration`已由使用者正式執行；OOS evaluable rows=`608,204`。MR-13F Predicted/Actual mean=`0.3548/1.0052R`，pre-OOS frozen constant為Selection-inner Validation target mean=`1.0314R`。
- Magnitude comparison：MR-13F Huber/MAE/RMSE=`0.7888/1.1547/2.0039R`，frozen constant=`0.6901/1.1005/1.7956R`；Model全部較差。MR-13F bias=`-0.6505R`，constant bias=`+0.0261R`。
- Post-hoc calibration structure：`Actual R = 0.9304R + 0.2110 × Predicted R`，R²=`0.0088`；Predicted-R bucket Actual-mean Spearman=`0.9879`，相鄰bucket上升=`8/9`。Predicted R `>0`組Actual mean=`1.1563R`，`<=0`組=`0.7430R`。正式classification=`RANKING_STRUCTURE_WITHOUT_MAGNITUDE_ERROR_GAIN`、Magnitude=`FROZEN_CONSTANT_DOMINATES_MODEL`、Calibration structure=`POSITIVE_CALIBRATION_STRUCTURE`。
- MR-13F判定：`REJECTED_AT_DIRECT_R_MAGNITUDE_GATE / NO_PIT / NO_RUNTIME_DL_SOURCE`。模型仍有relative ranking structure，但Huber輸出沒有打贏最簡單的pre-OOS constant magnitude baseline，因此不值得建立Selection PIT，也不把OOS fitted intercept/slope回灌runtime。
- Audit lifecycle：`AUD-mr13f-direct-r-calibration`待決策問題已結案；formal config/catalog/implementation與dedicated synthetic依disposable lifecycle退役，結果只保留於Registry／Log與既有`outputs/audit`工件。
- 新實驗：`MR-13G / daily_universal_no_time_r_mse`，狀態=`IMPLEMENTED / AWAITING_FORWARD_MODEL_RESULT`。程式基準=`test-branch-1_20260815_000304_0e651bc.zip`，SHA256=`38c28940b90ca9a5a9d8fc08b0fa8fe37a5059499aff5937d0e6c569c73901c4` + 本輪patch。
- 唯一scientific change：固定MR-13F的`daily_eligible_stock_days`、`daily_opportunity_no_time_r_v1`、`ARCH-inception_time_v1`、Adam/LR/weight decay、Seed42、Selection-inner Validation／Forward split、two-logit `PASS−REJECT` Predicted-R output、無target clipping、無sample weighting；只把loss由`huber_raw_r(delta=1R)`改為`mse_raw_r`。Epoch selection同步由Validation Huber最小值改成Validation raw-R MSE最小值，同值仍用Validation Daily Spearman tie-break。
- 研究理由：MSE的population optimum對應conditional mean `E[R|X]`，與後續可能的`Predicted/Expected R × planned initial risk` portfolio objective在數學上更直接一致；本輪不加入planned risk、K/R0、breakout membership或任何portfolio資訊進模型。
- Dataset／Target重建：不需要；沿用canonical daily-universal source與既有raw-R target。
- 下一步：只執行MR-13G Seed42 Forward Model Gate。第一優先比較raw-R MSE/RMSE、MAE、bias與pre-OOS frozen constant的magnitude error；ranking Daily Spearman／Pair／Top-K作必要secondary signal。若magnitude仍無gain則直接REJECT，不建PIT；若成立才另輪進Selection PIT。

### 2026-08-15 — MR-13G Forward REJECT → SR-C39 Relative Excess-Alpha Selection experiment

- MR-13G正式Forward結果：`daily_universal_no_time_r_mse`，Seed42，selected epoch=`1`。Epoch1 Train/Val MSE=`1.275490/4.578933`；Epoch2 Train/Val MSE=`0.751347/6.958489`，故依Validation MSE正確停在epoch1，第二epoch已明顯overfit。
- Forward OOS：MSE=`4.5923`、RMSE=`2.1430R`、MAE=`1.4311R`、bias=`+0.3390R`；Daily/Global rho=`0.1259/0.0569`、Pair=`54.25%`、Top10/Bottom10 Target=`1.0709/0.9719R`、spread=`+0.0990R`、Top-K Lift=`+0.5117R`、boundary gap=`-0.0065R`。Breakout slice Daily/Global rho=`0.1079/0.0206`、Pair=`53.90%`、Top-K Lift=`+0.1842R`。
- 判定：`MR-13G REJECTED_AT_DIRECT_R_MAGNITUDE_GATE / NO_PIT / NO_RUNTIME_DL_SOURCE`。相對MR-13F，MSE雖把bias由`-0.6505R`改善為`+0.3390R`，但RMSE=`2.1430R`與MAE=`1.4311R`更差，且仍輸給MR-13F Calibration Audit事前固定的pre-OOS constant RMSE=`1.7956R`；ranking亦退化。故「13F只是Huber壓縮magnitude」假說被否定，direct raw-R regression路線在13G停止，不再以另一個regression loss延伸。
- Current model-research entry回到frozen `MR-13E / daily_universal_no_time_full_list_ndcg_pairwise`作relative-ranking reference；這不是runtime promotion，current runtime anchor仍`MR-12B`。
- 新策略實驗：`SR-C39 / Min MR-13E Excess-Alpha`，程式基準=`test-branch-1_20260815_003742_8db650c.zip`，SHA256=`7ba87eb72c49ca7537136664d1f5370fa55bad48b5c1e6b578995edc4ad55628` + 本輪patch。
- Controlled change：C39完全重用C35的`DL-CONT13E-PIT`、historical Min params、K/R0、sizing、cash、orderability與execution；唯一portfolio objective由`Σ frozen score`改為`Σ(Expected Excess-R × canonical planned initial risk)`。不重訓MR-13E、不改target、不改K/R0、不以negative excess減少order count。
- Relative target：先在完整canonical `daily_eligible_target_valid` universe按日計算`daily mean raw R`，再定義`target_excess_r = raw R - same-day mean raw R`；共同absolute-R market baseline因此不會被planned risk放大。MR-13E score只用來取得同日daily percentile，不參與target mean。
- Calibration：Selection逐calendar strategy year expanding fit；每個cutoff只能用`date < cutoff`且`label_eval_end_date < cutoff`的成熟rows。Mapping=`daily percentile -> Expected Excess-R`的nondecreasing weighted PAVA isotonic curve；無人工Top-X bucket、slope/intercept、lambda、Forward target或OOS fit。
- Strategy Gate：只新增C39 Selection arm與C39-C35／C39-C23／C39-C25 contrasts，robustness=`off`。本輪**不建立C40**；只有C39 Selection結果支持relative alpha×capacity假說後，才另輪占新的Forward SR-C ID。

### 2026-08-15 — SR-C39 Selection result → SR-C40 Excess-Alpha No-R0 ablation

- SR-C39正式Selection PIT結果已完成，run=`20260815_014807_C32-C23-C25-C28-C35-C39_1f8085b42d35`。C39 frozen重用MR-13E PIT與PIT isotonic Expected Excess-R calibration，保留C35相同historical Min params、K/R0、sizing、cash、orderability與execution，只把basket objective改為`Σ(Expected Excess-R × canonical planned initial risk)`。
- C39主要結果：Return=`131.44%`、MDD=`20.21%`、RoMD=`6.50`、Annual=`12.74%`、EV=`0.58R`、DL selection R=`+51.02R`；C35為Return=`153.96%`、MDD=`19.88%`、RoMD=`7.74`、EV=`0.54R`、DL selection R=`+35.96R`。C39−C35=`-22.52pp Return / +0.32pp MDD / -1.24 RoMD / +0.03R EV / +15.06R DL selection R`。因此C39不能取代C35，也不得建立Forward arm。
- Resource diagnostics顯示C39仍受Min ROOS resource-preservation強烈影響：C35/C39的DL days=`274/275`、raw-direct-feasible days=`13/7`、minimum-repair days=`261/268`、actual changed-order days=`129/132`、reserved-capital delta=`+533,890/+716,817`。因此C39的失敗不能乾淨區分「Expected Excess-R×Risk objective本身不好」或「objective已含planned risk後又被R0 hard floor + R0 repair二次強調capacity」。
- 新受控實驗=`SR-C40 / Min MR-13E Excess-Alpha No-R0`，程式基準=`test-branch-1_20260815_014353_6a458df.zip`，SHA256=`98cfeec4868933f0334a9cf47e8c3e5336c875dfb2a2fb2eeacdf363e78b675d` + 本輪patch。
- C40與C39固定相同：`DL-CONT13E-PIT`、Expected Excess-R calibration identity、historical Min params、K、`Expected Excess-R × canonical planned initial risk` objective、sizing、actual cash、orderability、execution。唯一controlled change為`preserve_r0=False`、`r0_minimum_repair=False`。
- No-R0精確語意：不是取消真正資金限制。raw objective Top-K若在canonical cash-capped reservation下能保留K則直接接受，即使reserved capital低於baseline R0；若raw Top-K因真正cash feasibility無法保留K，只允許同參數Min ROOS baseline basket作K-only cash-feasible seed，再做Expected Excess-Alpha one-for-one ascent。seed只保證K/actual-cash feasibility，不做R0 restoration；`minimum_repair_steps/evaluations`必須保持0，`resource_preservation_required=False`。
- Decision contrasts：`C40-C39`只回答R0 hard floor + R0-driven repair純效果；`C40-C35`回答移除R0干擾後，Excess-Alpha×Risk objective是否真的優於原始MR-13E score objective。若C40只勝C39但仍輸C35，代表R0確有傷害但rank×capacity objective仍不值得採用；若C40仍未改善則停止此路線。Selection結果支持前不建立Forward arm、不跑multi-seed、不改模型或calibration。

### 2026-08-15 — SR-C40 Selection REJECT → SR-C41 Exact Constrained Excess-Alpha solver

- SR-C40正式Selection PIT結果已完成，run=`20260815_031604_C32-C23-C25-C28-C35-C39-C40_07556294e644`。C40與C39固定同一MR-13E PIT、Expected Excess-R calibration、historical Min params、K、Excess-Alpha×canonical planned-risk objective、sizing/cash/orderability/execution，只移除R0 hard floor與R0-driven minimum repair。
- C40主要結果：Return=`73.68%`、MDD=`18.83%`、RoMD=`3.91`、Annual=`8.21%`、EV=`0.27R`、Exposure=`59.30%`、DL selection R=`-81.67R`。相對C39：Return=`-57.76pp`、MDD=`-1.37pp`、RoMD=`-2.59`、Annual=`-4.53pp`、EV=`-0.31R`、Exposure=`-28.81pp`、DL selection R=`-132.69R`。因此No-R0明確REJECT，不建立Forward arm。
- 同時C40提供關鍵機制證據：R0 repair days由C39=`268`降為`0`、direct feasible days由`7`升為`288`；post-hoc Selected Target percentile=`0.6311`、top-K retention=`0.2148`、Target mean=`0.9951R`、opportunity gap=`3.0435R`，均顯示移除R0/repair後selection quality提高。然而總reserved-capital delta=`-73,969,424`、Exposure只有`59.30%`，說明R0具有必要capital-deployment功能，不能直接刪除。
- 新受控實驗=`SR-C41 / Min MR-13E Excess-Alpha Constrained`，程式基準=`test-branch-1_20260815_031548_4cb2516.zip`，SHA256=`f7ea9d0fe8b6dff5b07c397aeba84dca82729a1aa2474eccceaf5d428d6e36d0` + 本輪patch。
- C41與C39固定相同：MR-13E PIT、Expected Excess-R mapping、historical Min params、K/R0、`Σ(Expected Excess-R × canonical planned initial risk)` objective、canonical sizing/cash/orderability/execution。唯一controlled change為solver：C39的`raw Top-K → deterministic R0 minimum repair → best 1-swap feasible ascent`改成完整候選universe deterministic exact branch-and-bound，直接求K/R0/canonical-cash feasible global optimum。
- Exact solver contract：C39 feasible-ascent只提供incumbent lower bound，不能裁掉任何candidate membership；每個include state由canonical cash-capped reservation重新計算planned risk與reserved capital。Pruning只允許候選數不足、standalone maximum reserve不足、或admissible optimistic Excess-Alpha upper bound已不可能勝incumbent；禁止candidate truncation、beam width、time budget、lambda、R0比例或OOS-tuned tolerance。搜尋完整結束才輸出`constrained_solver_optimality_certified=True`。
- GPT獨立oracle：固定小型case直接暴力枚舉全部K-baskets，C41與oracle完全相同；另以30個randomized 8-candidate/K=3 case逐一完整枚舉，30/30皆與C41一致。另固定一個10-candidate/K=4 multi-swap trap：C39停在`MS2/MS5/MS8/MS9`，C41取得`MS0/MS3/MS8/MS9`，需同時替換2檔且canonical Excess-Alpha嚴格更高；2,000個小型randomized可行案例中另有11例C39/C41 membership不同且全部由C41取得更高objective，只作solver distinctness驗證。60-candidate synthetic benchmark在K=5/8/10的branch-and-bound states保持有限且均完成certificate；此benchmark只驗工程可行性，不作scientific gate。
- Decision Gate：只跑Selection C41，不建Forward、不跑multi-seed。主要contrast=`C41-C39`純solver effect與`C41-C35`最終Excess-Alpha策略經濟Gate。若exact constrained仍低於C35，停止`rank × capacity` objective路線；若明顯改善C39且勝C35，才另輪決定是否建立Forward arm。

### 2026-08-15 — SR-C41 Selection REJECT → SR-C42 MR-13E Score Exact Constrained

- SR-C41正式Selection PIT結果已完成，run=`20260815_034944_C32-C23-C25-C28-C35-C39-C40-C41_a3076d792ad2`。C41固定C39相同MR-13E PIT、Expected Excess-R calibration、historical Min params、K/R0、`Σ(Expected Excess-R × canonical planned initial risk)` objective、sizing/cash/orderability/execution，只把repair+1-swap heuristic替換為完整候選exact constrained branch-and-bound。
- C41主要結果：Return=`115.23%`、MDD=`20.65%`、RoMD=`5.58`、Annual=`11.58%`、EV=`0.51R`、Exposure=`88.20%`、DL selection R=`+26.86R`。相對C39：Return=`-16.21pp`、MDD=`+0.44pp`、RoMD=`-0.92`、Annual=`-1.16pp`、EV=`-0.07R`、Exposure=`+0.09pp`、DL selection R=`-24.16R`；相對C35：Return=`-38.73pp`、MDD=`+0.77pp`、RoMD=`-2.16`、Annual=`-2.67pp`、EV=`-0.03R`、Exposure=`-0.13pp`、DL selection R=`-9.10R`。
- Solver evidence：C41 repair/ascent eval均=`0`，Exact states=`66,283`、pruned=`31,288`、certified days=`1,018`；Exposure已恢復至與C35/C39相近水準，仍明顯落後。因此C39 failure不能再歸因於minimum-repair/1-swap local optimum，`Expected Excess-R × PlannedRisk` additive basket objective正式`REJECTED`；C39/C40/C41都不得建立Forward或multi-seed arm。
- 尚未回答的最小必要問題：exact constrained solver本身是否能改善目前最強的原始MR-13E score selector。既有C35=`score + K/R0 + repair/1-swap`，C41=`Excess-Alpha + K/R0 + exact`，因此仍缺`score + K/R0 + exact`這一格；不能由C41結果判定exact solver對score objective的效果。
- 新受控實驗=`SR-C42 / Min MR-13E Constrained`，程式基準=`test-branch-1_20260815_034918_37e4db2.zip`，SHA256=`33f604ded3eea285ad2c2ce4ffce30291c0cfce9e882738989d4da1ec65d5ac4` + 本輪patch。
- C42與C35固定相同：`DL-CONT13E-PIT`、historical Min params、K/R0、原始MR-13E continuous score objective、canonical sizing/cash/orderability/execution。唯一controlled change：C35的`score Top-K → deterministic minimum repair → best feasible 1-swap ascent`改為C41同源generic exact branch-and-bound，在完整候選universe直接最大化lexicographic MR-13E score quality subject to exact K/R0/canonical cash feasibility。
- C42不依賴Expected-R或Expected Excess-R calibration，不新增mapping、lambda、R0比例、candidate truncation、beam width或time cutoff。C41/C42共用同一exact solver owner，只由`objective_mode=excess_alpha/score`切換objective，避免solver implementation分叉。
- Decision Gate：只跑Selection。`C42-C35`為純solver effect；`C42-C41`在同MR-13E PIT、K/R0、exact solver與execution下只隔離score vs Excess-Alpha objective。若C42未明顯改善C35，停止selector exact-solver研究線；若明顯改善，才另輪決定是否建立Forward對應arm。

### 2026-08-15 — SR-C42 Selection PASS → SR-C43 MR-13A Score Exact Constrained

- SR-C42正式Selection PIT結果已完成，run=`20260815_041835_C32-C23-C25-C28-C35-C39-C40-C41-C42_d81955de516a`。C42固定C35相同`DL-CONT13E-PIT`、historical Min params、K/R0、原始MR-13E score、sizing/cash/orderability/execution，只把repair+1-swap heuristic替換為同源exact constrained branch-and-bound。
- C42主要結果：Return=`199.32%`、MDD=`19.88%`、RoMD=`10.03`、Annual=`16.96%`、EV=`0.75R`、Exposure=`88.16%`、DL selection R=`+117.16R`。相對C35：Return=`+45.36pp`、MDD=`0.00pp`、RoMD=`+2.28`、Annual=`+2.71pp`、EV=`+0.21R`、Exposure=`-0.17pp`、DL selection R=`+81.20R`。因此MR-13E pure score exact constrained solver Selection Gate明確PASS。
- Solver evidence：C42 repair/ascent正式path均=`0`，Exact states=`10,407`、pruned=`3,603`、certified days=`914`；selector total=`2206.33ms`、Median/day=`0.526ms`、P95/day=`4.983ms`，未出現不可接受的工程成本。C42相對C41同exact solver但objective改回原始score後Return=`+84.09pp`、RoMD=`+4.45`，進一步支持Expected Excess-R×Risk objective已REJECT而exact solver本身有價值。
- 尚缺的最小必要control：C28 MR-13A亦使用舊`Top-K → R0 minimum repair → 1-swap` heuristic；既然C42已證明solver可大幅改變MR-13E策略結果，不能再直接用C28 vs C42判定13A/13E純模型差異。故Forward C42暫緩一輪，先補13A exact constrained Selection。
- 新受控實驗=`SR-C43 / Min MR-13A Constrained`，程式基準=`test-branch-1_20260815_041825_439b851.zip`，SHA256=`627cdc64168c4d82b9a8c12c8c92efa1504f64597bdf4ada2403755b847d0533` + 本輪patch。
- C43與C28固定相同：`DL-CONT13A-PIT`、historical Min params、K/R0、原始MR-13A continuous score objective、canonical sizing/cash/orderability/execution。唯一controlled change：C28 repair+1-swap heuristic改為C42同一generic exact branch-and-bound，`objective_mode=score`，完整candidate universe直接求K/R0/canonical-cash feasible global optimum。
- C43不依賴Expected-R/Expected Excess-R calibration，不新增lambda、R0比例、candidate truncation、beam width、time cutoff，不重訓MR-13A、不重建PIT；robustness=`off`、Selection-only、無Forward arm。MR-12B C25/C20保留為historical/Forward anchor，不新增12B constrained。
- Decision Gate：`C43-C28`回答MR-13A純solver effect；`C42-C43`在同K/R0、同exact solver與execution下回答MR-13E vs MR-13A純DL source effect。C43結果取得後即停止新增Selection selector變體，直接決定哪一個daily exact-constrained candidate進single-seed frozen Forward。

### 2026-08-15 — SR-C43 Selection NOT PROMOTED → SR-C44 MR-13E Forward Score Exact Constrained

- SR-C43正式Selection PIT結果已完成，run=`20260815_044157_C32-C23-C25-C28-C35-C39-C40-C41-C42-C43_cf6bf772b3af`。C43固定C28相同`DL-CONT13A-PIT`、historical Min params、K/R0、原始MR-13A score、sizing/cash/orderability/execution，只把repair+1-swap heuristic替換為C42同源exact constrained branch-and-bound。
- C43主要結果：Return=`146.22%`、MDD=`18.86%`、RoMD=`7.75`、Annual=`13.74%`、EV=`0.68R`、Exposure=`88.09%`、DL selection R=`+103.25R`。相對C28：Return=`-5.66pp`、MDD=`0.00pp`、RoMD=`-0.30`、Annual=`-0.37pp`、EV=`-0.03R`、Exposure=`+0.29pp`、DL selection R=`-6.67R`；因此MR-13A exact constrained不提供promotion證據，不建Forward arm。
- Exact solver evidence：C43 repair/ascent eval=`0/0`，Exact states=`10,831`、pruned=`3,600`、certified days=`849`；同一exact solver下C42−C43=`+53.10pp Return / +1.02pp MDD / +2.27 RoMD / +3.22pp Annual / +0.07R EV / +13.91R DL selection R`，支持MR-13E source在此solver contract下明顯優於MR-13A。這也證明C42的Selection改善不是exact solver對所有daily model的普遍偏差。
- 新受控實驗=`SR-C44 / Min MR-13E Constrained Forward-OOS`。Effective source basis=`test-branch-1_20260815_041825_439b851(1).zip`，SHA256=`627cdc64168c4d82b9a8c12c8c92efa1504f64597bdf4ada2403755b847d0533` + 已成功執行的SR-C43 patch + 本輪patch；使用者本輪提供的ZIP本身仍是C43前source，因此只疊加已被正式Selection輸出證明已執行的C43 patch，不推測其他本機變更。
- C44與C36固定相同：`DL-CONT13E` frozen Forward-OOS score、current Min ROOS params、K/R0、原始MR-13E continuous score objective、canonical sizing/cash/orderability/execution。唯一controlled change：C36的`score Top-K → R0 minimum repair → best feasible 1-swap ascent`改為C42/C43同源generic exact branch-and-bound，`objective_mode=score`，完整candidate universe直接求K/R0/canonical-cash feasible global optimum。
- Stage contract：C42/C43的`selection_only=True`因source為`selection_point_in_time`；C44的`selection_only=False`因source為`continuous_ranker_oos`。Score-constrained validator改為要求此bool與DL source stage一致，而非硬編Selection-only；C44不依賴Expected-R／Expected Excess-R calibration、不重訓MR-13E、不重建Forward score。
- Forward Gate只新增C44，保留C1/C3/C20/C29/C36全部既有control。必要contrasts=`C44-C36`純MR-13E solver effect、`C44-C20`相對MR-12B current runtime anchor、`C44-C3`相對DL-off Min baseline。先跑single-seed frozen Forward；結果支持前不建立multi-seed robustness，也不新增MR-13A/12B constrained Forward。

### 2026-08-15 — SR-C44 Forward PASS → Selection / Forward 8-seed exact-constrained robustness activated

- SR-C44正式Forward-OOS結果完成，run=`20260815_053617_C1-C3-C20-C29-C36-C44_182b73f08149`。C44固定C36相同`DL-CONT13E` frozen Forward score、current Min ROOS params、K/R0、原始continuous score objective、sizing/cash/orderability/execution，只把repair+1-swap heuristic替換為generic exact constrained branch-and-bound。
- C44主要結果：Return=`214.29%`、MDD=`15.13%`、RoMD=`14.17`、Annual=`24.86%`、EV=`1.03R`、Exposure=`92.80%`。相對C36：Return=`+85.52pp`、MDD=`-2.06pp`、RoMD=`+6.67`、Annual=`+7.46pp`、EV=`+0.16R`；相對MR-12B C20：Return=`+24.03pp`、MDD=`-3.62pp`、RoMD=`+4.02`、Annual=`+1.91pp`。因此C44 single-seed Forward Gate PASS，但current runtime anchor在multi-seed完成前仍維持MR-12B。
- 下一個最小必要Gate依stage順序固定為`[3] Selection PIT Multi-seed robustness`後`[4] Forward-OOS Multi-seed robustness`。兩stage均沿用`seed_count=8`、`seed_generator_seed=20260810`、deterministic generated seeds；禁止best-seed selection、seed-specific調參或結果後改objective。
- Selection robustness matrix由robustness profile顯式指定：fixed=`C32 Full ROOS / C23 Min ROOS`；stochastic=`C25 MR-12B / C35 MR-13E heuristic / C42 MR-13E exact constrained`。Forward robustness matrix同樣由robustness profile顯式指定：fixed=`C1 Full ROOS / C3 Min ROOS`；stochastic=`C20 MR-12B / C36 MR-13E heuristic / C44 MR-13E exact constrained`。MR-13A C28/C29與C43不進本輪，C39/C40/C41亦不回流；single-seed arms既有`robustness_role`與Strategy Compare schema/fingerprint不因本輪robustness activation改寫。
- Same-seed paired reporting改為config-driven，只保留兩個決策contrast：`MR-13E exact vs MR-12B anchor`與`MR-13E exact vs same-source heuristic`；每個contrast獨立輸出RoMD wins/delta、DL selection R translation、cross-seed distribution與yearly same-seed delta，不再假設stochastic arms固定為2個，也不建立全排列比較噪音。
- Training orchestration按`(dl_id, seed)`去重。同stage有3 stochastic arms×8 seeds=`24` strategy replay observations，但只有2 unique DL sources×8=`16` model trainings：Selection為`CONT12B_PIT`與`CONT13E_PIT`，Forward為`CONT12B`與`CONT13E`。同一MR-13E seed只訓練/產生score一次，再fan-out到heuristic與exact runtime replay；shared training artifacts在全部fan-out replay完成後才依retention policy清理，避免第一個arm提前刪除第二個arm仍需的score/checkpoint。
- 此輪只改robustness orchestration/reporting與robustness profile membership；single-seed arm identity／`robustness_role`／Strategy Compare schema維持既有值，因此[1]/[2] scientific fingerprint不因啟用[3]/[4]而漂移。不改MR-13E/MR-12B training semantics、dataset/target、exact solver、K/R0、cash/sizing/execution，也不新增新的SR-C arm。結果尚未產生，不預先判定promotion。

### 2026-08-15 — 8-seed exact robustness closed → SR-C45 Full ROOS MR-13E Exact Selection

- 前一輪Selection PIT與Forward-OOS same-generated-8-seed robustness均已完成，固定`seed_count=8`、`seed_generator_seed=20260810`，沒有best-seed selection或seed-specific調參。Forward主要結果：MR-12B C20 RoMD Mean=`8.21`、MR-13E heuristic C36=`10.75`、MR-13E exact C44=`10.80`；C44-C20 same-seed ΔRoMD Mean=`+2.59`且C44勝=`6/8`，支持MR-13E model family相對MR-12B；C44-C36 ΔRoMD Mean僅=`+0.05`且雙方各勝=`4/8`，不支持宣稱exact solver本身具有跨seed穩健績效優勢。
- 因此後續把MR-13E Exact當作主要research architecture base的理由限定為scientific definition：完整candidate universe下直接求`K / R0 / canonical cash feasibility`的global optimum並輸出optimality certificate；不是把8-seed結果解讀成exact必然勝heuristic。MR-12B暫維持current runtime anchor／benchmark。
- 新受控實驗=`SR-C45 / Selection PIT: Full ROOS + Frozen MR-13E Score Exact Constrained`。程式基準=`test-branch-1_20260815_062408_fd2bde7(1).zip`，SHA256=`57f2ae2636b2c38d53c33c1b5b53930c2eb49d429a4987e4c40c326009bbbd9b` + 本輪patch。
- C45不新增任何selector implementation。完全重用C42的`DL-CONT13E-PIT`、原始`Σ MR-13E score` objective、`resource-aware-continuous-score-constrained-optimal`與`exact_branch_and_bound_v1`；`dl_runtime_options`與C42 byte-semantic一致：`preserve_k_r0=True / constrained_solver=exact_branch_and_bound_v1 / selection_only=True`。
- 唯一controlled strategy-system change：`param_source: selection_min_roos → selection_full_roos`、`rule_policy: all_off → formal`。C45與C32形成同一`selection_full_roos/formal` execution group，所以每日K、R0、canonical sizing、cash feasibility、orderability與execution全部由同日Full ROOS DL-off baseline建立；不得沿用C42的Min K/R0或Min all-off candidate universe。
- 必要contrasts只新增兩個：`C45-C32`回答Full ROOS系統內加入MR-13E Exact的策略經濟增量；`C45-C42`在同MR-13E PIT source與同exact solver下比較Full ROOS/formal vs Min ROOS/all-off的整體策略體系差異，明確不是單一參數效果。
- Gate只跑single-seed Selection。若C45沒有提供足以支持Full體系的證據，停止此分支，不占用`SR-C46`、不建立Full Forward arm、不跑8-seed；只有C45 Selection支持後，才另輪建立對稱的Full Forward-OOS MR-13E Exact arm。

### 2026-08-15 — SR-C45 Selection REJECT → Full ROOS transfer closed

- SR-C45正式Selection PIT結果已完成，run=`20260815_151550_C32-C23-C25-C28-C35-C39-C40-C41-C42-C43-C45_b90ab77b496d`，config fingerprint=`b90ab77b496d`。本輪結果回寫基準ZIP=`test-branch-1_20260815_151531_e0a8e0b.zip`，SHA256=`f000a82448a269df6b6fe22ce11952a2da9560486653df858c53a60a7834a29a`。
- C45固定MR-13E PIT score、原始score objective與`exact_branch_and_bound_v1`，只把策略體系從C42的`selection_min_roos + all_off`切換為`selection_full_roos + formal`；K/R0、canonical sizing、cash feasibility、orderability與execution仍由同日C32 Full baseline semantics建立，因此此次失敗不能歸因為誤用Min K/R0。
- C45主要結果：Return=`107.65%`、MDD=`19.02%`、RoMD=`5.66`、Annual=`11.01%`、EV=`0.64R`、Exposure=`79.60%`、trades=`422`、same-param DL selection R=`-29.22R`。相對同體系C32 Full ROOS：Return=`-24.60pp`、MDD=`+3.16pp`、RoMD=`-2.68`、Annual=`-1.79pp`、EV=`-0.07R`；相對Min Exact C42：Return=`-91.67pp`、MDD=`-0.86pp`、RoMD=`-4.36`、Annual=`-5.95pp`、EV=`-0.11R`、Exposure=`-8.56pp`。Full/formal transfer屬明確negative result，不需再補Forward或8-seed。
- Decision：`SR-C45 = REJECTED_FULL_TRANSFER / NO_FORWARD / HISTORICAL_ONLY`。不建立`SR-C46`，不做Full ROOS MR-13E Exact Forward，也不做Full multi-seed。C45 arm與`C45-C32/C45-C42` contrasts從current Selection profile與active catalog移出，移到`config/compatibility/strategy_compare_history.py`只供舊工件重現；Strategy Compare schema `28 → 29`。
- 架構結論：此結果只否決Full/formal transfer，不否決Min MR-13E Exact。既有same-seed 8-seed證據中，Min Exact相對MR-12B的RoMD為Selection `6/8`勝、ΔMean=`+0.19`，Forward `6/8`勝、ΔMean=`+2.59`；但Selection DL selection R相對MR-12B只有`1/8`勝、ΔMean=`-17.72R`，而Exact相對MR-13E heuristic在Selection/Forward都只有`4/8`勝（ΔRoMD Mean分別`+0.22/+0.05`）。因此後續以`Min/all-off + MR-13E raw score + exact constrained solver`作strategy research base，理由是完整global optimum/certificate與Forward strategy evidence，而不是宣稱MR-13E純DL source或exact solver已普遍優於所有anchor。
- Runtime boundary：本輪不修改`config/breakout_quality.py`的production/default workflow，不把研究結論自動變成部署切換。MR-12B仍可作既有runtime/default benchmark；若要正式把Min MR-13E Exact接成production/runtime path，必須另開一輪integration，明確驗證live-like score availability、exact selector owner、artifact identity與fallback semantics，不在此次C45 research closure中偷改。

### 2026-08-16 — SR-C46/C47 Selection result closure → SR-C48 Basket-level Pareto Exact No-R0 implementation

- 最新程式基準=`test-branch-1_20260816_152710_e7e45c1.zip`，SHA256=`6eff4eea4e4de0907c99dfa766710d76c8039d3d464c581e096054fe9548fca1` + 本輪patch。先補齊前一輪已取得但Registry仍標PENDING的正式Selection結果：run=`20260815_221854_C32-C23-C42-C46-C47_93d965ca2379`。
- `SR-C46 / Frozen MR-13E Score Exact No-R0`：Return=`95.89%`、MDD=`16.97%`、RoMD=`5.65`、Annual=`10.09%`、EV=`0.33R`、Exposure=`56.76%`、DL selection R=`-51.12R`。相對C42 Return=`-103.42pp`、RoMD=`-4.37`、Exposure=`-31.40pp`；但Target mean=`1.20R`、Target %ile=`0.648`、Top-K=`21.75%`、Opp gap=`3.27R`反而較強。Decision=`REJECTED_DEPLOYMENT_COLLAPSE / NO_FORWARD / HISTORICAL_ONLY`：純ΣScore No-R0會偏向低reserved-capital basket，模型selection quality本身不是主要失敗點。
- `SR-C47 / Frozen MR-13E Score×Capital Exact No-R0`：Return=`168.86%`、MDD=`20.00%`、RoMD=`8.44`、Annual=`15.18%`、EV=`0.69R`、Exposure=`88.05%`、DL selection R=`+105.30R`。相對C46 Exposure=`+31.28pp`、Return=`+72.97pp`，證明capital deployment確實是No-R0缺少的第二個目標；但相對C42 Return=`-30.46pp`、RoMD=`-1.58`、EV=`-0.06R`且Exposure只差`-0.11pp`。Decision=`REJECTED_SIMPLE_PRODUCT / NO_FORWARD / HISTORICAL_ONLY`：candidate-level `Σ(score_i×capital_i)`會過度把capital重寫進股票排序，不能取代R0。
- 新受控實驗=`SR-C48 / Frozen MR-13E Basket-level Pareto Exact No-R0`。固定：`DL-CONT13E-PIT`、historical Min ROOS params、all-off rules、baseline K、canonical 1% risk sizing、true-cash reservation、orderability、execution；不重訓模型、不改Target/Dataset/Label、不用Expected-R calibration。唯一portfolio研究變數：移除baseline-derived R0，改在basket level同時處理`Q=ΣScore`與`C=canonical reserved_cost_milli`。
- C48 exact定義：所有search pass先lexicographic最大化Score coverage，避免missing Score因視為0而打敗負Score；在最大coverage的K/cash-feasible universe中，pass-1 exact求score endpoint `(Q_max, C_min)`（Q優先、同Q取較高C），pass-2 exact求capital endpoint `(Q_min, C_max)`（C優先、同C取較高Q）。以兩端定義`Q_norm=(Q-Q_min)/(Q_max-Q_min)`與`C_norm=(C-C_min)/(C_max-C_min)`；若某軸沒有range，該軸固定為1。pass-3 exact最大化`Q_norm×C_norm`，同product tie依Q、C、baseline deterministic order決定。因scalar對Q/C單調且tie不偏好被支配點，final basket位於Pareto-efficient set。無lambda、無score shift、無candidate-level Score×Capital、無R0/minimum-repair/feasible-ascent。
- Exact implementation使用三個完整candidate include/exclude branch-and-bound pass；每個included partial state都重跑canonical cash-capped reservation。Pruning只使用admissible orderable-count、Score coverage、standalone optimistic Score upper與reserved-capital upper；沒有candidate truncation、beam width、time cutoff。Diagnostics保存score/capital/product三pass states、兩個normalization端點、final normalized Q/C/product與optimality certificate。
- Selection current matrix由schema `31 → 32`，並依研究生命週期把已REJECT的C46/C47及其contrasts移到`config/compatibility/strategy_compare_history.py`。Current active=`C32 / C23 / C42 / C48`；contrasts=`C32-C23 / C42-C23 / C42-C32 / C48-C42`。Forward仍=`C1 / C3 / C44`，robustness與production candidate仍固定C42/C44，不因C48 implementation改動。
- `validate_strategy_compare_config_driven_app_contract_case`新增C48 exhaustive oracle：小型K=2案例完整枚舉所有canonical cash-feasible baskets，獨立求score endpoint、capital endpoint與normalized-product global optimum，要求runtime C48選解、normalization端點與certificate完全一致；本輪implementation階段狀態只能是`IMPLEMENTED / SELECTION_PIT_PENDING / NO_FORWARD_YET / ROBUSTNESS_OFF`。
- C48 decision gate：先只跑Selection `C48-C42`。只有C48能在Exposure不再崩塌的前提下，至少接近／超過C42的Return、RoMD與EV，才另輪建立Forward；若仍明顯輸C42，停止No-R0 allocator路線，不繼續試lambda或其他Score×Capital變形。
- Implementation validation（非正式策略結果）：config-driven contract=`118/118 PASS`、Strategy Compare=`45/45 PASS`、readable report=`6/6 PASS`、Daily PIT runtime=`9/9 PASS`、MR-13E source gate=`4/4 PASS`、Runtime Integration=`13/13 PASS`；另以48組隨機小型universe逐組完整枚舉全部合法basket，三pass C48 runtime與獨立exhaustive oracle=`48/48`一致。依`PROJECT_SETTINGS`未執行`apps/run_bundle.py`或`apps/test_suite.py`。

### 2026-08-16 — SR-C48 Selection result closure / No-R0 allocator line stopped

- 結果來源：使用者由正式`apps/research.py → 策略組合比較 → Selection PIT`取得aggregate，period=`2014-01-01～2020-12-31`、config fingerprint=`27354349956f`。C42與C48使用同一frozen `DL-CONT13E-PIT` model-prediction metrics：Daily rho=`0.163`、Global rho=`0.053`、Pair一致=`55.75%`、Top-R=`0.95R`、Bottom-R=`0.67R`、Top-BottomR=`0.28R`，因此差異歸因於allocator，不是模型預測力改變。
- `SR-C48`結果：Return=`135.64%`、MDD=`17.52%`、RoMD=`7.74`、Annual=`13.03%`、Worst full year=`-9.66%`、Log R²=`0.8946`、Monthly win=`61.90%`、Win rate=`37.74%`、Payoff=`3.16`、EV=`0.51R`、Trades=`461`、Exposure=`84.97%`。相對C42：Return=`-63.68pp`、MDD=`-2.36pp`（較低）、RoMD=`-2.29`、Annual=`-3.93pp`、Worst full year=`-11.46pp`、Log R²=`-0.0512`、Monthly win=`-8.34pp`、EV=`-0.24R`、Exposure=`-3.19pp`。年度Return只有2018/2019高於C42，其餘5/7完整年度較低。
- R轉化：C48平均R=`0.51R` vs C42=`0.75R`，中位R=`-0.36R` vs `-0.28R`，DL選擇R=`+38.19R` vs `+117.16R`，RCE=`15.96%` vs `138.63%`。另一方面C48 Target mean=`0.87R` vs `0.71R`、Target percentile=`0.566` vs `0.546`、Opp gap=`3.61R` vs `3.97R`，顯示basket-level Pareto確實稍改善future Target selection quality，但這個改善沒有轉成realized trade R與portfolio profit。
- Execution：C48 fill rate=`97.46%`、daily orderable candidates=`57.92`、candidate shortage=`118 days`、underfilled end=`835 days`、position gap=`1499 slot-days`；Exposure仍為`84.97%`，不像C46崩到`56.76%`，所以C48失敗已不是單純deployment collapse。
- Decision=`REJECTED_NO_R0_PARETO / NO_FORWARD / HISTORICAL_ONLY`。C46證明純Score No-R0會deployment collapse；C47證明candidate-level Score×Capital能恢復deployment但仍輸C42；C48進一步證明basket-level normalized Pareto雖能兼顧capital與Target quality，仍無法恢復C42的realized-R conversion與portfolio economics。因此No-R0 allocator研究線在C48後停止，不建立Forward、不跑multi-seed、不再試lambda或其他Score×Capital/Pareto變形。Current strategy research anchor維持Min/all-off + frozen MR-13E + exact K/R0 constrained C42/C44。
- 研究生命週期同步：Strategy Compare config schema=`33`，current Selection matrix回到`C32/C23/C42`，contrasts=`C32-C23 / C42-C23 / C42-C32`；C48及`C48-C42`移入`config/compatibility/strategy_compare_history.py`供歷史重現。Production/runtime仍為C42/C44，不受本次closure影響。

### 2026-08-16 — AUD-min-roos-planned-risk-40d-alignment result / closure

- 使用者本機正式執行結果：Decision=`GO`；source=`outputs/strategy_compare/selection_pit/runs/20260816_164730_C32-C23-C42-C48_27354349956f/pairs/selection_min_roos__all_off__CONT13E_PIT__resource_aware_continuous_score_constrained_optimal`；completed trades=`420`、comparable=`414`、horizon=`40 trading bars after entry date`、fixed-risk control=`10.00%`。
- Headline：MR-13E existing Spearman=`0.5288`、Pair=`69.10%`、Top-Bottom realized R=`4.13R`、MAE=`1.63R`；entry-date fixed-risk 40D=`0.3689 / 64.45% / 4.43R / 1.58R`；planned-risk 40D=`0.5267 / 70.03% / 5.00R / 1.57R`。planned-risk相對same-entry fixed-risk同時改善Spearman與Top-Bottom separation，依事前gate正式GO；其整體Spearman與existing MR-13E近乎相同，不得解讀成全面取代MR-13E。
- 年度Spearman planned-risk 2014～2020=`0.608, 0.543, 0.629, 0.629, 0.524, 0.654, 0.117`；相對entry-fixed在2014～2019皆不差且多數明顯更好，2020仍弱，故後續模型必須保留year/regime reporting但不得新增year gate。
- Capital/Risk quintile planned-risk rho Q1～Q5=`0.477, 0.527, 0.566, 0.627, 0.453`；沒有只在單一capital bucket成立，但Q1/Q5弱於中間區，後續不得假設capital intensity與utility為單調線性關係。
- Entry-age quartile planned-risk rho Q1～Q4=`0.436, 0.328, 0.703, 0.540`；相對MR-13E Q1/Q2較弱、Q3/Q4較強，說明planned-risk geometry在較晚entry候選的alignment改善最明顯；這只作diagnostic，不建立entry-age gate。
- 決策：`AUD-min-roos-planned-risk-40d-alignment`完成使命，停止Audit鏈。下一步是受控模型實驗，不再新增同題Audit；為隔離candidate-universe改變與planned-risk target效果，下一模型研究必須先保留同candidate universe的existing-target control，再只切換planned-risk target。
- Lifecycle：依`PROJECT_SETTINGS.md C11`，formal Audit implementation/config/catalog與只服務本Audit的dedicated synthetic退役；Registry/Log保留結果與identity。

### 2026-08-16 — MR-13I Forward model result / MR-13J authorized

- 使用者由正式`apps/research.py → 模型訓練 → 訓練目前模型`完成`MR-13I / daily_universal_risk_normalized_net_full_list_ndcg_pairwise` Seed42 Forward model Gate。Dataset/index：`score_eligible=1,600,734`、`target_valid=1,176,931`、`tickers=563`，historical-effective Min ROOS risk-param coverage start=`2014-01-01`；selected epoch=`1/200`，epoch-selection Validation daily rho=`0.0495`，完整Selection重訓後原Validation daily/global rho=`0.0850/0.1436`、pair=`52.99%`。
- Forward all-stock OOS：daily/global rho=`0.0645/0.0150`、pair=`52.30%`；Top10% Target=`0.8042R`、Bottom10%=`0.8430R`，spread=`-0.0388R`；NDCG@10=`0.5162`、Top-K Target=`0.8782R`、Lift=`+0.0137R`、Oracle overlap=`3.29%`、Boundary=`47.50%`、Boundary gap=`-0.1468R`。Breakout-candidate OOS：daily/global rho=`0.0179/-0.0722`、pair=`51.29%`；Top10%=`0.9185R`、Bottom10%=`1.1579R`，spread=`-0.2394R`；NDCG@10=`0.6819`、Top-K Lift=`+0.0405R`、Oracle overlap=`55.82%`、Boundary gap=`-0.0034R`。
- 與frozen MR-13E Forward model reference比較：13E all-stock daily/global rho=`0.2151/0.1042`、pair=`57.42%`、Top10%-Bottom10% spread=`+0.8234R`、Top-K Lift=`+0.9951R`；breakout daily/global rho=`0.1721/0.1535`、pair=`57.43%`、spread=`+1.4294R`、Top-K Lift=`+0.4489R`。13I相對13E all-stock daily rho=`-0.1506`、pair=`-5.12pp`、spread=`-0.8622R`、Top-K Lift=`-0.9814R`；breakout daily rho=`-0.1542`、pair=`-6.14pp`、spread=`-1.6688R`。因此`MR-13I = FORWARD_OOS_MODEL_GATE_FAIL`，不得進Selection PIT、runtime source或Strategy Compare。
- 13I失敗不推翻A0 target-mechanism GO：A0在同entry completed trades上已證明planned-risk normalization相對entry-fixed control改善Spearman／Pair／Top-Bottom realized-R；13I目前唯一scientific change是把這個risk-normalized target交給**沒有顯式risk-param/geometry context**的sequence-only MR-13E backbone。尤其historical `atr_len/atr_times_init`會跨period改變，而13I input不直接提供這些risk/economic geometry，因此目前關鍵不確定性已縮成「顯式decision-time geometry能否讓模型學到同一Target」。
- 決策：`MR-13I = RESULT_AVAILABLE / REJECTED_AS_SEQUENCE_ONLY / NO_PIT / NO_RUNTIME_DL_SOURCE`；`MR-13J = NEXT_EXPERIMENT_AUTHORIZED / ACTIVE_MODEL_RESEARCH / FORWARD_MODEL_GATE_PENDING`。13J Target/universe/horizon/loss必須與13I完全相同，只新增既定5維context；若13J仍無法在Forward all-stock與breakout slice形成有意義的正向ranking/separation，則停止此risk-normalized DL路線，不再以Selector或Target微調救援。Production workflow與C42/C44仍固定MR-13E。

### 2026-08-16 — MR-13J Forward model result / C49-C50 Selection strategy gate authorized

- 使用者由正式`apps/research.py → 模型訓練 → 訓練目前模型`完成`MR-13J / daily_universal_risk_context_net_full_list_ndcg_pairwise` Seed42 Forward model Gate。Dataset/index：`score_eligible=1,198,243`、`target_valid=1,176,931`、`tickers=563`，historical-effective Min ROOS risk-param coverage start=`2014-01-01`；context=`risk_distance_pct / risk_distance_atr / capital_per_risk / cost_per_risk / risk_capacity`；selected epoch=`1/200`，epoch-selection Validation daily rho=`0.0265`，完整Selection重訓後Validation daily/global rho=`0.1162/0.0202`、pair=`54.01%`。
- Forward all-stock OOS：daily/global rho=`0.1086/0.1280`、pair=`53.89%`；Top10% Target=`1.1727R`、Bottom10%=`0.6363R`，spread=`+0.5364R`；NDCG@10=`0.5600`、Top-K Target=`1.1348R`、Lift=`+0.2702R`、Oracle overlap=`4.04%`、Boundary=`48.76%`、Boundary gap=`-0.0673R`。Breakout-candidate OOS：daily/global rho=`0.0825/0.1179`、pair=`54.22%`；Top10%=`1.1875R`、Bottom10%=`0.6842R`，spread=`+0.5033R`；NDCG@10=`0.7077`、Top-K Lift=`+0.1217R`、Oracle overlap=`58.00%`、Boundary gap=`+0.0290R`。
- 相對13I，13J all-stock daily rho=`+0.0441`、global rho=`+0.1130`、pair=`+1.59pp`、Top-Bottom spread=`+0.5752R`、Top-K Lift=`+0.2565R`；breakout daily rho=`+0.0646`、global rho=`+0.1901`、pair=`+2.93pp`、spread=`+0.7427R`。因Target/universe/horizon/loss完全相同，改善可歸因於顯式risk/economic geometry input；`MR-13J = FORWARD_MODEL_GATE_PASS`。但其all-stock daily rho仍低於frozen MR-13E=`0.2151`，所以不直接promotion，下一關必須看策略轉化。
- 下一個最小必要實驗採三個核心arm：C42=`MR-13E + K/R0 exact` production research base；C49=`MR-13J + 完全相同K/R0 exact`，只換DL source以隔離model effect；C50=`MR-13J + K/exact/canonical cash + No-R0`，只相對C49移除R0以檢驗13J是否已把capital/risk/cost economics upstream到Score。既有`selection_pit` profile仍固定2014～2020、C32/C23/C42，不因新target改寫歷史Gate；C49/C50放在獨立`selection_risk_context` profile；C23只作同Min/all-off DL-off共同基準，核心scientific comparison仍是C42/C49/C50三臂。
- 新identity：`DL-CONT13J-PIT / CONT13J_PIT`、`SR-C49`、`SR-C50`。由於13J risk-normalized target/context的historical-effective risk-param coverage從2014-01-01才開始，而Selection PIT fold本身還需要更早的train/validation history，因此不得宣稱13J PIT可從2014-01-01合法出分。`selection_risk_context`不設定固定起訖，執行時由CONT13E/CONT13J PIT runtime共同coverage自動取交集；禁止current/future Min params backfill到更早日期。C49/C50只允許Selection PIT；在結果前不建立Forward、不開multi-seed、不改production runtime candidate C42/C44。C46/C47/C48的MR-13E No-R0 allocator結論仍維持REJECT。
- C49 primary attribution=`C49-C42`；C50 primary attribution=`C50-C49`，並保留`C50-C42`作最終整體架構比較。判斷R0是否可退休不能只看Return：至少同看Average Exposure、underfilled/position-gap、reserved-capital diagnostics、realized average R、EV、RCE、Return與RoMD。若C50再次出現明顯deployment collapse，即判定13J仍未取代R0；若C49本身已無策略增量，亦不再為13J調context/selector救援。

### 2026-08-16 — MR-13J Selection PIT result / Strategy Compare PIT ownership boundary closure

- 使用者在`[3] 策略組合比較 → Selection PIT 策略比較`執行C42/C49/C50前置時，舊planner因`CONT13J_PIT` top-level工件缺少而自行呼叫checkpoint-only PIT builder；該builder除了重建score／manifest，也直接執行PIT Audit／Model Gate，造成模型驗證報表從Strategy Compare入口輸出。這是工作類型ownership錯誤，不是報表renderer差異。
- 該次實際產出的MR-13J Selection PIT工件合法完成：score period=`2016-04-01～2020-12-31`、folds=`5`、coverage=`455,971/455,971 (100%)`。All eligible stock-days Global Spearman=`-0.0321`、Daily rho=`+0.0490`、Top=`0.7730R`、Bottom=`0.8277R`、spread=`-0.0547R`；Breakout candidates Global/Daily rho=`-0.0250/+0.0770`、spread=`+0.0050R`。年度Global Spearman正向=`3/5`，年度Top-Bottom spread正向=`4/5`。
- 事前`derive_point_in_time_model_validation_gate()`要求Primary Global rho>0、Daily rho>0、正向Spearman年度嚴格過半、正向spread年度嚴格過半全部成立。MR-13J只有Global rho=`-0.0321`未通過，其餘三條成立，因此`MR-13J Selection PIT Model Gate = FAIL`。決策：C49/C50不得進strategy replay，不以R0／No-R0 selector結果繞過模型Gate；Production仍是MR-13E + C42/C44。
- Engineering ownership同步收斂：一般Strategy Compare對Selection PIT source改成consumer-only。Gate=PASS時只REUSE既有score／manifest／audit；缺失／無效時直接BLOCK並導向`Research → [1] 模型訓練 → [2] 建立／更新 Selection PIT Scores`；已有工件但Gate=FAIL時直接BLOCK策略績效驗證。即使既有fold score／checkpoint可checkpoint-only重建，Strategy Compare也不得執行PIT builder或PIT Audit。Forward-OOS deterministic score export與策略參數auto builder契約不變。

### 2026-08-17 — MR-13J PIT quality gate reclassified as advisory for controlled strategy conversion

- 使用者明確修正研究治理：PIT模型排序Gate不能自動取代最終策略經濟驗證。MR-13J Selection PIT工件本身已完整合法（2016-04-01～2020-12-31、5 folds、coverage=100%、identity/hash/PIT chronology合法），因此Global rho=-0.0321造成的Model Gate=FAIL保留為重要WARN，但不得單獨阻止已事前定義的C49/C50 controlled strategy replay。
- Ownership邊界不變：Selection PIT score/manifest/audit與Model Gate仍只由`[1] 模型訓練 → [2] 建立／更新 Selection PIT Scores`產生；Strategy Compare不得建立、重建或重跑PIT。真正hard blocker仍是missing/invalid identity/hash/coverage/period/look-ahead。
- Current Selection研究profile恢復`selection_risk_context`：C23共同DL-off baseline、C42=`MR-13E+R0`、C49=`MR-13J+R0`、C50=`MR-13J+No-R0`；實際比較期間由CONT13E/CONT13J PIT共同合法coverage自動取交集。核心歸因仍為C49-C42（13J model effect）、C50-C49（R0 ablation）、C50-C42（最終架構）。
- Production/runtime promotion規則不放寬：MR-13J仍`NO_RUNTIME_PROMOTION`，只有本次明確configured strategy-conversion研究可在WARN下消費既有PIT scores。最終是否繼續13J只看實際strategy replay的Return/MDD/RoMD/EV/realized R/RCE/exposure/underfill，而不是單靠PIT rho提前決定。

### 2026-08-17 — MR-13J C49/C50 strategy-conversion closure

- 使用者由正式`apps/research.py → [3] 策略組合比較 → Selection PIT 策略比較`完成`selection_risk_context`，共同合法期間=`2016-04-01～2020-12-31`、config fingerprint=`66b78fa6407a`。C23=`Min ROOS`、C42=`MR-13E + K/R0 exact`、C49=`MR-13J + 完全相同K/R0 exact`、C50=`MR-13J + K/exact/canonical cash + No-R0`；因此C49-C42只隔離模型source，C50-C49只隔離R0。
- 核心結果：C42 Return/MDD/RoMD=`104.29% / 12.43% / 8.39`、EV=`0.83R`、平均曝險=`90.42%`；C49=`97.21% / 12.33% / 7.88`、EV=`0.87R`、平均曝險=`88.97%`。C49相對C42 Return=`-7.08pp`、RoMD=`-0.51`、年化=`-0.86pp`，雖MDD改善`0.10pp`、EV `+0.04R`、Log R²=`0.9208 vs 0.8622`、最差完整年度=`+0.32% vs -0.42%`，整體未形成可取代MR-13E的策略增量。資金端C49期末未滿倉日=`750 vs 643`、持股缺口=`1899 vs 1214`，亦顯示13J選股轉化帶來較弱部署。
- R診斷同方向：C49實際平均R=`0.87R`略高於C42=`0.83R`，exclusive DL選擇R=`42.16R vs 23.20R`、RCE=`59.58% vs 33.03%`；但13J PIT本身Daily/Global rho=`0.049/-0.032`、Top-Bottom=`-0.05R`，且較大的exclusive edge沒有轉成較高portfolio Return/RoMD。這足以判定「局部交易品質改善 ≠ 整體資金配置改善」，不需要再新增Audit。
- C50 No-R0明確失敗：Return/MDD/RoMD=`55.61% / 12.43% / 4.47`、EV=`0.47R`、平均曝險=`74.31%`、DL選擇R=`-49.10R`、RCE=`-262.26%`。相對C49 Return=`-41.60pp`、RoMD=`-3.41`、EV=`-0.40R`、平均曝險=`-14.66pp`。雖期末未滿倉日由`750`降至`331`、持股缺口由`1899`降至`646`，代表slot數更容易填滿，但每格實際部署的資金顯著縮小；這直接證明R0的功能不是單純「避免沒股票可買」，而是維持同參數DL-off baseline的reserved-capital floor。13J的`capital_per_risk / cost_per_risk / risk_capacity` context沒有把這個portfolio-level hard constraint可靠地學進Score。
- 決策：`MR-13J = STRATEGY_CONVERSION_REJECTED / NO_RUNTIME_PROMOTION`；`SR-C49 = REJECTED / NO_FORWARD`；`SR-C50 = REJECTED_NO_R0 / NO_FORWARD`。依事前停止條件，13J若C49本身無策略增量就不再以context/selector救援；C50又重現No-R0 deployment collapse，因此不建Forward C49/C50、不開multi-seed、不新增Audit。risk-normalized/context路線結案，current model research回到已promotion的MR-13E，C42/C44與R0維持。
- 工程收斂：`selection_risk_context`退出current Strategy Compare menu；`CONT13J_PIT / C49 / C50 / C49-C42 / C50-C49 / C50-C42`移至`config/compatibility/strategy_compare_history.py`只供歷史工件解讀／重現；schema升至`37`，current Selection恢復`selection_pit: C32/C23/C42`。不刪generic risk-context architecture、risk-normalized target與No-R0 solver implementation，因仍需歷史artifact reconstruction／compatibility；只退役一次性active registration與其experiment-specific synthetic expectation。

### 2026-08-17 — MR-13H Label Audit + Seed42 Forward GO → Selection PIT authorized

- 程式基準：`test-branch-1_20260817_015333_b309e28(2).zip`，SHA256=`a89ae89d18fdae8d01321423cf7cfae7b6511cd1fb1e31d91a946050dad32eb8`。本輪不改MR-13H Target、architecture、loss、Seed或production workflow，只依已取得證據推進研究stage。
- Label Audit正式結果：common stock-days=`1,578,349`、40D risk breach=`31.31%`、Target changed=`6.65%`、candidate-vs-reference Daily rank correlation=`0.9435`、Top-10 overlap=`94.53%`、Top-K overlap=`94.53%`、Mean abs delta=`0.0714R`。這表示first-breach truncation只改變少量樣本，但仍保留足夠受控差異供實際訓練驗證。
- Seed42 Forward正式結果：selected epoch=`1`；validation refit Daily rho=`0.2603`；all-stock OOS Daily/Global rho=`0.2332/0.1369`、Pair=`58.04%`、Top10%-Bottom10% Target=`+1.0442R`、Top-K Lift=`+1.3400R`；breakout slice Daily/Global rho=`0.2056/0.1775`、Pair=`58.86%`、Top10%-Bottom10% Target=`+1.6044R`、Top-K Lift=`+0.5061R`。
- Frozen MR-13E Forward reference為all-stock Daily/Global rho=`0.2151/0.1042`、Pair=`57.42%`、Top10%-Bottom10%=`+0.8234R`、Top-K Lift=`+0.9951R`；MR-13H在self-target主要ranking指標一致改善。更重要的是同一MR-13H score對MR-13E reference Target仍有Daily rho=`0.2140`、Pair=`57.41%`，與MR-13E本身`0.2151/57.42%`幾乎相同，因此目前沒有看到以較容易Target換取舊ranking能力流失。
- Decision：`MR-13H = FORWARD_OOS_MODEL_GATE_PASS / SELECTION_PIT_AUTHORIZED`。PIT不加入13C/D/E batch，因其Target identity不同；由`ContinuousRankerResearchSpec.selection_pit_authorized=True`走單profile `[1] 模型訓練 → [2] 建立／更新 Selection PIT Scores → PIT模型驗證`。
- Selection strategy controlled arm新增`DL-CONT13H-PIT / SR-C51`。C51與C42固定完全相同historical Min params、all-off、K/R0、canonical sizing/cash/orderability/execution與`exact_branch_and_bound_v1`，唯一變更為MR-13H PIT score。Primary contrast=`C51-C42`；C51未進multi-seed。
- 階段順序依使用者要求改為先把MR-13H整條流程做完才開下一個Label：Selection PIT Model Gate → C51 Selection strategy translation → 若Selection支持才建立MR-13H Forward strategy arm → 若Forward亦支持才進same-generated-8-seed robustness／promotion判讀。任何一關失敗即結案MR-13H，不提前開Pure-MFE或cross-sectional新MR。Production/runtime目前仍固定MR-13E C42/C44。

### 2026-08-17 — MR-13H Selection result → full Forward + robustness evaluation authorized

- 程式基準：`test-branch-1_20260817_022948_d464717.zip`，SHA256=`872d3388407c815f770573a94694f22d6b4a950583b3d4fb041e7021132203b6`。本輪不改MR-13H Target、architecture、loss、Seed policy、K/R0或exact solver；只擴充Forward與multi-seed評估矩陣及一次串跑UI。
- Selection Strategy結果：C42 MR-13E Return=`199.32%`、MDD=`19.88%`、RoMD=`10.03`、EV=`0.75R`；C51 MR-13H Return=`158.21%`、MDD=`19.64%`、RoMD=`8.06`、EV=`0.80R`。C51單次Selection在Return/RoMD低於C42，雖EV略高，不足以單獨支持promotion。
- 使用者明確決策：不在Selection單次結果提前REJECT；先完整測完MR-13H整條流程，再決定是否保留此Label。故建立`DL-CONT13H / SR-C52` Forward source-only exact arm，與C44唯一差異為MR-13H vs MR-13E Forward model source。
- Current Forward matrix=`C1/C3/C44/C52`；primary source-only contrast=`C52-C44`。Selection robustness stochastic=`C42/C51`、paired=`C51-C42`；Forward robustness stochastic=`C44/C52`、paired=`C52-C44`；fixed references仍分別為Selection `C32/C23`、Forward `C1/C3`。兩stage使用同一deterministic generated 8 seeds與generator seed=`20260810`，不挑best seed、不ensemble。
- `apps/research.py → [3] 策略組合比較`新增泛化工作項`[5] 一次執行全部 Multi-seed robustness`：依config啟用profile順序串行執行Selection PIT與Forward-OOS robustness，兩stage之間不再要求確認；每一stage仍先顯示自己的canonical execution plan，任一stage BLOCKED/FAIL立即停止，不執行後續stage。原Runtime Integration Gate與全部狀態頁順延。
- Production/runtime candidate仍固定Selection C42／Forward C44；本輪只擴張research evaluation matrix，不構成promotion。下一步：先完成C52 Forward single-seed strategy comparison，再由`[5]`一次跑完Selection+Forward 8-seed robustness，最後依Selection/Forward/paired-seed整體證據決定MR-13H。

### 2026-08-17 — MR-13H full-flow closure → MR-13K Pure-MFE implementation

- MR-13H已完成完整策略robustness，正式seed count=`16`。Selection 2014～2020：C42 MR-13E Mean Return/MDD/RoMD=`117.06%/19.69%/5.98`，C51 MR-13H=`123.45%/19.73%/6.28`；same-seed RoMD為13H勝`10/16`，DL選擇R為13H勝`13/16`。Forward 2021～2026-03-02：C44 MR-13E=`173.92%/15.89%/11.10`，C52 MR-13H=`171.21%/16.60%/10.32`；same-seed RoMD為13E勝`9/16`、13H勝`7/16`。
- 決策：`MR-13H = VALID_LABEL_SIMPLIFICATION / NOT_SELECTED_FOR_PROMOTION`。移除first -10% breach truncation確實提高Seed42 Forward self-target predictability，但Selection小幅優勢未在Forward robustness延續，因此不取代MR-13E production anchor，不再增加13H seeds或調整selector。
- 新實驗identity：`MR-13K / PROFILE-daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise / TARGET-daily_full_horizon_pure_mfe_r_v1`。Reference固定MR-13H。
- MR-13K唯一scientific change：保留完整40D earliest maximum High、相同breach diagnostics與相同adverse-to-peak component計算，但Target不再扣除adverse；公式固定為`favorable_return / abs(max_adverse_return)`，目前canonical R scale仍為10%。因此candidate-reference Target差值必須逐筆精確等於MR-13H adverse R penalty。
- 固定條件：daily eligible stock-days、300×10 stock+0050 input、`inception_time_v1`、40D horizon、Full-list Delta-NDCG weighted RankNet pairwise loss、mean-daily-Spearman epoch selection、all-label scope與Seed42全部沿用MR-13H。不得加入strategy state、K、R0、cash、holdings或cross-sectional architecture。
- Label Audit已泛化為受控Target-pair contract：MR-13H vs MR-13E仍驗no-breach invariant；MR-13K vs MR-13H則驗selected peak/favorable/adverse/breach diagnostics完全相同，且Target delta=removed adverse R。Audit只讀、不訓練、不設GO/REJECT magic threshold。
- Stage governance：MR-13K `selection_pit_authorized=False`，目前只允許`[1] 模型訓練 → [6] Target comparison`與`[1] 訓練目前模型 → Forward-OOS`。Forward結果出來後才決定PIT，不預先建立13K strategy arm。Production/runtime維持MR-13E C42/C44。

### 2026-08-17 — MR-13K Target Audit + Seed42 Forward GO → Selection PIT authorized

- 程式基準：`test-branch-1_20260817_161310_f873c12(1).zip`，SHA256=`589a2ec313e5bc399413c6a3fbe46f19752395378e782f8a76a9f856c96e910c`。本輪只推進research stage與current comparison matrix；不改MR-13K Target、architecture、loss、Seed、K/R0、exact solver或production C42/C44。
- Target comparison正式結果：common stock-days=`1,578,349`、40D risk breach=`31.31%`、Target changed=`86.92%`、candidate-vs-MR-13H Daily rank correlation=`0.8609`、Top-10 overlap=`90.23%`、Mean abs delta=`0.3348R`。這符合pure-MFE移除adverse penalty造成大量數值改變、但大部分cross-sectional order仍保留的受控Label變化。
- Seed42 Forward正式結果：selected epoch=`1`；all-stock OOS Daily/Global rho=`0.3833/0.2659`、Pair=`63.19%`、Top-K Lift=`+1.6401R`；breakout-candidate slice Daily rho=`0.3621`、Pair=`64.09%`。相對MR-13H Seed42 self-target `0.2332/58.04%`，MR-13K的learnability明顯提高。
- 關鍵反證檢查：同一MR-13K Forward score對MR-13H reference Target仍有Daily rho=`0.2381`、Pair=`58.24%`，略高於MR-13H自身Seed42 Forward `0.2332/58.04%`。因此目前證據不支持「只是把Target變容易、同時犧牲原本opportunity ranking」這個解釋；但pure-MFE刻意忽略adverse path，是否轉成更好的portfolio economics仍只能由PIT strategy replay回答。
- Decision：`MR-13K = FORWARD_OOS_MODEL_GO / SELECTION_PIT_AUTHORIZED`。`ContinuousRankerResearchSpec.selection_pit_authorized=True`，新增active `DL-CONT13K-PIT / SR-C53`。C53與C42固定完全相同historical Min params、all-off、K/R0、canonical sizing/cash/orderability/execution與`exact_branch_and_bound_v1`，唯一替換MR-13E PIT score為MR-13K PIT score；primary contrast=`C53-C42`。
- Minimal-stage governance：MR-13H已以16-seed完整流程結案為`VALID_LABEL_SIMPLIFICATION / NOT_SELECTED_FOR_PROMOTION`，所以`CONT13H[_PIT] / C51 / C52`退出current config並移至historical compatibility。MR-13K尚無Selection strategy證據，因此不預先建立Forward strategy arm，也暫停Selection/Forward multi-seed robustness；先以C53回答pure-MFE是否能把更高ranking learnability轉成portfolio增量。
- Production/runtime不變：Selection C42／Forward C44仍是current production candidate。下一步正式操作為`apps/research.py → [1] 模型訓練 → [2] 建立／更新 Selection PIT Scores`完成PIT與Model Gate；之後再到`[3] 策略組合比較`執行Selection PIT目前設定。合法完整PIT若Model Gate品質FAIL，依current研究治理以WARN揭露且C53仍可做受控strategy conversion；missing/invalid PIT仍BLOCK。

### 2026-08-17 — MR-13K Selection PIT Model Gate PASS → C53 strategy translation ready

- 程式基準：`test-branch-1_20260817_174846_9cffdd4.zip`，SHA256=`dd2ecaef67481284ed64cd0f0b183739d2810569404fb756390f7a19b38f59d9`。本輪不改MR-13K Target、architecture、loss、Seed、PIT fold contract、K/R0、exact solver或production C42/C44；只記錄正式PIT結果並修正App簡易報表在audit尚未產生前的coverage顯示。
- Selection PIT正式結果：自動解析最早合法score date=`2013-04-01`；period=`2013-04-01～2020-12-31`、folds=`8`、score groups=`778,532`、coverage=`100.00%`。首fold best epoch=`3`，其餘7 folds best epoch=`1`；各fold Validation rho依序=`0.1221/0.1861/0.2909/0.2836/0.2467/0.2187/0.2946/0.3368`。
- PIT Model Gate：All eligible stock-days Global/Daily rho=`0.1869/0.2877`、Pair=`60.13%`、Top-Bottom spread=`+0.8284R`；年度rho>0=`8/8`、spread>0=`8/8`。Breakout slice Global/Daily rho=`0.1970/0.2081`、Pair=`59.55%`、Top-K Lift=`+0.2519R`、Boundary=`54.92%`。四項事前model Gate方向條件全部PASS。
- `drift=True`只代表至少一組相鄰fold score平均值位移達`>=1.0 pooled score SD`；current Gate刻意只判斷global/daily ranking正向與多數年度rho/spread正向，因此drift是calibration/distribution diagnostic，不是strategy replay blocker。若C53後續策略結果異常，才有理由回頭讀fold drift明細；目前不新增Audit。
- 相對MR-13E歷史Selection PIT模型層：MR-13K Daily rho=`0.2877` vs MR-13E=`0.1627`，Pair=`60.13%` vs `55.75%`，breakout Daily rho=`0.2081` vs `0.1019`，顯示Pure-MFE在PIT條件下的ranking learnability提升仍存在；但這不等於portfolio economics改善，最終只由C53-C42 controlled replay判定。
- Decision：`MR-13K = SELECTION_PIT_MODEL_GATE_PASS / SELECTION_STRATEGY_READY`，`DL-CONT13K-PIT = RESULT_AVAILABLE / ACTIVE_RESEARCH_SOURCE`，`SR-C53 = READY_FOR_SELECTION_STRATEGY_REPLAY`。下一步只執行`apps/research.py → [3] 策略組合比較 → Selection PIT 策略比較 → 執行目前比較設定`，Primary contrast=`C53-C42`。在C53結果出來前，不建立MR-13K Forward arm、不開multi-seed robustness、不改Target或selector。
- App層輸出修正：首次build PIT時audit JSON尚未存在，`build-point-in-time-scores`簡易報表原本因此顯示`Score coverage=- / Scored groups=None`；改為只在build command且audit coverage尚不存在時，直接讀同一canonical `selection_point_in_time_manifest.json.coverage`。Daily/Global rho仍只在audit完成後由audit canonical payload顯示，不另算第二套指標。

### 2026-08-17 — MR-13K C53 Selection result → full Forward + 16-seed robustness authorized

- 程式基準：`test-branch-1_20260817_184430_4ffd959(1).zip`，SHA256=`54ba69db9821793d97cd948dfc95158ea74a50f2ccca7425630ec471a87d74e7`。本輪不改MR-13K Target、architecture、loss、K/R0、exact solver、cash/execution或production C42/C44；只依使用者明確要求擴充完整研究評估矩陣。
- C53 Seed42 Selection正式結果：Return=`143.55%`、MDD=`19.11%`、RoMD=`7.51`、Annual=`13.56%`、EV=`0.70R`、DL選擇R=`100.72R`；C42 reference為Return=`199.32%`、MDD=`19.88%`、RoMD=`10.03`、Annual=`16.96%`、EV=`0.75R`、DL選擇R=`117.16R`。MR-13K模型層Daily rho/Pair明顯較高，但Seed42 Selection portfolio translation低於MR-13E。
- 使用者決策：不在單次Selection提前結案，要求把Forward-OOS與robustness全部測完。因此新增`DL-CONT13K / SR-C54` Forward source-only exact arm；C54與C44除MR-13K vs MR-13E frozen Forward source外，Min/all-off、K/R0、canonical sizing/cash/orderability/execution與`exact_branch_and_bound_v1`完全固定。
- Current Forward matrix=`C1/C3/C44/C54`，primary source-only contrast=`C54-C44`。Selection robustness stochastic=`C42/C53`、paired=`C53-C42`；Forward robustness stochastic=`C44/C54`、paired=`C54-C44`；fixed references分別為Selection `C32/C23`、Forward `C1/C3`。
- Robustness完全沿用current config，不覆寫使用者設定：`seed_count=16`、`seed_generator_seed=20260810`、deterministic generated seeds、不挑best seed、不ensemble、completed unit可reuse。Production/runtime candidate仍固定Selection C42／Forward C44。
- 正式執行順序：先`apps/research.py → [3] 策略組合比較 → Forward-OOS 策略比較 → 執行目前比較設定`取得C54 single-seed；再由同一策略比較選單的`一次執行全部 Multi-seed robustness`依config串行完成Selection與Forward robustness。完成兩stage後再依single-seed＋paired 16-seed整體證據結案MR-13K。

### 2026-08-17 — MR-13K C54 Forward single-seed result

- Forward-OOS controlled replay期間=`2021-01-01～2026-03-02`；C54與C44固定同一Min params、all-off rules、K/R0、canonical sizing/cash/orderability/execution與`exact_branch_and_bound_v1`，唯一變因是frozen DL source `MR-13K` vs `MR-13E`。
- C54結果：Return=`168.67%`、MDD=`19.95%`、RoMD=`8.45`、Annual=`21.12%`、worst full year=`-12.07%`、Log R²=`0.8384`、monthly win=`60.32%`、win rate=`42.55%`、Payoff=`3.41`、EV=`0.80R`、trades=`329`、average exposure=`92.96%`。
- C44 reference：Return=`214.29%`、MDD=`15.13%`、RoMD=`14.17`、Annual=`24.86%`、worst full year=`-4.07%`、Log R²=`0.9090`、monthly win=`69.84%`、win rate=`46.13%`、Payoff=`3.40`、EV=`1.03R`、trades=`310`、average exposure=`92.80%`。
- 模型ranking與portfolio economics呈反向：C54 Daily/Global rho=`0.383/0.266`、Pair=`63.19%`、Top-Bottom=`1.74R`，高於C44的`0.215/0.104`、`57.42%`、`0.82R`；但C54 Return/RoMD/EV/MDD均劣於C44。Payoff幾乎相同（`3.41 vs 3.40`）而win rate下降（`42.55% vs 46.13%`），支持Pure-MFE能抓upside magnitude但缺少path realizability資訊的機制解釋。
- 年度C54-C44：2021=`8.85%-19.53%`、2022=`-12.07%-(-4.07%)`、2023=`110.58%-85.87%`、2024=`10.69%-22.51%`、2025=`10.30%-11.64%`、2026 YTD=`9.19%-7.82%`；13K具有regime-dependent upside capture，但不是跨期穩定優勢。
- Decision：單次Selection與Forward均不支持MR-13K promotion，但依使用者明確要求不在single-seed提前結案；current Selection `C42/C53`與Forward `C44/C54` 16-seed paired robustness仍照既定設定完成，再做13K最終判讀。Production/runtime仍固定C42/C44。

### 2026-08-17 — MR-13K mechanism diagnosis → MR-13L decomposed MFE/adverse learning formulation implemented

- 程式基準：`test-branch-1_20260817_192304_690164e.zip`，SHA256=`a4bbc11ff7451400736314a37c9d3453a686d71797c1d02986ca17849f765da1`；fresh extraction=`/tmp/mqp13k_next`（僅GPT工作環境）。依`doc/PROJECT_SETTINGS.md`先讀Registry/Log後才新增identity；本輪不執行`apps/run_bundle.py`或`apps/test_suite.py`。
- 研究動機：MR-13K Seed42已形成清楚mechanism split：Pure-MFE self-target learnability大幅提高（Forward Daily rho=`0.3833`、Pair=`63.19%`；Selection PIT Daily rho=`0.2877`、Pair=`60.13%`），但C53 Selection與C54 Forward的portfolio translation均低於同條件MR-13E exact reference。這支持「upside signal可學，但path realizability資訊不能直接刪除」；下一個最小問題是MR-13H難學是否來自把性質不同的MFE與adverse直接壓成單一scalar ranking target，而不是adverse component本身完全不可學。
- 新identity：`MR-13L / PROFILE-daily_universal_full_horizon_mfe_adverse_dual_mse / ARCH-inception_time_v1`。Reference固定MR-13H；Target仍是`daily_full_horizon_opportunity_r_v1`。MR-13L不是新的economic Target，也不是architecture變更。
- 唯一scientific change：原MR-13H直接以pairwise ranking學`MFE R - adverse-to-peak R`；MR-13L改用`daily_dual_component_r_regression / dual_mse_raw_r`，將InceptionTime既有兩個output neurons分別定義為`predicted adverse-to-peak R`與`predicted favorable MFE R`。兩者都是primary supervised outputs，loss固定為兩分量MSE的同一mean reduction；沒有auxiliary loss、沒有auxiliary weight、沒有lambda／penalty coefficient掃描。正式model score固定為`Predicted favorable MFE R - Predicted adverse R`，epoch selection仍只依recombined score對同一MR-13H Target的Validation mean daily Spearman。
- Dataset contract：canonical daily feature-eligible stock-day universe、300×10 lazy feature window、40D full horizon、fixed canonical risk-budget R scale、Seed policy與`inception_time_v1`全部固定。`daily_ranker_data.py`只把已存在的`target_favorable_return / target_adverse_return_to_peak`按同一fixed risk-budget轉成`target_favorable_r / target_adverse_r`供primary regression；兩分量逐筆重組必須等於同一MR-13H raw Target。Future component targets只進training/evaluation，不進runtime feature或decision-time input。
- 與舊v9 auxiliary-head實驗的差異：v9是binary classifier主目標外掛`mfe_progress / mae_risk_fraction / reward_risk_progress` auxiliary loss，且有額外loss weight；MR-13L則直接把兩個物理R分量本身設為primary objective，沒有binary主loss、沒有auxiliary branch、沒有調權重。因此不是重做已REJECT的v9 auxiliary路線。
- 報表／artifact contract：Forward OOS score額外寫出`predicted_favorable_r / predicted_adverse_r / predicted_composite_r`；report新增Validation、Forward OOS與breakout-candidate slice的兩分量RMSE/MAE/bias/global rho/daily rho。Frozen model manifest/report透過`dual_component_r_regression_contract`固定output mapping、equal-mean loss與runtime score semantics。
- Stage governance：`selection_pit_authorized=False`。MR-13L目前只授權Seed42 Forward model Gate，不建立PIT DL source、不建立Strategy Compare arm、不改production C42/C44。只有當adverse component呈現可重現正向signal，且recombined same-target ranking對MR-13H形成可信增量時，才討論Selection PIT；否則直接在模型層結案。MR-13K既有C42/C53與C44/C54 16-seed full-flow robustness是獨立current strategy evaluation，仍照使用者既定決策完成，不因MR-13L implementation提前停止。

### 2026-08-17 — MR-13L Seed42 Forward result → rejected at model Gate

- 程式基準：`test-branch-1_20260817_213957_d6bddc0.zip`，SHA256=`c5c96c26ad643ba6c75e3f4ab630986974c454c14a137090c630b05026881c37`。使用者完成`daily_universal_full_horizon_mfe_adverse_dual_mse / MR-13L` Seed42 Forward training；Target、40D horizon、fixed R scale、daily universe與`inception_time_v1`均維持MR-13H不變。
- Training：epoch1 Train Loss=`0.614691`、選模Validation Daily rho=`0.0227`為最佳；epoch2 Train Loss降至`0.322042`但Validation Daily rho降至`0.0218`，patience=1後停止，selected epoch=`1`。完整Selection refit固定1 epoch。Final refit後原Validation Daily rho=`0.3069`因該rows已被納入final refit，只是描述性in-sample metric，不是選模或泛化證據。
- Frozen Forward all-stock：Daily rho=`0.1052`、Global rho=`0.0254`、Pair=`53.51%`；Top 10% Target=`0.9142R`、Bottom 10%=`0.9292R`，故Top-Bottom=`-0.0150R`；NDCG@10=`0.5243`、Top-K Target=`1.2386R`、Lift=`+0.2274R`、Oracle overlap=`4.02%`、Boundary=`48.62%`、Boundary gap=`-0.0386R`。
- Breakout-candidate OOS：groups=`17,346`、Daily/Global rho=`0.0647/-0.0301`、Pair=`52.82%`；Top 10%=`1.1016R`、Bottom 10%=`1.3576R`，Top-Bottom=`-0.2560R`；NDCG@10=`0.6779`、Top-K Lift=`+0.1196R`。
- Controlled comparison：MR-13L與MR-13H使用完全相同economic Target。MR-13H Seed42 Forward all-stock Daily rho=`0.2332`、Pair=`58.04%`、Top-K Lift=`+1.3400R`，因此13L的dual raw-R MSE formulation在same-target ranking明顯退化。更強的反證是MR-13K雖只訓練Pure-MFE Target，其frozen score對MR-13H reference Target仍有Daily rho=`0.2381`、Pair=`58.24%`，亦明顯高於13L直接分解同一MR-13H Target後的`0.1052/53.51%`。
- Interpretation：這次結果否決的是「以兩個primary raw-R MSE heads等權回歸，再直接相減即可改善MR-13H learnability」這個learning formulation；不能僅由composite失敗推論adverse-to-peak本身不可預測。component-level MFE/adverse individual rho/RMSE已由詳細`continuous_ranker_report.md`保存，但使用者本次貼出的console沒有列出精確component值，故本Log不自行補值。
- Decision：`REJECTED_AT_FORWARD_MODEL_GATE / NO_PIT / NO_RUNTIME_DL_SOURCE`。不建立MR-13L PIT、不做Selection Strategy Compare、不做MR-13L multi-seed。Production/runtime仍固定MR-13E + Min/all-off + exact K/R0；MR-13K既定C42/C53與C44/C54 robustness仍獨立完成。
- Current config reconciliation：本輪最新基準`config/strategy_compare.py`的正式SSOT目前為`STRATEGY_COMPARE_ROBUSTNESS_SEED_COUNT=8`、`STRATEGY_COMPARE_ROBUSTNESS_SEED_GENERATOR_SEED=20260810`。因此後續由`[5] 一次執行全部 Multi-seed robustness`啟動的新run應以8 seeds為準；先前Log中「16-seed authorized」保留為當時研究計畫歷史，不得反向覆寫目前config。
- 下一個最小研究問題：先隔離「adverse-to-peak是否具有可重現cross-sectional ranking signal」，而不是再調dual-MSE權重或回歸loss。若下一個實驗啟動，優先使用單一adverse-only、same-day pairwise ranking的受控模型，固定現有features/architecture/horizon/risk scale；只有adverse-only Forward signal成立，才有理由研究與已證實可學的Pure-MFE ranking如何組合。

### 2026-08-17 — MR-13M Seed42 Forward result → path-risk learnability confirmed

- 程式基準：`test-branch-1_20260817_221751_79dc1d2.zip`，SHA256=`8e800f703d825bdbfcf165f9b3f6e8023be9570db347bbee6cbb533d5074a3f3`；fresh extraction=`/tmp/mqp13m_result`（僅GPT工作環境）。MR-13M維持MR-13H/13K相同40D full horizon、earliest max-MFE peak、fixed canonical R scale、daily feature-eligible stock-day universe、300×10 lazy sequence、InceptionTime與Full-list Delta-NDCG RankNet；唯一Target=`-adverse-to-peak R`。
- Training：epoch1 Validation Daily rho=`0.2861`；epoch2=`0.3443`為最佳；epoch3=`0.3270`，selected epoch=`2`。完整Selection refit固定2 epochs。
- Frozen Forward all-stock：Daily/Global rho=`0.3558/0.2999`、Pair=`62.75%`；Top 10% Target=`-0.2069R`、Bottom 10%=`-0.5961R`，故Top-Bottom=`+0.3892R`；NDCG@10=`0.7616`、Top-K Target=`-0.1190R`、Lift=`+0.2300R`、Oracle overlap=`6.79%`、Boundary=`48.44%`、Boundary gap=`-0.0066R`。
- Breakout-candidate OOS：groups=`17,346`、Daily/Global rho=`0.3388/0.3737`、Pair=`63.70%`；Top 10%=`-0.1413R`、Bottom 10%=`-0.6269R`，Top-Bottom=`+0.4856R`；NDCG@10=`0.7859`、Top-K Lift=`+0.1115R`、Oracle overlap=`64.25%`、Boundary=`54.90%`、Boundary gap=`+0.0782R`。
- Controlled interpretation：MR-13K Pure-MFE Forward Daily rho=`0.3833`、Pair=`63.19%`；MR-13M low-adverse=`0.3558/62.75%`，兩個分量都具有很強且相近的cross-sectional learnability。MR-13H直接排序`MFE R - adverse R`只有Daily rho=`0.2332`、Pair=`58.04%`，因此13H較難學不能再歸因於adverse component沒有signal；更合理的機制是假說為「兩個各自可學、但在raw-R差值所誘導的composite ordering中形成較難預測的trade-off/residual」。此機制仍屬推論，尚未由component joint-distribution Audit直接證實。
- Decision：`PATH_RISK_LEARNABILITY_CONFIRMED / COMPONENT_FORWARD_GO / NO_STANDALONE_PIT`。MR-13M不建立standalone PIT或strategy arm，因其只獎勵path safety而沒有upside magnitude，單獨portfolio結果不能回答最終選股目標。下一個最小必要實驗應保留MR-13K與MR-13M已證實的個別rank signal，將MFE與low-adverse轉成同日共同rank scale後作固定等權組合；不掃lambda、不改K/R0/cash/execution。若該組合仍無法改善同一MR-13H economic Target的Forward ranking，才考慮更複雜的multi-head／score ensemble；不要先做MR-13M standalone PIT。
- MR-13K既有Selection `C42/C53`與Forward `C44/C54` current 8-seed paired robustness仍是獨立研究線，照current config完成；production/runtime仍固定MR-13E C42/C44。

### 2026-08-17 — MR-13M result → MR-13N fixed equal-rank MFE + low-adverse experiment implemented

- 程式基準：`test-branch-1_20260817_224751_faf8b5e.zip`，SHA256=`2d7cde7a9f8675168ae6eef92749eabb9cbdadd86b66b965a6134fc6e26c1820`；fresh extraction=`/tmp/mqp13n`（僅GPT工作環境）。依`doc/PROJECT_SETTINGS.md`先讀Registry/Log並先在Registry占用`MR-13N`後才修改程式；本輪不執行`apps/run_bundle.py`或`apps/test_suite.py`。
- 待決策問題：MR-13K Pure-MFE Forward Daily rho=`0.3833`、Pair=`63.19%`；MR-13M low-adverse Forward Daily rho=`0.3558`、Pair=`62.75%`，證明兩個component個別都具強cross-sectional learnability。MR-13H直接排序raw `MFE R - adverse R`只有Daily rho=`0.2332`、Pair=`58.04%`，MR-13L raw-R dual MSE再相減更降至`0.1052/53.51%`。下一個最小問題是把兩個可學component放到共同rank scale後固定等權組合，是否能避免raw-R magnitude/calibration trade-off。
- 新identity：`MR-13N / PROFILE-daily_universal_full_horizon_equal_rank_mfe_low_adverse_full_list_ndcg_pairwise / TARGET-daily_full_horizon_equal_rank_mfe_low_adverse_v1 / ARCH-inception_time_v1`。
- Target contract：每個target-valid stock-day保留MR-13K/MR-13M相同40D complete horizon、earliest maximum-MFE peak、risk diagnostics與fixed canonical R scale。每天先各自建立`MFE R`與`-adverse-to-peak R`的average-rank percentile `[0,1]`，再固定`0.5 × MFE percentile + 0.5 × low-adverse percentile`；single-stock day兩component percentile固定`0.5`。權重是實驗定義，不由Train/Validation/OOS調整，不掃lambda。
- Training contract：daily feature-eligible universe、300×10 lazy sequence、`inception_time_v1`、Adam、Full-list Delta-NDCG weighted RankNet、mean-daily-Spearman epoch selection、all-label scope與Seed42全部沿用MR-13H/K/M。pairwise training對MR-13N composite ordering本身學習；`MR-13H daily_full_horizon_opportunity_r_v1`只透過新增`evaluation_reference_profile_name`在checkpoint固定後作同一frozen score的reference OOS／breakout diagnostics，不進loss、gradient、early stopping或epoch selection，也不啟用Target comparison menu。
- SSOT調整：same-date percentile計算抽到`filters/breakout_quality/continuous_ranker_data.py::build_same_date_percentile_targets()`；既有trainer API `build_daily_percentile_targets()`只作compatibility façade，MR-13N component rank與既有daily ranker percentile共用同一實作，避免排名口徑分叉。
- Stage governance：`selection_pit_authorized=False`；目前只跑Seed42 Forward model Gate，不建立PIT、DL source或Strategy Compare arm，不改MR-13K current 8-seed robustness，也不改production MR-13E C42/C44。Forward審查同時看MR-13N self-target ranking與同一frozen score對MR-13H economic reference Target的Daily rho／Pair／Top-K；只有reference economic ordering形成可信增量才考慮PIT。

### 2026-08-17 — MR-13N Seed42 Forward result → rejected at model Gate

- 程式基準：`test-branch-1_20260817_230217_ea9903d.zip`，SHA256=`edf4033d43f3ecd60091198d4a6ad1c105f39d83c8eddfccd6aad96f46c2ba22`；fresh extraction=`/tmp/mqp13n_result`（僅GPT工作環境）。MR-13N維持MR-13H/K/M同一40D complete horizon、earliest max-MFE peak、daily feature-eligible universe、300×10 lazy sequence、InceptionTime與Full-list Delta-NDCG RankNet；Target固定=`0.5 × same-date MFE percentile + 0.5 × same-date low-adverse percentile`，無lambda/weight tuning。
- Seed42 result：selected epoch=`2`；獨立Validation Daily rho=`0.0437`。Frozen Forward all-stock self-target Daily/Global rho=`0.0150/0.0114`、Pair=`50.53%`、Top10=`0.5039`、Bottom10=`0.4931`、Top-Bottom=`+0.0108`、Top-K Lift=`-0.0124`；breakout slice Daily/Global rho=`-0.0085/-0.0211`、Pair=`50.82%`、Top-K Lift=`+0.0038`。
- Economic reference：同一frozen MR-13N score對MR-13H `MFE R - adverse R` reference Target只有Daily rho=`0.0053`、Pair=`50.25%`，沒有形成economic ordering。Target與trainer方向已核對：MFE percentile與`-adverse` percentile皆higher-is-better，固定0.5/0.5，故不是sign/percentile implementation bug。
- Decision：`REJECTED_AT_FORWARD_MODEL_GATE / NO_PIT / NO_RUNTIME_DL_SOURCE`。不建立MR-13N PIT、不做Strategy Compare、不做multi-seed。此結果否決的是「讓單一模型直接學fixed equal-rank composite ordering」；尚不能單獨區分equal-rank Target geometry本身是否無效，或兩個各自可學task在single-model joint ordering中互相干擾。

### 2026-08-18 — AUD-mr13km-frozen-rank-fusion result → fixed equal-rank expert fusion rejected

- 程式基準：`test-branch-1_20260818_001037_841d25c.zip`，SHA256=`ae6c2327c0bc84bf2c757a3794cfeecb737ab4e6c74f09862c671ec0955dd1aa`；fresh extraction=`/tmp/mqp_fusion_audit`（僅GPT工作環境）。
- Audit期間=`2021-01-04～2025-12-22`，Common rows=`608,204`；完全只讀既有MR-13K / MR-13M frozen Forward scores，不重訓、不fit權重、不重跑策略。
- Economic-reference結果：MR-13K對MR-13H economic Target Daily rho/Pair/Top-K Lift/Top-Bottom=`0.2381/58.24%/+1.3117R/+1.4179R`；MR-13M=`-0.2382/41.73%/-0.7746R/-1.4914R`；固定`0.5/0.5` same-date rank fusion=`0.0056/50.12%/-0.2777R/-0.1403R`。依事前stop rule判定`REJECT_EQUAL_RANK_SCORE_FUSION`。
- 機制判讀：true MFE vs true low-adverse component的mean Daily rho只有`-0.1261`，顯示真實component只存在輕度trade-off；但MR-13K vs MR-13M frozen score的mean Daily rho=`-0.8649`，代表兩個模型從現有300×10 sequence中抽到的「可預測部分」幾乎成為鏡像反向。與此同時，true equal-rank Target對MR-13H economic Target的mean Daily rho=`0.8537`，因此不能把MR-13N / frozen fusion失敗歸因於equal-rank economic geometry本身；失敗更精確地位於learned representation／predictable-component interaction。
- Decision：不做fixed equal-rank multi-expert score fusion、不為13K/13M掃fusion weight，也不讓13N進PIT。下一個受控模型若繼續此主線，應避免再對兩個scalar expert score做線性加權；較乾淨的候選是假設「只在MFE與low-adverse同方向的Pareto-comparable pairs提供pairwise supervision」，直接學joint dominance，而非再選一個magnitude/rank mixing coefficient。此候選尚未建立MR identity，需另輪正式實作前再查Registry。
- Lifecycle：Audit待決策問題已完全回答；`AUD-mr13km-frozen-rank-fusion`的formal config/catalog/implementation與dedicated synthetic依`PROJECT_SETTINGS C11`退役，結果只保留於Registry／Log。Generic Audit framework與research utilities仍保留。

### 2026-08-18 — MR-13O Seed42 Forward result → weak Pareto edge, rejected at model Gate

- 程式基準：`test-branch-1_20260818_100207_a009b17.zip`，SHA256=`82a44233325d97281f4a62ccfc59b8f7a777a2ad3d2a84f349357de5e88f3d74`。MR-13O固定MR-13H economic Target作描述性評估，training只使用strict Pareto-comparable MFE/low-adverse pairs，epoch selection只看Validation Pareto concordance。
- 結果：selected epoch=`1`；Validation Pareto=`51.14%`、selected-epoch economic rho=`0.0896`；Forward economic Daily/Global rho=`0.0782/0.1008`、Pair=`52.59%`、Top10−Bottom10=`+0.6042R`、Top-K Lift=`+0.1120R`；Forward Pareto=`51.81%`。Breakout economic Daily rho=`0.0376`、Pareto=`50.30%`。
- 判讀：Validation→Forward Pareto沒有明顯崩壞，代表不是典型overfit；但edge穩定地只有約`+1.8pp`，breakout幾乎隨機，且economic ordering遠低於MR-13H/K。`REJECTED_AT_FORWARD_MODEL_GATE / NO_PIT / NO_RUNTIME_DL_SOURCE`。至此停止繼續修改MFE/adverse scalar Target、equal-rank、Pareto pair mask或loss小改。
- 下一研究分流：方案A＝固定MR-13H Target，只改parameter-free path-dynamics input representation；方案B＝保留13K/13M specialist models，在portfolio layer分工。使用者決定先進行方案B。

### 2026-08-18 — SR-C55 Forward result → raw MR-13M safety floor stops; B2 residual-safety selected

- 程式／結果基準：`test-branch-1_20260818_131753_116a310.zip`，SHA256=`aa725f0dca960a0b3285343f87dafe3cf04abc987df085f8be235e9b2092a2fe`。Forward期間=`2021-01-01～2026-03-02`。
- C55相對C54：Return=`97.87% vs 168.67%`、MDD=`12.03% vs 19.95%`、RoMD=`8.13 vs 8.45`、Annual=`14.15% vs 21.12%`、Win Rate=`44.69% vs 42.55%`、Payoff=`2.77 vs 3.41`、EV=`0.75R vs 0.80R`。年度2022由`-12.07%`改善為`+0.38%`，但2023由`110.58%`降為`29.41%`。
- Selection-translation diagnostic同方向：C55 Target mean R=`0.74R`、Target percentile=`0.398`、Top-K=`7.47%`，均低於C54的`1.01R/0.486/8.32%`；Average exposure近似`92.32% vs 92.96%`，因此不是單純資金沒投出去，而是raw 13M basket floor改變membership後把部分高-upside winners一併排除。
- Decision：`SR-C55 = RESULT_AVAILABLE / STOP_RAW_SAFETY_FLOOR / NO_SELECTION_PIT / ROBUSTNESS_OFF`。13M確實有降path-risk效果，但直接要求整basket raw safety不低於Min baseline過硬；不建立13M Selection PIT，不做C55 robustness。
- 下一個最小controlled strategy experiment改為Plan B2：保留13K作唯一objective，先在同日候選橫截面把13M相對13K正常反向關係residualize，只用「同upside條件下異常安全／危險」的Residual Safety作baseline-relative hard floor。

### 2026-08-18 — SR-C56 Seed42 Forward result → residual safety works; B2 strategy-layer concept GO, no production promotion

- 結果來源：Forward-OOS run fingerprint=`0ed9def6972c`，期間=`2021-01-01～2026-03-02`；current arms=`C1/C3/C44/C54/C55/C56`。C56實際RUN，C1/C44/C54/C55合法REUSE。
- C56相對C54：Return=`148.34% vs 168.67%`（保留約`87.9%`）、MDD=`12.86% vs 19.95%`（`-7.09pp`）、RoMD=`11.53 vs 8.45`（`+3.08`）、Annual=`19.29% vs 21.12%`、Worst Full Year=`-3.19% vs -12.07%`、Win Rate=`47.90% vs 42.55%`（`+5.35pp`）、Payoff=`2.90 vs 3.41`、EV=`1.14R vs 0.80R`（`+0.34R`）、Trades=`309 vs 329`、Average Exposure=`92.98% vs 92.96%`。因此改善不是靠降低資金使用，而是membership/path-quality改變。
- C56相對C55 raw-safety：Return=`148.34% vs 97.87%`、MDD=`12.86% vs 12.03%`、RoMD=`11.53 vs 8.13`、Win Rate=`47.90% vs 44.69%`、Payoff=`2.90 vs 2.77`、EV=`1.14R vs 0.75R`。Residualizing 13M明顯修復C55把高-upside winner一併排除的問題，只用約`+0.83pp` MDD換回大量Return與RoMD。
- Conversion diagnostic：C56 `DL選擇R=+1.27R`、`RCE=+21.81%`，相對C54的`-87.57R/-60.07%`大幅改善；但Pure-MFE Target mean/percentile=`0.88R/0.440`仍低於C54的`1.01R/0.486`。這表示Residual Safety的價值主要不是把13K Pure-MFE Target本身推得更高，而是改善13K未刻畫的path／realized conversion。
- 年度：C56相對C54在2021、2022、2024、2026較好，2023顯著少賺（`63.45% vs 110.58%`），2025亦較弱；因此B2形成可信的upside-vs-path-risk trade-off，而不是全面支配C54。
- Decision：`SR-C56 = RESULT_AVAILABLE / FORWARD_GO_STRATEGY_LAYER_CONCEPT_CONFIRMED / NOT_PROMOTED / NO_SELECTION_PIT / ROBUSTNESS_OFF`。B2待決策問題已回答「YES」：同upside條件下的Residual Safety有實際portfolio conversion價值；但C56仍低於production C44的Return/RoMD（C44=`214.29%/14.17`），且只有Seed42 Forward證據，因此不改production C42/C44、不加入current 8-seed robustness。下一主線依Research Queue轉Plan C-M：13K PIT-safe upside context → Conditional Low-Adverse model。

### 2026-08-23 — First-Passage evidence後改採 Conversion-first 研究優先序

- 使用者實機 current Extending-Window single-seed evidence 已完成。OOS：C59=`166.49% / MDD 15.92% / RoMD 10.45 / EV 1.24R`，C60=`150.20% / 12.90% / 11.64 / 1.14R`；Rolling：C59=`201.10% / 15.92% / 12.63 / 1.11R`，C60=`136.12% / 15.54% / 8.76 / 0.81R`。
- First-Passage 的 actual initial-stop-before-upside 指標六個 OOS／Rolling × `+1R/+2R/+3R` 比較皆為 C59 較低。最具辨識力的 `+2R前初始Stop`：OOS C59=`3/34 (8.82%)`、C60=`9/40 (22.50%)`；Rolling C59=`6/41 (14.63%)`、C60=`9/37 (24.32%)`。這支持「full-horizon upside 可學，但較低 adverse 本身不等於較高可實現轉換；survivability / risk-truncated upside 更接近策略需要」的研究假說。
- 使用者研究優先序決策：**先解決模型訊號到實際交易的轉換率；只有找到相對 C59 更好的 candidate，才投入 Fixed-Window Stability 或 Extending-Window Multi-seed Robustness。** 因此既定 robustness 長跑與 Fixed-Window 暫時降為 conversion candidate 之後的 validation gate，不再作目前開發前置。
- 下一個受控方向先採 provisional **Survivable Pure-MFE / risk-truncated Pure-MFE**：只隔離 `first canonical risk breach truncation`，不再額外扣 continuous adverse penalty；尚未開始實作，故**不預占 MR identity**。若它不能改善 conversion，後續才依結果判斷是否需要相同 survivable Target 的 causal path-dynamics representation、first-passage / competing-risk Target，或重新評估 Plan C-M 的 conditional adverse stacking。
- 研究治理：目前 2021→latest OOS 已屬**迭代研究 OOS 證據**，可用於人類 GO/STOP 與下一假說，但不得把 OOS rows／labels／scores／統計輸入 training、selection、threshold、normalization 或其他 computational fitting pipeline。
- Decision：**CONVERSION_FIRST / C59_SINGLE_SEED_REFERENCE / ROBUSTNESS_DEFERRED_UNTIL_BETTER_CANDIDATE / NO_NEW_ID_ALLOCATED**。

### 2026-08-23 — MR-13P Seed42 Forward training completed；Conditional Forward metrics簡報可見性修正

- 使用者已透過正式 `apps/research.py → 模型訓練 → 訓練目前模型 → Forward-OOS模型報表` 完成 MR-13P Seed42 training；耗時 `10:37.1`，selected epoch=`1`。Primary MFE Forward evidence：OOS Daily rho=`0.3953`、Pair=`63.64%`；breakout-candidate OOS Daily rho=`0.3736`、Pair=`64.41%`。選模時 Conditional Safety Validation rho=`0.3125`，顯示 secondary signal 在 Validation 可學。
- Canonical `continuous_ranker_report` 已持久化 `conditional_mfe_safety_evaluation`，其中包含 Validation／Forward OOS／breakout-candidate 的 `Primary MFE` 與 `Conditional Safety` Daily rho、Global rho、Pair concordance；但 App 層簡易 console／`simple_reports/train-continuous-ranker.md` 只顯示 primary `split_metrics`，導致 Conditional Forward Model Gate 指標雖已計算卻未在正式互動摘要顯示。
- Engineering fix：正式簡易報表新增精簡 `Conditional MFE-Safety Model Gate` 表，固定顯示三個 split × 兩個 head 的 `Daily rho / Global rho / Pair`，直接消費既有 canonical report payload，不重算 metric、不重訓模型。模型選單 `[3] 查看目前Workflow、工件與模型報表` 在既有 report 存在時也直接重畫同一摘要，因此本次既有 Seed42 結果不需要重新訓練；詳細 Top/Bottom 等仍留在 `continuous_ranker_report.md`，避免長流程洗版。
- Scientific status：**FORWARD_RESULT_AVAILABLE_PENDING_REVIEW / MODEL_GATE_ONLY**。本輪只修報表可見性；尚未依 Conditional Safety Forward metrics做 GO／REJECT，也不授權 PIT、runtime source、strategy arm、robustness 或 promotion。

### 2026-08-23 — MR-13P Forward Model Gate GO；C64 single-model conversion arm implementation

- 使用者透過正式模型報表審閱MR-13P dual-head Forward evidence。Primary MFE：Validation Daily/Global/Pair=`0.4348/0.4335/65.07%`、Forward OOS=`0.3953/0.3019/63.64%`、breakout slice=`0.3736/0.3771/64.41%`；Conditional Safety：Validation=`0.3199/0.3067/61.05%`、Forward OOS=`0.3234/0.2900/61.17%`、breakout slice=`0.2824/0.3179/62.33%`。Conditional Forward沒有validation-only collapse，Primary MFE亦未相對MR-13K既有Forward evidence明顯崩壞，因此**MR-13P Forward Model Gate = GO**。
- 同次Primary MFE OOS Top-K Lift=`+1.2486R`，未因global/daily rho改善而同步擴張；因此不把Model Gate GO解讀成策略GO，依conversion-first原則直接進最小portfolio conversion contrast，不新增模型層Audit。
- 正式配置`DL-CONT13P-ROLL / CONT13P_ROLL`與`SR-C64`。C64與C58共用Min `base-finalist-best`策略參數與既有exact constrained selector；Primary最大化MR-13P Pure-MFE head，secondary直接讀**同一PIT artifact**的`conditional_safety_score`並套`baseline_coverage_and_score_sum_floor_v1`。因secondary target已在training時對true MFE residualize，runtime明確禁止再做C60的`same_day_rank_ols_v1`。
- PIT producer擴充為single-model dual-head score contract：每個fold只訓一個MR-13P checkpoint、一次shared-encoder inference同時輸出`breakout_quality_score`/`primary_mfe_score`/`conditional_safety_score`；artifact manifest明確宣告score columns。Strategy runtime允許primary與secondary指向同一DL source，但只有明確指定不同`conditional_safety_score`欄位才合法，避免把同一primary score誤當safety。
- Current Compare Suite加入C64與`C64-C58/C64-C59/C64-C60`三個controlled contrasts；OOS/Rolling/single/multi仍共用同一suite SSOT。Robustness framework會由suite自然辨識C64為model-seed-sensitive，但**目前研究決策仍是ROBUSTNESS_DEFERRED**，不因framework可執行而自動長跑。
- C64 conversion Gate尚無策略結果。下一個最小證據是single-seed OOS：主判斷`+2R前初始Stop↓`，並共同檢查`+1R/+3R`、Full-MFE、Adverse、Realized EV、Return、RoMD；不能明顯改善C59即停止，不先做robustness/stability。
- Decision：**MR-13P FORWARD_MODEL_GATE_GO / C64 IMPLEMENTED_RESULT_PENDING / SINGLE_SEED_CONVERSION_FIRST / ROBUSTNESS_DEFERRED / NOT_PROMOTED**。

### 2026-08-24 — C64 conversion evidence：hard safety floor rejected；Conditional research continues

- 使用者完成C64 single-seed Extending OOS與Rolling。OOS C64：Return=`134.00%`、MDD=`13.20%`、RoMD=`10.15`、EV=`0.78R`；`+1/+2/+3R前初始Stop=15.38%/17.39%/7.69%`、Full-MFE=`0.82R`、Adverse=`0.19R`、Realized EV=`0.84R`。Rolling C64：Return=`180.29%`、MDD=`13.20%`、RoMD=`13.66`、EV=`1.15R`；`+1/+2/+3R前初始Stop=20.69%/16.67%/11.11%`、Full-MFE=`0.78R`、Adverse=`0.20R`、Realized EV=`1.17R`.
- 主要reference C59：OOS `+2R前初始Stop=8.82%`、Full-MFE=`0.98R`、Adverse=`0.23R`、Realized EV=`1.28R`、Return=`166.49%`、RoMD=`10.45`；Rolling `14.63% / 1.04R / 0.27R / 1.14R / 201.10% / 12.63`。因此C64在兩mode都未改善預先固定的`+2R前初始Stop`主Gate，且Full-MFE分別下降`0.16R/0.26R`；OOS Realized EV/Return亦明顯退化。
- 同時C64 Adverse在OOS/ Rolling均降至`0.19R/0.20R`，且Rolling RoMD=`13.66`高於C59=`12.63`，顯示MR-13P Conditional Safety不是無訊號；失敗集中在**basket-wide baseline-relative safety hard floor對upside侵入過強**，而非Model Gate反轉。
- 使用者決策：不切換到Survivable Pure-MFE，**持續改善Conditional路線**。下一個最小controlled experiment先固定MR-13P模型與dual-head PIT scores，只改runtime application為primary-preserving/minimum-primary-regret secondary condition；不得使用score fusion、固定權重、OOS-derived threshold或新的basket-wide safety floor。只有該runtime formulation仍不足時，才開啟下一個單一conditional model formulation實驗；未開始實作前不配置新MR/SR identity。
- Decision：**C64_HARD_FLOOR_FORMULATION_REJECTED / MR13P_MODEL_SIGNAL_RETAINED / CONDITIONAL_RESEARCH_CONTINUES / ROBUSTNESS_STILL_DEFERRED / NOT_PROMOTED**。

### 2026-08-25 — MR-13Q/R Forward Model Gate + C65/C66 single-seed conversion完成；轉入MFE × Safety target-geometry Audit

- MR-13Q Forward Conditional-MFE：OOS Daily rho=`0.3181`、Global rho=`0.1980`、Pair=`60.86%`、breakout Daily rho=`0.2818`。MR-13R Raw Safety OOS Daily rho=`0.3612`；Conditional-MFE OOS Daily rho=`0.3509`、Global rho=`0.3233`、Pair=`62.07%`、breakout Conditional-MFE Daily rho=`0.3107`。因此顯式Safety-conditioned duo-head的J learnability明顯優於single-head，但這只通過model learnability層。
- OOS conversion：C59 Return/MDD/RoMD/EV=`166.49%/15.92%/10.45/1.24R`；C65=`181.75%/14.23%/12.77/0.87R`；C66=`131.41%/16.24%/8.09/0.73R`。Full-MFE C59/C65/C66=`0.98/1.03/1.03R`，Adverse=`0.23/0.24/0.30R`，`+2R前初始Stop=8.82%/14.71%/9.09%`，Realized EV=`1.28/0.89/0.75R`。C65 OOS headline較佳但conversion quality沒有同步改善；C66雖模型最好學，Adverse反而最高。
- Rolling conversion：C59 Return/RoMD/Full-MFE/Adverse/Realized EV=`201.10%/12.63/1.04R/0.27R/1.14R`；C65=`182.22%/11.67/1.02R/0.26R/0.79R`；C66=`201.29%/12.40/1.13R/0.28R/0.90R`，C66最差完整年度=`-1.94%`、Log R²=`0.9115`。C66 Rolling接近C59 headline，但仍沒有Realized-EV dominance。
- Decision：Q/R都保留research value，MR-13R architecture的顯式Safety conditioning值得保留；但目前沒有candidate對C59形成OOS+Rolling乾淨dominance，故**不進Multi-seed robustness、不進Fixed-Window Stability**。最可能剩餘不確定性轉為target geometry：`J=U-E(U|S)`只表示「相對該Safety水準異常高MFE」，並不要求absolute Safety高。
- 下一個最小必要evidence固定為`AUD-mfe-safety-target-geometry`：只讀canonical Pure-MFE/Safety future truth與既有OOS/Rolling Strategy Compare row-level evidence，使用同日cross-sectional percentile 0.50切High/Low，比較All eligible stock-days、C58 orderable pool、C58/C59/C65/C66四象限分布與相對母體enrichment。Audit不重訓、不改target、不改selector；結果尚未取得，因此本Log不預標GO/REJECT。

### 2026-08-25 — AUD-mfe-safety-target-geometry 結果完成；原C66 HM/LS enrichment假說不支持

- 使用者由正式 `Selection／Truth Geometry` Audit 取得OOS與Rolling四象限結果。OOS母體 High-MFE=`50.05%`；C58 orderable pool=`58.75%`，但C58 final selection降至=`27.35%`，同時Low-MFE/High-Safety由pool `19.45%`升至C58 `49.24%`。C59/C65/C66 High-MFE只部分恢復為`33.22% / 36.45% / 35.99%`。
- Rolling同方向：C58 orderable High-MFE=`58.17%`，C58 final=`25.63%`，C59/C65/C66=`33.33% / 38.08% / 40.14%`；C58 LM/HS=`49.37%`。
- 原預註冊疑慮「C66因`J=U-E(U|S)`而相對母體大量富集High-MFE/Low-Safety」**不成立**：C66 HM/LS OOS=`16.56% (0.60×母體)`、Rolling=`20.07% (0.72×)`；不得事後改Gate把此Audit標成target-geometry成功。
- 新的最小不確定性集中在`orderable pool → final selection`：C58本身造成最大MFE向LM/HS位移，而C59/C65/C66使用same-parameter C58-derived exact K/R0 resource envelope。由現有final-selection表無法分辨模型Raw Top-K本身不足，或K/R0/cash feasibility在Raw→Planned階段改寫basket。
- Decision：**ORIGINAL_HM_LS_HYPOTHESIS_NOT_SUPPORTED / AUDIT_CLOSED / ONE_SHOT_IMPLEMENTATION_RETIRED / NEXT=K_R0_STAGE_ATTRIBUTION / NO_NEW_MODEL / ROBUSTNESS_STILL_DEFERRED**。

### 2026-08-25 — AUD-selection-k-r0-attribution 結果完成；Raw DL ranking強、主要MFE流失發生於resource conversion

- OOS Raw Top-K High-MFE：C59/C65/C66=`73.26/73.81/77.29%`；Planned=`35.69/38.41/39.17%`。Rolling Raw=`71.58/72.44/78.66%`；Planned=`37.28/41.94/43.91%`。Raw→Planned約流失`30–40pp` High-MFE，LM/HS同步增加約`33–36pp`。
- Raw direct-infeasible days：OOS=`97.25/98.25/95.06%`，Rolling=`95.91/97.83/92.27%`；相反地，少數direct-feasible days的Raw→Planned High-MFE變化均為`0pp`且membership retention=`100%`。因此目前主瓶頸不支持「模型Raw ranking找不到High-MFE」，而支持現行resource feasibility contract大幅改寫basket。
- K headroom：OOS約`60–71%`、Rolling約`65–72%`，median C58 K=`1`、median physical free slots=`3`；Final reserved==R0約`31–51%`，median final R0 slack約`0–0.75%`。但既有sidecar只能證明`K/R0/canonical cash`聯合direct-infeasible，不將失敗硬歸因為R0-only。
- Decision：**AUDIT_RESULT_AVAILABLE / RESOURCE_CONVERSION_BOTTLENECK_SUPPORTED / AUDIT_CLOSED / ONE_SHOT_IMPLEMENTATION_RETIRED / NEXT=SR-C67_K_FLEX_R0_PRESERVED / NO_NEW_MODEL / ROBUSTNESS_DEFERRED**。

### 2026-08-25 — SR-C68/C69 single-seed OOS + Rolling 結果；K-Flex增加capacity但conversion惡化

- OOS：C68 Return/MDD/RoMD/EV=`164.04%/14.60%/11.23/0.81R`；C69=`142.80%/17.38%/8.22/0.71R`。C68→C69 Full-MFE=`1.00→1.14R`、Adverse=`0.25→0.32R`、交易數=`326→349`、position-gap slot-days=`2403→1644`。
- Rolling：C68 Return/MDD/RoMD/EV=`178.98%/16.92%/10.58/0.98R`；C69=`74.35%/22.99%/3.23/0.49R`。Full-MFE=`1.04→1.09R`、Adverse=`0.24→0.31R`、交易數=`307→368`、position-gap=`2686→1329`。
- Interpretation：K-Flex確實解除部分C58-derived count restriction並增加實際position utilization，故不是「沒有真正增加投資」；但新增capacity沒有轉成更好經濟結果，特別Rolling大幅惡化。Portfolio average顯示upside略升、adverse同步升，但不能只由aggregate推定marginal positions本身的truth/path品質。
- Decision：**C68_SOLVER_CONTROL_COMPLETE / C69_PERFORMANCE_NOT_SUPPORTED / NO_ROBUSTNESS / NEXT=AUD-c69-marginal-position-attribution / KEEP_AUD-selection-k-r0-attribution_ACTIVE**。不訓練新模型、不移植K-Flex到C65/C66。

### 2026-08-25 — C69 marginal Audit結果；轉向MR-13R joint K/R0 ablation

- C69 K-relative extras提供足夠方向證據：OOS `K+1+` High-MFE=`72.41%`、HM/LS=`55.17%`、Full-MFE=`2.29R`、Adverse=`0.57R`、Realized EV=`-0.04R`；Rolling=`57.14%/40.00%/1.30R/0.53R/-0.05R`。放寬K確實增加upside候選，但MR-13E marginal positions大量落在Low-Safety且conversion差；因此C69不promotion。strict matched candidate-state只有OOS/Rolling各4日，僅作補充而不宣稱完整因果反事實。
- 使用者提出交互作用假說：歷史No-R0失敗可能被fixed K限制資金投入混淆；下一輪不再用MR-13E，而用目前conditional learnability較強的MR-13R/C66，先只共同移除K與R0，其餘全部不動。
- Decision：**NEXT=SR-C70 / SAME_MR13R_SCORE / NO_K / NO_R0 / NO_NEW_SAFETY_GATE / SINGLE_SEED_OOS_PLUS_ROLLING_FIRST**。

### 2026-08-25 — SR-C70 OOS + Rolling結果；No-K/No-R0把upside主要移到Low-Safety

- 使用者完成focused current suite C61/C58/C59/C64/C66/C70的OOS與Rolling。C70與C66完全相同MR-13R final Conditional-MFE ranking，只共同移除K/R0，因此`C70-C66`可直接解讀joint resource-restriction effect。
- OOS C66→C70：Return=`131.41→74.18%`、MDD=`16.24→21.82%`、RoMD=`8.09→3.40`、EV=`0.73→0.25R`、平均曝險=`91.48→57.19%`、position gap=`2020→640`。Filled High-MFE=`35.99→62.09%`、HM/LS=`16.56→42.58%`、LM/HS=`39.17→15.93%`，但HM/HS幾乎不動=`19.43→19.51%`、High-Safety=`58.60→35.44%`。Full-MFE=`1.03→1.93R`、Adverse=`0.30→0.48R`、Realized EV=`0.75→0.24R`。
- Rolling C66→C70：Return=`201.29→101.33%`、MDD=`16.24→33.21%`、RoMD=`12.40→3.05`、EV=`0.89→0.36R`、平均曝險=`93.00→59.56%`、position gap=`2514→604`。High-MFE=`40.13→60.26%`、HM/LS=`20.07→40.26%`、LM/HS=`41.12→16.88%`、HM/HS=`20.07→20.00%`、High-Safety=`61.18→36.88%`。Full-MFE=`1.13→1.97R`、Adverse=`0.28→0.51R`、Realized EV=`0.90→0.35R`。
- Interpretation：Conditional-MFE ranking在沒有K/R0時確實成功取得更多upside，但增加的High-MFE幾乎全部來自Low-Safety，而非增加HM/HS；同時R0移除後即使position count更滿，資金曝險仍降到約58–60%，說明K不能補償R0的capital-deployment功能。
- Decision：**C70_RESULT_AVAILABLE / PERFORMANCE_NOT_SUPPORTED / MECHANISM_EVIDENCE_SUPPORTS_ABSOLUTE_SAFETY_GATE / NO_ROBUSTNESS / NOT_PROMOTED / NEXT=SR-C71**。

### 2026-08-26 — SR-C71 OOS + Rolling結果；Raw Safety gate有效，下一步改測P50/P60/P70 sensitivity

- C71與C70完全相同MR-13R final Conditional-MFE ranking、No-K/No-R0與canonical execution，只新增Raw Safety same-day orderable percentile>=0.50 eligibility。
- OOS C70→C71：Return=`74.18→128.62%`、MDD=`21.82→19.30%`、RoMD=`3.40→6.66`、EV=`0.25→0.53R`、平均曝險=`57.19→62.77%`；HM/HS=`19.51→24.87%`、HM/LS=`42.58→25.40%`、High-MFE=`62.09→50.27%`、High-Safety=`35.44→49.20%`；Full-MFE=`1.93→1.43R`、Adverse=`0.48→0.34R`、Realized EV=`0.24→0.54R`。
- Rolling C70→C71：Return=`101.33→123.57%`、MDD=`33.21→17.57%`、RoMD=`3.05→7.03`、EV=`0.36→0.50R`、平均曝險=`59.56→63.97%`；HM/HS=`20.00→21.75%`、HM/LS=`40.26→29.38%`、High-MFE=`60.26→51.13%`、High-Safety=`36.88→47.46%`；Full-MFE=`1.97→1.30R`、Adverse=`0.51→0.36R`、Realized EV=`0.35→0.50R`。
- Decision：**C71_MECHANISM_GO / PERFORMANCE_NOT_PROMOTED / DO_NOT_PRIORITIZE_R0_REINTRODUCTION / NEXT=SAFETY_GATE_SENSITIVITY**。Raw Safety不只改善path與geometry，也在兩mode提高平均曝險，因此No-R0低曝險不能單獨歸因為缺R0。

### 2026-08-26 — SR-C72/C73 OOS + Rolling結果；停止Safety threshold tuning，下一步改測C74 joint product selector

- C72/C73與C71共用同一MR-13R checkpoint/PIT source、final Conditional-MFE ranking、No-K/No-R0與canonical execution；唯一差異為Raw Safety same-day orderable percentile cutoff=`0.60/0.70`。
- OOS：C71/C72/C73 Exposure=`62.77/68.91/76.04%`、HM/HS=`24.87/23.22/23.38%`、HM/LS=`25.40/19.40/15.32%`、High-MFE=`50.27/42.62/38.70%`、High-Safety=`49.20/55.46/59.22%`、Full-MFE=`1.43/1.06/0.99R`、Adverse=`0.34/0.29/0.27R`、Realized EV=`0.54/0.53/0.65R`、RoMD=`6.66/5.68/5.11`。
- Rolling：Exposure=`63.97/72.31/80.56%`、HM/HS=`21.75/23.35/24.36%`、HM/LS=`29.38/21.56/13.47%`、High-MFE=`51.13/44.91/37.82%`、High-Safety=`47.46/51.50/61.32%`、Full-MFE=`1.30/1.22/0.91R`、Adverse=`0.36/0.32/0.25R`、Realized EV=`0.50/0.72/0.53R`、RoMD=`7.03/6.52/4.77`。
- Mechanism：Safety constraint確實改善Low-Safety排除與capital utilization，但HM/LS下降主要轉為LM/HS，而不是等量轉成HM/HS；constraint愈強時High-MFE/Full-MFE與RoMD整體走弱。這支持「兩個head都有訊號，但目前hard-gate + Conditional-MFE ranking不等於joint HM/HS objective」。
- 使用者明確指出繼續在C71-C73之間調Safety threshold等同迭代OOS上挑threshold，且最多只會在既有Safety/upside trade-off曲線內移動；因此停止0.55/0.65等threshold搜尋。
- Decision：**C72_C73_RESULT_AVAILABLE / SAFETY_CONSTRAINT_MECHANISM_CONFIRMED / THRESHOLD_TUNING_STOPPED / NO_MULTI_SEED / NEXT=SR-C74_PARAMETER_FREE_JOINT_SELECTOR**。

### 2026-08-26 — SR-C74 OOS + Rolling結果；parameter-free joint product未突破既有trade-off

- C74沿用MR-13R兩head、No-K/No-R0與canonical execution，不設Safety gate；唯一ranking=`same-day Raw Safety percentile × Conditional-MFE percentile`。
- OOS：Return=`61.71%`、MDD=`20.16%`、RoMD=`3.06`、EV=`0.42R`、Exposure=`72.08%`；HM/HS=`21.84%`、HM/LS=`21.58%`、High-MFE=`43.42%`、High-Safety=`54.21%`、Full-MFE=`1.15R`、Adverse=`0.29R`。
- Rolling：Return=`108.52%`、MDD=`15.03%`、RoMD=`7.22`、EV=`0.62R`、Exposure=`67.13%`；HM/HS=`22.85%`、HM/LS=`26.71%`、High-MFE=`49.55%`、High-Safety=`48.96%`、Full-MFE=`1.25R`、Adverse=`0.33R`。
- Interpretation：joint product沒有把actual HM/HS明顯推高，也沒有同時改善High-Safety與High-MFE；Rolling RoMD雖略高於C71，但仍遠低於C64/C66，且geometry落回C71附近。故「直接對兩個predicted percentile做乘法」不支持作下一個selector base。
- Decision：**C74_RESULT_AVAILABLE / JOINT_PRODUCT_NOT_SUPPORTED / NO_ROBUSTNESS / STOP_SELECTOR_ARITHMETIC / NEXT=AUD-mr13r-joint-capital-drawdown**。

### 2026-08-26 — AUD-mr13r-joint-capital-drawdown schema-v2修正；作廢Overlap ΣR drawdown語意

- 初版正式Audit已證實MR-13R兩head本身仍有signal、Safety→較窄stop→較高capital/exposure conversion也存在；但terminal沒有把程式已有的5×5 cells、Safety cohorts與upper-right N完整顯示，無法直接判讀joint enrichment gradient。
- 更重要的是初版drawdown `Overlap ΣR`只把「與peak→trough期間有重疊的trade」之**最終完整 realized R**加總；OOS C72 Max DD=`17.19%`卻可得`+8.29R`、Rolling C73 Max DD=`16.11%`卻可得`+43.07R`，因此該欄位明確**不是**peak→trough portfolio drawdown contribution，後續研究推論不得再引用。
- 同一formal Audit identity維持`AUD-mr13r-joint-capital-drawdown`，不新增Audit、不新增C75。output schema升為v2；Joint terminal/report新增S5×M5 upper-right `N`、完整5×5 `N / actual HM/HS%`、Safety quintile `N / Conditional-MFE→actual-MFE Dailyρ / High-MFE / HM/HS`及joint-product→HM/HS Dailyρ。
- Drawdown改用canonical Equity peak EOD→trough EOD的**true position MTM Δ**：由既有`score_ranking_trades.csv`實際買賣cashflow、pair `strategy_comparison.json`封存fee/tax contract、canonical cleaned OHLCV Close與核心long-sell tick/accounting規則重建每筆trade account value；交易在trough後的final realized R/PnL完全不參與contribution。
- 每個episode硬性要求`Σ position MTM Δ == Equity(trough)-Equity(peak)`（毫級容許誤差，失配即BLOCK），並分別標記peak-held、drawdown期間entry、期間exit、trough-held，再按HM/HS、HM/LS、LM/HS、LM/LS彙總MTM contribution；future truth只在MTM完成後作post-replay分組。詳細逐position輸出新增`drawdown_position_contributions.csv`。
- Synthetic新增反例：三筆trade最終`realized_r`合計=`+2R`，但portfolio peak=`1000`→trough=`950`；修正版得到`ΣMTM=-50`、Equity Δ=`-50`、reconcile Δ=`0`，HM/HS/HM/LS/LM/HS contribution=`-20/-20/-10`，證明已封住舊語意。Audit framework targeted synthetic=`15/15 PASS`。
- Diagnostic boundary仍為**READ_ONLY / NO_STRATEGY_REPLAY / NO_FUTURE_TRUTH_RUNTIME_USE / NO_THRESHOLD_FIT / NO_MODEL_TRAINING**；正式實機schema-v2結果仍待使用者重跑同一Audit後審閱，再依既定Case 1/2/3決策。
- Status：**ACTIVE / IMPLEMENTED_SCHEMA_V2 / RESULT_PENDING / OLD_OVERLAP_SUM_R_INVALIDATED**。

### 2026-08-26 — AUD-mr13r-joint-capital-drawdown schema-v2結果；MR-13R marginal signal存在但joint geometry collapse

- 同一read-only Audit已完成OOS fingerprint=`b12582a9de23`與Rolling=`3bca1e932f2e`。Joint model evidence：Conditional-MFE→actual MFE Dailyρ=`0.342/0.376`、Raw Safety→actual Safety Dailyρ=`0.339/0.364`（OOS/Rolling），故兩個head marginal learnability均存在。
- 但joint-product→actual HM/HS Dailyρ只有=`0.035/0.058`，且predicted S5×M5 upper-right兩mode皆只有`N=1`。5×5分佈近乎反對角；高Safety S5 cohort內Conditional-MFE→actual MFE rho仍=`0.151/0.169`，因此不能解讀成高Safety區MFE head完全失效，而是residual J沒有形成可供portfolio使用的absolute Safety×MFE joint coordinate。
- Capital mechanism維持前一輪結論：Safety提高時selected stop-distance通常縮窄，fixed-risk sizing下projected capital／reserved-notional與Exposure提高；holding days沒有同方向單調增加，故Exposure增加主要是sizing geometry而非holding-duration effect。
- Schema-v2 exact peak→trough MTM attribution全部硬性reconcile canonical Equity Δ=`0.000`。最大回撤損失主要落在Low-MFE quadrants；例如OOS C74 LM/HS+LM/LS=`-16.67%Peak`對總DD=`-20.16%`，Rolling C73=`-14.84%Peak`對總DD=`-16.11%`。這不支持把entry clustering／portfolio concentration列為current primary blocker。
- Decision：**AUDIT_RESULT_AVAILABLE / DECISION_COMPLETE / MARGINAL_SIGNAL_PRESENT / JOINT_GEOMETRY_COLLAPSED / STOP_SELECTOR_ARITHMETIC / DO_NOT_PRIORITIZE_PORTFOLIO_CONSTRUCTION / NEXT=MR-13S_MODEL_ONLY**。Audit identity/result保留read-only；不新增Audit、不恢復K/R0、不做threshold sweep／Multi-seed／Fixed-window。

### 2026-08-26 — MR-13S Forward Model Gate結果；先補Actual Daily/Breakout Truth Geometry Control

- 使用者完成MR-13S Seed42 frozen Forward Model Gate。Validation Raw Safety/Raw-MFE Dailyρ=`0.3505/0.4354`；Forward OOS=`0.3556/0.3948`，Breakout slice=`0.3165/0.3567`，兩個absolute heads的marginal learnability都保留。
- 但joint conversion仍collapse：Forward OOS joint-product→actual HM/HS Dailyρ=`0.0380`，predicted S5×M5=`N=0`；5×5大量集中在S1×M4/M5與S4/S5×M1/M2，absolute U並未解除MR-13R已有的近反對角geometry。故MR-13S不進PIT、不建立C75，也不做threshold/weight/calibration。
- 高Safety S5內Raw-MFE→actual MFE Dailyρ仍=`0.2587`，顯示local MFE information沒有消失；因此目前不能只用predicted geometry判定是input representation問題，還需要比較canonical **actual** Safety×Pure-MFE joint support。
- 本輪實作同一MR-13S identity的read-only Truth Geometry Control，不新增Audit或scientific identity：直接讀frozen `daily_ranker_oos_scores.csv.gz`已保存的daily-universal `target_low_adverse_safety_percentile` / `target_pure_mfe_percentile`；Breakout只依canonical event ticker/date membership filter，**不得在breakout subset內重算percentile**。
- 報表固定輸出Daily universal OOS與Breakout candidate OOS actual 5×5，每格=`N / population% / independence enrichment×`，另列actual Safety↔MFE Dailyρ、predicted Safety↔Raw-MFE Dailyρ、S5×M5與S4+×M4+ support。Independence expected=`N × P(S-bin) × P(M-bin)`，Breakout使用該subset實際marginals。
- Decision：**MR13S_MODEL_GATE_FAIL_JOINT_COLLAPSE / NO_STRATEGY_CONVERSION / ACTUAL_TRUTH_GEOMETRY_CONTROL_IMPLEMENTED_RESULT_PENDING**。取得Daily/Breakout actual geometry後，才決定下一步是representation-level controlled experiment、breakout-universe structural trade-off，或兩者並存。

### 2026-08-26 — MR-13S Actual Truth Geometry結果；真實upper-right存在，breakout structural trade-off排除

- Read-only Truth Geometry已完成且Breakout沒有subset rerank。Daily universal OOS `N=608,204`：actual Safety↔Pure-MFE Dailyρ=`-0.1261`；S5×M5=`23,242 / 3.82% / 0.92× independence`；S4+×M4+=`85,047 / 13.98% / 0.87×`。
- Breakout candidate OOS `N=17,346`：actual Safety↔Pure-MFE Dailyρ=`-0.0547`；S5×M5=`986 / 5.68% / 1.09×`；S4+×M4+=`2,725 / 15.71% / 0.93×`。Breakout upper-right並未被candidate qualification抽空，S5×M5甚至略高於依該slice marginals計算的independence expected。
- 同一frozen MR-13S predicted Safety↔Raw-MFE Dailyρ為Daily=`-0.9373`、Breakout=`-0.9029`，而predicted S5×M5=`0`；兩個marginal heads仍有OOS edge（Safety/MFE Dailyρ=`0.3556/0.3948`），因此主要缺口不是target truth absence、breakout structural trade-off或marginal learnability，而是模型能穩定抽出的predictable components被壓成近鏡像axis。
- Decision：**ACTUAL_JOINT_SUPPORT_CONFIRMED / BREAKOUT_STRUCTURAL_TRADEOFF_REJECTED / PREDICTED_COMPONENT_COLLAPSE_CONFIRMED / MR13S_STOP_NO_CONVERSION**。不建立C75、不做threshold/weight/calibration/K/R0/portfolio repair；下一個scientific問題可進只改input representation的controlled experiment，但不預占新MR。

### 2026-08-27 — MR-13T正式Model Gate FAIL；MR-13U H-only learnability control實作

- MR-13T Seed42正式Forward SOP結果：OOS Direct HM/HS相對same-model marginal product僅弱改善，Pair=`53.42% vs 52.53%`、PR-AUC=`0.2366 vs 0.2321`、Top10 HM/HS=`24.89% / 1.1159×` vs `22.95% / 1.0289×`。Breakout slice未確認，Direct反而落後product：Pair=`50.25% vs 51.47%`、PR-AUC=`0.2441 vs 0.2481`、Top10=`24.61% / 1.0299×` vs `24.84% / 1.0396×`。
- MR-13T marginal signal沒有崩潰：Forward Raw Safety/Raw-MFE Dailyρ=`0.3497/0.3876`、Breakout=`0.3051/0.3595`；但原mirror geometry幾乎不變，Daily predicted Safety↔MFE=`-0.9375`、S5×M5=`0`，Breakout=`-0.8962`、S5×M5=`8`。因此Direct H supervision只證明raw latent存在弱H-specific information，沒有證明shared tri-head已解joint problem。
- Decision：**MR13T_MODEL_GATE_FAIL / DIRECT_JOINT_SIGNAL_WEAKLY_PRESENT / FORWARD_ONLY_GAIN / BREAKOUT_NOT_CONFIRMED / SHARED_MARGINAL_MIRROR_UNCHANGED / NO_PIT / NO_STRATEGY_CONVERSION**。不做multi-seed/Fixed/Strategy conversion。
- 下一個最小scientific control=`MR-13U`，profile=`daily_universal_hmhs_single_head_full_list_ndcg_pairwise`。不建立新ARCH identity：H-only模型使用既有`inception_time_v1` single-head結構與canonical raw `300×10` sequence；與MR-13T保持同一InceptionTime trunk超參數、daily-universal universe、Seed42、split、optimizer、H target與full-list Delta-NDCG reduction，唯一變更是移除Safety/MFE heads與loss，encoder只受H supervision。
- MR-13U epoch selection事前固定Validation HM/HS Pair concordance；只有Pair完全同值才用Validation global PR-AUC tie-break。Forward/Breakout完全frozen，不回流selection/fitting。HM/HS population、Pair、Global/Mean-Daily PR-AUC、Top10/20 HM/HS rate/enrichment屬MR-13U Model-specific Extension evidence；不得改寫Standard Model SOP 1～6 schema。
- Predeclared decision：若MR-13U在Forward與Breakout都明顯高於MR-13T Direct H，支持multi-task gradient interference，下一步研究dedicated joint representation；若H-only仍弱且Breakout無增益，才支持current raw 300×10/current InceptionTime對joint intersection表示能力不足並重開raw-data architecture comparison。
- Status：**IMPLEMENTED / RESULT_PENDING / H_ONLY_LEARNABILITY_ABLATION / RAW_300x10_UNCHANGED / NO_OOS_FIT / NO_PIT / NO_STRATEGY_CONVERSION**。

### 2026-08-27 — MR-13U H-only Model Gate FAIL；MR-13V nonlinear Direct-H readout實作

- MR-13U Seed42正式結果：selected epoch=`1`；Validation HM/HS Pair/PR-AUC=`53.28%/0.2336`。Frozen Forward OOS Pair=`52.34%`、PR-AUC=`0.2306`、Top10 HM/HS=`23.41% / 1.0495×`、Top20=`23.24% / 1.0418×`；Breakout Pair=`50.17%`、PR-AUC=`0.2465`、Top10=`23.69% / 0.9913×`、Top20=`24.70% / 1.0335×`。
- 相對MR-13T Direct H，Forward `53.42% / 0.2366 / 1.1159×`在Pair/PR-AUC/Top10均較強；Breakout MR-13T Pair/Top10=`50.25%/1.0299×`也未被H-only改善，只有MR-13U Breakout PR-AUC小幅高於`0.2441`。因此不支持Safety/MFE auxiliary gradients是HM/HS主要blocker；H-only路線停止，不做loss-weight或single-head延伸。Decision=`MR13U_MODEL_GATE_FAIL / H_ONLY_NO_GAIN / MULTITASK_INTERFERENCE_NOT_SUPPORTED_AS_PRIMARY_BLOCKER / NO_PIT / NO_STRATEGY_CONVERSION`。
- 下一個最小mechanism experiment=`MR-13V`。回到MR-13T相同shared tri-head、raw `300×10`、InceptionTime trunk、S/U/H targets、Seed42、split、optimizer、full-list Delta-NDCG、三head固定等權loss與Raw-MFE Validation epoch selection；唯一scientific change為Direct H readout由`Linear(latent,2)`改為固定`Linear(latent,latent)→ReLU→Linear(latent,2)`。目前latent width由既有filters/kernels決定為`128`，故architecture=`ARCH-inception_time_safety_raw_mfe_hmhs_mlp_v1`。不增加dropout，不掃width/depth/activation/lambda，不讀marginal predicted score。
- Scientific question：MR-13T shared latent已可各自支撐Safety/MFE OOS Dailyρ約`0.35/0.39`，但HM/HS是upper-right conjunction；Direct H weak edge是否主要來自single linear readout capacity不足，而非temporal representation本身缺失。
- Predeclared Gate：沿用既有Direct-vs-product model-specific evidence。Forward Direct Pair/PR-AUC/Top10 enrichment需相對MR-13T明顯改善，Breakout必須同方向，Raw Safety/Raw-MFE marginal learnability不得實質崩潰；若FAIL，不再做另一個head arithmetic/width/activation小改，下一步才進learned temporal pooling或raw-data joint architecture。
- Research report governance：本輪**不得修改**Standard Model SOP 1～6、欄位／順序、Generalization/Evidence schema、menu role、`core/research_report_contract.py` approved persistent fingerprint。MR-13V因輸出payload與MR-13T同型，直接重用既有`Model-specific Extension｜MR-13V｜Direct HM/HS Joint Retrieval` schema；不新增常駐report contract。
- Status：**IMPLEMENTED / RESULT_PENDING / NONLINEAR_JOINT_READOUT_ONLY / RAW_300x10_UNCHANGED / STANDARD_REPORT_FROZEN / NO_OOS_FIT / NO_PIT / NO_STRATEGY_CONVERSION**。

### 2026-08-27 — MR-13V Global-rank gain但Top-tail Gate FAIL；MR-13W Continuous Joint-Min target實作

- MR-13V Seed42正式結果：Validation Direct HM/HS Pair/PR-AUC/Top10=`57.73% / 0.2642 / 28.44% (1.2518×)`；Forward=`54.30% / 0.2388 / 24.53% (1.0997×)`，優於same-model marginal product=`53.06% / 0.2325 / 23.07% (1.0343×)`；Breakout=`51.90% / 0.2484 / 24.15% (1.0106×)`，亦高於product=`51.00% / 0.2435 / 23.40% (0.9793×)`。
- 相對MR-13T Linear-H，MR-13V確實改善Forward Pair `53.42→54.30%`、PR-AUC `0.2366→0.2388`與Breakout Pair `50.25→51.90%`、PR-AUC `0.2441→0.2484`；但真正predeclared top-tail conversion未改善：Forward Top10 enrichment `1.1159×→1.0997×`，Breakout `1.0299×→1.0106×`。Marginal geometry仍Daily predicted Safety↔MFE=`-0.9425`、S5×M5=`0`，Breakout=`-0.9005`、S5×M5=`7`。Decision=`MR13V_MODEL_GATE_FAIL / NONLINEAR_READOUT_GLOBAL_RANK_GAIN / TOP_TAIL_GAIN_NOT_CONFIRMED / MARGINAL_MIRROR_UNCHANGED / STOP_HEAD_CAPACITY_TUNING`。
- 使用者指出binary HM/HS target本身存在information-loss：`0.51/0.51`與`0.90/0.90`都被標成positive，pairwise training無法要求後者排序更高。研究順序因此在learned temporal pooling前插入一個更小的target-only contrast，先確認learning problem是否定義過粗。
- 新identity=`MR-13W`，profile=`daily_universal_safety_raw_mfe_joint_min_mlp_head_full_list_ndcg_pairwise`；**architecture不變**，沿用MR-13V `ARCH-inception_time_safety_raw_mfe_hmhs_mlp_v1`，因模型結構/input representation完全相同。第三head target唯一改為`Jmin=min(S,U)`，S/U分別為canonical same-date Low-Adverse Safety percentile與Pure-MFE percentile。`min`不再做subset/date rerank、不乘0.5 indicator、不使用soft-min/product/harmonic mean，也不以OOS挑aggregation。
- MR-13W strict controls：raw `300×10`、shared InceptionTime trunk、Raw Safety head、stop-gradient Safety-conditioned Raw-MFE head、128→128→2 ReLU joint MLP、Seed42、daily-universal split、optimizer、full-list Delta-NDCG、三head固定等權loss與**Raw-MFE Validation Dailyρ epoch selection全部維持MR-13V**。因此本輪只測target information content，不同時改readout、pooling或selection rule。
- Model-specific Extension固定顯示Joint-Min Dailyρ/Pair、population mean min、Top10/20 actual min、Top10 mean Safety/MFE、binary HM/HS rate/enrichment。Binary HM/HS從training target降為interpretation truth，用來回答continuous maximin ranking是否自然把真正joint-good樣本推到top tail。
- Predeclared decision：Forward與Breakout的Joint-Min ordering與Top10/20 floor需同方向成立，且Top10 Safety/MFE兩邊不能只靠單邊極端、marginal S/U learnability不得實質崩潰。若GO，支持binary HM/HS target information loss是主要blocker；若FAIL，不再用OOS嘗試其他aggregation formula，下一步才進learned temporal pooling／raw-data joint representation。
- Research report governance：依B280，Standard Model SOP 1～6、Generalization/Evidence schema與approved persistent fingerprint不變；新增evidence只註冊`Model-specific Extension / joint_min_retrieval`，不得取得Standard section number。
- Status：**IMPLEMENTED / RESULT_PENDING / TARGET_ONLY_CONTINUOUS_JOINT_MIN / SAME_ARCH_AS_MR13V / STANDARD_REPORT_FROZEN / NO_OOS_FIT / NO_PIT / NO_STRATEGY_CONVERSION**。

### 2026-08-28 — MR-13W Forward generalization FAIL；MR-13X Joint learned temporal pooling實作

- MR-13W Seed42正式結果：Validation Joint-Min Dailyρ/Pair=`0.1417/54.78%`，population mean min=`0.3224`、Top10 Min=`0.3612`、Top10 S/MFE=`0.5031/0.5370`、HM/HS enrichment=`1.2070×`；Forward OOS=`0.0431/51.38%`，Pop=`0.3199`、Top10 Min=`0.3332`、S/MFE=`0.4611/0.5521`、HM/HS=`1.0702×`；Breakout=`0.0176/49.58%`，Pop=`0.3267`、Top10 Min=`0.3261`、S/MFE=`0.4164/0.5936`、HM/HS=`0.9986×`。
- Marginal heads未崩：Forward Raw Safety/Raw-MFE Dailyρ=`0.3548/0.3855`、Breakout=`0.3109/0.3610`；但predicted Safety↔Raw-MFE仍Daily=`-0.9393`、Breakout=`-0.8959`，Daily predicted S5×M5=`0`。因此`min(S,U)`修正binary target depth後只在Validation出現弱joint learnability，沒有形成usable Forward/Breakout ranking。Decision=`MR13W_MODEL_GATE_FAIL / VALIDATION_JOINT_MIN_SIGNAL / FORWARD_GENERALIZATION_COLLAPSE / BREAKOUT_RANDOM / TARGET_INFORMATION_LOSS_NOT_PRIMARY_BLOCKER / STOP_AGGREGATION_TUNING`。不再以OOS嘗試soft-min/product/harmonic mean。
- 下一個最小representation experiment=`MR-13X`，profile=`daily_universal_safety_raw_mfe_joint_min_attn_pool_mlp_head_full_list_ndcg_pairwise`；new architecture identity=`ARCH-inception_time_safety_raw_mfe_joint_attn_mlp_v1`。Scientific question：InceptionTime temporal feature map是否含joint path information，但MR-13W對300 bars做global mean後才送Joint MLP而提前丟失。
- MR-13X strict controls：raw `300×10`、InceptionTime trunk全部模組/filters/kernels/residual/BatchNorm、Safety/Raw-MFE heads、S/U/Jmin targets、128→128→2 ReLU Joint MLP、Seed42、split、optimizer、full-list Delta-NDCG、三head固定等權loss與Raw-MFE Validation Dailyρ epoch selection全部不變。唯一scientific change為Joint branch從global mean改成shared feature map上的`Conv1d(128,1,k=1)` scalar score、time softmax、weighted temporal sum。Marginal heads仍讀原global mean。
- 不加入positional encoding、多頭QKV attention、temperature、dropout、attention hidden width、window mask或任何sweep。Same-seed contract要求MR-13W/X所有既有parameter tensors初始化byte-identical，X只多`joint_attention_scorer.weight/bias`，使observed差異能歸因Joint pooling mechanism。
- Predeclared Gate：MR-13X需相對MR-13W在Forward Joint-Min rho=`0.0431`/Pair=`51.38%`/Top10 Min=`0.3332`與Breakout rho=`0.0176`/Pair=`49.58%`/Top10 Min=`0.3261`有共同且實質改善；Breakout Top10 Min必須真正高於population `0.3267`，Top10 Safety/MFE不能靠單邊極端，marginal S/U不可實質退化。若FAIL，不再做另一個InceptionTime attention/pooling小變體，下一步直接進ModernTCN/Patch Transformer以同一Jmin objective比較raw-data temporal representation。
- Research report governance：Standard Model SOP 1～6、Generalization/Evidence schema與approved persistent fingerprint完全不改；MR-13X直接重用既有`Model-specific Extension / joint_min_retrieval`，不新增report contract。PIT/current Rolling/runtime strategy維持未授權。
- Status：**IMPLEMENTED / RESULT_PENDING / JOINT_TEMPORAL_POOLING_ONLY / SAME_JMIN_TARGET_AS_MR13W / SAME_SEED_EXISTING_PARAMS_IDENTICAL / STANDARD_REPORT_FROZEN / NO_OOS_FIT / NO_PIT / NO_STRATEGY_CONVERSION**。

### 2026-08-28 — MR-13X small positive pooling signal但Gate FAIL；MR-13Y ModernTCN architecture-family comparison實作

- MR-13X Seed42正式結果：Validation Joint-Min rho/Pair/Top10 Min=`0.1466 / 54.99% / 0.3708` vs Pop=`0.3224`；Forward=`0.0528 / 51.73% / 0.3358` vs Pop=`0.3199`，Top10 Safety/MFE=`0.5188/0.4840`、HM/HS enrichment=`1.0477×`；Breakout=`0.0349 / 49.81% / 0.3364` vs Pop=`0.3267`，Top10 Safety/MFE=`0.5033/0.5026`、HM/HS=`1.0179×`。
- 相對MR-13W，learned temporal attention確實帶來小幅mechanism gain：Forward rho `0.0431→0.0528`、Pair `51.38→51.73%`、Top10 Min `0.3332→0.3358`；Breakout rho `0.0176→0.0349`、Pair `49.58→49.81%`、Top10 Min `0.3261→0.3364`。更重要的是Breakout Top10 S/MFE由失衡`0.4164/0.5936`收斂為`0.5033/0.5026`，predicted Safety↔MFE由Daily `-0.9393→-0.9190`、Breakout `-0.8959→-0.8663`。但Forward/Breakout ordering仍太弱，Breakout Pair仍未離開50%，不得進Strategy conversion。Decision=`MR13X_MODEL_GATE_FAIL / LEARNED_POOLING_SMALL_POSITIVE_SIGNAL / JOINT_FLOOR_BALANCE_IMPROVED / FORWARD_STILL_WEAK / BREAKOUT_NEAR_RANDOM / INCEPTIONTIME_REPRESENTATION_GATE_FAIL / STOP_LOCAL_INCEPTIONTIME_PATCHING`。
- 依MR-13X predeclared stop rule，後續不再試第二種InceptionTime attention、temperature、window mask、positional encoding或pooling小變體；正式進raw-data architecture-family comparison。第一個新identity=`MR-13Y`。
- MR-13Y profile=`daily_universal_safety_raw_mfe_joint_min_modern_tcn_attn_pool_mlp_head_full_list_ndcg_pairwise`；new architecture=`ARCH-modern_tcn_safety_raw_mfe_joint_attn_mlp_v1`。不能直接復活historical `modern_tcn_v1`，因9B已REJECTED/legacy read-only，且MR-13Y結構另含Safety/Raw-MFE tri-head、Joint temporal attention與Joint-Min MLP。
- Strict controls：canonical raw `300×10`、S/U/Jmin targets、Safety與stop-gradient Safety-conditioned Raw-MFE global-average semantics、Joint scalar `1×1` temporal attention softmax pooling、latent-width→latent-width→2 ReLU joint MLP、Seed42、daily-universal split、optimizer、full-list Delta-NDCG、三head固定等權loss與Raw-MFE Validation Dailyρ epoch selection全部沿用MR-13X。唯一research dimension是temporal trunk family。
- ModernTCN trunk不重新調參，固定沿用historical 9B容量近似recipe：depth=`6`、channels=`96`、kernel=`51`、pointwise expansion=`4×`、BatchNorm、dropout=`0.10`、receptive field=`301`。Head與attention width只依trunk latent自然為96，不加96→128 adapter、不掃head width。Synthetic以same Seed42重建legacy 9B與MR-13Y，要求`stem.*`/`blocks.*`所有shared trunk tensors exact equal，確保不是偷偷改ModernTCN recipe。
- Predeclared Gate：相對MR-13X Forward Joint-Min rho/Pair/Top10 Min=`0.0528/51.73%/0.3358`、Breakout=`0.0349/49.81%/0.3364`，MR-13Y需在Forward與Breakout對Joint-Min ordering/top-tail floor都有共同且實質改善；Breakout Pair需離開random、Top10 Min穩定高於Pop=`0.3267`，marginal Safety/Raw-MFE不得實質崩潰。若FAIL，不做ModernTCN depth/kernel/channels/expansion/dropout tuning，下一步直接Patch Transformer同一Jmin/head contract。
- Research report governance：Standard Model SOP 1～6與approved persistent fingerprint完全不變；MR-13Y重用既有`Model-specific Extension｜MR-13Y｜Continuous Joint-Min Retrieval`，不新增persistent report schema。
- Status：**IMPLEMENTED / RESULT_PENDING / RAW_DATA_ARCHITECTURE_FAMILY_COMPARISON / HISTORICAL_9B_TRUNK_RECIPE_FROZEN / STANDARD_REPORT_FROZEN / NO_OOS_FIT / NO_PIT / NO_STRATEGY_CONVERSION**。

### 2026-08-28 — MR-13Y ModernTCN Gate FAIL；MR-13Z Patch Transformer architecture comparison實作

- MR-13Y Seed42正式結果：Validation Joint-Min rho/Pair/Top10 Min=`0.1825 / 56.15% / 0.3701`；Forward=`0.0482 / 51.54% / 0.3284`、HM/HS enrichment=`1.0332×`；Breakout=`0.0299 / 50.32% / 0.3396`、Top20 Min=`0.3417`。相對MR-13X，Forward Joint-Min `0.0528/51.73%/0.3358`全面較弱，Breakout只有Top10/20 floor局部小升。
- Primary blocker是marginal generalization collapse：MR-13Y Forward Raw Safety/Raw-MFE rho=`0.3001/0.2601`，明顯低於MR-13X=`0.3541/0.3764`；Breakout Raw-MFE=`0.2347`。Predicted Safety↔MFE雖Daily/Breakout由MR-13X `-0.9190/-0.8663`變為`-0.8294/-0.7549`，S5×M5增至`9/36`，但這與truth-correlation下降同時發生，不能解讀為joint geometry改善。Decision=`MR13Y_MODEL_GATE_FAIL / MODERNTCN_MARGINAL_GENERALIZATION_COLLAPSE / JOINT_MIN_NO_FORWARD_GAIN / BREAKOUT_LOCAL_FLOOR_GAIN_ONLY / GEOMETRY_DECORRELATION_NOT_USEFUL / NO_MODERNTCN_TUNING`。
- 依事前stop rule不調ModernTCN kernel/channels/depth/dropout，直接進第二個raw-data family。新identity=`MR-13Z`，profile=`daily_universal_safety_raw_mfe_joint_min_patch_transformer_attn_pool_mlp_head_full_list_ndcg_pairwise`，new architecture=`ARCH-patch_token_transformer_safety_raw_mfe_joint_attn_mlp_v1`。historical `patch_transformer_v1`仍legacy read-only，不復活。
- MR-13Z strict controls：raw `300×10`、S/U/Jmin truth、Raw Safety head、stop-gradient Safety-conditioned Raw-MFE、marginal mean pooling語意、Joint scalar softmax attention、latent→latent→2 ReLU joint MLP、Seed42、split、optimizer、full-list Delta-NDCG、三head固定等權loss與Raw-MFE Validation epoch selection全部沿用MR-13X。唯一research dimension為temporal trunk family。
- Patch Transformer固定historical 9F未調參recipe：10-bar non-overlap patch（300 bars→30 tokens）、embedding=128、3 TransformerEncoder layers、4 heads、FFN=256、GELU、sinusoidal positional encoding、dropout=0.10、LayerNorm。Marginal heads對normalized encoded tokens取mean；Joint head對同一token map用單一`1×1` scalar scorer + softmax(token) weighted sum。不得調patch size/stride、embedding、depth、heads、FFN、position、dropout或另加CLS/token pooling變體。
- Same Seed42 reconstruction contract：新MR-13Z的`patch_projection.* / patch_normalization.* / encoder.* / output_normalization.*`必須與legacy 9F逐tensor exact equal；active builder不得import legacy `models.patch_transformer`。
- Predeclared Gate仍以MR-13X為主要baseline（MR-13Y已因marginal collapse淘汰）：Forward Joint-Min rho/Pair/Top10 Min=`0.0528/51.73%/0.3358`，Breakout=`0.0349/49.81%/0.3364`。MR-13Z需在Forward與Breakout joint ordering/top-tail floor共同且實質改善、Breakout Pair離開random且marginal S/U不崩。若FAIL，不再逐一搜尋更多backbone family，下一步重開input representation / target observability問題。
- Research report governance：Standard Model SOP 1～6、Generalization/Evidence schema與approved fingerprint完全不變；MR-13Z直接重用`Model-specific Extension / joint_min_retrieval`。
- Status：**IMPLEMENTED / RESULT_PENDING / PATCH_TRANSFORMER_FAMILY_ONLY / SAME_JMIN_CONTRACT / STANDARD_REPORT_FROZEN / NO_OOS_FIT / NO_PIT / NO_STRATEGY_CONVERSION**。

### 2026-08-28 — MR-13Z Model Gate PASS；授權PIT-safe Direct Joint-Min conversion並建立SR-C75

- MR-13Z Seed42正式Model SOP完成。Forward Joint-Min Population/ρ/Pair/Top10 Min/HMHS×=`0.3199 / 0.1148 / 53.84% / 0.3495 / 1.1693×`；Breakout=`0.3267 / 0.0677 / 53.75% / 0.3639 / 1.1698×`。相對predeclared MR-13X control Forward=`0.0528/51.73%/0.3358/1.0477×`、Breakout=`0.0349/49.81%/0.3364/1.0179×`，ordering與top-tail joint floor均同方向明顯改善。
- Marginal learnability未觸發collapse：MR-13Z Forward Raw Safety/Raw-MFE Dailyρ=`0.3350/0.3857`、Breakout=`0.3300/0.3519`；不同於MR-13Y ModernTCN的Forward Raw-MFE=`0.2601`。Decision=`PATCH_TRANSFORMER_JOINT_SIGNAL_CONFIRMED / FORWARD_AND_BREAKOUT_GATE_GO / TOP_TAIL_JOINT_FLOOR_IMPROVED / MARGINAL_LEARNABILITY_PRESERVED / MODEL_GATE_PASS`。Patch Transformer architecture/recipe就此freeze，不做patch/head/depth/embedding/dropout tuning，也不繼續backbone shopping。
- Model Gate PASS只授權下一階段PIT-safe research evaluation，不等於production promotion。MR-13Z research spec因此正式開啟`selection_pit_authorized=True`與`current_time_validation_authorized=True`，production runtime仍未promotion。
- PIT score contract保持既有primary語意相容：`breakout_quality_score`仍是Raw-MFE legacy primary；MR-13Z fold artifact另保存`raw_safety_score`、`raw_mfe_score`、`joint_min_score`。這避免為C75偷偷重定義舊欄位。
- 新DL source=`CONT13Z_ROLL`，evaluation stage=`selection_point_in_time`。Model Research canonical Rolling/PIT producer負責建立合法fold checkpoints/scores；Strategy Compare的`selection_pit_from_existing_folds`只允許重組/重評既有fold，不得因缺fold自行訓練模型。
- 新strategy identity=`SR-C75`：Min MR-13Z Direct Joint-Min No-K/No-R0。與C74固定共用Min `base-finalist-best`、all-off、No-K/No-R0、physical max positions=10、canonical sizing/cash/orderability/execution；唯一selector treatment為formal `primary_score_column=joint_min_score` descending + canonical tie-break。無Safety gate、threshold、product、fitted weight或calibration。
- `extending_current` Compare Suite schema=`62`，arms新增C75；primary contrast=`C75-C74`（learned Direct Joint-Min vs parameter-free Safety×Conditional-MFE percentile product），secondary=`C75-C71`。C75 score-projection cache identity只綁實際消費的`joint_min_score`，不以Raw-MFE `breakout_quality_score`冒充。
- Predeclared conversion decision：先做single-seed OOS+Rolling。若沒有形成具決策價值的Return/MDD/RoMD與Joint/Trade-quality改善，關閉C75且不投入Multi-seed；只有成為真正candidate才進Extending Multi-seed，再依必要性做Fixed。OOS/strategy結果不得回頭修改MR-13Z fitting、target、architecture或threshold。
- Status：**MR13Z_MODEL_GATE_PASS / PIT_AUTHORIZED / C75_IMPLEMENTED_RESULT_PENDING / SINGLE_SEED_CONVERSION_NEXT / NO_PRODUCTION_PROMOTION / NO_ROBUSTNESS_YET**。

### 2026-08-28 — C75 Direct Joint-Min conversion Gate FAIL；MR-13Z Model Gate PASS維持

- C75=`Min MR-13Z Direct Joint-Min No-K / No-R0`，與C74固定共用Min `base-finalist-best`、all-off、No-K/No-R0、canonical sizing/cash/orderability/execution；唯一selector treatment為`CONT13Z_ROLL.joint_min_score` descending，沒有Safety gate、threshold、product、calibration或fitted weights。
- OOS：C74 Return/MDD/RoMD/EV=`61.71% / 20.16% / 3.06 / 0.42R`；C75=`91.20% / 17.88% / 5.10 / 0.51R`，形成局部改善。
- Rolling：C74=`108.52% / 15.03% / 7.22 / 0.62R`；C75=`44.66% / 19.95% / 2.24 / 0.33R`，核心strategy economics collapse。
- Rolling mechanism：C74 HM/HS=`22.85%`、HM/LS=`26.71%`、High-MFE=`49.55%`、+2R前initial stop=`12.50%`、+3R前=`14.29%`；C75=`24.33% / 18.45% / 42.78% / 26.87% / 26.67%`。因此learned Joint-Min確實改善部分Safety×MFE geometry，但沒有改善「先活過initial stop再取得upside」的path survival。
- Decision：**OOS_LOCAL_GAIN / ROLLING_COLLAPSE / JOINT_GEOMETRY_IMPROVED_BUT_PATH_SURVIVAL_FAILED / CONVERSION_GATE_FAIL / NO_ROBUSTNESS**。C75不進Multi-seed、Fixed Window或C75 tuning。
- MR-13Z scientific layer維持 **PATCH_TRANSFORMER_JOINT_SIGNAL_CONFIRMED / MODEL_GATE_PASS**；strategy conversion FAIL不得反向把model判成FAIL。

### 2026-08-28 — MR-13AA Model Gate：Patch Transformer未形成MR-13H target的通用architecture gain；MR-13AB first-passage Pure-MFE control實作

- MR-13AA Seed42正式Standard Model SOP完成：selected epoch=`2`；Validation post-refit Daily rho=`0.2410`。Forward OOS Daily rho/Pair/Top-K Lift=`0.1974 / 56.79% / +1.1405R`；Breakout candidate slice=`0.2044 / 58.69% / +0.4672R`。
- Primary historical MR-13H InceptionTime reference為Forward=`0.2332 / 58.04% / +1.3400R`、Breakout=`0.2056 / 58.86% / +0.5061R`。MR-13AA相對reference：Forward Δrho=`-0.0358`、ΔPair=`-1.25pp`、ΔTop-K=`-0.1995R`；Breakout Δrho=`-0.0012`、ΔPair=`-0.17pp`、ΔTop-K=`-0.0389R`。因此Patch Transformer在成熟MR-13H economic scalar target上沒有形成一致增量。
- Decision：**LEARNABILITY_CONFIRMED / NO_GENERAL_ARCHITECTURE_GAIN / PATCH_GAIN_JOINT_MIN_SPECIFIC_OR_INTERACTION / MODEL_GATE_FAIL_FOR_GENERAL_ARCHITECTURE_PROMOTION / NO_PIT / NO_STRATEGY_CONVERSION / NOT_PROMOTED**。MR-13Z的Patch Transformer仍保留其Joint-Min Model Gate PASS，但不得外推為所有target的通用最佳backbone；不做MR-13AA PIT、strategy conversion、Patch tuning或backbone sweep。
- 下一個最小缺失cell分配合法identity=`MR-13AB`，profile=`daily_universal_first_risk_breach_pure_mfe_full_list_ndcg_pairwise`，architecture回到MR-13K相同`inception_time_v1`。其唯一scientific change為把MR-13K full-horizon Pure-MFE future path改成MR-13E既有的first-risk-breach truncation；target identity=`daily_first_risk_breach_pure_mfe_r_v1`。
- MR-13AB scalar固定=`favorable_return_before_first_risk_breach / risk_budget_return`。first-risk-breach path規則直接重用MR-13E canonical semantics：same-bar adverse-first、breach-day high不計、breach後upside不計；若第一根future bar即breach則target=`0R`。若40D完全無breach，target與MR-13K `daily_full_horizon_pure_mfe_r_v1`完全同義。Adverse-to-peak仍可作diagnostic component，但**不從target扣除**。
- Strict control固定MR-13K：daily-universal universe、300×10 input、InceptionTime、40D horizon、fixed canonical R scale、full-list ΔNDCG pairwise objective/reduction、Seed42、split、Adam optimizer、`mean_daily_spearman` epoch selection、training/sample scope全部不變。禁止Safety head、threshold、lambda、time penalty、product/fusion、portfolio state、PIT、strategy conversion、Multi-seed與Fixed Window。
- Primary historical reference固定MR-13K Forward=`rho 0.3833 / Pair 63.19% / Top-K Lift +1.6401R`，Breakout=`rho 0.3621 / Pair 64.09%`。MR-13AB先只回答三件事：(1) first-passage constraint使learnability損失多少；(2) high-score ranking是否仍保留明確upside separation；(3) survival資訊能否用比MR-13E/H composite magnitude更低的learnability成本嵌入single scalar。若learnability直接退回約MR-13E/H水準而無新的可決策evidence，停止first-passage scalar target方向；若仍保留顯著Pure-MFE learnability且path-survival constraint有價值，才另行決定是否建立strategy conversion，不預先授權。
- Standard Model SOP 1～6與approved persistent report fingerprint `2b372bd465258234`完全不變。
- Status：**MR13AA_CLOSED / MR13AB_IMPLEMENTED / RESULT_PENDING / PURE_MFE_FIRST_PASSAGE_TARGET_ONLY / INCEPTIONTIME / MODEL_GATE_ONLY / NO_PIT / NO_STRATEGY_CONVERSION**。

### 2026-08-28 — MR-13AB Forward learnability／Target geometry PASS；啟動一次性 frozen survival-increment Audit

- MR-13AB Seed42正式Standard Model SOP完成：selected epoch=`1`；Validation post-refit Daily rho=`0.3439`。Forward OOS self-target Daily rho/Pair/Top-K Lift=`0.3062 / 60.63% / +0.8959R`；Breakout candidate slice=`0.2570 / 60.58% / +0.4262R`。相對MR-13K historical Forward `0.3833 / 63.19% / +1.6401R`，first-passage constraint降低部分self-target learnability，但仍顯著高於MR-13E/H約0.21～0.23的economic-scalar水準。
- 同一MR-13AB frozen score對MR-13K full-horizon Pure-MFE reference仍有Forward Daily rho/Pair=`0.3782 / 63.04%`，只比MR-13K historical `0.3833 / 63.19%`低`0.0051 / 0.15pp`；因此upside ranking能力幾乎完整保留。
- 正式Target comparison完成：common stock-days=`2,052,791`；40D risk breach=`33.56%`；first-breach truncation真正改變target=`7.63%`；candidate/reference mean daily rank correlation=`0.9059`；Top-10 overlap=`88.39%`；mean absolute delta=`0.1007R`。這表示first-breach target是sparse但具經濟幅度的path correction，而不是全面重寫Pure-MFE。
- 目前不能僅由aggregate rho宣告完整Model Gate PASS：92%+ rows target未改，故MR-13AB可能主要複製MR-13K ranking。最後的關鍵不確定性收斂為「在7.63% changed rows與兩Target要求相反排序的same-date conflict pairs上，MR-13AB frozen model是否相對MR-13K frozen model增加survival ordering」。
- 分配一次性Audit identity=`AUD-mr13ab-survival-increment`。Audit只讀兩個canonical `daily_ranker_oos_scores.csv.gz`與各自`continuous_ranker_report.json`，不重建Target、不訓練、不建PIT、不Replay。Evidence固定為：(1) 2×2 frozen model×target Daily rho；(2) changed-row同日score percentile demotion與correction magnitude relation；(3) target-order conflict-pair concordance。
- Stopping rule：一次Audit足以決定`SURVIVAL_INCREMENT_CONFIRMED / MODEL_GATE_PASS → PIT-safe Extending-Window`，或`NO_INCREMENT / STOP`；不得由結果展開barrier、lambda、threshold、calibration、backbone或target-family sweep。
- Status：**MR13AB_LEARNABILITY_PASS / TARGET_GEOMETRY_PASS / AUDIT_IMPLEMENTED / SURVIVAL_INCREMENT_RESULT_PENDING / NO_PIT / NO_STRATEGY_CONVERSION**。

### 2026-08-28 — MR-13AB Final Survival Increment Audit：conflict ordering明確偏向full-horizon Pure-MFE，Model Gate FAIL

- Baseline：`test-branch-1_20260828_194453_50fa5a89.zip`，SHA256 `28706cd37b3495d267441b35c6c86fa94b41b54ee7253ee69af53ca8a3a5de4d`。
- `AUD-mr13ab-survival-increment`最終成功執行；Common OOS=`608,204`。
- Frozen score aggregate：對MR-13AB first-breach Pure-MFE truth的Mean Daily rho=`0.3062`；對MR-13K full-horizon Pure-MFE reference truth=`0.3782`；兩Target本身Daily rho=`0.9119`。
- OOS changed rows=`43,487 (7.15%)`，其中survival-demotion rows=`43,373`、empty-prebreach zero-floor rows=`114 (0.02%)`；mean target correction=`1.2809R`。
- changed rows的mean percentile：MR-13K reference truth=`0.6161`、MR-13AB first-breach truth=`0.2887`，應有rank correction=`-0.3274`；但MR-13AB frozen score percentile反而=`0.6768`，score residual vs reference rank=`+0.0607`。Reference-only top-10% rows=`6,754`，仍有`33.37%`被MR-13AB score留在top-10%。
- 最具判別力的target-order conflict control：same-date strict conflict pairs=`7,889,338`，涵蓋`1,198` dates。MR-13AB frozen score對AB survival truth concordance=`26.51%`，對K full-horizon truth=`73.49%`，AB-truth advantage=`-46.99pp`；mean daily AB-truth concordance=`23.52%`，只有`1.59%` conflict dates高於50% natural null。
- 因此MR-13AB雖有`LEARNABILITY_PASS`與`TARGET_GEOMETRY_PASS`，且幾乎保留MR-13K upside ranking能力，但**沒有把first-passage correction學進ranking**；在真正需要二選一的conflict pairs上反而強烈遵循full-horizon Pure-MFE ordering。這否決「sparse first-breach scalar correction可在不犧牲upside learnability下自然加入survival ordering」的假說。
- Final Decision：`NO_SURVIVAL_INCREMENT / FULL_HORIZON_ORDERING_DOMINATES_CONFLICTS / MODEL_GATE_FAIL / NO_PIT / NO_STRATEGY_CONVERSION / NO_ROBUSTNESS / STOP`。
- 依事前stop rule，不再做MR-13AB PIT、Strategy arm、Multi-seed、Fixed Window，也不展開first-passage barrier/lambda/threshold/calibration/backbone/aggregation sweep。下一個尚未實作的model問題改為Plan C-M：PIT-safe predicted-upside context下的Conditional Low-Adverse；其目的不是再壓一個global scalar，而是測path-risk residual在upside context條件化後是否成為可辨識訊號。未開始實作前不預占新的MR identity。
- `AUD-mr13ab-survival-increment`已完成唯一decision question；依最小必要Audit與disposable lifecycle，formal config/catalog/implementation/dedicated synthetic退役，歷史identity與以上結果永久保留。

### 2026-08-28 — MR-13AC Model Gate GO後先做actual conversion：current matrix收斂為C61/C58/C59/C76/C77/C78

- MR-13AC正式Seed42 Model Gate結果：selected epoch=`2`；Validation/OOS/Breakout Daily rho=`0.2278/0.1650/0.1900`，Global rho=`0.2190/0.1486/0.1857`，Pair=`57.85/55.50/56.60%`，Top-Bottom target spread=`+0.2271/+0.1513/+0.1942`。OOS→Breakout rho=`+0.0251`、Pair=`+1.10pp`，故Decision=`CONDITIONAL_PATH_RISK_LEARNABILITY_CONFIRMED / FORWARD_MODEL_GATE_GO`。
- 使用者重新確認研究終局偏好：希望一個head直接表示「漲多跌少」；13系列中同時把upside/downside放進單head的target普遍較難學，但MR-13H smooth full-horizon economic scalar仍有Forward rho=`0.2332`與歷史接近MR-13E的16-seed portfolio evidence。因此在開新的joint-head model之前，先用current corrected portfolio contract重測MR-13H actual conversion，並同時量MR-13AC standalone residual-safety actual performance。
- 使用者明確指出：C58本身就是正確Min DL-off baseline，不需重建；K限制已知有問題，R0亦已被resource attribution證明會實質改寫DL main indicator/basket membership。因此新的direct-score comparison不得再把K/R0綁進C76/C77/C78。C59則刻意保留舊MR-13E K/R0 constrained reference，用`C76-C59`量同一13E score下移除K/R0的淨效果。
- Current Strategy Compare schema=`63`，active arms收斂為6組：`C61` Full broad reference、`C58` Min DL-off causal baseline、`C59` MR-13E K/R0 constrained、`C76` MR-13E No-K/No-R0、`C77` MR-13H No-K/No-R0、`C78` MR-13AC No-K/No-R0。C76/C77/C78共用Min `base-finalist-best`、all-off、physical max positions=10、canonical sizing/cash/orderability/execution與同一`resource-aware-continuous-score-no-k-no-r0` runtime；唯一差異為DL source。
- Current sources：`CONT13E_ROLL`保留；新增`CONT13H_ROLL`與`CONT13AC_ROLL`。MR-13H與MR-13AC正式授權current research OOS/Rolling PIT source，不代表production promotion。MR-13AC Rolling只refit Stage-2 ranker；Stage-1 predicted-upside context仍固定既有Selection cross-fit + fixed-pre-OOS scientific contract，strategy結果不得回流改Stage-1或target。
- C64/C66/C71/C72/C73/C74/C75及其舊sources/contrasts全部退出active catalog並進`config/compatibility/strategy_compare_history.py`；歷史ID、結果、Audit可讀性完整保留，不刪證據。
- Primary contrasts：`C76-C58`=13E direct score增量、`C76-C59`=同13E source拿掉K/R0的resource effect、`C77-C58`與`C77-C76`=MR-13H single-head economic score actual value、`C78-C58/C76/C77`=MR-13AC standalone conditional-safety diagnostic。C78績效FAIL不得反向推翻MR-13AC Model Gate。
- Execution gate：只先跑single-seed Extending OOS + Rolling。未看到結果前不做Multi-seed、Fixed、不改target/architecture、不新增MR-13AD，不改production C42/C44。
- Decision：**SIX_ARM_DIRECT_CONVERSION_MATRIX_IMPLEMENTED / C58_UNCHANGED / K_R0_REMOVED_FOR_C76_C77_C78 / MR13H_PRIMARY_COMPARISON / MR13AC_DIAGNOSTIC / RESULT_PENDING / NO_ROBUSTNESS_YET**。

### 2026-08-29 — C76/C77/C78 direct No-K/No-R0 OOS + Rolling：MR-13H無穩定優勢，MR-13AC standalone FAIL；C76-C59純K/R0解讀撤回

#### 狀態

`RESULT_AVAILABLE / DIRECT_CONVERSION_GATE_CLOSED / NO_ROBUSTNESS / NOT_PROMOTED`

#### 程式基準

- 使用者最新 baseline ZIP：`test-branch-1_20260828_232818_55c1bd80.zip`
- SHA256：`033151effee508946ed0a71d5d252b84a39c4443acf450c6d039ec2d0c2666db`
- Strategy Compare period：`2021-01-01 ～ 2026-03-02`
- Current matrix：`C61/C58/C59/C76/C77/C78`
- C58：Min DL-off baseline，原樣保留。
- C59：MR-13E exact K/R0 constrained reference。
- C76/C77/C78：同一 `resource-aware-continuous-score-no-k-no-r0` direct allocator；分別使用MR-13E / MR-13H / MR-13AC score。
- Production C42/C44不變。

#### Model/PIT evidence

MR-13H current PIT：

- OOS：Daily rho=`0.2310`，Global rho=`0.1745`；Breakout Daily rho=`0.2211`。
- Rolling：Daily rho=`0.2295`，Global rho=`0.1702`；Breakout Daily rho=`0.2156`。

MR-13AC current PIT：

- OOS：Daily rho=`0.1650`，Global rho=`0.1486`；Breakout Daily rho=`0.1900`。
- Rolling：Daily rho=`0.1823`，Global rho=`0.1352`；Breakout Daily rho=`0.2064`。

兩者Rolling learnability均維持正向；後續strategy失敗不能歸因為model score完全失去ordering signal。

#### Strategy core performance

| Arm | OOS Return | OOS MDD | OOS RoMD | OOS EV | OOS Exposure | Rolling Return | Rolling MDD | Rolling RoMD | Rolling EV | Rolling Exposure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C58 Min baseline | 136.47% | 12.52% | 10.90 | 0.74R | 90.14% | 101.60% | 13.12% | 7.74 | 1.02R | 92.24% |
| C59 MR-13E exact K/R0 | 166.49% | 15.92% | 10.45 | 1.24R | 90.85% | 201.10% | 15.92% | 12.63 | 1.11R | 93.28% |
| C76 MR-13E direct No-K/No-R0 | 103.47% | 28.64% | 3.61 | 0.40R | 53.66% | 73.94% | 36.82% | 2.01 | 0.31R | 57.99% |
| C77 MR-13H direct No-K/No-R0 | 79.62% | 29.32% | 2.72 | 0.25R | 55.42% | 87.66% | 29.96% | 2.93 | 0.51R | 57.19% |
| C78 MR-13AC direct No-K/No-R0 | 83.35% | 12.09% | 6.89 | 0.43R | 86.05% | 77.01% | 15.07% | 5.11 | 0.54R | 88.53% |

#### MR-13H vs MR-13E direct same-allocator interpretation

`C77-C76`是本輪最乾淨的single-head economic score contrast，因兩者allocator/cash/sizing/orderability/execution完全相同，只換score source。

- OOS：H Return=`79.62%` < E=`103.47%`；RoMD=`2.72 < 3.61`；EV=`0.25R < 0.40R`；`+2R`前initial-stop=`12.00% > 9.26%`；Full-MFE=`1.78R < 1.85R`。
- Rolling：H Return=`87.66%` > E=`73.94%`；RoMD=`2.93 > 2.01`；EV=`0.51R > 0.31R`；`+2R`前initial-stop=`7.69% < 11.86%`；Full-MFE=`1.90R > 1.85R`。

方向在OOS與Rolling反轉，沒有形成MR-13H對MR-13E的stable strategy dominance。兩者又都明顯低於C58 RoMD，因此不值得再投入Multi-seed或Fixed。這與historical MR-13H constrained 16-seed「Selection略優、Forward略輸、無穩定promotion」結論一致。

#### Direct No-K/No-R0 deployment evidence

C76/C77雖把Filled High-MFE提高到：

- C76 OOS/Rolling=`59.20% / 57.34%`
- C77 OOS/Rolling=`56.46% / 58.58%`

且Full-MFE約=`1.78～1.90R`，但平均曝險只剩約`53～58%`，Adverse約=`0.45～0.51R`，Realized EV只=`0.25～0.51R`。這是「ranking找得到upside，但capital/path conversion失敗」，不是模型完全沒有upside signal。

此現象與historical C46 exact No-R0（Exposure=`56.76%`且selection target quality反而更強）以及C70 direct No-K/No-R0的deployment collapse一致；R0確實深度改變capital deployment。不能只把R0視為無關的報表限制。

#### C76-C59 interpretation correction

原Registry/Compare description曾把`C76-C59`稱為「同一MR-13E score下移除K/R0的resource-contract淨效果」。正式結果複查程式後確認此描述不成立：

- C59 runtime=`resource-aware-continuous-score-constrained-optimal`，使用`exact_branch_and_bound_v1`、固定baseline K並保留R0。
- C76 runtime=`resource-aware-continuous-score-no-k-no-r0`，使用`model_score desc` direct order後的canonical greedy reservation，沒有exact basket search。

因此C76-C59同時改變`K / R0 / allocator / solver`，不得再被引用為pure K/R0 ablation。純R0 controlled evidence仍是historical C46；K-Flex/R0-preserved exact C67則因large-K exact certification COMPUTE_BLOCKED，不能把timeout解讀成策略結果。

#### MR-13AC standalone diagnostic

C78沒有upside reward，結果符合其signal語意：

- Filled High-Safety：OOS=`71.31%`、Rolling=`69.58%`，高於C58=`68.09/67.72%`。
- OOS MDD=`12.09%`略優於C58=`12.52%`，但Return/RoMD=`83.35%/6.89`明顯低於`136.47%/10.90`。
- Rolling Return/MDD/RoMD=`77.01%/15.07%/5.11`，亦低於C58=`101.60%/13.12%/7.74`。

Decision：`STANDALONE_CONVERSION_FAIL / MODEL_GATE_RETAINED`。不得以C78 FAIL反向否定MR-13AC conditional residual learnability。

#### C78 First-Passage reporting bug

本輪OOS/Rolling C78 Upside Survival顯示：

`UNAVAILABLE | ValueError: MR-13AC predicted-upside architecture與conditional low-adverse target必須成對`

根因是Strategy Compare offline Upside Survival雖明確宣告Pure-MFE diagnostic profile owner，實作卻把active strategy arm的`model_architecture`傳入Pure-MFE sample provider。C59/C76/C77均為`inception_time_v1`所以未暴露；C78使用`inception_time_predicted_upside_context_v1`才觸發profile/architecture pairing guard。

工程修正：Pure-MFE path diagnostics現在從其**自己的explicit diagnostic profile + canonical default architecture**解析provider，不再繼承active arm architecture。這只影響read-only post-replay diagnostics；strategy replay、score、selection、accounting與Model SOP均不變。既有C78結果可在REUSE時backfill，不需重跑策略或重訓模型。

#### Final decision

`MR13H_DIRECT_NO_STABLE_GAIN / MR13AC_STANDALONE_FAIL / DIRECT_NO_K_NO_R0_DEPLOYMENT_COLLAPSE / C76_C59_PURE_KR0_INTERPRETATION_RETRACTED / NO_ROBUSTNESS / STOP`

下一步不自動做Multi-seed／Fixed，也不因本結果修改MR-13AC Model Gate。若使用者仍要研究單一head「漲多跌少」，應重新定義下一個joint economic target問題；若要研究K/R0本身，必須另建只改constraint、且明確固定allocator/solver semantics的新strategy identity，不得覆寫C76-C78。

### 2026-08-29 — C79 single-seed OOS + Rolling：MR-13AC constrained conversion確認Safety增益，但存在upside regret

- Controlled contract成立：C79逐欄複製C59的Min/all-off/K/R0/exact branch-and-bound portfolio construction，唯一scientific treatment=`CONT13E_ROLL → CONT13AC_ROLL`，因此`C79-C59`可解讀為DL-source-only effect。
- OOS：C59 Return/MDD/RoMD/EV/Exposure=`166.49%/15.92%/10.45/1.24R/90.85%`；C79=`116.80%/12.60%/9.27/0.65R/90.24%`。C79降低MDD但Return/EV/RoMD較弱。
- Rolling：C59=`201.10%/15.92%/12.63/1.11R/93.28%`；C79=`181.60%/12.63%/14.37/1.05R/93.14%`。C79 Return略低，但MDD與RoMD較佳且Exposure幾乎完全一致，排除先前direct allocator的capital-deployment confound。
- Filled truth：C79 High-Safety OOS/Rolling=`72.67/75.99%`高於C59=`62.33/62.68%`；Adverse=`0.19/0.18R`低於C59=`0.23/0.27R`。代價是Full-MFE=`0.82/0.87R`低於C59=`0.98/1.04R`。C79 OOS/Rolling selection target percentile=`0.657/0.674`，證明conditional safety ordering確實被exact constrained portfolio使用。
- First-Passage並非全面改善：C79 OOS `+2R/+3R`前initial-stop=`39.13/25.00%`高於C59=`8.82/5.00%`；Rolling `+1R/+2R/+3R`=`20.45/25.00/10.00%` vs C59=`14.71/14.63/12.50%`。因此「低Adverse」不等價於「更好的高R path survival」，不得只以MDD/Safety宣稱clean dominance。
- Decision=`CONTROLLED_CONVERSION_SIGNAL_CONFIRMED / SAFETY_GAIN_CONFIRMED / UPSIDE_REGRET_PRESENT / OOS_ROLLING_TRADEOFF / ROBUSTNESS_DEFERRED_FOR_MR13AD / NOT_PROMOTED`。C79具robustness資訊價值，但使用者選擇先補完全對稱的reverse conditional model cell，以更完整比較upside/downside/learnability/conversion。

### 2026-08-29 — MR-13AD Seed42 Forward Model Gate：reverse conditional MFE可學性不足，依事前stop rule結案

#### 狀態

`RESULT_AVAILABLE / REVERSE_CONDITIONAL_LEARNABILITY_WEAK / FORWARD_MODEL_GATE_FAIL / NO_PIT / NO_STRATEGY_CONVERSION / STOP`

#### 程式基準

- 使用者最新 baseline ZIP：`test-branch-1_20260829_040025_5fc44bae.zip`
- SHA256：`5eeb85896f8b881cd3e1ea858a3ef175dcf690b2abb697885cef136575df15dc`
- Profile：`daily_universal_predicted_safety_conditional_mfe_full_list_ndcg_pairwise`
- Architecture：`inception_time_predicted_safety_context_v1`
- Seed：`42`
- Stage-1：MR-13M PIT-safe predicted-Safety context；Selection 10-fold expanding cross-fit，Forward single fixed pre-2021 fit，兩者coverage=`100%`。
- Stage-2：MR-13K recipe + 1個direct predicted-safety percentile scalar；target=`same-date Pure-MFE percentile residual given predicted Safety percentile`再same-date percentile化。

#### Forward Model Gate結果

- selected epoch=`1`；選模Validation rho=`0.0770`。
- post-refit Validation：Daily/Global rho=`0.1591/0.1129`，Pair=`55.32%`，Top10/Bottom10=`0.5488/0.4213`，spread=`+0.1275`。
- Forward OOS：Daily/Global rho=`0.0999/0.0478`，Pair=`53.39%`，Top10/Bottom10=`0.5221/0.4627`，spread=`+0.0594`。
- Breakout OOS：Daily/Global rho=`0.0895/0.0376`，Pair=`53.46%`，Top10/Bottom10=`0.5232/0.4620`，spread=`+0.0612`。
- OOS Ranking：NDCG@10=`0.5465`，Top-K Target=`0.5444`，Lift=`+0.0444`，Boundary=`48.14%`，Boundary gap=`-0.0159`。
- Breakout Ranking：NDCG@10=`0.6970`，Top-K Target=`0.5251`，Lift=`+0.0187`，Boundary=`51.96%`，Boundary gap=`+0.0141`。
- Generalization：Validation→OOS ΔDaily rho=`-0.0592`、ΔPair=`-1.93pp`；OOS→Breakout ΔDaily rho=`-0.0104`、ΔPair=`+0.07pp`，表示application slice沒有額外collapse，但基礎signal本身偏弱。

#### 與MR-13AC的對稱比較

MR-13AC同樣是PIT-safe兩階段conditional residual設計，但方向為Predicted-Upside→Conditional Safety；其Seed42 Validation/OOS/Breakout Daily rho=`0.2278/0.1650/0.1900`。MR-13AD反向後為`0.1591/0.0999/0.0895`，OOS與Breakout分別低`0.0651`與`0.1005`。因此「先預測Safety，再學conditional MFE residual」沒有帶來預期的learnability優勢，反而明顯弱於AC方向。

#### 重要解讀邊界

使用者提出的原始直覺包含「Safety有下限／MFE右尾相對無上限」。MR-13AD作為**精確MR-13AC methodology reverse-control**，Stage-2在OLS residualization之前先把Pure-MFE轉成same-date percentile，因此原始MFE magnitude/right-tail已被rank normalization壓縮到`[0,1]`。所以本實驗可以否定的是：

- 在相同percentile-residual methodology下，交換conditioning direction並沒有改善可學性；
- 不應把本FAIL外推成「raw MFE magnitude/right-tail沒有研究價值」。若未來要測該假說，必須另立新的MR identity與預註冊target，不得覆寫MR-13AD。

#### Decision

`REVERSE_CONDITIONAL_LEARNABILITY_WEAK / FORWARD_MODEL_GATE_FAIL / NO_PIT / NO_STRATEGY_CONVERSION / STOP`

依事前stop rule，不建立`CONT13AD_ROLL`、不建立Strategy arm、不做MR-13AD Multi-seed或Fixed。MR-13AC/C79既有Safety conversion evidence不受本結果否定。由於C79 robustness先前只為等待MR-13AD對稱cell而延後，MR-13AD結案後，下一個最小必要證據回到**C59 vs C79 DL-source-only Extending Multi-seed robustness**；不先開新模型或Fixed。

### 2026-08-29 — MR-13AE Seed42 Model Gate：Pure-MFE learnability保留，但Safety context被主動反向利用

- Seed42 selected epoch=`1`。Validation/OOS/Breakout Daily rho=`0.4225/0.3937/0.3542`；OOS/Breakout Pair=`63.60/64.05%`。相較MR-13K，Pure-MFE ordering沒有collapse，證明「Safety context + 原始Pure-MFE supervision」的learnability本身成立。
- Model-specific geometry揭露相反機制：Validation/OOS/Breakout `Score→Low-Adverse Daily rho=-0.3091/-0.3242/-0.3083`；`Pred-Safety→Score=-0.8277/-0.8269/-0.7893`。因此Safety並非被忽略，而是被network當成「越不安全越可能高MFE」的shortcut。
- OOS Top10 MFE/Adverse=`2.5243R/0.5959R`，High-MFE/High-Safety/HMHS=`75.66/29.01/22.07%`；Breakout=`2.9345R/0.5756R`、`75.22/31.56/25.06%`。Downside geometry與研究目的方向相反。
- Decision=`PURE_MFE_LEARNABILITY_PRESERVED / SAFETY_CONTEXT_ACTIVELY_INVERTED / DOWNSIDE_GEOMETRY_FAIL / FORWARD_MODEL_GATE_FAIL / NO_PIT / NO_STRATEGY_CONVERSION / STOP`。不做AE Multi-seed/Fixed/Strategy。
- 使用者再次指出joint-head/joint-target family先前已大量驗證：component heads各自可學，但HM/HS joint交集learnability差；conditional family之所以保留，是因至少仍有可學訊號。因此下一步不得回到MR-13O/MR-13Z式joint objective。

### 2026-08-29 — MR-13AF formal Model Gate result / MR-13AG directional pair-weight follow-up

- **MR-13AF formal result**：Seed42 selected epoch=`1`；post-refit Validation/OOS/Breakout Pure-MFE Daily rho=`0.4351/0.3764/0.3479`，OOS/Breakout Pair=`62.97/63.84%`。OOS Model-specific diagnostics：`Pred-Safety→Score=-0.7763`、`Score→Low-Adverse=-0.3038`、Top10 MFE=`2.5154R`、Top10 Adverse=`0.5818R`、High-MFE=`75.24%`、High-Safety=`29.86%`、HM/HS=`22.43%`；Breakout=`-0.7600/-0.3011`、Top10 MFE/Adverse=`2.9144R/0.5572R`、High-MFE/High-Safety/HMHS=`73.69/32.63/25.11%`。
- **AF判讀**：相較MR-13AE OOS `Pred-Safety→Score=-0.8269`、`Score→Low-Adverse=-0.3242`，AF方向一致改善且MFE learnability只小幅下降；相較既有joint/multi-head常見`Pred-Safety↔Pred-MFE=-0.91~-0.94`，AF在不造成MR-13Y式MFE learnability collapse下把anti-correlation推至`-0.7763`，屬有效frontier evidence。但absolute Safety inversion仍強，High-Safety/HMHS只小幅提高，因此Decision=`PURE_MFE_LEARNABILITY_PRESERVED / SAFETY_INVERSION_PARTIALLY_REDUCED / EFFECT_PRESENT_BUT_INSUFFICIENT / FORWARD_MODEL_GATE_FAIL / NO_PIT / NO_STRATEGY / NO_ROBUSTNESS / STOP`。
- **AF結構缺口**：`min(S_i,S_j)`是對稱pair weight。它正確降權`unsafe MFE winner > safe loser`，但也同時降權我們真正希望強學的`safe MFE winner > unsafe loser`；因此下一步不調平方/exponent/cutoff，而只修正這個directionality。
- **新identity**：`MR-13AG`；profile=`daily_universal_predicted_safety_winner_weighted_pure_mfe_full_list_ndcg_pairwise`；architecture=`inception_time_v1`；target仍為MR-13K canonical `daily_full_horizon_pure_mfe_r_v1`。
- **Scientific change only**：Stage-1固定MR-13M/Seed42 canonical PIT-safe predicted-Safety context；Stage-2固定MR-13K InceptionTime、Pure-MFE target/order、split、optimizer、epoch selection與full-list ΔNDCG。Predicted Safety不進network。對每個同日non-tied Pure-MFE pair，base relevance再乘較高Pure-MFE item的Safety percentile：`w = ΔNDCG_MFE × S_winner`；safe winner/unsafe loser保留高權重，unsafe winner/safe loser低權重。Pair truth永不反轉，最後仍為`sum(loss*w)/sum(w)`。
- **No magic-strength extension**：不增加`S^p`、mean normalization、bucket、Safety cutoff、lambda、temperature、joint head、residual target、product或portfolio state。由於weighted mean本身除以weight sum，額外`S_winner / mean(S_winner)`只會整體乘除相消，故明確不加入。
- **Major-refactor integration review**：依使用者要求先重新盤點大幅重構後完整鏈路。確認scientific profile owner=`config/breakout_quality.py`、generic runtime capability owner=`config/breakout_quality_runtime.py`、acyclic resolver gateway=`config/breakout_quality_runtime_resolver.py`，data/training/dependency/report consumers均經`ContinuousRankerExecutionRecipe`。AG不新增profile-specific trainer branch；補齊既有objective seam，將base `pairwise_reduction`與generic `pair_weight_policy`分離。AF legacy combined reduction仍保留相同scientific/artifact identity，runtime只做compatibility normalization到`min_predicted_safety` policy。
- **Model Gate authorization**：只先跑Seed42 Forward Model Gate；`selection_pit_authorized=False`、`current_time_validation_authorized=False`，不建立`CONT13AG_ROLL`、Strategy Compare arm、Fixed或Multi-seed。只有在MFE learnability仍明顯、Safety anti-correlation/Actual Safety/Top10 Adverse/High-Safety/HMHS形成足夠大的共同改善後，才另行考慮conversion；robustness仍受使用者effect-size gate約束。

### 2026-08-29 — MR-13AG formal Model Gate result / MR-13AH symmetric Safety-product follow-up

- **MR-13AG formal result**：Seed42 selected epoch=`3`；epoch-selection best Validation Daily rho=`0.0269`，post-refit Validation=`0.3159`。Forward OOS Daily rho/Pair=`-0.0266/49.04%`，Breakout=`-0.1415/44.61%`。OOS Top10/Bottom10 target=`1.2716R/1.3840R`、Top-K Lift=`-0.2019R`；Breakout Top10/Bottom10=`1.0748R/1.8818R`、Top-K Lift=`-0.2098R`。Train loss從`0.513764`快速降至`0.052090`，但forward MFE ordering反轉。
- **AG判讀**：directional `S_winner` weighting不只是降低unsafe shortcut，而是讓future MFE winner的Safety直接決定supervision strength；在actual MFE/Safety存在結構性衝突時，這會系統性弱化大量真實high-MFE winner，造成Pure-MFE learnability collapse。Decision=`MFE_LEARNABILITY_COLLAPSE / FORWARD_ORDERING_REVERSED / APPLICATION_SLICE_COLLAPSE / FORWARD_MODEL_GATE_FAIL / NO_PIT / NO_STRATEGY / NO_ROBUSTNESS / STOP`。不需以Safety decorrelation是否改善來救回此cell。
- **新identity**：`MR-13AH`；profile=`daily_universal_predicted_safety_product_weighted_pure_mfe_full_list_ndcg_pairwise`；architecture=`inception_time_v1`；target仍為MR-13K canonical `daily_full_horizon_pure_mfe_r_v1`。
- **Scientific change only**：Stage-1固定MR-13M/Seed42 canonical PIT-safe predicted-Safety context；Stage-2固定MR-13K InceptionTime、Pure-MFE target/order、split、optimizer、epoch selection與full-list ΔNDCG。Predicted Safety不進network。唯一改動為對稱pair multiplier：`w = ΔNDCG_MFE × S_i × S_j`，最後仍為`sum(loss*w)/sum(w)`。相較AF `min(S_i,S_j)`，product更強壓低low/low pairs並集中high-Safety manifold；相較AG，weight不依MFE winner方向而變。
- **No magic-strength extension**：不增加bucket、cutoff、lambda、Safety exponent、temperature、joint head、residual target、product score或portfolio state。Safety全1時loss必須精確退化MR-13K。
- **Round 7 integration**：最新baseline已把pair-weight收斂為`config/breakout_quality_runtime.py` plugin registry，因此AH不修改generic trainer/data/report；只新增一個reusable product multiplier policy與declarative scientific profile/spec，consumer透過`ContinuousRankerExecutionRecipe`自動取得context/dependency/loss/report metadata。
- **Model Gate authorization**：只先跑Seed42 Forward Model Gate；`selection_pit_authorized=False`、`current_time_validation_authorized=False`，不建立PIT/Strategy/Fixed/Multi-seed。

### 2026-08-29 — MR-13AH formal Model Gate result / MR-13AI Phase-0 local-conflict follow-up

- **MR-13AH result**：Seed42 selected epoch=`1`；Validation/OOS/Breakout Daily rho=`0.4451/0.3522/0.3195`，OOS/Breakout Pair=`62.13/62.15%`。OOS `Pred-Safety→Score=-0.7189`、`Score→Low-Adverse=-0.2891`、Top10 MFE/Adverse=`2.3473R/0.5773R`、High-MFE/High-Safety/HMHS=`73.92/30.29/22.48%`；Breakout=`-0.6916/-0.2902`、Top10 MFE/Adverse=`2.7127R/0.5528R`、High-MFE/High-Safety/HMHS=`71.66/32.46/24.40%`。
- **Controlled interpretation**：相較MR-13AF，OOS `Pred-Safety→Score`改善`-0.7763→-0.7189`，但High-Safety只`29.86→30.29%`、HM/HS只`22.43→22.48%`、Adverse只`0.5818→0.5773R`；Breakout HM/HS反而`25.11→24.40%`。同時OOS MFE rho=`0.3764→0.3522`、Top-K Lift=`1.6503→1.3103R`。因此改善主要是prediction-space decorrelation，不是realized joint enrichment。
- **Decision**：`PREDICTED_SAFETY_ANTICORRELATION_REDUCED / MFE_LEARNABILITY_COST_MATERIAL / ACTUAL_SAFETY_GAIN_MINIMAL / HMHS_NOT_IMPROVED / FORWARD_MODEL_GATE_FAIL / STOP_PAIR_WEIGHT_STRENGTH_FAMILY / NO_PIT / NO_STRATEGY / NO_ROBUSTNESS`。
- **MR-13AI Phase-0 authorization**：不直接訓練新CNN。只讀MR-13AF frozen Forward-OOS score與canonical PIT-safe predicted-Safety，在每天canonical `K=10`邊界只比較rank K與K+1。為遵守no-lookahead，local MFE-confidence不用realized MFE gap，而用prediction-time AF score gap：`d_score=AF_score_K-AF_score_K+1`；`d_safety=S_K-S_K+1`；`g=d_score+(1-|d_score|)d_safety`；僅`g<0`交換。其餘rank完全保留AF。
- **Evidence surface**：Forward OOS與Breakout各自輸出AF vs AI的MFE Daily rho/Pair、Pred-Safety→Score、Score→Low-Adverse、Top-K MFE/Adverse/High-MFE/High-Safety/HMHS/Top-K Lift，另輸出swap rate與被交換pairs的realized MFE/Adverse/HMHS conversion。realized truth只在decision後評估。
- **Gate**：若Top-K MFE只小幅下降且Adverse/High-Safety/HMHS有實質改善，才另行定義trainable Phase-1；若zero-retrain boundary reorder都無joint gain，MR-13AI直接STOP。
- **Implementation boundary**：一次性read-only research diagnostic；不修改Standard Model SOP fingerprint、不新增model profile、不改正式model menu、不建立PIT/Strategy arm。

### 2026-08-29 — MR-13AI Phase-0 result / disposable implementation retirement / MR-13AJ direct DL follow-up

- **MR-13AI result**：Forward OOS boundary swaps=`578/1,203 (48.05%)`，AI−AF ΔMFE=`-0.0132R`、ΔAdverse=`-0.0059R`、ΔHigh-Safety=`+0.36pp`、ΔHM/HS=`+0.24pp`；Breakout swaps=`308/603 (51.08%)`，ΔMFE=`-0.0264R`、ΔAdverse=`-0.0071R`、ΔHigh-Safety=`+0.85pp`、ΔHM/HS=`-0.03pp`。近半數boundary decision被改寫，但joint outcome改善很小且Breakout HM/HS不重現。Decision=`LOCAL_BOUNDARY_REORDER_EFFECT_WEAK / ACTUAL_SAFETY_GAIN_SMALL / HMHS_NOT_REPLICATED_ON_BREAKOUT / NO_PHASE1 / STOP`。
- **Engineering lifecycle**：使用者明確要求不再為此類一次性Audit投入超過模型訓練本身的工程時間，且已完成Phase-0的service/CLI/dedicated synthetic沒有後續runtime/compatibility dependency。依PROJECT_SETTINGS C13，本輪刪除`services/breakout_quality/mr13ai_phase0.py`、`tools/filters/breakout_quality/mr13ai_phase0.py`與專屬synthetic registration/checklist current contract；MR-13AI scientific identity、公式、結果與decision只由Registry/Log保留歷史追溯。
- **MR-13AJ identity**：profile=`daily_universal_predicted_safety_conflict_discounted_pure_mfe_full_list_ndcg_pairwise`；architecture=`inception_time_v1`；target仍為MR-13K `daily_full_horizon_pure_mfe_r_v1`；Stage-1仍重用MR-13M/Seed42 canonical PIT-safe predicted-Safety context，Safety不進network。
- **唯一scientific change**：base仍為full-list ΔNDCG Pure-MFE pair loss。若MFE winner同時較安全（MFE與Safety ordering一致）或Safety tie，pair multiplier=`1`；只有MFE winner較不安全的conflict pair才 multiplier=`S_winner`。因此保留AG曾削弱的aligned MFE supervision，只降低unsafe-high-MFE conflict pressure；pair truth永不反轉。沒有bucket/cutoff/lambda/exponent/temperature/joint head/residual target/portfolio state/OOS fitting。
- **Authorization**：只先跑Seed42 Forward Model Gate；`selection_pit_authorized=False`、`current_time_validation_authorized=False`。若MFE learnability仍大幅下降或actual Safety/HMHS沒有實質改善即STOP；不再做pair-weight strength sweep或Phase-1 Audit。

### 2026-08-29 — C80/C81 single-seed result / allocator-path mechanism Audit launch

- **Status**：`RESULT_AVAILABLE / C80-C81 SAME-SOURCE ALLOCATOR CONTRAST CONFIRMED / MECHANISM_AUDIT_ACTIVE / NO_PROMOTION`。本輪不建立新模型、不改MR-13AH scientific identity、不調K/R0；只新增一次性read-only `AUD-c80-c81-allocator-path-attribution`，消費既有completed OOS/Rolling sidecars與canonical truth/Close。
- **C80 vs C81 core**：OOS Return=`165.13 vs 75.15%`、MDD=`14.93 vs 22.37%`、RoMD=`11.06 vs 3.36`、EV=`0.75 vs 0.22R`；Rolling Return=`173.87 vs 129.79%`、MDD=`16.06 vs 29.67%`、RoMD=`10.83 vs 4.37`、EV=`0.68 vs 0.45R`。C80 exposure=`90.56/92.63%`，C81=`59.78/60.71%`，所以C81劣化不能以投入較少單獨解釋。
- **Selection/path paradox**：C81 Selection Target mean OOS/Rolling=`1.79/2.07R`、Full-MFE=`1.71/1.93R`，且既有報表的+1/+2/+3R前初始Stop並不差；但Filled HM/LS=`35.31/39.29%`、High-Safety=`42.32/39.84%`、Adverse=`0.40/0.49R`，相較C80 HM/LS=`14.90/14.01%`、High-Safety=`65.23/63.52%`、Adverse=`0.25/0.27R`明顯惡化。C80同時把High-MFE由C81的`56.87/57.97%`降至`36.09/34.20%`，卻大幅改善RoMD/MDD，形成明確mechanism question。
- **Audit design**：一次性Audit固定比較同一`CONT13AH_ROLL`的C80 exact `preserve_k_r0=True`與C81 `preserve_k=False/preserve_r0=False`。輸出：(1) unconditional +1/+2/+3R reached；(2) canonical risk、all stop、initial stop、raised stop before first passage；(3) winner median MFE capture與giveback；(4) C80-only/C81-only/common planned cohorts；(5) C80 direct-infeasible日Raw Top-K dropped vs constrained-added MFE×Safety geometry與K/R0 binding；(6) MFE×Adverse→Realized EV matrix；(7) exact peak-EOD→trough-EOD MTM quadrant contribution。
- **Attribution boundary**：C80的K、R0與canonical cash feasibility/exact basket search是joint contract；沒有K-only/R0-only counterfactual前不得宣稱單一K或單一R0因果。Future truth只作post-replay diagnosis，不進runtime selection。
- **Stop rule**：OOS與Rolling若能共同定位membership substitution→path conversion→drawdown contributor，即停止Audit並做下一步決策；不得為補證據重訓模型、backfill sidecar或strategy replay。
- **GPT端工程驗證**：Audit config/catalog/handler contract PASS；`AUDIT_FRAMEWORK` dedicated synthetic=`23 checks / 0 FAIL`；新增deep first-passage/MFE×Adverse helper synthetic PASS；C80 Raw Top-K→exact K/R0 substitution synthetic PASS；drawdown quadrant key mapping PASS；修改Python檔`py_compile` PASS。使用者本機completed Strategy sidecars/market Close未包含於交付程式ZIP，因此真正Audit result仍維持`RESULT_PENDING`，不得預標機制成立。

### 2026-08-30 — MR-13AJ Seed42 Forward Model Gate result / STOP / MR-13H Standard SOP v2 BASE comparison next

- **Program basis**：使用者提供 `test-branch-1_20260829_235339_7c6dbacb.zip`；正式 `[1]→[1]` 執行MR-13AJ，Standard Model SOP v2共同MFE×Safety欄位已正常輸出。
- **MR-13AJ primary learnability**：selected epoch=`1`；Validation/OOS/Breakout Pure-MFE Daily rho=`0.0598/-0.0633/-0.1945`、Pair=`52.11/47.94/43.54%`。OOS/Breakout Top-K Lift=`-0.2997/-0.3118R`，已低於random ordering並在Breakout顯著惡化。
- **Standard SOP v2 mechanism evidence**：MR-13AJ target是canonical Pure-MFE，因此OOS `Target→Full-MFE rho=1.0000`；但`Score→Full-MFE=-0.0633`。同時OOS `Pred-Safety→Score=+0.2615`、`Score→Low-Adverse=+0.0994`；Breakout為`Score→Full-MFE=-0.1945`、`Pred-Safety→Score=+0.4362`、`Score→Low-Adverse=+0.1430`。Top10 OOS MFE/Adverse=`1.1883R/0.2856R`、High-MFE/High-Safety/HMHS=`47.04/57.44/26.06%`。這不是單純MFE rho小幅付費，而是score ordering由Pure-MFE方向反轉並轉向Safety。
- **Decision**：`MFE_ORDERING_COLLAPSE / SAFETY_DIRECTION_FLIPPED_POSITIVE / FORWARD_MODEL_GATE_FAIL / STOP`。依AJ事前stop rule，不做PIT、Strategy conversion、Multi-seed、Fixed或weight-strength sweep；conflict-only unsafe-winner discount不再延伸。
- **Next BASE decision**：使用者已要求在MR-13H與MR-13AF間選長期改善BASE，並已授權Standard SOP v2共同比較欄位。既有AF evidence保留；下一步只把Active Model Research Profile切回既有MR-13H scientific identity，正式重跑一次Seed42 `[1]→[1]`，補齊同口徑`Target/Score→MFE/Low-Adverse`、`Pred-Safety→Target/Score`與Top10 MFE/Adverse/High-MFE/High-Safety/HMHS，再做H vs AF BASE decision。此切換不是promotion，也不改production workflow anchor。

### 2026-08-30 — MR-13H Standard SOP v2 rerun / H vs AF BASE decision complete

- **Program basis**：使用者提供 `test-branch-1_20260830_000548_d630bc77.zip` 與正式 `[1]→[1]` MR-13H輸出；MR-13H scientific identity未改，Standard Model SOP fingerprint=`f220fdd4542ee32b`。
- **MR-13H rerun**：selected epoch=`1`；post-refit Validation/OOS/Breakout self-target Daily rho=`0.2659/0.2310/0.2211`，OOS/Breakout Pair=`57.93/59.32%`，Top-K Lift=`+1.1759/+0.5355R`。結果與既有H headline一致，故不是模型重訓異常。
- **Common upside/downside evidence**：OOS `Target→MFE=0.8638`、`Target→Low-Adverse=+0.2493`，證明H truth本身確實同時含upside/downside；但frozen `Score→MFE=0.3853`、`Score→Low-Adverse=-0.3230`、`Pred-Safety→Score=-0.8193`。Breakout為`0.3581/-0.3017/-0.7646`。也就是H模型主要保留upside relation，downside relation相對target方向反轉。
- **Top-tail geometry**：H OOS Top10 MFE/Adverse=`2.4702R/0.5729R`、High-MFE/High-Safety/HMHS=`74.25/30.22/22.40%`；Breakout=`2.9349R/0.5605R`、`74.04/30.87/24.72%`。AF OOS對照=`2.5154R/0.5818R`、`75.24/29.86/22.43%`；Breakout=`2.9144R/0.5572R`、`73.69/32.63/25.11%`。兩者actual joint tail基本同級，H沒有因downside-aware target形成可重複Safety/HMHS增量。
- **Precommitted BASE rule application**：事前規則要求H必須在保有可接受MFE learnability時，Safety/HMHS geometry明顯優於AF才選H；若H score丟失Safety或放大unsafe bias則重新評估AF。實測H `Pred-Safety→Score` OOS比AF更負`-0.8193 vs -0.7763`，`Score→Low-Adverse`也更負`-0.3230 vs -0.3038`，HM/HS OOS近乎相同且Breakout略低。故H不符合選BASE條件。
- **Decision**：`MR-13AF = SELECTED_LONG_TERM_RESEARCH_BASE`；`MR-13H = BASE_NOT_SELECTED`。此決策只定義後續DL改善的scientific control，不撤銷AF歷史`FORWARD_MODEL_GATE_FAIL`，不授權AF本身PIT/Strategy/robustness，不改production MR-13E。Active Model Research Profile切回既有MR-13AF；下一個改善需另建新MR identity，具體scientific design尚未授權。

### 2026-08-30 — B314 Model Standard SOP true report SSOT across Forward / Rolling / Robustness

- **Authorization / observed defect**：使用者實際執行確認`[1] Forward`與`[2] Rolling`、`[3] Forward comparison`與`[4] Rolling comparison`雖共享model設定，報表payload/producer仍不同源；並要求`[5]/[6]`同樣與`[3]/[4]`共用報表且刪除舊不同源code，避免未來誤接。
- **Single metric owner**：新增`services/breakout_quality/standard_model_sop.py`作唯一Standard SOP metric builder。Forward、Rolling、Robustness只提供各自evaluation score frames；Learnability、Generalization、MFE/Safety alignment、Top-tail、Ranking/Boundary與Evidence Coverage全部由同一builder產生，不准PIT audit、trainer或renderer另算第二套同名metric。
- **Rolling evidence parity**：Rolling fold新增post-refit `validation_scores.csv` canonical sidecar，因此每個Rolling report同樣具Validation/OOS/Breakout三個scope與`Validation → OOS`、`OOS → Breakout slice`。既有fitting-compatible fold缺sidecar時可checkpoint-only rescore補齊，不需重訓；cache-sync manifest同時驗checkpoint、score-period scores與validation scores。
- **Deleted split report ownership**：PIT Audit內舊Standard-SOP組裝、獨立Forward/Rolling comparison runners、舊Strategy-Robustness report menu route與rolling-specific persistent report contracts全部退出runtime。歷史output可保留作evidence，但runtime不留第二套producer/renderer alias。
- **Persistent contract**：唯一`model.standard_sop`升v7，approved fingerprint=`56e5fb1173404d7f`；唯一`model.standard_comparison`升v6，approved fingerprint=`7b11c0b9ddf50e41`。Rolling只追加`Rolling-specific Extension｜Fold / Year Stability`；Robustness只追加`Robustness-specific Extension｜Across-seed Stability`，extension不得改Standard 1～6 schema。
- **Robustness**：`[5]/[6]`改成model-level robustness route，從同一Model Compare/Test List取得模型；各benchmark seed先建立相同Standard SOP payload，再由generic aggregate建立同schema summary。Rolling robustness同時可帶Rolling extension＋Across-seed extension，不再接Strategy robustness報表。
- **Independent GPT validation**：以formal bundle retained primary params執行完整synthetic consistency=`4710 checks / 244 validators / 0 FAIL`；source compileall PASS。正式`apps/run_bundle.py`/`apps/test_suite.py`依PROJECT_SETTINGS不由GPT執行。

### 2026-09-01 — MR-13AK A1 decision → MR-13AL A2 + B327 Rolling multi-head producer closure

- **Authoritative baseline**：`test-branch-1_20260901_182733_0eecf87e(1).zip`，SHA256=`53f84fca6aa257cf80d829dfb13c3365cee9db2562ae993923ecf2f7f2665ee2`；本輪fresh extract=`/mnt/data/mr13_a2_work_20260901_182733_0eecf87e_1`。
- **A1 result / decision**：MR-13AK Forward OOS Daily rho/Pair/Top-Bottom=`0.3843/63.19%/1.8360R`，Breakout=`0.3714/63.65%/2.6001R`；Rolling=`0.3844/63.26%/1.6985R`，Breakout=`0.3633/64.07%/2.3880R`，max adjacent drift=`0.7954 pooled σ / NO flag`。相對AH，shared encoder明顯提高MFE representation與time stability，因此A1 architecture question判PASS。
- **A1 Safety mechanism**：Forward Raw Safety head本身可學，OOS rho/Pair=`0.3509/62.58%`、Breakout=`0.3108/63.65%`；但Pred Safety↔Raw-MFE Daily rho=`-0.9223`（Breakout=`-0.8848`）、Pred S5×M5=`0`（Breakout=`6`）、joint product→actual HM/HS rho=`0.0310/0.0661`，final Score→Safety=`-0.3417`（Rolling=`-0.3351`）。Decision=`ARCHITECTURE_GATE_PASS / SAFETY_GEOMETRY_COLLAPSE_PERSISTS / GO_A2 / NO_AK_ROBUSTNESS_FIRST / NOT_PROMOTED`。
- **MR-13AL A2 identity**：分配下一個未占用`MR-13AL`，profile=`daily_universal_shared_safety_context_weighted_pure_mfe_full_list_ndcg_pairwise`。依Architecture E6/C22不建立新ARCH；直接重用既有`ARCH-inception_time_safety_conditional_mfe_v1` topology。A1的training objective、target builder、Safety/MFE truth、AH same-date Safety-percentile product weighting、pair direction、equal head-loss mean、optimizer/split/Seed固定；唯一executable change為final MFE head從`shared_latent`改為`concat(shared_latent, detach(Raw Safety probability))`。
- **Generic composition seam**：`forward_safety_mfe_heads`改為capability interface，可由model topology選擇independent Raw-MFE head或Safety-conditioned MFE head；trainer仍只消費Raw-Safety+final-MFE capability，不辨識MR/profile/architecture。shared-Safety-weighted training semantics中的head topology改由model spec擁有，loss contract本身保持topology-agnostic。
- **B327 Rolling extension closure**：B326 renderer本身已capability-driven；真正缺口是Rolling PIT score producer沒有為A1 `safety_raw_mfe` target builder持久化Raw Safety/Raw-MFE sidecars。現在score-output capability由`training_policy.target_builder`單一解析，A1/A2自動得到`raw_safety_score/raw_mfe_score`；`score_output_contract`仍被既有fitting identity排除，所以compatible舊AK fold checkpoint只checkpoint-only rescore，不因補head evidence重訓。PIT Audit直接消費sidecars，用canonical same-date truth percentile與既有`safety_raw_mfe_metrics`產生`safety_raw_mfe_evaluation`，B326 [4] comparison即可顯示Multi-head Learnability / Truth-Prediction Geometry。
- **Persistent report boundary**：本輪不改`model.standard_sop v7`或`model.standard_comparison v7` schema/fingerprint；只補B326已授權extension的Rolling evidence producer。
- **Validation before formal local suite**：Shared-AH composition synthetic=`16/16 PASS`；PIT builder/canonical score-only reuse=`26/26 PASS`；Continuous DL workflow=`44/44 PASS`；persistent report freeze=`25/25 PASS`。正式`apps/run_bundle.py` / `apps/test_suite.py`依PROJECT_SETTINGS不由GPT執行。

Decision：`MR13AK_A1_CLOSED_PASS_TO_A2 / MR13AL_IMPLEMENTED_RESULT_PENDING / ROLLING_MULTIHEAD_EVIDENCE_CLOSED / NO_REPORT_FINGERPRINT_CHANGE / FORWARD_SEED42_NEXT`。

### 2026-09-01 — MR-13AL A2 decision / MR-13AK robustness / MR-13AM A3 implementation

- **Observed A2 evidence**：MR-13AL相對MR-13AK只在部分MFE指標小幅改善。Forward Daily rho/Pair/Top-Bottom=`0.3909/63.50%/1.9136R` vs AK=`0.3843/63.19%/1.8360R`；Rolling=`0.3848/63.29%/1.7440R` vs AK=`0.3844/63.26%/1.6985R`。但A2主要Safety mechanism沒有改善：Forward Pred Safety↔Raw-MFE=`-0.9446`（AK=`-0.9223`）、Pred S5×M5=`0`、Joint→actual HM/HS rho=`0.0343`、Score→Safety=`-0.3464`；Rolling Score→Safety=`-0.3401`、High-Safety/HM/HS=`29.58/21.62%`。Decision=`CONTEXT_INCREMENT_NOT_SUPPORTED / SAFETY_GEOMETRY_NOT_IMPROVED / NO_AL_ROBUSTNESS / CLOSED / NOT_PROMOTED`。
- **AK robustness evidence**：8-seed Rolling robustness完成後，MR-13AK OOS Daily rho mean/σ=`0.3893/0.0147`、Pair mean/σ=`63.44%/0.56%`、Top-Bottom mean/σ=`1.5567R/0.2664R`；Breakout Daily rho/Pair/Top-Bottom=`0.3668/64.19%/2.3143R`。相對H/AF/AH，AK同時具有最高mean Daily rho/Pair與最低rho/Pair seed dispersion，確認A1 shared representation gain不是Seed42偶然。Safety仍弱：robust Score→Safety=`-0.3381`、High-Safety=`29.59%`。Decision=`ROBUST_MFE_BASE_CONFIRMED / SAFETY_WEAKNESS_PERSISTS / A3_BASE / NO_RUNTIME_PROMOTION`。
- **User decision**：A3不由失敗的AL context topology延伸，改成**AK → A3**；並將MR-13AL從current Model Compare/Test List移除，但保留其historical identity/results/artifacts。
- **MR-13AM identity**：分配下一個未占用`MR-13AM`，profile=`daily_universal_shared_safety_weighted_full_horizon_opportunity_full_list_ndcg_pairwise`。重用AK `ARCH-inception_time_shared_safety_mfe_v1` topology，不使用AL Safety-context input。AK的Low-Adverse Safety truth、Safety full-list ΔNDCG、same-date predicted-Safety percentile `detach(S_i)×detach(S_j)` pair weighting、pair direction、equal head-loss mean、Seed/split/optimizer全部固定；唯一scientific change為final primary truth由Pure-MFE percentile改成MR-13H `daily_full_horizon_opportunity_r_v1` percentile。
- **Generic capability implementation**：A3使用generic `safety_primary` target builder／shared-Safety-weighted-primary composition，使第二head可由runtime contract標示`Economic Target`，而非錯稱Raw-MFE；Raw-MFE-specific Truth/Prediction Geometry extension不套用AM。model topology、score-output與training semantics仍由canonical capability/model-spec owner派生；不在trainer/menu/renderer增加MR-13AM identity branch。
- **Membership SSOT**：current training pair切至`MR-13AM`後，B325 ordered merge自動得到`MR-13H / MR-13AF / MR-13AH / MR-13AK / MR-13AM`；MR-13AL不再是reference control，因此自然退出[3]～[6] current comparison/robustness list，沒有第二份exclude清單。
- **Report boundary**：B329 `model.standard_comparison v8` / fingerprint=`aa7f812b9d1d936a`不變；本輪沒有新增、刪除或重排persistent report section。
- **Implementation provenance**：B329文件記錄的完整baseline `test-branch-1_20260901_193732_d9e902aa.zip` binary在本輪sandbox不可取，因此工作樹由可用`test-branch-1_20260901_182733_0eecf87e(1).zip` + 已交付MR-13AL/B327 patch + B329 patch重建，並依B327 capability-owner contract補回owner closure後實作A3；交付內容以replacement patch形式包含所有相對該可驗證重建基準的必要changed files。
- **Targeted validation before formal local suite**：Shared-AH/A3 composition=`19/19 PASS`；PIT score-output ownership=`26/26 PASS`；Continuous DL workflow=`44/44 PASS`；simple report=`15/15 PASS`；persistent report freeze=`27/27 PASS`。正式`apps/run_bundle.py` / `apps/test_suite.py`依PROJECT_SETTINGS不由GPT執行。

Decision：`MR13AL_CLOSED_NEGATIVE_CONTROL / MR13AK_ROBUST_MFE_BASE / MR13AM_IMPLEMENTED_RESULT_PENDING / AL_REMOVED_FROM_CURRENT_COMPARE_LIST / FORWARD_SEED42_NEXT`。

### 2026-09-01 — MR-13AN decision / MR-13AK strategy conversion C82/C83 implementation

- **MR-13AN result**：Seed42 Forward own-target Daily rho/Global rho/Pair/Top-Bottom=`0.2211/0.1945/57.62%/1.3947R`；Breakout=`0.2055/0.2272/58.60%/2.1071R`。相對MR-13AM，Score→Safety由`-0.3442`改善至`-0.3308`、High-Safety=`29.50→30.92%`、Top10 Low-Adverse=`-0.5854→-0.5563R`，但Score→MFE=`0.3883→0.3775`、Top-K Lift=`1.0300→0.7755R`、Oracle overlap=`7.97→5.28%`；Breakout HM/HS=`24.95→23.89%`反向退化。Decision=`SAFETY_GAIN_SMALL_WITH_RANKING_COST / MODEL_GATE_NO_GO / NO_ROLLING / NO_ROBUSTNESS / CLOSED / NOT_PROMOTED`。
- **User decision**：使用者明確要求進行MR-13AK strategy conversion。模型層不再新增Safety head/target cell；current model research focus回到AK robust MFE base。
- **New strategy source**：`CONT13AK_ROLL`=`MR-13AK / inception_time_shared_safety_mfe_v1 / daily_universal_shared_safety_weighted_pure_mfe_full_list_ndcg_pairwise / selection_point_in_time`。只建立downstream strategy binding，不改AK target、architecture、loss、score semantics或production。
- **SR-C82**：完整複製C80 constrained contract，唯一change=`CONT13AH_ROLL → CONT13AK_ROLL`。Primary contrast=`C82-C80`。
- **SR-C83**：完整複製C81 No-K/No-R0 direct contract，唯一change=`CONT13AH_ROLL → CONT13AK_ROLL`。Primary contrast=`C83-C81`；同AK allocator secondary=`C82-C83`。
- **Compare Suite**：`extending_current` schema=`67`；catalog新增C82/C83與三個controlled contrasts。current effective membership仍由shared Model Compare/Test List＋strategy binding派生，不建立第二份手寫current matrix。
- **Initial scope / stop rule**：先只跑Seed42 Forward OOS與Rolling OOS。核心看Return/MDD/RoMD/EV、Exposure、High-MFE/High-Safety/HMHS、Adverse與initial-stop/path conversion；只有相對AH direct controls形成足夠大的實務差異才另行決定strategy robustness，不因小幅領先自動進8-seed。
- **Production boundary**：C42/C44與runtime promotion truth完全不變。

### 2026-09-02 — MR-13AO Seed42 Forward Model Gate result

- **True-HS learnability**：Forward true-HS Conditional-MFE Daily rho/Pair=`0.3969/63.90%`；Breakout=`0.4116/67.22%`。因此「移除HM/LS MFE supervision後，HS population內是否仍有right-tail upside signal」答案為YES。
- **Lexicographic Forward**：MR-13AO Pred-Safety P50→Conditional-MFE的High-MFE/High-Safety/HMHS/HMLS=`59.66/44.22/25.94/33.72%`；AK same-gate=`52.15/47.55/23.87/28.28%`。AO的HMHS `+2.07pp`伴隨更大的HMLS `+5.44pp`，主要是更強MFE ranking而非更純HMHS retrieval。
- **LS contamination**：Pred-HS true-LS=`36.84%`；AO Top-tail LS contamination lift=`1.51×`，true-LS Conditional-rank P50/P90/P99=`0.616/0.934/0.996`。LS並非隨機亂飄，而是high-MFE representation合理外推到LS。
- **Breakout replication**：AO vs AK same-gate High-MFE=`40.49 vs 39.26%`、High-Safety=`59.34 vs 60.55%`、HMHS=`24.07 vs 23.92% (+0.15pp)`、HMLS=`16.43 vs 15.34% (+1.09pp)`；HMHS增量未replicate。
- **Decision**：`TRUE_HS_MFE_LEARNABILITY_CONFIRMED / CONDITIONAL_MFE_EXTRAPOLATES_TO_LS / LS_CONTAMINATION_AMPLIFIED / FORWARD_HMHS_GAIN_MOSTLY_MFE_DRIVEN / BREAKOUT_HMHS_INCREMENT_NOT_REPLICATED / MODEL_GATE_FAIL / STOP / NO_ROLLING / NO_ROBUSTNESS / NO_STRATEGY`。

### 2026-09-02 — MR-13AP Seed42 Forward Model Gate result

- **Result**：selected epoch=`2`；HS-Priority Forward OOS rho/Pair=`0.1863/57.47%`，Breakout=`0.1434/56.73%`。
- **Direction collapse**：Forward Standard Score→MFE/Safety=`-0.3610/+0.3462`；Top10 High-MFE/High-Safety=`19.50/77.45%`，HM/HS/HM/LS=`14.93/4.57%`。Breakout Score→MFE/Safety=`-0.3609/+0.3197`，High-MFE/High-Safety=`26.59/71.48%`，HM/HS/HM/LS=`18.80/7.79%`。Validation epoch-2 HS-only MFE rho=`-0.3813`。
- **Interpretation**：AP成功讓HM/LS大幅下降，但方式是把final head學成Safety selector；LM/HS Forward=`62.52%`，未達「HS先qualify後在HS內最大化upside」。程式target與pair direction audit未發現MFE sign反轉。
- **Mechanism**：同日約半數HS時，HS↔LS pair數約為HS↔HS的2×，AP relevance gap又約4.5×；canonical full-list ΔNDCG synthetic顯示boundary/within-HS supervision weight mass約8.5～9.1×。因此single normalized loss天然由boundary主導。Decision=`HS_LS_BOUNDARY_LEARNABLE / HS_MFE_ORDERING_COLLAPSED_AND_REVERSED / SAFETY_DOMINATED_PRIORITY_OBJECTIVE / MODEL_GATE_FAIL / STOP / NO_ROLLING / NO_ROBUSTNESS`。

### 2026-09-02 — MR-13AQ Seed42 Forward Model Gate result

- **Result**：selected epoch=`4`；Forward HS-Priority rho/Pair=`-0.0127/49.47%`，Breakout=`+0.0165/50.61%`。pair-stratified normalization已消除MR-13AP的Safety-dominant extreme，但final HS-Priority ordering接近random。
- **Alignment**：Forward Standard Score→MFE/Safety=`+0.1230/-0.0623`、High-MFE/High-Safety=`54.78/48.74%`、HM/HS/HM/LS=`24.67/30.10%`；Breakout Score→MFE/Safety=`+0.0348/-0.0025`、HM/HS/HM/LS=`24.74/26.39%`，HM/HS enrichment僅約`1.04×`。
- **Epoch geometry**：Validation E1→E4 HS-Priority rho由`-0.0769→+0.0673`時，HS-only MFE rho由`+0.2916→-0.0253`；兩個stratum等權後不是共同改善，而是明顯互相抵消。
- **Decision**：`PAIR_MASS_DOMINANCE_REMOVED / HS_PRIORITY_ORDERING_NEAR_RANDOM / HS_MFE_AND_HS_LS_OBJECTIVES_CONFLICT / BREAKOUT_HMHS_ENRICHMENT_NOT_CONFIRMED / SINGLE_SCORE_HS_PRIORITY_FAMILY_STOP / MODEL_GATE_FAIL / NO_ROLLING / NO_ROBUSTNESS / NO_STRATEGY`。不做0.4/0.6等weight sweep。

### 2026-09-02 — MR-13AR Seed42 Forward Model Gate result

- **Qualification learnability**：Forward / Breakout HS-Qualification Pair=`67.37% / 69.46%`，證明binary HS↔LS ordering本身可學；但真正Pred-HS P50 gate的true-LS contamination=`37.05% / 36.59%`，Forward較MR-13AO reference `36.84%`反而`+0.21pp`，Breakout只`-0.35pp`。
- **Conditional-MFE preserved**：true-HS Forward rho/Pair=`0.3770/63.13%`，Breakout=`0.4077/66.25%`；相對AO `0.3969/63.90%`與`0.4116/67.22%`僅小幅下降，故第二head不是本輪blocker。
- **Lexicographic outcome**：Forward AR vs AO same-gate High-MFE=`57.61 vs 59.66%`、High-Safety=`43.49 vs 44.22%`、HM/HS=`24.49 vs 25.94%`、HM/LS=`33.13 vs 33.72%`；Breakout HM/HS=`23.98 vs 24.07%`、HM/LS=`16.13 vs 16.43%`。沒有HM/HS增量。
- **Decision**：`HS_QUALIFICATION_PAIR_LEARNABILITY_CONFIRMED / P50_BOUNDARY_PURITY_NOT_IMPROVED / CONDITIONAL_MFE_PRESERVED / FORWARD_HMHS_WORSE_THAN_AO / BREAKOUT_HMHS_FLAT / MODEL_GATE_FAIL / STOP / NO_ROLLING / NO_ROBUSTNESS`。

### 2026-09-02 — MR-13AV closure + MR-13AW AO-objective Safety temporal-attention pooling

- **Authoritative baseline**：`test-branch-1_20260902_202436_3a90020c(1).zip`，SHA256=`ad25e09a1540829a1a508fcffe74950681e53dcff001c5556434970b5ba423d1`；fresh workspace=`/mnt/data/stock_aw`。
- **MR-13AV closure**：AO objective固定、只分離最後完整residual group後，Forward final Daily rho=`0.4010` vs AO=`0.4042`；Breakout=`0.3868` vs `0.3850`。Forward Pred-HS true-LS僅`36.65%` vs AO=`36.84%`、P45–P55=`52.34%`；HM/HS/HM/LS=`26.70/32.00%` vs AO=`25.94/33.72%`，但Breakout true-LS惡化至`37.53%`且HM/LS=`17.50%` vs AO=`16.43%`。Rolling Daily rho=`0.4105` vs AO=`0.4136`，overall HM/HS/HM/LS=`21.68/54.94%` vs `21.79/54.30%`；fold score drift雖由AO `0.8551`降至AV `0.3399 pooledσ`，仍未形成一致Safety learnability突破。Decision=`TASK_SPECIFIC_HIGH_LEVEL_REPRESENTATION_VALID / HS_SAFETY_GAIN_NOT_CONSISTENT / MODEL_GATE_FAIL / NO_MULTI_SEED`。
- **Research direction correction**：使用者明確指出研究重點不應綁定P50或某個boundary區間；實際deployment未必使用P50 classification。下一階段primary objective改為提升**完整continuous Safety ranking learnability**，P50/Pred-HS/HMHS僅保留下游operating-point diagnostic。
- **MR-13AW identity**：profile=`daily_universal_shared_safety_attn_hs_conditional_mfe_full_list_ndcg_pairwise`；architecture=`inception_time_shared_safety_attn_mfe_v1`。MR-13AO continuous Safety full-list ΔNDCG、true-HS Conditional-MFE sublist、1:1 loss、raw 300×10、Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference完全固定；不沿用AV task-specific branch。
- **唯一scientific change**：共享InceptionTime feature map後，MFE仍使用AO Global Average Pool；Safety改用parameter-minimal scalar temporal attention pooling：`1×1 Conv scorer → softmax over time → weighted sum`。沒有attention heads、temperature、query count、window length、hidden width或其他新tuning knob。
- **Controlled-path invariant**：同seed下AO/AW共同state-dict參數初始化bit-identical，AW只多`safety_attention_scorer.weight/bias`；eval MFE logits與AO逐位相同。Safety loss對scorer有gradient，MFE-only loss對scorer為0，避免Safety pooling改動污染Conditional-MFE representation path。
- **Model-specific evidence**：依既有使用者授權，`hs_conditional_mfe_gate` cross-model extension第一表加入canonical Raw Safety `Daily rho / Global rho / Pair`，直接比較AO vs AW完整Safety learnability；Standard SOP 1～6與`model.standard_sop v7`不變，`model.standard_comparison`因extension schema升v10，approved comparison fingerprint=`2a47b1928a60f697`。
- **Current membership**：explicit references維持`MR-13H / MR-13AH / MR-13AK / MR-13AO`，B325 current-pair merge自動加入`MR-13AW`，因此[3]～[6] list=`H / AH / AK / AO / AW`。
- **Gate**：只先Seed42 Forward。Primary看Safety Daily rho、Global rho、Pair與Safety top-tail capture是否相對AO有實質提升，同時AO-style HS-only Conditional-MFE不可崩；P50 qualification/HMHS不作architecture primary success criterion。若Safety ranking仍無實質提升，停止InceptionTime pooling小修，再升級更強temporal representation/backbone；不得同輪做attention/cutoff/loss/architecture sweep。

Decision：`MR13AV_CLOSED / MR13AW_IMPLEMENTED_RESULT_PENDING / FULL_CONTINUOUS_SAFETY_LEARNABILITY_PRIMARY / AO_OBJECTIVE_EXACT_CONTROL / SAFETY_SCALAR_TEMPORAL_ATTENTION_ONLY / H_AH_AK_AO_AW_CURRENT_LIST / SEED42_FORWARD_NEXT`。

### 2026-09-02 — MR-13AW closure + MR-13AX task-specific Safety attention interaction cell

- **Authoritative baseline**：`test-branch-1_20260902_210114_1f0320ad(1).zip`，SHA256=`b63c4f9a694089fa117375a5d8bf290bab1161dea18fd9ee7161a8a934fe0ec1`；fresh workspace=`/mnt/data/stock_ax`。
- **MR-13AW closure**：MR-13AO objective/target/inference固定，只將shared feature map上的Safety GAP換成single-scalar temporal attention。Forward Safety Daily/Global rho/Pair=`0.3532/0.3009/62.65%` vs AO=`0.3532/0.2929/62.68%`；Breakout=`0.3218/0.3673/63.81%` vs AO=`0.3135/0.3554/63.65%`。Forward Daily rho完全未提升、Pair微降，僅Global rho `+0.0080`；Breakout三項有小幅同向訊號。Forward HM/HS/HM/LS=`26.28/33.14%` vs AO=`25.94/33.72%`，但Breakout HM/HS=`23.30%<24.07% AO`且High-MFE=`35.57%<40.49%`。Decision=`SAFETY_GLOBAL_BREAKOUT_SIGNAL_SMALL / FORWARD_DAILY_RANKING_UNCHANGED / MODEL_GATE_FAIL / NO_ROLLING / NO_MULTI_SEED`。
- **2×2 architecture rationale**：目前已完成AO=`shared+GAP`、AV=`task-specific+GAP`、AW=`shared+Safety attention`三格。使用者授權補最後一格，分配`MR-13AX`，直接測Safety temporal attention是否只有作用在Safety-specific high-level feature map時才有interaction gain；不再新增cutoff/loss/attention knob。
- **MR-13AX identity**：profile=`daily_universal_task_specific_safety_attn_hs_conditional_mfe_full_list_ndcg_pairwise`；architecture=`inception_time_task_specific_safety_attn_mfe_v1`。MR-13AV較低層shared residual groups、最後一個Safety/MFE task-specific residual group、MR-13AO continuous Safety full-list ΔNDCG、true-HS Conditional-MFE sublist、1:1 loss、raw 300×10、Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。
- **唯一scientific change**：Safety-specific feature map由Global Average Pool改成MR-13AW已存在的`1×1 Conv scalar scorer → softmax over time → weighted sum`；MFE-specific feature map仍使用GAP。沒有head count、temperature、query count、window、hidden width或branch-depth新參數。
- **Generic composition implementation**：architecture descriptor同時宣告既有`task_specific_safety_mfe + safety_attention_pool` capabilities；InceptionTime generic runtime在task-specific mode將attention作用於Safety-specific feature map，而不是shared map。AV與AW舊路徑完全不變。Canonical training semantics額外記錄AX的Safety pooling，但AV既有payload不漂移。
- **Controlled-path invariant**：same-seed AV/AX共同state-dict參數bit-identical，AX只多`safety_attention_scorer.weight/bias`；eval MFE logits必須逐位相同。Safety loss可更新shared stem＋Safety branch＋attention scorer，不得更新MFE branch；MFE loss可更新shared stem＋MFE branch，不得更新Safety branch/scorer。
- **Current membership**：explicit references維持`MR-13H / MR-13AH / MR-13AK / MR-13AO`，B325 current-pair merge自動加入`MR-13AX`，因此[3]～[6] list=`H / AH / AK / AO / AX`。
- **Gate**：Seed42 Forward first。Primary看Safety Daily rho、Global rho、Pair是否實質離開AO/AW平台且Breakout同方向；HS-only Conditional-MFE不得崩。P50 qualification/HMHS只作downstream diagnostic。若AX仍無明顯增量，停止high-level split/pooling組合小修，再升級feature-map內部temporal interaction；不做Multi-seed。

Decision：`MR13AW_CLOSED / MR13AX_IMPLEMENTED_RESULT_PENDING / AO_OBJECTIVE_FIXED / TASK_SPECIFIC_SAFETY_ATTENTION_INTERACTION_ONLY / H_AH_AK_AO_AX_CURRENT_LIST / SEED42_FORWARD_NEXT`。

### 2026-09-02 — MR-13AX closure + MR-13AY Safety temporal self-attention interaction

- **MR-13AX closure**：補齊AO/AV/AW/AX 2×2 split×pooling最後一格。Forward Safety Daily/Global rho/Pair=`0.3569/0.2891/62.86%` vs AO=`0.3532/0.2929/62.68%`；Breakout=`0.3054/0.3569/63.87%` vs AO=`0.3135/0.3554/63.65%`。Forward只有Daily `+0.0037`、Pair `+0.18pp`，Global反降；Breakout Daily `-0.0081`。Downstream Forward HM/HS/HM/LS=`26.79/29.14%`伴隨High-MFE降至`55.94%`；Breakout contamination=`33.96%`但recall=`56.60%`、HM/HS=`23.55%`。Decision=`SPLIT_ATTENTION_INTERACTION_NO_BREAKTHROUGH / MODEL_GATE_FAIL / NO_ROLLING / NO_MULTI_SEED`。
- **Research-level transition**：停止shared/task-specific × GAP/scalar-pooling組合小修。使用者明確要求繼續提升DL本身，而不是把研究綁在P50 operating point；下一個mechanism升級到feature-map內部temporal interaction。
- **MR-13AY identity**：profile=`daily_universal_shared_safety_self_attn_hs_conditional_mfe_full_list_ndcg_pairwise`；architecture=`inception_time_shared_safety_self_attn_mfe_v1`。回到MR-13AO shared InceptionTime strict control；continuous Safety full-list ΔNDCG、true-HS Conditional-MFE、1:1 loss、raw 300×10、Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。
- **唯一scientific change**：Safety path在AO shared feature map加入single-head full-width Q/K/V temporal self-attention，使用PyTorch canonical scaled-dot-product attention（`1/sqrt(C)`、dropout=0、non-causal），context residual加回feature map後仍GAP；MFE path仍原shared feature map→GAP。沒有FFN、positional encoding、attention window、head-count、hidden-width、temperature或額外dropout knob。
- **Primary Gate**：完整Safety Daily/Global rho與Pair需實質離開AO/AW/AX平台，且Forward/Breakout同方向；Conditional-MFE不可崩。P50 boundary、Pred-HS、HM/HS只作downstream診斷。若AY仍無明顯增量，不再做InceptionTime局部pooling/split/interaction小修，才進真正不同temporal architecture。
- **Current membership**：explicit references維持`MR-13H / MR-13AH / MR-13AK / MR-13AO`，B325 current-pair merge自動加入`MR-13AY`，因此[3]～[6] list=`H / AH / AK / AO / AY`。

### 2026-09-02 — MR-13AY result closure + MR-13AZ Safety-Patch / MFE-Inception implementation

- **Authoritative baseline**：`test-branch-1_20260902_220513_2afb359c(1).zip`，SHA256=`9054fae8c3a7d7425c78dce113cd4d6897bba2d4ada343222c4c22b9a1ab8212`；fresh workspace=`/mnt/data/stock_az`。
- **MR-13AY result**：Forward Safety Daily/Global rho/Pair=`0.3450/0.2917/62.38%` vs AO=`0.3532/0.2929/62.68%`；Breakout=`0.2906/0.3478/63.38%` vs AO=`0.3135/0.3554/63.65%`。HS-only Conditional-MFE Forward/Breakout rho=`0.3848/0.3992` vs AO=`0.3969/0.4116`。Decision=`TEMPORAL_SELF_ATTENTION_DEGRADES_SAFETY_RANKING / CONDITIONAL_MFE_DEGRADES / INCEPTIONTIME_LOCAL_ARCHITECTURE_PATCH_FAMILY_STOP / MODEL_GATE_FAIL / NO_ROLLING / NO_ROBUSTNESS`。
- **MR-13AZ identity**：profile=`daily_universal_patch_safety_inception_hs_conditional_mfe_full_list_ndcg_pairwise`，architecture=`patch_transformer_safety_inception_mfe_v1`。AO target/loss、true-HS scope、1:1 heads、Seed42、split/optimizer、epoch selection與inference固定。
- **Safety encoder**：沿用historical 9F/MR-13Z freeze recipe：non-overlap patch=10、embedding=128、3 encoder layers、4 heads、FFN=256、sinusoidal position、dropout=0.10、patch-token mean。無architecture sweep。
- **Conditional-MFE encoder**：AO-form InceptionTime + GAP；兩encoder完全獨立。Safety loss只更新Patch encoder/head，Conditional-MFE loss只更新InceptionTime encoder/MFE head；此gradient ownership是AZ topology contract的一部分，不誤稱為AO shared-gradient exact control。
- **Engineering primitive B347**：新增descriptor-driven hybrid builder，並將既有frozen Patch token encoder constructor正式暴露為public reusable primitive供Z/AA/AZ共用；generic trainer/inference仍只消費既有`forward_safety_mfe_heads`／output-head contract，不辨識MR-13AZ identity。same-seed hybrid內完整AO MFE submodel初始化可與AO standalone bit-identical，eval MFE logits exact；Patch/MFE gradients雙向隔離。
- **Authorization**：Seed42 Forward Model Gate first；Rolling/Robustness/PIT/strategy均不自動授權。current Model Compare/Test List=`MR-13H / MR-13AH / MR-13AK / MR-13AO / MR-13AZ`。

Decision：`MR13AZ_IMPLEMENTED_RESULT_PENDING / SAFETY_BACKBONE_REPRESENTATION_CONTRAST / FROZEN_PATCH_RECIPE / INDEPENDENT_MFE_INCEPTION_ENCODER / SEED42_FORWARD_NEXT`。

### 2026-09-02 — MR-13AZ closure + MR-13BA full-window receptive-field implementation

- **Authoritative baseline**：`test-branch-1_20260902_232551_3907c6e4(1).zip`，SHA256=`1c78d19bfdb25082444f6d47e6e52f61cd880a8220ea84733a494425b4683eb8`。
- **MR-13AZ result closure**：Forward Safety Daily rho=`0.3260`、Pred-HS true-LS=`38.13%`、HS-only Conditional-MFE rho=`0.3772`、TopK HM/HS/HM/LS=`23.18/37.78%`；相對AO未形成Safety突破，且qualification污染更差。Decision=`PATCH_SAFETY_REPRESENTATION_NO_BREAKTHROUGH / MODEL_GATE_FAIL / NO_ROLLING / NO_ROBUSTNESS`。
- **Research transition**：使用者明確停止繼續排列head/encoder representation，改查物理性限制；順序固定為receptive field → pure capacity → backbone family → input horizon。
- **MR-13BA identity**：profile=`daily_universal_shared_safety_hs_conditional_mfe_full_window_rf_full_list_ndcg_pairwise`；architecture=`inception_time_shared_safety_mfe_full_window_rf_v1`。AO continuous Safety full-list ΔNDCG、true-HS=P50 Conditional-MFE、1:1 loss、raw 300×10、Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。
- **唯一scientific change**：AO shared InceptionTime kernels/depth/width不變，只把最後一個residual group的module dilation由1改為2：`(1,1,1,1,1,1) → (1,1,1,1,2,2)`。因此實際receptive field=`229 → 305` bars；kernels仍=`(39,19,9)`，trainable parameter count與AO exact相同。
- **Primary Gate**：完整Safety Daily/Global rho與Pair需實質離開AO平台且Breakout同方向；HS-only Conditional-MFE不得崩。若FAIL，停止「AO有效RF未覆蓋300-bar input」假說，下一輪只測pure capacity scaling，不做dilation/RF sweep。

Decision：`MR13BA_IMPLEMENTED_RESULT_PENDING / FULL_WINDOW_RECEPTIVE_FIELD_CONTROL / SAME_PARAMETER_COUNT_AS_AO / SEED42_FORWARD_NEXT`。

### 2026-09-03 — MR-13BA result closure + MR-13BB pure capacity scaling implementation

- **Authoritative baseline**：`test-branch-1_20260903_013343_17acff86.zip`，SHA256=`61283d47fb85541e56f7359cd1b95e570c535a7f4808cd2bca5ffdfe8be2ce7e`；formal Forward comparison結果由使用者貼上之2026-09-03輸出提供。
- **MR-13BA Forward result**：完整Safety Validation Daily/Global rho/Pair=`0.3414/0.3031/62.29%`；Forward=`0.3529/0.2975/62.67%` vs AO=`0.3532/0.2929/62.68%`；Breakout=`0.3164/0.3572/63.54%` vs AO=`0.3135/0.3554/63.65%`。Forward Pred-HS true-LS=`36.84%`與AO完全相同，True-HS recall=`63.00%`，P45–P55 Pair=`52.39%`。HS-only Conditional-MFE Forward rho/Pair=`0.3875/63.54%`低於AO=`0.3969/63.90%`。Final Forward Daily rho/Pair/Top-Bottom=`0.3974/63.70%/1.7231R`亦低於AO=`0.4042/63.96%/1.8767R`。
- **MR-13BA Decision**：`FULL_WINDOW_RF_NO_SAFETY_BREAKTHROUGH / RF_INSUFFICIENCY_NOT_SUPPORTED / MODEL_GATE_FAIL / NO_ROLLING / NO_ROBUSTNESS / STOP_DILATION_KERNEL_RF_SWEEP`。RF由229擴到305且trainable params exact不變，Safety ranking與qualification purity未離開AO平台，因此不再把300-bar input未完整覆蓋視為主要瓶頸。
- **MR-13BB identity**：profile=`daily_universal_shared_safety_hs_conditional_mfe_wide_full_list_ndcg_pairwise`；architecture=`inception_time_shared_safety_mfe_wide_v1`。MR-13AO target/loss/true-HS scope、raw 300×10 input、shared InceptionTime topology、depth=6、kernels=(39,19,9)、module dilations全1、RF=229、GAP/heads、Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。
- **唯一scientific change**：channel/parameter capacity，filters=`32→64`、bottleneck=`32→64`；feature_count=10 trainable params=`473,734→1,885,446`（約`3.98×`）。不改RF、不改input bars、不改backbone family，也不做32/48/64等width sweep。
- **Primary Gate**：完整Safety Daily/Global rho/Pair需實質離開AO且Breakout同方向，Pred-HS true-LS需明顯下降；HS-only Conditional-MFE不得崩。若FAIL，停止單純InceptionTime parameter-capacity不足假說，下一步才允許backbone-family controlled contrast。

Decision：`MR13BA_CLOSED_MODEL_GATE_FAIL / MR13BB_IMPLEMENTED_RESULT_PENDING / PURE_CAPACITY_CONTROL / 3P98X_PARAMS / SEED42_FORWARD_NEXT`。

### 2026-09-03 — MR-13BB result closure + MR-13BC 600-bar long-horizon implementation

- **Authoritative baseline**：`test-branch-1_20260903_014811_f0c8c991.zip`，SHA256=`94201a0b8d194be3a4cbc2a5902782ad66015c5a9d726613f82e7e58ee99d793`；formal Forward comparison結果由使用者貼上之2026-09-03輸出提供。
- **MR-13BB Forward result**：完整Safety Validation Daily/Global rho/Pair=`0.3250/0.2648/61.65%`；Forward=`0.3325/0.2423/61.92%` vs AO=`0.3532/0.2929/62.68%`；Breakout=`0.2942/0.2795/63.17%` vs AO=`0.3135/0.3554/63.65%`。Forward Pred-HS true-LS=`37.52%` vs AO `36.84%`，True-HS recall=`62.33%`，P45–P55 Pair=`52.21%`。HS-only Conditional-MFE Forward rho/Pair=`0.3670/62.84%`低於AO=`0.3969/63.90%`。
- **MR-13BB Decision**：`PURE_CAPACITY_AT_300BARS_NOT_SUPPORTED / MODEL_GATE_FAIL / NO_ROLLING / NO_ROBUSTNESS / STOP_WIDTH_CAPACITY_SWEEP`。filters/bottleneck 32/32→64/64、trainable params約3.98×未改善Safety，且qualification purity更差；但此結果只回答capacity@300 bars，不足以否定「較長input提供新資訊後capacity才有作用」的interaction。
- **Research transition**：使用者明確提出capacity提升可能需與更長input共同存在才看得出效果，因此原先「BB FAIL後直接backbone family」順序被新的研究決策覆寫：先做600/32 long-horizon control；只有600/32顯著改善才允許600/64 capacity interaction，若600/32無增量才轉backbone family。這是人類基於迭代OOS evidence形成的新假說，不把OOS統計輸入fitting/selection pipeline。
- **MR-13BC identity**：profile=`daily_universal_shared_safety_hs_conditional_mfe_600bar_full_list_ndcg_pairwise`；architecture=`inception_time_shared_safety_mfe_600bar_v1`。AO continuous Safety full-list ΔNDCG、true-HS=P50 Conditional-MFE、1:1 loss、filters/bottleneck=32/32、depth=6、kernels=(39,19,9)、GAP/heads、Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。
- **Scientific treatment**：可利用OHLCV history由raw 300×10擴為raw 600×10。architecture descriptor新增generic `input_window_bars=600`，daily-universal lazy feature provider依model spec materialize 600-bar window；canonical dataset/label policy仍維持300-bar source/target SSOT，不改既有模型artifact identity。為使600-bar input確實可由高階representation共同利用，module dilations設為`(1,1,1,1,6,6)`，RF=`609`；此為long-horizon input的必要工程條件，params仍exact=`473,734`，不獨立解讀為RF treatment（BA已在300-bar下單獨否定RF不足）。
- **Fair-comparison rule**：600-bar eligibility會排除歷史不足600 bars的stock-days，因此Primary Gate不得只比較各模型不同universe的raw aggregate；最終decision需優先使用MR-13BC與AO/BA **common eligible stock-days**。既有Standard SOP persistent schema不因本實驗擅自變更；common-universe contrast可由完成後的score artifact作一次性read-only diagnostic。
- **Primary Gate**：common-universe Safety Daily/Global rho與Pair需實質離開AO/BA且Breakout同方向，Pred-HS true-LS需下降、P45–P55需改善，HS-only Conditional-MFE不得崩。若FAIL，停止input-horizon不足假說並轉backbone-family controlled contrast；若GO，下一輪才測600/64 capacity interaction。

Decision：`MR13BB_CLOSED_MODEL_GATE_FAIL / MR13BC_IMPLEMENTED_RESULT_PENDING / 600BAR_LONG_HORIZON_CONTROL / SAME_473734_PARAMS / RF609 / SEED42_FORWARD_NEXT`。

### 2026-09-03 — MR-13BC result closure + MR-13BD deep hierarchy matched-control implementation

- **Authoritative baseline**：`test-branch-1_20260903_193917_cd7e7236.zip`，SHA256=`d2dba5af3a7785bd6a304db0541250a21b9952b6636705c7bd0fe26af2370ce6`；Forward與Rolling comparison結果由使用者貼上之2026-09-03輸出提供。
- **MR-13BC Forward result**：Groups=`585,573`（AO=`608,204`）；Safety Daily/Global rho/Pair=`0.3436/0.2585/62.26%` vs AO=`0.3532/0.2929/62.68%`；Pred-HS true-LS=`37.12%` vs `36.84%`，True-HS recall=`62.73%`，P45–P55 Pair=`52.21%`；HS-only rho/Pair=`0.3490/62.04%` vs AO=`0.3969/63.90%`。Final Daily rho/Pair/Top-Bottom=`0.3639/62.42%/1.4762R` vs AO=`0.4042/63.96%/1.8767R`。
- **MR-13BC Rolling result**：Safety Daily/Global rho/Pair=`0.3506/0.2648/62.55%` vs AO=`0.3601/0.3141/62.91%`；Pred-HS true-LS=`36.94%` vs `36.67%`；HS-only rho/Pair=`0.3715/62.94%` vs `0.4113/64.43%`。Rolling final Daily rho/Pair/Top-Bottom=`0.3817/63.12%/1.5663R` vs AO=`0.4136/64.33%/1.8203R`；fold drift=`YES`、max adjacent score-mean drift=`1.5439 pooled σ`，AO=`NO / 0.8551σ`。
- **MR-13BC interpretation boundary**：600-bar eligibility排除約3.7% Forward rows，現有貼上報表未另列AO/BC common-eligible exact contrast，因此不宣稱精確的純horizon effect magnitude；但事前GO要求「明顯Safety突破」，而Forward與Rolling所有Safety/qualification主指標均無此證據，且Conditional-MFE與final ranking一致較弱。故依stop rule不做600/64 interaction。Decision=`LONG_HORIZON_NO_VISIBLE_SAFETY_GAIN / INPUT_HORIZON_INSUFFICIENCY_NOT_SUPPORTED / MODEL_GATE_FAIL / CLOSED / NO_600x64`。
- **Research transition**：使用者先前提出的物理限制假說包含depth/backbone/parameter/input length。BA/BB/BC已分別回答RF、capacity@300與long-horizon input；current AO objective尚未受控隔離的是depth/hierarchical composition。歷史ModernTCN、Patch與Residual-TCN已有負evidence，因此在backbone-family shopping前先做一個parameter/RF-matched depth control。
- **MR-13BD identity**：profile=`daily_universal_shared_safety_hs_conditional_mfe_deep_full_list_ndcg_pairwise`；architecture=`inception_time_shared_safety_mfe_deep_v1`。AO target/loss/true-HS scope、raw 300×10、shared InceptionTime family、filters=32、GAP/heads、Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。
- **Scientific treatment**：hierarchical depth=`6→12`。為避免把純參數量/RF再次帶入，descriptor以matching controls固定`bottleneck=24`、kernels=`(21,11,5)`、residual_every=3、all dilation=1；feature_count=10下trainable params=`475,702` vs AO `473,734`（`+0.42%`），RF=`241` vs `229`（`+5.2%`），final latent width仍128 channels。這些matching controls不是待調knobs，不做depth/kernel/bottleneck sweep。
- **Primary Gate**：完整Safety Daily/Global rho與Pair需實質離開AO、Pred-HS true-LS下降、P45–P55改善且Breakout同方向；HS-only Conditional-MFE不得崩。若FAIL，關閉InceptionTime RF/width/input/depth物理限制族，下一步只能是具有新inductive bias的backbone family，而非再微調InceptionTime。

Decision：`MR13BC_CLOSED_MODEL_GATE_FAIL / MR13BD_IMPLEMENTED_RESULT_PENDING / DEEP_HIERARCHY_MATCHED_CONTROL / PARAMS_PLUS_0P42_PERCENT / RF_PLUS_5P2_PERCENT / SEED42_FORWARD_NEXT`。

### 2026-09-03 — MR-13BD result closure + MR-13BE 600-bar × wide-capacity interaction implementation

- **Authoritative baseline**：`test-branch-1_20260903_200808_dbbcc559.zip`，SHA256=`59fde5175dc0e0983016a31d151b71f20fe5743679f17e5fa06ca84e697bf310`；formal Forward comparison結果由使用者貼上之2026-09-03輸出提供。
- **MR-13BD Forward result**：Safety Validation Daily/Global rho/Pair=`0.3303/0.2890/61.85%`；Forward=`0.3391/0.2848/62.18%` vs AO=`0.3532/0.2929/62.68%`；Breakout=`0.2784/0.3408/62.99%` vs AO=`0.3135/0.3554/63.65%`。Forward Pred-HS true-LS=`37.19%` vs AO `36.84%`，True-HS recall=`62.65%`，P45–P55 Pair=`52.20%`。HS-only Conditional-MFE Forward rho/Pair=`0.3434/61.83%`低於AO=`0.3969/63.90%`；final Daily rho/Pair=`0.3577/62.19%`亦低於AO=`0.4042/63.96%`。
- **MR-13BD fit/generalization evidence**：epoch1→2 Train Loss=`0.554436→0.472219`明顯下降，但Val HS-MFE rho=`0.3662→0.2551`、Val Safety rho=`0.3185→0.3102`同步惡化。較深hierarchy具有更強training fit能力但未形成可泛化Safety signal。Decision=`DEEP_HIERARCHY_NO_SAFETY_BREAKTHROUGH / MODEL_GATE_FAIL / CLOSED / NO_ROLLING / NO_ROBUSTNESS`。
- **Research decision override**：原BC stop rule為600/32無突破則不做600/64；使用者2026-09-03明確指出BB只回答capacity@300、BC只回答long-input@32，兩者都不能排除`long input × large capacity` interaction，並授權補齊factorial最後一格。此人類research decision取代舊`NO_600x64`下一步，不改寫BC原始結果。
- **MR-13BE identity**：profile=`daily_universal_shared_safety_hs_conditional_mfe_600bar_wide_full_list_ndcg_pairwise`；architecture=`inception_time_shared_safety_mfe_600bar_wide_v1`。AO continuous Safety + true-HS Conditional-MFE objective、1:1 loss、shared InceptionTime depth=6、kernels=(39,19,9)、GAP/heads、Seed42/split/optimizer/epoch selection/inference全部固定。
- **Factorial cell**：BE合併兩個已完成main-effect treatment且不增加第三變因：BC的`input_window_bars=600`與`module_dilations=(1,1,1,1,6,6)`（RF=609），加BB的`filters/bottleneck=64/64`；trainable params=`1,885,446`。因此BE vs BC在同600-bar eligible universe、同RF下只差capacity；BB vs AO則是既有300-bar capacity main effect。
- **Fair-comparison contract**：BE `model_gate_reference_profile_name`固定指向MR-13BC，而非AO。既有HS-Qualification Model-specific Extension會直接讀BC frozen Forward score artifact，於完全相同600-bar eligibility下輸出reference control；不修改Standard SOP persistent schema、不把BB/BC硬編成永久comparison list。
- **Stopping rule**：Seed42 Forward first。只有BE相對BC在Safety Daily/Global rho、Pair、Pred-HS true-LS與boundary上形成明顯且Breakout同方向的提升，且HS-only Conditional-MFE不崩，才支持capacity×long-horizon interaction；否則關閉InceptionTime RF/width/input/depth/interaction physical scaling族，下一步轉真正不同backbone family，不做600-bar width/horizon sweep。

Decision：`MR13BD_CLOSED_MODEL_GATE_FAIL / MR13BE_IMPLEMENTED_RESULT_PENDING / 600BAR_WIDE_CAPACITY_FACTORIAL_INTERACTION / BC_SAME_UNIVERSE_REFERENCE / SEED42_FORWARD_NEXT`。

### 2026-09-03 — MR-13BE result closure + MR-13BF full-resolution day-token Transformer implementation

- **Authoritative baseline**：`test-branch-1_20260903_210132_e05bb4d6.zip`，SHA256=`b5b3a8fda5bb9493e16ae2792863c007e44121b8ae15c9c6d8c59d4a89775c96`；MR-13BE正式Forward結果由使用者2026-09-03貼上輸出提供。
- **MR-13BE same-universe factorial result**：BE vs BC皆為600-bar eligibility/RF609，僅capacity 64/64 vs 32/32。Forward Safety Daily rho/Pair=`0.3453/62.37%` vs BC=`0.3436/62.26%`，Pred-HS true-LS=`37.10%` vs `37.12%`，P45–P55 Pair=`52.12%` vs `52.21%`，沒有實質qualification提升；Breakout Safety Daily rho=`0.3059`低於BC=`0.3108`。但HS-only Conditional-MFE Forward rho/Pair=`0.3818/63.32%` vs BC=`0.3490/62.04%`，Breakout rho/Pair=`0.4057/66.31%` vs `0.3658/63.25%`，final Forward Daily rho亦`0.3919` vs BC `0.3639`。
- **Interpretation**：`large capacity × long input` interaction對upside/Conditional-MFE representation確實有正效果，但Safety Daily/Pair/P50 purity幾乎不動。因此BB與BC各自失敗並不是因二者必須一起放大才會改善Safety；InceptionTime的RF、width、input horizon、depth及input×capacity五個physical-scaling假說均未突破Safety ceiling。Decision=`LONG_INPUT_X_CAPACITY_MFE_INTERACTION_PRESENT / SAFETY_CEILING_UNCHANGED / MODEL_GATE_FAIL / CLOSED / NO_ROLLING / NO_ROBUSTNESS`。
- **MR-13BF identity**：profile=`daily_universal_day_token_transformer_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise`；architecture=`day_token_transformer_shared_safety_mfe_v1`。AO target、continuous Safety supervision、true-HS=P50 Conditional-MFE sublist、1:1 dual-head loss、independent linear heads、raw 300×10 input、Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。
- **Only primary treatment**：shared InceptionTime改為full-resolution temporal Transformer；每個交易日直接成為1 token（patch size=stride=1），因此沒有歷史9F/AZ的10-bar patch temporal compression。300 tokens由第一層開始可全域self-attention；sinusoidal positional encoding、mean pool。固定`d_model=96 / depth=6 / heads=4 / FFN=212 / dropout=0`。
- **Capacity matching**：feature_count=10、雙linear heads下BF trainable params=`473,692`，AO=`473,734`，差`-42 / -0.009%`。FFN=212只作parameter-matching control，不是待調hyperparameter。
- **Gate / stop**：Seed42 Forward only。Primary要求BF相對AO在Safety Daily/Global rho、Pair、Pred-HS true-LS與P45–P55形成明顯改善，Breakout同方向且HS-only Conditional-MFE不得崩。若FAIL，不做token size/head/d_model/depth/FFN sweep，停止raw global-attention Transformer family；不得以iterative OOS再調Transformer recipe。
- **Engineering display closure**：使用者BE正式輸出顯示`input_window=300`，但600-bar loader/eligibility與`target_manifest.input_window_bars=600`為真；原因是console/Markdown renderer誤從不持有該欄位的`bundle.summary/source_dataset` fallback至canonical 300。renderer改為讀既有canonical target manifest；不改data membership、window materialization、fitting identity、checkpoint或report schema。

Decision：`MR13BE_CLOSED_MODEL_GATE_FAIL / INCEPTIONTIME_PHYSICAL_SCALING_FAMILY_STOP / MR13BF_IMPLEMENTED_RESULT_PENDING / PARAMETER_MATCHED_DAY_TOKEN_GLOBAL_ATTENTION / SEED42_FORWARD_NEXT`。

### 2026-09-04 — MR-13BF Day-Token Global Transformer Model Gate FAIL；使用者授權MR-13BG GRU backbone control

- **MR-13BF Forward**：Safety Daily/Global rho/Pair=`0.3432/0.2988/62.22%`，低於或未穩定優於AO=`0.3532/0.2929/62.68%`；Pred-HS true-LS=`37.30%` vs AO `36.84%`，P45–P55 Pair=`52.35%`，未脫離boundary ceiling。HS-only Conditional-MFE rho/Pair=`0.4007/64.10%`，與AO=`0.3969/63.90%`相近，表示global attention可保留upside ranking但未改善Safety qualification。
- **MR-13BF Rolling**：Safety Daily/Global rho/Pair=`0.3489/0.2055/62.44%` vs AO=`0.3601/0.3141/62.91%`；Pred-HS true-LS=`37.01%` vs `36.67%`，P45–P55=`52.33% vs 52.36%`；HS-only=`0.4112/64.46%` vs AO=`0.4113/64.43%`。Rolling drift仍NO，但沒有Safety增量。
- **Breakout qualification**：BF Forward true-LS=`33.06%`、Rolling=`33.56%`看似較低，但true-HS recall只有`52.62%/54.57%`，且TopK HM/HS沒有同步突破，因此屬precision/recall trade-off，不視為通用Safety improvement。
- **Decision**：`RAW_GLOBAL_ATTENTION_DOES_NOT_BREAK_SAFETY_CEILING / CONDITIONAL_MFE_PRESERVED / MODEL_GATE_FAIL / NO_TRANSFORMER_SWEEP / CLOSED / NOT_PROMOTED`。不得再以token size、attention heads、d_model、depth、FFN做迭代OOS sweep。
- **MR-13BG user-authorized override**：BF事前stop rule原應轉information/target uncertainty；使用者明確要求先試GRU，因此另立`MR-13BG / ARCH-gru_shared_safety_mfe_v1`，不改寫BF結論。BG固定AO raw `300×10`、continuous Safety + true-HS Conditional-MFE、1:1 loss、independent heads、Seed42/split/optimizer/epoch-selection/inference，只換shared backbone為3-layer unidirectional GRU hidden=`176`、dropout=`0`、final-state pooling。feature_count=10 trainable params=`473,796` vs AO=`473,734`（`+0.013%`）。先做Seed42 Forward Model Gate；若FAIL，不做GRU hidden/layer/bidirectional sweep，回information/target uncertainty。

### 2026-09-04 — MR-13BH result closure + MR-13BI pure-BF16 dynamic backward-scaling control

- **BH completed Forward evidence**：MR-13BH epoch-selection完全重現先前BF16 trajectory，best epoch5 Validation HS-MFE/Safety/Pred-HS TopK HM/HS=`0.4191/0.3574/26.40%`。Selection完整重訓前3 epoch Batch Loss=`0.665070→0.553312→0.540414`；第4 epoch首次偵測`non-finite BF16 gradient`並依BH contract做same-batch FP32 recompute。該retry後epoch4/5 Loss反而升至`0.579697/0.615215`，顯示rescue step沒有維持原BF16 optimizer trajectory。
- **BH final result**：Forward overall Daily/Global/Pair=`0.3873/0.3786/63.42%`；Safety Daily/Global/Pair=`0.3272/0.2867/61.58%`，Pred-HS true-LS=`38.04%`，HS-only rho/Pair=`0.3937/63.79%`。Breakout Safety=`0.3298/0.3738/63.48%`，purity改善伴隨true-HS recall僅`38.21%`。相對BG Safety=`0.3683/0.3231/63.26%`與true-LS=`36.30%`明顯退化，因此BH不能promotion。
- **Interpretation**：BH證明「阻止non-finite optimizer step」本身不足；在BF16 trajectory中插入完整FP32 forward/backward gradient會改變gradient direction與Adam moments，雖然optimizer update count仍為1，scientific optimization path仍受到material disturbance。Decision=`SAME_BATCH_FP32_RETRY_DISTORTS_REFIT_TRAJECTORY / MODEL_CONTROL_FAIL / CLOSED / NO_ROLLING`。
- **MR-13BI design**：使用者授權下一個numerical control，不新增GRU topology。BI直接重用`ARCH-gru_shared_safety_mfe_v3`（1-layer/hidden391、474,287 params、outer BF16）與BH全部scientific contract；新增profile-owned `numerical_execution_policy=bf16_dynamic_backward_scaling`，retry scales固定`(1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256)`。正常batch完全不變；只有原始backward gradient non-finite時才清grad、同batch同BF16 autocast重算loss並逐級縮放backward。一旦scaled gradient finite，在FP32 parameter-gradient buffer除回scale，重新確認finite、套原`clip_norm`，再執行唯一一次Adam step。
- **Fail-fast boundary**：若BF16 forward或canonical loss本身non-finite，BI不切FP32而直接失敗；若全部8個backward scales耗盡仍non-finite亦fail-fast。禁止skip batch、降低LR、改clip、固定epoch5、改hidden/layer/bidirectional或任何OOS-guided tuning。
- **Identity/ownership**：numerical execution policy由Experiment Profile持有並進`experiment_settings` fitted identity；BI與BH可共用同一architecture descriptor，避免再把training recipe誤建成architecture identity。既有68個profile manifest payload與48個architecture manifests保持不變；只新增BI profile，無新architecture。Current Training Model仍BG；BI加入shared `[3]～[6]` comparison membership作Forward-only numerical control。
- **Independent checks**：synthetic強制原始gradient與`1/2` retry皆判non-finite、`1/4`才成功；確認未呼叫FP32 retry、exactly one optimizer step、rescue scale=`0.25`、parameters finite，且scaled/unscaled更新與正常BF16 reference保持在`5e-5` max-parameter tolerance內。

Decision：`MR13BH_CONTROL_FAIL / SAME_BATCH_FP32_RETRY_TRAJECTORY_DISTORTION_CONFIRMED / MR13BI_PURE_BF16_DYNAMIC_BACKWARD_SCALING_IMPLEMENTED / NO_FP32_FALLBACK / ONE_BATCH_ONE_OPTIMIZER_STEP / BG_ROLLING_PRIORITY_UNCHANGED`。

### 2026-09-04 — MR-13BI numerical-control closure + MR-13BG Rolling confirmation closure

- **MR-13BI completed Forward numerical control**：epoch-selection再次精確重現原pure-BF16 curve，best epoch5 Val HS-MFE/Safety/Pred-HS TopK HM/HS=`0.4191/0.3574/26.40%`。Selection refit前三epoch Loss=`0.665070→0.553312→0.540414`；第4 epoch第一次non-finite gradient需dynamic backward scaling至`scale=0.0625`（4 attempts）才finite，但epoch4/5 Loss仍反轉為`0.580198/0.606037`。Final Forward overall Daily/Global/Pair=`0.3947/0.3798/63.68%`；Safety=`0.3435/0.2982/62.24%`、Pred-HS true-LS=`37.27%`、HS-only rho/Pair=`0.3990/64.01%`。因此即使不切FP32，late-epoch BF16 refit仍位於numerical-instability boundary附近，未形成對BG的可靠增量。
- **BI decision**：`BACKWARD_SCALING_DOES_NOT_PRESERVE_REFIT_TRAJECTORY / MODEL_CONTROL_FAIL / CLOSED / HISTORICAL_ONLY / NO_ROLLING / NO_ROBUSTNESS / NOT_PROMOTED`。BH與BI均只保留歷史artifact/evidence，不再進current `[3]～[6]` membership。
- **MR-13BG Rolling evidence**：overall Rolling Daily/Global rho/Pair=`0.4146/0.3686/64.35%` vs AO=`0.4136/0.3365/64.33%`，Top-Bottom=`1.9043R vs 1.8203R`；表面overall ranking略高。Safety Rolling Daily/Global rho/Pair=`0.3639/0.2536/63.06%` vs AO=`0.3601/0.3141/62.91%`，Pred-HS true-LS=`36.44% vs 36.67%`，HS-only rho/Pair=`0.4130/64.51% vs 0.4113/64.43%`。Daily/Pair/purity只有小幅正向，但Safety Global rho明顯退化。
- **Top-tail / stability evidence**：BG Pred-HS TopK HM/HS=`24.76%`、Pred-HS High-MFE=`50.81%`、Pred-HS mean MFE=`1.267R`，均低於AO `27.21%/56.99%/1.468R`；Rolling drift=`1.0176 pooled σ / YES`，AO=`0.8551 / NO`。Breakout Safety Daily/Global/Pair=`0.3437/0.3690/64.06%` vs AO=`0.3370/0.3782/64.03%`，true-LS=`34.14% vs 35.96%`但recall=`57.53% vs 60.85%`，P45–P55=`51.46% vs 53.56%`，仍是purity/recall與boundary trade-off。
- **BG decision against predeclared gate**：事前要求Rolling同時確認Safety Daily/Global rho/Pair與true-LS優勢、HS-only MFE不崩、drift不惡化。BG在HS-only與Daily/Pair/purity上沒有崩，但**Safety Global與drift兩項明確不通過**，且Pred-HS top-tail economic quality較AO弱。因此Decision=`FORWARD_SIGNAL_NOT_ROLLING_CONFIRMED / ROLLING_MIXED / TUNING_CLOSED / NO_ROBUSTNESS / NO_GRU_SWEEP / NO_STRATEGY / NOT_PROMOTED`。不以single-seed小幅overall gain啟動robustness。
- **Current workflow boundary**：使用者先前明確要求Training Model維持`MR-13BG`並從compare list移除BH/BI；故current pair仍是BG，scientific reference base仍AO。這只是current workflow holder，不代表BG promotion。下一個模型研究回到BF前已延後的`information content / target uncertainty`問題；具體單一treatment尚未授權，不預占新MR identity。

### 2026-09-04 — MR-13BJ Forward result closure + MR-13BK MR-13M-target BiGRU backbone isolation

- **Authoritative implementation baseline**：`test-branch-1_20260904_193435_243f078c.zip`，SHA256=`1d3527f7d614dee4eb74a88910ad07f9a5e0885c11db9eda6f342e30e8ada649`。
- **MR-13BJ completed Forward evidence**：selected epoch=`3`；Validation HS-MFE/Safety rho=`0.4312/0.3581`。Forward overall Daily/Global rho/Pair=`0.4189/0.3997/64.51%` vs BG=`0.4124/0.3815/64.25%`，Top-Bottom=`2.2941R vs 2.1017R`。Forward Safety Daily/Global rho/Pair=`0.3697/0.3074/63.33%` vs BG=`0.3683/0.3231/63.26%`；Pred-HS true-LS=`36.14%`、true-HS recall=`63.70%`。HS-only Conditional-MFE rho/Pair=`0.4186/64.71%`、Pred-HS TopK HM/HS=`26.24%`、High-MFE=`55.25%`、mean MFE=`1.376R`，均較BG的`0.4095/64.37%`、`24.43%`、`50.93%`、`1.258R`改善。
- **MR-13BJ Breakout evidence**：overall Daily/Global/Pair=`0.3916/0.4365/65.17%`；Safety=`0.3458/0.3834/64.47%`。Pred-HS true-LS=`32.63%`明顯低於BG `34.95%`，但true-HS recall同步降至`56.12%`（BG `62.03%`）；HS-only rho=`0.4317`低於BG `0.4491`，Pred-HS HM/HS=`23.31%`亦未改善。
- **MR-13BJ decision**：bidirectionality確實改善overall ranking、Global representation與Forward MFE/top-tail recovery，但沒有通過事前要求的material joint Safety gate；尤其Safety Global相對BG退化，Breakout purity帶有recall trade-off。Decision=`BIDIRECTIONALITY_IMPROVES_OVERALL_AND_MFE_REPRESENTATION / SAFETY_GLOBAL_NOT_IMPROVED / BREAKOUT_PURITY_RECALL_TRADEOFF / FORWARD_MIXED / ROLLING_DEFERRED / NOT_PROMOTED`。不因overall漂亮而啟動hidden/layer/LR/precision sweep。
- **User research decision**：使用者明確要求「用這個來跑看看單純的13M」，因此建立新的target-fixed backbone isolation；不是繼續調BJ dual-head objective，也不是改寫MR-13M歷史結果。
- **MR-13BK identity**：profile=`daily_universal_bigru_full_horizon_low_adverse_full_list_ndcg_pairwise`；architecture=`ARCH-bigru_ranker_v1`。MR-13M `daily_full_horizon_low_adverse_r_v1`、40D full-horizon earliest-max-MFE peak、negative adverse-to-peak R ordering、daily-universal stock-day universe、full-list Delta-NDCG weighted RankNet、single rank head、Seed42、split、Adam、gradient clip、date-coherent batch與mean-Daily-Spearman epoch selection全部固定。
- **Only scientific treatment**：MR-13M `ARCH-inception_time_v1` global-average representation → MR-13BJ同款1-layer bidirectional GRU hidden=`274`/direction，以forward/backward final recurrent states concat成548-d latent。兩方向只讀同一300個decision-time bars，PIT合法。BK不帶Safety/MFE auxiliary heads或multi-task loss；single rank head直接學Low-Adverse target。
- **Capacity control**：feature_count=10，MR-13BK trainable params=`471,282` vs MR-13M InceptionTime=`473,218`（`-0.409%`）。不加入projection或unused auxiliary head湊參數。
- **Architecture ownership**：新增descriptor-driven `ARCH-bigru_ranker_v1`，重用既有generic split-direction CUDA-parallel GRU execution primitive；canonical architecture capability宣告`single_rank_head`，spec/runtime owner依capability建單一rank head。既有GRU v1-v4 construction/state/output semantics不變。
- **Workflow membership**：Current Training Model切到MR-13BK；shared compare/test list=`MR-13H / MR-13M / MR-13AH / MR-13AK / MR-13AO / MR-13BK`。MR-13M是BK primary target-matched reference；BG/BJ結果保留但不作本輪active comparison必要member，BH/BI仍historical-only。
- **Predeclared Gate**：Seed42 Forward only first。BK vs M必須在all-stock Daily/Global rho與Pair形成material共同改善，且Breakout同方向；單一Global、Top-Bottom或slice改善不足以進Rolling。若未明顯拉開，不做BK Rolling/robustness，也不做BiGRU hidden/layer/head sweep。

Decision：`MR13BJ_FORWARD_MIXED_RESULT_AVAILABLE / MR13BK_IMPLEMENTED_RESULT_PENDING / EXACT_MR13M_TARGET_LOSS_CONTROL / SINGLE_HEAD_BIGRU / FORWARD_GATE_FIRST / NO_ROLLING_UNLESS_MATERIAL_BACKBONE_GAIN`。

### 2026-09-04 — MR-13BK closure + MR-13BL Price-Volume Structural Safety Representation

- **Authoritative implementation baseline**：`test-branch-1_20260904_211341_1ac14739(2).zip`，SHA256=`744a89cce50e6e2ea3b21d5d14a39431b30c33f42254b5d9a5e5fd6a1a12f5c8`。
- **MR-13BK completed Forward evidence**：MR-13M InceptionTime vs BK BiGRU on identical Low-Adverse target：all-stock Daily/Global rho/Pair=`0.3687/0.3287/63.21%` vs `0.3425/0.2201/62.22%`；Breakout=`0.3396/0.3938/64.27%` vs `0.3386/0.2870/63.72%`。Top10 High-Safety=`79.32%` vs `76.44%`，Top10 Low-Adverse=`-0.1406R` vs `-0.1587R`。Decision=`BACKBONE_ISOLATION_FAIL / BIGRU_NOT_PURE_SAFETY_LEARNABILITY_SOLUTION / CLOSED / NO_ROLLING / NO_ROBUSTNESS / NO_GRU_TOPOLOGY_SWEEP`。
- **User research decision**：使用者明確表示先不要再做target engineering，研究目標仍是提高Safety learnability；提出技術分析中的支撐區、壓力區、趨勢線以及「該價位區的成交量」這類price×volume結構。使用者確認計算量設計後回覆「進行」，因此建立新的representation family，而不是重開backbone/target/horizon sweep。
- **MR-13BL identity**：profile=`daily_universal_price_volume_structure_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise`；architecture=`ARCH-inception_time_shared_safety_mfe_price_volume_structure_v1`。Primary reference=`MR-13AO`。
- **Scientific contract frozen from AO**：same-date full-universe Raw Safety target、true-HS P50 Conditional-MFE target、300×10 canonical normalized sequence、training universe、Seed42、split、Adam、gradient clip、date-coherent batch、dual-head 1:1 full-list Delta-NDCG pairwise objective、epoch selection、inference與MFE head topology全部固定。
- **Only scientific treatment**：同一300-bar stock OHLCV前5個normalized channels在batch-time deterministic rasterize成Price-Time-Volume structure。Price axis以current 14-bar true-range ATR正規化，固定`±16 ATR / 64 bins`；time axis完整使用300 bars後pool到`128 bins`。4 channels固定=`body occupancy / wick occupancy / relative-volume×body / relative-volume×wick`。Relative volume由既有window-normalized log-volume作parameter-free monotone softplus後mean=1 normalization，不使用absolute ticker volume scale。Volume-at-Price=`64 bins`由同一volume-weighted candle occupancy沿time累積並mean-normalize。
- **Representation network**：geometry=`Conv2d 4→12→24→32 + GAP = 32D`；VAP=`Conv1d 1→8→16 + GAP = 16D`；兩者concat後由bias-free `48→128` projection形成Safety residual，與AO shared latent相加後送入原Raw Safety head。Raw-MFE head仍只讀AO shared latent；structure branch不接MFE loss gradient。沒有人工support/resistance label、pivot threshold、trendline selection、RSI/MACD、Fib、portfolio state或post-decision資訊。
- **Storage / compute contract**：不建立expanded daily geometry artifact；map只在batch forward由既有lazy canonical OHLCV sequence生成。Per-sample persistent storage增量=`0`。Formal map shape=`4×128×64`；若以BF16 tensor估算約64 KiB/sample，batch128約8 MiB正式map；FP32 raster intermediates為transient。Trainable params=`490,586` vs AO=`473,734`，增量=`+16,852 / +3.56%`。
- **Controlled identity evidence**：same seed下AO所有common state tensors與BL完全exact；AO vs BL MFE logits bitwise exact；Safety logits因structure residual而不同。MFE-only loss對structure parameters gradient=`0`，Safety-only loss對structure parameters gradient=`>0`。Synthetic相同價格幾何下，抬高單一bar相對成交量會提高其對應local volume-weighted geometry mass，確認volume不是獨立旁路而是實際乘進price-zone representation。
- **Predeclared Gate**：Seed42 Forward only first。Primary看Raw Safety而非overall score；要求Safety Daily/Global rho/Pair形成material共同改善、Breakout同方向、Pred-HS true-LS下降或至少不惡化，HS-only Conditional-MFE與Pred-HS top-tail不得明顯受損。若只改善overall或單一slice，不進Rolling。若FAIL，不做time-bin/price-bin/ATR-span/channel-count或CNN width sweep。

Decision：`MR13BK_CLOSED_BACKBONE_FAIL / MR13BL_IMPLEMENTED_RESULT_PENDING / AO_EXACT_TARGET_LOSS_BACKBONE_CONTROL / PRICE_VOLUME_STRUCTURE_SAFETY_ONLY / MFE_PATH_EXACT / LAZY_BATCH_RASTER / FORWARD_GATE_FIRST / NO_GEOMETRY_SWEEP_IF_FAIL`。

### 2026-09-05 — MR-13BM Forward closure + MR-13BN Multi-scale Price-Volume 2D implementation

- **Baseline**：`test-branch-1_20260905_000945_99a09d64.zip`，SHA256=`4099b2349cae1a2488899575382d5fee8dadf051232f34f3d23d0bf861c399ef`。
- **MR-13BM Forward result**：Raw Safety Daily/Global/Pair=`0.3586/0.3077/62.87%` vs BL=`0.3568/0.3028/62.81%`；P45–P55=`52.44%` vs `52.39%`。Breakout Raw Safety=`0.3282/0.3696/63.66%` vs BL=`0.3283/0.3708/63.83%`，Breakout P45–P55=`55.21%` vs `54.78%`。Pred-HS HM/HS=`25.00%` vs `25.16%`、High-MFE=`54.21%` vs `55.79%`、MFE=`1.481R` vs `1.459R`。Decision=`LOCAL_ZONE_TOKEN_WEAK_INCREMENT / BREAKOUT_MIXED / MATERIAL_GATE_FAIL / NO_TOKEN_SWEEP / NOT_PROMOTED`。
- **Interpretation**：BM沒有否定global+local互補；它顯示把local資訊先壓成price-zone statistics只留下很小增量，time×price線型可能是被丟失的資訊。
- **MR-13BN identity**：profile=`daily_universal_price_volume_multiscale_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise`；architecture=`ARCH-inception_time_shared_safety_mfe_price_volume_multiscale_v1`。Primary reference=`MR-13BL`，BM保留local-zone control。
- **唯一scientific treatment**：BL global `128×64×4 / ±16 ATR / 300 bars`與64-bin VAP完全保留；新增recent80-bar、±4 ATR、`96×64×4` high-resolution local Price-Time-Volume 2D map。Local 2D CNN輸出32D；projected global residual與local latent做element-wise interaction，`[local, local×global]`經bias-free projection形成Safety-only residual。AO target/loss/split/Seed42/Adam/clip/date-coherent batch/epoch-selection及完整MFE path不變。
- **工程／PIT**：local map只由既有300×10前5個stock OHLCV channels在batch-time deterministic生成；不落盤expanded map，不新增資料源、pivot/support/resistance label或technical indicator。
- **Same-seed isolation**：BL common state exact、MFE logits bitwise exact；BN params=`512,878` vs BL=`490,586`（`+4.54%`）。
- **Gate**：Seed42 Forward first；BN需在BL上形成material Raw Safety Daily/Global/Pair共同改善且Breakout boundary不退、MFE top-tail不再惡化。若只有小增量，不做local window/raster/span/CNN/fusion sweep。

### 2026-09-05 — MR-13BN Forward closure + MR-13BO Position-Aware Multi-scale Price-Volume implementation

- **Authoritative implementation baseline**：`test-branch-1_20260905_002703_04fb31a8.zip`，SHA256=`a3ce82b4d43637052104757522dafba85fbf8602e22b9c795fa113d45f7de315`。
- **MR-13BN Forward result**：Raw Safety Daily/Global/Pair=`0.3538/0.3030/62.67%` vs BL=`0.3568/0.3028/62.81%`；Breakout Raw Safety=`0.3208/0.3725/63.74%` vs BL=`0.3283/0.3708/63.83%`。P45–P55=`52.32%` vs BL `52.39%`；Breakout P45–P55=`55.21%` vs BL `54.78%`。Pred-HS true-LS=`36.85%`，Breakout=`33.92%`；Forward Pred-HS MFE=`1.624R` vs BL `1.459R`，但Breakout MFE=`0.868R` vs BL `0.883R`。Decision=`MULTISCALE_GEOMETRY_FORWARD_GATE_FAIL / NO_ROLLING / NO_RESOLUTION_SPAN_CNN_SWEEP / NOT_PROMOTED`。
- **Representation diagnosis**：canonical OHLC price channels已以decision-day close正規化，因此current price anchor本身存在；問題在Global/Local Conv2d與VAP Conv1d最後皆以adaptive global average pooling壓縮，且原4-channel map沒有顯式signed price/time coordinate。相同局部形狀位於current price上方/下方或recent/old區域時，network需靠boundary effects間接推回位置，與支撐/壓力Safety語意不匹配。
- **MR-13BO identity**：profile=`daily_universal_price_volume_position_aware_multiscale_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise`；architecture=`ARCH-inception_time_shared_safety_mfe_price_volume_position_aware_multiscale_v1`。Primary reference=`MR-13BN`，BL保留global-only reference，AO保留scientific base。
- **Only scientific treatment**：BN Global/Local spans/resolutions及cross-scale fusion全部固定。Global/Local原4 evidence channels=`body / wick / relative-volume×body / relative-volume×wick`各追加2個deterministic CoordConv channels：signed price coordinate由branch最低價位`-1`到最高`+1`、current-price附近為0；time-age由oldest=`-1`到newest=`+1`。Global shape=`6×128×64`；Local=`6×96×64`。VAP由`1×64 mass`改為`2×64 [mass,signed-price]`。沒有pivot/support/resistance label、technical indicator、新資料源或target/loss改動。
- **Isolation / capacity**：同seed下BN與BO所有非Price-Volume common state tensors exact；MFE logits bitwise exact；MFE-only loss對position-aware encoder gradient=`0`、Safety-only loss=`>0`。BO params=`513,350` vs BN=`512,878`，只增加`472 / +0.09%`，因此不是capacity scaling。
- **Workflow membership**：Current Training Model=`MR-13BO`；shared compare/test list=`MR-13H / MR-13AH / MR-13AK / MR-13AO / MR-13BL / MR-13BM / MR-13BN / MR-13BO`，確保BO primary reference BN可在[3]～[6]直接比較。
- **Gate**：Seed42 Forward first。BO需相對BN讓Raw Safety Daily/Global/Pair形成material共同改善，Breakout Safety/boundary與Pred-HS true-LS不得退，且HS-only Conditional-MFE/top-tail不得以明顯trade-off換取。若仍只是小幅變動，不做coordinate scaling、map resolution/span、CNN width或fusion sweep，整條position-aware Price×Volume第一版結案。

Decision：`MR13BN_FORWARD_GATE_FAIL / MR13BO_IMPLEMENTED_RESULT_PENDING / EXPLICIT_PRICE_TIME_COORDINATES / POSITION_AWARE_VAP / MFE_PATH_EXACT / CAPACITY_NEUTRAL / FORWARD_GATE_FIRST`。

### 2026-09-05 — MR-13BO Rolling closure + MR-13BP Explicit Pairwise Temporal-Relation Safety implementation

- **Authoritative implementation baseline**：`test-branch-1_20260905_131306_038203a4.zip`，SHA256=`35b786759d2258c3dfe87d0722c4cae15ca4ed231a521516fdeb7ded3c12d9b4`。本輪以全新隔離解壓目錄開始，未overlay任何舊patch。
- **MR-13BO Rolling closure**：Raw Safety Daily/Global/Pair=`0.3577/0.2908/62.80%` vs AO=`0.3601/0.3141/62.91%`；Pred-HS true-LS=`36.66%`、True-HS recall=`63.18%`、drift=`1.1735σ YES`。Breakout Pred-HS true-LS雖到`32.45%`，但True-HS recall同時降至`51.93%`，HS-only rho=`0.4088`、Pred-HS MFE=`0.790R`，因此不是material Safety learnability。Decision=`MR13BO_ROLLING_GATE_FAIL / PRICE_VOLUME_STRUCTURE_FAMILY_CLOSED / NOT_PROMOTED`；BL只保留weak-positive structural-information control，不再做raster/token/coordinate/ATR-span/resolution/CNN-width sweep。
- **Equivalent-treatment audit**：歷史AY只做learned Q/K/V temporal content interaction；BF global day-token Transformer、BG/BJ GRU/BiGRU與歷史TS2Vec都不是explicit stock-bar relation representation。現有code/Registry沒有真正等價的bar-to-bar `price/volume/time` relation-bias treatment，因此分配新identity `MR-13BP`。
- **MR-13BP identity**：profile=`daily_universal_shared_safety_pairwise_relation_hs_conditional_mfe_full_list_ndcg_pairwise`；architecture=`ARCH-inception_time_shared_safety_pairwise_relation_mfe_v1`。Primary scientific reference=`MR-13AO`；`MR-13AY`為mechanistic self-attention-without-explicit-relation control。
- **唯一scientific treatment**：AO的300×10、full-universe Safety、true-HS-only Conditional-MFE、dual-head 1:1 full-list ΔNDCG、Seed42、Adam、split/PIT、epoch-selection與inference語意全部固定。Safety single-head temporal self-attention score新增5D signed relation：`Δlog Close / Δlog High / Δlog Low / Δcanonical normalized-log-volume / normalized signed time distance`；固定`5→8→1` MLP輸出scalar additive attention bias。Price relation直接由canonical `P/anchor_close-1` 的`log1p`差取得log ratio；沒有新normalization、pivot/extrema/support label、future-confirmed structure或persistent relation artifact。
- **Isolation / capacity**：relation capability只進Safety；Conditional-MFE維持AO shared feature map→GAP。Same-seed BP對AY所有common state tensors exact，僅新增relation MLP三個state tensors／`56` params；AO/AY/BP MFE logits bitwise exact。relation branch Safety-only gradient=`>0`、MFE-only gradient=`0`。
- **Execution-only memory semantics**：relation owner利用一階線性層對signed differences的代數等價性，避免materialize完整`T×T×8` hidden tensor；relation-biased attention只按batch維度做固定pair-cell cap chunking，relation/output/order完全不變，也不形成scientific knob。小型fixture與直接5D MLP計算allclose=`1e-6`。
- **Refactor boundary**：沒有在trainer/planner/renderer加入`MR-13BP` identity branch；新增scientific composition只落在declarative architecture/profile、單一pairwise-relation capability owner與既有capability-oriented synthetic validator。Current compare/test membership由B325 SSOT自動派生為`MR-13H / MR-13AH / MR-13AK / MR-13AO / MR-13AY / MR-13BP`，已關閉BL～BO退出current controls。
- **GPT targeted checks**：`validate_breakout_quality_reusable_model_component_contract_case=75/75 PASS`；current-membership contract=`6/6 PASS`；`validate_breakout_quality_policy_single_source_case=13/13 PASS`；registry/config current pair解析=`PASS`；Python compile/import=`PASS`。GPT未執行`apps/test_suite.py`或`apps/run_bundle.py`。
- **Gate / stop rule**：Seed42 Forward first。Raw Safety Daily/Global/Pair需material共同改善，purity改善不可靠True-HS recall collapse，Breakout/P45–P55與drift需同方向，HS-only Conditional-MFE與Pred-HS top-tail不可material惡化。只有小幅單seed增量時不做robustness，也不做relation feature/window/hidden/head/topology sweep。

Decision：`MR13BO_CLOSED / PRICE_VOLUME_STRUCTURE_FAMILY_CLOSED / MR13BP_IMPLEMENTED_RESULT_PENDING / AO_SCIENTIFIC_BASE / AY_MECHANISTIC_CONTROL / EXPLICIT_PAIRWISE_RELATION_SAFETY_ONLY / MFE_PATH_EXACT / FORWARD_GATE_FIRST`。

### 2026-09-05 — MR-13BQ Seed42 Forward Model Gate closure

- **User evidence**：使用者於`apps/research.py → Continuous DL → [3] Forward OOS模型比較`完成MR-13BQ缺失Forward工件BUILD與Standard Model SOP comparison；BQ selected epoch=`1`，Selection完整重訓=`1 epoch`。
- **Primary reference**：MR-13AO。BQ與AO的正式supervised architecture/target/loss/split/inference相同，唯一treatment為`stock_code_classification_v1` encoder pretraining。
- **Overall output**：BQ Forward Daily/Global/Pair=`0.3937/0.3383/63.61%` vs AO=`0.4042/0.3368/63.96%`；Breakout=`0.3712/0.4142/64.62%` vs AO=`0.3850/0.3804/64.73%`。Breakout Global rho與Top-Bottom存在局部正向，但非primary Safety learnability突破。
- **Raw Safety primary Gate**：BQ Forward Raw Safety Daily/Global/Pair=`0.3332/0.2761/61.95%`，全面低於AO=`0.3532/0.2929/62.68%`；Breakout=`0.2892/0.3260/63.09%`亦全面低於AO=`0.3135/0.3554/63.65%`。因此SCC initialization沒有改善Safety representation，且方向不是marginal positive。
- **Conditional-MFE**：BQ Forward HS-only Daily/Global/Pair=`0.3779/0.2993/63.24%` vs AO=`0.3969/0.3104/63.90%`；Breakout=`0.4046/0.4633/66.94%` vs AO=`0.4116/0.4353/67.22%`。Breakout Global rho局部提高，但Daily/Pair與Forward皆未形成共同增量。
- **Qualification / top-tail**：BQ Pred-HS true-LS Forward=`37.42%` vs AO=`36.84%`，Breakout=`39.53%` vs `36.94%`；Pred-HS HM/HS Forward=`25.47%` vs `25.94%`，Breakout=`23.96%` vs `24.07%`。BQ Pred-HS MFE雖Forward=`1.662R`、Breakout=`1.086R`高於AO=`1.585R/1.000R`，但伴隨更高LS contamination，不能視為Safety成功。
- **Stop-rule application**：事前已明定若BQ仍只有約1～3% marginal shift即不展開pretraining hyperparameter/task sweep；本次實際primary Safety更是共同退化，因此證據已足夠直接結案，不需要Rolling、Robustness或額外Audit。SCC LR/epoch/freeze/batch、MAP/SSC/其他pretext task均不啟動。
- **Current workflow**：MR-13BQ保留完整identity/result/implementation以供歷史重現，但退出current Training/Compare focus；canonical Training pair回MR-13AO作scientific reference/fallback，這不重開AO歷史Gate。MR-13BP維持`IMPLEMENTED / RESULT_PENDING / DEFERRED_BY_USER_PRIORITY_CHANGE`，不因BQ FAIL自動恢復。下一個尚未分配identity的研究問題為PIT-safe orthogonal information expansion，需先確認information family與PIT source contract後才實作。

Decision：`SCC_PRETRAINING_DOES_NOT_BREAK_SAFETY_CEILING / RAW_SAFETY_DEGRADES / CONDITIONAL_MFE_NOT_IMPROVED / MODEL_GATE_FAIL / CLOSED / NO_ROLLING / NO_ROBUSTNESS / NO_PRETEXT_SWEEP / AO_REFERENCE_FALLBACK / NEXT_ORTHOGONAL_INFORMATION_DESIGN`。

### 2026-09-05 — MR-13BQ post-closure Rolling evidence（Decision unchanged）

- **Context**：BQ已依Seed42 Forward事前stop rule正式CLOSED；使用者其後仍提供額外Rolling OOS comparison。此evidence只補充歷史可追溯性，不撤銷原research stop decision，也不重開SCC tuning。
- **Rolling overall**：BQ Daily/Global/Pair=`0.3942/0.3142/63.63%` vs AO=`0.4136/0.3365/64.33%`；Breakout=`0.3805/0.3902/64.71%` vs AO=`0.3821/0.4088/65.12%`。
- **Raw Safety / Conditional-MFE**：BQ Raw Safety=`0.3423/0.2925/62.31%` vs AO=`0.3601/0.3141/62.91%`；Breakout=`0.3104/0.3545/63.25%` vs AO=`0.3370/0.3782/64.03%`。HS-only Conditional-MFE=`0.3848/0.2861/63.49%` vs AO=`0.4113/0.3081/64.43%`；Pred-HS true-LS=`37.13%` vs AO=`36.67%`。
- **Stability**：BQ max adjacent score-mean drift=`1.2028 pooled σ / YES` vs AO=`0.8551 / NO`。額外Rolling進一步確認SCC pretraining不是Safety ceiling breaker。

Decision：`POST_CLOSURE_EVIDENCE_ONLY / SCC_ROLLING_CONFIRMS_FAIL / ORIGINAL_BQ_CLOSED_DECISION_UNCHANGED / NO_REOPEN / NO_ROBUSTNESS`。

### 2026-09-05 — MR-13BR Rolling closure + MR-13BS Same-Date Dynamic Hypergraph implementation

- **BR closure evidence**：使用者完成MR-13BR Rolling OOS。BR overall Daily/Global/Pair=`0.4132/0.3457/64.32%` vs AO=`0.4136/0.3365/64.33%`，Top-Bottom=`1.9988R vs 1.8203R`；Breakout overall=`0.3871/0.4233/65.18%` vs AO=`0.3821/0.4088/65.12%`。但primary Raw Safety Rolling=`0.3590/0.3210/62.89%` vs AO=`0.3601/0.3141/62.91%`，Breakout=`0.3244/0.3790/63.95%` vs `0.3370/0.3782/64.03%`；Pred-HS true-LS=`36.69% vs 36.67%`、Breakout=`35.77% vs 35.96%`。Conditional-MFE只在Breakout局部升至`0.4377/0.4843/67.93%` vs AO=`0.4195/0.4649/67.58%`。Decision=`FUTURE_PATH_UPSIDE_SIGNAL_PRESENT / SAFETY_CEILING_NOT_BROKEN / MODEL_GATE_FAIL / CLOSED / NO_PRETRAINING_SWEEP`。
- **New user decision**：使用者要求調查近期Dynamic graph / hypergraph跨股票研究，並在比較後明確「進行」第一順位：AO + detached latent + k=16 low-rank Dynamic Hypergraph + zero-init gated Safety residual。Registry分配下一個合法identity `MR-13BS`；BR/BQ歷史Decision不改寫，BP仍deferred。
- **Scientific base / exact controls**：MR-13AO的300×10、InceptionTime shared encoder、full-universe continuous Safety full-list ΔNDCG、true-HS=P50 Conditional-MFE sublist、1:1 loss、Seed42/Adam/downstream split/epoch-selection與Pred-Safety→Conditional-MFE inference固定。新architecture=`ARCH-inception_time_shared_safety_dynamic_hypergraph_mfe_v1`；只有Safety relational path新增topology。
- **Relational primitive**：每個交易日的AO shared latent形成`N×d` nodes；graph branch固定stop-gradient，`A=softmax(XW)`產生`N×16` learned soft incidence。Hyperedge state=`A^T X / membership_mass`，stock relational state=`A E`。`concat(detached X, X_rel)`經`Linear(2d→d)+ReLU`與learned sigmoid gate後，產生2-logit Safety residual。Final residual projection weight/bias固定zero-init；因此same-seed step-0 Raw Safety與MFE logits皆與AO bitwise exact。AO base Safety仍正常對shared encoder反傳；graph branch本身不得對encoder latent反傳。Conditional-MFE只讀原shared latent，不讀graph state。
- **Date coherence / PIT**：training generic date-coherent batcher在model宣告`requires_same_date_relations`時固定one complete date per batch；不因configured batch size切斷同日universe。Inference generic engine新增date-label capability，relational model缺少date labels即fail-fast，且同日rows即使超過batch size也保持完整單date job。Graph只用該decision date合法300-bar node representations，不使用future return/correlation、breakout outcome、portfolio state或跨date static graph。
- **Scope intentionally excluded**：第一輪不做Hawkes/event intensity、relation-trend、pairwise graph、sector/static relation prior、HGAT/multi-head/multi-layer、hyperedge-count sweep或external data。只有BS先對Safety產生material signal才允許第二階段研究relation change / tail-event。
- **Independent GPT checks**：reusable model component contract=`87/87 PASS`；policy SSOT=`13/13 PASS`；true-HS pair/qualification/boundary=`7/7 / 8/8 / 6/6 PASS`。Same-seed AO↔BS所有common state tensors exact，新增params=`35,459`且只屬hypergraph branch；zero-init step-0 Safety/MFE logits bitwise exact。Stop-gradient synthetic確認latent無gradient且incidence有gradient；date-coherent training/inference synthetic PASS。真實2-date、12-row、one-epoch optimization smoke loss=`0.693863` finite，residual projection與incidence皆完成非零update。Python compile/import PASS。GPT未執行`apps/run_bundle.py`或`apps/test_suite.py`。
- **Forward Gate**：先只跑Seed42 Forward並直接對MR-13AO。成功要求Raw Safety Daily/Global/Pair共同material改善、Pred-HS true-LS與P45-P55明確改善、Breakout同方向，Conditional-MFE不可material退化。若仍只有小幅rho／Pair增量則直接STOP，不做graph architecture sweep；只有material positive才進relation-trend/tail-event。

Decision：`MR13BS_IMPLEMENTED_RESULT_PENDING / AO_SUPERVISED_CONTRACT_EXACT / SAME_DATE_LOW_RANK_DYNAMIC_HYPERGRAPH / K16_FIXED / DETACHED_RELATIONAL_LATENT / ZERO_INIT_GATED_SAFETY_RESIDUAL / MFE_PATH_NO_GRAPH_INPUT / DATE_COHERENT_TRAIN_AND_INFERENCE / FORWARD_GATE_FIRST`。

### 2026-09-05 — MR-13BS Forward closure + MR-13BT Relation-Change implementation

- **BS user evidence**：使用者完成`Forward OOS模型比較`。MR-13BS overall Daily/Global/Pair=`0.4041/0.3350/63.96%` vs AO=`0.4042/0.3368/63.96%`；Breakout=`0.3887/0.3934/64.90%` vs AO=`0.3850/0.3804/64.73%`。
- **BS primary Safety Gate**：Raw Safety Forward=`0.3533/0.2931/62.68%` vs AO=`0.3532/0.2929/62.68%`，Pred-HS true-LS=`36.81%` vs `36.84%`；Breakout Raw Safety=`0.3166/0.3558/63.70%` vs AO=`0.3135/0.3554/63.65%`、true-LS=`36.50%` vs `36.94%`。這些僅屬`+0.00x / +0.0xpp`級，遠低於事前material Gate。
- **BS secondary evidence**：Breakout Conditional-MFE=`0.4248/0.4512/67.47%` vs AO=`0.4116/0.4353/67.22%`，顯示same-date relational state對upside/path representation有局部訊號；但full-universe NDCG/Top-K Lift/Oracle overlap=`0.7462/1.7014R/9.88%` vs AO=`0.7475/1.8273R/10.73%`，Boundary gap=`-0.0197` vs `0.2068`。因此不能把Breakout MFE增量解讀成Safety突破。
- **BS Decision**：`SAME_DATE_RELATIONAL_STATE_SAFETY_SIGNAL_NEAR_ZERO / MODEL_GATE_FAIL / CLOSED / NO_ROLLING / NO_GRAPH_ARCHITECTURE_SWEEP / NOT_PROMOTED`。不做k/layer/head/HGAT/Hawkes/pairwise/sector sweep。
- **New user decision**：使用者同意只再測cross-stock OHLCV relational family最後一個機制：relation change/deterioration。Registry分配`MR-13BT`。
- **BT exact controls**：MR-13AO的300×10、shared InceptionTime、full-universe Safety full-list ΔNDCG、true-HS=P50 Conditional-MFE、1:1 objective、Seed42/Adam/split/epoch-selection/inference不變；MR-13BS的k=16 same-date low-rank soft incidence、current-state Safety residual、stop-gradient、zero-init與MFE-no-graph contract全部保留。
- **BT only treatment**：對每個current decision date，另外materialize前一交易日**完整score-eligible universe**的canonical 300×10 windows；不能從current normalized window切片偽造previous-day input，因價格anchor與volume median/IQR normalization均依完整window重算。Current與prior relation-only views以同一shared encoder在`no_grad + encoder eval`下建立，因此不更新AO BatchNorm running state也不回傳encoder gradient。兩日共用同一incidence projection錨定16 latent hyperedge identity；固定`edge_change=current_edge_state-previous_edge_state`，再用current membership映回stock，經獨立projection + learned gate + zero-init 2-logit residual只修正Safety。最早無prior完整date時change residual精確為0，不建立missingness shortcut。
- **PIT / batching**：canonical `same_date_relations.py`唯一持有previous-trading-date index semantics；prior ids從完整group-date universe解析，必須strictly earlier。Training current graph仍one complete date per batch；previous relation tensor讀完整prior-date features。Inference即使configured batch size切斷日期，也按完整current date job並對應完整prior date。
- **Scope excluded**：不加入Hawkes/event intensity、pairwise graph、sector/static prior、EMA/lag-length、k、layer、head、detach、graph topology sweep。BT若仍marginal，即`CROSS_STOCK_OHLCV_RELATIONAL_CEILING_NOT_BROKEN`並終止整family。
- **Independent GPT checks**：same-seed BS→BT所有BS-common state tensors bitwise exact；BT只新增6個`relation_change_*` tensors。Zero-init step-0 Safety/MFE logits與BS bitwise exact。Synthetic previous-date planner確認`D_t→full D_{t-1}`且首日empty；relation-change current/previous latent stop-gradient，incidence/change projection具gradient；relation-only encoder view不改BS-common BatchNorm running buffers；strict batched inference與manual current+previous-date forward exact。Reusable model component contract=`94/94 PASS`。真實3-date×6-stock、one-epoch training smoke使用正式full-list ΔNDCG reduction，loss=`0.700168` finite，relation-change residual weight由0更新至norm=`0.009785`。GPT未執行`apps/run_bundle.py`或`apps/test_suite.py`。

Decision：`MR13BT_IMPLEMENTED_RESULT_PENDING / BS_STATE_CONTROL_EXACT / PREVIOUS_TRADING_DATE_RELATION_CHANGE_ONLY / FULL_PRIOR_DATE_CANONICAL_300X10 / SHARED_INCIDENCE_EDGE_IDENTITY / NO_GRAD_NO_BN_STATE_RELATION_VIEWS / SECOND_ZERO_INIT_GATED_SAFETY_RESIDUAL / MFE_UNTOUCHED / FINAL_OHLCV_RELATIONAL_GATE / FORWARD_FIRST`。

### 2026-09-05 — MR-13BT Forward result / final OHLCV cross-stock relational family closure

- **User evidence**：使用者完成`Forward OOS模型比較`。MR-13BT overall Daily/Global/Pair=`0.4025/0.3088/63.91%` vs AO=`0.4042/0.3368/63.96%`；Breakout=`0.3848/0.3843/64.96%` vs AO=`0.3850/0.3804/64.73%`。
- **Primary Raw Safety Gate**：BT Forward Daily/Global/Pair=`0.3566/0.2882/62.79%` vs AO=`0.3532/0.2929/62.68%`，即`+0.0034/-0.0047/+0.11pp`，方向不一致且遠低於事前material Gate；Pred-HS true-LS=`36.81%` vs AO=`36.84%`僅改善`0.03pp`。Breakout Raw Safety=`0.3209/0.3639/63.62%` vs AO=`0.3135/0.3554/63.65%`，Daily/Global局部上升但Pair反而`-0.03pp`；Pred-HS true-LS=`35.85%` vs `36.94%`改善`1.09pp`，仍未達約2pp Gate且不是full-universe共同改善。
- **Conditional-MFE / economic geometry**：BT HS-only Forward=`0.3971/0.2789/63.91%` vs AO=`0.3969/0.3104/63.90%`；Global rho明顯下降。Breakout=`0.4233/0.4438/67.59%` vs AO=`0.4116/0.4353/67.22%`局部改善，但Pred-HS MFE=`0.940R`低於AO `1.000R`。因此relation-change仍呈現「局部Breakout/path訊號，不是Safety ceiling breaker」。
- **Ranking cost**：full-universe NDCG@K=`0.7397` vs AO=`0.7475`、Top-K Lift=`1.5805R` vs `1.8273R`、Oracle overlap=`8.69%` vs `10.73%`、Boundary=`49.24%` vs `50.84%`、Boundary gap=`-0.1140` vs `+0.2068`；overall Top-Bottom=`1.7772R`亦低於AO `1.8767R`。這些負面證據阻止以Breakout局部gain進Rolling。
- **Scientific conclusion**：BS已證明same-date learned group state幾乎不移動Safety；BT再加入previous-trading-date hyperedge deterioration後仍只得到marginal/inconsistent Safety shift。依BT建立前明定的final-gate stop rule，這足以停止**OHLCV-derived cross-stock Dynamic Graph/Hypergraph relational family**。不做BT Rolling、robustness、lag/EMA、Hawkes、pairwise graph、sector prior、k/layer/head/detach或graph topology sweep。
- **MR-13BP disposition**：BP從未取得Forward結果，因此不寫`MODEL_GATE_FAIL`；但BT已是使用者授權的family final gate，故BP改為`IMPLEMENTED / UNEXECUTED / CLOSED_BY_FAMILY_STOP / NOT_A_MODEL_FAIL`，保留identity/code供歷史重現，不再作pending extension。
- **Current workflow**：Training pair回MR-13AO scientific reference/fallback；BS/BT artifacts與implementation保留歷史重現，不自動promotion。下一個Priority 1只能是**真正新增PIT-safe orthogonal information source**的設計/Data Gate；尚未選定information family、尚未分配新MR identity。

Decision：`RELATION_CHANGE_SAFETY_SIGNAL_MARGINAL_AND_INCONSISTENT / MODEL_GATE_FAIL / CLOSED / NO_ROLLING / NO_ROBUSTNESS / CROSS_STOCK_OHLCV_RELATIONAL_FAMILY_STOP / BP_UNEXECUTED_CLOSED_BY_FAMILY_STOP / AO_REFERENCE_FALLBACK / NEXT_ORTHOGONAL_INFORMATION_DESIGN`。

### 2026-09-06 — MR-13BU Forward closure / MR-13BV Adaptive Input Context authorization

- **BU completed Forward evidence**：overall Daily/Global/Pair=`0.3989/0.3090/63.77%` vs AO=`0.4042/0.3368/63.96%`；Breakout=`0.3733/0.3566/64.50%` vs AO=`0.3850/0.3804/64.73%`。Primary Raw Safety=`0.3484/0.2874/62.51%` vs AO=`0.3532/0.2929/62.68%`；Breakout Raw Safety=`0.3043/0.3459/63.61%` vs AO=`0.3135/0.3554/63.65%`。Pred-HS true-LS=`36.91%` vs `36.84%`，Breakout=`38.04%` vs `36.94%`。HS-only Conditional-MFE Forward=`0.3879/0.2768/63.56%` vs AO=`0.3969/0.3104/63.90%`，Breakout=`0.3974/0.3968/66.72%` vs `0.4116/0.4353/67.22%`。因此adaptive future-horizon supervision不是Safety ceiling breaker，且有MFE collateral damage。
- **BU decision**：`ADAPTIVE_HORIZON_SAFETY_DEGRADES / CONDITIONAL_MFE_DEGRADES / MODEL_GATE_FAIL / CLOSED / NO_ROLLING / NO_ROBUSTNESS / NO_HORIZON_SWEEP / NOT_PROMOTED`。效能優化後Epoch 1/2=`116.0s/116.8s`、Selection refit=`124.3s`，故本次scientific FAIL與先前CPU target-feeding問題已乾淨分離。
- **New user authorization**：BU結案後使用者明確追問「input length自己學剛才試了嗎」；確認BU只學future horizon、未測adaptive past context後，使用者明確回覆「進行」。這構成對BU原stop rule中`NO_INPUT_LENGTH_SWEEP`的具名新scientific identity override；不得覆寫BU或當作BU tuning。分配`MR-13BV`。
- **MR-13BV scientific contract**：max input仍300 bars，固定look-back candidates=`30/60/120/300`，future max horizon/target/loss全部恢復AO：40-bar full-universe Safety + true-HS Conditional-MFE，Seed42/Adam/split/PIT/epoch-selection/inference固定。相同AO encoder weights產生各suffix view；30/60/120 auxiliary views stop-gradient並禁止BatchNorm running-state mutation。Sample-specific softmax gate從canonical 300-bar latent決定4個look-back weights；adaptive latent與300-bar latent之差只進zero-init two-logit Safety residual。Conditional-MFE仍只讀canonical 300-bar latent。
- **Attribution / stopping**：BV不加future-horizon trajectory、aux loss、600-bar、temperature、hidden gate、window-count sweep、remove-detach或其他input architecture。Seed42 Forward first，primary reference=AO。只有Raw Safety Daily/Global/Pair、Pred-HS purity/P45-P55及Breakout共同material改善且Conditional-MFE不material退化才允許後續；否則直接關閉adaptive-input family。

Decision：`MR13BV_IMPLEMENTED_RESULT_PENDING / ADAPTIVE_30_60_120_300BAR_PAST_CONTEXT / AO_40BAR_TARGET_AND_LOSS_EXACT / SAFETY_ONLY_ZERO_INIT_RESIDUAL / SHORT_VIEW_STOP_GRADIENT / CONDITIONAL_MFE_300BAR_ONLY / AO_RETAINED_IN_COMPARE / FORWARD_GATE_FIRST`。

### 2026-09-27 — Production-PIT Phase 1 empirical closure / Phase 2 PARAM-P5 + C84 implementation

- **Phase 1 evidence closure**：使用者完成Extending-Window Rolling 2021-01-01～2026-03-02的8-seed robustness。C62 Production-PIT `base-finalists-agree` mean Return/MDD/RoMD/Annual=`184.99%/16.66%/11.40/22.12%`；C61 Full `base-finalist-best`=`162.43%/15.69%/10.64/20.27%`；C58 Min `base-finalist-best`=`132.01%/13.84%/9.91/17.47%`。
- **Stability interpretation**：C62對C61 same-seed RoMD勝負=`4/8 vs 4/8`、all-pairs `P(C62>C61)=53.12%`；C62對C58亦=`4/8 vs 4/8`、all-pairs=`57.81%`。因此C62是有效且具競爭力的production reference，但現有證據**不支持**把`base-finalists-agree`宣稱為跨seed穩定優於`base-finalist-best`。Phase 1正式結案並凍結，不以已觀察OOS回頭調production selector、trial budget、window或rules。
- **Phase 2 identity**：使用者明確授權進行non-DL validation control，分配`PARAM-P5`與`SR-C84`。唯一treatment為`validation_budget_refit_v1`：selection window最後12M作inner validation；事前固定trial checkpoints=`50/100/150/200/250/300`；每個checkpoint只使用前N個inner-train trials形成同一`base-finalists-agree` policy並在validation replay canonical optimizer portfolio score；選分數最高N*，exact tie取較小N。
- **Full refit semantics**：選N*後丟棄inner study作為正式params來源，另建fresh Optimizer study，在完整120M Train+Validation selection window只跑N* trials，再產生正式`base-finalists-agree` params。每個fold的validation period、checkpoint scores與selected N*保存於schedule fold evidence；OOS rows不參與budget selection。
- **SSOT / evaluation**：PARAM-P5由Optimizer唯一產生，research-only physical truth位於`models/research/breakout_quality/strategy_params/param_p5_validation_budget_refit/`。C84 OOS與Rolling都消費同一physical schedule，只分別採freeze與date-effective consumption semantics；C62/C61/C58既有canonical truth與fingerprint保持不變。C84只加入Phase 2 dedicated research Compare Suite，不加入`extending_current`。Forward／Rolling與其Multi-seed Robustness都由同一current config/Registry可直接執行；不以計算成本另設人工authorization block。Phase 2 robustness採same-seed end-to-end parameter benchmark：C62/C61/C58使用canonical benchmark Optimizer params，C84以相同benchmark seed重跑`validation_budget_refit_v1` PARAM-P5 producer。
- **Execution surface**：維持既有Research單一入口與穩定工作類型；Phase 2 current profiles由`config/strategy_compare.py`選定，使用`apps/research.py → 策略組合比較 → Forward OOS Test / Rolling OOS Test`，沿用Strategy Compare canonical orchestration／renderer；不新增experiment-specific App。

Decision：`PHASE1_RESULT_AVAILABLE_BASELINE_FROZEN / C62_COMPETITIVE_NOT_STABLY_DOMINANT / PARAM_P5_C84_IMPLEMENTED_RESULT_PENDING / VALIDATION_ONLY_BUDGET_SELECTION / FRESH_FULL_REFIT / OOS_FEEDBACK_PROHIBITED / SINGLE_PHYSICAL_PARAM_TRUTH / ROBUSTNESS_CONFIGURED / NOT_PROMOTED`。

### 2026-10-10 — MR-13BX Seed42 Forward + 6-fold Rolling OOS completed; AO-relative model Gate FAIL / CLOSED

- **Evidence origin / provenance**：使用者本機日誌 `貼上的文字 (1)(20261010-023028).txt`，對應本輪專案ZIP `test-branch-1_20261010_093202_734ad14e.zip`（SHA256=`55c1fcfc31cdeaa8529f5cedc868062c0fb0907c9793bd926543e2a246acd623`）。本輪僅讀取本機人讀輸出，未收到完整checkpoint／score CSV／model report JSON，故不宣稱獨立重算結果。這些數值屬反覆使用的**迭代研究OOS**，不是untouched holdout。
- **Identity and controlled contrast**：`MR-13BX / ARCH-inception_time_shared_safety_mfe_raw_quote_mask_v1 / PROFILE-daily_universal_raw_mask_adjusted_target_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise`；個股+0050的Raw OHLCV加各自availability mask，`300×12`，無報價日只使用中性正規化feature並保留mask。AO Adjusted ticker/date/Volume、40-bar Safety/MFE target、InceptionTime雙頭／Loss、Seed42、時間切分維持原設計；正式模型與PIT工件在BX獨立namespace，不修改AO／BW。
- **Forward / fitting**：Seed42 Forward 完成（CUDA/bfloat16，Selection best epoch=1/200；refit 1 epoch）。Forward OOS SOP overall Daily/Global rho/Pair=`0.3935/0.1992/63.57%`；Breakout=`0.3771/0.2019/64.86%`。模型雙頭Forward Raw Safety Daily/Global/Pair=`0.3506/0.2404/62.57%`，Conditional-MFE=`0.3873/0.1825/63.52%`。Forward報表與模型checkpoint存在；單次Console完成列出`OOS daily rho=0.3873`對應head-specific Conditional-MFE，不能誤抄為Standard SOP final-score Daily rho。
- **Rolling / PIT integrity**：2021-01-01～2026-03-02、六fold、`630,589/630,589 = 100%` PIT coverage，fold1符合exact fitting identity而Forward-import checkpoint，另外5 folds新建。`score`對照樣本全市場`608,204`、Breakout slice`17,346`，與AO為同scope比較。Rolling輸出自有`模型 Gate=PASS`表示跑通其內部模型檢核；並非對AO的material-improvement Gate，更非策略Gate／promotion。BX Rolling stability：valid years=5、positive rho years=4/5、positive top-bottom years=5/5、max adjacent score mean drift=`1.0583 pooled σ`、drift=`YES`；AO對照 positive rho years=5/5、drift=`NO`（`0.8551σ`）。
- **Rolling overall AO vs BX**：Daily rho=`0.4136` vs `0.3957`（BX−AO `−0.0179`），Global rho=`0.3365` vs `0.2111`（`−0.1254`），Pair=`64.33%` vs `63.60%`（`−0.73pp`）；Top-Bottom Target=`1.8203R` vs `0.6699R`；NDCG@10=`0.7480` vs `0.7067`、Top-K Lift=`1.7705` vs `1.5622`。Breakout Daily/Global rho=`0.3821/0.4088` vs `0.3675/0.2045`，Pair=`65.12%` vs `63.94%`，Top-K Lift=`0.6502` vs `0.6138`。
- **Joint Safety / MFE gate**：Raw Safety Daily/Global/Pair AO=`0.3601/0.3141/62.91%` vs BX=`0.3451/0.2117/62.30%`；Conditional-MFE Daily/Global/Pair AO=`0.4113/0.3081/64.43%` vs BX=`0.3914/0.1971/63.61%`。Pred-HS true-LS AO=`36.67%` vs BX=`37.07%`（惡化0.40pp），True-HS recall=`63.17%` vs `62.77%`；Top10 High-Safety=`28.18%` vs `28.47%`（微升0.29pp）但Top10 HM/HS=`21.79%` vs `21.52%`、High-MFE=`76.10%` vs `74.84%`。Breakout Top10 HM/HS=`24.06%` vs `24.36%`、High-Safety=`29.78%` vs `30.28%`雖有小幅正差，但Breakout Raw Safety Global=`0.3782` vs `0.2022`、Conditional-MFE Global=`0.4649` vs `0.2291`與Pair=`67.58%` vs `65.83%`顯著退化，Pred-HS True-HS recall=`60.85%` vs `57.02%`亦降低。不能以局部top-tail uplift掩蓋共同Safety／MFE損失。
- **Decision / action**：`RESULT_AVAILABLE / AO_RELATIVE_MODEL_GATE_FAIL / RAW_QUOTE_MASK_DOES_NOT_BREAK_SAFETY_CEILING / DRIFT_REGRESSION / CLOSED / NO_BX_ROBUSTNESS / NO_TUNING_SWEEP / NO_STRATEGY_PROMOTION`。這是對「Raw OHLC＋雙availability mask替代Adjusted input」這一**特定組合**的失敗，不證明所有非還原價資料或PIT-safe正交資訊沒有研究價值。AO回到Research model reference，原AO舊`MODEL_GATE_FAIL/CLOSED`評估不被重新定義、也不自動進Trading。BX checkpoint、PIT scores、report與identity保持可稽核；下次需要全新資料／機制，先做PIT Data Gate、未經授權不預占新MR ID。`MR-13BW`原300×10 direct Raw仍DATA_BLOCKED、未取得模型結果，不把BX FAIL冒充BW FAIL。

### 2026-10-10 — 融資融券 PIT Data Gate：本機實測覆蓋已取得，正式 Input 仍 BLOCKED

- **本輪基準／來源**：`test-branch-1_20261010_111847_85f9eb6c.zip`，SHA256=`16b0c6550ea1974eb1d3d78fe2fc3e71530d297c411380479e450cb8376b3fb8`；專案設定、Experiment Registry／Log／Queue已依序讀取。使用者提供本機產出的 `data_gate.md` 原文（非本輪重跑 Provider Full Snapshot）；本輪只同步歷史研究證據與條件，不覆寫數據／工件、無MR分配／模型訓練。
- **Snapshot／日曆範圍**：`TaiwanStockMarginPurchaseShortSale`，Research cutoff `2026-03-02`，來源日期 `2001-01-05～2026-03-02`；READY Provider台股日曆 `1999-01-05～2026-12-31`。原Full Adjusted控制資料中更早 `1,984` 日期／`179,275` 股票日未涵蓋，明確排除於被驗證日曆與coverage分母，不能解讀為來源不存在或PIT驗證通過。T日只用正式交易日T−1的已記錄來源列，T日資料不得當日使用。
- **OOS 控制格實測**：`667,613` 股票日、全部有可驗證前一台股交易日；T−1來源列存在率=`98.009%`；融資使用率可計算=`96.346%`、融券使用率可計算=`96.346%`、雙者同時=`96.346%`。融資／融券限額為0各=`11,104`個控制觀測；比率>1：融資=`48`、融券=`0`（僅供稽核，不自動判錯）。此為**Full Adjusted control grid**，不是精確AO `daily_eligible_stock_days`、不代表每個300日模型輸入窗皆完整，更不是合格PIT版本證據。
- **逐年覆蓋訊號**：1999、2000因margin provider資料尚不存在可計算率=`0%`；2001兩率可計算=`87.343%`；2005雙率=`79.045%`、2006=`77.602%`，2020=`94.243%`、2021=`94.801%`、2022=`95.916%`、2023=`96.546%`、2024=`96.990%`、2025=`97.065%`、2026截至研究cutoff=`97.580%`。不能以OOS平均掩蓋Train／Validation早期缺口；若將來獲得合法資料契約，仍需保留Missing vs Zero Limit的模型Input語意。
- **來源異常**：使用者本機Audit顯示無效來源日期=`0`、股票代碼=`0`、同股票同日期重複=`0`、schema問題=`[]`。官方FinMind文件 `https://finmind.github.io/tutor/TaiwanMarket/Chip/` 記載2001起始、平日21:00一般更新時間，**沒有給出每筆歷史首發與修訂vintage**。FinMind `https://finmind.github.io/WhatIsNew/` 在2026-09-01公告TPEX `ShortSaleBuy/ShortSaleSell`歷史欄位對調校正，並說明融券餘額不受該次修正影響；不能因此擴張推論歷史`MarginPurchaseTodayBalance/ShortSaleTodayBalance/Limit`永不修訂。候選feature先限定餘額與正限額比率；Buy/Sell Flow不納入本研究。
- **科學決策**：`COVERAGE_EVIDENCE_AVAILABLE / T_MINUS_ONE_ALIGNMENT_AUDITED / HISTORICAL_PUBLICATION_REVISION_VINTAGE_UNPROVEN / PIT_REVIEW_REQUIRED / MODEL_INPUT_NO / NO_MR / NO_TRAINING`。本輪有足夠證據**停止重複coverage Audit**，而不是授權DL；FinMind截至2026的現時快照不能當成2021年的as-of快照。只有可核對歷史決策時值的權威來源vintages／修訂證據，或另行取得明確且符合專案D1的scientific contract，才能繼續此來源研究。若版本證據不可取得，下一個研究假說應另選PIT合法的新資訊來源；既有AO checkpoint／Target／策略與報表維持不變。

### 2026-10-10 — TWSE 2024-02-01 單日官方確認餘額比較成功：來源交叉核對完成，Vintage 仍未證實

- **本輪程式基準／證據性質**：使用者上傳 `test-branch-1_20261010_120843_c9e8c3d0.zip`，SHA256=`0f5f2621697c380c4b94182f7b6c0351e297feedcf1763f139b4e63d7ad875e2`；本輪新解壓目錄 `stock_margin_1131_review_20261010_u1/`。使用者另提供本機一次性 `publication_probe.md` 實際結果文字；GPT 未取得對應原始本機 Provider Snapshot、TWSE response bytes、`publication_probe.json` 或 5 筆股票差異內容，**不能杜撰差異股票或差額**。
- **實測比對與時間語意**：已固定官方報表 T−1=`2024-02-01`，該報表「前日餘額」對應 T−2=`2024-01-31`，預測日 T=`2024-02-02`。TWSE個股=`1,136`，可比較=`1,136`，融資一致=`1,131/1,136`（`99.56%`），融券一致=`1,136/1,136`（`100%`），雙欄一致=`1,131/1,136`（`99.56%`）；融資差異=`5` 檔。TWSE來源TLS模式=`CA_CHAIN_AND_HOSTNAME_REQUIRED_X509_FORMAT_COMPAT`，確認局部 X.509 格式相容處理可在使用者本機完成安全連線；它不是舊版原始發布時間的證據。
- **範圍與缺口**：這是今天重查官方2024-02-01報表的 **current-vintage** 數值交叉核對，只證明該日當前可得版本與FinMind相符程度，未證明2024-02-02盤前可得的歷史不可變版本、真正首發/後續修訂紀錄，亦未稽核TPEx、跨年份全部股票日、限額effective date／使用率。五筆融資差異已由現有 `comparison.sample_mismatches` 保留於本機 `publication_probe.json`；如需來源歸因，僅讀既有證據，不重新抓六日或重算覆蓋率。**不得把2024單日一致率外推為全歷史資料準確率或宣稱已通過PIT。**
- **決策／stop rule**：`OFFICIAL_CURRENT_VINTAGE_SOURCE_COMPARISON_OBSERVED / TLS_ACCESS_RESOLVED / 5_MARGIN_DISCREPANCIES_UNATTRIBUTED / HISTORICAL_VINTAGE_NOT_PROVEN / PIT_REVIEW_REQUIRED / MODEL_INPUT_NO / NO_NEW_MR / NO_FITTING`。來源聯通與單日值對照問題已足夠回答，**結束重複的current-vintage官方網頁取樣鏈**。只有取得具可信封存／交付時間的當年TWSE/TPEx原始版本與適用日期證據（或另行授權且符合D1的新科學資料契約）才重啟margin Input科學授權審查；不更動AO、Provider Snapshot、Research V1、Target、checkpoint、正式SOP/Compare或Trading。研究紀錄同步不代表本機formal suite PASS。

### 2026-10-10 — FinMind T−1 YesterdayBalance 對照本機實測：TWSE 確認餘額 1,136/1,136 精確一致；數值語意結案、PIT 仍阻擋

- **本輪程式基準與證據**：使用者上傳 `test-branch-1_20261010_125708_f594b3c0.zip`（SHA256=`0763013a67c5fbe94754244011e7930209845007a7c43983066231adf514d621`）；另提供本機 `FinMind YesterdayBalance 同日 TWSE 確認餘額對照（唯讀）` 人讀報表全文。此為**使用者本機已執行的真實數值結果**，不是 GPT 在不含完整 Provider Snapshot 的 ZIP 上自行重跑；該報表的 full raw bytes／報表 JSON／單位來源仍未隨本輪 ZIP 提供，不能追加未觀測的逐年結論。
- **時間與對照合約**：已保存 TWSE 報表日 T−1=`2024-02-01`，其「前日餘額」代表 T−2=`2024-01-31` 確認餘額，對應預測日 T=`2024-02-02`。舊比較讀 FinMind T−2 `MarginPurchaseTodayBalance/ShortSaleTodayBalance`；新比較**同報表日 T−1** FinMind `MarginPurchaseYesterdayBalance/ShortSaleYesterdayBalance`。兩者均與 TWSE 原始數字作完全相等比較，不設容忍誤差、不修改原來源、目前也未單獨完成數字單位驗證。
- **實測結果（限 TWSE 上市且單一 current-vintage 重查日期）**：官方個股及可比較=`1,136/1,136`。舊 T−2 TodayBalance 融資=`1,131/1,136`、融券=`1,136/1,136`、雙欄=`1,131/1,136`；新 T−1 YesterdayBalance 三項全部為 **`1,136/1,136`（100.00%）**。舊融資差異五檔 `2409/2535/2614/3481/6271` 均以新欄位取得**精確零差值（5/5 MATCHED）**：新值依序=`89,670/104/4,228/162,388/8,221`。這支持 **T−1 YesterdayBalance 比 T−2 TodayBalance 更貼近已確認餘額的欄位語意**；不能單憑結果斷言五筆差異皆由調帳／修訂造成。
- **證據級別／止損規則**：`LOCAL_CURRENT_VINTAGE_SAME_DATE_SEMANTICS_CONFIRMED / FIVE_DIFFERENCES_RECONCILED / SOURCE_NUMERIC_AUDIT_CLOSED`。這不是歷史 contemporaneous vintage 的首次發布／修訂證據，不能外推其他年度、TPEx、所有 AO Eligible 樣本或預測期實際使用時點；先前 OOS `98.009%` 與 `96.346%` 來自**不同 T−1 來源／限額使用率口徑**，不是 T−1 YesterdayBalance 的歷史覆蓋率。正式仍 `PIT_REVIEW_REQUIRED / INPUT_NOT_AUTHORIZED / NO_NEW_MR / NO_FITTING / AO_UNCHANGED`。本條語意對照**已達決策充分，停止重跑相同日期的 TWSE/FinMind 數值比對、TLS、JSON、重複 Coverage Audit**。
- **下一個科學決策（待使用者明確授權）**：可以提出**獨立、非正式 PIT 的 current-vintage 探索性研究**，候選只用 T−1 的融資／融券 `YesterdayBalance` 表達 T−2 確認餘額，加上來源存在／不適用 Mask；不能將缺值當 0、不能以 T−1 的 next-business-day 限額除以 T−2 餘額、不能讀 T 當日資料；Train/Validation/OOS 隔離及 iterative OOS contamination 必須另訂新 scientific identity 與研究授權，探索結果不得宣稱嚴格 PIT 或升格 Trading。**這一輪不實作特徵、不分配 MR、不啟動學習或重新規劃 AO 工件。** 若需嚴格 PIT，仍要可證明歷史版本的第三方時間證據；若使用者不授權探索路線，此候選維持 BLOCKED，不再徒增診斷。

### 2026-10-10 — MR-13BY Seed42 Forward 同樣本AO比較：使用者本機結果與下一步

- **證據來源／邊界**：使用者2026-10-10貼出的`apps/research.py → [4] Audit → [2] 一次性專題 → MR-13BY → [3] Forward OOS模型比較`終端輸出。BY與frozen AO以同一`608,204`個OOS group進行比較；BY fitting fingerprint=`d0619eff9ed392eaf286ca19b2a3e097c43908fa9adb7e2d4c80cdee3ed21486`。BY Forward `[1]` 顯示REUSE，`[3]` 顯示REUSE/DONE。使用者提供的ZIP沒有該環境原始score、JSON或Provider Snapshot可供GPT獨立驗證，因此以下註明**USER_REPORTED_RESULT**，不可冒稱本輪重新完成訓練或實體資料QA。
- **受控單因素對比**：AO仍是Adjusted 300×10；BY在相同AO Target/Loss/Seed42/Forward劃分上只增加FinMind READY Snapshot current-vintage T−1 YesterdayBalance衍生的融資融券偏差與各自mask四欄（300×14）。未證明歷史首次發布／修訂Vintage，故正式PIT/Trading授權仍是NO。

| 同樣本Forward指標 | AO | BY | BY−AO | 越高越好 |
|---|---:|---:|---:|:---:|
| Raw Safety Daily rho | 0.3532 | 0.3201 | -0.0331 | 是 |
| Raw Safety Global rho | 0.2929 | 0.2668 | -0.0261 | 是 |
| Raw Safety Pair | 0.6268 | 0.6138 | -0.0130 | 是 |
| Conditional-MFE Daily rho | 0.3969 | 0.3723 | -0.0247 | 是 |
| Conditional-MFE Global rho | 0.3104 | 0.2987 | -0.0117 | 是 |
| Conditional-MFE Pair | 0.6390 | 0.6295 | -0.0095 | 是 |
| Pred-HS true-LS (%) | 36.8430 | 38.0132 | +1.1703 pp | 否 |
| Selected HM/HS (%) | 25.9352 | 25.0540 | -0.8811 pp | 是 |
| Selected High-Safety (%) | 44.2228 | 43.3250 | -0.8978 pp | 是 |

- **觀察與判斷**：九項方向全不利；Raw Safety Daily rho約降低`0.0331`、Pred-HS的true-LS污染約增加`1.17pp`，所以預先設定的「Safety與選股純度共同改善、Conditional-MFE不退步」Forward Gate **未通過**。這是同一迭代OOS的模型對照，不能從608,204股票日直接當成608,204個獨立統計樣本，也不能由九項負面差異推論新的正交資訊永遠不可能有效。
- **使用者後續指示與最小下一步**：使用者要求不再花時間調整一次性BY Audit報表，繼續BY研究。既有Forward結果不可覆寫重算；下一步限定既有**Seed42 Rolling OOS → AO同樣本Rolling比較**，用來辨別Forward負面差異是否在年度再訓練後仍持續，不讓Rolling的OOS指標回流到 fitting、loss、epoch選擇或特徵設計。只有Rolling證明相對Safety與選股純度改善且Conditional-MFE不惡化時，才另外決定是否有必要研究跨seed穩健性；若Rolling仍全面退步，BY停止並保留負面結果。這不是解除PIT vintage gate的授權。

### 2026-10-11 — 使用者研究排序決策：No-DL 新券商費率重訓優先；DL 策略歷史結果標記過時

- **本次授權**：短期繼續 `MR-13H／AH／AK／AO` **DL 模型本身**的 Target／輸入／模型與模型預測能力研究；暫緩 DL→策略重新最佳化。先處理 Research `[3]` 既有 No-DL 比較對象 `SR-C58／C61／C95` 對應的 Min Base-Finalist-Best、Full Base-Finalist-Best、Full Corrected Same-Original-Signal Agree 4×300，在相同 `broker_cash_v1` 券商現金費稅下以新策略科學身分重新訓練、回測並比較。
- **DL 經濟結果效力**：過往 `SR-C77`（H）、`SR-C80/C81`（AH）之 Return／MDD／RoMD／EV(A) 等屬歷史費稅語意，**對現行 broker-cash 決策已過時**；`SR-C82/C83`（AK）原本就是 `RESULT_PENDING`，不得寫成已取得正式策略結果；AO 尚無可據以宣稱完成的策略轉換績效。未來 DL 對應策略必須使用新 `SR-C*`／必要時新 `PARAM-*` identity，經 canonical Optimizer／Portfolio producer 重訓／重算 OOS、Rolling，不能覆寫舊經濟結果。
- **不受此次決策否定的證據**：H／AH／AK／AO 的原模型 checkpoint、預測分數、Spearman／Pair／Target 等，在各自原 scientific fitting/PIT 契約有效的前提下仍可用於模型研究；若後續使用同一 fitting identity，可驗證後 REUSE 模型工件，不能以券商費率變更為理由自動宣告需重新訓練 DL 權重。舊策略 `RESULT_AVAILABLE`、`RESULT_PENDING`、實驗決策及 original fingerprints 一律保持歷史原貌。
- **No-DL 執行限制／實際狀態**：`PARAM-P15` 是已登記的新券商口徑 Full Agree producer，但 `RESULT_PENDING`；舊 `C58/C61` 的單 seed canonical producer 仍綁定 frozen rebate 身分，不可直接換費率重建。新的 Min Best／Full Best 需要先完成獨立 scientific registration、合法單 seed canonical producer、physical artifact isolation 與新 Compare Suite，才可經既有 Research `[2]/[3]` 合法執行；**本輪沒有取得真實訓練工件、沒有執行正式 No-DL retraining/OOS/Rolling，也沒有產生新的策略績效或分配尚未實作的新 ID**。對應現行身分狀態以 Registry 為準。
- **研究治理**：這是經濟口徑變更與研究優先順序決策，非模型品質倒退的 evidence；舊 C58/C61/C95 及 DL Strategy conversion historical result 不得被新 execution 靜默覆寫，亦不得用修改報表數字代替以新費率重算完整交易路徑與現金時點。

### 2026-10-14 -- MR-13CC | User-reported Rolling OOS result and comparison caveat

- **Source/evidence level**: User pasted the Research [1] -> [4] Rolling OOS Model Comparison console (six models REUSE READY); source code ZIP `test-branch-1_20261014_115208_55a4328c.zip` SHA256=`87ab6b3389aba4c494d5c12095718dbb27b6cf5af02b65bc21dbe4977bea4a6a`. The ZIP does not include full historical score sidecars or the canonical comparison JSON. Values below are *user-reported aggregate OOS evidence*, independently inspected for metric semantics in source, not independently recalculated. Historical Forward comparison results not provided this turn.
- **Paired D40 truth (same ticker/date; 608,204 stock-days, 1,203 dates)**: AO vs CC D40 Daily rho=`0.4570 / 0.4621`, D40 Pair=`66.17% / 66.35%`, Pred-P50 low-MAE precision=`67.42% / 67.80%`, top predicted-Safety decile actual D40=`0.376R / 0.384R` (lower is safer). Thus the D40 correlation/precision gain is small and the top-decile adverse outcome is marginally worse. Old-vs-D40 True-HS P50 overlap=`64.94%`; post-earliest-peak additional adverse incidence=`45.95%`, median extra adverse=`0.000R`.
- **Common old-Safety and original final-score evidence**: AO vs CC original adverse-to-peak Raw Safety Daily rho=`0.3601 / 0.3455`, Pair=`62.91% / 62.35%`, Pred-HS true-LS=`36.67% / 37.06%` (lower is better), true-HS Conditional-MFE Daily rho=`0.4113 / 0.4070`. Final overall Daily rho=`0.4136 / 0.4074`, Pair=`64.33% / 64.07%`, Top-Bottom=`1.8203R / 1.6832R`, NDCG@K=`0.7480 / 0.7338`, Top-K Lift=`1.7705 / 1.5789`. Breakout final Daily rho=`0.3821 / 0.3681` and HM/HS top decile=`24.06% / 23.63%`. Adjacent fold score-mean drift AO=`0.8551 pooled sigma / NO`, CC=`1.7264 pooled sigma / YES`. Results do not support CC replacing AO; no promotion or automatic current-config change.
- **Confirmed reporting comparability defect, NOT model result**: `hs_conditional_mfe_metrics()` currently constructs `p45_p55_pair` membership from `targets.low_adverse_safety_percentile` but binary `true_hs` from `targets.true_hs_mask`. `build_hs_conditional_mfe_targets()` preserves `true_hs_mask` on AO adverse-to-peak percentiles for both AO/CC, while `low_adverse_safety_percentile` is old Safety for AO and D40 Safety for CC. Consequently AO=`52.36%` and CC=`68.01%` P45-P55 Pair scores are evaluated on *different stock subsets* and cannot be interpreted as a paired boundary gain. Independent synthetic reproduction confirms original True-HS membership can match exactly while P45-P55 membership differs. Preserve original displayed values as historical evidence; repair must define and re-evaluate a shared boundary cohort from raw sidecars, without secretly modifying a frozen report or assuming RUN/REUSE automatically recomputes persisted metrics.
- **Research interpretation**: This is iterative 2021-2026 OOS, not untouched holdout. A 0.0051 rho delta is not statistical proof without paired date/block uncertainty and multi-seed evidence. Prefer no additional CC weight/threshold/epoch tuning on this OOS. Any A2 full-horizon True-HS relabeling or new joint-label design changes the scientific target and requires a separate authorized identity and Train/Validation-only model-selection protocol. Model truth, previously frozen AO artifacts, Standard SOP contract, Research menu, and Trading production remain unchanged.

Assessment: `USER_REPORTED_ROLLING_RESULT_AVAILABLE / FULL_HORIZON_MAE_SMALL_PAIRED_D40_GAIN / NOT_AO_REPLACEMENT / P45_P55_BOUNDARY_COMPARABILITY_DEFECT / NO_PROMOTION / NO_CONFIG_CHANGE`.

### 2026-10-14 -- MR-13CC | User-reported shared AO P45-P55 Rolling result (post-fix)

- **Evidence provenance:** User pasted the Research `[1] -> [4] Rolling OOS` report after the comparison-only extension fix and supplied `test-branch-1_20261014_133101_13729a74.zip` (SHA256 `95041db23c103f2d7c8201141737bb22b047597f2d062df18182d4d0bbbc2945`). All H/AH/AK/AO/BZ/CC model rows were `REUSE READY`. The ZIP contains no complete original OOS score sidecars, so these numerical results are user-reported formal-console output, not independently recalculated from raw scores in this review.
- **Same-cohort full-universe result:** The comparison-only `ao_safety_p45_p55_shared_cohort` extension reports identical AO/CC population size (`59,937` AO-band stock-days, `1,196` dates), `1,188` rankable dates, and `755,275` comparable pairs. The Shared P45-P55 Pair is AO=`52.36%`, CC=`52.30%`, CC-AO=`-0.06pp`. Both models use the original AO adverse-to-peak truth for the band and P50 class, filtered to the same OOS stock-day intersection; this is a *post-fit diagnostic*, not an executable prediction-time band.
- **Same-cohort Breakout result:** Official Breakout membership only filters the already-defined AO band; no percentile reranking occurs. Both models share `1,612` stock-days, `702` dates, `268` rankable dates, and `1,026` comparable pairs. Shared P45-P55 Pair AO=`53.56%`, CC=`54.29%` (`+0.73pp`). This small improvement, from clustered and sparse breakout pairs, is not material evidence of a better boundary classifier; no paired date-block uncertainty or multi-seed robustness was provided.
- **Existing paired context unchanged:** On 608,204 common Rolling stock-days and 1,203 dates, D40 Daily rho AO=`0.4570` vs CC=`0.4621` (`+0.0051`), D40 Pair `66.17%` vs `66.35%` (`+0.18pp`). Original adverse-to-peak Raw Safety Daily rho AO=`0.3601` vs CC=`0.3455`; overall ranking Daily rho `0.4136` vs `0.4074`; Top-K Lift `1.7705` vs `1.5789`; fold score drift AO=`0.8551 pooled sigma / NO` vs CC=`1.7264 / YES`. The old per-model HS row CC=`68.01%` remains historical **non-paired** evidence and is not the new comparable result.
- **Code/contract check (separate from empirical score re-evaluation):** Inspected `services/breakout_quality/full_horizon_mae_comparison.py` and canonical `daily_rank_metrics`: original-AO percentile is computed on full mature daily-universal truth before score intersection and Breakout filtering, and pairwise concordance uses identical AO P50 labels. Ran only the isolated independent capability test `tools.validate.breakout_quality_full_horizon_mae_extension_cases.case()` -> PASS. No `apps/run_bundle.py`, `apps/test_suite.py`, original score reconstruction, new training, or new comparison run was performed by GPT.
- **Assessment and limits:** `USER_REPORTED_SHARED_AO_BOUNDARY_ROLLING_RESULT_AVAILABLE / MATERIAL_SHARED_BOUNDARY_GAIN_NOT_SUPPORTED / SMALL_D40_TARGET_GAIN / AO_REPLACEMENT_NOT_SUPPORTED / NO_PROMOTION`. CC robustness/strategy transfer is **not recommended** on current evidence; no automatic termination of the current Training Model or model research identity. The user has not supplied a new Forward paired result in this turn, and repeated OOS is iterative research evidence rather than an untouched holdout. No changes to model, report schema, menu, formal tests, or Trading are authorized or implied.

### 2026-10-14 -- MR-13CD | User-reported six-fold Rolling OOS outcome (joint MFE40 x D40)

- **Evidence source and limitations:** User supplied the Research `[1] -> [4]` console transcript and `test-branch-1_20261014_161648_389e8dff.zip` (SHA256=`686e4270deeb16ec1132bca426a9fe83459cbe93409d84149505b990f3ec7b29`). The ZIP includes source and reduced market data, not the complete original 608,204-row canonical OOS score sidecar or comparison JSON. Results below are user-reported formal-console aggregate evidence, reviewed against the target, scoring, report, training and model-gate implementation; they have **not** been independently reproduced on the full sample. No Forward or cross-seed result was provided in this turn. The 2021-2026 period is iterative OOS evidence, not an untouched holdout.
- **Execution:** Rolling model `MR-13CD`, 6 annual folds, score period 2021-01-01 through 2026-03-02, all six new fits, PIT score coverage `630,589 / 630,589 = 100.00%`, duration `37:36.7`. All six folds selected epoch `1` (patience config is `1`); printed `val rho` (`0.2640 / 0.0459 / 0.1184 / 0.0938 / 0.2282 / 0.3384`) is legacy MFE-relative descriptive ranking, **not** the categorical CE objective used to choose epoch. Individual per-epoch CE and Brier histories were not included; do not diagnose the cause of epoch-one selection from the summary alone.
- **Proper scores and event ranking (joint event = high MFE40 / low full-horizon D40 on same-date P50):** Full Rolling OOS `608,204` stock-days, `1,203` dates, 4-class CE=`1.2991`, joint-event Brier=`0.2224`, independence-projection Brier=`0.2226`, actual class-3 rate=`29.43%`, mean predicted=`28.64%`, top predicted 10% class-3 hit=`35.15%` (`+5.72pp` vs full-universe event frequency). Breakout slice `17,346` stock-days, CE=`1.2594`, joint Brier=`0.2265`, independence projection=`0.2271`, event rate=`30.70%`, mean prediction=`27.96%`, top10 hit=`36.49%` (`+5.79pp` vs slice frequency). Independence-projection is derived from the **same model's marginals**, not separately fitted marginal models; the small Brier gaps (`0.0002` and `0.0006`) do not establish material dependence learning.
- **Descriptive, not deployable, baseline diagnostic:** Using the observed OOS class-3 rate itself as a hindsight constant predictor gives binary event Brier `p*(1-p)` ~= `0.2077` (full) / `0.2128` (Breakout), numerically below CD's `0.2224 / 0.2265`. These are OOS-fitted oracle-constant diagnostics, **not valid train-frozen model controls and must not enter fitting or selection**. To judge probabilistic gain, a separately frozen train-fold or train/validation climatology, with the same ticker/date and fold sample weighting, is needed. Uniform four-class CE log(4) is likewise not a substitute for a lawful train-fold empirical class-frequency CE baseline.
- **Original Standard SOP MFE-control comparison:** On the original MFE/R outcome (not CD's categorical target), AO vs CD Rolling overall Daily rho=`0.4136 / 0.2152`, Pair=`64.33% / 57.67%`, Top10 MFE=`2.5680R / 1.5539R`, NDCG@K=`0.7480 / 0.5609`, Top-K lift=`1.7705 / 0.1796`; Breakout Daily rho=`0.3821 / 0.1697`, NDCG@K=`0.8117 / 0.7281`. The same Standard SOP Top10 Low-Adverse=`-0.5973R / -0.3886R` refers to original adverse-to-peak, **not** full D40. CD's Top10 original HM/HS (`25.02%`) vs AO (`21.79%`) also uses original AO truth and is NOT the class-3 event rate. Thus CD has substantially weaker old MFE ranking, but the report does not give AO's top-decile hit under the new **common class-3 truth**; a valid direct cross-model joint-event ranking comparison remains missing.
- **Gate semantics:** `derive_point_in_time_model_validation_gate` checks positive original primary/MFE global and daily Spearman plus majority positive years and spread. It does not test categorical proper scores, class-3 event skill against a train-frozen baseline, or reference-model superiority. Therefore local `Model Gate PASS` is only legacy minimal ranking viability, not scientific acceptance of CD's joint-probability objective; `drift=YES` (`1.0744 pooled sigma`) remains a warning. The model is not authorized for Trading or promoted.
- **Research assessment:** `USER_REPORTED_ROLLING_RESULT_AVAILABLE / PROPER_SCORE_GAIN_NOT_ESTABLISHED / AO_REPLACEMENT_NOT_SUPPORTED_ON_MFE_REFERENCE / DEPENDENCE_GAIN_MARGINAL / FROZEN_BENCHMARK_AND_SAME_TRUTH_COMPARISON_NEEDED / NO_AUTO_CLOSE / NO_ROBUSTNESS_OR_PROMOTION_DECISION`. Do not OOS-tune event thresholds, loss or epoch selection. Keep existing AO/CC/CD identities, model compare list, source code, Standard SOP and current training settings unchanged; no research decision to start a new MR or replace CD current config has been authorized.

### 2026-10-14 -- MR-13CD | User-reported real Rolling same-event ranking, Train-only proper scoring and calibration

- **Evidence source / reproducibility:** User supplied Research `[1] -> [4]` complete Rolling console transcript (`貼上的文字 (1)(20261014-110455).txt`) and source ZIP `test-branch-1_20261014_190108_32bdbaf7.zip` (SHA256 `78d7898691b442bb995291d580631236835ad1af4ec8d63c91dc712ed89f2546`). All seven models reported `REUSE READY`; the new `joint_mfe_d40_incremental_skill_v1` comparison rendered in the official menu. The ZIP contains source/reduced data but no complete full-market AO/CC/CD score sidecars; numbers below are **user-reported formal-console evidence**, with GPT independently inspecting code contracts and arithmetically recomputing weighted statistics from displayed rounded fold rows, **not** independently rerunning the 608,204-stock-day OOS. This same 2021-2025 OOS has been repeatedly studied and remains **iterative research OOS, not untouched holdout**. No Forward evidence or cross-seed evidence supplied.
- **Same truth / matched selection:** Class 3 is same-date P50 **high MFE40 AND low full-horizon D40**; all models are evaluated on identical matured ticker/date/group-index rows, not original AO adverse-to-peak HM/HS truth. Rolling matched sample=`608,204` stock-days / `1,203` dates, top-decile selected N=`61,342`, observed base event=`29.43%`: AO=`24.13%` hit, CC=`24.30%`, CD=`35.15%`; CD minus AO=`+11.02pp` (displayed `vs original`=`+11.03pp` due to per-row rounding). Breakout matched sample=`17,346` stock-days / `1,176` dates, top-decile N=`1,787`, base event=`30.70%`: AO=`25.85%`, CC=`25.69%`, CD=`36.49%` (CD minus AO=`+10.64pp`, displayed `+10.63pp` by unrounded computation). Rankings use each model's own **different scientific score objective** on identical *Class-3 evaluation truth*; this proves task-specific selection advantage on evaluated data, **not** superiority on original AO MFE/economic performance.
- **Across-fold top-decile uplift vs AO:** five mature evaluable OOS folds 2021/2022/2023/2024/2025, displayed CD-minus-AO=`+17.75/+10.78/+15.10/+7.18/+4.99pp`; Breakout=`+13.00/+12.27/+19.56/+0.27/+6.80pp`. All five full-universe folds positive, but uplift attenuates late in the window; Breakout 2024 advantage is only +0.27pp. Six PIT folds were trained including 2026, but only five mature-evidence folds appear in the new per-fold quality table. Per-date and per-fold matched score samples are not independent trading returns; statistical/financial superiority is **not** established by the point estimates alone.
- **Legal train-frozen proper scores, all-market:** per-fold prior generated from canonical inner-train mature Class 0..3 labels, with frozen splitter and manifest count checks; no validation/OOS class frequency used to fit the constant prior. CD four-class CE=`1.2991`; stock-day-weighted Train-only four-class CE about `1.3745` (about `5.49%` lower loss). **Every** evaluable fold has lower four-class CE than frozen Train prior; 2025's gain is small (`1.3677` vs `1.3726`). Binary **Class-3-only** Brier CD=`0.2224` vs weighted Train-only `~0.2081`; pooled skill `~−6.85%`. Yearly class-3 Brier Skill 2021=`+2.97%`, 2022=`−20.35%`, 2023=`+1.25%`, 2024=`−10.48%`, 2025=`−6.98%`: two positive, three negative. Four-class CE and Class-3 Brier assess **different** proper-scoring objectives; CE improvement does not imply calibrated Class-3 event probability.
- **Legal train-frozen proper scores, Breakout:** CD four-class CE=`1.2594` vs weighted Train-only `~1.3890` (`~9.33%` lower); Class-3 Brier=`0.2265` vs weighted Train-only `~0.2129`, skill=`~−6.43%`; fold-wise Brier Skill=`+2.32/−23.03/+1.58/−15.12/−6.46%` in 2021..2025. Breakout scope only filters canonical eligible candidates and **does not redefine same-date percentiles**.
- **Calibration defect:** Full-market fixed-bin expected absolute calibration error (ECE), recomputed from rounded 9 displayed nonempty bins, `~9.40pp`; 0-10% predicted bin N=`78,664` has mean predicted=`4.90%` versus observed=`23.54%` (under-predict by `18.65pp`); 70-80% bin N=`6,050` has mean predicted=`73.32%` versus observed=`33.40%` (over-predict by `39.92pp`). Event incidence flattens near ~33% in the 30-80% predicted range, undermining interpretation of softmax event scores as calibrated probabilities. Per-fold ECE=`2.41/17.06/3.92/13.54/9.95pp`; years 2022 and 2024 are particularly unstable. These OOS observations are solely **post-fit diagnostics**; no thresholds, calibration mapping, temperature, loss, weights, selection or hyperparameters may be fitted with OOS data or statistics.
- **Original MFE/economic trade-off remains:** CD's original-MFE OOS Daily rho=`0.2152` vs AO=`0.4136`; Top10 MFE=`1.5539R` vs `2.5680R`; NDCG@10=`0.5609` vs `0.7480`. Original-report Low-Adverse is adverse-to-earliest-peak, **not** full D40. Better Class-3 Top10 selection therefore **does not authorize** claims of better realized portfolio/Trading returns, full-horizon loss containment in the selected trades, or risk-adjusted utility; those require their own fair, preregistered evaluation.
- **Science assessment / governance:** `SAME_TRUTH_CLASS3_RANKING_GAIN_REPORTED / FOUR_CLASS_CE_GAIN_VS_LEGAL_TRAIN_PRIOR / CLASS3_BRIER_SKILL_NEGATIVE / SEVERE_PROBABILITY_MISCALIBRATION / AO_MFE_RANKING_REFERENCE_STILL_STRONGER / ITERATIVE_OOS / RESEARCH_ONLY / NO_AUTOMATIC_PROMOTION_OR_CLOSE`. No new MR identity, scientific settings, Research `[1]` menu, model checkpoints, PIT producer, Standard SOP, approved report fingerprints or Trading behavior changed. Practical next hypothesis (not authorized for automatic implementation): a separately declared PIT-legal Train/Validation-only probability calibration or multi-objective rank-plus-event training experiment, with frozen fitting semantics and untouched OOS fitting prohibition; all decisions must remain explicit and cannot use the current OOS bins to choose thresholds.

### 2026-10-14 -- MR-13CE | User-reported completed Rolling OOS, direct signed U40 Huber model-gate FAIL

- **Evidence and scope:** User-supplied official Research `[1] -> [4]` six-fold console transcript `貼上的文字 (1)(20261014-144358).txt` and latest source ZIP `test-branch-1_20261014_213430_e00ddbcd.zip` (SHA256=`4dba7760481d44221dd041e989ac4b77b0152dfff039a06f876d15957460d969`). Source extracted afresh to `/mnt/data/stock_MR13CE_20261014_213430_review`; per project contract `PROJECT_SETTINGS`, Experiment Registry and Experiment Log read before analysis. User's full original 608,204-stock-day PIT score sidecar and model checkpoints were **not** included; the numeric evaluation below is console evidence, **not GPT-recomputed full-market OOS**. Repeated 2021-2025 OOS is iterative research evidence, not an untouched holdout.
- **Completed six folds:** fit 6 new fold checkpoints, no reused fits, 630,589 / 630,589 PIT scores, 2021-01-01 through 2026-03-02, elapsed `58:51.2`, selected epoch sequence `2/1/1/2/1/1`. Real selection-stage, pre-refit independent Validation daily rho by fold `0.0082, -0.0161, 0.0254, -0.0142, 0.0120, -0.0001`. The displayed Rolling Standard SOP `validation` Daily rho=`0.5957` is **after the final model was refit on train plus original validation samples**: `point_in_time_scores._train_fold` calls `select_epoch(train_ids, validation_ids)`, then `fit_final(final_ids)`, then scores `validation_ids` using the refitted model; `point_in_time_audit` concatenates the resulting `validation_scores.csv`. This is an in-sample descriptive statistic, **not the independent epoch-selection Validation score**, and its delta against OOS cannot alone establish independent validation-to-OOS generalization. No proof of OOS contamination from this behavior; historical Standard SOP report schema is frozen and unchanged.
- **Actual mature OOS:** 608,204 legal stock-days; target is signed `U40=MFE40-D40`, with 40-day full-horizon R scaling. Primary daily/global Spearman=`0.0002 / -0.0182`; Pair=`50.03%`, mean daily Top10 minus Bottom10 **U40**=`-0.3548R`. Breakout subset 17,346 stock-days daily/global rho=`-0.0109 / -0.0722`, Pair=`49.55%`, Top-Bottom U40=`-0.9020R`, Top-K Lift=`-0.1516R`. Five evaluable mature years 2021-2025: primary yearly rho positive `4/5` but only `2/5` positive Top-Bottom; drift=`YES` (`1.3303 pooled sigma`). Model Gate=`FAIL`; **no strategy performance test or Trading promotion authorized**. Sixth fold includes 2026 score coverage but contributes no extra evaluable mature annual quality evidence.
- **Interpreting other reported numbers:** CE own raw-U40 Top10 Target=`0.4012R`, Bottom10=`0.7560R` on OOS, consistent with negative spread; post-refit in-sample Validation Top-Bottom=`+5.2685R` is not a held-out control. CE Top10 original MFE=`1.3982R` versus AO=`2.5680R`, although this is not a full matched-U40 test. AO's main Standard SOP Target uses its original scientific objective, so direct comparison between AO's `Top10 Target` and CE's `Top10 Target` would be **different-truth** and scientifically invalid. Original `Low-Adverse` metric is pre-peak adverse, not full D40.
- **Code-contract review, without changing scientific semantics:** `compute_daily_opportunity_target_batch` correctly subtracts the full 40-bar D40/R and retains negative U40. Raw-R training uses the signed two-logit `PASS−REJECT` margin with Huber delta=1R; validation epoch selection prioritizes holdout Huber loss, then rho, and inference returns that signed margin. Point-in-time selection/final refit separation follows existing canonical behavior. The dedicated direct-U40 target/CPU Huber regression synthetic, signed PIT/Forward producer/reader/audit regression and approved persistent report contract fingerprints pass. Nothing in the provided source proves a wrong target, scoring-domain regression or PIT future-label leak sufficient to explain the observed OOS collapse; poor generalization of this Huber-fit **is directly observed**, whereas causal attribution among target noise, regression loss, distribution shifts or model capacity requires controlled evidence.
- **Scientific disposition:** `ROLLING_RESULT_AVAILABLE / DIRECT_U40_HUBER_OOS_SORTING_FAIL / MODEL_GATE_FAIL / RESEARCH_ONLY / NOT_PROMOTED`. Preserve model identity/checkpoint/artifact fingerprints and current H/AH/AK/AO/CE compare config unchanged. This is a failure of **this single U40 numerical Huber learning formulation**, not proof that U40 itself has no forecastability. Minimal next read-only decision evidence: score the frozen AO and CE rankings against the same matured true U40 on the identical PIT stock-days/dates, reporting only Top10 U40, MFE40, D40 and paired per-year differences. That result can decide whether the U40 target is useful under an existing ranking model versus needing a new explicitly authorized rank-oriented U40 objective. Do **not** alter CE Loss, validation, fitted artifacts or permanent Standard SOP after inspecting the current OOS evidence; any new training semantics need a fresh registered MR identity and user research decision.

### 2026-10-15 -- MR-13CF | User-reported six-fold Rolling U40 RankNet result and AO/CE/CF common-truth decision

- **Evidence and scope:** User provided the official Research `[1] -> [4] Rolling OOS` console output `貼上的文字 (1)(20261015-105649).txt` and source ZIP `test-branch-1_20261015_102503_fa4c3553.zip` (SHA256 `99b6e4c8224fd03bf6b211db41898bbb3f6678b1ace2e143fbf8a896108905be`). This ZIP contains source and governance docs but **not** the full market OOS score/checkpoint sidecars, so numerical results below are **user-reported official-console evidence**, not independently recomputed from the 608,204 individual scored rows. GPT inspected canonical U40 producer, MR-13CF/CE scientific specs, matching score reader and approved report logic; U40 synthetic contract checks passed in isolation. **No proof of full Windows `apps/run_bundle.py` formal PASS is provided in this turn.** Already repeatedly used 2021–2025 OOS is **iterative research OOS**, not an untouched final holdout.
- **Six-fold PIT completeness and independent selection evidence:** Annual folds 2021-01-01 to 2026-03-02; total scored groups=`630,589`, actual mature evaluable OOS groups=`608,204` across 2021–2025 and `1,203` stock-trading dates. Six fits built, no checkpoint reuses, all six chose epoch 1. Independent epoch-selection validation daily rho by fold=`+0.0661/−0.0068/+0.0473/+0.0478/+0.1025/+0.0501` (2021..2026 folds). Separately reported combined post-refit descriptive Validation rho=`0.1144`; it is **not** an independent validation of the final refitted checkpoints and must not be substituted for fold-specific selection evidence.
- **MR-13CF own-target Rolling OOS:** All-stock U40 Daily rho=`0.0515`, global rho=`0.0221`, Pair=`51.76%`, Top10/Bottom10 target=`0.7180R/0.3201R`, spread=`+0.3979R`; Breakout slice N=`17,346`, daily rho=`0.0545`, global rho=`0.1123`, Pair=`52.21%`, Top-Bottom=`+1.0659R`; yearly daily rho positive only `2/5`, Top-Bottom positive only `2/5`, drift=`YES` (`1.6290` pooled sigma); **Model Gate FAIL**, no strategy performance authorization. The corresponding same-target CE-Huber control had full-market daily rho=`0.0002`, Pair=`50.03%`, Top-Bottom=`−0.3548R`, Gate FAIL. Ranking training package improves U40 learnability relative to numerical Huber in this controlled single-seed comparison, yet absolute robustness remains inadequate.
- **Primary matched mature U40 truth, all-stock Top10:** one common `ticker/date/group_index` and Rolling fold/cutoff cohort, N=`608,204` evaluable stock-days / `1,203` dates, identical `61,342` top-decile selections per model. AO actual `MFE40=2.5673R`, `D40=1.3394R`, `U40=1.2279R`; CE `MFE40=1.3985R`, `D40=0.8591R`, `U40=0.5394R`; CF `MFE40=1.6732R`, `D40=0.9474R`, `U40=0.7257R`. CF-minus-CE U40=`+0.1863R`, CF-minus-AO=`−0.5021R`; the relative disadvantage arises because CF's MFE vs AO falls `−0.8941R` while D40 improves `−0.3920R`. These are *observed future extrema*, not realized tradable PnL; their calendar order is not established by U40.
- **Breakout candidates, same truth:** matched N=`17,346` stock-days / 1,176 dates; only 603 days have sufficient candidate competition; identical per-model Top10 N=`1,739`. AO `MFE40=3.1221R`, `D40=1.3606R`, `U40=1.7615R`; CE `1.2402R/0.6029R/0.6373R`; CF `1.9951R/0.9110R/1.0841R`. CF-minus-CE=`+0.4468R`, CF-minus-AO=`−0.6774R`. Candidate restriction is report-only and does not alter universal model-fitting truth.
- **Annual stability on matched all-market Top10 U40:** 2021 AO/CE/CF=`1.5428/0.6778/0.9785R`; 2022=`0.2634/0.0925/0.0801R`; 2023=`1.7262/0.7744/1.2241R`; 2024=`1.1060/0.4258/0.5756R`; 2025=`1.5003/0.7259/0.7800R`. CF trails AO `5/5` mature evaluation years; CF beats CE `4/5` (except 2022). In Breakout, CF trails AO `5/5` years and CE is not consistently dominated. These are repeated within-cohort annual observations, not independent multi-seed experiments.
- **Science interpretation:** MR-13CE versus MR-13CF holds target, input architecture, seed and PIT schedule fixed, but swaps the *training objective/epoch-selection package* (Huber vs Full-list Delta-NDCG Pairwise); it **does not** isolate only the mathematical Loss, nor prove all pairwise learning is inferior. MR-13AO uses a materially different shared-Safety/Conditional-MFE architecture/objective, so AO-vs-CF is a **matched-outcome selection contrast**, not a single-factor architecture/Loss causal claim. Numerical U40 may still be useful as the evaluation objective even when direct learning fails. No new fitting hyperparameters, mixture weights or thresholds may be estimated from this observed OOS.
- **Decision / project state:** `MR13CF_ROLLING_6FOLD_AVAILABLE / CF_GAIN_VS_CE_BUT_NOT_AO / AO_TOP10_U40_GREATER_IN_ALL_5_YEARS / MODEL_GATE_FAIL / SINGLE_SEED / ITERATIVE_OOS / AO_REPLACEMENT_NOT_SUPPORTED / NO_PROMOTION / NO_TRADING`. Current Research training identity/config and historical AO/CE/CF artifacts and compare membership remain unchanged; no additional U40 exploratory training or report extension is authorized by this result alone. Next substantive research direction, if explicitly chosen, should preserve AO's learnable upside ranking while using an independently PIT-legal D40 risk representation rather than assuming a single directly trained U40 scalar must dominate.

### 2026-10-15 — MR-13CG | User-reported six-fold Rolling full-D40 Safety result versus frozen AO

- **Evidence provenance:** User-provided official Research `[1] → [4] Rolling OOS 模型比較` transcript `貼上的文字 (1)(20261015-122247).txt` and latest project ZIP `test-branch-1_20261015_193311_eb2eef4d.zip` (SHA256 `c17a90e3ba12ca88e27f1e1e2f48bc876e3249afdc147c6eeec5553ec6f16297`). Project settings, Experiment Registry and Log read before inspection. This source archive contains code and governance docs but **no full-market PIT score/checkpoint sidecars**. The numbers below come from the user's official Console, not an independent recomputation from scored rows, and no Windows formal `run_bundle.py` success evidence is supplied. Evaluated 2021–2025 OOS is **iterative research OOS**, not an untouched holdout.
- **Scientific control and maturity:** MR-13CG retains original AO 300×10 adjusted stock+0050 input, shared two-head InceptionTime, Full-list Delta-NDCG Pairwise, target MFE, Seed42, daily-universal sample, six yearly PIT folds and conditional-MFE-based inference. Compared with AO, **both the Safety head supervision AND the True-HS P50 Conditional-MFE sublist use full-horizon D40**; historical CC only switched Safety head. Complete scored PIT groups=`630,589/630,589` across six folds (`2021-01-01` to `2026-03-02`), six new fits, no reuse, all selected best epoch 1, elapsed=`43:09.0`. Five mature assessment years 2021–2025; the partial 2026 fold is not a sixth fully mature assessment year. Fold selection-stage `val rho` as printed was `0.6098/0.6299/0.5747/0.5846/0.6387/0.6621`; do not substitute any post-refit Standard-SOP validation aggregates for these independent selection diagnostics or presume identical metric populations without checking contract.
- **Own-objective Rolling quality / Gate:** CG all-market OOS (`608,204` mature stock-days) MFE-target Daily rho=`0.4044`, global rho=`0.3095`, Pair=`64.01%`, Top10–Bottom10 raw-MFE spread=`+1.8399R`, all `5/5` mature years positive by daily rho and spread; Breakout daily rho=`0.3881`, Pair=`65.35%`, spread=`+2.6832R`. Drift=`YES` (`1.0773 pooled σ`); **Model Gate PASS**, which authorizes the next research stage, not automatic strategy performance certification or Trading promotion. Relative to frozen AO, same-objective all-market Daily rho CG/AO=`0.4044/0.4136` (−0.0092), Pair=`64.01/64.33%` (−0.32pp); MFE NDCG@K=`0.7607/0.7480` (+0.0127). Such metrics are not direct U40 differences.
- **PRIMARY same-truth U40 all-market comparison:** Exact common mature ticker/date/group/fold/cutoff cohort N=`608,204`, dates=`1,203`, equal daily Top10 selection slots N=`61,342` per arm. AO actual `(MFE40,D40,U40)=(2.5673,1.3394,1.2279)R`; CG `(2.6469,1.3827,1.2642)R`. Hence CG−AO MFE=`+0.0796R`, actual D40=`+0.0433R` (worse risk), resulting U40=`+0.0363R` (~`+2.96%` relative to AO U40). This is a **small opportunity-minus-risk improvement**, *not* a drawdown reduction or realized tradable PnL. CE/CF on the same cohort remained U40=`0.5394R/0.7257R`.
- **Breakout subset:** Same exact cohort N=`17,346` stock-days, dates=`1,176`, rankable dates=`603`, matched Top10 N=`1,739`. AO `(MFE40,D40,U40)=(3.1221,1.3606,1.7615)R` vs CG `(3.1633,1.3708,1.7925)R`; CG−AO U40=`+0.0310R` from MFE=`+0.0412R` less D40=`+0.0102R` worse. The breakout candidate subset is an evaluation slice, not an alternate training universe.
- **Yearly stability of *matched* Top10 U40:** Full-market CG−AO by 2021..2025=`+0.0401/−0.0133/−0.0466/+0.0633/+0.1328R` (`3/5` positive). Breakout CG−AO=`+0.3044/+0.2375/−0.1244/+0.0662/−0.0949R` (`3/5` positive), and CG's 2022 Breakout U40=`+0.0219R` versus AO=`−0.2156R`; results are sensitive to year/market slice. Single Seed42 with six time folds does **not** establish multi-seed robustness or statistical significance; OOS has been repeatedly consulted in the research process. Any bootstrap/uncertainty estimate would require the underlying paired day-level OOS rows, unavailable in this source archive.
- **Risk-head evidence distinguished from ultimate selection:** Canonical common-truth full-horizon D40 head Daily rho AO=`0.4570`, CG=`0.4663` (`+0.0093`); Pair AO=`66.17%`, CG=`66.50%`; P50 low-MAE hit AO=`67.42%`, CG=`67.95%`. These measure D40 risk **head quality**, not the MFE-driven final Top10 portfolio; final Top10 actual D40 increased in CG. Original Standard-SOP `Top10 Low-Adverse` is adverse-to-peak, not full D40, and must not be substituted into U40.
- **Code-contract review and focused GPT tests:** Existing canonical `u40_paired_comparison.py` uses one matured U40 Producer, common ticker/date/group-index intersection, Rolling fold & information cutoff equivalence, equal per-date Top10 slots and independent AO/CG score ordering. Focused `breakout_quality_u40_paired_cases.py`, `breakout_quality_full_horizon_mae_primitive_cases.py`, and `breakout_quality_direct_u40_cases.py` passed in the extracted source. These tests establish engineering consistency, not market-effect confidence or Windows formal suite PASS. No model, checkpoint, loss, configured compare list or Trading changes.
- **Decision:** `MR13CG_ROLLING_6FOLD_RESULT_AVAILABLE / MODEL_GATE_PASS / MATCHED_U40_SMALL_POSITIVE_VS_AO / POSITIVE_3_OF_5_YEARS_IN_BOTH_SCOPES / SINGLE_SEED / ITERATIVE_OOS / NOT_PROVEN_SUPERIOR / RESEARCH_ONLY / NOT_PROMOTED / NO_TRADING`. Retain AO as frozen scientific reference, CG as a viable next-stage research candidate, and the current authorized Research menu/config as-is. Before an adoption decision, require pre-specified non-OOS-fitted stability evidence (e.g., officially authorized Rolling robustness plus identical downstream strategy-cash/execution comparison); avoid choosing risk weights, thresholds or new fitting hyperparameters from the observed iterative OOS improvement. No further one-off diagnostic is needed to establish that the current gain is small and inconsistent by year.

### 2026-10-15 — MR-13CH | User-reported 20D Rolling completion; 20D-versus-40D D40 Extension compatibility fix

- **User evidence and source:** The user uploaded official Research `[1] → [4]` output `貼上的文字 (1)(20261015-151602).txt` and latest source `test-branch-1_20261015_214831_c1bd23b9.zip` (SHA256 `277e27f0ea9ad743c95e31ef1d23e89fbea23c57ad796067541daa4fb84f574b`). The official console confirms CH six annual PIT folds covering `2021-01-01` through `2026-03-02`, groups `630,589/630,589`, 6 new fits, 0 reused, full coverage. Epoch selected per fold=`2/2/2/1/2/2`, independent selection-stage validation rho=`0.6364/0.6642/0.6292/0.6264/0.6654/0.6750`, elapsed=`01:13:56.0`. These are user-reported console results; the ZIP does not contain the full-market checkpoint/score files needed for independent reconstruction. The 2026 fold is partial and should not be described as a sixth fully mature evidence year.
- **20D own-target Rolling model evidence:** Evaluated OOS stock-days `608,204`, CH 20D daily rho=`0.4326`, global rho=`0.3399`, pairwise correctness=`65.03%`, Top10–Bottom10 target spread=`+1.2106R`; Breakout slice daily rho=`0.4096`, pair=`66.28%`, NDCG@K=`0.8235`; full-universe NDCG@K=`0.7678`; `5/5` positive-rho mature years, `5/5` positive-spread mature years, score drift=`NO`, **Model Gate PASS**. This passes the *model* threshold, not strategy/Trading validation. The labels, loss, metric and report describe a **20D target**; these cannot be read as evidence that CH dominates CG or AO on 40D U40, nor as an untested 20D-vs-40D economic comparison.
- **Comparison exception after successful fits:** Research `[1] → [4]` then raised `ValueError: D40 paired model comparison混入不同continuous target id` while building `full_horizon_mae_common_truth` before standard model comparison output. Canonical D40 extension previously selected all models having the same dual-head objective and a Raw Safety column; CH uses this objective but Raw Safety supervises **D20**, with a separate `daily_20bar_full_horizon_pure_mfe_r_v1` target. CH legitimately has a different target identity; this is not a PIT/model training error or missing trained checkpoint.
- **Scope-limited engineering repair:** The D40-only Extension now selects its anchor from registered full-horizon MAE-supervised models with the canonical **40D pure-MFE target**, then includes only models sharing that 40D target and training objective. It skips CH's D20 Raw Safety score instead of evaluating it as D40. Any mismatched `settings.continuous_target_id` versus validated score-contract identity remains fail-closed, as do same-target filter mismatches; only compatible 40D score frames are read. AO/CG retain paired D40/P45–P55 rows; CH remains in the user-authorized current H/AH/AK/AO/CG/CH Model Compare list and in the **separate** matched U20 and U40 utility Extensions. Model-specific extension contracts, Standard SOP six sections, approved fingerprints, U20/U40 cohort logic and scientific fitting identities are not changed.
- **GPT-side verification / remaining evidence:** The original D40 AO/CC Forward/Rolling synthetic contracts pass, plus new AO/CG/CH mixed-target Forward/Rolling cases showing correct CH exclusion, D40 anchor selection, and malformed CH contract fail-fast. Existing U40 paired and D40 primitive tests and approved report fingerprints pass; Continuous Ranker capability cases `67/67`, PIT Builder `33/33`, Research reverse-conditional route `16/16`, and syntax compilation of `642` Python sources all pass independently. **No Windows formal run_bundle PASS or actual AO/CG/CH Top10 U20/U40 numbers are claimed.** Re-run Research `[1] → [4]` using existing scored model artifacts to generate those results without retraining CH.
- **Decision:** `CH_ROLLING_6FOLD_REPORTED / MODEL_GATE_PASS / D40_ONLY_COMPARE_MIXED_TARGET_BUG_FIXED / AO_CG_CH_SAME_TRUTH_U20_U40_PENDING / NO_PROMOTION / RESEARCH_ONLY / NOT_TRADING`. Evaluation OOS remains iterative research evidence, not untouched holdout.

### 2026-10-15 — MR-13CH | Full AO/CG/CH matched U20 and U40 Rolling result, after D40 extension fix

- **Evidence and provenance:** User supplied official Research `[1] → [4] Rolling OOS 模型比較` console `貼上的文字 (1)(20261015-160231).txt` plus latest project source `test-branch-1_20261015_233308_658ec740.zip` (SHA256 `56a4eff6d553eabeeb10db96112e19c546c2204440aaab01573bfadc4083e997`). H/AH/AK/AO/CG/CH each reported `REUSE READY`; no newly selected epoch, PIT score rebuild or model retraining is reported in this comparison run. The ZIP is source/governance evidence without the complete 608,204-row OOS score/checkpoint sidecars. Values below come from **user-reported official Console**, not an independently recomputed actual full-market evaluation. No Windows formal `apps/run_bundle.py` PASS is claimed.
- **Exact comparison cohort / horizon separation:** All-market 2021–2025 has `608,204` common mature stock-days, `1,203` dates and `61,342` identical Top10 quotas per model. Breakout has `17,346` common stock-days / `1,176` dates, of which `603` are rankable (enough competing names), with `1,739` identical Top10 quotas. AO/CG/CH are ranked by their separately fitted models on identical ticker/date/group/fold/information-cutoff cohorts. The U20 report reads canonical CH20-truth; the U40 report reads canonical AO40-truth; `U20=MFE20−D20` and `U40=MFE40−D40`, all in R, use independent horizon-specific truth contracts. Both evaluations keep **40D-mature stock-day eligibility**, so this experiment does **not** test earlier 20D label availability or faster retraining.
- **All-market matched Top10 (AO / CG / CH):** U20=`0.6720 / 0.6779 / 0.6985R`; MFE20=`1.6720 / 1.7106 / 1.7296R`; D20=`1.0000 / 1.0327 / 1.0311R`. U40=`1.2279 / 1.2642 / 1.2848R`; MFE40=`2.5673 / 2.6469 / 2.6644R`; D40=`1.3394 / 1.3827 / 1.3796R`. CH vs CG: U20=`+0.0206R` and U40=`+0.0206R` (~`+1.63%` relative U40), MFE40=`+0.0175R`, D40=`−0.0031R` (lower D40 is preferable). CH vs AO: U20=`+0.0265R`, U40=`+0.0569R`. Only observed future extreme-based utility is measured, not executable/realized strategy PnL.
- **Breakout matched Top10 (AO / CG / CH):** U20=`1.0292 / 1.0830 / 1.0489R`, MFE20=`2.0530 / 2.1084 / 2.0914R`, D20=`1.0238 / 1.0253 / 1.0425R`. U40=`1.7615 / 1.7925 / 1.7534R`; MFE40=`3.1221 / 3.1633 / 3.1470R`; D40=`1.3606 / 1.3708 / 1.3936R`. CH vs CG: U20=`−0.0341R`, U40=`−0.0391R`, MFE40=`−0.0163R`, D40=`+0.0228R`, i.e. CH is worse on both realized upside and full-horizon adverse excursion. CH also trails AO U40 in Breakout by about `−0.0082R` (Console based on unrounded values).
- **Mature-year U40 CH-minus-CG consistency:** All-market 2021=`−0.0038R`, 2022=`+0.0498R`, 2023=`+0.0399R`, 2024=`+0.0067R`, 2025=`+0.0107R`; CH beats CG in `4/5` mature years, but gains are small. Breakout 2021=`−0.2063R`, 2022=`−0.2476R`, 2023=`+0.0472R`, 2024=`−0.0107R`, 2025=`+0.0362R`; CH beats CG in only `2/5` years. U20 CH-vs-CG likewise positive `4/5` all-market and `2/5` Breakout years. Annual means share the same iterative research OOS and are **not independent repeated-seed confirmations**.
- **Model-quality context and reporting caveat:** CH's already verified *20D own-target* six-fold Rolling Gate remains `PASS`: Daily rho=`0.4326`, Pair=`65.03%`, five mature positive-rho/spread years, score drift=`NO`. CG's reported `0.4044` Daily rho relates to the *40D target*; cross-horizon Daily rho values and original Standard SOP Target columns cannot be treated as directly comparable target truth. CH model Gate PASS does not establish a superior strategy or positive realized broker-cash payoff.
- **Decision and authorization:** `MR13CH_ROLLING_6FOLD_MODEL_GATE_PASS / MATCHED_U20_U40_AVAILABLE / ALL_MARKET_SMALL_GAIN_VS_CG / BREAKOUT_NO_GAIN_VS_CG / SINGLE_SEED42 / ITERATIVE_OOS / FORWARD_NOT_SUPPLIED / ROBUSTNESS_PENDING / NO_PROMOTION / NO_TRADING`. Retain both CG and CH as Research candidates, keep existing Research `[1]` training identity and H/AH/AK/AO/CG/CH Compare/Test membership unchanged. No OOS-derived optimization of risk weights/horizon/threshold or post hoc winner promotion. Confirm multi-seed repeatability and, only if warranted, identical-rule strategy economics before any replacement decision. This report is sufficient to characterize this specific 20D-versus-40D Seed42 contrast; further Audit is not required absent a concrete blocker.
- **GPT-side scope verification:** Current U20/U40 paired comparison producer is read-only and preserves independently frozen 20D/40D target contracts, shared matured cohort, matching OOS fold identities and Top10 quotas. Dedicated paired U20/U40 synthetic Forward/Rolling cases and full-horizon D40 controlled cases PASS. This turn only records official evidence; no model, loss, PIT, training, menu, Standard SOP, strategy or Trading code is changed.

### 2026-10-16 — Research [1] Rolling OOS Robustness | H/AH/AK/AO/CG/CH four-seed reported results

- **Source / boundary:** User supplied `貼上的文字 (1)(20261016-054843).txt`, displaying official Research `[1] → [6]` Rolling OOS Robustness Standard Model SOP Comparison, and accompanying `test-branch-1_20261016_001202_2c1e69e1(2).zip` (SHA256 `5655d8b6f2c2785f0acf57fa53f0e50e3891183a08547b73271b8f73afe2051d`). The file is a source/governance ZIP; it does **not** contain the full local `models/` or `outputs/` PIT score/checkpoint sidecars. Numbers below are **user-provided console results**, not independently recomputed seed-level scores or a Windows formal `apps/run_bundle.py` PASS. Four existing configured benchmark seeds from the earlier user console: `693545351 / 2014432738 / 104504529 / 762252050`. Research `MR-13CH` remains current training profile; current compare H/AH/AK/AO/CG/CH, frozen historical fits unchanged.
- **Aggregator semantics:** Standard SOP values shown in the Robustness comparison are arithmetic **row means across four per-seed model SOPs**; the Across-seed Stability σ uses the **population** standard deviation (`ddof=0`) of each seed's OOS metric. Rolling aggregate drift is a conservative **any-seed** flag, so `YES` for six models does not mean every seed drifted. The source prints `608,204` full market and `17,346` Breakout sample groups in the common OOS report; without full sidecars, per-seed row pairing and per-seed U40 cannot be reconstructed here.

| Model | Target | Daily rho mean | rho σ | Pair mean | Pair σ | Top-Bottom mean (R) | Top-Bottom σ (R) |
|---|---|---:|---:|---:|---:|---:|---:|
| MR-13H | full-horizon no-breach | 0.1996 | 0.0238 | 56.88% | 0.86pp | 1.1525 | 0.1720 |
| MR-13AH | MFE40 | 0.3221 | 0.0286 | 61.02% | 1.06pp | 1.2720 | 0.2438 |
| MR-13AK | MFE40 | 0.3852 | 0.0190 | 63.30% | 0.73pp | 1.4709 | 0.3400 |
| MR-13AO | MFE40 | 0.3978 | 0.0069 | 63.78% | 0.22pp | 1.6839 | 0.0865 |
| MR-13CG | MFE40 | 0.3918 | 0.0045 | 63.49% | 0.18pp | 1.7434 | 0.1519 |
| MR-13CH | **MFE20** | 0.4296 | 0.0033 | 64.89% | 0.11pp | 1.2705 | 0.0435 |

- **Same-target AO/CG controlled reading:** AO versus CG, both MFE40 self-target, Rolling Daily rho=`0.3978 vs 0.3918` (CG−AO=`−0.0060`), Pair=`63.78% vs 63.49%` (`−0.29pp`), Top-Bottom=`1.6839R vs 1.7434R` (`+0.0595R`); NDCG@K=`0.7422 vs 0.7286`, Top-K Lift=`1.7609 vs 1.6629`. Breakout Daily rho=`0.3803 vs 0.3685`, Breakout Top-K Lift=`0.6351 vs 0.6155`. CG shows less rho dispersion (`0.0045 vs 0.0069`) but does not consistently dominate AO in 40D MFE ranking or Breakout model ranking. Robustness Top10 MFE=`2.5699R vs 2.5522R`; Top10 Low-Adverse=`−0.5947R vs −0.6012R` uses the existing **Low-Adverse SOP metric**, and must **not be relabeled D40**. The earlier matched Seed42 U40 CG−AO=`+0.0363R` remains valid iterative research evidence but its sign/repeatability **cannot** be inferred from these rows.
- **CH distinct horizon:** CH four-seed own-**MFE20** Daily rho=`0.4296` (σ=`0.0033`) and Pair=`64.89%` indicate cross-seed 20D model-quality consistency. AO/CG are own-**MFE40**; their Daily rho, Top-Bottom and Top10 target metrics are **not direct cross-horizon economic comparisons**. CH's earlier Seed42 matched same-truth U40 (`CH 1.2848R`, `CG 1.2642R`, `AO 1.2279R` all market; Breakout `CH 1.7534R`, `CG 1.7925R`, `AO 1.7615R`) remains **single-seed only**, notwithstanding CH's four-seed 20D self-target stability.
- **Extension limitations:** Robustness table explicitly prints `MFE40−D40 / Paired AO vs Direct U40`, `MFE20−D20 / Same 40D-mature Cohort`, `AO Safety P45–P55 / Paired Common Cohort` and `Full-Horizon MAE / Paired Common Truth` as **`Single-seed only / NOT AGGREGATED`**; only Multi-head Learnability and HS Qualification/Conditional-MFE Quality are aggregated by arithmetic row mean. Conditional-MFE across-model rho (`AO 0.3936`, `CG 0.6182`, `CH 0.6591`) uses different True-HS cohort and, for CH, different horizon, and **does not answer** the pending fixed-cohort AO/CG 2×2 diagnostic. The supplied console is a Robustness report, **not** the AO/CG True-HS crossover Audit result.
- **Decision / status:** `USER_REPORTED_FOUR_SEED_ROLLING_MODEL_ROBUSTNESS_AVAILABLE / SAME_TARGET_AO_VS_CG_MIXED_WITH_AO_RANKING_EDGE / CH_20D_SELF_TARGET_CROSS_SEED_CONSISTENT / SHARED_U20_U40_NOT_AGGREGATED / ITERATIVE_OOS / NO_PROMOTION / NO_TRADING`. Retain H/AH/AK/AO/CG/CH current Research list and CH current training identity; do **not** auto-select a seed or make a strategy/Trading replacement decision. Remaining decisive economic question is the **per-seed matched AO/CG/CH common-truth U40/U20 Top10 differences**, computed read-only from existing legitimate per-seed scores if available, with the AO/CG cross-True-HS diagnostic still separate and not observed in this submission. No training, Standard SOP contract, user menu, scoring/runtime code, or Trading change in this documentation-only evidence update.

### 2026-10-16 — AUD-ao-cg-ch-four-seed-common-u40 | Real four-seed same-truth U40 result and decision

- **Source and artifact validation:** User ran `tools/audit/breakout_quality/ao_cg_ch_multiseed_u40.py` on Windows and supplied the complete `result.md` console output. Working source/governance baseline is `test-branch-1_20261016_141722_a8bb6fb7.zip`, SHA256 `4bc6e130d4f5b289adb9ce26c6cfadfbe8955bdf4ec07cf57e0bf119b2a1ac3f`. Four configured benchmark seeds `693545351 / 2014432738 / 104504529 / 762252050` each reported official `READY/PIT SHA/folds` checks and exactly `608,204` same 40D-mature stock-days. This turn treats the user-provided local Audit as results evidence; source ZIP does **not** carry physical Windows scores/checkpoints, so GPT has independently checked printed arithmetic only, not rescored full PIT sidecars or claimed a local Windows formal-suite PASS.
- **Metric contract:** Every model/seed ranks the same aligned stock-days using identical daily Top10 quotas; all evaluations use canonical 40D matured `U40=MFE40-D40` truth, not the model's different training horizon. Report `mean±σ` is across four model-seed U40 values; paired contrasts report the **mean of same-seed differences**, population σ (`ddof=0`), and winning seeds. Reports are iterative research OOS proxies, **not realized broker-cash trades**, and the official Standard SOP/Robustness extension remains unchanged (`single-seed-only` for its persistent U40 extension).

| Slice | Model | Top10 U40 mean ± seed σ (R) | MFE40 mean (R) | D40 mean (R) |
|---|---|---:|---:|---:|
| All market | AO | 1.2306 ± 0.0388 | 2.5693 | 1.3388 |
| All market | CG | 1.1764 ± 0.0799 | 2.5517 | 1.3753 |
| All market | CH | 1.2321 ± 0.0239 | 2.6242 | 1.3921 |
| Breakout | AO | 1.6995 ± 0.0617 | 3.0474 | 1.3478 |
| Breakout | CG | 1.6694 ± 0.0610 | 3.0469 | 1.3776 |
| Breakout | CH | 1.7659 ± 0.0300 | 3.1591 | 1.3931 |

| Slice | Paired contrast | Mean ΔU40 (R) | σ of per-seed Δ (R) | Winning seeds | Seed-wise ΔU40 (R), configured seed order |
|---|---|---:|---:|---:|---|
| All market | CG−AO | -0.0542 | 0.0787 | 1/4 | +0.0411, -0.1777, -0.0426, -0.0375 |
| All market | CH−AO | +0.0015 | 0.0305 | 2/4 | +0.0159, -0.0427, +0.0402, -0.0074 |
| All market | CH−CG | +0.0557 | 0.0597 | 3/4 | -0.0252, +0.1351, +0.0828, +0.0301 |
| Breakout | CG−AO | -0.0302 | 0.1094 | 2/4 | +0.1144, -0.1768, -0.0814, +0.0231 |
| Breakout | CH−AO | +0.0664 | 0.0799 | 3/4 | +0.1016, -0.0477, +0.0420, +0.1697 |
| Breakout | CH−CG | +0.0966 | 0.0637 | 3/4 | -0.0128, +0.1290, +0.1234, +0.1466 |

- **Year × seed evidence:** All-market CG−AO year mean ΔU40 (winning seeds): 2021 `−0.0158 (1/4)`, 2022 `−0.0154 (1/4)`, 2023 `−0.0582 (1/4)`, 2024 `−0.0121 (1/4)`, 2025 `−0.1651 (1/4)`; CG underperforms AO on the across-seed year mean in **5/5 years**. All-market CH−AO: 2021 `−0.0073 (2/4)`, 2022 `+0.0120 (1/4)`, 2023 `−0.0212 (2/4)`, 2024 `+0.0009 (2/4)`, 2025 `+0.0222 (3/4)`: near tie, not consistent gains. Breakout CH−AO: 2021 `+0.1776 (3/4)`, 2022 `−0.0520 (0/4)`, 2023 `+0.0119 (3/4)`, 2024 `+0.0467 (3/4)`, 2025 `+0.1053 (3/4)`; positive year means in 4/5 but 2022 loses in **all four seeds**. Breakout CH−CG: 2021 `+0.0953 (4/4)`, 2022 `−0.0961 (1/4)`, 2023 `+0.0448 (4/4)`, 2024 `+0.0872 (3/4)`, 2025 `+0.2322 (3/4)`.
- **Attribution:** CH Breakout U40 versus AO `+0.0664R` comes from `+0.1117R` higher MFE40 offset by `+0.0453R` **higher (worse) D40**; it does not establish improved downside protection. Across-seed CH Breakout U40 σ (`0.0300R`) is below AO's (`0.0617R`), but four common-seed replicates and repeatedly inspected overlapping 2021–2025 regimes do not establish significance or independent out-of-sample validation. Four-seed U20 same-truth economics have **not** been produced; do not infer them from CH's own-target 20D rho.
- **Decision and stopping condition:** `AUDIT_RESULT_AVAILABLE / DECISION_COMPLETE / CG_REPLACEMENT_OF_AO_NOT_SUPPORTED / CH_BREAKOUT_SMALL_POSITIVE_RESEARCH_SIGNAL / CH_ALL_MARKET_APPROX_AO / ITERATIVE_OOS / NO_HYPERPARAMETER_TUNING / NO_MODEL_OR_TRADING_PROMOTION`. Keep AO as frozen MFE40 reference; retain CH as controlled 20D candidate, CG as historical/current comparison reference (no deletion or change to current model config/list). This Audit answers the previously pending across-seed U40 decision, so **do not create additional U40 Audit chains**. If the user elects a next scientific step, compare AO vs CH under identical, independently predeclared strategy conversion/portfolio execution rules and broker-cash costs, preserving PIT and avoiding any OOS-driven score mixing, thresholds or seed selection. The separate AO/CG True-HS 2×2 crossover question remains unevaluated by this report, and is not required to interpret the matched U40 selection result.
- **Scope of this turn:** Documentation/evidence status only (Registry + Experiment Log). No source code, PIT scores, optimizer, model fitting, comparison schema, Research menu or Trading was altered.

### 2026-10-17 - Broker-Cash AO/CG/CH Rolling strategy results and four-seed robustness (user-reported)

- **Evidence and provenance:** User-supplied Research [3] -> [2] Rolling and [4] Rolling Robustness console: `貼上的文字 (1)(20261017-011247).txt`; companion code/governance ZIP `test-branch-1_20261017_050125_f4d81cc0(1).zip`, SHA256 `551dbbb25663714a4f0afdd263be48db44a842940085523d95a17da6d1b1995e`. The ZIP carries source and governance documents, not the local completed strategy result, per-seed CSV, score files or lineage manifests. The following figures are **user-reported local results**, not GPT-recomputed evidence or independent audit of physical artifact hashes.
- **Registered scope:** `broker_cash_dl_conversion_v1`, nine arms C96-C104, 2021-01-01 through 2026-03-02, PIT extending-window Rolling; benchmark `end_to_end_v1` with four fixed seeds `693545351, 2014432738, 104504529, 762252050`. Strategy account economics remain `broker_cash_v1`; C99-C104 retain P16 and their frozen Exact or Direct allocator. No new fitting/selection experiment is authorized by this report.
- **Execution reuse:** The single Rolling console lists `REUSE` for all nine completed arms, fingerprint `be30a608e5c6`, including the nine previously validated PIT score/audit/manifest prerequisites. The four-seed strategy report lists all nine arms as `4/4 seed scientific results identity` REUSE and the whole completed robustness scientific result as `[ROBUSTNESS REUSE]` with fingerprint `69114bfd04a47891`; it did **not** train or replay in this invocation. A displayed `Research model score REUSE=0` is not evidence of missing models: all strategy scientific observations were already complete, so there was no outstanding model-score dependency in this run. The source ZIP cannot independently re-verify the user's local original prior run.

| Arm | Selector / model | Single Rolling Return / MDD / RoMD | Four-seed RoMD mean +/- Std | Four-seed aggregated Return / MDD |
|---|---|---|---|---|
| C96 | No-DL Min, P16 | 114.33% / 13.68% / 8.36 | 9.18 +/- 0.49 | 137.88% / 15.05% |
| C97 | No-DL Full Best, P17 | 203.40% / 14.05% / 14.48 | 6.44 +/- 2.46 | 151.07% / 23.72% |
| C98 | No-DL Full Agree 4x300, P15 | 141.92% / 18.62% / 7.62 | 5.81 +/- 1.54 | 120.89% / 20.96% |
| C99 | AO Exact K/R0 | 108.82% / 17.06% / 6.38 | 9.11 +/- 2.29 | 147.29% / 16.51% |
| C100 | CH Exact K/R0 | 107.70% / 19.07% / 5.65 | 9.29 +/- 2.70 | 151.20% / 16.91% |
| C101 | AO No-K/No-R0 | 120.52% / 26.09% / 4.62 | 2.18 +/- 0.76 | 67.47% / 31.00% |
| C102 | CH No-K/No-R0 | 130.81% / 33.73% / 3.88 | 2.39 +/- 1.56 | 69.46% / 30.76% |
| C103 | CG Exact K/R0 | 156.95% / 18.83% / 8.33 | 9.21 +/- 2.77 | 152.84% / 17.01% |
| C104 | CG No-K/No-R0 | 132.10% / 32.05% / 4.12 | 2.70 +/- 1.62 | 76.36% / 30.32% |

- **Primary controlled same-seed contrast:** Four-seed mean RoMD Direct-minus-Exact is AO C101-C99 `-6.93`, CH C102-C100 `-6.90`, CG C104-C103 `-6.51`; Exact wins `4/4` seeds in each contrast. In the single Rolling report, No-K/No-R0 selected-target mean R and Top-K percentages increase, but realized RoMD is lower, with much higher MDD and lower mean exposure. Selection metrics and actual portfolio PnL answer different questions.
- **Baseline and model contrast:** C96 has RoMD mean `9.18` with seed Std `0.49`, while AO/CH/CG Exact have `9.11 +/- 2.29`, `9.29 +/- 2.70`, `9.21 +/- 2.77`. CH and CG Exact improve mean RoMD over C96 by only `+0.11` and `+0.03`, while having larger MDD and seed dispersion. C97 single Rolling has high RoMD `14.48`, but its four-seed mean is only `6.44` (Std `2.46`), cautioning against single-run conclusions. Four seeds and iteratively reused OOS are insufficient for promotion or statistically reliable small-difference selection.
- **Report availability and terminology:** Reused four-seed robustness source lacks the older selection/target attribution sidecars needed for the Trade Quality, Selection Quality and First-Passage sections; the canonical renderer reports these unavailable instead of fabricating metrics or retraining. For Same-seed Contrasts, the displayed expression is `internal right - internal left`; `right_wins` and `left_wins` refer to underlying registered contrast positions, **not the visual placement around the minus sign**. This naming is potentially confusing but is not evidence that the signed deltas or win counts are computed incorrectly. No approved persistent report schema or wording is changed in this evidence update.
- **Scientific decision:** `USER_REPORTED_ROLLING_AND_4SEED_ROBUSTNESS_AVAILABLE / DIRECT_NO_K_NO_R0_DISFAVORED_IN_CURRENT_CONFIG / EXACT_DL_MARGIN_OVER_C96_NOT_ESTABLISHED / C96_RETENTION_AS_ROBUST_REFERENCE / ITERATIVE_OOS / NO_AUTO_PROMOTION / NO_TRADING`. Keep all registered identities and original observations immutable. Do not tune model, selector thresholds, seeds or training configuration using these repeatedly observed OOS results. More extensive root-cause audit is not authorized by these metrics alone; no re-training/replay is needed merely to reproduce the shown REUSE figures.
- **Scope of changes:** Registry current evidence statuses and Experiment Log append-only new result entry only. No executable code, report contract, PIT/score producer, research menu, optimizer, strategy parameters or Trading artifacts were changed.

### 2026-10-23 — Oracle Ground Truth RoMD V1: user-reported four-seed completed result

- **Evidence provenance**: User directly supplied the finished Windows `AUD-target-ground-truth-romd` Console table; GPT has not independently replayed full-market outcomes. Status `V1_USER_REPORTED_COMPLETE / HINDSIGHT_ORACLE / ITERATIVE_OOS / NOT_PIT / NO_PROMOTION`. Historical seven-treatment implementation and archived per-run results are retained; never rewrite them with a later schema.
- **Period and comparability**: `2021-01-01` to `2025-12-22` (common 40D mature stock-day cohort `643,617`), `end_to_end_v1`, 4 P16 benchmark parameter seeds, canonical Exact K/R0 / Broker-Cash, with same-window C96 mean RoMD=`8.14` (SD=`0.44`). This is *oracle ex-post selection*, not a deployable performance forecast.
- **Target mean RoMD / SD / Delta vs paired C96 / wins over C96**:
  - AO40_EqualRank: `17.04 / 1.98 / +8.91 / 4/4`.
  - CG40_EqualRank: `18.32 / 1.40 / +10.18 / 4/4`.
  - CH20_EqualRank: `16.43 / 1.74 / +8.29 / 4/4`.
  - U40: `22.45 / 5.29 / +14.32 / 4/4`.
  - U20: `22.11 / 3.44 / +13.97 / 4/4`.
  - FirstBreach_NetR: `21.21 / 4.26 / +13.07 / 4/4`.
  - MFE40_Control: `19.65 / 3.10 / +11.51 / 4/4`.
- **Interpretation limit**: These are pooled across *strategy parameter seeds* and preselected future-truth rankings. The seven target geometries were not made available at actual trade decision time; Oracle RoMD is neither a mathematical upper bound on all strategies nor evidence that a DL model can learn a target. Higher Oracle RoMD must be balanced with separately observed learnability and common-truth joint quality. This repeatedly observed OOS interval is iterative evidence, not a new untouched holdout.

### 2026-10-24 — P0 收斂：40D Oracle / AK-CG-CI-CJ-CK Rolling / C105-C108 Broker-Cash 結果，退役 CI/C106 現行資格

- **基準與證據**：本輪使用 `test-branch-1_20261024_184451_4c989151.zip`，SHA256=`46c9aeb63f3dd56bd7b1feb64b15367548f2cd88695588529e195a702ba35596`，及使用者提供之 `貼上的文字 (1)(20261024-110607).txt` Console。ZIP 沒有 `outputs/` 或本機正式測試通過證據；下列為 **user-reported REUSE 結果**，並非 GPT 重新訓練／重新 Replay／獨立重算，沒有新 PIT/Trading 授權。所有觀測均為反覆使用的 iterative OOS。
- **Oracle `[1]` V8 完成**：相同 2021-01-01～2025-12-22 成熟期間、`643,617` stock-days、四個 end-to-end_v1 same-seed benchmark（S1=693545351 / S2=2014432738 / S3=104504529 / S4=762252050）；C96 同期間 RoMD=`8.14`；5 Exact + 4 Direct treatments，合計 40/40 Seed×Arm `REUSE`、RUN=`0`、BLOCKED=`0`。Oracle 的 RoMD Mean：CG40 EqualRank=`18.32`、MFE40=`19.65`、R40=`19.69`、T40=`20.33`、T40-MFE=`18.79`；Direct MFE40=`80.02`、R40=`77.66`、T40=`96.67`、T40-MFE=`100.13`。最後一項的 SD=`63.29`，不能把 hindsight future-label 排名效益宣稱為 PIT 或可預測的 Strategy 提升；T40-MFE Exact 未超越 T40 Exact。舊目標與歷史報表版型／科學工件永久保留。
- **Rolling OOS `[2]→[4]` 完成**：5/5 `REUSE READY`，模型 `AK/CG/CI/CJ/CK` 的 Global Daily rho=`0.3844/0.4044/0.0181/0.2194/0.1508`；Breakout Daily rho=`0.3633/0.3881/-0.0138/0.1941/0.1249`，Breakout Pair=`64.07%/65.35%/50.40%/57.56%/55.17%`，Breakout self-target Top−Bottom=`2.3880/2.6832/-0.2996/-0.1360/-0.1114`，Breakout Top10 MFE=`27.60%/30.48%/15.41%/9.74%/9.15%`，Top10 adverse-to-peak=`5.57%/5.77%/3.57%/2.28%/2.27%`。CI/CJ/CK 5年正向 Top−Bottom 年份=`1/5, 2/5, 1/5`；CK 5年正 rho=`4/5`、drift=`YES`。不同 Target 的 self-target rho、Target 值不可直接跨模型視為同尺度效益；Adverse-to-peak 亦不能冒充完整 40D MAE40 真值。
- **Strategy Compare `[4]→[2]` 完成**：原 canonical suite 8/8 `REUSE`，模式為 Broker-Cash Rolling 2021-01-01～2026-03-02、Max positions 10、Rotation off、相同 P16 Min Exact K/R0 DL conversion 與 0050 price-return 參考。報表記錄如下；`C97` Full No-DL 不屬 P16 Min 的等控制對比。

  | Arm | Total Return | MDD | RoMD | Trades | Eval note |
  |---|---:|---:|---:|---:|---|
  | C96 Min No-DL | +114.33% | 13.68% | 8.36 | 350 | P16 Exact/Min control |
  | C97 Full No-DL | +203.40% | 14.05% | 14.48 | 260 | different Full policy |
  | C98 Full Agree No-DL | +141.92% | 18.62% | 7.62 | 303 | different Full policy |
  | C103 CG Exact | +156.95% | 18.83% | 8.33 | 338 | current reference |
  | C105 AK Exact | +153.98% | 18.04% | 8.54 | 361 | result available |
  | C106 CI Exact | +127.96% | 15.92% | 8.04 | 337 | user-requested current exclusion |
  | C107 CJ Exact | +143.78% | 13.01% | 11.05 | 359 | four-seed strategy robustness not supplied |
  | C108 CK Exact | +112.25% | 14.80% | 7.59 | 342 | negative single-seed relative control |

- **P0 科學決策**：CI terminal R40 模型排序近隨機且 C106 RoMD=`8.04` < C96=`8.36`，結案並由使用者明確要求**移除現行模型／策略比較資格**。CK T40-MFE 模型 Breakout rho/Pair、self-target Top−Bottom 與 C108 RoMD=`7.59` 未支持後續延伸同族 Target，也不 promotion；**使用者本輪沒有要求把 CK/C108 從配置移除**，因此只記錄目前負面證據並保留當前成員資格。CJ 的 C107 單 Seed RoMD=`11.05` 較 C96 高 `+2.69`，但 `4-seed` same-seed Strategy Robustness 尚未提供、模型 Score→MFE rho 在 Breakout 為 `-0.2567`，不能視為穩健優勢或模型 Gate PASS。未獨立取得 Model Gate 結論：策略依賴計畫對 CI/CJ/CK 顯示 `WARN: Model Gate=FAIL`，只是使用者 Console 記錄，不能當作獨立重新驗證。
- **變更邊界與證據保全**：只有 `config/breakout_quality.py::BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES` 刪除 active `MR-13CI` 引用；active model test list 由既有 SSOT 派生為 `AK/CG/CJ/CK`，`core.strategy_compare_policy.get_strategy_compare_suite` 既有的 `filter_by_current_model_list` 自動令 C106 和 C106-C96 從 **current** OOS／Rolling／Robustness 移除（新 current suite 理應剩 C96/C97/C98/C103/C105/C107/C108）；不刪 `MR-13CI`、`DL-CONT13CI-ROLL`、`SR-C106`、歷史 Replay／PIT Score／model checkpoint、過往的 comparison 或 results。未改模型／Target／Loss、K/R0、參數工件／SSOT、SOP schema、menu 或 Trading；新 suite 組合生成的是新比較結果，不能覆寫舊 8-arm 結果。GPT 未執行 `apps/run_bundle.py`、`apps/test_suite.py`，正式 Windows double check 仍需由使用者執行。

### 2026-10-26 — Research [1] Oracle 原始定義確認／Parity 缺陷分類（使用者決策；尚未修復程式）

- **原有固定研究問題**：使用者重申 Research `[1] Target 驗證` 從一開始就是「假設 DL 能完美預測 Target，在與正式 Research DL 策略完全相同的選股與交易條件下，RoMD 可以達到多少？」。正式長期治理條款已加入 `doc/PROJECT_SETTINGS.md` E13；各 Target 的真值定義不同，**不代表可自行改變 Oracle 與 DL 的評分資格、分數介面或交易條件**。
- **本次發現與分類**：來源 ZIP `test-branch-1_20261026_015913_e8abe24e(1).zip`（SHA256 `2e108b26170fb8b696ea89c9152557fa0bd9aa9424673b1f6d3014daa80db8bf`）比對指出：舊價格型 Oracle 可評分股票日範圍與正式 Daily-universal DL 不一致；Oracle 預載 `2020-12-31` 資訊日但正式 DL 的 `score_start=2021-01-01` 不包含；Oracle 的每日百分位分數與部分正式 Exact DL raw-score 消費契約不同，尚未證明數值完美預測等價。這些屬偏離原始驗收定義的實作缺陷或契約未閉合，**不是新增的科學定義或另一套有效 Oracle 假說**；不得以既有合成測試 PASS 取代端到端 parity 驗收。
- **既有結果狀態與處置**：本次使用者提供的四 Seed Oracle RoMD 表及任何受上述差異影響的舊快取，均暫列 `NOT_VALIDATED_AS_PERFECT_DL_ORACLE / PARITY_FIX_REQUIRED`；不得宣稱為同條件「完美 DL RoMD」、作為有效模型／Target 優劣結論或升格依據。歷史報表數字與原始工件只保留追溯，不在有效比較中 REUSE；修復後依原核准的科學問題、相同正式 Research DL arm 與正確 artifact identity 重算實際受影響 arms，不因本次缺陷逐 Target 發行新 SC／Label／report version，也不全面強制 DL 重訓。
- **工作邊界**：本次僅補專案治理與 append-only 決策紀錄；未改 Target／Score／Replay 程式，未重新跑真實四 Seed 市場 Replay，亦未執行 Windows formal Double Check。原始報表是否每一個 arm 的實際 RoMD 都受到多少影響，仍須修復後由正式端到端比對及 Replay 證實。

### 2026-10-26 — Research [1] Target Oracle E13 改版後四 Seed 結果登錄及與修復前歷史對照（使用者實際報表）

- **本輪來源／基準**：使用者提供 `貼上的 Markdown(2).md`（SHA256 `34e28902ef5455927912abb5d29a314cd5269fee590083ccc562a9c75849fce7`）及 `test-branch-1_20261026_144657_6c957189(2).zip`（SHA256 `586de78fff77ab0fec2c92cb3dec0a6ee65c953ce2a34e3480cfddfeba8a571d`），原 ZIP `doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md` 共 13,793 行；本紀錄僅追加，不替換 2026-10-23～26 既有歷史與結果。依序核對 `doc/PROJECT_SETTINGS.md`、Registry、Log；這是接收使用者回報的真實市場報表，而非 GPT 重新跑全市場 Replay。
- **研究／報表條件**：Research `[1] Target 驗證`，`audit.target_oracle_romd@v9`（source payload `schema_version=8`），11 Oracle treatments（6 Exact K/R0 + 5 Direct No-K/No-R0），另有 same-window C96；Period=`2021-01-01～2025-12-22`；`end_to_end_v1`；四個正式 P16 策略參數 Seed=`693545351/2014432738/104504529/762252050`；40D mature、已核准 perfect Target daily-percentile score、Portfolio Broker-Cash。2025 截止日早於年底，`Full Year=False`，不能視為完整年度。
- **最重要的樣本口徑變更**：修復前 2026-10-24 Oracle V8 報表 common cohort=`643,617`，本報表=`608,204` stock-days，減少 `35,413`（`5.50%`）。當前程式以正式 feature-eligible DL stock-day 與 score start 遮罩價格真值，不能額外給不可評分股票日 Oracle 分數；C96、DailyUniversal 的舊／新 RoMD 恰相同，對應「未消費改變之價格型分數」的預期，但僅總和相同**不能**取代逐筆 Cache SHA／交易等價證明。修復前數值為歷史 `PARITY_FIX_REQUIRED`，不能和新值當作同一有效科學契約的 controlled outcome 比較。
- **逐 Target 4-seed aggregate，改前 RoMD 為歷史使用者報表，改後 8 欄來自本次使用者提供 Markdown**（Return/MDD 皆為百分比；Delta 是新 RoMD − 舊 RoMD）：

  | Target | Old RoMD | New Return | New Ann. | New MDD | New RoMD | Δ RoMD | New SD | New EV(A) |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | C96 Same-Window | 8.14 | +122.24% | +17.42% | 15.05% | 8.14 | +0.00 | 0.44 | +0.81 R |
  | CG40_EqualRank | 18.32 | +258.95% | +29.01% | 13.41% | 19.33 | +1.01 | 5.92 | +1.39 R |
  | MFE40 | 19.65 | +263.90% | +29.50% | 14.07% | 18.69 | -0.96 | 3.61 | +1.43 R |
  | R40 | 19.69 | +242.16% | +28.09% | 13.27% | 18.26 | -1.43 | 1.21 | +1.34 R |
  | T40 | 20.33 | +268.66% | +29.86% | 12.77% | 21.02 | +0.69 | 3.87 | +1.40 R |
  | T40-MFE | 18.79 | +235.43% | +27.49% | 12.79% | 18.37 | -0.42 | 2.59 | +1.39 R |
  | MFE40_Direct | 80.02 | +1014.92% | +60.61% | 14.36% | 75.05 | -4.97 | 41.26 | +2.18 R |
  | R40_Direct | 77.66 | +966.00% | +59.46% | 12.83% | 74.83 | -2.83 | 31.80 | +2.33 R |
  | T40_Direct | 96.67 | +1099.12% | +62.88% | 11.71% | 91.88 | -4.79 | 39.07 | +3.01 R |
  | T40-MFE_Direct | 100.13 | +1242.04% | +65.57% | 13.12% | 93.21 | -6.92 | 53.78 | +3.07 R |
  | DailyUniversal_MinTrade40_NetR | 20.32 | +269.60% | +29.73% | 13.46% | 20.32 | +0.00 | 6.97 | +1.56 R |
  | DailyUniversal_MinTrade40_NetR_Direct | 122.23 | +1486.69% | +71.50% | 12.02% | 122.23 | +0.00 | 71.21 | +3.83 R |

- **完整四 Seed RoMD 觀測**（依序 S1/S2/S3/S4；C96 baseline S1/S2/S3/S4=`8.38/8.18/8.48/7.51`；其餘 11 Target 對同 Seed C96 均為 `4/4`，但這**不**表示 Target 兩兩比較也是 4/4）：

  | Target | S1 | S2 | S3 | S4 |
  |---|---:|---:|---:|---:|
  | C96 Same-Window | 8.38 | 8.18 | 8.48 | 7.51 |
  | CG40_EqualRank | 27.88 | 14.32 | 16.96 | 18.18 |
  | MFE40 | 23.45 | 15.54 | 19.53 | 16.26 |
  | R40 | 18.08 | 16.63 | 19.31 | 19.02 |
  | T40 | 26.32 | 19.48 | 21.08 | 17.20 |
  | T40-MFE | 21.90 | 16.19 | 18.69 | 16.70 |
  | MFE40_Direct | 126.97 | 30.52 | 85.50 | 57.19 |
  | R40_Direct | 117.82 | 42.15 | 75.35 | 64.01 |
  | T40_Direct | 146.16 | 55.30 | 91.33 | 74.72 |
  | T40-MFE_Direct | 169.32 | 42.65 | 80.63 | 80.24 |
  | DailyUniversal_MinTrade40_NetR | 29.72 | 16.28 | 21.28 | 14.01 |
  | DailyUniversal_MinTrade40_NetR_Direct | 227.35 | 69.32 | 95.18 | 97.07 |

- **主要結果與統計穩健性**：Exact 新 RoMD 最佳 `T40=21.02 ± 3.87`（Return=`+268.66%`，MDD=`12.77%`），次高 `DailyUniversal_MinTrade40_NetR=20.32 ± 6.97`（Return=`+269.60%`，MDD=`13.46%`）；價格型 `R40` 從 `19.69` 下降到 `18.26`，`CG40_EqualRank` 從 `18.32 ± 1.40` 升為 `19.33 ± 5.92`（Seed dispersion 明顯擴大）。Direct 最佳 `DailyUniversal_MinTrade40_NetR_Direct=122.23 ± 71.21`（Return=`+1486.69%`，MDD=`12.02%`），次高 `T40-MFE_Direct=93.21 ± 53.78`。DailyUniversal Direct 同四 Seed 為 `227.35/69.32/95.18/97.07`；Direct 減同 Target Exact 的逐 Seed RoMD 差為 `+197.63/+53.04/+73.90/+83.06`（平均 `+101.91`），不應宣稱是同配置的獨立 Target 提升。
- **同 Selector 的 same-seed 配對**：Exact `T40` 對 `DailyUniversal_MinTrade40_NetR` 新 RoMD 平均僅 `+0.70`，四 Seed 中 **2/4** 勝（不等於 `Win/Seed=4/4`，後者只與 C96 比）；Direct `DailyUniversal_MinTrade40_NetR_Direct` 對次高 `T40-MFE_Direct` 平均 `+29.02`，**4/4** Seed 勝。前者的「Exact 最佳」僅四 Seed 平均排名，後者雖同 Seed 皆勝，仍有高 SD 與 hindsight 不可預測限制。
- **年度穩健性**：2022 C96=`-1.26%`、T40 Exact=`+8.62%`、DailyUniversal Exact=`+4.81%`、T40 Direct=`+23.32%`、DailyUniversal Direct=`+22.92%`；2023 DailyUniversal Direct=`+183.65%`；2025 全部 Target `Full Year=False`，其中 DailyUniversal Direct=`+44.00%` 僅是至 12/22 的部分年度觀測，非完整年度。
- **與修復前對照的來源邊界**：舊 CG/MFE/R/T/T40-MFE 等歷史 RoMD 來自本 Log 2026-10-23／24 的 V8 user-reported evidence；DailyUniversal 舊 Exact/Direct `20.32/122.23` 為 2026-10-26 稍早對話中使用者報表的既有回報（非此原 ZIP 內獨立 replay sidecar）。C96 舊同窗 4-seed 也是本 Log 歷史數值。舊的 broader price oracle results 已因 E13 parity bug 標記無效，保留原文供追溯；本紀錄不回填舊 Return/MDD 缺失值。
- **解讀與資格**：Exact／Direct 沿用同一 Broker-Cash portfolio（現金、倉位、費稅、P16），Direct 更動的是既有 selector 的 K/R0 約束；資金可由已實現損益滾動複利，但報表**未提供逐日現金利用率、部位、成交頻次或持股差異歸因**，因此不得把 Direct 大幅差距全歸於資金變多。C96 為 Min no-DL 原 selector control，並非 Direct 的同配置 no-DL 純分數效果控制。全表均為 hindsight future truth、NON-PIT、迭代 OOS，並非可學性、可部署上限或 Trading 升格。
- **驗收與不改動範圍**：已用本次 Markdown 的 44 筆 paired seed 資料獨立核對 11 組 RoMD Mean／sample SD（顯示精度容差）、四 Seed 集合、12 組 aggregate、年度 2025 False，全部算術一致。提供的 ZIP 不含正式 `outputs/` per-arm JSON、完整 `data/full/`、`models/` 與本機執行 QA，不能獨立認證原始四 Seed Replay SHA、逐日評分／持倉／現金 parity 或 Windows double check。狀態=`USER_REPORTED_POST_E13_FOUR_SEED_RESULT_AVAILABLE / AGGREGATE_ARITHMETIC_VERIFIED / PHYSICAL_REPLAY_PARITY_NOT_IN_UPLOAD / HINDSIGHT_NON_PIT / ITERATIVE_OOS / NO_PROMOTION`。僅登錄新觀測並更新 Registry 結果狀態；不改 SC、模型、策略、runtime、report contract、任何原始已存結果或 Golden。

### 2026-10-26 — 新增 DailyUniversal Formal-Min 60-session Oracle Exact / Direct（使用者授權；尚無實際 Replay 結果）

- **狀態**：`IMPLEMENTED / SYNTHETIC_INTEGRATION_VERIFIED / MARKET_RESULTS_PENDING / HINDSIGHT_NON_PIT / NO_FITTING / NO_TRADING`。使用者明確授權新增 `DailyUniversal_MinTrade60_NetR`、`DailyUniversal_MinTrade60_NetR_Direct` 於 Research `[1] Target 驗證` 正式比較清單；不是對歷史 MFE60／R60／T60 研究的重新啟用，也沒有新增正式 DL 模型。
- **唯一基準**：`test-branch-1_20261026_160135_cc3ac068.zip`，SHA256=`b90d9ee9a4c13f7865672e5ac76959af5c098cc30fbe0dc18bf9e71db9c09ab5`，已先讀取 `doc/PROJECT_SETTINGS.md`、Experiment Registry、Experiment Log。補丁以本 ZIP 作為唯一 source baseline，未用先前 GPT patch 覆蓋新版本。
- **新身分**：Target=`daily_universal_formal_min_roos_trade60_netr_v1`；Label=`LABEL-daily_universal_formal_min_roos_trade60_netr_v1`；比較 treatment 名稱為上述兩個，都是 ex-post Oracle。保留已註冊 40D `daily_universal_formal_min_roos_trade40_netr_v1` 的身分及原始結果。
- **唯一 Target 科學變更**：沿用 40D 的正式 feature-eligible stock-day universe、每 Seed 正式 Min ROOS P16 date-effective 参数、合併交易所交易日、當天訊號後首個合法個股 order day、ATR/限價/停損/追蹤/券商費稅、真正單股成交淨 R、未成交最後名次、逐日 `average_zero_based` percentile；**只把假設交易路徑的到期結算及成熟遮罩改為第 60 個合併交易所交易日**。60D Exact/Direct 真值與同一 Seed 的 score-map SHA 相同，兩組只使用各自既有 K/R0 Selector；Broker-Cash Portfolio Replay 與 C96 沒有調整。
- **共同期間／Cache**：包含 60D 的比較，所有 treatment 與 C96 對齊最長 60D 成熟截止日、相同情報日與相同可評分股票日，**不可拿不同結束日 40D 舊工件當作同窗 REUSE**，且不允許默默取不同股票母體交集或用 40D 路徑偽裝 60D。只有 40D 的比較清單保留原 40D end、`2021-01-01` 起點及原快取身份。新正式清單為 **13 treatments（7 Exact + 6 Direct）加 C96；四 Seed 最多 56 個逐 arm replay**；實際 RUN/REUSE 由 SSOT 逐 arm 審核，不預設全部重跑或重用。
- **未更動**：Report Contract=`audit.target_oracle_romd@v9`、payload schema=`8`，模型、訓練、最佳化器、研究 Seed、正式交易實作、歷史報表、原有 40D 科學定義與 golden schema 一律保持。不是可交易、PIT-legal 或 ML 實測結果。
- **GPT 端驗證範圍**：以獨立 synthetic 驗證 40/60 session 成熟邊界與同一 canonical 路徑強制出場日期、60D 每 Seed target/label/evidence、40D-only 快取重用、混合 60D 縮短共同 window 後避免重用不同期間 40D results、Strict feature cohort、分數注入與 Direct score 共用，並執行所涉 Oracle 合約 unittest。ZIP 不含市場全價資料、正式 P16 多 Seed 輸出及 Replay sidecars；**尚無 60D RoMD／年度／對 C96 實際差值，也未執行使用者本機 Formal Double Check 或 E13 逐日完整現金/成交 parity**。故不宣稱 Result Available 或正式驗收通過。

### 2026-10-26 — Research [1] MinTrade Oracle 統一期限／截止結算架構 Bug Fix（使用者決策）

- **基準與決策**：以使用者回退的 `test-branch-1_20261026_190837_ba534550.zip`（SHA256 `fd918258fd2cdb8afb447a594f9328d0e4b7c15360c861236391a650744fb15b`）作本輪唯一程式基準。使用者明確否定「為較長 Target 自動縮短共同成熟窗口」是合法科學變更，授權在**現有 Oracle 實作與已核准身份上直接修復 Bug**，不得建立平行 maturity 模式、逐 Target 新 SC／report 版本或覆寫歷史績效紀錄。本輪不修改 REUSE 架構、Trading、DL FIT、Optimizer 或現有價格型 MFE40／R40／T40 真值定義。
- **唯一真值結算規則**：正式 Min 單股模擬 owner `filters/breakout_quality/daily_trade40_target.py` 對任何正整數 `H` 和完整週期 `H=None` 使用相同交易路徑與風控／費稅 SSOT；結算事件為有效自然出場、H 期限（若有）、Research 資料截止中最早者。正式原有 Research 截止為 `2026-03-02`。資料尾端還有合法可執行部位時，以原正式 `finalize_open_position_at_end` 結算；停牌／沒有可合法平倉價、未成交等情境不虛構 R，保留 NULL 及 censored-last 排名。截斷、資料尾端結算以獨立數量證據記錄，不能說成每筆都自然持有 H 日。
- **唯一評估時間／母體**：Oracle Registry 的 `oracle_evaluation_end_date=2025-12-22` 是此 Research DL 受控同窗比較的固定評估截止，起始 `2021-01-01`，與目前有多少 Target、最長 H 或是否只選 Direct **無關**。研究資料可向後觀測至 `2026-03-02` 計算每個歷史股票日的 Target 結果。原本 `2025-11-24` 的 60D 最長成熟截止條件、以 target maturity 刪除較晚股票日、讓 40D / C96 跟著整批改期間均屬本次修復的**不合規工程缺陷**，不是另立 Target 科學身分的理由。
- **現行清單**：四 Seed，`C96` 加 **15 個 Target（8 Exact／7 Direct）**，其中既有 MinTrade40、MinTrade60，以及新增無期限 `DailyUniversal_MinTrade_NetR` 與 `DailyUniversal_MinTrade_NetR_Direct`。同 Seed Exact/Direct 真值來源、排名與分數 SHA 一致；Portfolio 持有期限、現金、K/R0 Selector、P16 有效期、掛單與費稅不因 Target 期限改動。任意 H（例如 87D）只需在 Registry 宣告新的合法 Target 名稱與 H，並不需加新的 Producer 分支。
- **失效／追溯**：2026-10-26 先前 60D-mature 跨 Target 同窗結果及其縮短期間 40D Replay 是歷史不合規實作 evidence，不可視為現行同期間績效，也不得當成可用快取強行 REUSE；原始結果保留、標記為 `SUPERSEDED_BY_UNIFIED_CUTOFF_BUG_FIX`。此次現行 Output／Cache 必須按實際真值、評估日期、cohort、價格 SHA、Seed/P16 及 Selector 完整身分檢查，不能把舊答案直接重貼；**既有正確且同一實際身份的 40D 歷史工件應按原規則依法 REUSE**，其重算治理留在下一輪獨立處理。
- **工程驗收**：獨立 Synthetic 驗證自然出場／H 期限／Research 期末結算，40／60／73／87／100／全週期單一演算法，晚於資料截止價格不得被使用，沒有可交易下一日不捏造結果；40D → 40+60D → 40+60D+完整週期三階段保持相同 Portfolio 期間／共同母體，僅新 Target Arm 需新增回放；Direct-only 可共用完整週期真值而不偷偷執行 Exact。相關 `unittest` **130/130 PASS**（獨立合成測試，非 Windows formal double check）。ZIP 未包含本機完整 `data/full`、全市場 P16 工件及完整 Portfolio Replay sidecars，故**尚未認證真實市場四 Seed RoMD 或逐日 E13 parity，也沒有產生新績效數字**。Report `audit.target_oracle_romd@v9`／payload schema 8 均維持既有正式欄位與格式。

### 2026-10-27 — 全期間結果後 Target 算法獨立審查與修復（同一既有科學契約）

#### 基準、輸入與範圍

- 唯一程式基準：`test-branch-1_20261027_211502_47952b65.zip`，SHA256 `b1726582530a636cea49f13dec3d1894514a55e91e17dc1b38eb3fbe12d2ccfe`；獨立解壓工作目錄 `/workspace/scratch/b884d04c0e9f/target_review_20261027_211502`。已依序讀取Project Settings、Registry、Experiment Log及相關Architecture/執行owner；本次只追加原始Log，未刪改舊bytes。
- 使用者結果：`貼上的 Markdown(3).md`，SHA256 `15f88aa2050682c3e17d292cb62e7549c5bd44bef7acd30577523864935a557d`；評估2021-01-01～2026-03-02，common cohort630,589，4 Seeds（693545351、2014432738、104504529、762252050），15 Oracle treatments＋C96。ZIP沒有完整市場行情/P16/逐日Replay工件，僅附10檔reduced OHLCV，不能以貼出的summary重算真實drawdown、cash、trades或認證其source receipt。
- 狀態：`USER_REPORTED_PRE_FIX`。以下數字為使用者貼出的歷史結果，不是本輪修復後重跑。60筆seed-paired RoMD、15組mean/sample SD/ddof=1/delta/wins在顯示rounding範圍內一致；16組年度表都覆蓋2021～2026、每年4 Seeds、2026為partial year。此算術檢查不等於逐日經濟驗收。

#### 已確認的錯誤、反例與修復

1. **Price同分被重組浮點拆開**：正式T40/T40-MFE三條路徑均輸出float32 `0.25R`，正確percentile `[.5,.5,.5]`；原Oracle從已float32的components重組出`0.24999998137354806 / 0.2500000069849186 / 0.2499999999999991`，錯排`[0,1,.5]`。R/MFE也有獨立反例。改為直接使用正式canonical batch的`target_raw_r`，不用epsilon修名次或第二套公式；CG尾端component除法精度同步正式Daily loader，原50/50 geometry、First-Breach路徑保持。
2. **MinTrade跨期訊號context凍結**：12/30原訊號、12/31未成交，1/1 P16 high_len由30改3；原Target仍把舊pending算成1/5買100、最終`+.3165886448R`，正式Portfolio已在1/2由新setup取代，原事件應為NULL且1/2成熟。該新setup的另一筆交易不能移接到舊標籤。另有current indicator清pending（原+.31659→NULL）、已成交indicator出場（原+.31659→-.1470411715R，淨損益-1470）、current ATR移停（原+.32249→-.1215694125R，淨損益-1214.6）四案。已以真實signals/Portfolio engine兩種single-member P16 payload比對；origin numeric params/entry geometry固定，逐session更新有效signal context，缺期fail closed。40/60/fullcycle同一owner，原A2無resolver行為不變。
3. **Warm REUSE來源證據不完整**：price舊signed Replay在current formula SHA改變後仍REUSE；MinTrade source SHA未含position_management等實際owner；Dataset summary end由2026-03-02改2025-12-31時Daily Exact/Direct也仍REUSE。修正shared source closure及summary唯一witness，將price formula/label/source/summary與MinTrade實際owners/summary全程接通truth cache、共用feature cohort cache與Replay policy。修前反例先FAIL、修後PASS；無current witness的舊Target RUN，歷史bytes不變，C96依自己的有效身份REUSE，規劃期間沒有真值／Portfolio precompute；summary確認期間異動在producer前停止。

#### 獨立驗證與限制

- 最終合併：`python3 -m unittest tools.validate.synthetic_target_oracle_cases tools.validate.synthetic_daily_trade40_oracle_cases tools.validate.synthetic_daily_trade40_cases -q`，183/183 PASS，6.729s。這不是`apps/run_bundle.py`或`apps/test_suite.py`，兩者均未執行。
- 既有`validate_breakout_quality_trade_path_label_contract_case(None)`直接驗證16/16 assertions PASS：A2 continuation、partial TP、fee/risk、natural/forced exit、blocked/unfilled及optional deadline保持。
- 獨立price手算2,402次精確float32比對PASS：H=1/2/7/40/60、四scalar formulas、排除information OHLC、峰後MAE、one-bar clip；不是將production函式當成expected。
- 獨立MinTrade Decimal/time/ranking15項PASS：948股買100/stop90/賣103，淨損益2278、initial net risk9991、NetR=.22800520468421578；partial TP總淨損益6526、NetR=.653187869082174；自然stop=-1R、entry-day不能當沖、不可結算NULL、H40/60/87/100/None、cutoff後行情不影響、NULL-last/ties。另6組stock/ETF Decimal fee-min-tax案例PASS；現行raw fee=.001425，無自動6折/72%回饋，未變更fee config。
- 真實Portfolio engine合成Oracle／DL score-boundary parity：3情境（首OOS資訊缺分、跨年effective參數、cutoff inference rows）×2獨立sizing fixtures×Exact/Direct＝12 pairs／24 replays；292組每日cash/positions/equity/params狀態、168筆execution配對、24筆trades配對一致。兩組fixtures不是本機真實4 Seeds的P16；僅排除刻意不同的score來源provenance欄位；無numba環境走既有reference fallback。
- 原ZIP647個Python files AST parse PASS。附帶reduced9股（排0050）、10,859成熟stock-days的MFE/R/T/T-MFE新舊daily rank差異為0。這不涵蓋全市場630,589 feature cohort，也不含CG尾端；因此不能聲稱報表所有數值必變，更不能估計新RoMD。
- 0050是feature benchmark、沒有DL score；Portfolio正式缺分fallback仍可選它，不能因Oracle自行排除。Exact/Direct共用同一Target分數，但Selector按各自既有契約；C96未加Direct語義或Target依賴。

#### 本輪決策與待驗收

這是對既有Target算法／E13一致性及C23來源認證的工程修復，不新增SC、Target/Label、Report schema或實驗版本。原績效只保留作修復前歷史觀測，尚不能據此選定最佳Target。套用patch後由既有Research `[1] Target 驗證`重建9 price×4＋6 MinTrade×4＝60個scored Replay；C96的4格由自身價格/P16/cohort/交易完整身份決定REUSE/RUN，不強制重訓DL或跑Optimizer。正式Windows Bundle、全市場60個修復後Replay與真實4 Seeds E13逐日parity待本機驗收。沒有整檔刪除；PROJECT_SETTINGS、CMD、private ToDo/token、現行費率/比較清單/選單/模型與Portfolio engine未改。

#### 使用者回報的修復前聚合結果（完整16列，僅歷史證據）

| Target | Return Mean | Ann. Mean | MDD Mean | RoMD Mean | RoMD SD | Delta RoMD | Win/Seed | EV(A) Mean |

|---|---|---|---|---|---|---|---|---|

| C96 Same-Window | +137.88% | +18.27% | 15.05% | 9.18 | 0.49 | +0.00 | - | +0.89 R |

| CG40_EqualRank | +310.60% | +31.02% | 13.41% | 23.22 | 8.35 | +14.05 | 4/4 | +1.53 R |

| MFE40 | +315.55% | +31.55% | 14.07% | 22.34 | 4.77 | +13.17 | 4/4 | +1.55 R |

| R40 | +290.85% | +30.23% | 13.27% | 21.92 | 1.42 | +12.74 | 4/4 | +1.52 R |

| T40 | +327.51% | +32.18% | 12.77% | 25.57 | 6.35 | +16.39 | 4/4 | +1.53 R |

| T40-MFE | +277.63% | +29.21% | 12.79% | 21.63 | 3.86 | +12.45 | 4/4 | +1.54 R |

| MFE40_Direct | +1339.25% | +65.59% | 14.36% | 98.76 | 56.20 | +89.58 | 4/4 | +2.41 R |

| R40_Direct | +1312.01% | +65.07% | 12.83% | 101.49 | 47.97 | +92.32 | 4/4 | +2.62 R |

| T40_Direct | +1398.61% | +66.78% | 11.71% | 116.71 | 51.78 | +107.53 | 4/4 | +3.32 R |

| T40-MFE_Direct | +1644.24% | +70.44% | 13.12% | 123.34 | 77.07 | +114.17 | 4/4 | +3.40 R |

| DailyUniversal_MinTrade40_NetR | +318.97% | +31.52% | 13.46% | 24.00 | 8.91 | +14.82 | 4/4 | +1.71 R |

| DailyUniversal_MinTrade40_NetR_Direct | +1918.64% | +75.73% | 12.02% | 157.64 | 97.15 | +148.46 | 4/4 | +4.14 R |

| DailyUniversal_MinTrade60_NetR | +321.21% | +31.93% | 13.28% | 24.25 | 5.52 | +15.07 | 4/4 | +1.77 R |

| DailyUniversal_MinTrade60_NetR_Direct | +1741.22% | +74.05% | 11.77% | 146.51 | 63.57 | +137.33 | 4/4 | +4.02 R |

| DailyUniversal_MinTrade_NetR | +336.92% | +32.70% | 13.53% | 25.40 | 8.23 | +16.23 | 4/4 | +1.74 R |

| DailyUniversal_MinTrade_NetR_Direct | +1360.67% | +66.39% | 11.10% | 122.64 | 59.76 | +113.47 | 4/4 | +3.59 R |

### 2026-10-28 — Research [1] Target Oracle 四 Seed 正式回放結果與 64/64 REUSE 登錄（使用者本機輸出）

- **基準／證據**：`test-branch-1_20261028_124740_3c745ada.zip`（SHA256 `bc8baf12fedc19096f3f209bc0f04a01c3379c7c46eab649c95aeaac9f7e7120`）；本機 Console `貼上的文字 (1)(20261028-051732).txt`（SHA256 `6b3bed7f4c8eafffa8b5e2d1020ccbe3e32c41ab8df6b470960efe19029c33c5`），先前完整人讀報表 `貼上的 Markdown.md`（SHA256 `347e3c895c30db382e274a9ab5c0d2f84add5257d3ee1a5e55d9a8b6b4cb3a68`）。已依序核對 `doc/PROJECT_SETTINGS.md`、本 Registry、既有 Experiment Log，並核對 `core/audit_registry.py`、正式 Replay／Score owner 與當前 Report Contract。本次由 GPT 登錄**使用者本機實際執行輸出**，不是 GPT 自行執行完整市場 Replay。
- **正式研究條件**：Research `[1] Target 驗證`；`AUD-target-ground-truth-romd`，Report `audit.target_oracle_romd@v9`／payload schema 8，不變更任何 Scientific／Report identity。期間 `2021-01-01～2026-03-02`；Score start `2021-01-01`、實際首日 `2021-01-04`；正式 DL feature-eligible **630,589** 個股票日；C96 加 15 Oracle treatments（9 price + 6 DailyUniversal MinTrade，Exact K/R0 與 Direct No-K/No-R0 各循原已核准 Selector）、固定 P16／Broker-Cash，Seed=S1 `693545351`、S2 `2014432738`、S3 `104504529`、S4 `762252050`。2026 年報酬是截至 03/02 的部分年度，不是完整年度；Oracle 使用未來真值，**HINDSIGHT／NON-PIT**。
- **重播與 REUSE 證據**：正式選單第一次規劃 `REUSE 37 / RUN 27 / BLOCKED 0`；使用者授權後僅 S2/S3/S4 各執行 9 組價格型 Target（共 **27** 組；原始 Price Score **REUSE**，未重新訓練 DL、未執行 Optimizer）；完成後重新進入原選單為 `REUSE 64 / RUN 0 / BLOCKED 0`。四個 C96 及六個 MinTrade×4 Seed 均由原身份 REUSE，沒有因來源修訂全面重算。正式規劃器只證明這64格當前 artifact identity 可用，不等於完整市場逐日交易等價驗收。
- **來源修訂背景**：2026-10-27 price-target 精度／同分與來源閉包 Bug 修正使 S2～S4 的 `score_algorithm_sha256`、`truth_source_sha256` 舊身分失效；先前唯讀 V3 證據顯示 9 組已匹配現行來源、27 組 Score SHA／母體／價格資料／相同 Seed C96 身分一致但來源 SHA 不同。因此仍依正式 E13 規則重新執行這27格，**沒有覆寫歷史身份或放寬 REUSE**；新報表與重跑前報表的 16 組彙總值於顯示精度完全相同，不宣稱新舊逐筆交易 bytes 已獨立比較。

#### 正式報表：四 Seed 平均（直接保存使用者 2026-10-28 Console）

報酬率／年化／MDD 單位為百分比；RoMD、RoMD SD／Δ 為比值；EV(A) 單位為 R；`Win/Seed` 為**相對同 Seed C96**，不是任兩 Target 互比。

| Target | Return | Ann. | MDD | RoMD | RoMD SD | Δ vs C96 | Win/Seed | EV(A) |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| C96 Same-Window | +137.88% | +18.27% | 15.05% | 9.18 | 0.49 | +0.00 | - | +0.89 R |
| CG40_EqualRank | +310.60% | +31.02% | 13.41% | 23.22 | 8.35 | +14.05 | 4/4 | +1.53 R |
| MFE40 | +315.55% | +31.55% | 14.07% | 22.34 | 4.77 | +13.17 | 4/4 | +1.55 R |
| R40 | +290.85% | +30.23% | 13.27% | 21.92 | 1.42 | +12.74 | 4/4 | +1.52 R |
| T40 | +327.51% | +32.18% | 12.77% | 25.57 | 6.35 | +16.39 | 4/4 | +1.53 R |
| T40-MFE | +277.63% | +29.21% | 12.79% | 21.63 | 3.86 | +12.45 | 4/4 | +1.54 R |
| MFE40_Direct | +1339.25% | +65.59% | 14.36% | 98.76 | 56.20 | +89.58 | 4/4 | +2.41 R |
| R40_Direct | +1312.01% | +65.07% | 12.83% | 101.49 | 47.97 | +92.32 | 4/4 | +2.62 R |
| T40_Direct | +1398.61% | +66.78% | 11.71% | 116.71 | 51.78 | +107.53 | 4/4 | +3.32 R |
| T40-MFE_Direct | +1644.24% | +70.44% | 13.12% | 123.34 | 77.07 | +114.17 | 4/4 | +3.40 R |
| DailyUniversal_MinTrade40_NetR | +319.50% | +31.55% | 13.46% | 24.04 | 8.90 | +14.86 | 4/4 | +1.74 R |
| DailyUniversal_MinTrade40_NetR_Direct | +1989.40% | +76.58% | 11.99% | 164.00 | 106.79 | +154.82 | 4/4 | +4.17 R |
| DailyUniversal_MinTrade60_NetR | +327.26% | +32.38% | 12.73% | 25.60 | 3.37 | +16.42 | 4/4 | +1.80 R |
| DailyUniversal_MinTrade60_NetR_Direct | +1694.71% | +73.37% | 11.54% | 145.47 | 58.65 | +136.30 | 4/4 | +4.05 R |
| DailyUniversal_MinTrade_NetR | +314.91% | +31.65% | 13.54% | 23.38 | 4.12 | +14.20 | 4/4 | +1.73 R |
| DailyUniversal_MinTrade_NetR_Direct | +1369.92% | +66.62% | 11.09% | 122.76 | 57.31 | +113.58 | 4/4 | +3.73 R |

#### 同 Seed RoMD（正式報表原始四 Seed 明細）

| Target | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|
| C96 Same-Window | 9.57 | 9.00 | 9.58 | 8.56 |
| CG40_EqualRank | 35.57 | 17.16 | 20.18 | 19.99 |
| MFE40 | 28.68 | 18.26 | 23.34 | 19.10 |
| R40 | 23.40 | 20.00 | 22.23 | 22.05 |
| T40 | 34.41 | 24.25 | 24.30 | 19.29 |
| T40-MFE | 27.25 | 19.61 | 20.97 | 18.68 |
| MFE40_Direct | 171.75 | 41.52 | 110.21 | 71.56 |
| R40_Direct | 168.26 | 55.54 | 98.74 | 83.43 |
| T40_Direct | 189.08 | 70.72 | 116.60 | 90.44 |
| T40-MFE_Direct | 234.34 | 56.00 | 103.93 | 99.11 |
| DailyUniversal_MinTrade40_NetR | 36.61 | 19.84 | 23.50 | 16.20 |
| DailyUniversal_MinTrade40_NetR_Direct | 322.34 | 88.47 | 120.88 | 124.31 |
| DailyUniversal_MinTrade60_NetR | 29.91 | 22.68 | 26.61 | 23.19 |
| DailyUniversal_MinTrade60_NetR_Direct | 228.04 | 90.66 | 139.04 | 124.16 |
| DailyUniversal_MinTrade_NetR | 28.86 | 23.53 | 22.13 | 18.99 |
| DailyUniversal_MinTrade_NetR_Direct | 207.48 | 81.11 | 98.59 | 103.87 |

- **結果解讀**：Exact 的最高四 Seed 平均 RoMD 為 `DailyUniversal_MinTrade60_NetR=25.60 (SD 3.37)`，與 `T40=25.57 (SD 6.35)` 僅差 `0.03`，不足以視為穩健的統計優勢。Direct 最高為 `DailyUniversal_MinTrade40_NetR_Direct=164.00 (SD 106.79)`，四 Seed `322.34/88.47/120.88/124.31` 差異很大；Direct 使用另一種既有 Selector（No-K/No-R0），不應把全部 RoMD 差異歸因於 Target 的可學性或 DL 自身提升。皆為迭代研究 OOS 的事後比較，不作模型選擇或 Trading promotion。
- **附加驗證與未完成項**：先前使用者 `report.json` 之正式 DL 兩 Provider 與 Oracle 母體 `630,589/630,589` 逐股票日相同、`cache_minus_formal=0`／`formal_minus_cache=0`；舊 `643,617` 母體原始鍵集合仍未取得，不能斷定其差異根因。新 ZIP 的 Oracle／MinTrade 167 個獨立合成 unittest 於 GPT 端 PASS，**非** Windows 正式 `apps/run_bundle.py`／`apps/test_suite.py` 及非使用者完整市場逐日 E13 parity；本 ZIP 不含全部原市場 Replay 逐筆帳本／逐日 cash、positions、equity 工件，不能僅憑總 RoMD 相同與 `REUSE 64` 宣告端到端等價驗收完成。舊 Oracle/修復前數值、所有歷史 identity／Log 仍保留且不可倒填為當前有效結果；輸出真理仍是原 `outputs/audit/breakout_quality/reusable/target_oracle_romd/latest/audit.md`／對應正式 artifacts。
- **處置**：`USER_REPORTED_REAL_MARKET_REPLAY_COMPLETED / CURRENT_CANONICAL_REUSE_64_RUN_0_BLOCKED_0 / FOUR_SEED_AGGREGATES_VERIFIED_AT_DISPLAY_PRECISION / E13_FULL_MARKET_DAILY_PARITY_NOT_YET_CERTIFIED / HINDSIGHT_NON_PIT / ITERATIVE_OOS / NO_NEW_SCIENTIFIC_ID / NO_PROMOTION`。本次只登錄結果與同步 Registry 現況，不修改 Target、策略、費稅、計算程序、REUSE 判定或正式報表版型。

### 2026-10-28 — MinTrade Oracle 單股交易事件／延續單 SSOT 對齊修復（實作驗證，市場結果待重跑）

- **既有要求／錯誤定性**：40D／60D／完整週期 MinTrade Net-R 與完整週期 NetPnL 的原始問題是「依正式單股 Min 交易路徑，以完美事後分數評價每個 DL 可評分股票日的合法候選」，而不是每日創造一筆不存在的新單。舊 Producer 在所有股票日以當日 Close+ATR 新造訂單，沒有要求正式突破買訊，也未繼承原始延續單的限價／shadow；導致同一股票資訊日可能將正式賺錢的延續單評為虧損。原先以 Registry 的 daily-fresh 語句認定其正確，未能符合使用者已要求的單股交易對齊。此為 **E13 既有契約的實作修復**，不新增 Target 科學身分，不變動 Exact／Direct Selector、不追加 `NetPnL>0` 門檻，也不觸碰 Trading。
- **唯一交易路徑／修復方式**：`filters/breakout_quality/daily_trade40_target.py::compute_counterfactual_trade40_batch()` 依正式 P16 生效日期的 `build_signal_cache()`，用 `core.extended_signals` 的 canonical `create_signal_tracking_state()`／`should_clear_extended_signal()` 維持尚未被投組選擇成交的單股有效訊號；正式買訊時建立原始狀態，延續日沿用原始買訊日期、原始限價、原參數與 shadow，無訊號記 NULL（不另造 0R／0 元）。`trade_path_label.py::simulate_realized_trade_path_label()` 只新增接受 decision-time signal-state snapshot 的可選入口，後續成交、停損、出場、風控 sizing、費稅、毫元帳務及 closeout 繼續呼叫既有 canonical owner。40／60 交易所日、自然出場、Research 截止 2026-03-02 三種期限仍依原本最早規則，Full-Cycle 無固定 H；DL 可評分股票日母體不縮小。正式 Portfolio 仍擁有現金、已有持倉與 Selector 選股競爭。
- **SSOT 精確邊界**：同股票資訊日可能沒有可執行 Min 訊號，這時雖有合法 DL Score 鍵，**不能**宣稱存在一筆可獲利交易；記 NULL 並排在已合法成交經濟結果之後。延續評價以「此前尚未因投組選擇而成交」為單股條件，不使用 Portfolio 過往實際現金／持倉／成交來決定監督標籤。固定單股參考本金只隔離投組資金差異，仍使用同一入場規劃與精確帳本；NetPnL 不等同動態 Portfolio 實際投入金額。PIT／模型訓練一律不可把事後淨利或未來成交資訊作輸入。
- **獨立驗收證據（本輪）**：`synthetic_daily_trade40_cases`、`synthetic_daily_trade40_oracle_cases`、`synthetic_target_oracle_cases` 共 **197／197 PASS**。另有直接使用正式 `core.portfolio_engine.run_portfolio_timeline()` 的真實合成買訊驗收：新單與延續單的原買訊／成交／出場、NetPnL、Net-R 同步一致；延續單來源價格與原限價不被新資訊日重建。沒有買訊時 Target 保持 NULL；跨年 P16 原參數凍結與新有效訊號、錯誤快照 fail-fast 也通過。另非正式測試，不等同使用者 Windows Formal Suite 或全市場 E13 每日交易／現金／權益逐筆等價。
- **對舊正式績效的處理／REUSE**：之前 `2026-10-28` 已登錄的 MinTrade 40D／60D／Full-Cycle Net-R 及 Full-Cycle NetPnL 的四 Seed RoMD **全部保留為歷史 daily-fresh 錯誤實作證據，不再視為本次修復後現行有效結果**。因 Net-R 與 NetPnL 共用的單股起始交易身分確實改變，兩者 Producer 來源 SHA 必須更新，舊 8 targets × 4 seeds = 32 個正式 Portfolio Replay 在新的身份下原則應 **RUN 32**；價格型 9 × 4 及 C96 4 × 1 的 40 組不受本修正，預期保持 **REUSE 40**（以使用者本地正式規劃為準）。**禁止**使用先前為「僅加 NetPnL 輸出」建立的 Net-R 舊 SHA compatibility attestation；新的歷史成果不得由舊 daily-fresh receipt 直接補簽。原研究問題、報表 `audit.target_oracle_romd@v9` 及 payload schema 8 保持不變。
- **交付狀態**：`IMPLEMENTATION_CORRECTED / 197_SYNTHETIC_PASS / REAL_MARKET_32_REPLAYS_NOT_RUN_HERE / E13_FULL_MARKET_DAILY_PARITY_PENDING / HISTORICAL_DAILY_FRESH_RESULTS_ARCHIVED_NOT_CURRENT / NO_DL_FIT / NO_OPTIMIZER / TRADING_UNCHANGED`。

### 2026-10-29 — 修正正式買訊／延續單後的 MinTrade Oracle 四 Seed 市場報表登錄（使用者提供）

- **來源與證據層級**：使用者提供 `貼上的 Markdown(5).md`（SHA256 `4907da573718f5b647eebe637795dd4b04ca1d8962f8de0a0e263d2ca24a05b2`），以及 `test-branch-1_20261028_185950_667f8453(1).zip`（SHA256 `bfe6dd543352cd7ae8e476287b90b856bf85f06a64568f6532fc16fd5a9b11bd`）。這是 **USER_REPORTED_MARKET_REPORT**，非 GPT 自行執行四 Seed 市場 Replay；ZIP 不含 `outputs/`、完整市場資料或逐日交易／現金／部位／權益 receipts。
- **覆蓋的舊狀態**：本 Log 上一節「市場結果待重跑」是 2026-10-28 修補交付時的歷史狀態。本次已有修補後四 Seed 績效報表，故將**績效報表待產生**更新為**使用者已提供**；**不代表**已核對 32 個 RUN 的 signed receipt／當前 72 個逐 Arm REUSE，亦不代表 E13 逐日等價驗收完成。更早 `2026-10-28` 的 daily-fresh 64/64 REUSE 及其 MinTrade 績效只能當歷史證據，不得冒充修正後有效結果。
- **不變的正式研究條件**：Research `[1] Target 驗證`，報表 `audit.target_oracle_romd@v9`／payload schema `8`；`2021-01-01～2026-03-02`、630,589 個共同股票日、四個 P16 Seed `693545351 / 2014432738 / 104504529 / 762252050`。C96＋17 Oracle treatments（9 價格型、8 MinTrade，計 18 列），Exact K/R0／Direct No-K/No-R0，原 Broker-Cash、參數／Selector／百分位映射不變。2026 為截至 03/02 的**部分年度**；全部 Target 是 hindsight/non-PIT，不能用於當日交易或作模型 fitting。

#### 修正後 MinTrade 四 Seed 平均（直接轉錄使用者正式人讀報表）

| Target | Return | Ann. | MDD | RoMD | RoMD SD | EV(A) |
|---|---:|---:|---:|---:|---:|---:|
| `DailyUniversal_MinTrade40_NetR` | +345.76% | +33.28% | 12.58% | 27.51 | 7.38 | +1.92 R |
| `DailyUniversal_MinTrade40_NetR_Direct` | +1231.92% | +61.70% | 11.88% | 99.96 | 57.68 | +4.11 R |
| `DailyUniversal_MinTrade60_NetR` | +282.41% | +29.69% | 12.76% | 22.23 | 2.18 | +1.89 R |
| `DailyUniversal_MinTrade60_NetR_Direct` | +1293.06% | +62.90% | 10.82% | 117.77 | 76.20 | +4.23 R |
| `DailyUniversal_MinTrade_NetR` | +311.69% | +31.22% | 12.99% | 24.63 | 8.20 | +2.09 R |
| `DailyUniversal_MinTrade_NetR_Direct` | +1343.07% | +63.99% | 11.56% | 114.26 | 69.81 | +4.07 R |
| `DailyUniversal_MinTrade_NetPnL` | +325.49% | +32.17% | 12.58% | 25.81 | 5.17 | +1.67 R |
| `DailyUniversal_MinTrade_NetPnL_Direct` | +1520.29% | +68.61% | 11.69% | 131.66 | 74.11 | +3.42 R |

#### 完整週期 Net-R／NetPnL 同 Seed 對照

| Seed | Net-R Exact | NetPnL Exact | Net-R Direct | NetPnL Direct | Direct 差（PnL−R） |
|---|---:|---:|---:|---:|---:|
| S1 `693545351` | 33.98 | 32.65 | 206.66 | 224.99 | +18.33 |
| S2 `2014432738` | 15.47 | 22.10 | 43.78 | 51.04 | +7.26 |
| S3 `104504529` | 28.44 | 26.97 | 124.19 | 149.63 | +25.44 |
| S4 `762252050` | 20.64 | 21.52 | 82.40 | 100.97 | +18.57 |
| **四 Seed 平均** | **24.63** | **25.81** | **114.26** | **131.66** | **+17.40** |

#### 與修補前 daily-fresh 歷史結果之差（非同算法受控比較）

| MinTrade Target | 舊 RoMD（historical） | 修正後 RoMD（user-reported） | 顯示值差 |
|---|---:|---:|---:|
| `DailyUniversal_MinTrade40_NetR` | 24.04 | 27.51 | +3.47 |
| `DailyUniversal_MinTrade40_NetR_Direct` | 164.00 | 99.96 | -64.04 |
| `DailyUniversal_MinTrade60_NetR` | 25.60 | 22.23 | -3.37 |
| `DailyUniversal_MinTrade60_NetR_Direct` | 145.47 | 117.77 | -27.70 |
| `DailyUniversal_MinTrade_NetR` | 23.38 | 24.63 | +1.25 |
| `DailyUniversal_MinTrade_NetR_Direct` | 122.76 | 114.26 | -8.50 |
| `DailyUniversal_MinTrade_NetPnL` | 24.73 | 25.81 | +1.08 |
| `DailyUniversal_MinTrade_NetPnL_Direct` | 123.14 | 131.66 | +8.52 |

- **重要讀法**：Full-Cycle NetPnL Direct `131.66`，相對 Full-Cycle Net-R Direct `114.26`，四 Seed 分別增加 `18.33 / 7.26 / 25.44 / 18.57`，**4/4 Seed 同向**；在本次 17 組 Oracle treatment 中四 Seed 平均 RoMD 最高。但 NetPnL Direct 對價格型 `T40-MFE_Direct=123.34` 並非 4/4 都領先；Direct 的 NetPnL RoMD SD `74.11`，不應解讀為穩健可部署的模型或全期間資金最優。Exact 的四 Seed 平均為 `25.81` 對 `24.63`，但同 Seed 僅 **2/4** 較好；本次 Exact 全表最高為 `MinTrade40_NetR=27.51`。
- **舊結果的解讀修正**：修補前 `MinTrade40_NetR_Direct=164.00`，修補後為 `99.96`；因 Target 起點從「每日假設新單」改為「實際 Min 買訊／延續狀態」，這些績效不可當作同算法的純參數或 Target 排名提升／退步。C96 `9.18` 與價格型 `T40-MFE_Direct=123.34` 維持原顯示值，是有限的非 MinTrade 同窗對照，**不能替代** signed source／score／Replay 身分核對。
- **驗證狀態**：直接解析報表 18 列（C96＋17 treatments），17 組 treatment ×4 Seed 的 RoMD mean 與樣本 SD 全部在 `±0.015` 的兩位數顯示精度容許範圍內一致；上節 197/197 合成與單股引擎對照係修補交付時工程證據，不是本次的真實市場逐日 E13 驗收。此 ZIP 無完整 receipt/ledger/position/equity sidecars；不得宣稱使用者 Windows Formal Test Suite PASS、72/72 REUSE、或正式 DL／Oracle score 入口逐日獨立 parity PASS。
- **現行處置**：`USER_REPORTED_POST_SIGNAL_STATE_FOUR_SEED_RESULTS_AVAILABLE / EIGHT_MINTRADE_TREATMENTS_REPORTED / OLD_DAILY_FRESH_RESULTS_HISTORICAL_INVALID_FOR_CURRENT / REUSE_72_NOT_INDEPENDENTLY_ATTESTED / E13_FULL_MARKET_DAILY_PARITY_PENDING / NON_PIT / ITERATIVE_OOS / NO_SCIENTIFIC_ID_CHANGE / NO_REPORT_VERSION_CHANGE / NO_PROMOTION`。本次只登錄結果與同步 Registry，**不**修改正式 Target、交易核心、選單、Report Contract、費稅、模型、Replay 工件或 `doc/CMD.md`。

## 4. 2026-10-29～30 Oracle 最新實測與 Positive Gate 歸因（歷史 NON-PIT）

### 2026-10-29 · 60-session NetPnL Exact／Direct（四 Seed user-reported）

與同一 60-session MinTrade Net-R 的交易／結算路徑一致，但使用 broker-ledger 參考 NetPnL 做每日相對排序；60-session 與 full-cycle 結算期限不同，不是原版同一筆交易的單純參數比較。

| Target | Return Mean | RoMD Mean | Win Rate | RoMD SD |
|---|---:|---:|---:|---:|
| `DailyUniversal_MinTrade60_NetPnL` | +309.53% | 21.42 | 54.40% | 2.28 |
| `DailyUniversal_MinTrade60_NetPnL_Direct` | +1719.69% | 137.77 | 72.14% | 87.12 |

期間 2021-01-01～2026-03-02，共同股票日 630,589，四個 P16 seeds：693545351／2014432738／104504529／762252050；來源為使用者 2026-10-29 正式 Oracle 報表，不冒稱獨立逐日 E13 parity 核對已完成。

### 2026-10-30 · `DailyUniversal_MinTrade_NetPnL_Positive_Direct`（完整週期 `NetPnL > 0`，NON-PIT 額外資格研究對照）

- 四 Seed Return Mean **+2001.37%**，Ann. **+78.71%**，MDD **9.97%**，RoMD **199.52**（SD **82.53**），Win Rate **99.71%**，EV(A) **+4.38R**；原完整週期無正獲利門檻 Direct RoMD **131.66**。**此項額外使用未來正負資訊，因此不能視為 Score-only E13 完美預測或 PIT 可執行策略**。
- 已完成 **701** Round-Trips：699 勝、2 敗、0 平。正獲利資格違規（NULL／非正仍買入）**0**；與參考進場事件不同 **21**、股數不同 **637**、最終出場日／價不同 **1**（因素可能重疊）；只列完整 Round-Trips，不推測未成交為虧損。
- Seed 693545351：202/202 勝；2014432738：155/155 勝；104504529：169/170 勝；762252050：173/174 勝。
- **4114／Seed104504529**：參考淨利 **+1443.40 元**、Portfolio **−1454 元**。參考於 2022-07-19 以 22.20 買入，Portfolio 於 2022-07-15 以 23.35 買入；參考 7/15 為 `ZERO_SIZED_CANDIDATE`，與固定參考本金下最低進場金額限制、動態 Portfolio 資金使下單量不同的機制吻合；不假裝已完成行情／P16 所有逐欄獨立認證。
- **00770／Seed762252050**：參考 **2,000 股、+1 元**，Portfolio **22,000 股、−10 元**；相同價格路徑在正式券商費稅整數取整後可翻轉獲利正負號。兩筆均非 NULL 漏過門檻。
- 一次性五因素 Audit **已結案並退役 implementation**；上述數字是當時使用者提供的 Market Replay／Audit 結果，不意味長期部署資格；詳細 signed replay 與既有 `outputs/audit/breakout_quality/one_time/fullcycle_positive_direct/latest/` 為原有研究工件，不是第二份日誌。

### 2026-10-30 · `[1] Target 驗證` `end_to_end_v2` 八 Seed 結果（使用者提供；現行比較基準）

- **來源**：使用者 `貼上的 Markdown(8).md`，SHA256 `911a94fbca917b0c854a249ec514a112b53282fed1ad85185a3bf5faafac1780`；此處只登錄重要結果，原報表保留逐年與 160 筆 Target×Seed 的 RoMD。正式結果原有入口為 `outputs/audit/breakout_quality/reusable/target_oracle_romd/latest/audit.md`（本 ZIP 不含該輸出與簽章 Replay 收據，不能自行驗證其位元組一致性）。
- **固定口徑**：`AUD-target-ground-truth-romd`／`audit.target_oracle_romd@v11`、payload schema 8；`2021-01-01～2026-03-02`（2026 非全年）、共同 feature-eligible 股票日 **630,589**、`end_to_end_v2` 的八個 P16 parameter Seeds `693545351 / 2014432738 / 104504529 / 762252050 / 9734312 / 1020806211 / 116293459 / 1234683984`。C96 同窗基準＋20 treatments（Exact K/R0 及 Direct No-K/No-R0，含一項特殊 Positive_Direct）；正式 Broker-Cash、原 Oracle percentile Score／Target 定義不變。Hindsight **NON-PIT**，OOS 為迭代研究證據。

#### 八 Seed 彙總績效（RoMD Mean 小→大；直接轉錄使用者報表）

| Target | Return Mean | Ann. Mean | MDD Mean | RoMD Mean | RoMD SD | Win Rate | EV(A) Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C96 Same-Window` | +130.87% | +17.29% | 14.27% | 9.17 | 3.16 | 43.86% | +0.85 R |
| `R40` | +311.61% | +31.50% | 12.94% | 24.14 | 2.90 | 52.53% | +1.55 R |
| `DailyUniversal_MinTrade60_NetPnL` | +322.80% | +32.19% | 13.46% | 24.47 | 5.11 | 55.10% | +1.72 R |
| `MFE40_Control` | +331.28% | +32.48% | 13.65% | 24.62 | 6.84 | 52.68% | +1.50 R |
| `CG40_EqualRank` | +303.73% | +30.63% | 12.15% | 25.22 | 7.54 | 54.45% | +1.51 R |
| `DailyUniversal_MinTrade60_NetR` | +326.08% | +32.30% | 12.26% | 27.07 | 6.78 | 57.05% | +1.96 R |
| `DailyUniversal_MinTrade40_NetR` | +335.69% | +32.80% | 12.27% | 27.59 | 6.18 | 57.96% | +1.98 R |
| `DailyUniversal_MinTrade_NetR` | +327.66% | +32.33% | 12.09% | 27.67 | 6.50 | 55.80% | +2.04 R |
| `DailyUniversal_MinTrade_NetPnL` | +323.02% | +31.84% | 11.56% | 28.20 | 8.42 | 54.89% | +1.67 R |
| `T40-MFE` | +330.91% | +32.33% | 11.97% | 28.23 | 9.71 | 54.65% | +1.58 R |
| `T40` | +343.71% | +33.23% | 12.23% | 28.34 | 6.62 | 53.75% | +1.59 R |
| `MFE40_Direct` | +1418.14% | +67.50% | 14.42% | 102.38 | 50.14 | 60.63% | +2.54 R |
| `R40_Direct` | +1381.28% | +66.93% | 12.89% | 108.08 | 46.88 | 62.02% | +2.75 R |
| `DailyUniversal_MinTrade40_NetR_Direct` | +1298.68% | +64.59% | 11.58% | 112.09 | 52.40 | 75.41% | +4.02 R |
| `DailyUniversal_MinTrade_NetR_Direct` | +1348.74% | +65.56% | 11.12% | 124.42 | 66.23 | 74.03% | +4.11 R |
| `DailyUniversal_MinTrade60_NetR_Direct` | +1300.05% | +64.14% | 10.55% | 126.21 | 74.58 | 76.13% | +4.18 R |
| `T40-MFE_Direct` | +1675.23% | +71.50% | 12.46% | 134.80 | 74.86 | 73.66% | +3.50 R |
| `T40_Direct` | +1509.19% | +68.88% | 11.35% | 137.44 | 81.59 | 72.82% | +3.53 R |
| `DailyUniversal_MinTrade_NetPnL_Direct` | +1655.37% | +71.98% | 11.79% | 146.54 | 82.19 | 72.24% | +3.43 R |
| `DailyUniversal_MinTrade60_NetPnL_Direct` | +1852.77% | +75.67% | 12.47% | 152.95 | 80.90 | 73.88% | +3.49 R |
| `DailyUniversal_MinTrade_NetPnL_Positive_Direct` | +2073.62% | +79.91% | 10.18% | 204.88 | 88.26 | 99.71% | +4.36 R |

- **名稱對應**：來源報表的 `MFE40` 對應正式選單的 `MFE40_Control`，僅為顯示名稱差異。
- **配對及變動**：20 個 Target 皆為 `8/8` Seed RoMD 高於 same-seed C96（非交易 Win Rate）。四 Seed→八 Seed RoMD：C96 `9.18→9.17`、T40 `25.57→28.34`、`MinTrade40_NetR 27.51→27.59`、`MinTrade60_NetPnL_Direct 137.77→152.95`、Positive_Direct `199.52→204.88`。只是擴充四個 P16 replicate 後的**跨 Seed 彙總變化**，不是更換 Target/演算法後的績效增量；舊四 Seed 報表與其獨立歷史身分保留。
- **研究判讀／決策**：一般 Exact 以 `T40` RoMD `28.34`（SD `6.62`）最高，僅比 `MinTrade40_NetR` `27.59` 高 `0.75`，不足以單憑名次宣稱穩健優勢；不含 Positive 未來入場資格的 Direct 以 `MinTrade60_NetPnL_Direct` `152.95`（SD `80.90`）最高，跨 Seed 分散明顯。`DailyUniversal_MinTrade_NetPnL_Positive_Direct` 雖達 RoMD `204.88`、Win Rate `99.71%`，**額外以未來 NetPnL>0 拒單**，不是純 Score-only Oracle、更不是可部署 PIT 策略；不得與一般 Direct 當成同條件的可學 Target 上限。
- **核對與限制**：直接解析新報表的 **21** 個 aggregate（C96＋20 treatments）、20×8＝**160** 筆 paired Seed RoMD；20 組 RoMD mean／sample SD 與八個 Seed 順序皆在顯示精度內一致，C96 same-seed 值一致。這只驗證**報表算術與配對**，不代表使用者本機 168/168 REUSE 已經獨立認證，也不能取代逐筆 Portfolio Ledger、score identity／signed replay 或正式 DL↔Oracle **E13 每日等價驗收**。不因此修改 Target、模型、Trading、報表版本、科學身分或作 promotion。

**結果狀態**：`USER_REPORTED_EIGHT_SEED_MARKET_REPORT_AVAILABLE / TWENTY_TREATMENTS_8_OF_8_ROMD_OVER_C96 / ALL_160_SEED_PAIR_AGGREGATES_ARITHMETIC_VERIFIED / PHYSICAL_REPLAY_RECEIPTS_NOT_IN_UPLOAD / E13_FULL_MARKET_PARITY_PENDING / HINDSIGHT_NON_PIT / ITERATIVE_OOS / NO_PROMOTION`。

## 5. 後續新增規則

**只有新科學證據或舊科學證據有效性更正才能增加本文件內容。** 記錄必須有實驗／Target 身分、期間／Seed、控制變數、原始結果、負結果與決策，並指向正式證據。沒有市場／模型結果的工程測試、提交紀錄、打包、格式調整或一次性 Audit 清理，不可追加研究章節。本文件不再生成封存副本或研究待辦。

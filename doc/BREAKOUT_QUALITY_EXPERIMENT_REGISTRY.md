# Breakout Quality 實驗編號與版本 Registry

## 1. 用途

本文件是 `breakout_quality` 的**實驗 identity、命名、版本語意、namespace、目前基準與 ID 佔用狀態**單一真理來源。

凡進行 breakout-quality 分析、設計、實作、比較、Audit 或新增實驗名稱，開始前固定依序讀取：

1. `doc/PROJECT_SETTINGS.md`
2. `doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md`
3. `doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md`

Registry 回答「**這個 ID 是什麼、屬於哪一層、是否已被占用**」；Experiment Log 回答「**做了什麼、結果如何、為什麼採用或淘汰**」。歷史 ID 一經使用即永久保留，不得回收重用。

---

## 2. Canonical namespace

| Namespace | 定義 | 例子 | 編號規則 |
|---|---|---|---|
| `MR-*` | 模型、學習目標、target-learning 等研究實驗 identity | `MR-9A`、`MR-10A`、`MR-11G` | 既有數字＋字母系列永久保留；只有真正的模型／學習研究才取得此 ID。 |
| `ARCH-*` | 神經網路 architecture 或 input representation identity | `ARCH-inception_time_v1` | `_vN` 只在 architecture 或輸入表示改變時增加。 |
| `PROFILE-*` | Training experiment profile | `PROFILE-unique_group_sampling` | optimizer、LR schedule、augmentation、sampling、loss weighting 等放在 profile，不建立新 architecture version。 |
| `DL-*` | 策略 runtime 使用的 DL source identity | `DL-A9`、`DL-TP1` | 綁定具體 model／label／threshold／score source。程式可保留 `A9`、`TP1` alias，相同文件若可能混淆必須使用 canonical prefix。 |
| `SR-C*` | Strategy runtime／portfolio 使用方式 arm | `SR-C11`、`SR-C12` | 既有 DL 的 refresh timing、gate、ranking、allocation 等使用方式變更全部放這一層。報表可繼續顯示 `C11`、`C12`。 |
| `PARAM-P*` | 策略參數訓練／工件 stage | `PARAM-P2`、`PARAM-P3` | 代表參數 stage，不是 model experiment，也不是策略 runtime arm。 |
| `AUD-*` | Read-only Audit／attribution identity | `AUD-a9-pass-quality` | 使用 `config/audit.py` profile slug；Audit 不占用 `MR-*` 或 `SR-C*`。 |
| `DATA-*` | Dataset identity | `DATA-breakout_quality_v1` | row identity、feature-bank dataset semantics 改變時才改版。 |
| `LABEL-*` | Label／Target identity | `LABEL-a2_realized_trade_path_v1` | Target 語意改變時才改版。 |

### 命名硬規則

1. **不得依對話記憶直接編號；任何新 ID 都必須先查本 Registry。**
2. 已使用 ID 即使 `REJECTED`、`STOPPED` 或 legacy-only，也**永久不得重用**。
3. 只改既有 DL 的 score refresh timing、gate、ranking 或 allocation，屬 `SR-C*`，**不是新 `MR-*` model experiment**。
4. 新模型訓練、architecture、target-learning 或 training-data semantics 的受控研究才屬 `MR-*`。
5. Audit 使用 `AUD-*`；一般 infrastructure／bug fix 不占 scientific experiment ID，除非該變更本身就是受控研究變數。
6. Architecture `_vN` 只代表結構／輸入表示差異；optimizer、LR、augmentation、loss weighting、sampling、threshold、runtime usage 不得 bump architecture version。
7. 有歧義的 bare ID 必須展開。例如同段文字應寫 `MR-9A` 與 `DL-A9`，不得只靠 `9A`／`A9` 猜語意。
8. Registry 與 Experiment Log 若 identity、名稱或狀態衝突，當輪必須先修正一致後才能新增實驗。

---

## 3. 目前 canonical identity

| 層級 | Canonical identity | 目前語意 | 狀態 |
|---|---|---|---|
| Dataset | `DATA-breakout_quality_v1` | 既有 300×10 breakout-quality feature-bank dataset | ACTIVE |
| Main Label | legacy fixed-percentage breakout-event label | A9 使用的 MFE／MAE 固定百分比 breakout opportunity Label；目前沒有獨立 `label_id` | ACTIVE |
| Model research | `MR-9A` | InceptionTime 研究實驗；architecture=`inception_time_v1` | ACCEPTED model-quality baseline |
| Architecture | `ARCH-inception_time_v1` | A9 active architecture | ACTIVE |
| Training profile | `PROFILE-unique_group_sampling` | Unique-group training profile | ACTIVE |
| Runtime DL source | `DL-A9` | `DATA-breakout_quality_v1 / ARCH-inception_time_v1 / PROFILE-unique_group_sampling / threshold 0.5` | ACTIVE research DL source |
| Alternative DL source | `DL-TP1` | `LABEL-a2_realized_trade_path_v1` realized trade-path binary source | HARD-FILTER REJECTED；保留歷史重現 |
| Strategy params baseline | `PARAM-P2 / Min ROOS` | rules全關、DL-off；每fold只搜尋`high_len`＋`atr_len/atr_buy_tol/atr_times_init/atr_times_trail`，其餘optimizer維度由canonical config/schema固定；trials直接讀`config/training_policy.py`；Outer Rolling首尾邊界以month-bucket語意驗證，Strategy Compare月底period end與optimizer月初canonical boundary不得被誤判為不同schedule | ACTIVE strategy parameter baseline／2026-08-09 semantic correction；舊4-ATR且high_len凍結工件只供歷史重現，不得作current Min ROOS cache；完成optimizer後若僅post-run validation/stamp中止，current preflight identity與完整五欄參數契約一致時可直接接續，不得重跑fold |
| Resource-aware risk-quality reference | `SR-C15` | All-event continuous capital-utilization-first | RESULT_AVAILABLE／PROMISING_NOT_PROMOTED；C16雖有更高Return，但C15仍有較低MDD與較高RoMD／EV；AUD-c15確認其相對C3/C12優勢高度集中2024，因此仍不升格正式基準 |
| 最大化 PASS 研究基準 | `SR-C12` | A9 resource-aware best-improvement basket | ACTIVE；研究方向固定為「最大化 PASS 使用，再提高 PASS 品質」 |
| 最新已完成Audit結果 | `AUD-cross-period-year-regime-attribution` | `MR-13A` vs `MR-12B` Selection PIT + Forward-OOS / same 8 seeds | RESULT_AVAILABLE／RANKING_EDGE_DIRECTION_REVERSAL／BROAD_SELECTION_PERIOD_WEAKNESS／MR13A_NOT_PROMOTED。Selection逐年度direct-pair Exclusive ΔR為5負2正，Forward為1負5正；Selection worst year=2014 `-13.96R`，只占Selection負年度R的`42.3%`，故edge reversal不是單一年份可解釋。下一步不建立year/regime gate；Daily Score仍是長期共同DL方向。MR-13B前先完成continuous ranker的profile-driven多DL架構整理，再以新的獨立MR研究daily-universal training semantics；MR-12B daily-universal inference bridge降為可選causal control，不作主線。 |
| Latest cross-period robustness | `MR-13A` vs `MR-12B` / same generated 8 seeds | Selection PIT 2014-01-01～2020-12-31 vs Forward-OOS 2021-01-01～2026-03-02 | RESULT_AVAILABLE／CROSS_PERIOD_DIRECTION_REVERSAL／YEAR_REGIME_AUDIT_COMPLETE。Selection PIT：MR-13A−MR-12B ΔDL選擇R Mean=`-20.76R`（3勝5敗）、ΔRoMD Mean=`-0.93`（2勝6敗）、方向一致=`7/8`；Forward：ΔDL選擇R Mean=`+50.64R`（6勝2敗）、ΔRoMD Mean=`+1.76`（4勝4敗）、方向一致=`4/8`。逐年度direct-pair Audit顯示Selection為5負2正、Forward為1負5正，Selection worst year=2014 `-13.96R`且只占負年度R `42.3%`，因此不是單一regime outlier；current runtime anchor維持MR-12B。 |
| Continuous research DL source | `DL-CONT11G` | `MR-11G / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_pass_magnitude_mse` frozen OOS continuous score | HISTORICAL controlled-replay source；SR-C14未採用 |
| Current model research | `MR-13E` | Daily Universal Full-list Delta-NDCG-weighted Pairwise Ranker / `daily_universal_no_time_full_list_ndcg_pairwise`；固定MR-13A全部daily-universal data／Target／architecture／optimizer／epoch-selection，唯一scientific change為同日RankNet pair依目前預測完整榜單交換造成的raw-percentile `abs(Delta NDCG)`加權 | RESULT_AVAILABLE／FORWARD_OOS_MODEL_GATE_PASS／SELECTION_PIT_MODEL_GATE_PASS／CROSS_PERIOD_SINGLE_SEED_MODEL_STABILITY_SUPPORTED／STRATEGY_GATE_RUNTIME_SOURCES_CONFIGURED／RUNTIME_CONTRACT_VALIDATOR_REPAIRED／AWAITING_SINGLE_SEED_STRATEGY_GATE。Seed42 Selection PIT 2013-04-01～2020-12-31：coverage=`100%`、all-stock daily/global rho=`0.1627/0.0533`、pair=`55.75%`、Top-K Lift=`+0.4456R`，年度rho/spread皆`8/8`為正；breakout daily/global rho=`0.1019/0.0833`、pair=`54.77%`、Top-K Lift=`+0.1513R`。Forward all-stock daily rho=`0.2151`、Top-K Lift=`+0.9951R`；breakout daily rho=`0.1721`、Top-K Lift=`+0.4489R`。相較13C/13D，13E在shared PIT的Daily rho、Pair、Top-K與breakout主要指標整體最強，且Selection→Forward未再出現MR-13A的方向反轉。`drift=True`僅為fold score-mean位移warning，不是Gate veto。下一步只授權建立13E runtime source並做controlled strategy compare；尚未promotion，`MR-12B / DL-CONT12B`仍為CURRENT_RUNTIME_ANCHOR。 |
| Daily DL research infrastructure | `ContinuousRankerResearchSpec` / profile-driven dispatcher | Continuous experiment profile只描述scientific profile；額外research metadata集中保存`model_research_id / trainer_family / score_semantic_id / pairwise_reduction`，event與daily trainer共享learning API但各自擁有sample-universe orchestration | IMPLEMENTED／MR13C-D-E_PIT_BATCH_COMPLETE／PROFILE_DRIVEN。MR-12B與MR-13A固定`equal_pair_weight`；MR-13B歷史gap weighting已淘汰；MR-13C重用canonical percentile-MSE；MR-13D=`upper_tail_relevance_weighted`；MR-13E=`full_list_delta_ndcg_weighted`。13C/13D/13E已使用同一Dataset／Target／Seed／period／fold contract完成Selection PIT Model Gate，三者Gate皆PASS；13E在shared PIT的all-stock Daily rho=`0.1627`、Pair=`55.75%`、Top-K Lift=`+0.4456R`與breakout Daily rho=`0.1019`均為三者最佳，且年度rho/spread=`8/8`正。Batch只並列canonical audit原始指標，未建立人工總分或挑seed。Strategy arm仍只綁一個`dl_id`；trainer artifact producer與runtime OOS validator現共用`ranker_training_contract.training_semantics()`，非default pairwise reduction不再被舊`equal_pair_weight` hard-code誤判；下一步由13E單獨進strategy gate。 |
| Previous continuous DL anchor | `DL-CONT12A` | `MR-12A / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_mse` frozen OOS continuous score | SOURCE_SUPPORTED historical anchor；保留作MSE基準 |
| Current continuous DL anchor | `DL-CONT12B` | `MR-12B / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_pairwise` frozen OOS continuous score | CURRENT_RUNTIME_ANCHOR／CROSS_PERIOD_ROBUSTNESS_FAVORS_KEEPING_ANCHOR。MR-13A在Forward 8-seed有selection edge但RoMD僅4勝4敗；Selection PIT 8-seed又反轉為MR-13A ΔDL選擇R Mean=`-20.76R`、3勝5敗，ΔRoMD Mean=`-0.93`、2勝6敗。故DL-CONT12B繼續維持canonical runtime anchor；不得以單一seed、單一時段Mean或Forward Audit平均PnL promotion MR-13A。Cross-period Audit已排除單一年份解釋；runtime anchor仍維持MR-12B，但model research主線維持Daily Score。MR-13B前先完成profile-driven多DL架構整理；之後新的Daily MR仍須通過Selection／Forward與multi-seed gate才可取代anchor。 |
| Current Selection PIT event source | `DL-CONT12B-PIT` | `MR-12B / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_pairwise` Selection point-in-time score | RESULT_AVAILABLE／PIT_GATE_PASS／CURRENT_FIVE_FIELD_MIN_ROOS_REPLAY_COMPLETE／NOT_PROMOTED。Current C24/C25同批五欄Min ROOS結果：C24 Return=82.98%、MDD=18.99%、RoMD=4.37、EV=0.50R、selection R=+12.19R；C25 Return=101.55%、MDD=19.66%、RoMD=5.17、EV=0.52R、selection R=+26.19R；兩者仍未超越C23 Return=107.98%、RoMD=5.25。舊4-ATR結果與其Audit保留read-only attribution，不得與current五欄結果混用 |
| Previous Selection runtime microtuning | `SR-C26` | 舊4-ATR universe下`SR-C25` feasible-ascent + Selection-frozen 22-calendar-day stale-score membership guard；candidate保留、只禁止stale score驅動membership change | RESULT_AVAILABLE／HISTORICAL_ONLY／NOT_PROMOTED／RUNTIME_MICROTUNING_STOPPED；C26與其Audit只解釋舊4-ATR Selection universe，不得套用到current五欄C23～C28結果；22日guard不繼承到C28／C29 |
| Current Selection PIT daily source | `DL-CONT13A-PIT` | `MR-13A / ARCH-inception_time_v1 / PROFILE-daily_universal_no_time_pairwise` daily Selection PIT score | RESULT_AVAILABLE／PIT_MODEL_GATE_PASS／STRATEGY_RUNTIME_READY／SINGLE_SEED_SELECTION_GATE_PASS／MULTI_SEED_RELATIVE_EDGE_NOT_CONFIRMED。策略端只接受`score_eligibility_contract=feature_history_only`的新PIT manifest，每個盤前決策查最新已完成交易日score。既有current五欄單seed C28-C25 source-only Return `+50.33pp`、RoMD `+2.89`、selection R `+83.72R`仍保留為有效歷史觀察；但正式8-seed Selection PIT robustness中MR-13A相對MR-12B的ΔDL選擇R只有`3/8`勝、Mean=`-20.76R`，ΔRoMD只有`2/8`勝、Mean=`-0.93`，故不得再以單seed Selection gate外推為跨seed source superiority。 |
| Latest Selection strategy translation | `SR-C27 / SR-C28` | C27完全沿用C24 minimum-repair、C28完全沿用C25 feasible-ascent；唯一DL source改為`DL-CONT13A-PIT`，不繼承C26 22-day stale guard | RESULT_AVAILABLE／C27_NOT_PROMOTED／C28_SINGLE_SEED_GATE_PASS／MULTI_SEED_EDGE_NOT_STABLE。既有單seedC28-C25 Return `+50.33pp`、MDD `-0.80pp`、RoMD `+2.89`、EV `+0.19R`、selection R `+83.72R`仍有效，但8-seed Selection PIT robustness的MR-13A vs MR-12B平均Return=`102.98% vs 117.08%`、RoMD=`5.03 vs 5.96`、DL選擇R=`43.32R vs 64.08R`，matched ΔRoMD=`-0.93`、matched ΔDL選擇R=`-20.76R`。因此C28不再提供跨seed promotion證據，亦不回頭調selector／stale cutoff／capital objective。 |
| Current Selection strategy matrix | `SR-C32 / C23 / C25 / C28 / C35` | Current單seed策略Gate顯示`Full ROOS`、`Min ROOS`、`Min MR-12B`、`Min MR-13A`、`Min MR-13E`；三個DL arms固定相同historical Min params與feasible-ascent，只有score source不同。 | IMPLEMENTED／MR13E_SINGLE_SEED_GATE_PENDING。既有12B/13A 8-seed結果維持原scientific fingerprint；新C35 `robustness_role=off`，不得污染或自動擴充既有robustness。先比較C35-C25與C35-C28，再決定是否授權13E multi-seed。 |
| Current Forward-OOS strategy matrix | `SR-C1 / C3 / C20 / C29 / C36` | Current單seed策略Gate顯示`Full ROOS`、`Min ROOS`、`Min MR-12B`、`Min MR-13A`、`Min MR-13E`；三個DL arms固定相同current Min params與feasible-ascent，只有score source不同。 | IMPLEMENTED／MR13E_SINGLE_SEED_GATE_PENDING。既有12B/13A 8-seed結果與anchor判定不變；新C36 `robustness_role=off`。先比較C36-C20與C36-C29，若Selection/Forward都支持13E才授權下一階段8-seed。 |
| Rejected continuous DL source | `DL-CONT12C` | `MR-12C / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_listwise` frozen OOS continuous score | RESULT_AVAILABLE／REJECTED_FOR_MODEL_RESEARCH；C21略弱於C19，C22相對C20大幅退步；保留歷史重現，不作current anchor |
| Current strategy validation harness | `SR-C17 / SR-C18` | 同一K/R0與Min ROOS execution order的兩個固定selector：C17 minimum-repair、C18 feasible-ascent | FROZEN_FOR_MODEL_COMPARISON；MR-12A用C17/C18、MR-12B用C19/C20、MR-12C用C21/C22套完全相同兩個selector，同時驗證model-only gain與較Max selector的轉化；不得依新模型OOS修改C17/C18 |
| 最新model-validation策略結果 | `SR-C21 / SR-C22` | `DL-CONT12C / MR-12C`分別套C17／C18固定selector | RESULT_AVAILABLE／MODEL_REJECTED；C21 Return=178.68%、MDD=16.22%、RoMD=11.02、EV=0.67R、selection R=+14.09R；C22 Return=116.51%、MDD=16.49%、RoMD=7.07、EV=0.45R、selection R=-59.48R。相對MR-12B anchors，C21-C19 Return -3.08pp／selection R -10.95R，C22-C20 Return -63.03pp／RoMD -5.33／EV -0.39R／selection R -114.79R；ListNet未改善Pairwise，且較Max-DL C18強烈放大其ranking弱點 |

### `MR-9A` 與 `DL-A9` 必須分開

- `MR-9A`：建立／驗證 InceptionTime 的**模型研究實驗**。
- `DL-A9`：策略比較所使用的**runtime DL source alias**。
- 兩者有來源關係，但不是同一個 namespace，也不能互相當作下一個實驗編號。

---

## 4. 已占用 Model／Research ID

下列歷史 ID 均已永久占用；詳細設計、數值與判定以 `doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md` 為準。

### 4.1 Training／Architecture 系列

已占用：

`MR-6A`、`MR-6B`、`MR-7A`、`MR-8A`、`MR-8B`、`MR-8C`、`MR-8D`、`MR-8E`、`MR-8F`、`MR-8G`、`MR-8H`、`MR-8I`、`MR-8J`、`MR-8K`、`MR-8L`、`MR-8M`、`MR-8O`、`MR-8P`、`MR-9A`、`MR-9A-GN`、`MR-9B`、`MR-9C`、`MR-9D`、`MR-9E`、`MR-9F`、`MR-10A`。

| ID | 名稱 | 狀態 |
|---|---|---|
| `MR-8F` | Unique ticker/date group training | ACCEPTED；歷史高 coverage baseline |
| `MR-9A` | InceptionTime | ACCEPTED；目前 model-quality baseline |
| `MR-10A` | Candidate-conditioned Query / `inception_time_market_set_candidate_v1` | REJECTED；legacy read-only；**ID 永久占用** |

`MR-10A` 已在 2026-07-29 使用。**不得再把新的 candidate-state／daily-quality 工作命名為 `A10` 或 `10A`。**

### 4.2 Strategy-aligned Target／Ranking 11 系列

`MR-11A`～`MR-11K` 亦全部已占用，不得拿來命名新的 candidate-day scoring 模型。

| ID | 名稱 | 歷史狀態 |
|---|---|---|
| `MR-11A` | Strategy-aligned Continuous Outcome Target audit | RESULT_AVAILABLE |
| `MR-11B` | Daily Percentile Regression | REJECTED |
| `MR-11C` | Candidate-set Coverage Audit | RESULT_AVAILABLE |
| `MR-11D` | Label-conditional Target Attribution | RESULT_AVAILABLE |
| `MR-11E` | Time-penalty Ablation | PASSED audit gate |
| `MR-11F` | No-time Target Learnability | PASSED Selection learnability gate |
| `MR-11G` | PASS-conditional Magnitude Ranker | REJECTED |
| `MR-11H` | Realization-gap Attribution | RESULT_AVAILABLE |
| `MR-11I` | Nested Selection Coverage Audit | RESULT_AVAILABLE |
| `MR-11J` | Canonical Counterfactual Execution Audit | STOPPED |
| `MR-11K` | Portfolio Selection-pressure Audit | IMPLEMENTED／historical |

### 4.3 All-event Continuous 12 系列

| ID | 名稱 | 狀態 |
|---|---|---|
| `MR-12A` | No-time All-event Continuous Ranker / `strategy_aligned_no_time_all_event_mse` | SOURCE_SUPPORTED；同一`strategy_aligned_opportunity_no_time_r_v1`與`ARCH-inception_time_v1`，唯一模型變數為training scope `pass_only → all_labels`；AUD-c15-source-attribution確認相同runtime下C15相對C14 final relative wealth +4.52%，且非2024仍+1.04% |
| `MR-12B` | No-time All-event Pairwise Ranker / `strategy_aligned_no_time_all_event_pairwise` | RESULT_AVAILABLE／PIT_MODEL_VALIDATION_PASS／MODEL_ECONOMIC_IMPROVEMENT_SUPPORTED；固定MR-12A Target／all-label／InceptionTime，MSE→within-day RankNet pairwise logistic。Selection PIT 2011-01-01～2020-12-31：10 folds、coverage 100%、PASS-only global rho 0.2336、daily rho 0.1859、spread 1.1386R、年度rho>0 10/10、spread>0 9/10，Gate PASS；這是MR-12B單模型絕對Gate，不代表每個PIT fold皆B>A；score-level drift=True為warning而非Gate veto。正式C19-C17：Return +17.62pp；C20-C18：Return +38.98pp、MDD -0.25pp、RoMD +2.86、EV +0.27R、selection R +81.57R，支持pairwise objective |
| `MR-12C` | No-time All-event ListNet Top-one Listwise Ranker / `strategy_aligned_no_time_all_event_listwise` | RESULT_AVAILABLE／REJECTED；固定MR-12B其餘條件、唯一scientific change為within-day pairwise logistic→same-date full-list ListNet top-one cross-entropy。C21-C19 Return -3.08pp、RoMD -0.21、EV -0.02R、selection R -10.95R；C22-C20 Return -63.03pp、MDD +2.00pp、RoMD -5.33、EV -0.39R、selection R -114.79R。較Max-DL C18下惡化顯著放大，故不取代MR-12B；工件保留歷史重現 |

`MR-12A`、`MR-12B`、`MR-12C` 已正式占用。後續不得重用；若模型權重／target／loss／training-data semantics再變更，須重新查 Registry 取得新的 `MR-*`。

### 4.4 Daily Universal 13 系列

| ID | 名稱 | 狀態 |
|---|---|---|
| `MR-13A` | Daily Universal No-time Pairwise Ranker / `daily_universal_no_time_pairwise` | RESULT_AVAILABLE／FORWARD_OOS_MODEL_GATE_PASS／SELECTION_PIT_MODEL_GATE_PASS／STAGE3_PIT_RUNTIME_READY／CROSS_PERIOD_MULTI_SEED_EDGE_NOT_STABLE／NOT_PROMOTED。單模型Gate與單seed策略結果仍保留；正式8-seed evidence分時段反轉：Forward MR-13A−MR-12B ΔDL選擇R=`+50.64R`（6勝2敗）、ΔRoMD=`+1.76`（4勝4敗）；Selection PIT則ΔDL選擇R=`-20.76R`（3勝5敗）、ΔRoMD=`-0.93`（2勝6敗）。Selection PIT的ranking/RoMD方向一致=`7/8`，故該時段不是主要portfolio translation失真，而是daily-universal relative ranking edge沒有跨時段穩定。Forward schema v5又未找到可泛化的13A-specific sizing/binding機制。結論：不得promotion、不得挑best seed或依單一period調參；Cross-period Audit已確認負edge廣泛分布。Daily Score仍為長期共同模型方向，下一個Daily MR前先完成profile-driven多DL trainer／research identity／loss-reduction架構整理，再一次只改一個training semantic。 |
| `MR-13B` | Daily Universal Target-gap-weighted Pairwise Ranker / `daily_universal_no_time_pairwise_gap_weighted` | RESULT_AVAILABLE／REJECTED_AT_FORWARD_MODEL_GATE／NO_PIT／NO_RUNTIME_DL_SOURCE。Seed42 Forward all-stock daily/global rho=`0.0303/-0.0089`、pair=`51.02%`、Top-bottom spread=`+0.0081R`；breakout daily/global rho=`0.0091/-0.0160`、pair=`51.34%`、Top-bottom spread=`-0.0550R`，相較MR-13A明確退步。不得進PIT或對gap weighting追threshold／exponent／floor。 |
| `MR-13C` | Daily Universal Percentile Regression / `daily_universal_no_time_percentile_mse` | RESULT_AVAILABLE／FORWARD_OOS_MODEL_GATE_PASS／SELECTION_PIT_MODEL_GATE_PASS／NO_RUNTIME_DL_SOURCE／NOT_SELECTED_FOR_NEXT_STRATEGY_GATE。Seed42 Selection PIT：all-stock daily/global rho=`0.0841/0.0463`、pair=`53.15%`、Top-K Lift=`+0.1550R`、Boundary gap=`+0.0858R`，年度rho/spread=`7/8`正；breakout daily rho=`0.0773`、Top-K Lift=`+0.1139R`。Forward all-stock daily rho=`0.1877`、Top-K Lift=`+0.3870R`。PIT絕對Gate通過，但shared 13C/13D/13E PIT中主要ranking與top-tail證據弱於13E，因此本輪不建立`DL-CONT13C`或策略arm。 |
| `MR-13D` | Daily Universal Upper-tail Relevance-weighted Pairwise Ranker / `daily_universal_no_time_upper_tail_pairwise` | RESULT_AVAILABLE／FORWARD_OOS_MODEL_GATE_PASS／SELECTION_PIT_MODEL_GATE_PASS／NO_RUNTIME_DL_SOURCE／NOT_SELECTED_FOR_NEXT_STRATEGY_GATE。Seed42 Selection PIT：all-stock daily/global rho=`0.0842/0.0630`、pair=`52.85%`、Top-K Lift=`+0.1776R`、Boundary gap=`-0.0093R`，年度rho=`7/8`、spread=`6/8`為正；breakout daily rho=`0.0233`、Top-K Lift=`+0.0657R`。Forward all-stock Top-K較13C強，但PIT breakout與整體Daily ranking弱於13E，因此本輪不建立`DL-CONT13D`或策略arm。 |
| `MR-13E` | Daily Universal Full-list Delta-NDCG-weighted Pairwise Ranker / `daily_universal_no_time_full_list_ndcg_pairwise` | RESULT_AVAILABLE／FORWARD_OOS_MODEL_GATE_PASS／SELECTION_PIT_MODEL_GATE_PASS／CROSS_PERIOD_SINGLE_SEED_MODEL_STABILITY_SUPPORTED／STRATEGY_GATE_RUNTIME_SOURCES_CONFIGURED／RUNTIME_CONTRACT_VALIDATOR_REPAIRED／AWAITING_SINGLE_SEED_STRATEGY_GATE。Seed42 Selection PIT：all-stock daily/global rho=`0.1627/0.0533`、pair=`55.75%`、Top-bottom spread=`+0.2813R`、Top-K Lift=`+0.4456R`、Boundary gap=`-0.0043R`，年度rho/spread均=`8/8`正；breakout daily/global rho=`0.1019/0.0833`、pair=`54.77%`、Top-K Lift=`+0.1513R`。Forward all-stock daily/global rho=`0.2151/0.1042`、pair=`57.42%`、Top-K Lift=`+0.9951R`；breakout daily rho=`0.1721`、Top-K Lift=`+0.4489R`。13E在Selection與Forward都維持同方向正edge，shared PIT主要指標整體優於13C/13D，足以授權下一階段controlled strategy gate；仍未經strategy或multi-seed驗證，故不promotion。 |

`MR-13A`、`MR-13B`、`MR-13C`、`MR-13D`、`MR-13E` 已正式占用。後續daily-universal target、loss、architecture或training-data semantics若再變更，必須使用新的 `MR-*`，不得覆寫13A/13B/13C/13D/13E。

13系列採profile-driven trainer：新增Daily MR只新增獨立experiment profile＋research spec並選擇其training reduction／target contract，不複製`train_daily_ranker.py`。多個DL「研究版本並存」已支援，但同一strategy arm同時融合多個DL尚未啟用，因fusion本身屬新的SR scientific variable。

---

## 5. Runtime DL source Registry

| Canonical ID | 相容 alias | Backing identity | Runtime 語意 | 狀態 |
|---|---|---|---|---|
| `DL-A9` | `A9` | `MR-9A / ARCH-inception_time_v1 / PROFILE-unique_group_sampling` | Binary breakout-quality score，threshold 0.5 | ACTIVE research source |
| `DL-TP1` | `TP1` | `LABEL-a2_realized_trade_path_v1` training line | Realized trade-path binary score，threshold 0.5 | Hard-filter use rejected；歷史保留 |
| `DL-CONT11G` | `CONT11G` | `MR-11G / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_pass_magnitude_mse` | Frozen OOS continuous breakout-event score；只供capital-utilization-first controlled strategy research。歷史training scope為PASS-only，SR-C14在全部orderable breakout events上的使用屬受控deployment hypothesis | HISTORICAL research-only score source；SR-C14未採用 |
| `DL-CONT12A` | `CONT12A` | `MR-12A / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_mse` | Frozen OOS all-event continuous breakout-event score；training scope=`all_labels`，Target仍為`strategy_aligned_opportunity_no_time_r_v1`；作為後續controlled selector研究的固定score source | SOURCE_SUPPORTED |
| `DL-CONT12B` | `CONT12B` | `MR-12B / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_pairwise` | Frozen OOS all-event continuous model checkpoint；training為同日pairwise ranking；runtime使用feature-history-only inference universe | RESULT_AVAILABLE／CURRENT_FORWARD_STRATEGY_SOURCE；future-independent Forward score已重建並完成C20 controlled replay，2021-01-01～2026-03-02 Return=190.26%、RoMD=10.15、EV=1.17R、selection R=+35.93R。舊target-complete-only score只保留PRE_FIX歷史追溯。 |
| `DL-CONT12B-PIT` | `CONT12B_PIT` | `MR-12B / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_pairwise` | Selection point-in-time continuous score；每個歷史日期只使用當時合法fold模型輸出 | RESULT_AVAILABLE／PIT_GATE_PASS／CURRENT_FIVE_FIELD_MIN_ROOS_REPLAY_COMPLETE／NOT_PROMOTED；C24/C25相對C23 Return `-25.00pp/-6.43pp`，selection R `+12.19R/+26.19R`；selection R改善未充分轉成portfolio economics |
| `DL-CONT13A-PIT` | `CONT13A_PIT` | `MR-13A / ARCH-inception_time_v1 / PROFILE-daily_universal_no_time_pairwise` | Daily Selection PIT continuous score；score eligibility只依截至information date已有足夠feature history，不要求future 40-bar target；策略每個盤前決策使用最新已完成交易日score | RESULT_AVAILABLE／PIT_MODEL_GATE_PASS／STRATEGY_RUNTIME_READY／SELECTION_STRATEGY_GATE_PASS；C28-C25 source-only Return `+50.33pp`、RoMD `+2.89`、EV `+0.19R`、selection R `+83.72R` |
| `DL-CONT13A` | `CONT13A` | `MR-13A / ARCH-inception_time_v1 / PROFILE-daily_universal_no_time_pairwise` | Frozen Forward-OOS daily continuous model checkpoint；策略盤前使用最新已完成交易日score；runtime score eligibility只依feature history、不要求future target | RESULT_AVAILABLE／SINGLE_SEED_FORWARD_FAIL／MULTI_SEED_RANKING_EDGE／PORTFOLIO_TRANSLATION_HETEROGENEOUS／NOT_PROMOTED；seed 42 C29仍為83.81%/RoMD5.34/EV0.52R/selection R -183.42R，但8-seed MR-13A相對MR-12B同seedΔDL選擇R 6勝2敗，且全8-seed direct-pair Exclusive ΔR Mean=`+50.64R`、Exclusive ΔPnL Mean=`+173,208.15`。因ΔRoMD仍4勝4敗且S1/S8存在正R負PnL的risk-dollar反轉，尚不取代DL-CONT12B；舊target-complete-only score只保留PRE_FIX歷史追溯。 |
| `DL-CONT13E-PIT` | `CONT13E_PIT` | `MR-13E / ARCH-inception_time_v1 / PROFILE-daily_universal_no_time_full_list_ndcg_pairwise` | Daily Selection PIT continuous score；每個盤前決策只使用最新已完成交易日score，score eligibility沿用daily feature-history-only contract | STRATEGY_GATE_SOURCE_AUTHORIZED／PIT_MODEL_GATE_PASS／AWAITING_SR-C35。僅供與`DL-CONT12B-PIT`、`DL-CONT13A-PIT`在相同Min params／feasible-ascent selector下做source-only Selection策略Gate；策略結果前不得promotion。 |
| `DL-CONT13E` | `CONT13E` | `MR-13E / ARCH-inception_time_v1 / PROFILE-daily_universal_no_time_full_list_ndcg_pairwise` | Frozen Forward-OOS daily continuous score；策略盤前使用最新已完成交易日score，runtime score eligibility沿用feature-history-only contract | STRATEGY_GATE_SOURCE_AUTHORIZED／FORWARD_MODEL_GATE_PASS／AWAITING_SR-C36。僅供與`DL-CONT12B`、`DL-CONT13A`在相同current Min params／feasible-ascent selector下做source-only Forward策略Gate；尚未取代`DL-CONT12B`。 |
| `DL-CONT12C` | `CONT12C` | `MR-12C / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_listwise` | Frozen OOS all-event continuous score；runtime仍使用softmax PASS probability，training為same-date full-list ListNet top-one cross-entropy；只供固定C17/C18 selector的controlled Pairwise-vs-Listwise歷史重現 | RESULT_AVAILABLE／REJECTED_FOR_MODEL_RESEARCH |

**改變 DL-A9 的策略使用方式不會自動產生 `DL-A10`，也不會自動成為新 `MR-*`。**只有模型權重、training target、architecture 或 training-data semantics 真正改變，才需要另立 model research identity。

---

## 6. Strategy Runtime Arm Registry

| Arm | Canonical name | 唯一主要差異 | 結果／狀態 |
|---|---|---|---|
| `SR-C1` | Full ROOS | Full rules，DL off | Historical comparator |
| `SR-C2` | Full ROOS: TP1-on | TP1 hard filter | REJECTED |
| `SR-C3` | Min ROOS | All-off Min params，DL off | Core strategy baseline |
| `SR-C4` | Min ROOS: TP1-on | TP1 hard filter | REJECTED |
| `SR-C5` | Min-TP1 ROOS | TP1-trained params，DL off | NOT PROMOTED |
| `SR-C6` | Min-TP1 ROOS: DL-on | TP1-trained params + TP1 runtime | REJECTED |
| `SR-C7` | Full ROOS: A9-on | A9 hard filter | REJECTED |
| `SR-C8` | Min ROOS: A9-on | A9 hard filter | REJECTED |
| `SR-C9` | Min-A9 ROOS | A9-trained params，DL off | NOT PROMOTED |
| `SR-C10` | Min-A9 ROOS: DL-on | A9-trained params + A9 hard filter | REJECTED |
| `SR-C11` | Min ROOS: A9 resource-aware | Cash-bottleneck gate + first-improvement PASS promotion | POSITIVE；目前 resource-aware 已知經濟結果最佳 |
| `SR-C12` | Min ROOS: A9 resource-aware basket | 相同 resource gate + best-improvement／最大化 PASS basket 方向 | ACTIVE max-PASS research base；PASS 使用增加，但經濟結果低於 C11 |
| `SR-C13` | Min ROOS: A9 resource-aware basket + candidate-day re-score | 同一DL-A9權重／threshold與C12 allocation，把A9 quality改為每個策略VALID candidate day重新計算 | **CANCELLED_BEFORE_IMPLEMENTATION**；A9是breakout-event classifier，extended candidate-day通常不是breakout形態，直接re-score語意／distribution不成立；ID永久保留不得重用 |
| `SR-C14` | Min ROOS: Continuous resource-aware | Min ROOS exact cash-cap先判resource mode；capital-utilization mode完全維持Min ROOS，只有cash-binding的DL Selection Mode才使用`DL-CONT11G` frozen OOS continuous score，且排序不得破壞cash-binding資源契約 | RESULT_AVAILABLE／NOT_ADOPTED；曝險接近C3，但RoMD／EV／同參數DL選擇R明顯低於C12；MR-11G PASS-only score的all-event deployment不成立 |
| `SR-C15` | Min ROOS: All-event Continuous resource-aware | 完全沿用SR-C14 capital-utilization-first與cash-binding契約；唯一差異為score source改成`DL-CONT12A / MR-12A`真正all-event訓練的continuous ranker | RESULT_AVAILABLE／PROMISING_NOT_PROMOTED；同期間相對C3 +12.35pp Return、-0.60pp MDD、+1.24 RoMD；相對C12 +6.10pp Return、-1.66pp MDD、+1.51 RoMD，但EV與同參數DL選擇R較弱且年度結果集中 |
| `SR-C16` | Min ROOS: All-event Continuous capital-preserving | 固定`PARAM-P2 / DL-CONT12A / MR-12A`；Min ROOS先建立exact reservation baseline，continuous score可跨cash／slot-binding重排，但接受basket必須同時滿足`selected_count >= baseline`與`reserved_cost >= baseline`；不新增threshold／blend／future資料 | RESULT_AVAILABLE／RESOURCE_CONTRACT_PASSED／NOT_PROMOTED；Return=175.15%、MDD=15.86%、RoMD=11.05、EV=0.48R；相對C15 Return +6.46pp但RoMD與EV退步；DL-selection 270日、selected-count delta +70、reserved-capital delta +1,361,933、resource violation 0 |
| `SR-C17` | Min ROOS: All-event Continuous max-DL constrained basket | 固定`PARAM-P2 / DL-CONT12A / MR-12A`；Min ROOS只建立同日K與reserved-capital floor，DL score決定K-stock basket membership；純DL Top-K若不合法，以最多K次deterministic minimum-repair替換Top-K成員；basket內正式執行順序沿用Min ROOS rank，且實際盤前只允許K筆預留單 | RESULT_AVAILABLE／FROZEN_AS_MODEL_VALIDATION_HARNESS／NOT_PROMOTED_AS_STRATEGY；Return=164.13%、MDD=14.38%、RoMD=11.42、EV=0.72R；Max-DL eligible 223日中repair 209日、fallback 26日，K/resource violation=0。雖搜尋不如C18完整，但依使用者要求保留既有語意作固定較低Max-DL harness，用於判斷新DL改善是否只在C17或也能轉化到C18；不得依新模型OOS修改 |
| `SR-C18` | Min ROOS: All-event Continuous max-DL feasible-ascent | 固定`PARAM-P2 / DL-CONT12A / MR-12A`與C17完全相同K／R0 hard constraints；先取得C17合法seed（含原fallback Min ROOS seed），再對所有selected↔unselected single swaps逐一用canonical exact reservation驗證，只接受可行且DL quality更高者，每輪取最佳改善直到無任何1-swap改善；capital只作feasibility，不參與objective | RESULT_AVAILABLE／SELECTOR_FROZEN_FOR_MODEL_RESEARCH／NOT_PROMOTED_AS_STRATEGY；Return=140.56%、MDD=14.73%、RoMD=9.54、EV=0.57R、same-param DL selection R=-26.25R；242/242 eligible days達1-swap local optimum、final fallback=0、K/resource violation=0。Selector total=1668.23ms、median=0.829ms、p95=4.065ms，相對C17 total slowdown=1.277×；成本可接受，後續固定此selector檢驗新DL source，不再新增capital/DL比例或搜尋規則 |
| `SR-C19` | Min ROOS: MR-12B max-DL constrained basket | 完全沿用SR-C17 K/R0、minimum-repair、action-prefix與execution-order語意；唯一差異為score source `DL-CONT12A → DL-CONT12B` | RESULT_AVAILABLE／MODEL_VALIDATION_ARM；Return=181.75%、MDD=16.19%、RoMD=11.23、EV=0.70R、same-param DL selection R=+25.05R；相對C17 Return +17.62pp但MDD/RoMD/EV略退步 |
| `SR-C20` | Min ROOS: MR-12B max-DL feasible-ascent | current五欄Min ROOS、all-off rules、frozen feasible-ascent；K/R0與reserved-capital floor由同一Min DL-off baseline逐日建立；score source=`DL-CONT12B` | RESULT_AVAILABLE／CURRENT_FORWARD_STRATEGY_ANCHOR；future-independent score current replay 2021-01-01～2026-03-02：Return=190.26%、MDD=18.75%、RoMD=10.15、Annual=22.95%、EV=1.17R、selection R=+35.93R。相對C3 Return +88.66pp、MDD +5.62pp、RoMD +2.41、Annual +8.39pp、EV +0.15R。舊2021～2025 target-complete-only結果保留PRE_FIX歷史。 |
| `SR-C21` | Min ROOS: MR-12C max-DL constrained basket | 完全沿用SR-C19／SR-C17 K/R0、minimum-repair、action-prefix與execution-order語意；唯一差異為score source `DL-CONT12B → DL-CONT12C` | RESULT_AVAILABLE／MODEL_VALIDATION_ARM／NOT_ADOPTED；Return=178.68%、MDD=16.22%、RoMD=11.02、EV=0.67R、selection R=+14.09R；相對C19全面小幅退步 |
| `SR-C22` | Min ROOS: MR-12C max-DL feasible-ascent | 完全沿用SR-C20／SR-C18 K/R0、feasible-ascent與execution-order語意；唯一差異為score source `DL-CONT12B → DL-CONT12C` | RESULT_AVAILABLE／MODEL_VALIDATION_ARM／NOT_ADOPTED；Return=116.51%、MDD=16.49%、RoMD=7.07、EV=0.45R、selection R=-59.48R；相對C20 Return -63.03pp、RoMD -5.33、EV -0.39R、selection R -114.79R，顯示較Max selector強烈放大MR-12C ranking弱點 |
| `SR-C23` | Selection Min ROOS PIT baseline | 固定2014～2020 current五欄Min ROOS active params、rules全關、DL off；作Selection PIT策略經濟驗證共同baseline | RESULT_AVAILABLE／CURRENT_FIVE_FIELD_BASELINE；Return=107.98%、MDD=20.57%、RoMD=5.25、Annual=11.03%、EV=0.42R、Exposure=87.19%、475 trades |
| `SR-C24` | Selection PIT: MR-12B minimum-repair | 與SR-C23完全相同current五欄Min ROOS params；唯一新增為`DL-CONT12B-PIT`，selector完全沿用SR-C17 minimum-repair K/R0 contract | RESULT_AVAILABLE／NOT_ADOPTED；Return=82.98%、MDD=18.99%、RoMD=4.37、EV=0.50R、selection R=+12.19R；相對C23 Return -25.00pp、MDD -1.58pp、RoMD -0.88、EV +0.08R，selection R未轉成portfolio改善 |
| `SR-C25` | Selection PIT: MR-12B feasible-ascent | 與SR-C23完全相同current五欄Min ROOS params；使用`DL-CONT12B-PIT`並完全沿用SR-C18 feasible-ascent K/R0 contract | RESULT_AVAILABLE／NOT_ADOPTED；Return=101.55%、MDD=19.66%、RoMD=5.17、EV=0.52R、selection R=+26.19R；相對C23 Return -6.43pp、MDD -0.91pp、RoMD -0.08、EV +0.10R。相對C24 Return +18.57pp、RoMD +0.80、selection R +14.00R，feasible-ascent改善轉化但仍未超越baseline |
| `SR-C26` | Selection PIT: MR-12B feasible-ascent stale-score membership guard | 完全沿用SR-C25的historical P2 params、`DL-CONT12B-PIT`、K/R0與feasible-ascent；唯一runtime變更為Selection預先凍結`score age > 22 calendar days`的有效舊PIT score不得參與DL造成的basket membership change。候選不刪除、不過期、不hard reject，仍保留Min ROOS membership／orderability；fresh↔fresh feasible-ascent維持原C25語意 | RESULT_AVAILABLE／STALE_GUARD_SUPPORTED／NOT_PROMOTED／CUTOFF_FROZEN_22D／RUNTIME_MICROTUNING_STOPPED；C26-C25 Return `+4.32pp`、EV `+0.08R`、selection R `+27.51R`且exclusive ΔPnL=`+57,383.45`，但C26-C23 Return `-7.71pp`、MDD `+1.66pp`、RoMD `-0.59`；risk-dollar attribution顯示C26-only risk-weighted R=`0.21R`低於C23-only=`0.27R`，common PnL差則幾乎全為既有sizing/path effect，無新的事前泛化runtime修正依據，故不進Forward-OOS且22日門檻維持凍結 |
| `SR-C27` | Selection PIT: MR-13A daily minimum-repair | 與SR-C24完全相同current五欄Min ROOS params、K/R0、minimum-repair selector與execution order；唯一DL source差異為`DL-CONT12B-PIT → DL-CONT13A-PIT`，盤前每日使用最新已完成交易日score | RESULT_AVAILABLE／NOT_PROMOTED；Return=102.26%、MDD=21.57%、RoMD=4.74、EV=0.50R。相對C24 Return +19.27pp／RoMD +0.37／selection R +6.22R，但相對C23 baseline仍Return -5.73pp、MDD +1.00pp、RoMD -0.51，故minimum-repair不進Forward-OOS |
| `SR-C28` | Selection PIT: MR-13A daily feasible-ascent | 與SR-C25完全相同current五欄Min ROOS params、K/R0、feasible-ascent selector與execution order；唯一DL source差異為`DL-CONT12B-PIT → DL-CONT13A-PIT`，盤前每日使用最新已完成交易日score | RESULT_AVAILABLE／SELECTION_GATE_PASS／FORWARD_OOS_AUTHORIZED／FROZEN_SELECTOR；Return=151.88%、MDD=18.86%、RoMD=8.05、EV=0.71R。C28-C25 source-only Return +50.33pp、MDD -0.80pp、RoMD +2.89、EV +0.19R、selection R +83.72R；C28-C23 Return +43.90pp、MDD -1.71pp、RoMD +2.80、EV +0.29R、selection R +109.91R；C28-C27 Return +49.62pp、RoMD +3.31、EV +0.21R。Selection後不得再調feasible-ascent／stale guard／capital objective |
| `SR-C29` | Forward-OOS: Min ROOS + MR-13A daily feasible-ascent | current五欄Min ROOS、all-off rules、frozen feasible-ascent與daily information-date lookup；K/R0與reserved-capital floor由同一C3 Min DL-off baseline逐日建立；score source=`DL-CONT13A` | RESULT_AVAILABLE／FORWARD_STRATEGY_GATE_FAIL／NOT_PROMOTED。future-independent score current replay 2021-01-01～2026-03-02：Return=83.81%、MDD=15.71%、RoMD=5.34、Annual=12.53%、EV=0.52R、selection R=-183.42R。相對C3 Return -17.79pp、MDD +2.58pp、RoMD -2.41、EV -0.51R；相對C20純source Return -106.45pp、MDD -3.04pp、RoMD -4.81、EV -0.65R、selection R -219.35R。Selector保持frozen；後續只做seed robustness與錯誤歸因，不用Forward結果微調selector／loss weight／hyperparameter。 |
| `SR-C30` | Forward-OOS: Full ROOS + MR-12B feasible-ascent | Full ROOS active params/formal rules + frozen feasible-ascent + `DL-CONT12B`；same-param K/R0 hard feasibility | RESULT_AVAILABLE／HISTORICAL_FULL_DL_COMPARATOR／INACTIVE_CURRENT_MATRIX；Return=133.21%、MDD=17.92%、RoMD=7.43、Annual=18.59%、EV=0.61R、selection R=+25.22R。永久保留重現，不再列入current四arm改善比較。 |
| `SR-C31` | Forward-OOS: Full ROOS + MR-13A daily feasible-ascent | 與C30同Full ROOS/formal/same-param feasible-ascent，唯一DL source=`DL-CONT13A` | RESULT_AVAILABLE／HISTORICAL_FULL_DL_COMPARATOR／INACTIVE_CURRENT_MATRIX；Return=122.00%、MDD=16.21%、RoMD=7.53、Annual=17.42%、EV=0.59R、selection R=+13.52R；C31-C30 Return -11.21pp、RoMD +0.09、EV -0.02R。永久保留重現，不作current MR-13A主要Gate。 |
| `SR-C32` | Selection PIT: Full ROOS baseline | `PARAM-P4` historical Full ROOS rolling active params；formal rules；DL-off；2014～2020無前視active-param replay | RESULT_AVAILABLE／ACTIVE_FULL_SELECTION_BASELINE；Return=132.25%、MDD=15.85%、RoMD=8.34、Annual=12.80%、EV=0.71R。Current Selection四arm保留C32作完整策略體系baseline；engine允許standalone DL-off comparator，不要求隱藏Full+DL arm。 |
| `SR-C33` | Selection PIT: Full ROOS + MR-12B feasible-ascent | `PARAM-P4`/formal + `DL-CONT12B-PIT` + frozen feasible-ascent | RESULT_AVAILABLE／HISTORICAL_FULL_DL_SELECTION／INACTIVE_CURRENT_MATRIX；Return=146.25%、MDD=18.91%、RoMD=7.73、Annual=13.74%、EV=0.56R、selection R=-63.74R。永久保留結果與ID，但不再列入current四arm。 |
| `SR-C34` | Selection PIT: Full ROOS + MR-13A daily feasible-ascent | `PARAM-P4`/formal + `DL-CONT13A-PIT` + frozen feasible-ascent | RESULT_AVAILABLE／HISTORICAL_FULL_DL_SELECTION／INACTIVE_CURRENT_MATRIX；Return=167.12%、MDD=18.44%、RoMD=9.06、Annual=15.07%、EV=0.75R、selection R=+1.99R；C34-C33 source-only Return +20.87pp、RoMD +1.33、EV +0.19R。永久保留結果與ID，但current MR-13A改善聚焦Min row。 |
| `SR-C35` | Selection PIT: Min ROOS + MR-13E daily feasible-ascent | 與`SR-C25/SR-C28`完全相同historical Min ROOS params、all-off rules、K/R0、feasible-ascent與execution；唯一DL source=`DL-CONT13E-PIT` | IMPLEMENTED／AWAITING_SINGLE_SEED_SELECTION_STRATEGY_GATE／ROBUSTNESS_OFF。與C25/C28做12B/13A/13E純source比較；single-seed策略結果前不得加入multi-seed robustness。 |
| `SR-C36` | Forward-OOS: Min ROOS + MR-13E daily feasible-ascent | 與`SR-C20/SR-C29`完全相同current Min ROOS、all-off rules、K/R0、feasible-ascent與execution；唯一DL source=`DL-CONT13E` | IMPLEMENTED／AWAITING_SINGLE_SEED_FORWARD_STRATEGY_GATE／ROBUSTNESS_OFF。與C20/C29做12B/13A/13E純source比較；single-seed策略結果前不得加入multi-seed robustness。 |

### `SR-C13` identity boundary

`SR-C13`曾正式規劃為candidate-day re-score，依Registry「已使用ID永久保留」規則不得改名重用。2026-08-07在實作前取消：`DL-A9`是以breakout-event snapshot訓練的classifier，而extended candidate-day通常不是breakout線型；把非breakout state直接餵回同一模型會改變輸入分布與score語意，因此不把這條路徑當成既有DL的乾淨策略使用方式。Candidate validity仍由原策略唯一負責。

若未來真的要學extended／candidate-state品質，必須另立新的`MR-*`模型研究；若只改既有`DL-A9`在portfolio中的排序／allocation，則使用下一個新的`SR-C*`，不得重用`SR-C13`。


---

### Multiple-seed robustness workflow contract（2026-08-11）

- `Forward-OOS Multi-seed robustness`與`Selection PIT Multi-seed robustness`都是`config/strategy_compare.py`驅動的正式Strategy Compare子工作類型；App不得硬編MR/C/模型名稱。
- Selection robustness固定沿用`selection_pit`策略比較期間；目前為`2014-01-01～2020-12-31`，PIT builder只建立此策略期間所需的12-month score folds，因此目前是7 folds/model/seed，而不是重跑各模型canonical PIT歷史的10/8 folds。
- 每個seed的PIT模型／Scores必須寫入robustness isolated work root；不得覆寫canonical Selection PIT工件或workflow seed。Strategy replay可透過明確override讀取isolated PIT score+manifest，但仍沿用相同Target、architecture、training sample scope、selector與策略accounting。
- Strategy runtime共用值`dataset / param_policy / max_positions / rotation`由`get_breakout_quality_workflow_settings()`單一來源供給；`config/strategy_compare.py`不得複製第二份magic value。
- Robustness永久raw aggregate固定包含`seed_results.csv`與`seed_yearly_returns.csv`；`yearly_report`只控制年度表顯示，不控制raw年度資料保存，因此切換report/console設定不需因缺年度raw而重訓。成功完成預設清除per-seed checkpoints／scores／replay work，FAILED／INTERRUPTED保留resumable work。
- 年度aggregate的side也是正式契約：fixed baseline只讀`no_filter_return_pct`，stochastic DL ranking只讀`score_ranking_return_pct`；controlled-pair cache同時存在兩欄時不得互換。
- Scientific fingerprint只包含scientific identity；console mode、progress interval、worker數、report schema、yearly renderer、retention policy、選單/arm顯示名稱與description不得造成模型scientific fingerprint改變。
- Console進度與Strategy Compare報表共用同一色彩語意：active info=cyan，DONE/REUSE=green，warning/resume=yellow，FAILED/BLOCKED=red；績效delta仍由`strategy_report_style.signal_for_delta()`依指標方向判讀，不能因工作完成就把負向績效染綠。
- 狀態：`IMPLEMENTED / RESULT_PENDING_FOR_SELECTION_ROBUSTNESS`；不新增`MR-*`／`DL-*`／`SR-C*` identity，不改既有Forward multi-seed結果。

---

## 7. Audit Registry

| Canonical ID | Config ID | Source | 狀態 | 主要結論 |
|---|---|---|---|---|
| `AUD-a9-pass-quality` | `a9-pass-quality` | `SR-C12 / DL-A9` | RESULT_AVAILABLE | Raw A9 PASS score 整體單調性弱；不支持直接 score sorting 或 age cutoff |
| `AUD-a9-pass-persistence` | `a9-pass-persistence` | `SR-C12 / DL-A9` | RESULT_AVAILABLE | False PASS persistence=1.60×；candidate-day FP amplification=1.33×；selector 不是主要放大來源 |
| `AUD-a9-selection-confidence` | `a9-selection-confidence` | `SR-C12 / DL-A9` | RESULT_AVAILABLE／NOT_USED_FOR_PRIMARY_RANK | Candidate-day rho=0.071、unique-event rho=0.102、selected R rho=0.082；每日平均Label concordance=49.18%，不支持A9 confidence作主排序 |
| `AUD-c15-strategy-attribution` | `c15-strategy-attribution` | `SR-C15` vs `SR-C3 / SR-C12` | RESULT_AVAILABLE | 2024單獨relative wealth effect約+18.10%/+11.46%，非2024約-11.25%/-8.19%；C15全期優勢主要由portfolio geometry／slot occupancy／compounding解釋，非平均R提升 |
| `AUD-c15-source-attribution` | `c15-source-attribution` | `SR-C15 / MR-12A` vs `SR-C14 / MR-11G` | RESULT_AVAILABLE | C15相對C14 Return +11.61pp、MDD -2.36pp、RoMD +2.24、EV +0.06R、relative wealth +4.52%；2024 +3.44%、非2024 +1.04%，exclusive selection +12.16R／+196,527.93 PnL；MR-12A all-label source支持保留 |
| `AUD-c23-c25-pit-realization` | `c23-c25-pit-realization` | `SR-C23` vs `SR-C24 / SR-C25` | RESULT_AVAILABLE／HISTORICAL_4_ATR_UNIVERSE／REALIZATION_GAP_CONFIRMED／PARAM_ADAPTATION_NOT_SUPPORTED | C24/C25平均Target R分別`+0.04R/+0.06R`，但平均Realized R`-0.09R/-0.03R`、aggregate capture`-0.41/-0.26`、exclusive selection R=`-35.78R/-19.52R`。Exclusive R分解顯示兩者主要損失皆來自winner capture不足：C24 winner contribution約`-29.52R`、loser contribution約`-6.26R`；C25 winner contribution約`-22.03R`、loser contribution反而`+2.51R`。fill/sizing/holding/exposure皆未達既定mechanical-gap門檻，故capture惡化只證明realization gap，不足以直接支持Selection參數適應 |
| `AUD-c23-c25-pit-fold-runtime` | `c23-c25-pit-fold-runtime` | `SR-C23` vs `SR-C24 / SR-C25` + `DL-CONT12B-PIT` fold identity | RESULT_AVAILABLE／HISTORICAL_4_ATR_UNIVERSE／FOLD_DRIFT_NOT_PRIMARY_CAUSE | C24 mixed-fold `+13.91R`、single-fold `-46.05R`、boundary outside `-37.95R`；C25 mixed-fold `-4.25R`、single-fold `-10.65R`、boundary outside `-24.45R`。fold drift存在但負Selection R並未集中於mixed-fold／fold boundary，故不支持cross-fold normalization作下一步。 |
| `AUD-c23-c25-pit-target-realization` | `c23-c25-pit-target-realization` | `SR-C23` vs `SR-C24 / SR-C25` + `DL-CONT12B-PIT` + original event Target | RESULT_AVAILABLE／HISTORICAL_4_ATR_UNIVERSE／C25_OLD_SIGNAL_TAIL_SUPPORTED／C24_NON_MONOTONIC | Actual exclusive trades coverage C24/C25=`93.85%/93.66%`。C25 Q1～Q3 Selection ΔR=`+5.67/+13.31/+2.76R`，合計`+21.74R`；Q4(age `23～288`日、median `38.5`)=`-40.18R`，且C25-only Q4 realized mean=`0.42R` vs C23-only=`1.50R`。C24負R則集中Q2與Q4而非單調隨age惡化。Target↔Realized rho仍為正（C24-only `0.23`、C25-only `0.35`），Age↔Realized接近0，故只支持C25尾端stale-event runtime假說，不支持全域age cutoff或直接改Target公式；未成交／未選候選仍不建立counterfactual R。 |
| `AUD-c23-c26-pit-portfolio-translation` | `c23-c26-pit-portfolio-translation` | `SR-C26` vs `SR-C23 / SR-C25` | RESULT_AVAILABLE／HISTORICAL_4_ATR_UNIVERSE／RISK_DOLLAR_PATH_DECOMPOSITION_CONFIRMED／C26_NOT_PROMOTED／RUNTIME_MICROTUNING_STOPPED | C26-C23：exclusive ΔR=`+7.99R`但ΔPnL=`-35,524.03`；C23-only/C26-only risk-weighted R=`0.27R/0.21R`，故equal-weighted R優勢沒有轉成實際risk-dollar品質。206筆common trades的ΔR約0、ΔPnL=`-41,604.23`，risk-size/path effect=`-41,612.25`、R-difference effect=`+8.02`、residual約0；且C26平均implied risk略高而非整體de-risk，故common loss是trade-specific portfolio sizing/path分布，不是共同交易R變差。C26-C25仍證明22日guard在R與dollar PnL均有效，但不足以超越C23；不據此調fixed risk、selector、22日cutoff或進Forward-OOS。 |
| `AUD-forward-robustness-portfolio-translation` | `forward-robustness-portfolio-translation` | `MR-13A` vs `MR-12B` Forward 8-seed robustness | RESULT_AVAILABLE／READ_ONLY／ALL_SEEDS_COMPLETE／WITHIN_SET_WEIGHTING_DILUTION_CONFIRMED／SCHEMA5_IMPLEMENTED／COMPACT_SCHEMA2_REBUILD_REQUIRED | schema v4 exact bridge：Direct-pair Exclusive ΔR Mean=`+50.64R`、Exclusive ΔPnL Mean=`+173,208.15`；equal-risk selection=`+355,214.76`、average-risk scale=`+14,995.80`、within-set weighting=`-197,002.42`、uncovered/residual=`0`。S1 selection `+78,069`但within-set=`-166,717`→PnL `-103,422`；S8 `+523,425/-352,399/-605,665`→`-434,639`；S6 selection `-289,839`但scale/within=`+432,811/+377,298`→`+520,269`。schema v5沿用同Audit ID，compact attribution schema v2新增entry-execution sidecar，保存預定risk budget、actual initial risk、candidate/chosen/filled qty與canonical counterfactual binding；重建仍須同scientific fingerprint／same 8 seeds並驗證原observation後才VERIFIED，Audit自身不train/replay。 |
| `AUD-cross-period-year-regime-attribution` | `cross-period-year-regime-attribution` | `MR-13A` vs `MR-12B` Selection PIT + Forward-OOS 8-seed robustness | RESULT_AVAILABLE／READ_ONLY／ALL_SAME_SEEDS_COMPLETE／RANKING_EDGE_DIRECTION_REVERSAL／BROAD_SELECTION_PERIOD_WEAKNESS | Selection baseline-relative ΔDL選擇R Mean=`-20.76R`、Forward=`+50.64R`；逐年度direct-pair Exclusive ΔR Selection為5負2正、Forward為1負5正。Selection worst year=2014 `-13.96R`，只占Selection負年度R `42.3%`，故cross-period reversal不是單一年度集中。Period aggregate與year bucket維持不同basis，不建立year/regime gate、不回流training。下一步優先用既有MR-12B權重做daily-universal inference bridge，隔離training universe與inference universe。 |

> 2026-08-12 attribution infra correction：`c15_strategy_attribution`過去曾以signal-date-aware key做cross-arm trade matching，現已改回與Strategy Compare同源的actual-trade canonical key。修正前既有Audit數值保留為`PRE_FIX_HISTORICAL`證據；若未來要用舊Audit重新做promotion／淘汰決策，必須以修正後primitive重跑，不得把舊數值視為current canonical attribution。

Audit 固定 read-only。Audit 結果可形成 `SR-*` 或 `MR-*` 假設，但 Audit 自己不占用這兩種 ID。

---

## 8. Parameter／Artifact Stage Registry

| ID | 定義 | 狀態／用途 |
|---|---|---|
| `PARAM-P2` | DL-off Min ROOS rolling active params；每fold只搜尋`high_len`＋4個ATR欄位，其餘由canonical config/schema固定，trials讀`config/training_policy.py` | Current Min ROOS parameter baseline |
| `PARAM-P3-TP1` | TP1-on-trained parameter stage | Historical；not promoted |
| `PARAM-P3-A9` | A9-on-trained parameter stage | Historical；not promoted |
| `PARAM-P4` | Selection historical Full ROOS rolling active params；2014～2020、120m train／12m OOS、trials直接讀`config/training_policy.py`；使用canonical Full optimizer search space，TP/DL/History threshold依current optimizer policy固定OFF | RESULT_AVAILABLE／ACTIVE_FULL_SELECTION_BASELINE_SOURCE；7 folds已完成，供C32及歷史C33/C34重現；current四arm只需要C32，但P4 lifecycle／cache identity永久保留。 |

`P1/P2/P3` 是策略參數訓練 stage／artifact identity，不是 model version，也不得拿來當 scientific experiment ID。

### 8.1 Multiple-seed strategy robustness infrastructure

`Multiple-seed robustness`是Strategy Compare的final-strategy穩健性診斷，不新增`MR-*`、`DL-*`或`SR-C*` identity，也不做seed ensemble／best-seed selection。比較對象由`config/strategy_compare.py`中目前profile的arm `robustness_role`解析：`fixed_baseline`只回放一次，`stochastic`依deterministic generated seeds逐一使用該階段canonical trainer/PIT builder產生隔離score，再套用相同strategy replay；目前正式子工作類型為Forward-OOS與Selection PIT兩個config-driven profiles。Per-seed isolated Forward score replay必須把calendar execution start與第一個實際score交易日分開：execution start取自isolated model manifest的`outer_oos_policy.oos_start_date`，score table `available_from`可因假日／非交易日晚於該日，兩者不得互相取代。若CPU replay在與下一個GPU training重疊時失敗，run manifest必須標記`FAILED`且`resumable=true`，並停止仍在執行的trainer process。永久只保留fingerprint-scoped `manifest.json`、`seed_results.csv`、`seed_yearly_returns.csv`、`robustness_summary.json`與`robustness_report.md`；年度raw永遠保存，年度表顯示可關閉；seed checkpoint／score／replay detail預設為暫存並於數值落盤後清除。報表第一表以各seed最終策略指標Mean比較並同列Full／Min fixed baseline；第二表專門列RoMD Mean／Median／Std／CV／Min／P25／P75／Max與勝baseline比例，並額外保存同seed matched-pair RoMD差異及任意seed distribution comparison。正式執行前只針對本次required parameter sources與model upstream顯示依賴計畫並確認一次；可建立的策略參數透過canonical parameter builder自動建立／接續，isolated seed workflow不得因normal Strategy Compare canonical DL source缺件而被阻擋。Forward共同期間由canonical Dataset與walk-forward policy推導，不綁定目前canonical score artifact尾端；resume seed results必須驗證arm/seed鍵集合，所有run-stage failure均留下`FAILED`／`resumable` manifest。2026-08-11首批8-seed結果：Full RoMD=7.42、Min=7.74；MR-12B RoMD Mean/Median/Std=`8.21/7.70/2.63`、勝Min/Full=`4/8`；MR-13A=`9.97/10.87/3.68`、勝Min/Full=`5/8`。MR-13A平均與median均高於MR-12B，但CV亦較高（0.37 vs 0.32），證明seed variance足以推翻「單一seed 42可代表模型語意優劣」的推論；在同seed paired delta完成前不升格MR-13A，也不維持「MR-13A普遍Forward失敗」的強結論。首批report另暴露`DL選擇R Mean`在score-ranking路徑未接canonical trade reconstruction而顯示`-`，schema v3改用Strategy Compare既有同參數直接選擇R SSOT並加入matched-seed RoMD比較。狀態：**RESULT_AVAILABLE／SINGLE-SEED_CONCLUSION_REOPENED／PAIRWISE_RECHECK_REQUIRED**。

---

## 9. 目前研究決策鏈

截至 2026-08-12：

1. 目前continuous模型研究以`DL-CONT12B / MR-12B`作current anchor；`DL-CONT12C / MR-12C`已完成controlled replay並REJECTED，保留歷史重現；`DL-CONT12A / MR-12A`保留為previous MSE anchor。
2. A9 hard-filter 使用方式 `SR-C7 / SR-C8 / SR-C10` 已淘汰。
3. Resource-aware 使用方式已解掉主要曝險問題；`SR-C11`仍是A9 resource-aware variants中的歷史較佳經濟結果，`SR-C15`則是不同continuous source的最新promising arm，兩者期間／source不同不得直接混成單一排名。
4. 使用者目前研究原則不是最佳化「Min ROOS sorting 與 DL sorting 的比例」，而是**固定在資源契約下最大化 PASS 使用，再改善 PASS 品質**。
5. 因此 `SR-C12` 雖經濟績效低於 C11，仍是「最大化 PASS」方向的 runtime research base。
6. `AUD-a9-pass-persistence` 已確認 false PASS 平均存活較久，並在 candidate-day pool 被放大。
7. `SR-C13` candidate-day re-score 已在實作前取消並永久保留ID，原因是extended candidate通常不是A9的breakout-event訓練分布。
8. `AUD-a9-selection-confidence`已完成；A9 confidence在真正DL Selection Mode競爭場景的整體排序力弱，不建立confidence-priority arm。
9. 舊continuous ranker曾有正向Target排序能力，但歷史raw score sort同時顯著降低資金利用；因此不能把舊策略失敗直接等同模型無效。
10. `SR-C14`已分配並實作為capital-utilization-first continuous controlled arm：Min ROOS先決定資源模式；capital-utilization mode完全不改，只有cash-binding的DL Selection Mode才使用`DL-CONT11G` frozen OOS continuous score。
11. `SR-C14`不使用A9 PASS／REJECT、不重訓`MR-11G`、不新增score threshold或Min ROOS／DL混合權重；若本機缺MR-11G既有research OOS工件，策略比較必須BLOCKED而不是自動重訓。
12. `SR-C14`正式結果期間為2021-01-01～2025-12-22（受DL-CONT11G frozen OOS coverage限制）：相對SR-C3報酬+0.73pp、MDD+1.76pp、RoMD-1.00、EV-0.07R、曝險-0.25pp、同參數DL選擇R+14.62R；相對SR-C12報酬-5.52pp、RoMD-0.73、EV-0.24R、同參數DL選擇R-79.99R。判定capital-utilization-first已大幅消除舊raw continuous sort的macro曝險問題，但現有MR-11G score在此deployment仍未形成足夠經濟排序力。
13. `MR-12A`已完成工件並由`DL-CONT12A / SR-C15`進行正式controlled replay；相同capital-first runtime下，all-event training明顯優於MR-11G PASS-only deployment：C15相對C14報酬+11.62pp、MDD-2.36pp、RoMD+2.24、EV+0.06R，曝險只差-0.03pp，支持`all_labels` training semantics。
14. `SR-C15`正式結果為Return=168.69%、MDD=14.81%、RoMD=11.39；同期間相對C3為+12.35pp Return／-0.60pp MDD／+1.24 RoMD，相對C12為+6.10pp Return／-1.66pp MDD／+1.51 RoMD。惟EV=0.64R、same-param DL selection R=+26.78R均弱於C12，且`AUD-c15-strategy-attribution`已確認相對C3/C12的全期優勢高度依賴2024，因此目前標記`PROMISING_NOT_PROMOTED`。
15. `AUD-c15-source-attribution`已完成：C15相對C14 Return +11.61pp、MDD -2.36pp、RoMD +2.24、EV +0.06R、final relative wealth +4.52%；2024約+3.44%，非2024仍+1.04%，因此`MR-12A / DL-CONT12A`升為`SOURCE_SUPPORTED`，但不等同`SR-C15` promotion。
16. C15自身盤前診斷仍顯示相對Min ROOS預計選入單數`+24`但reserved capital累計`-1,063,551`，正式證明cash-binding不等於basket-level capital preservation。`SR-C16`因此只改selector feasibility：任何DL basket不得降低Min ROOS同日selected count或exact reserved capital；模型／Target／score source／參數全部固定。
17. `SR-C16`正式結果已取得：Return=175.15%、MDD=15.86%、RoMD=11.05、EV=0.48R；盤前resource violation=0證明資源保護契約成立，但相對C15的RoMD／EV退步，因此標記`RESOURCE_CONTRACT_PASSED / NOT_PROMOTED`。
18. `SR-C17`正式結果已取得：Return=164.13%、MDD=14.38%、RoMD=11.42、EV=0.72R；相對C3為+7.79pp Return、-1.03pp MDD、+1.27 RoMD、+0.06R EV，且K/resource violation=0；惟223個Max-DL eligible日中209日需repair、26日最終fallback Min ROOS，故標記`RESULT_AVAILABLE / SELECTOR_NOT_FROZEN`。
19. `SR-C18`只處理C17搜尋完整性與計算時間：K、R0、MR-12A、execution order皆不變；exact global search因N=30／80壓力案例超過10秒而拒絕進production，正式實作改採C17合法seed後的best-feasible single-swap ascent直到1-swap local optimum，並直接量測selector total／median／p95／max CPU time。
20. 使用者要求後續同時保留`SR-C17`與`SR-C18`作固定validation harness：每個新DL source都分別跑兩個selector，先看同selector下model-only gain，再看同一新模型是否在較Max-DL的C18上轉化更好；C17/C18不得依新模型OOS修改。
21. `MR-12B / DL-CONT12B / SR-C19 / SR-C20`正式結果已取得：C19相對C17 Return +17.62pp但MDD +1.81pp、RoMD -0.19、EV -0.02R；C20相對C18 Return +38.98pp、MDD -0.25pp、RoMD +2.86、EV +0.27R、same-param DL selection R +81.57R。Pairwise objective判定`MODEL_ECONOMIC_IMPROVEMENT_SUPPORTED`，且C18較Max-DL harness對模型品質差異的轉化更強；C17/C18因此持續雙保留。
22. `MR-12C / DL-CONT12C / SR-C21 / SR-C22`正式結果已取得並REJECTED：C21-C19 Return -3.08pp、RoMD -0.21、EV -0.02R、selection R -10.95R；C22-C20 Return -63.03pp、MDD +2.00pp、RoMD -5.33、EV -0.39R、selection R -114.79R。C22 Selected Score總和增量仍達+15.028且251/251 eligible日為1-swap local optimum、K/resource violation=0，說明C18確實更完整最大化MR-12C score，但該score與實現選股品質失配；故淘汰ListNet objective、恢復MR-12B為active model anchor，C17/C18雙harness維持凍結。
23. `MR-12B` Selection PIT模型Gate已PASS（2011～2020、10 folds、coverage 100%、年度rho>0=10/10），但`SR-C24 / SR-C25`固定historical P2參數的PIT直接部署相對`SR-C23`分別Return `-19.40pp/-12.03pp`、selection R `-35.78R/-19.52R`，故`DL-CONT12B-PIT`不升格策略source。
24. `AUD-c23-c25-pit-realization`已確認PIT失敗是Target→Realized realization gap，而非已被證實的參數機械瓶頸。C24/C25雖提高平均Target R，exclusive loss主要由winner capture不足造成；fill/sizing/holding/exposure均未達既定機械門檻，因此**目前不得進參數適應**。下一個read-only優先診斷是PIT-specific fold/score-level drift與mixed-fold runtime conditioning；若不能解釋winner capture loss，再回到Target／realized outcome語意研究。
25. `AUD-c23-c25-pit-fold-runtime`結果已取得，`drift=True`不是主要失敗來源：C24 mixed-fold days反而`+13.91R`、single-fold`-46.05R`，fold-boundary window內`+2.17R`而outside`-37.95R`；C25 boundary內`+4.93R`而outside`-24.45R`。因此淘汰cross-fold score normalization／fold-transition runtime作目前主線，不進校正。
26. `AUD-c23-c25-pit-target-realization`已取得結果：C25在共同age quantile的Q1～Q3 Selection ΔR合計`+21.74R`，Q4(age `23～288`日、median `38.5`)單獨`-40.18R`，支持**C25 feasible-ascent的old-signal tail**是下一個Selection-only runtime假說；C24則Q2與Q4皆大幅為負、不是單調age效應。Target↔Realized仍為正且Age Spearman接近0，因此不得把結果解讀成Target全面失效或建立全域candidate expiry；下一步若實作，只允許測試「過舊score不得驅動DL membership change、候選本身仍保留並回退Min ROOS」的受控runtime guard，之後再凍結至OOS。不得重啟已停止的11J counterfactual。
27. `SR-C26`已依上述Selection假說實作：唯一scientific change是`stale_score_membership_guard_max_age_days=22` calendar days。C26先保留C25的K/R0與canonical exact reservation；若C17/C25 seed membership變化牽涉stale scored candidate則回到Min ROOS seed，之後只允許不牽涉stale scored candidate的hard-feasible DL-improving single swap。Unscored candidate不因本guard被視為stale；有有效score但日期不可稽核則保守禁止其改membership。C26不得刪候選、改score、改Target、改模型、改參數或依OOS重調22日。
28. `SR-C26` Selection結果已取得：相對C25 Return `+4.32pp`、RoMD `+0.04`、EV `+0.08R`、same-param selection R `+27.51R`，underfilled end days `-40`、position-gap slot-days `-146`，證明stale-score guard方向有效；但相對C23仍Return `-7.71pp`、MDD `+1.66pp`、RoMD `-0.59`，僅EV `+0.05R`與selection R `+7.99R`為正，因此`STALE_GUARD_SUPPORTED / NOT_PROMOTED`，**不進Forward-OOS，也不得調22日門檻**。後續`AUD-c23-c26-pit-portfolio-translation`已完成risk-dollar分解：C26-only risk-weighted R=`0.21R`低於C23-only=`0.27R`，206筆common trades的`-41.6k` PnL差幾乎完全由risk-size/path effect解釋且common ΔR≈0；沒有新的可事前觀測泛化runtime機械原因，故C26微調鏈正式停止。
29. `MR-13A` Stage 1 Forward-OOS模型Gate已PASS：all-stock OOS daily rho=`0.1755`、pair=`56.12%`；breakout slice daily rho=`0.1490`、pair=`56.91%`，raw Top-K Target/Lift=`1.6198R/+0.4056R`。相對MR-12B breakout raw Top-K近乎持平、pair/boundary略升但daily rho略降，因此授權歷史PIT，未升格continuous anchor。
30. `MR-13A` Stage 2 Selection PIT模型Gate已PASS：2013-04-01～2020-12-31、8 folds、778,532 target-valid groups、100% coverage；all-stock global/daily rho=`0.0783/0.1111`、pair=`53.84%`、spread=`+0.3319R`，年度rho與spread皆8/8正向；breakout slice global/daily rho=`0.0580/0.0784`、pair=`53.98%`、Top-K Lift=`+0.1102R`、Boundary=`51.49%`。此結果授權Selection strategy translation，但不代表已取代MR-12B。
31. Stage 3 preflight發現兩個strategy-runtime契約問題：Stage 2 daily PIT score universe仍只含future target完整rows，使score存在與否間接受未來40 bars影響；既有ranking lookup又固定使用breakout `signal_date`，會把daily model退化成事件日score。這兩點不推翻target-valid model audit，但舊778,532-row PIT table不得進策略。已將daily score eligibility改成feature-history-only、target-valid僅用於train/audit，並讓daily runtime每個盤前決策使用最新已完成交易日score；舊manifest缺此contract時fail-fast。新增`DL-CONT13A-PIT`、`SR-C27/C28`作C24/C25 source-only同參數對照，**不繼承C26 22-day stale guard**。PIT runtime重建已完成；取得C27/C28 Selection結果前仍不得進ROOS或promotion。 PIT重建另允許一條performance-only checkpoint重評路徑：只有同fold的train／validation／final-refit contract、model/training/source identity、selected epoch與checkpoint hash全部不變時，才可重用Stage 2權重並只對新增feature-eligible rows重新inference；任一training-side差異即完整重訓。
32. `MR-13A` Stage 3 PIT runtime rebuild已完成：8 folds全部走checkpoint重評、完整重訓=0，耗時約01:14；PIT score groups=`778,532`、coverage=`100.00%`，Audit all-stock global/daily rho=`0.0783/0.1111`、pair=`53.84%`，breakout slice daily rho=`0.0784`、pair=`53.98%`，與Stage 2 target-valid模型證據一致。score groups未增加不代表future-independent contract未生效；在2013-04-01～2020-12-31實際Selection期間，feature-eligible集合可與target-valid集合等大。Stage 3策略Gate以新manifest的`feature_history_only` contract及daily information-date lookup為準，因此`DL-CONT13A-PIT`已可供C27/C28 Selection比較；仍不得據此promotion或直接進ROOS。
33. Current五欄Min ROOS下C23/C24/C25/C27/C28 Selection Strategy Compare已完成。C27 minimum-repair相對C24 Return `+19.27pp`，但相對C23仍Return `-5.73pp`、MDD `+1.00pp`、RoMD `-0.51`，故不採用。C28 feasible-ascent相對C25 source-only Return `+50.33pp`、MDD `-0.80pp`、RoMD `+2.89`、EV `+0.19R`、same-param selection R `+83.72R`；相對C23 Return `+43.90pp`、MDD `-1.71pp`、RoMD `+2.80`、EV `+0.29R`、selection R `+109.91R`，年度Return改善6/7；相對C27 Return `+49.62pp`、RoMD `+3.31`、EV `+0.21R`。因此MR-13A Selection strategy Gate PASS，凍結C28 feasible-ascent語意並保留`DL-CONT12B/MR-12B`為正式anchor直到Forward-OOS完成。下一步已保留`DL-CONT13A`與`SR-C29`作2021+ Forward-OOS controlled validation；不得再以Selection結果調selector、stale cutoff、capital objective或模型。
34. Forward-OOS 2×3 historical矩陣已完成並產生正式結果：C1 Full=123.26%/MDD17.41/RoMD7.08/EV0.53R；C3 Min=86.21%/13.12/6.57/0.91R；C20 Min+MR-12B=172.40%/18.75/9.20/1.08R；C29 Min+MR-13A=71.86%/15.71/4.57/0.46R；C30 Full+MR-12B=133.21%/17.92/7.43/0.61R；C31 Full+MR-13A=122.00%/16.21/7.53/0.59R。C29-C20純source Return `-100.54pp`、RoMD `-4.62`、EV `-0.62R`，C31-C30 Return `-11.21pp`、RoMD `+0.09`、EV `-0.02R`；因此MR-13A不升級。C30/C31結果永久保留，但current改善矩陣縮為C1/C3/C20/C29四arm。
35. Selection Full historical row `PARAM-P4 / C32/C33/C34`已完成：C32 Full baseline Return=132.25%、MDD=15.85%、RoMD=8.34、EV=0.71R；C33 Full+MR-12B=146.25%/18.91/7.73/0.56R；C34 Full+MR-13A=167.12%/18.44/9.06/0.75R，C34-C33 source-only Return `+20.87pp`、RoMD `+1.33`、EV `+0.19R`。Selection支持MR-13A但與Forward結論反轉；C33/C34保留永久歷史結果，current Selection改善矩陣縮為C32/C23/C25/C28四arm。
36. `apps/research.py → [3] 策略組合比較`永久拆成config-driven `selection_pit`與`forward_oos`兩個profile；兩者共用同一strategy comparison engine，但各自持有period、enabled arms/contrasts、parameter sources與獨立output root（`outputs/strategy_compare/selection_pit/`、`outputs/strategy_compare/forward_oos/`）；兩profile只可透過config宣告的`reuse_output_roots`唯讀掃描舊`outputs/strategy_compare/`作pair/P2 migration，不得把新輸出寫回legacy root。模型工作類型的「準備策略比較所需模型工件」必須解析所有profiles的model dependencies並去重；不得再靠改寫單一config的enabled arms在Selection與Forward之間切換。
37. Selection與Forward結果反轉後，current MR-13A改善研究將兩階段比較對象統一縮減為四arm：Selection=`C32 Full / C23 Min / C25 Min+MR-12B / C28 Min+MR-13A`；Forward=`C1 Full / C3 Min / C20 Min+MR-12B / C29 Min+MR-13A`。Full只保留DL-off策略體系baseline；C33/C34/C30/C31仍是永久scientific IDs與可重現歷史結果，但inactive。Engine正式允許standalone DL-off comparator，不得為滿足舊pair schema而暗中啟用Full+DL arm。
38. 2026-08-11首批8-seed Forward robustness把單一seed結論重開：MR-12B RoMD Mean/Median=`8.21/7.70`，MR-13A=`9.97/10.87`；MR-13A勝Min／Full=`5/8`，MR-12B=`4/8`，但MR-13A CV較高`0.37 vs 0.32`。因此seed 42的C20/C29差異仍是canonical單一run證據，不能再外推成跨seed模型優劣；MR-13A維持NOT_PROMOTED，下一個正式判讀改以相同seed matched-pair ΔRoMD與同參數DL選擇R為主。Robustness schema v3已補canonical直接選擇R與matched-pair輸出。
39. 2026-08-12 robustness schema v5從既有raw `seed_results.csv`補出同seedDL選擇R：MR-13A相對MR-12B為`6/8`勝、`2/8`敗，Δ Mean/Median/Std=`+50.64R/+46.36R/77.12R`、Min/P25/P75/Max=`-41.02R/+4.25R/+74.36R/+210.64R`；相對地ΔRoMD仍`4/8`勝、`4/8`敗。故MR-13A更新為`MATCHED_SELECTION_R_EDGE / MATCHED_ROMD_INCONCLUSIVE / PORTFOLIO_TRANSLATION_UNRESOLVED / NOT_PROMOTED`：ranking跨seed改善已有多數方向支持，但尚不能證明能穩定轉成wealth-path風險報酬。下一個零重訓判讀先以同一raw seed aggregate直接配對`ΔDL選擇R`與`ΔReturn/ΔMDD/ΔRoMD/ΔEV`；只有確認ranking改善與portfolio結果明顯脫鉤後，才值得重建trade/path級attribution，不先擴seed、不跑Selection PIT multi-seed、不開新MR。
40. 2026-08-12 robustness schema v6完成零重訓ranking→portfolio配對：6個`ΔDL選擇R>0` seeds中只有3個`ΔRoMD>0`；ranking/RoMD方向一致=`4/8`、相反=`4/8`，Spearman(`ΔDL選擇R`,`ΔRoMD`)=`0.476`、Spearman(`ΔDL選擇R`,`ΔReturn`)=`0.333`。S1/S7/S8為ranking↑／RoMD↓，S6為ranking↓／RoMD↑；其中S1與S8主要是Return反向，S7則Return為正但MDD惡化使RoMD轉負。故上一項的條件已成立：MR-13A ranking edge存在但經濟轉化明顯受trade selection、risk-dollar sizing與wealth path影響，下一步固定進`AUD-forward-robustness-portfolio-translation`全8-seed attribution；MR-12B繼續作current anchor。在該Audit前不擴seed、不跑Selection PIT multi-seed、不開新MR，也不依discordant seeds調selector／loss／hyperparameter。
41. 2026-08-12 Forward robustness portfolio Audit首次兩次執行均在S1被consistency gate擋下。第一層closed-trade match key分叉已修正；第二次仍失敗後確認真正剩餘問題是comparison basis不同，而非compact evidence損壞：`seed_results.csv`的同seed`ΔDL選擇R`等於「MR-13A相對共同Min DL-off baseline的direct-selection R」減「MR-12B相對同一baseline的direct-selection R」；Audit `direct-pair exclusive ΔR`則直接比較MR-13A與MR-12B trade partitions。兩者都合法但不具代數恆等關係，故不再作equality gate。修正後Audit同時保留baseline-relative `ΔDL選擇R`、direct-pair exclusive/common/all-trade ΔR及`selection_basis_gap_r`，只把同一basis內可閉合的trade decomposition當一致性證據；scientific fingerprint、8 seeds與既有compact source均不變，不需重訓／replay。

---

## 10. Registry 維護契約

每次提出／實作／完成實驗或 Audit 時：

1. 編號前先讀本 Registry。
2. 新 ID 必須先在 Registry 建立 identity，再出現在新程式註解、設計文件或 Experiment Log 的後續實驗名稱。
3. 詳細設計、基準 ZIP／SHA256、唯一變更、固定條件、結果、採用／淘汰理由仍寫入 `doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md`。
4. Reject／Stop／legacy ID 不得刪除或回收。
5. 程式若因相容性保留舊 alias，文件在可能歧義的情況下仍必須使用 canonical namespace。
6. Registry 與 Experiment Log 衝突時，先停止新實驗，於同一輪完成 identity／狀態 reconciliation。

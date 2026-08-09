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
| Strategy params baseline | `PARAM-P2 / Min ROOS` | DL-off-trained rolling active params | ACTIVE strategy parameter baseline |
| Resource-aware risk-quality reference | `SR-C15` | All-event continuous capital-utilization-first | RESULT_AVAILABLE／PROMISING_NOT_PROMOTED；C16雖有更高Return，但C15仍有較低MDD與較高RoMD／EV；AUD-c15確認其相對C3/C12優勢高度集中2024，因此仍不升格正式基準 |
| 最大化 PASS 研究基準 | `SR-C12` | A9 resource-aware best-improvement basket | ACTIVE；研究方向固定為「最大化 PASS 使用，再提高 PASS 品質」 |
| 最新已完成Audit結果 | `AUD-c23-c25-pit-target-realization` | `SR-C23` vs `SR-C24 / SR-C25` Selection PIT Target／Score→Realized R與score-age attribution | RESULT_AVAILABLE／C25_OLD_SIGNAL_TAIL_SUPPORTED；C25 covered exclusive trades中Q1～Q3 Selection ΔR合計=`+21.74R`，Q4(age `23～288` calendar days，median `38.5`)單獨=`-40.18R`，使整體selection R維持負值；C24則Q1=`+31.73R`、Q2=`-50.95R`、Q3=`-9.41R`、Q4=`-21.61R`，不支持通用單調age cutoff。Target↔Realized仍為正（C25-only `rho=0.35`），故目前優先形成C25 feasible-ascent的Selection-only stale-score runtime guard假說，而非重做Target／MR-12B |
| Continuous research DL source | `DL-CONT11G` | `MR-11G / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_pass_magnitude_mse` frozen OOS continuous score | HISTORICAL controlled-replay source；SR-C14未採用 |
| Current model research | `MR-12B` | No-time all-event within-day Pairwise RankNet ranker；固定all-event Target／InceptionTime，以同日pairwise logistic直接學cross-sectional ordering | RESULT_AVAILABLE／PIT_MODEL_VALIDATION_PASS／FROZEN_OOS_MODEL_ECONOMIC_IMPROVEMENT_SUPPORTED／SELECTION_PIT_STRATEGY_NOT_PROMOTED；Selection PIT 2011-01-01～2020-12-31共10 folds、23,932/23,932 groups、coverage `100%`，PASS-only global rho=`0.2336`、mean daily rho=`0.1859`、top-bottom spread=`1.1386R`；年度rho>0=`10/10`、spread>0=`9/10`，Gate=`PASS`。此audit是MR-12B單模型絕對Gate，證明歷史PIT排序訊號穩定為正，但**不等同MR-12B在每個PIT fold都優於MR-12A**。Frozen Forward-OOS的C19/C20仍支持Pairwise objective；但Selection PIT固定historical P2 params的C24/C25相對C23分別Return `-19.40pp/-12.03pp`、same-param DL selection R `-35.78R/-19.52R`，故PIT score直接部署不升格策略。C24/C25仍改善post-replay Future Target，但經濟結果反向，支持Target→realized portfolio R存在realization gap；`AUD-c23-c25-pit-realization`已取得結果：C24/C25 aggregate capture分別較C23下降`-0.41/-0.26`、exclusive selection R=`-35.78R/-19.52R`。fill、平均投入、holding與平均曝險均未跨既定mechanical-gap門檻，因此目前**不支持直接進入參數適應**；不據此修改MR-12B loss、architecture或OOS selector。 |
| Previous continuous DL anchor | `DL-CONT12A` | `MR-12A / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_mse` frozen OOS continuous score | SOURCE_SUPPORTED historical anchor；保留作MSE基準 |
| Current continuous DL anchor | `DL-CONT12B` | `MR-12B / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_pairwise` frozen OOS continuous score | SOURCE_SUPPORTED_FOR_MODEL_RESEARCH；C19/C20正式controlled replay支持pairwise learning objective |
| Current Selection PIT strategy source | `DL-CONT12B-PIT` | `MR-12B / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_pairwise` Selection point-in-time score | RESULT_AVAILABLE／PIT_GATE_PASS／STRATEGY_NOT_PROMOTED；2014～2020固定historical P2 Min ROOS下，C24/C25皆改善post-replay Future Target但相對C23經濟績效與same-param DL selection R皆為負；保留作read-only attribution與歷史重現，不作正式策略source |
| Current Selection runtime experiment | `SR-C26` | `SR-C25` feasible-ascent + Selection-frozen 22-calendar-day stale-score membership guard；candidate保留、只禁止stale score驅動membership change | IMPLEMENTED／RESULT_PENDING；只做2014～2020 Selection controlled replay，先比C26-C25純runtime效果與C26-C23是否恢復經濟優勢；OOS前門檻已凍結 |
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

---

## 5. Runtime DL source Registry

| Canonical ID | 相容 alias | Backing identity | Runtime 語意 | 狀態 |
|---|---|---|---|---|
| `DL-A9` | `A9` | `MR-9A / ARCH-inception_time_v1 / PROFILE-unique_group_sampling` | Binary breakout-quality score，threshold 0.5 | ACTIVE research source |
| `DL-TP1` | `TP1` | `LABEL-a2_realized_trade_path_v1` training line | Realized trade-path binary score，threshold 0.5 | Hard-filter use rejected；歷史保留 |
| `DL-CONT11G` | `CONT11G` | `MR-11G / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_pass_magnitude_mse` | Frozen OOS continuous breakout-event score；只供capital-utilization-first controlled strategy research。歷史training scope為PASS-only，SR-C14在全部orderable breakout events上的使用屬受控deployment hypothesis | HISTORICAL research-only score source；SR-C14未採用 |
| `DL-CONT12A` | `CONT12A` | `MR-12A / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_mse` | Frozen OOS all-event continuous breakout-event score；training scope=`all_labels`，Target仍為`strategy_aligned_opportunity_no_time_r_v1`；作為後續controlled selector研究的固定score source | SOURCE_SUPPORTED |
| `DL-CONT12B` | `CONT12B` | `MR-12B / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_pairwise` | Frozen OOS all-event continuous score；runtime仍使用softmax PASS probability，training為同日pairwise ranking；固定C17/C18 selector controlled replay已支持其模型經濟改善 | SOURCE_SUPPORTED_FOR_MODEL_RESEARCH |
| `DL-CONT12B-PIT` | `CONT12B_PIT` | `MR-12B / ARCH-inception_time_v1 / PROFILE-strategy_aligned_no_time_all_event_pairwise` | Selection point-in-time continuous score；每個歷史日期只使用當時合法fold模型輸出；Strategy Compare只讀score／manifest／audit且要求PIT model Gate PASS，不得自動訓練或重建 | RESULT_AVAILABLE／PIT_GATE_PASS／STRATEGY_NOT_PROMOTED；C24/C25相對C23 Return `-19.40pp/-12.03pp`、selection R `-35.78R/-19.52R` |
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
| `SR-C20` | Min ROOS: MR-12B max-DL feasible-ascent | 完全沿用SR-C18 K/R0、feasible-ascent與execution-order語意；唯一差異為score source `DL-CONT12A → DL-CONT12B` | RESULT_AVAILABLE／MODEL_VALIDATION_ARM；Return=179.54%、MDD=14.48%、RoMD=12.40、EV=0.84R、same-param DL selection R=+55.31R；相對C18 Return +38.98pp、MDD -0.25pp、RoMD +2.86、EV +0.27R，顯示較Max selector能強烈放大DL品質差異 |
| `SR-C21` | Min ROOS: MR-12C max-DL constrained basket | 完全沿用SR-C19／SR-C17 K/R0、minimum-repair、action-prefix與execution-order語意；唯一差異為score source `DL-CONT12B → DL-CONT12C` | RESULT_AVAILABLE／MODEL_VALIDATION_ARM／NOT_ADOPTED；Return=178.68%、MDD=16.22%、RoMD=11.02、EV=0.67R、selection R=+14.09R；相對C19全面小幅退步 |
| `SR-C22` | Min ROOS: MR-12C max-DL feasible-ascent | 完全沿用SR-C20／SR-C18 K/R0、feasible-ascent與execution-order語意；唯一差異為score source `DL-CONT12B → DL-CONT12C` | RESULT_AVAILABLE／MODEL_VALIDATION_ARM／NOT_ADOPTED；Return=116.51%、MDD=16.49%、RoMD=7.07、EV=0.45R、selection R=-59.48R；相對C20 Return -63.03pp、RoMD -5.33、EV -0.39R、selection R -114.79R，顯示較Max selector強烈放大MR-12C ranking弱點 |
| `SR-C23` | Selection Min ROOS PIT baseline | 固定2014～2020 historical A2-teacher P2 Min ROOS active params、rules全關、DL off；作Selection PIT策略經濟驗證共同baseline | RESULT_AVAILABLE／BASELINE；Return=127.45%、MDD=25.45%、RoMD=5.01、EV=0.62R、Exposure=87.68%、379 trades |
| `SR-C24` | Selection PIT: MR-12B minimum-repair | 與SR-C23完全相同historical P2 params；唯一新增為`DL-CONT12B-PIT`，selector完全沿用SR-C17 minimum-repair K/R0 contract | RESULT_AVAILABLE／NOT_ADOPTED；Return=108.05%、MDD=24.51%、RoMD=4.41、EV=0.53R、selection R=-35.78R；相對C23 Return -19.40pp、RoMD -0.60、EV -0.09R；Future Target mean +0.0476R但未轉成realized economics |
| `SR-C25` | Selection PIT: MR-12B feasible-ascent | 與SR-C23完全相同historical P2 params；使用`DL-CONT12B-PIT`並完全沿用SR-C18 feasible-ascent K/R0 contract | RESULT_AVAILABLE／NOT_ADOPTED；Return=115.42%、MDD=26.36%、RoMD=4.38、EV=0.59R、selection R=-19.52R；相對C23 Return -12.03pp、RoMD -0.63、EV -0.03R。相對C24 Return +7.37pp、selection R +16.26R，顯示feasible-ascent改善selector轉化但不足以扭轉PIT source的經濟劣勢 |
| `SR-C26` | Selection PIT: MR-12B feasible-ascent stale-score membership guard | 完全沿用SR-C25的historical P2 params、`DL-CONT12B-PIT`、K/R0與feasible-ascent；唯一runtime變更為Selection預先凍結`score age > 22 calendar days`的有效舊PIT score不得參與DL造成的basket membership change。候選不刪除、不過期、不hard reject，仍保留Min ROOS membership／orderability；fresh↔fresh feasible-ascent維持原C25語意 | IMPLEMENTED／RESULT_PENDING／SELECTION_ONLY／CUTOFF_FROZEN_22D；門檻只來自AUD-c23-c25-pit-target-realization的Selection evidence（C25 Q1～Q3合計+21.74R、Q4 age 23～288日=-40.18R），OOS不得再調22日門檻 |

### `SR-C13` identity boundary

`SR-C13`曾正式規劃為candidate-day re-score，依Registry「已使用ID永久保留」規則不得改名重用。2026-08-07在實作前取消：`DL-A9`是以breakout-event snapshot訓練的classifier，而extended candidate-day通常不是breakout線型；把非breakout state直接餵回同一模型會改變輸入分布與score語意，因此不把這條路徑當成既有DL的乾淨策略使用方式。Candidate validity仍由原策略唯一負責。

若未來真的要學extended／candidate-state品質，必須另立新的`MR-*`模型研究；若只改既有`DL-A9`在portfolio中的排序／allocation，則使用下一個新的`SR-C*`，不得重用`SR-C13`。

---

## 7. Audit Registry

| Canonical ID | Config ID | Source | 狀態 | 主要結論 |
|---|---|---|---|---|
| `AUD-a9-pass-quality` | `a9-pass-quality` | `SR-C12 / DL-A9` | RESULT_AVAILABLE | Raw A9 PASS score 整體單調性弱；不支持直接 score sorting 或 age cutoff |
| `AUD-a9-pass-persistence` | `a9-pass-persistence` | `SR-C12 / DL-A9` | RESULT_AVAILABLE | False PASS persistence=1.60×；candidate-day FP amplification=1.33×；selector 不是主要放大來源 |
| `AUD-a9-selection-confidence` | `a9-selection-confidence` | `SR-C12 / DL-A9` | RESULT_AVAILABLE／NOT_USED_FOR_PRIMARY_RANK | Candidate-day rho=0.071、unique-event rho=0.102、selected R rho=0.082；每日平均Label concordance=49.18%，不支持A9 confidence作主排序 |
| `AUD-c15-strategy-attribution` | `c15-strategy-attribution` | `SR-C15` vs `SR-C3 / SR-C12` | RESULT_AVAILABLE | 2024單獨relative wealth effect約+18.10%/+11.46%，非2024約-11.25%/-8.19%；C15全期優勢主要由portfolio geometry／slot occupancy／compounding解釋，非平均R提升 |
| `AUD-c15-source-attribution` | `c15-source-attribution` | `SR-C15 / MR-12A` vs `SR-C14 / MR-11G` | RESULT_AVAILABLE | C15相對C14 Return +11.61pp、MDD -2.36pp、RoMD +2.24、EV +0.06R、relative wealth +4.52%；2024 +3.44%、非2024 +1.04%，exclusive selection +12.16R／+196,527.93 PnL；MR-12A all-label source支持保留 |
| `AUD-c23-c25-pit-realization` | `c23-c25-pit-realization` | `SR-C23` vs `SR-C24 / SR-C25` | RESULT_AVAILABLE／REALIZATION_GAP_CONFIRMED／PARAM_ADAPTATION_NOT_SUPPORTED | C24/C25平均Target R分別`+0.04R/+0.06R`，但平均Realized R`-0.09R/-0.03R`、aggregate capture`-0.41/-0.26`、exclusive selection R=`-35.78R/-19.52R`。Exclusive R分解顯示兩者主要損失皆來自winner capture不足：C24 winner contribution約`-29.52R`、loser contribution約`-6.26R`；C25 winner contribution約`-22.03R`、loser contribution反而`+2.51R`。fill/sizing/holding/exposure皆未達既定mechanical-gap門檻，故capture惡化只證明realization gap，不足以直接支持Selection參數適應 |
| `AUD-c23-c25-pit-fold-runtime` | `c23-c25-pit-fold-runtime` | `SR-C23` vs `SR-C24 / SR-C25` + `DL-CONT12B-PIT` fold identity | RESULT_AVAILABLE／FOLD_DRIFT_NOT_PRIMARY_CAUSE | C24 mixed-fold `+13.91R`、single-fold `-46.05R`、boundary outside `-37.95R`；C25 mixed-fold `-4.25R`、single-fold `-10.65R`、boundary outside `-24.45R`。fold drift存在但負Selection R並未集中於mixed-fold／fold boundary，故不支持cross-fold normalization作下一步。 |
| `AUD-c23-c25-pit-target-realization` | `c23-c25-pit-target-realization` | `SR-C23` vs `SR-C24 / SR-C25` + `DL-CONT12B-PIT` + original event Target | RESULT_AVAILABLE／C25_OLD_SIGNAL_TAIL_SUPPORTED／C24_NON_MONOTONIC | Actual exclusive trades coverage C24/C25=`93.85%/93.66%`。C25 Q1～Q3 Selection ΔR=`+5.67/+13.31/+2.76R`，合計`+21.74R`；Q4(age `23～288`日、median `38.5`)=`-40.18R`，且C25-only Q4 realized mean=`0.42R` vs C23-only=`1.50R`。C24負R則集中Q2與Q4而非單調隨age惡化。Target↔Realized rho仍為正（C24-only `0.23`、C25-only `0.35`），Age↔Realized接近0，故只支持C25尾端stale-event runtime假說，不支持全域age cutoff或直接改Target公式；未成交／未選候選仍不建立counterfactual R。 |

Audit 固定 read-only。Audit 結果可形成 `SR-*` 或 `MR-*` 假設，但 Audit 自己不占用這兩種 ID。

---

## 8. Parameter／Artifact Stage Registry

| ID | 定義 | 狀態／用途 |
|---|---|---|
| `PARAM-P2` | DL-off-trained risk-only rolling active params | Current Min ROOS parameter baseline |
| `PARAM-P3-TP1` | TP1-on-trained parameter stage | Historical；not promoted |
| `PARAM-P3-A9` | A9-on-trained parameter stage | Historical；not promoted |

`P1/P2/P3` 是策略參數訓練 stage／artifact identity，不是 model version，也不得拿來當 scientific experiment ID。

---

## 9. 目前研究決策鏈

截至 2026-08-08：

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

---

## 10. Registry 維護契約

每次提出／實作／完成實驗或 Audit 時：

1. 編號前先讀本 Registry。
2. 新 ID 必須先在 Registry 建立 identity，再出現在新程式註解、設計文件或 Experiment Log 的後續實驗名稱。
3. 詳細設計、基準 ZIP／SHA256、唯一變更、固定條件、結果、採用／淘汰理由仍寫入 `doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md`。
4. Reject／Stop／legacy ID 不得刪除或回收。
5. 程式若因相容性保留舊 alias，文件在可能歧義的情況下仍必須使用 canonical namespace。
6. Registry 與 Experiment Log 衝突時，先停止新實驗，於同一輪完成 identity／狀態 reconciliation。

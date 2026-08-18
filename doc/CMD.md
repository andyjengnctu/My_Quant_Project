# 常用指令

python apps/research.py optimizer --dataset full --timing --trials 10 `效能驗證`
python apps/research.py optimizer --dataset full --outer-oos --timing --trials 10 --outer-first-oos-date 2021-01-01 --outer-last-oos-date 2026-01-01 --outer-window-mode fixed --outer-train-window-months 120 --outer-oos-months 12 --yes `rolling效能驗證`

## 環境 / 測試

```bash
python requirements/export_requirements_lock.py
python apps/run_bundle.py
python tools/local_regression/run_all.py --only quick_gate
python tools/validate/preflight_env.py
```

- 正式對外入口為 `apps/run_bundle.py`；一般本機 double check 與交付打包直接執行：`python apps/run_bundle.py`。
- `apps/run_bundle.py`順序固定為 stage → commit current snapshot → package ZIP → formal test；formal test失敗時仍保留已建立的commit與ZIP，供閉環修正與交付。
- `apps/test_suite.py`是`run_bundle.py`內部formal test runner；只有在針對正式測試器本身除錯時才直接執行。
- 只有正式入口已指出失敗步驟時，才用 `python tools/local_regression/run_all.py --only ...` 重跑指定步驟。
- `python tools/validate/preflight_env.py` 只檢查環境，不自動安裝依賴。
- `outputs/local_regression/_staging/`：formal / validate 暫存 staging；屬 `local_regression` 內部子目錄，會由 retention 自動清理。

## 打包

```bash
python apps/run_bundle.py
python apps/package_zip.py
```

- 一般交付使用`apps/run_bundle.py`，先commit當前snapshot並建立ZIP，再執行formal test；測試FAIL不回滾該commit或刪除ZIP。
- `apps/package_zip.py`直接模式保留給單純snapshot／歷史相容用途，不取代正式整合入口。

## 主工具入口

```bash
python apps/research.py optimizer
python apps/portfolio_sim.py
python apps/smart_downloader.py
python apps/vip_scanner.py
python apps/workbench.py
```

- `apps/` 只作正式入口；模組責任與依賴方向以 `doc/ARCHITECTURE.md` 為準。

# Workbench

- `apps/workbench.py` 為 GUI 正式入口，也是單股 trade-analysis 的單一使用者入口。
- Workbench 上方控制列提供股票代號輸入、常用股票下拉、候選股掃描與歷史績效股掃描。
- K 線檢視中，交易明細與 Console 為獨立分頁。
- 日常 GUI 問題先檢查 `tools/workbench_ui/single_stock_inspector.py`，再看 `tools/workbench_ui/workbench.py`。

# Trade analysis

- `tools/trade_analysis/trade_log.py` 提供單股 trade-analysis 共用 backend / 開發輔助 CLI；正式使用者入口仍為 `apps/workbench.py`。
- 為維持相容性，保留 legacy `run_debug_*` API 名稱。
- 對外建議使用 canonical `run_trade_analysis` / `run_trade_backtest` / `run_prepared_trade_backtest` / `run_ticker_analysis` aliases。


# Breakout quality filter

- 所有簡易報表共用統一console格式；使用者可見的工件路徑一律從專案根目錄顯示相對路徑並使用`/`分隔。易讀內容直接顯示於console，持久工件依命令保留Markdown／JSON／CSV，不產生HTML。Breakout Quality策略層正式結果一律保證console摘要＋Markdown簡易報表；JSON／CSV／manifest只作詳細工件。Strategy Compare即使命中REUSE/cache，也會由canonical pair JSON重新產生本次run的Markdown並顯示相同簡報，不會因重用而只列工件路徑。

## 研究資料、訓練與評估

研究工作統一由`apps/research.py`進入；主選單只選工作類型。模型標的由`config/research.py`指定，Audit module由`config/audit.py`指定，策略比較arms／contrasts由`config/strategy_compare.py`指定。`tools/filters/breakout_quality/`的直接CLI只保留開發與歷史研究用途，不保留Audit或strategy-compare legacy相容入口。

互動式 PowerShell／Terminal 直接執行下列指令會開啟 Research 單一正式入口；主選單只選工作類型。選擇 `[1]  模型訓練  (Enter)` 後，才進入 `config/research.py` 指定 active model 的既有模型選單。Dataset、單獨 train、export及歷史版本化research audit仍可透過 `python apps/research.py model <command>` 執行。目前 Binary 模型研究固定研究 `a2_realized_trade_path_v1`：只建立新Label、訓練並顯示模型預測報表；策略經濟效果由「策略組合比較」工作類型依`config/strategy_compare.py`執行；舊Label／A2 no-DL專用Gate只保留研究CLI。continuous workflow仍維持模型與策略分開。

```bash
python apps/research.py
```

```text
====================================================================================================
 Research
====================================================================================================
[1]  模型訓練  (Enter)
[2]  策略參數最佳化
[3]  策略組合比較
[4]  Audit／診斷
[5]  查看目前設定與工件狀態
[0]  離開
```

策略組合比較的Multiple-seed robustness由同一`config/strategy_compare.py`驅動；正式CLI可用：

```bash
python apps/research.py compare robustness status
python apps/research.py compare robustness run
python apps/research.py compare robustness latest
```

`run`會依`config/strategy_compare.py`對應robustness profile的`stochastic_arm_ids`，以deterministic generated seeds逐一建立隔離模型／score並做final strategy replay；`fixed_arm_ids`只計算一次。對C56/C57這種dual-model arm，每個seed會用同一seed建立primary與secondary兩個source（Selection=`CONT13K_PIT + CONT13M_PIT`；Forward=`CONT13K + CONT13M`），兩者都READY後才做該seed唯一一次strategy replay；不得只vary primary而固定secondary。robustness membership與single-seed arm `robustness_role`分離，啟用[3]/[4]不得改寫[1]/[2] scientific identity。GPU training queue與CPU replay queue依config worker數重疊。checkpoint、full score與完整replay tree仍依retention清除；`keep_attribution_source=true`時會在清除前只永久抽取active `trades/equity/daily_capacity/selected_buys/execution`的gzip compact attribution source；`execution`只保存entry execution的qty、risk-budget、actual initial risk與binding診斷，不保存完整orderable universe。舊completed robustness若scientific observation已存在但compact source缺失，再次`run`會顯示`REBUILD ATTRIBUTION`，使用完全相同scientific fingerprint／resolved seeds重建缺失工件，compact unit先以`PENDING`落盤，並驗證selected epoch／fold count、正式策略metrics與逐年報酬與既有observation一致；既有model/score SHA可得時亦須一致，全部通過後才標記`VERIFIED`供resume/Audit使用。程序若在驗證前中止，PENDING unit下次不得列READY；整個流程不建立新的scientific結果。

Robustness報表前四區與一般Selection PIT／Forward-OOS Strategy Compare共用同一canonical metric registry與renderer；Multi-seed專屬的RoMD分布、same-seed contrasts、seed-by-seed delta與跨seed年度統計保留在後段。`latest`若偵測到舊report schema，會直接以既有`seed_results.csv`、`seed_yearly_returns.csv`與compact attribution source做report-only refresh並覆寫同run的summary/report；此refresh不得呼叫trainer、score builder或strategy replay，因此已完成或正在執行中的scientific run不需為報表格式更新重跑。舊run當時未永久保存的per-seed model prediction／Future Target conversion欄位會顯示`-`；未來新run會在清理暫存model report前保留可直接取用的小型canonical model metrics。

選擇 `[4] Audit／診斷` 會進入固定Audit子選單；Audit module由`config/audit.py`指定，不在選單中選擇：

```text
=== Audit／診斷 ===
[1]  執行目前 Audit 設定  (Enter)
[2]  查看 Audit 設定、工件與預計動作
[3]  查看最近 Audit 結果
[0]  返回
```

全專案Audit inventory與dispatch的單一真理位於`tools/audit/catalog.py`；`config/audit.py`只保存目前active formal policy。正式入口為Research主選單的`[4] Audit／診斷`或`python apps/research.py audit`，由`tools/audit/runner.py`派送。Formal Audit只能使用catalog中`mode=formal`且`read_only=true`的handler，缺件顯示`BLOCKED`，不得自行重跑策略、建立Label、訓練模型或修改runtime。一次性Audit／research diagnostic在決策完成、Registry／Log已留證且current runtime／active Audit／必要compatibility無依賴後即退役，不以`enabled=False`或historical CLI永久累積。`meta quality`會執行advisory slimming scan並在summary列出`CLEAN／REVIEW`與候選數；該訊號不單獨造成formal FAIL。Candidate validity仍維持Strategy owns validity / DL owns quality / Portfolio selector owns allocation。

Audit也可直接由Research CLI子命令執行：

```bash
python apps/research.py audit
```

目前`config/audit.py`沒有啟用中的formal Audit；Research `[4] Audit／診斷`仍保留泛化入口，當沒有enabled definition時顯示停用狀態。最近一次`AUD-mr13km-frozen-rank-fusion`已於2026-08-18取得`REJECT_EQUAL_RANK_SCORE_FUSION`結果並依一次性研究生命週期退役；其identity與結果只保留於Registry／Experiment Log，不再留formal handler或dedicated synthetic。

模型訓練與策略比較維持不同工作類型與service責任。正式策略組合比較可由主選單 `[3]` 進入，或執行：

```bash
python apps/research.py compare
```

Current CLI profile：

```bash
python apps/research.py compare extending_window_rolling status
```

```text
=== 策略組合比較 ===
[1]  Pre-Test 策略比較  (Enter)
[2]  Extending-Window Rolling 策略比較
[3]  Extending-Window Rolling Multi-seed robustness
[4]  查看目前Framework設定與工件狀態
[0]  返回
```

若未來重新啟用Runtime整合Gate，該項目由config動態插入，不固定選單號碼。已移除「一次執行全部 Multi-seed robustness」入口。

進入策略比較後，第二層選單固定為：

```text
[1]  執行目前比較設定  (Enter)
[2]  查看設定、工件與預計動作
[0]  返回
```

Current Extending-Window Rolling自2016起，以完整合法歷史的expanding training + annual PIT-safe refit形成單一策略績效主線；Fixed-Window Rolling屬模型研究選單中的120M歷史穩定性診斷，不建立第二套Strategy Compare truth。舊Selection PIT／Frozen Forward profile與工件只供歷史解讀／重現，不暴露於current主選單。

目前比較profiles、比較對象、參數來源、DL來源、差異比較與前置建立政策全部條列於`config/strategy_compare.py`；每個profile用`arm_ids`／`contrast_ids`決定當階段啟用集合，arm／contrast定義本身仍可保留歷史項目，不存在代表整套實驗的`ACTIVE_STRATEGY_COMPARISON_ID`。同一param source／rule policy使用一個共用DL-off基準，可同時掛多個DL-on模型；各DL-on arm可獨立開關，執行引擎會逐一與同一基準形成controlled pair，並驗證重複基準結果一致。選擇一般Strategy Compare執行後，App先顯示`READY／PREPARABLE／BLOCKED`依賴計畫並只確認一次；對可由既有正式工件確定產生的缺件，依config自動重用、重建或接續，包括既有模型的forward-OOS scores與比較所需的策略參數。App不建立Label、不選模型、不訓練模型權重；Selection PIT scores／manifest／audit與PIT Model Gate一律由`[1] 模型訓練`工作類型建立／更新，Strategy Compare缺少或identity/coverage不合法時直接BLOCKED並導向該正式入口，不做checkpoint-only PIT重建或重跑Audit。執行前會由全部啟用DL runtime工件解析共同比較期間，先驗證Full／Min／Min-DL rolling active params是否完整覆蓋；Min ROOS固定使用forward P2 DL-off-trained工件，不得使用只涵蓋Selection的歷史Label teacher params。若forward scores建立後才得知正式期間，App會重新規劃下一波前置並自動建立／接續缺少或過期的P2／P3，全部READY後才開始第一組replay。DL-aware參數必須與訓練時相同的DL版本配對：TP1-trained只允許TP1-on，A9-trained只允許A9-on；跨版本runtime組合在config驗證階段直接拒絕。A9 P3使用獨立`p3_dl_on_trained/A9/`工件，不覆蓋TP1 P3。console／報表採簡稱`Min ROOS`、`Min ROOS: TP1-on`、`Min ROOS: A9-on`、`Min-TP1 ROOS`、`Min-TP1 ROOS: DL-on`、`Min-A9 ROOS`、`Min-A9 ROOS: DL-on`。報表的`同參數DL選擇R`只在相同`param_source`與`rule_policy`的arms間具共同attribution基準；跨參數contrast的`Δ同參數DL選擇R`固定顯示`-`。預設`reuse_completed_results=True`與`reuse_shared_baseline=True`：選單的執行計畫會把replay identity與目前工件SHA完全一致的既有arm顯示為`REUSE`，只有新／失效arm顯示`RUN`；同一param/rules群組的DL-off baseline最多執行一次。例如新增MR-12B的C19/C20時，若C3/C17/C18已有compatible正式結果，計畫應直接重用C3/C17/C18，只執行C19/C20，再組合全部contrasts。修改contrast或報表說明不會使cache失效；param、model、manifest、forward score、期間或runtime contract任何一項改變都必須重新replay。

當目前 workflow 是 Binary classification 時，選擇 `[1]  模型研究與驗證  (Enter)` 後會顯示：

```text
=== Binary DL Filter 模型研究與驗證 ===
Active Research Label：a2_realized_trade_path_v1
[1]  建立新Label → 重新訓練 → 模型預測報表  (Enter)
[2]  使用既有模型 → 更新Scores → 模型預測報表
[3]  查看Label與事件生命週期摘要
[0]  返回
```

`[1]` 固定依序執行：建立／接續A2 realized trade-path Label Dataset、train、research score export、Selection／OOS模型預測報表、forward-OOS runtime score export；到此停止，不執行策略回放。`[2]` 不重新訓練，只更新research scores、重建同一份預測報表並更新forward-OOS runtime scores。`[3]` 顯示PASS／REJECT／EXCLUDED、事件group及初次miss buy／未成交終止契約。新Label使用獨立`filter_id=breakout_quality_a2_trade_path_v1`，不得覆蓋現有9A模型。策略經濟效果由`apps/research.py`的「策略組合比較」依目前啟用arms與contrasts比較；舊`strategy-trade-path-label-gate`只保留歷史研究診斷。

Breakout-quality 模型／Label／training／workflow設定只編輯 `config/breakout_quality.py`；正式Audit對象與診斷設定集中於 `config/audit.py`。檔案上半部是可調設定；下半部集中命名profile、驗證、衍生值與helper。舊`breakout_quality_policy.py`、`breakout_quality_experiments.py`與`breakout_quality_workflow.py`已刪除；任何新舊程式都必須直接import `config.breakout_quality`。

模型研究與策略 workflow 使用**分離的config-driven identity**。`BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE`只決定 `apps/research.py → 模型訓練` 的Active Profile；`BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE`則保留目前已驗證的策略／PIT runtime anchor，模型研究往前推進時不得自動改變策略基準。主選單不硬編MR／model名稱，會依模型研究profile的 `training_objective` 與 `training_sample_scope` 自動派送。event-based continuous profile會先依canonical Dataset refresh contract確認Dataset，再由`prepare-continuous-target`建立／驗證event-style Target artifact；daily-universal profile的Target直接由canonical OHLCV按ticker/date建立，feature採lazy materialization，因此**不建立expanded 300×10 daily feature artifact，也不建立event-style Continuous Target artifact**。兩者均可沿用同一Selection PIT builder／audit；PIT split固定要求training label完成日早於validation／score cutoff，score table不得含Future Target。模型流程到模型報表／PIT audit為止，策略經濟比較只由獨立「策略組合比較」入口執行。

目前模型研究與策略anchor設定為：

```python
BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE = "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise"
BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE = "daily_universal_no_time_full_list_ndcg_pairwise"
BREAKOUT_QUALITY_RANDOM_SEED = 42
```

MR-13E仍是production／reference anchor；current Extending-Window Rolling研究使用MR-13K作primary upside ranking、MR-13M作residual-safety secondary source，策略arms為`C58/C59/C60`。模型訓練的單一Active Profile因此設為MR-13K；`[5] 準備策略比較所需模型工件`再依current Strategy Compare config準備／重用MR-13E、MR-13K、MR-13M三個模型來源。Current Extending robustness預設4 seeds；production identity仍保留既有C42/C44，未經Rolling evidence與明確promotion decision不得自動切換。後續順序以`doc/BREAKOUT_QUALITY_RESEARCH_QUEUE.md`為準，`doc/ToDo.md`只屬使用者私人筆記。

Active continuous model menu由config／research spec動態產生；目前MR-13K會顯示：

```text
=== Continuous DL 模型研究與驗證 ===
Active Profile：daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise
[1]  Pre-Test｜單模型快速驗證  (Enter)
[2]  Extending-Window Rolling 模型驗證
[3]  Fixed-Window Rolling 模型驗證
[4]  查看目前Workflow與工件狀態
[5]  準備策略比較所需模型工件
[6]  比較目前 Target 與 reference Target
[0]  返回
```

MR-13O已於Forward model Gate結案為`REJECTED / NO PIT`，不再作Active Profile。它沿用MR-13H `daily_full_horizon_opportunity_r_v1`作economic result truth，但training target不是該scalar R；trainer會從同一canonical stock-day rows建立`[MFE percentile, low-adverse percentile]`，僅strict Pareto-comparable pairs進`pairwise_logistic` loss。Epoch selection固定看`mean_daily_pareto_pair_concordance`，global Pareto concordance只作tie-break；MR-13H economic Daily rho／Pair／Top-K只能在候選checkpoint評估中作描述性 model-gate evidence，不能參與選模。`selection_pit_authorized=False`；若為歷史重現手動切回MR-13O，Extending／Fixed工作類型入口仍固定顯示，但選入後會明確BLOCKED且不得建立Rolling工件。MR-13O沒有新增scalar Target identity，所以不顯示`[6] Target comparison`。

Forward console／簡易報表除既有economic ranking品質外，MR-13O會額外顯示Validation／Forward／breakout的Pareto pair concordance與comparable-pair coverage。若Pareto supervision本身學不到（接近隨機），此Target-formulation直接在model gate停止；若Pareto可學但economic ordering仍不改善，表示joint dominance supervision與最終economic ordering仍有落差，也不應直接進PIT。只有兩層證據都形成可信增量，才另輪授權Selection PIT。
### Current Rolling 驗證順序（2026-08-18 current）

1. `apps/research.py → [1] 模型訓練 → [1] Pre-Test｜單模型快速驗證`：沿用原本單模型流程；所有合法歷史→2020 Selection refit，2021+只作快速OOS研究Gate。
2. `apps/research.py → [3] 策略組合比較 → [1] Pre-Test 策略比較`：直接REUSE單模型OOS scores，快速比較DL-off/reference/current candidate；只決定是否值得進Rolling。
3. `apps/research.py → [1] 模型訓練 → [2] Extending-Window Rolling 模型驗證`：2016～2025共10個完整年度fold，使用完整合法歷史、annual refit。
4. `apps/research.py → [1] 模型訓練 → [3] Fixed-Window Rolling 模型驗證`：2016～2025固定120M history、annual refit，作歷史learnability診斷。
5. `apps/research.py → [1] 模型訓練 → [5] 準備策略比較所需模型工件`：準備Pre-Test與Extending current sources。
6. `apps/research.py → [3] 策略組合比較 → [2] Extending-Window Rolling 策略比較`：正式執行C58/C59/C60。
7. 若正式Rolling Seed42結果仍有決策價值，再執行`[3] Extending-Window Rolling Multi-seed robustness`；預設4 seeds。沒有combined robustness入口。

Continuous-ranker / Rolling CUDA feeding的current execution default為`train_prefetch_batches=8`、`train_prefetch_workers=4`，維持feature-only ordered prefetch，並使用pinned feature + non-blocking H2D + dedicated CUDA copy stream；complete-host prefetch因實機更慢已回退。Rolling缺少fold另以`fold_workers=2`的spawn獨立process平行執行；已完成fold仍先REUSE，`fold_workers=1`可恢復serial。Multi-seed robustness已有外層GPU trainer平行，因此其nested PIT固定使用1個fold worker。以上都只改execution，不改fold scientific identity或optimizer semantics。


### Continuous Target自動準備

主選單會自動執行；CLI-only入口如下：

```bash
python apps/research.py model prepare-continuous-target --filter-id breakout_quality_v1 --target-id strategy_aligned_opportunity_no_time_r_v1
```

此命令只依目前Dataset與profile準備Target，不訓練模型。已存在Target必須與目前Dataset artifact SHA256一致才會跳過；Dataset重新掃描或group排列改變時會重建。No-time Target直接依固定公式與canonical Dataset identity重建；歷史採用決策保留在Experiment Registry／Log，不再以11E／11F Audit report作runtime prerequisite。命令完成或確認Target已是current時，終端會顯示既有`continuous_target_audit.md`易讀報表路徑；主選單狀態頁同時列出Target manifest與Target audit Markdown。

### Selection point-in-time continuous-ranker Scores

先由目前Dataset／Target自動找出最早合法PIT日期並驗證fold計畫，不訓練或寫入正式Score：

```bash
python apps/research.py model build-point-in-time-scores --score-start-date auto --plan-only
```

確認計畫後批次建立／向前補齊PIT Scores：

```bash
python apps/research.py model build-point-in-time-scores --score-start-date auto --resume
```

`auto`會依實際group、Target valid、label completion、inner validation與最小group門檻逐月解析最早合法日期。Fold目錄採`fold_YYYYMMDD_YYYYMMDD`穩定日期ID；向前延伸時，既有相同日期與完整契約的舊`fold_000`類checkpoint／scores會先驗證hash，再自動遷移重用，不因前面新增fold而全部重訓。 MR-13A由Stage 2 target-valid score universe升級為Stage 3 feature-eligible score universe時，若同fold的inner-train／validation／final-refit identity、model spec、training settings、source contract、selected epoch與checkpoint hash完全一致，會保留既有模型權重並只重評擴充後的score rows；任何training-side差異都會自動退回完整fold重訓。

模型層 audit：

```bash
python apps/research.py model audit-point-in-time-scores
```

模型audit通過且Seed／identity一致後，執行Selection策略比較：

```bash
python -m filters.breakout_quality.strategy_compare_engine --dataset full --comparison-mode score-ranking --filter-id breakout_quality_v1 --score-source selection_point_in_time --model-architecture inception_time_v1 --experiment-profile strategy_aligned_no_time_pass_magnitude_mse --param-policy base-finalist-best --max-positions 10 --rotation off
```

此流程使用`models/research/breakout_quality/selection_strategy_realization/roos_base_best.json`的歷史active params，期間由PIT manifest決定。Score缺失不排除候選、不填0，改為回退原buy-sort；Future Target只在兩組replay完成後離線join，輸出orderable coverage、selected Target percentile、top-k retention與opportunity gap。

Audit完成後會直接在終端輸出表格化易讀摘要，依序呈現執行設定與Score coverage、PASS-only／all-valid核心排序能力、逐年Spearman與top-bottom spread、各fold Score分布與drift、PASS／REJECT重疊、orderable candidate coverage及研究邊界；同一份payload同步輸出完整Markdown與JSON，不另算第二套指標。Audit JSON另以SHA256綁定PIT manifest、Scores、coverage與Continuous Target manifest；任何來源工件改變後都必須重新audit，策略入口不得沿用舊模型gate。報表只評估模型層排序能力，明確標示策略optimizer尚未執行、Future Target未進runtime排序、PIT工件不可直接作forward-OOS runtime。

每個 expanding-window fold 只使用該 score period 以前、且 `label_eval_end_date < score_start` 的資料；Inner Validation 與 epoch selection 也限制在歷史窗內。每個事件只保留模型尚未看過該事件時產生的 Score。串接 Score 工件不含 Future Target，builder manifest 預設 `eligible=false`，只允許模型驗證；策略使用必須等待模型驗證與後續 Score buy-sort 接線完成。

主要工件：

```text
models/filters/breakout_quality/<filter_id>/<architecture>/<profile>/point_in_time/
  selection_point_in_time_scores.csv
  selection_point_in_time_manifest.json
  selection_point_in_time_coverage.csv
  folds/<fold_id>/model.pt
  folds/<fold_id>/scores.csv
  folds/<fold_id>/manifest.json
```

Audit 工件：

```text
outputs/filters/breakout_quality/<filter_id>/<architecture>/<profile>/point_in_time_audit/
  selection_point_in_time_audit.md    # 易讀完整報表
  selection_point_in_time_audit.json  # 完整結構化指標
```

正式策略比較以選單操作為主：

```bash
python apps/research.py compare
```

先選`[2] 查看設定、工件與預計動作`，再選`[1]  執行目前比較設定  (Enter)`並按Enter確認一次。`status`／`run`子命令只供自動化與非互動環境相容，不作一般使用者主要操作流程。

目前策略研究比較聚焦`C3 Min ROOS`、`C16 Min ROOS: All-event Continuous capital-preserving`與`C17 Min ROOS: All-event Continuous max-DL constrained basket`。C16保留既有capital-preserving heuristic作同source selector comparator；C17固定相同`DL-CONT12A / MR-12A` frozen OOS score與Min ROOS參數，Min ROOS只提供每日K筆預留單數與exact reserved-capital floor，stock membership先由DL score Top-K決定，不合法時才作deterministic minimum-repair；basket內執行順序仍沿用Min ROOS rank，且正式action只允許K筆盤前預留單。C17不設score threshold、不加Min ROOS／DL混合權重、不使用Future Target。若MR-12A model／manifest／report／OOS score缺失或identity/hash不一致，正式策略比較顯示`BLOCKED`而不得自動訓練模型。正式比較設定只重跑C3／C16／C17，核心contrast為C17-C16與C17-C3；C12/C14/C15保留歷史對照但目前disabled。

低階研究如需直接檢查canonical engine，可執行`python -m filters.breakout_quality.strategy_compare_engine --help`；正式比較仍一律使用`apps/research.py`的「策略組合比較」。

只有需要重建9D MantisV2 legacy工件時才需安裝固定相依套件。官方 `mantis-tsfm==1.0.0` 宣告 `pandas<3.0`，而本專案鎖定 pandas 3.x，因此必須先安裝相容依賴，再以 `--no-deps` 安裝 Mantis，避免 pip 降級既有資料鏈：

```bash
python -m pip install -r requirements/requirements-mantis-v2.txt
python -m pip install --no-deps mantis-tsfm==1.0.0
python -c "from importlib.metadata import version; from mantis.architecture import MantisV2; print('mantis-tsfm', version('mantis-tsfm'))"
```


只有需要重建9E MOMENT-1-base legacy工件時才需安裝官方選配套件。`momentfm==0.1.4` 的 PyPI metadata 釘死舊版 NumPy、Hugging Face Hub 與 Transformers；本專案不得因此降級既有資料鏈，所以先安裝專案驗證過的 Transformers runtime，再以 `--no-deps` 安裝 MOMENT。使用者目前 Python 3.14 環境仍須由本地正式測試確認相容性：

```bash
python -m pip install --index-url https://pypi.org/simple -r requirements/requirements-moment.txt
python -m pip install --index-url https://pypi.org/simple --no-deps momentfm==0.1.4
python -c "from importlib.metadata import version; from momentfm import MOMENTPipeline; print('momentfm', version('momentfm')); print('transformers', version('transformers')); print('MOMENTPipeline import: OK')"
```

9E legacy runtime 契約固定為 `momentfm==0.1.4` 與 `transformers==5.5.0`。不要直接執行 `pip install momentfm==0.1.4`，否則 pip 可能嘗試把本專案的 NumPy／Hub／Transformers 降到該套件 metadata 所列的舊版本。由於採刻意隔離安裝，`pip check` 仍會依舊 metadata 報告版本不相容，不能用它取代上方版本檢查與正式 `apps/test_suite.py`。

既有 `workflow` CLI相容流程會依active architecture分流：必要時建立 supervised dataset；目前policy使用9A `inception_time_v1`，CLI相容流程依序執行train → export research scores → 產生易讀研究報表。Binary正式互動選單目前改為A2 realized trade-path Label研究，於Selection／OOS模型預測報表後停止；forward-OOS score export與A2 Base／舊Label／新Label策略比較只由明確CLI Gate執行。8F sequence-only保留高覆蓋基準。10A Candidate-conditioned Market Set與Stage 1 Global Market Set均為legacy read-only。9C TS2Vec、9D MantisV2、9E MOMENT與9F Patch Transformer已轉為legacy read-only：Selection-only pretraining chain與外部checkpoint下載／驗證只供歷史工件重建，不再由正式新實驗workflow啟動；報表預設納入 OOS。互動式「產生易讀研究報表」固定讀取最終 OOS 並納入報表，不再詢問；讀取後不得依同一段 OOS 回頭調整 threshold、epochs、learning rate、feature、label 或模型。報表開頭將 Filter ID、統計口徑、Selection/OOS 日期與固定訓練參數合併顯示；後續依序呈現 Epoch 選擇、Selection Confusion Matrix、OOS Confusion Matrix、各資料區段比較、排序與校準診斷、OOS 年度診斷及 OOS 綜合判定。排序診斷固定包含PR-AUC、Precision@50/60/70% coverage、Recall@60% Precision、Brier與ECE，診斷threshold不得用於回頭調整OOS。OOS 綜合判定合併原本的 Selection/OOS 差異與部署判定，依「主要成效、過度篩選防線、輔助診斷」三類編排，並新增逐項判讀欄。資料區段與日期分欄；第 4 區固定精簡為「原始 PASS、模型 PASS、PASS Precision、Precision 絕對、PASS Recall、平均 Score」，依此順序呈現。REJECT Specificity、REJECT NPV、Accuracy 與 Precision 相對僅保留在 Confusion Matrix 下方或 OOS 綜合判定。Confusion Matrix 中央只保留 TP／FN／FP／TN；右側依序顯示「原始PASS → TP + FN」與「原始REJECT → FP + TN」，底部依序顯示「TP + FP → 模型PASS」與「FN + TN → 模型REJECT」。分類品質另以「指標、公式、結果、解釋」表呈現；Precision 絕對／相對另以「指標、公式、結果」表呈現。Confusion Matrix 前不再重複顯示統計口徑或列／欄說明。終端會以淡藍、綠、黃、紅標示重點；Confusion Matrix 僅以綠色標示 TP／TN、紅色標示 FP／FN，原始／模型類別與合計維持中性色，且每一行獨立重設 ANSI 色碼，避免跨格污染。重新導向或測試輸出不插入 ANSI 色碼。Markdown 以相同語意顏色呈現，完整 metrics JSON 會寫入 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/reports/`。批次或需要可重現命令時使用 `workflow`：

```bash
python apps/research.py model workflow --filter-id breakout_quality_v1 --dataset full --experiment-profile unique_group_sampling --epochs 200 --batch-size 128 --evaluation-batch-size 4096 --evaluation-workers 4 --no-parallel-split-evaluation --train-prefetch-batches 0 --preload-feature-bank --device auto --mixed-precision --mixed-precision-dtype auto --deterministic-algorithms --no-allow-tf32 --lr 0.0003 --weight-decay 0.0001 --gradient-clip-norm 1.0 --final-refit-mode selected_epochs --class-weight-mode none --time-weight-mode none --seed 42 --fixed-threshold 0.50 --use-inner-validation --inner-validation-months 24
```

### Event-style Continuous Target 建置

正式workflow只透過`prepare-continuous-target`建立versioned Target arrays；builder位於`services/breakout_quality/continuous_target_builder.py`，不依賴Strategy Compare輸出或歷史Audit report。

```bash
python apps/research.py model prepare-continuous-target --filter-id breakout_quality_v1 --target-id strategy_aligned_opportunity_r_v1
python apps/research.py model prepare-continuous-target --filter-id breakout_quality_v1 --target-id strategy_aligned_opportunity_no_time_r_v1
```

Target公式與array identity仍由`filters/breakout_quality/continuous_target.py`定義；builder只負責依canonical Dataset建立arrays、manifest與描述性Markdown/JSON。歷史11A～11F研究結論留在Experiment Registry／Log，已退役CLI不再是current workflow的一部分。

### 11B同日Percentile Regression

`strategy_aligned_opportunity_r_v1` Target準備完成後，此歷史percentile profile可用既有CLI重現；它是research-only實驗，不加入互動選單：


```bash

MR-12A No-time All-event Continuous Ranker 使用同一No-time Target與InceptionTime，只把training scope改為all-labels；此研究仍為CLI-only：

```powershell
python apps/research.py model prepare-continuous-target `
  --filter-id breakout_quality_v1 `
  --target-id strategy_aligned_opportunity_no_time_r_v1

python apps/research.py model train-continuous-ranker `
  --filter-id breakout_quality_v1 `
  --model-architecture inception_time_v1 `
  --experiment-profile strategy_aligned_no_time_all_event_mse `
  --seed 42
```

完成後由`apps/research.py`的「策略組合比較」正式選單執行`C3 / C12 / C15`；策略比較只重用frozen OOS score，不會自動重訓MR-12A。

python apps/research.py model train-continuous-ranker --filter-id breakout_quality_v1
```

預設profile固定為`strategy_aligned_daily_percentile_mse`，保留9A `inception_time_v1`與2-logit head，以PASS softmax probability回歸同日11A target percentile。訓練與epoch選擇只使用Selection內Inner Train／Validation；OOS在完整Selection重訓與checkpoint寫入後才評估。此命令不重建Dataset、不relabel、不設定threshold，也不產生可供scanner使用的`forward_oos scores.csv`；research scores每個group唯一一列，Selection內另以`selection_role`標示Inner Train／Validation。

主要輸出：

```text
models/filters/breakout_quality/<filter_id>/inception_time_v1/strategy_aligned_daily_percentile_mse/model.pt
models/filters/breakout_quality/<filter_id>/inception_time_v1/strategy_aligned_daily_percentile_mse/manifest.json
outputs/filters/breakout_quality/<filter_id>/inception_time_v1/strategy_aligned_daily_percentile_mse/continuous_ranker_report.md
outputs/filters/breakout_quality/<filter_id>/inception_time_v1/strategy_aligned_daily_percentile_mse/continuous_ranker_report.json
outputs/filters/breakout_quality/<filter_id>/inception_time_v1/strategy_aligned_daily_percentile_mse/continuous_ranker_scores.csv
```

不得把此profile傳給一般`train`／`workflow`或`export-scores --scope forward_oos`；這些入口只接受binary classification profiles。

- `outputs/portfolio_sim/`：投組報表與載入摘要。
- `outputs/vip_scanner/`：scanner issue log。
- `outputs/smart_downloader/`：下載器 issue log。
- `outputs/debug_trade_log/`：`trade_analysis` 單股分析輸出；為維持既有工具鏈相容，暫沿用 legacy 目錄名 `debug_trade_log`。
- `outputs/debug_trade_log/`（trade_analysis legacy output dir）屬既有工具鏈相容邊界。
- `outputs/workbench_ui/`：Workbench GUI runtime 快取；目前用於常用股票中文名稱快取；若 reduced 代碼組變動或缺名，Workbench 會優先查官方 CSV / ISIN 名錄並於必要時做 SSL 容錯與 HTTP fallback。

## 其他文件

- `ARCHITECTURE.md`：分層、正式入口、依賴方向與共享邊界。
- `TEST_SUITE_CHECKLIST.md`：formal test suite 主表、狀態與收斂索引。

### 11G PASS-conditional No-time Magnitude Ranker

No-time Target研究採用固定公式後，此歷史PASS-conditional profile使用既有continuous-ranker CLI重現；不加入互動選單：

```bash
python apps/research.py model train-continuous-ranker \
  --filter-id breakout_quality_v1 \
  --experiment-profile strategy_aligned_no_time_pass_magnitude_mse
```

固定契約：

- Architecture仍為`inception_time_v1`，不新增模型版本。
- Target為`strategy_aligned_opportunity_no_time_r_v1`。
- 同日percentile只由原始Label=PASS groups建立；Inner Train、Validation與完整Selection refit也只使用PASS groups。
- Epoch只依Validation PASS-only mean daily Spearman選擇，同分才比較PASS-only MSE。
- OOS percentile與推論只在完整Selection refit及checkpoint寫入後建立。
- Overall與REJECT結果只作診斷；主要判定為OOS PASS-only排序與actual PASS trades的Score↔R。
- Research-only、無threshold、不得匯出forward-OOS runtime scores或覆蓋9A／11B工件。

輸出位於：

```text
models/filters/breakout_quality/breakout_quality_v1/inception_time_v1/strategy_aligned_no_time_pass_magnitude_mse/
outputs/filters/breakout_quality/breakout_quality_v1/inception_time_v1/strategy_aligned_no_time_pass_magnitude_mse/
```

### 11H～11K 歷史診斷（已完成並退役）

11H PASS realization-gap、11I Selection strategy-realization、11J per-candidate counterfactual、11K portfolio selection-pressure已完成研究決策並退出current CLI。對應implementation、catalog registration與專屬synthetic tests不再隨正式程式維護；歷史結論請查`doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md`與`doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md`。若未來需要重新回答相似問題，應建立新的最小Audit，而不是復活舊命令。

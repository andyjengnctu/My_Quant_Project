# 常用指令

python apps/ml_optimizer.py --dataset full --timing --trials 10 `效能驗證`
python apps\ml_optimizer.py --dataset full --outer-oos --timing --trials 10 --outer-first-oos-date 2021-01-01 --outer-last-oos-date 2026-01-01 --outer-window-mode fixed --outer-train-window-months 120 --outer-oos-months 12 --yes `rolling效能驗證`

## 環境 / 測試

```bash
python requirements/export_requirements_lock.py
python apps/test_suite.py
python tools/local_regression/run_all.py --only quick_gate
python tools/validate/preflight_env.py
```

- 日常一鍵入口：`python apps/test_suite.py`
- 正式對外入口為 `apps/test_suite.py`。
- 只有正式入口已指出失敗步驟時，才用 `python tools/local_regression/run_all.py --only ...` 重跑指定步驟。
- `python tools/validate/preflight_env.py` 只檢查環境，不自動安裝依賴。

## 打包

```bash
python apps/package_zip.py
python apps/package_zip.py --run-test-suite
python apps/package_zip.py --commit-message "chore: package before delivery" --run-test-suite
```

## 主工具入口

```bash
python apps/ml_optimizer.py
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

- 所有簡易報表共用統一console格式；使用者可見的工件路徑一律從專案根目錄顯示相對路徑並使用`/`分隔。易讀內容直接顯示於console，持久工件依命令保留Markdown／JSON／CSV，不產生HTML。

## 研究資料、訓練與評估

模型研究與訓練由`apps/breakout_quality.py`進入；全專案正式Audit由`apps/audit.py`進入，Breakout Quality主選單的`Audit／診斷`只是同一個project-wide backend的domain facade；策略績效比較由獨立`apps/strategy_compare.py`進入。`tools/filters/breakout_quality/`的直接CLI只保留開發與歷史研究用途，不保留Audit或strategy-compare legacy相容入口。

互動式 PowerShell／Terminal 直接執行下列指令會開啟 Breakout Quality 正式模型選單。主選單只保留完整模型工作流程、config-driven Audit與模型設定／工件狀態；Dataset、單獨 train、export及歷史版本化research audit仍使用明確 CLI 子命令。目前 Binary 模型研究選單固定研究 `a2_realized_trade_path_v1`：只建立新Label、訓練並顯示模型預測報表；策略經濟效果由獨立策略比較App依`config/strategy_compare.py`執行；舊Label／A2 no-DL專用Gate只保留研究CLI。continuous workflow仍維持模型與策略分開。

```bash
python apps/breakout_quality.py
```

```text
=== Breakout Quality ===
[1/Enter] 模型研究與驗證
[2] Audit／診斷
[3] 查看模型設定與工件狀態
[0] 離開
```

選擇 `[2] Audit／診斷` 會進入固定Audit子選單：

```text
=== Audit／診斷 ===
[1/Enter] 執行目前 Audit 設定
[2] 查看 Audit 設定、工件與預計動作
[3] 查看最近 Audit 結果
[0] 返回
```

全專案Audit inventory與dispatch的單一真理位於`tools/audit/catalog.py`；`config/audit.py`只保存目前反覆執行的正式active policy，不再承擔所有歷史Audit命令的inventory。正式入口可直接執行`python apps/audit.py`；Breakout Quality主選單的Audit子選單呼叫完全相同的`tools/audit/runner.py`，App本身不硬編碼A9／C15或個別Audit module。`tools/audit/`可跨filter、portfolio、optimizer、data與其他domain；正式`core/`／`filters/`不得反向依賴Audit。Formal Audit只能使用catalog中`mode=formal`且`read_only=true`的handler，只讀既有正式工件；缺件顯示`BLOCKED`，不得自行重跑策略、建立Label、訓練模型或修改runtime。歷史research audit仍由catalog統一登記，但可明確標示`research`／`historical`與`read_only=false`，不會被formal runner誤執行。Candidate validity仍維持Strategy owns validity / DL owns quality / Portfolio selector owns allocation。


全專案Audit也可由獨立正式入口執行：

```bash
python apps/audit.py
```

目前`config/audit.py`預設啟用`c15-strategy-attribution`。它只讀`outputs/strategy_compare/latest/manifest.json`所指向的最新正式比較工件，比較`C15 vs C3`與`C15 vs C12`，使用exact daily `Δ log wealth`做可加總的portfolio wealth-path attribution，並拆解changed selection days、common／exclusive trades、reserved／invested／stop-distance capital geometry、slot occupancy與2024集中度。它不重新replay、不重建score、不訓練模型，trade R／PnL只作診斷，正式portfolio超額wealth以`Δ log wealth`為主。輸出位於`outputs/audit/breakout_quality/c15_strategy_attribution/`的`runs/<timestamp>/`與`latest/`。

模型訓練與策略比較使用分離入口。正式策略比較執行：

```bash
python apps/strategy_compare.py
```

```text
=== 策略績效比較 ===
[1/Enter] 執行目前比較設定
[2] 查看設定、工件與預計動作
[0] 離開
```

目前比較對象、參數來源、DL來源、差異比較與前置建立政策全部條列於`config/strategy_compare.py`；arms與contrasts以`enabled`開關，不存在代表整套實驗的`ACTIVE_STRATEGY_COMPARISON_ID`。同一param source／rule policy使用一個共用DL-off基準，可同時掛多個DL-on模型；各DL-on arm可獨立開關，執行引擎會逐一與同一基準形成controlled pair，並驗證重複基準結果一致。選擇執行後，App先顯示`READY／PREPARABLE／BLOCKED`依賴計畫並只確認一次；對可由既有正式工件確定產生的缺件，依config自動重用、重建或接續，包括既有模型的forward-OOS scores與比較所需的策略參數。App不建立Label、不選模型、不訓練模型權重；缺少模型等上游真理工件時才停止並導向模型正式入口。執行前會由全部啟用DL runtime工件解析共同比較期間，先驗證Full／Min／Min-DL rolling active params是否完整覆蓋；Min ROOS固定使用forward P2 DL-off-trained工件，不得使用只涵蓋Selection的歷史Label teacher params。若forward scores建立後才得知正式期間，App會重新規劃下一波前置並自動建立／接續缺少或過期的P2／P3，全部READY後才開始第一組replay。DL-aware參數必須與訓練時相同的DL版本配對：TP1-trained只允許TP1-on，A9-trained只允許A9-on；跨版本runtime組合在config驗證階段直接拒絕。A9 P3使用獨立`p3_dl_on_trained/A9/`工件，不覆蓋TP1 P3。console／報表採簡稱`Min ROOS`、`Min ROOS: TP1-on`、`Min ROOS: A9-on`、`Min-TP1 ROOS`、`Min-TP1 ROOS: DL-on`、`Min-A9 ROOS`、`Min-A9 ROOS: DL-on`。報表的`同參數DL選擇R`只在相同`param_source`與`rule_policy`的arms間具共同attribution基準；跨參數contrast的`Δ同參數DL選擇R`固定顯示`-`。預設`reuse_completed_results=True`與`reuse_shared_baseline=True`：選單的執行計畫會把replay identity與目前工件SHA完全一致的既有arm顯示為`REUSE`，只有新／失效arm顯示`RUN`；同一param/rules群組的DL-off baseline最多執行一次。例如新增MR-12B的C19/C20時，若C3/C17/C18已有compatible正式結果，計畫應直接重用C3/C17/C18，只執行C19/C20，再組合全部contrasts。修改contrast或報表說明不會使cache失效；param、model、manifest、forward score、期間或runtime contract任何一項改變都必須重新replay。

當目前 workflow 是 Binary classification 時，選擇 `[1/Enter] 模型研究與驗證` 後會顯示：

```text
=== Binary DL Filter 模型研究與驗證 ===
Active Research Label：a2_realized_trade_path_v1
[1/Enter] 建立新Label → 重新訓練 → 模型預測報表
[2] 使用既有模型 → 更新Scores → 模型預測報表
[3] 查看Label與事件生命週期摘要
[0] 返回
```

`[1]` 固定依序執行：建立／接續A2 realized trade-path Label Dataset、train、research score export、Selection／OOS模型預測報表、forward-OOS runtime score export；到此停止，不執行策略回放。`[2]` 不重新訓練，只更新research scores、重建同一份預測報表並更新forward-OOS runtime scores。`[3]` 顯示PASS／REJECT／EXCLUDED、事件group及初次miss buy／未成交終止契約。新Label使用獨立`filter_id=breakout_quality_a2_trade_path_v1`，不得覆蓋現有9A模型。策略經濟效果由`apps/strategy_compare.py`依目前啟用arms與contrasts比較；舊`strategy-trade-path-label-gate`只保留歷史研究診斷。

Breakout-quality 模型／Label／training／workflow設定只編輯 `config/breakout_quality.py`；正式Audit對象與診斷設定集中於 `config/audit.py`。檔案上半部是可調設定；下半部集中命名profile、驗證、衍生值與helper。舊`breakout_quality_policy.py`、`breakout_quality_experiments.py`與`breakout_quality_workflow.py`已刪除；任何新舊程式都必須直接import `config.breakout_quality`。

模型研究 workflow 由 `config/breakout_quality.py` 的 `BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE` 決定，主選單不綁定9A或11G名稱。程式讀取該profile的 `training_objective` 自動派送：目前binary classification主選單固定執行A2 realized trade-path Label的Dataset／train／research score／Selection與OOS模型報表／forward-OOS score，策略比較不在選單自動執行；daily percentile regression先以同一套Dataset refresh contract檢查Full dataset與全部股票，缺少、過期或policy不一致時自動完整重建，只有Label policy改變時快速relabel。接著執行泛用`prepare-continuous-target`：依profile的`continuous_target_id`檢查manifest、group count、Dataset policy與來源artifact SHA256，缺少或stale時自動建立目前Target，再執行Selection point-in-time Score builder與模型audit。策略設定預設為`auto`：binary自動解析為`hard-filter / canonical_runtime / original buy-sort`，continuous自動解析為`score-ranking / selection_point_in_time / breakout_quality_score_desc`。Binary新Label模型流程在Prediction報表與forward-OOS模型score工件完成後停止；策略經濟比較僅由獨立`apps/strategy_compare.py`執行。continuous策略選項會讀取Selection PIT manifest／audit並要求模型層gate通過，使用歷史nested active params比較Baseline與Score Sort；不會回退誤用canonical runtime scores。


目前正式 workflow 已切回9A binary classification。設定位置仍只有 `config/breakout_quality.py`：

```python
BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE = "unique_group_sampling"
BREAKOUT_QUALITY_RANDOM_SEED = 42  # 所有breakout-quality正式模型流程共用
```

若策略三項維持 `auto`，選單會自動顯示並執行hard-filter對照。切回continuous ranker時，只將profile改回 `strategy_aligned_no_time_pass_magnitude_mse`。Binary、continuous、pretraining與Selection PIT workflow全部共用`BREAKOUT_QUALITY_RANDOM_SEED`；目前值為42。只有單次重現特殊實驗時才用CLI `--seed`覆寫。PIT日期／fold設定只在continuous objective下生效。


### A2 Realized Trade-path Label研究

新Label以一個原始breakout event的完整生命週期為單位。初次`Low > orig_limit`只把事件留在pending／continuation，不下REJECT；後續回到原始limit成交後，直接重用正式initial stop、trailing、indicator exit與費稅帳務，只有終局`realized_net_r > 0`標PASS，其餘已成交完整交易標REJECT。新setup覆蓋舊延續訊號、shadow completion／invalidation、資料結尾仍未成交或成交後尚未結算者標INVALID並排除Binary訓練。同一ticker/date group只讓當日A2 active `high_len`事件取得有效Label，其餘high_len rows保持INVALID。Feature snapshot固定原始signal date，不因延續等待日重建。

選單`[1/Enter]`會自動執行等價於：

```powershell
python apps/breakout_quality.py build-trade-path-labels `
  --dataset full `
  --filter-id breakout_quality_a2_trade_path_v1 `
  --resume
```

Builder先以2014～2020 Selection rolling基準建立rules全關／DL關的risk-only A2 teacher，再合併既有2021～2026 P2 active params。每個Label日期只能使用當時已生效teacher params；衍生Dataset沿用9A 300×10 feature bank，但以獨立filter目錄保存Label、events與summary。

確認新模型Prediction報表後，正式策略比較使用獨立App。舊Old／New專用Gate僅保留歷史診斷；若需重現，可直接執行：

```powershell
python -m tools.filters.breakout_quality.strategy_trade_path_label_gate `
  --dataset full `
  --param-policy base-finalist-best `
  --max-positions 10 `
  --rotation off
```

Gate不重新訓練，只更新兩個既有凍結模型的forward-OOS Scores，並在同一套A2／P2 active params、rules全關、原position-aware buy-sort與threshold 0.5下比較：A2 Base（DL關）、Old Label 9A（DL開）、New Trade-path Label（DL開）。主判定為`New−Base`，輔助判定為`New−Old`；不得依同一段OOS回頭調整threshold、Label或模型。輸出位於`models/research/breakout_quality/trade_path_label_gate/a2_realized_trade_path_v1/`。

### Continuous Target自動準備

主選單會自動執行；CLI-only入口如下：

```bash
python apps/breakout_quality.py prepare-continuous-target --filter-id breakout_quality_v1 --target-id strategy_aligned_opportunity_no_time_r_v1
```

此命令只依目前Dataset與profile準備Target，不訓練模型。已存在Target必須與目前Dataset artifact SHA256一致才會跳過；Dataset重新掃描或group排列改變時會重建。No-time Target為目前active workflow已接受的固定公式時，可直接重建既定arrays，不要求每次重新執行歷史11E研究gate；原`audit-no-time-target`未帶workflow rebuild flag時仍保留原research-only gate語意。命令完成或確認Target已是current時，終端會顯示既有`continuous_target_audit.md`易讀報表路徑；主選單狀態頁同時列出Target manifest與Target audit Markdown。

### Selection point-in-time continuous-ranker Scores

先由目前Dataset／Target自動找出最早合法PIT日期並驗證fold計畫，不訓練或寫入正式Score：

```bash
python apps/breakout_quality.py build-point-in-time-scores --score-start-date auto --plan-only
```

確認計畫後批次建立／向前補齊PIT Scores：

```bash
python apps/breakout_quality.py build-point-in-time-scores --score-start-date auto --resume
```

`auto`會依實際group、Target valid、label completion、inner validation與最小group門檻逐月解析最早合法日期。Fold目錄採`fold_YYYYMMDD_YYYYMMDD`穩定日期ID；向前延伸時，既有相同日期與完整契約的舊`fold_000`類checkpoint／scores會先驗證hash，再自動遷移重用，不因前面新增fold而全部重訓。

模型層 audit：

```bash
python apps/breakout_quality.py audit-point-in-time-scores
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
python apps/strategy_compare.py
```

先選`[2] 查看設定、工件與預計動作`，再選`[1/Enter] 執行目前比較設定`並按Enter確認一次。`status`／`run`子命令只供自動化與非互動環境相容，不作一般使用者主要操作流程。

目前策略研究比較聚焦`C3 Min ROOS`、`C16 Min ROOS: All-event Continuous capital-preserving`與`C17 Min ROOS: All-event Continuous max-DL constrained basket`。C16保留既有capital-preserving heuristic作同source selector comparator；C17固定相同`DL-CONT12A / MR-12A` frozen OOS score與Min ROOS參數，Min ROOS只提供每日K筆預留單數與exact reserved-capital floor，stock membership先由DL score Top-K決定，不合法時才作deterministic minimum-repair；basket內執行順序仍沿用Min ROOS rank，且正式action只允許K筆盤前預留單。C17不設score threshold、不加Min ROOS／DL混合權重、不使用Future Target。若MR-12A model／manifest／report／OOS score缺失或identity/hash不一致，正式策略比較顯示`BLOCKED`而不得自動訓練模型。正式比較設定只重跑C3／C16／C17，核心contrast為C17-C16與C17-C3；C12/C14/C15保留歷史對照但目前disabled。

低階研究如需直接檢查canonical engine，可執行`python -m filters.breakout_quality.strategy_compare_engine --help`；正式比較仍一律使用`apps/strategy_compare.py`。

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
python apps/breakout_quality.py workflow --filter-id breakout_quality_v1 --dataset full --experiment-profile unique_group_sampling --epochs 200 --batch-size 128 --evaluation-batch-size 4096 --evaluation-workers 4 --no-parallel-split-evaluation --train-prefetch-batches 0 --preload-feature-bank --device auto --mixed-precision --mixed-precision-dtype auto --deterministic-algorithms --no-allow-tf32 --lr 0.0003 --weight-decay 0.0001 --gradient-clip-norm 1.0 --final-refit-mode selected_epochs --class-weight-mode none --time-weight-mode none --seed 42 --fixed-threshold 0.50 --use-inner-validation --inner-validation-months 24
```

### 11A連續目標可學性稽核

11A第一階段只由既有future-path cache建立固定連續target與稽核報表，不訓練模型、不選epoch、不調threshold，也不重建feature bank或Label：

```bash
python apps/breakout_quality.py audit-continuous-target
```

預設會在active 9A的正式模型輸出樹依序搜尋hard-filter `strategy_compare`、`base_finalists_agree`／`base_finalist_best` score-ranking及其他`strategy_compare*`目錄；有metadata時只接受目前filter／architecture／experiment profile且`comparison_design=historical_active_param_oos`的工件。每個目錄先讀`no_filter_round_trips.csv`，若只有`no_filter_trades.csv`則重用canonical交易歸因邏輯在記憶體重建round trips。hard-filter標準目錄優先，避免因舊版只查單一路徑而漏掉既有正式比較工件。也可顯式指定任一來源：

```bash
python apps/breakout_quality.py audit-continuous-target --round-trips <no_filter_round_trips.csv>
python apps/breakout_quality.py audit-continuous-target --trade-history <no_filter_trades.csv>
```

輸出固定在：

```text
outputs/filters/breakout_quality/<filter_id>/continuous_targets/strategy_aligned_opportunity_r_v1/
```

主要回傳`continuous_target_audit.md`與`continuous_target_audit.json`。若來源CSV inventory比既有Dataset新，預設fail-fast；`--allow-stale-source`只供明確知道風險的診斷，不得用於正式比較。

### 11B同日Percentile Regression

11A完整audit通過後，11B固定使用CLI執行；它是research-only臨時實驗，不加入互動選單：


```bash

MR-12A No-time All-event Continuous Ranker 使用同一No-time Target與InceptionTime，只把training scope改為all-labels；此研究仍為CLI-only：

```powershell
python apps/breakout_quality.py prepare-continuous-target `
  --filter-id breakout_quality_v1 `
  --target-id strategy_aligned_opportunity_no_time_r_v1

python apps/breakout_quality.py train-continuous-ranker `
  --filter-id breakout_quality_v1 `
  --model-architecture inception_time_v1 `
  --experiment-profile strategy_aligned_no_time_all_event_mse `
  --seed 42
```

完成後由`apps/strategy_compare.py`正式選單執行`C3 / C12 / C15`；策略比較只重用frozen OOS score，不會自動重訓MR-12A。

python apps/breakout_quality.py train-continuous-ranker --filter-id breakout_quality_v1
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

### 11C Qualified Candidate-set Coverage Audit

11B結果淘汰後，使用既有hard-filter historical active-param strategy comparison重播正式no-filter候選鏈，建立全部OOS／qualified／orderable／actual-trade四層診斷。11C固定使用CLI執行，不加入互動選單：


```bash
python apps/breakout_quality.py audit-qualified-candidate-set --filter-id breakout_quality_v1
```

前置工件：

```text
outputs/filters/breakout_quality/breakout_quality_v1/inception_time_v1/unique_group_sampling/strategy_compare/strategy_comparison.json
outputs/filters/breakout_quality/breakout_quality_v1/inception_time_v1/strategy_aligned_daily_percentile_mse/continuous_ranker_scores.csv
outputs/filters/breakout_quality/breakout_quality_v1/continuous_targets/strategy_aligned_opportunity_r_v1/continuous_target_trade_matches.csv
```

若strategy comparison位於其他明確目錄，可使用：

```bash
python apps/breakout_quality.py audit-qualified-candidate-set --filter-id breakout_quality_v1 --strategy-compare-dir <目錄>
```

輸出位於：

```text
outputs/filters/breakout_quality/breakout_quality_v1/inception_time_v1/strategy_aligned_daily_percentile_mse/qualified_candidate_set_audit/
```

本命令會重播no-filter OOS以取得candidate membership，但不重訓模型、不relabel、不修改9A／11B工件，也不產生runtime scores。qualified候選資格、orderable限制與active params全部由既有portfolio replay決定；audit只讀取並對齊原始`signal_date`。重播後會核對報酬、MDD、RoMD、曝險、PnL與trade counts均未因diagnostic capture改變，並輸出qualified／orderable兩層occurrence與unique-group CSV、每日coverage、actual membership及JSON／Markdown報表。

### 11D Target Component Attribution Audit

11C完成後，使用既有11A component arrays、11B OOS scores與11C qualified／actual-trade工件做Label條件分解。此命令只提供CLI，不加入互動選單：

```bash
python apps/breakout_quality.py audit-target-attribution --filter-id breakout_quality_v1
```

輸出位於：

```text
outputs/filters/breakout_quality/breakout_quality_v1/inception_time_v1/strategy_aligned_daily_percentile_mse/target_component_attribution_audit/
```

主要工件：

```text
target_component_attribution_audit.md
target_component_attribution_audit.json
qualified_target_component_attribution.csv
actual_trade_target_component_attribution.csv
```

本命令固定驗證`target=favorable_r-adverse_r-time_penalty_r`，並在qualified與actual trades分別計算Score／Target成分及realized R關係；另依PASS／REJECT分層。它不訓練、不改target、不調threshold或任何OOS參數。

### 11E Fixed Time-penalty Ablation Audit

11D完成後，只移除11A固定time penalty，檢查`target_no_time_r=favorable_r-adverse_r`是否更貼近actual R。此命令只提供CLI，不加入互動選單：

```bash
python apps/breakout_quality.py audit-target-time-ablation --filter-id breakout_quality_v1
```

輸出位於：

```text
outputs/filters/breakout_quality/breakout_quality_v1/inception_time_v1/strategy_aligned_daily_percentile_mse/target_time_penalty_ablation_audit/
```

主要工件：

```text
target_time_penalty_ablation_audit.md
target_time_penalty_ablation_audit.json
qualified_time_penalty_ablation.csv
actual_trade_time_penalty_ablation.csv
```

本命令只比較原11A Target與固定No-time Target，逐筆驗證`original=no_time-time_penalty`，並輸出overall與PASS／REJECT條件下的Target↔R、Score↔Target及decile差距。它不反向加分time penalty、不搜尋係數、不建立新target arrays、不訓練或修改runtime。

模型架構與訓練實驗分開管理：

```python
# config/breakout_quality.py
BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "inception_time_v1"
BREAKOUT_QUALITY_EXPERIMENT_PROFILE = "unique_group_sampling"
BREAKOUT_QUALITY_PRETRAINING_PROFILE = "ts2vec_selection_only"
```

- `BREAKOUT_QUALITY_MODEL_ARCHITECTURE` 只描述網路與輸入結構。目前policy使用9A-BN `inception_time_v1`與filter id `breakout_quality_v1`；8F `multiscale_cnn_sequence_only_v1`保留高覆蓋基準。10A `inception_time_market_set_candidate_v1`與Stage 1 `inception_time_market_set_v1`均已由完整OOS淘汰並轉為legacy read-only。Stage 1 `inception_time_market_set_v1`修正logical-batch後完整OOS仍低於9A，已轉為legacy read-only，且不具正式forward runtime資格。9F `patch_transformer_v1`完整OOS排序低於9A，已轉為legacy read-only；其10-bar非重疊patch、128維embedding、3層／4-head Transformer、256維MLP、sinusoidal position與patch-mean pooling規格只供舊工件重建。9E MOMENT、9D MantisV2、9C TS2Vec、9B ModernTCN、9A-GN及8P亦為legacy read-only。
- `BREAKOUT_QUALITY_EXPERIMENT_PROFILE` 描述下游supervised optimizer、LR schedule、augmentation與training sampling unit；`BREAKOUT_QUALITY_PRETRAINING_PROFILE`只保留9C legacy TS2Vec encoder的optimizer、epochs、batch、LR、weight decay、gradient clip、crop、mask與contrastive loss；active 9A／8F及legacy 9D／9E／9F均不讀取此profile。profile定義集中在`config/breakout_quality.py`：已接受的9A-BN與高覆蓋基準8F均沿用`unique_group_sampling / time_weight=none`；9B ModernTCN雖沿用相同profile，但完整OOS固定coverage排序全面低於9A，已轉為legacy。8K `unique_group_date_balanced` 已由完整 OOS 淘汰，只保留歷史重現；8J direct best inner checkpoint 同樣只供歷史重現。`baseline`、`adamw_only`、`adam_warmup_cosine`、`history_masking_only` 保留為歷史 profile。
- `multiscale_cnn_sequence_only_dual_path_v1`、`multiscale_cnn_regime_context_v1`、`multiscale_cnn_v2～v8`、`tiny_cnn_v1` 與 `residual_tcn_v1` 保留為 legacy architecture，只供讀取舊 checkpoint、重現既有實驗與稽核歷史 manifest；正常 workflow 不再用它們建立新實驗。
- AdamW、scheduler、augmentation與sampling等訓練方法不建立假模型版本。9A-GN屬正規化結構變更，因此使用獨立architecture `inception_time_group_norm_v1`；唯一差異是8個`BatchNorm1d(128)`改為`GroupNorm(8, 128)`。完整OOS顯示固定coverage排序明顯低於9A-BN，因此已轉為legacy，不再允許正式新訓練；9A-BN與8F保留為accepted比較基準。
- 既有supervised Dataset、feature bank、future-path cache、4維event context arrays與labels可直接沿用，不需重建或relabel。9D legacy工件固定把每個300-bar feature channel獨立線性插值至512，送入釘死 `paris-noah/MantisV2` revision的frozen encoder，取第3層（index 2）CLS＋mean combined embedding，再串接10個channel embedding並只訓練單一linear head；encoder內部以固定chunk切分，避免evaluation batch 4096一次展開40,960條單變量序列造成GPU記憶體尖峰。9C獨立Selection-only rolling-window dataset與pretrained encoder工件同樣只保留歷史重建；9E MOMENT checkpoint下載、凍結encoder與linear head鏈也只供legacy重建。正式9F workflow不執行project pretraining或任何外部checkpoint下載；既有supervised Dataset不需重建。9C、9D與9E鏈只供legacy重建。MantisV2、TS2Vec frozen probe與sequence-only models在forward時明確不讀取event context。8F 在 Inner Train／Final Refit 只保留每個 `ticker/date` 的最小原始 row index，batch size 128 因而代表 128 個 unique groups；early-stopping patience 固定為 1。Validation、Selection、OOS 與報表仍使用完整 rows及既有 `1/group_size` 口徑。每個 architecture/profile 使用獨立工件路徑。

- `multiscale_cnn_sequence_only_v1` 的 trainable parameters 比 v1 少 `4 × 32 = 128`，差異只來自 head 第一層不再接收 `high_len_norm`、`breakout_level_to_close`、`close_to_breakout_level`、`high_to_breakout_level`。


### 10A Candidate-conditioned Market Set 歷史結果

10A已完成完整Selection／OOS，OOS PR-AUC為0.6009，低於9A的0.6257；P@50／60／70%亦分別低0.75／0.41／0.47 pp。雖然threshold 0.5 Recall提高7.30 pp，但模型PASS同步提高7.34 pp，屬coverage放寬，不是排序改善。

因此10A已轉為legacy read-only：

```python
# 正式policy已退回
BREAKOUT_QUALITY_DEFAULT_FILTER_ID = "breakout_quality_v1"
BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "inception_time_v1"
BREAKOUT_QUALITY_EXPERIMENT_PROFILE = "unique_group_sampling"
```

既有10A Market Bank、checkpoint、manifest、research scores與report不需刪除，保留於獨立路徑供歷史重現。正常workflow不得再用10A建立新訓練工件，也不啟動Candidate-conditioned Learned Lag、query數／heads／embedding或Market Set微調。


工件隔離方式：

```text
models/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/
outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/
```

舊版 v1 baseline 工件若仍在沒有 `baseline/` 子目錄的歷史路徑，讀取端會以唯讀相容方式載入；該 fallback 不可作為正式 `forward_oos` 的寫入目標。要更新正式分數，必須先以 `baseline` profile 重新訓練到 canonical profile 子目錄；所有新工件一律寫入 profile 子目錄。

- `workflow` 會自動分成三種處理：工件、profile、ticker coverage、feature/high_len/benchmark/path-cache、欄位契約或來源 CSV inventory 改變時完整重建；只有 label horizon/PASS/REJECT 改變且 horizon 未超過 future path cache 時執行快速 relabel；全部一致時跳過。 完整重建採用 `ticker/date` feature bank 去重、逐檔 CSV 讀取與 per-ticker chunk 合併，正式陣列可 mmap 載入。互動選單只有在判定不需更新時，才詢問「是否強制完整重建 dataset」，預設 N；選 Y 等同 `--rebuild-dataset`。單獨執行 `train` 時若偵測到來源已更新，會 fail-fast 並要求先重建，避免靜默使用過期 dataset。 Dataset 完整重建的逐股票進度固定在同一行刷新，避免大量輸出洗版；重新導向輸出時只保留最終進度摘要與必要的 skip 訊息。
- 尚未準備最終 OOS 評估時，可加 `--no-evaluate-oos`；報表只包含 Selection 內診斷，並明確標示不能作為正式泛化結論。
- 選單只是正式 UI orchestration；dataset、split、training、export 與 evaluation 規則仍只實作在既有子系統，不在 app 複製。
- `BREAKOUT_QUALITY_MODEL_ARCHITECTURE`、`BREAKOUT_QUALITY_EXPERIMENT_PROFILE`、`BREAKOUT_QUALITY_PRETRAINING_PROFILE`、epochs、training batch size、evaluation batch size、evaluation workers、parallel split evaluation、training prefetch、feature-bank preload、`learning rate`、`weight decay`、`gradient clip norm`、final refit mode、class weight mode、time weight mode、`random seed`、最少 train/validation rows、threshold 與 inner-validation 預設均集中於 `config/breakout_quality.py`；互動選單直接採用 policy，不再逐項詢問，CLI 可單次覆蓋且不回寫 policy。Inner Validation 開啟時，正式模型模式依 policy 選擇：`selected_epochs` 在完整 eligible Selection重訓相同 epoch；`matched_optimizer_steps` 匹配更新量；`best_inner_checkpoint` 則直接恢復 best epoch state，不重新初始化、不執行 Final Refit。`class_weight_mode=none` 不對約 55/45 的 Label 額外做 inverse-frequency balancing；`time_weight_mode=none` 是第一階段基準，後續可單獨改為 `year_balanced_sqrt`，以溫和平方根權重降低事件密集年份對 loss 的支配。`evaluation batch size` 與 `evaluation workers` 控制 read-only 推論。完整 Train／Validation／Selection 評估仍以 event rows 為單位，維持原 batch boundaries、全部 requested rows、輸出列序與最終 reduction 順序；score export 對 `use_dataset_context=false` 的模型則以 unique ticker/date feature groups 定義固定 batches，每個 group 只推論一次並廣播至原 event-row 輸出列序。開啟 `parallel split evaluation` 時，Inner Train 與 Validation 的完整評估同時執行，峰值最多使用 `2 × evaluation workers`，但各自仍使用原本的資料列、batch 與模型快照。訓練仍固定單執行緒；`zero_grad(set_to_none=True)`、experiment profile 指定的 Adam／AdamW、weight decay、gradient clipping、RAM preload 與可選的 batch prefetch 均忠實套用並寫入 manifest；除已明確設定的 regularization 外，不改訓練 rows、shuffle 或 batch 邊界。Final refit 模式最後一輪的完整指標會直接沿用；`best_inner_checkpoint` 會在恢復 checkpoint 後對完整 Selection 做一次完整 rows、既有 `1/group_size` 的 group-weighted 研究評估。Research／forward-OOS score export 對 `use_dataset_context=false` 的 sequence-only active model 改為每個 unique ticker/date feature group 只推論一次，再精確廣播到全部 high_len event rows；使用 Dataset context 的 legacy model 才維持逐 event-row 固定 batch 推論。兩條路徑都保留原輸出列序。9A起GPU是正式research路徑；目前policy architecture使用9A `inception_time_v1`與filter id `breakout_quality_v1`；8F保留高覆蓋比較基準，10A `inception_time_market_set_candidate_v1`與Stage 1 `inception_time_market_set_v1`均已轉為legacy read-only，9C `ts2vec_frozen_linear_v1`、9D `mantis_v2_frozen_linear_v1`、9E `moment_1_base_frozen_linear_v1`與9B `modern_tcn_v1`只保留歷史重建：`device=auto`優先CUDA，mixed precision自動優先BF16、否則FP16；deterministic algorithms預設開啟、TF32預設關閉。training execution會寫入checkpoint/manifest；CPU fallback仍保留，但不同device/dtype結果須視為不同execution contract。`train` 終端的 Epoch 選擇每輪顯示完整 Inner Train Loss、Validation Loss、新最佳標記與該 Epoch 總耗時；完整 Selection 重訓沒有獨立 Validation，因此每輪顯示 Train Loss 與耗時。完整研究流程會在每一階段結束後顯示階段耗時，並在最後顯示 workflow 總耗時；互動終端中的時間值使用淡藍色，重新導向或測試輸出不插入 ANSI 色碼。關鍵資料區段與工件路徑保留在終端，完整 history、split policy、overlap、counts 與 Epoch `elapsed_sec` 仍保留於 `manifest.json`，不再將整包 Python dict 印到終端。

也可逐步執行：

```bash
python apps/breakout_quality.py build-dataset --dataset full --filter-id breakout_quality_v1

# 9C TS2Vec已淘汰；build-pretrain-dataset／pretrain只保留歷史工件重建，active 9A workflow不執行。

# 只更新 label；通常由 workflow 自動偵測，不需手動執行
python apps/breakout_quality.py build-dataset --dataset full --filter-id breakout_quality_v1 --relabel-only
# 預設關閉 inner validation：epochs 是完整 Selection 的正式固定訓練次數
python apps/breakout_quality.py train --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --epochs 20 --lr 0.001 --time-weight-mode none --seed 42 --fixed-threshold 0.50 --no-use-inner-validation
# 開啟時：epochs 是搜尋上限；以 Selection 尾端 N 個月選 best epoch，之後依 final-refit-mode 直接採用 best checkpoint 或進行完整 Selection 重訓
python apps/breakout_quality.py train --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --epochs 200 --device auto --mixed-precision --mixed-precision-dtype auto --deterministic-algorithms --no-allow-tf32 --lr 0.0003 --weight-decay 0.0001 --gradient-clip-norm 1.0 --final-refit-mode selected_epochs --class-weight-mode none --time-weight-mode none --seed 42 --fixed-threshold 0.50 --use-inner-validation --inner-validation-months 24
python apps/breakout_quality.py export-scores --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --scope research --device auto --mixed-precision --mixed-precision-dtype auto --deterministic-algorithms --no-allow-tf32 --inference-batch-size 4096 --inference-workers 4 --preload-feature-bank
# 建議日常使用：終端表格報表 + Markdown 解釋報表 + 完整 metrics JSON
python apps/breakout_quality.py report --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --no-include-oos
# 參數與模型已鎖定後，才把最終 OOS 納入報表
python apps/breakout_quality.py report --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --include-oos
# 需要稽核單一 split 的完整原始 JSON 時才使用 evaluate
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --split train
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --split validation
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --split selection
python apps/breakout_quality.py evaluate --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --split oos
# 稽核 Selection 是否涵蓋 OOS 的市場狀態，並歸因指定年度；不重建 Dataset、不重訓、不改 Label
python apps/breakout_quality.py regime-audit --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --focus-year 2022 --min-selection-groups 100 --min-support-share-ratio 0.5
```

`regime-audit` 也可由 `python apps/breakout_quality.py` 的互動選單執行。它以 canonical 300-bar feature bank 中的0050序列，在每個 `ticker/date` breakout group只計算一次事件日可觀測市場狀態。Trend使用0050相對200日均線與60日報酬；Drawdown使用距252日高點；Volatility只以Selection的20日年化波動率三分位數定義low／medium／high，OOS不得參與分箱。輸出固定包含年度覆蓋、各regime的Selection/OOS event share、PASS率、Precision、Recall、PR-AUC、combined-regime支撐數與低代表性標記；`--focus-year` 另輸出該年度各combined regime的TP／FP／TN／FN、年度錯誤貢獻，以及描述性排除low-support事件後的指標。排除比較只作歸因，不得回頭建立年度／regime gate或調整threshold。

- Dataset 與 future-path cache 固定在 `outputs/filters/breakout_quality/<filter_id>/`；research scores 與易讀報表依架構及實驗放在 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/`。一般評估固定輸出 `reports/evaluation_report.md` 與 `reports/evaluation_metrics.json`；regime稽核固定輸出 `reports/regime_coverage_audit.md`、`reports/regime_coverage_audit.json`、`reports/regime_coverage_cells.csv`、`reports/regime_focus_year_cells.csv` 與 `reports/regime_event_groups.csv`。
- Model、manifest 與 `split_assignments.csv` 依架構及實驗固定在 `models/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/`；固定 threshold 也寫入 manifest，OOS 評估不得改用其他值；正式啟用時 active `breakout_quality_score_threshold` 應與該固定值一致。
- 外層正式期間只有 `selection / oos`，日期直接讀 `core.walk_forward_policy`。
- `BREAKOUT_QUALITY_USE_INNER_VALIDATION=False` 時，全部 eligible Selection 固定 epochs 訓練；開啟時，Selection 尾端 `BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS` 個月用來選 epoch。正式預設為 8F `BREAKOUT_QUALITY_FINAL_REFIT_MODE="selected_epochs"`；8I matched optimizer steps與 8J best inner checkpoint 均只保留歷史重現。
- inner validation 只可選 epoch；fixed threshold 仍在 OOS 前鎖定，不可由 validation 或 OOS 自動最佳化。
- Rolling OOS fold 必須沿用既有 `V16_WF_SELECTION_START_DATE`、`V16_WF_SEARCH_TRAIN_END_DATE`、`V16_WF_OOS_START_DATE`、`V16_WF_OOS_END_DATE` policy override；不得另傳一套 breakout-quality 專用日期。
- `research` 分數固定寫到 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/research_scores.csv`，不會改動正式 `scores.csv` 或 model manifest。
- 正式 `models/.../scores.csv` 只能由明確的 `--scope forward_oos` 建立。

## 建立正式 forward-OOS score table

> 10A `inception_time_market_set_candidate_v1`已轉為legacy read-only，不能執行本節的`forward_oos`匯出。以下流程使用目前active 9A。

1. 先以正式 app 的 `build-dataset` 與 `train` 子命令建立完整研究資料並訓練；`train` 依既有 walk-forward policy 執行固定 epoch 模式，或以 Selection 內 validation 選 epoch 後完整重訓，並保留 `model.pt`、`split_assignments.csv`、`manifest.json` 與 `model_information_cutoff`。
2. 若目前 dataset 已包含 outer OOS，可直接匯出；若需延伸到更新資料，只重新執行 `build-dataset`，不可重新 train 同一模型。
3. 執行：

```bash
python apps/breakout_quality.py export-scores --filter-id breakout_quality_v1 --experiment-profile unique_group_sampling --scope forward_oos
```

- `forward_oos` 的策略執行期仍使用同一份 walk-forward OOS window；但盤前下單使用前一交易日訊號，因此 Score 匯出會從 `model_information_cutoff` 當日開始建立訊號 coverage，包含 OOS 首個執行日前必要的 signal anchor。這些前置 Score 只供首日盤前排序／過濾，不會把策略回放或 OOS 評估提前。
- Manifest 會分別記錄 `required_signal_start`、`execution_start`、實際第一個 Score 事件日 `available_from` 與 `available_through`。`required_signal_start` 等於 `model_information_cutoff`；cutoff 當日 Score 只可供下一交易日盤前訂單使用，早於該日不得使用模型；`available_through` 後若出現候選事件則 fail-fast。
- 正式 runtime 只讀目前 policy 架構與 experiment profile 的 `models/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/scores.csv`，並驗證 model/score SHA256、schema、high_len coverage、OOS eligibility 與可用日期。
- `forward_oos` 會以目前原始資料重新建立 runtime 突破候選全集，不再只沿用訓練 Dataset 內可建立特徵的事件。可評分事件使用模型 score；因 benchmark 日期缺失、300-bar feature history 不足或特徵無效而不可評分的正式候選，會寫入同目錄 `unavailable_scores.csv`，並在 `scores.csv` 固定記為 `0.0`（保守 REJECT）。未知缺分仍會 fail-fast。
- 每次原始 CSV inventory 改變或套用本契約修正後，都必須重新執行本命令；不需重訓模型或重跑 Rolling optimizer。
- 現有舊版 `breakout_quality_v1` manifest 不符合新版 artifact contract 時，必須依上述流程重建，不得由 runtime 猜測或自動相容。

## 固定 9A 策略層經濟效果對照

### 正式無前視 OOS 對照

正式比較不能拿「今天才訓練完成」的單一 `models/run_best_params.json` 回放整段 2021～2025；該參數已看過後期資料，不符合歷史交易日使用當時已生效 active param 的原則。應先由 Rolling OOS Optimizer 產生按生效日切換的參數組。不要加 `--timing`，因 timing mode 不寫出正式 `roos_*.json`：

```bash
python apps/ml_optimizer.py --dataset full --outer-oos --trials 10 --outer-first-oos-date 2021-01-01 --outer-last-oos-date 2026-01-01 --outer-window-mode fixed --outer-train-window-months 120 --outer-oos-months 12 --yes
```

目前 Trade selector 對應的正式參數組為：

```text
models/roos_base_finalists_agree.json
```

完成上方 `forward_oos` score 匯出與 Rolling OOS 參數組後執行：

```bash
python -m filters.breakout_quality.strategy_compare_engine --dataset full --params models/roos_base_finalists_agree.json --max-positions 10 --rotation off
```

- 比較期間固定為 runtime manifest 的 `execution_start` ～ `available_through`；Score table 可從更早的 `required_signal_start` 開始，只用來供應首個執行日前的原始 breakout signal Score。Rolling active-param 生效期間只需完整覆蓋實際策略執行期，不需覆蓋前置 Score anchor。
- 每個歷史交易日都使用 Rolling OOS 檔內當日已生效的單一參數或 seed ensemble；兩組交易日期與 0050 benchmark 必須完全一致。
- 工具與 Portfolio Simulator 共用 `core.portfolio_engine`、正式 signal generation、成交／費用／資金／持股延續及 `core.portfolio_stats`；Optimizer 也共用相同核心，但 Optimizer 是參數搜尋流程，策略對照是固定參數 OOS replay，兩者不是相同工作流。
- 兩組完整參數／每個 ensemble member 只能有 `use_breakout_quality_filter=False/True` 一項差異；filter ID 與 threshold 均固定為 active policy／manifest 值，不重新最佳化任何策略參數。
- `Candidate_Supply_Gap` 是「盤前可用持股格數 − 當日可掛單候選數」的非負值，只表示候選供給是否足夠；`End_Position_Gap` 才是成交執行後仍未滿倉的格數。
- 輸出固定在 `outputs/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/strategy_compare/`，包含 Markdown／JSON 主報表、兩組 equity／trade／daily-capacity CSV、年度報酬比較，以及一筆一列的 round-trip 交易歸因報表。
- 正常策略比較完成後會自動產生 `trade_attribution.md/.json`、`trade_attribution_trades.csv`、`trade_attribution_yearly.csv` 與兩組 `*_round_trips.csv`。配對鍵固定為 ticker＋實際進場日＋進場類型；R、PnL、費稅與結算沿用 Portfolio Engine 的 closed-trade 真理來源。
- 若策略比較已經跑完，只需重建歸因與修正部分年度標記，不必再次執行 replay：

```bash
python -m filters.breakout_quality.strategy_compare_engine --attribution-only
```

- `--attribution-only` 只讀取既有 `strategy_comparison.json`、`no_filter_trades.csv`、`quality_filter_trades.csv` 與正式 runtime score；它會把只到 2026-03-02 的 2026 年標為非完整年度，再輸出交易歸因。
- 此對照只判斷固定 9A 是否改善淨報酬、回撤、穩定性及資金使用；不得依結果回頭調整 threshold、epochs、feature、Label 或模型。

### Breakout Quality Score 候選排序探索性比較

### 實驗編號／版本 Registry

Breakout Quality 的實驗 ID、namespace、model architecture、DL source、策略 arm、Audit 與參數 stage 統一登錄於 `doc/BREAKOUT_QUALITY_EXPERIMENT_REGISTRY.md`。任何新實驗或新編號都必須先查 Registry；`MR-9A`（模型研究）、`DL-A9`（runtime source）、`SR-C12`（策略使用方式）屬不同 namespace。`MR-10A` 已被歷史 Candidate-conditioned Query 永久占用，不得重用。詳細結果與理由仍記錄於 `doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md`。

固定 threshold 0.5 hard filter 已由策略 OOS 淘汰；若要測試 9A 的相對排序能力，只使用下列隔離模式：

以 `base_finalist_best` 單一 runtime member 做較純的 Score Ranking ablation：

```bash
python -m filters.breakout_quality.strategy_compare_engine --comparison-mode score-ranking --param-policy base-finalist-best --dataset full --max-positions 10 --rotation off
```

工具會自動使用 `models/roos_base_best.json`，並驗證每個生效日恰有 1 個 member、`min_agree=1`。Baseline 與 score-ranking 兩組都固定 `use_breakout_quality_filter=False`；唯一差異為 `use_breakout_quality_ranking=False/True`。因所有候選票數皆為 1，實際有效排序為「Quality Score 由高到低 → 既有買入排序 → deterministic ticker」。輸出位於 `strategy_compare_score_ranking_base_finalist_best/`。

保留 `base_finalists_agree` 的既有探索性比較時使用：

```bash
python -m filters.breakout_quality.strategy_compare_engine --comparison-mode score-ranking --param-policy base-finalists-agree --dataset full --max-positions 10 --rotation off
```

此模式自動使用 `models/roos_base_finalists_agree.json`；候選先通過 `min_agree`，再依「finalist同意數由高到低 → 同票Quality Score由高到低 → 既有買入排序 → deterministic ticker」，輸出位於 `strategy_compare_score_ranking_base_finalists_agree/`。

Capital-aware ranking為CLI-only研究消融，不加入互動選單、不重訓模型、不執行optimizer。固定既有`base-finalist-best` ROOS、交易規則、fixed risk、position cap、max positions與Selection PIT Score，只改排序政策：

R2 `capital-adjusted-score`：

```bash
python -m filters.breakout_quality.strategy_compare_engine --dataset full --comparison-mode score-ranking --score-source selection_point_in_time --model-architecture inception_time_v1 --experiment-profile strategy_aligned_no_time_pass_magnitude_mse --param-policy base-finalist-best --ranking-policy capital-adjusted-score --start-date 2014-01-01 --end-date 2020-12-31 --max-positions 10 --rotation off
```

R2以正式盤前sizing結果計算`projected_capital_fraction = proj_cost / sizing_capital`，再計算`deployment_rate = min(1, projected_capital_fraction / max_position_cap_pct)`，排序鍵為「`Score × deployment_rate`由高到低 → 既有buy-sort → deterministic ticker」。不得另以停損距離近似`proj_cost`，也不得使用Future Target。輸出隔離於`strategy_compare_score_ranking_base_finalist_best_capital_adjusted_score_selection_point_in_time/`。

R3 `capital-bucket-then-score`：

```bash
python -m filters.breakout_quality.strategy_compare_engine --dataset full --comparison-mode score-ranking --score-source selection_point_in_time --model-architecture inception_time_v1 --experiment-profile strategy_aligned_no_time_pass_magnitude_mse --param-policy base-finalist-best --ranking-policy capital-bucket-then-score --start-date 2014-01-01 --end-date 2020-12-31 --max-positions 10 --rotation off
```

R3只對當日具有有效PIT Score的可掛單候選，依正式`deployment_rate`的當日橫斷面1/3與2/3分位切成高／中／低三桶，先按部署桶高到低，再於桶內按Score高到低，最後沿用既有buy-sort。分桶只使用當日盤前已知候選與正式sizing，不使用Future Target或回放績效調整邊界；同部署率跨分位時保持同桶。輸出隔離於`strategy_compare_score_ranking_base_finalist_best_capital_bucket_then_score_selection_point_in_time/`。

Optional entry filters × Ranking A～E Gate為CLI-only研究，不加入互動選單。Gate固定使用continuous-ranker的filter／architecture／experiment profile；dataset／部位與rotation採正式config，不會因主workflow切回binary `unique_group_sampling`而改錯PIT profile。它固定舊正式ROOS與Selection PIT Scores，依序執行：A目前filters＋原buy-sort、B目前filters＋R3、C五個optional entry filters全關＋原buy-sort、D五個filters全關＋R3、E五個filters全關＋原始Score sort：

```bash
python -m tools.filters.breakout_quality.strategy_filter_gate --dataset full --param-policy base-finalist-best --start-date 2014-01-01 --end-date 2020-12-31 --max-positions 10 --rotation off
```

全關欄位固定為`use_breakout_ema_filter`、`use_bb`、`use_vol`、`use_breakout_return_filter`與`use_breakout_false_filter`；不關閉`high_len`、ATR buy／stop／trail、`use_kc` exit、reclaim re-entry、fixed risk或position cap。主要判讀為`B−A`、`D−C`、`E−C`、`D−E`及R3交互作用`(D−C)−(B−A)`。E用來確認關閉filters後原始Score是否恢復，D−E則判斷R3資金分桶是否仍有必要。Gate不重訓模型、不重建PIT Scores、不執行optimizer；三個pair工件與合併`strategy_filter_gate.md/json`均輸出至`strategy_filter_gate_base_finalist_best_selection_point_in_time/`隔離目錄。

Binary DL Filter Rule Ablation Gate同樣為CLI-only研究，使用9A binary canonical runtime score與固定threshold，不使用continuous PIT Score或R3。它固定正式rolling params與原position-aware buy-sort，將原A／B／C／F重新命名為A0／B0／A1／B1，並加入A2／B2關歷史門檻、A3／B3再關Re-entry、A4／B4再關KC出場。每層A為DL關、B為DL開；未指定日期時自動使用9A runtime manifest宣告的正式execution start與available through：

```bash
python -m tools.filters.breakout_quality.strategy_dl_filter_gate --dataset full --param-policy base-finalist-best --max-positions 10 --rotation off
```

Gate只關閉`use_breakout_ema_filter`、`use_bb`、`use_vol`、`use_breakout_return_filter`與`use_breakout_false_filter`；保留`high_len`突破事件、ATR buy／initial stop／trail、`use_kc` exit、reclaim re-entry、fixed risk、position cap及原buy-sort。`B−A`檢查DL疊加現有filters，`F−C`檢查DL作唯一品質Gate，`F−A`才是DL-only replacement對目前正式策略的採用比較，interaction=`(F−C)−(B−A)`只作機制判讀。兩個hard-filter pair都固定threshold 0.5、canonical runtime score與相同active params；不重訓模型、不調threshold、不執行optimizer、不使用Future Target。輸出隔離於`strategy_dl_filter_gate_<param_policy>_canonical_runtime/`，包含A/B與C/F pair的完整策略比較、交易歸因及合併`strategy_dl_filter_gate.md/json`。

Binary DL risk-only parameter adaptation同樣為CLI-only暫時研究，不加入互動選單。正式比較固定為4種參數 × Binary DL關／開，共8個操作點：P0原ROOS＋原正式規則、P1原ROOS＋Rule-based filters全關、P2在rules全關／DL關環境只重訓四個風險參數、P3在rules全關／DL開環境以Binary PIT Scores只重訓同四個風險參數。每套參數各回放DL關／開，命名為A0／B0至A3／B3：

```powershell
python -m filters.breakout_quality.strategy_param_training `
  --dataset full `
  --param-policy base-finalist-best `
  --max-positions 10 `
  --rotation off
```

風險搜尋欄位固定為`atr_len`、`atr_buy_tol`、`atr_times_init`與`atr_times_trail`；`high_len`、TP、fixed risk、position cap、max positions、rotation、費稅、原position-aware buy-sort及其餘非風險值依各rolling effective date凍結。未指定`--trials-per-fold`時採目前正式training policy；原ROOS歷史trial數只作診斷，不要求與本次相等。

P3訓練必須使用`build-binary-point-in-time-scores`建立的expanding-window Binary PIT Scores。每個score period的模型只可使用該期開始日前已完成Label的歷史資料；optimizer runtime與平行fold workers都必須驗證同一PIT manifest／scores identity。禁止使用最終9A forward-OOS、`research_scores.csv`或Selection in-sample score回灌歷史訓練。Binary PIT最早合法日期不必倒推覆蓋完整120個月Selection：PIT開始日前固定pass-through，等同DL-off；PIT期間內缺少候選分數採保守REJECT；PIT尾端早於optimizer最新Selection則fail-fast。P3 preflight會輸出逐fold bootstrap／partial／full coverage與`binary_pit_optimizer_coverage.csv`，並將coverage policy納入runtime identity。預設Gate在PIT缺失時自動建立；可用`--no-build-binary-pit`只做前置檢查。PIT獨立CLI為：

```powershell
python apps/breakout_quality.py build-binary-point-in-time-scores `
  --filter-id breakout_quality_v1 `
  --experiment-profile unique_group_sampling `
  --score-start-date auto `
  --score-end-date auto
```

報表固定包含八操作點總表、`B0−A0`至`B3−A3`的DL增量、`A1−A0`、`A2−A1`、`B2−B1`、`A3−A1`、`B3−B1`、最終公平比較`B3−A2`及interaction=`(B3−A3)−(B2−A2)`。輸出位於`models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/`；Binary PIT位於`models/research/breakout_quality/binary_point_in_time_scores/<filter>/<architecture>/<profile>/`。

兩種政策都維持Score缺失契約：有效Score候選優先；缺分候選不排除、不填0，並完整回退原buy-sort。正式比較先看相較原始Score ranking能否恢復平均投入與曝險，再判斷總報酬、Return／MDD、Target mean、Realized R與capture；不得只因R2／R3優於原始Score就直接採用，仍須至少對照Baseline。

- `--param-policy` 與參數檔內 `selector` 不一致時直接拒絕；`base-finalist-best` 另要求每期 `1 member / min_agree=1`。
- 可正常評分但低 Score 的候選仍保留，只是順位靠後。Hard-filter模式的正式不可評分事件仍保守REJECT；Score-ranking模式的缺分候選不得排除或填0，必須保存`available=false`與原始Score來源，排在有效Score後並完整回退既有buy-sort。Continuation與STOP後Re-entry沿用原始breakout Score及原始Score事件日期。
- Score-ranking capture attribution已從正式strategy compare runtime拆離。`filters/breakout_quality/strategy_compare_engine.py`只產生canonical策略比較工件，不import `tools/audit/`、不在replay流程內自動產生Audit，也不提供`--capture-audit-only`。
- 通用capture實作位於`tools/audit/portfolio/score_ranking_capture.py`，只能從**已完成的pair工件**物化read-only報表；需要此歷史研究診斷的`strategy_filter_gate.py`／`strategy_adapt.py`會在正式比較完成或重用既有pair後顯式呼叫它，不因此重跑portfolio。Future Target仍只可在replay後join，不進候選排序、資金配置、成交或optimizer。
- 原策略比較主要工件維持`strategy_comparison.md`、`strategy_comparison.json`及canonical equity／trades／daily-capacity／selected diagnostics；capture audit如被研究工具要求，另輸出`score_ranking_capture_audit.md`、`score_ranking_capture_audit.json`與lifecycle／年度／scenario CSV。兩份報表彼此獨立，strategy comparison JSON不再內嵌Audit payload。
- Capture audit可分解平均實際投入、預留／投入比例、stop distance、保留買單成交率、持有期、首次半倉時間、尾倉slot-days、entry-date／月份集中度、可用時的產業集中度、exit reason、Realized R、Target R、Target capture ratio、realization gap與年度差異；若交易列沒有canonical產業欄位則顯示N/A，不自行推測類股。
- 一般Optimizer search space固定ranking=`False`，不得把ranking開關設成trial維度。策略適應使用專用固定context，而不是搜尋ranking開關。
- PIT build與model audit完成後，Selection ranking參數適應仍屬明確research CLI，不加入模型或正式策略比較選單：

```bash
python -m tools.filters.breakout_quality.strategy_adapt --dataset full --param-policy base-finalist-best
```

  上述未指定`--ranking-policy`時維持既有原始Score Adapted流程。R3參數適應為CLI-only研究，不加入互動選單；執行：

```bash
python -m tools.filters.breakout_quality.strategy_adapt --dataset full --param-policy base-finalist-best --ranking-policy capital-bucket-then-score
```

  R3流程固定`capital-bucket-then-score`於optimizer trial外，不把ranking policy、桶數或桶邊界放入搜尋；Baseline／R3 Sort Only使用舊正式ROOS，Param Only／R3 Adapted共用同一套R3 Adapted active params。工件隔離於`models/research/breakout_quality/score_ranking_adaptation/capital_bucket_then_score/rolling_validation/`，不得重用或覆蓋原始Score Adapted study。

  流程先對原Baseline全部rolling folds輸出training Score coverage，並以`BREAKOUT_QUALITY_POINT_IN_TIME_COVERAGE_REFERENCE_START_DATE`代表延伸前的正式PIT起點。全部Baseline folds與原OOS期間均保留；actual coverage必須逐fold不低於reference、至少一個fold嚴格改善且加權總coverage提高。實際PIT起點以前可依正式缺分契約回退existing buy-sort，實際PIT期間內缺口仍fail-fast，全部OOS replay必須完整位於PIT期間。接著在相同期間自動建立或重用Baseline／Sort Only，只訓練一套固定`use_breakout_quality_ranking=True`、hard filter=False的新active params，再輸出Param Only／Adapted。四組fold schedule、期間、PIT identity、risk、position cap與交易規則完全一致；Param Only與Adapted共用同一套新active params，只有ranking不同。結果只屬`ROLLING_SELECTION_DIAGNOSTIC`，不執行完整Selection final refit或正式OOS。未達100% coverage不會單獨構成拒絕，但若延伸後沒有提升、任何fold退步或PIT期間內缺分，流程會在optimizer前拒絕。
- 此研究是在已查看既有OOS後進行的迭代證據；任何候選改法仍須凍結契約後再做無前視驗證，不能由Selection結果直接部署。

### `run_best_params.json` 的用途與產生方式

Trade Mode 會先輸出 `models/candidate_best_params.json`；目前 `TRADE_MODE_AUTO_PROMOTE_RUN_BEST=True`，候選通過正式 promotion 契約時才建立或更新 `models/run_best_params.json`：

```bash
python apps/ml_optimizer.py --dataset full --model trade --trials 10
```

目前 random-seed ensemble 已啟用，因此 `run_best_params.json` 可能是 static active-param ensemble，而不是單一參數 JSON；策略對照工具支援此格式，但僅可明確標記為非 OOS 敏感度診斷：

```bash
python -m filters.breakout_quality.strategy_compare_engine --dataset full --params models/run_best_params.json --allow-static-diagnostic --max-positions 10 --rotation off
```

不指定 `--params` 時，也只有加上 `--allow-static-diagnostic` 才會使用正式 primary param source。此結果不可作為 2021～2025 無前視 OOS 部署證據。

# 輸出分類

- `outputs/local_regression/`：test suite 歷史 bundle。
- `outputs/local_regression/_staging/`：formal / validate 暫存 staging；屬 `local_regression` 內部子目錄，會由 retention 自動清理。
- `outputs/validate_consistency/`：standalone consistency 報表。
- `outputs/ml_optimizer/`：optimizer profiling / 載入摘要。
- `outputs/portfolio_sim/`：投組報表與載入摘要。
- `outputs/vip_scanner/`：scanner issue log。
- `outputs/smart_downloader/`：下載器 issue log。
- `outputs/debug_trade_log/`：`trade_analysis` 單股分析輸出；為維持既有工具鏈相容，暫沿用 legacy 目錄名 `debug_trade_log`。
- `outputs/debug_trade_log/`（trade_analysis legacy output dir）屬既有工具鏈相容邊界。
- `outputs/workbench_ui/`：Workbench GUI runtime 快取；目前用於常用股票中文名稱快取；若 reduced 代碼組變動或缺名，Workbench 會優先查官方 CSV / ISIN 名錄並於必要時做 SSL 容錯與 HTTP fallback。

## 其他文件

- `ARCHITECTURE.md`：分層、正式入口、依賴方向與共享邊界。
- `TEST_SUITE_CHECKLIST.md`：formal test suite 主表、狀態與收斂索引。

### 11F No-time Target Arrays＋Selection-only Learnability Audit

11E固定消融通過後，建立獨立No-time Target version arrays，並只稽核Selection內分布與同日可排序性。CLI-only，不加入互動選單：

```bash
python apps/breakout_quality.py audit-no-time-target --filter-id breakout_quality_v1
```

Target ID：

```text
strategy_aligned_opportunity_no_time_r_v1
```

輸出位於：

```text
outputs/filters/breakout_quality/breakout_quality_v1/continuous_targets/strategy_aligned_opportunity_no_time_r_v1/
```

本命令strict讀取11A component arrays及11E report／CSV SHA256，固定推導`target_raw_r=favorable_r-adverse_r`；只輸出Inner Train、Validation與Selection指標，明確`oos_evaluated=false`。不建立模型、profile、checkpoint、threshold或runtime score。

### 11G PASS-conditional No-time Magnitude Ranker

11F Selection-only稽核通過後，11G使用既有continuous-ranker CLI與新的research profile；不加入互動選單：

```bash
python apps/breakout_quality.py train-continuous-ranker \
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

### 11H PASS-only Realization-gap Attribution Audit

11G已能排序PASS-only No-time Target，但actual PASS trades的Score↔R為負，因此11H只做凍結工件歸因；CLI-only，不加入互動選單：

```bash
python apps/breakout_quality.py audit-pass-realization-gap --filter-id breakout_quality_v1
```

固定輸入：

- 11G `continuous_ranker_report.json`與`continuous_ranker_scores.csv`。
- 11F No-time Target component arrays。
- 11A canonical `continuous_target_trade_matches.csv`。

Audit聚焦原始Label=PASS，固定計算`realization_gap_r=target_raw_r-r_multiple`與`favorable_capture_ratio=r_multiple/favorable_r`，並輸出Score↔Favorable／Adverse、Score↔gap／capture、控制Target後partial Score↔R，以及Score／Target top-bottom decile成分。來源SHA256、逐筆`target=favorable-adverse`與actual PASS配對數均須一致。

輸出位於：

```text
outputs/filters/breakout_quality/breakout_quality_v1/inception_time_v1/strategy_aligned_no_time_pass_magnitude_mse/pass_realization_gap_audit/
```

本命令不訓練、不建立新profile／checkpoint、不調Target、loss、epoch、sampling、threshold或runtime score。


### 11I Nested Selection Strategy-realization Coverage Audit

11H確認11G高Score對應更大的未實現機會落差；11I先建立Selection內nested rolling OOS參數鏈，再以canonical no-filter portfolio replay量化strategy-realization coverage。CLI-only，不加入互動選單。

先輸出準備腳本：

```powershell
python apps/breakout_quality.py audit-selection-strategy-realization --filter-id breakout_quality_v1 --prepare-only
```

未指定`--optimizer-trials`時，trial數直接讀取`config/training_policy.py`的`OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT`；不再維護11I專屬硬編碼預設。需要單次覆蓋時可明確加上`--optimizer-trials 100`。每次修改config或CLI值後都必須重新執行`--prepare-only`，因為既有`.ps1`是已生成的靜態腳本。

執行產生的`prepare_selection_nested_roos.ps1`後，再執行：

```powershell
python apps/breakout_quality.py audit-selection-strategy-realization --filter-id breakout_quality_v1 --quiet
```

預設以2014-01-01～2020-12-31、120個月固定訓練窗、12個月OOS建立research-only nested參數鏈；策略replay固定只跑2014-01-01～2020-11-05，並由11F manifest的`final_refit_date_range`再次驗證，避免年底事件Target跨入2021。`V16_MODELS_DIR`隔離到`models/research/breakout_quality/selection_strategy_realization`，不得覆蓋正式2021～2026 rolling params。11I不訓練、不建立Target arrays，且不得把未交易候選標成0R。

### 11J Canonical Per-candidate Counterfactual Execution Audit（已停止）

11I確認Selection nested-OOS No-time Target方向成立、但actual portfolio trade coverage不足後，使用同一nested params與canonical candidate replay，對每個qualified訊號建立獨立counterfactual execution：

```bash
python apps/breakout_quality.py audit-candidate-counterfactual --filter-id breakout_quality_v1 --quiet
```

11J採plain replay-counts＋execution sidecar：只以11I相同的2014-01-01～2020-11-05執行一次canonical replay，`replay_counts`固定使用普通dict，再用相同flatten／target-date／unique流程精確核對2,003筆。Orderable成交資料由獨立`replay_execution_rows` sidecar保存；不得把observer、自訂dict或counterfactual狀態機傳入canonical replay。Sidecar不遞迴複製`signal_state`或`params_obj`，只保留成交必要欄位、params reference、cloned shadow與單一ticker market array參照。通過2,003 guard後才離線執行counterfactual；每日只推進open positions，2020-11-06～2020-12-31由sidecar market calendars延伸，不重跑portfolio。

此audit忽略portfolio capacity與cash competition，但保留正式限價成交、locked-limit、shadow inheritance、半倉停利、停損、指標出場、賣出受阻、費稅與R口徑。輸出位於：

```text
outputs/filters/breakout_quality/breakout_quality_v1/continuous_targets/strategy_aligned_opportunity_no_time_r_v1/selection_strategy_realization_audit/candidate_counterfactual_execution_audit/
```

未成交訊號保持`r_multiple`空值，不填0R。11J不建立Target arrays、不訓練、不加入互動選單。

11J已於2026-08-01停止：plain dict＋execution sidecar仍只重現1,969／2,003，不再要求執行，也不得再修改core以追求重現。

### 11K Portfolio Selection-pressure Attribution Audit

直接讀取11I既有orderable candidates與actual trade matches，不重播市場：

```bash
python apps/breakout_quality.py audit-selection-pressure --filter-id breakout_quality_v1
```

輸出位於：

```text
outputs/filters/breakout_quality/breakout_quality_v1/continuous_targets/strategy_aligned_opportunity_no_time_r_v1/selection_strategy_realization_audit/portfolio_selection_pressure_audit/
```

固定輸出同日Target percentile、actual top-half／top-quartile比例、依每日實際買入數k的Target top-k retention、Target opportunity gap與候選壓力分桶。未選候選沒有realized R，維持缺值，不填0R、不推估反事實績效。11K為read-only、CLI-only，不訓練、不修改runtime。

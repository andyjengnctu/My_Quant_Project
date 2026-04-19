# Optimizer Rolling 與 Anti-Overfit 規格

> 本文件整理目前已定版之共識，作為後續逐步升級 optimizer 的依據。
> 原則：
> - 選參只看 train phase
> - anti-overfit 主要做在 train phase
> - 外部 rolling OOS 只做驗證與歷史績效展示
> - 專案仍在開發中，無法宣稱 OOS 完全不影響後續方法演進；但應盡量避免 OOS 直接回授到單輪最佳化過程

---

## 1. 方法總原則

### 1.1 Train / OOS 分工
- Train phase：
  - 用於 trial 搜尋、評分、選參
  - anti-overfit 機制也放在 train 內
- 外部 rolling OOS：
  - 只做該 period 的 OOS 驗證
  - 不直接參與該 period 的 trial ranking / best param 決策

### 1.2 專案現況下的務實口徑
- 本專案仍在開發中，且台股歷史資料有限。
- 因此不宣稱 OOS 完全不影響未來版本演進。
- 但必須盡量做到：
  - OOS 不直接進 objective
  - OOS 不直接排名 trial
  - OOS 不直接決定單輪 best param

---

## 2. 內層 anti-overfit：定版範圍

### 2.1 內層不做完整 rolling
內層先不做 train-dev 內部 rolling / internal walk-forward，理由：
- 計算成本高
- 每段 train 會變短
- 台股資料有限
- 容易為了防 overfit 反而把樣本切太碎

### 2.2 內層目前定版內容
內層先定版為：
- 既有 hard filters 繼續沿用
- 新增 `local_min_score` 作為局部穩健性檢查
- 新增不同時段 RoMD 檢查，使用同一次完整 train 回測後的年度切片結果
- 其他候選機制（收益集中度、複雜度、fee stress 等）先不納入第一版主決策流程

---

## 3. `local_min_score` 定義（已定版）

### 3.1 定義目的
`local_min_score` 用來回答：

> 最佳參數附近，只要動 1 個既有 search step，最差會掉到什麼程度。

它不是拿來做多層 break tie，而是作為局部 fragility / 尖峰過擬合檢查。

### 3.2 定義
對某個 finalist trial `p`，固定：
- 同一份 prep 結果
- 同一個 objective_mode
- 同一個 train 日期區間
- 同一套 `apply_filter_rules(...)`
- 同一個 `calc_portfolio_score(...)`

只對 `p` 的指定數值參數做「單一參數 ±1 個既有 search step」擾動，並重算 train objective score。

記所有合法鄰點集合為 `N(p)`，則：

```text
local_min_score(p) = min_{q in N(p)} S(q)
```

其中 `S(q)` 為現行 train objective score；若合法鄰點 `q` 在 objective 流程中 fail 現有 hard filters，則該點分數直接視為現行 `INVALID_TRIAL_VALUE`。

### 3.3 鄰域規則
- 僅做 **單一參數** 的 ±1 step 擾動
- step 完全沿用既有 search space 設定，不引入新 epsilon / 新 magic number
- 若某方向超出原本搜尋範圍，該方向跳過
- `local_min_score` 不包含中心點 `p` 自己

### 3.4 第一版永遠納入的核心參數
- `high_len`
- `atr_len`
- `atr_times_init`
- `atr_times_trail`
- `atr_buy_tol`
- `tp_percent`（僅在 `tp_percent` 確實為 optimizer 搜尋維度時納入）
- `use_bb=True`：納入 `bb_len`、`bb_mult`
- `use_kc=True`：納入 `kc_len`、`kc_mult`
- `use_vol=True`：納入 `vol_short_len`、`vol_long_len`

但第一版仍：
- **不翻動** `use_bb / use_kc / use_vol` 布林值本身
- 只檢查「已啟用結構下的數值參數局部穩健性」

### 3.5 `vol` 合法鄰點規則
`vol_short_len / vol_long_len` 必須遵守正式契約：

```text
vol_long_len >= vol_short_len
```

處理規則：
- 只保留符合原 search range 且滿足正式契約的合法鄰點
- 不做 clamp
- 不偷偷聯動改另一個參數
- 不合法點直接跳過，不納入 `N(p)`
- 合法鄰點若在 objective 流程中 fail hard filters，才記為 `INVALID_TRIAL_VALUE`

### 3.6 第一版角色
`local_min_score` 第一版的角色是 **gate / veto**，不是多層 break tie。

建議主規則：
- 先通過既有 hard filters
- 再通過 `local_min_score > 0`
- 通過後仍由 `base_score` 選最高

此處 `> 0` 的物理意義：
- 最佳點附近只動 1 格，最差鄰點仍不應落到負 edge / 無效 edge

---

## 4. 不同時段 RoMD 檢查（已定版方向）

### 4.1 目的
雖然內層不做 rolling，但仍需檢查 train 內部是否存在明顯時段依賴。

避免出現：
- 整段 train RoMD 看起來不錯
- 但其實只靠少數時段支撐
- 某些年份已明顯失效

### 4.2 作法
- 不重新訓練多次
- 不做內層 rolling
- 只對 **同一次完整 train 回測結果** 再做時間切片
- 第一版按 **完整年度** 切分

例如 train 為 `2015~2020`，則切：
- 2015
- 2016
- 2017
- 2018
- 2019
- 2020

每段計算：
- segment return
- segment MDD
- segment RoMD

### 4.3 第一版角色
第一版不把年度 RoMD 納入複雜加權 objective。

建議角色：
- 做時間穩健性 gate / 診斷欄位
- 不作為主排序分數

### 4.4 第一版最小 gate
第一版建議至少產出：
- `min_full_year_romd`
- `positive_full_year_romd_count`
- `negative_full_year_romd_count`

最小 gate 建議：

```text
positive_full_year_romd_count > negative_full_year_romd_count
```

物理意義：
- 多數完整年度仍為正 edge
- 不要求所有年份都必須為正，避免過度嚴苛

### 4.5 主決策口徑
第一版內層主決策可定為：
1. 既有 hard filters 通過
2. `local_min_score > 0`
3. `positive_full_year_romd_count > negative_full_year_romd_count`
4. 通過者以 `base_score` 最大者勝出

---

## 5. 外層 rolling OOS：定版方向

### 5.1 外層 rolling 的目標
外層 rolling 不是單一 holdout 報表，而是：

> 每個 period 獨立重訓、獨立選 winner、獨立驗 OOS，最後將各 period 的 OOS 串接成 pseudo-live 歷史績效。

### 5.2 外層 rolling 規格
每個 rolling period：
- `train = 固定長度視窗`
- `oos = 下一個完整年度`
- `step = 1 年`
- 只用完整年度

範例：
- `2015~2020 -> 2021`
- `2016~2021 -> 2022`
- `2017~2022 -> 2023`
- `2018~2023 -> 2024`
- `2019~2024 -> 2025`

### 5.3 每個 period 必須獨立
每個 period 必須各自擁有：
- 獨立 train dates
- 獨立 optimizer study
- 獨立 winner
- 獨立 OOS report

不可把不同 period 的 trial 混在同一個 study / 同一個 best resolver 直接比較。

### 5.4 period-aware 隔離原則
至少下列實體都必須 period-aware：
- study name
- DB path
- run_best params artifact
- OOS report artifact

避免不同 period 之 trial / winner / output 混寫。

---

## 6. 外層 rolling 的 period 預算一致性（已定版）

### 6.1 必須固定一致的項目
為確保不同 period 可比較，每個 period 至少必須固定：
- train window 長度
- OOS window 長度
- step 長度
- search space
- sampler / pruner 設定
- attempted `n_trials`
- finalist 數量
- 內層 anti-overfit 規則

### 6.2 `n_trials` 口徑
每個 period 應固定相同的 **attempted `n_trials`**。

不建議：
- 以 timeout 作為主控制
- 以「補到有效 trial 數一致」為目標

理由：
- timeout 會讓實際完成 trial 數因 period 而異
- 補到有效 trial 一致，等於給某些 period 更多搜尋機會

### 6.3 Seed 規則
sampler seed 規則應一致，建議可採：

```text
period_seed = base_seed + period_index
```

目的：
- 每個 period 可重現
- 規則一致
- 不需所有 period 完全同 seed

---

## 7. 外層 rolling 的歷史績效口徑

### 7.1 正式 stitched OOS
正式 stitched OOS 代表：

> 若歷史上每年重訓一次，並使用下一個 OOS 年作為實際運行期，所形成的 pseudo-live 歷史資產曲線。

### 7.2 串接方式
建議用 **連續資金口徑**：
- 2021 OOS 用初始資金起跑
- 2022 OOS 初始資金 = 2021 OOS 期末資金
- 2023 以此類推

此口徑最接近真實每年重訓一次後的連續運行結果。

---

## 8. 正式歷史驗證 vs 最新未完整觀察（版本 2，已定版）

### 8.1 正式版
正式歷史驗證只統計到最後一個 **完整 OOS 年**。

例如：
- 正式 stitched OOS：`2021~2025`

這一段作為：
- 主報表
- 主 KPI
- 主比較基準

### 8.2 延伸觀察版
若最新年度尚未完整，例如 `2026-01 ~ 2026-04`，則：
- 可另外列為 `2026 YTD partial OOS`
- 或 `partial OOS / live monitoring`
- 但 **不併入正式 stitched OOS 主指標**

### 8.3 版本 2 定版口徑
採用版本 2：
- 主報表：`Official stitched OOS (2021~2025)`
- 附錄 / 觀察：`2026 YTD partial OOS`

### 8.4 正式主指標不得混入 partial 年度
不得把未完整年度直接混入：
- 正式 stitched CAGR
- 正式 stitched MDD
- 正式 stitched RoMD
- 正式主比較結果

partial 年度應獨立標示。

---

## 9. 實際上場參數的口徑

### 9.1 歷史 stitched OOS 期間
歷史 stitched OOS 期間不是只用最後一組參數，而是：
- 每個 OOS 年，使用前一個 train period 訓出的 winner

例如：
- 2021 用 `2015~2020` 訓出的參數
- 2022 用 `2016~2021` 訓出的參數
- ...
- 2025 用 `2019~2024` 訓出的參數

### 9.2 真正當下上場
若現在要進入新年度，例如 2026：
- 用最新完整 train window 訓出最新 deployment params
- 該組參數才是當下實際上場用的參數

例如：
- `2020~2025 train -> deployment params for 2026`

### 9.3 與 partial OOS 的關係
- 2026 實際上場使用 `2020~2025` 訓出的參數
- `2026-01 ~ 最新日` 記為 `partial OOS / live monitoring`
- 不混入正式 stitched OOS 主報表

---

## 10. 第一版建議輸出物

### 10.1 period-level
每個 rolling period 至少輸出：
- train window
- OOS window
- winner params
- base_score
- local_min_score
- 年度 RoMD 檢查結果
- OOS return / MDD / RoMD
- trade_count
- winner trial number

### 10.2 official stitched report
- official stitched OOS period range
- total return
- CAGR
- MDD
- RoMD
- trade_count
- yearly summaries

### 10.3 partial / live monitoring
- latest partial period range
- YTD return
- YTD MDD
- YTD RoMD
- current deployed params version

---

## 11. 第一版實作優先順序

### 11.1 內層
1. 補 `local_min_score`
2. 補 train 內完整年度 RoMD 切片與 gate
3. 保持主排序仍由 `base_score` 決定

### 11.2 外層
1. 建 `build_rolling_periods()`
2. 將 study / db / artifact 改為 period-aware
3. 逐 period 跑獨立 optimizer
4. 每 period 產生獨立 OOS report
5. 串接 official stitched OOS
6. 額外輸出 partial OOS / live monitoring

---

## 12. 本文件的角色
本文件作為後續 optimizer 升級的定版規格依據。
後續若需修改：
- 先更新本文件
- 再依本文件逐步實作
- 避免實作與共識脫鉤

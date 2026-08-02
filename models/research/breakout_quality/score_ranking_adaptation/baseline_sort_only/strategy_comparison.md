# Breakout Quality Score 排序策略經濟效果對照

- 期間：`2014-01-01` ～ `2020-12-31`
- 參數檔：`C:\Users\User\Desktop\My_Quant_Project\models\research\breakout_quality\selection_strategy_realization\roos_base_best.json`
- 參數型態：`rolling_active_param_ensemble`
- 參數 selector：`base_finalist_best`
- Runtime members：`1`～`1`；min_agree=`1`
- 比較設計：`selection_point_in_time_active_param_replay`
- 歷史 active-param 無前視：`True`
- Dataset：`full`
- Score source：`selection_point_in_time`
- Benchmark：`0050`
- 唯一差異：`use_breakout_quality_ranking=False vs True（hard filter 兩組皆 False）`
- Ranking model：`breakout_quality_v1` / `inception_time_v1` / `strategy_aligned_no_time_pass_magnitude_mse`；threshold 不作 gate
- 排序鍵：`breakout_quality_score_desc → existing_buy_sort → ticker_deterministic`
- Ranking 範圍：`all_candidates_after_single_member_qualification`

## 主要結果

| 指標 | No filter | Quality score ranking | 差異 | 判讀 |
|---|---:|---:|---:|:---:|
| 淨總報酬 | 182.62% | 144.80% | -37.82% | 🔴 惡化 |
| 最大回撤 | 13.18% | 21.53% | +8.35% | 🔴 惡化 |
| 報酬／最大回撤 | 13.86 | 6.73 | -7.13 | 🔴 惡化 |
| 年化報酬 | 16.00% | 13.65% | -2.36% | 🔴 惡化 |
| Log R² | 0.9349 | 0.8599 | -0.0750 | 🔴 惡化 |
| 月勝率 | 61.90% | 63.10% | +1.19% | 🟢 改善 |
| 交易數 | 527 | 502 | -25 | ⚪ 中性 |
| 勝率 | 38.14% | 35.26% | -2.88% | 🔴 惡化 |
| Payoff | 2.92 | 3.07 | +0.15 | 🟢 改善 |
| EV | 0.28 R | 0.35 R | +0.07 R | 🟢 改善 |
| 平均曝險 | 77.33% | 54.20% | -23.13% | 🟡 注意 |
| 最差完整年度 | -0.35% | -10.53% | -10.18% | 🔴 惡化 |
| 平均每日可掛單候選 | 19.78 | 19.66 | -0.12 | ⚪ 中性 |
| 候選供給不足日 | 329 日 | 338 日 | +9 日 | 🔴 惡化 |
| 期末未滿倉日 | 755 日 | 630 日 | -125 日 | 🟢 改善 |
| 期末持股缺口總和 | 2336 格日 | 2121 格日 | -215 格日 | 🟢 改善 |

## 年度報酬

| 年度 | No filter | Quality score ranking | 差異 | 判讀 | 完整年度 |
|---:|---:|---:|---:|:---:|:---:|
| 2014 | 28.92% | 33.87% | +4.95% | 🟢 改善 | 是 |
| 2015 | 23.34% | 13.91% | -9.43% | 🔴 惡化 | 是 |
| 2016 | -0.35% | 4.97% | +5.32% | 🟢 改善 | 是 |
| 2017 | 36.77% | 39.72% | +2.95% | 🟢 改善 | 是 |
| 2018 | 1.55% | -10.53% | -12.08% | 🔴 惡化 | 是 |
| 2019 | 18.62% | 14.07% | -4.56% | 🔴 惡化 | 是 |
| 2020 | 8.26% | 7.25% | -1.01% | 🔴 惡化 | 是 |

## Selection 選股診斷（Future Target僅於回放後join）

| 指標 | Baseline | Score Sort | 差異 | 判讀 |
|---|---:|---:|---:|:---:|
| Orderable Score coverage | 0.9808 | 0.9942 | +0.0134 | 🟢 改善 |
| 選中候選 Target percentile | 0.6964 | 0.7170 | +0.0206 | 🟢 改善 |
| Target top-k retention | 0.3679 | 0.4694 | +0.1015 | 🟢 改善 |
| Target opportunity gap (R) | 1.9209 | 1.3577 | -0.5632 | 🟢 改善 |
| 選中候選 Target mean (R) | 1.0868 | 1.2079 | +0.1211 | 🟢 改善 |

> Future Target未進入候選排序、資金配置或成交決策；上述診斷只在兩組策略回放完成後離線計算。

> 本報表使用Selection point-in-time Scores與當期歷史active params比較排序機制；可用於Selection內決定是否進入參數適應，但正式效果仍須由凍結後OOS驗證。

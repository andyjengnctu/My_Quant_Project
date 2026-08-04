PS C:\Users\User\Desktop\My_Quant_Project> python apps/breakout_quality.py strategy-dl-filter-param-adapt-gate `
>>   --dataset full `
>>   --param-policy base-finalist-best `
>>   --max-positions 10 `
>>   --rotation off
====================================================================================================
 Binary DL Filter Point-in-Time Scores
====================================================================================================
期間      ：2006-12-01 ～ 2026-03-02
Folds     ：21
Validation：24 months
Profile   ：unique_group_sampling

Fold plan
---------
Fold                    Score period            Train  Validation  Refit  Score
----------------------  ----------------------  -----  ----------  -----  -----
fold_20061201_20061231  2006-12-01～2006-12-31  202    3095        3454   321  
fold_20070101_20071231  2007-01-01～2007-12-31  307    3271        3739   1822 
fold_20080101_20081231  2008-01-01～2008-12-31  1635   4033        6161   889  
fold_20090101_20091231  2009-01-01～2009-12-31  3739   2674        7028   4198 
fold_20100101_20101231  2010-01-01～2010-12-31  6161   4620        10796  1878 
fold_20110101_20111231  2011-01-01～2011-12-31  7028   5733        12798  948  
fold_20120101_20121231  2012-01-01～2012-12-31  10796  2767        14030  2250 
fold_20130101_20131231  2013-01-01～2013-12-31  12798  2984        16125  2487 
fold_20140101_20141231  2014-01-01～2014-12-31  14030  4446        18535  2077 
fold_20150101_20151231  2015-01-01～2015-12-31  16125  4354        20693  1803 
fold_20160101_20161231  2016-01-01～2016-12-31  18535  3651        22477  2521 
fold_20170101_20171231  2017-01-01～2017-12-31  20693  4009        24912  3068 
fold_20180101_20181231  2018-01-01～2018-12-31  22477  5234        27940  1853 
fold_20190101_20191231  2019-01-01～2019-12-31  24912  4746        29973  3507 
fold_20200101_20201231  2020-01-01～2020-12-31  27940  4870        33165  3418 
fold_20210101_20211231  2021-01-01～2021-12-31  29973  6065        36213  3145 
fold_20220101_20221231  2022-01-01～2022-12-31  33165  5980        39635  2016 
fold_20230101_20231231  2023-01-01～2023-12-31  36213  4734        41807  4585 
fold_20240101_20241231  2024-01-01～2024-12-31  39635  5695        45913  3476 
fold_20250101_20251231  2025-01-01～2025-12-31  41807  7705        49939  4239 
fold_20260101_20260302  2026-01-01～2026-03-02  45913  7180        53999  975  

[1/21] fold_20061201_20061231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.692622 | Val Loss 0.688538 | 耗時 00:00.6 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.692674 | Val Loss 0.685922 | 耗時 00:00.1 | ★ 新最佳
  Epoch  3/200 | Train Loss 0.693032 | Val Loss 0.682265 | 耗時 00:00.1 | ★ 新最佳
  Epoch  4/200 | Train Loss 0.694179 | Val Loss 0.677551 | 耗時 00:00.1 | ★ 新最佳
  Epoch  5/200 | Train Loss 0.697902 | Val Loss 0.672194 | 耗時 00:00.1 | ★ 新最佳
  Epoch  6/200 | Train Loss 0.707229 | Val Loss 0.667399 | 耗時 00:00.1 | ★ 新最佳
  Epoch  7/200 | Train Loss 0.729525 | Val Loss 0.666873 | 耗時 00:00.1 | ★ 新最佳
  Epoch  8/200 | Train Loss 0.785589 | Val Loss 0.684054 | 耗時 00:00.1
  結果：Early stopping 於 Epoch 8 | Best Epoch 7 | 最低 Val Loss 0.666873

完整 Selection 重訓（7 Epoch）
  Epoch  1/7 | Train Loss 0.674086 | 耗時 00:00.4
  Epoch  2/7 | Train Loss 0.614432 | 耗時 00:00.3
  Epoch  3/7 | Train Loss 0.605499 | 耗時 00:00.4
  Epoch  4/7 | Train Loss 0.743079 | 耗時 00:00.4
  Epoch  5/7 | Train Loss 0.579002 | 耗時 00:00.3
  Epoch  6/7 | Train Loss 0.529250 | 耗時 00:00.3
  Epoch  7/7 | Train Loss 0.520844 | 耗時 00:00.3

[2/21] fold_20070101_20071231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.697878 | Val Loss 0.685276 | 耗時 00:00.1 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.698788 | Val Loss 0.683478 | 耗時 00:00.1 | ★ 新最佳
  Epoch  3/200 | Train Loss 0.699297 | Val Loss 0.681221 | 耗時 00:00.1 | ★ 新最佳
  Epoch  4/200 | Train Loss 0.700694 | Val Loss 0.678112 | 耗時 00:00.1 | ★ 新最佳
  Epoch  5/200 | Train Loss 0.711424 | Val Loss 0.666772 | 耗時 00:00.1 | ★ 新最佳
  Epoch  6/200 | Train Loss 0.738482 | Val Loss 0.653391 | 耗時 00:00.1 | ★ 新最佳
  Epoch  7/200 | Train Loss 0.788974 | Val Loss 0.653758 | 耗時 00:00.1
  結果：Early stopping 於 Epoch 7 | Best Epoch 6 | 最低 Val Loss 0.653391

完整 Selection 重訓（6 Epoch）
  Epoch  1/6 | Train Loss 0.682061 | 耗時 00:00.4
  Epoch  2/6 | Train Loss 0.601111 | 耗時 00:00.4
  Epoch  3/6 | Train Loss 0.669048 | 耗時 00:00.4
  Epoch  4/6 | Train Loss 0.663844 | 耗時 00:00.4
  Epoch  5/6 | Train Loss 0.886611 | 耗時 00:00.4
  Epoch  6/6 | Train Loss 0.561803 | 耗時 00:00.4

[3/21] fold_20080101_20081231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.691687 | Val Loss 0.661462 | 耗時 00:00.3 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.714709 | Val Loss 0.629247 | 耗時 00:00.2 | ★ 新最佳
  Epoch  3/200 | Train Loss 0.689026 | Val Loss 0.637691 | 耗時 00:00.2
  結果：Early stopping 於 Epoch 3 | Best Epoch 2 | 最低 Val Loss 0.629247

完整 Selection 重訓（2 Epoch）
  Epoch  1/2 | Train Loss 0.621168 | 耗時 00:00.7
  Epoch  2/2 | Train Loss 0.644477 | 耗時 00:00.6

[4/21] fold_20090101_20091231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.682061 | Val Loss 0.693506 | 耗時 00:00.5 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.601111 | Val Loss 0.756781 | 耗時 00:00.4
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.693506

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.644381 | 耗時 00:00.7

[5/21] fold_20100101_20101231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.621168 | Val Loss 0.860940 | 耗時 00:00.8 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.644477 | Val Loss 0.900575 | 耗時 00:00.7
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.860940

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.614426 | 耗時 00:01.1

[6/21] fold_20110101_20111231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.644381 | Val Loss 0.668038 | 耗時 00:00.9 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.634832 | Val Loss 0.856028 | 耗時 00:00.8
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.668038

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.620769 | 耗時 00:01.3

[7/21] fold_20120101_20121231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.614426 | Val Loss 0.691471 | 耗時 00:01.2 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.613409 | Val Loss 0.759366 | 耗時 00:01.1
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.691471

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.639317 | 耗時 00:01.4

[8/21] fold_20130101_20131231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.620769 | Val Loss 0.714957 | 耗時 00:01.4 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.620336 | Val Loss 0.756630 | 耗時 00:01.4
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.714957

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.649480 | 耗時 00:01.6

[9/21] fold_20140101_20141231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.639317 | Val Loss 0.715716 | 耗時 00:01.6 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.623868 | Val Loss 0.725913 | 耗時 00:01.5
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.715716

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.646768 | 耗時 00:01.9

[10/21] fold_20150101_20151231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.649480 | Val Loss 0.680135 | 耗時 00:01.7 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.625986 | Val Loss 0.694790 | 耗時 00:01.7
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.680135

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.646486 | 耗時 00:02.1

[11/21] fold_20160101_20161231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.646768 | Val Loss 0.695555 | 耗時 00:01.9 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.633987 | Val Loss 0.695030 | 耗時 00:01.9 | ★ 新最佳
  Epoch  3/200 | Train Loss 0.627154 | Val Loss 0.703006 | 耗時 00:02.0
  結果：Early stopping 於 Epoch 3 | Best Epoch 2 | 最低 Val Loss 0.695030

完整 Selection 重訓（2 Epoch）
  Epoch  1/2 | Train Loss 0.653591 | 耗時 00:02.3
  Epoch  2/2 | Train Loss 0.678400 | 耗時 00:02.3

[12/21] fold_20170101_20171231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.646486 | Val Loss 0.694305 | 耗時 00:02.2 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.638256 | Val Loss 0.694572 | 耗時 00:02.2
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.694305

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.656043 | 耗時 00:02.5

[13/21] fold_20180101_20181231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.653591 | Val Loss 0.687743 | 耗時 00:02.4 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.678400 | Val Loss 0.693597 | 耗時 00:02.5
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.687743

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.666523 | 耗時 00:02.9

[14/21] fold_20190101_20191231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.656043 | Val Loss 0.680978 | 耗時 00:02.7 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.649718 | Val Loss 0.700546 | 耗時 00:02.7
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.680978

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.653838 | 耗時 00:03.1

[15/21] fold_20200101_20201231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.666523 | Val Loss 0.706078 | 耗時 00:03.0 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.646120 | Val Loss 0.726636 | 耗時 00:03.0
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.706078

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.664367 | 耗時 00:03.3

[16/21] fold_20210101_20211231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.653838 | Val Loss 0.695090 | 耗時 00:03.2 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.667197 | Val Loss 0.728509 | 耗時 00:03.2
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.695090

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.657349 | 耗時 00:03.6

[17/21] fold_20220101_20221231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.664367 | Val Loss 0.684052 | 耗時 00:03.6 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.660415 | Val Loss 0.690512 | 耗時 00:03.5
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.684052

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.657184 | 耗時 00:04.0

[18/21] fold_20230101_20231231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.657349 | Val Loss 0.723056 | 耗時 00:03.8 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.644806 | Val Loss 0.722745 | 耗時 00:03.8 | ★ 新最佳
  Epoch  3/200 | Train Loss 0.646881 | Val Loss 0.726748 | 耗時 00:03.8
  結果：Early stopping 於 Epoch 3 | Best Epoch 2 | 最低 Val Loss 0.722745

完整 Selection 重訓（2 Epoch）
  Epoch  1/2 | Train Loss 0.658641 | 耗時 00:04.2
  Epoch  2/2 | Train Loss 0.649018 | 耗時 00:04.2

[19/21] fold_20240101_20241231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.657184 | Val Loss 0.699462 | 耗時 00:04.2 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.647235 | Val Loss 0.702862 | 耗時 00:04.2
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.699462

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.662379 | 耗時 00:04.6

[20/21] fold_20250101_20251231｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.658641 | Val Loss 0.671148 | 耗時 00:04.4 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.649018 | Val Loss 0.686304 | 耗時 00:04.3
  結果：Early stopping 於 Epoch 2 | Best Epoch 1 | 最低 Val Loss 0.671148

完整 Selection 重訓（1 Epoch）
  Epoch  1/1 | Train Loss 0.659268 | 耗時 00:05.1

[21/21] fold_20260101_20260302｜train

Epoch 選擇（依 Validation Loss）
  Epoch  1/200 | Train Loss 0.662379 | Val Loss 0.683774 | 耗時 00:04.8 | ★ 新最佳
  Epoch  2/200 | Train Loss 0.651569 | Val Loss 0.669354 | 耗時 00:04.8 | ★ 新最佳
  Epoch  3/200 | Train Loss 0.643611 | Val Loss 0.756966 | 耗時 00:04.7
  結果：Early stopping 於 Epoch 3 | Best Epoch 2 | 最低 Val Loss 0.669354

完整 Selection 重訓（2 Epoch）
  Epoch  1/2 | Train Loss 0.656341 | 耗時 00:05.5
  Epoch  2/2 | Train Loss 0.649611 | 耗時 00:05.4

工件輸出
--------
Binary PIT Scores  ：models/research/breakout_quality/binary_point_in_time_scores/breakout_quality_v1/inception_time_v1/unique_group_sampling/scores.csv
Binary PIT Manifest：models/research/breakout_quality/binary_point_in_time_scores/breakout_quality_v1/inception_time_v1/unique_group_sampling/manifest.json
====================================================================================================
OUTER ROLLING OOS TEST | NEXT 12 MONTHS
====================================================================================================
window mode      : fixed
train window     : 120 months
oos feedback     : False
promotion        : disabled
oos horizon      : next 12 months
optimizer trials : 200 per fold
----------------------------------------------------------------------------------------------------
fold   | selection period | OOS test period
----------------------------------------------------------------------------------------------------
1/6    | 2011-01~20-12   | 2021-01~21-12  
2/6    | 2012-01~21-12   | 2022-01~22-12  
3/6    | 2013-01~22-12   | 2023-01~23-12  
4/6    | 2014-01~23-12   | 2024-01~24-12  
5/6    | 2015-01~24-12   | 2025-01~25-12  
6/6    | 2016-01~25-12   | 2026-01~26-12  
----------------------------------------------------------------------------------------------------
LOCAL_MIN_SCORE              : False
INNER_VALIDATE_RANK          : False
DOMINANT_YEAR_DEPENDENCY     : False
====================================================================================================
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:48:45
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:48:53
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:49:04
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:49:14
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:49:28
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:49:39
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:49:43
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:49:54
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:50:03
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:50:11
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:50:22
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:50:28
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:50:39
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:50:42
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:50:55
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:50:59
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:51:02
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:51:05
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:51:17
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:51:29
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:51:36
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:51:47
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:51:59
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:52:11
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:52:23
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:52:35
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:52:49
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:52:52
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:53:04
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:53:08
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:53:21
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:53:35
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:53:46
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:53:58
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:54:10
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:54:14
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:54:17
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:54:20
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:54:31
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:54:45
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:54:57
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:55:09
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:55:13
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:55:23
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=5/6 | total_time=00:55:28
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
⏱️ Rolling fold parallel | completed=6/6 | total_time=00:56:28
  [1/6] train=11-01-01~20-12-31 | OOS 21-01-01~21-12-31 | DONE | best_base=306.788 | best_local_min=306.788 | elapsed=46:20
  [2/6] train=12-01-01~21-12-31 | OOS 22-01-01~22-12-31 | DONE | best_base=713.672 | best_local_min=713.672 | elapsed=43:40
  [3/6] train=13-01-01~22-12-31 | OOS 23-01-01~23-12-31 | DONE | best_base=353.380 | best_local_min=353.380 | elapsed=48:44
  [4/6] train=14-01-01~23-12-31 | OOS 24-01-01~24-12-31 | DONE | best_base=1693.390 | best_local_min=1693.390 | elapsed=56:26
  [5/6] train=15-01-01~24-12-31 | OOS 25-01-01~25-12-31 | DONE | best_base=552.711 | best_local_min=552.711 | elapsed=44:15
  [6/6] train=16-01-01~25-12-31 | OOS 26-01-01~26-12-31 | DONE | best_base=273.667 | best_local_min=273.667 | elapsed=46:54

FINALIST BEST RESULTS
------------------------------------------------------------------------------------------------------------------------
  fold    |        train        |     oos_period      |         base best          | elapsed 
------------------------------------------------------------------------------------------------------------------------
          |                     |                     |   RoMD   |      0050       |         
------------------------------------------------------------------------------------------------------------------------
1/6       | 11-01-01~20-12-31   | 21-01-01~21-12-31   |     0.66 |    1.90 (-1.24) | 00:46:20
2/6       | 12-01-01~21-12-31   | 22-01-01~22-12-31   |    -0.79 |   -0.64 (-0.14) | 00:43:40
3/6       | 13-01-01~22-12-31   | 23-01-01~23-12-31   |     5.63 |    3.81 (+1.82) | 00:48:44
4/6       | 14-01-01~23-12-31   | 24-01-01~24-12-31   |     0.34 |    2.31 (-1.97) | 00:56:26
5/6       | 15-01-01~24-12-31   | 25-01-01~25-12-31   |     0.21 |    1.39 (-1.18) | 00:44:15
6/6       | 16-01-01~25-12-31   | 26-01-01~26-12-31   |     0.55 |    5.96 (-5.42) | 00:46:54
OOS_AVG   | 11-01-01~25-12-31   | 21-01-01~26-12-31   |     1.10 |    2.45 (-1.35) |         
------------------------------------------------------------------------------------------------------------------------

FINALIST AGREE RESULTS
------------------------------------------------------------------------------------------------------------------------
  fold    |        train        |     oos_period      |         base agree         | elapsed 
------------------------------------------------------------------------------------------------------------------------
          |                     |                     |   RoMD   |      0050       |         
------------------------------------------------------------------------------------------------------------------------
1/6       | 11-01-01~20-12-31   | 21-01-01~21-12-31   |     0.82 |    1.90 (-1.08) | 00:46:20
2/6       | 12-01-01~21-12-31   | 22-01-01~22-12-31   |    -0.87 |   -0.64 (-0.23) | 00:43:40
3/6       | 13-01-01~22-12-31   | 23-01-01~23-12-31   |     9.39 |    3.81 (+5.58) | 00:48:44
4/6       | 14-01-01~23-12-31   | 24-01-01~24-12-31   |     0.32 |    2.31 (-1.99) | 00:56:26
5/6       | 15-01-01~24-12-31   | 25-01-01~25-12-31   |     0.29 |    1.39 (-1.10) | 00:44:15
6/6       | 16-01-01~25-12-31   | 26-01-01~26-12-31   |     0.57 |    5.96 (-5.39) | 00:46:54
OOS_AVG   | 11-01-01~25-12-31   | 21-01-01~26-12-31   |     1.75 |    2.45 (-0.70) |         
------------------------------------------------------------------------------------------------------------------------

SEED ENSEMBLE RESULTS
------------------------------------------------------------------------------------------------------------------------
  fold    |        train        |     oos_period      |       base ensemble        | elapsed 
------------------------------------------------------------------------------------------------------------------------
          |                     |                     |   RoMD   |      0050       |         
------------------------------------------------------------------------------------------------------------------------
1/6       | 11-01-01~20-12-31   | 21-01-01~21-12-31   |     0.66 |    1.90 (-1.24) | 00:46:20
2/6       | 12-01-01~21-12-31   | 22-01-01~22-12-31   |    -0.79 |   -0.64 (-0.14) | 00:43:40
3/6       | 13-01-01~22-12-31   | 23-01-01~23-12-31   |     5.63 |    3.81 (+1.82) | 00:48:44
4/6       | 14-01-01~23-12-31   | 24-01-01~24-12-31   |     0.34 |    2.31 (-1.97) | 00:56:26
5/6       | 15-01-01~24-12-31   | 25-01-01~25-12-31   |     0.21 |    1.39 (-1.18) | 00:44:15
6/6       | 16-01-01~25-12-31   | 26-01-01~26-12-31   |     0.55 |    5.96 (-5.42) | 00:46:54
OOS_AVG   | 11-01-01~25-12-31   | 21-01-01~26-12-31   |     1.10 |    2.45 (-1.35) |         
------------------------------------------------------------------------------------------------------------------------

⏳ OOS_CHAIN active replay | schedule policies=4 | period=2021-01-01~2026-12-31 | total=00:56:30
⏱️ OOS_CHAIN active replay context 完成 | contexts=28/28 | years=2021~2026 | policies=base/base_finalist_best/base_finalists_agree/best | chain=00:02:37 | total=00:59:08                 
⏳ OOS_CHAIN active replay 完成 | policies=4/4 | chain=00:00:35 | total=00:59:44               

FINALIST BEST RESULTS
------------------------------------------------------------------------------------------------------------------------
  fold    |        train        |     oos_period      |         base best          | elapsed 
------------------------------------------------------------------------------------------------------------------------
          |                     |                     |   RoMD   |      0050       |         
------------------------------------------------------------------------------------------------------------------------
1/6       | 11-01-01~20-12-31   | 21-01-01~21-12-31   |     0.66 |    1.90 (-1.24) | 00:46:20
2/6       | 12-01-01~21-12-31   | 22-01-01~22-12-31   |    -0.79 |   -0.64 (-0.14) | 00:43:40
3/6       | 13-01-01~22-12-31   | 23-01-01~23-12-31   |     5.63 |    3.81 (+1.82) | 00:48:44
4/6       | 14-01-01~23-12-31   | 24-01-01~24-12-31   |     0.34 |    2.31 (-1.97) | 00:56:26
5/6       | 15-01-01~24-12-31   | 25-01-01~25-12-31   |     0.21 |    1.39 (-1.18) | 00:44:15
6/6       | 16-01-01~25-12-31   | 26-01-01~26-12-31   |     0.55 |    5.96 (-5.42) | 00:46:54
OOS_CHAIN | 11-01-01~25-12-31   | 21-01-01~26-12-31   |     5.46 |    6.00 (-0.54) | 00:59:44
------------------------------------------------------------------------------------------------------------------------

FINALIST AGREE RESULTS
------------------------------------------------------------------------------------------------------------------------
  fold    |        train        |     oos_period      |         base agree         | elapsed 
------------------------------------------------------------------------------------------------------------------------
          |                     |                     |   RoMD   |      0050       |         
------------------------------------------------------------------------------------------------------------------------
1/6       | 11-01-01~20-12-31   | 21-01-01~21-12-31   |     0.82 |    1.90 (-1.08) | 00:46:20
2/6       | 12-01-01~21-12-31   | 22-01-01~22-12-31   |    -0.87 |   -0.64 (-0.23) | 00:43:40
3/6       | 13-01-01~22-12-31   | 23-01-01~23-12-31   |     9.39 |    3.81 (+5.58) | 00:48:44
4/6       | 14-01-01~23-12-31   | 24-01-01~24-12-31   |     0.32 |    2.31 (-1.99) | 00:56:26
5/6       | 15-01-01~24-12-31   | 25-01-01~25-12-31   |     0.29 |    1.39 (-1.10) | 00:44:15
6/6       | 16-01-01~25-12-31   | 26-01-01~26-12-31   |     0.57 |    5.96 (-5.39) | 00:46:54
OOS_CHAIN | 11-01-01~25-12-31   | 21-01-01~26-12-31   |     5.14 |    6.00 (-0.86) | 00:59:44
------------------------------------------------------------------------------------------------------------------------

SEED ENSEMBLE RESULTS
------------------------------------------------------------------------------------------------------------------------
  fold    |        train        |     oos_period      |       base ensemble        | elapsed 
------------------------------------------------------------------------------------------------------------------------
          |                     |                     |   RoMD   |      0050       |         
------------------------------------------------------------------------------------------------------------------------
1/6       | 11-01-01~20-12-31   | 21-01-01~21-12-31   |     0.66 |    1.90 (-1.24) | 00:46:20
2/6       | 12-01-01~21-12-31   | 22-01-01~22-12-31   |    -0.79 |   -0.64 (-0.14) | 00:43:40
3/6       | 13-01-01~22-12-31   | 23-01-01~23-12-31   |     5.63 |    3.81 (+1.82) | 00:48:44
4/6       | 14-01-01~23-12-31   | 24-01-01~24-12-31   |     0.34 |    2.31 (-1.97) | 00:56:26
5/6       | 15-01-01~24-12-31   | 25-01-01~25-12-31   |     0.21 |    1.39 (-1.18) | 00:44:15
6/6       | 16-01-01~25-12-31   | 26-01-01~26-12-31   |     0.55 |    5.96 (-5.42) | 00:46:54
OOS_CHAIN | 11-01-01~25-12-31   | 21-01-01~26-12-31   |     5.46 |    6.00 (-0.54) | 00:59:44
------------------------------------------------------------------------------------------------------------------------

📏 訓練效能摘要: folds=6 | seeds=8 | min_agree=5 | completed=6/6 | total=00:59:44 | CPU avg=79.2% | MEM avg=86.3% | HD avg=12.1%
💾 輸出檔案
  base_best: models\research\breakout_quality\binary_dl_filter_param_adaptation\risk_only_rolling\p3_dl_on_trained\active_params\roos_base_best.json
  base_agree: models\research\breakout_quality\binary_dl_filter_param_adaptation\risk_only_rolling\p3_dl_on_trained\active_params\roos_base_finalists_agree.json
  base_ensemble: models\research\breakout_quality\binary_dl_filter_param_adaptation\risk_only_rolling\p3_dl_on_trained\active_params\roos_ensemble_base.json

[no_filter] 建立市場與訊號快取
[no_filter] 執行 active-param ensemble replay 2021-01-01 ～ 2026-03-02
📦 歷史資料：記憶體快取建立完成｜標的=556｜清洗移除=18165列（異常OHLCV=18165, 重複日期=0）｜資料載入/清洗摘要=153筆（已寫入 issue log）
📦 建立 ensemble active param 快取 [1/6] 生效日=2021-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [2/6] 生效日=2022-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [3/6] 生效日=2023-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [4/6] 生效日=2024-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [5/6] 生效日=2025-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [6/6] 生效日=2026-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
✅ active-param ensemble replay 準備完成：2021-01-01~2026-01-01，members=1，min_agree=1。
⏳ 推進中: 2026-02-25 | 日序: 1246/1248 | 資產: 2,288,724 | 水位:  98.1% | elapsed=00:00:03...
[quality_filter] 建立市場與訊號快取
[quality_filter] 執行 active-param ensemble replay 2021-01-01 ～ 2026-03-02
📦 歷史資料：記憶體快取建立完成｜標的=556｜清洗移除=18165列（異常OHLCV=18165, 重複日期=0）｜資料載入/清洗摘要=153筆（已寫入 issue log）
📦 建立 ensemble active param 快取 [1/6] 生效日=2021-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [2/6] 生效日=2022-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [3/6] 生效日=2023-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [4/6] 生效日=2024-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [5/6] 生效日=2025-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [6/6] 生效日=2026-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
✅ active-param ensemble replay 準備完成：2021-01-01~2026-01-01，members=1，min_agree=1。
⏳ 推進中: 2026-02-25 | 日序: 1246/1248 | 資產: 1,834,560 | 水位:  89.2% | elapsed=00:00:01...
====================================================================================================
 Breakout Quality 策略經濟效果對照
====================================================================================================
期間                    ：2021-01-01 ～ 2026-03-02
參數檔                  ：models/roos_base_best.json
參數型態                ：rolling_active_param_ensemble
參數 selector           ：base_finalist_best
Runtime members         ：1～1；min_agree=1
比較設計                ：historical_active_param_oos
歷史 active-param 無前視：True
Dataset                 ：full
Score source            ：canonical_runtime
Ranking policy          ：None
Optional entry filters  ：current
Benchmark               ：0050
唯一差異                ：use_breakout_quality_filter=False vs True

1. 主要結果
-----------
指標                No filter  Active quality filter  差異       判讀   
------------------  ---------  ---------------------  ---------  -------
淨總報酬              129.08%                 85.79%    -43.29%  🔴 惡化
最大回撤               17.41%                 26.34%     +8.93%  🔴 惡化
報酬／最大回撤           7.42                   3.26      -4.16  🔴 惡化
年化報酬               17.43%                 12.76%     -4.67%  🔴 惡化
Log R²                 0.9436                 0.7870    -0.1567  🔴 惡化
月勝率                 63.49%                 53.97%     -9.52%  🔴 惡化
交易數                    407                    396        -11  ⚪ 中性
勝率                   41.52%                 37.37%     -4.15%  🔴 惡化
Payoff                   2.95                   2.80      -0.16  🔴 惡化
EV                     0.53 R                 0.36 R    -0.18 R  🔴 惡化
平均曝險               84.02%                 68.22%    -15.80%  🟡 注意
最差完整年度            7.90%                 -6.86%    -14.77%  🔴 惡化
平均每日可掛單候選      50.55                  20.98     -29.57  ⚪ 中性
候選供給不足日         163 日                 383 日    +220 日  🔴 惡化
期末未滿倉日           745 日                 661 日     -84 日  🟢 改善
期末持股缺口總和    1871 格日              2360 格日  +489 格日  🔴 惡化

2. 年度報酬
-----------
年度  No filter  Active quality filter  差異     判讀     完整年度
----  ---------  ---------------------  -------  -------  --------
2021     21.28%                  5.64%  -15.65%  🔴 惡化     是   
2022      7.90%                 -6.86%  -14.77%  🔴 惡化     是   
2023     34.17%                 43.65%   +9.47%  🟢 改善     是   
2024     14.78%                 10.85%   -3.93%  🔴 惡化     是   
2025      9.80%                  5.77%   -4.04%  🔴 惡化     是   
2026      3.51%                 12.12%   +8.61%  🟢 改善     否   

3. 判讀限制
-----------
本報表只驗證固定 active 操作點，不得依結果回頭調整 threshold、模型或訓練條件。

工件輸出
--------
策略比較 Markdown：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p0_original_roos_formal/strategy_comparison.md
策略比較 JSON    ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p0_original_roos_formal/strategy_comparison.json
年度比較 CSV     ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p0_original_roos_formal/yearly_returns_comparison.csv
交易歸因 Markdown：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p0_original_roos_formal/trade_attribution.md

[no_filter] 建立市場與訊號快取
[no_filter] 執行 active-param ensemble replay 2021-01-01 ～ 2026-03-02
📦 歷史資料：記憶體快取建立完成｜標的=556｜清洗移除=18165列（異常OHLCV=18165, 重複日期=0）｜資料載入/清洗摘要=153筆（已寫入 issue log）
📦 建立 ensemble active param 快取 [1/6] 生效日=2021-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [2/6] 生效日=2022-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [3/6] 生效日=2023-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [4/6] 生效日=2024-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [5/6] 生效日=2025-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [6/6] 生效日=2026-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
✅ active-param ensemble replay 準備完成：2021-01-01~2026-01-01，members=1，min_agree=1。
⏳ 推進中: 2026-02-25 | 日序: 1246/1248 | 資產: 2,387,859 | 水位:  99.0% | elapsed=00:00:06...
[quality_filter] 建立市場與訊號快取
[quality_filter] 執行 active-param ensemble replay 2021-01-01 ～ 2026-03-02
📦 歷史資料：記憶體快取建立完成｜標的=556｜清洗移除=18165列（異常OHLCV=18165, 重複日期=0）｜資料載入/清洗摘要=153筆（已寫入 issue log）
📦 建立 ensemble active param 快取 [1/6] 生效日=2021-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [2/6] 生效日=2022-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [3/6] 生效日=2023-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [4/6] 生效日=2024-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [5/6] 生效日=2025-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [6/6] 生效日=2026-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
✅ active-param ensemble replay 準備完成：2021-01-01~2026-01-01，members=1，min_agree=1。
⏳ 推進中: 2026-02-25 | 日序: 1246/1248 | 資產: 2,451,405 | 水位:  74.4% | elapsed=00:00:03...
====================================================================================================
 Breakout Quality 策略經濟效果對照
====================================================================================================
期間                    ：2021-01-01 ～ 2026-03-02
參數檔                  ：models/roos_base_best.json
參數型態                ：rolling_active_param_ensemble
參數 selector           ：base_finalist_best
Runtime members         ：1～1；min_agree=1
比較設計                ：historical_active_param_oos
歷史 active-param 無前視：True
Dataset                 ：full
Score source            ：canonical_runtime
Ranking policy          ：None
Optional entry filters  ：all-off
Benchmark               ：0050
唯一差異                ：use_breakout_quality_filter=False vs True

1. 主要結果
-----------
指標                No filter  Active quality filter  差異        判讀   
------------------  ---------  ---------------------  ----------  -------
淨總報酬              138.09%                145.92%      +7.83%  🟢 改善
最大回撤               13.35%                 18.31%      +4.95%  🔴 惡化
報酬／最大回撤          10.34                   7.97       -2.37  🔴 惡化
年化報酬               18.31%                 19.06%      +0.74%  🟢 改善
Log R²                 0.8997                 0.8577     -0.0420  🔴 惡化
月勝率                 66.67%                 63.49%      -3.17%  🔴 惡化
交易數                    384                    446         +62  ⚪ 中性
勝率                   41.15%                 37.89%      -3.25%  🔴 惡化
Payoff                   3.31                   3.19       -0.12  🔴 惡化
EV                     0.47 R                 0.35 R     -0.11 R  🔴 惡化
平均曝險               92.01%                 79.04%     -12.97%  🟡 注意
最差完整年度           -0.01%                -10.23%     -10.22%  🔴 惡化
平均每日可掛單候選      82.84                  37.52      -45.31  ⚪ 中性
候選供給不足日          38 日                 110 日      +72 日  🔴 惡化
期末未滿倉日           926 日                 598 日     -328 日  🟢 改善
期末持股缺口總和    2710 格日              1380 格日  -1330 格日  🟢 改善

2. 年度報酬
-----------
年度  No filter  Active quality filter  差異     判讀     完整年度
----  ---------  ---------------------  -------  -------  --------
2021      8.31%                 14.34%   +6.03%  🟢 改善     是   
2022     -0.01%                -10.23%  -10.22%  🔴 惡化     是   
2023     66.29%                 80.78%  +14.49%  🟢 改善     是   
2024     14.14%                 13.70%   -0.44%  🔴 惡化     是   
2025      9.07%                  9.05%   -0.02%  🔴 惡化     是   
2026      6.20%                  6.90%   +0.69%  🟢 改善     否   

3. 判讀限制
-----------
本報表只驗證固定 active 操作點，不得依結果回頭調整 threshold、模型或訓練條件。

工件輸出
--------
策略比較 Markdown：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p1_original_roos_all_off/strategy_comparison.md
策略比較 JSON    ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p1_original_roos_all_off/strategy_comparison.json
年度比較 CSV     ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p1_original_roos_all_off/yearly_returns_comparison.csv
交易歸因 Markdown：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p1_original_roos_all_off/trade_attribution.md

[no_filter] 建立市場與訊號快取
[no_filter] 執行 active-param ensemble replay 2021-01-01 ～ 2026-03-02
📦 歷史資料：記憶體快取建立完成｜標的=556｜清洗移除=18165列（異常OHLCV=18165, 重複日期=0）｜資料載入/清洗摘要=153筆（已寫入 issue log）
📦 建立 ensemble active param 快取 [1/6] 生效日=2021-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [2/6] 生效日=2022-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [3/6] 生效日=2023-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [4/6] 生效日=2024-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [5/6] 生效日=2025-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [6/6] 生效日=2026-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
✅ active-param ensemble replay 準備完成：2021-01-01~2026-01-01，members=1，min_agree=1。
⏳ 推進中: 2026-02-25 | 日序: 1246/1248 | 資產: 2,688,163 | 水位:  98.7% | elapsed=00:00:07...
[quality_filter] 建立市場與訊號快取
[quality_filter] 執行 active-param ensemble replay 2021-01-01 ～ 2026-03-02
📦 歷史資料：記憶體快取建立完成｜標的=556｜清洗移除=18165列（異常OHLCV=18165, 重複日期=0）｜資料載入/清洗摘要=153筆（已寫入 issue log）
📦 建立 ensemble active param 快取 [1/6] 生效日=2021-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [2/6] 生效日=2022-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [3/6] 生效日=2023-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [4/6] 生效日=2024-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [5/6] 生效日=2025-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [6/6] 生效日=2026-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
✅ active-param ensemble replay 準備完成：2021-01-01~2026-01-01，members=1，min_agree=1。
⏳ 推進中: 2026-02-25 | 日序: 1246/1248 | 資產: 2,435,958 | 水位:  91.2% | elapsed=00:00:04...
====================================================================================================
 Breakout Quality 策略經濟效果對照
====================================================================================================
期間                    ：2021-01-01 ～ 2026-03-02
參數檔                  ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/p2_dl_off_trained/active_params/roos_base_best.json
參數型態                ：rolling_active_param_ensemble
參數 selector           ：base_finalist_best
Runtime members         ：1～1；min_agree=1
比較設計                ：historical_active_param_oos
歷史 active-param 無前視：True
Dataset                 ：full
Score source            ：canonical_runtime
Ranking policy          ：None
Optional entry filters  ：all-off
Benchmark               ：0050
唯一差異                ：use_breakout_quality_filter=False vs True

1. 主要結果
-----------
指標                No filter  Active quality filter  差異        判讀   
------------------  ---------  ---------------------  ----------  -------
淨總報酬              166.69%                142.70%     -23.99%  🔴 惡化
最大回撤               15.41%                 21.75%      +6.34%  🔴 惡化
報酬／最大回撤          10.82                   6.56       -4.26  🔴 惡化
年化報酬               20.95%                 18.76%      -2.19%  🔴 惡化
Log R²                 0.8842                 0.8523     -0.0319  🔴 惡化
月勝率                 68.25%                 61.90%      -6.35%  🔴 惡化
交易數                    336                    374         +38  ⚪ 中性
勝率                   42.26%                 39.84%      -2.42%  🔴 惡化
Payoff                   3.62                   3.07       -0.54  🔴 惡化
EV                     0.73 R                 0.64 R     -0.09 R  🔴 惡化
平均曝險               92.13%                 78.24%     -13.90%  🟡 注意
最差完整年度            0.84%                 -6.42%      -7.26%  🔴 惡化
平均每日可掛單候選      86.40                  40.01      -46.39  ⚪ 中性
候選供給不足日          33 日                 135 日     +102 日  🔴 惡化
期末未滿倉日           912 日                 563 日     -349 日  🟢 改善
期末持股缺口總和    2572 格日              1239 格日  -1333 格日  🟢 改善

2. 年度報酬
-----------
年度  No filter  Active quality filter  差異     判讀     完整年度
----  ---------  ---------------------  -------  -------  --------
2021     23.08%                  6.25%  -16.83%  🔴 惡化     是   
2022      0.84%                  0.64%   -0.20%  🔴 惡化     是   
2023     80.22%                 71.31%   -8.91%  🔴 惡化     是   
2024     11.21%                 29.99%  +18.78%  🟢 改善     是   
2025      2.25%                 -6.42%   -8.67%  🔴 惡化     是   
2026      4.85%                  8.91%   +4.06%  🟢 改善     否   

3. 判讀限制
-----------
本報表只驗證固定 active 操作點，不得依結果回頭調整 threshold、模型或訓練條件。

工件輸出
--------
策略比較 Markdown：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p2_dl_off_trained_all_off/strategy_comparison.md
策略比較 JSON    ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p2_dl_off_trained_all_off/strategy_comparison.json
年度比較 CSV     ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p2_dl_off_trained_all_off/yearly_returns_comparison.csv
交易歸因 Markdown：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p2_dl_off_trained_all_off/trade_attribution.md

[no_filter] 建立市場與訊號快取
[no_filter] 執行 active-param ensemble replay 2021-01-01 ～ 2026-03-02
📦 歷史資料：記憶體快取建立完成｜標的=556｜清洗移除=18165列（異常OHLCV=18165, 重複日期=0）｜資料載入/清洗摘要=153筆（已寫入 issue log）
📦 建立 ensemble active param 快取 [1/6] 生效日=2021-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [2/6] 生效日=2022-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [3/6] 生效日=2023-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [4/6] 生效日=2024-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [5/6] 生效日=2025-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [6/6] 生效日=2026-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
✅ active-param ensemble replay 準備完成：2021-01-01~2026-01-01，members=1，min_agree=1。
⏳ 推進中: 2026-02-25 | 日序: 1246/1248 | 資產: 2,212,177 | 水位:  98.7% | elapsed=00:00:07...
[quality_filter] 建立市場與訊號快取
[quality_filter] 執行 active-param ensemble replay 2021-01-01 ～ 2026-03-02
📦 歷史資料：記憶體快取建立完成｜標的=556｜清洗移除=18165列（異常OHLCV=18165, 重複日期=0）｜資料載入/清洗摘要=153筆（已寫入 issue log）
📦 建立 ensemble active param 快取 [1/6] 生效日=2021-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [2/6] 生效日=2022-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [3/6] 生效日=2023-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [4/6] 生效日=2024-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [5/6] 生效日=2025-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
📦 建立 ensemble active param 快取 [6/6] 生效日=2026-01-01 member=1...
✅ 投組記憶體 replay context 完成｜標的=556｜mode=parallel｜workers=8
✅ active-param ensemble replay 準備完成：2021-01-01~2026-01-01，members=1，min_agree=1。
⏳ 推進中: 2026-02-25 | 日序: 1246/1248 | 資產: 1,776,606 | 水位:  97.8% | elapsed=00:00:04...
====================================================================================================
 Breakout Quality 策略經濟效果對照
====================================================================================================
期間                    ：2021-01-01 ～ 2026-03-02
參數檔                  ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/p3_dl_on_trained/active_params/roos_base_best.json
參數型態                ：rolling_active_param_ensemble
參數 selector           ：base_finalist_best
Runtime members         ：1～1；min_agree=1
比較設計                ：historical_active_param_oos
歷史 active-param 無前視：True
Dataset                 ：full
Score source            ：canonical_runtime
Ranking policy          ：None
Optional entry filters  ：all-off
Benchmark               ：0050
唯一差異                ：use_breakout_quality_filter=False vs True

1. 主要結果
-----------
指標                No filter  Active quality filter  差異       判讀   
------------------  ---------  ---------------------  ---------  -------
淨總報酬              119.77%                 78.67%    -41.10%  🔴 惡化
最大回撤               11.83%                 21.53%     +9.70%  🔴 惡化
報酬／最大回撤          10.12                   3.65      -6.47  🔴 惡化
年化報酬               16.49%                 11.91%     -4.58%  🔴 惡化
Log R²                 0.9451                 0.7723    -0.1728  🔴 惡化
月勝率                 63.49%                 61.90%     -1.59%  🔴 惡化
交易數                    275                    278         +3  ⚪ 中性
勝率                   47.27%                 38.85%     -8.42%  🔴 惡化
Payoff                   3.28                   2.90      -0.38  🔴 惡化
EV                     1.16 R                 0.55 R    -0.61 R  🔴 惡化
平均曝險               92.06%                 76.79%    -15.27%  🟡 注意
最差完整年度            1.33%                 -7.57%     -8.90%  🔴 惡化
平均每日可掛單候選      82.80                  30.66     -52.13  ⚪ 中性
候選供給不足日          36 日                 114 日     +78 日  🔴 惡化
期末未滿倉日           860 日                 517 日    -343 日  🟢 改善
期末持股缺口總和    1859 格日              1156 格日  -703 格日  🟢 改善

2. 年度報酬
-----------
年度  No filter  Active quality filter  差異     判讀     完整年度
----  ---------  ---------------------  -------  -------  --------
2021     19.63%                  3.19%  -16.44%  🔴 惡化     是   
2022      1.33%                 -7.57%   -8.90%  🔴 惡化     是   
2023     21.06%                 54.09%  +33.03%  🟢 改善     是   
2024     33.40%                 18.29%  -15.11%  🔴 惡化     是   
2025      5.04%                 -3.02%   -8.06%  🔴 惡化     是   
2026      6.87%                  5.97%   -0.90%  🔴 惡化     否   

3. 判讀限制
-----------
本報表只驗證固定 active 操作點，不得依結果回頭調整 threshold、模型或訓練條件。

工件輸出
--------
策略比較 Markdown：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p3_dl_on_trained_all_off/strategy_comparison.md
策略比較 JSON    ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p3_dl_on_trained_all_off/strategy_comparison.json
年度比較 CSV     ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p3_dl_on_trained_all_off/yearly_returns_comparison.csv
交易歸因 Markdown：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/replay/p3_dl_on_trained_all_off/trade_attribution.md

====================================================================================================
 Binary DL Filter 4 Parameters × 2 DL States Gate
====================================================================================================

參數基準     ：base-finalist-best
搜尋參數     ：atr_len / atr_buy_tol / atr_times_init / atr_times_trail
P2訓練       ：rules全關／DL關
P3訓練       ：rules全關／DL開／Binary PIT
Binary PIT   ：READY
固定 Buy sort：原 position-aware buy-sort


1. 八個操作點
-------------

組別  參數               規則                    DL  報酬     MDD     RoMD   EV      曝險    交易
----  -----------------  ----------------------  --  -------  ------  -----  ------  ------  ----
A0    P0 原ROOS          原正式設定              關  129.08%  17.41%  7.42   0.53 R  84.02%  0   
B0    P0 原ROOS          原正式設定              開  85.79%   26.34%  3.26   0.36 R  68.22%  0   
A1    P1 原ROOS          Rule-based filters全關  關  138.09%  13.35%  10.34  0.47 R  92.01%  0   
B1    P1 原ROOS          Rule-based filters全關  開  145.92%  18.31%  7.97   0.35 R  79.04%  0   
A2    P2 DL-off-trained  Rule-based filters全關  關  166.69%  15.41%  10.82  0.73 R  92.13%  0   
B2    P2 DL-off-trained  Rule-based filters全關  開  142.70%  21.75%  6.56   0.64 R  78.24%  0   
A3    P3 DL-on-trained   Rule-based filters全關  關  119.77%  11.83%  10.12  1.16 R  92.06%  0   
B3    P3 DL-on-trained   Rule-based filters全關  開  78.67%   21.53%  3.65   0.55 R  76.79%  0   


2. 各參數下DL增量
-----------------

比較   參數               Δ報酬     ΔMDD    ΔRoMD  ΔEV      Δ曝險   
-----  -----------------  --------  ------  -----  -------  --------
B0−A0  P0 原ROOS          -43.29pp  8.93pp  -4.16  -0.18 R  -15.80pp
B1−A1  P1 原ROOS          7.83pp    4.95pp  -2.37  -0.11 R  -12.97pp
B2−A2  P2 DL-off-trained  -23.99pp  6.34pp  -4.26  -0.09 R  -13.90pp
B3−A3  P3 DL-on-trained   -41.10pp  9.70pp  -6.47  -0.61 R  -15.27pp


3. 規則與參數適應
-----------------

比較                          Δ報酬     ΔMDD     ΔRoMD  ΔEV      Δ曝險   
----------------------------  --------  -------  -----  -------  --------
A1−A0：原ROOS關閉rules        9.01pp    -4.05pp  2.93   -0.07 R  7.99pp  
A2−A1：DL-off-trained對No-DL  28.60pp   2.06pp   0.48   0.27 R   0.12pp  
B2−B1：DL-off-trained對DL     -3.22pp   3.44pp   -1.41  0.29 R   -0.81pp 
A3−A1：DL-on-trained拿掉DL    -18.32pp  -1.52pp  -0.22  0.70 R   0.05pp  
B3−B1：DL-on-trained對DL      -67.25pp  3.23pp   -4.32  0.20 R   -2.25pp 
B3−A2：最終公平比較           -88.01pp  6.12pp   -7.16  -0.18 R  -15.34pp
Interaction：(B3−A3)−(B2−A2)  -17.11pp  -        -      -        -       

主判定：B3−A2；Interaction只判斷DL-aware參數是否改善DL增量，不能取代絕對績效。


4. 風險參數
-----------

參數             P0/P1 原ROOS                  P2 DL-off-trained             P3 DL-on-trained            
---------------  ----------------------------  ----------------------------  ----------------------------
atr_len          19, 12, 5, 6, 4, 9            3, 25, 5, 7, 4, 26            4, 4, 3, 5, 8, 6            
atr_buy_tol      1.0, 3.3, 3.7, 4.3, 3.9, 4.5  2.8, 2.4, 2.8, 3.2, 3.4, 3.7  3.9, 3.9, 3.8, 3.0, 4.4, 4.1
atr_times_init   4.4, 3.8, 4.2, 4.2, 3.9, 4.2  4.2, 4.5, 4.1, 4.5, 3.9, 4.5  4.4, 4.4, 4.3, 4.4, 4.0, 4.2
atr_times_trail  3.7, 3.6, 3.4, 4.4, 3.6, 4.1  3.8, 4.3, 3.8, 3.9, 4.0, 4.4  4.4, 4.4, 4.4, 4.5, 4.5, 4.0


5. 無前視契約
-------------

PIT Manifest：models/research/breakout_quality/binary_point_in_time_scores/breakout_quality_v1/inception_time_v1/unique_group_sampling/manifest.json
PIT Scores  ：models/research/breakout_quality/binary_point_in_time_scores/breakout_quality_v1/inception_time_v1/unique_group_sampling/scores.csv
PIT period  ：{'start': '2006-12-01', 'end': '2026-03-02'}
禁止        ：final forward-OOS／research／Selection in-sample scores回灌optimizer

工件輸出
--------
4×2計畫 ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/strategy_dl_filter_param_adapt_plan.json
4×2報表 ：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/strategy_dl_filter_param_adapt_gate.md
4×2 JSON：models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/strategy_dl_filter_param_adapt_gate.json
PS C:\Users\User\Desktop\My_Quant_Project> 
PS C:\Users\User\Desktop\My_Quant_Project> 
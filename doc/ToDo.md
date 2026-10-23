====================================================================================================
 Target Ground Truth -> Strategy RoMD (HINDSIGHT ORACLE / NOT PIT)
====================================================================================================

Period: 2021-01-01 ~ 2025-12-22 (40D-mature common window)
Common stock-day cohort: 643,617
Benchmark: end_to_end_v1, same-seed pairs N=4
Report schema: 2, Oracle treatments: 11 (formal Research Audit; not Standard SOP)
Ranking: Exact K/R0, Broker-Cash, same seed/P16; future truth changes ranking only.
MFE40/D40 zero denominator: 35,313 positive-MFE (+infinity); 0 zero/zero (rank 0).
AO/CG/CH = equal-rank 50/50 geometry PROXIES, not neural-network model heads.

Target            Return Mean  Ann. Mean  MDD Mean  RoMD Mean  RoMD SD  Delta RoMD  Win/Seed  EV(A) Mean
----------------  -----------  ---------  --------  ---------  -------  ----------  --------  ----------
C96 Same-Window   +122.24%     +17.42%    15.05%    8.14       0.44     +0.00       -         +0.81 R   
AO40_EqualRank    +219.94%     +26.38%    13.02%    17.04      1.98     +8.91       4/4       +1.36 R   
CG40_EqualRank    +237.44%     +27.72%    12.96%    18.32      1.40     +10.18      4/4       +1.38 R   
CH20_EqualRank    +205.27%     +25.16%    12.50%    16.43      1.74     +8.29       4/4       +1.19 R   
U40               +299.06%     +31.86%    13.29%    22.45      5.29     +14.32      4/4       +1.38 R   
U20               +277.72%     +30.54%    12.56%    22.11      3.44     +13.97      4/4       +1.34 R   
FirstBreach_NetR  +295.29%     +31.72%    13.93%    21.21      4.26     +13.07      4/4       +1.31 R   
MFE40_Control     +260.46%     +29.25%    13.16%    19.65      3.10     +11.51      4/4       +1.29 R   
MFE40_over_D40    +179.65%     +22.90%    13.17%    13.84      3.44     +5.70       4/4       +1.24 R   
MFE20             +256.67%     +28.93%    13.68%    18.74      4.58     +10.60      4/4       +1.22 R   
MAE40             +159.91%     +21.19%    13.30%    12.02      0.60     +3.89       4/4       +1.14 R   
MAE20             +172.70%     +22.35%    14.23%    12.15      1.27     +4.01       4/4       +1.11 R   

NOTICE: Future labels are hindsight only; not legal for PIT/Trading.

====================================================================================================
 Target Ground Truth → Strategy RoMD (Oracle) 最近結果
====================================================================================================

Audit               狀態  Markdown                                                                  
------------------  ----  --------------------------------------------------------------------------
target_oracle_romd  存在  outputs/audit/breakout_quality/reusable/target_oracle_romd/latest/audit.md


- imrove DL learnability
    - aduit DL target vs. ROMD Correlation to improve DL 績效轉換率
    - input with 漲/跌家數
    - input with 還原/非還原價
    - input with 三大法人籌碼資訊
    - input with EPS財報/ 基本面 
    - attention (temperal / cross-secction)

- DL Branch Research
    - sell using DL
    - 0050 only DL
    - non breakout strategy


## To do

- LLM-based learning
- 改為線上自動交易模擬 (依當天資產決定持有股數、沒買到下一檔(現在就是))
- online虛擬交易 (小時交易)

## Pending (useless)
- 考慮資投入資金效率、持有天數 

## Done
- 實際交易模擬
- 並訓練增/減不同條件 (0/1)
- Git -> 匯入程式碼操作
- 俢正downloader (刪除不符合的)
- 是否打敗大盤
- 檔案架構整理
- 直接訓練最終指標與買進策略 (end-to-end approach)
- 過濾一字鎖死漲停還能買入
- 顯示買到/沒買到的次數
- 跌停鎖死當天是賣不掉的、後面可以繼續賣、顯示未成功賣出筆數
- 訓練歷史績效參數
- 買入訊號依"可投入最多"資產，也就是考慮初始停損與股價
- 將歷史績效過濾條件、風險比也納入end-to-end架構訓練
- 解決當天無法買入真實性問題 
- 使買入限價/初始停損/停利/移動停損符合開盤前操作的限制，並完美符合R的交互關係
- 更新交易次數的penality算法 (不使用固定值)
- 顯示年化報酬率
- 加入winrate * count買入排序
- 更新EV算法與對應訓練方法

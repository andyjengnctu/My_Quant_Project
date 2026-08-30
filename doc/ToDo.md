

- restreucture
    - research 報表是否需要集中管理架構? 包含報表用色規則是否需要集中管理? 目前架構現況?
    - report refine
        - [1][1]
            - 確認如果模型/報表已經在就不要重跑
            - Learnability 多一欄是 = Top 10% Target - Bottom 10% Target
            - Ranking / Boundary移除 Top-K Target，並將Lift更名為Top-K Lift，競爭日要對比該pool的母體數量
            - Upside / Downside Alignment
                - 移除Pred-Safety→Target rho / Pred-Safety→Score rho 這兩個非共通的指標
                - Target→Low-Adverse rho -> Target→Safety rho
                - Score→Low-Adverse rho -> Score→Safety rho
            - Top-tail Economic Quality
                - Top10 Adverse  -> Top10 Low-Adverse     
                - HM/HS HM/HSx合併顯示為 24.51% (1.08x) 
                - 也加入 HM/LS, LM/HS, LM/LS
            - 4/5往後移, Evidence Coverage 放到最後
        -[1] [4] 報表項目與sop相同，可任意加多個比較對像，先進行13H, 13AF, 13AH三個比較，如果模型已經在不要重訓，如果不在自動重訓



- imrove DL module
    - attention (temperal / cross-secction)
    - self-learned history and L lengths
    - hihger weight for high mfe or high safety   
    - retrain min parameters
    - simply label
        - the 40t day's R
        - self learn the label days and input days
    - sell using score    
    - non breakout strategy
    - 三大法人籌碼資訊/ EPS財報/ 基本面 

- trading
    - add a buy list to decide the stop prices using full roos base-fanlist-best
    - frozen 2026/3/2 data for research purpose, latest for trading


## To do
- 如何讓你依據投組結果，包含分析K線交易過程，提供我策略升級建議
    - 加入大盤過濾 +  加入低點買入 
    - 加入盤整期策略
- DRL-based learning
- LLM-based learning
- ajd/raw整合交易策略 
- 快速檢視不同獎勵函式結果 (排行）
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

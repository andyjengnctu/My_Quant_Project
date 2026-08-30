

- restreucture
    - research 報表是否需要集中管理架構? 包含報表用色規則是否需要集中管理? 目前架構現況?
    - [1] 下面換成
        - [1] Foward OOS 模型訓練 <- 也就是現在的[1]
        - [2] Rolling OOS 模型訓練 <- 報表與[1] 一致 + Rolling獨有
        - [3] Foward OOS 模型比較 <- 也就是現在的 [4]
        - [4] Rolling OOS 模型比較 <- 報表與[3] 一致 + Rolling獨有
        - [5] Timing Mode｜Rolling 訓練前後比較  [工程]

        原[3] Fixed-Winddow Rolling 刪除
        只需要設定一次模型訓練對像 [1]、[2] 都可以接通使用
        只需要設定一次模型比較與測試清單 [3]、[4] 都可接通使用



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

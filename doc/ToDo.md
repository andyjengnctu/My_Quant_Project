- Trading
    - 股票中文名稱
    - 自動更新太干擾

    - 在交易中心加入掛單區
        - 放在scanner pool與持股決策區之間
        - 支援策略/手動選股，試算預留成本/股數，確保不超過金費/持股上限，並算好買入限價/停損等盤前資訊
        - 加入手選股時，依然套相同參數公式，計算相關盤前訊息，並可以單股回測檢示對應的線圖與資訊，買訊日圖示就從"買訊"改為"掛單"，其他都一樣
        - 確定買入後，可以再點選轉移到持有決策中，轉移前可修實際成交價與股數，無有成交的可手動刪除
    - 支股決策區
        - 支援手動股套用相同參數計算停損/停利/traiing/sell等，並可在單股回測檢視
        - 依然可以不透過掛單區在這補單
        - 如果不是透過掛單區補的手動股，停利就依買入前日收盤套相同參數計算
        - 持股日終推進支援策略/手動股




- restreucture
    - 刪除不再需要的相容層的code，精簡程式也避免之後誤接


- imrove DL learnability
    - 還是應該廣義成圖形辦識的多層結構，而不是人為克意去分層
    - input with 還原/非還原價
    - input with 漲/跌家數
    - input with lowerbound k style
    - more epoch to avoid fast convergence
    - attention (temperal / cross-secction)
    - self-learned history and L lengths
    - simply label
        - the 40t day's R
        - self learn the label days and input days

- improve DL Transferability
    - capital-aware DL with risk%, cap, stop input
    - retrain min parameters
    - sell using score    
    - non breakout strategy
    - 三大法人籌碼資訊/ EPS財報/ 基本面 




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

- Trading
    - 股票中文名稱
    - 儲存與自動載入已計算過的侯選股，不要每次開啟workben都動新計算

    - 帳戶儀表: 現金餘額、持股市值、帳戶淨值
    - 績效統計移到帳戶併表下方，包含庫存股/平倉股/加總: 股票市值、持有成本、未實現損益、報酬率、勝率、期望值、風報比，其它不要
    - 券商的買賣手續費、稅金是取無調件捨去到整數，現金、淨值、市值、價金、持有成本、損益、手續費、稅金呈現皆無小數點，市價、成交價、均價、各種率呈現小數點第二位



    - 不是全部的數字都要紅綠，只有跟績效相關的，如損益、報酬率
    - 買入登入後，自動跳到帳務中心並更新狀態
    - 表格欄位按一下由小到大排序，再按一下反向，預測是日期由小到大排序
    - 點選表格某股的時候選股該股，再點一下取消選取
    - 買入明細預設是全部，點某檔庫存只顯示對應該庫存對應的明細，例如庫存3000股，對應1000股、2000股兩個明細，消消點選又回到全部顯示
    - 表格最多10列，超過會出現上一頁、下一頁按鍵來換頁
    - 每個股票表格最左邊都有一個圖示，點下去跳到該股的單股回測檢示
    - 點選交易日框出現日曆，不是另外有一個按鍵
    - 無論是手動/策略的都要可以刪改


- Trading (performance)
    - Scanner按下去後一開始會卡很久才開始計算
    - 下單後都要等很久才跳出訊息


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

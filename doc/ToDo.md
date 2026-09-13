

- 實際交易頁面
- 目前的Trading實際交易頁面設計太複雜了，我需要的核心功能如下，作改版: 
    - 每天依策略依序列出scanner掃出的股票pool，與類型/價位/風控/歷史效等相關資訊，我看完可以自已到卷商下單
    - 點選某一檔股票，可以在單股回測檢視頁面顯示
    - 單股回測檢示頁面按下計算侯選股也可以列出最新的pool，單股回測檢示援research/trading切換
    - 基本上單股回測檢視與實際交易由scanner檢示的資訊一致，只是前者有現有k線輔助視覺化，後者以表格/文字為主
    - 可設定持有股，持有股可以從scanner掃出的，也可以以是使用者自已隨意買的，點下時一樣可以在單股回測檢示，單股回測也有下拉選單檢示持有股
    - 帳戶資訊計算持有股價值 + 餘額等，提供風控計算
    - 績效統計區支援對持有、賣出、持有+賣出的相關期效統計
    - 帳戶/持有股/績效統計等相關資訊另外用一個account/資料匣紀錄
    - 最上層以資訊儀表版顯示重要資訊


我的券商會對持有股會顯示股票、均價、股數、持有成本、市價、市值、損益、報酬率 
針對某檔持股，點進明細會逐筆顯示成交日、成交價、數量、價金、買入手續費、持有成本
賣出的明細會顯示: 股票、日期、股數、成交價、價金、賣出手續費、交易稅、損益、報酬率，並會有沖抵明細

價金 = 成交價 * 股數 
持有成本 = 價金 * 手續費 
手續費是未打折前的0.001425
損益 = 賣出價金 - 賣出手續費 - 交易稅 - 持有成本 

買賣時應手動輸入股票、數量、成交價、成交日，其它都自動計算

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

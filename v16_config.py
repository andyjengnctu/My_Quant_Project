from dataclasses import dataclass

@dataclass
class V16StrategyParams:
    """v16 策略的核心參數設定檔 (完美對應 TradingView 的 Inputs)"""
    
    # 1. 資金與風險控管
    initial_capital: float = 4000000.0
    fixed_risk: float = 0.01            # 單筆固定風險限制 (嚴格鎖定 1%)
    tp_percent: float = 0.5             # 半平倉停利比例 (50%)s
    
    # 2. 停損與追價參數 (ATR)
    atr_len: int = 6                    # ATR 週期
    atr_times_init: float = 3.5         # 初始停損 (ATR倍數)
    atr_times_trail: float = 4.0        # 移動停損 (ATR倍數)
    atr_buy_tol: float = 1.5            # 買入追價容忍度 (ATR倍數)
    
    # 3. 技術指標參數
    high_len: int = 100                 # 創N日新高週期
    bb_len: int = 20                    # 布林通道週期
    bb_mult: float = 2.0                # 布林通道標準差
    kc_len: int = 20                    # 阿肯那通道週期
    kc_mult: float = 2.0                # 阿肯那通道倍數
    vol_short_len: int = 5              # 短天期均量
    vol_long_len: int = 19              # 長天期均量

    # 4. 濾網開關 (讓 AI 決定要不要啟動這些條件)
    use_bb: bool = True                 # 是否啟用布林通道突破濾網
    use_kc: bool = True                 # 是否啟用阿肯那通道跌破出場
    use_vol: bool = True                # 是否啟用均量爆發濾網
    
    # 5. 交易成本設定 (台股標準)
    buy_fee: float = 0.000855           # 買入手續費率
    sell_fee: float = 0.000855          # 賣出手續費率
    tax_rate: float = 0.003             # 證交稅率
    min_fee: int = 20                   # 最低手續費 (元)
    
    # 🌟 6. 核心環境開關：複利開關 (預設為 True，讓您的 Portfolio 模擬器能正常滾雪球)
    use_compounding: bool = True
# core/v16_config.py
from dataclasses import dataclass

# ==========================================
# 🌟 全域戰略切換開關 (System-Wide Strategy Switches)
# ==========================================
# 1. 期望值 (EV) 算法切換
# 'A' = 嚴格 R_Multiple 期望值 (Mean R)
# 'B' = 傳統實際盈虧期望值 (Win% * Payoff - Loss%)
EV_CALC_METHOD = 'A'

# 2. 買入優先序切換開關
# 'PROJ_COST' = 優先買入能消耗最多資金的標的 (資金效率極大化)
# 'EV'        = 優先買入期望值最高的標的 (單筆質量極大化)
BUY_SORT_METHOD = 'PROJ_COST'

# 3. 系統評分 (Score) 算法切換
# 'LOG_R2' = 結合對數 R 平方與月度勝率的不對稱模型 (容許暴漲，尋找平穩向上的聖杯)
# 'RoMD'   = 傳統報酬回撤比 (只看總報酬與最大回撤)
SCORE_CALC_METHOD = 'LOG_R2'
# ==========================================

@dataclass
class V16StrategyParams:
    # 1. 核心指標參數
    high_len: int = 201              
    atr_len: int = 14                
    
    # 2. 停損停利與進場風控
    atr_buy_tol: float = 1.5         
    atr_times_init: float = 2.0      
    atr_times_trail: float = 3.5     
    tp_percent: float = 0.5
    min_chase_rr: float = 0.5        # <--- (AI註: 新增：遲到追車的最低盈虧比門檻)          

    # 3. 三大濾網開關與參數
    use_bb: bool = True
    bb_len: int = 20
    bb_mult: float = 2.0
    
    use_kc: bool = False
    kc_len: int = 20
    kc_mult: float = 2.0
    
    use_vol: bool = True
    vol_short_len: int = 5
    vol_long_len: int = 19

    # 4. 資金管理參數
    initial_capital: float = 1000000    
    fixed_risk: float = 0.01            
    buy_fee: float = 0.001425 * 0.28    
    sell_fee: float = 0.001425 * 0.28   
    tax_rate: float = 0.003             
    min_fee: float = 20                 
    use_compounding: bool = True        

    # 5. 歷史績效濾網 (AI 將接管這些設定)
    min_history_trades: int = 0         
    min_history_ev: float = 0.0         
    min_history_win_rate: float = 0.30
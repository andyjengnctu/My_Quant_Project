# 打包
git archive -o test-branch-1.zip test-branch-1

# 重現環境
python -m pip install -r requirements/requirements-lock.txt

# 正式入口（建議使用）
python apps/validate_consistency.py --dataset reduced
python apps/validate_consistency.py --dataset full
python apps/portfolio_sim.py --dataset full
python apps/portfolio_sim.py --dataset reduced
python apps/vip_scanner.py --dataset full
python apps/vip_scanner.py --dataset reduced
python apps/ml_optimizer.py --dataset full
python apps/ml_optimizer.py --dataset reduced
python apps/smart_downloader.py

# 相容舊入口（暫時保留）
python tools/validate_v16_consistency.py --dataset reduced
python tools/validate_v16_consistency.py --dataset full
python v16_portfolio_sim.py --dataset full
python v16_portfolio_sim.py --dataset reduced
python v16_vip_scanner.py --dataset full
python v16_vip_scanner.py --dataset reduced
python v16_ml_optimizer2.py --dataset full
python v16_ml_optimizer2.py --dataset reduced
python vip_smart_downloader.py

# 除錯工具
python tools/debug/trade_log.py
python tools/debug_trade_log.py

# 環境變數切換
# validate 專用
set V16_VALIDATE_DATASET=reduced
# 主工具共用
set V16_DATASET_PROFILE=full

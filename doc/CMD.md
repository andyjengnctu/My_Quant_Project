# 打包
git archive -o test-branch-1.zip test-branch-1

# 重現環境
python -m pip install -r requirements/requirements-lock.txt

# 資料集切換
python tools/validate_v16_consistency.py --dataset reduced
python tools/validate_v16_consistency.py --dataset full
python v16_portfolio_sim.py --dataset full
python v16_portfolio_sim.py --dataset reduced
python v16_vip_scanner.py --dataset full
python v16_vip_scanner.py --dataset reduced
python v16_ml_optimizer2.py --dataset full
python v16_ml_optimizer2.py --dataset reduced

# 環境變數切換
# validate 專用
set V16_VALIDATE_DATASET=reduced
# 主工具共用
set V16_DATASET_PROFILE=full

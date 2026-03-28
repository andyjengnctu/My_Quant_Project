# 打包
git archive -o test-branch-1.zip test-branch-1

# 重現環境
python -m pip install -r requirements/requirements-lock.txt

# 預設資料集
python v16_portfolio_sim.py                  # 預設 full
python v16_vip_scanner.py                    # 預設 full
python v16_ml_optimizer2.py                  # 預設 full
python tools/validate_v16_consistency.py     # 預設 reduced（consistency 仍可在 UI 選）

# 指定資料集（CLI）
python v16_portfolio_sim.py --dataset=reduced
python v16_portfolio_sim.py --dataset=full
python v16_vip_scanner.py --dataset=reduced
python v16_vip_scanner.py --dataset=full
python v16_ml_optimizer2.py --dataset=reduced
python v16_ml_optimizer2.py --dataset=full
python tools/validate_v16_consistency.py --dataset=reduced
python tools/validate_v16_consistency.py --dataset=full

# 指定資料集（PowerShell）
$env:V16_DATASET_PROFILE = 'full'
python v16_portfolio_sim.py
python v16_vip_scanner.py
python v16_ml_optimizer2.py

$env:V16_VALIDATE_DATASET = 'full'
python tools/validate_v16_consistency.py

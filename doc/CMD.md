# 打包
$branch="test-branch-1"; $ts=Get-Date -Format "yyyyMMdd_HHmmss"; $sha=(git rev-parse --short $branch).Trim(); git archive --format=zip -o "${branch}_${ts}_${sha}.zip" $branch

# 重現環境
python -m pip install -r requirements/requirements-lock.txt

# 資料集切換
python apps/validate_consistency.py --dataset reduced
python apps/validate_consistency.py --dataset full
python apps/portfolio_sim.py --dataset full
python apps/portfolio_sim.py --dataset reduced
python apps/vip_scanner.py --dataset full
python apps/vip_scanner.py --dataset reduced
python apps/ml_optimizer.py --dataset full
python apps/ml_optimizer.py --dataset reduced

# 環境變數切換
# validate 專用
set V16_VALIDATE_DATASET=reduced
# 主工具共用
set V16_DATASET_PROFILE=full

# optimizer 架構
python apps/ml_optimizer.py --dataset full            # 正式入口
# apps/ml_optimizer.py 為薄入口；主流程在 tools/optimizer/main.py

# validate 架構
python apps/validate_consistency.py --dataset reduced    # 正式入口
# apps/validate_consistency.py 為薄入口；總控在 tools/validate/main.py
# tools/validate/checks.py / tool_adapters.py / synthetic_cases.py 分別負責共用檢查、工具載入與 synthetic suite


## 資料下載

```bash
python apps/smart_downloader.py
```

下載主流程已移至 `tools/downloader/main.py`；正式入口仍為 `apps/smart_downloader.py`。


## 交易除錯

```bash
python tools/debug/trade_log.py
```

`tools/debug/trade_log.py` 為 debug 正式入口；交易回放主邏輯在 `tools/debug/backtest.py`，輸出摘要在 `tools/debug/reporting.py`。


# portfolio engine 架構
# run_portfolio_timeline() 正式總控仍在 core/v16_portfolio_engine.py
# 快取市場資料/PIT 索引在 core/v16_portfolio_fast_data.py
# 曲線/年度/年化統計與分數在 core/v16_portfolio_stats.py

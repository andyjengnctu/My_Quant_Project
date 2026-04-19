# 升級計畫（Step 0 ~ Step 2）

## Step 0：清除不合目標架構的部分

目標：先把現行 optimizer 主流程收斂成單一路徑，避免 legacy split / promotion / champion 流程持續干擾 Step 1 驗證。

本階段調整：
- 固定 pre-deploy / OOS 驗證區間為 `2010~2019 / 2020~latest`
- 停用 legacy 全資料模式入口
- 停用 final holdout / champion / promotion 主流程
- 將正式輸出收斂為 `run_best + 單一連續 OOS 驗證報表`
- 舊 compare / promotion 函式先保留，但降為 legacy，不列為正式主流程

不在 Step 0 變更：
- 策略公式
- search space
- 停損 / 停利與 sizing 規則
- 參數意義

## Step 1：完成 anti-overfitting 訓練方式（對現有參數最小改動）

目標：不改參數意義，先把選參邏輯改成單一口徑、較抗 overfitting 的方式。

預計原則：
- 不再採用 train / validate 雙口徑作為正式 Step 1 方法
- 在 `2010~2019` 全部 pre-deploy 資料上，用同一口徑選參
- 優先採約束式選參：先過硬條件，再 maximize `total_romd`
- OOS 驗證只看 `2020~latest` 的真實結果
- `plateau_drop_pct` 只作候選複檢，不放主搜尋內圈

## Step 2：全面加入 rolling 驗證歷史績效

目標：在 Step 1 方法確認有效後，再搬到 rolling moving forward 架構，用來評估制度的真實歷史績效。

預計原則：
- 先用 annual rolling forward
- 每輪只用當時以前資料選參
- 下一輪 deploy 才算真 OOS
- 將所有 deploy 區間串成整體歷史績效曲線

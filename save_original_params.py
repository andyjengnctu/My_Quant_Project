import json

# 這是您在 TradingView (上班族嚴格實戰版) 上的原始設定
original_params = {
    "atr_len": 12,
    "atr_times_init": 2.5,
    "atr_times_trail": 4.0,
    "atr_buy_tol": 0.5,
    "high_len": 50,
    "tp_percent": 0.55
}

# 將其存入專屬的備份檔案中
backup_filename = "v16_original_tv_params.json"

with open(backup_filename, "w") as f:
    json.dump(original_params, f, indent=4)

print(f"✅ 成功！您的 TV 原始參數已安全備份至 '{backup_filename}'。")
print(f"💡 日後若想使用這組參數進行掃描，只需將其改名為 'v16_best_params.json' 覆蓋即可！")
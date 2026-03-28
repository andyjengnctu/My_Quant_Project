import os

import pandas as pd


def build_synthetic_baseline_frame(start_date, periods, base_price=100.0, volume=1000):
    dates = pd.bdate_range(start_date, periods=periods)
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": [base_price - 0.5] * periods,
        "High": [base_price + 0.5] * periods,
        "Low": [base_price - 1.0] * periods,
        "Close": [base_price] * periods,
        "Volume": [volume] * periods,
    })



def set_synthetic_bar(df, idx, *, open_price, high_price, low_price, close_price, volume=1000):
    df.loc[idx, ["Open", "High", "Low", "Close", "Volume"]] = [
        float(open_price),
        float(high_price),
        float(low_price),
        float(close_price),
        float(volume),
    ]



def write_synthetic_csv_bundle(temp_dir, frames_by_ticker):
    for ticker, frame in frames_by_ticker.items():
        frame.to_csv(os.path.join(temp_dir, f"{ticker}.csv"), index=False)

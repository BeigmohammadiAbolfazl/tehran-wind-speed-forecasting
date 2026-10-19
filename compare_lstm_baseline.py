
import os
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = r"C:\Users\user\Desktop\wind"

NC_FILE = os.path.join(
    BASE_DIR,
    "reanalysis-era5-single-levels-timeseries-sfcq1tz7sl1.nc"
)
PRED_FILE = os.path.join(
    BASE_DIR,
    "lstm_test_predictions.csv"
)

OUT_FILE = os.path.join(
    BASE_DIR,
    "lstm_vs_persistence_results.csv"
)
LOCATION_FILE = os.path.join(
    BASE_DIR,
    "lstm_vs_persistence_by_location.csv"
)

# 1. بارگذاری داده‌های آزمون و ERA5
results = pd.read_csv(PRED_FILE)
results["time"] = pd.to_datetime(results["time"])

ds = xr.open_dataset(NC_FILE)
wind = np.hypot(ds["u100"], ds["v100"]).transpose(
    "valid_time", "latitude", "longitude"
)

times = pd.to_datetime(ds["valid_time"].values)
time_to_idx = {t: i for i, t in enumerate(times)}

# 2. پیش‌بینی Persistence:
# سرعت باد ساعت قبل = پیش‌بینی سرعت باد ساعت آینده
baseline_values = []

for row in results.itertuples(index=False):
    t = pd.Timestamp(row.time)

    if t not in time_to_idx:
        raise ValueError(f"Time not found in ERA5: {t}")

    idx = time_to_idx[t]

    if idx == 0:
        raise ValueError("Previous hour is unavailable.")

    point = wind.sel(
        latitude=float(row.latitude),
        longitude=float(row.longitude),
        method="nearest"
    )

    previous_speed = float(point.values[idx - 1])
    baseline_values.append(previous_speed)

results["persistence_prediction"] = baseline_values

# 3. معیارهای مقایسه روی همان نمونه‌های آزمون
actual = results["actual_wind_100m"].to_numpy()
lstm_pred = results["predicted_wind_100m"].to_numpy()
pers_pred = results["persistence_prediction"].to_numpy()

def calculate_metrics(y_true, y_pred):
    return {
        "MAE_m_s": mean_absolute_error(y_true, y_pred),
        "RMSE_m_s": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

overall = pd.DataFrame([
    {"Model": "Persistence", **calculate_metrics(actual, pers_pred)},
    {"Model": "LSTM", **calculate_metrics(actual, lstm_pred)}
])

print("\n===== OVERALL COMPARISON =====")
print(overall.to_string(index=False))

# درصد کاهش خطا؛ مقدار مثبت یعنی LSTM بهتر است
p_mae = mean_absolute_error(actual, pers_pred)
l_mae = mean_absolute_error(actual, lstm_pred)
p_rmse = np.sqrt(mean_squared_error(actual, pers_pred))
l_rmse = np.sqrt(mean_squared_error(actual, lstm_pred))

print(f"\nMAE reduction by LSTM: {(p_mae-l_mae)/p_mae*100:.2f}%")
print(f"RMSE reduction by LSTM: {(p_rmse-l_rmse)/p_rmse*100:.2f}%")

# 4. مقایسه جداگانه هر نقطه
location_rows = []

for (lat, lon), group in results.groupby(["latitude", "longitude"]):
    y = group["actual_wind_100m"].to_numpy()
    lp = group["predicted_wind_100m"].to_numpy()
    pp = group["persistence_prediction"].to_numpy()

    lm = calculate_metrics(y, lp)
    pm = calculate_metrics(y, pp)

    location_rows.append({
        "latitude": lat,
        "longitude": lon,
        "LSTM_MAE": lm["MAE_m_s"],
        "Persistence_MAE": pm["MAE_m_s"],
        "LSTM_RMSE": lm["RMSE_m_s"],
        "Persistence_RMSE": pm["RMSE_m_s"],
        "LSTM_R2": lm["R2"],
        "Persistence_R2": pm["R2"]
    })

location_df = pd.DataFrame(location_rows)
location_df.to_csv(LOCATION_FILE, index=False)
results.to_csv(OUT_FILE, index=False)

print("\n===== LOCATION COMPARISON =====")
print(location_df.round(4).to_string(index=False))

print("\nSaved:", OUT_FILE)
print("Saved:", LOCATION_FILE)

ds.close()

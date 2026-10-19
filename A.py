
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==================================================
# 1. مسیر فایل‌ها
# ==================================================
BASE_DIR = r"C:\Users\user\Desktop\wind"

NC_FILE = os.path.join(
    BASE_DIR,
    "reanalysis-era5-single-levels-timeseries-sfcq1tz7sl1.nc"
)

TOP20_FILE = os.path.join(
    BASE_DIR,
    "era5_top20_wind_locations.csv"
)

# خروجی‌ها
RESULTS_FILE = os.path.join(BASE_DIR, "lstm_test_predictions.csv")
METRICS_FILE = os.path.join(BASE_DIR, "lstm_metrics_by_location.csv")
MODEL_FILE = os.path.join(BASE_DIR, "era5_lstm_model.keras")
PLOT_FILE = os.path.join(BASE_DIR, "lstm_prediction_example.png")

np.random.seed(42)
tf.random.set_seed(42)

# ==================================================
# 2. بارگذاری داده‌ها
# ==================================================
if not os.path.exists(NC_FILE):
    raise FileNotFoundError(f"NetCDF file not found: {NC_FILE}")

if not os.path.exists(TOP20_FILE):
    raise FileNotFoundError(f"Top 20 CSV not found: {TOP20_FILE}")

ds = xr.open_dataset(NC_FILE)

top20 = pd.read_csv(TOP20_FILE)
top20.columns = top20.columns.str.strip()

required_columns = {"latitude", "longitude"}
if not required_columns.issubset(top20.columns):
    raise ValueError(
        "Top-20 CSV must contain latitude and longitude columns."
    )

# ==================================================
# 3. محاسبه سرعت باد در ارتفاع 100 متر
# ==================================================
wind_speed = np.hypot(ds["u100"], ds["v100"])

# اطمینان از ترتیب ابعاد
wind_speed = wind_speed.transpose(
    "valid_time", "latitude", "longitude"
)

times = pd.to_datetime(ds["valid_time"].values)
n_times = len(times)

# 70% آموزش، 15% اعتبارسنجی، 15% آزمون
train_end = int(n_times * 0.70)
val_end = int(n_times * 0.85)

if n_times <= 24:
    raise ValueError("Not enough hourly observations.")

print("Total hourly observations:", n_times)
print("Training period:", times[0], "to", times[train_end - 1])
print("Validation period:", times[train_end], "to", times[val_end - 1])
print("Test period:", times[val_end], "to", times[-1])

# ==================================================
# 4. استخراج سری زمانی 20 نقطه
# ==================================================
series_list = []
location_records = []

for row in top20.itertuples(index=False):
    lat = float(row.latitude)
    lon = float(row.longitude)

    # نزدیک‌ترین مختصات موجود در شبکه ERA5
    point = wind_speed.sel(
        latitude=lat,
        longitude=lon,
        method="nearest"
    )

    actual_lat = float(point["latitude"].values)
    actual_lon = float(point["longitude"].values)

    values = np.asarray(point.values, dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Missing wind data at {actual_lat}, {actual_lon}"
        )

    series_list.append(values)

    location_records.append({
        "latitude": actual_lat,
        "longitude": actual_lon
    })

wind_matrix = np.stack(series_list, axis=1)
n_locations = wind_matrix.shape[1]

print("Wind matrix shape:", wind_matrix.shape)
print("Number of locations:", n_locations)

# ==================================================
# 5. نرمال‌سازی بدون نشت داده از آینده
# ==================================================
# فقط داده‌های دوره آموزش برای مقیاس سرعت استفاده می‌شوند.
train_values = wind_matrix[:train_end, :]

wind_min = float(train_values.min())
wind_max = float(train_values.max())

if wind_max <= wind_min:
    raise ValueError("Wind speed has no variation.")

wind_scaled = (wind_matrix - wind_min) / (wind_max - wind_min)

# مختصات مکانی به عنوان ویژگی ثابت هر نقطه
latitudes = np.array(
    [r["latitude"] for r in location_records],
    dtype=np.float32
)

longitudes = np.array(
    [r["longitude"] for r in location_records],
    dtype=np.float32
)

lat_min, lat_max = latitudes.min(), latitudes.max()
lon_min, lon_max = longitudes.min(), longitudes.max()

lat_scaled = (latitudes - lat_min) / max(lat_max - lat_min, 1e-6)
lon_scaled = (longitudes - lon_min) / max(lon_max - lon_min, 1e-6)

# ==================================================
# 6. ساخت نمونه‌های 24 ساعت گذشته -> ساعت آینده
# ==================================================
LOOKBACK = 24

X_list = []
y_list = []
target_time_indices = []
location_indices = []

for loc in range(n_locations):
    for target_idx in range(LOOKBACK, n_times):
        # سه ویژگی در هر ساعت:
        # سرعت باد نرمال‌شده، عرض جغرافیایی، طول جغرافیایی
        past_wind = wind_scaled[
            target_idx - LOOKBACK:target_idx, loc
        ]

        lat_feature = np.full(LOOKBACK, lat_scaled[loc])
        lon_feature = np.full(LOOKBACK, lon_scaled[loc])

        sequence = np.column_stack([
            past_wind,
            lat_feature,
            lon_feature
        ])

        X_list.append(sequence)
        y_list.append(wind_scaled[target_idx, loc])
        target_time_indices.append(target_idx)
        location_indices.append(loc)

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.float32)
target_time_indices = np.asarray(target_time_indices)
location_indices = np.asarray(location_indices)

# تقسیم بر اساس زمان هدف، نه تقسیم تصادفی
train_mask = target_time_indices < train_end
val_mask = (
    (target_time_indices >= train_end) &
    (target_time_indices < val_end)
)
test_mask = target_time_indices >= val_end

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

test_times_idx = target_time_indices[test_mask]
test_locations = location_indices[test_mask]

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))
print("Input shape:", X_train.shape)

# ==================================================
# 7. تعریف مدل LSTM
# ==================================================
model = Sequential([
    Input(shape=(LOOKBACK, 3)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ==================================================
# 8. آموزش مدل
# ==================================================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,
    callbacks=[early_stop],
    shuffle=True,
    verbose=1
)

model.save(MODEL_FILE)
print("\nSaved model:", MODEL_FILE)

# ==================================================
# 9. پیش‌بینی روی داده‌های آزمون
# ==================================================
pred_scaled = model.predict(X_test, verbose=0).reshape(-1)

# بازگرداندن مقیاس به متر بر ثانیه
y_test_real = y_test * (wind_max - wind_min) + wind_min
pred_real = pred_scaled * (wind_max - wind_min) + wind_min

# سرعت باد منفی از نظر فیزیکی معنی ندارد
pred_real = np.maximum(pred_real, 0)

mae = mean_absolute_error(y_test_real, pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, pred_real))
r2 = r2_score(y_test_real, pred_real)

print("\n========== OVERALL TEST RESULTS ==========")
print(f"MAE  : {mae:.4f} m/s")
print(f"RMSE : {rmse:.4f} m/s")
print(f"R2   : {r2:.4f}")

# ==================================================
# 10. ذخیره پیش‌بینی‌ها
# ==================================================
results = pd.DataFrame({
    "time": times[test_times_idx],
    "latitude": latitudes[test_locations],
    "longitude": longitudes[test_locations],
    "actual_wind_100m": y_test_real,
    "predicted_wind_100m": pred_real
})

results["absolute_error"] = np.abs(
    results["actual_wind_100m"] -
    results["predicted_wind_100m"]
)

results.to_csv(RESULTS_FILE, index=False)

# معیارهای جداگانه برای هر نقطه
metric_rows = []

for loc in range(n_locations):
    mask = test_locations == loc

    if mask.sum() == 0:
        continue

    actual = y_test_real[mask]
    predicted = pred_real[mask]

    metric_rows.append({
        "latitude": latitudes[loc],
        "longitude": longitudes[loc],
        "test_samples": int(mask.sum()),
        "MAE_m_s": mean_absolute_error(actual, predicted),
        "RMSE_m_s": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    })

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(METRICS_FILE, index=False)

print("\nSaved predictions:", RESULTS_FILE)
print("Saved per-location metrics:", METRICS_FILE)

print("\nMetrics by location:")
print(metrics_df.to_string(index=False))

# ==================================================
# 11. نمودار نمونه: نقطه اول از فهرست 20 نقطه
# ==================================================
selected_loc = 0
plot_mask = test_locations == selected_loc

plot_df = results.loc[plot_mask].sort_values("time")

plt.figure(figsize=(14, 6))
plt.plot(
    plot_df["time"],
    plot_df["actual_wind_100m"],
    label="Actual wind speed",
    linewidth=1.5
)
plt.plot(
    plot_df["time"],
    plot_df["predicted_wind_100m"],
    label="LSTM prediction",
    linewidth=1.5
)

plt.xlabel("Time")
plt.ylabel("Wind speed at 100 m (m/s)")
plt.title(
    f"LSTM Wind Speed Prediction "
    f"({latitudes[selected_loc]:.2f}, "
    f"{longitudes[selected_loc]:.2f})"
)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
plt.show()

print("\nSaved plot:", PLOT_FILE)

# ==================================================
# 12. نمودار روند آموزش
# ==================================================
plt.figure(figsize=(9, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("LSTM Training History")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "lstm_training_history.png"),
    dpi=300
)
plt.show()

ds.close()
print("\nLSTM training and evaluation completed.")


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

BASE = r"C:\Users\user\Desktop\wind"
NC_FILE = os.path.join(
    BASE,
    "reanalysis-era5-single-levels-timeseries-sfcq1tz7sl1.nc"
)
POINTS_FILE = os.path.join(
    BASE, "tehran_nearby_grid_points.csv"
)

np.random.seed(42)
tf.random.set_seed(42)

# 1. بارگذاری مختصات چهار نقطه
points = pd.read_csv(POINTS_FILE)

# 2. محاسبه سرعت باد در ارتفاع 100 متر
ds = xr.open_dataset(NC_FILE)
speed = np.hypot(ds["u100"], ds["v100"]).transpose(
    "valid_time", "latitude", "longitude"
)
times = pd.to_datetime(ds["valid_time"].values)
n_time = len(times)

# تقسیم زمانی: 70% آموزش، 15% اعتبارسنجی، 15% آزمون
train_end = int(n_time * 0.70)
val_end = int(n_time * 0.85)
lookback = 24

series = []
coords = []

for row in points.itertuples(index=False):
    point = speed.sel(
        latitude=float(row.latitude),
        longitude=float(row.longitude),
        method="nearest"
    )

    series.append(np.asarray(point.values, dtype=np.float32))
    coords.append((
        float(point["latitude"].values),
        float(point["longitude"].values)
    ))

wind = np.stack(series, axis=1)
n_points = wind.shape[1]

# مقیاس‌بندی فقط بر اساس داده‌های آموزش
vmin = float(wind[:train_end].min())
vmax = float(wind[:train_end].max())
if vmax <= vmin:
    raise ValueError("Invalid wind speed range.")

scaled = (wind - vmin) / (vmax - vmin)

lats = np.array([c[0] for c in coords])
lons = np.array([c[1] for c in coords])

lat_min, lat_max = lats.min(), lats.max()
lon_min, lon_max = lons.min(), lons.max()

lat_s = (lats - lat_min) / max(lat_max - lat_min, 1e-6)
lon_s = (lons - lon_min) / max(lon_max - lon_min, 1e-6)

# 3. ساخت نمونه‌ها: 24 ساعت قبل -> سرعت ساعت بعد
X_list, y_list, target_idx, loc_idx = [], [], [], []

for loc in range(n_points):
    for t in range(lookback, n_time):
        seq = np.column_stack([
            scaled[t-lookback:t, loc],
            np.full(lookback, lat_s[loc]),
            np.full(lookback, lon_s[loc])
        ])
        X_list.append(seq)
        y_list.append(scaled[t, loc])
        target_idx.append(t)
        loc_idx.append(loc)

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.float32)
target_idx = np.asarray(target_idx)
loc_idx = np.asarray(loc_idx)

train_mask = target_idx < train_end
val_mask = (target_idx >= train_end) & (target_idx < val_end)
test_mask = target_idx >= val_end

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

test_t = target_idx[test_mask]
test_loc = loc_idx[test_mask]

# 4. تعریف و آموزش LSTM
model = Sequential([
    Input(shape=(lookback, 3)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    shuffle=True,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# 5. پیش‌بینی روی آزمون
pred_scaled = model.predict(X_test, verbose=0).ravel()

actual = y_test * (vmax - vmin) + vmin
pred = np.maximum(pred_scaled * (vmax - vmin) + vmin, 0)

# Persistence: سرعت ساعت قبل برای پیش‌بینی ساعت جاری
baseline = np.array([
    wind[t - 1, loc] for t, loc in zip(test_t, test_loc)
])

def metrics(y_true, y_pred):
    return {
        "MAE_m_s": mean_absolute_error(y_true, y_pred),
        "RMSE_m_s": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

print("\n===== TEHRAN-NEARBY OVERALL RESULTS =====")
print("LSTM:", metrics(actual, pred))
print("Persistence:", metrics(actual, baseline))

for name, estimate in [
    ("LSTM", pred),
    ("Persistence", baseline)
]:
    pass

p_mae = mean_absolute_error(actual, baseline)
l_mae = mean_absolute_error(actual, pred)
p_rmse = np.sqrt(mean_squared_error(actual, baseline))
l_rmse = np.sqrt(mean_squared_error(actual, pred))

print(f"MAE reduction: {(p_mae-l_mae)/p_mae*100:.2f}%")
print(f"RMSE reduction: {(p_rmse-l_rmse)/p_rmse*100:.2f}%")

# 6. نتایج هر نقطه
rows = []
for loc, (lat, lon) in enumerate(coords):
    mask = test_loc == loc
    rows.append({
        "latitude": lat,
        "longitude": lon,
        "mean_wind_100m": float(wind[:, loc].mean()),
        "LSTM_MAE": mean_absolute_error(actual[mask], pred[mask]),
        "LSTM_RMSE": np.sqrt(
            mean_squared_error(actual[mask], pred[mask])
        ),
        "LSTM_R2": r2_score(actual[mask], pred[mask]),
        "Persistence_MAE": mean_absolute_error(
            actual[mask], baseline[mask]
        ),
        "Persistence_RMSE": np.sqrt(
            mean_squared_error(actual[mask], baseline[mask])
        )
    })

result = pd.DataFrame(rows)
out = os.path.join(BASE, "tehran_lstm_metrics.csv")
result.to_csv(out, index=False)

# 7. نمودار واقعی در برابر پیش‌بینی برای نزدیک‌ترین نقطه به مرکز تهران
nearest_loc = int(np.argmin([
    (lat - 35.6892)**2 + (lon - 51.3890)**2
    for lat, lon in coords
]))
mask = test_loc == nearest_loc
plot_times = times[test_t[mask]]

plt.figure(figsize=(13, 5))
plt.plot(plot_times, actual[mask], label="Actual")
plt.plot(plot_times, pred[mask], label="LSTM")
plt.plot(plot_times, baseline[mask], label="Persistence", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Wind speed at 100 m (m/s)")
plt.title(f"Wind Forecast Near Tehran: {coords[nearest_loc]}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(BASE, "tehran_lstm_comparison.png"),
    dpi=300
)
plt.show()

print("\n===== RESULTS BY POINT =====")
print(result.round(4).to_string(index=False))
print("\nSaved:", out)

ds.close()

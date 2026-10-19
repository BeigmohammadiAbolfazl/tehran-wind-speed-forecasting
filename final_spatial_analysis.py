
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\user\Desktop\wind"

TOP20_FILE = os.path.join(
    BASE_DIR, "era5_top20_wind_locations.csv"
)
METRICS_FILE = os.path.join(
    BASE_DIR, "lstm_vs_persistence_by_location.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR, "final_site_analysis.csv"
)

# 1. خواندن اطلاعات سرعت باد و معیارهای مدل
wind = pd.read_csv(TOP20_FILE)
metrics = pd.read_csv(METRICS_FILE)

wind.columns = wind.columns.str.strip()
metrics.columns = metrics.columns.str.strip()

# 2. ترکیب اطلاعات بر اساس مختصات
df = wind.merge(
    metrics,
    on=["latitude", "longitude"],
    how="left",
    validate="one_to_one"
)

if df["LSTM_MAE"].isna().any():
    raise ValueError(
        "Some locations did not match. Check latitude/longitude values."
    )

# 3. محاسبه میزان بهبود LSTM نسبت به Persistence
df["MAE_improvement_percent"] = (
    (df["Persistence_MAE"] - df["LSTM_MAE"])
    / df["Persistence_MAE"] * 100
)

df["RMSE_improvement_percent"] = (
    (df["Persistence_RMSE"] - df["LSTM_RMSE"])
    / df["Persistence_RMSE"] * 100
)

# رتبه‌بندی بر اساس میانگین باد، نه دقت مدل
df = df.sort_values("mean_wind_100m", ascending=False)
df["wind_rank"] = np.arange(1, len(df) + 1)

df.to_csv(OUTPUT_CSV, index=False)

print("\n===== FINAL SITE ANALYSIS =====")
cols = [
    "wind_rank", "latitude", "longitude",
    "mean_wind_100m", "p90_wind_100m",
    "LSTM_MAE", "LSTM_RMSE", "LSTM_R2",
    "MAE_improvement_percent"
]
print(df[cols].round(3).to_string(index=False))
print("\nSaved:", OUTPUT_CSV)

# 4. نمودار میانگین سرعت باد در 20 نقطه
plot_df = df.sort_values("mean_wind_100m")

labels = [
    f"{r.latitude:.2f}, {r.longitude:.2f}"
    for r in plot_df.itertuples()
]

plt.figure(figsize=(10, 9))
plt.barh(labels, plot_df["mean_wind_100m"])
plt.xlabel("Mean wind speed at 100 m (m/s)")
plt.ylabel("Latitude, Longitude")
plt.title("Mean Wind Speed at the 20 Selected ERA5 Grid Points")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "final_mean_wind_top20.png"),
    dpi=300
)
plt.show()

# 5. نقشه پراکندگی نقاط؛ رنگ نشان‌دهنده میانگین سرعت باد است
plt.figure(figsize=(10, 7))
sc = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["mean_wind_100m"],
    s=100,
    edgecolors="black"
)

for r in df.itertuples():
    plt.annotate(
        str(int(r.wind_rank)),
        (r.longitude, r.latitude),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

plt.colorbar(sc, label="Mean wind speed at 100 m (m/s)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of the 20 Selected ERA5 Grid Points")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "final_wind_spatial_map.png"),
    dpi=300
)
plt.show()

# 6. نقشه پراکندگی خطای MAE مدل LSTM
plt.figure(figsize=(10, 7))
sc = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["LSTM_MAE"],
    s=100,
    edgecolors="black"
)

for r in df.itertuples():
    plt.annotate(
        str(int(r.wind_rank)),
        (r.longitude, r.latitude),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

plt.colorbar(sc, label="LSTM MAE (m/s)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of LSTM Prediction Error")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "final_lstm_error_map.png"),
    dpi=300
)
plt.show()

print("\nAll spatial analysis figures were saved.")

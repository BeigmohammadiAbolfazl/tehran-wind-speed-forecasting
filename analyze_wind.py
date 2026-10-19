
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# =========================================
# 1. File paths
# =========================================
folder = r"C:\Users\user\Desktop\wind"

nc_file = os.path.join(
    folder,
    "reanalysis-era5-single-levels-timeseries-sfcq1tz7sl1.nc"
)

all_csv = os.path.join(folder, "era5_wind_all_locations.csv")
top_csv = os.path.join(folder, "era5_top20_wind_locations.csv")
map_file = os.path.join(folder, "era5_top20_wind_map.png")

if not os.path.exists(nc_file):
    raise FileNotFoundError(f"File not found: {nc_file}")

# =========================================
# 2. Open dataset
# =========================================
ds = xr.open_dataset(nc_file)

# Wind speed magnitude at 100m and 10m
speed100 = np.hypot(ds["u100"], ds["v100"])
speed10 = np.hypot(ds["u10"], ds["v10"])

# =========================================
# 3. Calculate statistics at each grid cell
# =========================================
mean100 = speed100.mean(dim="valid_time")
p90_100 = speed100.quantile(0.90, dim="valid_time")
std100 = speed100.std(dim="valid_time")

mean10 = speed10.mean(dim="valid_time")

df = pd.DataFrame({
    "latitude": mean100["latitude"].values.repeat(
        mean100.sizes["longitude"]
    ),
    "longitude": np.tile(
        mean100["longitude"].values,
        mean100.sizes["latitude"]
    ),
    "mean_wind_100m": mean100.values.flatten(),
    "p90_wind_100m": p90_100.values.flatten(),
    "std_wind_100m": std100.values.flatten(),
    "mean_wind_10m": mean10.values.flatten()
})

# =========================================
# 4. Rank locations by mean wind speed
# =========================================
df = df.sort_values(
    "mean_wind_100m",
    ascending=False
).reset_index(drop=True)

df.insert(0, "rank", np.arange(1, len(df) + 1))

# Save all 81 grid points
df.to_csv(all_csv, index=False, encoding="utf-8-sig")

# Save top 20 points
top20 = df.head(20).copy()
top20.to_csv(top_csv, index=False, encoding="utf-8-sig")

# =========================================
# 5. Print results
# =========================================
print("\n========== DATA SUMMARY ==========")
print("Number of grid points:", len(df))
print("Period:", str(ds["valid_time"].values[0]),
      "to", str(ds["valid_time"].values[-1]))

print("\n========== TOP 20 LOCATIONS ==========")
print(
    top20[
        [
            "rank",
            "latitude",
            "longitude",
            "mean_wind_100m",
            "p90_wind_100m",
            "mean_wind_10m"
        ]
    ].round(3).to_string(index=False)
)

print("\n========== OVERALL STATISTICS ==========")
print("Highest mean wind speed:",
      round(df["mean_wind_100m"].max(), 3), "m/s")
print("Lowest mean wind speed:",
      round(df["mean_wind_100m"].min(), 3), "m/s")
print("Average across grid:",
      round(df["mean_wind_100m"].mean(), 3), "m/s")

# =========================================
# 6. Plot all points and highlight top 20
# =========================================
plt.figure(figsize=(10, 8))

sc = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["mean_wind_100m"],
    cmap="viridis",
    s=90,
    edgecolors="black",
    linewidths=0.3
)

plt.scatter(
    top20["longitude"],
    top20["latitude"],
    facecolors="none",
    edgecolors="red",
    s=180,
    linewidths=1.5,
    label="Top 20 grid cells"
)

for _, row in top20.iterrows():
    plt.annotate(
        str(int(row["rank"])),
        (row["longitude"], row["latitude"]),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=8
    )

plt.colorbar(sc, label="Mean wind speed at 100 m (m/s)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("ERA5 Wind Resource Ranking - Tehran Region")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(map_file, dpi=300)
plt.show()

ds.close()

print("\n========== OUTPUT FILES ==========")
print(all_csv)
print(top_csv)
print(map_file)

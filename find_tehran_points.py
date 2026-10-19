
import os
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\user\Desktop\wind"
FILE = os.path.join(BASE_DIR, "era5_wind_all_locations.csv")

# مختصات تقریبی مرکز تهران
TEHRAN_LAT = 35.6892
TEHRAN_LON = 51.3890

df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()

# محاسبه فاصله جغرافیایی با فرمول هاورساین
R = 6371.0

lat1 = np.radians(TEHRAN_LAT)
lat2 = np.radians(df["latitude"].to_numpy())
dlat = lat2 - lat1

dlon = np.radians(
    df["longitude"].to_numpy() - TEHRAN_LON
)

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2)
    * np.sin(dlon / 2) ** 2
)

df["distance_km"] = (
    2 * R * np.arcsin(np.sqrt(a))
)

# 30 کیلومتر فقط یک شعاع اولیه برای بررسی است،
# نه مرز رسمی شهر تهران
near = df[df["distance_km"] <= 30].copy()
near = near.sort_values("distance_km")

columns = [
    "latitude",
    "longitude",
    "distance_km",
    "mean_wind_100m",
    "p90_wind_100m",
    "mean_wind_10m"
]

print("\n===== GRID POINTS NEAR TEHRAN =====")
print(near[columns].round(3).to_string(index=False))
print("\nNumber of nearby grid points:", len(near))

out_file = os.path.join(
    BASE_DIR, "tehran_nearby_grid_points.csv"
)
near[columns].to_csv(out_file, index=False)

print("\nSaved:", out_file)

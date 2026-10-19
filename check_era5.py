
import os
import xarray as xr

folder = r"C:\Users\user\Desktop\wind"
filename = "reanalysis-era5-single-levels-timeseries-sfcq1tz7sl1.nc"
file_path = os.path.join(folder, filename)

if not os.path.exists(file_path):
    print("File not found:", file_path)
    print("\nNetCDF files in folder:")
    for f in os.listdir(folder):
        if f.endswith(".nc"):
            print(f)
    raise SystemExit

ds = xr.open_dataset(file_path)

print("\n========== DATASET INFO ==========")
print(ds)

print("\n========== VARIABLES ==========")
for name in ds.data_vars:
    print(name, ds[name].dims, ds[name].shape)

print("\n========== COORDINATES ==========")
for name in ds.coords:
    print(name, ds[name].values if ds[name].size < 10
          else ds[name].shape)

print("\n========== TIME RANGE ==========")
for name in ds.coords:
    if "time" in name.lower() or "valid_time" in name.lower():
        print(name, ds[name].values.min(), "to", ds[name].values.max())

ds.close()

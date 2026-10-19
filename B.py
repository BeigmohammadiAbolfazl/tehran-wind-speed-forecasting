
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\user\Desktop\wind\tehran_wind_pixels.csv"
df = pd.read_csv(file_path)

# محدوده تقریبی مرکز شهر تهران؛ فقط برای مقایسه اولیه
city = df[
    df["longitude"].between(51.0, 51.8) &
    df["latitude"].between(35.5, 36.0)
].copy()

# نقاط بیرون از محدوده تقریبی شهر
surroundings = df.drop(city.index)

print("City pixels:", len(city))
print("Surrounding pixels:", len(surroundings))

print("\nTop points in approximate city area:")
print(
    city.nlargest(10, "wind_speed")[
        ["longitude", "latitude", "wind_speed"]
    ].to_string(index=False)
)

plt.figure(figsize=(10, 8))

sc = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["wind_speed"],
    cmap="turbo",
    s=45
)

plt.colorbar(sc, label="Wind speed (verify units)")
plt.scatter(
    [51.389],
    [35.689],
    marker="*",
    s=200,
    color="black",
    label="Approximate Tehran center"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Wind Speed Distribution - Tehran Region")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

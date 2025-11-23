import pandas as pd

df = pd.read_csv("data/raw/uber_dataset.csv")

# Remove missing values
df = df.dropna()

# Remove invalid coords
df = df[
    (df["pickup_lat"].between(-90, 90)) &
    (df["pickup_lng"].between(-180, 180)) &
    (df["dropoff_lat"].between(-90, 90)) &
    (df["dropoff_lng"].between(-180, 180))
]

df.to_csv("data/processed/cleaned_dataset.csv", index=False)

print("Cleaning complete!")

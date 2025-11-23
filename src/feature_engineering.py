import pandas as pd
import numpy as np
from math import radians, sin, cos, atan2, sqrt

# Distance function (Haversine Formula)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in KM
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_dataset.csv")

# Distance feature
df["distance_km"] = df.apply(
    lambda row: haversine(
        row["pickup_lat"], row["pickup_lng"],
        row["dropoff_lat"], row["dropoff_lng"]
    ),
    axis=1
)

# Time features
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["hour"] = df["pickup_datetime"].dt.hour
df["dayofweek"] = df["pickup_datetime"].dt.weekday
df["is_weekend"] = df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
df["is_night"] = df["hour"].apply(lambda x: 1 if x < 5 or x >= 23 else 0)
df["is_peak"] = df["hour"].apply(lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 20) else 0)

# Save processed features
df.to_csv("data/processed/feature_dataset.csv", index=False)

print("Feature engineering complete!")

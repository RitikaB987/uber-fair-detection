# src/train_eta_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import argparse

def main(input_path, output_model):
    # Load features: prefer processed feature file if available
    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
    else:
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Basic check
    print("Rows:", len(df))
    # Use distance_km, hour, is_peak as predictors if present, otherwise compute simple substitutes
    required = ["distance_km","hour","is_peak"]
    if not all(col in df.columns for col in required):
        # try fallback: compute hour/distance quickly if pickup_datetime and coords exist
        if "pickup_datetime" in df.columns:
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
            df["hour"] = df["pickup_datetime"].dt.hour
        # if distance not present, try to compute if coords exist (light compute)
        if "distance_km" not in df.columns and all(c in df.columns for c in ["pickup_lat","pickup_lng","dropoff_lat","dropoff_lng"]):
            from math import radians, sin, cos, atan2, sqrt
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371.0
                lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                return R * c
            df["distance_km"] = df.apply(lambda r: haversine(r["pickup_lat"], r["pickup_lng"], r["dropoff_lat"], r["dropoff_lng"]), axis=1)
        # create is_peak fallback
        if "is_peak" not in df.columns and "hour" in df.columns:
            df["is_peak"] = df["hour"].apply(lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 20) else 0)

    # Define X,y for ETA: if actual eta exists use it as label, else create proxy label
    if "eta_minutes" in df.columns:
        y = df["eta_minutes"]
    else:
        # fallback pseudo-label (rough): distance_km / avg_speed + small noise (avg_speed ~ 30 km/h)
        y = (df["distance_km"] / 30.0) * 60.0

    X = df[["distance_km","hour","is_peak"]].fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(model, output_model)
    print("ETA model training complete. Model saved to:", output_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/feature_dataset.csv", help="path to feature csv")
    parser.add_argument("--output", default="models/eta_model.joblib", help="output model path")
    args = parser.parse_args()
    main(args.input, args.output)

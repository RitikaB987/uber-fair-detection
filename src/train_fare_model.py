# src/train_fare_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib, os, argparse

def main(input_path, eta_model_path, output_model):
    df = pd.read_csv(input_path)
    # Ensure basic features present
    if "distance_km" not in df.columns:
        raise ValueError("distance_km missing - run feature_engineering first")
    # If ETA model exists, predict ETA and use as feature
    if eta_model_path and os.path.exists(eta_model_path):
        eta = joblib.load(eta_model_path)
        X_eta = df[["distance_km","hour","is_peak"]].fillna(0)
        df["eta_minutes_pred"] = eta.predict(X_eta)
    else:
        df["eta_minutes_pred"] = df.get("eta_minutes", (df["distance_km"]/30.0)*60.0)

    features = ["distance_km","hour","dayofweek","is_peak","is_night","is_weekend","passenger_count","eta_minutes_pred"]
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df["fare_amount"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train,y_train)
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(model, output_model)
    print("Fare model trained and saved to:", output_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/feature_dataset.csv")
    parser.add_argument("--eta_model", default="models/eta_model.joblib")
    parser.add_argument("--output", default="models/fare_model.joblib")
    args = parser.parse_args()
    main(args.input, args.eta_model, args.output)

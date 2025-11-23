# src/train_surge_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os, argparse

def main(input_path, output_model):
    df = pd.read_csv(input_path)
    # Basic features (ensure these columns exist)
    if "distance_km" not in df.columns:
        # fallback quick distance calc if needed (but ideally run feature_engineering first)
        pass
    # Use example features (tweak if you have others)
    features = ["distance_km","hour","is_peak","is_night","demand_score","supply_score"]
    X = df[[c for c in features if c in df.columns]].fillna(0)
    # target: surge_label or surge_multiplier>1.0
    if "surge_label" in df.columns:
        y = df["surge_label"]
    else:
        y = (df.get("surge_multiplier", 1.0) > 1.0).astype(int)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train,y_train)
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(model, output_model)
    print("Surge model trained and saved to:", output_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/feature_dataset.csv")
    parser.add_argument("--output", default="models/surge_model.joblib")
    args = parser.parse_args()
    main(args.input, args.output)

"""
SMART AGRI AI - Fertilizer Recommendation Model Training Pipeline
Trains a scikit-learn pipeline (preprocessing + classifier) on soil & nutrient datasets.
"""

import os
import sys
import json
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import FERTILIZER_DATA_PATH, FERTILIZER_MODEL_PATH, FERTILIZER_METRICS_PATH

def train():
    print("=" * 65)
    print("  SMART AGRI AI - FERTILIZER MODEL TRAINING PIPELINE")
    print("=" * 65)

    if not FERTILIZER_DATA_PATH.exists():
        print(f"[Info] Fertilizer dataset not found at: {FERTILIZER_DATA_PATH}")
        print("Please place 'fertilizer_recommendation.csv' in 'data/fertilizer_dataset/'.")
        print("Model remains in 'Not Trained' status.")
        return

    print(f"[*] Loading dataset: {FERTILIZER_DATA_PATH}")
    df = pd.read_csv(FERTILIZER_DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    # Map column names if needed
    col_map = {
        "Crop_Type": "crop",
        "Soil_Type": "soil_type",
        "Nitrogen_Level": "N",
        "Phosphorus_Level": "P",
        "Potassium_Level": "K",
        "Recommended_Fertilizer": "fertilizer"
    }
    
    # Check if standard or custom schema
    available_cols = set(df.columns)
    if not all(col in available_cols for col in col_map.keys()):
        print(f"[Notice] Dataset columns: {list(df.columns)}")
        # Check fallback simple columns
        simple_cols = ["Crop", "Soil Type", "Nitrogen", "Phosphorus", "Potassium", "Fertilizer Name"]
        if all(c in df.columns for c in simple_cols):
            df = df.rename(columns={
                "Crop": "crop",
                "Soil Type": "soil_type",
                "Nitrogen": "N",
                "Phosphorus": "P",
                "Potassium": "K",
                "Fertilizer Name": "fertilizer"
            })
        else:
            print("[Warning] Dataset does not match required schemas. Needs Crop, Soil_Type, N, P, K, Fertilizer.")
            return
    else:
        df = df.rename(columns=col_map)

    required = ["crop", "soil_type", "N", "P", "K", "fertilizer"]
    df = df.dropna(subset=required).copy()
    
    # Clean string columns
    df["crop"] = df["crop"].astype(str).str.strip().str.title()
    df["soil_type"] = df["soil_type"].astype(str).str.strip().str.title()
    df["fertilizer"] = df["fertilizer"].astype(str).str.strip()

    print(f"[*] Valid records: {len(df)} | Unique fertilizers: {df['fertilizer'].nunique()}")

    categorical_features = ["crop", "soil_type"]
    numerical_features = ["N", "P", "K"]

    X = df[categorical_features + numerical_features]
    y = df["fertilizer"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ]
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("[*] Training Fertilizer Recommendation Pipeline...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("\n--- FERTILIZER MODEL PERFORMANCE ---")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall   : {recall * 100:.2f}%")
    print(f"  F1 Score : {f1 * 100:.2f}%")
    print("------------------------------------")

    # Save Pipeline
    FERTILIZER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, FERTILIZER_MODEL_PATH)
    print(f"[+] Saved model to: {FERTILIZER_MODEL_PATH}")

    # Save Real Metrics for Dashboard
    metrics_payload = {
        "status": "ready",
        "model_name": "Random Forest Fertilizer Recommender",
        "algorithm": "Preprocessed Random Forest Pipeline",
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_classes": int(len(clf.classes_)),
        "classes": sorted(list(clf.classes_)),
        "accuracy": round(float(accuracy) * 100, 2),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "supported_crops": sorted(list(df["crop"].unique())),
        "supported_soils": sorted(list(df["soil_type"].unique()))
    }

    with open(FERTILIZER_METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[+] Saved metrics to: {FERTILIZER_METRICS_PATH}")
    print("[OK] Fertilizer model training complete.\n")

if __name__ == "__main__":
    train()

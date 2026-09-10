"""
SMART AGRI AI - Crop Recommendation Model Training Pipeline
Trains a multi-class Random Forest Classifier on genuine agronomic dataset.
Saves model weights and authentic evaluation metrics for the dashboard.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Ensure base project path is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CROP_DATA_PATH, CROP_MODEL_PATH, CROP_METRICS_PATH, CROP_FEATURES

def train():
    print("=" * 65)
    print("  SMART AGRI AI - CROP MODEL TRAINING PIPELINE")
    print("=" * 65)

    if not CROP_DATA_PATH.exists():
        print(f"[Error] Dataset not found at: {CROP_DATA_PATH}")
        print("Please place 'Crop_recommendation.csv' in the 'data/' folder.")
        sys.exit(1)

    print(f"[*] Loading dataset: {CROP_DATA_PATH}")
    df = pd.read_csv(CROP_DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    target_col = "label"
    required_cols = CROP_FEATURES + [target_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Data hygiene
    df = df.dropna(subset=required_cols).copy()
    print(f"[*] Dataset verified: {len(df)} records across {df[target_col].nunique()} distinct crops.")

    X = df[CROP_FEATURES]
    y = df[target_col].astype(str).str.strip().str.lower()

    # Stratified Train/Test Split
    test_size = 0.20
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[*] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    print(f"[*] Training Random Forest Classifier (n_estimators=150, random_state={random_state})...")

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=random_state,
        n_jobs=-1,
        criterion="gini"
    )
    model.fit(X_train, y_train)

    # Real Evaluation on Unseen Test Split
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("\n--- MODEL PERFORMANCE METRICS ---")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall   : {recall * 100:.2f}%")
    print(f"  F1 Score : {f1 * 100:.2f}%")
    print("---------------------------------")

    # Feature Importance
    importances = model.feature_importances_
    feat_imp = {feat: round(float(imp), 4) for feat, imp in zip(CROP_FEATURES, importances)}
    sorted_feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))

    print("\nFeature Importances:")
    for feat, imp in sorted_feat_imp.items():
        print(f"  - {feat:12s}: {imp * 100:.2f}%")

    # Save Model Weights
    CROP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CROP_MODEL_PATH)
    print(f"\n[+] Saved model artifact to: {CROP_MODEL_PATH}")

    # Save Authentic Metrics for the Dashboard
    metrics_payload = {
        "status": "ready",
        "model_name": "Random Forest Classifier",
        "algorithm": "Ensemble Decision Trees (Random Forest)",
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_classes": int(len(model.classes_)),
        "classes": sorted(list(model.classes_)),
        "accuracy": round(float(accuracy) * 100, 2),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "feature_importances": sorted_feat_imp
    }

    with open(CROP_METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[+] Saved evaluation metrics to: {CROP_METRICS_PATH}")
    print("[OK] Crop recommendation model training complete.\n")

if __name__ == "__main__":
    train()

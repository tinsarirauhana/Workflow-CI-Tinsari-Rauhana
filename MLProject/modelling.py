"""
modelling.py
============
Melatih model Random Forest Classifier menggunakan MLflow Tracking (autolog).
Dataset: Heart Disease UCI (sudah diproses oleh automate_Nama-siswa.py)

Cara menjalankan:
    pip install mlflow scikit-learn pandas
    python modelling.py

Hasil:
    MLflow Tracking UI → http://127.0.0.1:5000
    Jalankan: mlflow ui
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI PATH
# ============================================================
TRAIN_PATH = "./heart_disease_preprocessing/heart_train_preprocessed.csv"
TEST_PATH  = "./heart_disease_preprocessing/heart_test_preprocessed.csv"
TARGET_COL = "condition"
EXPERIMENT_NAME = "Heart-Disease-Classification"


def load_data():
    """Memuat dataset yang sudah diproses."""
    print("📂 Memuat data ...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    X_train = train.drop(TARGET_COL, axis=1)
    y_train = train[TARGET_COL]
    X_test  = test.drop(TARGET_COL, axis=1)
    y_test  = test[TARGET_COL]

    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run_experiment(X_train, X_test, y_train, y_test):
    """Melatih model dengan MLflow autolog."""
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    mlflow.end_run()
    print("\n🚀 Memulai MLflow run (autolog) ...")
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Cetak hasil evaluasi
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        print("\n📊 Hasil Evaluasi Model:")
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   F1-Score  : {f1:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   ROC-AUC   : {auc:.4f}")
        print("\n" + classification_report(y_test, y_pred,
              target_names=["Tidak Sakit", "Sakit Jantung"]))

    print("\n✅ Run selesai! Buka MLflow UI: mlflow ui --port 5000")


if __name__ == "__main__":
    print("=" * 50)
    print("  MODELLING - Heart Disease Classification")
    print("=" * 50)
    X_train, X_test, y_train, y_test = load_data()
    run_experiment(X_train, X_test, y_train, y_test)

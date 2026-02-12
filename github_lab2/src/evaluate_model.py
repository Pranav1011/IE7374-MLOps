#!/usr/bin/env python3
"""
Load the saved model and test set, evaluate metrics, and write {timestamp}_metrics.json.
Workflow expects the metrics file to be in the repo root, named exactly {timestamp}_metrics.json

Usage:
    python src/evaluate_model.py --timestamp 20260211123456
"""

import argparse
import json
import os
import pickle

from joblib import load
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def main(timestamp: str):
    # Model file expected name (keeps compatibility with the provided workflows)
    model_version = f"model_{timestamp}_gb_model"
    model_filename = f"{model_version}.joblib"

    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Expected model file not found: {model_filename}")

    model = load(model_filename)

    # Load the test set saved by train_model.py
    test_X_path = os.path.join("data", "test_X.pickle")
    test_y_path = os.path.join("data", "test_y.pickle")

    if not (os.path.exists(test_X_path) and os.path.exists(test_y_path)):
        # As a fallback, load the Breast Cancer dataset and create a test split deterministically
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        X, y = load_breast_cancer(return_X_y=True)
        _, test_X, _, test_y = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
    else:
        with open(test_X_path, "rb") as fx:
            test_X = pickle.load(fx)
        with open(test_y_path, "rb") as fy:
            test_y = pickle.load(fy)

    preds = model.predict(test_X)
    proba = model.predict_proba(test_X)[:, 1]

    acc = float(accuracy_score(test_y, preds))
    f1 = float(f1_score(test_y, preds, average="weighted"))
    prec = float(precision_score(test_y, preds, average="weighted"))
    rec = float(recall_score(test_y, preds, average="weighted"))
    roc_auc = float(roc_auc_score(test_y, proba))
    conf_mat = confusion_matrix(test_y, preds).tolist()

    metrics = {
        "accuracy": acc,
        "f1_weighted": f1,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_mat,
    }

    # write metrics JSON in repo root with the timestamped filename (workflow expects this)
    if not os.path.exists("metrics"):
        os.makedirs("metrics")

    metrics_filename = os.path.join("metrics", f"{timestamp}_metrics.json")
    with open(metrics_filename, "w") as mf:
        json.dump(metrics, mf, indent=4)

    print(f"Wrote metrics to {metrics_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    main(args.timestamp)

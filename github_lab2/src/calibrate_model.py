#!/usr/bin/env python3
"""
Calibrate a trained model using CalibratedClassifierCV (isotonic or sigmoid).
Saves the calibrated model and calibration metrics.

Usage:
    python src/calibrate_model.py --timestamp 20260211123456 --method isotonic
"""

import argparse
import json
import os
import pickle

import numpy as np
from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split


def main(timestamp: str, method: str):
    # Load the trained model
    model_version = f"model_{timestamp}_gb_model"
    model_filename = f"{model_version}.joblib"

    if not os.path.exists(model_filename):
        # Check in models/ folder
        model_filename = os.path.join("models", f"{model_version}.joblib")
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_version}.joblib")

    model = load(model_filename)

    # Load test data
    test_X_path = os.path.join("data", "test_X.pickle")
    test_y_path = os.path.join("data", "test_y.pickle")

    if not (os.path.exists(test_X_path) and os.path.exists(test_y_path)):
        X, y = load_breast_cancer(return_X_y=True)
        _, test_X, _, test_y = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
    else:
        with open(test_X_path, "rb") as fx:
            test_X = pickle.load(fx)
        with open(test_y_path, "rb") as fy:
            test_y = pickle.load(fy)

    # Get uncalibrated probabilities
    uncal_proba = model.predict_proba(test_X)[:, 1]
    uncal_brier = brier_score_loss(test_y, uncal_proba)
    uncal_logloss = log_loss(test_y, uncal_proba)

    # For calibration, we need to train a fresh model with calibration wrapper
    # Load the full dataset and use cross-validation to fit the calibrator
    X, y = load_breast_cancer(return_X_y=True)

    # Create a new GradientBoosting model with the same parameters
    from sklearn.ensemble import GradientBoostingClassifier
    base_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    # Calibrate using cross-validation (cv=5 performs 5-fold CV internally)
    calibrated_clf = CalibratedClassifierCV(base_model, method=method, cv=5)
    calibrated_clf.fit(X, y)

    # Get calibrated probabilities
    cal_proba = calibrated_clf.predict_proba(test_X)[:, 1]
    cal_brier = brier_score_loss(test_y, cal_proba)
    cal_logloss = log_loss(test_y, cal_proba)

    # Save calibrated model
    if not os.path.exists("models"):
        os.makedirs("models")

    cal_model_filename = f"model_{timestamp}_cal_{method}.joblib"
    dump(calibrated_clf, cal_model_filename)
    dump(calibrated_clf, os.path.join("models", cal_model_filename))
    print(f"Saved calibrated model to: {cal_model_filename}")

    # Save calibration metrics
    if not os.path.exists("metrics"):
        os.makedirs("metrics")

    calibration_metrics = {
        "method": method,
        "uncalibrated_brier_score": float(uncal_brier),
        "calibrated_brier_score": float(cal_brier),
        "brier_improvement": float(uncal_brier - cal_brier),
        "uncalibrated_log_loss": float(uncal_logloss),
        "calibrated_log_loss": float(cal_logloss),
        "log_loss_improvement": float(uncal_logloss - cal_logloss),
    }

    metrics_filename = os.path.join("metrics", f"{timestamp}_calibration_{method}.json")
    with open(metrics_filename, "w") as mf:
        json.dump(calibration_metrics, mf, indent=4)

    print(f"Wrote calibration metrics to {metrics_filename}")
    print(f"Brier score: {uncal_brier:.4f} -> {cal_brier:.4f} (improvement: {uncal_brier - cal_brier:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    parser.add_argument("--method", type=str, default="isotonic", choices=["isotonic", "sigmoid"],
                        help="Calibration method: isotonic or sigmoid (Platt scaling)")
    args = parser.parse_args()
    main(args.timestamp, args.method)

#!/usr/bin/env python3
"""
Train a Gradient Boosting Classifier on the Breast Cancer dataset, log with MLflow, and save the model.

Usage:
    python src/train_model.py --timestamp 20260211123456
"""

import argparse
import datetime
import os
import pickle

import mlflow
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def main(timestamp: str):
    # --- prepare data ---
    X, y = load_breast_cancer(return_X_y=True)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    # Save test split so evaluate_model.py can reuse it
    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/test_X.pickle", "wb") as fx:
        pickle.dump(test_X, fx)
    with open("data/test_y.pickle", "wb") as fy:
        pickle.dump(test_y, fy)

    # --- mlflow setup ---
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "BreastCancer"
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{dataset_name}_GradientBoosting"):
        mlflow.log_param("dataset", "breast_cancer")
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("model_type", "GradientBoostingClassifier")

        # --- train model ---
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        clf.fit(train_X, train_y)

        # --- metrics on test set ---
        preds = clf.predict(test_X)
        proba = clf.predict_proba(test_X)[:, 1]

        acc = accuracy_score(test_y, preds)
        f1 = f1_score(test_y, preds, average="weighted")
        prec = precision_score(test_y, preds, average="weighted")
        rec = recall_score(test_y, preds, average="weighted")
        roc_auc = roc_auc_score(test_y, proba)

        mlflow.log_metrics({
            "accuracy": acc,
            "f1_weighted": f1,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "roc_auc": roc_auc
        })

        # Ensure models/ folder exists
        if not os.path.exists("models"):
            os.makedirs("models")

        # Save the model using the naming convention expected by GitHub Actions
        model_version = f"model_{timestamp}_gb_model"
        model_filename = f"{model_version}.joblib"
        dump(clf, model_filename)
        print(f"Saved model to: {model_filename}")

        # Also save a copy under models/ for local listing
        dump(clf, os.path.join("models", model_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    main(args.timestamp)

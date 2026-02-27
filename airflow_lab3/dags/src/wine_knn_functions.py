import os
import pickle
import base64
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from kneed import KneeLocator


def load_data():
    """
    Loads Wine dataset, serializes it and returns base64 encoded data.
    """
    wine = load_wine(as_frame=True)
    df = wine.frame

    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes data, performs preprocessing and returns
    base64-encoded scaled feature data + labels.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()

    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    processed = {"X": X_scaled, "y": y}

    serialized = pickle.dumps(processed)
    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Trains KNN models for multiple K values, computes accuracy,
    saves the final model, and returns accuracy list (for elbow).
    """
    data_bytes = base64.b64decode(data_b64)
    data_dict = pickle.loads(data_bytes)

    X = data_dict["X"]
    y = data_dict["y"]

    k_values = range(1, 31)
    accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=5)
        accuracies.append(score.mean())

    # Fit final model with best K
    best_k = k_values[accuracies.index(max(accuracies))]
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X, y)

    # Save model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump(final_model, f)

    return accuracies


def load_model_elbow(filename: str, accuracies: list):
    """
    Loads saved KNN model, finds elbow K, and predicts first wine sample.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # Determine elbow
    k_range = range(1, len(accuracies) + 1)
    kl = KneeLocator(k_range, accuracies, curve="concave", direction="increasing")
    print(f"Optimal K (elbow): {kl.elbow}")

    # Load wine for prediction example
    wine = load_wine()
    X_test = wine.data

    pred = loaded_model.predict(X_test)[0]

    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred

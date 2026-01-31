from sklearn.ensemble import RandomForestClassifier
import joblib
from data import load_data, split_data


def fit_model(X_train, y_train):
    """
    Train a Random Forest Classifier and save the model to a file.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=12)
    rf_classifier.fit(X_train, y_train)
    joblib.dump(rf_classifier, "../model/wine_model.pkl")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

if __name__ == '__main__':
    # Load the Wine dataset
    wine = datasets.load_wine()
    X, y = wine.data, wine.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train a Gradient Boosting classifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model training completed with accuracy: {accuracy:.2f}")

    # Save both model and scaler for serving stage
    joblib.dump((model, sc), 'wine_gb_model.joblib')
    print("Model and scaler saved as wine_gb_model.joblib")

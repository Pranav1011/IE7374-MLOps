from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, static_folder='statics')

# Load the trained model and scaler
model, sc = joblib.load('wine_gb_model.joblib')
class_labels = ['Barolo (Class 0)', 'Grignolino (Class 1)', 'Barbera (Class 2)']


@app.route('/')
def home():
    return "Welcome to the Wine Gradient Boosting Classifier API!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form

            # Get all 13 wine features
            features = [
                float(data['alcohol']),
                float(data['malic_acid']),
                float(data['ash']),
                float(data['alcalinity_of_ash']),
                float(data['magnesium']),
                float(data['total_phenols']),
                float(data['flavanoids']),
                float(data['nonflavanoid_phenols']),
                float(data['proanthocyanins']),
                float(data['color_intensity']),
                float(data['hue']),
                float(data['od280_od315']),
                float(data['proline'])
            ]

            # Prepare and scale input
            input_data = np.array([features])
            input_scaled = sc.transform(input_data)

            # Make prediction
            predicted_class = class_labels[model.predict(input_scaled)[0]]

            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)

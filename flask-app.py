from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging

app = Flask(__name__)
model = joblib.load("rf_model.pkl")
feature_order = [
    "Pregnancy",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPeedigreeFunction",
    "Age"
]

CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/predict', methods=['POST'])
def predict_diabetes():
    data = request.get_json()
    height_value = data["Height"]
    weight_value = data["Weight"]

    BMI = float(weight_value) / pow((float(height_value)/100), 2)

    data["BMI"] = BMI

    for key in list(data.keys()):
        if key not in feature_order:
            data.pop(key)

    features = [data[feature] for feature in feature_order]
    features = np.array(features).reshape(1,-1)
    prediction = model.predict(features).tolist()
    return jsonify({"Result": prediction[0]}), 200

if __name__ == '__main__':
    app.run(
        port=1249,
        debug=True
    )


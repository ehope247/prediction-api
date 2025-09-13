from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('prediction_model.joblib')

@app.route('/')
def status(): return "Prediction API is online."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'home_id' not in data or 'away_id' not in data:
        return jsonify({"error": "Missing home_id or away_id"}), 400
    try:
        input_data = np.array([[int(data['home_id']), int(data['away_id'])]])
        prediction_code = model.predict(input_data)[0]
        result_map = {0: "Home Team Win", 1: "Away Team Win", 2: "Draw"}
        prediction_text = result_map.get(prediction_code, "Prediction undetermined.")
        return jsonify({"prediction": prediction_text})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

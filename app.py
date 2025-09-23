from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [data.get(f, 0.0) for f in ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']]
        features = np.array(features).reshape(1, -1)
        scaled = scaler.transform(features)
        pred = model.predict(scaled)
        is_fraud = bool(pred[0])
        score = model.predict_proba(scaled)[0][1]  # Probability of fraud
        return jsonify({'fraud': is_fraud, 'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) #This sets up a Flask app with a /predict endpoint that accepts JSON input (e.g., transaction data) and returns fraud predictions. It loads the saved model and scaler, ensuring consistency with training. The predict_proba method provides a confidence score, enhancing the API’s utility.
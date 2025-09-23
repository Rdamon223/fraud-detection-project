from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from prometheus_client import Counter, generate_latest, REGISTRY

app = Flask(__name__, template_folder='templates')
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

requests_total = Counter('requests_total', 'Total prediction requests')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = {k: float(v) for k, v in request.form.items()}
        features = [data.get(f, 0.0) for f in ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']]
        features = np.array(features).reshape(1, -1)
        scaled = scaler.transform(features)
        pred = model.predict(scaled)
        score = model.predict_proba(scaled)[0][1]
        is_fraud = bool(pred[0])
        requests_total.inc()
        return render_template('index.html', prediction={'fraud': is_fraud, 'score': score})
    requests_total.inc()
    return render_template('index.html')

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
        requests_total.inc()
        return jsonify({'fraud': is_fraud, 'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics')
def metrics():
    return generate_latest(REGISTRY)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Added use_reloader=False to prevent duplicate registration.  This sets up a Flask app with a /predict endpoint that accepts JSON input (e.g., transaction data) and returns fraud predictions. It loads the saved model and scaler, ensuring consistency with training. The predict_proba method provides a confidence score, enhancing the API’s utility.
import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained DecisionTreeClassifier model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'decision_pro.pkl')

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Decision Pro AI - Classifier Dashboard</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #f857a6 0%, #ff5858 100%);
            --card-bg: rgba(255, 255, 255, 0.92);
            --primary: #ff5858;
            --text-main: #2d3436;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body { background: var(--bg-gradient); background-attachment: fixed; min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 30px 15px; }
        .main-container { background: var(--card-bg); border-radius: 24px; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2); padding: 40px; width: 100%; max-width: 900px; backdrop-filter: blur(12px); }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { background: linear-gradient(45deg, #ff5858, #f857a6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 800; margin-bottom: 8px; }
        .header p { color: #636e72; font-size: 1.05rem; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; }
        .input-box { display: flex; flex-direction: column; }
        .input-box label { font-size: 0.85rem; font-weight: 700; color: var(--text-main); margin-bottom: 6px; text-transform: uppercase; }
        .input-box input, .input-box select { padding: 12px 15px; border: 2px solid #dfe6e9; border-radius: 12px; font-size: 1rem; outline: none; transition: all 0.3s ease; background: #fff; }
        .input-box input:focus, .input-box select:focus { border-color: var(--primary); box-shadow: 0 0 10px rgba(255, 88, 88, 0.25); }
        .submit-container { grid-column: 1 / -1; margin-top: 15px; }
        .btn-predict { width: 100%; padding: 15px; background: linear-gradient(135deg, #ff5858 0%, #f857a6 100%); border: none; border-radius: 12px; color: white; font-size: 1.15rem; font-weight: bold; cursor: pointer; box-shadow: 0 8px 20px rgba(248, 87, 166, 0.4); }
        .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 12px 25px rgba(248, 87, 166, 0.6); }
        .result-box { margin-top: 30px; padding: 20px; border-radius: 16px; background: linear-gradient(135deg, #00b09b, #96c93d); color: white; text-align: center; box-shadow: 0 8px 20px rgba(0, 176, 155, 0.3); }
        .error-box { margin-top: 30px; padding: 15px; border-radius: 12px; background: #ff4757; color: white; text-align: center; font-weight: 600; }
    </style>
</head>
<body>
<div class="main-container">
    <div class="header">
        <h1>Decision Pro Classifier</h1>
        <p>Interactive machine learning prediction dashboard</p>
    </div>
    <form action="/predict" method="POST" class="form-grid">
        <div class="input-box"><label>Age</label><input type="number" step="any" name="age" required placeholder="e.g. 30"></div>
        <div class="input-box"><label>Gender</label><select name="gender" required><option value="0">Female (0)</option><option value="1">Male (1)</option></select></div>
        <div class="input-box"><label>Subscription Type</label><input type="number" step="any" name="subscription_type" required placeholder="e.g. 1"></div>
        <div class="input-box"><label>Watch Hours</label><input type="number" step="any" name="watch_hours" required placeholder="e.g. 55.2"></div>
        <div class="input-box"><label>Last Login (Days)</label><input type="number" step="any" name="last_login_days" required placeholder="e.g. 2"></div>
        <div class="input-box"><label>Region Code</label><input type="number" step="any" name="region" required placeholder="e.g. 3"></div>
        <div class="input-box"><label>Device Type</label><input type="number" step="any" name="device" required placeholder="e.g. 1"></div>
        <div class="input-box"><label>Monthly Fee ($)</label><input type="number" step="any" name="monthly_fee" required placeholder="e.g. 12.99"></div>
        <div class="input-box"><label>Payment Method</label><input type="number" step="any" name="payment_method" required placeholder="e.g. 0"></div>
        <div class="input-box"><label>Number of Profiles</label><input type="number" step="any" name="number_of_profiles" required placeholder="e.g. 4"></div>
        <div class="input-box"><label>Avg Watch Time / Day</label><input type="number" step="any" name="avg_watch_time_per_day" required placeholder="e.g. 3.5"></div>
        <div class="input-box"><label>Favorite Genre Code</label><input type="number" step="any" name="favorite_genre" required placeholder="e.g. 2"></div>
        <div class="submit-container"><button type="submit" class="btn-predict">Run Prediction</button></div>
    </form>
    {% if submitted %}
    <div class="result-box">
        <h2>{{ prediction_text }}</h2>
        <p>{{ confidence_text }}</p>
    </div>
    {% endif %}
    {% if error_text %}
    <div class="error-box"><p>{{ error_text }}</p></div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_text='Error: decision_pro.pkl model file not found.')
    try:
        input_features = [
            float(request.form.get('age', 0)),
            float(request.form.get('gender', 0)),
            float(request.form.get('subscription_type', 0)),
            float(request.form.get('watch_hours', 0)),
            float(request.form.get('last_login_days', 0)),
            float(request.form.get('region', 0)),
            float(request.form.get('device', 0)),
            float(request.form.get('monthly_fee', 0)),
            float(request.form.get('payment_method', 0)),
            float(request.form.get('number_of_profiles', 0)),
            float(request.form.get('avg_watch_time_per_day', 0)),
            float(request.form.get('favorite_genre', 0))
        ]
        features_array = np.array([input_features])
        prediction = model.predict(features_array)[0]
        
        confidence_text = ""
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_array)[0]
            max_prob = round(float(np.max(probabilities)) * 100, 2)
            confidence_text = f"Confidence: {max_prob}%"

        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=f"Predicted Result: {prediction}",
            confidence_text=confidence_text,
            submitted=True
        )
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f'Prediction error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and scaler
try:
    models = {
        'a': joblib.load('model_a.pkl'),
        'b': joblib.load('model_b.pkl'),
        'combined': joblib.load('model_all.pkl')
    }
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    raise RuntimeError(f"Error loading models or scaler: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate and parse input data
        region = data.get('region')
        jun_sep = float(data.get('jun_sep', 0))
        curve_number = float(data.get('curve_number', 0))
        retention = float(data.get('retention', 0))
        runoff = float(data.get('runoff', 0))

        if region not in models:
            return jsonify({'error': 'Invalid region specified.'}), 400

        # Create feature DataFrame
        features = pd.DataFrame([[
            jun_sep, curve_number, retention, runoff
        ]], columns=[
            'Jun-Sep', 'Curve Number', 
            'Potential Maximum Retention', 'Surface Runoff'
        ])

        # Preprocess
        scaled_features = scaler.transform(features)

        # Predict
        model = models[region]
        prediction = int(model.predict(scaled_features)[0])
        probability = float(model.predict_proba(scaled_features)[0][1]) * 100

        return jsonify({
            'prediction': prediction,
            'probability': round(probability, 2),
            'model_used': f"Model {region.upper()}" if region != 'combined' else "Combined Model"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

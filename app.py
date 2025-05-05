from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and scaler
models = {
    'a': joblib.load('model_a.pkl'),
    'b': joblib.load('model_b.pkl'),
    'combined': joblib.load('model_all.pkl')
}
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Get inputs
        region = data['region']
        features = [
            float(data['jun_sep']),
            float(data['curve_number']),
            float(data['retention']),
            float(data['runoff'])
        ]
        
        # Create DataFrame
        sample_df = pd.DataFrame([features], columns=[
            'Jun-Sep', 'Curve Number', 
            'Potential Maximum Retention', 'Surface Runoff'
        ])
        
        # Scale features
        sample_scaled = scaler.transform(sample_df)
        
        # Get prediction
        model = models[region]
        proba = model.predict_proba(sample_scaled)[0][1]
        prediction = model.predict(sample_scaled)[0]
        
        return jsonify({
            'probability': round(proba * 100, 2),
            'prediction': int(prediction),
            'model_used': f"Model {region.upper()}" if region != 'combined' else "Combined Model"
        })
    
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 400
        

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

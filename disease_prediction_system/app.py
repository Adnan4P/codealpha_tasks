import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths
MODELS_DIR = 'models'
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics_summary.json')

# Cache for models, scalers and metadata
models = {}
scalers = {}
metrics_summary = {}

def load_resources():
    """Load models, scalers, and metrics on startup."""
    global metrics_summary
    
    # Load metrics summary
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_summary = json.load(f)
            
    # Load model and scaler for each disease
    for disease in ['heart_disease', 'diabetes', 'breast_cancer']:
        model_path = os.path.join(MODELS_DIR, f'{disease}_model.joblib')
        scaler_path = os.path.join(MODELS_DIR, f'{disease}_scaler.joblib')
        
        if os.path.exists(model_path):
            models[disease] = joblib.load(model_path)
            print(f"Loaded model for {disease}")
            
        if os.path.exists(scaler_path):
            scalers[disease] = joblib.load(scaler_path)
            print(f"Loaded scaler for {disease}")

# Initial load of resources
try:
    load_resources()
except Exception as e:
    print(f"Warning: Could not load models/scalers initially: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Endpoint to return validation metrics for all models."""
    global metrics_summary
    if not metrics_summary:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics_summary = json.load(f)
        else:
            return jsonify({'error': 'Metrics not generated yet. Please run training pipeline.'}), 500
            
    return jsonify(metrics_summary)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint to perform predictions based on disease and features."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request: No data provided'}), 400
        
    disease = data.get('disease')
    input_features = data.get('features')
    
    if not disease or not input_features:
        return jsonify({'error': 'Invalid request: disease and features are required'}), 400
        
    if disease not in models or disease not in scalers:
        # Try reloading resources (maybe training completed after startup)
        load_resources()
        if disease not in models or disease not in scalers:
            return jsonify({'error': f'Model/scaler for {disease} is not loaded'}), 500
            
    # Get the features in correct order
    feature_list = metrics_summary.get(disease, {}).get('features')
    if not feature_list:
        return jsonify({'error': f'Feature order not found for {disease}'}), 500
        
    # Construct ordered array/dataframe for scaling
    ordered_features = []
    try:
        for feat in feature_list:
            if feat not in input_features:
                return jsonify({'error': f'Missing feature in input: {feat}'}), 400
            ordered_features.append(float(input_features[feat]))
    except ValueError as e:
        return jsonify({'error': f'Invalid value in features: {str(e)}'}), 400
        
    # Create DataFrame
    features_df = pd.DataFrame([ordered_features], columns=feature_list)
    
    # Preprocess (Scale)
    scaled_features = scalers[disease].transform(features_df)
    
    # Predict
    model = models[disease]
    prediction = int(model.predict(scaled_features)[0])
    
    # Probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_features)[0]
        # Class 1 probability
        risk_probability = float(probabilities[1])
    else:
        # Fallback for models without predict_proba (some SVM setups)
        decision = model.decision_function(scaled_features)[0]
        # Use sigmoid scaling to get a rough probability
        risk_probability = float(1 / (1 + np.exp(-decision)))
        
    # Explain feature contributions (simple method: feature importance * scaled feature value)
    contributions = []
    best_model_name = metrics_summary[disease]['best_model']
    importances = metrics_summary[disease].get('importances')
            
    if importances is not None:
        # Contribution is scaled value * importance
        # (This shows what features pushed the model towards the outcome, relative to mean)
        scaled_row = scaled_features[0]
        for name, imp, s_val in zip(feature_list, importances, scaled_row):
            # Focus on absolute contribution size
            contrib_val = float(imp * s_val)
            contributions.append({
                'feature': name,
                'raw_importance': float(imp),
                'contribution': contrib_val
            })
        # Sort contributions by absolute size
        contributions = sorted(contributions, key=lambda x: abs(x['contribution']), reverse=True)
        
    return jsonify({
        'disease': disease,
        'prediction': prediction,
        'probability': risk_probability,
        'best_model': best_model_name,
        'contributions': contributions
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

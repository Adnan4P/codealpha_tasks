import os
import logging
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

from src.model import load_saved_model
from src.preprocessing import preprocess_raw_data
from src.feature_engineering import add_custom_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the Random Forest model pipeline globally at startup
MODEL_NAME = "Random_Forest"
model_pipeline = None

try:
    model_pipeline = load_saved_model(MODEL_NAME)
    logger.info(f"Successfully loaded model pipeline: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading model pipeline at startup: {e}. "
                 f"Ensure main.py has been run and models/ exists.")

@app.route('/')
def home():
    """Renders the dashboard UI."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON financial/applicant details, performs pre-processing, 
    runs prediction via the pipeline, and returns risk analysis results.
    """
    global model_pipeline
    
    # Reload model if loading failed at startup
    if model_pipeline is None:
        try:
            model_pipeline = load_saved_model(MODEL_NAME)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Model pipeline not available on server: {e}"
            }), 500
            
    try:
        data = request.get_json(force=True)
        logger.info(f"Received prediction request: {data}")
        
        # 1. Map input dictionary to DataFrame
        df_input = pd.DataFrame([data])
        
        # 2. Run preprocessing and feature engineering
        df_clean = preprocess_raw_data(df_input)
        df_engineered = add_custom_features(df_clean)
        
        # 3. Predict probability and class label
        # Column transformer handles scaling & one-hot encoding within pipeline
        prediction = model_pipeline.predict(df_engineered)[0]
        probabilities = model_pipeline.predict_proba(df_engineered)[0]
        
        # Risk probability for class 1 (Bad Risk)
        prob_bad = float(probabilities[1])
        prob_good = float(probabilities[0])
        
        # Generate model recommendations
        recommendation = "REJECT" if prediction == 1 else "APPROVE"
        risk_label = "Bad Risk (High probability of default)" if prediction == 1 else "Good Risk (Low probability of default)"
        
        # Determine credit grade mapped from probability
        # mapping: lower probability of default = higher credit grade
        if prob_bad < 0.15:
            credit_grade = "A (Excellent)"
        elif prob_bad < 0.25:
            credit_grade = "B (Good)"
        elif prob_bad < 0.40:
            credit_grade = "C (Fair)"
        elif prob_bad < 0.60:
            credit_grade = "D (Weak)"
        else:
            credit_grade = "F (High Risk)"
            
        response = {
            "status": "success",
            "prediction": int(prediction),
            "risk_label": risk_label,
            "probability_bad": prob_bad,
            "probability_good": prob_good,
            "recommendation": recommendation,
            "credit_grade": credit_grade
        }
        
        logger.info(f"Prediction response: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.exception("Error during web prediction request processing")
        return jsonify({
            "status": "error",
            "message": f"Failed to compute prediction: {str(e)}"
        }), 400

if __name__ == '__main__':
    # Local host server
    logger.info("Starting Credit Scoring Web Application...")
    app.run(host='127.0.0.1', port=5000, debug=True)

import os
import argparse
import pandas as pd
import numpy as np
from src.model import load_saved_model
from src.preprocessing import preprocess_raw_data
from src.feature_engineering import add_custom_features

# Set up sample profiles for quick testing
SAMPLE_PROFILES = {
    "low_risk": {
        "Age": 45,
        "Sex": "male",
        "Job": 2,  # Skilled worker
        "Housing": "own",
        "Saving accounts": "rich",
        "Checking account": "moderate",
        "Credit amount": 2500,
        "Duration": 12,
        "Purpose": "car"
    },
    "high_risk": {
        "Age": 21,
        "Sex": "female",
        "Job": 1,  # Unskilled resident
        "Housing": "rent",
        "Saving accounts": "little",
        "Checking account": "little",
        "Credit amount": 8000,
        "Duration": 48,
        "Purpose": "vacation/others"
    }
}

def predict_creditworthiness(profile_data, model_name="Random_Forest"):
    """
    Loads the specified trained pipeline, preprocesses the raw profile dict,
    and runs a prediction.
    """
    # 1. Convert profile dictionary to a single-row DataFrame
    df_raw = pd.DataFrame([profile_data])
    
    # 2. Apply preprocessing (imputing NaN checks) and feature engineering
    df_clean = preprocess_raw_data(df_raw)
    df_engineered = add_custom_features(df_clean)
    
    # 3. Load model pipeline (includes scaling, one-hot encoding, and classifier)
    try:
        pipeline = load_saved_model(model_name)
    except FileNotFoundError:
        print(f"Error: Saved pipeline '{model_name}' not found. Have you run 'main.py' yet?")
        return None
        
    # 4. Predict
    prediction = pipeline.predict(df_engineered)[0]
    probability_bad = pipeline.predict_proba(df_engineered)[0][1]
    
    return {
        "prediction_class": int(prediction),
        "prediction_label": "Bad Risk (High probability of default)" if prediction == 1 else "Good Risk (Low probability of default)",
        "probability_bad_risk": float(probability_bad)
    }

def main():
    parser = argparse.ArgumentParser(description="Predict an individual's creditworthiness.")
    parser.add_argument(
        "--profile", 
        type=str, 
        choices=["low_risk", "high_risk"], 
        default="low_risk",
        help="Choose a pre-defined sample profile to test."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["Logistic_Regression", "Decision_Tree", "Random_Forest"], 
        default="Random_Forest",
        help="The classification model to use."
    )
    args = parser.parse_args()
    
    print(f"--- Running Prediction using {args.model} ---")
    profile = SAMPLE_PROFILES[args.profile]
    print("\nApplicant Profile:")
    for k, v in profile.items():
        print(f"  {k}: {v}")
        
    result = predict_creditworthiness(profile, args.model)
    if result:
        print("\n--- Prediction Results ---")
        print(f"Result Label: {result['prediction_label']}")
        print(f"Probability of default (Bad Risk): {result['probability_bad_risk']:.2%}")
        print(f"Model Recommendation: {'REJECT LOAN' if result['prediction_class'] == 1 else 'APPROVE LOAN'}")
        print("--------------------------")

if __name__ == "__main__":
    main()

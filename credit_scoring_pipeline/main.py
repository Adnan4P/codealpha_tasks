import logging
import sys
from src.data_loader import load_dataset
from src.preprocessing import preprocess_raw_data, split_data
from src.feature_engineering import add_custom_features, get_feature_lists, build_preprocessor
from src.model import train_and_save_models
from src.evaluate import evaluate_models

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("=" * 60)
    logger.info("STARTING CREDIT SCORING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Acquire Data
    logger.info("[STEP 1/6] Loading data...")
    df_raw = load_dataset()
    logger.info(f"Loaded raw dataset of shape: {df_raw.shape}")
    
    # Step 2: Clean/Preprocess Data
    logger.info("[STEP 2/6] Cleaning and preprocessing data...")
    df_clean = preprocess_raw_data(df_raw)
    
    # Step 3: Feature Engineering
    logger.info("[STEP 3/6] Applying custom feature engineering...")
    df_engineered = add_custom_features(df_clean)
    
    # Step 4: Split Train/Test Sets
    logger.info("[STEP 4/6] Splitting train and test sets...")
    X_train, X_test, y_train, y_test = split_data(df_engineered, target_col='Risk')
    
    # Step 5: Initialize Preprocessing Pipelines & Train Models
    logger.info("[STEP 5/6] Fitting preprocessor and training models...")
    categorical_cols, numerical_cols = get_feature_lists()
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    
    # Train models (bundled with preprocessing into single pipelines) and save to disk
    trained_pipelines = train_and_save_models(X_train, y_train, preprocessor)
    
    # Step 6: Model Evaluation
    logger.info("[STEP 6/6] Evaluating models on test dataset...")
    metrics_summary = evaluate_models(trained_pipelines, X_test, y_test)
    
    logger.info("=" * 60)
    logger.info("CREDIT SCORING PIPELINE RUN COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    return metrics_summary

if __name__ == "__main__":
    run_pipeline()

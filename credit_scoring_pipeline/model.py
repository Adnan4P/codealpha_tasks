import os
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

MODELS_DIR = "models"

def get_models(random_state=42):
    """
    Returns a dictionary of classification models with balanced class weights.
    """
    models = {
        'Logistic_Regression': LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=random_state
        ),
        'Decision_Tree': DecisionTreeClassifier(
            class_weight='balanced', 
            max_depth=5, 
            min_samples_split=10, 
            random_state=random_state
        ),
        'Random_Forest': RandomForestClassifier(
            class_weight='balanced', 
            n_estimators=200, 
            max_depth=8, 
            min_samples_split=5,
            random_state=random_state
        )
    }
    return models

def train_and_save_models(X_train, y_train, preprocessor, random_state=42):
    """
    Trains each model, compiles them with the preprocessor pipeline, and saves them.
    Saving the entire pipeline (preprocessor + model) makes predictions on raw inputs easy.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = get_models(random_state)
    trained_pipelines = {}
    
    # Pre-process the training features
    logger.info("Fitting feature preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        model.fit(X_train_processed, y_train)
        
        # Bundle preprocessor and model together in a single Pipeline
        from sklearn.pipeline import Pipeline
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        model_path = os.path.join(MODELS_DIR, f"{name.lower()}_pipeline.joblib")
        joblib.dump(full_pipeline, model_path)
        logger.info(f"Saved complete pipeline for {name} to {model_path}")
        
        trained_pipelines[name] = full_pipeline
        
    return trained_pipelines

def load_saved_model(model_name):
    """
    Loads a saved pipeline model from the models directory.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_pipeline.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")
    logger.info(f"Loading model pipeline from {model_path}")
    return joblib.load(model_path)

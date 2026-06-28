import os
import json
import joblib
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

# Set plotting style for modern visuals
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.facecolor': '#1a1a24',
    'axes.facecolor': '#1a1a24',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#cbd5e1',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'grid.color': '#334155',
    'font.family': 'sans-serif'
})

# Define paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
CHARTS_DIR = os.path.join('static', 'images', 'charts')

# Create directories if they don't exist
for d in [DATA_DIR, MODELS_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Datasets raw URLs
HEART_DISEASE_URL = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
DIABETES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

def download_data():
    """Download CSV files if they are not cached locally."""
    heart_path = os.path.join(DATA_DIR, 'heart_disease.csv')
    diabetes_path = os.path.join(DATA_DIR, 'diabetes.csv')
    
    if not os.path.exists(heart_path):
        print(f"Downloading Heart Disease dataset from {HEART_DISEASE_URL}...")
        urllib.request.urlretrieve(HEART_DISEASE_URL, heart_path)
        print("Heart Disease dataset downloaded.")
        
    if not os.path.exists(diabetes_path):
        print(f"Downloading Diabetes dataset from {DIABETES_URL}...")
        urllib.request.urlretrieve(DIABETES_URL, diabetes_path)
        print("Diabetes dataset downloaded.")
        
    return heart_path, diabetes_path

def load_and_preprocess_heart(csv_path):
    """Load and preprocess Heart Disease dataset."""
    df = pd.read_csv(csv_path)
    # Ensure no missing values (replace with median if any)
    df = df.fillna(df.median())
    
    # Feature columns (all except target)
    X = df.drop(columns=['target'])
    y = 1 - df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'heart_disease_scaler.joblib'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

def load_and_preprocess_diabetes(csv_path):
    """Load and preprocess Diabetes dataset."""
    df = pd.read_csv(csv_path)
    # Ensure no missing values
    df = df.fillna(df.median())
    
    # Features and target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'diabetes_scaler.joblib'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

def load_and_preprocess_breast_cancer():
    """Load and preprocess Breast Cancer dataset (scikit-learn built-in)."""
    data = load_breast_cancer()
    df_full = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Use only first 10 columns (mean features) for simplified inputs & clean UI
    feature_cols = [name for name in data.feature_names if 'mean' in name][:10]
    X = df_full[feature_cols]
    y = 1 - data.target
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'breast_cancer_scaler.joblib'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

def train_and_evaluate(X_train, X_test, y_train, y_test, disease_name):
    """Train 4 models and return metrics & trained instances."""
    models = {
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name} for {disease_name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5
            
        results[name] = {
            'model_object': model,
            'metrics': {
                'Accuracy': float(acc),
                'Precision': float(prec),
                'Recall': float(rec),
                'F1-Score': float(f1),
                'ROC-AUC': float(auc)
            },
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
    return results

def generate_plots(results, X_train, y_test, disease_key, feature_names):
    """Generate ROC curves, Confusion Matrix, and Feature Importance plots."""
    # 1. ROC Curves Plot
    plt.figure(figsize=(8, 6), dpi=150)
    colors = {'SVM': '#a855f7', 'Logistic Regression': '#06b6d4', 'Random Forest': '#f97316', 'XGBoost': '#10b981'}
    
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        auc_val = res['metrics']['ROC-AUC']
        plt.plot(fpr, tpr, color=colors.get(name, '#64748b'), lw=2, label=f"{name} (AUC = {auc_val:.3f})")
        
    plt.plot([0, 1], [0, 1], color='#475569', linestyle='--', lw=1.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold', labelpad=10)
    plt.title(f'ROC Curves - {disease_key.replace("_", " ").title()}', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc='lower right', frameon=True, facecolor='#1e293b', edgecolor='#334155')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, f'{disease_key}_roc.png'), facecolor='#1a1a24', edgecolor='none')
    plt.close()
    
    # Find the best model based on Accuracy
    best_name = max(results, key=lambda k: results[k]['metrics']['Accuracy'])
    best_res = results[best_name]
    best_model = best_res['model_object']
    
    # 2. Confusion Matrix Plot for Best Model
    plt.figure(figsize=(6, 5), dpi=150)
    cm = confusion_matrix(y_test, best_res['y_pred'])
    
    # Custom color palette for the CM
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=sns.color_palette("rocket", as_cmap=True),
        xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
        cbar=False, annot_kws={"size": 13, "weight": "bold"}
    )
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel('True Label', fontsize=11, fontweight='bold', labelpad=10)
    plt.title(f'Confusion Matrix ({best_name})\n{disease_key.replace("_", " ").title()}', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, f'{disease_key}_cm.png'), facecolor='#1a1a24', edgecolor='none')
    plt.close()
    
    # 3. Feature Importance Plot
    plt.figure(figsize=(8, 5), dpi=150)
    importances = None
    
    # Extract importances/coefficients based on model type
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        title_suffix = "Feature Importance"
    elif hasattr(best_model, 'coef_'):
        # For multi-class or single coefficients
        if best_model.coef_.ndim > 1:
            importances = np.abs(best_model.coef_[0])
        else:
            importances = np.abs(best_model.coef_)
        title_suffix = "Feature Coefficients (Absolute)"
        
    if importances is not None:
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
        # Select top 10 features if there are more
        feat_imp = feat_imp.tail(10)
        
        feat_imp.plot(kind='barh', color='#8b5cf6', edgecolor='#a78bfa', width=0.6)
        plt.title(f'{title_suffix} ({best_name})\n{disease_key.replace("_", " ").title()}', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Importance Score', fontsize=11, fontweight='bold', labelpad=10)
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, f'{disease_key}_feat_imp.png'), facecolor='#1a1a24', edgecolor='none')
        plt.close()
        importances_list = [float(val) for val in importances]
    else:
        # If best model doesn't support coefficients/importances directly (like SVM with RBF kernel),
        # try using Random Forest for feature importance anyway as a proxy to show users
        print(f"Model {best_name} has no importances/coefficients. Generating proxy importance using Random Forest.")
        rf = results['Random Forest']['model_object']
        importances = rf.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(10)
        feat_imp.plot(kind='barh', color='#f59e0b', edgecolor='#fbbf24', width=0.6)
        plt.title(f'Feature Importance (Proxy via Random Forest)\n{disease_key.replace("_", " ").title()}', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Importance Score', fontsize=11, fontweight='bold', labelpad=10)
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, f'{disease_key}_feat_imp.png'), facecolor='#1a1a24', edgecolor='none')
        plt.close()
        importances_list = [float(val) for val in importances]

    return best_name, best_model, importances_list

def main():
    # 1. Download datasets
    heart_path, diabetes_path = download_data()
    
    # 2. Load & preprocess
    print("\nLoading and preprocessing datasets...")
    datasets = {
        'heart_disease': load_and_preprocess_heart(heart_path),
        'diabetes': load_and_preprocess_diabetes(diabetes_path),
        'breast_cancer': load_and_preprocess_breast_cancer()
    }
    
    metrics_summary = {}
    
    # 3. Train, Evaluate and Save
    for key, (X_train, X_test, y_train, y_test, feature_names) in datasets.items():
        print(f"\n--- Processing {key.upper()} ---")
        
        # Train & Evaluate all models
        results = train_and_evaluate(X_train, X_test, y_train, y_test, key)
        
        # Generate diagnostic plots & Find the best model
        best_name, best_model, importances_list = generate_plots(results, X_train, y_test, key, feature_names)
        print(f"Best model for {key}: {best_name}")
        
        # Save the best model
        model_filename = os.path.join(MODELS_DIR, f'{key}_model.joblib')
        joblib.dump(best_model, model_filename)
        print(f"Saved best model to {model_filename}")
        
        # Record metrics for the summary JSON
        metrics_summary[key] = {
            'best_model': best_name,
            'features': feature_names,
            'importances': importances_list,
            'models': {name: res['metrics'] for name, res in results.items()}
        }
        
    # Write metrics summary to JSON
    summary_path = os.path.join(MODELS_DIR, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nMetrics summary saved to {summary_path}")
    print("\nAll models trained and ready!")

if __name__ == '__main__':
    main()

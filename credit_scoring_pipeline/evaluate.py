import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for background task rendering
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Computes classification evaluation metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1_Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_true, y_prob)
    }
    return metrics

def evaluate_models(trained_pipelines, X_test, y_test):
    """
    Evaluates all trained model pipelines on the test set,
    prints a comparison table, and generates comparison plots.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results = {}
    curves = {}
    conf_matrices = {}
    
    for name, pipeline in trained_pipelines.items():
        logger.info(f"Evaluating {name} on test set...")
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        # Probability for the positive class (1 = Bad Risk)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Get metrics
        results[name] = calculate_metrics(y_test, y_pred, y_prob)
        
        # Store for plotting
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        curves[name] = (fpr, tpr, results[name]['ROC_AUC'])
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
        
    # Convert metrics dictionary to a DataFrame for clean formatting
    metrics_df = pd.DataFrame(results).T
    logger.info("Evaluation Complete. Performance Summary:\n" + metrics_df.to_string())
    
    # Save the table to a CSV report
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "performance_metrics.csv"))
    
    # Generate Plots
    plot_roc_curves(curves)
    plot_confusion_matrices(conf_matrices)
    
    return metrics_df

def plot_roc_curves(curves):
    """
    Plots the ROC curves for all models in a single clean figure.
    """
    plt.figure(figsize=(8, 6))
    
    # Sleek palette colors
    colors = {
        'Logistic_Regression': '#2b7bba',
        'Decision_Tree': '#e67e22',
        'Random_Forest': '#2ecc71'
    }
    
    for name, (fpr, tpr, auc_val) in curves.items():
        color = colors.get(name, None)
        plt.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC = {auc_val:.3f})", linewidth=2.5, color=color)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guess')
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, labelpad=10)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "roc_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve comparison plot to {plot_path}")

def plot_confusion_matrices(conf_matrices):
    """
    Plots confusion matrices side-by-side in a 1x3 grid.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    names = list(conf_matrices.keys())
    cmaps = {
        'Logistic_Regression': 'Blues',
        'Decision_Tree': 'Oranges',
        'Random_Forest': 'Greens'
    }
    
    for i, name in enumerate(names):
        ax = axes[i]
        cm = conf_matrices[name]
        cmap = cmaps.get(name, 'Blues')
        
        # Plot heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False,
            xticklabels=['Good Risk (0)', 'Bad Risk (1)'],
            yticklabels=['Good Risk (0)', 'Bad Risk (1)'],
            annot_kws={'size': 14, 'weight': 'bold'}
        )
        
        # Calculate some summary figures for annotations
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        ax.set_title(f"{name.replace('_', ' ')}", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
        if i == 0:
            ax.set_ylabel('True Label', fontsize=11, labelpad=8)
        else:
            ax.set_ylabel('')
            
        # Add summary stats under each plot
        ax.text(
            0.5, -0.2, 
            f"Recall (Bad Risk Identified): {recall:.1%}\nPrecision: {precision:.1%}",
            transform=ax.transAxes, ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        )

    plt.tight_layout()
    # Adjust top and bottom to fit text annotations
    plt.subplots_adjust(bottom=0.25)
    
    plot_path = os.path.join(REPORTS_DIR, "confusion_matrices.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrices plot to {plot_path}")

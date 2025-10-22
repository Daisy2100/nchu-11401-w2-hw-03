"""
Evaluation and Visualization Module for Spam Classification

This module provides functions for evaluating models and creating visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from typing import Dict, Any, Tuple
import os


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for printing
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    print(f"\n{model_name} Evaluation:")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    print(f"\n{model_name} - Detailed Classification Report:")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    model_name: str = "Model",
    save_path: str = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam'],
        ax=ax
    )
    
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    model_name: str = "Model",
    save_path: str = None
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (for positive class)
        model_name: Name of the model
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    return fig


def compare_models(
    results: Dict[str, Dict[str, float]], 
    save_path: str = None
) -> plt.Figure:
    """
    Create bar plot comparing multiple models.
    
    Args:
        results: Dictionary mapping model names to their metrics
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        df[metric].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    return fig


def plot_feature_importance(
    vectorizer, 
    model, 
    top_n: int = 20, 
    save_path: str = None
) -> plt.Figure:
    """
    Plot top features (words) most indicative of spam.
    
    Args:
        vectorizer: Fitted TfidfVectorizer
        model: Trained model with feature importance or coefficients
        top_n: Number of top features to display
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Get feature names
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get feature importance/coefficients
    if hasattr(model.model, 'coef_'):
        importance = model.model.coef_[0]
    elif hasattr(model.model, 'feature_log_prob_'):
        # For Naive Bayes, use log probability difference
        importance = model.model.feature_log_prob_[1] - model.model.feature_log_prob_[0]
    else:
        print("Model doesn't support feature importance visualization")
        return None
    
    # Get top spam indicators (positive coefficients)
    top_spam_idx = np.argsort(importance)[-top_n:]
    top_spam_features = feature_names[top_spam_idx]
    top_spam_scores = importance[top_spam_idx]
    
    # Get top ham indicators (negative coefficients)
    top_ham_idx = np.argsort(importance)[:top_n]
    top_ham_features = feature_names[top_ham_idx]
    top_ham_scores = importance[top_ham_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot spam features
    ax1.barh(range(top_n), top_spam_scores, color='red', alpha=0.7)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_spam_features)
    ax1.set_xlabel('Importance Score', fontsize=12)
    ax1.set_title(f'Top {top_n} Spam Indicators', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot ham features
    ax2.barh(range(top_n), np.abs(top_ham_scores), color='green', alpha=0.7)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_ham_features)
    ax2.set_xlabel('Importance Score (Absolute)', fontsize=12)
    ax2.set_title(f'Top {top_n} Ham Indicators', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Feature importance saved to {save_path}")
    
    return fig


def create_metrics_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a formatted table of model metrics.
    
    Args:
        results: Dictionary mapping model names to their metrics
        
    Returns:
        DataFrame with formatted metrics
    """
    df = pd.DataFrame(results).T
    df = df.round(4)
    
    # Add percentage column
    for col in df.columns:
        df[f'{col}_pct'] = (df[col] * 100).round(2).astype(str) + '%'
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Testing evaluation module...")
    
    # Create dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100)
    
    # Test evaluation
    metrics = evaluate_model(y_true, y_pred, "Test Model")
    
    # Test confusion matrix
    fig = plot_confusion_matrix(y_true, y_pred, "Test Model")
    plt.close(fig)
    
    print("\nEvaluation module test complete!")

"""
Evaluation metrics for classification and regression
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import pearsonr, spearmanr
from typing import Dict


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: True binary labels
        y_pred_prob: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_prob > threshold).astype(int)
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['sensitivity'] = metrics['recall']  # Same as recall
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC metrics
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_prob)
    except:
        metrics['auc_roc'] = 0.0
    
    try:
        metrics['auc_pr'] = average_precision_score(y_true, y_pred_prob)
    except:
        metrics['auc_pr'] = 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics
    
    Args:
        y_true: True scores
        y_pred: Predicted scores
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Error metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # R-squared
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Correlation metrics
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    metrics['pearson_r'] = pearson_r
    metrics['pearson_p'] = pearson_p
    
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    metrics['spearman_r'] = spearman_r
    metrics['spearman_p'] = spearman_p
    
    return metrics


def compute_all_metrics(
    y_true_class: np.ndarray,
    y_pred_prob: np.ndarray,
    y_true_score: np.ndarray,
    y_pred_score: np.ndarray,
    config: Dict
) -> Dict[str, float]:
    """
    Compute all metrics (classification + regression)
    
    Returns:
        Dictionary of all metrics
    """
    # Classification metrics
    class_metrics = compute_classification_metrics(y_true_class, y_pred_prob)
    
    # Regression metrics
    reg_metrics = compute_regression_metrics(y_true_score, y_pred_score)
    
    # Combine
    all_metrics = {**class_metrics, **reg_metrics}
    
    return all_metrics


def print_metrics(metrics: Dict):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\nClassification Metrics:")
    print("-" * 50)
    if 'accuracy' in metrics:
        print(f"  Accuracy:      {metrics['accuracy']:.3f}")
    if 'sensitivity' in metrics:
        print(f"  Sensitivity:   {metrics['sensitivity']:.3f}")
    if 'specificity' in metrics:
        print(f"  Specificity:   {metrics['specificity']:.3f}")
    if 'precision' in metrics:
        print(f"  Precision:     {metrics['precision']:.3f}")
    if 'f1_score' in metrics:
        print(f"  F1 Score:      {metrics['f1_score']:.3f}")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:       {metrics['auc_roc']:.3f}")
    if 'auc_pr' in metrics:
        print(f"  AUC-PR:        {metrics['auc_pr']:.3f}")
    
    print("\nRegression Metrics:")
    print("-" * 50)
    if 'mae' in metrics:
        print(f"  MAE:           {metrics['mae']:.3f}")
    if 'rmse' in metrics:
        print(f"  RMSE:          {metrics['rmse']:.3f}")
    if 'r2' in metrics:
        print(f"  R²:            {metrics['r2']:.3f}")
    if 'pearson_r' in metrics:
        print(f"  Pearson r:     {metrics['pearson_r']:.3f} (p={metrics.get('pearson_p', 0):.4f})")
    if 'spearman_r' in metrics:
        print(f"  Spearman ρ:    {metrics['spearman_r']:.3f} (p={metrics.get('spearman_p', 0):.4f})")
    
    if 'confusion_matrix' in metrics:
        print("\nConfusion Matrix:")
        print("-" * 50)
        cm = metrics['confusion_matrix']
        print(f"  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
        print(f"  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

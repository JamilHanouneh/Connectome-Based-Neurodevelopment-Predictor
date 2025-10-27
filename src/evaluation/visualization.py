"""
Visualization utilities for evaluation results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def plot_roc_curves(
    predictions: Dict,
    labels: Dict,
    outcomes: List[str],
    output_dir: Path,
    config: Dict
):
    """
    Plot ROC curves for all outcomes
    """
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6*len(outcomes), 5))
    
    if len(outcomes) == 1:
        axes = [axes]
    
    for idx, outcome in enumerate(outcomes):
        y_true = labels[outcome]['class']
        y_pred_prob = predictions[outcome]['probs'].flatten()
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        axes[idx].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')
        axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        axes[idx].set_xlabel('False Positive Rate', fontsize=12)
        axes[idx].set_ylabel('True Positive Rate', fontsize=12)
        axes[idx].set_title(f'{outcome.capitalize()} Outcome\nROC Curve', fontsize=14, fontweight='bold')
        axes[idx].legend(loc='lower right', fontsize=11)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 1])
        axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    output_file = output_dir / 'roc_curves.png'
    plt.savefig(output_file, dpi=config['visualization']['dpi'], bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(
    predictions: Dict,
    labels: Dict,
    outcomes: List[str],
    output_dir: Path,
    config: Dict
):
    """
    Plot confusion matrices for all outcomes
    """
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6*len(outcomes), 5))
    
    if len(outcomes) == 1:
        axes = [axes]
    
    for idx, outcome in enumerate(outcomes):
        y_true = labels[outcome]['class']
        y_pred = (predictions[outcome]['probs'].flatten() > 0.5).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[idx],
            cbar=True,
            square=True,
            xticklabels=['Low-Risk', 'High-Risk'],
            yticklabels=['Low-Risk', 'High-Risk']
        )
        
        axes[idx].set_xlabel('Predicted Label', fontsize=12)
        axes[idx].set_ylabel('True Label', fontsize=12)
        axes[idx].set_title(f'{outcome.capitalize()} Outcome\nConfusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'confusion_matrices.png'
    plt.savefig(output_file, dpi=config['visualization']['dpi'], bbox_inches='tight')
    plt.close()


def plot_scatter_plots(
    predictions: Dict,
    labels: Dict,
    outcomes: List[str],
    output_dir: Path,
    config: Dict
):
    """
    Plot scatter plots of predicted vs true scores
    """
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6*len(outcomes), 5))
    
    if len(outcomes) == 1:
        axes = [axes]
    
    for idx, outcome in enumerate(outcomes):
        y_true = labels[outcome]['score']
        y_pred = predictions[outcome]['scores'].flatten()
        
        # Compute correlation
        from scipy.stats import pearsonr
        r, p = pearsonr(y_true, y_pred)
        
        # Plot
        axes[idx].scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        # Add diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p_line = np.poly1d(z)
        axes[idx].plot(y_true, p_line(y_true), 'b-', linewidth=2, alpha=0.8, label='Regression line')
        
        axes[idx].set_xlabel('True Bayley-III Score', fontsize=12)
        axes[idx].set_ylabel('Predicted Bayley-III Score', fontsize=12)
        axes[idx].set_title(f'{outcome.capitalize()} Outcome\nr = {r:.3f}, p = {p:.4f}', fontsize=14, fontweight='bold')
        axes[idx].legend(loc='upper left', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'prediction_scatter.png'
    plt.savefig(output_file, dpi=config['visualization']['dpi'], bbox_inches='tight')
    plt.close()


def plot_bland_altman(
    predictions: Dict,
    labels: Dict,
    outcomes: List[str],
    output_dir: Path,
    config: Dict
):
    """
    Plot Bland-Altman plots for agreement analysis
    """
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6*len(outcomes), 5))
    
    if len(outcomes) == 1:
        axes = [axes]
    
    for idx, outcome in enumerate(outcomes):
        y_true = labels[outcome]['score']
        y_pred = predictions[outcome]['scores'].flatten()
        
        # Compute mean and difference
        mean = (y_true + y_pred) / 2
        diff = y_pred - y_true
        
        # Compute statistics
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        # Plot
        axes[idx].scatter(mean, diff, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        axes[idx].axhline(mean_diff, color='blue', linestyle='-', linewidth=2, label=f'Mean = {mean_diff:.2f}')
        axes[idx].axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', linewidth=2, label=f'+1.96 SD = {mean_diff + 1.96*std_diff:.2f}')
        axes[idx].axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', linewidth=2, label=f'-1.96 SD = {mean_diff - 1.96*std_diff:.2f}')
        
        axes[idx].set_xlabel('Mean of True and Predicted Score', fontsize=12)
        axes[idx].set_ylabel('Difference (Predicted - True)', fontsize=12)
        axes[idx].set_title(f'{outcome.capitalize()} Outcome\nBland-Altman Plot', fontsize=14, fontweight='bold')
        axes[idx].legend(loc='upper left', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'bland_altman.png'
    plt.savefig(output_file, dpi=config['visualization']['dpi'], bbox_inches='tight')
    plt.close()


def generate_html_report(
    all_metrics: Dict,
    predictions: Dict,
    labels: Dict,
    outcomes: List[str],
    output_dir: Path,
    config: Dict,
    timestamp: str
) -> Path:
    """
    Generate HTML report with all results
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neurodevelopmental Outcome Prediction - Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-bottom: 2px solid #bdc3c7;
                padding-bottom: 5px;
            }}
            h3 {{
                color: #7f8c8d;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .metric-value {{
                font-weight: bold;
                color: #2980b9;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            .summary {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Neurodevelopmental Outcome Prediction</h1>
            <p class="timestamp">Evaluation Report - {timestamp}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>This report presents the evaluation results of the multimodal deep learning model
                for predicting neurodevelopmental outcomes in very preterm infants.</p>
                <ul>
                    <li><strong>Test Samples:</strong> {len(labels[outcomes[0]]['class'])}</li>
                    <li><strong>Outcomes Evaluated:</strong> {', '.join([o.capitalize() for o in outcomes])}</li>
                    <li><strong>Model Architecture:</strong> 4-channel VGG-19 based multimodal network</li>
                </ul>
            </div>
    """
    
    # Add metrics for each outcome
    for outcome in outcomes:
        metrics = all_metrics[outcome]
        
        html_content += f"""
            <h2>{outcome.capitalize()} Outcome</h2>
            
            <h3>Classification Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td class="metric-value">{metrics['accuracy']:.3f}</td>
                </tr>
                <tr>
                    <td>Sensitivity (Recall)</td>
                    <td class="metric-value">{metrics['sensitivity']:.3f}</td>
                </tr>
                <tr>
                    <td>Specificity</td>
                    <td class="metric-value">{metrics['specificity']:.3f}</td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td class="metric-value">{metrics['precision']:.3f}</td>
                </tr>
                <tr>
                    <td>F1 Score</td>
                    <td class="metric-value">{metrics['f1_score']:.3f}</td>
                </tr>
                <tr>
                    <td>AUC-ROC</td>
                    <td class="metric-value">{metrics['auc_roc']:.3f}</td>
                </tr>
                <tr>
                    <td>AUC-PR</td>
                    <td class="metric-value">{metrics['auc_pr']:.3f}</td>
                </tr>
            </table>
            
            <h3>Regression Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Mean Absolute Error (MAE)</td>
                    <td class="metric-value">{metrics['mae']:.3f}</td>
                </tr>
                <tr>
                    <td>Root Mean Squared Error (RMSE)</td>
                    <td class="metric-value">{metrics['rmse']:.3f}</td>
                </tr>
                <tr>
                    <td>R² Score</td>
                    <td class="metric-value">{metrics['r2']:.3f}</td>
                </tr>
                <tr>
                    <td>Pearson Correlation (r)</td>
                    <td class="metric-value">{metrics['pearson_r']:.3f} (p={metrics['pearson_p']:.4f})</td>
                </tr>
                <tr>
                    <td>Spearman Correlation (ρ)</td>
                    <td class="metric-value">{metrics['spearman_r']:.3f} (p={metrics['spearman_p']:.4f})</td>
                </tr>
            </table>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_file = output_dir / f'evaluation_report_{timestamp}.html'
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    return report_file

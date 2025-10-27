#!/usr/bin/env python3
"""
Testing/Evaluation script for Neurodevelopmental Outcome Predictor
Evaluates trained model on test set
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.logging_utils import setup_logger
from src.data.data_loader import create_dataloaders
from src.models.multimodal_network import MultimodalNetwork
from src.evaluation.metrics import compute_all_metrics, print_metrics
from src.evaluation.visualization import (
    plot_roc_curves, plot_confusion_matrices, plot_scatter_plots,
    plot_bland_altman, generate_html_report
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate Neurodevelopmental Outcome Predictor"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/predictions",
        help="Directory to save predictions and metrics"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (overrides config)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_device(config: dict, args) -> torch.device:
    """Setup compute device"""
    device_str = args.device if args.device else config['training']['device']
    
    if device_str == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA requested but not available, using CPU")
        device_str = "cpu"
    elif device_str == "mps" and not (hasattr(torch.backends, 'mps') and 
                                       torch.backends.mps.is_available()):
        print("⚠ MPS requested but not available, using CPU")
        device_str = "cpu"
    
    return torch.device(device_str)


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model
    model = MultimodalNetwork(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model


@torch.no_grad()
def evaluate_model(model, test_loader, config, device, logger):
    """Evaluate model on test set"""
    model.eval()
    
    outcomes = config['model']['outcomes']
    all_predictions = {outcome: {'probs': [], 'scores': []} for outcome in outcomes}
    all_labels = {outcome: {'class': [], 'score': []} for outcome in outcomes}
    
    print("\nRunning inference on test set...")
    logger.info("Starting test set evaluation")
    
    from tqdm import tqdm
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        # Move data to device
        func_conn = batch['functional_connectivity'].to(device)
        struct_conn = batch['structural_connectivity'].to(device)
        dwma = batch['dwma_features'].to(device)
        clinical = batch['clinical_features'].to(device)
        
        # Forward pass
        outputs = model(func_conn, struct_conn, dwma, clinical)
        
        # Store predictions and labels
        for outcome in outcomes:
            # Classification predictions
            probs = torch.sigmoid(outputs[f'{outcome}_class']).cpu().numpy()
            all_predictions[outcome]['probs'].extend(probs)
            
            # Regression predictions
            scores = outputs[f'{outcome}_reg'].cpu().numpy()
            all_predictions[outcome]['scores'].extend(scores)
            
            # Labels
            class_labels = batch[f'{outcome}_class_label'].cpu().numpy()
            score_labels = batch[f'{outcome}_score_label'].cpu().numpy()
            all_labels[outcome]['class'].extend(class_labels)
            all_labels[outcome]['score'].extend(score_labels)
    
    # Convert to numpy arrays
    for outcome in outcomes:
        for key in ['probs', 'scores']:
            all_predictions[outcome][key] = np.array(all_predictions[outcome][key])
        for key in ['class', 'score']:
            all_labels[outcome][key] = np.array(all_labels[outcome][key])
    
    logger.info(f"Evaluated {len(test_loader.dataset)} test samples")
    
    return all_predictions, all_labels


def save_predictions(predictions, labels, outcomes, output_dir):
    """Save predictions to CSV files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame for each outcome
    for outcome in outcomes:
        df = pd.DataFrame({
            'subject_id': range(len(predictions[outcome]['probs'])),
            f'{outcome}_high_risk_probability': predictions[outcome]['probs'].flatten(),
            f'{outcome}_predicted_score': predictions[outcome]['scores'].flatten(),
            f'{outcome}_true_class': labels[outcome]['class'],
            f'{outcome}_true_score': labels[outcome]['score'],
            f'{outcome}_predicted_class': (predictions[outcome]['probs'].flatten() > 0.5).astype(int),
        })
        
        output_file = output_dir / f'{outcome}_predictions.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ Saved predictions: {output_file}")
    
    # Save combined predictions
    combined_data = {'subject_id': range(len(predictions[outcomes[0]]['probs']))}
    
    for outcome in outcomes:
        combined_data.update({
            f'{outcome}_high_risk_prob': predictions[outcome]['probs'].flatten(),
            f'{outcome}_pred_score': predictions[outcome]['scores'].flatten(),
            f'{outcome}_true_class': labels[outcome]['class'],
            f'{outcome}_true_score': labels[outcome]['score'],
        })
    
    combined_df = pd.DataFrame(combined_data)
    combined_file = output_dir / 'all_predictions.csv'
    combined_df.to_csv(combined_file, index=False)
    print(f"✓ Saved combined predictions: {combined_file}")


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("NEURODEVELOPMENTAL OUTCOME PREDICTOR - EVALUATION".center(70))
    print("="*70 + "\n")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(config, args)
    print(f"✓ Using device: {device}")
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(config['output']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"testing_{timestamp}.log"
    logger = setup_logger("testing", log_file, config['output']['log_level'])
    
    logger.info("="*70)
    logger.info("Testing Started")
    logger.info("="*70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    
    try:
        # Create data loaders
        print("\nPreparing data loaders...")
        _, _, test_loader = create_dataloaders(config, logger)
        print(f"✓ Test samples: {len(test_loader.dataset)}")
        
        # Load model
        print("\nLoading model...")
        model = load_model(args.checkpoint, config, device)
        
        # Evaluate model
        predictions, labels = evaluate_model(model, test_loader, config, device, logger)
        
        outcomes = config['model']['outcomes']
        
        # Compute metrics
        print("\n" + "="*70)
        print("COMPUTING METRICS".center(70))
        print("="*70 + "\n")
        
        all_metrics = {}
        for outcome in outcomes:
            print(f"\n{outcome.upper()} OUTCOME:")
            print("-" * 50)
            
            metrics = compute_all_metrics(
                y_true_class=labels[outcome]['class'],
                y_pred_prob=predictions[outcome]['probs'].flatten(),
                y_true_score=labels[outcome]['score'],
                y_pred_score=predictions[outcome]['scores'].flatten(),
                config=config
            )
            
            all_metrics[outcome] = metrics
            print_metrics(metrics)
            
            logger.info(f"\n{outcome.upper()} metrics:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric_name}: {value:.4f}")
        
        # Save predictions
        print("\n" + "="*70)
        print("SAVING RESULTS".center(70))
        print("="*70 + "\n")
        
        output_dir = Path(args.output_dir)
        save_predictions(predictions, labels, outcomes, output_dir)
        
        # Save metrics
        metrics_file = output_dir / 'test_metrics.csv'
        metrics_data = []
        for outcome in outcomes:
            for metric_name, value in all_metrics[outcome].items():
                if isinstance(value, (int, float)):
                    metrics_data.append({
                        'outcome': outcome,
                        'metric': metric_name,
                        'value': value
                    })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(metrics_file, index=False)
        print(f"✓ Saved metrics: {metrics_file}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        figure_dir = Path(config['output']['figure_dir'])
        figure_dir.mkdir(parents=True, exist_ok=True)
        
        # ROC curves
        plot_roc_curves(predictions, labels, outcomes, figure_dir, config)
        print(f"✓ ROC curves saved")
        
        # Confusion matrices
        plot_confusion_matrices(predictions, labels, outcomes, figure_dir, config)
        print(f"✓ Confusion matrices saved")
        
        # Scatter plots
        plot_scatter_plots(predictions, labels, outcomes, figure_dir, config)
        print(f"✓ Scatter plots saved")
        
        # Bland-Altman plots
        plot_bland_altman(predictions, labels, outcomes, figure_dir, config)
        print(f"✓ Bland-Altman plots saved")
        
        # Generate HTML report
        if config['output']['generate_html_report']:
            print("\nGenerating HTML report...")
            report_dir = Path(config['output']['report_dir'])
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = generate_html_report(
                all_metrics, predictions, labels, outcomes, 
                report_dir, config, timestamp
            )
            print(f"✓ HTML report: {report_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION COMPLETE".center(70))
        print("="*70 + "\n")
        
        print("Summary:")
        print(f"  Test samples: {len(test_loader.dataset)}")
        print(f"  Predictions saved: {output_dir}/")
        print(f"  Figures saved: {figure_dir}/")
        print(f"  Metrics saved: {metrics_file}")
        
        print("\n✓ Evaluation complete!")
        
        logger.info("\n" + "="*70)
        logger.info("Testing completed successfully")
        logger.info("="*70)
        
    except Exception as e:
        print(f"\n✗ Evaluation failed with error: {str(e)}")
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

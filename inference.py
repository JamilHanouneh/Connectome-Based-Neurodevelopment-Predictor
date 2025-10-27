#!/usr/bin/env python3
"""
Inference script for Neurodevelopmental Outcome Predictor
Makes predictions on new subjects
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
from src.models.multimodal_network import MultimodalNetwork
from src.utils.gradcam import generate_gradcam_visualization


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run inference with Neurodevelopmental Outcome Predictor"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/synthetic",
        help="Directory containing input data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/predictions",
        help="Directory to save predictions"
    )
    
    parser.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="Specific subject ID to process (processes all if not specified)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use"
    )
    
    parser.add_argument(
        "--generate_gradcam",
        action="store_true",
        help="Generate Grad-CAM visualizations"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = MultimodalNetwork(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def load_subject_data(subject_id: str, input_dir: Path, config: dict):
    """Load data for a single subject"""
    # In a real implementation, this would load actual MRI data
    # For now, we'll generate synthetic data as an example
    
    func_conn_size = config['model']['functional_connectivity_input_size']
    struct_conn_size = config['model']['structural_connectivity_input_size']
    dwma_size = config['model']['dwma_input_size']
    clinical_size = config['model']['clinical_input_size']
    
    # Generate synthetic data (replace with real data loading)
    np.random.seed(hash(subject_id) % (2**32))
    
    functional_connectivity = np.random.randn(1, func_conn_size[0], func_conn_size[1]).astype(np.float32)
    functional_connectivity = (functional_connectivity + functional_connectivity.transpose(0, 2, 1)) / 2
    
    structural_connectivity = np.random.randn(1, struct_conn_size[0], struct_conn_size[1]).astype(np.float32)
    structural_connectivity = (structural_connectivity + structural_connectivity.transpose(0, 2, 1)) / 2
    structural_connectivity = np.abs(structural_connectivity)
    
    dwma_features = np.random.randn(1, dwma_size).astype(np.float32)
    clinical_features = np.random.randn(1, clinical_size).astype(np.float32)
    
    return {
        'functional_connectivity': torch.from_numpy(functional_connectivity).unsqueeze(1),
        'structural_connectivity': torch.from_numpy(structural_connectivity).unsqueeze(1),
        'dwma_features': torch.from_numpy(dwma_features),
        'clinical_features': torch.from_numpy(clinical_features)
    }


@torch.no_grad()
def predict_subject(model, subject_data, config, device):
    """Make predictions for a single subject"""
    model.eval()
    
    # Move data to device
    func_conn = subject_data['functional_connectivity'].to(device)
    struct_conn = subject_data['structural_connectivity'].to(device)
    dwma = subject_data['dwma_features'].to(device)
    clinical = subject_data['clinical_features'].to(device)
    
    # Forward pass
    outputs = model(func_conn, struct_conn, dwma, clinical)
    
    # Process predictions
    outcomes = config['model']['outcomes']
    predictions = {}
    
    for outcome in outcomes:
        # Classification probability
        prob = torch.sigmoid(outputs[f'{outcome}_class']).cpu().item()
        
        # Risk category
        threshold = config['model']['task_types'][outcome]['classification_threshold']
        risk_category = "High-Risk" if prob > 0.5 else "Low-Risk"
        
        # Predicted score
        pred_score = outputs[f'{outcome}_reg'].cpu().item()
        
        # Interpretation
        if pred_score < 70:
            interpretation = "Severe delay"
        elif pred_score < 85:
            interpretation = "Moderate delay"
        elif pred_score < 100:
            interpretation = "Low-average"
        elif pred_score < 115:
            interpretation = "Average"
        else:
            interpretation = "Above average"
        
        predictions[outcome] = {
            'high_risk_probability': prob,
            'risk_category': risk_category,
            'predicted_score': pred_score,
            'interpretation': interpretation
        }
    
    return predictions


def save_predictions(subject_id: str, predictions: dict, output_dir: Path, timestamp: str):
    """Save predictions for a subject"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    data = {'subject_id': [subject_id]}
    
    for outcome, preds in predictions.items():
        data[f'{outcome}_risk_category'] = [preds['risk_category']]
        data[f'{outcome}_high_risk_prob'] = [f"{preds['high_risk_probability']:.3f}"]
        data[f'{outcome}_predicted_score'] = [f"{preds['predicted_score']:.1f}"]
        data[f'{outcome}_interpretation'] = [preds['interpretation']]
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = output_dir / f"{subject_id}_predictions_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    return output_file


def print_predictions(subject_id: str, predictions: dict):
    """Print predictions in a formatted way"""
    print("\n" + "="*70)
    print(f"PREDICTIONS FOR SUBJECT: {subject_id}".center(70))
    print("="*70 + "\n")
    
    for outcome, preds in predictions.items():
        print(f"{outcome.upper()} OUTCOME:")
        print("-" * 50)
        print(f"  Risk Category:          {preds['risk_category']}")
        print(f"  High-Risk Probability:  {preds['high_risk_probability']:.1%}")
        print(f"  Predicted Score:        {preds['predicted_score']:.1f}")
        print(f"  Interpretation:         {preds['interpretation']}")
        print()


def main():
    """Main inference function"""
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("NEURODEVELOPMENTAL OUTCOME PREDICTOR - INFERENCE".center(70))
    print("="*70 + "\n")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device)
    print(f"✓ Using device: {device}")
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(config['output']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"inference_{timestamp}.log"
    logger = setup_logger("inference", log_file, config['output']['log_level'])
    
    logger.info("="*70)
    logger.info("Inference Started")
    logger.info("="*70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load model
        print("\nLoading model...")
        model = load_model(args.checkpoint, config, device)
        print("✓ Model loaded successfully")
        
        # Determine subjects to process
        if args.subject_id:
            subject_ids = [args.subject_id]
        else:
            # Process all subjects in input directory
            # For synthetic data, create some example subjects
            subject_ids = [f"sub-{i:04d}" for i in range(1, 6)]
        
        print(f"\nProcessing {len(subject_ids)} subject(s)...")
        
        output_dir = Path(args.output_dir)
        all_predictions = []
        
        for subject_id in subject_ids:
            print(f"\nProcessing subject: {subject_id}")
            logger.info(f"Processing subject: {subject_id}")
            
            # Load subject data
            subject_data = load_subject_data(
                subject_id, 
                Path(args.input_dir), 
                config
            )
            
            # Make predictions
            predictions = predict_subject(model, subject_data, config, device)
            
            # Print predictions
            print_predictions(subject_id, predictions)
            
            # Save predictions
            output_file = save_predictions(subject_id, predictions, output_dir, timestamp)
            print(f"✓ Predictions saved: {output_file}")
            
            # Store for summary
            all_predictions.append({
                'subject_id': subject_id,
                **{f'{k}_{sk}': sv for k, v in predictions.items() for sk, sv in v.items()}
            })
            
            # Generate Grad-CAM if requested
            if args.generate_gradcam:
                print("  Generating Grad-CAM visualizations...")
                try:
                    gradcam_dir = output_dir / f"{subject_id}_gradcam"
                    generate_gradcam_visualization(
                        model, subject_data, config, device, gradcam_dir
                    )
                    print(f"  ✓ Grad-CAM saved: {gradcam_dir}/")
                except Exception as e:
                    print(f"  ⚠ Grad-CAM generation failed: {str(e)}")
        
        # Save summary
        if len(all_predictions) > 1:
            summary_df = pd.DataFrame(all_predictions)
            summary_file = output_dir / f"inference_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"\n✓ Summary saved: {summary_file}")
        
        print("\n" + "="*70)
        print("INFERENCE COMPLETE".center(70))
        print("="*70 + "\n")
        
        print(f"✓ Processed {len(subject_ids)} subject(s)")
        print(f"✓ Results saved to: {output_dir}/")
        
        logger.info("Inference completed successfully")
        
    except Exception as e:
        print(f"\n✗ Inference failed with error: {str(e)}")
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

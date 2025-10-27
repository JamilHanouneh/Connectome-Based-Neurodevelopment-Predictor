#!/usr/bin/env python3
"""
Training script for Neurodevelopmental Outcome Predictor
Trains the multimodal deep learning model
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.logging_utils import setup_logger
from src.data.data_loader import create_dataloaders
from src.models.multimodal_network import MultimodalNetwork
from src.training.trainer import Trainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Neurodevelopmental Outcome Predictor"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (overrides config)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training"
    )
    
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a fast development test with 2 batches"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make training deterministic (may slow down training)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_device(config: dict, args) -> torch.device:
    """Setup compute device"""
    # Command line argument overrides config
    device_str = args.device if args.device else config['training']['device']
    
    if device_str == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA requested but not available, using CPU")
        device_str = "cpu"
    elif device_str == "mps" and not (hasattr(torch.backends, 'mps') and 
                                       torch.backends.mps.is_available()):
        print("⚠ MPS requested but not available, using CPU")
        device_str = "cpu"
    
    device = torch.device(device_str)
    
    if device.type == "cuda":
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    elif device.type == "mps":
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        print("✓ Using CPU")
        print("  Note: Training on CPU will be slower")
    
    return device


def create_output_dirs(config: dict):
    """Create output directories"""
    for key in ['checkpoint_dir', 'log_dir', 'figure_dir']:
        Path(config['output'][key]).mkdir(parents=True, exist_ok=True)


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    print("\n" + "="*70)
    print("NEURODEVELOPMENTAL OUTCOME PREDICTOR - TRAINING".center(70))
    print("="*70 + "\n")
    
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.fast_dev_run:
        config['debug']['fast_dev_run'] = True
    
    # Set random seed for reproducibility
    seed = config['reproducibility']['seed']
    print(f"Setting random seed: {seed}")
    set_seed(seed)
    
    # Setup device
    device = setup_device(config, args)
    
    # Create output directories
    create_output_dirs(config)
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = Path(config['output']['log_dir']) / f"training_{timestamp}.log"
    logger = setup_logger("training", log_file, config['output']['log_level'])
    
    logger.info("="*70)
    logger.info("Training Started")
    logger.info("="*70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {seed}")
    
    # Log configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Data mode: {config['data']['mode']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Optimizer: {config['training']['optimizer']}")
    
    try:
        # Create data loaders
        print("\nPreparing data loaders...")
        logger.info("\nCreating data loaders...")
        
        train_loader, val_loader, test_loader = create_dataloaders(config, logger)
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        print(f"✓ Training samples: {len(train_loader.dataset)}")
        print(f"✓ Validation samples: {len(val_loader.dataset)}")
        print(f"✓ Test samples: {len(test_loader.dataset)}")
        
        # Create model
        print("\nInitializing model...")
        logger.info("\nInitializing model...")
        
        model = MultimodalNetwork(config)
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model architecture: {config['model']['architecture']}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        print(f"✓ Model initialized")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Create trainer
        print("\nInitializing trainer...")
        logger.info("\nInitializing trainer...")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            logger=logger
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            print(f"\nResuming from checkpoint: {args.resume}")
            logger.info(f"Resuming from checkpoint: {args.resume}")
            start_epoch = trainer.load_checkpoint(args.resume)
        
        # Train model
        print("\n" + "="*70)
        print("STARTING TRAINING".center(70))
        print("="*70 + "\n")
        
        logger.info("\n" + "="*70)
        logger.info("Starting training loop")
        logger.info("="*70)
        
        trainer.train(start_epoch=start_epoch)
        
        # Training complete
        print("\n" + "="*70)
        print("TRAINING COMPLETE".center(70))
        print("="*70 + "\n")
        
        logger.info("\n" + "="*70)
        logger.info("Training completed successfully")
        logger.info("="*70)
        
        # Print summary
        print("Training Summary:")
        print(f"  Best epoch: {trainer.best_epoch}")
        print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"  Model saved to: {trainer.best_checkpoint_path}")
        print(f"  Training log: {log_file}")
        print(f"  Training curves: {config['output']['figure_dir']}/")
        
        logger.info("\nTraining summary:")
        logger.info(f"  Best epoch: {trainer.best_epoch}")
        logger.info(f"  Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"  Best checkpoint: {trainer.best_checkpoint_path}")
        
        print("\n✓ Training complete!")
        print("\nNext steps:")
        print("  1. Review training curves in outputs/figures/")
        print("  2. Evaluate on test set: python test.py")
        print("  3. Run inference: python inference.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        logger.warning("Training interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {str(e)}")
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

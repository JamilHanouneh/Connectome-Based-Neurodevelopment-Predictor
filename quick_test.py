#!/usr/bin/env python3
"""
Quick validation test for the entire project
Tests all components without requiring real MRI data
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("QUICK PROJECT VALIDATION TEST".center(70))
print("="*70 + "\n")

# Test 1: Import all modules
print("Test 1: Checking imports...")
try:
    from src.data.data_loader import generate_synthetic_data, NeurodevelopmentDataset
    from src.models.multimodal_network import MultimodalNetwork
    from src.models.loss import MultiTaskLoss
    from src.training.trainer import Trainer
    from src.evaluation.metrics import compute_all_metrics
    from src.utils.logging_utils import setup_logger
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {str(e)}")
    sys.exit(1)

# Test 2: Load configuration
print("\nTest 2: Loading configuration...")
try:
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Configuration loaded")
    print(f"  Data mode: {config['data']['mode']}")
    print(f"  Model: {config['model']['architecture']}")
except Exception as e:
    print(f"✗ Configuration loading failed: {str(e)}")
    sys.exit(1)

# Test 3: Generate synthetic data
print("\nTest 3: Generating synthetic data...")
try:
    # Use small dataset for quick test
    config['data']['n_synthetic_subjects'] = 30  # Small for testing
    data = generate_synthetic_data(config, logger=None)
    
    print(f"✓ Synthetic data generated")
    print(f"  Functional connectivity: {data['functional_connectivity'].shape}")
    print(f"  Structural connectivity: {data['structural_connectivity'].shape}")
    print(f"  DWMA features: {data['dwma_features'].shape}")
    print(f"  Clinical features: {data['clinical_features'].shape}")
except Exception as e:
    print(f"✗ Data generation failed: {str(e)}")
    sys.exit(1)

# Test 4: Create dataset
print("\nTest 4: Creating dataset...")
try:
    dataset = NeurodevelopmentDataset(data, config)
    sample = dataset[0]
    
    print(f"✓ Dataset created successfully")
    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Sample keys: {list(sample.keys())}")
except Exception as e:
    print(f"✗ Dataset creation failed: {str(e)}")
    sys.exit(1)

# Test 5: Initialize model
print("\nTest 5: Initializing model...")
try:
    model = MultimodalNetwork(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"✗ Model initialization failed: {str(e)}")
    sys.exit(1)

# Test 6: Forward pass
print("\nTest 6: Testing forward pass...")
try:
    model.eval()
    
    func_conn = sample['functional_connectivity'].unsqueeze(0)
    struct_conn = sample['structural_connectivity'].unsqueeze(0)
    dwma = sample['dwma_features'].unsqueeze(0)
    clinical = sample['clinical_features'].unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(func_conn, struct_conn, dwma, clinical)
    
    print(f"✓ Forward pass successful")
    print(f"  Output keys: {list(outputs.keys())}")
    for key, value in outputs.items():
        print(f"    {key}: {value.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {str(e)}")
    sys.exit(1)

# Test 7: Loss computation
print("\nTest 7: Testing loss computation...")
try:
    # Set model to training mode for loss computation
    model.train()
    
    # Create a small batch
    batch_size = 4
    func_conn = torch.cat([sample['functional_connectivity'].unsqueeze(0) for _ in range(batch_size)])
    struct_conn = torch.cat([sample['structural_connectivity'].unsqueeze(0) for _ in range(batch_size)])
    dwma = torch.cat([sample['dwma_features'].unsqueeze(0) for _ in range(batch_size)])
    clinical = torch.cat([sample['clinical_features'].unsqueeze(0) for _ in range(batch_size)])
    
    # Forward pass (with gradients enabled)
    outputs = model(func_conn, struct_conn, dwma, clinical)
    
    # Create labels
    criterion = MultiTaskLoss(config)
    labels = {}
    for outcome in config['model']['outcomes']:
        labels[f'{outcome}_class'] = torch.cat([sample[f'{outcome}_class_label'].unsqueeze(0) for _ in range(batch_size)])
        labels[f'{outcome}_score'] = torch.cat([sample[f'{outcome}_score_label'].unsqueeze(0) for _ in range(batch_size)])
    
    loss, loss_dict = criterion(outputs, labels)
    
    print(f"✓ Loss computation successful")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Classification losses:")
    for outcome in config['model']['outcomes']:
        print(f"    {outcome}: {loss_dict[f'{outcome}_class_loss']:.4f}")
    print(f"  Regression losses:")
    for outcome in config['model']['outcomes']:
        print(f"    {outcome}: {loss_dict[f'{outcome}_reg_loss']:.4f}")
except Exception as e:
    print(f"✗ Loss computation failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Backward pass
print("\nTest 8: Testing backward pass...")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer.zero_grad()
    
    # Compute loss again to ensure computational graph exists
    outputs = model(func_conn, struct_conn, dwma, clinical)
    loss, _ = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients are computed
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    
    if has_gradients:
        # Update weights
        optimizer.step()
        print(f"✓ Backward pass successful")
        print(f"  Gradients computed: Yes")
        print(f"  Weights updated: Yes")
    else:
        print(f"⚠ Warning: No gradients computed (model might be frozen)")
except Exception as e:
    print(f"✗ Backward pass failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Metrics computation
print("\nTest 9: Testing metrics computation...")
try:
    # Generate dummy predictions (realistic values)
    np.random.seed(42)
    n_samples = 20
    
    y_true_class = np.random.randint(0, 2, n_samples)
    y_pred_prob = np.random.rand(n_samples)
    y_true_score = np.random.randn(n_samples) * 15 + 100
    y_pred_score = y_true_score + np.random.randn(n_samples) * 10
    
    metrics = compute_all_metrics(
        y_true_class, y_pred_prob,
        y_true_score, y_pred_score,
        config
    )
    
    print(f"✓ Metrics computation successful")
    print(f"  Classification metrics:")
    print(f"    Accuracy: {metrics['accuracy']:.3f}")
    print(f"    AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"  Regression metrics:")
    print(f"    MAE: {metrics['mae']:.3f}")
    print(f"    Pearson r: {metrics['pearson_r']:.3f}")
except Exception as e:
    print(f"✗ Metrics computation failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Directory structure
print("\nTest 10: Checking directory structure...")
try:
    required_dirs = [
        'data/raw', 'data/processed', 'data/synthetic', 'data/clinical',
        'outputs/checkpoints', 'outputs/logs', 'outputs/predictions', 'outputs/figures'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    if missing_dirs:
        print(f"✓ Created missing directories:")
        for dir_path in missing_dirs:
            print(f"    - {dir_path}")
    else:
        print(f"✓ All directories present")
except Exception as e:
    print(f"✗ Directory check failed: {str(e)}")

# Final summary
print("\n" + "="*70)
print("TEST SUMMARY".center(70))
print("="*70)
print("\n✓ ALL TESTS PASSED!")
print("\nYour project is ready to use with synthetic data!")
print("\nNext steps:")
print("  1. Run mini training test: python mini_train_test.py")
print("  2. Run full training: python train.py")
print("  3. Evaluate model: python test.py")
print("\n" + "="*70 + "\n")

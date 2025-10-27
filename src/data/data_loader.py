"""
Data loader for multimodal neurodevelopmental prediction
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging


class NeurodevelopmentDataset(Dataset):
    """
    Dataset for multimodal neurodevelopmental outcome prediction
    
    Loads and provides access to:
    - Functional connectivity matrices (223x223)
    - Structural connectivity matrices (90x90)
    - DWMA features (11 features)
    - Clinical features (72 features)
    - Outcome labels (cognitive, language, motor)
    """
    
    def __init__(
        self,
        data_dict: Dict,
        config: Dict,
        transform=None,
        augmentation=None
    ):
        """
        Args:
            data_dict: Dictionary containing all data arrays
            config: Configuration dictionary
            transform: Optional transform to apply
            augmentation: Optional augmentation to apply
        """
        self.config = config
        self.transform = transform
        self.augmentation = augmentation
        
        # Store data
        self.functional_connectivity = data_dict['functional_connectivity']
        self.structural_connectivity = data_dict['structural_connectivity']
        self.dwma_features = data_dict['dwma_features']
        self.clinical_features = data_dict['clinical_features']
        
        # Store labels
        self.outcomes = config['model']['outcomes']
        self.labels = {}
        for outcome in self.outcomes:
            self.labels[f'{outcome}_class'] = data_dict[f'{outcome}_class_label']
            self.labels[f'{outcome}_score'] = data_dict[f'{outcome}_score_label']
        
        self.n_samples = len(self.functional_connectivity)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get data
        func_conn = self.functional_connectivity[idx]
        struct_conn = self.structural_connectivity[idx]
        dwma = self.dwma_features[idx]
        clinical = self.clinical_features[idx]
        
        # Apply augmentation if in training mode
        if self.augmentation is not None:
            func_conn, struct_conn, dwma, clinical = self.augmentation(
                func_conn, struct_conn, dwma, clinical
            )
        
        # Convert to tensors and add channel dimension for connectivity matrices
        func_conn = torch.from_numpy(func_conn).float().unsqueeze(0)  # [1, 223, 223]
        struct_conn = torch.from_numpy(struct_conn).float().unsqueeze(0)  # [1, 90, 90]
        dwma = torch.from_numpy(dwma).float()  # [11]
        clinical = torch.from_numpy(clinical).float()  # [72]
        
        # Get labels
        sample = {
            'functional_connectivity': func_conn,
            'structural_connectivity': struct_conn,
            'dwma_features': dwma,
            'clinical_features': clinical,
        }
        
        # Add labels for each outcome
        for outcome in self.outcomes:
            sample[f'{outcome}_class_label'] = torch.tensor(
                self.labels[f'{outcome}_class'][idx], dtype=torch.float32
            )
            sample[f'{outcome}_score_label'] = torch.tensor(
                self.labels[f'{outcome}_score'][idx], dtype=torch.float32
            )
        
        return sample


def generate_synthetic_data(config: Dict, logger: Optional[logging.Logger] = None) -> Dict:
    """
    Generate synthetic multimodal data for testing
    
    Args:
        config: Configuration dictionary
        logger: Optional logger
        
    Returns:
        Dictionary containing synthetic data arrays
    """
    if logger:
        logger.info("Generating synthetic data...")
    
    np.random.seed(config['data']['random_seed'])
    
    n_subjects = config['data']['n_synthetic_subjects']
    outcomes = config['model']['outcomes']
    
    # Generate connectivity matrices
    func_conn_size = tuple(config['model']['functional_connectivity_input_size'])
    struct_conn_size = tuple(config['model']['structural_connectivity_input_size'])
    dwma_size = config['model']['dwma_input_size']
    clinical_size = config['model']['clinical_input_size']
    
    # Functional connectivity (223x223)
    functional_connectivity = np.random.randn(n_subjects, func_conn_size[0], func_conn_size[1]).astype(np.float32)
    # Make symmetric
    for i in range(n_subjects):
        functional_connectivity[i] = (functional_connectivity[i] + functional_connectivity[i].T) / 2
    # Normalize
    functional_connectivity = (functional_connectivity - functional_connectivity.mean()) / functional_connectivity.std()
    
    # Structural connectivity (90x90)
    structural_connectivity = np.random.randn(n_subjects, struct_conn_size[0], struct_conn_size[1]).astype(np.float32)
    # Make symmetric and positive
    for i in range(n_subjects):
        structural_connectivity[i] = (structural_connectivity[i] + structural_connectivity[i].T) / 2
        structural_connectivity[i] = np.abs(structural_connectivity[i])
    # Normalize
    structural_connectivity = (structural_connectivity - structural_connectivity.mean()) / structural_connectivity.std()
    
    # DWMA features (11 features)
    dwma_features = np.random.randn(n_subjects, dwma_size).astype(np.float32)
    dwma_features = (dwma_features - dwma_features.mean(axis=0)) / dwma_features.std(axis=0)
    
    # Clinical features (72 features)
    clinical_features = np.random.randn(n_subjects, clinical_size).astype(np.float32)
    clinical_features = (clinical_features - clinical_features.mean(axis=0)) / clinical_features.std(axis=0)
    
    # Generate outcome labels
    data = {
        'functional_connectivity': functional_connectivity,
        'structural_connectivity': structural_connectivity,
        'dwma_features': dwma_features,
        'clinical_features': clinical_features,
    }
    
    # Generate correlated outcome scores
    # Create a base "developmental" score influenced by all modalities
    base_score = (
        np.mean(functional_connectivity.reshape(n_subjects, -1), axis=1) * 0.3 +
        np.mean(structural_connectivity.reshape(n_subjects, -1), axis=1) * 0.3 +
        np.mean(dwma_features, axis=1) * 0.2 +
        np.mean(clinical_features, axis=1) * 0.2
    )
    
    for outcome in outcomes:
        # Generate Bayley-III scores (mean=100, std=15)
        threshold = config['model']['task_types'][outcome]['classification_threshold']
        
        # Add outcome-specific variation
        outcome_variation = np.random.randn(n_subjects) * 0.5
        scores = 100 + (base_score + outcome_variation) * 15
        
        # Clip to realistic range [55, 145]
        scores = np.clip(scores, 55, 145)
        
        # Binary classification labels (high-risk if score < threshold)
        class_labels = (scores < threshold).astype(np.float32)
        
        data[f'{outcome}_score_label'] = scores.astype(np.float32)
        data[f'{outcome}_class_label'] = class_labels
        
        if logger:
            logger.info(f"  {outcome}: {(class_labels == 1).sum()} high-risk, "
                       f"{(class_labels == 0).sum()} low-risk")
    
    if logger:
        logger.info(f"Generated {n_subjects} synthetic subjects")
    
    return data


def split_data(data: Dict, config: Dict, logger: Optional[logging.Logger] = None) -> Tuple[Dict, Dict, Dict]:
    """
    Split data into train, validation, and test sets
    
    Args:
        data: Dictionary containing all data arrays
        config: Configuration dictionary
        logger: Optional logger
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n_subjects = len(data['functional_connectivity'])
    
    # Get split ratios
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    
    # Calculate split indices
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    n_test = n_subjects - n_train - n_val
    
    # Shuffle indices
    np.random.seed(config['data']['random_seed'])
    indices = np.random.permutation(n_subjects)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Split data
    def split_dict(data_dict, idx):
        return {k: v[idx] if isinstance(v, np.ndarray) else v 
                for k, v in data_dict.items()}
    
    train_data = split_dict(data, train_idx)
    val_data = split_dict(data, val_idx)
    test_data = split_dict(data, test_idx)
    
    if logger:
        logger.info(f"Data split: train={n_train}, val={n_val}, test={n_test}")
    
    return train_data, val_data, test_data


def create_dataloaders(config: Dict, logger: Optional[logging.Logger] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        config: Configuration dictionary
        logger: Optional logger
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .augmentation import DataAugmentation
    
    # Generate or load data
    if config['data']['mode'] == 'synthetic':
        if logger:
            logger.info("Using synthetic data")
        data = generate_synthetic_data(config, logger)
    else:
        # Load real data (implement based on actual data format)
        raise NotImplementedError("Real data loading not yet implemented")
    
    # Split data
    train_data, val_data, test_data = split_data(data, config, logger)
    
    # Create augmentation for training
    augmentation = DataAugmentation(config) if config['augmentation']['enabled'] else None
    
    # Create datasets
    train_dataset = NeurodevelopmentDataset(train_data, config, augmentation=augmentation)
    val_dataset = NeurodevelopmentDataset(val_data, config, augmentation=None)
    test_dataset = NeurodevelopmentDataset(test_data, config, augmentation=None)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True if config['training']['device'] != 'cpu' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True if config['training']['device'] != 'cpu' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True if config['training']['device'] != 'cpu' else False
    )
    
    return train_loader, val_loader, test_loader

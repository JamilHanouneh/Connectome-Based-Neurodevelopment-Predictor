"""
MRI preprocessing utilities
"""

import numpy as np
from typing import Tuple, Optional
import logging


def normalize_connectivity_matrix(matrix: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize connectivity matrix
    
    Args:
        matrix: Input connectivity matrix
        method: Normalization method ('zscore', 'minmax', 'log')
        
    Returns:
        Normalized matrix
    """
    if method == 'zscore':
        mean = np.mean(matrix)
        std = np.std(matrix)
        return (matrix - mean) / (std + 1e-8)
    
    elif method == 'minmax':
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        return (matrix - min_val) / (max_val - min_val + 1e-8)
    
    elif method == 'log':
        return np.log(matrix + 1.0)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def threshold_connectivity(matrix: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Threshold connectivity matrix
    
    Args:
        matrix: Input connectivity matrix
        threshold: Threshold value
        
    Returns:
        Thresholded matrix
    """
    matrix_copy = matrix.copy()
    matrix_copy[np.abs(matrix_copy) < threshold] = 0
    return matrix_copy


def fisher_z_transform(correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Apply Fisher Z-transformation to correlation matrix
    
    Args:
        correlation_matrix: Correlation matrix with values in [-1, 1]
        
    Returns:
        Fisher Z-transformed matrix
    """
    # Clip to avoid numerical issues
    clipped = np.clip(correlation_matrix, -0.999, 0.999)
    return np.arctanh(clipped)


def symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Make matrix symmetric
    
    Args:
        matrix: Input matrix
        
    Returns:
        Symmetric matrix
    """
    return (matrix + matrix.T) / 2


def handle_missing_values(data: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Handle missing values in data
    
    Args:
        data: Input data with potential NaN values
        method: Method to handle missing values
        
    Returns:
        Data with missing values handled
    """
    if method == 'mean':
        mask = np.isnan(data)
        data[mask] = np.nanmean(data)
    
    elif method == 'median':
        mask = np.isnan(data)
        data[mask] = np.nanmedian(data)
    
    elif method == 'zero':
        data = np.nan_to_num(data, nan=0.0)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return data


def standardize_features(features: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Standardize features to zero mean and unit variance
    
    Args:
        features: Input features [n_samples, n_features]
        
    Returns:
        Tuple of (standardized_features, statistics_dict)
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8
    
    standardized = (features - mean) / std
    
    stats = {
        'mean': mean,
        'std': std
    }
    
    return standardized, stats


def apply_standardization(features: np.ndarray, stats: dict) -> np.ndarray:
    """
    Apply pre-computed standardization to features
    
    Args:
        features: Input features
        stats: Statistics dictionary with 'mean' and 'std'
        
    Returns:
        Standardized features
    """
    return (features - stats['mean']) / stats['std']

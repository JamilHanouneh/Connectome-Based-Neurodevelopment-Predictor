"""
Data augmentation for multimodal brain data
"""

import numpy as np
from typing import Tuple


class DataAugmentation:
    """
    Data augmentation for connectivity matrices and clinical features
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.aug_config = config['augmentation']
        
    def __call__(
        self,
        func_conn: np.ndarray,
        struct_conn: np.ndarray,
        dwma: np.ndarray,
        clinical: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply augmentation to all modalities
        
        Args:
            func_conn: Functional connectivity matrix
            struct_conn: Structural connectivity matrix
            dwma: DWMA features
            clinical: Clinical features
            
        Returns:
            Tuple of augmented data
        """
        if not self.aug_config['enabled']:
            return func_conn, struct_conn, dwma, clinical
        
        # Augment connectivity matrices
        if self.aug_config['connectivity_augmentation']['gaussian_noise']['enabled']:
            func_conn = self._add_gaussian_noise(
                func_conn,
                std=self.aug_config['connectivity_augmentation']['gaussian_noise']['std']
            )
            struct_conn = self._add_gaussian_noise(
                struct_conn,
                std=self.aug_config['connectivity_augmentation']['gaussian_noise']['std']
            )
        
        if self.aug_config['connectivity_augmentation']['dropout_connections']['enabled']:
            func_conn = self._dropout_connections(
                func_conn,
                rate=self.aug_config['connectivity_augmentation']['dropout_connections']['dropout_rate']
            )
            struct_conn = self._dropout_connections(
                struct_conn,
                rate=self.aug_config['connectivity_augmentation']['dropout_connections']['dropout_rate']
            )
        
        # Augment clinical features
        if self.aug_config['clinical_augmentation']['gaussian_noise']['enabled']:
            dwma = self._add_gaussian_noise(
                dwma,
                std=self.aug_config['clinical_augmentation']['gaussian_noise']['std']
            )
            clinical = self._add_gaussian_noise(
                clinical,
                std=self.aug_config['clinical_augmentation']['gaussian_noise']['std']
            )
        
        if self.aug_config['clinical_augmentation']['feature_dropout']['enabled']:
            dwma = self._feature_dropout(
                dwma,
                rate=self.aug_config['clinical_augmentation']['feature_dropout']['dropout_rate']
            )
            clinical = self._feature_dropout(
                clinical,
                rate=self.aug_config['clinical_augmentation']['feature_dropout']['dropout_rate']
            )
        
        return func_conn, struct_conn, dwma, clinical
    
    @staticmethod
    def _add_gaussian_noise(data: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to data"""
        noise = np.random.randn(*data.shape) * std
        return data + noise
    
    @staticmethod
    def _dropout_connections(matrix: np.ndarray, rate: float) -> np.ndarray:
        """Randomly drop connections in connectivity matrix"""
        mask = np.random.rand(*matrix.shape) > rate
        augmented = matrix * mask
        # Keep matrix symmetric
        augmented = (augmented + augmented.T) / 2
        return augmented
    
    @staticmethod
    def _feature_dropout(features: np.ndarray, rate: float) -> np.ndarray:
        """Randomly drop features"""
        mask = np.random.rand(*features.shape) > rate
        return features * mask

"""
Neural network models
"""

from .multimodal_network import MultimodalNetwork
from .feature_extractor import VGG19FeatureExtractor

__all__ = [
    'MultimodalNetwork',
    'VGG19FeatureExtractor'
]

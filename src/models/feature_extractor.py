"""
Feature extractor based on VGG-19
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGG19FeatureExtractor(nn.Module):
    """
    VGG-19 based feature extractor for connectivity matrices
    
    Uses pre-trained VGG-19 on ImageNet and adapts it for 
    grayscale connectivity matrices
    """
    
    def __init__(self, pretrained: bool = True, freeze_layers: bool = False):
        """
        Args:
            pretrained: Whether to use pre-trained ImageNet weights
            freeze_layers: Whether to freeze convolutional layers
        """
        super(VGG19FeatureExtractor, self).__init__()
        
        # Load pre-trained VGG-19
        vgg19 = models.vgg19(pretrained=pretrained)
        
        # Modify first layer to accept single-channel input
        # Original: Conv2d(3, 64, kernel_size=3, padding=1)
        original_first_conv = vgg19.features[0]
        
        self.first_conv = nn.Conv2d(
            1, 64,  # Change from 3 channels to 1
            kernel_size=3,
            padding=1
        )
        
        # Initialize new conv layer
        if pretrained:
            # Average the RGB weights to initialize single-channel weights
            with torch.no_grad():
                self.first_conv.weight[:, 0, :, :] = original_first_conv.weight.mean(dim=1)
                self.first_conv.bias = original_first_conv.bias
        
        # Use remaining VGG-19 convolutional layers
        self.features = nn.Sequential(
            self.first_conv,
            *list(vgg19.features.children())[1:]  # Skip original first conv
        )
        
        # Freeze convolutional layers if specified
        if freeze_layers:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final feature dimension after GAP
        self.feature_dim = 512
        
        # Additional fully connected layer for dimensionality reduction
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input connectivity matrix [batch, 1, H, W]
            
        Returns:
            Feature vector [batch, 512]
        """
        # Extract convolutional features
        features = self.features(x)  # [batch, 512, H', W']
        
        # Global average pooling
        pooled = self.global_avg_pool(features)  # [batch, 512, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [batch, 512]
        
        # Final FC layer
        output = self.fc(pooled)  # [batch, 512]
        
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate feature maps (before global pooling)
        Useful for Grad-CAM visualization
        
        Returns:
            Feature maps [batch, 512, H', W']
        """
        return self.features(x)

"""
Multimodal neural network for neurodevelopmental outcome prediction
"""

import torch
import torch.nn as nn
from typing import Dict, List
from .feature_extractor import VGG19FeatureExtractor


class MultimodalNetwork(nn.Module):
    """
    4-channel multimodal neural network combining:
    1. Functional connectivity (VGG-19 features)
    2. Structural connectivity (VGG-19 features)
    3. DWMA features (fully connected)
    4. Clinical features (fully connected)
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super(MultimodalNetwork, self).__init__()
        
        self.config = config
        self.model_config = config['model']
        self.outcomes = self.model_config['outcomes']
        
        # Channel 1: Functional connectivity feature extractor
        self.func_conn_extractor = VGG19FeatureExtractor(
            pretrained=self.model_config['pretrained_vgg19'],
            freeze_layers=self.model_config['freeze_vgg19_layers']
        )
        
        # Channel 2: Structural connectivity feature extractor
        self.struct_conn_extractor = VGG19FeatureExtractor(
            pretrained=self.model_config['pretrained_vgg19'],
            freeze_layers=self.model_config['freeze_vgg19_layers']
        )
        
        # Channel 3: DWMA feature processor
        dwma_size = self.model_config['dwma_input_size']
        self.dwma_fc = nn.Sequential(
            nn.Linear(dwma_size, 128),
            nn.ReLU(),
            nn.Dropout(self.model_config['dropout_rate']),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Channel 4: Clinical feature processor
        clinical_size = self.model_config['clinical_input_size']
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_size, 256),
            nn.ReLU(),
            nn.Dropout(self.model_config['dropout_rate']),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Fusion layer dimensions
        # VGG-19 output: 512 features from each connectivity channel
        # DWMA: 128, Clinical: 256
        fusion_input_size = 512 + 512 + 128 + 256  # = 1408
        
        # Fusion network
        fc_dims = self.model_config['fc_dims']
        dropout_rate = self.model_config['dropout_rate']
        
        fusion_layers = []
        in_features = fusion_input_size
        
        for out_features in fc_dims:
            fusion_layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = out_features
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # Output heads for each outcome (classification + regression)
        self.output_heads = nn.ModuleDict()
        
        for outcome in self.outcomes:
            # Classification head (binary)
            self.output_heads[f'{outcome}_class'] = nn.Linear(fc_dims[-1], 1)
            
            # Regression head (Bayley-III score)
            self.output_heads[f'{outcome}_reg'] = nn.Linear(fc_dims[-1], 1)
    
    def forward(
        self,
        func_conn: torch.Tensor,
        struct_conn: torch.Tensor,
        dwma: torch.Tensor,
        clinical: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            func_conn: Functional connectivity [batch, 1, 223, 223]
            struct_conn: Structural connectivity [batch, 1, 90, 90]
            dwma: DWMA features [batch, 11]
            clinical: Clinical features [batch, 72]
            
        Returns:
            Dictionary of outputs for each outcome
        """
        # Extract features from connectivity matrices
        func_features = self.func_conn_extractor(func_conn)  # [batch, 512]
        struct_features = self.struct_conn_extractor(struct_conn)  # [batch, 512]
        
        # Process DWMA and clinical features
        dwma_features = self.dwma_fc(dwma)  # [batch, 128]
        clinical_features = self.clinical_fc(clinical)  # [batch, 256]
        
        # Concatenate all features
        fused_features = torch.cat([
            func_features,
            struct_features,
            dwma_features,
            clinical_features
        ], dim=1)  # [batch, 1408]
        
        # Pass through fusion network
        shared_representation = self.fusion_network(fused_features)  # [batch, fc_dims[-1]]
        
        # Generate outputs for each outcome
        outputs = {}
        
        for outcome in self.outcomes:
            # Classification output (logits)
            outputs[f'{outcome}_class'] = self.output_heads[f'{outcome}_class'](
                shared_representation
            ).squeeze(-1)
            
            # Regression output (predicted score)
            outputs[f'{outcome}_reg'] = self.output_heads[f'{outcome}_reg'](
                shared_representation
            ).squeeze(-1)
        
        return outputs
    
    def get_feature_representation(
        self,
        func_conn: torch.Tensor,
        struct_conn: torch.Tensor,
        dwma: torch.Tensor,
        clinical: torch.Tensor
    ) -> torch.Tensor:
        """
        Get fused feature representation before output heads
        Useful for visualization and interpretability
        
        Returns:
            Fused feature representation [batch, fc_dims[-1]]
        """
        func_features = self.func_conn_extractor(func_conn)
        struct_features = self.struct_conn_extractor(struct_conn)
        dwma_features = self.dwma_fc(dwma)
        clinical_features = self.clinical_fc(clinical)
        
        fused_features = torch.cat([
            func_features,
            struct_features,
            dwma_features,
            clinical_features
        ], dim=1)
        
        shared_representation = self.fusion_network(fused_features)
        
        return shared_representation

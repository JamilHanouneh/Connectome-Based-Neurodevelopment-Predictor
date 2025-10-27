"""
Loss functions for multimodal multitask learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning
    Handles both classification and regression for multiple outcomes
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super(MultiTaskLoss, self).__init__()
        
        self.config = config
        self.outcomes = config['model']['outcomes']
        self.loss_weights = config['training']['loss_weights']
        
        # Classification loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Regression loss
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Compute combined loss
        
        Args:
            predictions: Dictionary of model outputs
            labels: Dictionary of ground truth labels
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        total_loss = 0.0
        loss_dict = {}
        
        for outcome in self.outcomes:
            # Classification loss
            class_pred = predictions[f'{outcome}_class']
            class_label = labels[f'{outcome}_class']
            
            class_loss = self.bce_loss(class_pred, class_label)
            loss_dict[f'{outcome}_class_loss'] = class_loss.item()
            
            # Regression loss
            reg_pred = predictions[f'{outcome}_reg']
            reg_label = labels[f'{outcome}_score']
            
            reg_loss = self.mse_loss(reg_pred, reg_label)
            loss_dict[f'{outcome}_reg_loss'] = reg_loss.item()
            
            # Combined loss for this outcome
            outcome_loss = (
                self.loss_weights['classification'] * class_loss +
                self.loss_weights['regression'] * reg_loss
            )
            
            loss_dict[f'{outcome}_total_loss'] = outcome_loss.item()
            total_loss += outcome_loss
        
        # Average over outcomes
        total_loss = total_loss / len(self.outcomes)
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

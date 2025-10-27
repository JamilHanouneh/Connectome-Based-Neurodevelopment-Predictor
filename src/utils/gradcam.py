"""
Grad-CAM visualization for model interpretability
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


class GradCAM:
    """
    Grad-CAM for visualizing model attention on connectivity matrices
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Neural network model
            target_layer: Target layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activation maps"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, outcome='cognitive', task='class'):
        """
        Generate class activation map
        
        Args:
            input_tensor: Input tensor
            outcome: Which outcome to visualize
            task: 'class' or 'reg'
            
        Returns:
            CAM heatmap
        """
        # Forward pass
        self.model.eval()
        output = self.model(*input_tensor)
        
        # Get target output
        target_output = output[f'{outcome}_{task}']
        
        # Backward pass
        self.model.zero_grad()
        target_output.backward(torch.ones_like(target_output))
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # Compute weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Compute CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam


def generate_gradcam_visualization(
    model: torch.nn.Module,
    subject_data: Dict,
    config: Dict,
    device: torch.device,
    output_dir: Path
):
    """
    Generate Grad-CAM visualizations for a subject
    
    Args:
        model: Trained model
        subject_data: Subject's input data
        config: Configuration dictionary
        device: Compute device
        output_dir: Directory to save visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Move data to device
    func_conn = subject_data['functional_connectivity'].to(device)
    struct_conn = subject_data['structural_connectivity'].to(device)
    dwma = subject_data['dwma_features'].to(device)
    clinical = subject_data['clinical_features'].to(device)
    
    input_tensor = (func_conn, struct_conn, dwma, clinical)
    
    outcomes = config['model']['outcomes']
    
    # Generate Grad-CAM for functional connectivity
    try:
        target_layer = model.func_conn_extractor.features[-1]
        gradcam = GradCAM(model, target_layer)
        
        for outcome in outcomes:
            cam = gradcam.generate_cam(input_tensor, outcome=outcome, task='class')
            
            # Resize CAM to input size
            cam_resized = F.interpolate(
                torch.from_numpy(cam).unsqueeze(0).unsqueeze(0),
                size=func_conn.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original connectivity
            im1 = axes[0].imshow(func_conn[0, 0].cpu().numpy(), cmap='RdBu_r')
            axes[0].set_title('Functional Connectivity')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0])
            
            # Grad-CAM overlay
            axes[1].imshow(func_conn[0, 0].cpu().numpy(), cmap='gray', alpha=0.5)
            im2 = axes[1].imshow(cam_resized, cmap='jet', alpha=0.5)
            axes[1].set_title(f'Grad-CAM: {outcome.capitalize()}')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(output_dir / f'gradcam_func_{outcome}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    except Exception as e:
        print(f"Warning: Grad-CAM generation failed: {str(e)}")

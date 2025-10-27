"""
Training loop and utilities
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm

from ..models.loss import MultiTaskLoss
from .callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    """
    Trainer class for multimodal neural network
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        logger: logging.Logger
    ):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Compute device
            logger: Logger instance
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        self.training_config = config['training']
        self.outcomes = config['model']['outcomes']
        
        # Loss function
        self.criterion = MultiTaskLoss(config)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            mode=config['training']['early_stopping']['mode']
        ) if config['training']['early_stopping']['enabled'] else None
        
        self.model_checkpoint = ModelCheckpoint(
            checkpoint_dir=config['output']['checkpoint_dir'],
            monitor=config['output']['monitor_metric'],
            mode='max',
            save_best_only=config['output']['save_best_only']
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
        for outcome in self.outcomes:
            self.history[f'{outcome}_train_class_loss'] = []
            self.history[f'{outcome}_train_reg_loss'] = []
            self.history[f'{outcome}_val_class_loss'] = []
            self.history[f'{outcome}_val_reg_loss'] = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_checkpoint_path = None
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_name = self.training_config['optimizer'].lower()
        lr = self.training_config['learning_rate']
        weight_decay = self.training_config['weight_decay']
        
        if optimizer_name == 'adam':
            params = self.training_config.get('optimizer_params', {})
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **params
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.training_config['scheduler']
        params = self.training_config.get('scheduler_params', {})
        
        if scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get('step_size', 30),
                gamma=params.get('gamma', 0.1)
            )
        elif scheduler_name == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=params.get('gamma', 0.95)
            )
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.get('T_max', 50)
            )
        elif scheduler_name == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=params.get('mode', 'min'),
                factor=params.get('factor', 0.5),
                patience=params.get('patience', 10),
                min_lr=params.get('min_lr', 1e-6)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
        }
        for outcome in self.outcomes:
            epoch_losses[f'{outcome}_class_loss'] = 0.0
            epoch_losses[f'{outcome}_reg_loss'] = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            func_conn = batch['functional_connectivity'].to(self.device)
            struct_conn = batch['structural_connectivity'].to(self.device)
            dwma = batch['dwma_features'].to(self.device)
            clinical = batch['clinical_features'].to(self.device)
            
            labels = {}
            for outcome in self.outcomes:
                labels[f'{outcome}_class'] = batch[f'{outcome}_class_label'].to(self.device)
                labels[f'{outcome}_score'] = batch[f'{outcome}_score_label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(func_conn, struct_conn, dwma, clinical)
            
            # Compute loss
            loss, loss_dict = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.training_config['gradient_clipping']['enabled']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['gradient_clipping']['max_norm']
                )
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key] += value
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average losses
        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
        }
        for outcome in self.outcomes:
            epoch_losses[f'{outcome}_class_loss'] = 0.0
            epoch_losses[f'{outcome}_reg_loss'] = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]  ")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            func_conn = batch['functional_connectivity'].to(self.device)
            struct_conn = batch['structural_connectivity'].to(self.device)
            dwma = batch['dwma_features'].to(self.device)
            clinical = batch['clinical_features'].to(self.device)
            
            labels = {}
            for outcome in self.outcomes:
                labels[f'{outcome}_class'] = batch[f'{outcome}_class_label'].to(self.device)
                labels[f'{outcome}_score'] = batch[f'{outcome}_score_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(func_conn, struct_conn, dwma, clinical)
            
            # Compute loss
            loss, loss_dict = self.criterion(outputs, labels)
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key] += value
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average losses
        n_batches = len(self.val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def train(self, start_epoch: int = 0):
        """Main training loop"""
        num_epochs = self.training_config['num_epochs']
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            self.logger.info(f"LR: {current_lr:.6f}")
            self.logger.info(f"Train Loss: {train_losses['total_loss']:.4f}")
            self.logger.info(f"Val Loss: {val_losses['total_loss']:.4f}")
            
            for outcome in self.outcomes:
                self.logger.info(f"  {outcome} - Train Class: {train_losses[f'{outcome}_class_loss']:.4f}, "
                               f"Reg: {train_losses[f'{outcome}_reg_loss']:.4f}")
                self.logger.info(f"  {outcome} - Val Class: {val_losses[f'{outcome}_class_loss']:.4f}, "
                               f"Reg: {val_losses[f'{outcome}_reg_loss']:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['val_loss'].append(val_losses['total_loss'])
            
            # Save checkpoint
            checkpoint_path = self.model_checkpoint.step(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                val_loss=val_losses['total_loss']
            )
            
            if checkpoint_path:
                self.best_checkpoint_path = checkpoint_path
                self.best_val_loss = val_losses['total_loss']
                self.best_epoch = epoch + 1
                self.logger.info(f"✓ Saved best model: {checkpoint_path}")
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping.step(val_losses['total_loss']):
                    self.logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        self.logger.info("\n" + "="*70)
        self.logger.info("Training completed")
        self.logger.info(f"Best epoch: {self.best_epoch}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best model: {self.best_checkpoint_path}")
        
        # Save training curves
        self._save_training_curves()
    
    def _save_training_curves(self):
        """Save training curves"""
        import matplotlib.pyplot as plt
        
        figure_dir = Path(self.config['output']['figure_dir'])
        figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figure_dir / 'loss_curves.png', dpi=300)
        plt.close()
        
        self.logger.info(f"Saved training curves to {figure_dir}/loss_curves.png")
    
    def save_checkpoint(self, epoch: int, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': self.history['train_loss'][-1],
            'val_loss': self.history['val_loss'][-1],
            'config': self.config,
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        self.logger.info(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch + 1

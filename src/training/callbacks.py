"""
Training callbacks
"""

import torch
from pathlib import Path
from typing import Optional


class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def step(self, current_value: float) -> bool:
        """
        Check if training should stop
        
        Args:
            current_value: Current metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        # Check if improved
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        
        return False


class ModelCheckpoint:
    """
    Model checkpoint callback to save best models
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best model
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        
        self.best_value = None
    
    def step(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        val_loss: float
    ) -> Optional[str]:
        """
        Check if model should be saved
        
        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        current_value = val_loss
        
        # Check if this is the best model
        is_best = False
        if self.best_value is None:
            is_best = True
            self.best_value = current_value
        else:
            if self.mode == 'min':
                is_best = current_value < self.best_value
            else:
                is_best = current_value > self.best_value
            
            if is_best:
                self.best_value = current_value
        
        # Save checkpoint
        if is_best or not self.save_best_only:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
            }
            
            if is_best:
                filepath = self.checkpoint_dir / 'best_model.pth'
            else:
                filepath = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            
            torch.save(checkpoint, filepath)
            
            return str(filepath)
        
        return None

"""Training loop and model trainer."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm


def compute_positions(
    predictions: torch.Tensor,
    use_tanh: bool = True
) -> torch.Tensor:
    """
    Convert model predictions to trading positions.
    
    Args:
        predictions: Model output (logits or values)
        use_tanh: If True, apply w = tanh(p/2); else use predictions directly
    
    Returns:
        Trading positions
    """
    if use_tanh:
        return torch.tanh(predictions / 2.0)
    else:
        return predictions


def compute_roi(
    positions: np.ndarray,
    log_returns: np.ndarray
) -> float:
    """
    Compute Return on Investment.
    
    Args:
        positions: Trading positions
        log_returns: Log returns
    
    Returns:
        Compound ROI
    """
    pnl = positions * log_returns
    roi = np.sum(pnl)
    return roi


def compute_sharpe_ratio(
    positions: np.ndarray,
    log_returns: np.ndarray
) -> float:
    """
    Compute Sharpe ratio.
    
    Args:
        positions: Trading positions
        log_returns: Log returns
    
    Returns:
        Sharpe ratio
    """
    pnl = positions * log_returns
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)
    
    if std_pnl > 0:
        sharpe = mean_pnl / std_pnl
    else:
        sharpe = 0.0
    
    return sharpe


def compute_max_drawdown(
    positions: np.ndarray,
    log_returns: np.ndarray
) -> float:
    """
    Compute Maximum Drawdown.
    
    Args:
        positions: Trading positions
        log_returns: Log returns
    
    Returns:
        Maximum drawdown (positive value)
    """
    pnl = positions * log_returns
    cumulative = np.cumsum(pnl)
    
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = np.max(drawdown)
    
    return max_dd


def apply_risk_budget_scaling(
    positions: np.ndarray,
    scale_factor: float,
    dead_zone: float = 0.01,
    max_leverage: float = 1.0
) -> np.ndarray:
    """
    Apply risk-budget scaling with dead-zone and leverage constraints.
    
    Args:
        positions: Raw positions
        scale_factor: Scaling factor from validation set
        dead_zone: Dead-zone threshold
        max_leverage: Maximum leverage
    
    Returns:
        Scaled positions
    """
    # Scale positions
    scaled = positions / (scale_factor + 1e-8)
    
    # Apply dead-zone
    scaled = np.where(np.abs(scaled) < dead_zone, 0.0, scaled)
    
    # Clip to leverage
    scaled = np.clip(scaled, -max_leverage, max_leverage)
    
    return scaled


class Trainer:
    """
    Model trainer for trading experiments.
    
    Custom - Context7 found no library equivalent (paper training protocol)
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        use_tanh: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device to use
            use_tanh: Whether to use tanh for position conversion
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.use_tanh = use_tanh
        
        self.model.to(device)
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        epoch_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            predictions = self.model(batch_x)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(predictions, batch_y, self.model)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([d[key] for d in epoch_losses])
        
        return avg_losses
    
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader
            return_predictions: If True, return predictions and targets
        
        Returns:
            Dictionary of metrics (and optionally predictions)
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        # Convert to positions
        if self.use_tanh:
            positions = np.tanh(predictions / 2.0)
        else:
            positions = predictions
        
        # Compute metrics
        roi = compute_roi(positions, targets)
        sharpe = compute_sharpe_ratio(positions, targets)
        mdd = compute_max_drawdown(positions, targets)
        
        metrics = {
            'roi': roi,
            'sharpe': sharpe,
            'mdd': mdd
        }
        
        if return_predictions:
            metrics['predictions'] = predictions
            metrics['targets'] = targets
            metrics['positions'] = positions
        
        return metrics
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        n_epochs: int = 80,
        early_stopping_start: int = 30,
        verbose: bool = True
    ) -> Tuple[Dict[str, float], float]:
        """
        Complete training loop with validation-based checkpointing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            early_stopping_start: Epoch to start validation-based selection
            verbose: Print progress
        
        Returns:
            Tuple of (best_val_metrics, best_scale_factor)
        """
        best_val_roi = -np.inf
        best_epoch = 0
        best_state = None
        best_scale_factor = 1.0
        
        for epoch in range(n_epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Evaluate on validation set after early_stopping_start
            if epoch >= early_stopping_start:
                val_metrics = self.evaluate(val_loader, return_predictions=True)
                val_roi = val_metrics['roi']
                
                # Compute scale factor
                positions = val_metrics['positions']
                scale_factor = np.mean(np.abs(positions))
                
                if val_roi > best_val_roi:
                    best_val_roi = val_roi
                    best_epoch = epoch
                    best_state = self.model.state_dict().copy()
                    best_scale_factor = scale_factor
                
                if verbose:
                    print(f"Epoch {epoch+1}/{n_epochs} - "
                          f"Train Loss: {train_losses['total']:.4f} - "
                          f"Val ROI: {val_roi:.4f} - "
                          f"Val Sharpe: {val_metrics['sharpe']:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{n_epochs} - "
                          f"Train Loss: {train_losses['total']:.4f}")
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"\nBest model from epoch {best_epoch+1}")
        
        # Final validation metrics
        val_metrics = self.evaluate(val_loader)
        
        return val_metrics, best_scale_factor

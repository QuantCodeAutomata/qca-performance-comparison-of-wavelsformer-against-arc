"""Base experiment class with common functionality."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
import os
import json

from ..models.wavelsformer import (
    WaveLSFormer, WaveletMLPModel, WaveletLSTMModel,
    ClassicWaveletTransformer, WaveLSFormerConcatFusion,
    WaveLSFormerLowOnly, WaveLSFormerHighOnly
)
from ..models.backbones import MLPBackbone, LSTMBackbone, TransformerBackbone
from ..losses.trading_losses import CompositeTradingLoss, MSELoss, MAELoss
from ..training.trainer import Trainer, apply_risk_budget_scaling, compute_roi, compute_sharpe_ratio, compute_max_drawdown
from ..utils.dataset import TimeSeriesDataset
from ..data.data_loader import create_windows


class BaseExperiment:
    """
    Base class for all experiments.
    
    Custom - Context7 found no library equivalent (paper experiment protocol)
    """
    
    def __init__(
        self,
        experiment_id: str,
        n_features: int,
        window_size: int = 96,
        batch_size: int = 256,
        learning_rate: float = 1e-5,
        n_epochs: int = 80,
        early_stopping_start: int = 30,
        device: str = 'cpu',
        results_dir: str = 'results'
    ):
        """
        Initialize base experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            n_features: Number of input features
            window_size: Input window size
            batch_size: Training batch size
            learning_rate: Learning rate
            n_epochs: Number of training epochs
            early_stopping_start: Epoch to start validation-based selection
            device: Device to use
            results_dir: Directory to save results
        """
        self.experiment_id = experiment_id
        self.n_features = n_features
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.early_stopping_start = early_stopping_start
        self.device = device
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
    
    def create_model(self, model_type: str, **kwargs) -> torch.nn.Module:
        """
        Create model based on type.
        
        Args:
            model_type: Model type identifier
            **kwargs: Additional model arguments
        
        Returns:
            Model instance
        """
        if model_type == 'wavelsformer':
            return WaveLSFormer(
                n_features=self.n_features,
                window_size=self.window_size,
                **kwargs
            )
        elif model_type == 'mlp':
            input_dim = self.n_features * self.window_size
            return MLPBackbone(input_dim=input_dim, **kwargs)
        elif model_type == 'wavelet_mlp':
            return WaveletMLPModel(
                n_features=self.n_features,
                window_size=self.window_size,
                **kwargs
            )
        elif model_type == 'lstm':
            return LSTMBackbone(input_dim=self.n_features, **kwargs)
        elif model_type == 'wavelet_lstm':
            return WaveletLSTMModel(
                n_features=self.n_features,
                window_size=self.window_size,
                **kwargs
            )
        elif model_type == 'transformer':
            return TransformerBackbone(input_dim=self.n_features, **kwargs)
        elif model_type == 'classic_wavelet_transformer':
            return ClassicWaveletTransformer(
                n_features=self.n_features,
                window_size=self.window_size,
                **kwargs
            )
        elif model_type == 'wavelsformer_concat':
            return WaveLSFormerConcatFusion(
                n_features=self.n_features,
                window_size=self.window_size,
                **kwargs
            )
        elif model_type == 'wavelsformer_low':
            return WaveLSFormerLowOnly(
                n_features=self.n_features,
                window_size=self.window_size,
                **kwargs
            )
        elif model_type == 'wavelsformer_high':
            return WaveLSFormerHighOnly(
                n_features=self.n_features,
                window_size=self.window_size,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_loss_fn(self, loss_type: str, **kwargs) -> torch.nn.Module:
        """
        Create loss function based on type.
        
        Args:
            loss_type: Loss type identifier
            **kwargs: Additional loss arguments
        
        Returns:
            Loss function instance
        """
        if loss_type == 'composite':
            return CompositeTradingLoss(**kwargs)
        elif loss_type == 'mse':
            return MSELoss()
        elif loss_type == 'mae':
            return MAELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def prepare_data(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        test_data: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders.
        
        Args:
            train_data: Training data (T, d) where T is time steps, d is features
            val_data: Validation data
            test_data: Test data
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create windows
        train_windows = create_windows(train_data[:-1], self.window_size)
        train_targets = train_data[self.window_size:, -1]  # Target is last feature (log return)
        
        val_windows = create_windows(val_data[:-1], self.window_size)
        val_targets = val_data[self.window_size:, -1]
        
        test_windows = create_windows(test_data[:-1], self.window_size)
        test_targets = test_data[self.window_size:, -1]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(train_windows, train_targets)
        val_dataset = TimeSeriesDataset(val_windows, val_targets)
        test_dataset = TimeSeriesDataset(test_windows, test_targets)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        use_tanh: bool = True,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Train and evaluate a model.
        
        Args:
            model: Model to train
            loss_fn: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            use_tanh: Whether to use tanh for position conversion
            verbose: Print progress
        
        Returns:
            Dictionary of test metrics
        """
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=self.device,
            use_tanh=use_tanh
        )
        
        # Train
        val_metrics, scale_factor = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=self.n_epochs,
            early_stopping_start=self.early_stopping_start,
            verbose=verbose
        )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(test_loader, return_predictions=True)
        
        # Apply risk-budget scaling
        raw_positions = test_metrics['positions']
        scaled_positions = apply_risk_budget_scaling(
            raw_positions,
            scale_factor,
            dead_zone=0.01,
            max_leverage=1.0
        )
        
        # Recompute metrics with scaled positions
        test_targets = test_metrics['targets']
        test_roi = compute_roi(scaled_positions, test_targets)
        test_sharpe = compute_sharpe_ratio(scaled_positions, test_targets)
        test_mdd = compute_max_drawdown(scaled_positions, test_targets)
        
        return {
            'roi': test_roi,
            'sharpe': test_sharpe,
            'mdd': test_mdd,
            'scale_factor': scale_factor
        }
    
    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def run(self, *args, **kwargs):
        """Run experiment. To be implemented by subclasses."""
        raise NotImplementedError

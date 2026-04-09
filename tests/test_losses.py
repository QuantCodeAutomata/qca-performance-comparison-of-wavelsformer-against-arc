"""Tests for loss functions."""

import torch
import numpy as np
import pytest

from src.losses.trading_losses import (
    SoftLabelLoss, SharpeRegularizer, ROIPenalty,
    MSELoss, MAELoss, CompositeTradingLoss
)


def test_soft_label_loss():
    """Test soft-label loss function."""
    loss_fn = SoftLabelLoss(temperature=45.0)
    
    predictions = torch.randn(10)
    log_returns = torch.randn(10) * 0.01
    
    loss = loss_fn(predictions, log_returns)
    
    # Check loss is scalar and positive
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_sharpe_regularizer():
    """Test Sharpe regularizer."""
    reg = SharpeRegularizer(alpha=1.0)
    
    # Test with profitable positions
    positions = torch.ones(100)
    log_returns = torch.randn(100) * 0.01 + 0.001  # Positive mean
    
    loss = reg(positions, log_returns)
    
    # Loss should be negative (we want to maximize Sharpe)
    assert loss.ndim == 0


def test_roi_penalty():
    """Test ROI penalty."""
    penalty = ROIPenalty(lambda_roi=0.5)
    
    positions = torch.randn(100)
    log_returns = torch.randn(100) * 0.01
    
    loss = penalty(positions, log_returns)
    
    # Check loss is scalar
    assert loss.ndim == 0


def test_mse_loss():
    """Test MSE loss."""
    loss_fn = MSELoss()
    
    predictions = torch.randn(10)
    targets = torch.randn(10)
    
    loss = loss_fn(predictions, targets)
    
    # Check loss is scalar and positive
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_mae_loss():
    """Test MAE loss."""
    loss_fn = MAELoss()
    
    predictions = torch.randn(10)
    targets = torch.randn(10)
    
    loss = loss_fn(predictions, targets)
    
    # Check loss is scalar and positive
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_composite_loss_soft_label():
    """Test composite loss with soft-label."""
    loss_fn = CompositeTradingLoss(
        use_soft_label=True,
        use_sharpe=True,
        use_penalty=True,
        use_wavelet=False
    )
    
    predictions = torch.randn(100)
    log_returns = torch.randn(100) * 0.01
    
    total_loss, loss_dict = loss_fn(predictions, log_returns)
    
    # Check loss components
    assert 'trade' in loss_dict
    assert 'sharpe' in loss_dict
    assert 'penalty' in loss_dict
    assert 'total' in loss_dict
    
    # Check total loss is scalar
    assert total_loss.ndim == 0


def test_composite_loss_mse():
    """Test composite loss with MSE."""
    loss_fn = CompositeTradingLoss(
        use_soft_label=False,
        use_sharpe=False,
        use_penalty=False,
        use_wavelet=False
    )
    
    predictions = torch.randn(100)
    log_returns = torch.randn(100) * 0.01
    
    total_loss, loss_dict = loss_fn(predictions, log_returns)
    
    # Check loss components
    assert 'trade' in loss_dict
    assert 'total' in loss_dict
    
    # Check total loss is scalar
    assert total_loss.ndim == 0


def test_loss_backward():
    """Test that loss can be backpropagated."""
    loss_fn = CompositeTradingLoss(
        use_soft_label=True,
        use_sharpe=True,
        use_penalty=True,
        use_wavelet=False
    )
    
    predictions = torch.randn(100, requires_grad=True)
    log_returns = torch.randn(100)
    
    total_loss, _ = loss_fn(predictions, log_returns)
    total_loss.backward()
    
    # Check gradients exist
    assert predictions.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

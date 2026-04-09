"""Tests for training module."""

import torch
import numpy as np
import pytest

from src.training.trainer import (
    compute_positions, compute_roi, compute_sharpe_ratio,
    compute_max_drawdown, apply_risk_budget_scaling
)


def test_compute_positions_tanh():
    """Test position computation with tanh."""
    predictions = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])
    positions = compute_positions(predictions, use_tanh=True)
    
    # Check all positions are in [-1, 1]
    assert torch.all(positions >= -1.0)
    assert torch.all(positions <= 1.0)


def test_compute_positions_direct():
    """Test position computation without tanh."""
    predictions = torch.tensor([0.5, -0.5, 0.0])
    positions = compute_positions(predictions, use_tanh=False)
    
    # Should be identical to predictions
    assert torch.allclose(positions, predictions)


def test_compute_roi():
    """Test ROI computation."""
    positions = np.array([1.0, 1.0, -1.0, -1.0])
    log_returns = np.array([0.01, 0.02, 0.01, -0.01])
    
    roi = compute_roi(positions, log_returns)
    
    # Expected: 1*0.01 + 1*0.02 + (-1)*0.01 + (-1)*(-0.01) = 0.03
    assert np.isclose(roi, 0.03)


def test_compute_sharpe_ratio():
    """Test Sharpe ratio computation."""
    positions = np.ones(100)
    log_returns = np.random.randn(100) * 0.01 + 0.001
    
    sharpe = compute_sharpe_ratio(positions, log_returns)
    
    # Sharpe should be a scalar
    assert isinstance(sharpe, float)


def test_compute_sharpe_ratio_zero_std():
    """Test Sharpe ratio with zero std."""
    positions = np.ones(10)
    log_returns = np.zeros(10)
    
    sharpe = compute_sharpe_ratio(positions, log_returns)
    
    # Should return 0 when std is 0
    assert sharpe == 0.0


def test_compute_max_drawdown():
    """Test maximum drawdown computation."""
    positions = np.ones(100)
    log_returns = np.random.randn(100) * 0.01
    
    mdd = compute_max_drawdown(positions, log_returns)
    
    # MDD should be non-negative
    assert mdd >= 0


def test_apply_risk_budget_scaling():
    """Test risk-budget scaling."""
    positions = np.array([0.5, -0.3, 0.1, -0.05, 0.8])
    scale_factor = 0.2
    
    scaled = apply_risk_budget_scaling(
        positions,
        scale_factor,
        dead_zone=0.1,
        max_leverage=1.0
    )
    
    # Check all positions are in [-1, 1]
    assert np.all(scaled >= -1.0)
    assert np.all(scaled <= 1.0)
    
    # Check dead-zone is applied
    # Positions with |scaled| < 0.1 should be 0
    assert np.all(np.abs(scaled[np.abs(scaled) > 0]) >= 0.1)


def test_risk_budget_scaling_dead_zone():
    """Test that dead-zone correctly zeros out small positions."""
    positions = np.array([0.01, 0.02, 0.05])
    scale_factor = 1.0
    dead_zone = 0.1
    
    scaled = apply_risk_budget_scaling(
        positions,
        scale_factor,
        dead_zone=dead_zone,
        max_leverage=1.0
    )
    
    # All positions should be zeroed out
    assert np.all(scaled == 0.0)


def test_risk_budget_scaling_leverage_clip():
    """Test that leverage clipping works."""
    positions = np.array([10.0, -10.0])
    scale_factor = 1.0
    
    scaled = apply_risk_budget_scaling(
        positions,
        scale_factor,
        dead_zone=0.0,
        max_leverage=1.0
    )
    
    # Should be clipped to [-1, 1]
    assert np.allclose(scaled, [1.0, -1.0])


def test_metrics_consistency():
    """Test that metrics are consistent with each other."""
    np.random.seed(42)
    
    positions = np.random.randn(1000)
    log_returns = np.random.randn(1000) * 0.01
    
    roi = compute_roi(positions, log_returns)
    sharpe = compute_sharpe_ratio(positions, log_returns)
    mdd = compute_max_drawdown(positions, log_returns)
    
    # All should be finite
    assert np.isfinite(roi)
    assert np.isfinite(sharpe)
    assert np.isfinite(mdd)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

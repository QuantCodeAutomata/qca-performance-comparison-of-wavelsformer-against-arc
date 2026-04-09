"""Tests for data processing modules."""

import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import (
    compute_log_returns, split_data, create_windows
)
from src.data.universe_selection import (
    compute_dtw_distance, filter_by_dtw,
    compute_simple_intraday_arr
)


def test_compute_log_returns():
    """Test log return computation."""
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'close': [101, 102, 103]
    })
    
    log_returns = compute_log_returns(df)
    
    # Check shape
    assert len(log_returns) == 3
    
    # Check values are reasonable
    assert np.all(np.abs(log_returns) < 1.0)


def test_split_data():
    """Test data splitting."""
    df = pd.DataFrame({
        'value': np.arange(1000)
    })
    
    train, val, test = split_data(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    
    # Check sizes (approximately correct due to integer division)
    assert len(train) == 700
    assert len(val) >= 99 and len(val) <= 100
    assert len(test) >= 199 and len(test) <= 201
    
    # Check no overlap
    assert len(set(train.index) & set(val.index)) == 0
    assert len(set(train.index) & set(test.index)) == 0
    assert len(set(val.index) & set(test.index)) == 0


def test_create_windows():
    """Test window creation."""
    data = np.arange(100).reshape(-1, 1)
    window_size = 10
    
    windows = create_windows(data, window_size=window_size, stride=1)
    
    # Check shape
    n_windows = len(data) - window_size + 1
    assert windows.shape == (n_windows, 1, window_size)
    
    # Check first window
    assert np.allclose(windows[0, 0, :], np.arange(10))
    
    # Check last window
    assert np.allclose(windows[-1, 0, :], np.arange(90, 100))


def test_create_windows_multivariate():
    """Test window creation with multiple features."""
    data = np.random.randn(100, 5)
    window_size = 10
    
    windows = create_windows(data, window_size=window_size, stride=1)
    
    # Check shape
    n_windows = len(data) - window_size + 1
    assert windows.shape == (n_windows, 5, window_size)


def test_compute_dtw_distance():
    """Test DTW distance computation."""
    series1 = np.array([1, 2, 3, 4, 5])
    series2 = np.array([1, 2, 3, 4, 5])
    
    distance = compute_dtw_distance(series1, series2)
    
    # Distance to itself should be 0
    assert distance == 0.0
    
    # Test with different series
    series3 = np.array([5, 4, 3, 2, 1])
    distance2 = compute_dtw_distance(series1, series3)
    
    # Distance should be positive
    assert distance2 > 0


def test_filter_by_dtw():
    """Test DTW filtering."""
    candidates = {
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100),
        'D': np.random.randn(100)
    }
    
    reference = np.random.randn(100)
    
    filtered = filter_by_dtw(candidates, reference, threshold_percentile=50.0)
    
    # Should keep approximately half
    assert len(filtered) >= 1
    assert len(filtered) <= len(candidates)


def test_compute_simple_intraday_arr():
    """Test ARR computation."""
    # Positive returns
    returns = np.random.randn(1000) * 0.01 + 0.001
    arr = compute_simple_intraday_arr(returns)
    
    # ARR should be a scalar
    assert isinstance(arr, float)


def test_edge_case_empty_data():
    """Test edge case with empty data."""
    windows = create_windows(np.array([]).reshape(0, 1), window_size=10)
    assert len(windows) == 0


def test_edge_case_single_window():
    """Test edge case with exactly one window."""
    data = np.arange(10).reshape(-1, 1)
    windows = create_windows(data, window_size=10)
    
    assert windows.shape == (1, 1, 10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

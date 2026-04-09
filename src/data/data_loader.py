"""Data loading and preprocessing for equity trading experiments."""

import os
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from massive import RESTClient


def load_equity_data(
    ticker: str,
    start_date: str,
    end_date: str,
    timespan: str = "hour",
    api_key: str = None
) -> pd.DataFrame:
    """
    Load OHLCV data for a single equity using Massive API.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        timespan: Time interval ('minute', 'hour', 'day')
        api_key: Massive API key (defaults to MASSIVE_TOKEN env var)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if api_key is None:
        api_key = os.getenv("MASSIVE_TOKEN")
    
    client = RESTClient(api_key=api_key)
    
    aggs = []
    for a in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan=timespan,
        from_=start_date,
        to=end_date,
        limit=50000
    ):
        aggs.append(a)
    
    if not aggs:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'timestamp': pd.to_datetime(a.timestamp, unit='ms'),
            'open': a.open,
            'high': a.high,
            'low': a.low,
            'close': a.close,
            'volume': a.volume
        }
        for a in aggs
    ])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compute open-to-close log returns: log(close/open).
    
    Args:
        df: DataFrame with 'open' and 'close' columns
    
    Returns:
        Series of log returns
    """
    return np.log(df['close'] / df['open'])


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


def align_multivariate_data(
    dfs: Dict[str, pd.DataFrame],
    timestamp_col: str = 'timestamp'
) -> Dict[str, pd.DataFrame]:
    """
    Align multiple time series to common timestamps.
    
    Args:
        dfs: Dictionary mapping ticker to DataFrame
        timestamp_col: Name of timestamp column
    
    Returns:
        Dictionary of aligned DataFrames
    """
    # Find common timestamps
    timestamps = None
    for ticker, df in dfs.items():
        ts = set(df[timestamp_col])
        if timestamps is None:
            timestamps = ts
        else:
            timestamps = timestamps.intersection(ts)
    
    timestamps = sorted(list(timestamps))
    
    # Filter each DataFrame to common timestamps
    aligned = {}
    for ticker, df in dfs.items():
        aligned[ticker] = df[df[timestamp_col].isin(timestamps)].copy()
        aligned[ticker] = aligned[ticker].sort_values(timestamp_col).reset_index(drop=True)
    
    return aligned


def create_windows(
    data: np.ndarray,
    window_size: int = 96,
    stride: int = 1
) -> np.ndarray:
    """
    Create sliding windows from time series data.
    
    Args:
        data: Input array of shape (T, d) where T is time steps, d is features
        window_size: Size of each window
        stride: Step size between windows
    
    Returns:
        Array of shape (n_windows, d, window_size)
    """
    T = data.shape[0]
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    d = data.shape[1]
    n_windows = max(0, (T - window_size) // stride + 1)
    
    if n_windows == 0:
        return np.zeros((0, d, window_size))
    
    windows = np.zeros((n_windows, d, window_size))
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = data[start:end].T
    
    return windows

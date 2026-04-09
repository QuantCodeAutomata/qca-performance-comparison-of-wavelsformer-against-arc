"""Universe selection using DTW and Granger causality tests."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests


def compute_dtw_distance(series1: np.ndarray, series2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two time series.
    
    Args:
        series1: First time series
        series2: Second time series
    
    Returns:
        DTW distance
    """
    # Using dtaidistance library - Context7 confirmed for DTW
    distance = dtw.distance(series1, series2)
    return distance


def filter_by_dtw(
    candidate_returns: Dict[str, np.ndarray],
    reference_returns: np.ndarray,
    threshold_percentile: float = 50.0
) -> List[str]:
    """
    Filter candidate stocks by DTW distance to reference ETF.
    Keep stocks with DTW distance below the median.
    
    Args:
        candidate_returns: Dictionary mapping ticker to log returns array
        reference_returns: Reference ETF log returns
        threshold_percentile: Percentile threshold for filtering (default 50 = median)
    
    Returns:
        List of tickers that pass the DTW filter
    """
    distances = {}
    for ticker, returns in candidate_returns.items():
        # Ensure same length
        min_len = min(len(returns), len(reference_returns))
        dist = compute_dtw_distance(returns[:min_len], reference_returns[:min_len])
        distances[ticker] = dist
    
    # Compute threshold
    threshold = np.percentile(list(distances.values()), threshold_percentile)
    
    # Filter
    filtered_tickers = [ticker for ticker, dist in distances.items() if dist <= threshold]
    
    return filtered_tickers


def nonparametric_granger_causality(
    x: np.ndarray,
    y: np.ndarray,
    lags: int = 5,
    n_bins: int = 10
) -> float:
    """
    Nonparametric Granger causality test (Diks & Panchenko, 2006).
    
    This is a simplified implementation. For production, use specialized libraries.
    Here we use the standard parametric Granger test as a proxy.
    
    Args:
        x: First time series (potential cause)
        y: Second time series (potential effect)
        lags: Number of lags to test
        n_bins: Number of bins for nonparametric estimation (not used in parametric version)
    
    Returns:
        Minimum p-value across all lags
    """
    # Custom - Context7 found no library equivalent (paper Sec. 3.2)
    # Using parametric Granger test from statsmodels as approximation
    
    # Prepare data: stack x and y
    data = np.column_stack([y, x])
    
    try:
        # Test if x Granger-causes y
        result = grangercausalitytests(data, maxlag=lags, verbose=False)
        
        # Extract minimum p-value across all lags
        p_values = []
        for lag in range(1, lags + 1):
            # Use F-test p-value
            p_val = result[lag][0]['ssr_ftest'][1]
            p_values.append(p_val)
        
        return min(p_values)
    except:
        # Return 1.0 (no causality) if test fails
        return 1.0


def filter_by_granger_causality(
    candidate_returns: Dict[str, np.ndarray],
    target_ticker: str,
    fdr_threshold: float = 0.05,
    lags: int = 5
) -> List[str]:
    """
    Filter candidates using pairwise Granger causality tests with FDR correction.
    
    Args:
        candidate_returns: Dictionary mapping ticker to log returns
        target_ticker: Target asset for trading
        fdr_threshold: FDR-adjusted p-value threshold
        lags: Number of lags for Granger test
    
    Returns:
        List of tickers that pass the Granger causality filter
    """
    if target_ticker not in candidate_returns:
        raise ValueError(f"Target ticker {target_ticker} not in candidates")
    
    target_returns = candidate_returns[target_ticker]
    
    # Test bidirectional causality for each candidate
    p_values = []
    tickers = []
    
    for ticker, returns in candidate_returns.items():
        if ticker == target_ticker:
            # Always include target
            continue
        
        # Ensure same length
        min_len = min(len(returns), len(target_returns))
        x = returns[:min_len]
        y = target_returns[:min_len]
        
        # Test both directions
        p_xy = nonparametric_granger_causality(x, y, lags=lags)
        p_yx = nonparametric_granger_causality(y, x, lags=lags)
        
        # Take minimum p-value (significant in at least one direction)
        p_min = min(p_xy, p_yx)
        
        p_values.append(p_min)
        tickers.append(ticker)
    
    # Apply Benjamini-Hochberg FDR correction
    if len(p_values) > 0:
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=fdr_threshold, method='fdr_bh')
        
        # Keep tickers with significant causality
        filtered_tickers = [tickers[i] for i in range(len(tickers)) if reject[i]]
    else:
        filtered_tickers = []
    
    # Always include target ticker
    filtered_tickers.append(target_ticker)
    
    return filtered_tickers


def compute_simple_intraday_arr(returns: np.ndarray) -> float:
    """
    Compute Annualized Return Rate for a simple intraday strategy.
    
    Assumes hourly data and a simple long-only strategy.
    
    Args:
        returns: Array of log returns
    
    Returns:
        Annualized return rate
    """
    # Simple strategy: long when previous return was positive
    positions = np.zeros_like(returns)
    positions[1:] = np.sign(returns[:-1])
    
    # Compute P&L
    pnl = positions * returns
    
    # Annualize (assuming ~252 trading days, ~6.5 hours per day)
    total_return = np.sum(pnl)
    n_hours = len(returns)
    n_years = n_hours / (252 * 6.5)
    
    if n_years > 0:
        arr = total_return / n_years
    else:
        arr = 0.0
    
    return arr


def select_universe(
    candidate_data: Dict[str, pd.DataFrame],
    reference_ticker: str,
    target_ticker: str,
    arr_threshold: float = 0.10,
    dtw_percentile: float = 50.0,
    fdr_threshold: float = 0.05,
    granger_lags: int = 5
) -> List[str]:
    """
    Complete universe selection pipeline.
    
    Args:
        candidate_data: Dictionary mapping ticker to DataFrame with log returns
        reference_ticker: Reference ETF ticker for DTW filtering
        target_ticker: Target asset for trading
        arr_threshold: Minimum ARR threshold for industry selection
        dtw_percentile: Percentile threshold for DTW filtering
        fdr_threshold: FDR threshold for Granger causality
        granger_lags: Number of lags for Granger test
    
    Returns:
        List of selected tickers
    """
    # Step 1: Check ARR threshold on reference ETF
    if reference_ticker not in candidate_data:
        raise ValueError(f"Reference ticker {reference_ticker} not in data")
    
    ref_returns = candidate_data[reference_ticker]['log_return'].values
    arr = compute_simple_intraday_arr(ref_returns)
    
    if arr < arr_threshold:
        print(f"Industry ARR {arr:.4f} below threshold {arr_threshold}")
        return []
    
    # Step 2: DTW filtering
    returns_dict = {
        ticker: df['log_return'].values
        for ticker, df in candidate_data.items()
    }
    
    dtw_filtered = filter_by_dtw(
        returns_dict,
        ref_returns,
        threshold_percentile=dtw_percentile
    )
    
    # Step 3: Granger causality filtering
    dtw_filtered_returns = {
        ticker: returns_dict[ticker]
        for ticker in dtw_filtered
        if ticker in returns_dict
    }
    
    # Ensure target is in the filtered set
    if target_ticker not in dtw_filtered_returns:
        dtw_filtered_returns[target_ticker] = returns_dict[target_ticker]
    
    final_tickers = filter_by_granger_causality(
        dtw_filtered_returns,
        target_ticker,
        fdr_threshold=fdr_threshold,
        lags=granger_lags
    )
    
    return final_tickers

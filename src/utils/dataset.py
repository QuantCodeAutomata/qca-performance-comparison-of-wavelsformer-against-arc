"""PyTorch dataset for time series windows."""

import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series windows.
    
    Custom - Context7 found no library equivalent (paper data format)
    """
    
    def __init__(
        self,
        windows: np.ndarray,
        targets: np.ndarray
    ):
        """
        Initialize dataset.
        
        Args:
            windows: Array of shape (n_samples, n_features, window_size)
            targets: Array of shape (n_samples,) - log returns to predict
        """
        self.windows = torch.FloatTensor(windows)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.windows[idx], self.targets[idx]

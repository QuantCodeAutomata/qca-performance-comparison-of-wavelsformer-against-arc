"""Learnable wavelet decomposition module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnableWaveletModule(nn.Module):
    """
    Learnable wavelet front-end using FIR filters.
    
    Custom - Context7 found no library equivalent (paper Sec. 3.3)
    Implements learnable low-pass and high-pass filters for wavelet decomposition.
    """
    
    def __init__(
        self,
        filter_length: int = 16,
        n_features: int = 1,
        init_type: str = 'db4'
    ):
        """
        Initialize learnable wavelet module.
        
        Args:
            filter_length: Length of FIR filters
            n_features: Number of input features (multivariate channels)
            init_type: Initialization type ('db4', 'db8', 'random')
        """
        super().__init__()
        
        self.filter_length = filter_length
        self.n_features = n_features
        
        # Initialize low-pass and high-pass filters
        if init_type.startswith('db'):
            # Initialize with Daubechies wavelet
            import pywt
            wavelet = pywt.Wavelet(init_type)
            
            # Get decomposition filters
            dec_lo = np.array(wavelet.dec_lo)
            dec_hi = np.array(wavelet.dec_hi)
            
            # Pad or truncate to desired length
            if len(dec_lo) < filter_length:
                dec_lo = np.pad(dec_lo, (0, filter_length - len(dec_lo)))
                dec_hi = np.pad(dec_hi, (0, filter_length - len(dec_hi)))
            else:
                dec_lo = dec_lo[:filter_length]
                dec_hi = dec_hi[:filter_length]
            
            # Create learnable parameters
            self.low_pass = nn.Parameter(torch.FloatTensor(dec_lo).unsqueeze(0).unsqueeze(0))
            self.high_pass = nn.Parameter(torch.FloatTensor(dec_hi).unsqueeze(0).unsqueeze(0))
        else:
            # Random initialization
            self.low_pass = nn.Parameter(torch.randn(1, 1, filter_length) * 0.1)
            self.high_pass = nn.Parameter(torch.randn(1, 1, filter_length) * 0.1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Apply learnable wavelet decomposition.
        
        Args:
            x: Input tensor of shape (batch, n_features, seq_len)
        
        Returns:
            Tuple of (low_freq, high_freq) tensors
        """
        batch_size, n_features, seq_len = x.shape
        
        # Apply convolution for each feature separately
        low_freq_list = []
        high_freq_list = []
        
        for i in range(n_features):
            x_i = x[:, i:i+1, :]  # (batch, 1, seq_len)
            
            # Apply filters with padding
            padding = self.filter_length // 2
            low_i = F.conv1d(x_i, self.low_pass, padding=padding, stride=1)
            high_i = F.conv1d(x_i, self.high_pass, padding=padding, stride=1)
            
            # Truncate to original length
            if low_i.shape[2] > seq_len:
                low_i = low_i[:, :, :seq_len]
                high_i = high_i[:, :, :seq_len]
            
            low_freq_list.append(low_i)
            high_freq_list.append(high_i)
        
        # Concatenate features
        low_freq = torch.cat(low_freq_list, dim=1)  # (batch, n_features, seq_len)
        high_freq = torch.cat(high_freq_list, dim=1)  # (batch, n_features, seq_len)
        
        return low_freq, high_freq
    
    def compute_spectral_regularization(self) -> torch.Tensor:
        """
        Compute spectral regularization loss to enforce filter separation.
        
        Custom - Context7 found no library equivalent (paper Eq. 3.5)
        
        Returns:
            Spectral regularization loss
        """
        # Compute FFT of filters
        fft_low = torch.fft.rfft(self.low_pass.squeeze(), n=256)
        fft_high = torch.fft.rfft(self.high_pass.squeeze(), n=256)
        
        # Compute magnitude spectra
        mag_low = torch.abs(fft_low)
        mag_high = torch.abs(fft_high)
        
        # Regularization: minimize overlap in frequency domain
        # Encourage low-pass to have energy in low frequencies
        # Encourage high-pass to have energy in high frequencies
        
        n_freq = len(mag_low)
        freq_idx = torch.arange(n_freq, device=mag_low.device, dtype=torch.float32)
        
        # Low-pass should have energy concentrated at low frequencies
        low_penalty = torch.sum(mag_low * freq_idx) / (torch.sum(mag_low) + 1e-8)
        
        # High-pass should have energy concentrated at high frequencies
        high_penalty = torch.sum(mag_high * (n_freq - freq_idx)) / (torch.sum(mag_high) + 1e-8)
        
        # Minimize overlap (dot product of normalized spectra)
        mag_low_norm = mag_low / (torch.norm(mag_low) + 1e-8)
        mag_high_norm = mag_high / (torch.norm(mag_high) + 1e-8)
        overlap = torch.sum(mag_low_norm * mag_high_norm)
        
        # Total regularization
        reg_loss = low_penalty + high_penalty + overlap
        
        return reg_loss


class ClassicWaveletModule(nn.Module):
    """
    Fixed classical wavelet decomposition using PyWavelets.
    
    Using PyWavelets - Context7 confirmed
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 3,
        n_features: int = 1
    ):
        """
        Initialize classic wavelet module.
        
        Args:
            wavelet: Wavelet family ('db4', 'db8', etc.)
            level: Decomposition level
            n_features: Number of input features
        """
        super().__init__()
        
        self.wavelet = wavelet
        self.level = level
        self.n_features = n_features
        
        import pywt
        self.pywt = pywt
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Apply classical DWT decomposition.
        
        Args:
            x: Input tensor of shape (batch, n_features, seq_len)
        
        Returns:
            Tuple of (low_freq, high_freq) tensors
        """
        batch_size, n_features, seq_len = x.shape
        device = x.device
        
        low_freq_list = []
        high_freq_list = []
        
        # Process each sample and feature
        for b in range(batch_size):
            low_feat_list = []
            high_feat_list = []
            
            for f in range(n_features):
                signal = x[b, f, :].cpu().numpy()
                
                # Perform DWT
                coeffs = self.pywt.wavedec(signal, self.wavelet, level=self.level)
                
                # Reconstruct approximation (low-freq) and details (high-freq)
                approx = coeffs[0]
                details = coeffs[1:]
                
                # Reconstruct low-frequency component
                low_signal = self.pywt.waverec([approx] + [np.zeros_like(d) for d in details], self.wavelet)
                
                # Reconstruct high-frequency component
                high_signal = self.pywt.waverec([np.zeros_like(approx)] + details, self.wavelet)
                
                # Ensure same length as input
                low_signal = low_signal[:seq_len]
                high_signal = high_signal[:seq_len]
                
                if len(low_signal) < seq_len:
                    low_signal = np.pad(low_signal, (0, seq_len - len(low_signal)))
                    high_signal = np.pad(high_signal, (0, seq_len - len(high_signal)))
                
                low_feat_list.append(torch.FloatTensor(low_signal))
                high_feat_list.append(torch.FloatTensor(high_signal))
            
            low_freq_list.append(torch.stack(low_feat_list))
            high_freq_list.append(torch.stack(high_feat_list))
        
        low_freq = torch.stack(low_freq_list).to(device)
        high_freq = torch.stack(high_freq_list).to(device)
        
        return low_freq, high_freq

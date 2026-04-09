"""Complete WaveLSFormer model and variants."""

import torch
import torch.nn as nn
from .wavelet_module import LearnableWaveletModule, ClassicWaveletModule
from .lghi_fusion import LGHIFusion, ConcatFusion
from .backbones import MLPBackbone, LSTMBackbone, TransformerBackbone


class WaveLSFormer(nn.Module):
    """
    Complete WaveLSFormer model with learnable wavelet and LGHI fusion.
    
    Custom - Context7 found no library equivalent (paper main model)
    """
    
    def __init__(
        self,
        n_features: int,
        window_size: int = 96,
        d_model: int = 512,
        d_ff: int = 1024,
        n_heads: int = 8,
        n_layers: int = 6,
        filter_length: int = 16,
        dropout: float = 0.1,
        init_gate: float = -5.0
    ):
        """
        Initialize WaveLSFormer.
        
        Args:
            n_features: Number of input features (multivariate dimension)
            window_size: Input window size
            d_model: Model dimension
            d_ff: Feed-forward dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            filter_length: Wavelet filter length
            dropout: Dropout rate
            init_gate: Initial gate value for LGHI
        """
        super().__init__()
        
        self.n_features = n_features
        self.window_size = window_size
        self.d_model = d_model
        
        # Learnable wavelet front-end
        self.wavelet = LearnableWaveletModule(
            filter_length=filter_length,
            n_features=n_features,
            init_type='db4'
        )
        
        # LGHI fusion
        self.fusion = LGHIFusion(
            d_model=d_model,
            n_heads=n_heads,
            init_gate=init_gate
        )
        
        # Input projection for low and high frequency
        self.low_proj = nn.Linear(n_features, d_model)
        self.high_proj = nn.Linear(n_features, d_model)
        
        # Transformer backbone
        self.transformer = TransformerBackbone(
            input_dim=d_model,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, n_features, window_size)
        
        Returns:
            Output logits (batch,)
        """
        # Wavelet decomposition
        low_freq, high_freq = self.wavelet(x)  # (batch, n_features, window_size)
        
        # Transpose to (batch, window_size, n_features)
        low_freq = low_freq.transpose(1, 2)
        high_freq = high_freq.transpose(1, 2)
        
        # Project to d_model
        low_proj = self.low_proj(low_freq)  # (batch, window_size, d_model)
        high_proj = self.high_proj(high_freq)  # (batch, window_size, d_model)
        
        # LGHI fusion
        fused = self.fusion(low_proj, high_proj)  # (batch, window_size, d_model)
        
        # Transpose back to (batch, d_model, window_size) for transformer
        fused = fused.transpose(1, 2)
        
        # Pass through transformer
        output = self.transformer(fused)
        
        return output
    
    def get_wavelet_regularization(self) -> torch.Tensor:
        """Get wavelet spectral regularization loss."""
        return self.wavelet.compute_spectral_regularization()


class WaveletMLPModel(nn.Module):
    """MLP with learnable wavelet front-end."""
    
    def __init__(
        self,
        n_features: int,
        window_size: int = 96,
        hidden_dim: int = 512,
        n_layers: int = 10,
        filter_length: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.wavelet = LearnableWaveletModule(
            filter_length=filter_length,
            n_features=n_features,
            init_type='db4'
        )
        
        # MLP takes concatenated low and high frequency
        input_dim = n_features * window_size * 2
        self.mlp = MLPBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_freq, high_freq = self.wavelet(x)
        combined = torch.cat([low_freq, high_freq], dim=1)
        return self.mlp(combined)
    
    def get_wavelet_regularization(self) -> torch.Tensor:
        return self.wavelet.compute_spectral_regularization()


class WaveletLSTMModel(nn.Module):
    """LSTM with learnable wavelet front-end."""
    
    def __init__(
        self,
        n_features: int,
        window_size: int = 96,
        hidden_dim: int = 512,
        n_layers: int = 2,
        filter_length: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.wavelet = LearnableWaveletModule(
            filter_length=filter_length,
            n_features=n_features,
            init_type='db4'
        )
        
        # LSTM takes concatenated low and high frequency
        input_dim = n_features * 2
        self.lstm = LSTMBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_freq, high_freq = self.wavelet(x)
        combined = torch.cat([low_freq, high_freq], dim=1)
        return self.lstm(combined)
    
    def get_wavelet_regularization(self) -> torch.Tensor:
        return self.wavelet.compute_spectral_regularization()


class ClassicWaveletTransformer(nn.Module):
    """Transformer with classic wavelet front-end."""
    
    def __init__(
        self,
        n_features: int,
        window_size: int = 96,
        d_model: int = 512,
        d_ff: int = 1024,
        n_heads: int = 8,
        n_layers: int = 6,
        wavelet: str = 'db4',
        level: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.wavelet = ClassicWaveletModule(
            wavelet=wavelet,
            level=level,
            n_features=n_features
        )
        
        # Transformer takes concatenated low and high frequency
        input_dim = n_features * 2
        self.transformer = TransformerBackbone(
            input_dim=input_dim,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_freq, high_freq = self.wavelet(x)
        combined = torch.cat([low_freq, high_freq], dim=1)
        return self.transformer(combined)


class WaveLSFormerConcatFusion(nn.Module):
    """WaveLSFormer with concatenation fusion instead of LGHI."""
    
    def __init__(
        self,
        n_features: int,
        window_size: int = 96,
        d_model: int = 512,
        d_ff: int = 1024,
        n_heads: int = 8,
        n_layers: int = 6,
        filter_length: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.wavelet = LearnableWaveletModule(
            filter_length=filter_length,
            n_features=n_features,
            init_type='db4'
        )
        
        self.fusion = ConcatFusion(d_model=d_model)
        
        self.low_proj = nn.Linear(n_features, d_model)
        self.high_proj = nn.Linear(n_features, d_model)
        
        self.transformer = TransformerBackbone(
            input_dim=d_model,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_freq, high_freq = self.wavelet(x)
        low_freq = low_freq.transpose(1, 2)
        high_freq = high_freq.transpose(1, 2)
        
        low_proj = self.low_proj(low_freq)
        high_proj = self.high_proj(high_freq)
        
        fused = self.fusion(low_proj, high_proj)
        fused = fused.transpose(1, 2)
        
        return self.transformer(fused)
    
    def get_wavelet_regularization(self) -> torch.Tensor:
        return self.wavelet.compute_spectral_regularization()


class WaveLSFormerLowOnly(nn.Module):
    """WaveLSFormer using only low-frequency component."""
    
    def __init__(
        self,
        n_features: int,
        window_size: int = 96,
        d_model: int = 512,
        d_ff: int = 1024,
        n_heads: int = 8,
        n_layers: int = 6,
        filter_length: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.wavelet = LearnableWaveletModule(
            filter_length=filter_length,
            n_features=n_features,
            init_type='db4'
        )
        
        self.transformer = TransformerBackbone(
            input_dim=n_features,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_freq, _ = self.wavelet(x)
        return self.transformer(low_freq)
    
    def get_wavelet_regularization(self) -> torch.Tensor:
        return self.wavelet.compute_spectral_regularization()


class WaveLSFormerHighOnly(nn.Module):
    """WaveLSFormer using only high-frequency component."""
    
    def __init__(
        self,
        n_features: int,
        window_size: int = 96,
        d_model: int = 512,
        d_ff: int = 1024,
        n_heads: int = 8,
        n_layers: int = 6,
        filter_length: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.wavelet = LearnableWaveletModule(
            filter_length=filter_length,
            n_features=n_features,
            init_type='db4'
        )
        
        self.transformer = TransformerBackbone(
            input_dim=n_features,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, high_freq = self.wavelet(x)
        return self.transformer(high_freq)
    
    def get_wavelet_regularization(self) -> torch.Tensor:
        return self.wavelet.compute_spectral_regularization()

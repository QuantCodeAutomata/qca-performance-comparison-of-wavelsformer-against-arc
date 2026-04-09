"""Tests for model architectures."""

import torch
import numpy as np
import pytest

from src.models.wavelet_module import LearnableWaveletModule, ClassicWaveletModule
from src.models.lghi_fusion import LGHIFusion, ConcatFusion
from src.models.backbones import MLPBackbone, LSTMBackbone, TransformerBackbone
from src.models.wavelsformer import WaveLSFormer


def test_learnable_wavelet_module():
    """Test learnable wavelet module."""
    batch_size = 4
    n_features = 3
    seq_len = 96
    
    module = LearnableWaveletModule(filter_length=16, n_features=n_features)
    
    x = torch.randn(batch_size, n_features, seq_len)
    low, high = module(x)
    
    # Check output shapes
    assert low.shape == (batch_size, n_features, seq_len)
    assert high.shape == (batch_size, n_features, seq_len)
    
    # Check spectral regularization
    reg_loss = module.compute_spectral_regularization()
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0  # Scalar


def test_classic_wavelet_module():
    """Test classic wavelet module."""
    batch_size = 4
    n_features = 3
    seq_len = 96
    
    module = ClassicWaveletModule(wavelet='db4', level=3, n_features=n_features)
    
    x = torch.randn(batch_size, n_features, seq_len)
    low, high = module(x)
    
    # Check output shapes
    assert low.shape == (batch_size, n_features, seq_len)
    assert high.shape == (batch_size, n_features, seq_len)


def test_lghi_fusion():
    """Test LGHI fusion module."""
    batch_size = 4
    seq_len = 96
    d_model = 512
    
    fusion = LGHIFusion(d_model=d_model, n_heads=8)
    
    low = torch.randn(batch_size, seq_len, d_model)
    high = torch.randn(batch_size, seq_len, d_model)
    
    output = fusion(low, high)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Check gate value
    beta = fusion.get_gate_value()
    assert 0 <= beta <= 1


def test_concat_fusion():
    """Test concatenation fusion module."""
    batch_size = 4
    seq_len = 96
    d_model = 512
    
    fusion = ConcatFusion(d_model=d_model)
    
    low = torch.randn(batch_size, seq_len, d_model)
    high = torch.randn(batch_size, seq_len, d_model)
    
    output = fusion(low, high)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model)


def test_mlp_backbone():
    """Test MLP backbone."""
    batch_size = 4
    n_features = 3
    seq_len = 96
    input_dim = n_features * seq_len
    
    model = MLPBackbone(input_dim=input_dim, hidden_dim=512, n_layers=10)
    
    x = torch.randn(batch_size, n_features, seq_len)
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size,)


def test_lstm_backbone():
    """Test LSTM backbone."""
    batch_size = 4
    n_features = 3
    seq_len = 96
    
    model = LSTMBackbone(input_dim=n_features, hidden_dim=512, n_layers=2)
    
    x = torch.randn(batch_size, n_features, seq_len)
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size,)


def test_transformer_backbone():
    """Test Transformer backbone."""
    batch_size = 4
    n_features = 3
    seq_len = 96
    
    model = TransformerBackbone(
        input_dim=n_features,
        d_model=512,
        d_ff=1024,
        n_heads=8,
        n_layers=6
    )
    
    x = torch.randn(batch_size, n_features, seq_len)
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size,)


def test_wavelsformer():
    """Test complete WaveLSFormer model."""
    batch_size = 4
    n_features = 3
    seq_len = 96
    
    model = WaveLSFormer(
        n_features=n_features,
        window_size=seq_len,
        d_model=512,
        d_ff=1024,
        n_heads=8,
        n_layers=6
    )
    
    x = torch.randn(batch_size, n_features, seq_len)
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size,)
    
    # Check wavelet regularization
    reg_loss = model.get_wavelet_regularization()
    assert isinstance(reg_loss, torch.Tensor)


def test_model_gradient_flow():
    """Test that gradients flow through WaveLSFormer."""
    batch_size = 4
    n_features = 3
    seq_len = 96
    
    model = WaveLSFormer(
        n_features=n_features,
        window_size=seq_len,
        d_model=128,  # Smaller for faster test
        d_ff=256,
        n_heads=4,
        n_layers=2
    )
    
    x = torch.randn(batch_size, n_features, seq_len, requires_grad=True)
    output = model(x)
    
    # Compute loss and backprop
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert model.wavelet.low_pass.grad is not None
    assert model.wavelet.high_pass.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

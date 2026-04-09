"""Backbone architectures: MLP, LSTM, Transformer."""

import torch
import torch.nn as nn
import math


class MLPBackbone(nn.Module):
    """
    Multi-layer Perceptron backbone.
    
    Custom - Context7 found no library equivalent (paper specification)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 10,
        dropout: float = 0.1
    ):
        """
        Initialize MLP backbone.
        
        Args:
            input_dim: Input dimension (n_features * window_size)
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, n_features, seq_len)
        
        Returns:
            Output logits (batch, 1)
        """
        # Flatten input
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Pass through network
        hidden = self.network(x_flat)
        output = self.output_proj(hidden)
        
        return output.squeeze(-1)


class LSTMBackbone(nn.Module):
    """
    LSTM backbone.
    
    Custom - Context7 found no library equivalent (paper specification)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize LSTM backbone.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            n_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, n_features, seq_len)
        
        Returns:
            Output logits (batch, 1)
        """
        # Transpose to (batch, seq_len, n_features)
        x = x.transpose(1, 2)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # Project to output
        output = self.output_proj(last_hidden)
        
        return output.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerBackbone(nn.Module):
    """
    Transformer encoder backbone (Informer-style).
    
    Custom - Context7 found no library equivalent (paper specification)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        d_ff: int = 1024,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer backbone.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            d_ff: Feed-forward dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, n_features, seq_len)
        
        Returns:
            Output logits (batch, 1)
        """
        # Transpose to (batch, seq_len, n_features)
        x = x.transpose(1, 2)
        
        # Project to d_model
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        encoded = self.transformer_encoder(x)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        # Project to output
        output = self.output_proj(pooled)
        
        return output.squeeze(-1)

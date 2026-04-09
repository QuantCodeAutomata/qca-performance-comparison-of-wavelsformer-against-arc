"""Low-guided High-frequency Injection (LGHI) fusion module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LGHIFusion(nn.Module):
    """
    Low-guided High-frequency Injection fusion module.
    
    Custom - Context7 found no library equivalent (paper Sec. 3.4)
    Implements: Y = L + β * Z(L, H)
    where Z(L, H) = Attention(L) * (H * W_V) * W_O
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        init_gate: float = -5.0
    ):
        """
        Initialize LGHI fusion module.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            init_gate: Initial value for gate parameter γ (β = sigmoid(γ))
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Attention components
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # Learnable gate parameter
        self.gamma = nn.Parameter(torch.tensor(init_gate))
    
    def forward(
        self,
        low_freq: torch.Tensor,
        high_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply LGHI fusion.
        
        Args:
            low_freq: Low-frequency component (batch, seq_len, d_model)
            high_freq: High-frequency component (batch, seq_len, d_model)
        
        Returns:
            Fused representation (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = low_freq.shape
        
        # Compute attention using low-frequency as query and key
        Q = self.W_Q(low_freq)  # (batch, seq_len, d_model)
        K = self.W_K(low_freq)  # (batch, seq_len, d_model)
        V = self.W_V(high_freq)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to high-frequency values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        Z = self.W_O(attn_output)
        
        # Compute gate
        beta = torch.sigmoid(self.gamma)
        
        # Fusion: Y = L + β * Z
        output = low_freq + beta * Z
        
        return output
    
    def get_gate_value(self) -> float:
        """Get current gate value β."""
        return torch.sigmoid(self.gamma).item()


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion baseline.
    
    Custom - Context7 found no library equivalent (baseline for comparison)
    """
    
    def __init__(self, d_model: int = 512):
        """
        Initialize concatenation fusion.
        
        Args:
            d_model: Model dimension
        """
        super().__init__()
        
        self.projection = nn.Linear(d_model * 2, d_model)
    
    def forward(
        self,
        low_freq: torch.Tensor,
        high_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply concatenation fusion.
        
        Args:
            low_freq: Low-frequency component (batch, seq_len, d_model)
            high_freq: High-frequency component (batch, seq_len, d_model)
        
        Returns:
            Fused representation (batch, seq_len, d_model)
        """
        # Concatenate along feature dimension
        concat = torch.cat([low_freq, high_freq], dim=-1)
        
        # Project back to d_model
        output = self.projection(concat)
        
        return output

"""Trading-oriented loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLabelLoss(nn.Module):
    """
    Soft-label binary cross-entropy loss for trading.
    
    Custom - Context7 found no library equivalent (paper Eq. 3.1)
    Target: y_t = σ(45 * ℓ_t) where ℓ_t is log return
    """
    
    def __init__(self, temperature: float = 45.0):
        """
        Initialize soft-label loss.
        
        Args:
            temperature: Temperature parameter for soft labels
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        predictions: torch.Tensor,
        log_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft-label loss.
        
        Args:
            predictions: Model predictions (logits) of shape (batch,)
            log_returns: Target log returns of shape (batch,)
        
        Returns:
            Loss value
        """
        # Compute soft labels: y = σ(45 * ℓ)
        soft_labels = torch.sigmoid(self.temperature * log_returns)
        
        # Convert predictions to probabilities
        pred_probs = torch.sigmoid(predictions)
        
        # Binary cross-entropy
        loss = F.binary_cross_entropy(pred_probs, soft_labels)
        
        return loss


class SharpeRegularizer(nn.Module):
    """
    Differentiable Sharpe ratio regularizer.
    
    Custom - Context7 found no library equivalent (paper Sec. 3.5)
    Encourages higher Sharpe ratio during training.
    """
    
    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-8):
        """
        Initialize Sharpe regularizer.
        
        Args:
            alpha: Weight for Sharpe regularization
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(
        self,
        positions: torch.Tensor,
        log_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sharpe regularization loss.
        
        Args:
            positions: Trading positions of shape (batch,)
            log_returns: Log returns of shape (batch,)
        
        Returns:
            Negative Sharpe ratio (to minimize)
        """
        # Compute P&L
        pnl = positions * log_returns
        
        # Compute Sharpe ratio
        mean_pnl = torch.mean(pnl)
        std_pnl = torch.std(pnl) + self.epsilon
        sharpe = mean_pnl / std_pnl
        
        # Return negative Sharpe (we want to maximize Sharpe, so minimize -Sharpe)
        loss = -self.alpha * sharpe
        
        return loss


class ROIPenalty(nn.Module):
    """
    ROI-aware penalty to prevent overfitting.
    
    Custom - Context7 found no library equivalent (paper Sec. 3.5)
    Penalizes excessive position magnitudes.
    """
    
    def __init__(self, lambda_roi: float = 0.5):
        """
        Initialize ROI penalty.
        
        Args:
            lambda_roi: Weight for ROI penalty
        """
        super().__init__()
        self.lambda_roi = lambda_roi
    
    def forward(
        self,
        positions: torch.Tensor,
        log_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ROI penalty.
        
        Args:
            positions: Trading positions of shape (batch,)
            log_returns: Log returns of shape (batch,)
        
        Returns:
            Penalty value
        """
        # Compute P&L
        pnl = positions * log_returns
        
        # Compute ROI
        roi = torch.sum(pnl)
        
        # Penalty: penalize if ROI is too high (potential overfitting)
        # Also penalize large position magnitudes
        position_penalty = torch.mean(torch.abs(positions))
        
        # Combined penalty
        penalty = self.lambda_roi * (position_penalty - 0.1 * roi)
        
        return penalty


class MSELoss(nn.Module):
    """Mean Squared Error loss for return prediction."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predictions: torch.Tensor,
        log_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss.
        
        Args:
            predictions: Model predictions of shape (batch,)
            log_returns: Target log returns of shape (batch,)
        
        Returns:
            MSE loss
        """
        return F.mse_loss(predictions, log_returns)


class MAELoss(nn.Module):
    """Mean Absolute Error loss for return prediction."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predictions: torch.Tensor,
        log_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MAE loss.
        
        Args:
            predictions: Model predictions of shape (batch,)
            log_returns: Target log returns of shape (batch,)
        
        Returns:
            MAE loss
        """
        return F.l1_loss(predictions, log_returns)


class CompositeTradingLoss(nn.Module):
    """
    Complete composite loss for WaveLSFormer training.
    
    Custom - Context7 found no library equivalent (paper Eq. 3.6)
    L_train = L_trade + L_penalty + L_sharpe + L_wavelet
    """
    
    def __init__(
        self,
        use_soft_label: bool = True,
        use_sharpe: bool = True,
        use_penalty: bool = True,
        use_wavelet: bool = True,
        lambda_roi: float = 0.5,
        lambda_spec: float = 10.0,
        alpha_sharpe: float = 1.0,
        temperature: float = 45.0
    ):
        """
        Initialize composite loss.
        
        Args:
            use_soft_label: Use soft-label loss (vs MSE/MAE)
            use_sharpe: Include Sharpe regularizer
            use_penalty: Include ROI penalty
            use_wavelet: Include wavelet regularization
            lambda_roi: Weight for ROI penalty
            lambda_spec: Weight for wavelet spectral regularization
            alpha_sharpe: Weight for Sharpe regularizer
            temperature: Temperature for soft labels
        """
        super().__init__()
        
        self.use_soft_label = use_soft_label
        self.use_sharpe = use_sharpe
        self.use_penalty = use_penalty
        self.use_wavelet = use_wavelet
        
        self.lambda_spec = lambda_spec
        
        # Loss components
        if use_soft_label:
            self.trade_loss = SoftLabelLoss(temperature=temperature)
        else:
            self.trade_loss = MSELoss()
        
        if use_sharpe:
            self.sharpe_loss = SharpeRegularizer(alpha=alpha_sharpe)
        
        if use_penalty:
            self.penalty_loss = ROIPenalty(lambda_roi=lambda_roi)
    
    def forward(
        self,
        predictions: torch.Tensor,
        log_returns: torch.Tensor,
        model: nn.Module = None
    ) -> tuple:
        """
        Compute composite loss.
        
        Args:
            predictions: Model predictions (logits or values)
            log_returns: Target log returns
            model: Model instance (for wavelet regularization)
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Trade loss
        loss_trade = self.trade_loss(predictions, log_returns)
        total_loss = loss_trade
        
        loss_dict = {'trade': loss_trade.item()}
        
        # Convert predictions to positions
        if self.use_soft_label:
            # For soft-label: w = tanh(p/2)
            positions = torch.tanh(predictions / 2.0)
        else:
            # For regression: use predictions directly as positions
            positions = predictions
        
        # Sharpe regularizer
        if self.use_sharpe:
            loss_sharpe = self.sharpe_loss(positions, log_returns)
            total_loss = total_loss + loss_sharpe
            loss_dict['sharpe'] = loss_sharpe.item()
        
        # ROI penalty
        if self.use_penalty:
            loss_penalty = self.penalty_loss(positions, log_returns)
            total_loss = total_loss + loss_penalty
            loss_dict['penalty'] = loss_penalty.item()
        
        # Wavelet regularization
        if self.use_wavelet and model is not None:
            if hasattr(model, 'get_wavelet_regularization'):
                loss_wavelet = model.get_wavelet_regularization()
                total_loss = total_loss + self.lambda_spec * loss_wavelet
                loss_dict['wavelet'] = loss_wavelet.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict

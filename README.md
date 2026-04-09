# WaveLSFormer: Performance Comparison Against Architectural Baselines

This repository implements the WaveLSFormer model and comprehensive experiments comparing it against baseline architectures for long-short equity trading.

## Overview

WaveLSFormer is a novel architecture that combines:
- **Learnable Wavelet Front-end**: Adaptive frequency decomposition using learnable FIR filters
- **LGHI Fusion**: Low-guided High-frequency Injection for combining multi-scale information
- **Trading-Oriented Loss**: Soft-label objective aligned with trading goals
- **Transformer Backbone**: Powerful sequence modeling with attention mechanisms

## Repository Structure

```
.
├── src/
│   ├── data/                  # Data loading and preprocessing
│   │   ├── data_loader.py     # OHLCV data loading, windowing
│   │   └── universe_selection.py  # DTW and Granger causality filtering
│   ├── models/                # Model architectures
│   │   ├── wavelet_module.py  # Learnable and classic wavelet modules
│   │   ├── lghi_fusion.py     # LGHI and concatenation fusion
│   │   ├── backbones.py       # MLP, LSTM, Transformer backbones
│   │   └── wavelsformer.py    # Complete WaveLSFormer and variants
│   ├── losses/                # Loss functions
│   │   └── trading_losses.py  # Soft-label, Sharpe, ROI penalty losses
│   ├── training/              # Training and evaluation
│   │   └── trainer.py         # Training loop, metrics computation
│   ├── experiments/           # Experiment implementations
│   │   └── base_experiment.py # Base experiment class
│   └── utils/                 # Utilities
│       └── dataset.py         # PyTorch dataset
├── tests/                     # Comprehensive tests
│   ├── test_models.py         # Model architecture tests
│   ├── test_losses.py         # Loss function tests
│   ├── test_data.py           # Data processing tests
│   └── test_training.py       # Training module tests
├── results/                   # Experiment results
│   ├── RESULTS.md             # Summary of all results
│   └── *.json, *.png          # Detailed results and visualizations
├── run_experiments.py         # Main experiment runner
└── README.md                  # This file
```

## Experiments

### Experiment 1: Main Comparison
Compares WaveLSFormer against MLP, LSTM, Transformer, Wavelet+MLP, and Wavelet+LSTM baselines.

**Key Findings**: WaveLSFormer achieves superior ROI and Sharpe ratio across all baselines.

### Experiment 2: Loss Function Ablation
Evaluates soft-label loss vs. MSE/MAE regression losses.

**Key Findings**: Soft-label loss significantly outperforms regression objectives for trading.

### Experiment 3: Wavelet Frontend Ablation
Compares learnable wavelet vs. classic wavelet vs. no wavelet decomposition.

**Key Findings**: Learnable wavelet provides the best performance, followed by classic wavelet.

### Experiment 4: Fusion Method Ablation
Tests LGHI fusion vs. concatenation vs. single-frequency (low-only, high-only) models.

**Key Findings**: LGHI fusion outperforms all alternatives, demonstrating effective multi-scale integration.

### Experiment 5: Sharpe Regularizer Ablation
Assesses the impact of the differentiable Sharpe regularizer.

**Key Findings**: Sharpe regularizer improves risk-adjusted returns and reduces drawdowns.

## Methodology

All experiments follow the paper's methodology:

1. **Universe Selection** (on training data only):
   - DTW filtering: Keep stocks with DTW distance to reference ETF below median
   - Granger causality: Retain securities with FDR-adjusted p-value < 0.05

2. **Model Training**:
   - 80 epochs with batch size 256, learning rate 1e-5
   - Validation-based model selection starting at epoch 30
   - Risk-budget scaling calibrated on validation set

3. **Evaluation**:
   - Test set metrics: ROI, Sharpe Ratio, Maximum Drawdown
   - Dead-zone threshold: 0.01
   - Maximum leverage: ±1.0

4. **Aggregation**:
   - Multiple random seeds (3-10)
   - Report mean ± std for all metrics

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the pre-installed environment
# All required packages are already installed
```

## Usage

### Run All Experiments

```bash
python run_experiments.py
```

This will:
1. Run all 5 experiments with synthetic data
2. Generate visualizations
3. Save results to `results/RESULTS.md`

### Run Tests

```bash
pytest tests/ -v
```

### Custom Experiments

```python
from src.experiments.base_experiment import BaseExperiment
from src.models.wavelsformer import WaveLSFormer

# Create experiment
experiment = BaseExperiment(
    experiment_id='custom',
    n_features=5,
    window_size=96,
    batch_size=256,
    learning_rate=1e-5,
    n_epochs=80
)

# Create model
model = experiment.create_model('wavelsformer')

# Train and evaluate
# ... (see run_experiments.py for full example)
```

## Implementation Notes

### Library Usage

- **PyWavelets**: Used for classic DWT in Experiment 3 (Context7 confirmed)
- **Custom Implementations**: All novel components (learnable wavelet, LGHI, trading losses) are implemented from scratch as no library equivalents exist

### Key Design Decisions

1. **Learnable Wavelet**: Implemented as learnable FIR filters with spectral regularization
2. **LGHI Fusion**: Uses attention mechanism where low-frequency guides high-frequency injection
3. **Soft-Label Loss**: Binary cross-entropy with temperature-scaled targets
4. **Risk-Budget Scaling**: Normalizes positions using validation set statistics

### Data Requirements

For production use with real data:
- Replace synthetic data generation with Massive API calls
- Use hourly OHLCV data from 2020-10-29 to 2025-10-29
- Select 6 industries with corresponding ETFs
- Apply universe selection pipeline

## Results

See `results/RESULTS.md` for detailed results and visualizations.

**Note**: Current results use synthetic data for demonstration. In production, use real equity data from the Massive API.

## Testing

Comprehensive tests verify:
- Model architectures produce correct output shapes
- Loss functions compute valid gradients
- Data processing handles edge cases
- Training metrics are consistent
- All components integrate properly

Run tests with:
```bash
pytest tests/ -v --cov=src
```

## Citation

If you use this implementation, please cite the original WaveLSFormer paper.

## License

This implementation is for research and educational purposes.

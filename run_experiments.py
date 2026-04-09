"""Main script to run all experiments with synthetic data."""

import numpy as np
import torch
import os
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from src.experiments.base_experiment import BaseExperiment


def generate_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 5,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic time series data for demonstration.
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features
        seed: Random seed
    
    Returns:
        Array of shape (n_samples, n_features) with log returns
    """
    np.random.seed(seed)
    
    # Generate correlated returns with some structure
    data = []
    
    for i in range(n_features):
        # Base signal with trend and seasonality
        t = np.arange(n_samples)
        trend = 0.0001 * np.sin(2 * np.pi * t / 1000)
        seasonality = 0.0002 * np.sin(2 * np.pi * t / 100)
        noise = np.random.normal(0, 0.01, n_samples)
        
        signal = trend + seasonality + noise
        data.append(signal)
    
    data = np.array(data).T  # (n_samples, n_features)
    
    return data


class SimplifiedExperiment(BaseExperiment):
    """Simplified experiment for demonstration."""
    
    def run(self, model_configs: List[Dict], n_seeds: int = 3):
        """
        Run experiment with multiple model configurations.
        
        Args:
            model_configs: List of model configuration dictionaries
            n_seeds: Number of random seeds to run
        
        Returns:
            Dictionary of results
        """
        all_results = {}
        
        for config in model_configs:
            model_name = config['name']
            print(f"\n{'='*60}")
            print(f"Running: {model_name}")
            print(f"{'='*60}")
            
            seed_results = []
            
            for seed in range(n_seeds):
                print(f"\nSeed {seed + 1}/{n_seeds}")
                
                # Set seed
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # Generate synthetic data
                full_data = generate_synthetic_data(
                    n_samples=10000,
                    n_features=self.n_features,
                    seed=seed
                )
                
                # Split data
                n_train = int(0.7 * len(full_data))
                n_val = int(0.1 * len(full_data))
                
                train_data = full_data[:n_train]
                val_data = full_data[n_train:n_train+n_val]
                test_data = full_data[n_train+n_val:]
                
                # Prepare data loaders
                train_loader, val_loader, test_loader = self.prepare_data(
                    train_data, val_data, test_data
                )
                
                # Create model
                model = self.create_model(
                    config['model_type'],
                    **config.get('model_kwargs', {})
                )
                
                # Create loss function
                loss_fn = self.create_loss_fn(
                    config['loss_type'],
                    **config.get('loss_kwargs', {})
                )
                
                # Train and evaluate
                metrics = self.train_and_evaluate(
                    model=model,
                    loss_fn=loss_fn,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    use_tanh=config.get('use_tanh', True),
                    verbose=False
                )
                
                seed_results.append(metrics)
                print(f"  ROI: {metrics['roi']:.4f}, Sharpe: {metrics['sharpe']:.4f}, MDD: {metrics['mdd']:.4f}")
            
            # Aggregate results
            all_results[model_name] = {
                'roi_mean': np.mean([r['roi'] for r in seed_results]),
                'roi_std': np.std([r['roi'] for r in seed_results]),
                'sharpe_mean': np.mean([r['sharpe'] for r in seed_results]),
                'sharpe_std': np.std([r['sharpe'] for r in seed_results]),
                'mdd_mean': np.mean([r['mdd'] for r in seed_results]),
                'mdd_std': np.std([r['mdd'] for r in seed_results]),
            }
        
        return all_results


def run_experiment_1():
    """Experiment 1: Main comparison of WaveLSFormer against baselines."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Performance Comparison of WaveLSFormer Against Baselines")
    print("="*80)
    
    experiment = SimplifiedExperiment(
        experiment_id='exp_1',
        n_features=5,
        window_size=96,
        batch_size=256,
        learning_rate=1e-5,
        n_epochs=20,  # Reduced for demo
        early_stopping_start=10,
        device='cpu'
    )
    
    model_configs = [
        {
            'name': 'MLP',
            'model_type': 'mlp',
            'loss_type': 'composite',
            'model_kwargs': {'hidden_dim': 512, 'n_layers': 10},
            'loss_kwargs': {'use_wavelet': False},
            'use_tanh': True
        },
        {
            'name': 'LSTM',
            'model_type': 'lstm',
            'loss_type': 'composite',
            'model_kwargs': {'hidden_dim': 512, 'n_layers': 2},
            'loss_kwargs': {'use_wavelet': False},
            'use_tanh': True
        },
        {
            'name': 'Transformer',
            'model_type': 'transformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {'use_wavelet': False},
            'use_tanh': True
        },
        {
            'name': 'Wavelet+MLP',
            'model_type': 'wavelet_mlp',
            'loss_type': 'composite',
            'model_kwargs': {'hidden_dim': 512, 'n_layers': 10},
            'loss_kwargs': {'use_wavelet': True, 'lambda_spec': 10.0},
            'use_tanh': True
        },
        {
            'name': 'Wavelet+LSTM',
            'model_type': 'wavelet_lstm',
            'loss_type': 'composite',
            'model_kwargs': {'hidden_dim': 512, 'n_layers': 2},
            'loss_kwargs': {'use_wavelet': True, 'lambda_spec': 10.0},
            'use_tanh': True
        },
        {
            'name': 'WaveLSFormer',
            'model_type': 'wavelsformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {'use_wavelet': True, 'lambda_spec': 10.0},
            'use_tanh': True
        }
    ]
    
    results = experiment.run(model_configs, n_seeds=3)
    experiment.save_results(results, 'exp_1_results.json')
    
    return results


def run_experiment_2():
    """Experiment 2: Loss function ablation."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Impact of Trading-Oriented Loss vs. Regression Loss")
    print("="*80)
    
    experiment = SimplifiedExperiment(
        experiment_id='exp_2',
        n_features=5,
        window_size=96,
        batch_size=256,
        learning_rate=1e-5,
        n_epochs=20,
        early_stopping_start=10,
        device='cpu'
    )
    
    model_configs = [
        {
            'name': 'Soft-Label',
            'model_type': 'wavelsformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {
                'use_soft_label': True,
                'use_sharpe': False,
                'use_penalty': False,
                'use_wavelet': True,
                'lambda_spec': 10.0
            },
            'use_tanh': True
        },
        {
            'name': 'MSE',
            'model_type': 'wavelsformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {
                'use_soft_label': False,
                'use_sharpe': False,
                'use_penalty': False,
                'use_wavelet': True,
                'lambda_spec': 10.0
            },
            'use_tanh': False
        }
    ]
    
    results = experiment.run(model_configs, n_seeds=3)
    experiment.save_results(results, 'exp_2_results.json')
    
    return results


def run_experiment_3():
    """Experiment 3: Wavelet frontend ablation."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Efficacy of Learnable vs. Classic vs. No Wavelet")
    print("="*80)
    
    experiment = SimplifiedExperiment(
        experiment_id='exp_3',
        n_features=5,
        window_size=96,
        batch_size=256,
        learning_rate=1e-5,
        n_epochs=20,
        early_stopping_start=10,
        device='cpu'
    )
    
    model_configs = [
        {
            'name': 'Plain Transformer',
            'model_type': 'transformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {'use_wavelet': False},
            'use_tanh': True
        },
        {
            'name': 'Classic Wavelet Transformer',
            'model_type': 'classic_wavelet_transformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6, 'wavelet': 'db4', 'level': 3},
            'loss_kwargs': {'use_wavelet': False},
            'use_tanh': True
        },
        {
            'name': 'Learnable Wavelet (WaveLSFormer)',
            'model_type': 'wavelsformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {'use_wavelet': True, 'lambda_spec': 10.0},
            'use_tanh': True
        }
    ]
    
    results = experiment.run(model_configs, n_seeds=3)
    experiment.save_results(results, 'exp_3_results.json')
    
    return results


def run_experiment_4():
    """Experiment 4: Fusion method ablation."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Comparison of LGHI Fusion Against Naive Concatenation")
    print("="*80)
    
    experiment = SimplifiedExperiment(
        experiment_id='exp_4',
        n_features=5,
        window_size=96,
        batch_size=256,
        learning_rate=1e-5,
        n_epochs=20,
        early_stopping_start=10,
        device='cpu'
    )
    
    model_configs = [
        {
            'name': 'LGHI Fusion',
            'model_type': 'wavelsformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {'use_wavelet': True, 'lambda_spec': 10.0},
            'use_tanh': True
        },
        {
            'name': 'Concat Fusion',
            'model_type': 'wavelsformer_concat',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {'use_wavelet': True, 'lambda_spec': 10.0},
            'use_tanh': True
        },
        {
            'name': 'Low-Only',
            'model_type': 'wavelsformer_low',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {'use_wavelet': True, 'lambda_spec': 10.0},
            'use_tanh': True
        },
        {
            'name': 'High-Only',
            'model_type': 'wavelsformer_high',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {'use_wavelet': True, 'lambda_spec': 10.0},
            'use_tanh': True
        }
    ]
    
    results = experiment.run(model_configs, n_seeds=3)
    experiment.save_results(results, 'exp_4_results.json')
    
    return results


def run_experiment_5():
    """Experiment 5: Sharpe regularizer ablation."""
    print("\n" + "="*80)
    print("EXPERIMENT 5: Impact of the Differentiable Sharpe Regularizer")
    print("="*80)
    
    experiment = SimplifiedExperiment(
        experiment_id='exp_5',
        n_features=5,
        window_size=96,
        batch_size=256,
        learning_rate=1e-5,
        n_epochs=20,
        early_stopping_start=10,
        device='cpu'
    )
    
    model_configs = [
        {
            'name': 'Without Sharpe Regularizer',
            'model_type': 'wavelsformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {
                'use_sharpe': False,
                'use_wavelet': True,
                'lambda_spec': 10.0
            },
            'use_tanh': True
        },
        {
            'name': 'With Sharpe Regularizer',
            'model_type': 'wavelsformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 512, 'd_ff': 1024, 'n_heads': 8, 'n_layers': 6},
            'loss_kwargs': {
                'use_sharpe': True,
                'use_wavelet': True,
                'lambda_spec': 10.0,
                'alpha_sharpe': 1.0
            },
            'use_tanh': True
        }
    ]
    
    results = experiment.run(model_configs, n_seeds=3)
    experiment.save_results(results, 'exp_5_results.json')
    
    return results


def create_visualizations(all_results: Dict):
    """Create visualizations for all experiments."""
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    sns.set_style("whitegrid")
    
    # Experiment 1: Main comparison
    if 'exp_1' in all_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = list(all_results['exp_1'].keys())
        roi_means = [all_results['exp_1'][m]['roi_mean'] for m in models]
        roi_stds = [all_results['exp_1'][m]['roi_std'] for m in models]
        sharpe_means = [all_results['exp_1'][m]['sharpe_mean'] for m in models]
        sharpe_stds = [all_results['exp_1'][m]['sharpe_std'] for m in models]
        mdd_means = [all_results['exp_1'][m]['mdd_mean'] for m in models]
        mdd_stds = [all_results['exp_1'][m]['mdd_std'] for m in models]
        
        axes[0].bar(range(len(models)), roi_means, yerr=roi_stds, capsize=5)
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylabel('ROI')
        axes[0].set_title('Return on Investment')
        
        axes[1].bar(range(len(models)), sharpe_means, yerr=sharpe_stds, capsize=5)
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].set_title('Sharpe Ratio')
        
        axes[2].bar(range(len(models)), mdd_means, yerr=mdd_stds, capsize=5)
        axes[2].set_xticks(range(len(models)))
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].set_ylabel('Max Drawdown')
        axes[2].set_title('Maximum Drawdown')
        
        plt.tight_layout()
        plt.savefig('results/exp_1_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: results/exp_1_comparison.png")
    
    # Similar visualizations for other experiments
    for exp_id in ['exp_2', 'exp_3', 'exp_4', 'exp_5']:
        if exp_id in all_results:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            models = list(all_results[exp_id].keys())
            roi_means = [all_results[exp_id][m]['roi_mean'] for m in models]
            roi_stds = [all_results[exp_id][m]['roi_std'] for m in models]
            sharpe_means = [all_results[exp_id][m]['sharpe_mean'] for m in models]
            sharpe_stds = [all_results[exp_id][m]['sharpe_std'] for m in models]
            
            axes[0].bar(range(len(models)), roi_means, yerr=roi_stds, capsize=5)
            axes[0].set_xticks(range(len(models)))
            axes[0].set_xticklabels(models, rotation=45, ha='right')
            axes[0].set_ylabel('ROI')
            axes[0].set_title('Return on Investment')
            
            axes[1].bar(range(len(models)), sharpe_means, yerr=sharpe_stds, capsize=5)
            axes[1].set_xticks(range(len(models)))
            axes[1].set_xticklabels(models, rotation=45, ha='right')
            axes[1].set_ylabel('Sharpe Ratio')
            axes[1].set_title('Sharpe Ratio')
            
            plt.tight_layout()
            plt.savefig(f'results/{exp_id}_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: results/{exp_id}_comparison.png")


def save_results_markdown(all_results: Dict):
    """Save results in markdown format."""
    print("\n" + "="*80)
    print("Saving Results to Markdown")
    print("="*80)
    
    with open('results/RESULTS.md', 'w') as f:
        f.write("# WaveLSFormer Experiment Results\n\n")
        f.write("This document contains the results of all experiments comparing WaveLSFormer against baseline architectures.\n\n")
        f.write("**Note:** These results are generated using synthetic data for demonstration purposes.\n\n")
        
        for exp_id, exp_results in all_results.items():
            f.write(f"## {exp_id.upper()}\n\n")
            
            # Create table
            f.write("| Model | ROI (mean ± std) | Sharpe (mean ± std) | MDD (mean ± std) |\n")
            f.write("|-------|------------------|---------------------|------------------|\n")
            
            for model_name, metrics in exp_results.items():
                roi_str = f"{metrics['roi_mean']:.4f} ± {metrics['roi_std']:.4f}"
                sharpe_str = f"{metrics['sharpe_mean']:.4f} ± {metrics['sharpe_std']:.4f}"
                mdd_str = f"{metrics['mdd_mean']:.4f} ± {metrics['mdd_std']:.4f}"
                f.write(f"| {model_name} | {roi_str} | {sharpe_str} | {mdd_str} |\n")
            
            f.write("\n")
            
            # Add visualization reference
            if os.path.exists(f'results/{exp_id}_comparison.png'):
                f.write(f"![{exp_id} Results]({exp_id}_comparison.png)\n\n")
        
        f.write("## Summary\n\n")
        f.write("These experiments demonstrate the implementation of the WaveLSFormer methodology:\n\n")
        f.write("1. **Experiment 1**: Compares WaveLSFormer against MLP, LSTM, and Transformer baselines\n")
        f.write("2. **Experiment 2**: Evaluates the impact of soft-label loss vs. regression losses\n")
        f.write("3. **Experiment 3**: Tests learnable wavelet vs. classic wavelet vs. no wavelet\n")
        f.write("4. **Experiment 4**: Compares LGHI fusion against concatenation and single-frequency models\n")
        f.write("5. **Experiment 5**: Assesses the impact of the Sharpe regularizer\n\n")
        f.write("All experiments follow the paper's methodology with proper train/val/test splits, ")
        f.write("validation-based model selection, and risk-budget scaling.\n")
    
    print("Saved: results/RESULTS.md")


def main():
    """Run all experiments."""
    print("\n" + "="*80)
    print("WaveLSFormer Experiments")
    print("="*80)
    print("\nNote: Using synthetic data for demonstration.")
    print("In production, replace with real equity data from Massive API.\n")
    
    all_results = {}
    
    # Run experiments
    all_results['exp_1'] = run_experiment_1()
    all_results['exp_2'] = run_experiment_2()
    all_results['exp_3'] = run_experiment_3()
    all_results['exp_4'] = run_experiment_4()
    all_results['exp_5'] = run_experiment_5()
    
    # Create visualizations
    create_visualizations(all_results)
    
    # Save results
    save_results_markdown(all_results)
    
    print("\n" + "="*80)
    print("All Experiments Completed!")
    print("="*80)
    print("\nResults saved to:")
    print("  - results/RESULTS.md")
    print("  - results/exp_*_results.json")
    print("  - results/exp_*_comparison.png")


if __name__ == '__main__':
    main()

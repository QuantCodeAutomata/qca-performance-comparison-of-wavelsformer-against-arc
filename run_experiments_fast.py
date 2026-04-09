"""Fast version of experiments with reduced epochs for demonstration."""

import sys
sys.path.insert(0, '/workspace/project')

from run_experiments import *

# Override experiment parameters for faster execution
def run_experiment_1_fast():
    """Experiment 1: Main comparison (fast version)."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Performance Comparison (Fast Version)")
    print("="*80)
    
    experiment = SimplifiedExperiment(
        experiment_id='exp_1',
        n_features=5,
        window_size=96,
        batch_size=256,
        learning_rate=1e-5,
        n_epochs=5,  # Reduced
        early_stopping_start=2,  # Reduced
        device='cpu'
    )
    
    # Test only key models
    model_configs = [
        {
            'name': 'MLP',
            'model_type': 'mlp',
            'loss_type': 'composite',
            'model_kwargs': {'hidden_dim': 256, 'n_layers': 5},  # Reduced
            'loss_kwargs': {'use_wavelet': False},
            'use_tanh': True
        },
        {
            'name': 'Transformer',
            'model_type': 'transformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 256, 'd_ff': 512, 'n_heads': 4, 'n_layers': 2},  # Reduced
            'loss_kwargs': {'use_wavelet': False},
            'use_tanh': True
        },
        {
            'name': 'WaveLSFormer',
            'model_type': 'wavelsformer',
            'loss_type': 'composite',
            'model_kwargs': {'d_model': 256, 'd_ff': 512, 'n_heads': 4, 'n_layers': 2},  # Reduced
            'loss_kwargs': {'use_wavelet': True, 'lambda_spec': 10.0},
            'use_tanh': True
        }
    ]
    
    results = experiment.run(model_configs, n_seeds=2)  # Reduced seeds
    experiment.save_results(results, 'exp_1_results.json')
    
    return results


def main_fast():
    """Run fast version of experiments."""
    print("\n" + "="*80)
    print("WaveLSFormer Experiments (Fast Version)")
    print("="*80)
    print("\nNote: Using reduced epochs and model sizes for demonstration.\n")
    
    all_results = {}
    
    # Run only experiment 1 for demonstration
    all_results['exp_1'] = run_experiment_1_fast()
    
    # Create visualizations
    create_visualizations(all_results)
    
    # Save results
    save_results_markdown(all_results)
    
    print("\n" + "="*80)
    print("Experiments Completed!")
    print("="*80)
    print("\nResults saved to:")
    print("  - results/RESULTS.md")
    print("  - results/exp_1_results.json")
    print("  - results/exp_1_comparison.png")


if __name__ == '__main__':
    main_fast()

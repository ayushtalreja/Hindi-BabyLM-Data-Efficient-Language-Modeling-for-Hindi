"""
Example Usage Scripts for Enhanced Evaluation Pipeline

This file demonstrates how to use the new evaluation features:
1. Metrics with confidence intervals
2. Confusion matrices and visualizations
3. Evaluation caching
4. Comparative analysis
5. Training integration with evaluation callbacks
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.evaluation import (
    IndicGLUEEvaluator,
    MetricsAggregator,
    EvaluationCache,
    ComparativeAnalyzer,
    compare_results
)


# ==============================================================================
# Example 1: Compute Metrics with Confidence Intervals
# ==============================================================================
def example_metrics_with_ci():
    """Example: Computing metrics with bootstrap confidence intervals"""
    print("\n" + "="*80)
    print("Example 1: Metrics with Confidence Intervals")
    print("="*80)

    # Sample predictions and labels
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2] * 50  # 450 samples
    y_pred = [0, 1, 2, 0, 2, 1, 0, 1, 2] * 50  # Some errors

    # Create aggregator
    aggregator = MetricsAggregator(
        bootstrap_samples=1000,
        confidence_level=0.95
    )

    # Compute accuracy with CI
    accuracy = aggregator.compute_metric(
        y_true, y_pred, 'accuracy', compute_ci=True
    )
    print(f"\n{accuracy}")

    # Compute F1 with CI
    f1_macro = aggregator.compute_metric(
        y_true, y_pred, 'f1', average='macro', compute_ci=True
    )
    print(f"{f1_macro}")

    # Compute per-class metrics
    print("\nPer-class metrics:")
    per_class = aggregator.compute_per_class_metrics(
        y_true, y_pred,
        class_names=['Class A', 'Class B', 'Class C'],
        compute_ci=True
    )

    for class_idx, metrics in per_class.items():
        print(f"\n  Class {class_idx}:")
        for metric_name, metric in metrics.items():
            if metric_name != 'support':
                print(f"    {metric}")


# ==============================================================================
# Example 2: IndicGLUE Evaluation with Visualizations
# ==============================================================================
def example_indicglue_with_viz():
    """Example: Running IndicGLUE evaluation with confusion matrices"""
    print("\n" + "="*80)
    print("Example 2: IndicGLUE Evaluation with Visualizations")
    print("="*80)

    # Note: This requires a trained model
    print("\nTo use this feature:")
    print("1. Load your trained model and tokenizer")
    print("2. Create evaluator with visualization config:")
    print("""
    config = {
        'evaluation': {
            'save_visualizations': True,
            'visualization_format': ['png', 'html'],
            'bootstrap_samples': 1000,
            'confidence_level': 0.95
        }
    }

    evaluator = IndicGLUEEvaluator(model, tokenizer, config)
    results = evaluator.evaluate_all_tasks()

    # Generate visualizations
    evaluator.save_visualizations(results, 'results/visualizations')
    """)


# ==============================================================================
# Example 3: Evaluation Caching
# ==============================================================================
def example_evaluation_caching():
    """Example: Using evaluation cache to speed up repeated evaluations"""
    print("\n" + "="*80)
    print("Example 3: Evaluation Caching")
    print("="*80)

    # Create cache manager
    cache = EvaluationCache(
        cache_dir='.eval_cache',
        max_cache_age_days=30,
        enable_cache=True
    )

    # Compute cache key
    cache_key = cache._compute_cache_key(
        model_path='checkpoints/checkpoint_best.pt',
        dataset_name='IndicGLUE',
        dataset_split='test',
        config={'batch_size': 32}
    )

    print(f"\nCache key: {cache_key}")

    # Check for cached predictions
    cached = cache.get_cached_predictions(cache_key)
    if cached:
        print("Cache hit! Loading predictions from cache")
        predictions = cached['predictions']
    else:
        print("Cache miss! Running evaluation...")
        # Simulate predictions
        predictions = {'task1': [0, 1, 2], 'task2': [1, 0, 1]}

        # Save to cache
        cache.save_predictions(
            cache_key,
            predictions,
            metadata={'model': 'gpt-small', 'dataset': 'IndicGLUE'}
        )

    # Get cache stats
    stats = cache.get_cache_stats()
    print(f"\nCache statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


# ==============================================================================
# Example 4: Comparative Analysis
# ==============================================================================
def example_comparative_analysis():
    """Example: Comparing multiple models/checkpoints"""
    print("\n" + "="*80)
    print("Example 4: Comparative Analysis")
    print("="*80)

    print("\nTo compare multiple models:")
    print("""
    # Define paths to evaluation results
    result_paths = {
        'gpt-small': 'results/gpt_small_eval/evaluation_results.json',
        'gpt-medium': 'results/gpt_medium_eval/evaluation_results.json',
        'deberta-small': 'results/deberta_small_eval/evaluation_results.json'
    }

    # Run comparison
    analyzer = compare_results(
        result_paths,
        output_dir='comparative_analysis',
        generate_html=True,
        generate_pdf=True
    )

    # Create comparison table
    comparison_df = analyzer.create_comparison_table(
        metrics=['accuracy', 'f1_macro']
    )
    print(comparison_df)

    # Create radar plot
    analyzer.create_radar_plot(
        metric='accuracy',
        save_path='comparative_analysis/radar_plot.html'
    )
    """)


# ==============================================================================
# Example 5: Training Integration
# ==============================================================================
def example_training_integration():
    """Example: Integrating evaluation into training"""
    print("\n" + "="*80)
    print("Example 5: Training Integration with Evaluation Callbacks")
    print("="*80)

    print("\nUpdate your training config:")
    print("""
    training:
      # Enable evaluation callback
      enable_eval_callback: true
      eval_frequency: 1  # Evaluate every epoch
      log_eval_to_wandb: true

      # Evaluation-based early stopping
      eval_early_stopping: true
      eval_early_stopping_metric: "overall.average_accuracy"
      eval_early_stopping_patience: 3
      eval_early_stopping_mode: "max"

      # Best checkpoint selection
      checkpoint_metric: "overall.average_accuracy"
      load_best_checkpoint_at_end: true
    """)

    print("\nThen train normally:")
    print("""
    from src.training import HindiLanguageModelTrainer

    trainer = HindiLanguageModelTrainer(model, tokenizer, config)
    trainer.train(train_dataloader, val_dataloader)

    # Evaluation will run automatically at the end of each epoch
    # Best checkpoint will be loaded at the end of training
    """)


# ==============================================================================
# Example 6: Statistical Significance Testing
# ==============================================================================
def example_statistical_significance():
    """Example: Testing statistical significance between models"""
    print("\n" + "="*80)
    print("Example 6: Statistical Significance Testing")
    print("="*80)

    import numpy as np

    # Sample predictions from two models
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 50)
    model1_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0] * 50)
    model2_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1] * 50)

    # Create aggregator
    aggregator = MetricsAggregator()

    # Test significance
    result = aggregator.compute_statistical_significance(
        y_true, model1_pred, model2_pred,
        metric_name='accuracy',
        test='mcnemar'
    )

    print("\nMcNemar's Test Results:")
    print(f"  Test statistic: {result['statistic']:.4f}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Significant: {result['significant']}")
    print(f"\nContingency table:")
    for key, value in result['contingency_table'].items():
        print(f"  {key}: {value}")


# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "#"*80)
    print("# Hindi BabyLM Enhanced Evaluation Pipeline - Examples")
    print("#"*80)

    # Run examples
    example_metrics_with_ci()
    example_indicglue_with_viz()
    example_evaluation_caching()
    example_comparative_analysis()
    example_training_integration()
    example_statistical_significance()

    print("\n" + "#"*80)
    print("# Examples completed!")
    print("#"*80 + "\n")

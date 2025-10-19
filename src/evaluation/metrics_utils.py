"""
Metrics Utilities for Hindi BabyLM Evaluation

This module provides standardized metrics computation with statistical rigor,
including bootstrap confidence intervals, per-class metrics, and aggregation methods.

Key Features:
- Standardized Metric dataclass with confidence intervals
- Bootstrap confidence interval computation (1000 samples, 95% CI)
- Multiple aggregation strategies (macro, micro, weighted)
- Per-class metrics with statistical confidence
- Comprehensive error handling and validation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """
    Standardized metric representation with confidence intervals

    Attributes:
        name: Metric name (e.g., 'accuracy', 'f1_macro')
        value: Primary metric value
        mean: Bootstrap mean (typically same as value)
        std: Standard deviation from bootstrap
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        n_samples: Number of samples used in computation
        metadata: Additional metric-specific information
    """
    name: str
    value: float
    mean: Optional[float] = None
    std: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    n_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set mean to value if not provided"""
        if self.mean is None:
            self.mean = self.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation"""
        return {
            'name': self.name,
            'value': self.value,
            'mean': self.mean,
            'std': self.std,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'n_samples': self.n_samples,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """Human-readable string representation"""
        if self.ci_lower is not None and self.ci_upper is not None:
            return f"{self.name}: {self.value:.4f} (95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        else:
            return f"{self.name}: {self.value:.4f}"


@dataclass
class AggregatedMetrics:
    """
    Collection of aggregated metrics

    Attributes:
        metrics: Dictionary mapping metric names to Metric objects
        aggregation_type: Type of aggregation used ('macro', 'micro', 'weighted')
        weights: Optional weights used for weighted aggregation
    """
    metrics: Dict[str, Metric]
    aggregation_type: str = 'macro'
    weights: Optional[Dict[str, float]] = None

    def get_metric(self, name: str) -> Optional[Metric]:
        """Retrieve a specific metric by name"""
        return self.metrics.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'metrics': {k: v.to_dict() for k, v in self.metrics.items()},
            'aggregation_type': self.aggregation_type,
            'weights': self.weights
        }

    def get_summary(self) -> str:
        """Get human-readable summary of all metrics"""
        lines = [f"Aggregation Type: {self.aggregation_type}"]
        lines.append("-" * 60)
        for metric in self.metrics.values():
            lines.append(str(metric))
        return "\n".join(lines)


class MetricsAggregator:
    """
    Advanced metrics aggregator with bootstrap confidence intervals

    This class provides comprehensive metric computation with statistical rigor,
    including bootstrap resampling for confidence intervals and multiple
    aggregation strategies.

    Features:
    - Bootstrap confidence intervals (configurable samples and confidence level)
    - Per-class metrics computation
    - Multiple aggregation methods (macro, micro, weighted)
    - Proper handling of edge cases (zero division, empty classes)
    """

    def __init__(
        self,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ):
        """
        Initialize metrics aggregator

        Args:
            bootstrap_samples: Number of bootstrap samples for CI computation
            confidence_level: Confidence level for intervals (default: 0.95 for 95% CI)
            random_seed: Random seed for reproducibility
        """
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        logger.debug(
            f"MetricsAggregator initialized: "
            f"bootstrap_samples={bootstrap_samples}, "
            f"confidence_level={confidence_level}"
        )

    def compute_metric(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        metric_name: str,
        average: Optional[str] = None,
        compute_ci: bool = True
    ) -> Metric:
        """
        Compute a single metric with optional bootstrap confidence interval

        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_name: Name of metric ('accuracy', 'f1', 'precision', 'recall')
            average: Averaging strategy for multi-class ('macro', 'micro', 'weighted', None)
            compute_ci: Whether to compute bootstrap confidence intervals

        Returns:
            Metric object with value and optional confidence interval
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) == 0:
            logger.warning(f"Empty input for metric {metric_name}")
            return Metric(
                name=f"{metric_name}_{average}" if average else metric_name,
                value=0.0,
                n_samples=0
            )

        # Compute primary metric value
        value = self._compute_metric_value(y_true, y_pred, metric_name, average)

        # Compute bootstrap confidence interval if requested
        if compute_ci and len(y_true) >= 10:  # Need minimum samples for bootstrap
            mean, std, ci_lower, ci_upper = self.bootstrap_ci(
                y_true, y_pred, metric_name, average
            )
        else:
            mean, std, ci_lower, ci_upper = value, None, None, None

        # Create metric name with averaging strategy
        full_name = f"{metric_name}_{average}" if average else metric_name

        return Metric(
            name=full_name,
            value=value,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=len(y_true),
            metadata={'average': average} if average else {}
        )

    def _compute_metric_value(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_name: str,
        average: Optional[str] = None
    ) -> float:
        """
        Compute the actual metric value

        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_name: Metric to compute
            average: Averaging strategy

        Returns:
            Computed metric value
        """
        metric_name_lower = metric_name.lower()

        try:
            if metric_name_lower == 'accuracy':
                return accuracy_score(y_true, y_pred)
            elif metric_name_lower == 'f1':
                return f1_score(y_true, y_pred, average=average, zero_division=0)
            elif metric_name_lower == 'precision':
                return precision_score(y_true, y_pred, average=average, zero_division=0)
            elif metric_name_lower == 'recall':
                return recall_score(y_true, y_pred, average=average, zero_division=0)
            else:
                logger.warning(f"Unknown metric: {metric_name}, defaulting to accuracy")
                return accuracy_score(y_true, y_pred)
        except Exception as e:
            logger.error(f"Error computing {metric_name}: {e}")
            return 0.0

    def bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_name: str,
        average: Optional[str] = None
    ) -> Tuple[float, float, float, float]:
        """
        Compute bootstrap confidence interval for a metric

        This method uses bootstrap resampling to estimate the sampling distribution
        of the metric and compute confidence intervals.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_name: Metric to compute
            average: Averaging strategy

        Returns:
            Tuple of (mean, std, ci_lower, ci_upper)
        """
        n = len(y_true)
        bootstrap_scores = []

        # Perform bootstrap resampling
        for _ in range(self.bootstrap_samples):
            # Sample with replacement
            indices = self.rng.choice(n, size=n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Compute metric on bootstrap sample
            score = self._compute_metric_value(y_true_boot, y_pred_boot, metric_name, average)
            bootstrap_scores.append(score)

        bootstrap_scores = np.array(bootstrap_scores)

        # Compute statistics
        mean = np.mean(bootstrap_scores)
        std = np.std(bootstrap_scores, ddof=1)

        # Compute confidence interval using percentile method
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

        return mean, std, ci_lower, ci_upper

    def compute_per_class_metrics(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        class_names: Optional[List[str]] = None,
        compute_ci: bool = True
    ) -> Dict[int, Dict[str, Metric]]:
        """
        Compute per-class precision, recall, and F1 with confidence intervals

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names for labeling
            compute_ci: Whether to compute confidence intervals

        Returns:
            Dictionary mapping class index to dict of metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))

        per_class_metrics = {}

        for class_idx in classes:
            # Create binary labels for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)

            # Class name
            class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f"class_{class_idx}"

            # Compute metrics for this class
            metrics = {}

            # Precision
            metrics['precision'] = self.compute_metric(
                y_true_binary, y_pred_binary, 'precision', average='binary', compute_ci=compute_ci
            )
            metrics['precision'].name = f"{class_name}_precision"

            # Recall
            metrics['recall'] = self.compute_metric(
                y_true_binary, y_pred_binary, 'recall', average='binary', compute_ci=compute_ci
            )
            metrics['recall'].name = f"{class_name}_recall"

            # F1
            metrics['f1'] = self.compute_metric(
                y_true_binary, y_pred_binary, 'f1', average='binary', compute_ci=compute_ci
            )
            metrics['f1'].name = f"{class_name}_f1"

            # Support (number of true instances)
            support = np.sum(y_true == class_idx)
            metrics['support'] = Metric(
                name=f"{class_name}_support",
                value=float(support),
                n_samples=len(y_true),
                metadata={'class': class_name}
            )

            per_class_metrics[int(class_idx)] = metrics

        return per_class_metrics

    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]],
        aggregation_type: str = 'macro',
        weights: Optional[List[float]] = None
    ) -> AggregatedMetrics:
        """
        Aggregate multiple metric dictionaries

        Args:
            metrics_list: List of metric dictionaries
            aggregation_type: Type of aggregation ('macro', 'micro', 'weighted')
            weights: Optional weights for weighted aggregation

        Returns:
            AggregatedMetrics object
        """
        if not metrics_list:
            logger.warning("Empty metrics list provided for aggregation")
            return AggregatedMetrics(metrics={}, aggregation_type=aggregation_type)

        # Get all metric names
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())

        aggregated = {}

        for metric_name in metric_names:
            # Collect values for this metric
            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_list
                if metric_name in metrics
            ]

            if not values:
                continue

            # Aggregate based on type
            if aggregation_type == 'macro':
                agg_value = np.mean(values)
            elif aggregation_type == 'micro':
                # For micro, we'd need raw counts, so fall back to macro
                logger.warning(f"Micro aggregation not fully supported, using macro for {metric_name}")
                agg_value = np.mean(values)
            elif aggregation_type == 'weighted':
                if weights and len(weights) == len(values):
                    agg_value = np.average(values, weights=weights)
                else:
                    logger.warning(f"Invalid weights for {metric_name}, using macro")
                    agg_value = np.mean(values)
            else:
                logger.warning(f"Unknown aggregation type: {aggregation_type}, using macro")
                agg_value = np.mean(values)

            # Create metric object
            aggregated[metric_name] = Metric(
                name=f"{metric_name}_{aggregation_type}",
                value=agg_value,
                n_samples=len(values),
                metadata={'aggregation_type': aggregation_type}
            )

        return AggregatedMetrics(
            metrics=aggregated,
            aggregation_type=aggregation_type,
            weights=dict(zip(metric_names, weights)) if weights else None
        )

    def compute_confusion_matrix(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        normalize: Optional[str] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', or None)

        Returns:
            Tuple of (confusion_matrix, labels)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Get all labels
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

        return cm, labels

    def compute_statistical_significance(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        metric_name: str = 'accuracy',
        test: str = 'mcnemar'
    ) -> Dict[str, Any]:
        """
        Test statistical significance between two models' predictions

        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            metric_name: Metric to compare
            test: Statistical test to use ('mcnemar' for classification)

        Returns:
            Dictionary with test results
        """
        if test == 'mcnemar':
            # McNemar's test for paired nominal data
            # Create contingency table
            correct1 = (y_pred1 == y_true)
            correct2 = (y_pred2 == y_true)

            # 2x2 table: both correct, 1 correct 2 wrong, 1 wrong 2 correct, both wrong
            both_correct = np.sum(correct1 & correct2)
            only1_correct = np.sum(correct1 & ~correct2)
            only2_correct = np.sum(~correct1 & correct2)
            both_wrong = np.sum(~correct1 & ~correct2)

            # McNemar's test (using continuity correction)
            if only1_correct + only2_correct > 0:
                statistic = (abs(only1_correct - only2_correct) - 1) ** 2 / (only1_correct + only2_correct)
                p_value = 1 - stats.chi2.cdf(statistic, df=1)
            else:
                statistic = 0.0
                p_value = 1.0

            return {
                'test': 'mcnemar',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'contingency_table': {
                    'both_correct': int(both_correct),
                    'only_model1_correct': int(only1_correct),
                    'only_model2_correct': int(only2_correct),
                    'both_wrong': int(both_wrong)
                }
            }
        else:
            logger.warning(f"Unknown test: {test}")
            return {'error': f'Unknown test: {test}'}


# Convenience functions for backward compatibility
def compute_metrics_with_ci(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    metrics: List[str] = ['accuracy', 'f1'],
    average: str = 'macro',
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Metric]:
    """
    Convenience function to compute multiple metrics with confidence intervals

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metrics: List of metric names to compute
        average: Averaging strategy
        bootstrap_samples: Number of bootstrap samples
        confidence_level: Confidence level for CIs

    Returns:
        Dictionary mapping metric names to Metric objects
    """
    aggregator = MetricsAggregator(
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level
    )

    results = {}
    for metric_name in metrics:
        results[metric_name] = aggregator.compute_metric(
            y_true, y_pred, metric_name, average=average
        )

    return results

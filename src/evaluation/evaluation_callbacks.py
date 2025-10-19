"""
Evaluation Callbacks for Training Integration

This module provides callback classes for integrating comprehensive evaluation
into the training loop, including periodic evaluation, evaluation-based early
stopping, and checkpoint selection based on evaluation metrics.

Key Features:
- Periodic evaluation during training (epoch or step-based)
- Hierarchical WandB logging of evaluation metrics
- Evaluation-based early stopping (beyond just validation loss)
- Checkpoint selection based on evaluation metrics
- Automatic best checkpoint loading
"""

import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class EvaluationCallback:
    """
    Callback for running periodic evaluation during training

    This callback runs comprehensive evaluation at specified intervals and
    logs results to WandB and local files.

    Features:
    - Epoch or step-based evaluation
    - Hierarchical WandB logging (eval/indicglue/task_name/metric)
    - Save evaluation results per epoch/step
    - Support for multiple evaluation functions
    """

    def __init__(
        self,
        evaluation_fn: Callable,
        frequency: int = 1,
        frequency_type: str = 'epoch',  # 'epoch' or 'steps'
        eval_on_steps: Optional[List[int]] = None,
        save_results: bool = True,
        results_dir: str = 'evaluation_results',
        log_to_wandb: bool = True,
        wandb_prefix: str = 'eval'
    ):
        """
        Initialize evaluation callback

        Args:
            evaluation_fn: Function to call for evaluation, should return Dict[str, Any]
            frequency: How often to evaluate (every N epochs or steps)
            frequency_type: Type of frequency ('epoch' or 'steps')
            eval_on_steps: Specific steps to evaluate on (overrides frequency if provided)
            save_results: Whether to save results to files
            results_dir: Directory to save evaluation results
            log_to_wandb: Whether to log metrics to WandB
            wandb_prefix: Prefix for WandB metric names
        """
        self.evaluation_fn = evaluation_fn
        self.frequency = frequency
        self.frequency_type = frequency_type
        self.eval_on_steps = set(eval_on_steps) if eval_on_steps else None
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        self.log_to_wandb = log_to_wandb
        self.wandb_prefix = wandb_prefix

        # Create results directory
        if self.save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        # Track evaluation history
        self.eval_history = []

        logger.info(f"EvaluationCallback initialized:")
        logger.info(f"  Frequency: Every {frequency} {frequency_type}")
        logger.info(f"  Save results: {save_results}")
        logger.info(f"  WandB logging: {log_to_wandb}")

    def should_evaluate(self, epoch: int, step: int) -> bool:
        """
        Determine if evaluation should run

        Args:
            epoch: Current epoch number
            step: Current global step

        Returns:
            True if evaluation should run
        """
        # Check specific steps list
        if self.eval_on_steps is not None:
            return step in self.eval_on_steps

        # Check frequency-based
        if self.frequency_type == 'epoch':
            return epoch % self.frequency == 0
        elif self.frequency_type == 'steps':
            return step % self.frequency == 0
        else:
            return False

    def on_epoch_end(
        self,
        epoch: int,
        step: int,
        model: torch.nn.Module,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Called at the end of each epoch

        Args:
            epoch: Current epoch number (0-indexed)
            step: Current global step
            model: Model being trained
            **kwargs: Additional arguments

        Returns:
            Evaluation results dictionary if evaluation was run, None otherwise
        """
        if not self.should_evaluate(epoch, step):
            return None

        logger.info(f"\n{'='*60}")
        logger.info(f"Running evaluation at epoch {epoch+1}, step {step}")
        logger.info(f"{'='*60}")

        # Run evaluation
        try:
            eval_results = self.evaluation_fn(model)

            # Add metadata
            eval_results['_metadata'] = {
                'epoch': epoch,
                'step': step,
                'timestamp': datetime.now().isoformat()
            }

            # Log to WandB
            if self.log_to_wandb:
                self._log_to_wandb(eval_results, epoch, step)

            # Save results
            if self.save_results:
                self._save_results(eval_results, epoch, step)

            # Add to history
            self.eval_history.append({
                'epoch': epoch,
                'step': step,
                'results': eval_results
            })

            logger.info(f"Evaluation complete")
            return eval_results

        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            return None

    def _log_to_wandb(
        self,
        eval_results: Dict[str, Any],
        epoch: int,
        step: int
    ):
        """
        Log evaluation results to WandB with hierarchical structure

        Args:
            eval_results: Evaluation results dictionary
            epoch: Current epoch
            step: Current step
        """
        try:
            import wandb

            # Flatten results for logging
            flattened = self._flatten_dict(eval_results, parent_key=self.wandb_prefix)

            # Add epoch and step
            flattened[f'{self.wandb_prefix}/epoch'] = epoch
            flattened[f'{self.wandb_prefix}/step'] = step

            # Log to wandb
            wandb.log(flattened, step=step)

            logger.debug(f"Logged {len(flattened)} metrics to WandB")

        except ImportError:
            logger.warning("WandB not available for logging")
        except Exception as e:
            logger.error(f"Error logging to WandB: {e}")

    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = '',
        sep: str = '/'
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary with hierarchical keys

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for hierarchical keys

        Returns:
            Flattened dictionary
        """
        items = []

        for k, v in d.items():
            # Skip metadata and complex objects
            if k.startswith('_') or k in ['confusion_matrix', 'per_class_metrics', 'metadata']:
                continue

            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                # Recursively flatten
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, str, bool)):
                # Log primitive types
                items.append((new_key, v))
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                # Log list/tuple as individual items or summary
                if all(isinstance(x, (int, float)) for x in v):
                    # Log mean if numeric
                    import numpy as np
                    items.append((f"{new_key}_mean", float(np.mean(v))))

        return dict(items)

    def _save_results(
        self,
        eval_results: Dict[str, Any],
        epoch: int,
        step: int
    ):
        """
        Save evaluation results to file

        Args:
            eval_results: Evaluation results
            epoch: Current epoch
            step: Current step
        """
        try:
            # Create filename
            filename = f"eval_epoch{epoch:03d}_step{step:06d}.json"
            filepath = self.results_dir / filename

            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(eval_results, f, indent=2, default=str)

            logger.info(f"Saved evaluation results: {filepath}")

        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")


class EvaluationBasedEarlyStopping:
    """
    Early stopping based on evaluation metrics (not just validation loss)

    This class monitors evaluation metrics and stops training when
    no improvement is observed for a specified number of evaluations.

    Features:
    - Monitor any evaluation metric
    - Configurable patience and min_delta
    - Support for both maximize and minimize modes
    - Track best metric value and checkpoint
    """

    def __init__(
        self,
        metric_name: str,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        Initialize early stopping

        Args:
            metric_name: Name of metric to monitor (e.g., 'eval_indicglue_avg_accuracy')
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics that should increase, 'min' for metrics that should decrease
            verbose: Whether to log messages
        """
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        # State tracking
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.evaluations_without_improvement = 0
        self.should_stop = False

        logger.info(f"EvaluationBasedEarlyStopping initialized:")
        logger.info(f"  Metric: {metric_name}")
        logger.info(f"  Patience: {patience}")
        logger.info(f"  Mode: {mode}")

    def __call__(self, eval_results: Dict[str, Any]) -> bool:
        """
        Check if training should stop

        Args:
            eval_results: Evaluation results dictionary

        Returns:
            True if training should stop, False otherwise
        """
        # Extract metric value from nested dict
        current_value = self._extract_metric(eval_results, self.metric_name)

        if current_value is None:
            logger.warning(f"Metric {self.metric_name} not found in evaluation results")
            return False

        # Check for improvement
        improved = self._has_improved(current_value)

        if improved:
            self.best_value = current_value
            self.evaluations_without_improvement = 0

            if self.verbose:
                logger.info(f"Evaluation metric improved: {self.metric_name} = {current_value:.4f}")

        else:
            self.evaluations_without_improvement += 1

            if self.verbose:
                logger.info(
                    f"No improvement in {self.metric_name} for "
                    f"{self.evaluations_without_improvement}/{self.patience} evaluations"
                )

            # Check if should stop
            if self.evaluations_without_improvement >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered! Best {self.metric_name}: {self.best_value:.4f}")
                return True

        return False

    def _has_improved(self, current_value: float) -> bool:
        """
        Check if current value is an improvement

        Args:
            current_value: Current metric value

        Returns:
            True if improved, False otherwise
        """
        if self.mode == 'max':
            return current_value > self.best_value + self.min_delta
        else:
            return current_value < self.best_value - self.min_delta

    def _extract_metric(
        self,
        results: Dict[str, Any],
        metric_path: str
    ) -> Optional[float]:
        """
        Extract metric from nested dictionary

        Args:
            results: Results dictionary
            metric_path: Dot-separated path to metric (e.g., 'overall.accuracy')

        Returns:
            Metric value or None if not found
        """
        # Try simple key first
        if metric_path in results:
            value = results[metric_path]
            if isinstance(value, (int, float)):
                return float(value)

        # Try nested path
        parts = metric_path.split('.')
        current = results

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        if isinstance(current, (int, float)):
            return float(current)

        return None


class CheckpointSelector:
    """
    Select best checkpoint based on evaluation metrics

    This class tracks all checkpoints with their evaluation scores
    and can select the best one based on specified metrics.

    Features:
    - Track checkpoint paths with evaluation metrics
    - Select best checkpoint by any metric
    - Support for multiple metric selection criteria
    - Return best checkpoint path for loading
    """

    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        metric_name: str = 'eval_indicglue_avg_accuracy',
        mode: str = 'max'
    ):
        """
        Initialize checkpoint selector

        Args:
            checkpoint_dir: Directory where checkpoints are saved
            metric_name: Metric to use for selection
            mode: 'max' or 'min'
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric_name = metric_name
        self.mode = mode

        # Track checkpoints
        self.checkpoints = []  # List of (checkpoint_path, metric_value, eval_results)
        self.best_checkpoint = None
        self.best_value = float('-inf') if mode == 'max' else float('inf')

        logger.info(f"CheckpointSelector initialized:")
        logger.info(f"  Metric: {metric_name}")
        logger.info(f"  Mode: {mode}")

    def register_checkpoint(
        self,
        checkpoint_path: str,
        eval_results: Dict[str, Any]
    ):
        """
        Register a checkpoint with its evaluation results

        Args:
            checkpoint_path: Path to checkpoint file
            eval_results: Evaluation results for this checkpoint
        """
        # Extract metric value
        metric_value = self._extract_metric(eval_results, self.metric_name)

        if metric_value is None:
            logger.warning(f"Could not extract metric {self.metric_name} from results")
            return

        # Add to tracking
        self.checkpoints.append((checkpoint_path, metric_value, eval_results))

        # Check if best
        is_best = False
        if self.mode == 'max':
            is_best = metric_value > self.best_value
        else:
            is_best = metric_value < self.best_value

        if is_best:
            self.best_value = metric_value
            self.best_checkpoint = checkpoint_path
            logger.info(f"New best checkpoint: {checkpoint_path} ({self.metric_name}={metric_value:.4f})")

    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get path to best checkpoint

        Returns:
            Path to best checkpoint or None if no checkpoints registered
        """
        return self.best_checkpoint

    def get_checkpoint_ranking(self, top_k: int = 5) -> List[tuple[str, float]]:
        """
        Get top-k checkpoints ranked by metric

        Args:
            top_k: Number of top checkpoints to return

        Returns:
            List of (checkpoint_path, metric_value) tuples
        """
        if not self.checkpoints:
            return []

        # Sort checkpoints
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x[1],
            reverse=(self.mode == 'max')
        )

        # Return top-k
        return [(path, value) for path, value, _ in sorted_checkpoints[:top_k]]

    def _extract_metric(
        self,
        results: Dict[str, Any],
        metric_path: str
    ) -> Optional[float]:
        """Extract metric from nested dictionary (same as EvaluationBasedEarlyStopping)"""
        # Try simple key first
        if metric_path in results:
            value = results[metric_path]
            if isinstance(value, (int, float)):
                return float(value)

        # Try nested path
        parts = metric_path.split('.')
        current = results

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        if isinstance(current, (int, float)):
            return float(current)

        return None


# Convenience function to create standard evaluation callback
def create_evaluation_callback(
    model,
    tokenizer,
    config: Dict[str, Any]
) -> EvaluationCallback:
    """
    Create standard evaluation callback from config

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        config: Configuration dictionary

    Returns:
        Configured EvaluationCallback
    """
    from .evaluation_manager import EvaluationManager

    # Create evaluation function
    def eval_fn(model):
        eval_manager = EvaluationManager(model, tokenizer, config)
        return eval_manager.run_comprehensive_evaluation()

    # Get config
    eval_config = config.get('evaluation', {})
    training_config = config.get('training', {})

    return EvaluationCallback(
        evaluation_fn=eval_fn,
        frequency=training_config.get('eval_frequency', 1),
        frequency_type='epoch',
        eval_on_steps=training_config.get('eval_on_steps'),
        save_results=True,
        results_dir=eval_config.get('results_dir', 'evaluation_results'),
        log_to_wandb=training_config.get('log_eval_to_wandb', True),
        wandb_prefix='eval'
    )

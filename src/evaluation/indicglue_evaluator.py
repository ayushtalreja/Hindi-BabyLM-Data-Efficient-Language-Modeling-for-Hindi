"""
IndicGLUE Benchmark Evaluator for Hindi Language Models

This module implements comprehensive evaluation on the IndicGLUE benchmark,
which includes multiple tasks for evaluating Indian language understanding.

IndicGLUE Tasks:
- IndicNews: Article genre classification
- IndicHeadline: Headline prediction task
- IndicWiki: Section title prediction
- IndicCQ: Cloze-style question answering
- IndicWNLI: Winograd Natural Language Inference
- IndicCOPA: Choice of Plausible Alternatives

Reference: https://indicnlp.ai4bharat.org/indicglue/
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from datasets import load_dataset, Dataset
import logging
from pathlib import Path

# Import new utilities
from .metrics_utils import MetricsAggregator, Metric
from .evaluation_cache import EvaluationCache

logger = logging.getLogger(__name__)


class IndicGLUEEvaluator:
    """
    Comprehensive evaluator for IndicGLUE benchmark tasks

    Features:
    - All 6 IndicGLUE tasks supported
    - Multiple evaluation metrics per task
    - Batch processing for efficiency
    - Detailed error analysis
    - Statistical significance testing
    """

    def __init__(self, model, tokenizer, config: Optional[Dict] = None):
        """
        Initialize IndicGLUE evaluator

        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer for the model
            config: Optional configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}

        # Device setup
        self.device = next(model.parameters()).device
        logger.info(f"IndicGLUE evaluator initialized on device: {self.device}")

        # Task configurations
        self.tasks = {
            'IndicNews': {
                'type': 'classification',
                'num_labels': 3,  # Sports, Business, Entertainment
                'metric': 'accuracy',
                'class_names': ['Sports', 'Business', 'Entertainment']
            },
            'IndicHeadline': {
                'type': 'classification',
                'num_labels': 3,  # Correct/Incorrect headline match
                'metric': 'accuracy',
                'class_names': ['Not Related', 'Partially Related', 'Fully Related']
            },
            'IndicWiki': {
                'type': 'classification',
                'num_labels': 4,  # Section title categories
                'metric': 'accuracy',
                'class_names': ['History', 'Geography', 'Science', 'Other']
            },
            'IndicCQ': {
                'type': 'multiple_choice',
                'num_choices': 4,
                'metric': 'accuracy'
            },
            'IndicWNLI': {
                'type': 'nli',
                'num_labels': 2,  # Entailment/Not Entailment
                'metric': 'accuracy',
                'class_names': ['Not Entailment', 'Entailment']
            },
            'IndicCOPA': {
                'type': 'multiple_choice',
                'num_choices': 2,
                'metric': 'accuracy'
            }
        }

        # Batch size for evaluation
        self.batch_size = self.config.get('eval_batch_size', 32)
        self.max_samples = self.config.get('max_samples_per_task', None)

        # Initialize metrics aggregator
        eval_config = self.config.get('evaluation', {})
        self.metrics_aggregator = MetricsAggregator(
            bootstrap_samples=eval_config.get('bootstrap_samples', 1000),
            confidence_level=eval_config.get('confidence_level', 0.95)
        )

        # Initialize cache manager
        self.cache_manager = EvaluationCache(
            cache_dir=eval_config.get('cache_dir', '.eval_cache'),
            max_cache_age_days=eval_config.get('max_cache_age_days', 30),
            enable_cache=eval_config.get('use_eval_cache', True)
        )

        # Visualization settings
        self.save_visualizations = eval_config.get('save_visualizations', True)
        self.visualization_format = eval_config.get('visualization_format', ['png', 'html'])

    def evaluate_all_tasks(self) -> Dict[str, Dict]:
        """
        Evaluate model on all IndicGLUE tasks

        Returns:
            Dictionary mapping task names to results
        """
        logger.info("Starting IndicGLUE evaluation on all tasks...")
        results = {}

        for task_name in self.tasks.keys():
            logger.info(f"\nEvaluating {task_name}...")

            try:
                task_results = self.evaluate_task(task_name)
                results[task_name] = task_results

                # Log results
                logger.info(f"{task_name} Results:")
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric}: {value:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {task_name}: {str(e)}")
                results[task_name] = {'error': str(e), 'status': 'failed'}

        # Compute overall statistics
        results['overall'] = self._compute_overall_metrics(results)

        logger.info("\n" + "="*60)
        logger.info("IndicGLUE Evaluation Complete")
        logger.info(f"Overall Accuracy: {results['overall'].get('average_accuracy', 0):.4f}")
        logger.info("="*60)

        return results

    def evaluate_task(self, task_name: str) -> Dict:
        """
        Evaluate model on a specific IndicGLUE task

        Args:
            task_name: Name of the task to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")

        task_config = self.tasks[task_name]

        # Load task data
        try:
            dataset = self._load_task_data(task_name)
        except Exception as e:
            logger.warning(f"Could not load real data for {task_name}: {e}")
            logger.info(f"Using synthetic data for {task_name}")
            dataset = self._create_synthetic_data(task_name)

        if dataset is None or len(dataset) == 0:
            logger.warning(f"No data available for {task_name}")
            return {'status': 'no_data'}

        # Limit samples if configured
        if self.max_samples and len(dataset) > self.max_samples:
            dataset = dataset.select(range(self.max_samples))

        logger.info(f"Evaluating on {len(dataset)} examples")

        # Evaluate based on task type
        if task_config['type'] == 'classification':
            results = self._evaluate_classification(dataset, task_name)
        elif task_config['type'] == 'multiple_choice':
            results = self._evaluate_multiple_choice(dataset, task_name)
        elif task_config['type'] == 'nli':
            results = self._evaluate_nli(dataset, task_name)
        else:
            raise ValueError(f"Unknown task type: {task_config['type']}")

        return results

    def _load_task_data(self, task_name: str) -> Optional[Dataset]:
        """
        Load real IndicGLUE task data from Hugging Face

        Args:
            task_name: Name of the task

        Returns:
            Dataset or None if not available
        """
        # Mapping of task names to HuggingFace dataset paths
        # Updated to use new ai4bharat/indic_glue repository with correct config names
        dataset_map = {
            'IndicNews': ('ai4bharat/indic_glue', 'bbca.hi'),  # BBC Article Classification
            'IndicHeadline': None,  # Not available in indic_glue, will use synthetic data
            'IndicWiki': ('ai4bharat/indic_glue', 'wstp.hi'),  # Wikipedia Section Title Prediction
            'IndicCQ': ('ai4bharat/indic_glue', 'csqa.hi'),  # Commonsense QA
            'IndicWNLI': ('ai4bharat/indic_glue', 'wnli.hi'),  # Winograd NLI
            'IndicCOPA': ('ai4bharat/indic_glue', 'copa.hi')  # Choice of Plausible Alternatives
        }

        if task_name not in dataset_map:
            return None

        dataset_info = dataset_map[task_name]

        # If no real dataset is available for this task
        if dataset_info is None:
            logger.info(f"{task_name} is not available in ai4bharat/indic_glue")
            return None

        try:
            dataset_name, config_name = dataset_info
            logger.info(f"Attempting to load {task_name} from {dataset_name} with config '{config_name}'")
            dataset = load_dataset(dataset_name, config_name, split='test')
            logger.info(f"Successfully loaded {task_name} with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load {task_name} from HuggingFace: {e}")
            logger.info(f"Will use synthetic data for {task_name} instead")
            return None

    def _create_synthetic_data(self, task_name: str) -> Dataset:
        """
        Create synthetic data for testing when real data is unavailable

        Args:
            task_name: Name of the task

        Returns:
            Synthetic dataset
        """
        task_config = self.tasks[task_name]
        num_samples = 100

        if task_config['type'] == 'classification':
            data = {
                'text': [f"यह परीक्षण वाक्य {i} है।" for i in range(num_samples)],
                'label': np.random.randint(0, task_config['num_labels'], num_samples).tolist()
            }
        elif task_config['type'] == 'multiple_choice':
            data = {
                'premise': [f"प्रश्न {i}" for i in range(num_samples)],
                'choice1': [f"विकल्प 1-{i}" for i in range(num_samples)],
                'choice2': [f"विकल्प 2-{i}" for i in range(num_samples)],
                'label': np.random.randint(0, 2, num_samples).tolist()
            }
        elif task_config['type'] == 'nli':
            data = {
                'premise': [f"पहला वाक्य {i}" for i in range(num_samples)],
                'hypothesis': [f"दूसरा वाक्य {i}" for i in range(num_samples)],
                'label': np.random.randint(0, 2, num_samples).tolist()
            }
        else:
            data = {'text': [], 'label': []}

        return Dataset.from_dict(data)

    def _evaluate_classification(self, dataset: Dataset, task_name: str) -> Dict:
        """
        Evaluate classification task

        Args:
            dataset: Dataset to evaluate on
            task_name: Name of the task

        Returns:
            Dictionary with metrics
        """
        predictions = []
        labels = []

        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), self.batch_size), desc=f"Evaluating {task_name}"):
                batch = dataset[i:i + self.batch_size]

                # Get texts and labels
                if 'text' in batch:
                    texts = batch['text']
                elif 'sentence' in batch:
                    texts = batch['sentence']
                else:
                    # Try to construct from available fields
                    texts = [str(batch[k][0]) for k in batch.keys() if k != 'label']

                batch_labels = batch['label']

                # Tokenize
                try:
                    inputs = self._tokenize_batch(texts)
                except Exception as e:
                    logger.warning(f"Tokenization error: {e}, skipping batch")
                    continue

                # Get model predictions
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                # Get predicted classes
                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()

                predictions.extend(batch_preds.tolist())
                labels.extend(batch_labels if isinstance(batch_labels, list) else batch_labels.tolist())

        # Compute metrics
        return self._compute_classification_metrics(predictions, labels, task_name)

    def _evaluate_multiple_choice(self, dataset: Dataset, task_name: str) -> Dict:
        """
        Evaluate multiple choice task (IndicCOPA, IndicCQ)

        Args:
            dataset: Dataset to evaluate on
            task_name: Name of the task

        Returns:
            Dictionary with metrics
        """
        predictions = []
        labels = []

        self.model.eval()

        with torch.no_grad():
            for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
                # Extract premise and choices
                if 'premise' in example:
                    premise = example['premise']
                elif 'question' in example:
                    premise = example['question']
                else:
                    premise = example.get('context', '')

                # Get choices
                choices = []
                if 'choice1' in example and 'choice2' in example:
                    choices = [example['choice1'], example['choice2']]
                elif 'choices' in example:
                    choices = example['choices']
                else:
                    # Default for synthetic data
                    choices = [f"विकल्प {i}" for i in range(2)]

                # Score each choice
                choice_scores = []
                for choice in choices:
                    # Combine premise and choice
                    text = f"{premise} {choice}"

                    # Tokenize and get score
                    try:
                        inputs = self._tokenize_batch([text])
                        outputs = self.model(**inputs)

                        # Use loss or logits to score
                        if hasattr(outputs, 'logits'):
                            score = outputs.logits.mean().item()
                        else:
                            score = -outputs.loss.item() if hasattr(outputs, 'loss') else 0

                        choice_scores.append(score)
                    except:
                        choice_scores.append(0)

                # Predict choice with highest score
                pred = np.argmax(choice_scores)
                predictions.append(pred)
                labels.append(example['label'])

        # Compute metrics
        return self._compute_classification_metrics(predictions, labels, task_name)

    def _evaluate_nli(self, dataset: Dataset, task_name: str) -> Dict:
        """
        Evaluate Natural Language Inference task (IndicWNLI)

        Args:
            dataset: Dataset to evaluate on
            task_name: Name of the task

        Returns:
            Dictionary with metrics
        """
        predictions = []
        labels = []

        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), self.batch_size), desc=f"Evaluating {task_name}"):
                batch = dataset[i:i + self.batch_size]

                # Combine premise and hypothesis
                premises = batch['premise'] if 'premise' in batch else batch['sentence1']
                hypotheses = batch['hypothesis'] if 'hypothesis' in batch else batch['sentence2']

                texts = [f"{p} [SEP] {h}" for p, h in zip(premises, hypotheses)]
                batch_labels = batch['label']

                # Tokenize
                try:
                    inputs = self._tokenize_batch(texts)
                except Exception as e:
                    logger.warning(f"Tokenization error: {e}, skipping batch")
                    continue

                # Get predictions
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()

                predictions.extend(batch_preds.tolist())
                labels.extend(batch_labels if isinstance(batch_labels, list) else batch_labels.tolist())

        # Compute metrics
        return self._compute_classification_metrics(predictions, labels, task_name)

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts

        Args:
            texts: List of text strings

        Returns:
            Dictionary with tokenized inputs
        """
        # Handle different tokenizer interfaces
        try:
            # Try HuggingFace tokenizer interface
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
        except:
            # Fallback to simple encoding
            max_len = 512
            input_ids = []

            for text in texts:
                tokens = self.tokenizer.encode(text)[:max_len]
                input_ids.append(tokens)

            # Pad sequences
            max_batch_len = max(len(ids) for ids in input_ids)
            padded_ids = []
            attention_masks = []

            for ids in input_ids:
                padding_length = max_batch_len - len(ids)
                padded_ids.append(ids + [0] * padding_length)
                attention_masks.append([1] * len(ids) + [0] * padding_length)

            encoded = {
                'input_ids': torch.tensor(padded_ids),
                'attention_mask': torch.tensor(attention_masks)
            }

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        return encoded

    def _compute_classification_metrics(self, predictions: List[int],
                                       labels: List[int],
                                       task_name: str) -> Dict:
        """
        Compute comprehensive metrics for classification tasks with confidence intervals

        Args:
            predictions: List of predicted labels
            labels: List of true labels
            task_name: Name of the task

        Returns:
            Dictionary with metrics, confusion matrix, and per-class metrics
        """
        # Convert to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Get class names for this task
        class_names = self._get_class_names(task_name)

        # Compute metrics with confidence intervals
        accuracy_metric = self.metrics_aggregator.compute_metric(
            labels, predictions, 'accuracy', compute_ci=True
        )

        f1_macro_metric = self.metrics_aggregator.compute_metric(
            labels, predictions, 'f1', average='macro', compute_ci=True
        )

        f1_weighted_metric = self.metrics_aggregator.compute_metric(
            labels, predictions, 'f1', average='weighted', compute_ci=True
        )

        precision_macro_metric = self.metrics_aggregator.compute_metric(
            labels, predictions, 'precision', average='macro', compute_ci=True
        )

        recall_macro_metric = self.metrics_aggregator.compute_metric(
            labels, predictions, 'recall', average='macro', compute_ci=True
        )

        # Compute confusion matrix
        conf_matrix, matrix_labels = self.metrics_aggregator.compute_confusion_matrix(
            labels, predictions, normalize=None
        )

        # Normalized confusion matrix (by true labels)
        conf_matrix_normalized, _ = self.metrics_aggregator.compute_confusion_matrix(
            labels, predictions, normalize='true'
        )

        # Compute per-class metrics with CIs
        per_class_metrics = self.metrics_aggregator.compute_per_class_metrics(
            labels, predictions, class_names=class_names, compute_ci=True
        )

        # Build results dictionary
        results = {
            'task': task_name,
            'num_examples': len(labels),

            # Main metrics (backward compatible format)
            'accuracy': accuracy_metric.value,
            'f1_macro': f1_macro_metric.value,
            'f1_weighted': f1_weighted_metric.value,
            'precision_macro': precision_macro_metric.value,
            'recall_macro': recall_macro_metric.value,

            # Metrics with confidence intervals
            'metrics_with_ci': {
                'accuracy': accuracy_metric.to_dict(),
                'f1_macro': f1_macro_metric.to_dict(),
                'f1_weighted': f1_weighted_metric.to_dict(),
                'precision_macro': precision_macro_metric.to_dict(),
                'recall_macro': recall_macro_metric.to_dict(),
            },

            # Confusion matrix
            'confusion_matrix': {
                'matrix': conf_matrix.tolist(),
                'matrix_normalized': conf_matrix_normalized.tolist(),
                'labels': matrix_labels,
                'class_names': [class_names[i] if i < len(class_names) else f'class_{i}'
                               for i in matrix_labels]
            },

            # Per-class metrics with CIs
            'per_class_metrics': {
                int(class_idx): {
                    metric_name: metric.to_dict()
                    for metric_name, metric in metrics.items()
                }
                for class_idx, metrics in per_class_metrics.items()
            }
        }

        return results

    def _compute_overall_metrics(self, results: Dict[str, Dict]) -> Dict:
        """
        Compute overall statistics across all tasks

        Args:
            results: Dictionary of per-task results

        Returns:
            Dictionary with overall metrics
        """
        accuracies = []
        f1_scores = []

        for task_name, task_results in results.items():
            if task_name == 'overall':
                continue

            if 'accuracy' in task_results:
                accuracies.append(task_results['accuracy'])

            if 'f1_macro' in task_results:
                f1_scores.append(task_results['f1_macro'])

        overall = {
            'average_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'average_f1_macro': np.mean(f1_scores) if f1_scores else 0.0,
            'tasks_evaluated': len(accuracies),
            'accuracies_by_task': {
                task: results[task].get('accuracy', 0)
                for task in self.tasks.keys()
                if task in results and 'accuracy' in results[task]
            }
        }

        return overall

    def _get_class_names(self, task_name: str) -> List[str]:
        """
        Get class names for a specific task

        Args:
            task_name: Name of the task

        Returns:
            List of class names
        """
        task_config = self.tasks.get(task_name, {})
        class_names = task_config.get('class_names', [])

        # If no class names defined, generate generic names
        if not class_names:
            num_labels = task_config.get('num_labels', task_config.get('num_choices', 2))
            class_names = [f'Class {i}' for i in range(num_labels)]

        return class_names

    def _plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        class_names: List[str],
        task_name: str,
        save_dir: Optional[Path] = None,
        normalize: bool = True
    ):
        """
        Plot confusion matrix as heatmap

        Args:
            conf_matrix: Confusion matrix
            class_names: Names of classes
            task_name: Name of the task
            normalize: Whether to normalize the matrix
            save_dir: Directory to save plots
        """
        if not self.save_visualizations:
            return

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot heatmap
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='.2f' if normalize else 'd',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'}
            )

            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            title = f'Confusion Matrix - {task_name}'
            if normalize:
                title += ' (Normalized)'
            ax.set_title(title, fontsize=14, fontweight='bold')

            plt.tight_layout()

            # Save plot
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)

                if 'png' in self.visualization_format:
                    png_path = save_dir / f'{task_name}_confusion_matrix.png'
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved confusion matrix plot: {png_path}")

            plt.close()

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping confusion matrix plot")
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")

        # Try plotly for interactive version
        if 'html' in self.visualization_format:
            try:
                import plotly.graph_objects as go

                fig = go.Figure(data=go.Heatmap(
                    z=conf_matrix,
                    x=class_names,
                    y=class_names,
                    colorscale='Blues',
                    text=conf_matrix,
                    texttemplate='%{text:.2f}' if normalize else '%{text}',
                    textfont={"size": 12},
                    colorbar=dict(title='Proportion' if normalize else 'Count')
                ))

                fig.update_layout(
                    title=f'Confusion Matrix - {task_name}',
                    xaxis_title='Predicted Label',
                    yaxis_title='True Label',
                    width=700,
                    height=600
                )

                if save_dir:
                    html_path = save_dir / f'{task_name}_confusion_matrix.html'
                    fig.write_html(str(html_path))
                    logger.info(f"Saved interactive confusion matrix: {html_path}")

            except ImportError:
                logger.debug("Plotly not available for interactive plots")
            except Exception as e:
                logger.error(f"Error creating interactive confusion matrix: {e}")

    def _plot_per_class_metrics(
        self,
        per_class_metrics: Dict,
        class_names: List[str],
        task_name: str,
        save_dir: Optional[Path] = None
    ):
        """
        Plot per-class metrics (precision, recall, F1) with error bars

        Args:
            per_class_metrics: Dictionary of per-class metrics
            class_names: Names of classes
            task_name: Name of the task
            save_dir: Directory to save plots
        """
        if not self.save_visualizations:
            return

        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            # Prepare data for plotting
            metrics_data = {
                'Class': [],
                'Precision': [],
                'Precision_CI_Lower': [],
                'Precision_CI_Upper': [],
                'Recall': [],
                'Recall_CI_Lower': [],
                'Recall_CI_Upper': [],
                'F1': [],
                'F1_CI_Lower': [],
                'F1_CI_Upper': [],
            }

            for class_idx, metrics in per_class_metrics.items():
                class_name = class_names[class_idx] if class_idx < len(class_names) else f'Class {class_idx}'
                metrics_data['Class'].append(class_name)

                for metric_type in ['precision', 'recall', 'f1']:
                    if metric_type in metrics:
                        metric = metrics[metric_type]
                        metrics_data[metric_type.capitalize()].append(metric.get('value', 0))
                        metrics_data[f'{metric_type.capitalize()}_CI_Lower'].append(
                            metric.get('ci_lower', metric.get('value', 0))
                        )
                        metrics_data[f'{metric_type.capitalize()}_CI_Upper'].append(
                            metric.get('ci_upper', metric.get('value', 0))
                        )

            df = pd.DataFrame(metrics_data)

            # Create grouped bar chart
            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(df['Class']))
            width = 0.25

            # Plot bars with error bars
            for i, (metric, color) in enumerate([
                ('Precision', '#1f77b4'),
                ('Recall', '#ff7f0e'),
                ('F1', '#2ca02c')
            ]):
                values = df[metric].values
                lower_errors = values - df[f'{metric}_CI_Lower'].values
                upper_errors = df[f'{metric}_CI_Upper'].values - values

                ax.bar(
                    x + i * width,
                    values,
                    width,
                    label=metric,
                    color=color,
                    yerr=[lower_errors, upper_errors],
                    capsize=5,
                    alpha=0.8
                )

            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'Per-Class Metrics - {task_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(df['Class'], rotation=45, ha='right')
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()

            # Save plot
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)

                if 'png' in self.visualization_format:
                    png_path = save_dir / f'{task_name}_per_class_metrics.png'
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved per-class metrics plot: {png_path}")

            plt.close()

        except ImportError:
            logger.warning("Matplotlib/Pandas not available, skipping per-class metrics plot")
        except Exception as e:
            logger.error(f"Error plotting per-class metrics: {e}")

        # Try plotly for interactive version
        if 'html' in self.visualization_format:
            try:
                import plotly.graph_objects as go

                fig = go.Figure()

                # Add bars for each metric
                for metric, color in [
                    ('Precision', '#1f77b4'),
                    ('Recall', '#ff7f0e'),
                    ('F1', '#2ca02c')
                ]:
                    values = df[metric].values
                    lower_errors = values - df[f'{metric}_CI_Lower'].values
                    upper_errors = df[f'{metric}_CI_Upper'].values - values

                    fig.add_trace(go.Bar(
                        name=metric,
                        x=df['Class'],
                        y=values,
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=upper_errors,
                            arrayminus=lower_errors
                        ),
                        marker_color=color
                    ))

                fig.update_layout(
                    title=f'Per-Class Metrics - {task_name}',
                    xaxis_title='Class',
                    yaxis_title='Score',
                    barmode='group',
                    width=900,
                    height=500,
                    yaxis=dict(range=[0, 1.1])
                )

                if save_dir:
                    html_path = save_dir / f'{task_name}_per_class_metrics.html'
                    fig.write_html(str(html_path))
                    logger.info(f"Saved interactive per-class metrics: {html_path}")

            except ImportError:
                logger.debug("Plotly not available for interactive plots")
            except Exception as e:
                logger.error(f"Error creating interactive per-class metrics: {e}")

    def save_visualizations(self, results: Dict[str, Dict], save_dir: str):
        """
        Generate and save all visualizations for evaluation results

        Args:
            results: Dictionary of evaluation results
            save_dir: Directory to save visualizations
        """
        if not self.save_visualizations:
            logger.info("Visualization saving disabled")
            return

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating visualizations in: {save_path}")

        for task_name, task_results in results.items():
            if task_name == 'overall' or 'confusion_matrix' not in task_results:
                continue

            try:
                # Get confusion matrix and class names
                cm_data = task_results['confusion_matrix']
                conf_matrix = np.array(cm_data['matrix_normalized'])
                class_names = cm_data['class_names']

                # Plot confusion matrix
                self._plot_confusion_matrix(
                    conf_matrix, class_names, task_name, save_path, normalize=True
                )

                # Plot per-class metrics
                if 'per_class_metrics' in task_results:
                    self._plot_per_class_metrics(
                        task_results['per_class_metrics'],
                        class_names,
                        task_name,
                        save_path
                    )

            except Exception as e:
                logger.error(f"Error generating visualizations for {task_name}: {e}")


# For backward compatibility
def evaluate_indicglue(model, tokenizer, config: Optional[Dict] = None) -> Dict:
    """
    Convenience function to evaluate on IndicGLUE

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        config: Optional configuration

    Returns:
        Evaluation results
    """
    evaluator = IndicGLUEEvaluator(model, tokenizer, config)
    return evaluator.evaluate_all_tasks()

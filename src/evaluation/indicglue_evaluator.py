"""
IndicGLUE Benchmark Evaluator for Hindi Language Models

This module implements comprehensive evaluation on the IndicGLUE benchmark,
which includes multiple tasks for evaluating Indian language understanding.

IndicGLUE Tasks:
- IndicNews: Article genre classification
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
import time
from torch.utils.data import DataLoader

# Import new utilities
from .metrics_utils import MetricsAggregator, Metric
from .evaluation_cache import EvaluationCache

# Import classification models for wrapping language models
from ..models.classification_models import wrap_model_for_classification

logger = logging.getLogger(__name__)


# BBCA (BBC Articles Classification) label mapping
# The IndicGLUE BBCA dataset uses string labels, but models return integer predictions
# This mapping converts string labels to integer indices for metric computation
BBCA_LABEL_MAP = {
    'business': 0,
    'china': 1,
    'entertainment': 2,
    'india': 3,
    'institutional': 4,
    'international': 5,
    'learningenglish': 6,
    'multimedia': 7,
    'news': 8,
    'pakistan': 9,
    'science': 10,
    'social': 11,
    'southasia': 12,
    'sport': 13
}

# DiscourseMode label mapping
# The IndicGLUE DiscourseMode dataset uses string labels, but models return integer predictions
# This mapping converts string labels to integer indices for metric computation
DISCOURSE_MODE_LABEL_MAP = {
    'Narrative': 0,
    'Descriptive': 1,
    'Dialogue': 2,
    'Informative': 3,
    'Argumentative': 4,
    'Other': 5
}

# Task-specific split remappings for corrupted datasets
# Format: {task_name: {requested_split: actual_split_to_load}}
SPLIT_REMAPPING = {
    'Choice of Plausible Alternatives': {
        'test': 'validation',      # Test corrupted → use validation (88 examples)
        'validation': 'test',       # Use corrupted test as validation (449 examples)
        'train': 'train'           # Train is fine, keep unchanged (362 examples)
    }
}


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
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config or {}

        # Device setup
        self.device = next(model.parameters()).device
        logger.info(f"IndicGLUE evaluator initialized on device: {self.device}")

        # Extract max_length from config (IndicBERT paper specifies 128 tokens)
        self.max_length = int(self.config.get('max_length', 128))
        logger.info(f"Max sequence length: {self.max_length} tokens (IndicBERT paper uses 128)")

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.warning(f"Tokenizer had no pad_token, using eos_token: {self.tokenizer.eos_token}")
        logger.info(f"Tokenizer pad_token: {self.tokenizer.pad_token} (id: {self.tokenizer.pad_token_id})")

        # Detect if the model is already a classification model or a language model
        # Language models have output shape [batch, seq_len, vocab_size]
        # Classification models have output shape [batch, num_classes]
        self.is_language_model = self._is_language_model(model)

        # Dictionary to store task-specific wrapped models
        self.wrapped_models = {}

        logger.info(f"Model type detected: {'Language Model' if self.is_language_model else 'Classification Model'}")

        # Task configurations
        # Using descriptive names mapped to HuggingFace configs
        self.tasks = {
            'BBCArticlesClassification': {
                'type': 'classification',
                'num_labels': 14,  # 14 classes in BBC dataset
                'metric': 'accuracy',
                'class_names': ['business', 'china', 'entertainment', 'india', 'institutional',
                               'international', 'learningenglish', 'multimedia', 'news', 'pakistan',
                               'science', 'social', 'southasia', 'sport'],  # All 14 classes in order matching BBCA_LABEL_MAP
                'hf_config': 'bbca.hi'
            },
            'Wikipedia Section Title Prediction': {
                'type': 'multiple_choice',
                'num_labels': 2,  # Binary classification per candidate (correct=1, incorrect=0)
                'num_candidates': 4,  # Four title choices total
                'use_binary_per_candidate': True,  # Process each candidate separately
                'metric': 'accuracy',
                'hf_config': 'wstp.hi'
            },
            'Cloze-style multiple-choice QA': {
                'type': 'multiple_choice',
                'num_labels': 4,  # Four answer choices (cloze-style fill-in-the-blank)
                'metric': 'accuracy',
                'hf_config': 'csqa.hi'
            },
            'WinogradNLI': {
                'type': 'nli',
                'num_labels': 3,  # Not Entailment, Entailment, None
                'metric': 'accuracy',
                'class_names': ['Not Entailment', 'Entailment', 'None'],
                'hf_config': 'wnli.hi'
            },
            'Choice of Plausible Alternatives': {
                'type': 'multiple_choice',
                'num_labels': 2,  # Two plausible alternatives
                'metric': 'accuracy',
                'hf_config': 'copa.hi'
            },
            # New tasks added
            'MovieReviewSentiment': {
                'type': 'classification',
                'num_labels': 3,  # Positive, Negative, Neutral
                'metric': 'accuracy',
                'class_names': ['Negative', 'Neutral', 'Positive'],
                'hf_config': 'iitp-mr.hi'
            },
            'ProductReviewSentiment': {
                'type': 'classification',
                'num_labels': 3,  # Positive, Negative, Neutral
                'metric': 'accuracy',
                'class_names': ['Negative', 'Neutral', 'Positive'],
                'hf_config': 'iitp-pr.hi'
            },
            'DiscourseMode': {
                'type': 'classification',
                'num_labels': 6,  # FIXED: 6 classes, not 9
                'metric': 'accuracy',
                'class_names': ['Narrative', 'Descriptive', 'Dialogue',
                               'Informative', 'Argumentative', 'Other'],
                'hf_config': 'md.hi'
            }
        }

        # Batch size for evaluation - explicitly convert to int
        self.batch_size = int(self.config.get('eval_batch_size', 32))
        # max_samples can be None or int
        max_samples_val = self.config.get('max_samples_per_task', None)
        self.max_samples = int(max_samples_val) if max_samples_val is not None else None

        # Initialize metrics aggregator
        eval_config = self.config.get('evaluation', {})
        self.metrics_aggregator = MetricsAggregator(
            bootstrap_samples=int(eval_config.get('bootstrap_samples', 1000)),
            confidence_level=float(eval_config.get('confidence_level', 0.95))
        )

        # Label validation mode
        # - strict (True): Raise ValueError on invalid labels (recommended for development)
        # - permissive (False): Log warnings and default to class 0 (for production with noisy data)
        self.strict_label_validation = bool(eval_config.get('strict_label_validation', True))
        logger.info(f"Label validation mode: {'STRICT' if self.strict_label_validation else 'PERMISSIVE'} "
                   f"(invalid labels will {'raise errors' if self.strict_label_validation else 'default to 0'})")

        # Initialize cache manager
        self.cache_manager = EvaluationCache(
            cache_dir=eval_config.get('cache_dir', '.eval_cache'),
            max_cache_age_days=int(eval_config.get('max_cache_age_days', 30)),
            enable_cache=bool(eval_config.get('use_eval_cache', True))
        )

        # Visualization settings
        self.save_visualizations = eval_config.get('save_visualizations', True)
        self.visualization_format = eval_config.get('visualization_format', ['png', 'html'])

        # Track evaluation mode (fine-tuned vs zero-shot)
        # This is critical for routing multiple-choice tasks correctly:
        # - Fine-tuned mode: Use trained classification heads
        # - Zero-shot mode: Use perplexity-based scoring
        self._is_finetuned_mode = False
        self._current_fine_tuning_info = None
        logger.info("Evaluation mode initialized: zero-shot (will switch to fine-tuned after training)")

    def _is_language_model(self, model) -> bool:
        """
        Detect if the model is a language model (outputs per-token predictions)
        or a classification model (outputs per-example predictions).

        Returns:
            True if language model, False if classification model
        """
        # Check if model has classification head attributes
        if hasattr(model, 'classifier'):
            return False

        # Check the model class name
        model_class_name = model.__class__.__name__
        if 'Classification' in model_class_name or 'Classifier' in model_class_name:
            return False

        # Check for language modeling head
        if 'LMHead' in model_class_name or 'MaskedLM' in model_class_name or 'GPT' in model_class_name:
            return True

        # Default: assume it's a language model for safety
        logger.warning(f"Could not definitively determine model type for {model_class_name}, assuming language model")
        return True

    def _get_model_config(self) -> Dict:
        """Extract model configuration for wrapping."""
        # Try to get hidden size from model config
        if hasattr(self.base_model, 'config'):
            hidden_size = self.base_model.config.hidden_size
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'config'):
            hidden_size = self.base_model.model.config.hidden_size
        else:
            # Default hidden size
            hidden_size = 768
            logger.warning(f"Could not determine hidden size, using default: {hidden_size}")

        # Try to detect model type
        model_class_name = self.base_model.__class__.__name__.lower()
        if 'gpt' in model_class_name:
            model_type = 'gpt'
        elif 'deberta' in model_class_name:
            model_type = 'deberta'
        else:
            model_type = 'gpt'  # Default
            logger.warning(f"Could not determine model type from {self.base_model.__class__.__name__}, using 'gpt'")

        return {
            'hidden_size': hidden_size,
            'model_type': model_type
        }

    def _convert_bbca_labels_to_int(self, labels):
        """
        Convert BBCA string labels to integer indices with validation.

        The IndicGLUE BBCA dataset stores labels as strings (e.g., 'india', 'pakistan'),
        but models return integer predictions. This method converts string labels to
        integers using BBCA_LABEL_MAP.

        Args:
            labels: Single label (str) or list of labels (List[str])

        Returns:
            Integer label or list of integer labels

        Raises:
            ValueError: If strict_label_validation=True and label is invalid
        """
        # Handle single label
        if isinstance(labels, str):
            if labels not in BBCA_LABEL_MAP:
                msg = (f"Invalid BBCA label '{labels}'. "
                      f"Valid labels: {list(BBCA_LABEL_MAP.keys())}")
                if self.strict_label_validation:
                    raise ValueError(msg)
                else:
                    logger.error(msg + " - Defaulting to 0 (business)")
                    return 0
            return BBCA_LABEL_MAP[labels]

        # Handle list of labels
        converted_labels = []
        for label in labels:
            if isinstance(label, str):
                if label not in BBCA_LABEL_MAP:
                    msg = (f"Invalid BBCA label '{label}'. "
                          f"Valid labels: {list(BBCA_LABEL_MAP.keys())}")
                    if self.strict_label_validation:
                        raise ValueError(msg)
                    else:
                        logger.error(msg + " - Defaulting to 0 (business)")
                        converted_labels.append(0)
                else:
                    converted_labels.append(BBCA_LABEL_MAP[label])
            else:
                # Already an integer
                converted_labels.append(label)

        return converted_labels

    def _convert_discourse_mode_labels_to_int(self, labels):
        """
        Convert DiscourseMode string labels to integer indices with validation.

        The IndicGLUE DiscourseMode dataset stores labels as strings (e.g., 'Narrative', 'Descriptive'),
        but models return integer predictions. This method converts string labels to
        integers using DISCOURSE_MODE_LABEL_MAP.

        Args:
            labels: Single label (str) or list of labels (List[str])

        Returns:
            Integer label or list of integer labels

        Raises:
            ValueError: If strict_label_validation=True and label is invalid
        """
        # Handle single label
        if isinstance(labels, str):
            if labels not in DISCOURSE_MODE_LABEL_MAP:
                msg = (f"Invalid DiscourseMode label '{labels}'. "
                      f"Valid labels: {list(DISCOURSE_MODE_LABEL_MAP.keys())}")
                if self.strict_label_validation:
                    raise ValueError(msg)
                else:
                    logger.error(msg + " - Defaulting to 0 (Narrative)")
                    return 0
            return DISCOURSE_MODE_LABEL_MAP[labels]

        # Handle list of labels
        converted_labels = []
        for label in labels:
            if isinstance(label, str):
                if label not in DISCOURSE_MODE_LABEL_MAP:
                    msg = (f"Invalid DiscourseMode label '{label}'. "
                          f"Valid labels: {list(DISCOURSE_MODE_LABEL_MAP.keys())}")
                    if self.strict_label_validation:
                        raise ValueError(msg)
                    else:
                        logger.error(msg + " - Defaulting to 0 (Narrative)")
                        converted_labels.append(0)
                else:
                    converted_labels.append(DISCOURSE_MODE_LABEL_MAP[label])
            else:
                # Already an integer
                converted_labels.append(label)

        return converted_labels

    def _get_model_for_task(self, task_name: str, for_training: bool = False):
        """
        Get the appropriate model for a specific task.
        If the base model is a language model, wrap it with a classification head.

        Args:
            task_name: Name of the task
            for_training: If True, creates model with trainable head (frozen base)
                         If False, creates fully frozen model for inference

        Returns:
            Model ready for the task
        """
        if not self.is_language_model:
            # Already a classification model, use as-is
            return self.base_model

        # Create separate cache keys for training vs evaluation
        cache_key = f"{task_name}_{'train' if for_training else 'eval'}"

        # Check if we already have a wrapped model for this task and mode
        if cache_key in self.wrapped_models:
            return self.wrapped_models[cache_key]

        # Wrap the language model with a classification head
        task_config = self.tasks[task_name]
        model_config = self._get_model_config()

        # Determine number of classes based on task type
        task_type = task_config['type']

        if task_type == 'classification':
            # Text classification tasks (BBC, Sentiment, Discourse)
            num_classes = task_config['num_labels']
            logger.info(f"Task '{task_name}' is classification with {num_classes} classes")

        elif task_type == 'nli':
            # Natural Language Inference tasks (WinogradNLI)
            num_classes = task_config['num_labels']
            logger.info(f"Task '{task_name}' is NLI with {num_classes} classes")

        elif task_type == 'multiple_choice':
            # Multiple choice tasks (COPA, CSQA, Wikipedia Section Title)
            # Treat as multi-class classification
            num_classes = task_config['num_labels']
            logger.info(f"Task '{task_name}' is multiple-choice, treating as {num_classes}-class classification")

        else:
            # Unknown task type
            raise ValueError(
                f"Unknown task type '{task_type}' for task '{task_name}'. "
                f"Supported types: 'classification', 'nli', 'multiple_choice'"
            )

        logger.info(f"Wrapping model for task '{task_name}' with {num_classes} classes "
                   f"(mode: {'training' if for_training else 'evaluation'})...")

        # Official IndicBERT uses zero dropout in base model config
        # (attention_probs_dropout_prob=0, hidden_dropout_prob=0)
        # Match this by using zero dropout everywhere to avoid train-test mismatch
        dropout = 0.0

        wrapped_model = wrap_model_for_classification(
            lm_model=self.base_model,
            model_type=model_config['model_type'],
            num_classes=num_classes,
            hidden_size=model_config['hidden_size'],
            dropout=dropout,
            freeze_base=True,  # Always freeze base model (only train heads)
            pooling_strategy='auto'
        )

        # Move to correct device
        wrapped_model = wrapped_model.to(self.device)

        # For training mode, ensure classification head has gradients enabled
        if for_training:
            for param in wrapped_model.classifier.parameters():
                param.requires_grad = True
            num_trainable = sum(p.numel() for p in wrapped_model.classifier.parameters())
            logger.info(f"Enabled gradients for classification head ({num_trainable:,} trainable parameters)")

        # Cache the wrapped model
        self.wrapped_models[cache_key] = wrapped_model

        return wrapped_model

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

    # Tasks to skip due to dataset issues
    SKIP_TASKS = {
        'WinogradNLI': 'WNLI contains only entailment class for train and validation sets while the test set contains None',
        # CSQA removed - now handled via task_specific_splits in config with custom splits
    }

    def evaluate_task(self, task_name: str) -> Dict:
        """
        Evaluate model on a specific IndicGLUE task (with optional fine-tuning)

        Args:
            task_name: Name of the task to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")

        # Check if this task should be skipped due to dataset issues
        if task_name in self.SKIP_TASKS:
            skip_reason = self.SKIP_TASKS[task_name]
            logger.warning(f"Skipping {task_name}: {skip_reason}")
            return {
                'status': 'skipped',
                'reason': skip_reason,
                'task': task_name
            }

        task_config = self.tasks[task_name]

        # Check if fine-tuning is enabled
        ft_config = self.config.get('evaluation', {}).get('benchmarks', {}) \
                        .get('indicglue', {}).get('fine_tuning', {})
        fine_tune_enabled = ft_config.get('enabled', False)

        # Fine-tuning applies to all task types now (classification, NLI, and multiple-choice)
        if fine_tune_enabled:
            logger.info(f"Fine-tuning mode enabled for {task_name}")

            # Load all splits
            splits = self._load_all_splits(task_name)

            if 'train' not in splits:
                logger.warning(f"Missing train split for {task_name}, "
                              f"falling back to zero-shot evaluation")
                fine_tune_enabled = False
            else:
                # Check if validation set is available
                has_validation = 'validation' in splits
                if not has_validation:
                    logger.info(f"Fine-tuning {task_name} without validation set")
                # Fine-tune on train (and val if available)
                logger.info(f"Fine-tuning on {task_name}...")
                start_time = time.time()

                finetuned_model = self.fine_tune_task(
                    task_name,
                    splits['train'],
                    splits.get('validation', None)  # Pass None if no validation set
                )

                fine_tune_time = time.time() - start_time

                # Evaluate on test split
                logger.info(f"Evaluating fine-tuned model on test set...")
                test_dataset = splits.get('test')

                if test_dataset is None:
                    logger.error(f"No test split for {task_name}")
                    return {'status': 'no_test_data'}

                # Apply sample limit if configured
                if self.max_samples and len(test_dataset) > self.max_samples:
                    test_dataset = test_dataset.select(range(self.max_samples))

                # Evaluate with fine-tuned model
                results = self._evaluate_with_model(
                    finetuned_model,
                    test_dataset,
                    task_name
                )

                results['fine_tuned'] = True
                results['fine_tuning_time_seconds'] = fine_tune_time

        if not fine_tune_enabled:
            # Zero-shot evaluation (original behavior)
            logger.info(f"Zero-shot mode for {task_name}")

            # Clear fine-tuning info for zero-shot mode
            # This enables perplexity-based scoring for multiple-choice tasks
            self._current_fine_tuning_info = None
            self._is_finetuned_mode = False

            # Load test data
            try:
                dataset = self._load_task_data(task_name, split='test')
            except Exception as e:
                logger.error(f"Could not load data for {task_name}: {e}")
                return {'error': str(e), 'status': 'failed'}

            if dataset is None or len(dataset) == 0:
                logger.warning(f"No data available for {task_name}")
                return {'status': 'no_data'}

            # Apply sample limit if configured
            if self.max_samples and len(dataset) > self.max_samples:
                dataset = dataset.select(range(self.max_samples))

            # Get model for zero-shot evaluation
            model = self._get_model_for_task(task_name, for_training=False)

            # Evaluate
            results = self._evaluate_with_model(model, dataset, task_name)
            results['fine_tuned'] = False

        return results

    def _evaluate_with_model(self, model, dataset: Dataset, task_name: str) -> Dict:
        """
        Unified evaluation routing based on task type AND evaluation mode.

        CRITICAL: Multiple-choice tasks can be evaluated two ways:
        - Fine-tuned mode: Use trained classification head (treat as N-class classification)
        - Zero-shot mode: Use perplexity-based scoring

        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            task_name: Name of the task

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {len(dataset)} examples")

        task_config = self.tasks[task_name]

        # Detect if we're in fine-tuned mode
        is_finetuned = self._is_finetuned_mode

        # Route based on task type AND evaluation mode
        # PRIORITY 1: Check for binary per-candidate tasks (WSTP)
        if task_config.get('use_binary_per_candidate', False):
            if is_finetuned:
                # Binary per-candidate evaluation (WSTP)
                logger.info(f"[ROUTING] {task_name}: Binary per-candidate mode → Scoring each candidate separately "
                           f"(binary classification with {task_config.get('num_candidates', 4)} candidates)")
                return self._evaluate_binary_candidates(dataset, task_name, model)
            else:
                # Zero-shot mode for binary tasks (fallback to perplexity)
                logger.info(f"[ROUTING] {task_name}: Zero-shot binary task → Using perplexity scoring")
                return self._evaluate_multiple_choice(dataset, task_name)

        # PRIORITY 2: Standard multiple-choice tasks (COPA, CSQA)
        elif task_config['type'] == 'multiple_choice':
            if is_finetuned:
                # Fine-tuned mode: Use trained classification head
                logger.info(f"[ROUTING] {task_name}: Fine-tuned mode → Using classification head "
                           f"({task_config['num_labels']}-class classification)")
                return self._evaluate_classification(dataset, task_name, model)
            else:
                # Zero-shot mode: Use perplexity-based scoring
                logger.info(f"[ROUTING] {task_name}: Zero-shot mode → Using perplexity scoring")
                return self._evaluate_multiple_choice(dataset, task_name)

        # PRIORITY 3: Classification and NLI tasks
        elif task_config['type'] in ['classification', 'nli']:
            # Classification and NLI tasks always use classification heads
            logger.info(f"[ROUTING] {task_name}: Classification/NLI task → Using classification head")
            return self._evaluate_classification(dataset, task_name, model)

        else:
            raise ValueError(f"Unknown task type: {task_config['type']}")

    def _load_task_data(self, task_name: str, split: str = 'test') -> Optional[Dataset]:
        """
        Load real IndicGLUE task data from Hugging Face

        Automatically applies split remapping for tasks with corrupted data.

        Args:
            task_name: Name of the task
            split: Dataset split to load ('train', 'validation', 'test')

        Returns:
            Dataset or None if not available
        """
        # Apply split remapping if configured for this task
        original_split = split
        if task_name in SPLIT_REMAPPING:
            remap_config = SPLIT_REMAPPING[task_name]
            split = remap_config.get(split, split)

            if split != original_split:
                logger.info(
                    f"{task_name}: Remapping split '{original_split}' → '{split}' "
                    f"(working around corrupted dataset)"
                )

        # Mapping of task names to HuggingFace dataset paths
        # Using descriptive task names mapped to ai4bharat/indic_glue configs
        dataset_map = {
            'BBCArticlesClassification': ('ai4bharat/indic_glue', 'bbca.hi'),
            'Wikipedia Section Title Prediction': ('ai4bharat/indic_glue', 'wstp.hi'),
            'Cloze-style multiple-choice QA': ('ai4bharat/indic_glue', 'csqa.hi'),
            'WinogradNLI': ('ai4bharat/indic_glue', 'wnli.hi'),
            'Choice of Plausible Alternatives': ('ai4bharat/indic_glue', 'copa.hi'),
            # New tasks
            'MovieReviewSentiment': ('ai4bharat/indic_glue', 'iitp-mr.hi'),
            'ProductReviewSentiment': ('ai4bharat/indic_glue', 'iitp-pr.hi'),
            'DiscourseMode': ('ai4bharat/indic_glue', 'md.hi')
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
            logger.info(f"Attempting to load {task_name} from {dataset_name} with config '{config_name}' (split={split})")
            dataset = load_dataset(dataset_name, config_name, split=split)
            logger.info(f"Successfully loaded {task_name} {split} split with {len(dataset)} examples")

            return dataset
        except Exception as e:
            logger.debug(f"Failed to load {task_name} {split} split from HuggingFace: {e}")
            return None

    def _load_complete_dataset(self, task_name: str) -> Optional[Dataset]:
        """
        Load ALL available splits from HuggingFace and combine them.

        This creates a complete dataset by concatenating train, validation,
        and test splits (whichever are available). This solves issues with:
        - Missing validation splits (BBCA)
        - Corrupted test sets (COPA, WNLI)
        - Missing train/val splits (CSQA)

        Args:
            task_name: Name of the task

        Returns:
            Combined dataset with all available examples, or None if no data
        """
        from datasets import concatenate_datasets

        logger.info(f"Loading complete dataset for {task_name} (all available splits)...")

        available_datasets = []
        split_info = []

        # Try loading all possible splits
        for split_name in ['train', 'validation', 'test']:
            try:
                dataset = self._load_task_data(task_name, split=split_name)
                if dataset is not None and len(dataset) > 0:
                    available_datasets.append(dataset)
                    split_info.append(f"{split_name}({len(dataset)})")
                    logger.info(f"  Loaded {split_name}: {len(dataset)} examples")
            except Exception as e:
                logger.debug(f"  Split '{split_name}' not available: {e}")

        if not available_datasets:
            logger.error(f"No data available for {task_name}")
            return None

        # Combine all available datasets
        if len(available_datasets) == 1:
            combined = available_datasets[0]
        else:
            combined = concatenate_datasets(available_datasets)

        logger.info(f"  Combined dataset: {len(combined)} total examples from [{', '.join(split_info)}]")

        return combined

    def _create_custom_splits_from_complete(self, task_name: str,
                                           override_config: Optional[Dict] = None) -> Dict[str, Dataset]:
        """
        Create custom train/val/test splits from complete dataset.

        Strategy:
        1. Load ALL available data from HuggingFace (combine all splits)
        2. Shuffle with reproducible seed
        3. Split according to configured ratios

        Args:
            task_name: Name of the task

        Returns:
            Dictionary with 'train', 'validation', 'test' keys
        """
        import numpy as np

        logger.info(f"Creating custom splits for {task_name}...")

        # Get split configuration from task-specific config (required parameter now)
        if override_config:
            # Use task-specific ratios from config task_specific_splits
            train_ratio = float(override_config.get('train_ratio', 0.7))
            val_ratio = float(override_config.get('val_ratio', 0.15))
            test_ratio = float(override_config.get('test_ratio', 0.15))
            split_seed = int(override_config.get('split_seed', 42))
            logger.info(f"Using task-specific split ratios for {task_name}")
            logger.info(f"  Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}, Seed: {split_seed}")
        else:
            # Fallback to defaults (should not happen if called correctly)
            train_ratio = 0.7
            val_ratio = 0.15
            test_ratio = 0.15
            split_seed = 42
            logger.warning(f"No override_config provided for {task_name}, using default ratios")

        # Validate ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        assert abs(total_ratio - 1.0) < 1e-6, \
            f"Split ratios must sum to 1.0, got {total_ratio} (train={train_ratio}, val={val_ratio}, test={test_ratio})"

        # Load complete dataset (all available splits combined)
        complete_dataset = self._load_complete_dataset(task_name)

        if complete_dataset is None or len(complete_dataset) == 0:
            logger.error(f"No data available for {task_name}")
            return {}

        # Create shuffled indices with reproducible seed
        total_size = len(complete_dataset)
        indices = np.arange(total_size)
        rng = np.random.RandomState(split_seed)
        rng.shuffle(indices)

        # Calculate split sizes
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        # test_size is the remainder to avoid rounding issues

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create splits
        custom_train = complete_dataset.select(train_indices.tolist())
        custom_val = complete_dataset.select(val_indices.tolist())
        custom_test = complete_dataset.select(test_indices.tolist())

        logger.info(f"Custom splits created for {task_name}:")
        logger.info(f"  Train:      {len(custom_train):5d} examples ({len(custom_train)/total_size*100:.1f}%)")
        logger.info(f"  Validation: {len(custom_val):5d} examples ({len(custom_val)/total_size*100:.1f}%)")
        logger.info(f"  Test:       {len(custom_test):5d} examples ({len(custom_test)/total_size*100:.1f}%)")
        logger.info(f"  Total:      {total_size:5d} examples")

        return {
            'train': custom_train,
            'validation': custom_val,
            'test': custom_test
        }

    def _load_all_splits(self, task_name: str) -> Dict[str, Dataset]:
        """
        Load train, validation, and test splits for a task.

        Always uses original HuggingFace splits by default.
        For tasks with task_specific_splits configured, creates custom splits.

        Args:
            task_name: Name of the task

        Returns:
            Dictionary with 'train', 'validation', and 'test' keys (if available)
        """
        # Get fine-tuning config
        ft_config = self.config.get('evaluation', {}).get('benchmarks', {}) \
                        .get('indicglue', {}).get('fine_tuning', {})

        # Check for task-specific split configuration
        task_specific_splits = ft_config.get('task_specific_splits', {})

        if task_name in task_specific_splits:
            # Task requires custom split creation (e.g., CSQA with only test split)
            task_config = task_specific_splits[task_name]
            reason = task_config.get('reason', 'Task-specific split configuration')

            logger.info(f"Creating custom splits for {task_name}")
            logger.info(f"  Reason: {reason}")

            # Use task-specific custom split ratios
            return self._create_custom_splits_from_complete(task_name, override_config=task_config)

        # Default: load original splits directly from HuggingFace
        logger.info(f"Using original HuggingFace splits for {task_name}")
        splits = {}
        for split_name in ['train', 'validation', 'test']:
            try:
                dataset = self._load_task_data(task_name, split=split_name)
                if dataset is not None:
                    splits[split_name] = dataset
                    logger.info(f"Loaded {split_name} split for {task_name}: {len(dataset)} examples")
            except Exception as e:
                logger.debug(f"Could not load {split_name} split for {task_name}: {e}")

        return splits

    def fine_tune_task(self, task_name: str, train_dataset: Dataset,
                       val_dataset: Optional[Dataset] = None) -> 'torch.nn.Module':
        """
        Fine-tune classification head on train/val splits

        Features:
        - Freezes base model (only train classification head)
        - Early stopping based on validation accuracy (if validation set available)
        - If no validation set: trains for fixed num_epochs without early stopping
        - 10 epoch maximum (configurable)
        - Batch size and LR from config

        Args:
            task_name: Name of the task
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional, if None trains without validation)

        Returns:
            Fine-tuned model (best checkpoint based on validation, or final model if no validation)
        """
        # Get fine-tuning config
        ft_config = self.config.get('evaluation', {}).get('benchmarks', {}) \
                        .get('indicglue', {}).get('fine_tuning', {})

        # Explicitly convert config values to correct types to avoid type comparison errors
        num_epochs = int(ft_config.get('num_epochs', 10))
        learning_rate = float(ft_config.get('learning_rate', 2e-5))
        batch_size = int(ft_config.get('batch_size', 32))
        weight_decay = float(ft_config.get('weight_decay', 0.0))  # Official IndicBERT uses 0.0

        # Early stopping config
        es_config = ft_config.get('early_stopping', {})
        patience = int(es_config.get('patience', 3))

        # Determine if we have validation data
        has_validation = val_dataset is not None and len(val_dataset) > 0

        logger.info(f"Starting fine-tuning for {task_name}")
        if has_validation:
            logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        else:
            logger.info(f"Train samples: {len(train_dataset)}, No validation set")
            logger.info(f"Training for {num_epochs} epochs (no early stopping)")
        logger.info(f"Epochs: {num_epochs}, LR: {learning_rate}, Batch size: {batch_size}")

        # Get model with trainable head, frozen base
        model = self._get_model_for_task(task_name, for_training=True)

        # Check if this task uses binary per-candidate evaluation
        task_config = self.tasks[task_name]
        use_binary = task_config.get('use_binary_per_candidate', False)

        # Create dataloaders (use binary dataloader for WSTP)
        if use_binary:
            logger.info(f"Using binary per-candidate dataloader for {task_name} "
                       f"(1 example → {task_config.get('num_candidates', 4)} training examples)")
            train_loader = self._create_binary_candidate_dataloader(train_dataset, task_name,
                                                                     shuffle=True, batch_size=batch_size)
            val_loader = None
            if has_validation:
                val_loader = self._create_binary_candidate_dataloader(val_dataset, task_name,
                                                                       shuffle=False, batch_size=batch_size*2)
        else:
            # Standard dataloader for non-binary tasks
            train_loader = self._create_task_dataloader(train_dataset, task_name,
                                                         shuffle=True, batch_size=batch_size)
            val_loader = None
            if has_validation:
                val_loader = self._create_task_dataloader(val_dataset, task_name,
                                                           shuffle=False, batch_size=batch_size*2)

        # Setup optimizer with parameter-specific weight decay (match official IndicBERT)
        # No weight decay for bias and LayerNorm parameters
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]

        # Get adam_epsilon from config (official IndicBERT uses 1e-8)
        adam_epsilon = float(ft_config.get('adam_epsilon', 1e-8))

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_epsilon
        )

        # Training loop with early stopping (if validation available)
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        best_epoch = 0
        final_epoch = 0

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                outputs = model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
            final_epoch = epoch + 1

            if has_validation:
                # Validation phase (only if validation set available)
                val_metrics = self._validate_model(model, val_loader, task_name)
                val_acc = val_metrics['accuracy']
                val_loss = val_metrics['loss']

                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"train_loss={avg_train_loss:.4f}, "
                           f"val_acc={val_acc:.4f}, "
                           f"val_loss={val_loss:.4f}")

                # Early stopping check
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                    best_epoch = epoch + 1
                    logger.info(f"  → New best validation accuracy: {best_val_acc:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"  → No improvement (patience: {patience_counter}/{patience})")

                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                # No validation set - just log training loss
                logger.info(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}")

        # Restore best model (if validation was used) or use final model
        if has_validation and best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch} with val_acc={best_val_acc:.4f}")
        else:
            logger.info(f"Using final model from epoch {final_epoch} (no validation set)")

        # Store fine-tuning metadata for later use
        self._current_fine_tuning_info = {
            'epochs_trained': final_epoch,
            'best_epoch': best_epoch if has_validation else final_epoch,
            'best_val_accuracy': best_val_acc if has_validation else None,
            'early_stopped': has_validation and patience_counter >= patience,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset) if has_validation else 0,
            'had_validation': has_validation
        }

        # Set fine-tuned mode flag
        # This enables classification head routing for multiple-choice tasks
        self._is_finetuned_mode = True
        logger.info("Evaluation mode switched to: fine-tuned (will use classification heads for all tasks)")

        return model

    def _validate_model(self, model: 'torch.nn.Module', val_loader: DataLoader,
                        task_name: str) -> Dict[str, float]:
        """
        Run validation and return metrics

        Args:
            model: Model to validate
            val_loader: Validation data loader
            task_name: Name of the task

        Returns:
            Dictionary with 'accuracy' and 'loss'
        """
        model.eval()
        predictions = []
        labels = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
                total_loss += outputs.loss.item()
                num_batches += 1

        accuracy = accuracy_score(labels, predictions)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {'accuracy': accuracy, 'loss': avg_loss}

    def _create_task_dataloader(self, dataset: Dataset, task_name: str,
                                shuffle: bool = False,
                                batch_size: Optional[int] = None) -> DataLoader:
        """
        Create DataLoader for a task dataset

        Args:
            dataset: Dataset to create loader for
            task_name: Name of the task
            shuffle: Whether to shuffle the data
            batch_size: Batch size (defaults to self.batch_size)

        Returns:
            DataLoader
        """
        if batch_size is None:
            batch_size = self.batch_size

        task_config = self.tasks[task_name]

        def collate_fn(examples):
            """
            Collate function that handles multiple dataset field name variations

            Supports:
            - BBC/Sentiment/Discourse: 'text' or 'sentence'
            - NLI tasks: 'premise' + 'hypothesis'
            - COPA: 'premise' + 'question' + 'choice1' + 'choice2' (concatenated)
            - CSQA: 'title' + 'category' + 'question' + 'options' + 'out_of_context_options' (concatenated)
            - WSTP: 'sectionText' + 'titleA' + 'titleB' + 'titleC' + 'titleD' (concatenated)
            """
            texts = []
            labels = []

            for example in examples:
                text = None

                # Check if this is a multiple-choice task (COPA, CSQA, WSTP)
                # Multiple-choice tasks need context + all choices concatenated

                if 'premise' in example and 'choice1' in example and 'choice2' in example:
                    # COPA: concatenate premise + question + all choices
                    question = example.get('question', '')
                    choices = [example['choice1'], example['choice2']]
                    text = f"{example['premise']} [SEP] {question} [SEP] " + " [SEP] ".join(choices)

                elif 'question' in example and 'options' in example:
                    # CSQA: concatenate title + category + question + options + out_of_context_options
                    # Include all available fields for maximum context
                    title = example.get('title', '')
                    category = example.get('category', '')
                    question = example['question']
                    options = example['options']
                    out_of_context = example.get('out_of_context_options', [])

                    # Build comprehensive input
                    parts = []
                    if title:
                        parts.append(f"Title: {title}")
                    if category:
                        parts.append(f"Category: {category}")
                    parts.append(question)

                    if isinstance(options, list):
                        parts.append(" [SEP] ".join(options))

                    # Add out-of-context options if available
                    if out_of_context and isinstance(out_of_context, list) and len(out_of_context) > 0:
                        parts.append(" [SEP] ".join(out_of_context))

                    text = " [SEP] ".join(parts)

                elif 'sectionText' in example and 'titleA' in example:
                    # WSTP: concatenate sectionText + all title choices
                    titles = [
                        example.get('titleA', ''),
                        example.get('titleB', ''),
                        example.get('titleC', ''),
                        example.get('titleD', '')
                    ]
                    # Filter out empty titles
                    titles = [t for t in titles if t]
                    text = f"{example['sectionText']} [SEP] " + " [SEP] ".join(titles)

                # Standard text fields (classification and NLI tasks)
                elif 'text' in example:
                    text = example['text']
                elif 'sentence' in example:
                    text = example['sentence']
                elif 'premise' in example and 'hypothesis' in example:
                    # NLI tasks (WinogradNLI)
                    text = f"{example['premise']} [SEP] {example['hypothesis']}"

                else:
                    # Fallback: use first non-label string field
                    text_fields = [k for k in example.keys() if k != 'label' and isinstance(example[k], str)]
                    if text_fields:
                        text = example[text_fields[0]]
                    else:
                        text = ""

                texts.append(text if text is not None else "")

                # Extract label based on task-specific field names
                label = None

                # Wikipedia Section Title Prediction: uses 'correctTitle' field
                if 'correctTitle' in example:
                    # Map titleA/B/C/D to indices 0/1/2/3
                    title_to_idx = {'titleA': 0, 'titleB': 1, 'titleC': 2, 'titleD': 3}
                    correct_title = example['correctTitle']
                    label = title_to_idx.get(correct_title, 0)
                    if correct_title not in title_to_idx:
                        logger.warning(f"Unknown correctTitle value '{correct_title}' for WSTP, defaulting to 0")

                # Cloze-style QA: uses 'answer' field (need to find index in 'options')
                elif 'answer' in example and 'options' in example:
                    answer = example['answer']
                    options = example['options']
                    try:
                        label = options.index(answer)
                    except (ValueError, AttributeError):
                        logger.warning(f"Answer '{answer}' not found in options for CSQA, defaulting to 0")
                        label = 0

                # DiscourseMode: uses 'discourse_mode' field with string values
                elif 'discourse_mode' in example:
                    label = example['discourse_mode']
                    # Convert string label to integer
                    label = self._convert_discourse_mode_labels_to_int(label)

                # Standard tasks: use 'label' field
                elif 'label' in example:
                    label = example['label']
                    # Convert BBCA string labels to integers
                    if task_name == 'BBCArticlesClassification' and isinstance(label, str):
                        label = self._convert_bbca_labels_to_int(label)

                else:
                    # Fallback: no label found
                    logger.warning(f"No label field found in example for {task_name}, using 0")
                    label = 0

                labels.append(label)

            # Tokenize
            encoded = self._tokenize_batch(texts)
            encoded['labels'] = torch.tensor(labels, dtype=torch.long).to(self.device)

            return encoded

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

    def _create_binary_candidate_dataloader(self, dataset: Dataset, task_name: str,
                                            shuffle: bool = False, batch_size: int = None) -> DataLoader:
        """
        Create DataLoader for binary per-candidate evaluation (WSTP)

        This is used for tasks that need to score each candidate separately:
        - WSTP: 1 example with 4 candidates → 4 training examples with binary labels

        Args:
            dataset: Dataset to create loader for
            task_name: Name of the task
            shuffle: Whether to shuffle the data
            batch_size: Batch size (defaults to self.batch_size)

        Returns:
            DataLoader that expands examples into per-candidate format
        """
        if batch_size is None:
            batch_size = self.batch_size

        task_config = self.tasks[task_name]

        def collate_fn_binary_candidates(examples):
            """
            Collate function for binary per-candidate classification.

            For WSTP:
            - Input: 1 example with (sectionText, titleA, titleB, titleC, titleD, correctTitle)
            - Output: 4 examples, each with (sectionText, single_title) and binary label (0 or 1)

            This allows the model to learn: "Given (section, title), is this the correct title?"
            Instead of: "Given (section, all_titles), which index is correct?"
            """
            expanded_texts = []
            expanded_labels = []
            example_indices = []  # Track which original example each expanded example came from

            for batch_idx, example in enumerate(examples):
                # WSTP: expand into 4 candidate examples
                if 'sectionText' in example and 'titleA' in example:
                    section_text = example['sectionText']
                    correct_title_key = example.get('correctTitle', '').strip()

                    # Normalize correct title to match key format (e.g., "A" → "titleA")
                    if correct_title_key and not correct_title_key.startswith('title'):
                        correct_title_key = f"title{correct_title_key.upper()}"

                    # Process each of the 4 title candidates
                    for title_key in ['titleA', 'titleB', 'titleC', 'titleD']:
                        candidate_title = example.get(title_key, '')

                        if not candidate_title:
                            continue  # Skip empty candidates

                        # Format: "section [SEP] candidate"
                        text = f"{section_text} [SEP] {candidate_title}"
                        expanded_texts.append(text)

                        # Binary label: 1 if this is the correct title, 0 otherwise
                        is_correct = (title_key == correct_title_key)
                        label = 1 if is_correct else 0
                        expanded_labels.append(label)

                        # Track which original example this came from (for evaluation grouping)
                        example_indices.append(batch_idx)

                else:
                    # This shouldn't happen for WSTP, but handle gracefully
                    logger.warning(f"Binary candidate collation called on non-WSTP example: {example.keys()}")
                    # Add a dummy example to avoid breaking
                    expanded_texts.append(example.get('text', example.get('sentence', '')))
                    expanded_labels.append(0)
                    example_indices.append(batch_idx)

            # Tokenize all expanded examples
            if not expanded_texts:
                # Handle empty batch
                logger.warning("Binary candidate collation produced no examples")
                return {
                    'input_ids': torch.empty((0, 10), dtype=torch.long, device=self.device),
                    'attention_mask': torch.empty((0, 10), dtype=torch.long, device=self.device),
                    'labels': torch.empty((0,), dtype=torch.long, device=self.device),
                    'example_indices': torch.empty((0,), dtype=torch.long, device=self.device),
                }

            encoded = self._tokenize_batch(expanded_texts)
            encoded['labels'] = torch.tensor(expanded_labels, dtype=torch.long).to(self.device)
            encoded['example_indices'] = torch.tensor(example_indices, dtype=torch.long).to(self.device)

            logger.debug(f"Binary collation: {len(examples)} examples → {len(expanded_texts)} expanded examples "
                        f"(labels: {set(expanded_labels)})")

            return encoded

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_binary_candidates,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

    def _evaluate_classification(self, dataset: Dataset, task_name: str, model=None) -> Dict:
        """
        Evaluate classification task

        Args:
            dataset: Dataset to evaluate on
            task_name: Name of the task
            model: Optional model to use (if None, creates one)

        Returns:
            Dictionary with metrics
        """
        predictions = []
        labels = []

        # Get the appropriate model for this task if not provided
        if model is None:
            model = self._get_model_for_task(task_name)

        model.eval()

        # Use dataloader with proper collate function to handle different field names
        dataloader = self._create_task_dataloader(
            dataset,
            task_name,
            shuffle=False,
            batch_size=self.batch_size
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
                # Get model predictions
                outputs = model(**batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                # Get predicted classes
                # For classification models: logits shape is [batch, num_classes]
                # For language models (if still used): logits shape is [batch, seq_len, vocab_size]
                if logits.dim() == 2:
                    # Classification model: [batch, num_classes]
                    batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                elif logits.dim() == 3:
                    # Language model (fallback): [batch, seq_len, vocab_size]
                    # This should not happen if wrapping worked correctly
                    logger.warning(f"Received 3D logits for {task_name}, using fallback (last token, first N classes)")
                    num_classes = self.tasks[task_name]['num_labels']
                    last_token_logits = logits[:, -1, :num_classes]
                    batch_preds = torch.argmax(last_token_logits, dim=-1).cpu().numpy()
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")

                predictions.extend(batch_preds.tolist())
                labels.extend(batch['labels'].cpu().numpy().tolist())

        # Validate predictions and labels have same length
        assert len(predictions) == len(labels), \
            f"Prediction count mismatch: {len(predictions)} preds vs {len(labels)} labels"

        # Compute metrics
        return self._compute_classification_metrics(predictions, labels, task_name)

    def _evaluate_multiple_choice(self, dataset: Dataset, task_name: str) -> Dict:
        """
        Evaluate multiple choice task (IndicCOPA, IndicCQ, IndicWiki)

        Args:
            dataset: Dataset to evaluate on
            task_name: Name of the task

        Returns:
            Dictionary with metrics
        """
        predictions = []
        labels = []

        # Use base model for multiple choice (no classification head needed)
        self.base_model.eval()

        with torch.no_grad():
            for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
                # Extract premise/question/text based on task
                if 'premise' in example:
                    premise = example['premise']
                elif 'question' in example:
                    premise = example['question']
                elif 'sectionText' in example:
                    # IndicWiki uses sectionText
                    premise = example['sectionText']
                else:
                    premise = example.get('context', '')

                # Get choices based on task format
                choices = []
                if 'choice1' in example and 'choice2' in example:
                    # IndicCOPA format
                    choices = [example['choice1'], example['choice2']]
                elif task_name == 'Wikipedia Section Title Prediction' and 'titleA' in example:
                    # Wikipedia Section Title Prediction format: titleA, titleB, titleC, titleD
                    choices = [
                        example['titleA'],
                        example['titleB'],
                        example['titleC'],
                        example['titleD']
                    ]
                elif task_name == 'Cloze-style multiple-choice QA' and 'options' in example:
                    # Cloze-style QA format: options is a list
                    choices = example['options']
                elif 'sectionText' in example and 'titleA' in example:
                    # WSTP format: field-based fallback (works even if task name changes)
                    choices = [
                        example.get('titleA', ''),
                        example.get('titleB', ''),
                        example.get('titleC', ''),
                        example.get('titleD', '')
                    ]
                elif 'choices' in example:
                    # Generic choices field
                    choices = example['choices']
                else:
                    # Fallback if choices field is missing
                    logger.warning(f"No choices field found for {task_name}, using placeholder")
                    choices = [f"विकल्प {i}" for i in range(2)]

                # Score each choice using perplexity
                choice_scores = []
                for choice in choices:
                    # Combine premise and choice
                    # For COPA, include the question if available
                    if 'question' in example and 'premise' in example:
                        # COPA format: premise + question + choice
                        text = f"{premise} {example['question']} {choice}"
                    else:
                        text = f"{premise} {choice}"

                    # Tokenize and compute perplexity
                    try:
                        inputs = self._tokenize_batch([text])
                        outputs = self.base_model(**inputs)

                        # Compute perplexity (negative log likelihood)
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits  # [1, seq_len, vocab_size]

                            # Shift logits and labels for next-token prediction
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = inputs['input_ids'][:, 1:].contiguous()

                            # Compute cross-entropy loss (negative log likelihood)
                            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                            loss = loss_fct(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )

                            # Lower loss = higher likelihood = better choice
                            # Use negative loss as score (higher is better)
                            score = -loss.item()
                        else:
                            # Fallback if logits not available
                            score = -outputs.loss.item() if hasattr(outputs, 'loss') else 0

                        choice_scores.append(score)
                    except Exception as e:
                        logger.warning(f"Error scoring choice for {task_name}: {e}")
                        choice_scores.append(-float('inf'))  # Very low score for failed choices

                # Predict choice with highest score
                pred = np.argmax(choice_scores)
                predictions.append(pred)

                # Convert label to numeric format based on task
                if task_name == 'Wikipedia Section Title Prediction' and 'correctTitle' in example:
                    # Wikipedia Section Title Prediction: correctTitle is 'titleA', 'titleB', 'titleC', or 'titleD'
                    # Map to indices: titleA->0, titleB->1, titleC->2, titleD->3
                    label_map = {'titleA': 0, 'titleB': 1, 'titleC': 2, 'titleD': 3}
                    numeric_label = label_map.get(example['correctTitle'], 0)
                    labels.append(numeric_label)
                elif task_name == 'Cloze-style multiple-choice QA' and 'answer' in example and 'options' in example:
                    # Cloze-style QA: answer is the text, find its index in options
                    try:
                        numeric_label = example['options'].index(example['answer'])
                        labels.append(numeric_label)
                    except (ValueError, AttributeError):
                        # If answer not in options, log warning and use 0
                        logger.warning(f"Answer '{example.get('answer')}' not found in options for Cloze-style QA")
                        labels.append(0)
                elif 'correctTitle' in example:
                    # WSTP: field-based fallback (works even if task name changes)
                    label_map = {'titleA': 0, 'titleB': 1, 'titleC': 2, 'titleD': 3}
                    numeric_label = label_map.get(example['correctTitle'], 0)
                    if example['correctTitle'] not in label_map:
                        logger.warning(f"Unknown correctTitle value '{example['correctTitle']}' for WSTP")
                    labels.append(numeric_label)
                elif 'label' in example:
                    # Standard label field (IndicCOPA)
                    labels.append(example['label'])
                else:
                    # Fallback for missing labels
                    logger.warning(f"No label field found for {task_name}, using 0")
                    labels.append(0)

        # Validate predictions and labels have same length
        assert len(predictions) == len(labels), \
            f"Prediction count mismatch: {len(predictions)} preds vs {len(labels)} labels"

        # Compute metrics
        return self._compute_classification_metrics(predictions, labels, task_name)

    def _evaluate_binary_candidates(self, dataset: Dataset, task_name: str, model=None) -> Dict:
        """
        Evaluate using binary per-candidate scoring (for WSTP)

        This method processes each candidate separately with a binary classifier:
        - For each example, run N forward passes (one per candidate)
        - Get probability of "correct" (class 1) for each candidate
        - Predict the candidate with highest score

        Args:
            dataset: Dataset to evaluate on
            task_name: Name of the task
            model: Optional model to use (if None, uses current model)

        Returns:
            Dictionary with metrics
        """
        logger.info(f"Evaluating {task_name} using binary per-candidate scoring")

        predictions = []
        labels = []

        # Get the appropriate model for this task if not provided
        if model is None:
            model = self._get_model_for_task(task_name)

        model.eval()

        # Process examples one at a time (each example generates multiple candidates)
        for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
            # WSTP: Process 4 title candidates
            if 'sectionText' in example and 'titleA' in example:
                section_text = example['sectionText']
                correct_title_key = example.get('correctTitle', '').strip()

                # Normalize correct title to match key format (e.g., "A" → "titleA")
                if correct_title_key and not correct_title_key.startswith('title'):
                    correct_title_key = f"title{correct_title_key.upper()}"

                # Create inputs for all 4 candidates
                candidate_texts = []
                candidate_keys = []
                for title_key in ['titleA', 'titleB', 'titleC', 'titleD']:
                    candidate_title = example.get(title_key, '')
                    if not candidate_title:
                        continue  # Skip empty candidates

                    # Format: "section [SEP] candidate"
                    text = f"{section_text} [SEP] {candidate_title}"
                    candidate_texts.append(text)
                    candidate_keys.append(title_key)

                if not candidate_texts:
                    logger.warning(f"No valid candidates found for example")
                    predictions.append(0)
                    labels.append(0)
                    continue

                # Tokenize all candidates
                encoded = self._tokenize_batch(candidate_texts)  # Shape: [num_candidates, seq_len]
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Forward pass on all candidates
                with torch.no_grad():
                    try:
                        outputs = model(**encoded)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                        # logits shape: [num_candidates, 2] for binary classification

                        # Get probability of "correct" (class 1) for each candidate
                        probs = torch.softmax(logits, dim=1)  # Shape: [num_candidates, 2]
                        correct_probs = probs[:, 1]  # Shape: [num_candidates] - prob of class 1

                        # Predict candidate with highest score
                        predicted_idx = torch.argmax(correct_probs).item()
                        predicted_key = candidate_keys[predicted_idx]

                    except Exception as e:
                        logger.error(f"Error during binary candidate evaluation: {e}")
                        predicted_key = candidate_keys[0] if candidate_keys else 'titleA'

                # Map prediction to 0/1/2/3 index for metrics computation
                key_to_idx = {'titleA': 0, 'titleB': 1, 'titleC': 2, 'titleD': 3}
                pred_idx = key_to_idx.get(predicted_key, 0)
                true_idx = key_to_idx.get(correct_title_key, 0)

                predictions.append(pred_idx)
                labels.append(true_idx)

            else:
                # This shouldn't happen for WSTP
                logger.warning(f"Binary candidate evaluation called on non-WSTP example: {example.keys()}")
                predictions.append(0)
                labels.append(0)

        # Validate predictions and labels have same length
        assert len(predictions) == len(labels), \
            f"Prediction count mismatch: {len(predictions)} preds vs {len(labels)} labels"

        logger.info(f"Binary candidate evaluation complete: {len(predictions)} examples evaluated")

        # Compute metrics
        return self._compute_classification_metrics(predictions, labels, task_name)

    def _evaluate_nli(self, dataset: Dataset, task_name: str, model=None) -> Dict:
        """
        Evaluate Natural Language Inference task (IndicWNLI)

        Args:
            dataset: Dataset to evaluate on
            task_name: Name of the task
            model: Optional model to use (if None, creates one)

        Returns:
            Dictionary with metrics
        """
        predictions = []
        labels = []

        # Get the appropriate model for this task if not provided
        if model is None:
            model = self._get_model_for_task(task_name)

        model.eval()

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
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                # Get predicted classes
                # For classification models: logits shape is [batch, num_classes]
                # For language models (if still used): logits shape is [batch, seq_len, vocab_size]
                if logits.dim() == 2:
                    # Classification model: [batch, num_classes]
                    batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                elif logits.dim() == 3:
                    # Language model (fallback): [batch, seq_len, vocab_size]
                    # This should not happen if wrapping worked correctly
                    logger.warning(f"Received 3D logits for {task_name}, using fallback (last token, first N classes)")
                    num_classes = self.tasks[task_name]['num_labels']
                    last_token_logits = logits[:, -1, :num_classes]
                    batch_preds = torch.argmax(last_token_logits, dim=-1).cpu().numpy()
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")

                predictions.extend(batch_preds.tolist())
                labels.extend(batch_labels if isinstance(batch_labels, list) else batch_labels.tolist())

        # Validate predictions and labels have same length
        assert len(predictions) == len(labels), \
            f"Prediction count mismatch: {len(predictions)} preds vs {len(labels)} labels"

        # Compute metrics
        return self._compute_classification_metrics(predictions, labels, task_name)

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts using configured max_length.

        Uses max_length from config (default 128 per IndicBERT paper).

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
                max_length=self.max_length,  # Use config value instead of hardcoded 512
                return_tensors='pt'
            )
        except:
            # Fallback to simple encoding (also uses config max_length)
            max_len = self.max_length  # Use config value instead of hardcoded 512
            input_ids = []

            for text in texts:
                tokens = self.tokenizer.encode(text)[:max_len]
                input_ids.append(tokens)

            # Pad sequences
            max_batch_len = max(len(ids) for ids in input_ids)
            padded_ids = []
            attention_masks = []

            # Use tokenizer's pad_token_id if available, otherwise fallback to 0
            pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else 0

            for ids in input_ids:
                padding_length = max_batch_len - len(ids)
                padded_ids.append(ids + [pad_token_id] * padding_length)
                attention_masks.append([1] * len(ids) + [0] * padding_length)

            encoded = {
                'input_ids': torch.tensor(padded_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
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
        # Convert to numpy arrays and flatten to ensure 1D shape
        predictions = np.array(predictions).flatten()
        labels = np.array(labels).flatten()

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

        # Add fine-tuning metadata if available
        if hasattr(self, '_current_fine_tuning_info') and self._current_fine_tuning_info:
            results['fine_tuning_info'] = self._current_fine_tuning_info

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

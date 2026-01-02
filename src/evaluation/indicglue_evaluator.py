"""
IndicGLUE Benchmark Evaluator for Hindi Language Models

This module implements comprehensive evaluation on the IndicGLUE benchmark,
which includes multiple tasks for evaluating Indian language understanding.

IndicGLUE Tasks:
- BBCA: BBC Articles genre classification
- WSTP: Wikipedia Section title prediction
- CSQA: Cloze-style question answering
- IndicWNLI: Winograd Natural Language Inference
- COPA: Choice of Plausible Alternatives
- Sentiment Analysis: Movie and Product reviews
- Discourse Mode Classification

Reference: https://indicnlp.ai4bharat.org/indicglue/
"""

import torch
import torch.nn as nn
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

# Import refactored IndicGLUE modules
from .indicglue import (
    TaskRegistry,
    TaskDataExtractor,
    DataLoaderFactory,
    ClassificationStrategy,
    MultipleChoiceStrategy,
    PerplexityStrategy,
    FineTuningManager,
    ResultVisualizer
)

# Import classification models for wrapping language models
from ..models.classification_models import wrap_model_for_classification

logger = logging.getLogger(__name__)


class MultipleChoiceWrapper(nn.Module):
    """
    Wrapper for multiple-choice tasks (WSTP, CSQA, COPA).

    Architecture matches transformers.AutoModelForMultipleChoice:
    - Processes each choice independently with base model
    - Outputs single score per choice
    - Uses softmax over choices for prediction

    This matches the official IndicBERT implementation for multiple-choice tasks.
    """

    def __init__(self, base_model, hidden_size, num_choices, pooling_strategy='first'):
        """
        Args:
            base_model: Pre-trained language model (ALBERT/IndicBERT)
            hidden_size: Hidden dimension of base model
            num_choices: Number of choices per example (e.g., 4 for WSTP, 2 for COPA)
            pooling_strategy: 'first' (CLS token) or 'mean' (mean pooling)
        """
        super().__init__()
        self.base_model = base_model
        self.num_choices = num_choices
        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy

        # Classifier: maps pooled representation to single score per choice
        # Official implementation uses a single linear layer
        self.classifier = nn.Linear(hidden_size, 1)

        # Initialize classifier weights (match transformers initialization)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        # Match base model dtype
        base_dtype = next(base_model.parameters()).dtype
        self.classifier = self.classifier.to(dtype=base_dtype)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        Forward pass for multiple-choice classification.

        Args:
            input_ids: [batch, num_choices, seq_len]
            attention_mask: [batch, num_choices, seq_len]
            labels: [batch] - index of correct choice (0 to num_choices-1)

        Returns:
            Object with:
                logits: [batch, num_choices]
                loss: scalar (if labels provided)
        """
        batch_size, num_choices, seq_len = input_ids.shape

        # Flatten to process all choices in one forward pass
        # [batch, num_choices, seq_len] -> [batch*num_choices, seq_len]
        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)

        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )

        # Extract hidden states
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        # Shape: [batch*num_choices, seq_len, hidden_size]

        # Pool to sequence representation
        if self.pooling_strategy == 'first':
            # Use [CLS] token (position 0) - standard for BERT/ALBERT
            pooled = hidden_states[:, 0, :]  # [batch*num_choices, hidden_size]
        elif self.pooling_strategy == 'mean':
            # Mean pooling over non-padding tokens
            mask_expanded = attention_mask_flat.unsqueeze(-1).to(hidden_states.dtype)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Classify each choice to get single score
        logits = self.classifier(pooled)  # [batch*num_choices, 1]

        # Reshape to [batch, num_choices]
        logits = logits.view(batch_size, num_choices)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        # Return in a format compatible with existing code
        class MultipleChoiceOutput:
            def __init__(self, logits, loss):
                self.logits = logits
                self.loss = loss

        return MultipleChoiceOutput(logits=logits, loss=loss)


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

        # Initialize task registry (refactored from self.tasks dictionary)
        self.task_registry = TaskRegistry()
        logger.info(f"Initialized TaskRegistry with {len(self.task_registry.get_all_task_names())} tasks")

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

        # ========== Initialize Refactored Components ==========
        # These components were extracted from the monolithic evaluator for better testability

        # Data extraction component
        self.data_extractor = TaskDataExtractor(self.task_registry)
        logger.info("Initialized TaskDataExtractor")

        # DataLoader factory component
        self.dataloader_factory = DataLoaderFactory(
            task_registry=self.task_registry,
            data_extractor=self.data_extractor,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            device=self.device
        )
        logger.info("Initialized DataLoaderFactory")

        # Evaluation strategy components
        self.classification_strategy = ClassificationStrategy(
            device=self.device,
            task_config_getter=self.task_registry.get_task_config
        )
        self.mc_strategy = MultipleChoiceStrategy(device=self.device)
        self.perplexity_strategy = PerplexityStrategy(
            device=self.device,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            data_extractor=self.data_extractor
        )
        logger.info("Initialized evaluation strategies (Classification, MC, Perplexity)")

        # Fine-tuning manager component
        # Pass full config, dataloader factory, and model provider
        self.fine_tuning_manager = FineTuningManager(
            config=self.config,
            dataloader_factory=self.dataloader_factory,
            model_provider=self._get_model_for_task
        )
        logger.info("Initialized FineTuningManager (refactored with config, dataloader_factory, model_provider)")

        # Result visualizer component
        self.result_visualizer = ResultVisualizer(
            save_visualizations=self.save_visualizations,
            visualization_formats=self.visualization_format
        )
        logger.info("Initialized ResultVisualizer")

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
        hidden_size = None

        if hasattr(self.base_model, 'config'):
            config = self.base_model.config
            # Handle both dict and object config
            if isinstance(config, dict):
                hidden_size = config.get('hidden_size')
            else:
                hidden_size = getattr(config, 'hidden_size', None)
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'config'):
            config = self.base_model.model.config
            # Handle both dict and object config
            if isinstance(config, dict):
                hidden_size = config.get('hidden_size')
            else:
                hidden_size = getattr(config, 'hidden_size', None)

        if hidden_size is None:
            # Default hidden size
            hidden_size = 768
            logger.warning(f"Could not determine hidden size, using default: {hidden_size}")

        # Try to detect model type
        model_class_name = self.base_model.__class__.__name__.lower()
        if 'gpt' in model_class_name:
            model_type = 'gpt'
        elif 'albert' in model_class_name or 'bert' in model_class_name or 'deberta' in model_class_name:
            model_type = 'bert'  # BERT-style models (BERT, ALBERT, DeBERTa) use [CLS] token
            logger.info(f"Detected BERT-style model: {self.base_model.__class__.__name__} → will use [CLS] token pooling")
        else:
            model_type = 'bert'  # Default to BERT (safer than GPT for most classification tasks)
            logger.warning(f"Could not determine model type from {self.base_model.__class__.__name__}, defaulting to 'bert'")

        return {
            'hidden_size': hidden_size,
            'model_type': model_type
        }

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
        task_config = self.task_registry.get_task_config(task_name)
        model_config = self._get_model_config()

        # Check if this task uses multiple-choice wrapper
        if task_config.use_multiple_choice_wrapper:
            num_choices = task_config.num_choices or 4

            logger.info(f"Creating MultipleChoiceWrapper for '{task_name}' "
                       f"with {num_choices} choices (matches official IndicBERT)")

            wrapped_model = MultipleChoiceWrapper(
                base_model=self.base_model,
                hidden_size=model_config['hidden_size'],
                num_choices=num_choices,
                pooling_strategy='first'  # CLS token (ALBERT/BERT standard)
            )

            # Move to device
            wrapped_model = wrapped_model.to(self.device)

            # For training mode, ensure classifier has gradients enabled
            if for_training:
                for param in wrapped_model.classifier.parameters():
                    param.requires_grad = True
                num_trainable = sum(p.numel() for p in wrapped_model.classifier.parameters())
                logger.info(f"Enabled gradients for MC classifier ({num_trainable:,} trainable parameters)")

            # Cache and return
            self.wrapped_models[cache_key] = wrapped_model
            return wrapped_model

        # Standard classification/NLI tasks
        # Determine number of classes based on task type
        task_type = task_config.task_type

        if task_type == 'classification':
            # Text classification tasks (BBC, Sentiment, Discourse)
            num_classes = task_config.num_labels or task_config.num_choices or 2
            logger.info(f"Task '{task_name}' is classification with {num_classes} classes")

        elif task_type == 'nli':
            # Natural Language Inference tasks (WinogradNLI)
            num_classes = task_config.num_labels or task_config.num_choices or 3
            logger.info(f"Task '{task_name}' is NLI with {num_classes} classes")

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

        for task_name in self.task_registry.get_all_task_names():
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
        results['overall'] = self.result_visualizer.compute_overall_metrics(results, self.task_registry)

        logger.info("\n" + "="*60)
        logger.info("IndicGLUE Evaluation Complete")
        logger.info(f"Overall Accuracy: {results['overall'].get('average_accuracy', 0):.4f}")
        logger.info("="*60)

        return results

    # Tasks to skip due to dataset issues
    SKIP_TASKS = {
        'WinogradNLI': 'WNLI contains only entailment class for train and validation sets while the test set contains None',
        # CSQA removed - now handled via task_specific_splits in config with custom splits. It does not contain train/val splits.
    }

    def evaluate_task(self, task_name: str) -> Dict:
        """
        Evaluate model on a specific IndicGLUE task (with optional fine-tuning)

        Args:
            task_name: Name of the task to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Validate task name by attempting to get config
            try:
                _ = self.task_registry.get_task_config(task_name)
            except ValueError:
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

            task_config = self.task_registry.get_task_config(task_name)

            # Check if fine-tuning is enabled
            ft_config = self.config.get('evaluation', {}).get('benchmarks', {}) \
                            .get('indicglue', {}).get('fine_tuning', {})
            fine_tune_enabled = ft_config.get('enabled', False)

            # Fine-tuning applies to all task types now (classification, sentiment, and multiple-choice)
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
                    return results

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

        finally:
            # Clean up wrapped models to free GPU memory after each task
            if hasattr(self, 'wrapped_models'):
                self.wrapped_models.clear()
                logger.debug(f"Cleared wrapped model cache for {task_name}")

            # Free GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Freed GPU cache")

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

        task_config = self.task_registry.get_task_config(task_name)

        # Detect if we're in fine-tuned mode
        is_finetuned = self._is_finetuned_mode

        # Route based on task type
        # All tasks (classification, NLI, and multiple-choice) use the same unified evaluation method
        if task_config.use_multiple_choice_wrapper:
            logger.info(f"[ROUTING] {task_name}: Multiple-choice wrapper → "
                       f"Process each choice independently (matches official IndicBERT)")
        elif task_config.task_type in ['classification', 'nli']:
            logger.info(f"[ROUTING] {task_name}: Classification/NLI task → Using classification head")
        else:
            raise ValueError(f"Unknown task type: {task_config.task_type}")

        # Use unified evaluation method for all task types
        return self._evaluate_classification(dataset, task_name, model)

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
        remap_config = self.task_registry.get_split_remapping(task_name)
        if remap_config is not None:
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

    def _load_available_splits(self, task_name: str) -> Dict[str, Dataset]:
        """
        Helper method to load all available splits from HuggingFace.

        This is the core split-loading logic used by both:
        - _load_complete_dataset() (combines splits)
        - _load_all_splits() (keeps splits separate)

        Args:
            task_name: Name of the task

        Returns:
            Dictionary mapping split names to datasets (only includes available splits)
        """
        splits = {}

        for split_name in ['train', 'validation', 'test']:
            try:
                dataset = self._load_task_data(task_name, split=split_name)
                if dataset is not None and len(dataset) > 0:
                    splits[split_name] = dataset
                    logger.info(f"  Loaded {split_name}: {len(dataset)} examples")
            except Exception as e:
                logger.debug(f"  Split '{split_name}' not available: {e}")

        return splits

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

        # Load all available splits using helper method
        splits = self._load_available_splits(task_name)

        if not splits:
            logger.error(f"No data available for {task_name}")
            return None

        # Combine all available datasets
        datasets_list = list(splits.values())
        if len(datasets_list) == 1:
            combined = datasets_list[0]
        else:
            combined = concatenate_datasets(datasets_list)

        # Create info string for logging
        split_info = [f"{name}({len(ds)})" for name, ds in splits.items()]
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

        # Default: load original splits directly from HuggingFace using helper method
        logger.info(f"Using original HuggingFace splits for {task_name}")
        return self._load_available_splits(task_name)

    def fine_tune_task(self, task_name: str, train_dataset: Dataset,
                       val_dataset: Optional[Dataset] = None) -> 'torch.nn.Module':
        """
        Fine-tune classification head on train/val splits.

        This method now delegates all fine-tuning logic to FineTuningManager,
        which handles:
        - Model creation (via model_provider callback)
        - Dataloader creation (via dataloader_factory)
        - Optimizer setup with learning rate warmup
        - Training loop with early stopping
        - Best model restoration

        Args:
            task_name: Name of the task
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)

        Returns:
            Fine-tuned model (best checkpoint or final model)
        """
        # Delegate entire fine-tuning workflow to FineTuningManager
        model, metadata = self.fine_tuning_manager.fine_tune_task(
            task_name=task_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )

        # Store metadata for later use (e.g., in result visualization)
        self._current_fine_tuning_info = metadata

        # Set fine-tuned mode flag
        self._is_finetuned_mode = True
        logger.info("Evaluation mode switched to: fine-tuned (will use classification heads for all tasks)")

        return model

    def _evaluate_classification(self, dataset: Dataset, task_name: str, model=None) -> Dict:
        """
        Unified evaluation method for classification and multiple-choice tasks.

        This method handles:
        - Standard classification tasks (sentiment, NLI, etc.)
        - Multiple-choice tasks with wrapper (WSTP, CSQA, COPA)

        Features:
        - Defensive logits handling (2D and 3D fallback)
        - Validation of predictions/labels count
        - Automatic task-type routing via DataLoaderFactory
        - Uses same metrics computation for both task types
        - Prediction caching for faster repeated evaluations

        Args:
            dataset: Dataset to evaluate on
            task_name: Name of the task
            model: Optional model to use (if None, creates one)

        Returns:
            Dictionary with metrics
        """
        # Get task configuration to determine task type
        task_config = self.task_registry.get_task_config(task_name)

        # Try to load from cache
        if self.cache_manager and self.cache_manager.enable_cache:
            # Generate cache key from task name, dataset size, and config
            cache_key = self.cache_manager._compute_cache_key(
                model_hash=getattr(self.model, 'name_or_path', 'unknown_model'),
                dataset_name=task_name,
                dataset_split='test',  # IndicGLUE evaluation uses test split
                config={
                    'batch_size': self.batch_size,
                    'max_samples': self.max_samples_per_task,
                    'num_examples': len(dataset)
                }
            )

            # Try to retrieve cached predictions
            cached_result = self.cache_manager.get_cached_predictions(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached predictions for {task_name}")
                # Return cached metrics directly
                return cached_result['metadata'].get('metrics', {})

        # Log appropriate message based on task type
        if task_config.use_multiple_choice_wrapper:
            logger.info(f"Evaluating {task_name} with MultipleChoiceWrapper on {len(dataset)} examples")
        else:
            logger.info(f"Evaluating {task_name} on {len(dataset)} examples")

        # Get the appropriate model for this task if not provided
        if model is None:
            model = self._get_model_for_task(task_name, for_training=False)

        model.eval()

        # Use DataLoaderFactory for automatic task-type routing
        dataloader = self.dataloader_factory.create_dataloader(
            dataset,
            task_name,
            shuffle=False,
            batch_size=self.batch_size
        )

        predictions = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
                # Get model predictions
                outputs = model(**batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                # Get predicted classes
                # For classification/MC models: logits shape is [batch, num_classes] or [batch, num_choices]
                # For language models (fallback): logits shape is [batch, seq_len, vocab_size]
                if logits.dim() == 2:
                    # Standard case: [batch, num_classes] or [batch, num_choices]
                    batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                elif logits.dim() == 3:
                    # Language model (fallback): [batch, seq_len, vocab_size]
                    # This should not happen if wrapping worked correctly
                    logger.warning(f"Received 3D logits for {task_name}, using fallback (last token, first N classes)")

                    # Determine number of classes based on task type
                    if task_config.use_multiple_choice_wrapper:
                        num_classes = task_config.num_choices
                    else:
                        num_classes = task_config.num_labels

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
        fine_tuning_info = self._current_fine_tuning_info if hasattr(self, '_current_fine_tuning_info') and self._current_fine_tuning_info else None
        metrics = self.result_visualizer.compute_classification_metrics(
            predictions, labels, task_name,
            self.metrics_aggregator, self.task_registry, fine_tuning_info
        )

        # Save predictions and metrics to cache
        if self.cache_manager and self.cache_manager.enable_cache:
            cache_metadata = {
                'task_name': task_name,
                'model_name': getattr(self.model, 'name_or_path', 'unknown_model'),
                'num_examples': len(dataset),
                'metrics': metrics
            }
            self.cache_manager.save_predictions(
                cache_key,
                predictions={'predictions': predictions, 'labels': labels},
                metadata=cache_metadata
            )

        return metrics

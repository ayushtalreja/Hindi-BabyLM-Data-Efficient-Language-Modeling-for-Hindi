"""
IndicBERT Evaluation on IndicGLUE Tasks

This standalone script evaluates the pre-trained IndicBERT model on IndicGLUE tasks
to verify the correctness of the existing IndicGLUE evaluation implementation by
comparing results with reported scores from the IndicBERT paper.

Date: 2025-12-02

Usage:
    python scripts/evaluate_indicbert.py
    python scripts/evaluate_indicbert.py --mode zero-shot
    python scripts/evaluate_indicbert.py --tasks WSTP COPA BBCA
    python scripts/evaluate_indicbert.py --max-samples 10  # Quick smoke test
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
import json
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

# HuggingFace imports
from transformers import AutoModel, AutoTokenizer

# Import existing evaluation infrastructure
from src.evaluation.indicglue_evaluator import IndicGLUEEvaluator
from src.evaluation.metrics_utils import MetricsAggregator


# ============================================================================
# Logger Setup
# ============================================================================

def setup_logging(log_file: Path):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# ClassificationOutput Dataclass (matching existing infrastructure)
# ============================================================================

@dataclass
class ClassificationOutput:
    """Output from classification models"""
    logits: torch.Tensor  # [batch, num_classes]
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None  # [batch, seq_len, hidden_size]
    pooled_output: Optional[torch.Tensor] = None  # [batch, hidden_size]


# ============================================================================
# Component A: IndicBERT Model Loader
# ============================================================================

class IndicBERTModelLoader:
    """
    Load IndicBERT from HuggingFace and prepare for evaluation.

    IndicBERT is an ALBERT-based multilingual model for 12 Indian languages.
    This loader handles model/tokenizer loading and configuration extraction.
    """

    def __init__(self, model_name: str = 'ai4bharat/indic-bert', device: str = 'cuda'):
        """
        Load IndicBERT model and tokenizer from HuggingFace.

        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading IndicBERT model: {model_name}")
        logger.info(f"Device: {device}")

        # Load model and tokenizer
        # Use AutoModel (not AutoModelForMaskedLM) - we'll add classification heads manually
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Move to device
        self.model.to(device)

        # Extract configuration
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.model_type = 'albert'  # IndicBERT is ALBERT-based

        # Log model information
        logger.info(f"  Architecture: {self.config.architectures}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Vocab size: {self.vocab_size}")
        logger.info(f"  Num layers: {self.config.num_hidden_layers}")
        logger.info(f"  Model type: {self.model_type}")

        logger.info("Model loaded successfully!")

    def get_model(self):
        """Return the base model"""
        return self.model

    def get_tokenizer(self):
        """Return the tokenizer"""
        return self.tokenizer


# ============================================================================
# Component B: ALBERT Classification Wrapper
# ============================================================================

class ALBERTForSequenceClassification(nn.Module):
    """
    ALBERT-based sequence classification wrapper for IndicBERT.

    Architecture (per IndicBERT paper):
    1. Input (max 128 tokens) → ALBERT Base Model
    2. Extract [CLS] token from last hidden layer
    3. Linear classifier with softmax
    4. Multi-class cross-entropy loss

    This wrapper is similar to DeBERTaForSequenceClassification but adapted
    for ALBERT architecture. Uses [CLS] token pooling as specified in paper.
    """

    def __init__(self,
                 lm_model: nn.Module,
                 num_classes: int,
                 hidden_size: int,
                 dropout: float = 0.1,
                 freeze_base: bool = True,
                 pooling_strategy: str = 'first'):
        """
        Args:
            lm_model: Pre-trained ALBERT/IndicBERT model
            num_classes: Number of classification classes
            hidden_size: Hidden size of the language model
            dropout: Dropout probability for classification head
            freeze_base: If True, freeze the base model parameters (only train head)
            pooling_strategy: 'first' (CLS token) or 'mean' (mean pooling)
        """
        super().__init__()

        self.lm_model = lm_model
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy

        # Freeze base model if requested (paper uses task-specific head training)
        if freeze_base:
            for param in self.lm_model.parameters():
                param.requires_grad = False
            logger.debug(f"Base model frozen. Only training classification head.")

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classification head (standard initialization)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        # Match dtype of base model to avoid dtype mismatch errors
        self._match_base_model_dtype()

    def _match_base_model_dtype(self):
        """Match the dtype of the classification head to the base model"""
        base_dtype = next(self.lm_model.parameters()).dtype
        self.classifier = self.classifier.to(dtype=base_dtype)

    def pool_hidden_states(self, hidden_states: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool hidden states to sequence representation.

        Strategies:
        - 'first': [CLS] token (position 0) - standard for BERT/ALBERT, used in paper
        - 'mean': Mean pooling over non-padding tokens

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]

        Returns:
            pooled: [batch, hidden_size]
        """
        if self.pooling_strategy == 'first':
            # [CLS] token is at position 0
            # This is the standard for BERT/ALBERT and specified in IndicBERT paper
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == 'mean':
            # Mean pooling over non-padding tokens
            if attention_mask is not None:
                # Expand mask for broadcasting: [batch, seq_len, 1]
                mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                # Sum hidden states: [batch, hidden_size]
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                # Sum mask: [batch, 1]
                sum_mask = mask_expanded.sum(dim=1)
                # Avoid division by zero
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                # Average: [batch, hidden_size]
                return sum_hidden / sum_mask
            else:
                # No mask provided, simple mean
                return hidden_states.mean(dim=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> ClassificationOutput:
        """
        Forward pass for classification.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Classification labels [batch] (optional)

        Returns:
            ClassificationOutput with logits, loss, hidden_states
        """
        # Get base model outputs
        outputs = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract hidden states from last layer
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Pool to sequence representation using [CLS] token
        pooled_output = self.pool_hidden_states(hidden_states, attention_mask)

        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Compute loss if labels provided (multi-class cross-entropy)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        # Return in expected format
        return ClassificationOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            pooled_output=pooled_output
        )


# ============================================================================
# Component C: IndicBERT Evaluation Wrapper
# ============================================================================

class IndicBERTEvaluationWrapper:
    """
    Adapter to make IndicBERT compatible with IndicGLUEEvaluator.

    This wrapper:
    1. Presents IndicBERT as a "language model" to the evaluator
    2. Provides interface methods expected by IndicGLUEEvaluator
    3. Handles task-specific model wrapping via custom logic
    4. Caches wrapped models per task for efficiency
    """

    def __init__(self, indicbert_loader: IndicBERTModelLoader, config: Dict):
        """
        Args:
            indicbert_loader: IndicBERTModelLoader instance
            config: Configuration dictionary for evaluation
        """
        self.base_model = indicbert_loader.get_model()
        self.tokenizer = indicbert_loader.get_tokenizer()
        self.hidden_size = indicbert_loader.hidden_size
        self.device = indicbert_loader.device
        self.config = config

        # Cache for task-specific wrapped models
        self.wrapped_models = {}

        # Model info for evaluator
        self.is_language_model = True  # Tell evaluator to wrap us
        self.model_type = 'albert'

        logger.info("IndicBERT evaluation wrapper created")

    def _wrap_for_task(self, task_name: str, num_classes: int, for_training: bool = False):
        """
        Create task-specific classification wrapper.

        Called by _get_model_for_task_override during evaluation.
        Creates ALBERT-specific wrapper with proper configuration.

        Args:
            task_name: Name of the task
            num_classes: Number of classes for this task
            for_training: If True, configure for training mode

        Returns:
            Wrapped model for the task
        """
        # Create cache key
        cache_key = f"{task_name}_{'train' if for_training else 'eval'}"

        # Return cached if available
        if cache_key in self.wrapped_models:
            logger.debug(f"Using cached model for {cache_key}")
            return self.wrapped_models[cache_key]

        logger.info(f"Creating new wrapped model for task: {task_name}")
        logger.info(f"  Num classes: {num_classes}")
        logger.info(f"  Mode: {'training' if for_training else 'evaluation'}")

        # Get fine-tuning config
        ft_config = self.config.get('evaluation', {}).get('benchmarks', {}) \
                        .get('indicglue', {}).get('fine_tuning', {})

        # Configure dropout based on mode
        if for_training:
            dropout = float(ft_config.get('dropout', 0.1))
        else:
            dropout = 0.0  # No dropout during evaluation

        # Create ALBERT classification wrapper
        wrapped = ALBERTForSequenceClassification(
            lm_model=self.base_model,
            num_classes=num_classes,
            hidden_size=self.hidden_size,
            dropout=dropout,
            freeze_base=True,  # Always freeze base for task-specific fine-tuning
            pooling_strategy='first'  # [CLS] token pooling (from paper)
        )

        # Move to device
        wrapped = wrapped.to(self.device)

        # Enable gradients for head if training
        if for_training:
            for param in wrapped.classifier.parameters():
                param.requires_grad = True
            logger.debug("Classification head gradients enabled")

        # Cache the wrapped model
        self.wrapped_models[cache_key] = wrapped

        return wrapped

    # Interface methods for IndicGLUEEvaluator compatibility

    def parameters(self):
        """Return model parameters"""
        return self.base_model.parameters()

    def to(self, device):
        """Move to device"""
        self.base_model.to(device)
        self.device = device
        return self

    def eval(self):
        """Set model to evaluation mode"""
        self.base_model.eval()
        # Also set any wrapped models to eval mode
        for wrapped_model in self.wrapped_models.values():
            wrapped_model.eval()
        return self

    def train(self, mode=True):
        """Set model to training mode"""
        self.base_model.train(mode)
        # Also set any wrapped models to train mode
        for wrapped_model in self.wrapped_models.values():
            wrapped_model.train(mode)
        return self

    def __call__(self, **kwargs):
        """Forward pass for base model (used by multiple-choice tasks)"""
        return self.base_model(**kwargs)


# ============================================================================
# Component D: Configuration Builder
# ============================================================================

def create_evaluation_config(fine_tune: bool = True, **overrides) -> Dict:
    """
    Create configuration dictionary for evaluation.

    This configuration matches the structure expected by IndicGLUEEvaluator
    and incorporates IndicBERT paper specifications.

    Args:
        fine_tune: Enable fine-tuning mode
        **overrides: CLI arguments to override defaults

    Returns:
        Configuration dictionary
    """
    config = {
        # CRITICAL: Max sequence length from IndicBERT paper
        'max_length': 128,  # NOT 512!

        # Evaluation settings
        'eval_batch_size': overrides.get('batch_size', 32),
        'max_samples_per_task': overrides.get('max_samples', None),
        'eval_dropout': 0.0,  # No dropout during evaluation

        'evaluation': {
            'benchmarks': {
                'indicglue': {
                    'fine_tuning': {
                        'enabled': fine_tune,
                        'split_strategy': 'original',  # Use HuggingFace splits as-is

                        # Training hyperparameters (from ALBERT paper defaults)
                        'num_epochs': overrides.get('epochs', 10),
                        'learning_rate': overrides.get('learning_rate', 2e-5),  # ALBERT default
                        'batch_size': overrides.get('batch_size', 32),
                        'weight_decay': overrides.get('weight_decay', 0.01),  # ALBERT default
                        'dropout': 0.1,

                        # Model settings
                        'freeze_base_model': True,  # Only train classification heads

                        # Early stopping
                        'early_stopping': {
                            'patience': 3,
                            'metric': 'accuracy',
                            'mode': 'max'
                        }
                    }
                }
            },

            # Metrics settings
            'bootstrap_samples': 1000,
            'confidence_level': 0.95,
            'save_visualizations': not overrides.get('no_visualizations', False),
            'visualization_format': ['png', 'html'],
            'use_eval_cache': False  # Fresh evaluation (no caching)
        }
    }

    return config


# ============================================================================
# Component E: Evaluator Creation
# ============================================================================

def create_indicbert_evaluator(model_wrapper: IndicBERTEvaluationWrapper,
                               tokenizer,
                               config: Dict) -> IndicGLUEEvaluator:
    """
    Create IndicGLUEEvaluator with custom _get_model_for_task override.

    This allows us to inject our ALBERT-specific wrapping logic without
    modifying the core evaluator code.

    Args:
        model_wrapper: IndicBERTEvaluationWrapper instance
        tokenizer: IndicBERT tokenizer
        config: Evaluation configuration

    Returns:
        IndicGLUEEvaluator instance with custom wrapping
    """
    logger.info("Creating IndicGLUEEvaluator with ALBERT wrapping...")

    # Create standard evaluator
    evaluator = IndicGLUEEvaluator(model_wrapper, tokenizer, config)

    # Store original method
    original_get_model = evaluator._get_model_for_task

    # Override with ALBERT-aware version
    def _get_model_for_task_override(task_name, for_training=False):
        """Custom task model getter that uses our ALBERT wrapper"""
        task_config = evaluator.tasks[task_name]
        num_classes = task_config['num_labels']

        # Use our custom ALBERT wrapper
        return model_wrapper._wrap_for_task(task_name, num_classes, for_training)

    # Monkey-patch (only for this instance)
    evaluator._get_model_for_task = _get_model_for_task_override

    logger.info("Evaluator created with custom ALBERT wrapping")

    return evaluator


# ============================================================================
# Component F: Results Processing
# ============================================================================

def compute_summary_stats(results: Dict[str, Dict]) -> Dict:
    """
    Compute summary statistics across all tasks.

    Args:
        results: Dictionary of task results

    Returns:
        Summary statistics dictionary
    """
    successful_tasks = []
    accuracies = []
    f1_scores = []

    for task_name, task_result in results.items():
        status = task_result.get('status', 'unknown')

        if status not in ['skipped', 'error', 'failed']:
            successful_tasks.append(task_name)

            if 'accuracy' in task_result:
                accuracies.append(task_result['accuracy'])
            if 'f1_macro' in task_result:
                f1_scores.append(task_result['f1_macro'])

    summary = {
        'total_tasks': len(results),
        'successful_tasks': len(successful_tasks),
        'skipped_tasks': sum(1 for r in results.values() if r.get('status') == 'skipped'),
        'failed_tasks': sum(1 for r in results.values() if r.get('status') in ['error', 'failed']),
        'average_accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
        'average_f1_macro': float(np.mean(f1_scores)) if f1_scores else 0.0,
        'successful_task_list': successful_tasks
    }

    return summary


def save_results(results: Dict, summary: Dict, config: Dict,
                model_name: str, output_dir: Path):
    """
    Save results in multiple formats.

    Saves:
    1. Detailed JSON with all metrics
    2. Summary CSV for easy comparison
    3. Configuration JSON for reproducibility

    Args:
        results: Task results dictionary
        summary: Summary statistics
        config: Configuration used
        model_name: Model identifier
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to: {output_path}")

    # 1. Detailed JSON
    detailed_output = {
        'metadata': {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'mode': 'fine-tuning' if config['evaluation']['benchmarks']['indicglue']['fine_tuning']['enabled'] else 'zero-shot',
            'max_length': config.get('max_length', 128),
            'config': config
        },
        'task_results': results,
        'summary': summary
    }

    detailed_file = output_path / 'evaluation_results.json'
    with open(detailed_file, 'w') as f:
        json.dump(detailed_output, f, indent=2, default=str)
    logger.info(f"  ✓ Saved: {detailed_file}")

    # 2. Summary CSV
    csv_data = []
    for task_name, task_result in results.items():
        if task_result.get('status') not in ['skipped', 'error', 'failed']:
            csv_data.append({
                'Task': task_name,
                'Accuracy': f"{task_result.get('accuracy', 0):.4f}",
                'F1_Macro': f"{task_result.get('f1_macro', 0):.4f}",
                'F1_Weighted': f"{task_result.get('f1_weighted', 0):.4f}",
                'Precision': f"{task_result.get('precision_macro', 0):.4f}",
                'Recall': f"{task_result.get('recall_macro', 0):.4f}",
                'Num_Examples': task_result.get('num_examples', 0),
                'Status': 'completed'
            })
        else:
            csv_data.append({
                'Task': task_name,
                'Accuracy': 'N/A',
                'F1_Macro': 'N/A',
                'F1_Weighted': 'N/A',
                'Precision': 'N/A',
                'Recall': 'N/A',
                'Num_Examples': 'N/A',
                'Status': task_result.get('status', 'unknown')
            })

    csv_file = output_path / 'evaluation_summary.csv'
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    logger.info(f"  ✓ Saved: {csv_file}")

    # 3. Config YAML
    config_file = output_path / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"  ✓ Saved: {config_file}")


def print_results_table(results: Dict, summary: Dict):
    """
    Print a formatted results table to console.

    Args:
        results: Task results dictionary
        summary: Summary statistics
    """
    print("\n" + "="*80)
    print("INDICBERT EVALUATION RESULTS")
    print("="*80)

    # Print header
    print(f"\n{'Task':<40} {'Accuracy':>10} {'F1-Macro':>10} {'Examples':>10} {'Status':>10}")
    print("-"*80)

    # Print each task
    for task_name, task_result in results.items():
        status = task_result.get('status', 'unknown')

        if status not in ['skipped', 'error', 'failed']:
            acc = f"{task_result.get('accuracy', 0)*100:.2f}%"
            f1 = f"{task_result.get('f1_macro', 0)*100:.2f}%"
            n = task_result.get('num_examples', 0)
            status_icon = "✓"
        else:
            acc = "N/A"
            f1 = "N/A"
            n = "N/A"
            status_icon = "✗"

        print(f"{task_name:<40} {acc:>10} {f1:>10} {str(n):>10} {status_icon:>10}")

    # Print summary
    print("-"*80)
    print(f"\nSummary:")
    print(f"  Successful: {summary['successful_tasks']} / {summary['total_tasks']} tasks")
    print(f"  Skipped: {summary['skipped_tasks']} tasks")
    print(f"  Failed: {summary['failed_tasks']} tasks")
    print(f"  Average Accuracy: {summary['average_accuracy']*100:.2f}%")
    print(f"  Average F1-Macro: {summary['average_f1_macro']*100:.2f}%")
    print("="*80 + "\n")


# ============================================================================
# Component G: Main Evaluation Function
# ============================================================================

def evaluate_indicbert_on_indicglue(
    model_name: str = 'ai4bharat/indic-bert',
    tasks: Optional[List[str]] = None,
    fine_tune: bool = True,
    device: str = 'cuda',
    output_dir: str = 'results/indicbert_evaluation',
    **kwargs
) -> Dict:
    """
    Main evaluation function for IndicBERT on IndicGLUE tasks.

    Args:
        model_name: HuggingFace model ID
        tasks: List of task short names (e.g., ['WSTP', 'COPA'])
               None = all 7 tasks
        fine_tune: True for fine-tuning, False for zero-shot
        device: 'cuda' or 'cpu'
        output_dir: Where to save results
        **kwargs: Additional config overrides (epochs, lr, batch_size, etc.)

    Returns:
        Dictionary with evaluation results
    """

    # Step 1: Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    setup_logging(output_path / 'indicbert_evaluation.log')
    set_seed(kwargs.get('seed', 42))

    logger.info("="*80)
    logger.info("IndicBERT IndicGLUE Evaluation")
    logger.info("="*80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Mode: {'Fine-tuning' if fine_tune else 'Zero-shot'}")
    logger.info(f"Device: {device}")
    logger.info(f"Max sequence length: 128 tokens (from IndicBERT paper)")

    # Step 2: Load Model
    logger.info("\n" + "-"*80)
    logger.info("STEP 1: Loading IndicBERT model...")
    logger.info("-"*80)
    indicbert_loader = IndicBERTModelLoader(model_name, device)

    # Step 3: Create Configuration
    logger.info("\n" + "-"*80)
    logger.info("STEP 2: Creating evaluation configuration...")
    logger.info("-"*80)
    config = create_evaluation_config(
        fine_tune=fine_tune,
        **kwargs  # CLI overrides
    )
    logger.info(f"Configuration created:")
    logger.info(f"  Fine-tuning: {config['evaluation']['benchmarks']['indicglue']['fine_tuning']['enabled']}")
    logger.info(f"  Epochs: {config['evaluation']['benchmarks']['indicglue']['fine_tuning']['num_epochs']}")
    logger.info(f"  Learning rate: {config['evaluation']['benchmarks']['indicglue']['fine_tuning']['learning_rate']}")
    logger.info(f"  Batch size: {config['evaluation']['benchmarks']['indicglue']['fine_tuning']['batch_size']}")
    logger.info(f"  Max samples per task: {config.get('max_samples_per_task', 'all')}")

    # Step 4: Create Evaluation Wrapper
    logger.info("\n" + "-"*80)
    logger.info("STEP 3: Creating evaluation wrapper...")
    logger.info("-"*80)
    model_wrapper = IndicBERTEvaluationWrapper(indicbert_loader, config)

    # Step 5: Custom IndicGLUEEvaluator with IndicBERT support
    logger.info("\n" + "-"*80)
    logger.info("STEP 4: Creating IndicGLUEEvaluator...")
    logger.info("-"*80)
    evaluator = create_indicbert_evaluator(
        model_wrapper,
        indicbert_loader.get_tokenizer(),
        config
    )

    # Step 6: Determine tasks to evaluate
    if tasks is None:
        tasks = ['WSTP', 'CSQA', 'BBCA', 'iitp-mr', 'iitp-pr', 'iitp-md', 'COPA']

    task_mapping = {
        'WSTP': 'Wikipedia Section Title Prediction',
        'CSQA': 'Cloze-style multiple-choice QA',
        'BBCA': 'BBCArticlesClassification',
        'iitp-mr': 'MovieReviewSentiment',
        'iitp-pr': 'ProductReviewSentiment',
        'iitp-md': 'DiscourseMode',
        'COPA': 'Choice of Plausible Alternatives'
    }

    # Step 7: Run Evaluation
    logger.info("\n" + "="*80)
    logger.info(f"STEP 5: Evaluating on {len(tasks)} tasks...")
    logger.info("="*80)

    results = {}

    for i, task_short in enumerate(tasks, 1):
        task_name = task_mapping[task_short]

        logger.info(f"\n{'='*80}")
        logger.info(f"Task {i}/{len(tasks)}: {task_name} ({task_short})")
        logger.info(f"{'='*80}")

        try:
            task_result = evaluator.evaluate_task(task_name)
            results[task_name] = task_result

            # Log summary
            if task_result.get('status') == 'skipped':
                logger.warning(f"  Status: SKIPPED - {task_result.get('reason')}")
            elif task_result.get('status') == 'failed':
                logger.error(f"  Status: FAILED - {task_result.get('error')}")
            else:
                acc = task_result.get('accuracy', 0)
                f1 = task_result.get('f1_macro', 0)
                n = task_result.get('num_examples', 0)
                logger.info(f"  Status: SUCCESS")
                logger.info(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
                logger.info(f"  F1-Macro: {f1:.4f} ({f1*100:.2f}%)")
                logger.info(f"  Examples: {n}")

        except Exception as e:
            logger.error(f"Error evaluating {task_name}: {e}", exc_info=True)
            results[task_name] = {
                'task': task_name,
                'status': 'error',
                'error': str(e)
            }

    # Step 8: Compute Summary
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Computing summary statistics...")
    logger.info("="*80)
    summary = compute_summary_stats(results)

    # Step 9: Save Results
    logger.info("\n" + "="*80)
    logger.info("STEP 7: Saving results...")
    logger.info("="*80)
    save_results(
        results=results,
        summary=summary,
        config=config,
        model_name=model_name,
        output_dir=output_path
    )

    # Step 10: Print Summary
    print_results_table(results, summary)

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation complete! Results saved to: {output_path}")
    logger.info(f"{'='*80}")

    return {'task_results': results, 'summary': summary}


# ============================================================================
# Component H: CLI Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate IndicBERT on IndicGLUE tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all tasks with fine-tuning (default)
  python scripts/evaluate_indicbert.py

  # Zero-shot evaluation
  python scripts/evaluate_indicbert.py --mode zero-shot

  # Specific tasks only
  python scripts/evaluate_indicbert.py --tasks WSTP COPA BBCA

  # Quick test (100 samples per task)
  python scripts/evaluate_indicbert.py --max-samples 100

  # Custom hyperparameters
  python scripts/evaluate_indicbert.py --epochs 5 --learning-rate 1e-5
        """
    )

    parser.add_argument('--model-name', default='ai4bharat/indic-bert',
                       help='HuggingFace model identifier (default: ai4bharat/indic-bert)')
    parser.add_argument('--tasks', nargs='+',
                       choices=['WSTP', 'CSQA', 'BBCA', 'iitp-mr', 'iitp-pr', 'iitp-md', 'COPA'],
                       help='Tasks to evaluate (default: all 7 tasks)')
    parser.add_argument('--mode', choices=['zero-shot', 'fine-tune'],
                       default='fine-tune', help='Evaluation mode (default: fine-tune)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of fine-tuning epochs (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate for fine-tuning (default: 2e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device: cuda or cpu (default: auto-detect)')
    parser.add_argument('--output-dir', default='results/indicbert_evaluation',
                       help='Output directory (default: results/indicbert_evaluation)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples per task for quick testing (default: None = use all)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generating visualizations')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    try:
        results = evaluate_indicbert_on_indicglue(
            model_name=args.model_name,
            tasks=args.tasks,
            fine_tune=(args.mode == 'fine-tune'),
            device=args.device,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            max_samples=args.max_samples,
            no_visualizations=args.no_visualizations
        )

        return 0  # Success

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1  # Failure


if __name__ == '__main__':
    sys.exit(main())

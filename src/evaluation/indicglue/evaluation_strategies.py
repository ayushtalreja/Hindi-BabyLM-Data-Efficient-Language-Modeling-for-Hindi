"""
Evaluation strategies for different IndicGLUE task types.

Uses strategy pattern to handle different evaluation approaches.
Extracted from indicglue_evaluator.py to improve testability and maintainability.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies"""

    @abstractmethod
    def evaluate(
        self,
        model,
        dataloader_or_dataset,
        task_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on task.

        Args:
            model: Model to evaluate
            dataloader_or_dataset: DataLoader or Dataset to evaluate on
            task_name: Name of the task
            **kwargs: Additional strategy-specific arguments

        Returns:
            Dictionary with 'predictions' and 'labels' keys
        """
        pass


class ClassificationStrategy(EvaluationStrategy):
    """
    Strategy for classification and NLI tasks.

    Handles standard classification models with output shape [batch, num_classes].
    Falls back gracefully for language models with 3D logits.
    """

    def __init__(self, device: torch.device, task_config_getter=None):
        """
        Initialize ClassificationStrategy

        Args:
            device: Device to run evaluation on
            task_config_getter: Optional callable to get task config (for fallback)
        """
        self.device = device
        self.task_config_getter = task_config_getter

    def evaluate(
        self,
        model,
        dataloader: DataLoader,
        task_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate classification task.

        Args:
            model: Classification model to evaluate
            dataloader: DataLoader with batches
            task_name: Name of the task

        Returns:
            Dict with 'predictions' and 'labels' lists
        """
        model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
                # Get model predictions
                outputs = model(**batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                # Get predicted classes
                # For classification models: logits shape is [batch, num_classes]
                # For language models (fallback): logits shape is [batch, seq_len, vocab_size]
                if logits.dim() == 2:
                    # Classification model: [batch, num_classes]
                    batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                elif logits.dim() == 3:
                    # Language model (fallback): [batch, seq_len, vocab_size]
                    logger.warning(
                        f"Received 3D logits for {task_name}, using fallback "
                        f"(last token, first N classes)"
                    )

                    # Try to get num_classes from task config
                    num_classes = kwargs.get('num_classes')
                    if num_classes is None and self.task_config_getter:
                        task_config = self.task_config_getter(task_name)
                        num_classes = task_config.num_labels

                    if num_classes is None:
                        raise ValueError(
                            f"Cannot handle 3D logits without num_classes. "
                            f"Logits shape: {logits.shape}"
                        )

                    last_token_logits = logits[:, -1, :num_classes]
                    batch_preds = torch.argmax(last_token_logits, dim=-1).cpu().numpy()
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")

                predictions.extend(batch_preds.tolist())
                labels.extend(batch['labels'].cpu().numpy().tolist())

        # Validate predictions and labels have same length
        assert len(predictions) == len(labels), \
            f"Prediction count mismatch: {len(predictions)} preds vs {len(labels)} labels"

        return {
            'predictions': predictions,
            'labels': labels
        }


class MultipleChoiceStrategy(EvaluationStrategy):
    """
    Strategy for multiple-choice tasks with MultipleChoiceWrapper.

    Uses the official IndicBERT approach: process each choice independently
    and select the one with highest score.

    Expected input shape: [batch, num_choices, seq_len]
    Expected output shape: [batch, num_choices]
    """

    def __init__(self, device: torch.device):
        """
        Initialize MultipleChoiceStrategy

        Args:
            device: Device to run evaluation on
        """
        self.device = device

    def evaluate(
        self,
        model,
        dataloader: DataLoader,
        task_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate multiple-choice task.

        Args:
            model: MultipleChoiceWrapper model
            dataloader: DataLoader with MC-formatted batches
            task_name: Name of the task

        Returns:
            Dict with 'predictions' and 'labels' lists
        """
        logger.info(f"Evaluating {task_name} with MultipleChoiceWrapper")

        model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
                # Forward pass
                outputs = model(**batch)
                logits = outputs.logits  # [batch, num_choices]

                # Get predicted choice (argmax over choices)
                batch_preds = torch.argmax(logits, dim=-1)  # [batch]

                predictions.extend(batch_preds.cpu().numpy().tolist())
                labels.extend(batch['labels'].cpu().numpy().tolist())

        # Validate predictions and labels have same length
        assert len(predictions) == len(labels), \
            f"Prediction count mismatch: {len(predictions)} preds vs {len(labels)} labels"

        return {
            'predictions': predictions,
            'labels': labels
        }


class PerplexityStrategy(EvaluationStrategy):
    """
    Strategy for zero-shot multiple-choice evaluation via perplexity.

    Scores each choice using perplexity (negative log likelihood) and
    selects the choice with lowest perplexity (highest likelihood).

    This is useful for evaluating base language models without fine-tuning.
    """

    def __init__(self, device: torch.device, tokenizer, max_length: int, data_extractor):
        """
        Initialize PerplexityStrategy

        Args:
            device: Device to run evaluation on
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            data_extractor: TaskDataExtractor instance for field extraction
        """
        self.device = device
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_extractor = data_extractor

    def evaluate(
        self,
        model,
        dataset: Dataset,
        task_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate using perplexity scoring.

        Args:
            model: Base language model (not wrapped)
            dataset: Dataset to evaluate (not DataLoader - we process one at a time)
            task_name: Name of the task

        Returns:
            Dict with 'predictions' and 'labels' lists
        """
        logger.info(f"Evaluating {task_name} using perplexity scoring")

        model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
                # Extract context and choices using TaskDataExtractor
                context = self.data_extractor.extract_text(example, task_name)
                choices = self.data_extractor.extract_choices(example, task_name)
                label = self.data_extractor.extract_label(example, task_name)

                # Score each choice using perplexity
                choice_scores = []
                for choice in choices:
                    # Combine context and choice
                    # For COPA, context already includes premise + question
                    text = f"{context} {choice}"

                    # Tokenize and compute perplexity
                    try:
                        inputs = self.tokenizer(
                            text,
                            max_length=self.max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        ).to(self.device)

                        # Get model outputs
                        outputs = model(**inputs)

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
                labels.append(label)

        # Validate predictions and labels have same length
        assert len(predictions) == len(labels), \
            f"Prediction count mismatch: {len(predictions)} preds vs {len(labels)} labels"

        return {
            'predictions': predictions,
            'labels': labels
        }

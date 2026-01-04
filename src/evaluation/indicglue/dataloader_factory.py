"""
DataLoader factory for IndicGLUE tasks.

Creates task-appropriate dataloaders with proper collation.
Uses TaskDataExtractor to eliminate 30+ duplicated field-checking patterns.
"""

from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset
import logging

from .task_registry import TaskRegistry
from .data_extractor import TaskDataExtractor

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory for creating task-specific DataLoaders.

    This class handles:
    - Standard classification/NLI tasks → [batch, seq_len]
    - Multiple-choice tasks with wrapper → [batch, num_choices, seq_len]
    - Proper tokenization matching official IndicBERT

    Uses TaskDataExtractor to eliminate duplicated field extraction logic
    that was previously scattered across collate functions.
    """

    def __init__(
        self,
        task_registry: TaskRegistry,
        data_extractor: TaskDataExtractor,
        tokenizer,
        max_length: int,
        device: torch.device
    ):
        """
        Initialize DataLoaderFactory

        Args:
            task_registry: TaskRegistry instance for task configuration
            data_extractor: TaskDataExtractor instance for field extraction
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length (128 for IndicBERT)
            device: Device to move tensors to
        """
        self.task_registry = task_registry
        self.data_extractor = data_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def create_dataloader(
        self,
        dataset: Dataset,
        task_name: str,
        shuffle: bool = False,
        batch_size: int = 32
    ) -> DataLoader:
        """
        Create appropriate dataloader based on task type.

        Automatically routes to correct dataloader type based on task configuration.

        Args:
            dataset: Dataset to create loader for
            task_name: Name of the task
            shuffle: Whether to shuffle the data
            batch_size: Batch size

        Returns:
            DataLoader with appropriate collation for the task
        """
        task_config = self.task_registry.get_task_config(task_name)

        if task_config.use_multiple_choice_wrapper:
            return self.create_multiple_choice_dataloader(
                dataset, task_name, shuffle, batch_size
            )
        else:
            return self.create_standard_dataloader(
                dataset, task_name, shuffle, batch_size
            )

    def create_standard_dataloader(
        self,
        dataset: Dataset,
        task_name: str,
        shuffle: bool = False,
        batch_size: int = 32
    ) -> DataLoader:
        """
        Create dataloader for classification/NLI tasks.

        Returns batches with:
        - input_ids: [batch, seq_len]
        - attention_mask: [batch, seq_len]
        - labels: [batch]

        Args:
            dataset: Dataset to create loader for
            task_name: Name of the task
            shuffle: Whether to shuffle
            batch_size: Batch size

        Returns:
            DataLoader for classification/NLI tasks
        """

        def collate_fn(examples):
            """
            Collate function for classification/NLI tasks.

            Uses TaskDataExtractor to eliminate duplicated field-checking patterns.
            Previously, this logic was duplicated ~30+ times across the codebase.
            """
            texts = []
            labels = []

            for example in examples:
                # Use TaskDataExtractor instead of duplicated field checks
                text = self.data_extractor.extract_text(example, task_name)
                label = self.data_extractor.extract_label(example, task_name)

                texts.append(text if text is not None else "")
                labels.append(label)

            # Tokenize batch
            # NOTE: Uses truncation=True (default right-truncation) for single text classification
            # This differs from MC tasks which use truncation='longest_first' to preserve choices
            encoded = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding='max_length',  # Official IndicBERT uses max_length, not 'longest'
                truncation=True,  # Right-truncation for single sequences
                return_tensors='pt'
            )

            return {
                'input_ids': encoded['input_ids'].to(self.device),
                'attention_mask': encoded['attention_mask'].to(self.device),
                'labels': torch.tensor(labels, dtype=torch.long).to(self.device)
            }

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0  # Avoid multiprocessing issues
        )

    def create_multiple_choice_dataloader(
        self,
        dataset: Dataset,
        task_name: str,
        shuffle: bool = False,
        batch_size: int = 32
    ) -> DataLoader:
        """
        Create dataloader for multiple-choice tasks.

        Formats data as [batch, num_choices, seq_len] to match official IndicBERT.

        Returns batches with:
        - input_ids: [batch, num_choices, seq_len]
        - attention_mask: [batch, num_choices, seq_len]
        - labels: [batch] (index of correct choice)

        Args:
            dataset: Dataset to create loader for
            task_name: Name of the task (WSTP, CSQA, or COPA)
            shuffle: Whether to shuffle
            batch_size: Batch size

        Returns:
            DataLoader for multiple-choice tasks
        """
        task_config = self.task_registry.get_task_config(task_name)
        num_choices = task_config.num_choices

        def collate_fn_multiple_choice(examples):
            """
            Collate function for multiple-choice tasks.

            Uses TaskDataExtractor to eliminate duplicated field-checking patterns.
            Tokenizes each (context, choice) pair separately, matching official IndicBERT.
            """
            batch_input_ids = []
            batch_attention_masks = []
            batch_labels = []

            for example in examples:
                # Use TaskDataExtractor instead of duplicated field checks
                context = self.data_extractor.extract_text(example, task_name)
                choices = self.data_extractor.extract_choices(example, task_name)
                label = self.data_extractor.extract_label(example, task_name)

                # Batch tokenize all (context, choice) pairs at once for efficiency
                # This is ~4x faster than sequential tokenization for 4-choice tasks
                # NOTE: Uses truncation='longest_first' to preserve choice text when context is long
                # This differs from classification tasks which use truncation=True (right-truncation)
                # This is correct: MC needs to preserve choices, classification only has one sequence
                encoded = self.tokenizer(
                    [context] * len(choices),  # Repeat context for each choice
                    choices,                    # List of all choices
                    max_length=self.max_length,
                    padding='max_length',
                    truncation='longest_first',  # Truncate longer sequence (usually context) first
                    return_tensors='pt'
                )

                # encoded['input_ids'] shape: [num_choices, seq_len]
                # encoded['attention_mask'] shape: [num_choices, seq_len]
                batch_input_ids.append(encoded['input_ids'])
                batch_attention_masks.append(encoded['attention_mask'])
                batch_labels.append(label)

            # Stack across batch
            return {
                'input_ids': torch.stack(batch_input_ids).to(self.device),
                'attention_mask': torch.stack(batch_attention_masks).to(self.device),
                'labels': torch.tensor(batch_labels, dtype=torch.long).to(self.device)
            }

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_multiple_choice,
            num_workers=0
        )
"""
Curriculum Learning Scheduler

This module manages the progression of curriculum learning over training.
It controls how the difficulty threshold increases over epochs and provides
data loaders with curriculum-filtered datasets.

Scheduling Strategies:
1. Linear: Linearly increase difficulty over epochs
2. Exponential: Exponentially increase difficulty
3. Step: Step-wise increases at specific epochs
4. Root: Square root progression (fast initial, slow later)
5. Performance-based: Adapt based on validation metrics

Reference:
- Platanios et al. (2019) "Competence-based Curriculum Learning"
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import logging
from .curriculum_strategies import CurriculumStrategy, create_curriculum_strategy

logger = logging.getLogger(__name__)


class CurriculumDataset(Dataset):
    """
    Dataset wrapper that applies curriculum filtering
    """

    def __init__(self, base_dataset: Dataset, difficulty_scores: List[float],
                 difficulty_threshold: float = 1.0):
        """
        Initialize curriculum dataset

        Args:
            base_dataset: Base dataset
            difficulty_scores: Difficulty score for each example
            difficulty_threshold: Current difficulty threshold
        """
        self.base_dataset = base_dataset
        self.difficulty_scores = difficulty_scores
        self.difficulty_threshold = difficulty_threshold

        # Filter indices based on threshold
        self.valid_indices = [
            i for i, score in enumerate(difficulty_scores)
            if score <= difficulty_threshold
        ]

        logger.info(f"Curriculum dataset: {len(self.valid_indices)}/{len(base_dataset)} examples "
                   f"(threshold={difficulty_threshold:.3f})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        return self.base_dataset[original_idx]

    def update_threshold(self, new_threshold: float):
        """
        Update difficulty threshold and recompute valid indices

        Args:
            new_threshold: New difficulty threshold
        """
        self.difficulty_threshold = new_threshold
        self.valid_indices = [
            i for i, score in enumerate(self.difficulty_scores)
            if score <= new_threshold
        ]

        logger.info(f"Updated curriculum threshold to {new_threshold:.3f}: "
                   f"{len(self.valid_indices)}/{len(self.base_dataset)} examples")


class CurriculumScheduler:
    """
    Manages curriculum progression over training
    """

    def __init__(self, strategy: CurriculumStrategy,
                 schedule_type: str = 'linear',
                 config: Optional[Dict] = None):
        """
        Initialize curriculum scheduler

        Args:
            strategy: Curriculum strategy to use
            schedule_type: Type of scheduling ('linear', 'exponential', 'step', 'root', 'performance')
            config: Configuration dictionary
        """
        self.strategy = strategy
        self.schedule_type = schedule_type
        self.config = config or {}

        # Scheduling parameters
        self.start_threshold = self.config.get('start_threshold', 0.2)
        self.end_threshold = self.config.get('end_threshold', 1.0)
        self.num_epochs = self.config.get('num_epochs', 10)
        self.warmup_epochs = self.config.get('warmup_epochs', 0)

        # Current state
        self.current_epoch = 0
        self.current_threshold = self.start_threshold

        # Performance tracking for dynamic scheduling
        self.performance_history = []

        # Step schedule specific
        if schedule_type == 'step':
            self.step_epochs = self.config.get('step_epochs', [3, 6, 9])
            self.step_thresholds = self.config.get('step_thresholds', [0.4, 0.7, 1.0])

        logger.info(f"Curriculum scheduler initialized: {schedule_type} schedule, "
                   f"start={self.start_threshold:.2f}, end={self.end_threshold:.2f}")

    def get_threshold_for_epoch(self, epoch: int) -> float:
        """
        Get difficulty threshold for a specific epoch

        Args:
            epoch: Current epoch number

        Returns:
            Difficulty threshold
        """
        # Warmup period: use minimum threshold
        if epoch < self.warmup_epochs:
            return self.start_threshold

        # Adjust epoch for warmup
        effective_epoch = epoch - self.warmup_epochs
        effective_total = self.num_epochs - self.warmup_epochs

        if self.schedule_type == 'linear':
            return self._linear_schedule(effective_epoch, effective_total)

        elif self.schedule_type == 'exponential':
            return self._exponential_schedule(effective_epoch, effective_total)

        elif self.schedule_type == 'step':
            return self._step_schedule(epoch)

        elif self.schedule_type == 'root':
            return self._root_schedule(effective_epoch, effective_total)

        elif self.schedule_type == 'performance':
            # For performance-based, use current threshold (updated separately)
            return self.current_threshold

        else:
            logger.warning(f"Unknown schedule type: {self.schedule_type}, using linear")
            return self._linear_schedule(effective_epoch, effective_total)

    def _linear_schedule(self, epoch: int, total_epochs: int) -> float:
        """Linear increase from start to end threshold"""
        if total_epochs <= 0:
            return self.end_threshold

        progress = epoch / total_epochs
        threshold = self.start_threshold + progress * (self.end_threshold - self.start_threshold)
        return min(threshold, self.end_threshold)

    def _exponential_schedule(self, epoch: int, total_epochs: int) -> float:
        """Exponential increase (slow start, fast end)"""
        if total_epochs <= 0:
            return self.end_threshold

        progress = epoch / total_epochs
        # Exponential curve: exp(progress) - 1) / (e - 1)
        exp_progress = (np.exp(progress) - 1) / (np.e - 1)
        threshold = self.start_threshold + exp_progress * (self.end_threshold - self.start_threshold)
        return min(threshold, self.end_threshold)

    def _step_schedule(self, epoch: int) -> float:
        """Step-wise increases at specific epochs"""
        for step_epoch, step_threshold in zip(self.step_epochs, self.step_thresholds):
            if epoch < step_epoch:
                return step_threshold

        return self.end_threshold

    def _root_schedule(self, epoch: int, total_epochs: int) -> float:
        """Square root progression (fast initial, slow later)"""
        if total_epochs <= 0:
            return self.end_threshold

        progress = epoch / total_epochs
        # Square root curve
        root_progress = np.sqrt(progress)
        threshold = self.start_threshold + root_progress * (self.end_threshold - self.start_threshold)
        return min(threshold, self.end_threshold)

    def step(self, epoch: Optional[int] = None, performance_metric: Optional[float] = None):
        """
        Step the scheduler to next epoch

        Args:
            epoch: Explicit epoch number (if None, increments current)
            performance_metric: Performance metric for adaptive scheduling
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        # Update threshold based on schedule
        self.current_threshold = self.get_threshold_for_epoch(self.current_epoch)

        # For performance-based scheduling, adjust based on metrics
        if self.schedule_type == 'performance' and performance_metric is not None:
            self._update_performance_based_threshold(performance_metric)

        logger.info(f"Epoch {self.current_epoch}: Curriculum threshold = {self.current_threshold:.3f}")

    def _update_performance_based_threshold(self, performance_metric: float):
        """
        Update threshold based on performance

        Args:
            performance_metric: Recent validation metric (lower is better)
        """
        self.performance_history.append(performance_metric)

        # Keep only recent history
        window_size = self.config.get('performance_window', 5)
        if len(self.performance_history) > window_size:
            self.performance_history = self.performance_history[-window_size:]

        # Check if performance is improving
        if len(self.performance_history) >= 3:
            recent_avg = np.mean(self.performance_history[-2:])
            older_avg = np.mean(self.performance_history[-window_size:-2])

            adaptation_rate = self.config.get('adaptation_rate', 0.05)

            # If improving significantly, increase difficulty
            if recent_avg < older_avg * 0.95:
                self.current_threshold = min(
                    self.end_threshold,
                    self.current_threshold + adaptation_rate
                )
                logger.info(f"Performance improving, increased threshold to {self.current_threshold:.3f}")

            # If worsening, decrease difficulty
            elif recent_avg > older_avg * 1.05:
                self.current_threshold = max(
                    self.start_threshold,
                    self.current_threshold - adaptation_rate * 0.5
                )
                logger.info(f"Performance worsening, decreased threshold to {self.current_threshold:.3f}")

    def create_curriculum_dataloader(self, dataset: Dataset,
                                    batch_size: int,
                                    examples: Optional[List[Dict]] = None,
                                    **dataloader_kwargs) -> Tuple[DataLoader, CurriculumDataset]:
        """
        Create a dataloader with curriculum filtering

        Args:
            dataset: Base dataset
            batch_size: Batch size
            examples: List of example dicts for difficulty computation (if None, uses dataset)
            **dataloader_kwargs: Additional arguments for DataLoader

        Returns:
            Tuple of (DataLoader, CurriculumDataset)
        """
        # Compute difficulty scores
        if examples is None:
            # Try to extract examples from dataset
            examples = []
            for i in range(len(dataset)):
                item = dataset[i]
                if isinstance(item, dict):
                    examples.append(item)
                elif isinstance(item, (tuple, list)) and len(item) > 0:
                    examples.append({'text': str(item[0])})
                else:
                    examples.append({'text': str(item)})

        difficulty_scores = self.strategy.compute_difficulty(examples)

        # Create curriculum dataset
        curriculum_dataset = CurriculumDataset(
            dataset,
            difficulty_scores,
            self.current_threshold
        )

        # Create dataloader
        dataloader = DataLoader(
            curriculum_dataset,
            batch_size=batch_size,
            **dataloader_kwargs
        )

        return dataloader, curriculum_dataset

    def update_dataloader(self, curriculum_dataset: CurriculumDataset,
                         batch_size: int,
                         **dataloader_kwargs) -> DataLoader:
        """
        Update dataloader with new threshold

        Args:
            curriculum_dataset: Existing curriculum dataset
            batch_size: Batch size
            **dataloader_kwargs: Additional arguments for DataLoader

        Returns:
            New DataLoader with updated threshold
        """
        curriculum_dataset.update_threshold(self.current_threshold)

        dataloader = DataLoader(
            curriculum_dataset,
            batch_size=batch_size,
            **dataloader_kwargs
        )

        return dataloader

    def get_state_dict(self) -> Dict:
        """
        Get scheduler state for checkpointing

        Returns:
            State dictionary
        """
        return {
            'current_epoch': self.current_epoch,
            'current_threshold': self.current_threshold,
            'performance_history': self.performance_history,
            'schedule_type': self.schedule_type,
            'config': self.config
        }

    def load_state_dict(self, state_dict: Dict):
        """
        Load scheduler state from checkpoint

        Args:
            state_dict: State dictionary
        """
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.current_threshold = state_dict.get('current_threshold', self.start_threshold)
        self.performance_history = state_dict.get('performance_history', [])

        logger.info(f"Loaded curriculum scheduler state: epoch={self.current_epoch}, "
                   f"threshold={self.current_threshold:.3f}")


class CurriculumTrainingManager:
    """
    High-level manager for curriculum-based training

    Integrates curriculum scheduler with training loop
    """

    def __init__(self, strategy_type: str = 'combined',
                 schedule_type: str = 'linear',
                 config: Optional[Dict] = None):
        """
        Initialize curriculum training manager

        Args:
            strategy_type: Type of curriculum strategy
            schedule_type: Type of scheduling
            config: Configuration dictionary
        """
        self.config = config or {}

        # Create strategy
        self.strategy = create_curriculum_strategy(strategy_type, self.config)

        # Create scheduler
        self.scheduler = CurriculumScheduler(
            self.strategy,
            schedule_type,
            self.config
        )

        # Training state
        self.curriculum_datasets = {}

        logger.info(f"Curriculum training manager initialized: "
                   f"strategy={strategy_type}, schedule={schedule_type}")

    def prepare_epoch(self, train_dataset: Dataset,
                     batch_size: int,
                     epoch: int,
                     examples: Optional[List[Dict]] = None,
                     **dataloader_kwargs) -> DataLoader:
        """
        Prepare dataloader for an epoch

        Args:
            train_dataset: Training dataset
            batch_size: Batch size
            epoch: Current epoch
            examples: Example dicts for difficulty computation
            **dataloader_kwargs: Additional DataLoader arguments

        Returns:
            DataLoader for the epoch
        """
        # Update scheduler
        self.scheduler.step(epoch)

        # Create or update dataloader
        if 'train' in self.curriculum_datasets:
            # Update existing
            dataloader = self.scheduler.update_dataloader(
                self.curriculum_datasets['train'],
                batch_size,
                **dataloader_kwargs
            )
        else:
            # Create new
            dataloader, curriculum_dataset = self.scheduler.create_curriculum_dataloader(
                train_dataset,
                batch_size,
                examples,
                **dataloader_kwargs
            )
            self.curriculum_datasets['train'] = curriculum_dataset

        return dataloader

    def report_performance(self, performance_metric: float):
        """
        Report performance for adaptive scheduling

        Args:
            performance_metric: Validation metric
        """
        self.scheduler.step(performance_metric=performance_metric)

    def get_state_dict(self) -> Dict:
        """Get manager state for checkpointing"""
        return {
            'scheduler': self.scheduler.get_state_dict(),
            'config': self.config
        }

    def load_state_dict(self, state_dict: Dict):
        """Load manager state from checkpoint"""
        if 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])


# Convenience function
def create_curriculum_manager(config: Dict) -> Optional[CurriculumTrainingManager]:
    """
    Create curriculum manager from config

    Args:
        config: Configuration dictionary

    Returns:
        CurriculumTrainingManager or None if curriculum disabled
    """
    curriculum_config = config.get('curriculum', {})

    if not curriculum_config.get('enabled', False):
        logger.info("Curriculum learning disabled")
        return None

    strategy_type = curriculum_config.get('strategy', 'combined')
    schedule_type = curriculum_config.get('schedule', 'linear')

    manager = CurriculumTrainingManager(
        strategy_type=strategy_type,
        schedule_type=schedule_type,
        config=curriculum_config
    )

    logger.info("Curriculum learning enabled")
    return manager

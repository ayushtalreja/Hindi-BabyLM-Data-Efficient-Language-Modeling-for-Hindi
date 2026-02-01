"""
Fine-tuning manager for IndicGLUE tasks.

Handles model fine-tuning with early stopping and best model restoration.
Extracted from indicglue_evaluator.py to improve testability and maintainability.
"""

import torch
import gc
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Callable
from tqdm import tqdm
import logging
from datasets import Dataset

from .memory_utils import cleanup_cuda_memory

logger = logging.getLogger(__name__)


class FineTuningManager:
    """
    Manages fine-tuning workflow for IndicGLUE tasks.

    Features:
    - Early stopping based on validation accuracy
    - Parameter-specific weight decay (no decay for bias/LayerNorm)
    - Best model checkpoint restoration
    - Learning rate warmup scheduler
    - Reads all hyperparameters from config
    - High-level fine_tune_task() method that handles dataloader creation
    """

    def __init__(
        self,
        config: Dict,
        dataloader_factory: Any,
        model_provider: Callable[[str, bool], torch.nn.Module]
    ):
        """
        Initialize FineTuningManager.

        Args:
            config: Full evaluation config dictionary
            dataloader_factory: DataLoaderFactory instance for creating dataloaders
            model_provider: Callable that creates task-specific model
                           Signature: (task_name: str, for_training: bool) -> nn.Module
        """
        # Extract fine-tuning config
        ft_config = config.get('evaluation', {}).get('benchmarks', {}) \
                        .get('indicglue', {}).get('fine_tuning', {})

        # Training strategy
        self.freeze_base_model = bool(ft_config.get('freeze_base_model', True))
        self.use_auto_models = bool(ft_config.get('use_auto_models', True))

        # Training hyperparameters (all from config)
        self.num_epochs = int(ft_config.get('num_epochs', 10))
        self.learning_rate = float(ft_config.get('learning_rate', 2e-5))
        self.weight_decay = float(ft_config.get('weight_decay', 0.01))
        self.adam_epsilon = float(ft_config.get('adam_epsilon', 1e-8))
        self.warmup_ratio = float(ft_config.get('warmup_ratio', 0.1))
        self.batch_size = int(ft_config.get('batch_size', 32))
        self.gradient_accumulation_steps = int(ft_config.get('gradient_accumulation_steps', 1))
        self.max_grad_norm = float(ft_config.get('max_grad_norm', 1.0))
        self.dropout = float(ft_config.get('dropout', 0.1))
        self.label_smoothing = float(ft_config.get('label_smoothing', 0.0))

        # Early stopping parameters
        early_stop_config = ft_config.get('early_stopping', {})
        self.patience = int(early_stop_config.get('patience', 3))
        self.min_delta = float(early_stop_config.get('min_delta', 0.0))

        self.scheduler_type = ft_config.get('scheduler_type', 'linear')  # 'linear' or 'cosine'

        # Dependencies
        self.dataloader_factory = dataloader_factory
        self.model_provider = model_provider

        logger.info(
            f"FineTuningManager initialized: "
            f"freeze_base={self.freeze_base_model}, "
            f"use_auto_models={self.use_auto_models}, "
            f"lr={self.learning_rate}, epochs={self.num_epochs}, "
            f"batch_size={self.batch_size}, grad_accum={self.gradient_accumulation_steps}, "
            f"warmup_ratio={self.warmup_ratio}, scheduler_type={self.scheduler_type}, "
            f"weight_decay={self.weight_decay}, dropout={self.dropout}, "
            f"max_grad_norm={self.max_grad_norm}, label_smoothing={self.label_smoothing}, "
            f"patience={self.patience}, min_delta={self.min_delta}"
        )

    def fine_tune_task(
        self,
        task_name: str,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ) -> tuple[torch.nn.Module, Dict[str, Any]]:
        """
        High-level method to fine-tune a model on a task.

        This is the main entry point for fine-tuning. It handles:
        - Model creation via model_provider
        - Dataloader creation via dataloader_factory
        - Training loop delegation to _fine_tune()
        - Metadata collection

        Args:
            task_name: Name of the task
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)

        Returns:
            Tuple of (fine_tuned_model, metadata_dict)
        """
        # Determine if we have validation data
        has_validation = val_dataset is not None and len(val_dataset) > 0

        logger.info(f"Starting fine-tuning for {task_name}")
        if has_validation:
            logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        else:
            logger.info(f"Train samples: {len(train_dataset)}, No validation set")

        # Get model with trainable head, frozen base
        model = self.model_provider(task_name, for_training=True)

        # Create dataloaders using DataLoaderFactory (automatic task-type routing)
        train_loader = self.dataloader_factory.create_dataloader(
            train_dataset, task_name, shuffle=True, batch_size=self.batch_size
        )
        val_loader = None
        if has_validation:
            val_loader = self.dataloader_factory.create_dataloader(
                val_dataset, task_name, shuffle=False, batch_size=self.batch_size*2
            )

        # Delegate to internal fine-tuning method
        fine_tuning_info = self._fine_tune(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            task_name=task_name
        )

        # Add dataset metadata
        metadata = {
            **fine_tuning_info,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset) if has_validation else 0,
            'had_validation': has_validation
        }

        return model, metadata

    def _fine_tune(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        task_name: str = ""
    ) -> Dict[str, Any]:
        """
        Internal fine-tuning method (low-level, accepts dataloaders).

        This method handles the actual training loop with early stopping.
        Called by fine_tune_task() after dataloaders are created.

        Args:
            model: Model to fine-tune (in-place modification)
            train_loader: Training data loader
            val_loader: Validation data loader (optional, enables early stopping)
            task_name: Name of task (for logging)

        Returns:
            Dict with fine-tuning metadata:
                - epochs_trained: Number of epochs completed
                - best_epoch: Epoch with best validation accuracy
                - best_val_accuracy: Best validation accuracy achieved
                - early_stopped: Whether early stopping was triggered
                - final_train_loss: Training loss from last epoch
                - final_val_loss: Validation loss from last epoch (None if no validation)
                - final_val_accuracy: Validation accuracy from last epoch (None if no validation)
                - metrics_history: Dict with per-epoch lists of train_loss, val_loss,
                                   val_accuracy, and learning_rate
        """
        logger.info(f"Starting training loop for {task_name}")

        # Setup optimizer with parameter groups
        optimizer = self._create_optimizer(model)

        # Setup learning rate scheduler with warmup
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = self._create_scheduler(optimizer, warmup_steps, total_steps)

        logger.info(
            f"Training config: total_steps={total_steps}, warmup_steps={warmup_steps} "
            f"({self.warmup_ratio:.1%})"
        )

        # Training loop state
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        best_epoch = 0

        has_validation = val_loader is not None

        # Metrics history for training summary
        metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        for epoch in range(self.num_epochs):
            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, scheduler, epoch)

            # Track metrics
            current_lr = optimizer.param_groups[0]['lr']
            metrics_history['train_loss'].append(train_loss)
            metrics_history['learning_rate'].append(current_lr)

            # Update learning rate scheduler (once per epoch, not per batch)
            scheduler.step()

            if has_validation:
                # Validation phase
                val_metrics = self._validate(model, val_loader)
                val_acc = val_metrics['accuracy']
                val_loss = val_metrics['loss']

                # Store validation metrics
                metrics_history['val_loss'].append(val_loss)
                metrics_history['val_accuracy'].append(val_acc)

                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_acc={val_acc:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )

                # Early stopping check with min_delta threshold
                improvement = val_acc - best_val_acc
                if improvement > self.min_delta:
                    best_val_acc = val_acc
                    # Save best model state (on CPU to avoid GPU memory buildup)
                    # Clear previous checkpoint to avoid accumulation across tasks
                    if best_model_state is not None:
                        del best_model_state
                        gc.collect()

                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                    best_epoch = epoch + 1
                    logger.info(f"  → New best validation accuracy: {best_val_acc:.4f} (improvement: {improvement:.4f})")
                else:
                    patience_counter += 1
                    if improvement > 0:
                        logger.info(f"  → No significant improvement (improvement: {improvement:.4f} < min_delta: {self.min_delta:.4f}, patience: {patience_counter}/{self.patience})")
                    else:
                        logger.info(f"  → No improvement (patience: {patience_counter}/{self.patience})")

                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                # No validation - just log training loss
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}: train_loss={train_loss:.4f}")
                # No validation metrics available
                metrics_history['val_loss'].append(None)
                metrics_history['val_accuracy'].append(None)

        # Restore best model if validation was used
        if has_validation and best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch} with val_acc={best_val_acc:.4f}")
        else:
            logger.info(f"Using final model from epoch {epoch+1}")

        # Clean up training state to free memory
        # Delete best_model_state to free CPU memory
        if best_model_state is not None:
            del best_model_state

        # Delete optimizer and scheduler to free GPU memory used by their states
        del optimizer
        del scheduler

        # Perform robust CUDA memory cleanup
        cleanup_cuda_memory(logger)
        logger.debug("Cleaned up optimizer, scheduler, and GPU memory after fine-tuning")

        return {
            'epochs_trained': epoch + 1,
            'best_epoch': best_epoch if has_validation else epoch + 1,
            'best_val_accuracy': best_val_acc if has_validation else None,
            'early_stopped': has_validation and patience_counter >= self.patience,
            # Training summary metrics
            'final_train_loss': metrics_history['train_loss'][-1] if metrics_history['train_loss'] else None,
            'final_val_loss': metrics_history['val_loss'][-1] if has_validation else None,
            'final_val_accuracy': metrics_history['val_accuracy'][-1] if has_validation else None,
            'metrics_history': metrics_history
        }

    def _create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with parameter-specific weight decay.

        Includes validation and logging for training strategy.

        Args:
            model: Model to optimize

        Returns:
            AdamW optimizer with parameter groups
        """
        # Count total and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            f"Model parameters: {total_params:,} total, "
            f"{trainable_params:,} trainable ({trainable_params/total_params*100:.1f}%)"
        )

        # Validate frozen base configuration
        if self.freeze_base_model:
            # Count base model trainable parameters
            base_trainable = sum(
                p.numel() for n, p in model.named_parameters()
                if ('albert' in n or 'base_model' in n) and p.requires_grad
            )
            if base_trainable > 0:
                logger.warning(
                    f"⚠️  freeze_base_model=True but {base_trainable:,} base params are trainable! "
                    f"Check model wrapping logic."
                )
            else:
                logger.info(f"✓ Base model correctly frozen (0 trainable base params)")
        else:
            logger.info(f"✓ End-to-end training mode (all parameters trainable)")

        # Create parameter groups (no decay for bias/LayerNorm)
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon
        )

        logger.info(
            f"Optimizer: AdamW with {len(optimizer_grouped_parameters[0]['params'])} "
            f"params with decay, {len(optimizer_grouped_parameters[1]['params'])} without decay"
        )

        return optimizer

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int
    ) -> Any:
        """
        Create learning rate scheduler with warmup and decay.

        Supports two scheduler types:
        - 'linear': Linear warmup + linear decay (official IndicBERT)
        - 'cosine': Linear warmup + cosine decay

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total training steps

        Returns:
            Learning rate scheduler
        """
        if self.scheduler_type == 'linear':
            # Linear warmup + linear decay (official IndicBERT implementation)
            from transformers import get_linear_schedule_with_warmup

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            logger.info(
                f"Scheduler created: Linear schedule with {warmup_steps} warmup steps "
                f"(total {total_steps} steps)"
            )

        elif self.scheduler_type == 'cosine':
            # Linear warmup + cosine decay (using HuggingFace implementation)
            from transformers import get_cosine_schedule_with_warmup

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            logger.info(
                f"Scheduler created: Cosine schedule with {warmup_steps} warmup steps "
                f"(total {total_steps} steps)"
            )

        else:
            raise ValueError(
                f"Unknown scheduler_type: {self.scheduler_type}. "
                f"Must be 'linear' or 'cosine'"
            )

        return scheduler

    def _train_epoch(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int
    ) -> float:
        """
        Train for one epoch with gradient accumulation and clipping.

        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number (for logging)

        Returns:
            Average training loss for the epoch
        """
        model.train()
        train_loss = 0.0
        num_batches = 0

        # Zero gradients at start of epoch
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Scale loss by gradient accumulation steps
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Track loss (unscaled)
            train_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = train_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def _validate(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model on validation set.

        Args:
            model: Model to validate
            val_loader: Validation data loader

        Returns:
            Dict with 'accuracy' and 'loss' keys
        """
        model.eval()
        predictions = []
        labels = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                # Forward pass
                outputs = model(**batch)
                logits = outputs.logits

                # Get predictions
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy().tolist())
                labels.extend(batch['labels'].cpu().numpy().tolist())

                # Track loss
                if hasattr(outputs, 'loss'):
                    total_loss += outputs.loss.item()
                    num_batches += 1

        # Compute accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(labels, predictions)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }

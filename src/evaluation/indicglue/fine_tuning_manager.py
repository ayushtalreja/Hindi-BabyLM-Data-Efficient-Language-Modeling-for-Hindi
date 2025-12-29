"""
Fine-tuning manager for IndicGLUE tasks.

Handles model fine-tuning with early stopping and best model restoration.
Extracted from indicglue_evaluator.py to improve testability and maintainability.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class FineTuningManager:
    """
    Manages fine-tuning workflow for IndicGLUE tasks.

    Features:
    - Early stopping based on validation accuracy
    - Parameter-specific weight decay (no decay for bias/LayerNorm)
    - Best model checkpoint restoration
    - Configurable training hyperparameters
    """

    def __init__(
        self,
        num_epochs: int = 10,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.0,
        adam_epsilon: float = 1e-8,
        patience: int = 3
    ):
        """
        Initialize FineTuningManager.

        Args:
            num_epochs: Maximum number of training epochs
            learning_rate: Learning rate for AdamW optimizer
            weight_decay: Weight decay coefficient (L2 regularization)
            adam_epsilon: Epsilon for AdamW optimizer
            patience: Number of epochs to wait for improvement before early stopping
        """
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.patience = patience

    def fine_tune(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        task_name: str = ""
    ) -> Dict[str, Any]:
        """
        Fine-tune model on task with optional validation and early stopping.

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
        """
        logger.info(f"Starting fine-tuning for {task_name}")
        logger.info(
            f"Config: lr={self.learning_rate}, epochs={self.num_epochs}, "
            f"weight_decay={self.weight_decay}, patience={self.patience}"
        )

        # Setup optimizer with parameter groups
        optimizer = self._create_optimizer(model)

        # Training loop state
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        best_epoch = 0

        has_validation = val_loader is not None

        for epoch in range(self.num_epochs):
            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, epoch)

            if has_validation:
                # Validation phase
                val_metrics = self._validate(model, val_loader)
                val_acc = val_metrics['accuracy']
                val_loss = val_metrics['loss']

                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_acc={val_acc:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )

                # Early stopping check
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save best model state (on CPU to avoid GPU memory buildup)
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                    best_epoch = epoch + 1
                    logger.info(f"  → New best validation accuracy: {best_val_acc:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"  → No improvement (patience: {patience_counter}/{self.patience})")

                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                # No validation - just log training loss
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}: train_loss={train_loss:.4f}")

        # Restore best model if validation was used
        if has_validation and best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch} with val_acc={best_val_acc:.4f}")
        else:
            logger.info(f"Using final model from epoch {epoch+1}")

        return {
            'epochs_trained': epoch + 1,
            'best_epoch': best_epoch if has_validation else epoch + 1,
            'best_val_accuracy': best_val_acc if has_validation else None,
            'early_stopped': has_validation and patience_counter >= self.patience
        }

    def _create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with parameter-specific weight decay.

        No weight decay is applied to bias terms and LayerNorm parameters,
        following BERT fine-tuning best practices.

        Args:
            model: Model to optimize

        Returns:
            AdamW optimizer with parameter groups
        """
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
            f"Optimizer created: AdamW with {len(optimizer_grouped_parameters[0]['params'])} "
            f"params with decay, {len(optimizer_grouped_parameters[1]['params'])} without decay"
        )

        return optimizer

    def _train_epoch(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> float:
        """
        Train for one epoch.

        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number (for logging)

        Returns:
            Average training loss for the epoch
        """
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Track loss
            train_loss += loss.item()
            num_batches += 1

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

"""
Enhanced Training Pipeline for Hindi BabyLM

This module provides a comprehensive training framework with:
- Learning rate scheduling (warmup + cosine/linear decay)
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Checkpoint resumption
- Early stopping
- Comprehensive logging and monitoring
- Weights & Biases integration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm
import os
import json
import logging
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

from ..utils.seed_manager import SeedManager

# Import evaluation callbacks
try:
    from ..evaluation.evaluation_callbacks import (
        EvaluationCallback,
        EvaluationBasedEarlyStopping,
        CheckpointSelector,
        create_evaluation_callback
    )
    EVAL_CALLBACKS_AVAILABLE = True
except ImportError:
    EVAL_CALLBACKS_AVAILABLE = False
    logger.warning("Evaluation callbacks not available")

logger = logging.getLogger(__name__)


class HindiLanguageModelTrainer:
    """
    Advanced trainer for Hindi language models with comprehensive features

    Features:
    - Multiple LR scheduler options (linear, cosine, constant)
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation for large effective batch sizes
    - Checkpoint saving and resumption
    - Early stopping with patience
    - Comprehensive metrics tracking
    - W&B integration
    """

    def __init__(self, model, tokenizer, config):
        """
        Initialize trainer

        Args:
            model: The language model to train
            tokenizer: Tokenizer for the model
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.experiment_name = config.get('experiment_name', 'default_experiment')

        # Store vocab_size and model_type for checkpoint saving
        self.vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else None
        self.model_type = config.get('model_type', 'gpt')

        # Setup seed for reproducibility
        self.seed_manager = SeedManager(
            seed=config.get('seed', 42),
            deterministic=config.get('deterministic', True)
        )
        self.seed_manager.set_all_seeds()

        # Device setup
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        # Training configuration
        self.batch_size = config.get('batch_size', 32)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.num_epochs = config.get('num_epochs', 10)
        self.max_steps = config.get('max_steps', -1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = None  # Will be created after knowing total steps
        self.lr_scheduler_config = config.get('lr_scheduler', {})

        # Mixed precision training
        self.use_amp = config.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        # Use model_dir/checkpoints to match ModelFactory's loading path
        model_dir = config.get('model_dir', 'models')
        self.checkpoint_dir = Path(model_dir) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_steps = config.get('save_steps', 1000)
        self.save_total_limit = config.get('save_total_limit', 3)

        # Evaluation
        self.eval_steps = config.get('eval_steps', 500)

        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 3)
        self.early_stopping_threshold = config.get('early_stopping_threshold', 0.001)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Metrics tracking
        self.global_step = 0
        self.current_epoch = 0
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'learning_rate': [],
            'gradient_norm': []
        }

        # W&B initialization flag
        self.wandb_initialized = False

        # Training state
        self.training_state = {
            'epoch': 0,
            'global_step': 0,
            'best_val_loss': float('inf')
        }

        # Initialize evaluation callbacks
        self.eval_callback = None
        self.eval_early_stopping = None
        self.checkpoint_selector = None
        self._init_evaluation_callbacks()

        logger.info("Trainer initialized successfully")

    def _init_evaluation_callbacks(self):
        """Initialize evaluation callbacks if enabled"""
        if not EVAL_CALLBACKS_AVAILABLE:
            logger.info("Evaluation callbacks not available, skipping initialization")
            return

        training_config = self.config.get('training', {})

        # Initialize evaluation callback
        if training_config.get('enable_eval_callback', False):
            try:
                self.eval_callback = create_evaluation_callback(
                    self.model,
                    self.tokenizer,
                    self.config
                )
                logger.info("Evaluation callback initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize evaluation callback: {e}")
                self.eval_callback = None

        # Initialize evaluation-based early stopping
        if training_config.get('eval_early_stopping', False):
            try:
                self.eval_early_stopping = EvaluationBasedEarlyStopping(
                    metric_name=training_config.get('eval_early_stopping_metric', 'overall.average_accuracy'),
                    patience=training_config.get('eval_early_stopping_patience', 3),
                    min_delta=training_config.get('eval_early_stopping_min_delta', 0.001),
                    mode=training_config.get('eval_early_stopping_mode', 'max'),
                    verbose=True
                )
                logger.info("Evaluation-based early stopping initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize evaluation-based early stopping: {e}")
                self.eval_early_stopping = None

        # Initialize checkpoint selector
        if training_config.get('checkpoint_metric'):
            try:
                self.checkpoint_selector = CheckpointSelector(
                    checkpoint_dir=str(self.checkpoint_dir),
                    metric_name=training_config.get('checkpoint_metric', 'overall.average_accuracy'),
                    mode=training_config.get('checkpoint_metric_mode', 'max')
                )
                logger.info("Checkpoint selector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize checkpoint selector: {e}")
                self.checkpoint_selector = None

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from configuration"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw').lower()
        lr = optimizer_config.get('learning_rate', 3e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.01)

        # Parameter groups (can add custom logic here for layer-wise LR)
        params = self.model.parameters()

        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999)),
                eps=optimizer_config.get('epsilon', 1e-8),
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                params,
                lr=lr,
                betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999)),
                eps=optimizer_config.get('epsilon', 1e-8),
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        logger.info(f"Created {optimizer_type.upper()} optimizer with lr={lr}")
        return optimizer

    def _create_scheduler(self, num_training_steps: int):
        """
        Create learning rate scheduler

        Args:
            num_training_steps: Total number of training steps
        """
        scheduler_type = self.lr_scheduler_config.get('type', 'cosine_with_warmup')
        warmup_steps = self.lr_scheduler_config.get('warmup_steps', 1000)
        warmup_ratio = self.lr_scheduler_config.get('warmup_ratio', None)

        # Calculate warmup steps from ratio if provided
        if warmup_ratio is not None:
            warmup_steps = int(num_training_steps * warmup_ratio)

        logger.info(f"Creating {scheduler_type} scheduler with {warmup_steps} warmup steps")

        if scheduler_type == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'cosine' or scheduler_type == 'cosine_with_warmup':
            num_cycles = self.lr_scheduler_config.get('num_cycles', 0.5)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles
            )
        elif scheduler_type == 'constant':
            from transformers import get_constant_schedule_with_warmup
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using linear")
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch

        Args:
            dataloader: Training data loader

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids  # For language modeling
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                # Zero gradients
                self.optimizer.zero_grad()

                # Update metrics
                self.global_step += 1
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log to W&B
                if self.wandb_initialized:
                    wandb.log({
                        'train/batch_loss': loss.item() * self.gradient_accumulation_steps,
                        'train/learning_rate': current_lr,
                        'train/gradient_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        'train/global_step': self.global_step,
                        'train/epoch': self.current_epoch
                    })

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # Check for max steps
            if self.max_steps > 0 and self.global_step >= self.max_steps:
                logger.info(f"Reached max_steps={self.max_steps}, stopping training")
                break

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation/test set

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )

                loss = outputs.loss
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            'val_loss': avg_loss,
            'perplexity': perplexity
        }

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              num_epochs: Optional[int] = None):
        """
        Main training loop

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of epochs (overrides config if provided)
        """
        if num_epochs is not None:
            self.num_epochs = num_epochs

        # Calculate total training steps
        steps_per_epoch = len(train_dataloader) // self.gradient_accumulation_steps
        if self.max_steps > 0:
            total_steps = self.max_steps
        else:
            total_steps = steps_per_epoch * self.num_epochs

        # Create scheduler
        self._create_scheduler(total_steps)

        # Initialize W&B
        self._initialize_wandb()

        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")

        # Training loop
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"{'='*60}")

            # Training
            train_loss = self.train_epoch(train_dataloader)
            self.metrics_history['train_loss'].append(train_loss)

            # Validation
            val_metrics = self.evaluate(val_dataloader)
            self.metrics_history['val_loss'].append(val_metrics['val_loss'])
            self.metrics_history['perplexity'].append(val_metrics['perplexity'])

            # Log epoch metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Perplexity: {val_metrics['perplexity']:.2f}")

            if self.wandb_initialized:
                wandb.log({
                    'epoch/train_loss': train_loss,
                    'epoch/val_loss': val_metrics['val_loss'],
                    'epoch/perplexity': val_metrics['perplexity'],
                    'epoch/number': epoch
                })

            # Run evaluation callback if enabled
            eval_results = None
            if self.eval_callback is not None:
                try:
                    eval_results = self.eval_callback.on_epoch_end(
                        epoch=epoch,
                        step=self.global_step,
                        model=self.model
                    )
                except Exception as e:
                    logger.error(f"Error in evaluation callback: {e}", exc_info=True)

            # Check for improvement
            improved = val_metrics['val_loss'] < (self.best_val_loss - self.early_stopping_threshold)

            if improved:
                logger.info(f"✓ Validation loss improved from {self.best_val_loss:.4f} to {val_metrics['val_loss']:.4f}")
                self.best_val_loss = val_metrics['val_loss']
                self.epochs_without_improvement = 0

                # Save best model
                checkpoint_path = self.save_checkpoint(epoch, val_metrics, is_best=True)

                # Register with checkpoint selector
                if self.checkpoint_selector is not None and eval_results is not None:
                    self.checkpoint_selector.register_checkpoint(checkpoint_path, eval_results)

            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement for {self.epochs_without_improvement} epoch(s)")

                # Save regular checkpoint
                checkpoint_path = self.save_checkpoint(epoch, val_metrics, is_best=False)

                # Register with checkpoint selector
                if self.checkpoint_selector is not None and eval_results is not None:
                    self.checkpoint_selector.register_checkpoint(checkpoint_path, eval_results)

            # Check evaluation-based early stopping
            should_stop_eval = False
            if self.eval_early_stopping is not None and eval_results is not None:
                try:
                    should_stop_eval = self.eval_early_stopping(eval_results)
                except Exception as e:
                    logger.error(f"Error in evaluation-based early stopping: {e}")

            # Early stopping check (validation loss)
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered (validation loss) after {epoch + 1} epochs")
                break

            # Early stopping check (evaluation metric)
            if should_stop_eval:
                logger.info(f"Early stopping triggered (evaluation metric) after {epoch + 1} epochs")
                break

            # Check max steps
            if self.max_steps > 0 and self.global_step >= self.max_steps:
                break

        logger.info("\n" + "="*60)
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        # Load best checkpoint if requested
        if self.config.get('training', {}).get('load_best_checkpoint_at_end', False):
            if self.checkpoint_selector is not None:
                best_checkpoint = self.checkpoint_selector.get_best_checkpoint()
                if best_checkpoint:
                    logger.info(f"Loading best checkpoint: {best_checkpoint}")
                    try:
                        self.load_checkpoint(best_checkpoint)
                        logger.info("Best checkpoint loaded successfully")
                    except Exception as e:
                        logger.error(f"Error loading best checkpoint: {e}")
                else:
                    logger.warning("No best checkpoint available")
            else:
                # Fall back to loading {experiment_name}_best.pt if it exists
                best_checkpoint = self.checkpoint_dir / f'{self.experiment_name}_best.pt'
                if best_checkpoint.exists():
                    logger.info(f"Loading best checkpoint: {best_checkpoint}")
                    try:
                        self.load_checkpoint(str(best_checkpoint))
                        logger.info("Best checkpoint loaded successfully")
                    except Exception as e:
                        logger.error(f"Error loading best checkpoint: {e}")

        logger.info("="*60)

        # Save final model
        self.save_checkpoint(self.current_epoch, val_metrics, is_final=True)

        # Finish W&B run
        if self.wandb_initialized:
            wandb.finish()

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float],
                       is_best: bool = False, is_final: bool = False) -> str:
        """
        Save training checkpoint

        Args:
            epoch: Current epoch number
            metrics: Current metrics
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'metrics_history': self.metrics_history,
            # Add vocab_size and model_type for ModelFactory compatibility
            'vocab_size': self.vocab_size,
            'model_type': self.model_type,
            'experiment_name': self.experiment_name
        }

        # Determine checkpoint name (include experiment_name to match ModelFactory)
        if is_final:
            checkpoint_name = f'{self.experiment_name}_final.pt'
        elif is_best:
            checkpoint_name = f'{self.experiment_name}_best.pt'
        else:
            checkpoint_name = f'{self.experiment_name}_epoch_{epoch}.pt'

        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Cleanup old checkpoints (keep only last N)
        if not is_best and not is_final:
            self._cleanup_checkpoints()

        return str(checkpoint_path)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the most recent N"""
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob(f'{self.experiment_name}_epoch_*.pt')],
            key=lambda x: x.stat().st_mtime
        )

        # Remove oldest checkpoints
        while len(checkpoints) > self.save_total_limit:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.debug(f"Removed old checkpoint: {oldest}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint and resume training

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.metrics_history = checkpoint.get('metrics_history', self.metrics_history)

        logger.info(f"Checkpoint loaded - Epoch: {self.current_epoch}, Step: {self.global_step}")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")

    def _initialize_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb_config = self.config.get('wandb', {})

        if not wandb_config.get('enabled', False):
            logger.info("W&B logging disabled")
            return

        try:
            wandb.init(
                project=wandb_config.get('project', 'hindi-babylm'),
                entity=wandb_config.get('entity'),
                name=self.config.get('experiment_name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                config=self.config,
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes'),
                mode=wandb_config.get('mode', 'online')
            )

            # Watch model (optional)
            if wandb_config.get('watch_model', False):
                wandb.watch(self.model, log='all', log_freq=100)

            self.wandb_initialized = True
            logger.info("✓ W&B initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.wandb_initialized = False

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training run"""
        return {
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.metrics_history['train_loss'][-1] if self.metrics_history['train_loss'] else None,
            'final_val_loss': self.metrics_history['val_loss'][-1] if self.metrics_history['val_loss'] else None,
            'final_perplexity': self.metrics_history['perplexity'][-1] if self.metrics_history['perplexity'] else None,
            'metrics_history': self.metrics_history
        }

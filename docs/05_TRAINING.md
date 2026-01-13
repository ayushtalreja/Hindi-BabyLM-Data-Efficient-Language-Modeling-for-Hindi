# Training Pipeline

## Overview

The training pipeline manages the complete model training process, including optimization, learning rate scheduling, checkpointing, monitoring, and evaluation. The pipeline is designed for efficient training with limited data (~10-100M words).

## Architecture

```
DataLoader → Enhanced Trainer → Model
              ↓
         Optimizer (AdamW/Adam/SGD)
              ↓
         LR Scheduler (Cosine/Linear/Constant with Warmup)
              ↓
         Mixed Precision (FP16/BF16 with Auto-casting)
              ↓
         Gradient Accumulation
              ↓
         Gradient Clipping
              ↓
         Logging (W&B with nested metrics)
              ↓
         Checkpointing (Full State + Metadata)
              ↓
         Early Stopping (Validation + Evaluation metrics)
              ↓
         Evaluation (Perplexity + Optional IndicGLUE and multiblimp callbacks)
```

## Data Loading

### HindiLanguageModelingDataset

**Location**: `src/training/data_loader.py:6-28`

**Purpose**: Simple PyTorch Dataset for language modeling with tokenization.

**Implementation**:
```python
class HindiLanguageModelingDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
```

**Features**:
- Handles text tokenization on-the-fly
- Fixed-length sequences with padding
- Configurable `max_length` (default: 512)
- Returns `input_ids` and `attention_mask`
- Compatible with PyTorch DataLoader

**Usage**:
```python
from torch.utils.data import DataLoader
from src.training.data_loader import HindiLanguageModelingDataset

# Create dataset
dataset = HindiLanguageModelingDataset(
    texts=train_texts,
    tokenizer=tokenizer,
    max_length=512
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

**Note**: The actual data preparation and splitting happens in `src/data_processing/corpus_builder.py`. This dataset class only handles the tokenization and batch preparation.

---

## HindiLanguageModelTrainer

**Location**: `src/training/trainer.py:46-749`

**Purpose**: Advanced trainer with comprehensive features for efficient training in data-limited regimes.

### Enhanced Features

The `HindiLanguageModelTrainer` extends basic training with:
- ✅ **Multiple optimizer options** (AdamW, Adam, SGD)
- ✅ **Multiple LR schedulers** (Linear, Cosine, Constant with warmup)
- ✅ **Mixed precision training** (FP16/BF16) with configurable dtype
- ✅ **Gradient accumulation** for larger effective batch sizes
- ✅ **Full checkpoint management** (model, optimizer, scheduler, scaler states)
- ✅ **Early stopping** with configurable patience and threshold
- ✅ **Comprehensive logging** (W&B integration with nested config support)
- ✅ **Gradient norm tracking** for training stability
- ✅ **Deterministic training** with seed management
- ✅ **Checkpoint resumption** with complete state restoration
- ✅ **Evaluation callbacks** (optional) for training-time evaluation
- ✅ **Checkpoint selector** for metric-based best model selection

### Initialization

```python
class HindiLanguageModelTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.experiment_name = config.get('experiment_name', 'default_experiment')

        # Setup seed for reproducibility
        self.seed_manager = SeedManager(
            seed=config.get('seed', 42),
            deterministic=config.get('deterministic', True)
        )
        self.seed_manager.set_all_seeds()

        # Device setup with mixed precision dtype
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Mixed precision dtype configuration (bf16 by default on CUDA)
        if self.device.type == 'cuda':
            dtype_str = config.get('mixed_precision_dtype', 'bf16')
            dtype_map = {
                'fp16': torch.float16,
                'bf16': torch.bfloat16,
                'float16': torch.float16,
                'float32': torch.float32
            }
            target_dtype = dtype_map.get(dtype_str, torch.bfloat16)
            self.model = self.model.to(dtype=target_dtype)

        self.model.to(self.device)

        # Training configuration
        self.batch_size = config.get('batch_size', 32)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.num_epochs = config.get('num_epochs', 10)
        self.max_steps = config.get('max_steps', -1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # Optimizer (created via _create_optimizer method)
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler (created after knowing total steps)
        self.scheduler = None
        self.lr_scheduler_config = config.get('lr_scheduler', {})

        # Mixed precision training with GradScaler
        self.use_amp = config.get('use_amp', False)
        self.scaler = grad_scaler() if self.use_amp else None

        # Checkpointing (saves to results/{experiment_name}/models/)
        results_dir = config.get('results_dir', 'results')
        self.checkpoint_dir = Path(results_dir) / self.experiment_name / 'models'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 3)
        self.early_stopping_threshold = config.get('early_stopping_threshold', 0.001)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'learning_rate': [],
            'gradient_norm': []
        }

        # Initialize evaluation callbacks (optional)
        self._init_evaluation_callbacks()
```

### Core Methods

#### `train(train_dataloader, val_dataloader, num_epochs)` (line 456)

**Purpose**: Main training loop with comprehensive features

**Process**:
1. Calculate total training steps and create LR scheduler
2. Initialize Weights & Biases (if enabled)
3. For each epoch:
   - Train on training set with gradient accumulation
   - Evaluate on validation set
   - Run evaluation callbacks (if enabled)
   - Check for improvement and early stopping
   - Save checkpoint if best
   - Log all metrics to W&B
4. Optionally load best checkpoint at end
5. Return training summary

**Key Features**:
- Automatic scheduler creation based on total steps
- Evaluation callbacks for training-time benchmark evaluation
- Dual early stopping (validation loss + evaluation metric)
- Checkpoint selection based on evaluation metrics
- Comprehensive logging with nested metric groups

```python
def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
          num_epochs: Optional[int] = None):
    """Main training loop"""
    if num_epochs is not None:
        self.num_epochs = num_epochs

    # Calculate total training steps
    steps_per_epoch = len(train_dataloader) // self.gradient_accumulation_steps
    total_steps = self.max_steps if self.max_steps > 0 else steps_per_epoch * self.num_epochs

    # Create scheduler
    self._create_scheduler(total_steps)

    # Initialize W&B
    self._initialize_wandb()

    # Training loop
    for epoch in range(self.num_epochs):
        self.current_epoch = epoch

        # Training
        train_loss = self.train_epoch(train_dataloader)
        self.metrics_history['train_loss'].append(train_loss)

        # Validation
        val_metrics = self.evaluate(val_dataloader)
        self.metrics_history['val_loss'].append(val_metrics['val_loss'])
        self.metrics_history['perplexity'].append(val_metrics['perplexity'])

        # Log to W&B
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
            eval_results = self.eval_callback.on_epoch_end(
                epoch=epoch,
                step=self.global_step,
                model=self.model
            )

        # Check for improvement
        improved = val_metrics['val_loss'] < (self.best_val_loss - self.early_stopping_threshold)

        if improved:
            self.best_val_loss = val_metrics['val_loss']
            self.epochs_without_improvement = 0
            checkpoint_path = self.save_checkpoint(epoch, val_metrics, is_best=True)

            # Register with checkpoint selector
            if self.checkpoint_selector is not None and eval_results is not None:
                self.checkpoint_selector.register_checkpoint(checkpoint_path, eval_results)
        else:
            self.epochs_without_improvement += 1

        # Early stopping checks (validation loss + evaluation metric)
        if self.epochs_without_improvement >= self.early_stopping_patience:
            break
        if self.eval_early_stopping is not None and eval_results is not None:
            if self.eval_early_stopping(eval_results):
                break

    # Load best checkpoint if requested
    if self.config.get('training', {}).get('load_best_checkpoint_at_end', False):
        best_checkpoint = self.checkpoint_dir / 'best.pt'
        if best_checkpoint.exists():
            self.load_checkpoint(str(best_checkpoint))

    # Finish W&B run
    if self.wandb_initialized:
        wandb.finish()
```

#### `train_epoch(dataloader)` (line 306)

**Purpose**: Train for one epoch with gradient accumulation and mixed precision

**Process**:
1. Set model to training mode
2. For each batch:
   - Move to device
   - Forward pass (with mixed precision if enabled)
   - Compute loss (scaled by accumulation steps)
   - Backward pass (with gradient scaling if using AMP)
   - Accumulate gradients
   - Every N batches:
     * Clip gradients
     * Optimizer step (with scaler if using AMP)
     * Scheduler step
     * Zero gradients
     * Log to Wandb
   - Check for max_steps
3. Return average loss

```python
def train_epoch(self, dataloader: DataLoader) -> float:
    """Train for one epoch"""
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
            if self.wandb_initialized and self.global_step % self.log_steps == 0:
                wandb.log({
                    'train/batch_loss': loss.item() * self.gradient_accumulation_steps,
                    'train/learning_rate': current_lr,
                    'train/gradient_norm': grad_norm.item(),
                    'train/global_step': self.global_step
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
            break

    return total_loss / num_batches
```

**Key Components**:
- **Mixed Precision**: Optional FP16/BF16 training with autocast and GradScaler
- **Gradient Accumulation**: Effective batch size = batch_size × gradient_accumulation_steps
- **Gradient Clipping**: Prevents exploding gradients (configurable max_norm)
- **LR Scheduling**: Per-step updates (not per-epoch)
- **Progress Bar**: Shows loss and learning rate in real-time
- **Max Steps**: Optional early stopping based on total steps

#### `evaluate(dataloader)` (line 410)

**Purpose**: Evaluate model on validation/test set with mixed precision support

**Returns**: Dictionary with `val_loss` and `perplexity`

```python
def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
    """Evaluate model on validation/test set"""
    self.model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Forward pass with mixed precision if enabled
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
```

**Perplexity**:
```
Perplexity = exp(loss)
```
- Lower is better
- Measures how "surprised" model is by the data
- Model properly accumulates loss by batch size (not tokens)

#### `save_checkpoint(epoch, metrics, is_best, is_final)` (line 605)

**Purpose**: Save comprehensive training checkpoint with full state

**Arguments**:
- `epoch`: Current epoch number
- `metrics`: Dictionary of current metrics
- `is_best`: Whether this is the best model so far
- `is_final`: Whether this is the final checkpoint

**Returns**: Path to saved checkpoint

```python
def save_checkpoint(self, epoch: int, metrics: Dict[str, float],
                   is_best: bool = False, is_final: bool = False) -> str:
    """Save training checkpoint"""
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

    # Determine checkpoint name
    if is_final:
        checkpoint_name = 'final.pt'
    elif is_best:
        checkpoint_name = 'best.pt'
    else:
        checkpoint_name = f'epoch_{epoch}.pt'

    checkpoint_path = self.checkpoint_dir / checkpoint_name
    torch.save(checkpoint, checkpoint_path)

    # Cleanup old checkpoints (keep only last N)
    if not is_best and not is_final:
        self._cleanup_checkpoints()

    return str(checkpoint_path)
```

**Checkpoint Contents**:
- Model state (weights and biases)
- Optimizer state (momentum, etc.)
- LR scheduler state (step count, schedule)
- Mixed precision scaler state (if using AMP)
- Training metrics and history
- Full configuration
- Experiment metadata (vocab_size, model_type, name)

## Training Configuration

### Optimizer Options

The trainer supports three optimizer types (`trainer.py:223-260`):

#### 1. AdamW (Recommended)

**Configuration** (from `configs/base_config.yaml:195-202`):
```yaml
optimizer:
  type: "adamw"
  learning_rate: 0.0005  # 5e-4, increased from 3e-4 for faster convergence
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  epsilon: 1.0e-8
  amsgrad: false
```

**Why AdamW?**
- ✅ Adaptive learning rates per parameter
- ✅ Effective for transformers
- ✅ Decoupled weight decay (better regularization than Adam)
- ✅ Industry standard for LLM training
- ✅ Well-tested on GPT and similar architectures

#### 2. Adam

**Configuration**:
```yaml
optimizer:
  type: "adam"
  learning_rate: 1e-3
  beta1: 0.9
  beta2: 0.999
  epsilon: 1e-8
  weight_decay: 0.01
```

**When to use**: Legacy support, generally AdamW is preferred.

#### 3. SGD with Momentum

**Configuration**:
```yaml
optimizer:
  type: "sgd"
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.01
```

**When to use**:
- Specific research requirements
- Comparison baseline
- Generally slower convergence than Adam/AdamW

---

### Learning Rate Schedulers

The trainer supports three LR scheduler types (`trainer.py:262-304`):

#### 1. Cosine with Warmup (Default)

**Description**: Linear warmup followed by cosine annealing (half cycle by default).

**Configuration** (from `configs/base_config.yaml:205-210`):
```yaml
lr_scheduler:
  type: "cosine_with_warmup"  # Or just "cosine"
  warmup_steps: 1500           # Increased from 1000 for more stable warmup
  warmup_ratio: 0.1            # Alternative: ratio of total steps (10%)
  num_cycles: 0.5              # Half cosine cycle (recommended)
  min_lr_ratio: 0.05           # Don't decay to zero (prevents "dead" final epochs)
```

**Schedule**:
```
LR │        /────────\
   │       /          \
   │      /            \
   │     /              \
   │____/                \____
   └────────────────────────────→ Steps
        ↑                ↑
     Warmup          Decay End
```

**Use Case**:
- **Recommended for this project** - Better final performance than linear
- Smooth convergence with gentle decay
- Standard choice for modern transformer training

**Implementation** (`trainer.py:285-292`):
```python
from transformers import get_cosine_schedule_with_warmup

scheduler_type = self.lr_scheduler_config.get('type', 'cosine_with_warmup')
warmup_steps = self.lr_scheduler_config.get('warmup_steps', 1000)
num_cycles = self.lr_scheduler_config.get('num_cycles', 0.5)

scheduler = get_cosine_schedule_with_warmup(
    self.optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    num_cycles=num_cycles
)
```

#### 2. Linear with Warmup

**Description**: Linear warmup followed by linear decay to zero (or min_lr).

**Configuration**:
```yaml
lr_scheduler:
  type: "linear"
  warmup_steps: 1000
  warmup_ratio: 0.1  # Alternative to warmup_steps
```

**Schedule**:
```
LR │        /────────╮
   │       /          ╲
   │      /            ╲
   │     /              ╲_
   │____/                  ──
   └────────────────────────────→ Steps
        ↑
     Warmup
```

**Use Case**:
- Smooth convergence
- Better final performance in many cases
- Recommended for fine-tuning

**Implementation**:
```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps,
    num_cycles=0.5
)
```

#### 3. Constant with Warmup

**Description**: Linear warmup then constant learning rate.

**Configuration**:
```yaml
lr_scheduler:
  type: "constant"
  warmup_steps: 1000
```

**Schedule**:
```
LR │        /──────────────
   │       /
   │      /
   │     /
   │____/
   └────────────────────────────→ Steps
        ↑
     Warmup
```

**Use Case**:
- When you want stable learning rate
- Short training runs
- Transfer learning

**Implementation**:
```python
from transformers import get_constant_schedule_with_warmup

scheduler = get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000
)
```

---

### Warmup Configuration

**Two options for specifying warmup**:

1. **Absolute steps**:
   ```yaml
   warmup_steps: 1000  # Warmup for exactly 1000 steps
   ```

2. **Ratio of total steps**:
   ```yaml
   warmup_ratio: 0.1  # Warmup for 10% of total steps
   ```

**Recommended warmup ratios**:
- Short training (<10 epochs): 0.05-0.1 (5-10%)
- Medium training (10-50 epochs): 0.06-0.08 (6-8%)
- Long training (>50 epochs): 0.03-0.05 (3-5%)

**Why warmup?**
- Prevents early training instability
- Allows model to explore parameter space gradually
- Reduces risk of divergence with large LRs
- Standard practice for transformer training

---

### Scheduler Comparison

| Scheduler | Convergence Speed | Final Performance | Use Case |
|-----------|------------------|-------------------|----------|
| **Linear** | Fast | Good | General purpose |
| **Cosine** | Medium | Better | When you want smooth convergence |
| **Constant** | Medium | Good (short runs) | Transfer learning, short training |

**Recommendation for Hindi BabyLM**: **Cosine with warmup** (currently used)
- Reason: Better final performance, smooth convergence, industry standard
- Prevents learning rate from decaying to zero (min_lr_ratio: 0.05)

### Batch Size and Gradient Accumulation

**Configuration** (from `configs/base_config.yaml:189-192`):
```yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 4  # Effective batch = 128
  num_epochs: 10
  max_steps: -1  # -1 for epoch-based training
```

**Effective Batch Size**: `batch_size × gradient_accumulation_steps`
- Actual batch: 32
- Accumulation steps: 4
- **Effective batch: 128**

**Gradient Accumulation Benefits**:
- ✅ Simulate larger batch sizes without OOM
- ✅ More stable training with limited memory
- ✅ Maintains model quality comparable to large batches
- ⚠️ Slightly slower due to more frequent forward passes

**Implementation** (see `trainer.py:293-377`):
```python
gradient_accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch).loss / gradient_accumulation_steps
    loss.backward()

    # Update weights every N batches
    if (i + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # Per-step scheduling
        optimizer.zero_grad()
```

### Training Duration

**Number of Epochs**: 10 (default)

**Words per Epoch**: ~10M or 100M per training, ~10M for val and test sets (current), other options inslude user defined ratios or word limits.

**Comparison**:
- BabyLM strict-small: 10M words (1 epoch equivalent)
- This setup: 100M words (10 epochs)
- Standard LLM: Trillions of tokens

### Regularization

**Techniques Applied**:

1. **Dropout**: 0.1
   - Applied to embeddings, attention, and FFN
   - Prevents overfitting

2. **Weight Decay**: 0.01
   - L2 regularization
   - Prevents large weights

3. **Gradient Clipping**: max_norm=1.0
   - Prevents exploding gradients
   - Stabilizes training

4. **Early Stopping** (optional):
   - Stop if validation loss doesn't improve for N epochs
   - Prevents overfitting

## Monitoring and Logging

### Weights & Biases Integration

**Status**: ✅ **Fully implemented** (see `trainer.py:697-736` for initialization, `trainer.py:384-391` for batch logging, `trainer.py:510-516` for epoch logging)

**Configuration** (W&B can be nested under `experiment_tracking` or at top level):
```yaml
experiment_tracking:
  wandb:
    enabled: true
    project: "hindi-babylm"
    entity: "your-wandb-entity"  # Optional
    mode: "online"  # Options: online, offline, disabled
    tags: ["baseline", "gpt", "10M"]
    notes: "Experiment description"
    watch_model: false  # Log model gradients/parameters (increases overhead)
```

**Initialization** (`trainer.py:697-736`):
```python
def _initialize_wandb(self):
    """Initialize Weights & Biases logging"""
    # Support nested config: experiment_tracking.wandb or direct wandb key
    experiment_tracking = self.config.get('experiment_tracking', {})
    wandb_config = experiment_tracking.get('wandb', self.config.get('wandb', {}))

    if not wandb_config.get('enabled', False):
        logger.info("W&B logging disabled in config")
        return

    wandb.init(
        project=wandb_config.get('project', 'hindi-babylm'),
        entity=wandb_config.get('entity'),
        name=self.config.get('experiment_name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
        config=self.config,
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes'),
        mode=wandb_config.get('mode', 'online')
    )

    # Optionally watch model (adds overhead)
    if wandb_config.get('watch_model', False):
        wandb.watch(self.model, log='all', log_freq=100)

    self.wandb_initialized = True
```

**Logged Metrics** (with nested organization):

1. **Batch-level** (every `log_steps` batches, default 100):
   - `train/batch_loss`
   - `train/learning_rate`
   - `train/gradient_norm`
   - `train/global_step`
   - `train/epoch`

2. **Epoch-level**:
   - `epoch/train_loss`
   - `epoch/val_loss`
   - `epoch/perplexity`
   - `epoch/number`

3. **Evaluation** (if callbacks enabled):
   - `eval/{task_name}/accuracy`
   - `eval/overall/average_accuracy`
   - Additional task-specific metrics

**Visualization**:
- Real-time loss curves (train and validation)
- Perplexity tracking over epochs
- Learning rate schedule visualization
- Gradient norm monitoring
- Evaluation metrics (if enabled)
- Hyperparameter tracking

### Console Output

Example output during training:
```
Epoch 1/10
Training: 100%|████████| 250/250 [05:23<00:00, 0.77it/s, loss=4.231]
Evaluating: 100%|████████| 31/31 [00:12<00:00, 2.51it/s]
Train Loss: 4.2310
Val Loss: 3.9856
Perplexity: 53.71
Checkpoint saved: results/{experiment_name}/models/checkpoint_epoch_1.pt
```

## Checkpointing Strategy

### Types of Checkpoints

1. **Best Model**:
   - Saved when validation loss improves
   - Used for final evaluation
   - File: `checkpoint_best.pt`

2. **Epoch Checkpoints**:
   - Saved after each epoch
   - Allows resuming training
   - File: `checkpoint_epoch_{N}.pt`

3. **Step Checkpoints** (optional):
   - Saved every N steps
   - For very long training runs
   - File: `checkpoint_step_{N}.pt`

### Checkpoint Contents

```python
checkpoint = {
    'epoch': current_epoch,
    'step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),  # If used
    'train_loss': train_loss,
    'val_loss': val_loss,
    'perplexity': perplexity,
    'config': config,
    'random_state': torch.get_rng_state()  # For reproducibility
}
```

### Resuming Training

```python
# Load checkpoint
checkpoint = torch.load(checkpoint_path)

# Restore model
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Restore scheduler
if scheduler:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Resume from epoch
start_epoch = checkpoint['epoch'] + 1
```

## Advanced Training Techniques

### 1. Mixed Precision Training (FP16/BF16)

**Status**: ✅ **Fully implemented** (see `trainer.py:89-105` for dtype setup, `trainer.py:328-377` for training loop)

**Benefits**:
- ~2× faster training
- ~50% less memory usage
- Minimal accuracy loss (especially with BF16)
- Better numerical stability with BF16 on modern GPUs (H100, A100)

**Configuration** (from `configs/base_config.yaml:217-221`):
```yaml
mixed_precision:
  enabled: true
  dtype: "bf16"  # Options: fp16, bf16, float32
  opt_level: "O1"  # For apex mixed precision (if used)
```

**Supported Datatypes**:
1. **BF16 (Recommended)**: Better numerical range, no gradient scaling needed, native H100 support
2. **FP16**: Wider compatibility, requires gradient scaling
3. **FP32**: Full precision fallback (no speedup)

**Implementation** (from actual `trainer.py`):
```python
from torch.cuda.amp import autocast
from torch.amp import grad_scaler

# Initialization (trainer.py:89-105)
if self.device.type == 'cuda':
    dtype_str = config.get('mixed_precision_dtype', 'bf16')
    dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    target_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    self.model = self.model.to(dtype=target_dtype)

# Training loop (trainer.py:328-346)
self.use_amp = config.get('use_amp', False)
self.scaler = grad_scaler() if self.use_amp else None

for batch in dataloader:
    # Forward pass with mixed precision
    if self.use_amp:
        with autocast():
            outputs = self.model(input_ids, attention_mask, labels=input_ids)
            loss = outputs.loss / gradient_accumulation_steps
        # Backward with gradient scaling
        self.scaler.scale(loss).backward()
    else:
        outputs = self.model(input_ids, attention_mask, labels=input_ids)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

    # Gradient clipping and optimizer step (with accumulation)
    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

        self.optimizer.zero_grad()
```

**Why BF16 is Recommended**:
- No overflow/underflow issues (wider exponent range than FP16)
- No need for loss scaling
- Native support on H100, A100 GPUs
- Same performance as FP16 with better stability

### 2. Gradient Accumulation

**When to Use**: Simulate larger batch sizes with limited memory

```python
accumulation_steps = 4  # Effective batch size = 32 * 4 = 128

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = outputs.loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

### 5. Evaluation Callbacks (Optional)

**Status**: ✅ **Fully implemented** (see `trainer.py:174-222` for initialization, `trainer.py:519-565` for usage)

**Purpose**: Run IndicGLUE or other benchmark evaluations during training for better checkpoint selection.

**Configuration** (from `configs/base_config.yaml:258-278`):
```yaml
training:
  # Evaluation Callbacks
  enable_eval_callback: false  # Set to true to enable
  eval_frequency: 1  # Evaluate every N epochs
  eval_on_steps: []  # Or specify steps: [1000, 2000, 3000]
  log_eval_to_wandb: true

  # Evaluation-Based Early Stopping
  eval_early_stopping: true
  eval_early_stopping_metric: "overall.average_accuracy"
  eval_early_stopping_patience: 3
  eval_early_stopping_mode: "max"  # 'max' or 'min'
  eval_early_stopping_min_delta: 0.001

  # Checkpoint Selection Based on Evaluation
  checkpoint_metric: "overall.average_accuracy"
  checkpoint_metric_mode: "max"
  load_best_checkpoint_at_end: true
```

**Features**:

1. **EvaluationCallback**: Runs benchmarks (IndicGLUE/MultiBLiMP) at specified intervals
   ```python
   # Automatically initialized if enabled in config (trainer.py:182-193)
   if training_config.get('enable_eval_callback', False):
       self.eval_callback = create_evaluation_callback(
           self.model,
           self.tokenizer,
           self.config
       )
   ```

2. **EvaluationBasedEarlyStopping**: Stop training based on evaluation metrics (not just val loss)
   ```python
   # Stops if accuracy plateaus (trainer.py:196-208)
   self.eval_early_stopping = EvaluationBasedEarlyStopping(
       metric_name='overall.average_accuracy',
       patience=3,
       min_delta=0.001,
       mode='max'
   )
   ```

3. **CheckpointSelector**: Track and select best checkpoint based on evaluation metrics
   ```python
   # Saves best model by evaluation metric, not validation loss (trainer.py:210-221)
   self.checkpoint_selector = CheckpointSelector(
       checkpoint_dir=str(self.checkpoint_dir),
       metric_name='overall.average_accuracy',
       mode='max'
   )
   ```

**Usage in Training Loop** (`trainer.py:519-565`):
```python
# After validation
eval_results = None
if self.eval_callback is not None:
    eval_results = self.eval_callback.on_epoch_end(
        epoch=epoch,
        step=self.global_step,
        model=self.model
    )

# Register checkpoint with selector
if improved and self.checkpoint_selector is not None and eval_results is not None:
    self.checkpoint_selector.register_checkpoint(checkpoint_path, eval_results)

# Check evaluation-based early stopping
if self.eval_early_stopping is not None and eval_results is not None:
    if self.eval_early_stopping(eval_results):
        logger.info("Early stopping triggered (evaluation metric)")
        break
```

**Benefits**:
- ✅ Select checkpoints based on downstream task performance
- ✅ Avoid overfitting to validation perplexity
- ✅ Early stop if model stops improving on actual tasks
- ✅ Better correlation with final model quality

**Trade-offs**:
- ⚠️ Slower training (evaluation overhead)
- ⚠️ Requires evaluation data and compute
- ⚠️ Disabled by default to prioritize training speed

## Complete Training Example

```python
from src.models.model_factory import ModelFactory
from src.tokenization.tokenizer_factory import TokenizerFactory
from src.data_processing.corpus_builder import CorpusBuilder
from src.training.trainer import HindiLanguageModelTrainer
from src.utils.experiment_config import ExperimentConfig

# 1. Load configuration
config = ExperimentConfig.load_config('configs/base_config.yaml')

# 2. Load data
corpus_builder = CorpusBuilder(config)
splits = corpus_builder.load_splits()

# 3. Create tokenizer
tokenizer_factory = TokenizerFactory(config)
tokenizer = tokenizer_factory.create_tokenizer(splits['train'])

# 4. Create dataloaders
train_loader = corpus_builder.create_dataloader(splits['train'], tokenizer, 'train')
val_loader = corpus_builder.create_dataloader(splits['val'], tokenizer, 'val')

# 5. Create model
model_factory = ModelFactory(config)
model = model_factory.create_model(tokenizer.vocab_size)

# 6. Create trainer
trainer = HindiLanguageModelTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config.__dict__
)

# 7. Train
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=config.num_epochs
)

# 8. Save final model
model_factory.save_model(model, tokenizer, "final_model")
```

## Troubleshooting

### Issue: Loss Not Decreasing

**Possible Causes**:
1. Learning rate too high/low
2. Batch size too small
3. Model too large for data
4. Data quality issues

**Solutions**:
1. Try learning rate in [1e-5, 1e-3]
2. Increase batch size or use gradient accumulation
3. Reduce model size
4. Check data preprocessing

### Issue: Overfitting (Train loss << Val loss)

**Solutions**:
1. Increase dropout (0.1 → 0.2)
2. Increase weight decay (0.01 → 0.1)
3. Reduce model size
4. Add more training data
5. Early stopping

### Issue: Out of Memory

**Solutions**:
1. Reduce batch size
2. Use gradient accumulation
3. Enable mixed precision (FP16)
4. Use gradient checkpointing
5. Reduce max_length

### Issue: Training Too Slow

**Solutions**:
1. Enable mixed precision
2. Increase batch size (if memory allows)
3. Reduce logging frequency
4. Use faster tokenizer

## Best Practices

1. **Start Small**: Train small model first to verify pipeline
2. **Monitor Closely**: Watch training curves for issues
3. **Save Often**: Frequent checkpointing prevents data loss
4. **Validate Regularly**: Check validation loss every epoch
5. **Document Everything**: Log hyperparameters and results
6. **Use Version Control**: Track code changes with git
7. **Reproducibility**: Set random seeds, save configurations

## Related Documentation

- [Model Architecture Documentation](04_MODELS.md)
- [Evaluation Framework Documentation](06_EVALUATION.md)
- [Configuration Guide](07_CONFIGURATION.md)
- [API Reference](08_API_REFERENCE.md)

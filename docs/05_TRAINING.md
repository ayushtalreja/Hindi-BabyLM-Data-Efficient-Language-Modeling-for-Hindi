# Training Pipeline

## Overview

The training pipeline manages the complete model training process, including optimization, learning rate scheduling, checkpointing, monitoring, and evaluation. The pipeline is designed for efficient training with limited data (~10M tokens).

## Architecture

```
DataLoader → Trainer → Model
              ↓
         Optimizer (AdamW)
              ↓
         LR Scheduler
              ↓
         Logging (Wandb)
              ↓
         Checkpointing
              ↓
         Evaluation
```

## HindiLanguageModelTrainer

**Location**: `src/training/trainer.py:9`

**Purpose**: Main training loop with monitoring and checkpointing.

### Initialization

```python
class HindiLanguageModelTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'learning_rate': []
        }
```

### Core Methods

#### `train(train_dataloader, val_dataloader, num_epochs)` (line 104)

**Purpose**: Main training loop

**Process**:
1. Initialize Weights & Biases
2. For each epoch:
   - Train on training set
   - Evaluate on validation set
   - Log metrics
   - Save checkpoint if best
3. Return final metrics

```python
def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int):
    """Main training loop"""

    # Initialize wandb
    wandb.init(
        project=self.config['project_name'],
        config=self.config
    )

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        train_loss = self.train_epoch(train_dataloader)

        # Validation
        val_metrics = self.evaluate(val_dataloader)

        # Logging
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_metrics['val_loss'])
        self.metrics['perplexity'].append(val_metrics['perplexity'])

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['val_loss'],
            'perplexity': val_metrics['perplexity']
        })

        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            self.save_checkpoint(epoch, val_metrics['val_loss'])
```

#### `train_epoch(dataloader)` (line 34)

**Purpose**: Train for one epoch

**Process**:
1. Set model to training mode
2. For each batch:
   - Move to device
   - Forward pass
   - Compute loss
   - Backward pass
   - Clip gradients
   - Update weights
   - Log to Wandb
3. Return average loss

```python
def train_epoch(self, dataloader: DataLoader) -> float:
    """Train for one epoch"""
    self.model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # For language modeling, labels = input_ids
        )

        loss = outputs.loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

        # Log to wandb
        wandb.log({
            'batch_loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })

    return total_loss / len(dataloader)
```

**Key Components**:
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **Progress Bar**: Visual feedback with tqdm
- **Batch Logging**: Real-time loss tracking

#### `evaluate(dataloader)` (line 75)

**Purpose**: Evaluate model on validation/test set

**Returns**: Dictionary with `val_loss` and `perplexity`

```python
def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
    """Evaluate model on validation set"""
    self.model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return {
        'val_loss': avg_loss,
        'perplexity': perplexity.item()
    }
```

**Perplexity**:
```
Perplexity = exp(loss)
```
- Lower is better
- Measures how "surprised" model is
- Good baseline: ~20-50 for Hindi

#### `save_checkpoint(epoch, val_loss)` (line 145)

**Purpose**: Save model checkpoint

```python
def save_checkpoint(self, epoch: int, val_loss: float):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'val_loss': val_loss,
        'config': self.config
    }

    checkpoint_path = os.path.join(
        self.config['checkpoint_dir'],
        f'checkpoint_epoch_{epoch}.pt'
    )
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
```

## Training Configuration

### Optimizer

**AdamW (Adam with Weight Decay)**

```python
torch.optim.AdamW(
    params=model.parameters(),
    lr=3e-4,              # Learning rate
    betas=(0.9, 0.999),   # Exponential decay rates
    eps=1e-8,             # Numerical stability
    weight_decay=0.01     # L2 regularization
)
```

**Why AdamW?**
- Adaptive learning rates per parameter
- Effective for transformers
- Decoupled weight decay (better than Adam)

### Learning Rate

**Base Learning Rate**: 3e-4

**Recommended Ranges**:
- Small models (<50M): 1e-3 to 3e-3
- Base models (~100M): 1e-4 to 1e-3
- Large models (>300M): 1e-5 to 1e-4

**Scheduling** (optional but recommended):

```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,    # Warmup for 1000 steps
    num_training_steps=total_steps
)
```

**Schedule Types**:
1. **Linear Warmup + Decay**: Gradual increase then linear decrease
2. **Cosine Annealing**: Smooth decay following cosine curve
3. **Constant with Warmup**: Constant after warmup

### Batch Size

**Default**: 32

**Memory Considerations**:
| Batch Size | GPU Memory | Speed | Convergence |
|------------|-----------|-------|-------------|
| 8          | ~4 GB     | Slow  | May be noisy |
| 16         | ~8 GB     | Good  | Stable |
| 32         | ~16 GB    | Fast  | Very stable |
| 64         | ~32 GB    | Faster| Most stable |

**Gradient Accumulation** (for larger effective batch size):
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch).loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Training Duration

**Number of Epochs**: 10 (default)

**Tokens per Epoch**: ~8M (with 10M total corpus, 80% train)

**Total Training Tokens**: ~80M (10 epochs × 8M)

**Comparison**:
- BabyLM strict-small: 10M tokens (1 epoch equivalent)
- This setup: 80M tokens (10 epochs)
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

**Initialization**:
```python
wandb.init(
    project="hindi-babylm",
    config=config,
    name=experiment_name
)
```

**Logged Metrics**:
- Batch loss (every batch)
- Epoch loss (train and validation)
- Perplexity
- Learning rate
- Gradient norms (optional)
- Model checkpoints (optional)

**Visualization**:
- Loss curves
- Perplexity over time
- Learning rate schedule
- Gradient statistics

### Console Output

Example output during training:
```
Epoch 1/10
Training: 100%|████████| 250/250 [05:23<00:00, 0.77it/s, loss=4.231]
Evaluating: 100%|████████| 31/31 [00:12<00:00, 2.51it/s]
Train Loss: 4.2310
Val Loss: 3.9856
Perplexity: 53.71
Checkpoint saved: checkpoints/checkpoint_epoch_1.pt
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

### 1. Mixed Precision Training (FP16)

**Benefits**:
- ~2× faster training
- ~50% less memory
- Minimal accuracy loss

**Implementation**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in FP16
    with autocast():
        outputs = model(batch)
        loss = outputs.loss

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
```

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

### 3. Curriculum Learning

**Concept**: Train on simpler examples first, gradually increase complexity

**Strategies**:
1. **Length-Based**: Short texts → Long texts
2. **Morphological**: Simple forms → Complex inflections
3. **Lexical**: Frequent words → Rare words

**Implementation** (example):
```python
def sort_by_difficulty(texts):
    # Sort by sentence length
    return sorted(texts, key=lambda x: len(x.split()))

# Create curriculum
easy_texts = texts[:len(texts)//3]
medium_texts = texts[len(texts)//3:2*len(texts)//3]
hard_texts = texts[2*len(texts)//3:]

# Train in stages
train_on(easy_texts, epochs=3)
train_on(medium_texts, epochs=4)
train_on(hard_texts, epochs=3)
```

### 4. Learning Rate Finder

**Purpose**: Find optimal learning rate before full training

```python
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()  # Visualize loss vs. learning rate
best_lr = lr_finder.suggest_lr()
```

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
3. Use more GPUs (DataParallel/DistributedDataParallel)
4. Reduce logging frequency
5. Use faster tokenizer

## Best Practices

1. **Start Small**: Train small model first to verify pipeline
2. **Monitor Closely**: Watch training curves for issues
3. **Save Often**: Frequent checkpointing prevents data loss
4. **Validate Regularly**: Check validation loss every epoch
5. **Document Everything**: Log hyperparameters and results
6. **Use Version Control**: Track code changes with git
7. **Reproducibility**: Set random seeds, save configurations

## Performance Benchmarks

### Training Speed (on V100 GPU)

| Model Size | Batch Size | Tokens/sec | Time/Epoch |
|------------|-----------|-----------|-----------|
| 40M        | 32        | ~8000      | ~15 min   |
| 110M       | 32        | ~5000      | ~25 min   |
| 220M       | 16        | ~2500      | ~50 min   |

### Convergence

Typical training curves:
- **Epoch 1**: Loss ~6-8, Perplexity ~400-3000
- **Epoch 5**: Loss ~3-4, Perplexity ~20-50
- **Epoch 10**: Loss ~2-3, Perplexity ~7-20

Good Hindi language model: **Perplexity < 30**

## Related Documentation

- [Model Architecture Documentation](04_MODELS.md)
- [Evaluation Framework Documentation](06_EVALUATION.md)
- [Configuration Guide](07_CONFIGURATION.md)
- [API Reference](08_API_REFERENCE.md)

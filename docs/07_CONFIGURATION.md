# Configuration Guide

## Overview

The Hindi BabyLM project uses a centralized configuration system based on YAML files and Python dataclasses. All experiments are defined through configuration files, enabling reproducibility and easy experimentation.

## Configuration System Architecture

```
YAML Configuration File
         ↓
  ExperimentConfig (dataclass)
         ↓
  Component Initialization
    ├─→ Data Processing
    ├─→ Tokenization
    ├─→ Model Creation
    ├─→ Training
    └─→ Evaluation
```

## ExperimentConfig Class

**Location**: `src/utils/experiment_config.py:8`

**Purpose**: Central configuration management using Python dataclass

### Full Configuration Schema

```python
@dataclass
class ExperimentConfig:
    # ===== Experiment Metadata =====
    experiment_name: str = "default_experiment"

    # ===== Directory Configuration =====
    data_dir: str = "data"
    model_dir: str = "models"
    tokenizer_dir: str = "tokenizers"
    results_dir: str = "results"

    # ===== Data Configuration =====
    max_tokens: int = 10_000_000    # Total tokens (~10M for BabyLM)
    train_ratio: float = 0.8         # Training split ratio
    val_ratio: float = 0.1           # Validation split ratio
    test_ratio: float = 0.1          # Test split ratio

    # ===== Tokenization Configuration =====
    tokenizer_type: str = "sentencepiece"  # sentencepiece, wordpiece, bpe
    vocab_size: int = 32000                 # Vocabulary size

    # ===== Model Configuration =====
    model_type: str = "gpt"          # gpt, bert, hybrid
    hidden_size: int = 768           # Hidden dimension
    num_layers: int = 12             # Number of transformer layers
    num_heads: int = 12              # Number of attention heads
    max_length: int = 512            # Maximum sequence length
    dropout: float = 0.1             # Dropout probability
    intermediate_size: int = 3072    # FFN intermediate size (4*hidden_size)

    # ===== Training Configuration =====
    batch_size: int = 32             # Training batch size
    learning_rate: float = 3e-4      # Initial learning rate
    num_epochs: int = 10             # Number of training epochs
    weight_decay: float = 0.01       # L2 regularization
    warmup_steps: int = 1000         # LR warmup steps

    # ===== Curriculum Learning =====
    use_curriculum: bool = False              # Enable curriculum learning
    curriculum_strategy: str = "morphological"  # morphological, length, random

    # ===== Evaluation Configuration =====
    eval_steps: int = 500            # Evaluate every N steps
    save_steps: int = 1000           # Save checkpoint every N steps
```

### Methods

#### `load_config(path)` (line 58)

**Purpose**: Load configuration from YAML file

```python
@classmethod
def load_config(cls, path: str):
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Handle nested YAML structure
    flat_config = {}

    # Parse nested sections
    if 'data' in config_dict:
        flat_config.update(config_dict.get('data', {}))
    if 'tokenization' in config_dict:
        tokenization = config_dict.get('tokenization', {})
        if 'vocab_size' in tokenization:
            flat_config['vocab_size'] = tokenization['vocab_size']
        if 'methods' in tokenization:
            flat_config['tokenizer_type'] = tokenization['methods'][0]
    if 'training' in config_dict:
        training = config_dict.get('training', {})
        flat_config.update(training)
        if 'max_epochs' in training:
            flat_config['num_epochs'] = training['max_epochs']
    if 'model' in config_dict:
        flat_config.update(config_dict.get('model', {}))

    # If config is flat (not nested), use directly
    if not any(key in config_dict for key in ['data', 'tokenization', 'training', 'model']):
        flat_config = config_dict

    return cls(**flat_config)
```

#### `save_config(path)` (line 52)

**Purpose**: Save configuration to YAML file

```python
def save_config(self, path: str):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(self.__dict__, f, default_flow_style=False)
```

## Base Configuration File

**Location**: `configs/base_config.yaml`

```yaml
project:
  name: "hindi-babylm"
  description: "Hindi BabyLM Challenge Implementation"

data:
  max_tokens: 10_000_000  # 10M tokens for strict-small track
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

tokenization:
  vocab_size: 32000
  methods: ["sentencepiece", "wordpiece", "bpe"]

training:
  batch_size: 32
  learning_rate: 3e-4
  max_epochs: 10
  save_steps: 1000
  eval_steps: 500
```

## Configuration Templates

### 1. Quick Experiment (Small Model)

**File**: `configs/quick_experiment.yaml`

```yaml
experiment_name: "quick_test"

# Data
data:
  max_tokens: 1_000_000  # 1M tokens for quick testing
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

# Tokenization
tokenization:
  tokenizer_type: "sentencepiece"
  vocab_size: 8000  # Smaller vocabulary

# Model
model:
  model_type: "gpt"
  hidden_size: 256      # Smaller model
  num_layers: 6         # Fewer layers
  num_heads: 4          # Fewer heads
  max_length: 256       # Shorter sequences
  dropout: 0.1
  intermediate_size: 1024

# Training
training:
  batch_size: 16        # Smaller batch
  learning_rate: 1e-3
  num_epochs: 3         # Fewer epochs
  weight_decay: 0.01
  warmup_steps: 100
```

**Use Case**: Quick prototyping, debugging, testing pipeline

**Training Time**: ~30 minutes on GPU

### 2. Standard Baseline

**File**: `configs/baseline.yaml`

```yaml
experiment_name: "hindi_babylm_baseline"

# Data
data:
  max_tokens: 10_000_000
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

# Tokenization
tokenization:
  tokenizer_type: "sentencepiece"
  vocab_size: 32000

# Model
model:
  model_type: "gpt"
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  max_length: 512
  dropout: 0.1
  intermediate_size: 3072

# Training
training:
  batch_size: 32
  learning_rate: 3e-4
  num_epochs: 10
  weight_decay: 0.01
  warmup_steps: 1000
  eval_steps: 500
  save_steps: 1000
```

**Use Case**: Standard BabyLM baseline

**Training Time**: ~4 hours on V100 GPU

### 3. BERT-Style Model

**File**: `configs/bert_baseline.yaml`

```yaml
experiment_name: "hindi_bert_baseline"

tokenization:
  tokenizer_type: "wordpiece"  # BERT uses WordPiece
  vocab_size: 32000

model:
  model_type: "bert"            # BERT architecture
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  max_length: 512
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 1e-4           # Lower LR for BERT
  num_epochs: 10
  weight_decay: 0.01
  warmup_steps: 1000
```

**Use Case**: Masked language modeling, understanding tasks

### 4. Hybrid Model

**File**: `configs/hybrid.yaml`

```yaml
experiment_name: "hindi_hybrid"

model:
  model_type: "hybrid"          # Combined GPT + BERT
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  max_length: 512

training:
  batch_size: 16                # Smaller batch (larger model)
  learning_rate: 3e-4
  num_epochs: 10
  weight_decay: 0.01
```

**Use Case**: Multi-task learning, versatile model

### 5. Curriculum Learning

**File**: `configs/curriculum.yaml`

```yaml
experiment_name: "curriculum_learning"

data:
  max_tokens: 10_000_000

curriculum:
  use_curriculum: true
  curriculum_strategy: "morphological"  # Options: morphological, length, random

training:
  batch_size: 32
  learning_rate: 3e-4
  num_epochs: 12                # More epochs for curriculum
```

**Use Case**: Developmental training progression

### 6. Large Model

**File**: `configs/large_model.yaml`

```yaml
experiment_name: "hindi_large"

tokenization:
  vocab_size: 50000             # Larger vocabulary

model:
  model_type: "gpt"
  hidden_size: 1024             # Larger hidden size
  num_layers: 24                # More layers
  num_heads: 16                 # More heads
  max_length: 1024              # Longer sequences
  intermediate_size: 4096

training:
  batch_size: 16                # Smaller batch (memory)
  learning_rate: 2e-4           # Lower LR
  num_epochs: 15
```

**Use Case**: Maximum performance (if compute available)

**Training Time**: ~12+ hours on V100

## Experiment Manager

**Location**: `src/utils/experiment_config.py:91`

**Purpose**: Generate multiple experiment configurations

### Methods

#### `create_tokenization_experiments()` (line 96)

**Purpose**: Generate experiments for tokenizer comparison

```python
def create_tokenization_experiments(self) -> List[ExperimentConfig]:
    """Create experiments for different tokenization strategies"""
    tokenizers = ["sentencepiece", "wordpiece", "bpe"]
    experiments = []

    for tokenizer in tokenizers:
        config = copy.deepcopy(self.base_config)
        config.tokenizer_type = tokenizer
        config.experiment_name = f"tokenization_{tokenizer}"
        experiments.append(config)

    return experiments
```

**Usage**:
```python
base_config = ExperimentConfig.load_config('configs/base_config.yaml')
manager = ExperimentManager(base_config)

# Generate tokenization experiments
tokenization_exps = manager.create_tokenization_experiments()

# Run each experiment
for config in tokenization_exps:
    run_experiment(config)
```

#### `create_model_architecture_experiments()` (line 109)

**Purpose**: Generate experiments for architecture comparison

```python
def create_model_architecture_experiments(self) -> List[ExperimentConfig]:
    """Create experiments for different model architectures"""
    architectures = ["gpt", "bert", "hybrid"]
    experiments = []

    for arch in architectures:
        config = copy.deepcopy(self.base_config)
        config.model_type = arch
        config.experiment_name = f"architecture_{arch}"
        experiments.append(config)

    return experiments
```

#### `create_curriculum_experiments()` (line 122)

**Purpose**: Generate experiments for curriculum learning strategies

```python
def create_curriculum_experiments(self) -> List[ExperimentConfig]:
    """Create experiments for curriculum learning strategies"""
    strategies = ["morphological", "length", "random", "none"]
    experiments = []

    for strategy in strategies:
        config = copy.deepcopy(self.base_config)
        config.use_curriculum = strategy != "none"
        config.curriculum_strategy = strategy
        config.experiment_name = f"curriculum_{strategy}"
        experiments.append(config)

    return experiments
```

## Configuration Best Practices

### 1. Naming Conventions

**Experiment Names**: Use descriptive, structured names
```yaml
experiment_name: "{task}_{variant}_{date}"
# Examples:
# - "hindi_babylm_baseline_20250109"
# - "tokenization_sentencepiece_v1"
# - "curriculum_morphological_test"
```

### 2. Directory Structure

**Organize experiments**:
```
experiments/
├── baseline/
│   └── base_config.yaml
├── tokenization/
│   ├── sentencepiece.yaml
│   ├── wordpiece.yaml
│   └── bpe.yaml
├── architectures/
│   ├── gpt.yaml
│   ├── bert.yaml
│   └── hybrid.yaml
└── curriculum/
    ├── morphological.yaml
    ├── length.yaml
    └── random.yaml
```

### 3. Version Control

**Track configurations in git**:
```bash
git add configs/
git commit -m "Add tokenization experiment configs"
```

### 4. Documentation

**Document each experiment**:
```yaml
# base_config.yaml
# Description: Standard baseline for Hindi BabyLM
# Author: Your Name
# Date: 2025-01-09
# Purpose: Establish baseline performance with standard settings
# Expected training time: 4 hours on V100
# Expected perplexity: ~20-30

experiment_name: "hindi_babylm_baseline"
# ... rest of config
```

### 5. Hyperparameter Ranges

**Safe ranges for key hyperparameters**:

| Parameter | Min | Recommended | Max | Notes |
|-----------|-----|-------------|-----|-------|
| `learning_rate` | 1e-5 | 3e-4 | 1e-3 | Lower for large models |
| `batch_size` | 8 | 32 | 128 | Depends on GPU memory |
| `num_layers` | 6 | 12 | 24 | More layers = slower |
| `hidden_size` | 256 | 768 | 1024 | Must be divisible by num_heads |
| `dropout` | 0.0 | 0.1 | 0.3 | Higher for overfitting |
| `weight_decay` | 0.0 | 0.01 | 0.1 | Regularization strength |
| `vocab_size` | 8000 | 32000 | 50000 | Larger = more parameters |

## Loading and Using Configurations

### From Python

```python
from src.utils.experiment_config import ExperimentConfig

# Load from file
config = ExperimentConfig.load_config('configs/base_config.yaml')

# Access fields
print(f"Experiment: {config.experiment_name}")
print(f"Model: {config.model_type}")
print(f"Batch size: {config.batch_size}")

# Modify and save
config.learning_rate = 1e-4
config.save_config('configs/modified_config.yaml')

# Create new config programmatically
custom_config = ExperimentConfig(
    experiment_name="custom_experiment",
    model_type="bert",
    batch_size=16,
    learning_rate=1e-4
)
```

### From Command Line

```bash
# Run with specific config
python main.py --config configs/base_config.yaml --stage all --experiment_name my_exp

# Override config values
python main.py --config configs/base_config.yaml --stage train \
    --learning_rate 1e-4 --batch_size 16
```

## Environment Variables

**Set via `.env` file or shell**:

```bash
# Data directories
export DATA_DIR=/path/to/data
export MODEL_DIR=/path/to/models
export RESULTS_DIR=/path/to/results

# Wandb configuration
export WANDB_PROJECT=hindi-babylm
export WANDB_ENTITY=your-team

# Hardware
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

## Configuration Validation

**Check configuration before running**:

```python
def validate_config(config: ExperimentConfig):
    """Validate configuration values"""
    assert config.hidden_size % config.num_heads == 0, \
        "hidden_size must be divisible by num_heads"

    assert config.intermediate_size == 4 * config.hidden_size, \
        "intermediate_size should be 4 * hidden_size"

    assert 0 <= config.dropout <= 1, \
        "dropout must be between 0 and 1"

    assert config.train_ratio + config.val_ratio + config.test_ratio == 1.0, \
        "Split ratios must sum to 1.0"

    print("✓ Configuration is valid")

# Use before training
config = ExperimentConfig.load_config('configs/base_config.yaml')
validate_config(config)
```

## Troubleshooting

### Issue: Config file not loading

**Check**:
1. File path is correct
2. YAML syntax is valid (use YAML validator)
3. Required fields are present

### Issue: Parameter not taking effect

**Check**:
1. Parameter name matches exactly (case-sensitive)
2. Parameter is in correct section of YAML
3. Not overridden by command-line argument

### Issue: Out of memory

**Solutions**:
1. Reduce `batch_size`
2. Reduce `max_length`
3. Reduce model size (`hidden_size`, `num_layers`)
4. Enable gradient checkpointing

## Related Documentation

- [Training Pipeline Documentation](05_TRAINING.md)
- [Model Architecture Documentation](04_MODELS.md)
- [API Reference](08_API_REFERENCE.md)
- [Setup and Usage Guide](09_SETUP_AND_USAGE.md)

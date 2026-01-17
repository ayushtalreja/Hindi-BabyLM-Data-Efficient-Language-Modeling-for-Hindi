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

**Location**: `src/utils/experiment_config.py:29`

**Purpose**: Central configuration management using Python dataclass with support for model-specific configurations.

### Configuration Schema

```python
@dataclass
class ExperimentConfig:
    # ===== Experiment Metadata =====
    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None
    experiment_tags: Optional[List[str]] = None

    # ===== Directory Configuration =====
    data_dir: Optional[str] = None
    model_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    results_dir: Optional[str] = None

    # ===== Data Configuration =====
    max_words: Optional[int] = None           # Total words in corpus
    max_tokens: Optional[int] = None          # Deprecated: use max_words instead

    # Separate word limits for each split
    train_word_limit: Optional[int] = None    # Words for training split
    val_word_limit: Optional[int] = None      # Words for validation split
    test_word_limit: Optional[int] = None     # Words for test split

    train_ratio: Optional[float] = None
    val_ratio: Optional[float] = None
    test_ratio: Optional[float] = None

    # ===== Tokenization Configuration =====
    tokenizer_type: Optional[str] = None      # sentencepiece, wordpiece, bpe
    vocab_size: Optional[int] = None

    # ===== Model Configuration =====
    model_type: Optional[str] = None          # gpt, deberta
    model_size: Optional[str] = None          # tiny, small, medium/base, large
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    max_length: Optional[int] = None
    dropout: Optional[float] = None
    intermediate_size: Optional[int] = None

    # Model-specific configurations
    gpt_config: Optional[GPTModelConfig] = None
    deberta_config: Optional[DeBERTaModelConfig] = None

    # ===== Training Configuration =====
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    num_epochs: Optional[int] = None
    weight_decay: Optional[float] = None
    warmup_steps: Optional[int] = None

    # ===== Evaluation Configuration =====
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None

    # ===== Resource Configuration =====
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    device: Optional[str] = None

    # ===== Nested Configuration Sections =====
    sources: Optional[Dict[str, Any]] = field(default_factory=dict)
    filtering: Optional[Dict[str, Any]] = field(default_factory=dict)
    deduplication: Optional[Dict[str, Any]] = field(default_factory=dict)
    train_source_ratios: Optional[Dict[str, float]] = field(default_factory=dict)
```

### Model-Specific Configurations

**GPTModelConfig** (line 10):
```python
@dataclass
class GPTModelConfig:
    use_cache: Optional[bool] = None
    scale_attn_weights: Optional[bool] = None
    reorder_and_upcast_attn: Optional[bool] = None
```

**DeBERTaModelConfig** (line 18):
```python
@dataclass
class DeBERTaModelConfig:
    position_buckets: Optional[int] = None
    relative_attention: Optional[bool] = None
    max_relative_positions: Optional[int] = None
    pooler_hidden_size: Optional[int] = None
    pooler_dropout: Optional[float] = None
    pooler_hidden_act: Optional[str] = None
```

### Key Methods

#### `load_config(path)` (line 152)

**Purpose**: Load configuration from YAML file with support for nested structures

```python
@classmethod
def load_config(cls, path: str):
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Handles nested YAML structure (experiment, directories, data, etc.)
    # Extracts and flattens configuration sections
    # Creates model-specific config instances
    # Filters out non-dataclass fields

    return cls(**filtered_config)
```

**Features**:
- Handles nested YAML sections (experiment, directories, data, tokenization, model, training, resources)
- Maps nested fields to flat dataclass attributes (e.g., `model.architecture.hidden_size` → `hidden_size`)
- Creates GPTModelConfig or DeBERTaModelConfig instances from nested model configs
- Filters out extra YAML fields not in dataclass schema

#### `save_config(path)` (line 146)

**Purpose**: Save configuration to YAML file with only relevant model-specific parameters

```python
def save_config(self, path: str):
    """Save configuration to YAML file (with clean model-specific params)"""
    with open(path, 'w') as f:
        yaml.dump(self.to_clean_dict(), f, default_flow_style=False)
```

#### `to_clean_dict()` (line 127)

**Purpose**: Return dictionary representation with only active model-specific configuration

```python
def to_clean_dict(self):
    """
    Return a dictionary representation with only relevant model-specific params.
    This ensures saved configs don't contain cross-contamination.
    """
    # Includes only the active model config (gpt_config OR deberta_config)
    # Skips inactive model configs to prevent contamination
```

#### `from_checkpoint_config(config_dict)` (line 302)

**Purpose**: Create ExperimentConfig from checkpoint with backward compatibility

```python
@classmethod
def from_checkpoint_config(cls, config_dict: Dict[str, Any]):
    """
    Create ExperimentConfig from a checkpoint config dict.
    Handles backward compatibility with old flat configs.
    """
    # Migrates old flat model-specific params to nested structure
    # Handles both dict and dataclass instances for nested configs
    # Removes cross-contamination from other model types
```

#### Path Helper Methods (lines 360-370)

```python
def get_tokenizer_path(self) -> Path:
    """Get the experiment-scoped tokenizer directory path"""
    return Path(self.results_dir) / self.experiment_name / 'tokenizer'

def get_model_path(self) -> Path:
    """Get the experiment-scoped model directory path"""
    return Path(self.results_dir) / self.experiment_name / 'models'

def get_results_path(self) -> Path:
    """Get the experiment-scoped results directory path"""
    return Path(self.results_dir) / self.experiment_name
```

## Base Configuration File

**Location**: `configs/base_config.yaml`

This is the comprehensive base configuration that defines all aspects of the pipeline.

### Structure Overview

```yaml
# Top-level sections
experiment:          # Experiment metadata
project:             # Project information
directories:         # Path configuration
data:               # Data sources and processing
tokenization:       # Tokenizer configuration
model:              # Model architecture
training:           # Training parameters
evaluation:         # Evaluation benchmarks
experiment_tracking: # Wandb/TensorBoard
resources:          # Hardware configuration
analysis:           # Statistics and visualization
reproducibility:    # Seed management
debugging:          # Debug/profiling options
```

### Experiment Configuration

```yaml
experiment:
  name: "gpt_10M_baseline_finetuned_8K_Vocab"
  description: "Baseline GPT model on 10M word corpus with fine tuning on IndicGLUE with 8K vocab size"
  tags: ["baseline", "gpt", "10M", "strict-small", "8K Vocab"]
```

**Purpose**: Identifies and describes the experiment. The name is used for result directory creation.

### Data Configuration

#### Core Data Settings

```yaml
data:
  # Word Budget (BabyLM Strict-Small Track)
  max_words: 10_000_000  # 10M words

  # Separate word limits for each split
  train_word_limit: 10_000_000  # 10M words for training
  val_word_limit: 10_000_000    # 10M words for validation
  test_word_limit: 10_000_000   # 10M words for test

  # Data Splits (proportional sizing if word limits not specified)
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

**Key Parameters**:
- `max_words`: Total word budget (use this, not `max_tokens`)
- `train_word_limit`, `val_word_limit`, `test_word_limit`: Independent limits per split
- `train_ratio`, `val_ratio`, `test_ratio`: Proportional splits when word limits not used

#### Source Configuration

```yaml
data:
  # Source Mixing Ratios (for training data only)
  train_source_ratios:
    indiccorp: 0.50        # 50% from IndicCorp (formal written Hindi)
    wikipedia: 0.30        # 30% from Wikipedia (encyclopedic Hindi)
    indicdialogue: 0.15    # 15% from IndicDialogue (conversational Hindi)
    childrens_books: 0.05  # 5% from children's books (simple narrative Hindi)

  sources:
    indiccorp:
      enabled: true
      ratio: 0.5
      max_samples: 150000

    wikipedia:
      enabled: true
      ratio: 0.3
      dataset_version: "20231101.hi"
      max_articles: 100000

    indicdialogue:
      enabled: true
      ratio: 0.15
      max_movies: null  # null = use all 3,542 movies
      combine_dialogues: false
      min_dialogue_length: 10

    childrens_books:
      enabled: true
      ratio: 0.05
      max_stories: 8000
```

**Available Data Sources**:
1. **IndicCorp**: Formal written Hindi from news and web
2. **Wikipedia**: Encyclopedic Hindi articles
3. **IndicDialogue**: Conversational Hindi from movies
4. **Children's Books**: Simple narrative Hindi

**For Data Source Experiments**: Modify `train_source_ratios` to test different distributions.

#### Quality Filtering

```yaml
data:
  filtering:
    min_length: 30              # Minimum characters
    max_length: 2000            # Maximum characters
    min_hindi_ratio: 0.8        # Devanagari character ratio
    min_word_count: 2           # Minimum words
    max_word_count: 10000       # Maximum words
```

#### Deduplication

```yaml
data:
  deduplication:
    enabled: true
    similarity_threshold: 0.8   # MinHash LSH threshold
    num_permutations: 256
```

### Tokenization Configuration

```yaml
tokenization:
  type: "bpe"  # Options: sentencepiece, wordpiece, bpe
  vocab_size: 8192
  character_coverage: 0.9995  # For SentencePiece

  special_tokens:
    pad_token: "<pad>"
    unk_token: "<unk>"
    bos_token: "<s>"
    eos_token: "</s>"
    mask_token: "<mask>"
    sep_token: "<sep>"
    cls_token: "<cls>"

  # SentencePiece Specific
  sentencepiece:
    model_type: "bpe"  # Options: unigram, bpe
    split_by_whitespace: true
    split_by_unicode_script: true
    byte_fallback: true

  # WordPiece Specific
  wordpiece:
    min_frequency: 2
    continuing_subword_prefix: "##"

  # BPE Specific
  bpe:
    min_frequency: 2
    dropout: 0.0
```

**For Tokenization Experiments**:
- Change `type` to: `bpe`, `wordpiece`, or `sentencepiece`
- For SentencePiece, set `model_type` to `unigram` or `bpe`
- Adjust `vocab_size`: 8192, 16384, 32768 (8K, 16K, 32K)

**Tokenizer Types**:
- **BPE**: Byte-Pair Encoding (good for morphologically rich languages)
- **WordPiece**: BERT-style with `##` prefix
- **SentencePiece (Unigram)**: Statistical language model approach
- **SentencePiece (BPE)**: SentencePiece implementation of BPE

### Model Configuration

```yaml
model:
  type: "gpt"  # Options: gpt, deberta
  model_size: "small"  # Options: tiny, small, medium/base, large

  # Core Architecture Hyperparameters
  architecture:
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    intermediate_size: 3072
    max_position_embeddings: 512

  # Regularization
  regularization:
    hidden_dropout_prob: 0.1
    attention_probs_dropout_prob: 0.1

  # Activation and Normalization
  activation: "gelu"
  layer_norm_type: "layernorm"

  # GPT-Specific Configuration
  gpt:
    use_cache: true
    scale_attn_weights: true
    reorder_and_upcast_attn: false
```

**Model Sizes**:

**GPT Models** (Causal Language Modeling):
- `tiny`: 50M parameters (6 layers, 512 hidden, 8 heads)
- `small`: 110M parameters (12 layers, 768 hidden, 12 heads)
- `medium`: 350M parameters (24 layers, 1024 hidden, 16 heads)

**DeBERTa Models** (Masked Language Modeling):
- `tiny`: 22M parameters (6 layers, 384 hidden)
- `small`: 86M parameters (12 layers, 768 hidden)
- `base`: 86M parameters (12 layers, 768 hidden)
- `large`: 304M parameters (24 layers, 1024 hidden)

**For Model Architecture Experiments**:
- Change `type` to `gpt` or `deberta`
- Adjust `model_size` for different parameter counts
- For DeBERTa, add `deberta:` section instead of `gpt:` section

### Training Configuration

```yaml
training:
  # Basic Parameters
  batch_size: 32
  gradient_accumulation_steps: 4  # Effective batch = 128
  num_epochs: 10
  max_steps: -1  # -1 for epoch-based

  # Optimization
  optimizer:
    type: "adamw"
    learning_rate: 0.0005
    weight_decay: 0.01
    beta1: 0.9
    beta2: 0.999

  # Learning Rate Scheduling
  lr_scheduler:
    type: "cosine_with_warmup"
    warmup_steps: 1500
    warmup_ratio: 0.1
    min_lr_ratio: 0.05

  # Gradient Management
  gradient:
    max_grad_norm: 1.0
    gradient_checkpointing: true

  # Mixed Precision
  mixed_precision:
    enabled: true
    dtype: "bf16"  # Options: fp16, bf16, float32

  # Checkpointing
  checkpointing:
    save_strategy: "steps"
    save_steps: 1000
    save_total_limit: 1
    load_best_model_at_end: true

  # Evaluation
  evaluation:
    eval_strategy: "steps"
    eval_steps: 500
    per_device_eval_batch_size: 32

  # Early Stopping
  early_stopping:
    enabled: true
    patience: 3
    threshold: 0.0005
```

**Key Training Parameters**:
- **Effective Batch Size**: `batch_size × gradient_accumulation_steps` = 128
- **Learning Rate**: 0.0005 for pretraining (lower for fine-tuning: 2e-5)
- **Mixed Precision**: BF16 for faster training and memory efficiency
- **Gradient Checkpointing**: Enabled to reduce memory usage

### Evaluation Configuration

```yaml
evaluation:
  benchmarks:
    indicglue:
      enabled: true
      tasks: [
        "BBCArticlesClassification",
        "Wikipedia Section Title Prediction",
        "Cloze-style multiple-choice QA",
        "WinogradNLI",
        "Choice of Plausible Alternatives",
        "MovieReviewSentiment",
        "ProductReviewSentiment",
        "DiscourseMode"
      ]
      batch_size: 64

      fine_tuning:
        enabled: true
        num_epochs: 10
        learning_rate: 3e-5
        batch_size: 16
        freeze_base_model: false  # End-to-end training
        use_auto_models: false    # Use custom wrappers

    multiblimp:
      enabled: true
      n_examples_per_phenomenon: null

  perplexity:
    enabled: true
    split: "test"
    report_per_source: true
```

**Evaluation Benchmarks**:
1. **IndicGLUE**: 8 Hindi NLP tasks (classification, sentiment, QA, NLI)
   - Zero-shot or fine-tuned evaluation
   - Fine-tuning uses separate training loop
2. **MultiBLiMP**: 1,447 minimal pairs for syntactic evaluation
3. **Perplexity**: Language modeling performance on test set

### Resource Configuration

```yaml
resources:
  device: "cuda"
  gpu_ids: [0,1,2,3,4,5,6,7,8,9,10]
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
```

**Important**: `num_workers: 4` is recommended to avoid excessive memory usage.

## Configuration Templates for Conducted Experiments

### 1. Vocabulary Size Experiments (GPT + BPE)

**Purpose**: Test impact of vocabulary size on GPT models

**Experiments**:
- `configs/base_config.yaml` # Refers to gpt bpe 8K  
- `configs/ablations/vocab_sizes/gpt_bpe_16k.yaml`
- `configs/ablations/vocab_sizes/gpt_bpe_32k.yaml`

**Key Changes** from base config:
```yaml
experiment:
  name: "gpt_10M_bpe_8K_vocab"  # Change: 8K, 16K, 32K
  tags: ["vocab-experiment", "gpt", "bpe"]

tokenization:
  type: "bpe"
  vocab_size: 8192  # Change: 8192, 16384, 32768

model:
  type: "gpt"
  model_size: "small"
```

### 2. Tokenization Strategy Experiments

**Purpose**: Compare tokenization methods across model architectures

**Experiments** (10 total):
- GPT: bpe, unigram, wordpiece, character-level, bigram (5 experiments)
- DeBERTa: bpe, unigram, wordpiece, character-level, bigram (5 experiments)

**Example**: `configs/tokenization_experiments/gpt_unigram_10M_32K.yaml`
```yaml
experiment:
  name: "gpt_10M_unigram_32K"
  tags: ["tokenization-experiment", "gpt", "unigram"]

tokenization:
  type: "sentencepiece"  # For unigram
  vocab_size: 32768
  sentencepiece:
    model_type: "unigram"  # Change: unigram, bpe

model:
  type: "gpt"  # Change: gpt, deberta
```

**Tokenizer Options**:
- `type: "bpe"` → Pure BPE
- `type: "wordpiece"` → WordPiece
- `type: "sentencepiece"` + `model_type: "unigram"` → Unigram LM
- `type: "sentencepiece"` + `model_type: "bpe"` → SentencePiece BPE

### 3. Model Architecture & Data Amount Experiments

**Purpose**: Compare GPT vs DeBERTa on different data scales

**Experiments** (4 total):
- GPT + 10M words
- GPT + 100M words
- DeBERTa + 10M words
- DeBERTa + 100M words

**Example**: `configs/data_amount/deberta_bpe_100M_32K.yaml`
```yaml
experiment:
  name: "deberta_100M_bpe_16K"
  tags: ["architecture-experiment", "deberta", "100M"]

data:
  max_words: 100_000_000  # Change: 10M or 100M

tokenization:
  type: "bpe"
  vocab_size: 16384

model:
  type: "deberta"  # Change: gpt, deberta
  model_size: "small"

  # Add DeBERTa-specific config, remove GPT config
  deberta:
    position_buckets: -1
    relative_attention: true
    max_relative_positions: -1
```

**Note**: Remove `gpt:` section and add `deberta:` section when using DeBERTa models.

### 4. Data Source Distribution Experiments

**Purpose**: Test developmentally plausible data distributions

**Base**: Use best-performing model from above experiments

**Example**: `configs/data_mixing/10M_dev_plausible_config.yaml`
```yaml
experiment:
  name: "best_model_developmental_sources"
  tags: ["source-experiment", "developmental"]

data:
  # Modify source ratios
  train_source_ratios:
    childrens_books: 0.60    # Increase children's content
    indicdialogue: 0.25      # Conversational input
    wikipedia: 0.10          # Formal text
    indiccorp: 0.05          # News/web content

  sources:
    childrens_books:
      enabled: true
      ratio: 0.60
      max_stories: null  # Use all available

    indicdialogue:
      enabled: true
      ratio: 0.25

    wikipedia:
      enabled: true
      ratio: 0.10

    indiccorp:
      enabled: true
      ratio: 0.05
```

**Other Distribution Patterns** to test:
- **Balanced**: Equal ratios (0.25 each)
- **Formal-Heavy**: IndicCorp 0.60, Wikipedia 0.30, others 0.10
- **Conversational-Heavy**: IndicDialogue 0.60, Children's 0.30, others 0.10

## ExperimentManager

**Location**: `src/utils/experiment_config.py:372`

**Purpose**: Generate multiple experiment configurations programmatically

### Methods

#### `create_tokenization_experiments()` (line 377)

```python
def create_tokenization_experiments(self) -> List[ExperimentConfig]:
    """Create experiments for different tokenization strategies"""
    tokenizers = ["sentencepiece", "wordpiece", "bpe"]
    experiments = []

    for tokenizer in tokenizers:
        config = self.base_config.__class__(**self.base_config.__dict__)
        config.tokenizer_type = tokenizer
        config.experiment_name = f"tokenization_{tokenizer}"
        experiments.append(config)

    return experiments
```

#### `create_model_architecture_experiments()` (line 390)

```python
def create_model_architecture_experiments(self) -> List[ExperimentConfig]:
    """Create experiments for different model architectures"""
    architectures = ["gpt", "deberta"]
    experiments = []

    for arch in architectures:
        config = self.base_config.__class__(**self.base_config.__dict__)
        config.model_type = arch
        config.experiment_name = f"architecture_{arch}"
        experiments.append(config)

    return experiments
```

**Usage Example**:
```python
from src.utils.experiment_config import ExperimentConfig, ExperimentManager

# Load base config
base_config = ExperimentConfig.load_config('configs/base_config.yaml')

# Create experiment manager
manager = ExperimentManager(base_config)

# Generate tokenization experiments
tokenization_exps = manager.create_tokenization_experiments()

# Run each experiment
for config in tokenization_exps:
    # Modify vocab_size, model_type, etc.
    config.vocab_size = 16384
    config.save_config(f'configs/generated/{config.experiment_name}.yaml')
```

## Configuration Best Practices

### 1. Naming Conventions

**Experiment Names**: Use descriptive, structured names
```yaml
experiment_name: "{model}_{{tokenizer}_data_size}_{vocab_size}"
# Examples:
# - "gpt_bpe_10M_8K"
# - "deberta_unigram_100M_16K"
```

### 2. Directory Organization

**Organize experiments by category**:
```
configs/
├── base_config.yaml
├── vocab_experiments/
│   ├── gpt_bpe_8k.yaml
│   ├── gpt_bpe_16k.yaml
│   └── gpt_bpe_32k.yaml
├── tokenization_experiments/
│   ├── gpt_bpe.yaml
│   ├── gpt_unigram.yaml
│   ├── deberta_bpe.yaml
│   └── deberta_wordpiece.yaml
└── data_mixing/
    ├── developmental.yaml
    ├── balanced.yaml
    └── formal_heavy.yaml
```

### 3. Version Control

**Track configurations in git**:
```bash
git add configs/
git commit -m "Add vocabulary size experiment configs"
```

### 4. Configuration Documentation

**Document each experiment** with comments:
```yaml
# gpt_bpe_8k.yaml
# Experiment: Vocabulary Size Impact
# Description: Test GPT with 8K BPE vocabulary on 10M word corpus
# Purpose: Establish baseline for vocab size experiments
# Related: gpt_bpe_16k.yaml, gpt_bpe_32k.yaml

experiment:
  name: "gpt_bpe_10M_8K"
  description: "GPT small with 8K BPE vocabulary"
  tags: ["vocab-experiment", "baseline", "8K"]
```

### 5. Loading and Using Configurations

### From Python

```python
from src.utils.experiment_config import ExperimentConfig

# Load from file
config = ExperimentConfig.load_config('configs/base_config.yaml')

# Access fields
print(f"Experiment: {config.experiment_name}")
print(f"Model: {config.model_type}")
print(f"Vocab size: {config.vocab_size}")

# Modify and save
config.vocab_size = 16384
config.experiment_name = "gpt_10M_bpe_16K"
config.save_config('configs/vocab_experiments/gpt_bpe_16k.yaml')

# Get experiment paths
tokenizer_path = config.get_tokenizer_path()
model_path = config.get_model_path()
results_path = config.get_results_path()
```

### From Command Line

```bash
# Run with specific config
python main.py --config configs/base_config.yaml \
    --stage all \
    --experiment_name my_experiment

# Override config values
python main.py --config configs/base_config.yaml \
    --stage train \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --vocab_size 16384

# Run vocabulary size experiment
python main.py --config configs/vocab_experiments/gpt_bpe_16k.yaml \
    --stage all

# Run data source experiment
python main.py --config configs/source_experiments/developmental.yaml \
    --stage all
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
export WANDB_ENTITY=your-username

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
1. File path is correct (absolute or relative to current directory)
2. YAML syntax is valid (use online YAML validator)
3. Required fields are present
4. No tabs in YAML file (use spaces only)

### Issue: Parameter not taking effect

**Check**:
1. Parameter name matches exactly (case-sensitive)
2. Parameter is in correct section of YAML
3. Not overridden by command-line argument
4. Field exists in ExperimentConfig dataclass

### Issue: Out of memory during training

**Solutions**:
1. Reduce `batch_size` (try 16 or 8)
2. Reduce `max_length` (try 256 instead of 512)
3. Reduce model size (`hidden_size`, `num_layers`)
4. Enable gradient checkpointing: `gradient_checkpointing: true`
5. Reduce `num_workers` (try 2 or 4)
6. Use mixed precision: `dtype: "bf16"` or `dtype: "fp16"`

### Issue: Model-specific config not loading

**Check**:
1. For GPT models, ensure `gpt:` section exists
2. For DeBERTa models, ensure `deberta:` section exists
3. Remove configs for other model types (e.g., remove `gpt:` when using DeBERTa)
4. Check that `model.type` matches the model-specific config section

## Related Documentation

- [Data Processing Documentation](02_DATA_PROCESSING.md)
- [Tokenization Documentation](03_TOKENIZATION.md)
- [Model Architecture Documentation](04_MODELS.md)
- [Training Pipeline Documentation](05_TRAINING.md)
- [Evaluation Documentation](06_EVALUATION.md)

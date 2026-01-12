# Model Architectures
<!-- Updated: 2025-12-01 -->

## Overview

This project implements two transformer-based model architectures for Hindi language modeling:

1. **GPT-2**: Autoregressive (causal) language modeling
2. **DeBERTa**: Disentangled attention with enhanced mask decoding

Additionally, classification adapters wrap these base models for downstream tasks (IndicGLUE evaluation).

Each architecture is designed to explore different aspects of language understanding and generation with limited training data (~10M words / ~100M words).

## Architecture Comparison

| Feature | GPT-2 | DeBERTa |
|---------|-------|---------|
| **Objective** | Causal LM | Masked LM |
| **Attention** | Causal (unidirectional) | Disentangled Bidirectional |
| **Position Encoding** | Learned absolute | Relative (disentangled) |
| **Generation** | Yes | No |
| **Understanding** | Moderate | Strong |
| **Training** | Simpler | More complex |
| **Parameters** | ~110M (Small) | ~110M (Small) |
| **Use Case** | Text generation, LM | Classification, understanding tasks |
| **Key Innovation** | Standard autoregressive | Disentangled attention mechanism |

## Model Factory

**Location**: `src/models/model_factory.py:10`

**Purpose**: Factory class for creating, saving, and loading models. Models are saved under `results/{experiment_name}/models/` directory.

### Usage

```python
from src.models.model_factory import ModelFactory

# Create factory with configuration
factory = ModelFactory(config)

# Create model
model = factory.create_model(vocab_size=32000)

# Wrap model with classification head for downstream tasks
classification_model = factory.wrap_for_classification(
    model=model,
    num_classes=14,
    dropout=0.1,
    freeze_base=False,
    pooling_strategy='auto'  # 'auto' uses 'last' for GPT, 'first' for DeBERTa
)

# Save model
factory.save_model(model, tokenizer, checkpoint_name="my_model", metrics=metrics)

# Load model
model = factory.load_model(checkpoint_path, vocab_size=32000)

# Or load by experiment name (looks in results/{experiment_name}/models/)
model = factory.load_trained_model(experiment_name="hindi_babylm_baseline")

# Get model information
info = factory.get_model_info(model)
# Returns: {'model_type', 'total_parameters', 'trainable_parameters', 'hidden_size', ...}
```

### Key Methods

#### `create_model(vocab_size)` (line 25)
```python
def create_model(self, vocab_size: int):
    """
    Create a model based on config

    Args:
        vocab_size: Size of vocabulary

    Returns:
        Model instance (HindiGPTModel or HindiDeBERTaModel)

    Model type determined by config.model_type:
        - "gpt": GPT-2 style autoregressive model
        - "deberta": DeBERTa style masked language model with disentangled attention
    """
```

Prints model statistics:
- Model size preset (tiny, small, medium, base, large)
- Vocabulary size
- Hidden size
- Number of layers
- Number of attention heads
- Total parameters
- Trainable parameters

The method also handles model-specific configurations from nested config objects:
- GPT: `use_cache`, `scale_attn_weights`, `reorder_and_upcast_attn`
- DeBERTa: `position_buckets`, `relative_attention`, `max_relative_positions`, `pooler_hidden_size`

#### `wrap_for_classification(model, num_classes, dropout, freeze_base, pooling_strategy)` (line 97)
```python
def wrap_for_classification(
    self, model, num_classes: int, dropout: float = 0.1,
    freeze_base: bool = False, pooling_strategy: str = 'auto'
):
    """
    Wrap a language model with a classification head for sequence classification tasks.

    Args:
        model: Pre-trained language model (HindiGPTModel or HindiDeBERTaModel)
        num_classes: Number of classification classes
        dropout: Dropout probability for classification head (default: 0.1)
        freeze_base: If True, freeze the language model parameters
        pooling_strategy: Pooling strategy ('auto', 'mean', 'max', 'first', 'last')
                         'auto' → 'last' for GPT, 'first' for DeBERTa

    Returns:
        Model wrapped with classification head (GPTForSequenceClassification or
        DeBERTaForSequenceClassification)
    """
```

#### `save_model(model, tokenizer, checkpoint_name, metrics)` (line 143)
```python
def save_model(self, model, tokenizer, checkpoint_name: Optional[str] = None,
               metrics: Optional[Dict[str, float]] = None):
    """
    Save model checkpoint with metadata

    Saves to: results/{experiment_name}/models/{checkpoint_name}.pt

    Checkpoint includes:
        - Model state dict
        - Model type
        - Vocabulary size
        - Configuration (using config.to_clean_dict())
        - Experiment name
        - Metrics (if provided)
    """
```

#### `save_checkpoint(model, optimizer, epoch, step, metrics)` (line 284)
```python
def save_checkpoint(self, model, optimizer, epoch: int, step: int,
                   metrics: Dict[str, float]):
    """
    Save training checkpoint with optimizer state

    Saves to: results/{experiment_name}/models/epoch{epoch}_step{step}.pt

    Used during training to save intermediate checkpoints
    Includes optimizer state for resuming training
    """
```

#### `load_model(checkpoint_path, vocab_size)` (line 168)
```python
def load_model(self, checkpoint_path: str, vocab_size: Optional[int] = None):
    """
    Load model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        vocab_size: Optional vocabulary size (auto-detected if not provided)

    Restores:
        - Model architecture
        - Trained weights
        - Configuration (with backward compatibility migration)

    Returns model (not in eval mode - caller should set mode)
    """
```

#### `load_trained_model(experiment_name)` (line 235)
```python
def load_trained_model(self, experiment_name: str):
    """
    Load a trained model by experiment name

    Args:
        experiment_name: Name of the experiment

    Looks for checkpoints in:
        1. results/{experiment_name}/models/final.pt
        2. results/{experiment_name}/models/best.pt

    Automatically loads tokenizer from multiple possible paths to extract vocab_size

    Returns:
        Loaded model
    """
```

#### `get_model_info(model)` (line 329)
```python
def get_model_info(self, model) -> Dict[str, Any]:
    """
    Get information about the model

    Returns:
        Dictionary with:
            - model_type: 'gpt' or 'deberta'
            - total_parameters: Total number of parameters
            - trainable_parameters: Number of trainable parameters
            - hidden_size, num_layers, num_heads, max_length
    """
```

## 1. GPT-2 Style Model

### Overview

**Location**: `src/models/gpt_model.py:14`

**Class**: `HindiGPTModel`

**Objective**: Autoregressive (causal) language modeling

**Training Task**: Predict next token given previous tokens

**Formula**:
```
P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ | x₁, ..., xᵢ₋₁)
```

**Available Model Sizes** (`MODEL_SIZES` presets at line 18):
- **tiny**: 512 hidden, 6 layers, 8 heads, 2048 intermediate (~50M params)
- **small**: 768 hidden, 12 layers, 12 heads, 3072 intermediate (~110M params) - Default
- **medium**: 1024 hidden, 24 layers, 16 heads, 4096 intermediate (~350M params)

### Architecture

```
Input Tokens
    ↓
Token Embeddings + Position Embeddings
    ↓
┌─────────────────────────┐
│  Transformer Layers     │
│  (with Causal Mask)     │
│                         │
│  ┌─────────────────┐   │
│  │ Self-Attention  │   │ ← Causal (looks only left)
│  └────────┬────────┘   │
│           ↓             │
│  ┌─────────────────┐   │
│  │   Feed-Forward  │   │
│  └────────┬────────┘   │
│           ↓             │
│  [Repeated 12x]         │
└─────────┬───────────────┘
          ↓
  Language Model Head
          ↓
  Next Token Probabilities
```

### Implementation

```python
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, Any, Optional

class HindiGPTModel(nn.Module):
    """GPT model for Hindi with configurable sizes"""

    # Model size presets
    MODEL_SIZES = {
        'tiny': {
            'hidden_size': 512,
            'num_layers': 6,
            'num_heads': 8,
            'intermediate_size': 2048,
        },
        'small': {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'intermediate_size': 3072,
        },
        'medium': {
            'hidden_size': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'intermediate_size': 4096,
        },
    }

    def __init__(self, vocab_size: int, config: Dict[str, Any]):
        """
        Initialize Hindi GPT model

        Args:
            vocab_size: Size of vocabulary
            config: Configuration dictionary with model parameters
        """
        super().__init__()

        # Get model size preset if specified
        model_size = config.get('model_size', 'small')
        if model_size in self.MODEL_SIZES:
            size_preset = self.MODEL_SIZES[model_size]
            # Apply preset, but allow config to override
            hidden_size = config.get('hidden_size', size_preset['hidden_size'])
            num_layers = config.get('num_layers', size_preset['num_layers'])
            num_heads = config.get('num_heads', size_preset['num_heads'])
            intermediate_size = config.get('intermediate_size', size_preset['intermediate_size'])
        else:
            # Use config values or defaults
            hidden_size = config.get('hidden_size', 768)
            num_layers = config.get('num_layers', 12)
            num_heads = config.get('num_heads', 12)
            intermediate_size = config.get('intermediate_size', 3072)

        # Create HuggingFace GPT2 config
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=config.get('max_length', 512),
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_inner=intermediate_size,
            activation_function=config.get('activation', 'gelu'),
            resid_pdrop=config.get('dropout', 0.1),
            embd_pdrop=config.get('dropout', 0.1),
            attn_pdrop=config.get('dropout', 0.1),
            use_cache=config.get('use_cache', True),
        )

        # Create the model
        self.model = GPT2LMHeadModel(self.config)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for language modeling [batch, seq_len]

        Returns:
            Model outputs with logits and optional loss
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate(self, input_ids: torch.Tensor,
                max_length: int = 50,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                **kwargs):
        """
        Generate text autoregressively

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_length: Maximum total length (input + generated)
            temperature: Sampling temperature
            top_k: Top-k filtering (optional)
            top_p: Top-p (nucleus) sampling (optional)
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs [batch, total_length]
        """
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.config.pad_token_id,
            **kwargs
        )

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Count number of parameters

        Args:
            only_trainable: If True, count only trainable parameters

        Returns:
            Total number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Convenience function for creating model
def create_gpt_model(vocab_size: int, model_size: str = 'small', **kwargs) -> HindiGPTModel:
    """
    Create a GPT model with specified size

    Args:
        vocab_size: Vocabulary size
        model_size: Model size ('tiny', 'small', 'medium')
        **kwargs: Additional config overrides

    Returns:
        HindiGPTModel instance
    """
    config = {'model_size': model_size, **kwargs}
    return HindiGPTModel(vocab_size, config)
```

### Configuration

```python
GPT2Config(
    vocab_size=32000,           # Tokenizer vocabulary size
    n_positions=512,            # Maximum sequence length
    n_embd=768,                 # Hidden size (embedding dimension)
    n_layer=12,                 # Number of transformer layers
    n_head=12,                  # Number of attention heads
    n_inner=3072,               # Feed-forward dimension (4 * n_embd)
    activation_function='gelu', # Activation function
    resid_pdrop=0.1,           # Residual dropout
    embd_pdrop=0.1,            # Embedding dropout
    attn_pdrop=0.1,            # Attention dropout
    layer_norm_epsilon=1e-5,   # Layer norm epsilon
    initializer_range=0.02,    # Weight initialization range
)
```

### Key Features

1. **Causal Self-Attention**:
   - Attends only to previous tokens
   - Implements autoregressive property
   - Masked attention matrix (lower triangular)

2. **Position Embeddings**:
   - Learned absolute position embeddings
   - Max sequence length: 512 tokens

3. **Layer Normalization**:
   - Pre-LN (before attention and FFN)
   - Stabilizes training

4. **Residual Connections**:
   - Around each sub-layer
   - Facilitates gradient flow

### Parameters

Parameter counts for different sizes (vocab_size=32000):

| Size   | Hidden | Layers | Heads | Intermediate | Approx Parameters |
|--------|--------|--------|-------|-------------|-------------------|
| **Tiny**   | 512    | 6      | 8     | 2048        | ~50M              |
| **Small**  | 768    | 12     | 12    | 3072        | ~110M             |
| **Medium** | 1024   | 24     | 16    | 4096        | ~350M             |

Use `model.num_parameters()` to get exact counts.

### Training

**Loss Function**: Cross-Entropy Loss
```python
loss = CrossEntropyLoss(logits, labels)
# labels = input_ids shifted by 1 position
```

**Masking**: Causal mask ensures token i can only attend to tokens 1..i-1

### Generation

```python
from src.models.gpt_model import create_gpt_model

# Create model
model = create_gpt_model(vocab_size=32000, model_size='small', max_length=512)
model.eval()

# Prepare prompt
prompt = "मैं एक"
input_ids = tokenizer.encode(prompt)

# Generate text with top-k and top-p sampling
output_ids = model.generate(
    input_ids=torch.tensor([input_ids]),
    max_length=50,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)

generated_text = tokenizer.decode(output_ids[0])
# Output: "मैं एक विद्यार्थी हूँ और मैं विश्वविद्यालय में पढ़ता हूँ..."

# Check parameter count
print(f"Total parameters: {model.num_parameters():,}")
print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
```

### Pros and Cons

**Pros**:
- ✅ Natural for text generation
- ✅ Simpler training (no masking needed)
- ✅ Good for creative tasks
- ✅ Efficient inference (kv-caching)

**Cons**:
- ❌ Unidirectional context
- ❌ Less effective for understanding tasks
- ❌ Cannot use future context

## 2. DeBERTa Model
<!-- Updated: 2025-01-11 -->

### Overview

**Location**: `src/models/deberta_model.py:19`

**Class**: `HindiDeBERTaModel`

**Objective**: Masked language modeling with disentangled attention

**Training Task**: Predict masked tokens using bidirectional context with enhanced position encoding

**Key Innovation**: DeBERTa (Decoding-enhanced BERT with disentangled attention) improves upon BERT:
- **Disentangled attention mechanism**: Separately encodes content and position
- **Enhanced mask decoder**: Better handling of absolute positions for predictions
- **Relative position encoding**: More flexible position representation (DebertaV2)

**Formula**:
```
P(xᵢ | x₁, ..., xᵢ₋₁, xᵢ₊₁, ..., xₙ) for masked positions
with disentangled content-position attention
```

**Available Model Sizes** (`MODEL_SIZES` presets at line 23):
- **tiny**: 384 hidden, 6 layers, 6 heads, 1536 intermediate (~22M params)
- **small**: 768 hidden, 12 layers, 12 heads, 3072 intermediate (~86M params) - Default
- **base**: 768 hidden, 12 layers, 12 heads, 3072 intermediate (~86M params, same as small)
- **large**: 1024 hidden, 24 layers, 16 heads, 4096 intermediate (~304M params)

### Architecture

```
Input Tokens (with [MASK])
    ↓
Token Embeddings (content)
    ↓
┌───────────────────────────────┐
│  DeBERTa Transformer Layers   │
│  (Disentangled Attention)     │
│                               │
│  ┌─────────────────────────┐ │
│  │ Disentangled Attention  │ │ ← Separate content & position
│  │  - Content-to-Content   │ │
│  │  - Content-to-Position  │ │
│  │  - Position-to-Content  │ │
│  └────────┬────────────────┘ │
│           ↓                   │
│  ┌─────────────────┐         │
│  │   Feed-Forward  │         │
│  └────────┬────────┘         │
│           ↓                   │
│  [Repeated 12x]               │
└─────────┬─────────────────────┘
          ↓
  Enhanced Mask Decoder
  (with absolute positions)
          ↓
  Masked Token Predictions
```

**Disentangled Attention**: Unlike BERT's standard attention, DeBERTa computes attention scores using three separate components:
1. **Content-to-Content** (C2C): How each word attends to other words
2. **Content-to-Position** (C2P): How each word attends to positions
3. **Position-to-Content** (P2C): How positions influence word attention

This allows the model to better capture both semantic and positional information.

### Implementation

```python
import torch
import torch.nn as nn
from transformers import DebertaV2Config, DebertaV2ForMaskedLM
from typing import Dict, Any, Optional

class HindiDeBERTaModel(nn.Module):
    """DeBERTa model for Hindi with configurable sizes"""

    # Model size presets
    MODEL_SIZES = {
        'tiny': {
            'hidden_size': 384,
            'num_layers': 6,
            'num_heads': 6,
            'intermediate_size': 1536,
        },
        'small': {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'intermediate_size': 3072,
        },
        'base': {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'intermediate_size': 3072,
        },
        'large': {
            'hidden_size': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'intermediate_size': 4096,
        },
    }

    def __init__(self, vocab_size: int, config: Dict[str, Any]):
        """
        Initialize Hindi DeBERTa model

        Args:
            vocab_size: Size of vocabulary
            config: Configuration dictionary with model parameters
        """
        super().__init__()

        # Get model size preset if specified
        model_size = config.get('model_size', 'small')
        if model_size in self.MODEL_SIZES:
            size_preset = self.MODEL_SIZES[model_size]
            # Apply preset, but allow config to override
            hidden_size = config.get('hidden_size', size_preset['hidden_size'])
            num_layers = config.get('num_layers', size_preset['num_layers'])
            num_heads = config.get('num_heads', size_preset['num_heads'])
            intermediate_size = config.get('intermediate_size', size_preset['intermediate_size'])
        else:
            # Use config values or defaults
            hidden_size = config.get('hidden_size', 768)
            num_layers = config.get('num_layers', 12)
            num_heads = config.get('num_heads', 12)
            intermediate_size = config.get('intermediate_size', 3072)

        # Get DeBERTa-specific parameters
        max_relative_positions = config.get('max_relative_positions', -1)
        if max_relative_positions == -1:
            max_relative_positions = config.get('max_length', 512)

        # Create HuggingFace DeBERTaV2 config
        self.config = DebertaV2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=config.get('max_length', 512),
            hidden_dropout_prob=config.get('dropout', 0.1),
            attention_probs_dropout_prob=config.get('dropout', 0.1),
            # DeBERTa-specific parameters
            position_buckets=config.get('position_buckets', 256),
            relative_attention=config.get('relative_attention', True),
            max_relative_positions=max_relative_positions,
            pooler_hidden_size=config.get('pooler_hidden_size', hidden_size),
            pooler_dropout=config.get('pooler_dropout', 0.1),
            pooler_hidden_act=config.get('pooler_hidden_act', 'gelu'),
        )

        # Create the model
        self.model = DebertaV2ForMaskedLM(self.config)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for masked language modeling [batch, seq_len]

        Returns:
            Model outputs with logits and optional loss
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Count number of parameters

        Args:
            only_trainable: If True, count only trainable parameters

        Returns:
            Total number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Convenience function for creating model
def create_deberta_model(vocab_size: int, model_size: str = 'small', **kwargs) -> HindiDeBERTaModel:
    """
    Create a DeBERTa model with specified size

    Args:
        vocab_size: Vocabulary size
        model_size: Model size ('tiny', 'small', 'base', 'large')
        **kwargs: Additional config overrides

    Returns:
        HindiDeBERTaModel instance
    """
    config = {'model_size': model_size, **kwargs}
    return HindiDeBERTaModel(vocab_size, config)
```

### Configuration

```python
DebertaV2Config(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,                 # 4 * hidden_size
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    # DeBERTa-specific parameters
    position_buckets=256,                   # Number of position buckets
    relative_attention=True,                # Enable relative attention
    max_relative_positions=512,             # Max relative positions (-1 uses max_length)
    pooler_hidden_size=768,                 # Pooler hidden size
    pooler_dropout=0.1,                     # Pooler dropout
    pooler_hidden_act='gelu',               # Pooler activation function
    initializer_range=0.02,
    layer_norm_eps=1e-7,                    # DeBERTa uses 1e-7 (vs BERT's 1e-12)
)
```

**Key DeBERTa Parameters**:
- `position_buckets`: Number of buckets for relative position encoding (default: 256)
- `relative_attention`: Enable disentangled attention with relative positions (default: True)
- `max_relative_positions`: Maximum relative positions to consider (default: -1, uses max_length)
- `pooler_*`: Parameters for the pooler layer used in classification tasks

### Key Features

1. **Bidirectional Self-Attention**:
   - Attends to all tokens (no causal mask)
   - Captures full context
   - Better for understanding tasks

2. **Masked Language Modeling**:
   - Random masking of input tokens
   - Predict masked tokens from context
   - 15% masking rate (standard)

3. **Embeddings**:
   - Token embeddings
   - Position embeddings (learned)
   - Segment embeddings (for sentence pairs)

4. **Special Tokens**:
   - `[CLS]`: Classification token (sentence representation)
   - `[SEP]`: Separator between sentences
   - `[MASK]`: Masked token placeholder

### Masking Strategy

Standard BERT masking:
```
Original: मैं विश्वविद्यालय जा रहा हूँ
Masked:   मैं [MASK] जा रहा हूँ

Task: Predict "विश्वविद्यालय" using bidirectional context
```

**Masking Details** (15% of tokens):
- 80%: Replace with `[MASK]`
- 10%: Replace with random token
- 10%: Keep original

### Training

**Loss Function**: Cross-Entropy on masked positions only
```python
loss = CrossEntropyLoss(logits[masked_positions], labels[masked_positions])
```

### Fine-tuning

BERT based models can be fine-tuned for various tasks:

**Classification**:
```python
# Use [CLS] token representation
cls_output = model(input_ids)[0][:, 0, :]  # [batch_size, hidden_size]
logits = classifier(cls_output)  # Task-specific head
```

**Token Classification** (NER, POS):
```python
# Use all token representations
token_outputs = model(input_ids)[0]  # [batch_size, seq_len, hidden_size]
logits = token_classifier(token_outputs)
```

### Pros and Cons

**Pros**:
- ✅ Bidirectional context
- ✅ Excellent for understanding tasks
- ✅ Strong representations
- ✅ Pre-training → fine-tuning paradigm

**Cons**:
- ❌ Cannot directly generate text
- ❌ More complex training (masking)
- ❌ Slower inference than GPT

## 3. Classification Model Adapters
<!-- Updated: 2025-01-11 -->

### Overview

**Location**: `src/models/classification_models.py`

**Classes**: `GPTForSequenceClassification`, `DeBERTaForSequenceClassification`

**Helper Function**: `wrap_model_for_classification()` (line 332)

**Purpose**: Adapt language models for sequence-level classification tasks (IndicGLUE evaluation)

**Architecture**: Language Model + Pooling Layer + Dropout + Classification Head

**Key Features**:
- **Label Smoothing**: Configurable label smoothing for regularization
- **Mixed Precision Support**: Automatic dtype matching for BFloat16/Float16
- **Flexible Pooling**: Multiple pooling strategies for DeBERTa
- **Base Model Freezing**: Option to freeze language model and only train classification head

### Architecture

**Problem**: Language models output per-token predictions `[batch, seq_len, hidden_size]`, but classification needs per-example predictions `[batch, num_classes]`.

**Solution**: Add pooling + classification head

```
Input Sequence: [batch, seq_len]
    ↓
Language Model (GPT or DeBERTa)
    ↓
Hidden States: [batch, seq_len, hidden_size]
    ↓
Pooling Layer (reduce sequence dimension)
    ↓
Pooled Output: [batch, hidden_size]
    ↓
Dropout
    ↓
Classification Head (Linear)
    ↓
Logits: [batch, num_classes]
```

### A. GPTForSequenceClassification

**Location**: `classification_models.py:29`

**Pooling Strategy**: Last-token pooling (causal LM pattern)

- Label smoothing support
- Automatic dtype matching with base model
- Better hidden state extraction across different model types

```python
class GPTForSequenceClassification(nn.Module):
    """
    GPT model with classification head for sequence-level classification.

    Uses last-token pooling since GPT is a causal language model where
    the last token has seen the full sequence context.
    """

    def __init__(self,
                 lm_model: nn.Module,
                 num_classes: int,
                 hidden_size: int,
                 dropout: float = 0.1,
                 freeze_base: bool = False,
                 label_smoothing: float = 0.0):
        """
        Args:
            lm_model: Pre-trained GPT language model
            num_classes: Number of classification classes
            hidden_size: Hidden size of the language model
            dropout: Dropout probability for classification head
            freeze_base: If True, freeze the language model parameters
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()

        self.lm_model = lm_model
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.label_smoothing = label_smoothing

        # Freeze base model if requested
        if freeze_base:
            for param in self.lm_model.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classification head
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        # Match dtype of base model to avoid dtype mismatch errors
        self._match_base_model_dtype()

    def _match_base_model_dtype(self):
        """Match the dtype of the classification head to the base model"""
        # Get dtype from first parameter of base model
        base_dtype = next(self.lm_model.parameters()).dtype
        # Convert classification head to same dtype
        self.classifier = self.classifier.to(dtype=base_dtype)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> ClassificationOutput:
        """
        Forward pass for classification.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Classification labels [batch] (optional)

        Returns:
            ClassificationOutput with logits [batch, num_classes]
        """
        # Get language model outputs
        lm_outputs = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract hidden states: [batch, seq_len, hidden_size]
        # Handles different model output formats
        if hasattr(lm_outputs, 'hidden_states') and lm_outputs.hidden_states is not None:
            hidden_states = lm_outputs.hidden_states[-1]
        elif hasattr(lm_outputs, 'last_hidden_state'):
            hidden_states = lm_outputs.last_hidden_state
        else:
            # For GPT2LMHeadModel, access transformer outputs directly
            transformer_outputs = self.lm_model.model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = transformer_outputs.last_hidden_state

        # Last token pooling: [batch, hidden_size]
        # For causal LM, the last token has seen the full context
        if attention_mask is not None:
            # Get the last non-padding token for each example
            sequence_lengths = attention_mask.sum(dim=1).long() - 1
            batch_size = hidden_states.shape[0]
            pooled_output = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]
        else:
            # If no attention mask, just use the last token
            pooled_output = hidden_states[:, -1, :]

        # Apply classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch, num_classes]

        # Compute loss if labels provided (with label smoothing)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fct(logits, labels)

        return ClassificationOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            pooled_output=pooled_output
        )
```

**Why Last-Token Pooling?**
- GPT is causal: last token has seen the entire sequence
- Natural for autoregressive models
- Matches how GPT processes sequences left-to-right

### B. DeBERTaForSequenceClassification

**Location**: `classification_models.py:156`

**Pooling Strategies**: Multiple options (mean, max, first, last)

**New in this version**:
- Label smoothing support
- Automatic dtype matching with base model
- Mixed precision support in pooling operations
- Better hidden state extraction across different model types

```python
class DeBERTaForSequenceClassification(nn.Module):
    """
    DeBERTa model with classification head for sequence-level classification.

    Uses flexible pooling strategies since DeBERTa is a masked language model
    with bidirectional context.
    """

    def __init__(self,
                 lm_model: nn.Module,
                 num_classes: int,
                 hidden_size: int,
                 dropout: float = 0.1,
                 pooling_strategy: str = 'mean',
                 freeze_base: bool = False,
                 label_smoothing: float = 0.0):
        """
        Args:
            lm_model: Pre-trained DeBERTa language model
            num_classes: Number of classification classes
            hidden_size: Hidden size of the language model
            dropout: Dropout probability for classification head
            pooling_strategy: Pooling strategy ('mean', 'max', 'first', 'last')
            freeze_base: If True, freeze the language model parameters
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()

        self.lm_model = lm_model
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy
        self.label_smoothing = label_smoothing

        # Freeze base model if requested
        if freeze_base:
            for param in self.lm_model.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classification head
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        # Match dtype of base model to avoid dtype mismatch errors
        self._match_base_model_dtype()

    def _match_base_model_dtype(self):
        """Match the dtype of the classification head to the base model"""
        # Get dtype from first parameter of base model
        base_dtype = next(self.lm_model.parameters()).dtype
        # Convert classification head to same dtype
        self.classifier = self.classifier.to(dtype=base_dtype)

    def pool_hidden_states(self,
                          hidden_states: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool hidden states to get sequence representation.

        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled representation [batch, hidden_size]
        """
        if self.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            if attention_mask is not None:
                # Expand mask for broadcasting: [batch, seq_len, 1]
                # Match dtype to hidden_states to support mixed precision (BFloat16/Float16)
                mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                # Sum over sequence length, weighted by mask
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                # Divide by number of non-padding tokens
                sum_mask = mask_expanded.sum(dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
                pooled = sum_hidden / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)

        elif self.pooling_strategy == 'max':
            # Max pooling
            if attention_mask is not None:
                # Set padding positions to large negative value
                # Match dtype to hidden_states to support mixed precision
                mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                hidden_states = hidden_states.clone()
                hidden_states[mask_expanded == 0] = -1e9
            pooled = hidden_states.max(dim=1)[0]

        elif self.pooling_strategy == 'first':
            # First token (CLS-like)
            pooled = hidden_states[:, 0, :]

        elif self.pooling_strategy == 'last':
            # Last non-padding token
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1).long() - 1
                batch_size = hidden_states.shape[0]
                pooled = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    sequence_lengths
                ]
            else:
                pooled = hidden_states[:, -1, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return pooled

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> ClassificationOutput:
        """
        Forward pass for classification.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Classification labels [batch] (optional)

        Returns:
            ClassificationOutput with logits [batch, num_classes]
        """
        # Get language model outputs
        lm_outputs = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract hidden states: [batch, seq_len, hidden_size]
        # Handles different model output formats
        if hasattr(lm_outputs, 'hidden_states') and lm_outputs.hidden_states is not None:
            hidden_states = lm_outputs.hidden_states[-1]
        elif hasattr(lm_outputs, 'last_hidden_state'):
            hidden_states = lm_outputs.last_hidden_state
        else:
            # For DebertaV2ForMaskedLM, access the base model
            base_outputs = self.lm_model.model.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = base_outputs.last_hidden_state

        # Pool to get sequence representation: [batch, hidden_size]
        pooled_output = self.pool_hidden_states(hidden_states, attention_mask)

        # Apply classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch, num_classes]

        # Compute loss if labels provided (with label smoothing)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fct(logits, labels)

        return ClassificationOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            pooled_output=pooled_output
        )
```

**Pooling Strategy Comparison**:

| Strategy | Description | Use Case | Notes |
|----------|-------------|----------|-------|
| **Mean** | Attention-weighted average | General purpose, stable | Default for DeBERTa |
| **Max** | Maximum across sequence | Focus on salient features | May ignore context |
| **First** | [CLS] token (first position) | Following BERT convention | Used by 'auto' for DeBERTa |
| **Last** | Last non-padding token | Similar to GPT approach | Used by 'auto' for GPT |
| **Auto** | Selects best for model type | Recommended | 'first' for DeBERTa, 'last' for GPT |

**Recommendation**: Use `pooling_strategy='auto'` to automatically select the appropriate pooling for your model type.

### C. wrap_model_for_classification Helper Function

**Location**: `classification_models.py:332`

```python
def wrap_model_for_classification(
    lm_model: nn.Module,
    model_type: str,
    num_classes: int,
    hidden_size: int,
    dropout: float = 0.1,
    freeze_base: bool = False,
    pooling_strategy: str = 'auto',
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Wrap a language model with a classification head.

    Args:
        lm_model: Pre-trained language model
        model_type: Type of model ('gpt', 'bert', or 'deberta')
        num_classes: Number of classification classes
        hidden_size: Hidden size of the language model
        dropout: Dropout probability for classification head
        freeze_base: If True, freeze the language model parameters
        pooling_strategy: Pooling strategy ('auto', 'mean', 'max', 'first', 'last')
                         'auto' → 'last' for GPT, 'first' for BERT/DeBERTa
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Model wrapped with classification head
    """
```

This is a convenience function that automatically selects the appropriate classification wrapper based on model type:
- **GPT models**: Returns `GPTForSequenceClassification` with last-token pooling
- **BERT/DeBERTa models**: Returns `DeBERTaForSequenceClassification` with flexible pooling (default: first token for 'auto')

### Usage Example

```python
from src.models.model_factory import ModelFactory
from src.models.classification_models import wrap_model_for_classification

# Method 1: Using ModelFactory (recommended)
factory = ModelFactory(config)
gpt_model = factory.create_model(vocab_size=32000)
classifier = factory.wrap_for_classification(
    model=gpt_model,
    num_classes=14,  # e.g., BBC Articles Classification
    dropout=0.1,
    freeze_base=False,  # Fine-tune entire model
    pooling_strategy='auto'  # Auto-selects 'last' for GPT
)

# Method 2: Using helper function directly
from src.models.gpt_model import create_gpt_model
gpt_model = create_gpt_model(vocab_size=32000, model_size='small')
classifier = wrap_model_for_classification(
    lm_model=gpt_model,
    model_type='gpt',
    num_classes=14,
    hidden_size=768,
    dropout=0.1,
    freeze_base=False,
    pooling_strategy='auto',
    label_smoothing=0.1  # Enable label smoothing
)

# Forward pass
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
labels = torch.tensor([3])

output = classifier(input_ids, attention_mask, labels)
print(output.logits.shape)  # [1, 14]
print(output.loss)  # scalar (with label smoothing)
print(output.pooled_output.shape)  # [1, 768]
```

### Key Features

1. **Label Smoothing**: Configurable label smoothing factor (default: 0.0) for regularization
2. **Base Model Freezing**: Option to freeze language model and only train classification head
3. **Automatic dtype Matching**: Classification head automatically matches base model dtype (supports BFloat16/Float16)
4. **Flexible Pooling**: Multiple pooling strategies for DeBERTa (mean, max, first, last, auto)
5. **Mixed Precision Support**: Pooling operations handle BFloat16/Float16 correctly
6. **Standard Output Format**: `ClassificationOutput` dataclass for consistency
   - `logits`: Classification logits [batch, num_classes]
   - `loss`: Cross-entropy loss (optional, if labels provided)
   - `hidden_states`: Full sequence hidden states [batch, seq_len, hidden_size]
   - `pooled_output`: Pooled representation [batch, hidden_size]
7. **Robust Hidden State Extraction**: Handles different model output formats automatically

### When to Use

**GPTForSequenceClassification**:
- Text generation tasks where classification is secondary
- When you want causal (unidirectional) context
- Following GPT fine-tuning paradigm

**DeBERTaForSequenceClassification**:
- Pure classification tasks (IndicGLUE)
- When bidirectional context is important
- Need flexible pooling strategies
- Following BERT/DeBERTa fine-tuning paradigm

## Model Selection Guide

| Task Type | Recommended Model | Reason |
|-----------|------------------|---------|
| **Text Generation** | GPT-2 | Natural autoregressive generation |
| **Classification** | BERT | Strong bidirectional context |
| **Named Entity Recognition** | BERT | Token-level understanding |
| **Question Answering** | BERT | Bidirectional context crucial |
| **General LM** | GPT-2 | Simpler, standard baseline |
| **Multi-Task** | Hybrid | Versatile but expensive |
| **Limited Compute** | GPT-2 or BERT | Half the parameters of Hybrid |

## Configuration Examples

### Tiny Model (for quick experiments, ~50M params)
```yaml
model:
  model_type: "gpt"
  model_size: "tiny"  # Use preset
  # Or specify manually:
  # hidden_size: 512
  # num_layers: 6
  # num_heads: 8
  # intermediate_size: 2048
  max_length: 512
  dropout: 0.1
  activation: "gelu"

# GPT-specific config (optional)
gpt_config:
  use_cache: true
  scale_attn_weights: true
  reorder_and_upcast_attn: false
```

### Small Model (default, ~110M params for GPT / ~86M for DeBERTa)
```yaml
model:
  model_type: "gpt"  # or "deberta"
  model_size: "small"  # Use preset (default)
  # Or specify manually:
  # hidden_size: 768
  # num_layers: 12
  # num_heads: 12
  # intermediate_size: 3072
  max_length: 512
  dropout: 0.1
  activation: "gelu"

# DeBERTa-specific config (only for deberta model_type)
deberta_config:
  position_buckets: 256
  relative_attention: true
  max_relative_positions: -1  # -1 means use max_length
  pooler_hidden_size: 768
  pooler_dropout: 0.1
  pooler_hidden_act: "gelu"
```

### Medium Model (for GPT, ~350M params)
```yaml
model:
  model_type: "gpt"
  model_size: "medium"
  max_length: 512
  dropout: 0.1
  activation: "gelu"
```

### Large Model (for DeBERTa, ~304M params, requires significant compute)
```yaml
model:
  model_type: "deberta"
  model_size: "large"
  max_length: 512
  dropout: 0.1

deberta_config:
  position_buckets: 256
  relative_attention: true
  max_relative_positions: -1
  pooler_hidden_size: 1024  # Match hidden_size
  pooler_dropout: 0.1
  pooler_hidden_act: "gelu"
```

**Notes**:
- Training memory includes gradients and optimizer states (≈4× model size)
- Mixed precision (BF16/FP16) can reduce memory by ~40-50%
- Gradient accumulation allows larger effective batch sizes with same memory
- Gradient checkpointing can reduce memory further with ~20% speed trade-off
- Add ~2-4 GB for data loading and system overhead
- Inference memory is for evaluation mode only

## Best Practices

### 1. Model Initialization
- Use default transformer initialization (normal distribution, std=0.02)
- Pre-trained embeddings can help (if available)

### 2. Regularization
- Dropout: 0.1 (standard)
- Weight decay: 0.01
- Gradient clipping: max_norm=1.0

### 3. Attention Patterns
- GPT: Causal mask is essential
- BERT: Full bidirectional attention
- Hybrid: Ensure correct masks for each component

### 4. Special Tokens
- Reserve IDs 0-4 for special tokens
- Ensure tokenizer and model agree on special tokens

## Related Documentation

- [Training Pipeline Documentation](05_TRAINING.md)
- [Evaluation Framework Documentation](06_EVALUATION.md)
- [Configuration Guide](07_CONFIGURATION.md)
- [API Reference](08_API_REFERENCE.md)

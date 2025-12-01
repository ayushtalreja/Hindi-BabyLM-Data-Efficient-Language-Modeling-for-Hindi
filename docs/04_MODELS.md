# Model Architectures
<!-- Updated: 2025-12-01 -->

## Overview

This project implements two transformer-based model architectures for Hindi language modeling:

1. **GPT-2**: Autoregressive (causal) language modeling
2. **DeBERTa**: Disentangled attention with enhanced mask decoding

Additionally, classification adapters wrap these base models for downstream tasks (IndicGLUE evaluation).

Each architecture is designed to explore different aspects of language understanding and generation with limited training data (~10M tokens).

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

**Purpose**: Factory class for creating, saving, and loading models.

### Usage

```python
from src.models.model_factory import ModelFactory

# Create factory with configuration
factory = ModelFactory(config)

# Create model
model = factory.create_model(vocab_size=32000)

# Save model
factory.save_model(model, tokenizer, checkpoint_name="my_model", metrics=metrics)

# Load model
model = factory.load_model(checkpoint_path)

# Or load by experiment name
model = factory.load_trained_model(experiment_name="hindi_babylm_baseline")
```

### Key Methods

#### `create_model(vocab_size)` (line 23)
```python
def create_model(self, vocab_size: int):
    """
    Create a model based on config

    Args:
        vocab_size: Size of vocabulary

    Returns:
        Model instance (HindiGPTModel, HindiBERTModel, or HybridGPTBERTModel)

    Model type determined by config.model_type:
        - "gpt": GPT-2 style autoregressive model
        - "bert": BERT style masked language model
        - "hybrid": Hybrid causal + masked model
    """
```

Prints model statistics:
- Vocabulary size
- Hidden size
- Number of layers
- Number of attention heads
- Total parameters
- Trainable parameters

#### `save_model(model, tokenizer, checkpoint_name, metrics)` (line 90)
```python
def save_model(self, model, tokenizer, checkpoint_name: Optional[str] = None,
               metrics: Optional[Dict[str, float]] = None):
    """
    Save model checkpoint with metadata

    Saves:
        - Model state dict
        - Model type
        - Vocabulary size
        - Configuration
        - Experiment name
        - Metrics (if provided)

    Output:
        - checkpoints/{checkpoint_name}.pt: Full checkpoint
        - {experiment_name}_model.pt: State dict only
    """
```

#### `save_checkpoint(model, optimizer, epoch, step, metrics)` (line 175)
```python
def save_checkpoint(self, model, optimizer, epoch: int, step: int,
                   metrics: Dict[str, float]):
    """
    Save training checkpoint with optimizer state

    Used during training to save intermediate checkpoints
    Includes optimizer state for resuming training
    """
```

#### `load_model(checkpoint_path, vocab_size)` (line 120)
```python
def load_model(self, checkpoint_path: str, vocab_size: Optional[int] = None):
    """
    Load model from checkpoint

    Restores:
        - Model architecture
        - Trained weights
        - Configuration

    Returns model in evaluation mode
    """
```

## 1. GPT-2 Style Model

### Overview

**Location**: `src/models/gpt_model.py:5`

**Class**: `HindiGPTModel`

**Objective**: Autoregressive (causal) language modeling

**Training Task**: Predict next token given previous tokens

**Formula**:
```
P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ | x₁, ..., xᵢ₋₁)
```

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

class HindiGPTModel(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=config.get('max_length', 512),
            n_embd=config.get('hidden_size', 768),
            n_layer=config.get('num_layers', 12),
            n_head=config.get('num_heads', 12),
            resid_pdrop=config.get('dropout', 0.1),
            embd_pdrop=config.get('dropout', 0.1),
            attn_pdrop=config.get('dropout', 0.1),
        )

        self.model = GPT2LMHeadModel(self.config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate(self, input_ids, max_length=50, temperature=1.0):
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.config.pad_token_id
        )
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

With standard configuration (vocab_size=32000):
- **Embedding Layer**: 32K × 768 = 24.5M
- **Transformer Layers**: 12 × 7M = 84M
- **Total**: ~110M parameters

### Training

**Loss Function**: Cross-Entropy Loss
```python
loss = CrossEntropyLoss(logits, labels)
# labels = input_ids shifted by 1 position
```

**Masking**: Causal mask ensures token i can only attend to tokens 1..i-1

### Generation

```python
model.eval()
prompt = "मैं एक"
input_ids = tokenizer.encode(prompt)

output_ids = model.generate(
    input_ids=torch.tensor([input_ids]),
    max_length=50,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

generated_text = tokenizer.decode(output_ids[0])
# Output: "मैं एक विद्यार्थी हूँ और मैं विश्वविद्यालय में पढ़ता हूँ..."
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
<!-- Updated: 2025-12-01 -->

### Overview

**Location**: `src/models/deberta_model.py:19`

**Class**: `HindiDeBERTaModel`

**Objective**: Masked language modeling with disentangled attention

**Training Task**: Predict masked tokens using bidirectional context with enhanced position encoding

**Key Innovation**: DeBERTa (Decoding-enhanced BERT with disentangled attention) improves upon BERT:
- **Disentangled attention mechanism**: Separately encodes content and position
- **Enhanced mask decoder**: Better handling of absolute positions for predictions
- **Relative position encoding**: More flexible position representation

**Formula**:
```
P(xᵢ | x₁, ..., xᵢ₋₁, xᵢ₊₁, ..., xₙ) for masked positions
with disentangled content-position attention
```

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

    def __init__(self, vocab_size: int, config: dict):
        super().__init__()

        # Get model size preset
        model_size = config.get('model_size', 'small')
        size_preset = self.MODEL_SIZES.get(model_size, self.MODEL_SIZES['small'])

        # Create DeBERTaV2 config
        self.config = DebertaV2Config(
            vocab_size=vocab_size,
            hidden_size=config.get('hidden_size', size_preset['hidden_size']),
            num_hidden_layers=config.get('num_layers', size_preset['num_layers']),
            num_attention_heads=config.get('num_heads', size_preset['num_heads']),
            intermediate_size=config.get('intermediate_size', size_preset['intermediate_size']),
            max_position_embeddings=config.get('max_length', 512),
            hidden_dropout_prob=config.get('dropout', 0.1),
            attention_probs_dropout_prob=config.get('dropout', 0.1),
            # DeBERTa-specific parameters
            position_buckets=config.get('position_buckets', 256),
            relative_attention=config.get('relative_attention', True),
            max_relative_positions=config.get('max_relative_positions', 512),
            pooler_hidden_size=config.get('pooler_hidden_size', size_preset['hidden_size']),
            pooler_dropout=config.get('pooler_dropout', 0.1),
            pooler_hidden_act=config.get('pooler_hidden_act', 'gelu'),
        )

        self.model = DebertaV2ForMaskedLM(self.config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
```

### Configuration

```python
BertConfig(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,             # 4 * hidden_size
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,                  # Segment embeddings
    initializer_range=0.02,
    layer_norm_eps=1e-12,
)
```

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

BERT can be fine-tuned for various tasks:

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
<!-- Updated: 2025-12-01 -->

### Overview

**Location**: `src/models/classification_models.py`

**Classes**: `GPTForSequenceClassification`, `DeBERTaForSequenceClassification`

**Purpose**: Adapt language models for sequence-level classification tasks (IndicGLUE evaluation)

**Architecture**: Language Model + Pooling Layer + Classification Head

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

```python
class GPTForSequenceClassification(nn.Module):
    """GPT model with classification head using last-token pooling"""

    def __init__(self,
                 lm_model: nn.Module,
                 num_classes: int,
                 hidden_size: int,
                 dropout: float = 0.1,
                 freeze_base: bool = False):
        super().__init__()

        self.lm_model = lm_model
        self.num_classes = num_classes

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

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get language model outputs
        lm_outputs = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = lm_outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Last-token pooling for causal LM
        # Find last non-padding token position
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            pooled = hidden_states[range(batch_size), sequence_lengths]
        else:
            pooled = hidden_states[:, -1, :]  # Simply use last token

        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [batch, num_classes]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return ClassificationOutput(logits=logits, loss=loss, hidden_states=pooled)
```

**Why Last-Token Pooling?**
- GPT is causal: last token has seen the entire sequence
- Natural for autoregressive models
- Matches how GPT processes sequences left-to-right

### B. DeBERTaForSequenceClassification

**Location**: `classification_models.py:139`

**Pooling Strategies**: Multiple options (mean, max, first, last)

```python
class DeBERTaForSequenceClassification(nn.Module):
    """DeBERTa model with classification head and flexible pooling"""

    def __init__(self,
                 lm_model: nn.Module,
                 num_classes: int,
                 hidden_size: int,
                 pooling_strategy: str = 'mean',
                 dropout: float = 0.1,
                 freeze_base: bool = False):
        super().__init__()

        self.lm_model = lm_model
        self.pooling_strategy = pooling_strategy  # 'mean', 'max', 'first', 'last'

        # Freeze base model if requested
        if freeze_base:
            for param in self.lm_model.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def pool_hidden_states(self, hidden_states, attention_mask):
        """Apply pooling strategy to hidden states"""
        if self.pooling_strategy == 'mean':
            # Attention-mask weighted mean
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask

        elif self.pooling_strategy == 'max':
            # Max pooling (set padding to -inf)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states[mask_expanded == 0] = -1e9
            return torch.max(hidden_states, dim=1)[0]

        elif self.pooling_strategy == 'first':
            # [CLS] token (first token)
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == 'last':
            # Last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            return hidden_states[range(batch_size), sequence_lengths]

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get language model outputs
        lm_outputs = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = lm_outputs.last_hidden_state

        # Pool hidden states
        pooled = self.pool_hidden_states(hidden_states, attention_mask)

        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return ClassificationOutput(logits=logits, loss=loss, hidden_states=pooled)
```

**Pooling Strategy Comparison**:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Mean** | Attention-weighted average | General purpose, stable |
| **Max** | Maximum across sequence | Focus on salient features |
| **First** | [CLS] token | Following BERT convention |
| **Last** | Last token | Similar to GPT approach |

**Recommendation**: Mean pooling for most tasks (best balance)

### Usage Example

```python
from src.models.model_factory import create_model
from src.models.classification_models import GPTForSequenceClassification

# Create base language model
gpt_model = create_model('gpt', vocab_size=32000, model_size='small')

# Wrap with classification head
classifier = GPTForSequenceClassification(
    lm_model=gpt_model,
    num_classes=14,  # e.g., BBC Articles Classification
    hidden_size=768,
    dropout=0.1,
    freeze_base=False  # Fine-tune entire model
)

# Forward pass
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
labels = torch.tensor([3])

output = classifier(input_ids, attention_mask, labels)
print(output.logits.shape)  # [1, 14]
print(output.loss)  # scalar
```

### Key Features

1. **Base Model Freezing**: Option to freeze language model and only train classification head
2. **Automatic dtype Matching**: Classification head matches base model dtype
3. **Flexible Pooling**: Multiple pooling strategies for DeBERTa
4. **Standard Output Format**: `ClassificationOutput` dataclass for consistency
5. **Loss Integration**: Automatic loss computation when labels provided

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

### Small Model (for quick experiments)
```yaml
model:
  model_type: "gpt"
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  max_length: 256
  dropout: 0.1
  intermediate_size: 2048
```

### Base Model (default)
```yaml
model:
  model_type: "gpt"
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  max_length: 512
  dropout: 0.1
  intermediate_size: 3072
```

### Large Model (if compute allows)
```yaml
model:
  model_type: "gpt"
  hidden_size: 1024
  num_layers: 24
  num_heads: 16
  max_length: 1024
  dropout: 0.1
  intermediate_size: 4096
```

## Computational Requirements

### Memory Estimates (per model)

| Config | Parameters | Training (batch=32) | Inference (batch=1) |
|--------|-----------|---------------------|---------------------|
| Small  | ~40M      | ~8 GB               | ~1 GB               |
| Base   | ~110M     | ~16 GB              | ~2 GB               |
| Large  | ~350M     | ~32 GB              | ~4 GB               |
| Hybrid | ~220M     | ~24 GB              | ~4 GB               |

**Notes**:
- Training memory includes gradients and optimizer states
- Mixed precision (fp16) can reduce by ~50%
- Gradient checkpointing can reduce further with speed trade-off

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

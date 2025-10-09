# Model Architectures

## Overview

This project implements three transformer-based model architectures for Hindi language modeling:

1. **GPT-2 Style**: Autoregressive (causal) language modeling
2. **BERT Style**: Masked language modeling (bidirectional)
3. **Hybrid**: Combined causal and masked objectives

Each architecture is designed to explore different aspects of language understanding and generation with limited training data.

## Architecture Comparison

| Feature | GPT-2 | BERT | Hybrid |
|---------|-------|------|--------|
| **Objective** | Causal LM | Masked LM | Both |
| **Attention** | Causal (unidirectional) | Bidirectional | Both |
| **Generation** | Yes | No | Yes |
| **Understanding** | Moderate | Strong | Strong |
| **Training** | Simpler | More complex | Most complex |
| **Parameters** | ~110M | ~110M | ~220M |
| **Use Case** | Text generation | Classification, NER | Versatile |

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

## 2. BERT Style Model

### Overview

**Location**: `src/models/bert_model.py:5`

**Class**: `HindiBERTModel`

**Objective**: Masked language modeling

**Training Task**: Predict masked tokens using bidirectional context

**Formula**:
```
P(xᵢ | x₁, ..., xᵢ₋₁, xᵢ₊₁, ..., xₙ) for masked positions
```

### Architecture

```
Input Tokens (with [MASK])
    ↓
Token + Position + Segment Embeddings
    ↓
┌─────────────────────────┐
│  Transformer Layers     │
│  (Bidirectional)        │
│                         │
│  ┌─────────────────┐   │
│  │ Self-Attention  │   │ ← Bidirectional (all positions)
│  └────────┬────────┘   │
│           ↓             │
│  ┌─────────────────┐   │
│  │   Feed-Forward  │   │
│  └────────┬────────┘   │
│           ↓             │
│  [Repeated 12x]         │
└─────────┬───────────────┘
          ↓
  Masked LM Head
          ↓
  Masked Token Predictions
```

### Implementation

```python
import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM

class HindiBERTModel(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=config.get('hidden_size', 768),
            num_hidden_layers=config.get('num_layers', 12),
            num_attention_heads=config.get('num_heads', 12),
            intermediate_size=config.get('intermediate_size', 3072),
            max_position_embeddings=config.get('max_length', 512),
            hidden_dropout_prob=config.get('dropout', 0.1),
            attention_probs_dropout_prob=config.get('dropout', 0.1),
        )

        self.model = BertForMaskedLM(self.config)

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

## 3. Hybrid Model

### Overview

**Location**: `src/models/hybrid_model.py:5`

**Class**: `HybridGPTBERTModel`

**Objective**: Combined causal and masked language modeling

**Training Task**: Learn both generation and understanding

### Architecture

```
Input Tokens
    ↓
Shared Embeddings
    ↓
    ├─────────────────────────┐
    │                         │
    ↓                         ↓
┌────────────────┐   ┌───────────────┐
│  BERT Encoder  │   │  GPT Decoder  │
│ (Bidirectional)│   │   (Causal)    │
└────────┬───────┘   └───────┬───────┘
         │                   │
         ↓                   ↓
    ┌────────┐         ┌─────────┐
    │ MLM    │         │  CLM    │
    │ Head   │         │  Head   │
    └────┬───┘         └────┬────┘
         │                  │
         ↓                  ↓
   Masked Preds       Next Token Preds
```

### Implementation

```python
import torch
import torch.nn as nn
from transformers import BertModel, GPT2Model

class HybridGPTBERTModel(nn.Module):
    """Hybrid model combining causal and masked language modeling"""

    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        self.vocab_size = vocab_size

        # Shared embedding layer
        self.embeddings = nn.Embedding(vocab_size, config['hidden_size'])

        # BERT encoder for bidirectional understanding
        self.bert_encoder = BertModel(config['bert_config'])

        # GPT decoder for causal generation
        self.gpt_decoder = GPT2Model(config['gpt_config'])

        # Task-specific heads
        self.mlm_head = nn.Linear(config['hidden_size'], vocab_size)  # Masked LM
        self.lm_head = nn.Linear(config['hidden_size'], vocab_size)   # Causal LM

        # Loss weights
        self.loss_weights = {'mlm': 0.5, 'clm': 0.5}

    def forward(self, input_ids, attention_mask=None, task='both'):
        if task == 'mlm' or task == 'both':
            # Bidirectional encoding
            mlm_outputs = self.bert_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            mlm_logits = self.mlm_head(mlm_outputs.last_hidden_state)

        if task == 'clm' or task == 'both':
            # Causal decoding
            clm_outputs = self.gpt_decoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            clm_logits = self.lm_head(clm_outputs.last_hidden_state)

        if task == 'both':
            return {'mlm_logits': mlm_logits, 'clm_logits': clm_logits}
        elif task == 'mlm':
            return {'mlm_logits': mlm_logits}
        else:
            return {'clm_logits': clm_logits}
```

### Key Features

1. **Dual Objectives**:
   - Masked LM (bidirectional understanding)
   - Causal LM (generation)

2. **Shared Embeddings** (optional):
   - Reduce parameter count
   - Shared semantic space

3. **Multi-Task Learning**:
   - Combined loss: `loss = α * mlm_loss + β * clm_loss`
   - Default: α = β = 0.5

4. **Flexible Inference**:
   - Use BERT encoder for understanding
   - Use GPT decoder for generation

### Training

**Combined Loss**:
```python
mlm_loss = CrossEntropyLoss(mlm_logits[masked_pos], labels[masked_pos])
clm_loss = CrossEntropyLoss(clm_logits[:-1], labels[1:])

total_loss = 0.5 * mlm_loss + 0.5 * clm_loss
```

### Parameters

With standard configuration (vocab_size=32000):
- **Embeddings**: 24.5M (shared)
- **BERT Encoder**: 84M
- **GPT Decoder**: 84M
- **Heads**: 2 × 24M = 48M
- **Total**: ~220M parameters (2× single model)

### Usage Modes

**Training** (both objectives):
```python
outputs = model(input_ids, task='both')
mlm_logits = outputs['mlm_logits']
clm_logits = outputs['clm_logits']
```

**Understanding Tasks** (use BERT):
```python
outputs = model(input_ids, task='mlm')
representations = outputs['mlm_logits']
```

**Generation Tasks** (use GPT):
```python
outputs = model(input_ids, task='clm')
next_token_probs = outputs['clm_logits']
```

### Pros and Cons

**Pros**:
- ✅ Combines strengths of both architectures
- ✅ Versatile (generation + understanding)
- ✅ Better representations from multi-task learning

**Cons**:
- ❌ Double the parameters (~220M)
- ❌ More complex training
- ❌ Longer training time
- ❌ May not excel at either task individually

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

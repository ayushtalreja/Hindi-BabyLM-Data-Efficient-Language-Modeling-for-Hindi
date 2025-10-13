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

## 4. Position Encodings

**Location**: `src/models/position_encodings.py`

**Purpose**: Provide positional information to transformer models, which are inherently position-agnostic.

### Overview

Position encodings are crucial for transformers to understand token order. This module implements 5 different strategies, each with unique characteristics:

| Type | Parameters | Extrapolation | Use Case |
|------|-----------|---------------|----------|
| **Sinusoidal** | None (fixed) | Good | Original Transformer |
| **Learned** | Trainable | Poor | BERT, GPT-2 |
| **RoPE** | None (fixed) | Excellent | GPT-Neo, LLaMA |
| **ALiBi** | None (fixed) | Excellent | BLOOM |
| **Relative** | Trainable | Good | T5 |

### 4.1. Sinusoidal Position Encoding

**Class**: `SinusoidalPositionEncoding` (`position_encodings.py:23-66`)

**Description**: Original Transformer approach using sine and cosine functions of different frequencies.

**Formula**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Implementation**:
```python
from src.models.position_encodings import SinusoidalPositionEncoding

# Initialize
pos_enc = SinusoidalPositionEncoding(d_model=768, max_len=512)

# Forward pass
hidden_states = pos_enc(token_embeddings)  # Adds position info
```

**Features**:
- ✅ No trainable parameters (zero cost)
- ✅ Deterministic and reproducible
- ✅ Good length extrapolation
- ✅ Encodes absolute positions
- ❌ Less flexible than learned embeddings

**When to use**: Good baseline, especially for limited data regimes.

---

### 4.2. Learned Position Encoding

**Class**: `LearnedPositionEncoding` (`position_encodings.py:68-107`)

**Description**: Trainable position embeddings (BERT-style).

**Implementation**:
```python
from src.models.position_encodings import LearnedPositionEncoding

# Initialize
pos_enc = LearnedPositionEncoding(d_model=768, max_len=512)

# Forward pass
hidden_states = pos_enc(token_embeddings)  # Adds learned position embeddings
```

**Features**:
- ✅ Adapts to data during training
- ✅ Good for shorter sequences
- ✅ Standard in BERT, GPT-2
- ❌ Fixed maximum length (512 in this case)
- ❌ Poor extrapolation to longer sequences
- ❌ Adds parameters: `max_len × d_model` (e.g., 512 × 768 = 393K)

**When to use**: Standard baseline for fixed-length tasks.

---

### 4.3. Rotary Position Embedding (RoPE)

**Class**: `RotaryPositionEncoding` (`position_encodings.py:109-199`)

**Description**: Encodes relative positions by rotating query and key vectors. Used in GPT-Neo, LLaMA, GPT-J.

**Theory**: Instead of adding position embeddings to inputs, RoPE rotates the query and key vectors in attention by an angle proportional to their position.

**Rotation Formula**:
```
q'_m = R(m) q_m
k'_n = R(n) k_n

where R(θ) is a rotation matrix with θ = m * base^(-2i/d)
```

**Implementation**:
```python
from src.models.position_encodings import RotaryPositionEncoding

# Initialize (per attention head)
head_dim = hidden_size // num_heads  # e.g., 768 // 12 = 64
rope = RotaryPositionEncoding(dim=head_dim, max_len=2048, base=10000)

# Forward pass (in attention layer)
# q, k: [batch, num_heads, seq_len, head_dim]
q_rotated, k_rotated = rope(q, k)

# Then compute attention with rotated q and k
attn_scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1))
```

**Features**:
- ✅ **Excellent length extrapolation** (key advantage)
- ✅ Encodes relative positions naturally
- ✅ No additional parameters
- ✅ Works at the attention level (more principled)
- ❌ Requires attention layer modification
- ❌ Slightly more complex implementation

**Configuration**:
- `dim`: Head dimension (hidden_size / num_heads)
- `max_len`: Maximum sequence length to cache (2048 typical)
- `base`: Base for frequency calculation (10000 standard, higher for longer sequences)

**When to use**: **Recommended for Hindi BabyLM** - excellent extrapolation and no parameters.

---

### 4.4. Attention with Linear Biases (ALiBi)

**Class**: `ALiBiPositionBias` (`position_encodings.py:201-306`)

**Description**: Adds linear biases to attention scores based on distance. Used in BLOOM.

**Theory**: Instead of position embeddings, adds a bias to attention scores:
```
attention_score(q_i, k_j) = q_i · k_j + m * (i - j)
```
where `m` is a head-specific slope.

**Slopes**: Different for each attention head, computed as:
```
m_h = (1/2)^(8h/H) where H is number of heads
```

**Implementation**:
```python
from src.models.position_encodings import ALiBiPositionBias

# Initialize
alibi = ALiBiPositionBias(num_heads=12, max_len=2048)

# Forward pass (in attention layer)
# attention_scores: [batch, num_heads, seq_len, seq_len]
attention_scores_with_bias = alibi(attention_scores)

# Then apply softmax
attn_weights = F.softmax(attention_scores_with_bias, dim=-1)
```

**Features**:
- ✅ **Best length extrapolation** (better than RoPE for very long sequences)
- ✅ No position embeddings needed
- ✅ No additional parameters
- ✅ Simple and elegant
- ❌ Requires attention layer modification
- ❌ Adds computation to attention

**When to use**: Best for tasks requiring strong length extrapolation (e.g., long document processing).

---

### 4.5. Relative Position Bias (T5-style)

**Class**: `RelativePositionBias` (`position_encodings.py:308-396`)

**Description**: Learned biases for relative positions using buckets. Used in T5.

**Theory**: Maps relative positions to buckets, learns a bias for each bucket:
```
bias_table: [num_buckets, num_heads]
relative_position = j - i
bucket = bucket_function(relative_position)
bias = bias_table[bucket]
```

**Bucketing**:
- Close positions: exact buckets
- Distant positions: logarithmic buckets

**Implementation**:
```python
from src.models.position_encodings import RelativePositionBias

# Initialize
rel_bias = RelativePositionBias(
    num_heads=12,
    num_buckets=32,
    max_distance=128,
    bidirectional=False  # True for BERT-style, False for GPT-style
)

# Forward pass
bias = rel_bias(seq_len=512, device=device)
# Returns: [1, num_heads, seq_len, seq_len]

# Add to attention scores
attention_scores = attention_scores + bias
```

**Features**:
- ✅ Learns relative position biases from data
- ✅ Good generalization via bucketing
- ✅ Flexible (bidirectional or causal)
- ❌ Adds parameters: `num_buckets × num_heads` (e.g., 32 × 12 = 384)
- ❌ More complex than ALiBi

**When to use**: When you want learned relative positions (T5-style models).

---

### Factory Function

**Function**: `create_position_encoding()` (`position_encodings.py:398-440`)

**Purpose**: Convenient factory for creating position encodings.

**Usage**:
```python
from src.models.position_encodings import create_position_encoding

# Sinusoidal
pos_enc = create_position_encoding('sinusoidal', d_model=768, max_len=512)

# Learned
pos_enc = create_position_encoding('learned', d_model=768, max_len=512)

# RoPE (requires num_heads)
pos_enc = create_position_encoding('rope', d_model=768, num_heads=12, max_len=2048, base=10000)

# ALiBi (requires num_heads)
pos_enc = create_position_encoding('alibi', d_model=768, num_heads=12, max_len=2048)

# Relative (requires num_heads)
pos_enc = create_position_encoding('relative', d_model=768, num_heads=12, max_len=512,
                                   num_buckets=32, max_distance=128, bidirectional=False)
```

---

### Position Encoding Comparison

**For Hindi BabyLM**:

| Criterion | Sinusoidal | Learned | RoPE | ALiBi | Relative |
|-----------|-----------|---------|------|-------|----------|
| **Parameters** | 0 | 393K | 0 | 0 | 384 |
| **Extrapolation** | Good | Poor | Excellent | Excellent | Good |
| **Data Efficiency** | Good | Moderate | Excellent | Excellent | Moderate |
| **Implementation** | Simple | Simple | Moderate | Moderate | Complex |
| **Training Speed** | Fast | Fast | Fast | Slightly slower | Fast |

**Recommendation**: **RoPE** is the best choice for Hindi BabyLM:
- Zero parameters (important for limited data)
- Excellent length extrapolation
- Modern architecture (used in LLaMA, GPT-Neo)
- Good for low-resource languages

**Alternative**: **Sinusoidal** for a simpler baseline.

---

## 5. Enhanced GPT Model

**Location**: `src/models/enhanced_gpt.py`

**Purpose**: Advanced GPT implementation with configurable position encodings, model sizes, and efficiency features.

### Overview

The `EnhancedGPTModel` extends the basic GPT-2 model with:
- **Multiple position encoding options** (RoPE, ALiBi, Sinusoidal, Learned)
- **Three model size variants** (Tiny: 50M, Small: 110M, Medium: 350M)
- **Gradient checkpointing** for memory efficiency
- **Flash Attention support** (if available)
- **RMS Norm** option (more efficient than LayerNorm)
- **SwiGLU activation** option (better than GELU)

### Model Size Variants

**Configuration**: `EnhancedGPTConfig.MODEL_SIZES` (`enhanced_gpt.py:41-60`)

| Variant | Hidden | Layers | Heads | FFN | Parameters |
|---------|--------|--------|-------|-----|------------|
| **Tiny** | 512 | 6 | 8 | 2048 | ~50M |
| **Small** | 768 | 12 | 12 | 3072 | ~110M |
| **Medium** | 1024 | 24 | 16 | 4096 | ~350M |

**Usage**:
```python
from src.models.enhanced_gpt import create_enhanced_gpt_model

# Create tiny model (for quick experiments)
model_tiny = create_enhanced_gpt_model(vocab_size=32000, model_size='tiny')

# Create small model (default, recommended for BabyLM)
model_small = create_enhanced_gpt_model(vocab_size=32000, model_size='small')

# Create medium model (if compute allows)
model_medium = create_enhanced_gpt_model(vocab_size=32000, model_size='medium')
```

---

### Enhanced GPT Configuration

**Class**: `EnhancedGPTConfig` (`enhanced_gpt.py:37-113`)

**Key Parameters**:

```python
config = EnhancedGPTConfig(
    vocab_size=32000,
    model_size='small',  # 'tiny', 'small', 'medium'

    # Position encoding
    position_encoding_type='rope',  # 'learned', 'sinusoidal', 'rope', 'alibi'
    max_position_embeddings=512,

    # Regularization
    dropout=0.1,
    attention_dropout=0.1,
    residual_dropout=0.1,

    # Normalization
    use_rms_norm=False,  # Use RMS Norm instead of LayerNorm
    layer_norm_eps=1e-5,

    # Activation
    activation='gelu',  # 'gelu', 'relu', 'swiglu'

    # Efficiency
    use_flash_attention=True,  # Use Flash Attention if available
    gradient_checkpointing=False,  # Enable to save memory

    # Attention
    attention_bias=True,  # Bias in attention projections
)
```

---

### Position Encoding Integration

**Configuration Examples**:

**With RoPE** (recommended):
```python
model = create_enhanced_gpt_model(
    vocab_size=32000,
    model_size='small',
    position_encoding_type='rope',
    max_position_embeddings=2048  # Can handle long sequences
)
```

**With ALiBi**:
```python
model = create_enhanced_gpt_model(
    vocab_size=32000,
    model_size='small',
    position_encoding_type='alibi',
    max_position_embeddings=2048
)
```

**With Sinusoidal**:
```python
model = create_enhanced_gpt_model(
    vocab_size=32000,
    model_size='small',
    position_encoding_type='sinusoidal',
    max_position_embeddings=512
)
```

**With Learned** (default GPT-2 style):
```python
model = create_enhanced_gpt_model(
    vocab_size=32000,
    model_size='small',
    position_encoding_type='learned',
    max_position_embeddings=512
)
```

---

### Gradient Checkpointing

**Purpose**: Trade compute for memory - recompute activations during backward pass instead of storing them.

**Memory Savings**: ~30-50% reduction

**Speed Cost**: ~20-30% slower training

**Configuration**:
```python
model = create_enhanced_gpt_model(
    vocab_size=32000,
    model_size='medium',  # Large model
    gradient_checkpointing=True  # Enable to fit in GPU memory
)
```

**Usage**: Automatically applied during training when enabled. No code changes needed.

**When to use**:
- Large models that don't fit in GPU memory
- When batch size is constrained by memory
- Training on consumer GPUs

---

### RMS Norm vs LayerNorm

**RMS Norm**: Root Mean Square Layer Normalization

**Class**: `RMSNorm` (`enhanced_gpt.py:115-126`)

**Formula**:
```
RMSNorm(x) = (x / RMS(x)) * γ
where RMS(x) = sqrt(mean(x²) + ε)
```

**Advantages over LayerNorm**:
- ✅ ~10-15% faster (no mean subtraction)
- ✅ Fewer operations
- ✅ Used in modern models (LLaMA, PaLM)

**Usage**:
```python
model = create_enhanced_gpt_model(
    vocab_size=32000,
    model_size='small',
    use_rms_norm=True  # Use RMS Norm instead of LayerNorm
)
```

---

### Flash Attention

**Description**: Optimized attention implementation that reduces memory and increases speed.

**Benefits**:
- ✅ 2-4x faster attention
- ✅ Reduced memory usage
- ✅ Exact (not approximate)

**Requirements**:
```bash
pip install flash-attn  # Requires CUDA
```

**Usage**: Automatically used if available and `use_flash_attention=True` (default).

**Fallback**: Falls back to standard attention if not available.

---

### Complete Configuration Example

**Small Model with RoPE** (recommended for Hindi BabyLM):
```yaml
model:
  model_type: "enhanced_gpt"
  vocab_size: 32000
  model_size: "small"

  # Position encoding
  position_encoding_type: "rope"
  max_position_embeddings: 512

  # Architecture
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  intermediate_size: 3072

  # Regularization
  dropout: 0.1
  attention_dropout: 0.1
  residual_dropout: 0.1

  # Normalization & Activation
  use_rms_norm: false
  activation: "gelu"

  # Efficiency
  use_flash_attention: true
  gradient_checkpointing: false

  # Initialization
  initializer_range: 0.02
```

**Medium Model with ALiBi and Gradient Checkpointing**:
```yaml
model:
  model_type: "enhanced_gpt"
  vocab_size: 32000
  model_size: "medium"

  # Position encoding
  position_encoding_type: "alibi"
  max_position_embeddings: 1024

  # Efficiency (needed for larger model)
  gradient_checkpointing: true
  use_flash_attention: true
  use_rms_norm: true  # Faster than LayerNorm

  # Activation
  activation: "swiglu"  # Better than GELU
```

---

### Training and Generation

**Training**:
```python
from src.models.enhanced_gpt import create_enhanced_gpt_model

# Create model
model = create_enhanced_gpt_model(
    vocab_size=32000,
    model_size='small',
    position_encoding_type='rope'
)

# Forward pass
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs['loss']
logits = outputs['logits']

# Backward
loss.backward()
```

**Generation**:
```python
# Generate text
model.eval()
prompt_ids = tokenizer.encode("मैं एक")
prompt_tensor = torch.tensor([prompt_ids])

generated_ids = model.generate(
    prompt_tensor,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)

generated_text = tokenizer.decode(generated_ids[0])
print(generated_text)
# Output: "मैं एक विद्यार्थी हूँ जो विश्वविद्यालय में पढ़ता है..."
```

---

### Parameter Counts

**By Model Size** (vocab_size=32000):

| Component | Tiny (50M) | Small (110M) | Medium (350M) |
|-----------|-----------|--------------|---------------|
| **Token Embeddings** | 16.4M | 24.6M | 32.8M |
| **Position Embeddings** | 262K† | 393K† | 524K† |
| **Transformer Blocks** | 32M | 84M | 315M |
| **LM Head** | Tied | Tied | Tied |
| **Total** | ~49M | ~109M | ~348M |

† Only for learned position embeddings. RoPE and ALiBi have zero position encoding parameters.

---

### Memory Requirements

**GPU Memory Estimates** (batch_size=32, seq_len=512):

| Model | Position Encoding | Training (FP32) | Training (FP16) | Inference |
|-------|------------------|-----------------|-----------------|-----------|
| **Tiny** | Any | ~8 GB | ~4 GB | ~1 GB |
| **Small** | Learned/Sinusoidal | ~18 GB | ~9 GB | ~2 GB |
| **Small** | RoPE/ALiBi | ~17 GB | ~8.5 GB | ~2 GB |
| **Small + GradChkpt** | RoPE | ~12 GB | ~6 GB | ~2 GB |
| **Medium** | Any | ~36 GB | ~18 GB | ~4 GB |
| **Medium + GradChkpt** | RoPE | ~24 GB | ~12 GB | ~4 GB |

**Notes**:
- Training includes model + gradients + optimizer states
- FP16 mixed precision approximately halves memory
- Gradient checkpointing saves ~30-40% memory
- Inference is much more memory-efficient

---

### Best Practices

**1. Position Encoding Selection**:
- **Limited data**: Use RoPE or Sinusoidal (no parameters)
- **Fixed lengths**: Learned is acceptable
- **Long sequences**: Use RoPE or ALiBi
- **Memory constrained**: Use RoPE or ALiBi (no position embedding parameters)

**2. Model Size Selection**:
- **10M token budget**: Small (110M) is recommended
- **Quick experiments**: Tiny (50M)
- **Maximum performance**: Medium (350M) if compute allows

**3. Memory Optimization**:
- Enable gradient checkpointing for large models
- Use FP16 mixed precision
- Use RMS Norm instead of LayerNorm
- Use Flash Attention if available
- Reduce batch size if needed

**4. Initialization**:
- Keep `initializer_range=0.02` (standard)
- Consider smaller range (0.01) for very large models

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

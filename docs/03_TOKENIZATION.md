# Tokenization for Hindi Language Models

## Overview

Tokenization is a critical component for Hindi language models due to the morphological richness of the language. This module implements and compares three popular tokenization strategies: **SentencePiece**, **WordPiece**, and **BPE** (Byte-Pair Encoding).

## Why Tokenization Matters for Hindi

Hindi presents unique tokenization challenges:

1. **Morphological Complexity**: Rich inflectional morphology (case markers, agreement, tense, aspect, mood)
2. **Compound Words**: Frequent use of compound nouns and verbs
3. **Script Characteristics**: Devanagari script with matras (vowel diacritics) and conjuncts
4. **Sandhi**: Phonetic changes at word boundaries
5. **Code-Mixing**: Frequent mixing with English and other languages

**Key Questions**:
- How to balance morpheme preservation vs. vocabulary size?
- Which algorithm best captures Hindi word structure?
- How do different tokenizers affect model performance?

## Tokenization Strategies

### Comparison Table

| Feature | SentencePiece | WordPiece | BPE |
|---------|--------------|-----------|-----|
| **Algorithm** | Unigram LM | Greedy frequency-based | Merge pairs |
| **Subword Units** | Variable | Variable | Variable |
| **Morphology** | Good | Moderate | Moderate |
| **Vocabulary** | Flexible | Fixed | Fixed |
| **Speed** | Fast | Fast | Moderate |
| **Use Case** | General LMs | BERT-style | GPT-style |

## Implementation

### TokenizerFactory

**Location**: `src/tokenization/tokenizer_factory.py:10`

**Purpose**: Factory class for creating and managing different tokenizers.

**Usage**:
```python
from src.tokenization.tokenizer_factory import TokenizerFactory

# Create factory with configuration
factory = TokenizerFactory(config)

# Train tokenizer on corpus
tokenizer = factory.create_tokenizer(training_texts)

# Save tokenizer
factory.save_tokenizer(tokenizer, experiment_name)

# Load tokenizer
tokenizer = TokenizerFactory.load_tokenizer(experiment_name)
```

**Methods**:

#### `create_tokenizer(training_texts)` (line 22)
```python
def create_tokenizer(self, training_texts: List[str]):
    """
    Create and train a tokenizer based on config

    Args:
        training_texts: List of training texts

    Returns:
        Trained tokenizer (type depends on config.tokenizer_type)

    Supported Types:
        - "sentencepiece": SentencePiece tokenizer
        - "wordpiece": WordPiece tokenizer (BERT-style)
        - "bpe": BPE tokenizer
    """
```

#### `load_tokenizer(experiment_name)` (line 134)
```python
@staticmethod
def load_tokenizer(experiment_name: str, tokenizer_dir: str = 'tokenizers'):
    """
    Load a saved tokenizer

    Args:
        experiment_name: Name of experiment
        tokenizer_dir: Directory containing tokenizers

    Returns:
        Loaded tokenizer

    Searches for:
        1. Metadata file with experiment name
        2. Standard tokenizer files (sentencepiece.model, etc.)
    """
```

## 1. SentencePiece Tokenizer

### Overview

**Algorithm**: Unigram Language Model
**Library**: sentencepiece
**Location**: `src/tokenization/sentencepiece_tokenizer.py:4`

**Advantages**:
- Language-agnostic (no pre-tokenization required)
- Handles spaces as tokens (good for Hindi)
- Reversible tokenization
- Efficient training and inference
- Best morphological preservation for Hindi

**Disadvantages**:
- Requires separate library
- May split morphemes unexpectedly

### Implementation

**Class**: `HindiSentencePieceTokenizer`

```python
class HindiSentencePieceTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.model = None

    def train_tokenizer(self, corpus: List[str], model_prefix: str):
        """Train SentencePiece tokenizer on Hindi corpus"""

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to string tokens"""

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
```

### Training Parameters

```python
spm.SentencePieceTrainer.train(
    input=training_file,
    model_prefix=model_prefix,
    vocab_size=32000,                # Vocabulary size
    character_coverage=0.995,        # High coverage for Hindi characters
    model_type='bpe',                # Can be 'unigram', 'bpe', 'char', 'word'
    pad_id=0,                        # Padding token ID
    unk_id=1,                        # Unknown token ID
    bos_id=2,                        # Beginning of sentence
    eos_id=3,                        # End of sentence
    user_defined_symbols=[],         # Custom symbols
    normalization_rule_name='nfkc'   # Unicode normalization
)
```

**Key Parameter: `character_coverage`**
- Set to 0.995 for Hindi
- Ensures rare Devanagari characters are included
- Balance between vocabulary size and coverage

### Example

```python
tokenizer = HindiSentencePieceTokenizer(vocab_size=32000)
tokenizer.train_tokenizer(training_texts, 'tokenizers/sentencepiece')

# Tokenization
text = "मैं विश्वविद्यालय जा रहा हूँ।"
tokens = tokenizer.tokenize(text)
# Output: ['▁मैं', '▁विश्व', 'विद्यालय', '▁जा', '▁रहा', '▁हूँ', '।']

# Encoding
ids = tokenizer.encode(text)
# Output: [145, 2341, 5678, 892, 1234, 3456, 12]

# Decoding
decoded = tokenizer.decode(ids)
# Output: "मैं विश्वविद्यालय जा रहा हूँ।"
```

### Morphological Behavior

**Compound Words**:
```
विश्वविद्यालय (university) → ['▁विश्व', 'विद्यालय']
(विश्व = world, विद्यालय = school)
```

**Inflected Forms**:
```
लड़का (boy, nominative) → ['▁लड़', 'का']
लड़के (boy, oblique) → ['▁लड़', 'के']
लड़कों (boys) → ['▁लड़', 'कों']
```

## 2. WordPiece Tokenizer

### Overview

**Algorithm**: Greedy frequency-based merging
**Library**: HuggingFace tokenizers
**Location**: `src/tokenization/tokenizer_factory.py:58`

**Advantages**:
- Used in BERT (proven effectiveness)
- Good subword segmentation
- Efficient inference

**Disadvantages**:
- Requires pre-tokenization (whitespace)
- May not preserve morpheme boundaries well
- Fixed vocabulary after training

### Implementation

**Created in**: `TokenizerFactory._create_wordpiece_tokenizer()` (line 58)

```python
def _create_wordpiece_tokenizer(self, training_texts: List[str]):
    """Create WordPiece tokenizer (BERT-style)"""
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.normalizers import NFD, Sequence

    # Initialize tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    # Set normalizer (NFD for Hindi)
    tokenizer.normalizer = Sequence([NFD()])

    # Set pre-tokenizer (whitespace splitting)
    tokenizer.pre_tokenizer = Whitespace()

    # Create trainer
    trainer = WordPieceTrainer(
        vocab_size=self.vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2
    )

    # Train on texts
    tokenizer.train_from_iterator(training_texts, trainer)

    return WordPieceTokenizerWrapper(tokenizer, self.vocab_size)
```

### Special Tokens

| Token | Purpose | ID |
|-------|---------|-----|
| `[PAD]` | Padding | 0 |
| `[UNK]` | Unknown tokens | 1 |
| `[CLS]` | Classification token | 2 |
| `[SEP]` | Separator token | 3 |
| `[MASK]` | Masked token (for MLM) | 4 |

### Example

```python
text = "मैं विश्वविद्यालय जा रहा हूँ।"
tokens = tokenizer.tokenize(text)
# Output: ['मैं', 'विश्व', '##विद्यालय', 'जा', 'रहा', 'हूँ', '।']
# Note: ## prefix indicates continuation of previous word

ids = tokenizer.encode(text)
# Output: [2, 145, 2341, 5678, 892, 1234, 3456, 12, 3]
# [CLS] ... [SEP]
```

### Morphological Behavior

**Handles morphology differently than SentencePiece**:
```
लड़का → ['लड़', '##का']
लड़के → ['लड़', '##के']
लड़कों → ['लड़', '##कों']
```

## 3. BPE Tokenizer

### Overview

**Algorithm**: Byte-Pair Encoding (iterative merge)
**Library**: HuggingFace tokenizers
**Location**: `src/tokenization/tokenizer_factory.py:96`

**Advantages**:
- Simple and effective
- Used in GPT models
- Good compression ratio

**Disadvantages**:
- May create arbitrary subwords
- Less interpretable than WordPiece
- Can split morphemes unpredictably

### Implementation

**Created in**: `TokenizerFactory._create_bpe_tokenizer()` (line 96)

```python
def _create_bpe_tokenizer(self, training_texts: List[str]):
    """Create BPE tokenizer"""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.normalizers import NFD, Sequence

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Set normalizer
    tokenizer.normalizer = Sequence([NFD()])

    # Set pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()

    # Create trainer
    trainer = BpeTrainer(
        vocab_size=self.vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
        min_frequency=2
    )

    # Train on texts
    tokenizer.train_from_iterator(training_texts, trainer)

    return BPETokenizerWrapper(tokenizer, self.vocab_size)
```

### Special Tokens

| Token | Purpose | ID |
|-------|---------|-----|
| `<pad>` | Padding | 0 |
| `<unk>` | Unknown tokens | 1 |
| `<s>` | Start of sequence | 2 |
| `</s>` | End of sequence | 3 |
| `<mask>` | Masked token | 4 |

### Example

```python
text = "मैं विश्वविद्यालय जा रहा हूँ।"
tokens = tokenizer.tokenize(text)
# Output: ['मैं', 'विश्', 'विद्', 'यालय', 'जा', 'रहा', 'हूँ', '।']

ids = tokenizer.encode(text)
# Output: [2, 145, 2341, 5678, 1123, 892, 1234, 3456, 12, 3]
# <s> ... </s>
```

## Tokenizer Wrappers

To provide a consistent interface, WordPiece and BPE tokenizers are wrapped:

### WordPieceTokenizerWrapper

**Location**: `src/tokenization/tokenizer_factory.py:213`

```python
class WordPieceTokenizerWrapper:
    """Wrapper for WordPiece tokenizer to provide consistent interface"""

    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text"""
        return self.tokenizer.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to tokens"""
        encoding = self.tokenizer.encode(text)
        return encoding.tokens
```

### BPETokenizerWrapper

**Location**: `src/tokenization/tokenizer_factory.py:235`

Similar interface to WordPieceTokenizerWrapper.

## Morphological Evaluation

**Location**: `src/tokenization/morphological_eval.py`

**Purpose**: Evaluate how well tokenizers preserve morphological information.

### Evaluation Metrics

1. **Morpheme Boundary Preservation**
   - How often do tokens align with morpheme boundaries?
   - Example: लड़का (boy) = लड़ + का (stem + marker)

2. **Consistency Across Paradigms**
   - Are inflected forms of the same lemma tokenized consistently?
   - Example: लड़का, लड़के, लड़कों should have common stem

3. **Compound Word Handling**
   - Are compound words segmented meaningfully?
   - Example: विश्वविद्यालय = विश्व + विद्यालय

4. **Vocabulary Efficiency**
   - How many tokens needed for common words?
   - Lower is generally better

### Example Evaluation

```python
from src.tokenization.morphological_eval import evaluate_morphology

# Test paradigm
paradigm = {
    'lemma': 'लड़का',
    'forms': ['लड़का', 'लड़के', 'लड़कों', 'लड़की', 'लड़कियाँ']
}

results = evaluate_morphology(tokenizer, paradigm)
# Output: {
#     'stem_consistency': 0.95,  # 95% share common stem
#     'avg_tokens_per_form': 2.3,
#     'boundary_alignment': 0.88
# }
```

## Tokenizer Comparison

**Location**: `src/tokenization/tokenizer_comparison.py`

**Purpose**: Benchmark and compare different tokenization strategies.

### Comparison Dimensions

1. **Performance Metrics**:
   - Tokenization speed (tokens/sec)
   - Compression ratio (chars/token)
   - Vocabulary coverage

2. **Linguistic Metrics**:
   - Morpheme preservation
   - Paradigm consistency
   - Rare word handling

3. **Model Metrics** (after training):
   - Downstream task performance
   - Training efficiency
   - Generalization

### Running Comparison

```python
from src.tokenization.tokenizer_comparison import compare_tokenizers

results = compare_tokenizers(
    training_texts,
    test_texts,
    vocab_size=32000
)

# Results include:
# - Speed comparison
# - Compression ratios
# - Morphological scores
# - Example tokenizations
```

## Best Practices

### 1. Vocabulary Size Selection

**Guidelines**:
- Small models (10M params): 16K-32K
- Medium models (100M params): 32K-50K
- Large models (1B+ params): 50K-100K

**For Hindi BabyLM** (110M params): **32,000 tokens**

**Rationale**:
- Balance between granularity and embedding table size
- Adequate coverage for Hindi morphology
- Compatible with limited data regime

### 2. Character Coverage

**For Hindi**: Set `character_coverage=0.995`

**Why**:
- Hindi uses Devanagari script with many diacritics
- Rare characters (e.g., archaic letters) should be included
- Higher coverage = better handling of rare words

### 3. Normalization

**Recommended**: Unicode NFD (Canonical Decomposition)

```python
from tokenizers.normalizers import NFD
tokenizer.normalizer = NFD()
```

**Why**:
- Separates base characters from diacritics
- Consistent representation of matras
- Better handling of variant spellings

### 4. Pre-tokenization

**For Hindi**: Whitespace pre-tokenization is usually sufficient

```python
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()
```

**Note**: Hindi uses Devanagari space (U+0020) like English.

### 5. Special Tokens

**Minimum Required**:
- `[PAD]` or `<pad>`: Padding
- `[UNK]` or `<unk>`: Unknown tokens

**For Masked LM (BERT)**:
- `[MASK]`: Masked tokens
- `[CLS]`: Classification
- `[SEP]`: Separator

**For Autoregressive LM (GPT)**:
- `<s>` or `<bos>`: Beginning of sequence
- `</s>` or `<eos>`: End of sequence

## Configuration

### YAML Configuration

```yaml
tokenization:
  tokenizer_type: "sentencepiece"  # or "wordpiece", "bpe"
  vocab_size: 32000
  character_coverage: 0.995
  model_type: "bpe"  # for SentencePiece: "bpe", "unigram", "char", "word"

  # Optional parameters
  normalization: "nfkc"
  split_by_whitespace: true
  remove_accents: false  # Keep Hindi diacritics
```

### Python Configuration

```python
from src.utils.experiment_config import ExperimentConfig

config = ExperimentConfig(
    tokenizer_type="sentencepiece",
    vocab_size=32000,
    tokenizer_dir="tokenizers"
)
```

## Troubleshooting

### Issue: Unknown Tokens (UNK) in Output

**Causes**:
1. Vocabulary too small
2. Character coverage too low
3. Text contains unseen scripts

**Solutions**:
1. Increase `vocab_size`
2. Increase `character_coverage` (for SentencePiece)
3. Add `user_defined_symbols` for special characters

### Issue: Poor Morphological Segmentation

**Causes**:
1. Wrong algorithm for Hindi
2. Insufficient training data
3. Inappropriate pre-tokenization

**Solutions**:
1. Try SentencePiece (generally best for Hindi)
2. Increase training corpus size
3. Adjust pre-tokenization strategy

### Issue: Inconsistent Tokenization

**Causes**:
1. Missing normalization
2. Different input encodings
3. Tokenizer not deterministic

**Solutions**:
1. Apply NFD normalization
2. Ensure UTF-8 encoding
3. Set random seeds if applicable

## Recommendations for Hindi

Based on experiments and linguistic analysis:

| Criterion | Best Choice | Reason |
|-----------|-------------|---------|
| **Morphology** | SentencePiece | Better morpheme boundaries |
| **Speed** | All similar | Minimal difference |
| **Simplicity** | SentencePiece | Language-agnostic |
| **BERT-style** | WordPiece | Standard for masked LM |
| **GPT-style** | SentencePiece/BPE | Standard for autoregressive |

**Overall Recommendation**: **SentencePiece with BPE model_type**

## Related Documentation

- [Data Processing Documentation](02_DATA_PROCESSING.md)
- [Model Architecture Documentation](04_MODELS.md)
- [Configuration Guide](07_CONFIGURATION.md)
- [API Reference](08_API_REFERENCE.md)

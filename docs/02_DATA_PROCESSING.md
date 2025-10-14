# Data Processing Pipeline

## Overview

The data processing pipeline is responsible for collecting, cleaning, filtering, and preparing Hindi text data for language model training. The goal is to create a high-quality corpus of approximately 10 million tokens from diverse sources while maintaining linguistic quality and removing duplicates.

## Architecture

```
Raw Data Sources → Collection → Cleaning → Quality Filtering → Deduplication → Token Limiting → Train/Val/Test Splits
```

## Data Sources

### 1. IndicCorp Hindi (`indiccorp_downloader.py`)

**Description**: Large-scale corpus of Hindi text from news articles and web crawls.

**Implementation**: `src/data_processing/indiccorp_downloader.py:14`

**Key Function**:
```python
def download_indiccorp_hindi():
    """Download IndicCorp Hindi dataset from HuggingFace"""
    # Uses HuggingFace datasets library
    # Returns dataset with 'text' field
```

**Characteristics**:
- Source: News articles, web content
- Size: Large (filtered to match token budget)
- Quality: Generally high, formal register
- Language: Modern Standard Hindi

**Pros**:
- Large scale
- Formal, well-edited text
- Diverse topics

**Cons**:
- May contain some noise from web scraping
- Formal register may not match child language

### 2. Hindi Wikipedia (`wiki_scraper.py`)

**Description**: Encyclopedia articles from Hindi Wikipedia.

**Implementation**: `src/data_processing/wiki_scraper.py:25`

**Key Function**:
```python
def scrape_hindi_wikipedia(categories: List[str], max_articles: int = 5000):
    """Scrape Hindi Wikipedia articles from specified categories"""
    # Categories: विज्ञान (Science), इतिहास (History), भूगोल (Geography),
    #             साहित्य (Literature), कला (Arts)
```

**Characteristics**:
- Source: Hindi Wikipedia
- Size: Medium (configurable, default 5000 articles)
- Quality: High, well-structured
- Language: Encyclopedic Hindi

**Pros**:
- High quality, edited content
- Structured information
- Diverse topics

**Cons**:
- Formal register
- May contain technical jargon
- Encyclopedia style may not match natural language

### 3. Children's Books (`childrens_books.py`)

**Description**: Collection of children's literature in Hindi.

**Implementation**: `src/data_processing/childrens_books.py:18`

**Key Function**:
```python
def collect_childrens_stories():
    """Collect children's stories from various sources"""
    # Simpler vocabulary and structures
    # Developmentally appropriate content
```

**Characteristics**:
- Source: Children's literature
- Size: Small to medium
- Quality: High, age-appropriate
- Language: Simple, natural Hindi

**Pros**:
- Developmentally plausible complexity
- Simple vocabulary and structures
- Natural language patterns

**Cons**:
- Smaller corpus size
- May be too simple for some tasks

## Pipeline Components

### 1. Corpus Builder (`corpus_builder.py`)

**Main Class**: `CorpusBuilder`

**Location**: `src/data_processing/corpus_builder.py:47`

**Purpose**: Orchestrates the entire data processing pipeline.

**Key Methods**:

#### `collect_all_data()` (line 68)
Collects data from all sources and combines them.

```python
def collect_all_data(self) -> Dict[str, List[str]]:
    """
    Collect data from all sources

    Returns:
        Dictionary mapping source names to lists of text samples
        {
            'indiccorp': [...],
            'wikipedia': [...],
            'childrens_books': [...]
        }
    """
```

**Process**:
1. Download IndicCorp Hindi dataset
2. Scrape Wikipedia articles from specified categories
3. Collect children's literature
4. Save raw data as pickle file

**Output**: `data/raw/raw_corpus.pkl`

#### `process_and_filter()` (line 114)
Applies cleaning and filtering to collected data.

```python
def process_and_filter(self, raw_data: Dict[str, List[str]]) -> List[str]:
    """
    Process and filter collected data

    Steps:
        1. Clean text (normalization, formatting)
        2. Filter by length
        3. Filter by language (Hindi character ratio)
        4. Deduplicate (exact and fuzzy matching)
        5. Limit to max tokens

    Returns:
        List of processed text samples
    """
```

**Filtering Pipeline**:
1. **Text Cleaning**: Normalize Unicode, remove extra whitespace
2. **Length Filtering**: Remove texts that are too short or too long
3. **Language Filtering**: Ensure high ratio of Devanagari characters
4. **Deduplication**: Remove exact and near-duplicate texts
5. **Token Limiting**: Limit corpus to ~10M tokens

#### `create_splits()` (line 171)
Creates train/validation/test splits.

```python
def create_splits(self, processed_data: List[str]) -> Dict[str, List[str]]:
    """
    Create train/val/test splits

    Split Ratios (configurable):
        - Train: 80%
        - Validation: 10%
        - Test: 10%

    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
```

**Process**:
1. Shuffle data with fixed random seed (42) for reproducibility
2. Calculate split sizes based on ratios
3. Create splits
4. Return dictionary with train/val/test data

#### `save_splits()` (line 198)
Saves processed splits to disk.

**Outputs**:
- `data/processed/train.pkl` - Training data (pickle)
- `data/processed/val.pkl` - Validation data (pickle)
- `data/processed/test.pkl` - Test data (pickle)
- `data/processed/train.txt` - First 100 samples as text (for inspection)
- `data/processed/val.txt` - First 100 samples as text
- `data/processed/test.txt` - First 100 samples as text
- `data/processed/metadata.json` - Metadata about splits

#### `load_splits()` (line 233)
Loads processed splits from disk.

#### `create_dataloader()` (line 251)
Creates PyTorch DataLoader for a split.

```python
def create_dataloader(self, texts: List[str], tokenizer, split: str = 'train') -> DataLoader:
    """
    Create PyTorch DataLoader

    Args:
        texts: List of text samples
        tokenizer: Trained tokenizer
        split: 'train', 'val', or 'test'

    Returns:
        DataLoader with batch_size from config
        shuffle=True for train, False otherwise
    """
```

### 2. Text Cleaner (`text_cleaner.py`)

**Location**: `src/data_processing/text_cleaner.py`

**Function**: `clean_text(text: str) -> str`

**Purpose**: Normalize and clean raw text data.

**Operations**:
1. **Unicode Normalization**: Convert to NFC form
2. **Whitespace Normalization**:
   - Remove extra spaces
   - Normalize line breaks
   - Remove leading/trailing whitespace
3. **Special Character Handling**:
   - Remove or normalize special characters
   - Keep Hindi punctuation (।, ॥)
4. **HTML Entity Removal**: Strip any HTML tags or entities

**Example**:
```python
raw_text = "यह    एक   \n\n   परीक्षण  है।  "
clean_text = clean_text(raw_text)
# Result: "यह एक परीक्षण है।"
```

### 3. Quality Filter (`quality_filter.py`)

**Class**: `QualityFilter`

**Location**: `src/data_processing/quality_filter.py:4`

**Purpose**: Apply quality checks to ensure high-quality corpus.

**Configuration**:
```python
QualityFilter(
    min_length=10,        # Minimum characters
    max_length=1000,      # Maximum characters
    min_hindi_ratio=0.8   # Minimum ratio of Hindi/Devanagari characters
)
```

#### Methods:

**`filter_by_length()` (line 10)**
Filters texts based on character length.

```python
def filter_by_length(self, texts: List[str]) -> List[str]:
    """
    Remove texts that are too short or too long

    - Too short: Likely incomplete or low-quality
    - Too long: May be concatenated documents or contain noise
    """
```

**`filter_by_language()` (line 15)**
Ensures texts are primarily in Hindi.

```python
def filter_by_language(self, texts: List[str]) -> List[str]:
    """
    Filter texts with high Hindi character ratio

    Devanagari Unicode Ranges:
        - 0x0900-0x097F: Devanagari
        - 0xA8E0-0xA8FF: Devanagari Extended

    Texts with < min_hindi_ratio are removed
    """
```

**Rationale**: Removes texts with too much English, numbers, or other scripts.

**`calculate_readability_score()` (line 48)**
Calculates text readability based on sentence and word complexity.

```python
def calculate_readability_score(self, text: str) -> float:
    """
    Calculate readability score

    Factors:
        - Average sentence length
        - Average word length

    Returns:
        Score from 0-1 (higher = more readable)
    """
```

**Readability Formula**:
```
score = 1.0 / (1.0 + (avg_sentence_length / 20.0) + (avg_word_length / 10.0))
```

### 4. Deduplicator (`deduplicator.py`)

**Class**: `TextDeduplicator`

**Location**: `src/data_processing/deduplicator.py:5`

**Purpose**: Remove duplicate and near-duplicate texts to ensure corpus diversity.

**Technology**: MinHash LSH (Locality-Sensitive Hashing)

**Configuration**:
```python
TextDeduplicator(
    threshold=0.8,    # Similarity threshold for near-duplicates
    num_perm=128      # Number of permutations for MinHash
)
```

#### Two-Pass Deduplication:

**Pass 1: Exact Duplicate Detection** (line 25)
```python
def get_text_hash(self, text: str) -> str:
    """Generate MD5 hash for exact duplicate detection"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()
```
- Uses MD5 hashing
- O(n) time complexity
- Removes identical texts

**Pass 2: Fuzzy Duplicate Detection** (line 36)
```python
def get_minhash(self, text: str) -> MinHash:
    """Generate MinHash for fuzzy duplicate detection"""
    m = MinHash(num_perm=self.num_perm)
    words = text.split()
    for word in words:
        m.update(word.encode('utf-8'))
    return m
```
- Uses MinHash LSH algorithm
- Detects near-duplicates (similarity ≥ threshold)
- Efficient for large corpora

**Algorithm**:
1. Compute hash for each text
2. Track exact duplicates via hash collisions
3. For non-duplicates:
   - Compute MinHash signature
   - Query LSH index for similar texts
   - If no similar texts found, add to corpus
   - Otherwise, mark as duplicate

**Result**: Corpus with no exact or near-duplicate texts

### 5. Data Mixer (`data_mixer.py`)

**Location**: `src/data_processing/data_mixer.py`

**Purpose**: Mix data from multiple sources with specified ratios.

**Mixing Strategies**:
1. **Uniform Mixing**: Equal representation from all sources
2. **Weighted Mixing**: Specified proportions (e.g., 60% IndicCorp, 30% Wikipedia, 10% Children's)
3. **Developmental Mixing**: Prioritize simpler texts early in training

**Example Configuration**:
```yaml
data_mixing:
  strategy: weighted
  ratios:
    indiccorp: 0.6
    wikipedia: 0.3
    childrens_books: 0.1
```

### 6. Corpus Analyzer (`corpus_analyzer.py`)

**Location**: `src/data_processing/corpus_analyzer.py`

**Purpose**: Generate statistics and insights about the corpus.

**Analyses**:
1. **Token Statistics**:
   - Total tokens
   - Unique tokens (types)
   - Type-Token Ratio (lexical diversity)

2. **Text Statistics**:
   - Number of documents
   - Average document length
   - Length distribution

3. **Character Statistics**:
   - Character frequency
   - Script distribution (Devanagari vs. Latin vs. Other)

4. **Linguistic Statistics**:
   - Sentence length distribution
   - Word length distribution
   - Punctuation frequency

5. **Source Distribution**:
   - Proportion from each data source
   - Topic distribution (if available)

**Output**: JSON file with comprehensive statistics

## PyTorch Dataset

**Class**: `TextDataset`

**Location**: `src/data_processing/corpus_builder.py:19`

**Purpose**: PyTorch Dataset wrapper for text data.

**Implementation**:
```python
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text)

        # Truncate or pad to max_length
        if len(encoding) > self.max_length:
            encoding = encoding[:self.max_length]
        else:
            encoding = encoding + [0] * (self.max_length - len(encoding))

        return {
            'input_ids': torch.tensor(encoding, dtype=torch.long),
            'attention_mask': torch.tensor([1 if x != 0 else 0 for x in encoding], dtype=torch.long)
        }
```

**Features**:
- Tokenization on-the-fly
- Automatic truncation/padding
- Attention mask generation
- Compatible with PyTorch DataLoader

## Complete Pipeline Example

```python
from src.data_processing.corpus_builder import CorpusBuilder
from src.utils.experiment_config import ExperimentConfig

# Load configuration
config = ExperimentConfig.load_config('configs/base_config.yaml')

# Create corpus builder
corpus_builder = CorpusBuilder(config)

# Step 1: Collect raw data
raw_data = corpus_builder.collect_all_data()
# Output: data/raw/raw_corpus.pkl

# Step 2: Process and filter
processed_data = corpus_builder.process_and_filter(raw_data)
# - Cleans text
# - Filters by length and language
# - Deduplicates
# - Limits to max_tokens

# Step 3: Create splits
splits = corpus_builder.create_splits(processed_data)
# Output: {'train': [...], 'val': [...], 'test': [...]}

# Step 4: Save splits
corpus_builder.save_splits(splits)
# Output: data/processed/{train,val,test}.pkl
#         data/processed/metadata.json

# Later: Load splits
splits = corpus_builder.load_splits()

# Create DataLoader for training
train_dataloader = corpus_builder.create_dataloader(
    splits['train'],
    tokenizer,
    'train'
)
```

## Data Quality Metrics

After processing, the following metrics are tracked:

| Metric | Description | Target Range |
|--------|-------------|--------------|
| Total Tokens | Approximate token count | ~10M |
| Unique Documents | Number of unique texts | Varies |
| Average Document Length | Characters per document | 100-500 |
| Devanagari Ratio | % of Devanagari characters | > 80% |
| Deduplication Rate | % of duplicates removed | 10-30% |
| Train/Val/Test Split | Proportion of each split | 80/10/10 |

## Best Practices

### 1. Data Collection
- **Diverse Sources**: Use multiple sources for better generalization
- **Quality over Quantity**: Prioritize clean, high-quality text
- **Source Balance**: Avoid over-representation from single source

### 2. Filtering
- **Conservative Length Bounds**: Don't filter too aggressively
- **Language Detection**: Ensure high Hindi content
- **Manual Inspection**: Spot-check filtered data

### 3. Deduplication
- **Threshold Tuning**: Adjust similarity threshold based on corpus
- **Exact + Fuzzy**: Use both exact and fuzzy matching
- **Preserve Diversity**: Don't over-deduplicate

### 4. Token Limiting
- **Approximation**: Use word count × 1.3 as token estimate
- **Source Balance**: Maintain source diversity when limiting
- **Stratified Sampling**: Sample proportionally from sources

## Troubleshooting

### Issue: Low Hindi Character Ratio
**Cause**: Source contains too much English or other scripts
**Solution**: Adjust `min_hindi_ratio` in QualityFilter or filter source

### Issue: Too Much Deduplication
**Cause**: Threshold too low or repetitive source
**Solution**: Increase similarity threshold or check source quality

### Issue: Insufficient Tokens
**Cause**: Aggressive filtering or small sources
**Solution**: Add more sources or relax filtering criteria

### Issue: Memory Errors
**Cause**: Large corpus processing in memory
**Solution**: Implement streaming processing or batch processing

## Interactive Corpus Analysis (Phase 2)

For detailed interactive analysis of the processed corpus, use the **Data Exploration Notebook**:

**Notebook**: `notebooks/01_data_exploration.ipynb`

**Features**:
1. **Basic Statistics**:
   - Total tokens, unique tokens, type-token ratio
   - Average/median sentence lengths
   - Corpus size by split (train/val/test)

2. **Length Distribution Analysis**:
   - Word count histograms
   - Character count distributions
   - Statistical summaries (mean, median, std)

3. **Character Analysis**:
   - Top 30 Devanagari characters
   - Hindi ratio (Devanagari / total characters)
   - Character frequency visualization

4. **Word Frequency Analysis**:
   - Top 30 most frequent words
   - Vocabulary distribution
   - Function word analysis

5. **Morphological Complexity**:
   - Case marker frequency (ने, को, से, में, पर, का, की, के)
   - Morphological richness assessment
   - Agreement marker distribution

6. **Data Quality Assessment**:
   - Quality distribution (clean vs. filtered)
   - Rejection reasons breakdown
   - Quality metrics visualization

**Generated Outputs**:
- `figures/length_distributions.png` - Sentence length histograms
- `figures/character_distribution.png` - Top Devanagari characters
- `figures/word_frequency.png` - Most frequent words
- `figures/case_markers.png` - Hindi case marker distribution
- `figures/data_quality.png` - Quality assessment pie chart
- `data/corpus_statistics.json` - Comprehensive statistics

**Usage**:
```bash
# Launch Jupyter Lab
jupyter lab notebooks/01_data_exploration.ipynb

# Or run non-interactively
jupyter nbconvert --execute --to notebook \
    --inplace notebooks/01_data_exploration.ipynb
```

**Key Statistics Example**:
```python
# From corpus_statistics.json
{
  "dataset_statistics": {
    "total_words": 10234567,
    "unique_words": 234567,
    "type_token_ratio": 0.0229,
    "avg_sentence_length": 15.3,
    "median_sentence_length": 12.0
  },
  "quality_assessment": {
    "clean": 87.5%,
    "too_short": 5.2%,
    "too_long": 2.1%,
    "low_hindi_ratio": 3.8%,
    "has_urls": 1.4%
  },
  "case_markers": {
    "का": 145234,
    "की": 98765,
    "के": 87654,
    "को": 76543,
    "ने": 65432,
    "से": 54321,
    "में": 43210,
    "पर": 32109
  }
}
```

This interactive analysis complements the automated pipeline by providing:
- Visual insights into corpus characteristics
- Statistical validation of data quality
- Publication-ready figures for thesis
- Reproducible analysis workflow

For more details, see [Jupyter Notebooks Documentation](09_JUPYTER_NOTEBOOKS.md).

## Future Improvements

1. **Streaming Processing**: Process data in chunks for memory efficiency
2. **Parallel Processing**: Utilize multiprocessing for filtering
3. **Advanced Filtering**:
   - Toxicity filtering
   - Topic classification
   - Quality scoring with ML models
4. **Data Augmentation**:
   - Paraphrasing
   - Back-translation
   - Synthetic data generation
5. **Curriculum Data**: Order by linguistic complexity

## Related Documentation

- [Tokenization Documentation](03_TOKENIZATION.md)
- [Configuration Guide](07_CONFIGURATION.md)
- [Jupyter Notebooks Documentation](09_JUPYTER_NOTEBOOKS.md) - Interactive corpus analysis
- [Project Overview](01_PROJECT_OVERVIEW.md) - Complete pipeline architecture

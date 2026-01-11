# Data Processing Pipeline

## Overview

The data processing pipeline is responsible for collecting, cleaning, filtering, and preparing Hindi text data for language model training. The goal is to create a high-quality corpus of approximately 10 million and 100 million words from diverse sources while maintaining linguistic quality and removing duplicates.

## Architecture

```
Raw Data Sources → Collection → Cleaning → Quality Filtering → Deduplication → Token Limiting → Train/Val/Test Splits
```

## Data Sources

### 1. IndicCorp Hindi (`indiccorp_downloader.py`)

**Description**: Large-scale corpus of Hindi text from news articles and web crawls.

**Implementation**: `src/data_processing/downloaders/indiccorp_downloader.py`

**Dataset Information**:
- **Source**: AI4Bharat/HuggingFace (`ai4bharat/IndicCorpusV2`)
- **Language**: Hindi (hi)
- **Files**: hi-1.txt (26.7 GB), hi-2.txt (26.7 GB), hi-3.txt (26.7 GB)
- **Total Size**: ~80.1 GB for all three files
- **License**: CC0-1.0 (Public Domain)
- **Content**: Web-crawled text, news articles, blogs

#### Main Class: `IndicCorpDownloader`

**Location**: `src/data_processing/downloaders/indiccorp_downloader.py:39`

A streamlined downloader class that downloads specific Hindi text files from IndicCorp V2 using HuggingFace Hub.

**Key Methods**:

**`download()`** (line 76) - Download Hindi files from HuggingFace Hub:
```python
def download(
    self,
    files: Optional[List[str]] = None,
    max_lines: Optional[int] = None,
    clean_texts: bool = True
) -> tuple[List[str], int]:
    """
    Download and read Hindi files from IndicCorp V2.

    Args:
        files: List of filenames to download (default: ['hi-1.txt'])
        max_lines: Maximum number of lines to read (None = all)
        clean_texts: Whether to apply text cleaning

    Returns:
        Tuple of (list of text strings, total word count)
    """
```

**`get_source_info()`** (line 199) - Get source metadata:
```python
def get_source_info(self) -> Dict[str, Any]:
    """
    Get information about the IndicCorp data source.

    Returns:
        Dict with source metadata (repository, language, license, etc.)
    """
```

#### Convenience Function

**`download_indiccorp_hindi()`** (line 216) - One-line download:

```python
from src.data_processing.downloaders.indiccorp_downloader import download_indiccorp_hindi

# Download single file (hi-1.txt ~26.7GB) - default for efficiency
cache_path = download_indiccorp_hindi(
    output_dir='data/raw',
    max_lines=100000,
    clean_texts=True,
    save_stats=True
)

# Download all three files if needed (~80GB total)
cache_path = download_indiccorp_hindi(
    output_dir='data/raw',
    files=['hi-1.txt', 'hi-2.txt', 'hi-3.txt'],
    max_lines=100000,
    save_stats=True
)

# Returns path to cached pickle file
```

#### Command Line Usage

```bash
# Download 100K lines (downloads hi-1.txt by default)
python src/data_processing/downloaders/indiccorp_downloader.py \
    --output-dir data/raw \
    --max-lines 100000

# Download all three files
python src/data_processing/downloaders/indiccorp_downloader.py \
    --output-dir data/raw \
    --files hi-1.txt hi-2.txt hi-3.txt \
    --max-lines 100000 \
    --cache-dir /custom/cache
```

#### Advanced Usage

```python
from src.data_processing.downloaders import IndicCorpDownloader

# Initialize downloader
downloader = IndicCorpDownloader(
    output_dir='data/raw',
    cache_dir='/custom/cache'  # Optional custom cache
)

# Download with full control
texts, total_words = downloader.download(
    files=['hi-1.txt'],
    max_lines=100000,
    clean_texts=True
)

print(f"Downloaded {len(texts)} texts with {total_words} words")

# Use BaseDownloader's process_and_cache for complete pipeline
cache_path = downloader.process_and_cache(
    max_samples=None,  # We limit during download
    save_stats=True,
    files=['hi-1.txt'],
    max_lines=100000,
    clean_texts=True
)
```

#### Output Files

When using `process_and_cache()`:
1. **`indiccorp.pkl`** - Python pickle file (list of strings)
2. **`indiccorp_metadata.json`** - Source metadata and statistics
3. **`indiccorp_sample.txt`** - First 100 samples for inspection

**Characteristics**:
- **Source**: News articles, web content, blogs
- **Size**: Configurable via `max_lines` parameter
- **Quality**: High, formal register
- **Language**: Modern Standard Hindi
- **Incremental Processing**: Word counts calculated during download to avoid memory spikes

**Pros**:
- Large scale, high quality
- Formal, well-edited text
- Diverse topics (news, web, blogs)
- HuggingFace Hub integration (automatic caching)
- Efficient file-based downloads
- Incremental word counting for memory efficiency

**Cons**:
- Formal register may not match child language
- May contain some web scraping noise
- Requires internet connection for initial download
- Large file sizes (26.7 GB per file)

### 2. Hindi Wikipedia (`wiki_downloader.py`)

**Description**: Encyclopedia articles from Hindi Wikipedia using HuggingFace Datasets.

**Implementation**: `src/data_processing/downloaders/wiki_downloader.py`

**Dataset Information**:
- **Source**: HuggingFace `wikimedia/wikipedia` (version: 20231101.hi)
- **Language**: Hindi (hi)
- **Content**: Pre-processed Wikipedia dump
- **Quality**: High, edited content
- **Format**: Parquet via datasets library

**Key Class: `WikiDownloader`**

**Location**: `src/data_processing/downloaders/wiki_downloader.py:22`

Extends `BaseDownloader` with Wikipedia-specific functionality:

```python
def download(
    self,
    max_articles: Optional[int] = None,
    min_length: int = 30,
    max_length: int = 2000
) -> tuple[List[str], int]:
    """
    Download Hindi Wikipedia from HuggingFace.

    Args:
        max_articles: Maximum number of articles to download (None = all)
        min_length: Minimum character length for an article
        max_length: Maximum character length for an article

    Returns:
        Tuple of (list of article texts, total word count)
    """
```

#### Convenience Function

**`download_wikipedia_hindi()`** (line 149) - One-line download:

```python
from src.data_processing.downloaders.wiki_downloader import download_wikipedia_hindi

# Download Wikipedia articles
cache_path = download_wikipedia_hindi(
    output_dir='data/raw',
    dataset_version='20231101.hi',
    max_articles=25000,
    min_length=50,
    max_length=50000,
    save_stats=True
)

# Returns path to cached pickle file
```

#### Advanced Usage

```python
from src.data_processing.downloaders import WikiDownloader

# Initialize downloader
downloader = WikiDownloader(
    output_dir='data/raw',
    dataset_version='20231101.hi',
    cache_dir='/custom/cache'  # Optional
)

# Download with full control
texts, total_words = downloader.download(
    max_articles=25000,
    min_length=30,
    max_length=2000
)

print(f"Downloaded {len(texts)} articles with {total_words} words")

# Use BaseDownloader's process_and_cache for complete pipeline
cache_path = downloader.process_and_cache(
    max_samples=None,  # We filter during download
    save_stats=True,
    max_articles=25000,
    min_length=30,
    max_length=2000
)
```

#### Output Files

When using `process_and_cache()`:
1. **`wikipedia.pkl`** - Python pickle file (list of article texts)
2. **`wikipedia_metadata.json`** - Source metadata and statistics
3. **`wikipedia_sample.txt`** - First 100 articles for inspection

**Filtering Parameters**:
- `min_length`: Minimum article length (default: 30 characters)
- `max_length`: Maximum article length (default: 2000 characters)
- Filters out stub articles automatically based on length

**Characteristics**:
- **Source**: Pre-processed Wikipedia dumps
- **Size**: Configurable via `max_articles` parameter (thousands of articles available)
- **Quality**: High, well-structured, edited content
- **Language**: Formal encyclopedic Hindi
- **Incremental Processing**: Word counts calculated during download to avoid memory spikes

**Pros**:
- ✅ No web scraping needed (uses HuggingFace dataset)
- ✅ Pre-processed and cleaned
- ✅ High quality, edited content
- ✅ Structured information
- ✅ Diverse topics
- ✅ Reliable and reproducible
- ✅ Incremental word counting for memory efficiency

**Cons**:
- ❌ Formal register
- ❌ May contain technical jargon
- ❌ Encyclopedia style may not match natural language

### 3. IndicDialogue - Conversational Hindi (`indicdialogue_loader.py`)

**Description**: Movie subtitle data providing conversational and informal Hindi.

**Implementation**: `src/data_processing/downloaders/indicdialogue_loader.py`

**Dataset Information**:
- **Source**: Local JSONL file (`Hindi.jsonl` or `hindi.jsonl`) from Mendeley dataset
- **URL**: https://data.mendeley.com/datasets/wcb4bxbyxx/2
- **Content**: Movie subtitles (dialogues)
- **Language**: Conversational/informal Hindi
- **Format**: JSONL (JSON Lines)
- **Note**: Must be manually downloaded and placed in `data/raw/` directory

**Key Class: `IndicDialogueLoader`**

**Location**: `src/data_processing/downloaders/indicdialogue_loader.py:23`

Extends `BaseDownloader` for dialogue data:

```python
def download(
    self,
    max_movies: Optional[int] = None,
    min_dialogue_length: int = 10,
    combine_dialogues: bool = False
) -> tuple[List[str], int]:
    """
    Load IndicDialogue from JSONL file.

    Args:
        max_movies: Maximum number of movies to process (None = all)
        min_dialogue_length: Minimum characters per dialogue line
        combine_dialogues: If True, combine all dialogues from a movie into one text.
                          If False, keep each dialogue line separate.

    Returns:
        Tuple of (list of dialogue texts, total word count)
    """
```

#### Convenience Function

**`load_indicdialogue_hindi()`** (line 172) - One-line load:

```python
from src.data_processing.downloaders.indicdialogue_loader import load_indicdialogue_hindi

# Load dialogue data
cache_path = load_indicdialogue_hindi(
    jsonl_path='data/raw/Hindi.jsonl',
    output_dir='data/raw',
    max_movies=None,  # All movies
    min_dialogue_length=10,
    combine_dialogues=False,  # Keep each dialogue separate
    save_stats=True
)

# Returns path to cached pickle file
```

#### Advanced Usage

```python
from src.data_processing.downloaders import IndicDialogueLoader

# Initialize loader
loader = IndicDialogueLoader(
    jsonl_path='data/raw/Hindi.jsonl',
    output_dir='data/raw'
)

# Load with full control
texts, total_words = loader.download(
    max_movies=100,
    min_dialogue_length=10,
    combine_dialogues=False
)

print(f"Loaded {len(texts)} dialogues with {total_words} words")

# Use BaseDownloader's process_and_cache for complete pipeline
cache_path = loader.process_and_cache(
    max_samples=None,  # We filter during loading
    save_stats=True,
    max_movies=None,
    min_dialogue_length=10,
    combine_dialogues=False
)
```

#### Output Files

When using `process_and_cache()`:
1. **`indicdialogue.pkl`** - Python pickle file (list of dialogue texts)
2. **`indicdialogue_metadata.json`** - Source metadata and statistics
3. **`indicdialogue_sample.txt`** - First 100 dialogues for inspection

**Aggregation Modes**:
1. **Dialogue-level** (`combine_dialogues=False`): Each dialogue turn as separate text
   - More samples, shorter texts
   - Better for learning conversational patterns

2. **Movie-level** (`combine_dialogues=True`): All dialogues from a movie aggregated
   - Fewer samples, longer texts
   - Preserves narrative context

**Characteristics**:
- **Register**: Conversational, informal
- **Style**: Natural spoken Hindi with code-mixing
- **Quality**: Variable (subtitle quality)
- **Unique feature**: Captures spoken language patterns not found in written corpora
- **Incremental Processing**: Word counts calculated during loading to avoid memory spikes

**Pros**:
- ✅ Conversational/informal register
- ✅ Natural spoken language patterns
- ✅ Complements formal written sources (IndicCorp, Wikipedia)
- ✅ Captures code-mixing and colloquialisms
- ✅ Dialogue structure
- ✅ Incremental word counting for memory efficiency

**Cons**:
- ❌ Subtitle quality varies
- ❌ May contain transcription errors
- ❌ Requires manual download (not automated)
- ❌ Limited to movie domain

**Why Include This Source?**
- Provides **register diversity**: Balances formal written Hindi with conversational patterns
- Important for **language learning**: Models should understand both formal and informal Hindi
- Captures **code-mixing**: Common in modern Hindi usage

### 4. Children's Books (`childrens_books.py`)

**Description**: Collection of children's literature in Hindi from StoryWeaver (Pratham Books).

**Implementation**: `src/data_processing/childrens_books.py`

**Dataset Information**:
- **Source**: StoryWeaver API (storyweaver.org.in by Pratham Books)
- **Content**: Openly-licensed children's stories
- **Language**: Simple, age-appropriate Hindi
- **Quality**: High, curated content
- **Access**: API-based collection with rate limiting

**Key Class: `ChildrensStoryCollector`**

**Location**: `src/data_processing/childrens_books.py:26`

Collects Hindi children's stories from StoryWeaver with robust error handling:

```python
def collect_all_stories(self) -> tuple[List[str], int]:
    """
    Main entry point: Collect stories from StoryWeaver
    Returns empty list if collection fails (non-blocking)

    Returns:
        Tuple of (list of story texts, total word count)
    """
```

**Key Methods**:

**`scrape_storyweaver_api()`** (line 99) - Collect stories via API:
```python
def scrape_storyweaver_api(self) -> List[Dict]:
    """
    Collect stories from StoryWeaver using their public API

    Uses: https://storyweaver.org.in/api/v1/books-search
    - Filters for Hindi language
    - Collects from multiple pages
    - Respects rate limits (2 second delay)

    Returns:
        List of story dictionaries with metadata
    """
```

**`filter_stories()`** (line 337) - Quality and age-appropriateness filtering:
```python
def filter_stories(self, stories: List[Dict]) -> List[Dict]:
    """
    Filter stories for quality and age-appropriateness

    Filters:
    - Length: 100-10,000 characters
    - Word count: 20-2,000 words
    - Hindi content check (≥80% Devanagari)
    - Age-appropriate complexity

    Returns:
        Filtered list of stories
    """
```

#### Main Entry Point

**`collect_childrens_stories()`** (line 404) - Convenience function:

```python
from src.data_processing.childrens_books import collect_childrens_stories

# Collect children's stories
stories, total_words = collect_childrens_stories(max_stories=2000)

print(f"Collected {len(stories)} stories with {total_words} words")
```

#### Advanced Usage

```python
from src.data_processing.childrens_books import ChildrensStoryCollector

# Initialize collector
collector = ChildrensStoryCollector(
    max_stories=2000,
    rate_limit_delay=2.0  # Seconds between requests
)

# Collect stories
stories, total_words = collector.collect_all_stories()

print(f"Collected {len(stories)} stories with {total_words} words")
```

**Filtering Criteria**:
- **Length**: 100-10,000 characters
- **Word Count**: 20-2,000 words
- **Hindi Ratio**: ≥80% Devanagari characters
- **Average Word Length**: ≤10 characters (age-appropriate)
- **Content**: No complex vocabulary (filtered for children's level)

**Characteristics**:
- **Source**: StoryWeaver (Pratham Books) - openly-licensed children's literature
- **Size**: Configurable (default: 2000 stories)
- **Quality**: High, curated, age-appropriate
- **Language**: Simple, natural Hindi suitable for children
- **Rate Limiting**: 2-second delay between API requests
- **Error Handling**: Non-blocking - returns empty list if collection fails
- **Incremental Processing**: Word counts calculated during collection to avoid memory spikes

**Pros**:
- Developmentally plausible complexity
- Simple vocabulary and structures
- Natural language patterns
- High-quality curated content
- API-based (reliable, no web scraping)
- Openly licensed (CC BY)
- Incremental word counting for memory efficiency
- Robust error handling (doesn't crash pipeline)

**Cons**:
- Smaller corpus size compared to other sources
- May be too simple for some tasks
- Requires internet connection
- Rate-limited API (slower collection)
- Collection may fail if API is unavailable (gracefully handles failures)

## Pipeline Components

### 1. Corpus Builder (`corpus_builder.py`)

**Main Class**: `CorpusBuilder`

**Location**: `src/data_processing/corpus_builder.py:67`

**Purpose**: Orchestrates the entire data processing pipeline with smart caching and balanced split creation.

**Key Features**:
- **Smart Caching**: Per-source cache files to avoid re-downloading
- **Balanced Splits**: New method maintains source distribution across train/val/test
- **Memory Optimization**: Index-based shuffling and incremental word counting
- **Data Provenance**: Tracks statistics and lineage through entire pipeline
- **Separate Word Limits**: Configurable word limits for train/val/test splits

**Key Methods**:

#### `collect_all_data()` (line 147)
Collects data from all sources with smart caching.

```python
def collect_all_data(self, force_redownload: bool = False) -> Dict[str, List[str]]:
    """
    Collect data from all sources with smart caching

    Args:
        force_redownload: If True, ignore cache and download fresh data

    Returns:
        Dictionary mapping source names to lists of text samples
        {
            'indiccorp': [...],
            'wikipedia': [...],
            'indicdialogue': [...],
            'childrens_books': [...]
        }
    """
```

**Process**:
1. Check per-source cache files (`.pkl` in `data/raw/`)
2. If cache exists and `force_redownload=False`, load from cache
3. Otherwise, download from source:
   - IndicCorp Hindi dataset (via HuggingFace Hub)
   - Wikipedia articles (via HuggingFace datasets)
   - IndicDialogue (from local JSONL)
   - Children's stories (via StoryWeaver API)
4. Save each source to separate cache file
5. Track raw data statistics with `DataProvenanceTracker`

**Cache Files**:
- `data/raw/indiccorp.pkl`
- `data/raw/wikipedia.pkl`
- `data/raw/indicdialogue.pkl`
- `data/raw/childrens_stories.pkl`

**Benefits of Smart Caching**:
- Avoid re-downloading data on subsequent runs
- Faster pipeline iteration during development
- Can force re-download with `force_redownload=True`

#### `process_and_filter()` (line 329)
Applies cleaning and filtering to collected data.

```python
def process_and_filter(
    self,
    raw_data: Dict[str, List[str]],
    preserve_sources: bool = False
) -> Dict[str, List[str]]:
    """
    Process and filter collected data

    Args:
        raw_data: Dictionary with source names as keys and lists of texts as values
        preserve_sources: If True, return processed data grouped by source (for new split creation)
                         If False, return combined list (for legacy split creation)

    Returns:
        If preserve_sources=True: Dictionary with source names as keys
        If preserve_sources=False: Dictionary with single key 'combined'
    """
```

**Filtering Pipeline (Per-Source)**:
1. **Text Cleaning**: Normalize Unicode, remove extra whitespace, URLs, noise patterns
2. **Length Filtering**: Remove texts outside bounds (default: 30-2000 characters)
3. **Language Filtering**: Ensure ≥80% Devanagari character ratio
4. **Deduplication**:
   - If `preserve_sources=True`: Deduplicate within each source separately
   - If `preserve_sources=False`: Combine all sources, then deduplicate
5. **Statistics Tracking**: Track filtering and deduplication stats with `DataProvenanceTracker`

**Key Change**: Now supports per-source processing to maintain source information for balanced split creation.

#### `create_balanced_splits_with_limits()` (line 424)
**NEW METHOD** - Creates balanced train/val/test splits with separate word limits.

```python
def create_balanced_splits_with_limits(
    self,
    raw_data: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Create balanced train/val/test splits with separate word limits

    Strategy:
    1. Apply global deduplication first (no text appears in multiple splits)
    2. Training split: Use configured train_source_ratios until train_word_limit reached
    3. Validation split: Take equal proportions from each source until val_word_limit reached
    4. Test split: Take equal proportions from remaining data until test_word_limit reached

    Args:
        raw_data: Dictionary with keys as source names and values as lists of texts

    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
```

**Process**:
1. **Analyze Capacity**: Calculate available words per source
2. **Check Constraints**: Compare capacity vs. demand for each source
3. **Adjust Targets**: For constrained sources, allocate capacity proportionally
4. **Create Training Split**: Use adjusted targets respecting source ratios
5. **Create Validation Split**: Equal representation from each source
6. **Create Test Split**: Equal representation from remaining data
7. **Track Statistics**: Save detailed split statistics with `DataProvenanceTracker`
8. **Memory Cleanup**: Clear intermediate data structures, run garbage collection

**Key Features**:
- **Separate Word Limits**: `train_word_limit`, `val_word_limit`, `test_word_limit` from config
- **Source Balance**: Maintains configured source ratios in training, equal ratios in val/test
- **Global Deduplication**: No text appears in multiple splits (hash-based tracking)
- **Memory Optimization**: Index-based shuffling instead of copying data (~20GB savings)
- **Capacity-Aware**: Adjusts targets when source data is insufficient

**Configuration** (from `configs/base_config.yaml`):
```yaml
train_word_limit: 10000000  # 10M words
val_word_limit: 10000000    # 10M words
test_word_limit: 10000000   # 10M words

train_source_ratios:
  indiccorp: 0.60
  wikipedia: 0.25
  indicdialogue: 0.10
  childrens_books: 0.05
```

#### `create_splits()` (line 731)
**DEPRECATED** - Legacy method for backward compatibility.

```python
def create_splits(self, processed_data: List[str]) -> Dict[str, List[str]]:
    """
    Create train/val/test splits (legacy method for backward compatibility)

    NOTE: This method is deprecated. Use create_balanced_splits_with_limits() for new pipeline.
    """
```

**Process**:
1. Shuffle combined data with fixed random seed (42)
2. Calculate split sizes based on ratios
3. Create splits

**Use**: Only for backward compatibility. New code should use `create_balanced_splits_with_limits()`.

#### `save_splits()` (line 762)
Saves processed splits to disk with comprehensive metadata.

**Outputs**:
- `data/splits/train.pkl` - Training data (pickle)
- `data/splits/val.pkl` - Validation data (pickle)
- `data/splits/test.pkl` - Test data (pickle)
- `data/splits/train.txt` - First 100 samples as text (for inspection)
- `data/splits/val.txt` - First 100 samples as text
- `data/splits/test.txt` - First 100 samples as text
- `data/splits/metadata.json` - Metadata about splits
- `results/data_processing/data_provenance_report.json` - Comprehensive provenance tracking

**New Feature**: Data provenance report includes detailed statistics about:
- Raw data collection per source
- Filtering and deduplication statistics
- Split creation details with per-source breakdowns
- Word counts and document counts

#### `load_splits()` (line 803)
Loads processed splits from disk.

```python
def load_splits(self) -> Dict[str, List[str]]:
    """
    Load processed splits from disk

    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
```

#### `create_dataloader()` (line 821)
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

**Configuration** (from config):
- `batch_size`: Batch size for DataLoader
- `num_workers`: Number of worker processes (default: 4, was 64)
- `pin_memory`: Pin memory for faster GPU transfer (default: True)
- `persistent_workers`: Set to False for memory efficiency

**Memory Optimization**: Reduced `num_workers` from 64 to 4 saves 30-60GB RAM with minimal performance impact.

### 2. Text Cleaner (`text_cleaner.py`)

**Location**: `src/data_processing/text_cleaner.py`

**Main Class**: `HindiTextCleaner`

**Location**: `src/data_processing/text_cleaner.py:6`

A comprehensive Hindi text cleaning class with configurable operations.

**Key Methods**:

```python
def clean_text(
    self,
    text: str,
    remove_urls: bool = True,
    remove_noise: bool = True,
    normalize_punctuation: bool = True
) -> str:
    """
    Complete text cleaning pipeline with configurable options.

    Args:
        text: Input text to clean
        remove_urls: Whether to remove URLs
        remove_noise: Whether to remove common noise patterns
        normalize_punctuation: Whether to normalize repeated punctuation

    Returns:
        Cleaned text string
    """
```

**Operations**:
1. **Unicode Normalization**: Convert to NFC form for consistency
2. **URL Removal**: Remove HTTP/HTTPS URLs (if enabled)
3. **Noise Pattern Removal**: Remove common noise like "click here", "download", "read more" (if enabled)
4. **Non-Hindi Character Removal**: Keep only:
   - Devanagari characters (0x0900-0x097F)
   - Devanagari Extended (0xA8E0-0xA8FF)
   - Basic punctuation (।, ॥, common punctuation marks)
   - Digits
   - Whitespace
5. **Whitespace Normalization**:
   - Replace multiple spaces with single space
   - Remove leading/trailing whitespace
   - Normalize spacing around punctuation
6. **Punctuation Normalization**: Normalize repeated punctuation marks (if enabled)
7. **Special Case Handling**:
   - Convert English digits to Devanagari (0-9 → ०-९)
   - Remove zero-width characters
   - Normalize danda (।) and double danda (॥)

**Module-Level Convenience Function**:

```python
from src.data_processing.text_cleaner import clean_text

# Simple usage (all cleaning operations enabled)
cleaned = clean_text(text)
```

**Centralized Utilities**:
- Uses `HindiValidator` from utils for consistent Devanagari range handling
- Shared Unicode range definitions across all modules

**Example**:
```python
from src.data_processing.text_cleaner import HindiTextCleaner

cleaner = HindiTextCleaner()

raw_text = "यह    एक   \n\n   परीक्षण  है।  http://example.com 123"
cleaned = cleaner.clean_text(
    raw_text,
    remove_urls=True,
    remove_noise=True,
    normalize_punctuation=True
)
# Result: "यह एक परीक्षण है। १२३"
```

### 3. Quality Filter (`quality_filter.py`)

**Class**: `QualityFilter`

**Location**: `src/data_processing/quality_filter.py:5`

**Purpose**: Apply quality checks to ensure high-quality corpus.

**Configuration**:
```python
QualityFilter(
    min_length=30,        # Minimum characters (default)
    max_length=2000,      # Maximum characters (default)
    min_hindi_ratio=0.8   # Minimum ratio of Hindi/Devanagari characters
)
```

#### Methods:

**`filter_by_length()` (line 11)**
Filters texts based on character length.

```python
def filter_by_length(self, texts: List[str]) -> List[str]:
    """
    Remove texts that are too short or too long

    - Too short: Likely incomplete or low-quality
    - Too long: May be concatenated documents or contain noise
    """
```

**`filter_by_language()` (line 16)**
Ensures texts are primarily in Hindi using centralized validator.

```python
def filter_by_language(self, texts: List[str]) -> List[str]:
    """
    Filter texts with high Hindi character ratio using centralized validator

    Uses HindiValidator.calculate_hindi_ratio() for consistency

    Texts with < min_hindi_ratio are removed
    """
```

**Key Change**: Now uses centralized `HindiValidator` from utils module for consistent Hindi detection across all components.

**Rationale**: Removes texts with too much English, numbers, or other scripts.

**`detect_duplicates()` (line 28)**
Detect near-duplicate texts using fuzzy matching (legacy method).

```python
def detect_duplicates(self, texts: List[str], threshold=0.9) -> List[int]:
    """
    Detect near-duplicate texts using difflib.SequenceMatcher

    Note: For production, use TextDeduplicator class instead (faster MinHash LSH)

    Returns:
        List of duplicate indices
    """
```

**`calculate_readability_score()` (line 46)**
Calculates text readability based on sentence and word complexity.

```python
def calculate_readability_score(self, text: str) -> float:
    """
    Calculate readability score

    Factors:
        - Average sentence length (split by Hindi danda ।)
        - Average word length

    Returns:
        Score from 0-1 (higher = more readable)
    """
```

**Readability Formula**:
```
score = 1.0 / (1.0 + (avg_sentence_length / 20.0) + (avg_word_length / 10.0))
```

**Sentence Detection**: Splits by Hindi danda (।) first, falls back to period (.) if no danda found.

### 4. Deduplicator (`deduplicator.py`)

**Class**: `TextDeduplicator`

**Location**: `src/data_processing/deduplicator.py:5`

**Purpose**: Remove duplicate and near-duplicate texts to ensure corpus diversity.

**Technology**: MinHash LSH (Locality-Sensitive Hashing) from `datasketch` library

**Configuration**:
```python
TextDeduplicator(
    threshold=0.8,    # Similarity threshold for near-duplicates
    num_perm=128      # Number of permutations for MinHash
)
```

**Note**: Default `num_perm=128` in code, though config typically uses 256. Higher values = more accurate but slower.

#### Two-Pass Deduplication:

**Pass 1: Exact Duplicate Detection** (line 11)
```python
def get_text_hash(self, text: str) -> str:
    """Generate MD5 hash for exact duplicate detection"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()
```
- Uses MD5 hashing
- O(n) time complexity
- Removes identical texts

**Pass 2: Fuzzy Duplicate Detection** (line 15)
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

**`deduplicate_corpus()` (line 23)**
Main deduplication method with LSH index reset.

```python
def deduplicate_corpus(self, texts: List[str]) -> Tuple[List[str], List[int]]:
    """
    Remove duplicates and return cleaned corpus with indices

    Returns:
        Tuple of (deduplicated_texts, removed_indices)
    """
```

**Algorithm**:
1. **Reset LSH Index**: Clears index to avoid key collisions from previous runs
2. **Exact Deduplication**: Compute MD5 hash for each text, track collisions
3. **Fuzzy Deduplication**:
   - Skip texts marked as exact duplicates
   - Compute MinHash signature for remaining texts
   - Query LSH index for similar texts
   - If no similar texts found, add to index and keep text
   - Otherwise, mark as duplicate
4. **Return Results**: Deduplicated texts and list of removed indices

**Key Change**: LSH index is reset at the start of each deduplication run to prevent key collisions.

**Result**: Corpus with no exact or near-duplicate texts

### 5. Base Downloader (`base_downloader.py`)

**Class**: `BaseDownloader` (Abstract Base Class)

**Location**: `src/data_processing/downloaders/base_downloader.py:25`

**Purpose**: Provides common interface and shared functionality for all data source downloaders.

**Key Feature**: All downloader classes (IndicCorpDownloader, WikiDownloader, IndicDialogueLoader, etc.) inherit from this base class for consistency.

#### Shared Methods:

**`process_and_cache()` (line 69)**
Complete pipeline: download → sample → calculate stats → cache.

```python
def process_and_cache(
    self,
    max_samples: Optional[int] = None,
    save_stats: bool = True,
    save_sample_text: bool = True,
    sample_text_size: int = 100,
    **download_kwargs
) -> Path:
    """
    Complete pipeline: download -> sample -> calculate stats -> cache.

    This is a convenience method that handles the full workflow.

    Args:
        max_samples: Maximum number of samples to keep (None = all)
        save_stats: Whether to calculate and save statistics
        save_sample_text: Whether to save a sample text file for inspection
        sample_text_size: Number of texts to include in sample file
        **download_kwargs: Additional arguments to pass to download()

    Returns:
        Path to the cached pickle file
    """
```

**Process**:
1. Call subclass's `download()` method
2. Sample if `max_samples` specified
3. Calculate corpus statistics (word/character counts, length distributions)
4. Save to pickle file (`.pkl`)
5. Save metadata (`.json`)
6. Save sample text file (`.txt`)

**Abstract Methods** (must be implemented by subclasses):
- `download(**kwargs) -> List[str]`: Download/load data from source
- `get_source_info() -> Dict[str, Any]`: Return source metadata

**Utilities Used**:
- Uses centralized utilities from `src/data_processing/utils/`:
  - `save_pickle()` / `load_pickle()`: File I/O
  - `calculate_corpus_stats()`: Statistics calculation
  - `generate_metadata()` / `save_metadata()`: Metadata handling
  - `sample_texts()`: Random sampling
  - `configure_logger()`: Logging

**Benefits**:
- Consistent interface across all downloaders
- Reduces code duplication
- Automatic metadata and statistics generation
- Standardized output format

### 6. Data Mixer (`data_mixer.py`)

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

### New Pipeline (Recommended)

```python
from src.data_processing.corpus_builder import CorpusBuilder
from src.utils.experiment_config import ExperimentConfig

# Load configuration
config = ExperimentConfig.load_config('configs/base_config.yaml')

# Create corpus builder
corpus_builder = CorpusBuilder(config)

# Step 1: Collect raw data (with smart caching)
raw_data = corpus_builder.collect_all_data(force_redownload=False)
# Output: data/raw/{indiccorp,wikipedia,indicdialogue,childrens_stories}.pkl
# Automatically uses cache if available

# Step 2: Process and filter (per-source, preserving source info)
processed_data = corpus_builder.process_and_filter(raw_data, preserve_sources=True)
# - Cleans text per source
# - Filters by length and language per source
# - Deduplicates within each source
# - Tracks statistics with DataProvenanceTracker

# Step 3: Create balanced splits with separate word limits (NEW METHOD)
splits = corpus_builder.create_balanced_splits_with_limits(processed_data)
# Output: {'train': [...], 'val': [...], 'test': [...]}
# - Maintains source distribution across splits
# - Applies global deduplication (no text in multiple splits)
# - Respects train/val/test word limits from config
# - Memory-optimized with index-based shuffling

# Step 4: Save splits
corpus_builder.save_splits(splits)
# Output: data/splits/{train,val,test}.pkl
#         data/splits/{train,val,test}.txt (first 100 samples)
#         data/splits/metadata.json
#         results/data_processing/data_provenance_report.json

# Later: Load splits
splits = corpus_builder.load_splits()

# Create DataLoader for training
train_dataloader = corpus_builder.create_dataloader(
    splits['train'],
    tokenizer,
    'train'
)
```

### Legacy Pipeline (Deprecated)

```python
# Step 2: Process and filter (combined, for legacy split creation)
processed_data = corpus_builder.process_and_filter(raw_data, preserve_sources=False)
# Returns: {'combined': [...]}

# Step 3: Create splits (legacy method)
splits = corpus_builder.create_splits(processed_data['combined'])
# Simple ratio-based splitting (80/10/10)
# Does not maintain source distribution

# Step 4: Save splits
corpus_builder.save_splits(splits)
```

### Key Differences:

| Feature | New Pipeline | Legacy Pipeline |
|---------|-------------|-----------------|
| Split Method | `create_balanced_splits_with_limits()` | `create_splits()` |
| Source Balance | ✅ Maintained across splits | ❌ Lost after mixing |
| Word Limits | ✅ Separate per split | ❌ Combined max_words |
| Memory Usage | ✅ Optimized (index-based) | ❌ Higher (data copying) |
| Provenance Tracking | ✅ Detailed per-source stats | ❌ Limited |
| Status | **Recommended** | Deprecated |

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

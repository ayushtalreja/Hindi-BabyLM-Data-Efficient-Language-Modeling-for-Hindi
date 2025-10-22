import os
import sys
import json
import pickle
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

# Add project root to path for proper imports
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import data processing modules
try:
    from .indiccorp_downloader import download_indiccorp_hindi
    from .wiki_scraper import scrape_hindi_wikipedia
    from .childrens_books import collect_childrens_stories
    from .quality_filter import QualityFilter
    from .deduplicator import TextDeduplicator
    from .text_cleaner import clean_text
    from .data_mixer import DataMixer
    from .cache_manager import check_cache_exists, load_from_cache, save_to_cache
except ImportError:
    # Fallback for when script is run directly
    from src.data_processing.indiccorp_downloader import download_indiccorp_hindi
    from src.data_processing.wiki_scraper import scrape_hindi_wikipedia
    from src.data_processing.childrens_books import collect_childrens_stories
    from src.data_processing.quality_filter import QualityFilter
    from src.data_processing.deduplicator import TextDeduplicator
    from src.data_processing.text_cleaner import clean_text
    from src.data_processing.data_mixer import DataMixer
    from src.data_processing.cache_manager import check_cache_exists, load_from_cache, save_to_cache


class TextDataset(Dataset):
    """PyTorch Dataset for text data"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize and encode
        encoding = self.tokenizer.encode(text)

        # Truncate or pad to max_length
        if len(encoding) > self.max_length:
            encoding = encoding[:self.max_length]
        else:
            # Pad with 0 (assuming 0 is pad token)
            encoding = encoding + [0] * (self.max_length - len(encoding))

        return {
            'input_ids': torch.tensor(encoding, dtype=torch.long),
            'attention_mask': torch.tensor([1 if x != 0 else 0 for x in encoding], dtype=torch.long)
        }


class CorpusBuilder:
    """Main class for building and managing Hindi corpus"""

    def __init__(self, config):
        self.config = config
        self.data_dir = config.__dict__.get('data_dir', 'data')
        self.max_words = config.__dict__.get('max_words', config.__dict__.get('max_tokens', 10_000_000))

        # Separate word limits for each split
        self.train_word_limit = config.__dict__.get('train_word_limit', 10_000_000)
        self.val_word_limit = config.__dict__.get('val_word_limit', 10_000_000)
        self.test_word_limit = config.__dict__.get('test_word_limit', 10_000_000)

        self.train_ratio = config.train_ratio
        self.val_ratio = config.val_ratio
        self.test_ratio = config.test_ratio

        # Initialize components
        self.quality_filter = QualityFilter()
        self.deduplicator = TextDeduplicator()
        self.data_mixer = DataMixer()

        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'splits'), exist_ok=True)

    def _load_cached_source(self, source_name: str) -> Optional[List[str]]:
        """
        Load source data from cache if it exists

        Args:
            source_name: Name of source (e.g., 'indiccorp', 'wikipedia', 'childrens_stories')

        Returns:
            List of texts if cache exists, None otherwise
        """
        cache_path = os.path.join(self.data_dir, 'raw', f'{source_name}.pkl')

        if not check_cache_exists(cache_path):
            return None

        try:
            data = load_from_cache(cache_path, format='pickle')
            if data is not None:
                print(f"   ✓ Loaded {len(data):,} samples from cache: {source_name}.pkl")
            return data
        except Exception as e:
            print(f"   ⚠ Failed to load cache for {source_name}: {e}")
            return None

    def _save_source_to_cache(self, data: List[str], source_name: str):
        """
        Save source data to separate cache file

        Args:
            data: List of text samples
            source_name: Name of source (e.g., 'indiccorp', 'wikipedia', 'childrens_stories')
        """
        cache_path = os.path.join(self.data_dir, 'raw', f'{source_name}.pkl')

        try:
            success = save_to_cache(data, cache_path, format='pickle')
            if success:
                print(f"   ✓ Saved {len(data):,} samples to cache: {source_name}.pkl")
        except Exception as e:
            print(f"   ⚠ Failed to save cache for {source_name}: {e}")

    def collect_all_data(self, force_redownload: bool = False) -> Dict[str, List[str]]:
        """
        Collect data from all sources with smart caching

        Args:
            force_redownload: If True, ignore cache and download fresh data

        Returns:
            Dictionary with data from all sources
        """
        print("Collecting data from all sources...")
        if force_redownload:
            print("(Force redownload enabled - ignoring cache)\n")
        else:
            print("(Checking cache before downloading...)\n")

        all_data = {
            'indiccorp': [],
            'wikipedia': [],
            'childrens_books': []
        }

        # 1. IndicCorp - check cache first
        print("1. IndicCorp Hindi")
        cached_indiccorp = None if force_redownload else self._load_cached_source('indiccorp')
        if cached_indiccorp is not None:
            all_data['indiccorp'] = cached_indiccorp
            print(f"   Loaded {len(cached_indiccorp):,} samples from cache (skipping download)")
        else:
            print("   Cache not found, downloading...")
            try:
                # Download IndicCorp (file will be ~26.5GB on disk)
                indiccorp_paths = download_indiccorp_hindi(
                    output_dir=os.path.join(self.data_dir, 'raw')
                )

                # Process IndicCorp line-by-line to avoid loading entire 26.5GB into memory
                # We'll stream and sample to fit within 50GB memory limit
                print("   Processing IndicCorp (streaming to avoid memory issues)...")
                indiccorp_texts = []
                max_samples = self.max_words // 10  # Rough estimate

                for filename, file_path in indiccorp_paths.items():
                    if filename != 'metadata' and not filename.endswith('_pickle'):
                        line_count = 0
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    indiccorp_texts.append(line)
                                    line_count += 1

                                    # Sample only what we need to stay under memory limit
                                    if line_count >= max_samples:
                                        print(f"   Reached sample limit ({max_samples:,} lines), stopping...")
                                        break

                        print(f"   Processed {line_count:,} lines from {filename}")

                all_data['indiccorp'] = indiccorp_texts
                print(f"   Total IndicCorp samples: {len(all_data['indiccorp']):,}")

                # Save to cache for future runs
                self._save_source_to_cache(all_data['indiccorp'], 'indiccorp')
            except Exception as e:
                print(f"   ⚠ Error downloading IndicCorp: {e}")

        # 2. Wikipedia - check cache first
        print("\n2. Hindi Wikipedia")
        cached_wikipedia = None if force_redownload else self._load_cached_source('wikipedia')
        if cached_wikipedia is not None:
            all_data['wikipedia'] = cached_wikipedia
            print(f"   Loaded {len(cached_wikipedia):,} articles from cache (skipping scraping)")
        else:
            print("   Cache not found, scraping...")
            try:
                wiki_categories = ['विज्ञान', 'इतिहास', 'भूगोल', 'साहित्य', 'कला']
                wiki_articles = scrape_hindi_wikipedia(wiki_categories, max_articles=5000)
                all_data['wikipedia'] = [article['text'] for article in wiki_articles]
                print(f"   Scraped {len(all_data['wikipedia']):,} Wikipedia articles")

                # Save to cache for future runs
                self._save_source_to_cache(all_data['wikipedia'], 'wikipedia')
            except Exception as e:
                print(f"   ⚠ Error scraping Wikipedia: {e}")

        # 3. Children's Stories - check cache first
        print("\n3. Children's Stories")
        cached_stories = None if force_redownload else self._load_cached_source('childrens_stories')
        if cached_stories is not None:
            all_data['childrens_books'] = cached_stories
            print(f"   Loaded {len(cached_stories):,} stories from cache (skipping collection)")
        else:
            print("   Cache not found, collecting...")
            try:
                stories = collect_childrens_stories()
                all_data['childrens_books'] = stories
                print(f"   Collected {len(all_data['childrens_books']):,} children's stories")

                # Save to cache for future runs (even if empty)
                self._save_source_to_cache(all_data['childrens_books'], 'childrens_stories')
            except Exception as e:
                print(f"   ⚠ Error collecting children's books: {e}")
                print(f"   Continuing without children's stories...")
                all_data['childrens_books'] = []

        # Print summary
        print("\n" + "=" * 60)
        print("Data Collection Summary:")
        print(f"  IndicCorp:        {len(all_data['indiccorp']):,} samples")
        print(f"  Wikipedia:        {len(all_data['wikipedia']):,} articles")
        print(f"  Children's Books: {len(all_data['childrens_books']):,} stories")
        print(f"  Total:            {sum(len(v) for v in all_data.values()):,} documents")
        print("=" * 60 + "\n")

        return all_data

    def process_and_filter(self, raw_data: Dict[str, List[str]], preserve_sources: bool = False) -> Dict[str, List[str]]:
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
        print("\nProcessing and filtering data...")

        processed_by_source = {}

        # Process each source separately
        for source, texts in raw_data.items():
            print(f"\nProcessing {source}...")

            # Clean texts
            cleaned_texts = [clean_text(text) for text in tqdm(texts, desc="Cleaning")]

            # Apply quality filters
            print("  Filtering by length...")
            filtered_texts = self.quality_filter.filter_by_length(cleaned_texts)

            print("  Filtering by language...")
            filtered_texts = self.quality_filter.filter_by_language(filtered_texts)

            print(f"  {len(filtered_texts)} texts passed quality filters")
            processed_by_source[source] = filtered_texts

        # Calculate total before deduplication
        total_texts = sum(len(texts) for texts in processed_by_source.values())
        print(f"\nTotal texts before deduplication: {total_texts}")

        if preserve_sources:
            # For new pipeline: deduplicate within each source, preserve source information
            deduplicated_by_source = {}
            total_removed = 0

            for source, texts in processed_by_source.items():
                print(f"Deduplicating {source}...")
                deduplicated_texts, removed_indices = self.deduplicator.deduplicate_corpus(texts)
                deduplicated_by_source[source] = deduplicated_texts
                total_removed += len(removed_indices)
                print(f"  {source}: Removed {len(removed_indices)} duplicates, {len(deduplicated_texts)} remaining")

            print(f"\nTotal duplicates removed: {total_removed}")
            total_after = sum(len(texts) for texts in deduplicated_by_source.values())
            print(f"Total texts after deduplication: {total_after}")

            return deduplicated_by_source
        else:
            # For legacy pipeline: combine all sources, then deduplicate
            all_texts = []
            for texts in processed_by_source.values():
                all_texts.extend(texts)

            print("Deduplicating corpus...")
            deduplicated_texts, removed_indices = self.deduplicator.deduplicate_corpus(all_texts)
            print(f"Removed {len(removed_indices)} duplicates")
            print(f"Final corpus size: {len(deduplicated_texts)} texts")

            # Limit to max words if specified
            if self.max_words:
                deduplicated_texts = self._limit_to_max_words(deduplicated_texts)

            return {'combined': deduplicated_texts}

    def _limit_to_max_words(self, texts: List[str]) -> List[str]:
        """Limit corpus to maximum number of words"""
        print(f"\nLimiting corpus to {self.max_words:,} words...")

        limited_texts = []
        total_words = 0

        for text in texts:
            # Count words (whitespace tokenization)
            word_count = len(text.split())

            if total_words + word_count <= self.max_words:
                limited_texts.append(text)
                total_words += word_count
            else:
                break

        print(f"Limited to {len(limited_texts)} texts (~{total_words:,} words)")
        return limited_texts

    def create_balanced_splits_with_limits(self, raw_data: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Create balanced train/val/test splits with separate word limits

        Strategy:
        1. Apply global deduplication first (no text appears in multiple splits)
        2. Validation split: Take equal proportions from each source until val_word_limit reached
        3. Test split: Take equal proportions from remaining data until test_word_limit reached
        4. Training split: Use configured train_source_ratios until train_word_limit reached

        Args:
            raw_data: Dictionary with keys as source names and values as lists of texts

        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        print("\n" + "="*80)
        print("Creating balanced train/val/test splits with separate word limits")
        print("="*80)

        # Filter out sources with no data
        available_sources = {k: v for k, v in raw_data.items() if len(v) > 0}

        if len(available_sources) != len(raw_data):
            missing = set(raw_data.keys()) - set(available_sources.keys())
            print(f"\n⚠️  Sources with no data: {missing}")
            print(f"   Using {len(available_sources)} available sources: {list(available_sources.keys())}")

        # Shuffle each source with seed for reproducibility
        np.random.seed(42)
        shuffled_sources = {
            source: np.random.permutation(texts).tolist()
            for source, texts in available_sources.items()
        }

        # Track which texts we've used (for global deduplication)
        used_texts = set()
        splits = {'train': [], 'val': [], 'test': []}
        source_indices = {source: 0 for source in available_sources.keys()}

        def get_next_unique_text(source: str) -> Optional[str]:
            """Get next text from source that hasn't been used yet"""
            while source_indices[source] < len(shuffled_sources[source]):
                text = shuffled_sources[source][source_indices[source]]
                source_indices[source] += 1

                # Check if text is duplicate (global deduplication)
                text_hash = hash(text)
                if text_hash not in used_texts:
                    used_texts.add(text_hash)
                    return text
            return None

        # STEP 1: Create balanced validation split (equal from each source)
        print(f"\n📊 Creating validation split (target: {self.val_word_limit:,} words, balanced across sources)...")
        val_words_per_source = self.val_word_limit // len(available_sources)
        val_word_counts = {source: 0 for source in available_sources.keys()}
        total_val_words = 0

        for source in available_sources.keys():
            print(f"   Collecting from {source} (target: {val_words_per_source:,} words)...", end=" ")
            source_val_count = 0

            while val_word_counts[source] < val_words_per_source:
                text = get_next_unique_text(source)
                if text is None:
                    print(f"\n   ⚠️  {source} exhausted at {val_word_counts[source]:,} words")
                    break

                text_words = len(text.split())
                if val_word_counts[source] + text_words <= val_words_per_source:
                    splits['val'].append(text)
                    val_word_counts[source] += text_words
                    total_val_words += text_words
                    source_val_count += 1
                else:
                    # Would exceed limit, skip this text
                    break

            print(f"{source_val_count} texts, {val_word_counts[source]:,} words")

        print(f"   ✓ Validation split: {len(splits['val'])} texts, {total_val_words:,} words")

        # STEP 2: Create balanced test split (equal from each source)
        print(f"\n📊 Creating test split (target: {self.test_word_limit:,} words, balanced across sources)...")
        test_words_per_source = self.test_word_limit // len(available_sources)
        test_word_counts = {source: 0 for source in available_sources.keys()}
        total_test_words = 0

        for source in available_sources.keys():
            print(f"   Collecting from {source} (target: {test_words_per_source:,} words)...", end=" ")
            source_test_count = 0

            while test_word_counts[source] < test_words_per_source:
                text = get_next_unique_text(source)
                if text is None:
                    print(f"\n   ⚠️  {source} exhausted at {test_word_counts[source]:,} words")
                    break

                text_words = len(text.split())
                if test_word_counts[source] + text_words <= test_words_per_source:
                    splits['test'].append(text)
                    test_word_counts[source] += text_words
                    total_test_words += text_words
                    source_test_count += 1
                else:
                    # Would exceed limit, skip this text
                    break

            print(f"{source_test_count} texts, {test_word_counts[source]:,} words")

        print(f"   ✓ Test split: {len(splits['test'])} texts, {total_test_words:,} words")

        # STEP 3: Create training split (use configured ratios)
        print(f"\n📊 Creating training split (target: {self.train_word_limit:,} words, using source ratios)...")

        # Get train source ratios from config
        train_source_ratios = self.config.__dict__.get('train_source_ratios', {
            source: 1.0 / len(available_sources) for source in available_sources.keys()
        })

        # Handle missing children's books by redistributing ratio
        if 'childrens_books' not in available_sources and 'childrens_books' in train_source_ratios:
            print(f"   ⚠️  Redistributing childrens_books ratio ({train_source_ratios['childrens_books']:.2%}) to other sources")
            childrens_ratio = train_source_ratios.pop('childrens_books')
            total_remaining = sum(train_source_ratios.values())
            for source in train_source_ratios.keys():
                train_source_ratios[source] += (train_source_ratios[source] / total_remaining) * childrens_ratio

        # Calculate target words per source for training
        train_words_per_source = {
            source: int(self.train_word_limit * ratio)
            for source, ratio in train_source_ratios.items()
            if source in available_sources
        }

        print(f"   Training source ratios:")
        for source, ratio in train_source_ratios.items():
            if source in available_sources:
                print(f"      {source}: {ratio:.1%} ({train_words_per_source[source]:,} words)")

        train_word_counts = {source: 0 for source in available_sources.keys()}
        total_train_words = 0

        for source in available_sources.keys():
            if source not in train_words_per_source:
                continue

            target_words = train_words_per_source[source]
            print(f"   Collecting from {source} (target: {target_words:,} words)...", end=" ")
            source_train_count = 0

            while train_word_counts[source] < target_words:
                text = get_next_unique_text(source)
                if text is None:
                    print(f"\n   ⚠️  {source} exhausted at {train_word_counts[source]:,} words")
                    break

                text_words = len(text.split())
                if train_word_counts[source] + text_words <= target_words:
                    splits['train'].append(text)
                    train_word_counts[source] += text_words
                    total_train_words += text_words
                    source_train_count += 1
                else:
                    # Would exceed limit, skip this text
                    break

            print(f"{source_train_count} texts, {train_word_counts[source]:,} words")

        print(f"   ✓ Training split: {len(splits['train'])} texts, {total_train_words:,} words")

        # Final summary
        print("\n" + "="*80)
        print("SPLIT CREATION SUMMARY")
        print("="*80)
        print(f"  Train: {len(splits['train']):,} texts, {total_train_words:,} words")
        for source in available_sources.keys():
            if source in train_word_counts:
                print(f"    - {source}: {train_word_counts[source]:,} words ({train_word_counts[source]/total_train_words*100:.1f}%)")

        print(f"\n  Val:   {len(splits['val']):,} texts, {total_val_words:,} words")
        for source in available_sources.keys():
            print(f"    - {source}: {val_word_counts[source]:,} words ({val_word_counts[source]/total_val_words*100:.1f}%)")

        print(f"\n  Test:  {len(splits['test']):,} texts, {total_test_words:,} words")
        for source in available_sources.keys():
            print(f"    - {source}: {test_word_counts[source]:,} words ({test_word_counts[source]/total_test_words*100:.1f}%)")

        print(f"\n  Total: {len(splits['train']) + len(splits['val']) + len(splits['test']):,} texts, {total_train_words + total_val_words + total_test_words:,} words")
        print("="*80 + "\n")

        return splits

    def create_splits(self, processed_data: List[str]) -> Dict[str, List[str]]:
        """
        Create train/val/test splits (legacy method for backward compatibility)

        NOTE: This method is deprecated. Use create_balanced_splits_with_limits() for new pipeline.
        """
        print("\nCreating train/val/test splits...")

        # Shuffle data
        np.random.seed(42)
        indices = np.random.permutation(len(processed_data))
        shuffled_data = [processed_data[i] for i in indices]

        # Calculate split sizes
        n_total = len(shuffled_data)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        # Create splits
        splits = {
            'train': shuffled_data[:n_train],
            'val': shuffled_data[n_train:n_train + n_val],
            'test': shuffled_data[n_train + n_val:]
        }

        print(f"  Train: {len(splits['train'])} samples")
        print(f"  Val: {len(splits['val'])} samples")
        print(f"  Test: {len(splits['test'])} samples")

        return splits

    def save_splits(self, splits: Dict[str, List[str]]):
        """Save processed splits to disk"""
        print("\nSaving splits...")

        for split_name, texts in splits.items():
            # Save as pickle
            pkl_path = os.path.join(self.data_dir, 'splits', f'{split_name}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(texts, f)

            # Also save as text file for inspection
            txt_path = os.path.join(self.data_dir, 'splits', f'{split_name}.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                for text in texts[:100]:  # Save first 100 samples as text
                    f.write(text + '\n' + '='*80 + '\n')

            print(f"  Saved {split_name} split to {pkl_path}")

        # Save metadata
        metadata = {
            'num_train': len(splits['train']),
            'num_val': len(splits['val']),
            'num_test': len(splits['test']),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'max_words': self.max_words
        }

        metadata_path = os.path.join(self.data_dir, 'splits', 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved metadata to {metadata_path}")

    def load_splits(self) -> Dict[str, List[str]]:
        """Load processed splits from disk"""
        print("Loading processed splits...")

        splits = {}
        for split_name in ['train', 'val', 'test']:
            pkl_path = os.path.join(self.data_dir, 'splits', f'{split_name}.pkl')

            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"Split file not found: {pkl_path}")

            with open(pkl_path, 'rb') as f:
                splits[split_name] = pickle.load(f)

            print(f"  Loaded {split_name}: {len(splits[split_name])} samples")

        return splits

    def create_dataloader(self, texts: List[str], tokenizer, split: str = 'train') -> DataLoader:
        """Create PyTorch DataLoader for a split"""
        dataset = TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=self.config.max_length
        )

        batch_size = self.config.batch_size
        shuffle = (split == 'train')

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )

        return dataloader

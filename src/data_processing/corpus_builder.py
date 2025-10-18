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
        self.max_tokens = config.max_tokens
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
                max_samples = self.config.__dict__.get('max_tokens', 10_000_000) // 10  # Rough estimate

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

    def process_and_filter(self, raw_data: Dict[str, List[str]]) -> List[str]:
        """Process and filter collected data"""
        print("\nProcessing and filtering data...")

        all_texts = []

        # Combine all sources
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
            all_texts.extend(filtered_texts)

        print(f"\nTotal texts before deduplication: {len(all_texts)}")

        # Deduplicate
        print("Deduplicating corpus...")
        deduplicated_texts, removed_indices = self.deduplicator.deduplicate_corpus(all_texts)
        print(f"Removed {len(removed_indices)} duplicates")
        print(f"Final corpus size: {len(deduplicated_texts)} texts")

        # Limit to max tokens if specified
        if self.max_tokens:
            deduplicated_texts = self._limit_to_max_tokens(deduplicated_texts)

        return deduplicated_texts

    def _limit_to_max_tokens(self, texts: List[str]) -> List[str]:
        """Limit corpus to maximum number of tokens"""
        print(f"\nLimiting corpus to {self.max_tokens:,} tokens...")

        limited_texts = []
        total_tokens = 0

        for text in texts:
            # Approximate token count (words * 1.3 for subword tokenization)
            approx_tokens = len(text.split()) * 1.3

            if total_tokens + approx_tokens <= self.max_tokens:
                limited_texts.append(text)
                total_tokens += approx_tokens
            else:
                break

        print(f"Limited to {len(limited_texts)} texts (~{total_tokens:,.0f} tokens)")
        return limited_texts

    def create_splits(self, processed_data: List[str]) -> Dict[str, List[str]]:
        """Create train/val/test splits"""
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
            'max_tokens': self.max_tokens
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

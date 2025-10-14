"""
IndicCorp Hindi Dataset Downloader

This module provides functionality to download and process the Hindi portion
of the IndicCorp V2 dataset from AI4Bharat/HuggingFace.

IndicCorp is a large-scale sentence-level monolingual corpus for Indian languages,
containing high-quality text from diverse sources including news, blogs, and websites.

Dataset Information:
- Source: AI4Bharat (https://huggingface.co/datasets/ai4bharat/IndicCorpusV2)
- Language: Hindi (hi)
- Size: ~3.5GB (billions of tokens)
- License: CC0-1.0 (Public Domain)
- Content: Web-crawled text, news articles, blogs

Usage:
    from src.data_processing.indiccorp_downloader import download_indiccorp_hindi

    dataset = download_indiccorp_hindi(
        output_dir='data/raw',
        num_samples=100000,  # Limit samples for BabyLM
        streaming=False
    )
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

from datasets import load_dataset, Dataset, IterableDataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndicCorpDownloader:
    """
    Downloader for IndicCorp Hindi dataset.

    Handles downloading, caching, and basic preprocessing of the
    IndicCorp V2 Hindi dataset from HuggingFace.
    """

    def __init__(
        self,
        output_dir: str = 'data/raw',
        cache_dir: Optional[str] = None
    ):
        """
        Initialize IndicCorp downloader.

        Args:
            output_dir: Directory to save downloaded data
            cache_dir: Directory for HuggingFace datasets cache
                      (defaults to ~/.cache/huggingface/datasets)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = cache_dir

        # IndicCorp V2 dataset identifier
        self.dataset_name = "ai4bharat/IndicCorpusV2"
        self.language_code = "hi"  # Hindi

        logger.info(f"Initialized IndicCorp downloader")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Cache directory: {self.cache_dir or 'default'}")

    def download(
        self,
        num_samples: Optional[int] = None,
        streaming: bool = False,
        split: str = 'train'
    ) -> Union[Dataset, IterableDataset]:
        """
        Download IndicCorp Hindi dataset.

        Args:
            num_samples: Number of samples to download (None for all)
            streaming: Whether to use streaming mode (memory-efficient)
            split: Dataset split to download ('train', 'validation', 'test')

        Returns:
            Dataset or IterableDataset object
        """
        logger.info(f"Downloading IndicCorp Hindi dataset...")
        logger.info(f"  Streaming: {streaming}")
        logger.info(f"  Split: {split}")
        logger.info(f"  Samples: {num_samples or 'all'}")

        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(
                self.dataset_name,
                self.language_code,
                split=split,
                streaming=streaming,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )

            logger.info(f"✓ Successfully loaded IndicCorp Hindi dataset")

            # If limiting samples
            if num_samples is not None:
                logger.info(f"Limiting to {num_samples} samples...")
                if streaming:
                    # For streaming datasets, use take()
                    dataset = dataset.take(num_samples)
                else:
                    # For regular datasets, use select()
                    dataset = dataset.select(range(min(num_samples, len(dataset))))
                logger.info(f"✓ Limited to {num_samples} samples")

            return dataset

        except Exception as e:
            logger.error(f"Failed to download IndicCorp dataset: {e}")
            raise

    def save_to_text(
        self,
        dataset: Union[Dataset, IterableDataset],
        output_filename: str = 'indiccorp_hindi.txt',
        text_field: str = 'sentence'
    ) -> Path:
        """
        Save dataset to a text file (one sentence per line).

        Args:
            dataset: HuggingFace dataset
            output_filename: Output filename
            text_field: Field name containing text (default: 'sentence')

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / output_filename

        logger.info(f"Saving dataset to {output_path}...")

        # Determine if streaming
        is_streaming = isinstance(dataset, IterableDataset)

        with open(output_path, 'w', encoding='utf-8') as f:
            if is_streaming:
                # Streaming dataset - iterate without length
                count = 0
                for example in tqdm(dataset, desc="Saving"):
                    text = example.get(text_field, '')
                    if text and text.strip():  # Skip empty lines
                        f.write(text.strip() + '\n')
                        count += 1
                logger.info(f"✓ Saved {count} samples")
            else:
                # Regular dataset - can show progress
                for example in tqdm(dataset, desc="Saving"):
                    text = example.get(text_field, '')
                    if text and text.strip():  # Skip empty lines
                        f.write(text.strip() + '\n')
                logger.info(f"✓ Saved {len(dataset)} samples")

        return output_path

    def save_to_pickle(
        self,
        dataset: Union[Dataset, IterableDataset],
        output_filename: str = 'indiccorp_hindi.pkl',
        text_field: str = 'sentence'
    ) -> Path:
        """
        Save dataset to pickle file (list of strings).

        Args:
            dataset: HuggingFace dataset
            output_filename: Output filename
            text_field: Field name containing text

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / output_filename

        logger.info(f"Saving dataset to {output_path}...")

        # Extract text samples
        texts = []
        is_streaming = isinstance(dataset, IterableDataset)

        if is_streaming:
            for example in tqdm(dataset, desc="Extracting text"):
                text = example.get(text_field, '')
                if text and text.strip():
                    texts.append(text.strip())
        else:
            for example in tqdm(dataset, desc="Extracting text"):
                text = example.get(text_field, '')
                if text and text.strip():
                    texts.append(text.strip())

        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(texts, f)

        logger.info(f"✓ Saved {len(texts)} samples to pickle")

        return output_path

    def get_statistics(
        self,
        dataset: Union[Dataset, IterableDataset],
        num_samples: int = 10000,
        text_field: str = 'sentence'
    ) -> Dict:
        """
        Calculate statistics from dataset.

        Args:
            dataset: HuggingFace dataset
            num_samples: Number of samples to analyze (for efficiency)
            text_field: Field name containing text

        Returns:
            Dictionary with dataset statistics
        """
        logger.info(f"Calculating dataset statistics (sampling {num_samples} texts)...")

        stats = {
            'dataset_name': self.dataset_name,
            'language': self.language_code,
            'timestamp': datetime.now().isoformat(),
            'samples_analyzed': 0,
            'total_characters': 0,
            'total_words': 0,
            'avg_chars_per_sample': 0.0,
            'avg_words_per_sample': 0.0,
            'min_length': float('inf'),
            'max_length': 0,
            'empty_samples': 0
        }

        is_streaming = isinstance(dataset, IterableDataset)

        # Collect samples for analysis
        sample_texts = []

        if is_streaming:
            # Streaming - take first num_samples
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                text = example.get(text_field, '')
                sample_texts.append(text)
        else:
            # Regular dataset - sample uniformly
            total_size = len(dataset)
            sample_size = min(num_samples, total_size)
            indices = range(0, total_size, max(1, total_size // sample_size))[:sample_size]

            for idx in tqdm(indices, desc="Sampling for statistics"):
                text = dataset[int(idx)].get(text_field, '')
                sample_texts.append(text)

        # Analyze samples
        for text in tqdm(sample_texts, desc="Analyzing"):
            if not text or not text.strip():
                stats['empty_samples'] += 1
                continue

            char_count = len(text)
            word_count = len(text.split())

            stats['total_characters'] += char_count
            stats['total_words'] += word_count
            stats['samples_analyzed'] += 1
            stats['min_length'] = min(stats['min_length'], char_count)
            stats['max_length'] = max(stats['max_length'], char_count)

        # Calculate averages
        if stats['samples_analyzed'] > 0:
            stats['avg_chars_per_sample'] = stats['total_characters'] / stats['samples_analyzed']
            stats['avg_words_per_sample'] = stats['total_words'] / stats['samples_analyzed']

        # Fix infinity for min_length if no valid samples
        if stats['min_length'] == float('inf'):
            stats['min_length'] = 0

        logger.info(f"✓ Statistics calculated")
        logger.info(f"  Samples analyzed: {stats['samples_analyzed']}")
        logger.info(f"  Avg words/sample: {stats['avg_words_per_sample']:.1f}")
        logger.info(f"  Avg chars/sample: {stats['avg_chars_per_sample']:.1f}")

        return stats

    def save_statistics(
        self,
        stats: Dict,
        output_filename: str = 'indiccorp_statistics.json'
    ) -> Path:
        """
        Save statistics to JSON file.

        Args:
            stats: Statistics dictionary
            output_filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / output_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Statistics saved to {output_path}")

        return output_path

    def save_metadata(
        self,
        dataset: Union[Dataset, IterableDataset],
        stats: Dict,
        output_filename: str = 'indiccorp_metadata.json'
    ) -> Path:
        """
        Save comprehensive metadata about the downloaded dataset.

        Args:
            dataset: HuggingFace dataset
            stats: Statistics dictionary
            output_filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / output_filename

        is_streaming = isinstance(dataset, IterableDataset)

        metadata = {
            'source': self.dataset_name,
            'language': self.language_code,
            'download_timestamp': datetime.now().isoformat(),
            'streaming_mode': is_streaming,
            'statistics': stats,
            'output_directory': str(self.output_dir),
            'dataset_features': list(dataset.features.keys()) if not is_streaming else ['sentence']
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Metadata saved to {output_path}")

        return output_path


# Convenience functions

def download_indiccorp_hindi(
    output_dir: str = 'data/raw',
    num_samples: Optional[int] = None,
    streaming: bool = False,
    save_format: str = 'both'
) -> Dict:
    """
    Download IndicCorp Hindi dataset (convenience function).

    Args:
        output_dir: Directory to save data
        num_samples: Number of samples to download (None for all)
        streaming: Use streaming mode (memory-efficient)
        save_format: 'text', 'pickle', or 'both'

    Returns:
        Dictionary with paths to saved files and statistics
    """
    downloader = IndicCorpDownloader(output_dir=output_dir)

    # Download dataset
    dataset = downloader.download(
        num_samples=num_samples,
        streaming=streaming
    )

    # Calculate statistics
    stats = downloader.get_statistics(dataset, num_samples=min(10000, num_samples or 10000))

    # Save statistics
    stats_path = downloader.save_statistics(stats)

    # Save dataset
    paths = {'statistics': stats_path}

    if save_format in ['text', 'both']:
        text_path = downloader.save_to_text(dataset)
        paths['text'] = text_path

    if save_format in ['pickle', 'both']:
        pickle_path = downloader.save_to_pickle(dataset)
        paths['pickle'] = pickle_path

    # Save metadata
    metadata_path = downloader.save_metadata(dataset, stats)
    paths['metadata'] = metadata_path

    logger.info("=" * 60)
    logger.info("IndicCorp Hindi download complete!")
    logger.info(f"  Samples: {stats['samples_analyzed']}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    return paths


def load_indiccorp_from_cache(
    cache_path: str,
    file_format: str = 'pickle'
) -> List[str]:
    """
    Load previously downloaded IndicCorp data from cache.

    Args:
        cache_path: Path to cached file
        file_format: 'text' or 'pickle'

    Returns:
        List of text samples
    """
    logger.info(f"Loading IndicCorp from cache: {cache_path}")

    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    if file_format == 'pickle':
        with open(cache_path, 'rb') as f:
            texts = pickle.load(f)
    elif file_format == 'text':
        with open(cache_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unknown format: {file_format}")

    logger.info(f"✓ Loaded {len(texts)} samples from cache")

    return texts


# Alias for backwards compatibility
def save_raw_data(dataset, output_path):
    """
    Save raw data to disk (legacy function).

    Args:
        dataset: HuggingFace dataset
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_dir = output_path.parent

    downloader = IndicCorpDownloader(output_dir=str(output_dir))

    if output_path.suffix == '.txt':
        return downloader.save_to_text(dataset, output_path.name)
    elif output_path.suffix == '.pkl':
        return downloader.save_to_pickle(dataset, output_path.name)
    else:
        raise ValueError(f"Unsupported format: {output_path.suffix}")


def get_dataset_statistics(dataset):
    """
    Calculate basic statistics (legacy function).

    Args:
        dataset: HuggingFace dataset

    Returns:
        Statistics dictionary
    """
    downloader = IndicCorpDownloader()
    return downloader.get_statistics(dataset)


if __name__ == '__main__':
    """
    Example usage:

    python src/data_processing/indiccorp_downloader.py
    """
    import argparse

    parser = argparse.ArgumentParser(description='Download IndicCorp Hindi dataset')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to download (default: all)')
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming mode')
    parser.add_argument('--format', type=str, default='both',
                       choices=['text', 'pickle', 'both'],
                       help='Save format')

    args = parser.parse_args()

    # Download dataset
    paths = download_indiccorp_hindi(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        streaming=args.streaming,
        save_format=args.format
    )

    print("\n" + "=" * 60)
    print("Download complete! Files saved:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    print("=" * 60)

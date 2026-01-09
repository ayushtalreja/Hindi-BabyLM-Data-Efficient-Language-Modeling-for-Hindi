"""
Wikipedia downloader using HuggingFace datasets.

Downloads Hindi Wikipedia articles from the wikimedia/wikipedia dataset
on HuggingFace Hub. This replaces the old scraping approach with a much
faster and more reliable method.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm

from .base_downloader import BaseDownloader
from ..text_cleaner import clean_text
from ..utils import configure_logger

logger = configure_logger(__name__)


class WikiDownloader(BaseDownloader):
    """
    Download Hindi Wikipedia from HuggingFace datasets.

    Uses the wikimedia/wikipedia dataset which provides pre-processed,
    cleaned Wikipedia dumps in various languages.
    """

    def __init__(
        self,
        output_dir: str = 'data/raw',
        dataset_version: str = "20231101.hi",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Wikipedia downloader.

        Args:
            output_dir: Directory for saving output files
            dataset_version: Version of Wikipedia dataset (format: YYYYMMDD.lang)
            cache_dir: Optional directory for HuggingFace dataset cache
        """
        super().__init__(output_dir)
        self.dataset_version = dataset_version
        self.cache_dir = cache_dir
        self.source_name = "wikipedia"

        logger.info(f"Initialized WikiDownloader (version: {dataset_version})")

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
        logger.info(f"\nLoading Wikipedia dataset: {self.dataset_version}")
        logger.info(f"  Filters: min_length={min_length}, max_length={max_length}")

        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(
                "wikimedia/wikipedia",
                self.dataset_version,
                cache_dir=self.cache_dir,
                trust_remote_code=False
            )

            # Extract texts from the train split
            train_data = dataset["train"]
            total_available = len(train_data)

            logger.info(f"✓ Dataset loaded: {total_available:,} articles available")

            # Process articles
            texts = []
            total_words = 0  # Track word count incrementally
            articles_processed = 0
            articles_filtered = 0

            # Determine how many to process
            num_to_process = min(max_articles, total_available) if max_articles else total_available

            logger.info(f"Processing {num_to_process:,} articles...")

            for i in tqdm(range(num_to_process), desc="Processing articles"):
                article = train_data[i]

                # Extract and clean text
                raw_text = article.get("text", "")
                cleaned_text = clean_text(raw_text)

                articles_processed += 1

                # Apply length filters
                text_length = len(cleaned_text)
                if min_length <= text_length <= max_length:
                    # Count words incrementally (one article at a time to avoid memory spike)
                    total_words += len(cleaned_text.split())
                    texts.append(cleaned_text)
                else:
                    articles_filtered += 1

                # Progress update every 1000 articles
                if articles_processed % 5000 == 0:
                    logger.info(f"  Processed: {articles_processed:,} | "
                               f"Kept: {len(texts):,} | "
                               f"Filtered: {articles_filtered:,}")

            logger.info(f"\n✓ Wikipedia download complete!")
            logger.info(f"  Total processed: {articles_processed:,}")
            logger.info(f"  Articles kept: {len(texts):,}")
            logger.info(f"  Articles filtered: {articles_filtered:,}")
            logger.info(f"  Total words: {total_words:,}")

            return texts, total_words

        except Exception as e:
            logger.error(f"✗ Failed to download Wikipedia: {e}")
            raise

    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about the Wikipedia data source.

        Returns:
            Dict with source metadata
        """
        return {
            "source": "HuggingFace wikimedia/wikipedia",
            "version": self.dataset_version,
            "language": "Hindi (hi)",
            "format": "Parquet via datasets library",
            "description": "Pre-processed Wikipedia dump from Wikimedia Foundation"
        }


def download_wikipedia_hindi(
    output_dir: str = 'data/raw',
    dataset_version: str = "20231101.hi",
    max_articles: Optional[int] = None,
    min_length: int = 50,
    max_length: int = 50000,
    save_stats: bool = True
) -> Path:
    """
    Convenience function to download Hindi Wikipedia.

    Args:
        output_dir: Directory for saving output files
        dataset_version: Version of Wikipedia dataset
        max_articles: Maximum number of articles (None = all)
        min_length: Minimum character length
        max_length: Maximum character length
        save_stats: Whether to calculate and save statistics

    Returns:
        Path to the cached pickle file
    """
    downloader = WikiDownloader(
        output_dir=output_dir,
        dataset_version=dataset_version
    )

    return downloader.process_and_cache(
        max_samples=None,  # We filter during download
        save_stats=save_stats,
        max_articles=max_articles,
        min_length=min_length,
        max_length=max_length
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("Downloading Hindi Wikipedia...")
    cache_path = download_wikipedia_hindi(
        max_articles=1000,  # Test with 1000 articles
        save_stats=True
    )
    print(f"\n✓ Cached at: {cache_path}")

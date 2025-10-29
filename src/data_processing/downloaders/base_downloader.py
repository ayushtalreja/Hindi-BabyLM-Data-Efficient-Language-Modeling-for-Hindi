"""
Abstract base class for data downloaders.

Provides a common interface and shared functionality for all data source downloaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import shared utilities using relative imports
from ..utils import (
    save_pickle,
    save_text_file,
    calculate_corpus_stats,
    generate_metadata,
    save_metadata,
    sample_texts,
    configure_logger
)

logger = configure_logger(__name__)


class BaseDownloader(ABC):
    """
    Abstract base class for all data downloaders.

    All downloader classes should inherit from this and implement:
    - download(): Load/download data and return list of texts
    - get_source_info(): Return information about the data source
    """

    def __init__(self, output_dir: str = 'data/raw'):
        """
        Initialize the downloader.

        Args:
            output_dir: Directory for saving output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Derive source name from class name
        self.source_name = self.__class__.__name__.replace('Downloader', '').replace('Loader', '').lower()

        logger.info(f"Initialized {self.__class__.__name__} (output_dir: {output_dir})")

    @abstractmethod
    def download(self, **kwargs) -> List[str]:
        """
        Download or load data from the source.

        Returns:
            List of text strings
        """
        pass

    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about this data source.

        Returns:
            Dict with source metadata (format, location, version, etc.)
        """
        pass

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
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {self.source_name}")
        logger.info(f"{'='*60}")

        # Step 1: Download/load data
        logger.info("\n[1/4] Downloading data...")
        texts = self.download(**download_kwargs)

        if not texts:
            logger.error(f"No texts downloaded from {self.source_name}")
            return None

        logger.info(f"✓ Downloaded {len(texts):,} texts")

        # Step 2: Sample if needed
        if max_samples:
            logger.info(f"\n[2/4] Sampling {max_samples:,} texts...")
            texts = sample_texts(texts, max_samples=max_samples)
        else:
            logger.info(f"\n[2/4] Skipping sampling (using all {len(texts):,} texts)")

        # Step 3: Calculate statistics
        stats = {}
        if save_stats:
            logger.info("\n[3/4] Calculating statistics...")
            stats = calculate_corpus_stats(texts, self.source_name)

        # Step 4: Save to cache
        logger.info("\n[4/4] Saving to cache...")

        # Save pickle file
        pickle_path = self.output_dir / f"{self.source_name}.pkl"
        save_pickle(texts, pickle_path)

        # Save metadata
        if save_stats:
            metadata = generate_metadata(
                texts,
                self.source_name,
                stats,
                additional_info=self.get_source_info()
            )
            metadata_path = self.output_dir / f"{self.source_name}_metadata.json"
            save_metadata(metadata, metadata_path)

        # Save sample text file
        if save_sample_text:
            sample_path = self.output_dir / f"{self.source_name}_sample.txt"
            save_text_file(texts, sample_path, sample_size=sample_text_size)

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ {self.source_name} processing complete!")
        logger.info(f"  Cached: {pickle_path}")
        logger.info(f"  Samples: {len(texts):,}")
        logger.info(f"{'='*60}\n")

        return pickle_path

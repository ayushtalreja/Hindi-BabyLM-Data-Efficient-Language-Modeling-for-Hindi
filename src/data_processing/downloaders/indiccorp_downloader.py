"""
IndicCorp Hindi Dataset Downloader (Refactored)

This module provides functionality to download and process the Hindi portion
of the IndicCorp V2 dataset from AI4Bharat/HuggingFace.

IndicCorp is a large-scale sentence-level monolingual corpus for Indian languages,
containing high-quality text from diverse sources including news, blogs, and websites.

Dataset Information:
- Source: AI4Bharat (https://huggingface.co/datasets/ai4bharat/IndicCorpusV2)
- Language: Hindi (hi)
- Files: hi-1.txt (26.7 GB), hi-2.txt (26.7 GB), hi-3.txt (26.7 GB)
- Total Size: ~80.1 GB for all three files
- License: CC0-1.0 (Public Domain)
- Content: Web-crawled text, news articles, blogs

Usage:
    from src.data_processing.downloaders import IndicCorpDownloader

    downloader = IndicCorpDownloader()
    texts = downloader.download(files=['hi-1.txt'], max_lines=100000)
"""

import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from .base_downloader import BaseDownloader
from ..text_cleaner import clean_text
from ..utils import configure_logger

logger = configure_logger(__name__)


class IndicCorpDownloader(BaseDownloader):
    """
    Downloader for IndicCorp Hindi dataset files.

    Downloads specific Hindi text files (hi-1.txt, hi-2.txt, hi-3.txt) from
    IndicCorp V2 dataset on HuggingFace Hub.
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
            cache_dir: Directory for HuggingFace Hub cache
                      (defaults to ~/.cache/huggingface/hub)
        """
        super().__init__(output_dir)
        self.cache_dir = cache_dir
        self.source_name = "indiccorp"

        # IndicCorp V2 dataset identifier
        self.repo_id = "ai4bharat/IndicCorpV2"
        self.repo_type = "dataset"

        # Available Hindi files
        self.available_hindi_files = ["hi-1.txt", "hi-2.txt", "hi-3.txt"]

        logger.info(f"Initialized IndicCorpDownloader")
        logger.info(f"  Repository: {self.repo_id}")
        logger.info(f"  Available files: {self.available_hindi_files}")
        logger.info(f"  Cache dir: {self.cache_dir or 'default'}")

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
        # Default to only hi-1.txt
        if files is None:
            files = ["hi-1.txt"]

        # Validate file names
        for filename in files:
            if filename not in self.available_hindi_files:
                logger.warning(f"File {filename} not in available files: {self.available_hindi_files}")

        logger.info(f"\nDownloading IndicCorp files...")
        logger.info(f"  Files: {files}")
        logger.info(f"  Max lines: {max_lines or 'all'}")

        all_texts = []
        total_words = 0  # Track word count incrementally

        for filename in files:
            logger.info(f"\n[Downloading {filename}]")

            try:
                # Download file from HuggingFace Hub
                file_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=f"data/{filename}",
                    repo_type=self.repo_type,
                    cache_dir=self.cache_dir,
                    resume_download=True
                )

                logger.info(f"✓ Downloaded to cache: {file_path}")

                # Copy to output directory
                output_path = self.output_dir / filename
                if not output_path.exists() or output_path.stat().st_size == 0:
                    logger.info(f"  Copying to {output_path}...")
                    shutil.copy2(file_path, output_path)
                    logger.info(f"  ✓ Copied to output directory")
                else:
                    logger.info(f"  File already exists in output: {output_path}")

                # Read texts from file (now returns texts and word count)
                logger.info(f"  Reading texts from {filename}...")
                texts, word_count = self._read_file(output_path, max_lines, clean_texts)

                logger.info(f"  ✓ Read {len(texts):,} texts ({word_count:,} words) from {filename}")
                all_texts.extend(texts)
                total_words += word_count

            except Exception as e:
                logger.error(f"✗ Failed to download {filename}: {e}")
                raise

        logger.info(f"\n✓ IndicCorp download complete!")
        logger.info(f"  Total texts: {len(all_texts):,}")
        logger.info(f"  Total words: {total_words:,}")

        return all_texts, total_words

    def _read_file(
        self,
        file_path: Path,
        max_lines: Optional[int] = None,
        clean_texts: bool = True
    ) -> tuple[List[str], int]:
        """
        Read texts from a file and count words incrementally.

        Args:
            file_path: Path to the text file
            max_lines: Maximum number of lines to read
            clean_texts: Whether to apply text cleaning

        Returns:
            Tuple of (list of text strings, total word count)
        """
        texts = []
        line_count = 0
        total_words = 0  # Track word count incrementally as we read

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Reading {file_path.name}"):
                # Check line limit
                if max_lines and line_count >= max_lines:
                    break

                # Skip empty lines
                text = line.strip()
                if not text:
                    continue

                # Clean text if requested
                if clean_texts:
                    text = clean_text(text)

                # Skip if cleaning removed everything
                if not text:
                    continue

                # Count words incrementally (one text at a time to avoid memory spike)
                total_words += len(text.split())

                texts.append(text)
                line_count += 1

        return texts, total_words

    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about the IndicCorp data source.

        Returns:
            Dict with source metadata
        """
        return {
            "source": "IndicCorp V2",
            "repository": self.repo_id,
            "language": "Hindi",
            "license": "CC0-1.0 (Public Domain)",
            "description": "Large-scale web-crawled Hindi corpus",
            "content_types": "News articles, blogs, web pages"
        }


def download_indiccorp_hindi(
    output_dir: str = 'data/raw',
    files: Optional[List[str]] = None,
    max_lines: Optional[int] = None,
    clean_texts: bool = True,
    save_stats: bool = True,
    cache_dir: Optional[str] = None
) -> Path:
    """
    Convenience function to download IndicCorp Hindi files.

    Args:
        output_dir: Directory to save data
        files: List of files to download (default: ['hi-1.txt'])
        max_lines: Maximum number of lines to read (None = all)
        clean_texts: Whether to apply text cleaning
        save_stats: Whether to calculate and save statistics
        cache_dir: HuggingFace Hub cache directory

    Returns:
        Path to the cached pickle file

    Example:
        # Download single file with sampling
        cache_path = download_indiccorp_hindi(
            max_lines=100000,
            save_stats=True
        )
    """
    downloader = IndicCorpDownloader(
        output_dir=output_dir,
        cache_dir=cache_dir
    )

    return downloader.process_and_cache(
        max_samples=None,  # We limit during download
        save_stats=save_stats,
        files=files,
        max_lines=max_lines,
        clean_texts=clean_texts
    )


if __name__ == '__main__':
    """
    Example usage:

    python src/data_processing/downloaders/indiccorp_downloader.py --max-lines 10000
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Download IndicCorp Hindi text files from HuggingFace'
    )
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory (default: data/raw)')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                       help='Hindi files to download (default: hi-1.txt)')
    parser.add_argument('--max-lines', type=int, default=None,
                       help='Maximum lines to read (default: all)')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='HuggingFace cache directory')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("IndicCorp Hindi Downloader")
    print("="*60)

    cache_path = download_indiccorp_hindi(
        output_dir=args.output_dir,
        files=args.files,
        max_lines=args.max_lines,
        cache_dir=args.cache_dir
    )

    print(f"\n✓ Cached at: {cache_path}")
    print("="*60)

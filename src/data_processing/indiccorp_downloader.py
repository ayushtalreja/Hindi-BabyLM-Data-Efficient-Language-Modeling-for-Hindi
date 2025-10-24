"""
IndicCorp Hindi Dataset Downloader

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
    from src.data_processing.indiccorp_downloader import download_indiccorp_hindi

    # Download single Hindi file (hi-1.txt) by default
    paths = download_indiccorp_hindi(
        output_dir='data/raw',
        num_samples=100000  # Limit samples for BabyLM
    )

    # Or specify which files to download (for multiple files)
    paths = download_indiccorp_hindi(
        output_dir='data/raw',
        files=['hi-1.txt', 'hi-2.txt', 'hi-3.txt'],  # Download all three files
        num_samples=100000
    )
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndicCorpDownloader:
    """
    Downloader for IndicCorp Hindi dataset files.

    Downloads specific Hindi text files (hi-1.txt, hi-2.txt) from
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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = cache_dir

        # IndicCorp V2 dataset identifier
        self.repo_id = "ai4bharat/IndicCorpV2"
        self.repo_type = "dataset"

        # Available Hindi files
        self.available_hindi_files = ["hi-1.txt", "hi-2.txt", "hi-3.txt"]

        logger.info(f"Initialized IndicCorp downloader")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Cache directory: {self.cache_dir or 'default (~/.cache/huggingface/hub)'}")
        logger.info(f"Available Hindi files: {self.available_hindi_files}")

    def download(
        self,
        files: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Download specific Hindi files from IndicCorp V2 repository.

        Args:
            files: List of filenames to download (e.g., ['hi-1.txt', 'hi-2.txt', 'hi-3.txt'])
                  If None, downloads only hi-1.txt by default

        Returns:
            Dictionary mapping filename to local file path
        """
        # Default to only hi-1.txt (single file)
        if files is None:
            files = ["hi-1.txt"]

        # Validate file names
        for filename in files:
            if filename not in self.available_hindi_files:
                logger.warning(f"File {filename} not in available Hindi files: {self.available_hindi_files}")

        logger.info(f"Downloading Hindi files from IndicCorp V2...")
        logger.info(f"  Files to download: {files}")
        logger.info(f"  Repository: {self.repo_id}")

        downloaded_files = {}

        for filename in files:
            try:
                logger.info(f"\nDownloading {filename}...")

                # Download file from HuggingFace Hub
                file_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=f"data/{filename}",
                    repo_type=self.repo_type,
                    cache_dir=self.cache_dir,
                    resume_download=True  # Resume if interrupted
                )

                logger.info(f"✓ Downloaded {filename} to cache: {file_path}")

                # Copy to output directory for easier access
                output_path = self.output_dir / filename
                if not output_path.exists() or output_path.stat().st_size == 0:
                    logger.info(f"Copying {filename} to {output_path}...")
                    shutil.copy2(file_path, output_path)
                    logger.info(f"✓ Copied to {output_path}")
                else:
                    logger.info(f"File already exists in output directory: {output_path}")

                downloaded_files[filename] = output_path

            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                raise

        logger.info(f"\n✓ Successfully downloaded {len(downloaded_files)} files")
        return downloaded_files

    def read_and_sample(
        self,
        file_path: Path,
        num_samples: Optional[int] = None,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Read a downloaded file and optionally sample lines from it.

        Args:
            file_path: Path to the downloaded file
            num_samples: Number of lines to sample (None for all)
            output_filename: Output filename for sampled data (if None, uses input filename)

        Returns:
            Path to output file (sampled or original)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # If no sampling needed, return original file
        if num_samples is None:
            logger.info(f"No sampling requested, using full file: {file_path}")
            return file_path

        # Create output filename
        if output_filename is None:
            output_filename = f"{file_path.stem}_sampled_{num_samples}.txt"

        output_path = self.output_dir / output_filename

        logger.info(f"Sampling {num_samples} lines from {file_path.name}...")

        count = 0
        with open(file_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for line in tqdm(f_in, desc=f"Sampling {file_path.name}"):
                    if count >= num_samples:
                        break
                    if line.strip():  # Skip empty lines
                        f_out.write(line)
                        count += 1

        logger.info(f"✓ Saved {count} samples to {output_path}")
        return output_path

    def convert_to_pickle(
        self,
        file_path: Path,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Convert text file to pickle format (list of strings).

        Args:
            file_path: Path to text file
            output_filename: Output pickle filename (if None, uses input filename)

        Returns:
            Path to saved pickle file
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Create output filename
        if output_filename is None:
            output_filename = f"{file_path.stem}.pkl"

        output_path = self.output_dir / output_filename

        logger.info(f"Converting {file_path.name} to pickle format...")

        # Read all lines
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading lines"):
                if line.strip():
                    texts.append(line.strip())

        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(texts, f)

        logger.info(f"✓ Saved {len(texts)} lines to {output_path}")

        return output_path

    def get_statistics(
        self,
        file_path: Path,
        num_samples: int = 100000
    ) -> Dict:
        """
        Calculate statistics from a text file.

        Args:
            file_path: Path to text file
            num_samples: Number of lines to analyze (for large files)

        Returns:
            Dictionary with file statistics
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Calculating statistics for {file_path.name} (sampling {num_samples} lines)...")

        stats = {
            'filename': file_path.name,
            'repo_id': self.repo_id,
            'language': 'Hindi',
            'timestamp': datetime.now().isoformat(),
            'lines_analyzed': 0,
            'total_characters': 0,
            'total_words': 0,
            'avg_chars_per_line': 0.0,
            'avg_words_per_line': 0.0,
            'min_character_length': float('inf'),
            'max_character_length': 0,
            'empty_lines': 0
        }

        # Read and analyze samples
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Analyzing", total=num_samples):
                if count >= num_samples:
                    break

                if not line.strip():
                    stats['empty_lines'] += 1
                    count += 1
                    continue

                char_count = len(line.strip())
                word_count = len(line.strip().split())

                stats['total_characters'] += char_count
                stats['total_words'] += word_count
                stats['lines_analyzed'] += 1
                stats['min_length'] = min(stats['min_character_length'], char_count)
                stats['max_length'] = max(stats['max_character_length'], char_count)

                count += 1

        # Calculate averages
        if stats['lines_analyzed'] > 0:
            stats['avg_chars_per_line'] = stats['total_characters'] / stats['lines_analyzed']
            stats['avg_words_per_line'] = stats['total_words'] / stats['lines_analyzed']

        # Fix infinity for min_length if no valid samples
        if stats['min_character_length'] == float('inf'):
            stats['min_character_length'] = 0

        logger.info(f"✓ Statistics calculated")
        logger.info(f"  Lines analyzed: {stats['lines_analyzed']}")
        logger.info(f"  Avg words/line: {stats['avg_words_per_line']:.1f}")
        logger.info(f"  Avg chars/line: {stats['avg_chars_per_line']:.1f}")

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
        downloaded_files: Dict[str, Path],
        stats_per_file: Dict[str, Dict],
        output_filename: str = 'indiccorp_metadata.json'
    ) -> Path:
        """
        Save comprehensive metadata about the downloaded files.

        Args:
            downloaded_files: Dictionary mapping filename to path
            stats_per_file: Dictionary mapping filename to statistics
            output_filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / output_filename

        metadata = {
            'source': self.repo_id,
            'language': 'Hindi',
            'download_timestamp': datetime.now().isoformat(),
            'downloaded_files': {name: str(path) for name, path in downloaded_files.items()},
            'statistics_per_file': stats_per_file,
            'output_directory': str(self.output_dir),
            'cache_directory': str(self.cache_dir) if self.cache_dir else 'default'
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Metadata saved to {output_path}")

        return output_path


# Convenience functions

def download_indiccorp_hindi(
    output_dir: str = 'data/raw',
    files: Optional[List[str]] = None,
    num_samples: Optional[int] = None,
    save_format: str = 'text',
    cache_dir: Optional[str] = None
) -> Dict:
    """
    Download IndicCorp Hindi files (convenience function).

    Args:
        output_dir: Directory to save data
        files: List of files to download (default: ['hi-1.txt'] - single file only)
        num_samples: Number of lines to sample from each file (None for all)
        save_format: 'text', 'pickle', or 'both'
        cache_dir: HuggingFace Hub cache directory

    Returns:
        Dictionary with paths to saved files and statistics

    Example:
        # Download single Hindi file (hi-1.txt) with sampling
        paths = download_indiccorp_hindi(
            output_dir='data/raw',
            num_samples=100000
        )

        # Download all three files if needed
        paths = download_indiccorp_hindi(
            output_dir='data/raw',
            files=['hi-1.txt', 'hi-2.txt', 'hi-3.txt'],
            num_samples=100000
        )
    """
    downloader = IndicCorpDownloader(output_dir=output_dir, cache_dir=cache_dir)

    # Download files from HuggingFace Hub
    logger.info("=" * 60)
    logger.info("Starting IndicCorp Hindi download...")
    logger.info("=" * 60)

    downloaded_files = downloader.download(files=files)

    # Process each file
    paths = {}
    stats_per_file = {}

    for filename, file_path in downloaded_files.items():
        logger.info(f"\nProcessing {filename}...")

        # Sample lines if requested
        if num_samples is not None:
            processed_path = downloader.read_and_sample(
                file_path,
                num_samples=num_samples,
                output_filename=f"{Path(filename).stem}_sampled.txt"
            )
        else:
            processed_path = file_path

        paths[filename] = processed_path

        # Calculate statistics
        stats = downloader.get_statistics(processed_path, num_samples=min(10000, num_samples or 10000))
        stats_per_file[filename] = stats

        # Save statistics for this file
        stats_filename = f"{Path(filename).stem}_statistics.json"
        downloader.save_statistics(stats, stats_filename)

        # Convert to pickle if requested
        if save_format in ['pickle', 'both']:
            pickle_path = downloader.convert_to_pickle(processed_path)
            paths[f"{filename}_pickle"] = pickle_path

    # Save overall metadata
    metadata_path = downloader.save_metadata(downloaded_files, stats_per_file)
    paths['metadata'] = metadata_path

    logger.info("\n" + "=" * 60)
    logger.info("IndicCorp Hindi download complete!")
    logger.info(f"  Files downloaded: {len(downloaded_files)}")
    logger.info(f"  Output directory: {output_dir}")
    for filename, stats in stats_per_file.items():
        logger.info(f"  {filename}: {stats['lines_analyzed']} lines")
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


# Utility function to merge multiple Hindi files
def merge_hindi_files(
    input_files: List[Path],
    output_file: Path,
    max_lines: Optional[int] = None
) -> Path:
    """
    Merge multiple Hindi text files into one.

    Args:
        input_files: List of input file paths
        output_file: Output file path
        max_lines: Maximum number of lines to include (None for all)

    Returns:
        Path to merged file
    """
    logger.info(f"Merging {len(input_files)} files into {output_file}...")

    line_count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for input_file in input_files:
            logger.info(f"  Reading {input_file.name}...")
            with open(input_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if max_lines and line_count >= max_lines:
                        break
                    if line.strip():
                        f_out.write(line)
                        line_count += 1

            if max_lines and line_count >= max_lines:
                break

    logger.info(f"✓ Merged {line_count} lines to {output_file}")
    return output_file


if __name__ == '__main__':
    """
    Example usage:

    # Download single Hindi file (hi-1.txt) - default behavior
    python src/data_processing/indiccorp_downloader.py --output-dir data/raw

    # Download with sampling
    python src/data_processing/indiccorp_downloader.py --num-samples 100000

    # Download all three files
    python src/data_processing/indiccorp_downloader.py --files hi-1.txt hi-2.txt hi-3.txt

    # Download only hi-1.txt and hi-2.txt
    python src/data_processing/indiccorp_downloader.py --files hi-1.txt hi-2.txt
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Download IndicCorp Hindi text files from HuggingFace',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory (default: data/raw)')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                       help='Hindi files to download (default: hi-1.txt only)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of lines to sample from each file (default: all)')
    parser.add_argument('--format', type=str, default='text',
                       choices=['text', 'pickle', 'both'],
                       help='Save format (default: text)')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='HuggingFace Hub cache directory (default: ~/.cache/huggingface/hub)')

    args = parser.parse_args()

    # Download files
    print("\n" + "=" * 60)
    print("IndicCorp Hindi Downloader")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Files to download: {args.files or ['hi-1.txt']}")
    print(f"Sampling: {args.num_samples if args.num_samples else 'No (downloading all)'}")
    print(f"Format: {args.format}")
    print("=" * 60 + "\n")

    paths = download_indiccorp_hindi(
        output_dir=args.output_dir,
        files=args.files,
        num_samples=args.num_samples,
        save_format=args.format,
        cache_dir=args.cache_dir
    )

    print("\n" + "=" * 60)
    print("Download complete! Files saved:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    print("=" * 60)

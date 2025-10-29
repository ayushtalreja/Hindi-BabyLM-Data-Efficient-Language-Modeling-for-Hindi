"""
File I/O utilities for the Hindi data processing pipeline.

Provides functions for saving and loading data in various formats (pickle, text),
as well as cache management functionality.
"""

import pickle
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
from datetime import datetime

from .logging_utils import configure_logger

logger = configure_logger(__name__)


def save_pickle(data: List[str], filepath: Union[str, Path]) -> bool:
    """
    Save a list of strings to a pickle file.

    Args:
        data: List of strings to save
        filepath: Path to the output pickle file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"✓ Saved {len(data):,} texts to {filepath}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to save pickle to {filepath}: {e}")
        return False


def load_pickle(filepath: Union[str, Path]) -> List[str]:
    """
    Load a list of strings from a pickle file.

    Args:
        filepath: Path to the pickle file

    Returns:
        List[str]: List of strings, or empty list if loading fails
    """
    try:
        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"Pickle file not found: {filepath}")
            return []

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        logger.info(f"✓ Loaded {len(data):,} texts from {filepath}")
        return data

    except Exception as e:
        logger.error(f"✗ Failed to load pickle from {filepath}: {e}")
        return []


def save_text_file(
    data: List[str],
    filepath: Union[str, Path],
    sample_size: int = 100,
    encoding: str = 'utf-8'
) -> bool:
    """
    Save a sample of texts to a plain text file (one per line).

    Useful for manual inspection of data quality.

    Args:
        data: List of strings to save
        filepath: Path to the output text file
        sample_size: Number of samples to save (default: 100)
        encoding: Text encoding (default: 'utf-8')

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Take sample
        sample = data[:min(sample_size, len(data))]

        with open(filepath, 'w', encoding=encoding) as f:
            for text in sample:
                # Replace newlines with spaces for single-line format
                text_oneline = text.replace('\n', ' ').strip()
                f.write(text_oneline + '\n')

        logger.info(f"✓ Saved {len(sample):,} sample texts to {filepath}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to save text file to {filepath}: {e}")
        return False


# ====================================================================================
# Cache Management Functions (formerly in cache_manager.py)
# ====================================================================================


def check_cache_exists(filepath: Union[str, Path]) -> bool:
    """
    Check if a cache file exists and is valid (non-empty).

    Args:
        filepath: Path to cache file

    Returns:
        True if file exists and has content, False otherwise
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.debug(f"Cache miss: {filepath} does not exist")
        return False

    if filepath.stat().st_size == 0:
        logger.warning(f"Cache invalid: {filepath} is empty")
        return False

    logger.debug(f"Cache hit: {filepath} exists ({filepath.stat().st_size:,} bytes)")
    return True


def get_cache_info(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Get information about a cache file.

    Args:
        filepath: Path to cache file

    Returns:
        Dictionary with file information, or None if file doesn't exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return None

    stat = filepath.stat()

    info = {
        'path': str(filepath),
        'exists': True,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 ** 2),
        'size_gb': stat.st_size / (1024 ** 3),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
    }

    # Try to get item count for pickle files
    try:
        if filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    info['item_count'] = len(data)
        elif filepath.suffix in ['.txt', '.text']:
            with open(filepath, 'r', encoding='utf-8') as f:
                info['line_count'] = sum(1 for _ in f)
    except Exception as e:
        logger.debug(f"Could not get item count for {filepath}: {e}")

    return info


def get_cache_summary(cache_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Get summary statistics for all cache files in a directory.

    Args:
        cache_dir: Directory containing cache files

    Returns:
        Dictionary with summary statistics
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return {
            'exists': False,
            'total_files': 0,
            'total_size_bytes': 0,
            'total_size_gb': 0.0
        }

    cache_files = list(cache_dir.glob('*.pkl')) + list(cache_dir.glob('*.txt'))

    total_size = sum(f.stat().st_size for f in cache_files)

    summary = {
        'exists': True,
        'directory': str(cache_dir),
        'total_files': len(cache_files),
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 ** 2),
        'total_size_gb': total_size / (1024 ** 3),
        'files': {}
    }

    for cache_file in cache_files:
        summary['files'][cache_file.name] = {
            'size_bytes': cache_file.stat().st_size,
            'size_mb': cache_file.stat().st_size / (1024 ** 2),
            'modified': datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat()
        }

    return summary


def clear_cache(cache_dir: Union[str, Path], pattern: str = '*.pkl') -> int:
    """
    Clear cache files matching a pattern.

    Args:
        cache_dir: Directory containing cache files
        pattern: Glob pattern for files to delete (default: '*.pkl')

    Returns:
        Number of files deleted
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        return 0

    deleted = 0
    for cache_file in cache_dir.glob(pattern):
        try:
            cache_file.unlink()
            logger.info(f"Deleted cache file: {cache_file}")
            deleted += 1
        except Exception as e:
            logger.error(f"Failed to delete {cache_file}: {e}")

    logger.info(f"Cleared {deleted} cache files from {cache_dir}")
    return deleted


def print_cache_summary(cache_dir: Union[str, Path]):
    """
    Print a formatted summary of cache directory.

    Args:
        cache_dir: Directory containing cache files
    """
    summary = get_cache_summary(cache_dir)

    print("\n" + "=" * 70)
    print("Cache Directory Summary")
    print("=" * 70)

    if not summary['exists']:
        print(f"Cache directory does not exist: {cache_dir}")
        print("=" * 70)
        return

    print(f"Directory: {summary['directory']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Total size: {summary['total_size_gb']:.2f} GB ({summary['total_size_mb']:.1f} MB)")
    print("\nFiles:")
    print("-" * 70)

    for filename, info in summary['files'].items():
        print(f"  {filename:<30} {info['size_mb']:>10.1f} MB  {info['modified']}")

    print("=" * 70 + "\n")

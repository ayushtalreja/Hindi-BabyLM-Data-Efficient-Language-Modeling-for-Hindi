"""
Cache Manager for Hindi BabyLM Data Processing

This module provides utilities for caching downloaded and processed data
to avoid redundant downloads and speed up development iterations.

Key Features:
- Check if cache files exist and are valid
- Load data from cache (pickle or text format)
- Save data to cache with metadata
- Get cache information (size, modification time, etc.)
"""

import os
import pickle
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def check_cache_exists(filepath: Union[str, Path]) -> bool:
    """
    Check if a cache file exists and is valid (non-empty)

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


def load_from_cache(filepath: Union[str, Path], format: str = 'pickle') -> Optional[Union[List[str], Dict]]:
    """
    Load data from cache file

    Args:
        filepath: Path to cache file
        format: Format of cache file ('pickle' or 'text')

    Returns:
        Loaded data if successful, None if failed

    Raises:
        ValueError: If format is not supported
    """
    filepath = Path(filepath)

    if not check_cache_exists(filepath):
        return None

    try:
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"✓ Loaded from cache: {filepath} ({len(data):,} items)")
            return data

        elif format == 'text':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = [line.strip() for line in f if line.strip()]
            logger.info(f"✓ Loaded from cache: {filepath} ({len(data):,} lines)")
            return data

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'text'")

    except Exception as e:
        logger.error(f"Failed to load cache {filepath}: {e}")
        return None


def save_to_cache(data: Union[List[str], Dict], filepath: Union[str, Path], format: str = 'pickle') -> bool:
    """
    Save data to cache file

    Args:
        data: Data to save (list of strings or dictionary)
        filepath: Path where to save cache
        format: Format to save ('pickle' or 'text')

    Returns:
        True if save successful, False otherwise

    Raises:
        ValueError: If format is not supported
    """
    filepath = Path(filepath)

    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            size = filepath.stat().st_size
            logger.info(f"✓ Saved to cache: {filepath} ({size:,} bytes)")
            return True

        elif format == 'text':
            if not isinstance(data, list):
                raise ValueError("Text format requires list of strings")
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(str(item) + '\n')
            size = filepath.stat().st_size
            logger.info(f"✓ Saved to cache: {filepath} ({len(data):,} lines, {size:,} bytes)")
            return True

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'text'")

    except Exception as e:
        logger.error(f"Failed to save cache {filepath}: {e}")
        return False


def get_cache_info(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Get information about a cache file

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

    # Try to get line count for text/pickle files
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


def clear_cache(cache_dir: Union[str, Path], pattern: str = '*.pkl') -> int:
    """
    Clear cache files matching a pattern

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


def get_cache_summary(cache_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Get summary statistics for all cache files in a directory

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


def print_cache_summary(cache_dir: Union[str, Path]):
    """
    Print a formatted summary of cache directory

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

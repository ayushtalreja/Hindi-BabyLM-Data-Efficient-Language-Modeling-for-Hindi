"""
Evaluation Cache for Hindi BabyLM

This module provides caching functionality for evaluation predictions to avoid
redundant inference runs and speed up iterative evaluation workflows.

Key Features:
- Hash-based cache key generation from model checkpoint + dataset + config
- Efficient file hashing for large checkpoints
- Age-based cache validation (configurable expiration)
- Metadata tracking (timestamps, model info, dataset info)
- Safe cache cleanup with age-based retention
- Pickle-based serialization for predictions
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationCache:
    """
    Cache manager for evaluation predictions

    This class handles caching of model predictions to avoid redundant
    inference runs. It uses hash-based keys combining model checkpoint,
    dataset, and configuration to ensure cache validity.

    Features:
    - Automatic cache key generation from model + data + config
    - Age-based cache expiration (default: 30 days)
    - Metadata tracking for provenance
    - Safe concurrent access (file-based locking)
    - Efficient storage with pickle serialization
    """

    def __init__(
        self,
        cache_dir: str = ".eval_cache",
        max_cache_age_days: int = 30,
        enable_cache: bool = True
    ):
        """
        Initialize evaluation cache

        Args:
            cache_dir: Directory to store cache files
            max_cache_age_days: Maximum age of cache entries in days
            enable_cache: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_age_days = max_cache_age_days
        self.enable_cache = enable_cache

        # Create cache directory if it doesn't exist
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Evaluation cache initialized at: {self.cache_dir}")
            logger.info(f"Cache expiration: {max_cache_age_days} days")
        else:
            logger.info("Evaluation caching disabled")

    def _compute_cache_key(
        self,
        model_path: Optional[str] = None,
        model_hash: Optional[str] = None,
        dataset_name: str = "",
        dataset_split: str = "",
        config: Optional[Dict] = None
    ) -> str:
        """
        Compute cache key from model, dataset, and configuration

        Args:
            model_path: Path to model checkpoint file
            model_hash: Pre-computed model hash (alternative to model_path)
            dataset_name: Name of the dataset
            dataset_split: Dataset split (train/val/test)
            config: Evaluation configuration dictionary

        Returns:
            Cache key (hex digest)
        """
        # Initialize hash
        hasher = hashlib.sha256()

        # Hash model
        if model_hash:
            # Use pre-computed hash
            hasher.update(model_hash.encode('utf-8'))
        elif model_path and os.path.exists(model_path):
            # Compute hash from model file
            model_file_hash = self._hash_file(model_path)
            hasher.update(model_file_hash.encode('utf-8'))
        else:
            # No model identifier - use timestamp (cache will be unique per run)
            logger.warning("No model path or hash provided, cache key may not be stable")
            hasher.update(str(datetime.now().timestamp()).encode('utf-8'))

        # Hash dataset identifier
        dataset_id = f"{dataset_name}_{dataset_split}"
        hasher.update(dataset_id.encode('utf-8'))

        # Hash relevant config parameters
        if config:
            # Extract evaluation-relevant config
            relevant_config = {
                'batch_size': config.get('batch_size', config.get('eval_batch_size', 32)),
                'max_samples': config.get('max_samples_per_task'),
                'task_config': config.get('task_config', {}),
            }
            config_str = json.dumps(relevant_config, sort_keys=True)
            hasher.update(config_str.encode('utf-8'))

        cache_key = hasher.hexdigest()
        logger.debug(f"Generated cache key: {cache_key}")
        return cache_key

    def _hash_file(self, file_path: str, chunk_size: int = 8192) -> str:
        """
        Compute SHA256 hash of a file efficiently

        Args:
            file_path: Path to file
            chunk_size: Size of chunks to read (8KB default)

        Returns:
            Hex digest of file hash
        """
        hasher = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)

            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            # Return a timestamp-based hash as fallback
            return hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()

    def get_cached_predictions(
        self,
        cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached predictions if valid

        Args:
            cache_key: Cache key to look up

        Returns:
            Dictionary with predictions and metadata, or None if not found/invalid
        """
        if not self.enable_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"

        # Check if cache files exist
        if not cache_file.exists() or not metadata_file.exists():
            logger.debug(f"Cache miss: {cache_key}")
            return None

        # Load metadata to check validity
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading cache metadata: {e}")
            return None

        # Check cache validity
        if not self._is_cache_valid(metadata):
            logger.info(f"Cache expired or invalid: {cache_key}")
            # Clean up expired cache
            self._remove_cache_entry(cache_key)
            return None

        # Load predictions
        try:
            with open(cache_file, 'rb') as f:
                predictions = pickle.load(f)

            logger.info(f"Cache hit: {cache_key}")
            logger.info(f"  Created: {metadata.get('created_at')}")
            logger.info(f"  Samples: {metadata.get('n_samples', 0)}")

            return {
                'predictions': predictions,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error loading cached predictions: {e}")
            return None

    def save_predictions(
        self,
        cache_key: str,
        predictions: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save predictions to cache with metadata

        Args:
            cache_key: Cache key
            predictions: Predictions to cache (typically dict or list)
            metadata: Additional metadata to store

        Returns:
            True if successful, False otherwise
        """
        if not self.enable_cache:
            return False

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"

        # Prepare metadata
        full_metadata = {
            'cache_key': cache_key,
            'created_at': datetime.now().isoformat(),
            'n_samples': len(predictions) if hasattr(predictions, '__len__') else None,
        }

        # Add user-provided metadata
        if metadata:
            full_metadata.update(metadata)

        # Save predictions
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(full_metadata, f, indent=2)

            logger.info(f"Predictions cached: {cache_key}")
            logger.debug(f"  Samples: {full_metadata.get('n_samples', 'unknown')}")

            return True

        except Exception as e:
            logger.error(f"Error saving predictions to cache: {e}")
            # Clean up partial writes
            if cache_file.exists():
                cache_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            return False

    def _is_cache_valid(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if cache entry is still valid

        Args:
            metadata: Cache metadata dictionary

        Returns:
            True if cache is valid, False otherwise
        """
        # Check if created_at exists
        if 'created_at' not in metadata:
            logger.debug("Cache metadata missing 'created_at'")
            return False

        try:
            # Parse creation time
            created_at = datetime.fromisoformat(metadata['created_at'])

            # Check age
            age = datetime.now() - created_at
            max_age = timedelta(days=self.max_cache_age_days)

            if age > max_age:
                logger.debug(f"Cache expired: {age.days} days old (max: {self.max_cache_age_days})")
                return False

            return True

        except Exception as e:
            logger.warning(f"Error validating cache: {e}")
            return False

    def _remove_cache_entry(self, cache_key: str):
        """
        Remove a cache entry and its metadata

        Args:
            cache_key: Cache key to remove
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"

        try:
            if cache_file.exists():
                cache_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            logger.debug(f"Removed cache entry: {cache_key}")
        except Exception as e:
            logger.warning(f"Error removing cache entry: {e}")

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cache entries

        Args:
            older_than_days: Only clear entries older than this many days.
                           If None, clear all entries.
        """
        if not self.enable_cache or not self.cache_dir.exists():
            logger.info("No cache to clear")
            return

        removed_count = 0
        total_count = 0

        # Iterate over cache files
        for cache_file in self.cache_dir.glob("*.pkl"):
            total_count += 1
            cache_key = cache_file.stem
            metadata_file = self.cache_dir / f"{cache_key}_metadata.json"

            # If age filter specified, check metadata
            if older_than_days is not None:
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        created_at = datetime.fromisoformat(metadata['created_at'])
                        age = datetime.now() - created_at

                        if age.days <= older_than_days:
                            continue  # Keep this entry
                    except:
                        pass  # Remove if we can't read metadata

            # Remove entry
            self._remove_cache_entry(cache_key)
            removed_count += 1

        logger.info(f"Cache cleanup: removed {removed_count}/{total_count} entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache

        Returns:
            Dictionary with cache statistics
        """
        if not self.enable_cache or not self.cache_dir.exists():
            return {
                'enabled': False,
                'total_entries': 0,
                'total_size_mb': 0
            }

        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        # Count expired entries
        expired_count = 0
        for cache_file in cache_files:
            cache_key = cache_file.stem
            metadata_file = self.cache_dir / f"{cache_key}_metadata.json"

            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    if not self._is_cache_valid(metadata):
                        expired_count += 1
                except:
                    expired_count += 1

        return {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'total_entries': len(cache_files),
            'expired_entries': expired_count,
            'total_size_mb': total_size / (1024 * 1024),
            'max_age_days': self.max_cache_age_days
        }

    def invalidate_by_model(self, model_path: str):
        """
        Invalidate all cache entries for a specific model

        Args:
            model_path: Path to model checkpoint
        """
        if not self.enable_cache or not os.path.exists(model_path):
            return

        # Compute model hash
        model_hash = self._hash_file(model_path)

        removed_count = 0

        # Check all cache entries
        for metadata_file in self.cache_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Check if this cache entry is for the specified model
                if metadata.get('model_hash') == model_hash:
                    cache_key = metadata_file.stem.replace('_metadata', '')
                    self._remove_cache_entry(cache_key)
                    removed_count += 1

            except Exception as e:
                logger.debug(f"Error checking cache metadata: {e}")

        logger.info(f"Invalidated {removed_count} cache entries for model: {model_path}")


# Convenience function
def create_cache_manager(config: Dict[str, Any]) -> EvaluationCache:
    """
    Create cache manager from configuration

    Args:
        config: Configuration dictionary with evaluation settings

    Returns:
        EvaluationCache instance
    """
    eval_config = config.get('evaluation', {})

    return EvaluationCache(
        cache_dir=eval_config.get('cache_dir', '.eval_cache'),
        max_cache_age_days=eval_config.get('max_cache_age_days', 30),
        enable_cache=eval_config.get('use_eval_cache', True)
    )

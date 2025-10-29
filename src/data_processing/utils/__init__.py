"""
Shared utilities for Hindi data processing pipeline.

This module provides common functionality used across different data downloaders:
- File I/O operations (pickle, text files)
- Corpus statistics calculation
- Metadata generation and saving
- Text sampling strategies
- Hindi text validation and language detection
- Logging configuration
"""

from .file_io import (
    save_pickle,
    load_pickle,
    save_text_file,
    check_cache_exists,
    get_cache_info,
    get_cache_summary,
    clear_cache,
    print_cache_summary
)
from .statistics import calculate_corpus_stats
from .metadata import generate_metadata, save_metadata
from .sampling import sample_texts
from .hindi_utils import HindiValidator
from .logging_utils import configure_logger, get_logger
from .http_utils import HTTPClient, get_url

__all__ = [
    "save_pickle",
    "load_pickle",
    "save_text_file",
    "check_cache_exists",
    "get_cache_info",
    "get_cache_summary",
    "clear_cache",
    "print_cache_summary",
    "calculate_corpus_stats",
    "generate_metadata",
    "save_metadata",
    "sample_texts",
    "HindiValidator",
    "configure_logger",
    "get_logger",
    "HTTPClient",
    "get_url"
]

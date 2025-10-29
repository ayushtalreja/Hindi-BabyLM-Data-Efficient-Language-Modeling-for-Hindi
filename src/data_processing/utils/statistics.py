"""
Corpus statistics calculation utilities.

Provides functions for calculating various statistics about Hindi text corpora.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from .hindi_utils import HindiValidator
from .logging_utils import configure_logger

logger = configure_logger(__name__)


def calculate_corpus_stats(
    texts: List[str],
    source_name: str,
    num_samples: Optional[int] = 100000
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a text corpus.

    Args:
        texts: List of text strings
        source_name: Name of the data source
        num_samples: Maximum number of texts to sample for statistics (None = all)

    Returns:
        Dict containing various corpus statistics
    """
    logger.info(f"Calculating statistics for {source_name}...")

    # Sample if needed
    sample_texts = texts[:num_samples] if num_samples else texts
    total_texts = len(texts)
    sampled_texts = len(sample_texts)

    # Initialize counters
    total_chars = 0
    total_words = 0
    total_devanagari_chars = 0
    char_lengths = []
    word_lengths = []

    for text in sample_texts:
        # Character count
        char_count = len(text)
        total_chars += char_count
        char_lengths.append(char_count)

        # Word count (split on whitespace)
        words = text.split()
        word_count = len(words)
        total_words += word_count
        word_lengths.append(word_count)

        # Use centralized HindiValidator for Devanagari character count
        devanagari_chars = HindiValidator.count_devanagari_chars(text)
        total_devanagari_chars += devanagari_chars

    # Calculate statistics
    avg_char_length = total_chars / sampled_texts if sampled_texts > 0 else 0
    avg_word_length = total_words / sampled_texts if sampled_texts > 0 else 0
    devanagari_ratio = total_devanagari_chars / total_chars if total_chars > 0 else 0

    # Min/max lengths
    min_char_length = min(char_lengths) if char_lengths else 0
    max_char_length = max(char_lengths) if char_lengths else 0
    min_word_length = min(word_lengths) if word_lengths else 0
    max_word_length = max(word_lengths) if word_lengths else 0

    stats = {
        "source": source_name,
        "total_samples": total_texts,
        "sampled_for_stats": sampled_texts,
        "total_characters": total_chars,
        "total_words": total_words,
        "total_devanagari_chars": total_devanagari_chars,
        "devanagari_ratio": round(devanagari_ratio, 4),
        "avg_char_length": round(avg_char_length, 2),
        "avg_word_length": round(avg_word_length, 2),
        "min_char_length": min_char_length,
        "max_char_length": max_char_length,
        "min_word_length": min_word_length,
        "max_word_length": max_word_length
    }

    logger.info(f"✓ Statistics calculated: {total_texts:,} texts, "
                f"{total_words:,} words, {devanagari_ratio:.1%} Devanagari")

    return stats

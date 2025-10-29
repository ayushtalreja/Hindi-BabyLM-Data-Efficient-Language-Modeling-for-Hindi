"""
Text sampling utilities.

Provides functions for sampling texts from a corpus based on various strategies.
"""

import random
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def sample_texts(
    texts: List[str],
    max_samples: Optional[int] = None,
    max_words: Optional[int] = None,
    strategy: str = "first",
    seed: Optional[int] = 42
) -> List[str]:
    """
    Sample texts from a corpus based on specified constraints and strategy.

    Args:
        texts: List of text strings
        max_samples: Maximum number of samples to return (None = no limit)
        max_words: Maximum total words across all samples (None = no limit)
        strategy: Sampling strategy: "first", "random", or "balanced"
        seed: Random seed for reproducibility (only used for "random" strategy)

    Returns:
        List of sampled texts

    Strategies:
        - "first": Take the first N texts (fastest, preserves order)
        - "random": Random sampling (more diverse, requires shuffling)
        - "balanced": Try to balance by length (experimental)
    """
    if not texts:
        return []

    # If no constraints, return all texts
    if max_samples is None and max_words is None:
        logger.info(f"No sampling constraints - returning all {len(texts):,} texts")
        return texts

    # Initialize based on strategy
    if strategy == "random":
        if seed is not None:
            random.seed(seed)
        texts_to_sample = texts.copy()
        random.shuffle(texts_to_sample)
    else:
        texts_to_sample = texts

    sampled = []
    total_words = 0

    for text in texts_to_sample:
        # Check max_samples constraint
        if max_samples and len(sampled) >= max_samples:
            break

        # Check max_words constraint
        if max_words:
            word_count = len(text.split())
            if total_words + word_count > max_words:
                break
            total_words += word_count

        sampled.append(text)

    logger.info(f"✓ Sampled {len(sampled):,} texts from {len(texts):,} "
                f"(strategy: {strategy})")

    return sampled

"""
Metadata generation and saving utilities.

Provides functions for generating and saving metadata about data sources.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_metadata(
    texts: List[str],
    source_name: str,
    stats: Dict[str, Any],
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for a data source.

    Args:
        texts: List of text strings
        source_name: Name of the data source
        stats: Statistics dictionary (from calculate_corpus_stats)
        additional_info: Optional additional information to include

    Returns:
        Dict containing metadata
    """
    metadata = {
        "source": source_name,
        "timestamp": datetime.now().isoformat(),
        "num_texts": len(texts),
        "statistics": stats
    }

    # Add additional info if provided
    if additional_info:
        metadata.update(additional_info)

    return metadata


def save_metadata(
    metadata: Dict[str, Any],
    filepath: Union[str, Path]
) -> bool:
    """
    Save metadata to a JSON file.

    Args:
        metadata: Metadata dictionary to save
        filepath: Path to the output JSON file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved metadata to {filepath}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to save metadata to {filepath}: {e}")
        return False

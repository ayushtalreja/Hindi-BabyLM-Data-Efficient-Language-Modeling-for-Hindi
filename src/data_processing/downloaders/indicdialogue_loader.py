"""
IndicDialogue dataset loader.

Loads the IndicDialogue dataset from a JSONL file. The dataset contains
Hindi movie subtitles, providing conversational/informal Hindi text.

Source: https://data.mendeley.com/datasets/wcb4bxbyxx/2
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .base_downloader import BaseDownloader
from ..text_cleaner import clean_text
from ..utils import configure_logger

logger = configure_logger(__name__)


class IndicDialogueLoader(BaseDownloader):
    """
    Load IndicDialogue dataset from JSONL file.

    The dataset contains movie subtitles in Hindi, providing conversational
    and informal language examples.
    """

    def __init__(
        self,
        jsonl_path: str = "data/raw/hindi.jsonl",
        output_dir: str = "data/raw"
    ):
        """
        Initialize IndicDialogue loader.

        Args:
            jsonl_path: Path to the hindi.jsonl file
            output_dir: Directory for saving output files
        """
        super().__init__(output_dir)
        self.jsonl_path = Path(jsonl_path)
        self.source_name = "indicdialogue"

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"IndicDialogue file not found: {jsonl_path}")

        logger.info(f"Initialized IndicDialogueLoader (file: {jsonl_path})")

    def download(
        self,
        max_movies: Optional[int] = None,
        min_dialogue_length: int = 10,
        combine_dialogues: bool = False
    ) -> tuple[List[str], int]:
        """
        Load IndicDialogue from JSONL file.

        Args:
            max_movies: Maximum number of movies to process (None = all)
            min_dialogue_length: Minimum characters per dialogue line
            combine_dialogues: If True, combine all dialogues from a movie into one text.
                              If False, keep each dialogue line separate.

        Returns:
            Tuple of (list of dialogue texts, total word count)
        """
        logger.info(f"\nLoading IndicDialogue from {self.jsonl_path}")
        logger.info(f"  Settings: combine_dialogues={combine_dialogues}, "
                   f"min_length={min_dialogue_length}")

        texts = []
        total_words = 0  # Track word count incrementally
        movies_processed = 0
        dialogues_extracted = 0
        dialogues_filtered = 0
        parse_errors = 0

        try:
            # Count total lines for progress bar
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)

            logger.info(f"  Total movies in file: {total_lines:,}")

            # Process JSONL file
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Loading movies"), 1):
                    # Check max_movies limit
                    if max_movies and movies_processed >= max_movies:
                        break

                    try:
                        # Parse JSON line
                        data = json.loads(line)
                        hindi_dialogues = data.get("dialogs", {}).get("hin", [])

                        if not hindi_dialogues:
                            continue

                        if combine_dialogues:
                            # Combine all dialogues from this movie
                            combined_text = " ".join(hindi_dialogues)
                            cleaned = clean_text(combined_text)

                            if len(cleaned) >= min_dialogue_length:
                                # Count words incrementally (one text at a time to avoid memory spike)
                                total_words += len(cleaned.split())
                                texts.append(cleaned)
                                dialogues_extracted += 1
                            else:
                                dialogues_filtered += 1
                        else:
                            # Keep each dialogue line separate
                            for dialogue in hindi_dialogues:
                                cleaned = clean_text(dialogue)

                                if len(cleaned) >= min_dialogue_length:
                                    # Count words incrementally (one dialogue at a time to avoid memory spike)
                                    total_words += len(cleaned.split())
                                    texts.append(cleaned)
                                    dialogues_extracted += 1
                                else:
                                    dialogues_filtered += 1

                        movies_processed += 1

                        # Progress update every 500 movies
                        if movies_processed % 500 == 0:
                            logger.info(f"  Processed: {movies_processed:,} movies | "
                                       f"Extracted: {dialogues_extracted:,} dialogues")

                    except json.JSONDecodeError as e:
                        parse_errors += 1
                        logger.warning(f"  Failed to parse line {line_num}: {e}")
                        continue

            logger.info(f"\n✓ IndicDialogue loading complete!")
            logger.info(f"  Movies processed: {movies_processed:,}")
            logger.info(f"  Dialogues extracted: {dialogues_extracted:,}")
            logger.info(f"  Dialogues filtered: {dialogues_filtered:,}")
            logger.info(f"  Total words: {total_words:,}")
            if parse_errors > 0:
                logger.warning(f"  Parse errors: {parse_errors}")

            return texts, total_words

        except Exception as e:
            logger.error(f"✗ Failed to load IndicDialogue: {e}")
            raise

    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about the IndicDialogue data source.

        Returns:
            Dict with source metadata
        """
        return {
            "source": "IndicDialogue Dataset",
            "url": "https://data.mendeley.com/datasets/wcb4bxbyxx/2",
            "file": str(self.jsonl_path),
            "format": "JSONL (JSON Lines)",
            "content_type": "Movie subtitles - conversational Hindi",
            "language": "Hindi",
            "description": "Dialogue data from Hindi movie subtitles"
        }


def load_indicdialogue_hindi(
    jsonl_path: str = "data/raw/hindi.jsonl",
    output_dir: str = 'data/raw',
    max_movies: Optional[int] = None,
    min_dialogue_length: int = 10,
    combine_dialogues: bool = False,
    save_stats: bool = True
) -> Path:
    """
    Convenience function to load IndicDialogue dataset.

    Args:
        jsonl_path: Path to the hindi.jsonl file
        output_dir: Directory for saving output files
        max_movies: Maximum number of movies (None = all)
        min_dialogue_length: Minimum characters per dialogue
        combine_dialogues: Whether to combine dialogues per movie
        save_stats: Whether to calculate and save statistics

    Returns:
        Path to the cached pickle file
    """
    loader = IndicDialogueLoader(
        jsonl_path=jsonl_path,
        output_dir=output_dir
    )

    return loader.process_and_cache(
        max_samples=None,  # We filter during loading
        save_stats=save_stats,
        max_movies=max_movies,
        min_dialogue_length=min_dialogue_length,
        combine_dialogues=combine_dialogues
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("Loading IndicDialogue dataset...")
    cache_path = load_indicdialogue_hindi(
        max_movies=100,  # Test with 100 movies
        combine_dialogues=False,
        save_stats=True
    )
    print(f"\n✓ Cached at: {cache_path}")

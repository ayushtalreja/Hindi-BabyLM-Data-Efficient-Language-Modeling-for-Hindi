"""
Data Provenance Tracking Utilities.

Provides comprehensive tracking of data sources throughout the corpus building pipeline,
including raw data statistics, deduplication metrics, and split distributions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DataProvenanceTracker:
    """
    Tracks data provenance statistics throughout the corpus building pipeline.

    This class incrementally tracks:
    - Raw data statistics (documents and words from each source)
    - Deduplication statistics (duplicates removed per source)
    - Split statistics (distribution of sources across train/val/test splits)

    Usage:
        tracker = DataProvenanceTracker(config)
        tracker.add_raw_data_stats('wikipedia', 25000, 8750000)
        tracker.add_deduplication_stats('wikipedia', 25000, 2000)
        tracker.add_split_stats('train', 'wikipedia', 15000, 5000000)
        tracker.save_report('results/data_processing')
    """

    def __init__(self, config):
        """
        Initialize the provenance tracker.

        Args:
            config: Configuration object containing pipeline settings
        """
        self.config = config
        self.raw_stats = {}
        self.deduplication_stats = {}
        self.split_stats = {
            'train': {'sources': {}, 'total_documents': 0, 'total_words': 0},
            'val': {'sources': {}, 'total_documents': 0, 'total_words': 0},
            'test': {'sources': {}, 'total_documents': 0, 'total_words': 0}
        }

    def add_raw_data_stats(self, source_name: str, documents: int, words: int):
        """
        Track raw data statistics for a source before processing.

        Args:
            source_name: Name of the data source (e.g., 'indiccorp', 'wikipedia')
            documents: Number of documents/texts collected
            words: Total word count across all documents
        """
        self.raw_stats[source_name] = {
            'total_documents': documents,
            'total_words': words
        }
        logger.debug(f"Tracked raw data for {source_name}: {documents:,} docs, {words:,} words")

    def add_deduplication_stats(self, source_name: str, original_count: int, removed_count: int):
        """
        Track deduplication statistics for a source.

        Args:
            source_name: Name of the data source
            original_count: Number of documents before deduplication
            removed_count: Number of duplicate documents removed
        """
        remaining_count = original_count - removed_count
        deduplication_rate = removed_count / original_count if original_count > 0 else 0.0

        self.deduplication_stats[source_name] = {
            'original_count': original_count,
            'duplicates_removed': removed_count,
            'remaining_count': remaining_count,
            'deduplication_rate': round(deduplication_rate, 4)
        }
        logger.debug(f"Tracked deduplication for {source_name}: {removed_count:,}/{original_count:,} removed")

    def add_split_stats(self, split_name: str, source_name: str, document_count: int, word_count: int):
        """
        Track statistics for a specific source within a split.

        Args:
            split_name: Name of the split ('train', 'val', or 'test')
            source_name: Name of the data source
            document_count: Number of documents from this source in this split
            word_count: Number of words from this source in this split
        """
        if split_name not in self.split_stats:
            logger.warning(f"Unknown split name: {split_name}")
            return

        self.split_stats[split_name]['sources'][source_name] = {
            'document_count': document_count,
            'word_count': word_count
        }

        # Update split totals
        self.split_stats[split_name]['total_documents'] += document_count
        self.split_stats[split_name]['total_words'] += word_count

        logger.debug(f"Tracked {split_name} split for {source_name}: {document_count:,} docs, {word_count:,} words")

    def _calculate_percentages(self):
        """
        Calculate percentage distributions for each split.

        Adds percentage_of_split_documents and percentage_of_split_words to each source
        within each split.
        """
        for split_name, split_data in self.split_stats.items():
            total_docs = split_data['total_documents']
            total_words = split_data['total_words']

            for source_name, source_data in split_data['sources'].items():
                # Calculate document percentage
                if total_docs > 0:
                    doc_percentage = (source_data['document_count'] / total_docs) * 100
                    source_data['percentage_of_split_documents'] = round(doc_percentage, 2)
                else:
                    source_data['percentage_of_split_documents'] = 0.0

                # Calculate word percentage
                if total_words > 0:
                    word_percentage = (source_data['word_count'] / total_words) * 100
                    source_data['percentage_of_split_words'] = round(word_percentage, 2)
                else:
                    source_data['percentage_of_split_words'] = 0.0

    def _identify_sources_used_and_empty(self) -> tuple:
        """
        Identify which sources were used and which were configured but returned no data.

        Returns:
            Tuple of (sources_used, sources_configured_but_empty)
        """
        # Get all sources that have any data
        sources_with_data = set(self.raw_stats.keys())

        # Get all sources mentioned in splits
        sources_in_splits = set()
        for split_data in self.split_stats.values():
            sources_in_splits.update(split_data['sources'].keys())

        # Sources used are those that appear in splits
        sources_used = sorted(list(sources_in_splits))

        # Sources configured but empty are those with 0 documents in raw stats
        sources_configured_but_empty = sorted([
            source for source, stats in self.raw_stats.items()
            if stats['total_documents'] == 0
        ])

        return sources_used, sources_configured_but_empty

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate complete statistics report with percentages and summary.

        Returns:
            Dictionary containing complete provenance report
        """
        # Calculate percentages for all splits
        self._calculate_percentages()

        # Identify sources used and empty
        sources_used, sources_empty = self._identify_sources_used_and_empty()

        # Calculate total statistics across all splits
        total_documents = sum(split_data['total_documents'] for split_data in self.split_stats.values())
        total_words = sum(split_data['total_words'] for split_data in self.split_stats.values())

        # Extract relevant config parameters
        config_dict = self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        relevant_config = {
            'train_word_limit': config_dict.get('train_word_limit', None),
            'val_word_limit': config_dict.get('val_word_limit', None),
            'test_word_limit': config_dict.get('test_word_limit', None),
            'train_source_ratios': config_dict.get('train_source_ratios', {}),
        }

        # Build complete report
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': relevant_config
            },
            'raw_data_statistics': self.raw_stats,
            'deduplication_statistics': self.deduplication_stats,
            'split_statistics': self.split_stats,
            'summary': {
                'total_documents': total_documents,
                'total_words': total_words,
                'sources_used': sources_used,
                'sources_configured_but_empty': sources_empty
            }
        }

        return report

    def save_report(self, output_dir: str) -> bool:
        """
        Save the provenance report to a JSON file.

        Args:
            output_dir: Directory where the report should be saved

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate the report
            report = self.generate_report()

            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save to JSON file
            report_file = output_path / 'data_provenance_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Saved data provenance report to {report_file}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to save data provenance report: {e}")
            return False

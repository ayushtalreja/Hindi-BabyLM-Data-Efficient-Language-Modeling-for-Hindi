"""
Tests for data processing functionality
"""

import pytest
from pathlib import Path
from src.data_processing.text_cleaner import clean_text
from src.data_processing.quality_filter import QualityFilter


@pytest.mark.unit
@pytest.mark.data
class TestTextCleaner:
    """Test text cleaning functionality"""

    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        text = "  यह  एक   परीक्षण   है  "
        cleaned = clean_text(text)

        # Should remove extra whitespace
        assert "  " not in cleaned
        assert cleaned.strip() == cleaned

    def test_clean_text_removes_urls(self):
        """Test that URLs are removed"""
        text = "यह एक परीक्षण है http://example.com और यहाँ"
        cleaned = clean_text(text)

        assert "http://" not in cleaned
        assert "example.com" not in cleaned

    def test_clean_text_preserves_hindi(self):
        """Test that Hindi text is preserved"""
        text = "यह हिंदी है"
        cleaned = clean_text(text)

        assert "यह" in cleaned
        assert "हिंदी" in cleaned
        assert "है" in cleaned

    def test_clean_text_empty_string(self):
        """Test cleaning empty string"""
        text = ""
        cleaned = clean_text(text)
        assert cleaned == ""

    def test_clean_text_only_whitespace(self):
        """Test cleaning whitespace-only string"""
        text = "   \n  \t  "
        cleaned = clean_text(text)
        assert len(cleaned) == 0 or cleaned.isspace()


@pytest.mark.unit
@pytest.mark.data
class TestQualityFilter:
    """Test quality filtering functionality"""

    def test_quality_filter_initialization(self):
        """Test QualityFilter can be initialized"""
        config = {
            'min_length': 10,
            'max_length': 1000,
            'min_hindi_ratio': 0.5
        }

        qf = QualityFilter(config)
        assert qf.min_length == 10
        assert qf.max_length == 1000

    def test_filter_by_length(self):
        """Test filtering by length"""
        config = {
            'min_length': 10,
            'max_length': 100,
            'min_hindi_ratio': 0.5
        }

        qf = QualityFilter(config)

        # Too short
        assert not qf.passes_length_filter("छोटा")

        # Just right
        assert qf.passes_length_filter("यह एक अच्छा वाक्य है जो परीक्षण के लिए है")

        # Too long
        long_text = "बहुत " * 100
        assert not qf.passes_length_filter(long_text)

    def test_hindi_ratio_check(self):
        """Test Hindi character ratio check"""
        config = {
            'min_length': 5,
            'max_length': 1000,
            'min_hindi_ratio': 0.7
        }

        qf = QualityFilter(config)

        # Mostly Hindi
        hindi_text = "यह पूरी तरह से हिंदी है"
        assert qf.is_hindi_text(hindi_text)

        # Mostly English
        english_text = "This is mostly English with थोड़ा Hindi"
        assert not qf.is_hindi_text(english_text)


@pytest.mark.unit
@pytest.mark.data
def test_sample_texts_are_valid(sample_hindi_texts):
    """Test that sample texts are valid Hindi"""
    for text in sample_hindi_texts:
        assert len(text) > 0
        # Check for Devanagari
        has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
        assert has_devanagari


@pytest.mark.unit
@pytest.mark.data
def test_sample_corpus_file_creation(sample_corpus_file):
    """Test that sample corpus file is created correctly"""
    assert sample_corpus_file.exists()
    assert sample_corpus_file.is_file()

    # Read and verify content
    with open(sample_corpus_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    assert len(lines) > 0

    # Each line should have content
    for line in lines:
        assert len(line.strip()) > 0

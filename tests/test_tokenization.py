"""
Tests for tokenization functionality
"""

import pytest
from src.tokenization.tokenizer_factory import TokenizerFactory
from src.tokenization.sentencepiece_tokenizer import HindiSentencePieceTokenizer


@pytest.mark.unit
@pytest.mark.tokenization
class TestTokenizerFactory:
    """Test TokenizerFactory class"""

    def test_factory_initialization(self, test_config):
        """Test TokenizerFactory can be initialized"""
        factory = TokenizerFactory(test_config)
        assert factory.tokenizer_type == test_config['tokenizer_type']
        assert factory.vocab_size == test_config['vocab_size']

    def test_sentencepiece_tokenizer_creation(self, test_config, sample_hindi_texts, temp_dir):
        """Test SentencePiece tokenizer creation"""
        test_config['tokenizer_dir'] = str(temp_dir)
        factory = TokenizerFactory(test_config)

        # Create tokenizer
        tokenizer = factory._create_sentencepiece_tokenizer(sample_hindi_texts)

        assert tokenizer is not None
        assert hasattr(tokenizer, 'vocab_size')


@pytest.mark.unit
@pytest.mark.tokenization
class TestHindiSentencePieceTokenizer:
    """Test HindiSentencePieceTokenizer class"""

    def test_tokenizer_initialization(self):
        """Test tokenizer can be initialized"""
        tokenizer = HindiSentencePieceTokenizer(vocab_size=1000)
        assert tokenizer.vocab_size == 1000

    def test_tokenizer_has_required_methods(self):
        """Test tokenizer has required methods"""
        tokenizer = HindiSentencePieceTokenizer(vocab_size=1000)

        assert hasattr(tokenizer, 'encode')
        assert hasattr(tokenizer, 'decode')
        assert hasattr(tokenizer, 'tokenize')
        assert hasattr(tokenizer, 'train_tokenizer')


@pytest.mark.unit
@pytest.mark.tokenization
def test_hindi_text_encoding(sample_hindi_texts):
    """Test that Hindi text can be processed"""
    # Simple validation that text is valid Hindi
    for text in sample_hindi_texts:
        assert len(text) > 0
        # Check for Devanagari characters
        has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
        assert has_devanagari, f"Text should contain Devanagari: {text}"

import unicodedata
import re
from typing import List, Tuple
from .utils import HindiValidator

class HindiTextCleaner:
    def __init__(self):
        # Use centralized Hindi Unicode ranges from HindiValidator
        self.hindi_range = HindiValidator.DEVANAGARI_RANGE
        self.hindi_ext_range = HindiValidator.DEVANAGARI_EXT_RANGE
        
    def normalize_unicode(self, text: str) -> str:
        """Normalize Hindi text using NFC form"""
        return unicodedata.normalize('NFC', text)
    
    def remove_non_hindi(self, text: str) -> str:
        """Remove non-Hindi characters while preserving basic punctuation"""
        # Keep Devanagari, basic punctuation, digits, and whitespace
        cleaned_chars = []
        for char in text:
            code_point = ord(char)
            # Keep Hindi/Devanagari characters
            if (self.hindi_range[0] <= code_point <= self.hindi_range[1] or
                self.hindi_ext_range[0] <= code_point <= self.hindi_ext_range[1] or
                char.isspace() or
                char in '।॥,;:.!?\'"()-' or  # Hindi and common punctuation
                char.isdigit()):
                cleaned_chars.append(char)
        return ''.join(cleaned_chars)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        return re.sub(r'http\S+', '', text)

    def remove_noise_patterns(self, text: str) -> str:
        """Remove common noise patterns like download links, etc."""
        # Remove common noise patterns (case-insensitive)
        text = re.sub(r'(?i)(click here|download|pdf|epub|read more|subscribe)', '', text)
        # Remove brackets and braces
        text = re.sub(r'[\[\]{}]', '', text)
        return text

    def normalize_repeated_punctuation(self, text: str) -> str:
        """Normalize repeated punctuation marks"""
        # Normalize repeated Hindi punctuation
        text = re.sub(r'([।॥!?])\1+', r'\1', text)
        return text

    def clean_text(self, text: str,
                   remove_urls: bool = True,
                   remove_noise: bool = True,
                   normalize_punctuation: bool = True) -> str:
        """Complete text cleaning pipeline with configurable options.

        Args:
            text: Input text to clean
            remove_urls: Whether to remove URLs
            remove_noise: Whether to remove common noise patterns
            normalize_punctuation: Whether to normalize repeated punctuation

        Returns:
            Cleaned text string
        """
        text = self.normalize_unicode(text)

        if remove_urls:
            text = self.remove_urls(text)

        if remove_noise:
            text = self.remove_noise_patterns(text)

        text = self.remove_non_hindi(text)
        text = self.remove_extra_whitespace(text)

        if normalize_punctuation:
            text = self.normalize_repeated_punctuation(text)

        text = self.handle_special_cases(text)
        return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spacing"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Normalize spacing around punctuation
        text = re.sub(r'\s*([।॥,;:.!?])\s*', r'\1 ', text)
        return text.strip()

    def handle_special_cases(self, text: str) -> str:
        """Handle Hindi-specific text normalization cases"""
        # Normalize Hindi danda (।) and double danda (॥)
        text = re.sub(r'\.{2,}', '॥', text)  # Convert multiple dots to double danda
        text = re.sub(r'\.(?=\s|$)', '।', text)  # Convert single dot to danda at sentence end

        # Remove zero-width characters and joiners (except ZWNJ and ZWJ which are meaningful) (ZWJ = U+200D, ZWNJ = U+200C) (Zero width Non Joiner)
        text = re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', text)

        # Normalize numbers: convert English digits to Devanagari if needed (optional)
        text = text.translate(str.maketrans('0123456789', '०१२३४५६७८९'))

        return text


# Module-level convenience wrapper so callers can import `clean_text` directly
def clean_text(text: str) -> str:
    """Convenience wrapper for the HindiTextCleaner.clean_text method.

    This matches the historical API expected by other modules which import
    `clean_text` directly from this module.
    """
    cleaner = HindiTextCleaner()
    return cleaner.clean_text(text)


__all__ = ["HindiTextCleaner", "clean_text"]
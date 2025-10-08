import unicodedata
import re
from typing import List, Tuple

class HindiTextCleaner:
    def __init__(self):
        # Define Hindi Unicode ranges
        self.hindi_range = (0x0900, 0x097F)  # Devanagari block
        self.hindi_ext_range = (0xA8E0, 0xA8FF)  # Extended Devanagari
        
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
    
    def clean_text(self, text: str) -> str:
        """Complete text cleaning pipeline"""
        text = self.normalize_unicode(text)
        text = self.remove_non_hindi(text)
        text = self.remove_extra_whitespace(text)
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

        # Remove zero-width characters and joiners (except ZWNJ and ZWJ which are meaningful)
        text = re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', text)

        # Normalize numbers: convert English digits to Devanagari if needed (optional)
        # text = text.translate(str.maketrans('0123456789', '०१२३४५६७८९'))

        return text
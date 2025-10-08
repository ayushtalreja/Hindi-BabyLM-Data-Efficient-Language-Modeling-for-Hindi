import pandas as pd
from typing import List, Dict, Tuple

class QualityFilter:
    def __init__(self, min_length=10, max_length=1000, min_hindi_ratio=0.8):
        self.min_length = min_length
        self.max_length = max_length
        self.min_hindi_ratio = min_hindi_ratio
    
    def filter_by_length(self, texts: List[str]) -> List[str]:
        """Filter texts by character length"""
        return [text for text in texts
                if self.min_length <= len(text) <= self.max_length]
    
    def filter_by_language(self, texts: List[str]) -> List[str]:
        """Filter texts with high Hindi character ratio"""
        filtered = []
        for text in texts:
            if len(text) == 0:
                continue
            # Count Hindi/Devanagari characters
            hindi_chars = sum(1 for char in text
                            if 0x0900 <= ord(char) <= 0x097F or
                               0xA8E0 <= ord(char) <= 0xA8FF)
            ratio = hindi_chars / len(text)
            if ratio >= self.min_hindi_ratio:
                filtered.append(text)
        return filtered
    
    def detect_duplicates(self, texts: List[str], threshold=0.9) -> List[int]:
        """Detect near-duplicate texts using fuzzy matching"""
        from difflib import SequenceMatcher
        duplicate_indices = set()

        for i in range(len(texts)):
            if i in duplicate_indices:
                continue
            for j in range(i + 1, len(texts)):
                if j in duplicate_indices:
                    continue
                # Calculate similarity ratio
                similarity = SequenceMatcher(None, texts[i], texts[j]).ratio()
                if similarity >= threshold:
                    duplicate_indices.add(j)

        return list(duplicate_indices)
    
    def calculate_readability_score(self, text: str) -> float:
        """Calculate text readability (sentence length, word complexity)"""
        # Split by Hindi and English sentence terminators
        sentences = [s.strip() for s in text.split('।') if s.strip()]
        if not sentences:
            sentences = [s.strip() for s in text.split('.') if s.strip()]

        if not sentences:
            return 0.0

        # Calculate average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Calculate average word length
        words = text.split()
        if not words:
            return 0.0
        avg_word_length = sum(len(w) for w in words) / len(words)

        # Simple readability score (lower is easier to read)
        # Normalize to 0-1 range where higher is more readable
        score = 1.0 / (1.0 + (avg_sentence_length / 20.0) + (avg_word_length / 10.0))

        return score
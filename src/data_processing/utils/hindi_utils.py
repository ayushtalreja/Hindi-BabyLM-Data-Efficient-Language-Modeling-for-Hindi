"""
Hindi language utilities for text validation and processing.

This module provides centralized utilities for detecting and validating Hindi (Devanagari) text,
eliminating code duplication across the data processing pipeline.
"""

from typing import Optional


class HindiValidator:
    """Centralized Hindi text validation utilities.

    This class provides methods for detecting Devanagari characters and validating
    whether text is primarily in Hindi. It consolidates logic previously duplicated
    across multiple modules.

    Unicode Ranges:
        - Devanagari: U+0900 to U+097F (128 characters)
        - Devanagari Extended: U+A8E0 to U+A8FF (32 characters)
    """

    # Unicode ranges for Devanagari script
    DEVANAGARI_RANGE = (0x0900, 0x097F)
    DEVANAGARI_EXT_RANGE = (0xA8E0, 0xA8FF)

    @staticmethod
    def is_devanagari_char(char: str) -> bool:
        """Check if a character is in Devanagari Unicode blocks.

        Args:
            char: A single character string

        Returns:
            True if character is Devanagari, False otherwise

        Examples:
            >>> HindiValidator.is_devanagari_char('अ')
            True
            >>> HindiValidator.is_devanagari_char('a')
            False
        """
        if not char or len(char) != 1:
            return False

        code = ord(char)
        return (HindiValidator.DEVANAGARI_RANGE[0] <= code <= HindiValidator.DEVANAGARI_RANGE[1] or
                HindiValidator.DEVANAGARI_EXT_RANGE[0] <= code <= HindiValidator.DEVANAGARI_EXT_RANGE[1])

    @staticmethod
    def calculate_hindi_ratio(text: str, ignore_whitespace: bool = True) -> float:
        """Calculate the ratio of Devanagari characters in text.

        Args:
            text: Input text to analyze
            ignore_whitespace: If True, exclude whitespace from total character count

        Returns:
            Float between 0.0 and 1.0 representing the proportion of Devanagari characters
            Returns 0.0 if text is empty or has no valid characters

        Examples:
            >>> HindiValidator.calculate_hindi_ratio('यह हिंदी है')
            1.0
            >>> HindiValidator.calculate_hindi_ratio('Hello दुनिया')
            0.5
        """
        if not text:
            return 0.0

        hindi_chars = sum(1 for char in text if HindiValidator.is_devanagari_char(char))

        if ignore_whitespace:
            total_chars = sum(1 for char in text if not char.isspace())
        else:
            total_chars = len(text)

        return hindi_chars / total_chars if total_chars > 0 else 0.0

    @staticmethod
    def is_hindi_text(text: str,
                     min_ratio: float = 0.8,
                     min_length: int = 50,
                     ignore_whitespace: bool = True) -> bool:
        """Check if text is primarily in Hindi.

        Args:
            text: Input text to validate
            min_ratio: Minimum ratio of Devanagari characters required (default: 0.8)
            min_length: Minimum text length in characters (default: 50)
            ignore_whitespace: If True, exclude whitespace from character counts

        Returns:
            True if text meets both length and ratio requirements, False otherwise

        Examples:
            >>> HindiValidator.is_hindi_text('यह एक लंबा हिंदी वाक्य है जो न्यूनतम लंबाई की आवश्यकता को पूरा करता है')
            True
            >>> HindiValidator.is_hindi_text('Short')
            False
        """
        if not text or len(text) < min_length:
            return False

        ratio = HindiValidator.calculate_hindi_ratio(text, ignore_whitespace=ignore_whitespace)
        return ratio >= min_ratio

    @staticmethod
    def count_devanagari_chars(text: str) -> int:
        """Count the number of Devanagari characters in text.

        Args:
            text: Input text to analyze

        Returns:
            Integer count of Devanagari characters

        Examples:
            >>> HindiValidator.count_devanagari_chars('Hello दुनिया')
            5
        """
        if not text:
            return 0
        return sum(1 for char in text if HindiValidator.is_devanagari_char(char))

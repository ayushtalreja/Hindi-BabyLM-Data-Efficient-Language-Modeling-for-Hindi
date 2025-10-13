"""
Curriculum Learning Strategies for Hindi Language Models

This module implements various curriculum learning strategies that progressively
increase training difficulty. Strategies are designed for morphologically rich
languages like Hindi.

Strategies Implemented:
1. Morphological Complexity: Start with simple morphology, progress to complex
2. Length-Based: Start with short sentences, progress to longer
3. Frequency-Based: Start with common words, progress to rare
4. Combined: Multiple strategies combined
5. Dynamic: Adapt curriculum based on model performance

Reference:
- Bengio et al. (2009) "Curriculum Learning"
- Platanios et al. (2019) "Competence-based Curriculum Learning"
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class CurriculumStrategy:
    """
    Base class for curriculum learning strategies
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize curriculum strategy

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.difficulty_computed = False

    def compute_difficulty(self, examples: List[Dict]) -> List[float]:
        """
        Compute difficulty scores for examples

        Args:
            examples: List of training examples

        Returns:
            List of difficulty scores (0=easy, 1=hard)
        """
        raise NotImplementedError("Subclasses must implement compute_difficulty")

    def sort_by_difficulty(self, examples: List[Dict]) -> Tuple[List[Dict], List[float]]:
        """
        Sort examples by difficulty

        Args:
            examples: List of training examples

        Returns:
            Tuple of (sorted_examples, difficulty_scores)
        """
        difficulties = self.compute_difficulty(examples)
        sorted_indices = np.argsort(difficulties)

        sorted_examples = [examples[i] for i in sorted_indices]
        sorted_difficulties = [difficulties[i] for i in sorted_indices]

        return sorted_examples, sorted_difficulties

    def get_curriculum_subset(self, examples: List[Dict],
                             difficulty_threshold: float) -> List[Dict]:
        """
        Get subset of examples below difficulty threshold

        Args:
            examples: List of training examples
            difficulty_threshold: Threshold (0-1)

        Returns:
            Filtered subset of examples
        """
        difficulties = self.compute_difficulty(examples)
        filtered = [ex for ex, diff in zip(examples, difficulties)
                   if diff <= difficulty_threshold]
        return filtered


class MorphologicalComplexityCurriculum(CurriculumStrategy):
    """
    Curriculum based on morphological complexity

    Complexity factors for Hindi:
    - Number of morphemes
    - Presence of case markers
    - Verb inflections (tense, aspect, mood)
    - Agreement features (gender, number, person)
    - Compound constructions
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        # Hindi case markers and their complexity weights
        self.case_markers = {
            'ने': 2,  # Ergative
            'को': 2,  # Accusative/Dative
            'से': 2,  # Instrumental/Ablative
            'में': 1,  # Locative
            'पर': 1,  # Locative
            'का': 1, 'की': 1, 'के': 1,  # Genitive
        }

        # Complexity weights
        self.weights = {
            'case_markers': self.config.get('case_weight', 1.0),
            'sentence_length': self.config.get('length_weight', 0.5),
            'verb_forms': self.config.get('verb_weight', 1.5),
            'compound_words': self.config.get('compound_weight', 1.0)
        }

    def compute_difficulty(self, examples: List[Dict]) -> List[float]:
        """
        Compute morphological complexity scores

        Args:
            examples: List of examples with 'text' field

        Returns:
            List of difficulty scores
        """
        difficulties = []

        for example in examples:
            text = example.get('text', '')
            difficulty = self._compute_text_difficulty(text)
            difficulties.append(difficulty)

        # Normalize to [0, 1]
        if difficulties:
            max_diff = max(difficulties)
            min_diff = min(difficulties)
            if max_diff > min_diff:
                difficulties = [(d - min_diff) / (max_diff - min_diff) for d in difficulties]

        return difficulties

    def _compute_text_difficulty(self, text: str) -> float:
        """
        Compute difficulty for a single text

        Args:
            text: Input text

        Returns:
            Difficulty score
        """
        if not text:
            return 0.0

        words = text.split()
        difficulty = 0.0

        # Factor 1: Case markers
        case_score = sum(self.case_markers.get(word, 0) for word in words)
        difficulty += case_score * self.weights['case_markers']

        # Factor 2: Sentence length (normalized)
        length_score = min(len(words) / 20.0, 1.0)  # Cap at 20 words
        difficulty += length_score * self.weights['sentence_length']

        # Factor 3: Complex verb forms (रहा है, चुका है, etc.)
        verb_complexity_markers = ['रहा', 'रही', 'रहे', 'चुका', 'चुकी', 'चुके', 'गया', 'गई', 'गए']
        verb_score = sum(1 for word in words if any(marker in word for marker in verb_complexity_markers))
        difficulty += verb_score * self.weights['verb_forms']

        # Factor 4: Compound words (heuristic: words with multiple Devanagari characters)
        compound_score = sum(1 for word in words if len(word) > 10)
        difficulty += compound_score * self.weights['compound_words']

        return difficulty


class LengthBasedCurriculum(CurriculumStrategy):
    """
    Curriculum based on sentence length

    Start with short sentences, gradually increase length
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.min_length = self.config.get('min_length', 5)
        self.max_length = self.config.get('max_length', 50)

    def compute_difficulty(self, examples: List[Dict]) -> List[float]:
        """
        Compute length-based difficulty scores

        Args:
            examples: List of examples

        Returns:
            List of difficulty scores
        """
        difficulties = []

        for example in examples:
            text = example.get('text', '')
            length = len(text.split())

            # Normalize to [0, 1]
            if self.max_length > self.min_length:
                difficulty = (length - self.min_length) / (self.max_length - self.min_length)
                difficulty = max(0.0, min(1.0, difficulty))
            else:
                difficulty = 0.5

            difficulties.append(difficulty)

        return difficulties


class FrequencyBasedCurriculum(CurriculumStrategy):
    """
    Curriculum based on word frequency

    Start with common words, progress to rare words
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.word_frequencies = None

    def compute_difficulty(self, examples: List[Dict]) -> List[float]:
        """
        Compute frequency-based difficulty scores

        Args:
            examples: List of examples

        Returns:
            List of difficulty scores
        """
        # Build word frequency dictionary if not already done
        if self.word_frequencies is None:
            self._build_frequency_dict(examples)

        difficulties = []

        for example in examples:
            text = example.get('text', '')
            difficulty = self._compute_text_rarity(text)
            difficulties.append(difficulty)

        return difficulties

    def _build_frequency_dict(self, examples: List[Dict]):
        """Build word frequency dictionary from examples"""
        all_words = []
        for example in examples:
            text = example.get('text', '')
            all_words.extend(text.split())

        self.word_frequencies = Counter(all_words)
        total_words = sum(self.word_frequencies.values())

        # Normalize to probabilities
        self.word_frequencies = {
            word: count / total_words
            for word, count in self.word_frequencies.items()
        }

        logger.info(f"Built frequency dictionary with {len(self.word_frequencies)} unique words")

    def _compute_text_rarity(self, text: str) -> float:
        """
        Compute rarity score for text

        Args:
            text: Input text

        Returns:
            Rarity score (higher = rarer)
        """
        if not text or not self.word_frequencies:
            return 0.5

        words = text.split()
        if not words:
            return 0.5

        # Average negative log probability (higher = rarer)
        rarities = []
        for word in words:
            freq = self.word_frequencies.get(word, 1e-6)
            rarity = -np.log(freq + 1e-10)
            rarities.append(rarity)

        avg_rarity = np.mean(rarities)

        # Normalize to [0, 1] using sigmoid
        difficulty = 1 / (1 + np.exp(-avg_rarity + 5))

        return difficulty


class CombinedCurriculum(CurriculumStrategy):
    """
    Combined curriculum using multiple strategies

    Combines morphological complexity, length, and frequency
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        # Initialize sub-strategies
        self.morphological_strategy = MorphologicalComplexityCurriculum(config)
        self.length_strategy = LengthBasedCurriculum(config)
        self.frequency_strategy = FrequencyBasedCurriculum(config)

        # Combination weights
        self.strategy_weights = {
            'morphological': self.config.get('morphological_weight', 0.5),
            'length': self.config.get('length_weight', 0.3),
            'frequency': self.config.get('frequency_weight', 0.2)
        }

        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v / total_weight for k, v in self.strategy_weights.items()}

        logger.info(f"Combined curriculum with weights: {self.strategy_weights}")

    def compute_difficulty(self, examples: List[Dict]) -> List[float]:
        """
        Compute combined difficulty scores

        Args:
            examples: List of examples

        Returns:
            List of difficulty scores
        """
        # Get difficulty from each strategy
        morph_difficulties = self.morphological_strategy.compute_difficulty(examples)
        length_difficulties = self.length_strategy.compute_difficulty(examples)
        freq_difficulties = self.frequency_strategy.compute_difficulty(examples)

        # Combine with weights
        combined_difficulties = []
        for morph, length, freq in zip(morph_difficulties, length_difficulties, freq_difficulties):
            combined = (
                morph * self.strategy_weights['morphological'] +
                length * self.strategy_weights['length'] +
                freq * self.strategy_weights['frequency']
            )
            combined_difficulties.append(combined)

        return combined_difficulties


class DynamicCurriculum(CurriculumStrategy):
    """
    Dynamic curriculum that adapts based on model performance

    Adjusts difficulty based on validation loss or accuracy
    """

    def __init__(self, base_strategy: CurriculumStrategy, config: Optional[Dict] = None):
        super().__init__(config)
        self.base_strategy = base_strategy

        # Performance tracking
        self.performance_history = []
        self.difficulty_threshold = 0.3  # Start with easy examples

        # Configuration
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)
        self.min_threshold = self.config.get('min_threshold', 0.1)
        self.max_threshold = self.config.get('max_threshold', 1.0)
        self.performance_window = self.config.get('performance_window', 5)

    def compute_difficulty(self, examples: List[Dict]) -> List[float]:
        """
        Use base strategy for difficulty computation

        Args:
            examples: List of examples

        Returns:
            List of difficulty scores
        """
        return self.base_strategy.compute_difficulty(examples)

    def update_difficulty_threshold(self, performance_metric: float):
        """
        Update difficulty threshold based on performance

        Args:
            performance_metric: Recent validation metric (lower is better for loss)
        """
        self.performance_history.append(performance_metric)

        # Keep only recent history
        if len(self.performance_history) > self.performance_window:
            self.performance_history = self.performance_history[-self.performance_window:]

        # Check if performance is improving
        if len(self.performance_history) >= 2:
            recent_avg = np.mean(self.performance_history[-3:])
            older_avg = np.mean(self.performance_history[-self.performance_window:-3]) if len(self.performance_history) > 3 else self.performance_history[0]

            # If improving, increase difficulty
            if recent_avg < older_avg:
                self.difficulty_threshold += self.adaptation_rate
                logger.info(f"Performance improving, increasing difficulty threshold to {self.difficulty_threshold:.3f}")
            else:
                # If not improving, decrease difficulty slightly
                self.difficulty_threshold -= self.adaptation_rate * 0.5
                logger.info(f"Performance plateauing, decreasing difficulty threshold to {self.difficulty_threshold:.3f}")

        # Clamp to valid range
        self.difficulty_threshold = max(self.min_threshold, min(self.max_threshold, self.difficulty_threshold))

    def get_current_subset(self, examples: List[Dict]) -> List[Dict]:
        """
        Get current curriculum subset based on adaptive threshold

        Args:
            examples: Full dataset

        Returns:
            Filtered subset
        """
        return self.get_curriculum_subset(examples, self.difficulty_threshold)


class AntiCurriculum(CurriculumStrategy):
    """
    Anti-curriculum: Start with hard examples first

    Useful for comparison and ablation studies
    """

    def __init__(self, base_strategy: CurriculumStrategy):
        super().__init__()
        self.base_strategy = base_strategy

    def compute_difficulty(self, examples: List[Dict]) -> List[float]:
        """
        Invert base strategy difficulties

        Args:
            examples: List of examples

        Returns:
            Inverted difficulty scores
        """
        base_difficulties = self.base_strategy.compute_difficulty(examples)
        # Invert: easy becomes hard, hard becomes easy
        inverted = [1.0 - d for d in base_difficulties]
        return inverted


# Factory function
def create_curriculum_strategy(strategy_type: str, config: Optional[Dict] = None) -> CurriculumStrategy:
    """
    Factory function to create curriculum strategies

    Args:
        strategy_type: Type of strategy ('morphological', 'length', 'frequency', 'combined', 'dynamic')
        config: Configuration dictionary

    Returns:
        CurriculumStrategy instance
    """
    strategies = {
        'morphological': MorphologicalComplexityCurriculum,
        'length': LengthBasedCurriculum,
        'frequency': FrequencyBasedCurriculum,
        'combined': CombinedCurriculum,
    }

    if strategy_type in strategies:
        return strategies[strategy_type](config)
    elif strategy_type == 'dynamic':
        # Dynamic requires a base strategy
        base_type = config.get('base_strategy', 'combined') if config else 'combined'
        base_strategy = create_curriculum_strategy(base_type, config)
        return DynamicCurriculum(base_strategy, config)
    elif strategy_type == 'anti':
        # Anti-curriculum for ablation
        base_type = config.get('base_strategy', 'combined') if config else 'combined'
        base_strategy = create_curriculum_strategy(base_type, config)
        return AntiCurriculum(base_strategy)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

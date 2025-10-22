from typing import Dict, List, Tuple
import random

class DataMixer:
    def __init__(self, max_words: int = 10_000_000):
        self.max_words = max_words
        self.word_count = 0
    
    def mix_corpora(self, corpora: Dict[str, List[str]],
                   ratios: Dict[str, float]) -> List[Tuple[str, str]]:
        """Mix different corpora according to specified ratios"""
        # corpora = {"indiccorp": [...], "wikipedia": [...], "stories": [...]}
        # ratios = {"indiccorp": 0.7, "wikipedia": 0.2, "stories": 0.1}

        # Validate ratios sum to 1.0
        total_ratio = sum(ratios.values())
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        mixed_data = []
        remaining_words = self.max_words

        # Calculate target words per corpus
        target_words = {name: int(self.max_words * ratio)
                       for name, ratio in ratios.items()}

        for corpus_name, texts in corpora.items():
            if corpus_name not in ratios:
                continue

            words_needed = target_words[corpus_name]
            corpus_words = 0

            # Shuffle for randomness
            shuffled_texts = texts.copy()
            random.shuffle(shuffled_texts)

            for text in shuffled_texts:
                # Count words (whitespace tokenization)
                text_words = len(text.split())

                if corpus_words + text_words <= words_needed:
                    mixed_data.append((corpus_name, text))
                    corpus_words += text_words
                    self.word_count += text_words
                else:
                    # Take partial text to meet exact ratio
                    remaining = words_needed - corpus_words
                    words = text.split()
                    if remaining > 0 and len(words) > 0:
                        partial_text = ' '.join(words[:remaining])
                        mixed_data.append((corpus_name, partial_text))
                        self.word_count += remaining
                    break

        # Shuffle the mixed data for training
        random.shuffle(mixed_data)
        return mixed_data
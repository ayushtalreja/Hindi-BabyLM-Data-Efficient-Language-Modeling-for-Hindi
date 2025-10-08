from typing import Dict, List, Tuple
import random

class DataMixer:
    def __init__(self, max_tokens: int = 10_000_000):
        self.max_tokens = max_tokens
        self.token_count = 0
    
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
        remaining_tokens = self.max_tokens

        # Calculate target tokens per corpus
        target_tokens = {name: int(self.max_tokens * ratio)
                        for name, ratio in ratios.items()}

        for corpus_name, texts in corpora.items():
            if corpus_name not in ratios:
                continue

            tokens_needed = target_tokens[corpus_name]
            corpus_tokens = 0

            # Shuffle for randomness
            shuffled_texts = texts.copy()
            random.shuffle(shuffled_texts)

            for text in shuffled_texts:
                # Approximate token count (whitespace tokenization)
                text_tokens = len(text.split())

                if corpus_tokens + text_tokens <= tokens_needed:
                    mixed_data.append((corpus_name, text))
                    corpus_tokens += text_tokens
                    self.token_count += text_tokens
                else:
                    # Take partial text to meet exact ratio
                    remaining = tokens_needed - corpus_tokens
                    words = text.split()
                    if remaining > 0 and len(words) > 0:
                        partial_text = ' '.join(words[:remaining])
                        mixed_data.append((corpus_name, partial_text))
                        self.token_count += remaining
                    break

        # Shuffle the mixed data for training
        random.shuffle(mixed_data)
        return mixed_data
    
    def create_curriculum_splits(self, mixed_data: List[Tuple[str, str]],
                               strategy: str = "morphological") -> List[List[str]]:
        """Create curriculum learning splits based on complexity"""
        texts = [text for _, text in mixed_data]

        if strategy == "length":
            # Sort by text length (simple to complex)
            sorted_texts = sorted(texts, key=len)

        elif strategy == "morphological":
            # Sort by morphological complexity
            def morphological_score(text):
                """Calculate morphological complexity score"""
                # Count inflectional markers
                markers = ['ों', 'ें', 'यां', 'ने', 'को', 'में', 'पर', 'से',
                          'का', 'की', 'के', 'ता', 'ती', 'ते']
                score = sum(text.count(marker) for marker in markers)
                # Normalize by text length
                return score / len(text.split()) if len(text.split()) > 0 else 0

            sorted_texts = sorted(texts, key=morphological_score)

        elif strategy == "word_length":
            # Sort by average word length
            def avg_word_len(text):
                words = text.split()
                return sum(len(w) for w in words) / len(words) if words else 0

            sorted_texts = sorted(texts, key=avg_word_len)

        else:
            # Default: random order
            sorted_texts = texts.copy()
            random.shuffle(sorted_texts)

        # Split into curriculum stages (3 stages: easy, medium, hard)
        n = len(sorted_texts)
        splits = [
            sorted_texts[:n//3],           # Stage 1: Easy
            sorted_texts[n//3:2*n//3],     # Stage 2: Medium
            sorted_texts[2*n//3:]          # Stage 3: Hard
        ]

        return splits
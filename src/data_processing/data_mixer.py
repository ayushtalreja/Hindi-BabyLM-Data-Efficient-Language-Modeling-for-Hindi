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
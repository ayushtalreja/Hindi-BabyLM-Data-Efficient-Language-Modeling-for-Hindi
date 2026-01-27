"""
Hybrid character-bigram tokenizer with morphological awareness for Hindi.

This tokenizer extends pure character-level tokenization by adding frequent
character bigrams as atomic tokens, with special attention to morphologically
meaningful patterns (suffixes, roots, etc.).
"""

import unicodedata
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
import pickle
import os
from .character_tokenizer import DevanagariCharacterTokenizer


class CharacterBigramTokenizer(DevanagariCharacterTokenizer):
    """Hybrid character-bigram tokenizer with morphological awareness"""

    def __init__(
        self,
        target_bigrams: int = 800,
        min_frequency: int = 100,
        morphological_aware: bool = True,
        preserve_grapheme_clusters: bool = True
    ):
        """
        Initialize hybrid character-bigram tokenizer.

        Args:
            target_bigrams: Target number of bigrams to extract
            min_frequency: Minimum frequency threshold for bigrams
            morphological_aware: Apply morphological boosting to bigram selection
            preserve_grapheme_clusters: Preserve grapheme clusters in base tokenization
        """
        # Initialize parent character tokenizer
        super().__init__(preserve_grapheme_clusters)

        self.target_bigrams = target_bigrams
        self.min_frequency = min_frequency
        self.morphological_aware = morphological_aware
        self.bigrams = set()

        # Known Hindi morphological patterns for boosting
        self.morphological_patterns = self._load_morphological_patterns()

    def _load_morphological_patterns(self) -> Set[str]:
        """
        Load known Hindi morphological bigrams for boosting.

        Returns:
            Set of morphologically significant bigrams
        """
        patterns = set()

        # Common nominal suffixes (2-char patterns)
        nominal_suffixes = [
            "ों",  # Plural marker
            "ने",  # Ergative case
            "को",  # Dative/accusative
            "से",  # Instrumental/ablative
            "में",  # Locative
            "पर",  # Locative
            "का",  # Genitive masculine singular
            "के",  # Genitive masculine plural
            "की",  # Genitive feminine
        ]
        patterns.update(nominal_suffixes)

        # Common verbal morphology (2-char patterns)
        verbal_morphology = [
            "ता",  # Habitual aspect (masculine)
            "ती",  # Habitual aspect (feminine)
            "ते",  # Habitual aspect (plural)
            "ेग",  # Future tense (part of ेगा, ेगी, ेगे)
            "रह",  # Progressive aspect (part of रहा, रही, रहे)
            "गा",  # Future masculine
            "गी",  # Future feminine
            "गे",  # Future plural
        ]
        patterns.update(verbal_morphology)

        # Derivational morphology
        derivational = [
            "वा",  # Causative marker (part of वाला, वाली)
            "पन",  # Abstract noun suffix (मीठापन, बचपन)
        ]
        patterns.update(derivational)

        # Common 2-character verb roots
        verb_roots = [
            "पढ़",  # Read
            "लिख",  # Write
            "दे",   # Give
            "ले",   # Take
            "जा",   # Go
            "आ",    # Come
            "कर",   # Do
        ]
        patterns.update(verb_roots)

        # Common function words (2-char)
        function_words = [
            "है",   # Is/are
            "हैं",  # Are (plural)
            "था",   # Was (masculine)
            "थी",   # Was (feminine)
            "थे",   # Were
            "और",   # And
            "या",   # Or
        ]
        patterns.update(function_words)

        return patterns

    def train_bigrams(self, training_texts: List[str]):
        """
        Extract frequent bigrams from training corpus.

        Args:
            training_texts: List of training text strings
        """
        print(f"Extracting bigrams from {len(training_texts)} training texts...")

        # Count bigrams
        bigram_counts = Counter()
        total_chars = 0

        for text in training_texts:
            # Normalize to NFC
            text = unicodedata.normalize('NFC', text)

            # Extract character sequence (not grapheme clusters for bigram counting)
            chars = list(text)
            total_chars += len(chars)

            # Count bigrams
            for i in range(len(chars) - 1):
                char1, char2 = chars[i], chars[i + 1]

                # Skip bigrams crossing word boundaries (space-adjacent)
                if char1 == ' ' or char2 == ' ':
                    continue

                # Skip bigrams with newlines or tabs
                if char1 in ['\n', '\t'] or char2 in ['\n', '\t']:
                    continue

                bigram = char1 + char2
                bigram_counts[bigram] += 1

        print(f"Found {len(bigram_counts)} unique bigrams from {total_chars:,} characters")

        # Apply morphological bonus
        if self.morphological_aware:
            boosted_counts = {}
            for bigram, count in bigram_counts.items():
                if bigram in self.morphological_patterns:
                    # Boost morphologically significant bigrams by 50%
                    boosted_counts[bigram] = int(count * 1.5)
                else:
                    boosted_counts[bigram] = count
            bigram_counts = Counter(boosted_counts)
            print(f"Applied morphological boosting to {len(self.morphological_patterns)} patterns")

        # Select top bigrams above threshold
        selected_bigrams = []

        for bigram, count in bigram_counts.most_common():
            # Stop if we've reached target
            if len(selected_bigrams) >= self.target_bigrams:
                break

            # Check frequency threshold
            if count < self.min_frequency:
                break

            # Ensure both characters are in base vocabulary
            if all(c in self.char_to_id for c in bigram):
                selected_bigrams.append(bigram)

        self.bigrams = set(selected_bigrams)

        # Update vocabulary with bigrams
        self._update_vocabulary_with_bigrams()

        print(f"Selected {len(self.bigrams)} bigrams (min freq: {self.min_frequency})")
        print(f"Final vocabulary size: {self.vocab_size}")

        # Print some example bigrams
        if selected_bigrams:
            print(f"\nTop 20 selected bigrams:")
            for i, (bigram, count) in enumerate(bigram_counts.most_common(20)):
                if bigram in self.bigrams:
                    morph_marker = " *" if bigram in self.morphological_patterns else ""
                    print(f"  {bigram}: {count:,}{morph_marker}")

    def _update_vocabulary_with_bigrams(self):
        """Add bigrams to vocabulary and rebuild mappings"""
        # Add bigrams to vocab (sorted for consistency)
        for bigram in sorted(self.bigrams):
            if bigram not in self.vocab:
                self.vocab.append(bigram)

        # Rebuild mappings
        self.vocab_size = len(self.vocab)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}

        # Update special token IDs
        self.pad_token_id = self.char_to_id.get("<pad>", 0)
        self.unk_token_id = self.char_to_id.get("<unk>", 1)
        self.bos_token_id = self.char_to_id.get("<s>", 2)
        self.eos_token_id = self.char_to_id.get("</s>", 3)
        self.mask_token_id = self.char_to_id.get("<mask>", 4)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize with greedy bigram matching.

        Args:
            text: Input text

        Returns:
            List of tokens (mix of bigrams and characters)
        """
        # Normalize to NFC
        text = unicodedata.normalize('NFC', text)

        tokens = []
        i = 0

        while i < len(text):
            matched = False

            # Try bigram match first (greedy longest match)
            if i + 2 <= len(text):
                bigram = text[i:i+2]
                if bigram in self.bigrams:
                    tokens.append(bigram)
                    i += 2
                    matched = True

            # Fall back to character (with grapheme cluster handling)
            if not matched:
                # Extract single character or grapheme cluster
                char = text[i]
                cluster = [char]

                if self.preserve_grapheme_clusters:
                    # Check for combining marks
                    j = i + 1
                    while j < len(text) and unicodedata.category(text[j]).startswith('M'):
                        cluster.append(text[j])
                        j += 1
                    tokens.append(''.join(cluster))
                    i = j
                else:
                    tokens.append(char)
                    i += 1

        return tokens

    def save(self, path: str):
        """
        Save tokenizer with bigrams.

        Args:
            path: Directory path to save tokenizer
        """
        os.makedirs(path, exist_ok=True)
        vocab_path = os.path.join(path, 'char_bigram_vocab.pkl')

        data = {
            'vocab': self.vocab,
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char,
            'bigrams': self.bigrams,
            'target_bigrams': self.target_bigrams,
            'min_frequency': self.min_frequency,
            'morphological_aware': self.morphological_aware,
            'preserve_grapheme_clusters': self.preserve_grapheme_clusters,
            'vocab_size': self.vocab_size,
            'morphological_patterns': self.morphological_patterns
        }

        with open(vocab_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Character-bigram tokenizer saved to {vocab_path}")

    @staticmethod
    def load(path: str) -> 'CharacterBigramTokenizer':
        """
        Load saved tokenizer.

        Args:
            path: Directory path where tokenizer is saved

        Returns:
            Loaded tokenizer instance
        """
        vocab_path = os.path.join(path, 'char_bigram_vocab.pkl')

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Tokenizer not found at {vocab_path}")

        with open(vocab_path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = CharacterBigramTokenizer(
            target_bigrams=data['target_bigrams'],
            min_frequency=data['min_frequency'],
            morphological_aware=data['morphological_aware'],
            preserve_grapheme_clusters=data['preserve_grapheme_clusters']
        )

        tokenizer.vocab = data['vocab']
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = data['id_to_char']
        tokenizer.bigrams = data['bigrams']
        tokenizer.vocab_size = data['vocab_size']

        # Load morphological patterns if available
        if 'morphological_patterns' in data:
            tokenizer.morphological_patterns = data['morphological_patterns']

        # Update special token IDs (including mask_token_id fix)
        tokenizer.pad_token_id = tokenizer.char_to_id.get("<pad>", 0)
        tokenizer.unk_token_id = tokenizer.char_to_id.get("<unk>", 1)
        tokenizer.bos_token_id = tokenizer.char_to_id.get("<s>", 2)
        tokenizer.eos_token_id = tokenizer.char_to_id.get("</s>", 3)
        tokenizer.mask_token_id = tokenizer.char_to_id.get("<mask>", 4)

        # Ensure BERT compatibility after loading
        tokenizer._ensure_bert_compatibility()

        print(f"Character-bigram tokenizer loaded from {vocab_path}")
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
        print(f"  Bigrams: {len(tokenizer.bigrams)}")

        return tokenizer

    def get_bigram_stats(self) -> Dict[str, any]:
        """
        Get statistics about bigrams.

        Returns:
            Dictionary with bigram statistics
        """
        morphological_bigrams = [b for b in self.bigrams if b in self.morphological_patterns]

        return {
            'total_bigrams': len(self.bigrams),
            'morphological_bigrams': len(morphological_bigrams),
            'total_vocab_size': self.vocab_size,
            'character_vocab_size': self.vocab_size - len(self.bigrams),
            'bigram_ratio': len(self.bigrams) / self.vocab_size if self.vocab_size > 0 else 0
        }

    def __repr__(self) -> str:
        return (f"CharacterBigramTokenizer(vocab_size={self.vocab_size}, "
                f"bigrams={len(self.bigrams)}, "
                f"morphological_aware={self.morphological_aware})")

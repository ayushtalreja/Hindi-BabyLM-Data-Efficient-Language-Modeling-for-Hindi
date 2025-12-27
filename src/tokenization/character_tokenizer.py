"""
Pure character-level tokenizer with Devanagari grapheme cluster preservation.

This tokenizer implements Unicode-Aware Character Tokenization (UACT) for Hindi,
treating each character as a token while preserving visual grapheme clusters
(combining marks, matras, conjuncts).
"""

import unicodedata
from typing import List, Dict, Optional, Union
import pickle
import os
import torch


class DevanagariCharacterTokenizer:
    """Pure character-level tokenizer with Devanagari grapheme awareness"""

    def __init__(self, preserve_grapheme_clusters: bool = True):
        """
        Initialize character-level tokenizer.

        Args:
            preserve_grapheme_clusters: If True, preserves grapheme clusters
                                       (e.g., क + ि = कि as single unit)
        """
        self.preserve_grapheme_clusters = preserve_grapheme_clusters
        self.vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}

        # Special token IDs for compatibility
        self.pad_token_id = self.char_to_id.get("<pad>", 0)
        self.unk_token_id = self.char_to_id.get("<unk>", 1)
        self.bos_token_id = self.char_to_id.get("<s>", 2)
        self.eos_token_id = self.char_to_id.get("</s>", 3)

        # Special token strings (for HuggingFace compatibility)
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.mask_token = "<mask>"

    def _build_vocabulary(self) -> List[str]:
        """Build character vocabulary for Devanagari + common characters"""
        vocab = []

        # Special tokens (must be first for consistent indexing)
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>", "<sep>", "<cls>"]
        vocab.extend(special_tokens)

        # Space and common whitespace
        vocab.extend([" ", "\n", "\t"])

        # Devanagari Unicode block (U+0900 to U+097F)
        # This includes vowels, consonants, matras, numerals, and special marks
        for code in range(0x0900, 0x0980):
            char = chr(code)
            # Only add valid characters (skip unassigned codepoints)
            if unicodedata.category(char) != 'Cn':  # Cn = unassigned
                vocab.append(char)

        # ASCII digits (0-9)
        vocab.extend([str(i) for i in range(10)])

        # ASCII lowercase letters (a-z)
        vocab.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])

        # ASCII uppercase letters (A-Z)
        vocab.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])

        # Common punctuation and symbols
        punctuation = ".,!?;:'\"-()[]{}/@#$%&*+=<>।॥"
        vocab.extend(list(punctuation))

        return vocab

    def _extract_grapheme_clusters(self, text: str) -> List[str]:
        """
        Extract grapheme clusters preserving Devanagari conjuncts and combining marks.

        A grapheme cluster is a base character followed by any combining marks
        (matras, nukta, virama, etc.).

        Args:
            text: Input text

        Returns:
            List of grapheme clusters (or individual characters if preserve_grapheme_clusters=False)
        """
        if not self.preserve_grapheme_clusters:
            return list(text)

        clusters = []
        i = 0
        while i < len(text):
            char = text[i]
            cluster = [char]

            # Check for combining marks following the base character
            # Unicode category M* = marks (Mn = nonspacing, Mc = spacing combining, Me = enclosing)
            j = i + 1
            while j < len(text) and unicodedata.category(text[j]).startswith('M'):
                cluster.append(text[j])
                j += 1

            clusters.append(''.join(cluster))
            i = j

        return clusters

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into characters/grapheme clusters.

        Args:
            text: Input text

        Returns:
            List of character tokens
        """
        # Normalize to NFC (canonical composition)
        # This ensures consistent Unicode representation
        text = unicodedata.normalize('NFC', text)

        # Extract grapheme clusters
        tokens = self._extract_grapheme_clusters(text)

        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: If True, add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        ids = []

        # Add BOS token if requested
        if add_special_tokens:
            ids.append(self.bos_token_id)

        # Encode each token
        for token in tokens:
            if token in self.char_to_id:
                ids.append(self.char_to_id[token])
            else:
                # Unknown token - use <unk>
                ids.append(self.unk_token_id)

        # Add EOS token if requested
        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: If True, skip special tokens in output

        Returns:
            Decoded text string
        """
        chars = []
        for id in ids:
            if id in self.id_to_char:
                char = self.id_to_char[id]
                # Skip special tokens if requested
                if skip_special_tokens and char.startswith("<") and char.endswith(">"):
                    continue
                chars.append(char)
            # Skip unknown IDs silently

        return ''.join(chars)

    def __call__(self,
                 text: Union[str, List[str]],
                 return_tensors: Optional[str] = None,
                 padding: Union[bool, str] = False,
                 truncation: bool = False,
                 max_length: Optional[int] = None,
                 add_special_tokens: bool = False,
                 **kwargs) -> Dict[str, Union[List, torch.Tensor]]:
        """
        HuggingFace-style tokenizer interface for compatibility.

        Args:
            text: Input text or list of texts
            return_tensors: 'pt' for PyTorch tensors, None for lists
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Handle single string vs list of strings
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # Encode all texts
        encoded_batch = []
        for t in texts:
            ids = self.encode(t, add_special_tokens=add_special_tokens)

            # Truncate if needed
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]

            encoded_batch.append(ids)

        # Determine max length for padding
        if padding:
            if max_length:
                pad_to_length = max_length
            else:
                pad_to_length = max(len(ids) for ids in encoded_batch)
        else:
            pad_to_length = None

        # Pad sequences and create attention masks
        input_ids = []
        attention_mask = []

        for ids in encoded_batch:
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1] * len(ids)

            # Pad if needed
            if pad_to_length and len(ids) < pad_to_length:
                padding_length = pad_to_length - len(ids)
                ids = ids + [self.pad_token_id] * padding_length
                mask = mask + [0] * padding_length

            input_ids.append(ids)
            attention_mask.append(mask)

        # Convert to tensors if requested
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        return result

    def save(self, path: str):
        """
        Save tokenizer vocabulary and configuration.

        Args:
            path: Directory path to save tokenizer
        """
        os.makedirs(path, exist_ok=True)
        vocab_path = os.path.join(path, 'char_vocab.pkl')

        data = {
            'vocab': self.vocab,
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char,
            'preserve_grapheme_clusters': self.preserve_grapheme_clusters,
            'vocab_size': self.vocab_size
        }

        with open(vocab_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Character tokenizer saved to {vocab_path}")

    @staticmethod
    def load(path: str) -> 'DevanagariCharacterTokenizer':
        """
        Load saved tokenizer.

        Args:
            path: Directory path where tokenizer is saved

        Returns:
            Loaded tokenizer instance
        """
        vocab_path = os.path.join(path, 'char_vocab.pkl')

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Tokenizer not found at {vocab_path}")

        with open(vocab_path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = DevanagariCharacterTokenizer(
            preserve_grapheme_clusters=data['preserve_grapheme_clusters']
        )
        tokenizer.vocab = data['vocab']
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = data['id_to_char']
        tokenizer.vocab_size = data['vocab_size']

        # Update special token IDs
        tokenizer.pad_token_id = tokenizer.char_to_id.get("<pad>", 0)
        tokenizer.unk_token_id = tokenizer.char_to_id.get("<unk>", 1)
        tokenizer.bos_token_id = tokenizer.char_to_id.get("<s>", 2)
        tokenizer.eos_token_id = tokenizer.char_to_id.get("</s>", 3)

        print(f"Character tokenizer loaded from {vocab_path}")
        return tokenizer

    def get_vocab(self) -> Dict[str, int]:
        """
        Get vocabulary as dictionary.

        Returns:
            Dictionary mapping tokens to IDs
        """
        return self.char_to_id.copy()

    def __len__(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"DevanagariCharacterTokenizer(vocab_size={self.vocab_size}, preserve_grapheme_clusters={self.preserve_grapheme_clusters})"

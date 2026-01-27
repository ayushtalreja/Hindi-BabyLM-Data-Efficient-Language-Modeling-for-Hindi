"""
Abstract base class for all tokenizers in the Hindi BabyLM project.

This module provides shared functionality for tokenizers including:
- Padding (pad method for DataCollator compatibility)
- Special tokens mask generation
- BERT compatibility helpers (cls_token, sep_token aliasing)
- Token conversion utilities
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Set
import torch


class BaseTokenizer(ABC):
    """Abstract base class providing shared tokenizer functionality."""

    # Default special token strings (can be overridden by subclasses)
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    mask_token: str = "<mask>"

    # BERT-style tokens (set by _ensure_bert_compatibility)
    cls_token: Optional[str] = None
    sep_token: Optional[str] = None

    # Token IDs (must be set by subclass)
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    mask_token_id: int = 4
    cls_token_id: Optional[int] = None
    sep_token_id: Optional[int] = None

    # Vocabulary size (must be set by subclass)
    vocab_size: int = 0

    # -------------------------------------------------------------------------
    # Abstract methods that subclasses must implement
    # -------------------------------------------------------------------------

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into token strings.

        Args:
            text: Input text

        Returns:
            List of token strings
        """
        pass

    @abstractmethod
    def _get_special_token_ids(self) -> Set[int]:
        """
        Return the set of all special token IDs.

        Returns:
            Set of special token IDs (pad, unk, bos, eos, mask, cls, sep, etc.)
        """
        pass

    # -------------------------------------------------------------------------
    # Shared implementations
    # -------------------------------------------------------------------------

    def _ensure_bert_compatibility(self):
        """
        Set up BERT-style cls_token and sep_token by aliasing to bos_token and eos_token.

        Call this at the end of subclass __init__ for tokenizers that use GPT-style
        tokens (<s>, </s>) but need BERT-style (cls, sep) compatibility.
        """
        # Map bos_token -> cls_token, eos_token -> sep_token
        if self.cls_token is None:
            self.cls_token = self.bos_token
        if self.sep_token is None:
            self.sep_token = self.eos_token
        if self.cls_token_id is None:
            self.cls_token_id = self.bos_token_id
        if self.sep_token_id is None:
            self.sep_token_id = self.eos_token_id

    def pad(
        self,
        encoded_inputs,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Pad a batch of encoded inputs (for DataCollator compatibility).

        Args:
            encoded_inputs: List of dicts with 'input_ids' (and optionally 'attention_mask')
            padding: Padding strategy
            max_length: Maximum length to pad to
            pad_to_multiple_of: Pad to a multiple of this value
            return_attention_mask: Whether to return attention masks
            return_tensors: 'pt' for PyTorch tensors

        Returns:
            Dictionary with padded 'input_ids' and 'attention_mask'
        """
        # Extract input_ids from the batch
        if isinstance(encoded_inputs, dict):
            # Single encoding
            batch_input_ids = [encoded_inputs['input_ids']]
        else:
            # List of encodings
            batch_input_ids = []
            for item in encoded_inputs:
                if isinstance(item, dict):
                    ids = item.get('input_ids', item)
                else:
                    ids = item
                # Convert tensor to list if needed
                if hasattr(ids, 'tolist'):
                    ids = ids.tolist()
                batch_input_ids.append(ids)

        # Calculate target length
        if max_length is not None:
            target_length = max_length
        else:
            target_length = max(len(ids) for ids in batch_input_ids)

        # Apply pad_to_multiple_of
        if pad_to_multiple_of is not None and target_length % pad_to_multiple_of != 0:
            target_length = ((target_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # Pad all sequences
        padded_input_ids = []
        attention_masks = []
        for input_ids in batch_input_ids:
            # Create attention mask
            attention_mask = [1] * len(input_ids)

            # Pad to target length
            padding_length = target_length - len(input_ids)
            if padding_length > 0:
                input_ids = list(input_ids) + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            padded_input_ids.append(input_ids)
            attention_masks.append(attention_mask)

        # Build result
        result = {'input_ids': padded_input_ids}
        if return_attention_mask:
            result['attention_mask'] = attention_masks

        # Convert to tensors if requested
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.long)
            if return_attention_mask:
                result['attention_mask'] = torch.tensor(result['attention_mask'], dtype=torch.long)

        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Return a mask identifying special tokens.

        Args:
            token_ids_0: List of token IDs
            token_ids_1: Optional second list of token IDs (for sequence pairs)
            already_has_special_tokens: Whether the sequence already has special tokens

        Returns:
            List of 0s and 1s where 1 indicates a special token position
        """
        special_token_ids = self._get_special_token_ids()
        # Remove None values (in case some tokens weren't found)
        special_token_ids = {tid for tid in special_token_ids if tid is not None}

        if already_has_special_tokens:
            # Mark positions of special tokens in the sequence
            return [1 if token_id in special_token_ids else 0 for token_id in token_ids_0]

        # If special tokens haven't been added yet, return all zeros
        if token_ids_1 is not None:
            return [0] * (len(token_ids_0) + len(token_ids_1))
        return [0] * len(token_ids_0)

    def add_special_tokens(self, special_tokens_dict: Dict[str, str]) -> int:
        """
        Add special tokens to the tokenizer vocabulary.

        This method is required for HuggingFace DataCollator compatibility.
        For our tokenizers, special tokens are already defined, so this is mostly a no-op.

        Args:
            special_tokens_dict: Dictionary mapping special token names to their string values
                                 e.g., {'pad_token': '<pad>', 'mask_token': '<mask>'}

        Returns:
            Number of tokens added (0 if all tokens already exist)
        """
        # Our tokenizers have pre-defined special tokens, so we just verify they exist
        # This satisfies the DataCollator interface requirement
        added = 0
        for token_name, token_value in special_tokens_dict.items():
            if hasattr(self, token_name):
                current_value = getattr(self, token_name)
                if current_value is None:
                    setattr(self, token_name, token_value)
                    added += 1
        return added

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Convert token string(s) to their corresponding IDs.

        Required for MultiBLiMP evaluator compatibility.

        Args:
            tokens: Single token string or list of token strings

        Returns:
            Single token ID or list of token IDs
        """
        if isinstance(tokens, str):
            # Single token
            return self._convert_token_to_id(tokens)
        else:
            # List of tokens
            return [self._convert_token_to_id(token) for token in tokens]

    def _convert_token_to_id(self, token: str) -> int:
        """
        Convert a single token to its ID.

        Args:
            token: Token string

        Returns:
            Token ID (returns unk_token_id if token not found)
        """
        # Check for special tokens first
        special_token_map = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.mask_token: self.mask_token_id,
        }
        if self.cls_token is not None:
            special_token_map[self.cls_token] = self.cls_token_id
        if self.sep_token is not None:
            special_token_map[self.sep_token] = self.sep_token_id

        if token in special_token_map:
            return special_token_map[token]

        # Subclass should override this for actual vocabulary lookup
        # Default behavior: return unk_token_id
        return self.unk_token_id

    def get_vocab(self) -> Dict[str, int]:
        """
        Get vocabulary as dictionary.

        Subclasses should override this to return their actual vocabulary.

        Returns:
            Dictionary mapping tokens to IDs
        """
        # Default implementation returns special tokens only
        vocab = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.mask_token: self.mask_token_id,
        }
        if self.cls_token is not None and self.cls_token not in vocab:
            vocab[self.cls_token] = self.cls_token_id
        if self.sep_token is not None and self.sep_token not in vocab:
            vocab[self.sep_token] = self.sep_token_id
        return vocab

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

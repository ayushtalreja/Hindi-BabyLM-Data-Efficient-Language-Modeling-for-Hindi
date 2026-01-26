import os
import pickle
from typing import List, Union, Optional
from transformers import AutoTokenizer
import sentencepiece as spm
import torch

from .sentencepiece_tokenizer import HindiSentencePieceTokenizer
from .character_tokenizer import DevanagariCharacterTokenizer
from .character_bigram_tokenizer import CharacterBigramTokenizer


class TokenizerFactory:
    """Factory class for creating and managing different tokenizers"""

    def __init__(self, config):
        self.config = config
        self.tokenizer_type = config.tokenizer_type
        self.vocab_size = config.vocab_size
        self.tokenizer_dir = config.__dict__.get('tokenizer_dir', 'tokenizers')

        # Create directories
        os.makedirs(self.tokenizer_dir, exist_ok=True)

    def create_tokenizer(self, training_texts: List[str]):
        """Create and train a tokenizer based on config"""
        print(f"\nCreating {self.tokenizer_type} tokenizer with vocab size {self.vocab_size}...")

        if self.tokenizer_type == "sentencepiece":
            return self._create_sentencepiece_tokenizer(training_texts)
        elif self.tokenizer_type == "wordpiece":
            return self._create_wordpiece_tokenizer(training_texts)
        elif self.tokenizer_type == "bpe":
            return self._create_bpe_tokenizer(training_texts)
        elif self.tokenizer_type == "character":
            return self._create_character_tokenizer(training_texts)
        elif self.tokenizer_type == "character_bigram":
            return self._create_character_bigram_tokenizer(training_texts)
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

    def _create_sentencepiece_tokenizer(self, training_texts: List[str]):
        """Create SentencePiece tokenizer"""
        print("Training SentencePiece tokenizer...")

        tokenizer = HindiSentencePieceTokenizer(vocab_size=self.vocab_size)

        # Train tokenizer to a temporary location
        # The save_tokenizer method will copy it to the correct experiment-specific path
        model_prefix = os.path.join(self.tokenizer_dir, 'sentencepiece')
        tokenizer.train_tokenizer(training_texts, model_prefix)

        print(f"SentencePiece tokenizer trained successfully")

        # Add vocab_size attribute for compatibility
        tokenizer.vocab_size = self.vocab_size

        return tokenizer

    def _create_wordpiece_tokenizer(self, training_texts: List[str]):
        """Create WordPiece tokenizer (BERT-style)"""
        print("Training WordPiece tokenizer...")

        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.normalizers import NFC, Lowercase, StripAccents, Sequence

        # Initialize tokenizer
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

        # Set normalizer (don't lowercase for Hindi)
        tokenizer.normalizer = Sequence([NFC()])

        # Set pre-tokenizer
        tokenizer.pre_tokenizer = Whitespace()

        # Create trainer
        trainer = WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            min_frequency=2
        )

        # Train on texts
        tokenizer.train_from_iterator(training_texts, trainer)

        print(f"WordPiece tokenizer trained successfully")

        # Wrap in a class for consistent interface
        return WordPieceTokenizerWrapper(tokenizer, self.vocab_size)

    def _create_bpe_tokenizer(self, training_texts: List[str]):
        """Create BPE tokenizer"""
        print("Training BPE tokenizer...")

        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.normalizers import NFC, Sequence

        # Initialize tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))

        # Set normalizer
        tokenizer.normalizer = Sequence([NFC()])

        # Set pre-tokenizer
        tokenizer.pre_tokenizer = Whitespace()

        # Create trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
            min_frequency=2
        )

        # Train on texts
        tokenizer.train_from_iterator(training_texts, trainer)

        print(f"BPE tokenizer trained successfully")

        # Wrap in a class for consistent interface
        return BPETokenizerWrapper(tokenizer, self.vocab_size)

    def _create_character_tokenizer(self, training_texts: List[str]):
        """Create pure character-level tokenizer"""
        print("Creating character-level tokenizer...")

        tokenizer = DevanagariCharacterTokenizer(
            preserve_grapheme_clusters=True
        )

        print(f"Character tokenizer created with vocab size: {tokenizer.vocab_size}")
        return tokenizer

    def _create_character_bigram_tokenizer(self, training_texts: List[str]):
        """Create hybrid character-bigram tokenizer"""
        print("Training character-bigram tokenizer...")

        # Determine bigram count based on vocab_size config
        # vocab_size includes ~200 base chars, rest are bigrams
        target_bigrams = max(self.vocab_size - 200, 500)

        tokenizer = CharacterBigramTokenizer(
            target_bigrams=target_bigrams,
            min_frequency=100,
            morphological_aware=True,
            preserve_grapheme_clusters=True
        )

        # Train on corpus to extract bigrams
        tokenizer.train_bigrams(training_texts)

        print(f"Character-bigram tokenizer trained with vocab size: {tokenizer.vocab_size}")
        return tokenizer

    @staticmethod
    def load_tokenizer(experiment_name: str, results_dir: str = 'results'):
        """Load a saved tokenizer

        Args:
            experiment_name: Either an experiment name (e.g., 'my_experiment') or
                           a full directory path (e.g., 'results/my_experiment/tokenizer')
            results_dir: Base results directory. Defaults to 'results'.

        Returns:
            Loaded tokenizer instance
        """
        print(f"Loading tokenizer for experiment: {experiment_name}")

        # Check if experiment_name is actually a directory path
        if os.path.isdir(experiment_name):
            # experiment_name is a full directory path (e.g., 'results/exp/tokenizer')
            tokenizer_dir = experiment_name
        else:
            # Construct path: results/{experiment_name}/tokenizer/
            tokenizer_dir = os.path.join(results_dir, experiment_name, 'tokenizer')

        metadata_path = os.path.join(tokenizer_dir, 'tokenizer_metadata.pkl')

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No tokenizer found for experiment: {experiment_name}\n"
                f"Expected metadata at: {metadata_path}\n"
                f"Make sure the tokenizer has been saved to: {tokenizer_dir}"
            )

        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        tokenizer_type = metadata['tokenizer_type']

        if tokenizer_type == "sentencepiece":
            model_path = os.path.join(tokenizer_dir, 'sentencepiece.model')
            tokenizer = HindiSentencePieceTokenizer()
            tokenizer.model = spm.SentencePieceProcessor(model_file=model_path)
            tokenizer.vocab_size = metadata['vocab_size']
            return tokenizer
        elif tokenizer_type == "wordpiece":
            from tokenizers import Tokenizer
            tokenizer_path = os.path.join(tokenizer_dir, 'wordpiece.json')
            tokenizer = Tokenizer.from_file(tokenizer_path)
            return WordPieceTokenizerWrapper(tokenizer, metadata['vocab_size'])
        elif tokenizer_type == "bpe":
            from tokenizers import Tokenizer
            tokenizer_path = os.path.join(tokenizer_dir, 'bpe.json')
            tokenizer = Tokenizer.from_file(tokenizer_path)
            return BPETokenizerWrapper(tokenizer, metadata['vocab_size'])
        elif tokenizer_type == "character":
            return DevanagariCharacterTokenizer.load(tokenizer_dir)
        elif tokenizer_type == "character_bigram":
            return CharacterBigramTokenizer.load(tokenizer_dir)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    def save_tokenizer(self, tokenizer, save_path: str):
        """Save tokenizer with metadata

        Args:
            tokenizer: The tokenizer to save
            save_path: Directory path where tokenizer should be saved
                      (e.g., 'results/experiment_name/tokenizer')
        """
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Extract experiment name from path (last directory component or use full path)
        experiment_name = os.path.basename(save_path.rstrip('/'))

        # Save the actual tokenizer files to the correct path
        if self.tokenizer_type == "sentencepiece":
            # For SentencePiece, save the model file
            model_path = os.path.join(save_path, 'sentencepiece.model')
            if hasattr(tokenizer, 'model') and tokenizer.model is not None:
                # Copy the model file if it exists elsewhere
                import shutil
                temp_model_path = os.path.join(self.tokenizer_dir, 'sentencepiece.model')
                if os.path.exists(temp_model_path):
                    shutil.copy(temp_model_path, model_path)
                    print(f"SentencePiece model saved to {model_path}")
        elif self.tokenizer_type == "wordpiece":
            # For WordPiece, save the tokenizer JSON
            tokenizer_path = os.path.join(save_path, 'wordpiece.json')
            if hasattr(tokenizer, 'tokenizer'):
                tokenizer.tokenizer.save(tokenizer_path)
                print(f"WordPiece tokenizer saved to {tokenizer_path}")
        elif self.tokenizer_type == "bpe":
            # For BPE, save the tokenizer JSON
            tokenizer_path = os.path.join(save_path, 'bpe.json')
            if hasattr(tokenizer, 'tokenizer'):
                tokenizer.tokenizer.save(tokenizer_path)
                print(f"BPE tokenizer saved to {tokenizer_path}")
        elif self.tokenizer_type == "character":
            # For character tokenizer, use its save method
            tokenizer.save(save_path)
        elif self.tokenizer_type == "character_bigram":
            # For character-bigram tokenizer, use its save method
            tokenizer.save(save_path)

        # Save metadata
        metadata = {
            'tokenizer_type': self.tokenizer_type,
            'vocab_size': self.vocab_size,
            'experiment_name': experiment_name
        }

        metadata_path = os.path.join(save_path, 'tokenizer_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Tokenizer metadata saved to {metadata_path}")


class WordPieceTokenizerWrapper:
    """Wrapper for WordPiece tokenizer to provide consistent interface"""

    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

        # Special tokens for WordPiece (BERT-style)
        # Special tokens: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        # Get token IDs from the tokenizer vocabulary
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_token_id = self.tokenizer.token_to_id("[UNK]")
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")

        # For compatibility, set eos_token to [SEP] and bos_token to [CLS]
        self.eos_token = "[SEP]"
        self.eos_token_id = self.sep_token_id
        self.bos_token = "[CLS]"
        self.bos_token_id = self.cls_token_id

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text"""
        return self.tokenizer.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to tokens"""
        encoding = self.tokenizer.encode(text)
        return encoding.tokens

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
        """
        Tokenize text using HuggingFace-style interface.

        Args:
            text: Text or list of texts to tokenize
            text_pair: Optional second text for text pair tasks
            padding: Padding strategy ('max_length', True, or False)
            truncation: Truncation strategy ('longest_first', True, or False)
            max_length: Maximum sequence length
            return_tensors: 'pt' for PyTorch tensors, None for lists

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Handle single text vs batch
        is_batched = isinstance(text, list)
        texts = text if is_batched else [text]
        text_pairs = text_pair if text_pair is not None else [None] * len(texts)
        if not isinstance(text_pairs, list):
            text_pairs = [text_pairs]

        # Set default max_length if not provided
        if max_length is None:
            max_length = 512

        # Encode all texts
        all_input_ids = []
        for txt, txt_pair in zip(texts, text_pairs):
            if txt_pair is not None:
                # Combine text and text_pair with [CLS] and [SEP] tokens (BERT-style)
                combined_text = f"{self.cls_token} {txt} {self.sep_token} {txt_pair} {self.sep_token}"
            else:
                combined_text = f"{self.cls_token} {txt} {self.sep_token}"

            # Encode the text
            encoding = self.tokenizer.encode(combined_text)
            input_ids = encoding.ids

            # Apply truncation
            if truncation:
                input_ids = input_ids[:max_length]

            all_input_ids.append(input_ids)

        # Apply padding
        if padding:
            if padding == 'max_length' or padding is True:
                target_length = max_length if padding == 'max_length' else max(len(ids) for ids in all_input_ids)
            else:
                target_length = max(len(ids) for ids in all_input_ids)

            # Pad all sequences
            padded_input_ids = []
            attention_masks = []
            for input_ids in all_input_ids:
                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1] * len(input_ids)

                # Pad to target length
                padding_length = target_length - len(input_ids)
                if padding_length > 0:
                    input_ids = input_ids + [self.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length

                padded_input_ids.append(input_ids)
                attention_masks.append(attention_mask)

            all_input_ids = padded_input_ids
        else:
            # No padding - create attention masks for actual tokens only
            attention_masks = [[1] * len(ids) for ids in all_input_ids]

        # Convert to tensors if requested
        if return_tensors == 'pt':
            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long)

        return {
            'input_ids': all_input_ids,
            'attention_mask': attention_masks
        }

    def pad(
        self,
        encoded_inputs,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
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


class BPETokenizerWrapper:
    """Wrapper for BPE tokenizer to provide consistent interface"""

    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

        # Special tokens for BPE (GPT-style)
        # Special tokens: ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.mask_token = "<mask>"

        # Get token IDs from the tokenizer vocabulary
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.unk_token_id = self.tokenizer.token_to_id("<unk>")
        self.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.eos_token_id = self.tokenizer.token_to_id("</s>")
        self.mask_token_id = self.tokenizer.token_to_id("<mask>")

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text"""
        return self.tokenizer.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to tokens"""
        encoding = self.tokenizer.encode(text)
        return encoding.tokens

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
        """
        Tokenize text using HuggingFace-style interface.

        Args:
            text: Text or list of texts to tokenize
            text_pair: Optional second text for text pair tasks
            padding: Padding strategy ('max_length', True, or False)
            truncation: Truncation strategy ('longest_first', True, or False)
            max_length: Maximum sequence length
            return_tensors: 'pt' for PyTorch tensors, None for lists

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Handle single text vs batch
        is_batched = isinstance(text, list)
        texts = text if is_batched else [text]
        text_pairs = text_pair if text_pair is not None else [None] * len(texts)
        if not isinstance(text_pairs, list):
            text_pairs = [text_pairs]

        # Set default max_length if not provided
        if max_length is None:
            max_length = 512

        # Encode all texts
        all_input_ids = []
        for txt, txt_pair in zip(texts, text_pairs):
            if txt_pair is not None:
                # Combine text and text_pair with separator
                combined_text = f"{txt} {self.eos_token} {txt_pair}"
            else:
                combined_text = txt

            # Encode the text
            encoding = self.tokenizer.encode(combined_text)
            input_ids = encoding.ids

            # Apply truncation
            if truncation:
                input_ids = input_ids[:max_length]

            all_input_ids.append(input_ids)

        # Apply padding
        if padding:
            if padding == 'max_length' or padding is True:
                target_length = max_length if padding == 'max_length' else max(len(ids) for ids in all_input_ids)
            else:
                target_length = max(len(ids) for ids in all_input_ids)

            # Pad all sequences
            padded_input_ids = []
            attention_masks = []
            for input_ids in all_input_ids:
                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1] * len(input_ids)

                # Pad to target length
                padding_length = target_length - len(input_ids)
                if padding_length > 0:
                    input_ids = input_ids + [self.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length

                padded_input_ids.append(input_ids)
                attention_masks.append(attention_mask)

            all_input_ids = padded_input_ids
        else:
            # No padding - create attention masks for actual tokens only
            attention_masks = [[1] * len(ids) for ids in all_input_ids]

        # Convert to tensors if requested
        if return_tensors == 'pt':
            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long)

        return {
            'input_ids': all_input_ids,
            'attention_mask': attention_masks
        }

    def pad(
        self,
        encoded_inputs,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
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
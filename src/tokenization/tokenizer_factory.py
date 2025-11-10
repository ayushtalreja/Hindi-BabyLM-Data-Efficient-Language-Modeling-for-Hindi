import os
import pickle
from typing import List, Union
from transformers import AutoTokenizer
import sentencepiece as spm

from .sentencepiece_tokenizer import HindiSentencePieceTokenizer


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


class BPETokenizerWrapper:
    """Wrapper for BPE tokenizer to provide consistent interface"""

    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

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

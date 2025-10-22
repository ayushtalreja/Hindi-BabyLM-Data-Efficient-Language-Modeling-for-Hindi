import sentencepiece as spm
from typing import List, Dict, Union, Optional
import torch

class HindiSentencePieceTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.model = None
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def train_tokenizer(self, corpus: List[str], model_prefix: str):
        """Train SentencePiece tokenizer on Hindi corpus"""
        # Create training data file
        training_file = f"{model_prefix}_training.txt"
        with open(training_file, 'w', encoding='utf-8') as f:
            for text in corpus:
                f.write(text + '\n')

        # Train tokenizer
        spm.SentencePieceTrainer.train(
            input=training_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=0.995,  # Important for Hindi
            model_type='bpe'
        )

        # Load trained model
        self.model = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    def __call__(self,
                 text: Union[str, List[str]],
                 return_tensors: Optional[str] = None,
                 padding: Union[bool, str] = False,
                 truncation: bool = False,
                 max_length: Optional[int] = None,
                 **kwargs) -> Dict[str, Union[List, torch.Tensor]]:
        """
        HuggingFace-style tokenizer interface

        Args:
            text: Input text or list of texts
            return_tensors: 'pt' for PyTorch tensors, None for lists
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if self.model is None:
            raise ValueError("Tokenizer model not loaded. Call train_tokenizer() or load a model first.")

        # Handle single string vs list of strings
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # Encode all texts
        encoded_batch = []
        for t in texts:
            ids = self.model.encode(t, out_type=int)

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

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using trained model"""
        return self.model.encode(text, out_type=str)

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        return self.model.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text"""
        return self.model.decode(ids)
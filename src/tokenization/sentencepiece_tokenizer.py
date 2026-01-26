import sentencepiece as spm
from typing import List, Dict, Union, Optional
import torch

class HindiSentencePieceTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.model = None

        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

        # Special token strings (for HuggingFace compatibility)
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.mask_token = "<mask>"

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
                 text_pair: Optional[Union[str, List[str]]] = None,
                 return_tensors: Optional[str] = None,
                 padding: Union[bool, str] = False,
                 truncation: Union[bool, str] = False,
                 max_length: Optional[int] = None,
                 **kwargs) -> Dict[str, Union[List, torch.Tensor]]:
        """
        HuggingFace-style tokenizer interface

        Args:
            text: Input text or list of texts
            text_pair: Optional second text for text pair tasks
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
        text_pairs = text_pair if text_pair is not None else [None] * len(texts)
        if not isinstance(text_pairs, list):
            text_pairs = [text_pair]

        # Encode all texts
        encoded_batch = []
        for t, t_pair in zip(texts, text_pairs):
            if t_pair is not None:
                # Combine text and text_pair with separator
                combined = f"{t} {self.eos_token} {t_pair}"
            else:
                combined = t

            ids = self.model.encode(combined, out_type=int)

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
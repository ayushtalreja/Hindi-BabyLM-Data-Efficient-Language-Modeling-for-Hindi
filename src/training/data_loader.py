import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HindiLanguageModelingDataset(Dataset):
    """Dataset for Hindi language modeling (both CLM and MLM)."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def create_mlm_collator(tokenizer, mlm_probability: float = 0.15) -> DataCollatorForLanguageModeling:
    """
    Create a data collator for Masked Language Modeling (MLM).

    For MLM training (DeBERTa, BERT), we need to:
    1. Randomly mask ~15% of tokens with [MASK]
    2. Set labels = original token IDs for masked positions
    3. Set labels = -100 for non-masked positions (ignored in loss)

    Args:
        tokenizer: Tokenizer with mask_token defined
        mlm_probability: Probability of masking tokens (default: 0.15)

    Returns:
        DataCollatorForLanguageModeling configured for MLM
    """
    # Ensure tokenizer has mask token
    if tokenizer.mask_token is None:
        logger.warning("Tokenizer has no mask_token set. Setting to '[MASK]'")
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=None,
        return_tensors='pt'
    )


def create_clm_collator(tokenizer) -> DataCollatorForLanguageModeling:
    """
    Create a data collator for Causal Language Modeling (CLM).

    For CLM training (GPT), the collator simply returns the input_ids
    without masking. Labels are set to input_ids in the training loop.

    Args:
        tokenizer: Tokenizer

    Returns:
        DataCollatorForLanguageModeling configured for CLM (no masking)
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors='pt'
    )


def create_data_collator(
    tokenizer,
    model_type: str,
    mlm_probability: float = 0.15
) -> Optional[DataCollatorForLanguageModeling]:
    """
    Create appropriate data collator based on model type.

    Args:
        tokenizer: Tokenizer
        model_type: Type of model ('gpt', 'deberta', etc.)
        mlm_probability: MLM masking probability (for MLM models)

    Returns:
        Data collator appropriate for the model type
    """
    model_type_lower = model_type.lower()

    if model_type_lower in ['deberta', 'bert', 'albert', 'roberta']:
        logger.info(f"Creating MLM data collator for {model_type} with mlm_probability={mlm_probability}")
        return create_mlm_collator(tokenizer, mlm_probability)
    elif model_type_lower in ['gpt', 'gpt2', 'gpt-2']:
        logger.info(f"Creating CLM data collator for {model_type} (no masking)")
        return create_clm_collator(tokenizer)
    else:
        logger.warning(f"Unknown model type '{model_type}', defaulting to CLM collator")
        return create_clm_collator(tokenizer)
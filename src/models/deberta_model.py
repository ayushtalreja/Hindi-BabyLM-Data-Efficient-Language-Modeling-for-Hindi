"""
DeBERTa Model for Hindi Language Modeling

This module provides a DeBERTa-based masked language model using HuggingFace's
DebertaV2ForMaskedLM with support for different model sizes.

DeBERTa (Decoding-enhanced BERT with disentangled attention) improves upon BERT with:
- Disentangled attention mechanism
- Enhanced mask decoder
- Relative position encoding
"""

import torch
import torch.nn as nn
from transformers import DebertaV2Config, DebertaV2ForMaskedLM
from typing import Dict, Any, Optional


class HindiDeBERTaModel(nn.Module):
    """DeBERTa model for Hindi with configurable sizes"""

    # Model size presets
    MODEL_SIZES = {
        'tiny': {
            'hidden_size': 384,
            'num_layers': 6,
            'num_heads': 6,
            'intermediate_size': 1536,
        },
        'small': {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'intermediate_size': 3072,
        },
        'base': {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'intermediate_size': 3072,
        },
        'large': {
            'hidden_size': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'intermediate_size': 4096,
        },
    }

    def __init__(self, vocab_size: int, config: Dict[str, Any]):
        """
        Initialize Hindi DeBERTa model

        Args:
            vocab_size: Size of vocabulary
            config: Configuration dictionary with model parameters
        """
        super().__init__()

        # Get model size preset if specified
        model_size = config.get('model_size', 'small')
        if model_size in self.MODEL_SIZES:
            size_preset = self.MODEL_SIZES[model_size]
            # Apply preset, but allow config to override
            hidden_size = config.get('hidden_size', size_preset['hidden_size'])
            num_layers = config.get('num_layers', size_preset['num_layers'])
            num_heads = config.get('num_heads', size_preset['num_heads'])
            intermediate_size = config.get('intermediate_size', size_preset['intermediate_size'])
        else:
            # Use config values or defaults
            hidden_size = config.get('hidden_size', 768)
            num_layers = config.get('num_layers', 12)
            num_heads = config.get('num_heads', 12)
            intermediate_size = config.get('intermediate_size', 3072)

        # Get DeBERTa-specific parameters
        max_relative_positions = config.get('max_relative_positions', -1)
        if max_relative_positions == -1:
            max_relative_positions = config.get('max_length', 512)

        # Create HuggingFace DeBERTaV2 config
        self.config = DebertaV2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=config.get('max_length', 512),
            hidden_dropout_prob=config.get('dropout', 0.1),
            attention_probs_dropout_prob=config.get('dropout', 0.1),
            # DeBERTa-specific parameters
            position_buckets=config.get('position_buckets', 256),
            relative_attention=config.get('relative_attention', True),
            max_relative_positions=max_relative_positions,
            pooler_hidden_size=config.get('pooler_hidden_size', hidden_size),
            pooler_dropout=config.get('pooler_dropout', 0.1),
            pooler_hidden_act=config.get('pooler_hidden_act', 'gelu'),
        )

        # Create the model
        self.model = DebertaV2ForMaskedLM(self.config)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for masked language modeling [batch, seq_len]

        Returns:
            Model outputs with logits and optional loss
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Count number of parameters

        Args:
            only_trainable: If True, count only trainable parameters

        Returns:
            Total number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Convenience function for creating model
def create_deberta_model(vocab_size: int, model_size: str = 'small', **kwargs) -> HindiDeBERTaModel:
    """
    Create a DeBERTa model with specified size

    Args:
        vocab_size: Vocabulary size
        model_size: Model size ('tiny', 'small', 'base', 'large')
        **kwargs: Additional config overrides

    Returns:
        HindiDeBERTaModel instance
    """
    config = {'model_size': model_size, **kwargs}
    return HindiDeBERTaModel(vocab_size, config)

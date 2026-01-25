"""
GPT Model for Hindi Language Modeling

This module provides a GPT-based language model using HuggingFace's GPT2LMHeadModel
with support for different model sizes (tiny, small, medium).
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, Any, Optional


class HindiGPTModel(nn.Module):
    """GPT model for Hindi with configurable sizes"""

    # Model size presets
    MODEL_SIZES = {
        'tiny': {
            'hidden_size': 512,
            'num_layers': 6,
            'num_heads': 8,
            'intermediate_size': 2048,
        },
        'small': {
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'intermediate_size': 3072,
        },
        'medium': {
            'hidden_size': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'intermediate_size': 4096,
        },
    }

    def __init__(self, vocab_size: int, config: Dict[str, Any]):
        """
        Initialize Hindi GPT model

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

        # Create HuggingFace GPT2 config
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=config.get('max_length', 512),
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_inner=intermediate_size,
            activation_function=config.get('activation', 'gelu'),
            resid_pdrop=config.get('dropout', 0.1),
            embd_pdrop=config.get('dropout', 0.1),
            attn_pdrop=config.get('dropout', 0.1),
            use_cache=config.get('use_cache', True),
        )

        # Create the model
        self.model = GPT2LMHeadModel(self.config)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs):
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for language modeling [batch, seq_len]
            **kwargs: Additional arguments passed to underlying model
                      (e.g., output_hidden_states, output_attentions)

        Returns:
            Model outputs with logits and optional loss
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, input_ids: torch.Tensor,
                max_length: int = 50,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                **kwargs):
        """
        Generate text autoregressively

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_length: Maximum total length (input + generated)
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) sampling
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs [batch, total_length]
        """
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.config.pad_token_id,
            **kwargs
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
def create_gpt_model(vocab_size: int, model_size: str = 'small', **kwargs) -> HindiGPTModel:
    """
    Create a GPT model with specified size

    Args:
        vocab_size: Vocabulary size
        model_size: Model size ('tiny', 'small', 'medium')
        **kwargs: Additional config overrides

    Returns:
        HindiGPTModel instance
    """
    config = {'model_size': model_size, **kwargs}
    return HindiGPTModel(vocab_size, config)

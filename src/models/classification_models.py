"""
Classification Wrappers for Language Models

This module provides classification heads on top of language models to enable
sequence-level classification tasks. The language models (GPT, DeBERTa) output
per-token predictions, but classification tasks need per-example predictions.

Architecture:
- Language Model: [batch, seq_len, vocab_size]
- Pooling: [batch, seq_len, hidden_size] -> [batch, hidden_size]
- Classification Head: [batch, hidden_size] -> [batch, num_classes]
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClassificationOutput:
    """Output from classification models"""
    logits: torch.Tensor  # [batch, num_classes]
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None  # [batch, hidden_size]
    pooled_output: Optional[torch.Tensor] = None  # [batch, hidden_size]


class GPTForSequenceClassification(nn.Module):
    """
    GPT model with classification head for sequence-level classification.

    Uses last-token pooling since GPT is a causal language model where
    the last token has seen the full sequence context.
    """

    def __init__(self,
                 lm_model: nn.Module,
                 num_classes: int,
                 hidden_size: int,
                 dropout: float = 0.1,
                 freeze_base: bool = False,
                 label_smoothing: float = 0.0):
        """
        Args:
            lm_model: Pre-trained GPT language model
            num_classes: Number of classification classes
            hidden_size: Hidden size of the language model
            dropout: Dropout probability for classification head
            freeze_base: If True, freeze the language model parameters
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()

        self.lm_model = lm_model
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.label_smoothing = label_smoothing

        # Freeze base model if requested
        if freeze_base:
            for param in self.lm_model.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classification head with Xavier uniform for better gradient flow
        # CRITICAL FIX: Standard normal with std=0.02 causes gradient saturation
        # Xavier initialization scales weights based on fan_in/fan_out, providing stronger initial gradients
        # This prevents the softmax over similar small values from producing vanishing gradients
        nn.init.xavier_uniform_(self.classifier.weight)
        # Small positive bias helps break symmetry and improves initial gradient flow
        nn.init.constant_(self.classifier.bias, 0.1)

        # Match dtype of base model to avoid dtype mismatch errors
        self._match_base_model_dtype()

        # Log initialization statistics for debugging
        weight_std = self.classifier.weight.std().item()
        logger.info(f"GPTForSequenceClassification initialized: num_classes={num_classes}, "
                   f"hidden_size={hidden_size}, weight_std={weight_std:.4f}, bias={self.classifier.bias[0].item():.2f}")

    def _match_base_model_dtype(self):
        """Match the dtype of the classification head to the base model"""
        # Get dtype from first parameter of base model
        base_dtype = next(self.lm_model.parameters()).dtype

        # Convert classification head to same dtype
        self.classifier = self.classifier.to(dtype=base_dtype)

        # Note: Dropout doesn't have parameters, so no conversion needed

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> ClassificationOutput:
        """
        Forward pass for classification.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Classification labels [batch] (optional)

        Returns:
            ClassificationOutput with logits [batch, num_classes]
        """
        # Get language model outputs
        lm_outputs = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract hidden states: [batch, seq_len, hidden_size]
        if hasattr(lm_outputs, 'hidden_states') and lm_outputs.hidden_states is not None:
            # If model outputs hidden states, use the last layer
            hidden_states = lm_outputs.hidden_states[-1]
        elif hasattr(lm_outputs, 'last_hidden_state'):
            hidden_states = lm_outputs.last_hidden_state
        else:
            # For GPT2LMHeadModel, we need to get transformer outputs
            # The logits shape is [batch, seq_len, vocab_size]
            # We need to access the transformer's last hidden state
            transformer_outputs = self.lm_model.model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = transformer_outputs.last_hidden_state

        # Last token pooling: [batch, hidden_size]
        # For causal LM, the last token has seen the full context
        if attention_mask is not None:
            # Get the last non-padding token for each example
            sequence_lengths = attention_mask.sum(dim=1).long() - 1
            batch_size = hidden_states.shape[0]
            pooled_output = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]
        else:
            # If no attention mask, just use the last token
            pooled_output = hidden_states[:, -1, :]

        # Apply classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch, num_classes]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fct(logits, labels)

        return ClassificationOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            pooled_output=pooled_output
        )


class DeBERTaForSequenceClassification(nn.Module):
    """
    DeBERTa model with classification head for sequence-level classification.

    Uses mean pooling over all tokens since DeBERTa is a masked language model
    without a special CLS token position.
    """

    def __init__(self,
                 lm_model: nn.Module,
                 num_classes: int,
                 hidden_size: int,
                 dropout: float = 0.1,
                 pooling_strategy: str = 'mean',
                 freeze_base: bool = False,
                 label_smoothing: float = 0.0):
        """
        Args:
            lm_model: Pre-trained DeBERTa language model
            num_classes: Number of classification classes
            hidden_size: Hidden size of the language model
            dropout: Dropout probability for classification head
            pooling_strategy: Pooling strategy ('mean', 'max', 'first', 'last')
            freeze_base: If True, freeze the language model parameters
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()

        self.lm_model = lm_model
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy
        self.label_smoothing = label_smoothing

        # Freeze base model if requested
        if freeze_base:
            for param in self.lm_model.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classification head with Xavier uniform for better gradient flow
        # CRITICAL FIX: Standard normal with std=0.02 causes gradient saturation
        # Xavier initialization scales weights based on fan_in/fan_out, providing stronger initial gradients
        nn.init.xavier_uniform_(self.classifier.weight)
        # Small positive bias helps break symmetry and improves initial gradient flow
        nn.init.constant_(self.classifier.bias, 0.1)

        # Match dtype of base model to avoid dtype mismatch errors
        self._match_base_model_dtype()

        # Log initialization statistics for debugging
        weight_std = self.classifier.weight.std().item()
        logger.info(f"DeBERTaForSequenceClassification initialized: num_classes={num_classes}, "
                   f"hidden_size={hidden_size}, pooling={pooling_strategy}, "
                   f"weight_std={weight_std:.4f}, bias={self.classifier.bias[0].item():.2f}")

    def _match_base_model_dtype(self):
        """Match the dtype of the classification head to the base model"""
        # Get dtype from first parameter of base model
        base_dtype = next(self.lm_model.parameters()).dtype

        # Convert classification head to same dtype
        self.classifier = self.classifier.to(dtype=base_dtype)

        # Note: Dropout doesn't have parameters, so no conversion needed

    def pool_hidden_states(self,
                          hidden_states: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool hidden states to get sequence representation.

        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled representation [batch, hidden_size]
        """
        if self.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            if attention_mask is not None:
                # Expand mask for broadcasting: [batch, seq_len, 1]
                # Match dtype to hidden_states to support mixed precision (BFloat16/Float16)
                mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                # Sum over sequence length, weighted by mask
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                # Divide by number of non-padding tokens
                sum_mask = mask_expanded.sum(dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
                pooled = sum_hidden / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)

        elif self.pooling_strategy == 'max':
            # Max pooling
            if attention_mask is not None:
                # Set padding positions to large negative value
                # Match dtype to hidden_states to support mixed precision (BFloat16/Float16)
                mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                hidden_states = hidden_states.clone()
                hidden_states[mask_expanded == 0] = -1e9
            pooled = hidden_states.max(dim=1)[0]

        elif self.pooling_strategy == 'first':
            # First token (CLS-like)
            pooled = hidden_states[:, 0, :]

        elif self.pooling_strategy == 'last':
            # Last non-padding token
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1).long() - 1
                batch_size = hidden_states.shape[0]
                pooled = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    sequence_lengths
                ]
            else:
                pooled = hidden_states[:, -1, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return pooled

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> ClassificationOutput:
        """
        Forward pass for classification.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Classification labels [batch] (optional)

        Returns:
            ClassificationOutput with logits [batch, num_classes]
        """
        # Get language model outputs
        lm_outputs = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract hidden states: [batch, seq_len, hidden_size]
        if hasattr(lm_outputs, 'hidden_states') and lm_outputs.hidden_states is not None:
            # If model outputs hidden states, use the last layer
            hidden_states = lm_outputs.hidden_states[-1]
        elif hasattr(lm_outputs, 'last_hidden_state'):
            hidden_states = lm_outputs.last_hidden_state
        else:
            # For DebertaV2ForMaskedLM, access the base model
            base_outputs = self.lm_model.model.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = base_outputs.last_hidden_state

        # Pool to get sequence representation: [batch, hidden_size]
        pooled_output = self.pool_hidden_states(hidden_states, attention_mask)

        # Apply classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch, num_classes]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fct(logits, labels)

        return ClassificationOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            pooled_output=pooled_output
        )


def wrap_model_for_classification(
    lm_model: nn.Module,
    model_type: str,
    num_classes: int,
    hidden_size: int,
    dropout: float = 0.1,
    freeze_base: bool = False,
    pooling_strategy: str = 'auto',
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Wrap a language model with a classification head.

    Args:
        lm_model: Pre-trained language model
        model_type: Type of model ('gpt', 'bert', or 'deberta')
        num_classes: Number of classification classes
        hidden_size: Hidden size of the language model
        dropout: Dropout probability for classification head
        freeze_base: If True, freeze the language model parameters
        pooling_strategy: Pooling strategy ('auto', 'mean', 'max', 'first', 'last')
                         'auto' → 'last' for GPT, 'first' for BERT/DeBERTa
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Model wrapped with classification head
    """
    model_type = model_type.lower()

    if model_type == 'gpt':
        # GPT models use last-token pooling (causal LM)
        return GPTForSequenceClassification(
            lm_model=lm_model,
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
            freeze_base=freeze_base,
            label_smoothing=label_smoothing
        )

    elif model_type == 'bert':
        # BERT-style models (BERT, ALBERT, IndicBERT) use [CLS] token pooling
        # Auto-select pooling strategy
        if pooling_strategy == 'auto':
            pooling_strategy = 'first'  # Use [CLS] token (first position)

        return DeBERTaForSequenceClassification(
            lm_model=lm_model,
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
            pooling_strategy=pooling_strategy,
            freeze_base=freeze_base,
            label_smoothing=label_smoothing
        )

    elif model_type == 'deberta':
        # DeBERTa models - use flexible pooling (default to first for classification)
        # Auto-select pooling strategy
        if pooling_strategy == 'auto':
            pooling_strategy = 'first'  # Use [CLS] token for consistency with BERT

        return DeBERTaForSequenceClassification(
            lm_model=lm_model,
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
            pooling_strategy=pooling_strategy,
            freeze_base=freeze_base,
            label_smoothing=label_smoothing
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'gpt', 'bert', or 'deberta'")

"""
Enhanced GPT Model with Advanced Features

This module implements an enhanced GPT architecture with:
- Configurable position encodings (absolute, RoPE, ALiBi)
- Gradient checkpointing for memory efficiency
- Multiple model size variants
- Hindi-specific optimizations
- Flash attention support (if available)

Model Size Variants:
- Tiny: 50M parameters (6 layers, 512 hidden, 8 heads)
- Small: 110M parameters (12 layers, 768 hidden, 12 heads)
- Medium: 350M parameters (24 layers, 1024 hidden, 16 heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Tuple
import logging
from .position_encodings import create_position_encoding

logger = logging.getLogger(__name__)


# Try to import flash attention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    logger.info("Flash Attention not available, using standard attention")


class EnhancedGPTConfig:
    """Configuration for Enhanced GPT model"""

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

    def __init__(self, vocab_size: int, model_size: str = 'small', **kwargs):
        """
        Initialize config

        Args:
            vocab_size: Vocabulary size
            model_size: Model size preset ('tiny', 'small', 'medium')
            **kwargs: Override any config value
        """
        # Load size preset
        if model_size in self.MODEL_SIZES:
            size_config = self.MODEL_SIZES[model_size]
            for key, value in size_config.items():
                setattr(self, key, value)
        else:
            raise ValueError(f"Unknown model size: {model_size}")

        # Basic parameters
        self.vocab_size = vocab_size
        self.model_size = model_size
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 512)

        # Position encoding
        self.position_encoding_type = kwargs.get('position_encoding_type', 'learned')

        # Regularization
        self.dropout = kwargs.get('dropout', 0.1)
        self.attention_dropout = kwargs.get('attention_dropout', 0.1)
        self.residual_dropout = kwargs.get('residual_dropout', 0.1)

        # Layer normalization
        self.layer_norm_eps = kwargs.get('layer_norm_eps', 1e-5)
        self.use_rms_norm = kwargs.get('use_rms_norm', False)

        # Activation
        self.activation = kwargs.get('activation', 'gelu')

        # Attention
        self.use_flash_attention = kwargs.get('use_flash_attention', HAS_FLASH_ATTN)
        self.attention_bias = kwargs.get('attention_bias', True)

        # Gradient checkpointing
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)

        # Initialization
        self.initializer_range = kwargs.get('initializer_range', 0.02)

        # Override any remaining kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class EnhancedGPTAttention(nn.Module):
    """Multi-head self-attention with advanced features"""

    def __init__(self, config: EnhancedGPTConfig):
        super().__init__()
        self.config = config

        assert config.hidden_size % config.num_heads == 0
        self.head_dim = config.hidden_size // config.num_heads
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size

        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        # Position encoding (for RoPE)
        if config.position_encoding_type == 'rope':
            self.rotary_emb = create_position_encoding(
                'rope',
                config.hidden_size,
                config.num_heads,
                config.max_position_embeddings
            )
        else:
            self.rotary_emb = None

        # Scaling factor
        self.scale = self.head_dim ** -0.5

        # Flash attention flag
        self.use_flash_attention = config.use_flash_attention and HAS_FLASH_ATTN

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, 1, 1, seq_len]
            alibi_bias: ALiBi position bias (if using ALiBi)

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)

        # Compute attention
        if self.use_flash_attention:
            # Use flash attention (more efficient)
            attn_output = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                causal=True
            )
            attn_output = attn_output.transpose(1, 2)
        else:
            # Standard attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Add ALiBi bias if provided
            if alibi_bias is not None:
                attn_scores = attn_scores + alibi_bias

            # Apply causal mask
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class EnhancedGPTMLP(nn.Module):
    """Feed-forward network with configurable activation"""

    def __init__(self, config: EnhancedGPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.residual_dropout)

        # Activation function
        if config.activation == 'gelu':
            self.act = nn.GELU()
        elif config.activation == 'relu':
            self.act = nn.ReLU()
        elif config.activation == 'swiglu':
            self.act = nn.SiLU()
            # SwiGLU requires a gate
            self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        else:
            raise ValueError(f"Unknown activation: {config.activation}")

        self.use_swiglu = config.activation == 'swiglu'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.use_swiglu:
            # SwiGLU: (Wx) * σ(Vx)
            return self.dropout(self.fc2(self.act(self.fc1(x)) * self.gate(x)))
        else:
            return self.dropout(self.fc2(self.act(self.fc1(x))))


class EnhancedGPTBlock(nn.Module):
    """Transformer block with pre-norm architecture"""

    def __init__(self, config: EnhancedGPTConfig):
        super().__init__()
        self.config = config

        # Layer norms
        if config.use_rms_norm:
            self.ln_1 = RMSNorm(config.hidden_size, config.layer_norm_eps)
            self.ln_2 = RMSNorm(config.hidden_size, config.layer_norm_eps)
        else:
            self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Attention and MLP
        self.attn = EnhancedGPTAttention(config)
        self.mlp = EnhancedGPTMLP(config)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with pre-norm and residual connections"""
        # Attention block
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask, alibi_bias)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class EnhancedGPTModel(nn.Module):
    """Enhanced GPT model with advanced features"""

    def __init__(self, config: EnhancedGPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position encodings (if not using RoPE/ALiBi)
        if config.position_encoding_type in ['learned', 'sinusoidal']:
            self.position_embeddings = create_position_encoding(
                config.position_encoding_type,
                config.hidden_size,
                max_len=config.max_position_embeddings
            )
        else:
            self.position_embeddings = None

        # ALiBi bias (if using ALiBi)
        if config.position_encoding_type == 'alibi':
            self.alibi = create_position_encoding(
                'alibi',
                config.hidden_size,
                config.num_heads,
                config.max_position_embeddings
            )
        else:
            self.alibi = None

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EnhancedGPTBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer norm
        if config.use_rms_norm:
            self.ln_f = RMSNorm(config.hidden_size, config.layer_norm_eps)
        else:
            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights between token embeddings and LM head
        self.lm_head.weight = self.token_embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"Initialized Enhanced GPT model: {config.model_size} "
                   f"({self.num_parameters():,} parameters)")

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Count number of parameters"""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for language modeling [batch, seq_len]

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_ids.size()

        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)

        # Add position embeddings (if using learned or sinusoidal)
        if self.position_embeddings is not None:
            hidden_states = self.position_embeddings(hidden_states)

        hidden_states = self.drop(hidden_states)

        # Prepare attention mask
        if attention_mask is not None:
            # Convert to attention mask with large negative value for masked positions
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Prepare causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * torch.finfo(hidden_states.dtype).min,
            diagonal=1
        )
        if attention_mask is not None:
            attention_mask = attention_mask + causal_mask
        else:
            attention_mask = causal_mask

        # Compute ALiBi bias if needed
        alibi_bias = None
        if self.alibi is not None:
            # Create dummy attention scores to get bias
            dummy_scores = torch.zeros(batch_size, self.config.num_heads, seq_len, seq_len,
                                      device=input_ids.device)
            alibi_bias = self.alibi(dummy_scores, seq_len)

        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            if self.config.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    alibi_bias,
                    use_reentrant=False
                )
            else:
                hidden_states = block(hidden_states, attention_mask, alibi_bias)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for autoregressive loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {'logits': logits, 'loss': loss}

    def generate(self, input_ids: torch.Tensor,
                max_new_tokens: int = 50,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate text autoregressively

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) sampling

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get logits for next token
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Convenience function
def create_enhanced_gpt_model(vocab_size: int, model_size: str = 'small', **kwargs) -> EnhancedGPTModel:
    """
    Create an enhanced GPT model

    Args:
        vocab_size: Vocabulary size
        model_size: Model size ('tiny', 'small', 'medium')
        **kwargs: Additional config overrides

    Returns:
        EnhancedGPTModel instance
    """
    config = EnhancedGPTConfig(vocab_size, model_size, **kwargs)
    return EnhancedGPTModel(config)

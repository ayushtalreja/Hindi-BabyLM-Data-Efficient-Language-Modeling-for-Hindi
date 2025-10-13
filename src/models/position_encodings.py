"""
Position Encoding Implementations for Transformer Models

This module implements various position encoding strategies:
1. Absolute (Sinusoidal): Original Transformer position encoding
2. Learned Absolute: Trainable position embeddings
3. RoPE (Rotary Position Embedding): Used in GPT-Neo, LLaMA
4. ALiBi (Attention with Linear Biases): Used in BLOOM
5. Relative Position Encoding: T5-style relative positions

Reference:
- Vaswani et al. (2017) "Attention is All You Need" (Sinusoidal)
- Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Press et al. (2022) "Train Short, Test Long: Attention with Linear Biases"
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal position encoding (original Transformer)

    Features:
    - Deterministic, no parameters
    - Generalizes to longer sequences
    - Encodes absolute positions
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize sinusoidal position encoding

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()

        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add position encoding to input

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with position encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnedPositionEncoding(nn.Module):
    """
    Learned position embeddings (BERT-style)

    Features:
    - Trainable parameters
    - Fixed maximum length
    - Good for shorter sequences
    """

    def __init__(self, d_model: int, max_len: int = 512):
        """
        Initialize learned position encoding

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add position encoding to input

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with position encoding added
        """
        batch_size, seq_len, _ = x.size()

        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        position_embeds = self.position_embeddings(position_ids)
        return x + position_embeds


class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    Features:
    - Encodes relative positions in attention
    - No explicit position embeddings
    - Excellent length generalization
    - Used in GPT-Neo, LLaMA, GPT-J
    """

    def __init__(self, dim: int, max_len: int = 2048, base: int = 10000):
        """
        Initialize RoPE

        Args:
            dim: Dimension per attention head
            max_len: Maximum sequence length
            base: Base for frequency calculation
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Precompute rotation matrices
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached cos/sin values if sequence length changes"""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dims"""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor,
                            cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to queries and keys

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            cos: Cosine values
            sin: Sine values

        Returns:
            Rotated queries and keys
        """
        # Reshape cos/sin for broadcasting
        cos = cos[None, None, :, :]  # [1, 1, seq_len, dim]
        sin = sin[None, None, :, :]

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to queries and keys

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]

        Returns:
            Rotated queries and keys
        """
        seq_len = q.shape[2]
        self._update_cache(seq_len, q.device)

        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]

        return self.apply_rotary_pos_emb(q, k, cos, sin)


class ALiBiPositionBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi)

    Features:
    - No position embeddings
    - Adds bias to attention scores
    - Excellent length extrapolation
    - Used in BLOOM
    """

    def __init__(self, num_heads: int, max_len: int = 2048):
        """
        Initialize ALiBi

        Args:
            num_heads: Number of attention heads
            max_len: Maximum sequence length
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len

        # Compute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # Cache for efficiency
        self._bias_cached = None
        self._seq_len_cached = 0

    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """
        Compute ALiBi slopes

        Args:
            num_heads: Number of attention heads

        Returns:
            Slopes tensor [num_heads]
        """
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # If not a power of 2, interpolate
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = self._get_slopes(2 * closest_power_of_2)
            extra_slopes = extra_slopes[0::2][:num_heads - closest_power_of_2]
            slopes.extend(extra_slopes)

        return torch.tensor(slopes, dtype=torch.float32)

    def _build_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build ALiBi attention bias matrix

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            Bias tensor [1, num_heads, seq_len, seq_len]
        """
        # Create distance matrix
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = memory_position - context_position  # [seq_len, seq_len]

        # Only apply bias to past positions (causal mask)
        relative_position = torch.abs(relative_position).unsqueeze(0)  # [1, seq_len, seq_len]

        # Apply slopes
        slopes = self.slopes.to(device).view(-1, 1, 1)  # [num_heads, 1, 1]
        alibi = slopes * -relative_position  # [num_heads, seq_len, seq_len]

        return alibi.unsqueeze(0)  # [1, num_heads, seq_len, seq_len]

    def forward(self, attention_scores: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Add ALiBi bias to attention scores

        Args:
            attention_scores: Attention scores [batch, heads, seq_len, seq_len]
            seq_len: Sequence length (if None, inferred from attention_scores)

        Returns:
            Attention scores with ALiBi bias added
        """
        if seq_len is None:
            seq_len = attention_scores.size(-1)

        # Update cache if needed
        if seq_len > self._seq_len_cached or self._bias_cached is None:
            self._seq_len_cached = seq_len
            self._bias_cached = self._build_alibi_bias(seq_len, attention_scores.device)

        # Add bias
        bias = self._bias_cached[:, :, :seq_len, :seq_len]
        return attention_scores + bias


class RelativePositionBias(nn.Module):
    """
    Relative Position Bias (T5-style)

    Features:
    - Learned relative position biases
    - Bucket-based approach
    - Good for various sequence lengths
    """

    def __init__(self, num_heads: int, num_buckets: int = 32,
                 max_distance: int = 128, bidirectional: bool = False):
        """
        Initialize relative position bias

        Args:
            num_heads: Number of attention heads
            num_buckets: Number of relative position buckets
            max_distance: Maximum relative distance
            bidirectional: Whether to use bidirectional relative positions
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Map relative positions to buckets

        Args:
            relative_position: Relative position tensor

        Returns:
            Bucket indices
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if self.bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # Half of buckets for exact positions, half for log-scale
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Logarithmic scale for larger distances
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)

        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative position bias

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            Bias tensor [1, num_heads, seq_len, seq_len]
        """
        context_position = torch.arange(seq_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # [seq_len, seq_len]

        buckets = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(buckets)  # [seq_len, seq_len, num_heads]
        values = values.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, seq_len, seq_len]

        return values


def create_position_encoding(encoding_type: str, d_model: int,
                             num_heads: Optional[int] = None,
                             max_len: int = 512, **kwargs) -> nn.Module:
    """
    Factory function to create position encodings

    Args:
        encoding_type: Type of encoding ('sinusoidal', 'learned', 'rope', 'alibi', 'relative')
        d_model: Model dimension
        num_heads: Number of attention heads (required for ALiBi and relative)
        max_len: Maximum sequence length
        **kwargs: Additional arguments

    Returns:
        Position encoding module
    """
    if encoding_type == 'sinusoidal':
        return SinusoidalPositionEncoding(d_model, max_len)

    elif encoding_type == 'learned':
        return LearnedPositionEncoding(d_model, max_len)

    elif encoding_type == 'rope':
        head_dim = d_model // num_heads if num_heads else d_model
        return RotaryPositionEncoding(head_dim, max_len, kwargs.get('base', 10000))

    elif encoding_type == 'alibi':
        if num_heads is None:
            raise ValueError("num_heads is required for ALiBi")
        return ALiBiPositionBias(num_heads, max_len)

    elif encoding_type == 'relative':
        if num_heads is None:
            raise ValueError("num_heads is required for relative position bias")
        return RelativePositionBias(
            num_heads,
            kwargs.get('num_buckets', 32),
            kwargs.get('max_distance', 128),
            kwargs.get('bidirectional', False)
        )

    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

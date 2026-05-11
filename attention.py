"""
Minimal Grouped-Query Attention (GQA) in JAX/Flax NNX.

This file implements a simple, educational attention block used by the
Nemotron-style hybrid model:
- Causal self-attention (decoder-only masking)
- Grouped-query attention (more query heads than KV heads)
- Rotary Position Embeddings (RoPE, Su et al. 2021) applied to Q and K
- No dropout, bias-free projections

The implementation prioritizes readability over speed.
"""

import jax
import jax.numpy as jnp
from flax import nnx


def _apply_rope(x: jax.Array) -> jax.Array:
    """
    Apply Rotary Position Embeddings (RoPE, Su et al. 2021) to a Q or K tensor.

    RoPE encodes token positions as rotations of head-dimension pairs.
    For dimension pair (2i, 2i+1) at sequence position m, the rotation angle is:
        θ_i = m / 10000^(2i / head_dim)

    Key property: after RoPE, the dot product q·k depends only on the
    RELATIVE position (m − n), not absolute positions. This lets the model
    generalize to positions seen at training time.

    RoPE adds no learnable parameters — it is a deterministic function of position.

    Args:
        x: shape (batch, seqlen, num_heads, head_dim) — Q or K tensor
    Returns:
        shape (batch, seqlen, num_heads, head_dim) — position-encoded
    """
    batch, seqlen, num_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    half = head_dim // 2

    # Frequency for each dimension pair: θ_i = 1 / 10000^(2i / head_dim)
    freqs = 1.0 / (
        10000.0 ** (jnp.arange(half, dtype=jnp.float32) * 2 / head_dim)
    )

    # Position indices 0, 1, ..., seqlen-1
    positions = jnp.arange(seqlen, dtype=jnp.float32)

    # angles[m, i] = m * θ_i, shape (seqlen, half)
    angles = jnp.outer(positions, freqs)
    cos = jnp.cos(angles)  # (seqlen, half)
    sin = jnp.sin(angles)  # (seqlen, half)

    # Broadcast to (1, seqlen, 1, half) for batch and head dimensions
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Split into true even/odd dimension pairs: (0,1), (2,3), ...
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # 2D rotation per pair: [even, odd] -> [even*cos - odd*sin, even*sin + odd*cos]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    # Interleave rotated even/odd channels back to the original head_dim layout.
    x_rot = jnp.stack([x_rot_even, x_rot_odd], axis=-1)
    return jnp.reshape(x_rot, x.shape)


class GroupedQueryAttention(nnx.Module):
    """
    Minimal causal self-attention with grouped-query heads.

    Paper-aligned design choices used here:
    - Query heads and KV heads can be different (GQA)
    - Rotary Position Embeddings (RoPE) applied to Q and K
    - No dropout
    - Bias-free linear projections by default

    Args:
        d_model: Hidden/model dimension.
        num_query_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head channel dimension.
        use_bias: Whether projection layers use bias.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_bias: bool = False,
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_bias = use_bias

        assert self.num_query_heads % self.num_kv_heads == 0, (
            "num_query_heads must be divisible by num_kv_heads for GQA"
        )
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert self.num_query_heads * self.head_dim == self.d_model, (
            "d_model must equal num_query_heads * head_dim"
        )

        # Project input tokens into Q, K, and V spaces.
        self.q_proj = nnx.Linear(
            self.d_model,
            self.num_query_heads * self.head_dim,
            use_bias=self.use_bias,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            use_bias=self.use_bias,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            use_bias=self.use_bias,
            rngs=rngs,
        )

        # Project attention output back to model dimension.
        self.out_proj = nnx.Linear(
            self.num_query_heads * self.head_dim,
            self.d_model,
            use_bias=self.use_bias,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Token hidden states, shape (batch, seqlen, d_model)
        Returns:
            Output hidden states, shape (batch, seqlen, d_model)
        """
        batch, seqlen, d_model = x.shape
        assert d_model == self.d_model, "Input d_model does not match module config"

        # 1) Compute Q, K, V projections.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) Expose head dimensions.
        q = jnp.reshape(q, (batch, seqlen, self.num_query_heads, self.head_dim))
        k = jnp.reshape(k, (batch, seqlen, self.num_kv_heads, self.head_dim))
        v = jnp.reshape(v, (batch, seqlen, self.num_kv_heads, self.head_dim))

        # 3) Apply RoPE to Q and K (after head split, before GQA expansion).
        # RoPE encodes positions via rotation — applied per head, on the last dim.
        # V is NOT rotated: position encoding only affects attention score computation.
        q = _apply_rope(q)
        k = _apply_rope(k)

        # 4) Expand KV heads so each query head can attend to a matching KV head.
        kv_repeat = self.num_query_heads // self.num_kv_heads
        k = jnp.repeat(k, kv_repeat, axis=2)
        v = jnp.repeat(v, kv_repeat, axis=2)

        # 5) Move heads before sequence for standard attention math.
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, heads, seqlen, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # 6) Dot-product attention scores.
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # 7) Causal mask: position i cannot see future position j > i.
        causal_mask = jnp.tril(jnp.ones((seqlen, seqlen), dtype=bool))
        scores = jnp.where(causal_mask[None, None, :, :], scores, -1e30)

        # 8) Softmax over keys, then weighted sum of values.
        attn = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        # 9) Merge heads and project to model space.
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = jnp.reshape(
            context, (batch, seqlen, self.num_query_heads * self.head_dim)
        )
        out = self.out_proj(context)
        return out

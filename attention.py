"""
Minimal Grouped-Query Attention (GQA) in JAX/Flax NNX.

This file implements a simple, educational attention block used by the
Nemotron 3 Nano hybrid model:
- Causal self-attention (decoder-only masking)
- Grouped-query attention (more query heads than KV heads)
- No positional embeddings
- No dropout
- No bias on linear projections

The implementation prioritizes readability over speed.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from cache import KVCache

# =============================================================================
# Attention Block
# =============================================================================


class GroupedQueryAttention(nnx.Module):
    """
    Minimal causal self-attention with grouped-query heads.

    Nemotron 3 Nano design choices:
    - Query heads and KV heads can be different (GQA)
    - No positional embeddings
    - No dropout
    - No bias on linear projections

    LRM note: Attention is placed in select hybrid layers (not every layer) to
    give the model targeted retrieval capability inside long reasoning traces.
    A reasoning model often needs to reference a result it derived several
    paragraphs earlier (e.g., "as shown in step 3..."). The causal mask ensures
    the model can only attend to prior tokens — it cannot peek at future steps,
    which matches how a human writes a step-by-step solution left-to-right.
    GQA reduces the KV cache footprint during long-trace generation, making
    inference cheaper when reasoning traces span thousands of tokens.

    Args:
        d_model: Hidden/model dimension.
        num_query_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head channel dimension.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        assert self.num_query_heads % self.num_kv_heads == 0, (
            "num_query_heads must be divisible by num_kv_heads for GQA"
        )
        assert self.num_query_heads * self.head_dim == self.d_model, (
            "d_model must equal num_query_heads * head_dim"
        )

        # Project input tokens into Q, K, and V spaces.
        self.q_proj = nnx.Linear(
            self.d_model,
            self.num_query_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )

        # Project attention output back to model dimension.
        self.out_proj = nnx.Linear(
            self.num_query_heads * self.head_dim,
            self.d_model,
            use_bias=False,
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

        # 3) Expand KV heads so each query head can attend to a matching KV head.
        kv_repeat = self.num_query_heads // self.num_kv_heads
        k = jnp.repeat(k, kv_repeat, axis=2)
        v = jnp.repeat(v, kv_repeat, axis=2)

        # 4) Move heads before sequence for standard attention math.
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, heads, seqlen, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # 5) Dot-product attention scores.
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # 6) Causal mask: position i cannot see future position j > i.
        # LRM note: This mask is essential for reasoning models. During training
        # the full thinking trace is present in the sequence, so position i
        # (a later reasoning step) can freely attend to all prior steps j < i.
        # This allows the model to build on earlier derivations and catch errors
        # by comparing a current result against something stated earlier.
        # Positions cannot attend to the future, preserving the autoregressive
        # property needed for left-to-right generation at inference time.
        causal_mask = jnp.tril(jnp.ones((seqlen, seqlen), dtype=bool))
        scores = jnp.where(causal_mask[None, None, :, :], scores, -1e30)

        # 7) Softmax over keys, then weighted sum of values.
        attn = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        # 8) Merge heads and project to model space.
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = jnp.reshape(
            context, (batch, seqlen, self.num_query_heads * self.head_dim)
        )
        out = self.out_proj(context)
        return out

    def step(self, x: jax.Array, kv_cache: KVCache) -> tuple[jax.Array, KVCache]:
        """
        Single-token causal attention step with KV cache.

        Computes Q/K/V for one new token, appends K/V to the cache, then attends
        over all cached positions (causal mask keeps only valid filled slots).

        Args:
            x:        New token hidden state, shape (batch, d_model).
            kv_cache: KVCache carrying accumulated keys/values from prior tokens.

        Returns:
            out:       Output tensor, shape (batch, d_model).
            new_cache: Updated KVCache with the new token appended.
        """
        batch = x.shape[0]

        # Project the new single token.
        q = self.q_proj(x)  # (batch, num_query_heads * head_dim)
        k = self.k_proj(x)  # (batch, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, num_kv_heads * head_dim)

        q = jnp.reshape(q, (batch, self.num_query_heads, self.head_dim))
        k = jnp.reshape(k, (batch, self.num_kv_heads, self.head_dim))
        v = jnp.reshape(v, (batch, self.num_kv_heads, self.head_dim))

        # Append new K, V to the cache at the current fill position.
        pos = kv_cache.length
        new_k = jax.lax.dynamic_update_slice(
            kv_cache.k, k[:, :, None, :], (0, 0, pos, 0)
        )
        new_v = jax.lax.dynamic_update_slice(
            kv_cache.v, v[:, :, None, :], (0, 0, pos, 0)
        )

        # GQA: expand KV heads to match the number of query heads.
        kv_repeat = self.num_query_heads // self.num_kv_heads
        k_full = jnp.repeat(new_k, kv_repeat, axis=1)  # (batch, num_query_heads, max_len, head_dim)
        v_full = jnp.repeat(new_v, kv_repeat, axis=1)

        # q: (batch, num_query_heads, 1, head_dim) — single query token.
        q = q[:, :, None, :]
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_full) * scale  # (batch, h, 1, max_len)

        # Causal mask: only attend to positions 0..pos (pos is the newly written slot).
        max_len = kv_cache.k.shape[2]
        valid = jnp.arange(max_len)[None, None, None, :] <= pos  # (1, 1, 1, max_len)
        scores = jnp.where(valid, scores, -1e30)

        attn = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v_full)  # (batch, h, 1, head_dim)
        context = jnp.reshape(
            context[:, :, 0, :], (batch, self.num_query_heads * self.head_dim)
        )
        out = self.out_proj(context)
        return out, KVCache(k=new_k, v=new_v, length=pos + 1)


# =============================================================================
# Dot-Product Attention Block (uses jax.nn.dot_product_attention)
# =============================================================================


class DotProductGroupedQueryAttention(nnx.Module):
    """
    Causal GQA using jax.nn.dot_product_attention as the attention kernel.

    Identical in behaviour to GroupedQueryAttention but delegates the
    softmax + weighted-sum step to jax.nn.dot_product_attention, which XLA
    can lower to an optimised fused kernel (e.g. Flash Attention) on supported
    hardware. Requires JAX >= 0.4.20.

    Args:
        d_model: Hidden/model dimension.
        num_query_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head channel dimension.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        assert self.num_query_heads % self.num_kv_heads == 0, (
            "num_query_heads must be divisible by num_kv_heads for GQA"
        )
        assert self.num_query_heads * self.head_dim == self.d_model, (
            "d_model must equal num_query_heads * head_dim"
        )

        self.q_proj = nnx.Linear(
            self.d_model,
            self.num_query_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            self.num_query_heads * self.head_dim,
            self.d_model,
            use_bias=False,
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

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # jax.nn.dot_product_attention expects (batch, seq, heads, head_dim).
        q = jnp.reshape(q, (batch, seqlen, self.num_query_heads, self.head_dim))
        k = jnp.reshape(k, (batch, seqlen, self.num_kv_heads, self.head_dim))
        v = jnp.reshape(v, (batch, seqlen, self.num_kv_heads, self.head_dim))

        # Expand KV heads to match query heads for GQA.
        kv_repeat = self.num_query_heads // self.num_kv_heads
        k = jnp.repeat(k, kv_repeat, axis=2)
        v = jnp.repeat(v, kv_repeat, axis=2)

        # Fused causal attention; scale is applied internally.
        context = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        # context: (batch, seqlen, num_query_heads, head_dim)

        context = jnp.reshape(
            context, (batch, seqlen, self.num_query_heads * self.head_dim)
        )
        return self.out_proj(context)

    def step_chached(self, x: jax.Array, kv_cache: KVCache) -> tuple[jax.Array, KVCache]:
        """
        Single-token causal attention step with KV cache.

        Args:
            x:        New token hidden state, shape (batch, d_model).
            kv_cache: KVCache carrying accumulated keys/values from prior tokens.

        Returns:
            out:       Output tensor, shape (batch, d_model).
            new_cache: Updated KVCache with the new token appended.
        """
        batch = x.shape[0]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = jnp.reshape(q, (batch, self.num_query_heads, self.head_dim))
        k = jnp.reshape(k, (batch, self.num_kv_heads, self.head_dim))
        v = jnp.reshape(v, (batch, self.num_kv_heads, self.head_dim))

        pos = kv_cache.length
        new_k = jax.lax.dynamic_update_slice(
            kv_cache.k, k[:, :, None, :], (0, 0, pos, 0)
        )
        new_v = jax.lax.dynamic_update_slice(
            kv_cache.v, v[:, :, None, :], (0, 0, pos, 0)
        )

        kv_repeat = self.num_query_heads // self.num_kv_heads
        k_full = jnp.repeat(new_k, kv_repeat, axis=1)  # (batch, num_query_heads, max_len, head_dim)
        v_full = jnp.repeat(new_v, kv_repeat, axis=1)

        # Transpose from (batch, heads, seq, head_dim) to (batch, seq, heads, head_dim).
        q_dpa = q[:, None, :, :]                            # (batch, 1, num_query_heads, head_dim)
        k_dpa = jnp.transpose(k_full, (0, 2, 1, 3))        # (batch, max_len, num_query_heads, head_dim)
        v_dpa = jnp.transpose(v_full, (0, 2, 1, 3))        # (batch, max_len, num_query_heads, head_dim)

        # Causal mask: attend only to filled cache slots (0..pos).
        # mask shape must broadcast to (batch, num_heads, q_length, kv_length).
        max_len = kv_cache.k.shape[2]
        valid = jnp.arange(max_len) <= pos                  # (max_len,)
        mask = valid[None, None, None, :]                   # (1, 1, 1, max_len)

        context = jax.nn.dot_product_attention(q_dpa, k_dpa, v_dpa, mask=mask)
        # context: (batch, 1, num_query_heads, head_dim)

        context = jnp.reshape(
            context[:, 0, :, :], (batch, self.num_query_heads * self.head_dim)
        )
        out = self.out_proj(context)
        return out, KVCache(k=new_k, v=new_v, length=pos + 1)

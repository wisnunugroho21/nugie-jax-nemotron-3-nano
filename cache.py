"""
KV Cache and SSM State Cache for efficient autoregressive generation.

KVCache       — caches key/value tensors for GroupedQueryAttention layers.
SSMCache      — caches the SSM hidden state and causal conv buffer for Mamba2Block.
NemotronCache — bundles all per-layer caches for NemotronNanoBlock.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


class KVCache(NamedTuple):
    """
    Key-Value cache for one GroupedQueryAttention layer.

    Attributes:
        k:      Cached key tensors,   shape (batch, num_kv_heads, max_len, head_dim).
        v:      Cached value tensors, shape (batch, num_kv_heads, max_len, head_dim).
        length: Scalar int32 — number of tokens stored so far (fill pointer).
    """

    k: jax.Array
    v: jax.Array
    length: jax.Array


class SSMCache(NamedTuple):
    """
    State cache for one Mamba2Block.

    Attributes:
        h:        Running SSM hidden state h[t], shape (batch, nheads, headdim, d_state).
                  Updated at each step via the recurrence h[t] = A_bar[t]*h[t-1] + B[t]⊗X[t].
        conv_buf: Sliding causal conv input window, shape (batch, d_conv - 1, conv_dim).
                  Holds the last (d_conv - 1) inputs so the depthwise conv can be
                  applied without re-processing prior tokens.
    """

    h: jax.Array
    conv_buf: jax.Array


@dataclass
class NemotronCache:
    """
    Full per-step cache for NemotronNanoBlock.

    Attributes:
        ssm_caches: One SSMCache per block (indexed by block position).
        kv_caches:  One KVCache per block, or None for Mamba-only blocks without
                    an attention sub-layer.
    """

    ssm_caches: list[SSMCache]
    kv_caches: list[KVCache | None]

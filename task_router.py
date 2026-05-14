"""
Task routing and hybrid loss helpers for multimodal training.

This module is intentionally small and framework-light so it can be reused in
both quick experiments and fuller training loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax


@dataclass
class HybridBatch:
    """Unified batch container for text, VLM, and VLA tasks."""

    token_ids: jax.Array
    text_labels: jax.Array
    text_loss_mask: jax.Array | None = None
    pixel_values: jax.Array | None = None
    action_labels: jax.Array | None = None
    action_positions: jax.Array | None = None
    task_type: str = "text"


def masked_cross_entropy_loss(
    logits: jax.Array,
    labels: jax.Array,
    loss_mask: jax.Array | None = None,
) -> jax.Array:
    """Cross-entropy with optional token-level mask."""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    losses = optax.softmax_cross_entropy(logits, one_hot)

    if loss_mask is None:
        return losses.mean()

    mask = loss_mask.astype(losses.dtype)
    masked_sum = jnp.sum(losses * mask)
    denom = jnp.sum(mask)
    return jnp.where(denom > 0, masked_sum / denom, losses.mean())


def classification_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Per-sample action classification cross-entropy."""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    losses = optax.softmax_cross_entropy(logits, one_hot)
    return losses.mean()


def compute_hybrid_loss(
    text_logits: jax.Array,
    text_labels: jax.Array,
    text_loss_mask: jax.Array | None = None,
    action_logits: jax.Array | None = None,
    action_labels: jax.Array | None = None,
    text_loss_weight: float = 1.0,
    action_loss_weight: float = 1.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """
    Computes weighted text and optional action losses.

    Returns:
    - total_loss
    - metrics dict with component losses
    """
    text_loss = masked_cross_entropy_loss(
        logits=text_logits,
        labels=text_labels,
        loss_mask=text_loss_mask,
    )

    total_loss = text_loss_weight * text_loss
    metrics: dict[str, jax.Array] = {
        "text_loss": text_loss,
        "total_loss": total_loss,
    }

    if action_logits is not None or action_labels is not None:
        if action_logits is None or action_labels is None:
            raise ValueError("action_logits and action_labels must be provided together")

        action_loss = classification_loss(action_logits, action_labels)
        total_loss = total_loss + action_loss_weight * action_loss
        metrics["action_loss"] = action_loss
        metrics["total_loss"] = total_loss

    return total_loss, metrics

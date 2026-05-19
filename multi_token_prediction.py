"""
Multi-Token Prediction (MTP) for Nemotron 3 Super in JAX/Flax NNX.

Based on "Multi-Token Prediction" (Gloeckle et al., 2024) and DeepSeek-V3
(DeepSeek-AI, 2024), as incorporated in Nemotron 3 Super (arXiv:2604.12374, §2.5).

Standard language model training predicts one token ahead:
  Main model: hidden state at position n → predicts token n+1

MTP adds auxiliary heads that predict further into the future:
  MTP head 1: hidden state at position n → predicts token n+2
  MTP head 2: hidden state at position n → predicts token n+3
  ... for num_heads total extra prediction depths.

Each head uses teacher forcing during training: it receives the GROUND TRUTH
token at the intermediate positions as input, not the model's own prediction.
This ensures stable gradients — without teacher forcing, early errors would
cascade through all depths and produce an unlearnable training signal.

Why shared weights across all depths?
  At inference, the MTP head can act as a speculative decoder: it is applied
  recursively, with its OWN previous output hidden state as the next input.
  This creates a distribution mismatch with training (teacher forcing vs. self-
  generated states). Sharing weights across all training depths (k=1, k=2, ...)
  exposes the single head module to many offsets and makes it more robust to
  this mismatch. A single head trained on multiple horizons generalizes better
  to the self-generated distribution encountered at inference than N separate
  heads each trained on only one horizon.

Why a 0.3 loss scale?
  The MTP objective is auxiliary — it should improve representations without
  dominating the training signal. A scale of 0.3 keeps MTP contributions small
  relative to the main next-token cross-entropy. If MTP loss were unscaled, it
  would receive equal weight to the main objective despite being harder to
  optimize (predicting 2–3 tokens ahead is more uncertain).

Speculative decoding sketch (inference):
  1. The main model processes the prompt and generates token n+1 normally.
  2. Simultaneously, the MTP head drafts tokens n+2 and n+3.
  3. A fast verification step checks whether the drafts match what the main
     model would generate for those positions.
  4. If they match (often the case for predictable continuations), 2–3 tokens
     are accepted in one forward pass instead of running the model three times.

Training batch layout for 2 MTP heads:
  The training batch must contain T + num_heads + 1 tokens per sequence:
    - positions 0..T-1  → main model inputs
    - positions 1..T    → main model labels  (standard next-token)
    - positions 2..T+1  → MTP head 0 labels (predict 2 ahead)
    - positions 3..T+2  → MTP head 1 labels (predict 3 ahead)
  This is naturally handled by building batches of shape (B, T + num_heads + 1)
  and passing batch[:, 1:] as extended_token_ids to MultiTokenPrediction.__call__.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from mamba_2 import Mamba2Block


# =============================================================================
# MTP Head (single shared-weight module)
# =============================================================================


class MTPHead(nnx.Module):
    """
    A single shared-weight Multi-Token Prediction head.

    At each depth step k (k=0 → predicts n+2, k=1 → predicts n+3, etc.) this
    module takes two inputs:
      - h_prev:  the hidden state from the previous module at each position.
                 At k=0 this is the main model's final hidden state.
                 At k=1 it is the h_out returned by this module at k=0.
      - tok_emb: the embedding of the GROUND TRUTH token at position n+k+1
                 (teacher forcing). This gives the head a "peek" at the correct
                 intermediate token, stabilizing the gradient signal at depth 2+.

    The two inputs are normalized independently (their scales can differ widely),
    concatenated along the feature axis, and projected back to d_model before
    the Mamba-2 processing block.

    Architecture (per depth step):
        h_norm  = RMSNorm(h_prev)
        e_norm  = RMSNorm(tok_emb)
        fused   = input_proj( concat([h_norm, e_norm], dim=-1) )   # 2d → d
        h_out   = fused + block( block_norm(fused) )               # Mamba residual
        logits  = lm_head( out_norm(h_out) )                       # d → vocab_size
        return h_out, logits

    h_out is fed into the NEXT depth's call of this same module (shared weights).
    logits are used for the MTP cross-entropy loss at this depth.

    Args:
        d_model: Model hidden dimension.
        vocab_size: Vocabulary size for the logit projection.
        rms_norm_eps: Epsilon for RMSNorm numerical stability.
        mamba_*: Mamba-2 block hyperparameters — should match the main model
                 so the MTP processing block has the same inductive biases.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        vocab_size: int,
        rms_norm_eps: float,
        # Mamba-2 block hyperparameters (should match main model config)
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mamba_headdim: int,
        mamba_ngroups: int,
        mamba_chunk_size: int,
    ):
        # Normalize h_prev and tok_emb independently before fusing.
        # They inhabit different "spaces" — one is the output of a deep SSM stack,
        # the other is a raw token embedding — so separate norms prevent one from
        # drowning out the other when they are concatenated.
        self.h_norm = nnx.RMSNorm(d_model, epsilon=rms_norm_eps, rngs=rngs)
        self.emb_norm = nnx.RMSNorm(d_model, epsilon=rms_norm_eps, rngs=rngs)

        # Fuse the two d-dimensional inputs into one d-dimensional representation.
        # Concatenation (not addition) is used so the linear layer can learn
        # independently how much weight to assign each input before projecting.
        self.input_proj = nnx.Linear(
            2 * d_model, d_model, use_bias=False, rngs=rngs
        )

        # Pre-norm for the Mamba processing block — matches the main model's
        # pre-norm residual convention (x + block(norm(x))).
        self.block_norm = nnx.RMSNorm(d_model, epsilon=rms_norm_eps, rngs=rngs)

        # A single Mamba-2 block for sequence-level processing.
        # Using Mamba is consistent with the main hybrid model: the MTP head
        # processes the same type of long, causally-structured representations
        # that Mamba excels at. Using attention here would be inconsistent with
        # the rest of the architecture and would add unnecessary complexity.
        self.block = Mamba2Block(
            rngs=rngs,
            d_model=d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            ngroups=mamba_ngroups,
            chunk_size=mamba_chunk_size,
        )

        # Final norm before the vocabulary projection — standard in decoder LMs.
        self.out_norm = nnx.RMSNorm(d_model, epsilon=rms_norm_eps, rngs=rngs)

        # Vocabulary projection (untied from main model's lm_head).
        # Tying weights would force the MTP head to share the same output
        # representation as the main model, constraining quality at deeper horizons
        # where the target distribution differs more from the main objective.
        self.lm_head = nnx.Linear(d_model, vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        h_prev: jax.Array,
        tok_emb: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        One depth step of Multi-Token Prediction.

        Args:
            h_prev:  (batch, seq, d_model)
                Hidden state from the previous module:
                  - At depth k=0, this is the main model's final hidden state
                    (before the main model's final RMSNorm — see NemotronSuperBlock).
                  - At depth k>0, this is h_out returned by this call at k-1.
            tok_emb: (batch, seq, d_model)
                Embedding of the ground-truth token at position n + k + 1
                (teacher forcing during training).

        Returns:
            h_out:  (batch, seq, d_model)  — hidden state for the next depth.
            logits: (batch, seq, vocab_size) — predictions for position n + k + 2.
        """
        # Normalize both inputs independently before fusing.
        h_normed = self.h_norm(h_prev)     # (batch, seq, d_model)
        e_normed = self.emb_norm(tok_emb)  # (batch, seq, d_model)

        # Concatenate along the feature dimension, then project down to d_model.
        fused = self.input_proj(
            jnp.concatenate([h_normed, e_normed], axis=-1)
        )  # (batch, seq, d_model)

        # Mamba residual: fused + block(RMSNorm(fused)).
        # This pre-norm residual pattern matches the main model blocks exactly.
        h_out = fused + self.block(self.block_norm(fused))  # (batch, seq, d_model)

        # Project to vocabulary logits via a final normalization.
        logits = self.lm_head(self.out_norm(h_out))  # (batch, seq, vocab_size)

        return h_out, logits


# =============================================================================
# Multi-Token Prediction (shared-weight wrapper)
# =============================================================================


class MultiTokenPrediction(nnx.Module):
    """
    Multi-Token Prediction with shared weights across all prediction depths.

    Holds a SINGLE MTPHead instance that is called once per depth during both
    training and inference. This is the "shared weight" design described in the
    module docstring above — one head trained on multiple horizons generalizes
    better than N independent heads each trained on a single horizon.

    Nemotron 3 Super uses num_heads=2, so the model jointly trains to predict
    two tokens ahead in addition to the standard next-token objective.

    Training usage (called from NemotronSuperBlock.forward_train):

        main_logits, main_labels, mtp_outputs = model.forward_train(batch)

        main_loss = optax.softmax_cross_entropy_with_integer_labels(
            main_logits, main_labels
        ).mean()
        aux_loss  = mtp_loss(mtp_outputs, scale=config.mtp_loss_scale)
        total_loss = main_loss + aux_loss

    Args:
        d_model, vocab_size, rms_norm_eps, mamba_*: Forwarded directly to MTPHead.
        num_heads: Number of extra prediction depths (2 for Nemotron 3 Super).
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        vocab_size: int,
        rms_norm_eps: float,
        num_heads: int,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mamba_headdim: int,
        mamba_ngroups: int,
        mamba_chunk_size: int,
    ):
        self.num_heads = num_heads

        # ONE head module — shared across all depths.
        # During __call__, this same module is called num_heads times with
        # different inputs, so it trains on many (hidden-state, token) pairs
        # spanning different prediction horizons.
        self.head = MTPHead(
            rngs=rngs,
            d_model=d_model,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_headdim=mamba_headdim,
            mamba_ngroups=mamba_ngroups,
            mamba_chunk_size=mamba_chunk_size,
        )

    def __call__(
        self,
        main_hidden: jax.Array,
        extended_token_ids: jax.Array,
        embedder: nnx.Embed,
    ) -> list[tuple[jax.Array, jax.Array]]:
        """
        Compute MTP predictions for all depths using teacher forcing.

        Extended token ID layout
        ────────────────────────
        If main model input is positions 0..T-1, then extended_token_ids holds:
          index k   → token at position k+1 (relative to main model start)
        Concretely, if the full training batch is (B, T + num_heads + 1):
          main model inputs    = batch[:, :T]
          extended_token_ids   = batch[:, 1:]      shape (B, T + num_heads)

        At depth k (0-indexed):
          teacher forcing input: extended_token_ids[:, k : k+T]
            → tokens at positions k+1 .. k+T  (ground-truth intermediate tokens)
          MTP prediction target: extended_token_ids[:, k+1 : k+1+T]
            → tokens at positions k+2 .. k+T+1  (what the head should predict)

        The SAME head module (shared weights) is called for every k.

        Args:
            main_hidden: (batch, T, d_model)
                Final hidden states of the main model before final_norm.
                Using pre-norm hidden states lets the MTP head apply its own
                normalization (h_norm) and process a richer, unnormalized signal.
            extended_token_ids: (batch, T + num_heads)
                Token IDs for teacher forcing and MTP label extraction.
            embedder: The main model's nnx.Embed layer.
                Shared embedding table — MTP uses the same token vectors as the
                main model to keep the input representation consistent.

        Returns:
            List of (logits, labels) tuples, one per depth k.
              logits: (batch, T, vocab_size)
              labels: (batch, T)  — integer token IDs, already aligned to logits.
        """
        T = main_hidden.shape[1]
        outputs = []
        h = main_hidden  # updated to this head's output after each depth

        for k in range(self.num_heads):
            # Teacher forcing: the ground-truth token k+1 positions ahead.
            # These are the tokens the head should have "seen" at depth k, as if
            # reasoning were perfect up to position k.
            teacher_ids = extended_token_ids[:, k : k + T]  # (batch, T)
            tok_emb = embedder(teacher_ids)                  # (batch, T, d_model)

            # Call the SAME head module — shared weights for every depth k.
            h, logits = self.head(h, tok_emb)               # (batch, T, vocab/d)

            # Labels: the ground-truth token k+2 positions ahead.
            # These are what the head must predict at this depth.
            labels = extended_token_ids[:, k + 1 : k + 1 + T]  # (batch, T)

            outputs.append((logits, labels))

        return outputs  # list of (logits, labels), length = num_heads


# =============================================================================
# MTP Auxiliary Loss
# =============================================================================


def mtp_loss(
    mtp_outputs: list[tuple[jax.Array, jax.Array]],
    scale: float = 0.3,
) -> jax.Array:
    """
    Compute the scaled MTP auxiliary loss.

    For each depth k, computes cross-entropy of the head's predictions against
    the corresponding ground-truth labels, averages across depths, then scales
    by `scale`.

    The scaled result is intended to be ADDED to the main next-token loss:
        total_loss = main_loss + mtp_loss(mtp_outputs, scale=0.3)

    The 0.3 default scale keeps the MTP contribution auxiliary.  Predicting
    2–3 tokens ahead is inherently harder and more uncertain than predicting 1
    token ahead, so it should not receive the same weight as the main objective.

    Args:
        mtp_outputs: List of (logits, labels) tuples from MultiTokenPrediction.__call__.
            logits: (batch, seq, vocab_size)
            labels: (batch, seq)  — integer token IDs
        scale: Loss scaling factor (0.3 per Nemotron 3 Super §2.5).

    Returns:
        Scalar loss: mean cross-entropy across depths, multiplied by scale.
    """
    if not mtp_outputs:
        return jnp.array(0.0, dtype=jnp.float32)

    per_depth_losses = [
        optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        for logits, labels in mtp_outputs
    ]
    return jnp.mean(jnp.stack(per_depth_losses)) * scale

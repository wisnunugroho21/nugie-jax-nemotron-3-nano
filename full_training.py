"""
full_training.py — Complete Training Pipeline for Nemotron 3 Nano

This file implements the full training recipe described in:
  "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid
   Mamba-Transformer Model for Agentic Reasoning"
  https://arxiv.org/abs/2512.20848

Pipeline Overview
-----------------
The training is divided into two major phases:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                          PRE-TRAINING                                   │
  │                                                                         │
  │  Phase 1 — Diverse data   : 23.5 T tokens  (94% of total pre-training) │
  │  Phase 2 — High-quality   :  1.5 T tokens  (final 6%)                  │
  │  LC-Phase — Long context  :  121 B tokens  (continuous fine-tuning)     │
  │                                                                         │
  │  Optimizer  : AdamW  β₁=0.9, β₂=0.95, weight_decay=0.1                │
  │  LR schedule: Warmup → Stable → Cosine-Decay  (WSD)                    │
  │  MoE        : aux-loss-free load balancing + standard balance loss      │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         POST-TRAINING                                   │
  │                                                                         │
  │  Step 1 — SFT   : Supervised Fine-Tuning on diverse chat/agentic data  │
  │  Step 2 — RLVR  : Multi-environment RL from Verifiable Rewards (GRPO)  │
  │  Step 3 — RLHF  : RL from Human Feedback via a GenRM judge             │
  │  Step 4 — RLVR  : Second RLVR pass after RLHF for further refinement   │
  └─────────────────────────────────────────────────────────────────────────┘

Code style
----------
- Each section is self-contained and clearly labelled.
- All paper-specific constants are named and annotated.
- Simplifications for local runs are explicitly noted.
- We follow the same JAX/Flax NNX conventions as pretrained.py,
  finetune_cot.py, and finetune_cot_rl.py in this repo.

How to run
----------
  python full_training.py

  The script runs all phases in sequence, saving a checkpoint between each.
  You can safely comment out any phase(s) in main() to run a subset.

Paper reference
---------------
  NVIDIA (2025). "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts
  Hybrid Mamba-Transformer Model for Agentic Reasoning."
  arXiv:2512.20848.
"""

import math
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from datasets import load_dataset
from flax import nnx
from transformers import AutoTokenizer

from moe import SparseMoE
from nemotron import NemotronConfig, NemotronNanoBlock


# =============================================================================
# Shared constants (tokenizer & architecture)
# =============================================================================

# Vocabulary size of the official Nemotron tokenizer on HuggingFace.
VOCAB_SIZE = 131_072

# Mamba SSD kernel requires seq_len % chunk_size == 0.
# This must stay in sync with NemotronConfig.mamba_chunk_size.
CHUNK_SIZE = 64


# =============================================================================
# PRE-TRAINING Hyperparameters
# =============================================================================

# ── Sequence & batch ────────────────────────────────────────────────────────
# Paper §2.4: sequence length 8 192, batch size 3 072  ≈ 25 M tokens/batch.
# We use small local values; set to paper values for a full run.
PRETRAIN_SEQ_LEN   = 256   # paper: 8 192 (must be divisible by CHUNK_SIZE)
PRETRAIN_BATCH     = 2     # paper: 3 072

# ── Token budgets ─────────────────────────────────────────────────────────
# Paper §2.3: Phase 1 = 23.5 T diverse tokens; Phase 2 = 1.5 T high-quality;
# LC-Phase = 121 B long-context tokens.
# We express budgets as training step counts for a local demo.
PHASE1_STEPS       = 5_000   # paper: ~23.5 T / 25 M  ≈ 940 000 steps
PHASE2_STEPS       = 500     # paper: ~1.5 T / 25 M   ≈  60 000 steps
LC_PHASE_STEPS     = 100     # paper: ~121 B / 25 M   ≈   4 840 steps

# ── WSD Learning-Rate Schedule ─────────────────────────────────────────────
# Paper §2.4: Warmup-Stable-Decay (WSD) schedule.
#   Warmup  : linear ramp from 0 → PRETRAIN_PEAK_LR over 8.4 B tokens
#   Stable  : constant at PRETRAIN_PEAK_LR for 80% of training (20 T tokens)
#   Decay   : cosine decay to PRETRAIN_MIN_LR over final 20% (5 T tokens)
PRETRAIN_PEAK_LR   = 1e-3   # paper: 1e-3
PRETRAIN_MIN_LR    = 1e-5   # paper: 1e-5
PRETRAIN_WARMUP_STEPS  = max(1, int((PHASE1_STEPS + PHASE2_STEPS) * 0.05))
PRETRAIN_STABLE_STEPS  = max(1, int((PHASE1_STEPS + PHASE2_STEPS) * 0.75))
PRETRAIN_DECAY_STEPS   = max(1, int((PHASE1_STEPS + PHASE2_STEPS) * 0.20))

# ── Long-context extension (LC-Phase) ─────────────────────────────────────
# Paper §2.5: constant LR, mixture of 512k and 4k sequences.
# Simplified here: constant LR on a shorter context.
LC_PHASE_LR        = 1e-5   # paper: 1e-5 (constant)
LC_SEQ_LEN         = 512    # paper: mix of 512k and 4k; use short for demo

# ── AdamW optimiser ────────────────────────────────────────────────────────
# Paper §2.4: AdamW with β₁=0.9, β₂=0.95, weight_decay=0.1.
PRETRAIN_B1        = 0.9
PRETRAIN_B2        = 0.95
PRETRAIN_WD        = 0.1

# ── MoE load balancing ─────────────────────────────────────────────────────
# Paper §2.4: aux-loss-free bias update (rate 1e-3) AND standard balance loss
# (coefficient 1e-4).  Both are applied simultaneously during pretraining.
AUX_LOSS_COEFF     = 1e-4   # weight of the standard load-balancing loss term

# ── Checkpointing ──────────────────────────────────────────────────────────
PRETRAIN_CKPT_DIR  = "./checkpoints_pretrain"
PRETRAIN_CKPT_EVERY = 200   # save every N training steps

# ── Evaluation ─────────────────────────────────────────────────────────────
PRETRAIN_VAL_STEPS = 30     # batches used to estimate validation loss


# =============================================================================
# POST-TRAINING — SFT Hyperparameters
# =============================================================================

# Paper §3.1.6: 13 000 steps, batch 64, sequence packing to 256 K, LR 5e-5,
# 800 warmup steps, MoE load-balance loss coefficient 1e-4.

SFT_SEQ_LEN      = 256   # paper: 256 K (sequence packing; local demo uses 256)
SFT_BATCH        = 2     # paper: 64
SFT_STEPS        = 300   # paper: 13 000
SFT_LR           = 5e-5  # paper: 5e-5
SFT_MIN_LR       = 1e-7
SFT_WARMUP_STEPS = max(1, int(SFT_STEPS * 0.06))  # paper: 800 of 13 000 ≈ 6%
SFT_WD           = 0.1
SFT_B1           = 0.9
SFT_B2           = 0.95

# Paper §3.1.5: to enable "reasoning on/off" we randomly strip thinking traces
# from 10% of training samples.  To enable "budget control" we randomly
# truncate 3% of traces to a shorter length.
SFT_REASONING_OFF_PROB  = 0.10   # fraction of samples that lose <think> tokens
SFT_BUDGET_CONTROL_PROB = 0.03   # fraction of samples with truncated traces

SFT_CKPT_DIR     = "./checkpoints_sft"
SFT_CKPT_EVERY   = 100


# =============================================================================
# POST-TRAINING — RLVR Hyperparameters
# =============================================================================

# Paper §3.2.5: GRPO with 128 prompts × 16 generations per step,
# effective batch size 2 048.  We use smaller values for a local demo.

RLVR_NUM_PROMPTS     = 4    # paper: 128
RLVR_NUM_GENERATIONS = 4    # paper: 16  — completions per prompt
RLVR_STEPS           = 100  # number of GRPO gradient updates

# The KL coefficient β prevents the policy from drifting too far from the
# SFT checkpoint (the "reference model").
RLVR_KL_COEFF        = 0.04   # same as finetune_cot_rl.py
RLVR_CLIP_EPS        = 0.2    # PPO/GRPO clipping epsilon (paper §3.2.5)

# Paper §3.2.5: max generation length 49 K; overlong filtering is applied.
RLVR_MAX_NEW_TOKENS  = 150    # paper: 49 000 (reduced for demo)

# Sampling temperature during rollout generation.
RLVR_TEMPERATURE     = 0.8

# Paper §3.2.5: freeze MoE router weights during RLVR to stabilise training.
RLVR_FREEZE_ROUTER   = True

RLVR_LR              = 1e-5
RLVR_MIN_LR          = 1e-7
RLVR_WD              = 0.1
RLVR_B1              = 0.9
RLVR_B2              = 0.95

RLVR_CKPT_DIR        = "./checkpoints_rlvr"
RLVR_CKPT_EVERY      = 50

# Max cache length must exceed prompt length + RLVR_MAX_NEW_TOKENS.
RLVR_MAX_CACHE_LEN   = SFT_SEQ_LEN + RLVR_MAX_NEW_TOKENS + 64

# RL losses use token_ids[:, :-1] as model inputs. To keep that input length
# divisible by CHUNK_SIZE for Mamba, the packed RL sequence length is +1.
RL_TRAIN_SEQ_LEN     = SFT_SEQ_LEN + 1


# =============================================================================
# POST-TRAINING — RLHF Hyperparameters
# =============================================================================

# Paper §3.3.2: RLHF uses the same 128 prompts × 16 responses as RLVR.
RLHF_NUM_PROMPTS     = 4    # paper: 128
RLHF_NUM_RESPONSES   = 4    # paper: 16

# Paper §3.3.2, Eq. (6): Group Relative Length Control coefficients.
# λ_think and λ_answer penalise long reasoning / answer sections relative to
# the shortest response in the group.
RLHF_LAMBDA_THINK    = 0.5   # paper: 0.5
RLHF_LAMBDA_ANSWER   = 0.5   # paper: 0.5

# Paper §3.3.2: Quality-Gated Conciseness Bonus.
# An extra bonus is given to the *shortest* response that still achieves a
# score at or above the τ_p percentile threshold within the group.
RLHF_BETA_THINK      = 0.5   # paper: 0.5
RLHF_BETA_ANSWER     = 0.5   # paper: 0.5
RLHF_PERCENTILE      = 80    # paper: 80-th percentile

RLHF_LR              = 1e-5
RLHF_MIN_LR          = 1e-7
RLHF_WD              = 0.1
RLHF_B1              = 0.9
RLHF_B2              = 0.95
RLHF_STEPS           = 50

RLHF_CKPT_DIR        = "./checkpoints_rlhf"
RLHF_CKPT_EVERY      = 25
RLHF_MAX_CACHE_LEN   = SFT_SEQ_LEN + 200 + 64


# =============================================================================
# Sanity checks
# =============================================================================

assert PRETRAIN_SEQ_LEN % CHUNK_SIZE == 0, "PRETRAIN_SEQ_LEN must divide CHUNK_SIZE"
assert LC_SEQ_LEN       % CHUNK_SIZE == 0, "LC_SEQ_LEN must divide CHUNK_SIZE"
assert SFT_SEQ_LEN      % CHUNK_SIZE == 0, "SFT_SEQ_LEN must divide CHUNK_SIZE"
assert (RL_TRAIN_SEQ_LEN - 1) % CHUNK_SIZE == 0, "RL_TRAIN_SEQ_LEN - 1 must divide CHUNK_SIZE"


# =============================================================================
# 1. Model helpers
# =============================================================================


def build_model(seed: int = 0) -> NemotronNanoBlock:
    """Construct a tiny Nemotron model for local experimentation.

    In the paper, the model is a 31.6 B parameter MoE hybrid Mamba-Transformer
    with 128 experts (6 active per token) and 52 layers.  We use the tiny
    preset here so the whole pipeline can be tested on a laptop.

    To train at paper scale, replace NemotronConfig.from_preset("tiny") with
    the full config (d_model=4096, num_experts=128, top_k=6, …).
    """
    config = NemotronConfig.from_preset("tiny")
    # Keep this file robust across config updates: some presets may drift and
    # violate attention shape constraints (d_model != heads * head_dim).
    # We enforce a valid tiny shape here so full_training.py is runnable.
    expected_d_model = config.num_attention_heads * config.attention_head_dim
    if config.d_model != expected_d_model:
        config.d_model = expected_d_model
    config.vocab_size = VOCAB_SIZE
    config.mamba_chunk_size = CHUNK_SIZE
    config.validate()
    return NemotronNanoBlock(rngs=nnx.Rngs(seed), config=config)


def collect_moe_layers(model: NemotronNanoBlock) -> list[SparseMoE]:
    """Return every MoE sub-module in the model (one per block)."""
    return [block.moe for block in model.blocks]


# =============================================================================
# 2. Learning-rate schedules
# =============================================================================


def make_wsd_schedule(
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
) -> optax.Schedule:
    """Build the Warmup-Stable-Decay (WSD) learning-rate schedule.

    Paper §2.4 describes three segments:
      1. Warmup   : linear ramp  0 → peak_lr  over warmup_steps.
      2. Stable   : constant     peak_lr       over stable_steps.
      3. Decay    : cosine decay peak_lr → min_lr over decay_steps.

    The warmup prevents large gradient magnitudes from destabilising the model
    at the very start of training.  The stable plateau keeps the model learning
    at full speed for most of training.  The final cosine decay helps the model
    converge to a sharper minimum.
    """
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )
    stable = optax.constant_schedule(peak_lr)
    decay  = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=decay_steps,
        alpha=min_lr / peak_lr,   # final LR = peak_lr × alpha = min_lr
    )
    return optax.join_schedules(
        schedules=[warmup, stable, decay],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )


def make_adamw_optimizer(
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    weight_decay: float = 0.1,
    b1: float = 0.9,
    b2: float = 0.95,
) -> optax.GradientTransformation:
    """AdamW + gradient clipping + WSD schedule.

    Paper §2.4: AdamW with weight_decay=0.1, β₁=0.9, β₂=0.95.
    Gradient clipping (global norm ≤ 1.0) is standard practice to prevent
    occasional large gradients from destabilising training.
    """
    lr_schedule = make_wsd_schedule(
        peak_lr=peak_lr,
        min_lr=min_lr,
        warmup_steps=max(warmup_steps, 1),
        stable_steps=max(stable_steps, 1),
        decay_steps=max(decay_steps, 1),
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay, b1=b1, b2=b2),
    )


def make_constant_lr_optimizer(
    lr: float,
    weight_decay: float = 0.1,
    b1: float = 0.9,
    b2: float = 0.95,
) -> optax.GradientTransformation:
    """AdamW with a constant learning rate (used for LC-Phase and RLVR/RLHF).

    The LC-Phase (§2.5) and the RL stages use a constant LR rather than a
    full WSD schedule because the model is already well-trained and we only
    want small, targeted updates.
    """
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay, b1=b1, b2=b2),
    )


# =============================================================================
# 3. MoE load-balancing helpers
# =============================================================================


def compute_load_balance_loss(moe_layers: list[SparseMoE]) -> jax.Array:
    """Compute MoE load-balancing auxiliary loss.

    Paper §2.4: load-balancing loss coefficient 1e-4.
    The full paper loss is L_lb = num_experts × Σ_i f_i × P_i where f_i is
    the expert token fraction and P_i is the mean router probability.

    If router probabilities are unavailable, we fall back to a frequency-only
    proxy L_proxy = num_experts × Σ_i f_i².
    """
    total_loss = jnp.zeros(())
    for moe in moe_layers:
        # SparseMoE stores top-k indices as (num_tokens, routed_top_k).
        # Some variants may store (batch, seq, top_k); both are handled here.
        indices = moe.last_topk_indices.get_value()
        if indices is None or indices.size == 0:
            continue
        num_experts = moe.num_routed_experts
        if indices.ndim not in (2, 3):
            raise ValueError(f"Unexpected top-k index shape: {indices.shape}")

        # f_i: fraction of all (token, slot) assignments going to expert i.
        # We count each of the K slots independently (standard MoE convention).
        flat_indices = indices.reshape(-1)
        one_hot = jax.nn.one_hot(flat_indices, num_experts)          # (B*S*K, E)
        f = one_hot.mean(axis=0)                                      # (E,)

        # Standard paper term if router probabilities are available.
        if hasattr(moe, "last_router_probs"):
            router_probs = moe.last_router_probs.get_value()
            if router_probs is not None and router_probs.size > 0:
                P = router_probs.mean(axis=0)
                total_loss = total_loss + num_experts * (f * P).sum()
                continue

        # Fallback proxy when P_i is unavailable.
        total_loss = total_loss + num_experts * (f * f).sum()

    return total_loss / max(len(moe_layers), 1)


def update_moe_biases(moe_layers: list[SparseMoE]) -> None:
    """Apply the aux-loss-free expert bias update (paper §2.4).

    DeepSeek's aux-loss-free strategy increments a per-expert bias by a small
    amount (update_rate) whenever an expert is *underloaded* and decrements it
    when *overloaded*.  The biases are added to router logits at inference time
    to steer tokens toward underloaded experts, without any gradient signal.
    This keeps routing balanced without distorting the main training loss.
    """
    for moe in moe_layers:
        moe.update_expert_bias(moe.last_topk_indices.get_value())


# =============================================================================
# 4. Dataset helpers
# =============================================================================


def load_pretrain_data(
    split: str,
    max_samples: int,
    seq_len: int,
    tokenizer,
    skip: int = 0,
) -> np.ndarray:
    """Stream text from HuggingFaceFW/fineweb-edu and pack into fixed chunks.

    In the full paper:
      - Phase 1 uses a broad mixture (web crawl, code, math, multilingual…)
      - Phase 2 uses high-quality subsets (Wikipedia, curated synthetics…)
    We approximate both with different slices of fineweb-edu for simplicity.
    A production run would swap in the real multi-source data pipelines.

    Returns an array of shape (num_chunks, seq_len + 1) where each row is a
    chunk of (seq_len + 1) consecutive tokens.  During training, row[:seq_len]
    is the input and row[1:] is the target for next-token prediction.
    """
    print(f"  Loading {max_samples} texts from fineweb-edu ({split}, skip={skip}) …")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split=split, streaming=True)
    texts: list[str] = []
    for i, sample in enumerate(ds):
        if i < skip:
            continue
        texts.append(sample["text"])
        if len(texts) >= max_samples:
            break
    print(f"  Got {len(texts)} texts.")

    chunk_len = seq_len + 1
    all_tokens: list[int] = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text))
        all_tokens.append(tokenizer.eos_token_id)   # mark document boundaries

    n = (len(all_tokens) // chunk_len) * chunk_len
    if n == 0:
        raise RuntimeError("Not enough tokens for even one chunk; increase max_samples.")
    return np.array(all_tokens[:n], dtype=np.int32).reshape(-1, chunk_len)


def make_batches(chunks: np.ndarray, batch_size: int):
    """Shuffle chunks once, then yield (batch_size, chunk_len) batches."""
    idx = np.random.permutation(len(chunks))
    chunks = chunks[idx]
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        yield chunks[i : i + batch_size]


def load_sft_data(split: str = "train") -> list[dict]:
    """Load GSM8K as a stand-in for the paper's full SFT dataset.

    Paper §3.1.2 uses a much richer mixture:
      competition math, competition code, conversational tool use, long context,
      formal proofs, multilingual, general chat, instruction following, safety,
      software engineering, science, GenSelect, CUDA, …  (~18 M total samples).

    We use GSM8K here because it is publicly available and demonstrates all
    the key SFT mechanics (loss masking, reasoning traces, format control).
    """
    print(f"  Loading GSM8K ({split}) …")
    ds = load_dataset("gsm8k", "main", split=split)
    samples = [{"question": row["question"], "answer": row["answer"]} for row in ds]
    print(f"  Got {len(samples)} samples.")
    return samples


# =============================================================================
# 5. SFT data formatting and tokenisation
# =============================================================================


def maybe_strip_reasoning(response: str, rng: np.random.Generator) -> str:
    """Randomly remove the <think> block to enable 'reasoning off' mode.

    Paper §3.1.5: 10% of samples have their thinking trace stripped.
    This teaches the model to produce correct answers *without* explicit
    reasoning when the user opts out of the <think> trace.
    """
    if rng.random() < SFT_REASONING_OFF_PROB:
        # Remove the entire <think>…</think> block including whitespace.
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    return response


def maybe_truncate_reasoning(response: str, rng: np.random.Generator) -> str:
    """Randomly shorten the <think> block to teach reasoning budget control.

    Paper §3.1.5: 3% of samples have their reasoning trace truncated to a
    random fraction of its full length.  The model learns that it is valid
    to stop reasoning early when the budget is tight.
    """
    if rng.random() < SFT_BUDGET_CONTROL_PROB:
        think_match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
        if think_match:
            trace = think_match.group(1)
            # Keep only a random prefix of the reasoning trace.
            cut = rng.integers(1, max(len(trace), 2))
            truncated_trace = trace[:cut]
            response = response.replace(
                think_match.group(0),
                f"<think>{truncated_trace}</think>",
            )
    return response


def format_sft_example(
    question: str,
    answer: str,
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Format a GSM8K sample into a (prompt, response) pair.

    Prompt template  : "User: {question}\\nAssistant: "
    Response template: "<think>\\n{step_by_step}\\n</think>\\n{final_answer}"

    After building the base response we apply:
      1. maybe_strip_reasoning()    — removes trace for reasoning-off samples
      2. maybe_truncate_reasoning() — shortens trace for budget-control samples

    The two controls are part of the paper's SFT strategy (§3.1.5) and
    allow users to toggle reasoning at inference time via the chat template.
    """
    parts     = answer.split("####")
    reasoning = parts[0].strip()
    final_ans = parts[-1].strip()

    prompt   = f"User: {question}\nAssistant: "
    response = f"<think>\n{reasoning}\n</think>\n{final_ans}"

    # Apply reasoning control augmentations (§3.1.5).
    response = maybe_strip_reasoning(response, rng)
    response = maybe_truncate_reasoning(response, rng)

    return prompt, response


def tokenize_with_mask(
    tokenizer,
    prompt: str,
    response: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokenize a (prompt, response) pair and build a per-token loss mask.

    Why mask the prompt?
    --------------------
    In SFT we only want the model to learn to generate the *response*.
    The prompt is merely context; we exclude it from the gradient by setting
    its mask entries to 0.  This prevents the model from memorising questions.

    Token layout after a left-shift (next-token prediction):
        input_ids : [p₀ p₁ … p_{N-1}  r₀  r₁  … r_{M-2}]
        labels    : [p₁ p₂ … p_{N-1}  r₀  r₁  … r_{M-1}]
        mask      : [0  0  … 0         1   1   … 1       ]
                     ← N-1 zeros →      ← M ones →

    Returns (input_ids, labels, mask), each of shape (seq_len,).
    """
    prompt_ids   = tokenizer.encode(prompt,   add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    response_ids.append(tokenizer.eos_token_id)

    full_ids  = prompt_ids + response_ids
    input_ids = full_ids[:-1]
    labels    = full_ids[1:]

    # Loss mask: 1.0 for response tokens, 0.0 for prompt tokens.
    mask_start = max(len(prompt_ids) - 1, 0)
    mask = [0.0] * mask_start + [1.0] * (len(input_ids) - mask_start)

    pad_id = tokenizer.eos_token_id

    def pad_or_trunc(seq: list, fill) -> list:
        if len(seq) >= seq_len:
            return seq[:seq_len]
        return seq + [fill] * (seq_len - len(seq))

    return (
        np.array(pad_or_trunc(input_ids, pad_id), dtype=np.int32),
        np.array(pad_or_trunc(labels,    pad_id), dtype=np.int32),
        np.array(pad_or_trunc(mask,      0.0),    dtype=np.float32),
    )


def make_sft_batches(samples: list[dict], tokenizer, batch_size: int, seq_len: int):
    """Tokenize all SFT samples and yield (inputs, labels, mask) batches.

    Reasoning control augmentations (strip / truncate) are applied here with
    a seeded NumPy RNG so results are reproducible across epochs.
    """
    rng = np.random.default_rng(seed=42)

    all_inputs, all_labels, all_masks = [], [], []
    for sample in samples:
        prompt, response = format_sft_example(
            sample["question"], sample["answer"], rng
        )
        inp, lab, msk = tokenize_with_mask(tokenizer, prompt, response, seq_len)
        if msk.sum() == 0:
            continue   # response was fully truncated — skip this sample
        all_inputs.append(inp)
        all_labels.append(lab)
        all_masks.append(msk)

    if not all_inputs:
        raise RuntimeError("No valid SFT samples after tokenisation.")

    idx = np.random.permutation(len(all_inputs))
    stacked_inputs = np.stack(all_inputs)[idx]
    stacked_labels = np.stack(all_labels)[idx]
    stacked_masks  = np.stack(all_masks) [idx]

    for start in range(0, len(stacked_inputs) - batch_size + 1, batch_size):
        end = start + batch_size
        yield (
            jnp.array(stacked_inputs[start:end]),
            jnp.array(stacked_labels[start:end]),
            jnp.array(stacked_masks [start:end]),
        )


# =============================================================================
# 6. Loss functions
# =============================================================================


def pretrain_loss(
    model: NemotronNanoBlock,
    batch: jax.Array,
    moe_layers: list[SparseMoE],
) -> jax.Array:
    """Next-token prediction loss + MoE load-balance auxiliary loss.

    The main loss is standard cross-entropy.
    The auxiliary term (AUX_LOSS_COEFF × L_lb) encourages balanced expert usage
    across MoE layers (paper §2.4, load-balance loss coefficient 1e-4).
    """
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    logits = model(inputs)   # (B, seq_len, vocab_size)

    ce_loss  = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    lb_loss  = compute_load_balance_loss(moe_layers)

    return ce_loss + AUX_LOSS_COEFF * lb_loss


def sft_loss(
    model: NemotronNanoBlock,
    inputs: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
    moe_layers: list[SparseMoE] | None = None,
) -> jax.Array:
    """Masked cross-entropy for supervised fine-tuning.

    Only response tokens (mask == 1.0) contribute to the loss.
    Prompt tokens are excluded so the model learns *how to answer*, not
    to memorise the input questions.

    safe_mean divides by the number of unmasked tokens, not the total,
    so batches with many short responses get the same effective learning
    signal as batches with long responses.
    """
    logits = model(inputs)   # (B, seq_len, vocab_size)
    ce     = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    masked_sum   = (ce * mask).sum()
    unmasked_cnt = jnp.maximum(mask.sum(), 1.0)   # prevent division by zero
    sft_ce_loss = masked_sum / unmasked_cnt

    # Paper §3.1.6 keeps MoE load balancing active during SFT.
    if moe_layers is None:
        return sft_ce_loss
    lb_loss = compute_load_balance_loss(moe_layers)
    return sft_ce_loss + AUX_LOSS_COEFF * lb_loss


# =============================================================================
# 7. Training steps (JIT-compiled)
# =============================================================================


@nnx.jit
def pretrain_step(
    model: NemotronNanoBlock,
    optimizer: nnx.Optimizer,
    batch: jax.Array,
) -> jax.Array:
    """Single pre-training gradient update.  Returns the scalar loss."""
    moe_layers = collect_moe_layers(model)

    def _loss_fn(m):
        return pretrain_loss(m, batch, moe_layers)

    loss, grads = nnx.value_and_grad(_loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def sft_step(
    model: NemotronNanoBlock,
    optimizer: nnx.Optimizer,
    inputs: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Single SFT gradient update.  Returns total SFT loss."""
    moe_layers = collect_moe_layers(model)

    def _loss_fn(m):
        return sft_loss(m, inputs, labels, mask, moe_layers)

    loss, grads = nnx.value_and_grad(_loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    optimizer.update(model, grads)
    return loss


# =============================================================================
# 8. Checkpointing (Orbax)
# =============================================================================


def make_checkpoint_manager(ckpt_dir: str, max_to_keep: int = 1) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager that keeps the `max_to_keep` most recent checkpoints."""
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep)
    return ocp.CheckpointManager(pathlib.Path(ckpt_dir), options=options)


def save_checkpoint(manager: ocp.CheckpointManager, model: NemotronNanoBlock, step: int) -> None:
    """Serialise model parameters to disk at the given step."""
    _, state = nnx.split(model)
    manager.save(step, args=ocp.args.StandardSave(state), force=True)
    manager.wait_until_finished()
    print(f"    Checkpoint saved: step {step}")


def load_checkpoint(
    manager: ocp.CheckpointManager,
    model: NemotronNanoBlock,
    config: NemotronConfig,
) -> int:
    """Restore the latest checkpoint into `model` in-place.  Returns the step number."""
    latest = manager.latest_step()
    if latest is None:
        return 0
    abstract_model = nnx.eval_shape(lambda: NemotronNanoBlock(rngs=nnx.Rngs(0), config=config))
    _, abs_state = nnx.split(abstract_model)
    restored = manager.restore(latest, args=ocp.args.StandardRestore(abs_state))
    nnx.update(model, restored)
    print(f"    Resumed from checkpoint at step {latest}")
    return latest


def try_load_from_dir(src_dir: str, model: NemotronNanoBlock, config: NemotronConfig) -> bool:
    """Load the latest checkpoint from `src_dir` if it exists.  Returns True on success."""
    if not pathlib.Path(src_dir).exists():
        return False
    manager = make_checkpoint_manager(src_dir)
    step = load_checkpoint(manager, model, config)
    return step > 0


# =============================================================================
# 9. Evaluation helper
# =============================================================================


def evaluate_pretrain(
    model: NemotronNanoBlock,
    val_chunks: np.ndarray,
    val_steps: int,
    moe_layers: list[SparseMoE],
) -> tuple[float, float]:
    """Return (mean_loss, perplexity) over a few validation batches."""
    total_loss = 0.0
    count = 0
    for batch_np in make_batches(val_chunks, PRETRAIN_BATCH):
        if count >= val_steps:
            break
        loss = float(pretrain_loss(model, jnp.array(batch_np), moe_layers))
        total_loss += loss
        count += 1
    mean_loss = total_loss / max(count, 1)
    perplexity = math.exp(min(mean_loss, 20))   # clamp to prevent overflow
    return mean_loss, perplexity


# =============================================================================
# 10. PRE-TRAINING Phase 1 — Diverse data
# =============================================================================


def run_pretrain_phase1(model: NemotronNanoBlock, tokenizer) -> None:
    """Pre-train on a broad, diverse data mixture.

    Paper §2.3: Phase 1 uses 23.5 T tokens from 15 data categories including
    web crawl (medium → high quality), code, math, Wikipedia, academic text,
    multilingual text, and SFT-style synthetic data.

    The goal of Phase 1 is to give the model wide coverage of language,
    knowledge domains, and reasoning styles before Phase 2 sharpens it.

    Training uses the full Warmup-Stable-Decay schedule (§2.4).
    MoE load balancing is applied at every step (both bias update and aux loss).
    """
    print("\n=== Pre-Training Phase 1: Diverse Data ===")

    # ── Load training and validation data ───────────────────────────────────
    # In a real run Phase 1 data would be a weighted blend of 15 source types.
    # We approximate with two non-overlapping slices of fineweb-edu.
    train_chunks = load_pretrain_data(
        split="train", max_samples=200, seq_len=PRETRAIN_SEQ_LEN,
        tokenizer=tokenizer, skip=0,
    )
    val_chunks = load_pretrain_data(
        split="train", max_samples=50, seq_len=PRETRAIN_SEQ_LEN,
        tokenizer=tokenizer, skip=200,
    )

    # ── Optimizer: AdamW + WSD LR schedule ──────────────────────────────────
    tx = make_adamw_optimizer(
        peak_lr=PRETRAIN_PEAK_LR,
        min_lr=PRETRAIN_MIN_LR,
        warmup_steps=PRETRAIN_WARMUP_STEPS,
        stable_steps=PRETRAIN_STABLE_STEPS,
        decay_steps=PRETRAIN_DECAY_STEPS,
        weight_decay=PRETRAIN_WD,
        b1=PRETRAIN_B1,
        b2=PRETRAIN_B2,
    )
    optimizer  = nnx.Optimizer(model, tx, wrt=nnx.Param)
    moe_layers = collect_moe_layers(model)
    ckpt_mgr   = make_checkpoint_manager(PRETRAIN_CKPT_DIR)

    step = 0
    for step_i in range(PHASE1_STEPS):
        for batch_np in make_batches(train_chunks, PRETRAIN_BATCH):
            if step >= PHASE1_STEPS:
                break
            batch = jnp.array(batch_np)
            loss  = pretrain_step(model, optimizer, batch)

            # Aux-loss-free bias update runs OUTSIDE the gradient tape so the
            # bias nudges do not interfere with the gradient computation.
            update_moe_biases(moe_layers)

            step += 1

            if step % 100 == 0:
                val_loss, ppl = evaluate_pretrain(
                    model, val_chunks, PRETRAIN_VAL_STEPS, moe_layers
                )
                print(f"  Step {step:5d} | train_loss={float(loss):.4f} | "
                      f"val_loss={val_loss:.4f} | ppl={ppl:.1f}")

            if step % PRETRAIN_CKPT_EVERY == 0:
                save_checkpoint(ckpt_mgr, model, step)

    save_checkpoint(ckpt_mgr, model, step)
    print("Phase 1 complete.\n")


# =============================================================================
# 11. PRE-TRAINING Phase 2 — High-quality data
# =============================================================================


def run_pretrain_phase2(model: NemotronNanoBlock, tokenizer) -> None:
    """Continue pre-training on a high-quality subset.

    Paper §2.3: at the 94% point of training (after Phase 1), the data mixture
    shifts to emphasise high-quality sources such as Wikipedia, curated
    synthetic datasets, and premium web text.  This final 6% (1.5 T tokens)
    sharpens the model's knowledge and reduces noise from web-scale data.

    The LR schedule is shared across Phase 1 + Phase 2.  We restart the WSD
    schedule here for simplicity; in a real run the schedule would be a single
    continuous schedule spanning both phases.
    """
    print("\n=== Pre-Training Phase 2: High-Quality Data ===")

    # ── High-quality data: use a different slice / source ───────────────────
    # In the paper this would be Wikipedia + synthetic rephrases + curated text.
    train_chunks = load_pretrain_data(
        split="train", max_samples=100, seq_len=PRETRAIN_SEQ_LEN,
        tokenizer=tokenizer, skip=500,
    )
    val_chunks = load_pretrain_data(
        split="train", max_samples=30, seq_len=PRETRAIN_SEQ_LEN,
        tokenizer=tokenizer, skip=600,
    )

    # Re-use a WSD schedule tuned for the shorter Phase 2 budget.
    phase2_warmup = max(1, PHASE2_STEPS // 10)
    phase2_stable = max(1, PHASE2_STEPS // 2)
    phase2_decay  = max(1, PHASE2_STEPS - phase2_warmup - phase2_stable)

    tx = make_adamw_optimizer(
        peak_lr=PRETRAIN_PEAK_LR,
        min_lr=PRETRAIN_MIN_LR,
        warmup_steps=phase2_warmup,
        stable_steps=phase2_stable,
        decay_steps=phase2_decay,
        weight_decay=PRETRAIN_WD,
        b1=PRETRAIN_B1,
        b2=PRETRAIN_B2,
    )
    optimizer  = nnx.Optimizer(model, tx, wrt=nnx.Param)
    moe_layers = collect_moe_layers(model)
    ckpt_mgr   = make_checkpoint_manager(PRETRAIN_CKPT_DIR)

    step = 0
    for step_i in range(PHASE2_STEPS):
        for batch_np in make_batches(train_chunks, PRETRAIN_BATCH):
            if step >= PHASE2_STEPS:
                break
            batch = jnp.array(batch_np)
            loss  = pretrain_step(model, optimizer, batch)
            update_moe_biases(moe_layers)
            step += 1

            if step % 100 == 0:
                val_loss, ppl = evaluate_pretrain(
                    model, val_chunks, PRETRAIN_VAL_STEPS, moe_layers
                )
                print(f"  Step {step:4d} | train_loss={float(loss):.4f} | "
                      f"val_loss={val_loss:.4f} | ppl={ppl:.1f}")

            if step % PRETRAIN_CKPT_EVERY == 0:
                save_checkpoint(ckpt_mgr, model, step)

    save_checkpoint(ckpt_mgr, model, step)
    print("Phase 2 complete.\n")


# =============================================================================
# 12. PRE-TRAINING LC-Phase — Long-Context Extension
# =============================================================================


def run_lc_phase(model: NemotronNanoBlock, tokenizer) -> None:
    """Continuous pre-training to extend the model's context window.

    Paper §2.5: after the main pre-training run, the model is further trained
    on a mixture of:
      - Document QA at up to 512 K tokens (79% of the blend, scaled from Phase 2)
      - Synthetic retrieval-focused data at up to 256 K tokens (1%)
      - Long-context document QA (20%)

    The mix of 512 K and 4 K sequences is critical: using only very long
    sequences degraded short-context benchmarks (MMLU-Pro, Code).

    We use a constant LR of 1e-5 as described in §2.5.  The LC-Phase consumed
    121 B tokens in the paper; we run for LC_PHASE_STEPS here.
    """
    print("\n=== Pre-Training LC-Phase: Long-Context Extension ===")

    # We simulate the mixed-length nature of the LC-Phase by using a slightly
    # longer sequence length than pre-training.
    lc_chunks = load_pretrain_data(
        split="train", max_samples=100, seq_len=LC_SEQ_LEN,
        tokenizer=tokenizer, skip=700,
    )

    # Constant LR for the LC-Phase — no warmup or decay.
    tx         = make_constant_lr_optimizer(LC_PHASE_LR, PRETRAIN_WD, PRETRAIN_B1, PRETRAIN_B2)
    optimizer  = nnx.Optimizer(model, tx, wrt=nnx.Param)
    moe_layers = collect_moe_layers(model)
    ckpt_mgr   = make_checkpoint_manager(PRETRAIN_CKPT_DIR)

    step = 0
    for _ in range(LC_PHASE_STEPS):
        for batch_np in make_batches(lc_chunks, PRETRAIN_BATCH):
            if step >= LC_PHASE_STEPS:
                break
            batch = jnp.array(batch_np)
            loss  = pretrain_step(model, optimizer, batch)
            update_moe_biases(moe_layers)
            step += 1

            if step % 50 == 0:
                print(f"  LC-Phase step {step:3d} | loss={float(loss):.4f}")

    save_checkpoint(ckpt_mgr, model, step + 10_000)   # offset step for unique key
    print("LC-Phase complete.\n")


# =============================================================================
# 13. POST-TRAINING Step 1 — Supervised Fine-Tuning (SFT)
# =============================================================================


def run_sft(model: NemotronNanoBlock, tokenizer) -> None:
    """Fine-tune the base model on a supervised chat + reasoning dataset.

    Paper §3.1: SFT teaches the model reasoning, agentic tool use, instruction
    following, safety, and multilingual abilities.  All samples use the
    Nemotron chat template and only response tokens are included in the loss.

    Key SFT design choices from §3.1:
      • Loss mask: user/system tokens = 0, assistant tokens = 1.
      • Reasoning on/off control: 10% of samples have <think> stripped.
      • Reasoning budget control: 3% of samples have a truncated trace.
      • 13 000 steps at LR 5e-5 with 800-step warmup and MoE load-balance loss.
    """
    print("\n=== Post-Training Step 1: SFT ===")

    train_samples = load_sft_data("train")
    val_samples   = load_sft_data("test")

    # SFT uses a short warmup then a constant (no-decay) LR.
    tx = make_adamw_optimizer(
        peak_lr=SFT_LR,
        min_lr=SFT_MIN_LR,
        warmup_steps=SFT_WARMUP_STEPS,
        stable_steps=max(1, SFT_STEPS - SFT_WARMUP_STEPS - 1),
        decay_steps=1,   # effectively no decay after stable phase
        weight_decay=SFT_WD,
        b1=SFT_B1,
        b2=SFT_B2,
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ckpt_mgr  = make_checkpoint_manager(SFT_CKPT_DIR)
    moe_layers = collect_moe_layers(model)

    step = 0
    while step < SFT_STEPS:
        for inputs, labels, mask in make_sft_batches(
            train_samples, tokenizer, SFT_BATCH, SFT_SEQ_LEN
        ):
            if step >= SFT_STEPS:
                break
            loss = sft_step(model, optimizer, inputs, labels, mask)
            update_moe_biases(moe_layers)
            step += 1

            if step % 50 == 0:
                # Quick validation: run a few batches on the test split.
                val_loss = 0.0
                val_count = 0
                for vinputs, vlabels, vmask in make_sft_batches(
                    val_samples, tokenizer, SFT_BATCH, SFT_SEQ_LEN
                ):
                    val_loss += float(sft_loss(model, vinputs, vlabels, vmask, moe_layers))
                    val_count += 1
                    if val_count >= 10:
                        break
                val_loss /= max(val_count, 1)
                print(f"  SFT step {step:4d} | train_loss={float(loss):.4f} | "
                      f"val_loss={val_loss:.4f}")

            if step % SFT_CKPT_EVERY == 0:
                save_checkpoint(ckpt_mgr, model, step)

    save_checkpoint(ckpt_mgr, model, step)
    print("SFT complete.\n")


# =============================================================================
# 14. POST-TRAINING Step 2 — RLVR (GRPO)
# =============================================================================


def generate_completion_tokens(
    model: NemotronNanoBlock,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = RLVR_MAX_NEW_TOKENS,
    temperature: float = RLVR_TEMPERATURE,
    rng_seed: int = 0,
) -> list[int]:
    """Autoregressively sample a completion for the given prompt token IDs.

    Returns the completion token IDs only (prompt tokens excluded).
    Uses the model's KV/SSM cache for O(1) per-token cost during sampling.
    """
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]

    rng    = jax.random.PRNGKey(rng_seed)
    caches = model.init_caches(batch_size=1, max_attn_len=RLVR_MAX_CACHE_LEN)

    # Prefill: run every prompt token through the model to warm up caches.
    logits = None
    for tok in prompt_ids:
        logits, caches = model.step(jnp.array([tok]), caches)

    # Sampling: generate new tokens one at a time.
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_logits     = logits[0]   # (vocab_size,)
        rng, sample_rng = jax.random.split(rng)
        next_token      = int(jax.random.categorical(sample_rng, next_logits / temperature))
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
        logits, caches  = model.step(jnp.array([next_token]), caches)

    return generated


def compute_verifiable_reward(completion_text: str, ground_truth: str) -> float:
    """Score a completion against the verifiable ground-truth answer.

    Paper §3.2.1 uses different verifiers per environment:
      - Math    : exact answer match
      - Code    : unit test pass/fail
      - QA      : string match / LLM judge
      - etc.

    We implement a simple math verifier (exact numeric match) as a stand-in.

    Reward components:
      +0.2   for using <think>…</think> format (2 × 0.1)
      +1.0   for a correct numeric answer
    """
    reward = 0.0
    if "<think>" in completion_text:
        reward += 0.1
    if "</think>" in completion_text:
        reward += 0.1

    # Extract the last number after </think> (or anywhere in the text).
    answer = None
    if "</think>" in completion_text:
        after = completion_text.split("</think>")[-1]
        m = re.search(r"-?[\d,]+\.?\d*", after)
        if m:
            answer = m.group()
    if answer is None:
        nums = re.findall(r"-?[\d,]+\.?\d*", completion_text)
        if nums:
            answer = nums[-1]

    def to_int(s):
        try:
            return int(float(s.strip().replace(",", "")))
        except Exception:
            return None

    gt_num = to_int(ground_truth)
    pr_num = to_int(answer) if answer else None
    if gt_num is not None and pr_num is not None and gt_num == pr_num:
        reward += 1.0

    return reward


def compute_grpo_advantages(rewards: list[float]) -> np.ndarray:
    """Normalise a group of rewards to GRPO advantages.

    GRPO (Group Relative Policy Optimization, §3.2.5) avoids training a
    separate value network.  Instead it uses group-relative normalisation:
        A_j = (r_j − mean(r)) / (std(r) + ε)
    A positive advantage encourages the model to produce completion j more
    often; a negative advantage discourages it.

    When all rewards are equal (std ≈ 0), all advantages → 0 and the
    gradient signal disappears, relying only on the KL penalty.
    """
    r = np.array(rewards, dtype=np.float32)
    return (r - r.mean()) / (r.std() + 1e-6)


def curriculum_sample(
    samples: list[dict],
    pass_rates: np.ndarray,
    step: int,
    total_steps: int,
    batch_size: int,
) -> list[dict]:
    """Sample tasks according to the curriculum schedule from §3.2.2.

    The paper uses a Gaussian pass-rate target that shifts from easy (high
    pass-rate) tasks early in training toward hard (low pass-rate) tasks late
    in training.  This prevents the model from overfitting to trivial examples
    while still providing enough successes to learn from.

    Here we approximate this by drawing samples whose difficulty (1 - pass_rate)
    is close to a linearly increasing target difficulty, using softmax weights.
    """
    # Target difficulty increases linearly from 0.3 to 0.7 over training.
    target_difficulty = 0.3 + 0.4 * (step / max(total_steps, 1))
    difficulties = 1.0 - pass_rates   # convert pass-rate to difficulty

    # Gaussian weighting: prefer samples whose difficulty ≈ target.
    weights = np.exp(-0.5 * ((difficulties - target_difficulty) / 0.2) ** 2)
    weights = weights / weights.sum()

    chosen_idx = np.random.choice(len(samples), size=batch_size, replace=False, p=weights)
    return [samples[i] for i in chosen_idx]


def build_grpo_batch(
    prompt_ids_list: list[list[int]],
    completion_groups: list[list[list[int]]],
    advantage_groups: list[np.ndarray],
    seq_len: int,
    pad_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pack prompt + completions into a fixed-shape (B×G, seq_len) batch.

    Each prompt is paired with G completions (NUM_GENERATIONS).  We flatten
    the (batch, group) dimensions and produce arrays of shape (B×G, seq_len).

    Returns:
        token_ids   — (B×G, seq_len) int32   : full sequences (prompt + completion)
        masks       — (B×G, seq_len) float32 : 1.0 for completion tokens
        advantages  — (B×G,)         float32 : GRPO advantage per sequence
    """
    all_tokens:     list[np.ndarray] = []
    all_masks:      list[np.ndarray] = []
    all_advantages: list[float]      = []

    for prompt_ids, completions, advantages in zip(
        prompt_ids_list, completion_groups, advantage_groups
    ):
        n_prompt = len(prompt_ids)
        for comp_ids, adv in zip(completions, advantages):
            full = (prompt_ids + comp_ids)[:seq_len]
            mask_arr = [0.0] * min(n_prompt, seq_len)
            mask_arr += [1.0] * max(len(full) - n_prompt, 0)

            # Pad / truncate to exactly seq_len.
            pad_len = seq_len - len(full)
            full     += [pad_id] * pad_len
            mask_arr += [0.0]    * pad_len

            all_tokens.append(np.array(full[:seq_len],     dtype=np.int32))
            all_masks. append(np.array(mask_arr[:seq_len], dtype=np.float32))
            all_advantages.append(float(adv))

    return (
        jnp.array(np.stack(all_tokens)),
        jnp.array(np.stack(all_masks)),
        jnp.array(np.array(all_advantages, dtype=np.float32)),
    )


def grpo_loss(
    model: NemotronNanoBlock,
    token_ids: jax.Array,
    masks: jax.Array,
    advantages: jax.Array,
    ref_log_probs: jax.Array,
    old_log_probs: jax.Array,
    kl_coeff: float = RLVR_KL_COEFF,
    clip_eps: float = RLVR_CLIP_EPS,
) -> jax.Array:
    """GRPO objective: clipped-ratio policy gradient + unbiased KL penalty.

    For each completion token t in sequence j the loss is:
        L_j(t) = −min(r_t · A_j,  clip(r_t, 1−ε, 1+ε) · A_j)
               + β · (exp(δ_t) − δ_t − 1)    ← unbiased KL (always ≥ 0)
    where:
        r_t  = π_θ(t) / π_old(t)  — importance-sampling ratio
        δ_t  = log π_ref(t) − log π_θ(t)
        ε    = clip_eps (PPO-style trust-region clipping)

    `old_log_probs` are the per-token log-probs from the policy that
    generated the completions (computed before the gradient step, outside
    the gradient tape).  `ref_log_probs` come from the fixed SFT reference
    model used for the KL anchor.

    Reference: Shao et al. (2024), DeepSeek-AI (2025), §3.2.5 of the paper.
    """
    inputs  = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    masks_  = masks[:, 1:]          # shift to align with targets

    logits_policy = model(inputs)   # (B*G, seq-1, vocab)
    log_pi_policy = jax.nn.log_softmax(logits_policy, axis=-1)
    log_p_policy = jnp.take_along_axis(
        log_pi_policy, targets[:, :, None], axis=-1
    )[:, :, 0]

    # Importance-sampling ratio π_θ / π_old.
    # `old_log_probs` is computed outside the gradient tape so it acts as a
    # constant denominator; gradients flow only through log_p_policy.
    ratio = jnp.exp(log_p_policy - old_log_probs)     # (B*G, T)

    # Clipped surrogate policy gradient (PPO/GRPO objective).
    adv = advantages[:, None]                          # (B*G, 1) → broadcast
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_loss = -jnp.minimum(ratio * adv, clipped_ratio * adv)  # (B*G, T)

    # Unbiased KL penalty: e^δ − δ − 1  where δ = log π_ref − log π_θ
    delta   = ref_log_probs - log_p_policy
    kl_pen  = jnp.exp(delta) - delta - 1.0            # (B*G, T)

    total_loss = pg_loss + kl_coeff * kl_pen           # (B*G, T)
    masked_sum = (total_loss * masks_).sum()
    n_tokens   = jnp.maximum(masks_.sum(), 1.0)
    return masked_sum / n_tokens


@nnx.jit
def rlvr_step(
    model: NemotronNanoBlock,
    optimizer: nnx.Optimizer,
    token_ids: jax.Array,
    masks: jax.Array,
    advantages: jax.Array,
    ref_log_probs: jax.Array,
    old_log_probs: jax.Array,
) -> jax.Array:
    """Single GRPO gradient update for RLVR.  Returns the scalar loss."""
    def _loss_fn(m):
        return grpo_loss(m, token_ids, masks, advantages, ref_log_probs, old_log_probs)

    loss, grads = nnx.value_and_grad(_loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    optimizer.update(model, grads)
    return loss


def compute_ref_log_probs(ref_model: NemotronNanoBlock, token_ids: jax.Array) -> jax.Array:
    """Compute reference-policy token log-probs outside the gradient tape."""
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    logits_ref = ref_model(inputs)
    log_pi_ref = jax.nn.log_softmax(logits_ref, axis=-1)
    return jnp.take_along_axis(log_pi_ref, targets[:, :, None], axis=-1)[:, :, 0]


def snapshot_router_kernels(moe_layers: list[SparseMoE]) -> list[jax.Array]:
    """Copy router kernels so they can be restored after an optimizer step."""
    return [moe.router.kernel.get_value() for moe in moe_layers]


def restore_router_kernels(moe_layers: list[SparseMoE], kernels: list[jax.Array]) -> None:
    """Restore router kernels to enforce router freezing during RLVR."""
    for moe, kernel in zip(moe_layers, kernels):
        moe.router.kernel.set_value(kernel)


def run_rlvr(model: NemotronNanoBlock, tokenizer) -> None:
    """Multi-environment RLVR using synchronous GRPO.

    Paper §3.2: We train on all RL environments simultaneously using GRPO.
    Multi-environment training produces stable, uniform gains across benchmarks,
    whereas single-environment training often degrades other benchmarks.

    Environments (§3.2.1): competition math, competition code, QA, structured
    outputs, instruction following, long context, agentic tool use.
    We use GSM8K math as a stand-in for all environments.

    Key paper details implemented here:
      • 128 prompts × 16 generations per step  (scaled down for demo)
      • Curriculum sampling: difficulty increases from easy to hard  (§3.2.2)
      • Freeze MoE router weights  (§3.2.5)
      • Overlong filtering: discard completions exceeding max_length  (§3.2.5)
      • Aux-loss-free MoE bias update  (§2.4, kept during RL)
    """
    print("\n=== Post-Training Step 2: RLVR (GRPO) ===")

    train_samples = load_sft_data("train")
    moe_layers = collect_moe_layers(model)

    # ── Freeze MoE router weights (paper §3.2.5) ─────────────────────────────
    # Router weights are frozen during RL to stabilise training: the expert
    # routing pattern learned during pretraining / SFT is preserved, and only
    # the expert MLP weights and attention weights are updated. In this simple
    # implementation we enforce freezing by restoring router kernels after each
    # optimizer step.

    # Keep a frozen copy of the SFT model as the reference policy for KL.
    # The reference is never updated — it acts as an anchor to prevent the
    # RL policy from drifting too far from the SFT checkpoint.
    graphdef, ref_state = nnx.split(model)
    ref_model = nnx.merge(graphdef, ref_state)   # deep copy

    tx        = make_constant_lr_optimizer(RLVR_LR, RLVR_WD, RLVR_B1, RLVR_B2)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ckpt_mgr  = make_checkpoint_manager(RLVR_CKPT_DIR)

    # Estimate initial pass-rates for curriculum sampling.
    # In the paper these are computed with the SFT checkpoint; we initialise
    # to 0.5 (medium difficulty) for simplicity.
    pass_rates = np.full(len(train_samples), 0.5, dtype=np.float32)

    for step in range(RLVR_STEPS):
        # ── Curriculum sampling: prefer tasks near the current difficulty target
        batch_samples = curriculum_sample(
            train_samples, pass_rates, step, RLVR_STEPS, RLVR_NUM_PROMPTS
        )

        # ── Rollout: generate RLVR_NUM_GENERATIONS completions per prompt ────
        prompt_ids_list: list[list[int]] = []
        completion_groups: list[list[list[int]]] = []
        reward_groups: list[list[float]] = []

        for sample in batch_samples:
            prompt_text = f"User: {sample['question']}\nAssistant: "
            p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_ids_list.append(p_ids)

            completions: list[list[int]] = []
            rewards: list[float] = []
            ground_truth = sample["answer"].split("####")[-1].strip()

            for g in range(RLVR_NUM_GENERATIONS):
                comp_ids = generate_completion_tokens(
                    model, tokenizer, p_ids, rng_seed=step * 100 + g
                )

                # Overlong filtering (§3.2.5): discard completions that hit the
                # length cap — they indicate the model is "overthinking" and
                # should not receive a gradient signal.
                if len(comp_ids) >= RLVR_MAX_NEW_TOKENS:
                    comp_ids = []  # fully discarded from policy gradient (mask=0)
                    rewards.append(0.0)
                else:
                    comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                    rewards.append(compute_verifiable_reward(comp_text, ground_truth))

                completions.append(comp_ids)

            completion_groups.append(completions)
            reward_groups.append(rewards)

        # ── Compute group-relative advantages ────────────────────────────────
        advantage_groups = [compute_grpo_advantages(rg) for rg in reward_groups]

        # ── Build training batch and take a gradient step ─────────────────────
        token_ids, masks, advantages = build_grpo_batch(
            prompt_ids_list, completion_groups, advantage_groups,
            seq_len=RL_TRAIN_SEQ_LEN, pad_id=tokenizer.eos_token_id,
        )
        ref_log_probs = compute_ref_log_probs(ref_model, token_ids)
        old_log_probs = compute_ref_log_probs(model, token_ids)   # policy before update
        router_kernels_before = snapshot_router_kernels(moe_layers) if RLVR_FREEZE_ROUTER else None
        loss = rlvr_step(model, optimizer, token_ids, masks, advantages, ref_log_probs, old_log_probs)
        if router_kernels_before is not None:
            restore_router_kernels(moe_layers, router_kernels_before)

        # Update MoE expert biases (aux-loss-free, §2.4).
        update_moe_biases(moe_layers)

        # Update pass-rate estimates for curriculum sampling.
        for i, (sample, rewards) in enumerate(zip(batch_samples, reward_groups)):
            idx = train_samples.index(sample)
            pass_rates[idx] = float(np.mean([r > 0 for r in rewards]))

        if step % 10 == 0:
            mean_reward = float(np.mean([r for rg in reward_groups for r in rg]))
            print(f"  RLVR step {step:3d} | loss={float(loss):.4f} | "
                  f"mean_reward={mean_reward:.3f}")

        if step % RLVR_CKPT_EVERY == 0:
            save_checkpoint(ckpt_mgr, model, step)

    save_checkpoint(ckpt_mgr, model, RLVR_STEPS)
    print("RLVR complete.\n")


# =============================================================================
# 15. POST-TRAINING Step 3 — RLHF with GenRM
# =============================================================================


def simulated_genrm_score(response_text: str, prompt: str) -> float:
    """Simulate a GenRM (Generative Reward Model) helpfulness score in [1, 5].

    Paper §3.3.1: A full GenRM is a large language model (Qwen3-235B) trained
    with GRPO to evaluate pairwise response quality.  Given two candidate
    responses it produces:
      - Individual helpfulness scores s ∈ {1,2,3,4,5}
      - A ranking score sr ∈ {1,…,6}

    We approximate this with heuristic proxies for local testing:
      +1 point for each of: has <think>, has </think>, response is non-empty.
      Scaled to [1, 5].
    """
    score = 1.0
    if "<think>" in response_text:
        score += 1.0
    if "</think>" in response_text:
        score += 1.0
    if len(response_text.split()) > 10:
        score += 1.0
    return min(score, 5.0)


def apply_group_relative_length_control(
    base_scores: list[float],
    responses: list[str],
) -> list[float]:
    """Adjust base scores with the Group Relative Length Control mechanism.

    Paper §3.3.2: to prevent the model from growing its reasoning trace
    indefinitely during RLHF, rewards are adjusted by a length penalty that
    is *relative within the group* (not an absolute token budget).

    For each response we split the text into a <think> section and an answer
    section, compute normalised length weights within the group, and add a
    zero-mean correction term.

    Final reward (Eq. 6 in the paper):
        R_i = R_i^(base) + λ_think × w̃_i^(think) + λ_answer × w̃_i^(answer)

    The correction is zero-sum across the group: penalising long responses
    automatically rewards short responses by the same total amount.

    Additionally, a Quality-Gated Conciseness Bonus (§3.3.2) is awarded to
    the shortest high-quality response in each category (think / answer).
    """
    N = len(responses)
    if N == 0:
        return base_scores

    # Split each response into its <think> block and the answer after it.
    think_lens:  list[int] = []
    answer_lens: list[int] = []
    for r in responses:
        if "</think>" in r:
            think_part  = r.split("</think>")[0]
            answer_part = r.split("</think>")[1]
        else:
            think_part  = ""
            answer_part = r
        think_lens. append(len(think_part.split()))
        answer_lens.append(len(answer_part.split()))

    think_arr  = np.array(think_lens,  dtype=np.float32)
    answer_arr = np.array(answer_lens, dtype=np.float32)

    # ── Normalise lengths within the group (Eq. 4) ──────────────────────────
    def centered_weights(lengths: np.ndarray) -> np.ndarray:
        lo, hi = lengths.min(), lengths.max()
        if hi == lo:
            return np.zeros_like(lengths)
        # w_i = 1 − (ℓ_i − ℓ_min) / (ℓ_max − ℓ_min)
        # Shorter responses get a higher w_i (less penalised).
        w = 1.0 - (lengths - lo) / (hi - lo)
        return w - w.mean()   # zero-mean centring (Eq. 5)

    w_think  = centered_weights(think_arr)
    w_answer = centered_weights(answer_arr)

    # ── Length-adjusted rewards (Eq. 6) ─────────────────────────────────────
    scores = np.array(base_scores, dtype=np.float32)
    scores += RLHF_LAMBDA_THINK  * w_think
    scores += RLHF_LAMBDA_ANSWER * w_answer

    # ── Quality-Gated Conciseness Bonus (§3.3.2) ────────────────────────────
    # Award a bonus only to responses that are (a) the shortest and (b)
    # achieve a base score at or above the 80th-percentile threshold.
    threshold = float(np.percentile(base_scores, RLHF_PERCENTILE))

    min_think_idx  = int(np.argmin(think_arr))
    min_answer_idx = int(np.argmin(answer_arr))

    if base_scores[min_think_idx]  >= threshold:
        scores[min_think_idx]  += RLHF_BETA_THINK

    if base_scores[min_answer_idx] >= threshold:
        scores[min_answer_idx] += RLHF_BETA_ANSWER

    return scores.tolist()


def circular_pairwise_comparison(
    responses: list[str],
    prompt: str,
) -> list[float]:
    """Score all responses using circular comparison to reduce GenRM calls.

    Paper §3.3.2: naïve all-pairs comparison costs O(N²) GenRM calls.  The
    paper reduces this to O(N) by using a *circular* comparison:
        (r₁,r₂), (r₂,r₃), …, (r_{N-1},r_N), (r_N,r₁)
    Each response appears in exactly two comparisons (once as left, once as
    right), providing an unbiased score while cutting cost from 120 to 16
    comparisons for N=16 responses.

    We implement the circular scheme with a simple helpfulness scorer.
    """
    N = len(responses)
    # Each response accumulates scores from two comparison positions.
    accumulated = np.zeros(N, dtype=np.float32)

    for i in range(N):
        j = (i + 1) % N   # circular neighbour
        s_i = simulated_genrm_score(responses[i], prompt)
        s_j = simulated_genrm_score(responses[j], prompt)

        # Tiebreaker (Eq. 2-3 in the paper): when individual scores are equal,
        # the ranking score acts as a tiebreaker.
        if abs(s_i - s_j) < 0.01:
            # Use a linear interpolation of the two scores as the ranking signal.
            sr = 3.5   # neutral ranking (no clear winner)
            s_i += 3.5 - sr
            s_j += sr  - 3.5

        accumulated[i] += s_i
        accumulated[j] += s_j

    # Average over two appearances (each response was scored twice).
    return (accumulated / 2.0).tolist()


def run_rlhf(model: NemotronNanoBlock, tokenizer) -> None:
    """RLHF using a Generative Reward Model (GenRM) with Group Relative Length Control.

    Paper §3.3: after RLVR, RLHF is applied to improve the model's behaviour
    on chat-style tasks where rewards are harder to verify automatically.

    Pipeline:
      1. For each prompt, generate RLHF_NUM_RESPONSES responses.
      2. Use circular pairwise comparison to score responses with GenRM.
      3. Apply Group Relative Length Control to discourage verbosity.
      4. Compute GRPO advantages from the adjusted scores.
      5. Take a policy gradient step with a KL penalty.

    The paper observes that verbosity decreases 30% during RLHF without
    sacrificing accuracy — a direct result of the length control mechanism.
    """
    print("\n=== Post-Training Step 3: RLHF with GenRM ===")

    train_samples = load_sft_data("train")

    # Reference model (frozen SFT checkpoint) — same role as in RLVR.
    graphdef, ref_state = nnx.split(model)
    ref_model = nnx.merge(graphdef, ref_state)

    tx        = make_constant_lr_optimizer(RLHF_LR, RLHF_WD, RLHF_B1, RLHF_B2)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ckpt_mgr  = make_checkpoint_manager(RLHF_CKPT_DIR)

    for step in range(RLHF_STEPS):
        batch_samples = train_samples[
            step * RLHF_NUM_PROMPTS : (step + 1) * RLHF_NUM_PROMPTS
        ]
        if not batch_samples:
            break

        prompt_ids_list: list[list[int]] = []
        completion_groups: list[list[list[int]]] = []
        reward_groups: list[list[float]] = []

        for sample in batch_samples:
            prompt_text = f"User: {sample['question']}\nAssistant: "
            p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_ids_list.append(p_ids)

            # Generate N candidate responses for this prompt.
            responses_text: list[str] = []
            completions:    list[list[int]] = []
            for g in range(RLHF_NUM_RESPONSES):
                comp_ids  = generate_completion_tokens(
                    model, tokenizer, p_ids,
                    max_new_tokens=200, rng_seed=step * 50 + g,
                )
                comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                responses_text.append(comp_text)
                completions.append(comp_ids)

            # Score with circular comparison (O(N) GenRM calls, not O(N²)).
            base_scores = circular_pairwise_comparison(responses_text, prompt_text)

            # Apply Group Relative Length Control (§3.3.2).
            adjusted_scores = apply_group_relative_length_control(
                base_scores, responses_text
            )

            completion_groups.append(completions)
            reward_groups.append(adjusted_scores)

        # Compute advantages and run the GRPO update.
        advantage_groups = [compute_grpo_advantages(rg) for rg in reward_groups]
        token_ids, masks, advantages = build_grpo_batch(
            prompt_ids_list, completion_groups, advantage_groups,
            seq_len=RL_TRAIN_SEQ_LEN, pad_id=tokenizer.eos_token_id,
        )
        ref_log_probs = compute_ref_log_probs(ref_model, token_ids)
        loss = rlvr_step(model, optimizer, token_ids, masks, advantages, ref_log_probs)
        update_moe_biases(collect_moe_layers(model))

        if step % 10 == 0:
            mean_score = float(np.mean([s for rg in reward_groups for s in rg]))
            print(f"  RLHF step {step:3d} | loss={float(loss):.4f} | "
                  f"mean_genrm_score={mean_score:.3f}")

        if step % RLHF_CKPT_EVERY == 0:
            save_checkpoint(ckpt_mgr, model, step)

    save_checkpoint(ckpt_mgr, model, RLHF_STEPS)
    print("RLHF complete.\n")


# =============================================================================
# 16. Main — Full Training Orchestration
# =============================================================================


def main() -> None:
    """Run the complete Nemotron 3 Nano training pipeline.

    Phase order (§1 of the paper):
      Pre-Train Phase 1  →  Pre-Train Phase 2  →  LC-Phase
      →  SFT  →  RLVR  →  RLHF  →  RLVR (second pass)

    Each phase saves its checkpoint so the pipeline can be resumed after any
    stage by commenting out earlier phases in this function.
    """

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("Loading Nemotron tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ──────────────────────────────────────────────────────────────
    print("Building model …")
    model  = build_model(seed=0)
    config = model.config   # keep a reference for checkpoint restoration

    # ── Pre-Training ───────────────────────────────────────────────────────
    # Paper §2: 25 T token pretraining in two phases + long-context extension.
    run_pretrain_phase1(model, tokenizer)
    run_pretrain_phase2(model, tokenizer)
    run_lc_phase(model, tokenizer)

    # ── Post-Training ──────────────────────────────────────────────────────
    # Paper §3: SFT → RLVR → RLHF → RLVR (second pass).
    run_sft(model, tokenizer)
    run_rlvr(model, tokenizer)       # First RLVR pass (immediately after SFT)
    run_rlhf(model, tokenizer)
    run_rlvr(model, tokenizer)       # Second RLVR pass (after RLHF, §3.2)

    print("\n✓ Full training pipeline complete.")
    print("  Final checkpoint is in:", RLVR_CKPT_DIR)


if __name__ == "__main__":
    main()

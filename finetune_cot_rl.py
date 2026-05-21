"""
finetune_cot_rl.py — Chain-of-Thought RL Fine-Tuning via GRPO.

Overview
--------
This script is the *third* training phase:

    Phase 1 — pretrained.py   : Learn language from raw web text.
    Phase 2 — finetune_cot.py : Learn to reason by imitating annotated examples.
    Phase 3 — THIS FILE       : Improve reasoning by trial-and-error with rewards.

What is GRPO?
-------------
GRPO (Group Relative Policy Optimization) is the RL algorithm behind
DeepSeek-R1.  It avoids the need for a separate critic/value network (unlike
PPO) by using *group-relative advantages*:

  1. For each question, the model generates G candidate answers (a "group").
  2. A reward function scores each candidate.
  3. The scores are *normalized within the group* to get advantages:
         A_j = (r_j − mean(r)) / (std(r) + ε)
     This means: "how much better was completion j compared to the average
     of the group?"  A positive advantage → push the model to produce this
     more often.  A negative advantage → push the model away from this.
  4. A policy gradient update is applied:  loss = −A_j × log π_θ(o_j | q)
  5. A KL divergence penalty keeps the policy from drifting too far from a
     frozen reference model, preventing reward hacking.

Why RL after SFT?
-----------------
SFT (finetune_cot.py) teaches the model to imitate *existing* reasoning traces.
The model is limited by the quality and diversity of those examples.

RL with GRPO lets the model *explore* and discover reasoning strategies on its
own.  If a new approach gets a higher reward (correct answer), the model learns
to produce it — even if that approach never appeared in the training data.

GRPO Loss
---------
For each completion token t in a response o_j to question q:

    L = −A_j · log π_θ(t)          ← policy gradient
      + β · (exp(δ_t) − δ_t − 1)   ← unbiased KL penalty
                                       where δ_t = log π_ref(t) − log π_θ(t)

The KL term is always ≥ 0 (because e^x − x − 1 ≥ 0 for all x) and equals 0
when the policy matches the reference exactly.  β = KL_COEFF controls the
strength of the pull back toward the reference.

The loss is averaged over response tokens only (same mask logic as SFT).
Prompt/question tokens are excluded — we only train on what the model generates.

How to run
----------
  # Recommended: run the phases in order
  python pretrained.py      # Phase 1
  python finetune_cot.py    # Phase 2
  python finetune_cot_rl.py # Phase 3  ← this script

  If no SFT checkpoint is found, it falls back to the pretrained checkpoint.
  If no pretrained checkpoint is found either, it starts from random weights
  (useful for testing the pipeline end-to-end without a full training run).

References
----------
- DeepSeek-R1 (GRPO): DeepSeek-AI (2025)
    "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL"
    https://arxiv.org/abs/2501.12948

- GRPO algorithm: Shao et al. (2024)
    "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open LMs"
    https://arxiv.org/abs/2402.03300

- GSM8K dataset: Cobbe et al. (2021)
    "Training Verifiers to Solve Math Word Problems"
    https://arxiv.org/abs/2110.14168
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
# Hyperparameters
# =============================================================================

# ── Tokenizer / Model ─────────────────────────────────────────────────────────
VOCAB_SIZE = 131072     # nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 vocabulary
SEQ_LEN    = 256        # total token budget per sample (prompt + completion)
                        # must be divisible by CHUNK_SIZE for Mamba's SSD kernel
CHUNK_SIZE = 64         # Mamba SSD chunk size — must match NemotronConfig.mamba_chunk_size

# ── GRPO-specific ─────────────────────────────────────────────────────────────
NUM_COMPLETIONS = 4     # G — completions generated per question per episode.
                        # More completions → richer advantage signal but slower.
                        # Minimum useful value is 2 (need variance to normalize).

KL_COEFF = 0.04         # β — weight of the KL penalty in the GRPO loss.
                        # Larger → policy stays closer to the reference model.
                        # Smaller → policy can explore more freely (risk of
                        # reward hacking if set too low).

# ── Reward function ───────────────────────────────────────────────────────────
REWARD_CORRECT = 1.0    # given when the extracted numeric answer is correct.
REWARD_FORMAT  = 0.2    # given for correct use of <think>...</think> structure.
                        # Split: +0.1 for <think> present, +0.1 for </think> present.
                        # Format reward encourages the model to show its reasoning
                        # even before it learns to get answers right.

# ── RL optimiser ──────────────────────────────────────────────────────────────
# RL training uses an even smaller LR than SFT (1e-5) because:
# - RL gradients are noisier (based on sampled rewards, not ground-truth labels).
# - A large LR can cause the policy to collapse or forget language knowledge.
RL_LR         = 1e-5    # starting learning rate
RL_MIN_LR     = 1e-7    # cosine decay floor
RL_STEPS      = 200     # number of GRPO episodes (gradient updates)
RL_BATCH_SIZE = 2       # questions per episode.  Each episode produces
                        # RL_BATCH_SIZE × NUM_COMPLETIONS = 8 sequences.
WEIGHT_DECAY  = 0.1     # AdamW L2 regularisation (same as pretraining)
B1 = 0.9                # Adam first-moment coefficient
B2 = 0.95               # Adam second-moment coefficient

# ── Generation ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 200    # max completion length when sampling during training
TEMPERATURE    = 0.8    # sampling temperature; higher → more diverse completions
MAX_CACHE_LEN  = 512    # KV/SSM cache capacity; must be ≥ prompt + MAX_NEW_TOKENS

# ── Training logistics ────────────────────────────────────────────────────────
CHECKPOINT_EVERY = 50   # save every N episodes
EVAL_SAMPLES     = 30   # number of test questions for evaluation

# ── Paths ─────────────────────────────────────────────────────────────────────
# We prefer the SFT checkpoint as a starting point because it already knows
# how to produce <think> blocks.  The RL phase then refines *correctness*.
SFT_CHECKPOINT_DIR      = "./checkpoints_cot"     # Phase 2 output (preferred)
PRETRAIN_CHECKPOINT_DIR = "./checkpoints"          # Phase 1 fallback
RL_CHECKPOINT_DIR       = "./checkpoints_cot_rl"  # Phase 3 output (this script)

assert SEQ_LEN % CHUNK_SIZE == 0, "SEQ_LEN must be divisible by CHUNK_SIZE"


# =============================================================================
# 1. Dataset helpers  (same as finetune_cot.py)
# =============================================================================

def load_gsm8k(split: str = "train") -> list[dict]:
    """Load GSM8K from HuggingFace.

    Each sample contains:
      "question" — the math problem
      "answer"   — step-by-step solution ending with "#### <number>"

    We use "train" (~7 473 samples) for RL episodes and "test" (~1 319) for eval.
    """
    print(f"Loading GSM8K ({split} split) ...")
    ds = load_dataset("gsm8k", "main", split=split)
    samples = [{"question": row["question"], "answer": row["answer"]} for row in ds]
    print(f"  Loaded {len(samples)} samples.")
    return samples


# =============================================================================
# 2. Model helpers
# =============================================================================

def build_model(seed: int = 0) -> NemotronNanoBlock:
    """Build a fresh tiny Nemotron model (random weights)."""
    config = NemotronConfig.from_preset("tiny")
    config.vocab_size       = VOCAB_SIZE
    config.mamba_chunk_size = CHUNK_SIZE
    config.validate()
    return NemotronNanoBlock(rngs=nnx.Rngs(seed), config=config)


def collect_moe_layers(model: NemotronNanoBlock) -> list[SparseMoE]:
    """Return every SparseMoE sub-module from the model (one per block)."""
    return [block.moe for block in model.blocks]


# =============================================================================
# 3. Reward functions
# =============================================================================

def _normalize_number(s: str) -> int | None:
    """Try to parse a string as an integer, stripping commas and whitespace.

    Returns None if parsing fails.  We use int(float(...)) to handle both
    "42" and "42.0" formats that the model might output.
    """
    try:
        return int(float(s.strip().replace(",", "")))
    except (ValueError, TypeError, AttributeError):
        return None


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from the model's generated text.

    Strategy
    --------
    If the model used the <think>...</think> format correctly, the answer
    should appear *after* the closing </think> tag.  We look there first.

    If there is no </think> tag, we fall back to finding the last number
    anywhere in the text.

    Why "last number"?
    ------------------
    In GSM8K-style reasoning, intermediate calculations appear throughout the
    text, but the final answer is stated at the very end.  Scanning for the
    last number is a simple heuristic that captures this pattern.

    Returns the matched number as a raw string (before parsing), or None.
    """
    # Primary: check the part after </think>
    if "</think>" in text:
        after_think = text.split("</think>")[-1].strip()
        m = re.search(r"-?[\d,]+\.?\d*", after_think)
        if m:
            return m.group()

    # Fallback: find the last number anywhere in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return None


def compute_reward(completion_text: str, ground_truth: str) -> float:
    """Score one model completion against the ground-truth GSM8K answer.

    Reward components
    -----------------
    Format reward (+0.1 each):
      The model is rewarded for including <think> and </think> tags.
      This encourages the model to show its reasoning *even before* it has
      learned to get the right answer — the two skills can be learned together.

    Correctness reward (+1.0):
      The model is rewarded only if the extracted numeric answer matches the
      ground truth.  This is the primary signal that drives improvement.

    Total range: [0.0, 1.2]
      0.0  — wrong answer, no format
      0.2  — wrong answer but used <think>...</think>
      1.0  — right answer but no format
      1.2  — right answer AND used <think>...</think>  ← ideal output

    Args:
        completion_text : The decoded text generated by the model.
        ground_truth    : The numeric answer string from GSM8K (e.g. "42").

    Returns:
        Total reward as a plain Python float.
    """
    reward = 0.0

    # ── Format reward ────────────────────────────────────────────────────────
    if "<think>" in completion_text:
        reward += 0.1
    if "</think>" in completion_text:
        reward += 0.1

    # ── Correctness reward ───────────────────────────────────────────────────
    extracted = extract_answer(completion_text)
    gt_num    = _normalize_number(ground_truth)
    ext_num   = _normalize_number(extracted) if extracted is not None else None

    if gt_num is not None and ext_num is not None and gt_num == ext_num:
        reward += REWARD_CORRECT

    return reward


# =============================================================================
# 4. Generation — sample completion token IDs
# =============================================================================

def generate_completion_tokens(
    model:          NemotronNanoBlock,
    tokenizer,
    prompt_ids:     list[int],
    max_new_tokens: int   = MAX_NEW_TOKENS,
    temperature:    float = TEMPERATURE,
    rng_seed:       int   = 0,
) -> list[int]:
    """Autoregressively generate completion tokens for a pre-tokenized prompt.

    Unlike generate_with_cache() in pretrained.py and finetune_cot.py (which
    returns decoded text), this function returns the raw token IDs of the
    *completion only* (not the prompt).  We need raw IDs for two reasons:

      1. To concatenate with prompt_ids when building the GRPO training batch.
      2. To log-prob recompute the completion under policy and reference models.

    Generation uses the same two-phase approach as finetune_cot.py:
      Phase 1 — Prefill  : step() through every prompt token.
      Phase 2 — Sampling : generate one new token per step until EOS.

    Args:
        model          : NemotronNanoBlock (policy model, not reference).
        tokenizer      : HuggingFace tokenizer.
        prompt_ids     : Already-tokenized prompt as a list of int token IDs.
        max_new_tokens : Maximum number of new tokens to generate.
        temperature    : Sampling temperature. Lower → more deterministic.
        rng_seed       : Integer seed for reproducibility.

    Returns:
        List of int token IDs for the completion only (excludes prompt tokens).
        Always ends with eos_token_id (or is max_new_tokens long).
    """
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]

    rng    = jax.random.PRNGKey(rng_seed)
    caches = model.init_caches(batch_size=1, max_attn_len=MAX_CACHE_LEN)

    # ── Phase 1: Prefill ─────────────────────────────────────────────────────
    # Feed every prompt token through the model so its internal state (SSM
    # hidden state + KV cache) reflects the full prompt context.
    logits = None
    for tok in prompt_ids:
        logits, caches = model.step(jnp.array([tok]), caches)

    # ── Phase 2: Sample ──────────────────────────────────────────────────────
    # Generate new tokens one at a time.  Each call to model.step() is O(1)
    # because the caches carry all prior context.
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_logits        = logits[0]   # (vocab_size,)
        rng, sample_rng    = jax.random.split(rng)
        next_token         = int(jax.random.categorical(sample_rng, next_logits / temperature))
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        logits, caches = model.step(jnp.array([next_token]), caches)

    return generated


# =============================================================================
# 5. GRPO core
# =============================================================================

def compute_group_advantages(rewards: list[float]) -> np.ndarray:
    """Normalize rewards within a group to produce GRPO advantages.

    Formula
    -------
        A_j = (r_j − mean(r)) / (std(r) + ε)

    This makes the advantage relative to the group's average:
      A_j > 0  →  completion j was better than average   → reinforce it
      A_j < 0  →  completion j was worse than average    → discourage it
      A_j ≈ 0  →  completion j was about average         → neutral update

    When all completions receive the same reward (std ≈ 0), all advantages
    are ~0 and the policy gradient signal vanishes.  This happens when the
    model either gets every question right or every question wrong.  The KL
    term still contributes to the loss in that case, keeping the policy close
    to the reference.

    Args:
        rewards : List of G scalar reward values for one question group.

    Returns:
        NumPy array of shape (G,) with normalised advantages.
    """
    r      = np.array(rewards, dtype=np.float32)
    mean_r = r.mean()
    std_r  = r.std()
    return (r - mean_r) / (std_r + 1e-6)


def build_grpo_batch(
    prompt_ids_list:   list[list[int]],
    completion_groups: list[list[list[int]]],
    advantage_groups:  list[np.ndarray],
    seq_len:           int,
    pad_id:            int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pack completions into a fixed-shape batch for the GRPO gradient step.

    Layout
    ------
    For each (question, completion) pair we build one row:

        full_ids = [prompt_tok_0, ..., prompt_tok_{P-1},
                    comp_tok_0,  ..., comp_tok_{C-1}]

    This is the same concatenation strategy used in finetune_cot.py.
    We then pad or truncate to seq_len.

    Loss mask
    ---------
    The mask is built for *labels* = full_ids[1:] (the next-token targets).
    A label position t gets mask=1 when full_ids[t+1] is a completion token,
    i.e. when t >= P−1 (same derivation as tokenize_with_mask in finetune_cot.py).

    Fixed shape
    -----------
    This function ALWAYS produces exactly B×G rows (B = len(prompt_ids_list),
    G = NUM_COMPLETIONS).  If a prompt fills the entire seq_len window (leaving
    no room for completion tokens), the row is kept with advantage=0 and
    mask=all-zeros.  This ensures a stable shape for @nnx.jit compilation.

    Args:
        prompt_ids_list   : B lists of prompt token IDs.
        completion_groups : B groups, each containing G lists of completion IDs.
        advantage_groups  : B arrays of shape (G,), one advantage per completion.
        seq_len           : Target sequence length.
        pad_id            : Padding token ID (typically eos_token_id).

    Returns:
        all_seqs        : (B×G, seq_len)   int32   — full padded sequences.
        response_masks  : (B×G, seq_len-1) float32 — 1.0 for completion positions
                          in the label array (all_seqs[:, 1:]).
        flat_advantages : (B×G,)           float32 — one advantage per row.
    """
    all_seqs: list[list[int]]   = []
    all_masks: list[list[float]] = []
    flat_adv:  list[float]       = []

    for prompt_ids, comp_group, adv_group in zip(
        prompt_ids_list, completion_groups, advantage_groups
    ):
        P           = len(prompt_ids)
        mask_start  = max(P - 1, 0)   # position in labels where response starts

        for comp_ids, adv in zip(comp_group, adv_group):
            full_ids = prompt_ids + comp_ids   # [P + C] tokens
            C        = len(comp_ids)

            # ── Build the raw loss mask for labels ───────────────────────────
            # labels has length len(full_ids) − 1 = P + C − 1.
            # First P−1 positions are prompt targets → mask = 0.
            # Next C positions are completion targets → mask = 1.
            raw_mask = [0.0] * mask_start + [1.0] * C   # length = P + C − 1  ✓

            # ── Pad or truncate to fixed shape ───────────────────────────────
            if len(full_ids) >= seq_len:
                full_ids = full_ids[:seq_len]
            else:
                full_ids = full_ids + [pad_id] * (seq_len - len(full_ids))

            # Labels have length seq_len − 1.  Pad mask with 0.0 (no loss on
            # padding tokens beyond EOS).
            if len(raw_mask) >= seq_len - 1:
                raw_mask = raw_mask[:seq_len - 1]
            else:
                raw_mask = raw_mask + [0.0] * (seq_len - 1 - len(raw_mask))

            # If the entire window is prompt (mask_start ≥ seq_len-1), zero the
            # advantage so this row contributes nothing to the loss.
            if mask_start >= seq_len - 1:
                adv = 0.0

            all_seqs.append(full_ids)
            all_masks.append(raw_mask)
            flat_adv.append(float(adv))

    return (
        jnp.array(all_seqs,  dtype=jnp.int32),    # (B×G, seq_len)
        jnp.array(all_masks, dtype=jnp.float32),  # (B×G, seq_len-1)
        jnp.array(flat_adv,  dtype=jnp.float32),  # (B×G,)
    )


def compute_ref_log_probs(
    ref_model: NemotronNanoBlock,
    all_seqs:  jax.Array,
) -> jax.Array:
    """Compute per-token log-probabilities from the frozen reference model.

    Why we need this
    ----------------
    The KL penalty in GRPO is:  β × (exp(δ) − δ − 1)
    where δ = log π_ref(t) − log π_θ(t)  at each token t.

    We need log π_ref(t) for every token in every completion.  The reference
    model is frozen (never updated by the optimizer), so these log-probs serve
    as a stable anchor.

    This function is called OUTSIDE the JIT-compiled gradient step on purpose:
    we do NOT want gradients to flow through the reference model.  By computing
    ref_log_probs here (as a plain numpy/JAX array) and passing it as an
    argument to grpo_step(), we cleanly separate the reference from the
    differentiable policy path.

    How log-probs are computed
    --------------------------
    1. Forward pass: ref_model(all_seqs[:, :-1]) → logits (N, L-1, vocab_size)
    2. log_softmax(logits, axis=-1) → log_probs (N, L-1, vocab_size)
    3. Gather at the actual next token all_seqs[:, 1:] → token_log_probs (N, L-1)

    The gather step uses take_along_axis: for each (i, t) we select
    log_probs[i, t, all_seqs[i, t+1]], i.e., the log-prob the model assigned
    to the token that was actually generated next.

    Args:
        ref_model : Frozen NemotronNanoBlock (never passed to the optimizer).
        all_seqs  : (N, seq_len) int32 — padded full sequences.

    Returns:
        (N, seq_len-1) float32 — per-token log-probs from the reference model.
        NOT masked — the mask is applied later inside grpo_loss().
    """
    labels  = all_seqs[:, 1:]                               # (N, seq_len-1)
    logits  = ref_model(all_seqs[:, :-1])                   # (N, seq_len-1, vocab_size)

    log_probs = jax.nn.log_softmax(logits, axis=-1)         # (N, seq_len-1, vocab_size)

    # Gather the log-prob of each actual next token.
    # labels[:, :, None] expands to (N, seq_len-1, 1) for take_along_axis.
    token_log_probs = jnp.take_along_axis(
        log_probs,
        labels[:, :, None],
        axis=-1,
    )[:, :, 0]                                              # (N, seq_len-1)

    return token_log_probs


def grpo_loss(
    policy_model:   NemotronNanoBlock,
    all_seqs:       jax.Array,
    response_masks: jax.Array,
    flat_advantages: jax.Array,
    ref_log_probs:  jax.Array,
) -> jax.Array:
    """Compute the GRPO loss for one batch of completions.

    This function is called inside nnx.value_and_grad() in grpo_step().
    Gradients flow through policy_model only; ref_log_probs is treated as a
    constant (it was computed outside the gradient tape).

    The loss has two terms:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  1. Policy gradient                                                     │
    │                                                                         │
    │     pg = −A_j · log π_θ(t)                                             │
    │                                                                         │
    │  Minimising pg pushes log π_θ(t) up when A_j > 0 (good completion)    │
    │  and down when A_j < 0 (bad completion).                               │
    │                                                                         │
    │  2. KL penalty  (unbiased first-order approximation)                   │
    │                                                                         │
    │     kl = exp(log_ref − log_policy) − (log_ref − log_policy) − 1        │
    │                                                                         │
    │  This is always ≥ 0, equals 0 when policy == reference, and grows as  │
    │  the policy drifts away.  It is the regulariser that prevents the      │
    │  model from "reward hacking" (maximising reward by collapsing to a     │
    │  degenerate distribution).                                              │
    │                                                                         │
    │  Total:   loss = (pg + β × kl)  averaged over response tokens only.   │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        policy_model    : NemotronNanoBlock being trained (receives gradients).
        all_seqs        : (N, seq_len)   int32   — padded sequences.
        response_masks  : (N, seq_len-1) float32 — 1.0 for completion positions.
        flat_advantages : (N,)           float32 — group-relative advantages.
        ref_log_probs   : (N, seq_len-1) float32 — log π_ref(t), no gradient.

    Returns:
        Scalar loss.
    """
    labels  = all_seqs[:, 1:]                                # (N, seq_len-1)
    logits  = policy_model(all_seqs[:, :-1])                 # (N, seq_len-1, vocab_size)

    log_probs = jax.nn.log_softmax(logits, axis=-1)          # (N, seq_len-1, vocab_size)

    # Gather log-prob of each actual token (same as compute_ref_log_probs).
    policy_log_probs = jnp.take_along_axis(
        log_probs,
        labels[:, :, None],
        axis=-1,
    )[:, :, 0]                                               # (N, seq_len-1)

    # ── 1. Policy gradient loss ───────────────────────────────────────────────
    # Broadcast advantage from (N,) to (N, seq_len-1) so each token in a
    # completion carries the same advantage as its whole sequence.
    pg_per_token = -flat_advantages[:, None] * policy_log_probs   # (N, seq_len-1)

    # ── 2. KL divergence penalty (unbiased per-token approximation) ──────────
    # log_ratio = log π_ref(t) − log π_θ(t)
    # kl ≈ exp(log_ratio) − log_ratio − 1   (always ≥ 0)
    #
    # When policy == reference: exp(0) − 0 − 1 = 0  → no penalty
    # When policy diverges:    kl > 0               → penalty pulls it back
    log_ratio    = ref_log_probs - policy_log_probs           # (N, seq_len-1)
    kl_per_token = jnp.exp(log_ratio) - log_ratio - 1.0      # (N, seq_len-1)

    # ── Combined loss, averaged over response tokens only ────────────────────
    loss_per_token = pg_per_token + KL_COEFF * kl_per_token   # (N, seq_len-1)

    # response_masks zeroes out prompt positions (same mask logic as SFT).
    # We normalise by the count of active response tokens.
    masked_loss = (loss_per_token * response_masks).sum() / jnp.maximum(
        response_masks.sum(), 1.0
    )
    return masked_loss


@nnx.jit
def grpo_step(
    policy_model:    NemotronNanoBlock,
    optimizer:       nnx.Optimizer,
    all_seqs:        jax.Array,
    response_masks:  jax.Array,
    flat_advantages: jax.Array,
    ref_log_probs:   jax.Array,
) -> jax.Array:
    """One GRPO gradient update: forward pass → loss → backprop → weight update.

    This is structurally identical to finetune_step() in finetune_cot.py.
    The key difference is the loss function: instead of supervised cross-entropy,
    we use the GRPO objective (policy gradient + KL penalty).

    @nnx.jit compiles the function to XLA on the first call.  Because
    all_seqs always has shape (RL_BATCH_SIZE × NUM_COMPLETIONS, SEQ_LEN),
    JAX only compiles once.

    nnx.DiffState(0, nnx.Param) tells nnx.value_and_grad to differentiate
    only with respect to argument 0 (policy_model) and only its trainable
    parameters (nnx.Param).  The reference model is passed to grpo_loss as
    a precomputed array (ref_log_probs), so it is never differentiated.

    Args:
        policy_model    : NemotronNanoBlock (modified in-place by optimizer).
        optimizer       : nnx.Optimizer wrapping AdamW.
        all_seqs        : (N, seq_len)   int32.
        response_masks  : (N, seq_len-1) float32.
        flat_advantages : (N,)           float32.
        ref_log_probs   : (N, seq_len-1) float32 (no gradient flows through this).

    Returns:
        Scalar loss for logging.
    """
    loss, grads = nnx.value_and_grad(
        grpo_loss,
        argnums=nnx.DiffState(0, nnx.Param),  # differentiate only policy_model
    )(policy_model, all_seqs, response_masks, flat_advantages, ref_log_probs)

    optimizer.update(policy_model, grads)
    return loss


def update_moe_biases(moe_layers: list[SparseMoE]) -> None:
    """Nudge MoE expert biases to prevent expert collapse during RL training.

    Same mechanism as in pretrained.py and finetune_cot.py.  RL training
    can skew MoE routing even more than SFT because the loss distribution is
    narrower (only a few questions and completions per step).  Calling this
    after each grpo_step() keeps all experts active.

    Must be called OUTSIDE the JIT-compiled grpo_step(), after optimizer.update().
    """
    for moe in moe_layers:
        moe.update_expert_bias(moe.last_topk_indices.get_value())


# =============================================================================
# 6. Episode orchestration
# =============================================================================

def run_grpo_episode(
    policy_model:   NemotronNanoBlock,
    ref_model:      NemotronNanoBlock,
    optimizer:      nnx.Optimizer,
    moe_layers:     list[SparseMoE],
    questions:      list[str],
    ground_truths:  list[str],
    tokenizer,
    rng_key:        jax.Array,
) -> tuple[jax.Array, float, jax.Array]:
    """Run one full GRPO episode: generate → score → advantage → update.

    This function ties together all the GRPO components in the correct order.
    It is the heart of the RL training loop.

    Episode steps
    -------------
    1. Tokenize prompts.
    2. For each question, generate NUM_COMPLETIONS candidate responses using
       the *policy* model (not the reference).
    3. Score each response with the reward function.
    4. Normalise scores within each group → group-relative advantages.
    5. Pack everything into a fixed-shape batch.
    6. Compute reference log-probs OUTSIDE the gradient tape (no grad).
    7. Run grpo_step() to compute the GRPO loss and update policy weights.
    8. Update MoE expert biases.

    Why generate OUTSIDE the gradient tape?
    ----------------------------------------
    Generation requires Python control flow (loops, stopping at EOS) that
    cannot be differentiated through.  We treat the generated token sequences
    as fixed constants for the gradient step — we only differentiate the
    *log-probability* of those sequences under the current policy.  This is
    the "REINFORCE" trick at the core of GRPO.

    Args:
        policy_model  : NemotronNanoBlock being trained.
        ref_model     : NemotronNanoBlock, frozen (never passed to optimizer).
        optimizer     : nnx.Optimizer.
        moe_layers    : SparseMoE modules for bias updates.
        questions     : List of question strings (length = RL_BATCH_SIZE).
        ground_truths : List of ground-truth answer strings (same length).
        tokenizer     : HuggingFace tokenizer.
        rng_key       : JAX PRNGKey for reproducible sampling.

    Returns:
        loss         : Scalar GRPO loss for this episode.
        mean_reward  : Average reward across all completions (for logging).
        rng_key      : Updated PRNGKey (pass to the next episode).
    """
    B = len(questions)

    # ── Step 1: Tokenize prompts ───────────────────────────────────────────────
    prompt_ids_list = [
        tokenizer.encode(f"User: {q}\nAssistant: ", add_special_tokens=False)
        for q in questions
    ]

    # ── Step 2: Generate NUM_COMPLETIONS responses per question ───────────────
    # We split the rng_key to get a fresh seed for each (question, completion)
    # pair, ensuring diversity while keeping runs reproducible.
    completion_groups: list[list[list[int]]] = []

    for b_idx, prompt_ids in enumerate(prompt_ids_list):
        group: list[list[int]] = []
        for g in range(NUM_COMPLETIONS):
            rng_key, sub_key = jax.random.split(rng_key)
            # Convert the JAX key to a Python int for the generation seed.
            seed = int(sub_key[0])
            comp_tokens = generate_completion_tokens(
                policy_model, tokenizer, prompt_ids,
                rng_seed=seed,
            )
            group.append(comp_tokens)
        completion_groups.append(group)

    # ── Step 3: Score every completion ────────────────────────────────────────
    reward_groups: list[list[float]] = []
    all_rewards:   list[float]       = []   # flat list for mean logging

    for comp_group, gt in zip(completion_groups, ground_truths):
        group_rewards = []
        for comp_tokens in comp_group:
            text   = tokenizer.decode(comp_tokens)
            reward = compute_reward(text, gt)
            group_rewards.append(reward)
            all_rewards.append(reward)
        reward_groups.append(group_rewards)

    mean_reward = float(np.mean(all_rewards))

    # ── Step 4: Group-relative advantages ─────────────────────────────────────
    # Each group is normalised independently.  Completions for question A are
    # not compared against completions for question B.
    advantage_groups: list[np.ndarray] = [
        compute_group_advantages(rewards) for rewards in reward_groups
    ]

    # ── Step 5: Build the padded training batch ────────────────────────────────
    all_seqs, response_masks, flat_advantages = build_grpo_batch(
        prompt_ids_list  = prompt_ids_list,
        completion_groups = completion_groups,
        advantage_groups  = advantage_groups,
        seq_len           = SEQ_LEN,
        pad_id            = tokenizer.eos_token_id,
    )

    # ── Step 6: Reference log-probs (NO gradient) ─────────────────────────────
    # Computed here, outside grpo_step(), so the gradient tape in grpo_step()
    # never sees the reference model.  ref_log_probs is a plain JAX array.
    ref_log_probs = compute_ref_log_probs(ref_model, all_seqs)

    # ── Step 7: GRPO gradient update ──────────────────────────────────────────
    # grpo_step differentiates only through policy_model (nnx.DiffState).
    loss = grpo_step(
        policy_model, optimizer,
        all_seqs, response_masks, flat_advantages, ref_log_probs,
    )

    # ── Step 8: MoE bias correction ───────────────────────────────────────────
    update_moe_biases(moe_layers)

    return loss, mean_reward, rng_key


# =============================================================================
# 7. Checkpointing  (identical pattern to finetune_cot.py)
# =============================================================================

def make_checkpoint_manager(ckpt_dir: str, max_to_keep: int = 3) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager."""
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep)
    return ocp.CheckpointManager(pathlib.Path(ckpt_dir), options=options)


def save_checkpoint(
    manager: ocp.CheckpointManager,
    model:   NemotronNanoBlock,
    step:    int,
) -> None:
    """Serialise model weights to disk at the given training step."""
    _, state = nnx.split(model)
    manager.save(step, args=ocp.args.StandardSave(state))
    manager.wait_until_finished()
    print(f"  Checkpoint saved: step {step}")


def load_checkpoint(
    ckpt_dir: str,
    model:    NemotronNanoBlock,
    config:   NemotronConfig,
) -> bool:
    """Restore the most recent checkpoint from ckpt_dir into model in-place.

    How Orbax restoration works
    ---------------------------
    Orbax needs the *shape* of every array before reading from disk.  We get
    those shapes by running the model constructor in eval_shape mode — no real
    memory is allocated, only abstract shapes are recorded.  We split that
    abstract model into (graphdef, abstract_state) and hand abstract_state to
    Orbax as a template.

    Returns True if a checkpoint was found and loaded, False otherwise.
    """
    manager = make_checkpoint_manager(ckpt_dir)
    latest  = manager.latest_step()

    if latest is None:
        print(f"  No checkpoint found in '{ckpt_dir}'.")
        return False

    abstract_model = nnx.eval_shape(lambda: NemotronNanoBlock(rngs=nnx.Rngs(0), config=config))
    _, abs_state   = nnx.split(abstract_model)

    restored = manager.restore(latest, args=ocp.args.StandardRestore(abs_state))
    nnx.update(model, restored)
    manager.close()

    print(f"  Loaded checkpoint from step {latest}  ({ckpt_dir})")
    return True


# =============================================================================
# 8. Evaluation
# =============================================================================

def evaluate_rl(
    policy_model: NemotronNanoBlock,
    val_samples:  list[dict],
    tokenizer,
    n_samples:    int = EVAL_SAMPLES,
) -> dict:
    """Measure reasoning quality on a held-out set of GSM8K questions.

    For each question we generate one completion (low temperature for
    determinism) and score it with compute_reward().

    Returns a dict with three metrics:
      mean_reward  — average total reward (range 0.0 – 1.2)
      correct_rate — fraction of questions with the correct numeric answer
      format_rate  — fraction of completions that used <think>...</think>

    These metrics let you track whether RL is actually improving:
      - format_rate should be high early (the SFT phase taught this).
      - correct_rate should rise as RL discovers better reasoning strategies.
    """
    samples = val_samples[:n_samples]

    total_reward  = 0.0
    correct_count = 0
    format_count  = 0

    for sample in samples:
        question     = sample["question"]
        ground_truth = sample["answer"].split("####")[-1].strip()

        prompt_ids = tokenizer.encode(
            f"User: {question}\nAssistant: ", add_special_tokens=False
        )

        # Low temperature → more deterministic, better for evaluation
        comp_tokens = generate_completion_tokens(
            policy_model, tokenizer, prompt_ids,
            temperature=0.3, rng_seed=42,
        )
        text   = tokenizer.decode(comp_tokens)
        reward = compute_reward(text, ground_truth)

        total_reward += reward

        # Check correctness (same logic as compute_reward, but counted separately)
        extracted = extract_answer(text)
        gt_num    = _normalize_number(ground_truth)
        ext_num   = _normalize_number(extracted) if extracted is not None else None
        if gt_num is not None and ext_num is not None and gt_num == ext_num:
            correct_count += 1

        if "<think>" in text and "</think>" in text:
            format_count += 1

    n = max(len(samples), 1)
    return {
        "mean_reward":  total_reward  / n,
        "correct_rate": correct_count / n,
        "format_rate":  format_count  / n,
    }


# =============================================================================
# 9. LR schedule + optimiser
# =============================================================================

def make_rl_optimizer(
    start_lr:    float,
    end_lr:      float,
    total_steps: int,
) -> optax.GradientTransformation:
    """AdamW + cosine LR decay, same structure as finetune_cot.py.

    RL uses an even lower LR than SFT because:
    - GRPO gradients are noisy (reward signals are high-variance).
    - A large step could collapse the policy to a reward-hacking solution.
    """
    lr_schedule = optax.cosine_decay_schedule(
        init_value=start_lr,
        decay_steps=max(total_steps, 1),
        alpha=end_lr / start_lr,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, b1=B1, b2=B2),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("Loading Nemotron tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    train_samples = load_gsm8k(split="train")   # ~7 473 samples for RL episodes
    val_samples   = load_gsm8k(split="test")    # ~1 319 samples for evaluation

    # ── 3. Build model config ─────────────────────────────────────────────────
    config = NemotronConfig.from_preset("tiny")
    config.vocab_size       = VOCAB_SIZE
    config.mamba_chunk_size = CHUNK_SIZE
    config.validate()

    # ── 4. Policy model ───────────────────────────────────────────────────────
    # The policy model starts from the best available checkpoint.
    # Preferred order: SFT weights → pretrained weights → random (for testing).
    #
    # Starting from SFT weights is strongly recommended: the model already
    # knows the <think>...</think> format, so RL can focus on *correctness*
    # rather than learning the format from scratch.
    print("\nBuilding policy model ...")
    policy_model = NemotronNanoBlock(rngs=nnx.Rngs(0), config=config)
    moe_layers   = collect_moe_layers(policy_model)

    loaded = load_checkpoint(SFT_CHECKPOINT_DIR, policy_model, config)
    if not loaded:
        loaded = load_checkpoint(PRETRAIN_CHECKPOINT_DIR, policy_model, config)
    if not loaded:
        print("  Starting RL from random weights (pipeline test mode).")

    # ── 5. Reference model ────────────────────────────────────────────────────
    # The reference model is a frozen snapshot of the policy at the start of
    # RL training.  Its role is to provide the KL penalty anchor.
    #
    # IMPORTANT: ref_model is NEVER passed to the optimizer.  Its weights
    # never change.  It stays fixed throughout all RL training steps.
    # This is what keeps the model from drifting too far from its SFT baseline.
    print("\nBuilding reference model (frozen copy of policy) ...")
    ref_model = NemotronNanoBlock(rngs=nnx.Rngs(1), config=config)

    # Load the SAME checkpoint as the policy model.  Both models start
    # identical; only the policy model's weights are updated during RL.
    ref_loaded = load_checkpoint(SFT_CHECKPOINT_DIR, ref_model, config)
    if not ref_loaded:
        load_checkpoint(PRETRAIN_CHECKPOINT_DIR, ref_model, config)

    # ── 6. Optimiser (policy model only) ──────────────────────────────────────
    optimizer = nnx.Optimizer(
        policy_model,
        make_rl_optimizer(start_lr=RL_LR, end_lr=RL_MIN_LR, total_steps=RL_STEPS),
        wrt=nnx.Param,   # only trainable parameters, not MoE bias variables
    )

    # ── 7. RL training loop ───────────────────────────────────────────────────
    print(f"\nRL fine-tuning for {RL_STEPS} episodes ...")
    print(f"  {RL_BATCH_SIZE} questions × {NUM_COMPLETIONS} completions = "
          f"{RL_BATCH_SIZE * NUM_COMPLETIONS} sequences per gradient step.")
    print("(First episode is slow — JAX JIT-compiles the training function.)\n")

    ckpt_manager = make_checkpoint_manager(RL_CHECKPOINT_DIR)
    rng_key      = jax.random.PRNGKey(42)

    for step in range(1, RL_STEPS + 1):
        # Sample a random mini-batch of questions from the training set.
        batch_idx     = np.random.choice(len(train_samples), size=RL_BATCH_SIZE, replace=False)
        batch_samples = [train_samples[i] for i in batch_idx]
        questions     = [s["question"] for s in batch_samples]

        # Extract the ground-truth numeric answer (the part after "####").
        ground_truths = [s["answer"].split("####")[-1].strip() for s in batch_samples]

        # Run one full GRPO episode.
        loss, mean_reward, rng_key = run_grpo_episode(
            policy_model, ref_model, optimizer, moe_layers,
            questions, ground_truths, tokenizer, rng_key,
        )

        if step % 10 == 0:
            print(f"  step {step:4d} / {RL_STEPS}  |  "
                  f"loss {float(loss):.4f}  |  "
                  f"mean_reward {mean_reward:.3f}")

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(ckpt_manager, policy_model, step)

    # ── 8. Evaluation ─────────────────────────────────────────────────────────
    print("\nEvaluating on GSM8K test set ...")
    metrics = evaluate_rl(policy_model, val_samples, tokenizer)
    print(f"  Mean reward  : {metrics['mean_reward']:.4f}  (max 1.2)")
    print(f"  Correct rate : {metrics['correct_rate']:.2%}")
    print(f"  Format rate  : {metrics['format_rate']:.2%}")

    # ── 9. Final checkpoint ───────────────────────────────────────────────────
    save_checkpoint(ckpt_manager, policy_model, RL_STEPS)
    ckpt_manager.close()
    print(f"\nRL-tuned model saved to '{RL_CHECKPOINT_DIR}'")

    # ── 10. Reasoning test ────────────────────────────────────────────────────
    # Generate one answer to a sample question.
    # A well-trained RL model should produce a <think> block that leads to the
    # correct numeric answer — combining the format learned from SFT with the
    # correctness improved by RL.
    #
    # Expected output (after proper training):
    #   <think>
    #   5 + 3 = 8
    #   </think>
    #   8
    print("\n--- Reasoning test ---")
    test_question = "A baker made 5 pies in the morning and 3 pies in the afternoon. How many pies did the baker make in total?"
    prompt        = f"User: {test_question}\nAssistant: "
    prompt_ids    = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Question:\n  {test_question}\n")

    comp_tokens = generate_completion_tokens(
        policy_model, tokenizer, prompt_ids,
        max_new_tokens=300, temperature=0.3, rng_seed=0,
    )
    print(f"Model output:\n  {tokenizer.decode(comp_tokens)}")


if __name__ == "__main__":
    main()

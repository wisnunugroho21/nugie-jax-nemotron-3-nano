"""
finetune_cot.py — Chain-of-Thought (CoT) Supervised Fine-Tuning.

Overview
--------
This script is the *second* training phase, after pretraining.
While pretraining (pretrained.py) taught the model the basics of language from
raw web text, this script teaches the model *how to reason step-by-step* by
showing it worked-out math problems where every solution is fully annotated.

What is Chain-of-Thought (CoT)?
--------------------------------
Instead of jumping straight to an answer, a CoT model first writes down its
reasoning in a visible "thinking" block:

    User: If a train travels 60 km/h for 2 hours, how far does it go?
    Assistant: <think>
    Distance = speed × time = 60 × 2 = 120 km.
    </think>
    120

The model learns this pattern by supervised fine-tuning: we show it thousands
of question → <think>reasoning</think> → answer examples and train it to
reproduce those outputs.

What changes compared to pretrained.py?
----------------------------------------
  1. Dataset  : GSM8K (grade-school math) instead of FineWeb web text.
               Each sample is a (question, step-by-step solution) pair.

  2. Data format : Every example is wrapped in a prompt/response template.
                  Prompt  = "User: {question}\\nAssistant: "
                  Response = "<think>\\n{step_by_step}\\n</think>\\n{answer}"

  3. Loss mask : Only the response tokens contribute to the training loss.
                Prompt/question tokens are masked out (loss × 0).
                This is the key difference from plain next-token prediction:
                we want the model to learn *how to answer*, not to memorise
                the questions it is given.

  4. Lower LR  : Fine-tuning uses a 10× smaller learning rate than
                pretraining (3e-5 vs 3e-4) to avoid "catastrophic forgetting"
                — losing the general language knowledge acquired earlier.

  5. Fewer steps : GSM8K is much smaller than the pretraining corpus, so
                  fewer gradient updates are needed.

How to run
----------
  # Step 1: pretrain the model
  python pretrained.py

  # Step 2: fine-tune with CoT
  python finetune_cot.py

  The script will:
    1. Load the pretrained checkpoint from PRETRAIN_CHECKPOINT_DIR.
    2. Fine-tune on GSM8K for FINETUNE_STEPS steps using masked CoT loss.
    3. Save the fine-tuned model to FINETUNE_CHECKPOINT_DIR.
    4. Run a quick reasoning test and print the <think> trace.

References
----------
- GSM8K dataset : Cobbe et al. (2021)
    "Training Verifiers to Solve Math Word Problems"
    https://arxiv.org/abs/2110.14168

- Chain-of-Thought prompting : Wei et al. (2022)
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    https://arxiv.org/abs/2201.11903
"""

import math
import pathlib

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
VOCAB_SIZE = 131072     # nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 tokenizer vocab
SEQ_LEN    = 256        # max tokens per sample (prompt + response combined)
                        # must be divisible by CHUNK_SIZE for Mamba's SSD kernel
CHUNK_SIZE = 64         # Mamba SSD chunk size — must match NemotronConfig.mamba_chunk_size
BATCH_SIZE = 2          # samples per gradient update

# ── Learning rate ─────────────────────────────────────────────────────────────
# Fine-tuning uses a much smaller learning rate than pretraining (3e-4).
# Too large an LR here would overwrite the general knowledge already learned.
# "Cosine decay" means the LR starts at FINETUNE_LR and smoothly decreases
# to MIN_LR over all FINETUNE_STEPS — no warmup needed because the model
# weights are already stable from pretraining.
FINETUNE_LR    = 3e-5   # starting learning rate (10× lower than pretraining)
MIN_LR         = 1e-6   # final learning rate at the end of fine-tuning
FINETUNE_STEPS = 300    # total gradient updates; GSM8K has ~7 k samples,
                        # so this is roughly one partial pass over the data.
                        # Increase (e.g. 2000) for better reasoning quality.
CHECKPOINT_EVERY = 100  # save a checkpoint every N steps

# ── Optimiser ─────────────────────────────────────────────────────────────────
WEIGHT_DECAY = 0.1      # AdamW L2 penalty on weights (same as pretraining)
B1 = 0.9                # Adam first-moment (momentum) coefficient
B2 = 0.95               # Adam second-moment (variance) coefficient

# ── Paths ─────────────────────────────────────────────────────────────────────
PRETRAIN_CHECKPOINT_DIR = "./checkpoints"       # load weights from here
FINETUNE_CHECKPOINT_DIR = "./checkpoints_cot"   # save fine-tuned weights here

# ── Evaluation ────────────────────────────────────────────────────────────────
VAL_STEPS = 30          # number of validation batches to average for eval

assert SEQ_LEN % CHUNK_SIZE == 0, "SEQ_LEN must be divisible by CHUNK_SIZE"


# =============================================================================
# 1. Dataset helpers
# =============================================================================

def load_gsm8k(split: str = "train") -> list[dict]:
    """Load the GSM8K dataset from HuggingFace.

    GSM8K (Grade School Math 8K) contains ~8 500 grade-school math problems
    each paired with a step-by-step natural language solution.

    Each row in the dataset has exactly two fields:
      "question" — the math problem as plain text
      "answer"   — the worked solution, always ending with "#### <number>"
                   Example: "Janet has 3 apples...so the answer is 6.\\n#### 6"

    We use:
      split="train"  (~7 473 samples) for fine-tuning
      split="test"   (~1 319 samples) for validation
    """
    print(f"Loading GSM8K ({split} split) ...")
    ds = load_dataset("gsm8k", "main", split=split)
    samples = [{"question": row["question"], "answer": row["answer"]} for row in ds]
    print(f"  Loaded {len(samples)} samples.")
    return samples


def format_cot_example(question: str, answer: str) -> tuple[str, str]:
    """Format a GSM8K sample into a (prompt, response) pair for CoT training.

    GSM8K answer format
    -------------------
    The raw "answer" field in GSM8K looks like:
        "Janet makes 16 cups per day...so the answer is 6.\\n#### 6"
    It has two parts separated by "####":
        reasoning  = everything before "####"  (the step-by-step work)
        final_ans  = the number after "####"

    Output format
    -------------
    We produce two strings:

        prompt   = "User: {question}\\nAssistant: "
                    ↑ This is what the model receives as INPUT.
                    The loss mask will be 0 for all these tokens —
                    the model is NOT penalised for its predictions here.

        response = "<think>\\n{reasoning}\\n</think>\\n{final_ans}"
                    ↑ This is what the model must GENERATE.
                    The loss mask will be 1 for all these tokens —
                    every response token contributes to the training loss.

    Why return prompt and response separately?
    ------------------------------------------
    We need to tokenize them independently to measure the exact number of
    prompt tokens.  That boundary is needed in tokenize_with_mask() to place
    the 0 / 1 loss mask at exactly the right position.

    Args:
        question : Raw question string from GSM8K.
        answer   : Raw answer string (includes "#### N").

    Returns:
        (prompt, response) — two plain strings, not yet tokenized.
    """
    parts    = answer.split("####")
    reasoning = parts[0].strip()   # step-by-step solution text
    final_ans = parts[-1].strip()  # the numeric answer after "####"

    prompt   = f"User: {question}\nAssistant: "
    response = f"<think>\n{reasoning}\n</think>\n{final_ans}"
    return prompt, response


def tokenize_with_mask(
    tokenizer,
    prompt: str,
    response: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokenize a (prompt, response) pair and build a per-token loss mask.

    This is the core data-preparation step that makes CoT SFT different from
    plain next-token pretraining.

    Why do we need a loss mask?
    ---------------------------
    In pretraining we optimise every token equally: the model must learn to
    predict the next token at every position.  In SFT we only want the model
    to learn to generate the *response* (the reasoning trace + answer).  The
    question/prompt is just input context; we do not want the gradient to push
    the model towards memorising questions.

    We represent this as a binary float mask:
        mask[i] = 1.0  →  position i is a response token → include in loss
        mask[i] = 0.0  →  position i is a prompt token  → exclude from loss

    Sequence layout (before truncation / padding, length = N + M - 1)
    ------------------------------------------------------------------
    Let N = len(prompt_ids),  M = len(response_ids)  (includes EOS).

        full_ids   : [p0, p1, ..., p_{N-1},  r0,  r1, ..., r_{M-1}]
        input_ids  : [p0, p1, ..., p_{N-1},  r0,  r1, ..., r_{M-2}]   (drop last)
        labels     : [p1, p2, ..., p_{N-1},  r0,  r1, ..., r_{M-1}]   (drop first)
        mask       : [0,  0,  ..., 0,         1,   1,  ..., 1       ]
                      ╰── N-1 zeros ──╯         ╰──── M ones ────╯

    At position i the model predicts labels[i] from input_ids[i].
    The first response token appears in labels at index N-1, so
    mask_start = N-1 = len(prompt_ids) - 1.

    Args:
        tokenizer : HuggingFace tokenizer.
        prompt    : The user question string.
        response  : The <think>...answer string.
        seq_len   : Length of the returned arrays (truncate or pad to this).

    Returns:
        input_ids : (seq_len,) int32    — token ids fed into the model.
        labels    : (seq_len,) int32    — next-token targets.
        mask      : (seq_len,) float32  — 1.0 where loss is active.
    """
    # Tokenize prompt and response SEPARATELY so we know the exact boundary.
    # add_special_tokens=False prevents BOS/EOS insertion in the middle of a
    # concatenated sequence.
    prompt_ids   = tokenizer.encode(prompt,   add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    # Append EOS at the very end so the model learns to stop generating.
    response_ids.append(tokenizer.eos_token_id)

    # Concatenate into one flat token stream: [prompt tokens][response tokens]
    full_ids = prompt_ids + response_ids

    # Shift by one to create the (input, target) pair for next-token prediction.
    input_ids = full_ids[:-1]   # feed all tokens except the last
    labels    = full_ids[1:]    # predict all tokens except the first

    # ── Build the loss mask ──────────────────────────────────────────────────
    # labels[i] is the first response token when i == len(prompt_ids) - 1
    # (because the shift consumed one prompt token from the front).
    mask_start = max(len(prompt_ids) - 1, 0)
    mask = [0.0] * mask_start + [1.0] * (len(input_ids) - mask_start)

    # ── Truncate or pad all three arrays to exactly seq_len ─────────────────
    pad_id = tokenizer.eos_token_id  # EOS doubles as the padding token

    def pad_or_trunc(seq: list, fill) -> list:
        if len(seq) >= seq_len:
            return seq[:seq_len]
        return seq + [fill] * (seq_len - len(seq))

    input_ids = pad_or_trunc(input_ids, pad_id)
    labels    = pad_or_trunc(labels,    pad_id)
    mask      = pad_or_trunc(mask,      0.0)   # padding positions → no loss

    return (
        np.array(input_ids, dtype=np.int32),
        np.array(labels,    dtype=np.int32),
        np.array(mask,      dtype=np.float32),
    )


def make_cot_batches(
    samples:    list[dict],
    tokenizer,
    batch_size: int,
    seq_len:    int,
):
    """Tokenize all CoT samples, shuffle, and yield (inputs, labels, mask) batches.

    Yields
    ------
    inputs  : jax.Array  (batch_size, seq_len)  int32    — token ids
    labels  : jax.Array  (batch_size, seq_len)  int32    — next-token targets
    masks   : jax.Array  (batch_size, seq_len)  float32  — 1.0 for response tokens

    Processing steps
    ----------------
    1. Tokenize every sample with tokenize_with_mask().
    2. Discard any sample whose mask is all zeros (the response was entirely
       truncated by seq_len — nothing to learn from that sample).
    3. Shuffle remaining samples randomly (different order every call).
    4. Yield non-overlapping batches of size batch_size.

    Note: all tokenization happens upfront in Python, then batches are served
    from the pre-built NumPy arrays.  For ~7 k GSM8K samples this is fast
    (a few seconds) and keeps the training loop simple.
    """
    all_inputs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_masks:  list[np.ndarray] = []

    for sample in samples:
        prompt, response = format_cot_example(sample["question"], sample["answer"])
        inp, lab, msk = tokenize_with_mask(tokenizer, prompt, response, seq_len)

        # Skip samples where no response token survived truncation.
        # Training on such samples would compute 0 / 0 in the masked loss.
        if msk.sum() == 0:
            continue

        all_inputs.append(inp)
        all_labels.append(lab)
        all_masks.append(msk)

    if not all_inputs:
        raise RuntimeError(
            "No valid CoT samples found after tokenization. "
            "Try increasing SEQ_LEN or checking the dataset."
        )

    # Shuffle all samples so every epoch sees a different ordering.
    idx = np.random.permutation(len(all_inputs))
    stacked_inputs = np.stack(all_inputs)[idx]   # (N, seq_len)
    stacked_labels = np.stack(all_labels)[idx]   # (N, seq_len)
    stacked_masks  = np.stack(all_masks) [idx]   # (N, seq_len)

    # Yield non-overlapping slices of size batch_size.
    for start in range(0, len(stacked_inputs) - batch_size + 1, batch_size):
        end = start + batch_size
        yield (
            jnp.array(stacked_inputs[start:end]),
            jnp.array(stacked_labels[start:end]),
            jnp.array(stacked_masks [start:end]),
        )


# =============================================================================
# 2. Model helpers
# =============================================================================

def build_model(seed: int = 0) -> NemotronNanoBlock:
    """Build a fresh tiny Nemotron model (random weights, no checkpoint loaded)."""
    config = NemotronConfig.from_preset("tiny")
    config.vocab_size      = VOCAB_SIZE
    config.mamba_chunk_size = CHUNK_SIZE
    config.validate()
    return NemotronNanoBlock(rngs=nnx.Rngs(seed), config=config)


def collect_moe_layers(model: NemotronNanoBlock) -> list[SparseMoE]:
    """Return every SparseMoE sub-module from the model (one per block)."""
    return [block.moe for block in model.blocks]


# =============================================================================
# 3. Masked loss
# =============================================================================

def masked_cross_entropy_loss(
    model:  NemotronNanoBlock,
    inputs: jax.Array,
    labels: jax.Array,
    mask:   jax.Array,
) -> jax.Array:
    """Next-token cross-entropy loss, restricted to response tokens via mask.

    Why mask the loss?
    ------------------
    In pretraining every token is a target — the model must predict the next
    word at every position.  In CoT SFT we *only* want the model to learn how
    to produce the reasoning trace and the final answer.  The user question is
    just input context; penalising the model for its predictions there would
    send contradictory gradient signals (the question has no "correct" next
    word from the model's perspective).

    Masked loss formula
    -------------------
        logits  = model(inputs)                   # (B, L, vocab_size)
        per_tok = cross_entropy(logits, labels)   # (B, L)

        masked_loss = (per_tok * mask).sum()
                      ─────────────────────
                         mask.sum()

    Dividing by mask.sum() (number of response tokens) normalises the loss so
    it doesn't depend on how many response tokens happened to fit in the batch.
    We clip the denominator to ≥ 1 to guard against division by zero in case
    all tokens were masked (make_cot_batches already filters such samples, but
    it is good practice to be safe inside JIT-compiled code).

    Args:
        model  : NemotronNanoBlock.
        inputs : (B, L) int32   — token ids fed into the model.
        labels : (B, L) int32   — next-token targets.
        mask   : (B, L) float32 — 1.0 for response positions, 0.0 for prompt.

    Returns:
        Scalar loss averaged over response tokens.
    """
    logits = model(inputs)   # (B, L, vocab_size)

    # Per-token cross-entropy, shape (B, L).
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    # Apply mask: zero out prompt positions, then normalise by response token count.
    masked_loss = (per_token_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return masked_loss


# =============================================================================
# 4. Learning rate schedule
# =============================================================================

def create_cosine_lr_schedule(
    start_lr:    float,
    end_lr:      float,
    total_steps: int,
) -> optax.Schedule:
    """Single-phase cosine decay from start_lr down to end_lr.

    This is simpler than the three-phase schedule in pretrained.py because:
      - No warmup needed: the pretrained model already has stable gradients.
      - No stable plateau: every fine-tuning step should be gently reducing
        the LR so the model converges without overshooting.

    Visual shape:
        LR
        │▀▔╲
        │   ╲
        │    ╲___
        └─────────── step
               ↑ end_lr

    alpha = end_lr / start_lr makes cosine_decay_schedule land exactly at
    end_lr:  final_lr = start_lr × alpha = start_lr × (end_lr/start_lr) = end_lr.
    """
    return optax.cosine_decay_schedule(
        init_value=start_lr,
        decay_steps=max(total_steps, 1),
        alpha=end_lr / start_lr,
    )


def make_finetune_optimizer(
    start_lr:    float,
    end_lr:      float,
    total_steps: int,
) -> optax.GradientTransformation:
    """AdamW optimiser with cosine LR decay and gradient norm clipping.

    Identical structure to make_gradient_transform_optimizer() in pretrained.py
    but uses the simpler single-phase cosine schedule.

    optax.chain applies transformations left-to-right:
      1. clip_by_global_norm(1.0) — rescale gradients if their L2 norm > 1
         This prevents exploding gradients from destabilising training.
      2. adamw(...)               — adaptive learning rates + weight decay
    """
    lr_schedule = create_cosine_lr_schedule(start_lr, end_lr, total_steps)
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=WEIGHT_DECAY,
            b1=B1,
            b2=B2,
        ),
    )


# =============================================================================
# 5. Fine-tuning step
# =============================================================================

@nnx.jit
def finetune_step(
    model:     NemotronNanoBlock,
    optimizer: nnx.Optimizer,
    inputs:    jax.Array,
    labels:    jax.Array,
    mask:      jax.Array,
) -> jax.Array:
    """One fine-tuning step: forward pass → masked loss → backprop → weight update.

    Structurally identical to train_step() in pretrained.py, but passes the
    loss mask through to masked_cross_entropy_loss().

    @nnx.jit compiles this function to XLA once on the first call and caches
    the binary for all subsequent calls — so the first step is slow (~30 s)
    and every following step is fast.

    nnx.DiffState(0, nnx.Param) tells nnx.value_and_grad to differentiate
    only with respect to argument 0 (the model) and only the nnx.Param leaves
    (trainable weights), ignoring non-differentiable state like MoE bias
    variables.

    Args:
        model     : NemotronNanoBlock (modified in-place via optimizer).
        optimizer : nnx.Optimizer wrapping the AdamW gradient transform.
        inputs    : (B, L) int32   — token ids.
        labels    : (B, L) int32   — next-token targets.
        mask      : (B, L) float32 — 1.0 at response positions.

    Returns:
        Scalar loss for logging.
    """
    loss, grads = nnx.value_and_grad(
        masked_cross_entropy_loss,
        argnums=nnx.DiffState(0, nnx.Param),  # differentiate only model params
    )(model, inputs, labels, mask)
    optimizer.update(model, grads)
    return loss


def update_moe_biases(moe_layers: list[SparseMoE]) -> None:
    """Nudge expert load-balancing biases to prevent expert collapse.

    This is the same mechanism as in pretrained.py.  During fine-tuning the
    MoE routing can drift so that only a few experts get selected on the narrow
    CoT distribution.  The bias correction re-activates underused experts so
    they don't degrade from disuse.

    Must be called AFTER optimizer.update() and outside the JIT-compiled step,
    because it directly writes to mutable bias variables.
    """
    for moe in moe_layers:
        moe.update_expert_bias(moe.last_topk_indices.get_value())


# =============================================================================
# 6. Checkpointing  (Orbax — same pattern as pretrained.py)
# =============================================================================

def make_checkpoint_manager(ckpt_dir: str, max_to_keep: int = 1) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager that retains the last max_to_keep steps."""
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, overwrite=True)
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


def load_pretrained_checkpoint(
    pretrain_ckpt_dir: str,
    model:             NemotronNanoBlock,
    config:            NemotronConfig,
) -> bool:
    """Restore the most recent pretrained checkpoint into model in-place.

    How Orbax restoration works
    ---------------------------
    Orbax stores JAX arrays as raw binary blobs on disk.  Before loading,
    it needs to know the *shape* and *dtype* of every array so it can
    allocate the right buffers.  We provide those shapes by calling
    nnx.eval_shape() — this runs the model constructor but skips all actual
    array allocations, returning a "shape-only" model with abstract arrays.

    Steps:
      1. nnx.eval_shape(...)    → abstract model (shapes only, no real memory)
      2. nnx.split(abstract)    → (graphdef, abstract_state)
      3. manager.restore(...)   → restored_state (real arrays from disk)
      4. nnx.update(model, ...) → copy weights into the live model in-place

    Returns True if a checkpoint was found and loaded, False otherwise.
    """
    manager = make_checkpoint_manager(pretrain_ckpt_dir)
    latest  = manager.latest_step()

    if latest is None:
        print(f"  No pretrained checkpoint found in '{pretrain_ckpt_dir}'.")
        print("  Fine-tuning will start from random weights (still valid for testing).")
        return False

    # Build a shape-only model so Orbax knows the expected array shapes.
    abstract_model  = nnx.eval_shape(lambda: NemotronNanoBlock(rngs=nnx.Rngs(0), config=config))
    _, abs_state    = nnx.split(abstract_model)

    # Restore weights from disk and copy them into the live model.
    restored = manager.restore(latest, args=ocp.args.StandardRestore(abs_state))
    nnx.update(model, restored)
    manager.close()

    print(f"  Loaded pretrained checkpoint from step {latest}  ({pretrain_ckpt_dir})")
    return True


# =============================================================================
# 7. Evaluation on CoT data
# =============================================================================

def evaluate_cot(
    model:       NemotronNanoBlock,
    val_samples: list[dict],
    tokenizer,
    val_steps:   int,
) -> tuple[float, float]:
    """Compute masked validation loss and perplexity on the GSM8K test split.

    Only the response tokens (chain-of-thought + final answer) contribute to
    the reported numbers, which directly reflects how well the model has
    learned to produce correct reasoning — not how well it copies questions.

    Perplexity is defined as e^loss.  A perplexity of 1 means the model
    predicts every response token with 100 % certainty; higher values mean
    the model is more uncertain.  We clamp the exponent at 20 to avoid
    numerical overflow on an untrained model.

    Returns:
        (mean_loss, perplexity) averaged over up to val_steps batches.
    """
    total_loss = 0.0
    count      = 0

    for inputs, labels, mask in make_cot_batches(val_samples, tokenizer, BATCH_SIZE, SEQ_LEN):
        if count >= val_steps:
            break
        loss = masked_cross_entropy_loss(model, inputs, labels, mask)
        total_loss += float(loss)
        count      += 1

    mean_loss  = total_loss / max(count, 1)
    perplexity = math.exp(min(mean_loss, 20))   # clamp to avoid overflow
    return mean_loss, perplexity


# =============================================================================
# 8. Generation  (reused pattern from pretrained.py)
# =============================================================================

def generate_with_cache(
    model:          NemotronNanoBlock,
    tokenizer,
    prompt:         str,
    max_new_tokens: int   = 300,
    temperature:    float = 0.7,
    rng_seed:       int   = 0,
    max_cache_len:  int   = 512,
) -> str:
    """Efficient cache-based autoregressive generation.

    Identical in structure to generate_with_cache() in pretrained.py.
    We use the model's step() API (one token per call) rather than the bulk
    forward pass, so no CHUNK_SIZE padding is required.

    Two phases:
      1. Prefill  — step() through every prompt token to warm up the SSM
                    states and KV caches.
      2. Sampling — generate new tokens one at a time, sampling from the
                    scaled logit distribution, until EOS or max_new_tokens.

    For CoT inference we call this with a prompt like:
        "User: What is 5 + 7?\\nAssistant: "
    And expect the fine-tuned model to produce:
        "<think>\\n5 + 7 = 12\\n</think>\\n12"

    Args:
        model          : Fine-tuned NemotronNanoBlock.
        tokenizer      : HuggingFace tokenizer.
        prompt         : Input string (question + role prefix).
        max_new_tokens : Maximum tokens to generate.
        temperature    : Sampling temperature. Lower → more deterministic.
        rng_seed       : PRNG seed for reproducible outputs.
        max_cache_len  : KV cache capacity. Must be ≥ len(prompt) + max_new_tokens.

    Returns:
        Generated text (excluding the prompt itself).
    """
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_tokens:
        prompt_tokens = [tokenizer.eos_token_id]

    rng    = jax.random.PRNGKey(rng_seed)
    caches = model.init_caches(batch_size=1, max_attn_len=max_cache_len)

    # ── Phase 1 : Prefill ─────────────────────────────────────────────────────
    # Run every prompt token through step() so the SSM states and KV caches
    # are "warm" — they encode all the context the model needs to answer.
    logits = None
    for tok in prompt_tokens:
        logits, caches = model.step(jnp.array([tok]), caches)

    # ── Phase 2 : Sampling ────────────────────────────────────────────────────
    # `logits` now holds the model's prediction for the first new token.
    # We sample from it, then feed the chosen token back in, repeating until
    # EOS or until we hit max_new_tokens.
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_logits               = logits[0]   # (vocab_size,)
        rng, sample_rng           = jax.random.split(rng)
        next_token                = int(jax.random.categorical(sample_rng, next_logits / temperature))
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        logits, caches = model.step(jnp.array([next_token]), caches)

    return tokenizer.decode(generated)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("Loading Nemotron tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Note on <think> / </think> tokens
    # ----------------------------------
    # nemotron.py recommends adding <think> and </think> as *special tokens*
    # so each tag maps to a single dedicated token ID.  To do that in production:
    #
    #     tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    #     # then resize model.embedding and model.lm_head to match the new vocab size
    #
    # For this educational script we intentionally skip this step so the code
    # stays simple and the pretrained checkpoint loads without shape mismatches.
    # The BPE tokenizer will split <think> into a few sub-word tokens, which
    # still works — the model learns the pattern from the supervised examples.

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    train_samples = load_gsm8k(split="train")   # ~7 473 question-solution pairs
    val_samples   = load_gsm8k(split="test")    # ~1 319 question-solution pairs

    # ── 3. Model + pretrained checkpoint ─────────────────────────────────────
    print("\nBuilding model ...")
    config = NemotronConfig.from_preset("tiny")
    config.vocab_size       = VOCAB_SIZE
    config.mamba_chunk_size = CHUNK_SIZE
    config.validate()

    model      = NemotronNanoBlock(rngs=nnx.Rngs(0), config=config)
    moe_layers = collect_moe_layers(model)

    # Load pretrained weights so fine-tuning builds on top of learned language
    # knowledge rather than starting from scratch.
    # If no checkpoint is found the script continues with random weights
    # (useful for verifying the pipeline without a pretrained model).
    load_pretrained_checkpoint(PRETRAIN_CHECKPOINT_DIR, model, config)

    # ── 4. Optimiser ──────────────────────────────────────────────────────────
    # Lower LR (FINETUNE_LR) than pretraining (PEAK_LR in pretrained.py) to
    # preserve the base model's general language knowledge.
    optimizer = nnx.Optimizer(
        model,
        make_finetune_optimizer(
            start_lr=FINETUNE_LR,
            end_lr=MIN_LR,
            total_steps=FINETUNE_STEPS,
        ),
        wrt=nnx.Param,  # only optimise trainable parameters (not MoE bias vars)
    )

    # ── 5. Fine-tuning loop ───────────────────────────────────────────────────
    print(f"\nFine-tuning for {FINETUNE_STEPS} steps on GSM8K ...")
    print("(First step is slow — JAX JIT-compiles the training function.)\n")

    ckpt_manager = make_checkpoint_manager(FINETUNE_CHECKPOINT_DIR)

    step       = 0
    batch_iter = iter(make_cot_batches(train_samples, tokenizer, BATCH_SIZE, SEQ_LEN))

    while step < FINETUNE_STEPS:
        # When one pass over the data is exhausted, reshuffle and restart.
        try:
            inputs, labels, mask = next(batch_iter)
        except StopIteration:
            batch_iter           = iter(make_cot_batches(train_samples, tokenizer, BATCH_SIZE, SEQ_LEN))
            inputs, labels, mask = next(batch_iter)

        loss = finetune_step(model, optimizer, inputs, labels, mask)

        # MoE bias correction must happen OUTSIDE the JIT-compiled step because
        # it writes directly to mutable variables (not tracked by the gradient tape).
        update_moe_biases(moe_layers)

        step += 1

        if step % 10 == 0:
            print(f"  step {step:4d} / {FINETUNE_STEPS}  |  loss {float(loss):.4f}")

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(ckpt_manager, model, step)

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    print("\nEvaluating on GSM8K test set ...")
    val_loss, val_ppl = evaluate_cot(model, val_samples, tokenizer, VAL_STEPS)
    print(f"  Validation loss : {val_loss:.4f}")
    print(f"  Perplexity      : {val_ppl:.2f}")
    # A lower perplexity means the model assigns higher probability to the
    # correct reasoning tokens — it has learned to reason more confidently.

    # ── 7. Final checkpoint ───────────────────────────────────────────────────
    save_checkpoint(ckpt_manager, model, step)
    ckpt_manager.close()
    print(f"\nFine-tuned model saved to '{FINETUNE_CHECKPOINT_DIR}'")

    # ── 8. Reasoning test ─────────────────────────────────────────────────────
    # Feed a sample question and print the full generated response.
    # After sufficient training you should see a <think> block appear in the
    # output, showing the model's step-by-step reasoning before the answer.
    #
    # Example expected output (after proper training):
    #   <think>
    #   Janet has 3 sisters. Each owns 2 dolls → 3 × 2 = 6.
    #   Janet herself owns 5. Total = 6 + 5 = 11.
    #   </think>
    #   11
    print("\n--- Reasoning test ---")
    test_question = (
        "Janet has 3 sisters. Each sister owns 2 dolls. "
        "Janet herself owns 5 dolls. How many dolls are there in total?"
    )
    prompt   = f"User: {test_question}\nAssistant: "
    print(f"Question:\n  {test_question}\n")
    response = generate_with_cache(model, tokenizer, prompt, max_new_tokens=300)
    print(f"Model output:\n  {response}")


if __name__ == "__main__":
    main()

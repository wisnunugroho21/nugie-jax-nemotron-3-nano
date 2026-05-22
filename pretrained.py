"""
pretrained.py — Tiny Nemotron pretraining workflow.

Steps:
  1. Load HuggingFaceFW/fineweb-edu from Hugging Face (streaming)
  2. Train a tiny Nemotron model, saving checkpoints every N steps
  3. Evaluate with validation loss + perplexity
  4. Chat with the model in the terminal

Design goals:
  - Keep code easy to read and modify.
  - Keep control flow explicit.
  - Prefer clarity over speed/optimization.
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
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from moe import SparseMoE
from nemotron import NemotronConfig, NemotronNanoBlock

# =============================================================================
# Hyperparameters
# =============================================================================

VOCAB_SIZE = 131072  # nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 tokenizer vocabulary
SEQ_LEN = 256  # input tokens per sample — must be divisible by CHUNK_SIZE
CHUNK_SIZE = 64  # Mamba SSD chunk size — must match NemotronConfig.mamba_chunk_size
BATCH_SIZE = 2
CHECKPOINT_EVERY = 200  # save a checkpoint every N training steps
MAX_TRAIN_STEPS = 10000
WARMUP_STEPS = int(MAX_TRAIN_STEPS * 0.1)  # linear warmup for the first N steps
STABLE_STEPS = int(MAX_TRAIN_STEPS * 0.7)
DECAY_STEPS = int(MAX_TRAIN_STEPS * 0.2)
PEAK_LR = 3e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 0.1
B1 = 0.9  # Adam beta1
B2 = 0.95  # Adam beta2
VAL_STEPS = 50  # how many batches to average for validation
CHECKPOINT_DIR = "./checkpoints"
MAX_GEN_TOKENS = 200  # max new tokens per chat response
MAX_CTX_LEN = (
    512  # rolling context window during generation — must be % CHUNK_SIZE == 0
)

assert SEQ_LEN % CHUNK_SIZE == 0, "SEQ_LEN must be divisible by CHUNK_SIZE"
assert MAX_CTX_LEN % CHUNK_SIZE == 0, "MAX_CTX_LEN must be divisible by CHUNK_SIZE"

# =============================================================================
# 1. Dataset helpers
# =============================================================================


def load_raw_texts(max_samples: int, skip: int = 0) -> list[str]:
    """Stream texts from HuggingFaceFW/fineweb-edu.

    Uses the 10B-token sample split so we never have to download the full
    dataset.  `skip` lets us draw training and validation from non-overlapping
    portions of the stream.
    """
    print(f"Loading {max_samples} texts from fineweb-edu (skip={skip}) ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )
    texts: list[str] = []
    for i, sample in enumerate(ds):
        if i < skip:
            continue
        texts.append(sample["text"])
        if len(texts) >= max_samples:
            break
    print(f"  Got {len(texts)} texts.")
    return texts


def tokenize_and_pack(texts: list[str], tokenizer, seq_len: int) -> np.ndarray:
    """Tokenize all texts, concatenate into one token stream, then cut into
    non-overlapping chunks of (seq_len + 1) tokens.

    Each chunk has seq_len+1 tokens so the training code can slice:
      inputs = chunk[:, :-1]   — seq_len tokens fed into the model
      labels = chunk[:, 1:]    — seq_len next-tokens used as targets

    Returns an array of shape (n_chunks, seq_len + 1).
    """
    chunk_len = seq_len + 1
    all_tokens: list[int] = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text))
        all_tokens.append(tokenizer.eos_token_id)  # mark document boundaries

    # Trim the tail so the total length divides evenly into chunks.
    n = (len(all_tokens) // chunk_len) * chunk_len
    return np.array(all_tokens[:n], dtype=np.int32).reshape(-1, chunk_len)


def make_batches(chunks: np.ndarray, batch_size: int):
    """Shuffle chunks once, then yield (batch_size, chunk_len) arrays."""
    idx = np.random.permutation(len(chunks))
    chunks = chunks[idx]
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        yield chunks[i : i + batch_size]


# =============================================================================
# 2. Model
# =============================================================================


def build_model(seed: int = 0) -> NemotronNanoBlock:
    """Build a tiny Nemotron configured for the GPT-2 vocabulary."""
    config = NemotronConfig.from_preset("tiny")  # tiny defaults (d_model=128, etc.)
    config.vocab_size = VOCAB_SIZE
    config.mamba_chunk_size = CHUNK_SIZE
    config.validate()
    return NemotronNanoBlock(rngs=nnx.Rngs(seed), config=config)


def collect_moe_layers(model: NemotronNanoBlock) -> list[SparseMoE]:
    """Collect every SparseMoE sub-module in the model."""
    return [block.moe for block in model.blocks]


# =============================================================================
# 3. Loss
# =============================================================================


def cross_entropy_loss(model: NemotronNanoBlock, batch: jax.Array) -> jax.Array:
    """Standard next-token prediction loss.

    batch: (B, seq_len + 1)
      inputs = batch[:, :-1]  →  fed into the model
      labels = batch[:, 1:]   →  the shifted-by-one targets
    """
    inputs = batch[:, :-1]  # (B, seq_len)
    labels = batch[:, 1:]   # (B, seq_len)
    logits = model(inputs)  # (B, seq_len, vocab_size)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()


# =============================================================================
# 4. Learning Rate Schedule
# =============================================================================


def create_lr_schedule(
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
) -> optax.Schedule:
    """
    Creates a two-phase learning rate schedule:
        Phase 1 — Warmup  : linear ramp from 0 → peak_lr
        Phase 2 — Stable  : constant at peak_lr
        Phase 3 — Decay   : cosine decay from peak_lr → min_lr

    This avoids early training instability (warmup) while allowing the
    optimizer to fine-tune at smaller learning rates later (cosine decay).
    """
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )
    stable = optax.constant_schedule(peak_lr)
    decay = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=decay_steps,
        alpha=min_lr / peak_lr,  # final LR = peak_lr * alpha = min_lr
    )

    return optax.join_schedules(
        schedules=[warmup, stable, decay],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )


def make_gradient_transform_optimizer(
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    weight_decay: float = 0.1,
    b1: float = 0.9,
    b2: float = 0.95,
) -> optax.GradientTransformation:
    """
    Creates an Optax gradient transformation for optimizer with the custom learning rate schedule.
    We use AdamW with weight decay, which is common for transformer training.
    """
    warmup_steps = max(
        warmup_steps, 1
    )  # sanity check to avoid division by zero in warmup schedule
    stable_steps = max(
        stable_steps, 1
    )  # sanity check to avoid division by zero in stable schedule
    decay_steps = max(
        decay_steps, 1
    )  # sanity check to avoid division by zero in decay schedule

    lr_schedule = create_lr_schedule(
        peak_lr=peak_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
    )

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay, b1=b1, b2=b2),
    )


# =============================================================================
# 5. Training step
# =============================================================================


@nnx.jit
def train_step(
    model: NemotronNanoBlock,
    optimizer: nnx.Optimizer,
    batch: jax.Array,
) -> jax.Array:
    """Compute gradients and update the model. Returns the scalar loss."""
    loss, grads = nnx.value_and_grad(
        cross_entropy_loss, argnums=nnx.DiffState(0, nnx.Param)
    )(model, batch)
    optimizer.update(model, grads)
    return loss


def update_moe_biases(moe_layers: list[SparseMoE]) -> None:
    """Update expert load-balancing biases for every MoE layer.

    Must be called AFTER the optimizer step and outside the gradient tape.
    Each SparseMoE stashes the top-k routing indices from its most recent
    forward pass in `self.last_topk_indices`; we use those to nudge biases so
    underloaded experts become easier to pick and overloaded ones harder.
    """
    for moe in moe_layers:
        moe.update_expert_bias(moe.last_topk_indices.get_value())


# =============================================================================
# 6. Checkpointing  (powered by Orbax)
# =============================================================================


def make_checkpoint_manager(
    ckpt_dir: str, max_to_keep: int = 1
) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager that keeps the last `max_to_keep` steps."""
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep)
    return ocp.CheckpointManager(pathlib.Path(ckpt_dir), options=options)


def save_checkpoint(
    manager: ocp.CheckpointManager,
    model: NemotronNanoBlock,
    step: int,
) -> None:
    """Save model state at `step` via the checkpoint manager."""
    _, state = nnx.split(model)
    manager.save(step, args=ocp.args.StandardSave(state), force=True)
    manager.wait_until_finished()
    print(f"  Checkpoint saved: step {step}")


def load_latest_checkpoint(
    manager: ocp.CheckpointManager,
    model: NemotronNanoBlock,
    config: NemotronConfig,
) -> int:
    """Restore the most recent checkpoint into model in-place.

    Returns the step number of the loaded checkpoint, or 0 if none found.
    We build an abstract (shape-only) model to tell Orbax the expected array
    shapes before it reads the files.
    """
    latest = manager.latest_step()
    if latest is None:
        return 0

    abstract_model = nnx.eval_shape(
        lambda: NemotronNanoBlock(rngs=nnx.Rngs(0), config=config)
    )

    _, abs_state = nnx.split(abstract_model)
    restored = manager.restore(latest, args=ocp.args.StandardRestore(abs_state))
    nnx.update(model, restored)
    print(f"  Resumed from checkpoint at step {latest}")
    return latest


# =============================================================================
# 7. Evaluation
# =============================================================================


def evaluate(
    model: NemotronNanoBlock,
    val_chunks: np.ndarray,
    val_steps: int,
) -> tuple[float, float]:
    """Return (mean_loss, perplexity) averaged over val_steps batches."""
    total_loss = 0.0
    count = 0
    for batch_np in make_batches(val_chunks, BATCH_SIZE):
        if count >= val_steps:
            break
        loss = cross_entropy_loss(model, jnp.array(batch_np))
        total_loss += float(loss)
        count += 1
    mean_loss = total_loss / max(count, 1)
    perplexity = math.exp(
        min(mean_loss, 20)
    )  # clamp to avoid overflow on a fresh model
    return mean_loss, perplexity


# =============================================================================
# 8. Generation
# =============================================================================


def generate(
    model: NemotronNanoBlock,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_GEN_TOKENS,
    temperature: float = 0.8,
    rng_seed: int = 42,
) -> str:
    """Generate text autoregressively with temperature sampling.

    Because the Mamba SSD kernel requires seqlen % chunk_size == 0, the
    context is always left-padded with <eos> tokens to the nearest multiple
    of CHUNK_SIZE before each forward pass.

    Note: this re-runs the full forward pass for every new token — simple and
    correct, but not fast.  Prefer generate_with_cache() for efficient inference.
    """
    prompt_tokens = tokenizer.encode(prompt)
    tokens = list(prompt_tokens)
    rng = jax.random.PRNGKey(rng_seed)

    for _ in range(max_new_tokens):
        # Keep only the most recent MAX_CTX_LEN tokens to bound memory.
        ctx = tokens[-MAX_CTX_LEN:]

        # Left-pad so length is a multiple of CHUNK_SIZE (minimum CHUNK_SIZE).
        pad_len = (-len(ctx)) % CHUNK_SIZE
        padded = [tokenizer.eos_token_id] * pad_len + ctx
        if not padded:  # guard: empty prompt → pad to one full chunk
            padded = [tokenizer.eos_token_id] * CHUNK_SIZE

        input_ids = jnp.array([padded])  # (1, padded_len)
        logits = model(input_ids)  # (1, padded_len, vocab_size)
        next_logits = logits[0, -1, :]  # last position: (vocab_size,)

        # Sample from the distribution scaled by temperature.
        rng, sample_rng = jax.random.split(rng)
        next_token = int(jax.random.categorical(sample_rng, next_logits / temperature))
        tokens.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(tokens[len(prompt_tokens) :])


def generate_with_cache(
    model: NemotronNanoBlock,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_GEN_TOKENS,
    temperature: float = 0.8,
    rng_seed: int = 42,
    max_cache_len: int = MAX_CTX_LEN + MAX_GEN_TOKENS,
) -> str:
    """Efficient cache-based autoregressive text generation.

    Maintains a KV cache (for attention layers) and an SSM state cache (for
    Mamba layers) across steps.  Each new token therefore costs O(1) instead
    of re-running the full context from scratch.

    Generation has two phases:
      1. Prefill  — step() through every prompt token to warm up the caches.
      2. Sampling — generate new tokens one at a time, reusing cached state.

    No chunk-size padding is required because step() processes one token at a
    time rather than calling the parallel SSD algorithm.

    Args:
        model:          Trained NemotronNanoBlock.
        tokenizer:      HuggingFace tokenizer.
        prompt:         Input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature:    Sampling temperature (lower → more deterministic).
        rng_seed:       PRNG seed for reproducible sampling.
        max_cache_len:  KV cache capacity.  Must be ≥ len(prompt_tokens) +
                        max_new_tokens.

    Returns:
        Generated text (the prompt itself is excluded).
    """
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        prompt_tokens = [tokenizer.eos_token_id]

    rng = jax.random.PRNGKey(rng_seed)

    # Initialise all caches to zero for the first step.
    caches = model.init_caches(batch_size=1, max_attn_len=max_cache_len)

    # ── Phase 1: prefill ──────────────────────────────────────────────────────
    # Run every prompt token through step() so the SSM states and KV caches
    # reflect the full prompt context.  The final call returns logits that
    # predict the first generated token.
    logits = None
    for tok in prompt_tokens:
        logits, caches = model.step(jnp.array([tok]), caches)

    # ── Phase 2: sampling ─────────────────────────────────────────────────────
    # At this point `logits` holds the distribution over the next token.
    # We sample, append, then step() with the new token to get the next logits.
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_logits = logits[0]  # (vocab_size,)
        rng, sample_rng = jax.random.split(rng)
        next_token = int(jax.random.categorical(sample_rng, next_logits / temperature))
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        logits, caches = model.step(jnp.array([next_token]), caches)

    return tokenizer.decode(generated)


# =============================================================================
# 9. Chat loop
# =============================================================================


def chat(model: NemotronNanoBlock, tokenizer) -> None:
    print("\n--- Chat mode  (type 'quit' to exit) ---\n")
    seed = 0
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not prompt:
            continue
        response = generate_with_cache(model, tokenizer, prompt, rng_seed=seed)
        seed += 1
        print(f"Model: {response}\n")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("Loading Nemotron tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    # This tokenizer has no pad token by default; reuse eos for padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    # Stream 2 000 texts for training, then 200 more (skipped past train) for
    # validation so the two sets never overlap.
    # Increase max_samples for a longer/better training run.
    train_texts = load_raw_texts(max_samples=10000, skip=0)
    val_texts = load_raw_texts(max_samples=200, skip=10000)

    train_chunks = tokenize_and_pack(train_texts, tokenizer, SEQ_LEN)
    val_chunks = tokenize_and_pack(val_texts, tokenizer, SEQ_LEN)
    print(f"Train chunks: {len(train_chunks)},  Val chunks: {len(val_chunks)}")

    # ── 3. Model + optimizer ──────────────────────────────────────────────────
    print("\nBuilding model ...")
    config = NemotronConfig().from_preset("tiny")  # tiny defaults (d_model=128, etc.)
    config.vocab_size = VOCAB_SIZE
    config.mamba_chunk_size = CHUNK_SIZE
    config.validate()

    model = build_model(seed=0)
    moe_layers = collect_moe_layers(model)

    optimizer = nnx.Optimizer(
        model,
        make_gradient_transform_optimizer(
            PEAK_LR,
            MIN_LR,
            WARMUP_STEPS,
            STABLE_STEPS,
            DECAY_STEPS,
            weight_decay=WEIGHT_DECAY,
            b1=B1,
            b2=B2,
        ),
        wrt=nnx.Param,
    )

    # Create checkpoint manager; resume from the latest step if one exists.
    ckpt_manager = make_checkpoint_manager(CHECKPOINT_DIR)
    start_step = load_latest_checkpoint(ckpt_manager, model, config)

    # ── 4. Training loop ──────────────────────────────────────────────────────
    print(
        f"\nTraining for {MAX_TRAIN_STEPS} steps "
        f"(batch={BATCH_SIZE}, seq_len={SEQ_LEN}) ..."
    )
    print("(The first step is slow — JAX JIT-compiles the model.)\n")

    step = start_step
    batch_iter = iter(make_batches(train_chunks, BATCH_SIZE))

    while step < MAX_TRAIN_STEPS:
        # Refill the iterator when one pass over the data is done.
        try:
            batch_np = next(batch_iter)
        except StopIteration:
            batch_iter = iter(make_batches(train_chunks, BATCH_SIZE))
            batch_np = next(batch_iter)

        loss = train_step(model, optimizer, jnp.array(batch_np))

        # Update MoE expert biases outside the gradient tape.
        update_moe_biases(moe_layers)

        step += 1

        if step % 10 == 0:
            print(f"  step {step:5d} / {MAX_TRAIN_STEPS}  |  loss {float(loss):.4f}")

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(ckpt_manager, model, step)

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    print("\nEvaluating on validation set ...")
    val_loss, val_ppl = evaluate(model, val_chunks, VAL_STEPS)
    print(f"  Validation loss : {val_loss:.4f}")
    print(f"  Perplexity      : {val_ppl:.2f}")

    # ── 6. Final checkpoint ───────────────────────────────────────────────────
    save_checkpoint(ckpt_manager, model, step)
    ckpt_manager.close()

    # ── 7. Interactive chat ───────────────────────────────────────────────────
    chat(model, tokenizer)


if __name__ == "__main__":
    main()

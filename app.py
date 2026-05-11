"""
Simple, minimal, and explainable Nemotron app.

This script shows a full small workflow:
1) Load a real dataset (roneneldan/TinyStories)
2) Train a tiny Nemotron language model
3) Evaluate it with validation loss + perplexity
4) Chat with it in the terminal

Design goals:
- Keep code easy to read and modify.
- Keep control flow explicit.
- Prefer clarity over speed/optimization.
"""

from __future__ import annotations

import argparse
import math

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from nemotron import NemotronConfig, NemotronNanoBlock


class CharTokenizer:
    """
    Very small character-level tokenizer.

    Why character-level?
    - It is the simplest reversible tokenizer.
    - It is easy to explain and debug.
    - It avoids extra external dependencies.
    """

    def __init__(self) -> None:
        # Reserve a few special IDs that help batching/generation.
        self.special_tokens = ["<pad>", "<bos>", "<eos>"]
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def fit(self, texts: list[str]) -> None:
        """Builds a vocabulary from all unique characters in the corpus."""
        charset: set[str] = set()
        for text in texts:
            charset.update(text)

        # Keep character order deterministic by sorting.
        vocab = self.special_tokens + sorted(charset)
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Converts text to token IDs."""
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        for ch in text:
            if ch in self.token_to_id:
                ids.append(self.token_to_id[ch])

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Converts token IDs back to text."""
        chars: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "")
            if skip_special and token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)


def _extract_story_text(example: dict[str, object]) -> str:
    """
    Reads one TinyStories row and returns its text.

    We check a few keys defensively so this stays easy to understand even if
    the upstream schema changes slightly.
    """
    for key in ("text", "story", "content"):
        value = example.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
    return ""


def load_tinystories_texts(
    max_stories: int,
    split: str = "train",
    cache_dir: str | None = None,
) -> list[str]:
    """
    Loads a bounded number of stories from roneneldan/TinyStories.

    Why a bounded subset?
    - Keeps the demo easy to run locally.
    - Keeps the data flow simple and inspectable.
    - Streams rows and stops early at `max_stories`.
    """
    if max_stories < 2:
        raise ValueError("max_stories must be at least 2")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "TinyStories training requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    try:
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            cache_dir=cache_dir,
            streaming=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load roneneldan/TinyStories. "
            "Check your internet connection and cache permissions."
        ) from exc

    stories: list[str] = []

    # Intentionally iterate in plain Python for readability.
    for example in dataset:
        story = _extract_story_text(example)
        if story:
            stories.append(story)
        if len(stories) >= max_stories:
            break

    if len(stories) < 2:
        raise ValueError("TinyStories did not return enough non-empty stories")

    return stories


def split_train_val_texts(
    stories: list[str],
    train_ratio: float,
) -> tuple[list[str], list[str]]:
    """
    Splits stories into train and validation lists.

    This uses a deterministic ordered split to keep behavior simple and
    reproducible.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(stories) < 2:
        raise ValueError("Need at least 2 stories for train/validation split")

    split_index = int(len(stories) * train_ratio)
    split_index = max(1, split_index)
    split_index = min(split_index, len(stories) - 1)

    train_texts = stories[:split_index]
    val_texts = stories[split_index:]
    return train_texts, val_texts


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Standard language-model cross-entropy."""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return optax.softmax_cross_entropy(logits, one_hot).mean()


def total_training_loss(
    logits: jax.Array,
    labels: jax.Array,
    moe_aux_loss: jax.Array,
    moe_aux_loss_weight: float,
) -> jax.Array:
    """
    Combines cross-entropy loss with the MoE auxiliary loss term.

    With aux-loss-free load balancing, SparseMoE always returns jnp.zeros(())
    for the aux loss, so this is effectively just cross-entropy. The weight
    parameter and moe_aux_loss argument are kept for API compatibility.
    """
    ce = cross_entropy_loss(logits, labels)
    return ce + moe_aux_loss_weight * moe_aux_loss


def _update_all_expert_biases(model: NemotronNanoBlock) -> None:
    """
    Update expert biases across every MoE layer after a training step.

    This is the aux-loss-free load balancing step (Wang et al. 2024, §2.4).
    Each MoE layer stored the top-k indices from its last forward pass in
    `moe.last_topk_indices`. We read those indices here and call
    `update_expert_bias`, which nudges each expert's bias by +/-bias_update_rate
    depending on whether that expert was over- or under-utilized.

    IMPORTANT: Call this AFTER optimizer.update, outside the gradient computation.
    The expert_bias is an nnx.Variable (not nnx.Param) so the optimizer does
    not touch it — it is updated only here.
    """
    for i in range(model.num_layers):
        block = getattr(model, f"block_{i}")
        block.moe.update_expert_bias(block.moe.last_topk_indices.value)


def _ensure_min_length(tokens: jax.Array, min_length: int) -> jax.Array:
    """
    Repeats tokens until we have enough positions for batching.

    This keeps the data pipeline simple for very tiny demo corpora.
    """
    if int(tokens.shape[0]) >= min_length:
        return tokens

    repeat_count = int(math.ceil(min_length / int(tokens.shape[0])))
    return jnp.tile(tokens, repeat_count)


def prepare_datasets(
    tokenizer: CharTokenizer,
    train_texts: list[str],
    val_texts: list[str],
    seq_len: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Creates train/val token streams from TinyStories text.

    Output format:
    - train_tokens: shape (num_train_tokens,)
    - val_tokens: shape (num_val_tokens,)
    """
    if not train_texts or not val_texts:
        raise ValueError("train_texts and val_texts must both be non-empty")

    # Keep train/validation streams separate so evaluation stays honest.
    train_joined = "\n\n".join(train_texts)
    val_joined = "\n\n".join(val_texts)

    # Add BOS/EOS so the model can learn sequence boundaries.
    train_ids = tokenizer.encode(train_joined, add_bos=True, add_eos=True)
    val_ids = tokenizer.encode(val_joined, add_bos=True, add_eos=True)

    train_tokens = jnp.array(train_ids, dtype=jnp.int32)
    val_tokens = jnp.array(val_ids, dtype=jnp.int32)

    # Ensure both splits are large enough to sample (x, y) windows.
    min_stream_len = seq_len + 2
    train_tokens = _ensure_min_length(train_tokens, min_stream_len)
    val_tokens = _ensure_min_length(val_tokens, min_stream_len)

    return train_tokens, val_tokens


def sample_lm_batch(
    token_stream: jax.Array,
    batch_size: int,
    seq_len: int,
    rng_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Samples random contiguous windows for next-token prediction.

    For each sampled window of length (seq_len + 1):
    - x = first seq_len tokens
    - y = next seq_len tokens
    """
    max_start = int(token_stream.shape[0]) - (seq_len + 1)
    if max_start < 0:
        raise ValueError("token_stream is shorter than seq_len + 1")

    starts = jax.random.randint(rng_key, (batch_size,), 0, max_start + 1)

    x_list: list[jax.Array] = []
    y_list: list[jax.Array] = []
    for start in starts.tolist():
        window = token_stream[start : start + seq_len + 1]
        x_list.append(window[:-1])
        y_list.append(window[1:])

    x = jnp.stack(x_list, axis=0)
    y = jnp.stack(y_list, axis=0)
    return x, y


def train_model(
    model: NemotronNanoBlock,
    optimizer: nnx.Optimizer,
    train_tokens: jax.Array,
    config: NemotronConfig,
    steps: int,
    batch_size: int,
    seq_len: int,
    rng_key: jax.Array,
) -> jax.Array:
    """Runs a tiny training loop and prints readable metrics."""

    @nnx.jit
    def train_step(model, optimizer, x_batch, y_batch):
        def loss_fn(model):
            logits_local, moe_aux_local = model(x_batch, return_aux_loss=True)
            total = total_training_loss(
                logits=logits_local,
                labels=y_batch,
                moe_aux_loss=moe_aux_local,
                moe_aux_loss_weight=config.moe_aux_loss_weight,
            )
            ce = cross_entropy_loss(logits_local, y_batch)
            return total, (ce, moe_aux_local)

        (total_loss, (ce_loss, moe_aux_loss)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)
        optimizer.update(model, grads)

        # Aux-loss-free load balancing: update expert biases AFTER the gradient step.
        # Uses the top-k indices stored by each SparseMoE during the forward pass.
        _update_all_expert_biases(model)

        return total_loss, ce_loss, moe_aux_loss

    print("\nTraining:")
    for step in range(steps):
        rng_key, batch_key = jax.random.split(rng_key)
        x_batch, y_batch = sample_lm_batch(train_tokens, batch_size, seq_len, batch_key)
        total_loss, ce_loss, moe_aux = train_step(model, optimizer, x_batch, y_batch)

        print(
            f"  step {step + 1:>3}/{steps} | "
            f"total={float(total_loss):.4f} | "
            f"ce={float(ce_loss):.4f} | "
            f"moe_aux={float(moe_aux):.4f}"
        )

        """ if step == 0 or (step + 1) % max(1, steps // 5) == 0 or step == steps - 1:
            print(
                f"  step {step + 1:>3}/{steps} | "
                f"total={float(total_loss):.4f} | "
                f"ce={float(ce_loss):.4f} | "
                f"moe_aux={float(moe_aux):.4f}"
            ) """

    return rng_key


def evaluate_model(
    model: NemotronNanoBlock,
    val_tokens: jax.Array,
    config: NemotronConfig,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    rng_key: jax.Array,
) -> tuple[float, float, float, jax.Array]:
    """
    Evaluates model on validation batches.

    Returns:
    - mean_total_loss
    - mean_ce_loss
    - perplexity = exp(mean_ce_loss)
    - updated rng_key
    """
    total_losses: list[jax.Array] = []
    ce_losses: list[jax.Array] = []

    for _ in range(eval_batches):
        rng_key, batch_key = jax.random.split(rng_key)
        x_batch, y_batch = sample_lm_batch(val_tokens, batch_size, seq_len, batch_key)
        logits, moe_aux = model(x_batch, return_aux_loss=True)
        ce = cross_entropy_loss(logits, y_batch)
        total = ce + config.moe_aux_loss_weight * moe_aux

        total_losses.append(total)
        ce_losses.append(ce)

    mean_total = jnp.mean(jnp.stack(total_losses))
    mean_ce = jnp.mean(jnp.stack(ce_losses))
    ppl = jnp.exp(mean_ce)

    return float(mean_total), float(mean_ce), float(ppl), rng_key


def pad_or_trim_context(token_ids: list[int], seq_len: int, pad_id: int) -> jax.Array:
    """
    Makes context length exactly `seq_len` so Mamba chunking constraints hold.

    - If context is too long, keep the most recent tokens.
    - If context is too short, left-pad with <pad>.
    """
    if len(token_ids) >= seq_len:
        fixed = token_ids[-seq_len:]
    else:
        pad_count = seq_len - len(token_ids)
        fixed = [pad_id] * pad_count + token_ids

    return jnp.array([fixed], dtype=jnp.int32)


def sample_next_token(
    logits: jax.Array,
    temperature: float,
    rng_key: jax.Array,
) -> int:
    """
    Picks the next token.

    - temperature <= 0: greedy argmax
    - temperature > 0: categorical sampling after temperature scaling
    """
    if temperature <= 0.0:
        return int(jnp.argmax(logits))

    safe_temp = max(temperature, 1e-6)
    scaled = logits / safe_temp
    return int(jax.random.categorical(rng_key, scaled))


def generate_reply(
    model: NemotronNanoBlock,
    tokenizer: CharTokenizer,
    prompt_text: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    rng_key: jax.Array,
) -> tuple[str, jax.Array]:
    """Autoregressively generates assistant text from a prompt."""
    context_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        model_input = pad_or_trim_context(context_ids, seq_len, tokenizer.pad_id)
        logits = model(model_input)

        next_logits: jax.Array = jnp.zeros(0)
        if isinstance(logits, jax.Array):
            next_logits = logits[0, -1]

        # Avoid sampling these two control tokens during response generation.
        next_logits = next_logits.at[tokenizer.pad_id].set(-1e9)
        next_logits = next_logits.at[tokenizer.bos_id].set(-1e9)

        rng_key, sample_key = jax.random.split(rng_key)
        next_id = sample_next_token(next_logits, temperature, sample_key)

        if next_id == tokenizer.eos_id:
            break

        context_ids.append(next_id)
        generated_ids.append(next_id)

    return tokenizer.decode(generated_ids, skip_special=True), rng_key


def chat_loop(
    model: NemotronNanoBlock,
    tokenizer: CharTokenizer,
    seq_len: int,
    temperature: float,
    max_new_tokens: int,
    rng_key: jax.Array,
) -> None:
    """Runs a simple interactive terminal chatbot."""
    print("\nChatbot is ready. Type 'quit' or 'exit' to stop.")

    # Tiny system instruction to stabilize output format.
    history = "System: You are a concise and helpful assistant.\n"

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            print("Bot: bye.")
            break
        if not user_text:
            continue

        history += f"User: {user_text}\nAssistant: "
        reply, rng_key = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt_text=history,
            seq_len=seq_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            rng_key=rng_key,
        )
        reply = reply.strip() or "..."

        print(f"Bot: {reply}")
        history += reply + "\n"

        # Keep only recent history so context stays bounded and simple.
        history = history[-1200:]

def create_lr_schedule(max_steps: int, warmup_steps: int, peak_lr: float) -> optax.Schedule:
    """
    Creates a two-phase learning rate schedule:
      Phase 1 (steps 0 .. warmup_steps):        linear ramp  0 → peak_lr
      Phase 2 (steps warmup_steps .. max_steps): cosine decay peak_lr → 0

    This avoids early training instability (warmup) while allowing the
    optimizer to fine-tune at smaller learning rates later (cosine decay).
    """
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )
    decay = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=max(max_steps - warmup_steps, 1),
    )
    return optax.join_schedules(
        schedules=[warmup, decay],
        boundaries=[warmup_steps],
    )

def build_arg_parser() -> argparse.ArgumentParser:
    """CLI arguments kept intentionally small and beginner-friendly."""
    parser = argparse.ArgumentParser(description="Minimal Nemotron train/eval/chat app")
    parser.add_argument("--preset", type=str, default="tiny", help="Nemotron preset")
    parser.add_argument("--steps", type=int, default=80, help="Training steps")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Train/eval batch size"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length (must be divisible by Mamba chunk size)",
    )
    parser.add_argument(
        "--eval-batches", type=int, default=10, help="Validation batches"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 for greedy decoding, >0 for sampling",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=80, help="Max tokens per reply"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--tinystories-max-stories",
        type=int,
        default=5000,
        help="How many TinyStories stories to load for this run",
    )
    parser.add_argument(
        "--tinystories-train-ratio",
        type=float,
        default=0.9,
        help="Fraction of stories used for training (rest for validation)",
    )
    parser.add_argument(
        "--tinystories-split",
        type=str,
        default="train",
        help="Hugging Face split to read from TinyStories",
    )
    parser.add_argument(
        "--tinystories-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--preview-first-story",
        action="store_true",
        help="Print a short preview of the first loaded TinyStories sample",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Run train+eval only (useful for non-interactive testing)",
    )
    return parser


def main() -> None:
    print("Initializing minimal Nemotron app...")

    # 1) Load real text data from TinyStories.
    all_stories = load_tinystories_texts(
        max_stories=5000,
        split="train",
        cache_dir=None,
    )
    train_texts, val_texts = split_train_val_texts(
        stories=all_stories,
        train_ratio=0.9,
    )

    print(
        "Dataset setup: "
        f"total_stories={len(all_stories)}, "
        f"train_stories={len(train_texts)}, "
        f"val_stories={len(val_texts)}"
    )

    # Optional preview: helps beginners see real input text before tokenization.
    if True:
        preview = all_stories[0].replace("\n", " ").strip()
        preview_limit = 240
        if len(preview) > preview_limit:
            preview = preview[:preview_limit] + "..."

        print("\nTinyStories preview (first loaded sample):")
        print(f"  {preview}")

    # 2) Build tokenizer from dataset text.
    # We fit on all loaded stories so train and validation text are always encodable.
    tokenizer = CharTokenizer()
    tokenizer.fit(all_stories)

    # 3) Build Nemotron config/model.
    config = NemotronConfig.from_preset("tiny")
    config.vocab_size = tokenizer.vocab_size

    # if args.seq_len % config.mamba_chunk_size != 0:
    #     raise ValueError(
    #         f"seq_len must be divisible by mamba_chunk_size ({config.mamba_chunk_size})"
    #     )

    # print(
    #     "Model setup: "
    #     f"preset={args.preset}, vocab_size={config.vocab_size}, "
    #     f"seq_len={args.seq_len}, chunk_size={config.mamba_chunk_size}"
    # )

    rngs = nnx.Rngs(0)
    model = NemotronNanoBlock(rngs=rngs, config=config)

    # Build optimizer: AdamW with warmup + cosine decay
    lr_schedule = create_lr_schedule(
        max_steps=1000,
        warmup_steps=200,
        peak_lr=3e-4,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=0.1),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Use a separate key stream for data/generation randomness.
    rng_key = jax.random.PRNGKey(0 + 1)

    # 4) Prepare token streams from train/validation stories.
    train_tokens, val_tokens = prepare_datasets(
        tokenizer=tokenizer,
        train_texts=train_texts,
        val_texts=val_texts,
        seq_len=64,
    )

    # 5) Train.
    rng_key = train_model(
        model=model,
        optimizer=optimizer,
        train_tokens=train_tokens,
        config=config,
        steps=100000,
        batch_size=32,
        seq_len=64,
        rng_key=rng_key,
    )

    # 6) Evaluate.
    mean_total, mean_ce, perplexity, rng_key = evaluate_model(
        model=model,
        val_tokens=val_tokens,
        config=config,
        batch_size=32,
        seq_len=64,
        eval_batches=32,
        rng_key=rng_key,
    )

    print("\nEvaluation:")
    print(f"  mean total loss: {mean_total:.4f}")
    print(f"  mean CE loss:    {mean_ce:.4f}")
    print(f"  perplexity:      {perplexity:.4f}")

    # 7) Chat.
    if not False:
        chat_loop(
            model=model,
            tokenizer=tokenizer,
            seq_len=64,
            temperature=0.0,
            max_new_tokens=200,
            rng_key=rng_key,
        )


if __name__ == "__main__":
    main()

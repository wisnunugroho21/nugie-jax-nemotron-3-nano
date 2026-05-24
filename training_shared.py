"""
training_shared.py — Shared constants, helpers, and utilities for all training stages.

This module is imported by each individual training-stage script and by the
full-pipeline orchestrator (full_training.py).  Nothing in this file should
be run directly.
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
from jax.sharding import NamedSharding, PartitionSpec as P

from device_mesh import MESH, NUM_DEVICES, DATA_SHARDING, REPLICATED_SHARDING
from moe import SparseMoE
from nemotron import NemotronConfig, NemotronNanoBlock

# Aliases kept for internal use; device_mesh.py is the single source of truth.
_DATA_SHARDING: NamedSharding = DATA_SHARDING
_REPLICATED_SHARDING: NamedSharding = REPLICATED_SHARDING


def shard_batch(arr) -> jax.Array:
    """Place a (batch, ...) array on all devices, sharded along the batch axis.

    The batch dimension must be divisible by NUM_DEVICES.  When NUM_DEVICES==1
    this is equivalent to a plain ``jnp.asarray`` call.
    """
    arr = jnp.asarray(arr)
    if arr.shape[0] % NUM_DEVICES != 0:
        raise ValueError(
            f"Batch size {arr.shape[0]} must be divisible by NUM_DEVICES "
            f"({NUM_DEVICES}).  Multiply your batch-size constant by "
            f"{NUM_DEVICES} (or a multiple) so each device gets at least "
            f"one sample."
        )
    return jax.device_put(arr, _DATA_SHARDING)


def _replicate_to_devices(model: NemotronNanoBlock) -> None:
    """Replicate model parameters to all devices in-place.

    After this call every variable in ``model`` is backed by a replicated
    ``jax.Array`` (``NamedSharding(MESH, P())``) so that data-parallel
    training steps compiled by XLA can see consistent weights on every device.
    On a single device this is a cheap no-op.
    """
    if NUM_DEVICES == 1:
        return
    graphdef, state = nnx.split(model)
    state = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, _REPLICATED_SHARDING), state
    )
    nnx.update(model, state)
    print(f"  Model replicated across {NUM_DEVICES} devices: {jax.devices()}")


# =============================================================================
# Shared constants (tokenizer & architecture)
# =============================================================================

VOCAB_SIZE = 131_072
CHUNK_SIZE = 64


# =============================================================================
# PRE-TRAINING Hyperparameters
# =============================================================================

PRETRAIN_SEQ_LEN   = 256
PRETRAIN_BATCH     = 2

PHASE1_STEPS       = 5_000
PHASE2_STEPS       = 500
LC_PHASE_STEPS     = 100

PRETRAIN_PEAK_LR   = 1e-3
PRETRAIN_MIN_LR    = 1e-5
PRETRAIN_WARMUP_STEPS  = max(1, int((PHASE1_STEPS + PHASE2_STEPS) * 0.05))
PRETRAIN_STABLE_STEPS  = max(1, int((PHASE1_STEPS + PHASE2_STEPS) * 0.75))
PRETRAIN_DECAY_STEPS   = max(1, int((PHASE1_STEPS + PHASE2_STEPS) * 0.20))

LC_PHASE_LR        = 1e-5
LC_SEQ_LEN         = 512

PRETRAIN_B1        = 0.9
PRETRAIN_B2        = 0.95
PRETRAIN_WD        = 0.1

AUX_LOSS_COEFF     = 1e-4

PRETRAIN_CKPT_DIR  = "./checkpoints_pretrain"
PRETRAIN_CKPT_EVERY = 200

PRETRAIN_VAL_STEPS = 30


# =============================================================================
# POST-TRAINING — SFT Hyperparameters
# =============================================================================

SFT_SEQ_LEN      = 256
SFT_BATCH        = 2
SFT_STEPS        = 300
SFT_LR           = 5e-5
SFT_MIN_LR       = 1e-7
SFT_WARMUP_STEPS = max(1, int(SFT_STEPS * 0.06))
SFT_WD           = 0.1
SFT_B1           = 0.9
SFT_B2           = 0.95

SFT_REASONING_OFF_PROB  = 0.10
SFT_BUDGET_CONTROL_PROB = 0.03

SFT_CKPT_DIR     = "./checkpoints_sft"
SFT_CKPT_EVERY   = 100


# =============================================================================
# POST-TRAINING — RLVR Hyperparameters
# =============================================================================

RLVR_NUM_PROMPTS     = 4
RLVR_NUM_GENERATIONS = 4
RLVR_STEPS           = 100
RLVR_BATCH_STEPS     = 4

RLVR_KL_COEFF        = 0.04
RLVR_CLIP_EPS        = 0.2

RLVR_MAX_NEW_TOKENS  = 150
RLVR_TEMPERATURE     = 0.8
RLVR_FREEZE_ROUTER   = True

RLVR_LR              = 1e-5
RLVR_MIN_LR          = 1e-7
RLVR_WD              = 0.1
RLVR_B1              = 0.9
RLVR_B2              = 0.95

RLVR_CKPT_DIR        = "./checkpoints_rlvr"
RLVR_CKPT_EVERY      = 50

RLVR_MAX_CACHE_LEN   = SFT_SEQ_LEN + RLVR_MAX_NEW_TOKENS + 64
RL_TRAIN_SEQ_LEN     = SFT_SEQ_LEN + 1


# =============================================================================
# POST-TRAINING — RLHF Hyperparameters
# =============================================================================

RLHF_NUM_PROMPTS     = 4
RLHF_NUM_RESPONSES   = 4

RLHF_LAMBDA_THINK    = 0.5
RLHF_LAMBDA_ANSWER   = 0.5

RLHF_BETA_THINK      = 0.5
RLHF_BETA_ANSWER     = 0.5
RLHF_PERCENTILE      = 80

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
    """Construct a tiny Nemotron model replicated across all available devices."""
    config = NemotronConfig.from_preset("tiny")
    expected_d_model = config.num_attention_heads * config.attention_head_dim
    if config.d_model != expected_d_model:
        config.d_model = expected_d_model
    config.vocab_size = VOCAB_SIZE
    config.mamba_chunk_size = CHUNK_SIZE
    config.validate()
    model = NemotronNanoBlock(rngs=nnx.Rngs(seed), config=config)
    _replicate_to_devices(model)
    return model


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
    """Build the Warmup-Stable-Decay (WSD) learning-rate schedule."""
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )
    stable = optax.constant_schedule(peak_lr)
    decay  = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=decay_steps,
        alpha=min_lr / peak_lr,
    )
    return optax.join_schedules(
        schedules=[warmup, stable, decay],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )


def make_decayed_lr_optimizer(
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    weight_decay: float = 0.1,
    b1: float = 0.9,
    b2: float = 0.95,
) -> optax.GradientTransformation:
    """AdamW + gradient clipping + WSD schedule."""
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
    """AdamW with a constant learning rate (used for LC-Phase and RLVR/RLHF)."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay, b1=b1, b2=b2),
    )


# =============================================================================
# 3. MoE load-balancing helpers
# =============================================================================


def compute_load_balance_loss(moe_layers: list[SparseMoE]) -> jax.Array:
    """Compute MoE load-balancing auxiliary loss (paper §2.4)."""
    total_loss = jnp.zeros(())
    for moe in moe_layers:
        indices = moe.last_topk_indices.get_value()
        if indices is None or indices.size == 0:
            continue
        num_experts = moe.num_routed_experts
        if indices.ndim not in (2, 3):
            raise ValueError(f"Unexpected top-k index shape: {indices.shape}")

        flat_indices = indices.reshape(-1)
        one_hot = jax.nn.one_hot(flat_indices, num_experts)
        f = one_hot.mean(axis=0)

        if hasattr(moe, "last_router_probs"):
            router_probs = moe.last_router_probs.get_value()
            if router_probs is not None and router_probs.size > 0:
                P = router_probs.mean(axis=0)
                total_loss = total_loss + num_experts * (f * P).sum()
                continue

        total_loss = total_loss + num_experts * (f * f).sum()

    return total_loss / max(len(moe_layers), 1)


def update_moe_biases(moe_layers: list[SparseMoE]) -> None:
    """Apply the aux-loss-free expert bias update (paper §2.4)."""
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
    """Stream text from HuggingFaceFW/fineweb-edu and pack into fixed chunks."""
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
        all_tokens.append(tokenizer.eos_token_id)

    n = (len(all_tokens) // chunk_len) * chunk_len
    if n == 0:
        raise RuntimeError("Not enough tokens for even one chunk; increase max_samples.")
    return np.array(all_tokens[:n], dtype=np.int32).reshape(-1, chunk_len)


def make_batches(chunks: np.ndarray, batch_size: int):
    """Shuffle chunks once, then yield (batch_size, chunk_len) sharded batches."""
    idx = np.random.permutation(len(chunks))
    chunks = chunks[idx]
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        yield shard_batch(chunks[i : i + batch_size])


def load_sft_data(split: str = "train") -> list[dict]:
    """Load GSM8K as a stand-in for the paper's full SFT dataset."""
    print(f"  Loading GSM8K ({split}) …")
    ds = load_dataset("gsm8k", "main", split=split)
    samples = [{"question": row["question"], "answer": row["answer"]} for row in ds]
    print(f"  Got {len(samples)} samples.")
    return samples


# =============================================================================
# 5. SFT data formatting and tokenisation
# =============================================================================


def maybe_strip_reasoning(response: str, rng: np.random.Generator) -> str:
    """Randomly remove the <think> block to enable 'reasoning off' mode (§3.1.5)."""
    if rng.random() < SFT_REASONING_OFF_PROB:
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    return response


def maybe_truncate_reasoning(response: str, rng: np.random.Generator) -> str:
    """Randomly shorten the <think> block to teach reasoning budget control (§3.1.5)."""
    if rng.random() < SFT_BUDGET_CONTROL_PROB:
        think_match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
        if think_match:
            trace = think_match.group(1)
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
    """Format a GSM8K sample into a (prompt, response) pair."""
    parts     = answer.split("####")
    reasoning = parts[0].strip()
    final_ans = parts[-1].strip()

    prompt   = f"User: {question}\nAssistant: "
    response = f"<think>\n{reasoning}\n</think>\n{final_ans}"

    response = maybe_strip_reasoning(response, rng)
    response = maybe_truncate_reasoning(response, rng)

    return prompt, response


def tokenize_with_mask(
    tokenizer,
    prompt: str,
    response: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokenize a (prompt, response) pair and build a per-token loss mask."""
    prompt_ids   = tokenizer.encode(prompt,   add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    response_ids.append(tokenizer.eos_token_id)

    full_ids  = prompt_ids + response_ids
    input_ids = full_ids[:-1]
    labels    = full_ids[1:]

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
    """Tokenize all SFT samples and yield (inputs, labels, mask) batches."""
    rng = np.random.default_rng(seed=42)

    all_inputs, all_labels, all_masks = [], [], []
    for sample in samples:
        prompt, response = format_sft_example(
            sample["question"], sample["answer"], rng
        )
        inp, lab, msk = tokenize_with_mask(tokenizer, prompt, response, seq_len)
        if msk.sum() == 0:
            continue
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
            shard_batch(stacked_inputs[start:end]),
            shard_batch(stacked_labels[start:end]),
            shard_batch(stacked_masks [start:end]),
        )


# =============================================================================
# 6. Loss functions
# =============================================================================


def pretrain_loss(
    model: NemotronNanoBlock,
    batch: jax.Array,
    moe_layers: list[SparseMoE],
) -> jax.Array:
    """Next-token prediction loss + MoE load-balance auxiliary loss."""
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    logits = model(inputs)

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
    """Masked cross-entropy for supervised fine-tuning."""
    logits = model(inputs)
    ce     = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    masked_sum   = (ce * mask).sum()
    unmasked_cnt = jnp.maximum(mask.sum(), 1.0)
    sft_ce_loss = masked_sum / unmasked_cnt

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
        loss = float(pretrain_loss(model, batch_np, moe_layers))
        total_loss += loss
        count += 1
    mean_loss = total_loss / max(count, 1)
    perplexity = math.exp(min(mean_loss, 20))
    return mean_loss, perplexity


# =============================================================================
# 10. RL shared helpers
# =============================================================================


def generate_completion_tokens(
    model: NemotronNanoBlock,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = RLVR_MAX_NEW_TOKENS,
    temperature: float = RLVR_TEMPERATURE,
    rng_seed: int = 0,
) -> list[int]:
    """Autoregressively sample a completion for the given prompt token IDs."""
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]

    rng    = jax.random.PRNGKey(rng_seed)
    caches = model.init_caches(batch_size=1, max_attn_len=RLVR_MAX_CACHE_LEN)

    logits = None
    for tok in prompt_ids:
        logits, caches = model.step(jnp.array([tok]), caches)

    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_logits     = logits[0]
        rng, sample_rng = jax.random.split(rng)
        next_token      = int(jax.random.categorical(sample_rng, next_logits / temperature))
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
        logits, caches  = model.step(jnp.array([next_token]), caches)

    return generated


def compute_verifiable_reward(completion_text: str, ground_truth: str) -> float:
    """Score a completion against the verifiable ground-truth answer."""
    reward = 0.0
    if "<think>" in completion_text:
        reward += 0.1
    if "</think>" in completion_text:
        reward += 0.1

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
    """Normalise a group of rewards to GRPO advantages."""
    r = np.array(rewards, dtype=np.float32)
    return (r - r.mean()) / (r.std() + 1e-6)


def curriculum_sample_indices(
    pass_rates: np.ndarray,
    step: int,
    total_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Return sample indices according to the curriculum schedule (§3.2.2)."""
    if len(pass_rates) == 0:
        raise ValueError("pass_rates must not be empty.")

    target_difficulty = 0.3 + 0.4 * (step / max(total_steps, 1))
    difficulties = 1.0 - pass_rates

    weights = np.exp(-0.5 * ((difficulties - target_difficulty) / 0.2) ** 2).astype(np.float64)
    total_weight = float(weights.sum())
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        weights = np.full_like(weights, 1.0 / len(weights))
    else:
        weights /= total_weight

    replace = batch_size > len(pass_rates)
    chosen_idx = np.random.choice(len(pass_rates), size=batch_size, replace=replace, p=weights)
    return chosen_idx


def curriculum_sample(
    samples: list[dict],
    pass_rates: np.ndarray,
    step: int,
    total_steps: int,
    batch_size: int,
) -> list[dict]:
    """Sample tasks according to the curriculum schedule from §3.2.2."""
    chosen_idx = curriculum_sample_indices(
        pass_rates=pass_rates,
        step=step,
        total_steps=total_steps,
        batch_size=batch_size,
    )
    return [samples[i] for i in chosen_idx]


def build_grpo_batch(
    prompt_ids_list: list[list[int]],
    completion_groups: list[list[list[int]]],
    advantage_groups: list[np.ndarray],
    seq_len: int,
    pad_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pack prompt + completions into a fixed-shape (B×G, seq_len) batch."""
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

            pad_len = seq_len - len(full)
            full     += [pad_id] * pad_len
            mask_arr += [0.0]    * pad_len

            all_tokens.append(np.array(full[:seq_len],     dtype=np.int32))
            all_masks. append(np.array(mask_arr[:seq_len], dtype=np.float32))
            all_advantages.append(float(adv))

    return (
        shard_batch(np.stack(all_tokens)),
        shard_batch(np.stack(all_masks)),
        shard_batch(np.array(all_advantages, dtype=np.float32)),
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
    """GRPO objective: clipped-ratio policy gradient + unbiased KL penalty."""
    inputs  = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    masks_  = masks[:, 1:]

    logits_policy = model(inputs)
    log_pi_policy = jax.nn.log_softmax(logits_policy, axis=-1)
    log_p_policy = jnp.take_along_axis(
        log_pi_policy, targets[:, :, None], axis=-1
    )[:, :, 0]

    ratio = jnp.exp(log_p_policy - old_log_probs)

    adv = advantages[:, None]
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_loss = -jnp.minimum(ratio * adv, clipped_ratio * adv)

    delta   = ref_log_probs - log_p_policy
    kl_pen  = jnp.exp(delta) - delta - 1.0

    total_loss = pg_loss + kl_coeff * kl_pen
    masked_sum = (total_loss * masks_).sum()
    n_tokens   = jnp.maximum(masks_.sum(), 1.0)
    return masked_sum / n_tokens


@nnx.jit
def rl_step(
    model: NemotronNanoBlock,
    optimizer: nnx.Optimizer,
    token_ids: jax.Array,
    masks: jax.Array,
    advantages: jax.Array,
    ref_log_probs: jax.Array,
    old_log_probs: jax.Array,
) -> float:
    """Single GRPO gradient update.  Returns the scalar loss."""
    def _loss_fn(m):
        return grpo_loss(m, token_ids, masks, advantages, ref_log_probs, old_log_probs)

    avg_loss = 0.0
    for _ in range(RLVR_BATCH_STEPS):
        loss, grads = nnx.value_and_grad(_loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
        optimizer.update(model, grads)
        avg_loss += loss

    return avg_loss / RLVR_BATCH_STEPS


@nnx.jit
def compute_log_probs(ref_model: NemotronNanoBlock, token_ids: jax.Array) -> jax.Array:
    """Compute policy token log-probs outside the gradient tape."""
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    logits_ref = ref_model(inputs)
    log_pi_ref = jax.nn.log_softmax(logits_ref, axis=-1)
    return jnp.take_along_axis(log_pi_ref, targets[:, :, None], axis=-1)[:, :, 0]

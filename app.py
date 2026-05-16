"""
Simple, minimal, and explainable Nemotron app.

This script shows a full small workflow:
1) Load a conversational dataset (directly from Hugging Face, or from local JSONL files)
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
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nemotron import NemotronConfig, NemotronNanoBlock

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# =============================================================================
# Tokenizer
# =============================================================================


def load_hf_tokenizer(
    tokenizer_name: str,
    cache_dir: str | None = None,
) -> "PreTrainedTokenizerBase":
    """
    Loads a tokenizer from Hugging Face and guarantees core special tokens.

    We ensure PAD/BOS/EOS IDs exist because batching and generation rely on them.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Hugging Face tokenizer support requires the 'transformers' package. "
            "Install it with: pip install transformers"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            use_fast=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Hugging Face tokenizer '{tokenizer_name}'."
        ) from exc

    # Generation depends on EOS.
    if tokenizer.eos_token_id is None:
        if tokenizer.sep_token is not None:
            tokenizer.eos_token = tokenizer.sep_token
        else:
            tokenizer.add_special_tokens({"eos_token": "<eos>"})

    # We use BOS as a light conversation delimiter.
    if tokenizer.bos_token_id is None:
        if tokenizer.cls_token is not None:
            tokenizer.bos_token = tokenizer.cls_token
        else:
            tokenizer.add_special_tokens({"bos_token": "<bos>"})

    # Left padding is used for fixed-length context windows.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer


def _get_special_token_ids(
    tokenizer: "PreTrainedTokenizerBase",
) -> tuple[int, int, int]:
    """Returns guaranteed integer IDs for PAD/BOS/EOS tokens."""
    if (
        tokenizer.pad_token_id is None
        or tokenizer.bos_token_id is None
        or tokenizer.eos_token_id is None
    ):
        raise ValueError(
            "Tokenizer must define pad_token_id, bos_token_id, eos_token_id"
        )

    return (
        int(tokenizer.pad_token_id),  # type: ignore[arg-type]
        int(tokenizer.bos_token_id),  # type: ignore[arg-type]
        int(tokenizer.eos_token_id),  # type: ignore[arg-type]
    )


def encode_text(
    tokenizer: "PreTrainedTokenizerBase",
    text: str,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[int]:
    """Encodes text and optionally prepends/appends BOS/EOS."""
    _, bos_id, eos_id = _get_special_token_ids(tokenizer)

    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if add_bos:
        ids = [bos_id] + ids
    if add_eos:
        ids = ids + [eos_id]
    return ids


def _encode_segment(tokenizer: "PreTrainedTokenizerBase", text: str) -> list[int]:
    """Tokenizes raw text without adding any special tokens."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def encode_text_with_assistant_mask(
    tokenizer: "PreTrainedTokenizerBase",
    text: str,
    user_role_tag: str,
    assistant_role_tag: str,
) -> tuple[list[int], list[float]]:
    """
    Encodes one role-tagged conversation and returns per-token loss mask.

    Mask semantics:
    - 1.0 for assistant content tokens
    - 0.0 for user content tokens and role tag tokens
    """
    token_ids: list[int] = []
    loss_mask: list[float] = []

    if not user_role_tag or not assistant_role_tag:
        raise ValueError("user_role_tag and assistant_role_tag must be non-empty")

    cursor = 0
    current_role: str | None = None

    while cursor < len(text):
        next_user = text.find(user_role_tag, cursor)
        next_assistant = text.find(assistant_role_tag, cursor)

        candidates = [idx for idx in (next_user, next_assistant) if idx != -1]
        next_tag_index = min(candidates) if candidates else -1

        if next_tag_index == -1:
            chunk = text[cursor:]
            if chunk:
                chunk_ids = _encode_segment(tokenizer, chunk)
                mask_value = 1.0 if current_role == "assistant" else 0.0
                token_ids.extend(chunk_ids)
                loss_mask.extend([mask_value] * len(chunk_ids))
            break

        if next_tag_index > cursor:
            chunk = text[cursor:next_tag_index]
            if chunk:
                chunk_ids = _encode_segment(tokenizer, chunk)
                mask_value = 1.0 if current_role == "assistant" else 0.0
                token_ids.extend(chunk_ids)
                loss_mask.extend([mask_value] * len(chunk_ids))

        if text.startswith(user_role_tag, next_tag_index):
            tag_text = user_role_tag
            current_role = "user"
        else:
            tag_text = assistant_role_tag
            current_role = "assistant"

        tag_ids = _encode_segment(tokenizer, tag_text)
        token_ids.extend(tag_ids)
        loss_mask.extend([0.0] * len(tag_ids))

        cursor = next_tag_index + len(tag_text)

    return token_ids, loss_mask


# =============================================================================
# Dataset helpers (shared by JSONL and Hugging Face loaders)
# =============================================================================


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _serialize_turns(turns: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for turn in turns:
        chunks.append(f"<|{turn['role']}|>\n{turn['text']}\n")
    return "".join(chunks)


def _stable_hash_fraction(text: str) -> float:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _get_field(row: dict, keys: tuple) -> object:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _map_role(raw_role: object) -> str | None:
    if not isinstance(raw_role, str):
        return None
    role = raw_role.strip().lower()
    if role in {"prompter", "user", "human"}:
        return "user"
    if role in {"assistant", "gpt", "model"}:
        return "assistant"
    return None


def _iter_paths_from_roots(
    root_ids: list[str],
    children_by_parent: dict[str, list[str]],
) -> list[list[str]]:
    all_paths: list[list[str]] = []

    def dfs(node_id: str, path: list[str]) -> None:
        children = children_by_parent.get(node_id, [])
        if not children:
            all_paths.append(path.copy())
            return
        for child_id in children:
            path.append(child_id)
            dfs(child_id, path)
            path.pop()

    for root_id in root_ids:
        dfs(root_id, [root_id])
    return all_paths


def _passes_quality_filters(
    turns: list[dict[str, str]],
    min_turns: int,
    max_turns: int,
    min_chars: int,
    max_chars: int,
) -> bool:
    if not (min_turns <= len(turns) <= max_turns):
        return False
    char_len = len(_serialize_turns(turns))
    return min_chars <= char_len <= max_chars


# =============================================================================
# Dataset (Hugging Face direct load)
# =============================================================================


def load_hf_dataset_texts(
    dataset_name: str = "OpenAssistant/oasst2",
    lang: str = "en",
    val_ratio: float = 0.02,
    min_turns: int = 2,
    max_turns: int = 20,
    min_chars: int = 32,
    max_chars: int = 6000,
    max_train_records: int = 50000,
    max_val_records: int = 5000,
) -> tuple[list[str], list[str]]:
    """
    Loads a conversational dataset directly from Hugging Face and returns
    train/val lists of serialized chat texts — no file I/O required.

    Compatible with OpenAssistant/oasst2 and similarly-structured datasets.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading from Hugging Face requires the 'datasets' package. "
            "Install with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split="train")

    node_by_id: dict[str, dict] = {}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    root_ids: list[str] = []

    for row in dataset:
        if not isinstance(row, dict):
            continue

        row_lang = _get_field(row, ("lang", "language"))
        if isinstance(row_lang, str) and row_lang.lower() != lang.lower():
            continue

        node_id_raw = _get_field(row, ("message_id", "id"))
        parent_id_raw = _get_field(row, ("parent_id", "parent_message_id"))

        if node_id_raw is None:
            continue

        node_id = str(node_id_raw)
        parent_id = str(parent_id_raw) if parent_id_raw is not None else None

        node_by_id[node_id] = row
        if parent_id is None or parent_id == "None":
            root_ids.append(node_id)
        else:
            children_by_parent[parent_id].append(node_id)

    for parent_id in list(children_by_parent.keys()):
        children_by_parent[parent_id].sort()
    root_ids = sorted(set(root_ids))

    paths = _iter_paths_from_roots(root_ids, children_by_parent)

    train_texts: list[str] = []
    val_texts: list[str] = []

    for path_idx, path_node_ids in enumerate(paths):
        turns: list[dict[str, str]] = []

        for node_id in path_node_ids:
            row = node_by_id.get(node_id)
            if row is None:
                continue

            mapped_role = _map_role(_get_field(row, ("role", "speaker")))
            if mapped_role not in {"user", "assistant"}:
                continue

            text_raw = _get_field(row, ("text", "message", "content"))
            if not isinstance(text_raw, str):
                continue

            text = _normalize_text(text_raw)
            if not text:
                continue

            if turns and turns[-1]["role"] == mapped_role:
                turns[-1]["text"] = f"{turns[-1]['text']}\n\n{text}"
            else:
                turns.append({"role": mapped_role, "text": text})

        if not turns:
            continue

        while turns and turns[0]["role"] != "user":
            turns.pop(0)
        while turns and turns[-1]["role"] != "assistant":
            turns.pop()

        if not turns:
            continue

        if not _passes_quality_filters(turns, min_turns, max_turns, min_chars, max_chars):
            continue

        base_id = f"oasst2_path_{path_idx:08d}"
        split = "val" if _stable_hash_fraction(base_id) < val_ratio else "train"

        if split == "train" and len(train_texts) >= max_train_records:
            continue
        if split == "val" and len(val_texts) >= max_val_records:
            continue

        serialized_text = _serialize_turns(turns)
        if split == "val":
            val_texts.append(serialized_text)
        else:
            train_texts.append(serialized_text)

    if len(train_texts) < 2:
        raise ValueError(
            f"Loaded fewer than 2 train samples from '{dataset_name}'. "
            "Try relaxing quality filters or increasing max_train_records."
        )
    if len(val_texts) < 2:
        raise ValueError(
            f"Loaded fewer than 2 val samples from '{dataset_name}'. "
            "Try increasing val_ratio or max_val_records."
        )

    print(
        f"HF dataset loaded: source={dataset_name}, lang={lang}, "
        f"train={len(train_texts)}, val={len(val_texts)}"
    )
    return train_texts, val_texts


# =============================================================================
# Dataset (JSONL conversational text)
# =============================================================================


def load_jsonl_texts(
    jsonl_path: str,
    max_records: int,
    text_key: str = "serialized_text",
) -> list[str]:
    """
    Loads text samples from a JSONL file.

    Each non-empty line must contain a JSON object with a string field at
    `text_key` (default: serialized conversation text).
    """
    if max_records <= 0:
        raise ValueError("max_records must be > 0")

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    texts: list[str] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if len(texts) >= max_records:
                break

            stripped = line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}"
                ) from exc

            value = obj.get(text_key) if isinstance(obj, dict) else None
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    texts.append(cleaned)

    if len(texts) < 2:
        raise ValueError(
            f"Need at least 2 non-empty records in {jsonl_path} for training/eval"
        )

    return texts


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
    tokenizer: "PreTrainedTokenizerBase",
    train_texts: list[str],
    val_texts: list[str],
    seq_len: int,
    assistant_only_loss: bool = False,
    user_role_tag: str = "<|user|>",
    assistant_role_tag: str = "<|assistant|>",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Creates train/val token streams from a list of text samples.

    Works with role-tagged chat text from JSONL records.
    When assistant_only_loss=True, computes a per-token loss mask that
    marks only assistant content as supervised.

    Output format:
    - train_tokens: shape (num_train_tokens,)
    - val_tokens: shape (num_val_tokens,)
    - train_loss_mask: shape (num_train_tokens,)  — all 1.0 when masking is off
    - val_loss_mask: shape (num_val_tokens,)      — all 1.0 when masking is off
    """
    if not train_texts or not val_texts:
        raise ValueError("train_texts and val_texts must both be non-empty")

    def build_stream(texts: list[str]) -> tuple[list[int], list[float]]:
        if not assistant_only_loss:
            joined = "\n\n".join(texts)
            ids = encode_text(tokenizer, joined, add_bos=True, add_eos=True)
            return ids, [1.0] * len(ids)

        _, bos_id, eos_id = _get_special_token_ids(tokenizer)
        sep_ids = _encode_segment(tokenizer, "\n\n")

        all_ids: list[int] = [bos_id]
        all_mask: list[float] = [0.0]

        for text_index, text in enumerate(texts):
            ids, mask = encode_text_with_assistant_mask(
                tokenizer=tokenizer,
                text=text,
                user_role_tag=user_role_tag,
                assistant_role_tag=assistant_role_tag,
            )
            all_ids.extend(ids)
            all_mask.extend(mask)

            if text_index < len(texts) - 1:
                all_ids.extend(sep_ids)
                all_mask.extend([0.0] * len(sep_ids))

        all_ids.append(eos_id)
        all_mask.append(0.0)

        return all_ids, all_mask

    train_ids, train_mask = build_stream(train_texts)
    val_ids, val_mask = build_stream(val_texts)

    train_tokens = jnp.array(train_ids, dtype=jnp.int32)
    val_tokens = jnp.array(val_ids, dtype=jnp.int32)
    train_loss_mask = jnp.array(train_mask, dtype=jnp.float32)
    val_loss_mask = jnp.array(val_mask, dtype=jnp.float32)

    # Ensure both splits are large enough to sample (x, y) windows.
    min_stream_len = seq_len + 2
    train_tokens = _ensure_min_length(train_tokens, min_stream_len)
    val_tokens = _ensure_min_length(val_tokens, min_stream_len)
    train_loss_mask = _ensure_min_length(train_loss_mask, min_stream_len)
    val_loss_mask = _ensure_min_length(val_loss_mask, min_stream_len)

    return train_tokens, val_tokens, train_loss_mask, val_loss_mask


# =============================================================================
# Optimization
# =============================================================================


def cross_entropy_loss(
    logits: jax.Array,
    labels: jax.Array,
    loss_mask: jax.Array | None = None,
) -> jax.Array:
    """Language-model cross-entropy with optional token-level masking."""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    losses = optax.softmax_cross_entropy(logits, one_hot)

    if loss_mask is None:
        return losses.mean()

    mask = loss_mask.astype(losses.dtype)
    masked_sum = jnp.sum(losses * mask)
    denom = jnp.sum(mask)
    return jnp.where(denom > 0, masked_sum / denom, losses.mean())


def create_lr_schedule(
    max_steps: int, warmup_steps: int, peak_lr: float
) -> optax.Schedule:
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
    for block in model.blocks:
        block.moe.update_expert_bias(block.moe.last_topk_indices.get_value())


def sample_lm_batch(
    token_stream: jax.Array,
    loss_mask_stream: jax.Array,
    batch_size: int,
    seq_len: int,
    rng_key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
    m_list: list[jax.Array] = []
    for start in starts.tolist():
        token_window = token_stream[start : start + seq_len + 1]
        mask_window = loss_mask_stream[start : start + seq_len + 1]
        x_list.append(token_window[:-1])
        y_list.append(token_window[1:])
        m_list.append(mask_window[1:])

    x = jnp.stack(x_list, axis=0)
    y = jnp.stack(y_list, axis=0)
    y_mask = jnp.stack(m_list, axis=0)
    return x, y, y_mask


def train_model(
    model: NemotronNanoBlock,
    optimizer: nnx.Optimizer,
    train_tokens: jax.Array,
    train_loss_mask: jax.Array,
    steps: int,
    batch_size: int,
    seq_len: int,
    rng_key: jax.Array,
    debug_mask_ratio: bool = False,
    debug_mask_every: int = 1,
) -> jax.Array:
    """Runs a tiny training loop and prints readable metrics."""

    if debug_mask_every <= 0:
        raise ValueError("debug_mask_every must be > 0")

    mask_ratios: list[float] = []

    @nnx.jit
    def train_step(
        model: NemotronNanoBlock,
        optimizer: nnx.Optimizer,
        x_batch: jax.Array,
        y_batch: jax.Array,
        y_mask_batch: jax.Array,
    ) -> jax.Array:
        def loss_fn(model: NemotronNanoBlock) -> jax.Array:
            logits_local = model(x_batch)
            return cross_entropy_loss(logits_local, y_batch, loss_mask=y_mask_batch)

        total_loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

        # Aux-loss-free load balancing: update expert biases AFTER the gradient step.
        # Uses the top-k indices stored by each SparseMoE during the forward pass.
        _update_all_expert_biases(model)

        return total_loss

    print("\nTraining:")
    for step in range(steps):
        rng_key, batch_key = jax.random.split(rng_key)
        x_batch, y_batch, y_mask_batch = sample_lm_batch(
            train_tokens,
            train_loss_mask,
            batch_size,
            seq_len,
            batch_key,
        )
        total_loss = train_step(model, optimizer, x_batch, y_batch, y_mask_batch)

        print(f"  step {step + 1:>3}/{steps} | ce={float(total_loss):.4f}")
        if debug_mask_ratio:
            supervised_tokens = float(jnp.sum(y_mask_batch))
            total_tokens = int(y_mask_batch.size)
            ratio = supervised_tokens / max(total_tokens, 1)
            mask_ratios.append(ratio)
            
            if ((step + 1) % debug_mask_every == 0):
                print(
                    "    mask-debug "
                    f"supervised={supervised_tokens:.1f}/{total_tokens} "
                    f"ratio={ratio:.4f}"
                )

    if debug_mask_ratio and mask_ratios:
        ratio_min = min(mask_ratios)
        ratio_max = max(mask_ratios)
        ratio_mean = sum(mask_ratios) / len(mask_ratios)
        print(
            "Mask ratio summary "
            f"min={ratio_min:.4f} "
            f"max={ratio_max:.4f} "
            f"mean={ratio_mean:.4f}"
        )

    return rng_key


def evaluate_model(
    model: NemotronNanoBlock,
    val_tokens: jax.Array,
    val_loss_mask: jax.Array,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    rng_key: jax.Array,
) -> tuple[float, float, jax.Array]:
    """
    Evaluates model on validation batches.

    Returns:
    - mean_ce_loss
    - perplexity = exp(mean_ce_loss)
    - updated rng_key
    """
    ce_losses: list[jax.Array] = []

    for _ in range(eval_batches):
        rng_key, batch_key = jax.random.split(rng_key)
        x_batch, y_batch, y_mask_batch = sample_lm_batch(
            val_tokens,
            val_loss_mask,
            batch_size,
            seq_len,
            batch_key,
        )
        logits = model(x_batch)
        ce_losses.append(cross_entropy_loss(logits, y_batch, loss_mask=y_mask_batch))

    mean_ce = jnp.mean(jnp.stack(ce_losses))
    ppl = jnp.exp(mean_ce)

    return float(mean_ce), float(ppl), rng_key


# =============================================================================
# Chat
# =============================================================================


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
    tokenizer: "PreTrainedTokenizerBase",
    prompt_text: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    rng_key: jax.Array,
) -> tuple[str, jax.Array]:
    """Autoregressively generates assistant text from a prompt."""
    pad_id, bos_id, eos_id = _get_special_token_ids(tokenizer)

    context_ids = encode_text(tokenizer, prompt_text, add_bos=True, add_eos=False)
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        model_input = pad_or_trim_context(context_ids, seq_len, pad_id)
        logits = model(model_input)

        next_logits = logits[0, -1]

        # Avoid sampling control tokens during response generation.
        # Some tokenizers reuse eos as pad, so guard against masking eos by accident.
        if pad_id != eos_id:
            next_logits = next_logits.at[pad_id].set(-1e9)
        if bos_id != eos_id and bos_id != pad_id:
            next_logits = next_logits.at[bos_id].set(-1e9)

        rng_key, sample_key = jax.random.split(rng_key)
        next_id = sample_next_token(next_logits, temperature, sample_key)

        if next_id == eos_id:
            break

        context_ids.append(next_id)
        generated_ids.append(next_id)

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return str(decoded), rng_key


def build_chat_prompt(
    turns: list[tuple[str, str]],
    user_text: str,
    user_role_tag: str,
    assistant_role_tag: str,
) -> str:
    """
    Builds a role-tagged multi-turn prompt from chat history and latest user text.

    Format:
      <|user|>...<|assistant|>...<|user|>...<|assistant|>
    """
    parts: list[str] = []
    for prev_user, prev_assistant in turns:
        parts.append(f"{user_role_tag}\n{prev_user.strip()}\n")
        parts.append(f"{assistant_role_tag}\n{prev_assistant.strip()}\n")

    parts.append(f"{user_role_tag}\n{user_text.strip()}\n")
    parts.append(f"{assistant_role_tag}\n")
    return "".join(parts)


def chat_loop(
    model: NemotronNanoBlock,
    tokenizer: "PreTrainedTokenizerBase",
    seq_len: int,
    temperature: float,
    max_new_tokens: int,
    rng_key: jax.Array,
    user_role_tag: str,
    assistant_role_tag: str,
    history_turns: int,
) -> None:
    """Runs an interactive terminal chatbot with short conversation memory."""
    if history_turns <= 0:
        raise ValueError("history_turns must be > 0")

    print("\nChatbot is ready. Commands: /help, /reset, /exit")
    print(f"Using up to {history_turns} previous turns as prompt context.")

    turns: list[tuple[str, str]] = []

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: bye.")
            break

        lower_text = user_text.lower()
        if lower_text in {"quit", "exit", "/exit"}:
            print("Bot: bye.")
            break
        if not user_text:
            continue
        if lower_text == "/help":
            print("Bot: /reset clears memory, /exit ends chat.")
            continue
        if lower_text == "/reset":
            turns.clear()
            print("Bot: conversation memory cleared.")
            continue

        prompt_text = build_chat_prompt(
            turns=turns[-history_turns:],
            user_text=user_text,
            user_role_tag=user_role_tag,
            assistant_role_tag=assistant_role_tag,
        )

        reply, rng_key = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            seq_len=seq_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            rng_key=rng_key,
        )

        reply = reply.strip() or "..."
        turns.append((user_text, reply))
        print(f"Bot: {reply}")


def load_checkpoint_into_model(
    model: NemotronNanoBlock,
    checkpoint_path: str,
) -> int:
    """
    Loads model parameter leaves from a legacy .npz checkpoint.

    Expected checkpoint format:
    - Keys: param_0, param_1, ..., param_N
    - Values: numpy arrays matching current model parameter shapes in order
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    param_state = nnx.state(model, nnx.Param)
    param_leaves = jax.tree_util.tree_leaves(param_state)
    tree_def = jax.tree_util.tree_structure(param_state)

    with np.load(path, allow_pickle=False) as ckpt:
        keys = list(ckpt.files)
        if not keys:
            raise ValueError(f"Checkpoint is empty: {checkpoint_path}")

        if not all(key.startswith("param_") for key in keys):
            raise ValueError(
                "Checkpoint format mismatch: expected keys like 'param_0', 'param_1', ..."
            )

        try:
            ordered_keys = sorted(keys, key=lambda name: int(name.split("_", 1)[1]))
        except ValueError as exc:
            raise ValueError(
                "Checkpoint keys must follow 'param_<index>' format with integer index"
            ) from exc

        ckpt_leaves = [jnp.asarray(ckpt[key]) for key in ordered_keys]

    if len(ckpt_leaves) != len(param_leaves):
        raise ValueError(
            "Checkpoint parameter count mismatch: "
            f"checkpoint has {len(ckpt_leaves)} leaves, "
            f"model expects {len(param_leaves)} leaves"
        )

    restored_leaves: list[jax.Array] = []
    for leaf_index, (target_leaf, source_leaf) in enumerate(
        zip(param_leaves, ckpt_leaves)
    ):
        target_arr = jnp.asarray(target_leaf)
        if target_arr.shape != source_leaf.shape:
            raise ValueError(
                "Checkpoint shape mismatch at leaf "
                f"{leaf_index}: checkpoint {source_leaf.shape} vs model {target_arr.shape}"
            )

        if source_leaf.dtype != target_arr.dtype:
            source_leaf = source_leaf.astype(target_arr.dtype)

        restored_leaves.append(source_leaf)

    restored_params = jax.tree_util.tree_unflatten(tree_def, restored_leaves)
    nnx.update(model, restored_params)
    return len(restored_leaves)


# =============================================================================
# Main
# =============================================================================


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
        "--train-jsonl",
        type=str,
        default=None,
        help="Path to train JSONL file",
    )
    parser.add_argument(
        "--val-jsonl",
        type=str,
        default=None,
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--jsonl-text-key",
        type=str,
        default="serialized_text",
        help="JSON key to read text from in JSONL records",
    )
    parser.add_argument(
        "--jsonl-max-train-records",
        type=int,
        default=50000,
        help="Max train records to read from train JSONL",
    )
    parser.add_argument(
        "--jsonl-max-val-records",
        type=int,
        default=5000,
        help="Max validation records to read from val JSONL",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="OpenAssistant/oasst2",
        help="Hugging Face dataset name to load directly (used when --train-jsonl/--val-jsonl are not provided)",
    )
    parser.add_argument(
        "--hf-lang",
        type=str,
        default="en",
        help="Language filter for HF dataset (e.g. 'en')",
    )
    parser.add_argument(
        "--hf-val-ratio",
        type=float,
        default=0.02,
        help="Fraction of HF dataset samples to use for validation",
    )
    parser.add_argument(
        "--hf-min-turns",
        type=int,
        default=2,
        help="Minimum conversation turns to keep (HF dataset filter)",
    )
    parser.add_argument(
        "--hf-max-turns",
        type=int,
        default=20,
        help="Maximum conversation turns to keep (HF dataset filter)",
    )
    parser.add_argument(
        "--hf-min-chars",
        type=int,
        default=32,
        help="Minimum serialized char length to keep (HF dataset filter)",
    )
    parser.add_argument(
        "--hf-max-chars",
        type=int,
        default=6000,
        help="Maximum serialized char length to keep (HF dataset filter)",
    )
    parser.add_argument(
        "--hf-max-train-records",
        type=int,
        default=50000,
        help="Max train records to load from HF dataset",
    )
    parser.add_argument(
        "--hf-max-val-records",
        type=int,
        default=5000,
        help="Max validation records to load from HF dataset",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="google/byt5-small",
        help="Hugging Face tokenizer name or local path",
    )
    parser.add_argument(
        "--tokenizer-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory for tokenizer files",
    )
    parser.add_argument(
        "--preview-first-story",
        action="store_true",
        help="Print a short preview of the first loaded training sample",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Run train+eval only (useful for non-interactive testing)",
    )
    parser.add_argument(
        "--chat-only",
        action="store_true",
        help="Skip train/eval and start chat after loading --checkpoint-path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to .npz checkpoint with model parameter leaves",
    )
    parser.add_argument(
        "--assistant-only-loss",
        action="store_true",
        help=(
            "Mask loss to assistant content tokens only using role tags in text "
            "(recommended with role-tagged chat JSONL data)"
        ),
    )
    parser.add_argument(
        "--user-role-tag",
        type=str,
        default="<|user|>",
        help="Role tag used to mark user turns in serialized chat text",
    )
    parser.add_argument(
        "--assistant-role-tag",
        type=str,
        default="<|assistant|>",
        help="Role tag used to mark assistant turns in serialized chat text",
    )
    parser.add_argument(
        "--debug-mask-ratio",
        action="store_true",
        help="Print supervised-token mask ratio during training",
    )
    parser.add_argument(
        "--debug-mask-every",
        type=int,
        default=1,
        help="Print mask-debug line every N training steps",
    )
    parser.add_argument(
        "--chat-history-turns",
        type=int,
        default=6,
        help="How many previous user-assistant turns to include in chat context",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be > 0")
    if args.chat_history_turns <= 0:
        raise ValueError("--chat-history-turns must be > 0")
    if args.chat_only and not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required when --chat-only is set")
    if args.chat_only and args.skip_chat:
        raise ValueError("--chat-only and --skip-chat cannot be used together")

    print("Initializing minimal Nemotron app...")

    # 1) Load Hugging Face tokenizer.
    tokenizer = load_hf_tokenizer(
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.tokenizer_cache_dir,
    )

    # 2) Build Nemotron config/model.
    config = NemotronConfig.from_preset(args.preset)
    # len(tokenizer) includes any runtime-added special tokens.
    config.vocab_size = len(tokenizer)

    if args.seq_len % config.mamba_chunk_size != 0:
        raise ValueError(
            f"seq_len must be divisible by mamba_chunk_size ({config.mamba_chunk_size})"
        )

    print(
        "Model setup: "
        f"preset={args.preset}, tokenizer={args.tokenizer_name}, "
        f"vocab_size={config.vocab_size}, "
        f"seq_len={args.seq_len}, chunk_size={config.mamba_chunk_size}"
    )

    rngs = nnx.Rngs(args.seed)
    model = NemotronNanoBlock(rngs=rngs, config=config)

    # Chat-only mode: load checkpoint and jump straight to interactive chat.
    if args.chat_only:
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required for chat-only mode")

        restored_count = load_checkpoint_into_model(
            model=model,
            checkpoint_path=args.checkpoint_path,
        )
        print(
            f"Loaded checkpoint '{args.checkpoint_path}' "
            f"({restored_count} parameter leaves restored)."
        )

        chat_loop(
            model=model,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            rng_key=jax.random.PRNGKey(args.seed + 1),
            user_role_tag=args.user_role_tag,
            assistant_role_tag=args.assistant_role_tag,
            history_turns=args.chat_history_turns,
        )
        return

    # 3) Load conversational text data.
    if args.train_jsonl and args.val_jsonl:
        # Load from local JSONL files.
        train_texts = load_jsonl_texts(
            jsonl_path=args.train_jsonl,
            max_records=args.jsonl_max_train_records,
            text_key=args.jsonl_text_key,
        )
        val_texts = load_jsonl_texts(
            jsonl_path=args.val_jsonl,
            max_records=args.jsonl_max_val_records,
            text_key=args.jsonl_text_key,
        )
        print(
            "Dataset setup: "
            "source=jsonl, "
            f"train_records={len(train_texts)}, "
            f"val_records={len(val_texts)}, "
            f"text_key={args.jsonl_text_key}"
        )
    else:
        # Load directly from Hugging Face — no local files needed.
        train_texts, val_texts = load_hf_dataset_texts(
            dataset_name=args.hf_dataset,
            lang=args.hf_lang,
            val_ratio=args.hf_val_ratio,
            min_turns=args.hf_min_turns,
            max_turns=args.hf_max_turns,
            min_chars=args.hf_min_chars,
            max_chars=args.hf_max_chars,
            max_train_records=args.hf_max_train_records,
            max_val_records=args.hf_max_val_records,
        )

    if args.preview_first_story:
        preview = train_texts[0].replace("\n", " ").strip()
        preview_limit = 240
        if len(preview) > preview_limit:
            preview = preview[:preview_limit] + "..."

        print("\nDataset preview (first loaded sample):")
        print(f"  {preview}")

    # Build optimizer: AdamW with warmup + cosine decay
    max_steps = max(args.steps, 1)
    warmup_steps = max(1, max_steps // 10)
    lr_schedule = create_lr_schedule(
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        peak_lr=3e-4,
    )
    gradient_transformation = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=0.1),
    )
    optimizer = nnx.Optimizer(model, gradient_transformation, wrt=nnx.Param)

    # Use a separate key stream for data/generation randomness.
    rng_key = jax.random.PRNGKey(args.seed + 1)

    # 4) Prepare token streams from train/validation text samples.
    train_tokens, val_tokens, train_loss_mask, val_loss_mask = prepare_datasets(
        tokenizer=tokenizer,
        train_texts=train_texts,
        val_texts=val_texts,
        seq_len=args.seq_len,
        assistant_only_loss=args.assistant_only_loss,
        user_role_tag=args.user_role_tag,
        assistant_role_tag=args.assistant_role_tag,
    )

    # 5) Train.
    rng_key = train_model(
        model=model,
        optimizer=optimizer,
        train_tokens=train_tokens,
        train_loss_mask=train_loss_mask,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        rng_key=rng_key,
        debug_mask_ratio=args.debug_mask_ratio,
        debug_mask_every=args.debug_mask_every,
    )

    # 6) Evaluate.
    mean_ce, perplexity, rng_key = evaluate_model(
        model=model,
        val_tokens=val_tokens,
        val_loss_mask=val_loss_mask,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
        rng_key=rng_key,
    )

    print("\nEvaluation:")
    print(f"  mean CE loss: {mean_ce:.4f}")
    print(f"  perplexity:   {perplexity:.4f}")

    # 7) Chat.
    if not args.skip_chat:
        chat_loop(
            model=model,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            rng_key=rng_key,
            user_role_tag=args.user_role_tag,
            assistant_role_tag=args.assistant_role_tag,
            history_turns=args.chat_history_turns,
        )


if __name__ == "__main__":
    main()
